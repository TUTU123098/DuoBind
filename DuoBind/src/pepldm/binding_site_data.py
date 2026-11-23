from __future__ import annotations

import json
import hashlib
from functools import partial
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .tokenizer import ESMSeqTokenizer


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EmbeddingCacheStore:
    """Read and validate residue-only binding-site embeddings.

    New caches use two ESMCModel instances initialized from one shared
    ESMCpep checkpoint. Older manifests remain readable.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        expected_esm_args: dict[str, Any] | None = None,
        expected_weight_path: str | Path | None = None,
        expected_peptide_esm_args: dict[str, Any] | None = None,
        expected_peptide_weight_path: str | Path | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.manifest_path = self.cache_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Embedding cache manifest not found: {self.manifest_path}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        entries = self.manifest.get("entries")
        if not isinstance(entries, list):
            raise ValueError("Embedding cache manifest has no entries list")
        self.entries = {}
        for entry in entries:
            key = (str(entry["sequence_type"]), str(entry["hash"]))
            self.entries[key] = entry
        self._validate_metadata(
            expected_esm_args=expected_esm_args,
            expected_weight_path=expected_weight_path,
            expected_peptide_esm_args=expected_peptide_esm_args,
            expected_peptide_weight_path=expected_peptide_weight_path,
        )

    def _validate_metadata(
        self,
        expected_esm_args: dict[str, Any] | None,
        expected_weight_path: str | Path | None,
        expected_peptide_esm_args: dict[str, Any] | None,
        expected_peptide_weight_path: str | Path | None,
    ) -> None:
        encoders = self.manifest.get("encoders")
        if not isinstance(encoders, dict):
            legacy_encoder = self.manifest.get("encoder", {})
            encoders = {"protein": legacy_encoder, "peptide": legacy_encoder}
        for role in ("protein", "peptide"):
            if encoders.get(role, {}).get("class") != "ESMCModel":
                raise ValueError(
                    f"Embedding cache {role} encoder class mismatch: "
                    f"{encoders.get(role, {}).get('class')!r}"
                )
        encoder = encoders["protein"]
        peptide_encoder = encoders["peptide"]
        if encoder.get("role") not in (None, "protein"):
            raise ValueError("Embedding cache protein encoder role is invalid")
        if peptide_encoder.get("role") not in (None, "peptide"):
            raise ValueError("Embedding cache peptide encoder role is invalid")
        if encoder.get("name") not in (None, "ESMC", "ESMCpep"):
            raise ValueError("Embedding cache protein encoder name is invalid")
        if peptide_encoder.get("name") not in (None, "ESMCpep"):
            raise ValueError("Embedding cache peptide encoder name is invalid")
        if self.manifest.get("shared_checkpoint") is True:
            if (
                encoder.get("weight_sha256")
                != peptide_encoder.get("weight_sha256")
                or encoder.get("weight_identifier")
                != peptide_encoder.get("weight_identifier")
            ):
                raise ValueError(
                    "Shared-checkpoint cache has inconsistent protein/peptide "
                    "weight metadata"
                )

        self._validate_config(
            encoder.get("config", {}),
            expected_esm_args,
            "protein",
        )
        self._validate_config(
            peptide_encoder.get("config", {}),
            expected_peptide_esm_args or expected_esm_args,
            "peptide",
        )
        self._validate_weight(
            encoder,
            expected_weight_path,
            "protein",
        )
        self._validate_weight(
            peptide_encoder,
            expected_peptide_weight_path or expected_weight_path,
            "peptide",
        )

    @staticmethod
    def _validate_config(
        cached_config: dict[str, Any],
        expected_config: dict[str, Any] | None,
        role: str,
    ) -> None:
        if not expected_config:
            return
        for key in (
            "vocab_size",
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
        ):
            if key in expected_config and key in cached_config:
                if int(expected_config[key]) != int(cached_config[key]):
                    raise ValueError(
                        f"Embedding cache {role} config mismatch for {key}: "
                        f"cache={cached_config[key]} expected={expected_config[key]}"
                    )

    @staticmethod
    def _validate_weight(
        encoder: dict[str, Any],
        expected_weight_path: str | Path | None,
        role: str,
    ) -> None:
        cached_weight_hash = encoder.get("weight_sha256")
        if expected_weight_path is not None:
            expected_hash = _file_sha256(expected_weight_path)
            if cached_weight_hash != expected_hash:
                raise ValueError(
                    f"Embedding cache {role} weight mismatch: "
                    f"cache={cached_weight_hash} expected={expected_hash}"
                )
        elif cached_weight_hash:
            # A cache without an explicit weight path is usable, but this is
            # intentionally visible instead of silently treating it as current.
            import warnings

            warnings.warn(
                f"Embedding cache {role} weight was not checked because no "
                "expected weight path was provided",
                RuntimeWarning,
                stacklevel=3,
            )

    def load(
        self,
        sequence: str,
        sequence_type: str,
    ) -> torch.Tensor:
        key = (sequence_type, sequence_hash(sequence))
        if key not in self.entries:
            raise KeyError(
                f"Sequence is absent from embedding cache: type={sequence_type}"
            )
        entry = self.entries[key]
        if entry.get("sequence") != sequence:
            raise ValueError("Embedding cache hash collision or stale sequence metadata")
        if int(entry.get("length", -1)) != len(sequence):
            raise ValueError("Embedding cache sequence length metadata mismatch")
        path = self.cache_dir / str(entry["file"])
        embedding = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(embedding, torch.Tensor):
            raise TypeError(f"Cached embedding is not a tensor: {path}")
        expected_shape = tuple(int(value) for value in entry["embedding_shape"])
        if tuple(embedding.shape) != expected_shape:
            raise ValueError(
                f"Cached embedding shape mismatch for {path}: "
                f"{tuple(embedding.shape)} != {expected_shape}"
            )
        if embedding.ndim != 2 or embedding.shape[0] != len(sequence):
            raise ValueError(
                "Cached embeddings must be residue-only [Lsequence, D]; "
                f"got {tuple(embedding.shape)}"
            )
        return embedding.float()


class BindingSiteDataset(Dataset):
    """Protein-peptide pairs with residue labels aligned to protein tokens."""

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer: ESMSeqTokenizer,
        prot_max_length: int = 512,
        pep_max_length: int = 50,
        use_cached_embeddings: bool = False,
        embedding_cache_dir: str | Path | None = None,
        expected_esm_args: dict[str, Any] | None = None,
        expected_weight_path: str | Path | None = None,
        expected_peptide_esm_args: dict[str, Any] | None = None,
        expected_peptide_weight_path: str | Path | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.tokenizer = tokenizer
        self.prot_max_length = prot_max_length
        self.pep_max_length = pep_max_length
        self.use_cached_embeddings = use_cached_embeddings
        if use_cached_embeddings and embedding_cache_dir is None:
            raise ValueError(
                "embedding_cache_dir is required when use_cached_embeddings=True"
            )
        self.embedding_cache = (
            EmbeddingCacheStore(
                embedding_cache_dir,
                expected_esm_args=expected_esm_args,
                expected_weight_path=expected_weight_path,
                expected_peptide_esm_args=expected_peptide_esm_args,
                expected_peptide_weight_path=expected_peptide_weight_path,
            )
            if use_cached_embeddings
            else None
        )
        required = {
            "sample_id",
            "protein_id",
            "protein_sequence",
            "peptide_sequence",
            "site_labels",
        }
        missing = sorted(required - set(self.df.columns))
        if missing:
            raise KeyError(
                f"{self.csv_path} missing columns {missing}; "
                f"found {list(self.df.columns)}"
            )

        max_length = int(self.df["protein_sequence"].astype(str).str.len().max())
        if max_length + 2 > prot_max_length:
            raise ValueError(
                f"prot_max_length={prot_max_length} cannot contain the longest "
                f"protein ({max_length} residues plus CLS/EOS)"
            )

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def parse_labels(value: Any, protein_length: int) -> list[int]:
        labels = json.loads(str(value))
        if not isinstance(labels, list) or len(labels) != protein_length:
            raise ValueError(
                "site_labels must have exactly one value per protein residue"
            )
        if any(int(label) not in (0, 1) for label in labels):
            raise ValueError("site_labels must contain only 0/1 values")
        return [int(label) for label in labels]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        protein = str(row["protein_sequence"])
        peptide = str(row["peptide_sequence"])
        labels = self.parse_labels(row["site_labels"], len(protein))

        if self.use_cached_embeddings:
            assert self.embedding_cache is not None
            if len(protein) > self.prot_max_length:
                raise ValueError(
                    f"Protein at index {index} exceeds prot_max_length "
                    f"{self.prot_max_length}"
                )
            if len(peptide) > self.pep_max_length:
                raise ValueError(
                    f"Peptide at index {index} exceeds pep_max_length "
                    f"{self.pep_max_length}; cached sequences are never truncated"
                )
            protein_embeddings = self.embedding_cache.load(protein, "protein")
            peptide_embeddings = self.embedding_cache.load(peptide, "peptide")
            site_labels = torch.tensor(labels, dtype=torch.float32)
            site_mask = torch.ones(len(protein), dtype=torch.float32)
            return {
                "protein_embeddings": protein_embeddings,
                "peptide_embeddings": peptide_embeddings,
                "prot_attention_mask": site_mask.clone(),
                "pep_attention_mask": torch.ones(
                    len(peptide), dtype=torch.float32
                ),
                "site_labels": site_labels,
                "site_mask": site_mask,
                "sample_id": str(row["sample_id"]),
                "protein_id": str(row["protein_id"]),
                "protein_sequence": protein,
                "peptide_sequence": peptide,
            }

        protein_tokens = self.tokenizer(
            protein,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        peptide_tokens = self.tokenizer(
            peptide,
            padding=False,
            truncation=True,
            max_length=self.pep_max_length,
            return_tensors="pt",
        )
        protein_input_ids = protein_tokens["input_ids"].squeeze(0)
        protein_attention_mask = protein_tokens["attention_mask"].squeeze(0)
        peptide_input_ids = peptide_tokens["input_ids"].squeeze(0)
        peptide_attention_mask = peptide_tokens["attention_mask"].squeeze(0)

        if protein_input_ids.numel() > self.prot_max_length:
            raise ValueError(
                f"Protein at index {index} exceeds prot_max_length "
                f"{self.prot_max_length}"
            )

        # Token 0 is CLS; residue i is token i+1; EOS follows residues.
        site_labels = torch.zeros(protein_input_ids.numel(), dtype=torch.float32)
        site_mask = torch.zeros(protein_input_ids.numel(), dtype=torch.float32)
        site_labels[1 : len(protein) + 1] = torch.tensor(
            labels,
            dtype=torch.float32,
        )
        site_mask[1 : len(protein) + 1] = 1.0

        return {
            "prot_input_ids": protein_input_ids,
            "prot_attention_mask": protein_attention_mask,
            "pep_input_ids": peptide_input_ids,
            "pep_attention_mask": peptide_attention_mask,
            "site_labels": site_labels,
            "site_mask": site_mask,
            "sample_id": str(row["sample_id"]),
            "protein_id": str(row["protein_id"]),
            "protein_sequence": protein,
            "peptide_sequence": peptide,
        }


def collate_fn(
    batch: list[dict[str, Any]],
    pad_token_id: int = 1,
) -> dict[str, Any]:
    cached = "protein_embeddings" in batch[0]
    if cached:
        protein_embeddings = pad_sequence(
            [item["protein_embeddings"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        )
        prot_attention_mask = pad_sequence(
            [item["prot_attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        )
    else:
        prot_input_ids = pad_sequence(
            [item["prot_input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
        prot_attention_mask = pad_sequence(
            [item["prot_attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
    site_labels = pad_sequence(
        [item["site_labels"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    site_mask = pad_sequence(
        [item["site_mask"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    result = {
        "prot_attention_mask": prot_attention_mask,
        "site_labels": site_labels,
        "site_mask": site_mask,
        "sample_id": [item["sample_id"] for item in batch],
        "protein_id": [item["protein_id"] for item in batch],
        "protein_sequence": [item["protein_sequence"] for item in batch],
        "peptide_sequence": [item["peptide_sequence"] for item in batch],
    }
    if cached:
        result["protein_embeddings"] = protein_embeddings
        result["pep_attention_mask"] = pad_sequence(
            [item["pep_attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        )
        result["peptide_embeddings"] = pad_sequence(
            [item["peptide_embeddings"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        )
    else:
        result["prot_input_ids"] = prot_input_ids
        result["pep_input_ids"] = pad_sequence(
            [item["pep_input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
        result["pep_attention_mask"] = pad_sequence(
            [item["pep_attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
    return result


class BindingSiteDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_path: str | Path,
        val_data_path: str | Path,
        test_data_path: str | Path,
        tokenizer: ESMSeqTokenizer,
        batch_size: int = 8,
        num_workers: int = 0,
        prot_max_length: int = 512,
        pep_max_length: int = 50,
        use_cached_embeddings: bool = False,
        embedding_cache_dir: str | Path | None = None,
        esm_args: dict[str, Any] | None = None,
        expected_weight_path: str | Path | None = None,
        expected_peptide_esm_args: dict[str, Any] | None = None,
        expected_peptide_weight_path: str | Path | None = None,
    ):
        super().__init__()
        self.train_data_path = Path(train_data_path)
        self.val_data_path = Path(val_data_path)
        self.test_data_path = Path(test_data_path)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prot_max_length = prot_max_length
        self.pep_max_length = pep_max_length
        self.use_cached_embeddings = use_cached_embeddings
        self.embedding_cache_dir = embedding_cache_dir
        self.esm_args = esm_args
        self.expected_weight_path = expected_weight_path
        self.expected_peptide_esm_args = expected_peptide_esm_args
        self.expected_peptide_weight_path = expected_peptide_weight_path

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = BindingSiteDataset(
            self.train_data_path,
            self.tokenizer,
            self.prot_max_length,
            self.pep_max_length,
            use_cached_embeddings=self.use_cached_embeddings,
            embedding_cache_dir=self.embedding_cache_dir,
            expected_esm_args=self.esm_args,
            expected_weight_path=self.expected_weight_path,
            expected_peptide_esm_args=self.expected_peptide_esm_args,
            expected_peptide_weight_path=self.expected_peptide_weight_path,
        )
        self.val_dataset = BindingSiteDataset(
            self.val_data_path,
            self.tokenizer,
            self.prot_max_length,
            self.pep_max_length,
            use_cached_embeddings=self.use_cached_embeddings,
            embedding_cache_dir=self.embedding_cache_dir,
            expected_esm_args=self.esm_args,
            expected_weight_path=self.expected_weight_path,
            expected_peptide_esm_args=self.expected_peptide_esm_args,
            expected_peptide_weight_path=self.expected_peptide_weight_path,
        )
        self.test_dataset = BindingSiteDataset(
            self.test_data_path,
            self.tokenizer,
            self.prot_max_length,
            self.pep_max_length,
            use_cached_embeddings=self.use_cached_embeddings,
            embedding_cache_dir=self.embedding_cache_dir,
            expected_esm_args=self.esm_args,
            expected_weight_path=self.expected_weight_path,
            expected_peptide_esm_args=self.expected_peptide_esm_args,
            expected_peptide_weight_path=self.expected_peptide_weight_path,
        )

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=partial(
                collate_fn,
                pad_token_id=self.tokenizer.pad_token_id,
            ),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)


def compute_train_pos_weight(dataset: BindingSiteDataset) -> float:
    positive = 0
    negative = 0
    for value in dataset.df["site_labels"]:
        labels = torch.tensor(json.loads(str(value)), dtype=torch.long)
        positive += int(labels.sum())
        negative += int(labels.numel() - labels.sum())
    if positive <= 0:
        raise ValueError("Training split has no positive residue labels")
    return negative / positive
