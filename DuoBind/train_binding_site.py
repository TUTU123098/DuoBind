from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import lightning as L
import torch


from src.pepldm.binding_site import BindingSiteModel
from src.pepldm.binding_site_data import (
    BindingSiteDataModule,
    compute_train_pos_weight,
)
from src.pepldm.local_context import LOCAL_CONTEXT_NAMES
from src.pepldm.tokenizer import ESMSeqTokenizer


def parse_limit(value: str) -> int | float:
    return float(value) if "." in value else int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("site", "binding"), default="site")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/binding_site_5fold_cc50"),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--prot-max-length", type=int, default=512)
    parser.add_argument("--pep-max-length", type=int, default=50)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--save-dir", type=Path, default=Path("logs/binding_site"))
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--esmc-weights-path", type=str, default=None)
    parser.add_argument("--esmcpep-weights-path", type=str, default=None)
    parser.add_argument("--pretrained-ckpt", type=str, default=None)
    parser.add_argument(
        "--site-head",
        choices=(
            "mlp",
            "local_conv",
            "attention_guided",
            "gated",
            "multiscale",
            "evidence_only",
            "interaction_evidence",
        ),
        default="mlp",
    )
    parser.add_argument("--site-head-dropout", type=float, default=0.3)
    parser.add_argument(
        "--attention-aggregate",
        choices=("max", "entropy", "topk"),
        default="max",
    )
    parser.add_argument("--attention-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--interaction-source",
        choices=("attention", "logits", "both"),
        default="attention",
    )
    parser.add_argument(
        "--evidence-features",
        default="max,topk,entropy",
        help="Comma-separated evidence features.",
    )
    parser.add_argument(
        "--evidence-fusion",
        choices=("concat", "gate"),
        default="concat",
    )
    parser.add_argument("--evidence-loss-weight", type=float, default=0.0)
    parser.add_argument("--evidence-topk", type=int, default=3)
    parser.add_argument(
        "--local-context",
        choices=LOCAL_CONTEXT_NAMES,
        default="none",
        help="Independent local-context ablation: none, single, multiscale, or ffn.",
    )
    parser.add_argument("--local-context-width", type=int, default=256)
    parser.add_argument("--use-cached-embeddings", action="store_true")
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=Path("cache/binding_site_esmc"),
    )
    parser.add_argument("--skip-test", action="store_true", default=True)
    parser.add_argument("--run-test", action="store_false", dest="skip_test")
    parser.add_argument("--dev-mode", action="store_true")
    parser.add_argument("--load-from-esmc", action="store_true", default=True)
    parser.add_argument("--limit-train-batches", type=parse_limit, default=None)
    parser.add_argument("--limit-val-batches", type=parse_limit, default=None)
    parser.add_argument("--limit-test-batches", type=parse_limit, default=None)
    parser.add_argument(
        "--verify-transfer-only",
        action="store_true",
        help="Build the model, verify interaction transfer, and exit before Trainer creation.",
    )
    args = parser.parse_args()

    if args.task != "site":
        raise ValueError(
            "This standalone entry point implements the residue-level site task; "
            "use train_frag.py for the existing pair-level binding task."
        )

    if args.dev_mode:
        args.fold = 0
    if args.use_cached_embeddings:
        args.load_from_esmc = False

    L.seed_everything(args.seed, workers=True)
    tokenizer = ESMSeqTokenizer()
    esm_args = {
        "vocab_size": 64,
        "hidden_size": 1152,
        "num_attention_heads": 18,
        "num_hidden_layers": 36,
    }
    data_module = BindingSiteDataModule(
        train_data_path=args.data_dir / f"fold{args.fold}_train.csv",
        val_data_path=args.data_dir / f"fold{args.fold}_val.csv",
        test_data_path=args.data_dir / f"fold{args.fold}_test.csv",
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prot_max_length=args.prot_max_length,
        pep_max_length=args.pep_max_length,
        use_cached_embeddings=args.use_cached_embeddings,
        embedding_cache_dir=args.embedding_cache_dir,
        esm_args=esm_args,
        expected_weight_path=(
            args.esmcpep_weights_path or args.esmc_weights_path
        ),
        expected_peptide_esm_args=esm_args,
        expected_peptide_weight_path=(
            args.esmcpep_weights_path or args.esmc_weights_path
        ),
    )
    data_module.setup("fit")
    pos_weight = compute_train_pos_weight(data_module.train_dataset)

    prediction_dir = args.prediction_dir or (
        args.save_dir / args.site_head / f"fold{args.fold}" / "predictions"
    )
    model = BindingSiteModel(
        tokenizer=tokenizer,
        esm_args=esm_args,
        load_from_esmc=args.load_from_esmc,
        esmc_weights_path=args.esmcpep_weights_path or args.esmc_weights_path,
        strict_esmc_load=args.load_from_esmc,
        pretrained_ckpt=args.pretrained_ckpt,
        pos_weight=pos_weight,
        threshold=0.5,
        prediction_dir=str(prediction_dir),
        site_head=args.site_head,
        site_head_dropout=args.site_head_dropout,
        attention_aggregate=args.attention_aggregate,
        attention_loss_weight=args.attention_loss_weight,
        interaction_source=args.interaction_source,
        evidence_features=args.evidence_features,
        evidence_fusion=args.evidence_fusion,
        evidence_loss_weight=args.evidence_loss_weight,
        evidence_topk=args.evidence_topk,
        local_context=args.local_context,
        local_context_width=args.local_context_width,
        use_cached_embeddings=args.use_cached_embeddings,
        embedding_cache_dir=str(args.embedding_cache_dir)
        if args.use_cached_embeddings
        else None,
        learning_rate=args.learning_rate,
    )

    use_gpu = torch.cuda.is_available()
    devices = [int(value) for value in args.devices.split(",")]
    experiment_dir = args.save_dir / args.site_head / f"fold{args.fold}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=experiment_dir,
        filename="{epoch}-{val_site_prauc:.4f}",
        monitor="val_site_prauc",
        mode="max",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
    )
    if args.verify_transfer_only:
        if not args.pretrained_ckpt:
            raise ValueError(
                "--verify-transfer-only requires --pretrained-ckpt"
            )
        return

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices if use_gpu else 1,
        precision="bf16-mixed" if use_gpu else "32-true",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        limit_train_batches=args.limit_train_batches or 1.0,
        limit_val_batches=args.limit_val_batches or 1.0,
        limit_test_batches=args.limit_test_batches or 1.0,
        callbacks=[checkpoint],
        logger=L.pytorch.loggers.CSVLogger(
            save_dir=str(args.save_dir),
            name=f"{args.site_head}/fold{args.fold}",
        ),
    )
    config = {
        "site_head": args.site_head,
        "fold": args.fold,
        "seed": args.seed,
        "epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "site_head_dropout": args.site_head_dropout,
        "hidden_dim": model.hidden_size,
        "pos_weight": pos_weight,
        "threshold": 0.5,
        "cached_mode": args.use_cached_embeddings,
        "embedding_cache_dir": str(args.embedding_cache_dir)
        if args.use_cached_embeddings
        else None,
        "attention_aggregate": args.attention_aggregate,
        "attention_loss_weight": args.attention_loss_weight,
        "interaction_source": args.interaction_source,
        "evidence_features": args.evidence_features,
        "evidence_fusion": args.evidence_fusion,
        "evidence_loss_weight": args.evidence_loss_weight,
        "evidence_topk": args.evidence_topk,
        "local_context": args.local_context,
        "local_context_width": args.local_context_width,
        "local_context_params": sum(
            parameter.numel()
            for parameter in model.local_context.parameters()
            if parameter.requires_grad
        )
        if model.local_context is not None
        else 0,
        "trainable_params": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    (experiment_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    trainer.fit(model, datamodule=data_module)
    fit_callback_metrics = dict(trainer.callback_metrics)
    if not args.skip_test:
        trainer.test(model, datamodule=data_module, ckpt_path="best")

    metrics = {}
    for key, value in fit_callback_metrics.items():
        metrics[key] = (
            float(value.detach().cpu().item())
            if hasattr(value, "detach")
            else value
        )
    csv_source = Path(trainer.logger.log_dir) / "metrics.csv"
    best_row = None
    if csv_source.exists():
        import csv

        with csv_source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        validation_rows = [
            row
            for row in rows
            if row.get("val_site_prauc") not in (None, "", "nan")
        ]
        if validation_rows:
            best_row = max(
                validation_rows,
                key=lambda row: float(row["val_site_prauc"]),
            )
            for metric_name in (
                "val_site_prauc",
                "val_site_mcc",
                "val_site_f1",
                "val_site_auc",
            ):
                if best_row.get(metric_name) not in (None, "", "nan"):
                    metrics[metric_name] = float(best_row[metric_name])
            if best_row.get("epoch") not in (None, "", "nan"):
                metrics["best_epoch"] = int(float(best_row["epoch"]))
    metrics["best_model_path"] = checkpoint.best_model_path
    metrics["best_val_pr_auc"] = metrics.get("val_site_prauc")
    metrics["best_val_mcc"] = metrics.get("val_site_mcc")
    metrics["best_val_f1"] = metrics.get("val_site_f1")
    metrics["best_val_roc_auc"] = metrics.get("val_site_auc")
    (experiment_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    if csv_source.exists():
        shutil.copyfile(csv_source, experiment_dir / "training_history.csv")


if __name__ == "__main__":
    main()
