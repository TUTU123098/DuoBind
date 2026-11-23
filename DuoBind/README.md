# DuoBind

DuoBind is a sequence-based framework for protein–peptide binding classification and peptide-conditioned binding-site prediction. It uses ESMC sequence encoders and bidirectional cross-attention to model protein–peptide interactions.

## Setups

The environment follows [High-PepBinder](https://github.com/mqyii/High-PepBinder): `python==3.12` and `torch==2.7.1+cu126`.

```bash
conda create -n DuoBind python=3.12 -y
conda activate DuoBind
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install lightning torchmetrics transformers tokenizers numpy pandas einops loguru click rich
```

If you already have the High-PepBinder environment with these dependencies, you can use that environment directly.

Run all commands from the directory containing `train_frag.py`, `train_binding_site.py`, and `src/`. Replace `/path/to/...` with your local data and checkpoint paths. GPU training uses BF16 mixed precision.

## Models

- **Binding classification** predicts whether a protein–peptide pair binds. Frozen ESMC encoders extract sequence representations, followed by bidirectional cross-attention, pooling, gated fusion, and a binary classification head.
- **Binding-site prediction** predicts a binding probability for each protein residue conditioned on the peptide. It supports interaction-layer transfer from the classification model, multiple site heads, and optional local-context modules.

The two tasks are trained separately. Encoder weights and task checkpoints must be prepared locally. The default encoder configuration uses a hidden size of 1152, 18 attention heads, and 36 layers.

## Datasets

Prepare the datasets locally; the code does not download them automatically.

### Binding classification

Each CSV row contains a protein sequence, a peptide sequence, and a label:

```csv
protein_sequence,peptide_sequence,binding
ACDEFGHIKLMNPQRSTVWY,ACDE,1
ACDEFGHIKLMNPQRSTVWY,WWWW,0
```

- Supported protein columns: `protein_sequence`, `protein_seq`, or `seq1`.
- Supported peptide columns: `peptide_sequence`, `peptide_seq`, or `seq2`.
- Use `binding` or `interface_binding` labels in 0/1 format. If both columns are present, `binding` takes precedence.

### Binding-site prediction

Each CSV requires `sample_id`, `protein_id`, `protein_sequence`, `peptide_sequence`, and `site_labels`:

```csv
sample_id,protein_id,protein_sequence,peptide_sequence,site_labels
sample_001,protein_001,ACDEFG,ACD,"[0,1,1,0,0,0]"
```

`site_labels` is a JSON array with one binary label per protein residue. Residue indices start at 0.

Organize each fold as follows:

```text
datasets/binding_site_5fold_cc50/
├── fold0_train.csv
├── fold0_val.csv
├── fold0_test.csv
└── ...
```

Use `--data-dir` to select another directory. With the default `--prot-max-length 512`, online encoding requires the protein sequence plus two special tokens to fit within 512 tokens.

## Training

### Binding classification

```bash
python train_frag.py \
    --train_data_path datasets/ProtFragDB_train.csv \
    --val_data_path datasets/ProtFragDB_val.csv \
    --esmc_weights_path /path/to/esmc_weights.pth \
    --batch_size 16 \
    --num_workers 0 \
    --max_epochs 30 \
    --devices 0 \
    --save_dir logs/classification
```

Existing commands can still pass `--is_pretrain False`; it does not change label selection. Use `--pretrained_ckpt /path/to/model.ckpt` to load Task 1 weights while leaving the interaction head newly initialized. If `--esmc_weights_path` is also set, the encoders keep the weights from that file. Add `--test_data_path /path/to/test.csv` to evaluate the best validation-AUROC checkpoint after training.

### Binding-site prediction

```bash
python train_binding_site.py \
    --task site \
    --data-dir datasets/binding_site_5fold_cc50 \
    --fold 0 \
    --esmcpep-weights-path /path/to/esmcpep_weights.pth \
    --site-head mlp \
    --batch-size 2 \
    --max-epochs 30 \
    --devices 0 \
    --save-dir logs/binding_site
```

Add `--pretrained-ckpt /path/to/classification.ckpt` to initialize the interaction layers from a compatible classification checkpoint. Only `affinity_head.layers.*` is transferred. Add `--run-test` to evaluate the best validation-PR-AUC checkpoint after training; test evaluation is disabled by default.

To use an existing embedding cache, add `--use-cached-embeddings --embedding-cache-dir /path/to/cache`. The cache must contain a compatible `manifest.json` and embeddings for all input sequences.

Available site heads: `mlp`, `local_conv`, `attention_guided`, `gated`, `multiscale`, `evidence_only`, and `interaction_evidence`. Local-context options: `none`, `single`, `multiscale`, and `ffn`.

## Outputs

- Classification logs and checkpoints are saved under `--save_dir`. The best checkpoint is selected by validation AUROC.
- Binding-site outputs are saved under `<save-dir>/<site-head>/fold<N>/`, including checkpoints, `config.json`, `metrics.json`, and `training_history.csv`.
- With `--run-test`, binding-site predictions are exported to `predictions/per_residue_predictions.csv` and `predictions/per_protein_predictions.csv`. The default residue probability threshold is 0.5.

For all training options:

```bash
python train_frag.py --help
python train_binding_site.py --help
```

## Acknowledgements

This project builds on [High-PepBinder](https://github.com/mqyii/High-PepBinder). We thank the authors for making their implementation available. Upstream code is distributed under the [MIT License](https://github.com/mqyii/High-PepBinder/blob/main/LICENSE); retain the applicable copyright and license notices when redistributing derived code.
