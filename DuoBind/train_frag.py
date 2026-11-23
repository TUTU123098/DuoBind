import click
import lightning as L
import torch

from src.pepldm.data_module import ProtFragDataModule
from src.pepldm.lightning import PepLDM2
from src.pepldm.tokenizer import ESMSeqTokenizer


@click.command()
@click.option("--train_data_path", default="datasets/ProtFragDB_train.csv")
@click.option("--val_data_path", default="datasets/ProtFragDB_val.csv")
@click.option("--test_data_path", default=None, help="Optional test set path for final evaluation.")
@click.option("--batch_size", default=32, type=int)
@click.option("--num_workers", default=4, type=int)
@click.option("--max_epochs", default=3, type=int)
@click.option("--learning_rate", default=1e-4, type=float)
@click.option("--devices", default="0", type=str)
@click.option("--save_dir", default="./logs/pretrain")
@click.option(
    "--esmc_weights_path",
    default=None,
    type=str,
    help="Local path to esmc_600m_2024_12_v0.pth. If omitted, the ESM package cache is used.",
)
@click.option(
    "--pretrained_ckpt",
    default=None,
    type=str,
    help="Optional Lightning checkpoint to load, excluding old affinity_head weights.",
)
@click.option(
    "--is_pretrain",
    default=True,
    type=bool,
    expose_value=False,
    help="Accepted for compatibility; both values use the same binary labels.",
)
def train(
    train_data_path,
    val_data_path,
    test_data_path,
    batch_size,
    num_workers,
    max_epochs,
    learning_rate,
    devices,
    save_dir,
    esmc_weights_path,
    pretrained_ckpt,
):
    L.seed_everything(42)

    tokenizer = ESMSeqTokenizer()
    data_module = ProtFragDataModule(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        test_data_path=test_data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = PepLDM2(
        model_args={},
        esm_args={},
        tokenizer=tokenizer,
        load_from_esmc=True,
        esmc_weights_path=esmc_weights_path,
        strict_esmc_load=True,
        learning_rate=learning_rate,
    )

    if pretrained_ckpt is not None:
        print(f"Loading pretrained checkpoint: {pretrained_ckpt}")
        checkpoint = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        exclude_prefixes = ("affinity_head.",)
        if esmc_weights_path is not None:
            # 当使用自定义ESMC权重（如ESMC-pep continuing pretraining权重）时，
            # 不能让pretrained_ckpt里的encoder权重覆盖回原始ESMC
            exclude_prefixes = exclude_prefixes + ("prot_encoder.", "pep_encoder.")
            print("检测到自定义esmc_weights_path，加载pretrained_ckpt时将排除prot_encoder/pep_encoder权重")
        state_dict = {
            k: v for k, v in state_dict.items()
            if not any(k.startswith(p) for p in exclude_prefixes)
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            "Reinitialized affinity_head; missing affinity_head keys:",
            sum(k.startswith("affinity_head.") for k in missing),
        )
        print("Unexpected keys:", unexpected[:10])

    device_list = [int(d) for d in devices.split(",")]
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=device_list,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        logger=L.pytorch.loggers.CSVLogger(save_dir=save_dir, name="pretrain"),
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                filename="{step}-{val_auc:.4f}",
                save_top_k=1,
                monitor="val_auc",
                mode="max",
                save_last=True,
                save_weights_only=True,
            ),
            L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
            L.pytorch.callbacks.RichProgressBar(),
        ],
    )
    trainer.fit(model, datamodule=data_module)

    if test_data_path is not None:
        print(f"Running final test evaluation on: {test_data_path}")
        _original_torch_load = torch.load
        def _patched_load(path, **kwargs):
            kwargs["weights_only"] = False
            return _original_torch_load(path, **kwargs)
        torch.load = _patched_load
        trainer.test(model, datamodule=data_module, ckpt_path="best")
        torch.load = _original_torch_load


if __name__ == "__main__":
    train()
