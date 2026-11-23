import pandas as pd
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from .tokenizer import ESMSeqTokenizer


class ProtFragDataset(Dataset):  # 定义蛋白质片段数据集类
    PROTEIN_COLUMNS = ("protein_sequence", "protein_seq", "seq1")
    PEPTIDE_COLUMNS = ("peptide_sequence", "peptide_seq", "seq2")

    def __init__(
        self,
        csv_path: str,  # CSV文件路径
        tokenizer: ESMSeqTokenizer,  # ESM序列分词器
        prot_max_length: int = 512,  # 蛋白质序列最大长度
        pep_max_length: int = 50,  # 多肽序列最大长度
    ):
        self.df = pd.read_csv(csv_path)  # 读取CSV文件
        self.tokenizer = tokenizer  # 保存分词器
        self.prot_max_length = prot_max_length  # 保存蛋白质最大长度
        self.pep_max_length = pep_max_length  # 保存多肽最大长度

        if "binding" in self.df.columns:
            self.df["label"] = self.df["binding"].astype(float)
        elif "interface_binding" in self.df.columns:
            self.df["label"] = self.df["interface_binding"].astype(float)
        else:
            raise KeyError("Expected `binding` or `interface_binding` column.")
        if not self.df["label"].isin((0, 1)).all():
            raise ValueError("Binding labels must be 0 or 1.")

        self.protein_col = self._first_existing_column(self.PROTEIN_COLUMNS)
        self.peptide_col = self._first_existing_column(self.PEPTIDE_COLUMNS)

    def _first_existing_column(self, candidates):
        for column in candidates:
            if column in self.df.columns:
                return column
        raise KeyError(
            f"CSV missing required column. Expected one of {candidates}, "
            f"but found {list(self.df.columns)}"
        )

    def __len__(self):  # 返回数据集大小
        return len(self.df)

    def __getitem__(self, index):  # 按索引返回单条样本
        row = self.df.iloc[index]  # 取出对应行

        prot_tokenized = self.tokenizer(  # 对蛋白质序列编码
            row[self.protein_col],
            padding="max_length",
            truncation=True,
            max_length=self.prot_max_length,
            return_tensors="pt",
        )

        pep_tokenized = self.tokenizer(  # 对多肽序列编码
            row[self.peptide_col],
            padding="max_length",
            truncation=True,
            max_length=self.pep_max_length,
            return_tensors="pt",
        )

        return {
            "prot_input_ids": prot_tokenized["input_ids"].squeeze(0),  # 蛋白质token id
            "prot_attention_mask": prot_tokenized["attention_mask"].squeeze(0),  # 蛋白质mask
            "pep_input_ids": pep_tokenized["input_ids"].squeeze(0),  # 多肽token id
            "pep_attention_mask": pep_tokenized["attention_mask"].squeeze(0),  # 多肽mask
            "label": torch.tensor(row["label"], dtype=torch.float),  # 分类标签（0/1）
        }


class ProtFragDataModule(L.LightningDataModule):  # 定义Lightning数据模块
    def __init__(
        self,
        train_data_path: str,  # 训练集路径
        val_data_path: str,  # 验证集路径
        tokenizer: ESMSeqTokenizer,  # 分词器
        batch_size: int = 32,  # 批次大小
        num_workers: int = 4,  # 数据加载线程数
        prot_max_length: int = 512,  # 蛋白质最大长度
        pep_max_length: int = 50,  # 多肽最大长度
        test_data_path: str = None,  # 测试集路径（可选）
    ):
        super().__init__()
        self.train_data_path = train_data_path  # 保存训练集路径
        self.val_data_path = val_data_path  # 保存验证集路径
        self.test_data_path = test_data_path  # 保存测试集路径
        self.tokenizer = tokenizer  # 保存分词器
        self.batch_size = batch_size  # 保存批次大小
        self.num_workers = num_workers  # 保存线程数
        self.prot_max_length = prot_max_length  # 保存蛋白质最大长度
        self.pep_max_length = pep_max_length  # 保存多肽最大长度

    def setup(self, stage=None):  # 初始化数据集
        self.train_dataset = ProtFragDataset(  # 创建训练集
            csv_path=self.train_data_path,
            tokenizer=self.tokenizer,
            prot_max_length=self.prot_max_length,
            pep_max_length=self.pep_max_length,
        )
        self.val_dataset = ProtFragDataset(  # 创建验证集
            csv_path=self.val_data_path,
            tokenizer=self.tokenizer,
            prot_max_length=self.prot_max_length,
            pep_max_length=self.pep_max_length,
        )
        if self.test_data_path is not None:
            self.test_dataset = ProtFragDataset(  # 创建测试集
                csv_path=self.test_data_path,
                tokenizer=self.tokenizer,
                prot_max_length=self.prot_max_length,
                pep_max_length=self.pep_max_length,
            )

    def _make_weighted_sampler(self):
        """
        按类别频率倒数构造采样权重，重采样后正负样本接近1:1
        """
        labels = self.train_dataset.df["label"].values.astype(int)
        class_counts = np.bincount(labels)  # [neg_count, pos_count]
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return sampler

    def train_dataloader(self):  # 返回训练集DataLoader
        sampler = self._make_weighted_sampler()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):  # 返回验证集DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):  # 返回测试集DataLoader
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
