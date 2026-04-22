"""
================================================================================
Lab03 情感分析 — load_dataset.py（CSV → 清洗 → train/val/test 划分 → 词表 → [CLS]/[SEP] 向量）
================================================================================

数据处理函数与作用
  load_data               读 imdb CSV，返回 DataFrame。
  clean_text              去 HTML 噪声、保留字母数字与空格，小写。
  tokenize_text           有 HF tokenizer 则用其分词，否则按空格切词。
  load_and_preprocess_data
                          先划出 test（默认 1-train_ratio），再在原 train 中划出 val（对原 train 的 10%），
                          比例与原先一致：约 72% / 8% / 20%（train_ratio=0.8 时）。
  build_vocab             仅用训练集词频，special：<pad>=0,<unk>=1,[CLS]=2,[SEP]=3。
  text_to_indices         whitespace： [CLS]+内容+[SEP]，右补 <pad>；pretrained：句首无 CLS、max_length
                          右端以 pad（GPT-2 上常为 eos_id）填充。
  build_tensor_datasets   train / val / test 三套 TensorDataset。
  dataset_to_gpu          整集 .to(device)，训练时按 batch 切片。
  try_build_hf_tokenizer  可选加载 transformers AutoTokenizer。

张量形状示例（max_seq_len=L）
  train_x / val_x / test_x : (N_train, L)、(N_val, L)、(N_test, L)；y 为 (N,) float32。

图与表：本文件不画图。
"""

import os
import re
from collections import Counter

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def load_data(csv_path="/data/lujd/xuchenhao/Homework/26DeepLearing/Lab03/imdb_sentiment_data.csv"):
    """读取原始 CSV 文件，并返回一个包含文本和标签的 DataFrame。"""
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded. Total samples: {len(df)}")
    return df

def load_and_preprocess_data(
    csv_path="imdb_sentiment_data.csv",
):
    """先 train|test，再 train 内划分出 val；与原先逻辑一致。

    max_vocab_size / max_seq_len / tokenizer 仅保留与旧调用签名兼容，划分阶段未使用。
    """
    df = load_data(csv_path)

    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Positive samples: {(df['label'] == 1).sum()}")
    print(f"Negative samples: {(df['label'] == 0).sum()}")

    text_lengths = df["text"].apply(lambda x: len(clean_text(x).split()))
    print("\nText length statistics:")
    print(f"Mean: {text_lengths.mean():.2f}")
    print(f"Median: {text_lengths.median():.2f}")
    print(f"Max: {text_lengths.max()}")
    print(f"95th percentile: {text_lengths.quantile(0.95):.2f}")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"].values,
        df["label"].values,
        test_size=0.1,
        random_state=42,
        stratify=df["label"].values,
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=0.1,
        random_state=42,
        stratify=train_labels,
    )

    print(f"\nData split:")
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

def clean_text(text):
    """对原始评论做最基本的清洗，尽量保留情感信息，减少噪声。"""
    text = str(text)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^A-Za-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def tokenize_text(text, tokenizer=None):
    text = clean_text(text)
    # GPT2 tokenizer → 直接返回id
    if tokenizer is not None and hasattr(tokenizer, "__call__"):
        # 直接将text转换为id，返回值的形状：[text[0], 200]
        return tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=200,
            return_tensors=None
        )["input_ids"]

    return text.split()


def build_vocab(texts, max_vocab_size, tokenizer=None):
    # 如果用 GPT2 tokenizer → 不需要 vocab
    if tokenizer is not None and hasattr(tokenizer, "__call__"):
        print("[INFO] Using pretrained tokenizer, skip vocab building.")
        return None

    word_counts = Counter()
    for text in texts:
        tokens = tokenize_text(text, tokenizer=None)
        word_counts.update(tokens)

    vocab = {"<pad>": 0, "<unk>": 1, "[CLS]": 2, "[SEP]": 3}

    for token, _ in word_counts.most_common(max_vocab_size - len(vocab)):
        vocab[token] = len(vocab)

    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def text_to_indices(text, vocab, max_seq_len, tokenizer=None):
    # 输入的text是str 类型的字符串
    # ===== GPT2 路径 =====
    if tokenizer is not None and hasattr(tokenizer, "__call__"):
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_seq_len,
        )
        return encoded["input_ids"]

    # ===== 原始路径 =====
    # tokens是由字符串切分后产生的一维列表
    tokens = tokenize_text(text, tokenizer=None)

    pad_id, unk_id = vocab["<pad>"], vocab["<unk>"]
    cls_id, sep_id = vocab["[CLS]"], vocab["[SEP]"]

    max_content = max_seq_len - 2
    word_ids = [vocab.get(t, unk_id) for t in tokens]

    if len(word_ids) > max_content:
        word_ids = word_ids[:max_content]

    indices = [cls_id] + word_ids + [sep_id]

    if len(indices) < max_seq_len:
        indices += [pad_id] * (max_seq_len - len(indices))

    return indices


def build_tensor_datasets(
    train_texts,
    val_texts,
    test_texts,
    train_labels,
    val_labels,
    test_labels,
    vocab,
    max_seq_len,
    tokenizer=None,
):
    """在 CPU 上构建 train / val / test 三个 TensorDataset。"""
    train_x = torch.tensor(
        [text_to_indices(t, vocab, max_seq_len, tokenizer) for t in train_texts], dtype=torch.long
    )
    train_y = torch.tensor(train_labels, dtype=torch.float32)
    val_x = torch.tensor([text_to_indices(t, vocab, max_seq_len, tokenizer) for t in val_texts], dtype=torch.long)
    val_y = torch.tensor(val_labels, dtype=torch.float32)
    test_x = torch.tensor([text_to_indices(t, vocab, max_seq_len, tokenizer) for t in test_texts], dtype=torch.long)
    test_y = torch.tensor(test_labels, dtype=torch.float32)
    return TensorDataset(train_x, train_y), TensorDataset(val_x, val_y), TensorDataset(test_x, test_y)


def dataset_to_gpu(dataset, device):
    """整集搬到 GPU"""
    x, y = dataset.tensors[0], dataset.tensors[1]
    nb = device.type == "cuda"
    return x.to(device, non_blocking=nb), y.to(device, non_blocking=nb)


def try_build_hf_tokenizer(preferred_name="gpt2"):
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(preferred_name, use_fast=True)

        # GPT2 没有 pad token → 必须补
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # 分类任务：内容从左向右，在序列右侧用 pad（通常即 eos_id）填充。
        tokenizer.padding_side = "right"

        return tokenizer
    except Exception as exc:
        print(f"Pretrained tokenizer unavailable: {exc}")
        return None


if __name__ == "__main__":
    load_and_preprocess_data()
