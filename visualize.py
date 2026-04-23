"""
================================================================================
Lab03 情感分析 — visualize.py（matplotlib Agg；四曲线、验证集混淆矩阵、注意力 head0）
================================================================================

函数与作用
  ensure_parent_dir        保证 save_path 父目录存在。
  plot_training_four_curves  train/val 的 loss 与 F1 共四条折线。
  plot_confusion_heatmap    二分类混淆矩阵（验证集）。
  plot_attention_heatmap   仅单头注意力；attention 权重用 vmin/vmax=0,1 铺开颜色动态范围。

本模块由 train_eval 在每轮实验目录下调用；已删除多头网格与表格 PNG。
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _savefig_current_png(path, dpi=200):
    """经真实磁盘文件句柄保存，避免 Pillow 在部分环境对非 fileno 流报错（如 _idat / fileno）。"""
    with open(path, "wb") as fh:
        plt.savefig(fh, format="png", dpi=dpi)


def plot_training_four_curves(history, save_path, title):
    """绘制 train/val 的 loss 与 F1 四条曲线。"""
    ensure_parent_dir(save_path)

    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["train_loss"], marker="o", linewidth=2, label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", linewidth=2, label="Val Loss")
    plt.plot(epochs, history["train_f1"], marker="s", linewidth=2, label="Train F1")
    plt.plot(epochs, history["val_f1"], marker="s", linewidth=2, label="Val F1")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _savefig_current_png(save_path)
    plt.close()


def plot_confusion_heatmap(y_true, y_pred, save_path, title="Confusion matrix"):
    """二分类混淆热力图（行=真实，列=预测）。"""
    ensure_parent_dir(save_path)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    vmax = max(1, int(cm.max()))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    with open(save_path, "wb") as fh:
        fig.savefig(fh, format="png", dpi=200)
    plt.close(fig)


def _to_numpy_attention(attn_weights):
    attn = attn_weights
    if hasattr(attn_weights, "detach"):
        attn = attn_weights.detach().cpu().numpy()
    return np.asarray(attn)


def align_attention_prefix(attn_2d, tokens):
    """右 padding：有效 token 在前部，对齐注意力左上角 n×n。"""
    attn_2d = np.asarray(attn_2d)
    tokens = list(tokens)
    seq_len = attn_2d.shape[-1]
    n = min(len(tokens), seq_len)
    return attn_2d[:n, :n], tokens[:n], n


def plot_attention_heatmap(attn_weights, tokens, save_path, head_index=0, title=None):
    """单个注意力头热力图"""
    ensure_parent_dir(save_path)

    attn = _to_numpy_attention(attn_weights)
    if attn.ndim == 4:
        attn = attn[0, head_index]
    elif attn.ndim == 3:
        attn = attn[head_index]

    attn, tokens, token_count = align_attention_prefix(attn, tokens)

    plt.figure(figsize=(max(8, token_count * 0.35), max(6, token_count * 0.35)))
    plt.imshow(attn, cmap="viridis", aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(np.arange(token_count), tokens, rotation=90)
    plt.yticks(np.arange(token_count), tokens)
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.title(title or f"Attention head {head_index}")
    plt.tight_layout()
    _savefig_current_png(save_path)
    plt.close()
