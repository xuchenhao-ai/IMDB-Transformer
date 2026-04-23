"""
Lab03 情感分析 — train_eval.py（训练入口；按实验类型分子目录产出）

数据：TensorDataset 在 CPU；训练用 DataLoader（num_workers=4）按 batch 传 GPU。
当前训练、曲线、混淆矩阵、注意力图仅使用 train+val；test_loader 与 test_texts 保留供日后评估。

配置：tokenizer_mode（whitespace | pretrained）、pos_encoding_mode、causal 等在 ExperimentConfig / CLI 中保留。

产出：<output>/logs/training_log.txt；
  <output>/{single|sweep|position|compare|tokenizer_compare}/<run_name>/{four_curves,confusion,attn_head0}.png、best.pt、table.csv；
  tokenizer_compare 目录另有 summary.csv（whitespace vs pretrained 各一列对比）。
  sweep 另有 group_<param>.csv、summary.csv 等。
  test 模式：在 test_result/<run_name>/ 下产出与 single 相同文件，并额外有 confusion_test.png、table.csv 中 test_* 列与 test_error_samples.csv。
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from load_dataset import (
    DATALOADER_NUM_WORKERS,
    build_tensor_datasets,
    build_vocab,
    load_and_preprocess_data,
    make_dataloaders,
    try_build_hf_tokenizer,
)
from models import AttentionClassifier, RNNClassifier
from visualize import plot_attention_heatmap, plot_confusion_heatmap, plot_training_four_curves


class TeeLogger:
    """控制台和文件双写日志，行缓冲实时落盘。"""

    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.console = sys.__stdout__
        self.file = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def _make_unraisablehook(log_path: str):
    """避免默认 hook 向已关闭的 Tee 写 stderr 时嵌套报错；同时追加到 training_log。"""

    def unraisablehook(unraisable):
        try:
            if unraisable.exc_type is not None:
                if unraisable.exc_traceback is not None:
                    body = "".join(
                        traceback.format_exception(
                            unraisable.exc_type,
                            unraisable.exc_value,
                            unraisable.exc_traceback,
                        )
                    )
                else:
                    body = "".join(
                        traceback.format_exception_only(
                            unraisable.exc_type, unraisable.exc_value
                        )
                    )
            else:
                body = f"{getattr(unraisable, 'err_msg', '') or unraisable.exc_value or unraisable!r}\n"
            block = "\n[sys.unraisablehook]\n" + body
            sys.__stderr__.write(block)
            sys.__stderr__.flush()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(block)
        except Exception:
            pass

    return unraisablehook


@dataclass
class ExperimentConfig:
    model_type: str = "attention"
    tokenizer_mode: str = "pretrained"
    tokenizer_name: str = "gpt2"

    pos_encoding_mode: str = "rope"
    causal: bool = False

    embed_dim: int = 384 # Attention模型使用
    num_heads: int = 4
    num_layers: int = 4
    hidden_dim: int = 256 # RNN模型使用

    dropout: float = 0.1
    lr: float = 1e-4
    weight_decay: float = 1e-2

    batch_size: int = 128
    epochs: int = 10

    max_vocab_size: int = 60000
    max_seq_len: int = 200


class TrainingData(NamedTuple):
    """词表 + train/val/test 的 DataLoader；val_x0 为验证集首条 (1,L) CPU 张量，供注意力图。"""

    vocab: Optional[dict]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    val_x0: torch.Tensor

    test_texts: list
    pad_id: int
    vocab_size: int
    pooling: str
    hf_tokenizer: Any

def build_data_pipeline(config: ExperimentConfig):
    tokenizer = None
    if config.tokenizer_mode == "pretrained":
        tokenizer = try_build_hf_tokenizer(config.tokenizer_name)
        if tokenizer is None:
            config.tokenizer_mode = "whitespace"
            print("[WARN] pretrained tokenizer unavailable, fallback to whitespace", flush=True)
        else:
            print("[INFO] pretrained tokenizer available", flush=True)

    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = load_and_preprocess_data()

    vocab = build_vocab(train_texts, config.max_vocab_size, tokenizer)

    train_set, val_set, test_set = build_tensor_datasets(
        train_texts,
        val_texts,

        test_texts,
        train_labels,

        val_labels,
        test_labels,

        vocab,
        config.max_seq_len,
        tokenizer=tokenizer,
    )
    if tokenizer is not None:
        pad_id = int(tokenizer.pad_token_id)
        vocab_size = len(tokenizer)
        pooling = "mean" # 将最后一层的所有向量做mean pooling，得到一个向量，再通过分类器得到logits
    else:
        pad_id = int(vocab["<pad>"])
        vocab_size = len(vocab)
        pooling = "cls" # 只对最后一层[CLS]对应的token做分类
    return vocab, train_set, val_set, test_set, list(test_texts), tokenizer, pad_id, vocab_size, pooling

def load_training_data(config: ExperimentConfig, device) -> TrainingData:
    (
        vocab,
        train_set,
        val_set,
        test_set,
        test_texts,
        hf_tokenizer,
        pad_id,
        vocab_size,
        pooling,
    ) = build_data_pipeline(config)
    pin = device.type == "cuda"
    train_loader, val_loader, test_loader = make_dataloaders(
        train_set,
        val_set,
        test_set,
        batch_size=config.batch_size,
        pin_memory=pin,
    )
    vx, _ = val_set[0]
    val_x0 = vx.unsqueeze(0).contiguous()
    print(
        f"[data] tokenizer_mode={config.tokenizer_mode!r} pad_id={pad_id} pooling={pooling} |vocab|={vocab_size} "
        f"train={len(train_set)} val={len(val_set)} test={len(test_set)} | "
        f"DataLoader num_workers={DATALOADER_NUM_WORKERS} batch_size={config.batch_size} pin_memory={pin}",
        flush=True,
    )
    return TrainingData(
        vocab,
        train_loader,
        val_loader,
        test_loader,
        val_x0,
        test_texts,
        pad_id,
        vocab_size,
        pooling,
        hf_tokenizer,
    )


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_or_fail():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found.")
    return torch.device(f"cuda:0")


def ensure_logs_dir(output_root: str) -> str:
    # os.path.join表示将output_root和"logs"连接起来，如果output_root是"output"，则logs是"output/logs"
    logs = os.path.join(output_root, "logs")
    # exist_ok=True表示如果logs目录已经存在，则不创建，否则创建
    os.makedirs(logs, exist_ok=True)
    # 返回logs目录下的training_log.txt文件路径
    return os.path.join(logs, "training_log.txt")


def experiment_run_dir(output_root: str, experiment_tag: str, run_name: str) -> str:
    """用于构建实验结果保存的目录，例如 output/sweep/sweep_lr_1e-4/"""
    d = os.path.join(output_root, experiment_tag, run_name)
    os.makedirs(d, exist_ok=True)
    return d


def save_rows_csv(rows, csv_path: str):
    d = os.path.dirname(csv_path)
    if d:
        os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def compute_metrics(y_true, y_prob): # y_true是真实标签，y_prob是预测概率
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }



def build_model(config, vocab_size, pad_id, pooling: str = "cls"):
    if config.model_type == "attention":
        return AttentionClassifier(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_len=config.max_seq_len,
            pad_id=pad_id,
            dropout=config.dropout,
            use_sinusoidal_pos=(config.pos_encoding_mode == "sinusoidal"),
            use_rope=(config.pos_encoding_mode == "rope"),
            causal=config.causal,
            pooling=pooling,
        )

    if config.model_type == "rnn":
        return RNNClassifier(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            pad_id=pad_id,
            dropout=config.dropout,
        )

    raise ValueError(f"Unsupported model_type: {config.model_type}")


def unwrap_output(outputs):
    if isinstance(outputs, tuple):
        return outputs[0], outputs[1]
    return outputs, None


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    y_parts, p_parts = [], []
    total_loss, total_n = 0.0, 0
    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = unwrap_output(model(x))
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)
        y_parts.append(y.detach().cpu())
        p_parts.append(probs.detach().cpu())
        total_loss += loss.item() * y.size(0)
        total_n += y.size(0)

    y_true = torch.cat(y_parts).numpy()
    y_prob = torch.cat(p_parts).numpy()
    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / max(total_n, 1)
    return metrics


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    """仅评估指标；注意力可视化在 train_model 中对 val_x0 单独前向。"""
    model.eval()
    y_parts, p_parts = [], []
    total_loss, total_n = 0.0, 0
    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = unwrap_output(model(x))
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits)
        y_parts.append(y.detach().cpu())
        p_parts.append(probs.detach().cpu())
        total_loss += loss.item() * y.size(0)
        total_n += y.size(0)

    y_true = torch.cat(y_parts).numpy()
    y_prob = torch.cat(p_parts).numpy()
    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / max(total_n, 1)
    return metrics



@torch.no_grad()
def collect_loader_probs_and_labels(model, loader, device):
    """顺序遍历任意 DataLoader，返回 (probs, y_true) 一维 numpy。"""
    model.eval()
    p_parts, y_parts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits, _ = unwrap_output(model(x))
        p_parts.append(torch.sigmoid(logits).detach().cpu())
        y_parts.append(y)
    probs = torch.cat(p_parts).numpy().ravel()
    y_true = torch.cat(y_parts).numpy().ravel()
    return probs, y_true


def best_hyperparam_config() -> ExperimentConfig:
    """固定一组表现较好的超参，供 test 模式完整训练 + 测试集评估使用。"""
    return ExperimentConfig(
        model_type="attention",
        tokenizer_mode="pretrained",
        tokenizer_name="gpt2",
        pos_encoding_mode="rope",
        causal=False,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        hidden_dim=256,
        dropout=0.1,
        lr=3e-4,
        weight_decay=1e-3,
        batch_size=512,
        epochs=10,
        max_vocab_size=60000,
        max_seq_len=200,
    )
def _append_test_metrics_and_error_samples(
    run_dir: str,
    test_metrics: dict,
    test_texts: list,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    run_title: str,
    max_error_print: int = 8,
    max_error_csv: int = 50,
):
    """在 table.csv 中追加测试集指标；保存测试混淆矩阵图；打印并可选落盘错误样例。"""
    y_pred = (y_prob >= 0.5).astype(int)
    plot_confusion_heatmap(
        y_true,
        y_pred,
        os.path.join(run_dir, "confusion_test.png"),
        title=f"{run_title} test confusion",
    )

    table_path = os.path.join(run_dir, "table.csv")
    if os.path.isfile(table_path):
        df = pd.read_csv(table_path)
    else:
        df = pd.DataFrame()
    for k, v in test_metrics.items():
        df[f"test_{k}"] = [v]
    df.to_csv(table_path, index=False)

    err_rows = []
    for i in range(len(y_true)):
        if int(y_true[i]) != int(y_pred[i]):
            snippet = test_texts[i].replace("\n", " ").strip()
            if len(snippet) > 240:
                snippet = snippet[:240] + "…"
            err_rows.append(
                {
                    "index": i,
                    "text_snippet": snippet,
                    "y_true": int(y_true[i]),
                    "y_pred": int(y_pred[i]),
                    "prob_positive": float(y_prob[i]),
                }
            )

    if err_rows:
        err_csv = os.path.join(run_dir, "test_error_samples.csv")
        save_rows_csv(err_rows[:max_error_csv], err_csv)
        print(f"\n[{run_title}] 测试集错分样例（最多 {max_error_print} 条，详见 {err_csv}）:", flush=True)
        for row in err_rows[:max_error_print]:
            print(
                f"  idx={row['index']} true={row['y_true']} pred={row['y_pred']} "
                f"P(y=1)={row['prob_positive']:.4f} | {row['text_snippet']}",
                flush=True,
            )
    else:
        print(f"\n[{run_title}] 测试集无错分（或样本为空）。", flush=True)


def test_model(output_root: str, device):
    """使用最优超参跑一次与 single 相同的训练与产出，目录为 output/test_result/<run_name>/；最后在测试集上评估并记录混淆矩阵与错分样例。"""
    cfg = best_hyperparam_config()
    set_seed(42)
    data = load_training_data(cfg, device)
    run_name = "attention_best"
    result = run_one(cfg, output_root, "test_result", run_name, device, data)

    ckpt = torch.load(result["ckpt_path"], map_location=device)
    model = build_model(
        cfg,
        data.vocab_size,
        data.pad_id,
        pooling=data.pooling,
    ).to(device)
    model.load_state_dict(ckpt["final_best_state_dict"])

    criterion = nn.BCEWithLogitsLoss()
    test_m = evaluate(model, data.test_loader, criterion, device)
    probs, y_true = collect_loader_probs_and_labels(model, data.test_loader, device)

    print(
        f"\n[{run_name}] 测试集 | Accuracy={test_m['accuracy']:.4f} Precision={test_m['precision']:.4f} "
        f"Recall={test_m['recall']:.4f} F1-score={test_m['f1']:.4f} Loss={test_m['loss']:.4f}",
        flush=True,
    )
    _append_test_metrics_and_error_samples(
        result["run_dir"],
        test_m,
        data.test_texts,
        y_true,
        probs,
        run_title=run_name,
    )


def train_model(
    model,
    vocab: Optional[dict],
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_x0: torch.Tensor,
    config: ExperimentConfig,
    device,
    run_dir: str,  # 本次实验所有结果保存的目录
    run_title: str,
    pad_id: int = 0,
    hf_tokenizer: Any = None,
):
    """在 run_dir 下写入曲线、混淆矩阵（验证集）、head0（验证集首条）、checkpoint、table.csv。不使用 X_test。"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    print(
        f"DataLoader → {device}: train={len(train_loader.dataset)} val={len(val_loader.dataset)}",
        flush=True,
    )

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None

    t0 = time.time()
    for epoch in range(1, config.epochs + 1):
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_f1"].append(train_m["f1"])
        history["val_f1"].append(val_m["f1"])

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[{run_title}] epoch {epoch}/{config.epochs} | loss={train_m['loss']:.4f} | "
            f"Accuracy={val_m['accuracy']:.4f} Precision={val_m['precision']:.4f} "
            f"Recall={val_m['recall']:.4f} F1-score={val_m['f1']:.4f}",
            flush=True,
        )

    # 加载历史最优的模型参数
    model.load_state_dict(best_state)

    # 将历史最优在验证集上跑一遍
    val_m = evaluate(model, val_loader, criterion, device)

    ckpt_path = os.path.join(run_dir, "best.pt")
    # 1. 保存模型参数与配置到best.pt文件
    torch.save(
        {
            "final_best_state_dict": best_state,  # 历史最优的模型参数
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "config": asdict(config),
        },
        ckpt_path,
    )
    # 2. 画训练曲线four_curves.png，包括train_loss、val_loss、train_f1、val_f1
    plot_training_four_curves(history, os.path.join(run_dir, "four_curves.png"), f"{run_title} curves")

    # 3. 画验证集混淆矩阵 confusion.png
    probs, y_true_np = collect_loader_probs_and_labels(model, val_loader, device)
    y_pred_np = (probs >= 0.5).astype(int)
    plot_confusion_heatmap(
        y_true_np,
        y_pred_np,
        os.path.join(run_dir, "confusion.png"),
        title=f"{run_title} val confusion",
    )

    # 4. 画验证集注意力图attn_head0.png，选择第一张图的最后一层注意力的第一个注意力头
    if config.model_type == "attention":
        with torch.no_grad():
            x0 = val_x0.to(device, non_blocking=True)
            _, attn = unwrap_output(model(x0))
        if attn is not None:
            ids = val_x0[0].detach().cpu().tolist()
            if hf_tokenizer is not None:
                tokens = [hf_tokenizer.convert_ids_to_tokens(i) for i in ids if i != pad_id]
            else:
                assert vocab is not None
                ivocab = {idx: tk for tk, idx in vocab.items()}
                tokens = [ivocab.get(i, "<unk>") for i in ids if i != pad_id]
            plot_attention_heatmap(
                attn,
                tokens,
                os.path.join(run_dir, "attn_head0.png"),
                0,
                f"{run_title} head0 (val[0])",
            )
    # 5. 保存记录表table.csv
    row = {
        **{k: str(v) for k, v in asdict(config).items()}, # config 中的参数
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_accuracy": val_m["accuracy"],
        "val_precision": val_m["precision"],
        "val_recall": val_m["recall"],
        "val_f1": val_m["f1"],
        "val_loss": val_m["loss"],
    }
    save_rows_csv([row], os.path.join(run_dir, "table.csv"))

    return {
        "run_title": run_title,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_metrics": val_m,
        "history": history,
        "train_seconds": time.time() - t0,
        "ckpt_path": ckpt_path,
        "run_dir": run_dir,
    }

# 整理出所有配置和结果
def make_row(config, result, group_name, value):
    vm = result["val_metrics"] # 展开evaluate函数的结果，包括accuracy、precision、recall、f1、loss
    return {
        "group": group_name,
        "parameter_choice": value,
        "accuracy": vm["accuracy"],
        "precision": vm["precision"],
        "recall": vm["recall"],
        "f1": vm["f1"],
        "best_val_loss": result["best_val_loss"],
        "best_epoch": result["best_epoch"],
        "config": json.dumps(asdict(config), ensure_ascii=False),
    }


def run_one(
    config,
    output_root: str,
    experiment_tag: str,
    run_name: str,
    device,
    data: TrainingData,
):
    # 打印单次实验的日志
    run_dir = experiment_run_dir(output_root, experiment_tag, run_name)
    print(f"\n========== {run_name} ==========")
    print(f"run_dir: {run_dir}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Config: {json.dumps(asdict(config), ensure_ascii=False)}", flush=True)

    model = build_model(
        config,
        data.vocab_size,
        data.pad_id,
        pooling=data.pooling,
    ).to(device)
    return train_model(
        model,
        data.vocab,
        data.train_loader,
        data.val_loader,
        data.val_x0,
        config,
        device,
        run_dir,
        run_name,
        pad_id=data.pad_id,
        hf_tokenizer=data.hf_tokenizer,
    )


def run_hparam_sweep(base: ExperimentConfig, output_root: str, device, data: TrainingData):
    groups = {
        "num_heads": [2, 4, 8],
        "embed_dim": [128, 256, 384],
        "num_layers": [1, 2, 3],
        "weight_decay": [0.0, 1e-3, 1e-2],
        "lr": [1e-4, 3e-4, 1e-3],
    }
    # 输出总目录为output/sweep/
    # 子目录为output/sweep/"run_name", 例如output/sweep/sweep_num_heads_2，共3*5 = 15个子目录
    sweep_root = os.path.join(output_root, "sweep")
    os.makedirs(sweep_root, exist_ok=True)
    all_rows = []
    for g, vals in groups.items():
        group_rows = []
        print(f"\n===== Hyperparameter group: {g} =====", flush=True)
        for v in vals:
            cfg = ExperimentConfig(**asdict(base))
            setattr(cfg, g, v)
            run_name = f"sweep_{g}_{str(v).replace('.', '_')}" # 实验结果保存的目录名
            res = run_one(cfg, output_root, "sweep", run_name, device, data) # 例：output/sweep/sweep_num_heads_2
            row = make_row(cfg, res, g, v) # 整理出全部参数和结果
            group_rows.append(row)
            all_rows.append(row)
        if group_rows:
            # 整组对比表，直接存储于output/sweep/，例如output/sweep/group_num_heads.csv
            save_rows_csv(group_rows, os.path.join(sweep_root, f"group_{g}.csv"))

    if all_rows:
        # 所有组对比表，存储于output/sweep/，例如output/sweep/summary.csv
        save_rows_csv(all_rows, os.path.join(sweep_root, "summary.csv"))


def run_position_study(base: ExperimentConfig, output_root: str, device, data: TrainingData):
    rows = []
    for mode in ["sinusoidal", "rope", "none"]:
        cfg = ExperimentConfig(**asdict(base))
        cfg.pos_encoding_mode = mode
        run_name = f"pos_{mode}"
        res = run_one(cfg, output_root, "position", run_name, device, data)
        rows.append(make_row(cfg, res, "pos_encoding_mode", mode))
    save_rows_csv(rows, os.path.join(output_root, "position", "summary.csv"))


def run_model_compare(base: ExperimentConfig, output_root: str, device, data: TrainingData):
    rows = []
    for mt in ["attention", "rnn"]:
        cfg = ExperimentConfig(**asdict(base))
        cfg.model_type = mt
        if mt == "rnn":
            cfg.pos_encoding_mode = "none"
        run_name = f"{mt}_baseline"
        res = run_one(cfg, output_root, "compare", run_name, device, data)
        rows.append(make_row(cfg, res, "model_type", mt))
    save_rows_csv(rows, os.path.join(output_root, "compare", "summary.csv"))


def run_tokenizer_compare(base: ExperimentConfig, output_root: str, device):
    """whitespace 与 pretrained 各重建数据与 DataLoader 并各训一套；结果写入 tokenizer_compare/summary.csv。"""
    rows = []
    for tm in ["whitespace", "pretrained"]:
        cfg = ExperimentConfig(**asdict(base))
        if tm == "pretrained":
            cfg.tokenizer_name = "gpt2"
        else:
            cfg.tokenizer_name = "whitespace"
        cfg.tokenizer_mode = tm
        data = load_training_data(cfg, device)
        run_name = f"tok_{tm}"
        res = run_one(cfg, output_root, "tokenizer_compare", run_name, device, data)
        rows.append(make_row(cfg, res, "tokenizer_mode", tm))
    save_rows_csv(rows, os.path.join(output_root, "tokenizer_compare", "summary.csv"))


def parse_args():
    p = argparse.ArgumentParser(description="IMDB sentiment experiments")
    p.add_argument(
        "--mode",
        choices=["single", "sweep", "position", "compare", "tokenizer_compare", "test", "all"],
        default="all",
    )
    p.add_argument("--output-dir", type=str, default="output")

    p.add_argument("--model-type", choices=["attention", "rnn"], default="attention")
    p.add_argument("--tokenizer-mode", choices=["whitespace", "pretrained"], default="pretrained")
    p.add_argument("--tokenizer-name", type=str, default="gpt2")

    p.add_argument("--pos-encoding-mode", choices=["sinusoidal", "rope", "none"], default="rope")
    p.add_argument("--causal", action="store_true")

    p.add_argument("--embed-dim", type=int, default=256) # Attention模型使用
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--hidden-dim", type=int, default=256) # RNN模型使用

    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)

    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=10)

    p.add_argument("--max-vocab-size", type=int, default=60000)
    p.add_argument("--max-seq-len", type=int, default=200)
    return p.parse_args()


def build_config(args):
    return ExperimentConfig(
        model_type=args.model_type,
        tokenizer_mode=args.tokenizer_mode,
        tokenizer_name=args.tokenizer_name,
        pos_encoding_mode=args.pos_encoding_mode,
        causal=args.causal,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_vocab_size=args.max_vocab_size,
        max_seq_len=args.max_seq_len,
    )


def main():
    args = parse_args()
    cfg = build_config(args)
    set_seed(42)

    output_root = os.path.abspath(args.output_dir)
    log_file = ensure_logs_dir(output_root)
    logger = TeeLogger(log_file)
    sys.unraisablehook = _make_unraisablehook(log_file)
    sys.stdout = logger
    sys.stderr = logger

    try:
        print(f"Output root: {output_root}", flush=True)
        print("Start training pipeline...", flush=True)

        device = get_device_or_fail()

        if args.mode == "tokenizer_compare":
            run_tokenizer_compare(cfg, output_root, device)
            return

        if args.mode == "test":
            test_model(output_root, device)
            return

        data = load_training_data(cfg, device)

        if args.mode == "single":
            run_one(cfg, output_root, "single", f"{cfg.model_type}_single", device, data)
            return

        if args.mode == "sweep":
            run_hparam_sweep(cfg, output_root, device, data)
            return

        if args.mode == "position":
            run_position_study(cfg, output_root, device, data)
            return

        if args.mode == "compare":
            run_model_compare(cfg, output_root, device, data)
            return

        if args.mode == "all":
            run_hparam_sweep(cfg, output_root, device, data)
            run_position_study(cfg, output_root, device, data)
            run_model_compare(cfg, output_root, device, data)
            run_tokenizer_compare(cfg, output_root, device)
            return

    except Exception:
        # 直接写盘与真实 stderr，避免 Tee 在崩溃路径上未 flush 时丢栈
        err = traceback.format_exc()
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n[FATAL] uncaught in main:\n" + err)
        except OSError:
            pass
        sys.__stderr__.write("\n[FATAL] uncaught in main:\n" + err)
        raise
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        logger.flush()
        logger.close()


if __name__ == "__main__":
    main()
    # CUDA_VISIBLE_DEVICES=5 python train_eval.py --batch-size 512
    # 可自选参数：--mode all | single | sweep | position | compare | tokenizer_compare | test
