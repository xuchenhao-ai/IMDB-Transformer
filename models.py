"""
================================================================================
Lab03 情感分析 — models.py（位置编码、手动多头注意力、分类器；满足作业「不调 nn.MultiheadAttention」）
================================================================================

模块组成与作用
  SinusoidalPositionalEncoding    标准 sin/cos 位置向量，与 token 维对齐后相加；缓冲形状约 (1, max_len, d_model)。
  RotaryPositionalEncoding       RoPE：在 Q/K 上按 head_dim 旋转（见 forward(q,k)）。
  MultiHeadAttention             手写 MHA：Q/K/V 线性、缩放点积、padding mask、可选因果上三角 mask、softmax、输出投影；
                                 满足 README「手动实现 MHA、参照 Attention Is All You Need」；不使用 nn.MultiheadAttention。
  AttentionClassifier            Embedding →（可选正弦位置）→ 多层 (MHA + LayerNorm + 残差) → 池化（[CLS] 位 或
                                 非 padding 的 mean pooling）→ 线性二分类；causal=True 时因果 mask；return (logits, last_layer_attn)。
  RNNClassifier                  nn.RNN + 最后非 pad 时间步池化 + 线性，与 Attention 对比。

典型张量形状（batch=B，长=L，维=D，头数=H，层内 head_dim=D/H）
  输入 token id       x : (B, L)
  嵌入 / 隐状态       hidden : (B, L, D)
  注意力权重（每层）   attn_weights : (B, H, L, L)
   logits（二分类）    : (B,)，与 BCEWithLogitsLoss 单 logit 一致。

图与表：本文件不绘图；注意力权重由 train_eval 传入 visualize 保存为 PNG。
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """标准正弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1) # position类型为[max_len, 1]的tensor
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        ) # div_term类型为[d_model/2]的tensor

        position_encoder = torch.zeros(max_len, d_model) # position_encoder类型为[max_len, d_model]的tensor
        position_encoder[:, 0::2] = torch.sin(position * div_term) # position_encoder[:, 0::2]类型为[max_len, d_model/2]的tensor
        position_encoder[:, 1::2] = torch.cos(position * div_term) # position_encoder[:, 1::2]类型为[max_len, d_model/2]的tensor

        # 缓存为 buffer，模型保存和加载时会一起处理，但不会参与梯度更新。
        self.register_buffer("position_encoder", position_encoder.unsqueeze(0)) # unsqueeze(0)表示在第0维上增加一维，对应后文batch_size维度

    def forward(self, x: torch.Tensor):
        """
        把预先计算好的位置编码加到 token embedding 上
        x: [batch_size, seq_len, d_model]
        """
        return x + self.position_encoder[:, : x.size(1)]


class RotaryPositionalEncoding(nn.Module):
    """RoPE 位置编码，只旋转每个 head 内部的一部分维度。"""

    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        q, k: [batch_size, num_heads, seq_len, head_dim]
        RoPE 的核心思想不是把位置向量加到 token 上，而是把位置信息融入 query/key 的几何关系里。
        """
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even") # 奇数维度自动报错

        seq_len = q.size(-2)
        device = q.device
        dtype = q.dtype

        position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1) # position类型为[seq_len, 1]的tensor
        freq = torch.arange(0, self.head_dim, 2, device=device, dtype=dtype)
        freq = 1.0 / (10000 ** (freq / self.head_dim))
        angles = position * freq.unsqueeze(0) # angles类型为[seq_len, self.head_dim/2]的tensor

        # 每对相邻维度 (2i, 2i+1) 共用同一角度；需把 cos/sin 在最后一维扩成 head_dim，才能与 q、k 逐元相乘。
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0).to(dtype=q.dtype)  # [1, 1, seq_len, head_dim/2]
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0).to(dtype=q.dtype)
        # repeat_interleave(input, repeats, dim=None) 表示在指定维度上重复input的元素，repeats表示每个元素重复的次数
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # [1, 1, seq_len, head_dim]
        sin = torch.repeat_interleave(sin, 2, dim=-1)

        def rotate_half(x):
            x1 = x[..., ::2]  # 交错 pair，与上面对 (2i,2i+1) 的成对旋转一致
            x2 = x[..., 1::2]
            rotated = torch.stack((-x2, x1), dim=-1)
            return rotated.flatten(-2)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin 
        return q, k # q, k类型为[batch_size, num_heads, seq_len, head_dim]的tensor


class MultiHeadAttention(nn.Module):
    """手动实现的多头自注意力层，不依赖 nn.MultiheadAttention。"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_rope: bool = False):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        # 嵌入维度、头数、每个头维度、是否使用RoPE
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope

        # Q, K, V, 输出矩阵
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEncoding(self.head_dim) if use_rope else None

    def _reshape_to_heads(self, x: torch.Tensor):
        """把 [B, L, D] 重排成 [B, H, L, Hd]，便于每个 head 独立计算。"""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2) # transpose(1, 2)表示交换第1维和第2维

    # causal: 是否启用因果 mask，仅当需要自回归MHA时才设为True，BERT为False
    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, causal: bool = False):
        """
        x: [batch_size, seq_len, embed_dim]
        key_padding_mask: [batch_size, seq_len]，pad 位置为 True
        causal: 是否启用因果 mask
        return:
            attn_output: [batch_size, seq_len, embed_dim]
            attn_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        q = self._reshape_to_heads(self.q_proj(x))
        k = self._reshape_to_heads(self.k_proj(x))
        v = self._reshape_to_heads(self.v_proj(x))

        if self.rope is not None:
            q, k = self.rope(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # scores类型为[batch_size, num_heads, seq_len, seq_len]的tensor

        # 分类任务不需要因果mask
        if causal:
            causal_mask = torch.triu( # triu(a, diagonal=1)表示取上三角矩阵，diagonal=1表示对角线偏移1
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), 
                diagonal=1,
            )
            # float("-inf")表示将causal_mask为True的位置填充为-inf，只保留对角线及以下的部分
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")) 

        if key_padding_mask is not None:
            key_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # key_mask类型为[batch_size, 1, 1, seq_len]的tensor
            scores = scores.masked_fill(key_mask, float("-inf")) # padding对应的列填充为-inf
            query_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1) # query_mask类型为[batch_size, 1, seq_len, 1]的tensor
            scores = scores.masked_fill(query_mask, 0.0)

        attn_weights = torch.softmax(scores, dim=-1)

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(-1), 0.0)

        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        # 变形回[batch_size, seq_len, embed_dim]的tensor
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        # 输出层线性变换
        attn_output = self.out_proj(attn_output) 

        if key_padding_mask is not None:
            attn_output = attn_output.masked_fill(key_padding_mask.unsqueeze(-1), 0.0) # padding对应的行填充为0

        return attn_output, attn_weights # attn_output类型为[batch_size, seq_len, embed_dim]的tensor, attn_weights类型为[batch_size, num_heads, seq_len, seq_len]的tensor


class AttentionClassifier(nn.Module):
    """基于多层自注意力的文本二分类模型。"""

    # 词表大小，嵌入维度，头数，层数，最大长度，<padding> id，dropout率
    # 是否使用正弦位置编码，是否使用手动多头注意力，是否使用RoPE，是否启用因果 mask
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 8,
        max_len: int = 200,
        pad_id: int = 0,
        dropout: float = 0.1,

        use_sinusoidal_pos: bool = False,
        use_rope: bool = True,
        causal: bool = False,
        pooling: str = "cls",
    ):
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        self.pooling = pooling
        self.pad_id = pad_id
        self.causal = causal
        self.use_sinusoidal_pos = use_sinusoidal_pos
        self.use_rope = use_rope

        # self.embed可看成一个 (vocab_size × embed_dim) 矩阵，嵌入是可训练的
        # 指定pad token 在词表中次序为 0，意味着pad token永远不参与学习，并且嵌入向量固定为0
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=max_len)

        self.layers = nn.ModuleList(
            [
                MultiHeadAttention(embed_dim, num_heads, dropout=dropout, use_rope=use_rope)
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, seq_len]
        return:
            logits: [batch_size]
            last_attn: [batch_size, num_heads, seq_len, seq_len]
            hidden: [batch_size, seq_len, embed_dim]
        """
        key_padding_mask = x.eq(self.pad_id)
        hidden = self.embed(x)

        if self.use_sinusoidal_pos:
            hidden = self.pos_encoder(hidden)
        # RoPE 仅在 MultiHeadAttention 内对分头后的 Q/K 施加（见 self.layers[*].rope），
        # 不能在 [B,L,D] 的 hidden 上调用 RotaryPositionalEncoding.forward(q,k)。

        last_attn = None
        # 层数*(多头注意力+层归一化+残差连接+层归一化)，不使用因果mask
        for attn_layer, norm in zip(self.layers, self.norms):
            attn_out, attn_w = attn_layer(
                hidden,
                key_padding_mask=key_padding_mask,
                causal=self.causal,
            )
            hidden = norm(hidden + self.dropout(attn_out))
            last_attn = attn_w

        if self.pooling == "cls":
            # 右 padding：白空格分词时序列第 0 位为 [CLS]，与 BERT 一致取句首向量。
            pooled = hidden[:, 0, :]
        else:
            # 非 padding 位置做 mean pooling（如 GPT-2 用 EOS 作 pad 时，pad 位不计入平均）。
            mask = (~key_padding_mask).float().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            pooled = (hidden * mask).sum(dim=1) / denom
        logits = self.classifier(self.dropout(pooled)).squeeze(-1)
        return logits, last_attn


class RNNClassifier(nn.Module):
    """基于 RNN 的文本二分类模型，用于和 Attention 模型对比。"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 8,
        pad_id: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            # batch_first=True表示输入的形状为[batch_size, seq_len, embed_dim]
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, seq_len]
        return: [batch_size]
        """
        emb = self.embed(x)
        # .rnn的输出有两个
        # out.shape = [batch_size, seq_len, hidden_dim], 每个时间步的输出
        # hidden_states.shape = [num_layers, batch_size, hidden_dim], 每个时间步的隐藏状态
        out, _ = self.rnn(emb)
        # 右 padding：末位常为 pad，取每个样本最后一个非 pad 时间步。
        lengths = (x != self.pad_id).sum(dim=1)
        last_idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(x.size(0), device=x.device)
        pooled = out[batch_idx, last_idx, :] # 高级索引，根据batch_idx和last_idx索引出每个样本最后一个非 pad 时间步的输出
        logits = self.classifier(self.dropout(pooled)).squeeze(-1)
        return logits
