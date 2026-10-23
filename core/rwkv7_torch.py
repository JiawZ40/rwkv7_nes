# -*- coding: utf-8 -*-
"""纯 PyTorch 的 RWKV-7 推理实现 —— 不依赖 fla / triton / transformers。

为什么要自己写一份
------------------
训练用的是 `flash-linear-attention`(fla) 的 RWKV7ForCausalLM，它在 Windows 上
依赖 triton，而 triton 在没有 NVIDIA GPU 的机器上连 driver 都初始化不了
（`RuntimeError: 0 active drivers`）。也就是说原实现只能在本机这种有显卡、
装了 triton-windows 的环境里跑，没法打包分发给普通用户。

这里把 fla 0.3.1 的 `fused_recurrent` 数学路径用纯 torch 重写了一遍，
参数名与 fla 完全一致（`model.layers.N.attn.xxx` / `...ffn.xxx`），
因此可以直接 `load_state_dict(torch.load('best.pt'))`。

数值对齐由 `src/verify_torch_backend.py` 把关：与 GPU 上的 fla 前向逐点比对。

额外收益：原实现的生成循环每步都把整段序列填充到 256 长度重算（为了
triton 形状稳定）。RNN 本身是马尔可夫的，这里改成增量 state 推理，
每步只算 1 个 token，CPU 上快两个数量级。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# RWKV-7 对 decay LoRA 输出固定的缩放系数（fla 里写死的常数）
W_SCALE = -0.6065306597126334


# --------------------------------------------------------------------------- #
# 结构配置
# --------------------------------------------------------------------------- #
class RWKV7Cfg:
    """与 fla 的 RWKV7Config 默认值对齐（本项目 build_model 用的就是默认值）"""

    def __init__(
        self,
        vocab_size: int = 314,
        hidden_size: int = 384,
        num_hidden_layers: int = 6,
        num_heads: int = 6,
        head_dim: int = 64,
        value_dim: Optional[int] = None,
        hidden_ratio: int = 4,
        norm_eps: float = 1e-5,
        norm_first: bool = True,
        decay_low_rank_dim: int = 64,
        gate_low_rank_dim: int = 128,
        a_low_rank_dim: int = 64,
        v_low_rank_dim: int = 16,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_dim = value_dim if value_dim is not None else hidden_size
        inter = int(hidden_size * hidden_ratio)
        self.intermediate_size = 32 * ((inter + 31) // 32)
        self.norm_eps = norm_eps
        self.norm_first = norm_first
        self.decay_low_rank_dim = decay_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.a_low_rank_dim = a_low_rank_dim
        self.v_low_rank_dim = v_low_rank_dim

    @property
    def head_v_dim(self) -> int:
        return self.value_dim // self.num_heads


# --------------------------------------------------------------------------- #
# 基础模块
# --------------------------------------------------------------------------- #
class LoRA(nn.Module):
    """与 fla.layers.rwkv6.LoRA 同构：Linear -> act -> Linear

    内部 Sequential 命名为 `lora`，保证参数名是 lora.0.weight / lora.2.weight
    / lora.2.bias，正好对上 ckpt 里的键。
    """

    def __init__(self, d_in: int, d_out: int, rank: int,
                 act: Optional[str] = "tanh", bias: bool = True):
        super().__init__()
        mods: List[nn.Module] = [nn.Linear(d_in, rank, bias=False)]
        mods.append({"tanh": nn.Tanh, "sigmoid": nn.Sigmoid}.get(act, nn.Identity)())
        mods.append(nn.Linear(rank, d_out, bias=bias))
        self.lora = nn.Sequential(*mods)

    def forward(self, x):
        return self.lora(x)


def _sqrelu(x: torch.Tensor) -> torch.Tensor:
    r = F.relu(x)
    return r * r


class State:
    """跨步增量推理需要的循环状态

    attn_prev / ffn_prev  token_shift 用到的「上一个 token 的 x」
    h                     RWKV-7 的 recurrent 状态，形状 (B, H, K, V)
    """

    __slots__ = ("attn_prev", "ffn_prev", "h")

    def __init__(self, n_layer: int, b: int, h: int, k: int, v: int, d: int,
                 device=None, dtype=None):
        self.attn_prev = [torch.zeros(b, 1, d, device=device, dtype=dtype)
                          for _ in range(n_layer)]
        self.ffn_prev = [torch.zeros(b, 1, d, device=device, dtype=dtype)
                         for _ in range(n_layer)]
        self.h = [torch.zeros(b, h, k, v, device=device, dtype=dtype)
                  for _ in range(n_layer)]

    def clone(self) -> "State":
        s = State.__new__(State)
        s.attn_prev = [t.clone() for t in self.attn_prev]
        s.ffn_prev = [t.clone() for t in self.ffn_prev]
        s.h = [t.clone() for t in self.h]
        return s


def _token_shift(x: torch.Tensor, prev: torch.Tensor):
    """fla 的 token_shift 语义：delta = x_prev - x（注意符号是反的）

    序列开头 x_prev 取 0；增量推理时取上一步保存的值。
    """
    if x.shape[1] == 1:
        delta = prev - x
    else:
        delta = torch.cat([prev, x[:, :-1]], dim=1) - x
    return delta, x[:, -1:]


# --------------------------------------------------------------------------- #
# Attention / FFN / Block
# --------------------------------------------------------------------------- #
class Attn(nn.Module):

    def __init__(self, cfg: RWKV7Cfg, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.cfg = cfg
        d, hn, hd = cfg.hidden_size, cfg.num_heads, cfg.head_dim
        vdim = cfg.value_dim

        self.x_r = nn.Parameter(torch.zeros(1, 1, d))
        self.x_w = nn.Parameter(torch.zeros(1, 1, d))
        self.x_k = nn.Parameter(torch.zeros(1, 1, d))
        self.x_v = nn.Parameter(torch.zeros(1, 1, d))
        self.x_a = nn.Parameter(torch.zeros(1, 1, d))
        self.x_g = nn.Parameter(torch.zeros(1, 1, d))

        self.k_k = nn.Parameter(torch.zeros(d))
        self.k_a = nn.Parameter(torch.zeros(d))
        self.r_k = nn.Parameter(torch.zeros(hn, hd))

        self.r_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, vdim, bias=False)
        self.o_proj = nn.Linear(vdim, d, bias=False)

        self.w_lora = LoRA(d, d, cfg.decay_low_rank_dim, "tanh", True)
        if layer_idx != 0:                      # fla: 第 0 层没有 v_lora
            self.v_lora = LoRA(d, vdim, cfg.v_low_rank_dim, None, True)
        self.a_lora = LoRA(d, d, cfg.a_low_rank_dim, None, True)
        self.g_lora = LoRA(d, vdim, cfg.gate_low_rank_dim, "sigmoid", False)

        self.g_norm = nn.GroupNorm(hn, vdim, eps=hd * cfg.norm_eps, affine=True)

    def forward(self, x: torch.Tensor, st: State, v_first: torch.Tensor):
        cfg = self.cfg
        b, t, _ = x.shape
        hn, hd, vh = cfg.num_heads, cfg.head_dim, cfg.head_v_dim
        i = self.layer_idx

        delta, prev = _token_shift(x, st.attn_prev[i])
        st.attn_prev[i] = prev

        xr = x + delta * self.x_r
        xw = x + delta * self.x_w
        xk = x + delta * self.x_k
        xv = x + delta * self.x_v
        xa = x + delta * self.x_a
        xg = x + delta * self.x_g

        r = self.r_proj(xr)
        w = W_SCALE * torch.sigmoid(self.w_lora(xw))
        k = self.k_proj(xk)
        v = self.v_proj(xv)

        if i == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, torch.sigmoid(self.v_lora(xv)))

        a = torch.sigmoid(self.a_lora(xa))
        g = self.g_lora(xg)

        # kk 用的是「乘 k_a 之前」的 k
        kk = F.normalize((k * self.k_k).view(b, t, hn, hd), dim=-1, p=2.0)
        # k = k.addcmul(k * (a - 1), k_a)  <=>  k + k*(a-1)*k_a
        k = k + k * (a - 1) * self.k_a

        r = r.view(b, t, hn, hd)
        w = w.view(b, t, hn, hd)
        k = k.view(b, t, hn, hd)
        a = a.view(b, t, hn, hd)
        v = v.view(b, t, hn, vh)

        hs = st.h[i]
        o = torch.empty(b, t, hn, vh, device=x.device, dtype=x.dtype)
        for s in range(t):
            decay = torch.exp(w[:, s])                      # (b, hn, hd)
            # 对应 fused_recurrent kernel:
            #   s[v] = sum_k (-kk[k] * h[k, v]);  b = kk * a
            #   h = exp(w) * h + b ⊗ s ;  h += k ⊗ v ;  o = h^T r
            sv = torch.einsum("bhk,bhkv->bhv", -kk[:, s], hs)
            bb = kk[:, s] * a[:, s]
            hs = decay.unsqueeze(-1) * hs + bb.unsqueeze(-1) * sv.unsqueeze(2)
            hs = hs + k[:, s].unsqueeze(-1) * v[:, s].unsqueeze(-2)
            o[:, s] = torch.einsum("bhk,bhkv->bhv", r[:, s], hs)
        st.h[i] = hs

        o = self.g_norm(o.reshape(b * t, hn * vh)).view(b, t, hn * vh)
        corr = ((r * k * self.r_k).sum(-1, keepdim=True) * v).reshape(b, t, hn * vh)
        o = (o + corr) * g
        return self.o_proj(o), v_first


class FFN(nn.Module):

    def __init__(self, cfg: RWKV7Cfg, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        d = cfg.hidden_size
        self.x_k = nn.Parameter(torch.zeros(d))
        self.key = nn.Linear(d, cfg.intermediate_size, bias=False)
        self.value = nn.Linear(cfg.intermediate_size, d, bias=False)

    def forward(self, x: torch.Tensor, st: State):
        i = self.layer_idx
        delta, prev = _token_shift(x, st.ffn_prev[i])
        st.ffn_prev[i] = prev
        xx = x + delta * self.x_k
        return self.value(_sqrelu(self.key(xx)))


class Block(nn.Module):

    def __init__(self, cfg: RWKV7Cfg, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        d = cfg.hidden_size
        if cfg.norm_first and layer_idx == 0:      # 只有第 0 层有 pre_norm
            self.pre_norm = nn.LayerNorm(d, eps=cfg.norm_eps)
        self.attn_norm = nn.LayerNorm(d, eps=cfg.norm_eps)
        self.attn = Attn(cfg, layer_idx)
        self.ffn_norm = nn.LayerNorm(d, eps=cfg.norm_eps)
        self.ffn = FFN(cfg, layer_idx)

    def forward(self, x: torch.Tensor, st: State, v_first: torch.Tensor):
        residual = self.pre_norm(x) if self.layer_idx == 0 else x
        h = self.attn_norm(residual)
        h, v_first = self.attn(h, st, v_first)
        h = residual + h
        residual = h
        h = self.ffn_norm(h)
        h = self.ffn(h, st)
        return residual + h, v_first


class Backbone(nn.Module):

    def __init__(self, cfg: RWKV7Cfg):
        super().__init__()
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps)


class RWKV7Torch(nn.Module):
    """参数命名与 fla 的 RWKV7ForCausalLM 一致：`model.*` + `lm_head.weight`"""

    def __init__(self, cfg: RWKV7Cfg):
        super().__init__()
        self.cfg = cfg
        self.model = Backbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def new_state(self, b: int = 1, device=None, dtype=None) -> State:
        cfg = self.cfg
        return State(cfg.num_hidden_layers, b, cfg.num_heads, cfg.head_dim,
                     cfg.head_v_dim, cfg.hidden_size, device, dtype)

    def forward(self, input_ids: torch.Tensor, state: Optional[State] = None,
                use_cache: bool = False):
        x = self.model.embeddings(input_ids)
        b = x.shape[0]
        if state is None:
            state = self.new_state(b, x.device, x.dtype)
        v_first = torch.zeros_like(x)
        for blk in self.model.layers:
            x, v_first = blk(x, state, v_first)
        x = self.model.norm(x)
        out = SimpleNamespace(logits=self.lm_head(x))
        if use_cache:
            return out, state
        return out

    # ---- 增量推理 ---- #
    def _dev(self, device=None):
        return device if device is not None else next(self.parameters()).device

    def prefill(self, ids: Sequence[int], state: Optional[State] = None,
                device=None, dtype=None):
        """喂入整段前缀，返回最后一个位置的 logits 与更新后的 state"""
        device = self._dev(device)
        x = torch.as_tensor([list(ids)], dtype=torch.long, device=device)
        st = state if state is not None else self.new_state(1, device, dtype)
        out = self.forward(x, st)
        return out.logits[0, -1], st

    def decode(self, tok: int, state: State, device=None):
        """单步解码：只算 1 个 token"""
        device = self._dev(device)
        x = torch.as_tensor([[int(tok)]], dtype=torch.long, device=device)
        out = self.forward(x, state)
        return out.logits[0, -1], state


# --------------------------------------------------------------------------- #
def load_torch_model(ckpt_path: str, cfg: Optional[RWKV7Cfg] = None,
                     device: str = "cpu", dtype=torch.float32) -> RWKV7Torch:
    """加载 fla 训练出的 state_dict（bf16）到纯 torch 模型

    ckpt 里存的是 bf16；CPU 上用 fp32 跑更快也更稳，load_state_dict 会自动转换。
    """
    cfg = cfg or RWKV7Cfg()
    model = RWKV7Torch(cfg)
    sd = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"权重不匹配 missing={missing} unexpected={unexpected}")
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model
