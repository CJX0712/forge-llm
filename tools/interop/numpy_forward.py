"""
numpy_forward.py —— 用 numpy 独立实现 Llama forward，与 forge-llm 的 JS 实现交叉验证

这是 forge-llm 最硬的一条外部证据：
  - 权重来自 llama.cpp 官方 gguf-py 写出的 GGUF（不经 forge-llm 之手）
  - 本脚本用 numpy 从零实现同一套数学（RMSNorm / RoPE / GQA / SwiGLU / 因果注意力）
  - 两者 logits 若逐位（或 ≤1e-4）一致，说明 JS 实现与官方格式+数学都正确

用法：
  python numpy_forward.py <model.gguf.npz> --ids 2,3,4,5 --out logits.json
"""

import argparse
import json
import sys

import numpy as np


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope(x: np.ndarray, pos: int, n_heads: int, theta: float) -> np.ndarray:
    """llama.cpp 约定：相邻对旋转（interleaved），freq = 1/theta^(i/dim), i=0,2,4,…"""
    dim = x.shape[-1]
    dhead = dim // n_heads
    out = x.reshape(n_heads, dhead).copy()
    for i in range(0, dhead, 2):
        freq = 1.0 / (theta ** (i / dhead))
        ang = pos * freq
        c, s = np.cos(ang), np.sin(ang)
        # 关键：a/b 必须取副本，否则下方 out[:,i]=… 会就地改写 a 的共享内存，
        # 导致下一行 out[:,i+1]=a*s+… 读到已被污染的 a（视图别名 bug）。
        a = out[:, i].copy()
        b = out[:, i + 1].copy()
        out[:, i] = a * c - b * s
        out[:, i + 1] = a * s + b * c
    return out.reshape(dim)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--ids", default="2,3,4,5")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta_path = a.meta or (a.npz.replace(".npz", "") + ".meta.txt")
    meta = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                meta[k] = v
    dim = int(meta["dim"])
    hidden = int(meta["hidden"])
    nl = int(meta["layers"])
    nh = int(meta["heads"])
    nkv = int(meta["kvheads"])
    V = int(meta["vocab"])
    dhead = int(meta["dhead"])
    eps = 1e-5
    theta = 10000.0
    group = nh // nkv

    W = dict(np.load(a.npz))
    ids = [int(t) for t in a.ids.split(",") if t != ""]
    T = len(ids)

    # KV cache
    Kc = np.zeros((nl, T, nkv * dhead), dtype=np.float64)
    Vc = np.zeros((nl, T, nkv * dhead), dtype=np.float64)

    # 逐 token 自回归（与 JS 的 KV cache 增量解码同构）
    x = None
    for pos, tok in enumerate(ids):
        x = W["token_embd.weight"][tok].astype(np.float64)
        for l in range(nl):
            p = f"blk.{l}."
            xn = rmsnorm(x, W[p + "attn_norm.weight"].astype(np.float64), eps)
            q = (W[p + "attn_q.weight"].astype(np.float64) @ xn)
            k = (W[p + "attn_k.weight"].astype(np.float64) @ xn)
            v = (W[p + "attn_v.weight"].astype(np.float64) @ xn)
            q = rope(q, pos, nh, theta)
            k = rope(k, pos, nkv, theta)
            Kc[l, pos] = k
            Vc[l, pos] = v

            att_out = np.zeros(dim, dtype=np.float64)
            for h in range(nh):
                kvh = h // group
                qh = q[h * dhead:(h + 1) * dhead]
                Kh = Kc[l, :pos + 1, kvh * dhead:(kvh + 1) * dhead]
                Vh = Vc[l, :pos + 1, kvh * dhead:(kvh + 1) * dhead]
                scores = (Kh @ qh) / np.sqrt(dhead)
                scores = scores - scores.max()
                e = np.exp(scores)
                p_attn = e / e.sum()
                att_out[h * dhead:(h + 1) * dhead] = p_attn @ Vh
            x = x + W[p + "attn_output.weight"].astype(np.float64) @ att_out

            xn2 = rmsnorm(x, W[p + "ffn_norm.weight"].astype(np.float64), eps)
            g = W[p + "ffn_gate.weight"].astype(np.float64) @ xn2
            u = W[p + "ffn_up.weight"].astype(np.float64) @ xn2
            x = x + W[p + "ffn_down.weight"].astype(np.float64) @ (silu(g) * u)
    xf = rmsnorm(x, W["output_norm.weight"].astype(np.float64), eps)
    logits = W["output.weight"].astype(np.float64) @ xf

    res = {"ids": ids, "vocab": V, "logits": logits.tolist()}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f)
        print(f"OK 写入 {a.out}")
    print("argmax =", int(np.argmax(logits)))
    print("前 8 个 logits:", np.array2string(logits[:8], precision=6))


if __name__ == "__main__":
    main()
