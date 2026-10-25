"""
cmp_full.py —— 逐阶段对拍「forge JS 前向」与「独立 numpy 参考」

对拍点（每个 token 位置 × 每层）：
  embed → afterAttn → afterFFN → xFinal → logits
任何一环（RMSNorm / RoPE / GQA 注意力 / KV 缓存 / SwiGLU / LM head）出错，
都会在该阶段的 max|Δ| 上立刻暴露。

用法（在仓库根目录）：
  python tools/interop/cmp_full.py
退出码：0 = PASS，1 = FAIL
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TOL = 1e-5
IDS = [2, 3, 4, 5]


def rms(x, w, eps=1e-5):
    return x / np.sqrt(np.mean(x * x) + eps) * w


def rope(x, pos, n, theta=10000.0):
    """llama.cpp 约定：相邻对旋转，freq = 1/theta^(i/dhead)，i=0,2,4,…

    注意：a/b 必须取副本，否则 out[:,i]=… 会就地改写 a 的共享内存，
    导致下一行 out[:,i+1]=a*s+… 读到已被污染的 a（numpy 视图别名 bug）。
    """
    d = x.shape[-1] // n
    o = x.reshape(n, d).copy()
    for i in range(0, d, 2):
        f = 1.0 / (theta ** (i / d))
        a = pos * f
        c, s = np.cos(a), np.sin(a)
        p = o[:, i].copy()
        q = o[:, i + 1].copy()
        o[:, i] = p * c - q * s
        o[:, i + 1] = p * s + q * c
    return o.reshape(-1)


def silu(x):
    return x / (1.0 + np.exp(-x))


def main():
    npz = os.path.join(OUT, "llama-official.gguf.npz")
    js_path = os.path.join(OUT, "js-full.json")
    meta_path = os.path.join(OUT, "llama-official.gguf.meta.txt")
    for p in (npz, js_path, meta_path):
        if not os.path.exists(p):
            sys.exit(f"缺少 {p}；请先运行 tools/interop/run_interop.mjs")

    W = dict(np.load(npz))
    js = json.load(open(js_path, encoding="utf-8"))
    meta = dict(
        l.strip().split("=", 1)
        for l in open(meta_path, encoding="utf-8")
        if "=" in l
    )
    dim = int(meta["dim"])
    nl = int(meta["layers"])
    nh = int(meta["heads"])
    nkv = int(meta["kvheads"])
    dhead = int(meta["dhead"])
    group = nh // nkv
    eps, theta = 1e-5, 10000.0

    KC = [[] for _ in range(nl)]
    VC = [[] for _ in range(nl)]
    results = []

    def diff(a, b, name):
        d = float(np.max(np.abs(np.array(a, dtype=np.float64) - b)))
        results.append((name, d))
        return d

    for pos, tok in enumerate(IDS):
        x = W["token_embd.weight"][tok].astype(np.float64)
        if pos == 0:
            diff(js["embed"], x, "embed")
        for l in range(nl):
            p = f"blk.{l}."
            xn = rms(x, W[p + "attn_norm.weight"].astype(np.float64))
            q = rope(W[p + "attn_q.weight"].astype(np.float64) @ xn, pos, nh)
            k = rope(W[p + "attn_k.weight"].astype(np.float64) @ xn, pos, nkv)
            v = W[p + "attn_v.weight"].astype(np.float64) @ xn
            KC[l].append(k)
            VC[l].append(v)
            att = np.zeros(dim)
            for h in range(nh):
                kvh = h // group
                qh = q[h * dhead:(h + 1) * dhead]
                Kh = np.array([KC[l][pp][kvh * dhead:(kvh + 1) * dhead] for pp in range(pos + 1)])
                Vh = np.array([VC[l][pp][kvh * dhead:(kvh + 1) * dhead] for pp in range(pos + 1)])
                s = (Kh @ qh) / np.sqrt(dhead)
                s = s - s.max()
                e = np.exp(s)
                pr = e / e.sum()
                att[h * dhead:(h + 1) * dhead] = pr @ Vh
            x = x + W[p + "attn_output.weight"].astype(np.float64) @ att
            diff(js[f"P{pos}_L{l}_afterAttn"], x, f"P{pos}_L{l}_afterAttn")
            xn2 = rms(x, W[p + "ffn_norm.weight"].astype(np.float64))
            g = W[p + "ffn_gate.weight"].astype(np.float64) @ xn2
            u = W[p + "ffn_up.weight"].astype(np.float64) @ xn2
            x = x + W[p + "ffn_down.weight"].astype(np.float64) @ (silu(g) * u)
            diff(js[f"P{pos}_L{l}_afterFFN"], x, f"P{pos}_L{l}_afterFFN")

    xf = rms(x, W["output_norm.weight"].astype(np.float64))
    diff(js["xFinal"], xf, "xFinal")
    logits = W["output.weight"].astype(np.float64) @ xf
    diff(js["logits"], logits, "LOGITS")

    print("── 逐阶段对拍 (forge JS vs numpy) ───────────────────────────")
    for name, d in results:
        flag = "✅" if d < TOL else "❌"
        print(f"  {name:22s} maxAbsDiff={d:.3e}  {flag}")
    worst = max(d for _, d in results)
    ok = worst < TOL
    print(f"  {'最大偏差':22s} {worst:.3e}  (容差 {TOL:.0e})")
    print(f"  结果: {'PASS ✅' if ok else 'FAIL ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
