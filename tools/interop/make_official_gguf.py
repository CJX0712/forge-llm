"""
make_official_gguf.py —— 用 llama.cpp 官方 gguf-py 生成一个「官方命名」的 GGUF

用途：作为 forge-llm 的外部证伪输入。这个文件不是 forge-llm 写的，
而是 llama.cpp 官方工具链写出的，因此能证明 forge-llm 的 GGUF 读写器
真正兼容 llama.cpp 生态（而非只自圆其说）。

用法：
  python make_official_gguf.py <out.gguf> [--dim 64] [--layers 2] [--seed 7]

同时把权重 dump 成 .npz（供 Python 侧独立 numpy forward 做交叉验证）。
"""

import argparse
import os
import sys

import numpy as np

def _locate_ggufpy() -> str:
    """定位 llama.cpp 官方 gguf-py。优先环境变量，其次仓库内 .ref/gguf-py。"""
    env = os.environ.get("FORGE_GGUFPY")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.normpath(os.path.join(here, "..", "..", ".ref", "gguf-py"))
    return cand


GGUFPY = _locate_ggufpy()
if not os.path.isdir(GGUFPY):
    sys.exit(
        "找不到 llama.cpp 官方 gguf-py。请执行：\n"
        "  git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp\n"
        "  cp -r /tmp/llama.cpp/gguf-py <repo>/.ref/gguf-py\n"
        "或设置 FORGE_GGUFPY=/path/to/gguf-py"
    )
sys.path.insert(0, GGUFPY)
from gguf.gguf_writer import GGUFWriter  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402


def build_tokens(n_vocab: int):
    """构造一个可复现的小词表：单字节 + 少量多字符 token（含 <s>/</s>）"""
    toks = ["<s>", "</s>"]
    for i in range(n_vocab - 2):
        toks.append(f"t{i}")
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kvheads", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    # 输出目录不存在时自动创建（官方 writer 不会建目录）
    out_dir = os.path.dirname(os.path.abspath(a.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dim, hidden, nl, nh, nkv, V = a.dim, a.hidden, a.layers, a.heads, a.kvheads, a.vocab
    assert dim % nh == 0, "dim 必须能被 nHeads 整除"
    assert nh % nkv == 0, "nHeads 必须能被 nKvHeads 整除"
    dhead = dim // nh
    kvdim = nkv * dhead

    rng = np.random.default_rng(a.seed)
    # 权重按 ggml 约定：numpy shape = [N(out), K(in)]，官方 writer 直接写入该 shape
    g = lambda n_out, n_in: (rng.standard_normal((n_out, n_in), dtype=np.float32) * 0.08).astype(np.float32)

    W = {
        "token_embd.weight": g(V, dim),
        "output_norm.weight": (np.ones(dim, dtype=np.float32)),
        "output.weight": g(V, dim),
    }
    for l in range(nl):
        W[f"blk.{l}.attn_norm.weight"] = np.ones(dim, dtype=np.float32)
        W[f"blk.{l}.ffn_norm.weight"] = np.ones(dim, dtype=np.float32)
        W[f"blk.{l}.attn_q.weight"] = g(dim, dim)
        W[f"blk.{l}.attn_k.weight"] = g(kvdim, dim)
        W[f"blk.{l}.attn_v.weight"] = g(kvdim, dim)
        W[f"blk.{l}.attn_output.weight"] = g(dim, dim)
        W[f"blk.{l}.ffn_gate.weight"] = g(hidden, dim)
        W[f"blk.{l}.ffn_up.weight"] = g(hidden, dim)
        W[f"blk.{l}.ffn_down.weight"] = g(dim, hidden)

    toks = build_tokens(V)
    scores = [0.0] * V
    # 官方 TokenType：1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 5=UNUSED, 6=BYTE
    types = [3, 3] + [1] * (V - 2)

    w = GGUFWriter(path=a.out, arch="llama")
    w.add_name("llama.cpp-official-tiny")
    w.add_block_count(nl)
    w.add_embedding_length(dim)
    w.add_feed_forward_length(hidden)
    w.add_head_count(nh)
    w.add_head_count_kv(nkv)
    w.add_context_length(a.ctx)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_freq_base(10000.0)
    w.add_vocab_size(V)
    w.add_file_type(int(GGMLQuantizationType.F32))

    w.add_tokenizer_model("llama")
    w.add_tokenizer_pre("default")
    w.add_token_list(toks)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_bos_token_id(0)
    w.add_eos_token_id(1)
    w.add_add_bos_token(True)
    w.add_add_eos_token(False)

    for name, arr in W.items():
        w.add_tensor(name, arr)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    # dump 权重供 Python 侧独立 forward 交叉验证
    np.savez(a.out + ".npz", **{k: v for k, v in W.items()})
    meta = dict(dim=dim, hidden=hidden, layers=nl, heads=nh, kvheads=nkv, vocab=V,
                dhead=dhead, kvdim=kvdim, ctx=a.ctx, tokens=toks, seed=a.seed)
    with open(a.out + ".meta.txt", "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"{k}={v if not isinstance(v, list) else ' '.join(v)}\n")
    print(f"OK {a.out} ({os.path.getsize(a.out)} bytes)  tensors={len(W)}  dim={dim} layers={nl} vocab={V}")


if __name__ == "__main__":
    main()
