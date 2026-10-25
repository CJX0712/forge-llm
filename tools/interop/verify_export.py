"""
verify_export.py —— 反向互操作：用 **llama.cpp 官方 gguf-py** 读 forge 写出的 GGUF，
并与 forge 自己读取引擎的视图逐张量比对（dims / 字节 sha256 / 元数据）。

这一步证明的是「forge 的 GGUF **写**出来的文件，官方实现读得懂、读得一模一样」，
补上「官方写 → forge 读」之外的另一半闭环。

用法（在仓库根目录）：
  python tools/interop/verify_export.py
退出码：0 = PASS，1 = FAIL
"""

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def locate_ggufpy() -> str:
    env = os.environ.get("FORGE_GGUFPY")
    if env:
        return env
    return os.path.join(ROOT, ".ref", "gguf-py")


def main():
    gguf_path = os.path.join(OUT, "forge-export.gguf")
    view_path = os.path.join(OUT, "forge-view.json")
    for p in (gguf_path, view_path):
        if not os.path.exists(p):
            sys.exit(f"缺少 {p}；请先运行 tools/interop/run_interop.mjs")

    gguFPy = locate_ggufpy()
    if not os.path.isdir(gguFPy):
        sys.exit(f"找不到 gguf-py：{gguFPy}")
    sys.path.insert(0, gguFPy)
    from gguf import GGUFReader

    forge_view = json.load(open(view_path, encoding="utf-8"))
    reader = GGUFReader(gguf_path)

    print("── 反向互操作：forge 写 → llama.cpp 官方 gguf-py 读 ──────────")
    print(f"  文件: {os.path.basename(gguf_path)}  ({os.path.getsize(gguf_path)} 字节)")

    # 1) 张量集合一致
    off_names = [t.name for t in reader.tensors]
    fg_names = list(forge_view["tensors"].keys())
    ok_names = off_names == fg_names
    print(f"  张量数量            : official={len(off_names)}  forge={len(fg_names)}  "
          f"{'✅' if ok_names else '❌'}")
    if not ok_names:
        only_off = [n for n in off_names if n not in fg_names]
        only_fg = [n for n in fg_names if n not in off_names]
        if only_off:
            print(f"    仅官方读到: {only_off}")
        if only_fg:
            print(f"    仅 forge 读到: {only_fg}")
        sys.exit(1)

    # 2) 逐张量比对 dims / 类型 / 字节 sha256
    bad = []
    for t in reader.tensors:
        fv = forge_view["tensors"][t.name]
        # 注意：必须用 .tobytes() 取原始字节；
        # np.asarray(x, dtype=np.uint8) 是「值转换」而非「按字节重解释」，
        # 会把 float32 截断成 uint8，长度与内容都会错。
        raw = np.asarray(t.data).tobytes()
        off_sha = hashlib.sha256(raw).hexdigest()
        off_dims = [int(x) for x in t.shape]
        # gguf-py 的 shape 为文件内原始 dims 顺序（与 numpy 约定相反），此处与 forge 保持同一口径
        if off_dims != fv["dims"]:
            bad.append(f"{t.name}: dims official={off_dims} forge={fv['dims']}")
        elif off_sha != fv["sha256"]:
            bad.append(f"{t.name}: sha256 official={off_sha[:16]}… forge={fv['sha256'][:16]}…")
    if bad:
        for b in bad:
            print(f"    ❌ {b}")
        sys.exit(1)
    print(f"  逐张量 dims+sha256  : 全部 {len(off_names)} 个张量一致  ✅")

    # 3) 关键元数据一致（官方实现的读法 vs forge 的读法）
    def field(key):
        f = reader.get_field(key)
        if f is None:
            return None
        try:
            return f.contents()
        except Exception:
            return f.parts[-1] if f.parts else None

    checks = [
        ("general.architecture", "llama"),
        ("llama.block_count", None),
        ("llama.embedding_length", None),
        ("llama.attention.head_count", None),
        ("llama.vocab_size", None),
    ]
    ok_meta = True
    for key, expect in checks:
        off_v = field(key)
        fg_v = forge_view["fields"].get(key)
        same = off_v == fg_v and (expect is None or off_v == expect)
        ok_meta &= bool(same)
        shown = off_v if not isinstance(off_v, (bytes, bytearray)) else f"<{len(off_v)} bytes>"
        print(f"  {key:38s} official={shown!s:>12s}  forge={fg_v!s:>12s}  {'✅' if same else '❌'}")
    if not ok_meta:
        sys.exit(1)

    print("  结果               : PASS ✅  forge 写出的 GGUF 被 llama.cpp 官方实现完整、无损读出")
    sys.exit(0)


if __name__ == "__main__":
    main()
