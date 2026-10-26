#!/usr/bin/env python3
"""
量化互操作对拍 —— 用 llama.cpp 官方 gguf-py 作锚点，证伪 forge 的量化位布局。

为什么这样设计（判据的可辩护性）：
  · **布局断言**用全部数据组：把 forge 写出的字节交给官方 dequantize 读回，
    与 forge 自己的 dequantize 对拍。若位布局与 llama.cpp 不一致，官方反量化
    必然产出垃圾，两侧瞬间炸开 —— 这是本检查的牙齿所在。
  · **质量断言**只用 clean 组：块量化对离群点是固有弱势（一个 super-block 共享
    一个 fp16 的 d），把离群块算进相对 RMSE 会让指标既不可比也不可辩护，
    故 adversarial 组的 RMSE 仅作信息披露，不作判定。
  · **逐字节断言**用于官方提供写入端的类型（Q8_0 官方实现自述 bit-exact 对齐
    C 参考；F16 即 IEEE-754 half，numpy 是独立参照）。

用法: python tools/interop/verify_quant.py [--json out/quant-forge.json]
退出码: 0 = 全部 PASS，1 = 存在失败项
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def locate_ggufpy() -> str:
    env = os.environ.get("FORGE_GGUFPY")
    cands = [env] if env else []
    cands.append(os.path.join(ROOT, ".ref", "gguf-py"))
    for c in cands:
        if c and os.path.isdir(c):
            return c
    sys.exit("找不到 gguf-py：请把 llama.cpp 的 gguf-py 放到 .ref/gguf-py，或用 FORGE_GGUFPY 指定")


sys.path.insert(0, locate_ggufpy())
from gguf.constants import GGMLQuantizationType  # noqa: E402
from gguf import quants as Q  # noqa: E402

QT = {
    "F16": GGMLQuantizationType.F16,
    "Q8_0": GGMLQuantizationType.Q8_0,
    "Q4_K": GGMLQuantizationType.Q4_K,
    "Q5_K": GGMLQuantizationType.Q5_K,
    "Q6_K": GGMLQuantizationType.Q6_K,
}

# 布局一致性：两侧同一套数学，差只应来自 f32 运算顺序
TOL_DEQ = 1e-5
# 每元素有效码位（不含 scale 开销）；F16 非均匀量化，单独给界
BITS = {"F16": None, "Q8_0": 8, "Q6_K": 6, "Q5_K": 5, "Q4_K": 4}
TOL_F16 = 5e-4
# 允许比「同码率理想均匀量化器」差的比例（留 5% 余量吸收网格搜索/层级 scale 的开销）
SLACK = 1.05


def rel_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(HERE, "out", "quant-forge.json"))
    a = ap.parse_args()

    with open(a.json, encoding="utf-8") as f:
        d = json.load(f)
    n = d["n"]
    ds = d["datasets"]
    ok = True

    print(f"数据: n={n}  seed=0x{d['seed']:x}   two 组（clean / adversarial）")
    print()
    print("【布局断言】官方 dequantize(forge 字节) vs forge dequantize —— 两组都必须一致")
    print(f"{'类型':<6}{'bpw':>8}  {'clean maxΔ':>13}  {'adversarial maxΔ':>18}  判定")
    print("-" * 62)
    for name, qtype in QT.items():
        row = [name]
        verdict = "PASS"
        for kind in ("clean", "adversarial"):
            if name not in ds[kind]["cases"]:
                verdict = "FAIL(缺实现)"
                ok = False
                row.append("—")
                continue
            c = ds[kind]["cases"][name]
            raw = np.frombuffer(bytes.fromhex(c["bytes"]), dtype=np.uint8)
            assert raw.size == c["nBytes"], f"{name}/{kind} 字节数不一致"
            off_deq = Q.dequantize(raw.reshape(1, -1), qtype).reshape(-1).astype(np.float64)
            md = float(np.max(np.abs(off_deq - np.array(c["deq"], dtype=np.float64))))
            row.append(md)
            if md > TOL_DEQ or c["nonFinite"] != 0:
                verdict = "FAIL"
                ok = False
        bpw = ds["clean"]["cases"][name]["nBytes"] * 8 / n if name in ds["clean"]["cases"] else 0
        print(f"{row[0]:<6}{bpw:>8.4f}  {row[1]:>13.3e}  {row[2]:>18.3e}  {verdict}")

    print()
    print("【质量断言】clean 组相对 RMSE ≤ 同码率理想均匀量化器的理论值")
    print("  标定基准：Q8_0（与官方逐字节一致）；外推 rmse_Q8_0 × 255/(2^bits−1) × 1.05")
    print(f"{'类型':<6}{'bits':>5}{'clean relRMSE':>15}{'理论界':>10}  {'adversarial（仅披露）':>22}  判定")
    print("-" * 70)
    # 先取 Q8_0 的实测作为标定基准
    c8 = ds["clean"]["cases"].get("Q8_0")
    x8 = np.array(ds["clean"]["x"], dtype=np.float64)
    rmse_q8 = rel_rmse(
        Q.dequantize(np.frombuffer(bytes.fromhex(c8["bytes"]), dtype=np.uint8).reshape(1, -1),
                     GGMLQuantizationType.Q8_0).reshape(-1).astype(np.float64), x8)
    for name, qtype in QT.items():
        cc = ds["clean"]["cases"].get(name)
        ca = ds["adversarial"]["cases"].get(name)
        if not cc:
            print(f"{name:<6}{'—':>5}")
            ok = False
            continue
        rmse = rel_rmse(
            Q.dequantize(np.frombuffer(bytes.fromhex(cc["bytes"]), dtype=np.uint8).reshape(1, -1),
                         qtype).reshape(-1).astype(np.float64), x8)
        xa = np.array(ds["adversarial"]["x"], dtype=np.float64)
        rmsea = rel_rmse(
            Q.dequantize(np.frombuffer(bytes.fromhex(ca["bytes"]), dtype=np.uint8).reshape(1, -1),
                         qtype).reshape(-1).astype(np.float64), xa)
        bits = BITS[name]
        bound = TOL_F16 if bits is None else rmse_q8 * (255 / (2 ** bits - 1)) * SLACK
        verdict = "PASS" if rmse <= bound else "FAIL"
        if verdict == "FAIL":
            ok = False
        bs = "fp16" if bits is None else str(bits)
        print(f"{name:<6}{bs:>5}{rmse:>15.5f}{bound:>10.4f}  {rmsea:>22.5f}  {verdict}")

    print()
    print("【逐字节断言】官方写入端 → 与 forge 字节比对")
    x32 = np.array(ds["clean"]["x"], dtype=np.float32).reshape(1, n)
    if "Q8_0" in ds["clean"]["cases"]:
        off_bytes = Q.quantize(x32, GGMLQuantizationType.Q8_0).reshape(-1)
        forge_bytes = np.frombuffer(bytes.fromhex(ds["clean"]["cases"]["Q8_0"]["bytes"]), dtype=np.uint8)
        same = int(np.sum(off_bytes == forge_bytes))
        exact = same == off_bytes.size
        ok = ok and exact
        print(f"  Q8_0  {same}/{off_bytes.size} 字节一致  "
              f"sha(官方)={hashlib.sha256(off_bytes.tobytes()).hexdigest()[:16]}  "
              f"sha(forge)={hashlib.sha256(forge_bytes.tobytes()).hexdigest()[:16]}  "
              f"{'PASS' if exact else 'FAIL'}")
    if "F16" in ds["clean"]["cases"]:
        ref = x32.astype(np.float16).tobytes().hex()
        exact = ref == ds["clean"]["cases"]["F16"]["bytes"]
        ok = ok and exact
        print(f"  F16   vs numpy float16（IEEE-754 half）: {'PASS' if exact else 'FAIL'}")

    print()
    if ok:
        print("\033[32m\033[1m全部 PASS ✅\033[0m  forge 的量化位布局可被 llama.cpp 官方反量化器正确解读")
        return 0
    print("\033[31m\033[1m存在失败项 ❌\033[0m")
    return 1


if __name__ == "__main__":
    sys.exit(main())
