"""
cmp_logits.py —— 对拍「出厂前向」(verify_engine.mjs) 与「独立 numpy 参考」(numpy_forward.py)

判定标准：max|Δlogits| < 1e-5（float32 精度上限的 ~10 倍余量）。
若 forge 的 GGUF 解析或任一 transformer 数学环节有误，该值会迅速劣化到 1e-2 量级。

退出码：0 = PASS，1 = FAIL
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TOL = 1e-5


def main():
    eng_path = os.path.join(OUT, "engine-logits.json")
    np_path = os.path.join(OUT, "np-logits.json")
    for p in (eng_path, np_path):
        if not os.path.exists(p):
            sys.exit(f"缺少 {p}；请先运行 run_interop.mjs（或先跑 verify_engine.mjs / numpy_forward.py）")

    eng = np.array(json.load(open(eng_path, encoding="utf-8"))["logits"], dtype=np.float64)
    npl = np.array(json.load(open(np_path, encoding="utf-8"))["logits"], dtype=np.float64)

    if eng.shape != npl.shape:
        sys.exit(f"形状不一致：engine {eng.shape} vs numpy {npl.shape}")

    diff = np.abs(eng - npl)
    md = float(diff.max())
    idx = int(diff.argmax())
    a_e, a_n = int(np.argmax(eng)), int(np.argmax(npl))
    ok = md < TOL and a_e == a_n

    print("── 出厂前向 vs 独立 numpy 参考 ─────────────────────────────")
    print(f"  logits 维度        : {eng.shape[0]}")
    print(f"  max|Δlogits|       : {md:.3e}   (容差 {TOL:.0e})")
    print(f"  最大偏差位置       : idx={idx}  engine={eng[idx]:.6f}  numpy={npl[idx]:.6f}")
    print(f"  argmax             : engine={a_e}  numpy={a_n}")
    print(f"  engine logits[:8]  : {np.array2string(eng[:8], precision=6)}")
    print(f"  numpy  logits[:8]  : {np.array2string(npl[:8], precision=6)}")
    print(f"  结果               : {'PASS ✅' if ok else 'FAIL ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
