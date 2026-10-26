# GGUF 外部证伪（interop falsification）

> forge-llm 的可信度不能靠自证。这一目录用**第三方实现**给 forge 挑错：
> 权重由 llama.cpp 官方工具链产出，前向由独立 numpy 实现复算，
> 文件由 llama.cpp 官方 reader 回读——三个外部锚点，任意一个都能证伪 forge。

---

## 一、为什么需要这一层

forge 自带的不变量（I-10 ~ I-91）都是**自洽性**检验：
「写完自己读回来一致」「增量解码 == 全量重算」。
它们能抓实现 bug，但抓不到**格式级**与**语义级**的偏差——
如果读写两端犯的是同一个错，自洽性检查会全绿，而文件对外部生态是不兼容的。

已实证的两类「自洽但错误」（都已被本目录的检查抓住）：

| # | 症状 | 自洽测试 | 外部检查 | 根因 |
|---|------|---------|---------|------|
| 1 | 之前 pos>0 前向发散（Δlogits 0.41） | ✅ 全绿 | ❌ 失败 | **参照实现**的 numpy RoPE 视图别名 bug（`o[:,i]=…` 就地改写了 `p` 所指共享内存，下一行 `o[:,i+1]=p*s+…` 读到被污染的 `p`）。forge 侧是对的 |
| 2 | GGUF 字符串长度前缀宽度错误 | ✅ 全绿 | ❌ 官方 reader 直接崩溃 | `encodeMetaValue` 写 u32 长度、`readMetaValue` 读 u64 长度 → 解析错位；**forge 导出的 GGUF 此前 llama.cpp 完全读不了** |
| 3 | Q6_K 的 super-block scale `d` 放错位置 | ✅ 全绿 | ❌ 官方反量化读出 **2.3e8** 量级垃圾 | Q4_K/Q5_K 的 `d` 在块头，Q6_K 却在**块尾**（`ql[128] qh[64] scales[16] d[2]`）。forge 读写两端一致地放在块头 ⇒ 往返全绿，而 llama.cpp 读 forge 的 Q6_K 会炸 |

第 2 条尤其说明问题：同一个「写→读」往返测试在写端与读端同时错误时**必然全绿**，
只有拿官方实现去读才能暴露。

---

## 二、证据链

```
        llama.cpp 官方 gguf-py                    forge-llm (JS)              独立 numpy 参考
              │                                        │                           │
   ① 写 GGUF ─┼──────────► [llama-official.gguf] ───► loadGGUF ──┐               │
              │                                        │        │                 │
              │  ② 读同一文件 ────────────────────────────────┘        │                 │
              │                                        │        ▼                 ▼
              │                              ③ 逐阶段对拍 (embed/afterAttn/         │
              │                                 afterFFN/xFinal/logits) ◄── numpy 前向
              │                                        │                          │
              └────────────────────────────────────────┴──── ④ logits 终判 ────────┘
                                                            ≤1e-5 ✅
```

- **①** 权重来源：llama.cpp 官方 `gguf-py` 的 `GGUFWriter`，**未经 forge 之手**
- **②** 反向闭环：forge **写**出的 GGUF，交由官方 `GGUFReader` 读回，逐张量比对
  dims + 字节 sha256 + 关键元数据
- **③** 逐阶段：4 个位置 × 2 层 × 4 个对拍点，逐点 max|Δ|
- **④** 终判：`Transformer.forward`（出厂代码路径）vs numpy 参考的 logits

---

## 三、实测结果

| 对拍点 | max&#124;Δ&#124; | 判定 |
|--------|-------------|------|
| embed | 0.000e+00 | ✅ |
| P0~P3 · L0/L1 · afterAttn | ≤ 1.292e-07 | ✅ |
| P0~P3 · L0/L1 · afterFFN | ≤ 1.728e-07 | ✅ |
| xFinal | 4.902e-07 | ✅ |
| logits（出厂前向 vs numpy） | **3.149e-07** | ✅ |
| argmax | 99 vs 99 | ✅ |
| 反向：20 个张量 dims + sha256 | 全部一致 | ✅ |

容差 1e-5（float32 精度上限的 ~10 倍余量）。实测偏差全部落在 float32 舍入噪声内
（numpy 侧用 float64，forge 侧用 float32，故残差非零属预期）。

> 模型规格：dim=64, hidden=128, layers=2, heads=4, kv_heads=2（GQA group=2）, vocab=256,
> ctx=256, rope_base=10000 —— 21 个张量，434 KB。

---

## 四、怎么跑

前置：**Python 3 + numpy**，以及 llama.cpp 官方 `gguf-py`（不需要 pip 安装，
脚本会把 `.ref/gguf-py` 直接挂到 `sys.path`）。

```bash
# 1) 取 llama.cpp 官方 gguf-py（只需一次）
git clone --depth 1 https://github.com/ggml-org/llama.cpp .ref/llama.cpp
cp -r .ref/llama.cpp/gguf-py .ref/gguf-py

# 2) 一条命令跑完全流程
node tools/interop/run_interop.mjs --force
```

Windows / 指定解释器：

```bat
set FORGE_PYTHON=C:\path\to\python.exe
node tools/interop/run_interop.mjs --force
```

退出码 0 = 全部 PASS，1 = 有环节失败（CI 可直接用）。

---

## 五、文件清单

| 文件 | 角色 |
|------|------|
| `run_interop.mjs` | 一键编排（8 步），跨平台，输出 PASS/FAIL 总结 |
| `make_official_gguf.py` | 用官方 `gguf-py` 写出官方命名的 GGUF + 权重 `.npz` |
| `numpy_forward.py` | **独立** numpy 前向实现（对照组，不共享任何 forge 代码） |
| `dump_stages.mjs` | 手工展开每层每阶段的中间量，供逐点定位误差 |
| `verify_engine.mjs` | 走**出厂代码路径**（`Transformer.forward` + `KVCache` + `gatherKV`） |
| `cmp_full.py` | 逐阶段对拍（embed / afterAttn / afterFFN / xFinal / logits） |
| `cmp_logits.py` | 出厂前向 vs numpy 的 logits 终判（阈值 1e-5） |
| `verify_export.mjs` | forge 读取引擎对自家文件的「视图」（dims + sha256） |
| `verify_export.py` | 官方 `GGUFReader` 对同一文件的视图，并逐张量比对 |
| `verify_quant.mjs` | forge 侧：确定性数据（clean + adversarial）量化后转储字节与反量化值 |
| `verify_quant.py` | 官方 `gguf.quants.dequantize` 读同一字节并逐点对拍 + 质量判定 |

---

## 六、量化位布局对拍（第 8 步）

| 断言 | 判据 | 实测 |
|---|---|---|
| 布局 | 官方 `dequantize` 读 forge 字节 == forge `dequantize`，clean + adversarial 两组 | 5/5 格式 maxΔ = **0** |
| 逐字节 | Q8_0 与官方 `gguf.quants.quantize`（自述 bit-exact 对齐 C 参考） | 4352/4352 一致 |
| 逐字节 | F16 vs numpy `float16`（IEEE-754 half 是硬件规范） | 一致 |
| 质量 | clean 组相对 RMSE ≤ 同码率**理想均匀量化器**的理论值 | 5/5 PASS |

质量界不是拍脑袋定的：以与官方逐字节一致的 Q8_0 为标定基准，按码率外推
`rmse_Q8_0 × 255/(2^bits − 1) × 1.05`。实测各格式都略优于该理论界
（Q4_K 0.077 vs 界 0.093、Q5_K 0.036 vs 0.045、Q6_K 0.017 vs 0.022）。

离群组（全零块 / 常数块 / 单一 50× 离群块）的 RMSE **仅披露不判定**：
一个 super-block 共享一个 fp16 的 `d`，离群点会拉爆整块量程，这是块量化的固有
代价，把它算进指标既不可比也不可辩护。布局断言则两组都跑。

## 七、对照实现的教训：numpy 视图别名 bug

参照实现一度给出「forge 在 pos>0 发散」的结论。定位后发现**发散方是参照实现**：

```python
# 有 bug 的写法
p, q = o[:, i], o[:, i + 1]      # p 是 o 的视图，不是副本
o[:, i]     = p * c - q * s       # ← 就地改写 o[:,i]，p 随之被污染
o[:, i + 1] = p * s + q * c       # ← 此处 p 已是新值，算错

# 正确写法
p, q = o[:, i].copy(), o[:, i + 1].copy()
```

手工验算：pair=(0.5338, 0.83552)，pos=1，θ=1.0 →
正确旋转 new1 = 0.5338·sin1 + 0.83552·cos1 = **0.90074**；
错误写法得 (0.5338·cos1 − 0.83552·sin1)·sin1 + 0.83552·cos1 = **0.10244** —— 与实测错值逐位吻合。

**forge 侧不需要改一行代码。** 这条留档，是因为「对照实现的 bug」和「被测对象的 bug」
在结果上完全一样——排查时必须对双方同等怀疑。

### 另一个教训：自洽往返测试的结构性盲区（Q6_K 的 d）

「写→读往返一致」只能证明两端**互相**一致，不能证明任一端**正确**。
Q6_K 的 `d` 位置错误就是活例：写端写到块头、读端从块头读，往返必然全绿；
而 llama.cpp 从块尾读 `d`，读出来的是 `ql` 的头两个字节当 fp16 解 —— 幅值 1e8。

抓它靠的是**外部反量化器**，而判据设计成不依赖官方写入端也能生效：
把 forge 写出的字节交给 `gguf.quants.dequantize`，与 forge 自己的 `dequantize` 对拍。
布局一旦错位，两侧立刻分离（实测 5.3e7），而不是靠肉眼比对字节。

顺带一个可复用的**纯 JS 判据**（`test/l3-l5.test.js`）：K-quant 的 `ql/qh/scales`
对正标量缩放**不变**，只有 `d` 随缩放线性变化。因此把输入整体 ×4 后重新量化，
「块内前 208 字节应完全相同、末 2 字节应不同」—— 这个断言不依赖任何外部工具。

---

## 八、相关不变量

- `I-90` GGUF 导出 → 重加载，logits 逐位一致（`src/L10/verify.js`）
- `I-91` GGUF 元数据遵循官方命名（`general.architecture=llama` + `llama.*` 键）
- `I-92`（本目录）导出文件可被官方 `GGUFReader` 无损读出——**由外部检查承担**
