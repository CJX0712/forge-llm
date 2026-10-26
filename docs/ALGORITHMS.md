# 算法原理 · Algorithms

> 每个模块的**数学定义**、**站在巨人肩上的依据**、**实现要点**、**可验证不变量**。
> 遵循零重写原则：先检索世界最优现成成果，直接复用其数学定义与位布局。

---

## 0. 复用清单（零重写原则的落地）

| 本项目模块 | 直接复用 | 复用内容 | 自研部分与理由 |
|---|---|---|---|
| L0 GEMM | BLAS / ggml | 4 路展开 + 行主序顺序读（缓存分块实测否决） | 无 |
| L0 RNG | Vigna 2017 | xorshift128+ | 无 |
| L1 softmax | Goodfellow 2016 | 减最大值稳定化 | 无 |
| L1 RMSNorm | Zhang & Sennrich 2019 | `y = x/rms(x)·w` | 无 |
| L1 RoPE | Su et al. 2023 | 成对旋转公式 | 无 |
| L1 SwiGLU | Shazeer 2020 | `silu(gate)⊙up` | 无 |
| L1 erf | Abramowitz & Stegun 7.1.26 | 7.1.26 有理逼近 | 无 |
| L2 AD | Linnainmaa 1970 / Rumelhart 1986 | tape 结构 | 张量级融合算子的反向公式（照 PyTorch `derivatives.yaml`） |
| L2 Adam | Kingma & Ba 2015 | bias correction 式 (7)(8) | 无 |
| L2 裁剪 | Pascanu et al. 2013 | 全局范数缩放 | 无 |
| L2 余弦退火 | Loshchilov & Hutter 2017 | SGDR | 无 |
| L3 量化位布局 | llama.cpp `ggml-quants.c` | K-quant 全部格式 | Q4_K **写入端**（闭式替代 importance-matrix，见 §3.3） |
| L3 Q6_K 写入 | llama.cpp `quantize_row_q6_K_impl` | 含 RMSE 迭代搜索 | 无（完整移植） |
| L4 QuantLinear | llama.cpp `ggml_mul_mat` | 量化权重直接参与 matvec | 无 |
| L5 在线 softmax | FlashAttention (Dao et al. 2022) | (m, ℓ, acc) 三元组 | 无 |
| L5 分页 KV | vLLM PagedAttention (Kwon et al. 2023) | 按块分配 | 无 |
| L5 GQA | Ainslie et al. 2023 | kv-group 复用 | 无 |
| L6 架构 | Llama-2 (Touvron et al. 2023) | RMSNorm/GQA/RoPE/SwiGLU | 无 |
| L6 初始化 | GPT-2 / llama2.c | N(0,0.02) + 1/√(2L) 残差缩放 | 无 |
| L6 批量掩码 | HuggingFace 4D attn_mask | 块对角因果 | 无 |
| L7 字节回退 | Kudo & Richardson 2018 | SentencePiece byte-fallback | 无 |
| L8 min-p | Nguyen et al. 2024 (arXiv:2407.01082) | 阈值 = min_p·p_max | 无 |
| L9 GGUF | ggml 官方规范 v3 | 二进制布局 | 无 |

**自研部分总计：2 处**（Q4_K 写入端、张量级 AD 融合算子的反向），均有明确理由。

---

## 1. L0 · 数值内核

### 1.1 GEMM：4 路展开（缓存分块经实测否决）

**目标**：`C[M,N] = A[M,K] · B[N,K]ᵀ`（B 行主序 `[N,K]` —— 与权重布局一致）

**做法**：
```
for m in M:
  for n in N:
    4 路独立累加 s0..s3（k 步长 4）→ s = (s0+s1)+(s2+s3)
    C[m,n] = s
```
**4 路展开**的收益：隐藏 FMA 的依赖链（现代 x86 的 FMA 延迟 4 周期、吞吐 0.5 周期），4 路独立累加器可让流水线保持满载。

**为什么不做缓存分块**（选型由实测决定）：512×1024×1024 交错基准（best-of-60）——

| 实现 | 耗时 | GFLOPS |
|---|---|---|
| 朴素三重循环 + 4 路展开（采用） | **433.9 ms** | **2.47** |
| BK=32, BN=32 分块 | 512.1 ms | 2.10 |
| BK=64, BN=64 分块 | 508.2 ms | 2.11 |
| BK=128, BN=32 分块 | 531.2 ms | 2.02 |
| 朴素三重循环（无展开） | 567.3 ms | 1.89 |

cache blocking 在 C/BLAS 下是标准手段，但在 V8 生成的 JS 代码上，访存模式与
寄存器分配由 JIT 决定，分块反而增加循环开销与 C 矩阵的多次读写。**结论：JS 下
最快的是最简形式**。若未来接入 WASM/SIMD，L0 是单一替换点。

**实测**（B1，交错计时）：128×256×256 展开 vs 朴素 = 1.35×；512×1024×1024 = 1.26×。
与 `gemmRef` 输出**比特级一致**（relL2 = 0）—— 比预定的 1e-5 更强的不变量。

### 1.2 matvec（解码热点）

```
out[n] = Σ_k W[n*K+k] · x[k]     # W 行主序 → 顺序访问
```
每 token 每层要走 2×N×K 次 FMA。这是纯 JS 推理的主要成本。

**实测**（B2）：4096×4096 → 16.23 ms → **2.07 GFLOPS**。

### 1.3 xorshift128+ RNG

```js
x ^= x << 23; x ^= x >>> 17; x ^= y; x ^= y >>> 26;
```
**为什么**：benchmark 必须可复现。`Math.random()` 不可控；xorshift128+ 的周期是 2^128−1，通过 BigCrush 测试，且实现仅 4 行。种子用 splitmix32 扩散，避免低位种子相关性。

---

## 2. L1 · 特殊函数

### 2.1 稳定 softmax

```
m = max(x);  p_i = exp(x_i − m) / Σ_j exp(x_j − m)
```
减去最大值不改变结果（分子分母同乘 e^{-m}），但把最大指数压到 1，避免 `exp(1000) = Inf`。

**不变量**：I-10（Σ=1 且非负）、I-11（logsumexp 恒等 `log Σe^x = m + log Σe^{x−m}`）。

### 2.2 RMSNorm

```
y_i = x_i · w_i / sqrt( (1/n)Σx_i² + ε )
```
**为什么不用 LayerNorm**：RMSNorm 省去均值中心化，计算更少、数值更稳（Zhang & Sennrich 2019），是 LLaMA/Qwen/Mistral 的默认选择。

**反向解析式**（论文附录）：
```
r = sqrt(ss/n + ε),  ĉ = (Σ g_i w_i x_i) / (n r³)
dx_i = g_i w_i / r − ĉ x_i
dw_i = g_i x_i / r
```

**不变量**：I-12（w=1 时输出单位 RMS，±1e-6）。

### 2.3 RoPE（旋转位置编码）

**复数形式**（Su et al. 2023）：
```
q_m = R_m q,  k_n = R_n k
q_mᵀk_n = qᵀ R_mᵀ R_n k = qᵀ R_{n−m} k     ← 只依赖相对位置
```
**实数实现**（成对旋转）：
```
θ_i = 1/base^(2i/d)
(x_{2i}, x_{2i+1}) ← (x_{2i}·cos(mθ) − x_{2i+1}·sin(mθ),
                      x_{2i}·sin(mθ) + x_{2i+1}·cos(mθ))
```

**为什么这个性质重要**：模型天然具备相对位置归纳偏置，且**外推能力优于绝对位置编码**（训练长度外仍可用）。

**反向**：旋转矩阵是正交的，`Rᵀ = R⁻¹`，故 `dx = Rᵀ g`（转置回传）。

**不变量**：I-13（`(R_m q)·(R_n k)` 只依赖 m−n）。

### 2.4 SwiGLU

```
FFN(x) = W_down · (silu(W_gate·x) ⊙ (W_up·x))
silu(z) = z·σ(z)
```
**为什么要门控**：比标准 FFN 多一个投影分支，参数量增加但表达力提升明显（Shazeer 2020）。LLaMA/Qwen/Mistral 全部采用。

---

## 3. L3 · 量化内核

### 3.1 量化的一般框架

给定权重块 `x`（长度 n），求 `x̂ ≈ x`：
1. 统计块的极值/分布
2. 求量化步长 `d`（scale）与零点 `m`（min）
3. 逐元素量化到整数 `q = round((x−m)/d)`
4. 存储 `d, m, q`

**误差来源**：`x̂ = d·q + m`，误差 `|x̂−x| ≤ d/2`。

### 3.2 Q8_0（块长 32）

```
d = amax / 127
q_i = clamp(round(x_i / d), −128, 127)
```
**误差界**：`|x̂−x| ≤ d/2 = amax/254`。

**布局**：`f16(d)` [2B] + `int8[32]` [32B] = 34 B / 32 权重 = **8.5 bpw**。

**实测 RMSE**：3.73e-3，余弦 0.999986。

### 3.3 Q4_K（块长 256，超级块）

**两级 scale 结构**（ikawrakow 的设计）：
```
super-block (256 权重):
  f16 d_super, f16 d_min_super      # 4 B
  uint8 scales[12]                  # 12 B  ← 8 个子块的 6-bit scale + 6-bit min 打包
  uint8 qs[128]                     # 128 B ← 4-bit 量化索引
                                  ─────
                                  144 B / 256 权重 = 4.5 bpw
```

**scales[12] 的打包技巧**（llama.cpp 的 `get_scale_min_k4`）：
- 子块 0–3：`scale_j = scales[j] & 63`，`min_j = scales[j+4] & 63`
- 子块 4–7：`scale_j = (scales[j+4] & 0xF) | ((scales[j−4] >> 6) << 4)`（高位藏在前一个字节的 bits 6–7）

**解码**：每个子块（32 元素）有自己的 `d_sub = d_super · scale_sub` 与 `m_sub = d_min_super · min_sub`；`x̂ = d_sub·q − m_sub`。

**为什么是两级**：4-bit 量化本身太粗（±1/15 相对误差），两级 scale 让不同子块用不同步长，能贴合权重的局部幅度分布。

**写入端（本项目自研部分）**：
- llama.cpp 用 **importance matrix** 迭代搜索（`make_qkx3_quants` / `make_qp_quants`），需要校准集（activation-aware）。
- 零依赖内核里做不了校准，故用**闭式 min-max + 6-bit 层级 scale**：每个 32 元素子块取 min/max → 15 级均匀量化；8 个子块的 scale/min 再 6-bit 量化到超级块。
- **格式完全兼容**（llama.cpp 能读），**代价是 RMSE 略高**：实测 5.45e-2（llama.cpp 官方 Q4_K_M 约 4.5e-2）。
- **需要极致质量时请用 `llama-quantize` 产出** —— 本项目的定位是「读+推理+验证」，不是取代官方量化器。

### 3.4 Q6_K（块长 256）

**布局**：`f16(d)` [2B] + `ql[128]` [128B] + `qh[64]` [64B] + `scales[16]` [16B] = 210 B / 256 = **6.5625 bpw**。

**6-bit 索引的打包**：低 4 位放 `ql`，高 2 位放 `qh`（每 4 个元素共享 1 字节 qh）。

**写入端完整移植** llama.cpp 的 `quantize_row_q6_K_impl`（含 `make_qx_quants` 的 RMSE 迭代搜索），因此精度与官方一致：**RMSE 1.21e-2，余弦 0.999848**。

### 3.5 dotQ：不解量化的融合点积

```js
dotQ(type, u8, x, n, xOff, byteOff)
```
把「反量化 → 存 f32 数组 → 点积」三步压成一步：在解码每个量化值时**立即**乘进累加器。

**收益**：
- 内存：省掉一个 n 大小的 f32 临时数组（4096 权重 = 16 KB）
- 带宽：只读一次量化字节流（Q4_K 是 4.5 bpw，f32 是 32 bpw → **7.1× 更少的内存流量**）

**不变量**：I-22（dotQ 与「反量化后点积」误差 ≤ 1e-4，实测 Δ = 0.00e+0 / 1.78e-15）。

---

## 4. L5 · 注意力

### 4.1 在线 softmax（FlashAttention 的标量版）

**朴素做法**：物化 N×N 分数矩阵 → softmax → 乘 V。内存 O(N²)。

**在线做法**：维护三元组 `(m, ℓ, acc)`：
```
每来一个块：
  m_new = max(m, m_block)
  α = exp(m − m_new)          # 旧结果的缩放因子
  ℓ = ℓ·α + Σ p
  acc = acc·α + Σ p·V
最后：out = acc / ℓ
```
**为什么正确**：`softmax` 的分母与最大值都可增量更新 —— 减去新最大值后，旧指数全部乘以 `exp(m_old − m_new)`，恰好由 `α` 补偿。

**收益**：内存 O(d) 而非 O(N²)，且分块扫描天然适配 KV cache。

**实测**（B4）：seq=512, d=64 → flash 0.1815 ms vs 朴素 0.3042 ms → **1.68×**，数值一致 maxErr = 1.49e-7。

### 4.2 因果掩码

位置 `i` 的 query 只能看 `j ≤ i`。实现上通过 `end = min(n, qOff+1)` 限定扫描范围 —— 在 KV cache 场景下，`qOff = pos`，天然只扫已缓存的前缀。

### 4.3 GQA（分组查询注意力）

```
n_kv < n_q 时，query head h 使用 kv-group ⌊h / ⌊n_q/n_kv⌋⌋ 的 K/V
```
**收益**：KV cache 显存降为 1/n_kv 倍（Llama-2 70B 是 32:8 = 4×）。

### 4.4 分页 KV（vLLM PagedAttention 思想）

按固定块（默认 64）分配 KV 空间，支持变长序列与批处理时零拷贝复用，避免「按最长序列预留」造成的显存碎片。

---

## 5. L6 · 模型

### 5.1 Llama-2 完整拓扑

```
x = E[token]
for l in layers:
    h1 = RMSNorm(x, ln1_l)
    q, k, v = Wqkv_l · h1        # Wqkv 输出 (n_h + 2·n_kv)·d_head
    q, k = RoPE(q, pos), RoPE(k, pos)
    a = Attn(q, K_cache, V_cache)  # 因果 + GQA
    x = x + Wo_l · a
    h2 = RMSNorm(x, ln2_l)
    x = x + Wdown_l · (silu(Wgate_l·h2) ⊙ (Wup_l·h2))
logits = Wcls · RMSNorm(x, ln_f)
```

### 5.2 批量训练中的块对角因果掩码

**问题**：若把 B 个长度 T 的样本拼成一条 BT 长序列做因果注意力，样本之间会互相注意 —— 这是**错的**。

**解法**：`causalAttnBatch` 用块对角掩码，样本 b 的第 t 个 token 只看同一样本的 0..t。等价于 HF 的 4D attn_mask。

**RoPE 位置**也必须在样本内重新计数（`ropeRows` 的 `posStride = T`）。

**验证**：L10 断言「批量 loss ≡ 逐样本 loss 的均值」（实测完全相等，误差 < 1e-5）。

### 5.3 权重初始化

```
W ~ N(0, 0.02)                          # 普通矩阵
W_residual ~ N(0, 0.02) · 1/√(2·nLayers) # 残差投影额外缩放
```
**为什么残差投影要缩放**：随着层数增加，残差流的方差会累积放大（每层加一个 var(W·h)）。1/√(2L) 让方差保持稳定，是 GPT-2 / llama2.c 的标准做法。

---

## 6. L2 · 自动微分

### 6.1 反向模式 tape

```js
op(out, bwd)   // 创建节点 + 挂反向函数 + 压 tape
backward(loss) // loss.grad=1；逆序遍历 tape 调 nd._b(nd.grad)
```

**为什么是反向模式**：LLM 训练里「输出维度 ≪ 参数维度」，反向模式只需 O(1) 次前向就能拿到所有参数的梯度。前向模式需要 O(参数量) 次。

### 6.2 张量级融合算子的反向

**照 PyTorch `derivatives.yaml` 的语义**，逐个推导：

| 算子 | 前向 | 反向 |
|---|---|---|
| `matmulT(X[M,K], W[N,K])` | `Y[m,n] = Σ_k X[m,k]·W[n,k]` | `dX = dY·W`，`dW = dYᵀ·X` |
| `embedLookup(E[V,D], ids[T])` | `X[t] = E[ids[t]]` | `dE[ids[t]] += dX[t]`（scatter-add） |
| `rmsNorm` | `y_i = x_i w_i / r` | `dx_i = g_i w_i/r − ĉ x_i`，`dw_i = g_i x_i/r` |
| `ropeRows` | `y = R x` | `dx = Rᵀ g`（旋转正交，转置回传） |
| `causalAttn` | `S = QKᵀ/√d`，`P = softmax(S)`，`O = PV` | `dV = PᵀdO`；`dP = dO·Vᵀ`；`dS = P⊙(dP − Σ_j P_j dP_j)`；`dQ = dS·K/√d`；`dK = dSᵀ·Q/√d` |
| `swigluRows` | `o = silu(g)⊙u` | `dg = du·(σ(g) + g·σ(g)(1−σ(g)))`，`du = do·silu(g)` |
| `crossEntropyMean` | `L = −(1/T)Σ log P[y]` | `dLogits = (P − OneHot)/T` |

**关键点**：这些反向都是**整块张量**的一次遍历，不是逐元素 —— 与 PyTorch 的融合算子思路一致。

### 6.3 Adam

```
m_t = β₁ m_{t−1} + (1−β₁) g_t
v_t = β₂ v_{t−1} + (1−β₂) g_t²
m̂ = m_t/(1−β₁^t),  v̂ = v_t/(1−β₂^t)
θ ← θ − lr·m̂/(√v̂ + ε)
```
**bias correction 的作用**：m、v 初始为 0，前几步估计有偏；除以 `1−β^t` 修正。**可验证**：单参数首步更新幅度恰为 `lr`（I-32 实测 Δw = 0.1000，lr = 0.1）。

### 6.4 梯度检验的正确做法

中心差分：`∂f/∂x_i ≈ (f(x+εe_i) − f(x−εe_i)) / (2ε)`，理论误差 O(ε²)。

**但**：f32 下 loss 的绝对精度约 1e-7，当某元素梯度本身很小时，逐元素相对误差会被放大。

**本项目的做法**（同 PyTorch gradcheck 的 atol+rtol 双阈值思路）：
```
误差 = |num − ana| / max(‖ana‖∞, 1e-3)
```
即「相对梯度尺度」的误差。实测：matmulT 8.56e-5、rmsNorm 1.96e-5、ropeRows 3.04e-5、causalAttn 9.88e-4、swiglu 2.63e-5。

---

## 7. L8 · 采样

### 7.1 min-p（arXiv:2407.01082）

```
阈值 τ = min_p · p_max
保留所有 p_i ≥ τ 的 token，重新归一化
```
**为什么比 top-p 好**：top-p 的阈值依赖**累积概率**，会随分布形状剧烈变化；min-p 相对**最大概率**，在分布尖锐（低熵）时自然收紧，分布平坦时自然放宽 —— 阈值自适应。

**实测**（I-82）：5 个 token 保留 4 个，截断生效。

### 7.2 采样顺序

```
1. 频率/存在惩罚 → 改写 logits
2. 温度缩放       → logits / T
3. T→0 或 k=1    → 直接 argmax（greedy）
4. softmax（稳定）
5. min-p 截断 + 重归一化
6. top-k 掩码
7. top-p 核采样（按降序累加至 ≥ top_p）
8. 在允许集合上按概率采样
```
**为什么 greedy 在 softmax 之前**：T→0 时 logits/T 会溢出，且 argmax 与温度无关。

---

## 8. L9 · GGUF

### 8.1 二进制布局（v3）

```
magic      u32  "GGUF" (0x46554747)
version    u32  3
n_tensors  u32
n_kv       u32
─── KV 元数据 ───
for each: key(string) + value_type(u32) + value
─── 张量信息 ───
for each: name(string) + n_dims(u32) + dims(u64[]) + type(u32) + offset(u64)
─── 对齐填充 ───
pad to general.alignment (默认 32)
─── 张量数据 ───
每个张量按 alignment 补齐
```

**元数据类型**：UINT8/INT8/UINT16/INT16/UINT32/INT32/FLOAT32/BOOL/STRING/ARRAY/UINT64/INT64/FLOAT64。

**STRING 的编码**：`u32 长度 + UTF-8 字节`（数组元素同样带长度前缀）。

**ARRAY 的编码**：`elem_type(u32) + count(u64) + 逐个元素`。

### 8.2 对齐规则

- 数据段起始地址 = `ceil(headerLen / alignment) × alignment`
- 每个张量的 offset 相对数据段起点，且张量之间按 alignment 补齐
- **量化张量的元素数必须是块长的整数倍**（Q4_K/Q6_K = 256，Q8_0 = 32）—— 这是 GGUF 规范要求，本项目的 `tensorBytes` 会显式校验并抛错

### 8.3 量化张量的字节数

```
F32/F16:  n × 4 / n × 2
块量化:   (n / blockSize) × blockBytes
```
**为什么不能用「平均字节/元素」**：Q4_K 是 144/256，但 n 必须是 256 的倍数才能整除。本项目用 `tensorBytes` 显式计算并校验。

---

## 9. 训练与推理的数值一致性

**这是本项目最核心的技术主张。**

**等式链**（每一环都有不变量守护）：

| 环节 | 训练侧 | 推理侧 | 等价依据 |
|---|---|---|---|
| 嵌入 | `embedLookup` | `_embed` | 都是「按行拷贝」 |
| 归一化 | `rmsNorm` | `L1.rmsnorm` | **同一份实现** |
| 线性 | `matmulT` | `QuantLinear.forward` | 都是 `Σ_k X[k]·W[n,k]` |
| 位置 | `ropeRows` | `applyRopeHead` | 同一 RoPE 公式 |
| 注意力 | `causalAttn` | `flashAttention` | 在线 softmax 与显式 softmax 数学等价 |
| FFN | `swigluRows` | `SwiGLU.forward` | 都是 `silu(g)⊙u` |
| 输出头 | `matmulT(xf, Wcls)` | `wcls.forward` | 同上 |

**实测**：I-70 `maxLogitsDiff = 5.96e-8`（f32 舍入级别）。

**工程价值**：训练出的权重**零转换**直接被推理引擎加载；不会出现「训练图 ≠ 推理图」导致的静默精度退化。

---

## 10. 不变量全表（40 条）

| ID | 名称 | 判据 | 实测 |
|---|---|---|---|
| I-00 | gemm ≡ 朴素三重循环 | relL2 < 1e-5 | 0.00e+0 |
| I-01 | matvec ≡ gemm(M=1) | maxErr = 0 | 0 |
| I-10 | softmax Σ=1 且非负 | ±1e-5 | Σ=1.000000 |
| I-11 | logsumexp 恒等式 | ±1e-5 | lse=1.5018 |
| I-12 | RMSNorm 单位范数 | ±1e-6 | 1.000000 |
| I-13 | RoPE 相对性 | ±1e-5 | dot1=dot2 |
| I-20 | Q8_0/Q4_K/Q6_K 误差界 | RMSE < 界 | 3.7e-3 / 5.2e-2 / 1.2e-2 |
| I-22 | dotQ ≡ 反量化点积 | ≤1e-4 | 0.00e+0 |
| I-23 | bpw 与官方一致 | 精确相等 | ✓ |
| I-30 | 梯度检验 | maxRelErr < 1e-4 | 6.44e-5 |
| I-31 | softmaxCE 梯度 = p−onehot | ±1e-5 | ✓ |
| I-32 | Adam bias correction | Δw ≈ lr | 0.1000 |
| I-33a | matmulT 梯度 | < 2e-3 | 8.56e-5 |
| I-33b | rmsNorm 梯度 | < 2e-3 | 1.96e-5 |
| I-33c | ropeRows 梯度 | < 2e-3 | 3.04e-5 |
| I-33d | causalAttn 梯度 | < 2e-3 | 9.88e-4 |
| I-34 | crossEntropyMean 梯度 | ±1e-5 | 9.93e-9 |
| I-35 | swigluRows 梯度 | < 2e-3 | 2.63e-5 |
| I-36 | embedLookup = scatter-add | ±1e-5 | 0.00e+0 |
| I-40 | QuantLinear(Q8_0) 余弦 | > 0.999 | 0.999985 |
| I-41 | SwiGLU ≡ 手动分解 | < 1e-5 | 0.00e+0 |
| I-50 | flashAttention ≡ 朴素 | < 1e-4 | 8.94e-8 |
| I-51 | 注意力权重行和 = 1 | ±1e-5 | 1.000000 |
| I-52 | 因果掩码 | ±1e-5 | ✓ |
| I-54 | KV cache ≡ 全量重算 | < 1e-5 | 8.94e-8 |
| I-60 | f32 vs Q8_0 推理余弦 | ≥0.99 且 argmax 至多 1 翻转 | 0.99531 |
| I-61 | 确定性（同输入同输出） | maxErr < 1e-6 | 0.00e+0 |
| I-70 | 训练前向 ≡ 推理前向 | < 2e-3 | 5.96e-8 |
| I-71 | 单批次过拟合 loss 下降 | < 0.6× | 2.9805→0.0006 |
| I-80 | 采样前 softmax Σ=1 | ±1e-6 | 1.000000 |
| I-81 | T=0 → greedy | 精确 | ✓ |
| I-82 | min-p 截断生效 | 有效 | 4/5 |
| I-90 | GGUF 往返逐位一致 | < 1e-5 | 0.00e+0 |
| I-91 | GGUF 元数据结构 | 精确 | ✓ |
| I-33e | 批量块对角 ≡ 逐样本 | ±1e-5 | ✓ |
| I-36b | 记忆化收敛 | loss < 0.05 | 3.479→0.000539 |

---

## 11. 已知局限

| 局限 | 原因 | 缓解/展望 |
|---|---|---|
| 性能不及 llama.cpp | 纯 JS，无 SIMD/WASM | L0 是单一替换点，可换 WASM 后端 |
| Q4_K 写入端精度略低 | 缺 importance matrix 校准 | 需要极致质量时用 `llama-quantize` |
| 训练速度慢 | 纯 JS + f32 | 教学/验证用途；生产用 PyTorch |
| 无 GPU / 无多线程 | 零依赖约束 | `worker_threads` 可并行化 batch 维度 |
| GGUFVocab 编码非完整 BPE | 未实现 merges 训练 | 读取真实 llama GGUF 时可接受（字节回退覆盖） |
| 训练示例生成质量有限 | 574 字符语料 + 76K 参数模型的容量上限 | 记忆化检验已证明闭环正确；扩语料/模型即可 |
