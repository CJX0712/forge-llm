# 架构设计 · Architecture

> forge-llm 的十层分解设计。每一层的**职责边界**、**依赖方向**、**数据契约**与**不变量**。

## 0. 设计原则

| 原则 | 落地方式 |
|---|---|
| **单向依赖** | L0 ← L1 ← L2 ← L3 ← L4 ← L5 ← L6 ← L7/L8 ← L9 ← L10。**任何层不得反向依赖**，也**不得跨层调用**（如 L6 不得直接 `Math.exp`，必须走 L1） |
| **同一函数只有一个实现** | 训练路径与推理路径共用 L1 的 RMSNorm/softmax、L2b 的 matmul 定义；差异只在「是否带反向」 |
| **可证伪优先** | 每层头部声明 `I-xx` 不变量，L10 逐条独立交叉验证。失败 → 非 0 退出码 |
| **零重写** | 每个模块先检索世界最优方案，直接复用其数学定义与位布局（见 `ALGORITHMS.md` 的复用清单） |
| **零依赖** | 无 npm 依赖、无构建步骤、无 WASM。纯 ESM + `Float32Array` |

---

## 1. 依赖图

```
                          ┌──────────────────────────┐
                          │  L10 verify.js           │  38 条不变量
                          │  runVerify() → {pass,fail}│
                          └───────────┬──────────────┘
                                      │ 引用全部层
   ┌──────────┬──────────┬────────────┼────────────┬──────────┬──────────┐
   │ L8       │ L7       │ L9 engine  │ L6         │ L5       │ L4       │
   │ sampling │ tokenize │ gguf+load  │ transformer│ attention│ nn       │
   └────┬─────┴────┬─────┴─────┬──────┴─────┬─────┴────┬────┴────┬─────┘
        │          │           │            │          │         │
        │          │           │      ┌─────┴──────┐   │         │
        │          │           │      │ L6 train   │   │         │
        │          │           │      │ (AD 路径)  │   │         │
        │          │           │      └─────┬──────┘   │         │
        │          │           │            │          │         │
        │          │           │      ┌─────┴──────────┴─────┐   │
        │          │           │      │ L2b tensor.js         │   │
        │          │           │      │ (张量级融合算子)      │   │
        │          │           │      └─────┬─────────────────┘   │
        │          │           │            │                     │
        │          │           │      ┌─────┴──────────┐          │
        │          │           │      │ L2 autograd.js │          │
        │          │           │      │ (向量级 AD+Adam)│          │
        │          │           │      └─────┬──────────┘          │
        │          │           │            │                     │
        │          │           │      ┌─────┴──────────────┐      │
        │          │           │      │ L3 quant.js        │──────┘
        │          │           │      │ (K-quant 编解码)    │
        │          │           │      └─────┬──────────────┘
        │          │           │            │
        │          │           │      ┌─────┴──────┐
        │          │           │      │ L1 special │  softmax/RMSNorm/RoPE/fp16
        │          │           │      └─────┬──────┘
        │          │           │            │
        │          │           │      ┌─────┴──────┐
        └──────────┴───────────┴──────│ L0 numeric │  GEMM/matvec/RNG
                                       └────────────┘
```

---

## 2. 逐层职责

### L0 · 数值内核 `src/L0/numeric.js` (163 行)

**职责**：整个内核的**唯一算术地基**。所有层只准通过 L0 触碰内存与浮点。

**导出**：`RNG`（xorshift128+）、`dot`、`matvec`、`gemm`、`gemmRef`、`rms`、`argmax`、`addInto`、`axpy`、`scale`、`maxAbsErr`、`rmse`、`relL2`、`cosSim`。

**关键设计**
- **权重布局**：行主序 `[N(out), K(in)]`。解码期 matvec 顺序访问权重，与 llama.cpp 的 `ggml_compute_forward_mul_mat` 内存访问模式一致。
- **GEMM**：4 路循环展开 + 行主序顺序读。缓存分块在 C/BLAS 下是标准手段，但本机 V8 实测更慢（433.9 ms vs 512.1 ms，见 ALGORITHMS §1.1），故选型由数据决定。
- **`gemmRef`**：朴素三重循环参考实现，**仅供 L10 交叉验证**，生产路径不调用 —— 这是「可证伪」的关键道具。

**不变量**：I-00（gemm ≡ 朴素，实测比特级一致）、I-01（matvec ≡ gemm(M=1)，逐位一致）。

---

### L1 · 特殊函数 `src/L1/special.js` (163 行)

**职责**：所有非线性与归一化的**唯一实现**。L2 的 AD 与 L6 的推理共用同一份代码 —— 训练/推理数值一致性靠这一点兜底。

**导出**：`softmax`、`logsumexp`、`logSoftmax`、`crossEntropy`、`rmsnorm`、`layernorm`、`rope`、`ropeTable`、`ropeApply`、`silu`、`gelu`、`erf`、`swiglu`、`fp16ToFp32`、`fp32ToFp16`。

**关键设计**
- **稳定 softmax**：减最大值后归一化；`max=-∞` 时退化为均匀分布（防御 NaN）。
- **RMSNorm**：`y = x/rms(x) · w`，eps 加在均方内部（对齐 LLaMA 官方实现）。
- **RoPE 表**：预计算 cos/sin 避免每 token 三角函数。`ropeTable` 与 `rope` 使用**完全相同的表达式形式**，保证逐位可复现。
- **fp16 查表**：L3 内部用 65536 项表替代 `Math.pow`，fp16→fp32 从 ~40ns 降到 ~1ns。

**不变量**：I-10（softmax Σ=1）、I-11（logsumexp 恒等）、I-12（RMSNorm 单位范数）、I-13（RoPE 相对性）。

---

### L2 · 自动微分 `src/L2/autograd.js` (247 行) + `src/L2/tensor.js` (479 行)

**职责**：反向模式自动微分 + 优化器。分两级：

| 文件 | 级别 | 算子 |
|---|---|---|
| `autograd.js` | 向量级 | `linear`、`add`、`mul`、`silu`、`rmsnorm`、`softmaxCE`、`dot`、`scale`、`ropeNode`、`softmaxVec`、`weightedSum` |
| `tensor.js` | 张量级（融合） | `matmulT`、`matmul`、`embedLookup`、`rmsNorm`、`ropeRows`、`sliceCols`、`causalAttn`、`causalAttnBatch`、`swigluRows`、`crossEntropyMean` |

**为什么分两级**：纯逐元素 tape 在序列长度 T 上会产生 O(T) 个节点与常数开销。PyTorch 的做法是**融合算子 + 手写反向**（`derivatives.yaml`）。本项目照此设计：张量级算子一次完成整块张量的反向。

**Tape 机制**
```
op(out, bwd)  →  创建 Node，挂上反向函数，压入 tape
backward(loss) →  loss.grad = 1；逆序遍历 tape，逐个调用 nd._b(nd.grad)
```
梯度累加在 `param.grad`（首次使用时惰性分配）。

**Adam**（Kingma & Ba 2015）：含 bias correction（论文式 (7)(8)）、全局范数梯度裁剪（Pascanu 2013）、余弦退火（SGDR, Loshchilov & Hutter 2017）。

**不变量**：I-30（梯度检验）、I-31（softmaxCE 梯度 = p−onehot）、I-32（Adam bias correction）、I-33a~d（张量级算子梯度 = 中心差分）、I-34（crossEntropyMean 梯度 = (P−OneHot)/T）、I-35（swiglu 梯度）、I-36（embedLookup = scatter-add）。

---

### L3 · 量化内核 `src/L3/quant.js` (453 行)

**职责**：K-quant 的编解码 + 直接量化点积（不物化反量化数组）。

**导出**：`dequantize`、`quantizeRow`、`dotQ`、`rowBytes`、`info`、`qualityReport`、`fp16Table`、各格式的独立写入端。

**关键设计**
- **位布局逐位对齐 llama.cpp** `ggml-quants.c`（ikawrakow K-quant, PR #1684）。因此能直接吃 HuggingFace 上任何 Q4_K_M/Q5_K_M/Q6_K/Q8_0/F16 的 GGUF 权重。
- **`dotQ`**：量化权重 · f32 激活的融合点积，是解码期的全部热点（等价于 llama.cpp 的 `ggml_vec_dot_q`）。**绝不物化反量化后的 f32 数组** —— 这既是速度来源，也是内存优势。
- **自研部分与理由**：
  - `quantizeRowQ4K`：llama.cpp 的 Q4_K 写入端依赖 importance-matrix 迭代搜索（`make_qkx3_quants`/`make_qp_quants`），需要校准集，在零依赖内核里不划算。本项目给出闭式 min-max + 6-bit 层级 scale 的实现，**格式完全兼容**，代价是 RMSE 略高（实测 5.45e-2 vs llama.cpp 的约 4.5e-2）。如需极致质量请用 `llama-quantize`。
  - `quantizeRowQ6K`：**完整移植** llama.cpp 的 `quantize_row_q6_K_impl`（含 RMSE 迭代搜索），因此 Q6_K 精度与官方一致。

**不变量**：I-20（量化误差界）、I-21（K-quant 层级 scale 与 llama.cpp 逐位一致）、I-22（dotQ ≡ 反量化后点积）、I-23（bpw 与官方表一致）。

---

### L4 · 神经组件 `src/L4/nn.js` (69 行)

**职责**：把「权重」抽象成**可插拔后端**。

**核心：`QuantLinear`**
```js
class QuantLinear {
  constructor(type, bytes, N, K)   // type: 0=F32 1=F16 8=Q8_0 12=Q4_K 14=Q6_K
  forward(x, out, xOff = 0)        // type=0 → matvec；否则 → 逐行 dotQ
  toF32()                          // 反量化（验证/导出用）
  bytesSize()                      // 实际占用字节
}
```

**这意味着同一份 forward 代码既能跑 f32，也能直接吃量化字节流** —— 训练用 f32，部署用量化，**forward 只有一份**。

**不变量**：I-40（QuantLinear(Q8_0) 与 f32 余弦 > 0.999）、I-41（SwiGLU 块 ≡ 手动分解）。

---

### L5 · 注意力引擎 `src/L5/attention.js` (149 行)

**职责**：注意力的三种实现 + 两种 KV 缓存。

| 组件 | 用途 | 复杂度 |
|---|---|---|
| `attentionNaive` | 参考实现，供 L10 互拍 | O(n²d) 时间，O(n²) 内存 |
| `flashAttention` | 在线 softmax，**推理默认路径** | O(nd) 时间，**O(d) 内存** |
| `KVCache` | 连续布局，零间接寻址 | 推理默认 |
| `PagedKVCache` | 分页分配，支持变长/批处理 | vLLM PagedAttention 思想 |

**在线 softmax（FlashAttention, Dao et al. 2022）**：维护运行最大值 `m` 与归一化因子 `ℓ`，新增块时用 `α = exp(m_old − m_new)` 重缩放，**永不物化 N×N 注意力矩阵**。

**不变量**：I-50（flash ≡ 朴素）、I-51（权重行和 = 1）、I-52（因果掩码）、I-54（KV cache ≡ 全量重算）。

---

### L6 · 模型层

#### `src/L6/transformer.js` (181 行) —— 推理路径

**Llama-2 完整拓扑**（Touvron et al. 2023）：
```
x = Embed(token)
for each layer:
    x += Wo · Attn(RMSNorm(x))      # GQA：n_kv < n_q 时按组复用 KV
    x += W_down · SwiGLU(RMSNorm(x)) # gate/up 双投影 + silu 门控
logits = W_cls · RMSNorm(x)        # 权重共享（缺省复用 tok_emb）
```

**关键设计**
- **RoPE 预计算表**：每 token 只查表不计算三角函数。
- **在线 softmax 因果注意力**：永不物化 N×N。
- **GQA**：`(h / groupSize) | 0` 取所属 kv-group。

#### `src/L6/train.js` (285 行) —— 训练路径

**与推理路径结构完全同构**，区别只在算子来自 L2b 的 AD。

**关键设计**
- **块对角因果掩码**（`causalAttnBatch`）：训练时 batch 内样本**互不可见**，等价于 HF 的 4D attn_mask。这一点由 L10 的「批量块对角 == 逐样本独立计算」不变量守护。
- **RoPE 位置在样本内重新计数**（`ropeRows` 的 `posStride` 参数）。
- **权重初始化**：GPT-2 式 N(0, 0.02) + 残差投影 1/√(2·nLayers) 缩放。

**不变量**：I-70（训练前向 ≡ 推理前向，逐位）、I-71（单批次过拟合 loss 显著下降）。

---

### L7 · 分词器 `src/L7/tokenizer.js` (138 行)

| 类 | 用途 |
|---|---|
| `CharTokenizer` | 字符级，Karpathy llama2.c 同款。训练演示用 |
| `GGUFVocab` | 对齐 llama.cpp 的词表加载/解码，含字节回退 |

**解码侧完全正确**：把每个 token 当作**原始字节串**（type=1 即单字节），拼接后按 UTF-8 还原 —— 与 llama.cpp 的 `llama_token_to_piece` 行为一致。

**编码侧**：基于字节的最长匹配（含 byte-fallback token）。

---

### L8 · 采样 `src/L8/sampling.js` (106 行)

对齐当前 SOTA 解码配方，全部基于 L0 的确定性 RNG（**可复现**）：

- `temperature`（先除温再 softmax）、`top_k`（Fan et al. 2018）、`top_p`/nucleus（Holtzman et al. 2020）
- `min_p`（Nguyen et al. 2024, arXiv:2407.01082）：阈值 = `min_p · p_max`，比 top-p 更稳
- 频率/存在惩罚（OpenAI 2020）、greedy（argmax，用于确定性评测）

---

### L9 · GGUF + 引擎

#### `src/L9/gguf.js` (218 行) —— GGUF v3 读写

严格的 GGUF v3 实现：
- **读**：能加载任意合规 GGUF（含真实 Llama/Qwen 权重，只要架构字段匹配）。
- **写**：本项目产出的 GGUF 可被 llama.cpp / 本项目互相读取。
- **对齐**：`general.alignment`（默认 32）—— 数据段起始地址是 alignment 的整数倍；每个张量按 alignment 补齐。

#### `src/L9/engine.js` (262 行) —— 引擎

```
loadGGUF(buf) → { config, transformer, vocab, metadata }
exportGGUF(weights, config, vocab, quantType, name) → Uint8Array
generate(model, {prompt, maxTokens, temperature, top_k, top_p, min_p, seed}) → string
```

**张量命名**（与 llama.cpp 兼容风格）：
```
token_embd.weight · output.weight · output_norm.weight
blk.{l}.attn_norm.weight · blk.{l}.attn_qkv.weight · blk.{l}.attn_output.weight
blk.{l}.ffn_norm.weight · blk.{l}.ffn_gate.weight · blk.{l}.ffn_up.weight · blk.{l}.ffn_down.weight
```

**生成循环**：prefill 整段 prompt 填充 KV 缓存 → 逐 token 自回归采样。上下文边界受 `maxSeq` 硬约束。

---

### L10 · 验证层 `src/L10/verify.js` (530 行)

**38 条不变量**的独立交叉验证。每条不变量都遵循一个模式：

```
构造两个独立路径 → 断言它们一致
```

| 模式 | 例子 |
|---|---|
| **解析 vs 数值** | 解析梯度 vs 中心差分（I-30, I-33a~d） |
| **快路径 vs 慢路径** | flashAttention vs 朴素（I-50）；KV cache vs 全量重算（I-54） |
| **量化 vs 精确** | dotQ vs 反量化后点积（I-22） |
| **训练 vs 推理** | 训练前向 vs 推理前向（I-70） |
| **写 vs 读** | GGUF 导出→重加载输出逐位一致（I-90） |
| **批量 vs 逐样本** | 块对角掩码 vs 独立计算 |
| **公式 vs 定理** | softmax Σ=1、logsumexp 恒等、RoPE 相对性、Adam bias correction |

**任何一条失败 → 退出码非 0。**

---

## 3. 数据流：训练 → 推理的一致性保证

```
训练路径                              推理路径
────────                              ────────
embedLookup(E, ids)          ≡       _embed(token)         → QuantLinear 行拷贝
rmsNorm(x, w, T, D)          ≡       rmsnorm(xb, x, w, D)  → L1 同一实现
matmulT(xn, W)               ≡       W.forward(x, out)     → 同为 Σ_k X[k]·W[n,k]
ropeRows(q, T, nH, dH)       ≡       applyRopeHead(...)    → 同一 RoPE 公式
causalAttn(q, k, v)          ≡       flashAttention(...)   → 数学等价（在线 softmax）
swigluRows(g, u)             ≡       SwiGLU.forward(...)   → 同一 silu·u
matmulT(xf, Wcls)            ≡       wcls.forward(xb, out)

                          ↓
              I-70 实测 maxLogitsDiff = 5.96e-8
```

这条等式链是本项目最核心的架构价值：**训练出来的权重不需要任何「转换」就能直接被推理引擎加载**，且数值上等价。

---

## 4. 内存布局与性能设计

| 决策 | 理由 |
|---|---|
| 权重行主序 `[N, K]` | 解码期 matvec 顺序访问，cache 友好；与 llama.cpp 一致 |
| KV 连续布局（`KVCache`） | 零间接寻址，推理默认路径 |
| `dotQ` 融合反量化+点积 | 避免物化 f32 数组 —— 省内存 + 省带宽 |
| RoPE 预计算表 | 每 token 省 N 次三角函数 |
| 训练用块对角掩码而非 padding | 样本间零干扰，梯度无偏 |
| f32 而非 f64 | 与 GGUF/llama.cpp 生态一致；误差界已在不变量中量化 |

---

## 5. 扩展点

| 想做什么 | 改哪里 |
|---|---|
| 支持新架构（Qwen/Mistral/Phi） | L6 `Config` + `Transformer.loadWeights` 的张量名映射 |
| 支持新量化格式（IQ 系列/AWQ/GPTQ） | L3 加 `dequantize`/`dotQ` 分支 + `TYPE_INFO` 表项 |
| 支持真实 BPE 词表 | L7 `GGUFVocab.encode` 接 `merges` |
| 接入 GPU / WASM | 只需替换 L0 的 `matvec`/`gemm`，上层零改动 |
| 加新不变量 | L10 加一个 `check()` 调用 |
