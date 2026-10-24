# REFERENCES · 引用来源

本项目遵循「禁止从零造轮子」原则：每个模块先检索当前世界最优的现成成果，
直接复用其数学模型与位布局约定，只在必要时自研并说明理由。以下按模块列出
Standing on the shoulders of giants 的完整清单。

## 架构与模型

| 组件 | 来源 | 本项目如何复用 |
|---|---|---|
| Llama-2 整体架构 | Touvron et al. 2023, *Llama 2: Open Foundation and Fine-Tuned Chat Models* | RMSNorm + RoPE + SwiGLU + GQA 的组合方式（无 bias、预归一化） |
| GQA（分组查询注意力） | Ainslie et al. 2023, *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* | n_kv < n_q 时按 group 复用 KV；`Config._derive` 强制 n_heads % n_kv_heads == 0 |
| SwiGLU FFN | Shazeer 2020, *GLU Variants Improve Transformer* | gate/up/down 三投影 + SiLU 门控 |
| RoPE（旋转位置编码） | Su et al. 2021/2023, *RoFormer: Enhanced Transformer with Rotary Position Embedding* | 预计算 cos/sin 表，逐 head 原地旋转；I-53 验证相对性 |

## 注意力与推理

| 组件 | 来源 | 本项目如何复用 |
|---|---|---|
| 在线 softmax | Milakov & Gummadi 2018, *Online normalizer calculation for softmax*；Dao et al. 2022, *FlashAttention: Fast and Memory-efficient Exact Attention with IO-Awareness* | `flashAttention`：维护 (m, ℓ, acc)，新块 α=exp(m_old−m_new) 重缩放；与 naive 版数值互拍（I-50） |
| PagedAttention（分页 KV） | Kwon et al. 2023, *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM) | `PagedKVCache`：按块分配、变长序列、零拷贝复用的简洁版 |
| KV Cache 连续布局 | llama.cpp `ggml_compute_forward_mul_mat` 的访存模式 | `KVCache`：nLayers × maxSeq × kvDim 连续缓冲，kView/vView 零间接寻址 |

## 量化

| 组件 | 来源 | 本项目如何复用 |
|---|---|---|
| GGUF 容器格式 | ggerganov/llama.cpp, `gguf-py/gguf/constants.py` 与 `ggml/include/ggml.h` | 元数据键值、张量对齐（默认 32）、`ggml_type` 枚举与块大小，逐一对齐 |
| Q4_K / Q5_K / Q6_K / Q8_0 位布局 | llama.cpp `src/ggml-quants.c`（`quantize_row_q4_K_ref` / `dequantize_row_q4_K` 等） | super-block 256、12 字节 6-bit scales（`get_scale_min_k4` 的两种编码分支）、fp16 d/dmin、q4  nibble + qh 高位 —— 按官方参考实现逐行对齐 |
| Q8_0 | 同上（`quantize_row_q8_0_ref`） | fp16 d + 32×int8，块长 32 |
| fp16 ↔ fp32 | IEEE 754-2008 half precision | 65536 项查表（含 subnormal 分支），避免逐次转换开销 |

## 数值与优化

| 组件 | 来源 | 本项目如何复用 |
|---|---|---|
| 4 路循环展开 + FMA 依赖链隐藏 | 经典高性能计算手法（如 *Optimizing subroutines in assembly language*，Agner Fog） | `gemm`/`matvec` 的 s0..s3 独立累加器 |
| 梯度检验（中心差分） | 标准做法（CS231n / *Numerical Optimization*） | `gradCheck`：maxRelErr ≈ 1e-7（I-30 阈值 1e-5） |
| Adam | Kingma & Ba 2015, *Adam: A Method for Stochastic Optimization* | 含 bias 修正、余弦/线性衰减、梯度裁剪 |
| 交叉熵 + logsumexp | 数值稳定标准做法 | L1 `logsumexp`：max 平移防溢出 |
| xorshift128+ RNG | Vigna 2017, *Further scramblings of Marsaglia's xorshift* | 确定性伪随机：可复现基准的必要前提 |

## 采样

| 组件 | 来源 | 本项目如何复用 |
|---|---|---|
| min-p 采样 | Nguyen et al. 2024, *Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs* (arXiv:2407.01082) | 阈值 = min_p × 最大概率，替代 top-p 的绝对累积截断 |
| top-k / top-p（核采样） | Holtzman et al. 2020, *The Curious Case of Neural Text Degeneration*；Fan et al. 2018, *Hierarchical Neural Story Generation* | `sample()` 的温度/截断链 |
| 频率惩罚 | OpenAI GPT-3 2020, *Language Models are Few-Shot Learners* | presence/frequency penalty |

## 工程实践

| 组件 | 来源 | 本项目如何复用 |
|---|---|---|
| 「可证伪」的不变量测试法 | 性质测试 / metamorphic testing 思想 | L10 把每条数学性质变成可执行断言（38 项） |
| Apache 2.0 许可证 | Apache Software Foundation | 项目许可，署名晨星 |

## 与现有顶级方案的定位对比

| 方案 | 语言/生态 | 与 forge-llm 的关系 |
|---|---|---|
| **llama.cpp** | C/C++，GGUF 生态事实标准 | 位布局与格式的**兼容对象**：本项目读写同一份 GGUF 规范；性能目标不可比（C vs JS），但模型可互操作 |
| **ggml** | C，张量库 | 量化格式与 mul_mat 访存模式的参照实现 |
| **transformers (HF)** | Python，训练生态 | 架构定义（Llama-2 配置项命名）的来源；本项目只做推理+微型训练 |
| **micrograd** (Karpathy) | JS/Python，教学 | 自动微分的极简思想先驱；本项目的 L2 在其「标量 tape」思想上扩展为张量级算子 |
| **tinygrad** | Python，极简框架 | 「一层内核吃所有后端」的架构灵感 |
| **onnxruntime / TensorRT** | C++/多语言 | 生产部署的下游替换目标（本项目的 GGUF 可经转换接入） |

**结论**：forge-llm 不与上述任何方案竞争性能，它占据的生态位是
「**零依赖、零构建、可验证的 AI 内核教科书**」——用一份能跑的 JS 代码，
把从数值内核到 GGUF 推理的每一层数学契约变成可执行断言。
