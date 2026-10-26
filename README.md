# FORGE · Verifiable AI Kernel

> **零依赖 · 纯 JavaScript · 十层可验证 LLM 内核** —— 从数值底座到 GGUF 推理，全链路自带**可执行数学不变量**。
> 作者：**晨星 (Chenxing)** · License: Apache-2.0

```
forge-llm 不是一个「能跑」的 demo，而是一个「可被证伪」的 LLM 内核。
每一层都声明自己的数学契约，每一条契约都在 L10 验证层被独立交叉验证。
任何一条不变量失败 → 退出码非 0 → CI 直接红。
```

---

## 为什么是它

| 维度 | 主流方案 | forge-llm |
|---|---|---|
| 运行时 | Python + PyTorch（GB 级依赖） | 纯 JS，**零依赖零构建**，Node ≥ 20 |
| 正确性保障 | 靠 CI 跑测试 | **40 条数学不变量**直接断言数学性质（梯度=中心差分、softmax Σ=1、RoPE 相对性、量化误差界、GGUF 字节级往返…） |
| 量化 | 依赖 llama.cpp / AWQ / GPTQ | **自带** Q4_K/Q5_K/Q6_K/Q8_0/F16 编解码，位布局经 llama.cpp 官方反量化器**逐位验证**（max&#124;Δ&#124;=0；Q8_0/F16 与官方写入端逐字节一致） |
| 模型格式 | 各自实现 | **GGUF v3 读写器**，可与 llama.cpp 互操作 |
| 训练 | PyTorch | **自带反向模式 AD**（向量级 + 张量级融合算子），训练前向与推理前向**逐位一致** |
| 可复现 | 依赖环境 | 固定 RNG 种子 + 纯函数 + 无隐藏状态 |

**零重写原则**：本项目没有重新发明矩阵乘、注意力、量化格式或 GGUF 规范。每一个模块都先检索当前世界最优的现成成果，直接复用其数学定义与位布局：

- 量化位布局 → llama.cpp `ggml-quants.c`（ikawrakow K-quant, PR #1684）
- GGUF 格式 → ggml 官方规范 v3
- 注意力在线 softmax → FlashAttention (Dao et al. 2022)
- GQA → Ainslie et al. 2023
- RoPE → Su et al. 2023
- SwiGLU → Shazeer 2020
- min-p 采样 → Nguyen et al. 2024 (arXiv:2407.01082)
- Adam → Kingma & Ba 2015；余弦退火 → Loshchilov & Hutter 2017；梯度裁剪 → Pascanu et al. 2013
- 反向模式 AD → Linnainmaa 1970 / Rumelhart 1986

自研部分仅在**现成方案有硬性缺口**处，且均给出替代理由（见 `docs/ALGORITHMS.md`）。

---

## 30 秒上手

```bash
git clone https://github.com/CJX0712/forge-llm.git
cd forge-llm
node cli/forge.mjs verify          # 40 条不变量自检（应全部 ✅）
node cli/forge.mjs bench           # 性能基准（B1–B6）
node cli/forge.mjs train --steps 200 --out out/tiny.gguf
node cli/forge.mjs chat out/tiny.gguf --prompt "the " --tokens 120
```

零安装步骤：**不需要 `npm install`**（无依赖，Node ≥ 20 即可）。
或跑 [`examples/01-quickstart.mjs`](examples/01-quickstart.mjs) 看完整最小路径。

---

## 项目主页

- 仓库：<https://github.com/CJX0712/forge-llm>
- 文档入口：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) · [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) · [`docs/REFERENCES.md`](docs/REFERENCES.md)
- 一键安装脚本：[`install.sh`](install.sh) / [`install.ps1`](install.ps1)

---

## 十层架构

```
L10  验证层   runVerify()          40 条不变量，失败即退出码非 0
L9   引擎层   loadGGUF / generate  GGUF v3 读写 + 自回归生成
L8   采样层   temperature/top-k/top-p/min-p/贪心
L7   分词层   CharTokenizer / GGUFVocab（字节回退）
L6   模型层   Transformer（推理） + TrainableTransformer（训练）
L5   注意力   两趟扫描(解码) / 在线 softmax / KV 缓存 / 分页 KV / GQA
L4   组件层   QuantLinear（量化感知线性层）/ SwiGLU
L3   量化层   Q4_K/Q5_K/Q6_K/Q8_0/F16 编解码（对齐 llama.cpp，Q5_K 为官方 ref 移植）
L2   自动微分 向量级 AD + 张量级融合算子 + Adam
L1   特殊函数 softmax / RMSNorm / RoPE / SwiGLU / erf / fp16
L0   数值内核 GEMM(4路展开) / matvec / xorshift128+ RNG
```

每层只允许通过下层的公开接口触碰内存与浮点——这保证了「同一个函数只有一个实现」。

---

## 目录结构

```
forge-llm/
├── cli/forge.mjs              CLI 入口（verify/bench/train/chat/quantize/export/inspect/test）
├── src/
│   ├── index.js               公开 API 汇总导出
│   ├── L0/numeric.js          数值底座
│   ├── L1/special.js          softmax/RMSNorm/RoPE/fp16
│   ├── L2/autograd.js         向量级反向模式 AD + Adam
│   ├── L2/tensor.js           张量级融合算子（matmul/attention/cross-entropy）
│   ├── L3/quant.js            K-quant 编解码 + 质量报告
│   ├── L4/nn.js               QuantLinear / SwiGLU
│   ├── L5/attention.js        在线 softmax / KVCache / PagedKVCache
│   ├── L6/transformer.js      推理 Transformer
│   ├── L6/train.js            可训练 Transformer + 训练循环
│   ├── L7/tokenizer.js        CharTokenizer / GGUFVocab
│   ├── L8/sampling.js         采样策略
│   ├── L9/gguf.js             GGUF v3 读写
│   ├── L9/engine.js           加载 / 导出 / 生成
│   └── L10/verify.js          40 条不变量
├── bench/bench.mjs            B1–B6 性能基准
├── examples/                  01 快速上手 / 02 量化 / 03 训练闭环
├── test/                      59 个 node:test 用例（含 CLI 冒烟）
├── tools/quantize-gguf.mjs    GGUF 量化器
├── tools/interop/             GGUF 外部证伪（官方 gguf-py + 独立 numpy 对照）
├── docs/                      架构 / 算法 / 性能 / 引用
├── install.sh / install.ps1   一键安装与验证
├── LICENSE                    Apache-2.0
└── package.json               零依赖
```

---

## 核心能力

### 1. 训练 → 导出 → 推理 → 量化，全链路闭环

```bash
node cli/forge.mjs train --steps 300 --out out/tiny.gguf     # 训练并导出 GGUF
node cli/forge.mjs quantize out/tiny.gguf out/tiny-q4.gguf --type 12   # 量化
node cli/forge.mjs chat out/tiny-q4.gguf --prompt "the "     # 推理
node cli/forge.mjs inspect out/tiny-q4.gguf                  # 查看元数据
```

### 2. 40 条数学不变量（可执行）

```
I-00  gemm 与朴素三重循环一致            relL2 = 0.00e+0
I-01  matvec 与 gemm(M=1) 一致           maxErr = 0
I-10  softmax 输出 Σ=1 且非负             Σ = 1.000000
I-13  RoPE 相对性（只依赖 m−n）          dot1 = dot2 = −1.4090
I-20  Q8_0/Q4_K/Q6_K 量化误差界          RMSE = 3.72e-3 / 5.18e-2 / 1.21e-2
I-23  bit-per-weight 与官方一致          Q4_K=4.5 Q6_K=6.5625 Q8_0=8.5
I-30  梯度检验 maxRelErr < 1e-4          maxRelErr = 6.44e-5
I-33a matmulT 梯度 = 中心差分            maxRelErr = 8.56e-5
I-33d causalAttn 梯度 = 中心差分         maxRelErr = 9.88e-4
I-34  crossEntropyMean 梯度 = (P−OneHot)/T
I-50  flashAttention 与朴素一致          maxErr = 8.94e-8
I-70  训练前向 == 推理前向（逐位）        maxLogitsDiff = 5.96e-8
I-90  GGUF 导出→重加载 输出逐位一致      maxLogitsDiff = 0.00e+0
```

完整清单见 `node cli/forge.mjs verify` 输出与 `docs/ALGORITHMS.md`。

### 3. 量化内核对齐 llama.cpp

（n=4096，N(0,1)，RNG 种子 1，`qualityReport` 实跑）

| 格式 | bpw | 压缩比 | RMSE | 余弦 |
|---|---|---|---|---|
| F16 | 16 | 2.00× | 2.15e-4 | 1.000000 |
| Q8_0 | 8.5 | 3.76× | 5.54e-3 | 0.999985 |
| Q6_K | 6.5625 | 4.88× | 1.82e-2 | 0.999842 |
| Q5_K | 5.5 | 5.82× | 3.69e-2 | 0.999349 |
| Q4_K | 4.5 | 7.11× | 8.07e-2 | 0.996913 |

Q5_K 是 llama.cpp `quantize_row_q5_K_ref` 的忠实移植（含 `make_qkx2_quants` 加权网格搜索），
Q4_K 为自研闭式 min-max 版（格式兼容、质量略低）——两者的字节都已被官方反量化器验证可读。

### 4. GGUF 互操作：外部证伪（不是自证）

不变量是**自洽性**检验——写端与读端同时犯同一个错时，它们依然全绿。
所以本项目另设一条**外部证据链**：用第三方实现给 forge 挑错。

| 环节 | 外部锚点 | 结果 |
|---|---|---|
| ① 权重来源 | llama.cpp 官方 `gguf-py` 的 `GGUFWriter`（不经 forge 之手） | 21 张量 / 434 KB |
| ② 读官方文件 | forge `loadGGUF` → 独立 **numpy** 前向复算，逐点对拍 | embed / afterAttn / afterFFN / xFinal / logits 全 ≤ **1.7e-07** |
| ③ 出厂前向 | `Transformer.forward`（+KVCache + gatherKV）vs numpy 参考 logits | max&#124;Δ&#124; = **3.149e-07**，argmax 一致 |
| ④ **反向**闭环 | forge **写**出的 GGUF → 官方 `GGUFReader` 读回，逐张量比 dims + sha256 | 20/20 张量 **逐位一致** |
| ⑤ 量化位布局 | forge 量化出的字节 → 官方 `gguf.quants.dequantize` 读回，与 forge 反量化对拍 | 5/5 格式 max&#124;Δ&#124; = **0**；Q8_0/F16 与官方写入端**逐字节一致** |

```bash
node tools/interop/run_interop.mjs --force     # 一键跑完 8 步，退出码 0 = 全 PASS
```

详见 [`tools/interop/README.md`](tools/interop/README.md)。

> 这层检查**已实证抓到三个真实缺陷**：
> ① GGUF 字符串长度前缀宽度写错（导致 forge 导出的 GGUF 此前 llama.cpp **完全读不了**，
> 官方 reader 直接崩溃）；② **Q6_K 的 super-block scale `d` 被放在块头**——Q4_K/Q5_K 的
> `d` 在块首，Q6_K 却在块尾（`ql[128] qh[64] scales[16] d[2]`），写端读端同错所以自洽
> 往返全绿，而 llama.cpp 读 forge 的 Q6_K 会读出 **2.3e8** 量级的垃圾；
> ③ 一处参照实现自身的 numpy 视图别名 bug（曾一度误判为 forge 发散）。详见 interop README。

### 5. 作为库使用

```js
import { loadGGUF, generate, sample, dequantize } from "forge-llm";

const model = loadGGUF(new Uint8Array(buf));
console.log(generate(model, { prompt: "你好", maxTokens: 64, min_p: 0.05 }));
```

---

## 验收标准对照

| 标准 | 状态 | 证据 |
|---|---|---|
| 架构先进 | ✅ | Llama-2 全拓扑（GQA+RMSNorm+RoPE+SwiGLU）+ FlashAttention + K-quant + GGUF v3 |
| 性能可量化 | ✅ | B1–B6 六组基准，固定种子可复现（`docs/PERFORMANCE.md`） |
| 代码精简无冗余 | ✅ | 零依赖、零构建；3,453 行源码覆盖训练+量化+推理全栈（+852 行 CLI/示例/基准） |
| 文档齐全 | ✅ | 架构 / 算法原理 / 模块职责 / 使用方式 / 性能对比 / 引用来源 六份文档 |
| 他人可一键复现 | ✅ | `install.sh` / `install.ps1`；`verify` 全绿即证明环境正确 |
| **格式与数学可被外部证伪** | ✅ | 官方 `gguf-py` 双向互操作 + 独立 numpy 前向对拍，max&#124;Δlogits&#124; = 3.15e-07（`tools/interop/`） |

---

## 与现有顶级方案的对比依据

| 方案 | 强项 | forge-llm 的差异化 |
|---|---|---|
| **llama.cpp** | C++ 极致性能、完整量化生态 | 性能不在一个量级；本项目胜在**可验证性**（40 条不变量）、**零依赖**、**自带训练闭环**、可作为 JS 侧的教学/测试基准 |
| **PyTorch** | 训练/自动微分工业标准 | PyTorch 无法做「读 GGUF → 量化 → 推理」这件事的一体化轻量实现；本项目补上了这条链，且**训练前向与推理前向逐位一致**（I-70） |
| **nanoGPT** | 极简可读的 LLM 训练 | 仅 PyTorch；本项目把同样的架构落到零依赖 JS，并额外提供量化与 GGUF 互操作 |
| **ggml / gguf-py** | 格式与工具链的事实标准 | 本项目是**纯 JS 的等价实现**，位布局逐位对齐，可与 llama.cpp 互读 |

---

## License

Apache-2.0 · 作者 **晨星 (Chenxing)** · 贡献与问题反馈欢迎提 issue。
