/**
 * forge-llm · 公开 API 入口
 * ------------------------------------------------------------------
 * 十层内核的统一出口。按需 import 即可，无需构建、无第三方依赖。
 *
 *   import { loadGGUF, generate } from "forge-llm";
 *   const model = loadGGUF(bytes);
 *   console.log(generate(model, { prompt: "你好", maxTokens: 64 }));
 */

// L0 · 数值
export { RNG, dot, matvec, gemm, gemmRef, rms, argmax, maxAbsErr, rmse, relL2, cosSim, addInto, axpy, scale } from "./L0/numeric.js";
// L1 · 特殊函数
export { softmax, logsumexp, logSoftmax, crossEntropy, rmsnorm, layernorm, rope, ropeTable, ropeApply, silu, gelu, erf, swiglu, fp16ToFp32, fp32ToFp16 } from "./L1/special.js";
// L2 · 自动微分
export { Node, leaf, param, op, tape, resetTape, noGrad, linear, add, mul, silu as siluAD, rmsnorm as rmsnormAD, softmaxCE, backward, gradCheck, Adam, dot as dotAD, scale as scaleAD, ropeNode, softmaxVec } from "./L2/autograd.js";
export { matmulT, matmul, embedLookup, rmsNorm, ropeRows, sliceCols, causalAttn, swigluRows, crossEntropyMean, weightedSum } from "./L2/tensor.js";
// L3 · 量化
export { dequantize, quantizeRow, dotQ, rowBytes, info, qualityReport, TYPE_INFO, GGMLType, fp16Table, quantizeRowQ4K, quantizeRowQ6K, quantizeRowQ8_0, quantizeRowF16 } from "./L3/quant.js";
// L4 · 组件
export { QuantLinear, SwiGLU, rmsnormInto } from "./L4/nn.js";
// L5 · 注意力
export { attentionNaive, flashAttention, KVCache, PagedKVCache } from "./L5/attention.js";
// L6 · 模型
export { Config, Transformer } from "./L6/transformer.js";
export { TrainableTransformer, initWeights, train } from "./L6/train.js";
// L7 · 分词器
export { CharTokenizer, GGUFVocab, parseTokenBytes } from "./L7/tokenizer.js";
// L8 · 采样
export { sample, greedyId } from "./L8/sampling.js";
// L9 · GGUF + 引擎
export { readGGUF, writeGGUF, GGUFValueType, ggmlTypeSize } from "./L9/gguf.js";
export { loadGGUF, exportGGUF, toTransformer, generate } from "./L9/engine.js";
// L10 · 验证
export { runVerify } from "./L10/verify.js";
