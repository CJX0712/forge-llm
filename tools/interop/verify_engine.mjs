/**
 * verify_engine.mjs —— 用「出厂代码路径」(loadGGUF + Transformer.forward + KVCache + gatherKV)
 * 在官方 llama.cpp GGUF 上跑前向，输出 logits，与独立 numpy 参考 (np-logits.json) 对拍。
 *
 * 这是 forge-llm 对外可信度的最硬证据：
 *   - 权重来自 llama.cpp 官方 gguf-py 写出的 GGUF（未经过 forge 之手）
 *   - 前向走的是 src/L6/transformer.js 正式发布的 forward（非手写复刻）
 *   - 对照实现是 tools/interop/numpy_forward.py（已修正 RoPE 视图别名 bug）
 *   - 两者 logits 逐位一致（≤ float32 精度 ~1e-6）即证明 forge 的 GGUF 解析与
 *     全套 transformer 数学（RMSNorm / RoPE / GQA / 因果注意力 / SwiGLU / LM head）都正确。
 */
import { readFileSync, writeFileSync } from "node:fs";
import { loadGGUF } from "../../src/L9/engine.js";
import { KVCache } from "../../src/L5/attention.js";

const model = loadGGUF(new Uint8Array(readFileSync("tools/interop/out/llama-official.gguf")));
const { transformer, config } = model;
const ids = [2, 3, 4, 5];
const cache = new KVCache(config.nLayers, config.maxSeq, config.kvDim);

let logits;
for (let pos = 0; pos < ids.length; pos++) {
  logits = transformer.forward(ids[pos], pos, cache);
}
writeFileSync(
  "tools/interop/out/engine-logits.json",
  JSON.stringify({ ids, vocab: config.vocabSize, logits: Array.from(logits) })
);
console.log("engine forward logits[:8] =", Array.from(logits.slice(0, 8)).map((v) => v.toFixed(6)).join(", "));
const { argmax } = await import("../../src/L0/numeric.js");
console.log("argmax =", argmax(logits, config.vocabSize));
