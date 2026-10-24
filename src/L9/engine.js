/**
 * L9b · 推理引擎 (Engine)
 * ------------------------------------------------------------------
 * 把 L3/L4/L5/L6/L7/L8 串成「读 GGUF → 装配 Transformer → 自回归生成」的完整链路。
 *
 * 设计要点：
 *  - loadGGUF 是严格的 GGUF v3 读取器（见 gguf.js），能加载任意合规 GGUF；
 *    本项目的 exportGGUF 与之一一对应，因此「导出 → 重加载」可逐字节/逐数值自洽。
 *  - 权重后端可插拔：GGUF 里的 Q4_K/Q8_0/F16 张量被直接封装成 QuantLinear，
 *    无需反量化即可喂给 L6 的 forward。
 *  - 生成循环：prefill 整段 prompt 填充 KV 缓存，随后逐 token 自回归采样。
 */

import { Config, Transformer } from "../L6/transformer.js";
import { QuantLinear } from "../L4/nn.js";
import { KVCache } from "../L5/attention.js";
import { RNG } from "../L0/numeric.js";
import { sample } from "../L8/sampling.js";
import { GGUFVocab } from "../L7/tokenizer.js";
import {
  readGGUF, writeGGUF, GGUFValueType,
} from "./gguf.js";
import { quantizeRow, rowBytes, info } from "../L3/quant.js";

// ── 张量名约定（与 llama.cpp 兼容风格）────────────────────────────────────
const T = {
  emb: "token_embd.weight",
  lnF: "output_norm.weight",
  ln1: (l) => `blk.${l}.attn_norm.weight`,
  wqkv: (l) => `blk.${l}.attn_qkv.weight`,
  wo: (l) => `blk.${l}.attn_output.weight`,
  ln2: (l) => `blk.${l}.ffn_norm.weight`,
  fg: (l) => `blk.${l}.ffn_gate.weight`,
  fu: (l) => `blk.${l}.ffn_up.weight`,
  fd: (l) => `blk.${l}.ffn_down.weight`,
  out: "output.weight",
};

// ── 从 f32 权重构造 QuantLinear ─────────────────────────────────────────
function wm(tensor, N, K, quantType) {
  const type = quantType ?? 0;
  if (type === 0) {
    // 直接以 Float32Array 存储（QuantLinear.fromF32 对 type 0 即原样保留）
    const W = tensor instanceof Float32Array ? tensor : Float32Array.from(tensor);
    return QuantLinear.fromF32(W, N, K, 0);
  }
  const bytes = quantizeRow(type, tensor, N * K);
  return new QuantLinear(type, bytes, N, K);
}
function nm(tensor, dim) {
  return tensor instanceof Float32Array ? tensor : Float32Array.from(tensor);
}

/** 把权重对象灌进一个可推理的 Transformer */
export function toTransformer(weights, config, quantType = 0) {
  const c = config instanceof Config ? config : new Config(config);
  const tr = new Transformer(c);
  const w = {
    tok_emb: wm(weights.tok_emb, c.vocabSize, c.dim, quantType),
    ln_f: nm(weights.ln_f, c.dim),
    wcls: weights.wcls ? wm(weights.wcls, c.vocabSize, c.dim, quantType) : null,
  };
  for (let l = 0; l < c.nLayers; l++) {
    const L = weights[`layer.${l}`];
    w[`layer.${l}`] = {
      ln1: nm(L.ln1, c.dim),
      ln2: nm(L.ln2, c.dim),
      wqkv: wm(L.wqkv, c.qkvOut, c.dim, quantType),
      wo: wm(L.wo, c.dim, c.dim, quantType),
      ffn_gate: wm(L.ffn_gate, c.hidden, c.dim, quantType),
      ffn_up: wm(L.ffn_up, c.hidden, c.dim, quantType),
      ffn_down: wm(L.ffn_down, c.dim, c.hidden, quantType),
    };
  }
  return tr.loadWeights(w);
}

// ── 导出 GGUF ─────────────────────────────────────────────────────────────
/**
 * @param weights  f32 权重对象（见 toTransformer 字段）
 * @param config   Config
 * @param vocab    可选 GGUFVocab；用于写入 tokenizer 段
 * @param quantType 对线性权重采用的量化类型（默认 0=f32）
 */
export function exportGGUF(weights, config, vocab = null, quantType = 0, name = "forge-model") {
  const c = config instanceof Config ? config : new Config(config);
  const meta = new Map();
  meta.set("general.architecture", "forge");
  meta.set("general.name", name);
  meta.set("general.quantization_version", 2);
  meta.set("general.alignment", 32);
  meta.set("forge.block_count", c.nLayers);
  meta.set("forge.embedding_length", c.dim);
  meta.set("forge.feed_forward_length", c.hidden);
  meta.set("forge.attention.head_count", c.nHeads);
  meta.set("forge.attention.head_count_kv", c.nKvHeads);
  meta.set("forge.context_length", c.maxSeq);
  meta.set("forge.attention.layer_norm_rms_epsilon", c.normEps);
  meta.set("forge.vocab_size", c.vocabSize);
  meta.set("forge.rope.freq_base", c.ropeTheta);

  const tensors = [];
  const pushT = (name, mat, N, K) => {
    const type = quantType ?? 0;
    if (type === 0) {
      const f32 = mat instanceof Float32Array ? mat : Float32Array.from(mat);
      tensors.push({ name, dims: [N, K], type: 0, data: new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength) });
    } else {
      const bytes = quantizeRow(type, mat, N * K);
      tensors.push({ name, dims: [N, K], type, data: bytes });
    }
  };
  const pushN = (name, vec, dim) => {
    const f32 = vec instanceof Float32Array ? vec : Float32Array.from(vec);
    tensors.push({ name, dims: [dim], type: 0, data: new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength) });
  };

  pushT(T.emb, weights.tok_emb, c.vocabSize, c.dim);
  pushN(T.lnF, weights.ln_f, c.dim);
  for (let l = 0; l < c.nLayers; l++) {
    const L = weights[`layer.${l}`];
    pushN(T.ln1(l), L.ln1, c.dim);
    pushT(T.wqkv(l), L.wqkv, c.qkvOut, c.dim);
    pushT(T.wo(l), L.wo, c.dim, c.dim);
    pushN(T.ln2(l), L.ln2, c.dim);
    pushT(T.fg(l), L.ffn_gate, c.hidden, c.dim);
    pushT(T.fu(l), L.ffn_up, c.hidden, c.dim);
    pushT(T.fd(l), L.ffn_down, c.dim, c.hidden);
  }
  if (weights.wcls) pushT(T.out, weights.wcls, c.vocabSize, c.dim);

  // 词表段
  if (vocab) {
    meta.set("tokenizer.ggml.model", "llama");
    meta.set("tokenizer.ggml.tokens", vocab.tokenBytes.map((b) => new TextDecoder().decode(b)));
    meta.set("tokenizer.ggml.scores", vocab.scores.map((s) => s));
    meta.set("tokenizer.ggml.token_type", vocab.types);
    meta.set("tokenizer.ggml.bos_token_id", vocab.bos);
    meta.set("tokenizer.ggml.eos_token_id", vocab.eos);
    meta.set("tokenizer.ggml.add_bos_token", vocab.addBos);
    meta.set("tokenizer.ggml.add_eos_token", vocab.addEos);
  }

  const { bytes } = writeGGUF({ meta, tensors });
  return bytes;
}

// ── 加载 GGUF → 可推理模型 ────────────────────────────────────────────────
export function loadGGUF(buf) {
  const g = readGGUF(buf);
  const M = (k, d) => (g.meta.has(k) ? g.meta.get(k) : d);
  const config = new Config({
    dim: M("forge.embedding_length"),
    hidden: M("forge.feed_forward_length"),
    nLayers: M("forge.block_count"),
    nHeads: M("forge.attention.head_count"),
    nKvHeads: M("forge.attention.head_count_kv"),
    vocabSize: M("forge.vocab_size"),
    maxSeq: M("forge.context_length"),
    normEps: M("forge.attention.layer_norm_rms_epsilon"),
    ropeTheta: M("forge.rope.freq_base", 10000),
    bosId: M("tokenizer.ggml.bos_token_id", 1),
    eosId: M("tokenizer.ggml.eos_token_id", 2),
  });

  const gt = (name) => g.getTensor(name);
  const w = {
    tok_emb: tensorToQL(gt(T.emb), config.vocabSize, config.dim),
    ln_f: tensorToF32(gt(T.lnF), config.dim),
    wcls: gt(T.out) ? tensorToQL(gt(T.out), config.vocabSize, config.dim) : null,
  };
  for (let l = 0; l < config.nLayers; l++) {
    w[`layer.${l}`] = {
      ln1: tensorToF32(gt(T.ln1(l)), config.dim),
      ln2: tensorToF32(gt(T.ln2(l)), config.dim),
      wqkv: tensorToQL(gt(T.wqkv(l)), config.qkvOut, config.dim),
      wo: tensorToQL(gt(T.wo(l)), config.dim, config.dim),
      ffn_gate: tensorToQL(gt(T.fg(l)), config.hidden, config.dim),
      ffn_up: tensorToQL(gt(T.fu(l)), config.hidden, config.dim),
      ffn_down: tensorToQL(gt(T.fd(l)), config.dim, config.hidden),
    };
  }

  const vocab = (g.meta.has("tokenizer.ggml.tokens"))
    ? new GGUFVocab(
        g.meta.get("tokenizer.ggml.tokens").map((s) => new TextEncoder().encode(s)),
        g.meta.get("tokenizer.ggml.token_type") ?? g.meta.get("tokenizer.ggml.tokens").map(() => 0),
        g.meta.get("tokenizer.ggml.scores") ?? g.meta.get("tokenizer.ggml.tokens").map(() => 0),
        {
          bos: M("tokenizer.ggml.bos_token_id", 1),
          eos: M("tokenizer.ggml.eos_token_id", 2),
          addBos: M("tokenizer.ggml.add_bos_token", true),
          addEos: M("tokenizer.ggml.add_eos_token", false),
        })
    : null;

  const transformer = new Transformer(config).loadWeights(w);
  return { config, transformer, vocab, metadata: g.meta };
}

function tensorToQL(t, N, K) {
  if (!t) throw new Error("缺失张量");
  // type=0（f32）张量需把字节子视图包成 Float32Array，否则 matvec/_embed 会误读字节
  if (t.type === 0) {
    const f32 = new Float32Array(t.data.buffer, t.data.byteOffset, N * K);
    return new QuantLinear(0, f32, N, K);
  }
  return new QuantLinear(t.type, t.data, N, K);
}
function tensorToF32(t, n) {
  if (!t) throw new Error("缺失张量");
  // 期望 type=0（f32）
  const f = new Float32Array(t.data.buffer, t.data.byteOffset, n);
  return f;
}

// ── 生成循环 ─────────────────────────────────────────────────────────────
/**
 * @param model  loadGGUF 的返回（含 transformer / vocab）
 * @param opts  {prompt, maxTokens, temperature, top_k, top_p, min_p, seed}
 * @returns 生成的文本（已解码）
 */
export function generate(model, opts = {}) {
  const { transformer, vocab, config } = model;
  const prompt = opts.prompt ?? "";
  const seed = opts.seed ?? 0;
  const sampling = {
    temperature: opts.temperature ?? 0.8,
    top_k: opts.top_k ?? 40,
    top_p: opts.top_p ?? 0.95,
    min_p: opts.min_p ?? 0.0,
    seed,
  };
  const rng = new RNG(seed);
  const ids = vocab ? vocab.encode(prompt) : [config.bosId, ...prompt.split("").map((c) => c.charCodeAt(0) % config.vocabSize)];
  const cache = new KVCache(config.nLayers, config.maxSeq, config.kvDim);

  // 上下文上限：prompt 本身不得超长；生成阶段不得越过 maxSeq
  if (ids.length > config.maxSeq) {
    throw new Error(`prompt 长度 ${ids.length} 超过上下文窗口 ${config.maxSeq}`);
  }
  const budget = Math.max(0, Math.min(opts.maxTokens ?? 256, config.maxSeq - ids.length));

  let logits;
  for (let pos = 0; pos < ids.length; pos++) logits = transformer.forward(ids[pos], pos, cache);

  let token = sample(logits, sampling, rng);
  const out = [];
  let nGen = 0;
  while (nGen < budget) {
    if (token === config.eosId) break;
    out.push(token);
    nGen++;
    logits = transformer.forward(token, ids.length + nGen - 1, cache);
    token = sample(logits, sampling, rng);
  }
  return vocab ? vocab.decode(out) : out.map((i) => String.fromCharCode(i)).join("");
}
