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
/**
 * 张量命名 —— 与 llama.cpp 官方 GGUF 完全一致（blk.{l}.{name}.weight）。
 * 注：官方把 Q/K/V 存成三个独立张量（attn_q/attn_k/attn_v），而内核内部
 * 用融合的 wqkv 一次投影（少两次 GEMM 调用开销）。导出时拆开、加载时拼回。
 */
const T = {
  emb: "token_embd.weight",
  lnF: "output_norm.weight",
  ln1: (l) => `blk.${l}.attn_norm.weight`,
  wq: (l) => `blk.${l}.attn_q.weight`,
  wk: (l) => `blk.${l}.attn_k.weight`,
  wv: (l) => `blk.${l}.attn_v.weight`,
  wqkv: (l) => `blk.${l}.attn_qkv.weight`, // 非官方别名：加载旧文件时兼容
  wo: (l) => `blk.${l}.attn_output.weight`,
  ln2: (l) => `blk.${l}.ffn_norm.weight`,
  fg: (l) => `blk.${l}.ffn_gate.weight`,
  fu: (l) => `blk.${l}.ffn_up.weight`,
  fd: (l) => `blk.${l}.ffn_down.weight`,
  out: "output.weight",
};

/**
 * 官方 GGUF 的 dims 是**反转**的：numpy shape (N,K) → GGUF dims [K,N]
 * （ggml 的 ne[] 按列优先语义）。实测：官方 gguf-py 写 (3,5) 得 dims [5,3]。
 * 所以逻辑 [N(out),K(in)] 与 GGUF dims 的换算只有这一处，读写共用。
 */
const toGGUFDims = (N, K) => [K, N];
const fromGGUFDims = (dims) => (dims.length >= 2 ? [dims[1], dims[0]] : [dims[0], 1]);

/** 从（可能已量化的）矩阵字节里切出 [rowStart, rowStart+rowCount) 行 */
function sliceRows(type, data, K, rowStart, rowCount) {
  const rb = rowBytes(type, K);
  return data.subarray(rowStart * rb, (rowStart + rowCount) * rb);
}

/** 把 q/k/v 三个张量按行拼成融合的 wqkv 字节（要求三者 type 与 K 相同） */
function concatRows(parts, K) {
  const types = new Set(parts.map((p) => p.type));
  if (types.size !== 1) throw new Error(`Q/K/V 量化类型不一致: ${[...types]}`);
  const type = parts[0].type;
  const rb = rowBytes(type, K);
  let total = 0;
  for (const p of parts) total += p.data.byteLength;
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) { out.set(p.data, o); o += p.data.byteLength; }
  return { type, data: out, rows: total / rb };
}

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
/**
 * 导出为标准 llama 架构 GGUF —— 与 llama.cpp 官方命名/键名/维度约定一致，
 * 可被 llama.cpp、ollama、LM Studio 等生态直接加载。
 * @param quantType 0=f32，8=Q8_0，12=Q4_K，14=Q6_K …
 */
export function exportGGUF(weights, config, vocab = null, quantType = 0, name = "forge-model") {
  const c = config instanceof Config ? config : new Config(config);
  const A = "llama"; // 官方架构名
  const meta = new Map();
  meta.set("general.architecture", A);
  meta.set("general.name", name);
  meta.set("general.quantization_version", 2);
  meta.set("general.alignment", 32);
  meta.set(`${A}.block_count`, c.nLayers);
  meta.set(`${A}.embedding_length`, c.dim);
  meta.set(`${A}.feed_forward_length`, c.hidden);
  meta.set(`${A}.attention.head_count`, c.nHeads);
  meta.set(`${A}.attention.head_count_kv`, c.nKvHeads);
  meta.set(`${A}.context_length`, c.maxSeq);
  meta.set(`${A}.attention.layer_norm_rms_epsilon`, c.normEps);
  meta.set(`${A}.vocab_size`, c.vocabSize);
  meta.set(`${A}.rope.freq_base`, c.ropeTheta);
  meta.set("general.file_type", quantType ?? 0);

  const tensors = [];
  const pushT = (name, mat, N, K) => {
    const type = quantType ?? 0;
    if (type === 0) {
      const f32 = mat instanceof Float32Array ? mat : Float32Array.from(mat);
      tensors.push({
        name, dims: toGGUFDims(N, K), type: 0,
        data: new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength),
      });
    } else {
      // 块量化要求行长是 blockSize 的整数倍，否则该张量降级为 f32
      const t = info(type);
      if (K % t.blockSize === 0) {
        tensors.push({ name, dims: toGGUFDims(N, K), type, data: quantizeRow(type, mat, N * K) });
      } else {
        const f32 = mat instanceof Float32Array ? mat : Float32Array.from(mat);
        tensors.push({ name, dims: toGGUFDims(N, K), type: 0, data: new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength) });
      }
    }
  };
  const pushN = (name, vec, dim) => {
    const f32 = vec instanceof Float32Array ? vec : Float32Array.from(vec);
    tensors.push({ name, dims: [dim], type: 0, data: new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength) });
  };

  pushT(T.emb, weights.tok_emb, c.vocabSize, c.dim);
  pushN(T.lnF, weights.ln_f, c.dim);
  const qD = c.nHeads * c.dHead, kvD = c.nKvHeads * c.dHead;
  for (let l = 0; l < c.nLayers; l++) {
    const L = weights[`layer.${l}`];
    pushN(T.ln1(l), L.ln1, c.dim);
    // 内部融合 wqkv → 官方三个独立张量（按行切片，量化数据零反量化）
    const type = quantType ?? 0;
    const wqkv = L.wqkv instanceof Float32Array ? L.wqkv : Float32Array.from(L.wqkv);
    if (type === 0) {
      pushT(T.wq(l), wqkv.subarray(0, qD * c.dim), qD, c.dim);
      pushT(T.wk(l), wqkv.subarray(qD * c.dim, (qD + kvD) * c.dim), kvD, c.dim);
      pushT(T.wv(l), wqkv.subarray((qD + kvD) * c.dim, (qD + 2 * kvD) * c.dim), kvD, c.dim);
    } else {
      const t = info(type);
      if (c.dim % t.blockSize === 0) {
        const bytes = quantizeRow(type, wqkv, c.qkvOut * c.dim);
        const push = (name, rowStart, rows) => tensors.push({
          name, dims: toGGUFDims(rows, c.dim), type, data: sliceRows(type, bytes, c.dim, rowStart, rows),
        });
        push(T.wq(l), 0, qD);
        push(T.wk(l), qD, kvD);
        push(T.wv(l), qD + kvD, kvD);
      } else {
        pushT(T.wq(l), wqkv.subarray(0, qD * c.dim), qD, c.dim);
        pushT(T.wk(l), wqkv.subarray(qD * c.dim, (qD + kvD) * c.dim), kvD, c.dim);
        pushT(T.wv(l), wqkv.subarray((qD + kvD) * c.dim, (qD + 2 * kvD) * c.dim), kvD, c.dim);
      }
    }
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
/**
 * 加载 GGUF —— 按 llama.cpp 官方规范解析。
 * 架构前缀取自 `general.architecture`（llama / qwen2 / gemma / mistral…），
 * 元数据键为 `<arch>.*`，张量名为 `blk.{l}.*`；Q/K/V 三个独立张量会被
 * 按行拼回内核内部的融合 wqkv。
 */
export function loadGGUF(buf) {
  const g = readGGUF(buf);
  const M = (k, d) => (g.meta.has(k) ? g.meta.get(k) : d);
  const A = M("general.architecture", "llama");
  const K = (suffix) => `${A}.${suffix}`;
  const config = new Config({
    dim: M(K("embedding_length")),
    hidden: M(K("feed_forward_length")),
    nLayers: M(K("block_count")),
    nHeads: M(K("attention.head_count")),
    nKvHeads: M(K("attention.head_count_kv"), M(K("attention.head_count"))),
    vocabSize: M(K("vocab_size")),
    maxSeq: M(K("context_length"), 512),
    normEps: M(K("attention.layer_norm_rms_epsilon"), 1e-5),
    ropeTheta: M(K("rope.freq_base"), 10000),
    bosId: M("tokenizer.ggml.bos_token_id", 1),
    eosId: M("tokenizer.ggml.eos_token_id", 2),
  });
  const c = config;

  const gt = (name) => g.getTensor(name);
  /** 取矩阵张量：GGUF dims 是反转的，用 fromGGUFDims 还原 [N,K] */
  const matQL = (name) => {
    const t = gt(name);
    if (!t) throw new Error(`缺失张量: ${name}`);
    const [N, K2] = fromGGUFDims(t.dims);
    return tensorToQL(t, N, K2);
  };
  // Q/K/V：官方三张量 → 融合 wqkv（字节级按行拼接，量化数据不反量化）
  const fuseQKV = (l) => {
    const one = gt(T.wqkv(l));
    if (one) return matQL(T.wqkv(l));
    const q = gt(T.wq(l)), k = gt(T.wk(l)), v = gt(T.wv(l));
    if (!q || !k || !v) throw new Error(`第 ${l} 层缺少 attn_q/k/v`);
    const [, KQ] = fromGGUFDims(q.dims);
    const fused = concatRows([q, k, v], KQ);
    return tensorToQL(fused, c.qkvOut, KQ);
  };

  const w = {
    tok_emb: matQL(T.emb),
    ln_f: tensorToF32(gt(T.lnF), c.dim),
    wcls: gt(T.out) ? matQL(T.out) : null,
  };
  for (let l = 0; l < c.nLayers; l++) {
    w[`layer.${l}`] = {
      ln1: tensorToF32(gt(T.ln1(l)), c.dim),
      ln2: tensorToF32(gt(T.ln2(l)), c.dim),
      wqkv: fuseQKV(l),
      wo: matQL(T.wo(l)),
      ffn_gate: matQL(T.fg(l)),
      ffn_up: matQL(T.fu(l)),
      ffn_down: matQL(T.fd(l)),
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
