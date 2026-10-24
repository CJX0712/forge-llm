/**
 * L6b · 可训练 Transformer (Trainable Transformer)
 * ------------------------------------------------------------------
 * 与 L6/transformer.js 的推理路径**结构完全同构**：同样的 RMSNorm → QKV →
 * RoPE → 因果注意力 → SwiGLU FFN 拓扑、同样的权重命名与布局。区别只在
 * 「算子来自 L2 张量级 AD（带反向）而非 L0/L1 的裸函数」。
 *
 * 这样设计的关键价值：
 *  1) 训练与推理共用同一份架构定义，不存在「训练图 ≠ 推理图」的经典陷阱；
 *  2) 训练完的权重可**原样**导出 GGUF，被推理引擎或 llama.cpp 加载；
 *  3) L10 的不变量可交叉验证训练前向与推理前向的一致性（I-70）。
 *
 * 站在巨人肩上：
 *  - 权重初始化：GPT-2 式的 N(0, 0.02) + 残差投影 1/√(2L) 缩放
 *    (GPT-3 / llama2.c 的 rescaled 初始化)。
 *  - 权重衰减与 AdamW 解耦（Loshchilov & Hutter 2019；AdamW, Loshchilov 2018）。
 *
 * 不变量：
 *  I-70  训练前向在 eval 模式下与 L6 推理前向 logits 最大误差 < 1e-3
 *  I-71  loss 随参数更新单调下降（小批次过拟合测试）
 */

import { Config } from "./transformer.js";
import { RNG } from "../L0/numeric.js";
import { param, leaf, op, resetTape, backward, Adam } from "../L2/autograd.js";
import {
  matmulT, embedLookup, rmsNorm, ropeRows, sliceCols, causalAttn, causalAttnBatch,
  swigluRows, crossEntropyMean,
} from "../L2/tensor.js";

// ── 权重初始化 ───────────────────────────────────────────────────────
/** GPT-2 式初始化：正态(0, 0.02)；残差投影额外乘 1/√(2·nLayers) */
export function initWeights(cfg, seed = 1337) {
  const rng = new RNG(seed);
  const std = 0.02;
  const resScale = 1 / Math.sqrt(2 * cfg.nLayers);
  const norm = (n) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = rng.normal() * std; return a; };
  const ones = (n) => new Float32Array(n).fill(1);

  const w = {
    tok_emb: norm(cfg.vocabSize * cfg.dim),
    ln_f: ones(cfg.dim),
    wcls: null,
  };
  for (let l = 0; l < cfg.nLayers; l++) {
    w[`layer.${l}`] = {
      ln1: ones(cfg.dim),
      ln2: ones(cfg.dim),
      wqkv: norm(cfg.qkvOut * cfg.dim),
      wo: (() => { const a = norm(cfg.dim * cfg.dim); for (let i = 0; i < a.length; i++) a[i] *= resScale; return a; })(),
      ffn_gate: norm(cfg.hidden * cfg.dim),
      ffn_up: norm(cfg.hidden * cfg.dim),
      ffn_down: (() => { const a = norm(cfg.dim * cfg.hidden); for (let i = 0; i < a.length; i++) a[i] *= resScale; return a; })(),
    };
  }
  return w;
}

/** 把 f32 权重对象转为 AD 参数节点表 */
export function toParams(w, cfg) {
  const P = [];
  const mk = (arr) => { const p = param(arr); P.push(p); return p; };
  const ps = {
    tok_emb: mk(w.tok_emb),
    ln_f: mk(w.ln_f),
    wcls: w.wcls ? mk(w.wcls) : null,
    layers: [],
  };
  for (let l = 0; l < cfg.nLayers; l++) {
    const L = w[`layer.${l}`];
    ps.layers.push({
      ln1: mk(L.ln1), ln2: mk(L.ln2),
      wqkv: mk(L.wqkv), wo: mk(L.wo),
      ffn_gate: mk(L.ffn_gate), ffn_up: mk(L.ffn_up), ffn_down: mk(L.ffn_down),
    });
  }
  return { ps, all: P };
}

/**
 * 可训练 Transformer
 */
export class TrainableTransformer {
  constructor(config, weights = null, seed = 1337) {
    this.cfg = config instanceof Config ? config : new Config(config);
    const c = this.cfg;
    const w = weights ?? initWeights(c, seed);
    const { ps, all } = toParams(w, c);
    this.params = ps;
    this.paramList = all;
    this.w = w;
    this._build();
  }

  _build() {
    const c = this.cfg, ps = this.params;
    this.wclsNode = ps.wcls ?? ps.tok_emb; // 权重共享（GPT-2 风格）
  }

  /**
   * 前向：ids[T] → logits[T, V]（不做 loss）
   * @param ids  token id 序列（长度 T）
   * @param pos0 位置起点（默认 0）
   */
  forward(ids, pos0 = 0) {
    return this._forward(ids, 1, ids.length, pos0);
  }

  /**
   * 批量前向：ids[B][T] → logits[B, T, V]
   * 使用块对角因果掩码 —— 样本之间互不可见（与 HF 实现的 4D attn_mask 等价）。
   */
  forwardBatch(batchIds) {
    const B = batchIds.length, T = batchIds[0].length;
    const flat = [];
    for (const row of batchIds) for (const id of row) flat.push(id);
    return this._forward(flat, B, T, 0);
  }

  _forward(flatIds, B, T, pos0) {
    const c = this.cfg, ps = this.params;
    const qD = c.nHeads * c.dHead, kvD = c.nKvHeads * c.dHead;

    // 1) 词嵌入
    let x = embedLookup(ps.tok_emb, flatIds, c.vocabSize, c.dim);

    for (let l = 0; l < c.nLayers; l++) {
      const L = ps.layers[l];
      // 2) 注意力前 RMSNorm
      const xn = rmsNorm(x, L.ln1, B * T, c.dim, c.normEps);
      // 3) QKV 投影 → 切三段
      const qkv = matmulT(xn, L.wqkv, B * T, c.dim, c.qkvOut);
      const q = sliceCols(qkv, B * T, c.qkvOut, 0, qD);
      const k = sliceCols(qkv, B * T, c.qkvOut, qD, kvD);
      const v = sliceCols(qkv, B * T, c.qkvOut, qD + kvD, kvD);
      // 4) RoPE（位置在样本内重新计数）
      const qr = ropeRows(q, B * T, c.nHeads, c.dHead, pos0, c.ropeTheta, T);
      const kr = ropeRows(k, B * T, c.nKvHeads, c.dHead, pos0, c.ropeTheta, T);
      // 5) 因果注意力（批量：块对角掩码）
      const att = B === 1
        ? causalAttn(qr, kr, v, T, c.nHeads, c.nKvHeads, c.dHead)
        : causalAttnBatch(qr, kr, v, B, T, c.nHeads, c.nKvHeads, c.dHead);
      // 6) 输出投影 + 残差
      const ao = matmulT(att, L.wo, B * T, qD, c.dim);
      x = addTensors(x, ao, B * T * c.dim);
      // 7) FFN 前 RMSNorm
      const xn2 = rmsNorm(x, L.ln2, B * T, c.dim, c.normEps);
      // 8) SwiGLU FFN
      const g = matmulT(xn2, L.ffn_gate, B * T, c.dim, c.hidden);
      const u = matmulT(xn2, L.ffn_up, B * T, c.dim, c.hidden);
      const act = swigluRows(g, u, B * T, c.hidden);
      const ff = matmulT(act, L.ffn_down, B * T, c.hidden, c.dim);
      x = addTensors(x, ff, B * T * c.dim);
    }
    // 9) 最终归一化 + LM 头
    const xf = rmsNorm(x, ps.ln_f, B * T, c.dim, c.normEps);
    const logits = matmulT(xf, this.wclsNode, B * T, c.dim, c.vocabSize);
    return logits;
  }

  /**
   * 前向 + 交叉熵损失（next-token 预测，teacher forcing）
   * @param ids     输入序列（长度 T+1）
   * @param targets 目标序列（长度 T，通常为 ids.slice(1)）
   */
  loss(ids, targets) {
    // 输入取前 T 个 token，第 i 行 logits 预测 targets[i]
    const T = targets.length;
    const logits = this.forward(ids.slice(0, T));
    return crossEntropyMean(logits, targets, T, this.cfg.vocabSize);
  }

  /**
   * 批量前向 + 交叉熵损失（块对角掩码，样本互不可见）
   * @param batchIds  Array<B> of Array<T+1>
   */
  lossBatch(batchIds) {
    const B = batchIds.length, T = batchIds[0].length - 1;
    const flatIds = [], flatTgt = [];
    for (const row of batchIds) {
      for (let i = 0; i < T; i++) { flatIds.push(row[i]); flatTgt.push(row[i + 1]); }
    }
    const logits = this._forward(flatIds, B, T, 0);
    return crossEntropyMean(logits, flatTgt, B * T, this.cfg.vocabSize);
  }

  /** 反向传播并返回 loss 值 */
  backward(lossNode) {
    backward(lossNode);
    return lossNode.data[0];
  }

  /** 清空所有梯度（每个 batch 前调用） */
  zeroGrad() {
    for (const p of this.paramList) p.zeroGrad();
  }

  /** 导出为 f32 权重对象（可直接喂给 exportGGUF） */
  exportWeights() {
    const c = this.cfg;
    const out = {
      tok_emb: Float32Array.from(this.params.tok_emb.data),
      ln_f: Float32Array.from(this.params.ln_f.data),
      wcls: this.params.wcls ? Float32Array.from(this.params.wcls.data) : null,
    };
    for (let l = 0; l < c.nLayers; l++) {
      const L = this.params.layers[l];
      out[`layer.${l}`] = {
        ln1: Float32Array.from(L.ln1.data),
        ln2: Float32Array.from(L.ln2.data),
        wqkv: Float32Array.from(L.wqkv.data),
        wo: Float32Array.from(L.wo.data),
        ffn_gate: Float32Array.from(L.ffn_gate.data),
        ffn_up: Float32Array.from(L.ffn_up.data),
        ffn_down: Float32Array.from(L.ffn_down.data),
      };
    }
    return out;
  }

  /** 参数量 */
  countParams() {
    let n = 0;
    for (const p of this.paramList) n += p.data.length;
    return n;
  }
}

// 逐元素加（自实现，避免依赖 autograd.js 的标量版 add 的额外语义）
function addTensors(a, b, n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = a.data[i] + b.data[i];
  return op(out, (g) => {
    const ga = (a.grad ??= new Float32Array(n));
    const gb = (b.grad ??= new Float32Array(n));
    for (let i = 0; i < n; i++) { ga[i] += g[i]; gb[i] += g[i]; }
  });
}

/**
 * 训练循环（教学/演示用；生产训练请用 PyTorch/llama.cpp）
 * @param model   TrainableTransformer
 * @param data    扁平 token id 数组
 * @param opts    {steps, batchSize, lr, seqLen, logEvery, seed}
 */
export function train(model, data, opts = {}) {
  const steps = opts.steps ?? 100;
  const B = opts.batchSize ?? 8;
  const seqLen = opts.seqLen ?? 32;
  const lr = opts.lr ?? 1e-2;
  const logEvery = opts.logEvery ?? 10;
  const rng = new RNG(opts.seed ?? 7);

  const opt = new Adam(model.paramList, { lr, totalSteps: steps, clip: 1.0, wd: 0.01 });
  const history = [];
  for (let step = 0; step < steps; step++) {
    // 采样一个批次（每个样本独立切窗，样本间由块对角掩码隔离）
    const batch = [];
    for (let b = 0; b < B; b++) {
      const s = rng.int(Math.max(1, data.length - seqLen - 1));
      const row = new Array(seqLen + 1);
      for (let i = 0; i <= seqLen; i++) row[i] = data[s + i] ?? data[data.length - 1];
      batch.push(row);
    }
    resetTape();
    model.zeroGrad();
    const lossNode = model.lossBatch(batch);
    const l = model.backward(lossNode);
    const info = opt.step();
    if (step % logEvery === 0 || step === steps - 1) {
      history.push({ step, loss: l, lr: info.lr, gnorm: info.gnorm });
      console.log(`  step ${String(step).padStart(4)}  loss ${l.toFixed(4)}  lr ${info.lr.toExponential(2)}  |g| ${info.gnorm.toFixed(3)}`);
    }
  }
  return history;
}
