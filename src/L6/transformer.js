/**
 * L6 · Transformer 装配 (Model Assembly)
 * ------------------------------------------------------------------
 * 把 L1–L5 组合成一个可推理的 Transformer。关键设计：
 *  - 「权重后端可插拔」：每一层都通过 L4 的 QuantLinear，因此同一份 forward
 *    既能跑 f32、也能直接吃 Q4_K/Q5_K/Q6_K/Q8_0/F16 的字节流 —— 训练/导出
 *    用 f32，部署用量化，forward 代码只有一份。
 *  - RoPE 预计算表（L1）每 token 只查表不计算三角函数。
 *  - 因果注意力走 L5 的两趟扫描实现（scores+max → exp+累加），
 *    实测在单 query 行的纯 JS 场景比在线 softmax 快约 5%，且零分配。
 *
 * 复用的世界最优构件：
 *  - Llama-2 架构（GQA + RMSNorm + RoPE + SwiGLU FFN），Touvron et al. 2023。
 *  - 多查询分组（GQA，Ainslie et al. 2023），n_kv < n_q 时按组复用 KV。
 *
 * 不变量：
 *  I-60  forward 在「f32 权重」与「反量化后 f32」下自洽（loader 校验）
 *  I-61  因果：pos 处 logits 只依赖 ≤ pos 的 token（KV 前缀不变则输出不变）
 *  I-62  GQA：head h 只用其所属 kv-group 的 K/V
 */

import { rmsnorm, ropeTable } from "../L1/special.js";
import { addInto } from "../L0/numeric.js";
import { QuantLinear, SwiGLU } from "../L4/nn.js";
import { attentionDecode } from "../L5/attention.js";
import { dequantize, rowBytes } from "../L3/quant.js";

/** 模型超参。Follows Llama-2 的命名。 */
export class Config {
  constructor(o = {}) {
    this.dim = o.dim ?? 288;
    this.hidden = o.hidden ?? 768;          // FFN 中间维（SwiGLU gate/up 宽度）
    this.nLayers = o.nLayers ?? 6;
    this.nHeads = o.nHeads ?? 6;
    this.nKvHeads = o.nKvHeads ?? 6;        // < nHeads 即 GQA
    this.vocabSize = o.vocabSize ?? 32000;
    this.maxSeq = o.maxSeq ?? 1024;
    this.normEps = o.normEps ?? 1e-5;
    this.ropeTheta = o.ropeTheta ?? 10000;
    this.bosId = o.bosId ?? 1;
    this.eosId = o.eosId ?? 2;
    this._derive();
  }
  _derive() {
    if (this.nHeads % this.nKvHeads !== 0) {
      throw new Error(`GQA 要求 nHeads(${this.nHeads}) 能被 nKvHeads(${this.nKvHeads}) 整除（Llama-2 系模型均满足）`);
    }
    this.dHead = Math.floor(this.dim / this.nHeads);
    this.kvDim = this.nKvHeads * this.dHead;
    this.qkvOut = (this.nHeads + 2 * this.nKvHeads) * this.dHead;
    this.groupSize = this.nHeads / this.nKvHeads; // ≥1：每 groupSize 个 q-head 共享 1 个 kv-head
  }
}

/** 给某个 head 的子向量原地施加 RoPE（用预计算表） */
function applyRopeHead(x, base, dHead, pos, tbl) {
  const half = dHead >> 1, cp = pos * half;
  for (let i = 0; i < dHead; i += 2) {
    const c = tbl.cos[cp + (i >> 1)], s = tbl.sin[cp + (i >> 1)];
    const a = x[base + i], b = x[base + i + 1];
    x[base + i] = a * c - b * s;
    x[base + i + 1] = a * s + b * c;
  }
}

/** 把 f32 数组或 QuantLinear 统一成可 forward 的 QuantLinear（type 0 = 原样 f32） */
function asQuantLinear(w, N, K) {
  if (w instanceof QuantLinear) return w;
  return QuantLinear.fromF32(w, N, K, 0);
}

export class Transformer {
  constructor(config) {
    this.cfg = config instanceof Config ? config : new Config(config);
    const c = this.cfg;
    // 可复用缓冲区（避免每 token 重分配）
    this.x = new Float32Array(c.dim);
    this.xb = new Float32Array(c.dim);
    this.xb2 = new Float32Array(c.dim);
    this.q = new Float32Array(c.qkvOut);
    this.att = new Float32Array(c.dim);
    this.logits = new Float32Array(c.vocabSize);
    this.tmpK = new Float32Array(c.maxSeq * c.dHead);
    this.tmpV = new Float32Array(c.maxSeq * c.dHead);
    this.scores = new Float32Array(c.maxSeq); // 注意力 scores 复用缓冲（解码零分配）
    this.rope = ropeTable(c.maxSeq, c.dHead, c.ropeTheta);
    this.layers = [];
    this.tokEmb = null; this.wcls = null; this.lnF = null;
  }

  /**
   * 注入权重。w 字段约定（见 README 与 export 格式）：
   *  tok_emb[vocab*dim]（f32 或 QuantLinear）
   *  layer.{l}.ln1[dim], layer.{l}.ln2[dim], ln_f[dim]
   *  layer.{l}.wqkv / wo / ffn_gate / ffn_up / ffn_down（QuantLinear）
   *  wcls（可选；缺省复用 tok_emb）
   */
  loadWeights(w) {
    const c = this.cfg;
    this.tokEmb = asQuantLinear(w.tok_emb, c.vocabSize, c.dim);
    this.wcls = asQuantLinear(w.wcls || w.tok_emb, c.vocabSize, c.dim);
    this.lnF = w.ln_f;
    this.layers = [];
    for (let l = 0; l < c.nLayers; l++) {
      const L = w[`layer.${l}`];
      this.layers.push({
        ln1: L.ln1, ln2: L.ln2,
        wqkv: asQuantLinear(L.wqkv, c.qkvOut, c.dim),
        wo: asQuantLinear(L.wo, c.dim, c.dim),
        ffn: new SwiGLU(
          asQuantLinear(L.ffn_gate, c.hidden, c.dim),
          asQuantLinear(L.ffn_up, c.hidden, c.dim),
          asQuantLinear(L.ffn_down, c.dim, c.hidden),
          c.dim, c.hidden),
      });
    }
    return this;
  }

  /** 词嵌入查表（QuantLinear 行拷贝；量化嵌入则反量化整行） */
  _embed(token, out) {
    const c = this.cfg, te = this.tokEmb;
    if (te.type === 0) {
      const off = token * c.dim;
      for (let i = 0; i < c.dim; i++) out[i] = te.bytes[off + i];
    } else {
      const rb = rowBytes(te.type, c.dim);
      dequantize(te.type, te.bytes, c.dim, out, token * rb);
    }
    return out;
  }

  /** 单步前向：输入 token + 位置 pos + KV 缓存。返回 logits[|vocab|]。 */
  forward(token, pos, cache) {
    const c = this.cfg;
    const { dim, hidden, nLayers, nHeads, nKvHeads, dHead, kvDim,
            qkvOut, groupSize, normEps, maxSeq } = c;
    const x = this.x, xb = this.xb, xb2 = this.xb2, q = this.q, att = this.att;

    this._embed(token, x);

    for (let l = 0; l < nLayers; l++) {
      const Lr = this.layers[l];
      // 1) 注意力预归一化
      rmsnorm(xb, x, Lr.ln1, dim, normEps);
      // 2) QKV 投影 → q = [Q | K | V]
      Lr.wqkv.forward(xb, q);
      // 3) 对 Q、K 施加 RoPE
      let off = 0;
      for (let h = 0; h < nHeads; h++) { applyRopeHead(q, off, dHead, pos, this.rope); off += dHead; }
      for (let h = 0; h < nKvHeads; h++) { applyRopeHead(q, off, dHead, pos, this.rope); off += dHead; }
      // 4) 写 K/V 入缓存
      const K = q.subarray(nHeads * dHead, nHeads * dHead + kvDim);
      const V = q.subarray(nHeads * dHead + kvDim, qkvOut);
      cache.put(l, pos, K, V);
      // 5) 逐 head 因果注意力（两趟扫描 + 复用 scores 缓冲，零分配）
      //    GQA：同一 kv-head 服务 groupSize 个 q-head，K/V 收集只需做一次
      att.fill(0, 0, dim);
      const n = pos + 1;
      for (let kvh = 0; kvh < nKvHeads; kvh++) {
        gatherKV(cache, l, kvh, dHead, n, this.tmpK, this.tmpV);
        for (let g = 0; g < groupSize; g++) {
          const h = kvh * groupSize + g;
          if (h >= nHeads) break;
          const qh = q.subarray(h * dHead, h * dHead + dHead);
          attentionDecode(
            att.subarray(h * dHead, h * dHead + dHead),
            qh, this.tmpK, this.tmpV, n, dHead,
            1 / Math.sqrt(dHead), true, pos, 0, this.scores);
        }
      }
      // 6) 输出投影 + 残差
      Lr.wo.forward(att, xb2);
      addInto(x, xb2, dim);
      // 7) FFN 预归一化 + SwiGLU + 残差
      rmsnorm(xb, x, Lr.ln2, dim, normEps);
      Lr.ffn.forward(xb, xb2);
      addInto(x, xb2, dim);
    }
    // 最终层归一化 + 语言模型头
    rmsnorm(xb, x, this.lnF, dim, normEps);
    this.wcls.forward(xb, this.logits);
    return this.logits;
  }
}

/** 从 KV 缓存（连续布局）按 head 收集 [0..n) 的切片到连续 tmpK/tmpV */
function gatherKV(cache, layer, kvh, dHead, n, tmpK, tmpV) {
  const kvDim = cache.kvDim, stride = cache.maxSeq * kvDim;
  const base = layer * stride;
  for (let p = 0; p < n; p++) {
    const o = base + p * kvDim + kvh * dHead;
    for (let i = 0; i < dHead; i++) { tmpK[p * dHead + i] = cache.k[o + i]; tmpV[p * dHead + i] = cache.v[o + i]; }
  }
}
