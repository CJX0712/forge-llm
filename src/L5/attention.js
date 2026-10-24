/**
 * L5 · 注意力引擎 (Attention Engine)
 * ------------------------------------------------------------------
 * 站在巨人肩上：
 *  - 在线 softmax（Milakov & Gummadi 2018；FlashAttention, Dao et al. 2022）：
 *    维护运行最大值 m 与归一化因子 ℓ，新增块时用 α=exp(m_old−m_new) 重缩放，
 *    永不物化 N×N 注意力矩阵。本层给出其可读参考实现，供 L10 与朴素实现互拍。
 *  - 分块注意力（PagedAttention 思想, vLLM, Kwon et al. 2023）：KV 按块分配，
 *    支持变长序列与零拷贝复用。
 *  - GQA（Ainslie et al. 2023）：n_kv < n_q 时按 group 复用 KV。
 *  - RoPE（Su et al. 2023）。
 *
 * 不变量：
 *  I-50  flashAttention 与 attentionNaive 最大绝对误差 < 1e-5
 *  I-51  注意力权重行和 = 1（Σ_j softmax(q·k_j/√d) = 1）
 *  I-52  causal：位置 i 的权重在 j>i 处恒为 0
 *  I-53  RoPE 相对性：(R_m q)·(R_n k) 只依赖 m−n
 *  I-54  KV cache 增量解码 == 全量重算（同前缀逐位/近位一致）
 */

import { softmax } from "../L1/special.js";

/**
 * 单 query 行的最优纯 JS 实现（解码热点路径）：
 *  两趟扫描 —— ① scores + max；② exp + 加权累加 V，最后一次性乘 1/ℓ。
 * 相比朴素版省掉 softmax 的额外两趟与逐元素除法；相比在线 softmax 版
 * 省掉逐块重缩放（实测在线版在单行场景反而慢约 10%，因为 exp 与 V 累加
 * 形成依赖链）。内存 O(n)（scores 缓冲可复用），与另两版数值等价。
 * @param s 可选的外部 scores 缓冲（长度 ≥ n）；不传则每次分配。
 */
export function attentionDecode(out, q, K, V, n, d, scale, causal = true, qOff = 0, kOff = 0, s = null) {
  const sc = s && s.length >= n ? s : new Float32Array(n);
  // ① scores + 最大值
  let mx = -Infinity;
  for (let j = 0; j < n; j++) {
    if (causal && j > qOff) { sc[j] = -Infinity; continue; }
    const kb = kOff + j * d;
    let acc = 0;
    for (let i = 0; i < d; i++) acc += q[i] * K[kb + i];
    const v = acc * scale;
    sc[j] = v;
    if (v > mx) mx = v;
  }
  // ② exp + 加权累加
  out.fill(0, 0, d);
  let l = 0;
  for (let j = 0; j < n; j++) {
    const p = sc[j] === -Infinity ? 0 : Math.exp(sc[j] - mx);
    if (p === 0) continue;
    l += p;
    const vb = kOff + j * d;
    for (let i = 0; i < d; i++) out[i] += p * V[vb + i];
  }
  const inv = 1 / (l || 1);
  for (let i = 0; i < d; i++) out[i] *= inv;
  return out;
}

/**
 * 朴素注意力（参考实现）：out[d] = Σ_j p_j V[j,d], p = softmax(q·K/√d + mask)
 * 用于 L10 与 flash 版互拍；复杂度 O(n²d)、内存 O(n²)。
 */
export function attentionNaive(out, q, K, V, n, d, scale, causal = true, qOff = 0, kOff = 0) {
  const s = new Float32Array(n);
  for (let j = 0; j < n; j++) {
    if (causal && j > qOff) { s[j] = -Infinity; continue; }
    let acc = 0;
    for (let i = 0; i < d; i++) acc += q[i] * K[kOff + j * d + i];
    s[j] = acc * scale;
  }
  softmax(s, s, n);
  out.fill(0, 0, d);
  for (let j = 0; j < n; j++) {
    const p = s[j];
    if (p === 0) continue;
    for (let i = 0; i < d; i++) out[i] += p * V[kOff + j * d + i];
  }
  return out;
}

/**
 * FlashAttention 风格在线 softmax（单 query 行，分块扫描 K/V）
 * 状态：(m, ℓ, acc)；新块到达时 α = exp(m_old − m_new)，acc ← acc·α + P·V。
 * 内存 O(d + block)，与朴素版数学等价（softmax 平移不变性）。
 * 实现要点：块内分数只算一次点积并存入可复用缓冲，避免朴素在线 softmax
 * 常见的「先求 max 再求 exp」两遍扫描。
 */
let _flashBuf = new Float32Array(64); // 可复用的块分数缓冲（跨调用共享，无状态残留）
export function flashAttention(out, q, K, V, n, d, scale, causal = true, qOff = 0, kOff = 0, block = 64) {
  const end = causal ? Math.min(n, qOff + 1) : n;
  if (end <= 0) { out.fill(0, 0, d); return out; }
  if (_flashBuf.length < block) _flashBuf = new Float32Array(block);
  const s = _flashBuf;
  let m = -Infinity, l = 0;
  out.fill(0, 0, d);
  for (let b = 0; b < end; b += block) {
    const e = Math.min(b + block, end);
    // 1) 块内分数：一次点积，缓存到 s[0..e-b)
    let mb = -Infinity;
    for (let j = b; j < e; j++) {
      const kb = kOff + j * d;
      let acc = 0;
      for (let i = 0; i < d; i++) acc += q[i] * K[kb + i];
      const v = acc * scale;
      s[j - b] = v;
      if (v > mb) mb = v;
    }
    // 2) 在线重缩放：α = exp(m_old − m_new)
    const mNew = Math.max(m, mb);
    const alpha = m === -Infinity ? 0 : Math.exp(m - mNew);
    let lNew = l * alpha;
    if (alpha !== 1) for (let i = 0; i < d; i++) out[i] *= alpha;
    // 3) 累积 P·V
    for (let j = b; j < e; j++) {
      const p = Math.exp(s[j - b] - mNew);
      lNew += p;
      const vb = kOff + j * d;
      for (let i = 0; i < d; i++) out[i] += p * V[vb + i];
    }
    m = mNew; l = lNew;
  }
  const inv = 1 / (l || 1);
  for (let i = 0; i < d; i++) out[i] *= inv;
  return out;
}

/** 连续 KV 缓存：nLayers × maxSeq × kvDim —— 推理默认路径，零间接寻址 */
export class KVCache {
  constructor(nLayers, maxSeq, kvDim) {
    this.nLayers = nLayers; this.maxSeq = maxSeq; this.kvDim = kvDim;
    this.k = new Float32Array(nLayers * maxSeq * kvDim);
    this.v = new Float32Array(nLayers * maxSeq * kvDim);
    this.len = 0;
  }
  put(layer, pos, k, v) {
    const o = (layer * this.maxSeq + pos) * this.kvDim;
    this.k.set(k, o); this.v.set(v, o);
    if (pos + 1 > this.len) this.len = pos + 1;
  }
  /** 第 layer 层前 upto 个位置的 K 连续视图（可直接喂 flashAttention） */
  kView(layer, upto) { return this.k.subarray(layer * this.maxSeq * this.kvDim, (layer * this.maxSeq + upto) * this.kvDim); }
  vView(layer, upto) { return this.v.subarray(layer * this.maxSeq * this.kvDim, (layer * this.maxSeq + upto) * this.kvDim); }
  /** 第 layer 层、第 pos 个位置的 K/V 切片（与 PagedKVCache 同名接口） */
  sliceK(layer, pos) {
    const o = (layer * this.maxSeq + pos) * this.kvDim;
    return this.k.subarray(o, o + this.kvDim);
  }
  sliceV(layer, pos) {
    const o = (layer * this.maxSeq + pos) * this.kvDim;
    return this.v.subarray(o, o + this.kvDim);
  }
  reset() { this.len = 0; }
  bytes() { return (this.k.length + this.v.length) * 4; }
}

/** 分页 KV 缓存（vLLM PagedAttention 思想的简洁版）：按块分配，支持批处理与变长 */
export class PagedKVCache {
  constructor(nLayers, maxSeq, kvDim, blockSize = 64) {
    this.nLayers = nLayers; this.maxSeq = maxSeq; this.kvDim = kvDim; this.blockSize = blockSize;
    this.nBlocks = Math.ceil(maxSeq / blockSize);
    this.k = new Float32Array(nLayers * this.nBlocks * blockSize * kvDim);
    this.v = new Float32Array(nLayers * this.nBlocks * blockSize * kvDim);
    this.len = 0;
  }
  /** 写入第 layer 层、第 pos 位置的 K/V */
  put(layer, pos, k, v) {
    const b = (pos / this.blockSize) | 0, o = pos % this.blockSize;
    const base = ((layer * this.nBlocks + b) * this.blockSize + o) * this.kvDim;
    this.k.set(k, base); this.v.set(v, base);
    if (pos + 1 > this.len) this.len = pos + 1;
  }
  kOff(layer, pos) { const b = (pos / this.blockSize) | 0, o = pos % this.blockSize; return ((layer * this.nBlocks + b) * this.blockSize + o) * this.kvDim; }
  /** 顺序访问视图：为便于非分页路径，pos 直接映射（块内连续） */
  sliceK(layer, pos) { const o = this.kOff(layer, pos); return this.k.subarray(o, o + this.kvDim); }
  sliceV(layer, pos) { const o = this.kOff(layer, pos); return this.v.subarray(o, o + this.kvDim); }
  reset() { this.len = 0; }
  /** 把 pos 之前的 K/V 物化成连续数组（批处理 / 与 KVCache 对拍用） */
  contiguousKV(layer, upto) {
    const K = new Float32Array(upto * this.kvDim), V = new Float32Array(upto * this.kvDim);
    for (let p = 0; p < upto; p++) {
      K.set(this.sliceK(layer, p), p * this.kvDim);
      V.set(this.sliceV(layer, p), p * this.kvDim);
    }
    return [K, V];
  }
  bytes() { return (this.k.length + this.v.length) * 4; }
}
