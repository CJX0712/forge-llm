/**
 * L4 · 神经组件 (Neural Components)
 * ------------------------------------------------------------------
 * 把「权重」抽象成一个可插拔后端：同一份 forward 代码既能跑 f32，也能直接
 * 在 Q4_K/Q5_K/Q6_K/Q8_0/F16 的字节流上跑（llama.cpp 的 ggml_mul_mat 思路）。
 *
 * 组件清单：QuantLinear（量化感知线性层）、RMSNorm、SwiGLU FFN 块、RoPE。
 *
 * 不变量：
 *  I-40  QuantLinear.forward(量化) 与 f32 权重 matvec 的余弦相似度 ≥ 0.999
 *  I-41  SwiGLU 块前向与「分开算 gate/up 再逐元素乘」逐位一致
 */

import { dotQ, dequantize, quantizeRow, rowBytes } from "../L3/quant.js";
import { matvec } from "../L0/numeric.js";
import { rmsnorm as rmsnormF } from "../L1/special.js";

/** 量化感知线性层：W[N,K]，{type, bytes} 或 f32 数组 */
export class QuantLinear {
  constructor(type, bytes, N, K) {
    this.type = type; this.bytes = bytes; this.N = N; this.K = K;
  }
  static fromF32(W, N, K, type = 0) {
    return new QuantLinear(type, type === 0 ? Float32Array.from(W) : quantizeRow(type, W, N * K), N, K);
  }
  /** out[N] = W · x；byteRowOff 用于分片权重（张量并行预留） */
  forward(x, out, xOff = 0) {
    const { type, N, K, bytes } = this;
    if (type === 0) { matvec(out, bytes, x, N, K); return out; }
    const rb = rowBytes(type, K);
    for (let n = 0; n < N; n++) out[n] = dotQ(type, bytes, x, K, xOff, n * rb);
    return out;
  }
  /** 反量化为 f32（验证/导出用） */
  toF32() {
    if (this.type === 0) return Float32Array.from(this.bytes);
    return dequantize(this.type, this.bytes, this.N * this.K);
  }
  bytesSize() { return rowBytes(this.type, this.K) * this.N; }
}

export function rmsnormInto(out, x, w, n, eps = 1e-6) {
  return rmsnormF(out, x, w, n, eps);
}

/**
 * SwiGLU 前馈块（Shazeer 2020）：
 *   out = W_down · (silu(W_gate · x) ⊙ (W_up · x))
 */
export class SwiGLU {
  constructor(gate, up, down, dim, hidden) {
    this.gate = gate; this.up = up; this.down = down;
    this.dim = dim; this.hidden = hidden;
    this._g = new Float32Array(hidden); this._u = new Float32Array(hidden);
  }
  forward(x, out) {
    const { gate, up, down, hidden } = this;
    gate.forward(x, this._g);
    up.forward(x, this._u);
    for (let i = 0; i < hidden; i++) {
      const g = this._g[i];
      this._g[i] = (g / (1 + Math.exp(-g))) * this._u[i];
    }
    down.forward(this._g, out);
    return out;
  }
}
