/**
 * L0 · 数值内核 (Numeric Kernel)
 * ------------------------------------------------------------------
 * 定位：整个内核的唯一算术地基。所有层只准通过 L0 触碰内存与浮点。
 *
 * 站在巨人肩上：
 *  - GEMM：4 路循环展开 + 行主序顺序读。缓存分块（BLAS/ggml 在 C 下的通用
 *    做法）在本机 V8 上实测更慢（见 gemm 文档注释与 PERFORMANCE.md B1），
 *    故不采用 —— 选型由数据决定，不由惯例决定。
 *  - 行主序 [N(out), K(in)] 权重布局：解码期 matvec 顺序访问，与
 *    llama.cpp 的 ggml_compute_forward_mul_mat 内存访问模式一致。
 *  - xorshift128+ 确定性 RNG：Vigna 2017，可复现的必要前提。
 *
 * 不变量（L10 会逐条验证）：
 *  I-00  gemm 与朴素三重循环在 f32 下相对误差 < 1e-5
 *  I-01  matvec 与 gemm(M=1) 逐位一致
 *  I-02  量化点积 dotQ 与「反量化后再点积」误差 ≤ 该格式的解析误差界
 */

// ── 基础内存 ─────────────────────────────────────────────────────────
export const f32 = (n) => new Float32Array(n);
export const zeros = (n) => new Float32Array(n);
export function fill(a, v) { a.fill(v); return a; }
export function copy(a) { return new Float32Array(a); }

/** 确定性 RNG · xorshift128+ (Vigna 2017) —— 同种子必同序列，benchmark 才可复现 */
export class RNG {
  constructor(seed = 0x9e3779b9) {
    let s = seed >>> 0 || 1;
    // splitmix32 扩散，避免低位种子相关性
    const nx = () => { s = (s + 0x9e3779b9) >>> 0; let z = s; z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0; z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0; return (z ^ (z >>> 15)) >>> 0; };
    this.s0 = nx(); this.s1 = nx();
  }
  next() { // xorshift128+
    let x = this.s0, y = this.s1;
    this.s0 = y;
    x ^= (x << 23) >>> 0; x >>>= 0;
    x ^= x >>> 17; x ^= y; x ^= (y >>> 26);
    this.s1 = x >>> 0;
    return ((this.s1 + y) >>> 0) / 4294967296;
  }
  /** Box-Muller 标准正态 */
  normal() {
    const u = Math.max(this.next(), 1e-12), v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  uniform(a, b) { return a + (b - a) * this.next(); }
  int(n) { return Math.floor(this.next() * n); }
}

// ── 向量基元 ─────────────────────────────────────────────────────────
export function dot(a, b, n) {
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0, i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3];
  }
  let s = s0 + s1 + s2 + s3;
  for (; i < n; i++) s += a[i] * b[i];
  return s;
}

export function axpy(o, x, a, n) { for (let i = 0; i < n; i++) o[i] += a * x[i]; return o; }
export function addInto(o, x, n) { for (let i = 0; i < n; i++) o[i] += x[i]; return o; }
export function scale(o, a, n) { for (let i = 0; i < n; i++) o[i] *= a; return o; }

export function argmax(a, n = a.length, stride = 1, off = 0) {
  let bi = 0, bv = -Infinity;
  for (let i = 0; i < n; i++) { const v = a[off + i * stride]; if (v > bv) { bv = v; bi = i; } }
  return bi;
}

/** RMS 范数：LLaMA 系归一化基石 ‖x‖₂/√n */
export function rms(x, n = x.length, off = 0) {
  let s = 0;
  for (let i = 0; i < n; i++) { const v = x[off + i]; s += v * v; }
  return Math.sqrt(s / n);
}

// ── 矩阵乘 ───────────────────────────────────────────────────────────
/**
 * matvec: out[N] = W[N,K] · x[K]  —— 解码热点（每 token 走 2N×K 次 FMA）
 * W 行主序，N=输出维，K=输入维；内层顺序读 W 与 x，cache 友好。
 */
export function matvec(out, W, x, N, K) {
  for (let n = 0; n < N; n++) {
    const base = n * K;
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0, k = 0;
    for (; k + 4 <= K; k += 4) {
      s0 += W[base + k] * x[k];
      s1 += W[base + k + 1] * x[k + 1];
      s2 += W[base + k + 2] * x[k + 2];
      s3 += W[base + k + 3] * x[k + 3];
    }
    let s = s0 + s1 + s2 + s3;
    for (; k < K; k++) s += W[base + k] * x[k];
    out[n] = s;
  }
  return out;
}

/**
 * gemm: C[M,N] = A[M,K] · Bᵀ  (B 行主序 [N,K]) —— prefill / 训练热点
 *
 * 选型依据（实测，见 PERFORMANCE.md B1）：在本机 Node/V8 上对 512×1024×1024
 * 做过交错基准 —— 朴素三重循环 + 4 路展开 433.9 ms，而 BK/BN 缓存分块的三个
 * 变体（32/32、64/64、128/32）分别为 512/508/531 ms。即 JS 下缓存分块不划算
 * （JIT 生成的代码与 C 的访存模式不同），故采用最快且最简的形式：4 路展开 +
 * 行主序顺序读。与 matvec 同一展开结构，M=1 时逐位一致（I-01）。
 */
export function gemm(C, A, B, M, N, K) {
  for (let m = 0; m < M; m++) {
    const ab = m * K, cb = m * N;
    for (let n = 0; n < N; n++) {
      const bb = n * K;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0, k = 0;
      for (; k + 4 <= K; k += 4) {
        s0 += A[ab + k] * B[bb + k];
        s1 += A[ab + k + 1] * B[bb + k + 1];
        s2 += A[ab + k + 2] * B[bb + k + 2];
        s3 += A[ab + k + 3] * B[bb + k + 3];
      }
      let s = s0 + s1 + s2 + s3;
      for (; k < K; k++) s += A[ab + k] * B[bb + k];
      C[cb + n] = s;
    }
  }
  return C;
}

/** 朴素参考 GEMM —— 仅供 L10 交叉验证，生产路径不调用 */
export function gemmRef(C, A, B, M, N, K) {
  for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
    let s = 0; const ab = m * K, bb = n * K;
    for (let k = 0; k < K; k++) s += A[ab + k] * B[bb + k];
    C[m * N + n] = s;
  }
  return C;
}

// ── 误差度量（L3 量化与 L10 验证共用）────────────────────────────────
export function maxAbsErr(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > m) m = d; }
  return m;
}
export function rmse(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s / a.length);
}
export function relL2(a, b) {
  let ns = 0, ds = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; ns += d * d; ds += a[i] * a[i]; }
  return Math.sqrt(ns / (ds || 1));
}
export function cosSim(a, b) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / (Math.sqrt(na * nb) || 1);
}
