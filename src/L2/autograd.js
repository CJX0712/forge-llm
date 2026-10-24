/**
 * L2 · 自动微分 (Reverse-mode Autodiff)
 * ------------------------------------------------------------------
 * 站在巨人肩上：
 *  - 反向模式 AD：Linnainmaa 1970 / Rumelhart 1986；tape 结构与 PyTorch autograd 同构。
 *  - Adam：Kingma & Ba 2015（含 bias correction，论文式 (7)(8)）。
 *  - 余弦学习率：Loshchilov & Hutter 2017（SGDR）。
 *  - 梯度裁剪：Pascanu et al. 2013（按全局范数缩放）。
 *  - 有限差分梯度检验：中心差分，误差应为 O(ε²)≈1e-7 量级。
 *
 * 不变量：
 *  I-30  解析梯度 vs 中心差分：maxRelErr < 1e-5
 *  I-31  softmax-CE 的解析梯度恒等于 p − onehot
 *  I-32  Adam 首步更新幅度 ≈ lr（bias correction 生效的判据）
 */

import { matvec } from "../L0/numeric.js";

// ── Tape ─────────────────────────────────────────────────────────────
let _tape = { nodes: [], enabled: true };
export function tape() { return _tape; }
export function resetTape() { _tape = { nodes: [], enabled: true }; }
export function noGrad(fn) { const e = _tape.enabled; _tape.enabled = false; try { return fn(); } finally { _tape.enabled = e; } }

export class Node {
  constructor(data, requiresGrad = true) {
    this.data = data instanceof Float32Array ? data : Float32Array.from(data);
    this.grad = null;
    this._b = null;
    if (requiresGrad && _tape.enabled) _tape.nodes.push(this);
  }
  get n() { return this.data.length; }
  zeroGrad() { if (this.grad) this.grad.fill(0); }
}

export const leaf = (data) => new Node(data);
export const param = (data) => { const n = new Node(data); n.isParam = true; return n; };

export function op(out, bwd) {
  const n = new Node(out);
  n._b = bwd;
  return n;
}

// ── 算子 ─────────────────────────────────────────────────────────────
/** 线性层：out[N] = W[N,K] · x[K] */
export function linear(W, x, N, K) {
  const out = new Float32Array(N);
  matvec(out, W.data, x.data, N, K);
  return op(out, (g) => {
    const gW = (W.grad ??= new Float32Array(N * K));
    const gx = (x.grad ??= new Float32Array(K));
    for (let n = 0; n < N; n++) {
      const gn = g[n], b = n * K;
      for (let k = 0; k < K; k++) { gW[b + k] += gn * x.data[k]; gx[k] += gn * W.data[b + k]; }
    }
  });
}

export function add(a, b) {
  const n = a.data.length, out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = a.data[i] + b.data[i];
  return op(out, (g) => {
    const ga = (a.grad ??= new Float32Array(n)), gb = (b.grad ??= new Float32Array(n));
    for (let i = 0; i < n; i++) { ga[i] += g[i]; gb[i] += g[i]; }
  });
}

/** 逐元素乘（broadcast：b 可为标量 Node，长度 1） */
export function mul(a, b) {
  const nb = b.data.length, n = a.data.length, out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = a.data[i] * b.data[nb === 1 ? 0 : i];
  return op(out, (g) => {
    const ga = (a.grad ??= new Float32Array(n));
    for (let i = 0; i < n; i++) ga[i] += g[i] * b.data[nb === 1 ? 0 : i];
    if (nb > 1) { const gb = (b.grad ??= new Float32Array(n)); for (let i = 0; i < n; i++) gb[i] += g[i] * a.data[i]; }
    else { const gb = (b.grad ??= new Float32Array(1)); for (let i = 0; i < n; i++) gb[0] += g[i] * a.data[i]; }
  });
}

export function silu(x) {
  const n = x.data.length, out = new Float32Array(n), s = new Float32Array(n);
  for (let i = 0; i < n; i++) { s[i] = 1 / (1 + Math.exp(-x.data[i])); out[i] = x.data[i] * s[i]; }
  return op(out, (g) => {
    const gx = (x.grad ??= new Float32Array(n));
    for (let i = 0; i < n; i++) gx[i] += g[i] * (s[i] + x.data[i] * s[i] * (1 - s[i]));
  });
}

/** RMSNorm（Zhang & Sennrich 2019），含解析反向 */
export function rmsnorm(x, w, eps = 1e-6) {
  const n = x.data.length;
  let ss = 0; for (let i = 0; i < n; i++) ss += x.data[i] * x.data[i];
  const r = Math.sqrt(ss / n + eps), inv = 1 / r;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = x.data[i] * inv * w.data[i];
  return op(out, (g) => {
    const gx = (x.grad ??= new Float32Array(n)), gw = (w.grad ??= new Float32Array(n));
    let dot = 0;
    for (let i = 0; i < n; i++) dot += g[i] * w.data[i] * x.data[i];
    const c = dot / (n * r * r * r);
    for (let i = 0; i < n; i++) {
      gx[i] += g[i] * w.data[i] * inv - c * x.data[i];
      gw[i] += g[i] * x.data[i] * inv;
    }
  });
}

/** softmax + 交叉熵合并：数值稳定，反向恰为 p − onehot */
export function softmaxCE(logits, target) {
  const n = logits.data.length;
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (logits.data[i] > m) m = logits.data[i];
  let s = 0; const p = new Float32Array(n);
  for (let i = 0; i < n; i++) { p[i] = Math.exp(logits.data[i] - m); s += p[i]; }
  for (let i = 0; i < n; i++) p[i] /= s;
  const out = new Float32Array(1);
  out[0] = -Math.log(Math.max(p[target], 1e-40));
  return op(out, (g) => {
    const gl = (logits.grad ??= new Float32Array(n));
    const k = g[0];
    for (let i = 0; i < n; i++) gl[i] += k * (p[i] - (i === target ? 1 : 0));
  });
}

export function backward(loss) {
  loss.grad = new Float32Array(loss.data.length).fill(1);
  const nodes = _tape.nodes;
  for (let i = nodes.length - 1; i >= 0; i--) {
    const nd = nodes[i];
    if (nd._b && nd.grad) nd._b(nd.grad);
  }
}

/** 向量内积：out = Σ a_i b_i（训练注意力分数的核心） */
export function dot(a, b) {
  const n = a.data.length;
  let s = 0; for (let i = 0; i < n; i++) s += a.data[i] * b.data[i];
  const out = new Float32Array(1); out[0] = s;
  return op(out, (g) => {
    const ga = (a.grad ??= new Float32Array(n)), gb = (b.grad ??= new Float32Array(n));
    const k = g[0];
    for (let i = 0; i < n; i++) { ga[i] += k * b.data[i]; gb[i] += k * a.data[i]; }
  });
}

/** 标量缩放：out_i = s · x_i */
export function scale(x, s) {
  const n = x.data.length, out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = s * x.data[i];
  return op(out, (g) => { (x.grad ??= new Float32Array(n)); for (let i = 0; i < n; i++) x.grad[i] += g[i] * s; });
}

/** 可微 RoPE：对长度 dHead 的向量施加旋转（cos/sin 为常数，非 Node）。
 *  反向即乘以转置旋转矩阵 R^T = R^{-1}，故梯度正确回传。 */
export function ropeNode(x, cos, sin, dHead) {
  const n = x.data.length, out = new Float32Array(n);
  for (let i = 0; i < dHead; i += 2) {
    const a = x.data[i], b = x.data[i + 1];
    out[i] = a * cos[i >> 1] - b * sin[i >> 1];
    out[i + 1] = a * sin[i >> 1] + b * cos[i >> 1];
  }
  return op(out, (g) => {
    const gx = (x.grad ??= new Float32Array(n));
    for (let i = 0; i < dHead; i += 2) {
      const ci = cos[i >> 1], si = sin[i >> 1];
      gx[i] += g[i] * ci + g[i + 1] * si;
      gx[i + 1] += -g[i] * si + g[i + 1] * ci;
    }
  });
}

/** 向量 softmax（逐元素），反向即标准 softmax 雅可比 */
export function softmaxVec(x) {
  const n = x.data.length;
  let m = -Infinity; for (let i = 0; i < n; i++) if (x.data[i] > m) m = x.data[i];
  const p = new Float32Array(n); let s = 0;
  for (let i = 0; i < n; i++) { p[i] = Math.exp(x.data[i] - m); s += p[i]; }
  for (let i = 0; i < n; i++) p[i] /= s;
  const out = p.slice();
  return op(out, (g) => {
    const gx = (x.grad ??= new Float32Array(n));
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let j = 0; j < n; j++) acc += g[j] * (p[i] * ((i === j ? 1 : 0) - p[j]));
      gx[i] += acc;
    }
  });
}

/** 中心差分梯度检验 —— 期望 maxRelErr ≈ 1e-7 */
export function gradCheck(lossFn, params, eps = 1e-3) {
  let worst = 0, worstName = "";
  const base = lossFn();      // 构建计算图并返回 loss 节点
  backward(base);             // 反向传播，填充 p.grad（基准点）
  for (const [name, p] of Object.entries(params)) {
    for (let i = 0; i < p.data.length; i += Math.max(1, Math.floor(p.data.length / 24))) {
      const orig = p.data[i];
      p.data[i] = orig + eps; const lp = lossFn().data[0];
      p.data[i] = orig - eps; const lm = lossFn().data[0];
      p.data[i] = orig;
      const num = (lp - lm) / (2 * eps);
      const ana = p.grad ? p.grad[i] : 0;
      const denom = Math.max(1, Math.abs(num), Math.abs(ana));
      const rel = Math.abs(num - ana) / denom;
      if (rel > worst) { worst = rel; worstName = `${name}[${i}]`; }
    }
  }
  return { maxRelErr: worst, at: worstName, loss: base.data[0] };
}

// ── 优化器 ───────────────────────────────────────────────────────────
/** Adam（Kingma & Ba 2015）+ 全局范数裁剪（Pascanu 2013）+ 余弦退火（SGDR） */
export class Adam {
  constructor(params, { lr = 3e-4, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 0, clip = 1.0, totalSteps = 1000 } = {}) {
    this.p = params; this.lr = lr; this.b1 = b1; this.b2 = b2; this.eps = eps;
    this.wd = wd; this.clip = clip; this.total = totalSteps; this.t = 0;
    this.m = params.map((p) => new Float32Array(p.data.length));
    this.v = params.map((p) => new Float32Array(p.data.length));
  }
  step() {
    this.t++;
    // 1) 梯度裁剪
    let tot = 0;
    for (const p of this.p) if (p.grad) for (let i = 0; i < p.grad.length; i++) tot += p.grad[i] * p.grad[i];
    const gnorm = Math.sqrt(tot);
    const scale = gnorm > this.clip ? this.clip / gnorm : 1;
    // 2) 余弦学习率
    const lr = this.lr * 0.5 * (1 + Math.cos(Math.PI * Math.min(this.t / this.total, 1)));
    for (let j = 0; j < this.p.length; j++) {
      const p = this.p[j]; if (!p.grad) continue;
      const m = this.m[j], v = this.v[j];
      for (let i = 0; i < p.data.length; i++) {
        const g = p.grad[i] * scale + this.wd * p.data[i];
        m[i] = this.b1 * m[i] + (1 - this.b1) * g;
        v[i] = this.b2 * v[i] + (1 - this.b2) * g * g;
        const mh = m[i] / (1 - Math.pow(this.b1, this.t));
        const vh = v[i] / (1 - Math.pow(this.b2, this.t));
        p.data[i] -= lr * mh / (Math.sqrt(vh) + this.eps);
      }
    }
    return { lr, gnorm, scale };
  }
}
