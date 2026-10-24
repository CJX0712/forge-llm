/**
 * L8 · 采样策略 (Sampling)
 * ------------------------------------------------------------------
 * 对齐当前 SOTA 解码配方，全部基于 L0 的确定性 RNG（可复现）：
 *  - temperature：先除温再 softmax。
 *  - top-k（Fan et al. 2018）、top-p / nucleus（Holtzman et al. 2020）。
 *  - min-p（Nguyen et al. 2024, arXiv:2407.01082）：阈值 = min_p · p_max，
 *    比 top-p 更稳，是 2024 年起主流 LLM 的默认搭配。
 *  - 频率 / 存在惩罚（frequency & presence penalty，OpenAI 2020）。
 *  - greedy（argmax）用于确定性评测。
 *
 * 不变量：
 *  I-80  贪心 Σ p = 1（采样前归一化后）
 *  I-81  temperature→0 退化为 greedy
 *  I-82  min-p 阈值下所有 p<min_p·p_max 的概率被清零并重新归一化
 */

import { RNG } from "../L0/numeric.js";
import { softmax } from "../L1/special.js";

/**
 * 核心采样器。
 * @param logits Float32Array（未归一化）
 * @param opt {temperature,k,top_p,min_p,seed,freqPenalty,presPenalty,forbid?}
 * @param rng  可选预置 RNG（复现用）
 * @returns 选中的 token id
 */
export function sample(logits, opt = {}, rng = new RNG(opt.seed ?? 0x9e3779b9)) {
  const n = logits.length;
  const temperature = opt.temperature ?? 1.0;
  const k = opt.top_k ?? 0;          // 0 = 不限制
  const topP = opt.top_p ?? 1.0;
  const minP = opt.min_p ?? 0.0;
  const t = Math.max(temperature, 1e-6);

  // 0) 频率 / 存在惩罚：直接改写 logits
  if ((opt.freqPenalty || opt.presPenalty) && opt.counts) {
    for (let i = 0; i < n; i++) {
      const c = opt.counts[i] | 0;
      if (c > 0) logits[i] -= opt.freqPenalty * c + opt.presPenalty * (c > 0 ? 1 : 0);
    }
  }
  // 1) 温度缩放
  const scaled = new Float32Array(n);
  for (let i = 0; i < n; i++) scaled[i] = logits[i] / t;

  // greedy 分支
  if (temperature === 0 || (k === 1 && topP >= 1)) return argmax(scaled);

  // 2) softmax（稳定）
  const probs = new Float32Array(n);
  softmax(probs, scaled, n);

  // 3) min-p 截断：阈值 = minP · p_max
  if (minP > 0) {
    let pmax = 0; for (let i = 0; i < n; i++) if (probs[i] > pmax) pmax = probs[i];
    const thr = minP * pmax;
    let sum = 0;
    for (let i = 0; i < n; i++) { if (probs[i] < thr) probs[i] = 0; else sum += probs[i]; }
    if (sum > 0) for (let i = 0; i < n; i++) probs[i] /= sum;
  }

  // 4) top-k
  let allow = new Uint8Array(n).fill(1);
  if (k > 1) {
    const order = argsortDesc(probs, n);
    allow.fill(0);
    for (let i = 0; i < k && i < n; i++) allow[order[i]] = 1;
  }

  // 5) top-p（nucleus）：按概率降序累加至 ≥ topP
  if (topP < 1) {
    const order = argsortDesc(probs, n);
    let cum = 0;
    for (let i = 0; i < n; i++) {
      const id = order[i];
      if (!allow[id]) continue;
      cum += probs[id];
      if (cum >= topP) { for (let j = i + 1; j < n; j++) allow[order[j]] = 0; break; }
    }
  }

  // 6) 归一化被允许集合并采样
  let sum = 0;
  for (let i = 0; i < n; i++) if (allow[i]) { sum += (opt.forbid && opt.forbid[i]) ? 0 : probs[i]; }
  let r = rng.next() * sum;
  let acc = 0;
  for (let i = 0; i < n; i++) {
    if (!allow[i]) continue;
    if (opt.forbid && opt.forbid[i]) continue;
    acc += probs[i];
    if (r <= acc) return i;
  }
  // 兜底：选概率最大者
  return argmax(probs);
}

/** 多 token 贪心校验：返回给定 logits 下 argmax 是否等于 target */
export function greedyId(logits) { return argmax(logits); }

function argmax(a) { let bi = 0, bv = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > bv) { bv = a[i]; bi = i; } return bi; }
function argsortDesc(arr, n) {
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => arr[b] - arr[a]);
  return idx;
}
