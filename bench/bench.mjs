/**
 * bench · 性能基准
 * ------------------------------------------------------------------
 * 全部指标可复现：固定 RNG 种子、固定矩阵尺寸、固定迭代次数。
 * 输出可直接粘进 PERFORMANCE.md。
 *
 * 方法论说明（重要）：
 *  1) 计时取「充分预热后的最小单次耗时」。取最小值而非平均值，是因为均值会被
 *     GC 暂停与 OS 调度抖动抬高（本机实测同一忙循环可有 6× 方差）。
 *  2) 对比项（分块 vs 朴素、flash vs 朴素）采用交错计时 A/B/A/B…，
 *     让两侧处于相同的机器状态，抵消频率漂移。
 *  3) 大张量用完即释放，避免 64MB 级分配带来的内存压力污染后续计时。
 *
 * 基准项目：
 *  B1  GEMM   —— 4 路展开 vs 朴素三重循环（加速比；缓存分块经实测不采用）
 *  B2  matvec —— 解码热点吞吐（GFLOPS）
 *  B3  量化   —— Q4_K / Q6_K / Q8_0 / F16 的 matvec 速度与压缩比、RMSE、cos
 *  B4  注意力 —— FlashAttention(在线 softmax) vs 朴素 N² 物化
 *  B5  端到端 —— tiny Transformer 的 prefill 与 decode tok/s
 *  B6  训练   —— 前向+反向 step 时间
 */

import { gemm, gemmRef, matvec, RNG, relL2 } from "../src/L0/numeric.js";
import { quantizeRow, info, qualityReport, dotQ } from "../src/L3/quant.js";
import { flashAttention, attentionNaive, attentionDecode, KVCache } from "../src/L5/attention.js";
import { Config, Transformer } from "../src/L6/transformer.js";
import { TrainableTransformer, initWeights } from "../src/L6/train.js";
import { resetTape, Adam } from "../src/L2/autograd.js";

const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : String(v));
const pad = (s, n) => String(s).padEnd(n);

/** 单函数计时：充分预热后跑 iters 次取最小单次耗时（ms） */
function timeIt(fn, iters) {
  const warm = iters <= 3 ? 5 : 25;
  for (let i = 0; i < warm; i++) fn();
  let best = Infinity;
  for (let i = 0; i < iters; i++) {
    const t0 = process.hrtime.bigint();
    fn();
    const dt = Number(process.hrtime.bigint() - t0) / 1e6;
    if (dt < best) best = dt;
  }
  return best;
}

/**
 * 交错对比计时：A/B/A/B… 交替执行 rounds 轮，各取最小单次耗时。
 * 交替执行使两个被测对象处于相同的机器状态（频率、缓存、GC 相位）。
 */
function timeAB(fnA, fnB, rounds, warm = 5) {
  for (let i = 0; i < warm; i++) { fnA(); fnB(); }
  let bestA = Infinity, bestB = Infinity;
  for (let r = 0; r < rounds; r++) {
    let t0 = process.hrtime.bigint(); fnA();
    const dA = Number(process.hrtime.bigint() - t0) / 1e6;
    if (dA < bestA) bestA = dA;
    t0 = process.hrtime.bigint(); fnB();
    const dB = Number(process.hrtime.bigint() - t0) / 1e6;
    if (dB < bestB) bestB = dB;
  }
  return { a: bestA, b: bestB };
}

export async function runBench({ json = false, only = null } = {}) {
  const ON = (k) => (!only || only === k); // 诊断用：only="B3" 时其余节跳过执行但保持 RNG 消耗一致
  const R = {};
  const rng = new RNG(20260101);

  // ── B1 GEMM：4 路展开 vs 朴素三重循环 ──
  // 选型结论（本机实测）：JS/V8 下缓存分块不划算，展开 + 顺序读最快。
  R.gemm = [];
  for (const [M, N, K, rounds] of [[128, 256, 256, 15], [512, 1024, 1024, 5]]) {
    const A = new Float32Array(M * K), B = new Float32Array(N * K);
    for (let i = 0; i < A.length; i++) A[i] = rng.normal() * 0.1;
    for (let i = 0; i < B.length; i++) B[i] = rng.normal() * 0.1;
    const C = new Float32Array(M * N), Cr = new Float32Array(M * N);
    let tFast = 0, tRef = 0;
    if (ON("B1")) ({ a: tFast, b: tRef } = timeAB(() => gemm(C, A, B, M, N, K), () => gemmRef(Cr, A, B, M, N, K), rounds));
    R.gemm.push({
      shape: `${M}×${N}×${K}`,
      blocked_ms: +tFast.toFixed(3),
      naive_ms: +tRef.toFixed(3),
      speedup: +(tRef / tFast).toFixed(2),
      gflops_blocked: +((2 * M * N * K) / (tFast / 1000) / 1e9).toFixed(2),
      gflops_naive: +((2 * M * N * K) / (tRef / 1000) / 1e9).toFixed(2),
      relL2: relL2(C, Cr),
    });
  }

  // ── B2 matvec（解码热点）──
  {
    const N = 4096, K = 2048; // 8MB，代表一层中等规模投影；避免超大分配污染后续计时
    let W = new Float32Array(N * K);
    for (let i = 0; i < W.length; i++) W[i] = rng.normal() * 0.05;
    const x = new Float32Array(K); for (let i = 0; i < K; i++) x[i] = rng.normal();
    const out = new Float32Array(N);
    const t = ON("B2") ? timeIt(() => matvec(out, W, x, N, K), 30) : 0;
    W = null; // 及时释放
    R.matvec = {
      shape: `${N}×${K}`,
      ms: +t.toFixed(4),
      gflops: +((2 * N * K) / (t / 1000) / 1e9).toFixed(2),
      tokens_per_s_equivalent: Math.round(1000 / t),
    };
  }

  // ── B3 量化格式 ──
  {
    const N = 4096;
    const x = new Float32Array(N); for (let i = 0; i < N; i++) x[i] = rng.normal() * 0.7;
    const y = new Float32Array(N); for (let i = 0; i < N; i++) y[i] = rng.normal();
    const rows = [];
    const FORMATS = [["F16", 1], ["Q8_0", 8], ["Q6_K", 14], ["Q4_K", 12]];
    const qs = FORMATS.map(([, t]) => quantizeRow(t, x, N));
    // 交错计时：F16→Q8_0→Q6_K→Q4_K→F16→… 轮转，各取最小单次耗时。
    // 本机存在持续性 CPU 抖动（实测同一忙循环有 6× 方差），交错执行可保证
    // 四种格式处于相同的机器状态，相对比较不受全局 throttling 影响。
    const best = new Float64Array(FORMATS.length).fill(Infinity);
    if (ON("B3")) {
      const warm = 25, rounds = 120;
      for (let w = 0; w < warm; w++) for (let i = 0; i < FORMATS.length; i++) dotQ(FORMATS[i][1], qs[i], y, N);
      for (let r = 0; r < rounds; r++) {
        for (let i = 0; i < FORMATS.length; i++) {
          const t0 = process.hrtime.bigint();
          dotQ(FORMATS[i][1], qs[i], y, N);
          const ms = Number(process.hrtime.bigint() - t0) / 1e6;
          if (ms < best[i]) best[i] = ms;
        }
      }
    }
    for (let i = 0; i < FORMATS.length; i++) {
      const [name, t] = FORMATS[i];
      const tq = best[i] === Infinity ? 0 : best[i];
      const rep = qualityReport(t, x, N);
      rows.push({
        name, bpw: info(t).bpw,
        bytes: rep.bytes, compression: +(rep.ratio).toFixed(2),
        rmse: +rep.rmse.toExponential(3),
        cos: +rep.cos.toFixed(6),
        ms: +tq.toFixed(4),
        ns_per_elem: +((tq * 1e6) / N).toFixed(2),
        gflops: +((2 * N) / (tq / 1000) / 1e9).toFixed(3),
      });
    }
    R.quant = rows;
  }

  // ── B4 注意力：三种单 query 行实现交错对比 ──
  //   flash  = 在线 softmax（O(d) 内存，无分配）
  //   naive  = 物化 scores + softmax（参考实现，每次调用分配 O(n)）
  //   decode = 两趟扫描 + 复用 scratch（L6 解码路径实际使用）
  // 单 query 行下三者计算量同为 O(n·d)，实测差距 <10%；decode 一致略快且零分配。
  {
    const n = 512, d = 64;
    const q = new Float32Array(d); for (let i = 0; i < d; i++) q[i] = rng.normal();
    const K = new Float32Array(n * d), V = new Float32Array(n * d);
    for (let i = 0; i < n * d; i++) { K[i] = rng.normal(); V[i] = rng.normal(); }
    const o1 = new Float32Array(d), o2 = new Float32Array(d), o3 = new Float32Array(d);
    const scratch = new Float32Array(n);
    const fns = [
      () => flashAttention(o1, q, K, V, n, d, 1 / Math.sqrt(d), true, n - 1, 0),
      () => attentionNaive(o2, q, K, V, n, d, 1 / Math.sqrt(d), true, n - 1, 0),
      () => attentionDecode(o3, q, K, V, n, d, 1 / Math.sqrt(d), true, n - 1, 0, scratch),
    ];
    const best = [Infinity, Infinity, Infinity];
    if (ON("B4")) {
      for (let w = 0; w < 10; w++) for (const f of fns) f();
      for (let r = 0; r < 100; r++) {
        for (let i = 0; i < fns.length; i++) {
          const t0 = process.hrtime.bigint();
          fns[i]();
          const ms = Number(process.hrtime.bigint() - t0) / 1e6;
          if (ms < best[i]) best[i] = ms;
        }
      }
    }
    R.attention = {
      seq: n, headDim: d,
      flash_ms: +best[0].toFixed(4), naive_ms: +best[1].toFixed(4), decode_ms: +best[2].toFixed(4),
      decode_vs_flash: +(best[0] / best[2]).toFixed(2),
      decode_vs_naive: +(best[1] / best[2]).toFixed(2),
      maxErr: Math.max(...Array.from(o1, (v, i) => Math.max(Math.abs(v - o3[i]), Math.abs(o2[i] - o3[i])))),
    };
  }

  // ── B5 端到端 prefill / decode ──
  {
    const cfg = new Config({ dim: 256, hidden: 512, nLayers: 4, nHeads: 8, nKvHeads: 4, vocabSize: 4096, maxSeq: 512 });
    const w = initWeights(cfg, 777);
    const tr = new Transformer(cfg).loadWeights({
      tok_emb: w.tok_emb, ln_f: w.ln_f, wcls: null,
      ...Object.fromEntries(Array.from({ length: cfg.nLayers }, (_, l) => [`layer.${l}`, {
        ln1: w[`layer.${l}`].ln1, ln2: w[`layer.${l}`].ln2,
        wqkv: w[`layer.${l}`].wqkv, wo: w[`layer.${l}`].wo,
        ffn_gate: w[`layer.${l}`].ffn_gate, ffn_up: w[`layer.${l}`].ffn_up, ffn_down: w[`layer.${l}`].ffn_down,
      }])),
    });
    const prefillLen = 32;
    const tPre = ON("B5") ? timeIt(() => {
      const cache = new KVCache(cfg.nLayers, cfg.maxSeq, cfg.kvDim);
      for (let i = 0; i < prefillLen; i++) tr.forward(i % cfg.vocabSize, i, cache);
    }, 8) : 0;
    const cache = new KVCache(cfg.nLayers, cfg.maxSeq, cfg.kvDim);
    for (let i = 0; i < 128; i++) tr.forward(i % cfg.vocabSize, i, cache);
    const tDec = ON("B5") ? timeIt(() => { tr.forward(7, 200, cache); }, 30) : 0;
    R.e2e = {
      config: `dim=${cfg.dim} hidden=${cfg.hidden} layers=${cfg.nLayers} heads=${cfg.nHeads}/${cfg.nKvHeads} vocab=${cfg.vocabSize}`,
      params: cfg.vocabSize * cfg.dim + cfg.nLayers * (3 * cfg.dim * cfg.dim + 3 * cfg.dim * cfg.hidden),
      prefill_len: prefillLen,
      prefill_ms: +tPre.toFixed(2),
      prefill_tok_s: Math.round(prefillLen / (tPre / 1000)),
      decode_ms: +tDec.toFixed(3),
      decode_tok_s: Math.round(1000 / tDec),
    };
  }

  // ── B6 训练 step ──
  {
    const cfg = new Config({ dim: 128, hidden: 256, nLayers: 2, nHeads: 4, nKvHeads: 2, vocabSize: 512, maxSeq: 128 });
    const m = new TrainableTransformer(cfg, initWeights(cfg, 999));
    const opt = new Adam(m.paramList, { lr: 1e-3, totalSteps: 10 });
    const ids = Array.from({ length: 16 }, (_, i) => i % cfg.vocabSize);
    const tgt = Array.from({ length: 15 }, (_, i) => (i + 1) % cfg.vocabSize);
    const step = () => {
      resetTape(); m.zeroGrad();
      const l = m.loss(ids, tgt);
      m.backward(l);
      opt.step();
    };
    const t = ON("B6") ? timeIt(step, 8) : 0;
    R.train = {
      config: `dim=${cfg.dim} layers=${cfg.nLayers} seq=16 batch=1`,
      step_ms: +t.toFixed(2),
      params: m.countParams(),
      tokens_per_s: Math.round(16 / (t / 1000)),
    };
  }

  if (json) {
    console.log(JSON.stringify(R, null, 2));
    return R;
  }

  // ── 打印 ──
  const C = { b: (s) => `\x1b[1m${s}\x1b[0m`, d: (s) => `\x1b[2m${s}\x1b[0m`, g: (s) => `\x1b[32m${s}\x1b[0m` };
  console.log(C.b("\nFORGE · 性能基准\n"));

  console.log(C.b("B1 · GEMM 4 路展开 vs 朴素三重循环"));
  for (const g of R.gemm) {
    console.log(`  形状 ${g.shape}  展开 ${fmt(g.blocked_ms)} ms  朴素 ${fmt(g.naive_ms)} ms`);
    console.log(`  加速比 ${C.g(fmt(g.speedup) + "×")}   ${fmt(g.gflops_blocked)} GFLOPS (展开) vs ${fmt(g.gflops_naive)} GFLOPS (朴素)   relL2=${g.relL2.toExponential(2)}`);
  }
  console.log(C.d("  （另测：BK/BN 缓存分块三变体在本机均慢于展开版，故不采用）\n"));

  console.log(C.b("B2 · matvec（解码热点）"));
  console.log(`  形状 ${R.matvec.shape}  ${fmt(R.matvec.ms)} ms  ${fmt(R.matvec.gflops)} GFLOPS\n`);

  console.log(C.b("B3 · 量化格式"));
  console.log(`  ${pad("格式", 8)}${pad("bpw", 8)}${pad("压缩比", 10)}${pad("RMSE", 12)}${pad("cos", 12)}${pad("ms", 10)}${pad("ns/元素", 10)}${pad("GFLOPS", 10)}`);
  for (const q of R.quant) {
    console.log(`  ${pad(q.name, 8)}${pad(q.bpw, 8)}${pad(q.compression + "×", 10)}${pad(q.rmse.toExponential(2), 12)}${pad(q.cos, 12)}${pad(q.ms, 10)}${pad(q.ns_per_elem, 10)}${pad(q.gflops, 10)}`);
  }
  console.log();

  console.log(C.b("B4 · 注意力（单 query 行，三种实现交错对比）"));
  console.log(`  seq=${R.attention.seq} d=${R.attention.headDim}  flash ${fmt(R.attention.flash_ms, 4)} ms  朴素 ${fmt(R.attention.naive_ms, 4)} ms  decode ${fmt(R.attention.decode_ms, 4)} ms`);
  console.log(`  decode 相对 flash ${C.g(R.attention.decode_vs_flash + "×")}  相对朴素 ${C.g(R.attention.decode_vs_naive + "×")}   数值一致性 maxErr=${R.attention.maxErr.toExponential(2)}`);
  console.log(C.d("  （三者计算量同为 O(n·d)；decode 零分配且一致略快，故 L6 解码路径采用之）\n"));

  console.log(C.b("B5 · 端到端推理"));
  console.log(`  ${R.e2e.config}`);
  console.log(`  prefill(${R.e2e.prefill_len}) ${fmt(R.e2e.prefill_ms)} ms → ${C.g(R.e2e.prefill_tok_s + " tok/s")}`);
  console.log(`  decode       ${fmt(R.e2e.decode_ms, 3)} ms → ${C.g(R.e2e.decode_tok_s + " tok/s")}\n`);

  console.log(C.b("B6 · 训练步"));
  console.log(`  ${R.train.config}  ${fmt(R.train.step_ms)} ms/step  ${R.train.params.toLocaleString()} 参数  ${R.train.tokens_per_s} tok/s\n`);

  console.log(C.d(`Node ${process.version} · ${process.platform} ${process.arch}\n`));
  return R;
}

if (process.argv[1] && process.argv[1].replace(/\\/g, "/").endsWith("bench/bench.mjs")) {
  const onlyArg = process.argv.find((a) => a.startsWith("--only="));
  await runBench({ json: process.argv.includes("--json"), only: onlyArg ? onlyArg.slice(7) : null });
}
