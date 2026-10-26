/**
 * 量化互操作 · forge 侧转储
 * ------------------------------------------------------------------
 * 用确定性 RNG 造两组数据，跑 forge 的 quantizeRow / dequantize，把
 * 「原始值 + 量化字节 + forge 反量化值」落到 JSON，交给官方 gguf-py 对拍。
 *
 *   clean       纯高斯，无病态块 —— 用于量化质量的断言（干净数据的误差才有可比性）
 *   adversarial 埋三类边界块 —— 用于布局断言（K-quant 的经典 bug 源）
 *       块 4 = 全零         → d=0 除零 / scale 退化
 *       块 5 = 常数 0.75    → 所有 q 相同，scale 搜索退化；全正子块还考验 min 符号钳制
 *       块 6 = 斜坡+单一 50×离群 → super-block 共享 d 的量程被单点拉爆
 *
 * 用法: node tools/interop/verify_quant.mjs [--out <path>]
 */
import { writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { RNG } from "../../src/L0/numeric.js";
import { quantizeRow, dequantize, rowBytes, TYPE_NAME } from "../../src/L3/quant.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, "out");
mkdirSync(OUT, { recursive: true });

const arg = (k, d) => {
  const i = process.argv.indexOf(k);
  return i >= 0 ? process.argv[i + 1] : d;
};
const outPath = arg("--out", join(OUT, "quant-forge.json"));

const SEED = 0x5eed1234;
const N = 4096;   // 16 个 K-quant super-block；Q8_0 则为 128 个块
const BLK = 256;
const TYPES = [1, 8, 12, 13, 14]; // F16 Q8_0 Q4_K Q5_K Q6_K

function gaussian() {
  const rng = new RNG(SEED);
  const x = new Float32Array(N);
  for (let i = 0; i < N; i++) x[i] = rng.normal() * 0.5;
  return x;
}

function build(kind) {
  const x = gaussian();
  if (kind === "adversarial") {
    x.fill(0, 4 * BLK, 5 * BLK);      // 块 4：全零
    x.fill(0.75, 5 * BLK, 6 * BLK);   // 块 5：常数
    for (let i = 6 * BLK; i < 7 * BLK; i++) x[i] = 0.01 * (i - 6 * BLK);
    x[6 * BLK + 137] = 50;            // 块 6：斜坡 + 单一极端离群
  }
  const cases = {};
  for (const t of TYPES) {
    const name = TYPE_NAME[t];
    let bytes;
    try {
      bytes = quantizeRow(t, x, N);
    } catch (e) {
      console.log(`  ${kind.padEnd(12)} ${name.padEnd(5)} 量化未实现: ${e.message}`);
      continue;
    }
    const expect = rowBytes(t, N);
    if (bytes.length !== expect) {
      console.error(`  ${name.padEnd(5)} 字节数不符: got ${bytes.length} expect ${expect}`);
      process.exit(1);
    }
    const deq = dequantize(t, bytes, N);
    let nonFinite = 0;
    for (let i = 0; i < N; i++) if (!Number.isFinite(deq[i])) nonFinite++;
    cases[name] = {
      type: t, bytes: Buffer.from(bytes).toString("hex"),
      nBytes: bytes.length, deq: Array.from(deq), nonFinite,
    };
    console.log(`  ${kind.padEnd(12)} ${name.padEnd(5)} ${String(bytes.length).padStart(6)} B  bpw=${(bytes.length * 8 / N).toFixed(4)}  非有限值=${nonFinite}`);
  }
  return { x: Array.from(x), cases };
}

const payload = { seed: SEED, n: N, block: BLK, datasets: {} };
for (const kind of ["clean", "adversarial"]) payload.datasets[kind] = build(kind);

writeFileSync(outPath, JSON.stringify(payload), { encoding: "utf-8" });
console.log(`OK 写入 ${outPath}`);
