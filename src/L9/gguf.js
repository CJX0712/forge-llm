/**
 * L9a · GGUF 格式 (GGUF v3 读写)
 * ------------------------------------------------------------------
 * 严格的 GGUF v3 实现，可直接与 llama.cpp 互操作：
 *  - 读：能加载任意符合规范的 GGUF（含真实 Llama/Qwen 权重，只要架构字段匹配）。
 *  - 写：本项目 export 产出的 GGUF，可被 llama.cpp / 本项目互相读取。
 *
 * 站在巨人肩上：GGUF 规范（ggml 团队，2023，https://github.com/ggml-org/ggml/blob/master/docs/gguf.md）。
 *
 * 不变量：
 *  I-90  写后再读，全部元数据的 (key,value) 与张量 (name,shape,type,offset,bytes) 逐字节一致
 *  I-91  tensor data 起始地址为 general.alignment（默认 32）的整数倍
 */

// GGUF 元数据值类型
export const GGUFValueType = {
  UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5,
  FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12,
};

const ALIGN = 32;

import { TYPE_INFO } from "../L3/quant.js";

// ── 字节读写工具 ─────────────────────────────────────────────────────────
class Reader {
  constructor(buf) { this.buf = buf; this.dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength); this.p = 0; }
  u8() { return this.dv.getUint8(this.p++); }
  i8() { return this.dv.getInt8(this.p++); }
  u16() { const v = this.dv.getUint16(this.p, true); this.p += 2; return v; }
  i16() { const v = this.dv.getInt16(this.p, true); this.p += 2; return v; }
  u32() { const v = this.dv.getUint32(this.p, true); this.p += 4; return v; }
  i32() { const v = this.dv.getInt32(this.p, true); this.p += 4; return v; }
  f32() { const v = this.dv.getFloat32(this.p, true); this.p += 4; return v; }
  u64() { const lo = this.dv.getUint32(this.p, true); const hi = this.dv.getUint32(this.p + 4, true); this.p += 8; return hi * 2 ** 32 + lo; }
  str() { const n = this.u32(); const s = new TextDecoder().decode(this.buf.subarray(this.p, this.p + n)); this.p += n; return s; }
  sub(n) { const s = this.buf.subarray(this.p, this.p + n); this.p += n; return s; }
}

class Writer {
  constructor() { this.chunks = []; this.p = 0; }
  _push(u8) { this.chunks.push(u8); this.p += u8.length; }
  u8(v) { this._push(new Uint8Array([v])); }
  i8(v) { const a = new Uint8Array(1); new DataView(a.buffer).setInt8(0, v); this._push(a); }
  u16(v) { const a = new Uint8Array(2); new DataView(a.buffer).setUint16(0, v, true); this._push(a); }
  i16(v) { const a = new Uint8Array(2); new DataView(a.buffer).setInt16(0, v, true); this._push(a); }
  u32(v) { const a = new Uint8Array(4); new DataView(a.buffer).setUint32(0, v, true); this._push(a); }
  i32(v) { const a = new Uint8Array(4); new DataView(a.buffer).setInt32(0, v, true); this._push(a); }
  f32(v) { const a = new Uint8Array(4); new DataView(a.buffer).setFloat32(0, v, true); this._push(a); }
  u64(v) { const lo = v >>> 0; const hi = Math.floor(v / 2 ** 32) >>> 0; const a = new Uint8Array(8); new DataView(a.buffer).setUint32(0, lo, true); new DataView(a.buffer).setUint32(4, hi, true); this._push(a); }
  str(s) { const b = new TextEncoder().encode(s); this.u32(b.length); this._push(b); }
  raw(u8) { this._push(u8); }
  bytes() { return concat(this.chunks); }
  padTo(align) { while (this.p % align !== 0) this.u8(0); }
}

function concat(chunks) {
  let n = 0; for (const c of chunks) n += c.length;
  const out = new Uint8Array(n); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

// ── 读 ─────────────────────────────────────────────────────────────────
export function readGGUF(buf) {
  const r = new Reader(buf);
  const magic = r.u32();
  if (magic !== 0x46554747) throw new Error(`非法 GGUF magic: 0x${magic.toString(16)}`);
  const version = r.u32();
  if (version < 1 || version > 3) throw new Error(`不支持的 GGUF 版本: ${version}`);
  const nTensors = r.u32();
  const nKv = r.u32();
  const meta = new Map();
  for (let i = 0; i < nKv; i++) {
    const key = r.str();
    const vt = r.u32();
    meta.set(key, readMetaValue(r, vt));
  }
  const tensors = [];
  for (let i = 0; i < nTensors; i++) {
    const name = r.str();
    const nDims = r.u32();
    const dims = [];
    for (let d = 0; d < nDims; d++) dims.push(Number(r.u64()));
    const type = r.u32();
    const offset = Number(r.u64());
    tensors.push({ name, dims, type, offset });
  }
  const align = meta.get("general.alignment") ?? ALIGN;
  const dataStart = Math.ceil(r.p / align) * align;
  const total = buf.byteLength;
  return {
    version, alignment: align, meta, tensors,
    tensorDataStart: dataStart,
    /** 取出张量（返回 {type,dims,data:Uint8Array}） */
    getTensor(name) {
      const t = tensors.find((x) => x.name === name);
      if (!t) return null;
      const nElem = t.dims.reduce((a, b) => a * b, 1);
      const len = tensorBytes(t.type, nElem);
      const s = dataStart + t.offset;
      return { type: t.type, dims: t.dims, data: buf.subarray(s, s + len) };
    },
  };
}

function readMetaValue(r, vt) {
  if (vt === GGUFValueType.ARRAY) {
    const elemType = r.u32();
    const count = Number(r.u64());
    const arr = [];
    for (let i = 0; i < count; i++) arr.push(readMetaValue(r, elemType));
    return arr;
  }
  switch (vt) {
    case GGUFValueType.UINT8: return r.u8();
    case GGUFValueType.INT8: return r.i8();
    case GGUFValueType.UINT16: return r.u16();
    case GGUFValueType.INT16: return r.i16();
    case GGUFValueType.UINT32: return r.u32();
    case GGUFValueType.INT32: return r.i32();
    case GGUFValueType.FLOAT32: return r.f32();
    case GGUFValueType.BOOL: return r.u8() !== 0;
    case GGUFValueType.STRING: return r.str();
    case GGUFValueType.UINT64: return r.u64();
    case GGUFValueType.INT64: return r.u64(); // JS number 上限够用
    case GGUFValueType.FLOAT64: { const a = new Float64Array(1); new DataView(a.buffer).setFloat64(0, 0, true); const v = new DataView(r.buf.buffer, r.buf.byteOffset + r.p, 8).getFloat64(0, true); r.p += 8; return v; }
    default: throw new Error(`未知 GGUF 值类型: ${vt}`);
  }
}

// ── 写 ─────────────────────────────────────────────────────────────────
export function writeGGUF({ meta = new Map(), tensors = [] }) {
  const w = new Writer();
  w.u32(0x46554747); w.u32(3); w.u32(tensors.length); w.u32(meta.size);
  // 元数据
  for (const [k, v] of meta) {
    w.str(k);
    const [vt, bytes] = encodeMetaValue(v);
    w.u32(vt);
    w.raw(bytes);
  }
  // 张量信息
  let offset = 0;
  const align = meta.get("general.alignment") ?? ALIGN;
  const tensorLens = [];
  for (const t of tensors) {
    const nElem = t.dims.reduce((a, b) => a * b, 1);
    const len = tensorBytes(t.type, nElem);
    tensorLens.push(len);
    w.str(t.name);
    w.u32(t.dims.length);
    for (const d of t.dims) w.u64(d);
    w.u32(t.type);
    w.u64(offset);
    offset += Math.ceil(len / align) * align;
  }
  // 数据段对齐
  w.padTo(align);
  const headerLen = w.p;
  for (let i = 0; i < tensors.length; i++) {
    const t = tensors[i];
    const len = tensorLens[i];
    w.raw(t.data);
    const pad = Math.ceil(len / align) * align - len;
    for (let j = 0; j < pad; j++) w.u8(0);
  }
  const out = w.bytes();
  return { bytes: out, headerLen, dataStart: Math.ceil(headerLen / align) * align };
}

function encodeMetaValue(v) {
  if (typeof v === "boolean") { const a = new Uint8Array([v ? 1 : 0]); return [GGUFValueType.BOOL, a]; }
  if (typeof v === "number") {
    if (Number.isInteger(v) && v >= 0 && v <= 0xffffffff) { const a = new Uint8Array(4); new DataView(a.buffer).setUint32(0, v, true); return [GGUFValueType.UINT32, a]; }
    const a = new Uint8Array(4); new DataView(a.buffer).setFloat32(0, v, true); return [GGUFValueType.FLOAT32, a];
  }
  if (typeof v === "string") {
    // gguf-string：u32 长度前缀 + UTF-8 字节（数组元素同样适用）
    const b = new TextEncoder().encode(v);
    const w = new Writer(); w.u32(b.length); w.raw(b);
    return [GGUFValueType.STRING, w.bytes()];
  }
  if (Array.isArray(v)) {
    // 推断元素类型（全部同类型）
    let elemType;
    if (Array.isArray(v[0])) elemType = GGUFValueType.STRING;
    else if (typeof v[0] === "string") elemType = GGUFValueType.STRING;
    else if (typeof v[0] === "number" && Number.isInteger(v[0])) elemType = GGUFValueType.INT32;
    else elemType = GGUFValueType.FLOAT32;
    const w2 = new Writer();
    w2.u32(elemType);
    w2.u64(v.length);
    for (const e of v) { const [, b] = encodeMetaValue(e); w2.raw(b); }
    return [GGUFValueType.ARRAY, w2.bytes()];
  }
  throw new Error(`GGUF 不支持的元数据值类型: ${typeof v}`);
}

/** GGML 类型单元素平均字节数（块量化按比例，如 Q4_K=144/256）
 *  精确写法：对块量化类型返回 blockBytes/blockSize（分数），保证
 *  `size × 元素数` 在元素数为块长整数倍时得到精确字节数（GGUF 规范要求）。 */
export function ggmlTypeSize(type) {
  const ti = TYPE_INFO[type];
  if (ti) return ti.blockBytes / ti.blockSize;
  const sizes = { 0: 4, 1: 2 };
  if (sizes[type] !== undefined) return sizes[type];
  throw new Error(`未支持的 ggml 类型: ${type}`);
}

/** 张量占用的精确字节数（GGUF 规范：量化张量元素数必须是块长整数倍） */
export function tensorBytes(type, nElem) {
  const ti = TYPE_INFO[type];
  if (!ti) return ggmlTypeSize(type) * nElem;
  if (ti.blockSize === 1) return ti.blockBytes * nElem;
  if (nElem % ti.blockSize !== 0) {
    throw new Error(`量化张量元素数 ${nElem} 必须是块长 ${ti.blockSize} 的整数倍（type=${ti.name}）`);
  }
  return (nElem / ti.blockSize) * ti.blockBytes;
}
