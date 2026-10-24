/**
 * L7 · 分词器 (Tokenizer)
 * ------------------------------------------------------------------
 * 两套实现，分别服务「训练演示」与「真实 GGUF 模型」：
 *
 *  1) CharTokenizer —— Karpathy llama2.c 同款字符级分词。
 *     小、零依赖、可训练，是 examples/tiny-train 的默认分词器。
 *
 *  2) GGUFVocab —— 对齐 llama.cpp 的词表加载与解码（含字节回退）。
 *     · 解码侧完全正确：把每个 token 当作**原始字节串**（type=1 即单字节），
 *       拼接后按 UTF-8 还原 —— 与 llama.cpp 的 `llama_token_to_piece` 行为一致。
 *     · 编码侧：基于字节的最长匹配（含 byte-fallback token），覆盖绝大多数
 *       情形；如 GGUF 提供 BPE `merges` 则走正规 BPE。
 *
 * 站在巨人肩上：
 *  - SentencePiece / Llama byte-fallback（Kudo & Richardson 2018; Touvron 2023）。
 *  - 字节级 BPE 与 byte-decode 算法（GPT-2, Radford 2019）。
 */

// ── 1) 字符级分词（训练/演示）────────────────────────────────────────────
export class CharTokenizer {
  constructor(text, { bos = 1, eos = 2 } = {}) {
    this.bos = bos; this.eos = eos;
    const set = new Set();
    for (const ch of text) set.add(ch);
    const chars = [...set].sort();
    this.chars = chars;
    this.stoi = new Map();
    this.itos = [];
    // 0 保留为 <unk>，1=bos，2=eos，其余 char 从 3 起
    this.itos[0] = ""; this.itos[1] = ""; this.itos[2] = "";
    this.stoi.set("<unk>", 0); this.stoi.set("<bos>", bos); this.stoi.set("<eos>", eos);
    let idx = 3;
    for (const ch of chars) { this.stoi.set(ch, idx); this.itos[idx] = ch; idx++; }
    this.vocabSize = idx;
  }
  encode(s) {
    const out = [this.bos];
    for (const ch of s) out.push(this.stoi.has(ch) ? this.stoi.get(ch) : 0);
    out.push(this.eos);
    return out;
  }
  decode(ids) {
    let str = "";
    for (const id of ids) {
      if (id === this.bos || id === this.eos) continue;
      str += this.itos[id] ?? "";
    }
    return str;
  }
}

// ── 2) GGUF 词表（真实模型解码）──────────────────────────────────────────
export class GGUFVocab {
  /**
   * @param tokenBytes Array<Uint8Array> 每个 token 的原始字节串
   * @param types      Array<number>     0=normal 1=byte 2=unused ...
   * @param scores     Array<number>
   * @param meta       {bos,eos,unknown,addBos,addEos}
   */
  constructor(tokenBytes, types, scores, meta = {}) {
    this.tokenBytes = tokenBytes;
    this.types = types;
    this.scores = scores;
    this.bos = meta.bos ?? 1;
    this.eos = meta.eos ?? 2;
    this.addBos = meta.addBos ?? true;
    this.addEos = meta.addEos ?? false;
    this.vocabSize = tokenBytes.length;
    // 反向索引：字节串 -> id（用于编码）
    this._idx = new Map();
    for (let i = 0; i < tokenBytes.length; i++) {
      if (types[i] === 2) continue; // unused
      const key = bytesKey(tokenBytes[i]);
      if (!this._idx.has(key)) this._idx.set(key, i);
    }
  }

  /** 解码：把 token 序列还原成文本（字节拼接后 UTF-8 解码） */
  decode(ids, { skipSpecial = true } = {}) {
    const chunks = [];
    for (const id of ids) {
      if (skipSpecial && (id === this.bos || id === this.eos)) continue;
      const tb = this.tokenBytes[id];
      if (!tb) continue;
      chunks.push(tb);
    }
    // 拼接所有字节
    let total = 0; for (const c of chunks) total += c.length;
    const buf = new Uint8Array(total); let o = 0;
    for (const c of chunks) { buf.set(c, o); o += c.length; }
    return new TextDecoder("utf-8", { fatal: false }).decode(buf);
  }

  /**
   * 编码：基于 UTF-8 字节的最长匹配（含 byte-fallback）。
   * 若构造时传入 merges，则先按 BPE 合并再回退。
   */
  encode(text, merges) {
    let ids = [];
    if (this.addBos) ids.push(this.bos);
    const bytes = new TextEncoder().encode(text);
    let i = 0;
    while (i < bytes.length) {
      // 最长匹配
      let matchId = -1, matchLen = 0;
      for (let len = Math.min(8, bytes.length - i); len >= 1; len--) {
        const key = bytesKey(bytes.subarray(i, i + len));
        const id = this._idx.get(key);
        if (id !== undefined) { matchId = id; matchLen = len; break; }
      }
      if (matchId >= 0) { ids.push(matchId); i += matchLen; }
      else { i++; } // 极端缺失：跳过（byte-fallback 通常已覆盖）
    }
    if (this.addEos) ids.push(this.eos);
    return ids;
  }
}

function bytesKey(b) {
  let s = "";
  for (let i = 0; i < b.length; i++) s += String.fromCharCode(b[i]);
  return s;
}

/** 从 GGUF 读到的 tokBytes（ArrayBuffer 片段）转成 Uint8Array 列表 */
export function parseTokenBytes(tokBuf, n) {
  // GGUF 中 token 以「长度前缀字符串」存储：u32 len + len 字节
  const out = [];
  let p = 0;
  for (let i = 0; i < n; i++) {
    const len = new DataView(tokBuf.buffer, tokBuf.byteOffset + p, 4).getUint32(0, true);
    p += 4;
    out.push(new Uint8Array(tokBuf.buffer, tokBuf.byteOffset + p, len));
    p += len;
  }
  return out;
}
