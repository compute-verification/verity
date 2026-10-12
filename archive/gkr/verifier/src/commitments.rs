//! The commitment scheme of a committed-operands statement (`R+sha256`, `R+blake3`): scheme `frame-v3` of
//! `verity/ml/commitments/frame_v3` (its `PROTOCOL.md`; the framing is `merkle.py`'s, the row-leaf
//! schemas `rowleaf.py`'s), checked natively.  The proof publishes, per VU, the row digests of its x row and W column
//! as public words; this module rebuilds the tree leaves from them (and the `y` word leaves from the output words),
//! the three roots, and compares them with the pinned roots of the frozen input set.  The domains are derived here
//! from pinned fields only (B-Ligero's `hashauth.binding_digest`, so the roots are B-Ligero's for the same set and
//! leaf schema); nothing the prover sends names a domain.  One module per scheme: `vllm_v1.rs` (`R+vllm-v1`) is the
//! other, dispatched from `check`.
//!
//! `commitment.txt` (in the statement directory):
//!
//! ```text
//! scheme frame-v3|vllm-v1
//! leaf sha256|blake3|pos                   sha256/row/v1 | blake3-keyed/row/v2 (frame-v3); pos_leaf (vllm-v1)
//! relation R+sha256|R+blake3|R+vllm-v1
//! instances <dataset> <tier> <manifest_sha256>
//! lo L
//! vus N
//! K <words per row>
//! x <word bits>
//! w <word bits>
//! y u16|u32 <words per VU>
//! domain_a|domain_b|domain_y <hex>         vllm-v1 only: each port's domain_digest, which must be the derived one
//! ```
//!
//! Its SHA-256 is absorbed into the transcript after the binding digests (`verify::Statement::commit`).
//! Each VU's public words are its `y` words, then 16 limbs of its x row digest, then 16 of its W column digest
//! (limb `i` = digest bytes `2i, 2i+1` big-endian).  VU `v` is rank `v` of trees `a` (x rows, owner -1) and `b`
//! (W columns, owner -2); its `y` words are ranks `n_y v ..` of tree `y` (owner -1, `u16`/`u32` big-endian leaves).

use crate::sha256::Sha256;

pub type Digest = [u8; 32];

pub const SCHEME: &str = "frame-v3";
const FRAME: &[u8] = b"veritor/protocol/merkle/frame/v3\0";
const TAGGED: &[u8] = b"veritor/tagged-sha256/v1\0";
pub const ROLE_X: u8 = 1;
pub const ROLE_W: u8 = 2;
pub const LIMBS: usize = 16;
/// `archive/direct/ligero/hashauth.py BINDING_TAG`
pub const BINDING_TAG: &str = "verity/ligero-b/auth-binding/v2h";
const OWNER_A: i64 = -1;
const OWNER_B: i64 = -2;
const OWNER_Y: i64 = -1;

/// A value of a flat identity manifest (`identity.canonical_json` over string / integer fields).
pub enum Json<'a> { Int(u64), Str(&'a str), Null }

fn json_str(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Sorted keys (UTF-8 byte order = code point order), `,` / `:` separators, no whitespace, UTF-8 as is.
pub fn canonical_json(fields: &[(&str, Json)]) -> String {
    let mut f: Vec<&(&str, Json)> = fields.iter().collect();
    f.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let mut out = String::from("{");
    for (i, (k, v)) in f.iter().enumerate() {
        if i > 0 { out.push(','); }
        json_str(k, &mut out);
        out.push(':');
        match v { Json::Int(n) => out.push_str(&n.to_string()), Json::Str(s) => json_str(s, &mut out), Json::Null => out.push_str("null") }
    }
    out.push('}');
    out
}

pub fn tagged_sha256(tag: &str, payload: &[u8]) -> Digest {
    let mut h = Sha256::new();
    h.update(TAGGED);
    h.update(&(tag.len() as u32).to_be_bytes());
    h.update(tag.as_bytes());
    h.update(&(payload.len() as u64).to_be_bytes());
    h.update(payload);
    h.finalize()
}

pub fn identity_digest(tag: &str, fields: &[(&str, Json)]) -> Digest { tagged_sha256(tag, canonical_json(fields).as_bytes()) }

/// `merkle._uint`: minimal big-endian, one byte for zero.
fn uint(v: u64) -> Vec<u8> {
    let b = v.to_be_bytes();
    let skip = b.iter().take_while(|&&x| x == 0).count().min(7);
    b[skip..].to_vec()
}

fn frame(tag: &[u8], parts: &[&[u8]]) -> Digest {
    let mut h = Sha256::new();
    h.update(FRAME);
    h.update(&(tag.len() as u32).to_be_bytes());
    h.update(tag);
    for p in parts {
        h.update(&(p.len() as u64).to_be_bytes());
        h.update(p);
    }
    h.finalize()
}

/// `merkle_depth`
pub fn depth(count: u64) -> u32 { if count <= 1 { 0 } else { 64 - (count - 1).leading_zeros() } }

/// A `CommitmentDomain` over `RangeIndexedDomain(0, count)`: rank = position.
pub struct Domain { pub id: Digest, pub count: u64 }

impl Domain {
    pub fn range(binding: &Digest, owner: i64, count: u64) -> Domain {
        assert!(owner >= -2, "owner -2 (weights), -1 (boundary) or a replay unit");
        let positions = identity_digest("veritor/indexed-domain/range/v1",
                                        &[("start", Json::Int(0)), ("step", Json::Int(1)), ("stop", Json::Int(count))]);
        let id = frame(b"domain", &[binding, &uint((owner + 2) as u64), &positions, &uint(count)]);
        Domain { id, count }
    }
    pub fn leaf(&self, rank: u64, position: u64, schema: &str, value: &[u8]) -> Digest {
        frame(b"leaf", &[&self.id, &uint(rank), &uint(position), schema.as_bytes(), value])
    }
    pub fn node(&self, level: u32, index: u64, left: &Digest, right: &Digest) -> Digest {
        frame(b"node", &[&self.id, &uint(level as u64), &uint(index), left, right])
    }
    pub fn pad(&self, rank: u64) -> Digest { frame(b"pad", &[&self.id, &uint(rank)]) }
    pub fn empty(&self) -> Digest { frame(b"empty", &[&self.id]) }

    /// The levels over the leaves in rank order, padded to a power of two (`merkle.tree_levels`).
    fn levels(&self, leaves: &[Digest]) -> Result<Vec<Vec<Digest>>, String> {
        if leaves.len() as u64 != self.count { return Err(format!("expected {} leaves, got {}", self.count, leaves.len())); }
        if leaves.is_empty() { return Ok(vec![]); }
        let width = 1u64 << depth(self.count);
        let mut level: Vec<Digest> = leaves.to_vec();
        level.extend((self.count..width).map(|r| self.pad(r)));
        let mut out = vec![level];
        let mut d = 0;
        while out.last().unwrap().len() > 1 {
            let prev = out.last().unwrap();
            let next = (0..prev.len() / 2).map(|i| self.node(d, i as u64, &prev[2 * i], &prev[2 * i + 1])).collect();
            out.push(next);
            d += 1;
        }
        Ok(out)
    }
    pub fn root(&self, leaves: &[Digest]) -> Result<Digest, String> {
        Ok(self.levels(leaves)?.last().map(|l| l[0]).unwrap_or_else(|| self.empty()))
    }
    pub fn path(&self, leaves: &[Digest], mut index: usize) -> Result<Vec<Digest>, String> {
        let lv = self.levels(leaves)?;
        if index >= leaves.len() { return Err(format!("index {index} out of range for {} leaves", leaves.len())); }
        let mut out = Vec::new();
        for level in &lv[..lv.len() - 1] { out.push(level[index ^ 1]); index >>= 1; }
        Ok(out)
    }
    pub fn verify_path(&self, leaf: &Digest, index: u64, path: &[Digest], root: &Digest) -> bool {
        if index >= self.count || path.len() as u32 != depth(self.count) { return false; }
        let (mut acc, mut cur) = (*leaf, index);
        for (level, sib) in path.iter().enumerate() {
            acc = if cur & 1 == 0 { self.node(level as u32, cur >> 1, &acc, sib) } else { self.node(level as u32, cur >> 1, sib, &acc) };
            cur >>= 1;
        }
        &acc == root
    }
}

/// The row leaf rule: frame-v3's two row schemas, or vllm-v1's position leaf (`vllm_v1.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowHash { Sha256, Blake3, VllmPos }

impl RowHash {
    pub fn parse(s: &str) -> Option<RowHash> {
        match s { "sha256" => Some(RowHash::Sha256), "blake3" => Some(RowHash::Blake3), "pos" => Some(RowHash::VllmPos), _ => None }
    }
    pub fn name(self) -> &'static str { match self { RowHash::Sha256 => "sha256", RowHash::Blake3 => "blake3", RowHash::VllmPos => "pos" } }
    /// `rowleaf.SCHEMA_SHA256_ROW` / `SCHEMA_BLAKE3_ROW`; vllm-v1 leaves carry no schema (a label only)
    pub fn schema(self) -> &'static str {
        match self { RowHash::Sha256 => "sha256/row/v1", RowHash::Blake3 => "blake3-keyed/row/v2", RowHash::VllmPos => "vllm-v1/pos-leaf/v0" }
    }
    /// The relation suffix naming a statement committed under this leaf.
    pub fn suffix(self) -> &'static str { match self { RowHash::Sha256 => "+sha256", RowHash::Blake3 => "+blake3", RowHash::VllmPos => "+vllm-v1" } }
    pub fn scheme(self) -> &'static str { if self == RowHash::VllmPos { crate::vllm_v1::SCHEME } else { SCHEME } }
}

/// The relation suffix of a committed-operands relation name, if any.
pub fn relation_hash(relation: &str) -> Option<RowHash> {
    [RowHash::Sha256, RowHash::Blake3, RowHash::VllmPos].into_iter().find(|h| relation.ends_with(h.suffix()))
}

/// `rowleaf.sha256_row_prefix`: `"verity/sha256-row/v1\0" || role || word_bits || u32be(n_words)`, zero-padded to 64.
pub fn sha256_row_prefix(role: u8, word_bits: u8, n_words: u32) -> [u8; 64] {
    let mut p = [0u8; 64];
    let tag = b"verity/sha256-row/v1\0";
    p[..tag.len()].copy_from_slice(tag);
    p[tag.len()] = role;
    p[tag.len() + 1] = word_bits;
    p[tag.len() + 2..tag.len() + 6].copy_from_slice(&n_words.to_be_bytes());
    p
}

/// `rowleaf` BLAKE3 key: `"verity/blake3-leaf/v1/" + "x"|"w"`, zero-padded to 32.
pub fn blake3_row_key(role: u8) -> [u8; 32] {
    let mut k = [0u8; 32];
    let tag = b"verity/blake3-leaf/v1/";
    k[..tag.len()].copy_from_slice(tag);
    k[tag.len()] = if role == ROLE_X { b'x' } else { b'w' };
    k
}

/// `sha256/row/v1` of a row's little-endian words (the verifier never sees rows: the harness and the tests do).
pub fn sha256_row_digest(role: u8, word_bits: u8, n_words: u32, row_bytes: &[u8]) -> Digest {
    let mut h = Sha256::new();
    h.update(&sha256_row_prefix(role, word_bits, n_words));
    h.update(row_bytes);
    h.finalize()
}

/// The tree leaf of a row digest at `rank`.
pub fn row_leaf(d: &Domain, rank: u64, hash: RowHash, digest: &Digest) -> Digest { d.leaf(rank, rank, hash.schema(), digest) }

pub fn word_schema(width: usize) -> &'static str { if width == 2 { "u16" } else { "u32" } }

/// The tree leaf of an output word (`auth.py` word leaves: 2 or 4 bytes big-endian).
pub fn word_leaf(d: &Domain, position: u64, width: usize, word: u32) -> Digest {
    d.leaf(position, position, word_schema(width), &word.to_be_bytes()[4 - width..])
}

/// `hashauth.binding_digest`: what the committed object may not outlive.
#[allow(clippy::too_many_arguments)]
pub fn binding(dataset: &str, tier: &str, manifest_sha256: &str, lo: u64, hi: u64, k: u64, tree: &str, schema: &str) -> Digest {
    identity_digest(BINDING_TAG, &[("dataset", Json::Str(dataset)), ("tier", Json::Str(tier)), ("manifest_sha256", Json::Str(manifest_sha256)),
                                   ("lo", Json::Int(lo)), ("hi", Json::Int(hi)), ("K", Json::Int(k)), ("tree", Json::Str(tree)), ("schema", Json::Str(schema))])
}

pub fn hex(d: &[u8]) -> String { d.iter().map(|b| format!("{b:02x}")).collect() }

pub fn unhex32(s: &str) -> Option<Digest> {
    if s.len() != 64 { return None; }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() { *o = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?; }
    Some(out)
}

/// A digest from its 16 public limbs; `None` if a limb is not below `2^16`.
pub fn from_limbs(limbs: &[u32]) -> Option<Digest> {
    if limbs.len() != LIMBS || limbs.iter().any(|&l| l >> 16 != 0) { return None; }
    let mut out = [0u8; 32];
    for (i, &l) in limbs.iter().enumerate() { out[2 * i..2 * i + 2].copy_from_slice(&(l as u16).to_be_bytes()); }
    Some(out)
}

/// The pinned roots of a frozen input set under one leaf schema (`python -m gpu.commit pin`).
pub struct Pin {
    pub relation: &'static str,
    pub lo: u64,
    pub vus: u64,
    pub dataset: &'static str,
    pub tier: &'static str,
    pub manifest_sha256: &'static str,
    pub k: u64,
    pub x_bits: u8,
    pub w_bits: u8,
    pub y_width: usize,
    pub n_y: usize,
    pub a: &'static str,
    pub b: &'static str,
    pub y: &'static str,
    pub source: &'static str,
}

pub const PINS: &[Pin] = &[
    Pin {
        relation: "bf16-ampere+sha256", lo: 0, vus: 4096,
        dataset: "bench-instances/v1", tier: "vu-k1536",
        manifest_sha256: "059103cf9bd55ee83cbd4bb14ae6db2f60db2cb4ddf85cdc22b1cecee6e4eeea",
        k: 1536, x_bits: 16, w_bits: 16, y_width: 2, n_y: 1,
        a: "780aae0ac7471dfdbd3a5e4ec0ac7204ece23576cf55bdfb0d179c9c21c3b128",
        b: "1e8c3fd306ecf6a02be71a1306aac2b0dddb823e1b53f4f0f0866e843adaaa82",
        y: "9fff0fdc3d33d27fd358d42111e3a62856f1cf56cc1a81e2a9efab19adf0a0f6",
        source: "fixtures/bench-instances/v1/manifest.json tier vu-k1536 (python -m gpu.commit pin bf16-ampere --leaf sha256)",
    },
    Pin {
        relation: "fp8-hopper+blake3", lo: 0, vus: 4096,
        dataset: "bench-instances-fp8-hopper/v1", tier: "vu-k1536-fp8-hopper",
        manifest_sha256: "0ff750026b8d8df0418d3ec360ed6ade683a33b8264d66973021cbfbff7945bc",
        k: 1536, x_bits: 8, w_bits: 8, y_width: 4, n_y: 1,
        a: "300b209243d767f444a3531c05cfd9da3fb73d7dab216d0a8f3fe73889c24ec1",
        b: "69740dc3f6c06ed566b7da893472a7fa7d4b23236571086dca09f300f47d0347",
        y: "22fa3c6e7e514a35afb9c5d3041343179467e4338ee32ce3158d28d1885b70d1",
        source: "archive/direct/ligero/relchain.py instances() seed 20260922 (python -m gpu.commit pin fp8-hopper --leaf blake3)",
    },
    Pin {
        relation: "bf16-ampere+blake3", lo: 0, vus: 4096,
        dataset: "bench-instances/v1", tier: "vu-k1536",
        manifest_sha256: "059103cf9bd55ee83cbd4bb14ae6db2f60db2cb4ddf85cdc22b1cecee6e4eeea",
        k: 1536, x_bits: 16, w_bits: 16, y_width: 2, n_y: 1,
        a: "4867030e7e92eb5fcd1220660491bd0620888810e28888ab08401c6dcff547fa",
        b: "67137b46a2979f734711c47244abb6375ea23a7f2bbd12add62c5c524be044fd",
        y: "9fff0fdc3d33d27fd358d42111e3a62856f1cf56cc1a81e2a9efab19adf0a0f6",
        source: "fixtures/bench-instances/v1/manifest.json tier vu-k1536 (python -m gpu.commit pin bf16-ampere --leaf blake3)",
    },
    Pin {
        relation: "bf16-ampere+vllm-v1", lo: 0, vus: 4096,
        dataset: "bench-instances/v1", tier: "vu-k1536",
        manifest_sha256: "059103cf9bd55ee83cbd4bb14ae6db2f60db2cb4ddf85cdc22b1cecee6e4eeea",
        k: 1536, x_bits: 16, w_bits: 16, y_width: 2, n_y: 1,
        a: "d9be865a18fbdc55e39e38b00c72fdeafd9cb723a26b946eceef28ed51bf94bf",
        b: "8916cf1e9895f8a5fee9b03dcf806862fbbe1bd605f150d16b0ef12768e7d1ef",
        y: "ad6d8cec08637127585160ae50e449417c083c85c85a5eeb8eac23ef3c287ccf",
        source: "fixtures/bench-instances/v1/manifest.json tier vu-k1536 (python -m gpu.commit pin bf16-ampere --leaf vllm-v1)",
    },
    Pin {
        relation: "fp8-hopper+vllm-v1", lo: 0, vus: 4096,
        dataset: "bench-instances-fp8-hopper/v1", tier: "vu-k1536-fp8-hopper",
        manifest_sha256: "0ff750026b8d8df0418d3ec360ed6ade683a33b8264d66973021cbfbff7945bc",
        k: 1536, x_bits: 8, w_bits: 8, y_width: 4, n_y: 1,
        a: "da5f37a02f479bdc5609efa2ca067fec71edb1fa69e1eabe9ed732c1739bc6b7",
        b: "c77882c72b781a6e1edd26653b495f41c92a55db875ac99dede7424a83b0160e",
        y: "86b0dcaef4618ec50b7c7ec7717eadfc81472f26000b9c341d33c1708ff94657",
        source: "archive/direct/ligero/relchain.py instances() seed 20260922 (python -m gpu.commit pin fp8-hopper --leaf vllm-v1)",
    },    // route (a) at vLLM's reduction lengths: VUs [0, vus) of the #101 captured sets, the sizes a cell's sweep runs
    Pin {
        relation: "bf16-ampere-k2048+blake3", lo: 0, vus: 1024,
        dataset: "vllm-vu-set/v1:gemm-coordinate-ampere-bf16-k2048", tier: "gemm-coordinate-ampere-bf16-k2048",
        manifest_sha256: "b2911acffe4dc63321cd7d2a05eef1aa2234648b2cb76a1d66d0d458ee352dff",
        k: 2048, x_bits: 16, w_bits: 16, y_width: 2, n_y: 1,
        a: "4bc72a389e0433194919f9e826f827cd8fb009a4fd8c67826c9e155222db105f",
        b: "5f3ddb8f0685d895ee32a0343ed0df4f3e3f43a28cd1cbe12ae6242041b27a2c",
        y: "d6bbaad5afcdfeba38d80f312bb3412e9da389203a9f47aaa9460222c55e20cb",
        source: "input set gemm-coordinate-ampere-bf16-k2048 (captured; content digest 49d7658f823ab777...) (python -m gpu.commit pin bf16-ampere-k2048 --leaf blake3 --input-set ... --vus 1024)",
    },
    Pin {
        relation: "bf16-ampere-k2048+blake3", lo: 0, vus: 2048,
        dataset: "vllm-vu-set/v1:gemm-coordinate-ampere-bf16-k2048", tier: "gemm-coordinate-ampere-bf16-k2048",
        manifest_sha256: "b2911acffe4dc63321cd7d2a05eef1aa2234648b2cb76a1d66d0d458ee352dff",
        k: 2048, x_bits: 16, w_bits: 16, y_width: 2, n_y: 1,
        a: "b749140a8c0e67a69442e07a657a1cb3425fd10eb951fc982c2f849590f140c8",
        b: "969173dfe3492ad8f8ea2f853ecc9daf2ab52cfa824cb15284e1fb223c986f94",
        y: "0843ac999a3331faffb8efbb8d6e0413a60dc539517f44b88899cd7e9cbcb3c3",
        source: "input set gemm-coordinate-ampere-bf16-k2048 (captured; content digest 49d7658f823ab777...) (python -m gpu.commit pin bf16-ampere-k2048 --leaf blake3 --input-set ... --vus 2048)",
    },
    Pin {
        relation: "bf16-ampere-k2048+blake3", lo: 0, vus: 4096,
        dataset: "vllm-vu-set/v1:gemm-coordinate-ampere-bf16-k2048", tier: "gemm-coordinate-ampere-bf16-k2048",
        manifest_sha256: "b2911acffe4dc63321cd7d2a05eef1aa2234648b2cb76a1d66d0d458ee352dff",
        k: 2048, x_bits: 16, w_bits: 16, y_width: 2, n_y: 1,
        a: "eaec873cb367e6d2891b0b9145d577c80f7fdb1c901300ea64fef95cb71c6cfe",
        b: "8b045679097a3e20ee1261284552e185aa111ed1850e351db23276258a4f688c",
        y: "eff79558cec0e6b059d741ac5d1c2a418de2e7318ba2df77d1db7d71f918f5db",
        source: "input set gemm-coordinate-ampere-bf16-k2048 (captured; content digest 49d7658f823ab777...) (python -m gpu.commit pin bf16-ampere-k2048 --leaf blake3 --input-set ... --vus 4096)",
    },
    Pin {
        relation: "bf16-ampere-k2048+blake3", lo: 0, vus: 6272,
        dataset: "vllm-vu-set/v1:gemm-coordinate-ampere-bf16-k2048", tier: "gemm-coordinate-ampere-bf16-k2048",
        manifest_sha256: "b2911acffe4dc63321cd7d2a05eef1aa2234648b2cb76a1d66d0d458ee352dff",
        k: 2048, x_bits: 16, w_bits: 16, y_width: 2, n_y: 1,
        a: "50fb9a067529173128872b432cf621e6c4da9813188f15717fc8da130b4ba44e",
        b: "52c1370690c517d360dbe27de8996ff74469f6f6e7b85b6c78c7bf883cccd156",
        y: "e35be3a50bff58606932ef1e62ba1003c38aca7ba98b1365ac9817acc8c1d08d",
        source: "input set gemm-coordinate-ampere-bf16-k2048 (captured; content digest 49d7658f823ab777...) (python -m gpu.commit pin bf16-ampere-k2048 --leaf blake3 --input-set ... --vus 6272)",
    },
    Pin {
        relation: "bf16-ampere-k8192+blake3", lo: 0, vus: 256,
        dataset: "vllm-vu-set/v1:gemm-coordinate-ampere-bf16-k8192", tier: "gemm-coordinate-ampere-bf16-k8192",
        manifest_sha256: "9f24ed608f42c32052512d4881ecb4b480d130af8606dcbd4137bf3cfe869518",
        k: 8192, x_bits: 16, w_bits: 16, y_width: 2, n_y: 1,
        a: "58f89a9c76f8828e47c0b49d53d5429ce9a1e937d78699a9fb9a5e278f85a23e",
        b: "07604f04fbb0a2cc0a79f94e67d1538c69efe9c51ef7c9ad8623636f127ac6b1",
        y: "4a7e70d475c5bdb852f0caa66e64a56e66dba80443f3f334ac8fa93ffcaad875",
        source: "input set gemm-coordinate-ampere-bf16-k8192 (captured; content digest 282e42457833e1a1...) (python -m gpu.commit pin bf16-ampere-k8192 --leaf blake3 --input-set ... --vus 256)",
    },
    Pin {
        relation: "bf16-ampere-k8192+blake3", lo: 0, vus: 512,
        dataset: "vllm-vu-set/v1:gemm-coordinate-ampere-bf16-k8192", tier: "gemm-coordinate-ampere-bf16-k8192",
        manifest_sha256: "9f24ed608f42c32052512d4881ecb4b480d130af8606dcbd4137bf3cfe869518",
        k: 8192, x_bits: 16, w_bits: 16, y_width: 2, n_y: 1,
        a: "a5173e851397e4c24e218a1f7a691c136f094d455a6bcb0f40695df94e387b23",
        b: "2a40cc8aefbbb51cf1256fc7128797cf5ae95ed11223362136c1afeebff6804a",
        y: "6c0ee9bed5545a6b35a44d6d27e5f235687b396ed4245a79ff496e076353f0f5",
        source: "input set gemm-coordinate-ampere-bf16-k8192 (captured; content digest 282e42457833e1a1...) (python -m gpu.commit pin bf16-ampere-k8192 --leaf blake3 --input-set ... --vus 512)",
    },
    Pin {
        relation: "bf16-ampere-k8192+blake3", lo: 0, vus: 1024,
        dataset: "vllm-vu-set/v1:gemm-coordinate-ampere-bf16-k8192", tier: "gemm-coordinate-ampere-bf16-k8192",
        manifest_sha256: "9f24ed608f42c32052512d4881ecb4b480d130af8606dcbd4137bf3cfe869518",
        k: 8192, x_bits: 16, w_bits: 16, y_width: 2, n_y: 1,
        a: "cfb749cec03b05304ddbcbead56b717716802d92250d7b7e8dc034bde92d1991",
        b: "63ee5637f160e9c1878b586f2aa636277eba766fbc0278bc0f92a17d82b3cbed",
        y: "e28cdbb18dc249bce5497a598b3d30087ac1ffb2fd6321b64566bc798caeb719",
        source: "input set gemm-coordinate-ampere-bf16-k8192 (captured; content digest 282e42457833e1a1...) (python -m gpu.commit pin bf16-ampere-k8192 --leaf blake3 --input-set ... --vus 1024)",
    },
    Pin {
        relation: "bf16-ampere-k8192+blake3", lo: 0, vus: 1920,
        dataset: "vllm-vu-set/v1:gemm-coordinate-ampere-bf16-k8192", tier: "gemm-coordinate-ampere-bf16-k8192",
        manifest_sha256: "9f24ed608f42c32052512d4881ecb4b480d130af8606dcbd4137bf3cfe869518",
        k: 8192, x_bits: 16, w_bits: 16, y_width: 2, n_y: 1,
        a: "b9679506ef92b4c2a2361e93e0ce99fe363938c02df97cfc8ff9e151d884efe9",
        b: "2243341c7422a46111d9d8dcc1058412d943e6c8a8f00ad072d0bb9aa03b80e8",
        y: "099488d45ec77d1ecb97bedcb73ee622a22c705c15acd3c50203d3d68e43c32e",
        source: "input set gemm-coordinate-ampere-bf16-k8192 (captured; content digest 282e42457833e1a1...) (python -m gpu.commit pin bf16-ampere-k8192 --leaf blake3 --input-set ... --vus 1920)",
    },
];

pub fn find(relation: &str, lo: u64, vus: u64) -> Option<&'static Pin> {
    PINS.iter().find(|p| p.relation == relation && p.lo == lo && p.vus == vus)
}

/// `commitment.txt`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spec {
    pub hash: RowHash,
    pub relation: String,
    pub dataset: String,
    pub tier: String,
    pub manifest_sha256: String,
    pub lo: u64,
    pub vus: u64,
    pub k: u64,
    pub x_bits: u8,
    pub w_bits: u8,
    pub y_width: usize,
    pub n_y: usize,
    /// vllm-v1 only: the statement's `domain_a` / `domain_b` / `domain_y` digests (`vllm_v1::StepDomain::digest`)
    pub domains: Option<[Digest; 3]>,
}

impl Spec {
    pub fn parse(text: &str) -> Result<Spec, String> {
        let mut kv: std::collections::HashMap<&str, Vec<&str>> = std::collections::HashMap::new();
        for line in text.lines() {
            let t: Vec<&str> = line.split_whitespace().collect();
            let Some((k, v)) = t.split_first() else { continue };
            if kv.insert(k, v.to_vec()).is_some() { return Err(format!("commitment.txt: {k} twice")); }
        }
        const DOMAINS: [&str; 3] = ["domain_a", "domain_b", "domain_y"];
        let expect = ["scheme", "leaf", "relation", "instances", "lo", "vus", "K", "x", "w", "y"];
        if let Some(k) = kv.keys().find(|k| !expect.contains(k) && !DOMAINS.contains(k)) { return Err(format!("commitment.txt: unknown line {k}")); }
        let get = |k: &str, i: usize| -> Result<&str, String> { kv.get(k).and_then(|v| v.get(i).copied()).ok_or_else(|| format!("commitment.txt: {k}")) };
        let num = |k: &str, i: usize| -> Result<u64, String> { get(k, i)?.parse().map_err(|_| format!("commitment.txt: {k} {:?}", get(k, i))) };
        let bits = |k: &str| -> Result<u8, String> { match num(k, 0)? { b @ (8 | 16) => Ok(b as u8), b => Err(format!("commitment.txt: {k} words of {b} bits")) } };
        let hash = RowHash::parse(get("leaf", 0)?).ok_or_else(|| format!("commitment.txt: leaf {}", get("leaf", 0).unwrap_or("")))?;
        if get("scheme", 0)? != hash.scheme() {
            return Err(format!("commitment.txt: scheme {}, leaf {} is a {} leaf", get("scheme", 0)?, hash.name(), hash.scheme()));
        }
        let domains = if hash == RowHash::VllmPos {
            let d = |k: &str| get(k, 0).ok().and_then(unhex32).ok_or_else(|| format!("commitment.txt: {k} (a vllm-v1 statement names each port's domain)"));
            Some([d(DOMAINS[0])?, d(DOMAINS[1])?, d(DOMAINS[2])?])
        } else {
            if let Some(k) = DOMAINS.iter().find(|k| kv.contains_key(*k)) { return Err(format!("commitment.txt: {k} in a {SCHEME} statement")); }
            None
        };
        let relation = get("relation", 0)?.to_string();
        if relation_hash(&relation) != Some(hash) { return Err(format!("commitment.txt: relation {relation} is not a {} relation", hash.suffix())); }
        let y_width = match get("y", 0)? { "u16" => 2, "u32" => 4, s => return Err(format!("commitment.txt: y {s}")) };
        let manifest_sha256 = get("instances", 2)?.to_string();
        if unhex32(&manifest_sha256).is_none() { return Err("commitment.txt: instances manifest digest".into()); }
        let s = Spec {
            hash, relation, dataset: get("instances", 0)?.to_string(), tier: get("instances", 1)?.to_string(), manifest_sha256,
            lo: num("lo", 0)?, vus: num("vus", 0)?, k: num("K", 0)?, x_bits: bits("x")?, w_bits: bits("w")?, y_width, n_y: num("y", 1)? as usize,
            domains,
        };
        if s.vus == 0 || s.k == 0 || s.n_y == 0 { return Err("commitment.txt: vus, K and the y words per VU must be positive".into()); }
        Ok(s)
    }

    /// Public words per VU: the `y` words and the two digests.
    pub fn words_per_vu(&self) -> usize { self.n_y + 2 * LIMBS }

    pub fn bindings(&self) -> [Digest; 3] {
        let hi = self.lo + self.vus;
        let b = |tree: &str, schema: &str| binding(&self.dataset, &self.tier, &self.manifest_sha256, self.lo, hi, self.k, tree, schema);
        [b("a", self.hash.schema()), b("b", self.hash.schema()), b("y", word_schema(self.y_width))]
    }

    fn pin_problem(&self, p: &Pin) -> Option<String> {
        let same = p.dataset == self.dataset && p.tier == self.tier && p.manifest_sha256 == self.manifest_sha256 && p.k == self.k
            && p.x_bits == self.x_bits && p.w_bits == self.w_bits && p.y_width == self.y_width && p.n_y == self.n_y;
        (!same).then(|| format!("commitment: commitment.txt is not the instance set pinned for {} [{}, {})", p.relation, p.lo, p.lo + p.vus))
    }
}

/// What the tree check found.
pub struct Checked { pub spec: Spec, pub roots: [Digest; 3], pub pin: Option<&'static Pin> }

impl Checked {
    /// `None` when the roots are the pinned roots (or `allow_unpinned` and the tree check itself passed).
    pub fn problem(&self, allow_unpinned: bool) -> Option<String> {
        let Some(p) = self.pin else {
            return (!allow_unpinned).then(|| format!("commitment: {} [{}, {}) is not a pinned instance commitment", self.spec.relation, self.spec.lo, self.spec.lo + self.spec.vus));
        };
        if let Some(e) = self.spec.pin_problem(p) { return Some(e); }
        for (name, got, want) in [("a", &self.roots[0], p.a), ("b", &self.roots[1], p.b), ("y", &self.roots[2], p.y)] {
            if hex(got) != want { return Some(format!("commitment: tree {name} root {} is not the pinned {want}", hex(got))); }
        }
        None
    }
    pub fn pinned(&self) -> bool { self.pin.is_some() && self.problem(false).is_none() }
    pub fn json(&self) -> String {
        let domains = self.spec.domains.map(|d| format!(", \"domain_a\": \"{}\", \"domain_b\": \"{}\", \"domain_y\": \"{}\"", hex(&d[0]), hex(&d[1]), hex(&d[2])))
            .unwrap_or_default();
        format!("{{\"scheme\": \"{}\", \"leaf\": \"{}\", \"relation\": \"{}\", \"lo\": {}, \"vus\": {}, \"root_a\": \"{}\", \"root_b\": \"{}\", \"root_y\": \"{}\"{domains}, \"pinned\": {}, \"pin_source\": {}}}",
                self.spec.hash.scheme(), self.spec.hash.schema(), self.spec.relation, self.spec.lo, self.spec.vus, hex(&self.roots[0]), hex(&self.roots[1]),
                hex(&self.roots[2]), self.pinned(), self.pin.map(|p| format!("\"{}\"", p.source)).unwrap_or("null".into()))
    }
}

/// vllm-v1: the digests are the tree leaves, each root is its verifier-derived step domain's binding.
fn check_vllm(spec: &Spec, public: &[u32]) -> Result<[Digest; 3], String> {
    use crate::vllm_v1 as V;
    let doms: Vec<V::StepDomain> = V::PORTS.iter().map(|p| V::operand_domain(spec, p)).collect();
    let named = spec.domains.ok_or("commitment: a vllm-v1 statement without its domains")?;
    for ((port, dom), want) in V::PORTS.iter().zip(&doms).zip(&named) {
        if &dom.digest() != want { return Err(format!("commitment: domain_{port} {} is not this instance set's {}", hex(want), hex(&dom.digest()))); }
    }
    let m = spec.words_per_vu();
    let (mut la, mut lb, mut ly) = (Vec::new(), Vec::new(), Vec::new());
    for (v, words) in public.chunks_exact(m).enumerate() {
        for &y in &words[..spec.n_y] {
            if spec.y_width < 4 && y >> (8 * spec.y_width) != 0 { return Err(format!("commitment: VU {v}: a y word exceeds {}", word_schema(spec.y_width))); }
            ly.push(V::word_leaf(spec.y_width, y));
        }
        la.push(from_limbs(&words[spec.n_y..spec.n_y + LIMBS]).ok_or_else(|| format!("commitment: VU {v}: an x digest limb is not below 2^16"))?);
        lb.push(from_limbs(&words[spec.n_y + LIMBS..]).ok_or_else(|| format!("commitment: VU {v}: a W digest limb is not below 2^16"))?);
    }
    Ok([doms[0].root(&la)?, doms[1].root(&lb)?, doms[2].root(&ly)?])
}

/// The three roots of the statement's public words (`words_per_vu` per VU, VU-major).  An error is a malformed
/// statement (a limb of 16 bits or more, a `y` word wider than its schema, a count mismatch).
pub fn check(spec: Spec, public: &[u32]) -> Result<Checked, String> {
    let m = spec.words_per_vu();
    if public.len() as u64 != spec.vus * m as u64 {
        return Err(format!("commitment: {} public words, {} VUs of {m} expected", public.len(), spec.vus));
    }
    if spec.hash == RowHash::VllmPos {
        let roots = check_vllm(&spec, public)?;
        let pin = find(&spec.relation, spec.lo, spec.vus);
        return Ok(Checked { spec, roots, pin });
    }
    let [ba, bb, by] = spec.bindings();
    let (da, db) = (Domain::range(&ba, OWNER_A, spec.vus), Domain::range(&bb, OWNER_B, spec.vus));
    let dy = Domain::range(&by, OWNER_Y, spec.vus * spec.n_y as u64);
    let (mut la, mut lb, mut ly) = (Vec::new(), Vec::new(), Vec::new());
    for (v, words) in public.chunks_exact(m).enumerate() {
        for (j, &y) in words[..spec.n_y].iter().enumerate() {
            if spec.y_width < 4 && y >> (8 * spec.y_width) != 0 { return Err(format!("commitment: VU {v}: a y word exceeds {}", word_schema(spec.y_width))); }
            ly.push(word_leaf(&dy, (v * spec.n_y + j) as u64, spec.y_width, y));
        }
        let x = from_limbs(&words[spec.n_y..spec.n_y + LIMBS]).ok_or_else(|| format!("commitment: VU {v}: an x digest limb is not below 2^16"))?;
        let w = from_limbs(&words[spec.n_y + LIMBS..]).ok_or_else(|| format!("commitment: VU {v}: a W digest limb is not below 2^16"))?;
        la.push(row_leaf(&da, v as u64, spec.hash, &x));
        lb.push(row_leaf(&db, v as u64, spec.hash, &w));
    }
    let roots = [da.root(&la)?, db.root(&lb)?, dy.root(&ly)?];
    let pin = find(&spec.relation, spec.lo, spec.vus);
    Ok(Checked { spec, roots, pin })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Just enough JSON for `vectors.json` (objects, arrays, strings without escapes, integers, booleans, null).
    #[derive(Debug)]
    pub enum J { O(Vec<(String, J)>), A(Vec<J>), S(String), N(i64), B(bool), Null }

    impl J {
        pub fn get(&self, k: &str) -> &J { match self { J::O(f) => &f.iter().find(|(n, _)| n == k).unwrap_or_else(|| panic!("no key {k}")).1, _ => panic!("not an object") } }
        pub fn arr(&self) -> &[J] { match self { J::A(v) => v, _ => panic!("not an array") } }
        pub fn s(&self) -> &str { match self { J::S(s) => s, _ => panic!("not a string") } }
        pub fn n(&self) -> i64 { match self { J::N(n) => *n, _ => panic!("not a number") } }
        pub fn bytes(&self) -> Vec<u8> { let s = self.s(); (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect() }
        pub fn d32(&self) -> Digest { unhex32(self.s()).unwrap() }
    }

    pub fn parse_json(text: &str) -> J { parse(text.as_bytes(), &mut 0) }

    fn parse(s: &[u8], i: &mut usize) -> J {
        while s[*i].is_ascii_whitespace() { *i += 1; }
        match s[*i] {
            b'{' => {
                *i += 1;
                let mut f = Vec::new();
                loop {
                    while s[*i].is_ascii_whitespace() || s[*i] == b',' { *i += 1; }
                    if s[*i] == b'}' { *i += 1; return J::O(f); }
                    let J::S(k) = parse(s, i) else { panic!("key") };
                    while s[*i].is_ascii_whitespace() || s[*i] == b':' { *i += 1; }
                    f.push((k, parse(s, i)));
                }
            }
            b'[' => {
                *i += 1;
                let mut v = Vec::new();
                loop {
                    while s[*i].is_ascii_whitespace() || s[*i] == b',' { *i += 1; }
                    if s[*i] == b']' { *i += 1; return J::A(v); }
                    v.push(parse(s, i));
                }
            }
            b'"' => {
                let st = *i + 1;
                *i = st + s[st..].iter().position(|&c| c == b'"').unwrap();
                let out = std::str::from_utf8(&s[st..*i]).unwrap().to_string();
                assert!(!out.contains('\\'));
                *i += 1;
                J::S(out)
            }
            b'n' => { *i += 4; J::Null }
            b't' => { *i += 4; J::B(true) }
            b'f' => { *i += 5; J::B(false) }
            _ => {
                let st = *i;
                while s[*i] == b'-' || s[*i].is_ascii_digit() { *i += 1; }
                J::N(std::str::from_utf8(&s[st..*i]).unwrap().parse().unwrap())
            }
        }
    }

    fn vectors() -> J {
        let text = include_str!("../../../../verity/ml/commitments/frame_v3/vectors.json");
        parse(text.as_bytes(), &mut 0)
    }

    fn domain_of(d: &J) -> Domain {
        let r = d.get("range").arr();
        assert_eq!(r[0].n(), 0);
        let dom = Domain::range(&d.get("binding").d32(), d.get("owner").n(), r[1].n() as u64);
        assert_eq!(hex(&dom.id), d.get("domain_id").s(), "domain_id");
        dom
    }

    #[test]
    fn frame_v3_row_trees() {
        let v = vectors();
        assert_eq!(v.get("scheme").s(), SCHEME);
        for t in v.get("row_trees").arr() {
            let dom = domain_of(t.get("domain"));
            let hash = RowHash::parse(t.get("hash").s()).unwrap();
            assert_eq!(hash.schema(), t.get("schema").s());
            let (role, bits, n) = (t.get("role").n() as u8, t.get("word_bits").n() as u8, t.get("n_words").n() as u32);
            let digests: Vec<Digest> = t.get("digests").arr().iter().map(J::d32).collect();
            let leaves: Vec<Digest> = digests.iter().enumerate().map(|(r, d)| row_leaf(&dom, r as u64, hash, d)).collect();
            for (r, (got, want)) in leaves.iter().zip(t.get("leaves").arr()).enumerate() {
                assert_eq!(hex(got), want.s(), "leaf {r}");
                if hash == RowHash::Sha256 {
                    assert_eq!(sha256_row_digest(role, bits, n, &t.get("rows").arr()[r].bytes()), digests[r], "row digest {r}");
                }
            }
            let root = dom.root(&leaves).unwrap();
            assert_eq!(hex(&root), t.get("root").s(), "root");
            for (r, p) in t.get("paths").arr().iter().enumerate() {
                let want: Vec<Digest> = p.arr().iter().map(J::d32).collect();
                assert_eq!(dom.path(&leaves, r).unwrap(), want, "path {r}");
                assert!(dom.verify_path(&leaves[r], r as u64, &want, &root));
                let mut bad = leaves[r];
                bad[0] ^= 1;
                assert!(!dom.verify_path(&bad, r as u64, &want, &root));
            }
        }
    }

    #[test]
    fn frame_v3_row_leaves() {
        let v = vectors();
        let mut sha = 0;
        for c in v.get("row_leaves").arr() {
            let hash = RowHash::parse(c.get("hash").s()).unwrap();
            assert_eq!(hash.schema(), c.get("schema").s());
            let (role, bits, n) = (c.get("role").n() as u8, c.get("word_bits").n() as u8, c.get("n_words").n() as u32);
            match hash {
                RowHash::Sha256 => {
                    assert_eq!(hex(&sha256_row_prefix(role, bits, n)), c.get("prefix").s());
                    assert!(matches!(c.get("key"), J::Null));
                    assert_eq!(sha256_row_digest(role, bits, n, &c.get("row_bytes").bytes()), c.get("digest").d32());
                    sha += 1;
                }
                RowHash::Blake3 => assert_eq!(hex(&blake3_row_key(role)), c.get("key").s()),
                RowHash::VllmPos => unreachable!(),
            }
        }
        assert!(sha >= 14);
    }

    #[test]
    fn frame_v3_word_leaves() {
        let v = vectors();
        for c in v.get("word_leaves").arr() {
            let dom = domain_of(c.get("domain"));
            assert_eq!(c.get("schema").s(), "u16");
            let val = c.get("value").bytes();
            let word = u16::from_be_bytes([val[0], val[1]]) as u32;
            assert_eq!(hex(&word_leaf(&dom, c.get("position").n() as u64, 2, word)), c.get("leaf").s());
        }
    }

    #[test]
    fn empty_tree_and_uint() {
        assert_eq!(uint(0), vec![0]);
        assert_eq!(uint(255), vec![255]);
        assert_eq!(uint(256), vec![1, 0]);
        assert_eq!((depth(0), depth(1), depth(2), depth(5), depth(4096)), (0, 0, 1, 3, 12));
        let d = Domain::range(&[7u8; 32], -1, 0);
        assert_eq!(d.root(&[]).unwrap(), d.empty());
    }

    #[test]
    fn canonical_json_sorts_and_escapes() {
        let s = canonical_json(&[("tree", Json::Str("a")), ("K", Json::Int(1536)), ("dataset", Json::Str("q\"\\\n"))]);
        assert_eq!(s, "{\"K\":1536,\"dataset\":\"q\\\"\\\\\\n\",\"tree\":\"a\"}");
    }

    fn spec() -> Spec {
        Spec::parse(&format!("scheme frame-v3\nleaf sha256\nrelation bf16-ampere+sha256\ninstances synthetic dev {}\nlo 0\nvus 3\nK 4\nx 16\nw 16\ny u16 1\n", "ab".repeat(32))).unwrap()
    }

    fn limbs(d: &Digest) -> Vec<u32> { d.chunks(2).map(|c| u16::from_be_bytes([c[0], c[1]]) as u32).collect() }

    const VLLM_BASE: &str = "scheme vllm-v1\nleaf pos\nrelation bf16-ampere+vllm-v1\ninstances synthetic dev {m}\nlo 0\nvus 3\nK 4\nx 16\nw 16\ny u16 1\n";

    /// A vllm-v1 `commitment.txt` naming its own derived domains.
    pub fn vllm_spec_text() -> String {
        let base = VLLM_BASE.replace("{m}", &"ab".repeat(32));
        let mut s = Spec::parse(&format!("{base}domain_a {z}\ndomain_b {z}\ndomain_y {z}\n", z = "00".repeat(32))).unwrap();
        s.domains = None;
        let d: Vec<String> = crate::vllm_v1::PORTS.iter().map(|p| hex(&crate::vllm_v1::operand_domain(&s, p).digest())).collect();
        format!("{base}domain_a {}\ndomain_b {}\ndomain_y {}\n", d[0], d[1], d[2])
    }

    #[test]
    fn vllm_statement_check_rebuilds_the_roots() {
        use crate::vllm_v1 as V;
        let text = vllm_spec_text();
        let s = Spec::parse(&text).unwrap();
        let rows: Vec<Vec<u8>> = (0..3u8).map(|v| vec![v; 8]).collect();
        let mut public = Vec::new();
        for (v, r) in rows.iter().enumerate() {
            public.push(100 + v as u32);
            public.extend(limbs(&V::pos_leaf(r)));
            public.extend(limbs(&V::pos_leaf(&[r.as_slice(), &[9]].concat())));
        }
        let c = check(s.clone(), &public).unwrap();
        assert!(c.pin.is_none() && c.problem(true).is_none());
        let da = V::operand_domain(&s, "a");
        assert_eq!(c.roots[0], da.bind(&V::fold(&rows.iter().map(|r| V::pos_leaf(r)).collect::<Vec<_>>())));
        let dy = V::operand_domain(&s, "y");
        assert_eq!(c.roots[2], dy.bind(&V::fold(&(0..3).map(|v| V::pos_leaf(&(100u16 + v).to_le_bytes())).collect::<Vec<_>>())));
        assert!(c.json().contains("\"scheme\": \"vllm-v1\""));
        // a domain line that is not the derived one, a missing one, a frame-v3 statement naming domains
        let a_line = text.lines().find(|l| l.starts_with("domain_a")).unwrap().to_string();
        let bad = text.replace(&a_line, &format!("domain_a {}", "11".repeat(32)));
        assert!(check(Spec::parse(&bad).unwrap(), &public).is_err());
        assert!(Spec::parse(&text.replace(&format!("{a_line}\n"), "")).is_err());
        let fv3 = text.replace("scheme vllm-v1\nleaf pos", "scheme frame-v3\nleaf sha256").replace("+vllm-v1", "+sha256");
        assert!(Spec::parse(&fv3).is_err());
        assert!(Spec::parse(&text.replace("scheme vllm-v1", "scheme frame-v3")).is_err());
        let mut p2 = public.clone();
        p2[1 + 16] ^= 1;
        let c2 = check(s, &p2).unwrap();
        assert!(c2.roots[1] != c.roots[1] && c2.roots[0] == c.roots[0]);
    }

    #[test]
    fn statement_check_rebuilds_the_roots() {
        let s = spec();
        let rows: Vec<Vec<u8>> = (0..3u8).map(|v| vec![v; 8]).collect();
        let mut public = Vec::new();
        for (v, r) in rows.iter().enumerate() {
            public.push(100 + v as u32);
            public.extend(limbs(&sha256_row_digest(ROLE_X, 16, 4, r)));
            public.extend(limbs(&sha256_row_digest(ROLE_W, 16, 4, r)));
        }
        let c = check(s.clone(), &public).unwrap();
        assert!(c.pin.is_none() && c.problem(true).is_none() && c.problem(false).is_some());
        let [ba, bb, by] = s.bindings();
        let da = Domain::range(&ba, -1, 3);
        let la: Vec<Digest> = rows.iter().enumerate().map(|(v, r)| row_leaf(&da, v as u64, RowHash::Sha256, &sha256_row_digest(ROLE_X, 16, 4, r))).collect();
        assert_eq!(c.roots[0], da.root(&la).unwrap());
        let db = Domain::range(&bb, -2, 3);
        assert_ne!(c.roots[1], da.root(&la).unwrap());
        assert_eq!(db.count, 3);
        let dy = Domain::range(&by, -1, 3);
        assert_eq!(c.roots[2], dy.root(&(0..3).map(|v| word_leaf(&dy, v, 2, 100 + v as u32)).collect::<Vec<_>>()).unwrap());
        // one changed digest limb, one changed y word: other roots
        let mut p2 = public.clone();
        p2[1 + 33] ^= 1;
        assert_ne!(check(s.clone(), &p2).unwrap().roots[0], c.roots[0]);
        let mut p3 = public.clone();
        p3[0] += 1;
        assert_ne!(check(s.clone(), &p3).unwrap().roots[2], c.roots[2]);
        // malformed: a limb of 16 bits, a y word past u16, a missing word
        let mut p4 = public.clone();
        p4[1] = 1 << 16;
        assert!(check(s.clone(), &p4).is_err());
        let mut p5 = public.clone();
        p5[0] = 1 << 16;
        assert!(check(s.clone(), &p5).is_err());
        assert!(check(s, &public[1..]).is_err());
    }

    #[test]
    fn spec_refuses_malformed() {
        let ok = format!("scheme frame-v3\nleaf sha256\nrelation bf16-ampere+sha256\ninstances synthetic dev {}\nlo 0\nvus 3\nK 4\nx 16\nw 16\ny u16 1\n", "ab".repeat(32));
        assert!(Spec::parse(&ok).is_ok());
        for (from, to) in [("leaf sha256", "leaf blake3"), ("scheme frame-v3", "scheme vllm-v1"), ("x 16", "x 12"), ("y u16 1", "y u8 1"),
                           ("vus 3", "vus 0"), ("+sha256", "+bound"), ("K 4\n", "K 4\nK 4\n"), ("lo 0\n", "lo 0\nroot 00\n")] {
            assert!(Spec::parse(&ok.replacen(from, to, 1)).is_err(), "{from} -> {to}");
        }
    }

    #[test]
    fn pins_are_well_formed() {
        for (i, p) in PINS.iter().enumerate() {
            for d in [p.a, p.b, p.y, p.manifest_sha256] { assert!(unhex32(d).is_some()); }
            let h = relation_hash(p.relation).expect("a +sha256 / +blake3 relation");
            let base = p.relation.strip_suffix(h.suffix()).unwrap();
            assert!(crate::pins::relations().contains(&base) || crate::pins::relations().contains(&p.relation), "{} commits {base}, unknown to pins.txt", p.relation);
            for q in &PINS[i + 1..] { assert!(!(p.relation == q.relation && p.lo == q.lo && p.vus == q.vus)); }
        }
    }
}
