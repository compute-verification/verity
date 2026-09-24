//! `relation-bare/v2` (feature `relation-bare`): the Table 2 bare statement -- authentication excluded -- for a
//! batch of K = 1536 VUs of one numerical format.  Shared byte for byte by the stock-SP1 guest (`guest/`, built with
//! `--features relation-bare`) and the TC_DOT fork (`tcdot/`), so both prove the identical statement.
//!
//! **Statement.** For every `i < B`: `y[i]` is the output word of VU `i`'s chain under `format`.  For
//! [`Format::Bf16Ampere`] (the A100 row) that is `f32_to_bf16(chain(x_i, W_i))`, the body of `GemmCoordinate<1536>`
//! in `veritor.tc-ampere-bf16@1` ([`crate::tc`]: 96 `mma` steps from `+0`, groups `[8, 8]`, a 25-bit adder, exponent
//! floor `-132`; a non-finite operand or a saturation before the last step is outside the domain and rejects; a
//! saturation inside the last step is the valid output `+-inf`, and the rest of that step is not examined).  The
//! format, `y` and the instance-set identity are public; every `x_i`, `W_i` is private and NOT authenticated --
//! binding the rows to the frozen set is the authentication Table 2's `authentication = excluded` column leaves out,
//! so the identity is a label the guest carries through, not something it checks.
//!
//! **Public values** (33 bytes, the 33-byte layout of [`crate::split_public_values`]): `digest || verdict`,
//!
//! ```text
//! digest  = sha256(statement bytes)
//! statement bytes = "verity/sp1/relation-bare/v2" || u32le(format) || id || u32le(K) || u32le(B)
//!                   || y[0..B], each word y_bytes(format) little-endian
//! id      = sha256("<dataset>|<tier>|<lo>|<hi>|<manifest_sha256>")
//! verdict = 1 when every VU is in the domain and gives its y[i], else 0
//! ```
//!
//! A verifier writes the statement bytes from its own identity string and the committed `y` words
//! ([`statement_bytes`]) and accepts `sha256(statement bytes) || 1` under the pinned verifying key only.
//!
//! **Input** (`SP1Stdin` buffers): a header `"VYRB" || u32le(2) || u32le(format) || u32le(K) || u32le(B) ||
//! u32le(vus_per_read) || id || y[0..B]` ([`encode_header`]), then `ceil(B / vus_per_read)` chunks
//! ([`encode_chunk`]) of `vus_per_read` VUs (the last may be shorter), each VU its x row then its W row,
//! [`Format::row_bytes`] each.  Every format's row is a whole number of u64 words and SP1 places every `read_vec`
//! buffer 8-byte aligned, so the guest reads a chunk as little-endian u64 words and [`check_words`] hands each VU
//! to [`check_one`], the one place a format is dispatched.
//!
//! **Kernel** ([`Format::Bf16Ampere`]).  [`check_vu`] is the `tc` chain specialised to this shape: SWAR decoding of
//! four words per u64, two 8-bit significand products per multiply, biased integer exponents, and no packing between
//! steps (a group's `Term` round-trips `pack_fp32` / `fp32_term` exactly).  [`check_vu_reference`] is the kernel's
//! own gate-set evaluation (`relation_only::evaluate`); the tests hold the two equal on the frozen set, on every
//! negative and on adversarial random rows.

use sha2::{Digest, Sha256};

/// The domain-separation tag hashed in front of every statement.
pub const TAG: &[u8] = b"verity/sp1/relation-bare/v2";
/// The header magic.
pub const MAGIC: [u8; 4] = *b"VYRB";
/// The input format version.
pub const VERSION: u32 = 2;
/// Elements per row: the frozen tiers' K.
pub const K: usize = 1536;
/// u64 words per BF16 row (four words each).
pub const ROW_WORDS: usize = K / 4;
/// u64 words per BF16 VU: the x row then the W row.
pub const VU_WORDS: usize = 2 * ROW_WORDS;
/// Header bytes before the `y` words.
pub const HEADER_FIXED_BYTES: usize = 56;
/// `digest || verdict`.
pub const PUBLIC_VALUES_LEN: usize = 33;
pub const VERDICT_ACCEPT: u8 = 1;
pub const VERDICT_REJECT: u8 = 0;

/// The numerical format of a batch: which chain a VU's rows run through.  The id is part of the statement.  A
/// format whose arm in [`check_one`] is not written yet rejects every VU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Format {
    /// `veritor.tc-ampere-bf16@1`, the A100 BF16 row: BF16 words as u16 LE, `y` = `f32_to_bf16` of the result (u16).
    Bf16Ampere = 1,
    /// `bf16-hopper` (H100 BF16): BF16 words as u16 LE, `y` = `f32_to_bf16` of the final accumulator (u16).
    Bf16Hopper = 2,
    /// `fp8-hopper` (H100 FP8 wgmma): E4M3 bytes, `y` = the final FP32 accumulator word (u32).
    Fp8Hopper = 3,
    /// `fp8-ada` (4090 FP8 mma): E4M3 bytes, `y` = the final FP32 accumulator word (u32).
    Fp8Ada = 4,
    /// `fp4-nvf4` (5090 NVFP4 sm_120): 24 firings of 64 E2M1 code bytes then 4 UE4M3 scale bytes, `y` = the final
    /// FP32 accumulator word (u32).
    Fp4Nvf4 = 5,
}

impl Format {
    pub const ALL: [Format; 5] = [Format::Bf16Ampere, Format::Bf16Hopper, Format::Fp8Hopper, Format::Fp8Ada, Format::Fp4Nvf4];

    pub fn id(self) -> u32 {
        self as u32
    }

    pub fn from_id(id: u32) -> Option<Format> {
        Format::ALL.into_iter().find(|f| f.id() == id)
    }

    /// The name `verity_sp1.format_instances` files carry (`bf16-ampere` for the A100 fixture).
    pub fn name(self) -> &'static str {
        match self {
            Format::Bf16Ampere => "bf16-ampere",
            Format::Bf16Hopper => "bf16-hopper",
            Format::Fp8Hopper => "fp8-hopper",
            Format::Fp8Ada => "fp8-ada",
            Format::Fp4Nvf4 => "fp4-nvf4",
        }
    }

    pub fn from_name(name: &str) -> Option<Format> {
        Format::ALL.into_iter().find(|f| f.name() == name)
    }

    /// Bytes of one operand row (x or W), a multiple of 8.
    pub const fn row_bytes(self) -> usize {
        match self {
            Format::Bf16Ampere | Format::Bf16Hopper => 2 * K,
            Format::Fp8Hopper | Format::Fp8Ada => K,
            Format::Fp4Nvf4 => 24 * 68,
        }
    }

    /// Bytes of one output word `y`.
    pub const fn y_bytes(self) -> usize {
        match self {
            Format::Bf16Ampere | Format::Bf16Hopper => 2,
            Format::Fp8Hopper | Format::Fp8Ada | Format::Fp4Nvf4 => 4,
        }
    }

    /// u64 words of one VU in a chunk: its x row then its W row.
    pub const fn vu_words(self) -> usize {
        2 * self.row_bytes() / 8
    }
}

/// THE format dispatch: whether one VU gives the claimed word `y` (its `y_bytes` little-endian bytes,
/// zero-extended).  The VU is its x row then its W row, [`Format::row_bytes`] each, given twice: as bytes and as
/// the same bytes read as little-endian u64 words (the guest holds both views of its input buffer).  A format is a
/// [`Format`] variant plus one arm here.
pub fn check_one(format: Format, vu_bytes: &[u8], vu_words: &[u64], y: u32) -> bool {
    match format {
        Format::Bf16Ampere => {
            let (x, w) = vu_words.split_at(ROW_WORDS);
            y <= 0xFFFF && check_vu(x, w) == Some(y as u16)
        }
        Format::Bf16Hopper => {
            let (x, w) = vu_words.split_at(crate::tc_hopper_bf16::ROW_WORDS);
            y <= 0xFFFF && crate::tc_hopper_bf16::vu_words(x, w) == Some(y as u16)
        }
        Format::Fp8Hopper => {
            let ((x, w), (xw, ww)) = (vu_bytes.split_at(format.row_bytes()), vu_words.split_at(crate::tc_fp8::ROW_WORDS));
            crate::tc_fp8::vu_hopper_views(x, w, xw, ww) == Some(y)
        }
        Format::Fp8Ada => {
            let ((x, w), (xw, ww)) = (vu_bytes.split_at(format.row_bytes()), vu_words.split_at(crate::tc_fp8::ROW_WORDS));
            crate::tc_fp8::vu_ada_views(x, w, xw, ww) == Some(y)
        }
        Format::Fp4Nvf4 => {
            let (x, w) = vu_bytes.split_at(format.row_bytes());
            crate::nvfp4::vu(x, w) == Some(y)
        }
    }
}

/// The identity string of a contiguous id range `[lo, hi)` of a tier.
pub fn identity_string(dataset: &str, tier: &str, lo: u64, hi: u64, manifest_sha256: &str) -> String {
    format!("{dataset}|{tier}|{lo}|{hi}|{manifest_sha256}")
}

/// `id = sha256(identity string)`.
pub fn identity_digest(identity: &str) -> [u8; 32] {
    Sha256::digest(identity.as_bytes()).into()
}

/// The claimed words as the statement and the header carry them: `y_bytes(format)` little-endian bytes each, or
/// `None` when a word does not fit.
pub fn encode_y(format: Format, y: &[u32]) -> Option<Vec<u8>> {
    let n = format.y_bytes();
    let mut out = Vec::with_capacity(n * y.len());
    for &word in y {
        if n < 4 && word >> (8 * n) != 0 {
            return None;
        }
        out.extend_from_slice(&word.to_le_bytes()[..n]);
    }
    Some(out)
}

/// The statement bytes whose sha256 is the public digest (`y_le` from [`encode_y`]).
pub fn statement_bytes(format: Format, id: &[u8; 32], k: u32, y_le: &[u8]) -> Vec<u8> {
    let b = (y_le.len() / format.y_bytes()) as u32;
    let mut out = Vec::with_capacity(TAG.len() + 44 + y_le.len());
    out.extend_from_slice(TAG);
    out.extend_from_slice(&format.id().to_le_bytes());
    out.extend_from_slice(id);
    out.extend_from_slice(&k.to_le_bytes());
    out.extend_from_slice(&b.to_le_bytes());
    out.extend_from_slice(y_le);
    out
}

/// `sha256(statement bytes)` from the `y` bytes as they sit in the header (the guest hashes them in place).
pub fn statement_digest_le(format: Format, id: &[u8; 32], k: u32, b: u32, y_le: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(TAG);
    hasher.update(format.id().to_le_bytes());
    hasher.update(id);
    hasher.update(k.to_le_bytes());
    hasher.update(b.to_le_bytes());
    hasher.update(y_le);
    hasher.finalize().into()
}

/// `digest || verdict`.
pub fn public_values(digest: &[u8; 32], accept: bool) -> [u8; PUBLIC_VALUES_LEN] {
    let mut out = [0u8; PUBLIC_VALUES_LEN];
    out[..32].copy_from_slice(digest);
    out[32] = if accept { VERDICT_ACCEPT } else { VERDICT_REJECT };
    out
}

/// The header buffer (`y_le` from [`encode_y`]).
pub fn encode_header(format: Format, k: u32, vus_per_read: u32, id: &[u8; 32], y_le: &[u8]) -> Vec<u8> {
    let b = (y_le.len() / format.y_bytes()) as u32;
    let mut out = Vec::with_capacity(HEADER_FIXED_BYTES + y_le.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&format.id().to_le_bytes());
    out.extend_from_slice(&k.to_le_bytes());
    out.extend_from_slice(&b.to_le_bytes());
    out.extend_from_slice(&vus_per_read.to_le_bytes());
    out.extend_from_slice(id);
    out.extend_from_slice(y_le);
    out
}

/// A parsed header; `y_le` borrows the claimed words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header<'a> {
    pub format: Format,
    pub k: u32,
    pub b: u32,
    pub vus_per_read: u32,
    pub id: [u8; 32],
    pub y_le: &'a [u8],
}

/// Parse a header; `None` on a wrong magic, version, format or length.
pub fn parse_header(bytes: &[u8]) -> Option<Header<'_>> {
    if bytes.len() < HEADER_FIXED_BYTES || bytes[..4] != MAGIC {
        return None;
    }
    let u32_at = |at: usize| u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    if u32_at(4) != VERSION {
        return None;
    }
    let format = Format::from_id(u32_at(8))?;
    let (k, b, vus_per_read) = (u32_at(12), u32_at(16), u32_at(20));
    if (bytes.len() - HEADER_FIXED_BYTES) as u64 != format.y_bytes() as u64 * b as u64 {
        return None;
    }
    let mut id = [0u8; 32];
    id.copy_from_slice(&bytes[24..HEADER_FIXED_BYTES]);
    Some(Header { format, k, b, vus_per_read, id, y_le: &bytes[HEADER_FIXED_BYTES..] })
}

/// The chunk buffer of VUs `lo..hi` of VU-major operand blocks (`row_bytes` per row).
pub fn encode_chunk(format: Format, x: &[u8], w: &[u8], lo: usize, hi: usize) -> Vec<u8> {
    let r = format.row_bytes();
    let mut out = Vec::with_capacity((hi - lo) * 2 * r);
    for i in lo..hi {
        out.extend_from_slice(&x[r * i..r * (i + 1)]);
        out.extend_from_slice(&w[r * i..r * (i + 1)]);
    }
    out
}

/// Little-endian bytes as u64 words (the host-side and test-side view of a chunk; the guest reinterprets in place).
pub fn words_of(bytes: &[u8]) -> Vec<u64> {
    bytes.chunks_exact(8).map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])).collect()
}

/// Whether every VU of a chunk gives its claimed word (`y_le`, [`Format::y_bytes`] per VU).  The chunk is given as
/// its bytes and as the same bytes read as little-endian u64 words ([`Format::vu_words`] per VU).
pub fn check_words(format: Format, bytes: &[u8], words: &[u64], y_le: &[u8]) -> bool {
    let (per, yb) = (format.vu_words(), format.y_bytes());
    if y_le.len() % yb != 0 || words.len() != y_le.len() / yb * per || bytes.len() != 8 * words.len() {
        return false;
    }
    for ((vu_bytes, vu_words), y) in bytes.chunks_exact(8 * per).zip(words.chunks_exact(per)).zip(y_le.chunks_exact(yb)) {
        let mut word = [0u8; 4];
        word[..yb].copy_from_slice(y);
        if !check_one(format, vu_bytes, vu_words, u32::from_le_bytes(word)) {
            return false;
        }
    }
    true
}

/// What the guest commits for a header and its chunks (the host's expectation and the tests' reference).
pub fn decide(header: &[u8], chunks: &[Vec<u8>]) -> [u8; PUBLIC_VALUES_LEN] {
    let Some(h) = parse_header(header) else {
        return public_values(&[0u8; 32], false);
    };
    let digest = statement_digest_le(h.format, &h.id, h.k, h.b, h.y_le);
    let (b, per, yb) = (h.b as usize, h.vus_per_read as usize, h.format.y_bytes());
    let mut ok = h.k as usize == K && per > 0 && chunks.len() == b.div_ceil(per.max(1));
    let mut i = 0;
    for chunk in chunks {
        if !ok {
            break;
        }
        let n = per.min(b - i);
        ok = chunk.len() == 8 * n * h.format.vu_words() && check_words(h.format, chunk, &words_of(chunk), &h.y_le[yb * i..yb * (i + n)]);
        i += n;
    }
    public_values(&digest, ok)
}

// --- the kernel ------------------------------------------------------------------------------------------------------

const L7F: u64 = 0x007F_007F_007F_007F;
const L80: u64 = 0x0080_0080_0080_0080;
/// Every lane's exponent field, in place.
const L7F80: u64 = 0x7F80_7F80_7F80_7F80;
/// Bit 15 of every lane: of `field << 7 + 0x80`, set iff the field is `0xFF` (a non-finite word); of
/// `field << 7 + 0x7F80`, set iff the field is non-zero (the implicit bit).
const L8000: u64 = 0x8000_8000_8000_8000;
/// Lanes 0 and 2 of a SWAR word, one byte each: two products per 64x64 -> 128 multiply.
const M02: u64 = 0x0000_00FF_0000_00FF;
/// [`M02`] at bit 10, for the multiplier: the products come out as `p << 10`, aligned for the adder.
const M02S: u64 = M02 << 10;
/// `p << 10` of a product of two 8-bit significands.
const M26: u64 = (1 << 26) - 1;
/// Exponents are biased by 252: a product's `max(ea, 1) + max(eb, 1) - 2` is its true exponent + 252.
const BIAS: i64 = 252;
/// The adder's exponent floor `-132`, biased.
const FLOOR: u64 = (-132 + BIAS) as u64;
/// The smallest normal exponent `-126`, biased.
const SUBNORMAL: u64 = (-126 + BIAS) as u64;
/// `tc_dot`'s saturation test `exp > 127`, biased.
const SATURATED: u64 = (127 + BIAS) as u64;

/// A group's result: `(-1)^neg * mag * 2^(e - BIAS - 23)`, `mag` on the 24-bit grid (`tc::Term`), `e` biased, `neg`
/// 0 or 1; `mag == 0` is `+0` with `e == 0`, so `e > SATURATED` is exactly `tc_dot`'s saturation test.
#[derive(Clone, Copy)]
struct Acc {
    neg: u64,
    e: u64,
    mag: u64,
}

const ZERO: Acc = Acc { neg: 0, e: 0, mag: 0 };

/// `BIT_LEN[i]` is the bit length of `i`.
static BIT_LEN: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 1;
    while i < 256 {
        t[i] = t[i >> 1] + 1;
        i += 1;
    }
    t
};

/// `SHIFT[i]` for `1 <= i = m >> 23 < 128`: `bit_length(m) - 24`, the truncating shift of `m` to 24 bits.
static SHIFT: [u8; 128] = {
    let mut t = [0u8; 128];
    let mut i = 1;
    while i < 128 {
        t[i] = BIT_LEN[i] - 1;
        i += 1;
    }
    t
};

/// Whether all sixteen words of a group are finite: bit 15 of `field << 7 + 0x80` is set iff the field is `0xFF`.
/// Checked before the group's arithmetic, so the test is a branch there rather than an accumulator the compiler
/// carries (and spills) to the end of the loop.
#[inline(always)]
fn finite(x0: u64, x1: u64, w0: u64, w1: u64) -> bool {
    (((x0 & L7F80) + L80) | ((x1 & L7F80) + L80) | ((w0 & L7F80) + L80) | ((w1 & L7F80) + L80)) & L8000 == 0
}

/// Four products of a lane quadruple: significand products `<< 10`, the SWAR word of their biased exponents `<< 7`
/// (lane `j`, bits `7 ..= 15`, is product `j`'s), and the word whose bit `15 + 16 j` is product `j`'s sign.
#[inline(always)]
fn quad(x: u64, w: u64) -> ([u64; 4], u64, u64) {
    let ex = x & L7F80;
    let ew = w & L7F80;
    // 0x80 in the lanes whose exponent field is non-zero: the implicit bit
    let fx = ((ex + L7F80) & L8000) >> 8;
    let fw = ((ew + L7F80) & L8000) >> 8;
    let sx = (x & L7F) | fx;
    let sw = (w & L7F) | fw;
    // (max(e, 1) - 1) << 7 per operand, summed: at most 508 << 7 per lane, so no lane carries or borrows
    let g = ex + ew - (fx + fw);
    // low 64 bits: (p0 << 10) + (cross terms << 42), below 2^59, so the high 64 bits are exactly p2 << 10
    let t02 = (sx & M02) as u128 * ((sw << 10) & M02S) as u128;
    let t13 = ((sx >> 16) & M02) as u128 * ((sw >> 6) & M02S) as u128;
    ([t02 as u64 & M26, t13 as u64 & M26, (t02 >> 64) as u64, (t13 >> 64) as u64], g, x ^ w)
}

/// The four lanes of a [`quad`] exponent word (each below 2^9).
#[inline(always)]
fn lanes(g: u64) -> [u64; 4] {
    [(g >> 7) & 0x1FF, (g >> 23) & 0x1FF, (g >> 39) & 0x1FF, g >> 55]
}

/// Product `j`'s sign in a [`quad`] sign word.
#[inline(always)]
fn negative(s: u64, j: usize) -> bool {
    ((s << (48 - 16 * j)) as i64) < 0
}

/// The running maximum over a product: raised only by a non-zero product.  The update is rare once the accumulator
/// dominates, and `black_box` keeps it a branch (a select costs two branches and moves on every product).
#[inline(always)]
fn raise(mx: u64, e: u64, p: u64) -> u64 {
    if e > mx && p != 0 {
        core::hint::black_box(e)
    } else {
        mx
    }
}

/// Adds a product truncated to the group's maximum exponent `mx`: `p << 10 >> d` is zero from d = 26 (and a zero
/// product adds nothing at any d, so the wrapped `d` of a zero product above the maximum is harmless).
#[inline(always)]
fn add_term(total: i64, p: u64, e: u64, neg: bool, mx: u64) -> i64 {
    let d = mx.wrapping_sub(e);
    if d < 26 {
        let v = (p >> d) as i64;
        if neg {
            total - v
        } else {
            total + v
        }
    } else {
        total
    }
}

/// `tc::group_sum` of the accumulator and eight products (width 25, floor -132), on the terms as they come out of
/// [`quad`]: the largest exponent of a non-zero term, each term's `magnitude << 1` truncated to it, the exact sum,
/// truncating normalisation to 25 bits, denormalisation below -126, and the truncation back to 24 bits.
#[inline(always)]
fn group(acc: Acc, x0: u64, x1: u64, w0: u64, w1: u64) -> Acc {
    let (pa, ga, sa) = quad(x0, w0);
    let (pb, gb, sb) = quad(x1, w1);
    let (ea, eb) = (lanes(ga), lanes(gb));
    // a non-zero accumulator has exponent >= -126, above the floor
    let mut mx = if acc.mag != 0 { acc.e } else { FLOOR };
    for j in 0..4 {
        mx = raise(mx, ea[j], pa[j]);
    }
    for j in 0..4 {
        mx = raise(mx, eb[j], pb[j]);
    }
    let mut total: i64 = 0;
    if acc.mag != 0 {
        let d = mx - acc.e;
        if d < 26 {
            let v = ((acc.mag + acc.mag) >> d) as i64;
            total = if acc.neg != 0 { -v } else { v };
        }
    }
    for j in 0..4 {
        total = add_term(total, pa[j], ea[j], negative(sa, j), mx);
    }
    for j in 0..4 {
        total = add_term(total, pb[j], eb[j], negative(sb, j), mx);
    }
    normalize(total, mx)
}

/// `group_sum`'s tail on the biased maximum exponent `mx`: `|total| < 9 * 2^26`, so the bit length is at most 30.  The
/// usual sum is at least 2^23 and its exponent normal: then the truncation to 24 bits is one right shift by
/// `bit_length - 24` and the exponent is `mx + bit_length - 25`.  Everything else is [`normalize_slow`]'s.
#[inline(always)]
fn normalize(total: i64, mx: u64) -> Acc {
    let m = total.unsigned_abs();
    let top = m >> 23;
    if top != 0 {
        let shift = SHIFT[top as usize & 127] as u64;
        let e = mx + shift - 1;
        if e >= SUBNORMAL {
            return Acc { neg: (total < 0) as u64, e, mag: m >> shift };
        }
    }
    let (e, mag) = normalize_slow(total, mx);
    Acc { neg: (total < 0 && mag != 0) as u64, e, mag }
}

/// [`normalize`] in general, as `(e, mag)` (two registers, so the accumulator never goes through memory): normalising
/// to 25 bits, denormalising below -126 and truncating to 24 bits are three truncating shifts of one non-negative
/// integer, so they are one shift by their sum (a left shift where the sum is negative, exact).
#[cold]
#[inline(never)]
fn normalize_slow(total: i64, mx: u64) -> (u64, u64) {
    if total == 0 {
        return (0, 0);
    }
    let m = total.unsigned_abs();
    let (mut v, mut bits) = (m, 0u64);
    if v >> 16 != 0 {
        v >>= 16;
        bits += 16;
    }
    if v >> 8 != 0 {
        v >>= 8;
        bits += 8;
    }
    let bits = bits + BIT_LEN[v as usize & 255] as u64;
    // mx >= FLOOR = 120, so this does not wrap
    let e = mx + bits - 25;
    let (e, shift) = if e >= SUBNORMAL { (e, bits as i64 - 24) } else { (SUBNORMAL, bits as i64 - 24 + (SUBNORMAL - e) as i64) };
    let mag = if shift >= 0 { m >> shift } else { m << -shift };
    if mag == 0 {
        return (0, 0);
    }
    (e, mag)
}

/// `f32_to_bf16(pack_fp32(acc))` for a finite result.
fn to_bf16(acc: Acc) -> u16 {
    if acc.mag == 0 {
        return 0;
    }
    let mut field = acc.e as i64 - BIAS + 127;
    if acc.mag & (1 << 23) == 0 {
        field -= 1;
    }
    let bits = (acc.neg << 31) | ((field as u64) << 23) | (acc.mag & 0x7F_FFFF);
    (((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF) as u16
}

/// The BF16 infinity of a sign (the cast of the FP32 infinity a saturating last step returns).
fn infinity(neg: u64) -> u16 {
    0x7F80 | ((neg as u16) << 15)
}

/// One VU: the output word of the chain on an x row and a W row given as [`ROW_WORDS`] u64 words each (word `k` of
/// the row is bits `16 (k % 4) ..` of u64 `k / 4`), or `None` where the chain rejects.  Words `0..1528` must all be
/// finite (every one is examined before any valid output); words `1528..1536` only when the last step's first group
/// does not saturate.  A saturation before the last step rejects.
pub fn check_vu(x: &[u64], w: &[u64]) -> Option<u16> {
    let x: &[u64; ROW_WORDS] = x.try_into().ok()?;
    let w: &[u64; ROW_WORDS] = w.try_into().ok()?;
    let mut acc = ZERO;
    // every outcome before the last step is a rejection, so a non-finite word and a saturation may reject in either
    // order; in the last step the first group's words are examined before its saturation, the second group's only
    // after the first did not saturate
    for step in 0..K / 16 - 1 {
        let i = 4 * step;
        if !finite(x[i], x[i + 1], w[i], w[i + 1]) {
            return None;
        }
        acc = group(acc, x[i], x[i + 1], w[i], w[i + 1]);
        if acc.e > SATURATED {
            return None;
        }
        if !finite(x[i + 2], x[i + 3], w[i + 2], w[i + 3]) {
            return None;
        }
        acc = group(acc, x[i + 2], x[i + 3], w[i + 2], w[i + 3]);
        if acc.e > SATURATED {
            return None;
        }
    }
    let i = ROW_WORDS - 4;
    if !finite(x[i], x[i + 1], w[i], w[i + 1]) {
        return None;
    }
    acc = group(acc, x[i], x[i + 1], w[i], w[i + 1]);
    if acc.e > SATURATED {
        return Some(infinity(acc.neg));
    }
    if !finite(x[i + 2], x[i + 3], w[i + 2], w[i + 3]) {
        return None;
    }
    acc = group(acc, x[i + 2], x[i + 3], w[i + 2], w[i + 3]);
    if acc.e > SATURATED {
        return Some(infinity(acc.neg));
    }
    Some(to_bf16(acc))
}

/// A K-word row as [`ROW_WORDS`] u64 words.
pub fn row_words(row: &[u16]) -> Vec<u64> {
    row.chunks_exact(4)
        .map(|q| q[0] as u64 | (q[1] as u64) << 16 | (q[2] as u64) << 32 | (q[3] as u64) << 48)
        .collect()
}

/// The reference: the checker kernel's own gate-set evaluation of `GemmCoordinate<K>` on the rows
/// (`relation_only::evaluate`), as a word.
#[cfg(test)]
pub fn check_vu_reference(x: &[u16], w: &[u16]) -> Option<u16> {
    let x: Vec<u64> = x.iter().map(|&v| v as u64).collect();
    let w: Vec<u64> = w.iter().map(|&v| v as u64).collect();
    crate::relation_only::evaluate(&x, &w).map(|word| word as u16)
}

#[cfg(test)]
mod tests {
    //! The kernel equals the gate-set evaluation on the frozen `vu-k1536` set (when its built arrays are present:
    //! `VERITY_BENCH_INSTANCES=<dir>` or `fixtures/bench-instances/v1`), on all 52 committed negatives, and on
    //! adversarial random rows that exercise zeros, subnormals, the floor, cancellation, intermediate and final
    //! saturation, and non-finite words on either side of the last step's group boundary.

    use super::*;
    use std::path::PathBuf;

    const MANIFEST: &str = "059103cf9bd55ee83cbd4bb14ae6db2f60db2cb4ddf85cdc22b1cecee6e4eeea";

    fn fixture_dir() -> PathBuf {
        std::env::var_os("VERITY_BENCH_INSTANCES")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../fixtures/bench-instances/v1")))
    }

    fn u16s(path: &std::path::Path) -> Option<Vec<u16>> {
        let bytes = std::fs::read(path).ok()?;
        Some(bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
    }

    fn fast(x: &[u16], w: &[u16]) -> Option<u16> {
        check_vu(&row_words(x), &row_words(w))
    }

    /// A small deterministic generator (splitmix64).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// A BF16 word with a sign, an exponent field from `lo..=hi` and a random mantissa; zero / subnormal /
    /// non-finite with the given per-mille rates.
    fn word(r: &mut Rng, lo: u64, hi: u64, zero_pm: u64, sub_pm: u64, nonfinite_pm: u64) -> u16 {
        let sign = (r.next() & 1) << 15;
        let roll = r.below(1000);
        let mant = r.next() & 0x7F;
        let v = if roll < zero_pm {
            0
        } else if roll < zero_pm + sub_pm {
            mant.max(1)
        } else if roll < zero_pm + sub_pm + nonfinite_pm {
            (0xFF << 7) | if r.next() & 1 == 0 { 0 } else { mant.max(1) }
        } else {
            ((lo + r.below(hi - lo + 1)) << 7) | mant
        };
        (sign | v) as u16
    }

    fn y_le(y: &[u16]) -> Vec<u8> {
        encode_y(Format::Bf16Ampere, &y.iter().map(|&v| v as u32).collect::<Vec<_>>()).unwrap()
    }

    fn bytes_of(words: &[u16]) -> Vec<u8> {
        words.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn header_statement_and_public_values_round_trip() {
        let identity = identity_string("bench-instances/v1", "vu-k1536", 0, 4096, MANIFEST);
        assert_eq!(identity, format!("bench-instances/v1|vu-k1536|0|4096|{MANIFEST}"));
        let id = identity_digest(&identity);
        for format in Format::ALL {
            assert_eq!(Format::from_id(format.id()), Some(format));
            assert_eq!(Format::from_name(format.name()), Some(format));
            assert_eq!(format.row_bytes() % 8, 0);
            let y: Vec<u32> = (0..70u32).map(|i| i * 977 % (1 << (8 * format.y_bytes() as u32 - 1))).collect();
            let yl = encode_y(format, &y).unwrap();
            let header = encode_header(format, K as u32, 16, &id, &yl);
            assert_eq!(header.len(), HEADER_FIXED_BYTES + format.y_bytes() * y.len());
            let h = parse_header(&header).unwrap();
            assert_eq!((h.format, h.k, h.b, h.vus_per_read, h.id, h.y_le), (format, K as u32, 70, 16, id, &yl[..]));
            let stmt = statement_bytes(format, &id, K as u32, &yl);
            assert_eq!(&stmt[..TAG.len()], TAG);
            assert_eq!(statement_digest_le(format, &id, K as u32, 70, h.y_le), crate::frame::sha256(&stmt));
            // malformed headers
            assert!(parse_header(&header[..header.len() - 1]).is_none());
            for (at, v) in [(0, b'X'), (4, 1), (8, 0), (8, 6)] {
                let mut bad = header.clone();
                bad[at] = v;
                assert!(parse_header(&bad).is_none(), "byte {at} = {v}");
            }
        }
        // the format is bound: the same words under another format are another statement
        let yl = y_le(&[1, 2, 3]);
        assert_ne!(statement_bytes(Format::Bf16Ampere, &id, K as u32, &yl), statement_bytes(Format::Bf16Hopper, &id, K as u32, &yl));
        assert!(encode_y(Format::Bf16Ampere, &[0x1_0000]).is_none());
        assert_eq!(public_values(&[7; 32], true)[32], VERDICT_ACCEPT);
        assert_eq!(public_values(&[7; 32], false)[32], VERDICT_REJECT);
        // the 33-byte layout of the checker's public values: the host's PublicOutcome reads it unchanged
        let stmt = statement_bytes(Format::Bf16Ampere, &id, K as u32, &yl);
        let pv = public_values(&crate::frame::sha256(&stmt), true);
        let (object, digest, verdict) = crate::split_public_values(&pv).unwrap();
        assert!(object.is_none() && digest == crate::frame::sha256(&stmt) && verdict == VERDICT_ACCEPT);
    }

    #[test]
    fn an_all_zero_vu_is_plus_zero_in_every_format() {
        for format in Format::ALL {
            let chunk = vec![0u8; 8 * format.vu_words()];
            for (y, verdict) in [(0, VERDICT_ACCEPT), (1, VERDICT_REJECT)] {
                let yl = encode_y(format, &[y]).unwrap();
                assert_eq!(decide(&encode_header(format, K as u32, 1, &[3; 32], &yl), &[chunk.clone()])[32], verdict, "{format:?} y {y}");
            }
        }
    }

    #[test]
    fn frozen_set_matches_the_reference_and_the_committed_words() {
        let dir = fixture_dir();
        let (Some(x), Some(w), Some(y)) =
            (u16s(&dir.join("vu-k1536.x.u16")), u16s(&dir.join("vu-k1536.w.u16")), u16s(&dir.join("vu-k1536.y.u16")))
        else {
            eprintln!("SKIPPED: built vu-k1536 arrays not under {}", dir.display());
            return;
        };
        assert_eq!((x.len(), w.len(), y.len()), (4096 * K, 4096 * K, 4096));
        for i in 0..4096 {
            let (xr, wr) = (&x[K * i..K * (i + 1)], &w[K * i..K * (i + 1)]);
            let got = fast(xr, wr);
            assert_eq!(got, Some(y[i]), "instance {i}");
            assert_eq!(got, check_vu_reference(xr, wr), "instance {i}");
        }
        // the batch path: header + chunks as the guest reads them, several chunkings, and one flipped word
        let id = identity_digest(&identity_string("bench-instances/v1", "vu-k1536", 0, 300, MANIFEST));
        let (xb, wb) = (bytes_of(&x), bytes_of(&w));
        let f = Format::Bf16Ampere;
        for per in [1usize, 7, 64, 300] {
            let header = encode_header(f, K as u32, per as u32, &id, &y_le(&y[..300]));
            let chunks: Vec<Vec<u8>> = (0..300).step_by(per).map(|lo| encode_chunk(f, &xb, &wb, lo, (lo + per).min(300))).collect();
            let pv = decide(&header, &chunks);
            assert_eq!(pv[32], VERDICT_ACCEPT, "per {per}");
            assert_eq!(&pv[..32], &crate::frame::sha256(&statement_bytes(f, &id, K as u32, &y_le(&y[..300]))));
            let mut wrong = y[..300].to_vec();
            wrong[299] ^= 1;
            assert_eq!(decide(&encode_header(f, K as u32, per as u32, &id, &y_le(&wrong)), &chunks)[32], VERDICT_REJECT, "per {per}");
            // a chunk one VU short rejects
            let mut short = chunks.clone();
            let last = short.last_mut().unwrap();
            last.truncate(last.len() - 8 * VU_WORDS);
            assert_eq!(decide(&header, &short)[32], VERDICT_REJECT, "per {per}");
        }
    }

    #[test]
    fn every_negative_is_rejected_and_matches_the_reference() {
        let dir = fixture_dir();
        let x = u16s(&dir.join("vu-k1536-neg.x.u16")).unwrap();
        let w = u16s(&dir.join("vu-k1536-neg.w.u16")).unwrap();
        let claimed = u16s(&dir.join("vu-k1536-neg.y.u16")).unwrap();
        let index: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("vu-k1536-neg.index.json")).unwrap()).unwrap();
        let rows = index["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 52);
        for row in rows {
            let i = row[0].as_u64().unwrap() as usize;
            let (xr, wr) = (&x[K * i..K * (i + 1)], &w[K * i..K * (i + 1)]);
            let got = fast(xr, wr);
            assert_eq!(got, check_vu_reference(xr, wr), "negative {i}");
            assert_ne!(got, Some(claimed[i]), "negative {i} ({}) accepted its claimed word", row[1]);
            match row[5].as_str().unwrap() {
                "wrong" => assert_eq!(got, Some(row[4].as_u64().unwrap() as u16), "negative {i}"),
                "reject" => assert_eq!(got, None, "negative {i}"),
                other => panic!("verdict {other}"),
            }
            let id = identity_digest(&identity_string("bench-instances/v1", "vu-k1536-neg", i as u64, i as u64 + 1, MANIFEST));
            let f = Format::Bf16Ampere;
            let chunk = encode_chunk(f, &bytes_of(&x), &bytes_of(&w), i, i + 1);
            assert_eq!(decide(&encode_header(f, K as u32, 1, &id, &y_le(&claimed[i..i + 1])), &[chunk])[32], VERDICT_REJECT);
        }
    }

    /// Random rows from several regimes, each compared with the reference (verdicts included).
    #[test]
    fn adversarial_rows_match_the_reference() {
        let mut r = Rng(0x5eed_2026_0924);
        let mut counts = [0usize; 3]; // [accepted finite, accepted inf, rejected]
        let cases = if cfg!(debug_assertions) { 300 } else { 6000 };
        for case in 0..cases {
            let regime = case % 10;
            let mut x = vec![0u16; K];
            let mut w = vec![0u16; K];
            for k in 0..K {
                let (a, b) = match regime {
                    // the frozen set's regime: exponents around 1
                    0 => (word(&mut r, 110, 135, 5, 0, 0), word(&mut r, 110, 135, 5, 0, 0)),
                    // tiny magnitudes: subnormals, the floor, products below every other term
                    1 => (word(&mut r, 0, 20, 50, 200, 0), word(&mut r, 0, 20, 50, 200, 0)),
                    // wide exponents: truncation across the whole alignment range, saturation
                    2 => (word(&mut r, 0, 254, 20, 20, 0), word(&mut r, 60, 190, 20, 20, 0)),
                    // near the top: saturation somewhere in the chain
                    3 => (word(&mut r, 180, 254, 0, 0, 0), word(&mut r, 120, 140, 0, 0, 0)),
                    // sparse non-finite words
                    4 => (word(&mut r, 100, 150, 10, 10, 1), word(&mut r, 100, 150, 10, 10, 1)),
                    // mostly zeros
                    5 => (word(&mut r, 100, 150, 900, 30, 0), word(&mut r, 100, 150, 500, 30, 0)),
                    // around the subnormal / normal boundary of the products (sum of exponents near 127 + 1)
                    6 => (word(&mut r, 0, 70, 10, 100, 0), word(&mut r, 55, 70, 10, 100, 0)),
                    _ => (word(&mut r, 115, 140, 5, 5, 0), word(&mut r, 115, 140, 5, 5, 0)),
                };
                x[k] = a;
                w[k] = b;
            }
            match regime {
                // exact cancellation: some pairs of equal and opposite products
                7 => {
                    for k in (0..K).step_by(2) {
                        if r.next() & 3 != 0 {
                            x[k + 1] = x[k] ^ 0x8000;
                            w[k + 1] = w[k];
                        }
                    }
                }
                // saturation inside the last step's first group, with non-finite words after it
                8 => {
                    for k in 1520..1528 {
                        x[k] = 0x7F00 | (r.next() & 0x7F) as u16;
                        w[k] = 0x7F00 | (r.next() & 0x7F) as u16;
                    }
                    if case % 20 < 10 {
                        x[1528 + r.below(8) as usize] = 0x7FC0;
                    }
                }
                // saturation inside the last step's second group, or a non-finite word in it
                9 => {
                    let k = 1528 + r.below(8) as usize;
                    if case % 20 < 10 {
                        x[k] = 0x7F7F;
                        w[k] = 0x7F7F;
                    } else {
                        w[k] = if case % 3 == 0 { 0xFF80 } else { 0x7FFF };
                    }
                }
                _ => {}
            }
            let want = check_vu_reference(&x, &w);
            assert_eq!(fast(&x, &w), want, "case {case} (regime {regime})");
            counts[match want {
                None => 2,
                Some(v) if v & 0x7FFF == 0x7F80 => 1,
                Some(_) => 0,
            }] += 1;
        }
        eprintln!("adversarial rows: {} finite, {} +-inf, {} rejected", counts[0], counts[1], counts[2]);
        assert!(counts.iter().all(|&c| c > cases / 50), "every outcome exercised: {counts:?}");
    }

    #[test]
    fn short_rows_are_not_a_vu() {
        assert_eq!(check_vu(&[0; ROW_WORDS - 1], &[0; ROW_WORDS]), None);
        assert_eq!(check_vu(&[0; ROW_WORDS], &[0; ROW_WORDS]), Some(0));
        let bytes = [0u8; 8 * VU_WORDS];
        assert!(!check_words(Format::Bf16Ampere, &bytes, &[0; VU_WORDS], &[0]));
        assert!(check_words(Format::Bf16Ampere, &bytes, &[0; VU_WORDS], &[0, 0]));
        assert!(!check_words(Format::Bf16Ampere, &bytes[8..], &[0; VU_WORDS - 1], &[0, 0]));
        assert!(!check_words(Format::Bf16Ampere, &bytes[8..], &[0; VU_WORDS], &[0, 0]));
    }
}
