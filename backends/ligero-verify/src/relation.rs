//! The relations this verifier can check, each PINNED by the digests of its compiled constraint system: a statement
//! names its relation (`ligero-statement/v4`; a v2 / v3 file is the Ampere BF16 unit) and the verifier refuses any
//! system file whose `sys_id` (the Python `protocol.system_id`) and table digest (BLAKE3 of the exported tables) are
//! not the pinned pair -- unless `--allow-any-system`, which every report then flags as `system_pinned = false`.
//!
//! Everything else a relation fixes outside `system.bin` is mirrored here from the Python side
//! (`protocol.RelationHooks`, `relations.py` on lane b-hopper, `fp8/relation.py`): the statement-digest tag, the byte
//! widths the words are absorbed with, the range of the chain-end public word, and the operand decode the public pins
//! are computed from (`unit.public_vectors` / `fp8.relation.public_vectors_fp8`).  Nothing here is guessed from the
//! statement: a v4 file whose widths disagree with its named relation is rejected before a word is decoded.
//!
//! Provenance of the pinned digests (lane ligero-verify-relations, 2026-09-22; `ligero-verify system-digest`):
//!
//! * `bf16-ampere`: `compile_unit(Params.REAL, chain=True)` on main aecc298 (laptop, torch-free) == the committed
//!   `fixtures/system.bin` (lane b-verifier) == `proofs/system.bin` of every A100 / 4090 dump; the values pinned since
//!   lane b-verifier.
//! * `fp8-ada`: `compile_fp8_unit(ADA, chain=True)` on main aecc298 and, through `relations.relation("fp8-ada")`, on
//!   lane b-hopper a39de82 -- byte-identical files (sha256 cba9a09f...), two runs each; equal to `proofs/system.bin`
//!   of the pod dumps of lane ada-fp8-dumps (attempts r20260922-235542-295b, -235643-98f7, -235746-0929 at a4b0ac7,
//!   and r20260922-233711-6b9f at 7bb0ffa1; manifest `system_file.sha256` cba9a09f..., `sys_id` 6b570eef...).
//! * `bf16-hopper`, `fp8-hopper`: `relations.relation(name).compile(params, True)` from the lane b-hopper tree at
//!   a39de82 AND from the lane b-merge-h100 tree at 01fc64f (the merge that lands them on main): laptop, torch-free,
//!   two runs each, all four files byte-identical per relation (sha256 5bba40df... / 968d99d6...; 3292 / 3396 rows,
//!   the lane's reported counts).  Not yet cross-checked against a pod `system.bin`: b-hopper's own H100 dumps died
//!   with their pod and lane b-merge-h100's `h100-dumps-v1` had not landed when this was written.
//! * `fp4-nvf4` (lane ligero-verify-fp4, 2026-09-23): `fp4.relation.compile_fp4_unit()` on main 24bd337 (laptop,
//!   torch-free: compile.py + verity_numerical.checker), serialised by `serialize.system_bytes` with the `sys_id`
//!   from a transcription of `protocol.system_id` (protocol.py imports torch), two runs, byte-identical
//!   (`ligero-system/v2`, 95330 bytes, sha256 6bb9d77f...) == `dumps/system.bin` of the three RTX 5090 dumps of lane
//!   fp4-proof (r20260923-024035-f3cf, -024119-8da9, -024204-1f80 at 3b03017; manifest `sys_id` a825ba0b...).
//!   1583 rows, 306 linear, 1389 quadratic, 141 pins, 86 hints; the chain ends on (31, 1), (23, 8), (0, 23).

/// The operand word format: how a public word becomes the pinned `(s, m, e)` (`unit.public_vectors` for BF16,
/// `public_vectors_fp8` for E4M3), and which words are off the finite domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    /// sign 1 | exponent 8 | fraction 7; bias 127; the all-ones exponent field (infinities, NaN) is rejected.
    Bf16,
    /// sign 1 | exponent 4 | fraction 3; bias 7; the all-ones exponent field is finite except the NaN code 0x7F / 0xFF
    /// (`(w & 0x7F) == 0x7F` is rejected, as in `public_vectors_fp8`).
    E4m3,
}

impl Operand {
    /// `(s, m, e)` of a word: `s` the sign bit, `m = f + nz 2^frac` the significand, `e = t - nz - (bias - 1)` as a
    /// signed integer (the caller reduces mod p).  `Err` for a word off the finite domain.
    pub fn decode(self, w: u32) -> Result<(u32, u32, i64), String> {
        match self {
            Operand::Bf16 => {
                if w >> 16 != 0 {
                    return Err("operand word exceeds 16 bits".into());
                }
                let t = (w >> 7) & 0xFF;
                if t == 0xFF {
                    return Err("operand exponent field off the finite range".into());
                }
                let f = w & 0x7F;
                let nz = (t != 0) as u32;
                Ok((w >> 15, f + nz * 128, t as i64 - nz as i64 - 126))
            }
            Operand::E4m3 => {
                if w >> 8 != 0 {
                    return Err("operand word exceeds 8 bits".into());
                }
                if w & 0x7F == 0x7F {
                    return Err("E4M3 NaN operand".into());
                }
                let t = (w >> 3) & 0xF;
                let f = w & 0x7;
                let nz = (t != 0) as u32;
                Ok((w >> 7, f + nz * 8, t as i64 - nz as i64 - 6))
            }
        }
    }
}

/// The NVFP4 unit's public words (`fp4/relation.py`, `fp4/witness.py public_vectors_fp4`; the model
/// `BLACKWELL_SM120_NVF4.step`): per operand per unit 64 E2M1 codes (nibbles, one per byte) followed by 4 UE4M3 scale
/// bytes -- one scale per group of 16 codes.  The pins the verifier recomputes from them are the 128 numerators
/// `a[i].n` / `b[i].n`, per group the scale-mantissa product `g[g].sm`, the unit-weight exponent `g[g].X` and the
/// participation flag `g[g].part`, and the anchor `anc.G`.
pub mod nvf4 {
    /// E2M1 codes per operand per step (`mma.m16n8k64`), codes per scale block, blocks (= scale bytes), words per
    /// operand per step (the statement's K).
    pub const K: usize = 64;
    pub const GROUP: usize = 16;
    pub const GROUPS: usize = K / GROUP;
    pub const WORDS: usize = K + GROUPS;
    /// Adder bits below the leading participating group's unit weight, and the grid floor (`GRID_FLOOR`).
    pub const FRAC_BITS: i64 = 27;
    pub const GRID_FLOOR: i64 = -174;
    /// E2M1 magnitudes in halves: 0, .5, 1, 1.5, 2, 3, 4, 6.
    const NUMERATOR: [i64; 8] = [0, 1, 2, 3, 4, 6, 8, 12];

    /// The E2M1 nibble as a signed number of halves (`relation.numerator`); a word above 4 bits is off the domain.
    pub fn numerator(code: u32) -> Result<i64, String> {
        if code >= 16 {
            return Err("E2M1 code outside 4 bits".into());
        }
        let n = NUMERATOR[(code & 7) as usize];
        Ok(if code & 8 != 0 { -n } else { n })
    }

    /// The UE4M3 scale byte as `(mantissa, exponent)` with the value `mantissa 2^exponent` (`relation.decode_scale`,
    /// `models._scale_dyadic`): subnormal `m 2^-9` at exponent field 0, else `(8 + m) 2^(e - 10)`.  Bit 7 set (the
    /// PTX ISA's padding bit) and the NaN code 0x7F are off the domain.
    pub fn scale(byte: u32) -> Result<(i64, i64), String> {
        if byte >= 128 {
            return Err("UE4M3 scale byte with the padding bit set".into());
        }
        if byte == 0x7F {
            return Err("UE4M3 NaN scale".into());
        }
        let (e, m) = (((byte >> 3) & 0xF) as i64, (byte & 7) as i64);
        Ok(if e == 0 { (m, -9) } else { (8 + m, e - 10) })
    }

    /// The pinned values of one unit from its two operand rows of `WORDS` words each (`public_vectors_fp4` on one
    /// row), as signed integers in the order `a[0..K].n, b[0..K].n, g[0..GROUPS].(sm, X, part), anc.G`; the caller
    /// picks them by pin name (`pin_index`).
    pub fn unit_pins(a: &[u32], b: &[u32]) -> Result<Vec<i64>, String> {
        debug_assert!(a.len() == WORDS && b.len() == WORDS);
        let mut out = Vec::with_capacity(2 * K + 3 * GROUPS + 1);
        let mut na = [0i64; K];
        let mut nb = [0i64; K];
        for i in 0..K {
            na[i] = numerator(a[i])?;
            nb[i] = numerator(b[i])?;
        }
        out.extend_from_slice(&na);
        out.extend_from_slice(&nb);
        let mut g_anchor = GRID_FLOOR;
        for g in 0..GROUPS {
            let (ma, ea) = scale(a[K + g])?;
            let (mb, eb) = scale(b[K + g])?;
            let any_product = (GROUP * g..GROUP * (g + 1)).any(|i| na[i] * nb[i] != 0);
            let part = any_product && ma != 0 && mb != 0;
            let x = -2 + ea + eb;
            out.push(ma * mb);
            out.push(x);
            out.push(part as i64);
            if part {
                g_anchor = g_anchor.max(x - FRAC_BITS);
            }
        }
        out.push(g_anchor);
        Ok(out)
    }

    /// The position of a pin name in `unit_pins`' output: `a[i].n`, `b[i].n`, `g[g].sm` / `.X` / `.part`, `anc.G`.
    pub fn pin_index(name: &str) -> Result<usize, String> {
        let bad = || format!("pin {name}: not an fp4-nvf4 pin");
        if name == "anc.G" {
            return Ok(2 * K + 3 * GROUPS);
        }
        let (arr, rest) = name.split_at(1);
        let close = rest.find(']').ok_or_else(bad)?;
        if !rest.starts_with('[') {
            return Err(bad());
        }
        let i: usize = rest[1..close].parse().map_err(|_| bad())?;
        let comp = &rest[close + 1..];
        match (arr, comp) {
            ("a", ".n") if i < K => Ok(i),
            ("b", ".n") if i < K => Ok(K + i),
            ("g", ".sm") if i < GROUPS => Ok(2 * K + 3 * i),
            ("g", ".X") if i < GROUPS => Ok(2 * K + 3 * i + 1),
            ("g", ".part") if i < GROUPS => Ok(2 * K + 3 * i + 2),
            _ => Err(bad()),
        }
    }
}

/// The v2 units of lane relmin-lookup (`backends/direct/ligero/pubsel/relation.py`, `public_pins_int`): the verifier does the
/// public part of the tensor-core step itself and pins, per product group, the group maximum `g{g}.G` of the products'
/// exponents and, for every extra alignment shift `j in [0, PB)`, the signed sum `g{g}.Q[j]` of the products aligned to
/// `G + j` (truncating, as `AlignProduct`).  The prover's witness selects `Q[E - G]` with a lookup one-hot; what depends on
/// the private accumulator stays in the witness.  The BF16 Hopper sum exceeds p and is pinned in two limbs `.lo` (the low
/// `LIMB` bits) and `.hi` (the arithmetic shift, negative for a negative sum).
pub mod pubsel {
    use super::Operand;

    /// Limb width of the wide sum (`pubsel.relation.LIMB`).
    pub const LIMB: u32 = 16;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PubSel {
        pub op: Operand,
        /// `Params.floor`: the exponent of a zero product and the least group maximum.
        pub floor: i64,
        /// `prod_bits` / `prod_shift`: a product enters the adder as `u 2^PS < 2^PB`.
        pub pb: u32,
        pub ps: u32,
        /// The product groups of the step (`Params.groups`); K = their sum.
        pub groups: &'static [usize],
        /// The sum is pinned in two limbs.
        pub wide: bool,
    }

    impl PubSel {
        pub const fn k(&self) -> usize {
            let mut k = 0;
            let mut i = 0;
            while i < self.groups.len() {
                k += self.groups[i];
                i += 1;
            }
            k
        }

        /// Pins per group: `G` and `PB` sums (two limbs each when wide).
        pub const fn pins_per_group(&self) -> usize {
            1 + self.pb as usize * if self.wide { 2 } else { 1 }
        }

        /// `AlignProduct`: `u 2^PS` shifted right by `d`, truncating (zero at or beyond `PB`).
        pub fn aligned_product(&self, u: i64, d: i64) -> i64 {
            if d <= self.ps as i64 {
                u << (self.ps as i64 - d)
            } else if d < self.pb as i64 {
                u >> (d - self.ps as i64)
            } else {
                0
            }
        }

        /// The pinned values of one unit from its two operand rows of K words each (`public_pins_int`), as signed
        /// integers per group in the order `G, Q[0], .., Q[PB-1]` (`Q[j].lo, Q[j].hi` when wide); the caller picks
        /// them by pin name (`pin_index`).  A word off the finite domain is refused with the decode's reason.
        pub fn unit_pins(&self, a: &[u32], b: &[u32]) -> Result<Vec<i64>, String> {
            debug_assert!(a.len() == self.k() && b.len() == self.k());
            let mut out = Vec::with_capacity(self.groups.len() * self.pins_per_group());
            let mut start = 0usize;
            for &size in self.groups {
                let mut terms: Vec<(i64, i64, i64)> = Vec::with_capacity(size);
                for i in start..start + size {
                    let (sa, ma, ea) = self.op.decode(a[i])?;
                    let (sb, mb, eb) = self.op.decode(b[i])?;
                    let u = ma as i64 * mb as i64;
                    let e = if u != 0 { ea + eb } else { self.floor };
                    terms.push(((sa ^ sb) as i64, u, e));
                }
                start += size;
                let g = terms.iter().map(|t| t.2).fold(self.floor, i64::max);
                out.push(g);
                for j in 0..self.pb as i64 {
                    let q: i64 = terms.iter().map(|&(s, u, e)| (1 - 2 * s) * self.aligned_product(u, j + g - e)).sum();
                    if self.wide {
                        let lo = q & ((1i64 << LIMB) - 1);
                        out.push(lo);
                        out.push((q - lo) >> LIMB);
                    } else {
                        out.push(q);
                    }
                }
            }
            Ok(out)
        }

        /// The position of a pin name in `unit_pins`' output: `g{g}.G`, `g{g}.Q[j]` (`.lo` / `.hi` when wide).
        pub fn pin_index(&self, name: &str) -> Result<usize, String> {
            let bad = || format!("pin {name}: not a public-selection pin of this relation");
            let rest = name.strip_prefix('g').ok_or_else(bad)?;
            let dot = rest.find('.').ok_or_else(bad)?;
            let g: usize = rest[..dot].parse().map_err(|_| bad())?;
            if g >= self.groups.len() {
                return Err(bad());
            }
            let base = g * self.pins_per_group();
            let comp = &rest[dot..];
            if comp == ".G" {
                return Ok(base);
            }
            let inner = comp.strip_prefix(".Q[").ok_or_else(bad)?;
            let close = inner.find(']').ok_or_else(bad)?;
            let j: usize = inner[..close].parse().map_err(|_| bad())?;
            if j >= self.pb as usize {
                return Err(bad());
            }
            match (&inner[close + 1..], self.wide) {
                ("", false) => Ok(base + 1 + j),
                (".lo", true) => Ok(base + 1 + 2 * j),
                (".hi", true) => Ok(base + 2 + 2 * j),
                _ => Err(bad()),
            }
        }
    }

    /// `Params.from_model(HOPPER_BF16_M16N8K16)`: one group of 16, floor -133, products `u 2^11 < 2^27`, wide sum.
    pub static BF16_HOPPER: PubSel = PubSel { op: Operand::Bf16, floor: -133, pb: 27, ps: 11, groups: &[16], wide: true };
    /// `Params.from_model(AMPERE_BF16_M16N8K16)`: two groups of 8, floor -132, products `u 2^10 < 2^26`.
    pub static BF16_AMPERE: PubSel = PubSel { op: Operand::Bf16, floor: -132, pb: 26, ps: 10, groups: &[8, 8], wide: false };
    /// `FP8Params.from_model(HOPPER_E4M3_K32)`: one group of 32, floor -139, products `u 2^7 < 2^15`.
    pub static FP8_HOPPER: PubSel = PubSel { op: Operand::E4m3, floor: -139, pb: 15, ps: 7, groups: &[32], wide: false };
    /// `FP8Params.from_model(ADA_E4M3_M16N8K32)`: two groups of 16, floor -139, products `u 2^7 < 2^15`.
    pub static FP8_ADA: PubSel = PubSel { op: Operand::E4m3, floor: -139, pb: 15, ps: 7, groups: &[16, 16], wide: false };
}

/// The v3 units of lane relmin-private (`backends/direct/ligero/privsel/relation.py`, `public_vectors_v3`): the operand
/// words are committed witness wires; the verifier pins the words THEMSELVES -- `a[i]` / `b[i]` one word per pin, or
/// `a[lo:hi]` / `b[lo:hi]` the run of `words_per_pin` words packed `sum_j w_j 2^(word_bits (j - lo))` (3 E4M3 bytes per
/// pin, 1 BF16 word) -- after the same finiteness check as the v1 decode (a non-finite word is refused; the relation
/// itself excludes every other off-domain encoding).
pub mod privsel {
    use super::Operand;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PrivSel {
        pub op: Operand,
        /// Operands per unit (the sum of the relation's product groups).
        pub k: usize,
        /// Words packed into one pin row (`ShiftPlan.words_per_pin` = 30 / word bits).
        pub words_per_pin: usize,
    }

    impl PrivSel {
        pub const fn word_bits(&self) -> u32 {
            match self.op {
                Operand::Bf16 => 16,
                Operand::E4m3 => 8,
            }
        }

        /// Pins per unit: the packed runs of both operands.
        pub const fn pins(&self) -> usize {
            2 * self.k.div_ceil(self.words_per_pin)
        }

        /// The pinned values of one unit from its two operand rows: per operand, the packed runs in order
        /// (`a` runs then `b` runs); the caller picks them by pin name (`pin_index`).
        pub fn unit_pins(&self, a: &[u32], b: &[u32]) -> Result<Vec<i64>, String> {
            if a.len() != self.k || b.len() != self.k {
                return Err(format!("unit has {} + {} operand words, the relation takes {} + {}", a.len(), b.len(), self.k, self.k));
            }
            let mut out = Vec::with_capacity(self.pins());
            for words in [a, b] {
                for lo in (0..self.k).step_by(self.words_per_pin) {
                    let hi = (lo + self.words_per_pin).min(self.k);
                    let mut packed: i64 = 0;
                    for j in lo..hi {
                        self.op.decode(words[j])?;
                        packed += (words[j] as i64) << (self.word_bits() * (j - lo) as u32);
                    }
                    out.push(packed);
                }
            }
            Ok(out)
        }

        /// The position of a pin name in `unit_pins`' output: `a[i]` / `b[i]` (one word per pin) or `a[lo:hi]` /
        /// `b[lo:hi]` (exactly the run starting at a multiple of `words_per_pin`).
        pub fn pin_index(&self, name: &str) -> Result<usize, String> {
            let bad = || format!("pin {name}: not an operand-word pin of this relation");
            let (arr, rest) = name.split_at(1);
            let base = match arr {
                "a" => 0,
                "b" => self.k.div_ceil(self.words_per_pin),
                _ => return Err(bad()),
            };
            let inner = rest.strip_prefix('[').and_then(|r| r.strip_suffix(']')).ok_or_else(bad)?;
            let (lo, hi) = match inner.find(':') {
                None => {
                    let i: usize = inner.parse().map_err(|_| bad())?;
                    (i, i + 1)
                }
                Some(c) => (inner[..c].parse().map_err(|_| bad())?, inner[c + 1..].parse().map_err(|_| bad())?),
            };
            if lo % self.words_per_pin != 0 || lo >= self.k || hi != (lo + self.words_per_pin).min(self.k) {
                return Err(bad());
            }
            Ok(base + lo / self.words_per_pin)
        }
    }

    pub static BF16_HOPPER: PrivSel = PrivSel { op: Operand::Bf16, k: 16, words_per_pin: 1 };
    pub static BF16_AMPERE: PrivSel = PrivSel { op: Operand::Bf16, k: 16, words_per_pin: 1 };
    pub static FP8_HOPPER: PrivSel = PrivSel { op: Operand::E4m3, k: 32, words_per_pin: 3 };
    pub static FP8_ADA: PrivSel = PrivSel { op: Operand::E4m3, k: 32, words_per_pin: 3 };
}

/// How a relation's public pins are computed from the statement's operand words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decode {
    /// One word per operand, one pin per word component: `a[i].s|m|e` / `b[i].s|m|e` (`unit.public_vectors`,
    /// `public_vectors_fp8`); K = 1 + the largest pinned index.
    Word(Operand),
    /// The NVFP4 unit: K = 68 words per operand (64 nibbles + 4 scale bytes), pins per `nvf4::unit_pins`.
    Nvf4,
    /// A v2 public-selection unit: K = the sum of the groups, pins per `pubsel::PubSel::unit_pins`.
    PubSel(&'static pubsel::PubSel),
    /// A v3 private-operand unit: the (packed) words themselves, pins per `privsel::PrivSel::unit_pins`.
    PrivSel(&'static privsel::PrivSel),
}

#[derive(Debug)]
pub struct Relation {
    /// The `relation` string of a v4 statement (also the `--relation` value of `backends.direct.ligero.run`).
    pub name: &'static str,
    /// Other spellings a v4 statement may carry for the same relation.
    pub aliases: &'static [&'static str],
    /// `RelationHooks.tag`: absorbed right after the statement tag (`b""` for the BF16 relations).
    pub tag: &'static [u8],
    /// `RelationHooks.word_dtype` / `y16_dtype` item sizes: the widths the words are written AND absorbed with.
    pub word_bytes: u8,
    pub y_bytes: u8,
    /// `RelationHooks.y16_bits`: the chain-end public word is in `[0, 2^y_bits)`.
    pub y_bits: u32,
    pub decode: Decode,
    /// The chain end of the relation's compiled system: `None` = one public word compared as a residue (a
    /// `ligero-system/v1` chain, `y16 < 2^y_bits`); `Some` = the `(lo, bits)` bit fields of the public word a
    /// `ligero-system/v2` chain ends on (`compile_fp4_unit`: sign 31/1, exponent field 23/8, fraction 0/23).  A
    /// system file whose chain end is not this is refused for the relation whatever `--allow-any-system` says: the
    /// public values the chain test compares against would otherwise be read from the file.
    pub end_components: Option<&'static [(u32, u32)]>,
    /// The pinned `protocol.system_id` (the last 32 bytes of `system.bin`) and this verifier's own table digest.
    pub sys_id: &'static str,
    pub table_digest: &'static str,
    /// The same two digests of the HASHED system (`hashchain.compose`: operands private, two Poseidon2 sponges along
    /// the chain, the digests pinned at the chain end) a v5 statement (`authentication = included-hash`) is checked
    /// against; empty when the relation is not composed (bf16-ampere, fp4-nvf4).
    pub hashed_sys_id: &'static str,
    pub hashed_table_digest: &'static str,
}

pub static BF16_AMPERE: Relation = Relation {
    name: "bf16-ampere",
    aliases: &["bf16"],
    tag: b"",
    word_bytes: 2,
    y_bytes: 2,
    y_bits: 16,
    decode: Decode::Word(Operand::Bf16),
    end_components: None,
    sys_id: "44cb05b9e6988219beba2d66582124e67a1893aac39d0b66126ce0b9d4db89cb",
    table_digest: "c147cc4cece7d61cd3046828410d18f487dcde82f709044f4b4f97e282a73a95",
    hashed_sys_id: "",
    hashed_table_digest: "",
};

pub static FP8_ADA: Relation = Relation {
    name: "fp8-ada",
    aliases: &[],
    tag: b"|rel=fp8-ada|v1",
    word_bytes: 1,
    y_bytes: 4,
    y_bits: 22,
    decode: Decode::Word(Operand::E4m3),
    end_components: None,
    sys_id: "6b570eef0c7faa07ccd1dddb2508e2699bd5367b108b15920469d46d16073d37",
    table_digest: "443e4fc74c0ac0cc864e50f7125d9585d71986c0b95ff5336d93a6939f3cb952",
    hashed_sys_id: "c4c4b403e07da47f7747b947ad76899ef231112bcc6d62dc0448a83ef8696618",
    hashed_table_digest: "c0e2f48119e2c83ca81cbc8eb1cbba51a676b67a600b8125d75bf438b770e05d",
};

pub static BF16_HOPPER: Relation = Relation {
    name: "bf16-hopper",
    aliases: &[],
    tag: b"",
    word_bytes: 2,
    y_bytes: 2,
    y_bits: 16,
    decode: Decode::Word(Operand::Bf16),
    end_components: None,
    sys_id: "9dcc7cdc1377a00d877440cc01a0dcdb73b3e989d86df8244eccb35a81be0df4",
    table_digest: "74ce68a2801e14fbeaab2b241886bbe3fa9146234289e52c774fd4160a6c469c",
    hashed_sys_id: "3c7570fa029de39af30363599480c7a7bdbb51ba8e5fbe552a852b2ffb0ba1e1",
    hashed_table_digest: "def0ccdae79f092f41bc35ecd054ad64e54e2cfe9769c8cec41737c1ebab6617",
};

pub static FP8_HOPPER: Relation = Relation {
    name: "fp8-hopper",
    aliases: &[],
    tag: b"|rel=fp8-hopper|v1",
    word_bytes: 1,
    y_bytes: 4,
    y_bits: 22,
    decode: Decode::Word(Operand::E4m3),
    end_components: None,
    sys_id: "c7cbebe365cb5f773a679bc13b5382fb45714511b1912e4e4ab7fb48accd4bc3",
    table_digest: "d5b77463fee9b3d777174755144edaa84461b3eec6ac6a1750c887dc861d58e7",
    hashed_sys_id: "141a5fdd5ea2314ba7070f35d8eae1cb649c554589b1065033e5382580587c97",
    hashed_table_digest: "801ca79f0f8c5c77e6b90758614acadfc4e20688442e616e756dbd14a464144d",
};

/// `GemmAccumulator<1536>` over `sm120.mma.m16n8k64.e2m1.nvf4` (24 chained K = 64 steps; `fp4/chain.py FP4_HOOKS`:
/// 8-bit words, the full FP32 accumulator word in a 32-bit slot, compared as three components).
pub static FP4_NVF4: Relation = Relation {
    name: "fp4-nvf4",
    aliases: &[],
    tag: b"|rel=fp4-nvf4|v1",
    word_bytes: 1,
    y_bytes: 4,
    y_bits: 32,
    decode: Decode::Nvf4,
    end_components: Some(&[(31, 1), (23, 8), (0, 23)]),
    sys_id: "a825ba0b826a15366bfb8cdd720a5e9978603cbe589491abcbba7346ad35bf3c",
    table_digest: "b597b003e55b8cf1025b517a04d442d9c6d8704f38fd06df13a21b3273b2d2f6",
    hashed_sys_id: "",
    hashed_table_digest: "",
};

/// The v2 public-selection relations (lane relmin-lookup, 2026-09-23): the same claims as their v1 counterparts (pinned
/// instruction semantics, statement widths, chain end on one public word), compiled by `pubsel.relation.compile_v2_unit`
/// -- `relations.relation(name).compile(params, True)` from the lane relmin-lookup tree (the picked-value selection
/// without product rows, lookup keys ranged by their one-hots), laptop, torch-free, two runs each, byte-identical
/// (`ligero-system/v1`; sha256 b8c98eec... / fa3ee053... / 2f74bb3a... / 6e1d87cc...; 328 / 505 / 208 / 370 rows;
/// 55 / 54 / 16 / 32 pins), the `sys_id`s equal to `protocol.system_id` under torch.  `bf16-ampere-v2` is the
/// Ampere BF16 pipeline on the generic chain runner (`tag` empty like the other BF16 relations; the statement names it).
pub static BF16_HOPPER_V2: Relation = Relation {
    name: "bf16-hopper-v2",
    aliases: &[],
    tag: b"",
    word_bytes: 2,
    y_bytes: 2,
    y_bits: 16,
    decode: Decode::PubSel(&pubsel::BF16_HOPPER),
    end_components: None,
    sys_id: "cf99d1ce739e621fca7ef52d86cdec3c43aae58464e370da745901316a70b831",
    table_digest: "bc4b95ec3576a22bf16a28ae5d0e83bdbc8fe4f7806dc28b1f6b304c6602bf7f",
    hashed_sys_id: "",
    hashed_table_digest: "",
};

pub static BF16_AMPERE_V2: Relation = Relation {
    name: "bf16-ampere-v2",
    aliases: &[],
    tag: b"",
    word_bytes: 2,
    y_bytes: 2,
    y_bits: 16,
    decode: Decode::PubSel(&pubsel::BF16_AMPERE),
    end_components: None,
    sys_id: "15d793e74cc2b380555920c740c135bbd9e936fe925b2c4ca84f1da2943e7128",
    table_digest: "e70b77d58569b1de95d1304d71f5a4aac6df2ac88581054058a36d7fe03c1c66",
    hashed_sys_id: "",
    hashed_table_digest: "",
};

pub static FP8_HOPPER_V2: Relation = Relation {
    name: "fp8-hopper-v2",
    aliases: &[],
    tag: b"|rel=fp8-hopper-v2|v1",
    word_bytes: 1,
    y_bytes: 4,
    y_bits: 22,
    decode: Decode::PubSel(&pubsel::FP8_HOPPER),
    end_components: None,
    sys_id: "e2a1573bc37cbd86af3184181da7ca404e60dfab6e82f93ff6fff2177fdaf493",
    table_digest: "a473c735fa661e1f8752679b51b15d935c2a0857ec7f60055b07d7b140169123",
    hashed_sys_id: "",
    hashed_table_digest: "",
};

pub static FP8_ADA_V2: Relation = Relation {
    name: "fp8-ada-v2",
    aliases: &[],
    tag: b"|rel=fp8-ada-v2|v1",
    word_bytes: 1,
    y_bytes: 4,
    y_bits: 22,
    decode: Decode::PubSel(&pubsel::FP8_ADA),
    end_components: None,
    sys_id: "f23461f82dec60ba24ce05000cc91b72dabcce3c87c249d0cee61a6cdba9eb36",
    table_digest: "82b3041e8cbcf0fc7db77fa1b8693c4517b6d8e62d28e4f1f87ea8e8c5782b67",
    hashed_sys_id: "",
    hashed_table_digest: "",
};

/// The v3 private-operand relations (lane relmin-private, 2026-09-23): v1's statement on committed operand words, the
/// public words pinned as themselves (`privsel::PrivSel`), compiled by `privsel.relation.compile_v3_unit` --
/// `relations.relation(name).compile(params, True)` from the lane relmin-private tree, laptop, torch-free, two runs each,
/// byte-identical (`ligero-system/v1`; 1535 / 1645 / 1705 / 1838 rows per unit; 32 / 32 / 22 / 22 pins), the `sys_id`s
/// equal to `protocol.system_id`.
pub static BF16_HOPPER_V3: Relation = Relation {
    name: "bf16-hopper-v3",
    aliases: &[],
    tag: b"|rel=bf16-hopper-v3|v1",
    word_bytes: 2,
    y_bytes: 2,
    y_bits: 16,
    decode: Decode::PrivSel(&privsel::BF16_HOPPER),
    end_components: None,
    sys_id: "2279f2acae7fc92a903d0b47abf3b962541a917809e4fc8bc68cd061341e796e",
    table_digest: "0caba95acd43d3a795333a59e8826cf26a994cc9e34400b1be3c5502fd60a66a",
    hashed_sys_id: "",
    hashed_table_digest: "",
};

pub static BF16_AMPERE_V3: Relation = Relation {
    name: "bf16-ampere-v3",
    aliases: &[],
    tag: b"|rel=bf16-ampere-v3|v1",
    word_bytes: 2,
    y_bytes: 2,
    y_bits: 16,
    decode: Decode::PrivSel(&privsel::BF16_AMPERE),
    end_components: None,
    sys_id: "f962956b8d8d48f2711d9c452630394f919de6602171bcf31ecc2bfc7687af79",
    table_digest: "35fe22ff9b2f444497545389c42860a673208fb18f9ac2030c5e4125ba090f19",
    hashed_sys_id: "",
    hashed_table_digest: "",
};

pub static FP8_HOPPER_V3: Relation = Relation {
    name: "fp8-hopper-v3",
    aliases: &[],
    tag: b"|rel=fp8-hopper-v3|v1",
    word_bytes: 1,
    y_bytes: 4,
    y_bits: 22,
    decode: Decode::PrivSel(&privsel::FP8_HOPPER),
    end_components: None,
    sys_id: "bc6f87807650041969780bdd1909cf00fc8c203da25e73513fcc536e946ad1f5",
    table_digest: "a21f5b46aa80a824d82ef90bc68cd0c441949db1f0bf3dac598fcdeab49bd859",
    hashed_sys_id: "",
    hashed_table_digest: "",
};

pub static FP8_ADA_V3: Relation = Relation {
    name: "fp8-ada-v3",
    aliases: &[],
    tag: b"|rel=fp8-ada-v3|v1",
    word_bytes: 1,
    y_bytes: 4,
    y_bits: 22,
    decode: Decode::PrivSel(&privsel::FP8_ADA),
    end_components: None,
    sys_id: "adaea5d90ba579b5c70ac0871aace655644813df993f02dea5f7e0033cb6e30f",
    table_digest: "c62f82e063c7b690f7e7a6cdab288f9dcb518503123a3575a8181b1d0f8bb2aa",
    hashed_sys_id: "",
    hashed_table_digest: "",
};

pub static RELATIONS: [&Relation; 13] = [
    &BF16_AMPERE, &FP8_ADA, &BF16_HOPPER, &FP8_HOPPER, &FP4_NVF4, &BF16_HOPPER_V2, &BF16_AMPERE_V2, &FP8_HOPPER_V2, &FP8_ADA_V2,
    &BF16_HOPPER_V3, &BF16_AMPERE_V3, &FP8_HOPPER_V3, &FP8_ADA_V3,
];

/// The relation a statement names: `None` (a v2 / v3 file) is the Ampere BF16 unit; an unknown name is refused (its
/// tag and widths are not in the file, so nothing could be checked).
pub fn for_statement(name: Option<&str>) -> Result<&'static Relation, String> {
    match name {
        None => Ok(&BF16_AMPERE),
        Some(n) => RELATIONS
            .iter()
            .copied()
            .find(|r| r.name == n || r.aliases.contains(&n))
            .ok_or_else(|| format!("statement names relation {n:?}; this build pins {}", names().join(", "))),
    }
}

/// The relation whose pinned digests a system file carries, if any, and whether it is the hashed system.
pub fn by_digests(sys_id_hex: &str, table_digest_hex: &str) -> Option<(&'static Relation, bool)> {
    RELATIONS.iter().copied().find_map(|r| {
        if r.sys_id == sys_id_hex && r.table_digest == table_digest_hex {
            Some((r, false))
        } else if !r.hashed_sys_id.is_empty() && r.hashed_sys_id == sys_id_hex && r.hashed_table_digest == table_digest_hex {
            Some((r, true))
        } else {
            None
        }
    })
}

pub fn names() -> Vec<&'static str> {
    RELATIONS.iter().map(|r| r.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_decode_matches_unit_public_vectors() {
        // 1.0 = 0x3F80: t = 127, f = 0 -> m = 128, e = 0; -2.0 = 0xC000: t = 128 -> m = 128, e = 1; a subnormal
        // 0x0001: t = 0, f = 1 -> m = 1, e = -126; the smallest normal 0x0080: t = 1 -> m = 128, e = -126
        assert_eq!(Operand::Bf16.decode(0x3F80), Ok((0, 128, 0)));
        assert_eq!(Operand::Bf16.decode(0xC000), Ok((1, 128, 1)));
        assert_eq!(Operand::Bf16.decode(0x0001), Ok((0, 1, -126)));
        assert_eq!(Operand::Bf16.decode(0x0080), Ok((0, 128, -126)));
        assert_eq!(Operand::Bf16.decode(0x8000), Ok((1, 0, -126)));
        assert!(Operand::Bf16.decode(0x7F80).is_err(), "+inf");
        assert!(Operand::Bf16.decode(0xFFC0).is_err(), "NaN");
        assert!(Operand::Bf16.decode(0x1_0000).is_err());
    }

    #[test]
    fn e4m3_decode_matches_public_vectors_fp8() {
        // 1.0 = 0x38: t = 7, f = 0 -> m = 8, e = 0; 448 = 0x7E: t = 15, f = 6 -> m = 14, e = 8 (finite: E4M3 has no
        // infinities); 2^-9 = 0x01: t = 0, f = 1 -> m = 1, e = -6; -0 = 0x80
        assert_eq!(Operand::E4m3.decode(0x38), Ok((0, 8, 0)));
        assert_eq!(Operand::E4m3.decode(0x7E), Ok((0, 14, 8)));
        assert_eq!(Operand::E4m3.decode(0x01), Ok((0, 1, -6)));
        assert_eq!(Operand::E4m3.decode(0x80), Ok((1, 0, -6)));
        assert_eq!(Operand::E4m3.decode(0xB8), Ok((1, 8, 0)));
        assert!(Operand::E4m3.decode(0x7F).is_err(), "NaN");
        assert!(Operand::E4m3.decode(0xFF).is_err(), "-NaN");
        assert!(Operand::E4m3.decode(0x100).is_err());
    }

    #[test]
    fn e2m1_and_ue4m3_decodes_match_fp4_relation_py() {
        // relation.numerator: halves 0, 1, 2, 3, 4, 6, 8, 12; bit 3 the sign
        assert_eq!((0..8).map(|c| nvf4::numerator(c).unwrap()).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 6, 8, 12]);
        assert_eq!(nvf4::numerator(0x8), Ok(0), "-0");
        assert_eq!(nvf4::numerator(0xF), Ok(-12));
        assert_eq!(nvf4::numerator(0x9), Ok(-1));
        assert!(nvf4::numerator(16).is_err());
        assert!(nvf4::numerator(0xFF).is_err());
        // relation.decode_scale: 1.0 = 0x38 (e = 7): (8, -3); 0x00: (0, -9); 0x07: (7, -9) subnormal; 0x08 (e = 1): (8, -9);
        // 0x7E (e = 15, m = 6): (14, 5); 0x7F NaN; 0x80 padding bit
        assert_eq!(nvf4::scale(0x38), Ok((8, -3)));
        assert_eq!(nvf4::scale(0x00), Ok((0, -9)));
        assert_eq!(nvf4::scale(0x07), Ok((7, -9)));
        assert_eq!(nvf4::scale(0x08), Ok((8, -9)));
        assert_eq!(nvf4::scale(0x7E), Ok((14, 5)));
        assert_eq!(nvf4::scale(0x7F).unwrap_err(), "UE4M3 NaN scale");
        assert_eq!(nvf4::scale(0x80).unwrap_err(), "UE4M3 scale byte with the padding bit set");
        assert_eq!(nvf4::scale(0xFF).unwrap_err(), "UE4M3 scale byte with the padding bit set");
    }

    #[test]
    fn nvf4_unit_pins_match_public_vectors_fp4() {
        use nvf4::{pin_index, unit_pins, GROUPS, K, WORDS};
        // all codes 1.0 (0x2), scales 1.0 (0x38): every product 4 (halves squared), every group participates,
        // sm = 64, X = -2 + (-3) + (-3) = -8, G = -8 - 27 = -35
        let a = vec![0x2u32; K].into_iter().chain(vec![0x38u32; GROUPS]).collect::<Vec<_>>();
        let v = unit_pins(&a, &a).unwrap();
        assert_eq!(v.len(), 2 * K + 3 * GROUPS + 1);
        assert!(v[..2 * K].iter().all(|&n| n == 2));
        for g in 0..GROUPS {
            assert_eq!(&v[2 * K + 3 * g..2 * K + 3 * g + 3], &[64, -8, 1]);
        }
        assert_eq!(v[2 * K + 3 * GROUPS], -35);
        // group 1's codes all zero in a: no product -> not participating whatever the scales; a zero scale mantissa
        // (byte 0x00) in group 2 of b: not participating, sm = 0; group 3 with a subnormal scale (0x03: m = 3 at 2^-9)
        // and the top scale (0x7E: 14 at 2^5): X = -2 - 9 + 5 = -6, sm = 42, part; a -6 (0xF) x 6 (0x7) code: -144
        let mut a2 = a.clone();
        let mut b2 = a.clone();
        for i in 16..32 {
            a2[i] = 0x8; // -0
        }
        b2[K + 2] = 0x00;
        a2[K + 3] = 0x03;
        b2[K + 3] = 0x7E;
        a2[48] = 0xF;
        b2[48] = 0x7;
        let v = unit_pins(&a2, &b2).unwrap();
        assert_eq!(v[48], -12);
        assert_eq!(v[K + 48], 12);
        assert_eq!(&v[2 * K + 3..2 * K + 6], &[64, -8, 0], "group 1: no nonzero product");
        assert_eq!(&v[2 * K + 6..2 * K + 9], &[0, -14, 0], "group 2: zero scale mantissa in b (X = -2 - 3 - 9)");
        assert_eq!(&v[2 * K + 9..2 * K + 12], &[42, -6, 1], "group 3");
        assert_eq!(v[2 * K + 12], -6 - 27, "G = max participating X - 27 over groups 0 and 3");
        // nothing participates: G = -174
        let zero = vec![0u32; WORDS];
        let v = unit_pins(&zero, &a).unwrap();
        assert!(v[2 * K..2 * K + 3 * GROUPS].chunks(3).all(|c| c == [0, -2 - 9 - 3, 0]));
        assert_eq!(v[2 * K + 3 * GROUPS], -174);
        // off-domain words are refused with the decode's reason
        let mut bad = a.clone();
        bad[K] = 0x7F;
        assert_eq!(unit_pins(&bad, &a).unwrap_err(), "UE4M3 NaN scale");
        bad[K] = 0x80;
        assert_eq!(unit_pins(&a, &bad).unwrap_err(), "UE4M3 scale byte with the padding bit set");
        bad[K] = 0x38;
        bad[5] = 16;
        assert_eq!(unit_pins(&bad, &a).unwrap_err(), "E2M1 code outside 4 bits");
        // pin names -> positions
        assert_eq!(pin_index("a[0].n"), Ok(0));
        assert_eq!(pin_index("a[63].n"), Ok(63));
        assert_eq!(pin_index("b[0].n"), Ok(64));
        assert_eq!(pin_index("g[0].sm"), Ok(128));
        assert_eq!(pin_index("g[3].part"), Ok(128 + 11));
        assert_eq!(pin_index("anc.G"), Ok(140));
        for bad in ["a[64].n", "a[0].s", "g[4].X", "c.s", "y.t", "a[0]", "anc.lsb"] {
            assert!(pin_index(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn pubsel_unit_pins_match_public_pins_int() {
        // reference values: pubsel.relation.public_pins_int on the same words (lane relmin-lookup, bee4c6d)
        use pubsel::{BF16_AMPERE, BF16_HOPPER, FP8_ADA, FP8_HOPPER};
        assert_eq!((BF16_HOPPER.k(), BF16_AMPERE.k(), FP8_HOPPER.k(), FP8_ADA.k()), (16, 16, 32, 32));
        assert_eq!((BF16_HOPPER.pins_per_group(), BF16_AMPERE.pins_per_group(), FP8_ADA.pins_per_group()), (55, 27, 16));
        // bf16-hopper, all 1.0: 16 products 128^2 at exponent 0, G = 0, Q[j] = 16 * 2^14 * 2^(11 - j) in two 16-bit limbs
        let one = vec![0x3F80u32; 16];
        let v = BF16_HOPPER.unit_pins(&one, &one).unwrap();
        assert_eq!(v.len(), 55);
        assert_eq!(v[0], 0);
        let q: Vec<i64> = (0..27).map(|j| v[1 + 2 * j] + (v[2 + 2 * j] << 16)).collect();
        assert_eq!(&q[..3], &[1 << 29, 1 << 28, 1 << 27]);
        assert_eq!(q[11], 16 * 16384, "j = PS: no shift");
        assert_eq!(q[25], 16 * 16384 >> 14);
        assert_eq!(q[26], 0, "beyond PB - 1 - PS + 14 bits: truncated away");
        assert_eq!(&v[1..9], &[0, 8192, 0, 4096, 0, 2048, 0, 1024]);
        // mixed: a[0] = -2.0, a[1] = 0 (its exponent is the floor: G is set by the others), a[2] = 2^-133 (subnormal
        // 0x0001) x 2.0, a[3] = 1.5 x -3.0 (exponent 1 -> G = 1); a negative sum's .hi limb is negative
        let mut a = one.clone();
        let mut b = one.clone();
        a[0] = 0xC000;
        a[1] = 0x0000;
        a[2] = 0x0001;
        b[2] = 0x4000;
        a[3] = 0x3FC0;
        b[3] = 0xC040;
        let v = BF16_HOPPER.unit_pins(&a, &b).unwrap();
        assert_eq!(v[0], 1);
        assert_eq!(&v[1..9], &[0, 1408, 0, 704, 0, 352, 0, 176]);
        assert_eq!(&v[17..25], &[32768, 5, 49152, 2, 24576, 1, 45056, 0]);
        assert_eq!(&v[49..55], &[0, 6, 0, 65533, -1, 65535, -1][1..]);
        // all zero products: G = floor, every Q = 0
        let v = BF16_HOPPER.unit_pins(&vec![0u32; 16], &one).unwrap();
        assert_eq!(v[0], -133);
        assert!(v[1..].iter().all(|&x| x == 0));
        // all -1.0 x 1.0: Q[0] = -2^29 = lo 0, hi -8192; Q[14] = -2^15 = lo 32768, hi -1
        let v = BF16_HOPPER.unit_pins(&vec![0xBF80u32; 16], &one).unwrap();
        assert_eq!(&v[1..5], &[0, -8192, 0, -4096]);
        assert_eq!(&v[29..33], &[32768, -1, 49152, -1]);
        // bf16-ampere: group 0 all 1.0 (G = 0, Q[0] = 8 * 2^14 * 2^10), group 1 subnormal 2^-133 x 1.0 (G = -126,
        // Q[0] = 8 * 128 * 2^10)
        let a: Vec<u32> = vec![0x3F80; 8].into_iter().chain(vec![0x0001; 8]).collect();
        let v = BF16_AMPERE.unit_pins(&a, &one).unwrap();
        assert_eq!(v.len(), 54);
        assert_eq!(v[0], 0);
        assert_eq!(&v[1..4], &[134217728, 67108864, 33554432]);
        assert_eq!(v[25], 8, "Q[24] = 8 products of 128^2 >> 14");
        assert_eq!(v[26], 0);
        assert_eq!(v[27], -126);
        assert_eq!(&v[28..31], &[1048576, 524288, 262144]);
        assert_eq!(v[27 + 18], 8);
        assert!(v[27 + 19..].iter().all(|&x| x == 0));
        // fp8-hopper, all 1.0 (0x38): 32 x 64 x 2^7 = 2^18 at G = 0; mixed: 448 x -1.0 (14 x 8 at exponent 8 -> G = 8),
        // 2^-9 x 1.0 (1 x 8 at -6), a zero, the rest 1.0 at 0: truncation of the small terms
        let one8 = vec![0x38u32; 32];
        let v = FP8_HOPPER.unit_pins(&one8, &one8).unwrap();
        assert_eq!(v.len(), 16);
        assert_eq!(&v[..4], &[0, 262144, 131072, 65536]);
        assert_eq!(v[15], 0);
        let mut a = one8.clone();
        let mut b = one8.clone();
        a[0] = 0x7E;
        b[0] = 0xB8;
        a[1] = 0x01;
        a[2] = 0x00;
        let v = FP8_HOPPER.unit_pins(&a, &b).unwrap();
        assert_eq!(&v[..5], &[8, -13408, -6704, -3352, -1676]);
        assert_eq!(&v[12..16], &[-7, -3, -1, 0]);
        // fp8-ada: the same 16 words in group 0 (its sum without the other 16 ones), group 1 all zero (G = floor)
        let a2: Vec<u32> = a[..16].iter().copied().chain(vec![0; 16]).collect();
        let b2: Vec<u32> = b[..16].iter().copied().chain(vec![0x38; 16]).collect();
        let v = FP8_ADA.unit_pins(&a2, &b2).unwrap();
        assert_eq!(v.len(), 32);
        assert_eq!(&v[..4], &[8, -13920, -6960, -3480]);
        assert_eq!(v[16], -139);
        assert!(v[17..].iter().all(|&x| x == 0));
        // off-domain words are refused with the decode's reason
        let mut bad = one.clone();
        bad[7] = 0x7F80;
        assert_eq!(BF16_HOPPER.unit_pins(&one, &bad).unwrap_err(), "operand exponent field off the finite range");
        let mut bad8 = one8.clone();
        bad8[31] = 0xFF;
        assert_eq!(FP8_HOPPER.unit_pins(&bad8, &one8).unwrap_err(), "E4M3 NaN operand");
        // pin names -> positions
        assert_eq!(BF16_HOPPER.pin_index("g0.G"), Ok(0));
        assert_eq!(BF16_HOPPER.pin_index("g0.Q[0].lo"), Ok(1));
        assert_eq!(BF16_HOPPER.pin_index("g0.Q[0].hi"), Ok(2));
        assert_eq!(BF16_HOPPER.pin_index("g0.Q[26].hi"), Ok(54));
        assert_eq!(BF16_AMPERE.pin_index("g1.G"), Ok(27));
        assert_eq!(BF16_AMPERE.pin_index("g1.Q[25]"), Ok(53));
        assert_eq!(FP8_ADA.pin_index("g1.Q[14]"), Ok(31));
        assert_eq!(FP8_HOPPER.pin_index("g0.Q[14]"), Ok(15));
        for bad in ["g0.Q[27].lo", "g1.G", "g0.Q[0]", "a[0].s", "g0.Q[0].mid", "anc.G", "g0", "gx.G"] {
            assert!(BF16_HOPPER.pin_index(bad).is_err(), "{bad}");
        }
        for bad in ["g0.Q[0].lo", "g2.G", "g0.Q[26]"] {
            assert!(BF16_AMPERE.pin_index(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn names_are_distinct_and_digests_unique() {
        for (i, a) in RELATIONS.iter().enumerate() {
            for b in RELATIONS.iter().skip(i + 1) {
                assert_ne!(a.name, b.name);
                assert_ne!(a.sys_id, b.sys_id);
                assert_ne!(a.table_digest, b.table_digest);
            }
            assert_eq!(a.sys_id.len(), 64);
            assert_eq!(a.table_digest.len(), 64);
            assert_eq!(for_statement(Some(a.name)).unwrap().name, a.name);
            let (r0, h0) = by_digests(a.sys_id, a.table_digest).unwrap();
            assert!(r0.name == a.name && !h0);
            if !a.hashed_sys_id.is_empty() {
                assert_eq!(a.hashed_sys_id.len(), 64);
                assert_eq!(a.hashed_table_digest.len(), 64);
                assert_ne!(a.hashed_sys_id, a.sys_id);
                let (r1, h1) = by_digests(a.hashed_sys_id, a.hashed_table_digest).unwrap();
                assert!(r1.name == a.name && h1);
            }
        }
        assert!(FP8_ADA.hashed_sys_id.len() == 64 && BF16_HOPPER.hashed_sys_id.len() == 64 && FP8_HOPPER.hashed_sys_id.len() == 64);
        assert!(BF16_AMPERE.hashed_sys_id.is_empty() && FP4_NVF4.hashed_sys_id.is_empty());
        assert_eq!(for_statement(None).unwrap().name, "bf16-ampere");
        assert_eq!(for_statement(Some("bf16")).unwrap().name, "bf16-ampere");
        assert_eq!(for_statement(Some("fp4-nvf4")).unwrap().name, "fp4-nvf4");
        assert!(for_statement(Some("fp4-blackwell")).is_err());
        assert!(for_statement(Some("fp4-mxf4")).is_err());
        // the component ends are fields of a 32-bit word, disjoint and covering it (the FP32 accumulator)
        let ends = FP4_NVF4.end_components.unwrap();
        let mut covered = 0u32;
        for &(lo, bits) in ends {
            let mask = (((1u64 << bits) - 1) as u32) << lo;
            assert_eq!(covered & mask, 0);
            covered |= mask;
        }
        assert_eq!(covered, u32::MAX);
        assert_eq!(FP4_NVF4.y_bits, 32);
    }

    #[test]
    fn privsel_unit_pins_match_public_vectors_v3() {
        // reference values: privsel.relation.public_vectors_v3 on the same words (lane relmin-private).  The pins are
        // the operand words themselves: one word per pin for BF16, 3 FP8 words per pin (little-endian 8-bit fields).
        use privsel::{BF16_AMPERE, BF16_HOPPER, FP8_ADA, FP8_HOPPER};
        assert_eq!((BF16_HOPPER.k, BF16_AMPERE.k, FP8_HOPPER.k, FP8_ADA.k), (16, 16, 32, 32));
        assert_eq!((BF16_HOPPER.pins(), BF16_AMPERE.pins(), FP8_HOPPER.pins(), FP8_ADA.pins()), (32, 32, 22, 22));
        // bf16: a[i] -> pin i, b[i] -> pin 16 + i, any finite word (including -0, subnormals, the largest finite)
        let a: Vec<u32> = (0..16).map(|i| [0x3F80u32, 0xC000, 0x0000, 0x8000, 0x0001, 0x7F7F][i % 6]).collect();
        let b: Vec<u32> = (0..16).map(|i| 0x3F80 + i as u32).collect();
        let v = BF16_HOPPER.unit_pins(&a, &b).unwrap();
        assert_eq!(v.len(), 32);
        assert_eq!(v[..16], a.iter().map(|&x| x as i64).collect::<Vec<_>>()[..]);
        assert_eq!(v[16..], b.iter().map(|&x| x as i64).collect::<Vec<_>>()[..]);
        assert_eq!(BF16_HOPPER.pin_index("a[0]").unwrap(), 0);
        assert_eq!(BF16_HOPPER.pin_index("a[15]").unwrap(), 15);
        assert_eq!(BF16_HOPPER.pin_index("b[0]").unwrap(), 16);
        assert_eq!(BF16_HOPPER.pin_index("b[15]").unwrap(), 31);
        for bad in ["a[16]", "b[0:3]", "c[0]", "a[-1]", "a_words[0]", "a[0"] {
            assert!(BF16_HOPPER.pin_index(bad).is_err(), "{bad}");
        }
        // non-finite words are refused (inf, nan, either sign), and words outside 16 bits
        for bad in [0x7F80u32, 0xFF80, 0x7FC0, 0xFFFF, 0x7F81, 0x10000] {
            let mut a2 = a.clone();
            a2[5] = bad;
            assert!(BF16_HOPPER.unit_pins(&a2, &b).is_err(), "{bad:#x}");
            let mut b2 = b.clone();
            b2[0] = bad;
            assert!(BF16_HOPPER.unit_pins(&a, &b2).is_err(), "{bad:#x}");
        }
        assert!(BF16_HOPPER.unit_pins(&a[..15], &b).is_err(), "short unit");
        // fp8 e4m3: 32 words -> 11 pins per operand, runs [0:3], [3:6], ..., [30:32]; word i sits in byte (i % 3)
        let a: Vec<u32> = (0..32).map(|i| (i as u32 * 7 + 1) & 0x7E).collect(); // finite (exponent < 15 or mantissa != 7)
        let b: Vec<u32> = (0..32).map(|i| 0x80 | ((i as u32 * 5) & 0x7E)).collect();
        let v = FP8_ADA.unit_pins(&a, &b).unwrap();
        assert_eq!(v.len(), 22);
        for r in 0..11 {
            let lo = 3 * r;
            let hi = (lo + 3).min(32);
            let pack = |w: &[u32]| (lo..hi).fold(0i64, |acc, i| acc + ((w[i] as i64) << (8 * (i - lo))));
            assert_eq!(v[r], pack(&a), "a run {r}");
            assert_eq!(v[11 + r], pack(&b), "b run {r}");
            assert_eq!(FP8_ADA.pin_index(&format!("a[{lo}:{hi}]")).unwrap(), r);
            assert_eq!(FP8_ADA.pin_index(&format!("b[{lo}:{hi}]")).unwrap(), 11 + r);
        }
        assert_eq!(v[10], (a[30] as i64) + ((a[31] as i64) << 8), "last run holds 2 words");
        for bad in ["a[0]", "a[1:4]", "a[30:33]", "a[33:36]", "b[0:2]"] {
            assert!(FP8_ADA.pin_index(bad).is_err(), "{bad}");
        }
        // e4m3 NaN (0x7F / 0xFF) and 9-bit words refused; e4m3 0x7E (448.0) and 0xFE accepted (no infinities)
        for bad in [0x7Fu32, 0xFF, 0x100] {
            let mut a2 = a.clone();
            a2[31] = bad;
            assert!(FP8_ADA.unit_pins(&a2, &b).is_err(), "{bad:#x}");
        }
        let mut a2 = a.clone();
        a2[0] = 0x7E;
        a2[1] = 0xFE;
        assert!(FP8_ADA.unit_pins(&a2, &b).is_ok());
        assert_eq!(FP8_ADA.unit_pins(&a2, &b).unwrap()[0], 0x7E + (0xFE << 8) + ((a[2] as i64) << 16));
        // e4m3 is the same format on hopper
        assert_eq!(FP8_HOPPER.unit_pins(&a, &b).unwrap(), v);
    }
}
