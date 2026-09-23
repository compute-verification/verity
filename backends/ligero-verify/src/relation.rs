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

/// How a relation's public pins are computed from the statement's operand words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decode {
    /// One word per operand, one pin per word component: `a[i].s|m|e` / `b[i].s|m|e` (`unit.public_vectors`,
    /// `public_vectors_fp8`); K = 1 + the largest pinned index.
    Word(Operand),
    /// The NVFP4 unit: K = 68 words per operand (64 nibbles + 4 scale bytes), pins per `nvf4::unit_pins`.
    Nvf4,
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

pub static RELATIONS: [&Relation; 5] = [&BF16_AMPERE, &FP8_ADA, &BF16_HOPPER, &FP8_HOPPER, &FP4_NVF4];

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
}
