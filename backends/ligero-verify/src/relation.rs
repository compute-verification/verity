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
    pub operand: Operand,
    /// The pinned `protocol.system_id` (the last 32 bytes of `system.bin`) and this verifier's own table digest.
    pub sys_id: &'static str,
    pub table_digest: &'static str,
}

pub static BF16_AMPERE: Relation = Relation {
    name: "bf16-ampere",
    aliases: &["bf16"],
    tag: b"",
    word_bytes: 2,
    y_bytes: 2,
    y_bits: 16,
    operand: Operand::Bf16,
    sys_id: "44cb05b9e6988219beba2d66582124e67a1893aac39d0b66126ce0b9d4db89cb",
    table_digest: "c147cc4cece7d61cd3046828410d18f487dcde82f709044f4b4f97e282a73a95",
};

pub static FP8_ADA: Relation = Relation {
    name: "fp8-ada",
    aliases: &[],
    tag: b"|rel=fp8-ada|v1",
    word_bytes: 1,
    y_bytes: 4,
    y_bits: 22,
    operand: Operand::E4m3,
    sys_id: "6b570eef0c7faa07ccd1dddb2508e2699bd5367b108b15920469d46d16073d37",
    table_digest: "443e4fc74c0ac0cc864e50f7125d9585d71986c0b95ff5336d93a6939f3cb952",
};

pub static BF16_HOPPER: Relation = Relation {
    name: "bf16-hopper",
    aliases: &[],
    tag: b"",
    word_bytes: 2,
    y_bytes: 2,
    y_bits: 16,
    operand: Operand::Bf16,
    sys_id: "9dcc7cdc1377a00d877440cc01a0dcdb73b3e989d86df8244eccb35a81be0df4",
    table_digest: "74ce68a2801e14fbeaab2b241886bbe3fa9146234289e52c774fd4160a6c469c",
};

pub static FP8_HOPPER: Relation = Relation {
    name: "fp8-hopper",
    aliases: &[],
    tag: b"|rel=fp8-hopper|v1",
    word_bytes: 1,
    y_bytes: 4,
    y_bits: 22,
    operand: Operand::E4m3,
    sys_id: "c7cbebe365cb5f773a679bc13b5382fb45714511b1912e4e4ab7fb48accd4bc3",
    table_digest: "d5b77463fee9b3d777174755144edaa84461b3eec6ac6a1750c887dc861d58e7",
};

pub static RELATIONS: [&Relation; 4] = [&BF16_AMPERE, &FP8_ADA, &BF16_HOPPER, &FP8_HOPPER];

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

/// The relation whose pinned digests a system file carries, if any.
pub fn by_digests(sys_id_hex: &str, table_digest_hex: &str) -> Option<&'static Relation> {
    RELATIONS.iter().copied().find(|r| r.sys_id == sys_id_hex && r.table_digest == table_digest_hex)
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
            assert_eq!(by_digests(a.sys_id, a.table_digest).unwrap().name, a.name);
        }
        assert_eq!(for_statement(None).unwrap().name, "bf16-ampere");
        assert_eq!(for_statement(Some("bf16")).unwrap().name, "bf16-ampere");
        assert!(for_statement(Some("fp4-blackwell")).is_err());
    }
}
