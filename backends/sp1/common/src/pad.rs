//! The flag/mux gate semantics of the padded serving circuit (`veritor.pad-serve@1`,
//! lane PAD-IMPL, GUIDANCE §29), bit for bit as `veritor/core/pad_serve.py` declares
//! them and `lean/VeritorSpec/PadServe.lean` specifies them.
//!
//! The padded circuit (`Serve@3`, `docs/vllm-poc/serve3-spec.md`) carries one
//! live/finished bit per decode step and selects, at decode-step granularity,
//! between the step's computed outputs and the canonical pad values.  These
//! functions are total on their declared widths: the width checks live in
//! `GateSet::evaluate` (exactly where every other family's live), so every
//! function here assumes in-range words and is a plain word computation.
//!
//! * [`select`]: `c ? a : b` on raw words — representation-transparent (BF16,
//!   FP32 and I32 selection are the same copy; the gate set declares three
//!   typed gates so that argument widths are checked per encoding).
//! * [`eq_i32`]: `1` iff the two 32-bit words are equal (the stop test
//!   `token == EOS_ID` of the finished chain).
//! * [`bit_and`], [`bit_not`]: boolean algebra on `Value<1>` words; the
//!   finished chain `live' = bit_and(live, bit_not(eq_i32(tok, EOS)))` is
//!   absorbing because `bit_and(0, x) = 0` for every `x`.

/// `c ? a : b` on words; `c` is a `Value<1>` word (the width check upstream
/// guarantees `c <= 1`; the low bit decides regardless, as in the Python).
#[cfg_attr(feature = "attribution", inline(never))]
pub fn select(c: u64, a: u64, b: u64) -> u64 {
    if c & 1 == 1 {
        a
    } else {
        b
    }
}

/// `1` iff two 32-bit words are equal (`veritor_ir` `I32Eq_v1`; two's-complement
/// interpretation never matters for equality of words).
#[cfg_attr(feature = "attribution", inline(never))]
pub fn eq_i32(a: u64, b: u64) -> u64 {
    (a == b) as u64
}

/// Conjunction of two `Value<1>` words (`BitAnd_v1`).
#[cfg_attr(feature = "attribution", inline(never))]
pub fn bit_and(a: u64, b: u64) -> u64 {
    a & b & 1
}

/// Negation of one `Value<1>` word (`BitNot_v1`).
#[cfg_attr(feature = "attribution", inline(never))]
pub fn bit_not(a: u64) -> u64 {
    (a & 1) ^ 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_copies_the_named_operand() {
        assert_eq!(select(1, 0x3F80, 0x7FFF), 0x3F80);
        assert_eq!(select(0, 0x3F80, 0x7FFF), 0x7FFF);
        assert_eq!(select(1, 0xFFFF_FFFF, 0), 0xFFFF_FFFF);
        assert_eq!(select(0, 0xFFFF_FFFF, 0), 0);
    }

    #[test]
    fn eq_i32_is_word_equality() {
        assert_eq!(eq_i32(5, 5), 1);
        assert_eq!(eq_i32(5, 6), 0);
        assert_eq!(eq_i32(0xFFFF_FFFF, 0xFFFF_FFFF), 1);
        assert_eq!(eq_i32(0, 0x8000_0000), 0);
    }

    #[test]
    fn boolean_algebra_truth_tables_and_absorption() {
        assert_eq!(bit_and(0, 0), 0);
        assert_eq!(bit_and(0, 1), 0);
        assert_eq!(bit_and(1, 0), 0);
        assert_eq!(bit_and(1, 1), 1);
        assert_eq!(bit_not(0), 1);
        assert_eq!(bit_not(1), 0);
        // the finished chain is absorbing: once live falls, it never rises
        for tok_eq_eos in [0u64, 1] {
            assert_eq!(bit_and(0, bit_not(tok_eq_eos)), 0);
        }
    }
}
