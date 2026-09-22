//! The frozen target's widths, mirroring `verity_numerical.checker.params.REAL` formula for formula.
//! Every constant here is a *bound* used by a contract; `check_params()` asserts the derived values the
//! Python reference prints for `REAL` (see `tests/params.rs`).

pub const SIG: i128 = 8;
pub const EXP: i128 = 8;
pub const ACC_SIG: i128 = 24;
pub const WIDTH: i128 = 25;
pub const FLOOR: i128 = -132;
pub const GROUPS: [usize; 2] = [8, 8];
pub const K: usize = 16;

pub const BIAS: i128 = (1 << (EXP - 1)) - 1;
pub const E_MIN: i128 = 1 - BIAS;
pub const E_MAX: i128 = BIAS;
pub const T_FINITE: i128 = (1 << EXP) - 1;
pub const FRAC: i128 = SIG - 1;
pub const ACC_FRAC: i128 = ACC_SIG - 1;
pub const WORD_BITS: i128 = 1 + EXP + FRAC;
pub const ACC_WORD_BITS: i128 = 1 + EXP + ACC_FRAC;
pub const PROD_SHIFT: i128 = WIDTH - 2 * SIG + 1;
pub const PROD_BITS: i128 = 2 * SIG + PROD_SHIFT;
pub const ACC_SHIFT: i128 = WIDTH - ACC_SIG;
pub const ACC_BITS: i128 = ACC_SIG + ACC_SHIFT;
pub const EP_LO: i128 = 2 * E_MIN;
pub const EP_HI: i128 = 2 * E_MAX;
pub const EXP_LO: i128 = if FLOOR < EP_LO { FLOOR } else { EP_LO };
pub const N: i128 = 8;
pub const S_MAX: i128 = ((1 << ACC_BITS) - (1 << ACC_SHIFT)) + N * ((1 << PROD_BITS) - (1 << PROD_SHIFT));
pub const S_BITS: i128 = 128 - (S_MAX as u128).leading_zeros() as i128;
pub const L_HI: i128 = S_BITS;
pub const GROWTH: i128 = L_HI - WIDTH;
const LAST_G: i128 = GROUPS.len() as i128 - 1;
pub const fn e_hi(g: i128) -> i128 {
    (if E_MAX > EP_HI { E_MAX } else { EP_HI }) + g * GROWTH
}
pub const LAST: i128 = e_hi(LAST_G);
pub const EXP_HI: i128 = LAST + GROWTH;
pub const SHIFT_ROWS: i128 = LAST - EXP_LO + 1;
const fn max(a: i128, b: i128) -> i128 {
    if a > b { a } else { b }
}
pub const CMP_ROWS: i128 = {
    let s1 = E_MIN - FLOOR + WIDTH;
    let s2 = LAST + L_HI - WIDTH - E_MIN;
    let s3 = EXP_HI - E_MAX;
    let s4 = E_MAX - E_MIN;
    max(SHIFT_ROWS, max(max(s1, s2), max(s3, s4)) + 1)
};
pub const K_LO: i128 = {
    let a = ACC_SIG - L_HI;
    let b = FLOOR - E_MIN + ACC_SIG - WIDTH;
    if a < b { a } else { b }
};
pub const K_HI: i128 = ACC_SIG;
pub const NORM_REM_BITS: i128 = -K_LO;

/// The values `verity_numerical.checker.params.REAL` reports (checked by a unit test).
pub fn check_params() {
    assert_eq!(BIAS, 127);
    assert_eq!(E_MIN, -126);
    assert_eq!(PROD_SHIFT, 10);
    assert_eq!(PROD_BITS, 26);
    assert_eq!(ACC_BITS, 25);
    assert_eq!(S_BITS, 30);
    assert_eq!(LAST, 259);
    assert_eq!(EXP_HI, 264);
    assert_eq!(SHIFT_ROWS, 512);
    assert_eq!(CMP_ROWS, 512);
    assert_eq!(K_LO, -7);
    assert_eq!(NORM_REM_BITS, 7);
    assert_eq!(EXP_LO, -252);
}
