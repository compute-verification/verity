//! Statement binding for the chain AIR (PROTOCOL.md "Statement binding"; red-team findings F1-F3 of
//! `backends/numerical/reports/redteam_vu.md`).
//!
//! A B proof is a proof *of a statement*: `B` VUs of `K = 16 * STEPS` products each, with claimed output words
//! `y16[0..B]` (and, when the verifier holds them, the operands `x`, `w`).  Two mechanisms bind the trace to it:
//!
//! 1. **Header absorption.**  Before any challenge is drawn, prover and verifier absorb the same header into the
//!    Fiat-Shamir challenger: a domain tag, the AIR identity (a hash of the field, the column layout and the
//!    constraint program), the trace field, the challenge-extension degree, `B`, `K`, `STEPS`, `degree_bits`, the
//!    FRI parameters, and the statement (every claimed word; a hash of the operands).  The verifier builds the
//!    header from *its own* configuration and statement -- nothing in it is read from the proof.
//! 2. **Public columns checked at ζ.**  The claim column `vu.y16` (and the operand columns when bound) are
//!    committed trace columns; the verifier knows what each of them must be on every row (the statement's word
//!    of the VU that row belongs to, `0` on the zero-VU padding), i.e. a public polynomial `W(X)` of degree `< n`
//!    over the trace domain.  The verifier evaluates `W` itself at the out-of-domain point ζ (and at ζ·g) and
//!    requires the opened value of the committed column to equal it.  Both polynomials have degree `< n` and ζ is
//!    drawn after the trace commitment, so a column that differs from `W` anywhere passes with probability at most
//!    `n / |F_challenge|` (Schwartz-Zippel), the same order as the DEEP quotient term the proof already carries.
//!    The AIR's last-row constraint `vu.out: last * (epilogue - vu.y16) = 0` then ties the bound word to the
//!    epilogue of the chain, and the chain constraints tie the epilogue to the operands.
//!
//! Cost: no prover work, no proof bytes; the verifier does `O(n)` extension-field operations per bound column.

use p3_field::{batch_multiplicative_inverse, ExtensionField, PrimeField64, TwoAdicField};
use sha2::{Digest, Sha256};

use crate::air::Layout;
use crate::ctx::FieldKind;
use crate::program::{COL_CLAIM, STEPS};
use crate::stark::FriParams;

/// The transcript domain tag (versioned: any change to the header layout bumps it).
pub const DOMAIN_TAG: &[u8] = b"verity-direct/chain-air/statement/v1";
/// The trace has at least this many rows (uni-stark needs a two-row domain).
pub const MIN_ROWS: usize = 2;

/// The statement a chain proof is bound to.
#[derive(Clone, Debug)]
pub struct Statement {
    /// Claimed output word of every VU, in VU order.  `B = words.len()`.
    pub words: Vec<u64>,
    /// The operands, when the verifier holds them: `x[vu * K + j]`, `w[vu * K + j]` (bf16 words).  Binding them
    /// makes the operand columns public too (32 more columns checked at ζ; `authentication = excluded` numbers
    /// leave this `None` and the operand columns are witness).
    pub operands: Option<(Vec<u64>, Vec<u64>)>,
    /// The trace has at least this many rows before padding to a power of two: `MIN_ROWS` for the limb
    /// realisation; the preprocessed tables' `total_rows` for the LogUp realisation (PROTOCOL.md 14.3: the tables
    /// live in the same domain as the trace).  Part of the header (through `rows` / `degree_bits`).
    pub min_rows: usize,
}

impl Statement {
    pub fn new(words: Vec<u64>) -> Self {
        Self { words, operands: None, min_rows: MIN_ROWS }
    }
    pub fn with_operands(words: Vec<u64>, x: Vec<u64>, w: Vec<u64>) -> Self {
        let k = 16 * STEPS as usize;
        assert_eq!(x.len(), words.len() * k, "x has K words per VU");
        assert_eq!(w.len(), words.len() * k, "w has K words per VU");
        Self { words, operands: Some((x, w)), min_rows: MIN_ROWS }
    }
    /// The same statement for a realisation whose trace is at least `min_rows` tall (the LogUp tables).
    pub fn with_min_rows(mut self, min_rows: usize) -> Self {
        self.min_rows = min_rows.max(MIN_ROWS);
        self
    }
    pub fn b(&self) -> usize {
        self.words.len()
    }
    pub const fn k() -> usize {
        16 * STEPS as usize
    }
    /// The trace height the verifier expects: `B * STEPS` rows padded to a power of two (at least `MIN_ROWS`).
    pub fn rows(&self) -> usize {
        (self.b() * STEPS as usize).max(self.min_rows).next_power_of_two()
    }
    pub fn degree_bits(&self) -> usize {
        self.rows().trailing_zeros() as usize
    }
    /// The value of a public column on every row of the trace: `per_vu[vu]` on the VU's rows, `0` on padding.
    pub fn expand<G: PrimeField64>(&self, per_row: impl Fn(usize, usize) -> u64) -> Vec<G> {
        let steps = STEPS as usize;
        let n = self.rows();
        let real = self.b() * steps;
        (0..n).map(|r| if r < real { G::from_canonical_u64(per_row(r / steps, r % steps)) } else { G::zero() }).collect()
    }
    /// The bound columns: `(column index, column name, public values on every row)`.
    pub fn public_columns<G: PrimeField64>(&self, lay: &Layout) -> Vec<(usize, String, Vec<G>)> {
        let mut out = vec![(lay.col(COL_CLAIM), COL_CLAIM.to_string(), self.expand(|vu, _| self.words[vu]))];
        if let Some((x, w)) = &self.operands {
            let k = Self::k();
            for j in 0..16 {
                let name = format!("x.a[{j}]");
                out.push((lay.col(&name), name, self.expand(|vu, s| x[vu * k + 16 * s + j])));
                let name = format!("x.b[{j}]");
                out.push((lay.col(&name), name, self.expand(|vu, s| w[vu * k + 16 * s + j])));
            }
        }
        out
    }
}

/// The AIR identity: SHA-256 over the field, the width and every column and constraint name in layout order.
/// Two AIRs with different constraint programs, layouts or fields have different identities.
pub fn air_id(kind: FieldKind, lay: &Layout) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"verity-direct/chain-air/id/v1\0");
    h.update(kind.name().as_bytes());
    h.update(b"\0");
    h.update((lay.width as u64).to_le_bytes());
    h.update((lay.col_names.len() as u64).to_le_bytes());
    for n in &lay.col_names {
        h.update((n.len() as u64).to_le_bytes());
        h.update(n.as_bytes());
    }
    h.update((lay.constraint_names.len() as u64).to_le_bytes());
    for n in &lay.constraint_names {
        h.update((n.len() as u64).to_le_bytes());
        h.update(n.as_bytes());
    }
    h.finalize().into()
}

fn sha256_words(words: &[u64]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update((words.len() as u64).to_le_bytes());
    for w in words {
        h.update(w.to_le_bytes());
    }
    h.finalize().into()
}

/// The header both parties absorb before the first challenge, as field elements (bytes of tags and hashes one
/// element each; integers split into 32-bit halves so they are canonical in either field; the word list is
/// length-prefixed).
pub fn header<G: PrimeField64>(kind: FieldKind, air_id: &[u8; 32], lay_width: usize, challenge_ext_degree: usize, fri: FriParams, stmt: &Statement) -> Vec<G> {
    let mut out: Vec<G> = Vec::with_capacity(96 + 2 * stmt.b());
    let push_bytes = |out: &mut Vec<G>, b: &[u8]| {
        out.push(G::from_canonical_u64(b.len() as u64));
        out.extend(b.iter().map(|&x| G::from_canonical_u8(x)));
    };
    let push_u64 = |out: &mut Vec<G>, v: u64| {
        out.push(G::from_canonical_u32(v as u32));
        out.push(G::from_canonical_u32((v >> 32) as u32));
    };
    push_bytes(&mut out, DOMAIN_TAG);
    push_bytes(&mut out, air_id);
    let p = kind.prime();
    push_u64(&mut out, p as u64);
    push_u64(&mut out, (p >> 64) as u64);
    push_u64(&mut out, lay_width as u64);
    push_u64(&mut out, challenge_ext_degree as u64);
    push_u64(&mut out, stmt.b() as u64);
    push_u64(&mut out, Statement::k() as u64);
    push_u64(&mut out, STEPS);
    push_u64(&mut out, stmt.rows() as u64);
    push_u64(&mut out, stmt.degree_bits() as u64);
    push_u64(&mut out, fri.log_blowup as u64);
    push_u64(&mut out, fri.num_queries as u64);
    push_u64(&mut out, fri.proof_of_work_bits as u64);
    push_u64(&mut out, stmt.words.len() as u64);
    for &w in &stmt.words {
        push_u64(&mut out, w);
    }
    match &stmt.operands {
        None => push_u64(&mut out, 0),
        Some((x, w)) => {
            push_u64(&mut out, 1);
            push_bytes(&mut out, &sha256_words(x));
            push_bytes(&mut out, &sha256_words(w));
        }
    }
    out
}

/// The LogUp realisation's addition to the header (PROTOCOL.md 14.6), absorbed right after `header`: a tag, the
/// preprocessed and auxiliary widths, the challenge-extension degree the aux columns are laid out for, and the
/// digest of the table set the verifier holds -- the prover cannot substitute a table without changing the
/// transcript (the verifier also recomputes the preprocessed commitment and compares it).
pub const LOOKUP_DOMAIN_TAG: &[u8] = b"verity-direct/chain-air/logup/v1";

pub fn lookup_header<G: PrimeField64>(pre_width: usize, aux_width: usize, ext_degree: usize, table_digest: &[u8; 32]) -> Vec<G> {
    let mut out: Vec<G> = Vec::with_capacity(48);
    out.push(G::from_canonical_u64(LOOKUP_DOMAIN_TAG.len() as u64));
    out.extend(LOOKUP_DOMAIN_TAG.iter().map(|&x| G::from_canonical_u8(x)));
    for v in [pre_width as u64, aux_width as u64, ext_degree as u64] {
        out.push(G::from_canonical_u32(v as u32));
        out.push(G::from_canonical_u32((v >> 32) as u32));
    }
    out.push(G::from_canonical_u64(32));
    out.extend(table_digest.iter().map(|&x| G::from_canonical_u8(x)));
    out
}

/// SHA-256 of the table set: every table's name, tag, offset, row count, lowest key and every table row's tuple
/// (`lookup::TableSet::row`), then `total_rows`.  Two table sets that differ in any entry have different digests.
pub fn table_digest(tables: &crate::lookup::TableSet) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"verity-direct/logup/tables/v1\0");
    h.update((tables.tables.len() as u64).to_le_bytes());
    for t in &tables.tables {
        h.update((t.name.len() as u64).to_le_bytes());
        h.update(t.name.as_bytes());
        h.update(t.tag.to_le_bytes());
        h.update((t.offset as u64).to_le_bytes());
        h.update((t.rows as u64).to_le_bytes());
        h.update(t.key_lo.to_le_bytes());
    }
    h.update((tables.total_rows as u64).to_le_bytes());
    for r in 0..tables.total_rows {
        let (vals, tag) = tables.row(r);
        for v in vals {
            h.update(v.to_le_bytes());
        }
        h.update(tag.to_le_bytes());
    }
    h.finalize().into()
}

/// `(W(ζ), W(ζ·g))` for the polynomial `W` of degree `< n` that takes `values[i]` at `g^i` (the natural trace
/// domain, shift 1, `g = two_adic_generator(log n)`), by barycentric evaluation:
/// `W(ζ) = (ζ^n - 1)/n · Σ_i values[i] · g^i / (ζ - g^i)`.  `W(ζ g)` reuses the same inverse denominators
/// (`ζ g - g^i = g (ζ - g^{i-1})`): it is the evaluation of the cyclically shifted vector.
pub fn eval_pair<G: TwoAdicField + PrimeField64, EF: ExtensionField<G>>(values: &[G], zeta: EF) -> (EF, EF) {
    let n = values.len();
    assert!(n.is_power_of_two() && n >= 2);
    let log_n = n.trailing_zeros() as usize;
    let g = G::two_adic_generator(log_n);
    let pows: Vec<G> = g.powers().take(n).collect();
    let denoms: Vec<EF> = pows.iter().map(|&gi| zeta - EF::from_base(gi)).collect();
    let inv = batch_multiplicative_inverse(&denoms);
    // Σ_i v_i g^i / (ζ - g^i)  and  Σ_i v_{i+1} g^i / (ζ - g^i)
    let mut s0 = EF::zero();
    let mut s1 = EF::zero();
    for i in 0..n {
        let term = inv[i] * pows[i];
        if values[i] != G::zero() {
            s0 += term * values[i];
        }
        let v_next = values[(i + 1) % n];
        if v_next != G::zero() {
            s1 += term * v_next;
        }
    }
    let n_inv = G::from_canonical_usize(n).inverse();
    let zn = zeta.exp_power_of_2(log_n);
    let scale = (zn - EF::one()) * n_inv;
    (s0 * scale, s1 * scale)
}

/// `Statement::rows` for a bare VU count.
pub fn rows_for(b: usize) -> usize {
    (b * STEPS as usize).max(MIN_ROWS).next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::AbstractField;
    use p3_goldilocks::Goldilocks;

    fn check<G: TwoAdicField + PrimeField64, EF: ExtensionField<G>>() {
        // W with values v_i at g^i: evaluate at an in-domain point through the barycentric formula's limit is
        // not defined, so compare against direct Lagrange evaluation at a random-ish extension point.
        let n = 16usize;
        let vals: Vec<G> = (0..n).map(|i| G::from_canonical_u64((i * i + 3) as u64 % 7)).collect();
        let zeta = EF::from_base_fn(|k| G::from_canonical_u64(0x1234_5678 + 97 * k as u64));
        let g = G::two_adic_generator(4);
        // direct: Σ_i v_i Π_{j≠i} (ζ - g^j)/(g^i - g^j)
        let direct = |z: EF| {
            let mut acc = EF::zero();
            for i in 0..n {
                let mut li = EF::one();
                for j in 0..n {
                    if j != i {
                        let num = z - EF::from_base(g.exp_u64(j as u64));
                        let den = g.exp_u64(i as u64) - g.exp_u64(j as u64);
                        li *= num * den.inverse();
                    }
                }
                acc += li * vals[i];
            }
            acc
        };
        let (a, b) = eval_pair::<G, EF>(&vals, zeta);
        assert_eq!(a, direct(zeta));
        assert_eq!(b, direct(zeta * EF::from_base(g)));
    }

    #[test]
    fn barycentric_matches_lagrange_goldilocks() {
        check::<Goldilocks, BinomialExtensionField<Goldilocks, 2>>();
    }
    #[test]
    fn barycentric_matches_lagrange_babybear() {
        check::<BabyBear, BinomialExtensionField<BabyBear, 4>>();
    }

    #[test]
    fn header_changes_with_every_statement_field() {
        let (_, lay) = crate::air::ChainAir::new(FieldKind::Goldilocks);
        let id = air_id(FieldKind::Goldilocks, &lay);
        let fri = crate::stark::DEFAULT_FRI;
        let s = Statement::new(vec![1, 2]);
        let h0 = header::<Goldilocks>(FieldKind::Goldilocks, &id, lay.width, 2, fri, &s);
        assert_ne!(h0, header::<Goldilocks>(FieldKind::Goldilocks, &id, lay.width, 2, fri, &Statement::new(vec![1, 3])));
        assert_ne!(h0, header::<Goldilocks>(FieldKind::Goldilocks, &id, lay.width, 2, fri, &Statement::new(vec![1, 2, 0])));
        assert_ne!(h0, header::<Goldilocks>(FieldKind::Goldilocks, &[0u8; 32], lay.width, 2, fri, &s));
        let mut fri2 = fri;
        fri2.num_queries += 1;
        assert_ne!(h0, header::<Goldilocks>(FieldKind::Goldilocks, &id, lay.width, 2, fri2, &s));
        assert_ne!(h0, header::<Goldilocks>(FieldKind::Goldilocks, &id, lay.width, 3, fri, &s));
        // the BabyBear layout has a different identity
        let (_, lay_b) = crate::air::ChainAir::new(FieldKind::BabyBear);
        assert_ne!(id, air_id(FieldKind::BabyBear, &lay_b));
    }

    #[test]
    fn statement_rows_and_public_column() {
        let s = Statement::new(vec![0xbed1, 0x3fab]);
        assert_eq!(s.rows(), 256);
        assert_eq!(s.degree_bits(), 8);
        assert_eq!(rows_for(4096), 1 << 19);
        let (_, lay) = crate::air::ChainAir::new(FieldKind::Goldilocks);
        let cols = s.public_columns::<Goldilocks>(&lay);
        assert_eq!(cols.len(), 1);
        let w = &cols[0].2;
        assert_eq!(w.len(), 256);
        assert_eq!(w[0], Goldilocks::from_canonical_u64(0xbed1));
        assert_eq!(w[95], Goldilocks::from_canonical_u64(0xbed1));
        assert_eq!(w[96], Goldilocks::from_canonical_u64(0x3fab));
        assert_eq!(w[192], Goldilocks::zero());
    }
}
