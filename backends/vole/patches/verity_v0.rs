//! The `verity_v0` plugin (Verity campaign, track C, lane c-compress; swanky `1b05bdda` + this patch).
//!
//! Three operations on one prime field, none with output wires:
//!
//! * `assert_mul` -- inputs flattened as `(a, b, c)*`: checks `a * b = c` as one QuickSilver degree-2 triple with
//!   **no committed output** (the IR's `@mul` always commits its product; the protocol never needed it).  A boolean
//!   check is `assert_mul(b, b, b)`, a product identity `q * pow + r = M 2^s` is `assert_mul(q, pow, M 2^s - r)`.
//! * `range, W` -- every input wire is in `[0, 2^W)`, proved with a logUp-style lookup argument (Haböck 2022,
//!   ePrint 2022/1530, Lemma 5): the prover commits the multiplicities `m_j` of the table entries, the verifier sends
//!   `r`, the prover commits `inv_i = 1 / (r - x_i)` for every query (one `assert_mul(r - x_i, inv_i, 1)` each) and
//!   `g_j = m_j / (r - j)` for every table row (linear check `g_j (r - j) = m_j`), and `sum_i inv_i = sum_j g_j` is
//!   asserted.  Soundness: the identity `sum_i 1/(X - x_i) = sum_j m_j/(X - j)` holds as rational functions iff every
//!   `x_i` is a table entry (`p > N` so multiplicities never vanish mod p); a false identity survives the random `r`
//!   with probability `<= (N + R) / p` (Schwartz-Zippel on the numerator).  Cost: `N + 2R` committed elements.
//! * `lookup, C, R, t_11 .. t_RC` -- inputs flattened as `C`-tuples; each tuple is a row of the `R x C` public table.
//!   The verifier sends `s`, tuples and rows are combined as `sum_c s^c t_c`, then the same argument as `range`.
//!   Extra soundness term: a tuple that is not a row collides with some row's combination with probability
//!   `<= R (C - 1) / p` per query.  Cost: `N + 2R` committed elements.
//!
//! Everything runs over the existing `BackendT`: `random()` (the verifier's fresh challenge, sent only after the
//! prover's pending commitments are flushed and read), `input_private` (the prover supplies the value: multiplicities,
//! inverses and `g_j` are computed here, not read from the witness stream), `assert_mul` (added to the trait by this
//! patch), `mul_constant`, `add`, `add_constant`, `sub`, `assert_zero`.  The plaintext backend gets the default
//! `assert_mul` (a multiplication and a zero check).

use super::{Plugin, PluginExecution};
use crate::backend_trait::BackendT;
use crate::circuit_ir::{FunStore, TypeId, TypeSpecification, TypeStore, WireCount};
use crate::fields::SieveIrDeserialize;
use crate::number_to_u64;
use log::warn;
use std::collections::HashMap;
use swanky_error::{ErrorKind, Result, bail, ensure};
use swanky_field::{FiniteField, FiniteRing};
use swanky_sieve_ir_parser::{Number, PluginTypeArg};

#[derive(Clone, Debug)]
pub(crate) enum VerityOp {
    AssertMul,
    Range { bits: u32 },
    Lookup { ncols: usize, table: Vec<Number> },
}

#[derive(Clone, Debug)]
pub(crate) struct VerityV0 {
    type_id: TypeId,
    op: VerityOp,
}

impl VerityV0 {
    pub(crate) fn type_id(&self) -> TypeId {
        self.type_id
    }

    pub(crate) fn execute<B: BackendT>(&self, wires: Vec<B::Wire>, backend: &mut B) -> Result<()>
    where
        B::FieldElement: SieveIrDeserialize,
    {
        match &self.op {
            VerityOp::AssertMul => {
                ensure!(
                    wires.len() % 3 == 0,
                    ErrorKind::OtherError,
                    "verity_v0 assert_mul: {} wires, not a multiple of 3",
                    wires.len()
                );
                for t in wires.chunks(3) {
                    backend.assert_mul(&t[0], &t[1], &t[2])?;
                }
                Ok(())
            }
            VerityOp::Range { bits } => {
                let rows = (0..(1u64 << bits))
                    .map(|i| -> Result<Vec<B::FieldElement>> { Ok(vec![fe_from_u64::<B::FieldElement>(i)?]) })
                    .collect::<Result<Vec<_>>>()?;
                logup(backend, &wires, 1, &rows)
            }
            VerityOp::Lookup { ncols, table } => {
                let rows = table
                    .chunks(*ncols)
                    .map(|r| r.iter().map(B::FieldElement::from_number).collect::<Result<Vec<_>>>())
                    .collect::<Result<Vec<_>>>()?;
                logup(backend, &wires, *ncols, &rows)
            }
        }
    }
}

fn fe_bytes<F: FiniteField>(x: &F) -> Vec<u8> {
    x.to_bytes().to_vec()
}

fn fe_from_u64<F: FiniteField + SieveIrDeserialize>(x: u64) -> Result<F> {
    F::from_number(&Number::from_u64(x))
}

/// Inverses of every element of `xs` (all nonzero) with one field inversion.
fn batch_inverse<F: FiniteField>(xs: &[F]) -> Vec<F> {
    let mut prefix = Vec::with_capacity(xs.len());
    let mut acc = F::ONE;
    for x in xs {
        prefix.push(acc);
        acc *= *x;
    }
    let mut inv = if xs.is_empty() { F::ONE } else { acc.inverse() };
    let mut out = vec![F::ZERO; xs.len()];
    for i in (0..xs.len()).rev() {
        out[i] = inv * prefix[i];
        inv *= xs[i];
    }
    out
}

/// logUp membership of every `ncols`-tuple of `wires` in `rows`.
fn logup<B: BackendT>(
    backend: &mut B,
    wires: &[B::Wire],
    ncols: usize,
    rows: &[Vec<B::FieldElement>],
) -> Result<()>
where
    B::FieldElement: SieveIrDeserialize,
{
    ensure!(ncols >= 1, ErrorKind::OtherError, "verity_v0: ncols must be >= 1");
    ensure!(
        wires.len() % ncols == 0,
        ErrorKind::OtherError,
        "verity_v0 lookup: {} wires, not a multiple of {ncols}",
        wires.len()
    );
    ensure!(!rows.is_empty(), ErrorKind::OtherError, "verity_v0 lookup: empty table");
    ensure!(
        rows.iter().all(|r| r.len() == ncols),
        ErrorKind::OtherError,
        "verity_v0 lookup: ragged table"
    );
    let n = wires.len() / ncols;
    let one = backend.constant(B::FieldElement::ONE)?;
    let is_prover = backend.wire_value(&one).is_some();

    // 1. multiplicities (prover side; the verifier commits blindly)
    let mut mult = vec![0u64; rows.len()];
    if is_prover {
        let mut index: HashMap<Vec<u8>, usize> = HashMap::with_capacity(rows.len());
        for (j, row) in rows.iter().enumerate() {
            let key: Vec<u8> = row.iter().flat_map(fe_bytes).collect();
            index.insert(key, j);
        }
        let mut missing = 0usize;
        for i in 0..n {
            let key: Vec<u8> = (0..ncols)
                .flat_map(|c| fe_bytes(&backend.wire_value(&wires[i * ncols + c]).unwrap()))
                .collect();
            match index.get(&key) {
                Some(&j) => mult[j] += 1,
                None => missing += 1,
            }
        }
        if missing > 0 {
            warn!("verity_v0 lookup: {missing} of {n} queries are not in the table (the verifier will reject)");
        }
    }
    let m_wires = mult
        .iter()
        .map(|&m| backend.input_private(if is_prover { Some(fe_from_u64::<B::FieldElement>(m)?) } else { None }))
        .collect::<Result<Vec<_>>>()?;

    // 2. challenges: s combines the columns, r is the logUp evaluation point.  Both are sent by the verifier only
    //    after every commitment above (and every earlier gate) has been received: `random()` flushes the prover's
    //    channel first and the verifier reaches this point only after reading the pending fix messages.
    let s = if ncols > 1 { backend.random()? } else { B::FieldElement::ONE };
    let r = backend.random()?;

    // 3. one inverse per query.  The prover's inverses are computed with one field inversion per call (Montgomery's
    //    batch trick): a per-element `inverse()` costs ~30 us in F384p and dominated the online phase (measured).
    let mut d_wires = Vec::with_capacity(n);
    let mut d_vals: Vec<B::FieldElement> = Vec::with_capacity(if is_prover { n } else { 0 });
    for i in 0..n {
        let mut q = backend.copy(&wires[i * ncols])?;
        let mut pw = s;
        for c in 1..ncols {
            let t = backend.mul_constant(&wires[i * ncols + c], pw)?;
            q = backend.add(&q, &t)?;
            pw *= s;
        }
        let d = backend.add_constant(&q, -r)?;
        if let Some(v) = backend.wire_value(&d) {
            ensure!(
                v != B::FieldElement::ZERO,
                ErrorKind::OtherError,
                "verity_v0 lookup: the challenge hit a query (probability N/p); rerun"
            );
            d_vals.push(v);
        }
        d_wires.push(d);
    }
    let d_invs = batch_inverse(&d_vals);
    let mut sum_inv: Option<B::Wire> = None;
    for (i, d) in d_wires.iter().enumerate() {
        let inv_w = backend.input_private(if is_prover { Some(d_invs[i]) } else { None })?;
        backend.assert_mul(d, &inv_w, &one)?;
        sum_inv = Some(match sum_inv {
            None => inv_w,
            Some(acc) => backend.add(&acc, &inv_w)?,
        });
    }

    // 4. one committed g_j = m_j / (T_j - r) per row, tied linearly to m_j
    let mut denoms = Vec::with_capacity(rows.len());
    for row in rows.iter() {
        let mut t = B::FieldElement::ZERO;
        let mut pw = B::FieldElement::ONE;
        for c in row {
            t += *c * pw;
            pw *= s;
        }
        let denom = t - r; // same orientation as d = x_i - r above
        ensure!(
            denom != B::FieldElement::ZERO,
            ErrorKind::OtherError,
            "verity_v0 lookup: the challenge hit a table row (probability R/p); rerun"
        );
        denoms.push(denom);
    }
    let denom_invs = if is_prover { batch_inverse(&denoms) } else { Vec::new() };
    let mut sum_g: Option<B::Wire> = None;
    for (j, denom) in denoms.iter().enumerate() {
        let g_val = backend.wire_value(&m_wires[j]).map(|m| m * denom_invs[j]);
        let g = backend.input_private(g_val)?;
        let lhs = backend.mul_constant(&g, *denom)?;
        let diff = backend.sub(&lhs, &m_wires[j])?;
        backend.assert_zero(&diff)?;
        sum_g = Some(match sum_g {
            None => g,
            Some(acc) => backend.add(&acc, &g)?,
        });
    }

    // 5. sum_i 1/(x_i - r) = sum_j m_j / (T_j - r)
    let zero = backend.constant(B::FieldElement::ZERO)?;
    let lhs = sum_inv.unwrap_or(zero);
    let rhs = sum_g.unwrap_or(zero);
    let diff = backend.sub(&lhs, &rhs)?;
    backend.assert_zero(&diff)
}

impl Plugin for VerityV0 {
    const NAME: &'static str = "verity_v0";

    fn instantiate(
        operation: &str,
        params: &[PluginTypeArg],
        output_counts: &[(TypeId, WireCount)],
        input_counts: &[(TypeId, WireCount)],
        type_store: &TypeStore,
        _fun_store: &FunStore,
    ) -> Result<PluginExecution> {
        ensure!(
            output_counts.is_empty(),
            ErrorKind::OtherError,
            "{}: no outputs allowed",
            Self::NAME
        );
        ensure!(
            !input_counts.is_empty(),
            ErrorKind::OtherError,
            "{}: at least one input range",
            Self::NAME
        );
        let type_id = input_counts[0].0;
        ensure!(
            input_counts.iter().all(|c| c.0 == type_id),
            ErrorKind::OtherError,
            "{}: all inputs must have one type",
            Self::NAME
        );
        match type_store.get(&type_id)? {
            TypeSpecification::Field(_) => {}
            _ => bail!(ErrorKind::UnsupportedError, "{}: only field types", Self::NAME),
        }
        let total: u64 = input_counts.iter().map(|c| c.1).sum();
        let num = |i: usize| -> Result<u64> {
            match params.get(i) {
                Some(PluginTypeArg::Number(n)) => number_to_u64(n),
                _ => bail!(ErrorKind::OtherError, "{}: parameter {i} must be a number", Self::NAME),
            }
        };
        let op = match operation {
            "assert_mul" => {
                ensure!(params.is_empty(), ErrorKind::OtherError, "{}: assert_mul takes no params", Self::NAME);
                ensure!(total % 3 == 0, ErrorKind::OtherError, "{}: assert_mul needs 3k wires", Self::NAME);
                VerityOp::AssertMul
            }
            "range" => {
                ensure!(params.len() == 1, ErrorKind::OtherError, "{}: range takes one param", Self::NAME);
                let bits = num(0)?;
                ensure!((1..=20).contains(&bits), ErrorKind::OtherError, "{}: range bits in 1..=20", Self::NAME);
                VerityOp::Range { bits: bits as u32 }
            }
            "lookup" => {
                ensure!(params.len() >= 2, ErrorKind::OtherError, "{}: lookup needs ncols, nrows, table", Self::NAME);
                let ncols = num(0)? as usize;
                let nrows = num(1)? as usize;
                ensure!(ncols >= 1 && nrows >= 1, ErrorKind::OtherError, "{}: empty table", Self::NAME);
                ensure!(
                    params.len() == 2 + ncols * nrows,
                    ErrorKind::OtherError,
                    "{}: lookup table has {} entries, expected {}",
                    Self::NAME,
                    params.len() - 2,
                    ncols * nrows
                );
                ensure!(
                    total % (ncols as u64) == 0,
                    ErrorKind::OtherError,
                    "{}: lookup inputs must be {ncols}-tuples",
                    Self::NAME
                );
                let table = params[2..]
                    .iter()
                    .map(|a| match a {
                        PluginTypeArg::Number(n) => Ok(*n),
                        _ => bail!(ErrorKind::OtherError, "{}: table entries must be numbers", Self::NAME),
                    })
                    .collect::<Result<Vec<_>>>()?;
                VerityOp::Lookup { ncols, table }
            }
            _ => bail!(ErrorKind::OtherError, "{}: unknown operation {operation}", Self::NAME),
        };
        Ok(PluginExecution::Verity(VerityV0 { type_id, op }))
    }
}
