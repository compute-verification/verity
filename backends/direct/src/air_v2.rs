//! The checker-v2 chain AIR over BabyBear with LogUp (PROTOCOL.md 15, `V2_AIR.md`): one row per transition unit,
//! the state `(s, e_hat, M)` carried row to row, tuple lookups against the preprocessed table set, statement
//! binding (`statement.rs`).  Reuses `lookup_stark.rs` (the three-round fork) unchanged.

use std::sync::Arc;

use p3_air::{AirBuilder, BaseAir};
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, PrimeField64, TwoAdicField};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{StarkGenericConfig, Val, VerificationError};

use crate::air::{to_field, Evaluator, Layout, LookupEvaluator, RowSpec, VuIo};
use crate::ctx::{modp, AirCtx, FieldKind, IntCtx};
use crate::lookup::{aux_matrix, eval_lookup, group_queries, lookup_constraint_names, multiplicities, LookupAirBuilder, LookupLayout, TableSet, INVALID};
use crate::lookup_stark::{prove_lookup, verify_lookup, LookupAir, LookupProof, StageTimes};
use crate::program::{Word32, COL_CLAIM, COL_FIRST, COL_LAST, COL_STEP, STEPS};
use crate::program_v2::{chain_program_v2, gadget_program_v2, V2Io, V2Row, COL_ST_E, COL_ST_M, COL_ST_S};
use crate::stark::{Backend, FriParams};
use crate::statement::{self, Statement};

/// Column indices the row-to-row constraints read.
#[derive(Clone, Debug)]
pub struct V2Idx {
    pub st: [usize; 3],
    pub y: [usize; 2],
    pub step: usize,
    pub first: usize,
    pub last: usize,
}

pub struct V2ChainAir {
    /// Main width (program columns + 1 multiplicity column).
    pub width: usize,
    pub kind: FieldKind,
    pub idx: V2Idx,
    pub lay: LookupLayout,
    pub ext_degree: usize,
}

fn dummy_io() -> V2Io {
    V2Io { st: V2Io::ZERO_STATE, a: vec![0; 16], b: vec![0; 16], step: 0, y16: 0 }
}

/// Names of the row-to-row and boundary constraints, in the order `eval_lookup` emits them after the row program.
pub fn tail_names() -> Vec<String> {
    let mut n = vec!["vu.row0_step".to_string(), "vu.row0_first".to_string(), "vu.next_first".to_string(), "vu.next_step".to_string()];
    for i in 0..3 {
        n.push(format!("vu.link[{i}]"));
    }
    n
}

impl V2ChainAir {
    pub fn mult_col(&self) -> usize {
        self.width - 1
    }
    pub fn new(kind: FieldKind, ext_degree: usize) -> (Self, Layout) {
        assert!(kind.small(), "the v2 AIR is the BabyBear realisation");
        let (lay, llay) = layout(kind, ext_degree);
        let idx = V2Idx {
            st: [lay.col(COL_ST_S), lay.col(COL_ST_E), lay.col(COL_ST_M)],
            y: [lay.col("pack.y_hi"), lay.col("pack.y_lo")],
            step: lay.col(COL_STEP),
            first: lay.col(COL_FIRST),
            last: lay.col(COL_LAST),
        };
        (Self { width: lay.width, kind, idx, lay: llay, ext_degree }, lay)
    }
}

/// The layout: a collecting witness run discovers the tables, a second run records the queries (names, degrees ->
/// groups), a symbolic run names the constraints.
pub fn layout(kind: FieldKind, ext_degree: usize) -> (Layout, LookupLayout) {
    let io = dummy_io();
    let mut collect = IntCtx::new_kind(kind).with_lookup(None);
    chain_program_v2(&mut collect, &io);
    let tables = TableSet::from_needs(&collect.needs);
    let mut ctx = IntCtx::new_kind(kind).with_lookup(Some(tables.clone()));
    chain_program_v2(&mut ctx, &io);
    let groups = group_queries(&ctx.query_degs);
    let llay = LookupLayout { tables, query_names: ctx.query_names.clone(), query_degs: ctx.query_degs.clone(), groups };
    let width = ctx.cols.len() + 1;
    let mut col_names = ctx.names.clone();
    col_names.push("lookup.m".to_string());
    let row: Vec<Goldilocks> = ctx.cols.iter().map(|&v| to_field(kind, v)).chain([Goldilocks::zero()]).collect();
    let two: Vec<Goldilocks> = [row.clone(), row.clone()].concat();
    let aux_w = llay.aux_width(ext_degree);
    let pre_w = llay.tables.pre_width();
    let aux = vec![Goldilocks::zero(); 2 * aux_w];
    let pre = vec![Goldilocks::zero(); 2 * pre_w];
    let mut ev = LookupEvaluator::<Goldilocks, p3_field::extension::BinomialExtensionField<Goldilocks, 2>> {
        failing: vec![],
        count: 0,
        main: RowMajorMatrixView::new(&two, width),
        pre: RowMajorMatrixView::new(&pre, pre_w),
        aux: RowMajorMatrixView::new(&aux, aux_w),
        first_row: false,
        last_row: false,
        z: Default::default(),
        beta: Default::default(),
    };
    let mut names;
    {
        let mut actx = AirCtx::with_kind(&mut ev, &row[..width - 1], kind).with_lookup(llay.tables.clone());
        actx.constraint_names = Some(vec![]);
        chain_program_v2(&mut actx, &io);
        names = actx.constraint_names.take().unwrap();
        assert_eq!(actx.queries.len(), llay.n_queries());
    }
    names.extend(tail_names());
    names.extend(lookup_constraint_names(&llay));
    (Layout { width, col_names, constraint_names: names, n_hints: ctx.n_hints, n_mat: ctx.n_mat }, llay)
}

impl<F2> BaseAir<F2> for V2ChainAir {
    fn width(&self) -> usize {
        self.width
    }
}

/// The row-to-row and boundary constraints (`tail_names` order): the trace starts a VU; `next.first = last`;
/// the step increments inside a VU; the state is carried inside a VU.
fn tail<AB: AirBuilder>(builder: &mut AB, idx: &V2Idx, local: &[AB::Var], next: &[AB::Var], st_out: &[AB::Expr; 3]) {
    let one = AB::Expr::one();
    builder.when_first_row().assert_zero(local[idx.step]);
    builder.when_first_row().assert_one(local[idx.first]);
    let nf: AB::Expr = one.clone() - next[idx.first].into();
    let mut t = builder.when_transition();
    t.assert_eq(next[idx.first], local[idx.last]);
    let inc: AB::Expr = next[idx.step].into() - local[idx.step].into() - one;
    t.assert_zero(nf.clone() * inc);
    for i in 0..3 {
        let link: AB::Expr = next[idx.st[i]].into() - st_out[i].clone();
        t.assert_zero(nf.clone() * link);
    }
}

impl<G: PrimeField64> LookupAir<G> for V2ChainAir {
    fn preprocessed_width(&self) -> usize {
        self.lay.tables.pre_width()
    }
    fn aux_width(&self) -> usize {
        self.lay.aux_width(self.ext_degree)
    }
    fn max_degree(&self) -> usize {
        crate::program::MAX_DEG
    }
    fn eval_lookup<AB: LookupAirBuilder<F = G>>(&self, builder: &mut AB) {
        let main = builder.main();
        let local: Vec<AB::Var> = main.row_slice(0).to_vec();
        let next: Vec<AB::Var> = main.row_slice(1).to_vec();
        let w = self.width - 1;
        let (queries, st_out) = {
            let mut ctx = AirCtx::with_kind(builder, &local[..w], self.kind).with_lookup(self.lay.tables.clone());
            let row = chain_program_v2(&mut ctx, &dummy_io());
            assert_eq!(ctx.cursor, w, "layout mismatch");
            let st: [AB::Expr; 3] = [row.st_out[0].e.clone(), row.st_out[1].e.clone(), row.st_out[2].e.clone()];
            (std::mem::take(&mut ctx.queries), st)
        };
        tail(builder, &self.idx, &local, &next, &st_out);
        eval_lookup(builder, &self.lay, &queries, local[w], &mut None);
    }
}

// --- witness ----------------------------------------------------------------------------------------------------------

/// The integer value of a state expression (small, honest) folded from the field representative.
fn fold(kind: FieldKind, v: i128) -> i128 {
    let p = kind.prime() as i128;
    let r = modp(v, kind.prime()) as i128;
    if r > p / 2 { r - p } else { r }
}

/// The honest rows of one VU with their query indices, the per-row outgoing states, and the integer violations.
pub fn vu_rows(kind: FieldKind, lay_width: usize, tables: &Arc<TableSet>, vu: &VuIo) -> (Vec<Vec<i128>>, Vec<Vec<u32>>, Vec<(i128, i128, i128)>, Vec<String>) {
    let steps = STEPS as usize;
    let mut rows = Vec::with_capacity(steps);
    let mut qs = Vec::with_capacity(steps);
    let mut states = Vec::with_capacity(steps);
    let mut violations = vec![];
    let mut st = V2Io::ZERO_STATE;
    for s in 0..steps {
        let io = V2Io { st, a: vu.x[16 * s..16 * s + 16].to_vec(), b: vu.w[16 * s..16 * s + 16].to_vec(), step: s as u64, y16: vu.y16 };
        let mut ctx = IntCtx::for_trace(kind).with_lookup(Some(tables.clone()));
        let row: V2Row<_> = chain_program_v2(&mut ctx, &io);
        debug_assert_eq!(ctx.cols.len() + 1, lay_width);
        for v in ctx.violations {
            violations.push(format!("step {s}: {v}"));
        }
        st = (fold(kind, row.st_out[0].v), fold(kind, row.st_out[1].v), fold(kind, row.st_out[2].v));
        states.push(st);
        rows.push(ctx.cols);
        qs.push(ctx.queries);
    }
    (rows, qs, states, violations)
}

/// The FP32 accumulator word of a state (for the recorded-accumulator gate): `Pack` on `(s, e_hat, M, z = [M = 0])`
/// with no overflow.  A state whose exponent exceeds `E_MAX` is the overflow word (the recorded accumulators of
/// in-domain instances never reach it before the last unit).
pub fn state_word(st: (i128, i128, i128)) -> u64 {
    use crate::params::*;
    let (s, e, m) = st;
    if m == 0 {
        return 0;
    }
    if e > E_MAX {
        return ((s as u64) << 31) | ((T_FINITE as u64) << 23);
    }
    let fld = e + BIAS - 1 + (m >> ACC_FRAC);
    ((s as u64) << 31) | ((fld as u64) << 23) | ((m & ((1 << ACC_FRAC) - 1)) as u64)
}

/// The zero VU (the prescribed padding).
pub fn zero_vu() -> VuIo {
    crate::air::zero_vu()
}

/// The chain trace: main matrix (program columns + multiplicity), flat `n_rows x Q` query indices, every VU's
/// per-step accumulator words.  Rows in VU order, then the zero VU's rows tiled; height at least `min_rows` and
/// the table set's height.
pub fn trace<G: PrimeField64 + Send + Sync>(air: &V2ChainAir, vus: &[VuIo], min_rows: usize) -> (RowMajorMatrix<G>, Vec<u32>, Vec<Vec<u64>>) {
    let kind = air.kind;
    let lay_width = air.width;
    let tables = &air.lay.tables;
    let q = air.lay.n_queries();
    let steps = STEPS as usize;
    let real = vus.len() * steps;
    let n = real.max(min_rows).max(tables.total_rows).next_power_of_two();
    let mut values: Vec<G> = vec![G::zero(); n * lay_width];
    let mut queries: Vec<u32> = vec![INVALID; n * q];
    let (real_part, pad_part) = values.split_at_mut(real * lay_width);
    let (real_q, pad_q) = queries.split_at_mut(real * q);
    let accs: Vec<Vec<u64>> = real_part
        .par_chunks_mut(steps * lay_width)
        .zip(real_q.par_chunks_mut(steps * q))
        .zip(vus.par_iter())
        .map(|((chunk, qchunk), vu)| {
            let (rows, qs, states, viol) = vu_rows(kind, lay_width, tables, vu);
            assert!(viol.is_empty(), "honest VU violates the integer checker: {:?}", &viol[..viol.len().min(3)]);
            for (r, row) in rows.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    chunk[r * lay_width + j] = to_field(kind, v);
                }
                qchunk[r * q..(r + 1) * q].copy_from_slice(&qs[r]);
            }
            states.iter().map(|&s| state_word(s)).collect()
        })
        .collect();
    if !pad_part.is_empty() {
        let (zrows, zqs, _, viol) = vu_rows(kind, lay_width, tables, &zero_vu());
        assert!(viol.is_empty());
        let zrows: Vec<Vec<G>> = zrows.iter().map(|r| r.iter().map(|&v| to_field(kind, v)).collect()).collect();
        pad_part.par_chunks_mut(lay_width).zip(pad_q.par_chunks_mut(q)).enumerate().for_each(|(i, (row, qrow))| {
            row[..lay_width - 1].copy_from_slice(&zrows[i % steps]);
            qrow.copy_from_slice(&zqs[i % steps]);
        });
    }
    let mult = multiplicities(tables, &queries);
    let mc = lay_width - 1;
    for (r, &m) in mult.iter().enumerate() {
        values[r * lay_width + mc] = G::from_canonical_u32(m);
    }
    (RowMajorMatrix::new(values, lay_width), queries, accs)
}

// --- proving with statement binding ---------------------------------------------------------------------------------------

pub fn air_id(kind: FieldKind, lay: &Layout) -> [u8; 32] {
    statement::air_id(kind, lay)
}

fn absorb_header<B: Backend>(challenger: &mut <B::SC as StarkGenericConfig>::Challenger, kind: FieldKind, lay: &Layout, fri: FriParams, stmt: &Statement)
where
    Val<B::SC>: PrimeField64 + Send + Sync,
{
    use p3_challenger::CanObserve;
    let id = air_id(kind, lay);
    let ext = <<B::SC as StarkGenericConfig>::Challenge as AbstractExtensionField<Val<B::SC>>>::D;
    for v in statement::header_tagged::<Val<B::SC>>(statement::DOMAIN_TAG_V2, kind, &id, lay.width, ext, fri, stmt) {
        challenger.observe(v);
    }
}

/// The trace height a statement of `b` VUs needs under this AIR (the table set may exceed `B * STEPS`).
pub fn rows_for(air: &V2ChainAir, b: usize) -> usize {
    (b * STEPS as usize).max(statement::MIN_ROWS).max(air.lay.tables.total_rows).next_power_of_two()
}

/// Prove the v2 chain of `stmt` (the header absorbed before the first challenge; the trace must be the honest
/// trace for the statement -- checked).  Returns the proof, the stage timings and the preprocessed matrix.
pub fn prove_bound<B: Backend>(air: &V2ChainAir, lay: &Layout, main: RowMajorMatrix<Val<B::SC>>, queries: &[u32], fri: FriParams, stmt: &Statement) -> (LookupProof<B::SC>, StageTimes)
where
    Val<B::SC>: PrimeField64 + Send + Sync + TwoAdicField,
{
    let rows = main.height();
    assert_eq!(rows, rows_for(air, stmt.b()), "trace height differs from the statement's");
    let width = main.width();
    for (col, name, want) in public_columns::<Val<B::SC>>(stmt, lay, rows) {
        for (r, w) in want.iter().enumerate() {
            assert_eq!(main.values[r * width + col], *w, "prover: column {name} row {r} differs from the statement");
        }
    }
    let log_n = rows.trailing_zeros() as usize;
    let config = B::config(log_n, fri);
    let mut challenger = B::challenger();
    absorb_header::<B>(&mut challenger, air.kind, lay, fri, stmt);
    let pre: RowMajorMatrix<Val<B::SC>> = air.lay.tables.preprocessed(rows);
    let mult: Vec<Val<B::SC>> = (0..rows).map(|r| main.values[r * width + width - 1]).collect();
    let llay = air.lay.clone();
    prove_lookup(&config, air, &mut challenger, pre, main, |z, beta| aux_matrix(&llay, queries, &mult, rows, z, beta, true))
}

/// The statement's public columns on a trace of `rows` rows: the claim column, and the operand columns when bound.
pub fn public_columns<G: PrimeField64>(stmt: &Statement, lay: &Layout, rows: usize) -> Vec<(usize, String, Vec<G>)> {
    let mut out = vec![(lay.col(COL_CLAIM), COL_CLAIM.to_string(), stmt.expand_to(rows, |vu, _| stmt.words[vu]))];
    if let Some((x, w)) = &stmt.operands {
        let k = Statement::k();
        for j in 0..16 {
            let name = format!("x.a[{j}]");
            out.push((lay.col(&name), name, stmt.expand_to(rows, |vu, s| x[vu * k + 16 * s + j])));
            let name = format!("x.b[{j}]");
            out.push((lay.col(&name), name, stmt.expand_to(rows, |vu, s| w[vu * k + 16 * s + j])));
        }
    }
    out
}

/// Verify a v2 proof of `stmt`: `degree_bits` is fixed by the statement (and the table set), the header is
/// absorbed, the tables are recomputed, the claim column (and operand columns when bound) are checked at `zeta`.
pub fn verify_bound<B: Backend>(air: &V2ChainAir, lay: &Layout, proof: &LookupProof<B::SC>, fri: FriParams, stmt: &Statement) -> Result<(), VerificationError>
where
    Val<B::SC>: PrimeField64 + Send + Sync + TwoAdicField,
    <<B::SC as StarkGenericConfig>::Pcs as p3_commit::Pcs<<B::SC as StarkGenericConfig>::Challenge, <B::SC as StarkGenericConfig>::Challenger>>::Commitment: PartialEq,
{
    let rows = rows_for(air, stmt.b());
    let log_n = rows.trailing_zeros() as usize;
    let config = B::config(log_n, fri);
    let mut challenger = B::challenger();
    absorb_header::<B>(&mut challenger, air.kind, lay, fri, stmt);
    let pre: RowMajorMatrix<Val<B::SC>> = air.lay.tables.preprocessed(rows);
    let public: Vec<(usize, Vec<Val<B::SC>>)> = public_columns::<Val<B::SC>>(stmt, lay, rows).into_iter().map(|(col, _, want)| (col, want)).collect();
    verify_lookup(&config, air, &mut challenger, proof, pre, log_n, &public)
}

// --- evaluation of adversarial rows (check-rows, vu-neg) -------------------------------------------------------------------

/// Every failing constraint of the v2 AIR over a whole committed trace (main, preprocessed, aux) with the row index.
pub fn failing_trace<G: PrimeField64, EF: ExtensionField<G>>(air: &V2ChainAir, lay: &Layout, main: &RowMajorMatrix<G>, pre: &RowMajorMatrix<G>, aux: &RowMajorMatrix<G>, z: EF, beta: EF) -> Vec<(usize, String)> {
    let n = main.height();
    let (mw, pw, aw) = (main.width(), pre.width(), aux.width());
    (0..n)
        .into_par_iter()
        .flat_map_iter(|r| {
            let nx = (r + 1) % n;
            let pair = |m: &RowMajorMatrix<G>, w: usize| -> Vec<G> { [&m.values[r * w..(r + 1) * w], &m.values[nx * w..(nx + 1) * w]].concat() };
            let (m2, p2, a2) = (pair(main, mw), pair(pre, pw), pair(aux, aw));
            let mut ev = LookupEvaluator { failing: vec![], count: 0, main: RowMajorMatrixView::new(&m2, mw), pre: RowMajorMatrixView::new(&p2, pw), aux: RowMajorMatrixView::new(&a2, aw), first_row: r == 0, last_row: r + 1 == n, z, beta };
            air.eval_lookup(&mut ev);
            assert_eq!(ev.count, lay.constraint_names.len());
            ev.failing.iter().map(|&i| (r, lay.constraint_names[i].clone())).collect::<Vec<_>>()
        })
        .collect()
}

/// A row of an adversarial v2 trace: `RowSpec` with `c` read as the incoming FP32 accumulator word (decoded to
/// the state; `None` = the previous row's outgoing state), overrides by column name.
pub fn rows_spec(kind: FieldKind, lay_width: usize, tables: &Arc<TableSet>, specs: &[RowSpec]) -> (Vec<Vec<i128>>, Vec<Vec<u32>>, Vec<String>) {
    let mut rows: Vec<Vec<i128>> = Vec::with_capacity(specs.len());
    let mut qs = Vec::with_capacity(specs.len());
    let mut violations = vec![];
    let mut prev_state = V2Io::ZERO_STATE;
    for (i, sp) in specs.iter().enumerate() {
        let mut ctx = IntCtx::new_kind(kind).with_lookup(Some(tables.clone()));
        ctx.record_names = false;
        for (k, v) in &sp.overrides {
            ctx.overrides.insert(k.clone(), *v);
        }
        let st = match sp.c {
            Some(c) => decode_word(c),
            None => prev_state,
        };
        let io = V2Io { st, a: sp.a.clone(), b: sp.b.clone(), step: sp.step, y16: sp.y16 };
        let row = chain_program_v2(&mut ctx, &io);
        assert_eq!(ctx.cols.len() + 1, lay_width, "row {i}: width");
        for v in ctx.violations {
            violations.push(format!("row {i}: {v}"));
        }
        // the next row's state columns are exactly the field values the link constraint transports
        prev_state = (fold(kind, row.st_out[0].v), fold(kind, row.st_out[1].v), fold(kind, row.st_out[2].v));
        rows.push(ctx.cols);
        qs.push(ctx.queries);
    }
    (rows, qs, violations)
}

/// The state of an FP32 word (finite): `(s, e_hat, M)`; the zero state for +-0.
pub fn decode_word(c: u64) -> (i128, i128, i128) {
    use crate::params::*;
    let s = (c >> 31) as i128;
    let t = ((c >> 23) & 0xFF) as i128;
    let f = (c & ((1 << 23) - 1)) as i128;
    let m = f + if t != 0 { 1 << ACC_FRAC } else { 0 };
    if m == 0 {
        return (s, V2Io::ZERO_STATE.1, 0);
    }
    (s, t.max(1) - BIAS, m)
}

/// Row-local failing constraints of adversarial rows (program constraints, lookup helpers, out-of-table queries as
/// `lookup:<name>`); the global table-side / running-sum constraints need the whole trace and are skipped.
pub fn failing_rows<G: PrimeField64, EF: ExtensionField<G>>(air: &V2ChainAir, lay: &Layout, rows: &[Vec<i128>], queries: &[Vec<u32>], starts_trace: bool) -> Vec<(usize, String)> {
    let kind = air.kind;
    let n = rows.len();
    let q = air.lay.n_queries();
    let flat: Vec<u32> = queries.iter().flat_map(|r| r.iter().copied()).collect();
    assert_eq!(flat.len(), n * q);
    let z = EF::from_canonical_u64(0x1234_5678_9abc_def1 % kind.prime() as u64) + EF::monomial(1) * G::from_canonical_u32(7);
    let beta = EF::from_canonical_u64(0x0fed_cba9_8765_4321 % kind.prime() as u64) + EF::monomial(1) * G::from_canonical_u32(11);
    let mult = vec![G::zero(); n];
    let aux = aux_matrix::<G, EF>(&air.lay, &flat, &mult, n, z, beta, false);
    let mw = air.width;
    let pw = air.lay.tables.pre_width();
    let mut mvals = Vec::with_capacity(n * mw);
    for r in rows {
        mvals.extend(r.iter().map(|&v| to_field::<G>(kind, v)));
        mvals.push(G::zero());
    }
    let main = RowMajorMatrix::new(mvals, mw);
    let pre = RowMajorMatrix::new(vec![G::zero(); n * pw], pw);
    let global = ["lookup.table", "lookup.sum_first", "lookup.sum_step", "lookup.sum_last"];
    let tail = tail_names();
    let mut out = vec![];
    for r in 0..n {
        let nx = if r + 1 < n { r + 1 } else { r };
        let pair = |m: &RowMajorMatrix<G>, w: usize| -> Vec<G> { [&m.values[r * w..(r + 1) * w], &m.values[nx * w..(nx + 1) * w]].concat() };
        let (m2, p2, a2) = (pair(&main, mw), pair(&pre, pw), pair(&aux, aux.width()));
        let mut ev = LookupEvaluator { failing: vec![], count: 0, main: RowMajorMatrixView::new(&m2, mw), pre: RowMajorMatrixView::new(&p2, pw), aux: RowMajorMatrixView::new(&a2, aux.width()), first_row: starts_trace && r == 0, last_row: false, z, beta };
        air.eval_lookup(&mut ev);
        assert_eq!(ev.count, lay.constraint_names.len());
        let last = r + 1 == n;
        for &i in &ev.failing {
            let name = &lay.constraint_names[i];
            if global.contains(&name.as_str()) {
                continue;
            }
            if last && tail.contains(name) && name != "vu.row0_step" && name != "vu.row0_first" {
                continue;
            }
            out.push((r, name.clone()));
        }
        for (i, &qi) in queries[r].iter().enumerate() {
            if qi == INVALID {
                out.push((r, format!("lookup:{}", air.lay.query_names[i])));
            }
        }
    }
    out
}

/// The honest outputs of a v2 gadget on the case's inputs (the v2 semantics: e.g. the exponent of a zero result is
/// LEADNORM's, not v1's `E - 25`).
pub fn gadget_honest_outputs(kind: FieldKind, case: &crate::program::GadgetCase) -> Vec<i128> {
    let mut ctx = IntCtx::new_kind(kind).with_lookup(None);
    crate::program_v2::gadget_outputs_v2(&mut ctx, case).iter().map(|v| v.v).collect()
}

/// A gadget-level case on the v2 gadgets: `(failing constraint names incl. `lookup:<name>`, integer violations)`.
pub fn gadget_failing<G: PrimeField64>(kind: FieldKind, case: &crate::program::GadgetCase, overrides: &[(String, i128)]) -> (Vec<String>, Vec<String>) {
    let mut collect = IntCtx::new_kind(kind).with_lookup(None);
    gadget_program_v2(&mut collect, case);
    let tables = TableSet::from_needs(&collect.needs);
    let mut ctx = IntCtx::new_kind(kind).with_lookup(Some(tables.clone()));
    for (k, v) in overrides {
        ctx.overrides.insert(k.clone(), *v);
    }
    gadget_program_v2(&mut ctx, case);
    let width = ctx.cols.len();
    let row: Vec<G> = ctx.cols.iter().map(|&v| to_field(kind, v)).collect();
    let two: Vec<G> = [row.clone(), row.clone()].concat();
    let mut ev = Evaluator { failing: vec![], count: 0, main: RowMajorMatrixView::new(&two, width), first_row: false };
    let mut actx = AirCtx::with_kind(&mut ev, &row, kind).with_lookup(tables.clone());
    actx.constraint_names = Some(vec![]);
    gadget_program_v2(&mut actx, case);
    let names = actx.constraint_names.take().unwrap();
    assert_eq!(ev.count, names.len());
    let mut failing: Vec<String> = ev.failing.iter().map(|&i| names[i].clone()).collect();
    for (i, &q) in ctx.queries.iter().enumerate() {
        if q == INVALID {
            failing.push(format!("lookup:{}", ctx.query_names[i]));
        }
    }
    (failing, ctx.violations)
}

/// The output word of a row from its limb columns.
pub fn row_word(idx: &V2Idx, row: &[i128], kind: FieldKind) -> u64 {
    (modp(row[idx.y[0]], kind.prime()) << 16) | modp(row[idx.y[1]], kind.prime())
}

pub fn word_of(w: &Word32<crate::ctx::IVal>, kind: FieldKind) -> u64 {
    match w {
        Word32::Full(v) => modp(v.v, kind.prime()),
        Word32::Limbs { hi, lo } => (modp(hi.v, kind.prime()) << 16) | modp(lo.v, kind.prime()),
    }
}
