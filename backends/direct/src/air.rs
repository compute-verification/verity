//! The AIRs, the trace builders, and a concrete field evaluator of the constraints.
//!
//! * `UnitAir`: one independent row per transition unit (Goldilocks; the wave-1 diagnostic, kept as is).
//! * `ChainAir`: one row per transition unit with the accumulator carried row to row -- the K=1536 VU is 96
//!   consecutive rows from `c_0 = 0` to the epilogue, over Goldilocks or BabyBear (`FieldKind`).

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;

use std::sync::Arc;

use p3_field::{AbstractExtensionField, ExtensionField};

use crate::ctx::{modp, AirCtx, FieldKind, IntCtx};
use crate::lookup::{aux_matrix, eval_lookup, group_queries, lookup_constraint_names, multiplicities, LookupAirBuilder, LookupLayout, TableSet, INVALID, PRE_WIDTH};
use crate::lookup_stark::LookupAir;
use crate::program::{chain_program, unit_program, ChainIo, UnitIo, Word32, COL_FIRST, COL_LAST, COL_STEP, STEPS};

pub type F = Goldilocks;

/// The layout: column names in allocation order (from one witness run), constraint names in emission order.
pub struct Layout {
    pub width: usize,
    pub col_names: Vec<String>,
    pub constraint_names: Vec<String>,
    pub n_hints: usize,
    pub n_mat: usize,
}

impl Layout {
    pub fn col(&self, name: &str) -> usize {
        self.col_names.iter().position(|n| n == name).unwrap_or_else(|| panic!("no column {name}"))
    }
    pub fn has(&self, name: &str) -> bool {
        self.col_names.iter().any(|n| n == name)
    }
}

/// The transition-unit layout (Goldilocks).
pub fn layout() -> Layout {
    let io = UnitIo { c: 0, a: vec![0; 16], b: vec![0; 16], y: 0 };
    let mut ctx = IntCtx::new();
    unit_program(&mut ctx, &io);
    let width = ctx.cols.len();
    // constraint names from a symbolic pass on a concrete row (names are layout data, values irrelevant)
    let row: Vec<F> = ctx.cols.iter().map(|&v| to_f(v)).collect();
    let two: Vec<F> = [row.clone(), row.clone()].concat();
    let mut ev = Evaluator { failing: vec![], count: 0, main: RowMajorMatrixView::new(&two, width), first_row: false };
    let mut actx = AirCtx::new(&mut ev, &row);
    actx.constraint_names = Some(vec![]);
    unit_program(&mut actx, &io);
    let constraint_names = actx.constraint_names.take().unwrap();
    Layout { width, col_names: ctx.names, constraint_names, n_hints: ctx.n_hints, n_mat: ctx.n_mat }
}

pub fn to_f(v: i128) -> F {
    F::from_canonical_u64(modp(v, FieldKind::Goldilocks.prime()))
}

pub fn to_field<G: PrimeField64>(kind: FieldKind, v: i128) -> G {
    G::from_canonical_u64(modp(v, kind.prime()))
}

/// One honest row (or a mutated one when `overrides` is given).  Returns the integer violations too.
pub fn gen_row(io: &UnitIo, overrides: &[(&str, i128)]) -> (Vec<F>, Vec<String>) {
    let mut ctx = IntCtx::new();
    for (k, v) in overrides {
        ctx.overrides.insert(k.to_string(), *v);
    }
    unit_program(&mut ctx, io);
    (ctx.cols.iter().map(|&v| to_f(v)).collect(), ctx.violations)
}

/// The unit trace: one row per unit, padded to a power of two (at least `min_rows`) by repeating the last row.
pub fn trace(units: &[UnitIo], min_rows: usize) -> RowMajorMatrix<F> {
    let lay_width = layout().width;
    let n = units.len().max(min_rows).next_power_of_two();
    let mut values = Vec::with_capacity(n * lay_width);
    let mut last: Vec<F> = vec![];
    for u in units {
        let (row, viol) = gen_row(u, &[]);
        assert!(viol.is_empty(), "honest unit violates the integer checker: {:?}", viol);
        assert_eq!(row.len(), lay_width);
        values.extend_from_slice(&row);
        last = row;
    }
    for _ in units.len()..n {
        values.extend_from_slice(&last);
    }
    RowMajorMatrix::new(values, lay_width)
}

pub struct UnitAir {
    pub width: usize,
}

impl<F2> BaseAir<F2> for UnitAir {
    fn width(&self) -> usize {
        self.width
    }
}

impl<AB: AirBuilder> Air<AB> for UnitAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let row: Vec<AB::Var> = local.to_vec();
        drop(local);
        let mut ctx = AirCtx::new(builder, &row);
        // the AIR context never reads the io
        let io = UnitIo { c: 0, a: vec![0; 16], b: vec![0; 16], y: 0 };
        unit_program(&mut ctx, &io);
        assert_eq!(ctx.cursor, self.width, "layout mismatch");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The chain
// ---------------------------------------------------------------------------------------------------------------

/// Column indices the row-to-row constraints read.
#[derive(Clone, Debug)]
pub struct ChainIdx {
    pub c: Vec<usize>,
    pub y: Vec<usize>,
    pub step: usize,
    pub first: usize,
    pub last: usize,
}

pub struct ChainAir {
    pub width: usize,
    pub kind: FieldKind,
    pub idx: ChainIdx,
}

impl ChainAir {
    pub fn new(kind: FieldKind) -> (Self, Layout) {
        let lay = chain_layout(kind);
        let idx = chain_idx(kind, &lay);
        (Self { width: lay.width, kind, idx }, lay)
    }
}

fn chain_idx(kind: FieldKind, lay: &Layout) -> ChainIdx {
    let (c, y) = if kind.small() {
        (vec![lay.col("x.c_hi"), lay.col("x.c_lo")], vec![lay.col("pack.y_hi"), lay.col("pack.y_lo")])
    } else {
        (vec![lay.col("x.c")], vec![lay.col("pack.y")])
    };
    ChainIdx { c, y, step: lay.col(COL_STEP), first: lay.col(COL_FIRST), last: lay.col(COL_LAST) }
}

fn dummy_chain_io() -> ChainIo {
    ChainIo { c: 0, a: vec![0; 16], b: vec![0; 16], step: 0, y16: 0 }
}

/// Names of the row-to-row and boundary constraints, in the order `ChainAir::eval` emits them after the
/// row-local program.
pub fn chain_tail_names(kind: FieldKind) -> Vec<String> {
    let mut names = vec!["vu.row0_step".to_string(), "vu.row0_first".to_string(), "vu.next_first".to_string(), "vu.next_step".to_string()];
    let n_c = if kind.small() { 2 } else { 1 };
    for i in 0..n_c {
        names.push(format!("vu.link[{i}]"));
    }
    names
}

pub fn chain_layout(kind: FieldKind) -> Layout {
    let io = dummy_chain_io();
    let mut ctx = IntCtx::new_kind(kind);
    chain_program(&mut ctx, &io);
    let width = ctx.cols.len();
    let row: Vec<Goldilocks> = ctx.cols.iter().map(|&v| to_field(kind, v)).collect();
    let two: Vec<Goldilocks> = [row.clone(), row.clone()].concat();
    let mut ev = Evaluator { failing: vec![], count: 0, main: RowMajorMatrixView::new(&two, width), first_row: false };
    let mut actx = AirCtx::with_kind(&mut ev, &row, kind);
    actx.constraint_names = Some(vec![]);
    chain_program(&mut actx, &io);
    let mut constraint_names = actx.constraint_names.take().unwrap();
    constraint_names.extend(chain_tail_names(kind));
    Layout { width, col_names: ctx.names, constraint_names, n_hints: ctx.n_hints, n_mat: ctx.n_mat }
}

impl<F2> BaseAir<F2> for ChainAir {
    fn width(&self) -> usize {
        self.width
    }
}

/// The row-to-row and boundary constraints of the chain (`chain_tail_names` order).
fn chain_tail<AB: AirBuilder>(builder: &mut AB, idx: &ChainIdx, local: &[AB::Var], next: &[AB::Var]) {
    let one = AB::Expr::one();
    // boundary: the trace starts a VU
    builder.when_first_row().assert_zero(local[idx.step]);
    builder.when_first_row().assert_one(local[idx.first]);
    // row to row: next.first = last; step increments inside a VU; the accumulator is carried
    let nf: AB::Expr = one.clone() - next[idx.first].into();
    let mut t = builder.when_transition();
    t.assert_eq(next[idx.first], local[idx.last]);
    let inc: AB::Expr = next[idx.step].into() - local[idx.step].into() - one;
    t.assert_zero(nf.clone() * inc);
    for (ci, yi) in idx.c.iter().zip(idx.y.iter()) {
        let link: AB::Expr = next[*ci].into() - local[*yi].into();
        t.assert_zero(nf.clone() * link);
    }
}

impl<AB: AirBuilder> Air<AB> for ChainAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: Vec<AB::Var> = main.row_slice(0).to_vec();
        let next: Vec<AB::Var> = main.row_slice(1).to_vec();
        {
            let mut ctx = AirCtx::with_kind(builder, &local, self.kind);
            chain_program(&mut ctx, &dummy_chain_io());
            assert_eq!(ctx.cursor, self.width, "layout mismatch");
        }
        chain_tail(builder, &self.idx, &local, &next);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The chain with LogUp lookups (PROTOCOL.md section 14)
// ---------------------------------------------------------------------------------------------------------------

/// `ChainAir` with range checks and table lookups realised as LogUp queries.  The main trace is the program's
/// columns plus one multiplicity column (the last); the preprocessed matrix holds the tables; the auxiliary
/// matrix the helpers, `g` and the running sum.
pub struct LookupChainAir {
    /// Main width (program columns + 1 multiplicity column).
    pub width: usize,
    pub kind: FieldKind,
    pub idx: ChainIdx,
    pub lay: LookupLayout,
    pub ext_degree: usize,
}

impl LookupChainAir {
    /// Column of the multiplicity.
    pub fn mult_col(&self) -> usize {
        self.width - 1
    }
    pub fn new(kind: FieldKind, ext_degree: usize) -> (Self, Layout) {
        let (lay, llay) = chain_lookup_layout(kind, ext_degree);
        let mut idx = chain_idx(kind, &lay);
        // the multiplicity column is not a program column: `chain_idx` does not see it
        let _ = &mut idx;
        (Self { width: lay.width, kind, idx, lay: llay, ext_degree }, lay)
    }
}

/// The lookup layout of the chain: a collecting witness run discovers the tables, a second run records the
/// queries (names, degrees -> groups), a symbolic run names the constraints.
pub fn chain_lookup_layout(kind: FieldKind, ext_degree: usize) -> (Layout, LookupLayout) {
    let io = dummy_chain_io();
    let mut collect = IntCtx::new_kind(kind).with_lookup(None);
    chain_program(&mut collect, &io);
    let tables = TableSet::from_needs(&collect.needs);
    let mut ctx = IntCtx::new_kind(kind).with_lookup(Some(tables.clone()));
    chain_program(&mut ctx, &io);
    let groups = group_queries(&ctx.query_degs);
    let llay = LookupLayout { tables, query_names: ctx.query_names.clone(), query_degs: ctx.query_degs.clone(), groups };
    let width = ctx.cols.len() + 1;
    let mut col_names = ctx.names.clone();
    col_names.push("lookup.m".to_string());
    // constraint names: a symbolic pass on a concrete (zero) aux / preprocessed row
    let row: Vec<Goldilocks> = ctx.cols.iter().map(|&v| to_field(kind, v)).chain([Goldilocks::zero()]).collect();
    let two: Vec<Goldilocks> = [row.clone(), row.clone()].concat();
    let aux_w = llay.aux_width(ext_degree);
    let aux = vec![Goldilocks::zero(); 2 * aux_w];
    let pre = vec![Goldilocks::zero(); 2 * PRE_WIDTH];
    let mut ev = LookupEvaluator::<Goldilocks, p3_field::extension::BinomialExtensionField<Goldilocks, 2>> {
        failing: vec![],
        count: 0,
        main: RowMajorMatrixView::new(&two, width),
        pre: RowMajorMatrixView::new(&pre, PRE_WIDTH),
        aux: RowMajorMatrixView::new(&aux, aux_w),
        first_row: false,
        last_row: false,
        z: Default::default(),
        beta: Default::default(),
    };
    let mut names = vec![];
    {
        let mut actx = AirCtx::with_kind(&mut ev, &row[..width - 1], kind).with_lookup(llay.tables.clone());
        actx.constraint_names = Some(vec![]);
        chain_program(&mut actx, &io);
        names = actx.constraint_names.take().unwrap();
        assert_eq!(actx.queries.len(), llay.n_queries());
    }
    names.extend(chain_tail_names(kind));
    names.extend(lookup_constraint_names(&llay));
    (Layout { width, col_names, constraint_names: names, n_hints: ctx.n_hints, n_mat: ctx.n_mat }, llay)
}

impl<F2> BaseAir<F2> for LookupChainAir {
    fn width(&self) -> usize {
        self.width
    }
}

impl<G: PrimeField64> LookupAir<G> for LookupChainAir {
    fn preprocessed_width(&self) -> usize {
        PRE_WIDTH
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
        let queries = {
            let mut ctx = AirCtx::with_kind(builder, &local[..w], self.kind).with_lookup(self.lay.tables.clone());
            chain_program(&mut ctx, &dummy_chain_io());
            assert_eq!(ctx.cursor, w, "layout mismatch");
            std::mem::take(&mut ctx.queries)
        };
        chain_tail(builder, &self.idx, &local, &next);
        eval_lookup(builder, &self.lay, &queries, local[w], &mut None);
    }
}

/// The honest rows of one VU with their query indices (LogUp realisation).
pub fn chain_rows_lookup(kind: FieldKind, lay_width: usize, idx: &ChainIdx, tables: &Arc<TableSet>, vu: &VuIo) -> (Vec<Vec<i128>>, Vec<Vec<u32>>, Vec<u64>, Vec<String>) {
    let steps = STEPS as usize;
    let mut rows = Vec::with_capacity(steps);
    let mut qs = Vec::with_capacity(steps);
    let mut accs = Vec::with_capacity(steps);
    let mut violations = vec![];
    let mut c: u64 = 0;
    let p = kind.prime();
    for s in 0..steps {
        let io = ChainIo { c, a: vu.x[16 * s..16 * s + 16].to_vec(), b: vu.w[16 * s..16 * s + 16].to_vec(), step: s as u64, y16: vu.y16 };
        let mut ctx = IntCtx::for_trace(kind).with_lookup(Some(tables.clone()));
        chain_program(&mut ctx, &io);
        debug_assert_eq!(ctx.cols.len() + 1, lay_width);
        for v in ctx.violations {
            violations.push(format!("step {s}: {v}"));
        }
        c = if idx.y.len() == 1 { modp(ctx.cols[idx.y[0]], p) } else { (modp(ctx.cols[idx.y[0]], p) << 16) | modp(ctx.cols[idx.y[1]], p) };
        accs.push(c);
        rows.push(ctx.cols);
        qs.push(ctx.queries);
    }
    (rows, qs, accs, violations)
}

/// The chain trace in the LogUp realisation: the main matrix (program columns + multiplicity), the flat
/// `n_rows x Q` query index array, and every VU's intermediate accumulators.
pub fn chain_trace_lookup<G: PrimeField64 + Send + Sync>(air: &LookupChainAir, vus: &[VuIo], min_rows: usize) -> (RowMajorMatrix<G>, Vec<u32>, Vec<Vec<u64>>) {
    let kind = air.kind;
    let lay_width = air.width;
    let idx = &air.idx;
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
            let (rows, qs, accs, viol) = chain_rows_lookup(kind, lay_width, idx, tables, vu);
            assert!(viol.is_empty(), "honest VU violates the integer checker: {:?}", &viol[..viol.len().min(3)]);
            for (r, row) in rows.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    chunk[r * lay_width + j] = to_field(kind, v);
                }
                qchunk[r * q..(r + 1) * q].copy_from_slice(&qs[r]);
            }
            accs
        })
        .collect();
    if !pad_part.is_empty() {
        let (zrows, zqs, _, viol) = chain_rows_lookup(kind, lay_width, idx, tables, &zero_vu());
        assert!(viol.is_empty());
        let zrows: Vec<Vec<G>> = zrows.iter().map(|r| r.iter().map(|&v| to_field(kind, v)).collect()).collect();
        pad_part.par_chunks_mut(lay_width).zip(pad_q.par_chunks_mut(q)).enumerate().for_each(|(i, (row, qrow))| {
            row[..lay_width - 1].copy_from_slice(&zrows[i % steps]);
            qrow.copy_from_slice(&zqs[i % steps]);
        });
    }
    // the multiplicity column
    let mult = multiplicities(tables, &queries);
    let mc = lay_width - 1;
    for (r, &m) in mult.iter().enumerate() {
        values[r * lay_width + mc] = G::from_canonical_u32(m);
    }
    (RowMajorMatrix::new(values, lay_width), queries, accs)
}

/// A concrete `LookupAirBuilder` over one (local, next) row pair of main / preprocessed / aux.
pub struct LookupEvaluator<'a, G: PrimeField64, EF: ExtensionField<G>> {
    pub failing: Vec<usize>,
    pub count: usize,
    pub main: RowMajorMatrixView<'a, G>,
    pub pre: RowMajorMatrixView<'a, G>,
    pub aux: RowMajorMatrixView<'a, G>,
    pub first_row: bool,
    pub last_row: bool,
    pub z: EF,
    pub beta: EF,
}

impl<'a, G: PrimeField64, EF: ExtensionField<G>> AirBuilder for LookupEvaluator<'a, G, EF> {
    type F = G;
    type Expr = G;
    type Var = G;
    type M = RowMajorMatrixView<'a, G>;
    fn main(&self) -> Self::M {
        self.main
    }
    fn is_first_row(&self) -> G {
        if self.first_row { G::one() } else { G::zero() }
    }
    fn is_last_row(&self) -> G {
        if self.last_row { G::one() } else { G::zero() }
    }
    fn is_transition_window(&self, _size: usize) -> G {
        if self.last_row { G::zero() } else { G::one() }
    }
    fn assert_zero<I: Into<G>>(&mut self, x: I) {
        if x.into() != G::zero() {
            self.failing.push(self.count);
        }
        self.count += 1;
    }
}

impl<'a, G: PrimeField64, EF: ExtensionField<G>> LookupAirBuilder for LookupEvaluator<'a, G, EF> {
    type EF = EF;
    fn preprocessed(&self) -> Self::M {
        self.pre
    }
    fn aux(&self) -> Self::M {
        self.aux
    }
    fn challenges(&self) -> (EF, EF) {
        (self.z, self.beta)
    }
    fn ext_from_base(&self, x: G) -> EF {
        EF::from_base(x)
    }
    fn ext_from_cols(&self, cols: &[G]) -> EF {
        EF::from_base_slice(cols)
    }
    fn assert_zero_ext(&mut self, x: EF) {
        if x != EF::zero() {
            self.failing.push(self.count);
        }
        self.count += 1;
    }
    fn ext_degree(&self) -> usize {
        <EF as AbstractExtensionField<G>>::D
    }
}

/// Every failing constraint of the LogUp AIR over a whole committed trace (main, preprocessed, aux), with the
/// row index: the exhaustive check of the tests (all rows, all constraints, the verifier's own evaluation).
pub fn lookup_failing_trace<G: PrimeField64, EF: ExtensionField<G>>(air: &LookupChainAir, lay: &Layout, main: &RowMajorMatrix<G>, pre: &RowMajorMatrix<G>, aux: &RowMajorMatrix<G>, z: EF, beta: EF) -> Vec<(usize, String)> {
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

/// The row-local failing constraints of adversarial rows in the LogUp realisation (`check-rows`): the program
/// constraints and the lookup helper constraints (an out-of-table query leaves its group's helper unsatisfiable
/// -- the prover has no term for it -- which is exactly the verifier's constraint that rejects the row).  The
/// global table-side / running-sum constraints need the whole trace and are not evaluated here.
pub fn lookup_failing_rows<G: PrimeField64, EF: ExtensionField<G>>(air: &LookupChainAir, lay: &Layout, rows: &[Vec<i128>], queries: &[Vec<u32>], starts_trace: bool) -> Vec<(usize, String)> {
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
    let mut mvals = Vec::with_capacity(n * mw);
    for r in rows {
        mvals.extend(r.iter().map(|&v| to_field::<G>(kind, v)));
        mvals.push(G::zero());
    }
    let main = RowMajorMatrix::new(mvals, mw);
    let pre = RowMajorMatrix::new(vec![G::zero(); n * PRE_WIDTH], PRE_WIDTH);
    let global = ["lookup.table", "lookup.sum_first", "lookup.sum_step", "lookup.sum_last"];
    let tail = chain_tail_names(kind);
    let mut out = vec![];
    for r in 0..n {
        let nx = if r + 1 < n { r + 1 } else { r };
        let pair = |m: &RowMajorMatrix<G>, w: usize| -> Vec<G> { [&m.values[r * w..(r + 1) * w], &m.values[nx * w..(nx + 1) * w]].concat() };
        let (m2, p2, a2) = (pair(&main, mw), pair(&pre, PRE_WIDTH), pair(&aux, aux.width()));
        let mut ev = LookupEvaluator { failing: vec![], count: 0, main: RowMajorMatrixView::new(&m2, mw), pre: RowMajorMatrixView::new(&p2, PRE_WIDTH), aux: RowMajorMatrixView::new(&a2, aux.width()), first_row: starts_trace && r == 0, last_row: false, z, beta };
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
    }
    out
}

/// One VU's instance: operand words and the claimed output word.
#[derive(Clone, Debug)]
pub struct VuIo {
    pub x: Vec<u64>,
    pub w: Vec<u64>,
    pub y16: u64,
}

/// The 96 honest rows of one VU (integer columns), the chain's 96 intermediate accumulator words, and the
/// integer-checker violations (empty for an in-domain VU whose claimed word is right).
pub fn chain_rows(kind: FieldKind, lay_width: usize, idx: &ChainIdx, vu: &VuIo) -> (Vec<Vec<i128>>, Vec<u64>, Vec<String>) {
    let steps = STEPS as usize;
    assert_eq!(vu.x.len(), 16 * steps);
    assert_eq!(vu.w.len(), 16 * steps);
    let mut rows = Vec::with_capacity(steps);
    let mut accs = Vec::with_capacity(steps);
    let mut violations = vec![];
    let mut c: u64 = 0;
    let p = kind.prime();
    for s in 0..steps {
        let io = ChainIo { c, a: vu.x[16 * s..16 * s + 16].to_vec(), b: vu.w[16 * s..16 * s + 16].to_vec(), step: s as u64, y16: vu.y16 };
        let mut ctx = IntCtx::for_trace(kind);
        chain_program(&mut ctx, &io);
        debug_assert_eq!(ctx.cols.len(), lay_width);
        for v in ctx.violations {
            violations.push(format!("step {s}: {v}"));
        }
        // the output word from the materialised column(s), as the next row's input
        c = if idx.y.len() == 1 {
            modp(ctx.cols[idx.y[0]], p)
        } else {
            (modp(ctx.cols[idx.y[0]], p) << 16) | modp(ctx.cols[idx.y[1]], p)
        };
        accs.push(c);
        rows.push(ctx.cols);
    }
    (rows, accs, violations)
}

/// One row of an adversarial trace (`check-rows`): the unit's operands, its position, the claimed word and the
/// hints the adversary imposes by name (`IntCtx::overrides`; every realisation column -- limbs, selectors,
/// inverses, materialised copies -- is derived from the imposed values, the only way a prover could fill them).
/// `c = None` carries the previous row's output limbs (their canonical field values, whatever they are).
#[derive(Clone, Debug, serde::Deserialize)]
pub struct RowSpec {
    pub c: Option<u64>,
    pub a: Vec<u64>,
    pub b: Vec<u64>,
    pub step: u64,
    pub y16: u64,
    #[serde(default)]
    pub overrides: std::collections::BTreeMap<String, i128>,
}

/// The rows of an adversarial trace from `RowSpec`s (integer columns) and the integer-checker violations (for
/// information only: the field evaluator decides).
pub fn chain_rows_spec(kind: FieldKind, lay_width: usize, idx: &ChainIdx, specs: &[RowSpec]) -> (Vec<Vec<i128>>, Vec<String>) {
    let p = kind.prime();
    let mut rows: Vec<Vec<i128>> = Vec::with_capacity(specs.len());
    let mut violations = vec![];
    for (i, sp) in specs.iter().enumerate() {
        let mut ctx = IntCtx::new_kind(kind);
        ctx.record_names = false;
        for (k, v) in &sp.overrides {
            ctx.overrides.insert(k.clone(), *v);
        }
        let mut c = sp.c.unwrap_or(0);
        if sp.c.is_none() && i > 0 {
            let prev = &rows[i - 1];
            if idx.y.len() == 1 {
                c = modp(prev[idx.y[0]], p);
            } else {
                // the limbs as field elements, exactly as the link constraint transports them
                ctx.overrides.insert("x.c_hi".into(), modp(prev[idx.y[0]], p) as i128);
                ctx.overrides.insert("x.c_lo".into(), modp(prev[idx.y[1]], p) as i128);
            }
        }
        let io = ChainIo { c, a: sp.a.clone(), b: sp.b.clone(), step: sp.step, y16: sp.y16 };
        chain_program(&mut ctx, &io);
        assert_eq!(ctx.cols.len(), lay_width, "row {i}: width");
        for v in ctx.violations {
            violations.push(format!("row {i}: {v}"));
        }
        rows.push(ctx.cols);
    }
    (rows, violations)
}

/// A gadget-level case evaluated by the AIR's own constraints: `(failing constraint names, integer violations)`.
pub fn gadget_failing<G: PrimeField64>(kind: FieldKind, case: &crate::program::GadgetCase, overrides: &[(String, i128)]) -> (Vec<String>, Vec<String>) {
    let mut ctx = IntCtx::new_kind(kind);
    for (k, v) in overrides {
        ctx.overrides.insert(k.clone(), *v);
    }
    crate::program::gadget_program(&mut ctx, case);
    let width = ctx.cols.len();
    let row: Vec<G> = ctx.cols.iter().map(|&v| to_field(kind, v)).collect();
    let two: Vec<G> = [row.clone(), row.clone()].concat();
    let mut ev = Evaluator { failing: vec![], count: 0, main: RowMajorMatrixView::new(&two, width), first_row: false };
    let mut actx = AirCtx::with_kind(&mut ev, &row, kind);
    actx.constraint_names = Some(vec![]);
    crate::program::gadget_program(&mut actx, case);
    let names = actx.constraint_names.take().unwrap();
    assert_eq!(ev.count, names.len());
    (ev.failing.iter().map(|&i| names[i].clone()).collect(), ctx.violations)
}

/// The all-zero VU (`a = b = 0`, `c` stays `+0`, `y16 = 0`): the prescribed public padding.
pub fn zero_vu() -> VuIo {
    VuIo { x: vec![0; 16 * STEPS as usize], w: vec![0; 16 * STEPS as usize], y16: 0 }
}

/// The chain trace: `96 * vus.len()` rows in VU order, padded to a power of two (at least `min_rows`) with the
/// rows of the zero VU tiled cyclically (the padding may end inside a VU: the last row's transition is not
/// enforced by the AIR's transition window, and the padding never reaches an epilogue identity it cannot
/// satisfy).  Rows are generated in parallel over VUs.  Returns the trace and every VU's intermediate
/// accumulators; panics on an integer-checker violation (the caller checks the negatives with `chain_rows`).
pub fn chain_trace<G: PrimeField64 + Send + Sync>(kind: FieldKind, lay_width: usize, idx: &ChainIdx, vus: &[VuIo], min_rows: usize) -> (RowMajorMatrix<G>, Vec<Vec<u64>>) {
    let steps = STEPS as usize;
    let real = vus.len() * steps;
    let n = real.max(min_rows).next_power_of_two();
    let mut values: Vec<G> = vec![G::zero(); n * lay_width];
    let (real_part, pad_part) = values.split_at_mut(real * lay_width);
    let accs: Vec<Vec<u64>> = real_part
        .par_chunks_mut(steps * lay_width)
        .zip(vus.par_iter())
        .map(|(chunk, vu)| {
            let (rows, accs, viol) = chain_rows(kind, lay_width, idx, vu);
            assert!(viol.is_empty(), "honest VU violates the integer checker: {:?}", &viol[..viol.len().min(3)]);
            for (r, row) in rows.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    chunk[r * lay_width + j] = to_field(kind, v);
                }
            }
            accs
        })
        .collect();
    if !pad_part.is_empty() {
        let (zrows, _, viol) = chain_rows(kind, lay_width, idx, &zero_vu());
        assert!(viol.is_empty());
        let zrows: Vec<Vec<G>> = zrows.iter().map(|r| r.iter().map(|&v| to_field(kind, v)).collect()).collect();
        pad_part.par_chunks_mut(lay_width).enumerate().for_each(|(i, row)| {
            row.copy_from_slice(&zrows[i % steps]);
        });
    }
    (RowMajorMatrix::new(values, lay_width), accs)
}

// ---------------------------------------------------------------------------------------------------------------
// Concrete evaluation (tests, cross-checks)
// ---------------------------------------------------------------------------------------------------------------

/// A concrete `AirBuilder`: evaluates every constraint on one (local, next) row pair and records the failing ones.
/// `is_transition_window` is always 1 (a next row is always supplied); `is_first_row` is `first_row`.
pub struct Evaluator<'a, G: PrimeField64 = F> {
    pub failing: Vec<usize>,
    pub count: usize,
    pub main: RowMajorMatrixView<'a, G>,
    pub first_row: bool,
}

impl<'a, G: PrimeField64> AirBuilder for Evaluator<'a, G> {
    type F = G;
    type Expr = G;
    type Var = G;
    type M = RowMajorMatrixView<'a, G>;
    fn main(&self) -> Self::M {
        self.main
    }
    fn is_first_row(&self) -> G {
        if self.first_row { G::one() } else { G::zero() }
    }
    fn is_last_row(&self) -> G {
        G::zero()
    }
    fn is_transition_window(&self, _size: usize) -> G {
        G::one()
    }
    fn assert_zero<I: Into<G>>(&mut self, x: I) {
        if x.into() != G::zero() {
            self.failing.push(self.count);
        }
        self.count += 1;
    }
}

/// Names of the unit constraints that fail on `row` (empty = the row satisfies the AIR).
pub fn failing_constraints(lay: &Layout, row: &[F]) -> Vec<String> {
    let two: Vec<F> = [row.to_vec(), row.to_vec()].concat();
    let mut ev = Evaluator { failing: vec![], count: 0, main: RowMajorMatrixView::new(&two, lay.width), first_row: false };
    let air = UnitAir { width: lay.width };
    air.eval(&mut ev);
    ev.failing.iter().map(|&i| lay.constraint_names[i].clone()).collect()
}

/// Names of the chain constraints that fail on the pair (`local`, `next`); `first_row` enables the boundary
/// constraints.
pub fn chain_failing_constraints<G: PrimeField64>(air: &ChainAir, lay: &Layout, local: &[G], next: &[G], first_row: bool) -> Vec<String> {
    let two: Vec<G> = [local.to_vec(), next.to_vec()].concat();
    let mut ev = Evaluator { failing: vec![], count: 0, main: RowMajorMatrixView::new(&two, lay.width), first_row };
    air.eval(&mut ev);
    assert_eq!(ev.count, lay.constraint_names.len(), "constraint count differs from the layout's names");
    ev.failing.iter().map(|&i| lay.constraint_names[i].clone()).collect()
}

/// Every failing constraint over a whole set of integer rows (consecutive rows of one trace, first row = trace
/// start when `starts_trace`), with the row index; the last row is checked against `wrap_to` (the row that
/// follows it in the trace) when given, else only for its row-local constraints via a self-pair whose
/// transition constraints are ignored.
pub fn chain_failing_rows<G: PrimeField64>(kind: FieldKind, air: &ChainAir, lay: &Layout, rows: &[Vec<i128>], starts_trace: bool) -> Vec<(usize, String)> {
    let frows: Vec<Vec<G>> = rows.iter().map(|r| r.iter().map(|&v| to_field(kind, v)).collect()).collect();
    let mut out = vec![];
    let tail = chain_tail_names(kind);
    for i in 0..frows.len() {
        let next = if i + 1 < frows.len() { &frows[i + 1] } else { &frows[i] };
        let last = i + 1 == frows.len();
        for name in chain_failing_constraints(air, lay, &frows[i], next, starts_trace && i == 0) {
            // on the last supplied row only the row-local constraints are meaningful
            if last && tail.contains(&name) && name != "vu.row0_step" && name != "vu.row0_first" {
                continue;
            }
            out.push((i, name));
        }
    }
    out
}

pub fn f_to_u64(v: F) -> u64 {
    v.as_canonical_u64()
}

/// The 32-bit word of a row's output column(s).
pub fn row_word<G: PrimeField64>(idx: &ChainIdx, row: &[G]) -> u64 {
    if idx.y.len() == 1 {
        row[idx.y[0]].as_canonical_u64()
    } else {
        (row[idx.y[0]].as_canonical_u64() << 16) | row[idx.y[1]].as_canonical_u64()
    }
}

/// Convenience: `Word32` of the input accumulator for a row (unused by the AIR; for tests).
pub fn row_c<G: PrimeField64>(idx: &ChainIdx, row: &[G]) -> Word32<u64> {
    if idx.c.len() == 1 {
        Word32::Full(row[idx.c[0]].as_canonical_u64())
    } else {
        Word32::Limbs { hi: row[idx.c[0]].as_canonical_u64(), lo: row[idx.c[1]].as_canonical_u64() }
    }
}
