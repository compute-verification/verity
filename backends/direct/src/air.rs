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

use crate::ctx::{modp, AirCtx, FieldKind, IntCtx};
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
        let idx = &self.idx;
        let one = AB::Expr::one();
        // boundary: the trace starts a VU
        builder.when_first_row().assert_zero(local[idx.step]);
        builder.when_first_row().assert_one(local[idx.first]);
        // row to row: next.first = last; step increments inside a VU; the accumulator is carried
        let nf: AB::Expr = one.clone() - next[idx.first].into();
        {
            let mut t = builder.when_transition();
            t.assert_eq(next[idx.first], local[idx.last]);
            let inc: AB::Expr = next[idx.step].into() - local[idx.step].into() - one;
            t.assert_zero(nf.clone() * inc);
            for (ci, yi) in idx.c.iter().zip(idx.y.iter()) {
                let link: AB::Expr = next[*ci].into() - local[*yi].into();
                t.assert_zero(nf.clone() * link);
            }
        }
    }
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
