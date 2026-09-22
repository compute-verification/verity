//! The one-row-per-unit AIR, the trace builder, and a concrete field evaluator of the constraints.

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::Matrix;

use crate::ctx::{AirCtx, IntCtx};
use crate::program::{unit_program, UnitIo};

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
}

pub fn layout() -> Layout {
    let io = UnitIo { c: 0, a: vec![0; 16], b: vec![0; 16], y: 0 };
    let mut ctx = IntCtx::new();
    unit_program(&mut ctx, &io);
    let width = ctx.cols.len();
    // constraint names from a symbolic pass on a concrete row (names are layout data, values irrelevant)
    let row: Vec<F> = ctx.cols.iter().map(|&v| to_f(v)).collect();
    let mut ev = Evaluator { failing: vec![], count: 0, main: RowMajorMatrixView::new(&row, width) };
    let mut actx = AirCtx::new(&mut ev, &row);
    actx.constraint_names = Some(vec![]);
    unit_program(&mut actx, &io);
    let constraint_names = actx.constraint_names.take().unwrap();
    Layout { width, col_names: ctx.names, constraint_names, n_hints: ctx.n_hints, n_mat: ctx.n_mat }
}

pub fn to_f(v: i128) -> F {
    let p = crate::ctx::GOLDILOCKS_P as i128;
    F::from_canonical_u64((((v % p) + p) % p) as u64)
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

/// The trace: one row per unit, padded to a power of two (at least `min_rows`) by repeating the last row.
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

/// A concrete `AirBuilder` over Goldilocks: evaluates every constraint on one row and records the failing ones.
pub struct Evaluator<'a> {
    pub failing: Vec<usize>,
    pub count: usize,
    pub main: RowMajorMatrixView<'a, F>,
}

impl<'a> AirBuilder for Evaluator<'a> {
    type F = F;
    type Expr = F;
    type Var = F;
    type M = RowMajorMatrixView<'a, F>;
    fn main(&self) -> Self::M {
        self.main
    }
    fn is_first_row(&self) -> F {
        F::zero()
    }
    fn is_last_row(&self) -> F {
        F::zero()
    }
    fn is_transition_window(&self, _size: usize) -> F {
        F::one()
    }
    fn assert_zero<I: Into<F>>(&mut self, x: I) {
        if x.into() != F::zero() {
            self.failing.push(self.count);
        }
        self.count += 1;
    }
}

/// Names of the constraints that fail on `row` (empty = the row satisfies the AIR).
pub fn failing_constraints(lay: &Layout, row: &[F]) -> Vec<String> {
    let mut ev = Evaluator { failing: vec![], count: 0, main: RowMajorMatrixView::new(row, lay.width) };
    let air = UnitAir { width: lay.width };
    air.eval(&mut ev);
    ev.failing.iter().map(|&i| lay.constraint_names[i].clone()).collect()
}

pub fn f_to_u64(v: F) -> u64 {
    v.as_canonical_u64()
}
