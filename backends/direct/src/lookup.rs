//! LogUp range and table lookups (PROTOCOL.md section 14): the tables, the query bookkeeping shared by the two
//! interpreters of the program, the prover-side auxiliary columns and the constraints.
//!
//! One preprocessed table matrix holds every table used by the program (range tables `T_w = [0, 2^w)` and the
//! four fixed tables POW / ALIGN / LEAD / NORM, each row tagged), one multiplicity column lives in the main trace,
//! and the auxiliary stage (committed after the challenges `z, beta`) holds one extension-field *helper* per group
//! of queries, the table-side fraction `g = m / (z - t)` and the running sum `S`.  The identity
//!
//!     sum_{rows, queries} 1 / (z - v_i)  =  sum_{table rows} m_r / (z - t_r)
//!
//! holds iff every queried tuple is a table row (LogUp; soundness `(N_queries + N_table) / |F_ext|` per
//! challenge, PROTOCOL.md 14.4).  A tuple `(key, out_1, .., out_4)` of table `tag` is combined as
//! `key + beta out_1 + ... + beta^4 out_4 + beta^5 tag`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use p3_air::AirBuilder;
use p3_field::{batch_multiplicative_inverse, AbstractExtensionField, AbstractField, ExtensionField, Field, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;

use crate::program::Table;

/// Number of tuple columns (key + up to four outputs); the tag is column `N_COLS`.
pub const N_COLS: usize = 5;
/// Preprocessed width: the tuple and the tag.
pub const PRE_WIDTH: usize = N_COLS + 1;
/// Sum of query degrees a helper column may carry (the helper constraint has degree `1 + sum`, `MAX_DEG = 5`).
pub const GROUP_DEG: usize = 4;
/// Index of an out-of-table query (adversarial rows): never a valid table row.
pub const INVALID: u32 = u32::MAX;
/// Range queries are split into 16-bit pieces above this width.
pub const SPLIT_BITS: u32 = 16;

#[derive(Clone, Debug)]
pub enum TableKind {
    /// `[0, 2^w)`, key only.
    Range(u32),
    /// A fixed table expanded to contiguous keys (saturating rows repeated up to the slack bound).
    Fixed(Table),
}

#[derive(Clone, Debug)]
pub struct TableSpec {
    pub kind: TableKind,
    pub name: String,
    /// The tag (`>= 1`; `0` is the padding row).
    pub tag: u32,
    /// First row in the preprocessed matrix.
    pub offset: usize,
    pub rows: usize,
    /// Lowest key (row `offset + (key - key_lo)`).
    pub key_lo: i128,
    /// Output columns per row (`rows x (N_COLS - 1)`, zero-padded); empty for range tables.
    pub outs: Vec<Vec<i128>>,
}

impl TableSpec {
    /// Row index of `key` in the preprocessed matrix, or `INVALID`.
    pub fn index(&self, key: i128) -> u32 {
        let k = key - self.key_lo;
        if k < 0 || k >= self.rows as i128 {
            INVALID
        } else {
            (self.offset + k as usize) as u32
        }
    }
    pub fn out(&self, key: i128, c: usize) -> i128 {
        let k = key - self.key_lo;
        if k < 0 || k >= self.rows as i128 {
            0
        } else {
            self.outs[k as usize][c]
        }
    }
}

/// The tables a program needs, discovered by a collecting witness run.
#[derive(Clone, Debug, Default)]
pub struct Needs {
    pub ranges: BTreeSet<u32>,
    pub fixed: BTreeMap<String, Table>,
}

#[derive(Clone, Debug)]
pub struct TableSet {
    pub tables: Vec<TableSpec>,
    pub total_rows: usize,
}

impl TableSet {
    pub fn from_needs(needs: &Needs) -> Arc<Self> {
        let mut tables = vec![];
        let mut offset = 0usize;
        let mut tag = 1u32;
        for &w in &needs.ranges {
            let rows = 1usize << w;
            tables.push(TableSpec { kind: TableKind::Range(w), name: format!("T{w}"), tag, offset, rows, key_lo: 0, outs: vec![] });
            offset += rows;
            tag += 1;
        }
        for (name, t) in &needs.fixed {
            let key_lo = t.keys[0];
            for (i, &k) in t.keys.iter().enumerate() {
                assert_eq!(k, key_lo + i as i128, "{name}: keys must be contiguous");
            }
            let last = *t.keys.last().unwrap();
            let key_hi = if t.saturate { last + t.slack_hi } else { last + 1 }; // exclusive
            let rows = (key_hi - key_lo) as usize;
            let ncol = t.rows[0].len();
            assert!(ncol <= N_COLS - 1, "{name}: too many outputs");
            let outs = (0..rows)
                .map(|i| {
                    let src = i.min(t.keys.len() - 1);
                    let mut o = t.rows[src].clone();
                    o.resize(N_COLS - 1, 0);
                    o
                })
                .collect();
            tables.push(TableSpec { kind: TableKind::Fixed(t.clone()), name: name.clone(), tag, offset, rows, key_lo, outs });
            offset += rows;
            tag += 1;
        }
        Arc::new(TableSet { tables, total_rows: offset })
    }
    pub fn range(&self, w: u32) -> &TableSpec {
        self.tables.iter().find(|t| matches!(t.kind, TableKind::Range(x) if x == w)).unwrap_or_else(|| panic!("no range table T{w}"))
    }
    pub fn fixed(&self, name: &str) -> &TableSpec {
        self.tables.iter().find(|t| matches!(t.kind, TableKind::Fixed(_)) && t.name == name).unwrap_or_else(|| panic!("no fixed table {name}"))
    }
    /// The tuple of table row `r` (`N_COLS` values and the tag).
    pub fn row(&self, r: usize) -> ([i128; N_COLS], u32) {
        let mut vals = [0i128; N_COLS];
        for t in &self.tables {
            if r >= t.offset && r < t.offset + t.rows {
                let k = t.key_lo + (r - t.offset) as i128;
                vals[0] = k;
                if let TableKind::Fixed(_) = t.kind {
                    for c in 0..N_COLS - 1 {
                        vals[c + 1] = t.outs[r - t.offset][c];
                    }
                }
                return (vals, t.tag);
            }
        }
        (vals, 0)
    }
    /// The preprocessed matrix: every table row, then padding rows `(0, .., 0; tag 0)`.
    pub fn preprocessed<G: PrimeField64>(&self, n_rows: usize) -> RowMajorMatrix<G> {
        assert!(n_rows >= self.total_rows, "trace too short for the tables: {} < {}", n_rows, self.total_rows);
        let mut v = vec![G::zero(); n_rows * PRE_WIDTH];
        v.par_chunks_mut(PRE_WIDTH).enumerate().take(self.total_rows).for_each(|(r, row)| {
            let (vals, tag) = self.row(r);
            for c in 0..N_COLS {
                row[c] = int_to_field(vals[c]);
            }
            row[N_COLS] = G::from_canonical_u32(tag);
        });
        RowMajorMatrix::new(v, PRE_WIDTH)
    }
    /// The combined table values `t_r` for every table row.
    pub fn values<G: PrimeField64, EF: ExtensionField<G>>(&self, beta: EF) -> Vec<EF> {
        let pows: Vec<EF> = (0..=N_COLS).map(|i| beta.exp_u64(i as u64)).collect();
        (0..self.total_rows)
            .into_par_iter()
            .map(|r| {
                let (vals, tag) = self.row(r);
                let mut acc = EF::zero();
                for c in 0..N_COLS {
                    acc += pows[c] * int_to_field::<G>(vals[c]);
                }
                acc + pows[N_COLS] * G::from_canonical_u32(tag)
            })
            .collect()
    }
    pub fn describe(&self) -> Vec<(String, usize)> {
        self.tables.iter().map(|t| (t.name.clone(), t.rows)).collect()
    }
}

fn int_to_field<G: PrimeField64>(v: i128) -> G {
    if v >= 0 {
        G::from_canonical_u64(v as u64)
    } else {
        -G::from_canonical_u64((-v) as u64)
    }
}

/// A recorded query on the symbolic side: the tuple's expressions (key first) and its table tag.
#[derive(Clone)]
pub struct Query<E> {
    pub name: String,
    pub tag: u32,
    pub vals: Vec<E>,
    pub deg: usize,
}

/// Greedy in-order packing of queries into helper groups by degree budget (`GROUP_DEG`); identical on both
/// sides because both record the same degrees in the same order.
pub fn group_queries(degs: &[usize]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = vec![];
    let mut cur: Vec<usize> = vec![];
    let mut cur_deg = 0usize;
    for (i, &d) in degs.iter().enumerate() {
        assert!(d >= 1 && d <= GROUP_DEG, "query {i} degree {d}");
        if cur_deg + d > GROUP_DEG && !cur.is_empty() {
            groups.push(std::mem::take(&mut cur));
            cur_deg = 0;
        }
        cur.push(i);
        cur_deg += d;
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    groups
}

/// The lookup layout of a row program: query names/degrees in order, the groups, the tables.
#[derive(Clone, Debug)]
pub struct LookupLayout {
    pub tables: Arc<TableSet>,
    pub query_names: Vec<String>,
    pub query_degs: Vec<usize>,
    pub groups: Vec<Vec<usize>>,
}

impl LookupLayout {
    pub fn n_queries(&self) -> usize {
        self.query_names.len()
    }
    pub fn n_helpers(&self) -> usize {
        self.groups.len()
    }
    /// Auxiliary width in base columns for extension degree `d`: helpers, `g`, `S`.
    pub fn aux_width(&self, d: usize) -> usize {
        d * (self.groups.len() + 2)
    }
}

/// The red team's table substitution: the same layout with one entry of the first fixed table's row `row`
/// changed (its first output column +1).  A verifier holding this table set must reject a proof made against the
/// honest one (PROTOCOL.md 14.6: the table digest is in the header and the preprocessed commitment is recomputed).
pub fn substitute_table_row(lay: &LookupLayout, row: usize) -> LookupLayout {
    let mut tables: TableSet = (*lay.tables).clone();
    let t = tables.tables.iter_mut().find(|t| matches!(t.kind, TableKind::Fixed(_))).expect("a fixed table");
    let r = row.min(t.rows - 1);
    t.outs[r][0] += 1;
    LookupLayout { tables: Arc::new(tables), query_names: lay.query_names.clone(), query_degs: lay.query_degs.clone(), groups: lay.groups.clone() }
}

/// Multiplicities of every table row over a flat `n_rows x Q` query index array (`INVALID` entries skipped).
pub fn multiplicities(tables: &TableSet, queries: &[u32]) -> Vec<u32> {
    let n = tables.total_rows;
    queries
        .par_chunks(1 << 16)
        .map(|chunk| {
            let mut m = vec![0u32; n];
            for &q in chunk {
                if q != INVALID {
                    m[q as usize] += 1;
                }
            }
            m
        })
        .reduce(
            || vec![0u32; n],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        )
}

/// The auxiliary matrix (base columns) for the challenges: per row `[h_0 .. h_{H-1}, g, S]`, each `D` base
/// columns.  `queries` is the flat `n_rows x Q` index array, `mult` the multiplicity column of the main trace.
pub fn aux_matrix<G: PrimeField64, EF: ExtensionField<G>>(lay: &LookupLayout, queries: &[u32], mult: &[G], n_rows: usize, z: EF, beta: EF, strict: bool) -> RowMajorMatrix<G> {
    let d = <EF as AbstractExtensionField<G>>::D;
    let q = lay.n_queries();
    assert_eq!(queries.len(), n_rows * q);
    let tvals = lay.tables.values::<G, EF>(beta);
    let denoms: Vec<EF> = tvals.iter().map(|&t| z - t).collect();
    let inv = batch_multiplicative_inverse(&denoms);
    let h = lay.n_helpers();
    let width = lay.aux_width(d);
    let mut aux = vec![G::zero(); n_rows * width];
    // per-row helpers and g (parallel), row sums collected for the prefix
    let row_sums: Vec<EF> = aux
        .par_chunks_mut(width)
        .enumerate()
        .map(|(r, out)| {
            let qs = &queries[r * q..(r + 1) * q];
            let mut row_sum = EF::zero();
            for (j, grp) in lay.groups.iter().enumerate() {
                let mut hj = EF::zero();
                for &i in grp {
                    let idx = qs[i];
                    // an out-of-table query (adversarial rows under the evaluator) has no term: the helper
                    // constraint of its group then fails, which is how the evaluator reports it
                    if idx != INVALID {
                        hj += inv[idx as usize];
                    } else {
                        assert!(!strict, "row {r}: query {} out of table", lay.query_names.get(i).map(String::as_str).unwrap_or("?"));
                    }
                }
                out[j * d..(j + 1) * d].copy_from_slice(hj.as_base_slice());
                row_sum += hj;
            }
            let g = if r < inv.len() { inv[r] * mult[r] } else { EF::zero() };
            out[h * d..(h + 1) * d].copy_from_slice(g.as_base_slice());
            row_sum - g
        })
        .collect();
    let mut s = EF::zero();
    for (r, rs) in row_sums.iter().enumerate() {
        s += *rs;
        aux[r * width + (h + 1) * d..(r + 1) * width].copy_from_slice(s.as_base_slice());
    }
    if strict {
        assert_eq!(s, EF::zero(), "LogUp identity does not balance on the honest witness");
    }
    RowMajorMatrix::new(aux, width)
}

// ---------------------------------------------------------------------------------------------------------------
// Constraints
// ---------------------------------------------------------------------------------------------------------------

/// An `AirBuilder` with the preprocessed table matrix, the auxiliary matrix, the two lookup challenges and
/// extension-field arithmetic for the lookup constraints.  `EF` is an extension of the builder's expression type
/// (so `EF * Expr`, `EF - Expr` are the cheap extension-by-base operations).
pub trait LookupAirBuilder: AirBuilder {
    type EF: AbstractExtensionField<Self::Expr> + Clone;
    fn preprocessed(&self) -> Self::M;
    fn aux(&self) -> Self::M;
    /// `(z, beta)`
    fn challenges(&self) -> (Self::EF, Self::EF);
    fn ext_from_base(&self, x: Self::Expr) -> Self::EF;
    /// An extension element from its `D` consecutive base columns.
    fn ext_from_cols(&self, cols: &[Self::Var]) -> Self::EF;
    fn assert_zero_ext(&mut self, x: Self::EF);
    fn ext_degree(&self) -> usize;
}

/// `(prod_j t_j, sum_j prod_{k != j} t_k)` by pairwise combination: `(pA pB, sA pB + sB pA)`; a single term is
/// `(t, 1)` (the `None` sum), so four degree-1 terms cost 5 extension products instead of 3 + 4 * 2.
fn prod_and_sum<EF: AbstractField + Clone>(terms: &[EF]) -> (EF, Option<EF>) {
    match terms.len() {
        0 => (EF::one(), Some(EF::zero())),
        1 => (terms[0].clone(), None),
        n => {
            let (pa, sa) = prod_and_sum(&terms[..n / 2]);
            let (pb, sb) = prod_and_sum(&terms[n / 2..]);
            let prod = pa.clone() * pb.clone();
            let sum = match (sa, sb) {
                (None, None) => pa + pb,
                (Some(sa), None) => sa * pb + pa,
                (None, Some(sb)) => pb + sb * pa,
                (Some(sa), Some(sb)) => sa * pb + sb * pa,
            };
            (prod, Some(sum))
        }
    }
}

/// Emits the lookup constraints for one row pair.  `queries` are this row's recorded queries (program order),
/// `mult` the multiplicity column of the main trace (local row).  Constraint order: helpers, table side, running
/// sum (first, transition, last).
pub fn eval_lookup<AB: LookupAirBuilder>(builder: &mut AB, lay: &LookupLayout, queries: &[Query<AB::Expr>], mult: AB::Var, names: &mut Option<Vec<String>>) {
    let d = builder.ext_degree();
    let (z, beta) = builder.challenges();
    let mut pows: Vec<AB::EF> = vec![AB::EF::one()];
    for i in 1..=N_COLS {
        pows.push(pows[i - 1].clone() * beta.clone());
    }
    assert_eq!(queries.len(), lay.n_queries(), "query count differs from the layout");
    // z - beta^N_COLS * tag, once per table (the tags are the constants of the row)
    let mut z_tag: BTreeMap<u32, AB::EF> = BTreeMap::new();
    for q in queries {
        z_tag.entry(q.tag).or_insert_with(|| z.clone() - pows[N_COLS].clone() * builder.ext_from_base(AB::Expr::from_canonical_u32(q.tag)));
    }
    // z - q_j = (z - beta^5 tag) - v_0 - sum_{c >= 1} beta^c v_c : extension-by-base operations only
    let terms: Vec<AB::EF> = queries
        .iter()
        .map(|q| {
            let mut acc = z_tag[&q.tag].clone() - q.vals[0].clone();
            for (c, v) in q.vals.iter().enumerate().skip(1) {
                acc -= pows[c].clone() * v.clone();
            }
            acc
        })
        .collect();
    let aux = builder.aux();
    let aux_local: Vec<AB::Var> = aux.row_slice(0).to_vec();
    let aux_next: Vec<AB::Var> = aux.row_slice(1).to_vec();
    let pre = builder.preprocessed();
    let pre_local: Vec<AB::Var> = pre.row_slice(0).to_vec();
    let h = lay.n_helpers();
    let ext_at = |b: &AB, row: &[AB::Var], j: usize| b.ext_from_cols(&row[j * d..(j + 1) * d]);
    let mut row_sum_local = AB::EF::zero();
    let mut row_sum_next = AB::EF::zero();
    for (j, grp) in lay.groups.iter().enumerate() {
        let hj = ext_at(builder, &aux_local, j);
        // h_j * prod_i (z - v_i) - sum_i prod_{k != i} (z - v_k) = 0
        let ts: Vec<AB::EF> = grp.iter().map(|&i| terms[i].clone()).collect();
        let (prod, sum) = prod_and_sum(&ts);
        if let Some(n) = names.as_mut() {
            n.push(format!("lookup.helper[{j}]"));
        }
        let c = match sum {
            Some(sum) => hj.clone() * prod - sum,
            None => hj.clone() * prod - AB::EF::one(),
        };
        builder.assert_zero_ext(c);
        row_sum_local += hj;
        row_sum_next += ext_at(builder, &aux_next, j);
    }
    // table side: g (z - t) - m = 0, t = key + sum beta^c out_c + beta^5 tag from the preprocessed row
    let mut t = builder.ext_from_base(pre_local[0].into());
    for c in 1..=N_COLS {
        let v: AB::Expr = pre_local[c].into();
        t += pows[c].clone() * v;
    }
    let g = ext_at(builder, &aux_local, h);
    let g_next = ext_at(builder, &aux_next, h);
    if let Some(n) = names.as_mut() {
        n.push("lookup.table".to_string());
    }
    let m: AB::Expr = mult.into();
    let table_c = g.clone() * (z.clone() - t) - m;
    builder.assert_zero_ext(table_c);
    row_sum_local -= g;
    row_sum_next -= g_next;
    // running sum
    let s = ext_at(builder, &aux_local, h + 1);
    let s_next = ext_at(builder, &aux_next, h + 1);
    let first = builder.is_first_row();
    let trans = builder.is_transition();
    let last = builder.is_last_row();
    if let Some(n) = names.as_mut() {
        n.push("lookup.sum_first".to_string());
        n.push("lookup.sum_step".to_string());
        n.push("lookup.sum_last".to_string());
    }
    builder.assert_zero_ext((s.clone() - row_sum_local) * first);
    builder.assert_zero_ext((s_next - s.clone() - row_sum_next) * trans);
    builder.assert_zero_ext(s * last);
}

/// Names of the lookup constraints in emission order.
pub fn lookup_constraint_names(lay: &LookupLayout) -> Vec<String> {
    let mut n: Vec<String> = (0..lay.n_helpers()).map(|j| format!("lookup.helper[{j}]")).collect();
    n.push("lookup.table".into());
    n.extend(["lookup.sum_first", "lookup.sum_step", "lookup.sum_last"].map(String::from));
    n
}

/// Soundness of the lookup argument (PROTOCOL.md 14.4): the LogUp identity is a rational function identity in
/// `z` of degree `<= N_queries + N_table`, and the `beta`-combination of `N_COLS + 1` columns is a polynomial of
/// degree `N_COLS` per tuple: `(N_q + N_t) / |F_ext|  +  N_COLS * N_q / |F_ext|`, in bits.
pub fn lookup_soundness_bits(n_rows: usize, n_queries_per_row: usize, n_table: usize, ext_bits: f64) -> f64 {
    let nq = n_rows as f64 * n_queries_per_row as f64;
    let terms = (nq + n_table as f64) + N_COLS as f64 * nq;
    ext_bits - terms.log2()
}

/// Cast helper for `Field` extension arithmetic in the witness builder.
pub fn ext_from_u32<EF: Field>(v: u32) -> EF {
    EF::from_canonical_u32(v)
}
