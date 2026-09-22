//! The exported unit circuit (`verity_numerical.gkr_export`): columns, product wires, assertions, tables, lookup
//! queries; its layered form; and the flat witness file.

use crate::field::Fp;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Term { pub idx: u32, pub coef: Fp }

#[derive(Clone, Debug, Default)]
pub struct LinForm { pub terms: Vec<Term>, pub konst: Fp }

#[derive(Clone, Debug)]
pub enum Wire { Input(u32), Prod { depth: u32, a: LinForm, b: LinForm } }

/// A lookup table: public rows (listed in the circuit text or materialised from an `itable` definition) and the
/// index from a key (row column 0) to its row: `dense` when row `i` has key `i` (every v2 table, the `R<w>`
/// ranges), else a hash map (the sparse `T_OP`: the non-finite words have no row).
#[derive(Clone, Debug)]
pub struct Table { pub name: String, pub cols: usize, pub rows: Vec<Vec<Fp>>, pub dense: bool, pub index: HashMap<Fp, usize> }

impl Table {
    pub fn new(name: String, cols: usize, rows: Vec<Vec<Fp>>) -> Table {
        let dense = rows.iter().enumerate().all(|(i, r)| r[0] == Fp::new(i as u64));
        let index = if dense { HashMap::new() } else { rows.iter().enumerate().map(|(i, r)| (r[0], i)).collect() };
        Table { name, cols, rows, dense, index }
    }
    /// The row index of a key, if the key has a row.
    #[inline]
    pub fn find(&self, key: Fp) -> Option<usize> {
        if self.dense { let k = key.to_u64() as usize; (k < self.rows.len() && key.to_u64() < self.rows.len() as u64).then_some(k) } else { self.index.get(&key).copied() }
    }
}

/// `itable NAME cols rows kind params...`: the v2 tables `circuit.rs` materialises from their definition
/// (`gkr_export.v2.implicit_table_header`; PROTOCOL.md 14.2).  Values are integers reduced into the field.
///   shift <bits> <scale> <R>     key x·R + 2d + s (x < 2^bits, d < R/2, s a bit) -> (1 - 2s)·floor(x·2^scale / 2^d)
///   tnorm <vb> <k_rows> <k_lo>   key v + 2^vb·koff (v < 2^vb, koff < k_rows) -> floor(v·2^(koff + k_lo))
pub fn materialise(kind: &str, params: &[i64], rows: usize) -> Vec<Vec<Fp>> {
    let signed = |v: u128, neg: bool| -> Fp { let f = Fp::new((v % (crate::field::P as u128)) as u64); if neg { -f } else { f } };
    match kind {
        "shift" => {
            let (bits, scale, r) = (params[0] as u32, params[1] as u32, params[2] as usize);
            assert_eq!(rows, (1usize << bits) * r, "shift table row count");
            (0..rows).map(|k| { let (x, rem) = (k / r, k % r); let (d, s) = (rem / 2, rem % 2); let v = ((x as u128) << scale) >> d; vec![Fp::new(k as u64), signed(v, s == 1)] }).collect()
        }
        "tnorm" => {
            let (vb, k_rows, k_lo) = (params[0] as u32, params[1] as usize, params[2]);
            assert_eq!(rows, (1usize << vb) * k_rows, "tnorm table row count");
            (0..rows).map(|key| { let (koff, v) = (key >> vb, key & ((1usize << vb) - 1)); let k = koff as i64 + k_lo; let t = if k >= 0 { (v as u128) << k } else { (v as u128) >> (-k) }; vec![Fp::new(key as u64), signed(t, false)] }).collect()
        }
        other => panic!("unknown implicit table kind {other}"),
    }
}

#[derive(Clone, Debug)]
pub struct Query { pub table: String, pub cols: Vec<LinForm> } // linear forms over *columns*

#[derive(Debug)]
pub struct Circuit {
    pub ncols: usize,
    pub col_names: Vec<String>,
    pub one_col: u32,
    pub wires: Vec<Wire>,
    pub asserts: Vec<(String, LinForm)>, // over wires
    pub tables: Vec<Table>,
    pub queries: Vec<Query>,
    pub max_depth: u32,
}

/// `R<w>` (1 <= w <= 16) names an implicit exact-width range table.
pub fn range_width(name: &str) -> Option<u32> {
    let w: u32 = name.strip_prefix('R')?.parse().ok()?;
    (1..=16).contains(&w).then_some(w)
}

fn parse_lin(tok: &mut std::slice::Iter<'_, &str>) -> LinForm {
    let n: usize = tok.next().unwrap().parse().unwrap();
    let mut terms = Vec::with_capacity(n);
    for _ in 0..n {
        let idx: u32 = tok.next().unwrap().parse().unwrap();
        let coef: u64 = tok.next().unwrap().parse().unwrap();
        terms.push(Term { idx, coef: Fp::new(coef) });
    }
    let konst: u64 = tok.next().unwrap().parse().unwrap();
    LinForm { terms, konst: Fp::new(konst) }
}

impl Circuit {
    pub fn parse(text: &str) -> Circuit {
        let mut c = Circuit { ncols: 0, col_names: vec![], one_col: 0, wires: vec![], asserts: vec![], tables: vec![], queries: vec![], max_depth: 0 };
        let mut prods: HashMap<u32, Wire> = HashMap::new();
        let mut nwires = 0usize;
        for line in text.lines() {
            let toks: Vec<&str> = line.split_whitespace().collect();
            if toks.is_empty() { continue; }
            let mut it = toks[1..].iter();
            match toks[0] {
                "ncols" => { c.ncols = it.next().unwrap().parse().unwrap(); c.col_names = vec![String::new(); c.ncols]; }
                "nwires" => { nwires = it.next().unwrap().parse().unwrap(); }
                "one" => { c.one_col = it.next().unwrap().parse().unwrap(); }
                "col" => { let i: usize = it.next().unwrap().parse().unwrap(); c.col_names[i] = it.next().unwrap().to_string(); }
                "prod" => {
                    let w: u32 = it.next().unwrap().parse().unwrap();
                    let depth: u32 = it.next().unwrap().parse().unwrap();
                    let a = parse_lin(&mut it);
                    let b = parse_lin(&mut it);
                    c.max_depth = c.max_depth.max(depth);
                    prods.insert(w, Wire::Prod { depth, a, b });
                }
                "assert" => { let name = it.next().unwrap().to_string(); c.asserts.push((name, parse_lin(&mut it))); }
                "table" => {
                    let name = it.next().unwrap().to_string();
                    let cols: usize = it.next().unwrap().parse().unwrap();
                    c.tables.push(Table { name, cols, rows: vec![], dense: true, index: HashMap::new() });
                }
                "itable" => {
                    let name = it.next().unwrap().to_string();
                    let cols: usize = it.next().unwrap().parse().unwrap();
                    let rows: usize = it.next().unwrap().parse().unwrap();
                    let kind = it.next().unwrap();
                    let params: Vec<i64> = it.map(|s| s.parse().unwrap()).collect();
                    let rows = materialise(kind, &params, rows);
                    assert!(rows.iter().all(|r| r.len() == cols), "{name}: implicit table column count");
                    c.tables.push(Table::new(name, cols, rows));
                }
                "row" => { let t = c.tables.last_mut().unwrap(); t.rows.push(it.map(|s| Fp::new(s.parse().unwrap())).collect()); }
                "query" => {
                    let table = it.next().unwrap().to_string();
                    let n: usize = it.next().unwrap().parse().unwrap();
                    let cols = (0..n).map(|_| parse_lin(&mut it)).collect();
                    c.queries.push(Query { table, cols });
                }
                other => panic!("unknown line kind {other}"),
            }
        }
        // wires: inputs are the columns in order of wire id; the export numbers wires in creation order, where an
        // input wire's id is recoverable from the fact that non-product ids are inputs in column order.
        let mut next_col = 0u32;
        for w in 0..nwires as u32 {
            if let Some(p) = prods.remove(&w) { c.wires.push(p); } else { c.wires.push(Wire::Input(next_col)); next_col += 1; }
        }
        assert_eq!(next_col as usize, c.ncols, "column/wire mismatch");
        // the range tables R<w> = [0, 2^w), w <= 16, are implicit: every one the queries reference, narrowest first
        let mut widths: Vec<u32> = c.queries.iter().filter_map(|q| range_width(&q.table)).collect();
        widths.sort_unstable();
        widths.dedup();
        for w in widths {
            c.tables.push(Table::new(format!("R{w}"), 1, (0..1u64 << w).map(|v| vec![Fp::new(v)]).collect()));
        }
        // the row-listed tables get their key index now that every row is read
        for t in c.tables.iter_mut() { if t.index.is_empty() { let rebuilt = Table::new(std::mem::take(&mut t.name), t.cols, std::mem::take(&mut t.rows)); *t = rebuilt; } }
        c
    }

    pub fn nwires(&self) -> usize { self.wires.len() }
    pub fn one_wire(&self) -> u32 { self.wires.iter().position(|w| matches!(w, Wire::Input(c) if *c == self.one_col)).unwrap() as u32 }
    pub fn table(&self, name: &str) -> &Table { self.tables.iter().find(|t| t.name == name).unwrap() }

    /// Every wire of one unit from its columns (products evaluated in id order: operands always precede).
    pub fn eval_wires(&self, cols: &[Fp], out: &mut Vec<Fp>) {
        out.clear();
        for w in &self.wires {
            let v = match w {
                Wire::Input(c) => cols[*c as usize],
                Wire::Prod { a, b, .. } => eval_lin(a, out) * eval_lin(b, out),
            };
            out.push(v);
        }
    }
}

impl Circuit {
    /// The verifier's relation on one row in the clear: every assertion zero, every query a row of its table.
    pub fn check_row(&self, cols: &[Fp]) -> Result<(), String> {
        let mut w = Vec::with_capacity(self.nwires());
        self.eval_wires(cols, &mut w);
        for (name, l) in &self.asserts {
            if eval_lin(l, &w) != Fp::ZERO { return Err(format!("assert {name}")); }
        }
        for (i, q) in self.queries.iter().enumerate() {
            let vals: Vec<Fp> = q.cols.iter().map(|l| eval_lin(l, &w)).collect();
            let t = self.table(&q.table);
            if !t.find(vals[0]).map(|ri| t.rows[ri][..] == vals[..]).unwrap_or(false) { return Err(format!("query {i} {} {:?}", q.table, vals.iter().map(|v| v.to_u64()).collect::<Vec<_>>())); }
        }
        Ok(())
    }
}

pub fn eval_lin(l: &LinForm, vals: &[Fp]) -> Fp {
    let mut acc = l.konst;
    for t in &l.terms { acc += t.coef * vals[t.idx as usize]; }
    acc
}

/// One GKR layer: output g = (Σ a_g) · (Σ b_g) over the layer's input index space (size 2^s_in).
pub struct Gate { pub a: Vec<Term>, pub b: Vec<Term> }
pub struct Layer { pub s_out: usize, pub s_in: usize, pub gates: Vec<Gate>, pub name: String }

fn log2ceil(n: usize) -> usize { let mut s = 0; while (1usize << s) < n { s += 1; } s }

fn lin_terms(l: &LinForm, one: u32, remap: &dyn Fn(u32) -> u32) -> Vec<Term> {
    let mut v: Vec<Term> = l.terms.iter().map(|t| Term { idx: remap(t.idx), coef: t.coef }).collect();
    if l.konst != Fp::ZERO { v.push(Term { idx: one, coef: l.konst }); }
    v
}

/// Layers from the input columns (layer index max_depth+1, the leaves) up to the assertions (layer 0).
/// V^(k) for k>=1 is indexed by wire id and holds every wire of depth <= k (deeper wires are 0); V^(0) = columns.
pub fn layers(c: &Circuit) -> Vec<Layer> {
    let nw = c.nwires();
    let s_w = log2ceil(nw);
    let s_c = log2ceil(c.ncols);
    let one_w = c.one_wire();
    let mut out = Vec::new();
    // assertion layer: over V^(max_depth)
    let s_a = log2ceil(c.asserts.len().max(1));
    let gates = c.asserts.iter().map(|(_, l)| Gate { a: lin_terms(l, one_w, &|i| i), b: vec![Term { idx: one_w, coef: Fp::ONE }] }).collect();
    out.push(Layer { s_out: s_a, s_in: s_w, gates, name: "assertions".into() });
    for k in (1..=c.max_depth).rev() {
        // V^(k) from V^(k-1)
        let from_cols = k == 1;
        let s_in = if from_cols { s_c } else { s_w };
        let remap = |i: u32| -> u32 { if from_cols { match c.wires[i as usize] { Wire::Input(col) => col, _ => panic!("depth-1 product reads a product") } } else { i } };
        let one_in = if from_cols { c.one_col } else { one_w };
        let gates = c.wires.iter().enumerate().map(|(w, wire)| match wire {
            Wire::Input(col) => Gate { a: vec![Term { idx: if from_cols { *col } else { w as u32 }, coef: Fp::ONE }], b: vec![Term { idx: one_in, coef: Fp::ONE }] },
            Wire::Prod { depth, a, b } => {
                if *depth == k { Gate { a: lin_terms(a, one_in, &remap), b: lin_terms(b, one_in, &remap) } }
                else if *depth < k { Gate { a: vec![Term { idx: w as u32, coef: Fp::ONE }], b: vec![Term { idx: one_in, coef: Fp::ONE }] } }
                else { Gate { a: vec![], b: vec![] } }
            }
        }).collect();
        out.push(Layer { s_out: s_w, s_in, gates, name: format!("depth{k}") });
    }
    out
}

/// Flat LE u64 witness: n_units, n_cols, then rows.
pub fn read_witness(bytes: &[u8]) -> (usize, usize, Vec<Fp>) {
    let rd = |i: usize| u64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap());
    let n = rd(0) as usize;
    let m = rd(1) as usize;
    let mut v = Vec::with_capacity(n * m);
    for i in 0..n * m { v.push(Fp::new(rd(2 + i))); }
    (n, m, v)
}
