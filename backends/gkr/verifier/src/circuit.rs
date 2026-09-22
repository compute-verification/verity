//! The exported layered circuit as the GPU prover reads it (`gpu/circuit.py`): the text format of
//! `backends/gkr/src/circuit.rs` with coefficients reduced mod p, row-listed tables, and the implicit `R16` table
//! (65536 rows) appended last.  Layers follow `gpu/circuit.py::layers` (= the CPU crate's `layers()`).

use crate::field::{add, mul, P, P64};
use crate::sha256::Sha256;
use std::collections::HashMap;

pub const R16_ROWS: usize = 65536;

#[derive(Clone, Debug)]
pub struct Term { pub idx: u32, pub coef: u32 }

#[derive(Clone, Debug, Default)]
pub struct LinForm { pub terms: Vec<Term>, pub konst: u32 }

#[derive(Clone, Debug)]
pub enum Wire { Input(u32), Prod { depth: u32, a: LinForm, b: LinForm } }

#[derive(Clone, Debug)]
pub struct Table { pub name: String, pub cols: usize, pub rows: Vec<Vec<u32>> }

#[derive(Clone, Debug)]
pub struct Query { pub table: String, pub cols: Vec<LinForm> }

#[derive(Debug)]
pub struct Circuit {
    pub ncols: usize,
    pub col_names: Vec<String>,
    pub one_col: u32,
    pub wires: Vec<Wire>,
    pub asserts: Vec<(String, LinForm)>,
    pub tables: Vec<Table>,
    pub queries: Vec<Query>,
    pub max_depth: u32,
    /// SHA-256 of the circuit text (absorbed by the transcript preamble).
    pub hash: [u8; 32],
}

fn fe(s: &str) -> u32 {
    let v: i128 = s.parse().expect("integer coefficient");
    (v.rem_euclid(P64 as i128)) as u32
}

fn parse_lin(tok: &mut std::slice::Iter<'_, &str>) -> LinForm {
    let n: usize = tok.next().unwrap().parse().unwrap();
    let mut terms = Vec::with_capacity(n);
    for _ in 0..n {
        let idx: u32 = tok.next().unwrap().parse().unwrap();
        let coef = fe(tok.next().unwrap());
        terms.push(Term { idx, coef });
    }
    let konst = fe(tok.next().unwrap());
    LinForm { terms, konst }
}

impl Circuit {
    pub fn parse(text: &str) -> Circuit {
        let mut c = Circuit { ncols: 0, col_names: vec![], one_col: 0, wires: vec![], asserts: vec![], tables: vec![], queries: vec![], max_depth: 0, hash: Sha256::digest(text.as_bytes()) };
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
                    c.tables.push(Table { name, cols, rows: vec![] });
                }
                "row" => { let t = c.tables.last_mut().unwrap(); t.rows.push(it.map(|s| fe(s)).collect()); }
                "query" => {
                    let table = it.next().unwrap().to_string();
                    let n: usize = it.next().unwrap().parse().unwrap();
                    let cols = (0..n).map(|_| parse_lin(&mut it)).collect();
                    c.queries.push(Query { table, cols });
                }
                other => panic!("unknown line kind {other}"),
            }
        }
        let mut next_col = 0u32;
        for w in 0..nwires as u32 {
            if let Some(p) = prods.remove(&w) { c.wires.push(p); } else { c.wires.push(Wire::Input(next_col)); next_col += 1; }
        }
        assert_eq!(next_col as usize, c.ncols, "column/wire mismatch");
        // gpu/circuit.py appends the implicit 16-bit range table unconditionally
        c.tables.push(Table { name: "R16".into(), cols: 1, rows: (0..R16_ROWS as u32).map(|v| vec![v]).collect() });
        for t in &c.tables { assert!(t.rows.iter().all(|r| r.len() == t.cols), "{}: row width", t.name); }
        c
    }

    pub fn nwires(&self) -> usize { self.wires.len() }
    pub fn one_wire(&self) -> u32 { self.wires.iter().position(|w| matches!(w, Wire::Input(c) if *c == self.one_col)).unwrap() as u32 }
    pub fn col_index(&self, name: &str) -> Option<usize> { self.col_names.iter().position(|n| n == name) }
    pub fn table_queries(&self, name: &str) -> Vec<&Query> { self.queries.iter().filter(|q| q.table == name).collect() }
}

pub fn eval_lin(l: &LinForm, vals: &[u32]) -> u32 {
    let mut acc = l.konst;
    for t in &l.terms { acc = add(acc, mul(t.coef, vals[t.idx as usize])); }
    acc
}

/// One GKR layer: output `g = (sum a_g) * (sum b_g)` over the layer's input index space (size `2^s_in`).
pub struct Gate { pub a: Vec<Term>, pub b: Vec<Term> }
pub struct Layer { pub s_out: usize, pub s_in: usize, pub gates: Vec<Gate>, pub name: String }

pub fn log2ceil(n: usize) -> usize { let mut s = 0; while (1usize << s) < n { s += 1; } s }

fn lin_terms(l: &LinForm, one: u32, remap: &dyn Fn(u32) -> u32) -> Vec<Term> {
    let mut v: Vec<Term> = l.terms.iter().map(|t| Term { idx: remap(t.idx), coef: t.coef }).collect();
    if l.konst != 0 { v.push(Term { idx: one, coef: l.konst }); }
    v
}

/// Layer 0 = assertions over `V^(max_depth)`, then `depth k` layers computing `V^(k)` from `V^(k-1)`, the last
/// reading the columns.
pub fn layers(c: &Circuit) -> Vec<Layer> {
    let nw = c.nwires();
    let s_w = log2ceil(nw);
    let s_c = log2ceil(c.ncols);
    let one_w = c.one_wire();
    let mut out = Vec::new();
    let s_a = log2ceil(c.asserts.len().max(1));
    let gates = c.asserts.iter().map(|(_, l)| Gate { a: lin_terms(l, one_w, &|i| i), b: vec![Term { idx: one_w, coef: 1 }] }).collect();
    out.push(Layer { s_out: s_a, s_in: s_w, gates, name: "assertions".into() });
    for k in (1..=c.max_depth).rev() {
        let from_cols = k == 1;
        let s_in = if from_cols { s_c } else { s_w };
        let remap = |i: u32| -> u32 { if from_cols { match c.wires[i as usize] { Wire::Input(col) => col, _ => panic!("depth-1 product reads a product") } } else { i } };
        let one_in = if from_cols { c.one_col } else { one_w };
        let gates = c.wires.iter().enumerate().map(|(w, wire)| match wire {
            Wire::Input(col) => Gate { a: vec![Term { idx: if from_cols { *col } else { w as u32 }, coef: 1 }], b: vec![Term { idx: one_in, coef: 1 }] },
            Wire::Prod { depth, a, b } => {
                if *depth == k { Gate { a: lin_terms(a, one_in, &remap), b: lin_terms(b, one_in, &remap) } }
                else if *depth < k { Gate { a: vec![Term { idx: w as u32, coef: 1 }], b: vec![Term { idx: one_in, coef: 1 }] } }
                else { Gate { a: vec![], b: vec![] } }
            }
        }).collect();
        out.push(Layer { s_out: s_w, s_in, gates, name: format!("depth{k}") });
    }
    out
}

/// `<QQ>` header then `n x m` LE u32 (`gpu/bb_export.py::write_rows`); values must be canonical.
pub fn read_rows(bytes: &[u8]) -> Result<(usize, usize, Vec<u32>), String> {
    if bytes.len() < 16 { return Err("rows file: short header".into()); }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let m = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    if bytes.len() != 16 + 4 * n * m { return Err(format!("rows file: {} bytes for {n} x {m}", bytes.len())); }
    let mut v = Vec::with_capacity(n * m);
    for i in 0..n * m {
        let x = u32::from_le_bytes(bytes[16 + 4 * i..20 + 4 * i].try_into().unwrap());
        if x >= P { return Err(format!("rows file: non-canonical value at {i}")); }
        v.push(x);
    }
    Ok((n, m, v))
}
