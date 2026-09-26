//! A-GKR: prover + verifier for the WP2 transition unit (data-parallel over B units) and for the full
//! K=1536 Verity VU (PROTOCOL.md 12: B·96 data-parallel units + B epilogue rows over one commitment, chain
//! linkage and public output words as linear constraints on the committed vector).
//!
//!   verity-gkr run --circuit C --witness W --units B [--clear] [--out result.json]       transition units
//!   verity-gkr run-vu --dir D [--vus B] [--clear] [--out result.json]                     VUs (D from gkr_export.vu)
//!   verity-gkr negatives --circuit C --dir DIR                                            every unit witness rejected
//!   verity-gkr negatives-vu --dir D                                                       every D/neg/<id> rejected
//!   verity-gkr check-witness --circuit C --witness W                                      constraint check only (no proof)
//!   verity-gkr dump-transcript --dir D [--vus B] --out T.json                              prover transcript vectors
//!   every command takes `--logup union` (one batched logUp over the tagged table union, PROTOCOL.md 15) or
//!   `--logup per-table` (default; env VERITY_GKR_LOGUP).
//!
//! Field: Goldilocks + cubic extension by default; `--features babybear` builds the same prover over BabyBear with
//! the degree-6 extension (PROTOCOL.md 13; `src/field_babybear.rs`), reading the `fixtures/babybear/` exports.
//!
//! Default mode (PCS): the input/hint columns and the lookup multiplicities are committed with the hash-based
//! Ligero-style commitment of `ligero.rs` (no masking) → NON_ZK_PROOF_DIAGNOSTIC.  `--clear`: the columns are given
//! to the verifier in the clear and only the GKR + logUp part is proved → ARITHMETIC_DIAGNOSTIC.
//! authentication = excluded in both.

mod circuit;
#[cfg(not(feature = "babybear"))]
mod field;
#[cfg(feature = "babybear")]
#[path = "field_babybear.rs"]
mod field;
mod gkr;
mod ligero;
mod logup;
mod transcript;

#[cfg(test)]
use circuit::materialise;
use circuit::{layers, read_witness, Circuit, Layer, Query, Table};
use field::{eq_table, Ext, Fp};
use gkr::{eval_columns, threads, LayerClaim, Reader, Strided};
use logup::{LeafClaim, SegQueries, TableChallenges, UnionChallenges, UnionSeg};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use transcript::{Event, Sha256, Transcript};

/// `dump-transcript`: record every transcript event of the next proof (read by `start_transcript`).
static LOG_TRANSCRIPT: AtomicBool = AtomicBool::new(false);

/// One data-parallel circuit over `units` rows of `ncols` committed columns.
pub struct Segment<'a> {
    pub name: String,
    pub circ: &'a Circuit,
    pub layers: &'a [Layer],
    pub cols: Vec<Fp>,
    pub units: usize,
    pub ncols: usize,
    pub hash: [u8; 32],
}

/// One linked quantity of the VU chain (PROTOCOL.md 12.2 / 14.3).  With `col_c = Some(c)`: `X[(v,0),c] = init`,
/// `X[(v,i),c] = X[(v,i-1),y]` for `0 < i < steps`, `X_epi[v,epi] = X[(v,steps-1),y]`.  With `col_c = None` the
/// quantity is not an input of the next unit: `X_epi[v,epi] = X[(v,steps-1),y]` and, if `zero_nonlast`,
/// `X[(v,i),y] = 0` for every `i < steps-1` (v2's saturation flags: the domain excludes a saturated intermediate).
#[derive(Clone, Debug)]
pub struct Link {
    pub col_c: Option<usize>,
    pub col_y: usize,
    pub epi: usize,
    pub init: Fp,
    pub zero_nonlast: bool,
}

impl Link {
    /// A word linked as one column (v1 Goldilocks) or as one limb (v1 BabyBear): `[x.c, y, y32]`, init 0.
    pub fn word(cols: [usize; 3]) -> Link {
        Link {
            col_c: Some(cols[0]),
            col_y: cols[1],
            epi: cols[2],
            init: Fp::ZERO,
            zero_nonlast: false,
        }
    }
    pub fn constraints_per_vu(&self, steps: usize) -> usize {
        if self.col_c.is_some() {
            steps + 1
        } else {
            1 + if self.zero_nonlast { steps - 1 } else { 0 }
        }
    }
}

/// The VU chain over segment 0 (units, VU-major: row v·steps+i) and segment 1 (epilogue, row v): the links, the
/// epilogue's output-word column and the public words.
#[derive(Clone)]
pub struct Chain {
    pub steps: usize,
    pub limbs: Vec<Link>,
    pub epi_y16: usize,
    pub public_y16: Vec<Fp>,
}

impl Chain {
    pub fn constraints_per_vu(&self) -> usize {
        1 + self
            .limbs
            .iter()
            .map(|l| l.constraints_per_vu(self.steps))
            .sum::<usize>()
    }
}

/// The lookup tables of an instance: the union, by name, of every segment's tables (`Instance::new`).  Until
/// 2026-09-22 this was segment 0's table list alone, so a table queried only by the epilogue circuit (v2: `R16`,
/// `R15`) had no logUp instance and its range checks were never proved — a malicious epilogue hint forged any public
/// word (`v2_forged_public_word_via_epilogue_hint_is_rejected`; `fixtures/redteam-3/logup-union/`).
pub struct Instance<'a> {
    pub segs: Vec<Segment<'a>>,
    pub chain: Option<Chain>,
    pub tables: Vec<&'a Table>,
    pub lookup: Lookup,
}

/// The lookup argument: one logUp fractional-GKR instance per table (PROTOCOL.md 4.2, the default) or ONE batched
/// instance over the tagged union of every table (PROTOCOL.md 15, `--logup union`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lookup {
    PerTable,
    Union,
}

impl Lookup {
    pub fn name(self) -> &'static str {
        match self {
            Lookup::PerTable => "per_table",
            Lookup::Union => "union",
        }
    }
    pub fn parse(s: &str) -> Lookup {
        match s {
            "union" => Lookup::Union,
            "per-table" | "per_table" => Lookup::PerTable,
            other => panic!("--logup takes union | per-table, got {other}"),
        }
    }
}

impl<'a> Instance<'a> {
    pub fn with_lookup(mut self, lookup: Lookup) -> Instance<'a> {
        self.lookup = lookup;
        self
    }
    pub fn new(segs: Vec<Segment<'a>>, chain: Option<Chain>) -> Instance<'a> {
        let mut tables: Vec<&'a Table> = Vec::new();
        for s in &segs {
            for t in &s.circ.tables {
                match tables.iter().find(|u| u.name == t.name) {
                    Some(u) => assert!(
                        u.cols == t.cols && u.rows == t.rows,
                        "table {} differs between segments",
                        t.name
                    ),
                    None => tables.push(t),
                }
            }
        }
        Instance {
            segs,
            chain,
            tables,
            lookup: Lookup::PerTable,
        }
    }
    /// Every segment's queries with their table index (union mode's witness side).
    pub fn union_segs(&self) -> Vec<UnionSeg<'_>> {
        self.segs
            .iter()
            .map(|s| UnionSeg {
                queries: s
                    .circ
                    .queries
                    .iter()
                    .map(|q| {
                        (
                            self.tables
                                .iter()
                                .position(|t| t.name == q.table)
                                .expect("query table"),
                            q,
                        )
                    })
                    .collect(),
                cols: &s.cols,
                ncols: s.ncols,
                units: s.units,
            })
            .collect()
    }
    pub fn tables(&self) -> &[&'a Table] {
        &self.tables
    }
    pub fn vus(&self) -> usize {
        self.chain
            .as_ref()
            .map(|c| self.segs[0].units / c.steps)
            .unwrap_or(0)
    }
}

fn log2ceil(n: usize) -> usize {
    let mut s = 0;
    while (1usize << s) < n {
        s += 1;
    }
    s
}

#[derive(Default, Debug)]
pub struct Stats {
    pub t_lookup: f64,
    pub t_arith: f64,
    pub t_witness_wires: f64,
    pub proof_elems: usize,
    pub slots: u64,
    pub depth_circuit: u64,
    pub depth_lookup_max: u64,
    pub lookup_dims: Vec<(String, usize, usize)>,
    pub committed_elements: usize,
    pub t_commit: f64,
    pub t_open: f64,
    pub ligero_rows: usize,
    pub linkage_constraints: usize,
    pub transcript_log: Option<Vec<Event>>,
    /// Measured critical-path verifier messages (`rounds.sequential_depth`), the emulated-network sleep (s), and
    /// the two chains' depths (closed forms checked against `trips`).
    pub trips: u64,
    pub slept: f64,
    pub depth_lookup_chain: u64,
    pub depth_checker_chain: u64,
}

fn table_queries<'a>(circ: &'a Circuit, name: &str) -> Vec<&'a Query> {
    circ.queries.iter().filter(|q| q.table == name).collect()
}

fn seg_queries<'a>(inst: &'a Instance, name: &str) -> Vec<SegQueries<'a>> {
    inst.segs
        .iter()
        .map(|s| SegQueries {
            queries: table_queries(s.circ, name),
            cols: &s.cols,
            ncols: s.ncols,
            units: s.units,
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Clear,
    Pcs,
}

pub struct Proof {
    pub root: [u8; 32],
    pub msgs: Vec<Ext>,
    pub bytes: Vec<u8>,
}
impl Proof {
    pub fn size(&self) -> usize {
        (if self.root != [0u8; 32] { 32 } else { 0 })
            + self.msgs.len() * Ext::BYTES
            + self.bytes.len()
    }
}

pub const LIGERO: ligero::Params = ligero::Params {
    k: 4096,
    n: 16384,
    t: 192,
};

/// The lookup challenges of one proof: per table (z_T, beta_T), or the union's single (z, beta, gamma).
pub struct Challenges {
    pub per_table: Vec<TableChallenges>,
    pub union: Option<UnionChallenges>,
}

/// Transcript preamble: every public datum (circuits, batch shape, lookup mode, commitment root or the clear
/// witness, the public output words), then the table challenges and every segment's output-layer point.
fn start_transcript(
    inst: &Instance,
    mode: Mode,
    root: &[u8; 32],
) -> (Transcript, Challenges, Vec<(Vec<Ext>, Vec<Ext>)>) {
    let mut tr = if LOG_TRANSCRIPT.load(Ordering::Relaxed) {
        Transcript::with_log(field::DOMAIN)
    } else {
        Transcript::new(field::DOMAIN)
    };
    tr.label("preamble: per segment, sha256(circuit text) then units as u64 LE");
    for s in &inst.segs {
        tr.absorb(&s.hash);
        tr.absorb(&(s.units as u64).to_le_bytes());
    }
    if inst.lookup == Lookup::Union {
        // the lookup mode is part of the statement (a per-table transcript must not verify as a union one)
        tr.label("preamble: lookup argument = union (PROTOCOL.md 15), table order as tagged");
        tr.absorb(b"logup-union-v1");
        for (ti, t) in inst.tables().iter().enumerate() {
            tr.absorb(&(ti as u64 + 1).to_le_bytes());
            tr.absorb(t.name.as_bytes());
        }
    }
    match mode {
        Mode::Clear => {
            // the witness is public in this diagnostic: bind the challenges to it
            let mut h = Sha256::new();
            for s in &inst.segs {
                for v in &s.cols {
                    h.update(&v.to_bytes());
                }
            }
            tr.label("preamble: sha256 of every committed column value (clear mode)");
            tr.absorb(&h.finalize());
        }
        Mode::Pcs => {
            tr.label("preamble: Ligero Merkle root");
            tr.absorb(root)
        }
    }
    if let Some(ch) = &inst.chain {
        tr.label("preamble: chain steps (u64 LE), then the public y16 words (base field elements, concatenated)");
        tr.absorb(&(ch.steps as u64).to_le_bytes());
        let mut buf = Vec::with_capacity(Fp::BYTES * ch.public_y16.len());
        for y in &ch.public_y16 {
            buf.extend_from_slice(&y.to_bytes());
        }
        tr.absorb(&buf);
    }
    let mut chs = Challenges {
        per_table: Vec::new(),
        union: None,
    };
    match inst.lookup {
        Lookup::PerTable => {
            for t in inst.tables() {
                tr.label(&format!(
                    "table {}: z{}",
                    t.name,
                    if t.cols > 1 {
                        ", beta"
                    } else {
                        " (beta = 1, single column)"
                    }
                ));
                let z = tr.challenge();
                let beta = if t.cols > 1 { tr.challenge() } else { Ext::ONE };
                chs.per_table.push(TableChallenges { z, beta });
            }
        }
        Lookup::Union => {
            // gamma is drawn here, AFTER the multiplicities are committed (they are part of X, hence of the root)
            tr.label(&format!(
                "union logup over {} tables: z, beta, gamma (tag(t) = 1 + table index)",
                inst.tables().len()
            ));
            let z = tr.challenge();
            let beta = tr.challenge();
            let gamma = tr.challenge();
            chs.union = Some(UnionChallenges { z, beta, gamma });
        }
    }
    let mut pts = Vec::new();
    for s in &inst.segs {
        tr.label(&format!(
            "segment {}: output point z0 ({} coordinates), copy point gamma ({} coordinates)",
            s.name,
            s.layers[0].s_out,
            log2ceil(s.units)
        ));
        let z0 = tr.challenges(s.layers[0].s_out);
        let gamma = tr.challenges(log2ceil(s.units));
        pts.push((z0, gamma));
    }
    (tr, chs, pts)
}

/// Layout of the committed vector X: every segment's rows (row-major), then every table's multiplicities.
fn x_layout(inst: &Instance) -> (Vec<usize>, Vec<usize>, usize) {
    let mut seg_offs = Vec::new();
    let mut l = 0;
    for s in &inst.segs {
        seg_offs.push(l);
        l += s.units * s.ncols;
    }
    let mut offs = Vec::new();
    for t in inst.tables() {
        offs.push(l);
        l += t.rows.len();
    }
    (seg_offs, offs, l)
}

/// The batched linear functional ⟨a, X⟩ = b the Ligero opening proves.  Every residual claim is added with its
/// batching coefficient ρ (drawn once all claims are fixed); the per-row work is parallel over units.
struct Acc {
    a: Vec<Ext>,
    b: Ext,
}

fn par_rows<F: Fn(usize, &mut [Ext]) -> Ext + Sync>(
    a: &mut [Ext],
    ncols: usize,
    units: usize,
    f: F,
) -> Ext {
    let nthreads = threads().min(units.max(1));
    let chunk = units.div_ceil(nthreads).max(1);
    let f = &f;
    let parts: Vec<Ext> = std::thread::scope(|s| {
        let hs: Vec<_> = a[..units * ncols]
            .chunks_mut(chunk * ncols)
            .enumerate()
            .map(|(ci, ac)| {
                s.spawn(move || {
                    let mut k = Ext::ZERO;
                    for (i, row) in ac.chunks_mut(ncols).enumerate() {
                        k += f(ci * chunk + i, row);
                    }
                    k
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut k = Ext::ZERO;
    for p in parts {
        k += p;
    }
    k
}

/// Ṽ_in(x*, c*) = value on a segment's committed columns.
fn add_input_claim(
    acc: &mut Acc,
    rho: Ext,
    seg_off: usize,
    seg: &Segment,
    x_pt: &[Ext],
    c_pt: &[Ext],
    value: Ext,
) {
    let ec = eq_table(c_pt);
    let ex = eq_table(x_pt);
    par_rows(&mut acc.a[seg_off..], seg.ncols, seg.units, |u, row| {
        let e = rho * ec[u];
        for (i, x) in row.iter_mut().enumerate() {
            *x += e * ex[i];
        }
        Ext::ZERO
    });
    acc.b += rho * value;
}

/// p̃(ρ) + λ q̃(ρ) = value as a functional on X (table rows, padding and the linear-form constants are public).
fn add_lookup_claim(
    acc: &mut Acc,
    rho: Ext,
    inst: &Instance,
    seg_offs: &[usize],
    table_off: usize,
    table: &Table,
    segqs: &[SegQueries],
    ch: &TableChallenges,
    n: usize,
    c: &LeafClaim,
) {
    let eqt = eq_table(&c.rho);
    let bs = logup::betas(table, ch);
    let mut konst = Ext::ZERO;
    let mut base = 0usize;
    for (si, sq) in segqs.iter().enumerate() {
        let q_t = sq.queries.len();
        if q_t == 0 {
            continue;
        }
        let seg = &inst.segs[si];
        let (eqt, bs, queries) = (&eqt, &bs, &sq.queries);
        konst += par_rows(
            &mut acc.a[seg_offs[si]..],
            seg.ncols,
            seg.units,
            |u, row| {
                let mut k = Ext::ZERO;
                for (j, qu) in queries.iter().enumerate() {
                    let e = eqt[base + u * q_t + j];
                    let le = c.lambda * e;
                    k += e; // p = 1
                    let mut cst = ch.z;
                    for (kk, lin) in qu.cols.iter().enumerate() {
                        cst -= bs[kk].scale(lin.konst);
                        let lb = rho * le * bs[kk];
                        for t in &lin.terms {
                            row[t.idx as usize] -= lb.scale(t.coef);
                        }
                    }
                    k += le * cst;
                }
                k
            },
        );
        base += sq.n_wit();
    }
    for (v, r) in table.rows.iter().enumerate() {
        let e = eqt[base + v];
        acc.a[table_off + v] -= rho * e; // p = -m_v
        let mut comb = Ext::ZERO;
        for (k, x) in r.iter().enumerate() {
            comb += bs[k].scale(*x);
        }
        konst += c.lambda * e * (ch.z - comb);
    }
    for idx in base + table.rows.len()..(1usize << n) {
        konst += c.lambda * eqt[idx];
    } // padding (0, 1)
    acc.b += rho * (c.value - konst);
}

/// The union instance's leaf claim as a functional on X (PROTOCOL.md 15.4): the same as `add_lookup_claim` with
/// every table's queries in one leaf vector, the tag constant γ·tag(t) in every denominator, one β for all tables,
/// and the table side running over every multiplicity block in table order.
fn add_union_claim(
    acc: &mut Acc,
    rho: Ext,
    inst: &Instance,
    seg_offs: &[usize],
    offs: &[usize],
    usegs: &[UnionSeg],
    ch: &UnionChallenges,
    n: usize,
    c: &LeafClaim,
) {
    let eqt = eq_table(&c.rho);
    let tables = inst.tables();
    let bs = logup::union_betas(tables, ch.beta);
    let tags: Vec<Ext> = (0..tables.len())
        .map(|ti| ch.gamma * logup::union_tag(ti))
        .collect();
    let mut konst = Ext::ZERO;
    let mut base = 0usize;
    for (si, sq) in usegs.iter().enumerate() {
        let q_t = sq.queries.len();
        if q_t == 0 {
            continue;
        }
        let seg = &inst.segs[si];
        let (eqt, bs, tags, queries) = (&eqt, &bs, &tags, &sq.queries);
        konst += par_rows(
            &mut acc.a[seg_offs[si]..],
            seg.ncols,
            seg.units,
            |u, row| {
                let mut k = Ext::ZERO;
                for (j, (ti, qu)) in queries.iter().enumerate() {
                    let e = eqt[base + u * q_t + j];
                    let le = c.lambda * e;
                    k += e; // p = 1
                    let mut cst = ch.z - tags[*ti];
                    for (kk, lin) in qu.cols.iter().enumerate() {
                        cst -= bs[kk].scale(lin.konst);
                        let lb = rho * le * bs[kk];
                        for t in &lin.terms {
                            row[t.idx as usize] -= lb.scale(t.coef);
                        }
                    }
                    k += le * cst;
                }
                k
            },
        );
        base += sq.n_wit();
    }
    for (ti, table) in tables.iter().enumerate() {
        for (v, r) in table.rows.iter().enumerate() {
            let e = eqt[base + v];
            acc.a[offs[ti] + v] -= rho * e; // p = -m_v
            let mut comb = tags[ti];
            for (k, x) in r.iter().enumerate() {
                comb += bs[k].scale(*x);
            }
            konst += c.lambda * e * (ch.z - comb);
        }
        base += table.rows.len();
    }
    for idx in base..(1usize << n) {
        konst += c.lambda * eqt[idx];
    } // padding (0, 1)
    acc.b += rho * (c.value - konst);
}

/// Chain linkage and public endpoints as one functional with geometric coefficients ρ_geo^k (PROTOCOL.md 12.2):
/// c_{v,0} = 0; c_{v,i} = y_{v,i-1}; y32_v = y_{v,steps-1}; y16_v = the public word.  Returns the constraint count.
fn add_chain(acc: &mut Acc, rho: Ext, geo: Ext, inst: &Instance, seg_offs: &[usize]) -> usize {
    let ch = inst.chain.as_ref().unwrap();
    let (u_seg, e_seg) = (&inst.segs[0], &inst.segs[1]);
    let vus = u_seg.units / ch.steps;
    let mut pw = rho;
    let mut count = 0;
    for v in 0..vus {
        for lk in &ch.limbs {
            let (col_y, epi) = (lk.col_y, lk.epi);
            if let Some(col_c) = lk.col_c {
                for i in 0..ch.steps {
                    let u = v * ch.steps + i;
                    acc.a[seg_offs[0] + u * u_seg.ncols + col_c] += pw;
                    if i > 0 {
                        acc.a[seg_offs[0] + (u - 1) * u_seg.ncols + col_y] -= pw;
                    } else {
                        acc.b += pw.scale(lk.init);
                    }
                    pw = pw * geo;
                    count += 1;
                }
            } else if lk.zero_nonlast {
                for i in 0..ch.steps - 1 {
                    acc.a[seg_offs[0] + (v * ch.steps + i) * u_seg.ncols + col_y] += pw;
                    pw = pw * geo;
                    count += 1;
                }
            }
            acc.a[seg_offs[1] + v * e_seg.ncols + epi] += pw;
            acc.a[seg_offs[0] + (v * ch.steps + ch.steps - 1) * u_seg.ncols + col_y] -= pw;
            pw = pw * geo;
            count += 1;
        }
        acc.a[seg_offs[1] + v * e_seg.ncols + ch.epi_y16] += pw;
        acc.b += pw.scale(ch.public_y16[v]);
        pw = pw * geo;
        count += 1;
    }
    count
}

/// The chain on clear columns (ARITHMETIC_DIAGNOSTIC verifier).
fn check_chain_clear(inst: &Instance) -> Result<(), String> {
    let ch = inst.chain.as_ref().unwrap();
    let (u_seg, e_seg) = (&inst.segs[0], &inst.segs[1]);
    let vus = u_seg.units / ch.steps;
    let uc = |u: usize, c: usize| u_seg.cols[u * u_seg.ncols + c];
    for v in 0..vus {
        for lk in &ch.limbs {
            let (col_y, epi) = (lk.col_y, lk.epi);
            if let Some(col_c) = lk.col_c {
                for i in 0..ch.steps {
                    let u = v * ch.steps + i;
                    let want = if i == 0 { lk.init } else { uc(u - 1, col_y) };
                    if uc(u, col_c) != want {
                        return Err(format!("chain: VU {v} unit {i} incoming value differs from the previous outgoing value (column {col_c})"));
                    }
                }
            } else if lk.zero_nonlast {
                for i in 0..ch.steps - 1 {
                    if uc(v * ch.steps + i, col_y) != Fp::ZERO {
                        return Err(format!(
                            "chain: VU {v} unit {i}: column {col_y} must be 0 before the last unit"
                        ));
                    }
                }
            }
            if e_seg.cols[v * e_seg.ncols + epi] != uc(v * ch.steps + ch.steps - 1, col_y) {
                return Err(format!("chain: VU {v} epilogue input (column {epi}) differs from the last unit's value"));
            }
        }
        if e_seg.cols[v * e_seg.ncols + ch.epi_y16] != ch.public_y16[v] {
            return Err(format!(
                "chain: VU {v} output word differs from the public word"
            ));
        }
    }
    Ok(())
}

fn n_lookup_instances(inst: &Instance) -> usize {
    match inst.lookup {
        Lookup::PerTable => inst.tables().len(),
        Lookup::Union => 1,
    }
}
fn n_functionals(inst: &Instance) -> usize {
    2 * inst.segs.len() + n_lookup_instances(inst) + if inst.chain.is_some() { 1 } else { 0 }
}

/// Every wire of every unit of a segment (parallel over units).
fn eval_wires(seg: &Segment) -> Vec<Fp> {
    let nw = seg.circ.nwires();
    let mut wires = vec![Fp::ZERO; seg.units * nw];
    let nthreads = threads().min(seg.units.max(1));
    let chunk = seg.units.div_ceil(nthreads).max(1);
    std::thread::scope(|s| {
        for (ci, wc) in wires.chunks_mut(chunk * nw).enumerate() {
            s.spawn(move || {
                let mut tmp = Vec::with_capacity(nw);
                for (k, w) in wc.chunks_mut(nw).enumerate() {
                    let u = ci * chunk + k;
                    seg.circ
                        .eval_wires(&seg.cols[u * seg.ncols..(u + 1) * seg.ncols], &mut tmp);
                    w.copy_from_slice(&tmp);
                }
            });
        }
    });
    wires
}

/// The layer-k view of the wires: every wire of depth > k zeroed (parallel).
fn masked_wires(wires: &[Fp], depth_of: &[u32], k_in: u32) -> Vec<Fp> {
    let mut m = wires.to_vec();
    let nw = depth_of.len();
    let nthreads = threads();
    let chunk = (m.len() / nw).div_ceil(nthreads).max(1) * nw;
    std::thread::scope(|s| {
        for mc in m.chunks_mut(chunk) {
            s.spawn(move || {
                for (j, v) in mc.iter_mut().enumerate() {
                    if depth_of[j % nw] > k_in {
                        *v = Fp::ZERO;
                    }
                }
            });
        }
    });
    m
}

type FinalClaims = (Vec<Ext>, Ext, Vec<Ext>, Ext);

/// GKR over one segment from its output-layer claim down to the input columns; returns the two input claims
/// and the copy point, and the segment's sequential depth.
fn prove_segment(
    seg: &Segment,
    z0: &[Ext],
    gamma: &[Ext],
    tr: &mut Transcript,
    out: &mut Vec<Ext>,
    st: &mut Stats,
) -> (FinalClaims, Vec<Ext>) {
    let t1 = Instant::now();
    let wires = eval_wires(seg);
    st.t_witness_wires += t1.elapsed().as_secs_f64();
    let b = log2ceil(seg.units);
    let nw = seg.circ.nwires();
    let mut claim = LayerClaim {
        w: eq_table(z0),
        copy_point: gamma.to_vec(),
        value: Ext::ZERO,
    };
    let nl = seg.layers.len();
    let depth_of: Vec<u32> = seg
        .circ
        .wires
        .iter()
        .map(|w| match w {
            circuit::Wire::Input(_) => 0,
            circuit::Wire::Prod { depth, .. } => *depth,
        })
        .collect();
    let mut final_claims = None;
    let mut depth = 0u64;
    for (i, layer) in seg.layers.iter().enumerate() {
        let last = i == nl - 1;
        let k_in = seg.circ.max_depth - i as u32;
        let masked: Vec<Fp> = if last {
            vec![]
        } else {
            masked_wires(&wires, &depth_of, k_in)
        };
        let input_wires = Strided {
            data: &masked,
            width: nw,
            copies: seg.units,
        };
        let input_cols = Strided {
            data: &seg.cols,
            width: seg.ncols,
            copies: seg.units,
        };
        let inp: &(dyn gkr::LayerInput + Sync) = if last { &input_cols } else { &input_wires };
        let (next, fin) = gkr::prove_layer(layer, inp, b, &claim, tr, out, last);
        depth += (b + 2 * layer.s_in + if last { 0 } else { 1 }) as u64;
        if let Some(f) = fin {
            final_claims = Some((f, next.copy_point.clone()));
        }
        claim = next;
    }
    st.depth_circuit = st.depth_circuit.max(depth);
    final_claims.unwrap()
}

/// Verifier messages on the critical path of the logUp chain of dimension `n`: lambda after q_root, then per level
/// k its k round challenges and one (mu, lambda) message.  (The earlier derived counter `1 + sum (k + 2)` counted mu
/// and lambda as two messages; the transcript's `trips` counter is the measured truth and this is its closed form.)
fn lookup_chain_depth(n: usize) -> u64 {
    1 + (0..n as u64).map(|k| k + 1).sum::<u64>()
}

/// Verifier messages on the critical path of one segment's checker GKR: per layer b phase-1 rounds, 2 s_in phase-2
/// rounds and alpha (none on the last layer).
fn segment_chain_depth(seg: &Segment) -> u64 {
    let b = log2ceil(seg.units) as u64;
    let nl = seg.layers.len();
    seg.layers
        .iter()
        .enumerate()
        .map(|(i, l)| b + 2 * l.s_in as u64 + if i == nl - 1 { 0 } else { 1 })
        .sum()
}

pub fn prove(inst: &Instance, mode: Mode) -> (Proof, Stats) {
    prove_with_rtt(inst, mode, None)
}

/// `prove` with the network emulated at round-trip time `rtt` (`--rtt-ms`): the prover sleeps one RTT at every
/// critical-path verifier message (`Transcript::round_trip`); `Stats::trips` counts them, `Stats::slept` the time.
pub fn prove_with_rtt(
    inst: &Instance,
    mode: Mode,
    rtt: Option<std::time::Duration>,
) -> (Proof, Stats) {
    let mut st = Stats::default();
    // the critical chain is known before any challenge: the deepest logUp instance and the deepest segment
    let usegs = inst.union_segs();
    let table_depths: Vec<u64> = match inst.lookup {
        Lookup::PerTable => inst
            .tables()
            .iter()
            .map(|t| lookup_chain_depth(logup::hypercube_dim(&seg_queries(inst, &t.name), t)))
            .collect(),
        Lookup::Union => vec![lookup_chain_depth(logup::union_dim(&usegs, inst.tables()))],
    };
    let seg_depths: Vec<u64> = inst.segs.iter().map(segment_chain_depth).collect();
    let critical_table = (0..table_depths.len()).max_by_key(|&i| table_depths[i]);
    let critical_seg = (0..seg_depths.len()).max_by_key(|&i| seg_depths[i]);
    let lookup_is_critical = table_depths.iter().max().copied().unwrap_or(0)
        >= seg_depths.iter().max().copied().unwrap_or(0);
    let mut out: Vec<Ext> = Vec::new();
    let mut bytes: Vec<u8> = Vec::new();
    let (seg_offs, offs, l) = x_layout(inst);
    // multiplicities are challenge-independent: count them first so X can be committed before any challenge
    let tc = Instant::now();
    let committed = if mode == Mode::Pcs {
        let mut x: Vec<Fp> = Vec::with_capacity(l);
        for s in &inst.segs {
            x.extend_from_slice(&s.cols);
        }
        match inst.lookup {
            Lookup::PerTable => {
                for t in inst.tables() {
                    let m = logup::multiplicities(t, &seg_queries(inst, &t.name));
                    x.extend(m.iter().map(|v| Fp::new(*v)));
                }
            }
            Lookup::Union => {
                for m in logup::union_multiplicities(inst.tables(), &usegs) {
                    x.extend(m.iter().map(|v| Fp::new(*v)));
                }
            }
        }
        debug_assert_eq!(x.len(), l);
        Some(ligero::commit(&x, &LIGERO))
    } else {
        None
    };
    let root = committed.as_ref().map(|c| c.root()).unwrap_or([0u8; 32]);
    st.t_commit = tc.elapsed().as_secs_f64();
    let (mut tr, chs, pts) = start_transcript(inst, mode, &root);
    tr.set_rtt(rtt);
    // message 2a (all z, beta, z_0, gamma) was drawn in start_transcript after the root: one round trip
    tr.critical = true;
    tr.round_trip();
    let t0 = Instant::now();
    let mut lookup_claims: Vec<(usize, LeafClaim)> = Vec::new();
    match inst.lookup {
        Lookup::PerTable => {
            for (ti, t) in inst.tables().iter().enumerate() {
                let sq = seg_queries(inst, &t.name);
                tr.critical = lookup_is_critical && Some(ti) == critical_table;
                tr.label(&format!("logup table {}", t.name));
                let (p, q, n, _) = logup::leaves(t, &sq, &chs.per_table[ti]);
                let (s, lc) = logup::prove_table(p, q, n, &mut tr, &mut out);
                st.lookup_dims.push((
                    t.name.clone(),
                    s.n,
                    sq.iter().map(|x| x.queries.len()).sum(),
                ));
                lookup_claims.push((n, lc));
                let d: u64 = 1 + (0..n as u64).map(|k| k + 2).sum::<u64>();
                st.depth_lookup_max = st.depth_lookup_max.max(d);
            }
        }
        Lookup::Union => {
            tr.critical = lookup_is_critical;
            tr.label("logup union instance");
            let (p, q, n, _) =
                logup::union_leaves(inst.tables(), &usegs, chs.union.as_ref().unwrap());
            let (s, lc) = logup::prove_table(p, q, n, &mut tr, &mut out);
            st.lookup_dims.push((
                "UNION".into(),
                s.n,
                usegs.iter().map(|x| x.queries.len()).sum(),
            ));
            lookup_claims.push((n, lc));
            st.depth_lookup_max = 1 + (0..n as u64).map(|k| k + 2).sum::<u64>();
        }
    }
    st.t_lookup = t0.elapsed().as_secs_f64();
    let t2 = Instant::now();
    let mut finals = Vec::new();
    for (si, seg) in inst.segs.iter().enumerate() {
        tr.critical = !lookup_is_critical && Some(si) == critical_seg;
        tr.label(&format!("gkr segment {}", seg.name));
        finals.push(prove_segment(
            seg, &pts[si].0, &pts[si].1, &mut tr, &mut out, &mut st,
        ));
    }
    st.t_arith = t2.elapsed().as_secs_f64() - st.t_witness_wires;
    tr.critical = true;
    if let Some(cm) = &committed {
        let t3 = Instant::now();
        tr.label(&format!(
            "opening: {} functional weights rho{}",
            n_functionals(inst),
            if inst.chain.is_some() {
                ", then geo (chain)"
            } else {
                ""
            }
        ));
        tr.round_trip(); // rho, geo and the Ligero r are one verifier message (no prover message between them)
        let rhos = tr.challenges(n_functionals(inst));
        let geo = if inst.chain.is_some() {
            tr.challenge()
        } else {
            Ext::ZERO
        };
        let mut acc = Acc {
            a: vec![Ext::ZERO; l],
            b: Ext::ZERO,
        };
        let mut ri = 0;
        for (si, seg) in inst.segs.iter().enumerate() {
            let ((l_star, vl, r_star, vr), c_star) = &finals[si];
            add_input_claim(&mut acc, rhos[ri], seg_offs[si], seg, l_star, c_star, *vl);
            add_input_claim(
                &mut acc,
                rhos[ri + 1],
                seg_offs[si],
                seg,
                r_star,
                c_star,
                *vr,
            );
            ri += 2;
        }
        match inst.lookup {
            Lookup::PerTable => {
                for (ti, t) in inst.tables().iter().enumerate() {
                    let (n, lc) = &lookup_claims[ti];
                    add_lookup_claim(
                        &mut acc,
                        rhos[ri],
                        inst,
                        &seg_offs,
                        offs[ti],
                        t,
                        &seg_queries(inst, &t.name),
                        &chs.per_table[ti],
                        *n,
                        lc,
                    );
                    ri += 1;
                }
            }
            Lookup::Union => {
                let (n, lc) = &lookup_claims[0];
                add_union_claim(
                    &mut acc,
                    rhos[ri],
                    inst,
                    &seg_offs,
                    &offs,
                    &usegs,
                    chs.union.as_ref().unwrap(),
                    *n,
                    lc,
                );
                ri += 1;
            }
        }
        if inst.chain.is_some() {
            st.linkage_constraints = add_chain(&mut acc, rhos[ri], geo, inst, &seg_offs);
        }
        ligero::prove_open(cm, acc.a, &LIGERO, &mut tr, &mut out, &mut bytes);
        st.t_open = t3.elapsed().as_secs_f64();
        st.ligero_rows = cm.rows;
    }
    st.proof_elems = out.len();
    st.slots = tr.slots;
    st.trips = tr.trips;
    st.slept = tr.slept.as_secs_f64();
    st.depth_lookup_chain = table_depths.iter().max().copied().unwrap_or(0);
    st.depth_checker_chain = seg_depths.iter().max().copied().unwrap_or(0);
    let expected = 1
        + st.depth_lookup_chain.max(st.depth_checker_chain)
        + if mode == Mode::Pcs { 2 } else { 0 };
    assert_eq!(
        st.trips, expected,
        "critical-path trips counted by the transcript disagree with the closed form"
    );
    st.committed_elements = l;
    st.transcript_log = tr.log.take();
    (
        Proof {
            root,
            msgs: out,
            bytes,
        },
        st,
    )
}

/// Verifier.  In `Mode::Clear` it reads the segments' `cols`; in `Mode::Pcs` it never touches them (they may be empty).
pub fn verify(inst: &Instance, proof: &Proof, mode: Mode) -> Result<(), String> {
    let (mut tr, chs, pts) = start_transcript(inst, mode, &proof.root);
    let mut rd = Reader {
        msgs: &proof.msgs,
        pos: 0,
    };
    let (seg_offs, offs, l) = x_layout(inst);
    let mut lookup_claims = Vec::new();
    let usegs = inst.union_segs();
    match inst.lookup {
        Lookup::PerTable => {
            for (ti, t) in inst.tables().iter().enumerate() {
                let sq = seg_queries(inst, &t.name);
                let n = logup::hypercube_dim(&sq, t);
                let lc = logup::verify_table(n, &mut tr, &mut rd, &t.name)?;
                match mode {
                    Mode::Clear => {
                        let (p, q, n2, _) = logup::leaves(t, &sq, &chs.per_table[ti]);
                        debug_assert_eq!(n, n2);
                        logup::check_leaves(&p, &q, &lc, &t.name)?;
                    }
                    Mode::Pcs => lookup_claims.push((n, lc)),
                }
            }
        }
        Lookup::Union => {
            let n = logup::union_dim(&usegs, inst.tables());
            let lc = logup::verify_table(n, &mut tr, &mut rd, "UNION")?;
            match mode {
                Mode::Clear => {
                    let (p, q, n2, _) =
                        logup::union_leaves(inst.tables(), &usegs, chs.union.as_ref().unwrap());
                    debug_assert_eq!(n, n2);
                    logup::check_leaves(&p, &q, &lc, "UNION")?;
                }
                Mode::Pcs => lookup_claims.push((n, lc)),
            }
        }
    }
    let mut finals = Vec::new();
    for (si, seg) in inst.segs.iter().enumerate() {
        let b = log2ceil(seg.units);
        let mut claim = LayerClaim {
            w: eq_table(&pts[si].0),
            copy_point: pts[si].1.clone(),
            value: Ext::ZERO,
        };
        let nl = seg.layers.len();
        let s_c = seg.layers[nl - 1].s_in;
        for (i, layer) in seg.layers.iter().enumerate() {
            let last = i == nl - 1;
            let (next, fin) = gkr::verify_layer(layer, b, &claim, &mut tr, &mut rd, last)?;
            if let Some((l_star, vl, r_star, vr)) = fin {
                let c_star = &next.copy_point;
                match mode {
                    Mode::Clear => {
                        if eval_columns(&seg.cols, seg.ncols, seg.units, s_c, &l_star, c_star) != vl
                        {
                            return Err(format!(
                                "{}: input layer: left evaluation mismatch",
                                seg.name
                            ));
                        }
                        if eval_columns(&seg.cols, seg.ncols, seg.units, s_c, &r_star, c_star) != vr
                        {
                            return Err(format!(
                                "{}: input layer: right evaluation mismatch",
                                seg.name
                            ));
                        }
                    }
                    Mode::Pcs => finals.push(((l_star, vl, r_star, vr), c_star.clone())),
                }
            }
            claim = next;
        }
    }
    match mode {
        Mode::Clear => {
            if inst.chain.is_some() {
                check_chain_clear(inst)?;
            }
        }
        Mode::Pcs => {
            let rhos = tr.challenges(n_functionals(inst));
            let geo = if inst.chain.is_some() {
                tr.challenge()
            } else {
                Ext::ZERO
            };
            let mut acc = Acc {
                a: vec![Ext::ZERO; l],
                b: Ext::ZERO,
            };
            let mut ri = 0;
            for (si, seg) in inst.segs.iter().enumerate() {
                let ((l_star, vl, r_star, vr), c_star) = &finals[si];
                add_input_claim(&mut acc, rhos[ri], seg_offs[si], seg, l_star, c_star, *vl);
                add_input_claim(
                    &mut acc,
                    rhos[ri + 1],
                    seg_offs[si],
                    seg,
                    r_star,
                    c_star,
                    *vr,
                );
                ri += 2;
            }
            match inst.lookup {
                Lookup::PerTable => {
                    for (ti, t) in inst.tables().iter().enumerate() {
                        let (n, lc) = &lookup_claims[ti];
                        add_lookup_claim(
                            &mut acc,
                            rhos[ri],
                            inst,
                            &seg_offs,
                            offs[ti],
                            t,
                            &seg_queries(inst, &t.name),
                            &chs.per_table[ti],
                            *n,
                            lc,
                        );
                        ri += 1;
                    }
                }
                Lookup::Union => {
                    let (n, lc) = &lookup_claims[0];
                    add_union_claim(
                        &mut acc,
                        rhos[ri],
                        inst,
                        &seg_offs,
                        &offs,
                        &usegs,
                        chs.union.as_ref().unwrap(),
                        *n,
                        lc,
                    );
                    ri += 1;
                }
            }
            if inst.chain.is_some() {
                add_chain(&mut acc, rhos[ri], geo, inst, &seg_offs);
            }
            let rows = l.div_ceil(LIGERO.k);
            ligero::verify_open(
                &proof.root,
                rows,
                acc.a,
                acc.b,
                &LIGERO,
                &mut tr,
                &mut rd,
                &proof.bytes,
            )?;
        }
    }
    if rd.pos != proof.msgs.len() {
        return Err("trailing proof data".into());
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------------------------
// loading
// ------------------------------------------------------------------------------------------------------------------

pub struct Loaded {
    pub circ: Circuit,
    pub layers: Vec<Layer>,
    pub hash: [u8; 32],
}

fn load_circuit(path: &str) -> Loaded {
    let text = std::fs::read_to_string(path).expect("circuit");
    let hash = Sha256::digest(text.as_bytes());
    let circ = Circuit::parse(&text);
    let layers = layers(&circ);
    Loaded { circ, layers, hash }
}

fn load_rows(path: &str, ncols: usize, take: Option<usize>, tile: bool) -> (Vec<Fp>, usize) {
    let bytes = std::fs::read(path).expect("witness");
    let (n, m, mut cols) = read_witness(&bytes);
    assert_eq!(m, ncols, "{path}: column count");
    let units = take.unwrap_or(n);
    if units > n {
        assert!(tile, "{path}: {n} rows, {units} requested");
        // tile the available rows to the requested batch (timing runs only; each row is still a valid unit)
        let base = cols.clone();
        while cols.len() < units * m {
            let take = (units * m - cols.len()).min(base.len());
            cols.extend_from_slice(&base[..take]);
        }
    } else {
        cols.truncate(units * m);
    }
    (cols, units)
}

fn segment<'a>(name: &str, ld: &'a Loaded, cols: Vec<Fp>, units: usize) -> Segment<'a> {
    Segment {
        name: name.into(),
        circ: &ld.circ,
        layers: &ld.layers,
        cols,
        units,
        ncols: ld.circ.ncols,
        hash: ld.hash,
    }
}

fn col_index(c: &Circuit, name: &str) -> usize {
    c.col_names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("no column {name}"))
}

/// A VU batch directory written by `verity_numerical.gkr_export.vu` (or overrides of its files).
pub struct VuFiles {
    pub units: String,
    pub epilogue: String,
    pub public: String,
    pub chain: Option<String>,
}

impl VuFiles {
    /// The files of an export directory; `chain.txt` (checker v2, `gkr_export.v2.write_chain`) when present.
    pub fn in_dir(d: &str) -> VuFiles {
        let chain = format!("{d}/chain.txt");
        VuFiles {
            units: format!("{d}/units.bin"),
            epilogue: format!("{d}/epilogue.bin"),
            public: format!("{d}/public.bin"),
            chain: std::path::Path::new(&chain).is_file().then_some(chain),
        }
    }
}

/// `chain.txt`: one `link <in|-> <out> <epi> <init> <zero_nonlast>` per line, columns by name (PROTOCOL.md 14.3).
fn read_chain(text: &str, unit: &Circuit, epi: &Circuit) -> Vec<Link> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let t: Vec<&str> = l.split_whitespace().collect();
            assert!(t.len() == 6 && t[0] == "link", "chain.txt: bad line {l:?}");
            Link {
                col_c: (t[1] != "-").then(|| col_index(unit, t[1])),
                col_y: col_index(unit, t[2]),
                epi: col_index(epi, t[3]),
                init: Fp::new(t[4].parse().unwrap()),
                zero_nonlast: t[5] == "1",
            }
        })
        .collect()
}

/// The checker-v2 chain (PROTOCOL.md 14.3), owned by the VERIFIER: the four state links `(s, e, M, z)` from the zero
/// state `(0, floor, 0, 1)` (`checker.params.REAL.floor = -132`), the two saturation flags (zero before the last
/// unit) and the two signs, all by column name, exactly `gkr_export.v2.chain_spec`.  It is a constant here because
/// the linkage is part of the relation, not of the proof: red-team-v2 case 9 (`chain_spec_stripped`) shipped a
/// bundle whose `chain.txt` was empty and the verifier, which then took the chain from the bundle, accepted an
/// epilogue that published a non-final unit's state.
pub fn v2_chain_spec() -> Vec<(Option<&'static str>, &'static str, &'static str, Fp, bool)> {
    let floor = Fp::ZERO - Fp::new(132);
    vec![
        (Some("x.s"), "y.s", "s", Fp::ZERO, false),
        (Some("x.e"), "y.e", "e", floor, false),
        (Some("x.M"), "y.M", "M", Fp::ZERO, false),
        (Some("x.z"), "y.z", "z", Fp::ONE, false),
        (None, "g0.norm.leadnorm.ovf", "ovf[0]", Fp::ZERO, true),
        (None, "g1.norm.leadnorm.ovf", "ovf[1]", Fp::ZERO, true),
        (None, "h.groups[0].sum.sgn", "sgn[0]", Fp::ZERO, false),
        (None, "h.groups[1].sum.sgn", "sgn[1]", Fp::ZERO, false),
    ]
}

/// The verifier's v2 chain resolved against the two circuits (every named column must exist).
fn v2_chain(unit: &Circuit, epi: &Circuit) -> Vec<Link> {
    v2_chain_spec()
        .into_iter()
        .map(|(c, y, e, init, zn)| Link {
            col_c: c.map(|c| col_index(unit, c)),
            col_y: col_index(unit, y),
            epi: col_index(epi, e),
            init,
            zero_nonlast: zn,
        })
        .collect()
}

fn load_vu<'a>(
    unit: &'a Loaded,
    epi: &'a Loaded,
    files: &VuFiles,
    vus: Option<usize>,
) -> Instance<'a> {
    let header = |path: &str| -> usize {
        let b = std::fs::read(path).expect("witness");
        u64::from_le_bytes(b[..8].try_into().unwrap()) as usize
    };
    let n_pub = header(&files.public);
    let n_units = header(&files.units);
    assert!(
        n_pub > 0 && n_units % n_pub == 0,
        "units rows must be a multiple of the VU count"
    );
    let steps = n_units / n_pub;
    let b = vus.unwrap_or(n_pub).min(n_pub);
    let (ucols, nu) = load_rows(&files.units, unit.circ.ncols, Some(b * steps), false);
    let (ecols, ne) = load_rows(&files.epilogue, epi.circ.ncols, Some(b), false);
    let (pcols, np) = load_rows(&files.public, 1, Some(b), false);
    assert!(nu == b * steps && ne == b && np == b);
    let bundle_chain = files
        .chain
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|text| read_chain(&text, &unit.circ, &epi.circ));
    let limbs: Vec<Link> = if unit.circ.col_names.iter().any(|n| n == "x.s") {
        // checker v2: the chain is the verifier's (`v2_chain_spec`); a bundle's `chain.txt` is informative only and
        // must agree with it -- an empty or shortened one is a forgery attempt, not a configuration (red-team-v2 #9).
        let own = v2_chain(&unit.circ, &epi.circ);
        if let Some(theirs) = bundle_chain {
            assert!(theirs.len() == own.len() && theirs.iter().zip(&own).all(|(a, b)| a.col_c == b.col_c && a.col_y == b.col_y && a.epi == b.epi && a.init == b.init && a.zero_nonlast == b.zero_nonlast),
                "chain.txt disagrees with the verifier's checker-v2 chain (PROTOCOL.md 14.3): {} link(s) given, {} required", theirs.len(), own.len());
        }
        own
    } else if let Some(theirs) = bundle_chain {
        // v1 with an explicit chain file (tests): the word links are the only admissible content
        assert!(!theirs.is_empty(), "chain.txt is empty");
        theirs
    } else if unit.circ.col_names.iter().any(|n| n == "x.c") {
        vec![Link::word([
            col_index(&unit.circ, "x.c"),
            col_index(&unit.circ, "y"),
            col_index(&epi.circ, "y32"),
        ])]
    } else {
        ["lo", "hi"]
            .iter()
            .map(|l| {
                Link::word([
                    col_index(&unit.circ, &format!("x.c_{l}")),
                    col_index(&unit.circ, &format!("y_{l}")),
                    col_index(&epi.circ, &format!("y32_{l}")),
                ])
            })
            .collect()
    };
    let chain = Chain {
        steps,
        limbs,
        epi_y16: col_index(&epi.circ, "y16"),
        public_y16: pcols,
    };
    Instance::new(
        vec![
            segment("units", unit, ucols, nu),
            segment("epilogue", epi, ecols, ne),
        ],
        Some(chain),
    )
}

fn strip_cols<'a>(inst: &Instance<'a>) -> Instance<'a> {
    Instance::new(
        inst.segs
            .iter()
            .map(|s| Segment {
                name: s.name.clone(),
                circ: s.circ,
                layers: s.layers,
                cols: vec![],
                units: s.units,
                ncols: s.ncols,
                hash: s.hash,
            })
            .collect(),
        inst.chain.clone(),
    )
    .with_lookup(inst.lookup)
}

fn verifier_view<'a>(inst: &Instance<'a>, mode: Mode) -> Instance<'a> {
    if mode == Mode::Pcs {
        strip_cols(inst)
    } else {
        Instance::new(
            inst.segs
                .iter()
                .map(|s| Segment {
                    name: s.name.clone(),
                    circ: s.circ,
                    layers: s.layers,
                    cols: s.cols.clone(),
                    units: s.units,
                    ncols: s.ncols,
                    hash: s.hash,
                })
                .collect(),
            strip_cols(inst).chain,
        )
        .with_lookup(inst.lookup)
    }
}

// ------------------------------------------------------------------------------------------------------------------
// result.json, CLI
// ------------------------------------------------------------------------------------------------------------------

fn peak_rss_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

fn cpu_model() -> Option<String> {
    let s = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    s.lines()
        .find(|l| l.starts_with("model name"))
        .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .map(|i| args[i + 1].clone())
}

/// `t_prove` is the prover's compute time (wall minus the emulated-network sleep); `t_wall` the wall time with the
/// sleeps; `rtt_ms` the emulated RTT (0 when none).
struct RunReport<'a> {
    inst: &'a Instance<'a>,
    mode: Mode,
    proof: &'a Proof,
    st: &'a Stats,
    t_prove: f64,
    t_wall: f64,
    t_verify: f64,
    rtt_ms: f64,
    res: &'a Result<(), String>,
    k: usize,
    unit: &'a str,
    b: usize,
}

/// The contract's RTT sweep (`t.e2e.rtt_<x>ms`).
const RTT_SWEEP_MS: [(&str, f64); 5] = [
    ("0", 0.0),
    ("0.1", 0.1),
    ("1", 1.0),
    ("10", 10.0),
    ("50", 50.0),
];

fn write_result(out: &str, args: &[String], r: &RunReport) {
    let run_id = arg(args, "--run-id")
        .or_else(|| std::env::var("RESEARCH_RUN_ID").ok())
        .unwrap_or_else(|| "local".into());
    let commit = arg(args, "--commit").unwrap_or_else(|| "unknown".into());
    let threads = threads();
    let rss = peak_rss_bytes();
    let depth = r.st.trips;
    let depth_legacy =
        1 + r.st.depth_circuit.max(r.st.depth_lookup_max) + if r.mode == Mode::Pcs { 2 } else { 0 };
    let ok = r.res.is_ok();
    let cpu = cpu_model()
        .map(|m| {
            format!(
                "{{\"model\": \"{m}\", \"count\": {}}}",
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(0)
            )
        })
        .unwrap_or("null".into());
    let meas = |name: &str, value: f64, unit: &str| {
        format!("{{\"name\": \"{name}\", \"value\": {value}, \"unit\": \"{unit}\"}}")
    };
    let st = r.st;
    let mut ms = vec![
        meas("t.total", r.t_prove, "s"),
        meas("t.lookup", st.t_lookup, "s"),
        meas("t.arithmetic", st.t_arith, "s"),
        meas("t.witness", st.t_witness_wires, "s"),
        meas("t.encoding_commitment", st.t_commit + st.t_open, "s"),
        meas("t.zk_additional", 0.0, "s"),
        meas("t.serialization", 0.0, "s"),
        meas("verifier.seconds", r.t_verify, "s"),
        meas("transcript.bytes", r.proof.size() as f64, "B"),
        meas("rounds.protocol", st.slots as f64, "count"),
        meas("rounds.sequential_depth", depth as f64, "count"),
        meas(
            "census.committed_elements",
            st.committed_elements as f64,
            "count",
        ),
        meas("prove_seconds", r.t_prove, "s"),
        meas("verify_seconds", r.t_verify, "s"),
        meas("proof_bytes", r.proof.size() as f64, "B"),
        meas("ligero_rows", st.ligero_rows as f64, "count"),
        meas("threads", threads as f64, "count"),
        meas("units", r.inst.segs[0].units as f64, "count"),
        meas("columns_per_unit", r.inst.segs[0].ncols as f64, "count"),
        meas("t_commit", st.t_commit, "s"),
        meas("t_open", st.t_open, "s"),
        meas("depth_lookup_chain", st.depth_lookup_chain as f64, "count"),
        meas(
            "depth_checker_chain",
            st.depth_checker_chain as f64,
            "count",
        ),
        meas("depth_formula_legacy", depth_legacy as f64, "count"),
        meas("rtt_ms", r.rtt_ms, "ms"),
        meas("rtt_sleep_seconds", st.slept, "s"),
        meas("prove_wall_seconds", r.t_wall, "s"),
    ];
    // end-to-end wall time at the RTT sweep: measured at the emulated RTT (prover wall incl. the sleeps + verifier),
    // predicted at the others as compute + depth x RTT (the honest serial number; bytes / bandwidth not included here)
    let mut e2e_measured: Vec<String> = vec![];
    let mut e2e_predicted: Vec<String> = vec![];
    for (name, rtt) in RTT_SWEEP_MS {
        let key = format!("t.e2e.rtt_{name}ms");
        if (rtt - r.rtt_ms).abs() < 1e-9 {
            ms.push(meas(&key, r.t_wall + r.t_verify, "s"));
            e2e_measured.push(format!("\"{key}\""));
        } else {
            ms.push(meas(
                &key,
                r.t_prove + r.t_verify + depth as f64 * rtt / 1000.0,
                "s",
            ));
            e2e_predicted.push(format!("\"{key}\""));
        }
    }
    if r.unit == "vu" {
        let vus = r.inst.vus() as f64;
        ms.push(meas("vu.count", vus, "count"));
        ms.push(meas("vu.seconds_per_vu", r.t_prove / vus, "s"));
        ms.push(meas("vu.per_second", vus / r.t_prove, "1/s"));
        ms.push(meas(
            "linkage_constraints",
            st.linkage_constraints as f64,
            "count",
        ));
        ms.push(meas(
            "epilogue_columns",
            r.inst.segs[1].ncols as f64,
            "count",
        ));
    }
    if let Some(rss) = rss {
        ms.push(meas("mem.peak_host_bytes", rss as f64, "B"));
        ms.push(meas("peak_rss_bytes", rss as f64, "B"));
    }
    let lk: Vec<String> = st
        .lookup_dims
        .iter()
        .map(|(n, d, q)| format!("\"{n}\": {{\"hypercube_dim\": {d}, \"queries_per_unit\": {q}}}"))
        .collect();
    let segs: Vec<String> = r.inst.segs.iter().map(|s| format!("{{\"name\": \"{}\", \"rows\": {}, \"columns\": {}, \"wires\": {}, \"layers\": {}}}", s.name, s.units, s.ncols, s.circ.nwires(), s.layers.len())).collect();
    let json = format!(
        "{{\n\"schema\": \"research/result/v0.1\",\n\"run_id\": \"{run_id}\",\n\"workload_fingerprint\": {{\n  \"profile\": \"first-campaign-target/2026-09-21\", \"domain\": \"finite\", \"K\": {}, \"B\": {}, \"N_subbatches\": 1,\n  \"unit\": \"{}\",\n  \"proof_class\": \"{}\", \"authentication\": \"excluded\", \"zk_mode\": \"none\",\n  \"security\": {{\"target\": -128, \"trace_field\": \"{}\", \"challenge_field\": \"{}\", \"fiat_shamir\": \"sha256\", \"commitment\": \"{}\", \"note\": \"see report for the error terms this configuration achieves\"}},\n  \"hardware\": {{\"gpu\": null, \"cpu\": {cpu}}},\n  \"software\": {{\"backend\": {{\"name\": \"verity-gkr\", \"version\": \"0.2.0\", \"commit\": \"{commit}\"}}, \"toolchain\": {{\"rustc\": \"{}\"}}}},\n  \"preprocessing\": \"circuit export from the WP2 checker (verity_numerical.gkr_export) and witness generation by WP2 gen_step in Python; not timed here\",\n  \"retention\": \"retained\",\n  \"threads\": {threads},\n  \"latency\": {{\"protocol\": \"interactive (production: verifier coins over the network, ZK_CONSTRUCTION.md 3.3); measured here with the Fiat-Shamir transcript and an emulated RTT: the prover sleeps one RTT at every critical-path verifier message\", \"rtt_ms\": {}, \"e2e_measured\": [{}], \"e2e_predicted\": [{}], \"e2e_model\": \"t.total + verifier.seconds + rounds.sequential_depth x RTT (bytes / bandwidth not included)\", \"depth_note\": \"rounds.sequential_depth is the measured count of critical-path verifier messages (Transcript::trips); depth_formula_legacy is the earlier derived counter, which counted the logUp (mu, lambda) pair as two messages\"}},\n  \"segments\": [{}],\n  \"lookup_mode\": \"{}\",\n  \"lookups\": {{{}}}\n}},\n\"validation\": {{\"status\": \"{}\", \"checker\": \"verity-gkr verify\", \"detail\": \"{}\", \"evidence\": []}},\n\"measurements\": [\n{}\n]\n}}\n",
        r.k, r.b, r.unit,
        if r.mode == Mode::Pcs { "NON_ZK_PROOF_DIAGNOSTIC" } else { "ARITHMETIC_DIAGNOSTIC" }, field::NAME, field::EXT_NAME,
        if r.mode == Mode::Pcs { format!("ligero-style RS(k={},n={}) over {}, SHA-256 Merkle columns, t={} opened, no masking", LIGERO.k, LIGERO.n, field::NAME, LIGERO.t) } else { "none: columns in the clear".to_string() },
        std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".into()), r.rtt_ms, e2e_measured.join(", "), e2e_predicted.join(", "), segs.join(", "), r.inst.lookup.name(), lk.join(", "), if ok { "passed" } else { "failed" },
        match r.res { Ok(()) => "verifier accepted the batch".to_string(), Err(e) => e.replace('"', "'") }, ms.join(",\n"));
    std::fs::write(out, json).expect("write result");
}

fn run_and_report(
    inst: &Instance,
    mode: Mode,
    args: &[String],
    k: usize,
    unit: &str,
    b: usize,
) -> bool {
    // --rtt-ms R (or VERITY_GKR_RTT_MS): emulate the network; the prover sleeps R ms at every critical-path challenge
    let rtt_ms: f64 = arg(args, "--rtt-ms")
        .or_else(|| std::env::var("VERITY_GKR_RTT_MS").ok())
        .map(|s| s.parse().expect("--rtt-ms takes milliseconds"))
        .unwrap_or(0.0);
    let rtt = if rtt_ms > 0.0 {
        Some(std::time::Duration::from_secs_f64(rtt_ms / 1000.0))
    } else {
        None
    };
    let tp = Instant::now();
    let (proof, st) = prove_with_rtt(inst, mode, rtt);
    let t_wall = tp.elapsed().as_secs_f64();
    let t_prove = t_wall - st.slept; // compute: the wall time less the emulated waits for the verifier's coins
                                     // the bucket timers include the sleeps of their chain: take them out of the bucket that slept
    let mut st = st;
    if st.slept > 0.0 {
        let lookup_share = if st.depth_lookup_chain >= st.depth_checker_chain {
            st.depth_lookup_chain
        } else {
            0
        } as f64;
        let checker_share = if st.depth_lookup_chain >= st.depth_checker_chain {
            0
        } else {
            st.depth_checker_chain
        } as f64;
        let per_trip = st.slept / st.trips as f64;
        st.t_lookup -= lookup_share * per_trip;
        st.t_arith -= checker_share * per_trip;
        st.t_open -= (st.trips as f64 - lookup_share - checker_share - 1.0).max(0.0) * per_trip;
        // the Ligero r, Q trips
    }
    let tv = Instant::now();
    let vinst = verifier_view(inst, mode);
    let res = verify(&vinst, &proof, mode);
    let t_verify = tv.elapsed().as_secs_f64();
    let depth = st.trips;
    eprintln!("field={} mode={mode:?} lookup={} unit={unit} B={b} units={} verify={:?} prove_s={t_prove:.3} (commit {:.3} lookup {:.3} wires {:.3} arith {:.3} open {:.3}) verify_s={t_verify:.3} proof_bytes={} slots={} depth={depth} (lookup chain {}, checker chain {}) rtt_ms={rtt_ms} slept_s={:.3} wall_s={t_wall:.3} e2e_s={:.3} rows={} committed={} lookups={:?} rss={:?}",
        field::NAME, inst.lookup.name(), inst.segs[0].units, res, st.t_commit, st.t_lookup, st.t_witness_wires, st.t_arith, st.t_open, proof.size(), st.slots, st.depth_lookup_chain, st.depth_checker_chain, st.slept, t_wall + t_verify, st.ligero_rows, st.committed_elements, st.lookup_dims, peak_rss_bytes());
    if let Some(out) = arg(args, "--out") {
        write_result(
            &out,
            args,
            &RunReport {
                inst,
                mode,
                proof: &proof,
                st: &st,
                t_prove,
                t_wall,
                t_verify,
                rtt_ms,
                res: &res,
                k,
                unit,
                b,
            },
        );
    }
    res.is_ok()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: run|run-vu|negatives|negatives-vu ...");
        std::process::exit(2);
    }
    let mode = if args.iter().any(|a| a == "--clear") {
        Mode::Clear
    } else {
        Mode::Pcs
    };
    // --logup union | per-table (default per-table; or VERITY_GKR_LOGUP)
    let lookup = arg(&args, "--logup")
        .or_else(|| std::env::var("VERITY_GKR_LOGUP").ok())
        .map(|s| Lookup::parse(&s))
        .unwrap_or(Lookup::PerTable);
    match args[1].as_str() {
        "run" => {
            let ld = load_circuit(&arg(&args, "--circuit").unwrap());
            let units: Option<usize> = arg(&args, "--units").map(|s| s.parse().unwrap());
            let (cols, units) = load_rows(
                &arg(&args, "--witness").unwrap(),
                ld.circ.ncols,
                units,
                true,
            );
            let inst =
                Instance::new(vec![segment("units", &ld, cols, units)], None).with_lookup(lookup);
            if !run_and_report(&inst, mode, &args, 16, "transition", units) {
                std::process::exit(1);
            }
        }
        "run-vu" => {
            let dir = arg(&args, "--dir").unwrap();
            let unit = load_circuit(&format!("{dir}/circuit.txt"));
            let epi = load_circuit(&format!("{dir}/epilogue.txt"));
            let files = VuFiles::in_dir(&dir);
            let vus: Option<usize> = arg(&args, "--vus").map(|s| s.parse().unwrap());
            let inst = load_vu(&unit, &epi, &files, vus).with_lookup(lookup);
            let b = inst.vus();
            let k = 16 * inst.chain.as_ref().unwrap().steps;
            if !run_and_report(&inst, mode, &args, k, "vu", b) {
                std::process::exit(1);
            }
        }
        "negatives" => {
            // `negatives FIXTURES/` (README form: FIXTURES/circuit.txt + FIXTURES/negatives/*.bin) or `--circuit C --dir D`.
            let positional = args.get(2).filter(|a| !a.starts_with("--")).cloned();
            let circuit = arg(&args, "--circuit")
                .or_else(|| {
                    positional
                        .as_ref()
                        .map(|d| format!("{}/circuit.txt", d.trim_end_matches('/')))
                })
                .expect("negatives: pass FIXTURES/ or --circuit C --dir D");
            let dir = arg(&args, "--dir")
                .or_else(|| {
                    positional
                        .as_ref()
                        .map(|d| format!("{}/negatives", d.trim_end_matches('/')))
                })
                .unwrap();
            let ld = load_circuit(&circuit);
            let mut n_ok = 0;
            let mut n_bad = 0;
            let mut entries: Vec<_> = std::fs::read_dir(&dir)
                .unwrap()
                .map(|e| e.unwrap().path())
                .filter(|p| p.extension().map(|e| e == "bin").unwrap_or(false))
                .collect();
            entries.sort();
            for p in entries {
                let (cols, units) = load_rows(p.to_str().unwrap(), ld.circ.ncols, None, false);
                let inst = Instance::new(vec![segment("units", &ld, cols, units)], None)
                    .with_lookup(lookup);
                let (proof, _) = prove(&inst, mode);
                match verify(&verifier_view(&inst, mode), &proof, mode) {
                    Ok(()) => {
                        n_bad += 1;
                        println!("ACCEPTED (BUG) {}", p.display());
                    }
                    Err(e) => {
                        n_ok += 1;
                        println!(
                            "rejected {} : {e}",
                            p.file_name().unwrap().to_string_lossy()
                        );
                    }
                }
            }
            println!("negatives rejected: {n_ok}, wrongly accepted: {n_bad}");
            if n_bad > 0 {
                std::process::exit(1);
            }
        }
        "negatives-vu" => {
            let dir = arg(&args, "--dir").unwrap();
            let unit = load_circuit(&format!("{dir}/circuit.txt"));
            let epi = load_circuit(&format!("{dir}/epilogue.txt"));
            let mut ids: Vec<_> = std::fs::read_dir(format!("{dir}/neg"))
                .unwrap()
                .map(|e| e.unwrap().path())
                .collect();
            ids.sort();
            let (mut n_ok, mut n_bad) = (0, 0);
            for p in ids {
                let d = p.to_str().unwrap();
                let files = VuFiles::in_dir(d);
                let inst = load_vu(&unit, &epi, &files, None).with_lookup(lookup);
                let (proof, _) = prove(&inst, mode);
                match verify(&verifier_view(&inst, mode), &proof, mode) {
                    Ok(()) => {
                        n_bad += 1;
                        println!("ACCEPTED (BUG) {}", p.display());
                    }
                    Err(e) => {
                        n_ok += 1;
                        println!(
                            "rejected {} : {e}",
                            p.file_name().unwrap().to_string_lossy()
                        );
                    }
                }
            }
            println!("vu negatives rejected: {n_ok}, wrongly accepted: {n_bad}");
            if n_bad > 0 {
                std::process::exit(1);
            }
        }
        "check-witness" => {
            // the verifier's constraint check on a supplied witness (every row), without proving: exit 0 iff all rows satisfy
            let ld = load_circuit(&arg(&args, "--circuit").unwrap());
            let (cols, units) = load_rows(
                &arg(&args, "--witness").unwrap(),
                ld.circ.ncols,
                None,
                false,
            );
            let mut bad = 0;
            for u in 0..units {
                if let Err(e) = ld
                    .circ
                    .check_row(&cols[u * ld.circ.ncols..(u + 1) * ld.circ.ncols])
                {
                    bad += 1;
                    println!("row {u}: {e}");
                }
            }
            println!("check-witness: {units} rows, {bad} violated");
            if bad > 0 {
                std::process::exit(1);
            }
        }
        "dump-transcript" => {
            let dir = arg(&args, "--dir").unwrap();
            let unit = load_circuit(&format!("{dir}/circuit.txt"));
            let epi = load_circuit(&format!("{dir}/epilogue.txt"));
            let files = VuFiles::in_dir(&dir);
            let vus: Option<usize> = arg(&args, "--vus").map(|s| s.parse().unwrap());
            let inst = load_vu(&unit, &epi, &files, vus).with_lookup(lookup);
            LOG_TRANSCRIPT.store(true, Ordering::Relaxed);
            let (proof, st) = prove(&inst, mode);
            LOG_TRANSCRIPT.store(false, Ordering::Relaxed);
            verify(&verifier_view(&inst, mode), &proof, mode)
                .expect("the dumped proof must verify");
            let json = transcript_json(
                &inst,
                mode,
                &proof,
                st.transcript_log.as_deref().unwrap_or(&[]),
            );
            std::fs::write(arg(&args, "--out").unwrap(), json).expect("write transcript");
            eprintln!(
                "dump-transcript: field={} vus={} events={} msgs={} bytes={}",
                field::NAME,
                inst.vus(),
                st.transcript_log.map(|l| l.len()).unwrap_or(0),
                proof.msgs.len(),
                proof.bytes.len()
            );
        }
        _ => {
            eprintln!("unknown command");
            std::process::exit(2);
        }
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn ext_json(x: &Ext) -> String {
    format!(
        "[{}]",
        x.0.iter()
            .map(|c| c.to_u64().to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// The transcript vectors (fixtures/babybear-transcripts/README.md): every absorb with the state after it, every
/// challenge with the counter it was squeezed at, the phase labels, and the proof messages.
fn transcript_json(inst: &Instance, mode: Mode, proof: &Proof, log: &[Event]) -> String {
    let mut ev = Vec::with_capacity(log.len());
    for e in log {
        ev.push(match e {
            Event::Label(s) => format!(
                "{{\"label\": \"{}\"}}",
                s.replace('\\', "\\\\").replace('"', "\\\"")
            ),
            Event::Absorb { bytes, state } => format!(
                "{{\"absorb\": \"{}\", \"state\": \"{}\"}}",
                hex(bytes),
                hex(state)
            ),
            Event::Challenge { value, ctr_before } => format!(
                "{{\"challenge\": {}, \"ctr\": {ctr_before}}}",
                ext_json(value)
            ),
        });
    }
    let msgs: Vec<String> = proof.msgs.iter().map(ext_json).collect();
    let segs: Vec<String> = inst
        .segs
        .iter()
        .map(|s| {
            format!(
                "{{\"name\": \"{}\", \"circuit_sha256\": \"{}\", \"units\": {}, \"columns\": {}}}",
                s.name,
                hex(&s.hash),
                s.units,
                s.ncols
            )
        })
        .collect();
    let public: Vec<String> = inst
        .chain
        .as_ref()
        .map(|c| {
            c.public_y16
                .iter()
                .map(|y| y.to_u64().to_string())
                .collect()
        })
        .unwrap_or_default();
    format!("{{\n\"schema\": \"verity-gkr/transcript/v1\",\n\"field\": \"{}\", \"modulus\": {}, \"extension\": \"{}\", \"degree\": {}, \"base_bytes\": {}, \"hash\": \"sha256\",\n\"domain\": \"{}\",\n\"mode\": \"{}\", \"ligero\": {{\"k\": {}, \"n\": {}, \"t\": {}}},\n\"vus\": {}, \"steps\": {},\n\"segments\": [{}],\n\"public_y16\": [{}],\n\"root\": \"{}\",\n\"events\": [\n{}\n],\n\"proof_msgs\": [\n{}\n],\n\"proof_bytes\": \"{}\"\n}}\n",
        field::NAME, field::P, field::EXT_NAME, field::DEGREE, Fp::BYTES, String::from_utf8_lossy(field::DOMAIN),
        if mode == Mode::Pcs { "pcs" } else { "clear" }, LIGERO.k, LIGERO.n, LIGERO.t,
        inst.vus(), inst.chain.as_ref().map(|c| c.steps).unwrap_or(0), segs.join(", "), public.join(","), hex(&proof.root),
        ev.join(",\n"), msgs.join(",\n"), hex(&proof.bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Fixture path of the active field (`fixtures/` for Goldilocks, `fixtures/babybear/` for BabyBear).
    fn fx(p: &str) -> String {
        format!(
            "{}/fixtures/{}{p}",
            env!("CARGO_MANIFEST_DIR"),
            field::FIXTURES
        )
    }

    fn unit_instance<'a>(ld: &'a Loaded, witness: &str, units: Option<usize>) -> Instance<'a> {
        let (cols, units) = load_rows(witness, ld.circ.ncols, units, true);
        Instance::new(vec![segment("units", ld, cols, units)], None)
    }

    #[test]
    fn honest_units_accepted_at_several_batch_sizes() {
        let ld = load_circuit(&fx("circuit.txt"));
        for units in [1usize, 2, 8] {
            for lookup in [Lookup::PerTable, Lookup::Union] {
                let inst = unit_instance(&ld, &fx("honest8.bin"), Some(units)).with_lookup(lookup);
                for mode in [Mode::Clear, Mode::Pcs] {
                    let (proof, _) = prove(&inst, mode);
                    verify(&verifier_view(&inst, mode), &proof, mode).unwrap();
                }
            }
        }
    }

    #[test]
    fn every_exported_negative_is_rejected() {
        let ld = load_circuit(&fx("circuit.txt"));
        let dir = fx("negatives");
        let mut n = 0;
        for e in std::fs::read_dir(&dir).unwrap() {
            let p = e.unwrap().path();
            if p.extension().map(|e| e != "bin").unwrap_or(true) {
                continue;
            }
            let inst = unit_instance(&ld, p.to_str().unwrap(), None);
            for mode in [Mode::Clear, Mode::Pcs] {
                let (proof, _) = prove(&inst, mode);
                assert!(
                    verify(&verifier_view(&inst, mode), &proof, mode).is_err(),
                    "{} accepted in {mode:?}",
                    p.display()
                );
            }
            let inst = inst.with_lookup(Lookup::Union);
            let (proof, _) = prove(&inst, Mode::Pcs);
            assert!(
                verify(&verifier_view(&inst, Mode::Pcs), &proof, Mode::Pcs).is_err(),
                "{} accepted in union mode",
                p.display()
            );
            n += 1;
        }
        assert!(n >= 50);
    }

    #[test]
    fn tampered_proof_is_rejected() {
        let ld = load_circuit(&fx("circuit.txt"));
        for lookup in [Lookup::PerTable, Lookup::Union] {
            let inst = unit_instance(&ld, &fx("honest8.bin"), Some(2)).with_lookup(lookup);
            for mode in [Mode::Clear, Mode::Pcs] {
                let (mut proof, _) = prove(&inst, mode);
                let k = proof.msgs.len() / 2;
                proof.msgs[k] = proof.msgs[k] + Ext::ONE;
                assert!(verify(&inst, &proof, mode).is_err());
                let (mut proof, _) = prove(&inst, mode);
                if mode == Mode::Pcs {
                    proof.bytes[5] ^= 1;
                    assert!(verify(&inst, &proof, mode).is_err());
                }
            }
        }
        // a union proof is not a per-table proof and vice versa (the mode is in the statement preamble)
        let inst = unit_instance(&ld, &fx("honest8.bin"), Some(2));
        let (proof, _) = prove(&inst, Mode::Pcs);
        assert!(verify(
            &verifier_view(&inst, Mode::Pcs).with_lookup(Lookup::Union),
            &proof,
            Mode::Pcs
        )
        .is_err());
    }

    fn synth_files(units: &str, public: &str) -> VuFiles {
        VuFiles {
            units: fx(&format!("vu_synth/{units}")),
            epilogue: fx("vu_synth/epilogue.bin"),
            public: fx(&format!("vu_synth/{public}")),
            chain: None,
        }
    }

    #[test]
    fn synthetic_vus_accepted_with_chain_and_public_words() {
        let unit = load_circuit(&fx("vu_synth/circuit.txt"));
        let epi = load_circuit(&fx("vu_synth/epilogue.txt"));
        for vus in [1usize, 2] {
            let inst = load_vu(
                &unit,
                &epi,
                &synth_files("units.bin", "public.bin"),
                Some(vus),
            );
            assert_eq!(inst.chain.as_ref().unwrap().steps, 3);
            for mode in [Mode::Clear, Mode::Pcs] {
                let (proof, st) = prove(&inst, mode);
                verify(&verifier_view(&inst, mode), &proof, mode).unwrap();
                let limbs = inst.chain.as_ref().unwrap().limbs.len();
                if mode == Mode::Pcs {
                    assert_eq!(st.linkage_constraints, vus * (limbs * (3 + 1) + 1));
                    assert_eq!(
                        st.linkage_constraints,
                        vus * inst.chain.as_ref().unwrap().constraints_per_vu()
                    );
                }
            }
        }
    }

    #[test]
    fn synthetic_vu_negatives_rejected() {
        let unit = load_circuit(&fx("vu_synth/circuit.txt"));
        let epi = load_circuit(&fx("vu_synth/epilogue.txt"));
        for (u, p) in [
            ("units.bin", "negatives/public_wrong_word.bin"),
            ("negatives/units_broken_chain.bin", "public.bin"),
        ] {
            let inst = load_vu(&unit, &epi, &synth_files(u, p), None);
            for mode in [Mode::Clear, Mode::Pcs] {
                let (proof, _) = prove(&inst, mode);
                assert!(
                    verify(&verifier_view(&inst, mode), &proof, mode).is_err(),
                    "{u} / {p} accepted in {mode:?}"
                );
            }
        }
        // a prover that swaps the public words of two VUs (each word valid for the other) must fail too
        let inst = load_vu(&unit, &epi, &synth_files("units.bin", "public.bin"), None);
        let mut swapped = Instance::new(
            inst.segs
                .iter()
                .map(|s| Segment {
                    name: s.name.clone(),
                    circ: s.circ,
                    layers: s.layers,
                    cols: s.cols.clone(),
                    units: s.units,
                    ncols: s.ncols,
                    hash: s.hash,
                })
                .collect(),
            strip_cols(&inst).chain,
        );
        let ch = swapped.chain.as_mut().unwrap();
        ch.public_y16.swap(0, 1);
        let (proof, _) = prove(&swapped, Mode::Pcs);
        assert!(verify(&verifier_view(&swapped, Mode::Pcs), &proof, Mode::Pcs).is_err());
    }

    // ---- checker v2 (PROTOCOL.md 14): fixtures/v2_synth from `gkr_export.vu synth --variant v2` (Goldilocks only)

    #[cfg(not(feature = "babybear"))]
    fn v2_files(units: &str, epilogue: &str, public: &str) -> VuFiles {
        let f = |p: &str| format!("{}/fixtures/v2_synth/{p}", env!("CARGO_MANIFEST_DIR"));
        VuFiles {
            units: f(units),
            epilogue: f(epilogue),
            public: f(public),
            chain: Some(f("chain.txt")),
        }
    }

    #[cfg(not(feature = "babybear"))]
    #[test]
    fn implicit_tables_match_their_definition() {
        // SHIFT at REAL: bits 16, scale 10, R 54: key x·54 + 2d + s -> ±floor(x·2^10 / 2^d)
        let t = materialise("shift", &[16, 10, 54], 65536 * 54);
        let at = |x: u64, d: u64, s: u64| t[(x * 54 + 2 * d + s) as usize][1];
        assert_eq!(at(3, 0, 0), Fp::new(3 << 10));
        assert_eq!(at(3, 11, 0), Fp::new(1));
        assert_eq!(at(3, 12, 0), Fp::ZERO);
        assert_eq!(at(65535, 26, 1), -Fp::new((65535u64 << 10) >> 26));
        assert_eq!(at(7, 1, 1), -Fp::new(7 << 9));
        // TNORM at REAL: vb 15, k_rows 32, k_lo -7: key v + 2^15·koff -> floor(v·2^(koff-7))
        let t = materialise("tnorm", &[15, 32, -7], 32768 * 32);
        assert_eq!(t[(5 + 32768 * 0) as usize][1], Fp::new(5 >> 7));
        assert_eq!(t[(1000 + 32768 * 7) as usize][1], Fp::new(1000));
        assert_eq!(t[(32767 + 32768 * 31) as usize][1], Fp::new(32767u64 << 24));
        let tb = Table::new("T".into(), 2, t);
        assert!(tb.dense);
        assert_eq!(tb.find(Fp::new(12345)), Some(12345));
        assert_eq!(tb.find(Fp::new(32768 * 32)), None);
        assert_eq!(tb.find(-Fp::ONE), None);
    }

    #[cfg(not(feature = "babybear"))]
    #[test]
    fn v2_synthetic_vus_accepted_with_state_chain() {
        let f = |p: &str| format!("{}/fixtures/v2_synth/{p}", env!("CARGO_MANIFEST_DIR"));
        let unit = load_circuit(&f("circuit.txt"));
        let epi = load_circuit(&f("epilogue.txt"));
        assert!(unit
            .circ
            .tables
            .iter()
            .any(|t| t.name == "SHIFT" && t.rows.len() == 3_538_944 && t.dense));
        for vus in [1usize, 2] {
            let inst = load_vu(
                &unit,
                &epi,
                &v2_files("units.bin", "epilogue.bin", "public.bin"),
                Some(vus),
            );
            let ch = inst.chain.as_ref().unwrap();
            assert_eq!(ch.steps, 3);
            assert_eq!(ch.limbs.len(), 8);
            assert_eq!(ch.constraints_per_vu(), 1 + 4 * (3 + 1) + 2 * (1 + 2) + 2);
            assert_eq!(
                inst.tables().len(),
                12,
                "10 unit tables + the epilogue's R16, R15"
            );
            for mode in [Mode::Clear, Mode::Pcs] {
                let (proof, st) = prove(&inst, mode);
                verify(&verifier_view(&inst, mode), &proof, mode).unwrap();
                if mode == Mode::Pcs {
                    assert_eq!(st.linkage_constraints, vus * ch.constraints_per_vu());
                }
            }
            let inst = inst.with_lookup(Lookup::Union);
            for mode in [Mode::Clear, Mode::Pcs] {
                let (proof, st) = prove(&inst, mode);
                verify(&verifier_view(&inst, mode), &proof, mode).unwrap();
                assert_eq!(st.lookup_dims.len(), 1);
                assert_eq!(st.lookup_dims[0].0, "UNION");
                assert_eq!(st.lookup_dims[0].2, 86 + 5);
            }
        }
    }

    /// A malicious epilogue hint (lane a-logup-union, 2026-09-22): the public word is forged as y16 + 1 by shifting
    /// 2^16 from `lo` into `hi` (every epilogue *assertion* still holds; only the epilogue's `R16` / `R15` range
    /// lookups can catch it).  Those lookups belong to a table that exists only in the epilogue circuit, so the
    /// lookup argument must cover the tables of EVERY segment, not segment 0's alone.
    #[cfg(not(feature = "babybear"))]
    #[test]
    fn v2_forged_public_word_via_epilogue_hint_is_rejected() {
        let f = |p: &str| format!("{}/fixtures/v2_synth/{p}", env!("CARGO_MANIFEST_DIR"));
        let unit = load_circuit(&f("circuit.txt"));
        let epi = load_circuit(&f("epilogue.txt"));
        let honest = load_vu(
            &unit,
            &epi,
            &v2_files("units.bin", "epilogue.bin", "public.bin"),
            Some(1),
        );
        let mut forged = Instance::new(
            honest
                .segs
                .iter()
                .map(|s| Segment {
                    name: s.name.clone(),
                    circ: s.circ,
                    layers: s.layers,
                    cols: s.cols.clone(),
                    units: s.units,
                    ncols: s.ncols,
                    hash: s.hash,
                })
                .collect(),
            strip_cols(&honest).chain,
        );
        let ec = &epi.circ;
        let (y16, hi, lo, hi2, lsb, rem) = (
            col_index(ec, "y16"),
            col_index(ec, "h.hi"),
            col_index(ec, "h.lo"),
            col_index(ec, "h.hi2"),
            col_index(ec, "h.lsb"),
            col_index(ec, "h.rem"),
        );
        let row = &mut forged.segs[1].cols[..ec.ncols];
        let two16 = Fp::new(65536);
        let carry = col_index(ec, "h.carry");
        row[hi] += Fp::ONE;
        row[lo] -= two16;
        if row[lsb] == Fp::ZERO {
            row[lsb] = Fp::ONE;
        } else {
            row[lsb] = Fp::ZERO;
            row[hi2] += Fp::ONE;
        }
        row[rem] = row[lo] + row[lsb] + Fp::new(32767) - two16 * row[carry]; // cast.round solved for rem
        row[y16] += Fp::ONE;
        forged.chain.as_mut().unwrap().public_y16[0] += Fp::ONE;
        // every assertion of the epilogue row holds: only the lookups can reject
        let mut w = Vec::new();
        ec.eval_wires(&forged.segs[1].cols[..ec.ncols], &mut w);
        for (name, l) in &ec.asserts {
            assert_eq!(
                circuit::eval_lin(l, &w),
                Fp::ZERO,
                "assertion {name} should still hold in the forgery"
            );
        }
        assert!(
            ec.check_row(&forged.segs[1].cols[..ec.ncols]).is_err(),
            "the clear check must see the out-of-range hint"
        );
        for mode in [Mode::Clear, Mode::Pcs] {
            let (proof, _) = prove(&forged, mode);
            assert!(
                verify(&verifier_view(&forged, mode), &proof, mode).is_err(),
                "forged public word accepted in {mode:?}"
            );
        }
    }

    /// The same forgery as a stored fixture for the red team (`fixtures/redteam-3/logup-union/forged_epilogue_hint`,
    /// generated by `verity_numerical.redteam.logup_union`): the v2_synth circuits with a 1-VU witness whose epilogue
    /// hint moves 2^16 from `lo` to `hi` and claims the word y16 + 1.
    #[cfg(not(feature = "babybear"))]
    #[test]
    fn redteam3_forged_epilogue_hint_fixture_is_rejected() {
        let f = |p: &str| format!("{}/fixtures/v2_synth/{p}", env!("CARGO_MANIFEST_DIR"));
        let d = format!(
            "{}/../../fixtures/redteam-3/logup-union/forged_epilogue_hint",
            env!("CARGO_MANIFEST_DIR")
        );
        let unit = load_circuit(&f("circuit.txt"));
        let epi = load_circuit(&f("epilogue.txt"));
        let inst = load_vu(&unit, &epi, &VuFiles::in_dir(&d), None);
        assert_eq!(inst.vus(), 1);
        assert!(
            inst.tables().iter().any(|t| t.name == "R16")
                && inst.tables().iter().any(|t| t.name == "R15"),
            "the epilogue's tables must be in the instance"
        );
        for mode in [Mode::Clear, Mode::Pcs] {
            let (proof, _) = prove(&inst, mode);
            let err = verify(&verifier_view(&inst, mode), &proof, mode)
                .expect_err("forged fixture accepted");
            assert!(
                err.starts_with("R16") || err.starts_with("R15"),
                "expected a range-table rejection, got {err}"
            );
        }
        let inst = inst.with_lookup(Lookup::Union);
        for mode in [Mode::Clear, Mode::Pcs] {
            let (proof, _) = prove(&inst, mode);
            let err = verify(&verifier_view(&inst, mode), &proof, mode)
                .expect_err("forged fixture accepted in union mode");
            assert!(
                err.starts_with("UNION"),
                "expected the union instance to reject, got {err}"
            );
        }
    }

    /// `fixtures/redteam-3/logup-union/cross_table`: a query into table TA whose value is a row of TB only.  The union
    /// argument must separate the two by the tag; the per-table argument trivially does.  `manifest.json` names the
    /// cases and their expected outcome.
    #[test]
    fn redteam3_cross_table_fixtures() {
        let d = format!(
            "{}/../../fixtures/redteam-3/logup-union/cross_table",
            env!("CARGO_MANIFEST_DIR")
        );
        let ld = load_circuit(&format!("{d}/circuit.txt"));
        assert_eq!(ld.circ.tables.len(), 2);
        let manifest = std::fs::read_to_string(format!("{d}/manifest.json")).unwrap();
        let mut seen = 0;
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.extension().map(|e| e != "bin").unwrap_or(true) {
                continue;
            }
            let name = p.file_stem().unwrap().to_str().unwrap().to_string();
            let expect_accept =
                manifest.contains(&format!("\"cross_table/{name}\": {{\n  \"rows\"")) && {
                    let i = manifest.find(&format!("cross_table/{name}\"")).unwrap();
                    manifest[i..]
                        .find("\"expected\": \"accept\"")
                        .map(|j| j < manifest[i..].find("\"mechanism\"").unwrap())
                        .unwrap_or(false)
                };
            for lookup in [Lookup::PerTable, Lookup::Union] {
                let inst = unit_instance(&ld, p.to_str().unwrap(), None).with_lookup(lookup);
                for mode in [Mode::Clear, Mode::Pcs] {
                    let (proof, _) = prove(&inst, mode);
                    let res = verify(&verifier_view(&inst, mode), &proof, mode);
                    assert_eq!(
                        res.is_ok(),
                        expect_accept,
                        "{name} in {lookup:?}/{mode:?}: {res:?}"
                    );
                }
            }
            seen += 1;
        }
        assert_eq!(seen, 5);
    }

    /// red-team-v2 #9: the chain is the verifier's, not the bundle's -- no `chain.txt` gives the same 8 links, an
    /// empty one is refused, and the shipped `chain_spec_stripped` forgery (empty file, epilogue on unit 0) is refused.
    #[cfg(not(feature = "babybear"))]
    #[test]
    fn v2_chain_is_verifier_owned() {
        let f = |p: &str| format!("{}/fixtures/v2_synth/{p}", env!("CARGO_MANIFEST_DIR"));
        let unit = load_circuit(&f("circuit.txt"));
        let epi = load_circuit(&f("epilogue.txt"));
        let mut files = v2_files("units.bin", "epilogue.bin", "public.bin");
        files.chain = None;
        let inst = load_vu(&unit, &epi, &files, Some(1));
        assert_eq!(inst.chain.as_ref().unwrap().limbs.len(), 8);
        let own = v2_chain(&unit.circ, &epi.circ);
        let theirs = read_chain(
            &std::fs::read_to_string(f("chain.txt")).unwrap(),
            &unit.circ,
            &epi.circ,
        );
        assert!(own.iter().zip(&theirs).all(|(a, b)| a.col_c == b.col_c
            && a.col_y == b.col_y
            && a.epi == b.epi
            && a.init == b.init
            && a.zero_nonlast == b.zero_nonlast));
        let empty = std::env::temp_dir().join("verity-gkr-empty-chain.txt");
        std::fs::write(&empty, "").unwrap();
        files.chain = Some(empty.to_str().unwrap().into());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load_vu(
                &unit,
                &epi,
                &files,
                Some(1)
            )))
            .is_err(),
            "an empty chain.txt was accepted"
        );
        let rt = format!(
            "{}/../../fixtures/redteam-v2/gkr/neg/chain_spec_stripped",
            env!("CARGO_MANIFEST_DIR")
        );
        if std::path::Path::new(&rt).is_dir() {
            let (u2, e2) = (
                load_circuit(&format!("{rt}/circuit.txt")),
                load_circuit(&format!("{rt}/epilogue.txt")),
            );
            let files = VuFiles::in_dir(&rt);
            let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                load_vu(&u2, &e2, &files, None)
            }));
            if let Ok(inst) = loaded {
                for mode in [Mode::Clear, Mode::Pcs] {
                    let (proof, _) = prove(&inst, mode);
                    assert!(
                        verify(&verifier_view(&inst, mode), &proof, mode).is_err(),
                        "chain_spec_stripped accepted in {mode:?}"
                    );
                }
            }
        }
    }

    #[cfg(not(feature = "babybear"))]
    #[test]
    fn v2_synthetic_negatives_rejected() {
        let f = |p: &str| format!("{}/fixtures/v2_synth/{p}", env!("CARGO_MANIFEST_DIR"));
        let unit = load_circuit(&f("circuit.txt"));
        let epi = load_circuit(&f("epilogue.txt"));
        let cases = [
            (
                "units.bin",
                "epilogue.bin",
                "negatives/public_wrong_word.bin",
            ),
            (
                "negatives/units_broken_chain.bin",
                "epilogue.bin",
                "public.bin",
            ),
            (
                "negatives/saturated_intermediate/units.bin",
                "negatives/saturated_intermediate/epilogue.bin",
                "negatives/saturated_intermediate/public.bin",
            ),
            (
                "negatives/false_overflow_flag/units.bin",
                "negatives/false_overflow_flag/epilogue.bin",
                "negatives/false_overflow_flag/public.bin",
            ),
        ];
        for (u, e, p) in cases {
            let inst = load_vu(&unit, &epi, &v2_files(u, e, p), None);
            for mode in [Mode::Clear, Mode::Pcs] {
                let (proof, _) = prove(&inst, mode);
                assert!(
                    verify(&verifier_view(&inst, mode), &proof, mode).is_err(),
                    "{u} / {e} / {p} accepted in {mode:?}"
                );
            }
            let inst = inst.with_lookup(Lookup::Union);
            let (proof, _) = prove(&inst, Mode::Pcs);
            assert!(
                verify(&verifier_view(&inst, Mode::Pcs), &proof, Mode::Pcs).is_err(),
                "{u} / {e} / {p} accepted in union mode"
            );
        }
    }
}
