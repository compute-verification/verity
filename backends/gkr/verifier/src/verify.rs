//! The verifier for the GPU prover's proofs (`gpu/prover.py::verify` re-derived from the protocol, not ported):
//! transcript replay, LogUp fractional-sum GKR per table, the data-parallel GKR per segment, the batched linear
//! functional `<a, X> = b` over the committed vector, and the Ligero opening (proximity + linear tests on `t`
//! SHA-256-authenticated columns).  Every size (units, columns, table shapes, `k, n, t`, message counts) comes from
//! the statement; the proof is rejected if its counts disagree.
//!
//! The expensive part is the linear test, which needs `a_i(eta_j)` for every committed row `i` and opened column
//! `j`.  `a` is never materialised: rows are built chunk by chunk from the claims (input claims are outer products,
//! lookup claims are per-unit sparse scatters of `eq(rho_leaf, .)` weights, the chain is geometric), each row is
//! interpolated once (INTT_k) and evaluated on the opened cosets (3 NTT_k), and the partial sums over rows are
//! reduced across threads.  The `eq` weights over the LogUp hypercube use a high/low split (one product per leaf).

use crate::circuit::{eval_lin, log2ceil, Circuit, Gate, Layer, Query, Table};
use crate::field::{add, eq_point, eq_table, interpolate, mat_apply, mont_mul, mul, mul_matrix, planar_times_matrix, sub, to_mont, Ext, PlanarMats, DEG, DOMAIN, P, P64, R_MOD_P};

/// Montgomery form of `-1`.
const NEG_R_MOD_P: u32 = P - R_MOD_P;
use crate::ntt::{encode, RowEval};
use crate::proof::{Column, Proof};
use crate::sha256::Sha256;
use crate::transcript::Transcript;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

pub struct Params { pub k: usize, pub n: usize, pub t: usize }
pub const LIGERO: Params = Params { k: 4096, n: 16384, t: 192 };

pub struct Segment<'a> { pub name: String, pub circ: &'a Circuit, pub layers: &'a [Layer], pub units: usize }
impl<'a> Segment<'a> { pub fn ncols(&self) -> usize { self.circ.ncols } }

/// One linked quantity of the v2 chain (`chain.txt`: `link <in|-> <out> <epi> <init> <zero_nonlast>`,
/// `gpu/prover.py::Link`, PROTOCOL 14.3).  With `col_c`: `X[(v,0),c] = init`, `X[(v,i),c] = X[(v,i-1),y]`,
/// `X_epi[v,epi] = X[(v,steps-1),y]`; without: the epilogue link only and, if `zero_nonlast`, `X[(v,i),y] = 0` for
/// `i < steps-1`.
#[derive(Clone, Debug)]
pub struct Link { pub col_c: Option<usize>, pub col_y: usize, pub epi: usize, pub init: u32, pub zero_nonlast: bool }
impl Link {
    pub fn constraints_per_vu(&self, steps: usize) -> usize {
        if self.col_c.is_some() { steps + 1 } else { 1 + if self.zero_nonlast { steps - 1 } else { 0 } }
    }
}

/// The VU chain (`gpu/prover.py::Chain`).  v1: the word chain on limbed words (`col_c[j]` <-> `col_y[j]` <->
/// `epi_y32[j]`, limb-major within a step, `links == None`); v2: the `chain.txt` link list, link by link
/// (`add_chain_links`), with the file's SHA-256 absorbed in the preamble.
pub struct Chain {
    pub steps: usize, pub col_c: Vec<usize>, pub col_y: Vec<usize>, pub epi_y32: Vec<usize>, pub epi_y16: usize, pub public_y16: Vec<u32>,
    pub links: Option<Vec<Link>>, pub spec_hash: Option<[u8; 32]>,
}
impl Chain {
    pub fn limbs(&self) -> usize { self.col_c.len() }
    pub fn constraints_per_vu(&self) -> usize {
        match &self.links {
            None => (self.steps + 1) * self.limbs() + 1,
            Some(ls) => 1 + ls.iter().map(|l| l.constraints_per_vu(self.steps)).sum::<usize>(),
        }
    }
}

pub struct Statement<'a> { pub segs: Vec<Segment<'a>>, pub chain: Chain }
impl<'a> Statement<'a> {
    /// Every table any segment queries: segment 0's in order, then the later segments' new ones
    /// (`gpu/prover.py::union_tables`; a v1 statement's epilogue adds none).
    pub fn tables(&self) -> Vec<&'a Table> {
        let mut out: Vec<&'a Table> = Vec::new();
        for s in &self.segs {
            for t in &s.circ.tables { if !out.iter().any(|u| u.name == t.name) { out.push(t); } }
        }
        out
    }
    pub fn vus(&self) -> usize { self.chain.public_y16.len() }
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub bytes_read: usize, pub slots: u64, pub msgs: usize, pub committed_elements: usize, pub rows: usize,
    pub t_parse: f64, pub t_transcript: f64, pub t_konst: f64, pub t_merkle: f64, pub t_rows: f64, pub t_total: f64, pub t_rows_fill_cpu: f64, pub t_rows_eval_cpu: f64,
    pub threads: usize, pub opened: Vec<usize>,
}

struct Reader<'a> { msgs: &'a [Ext], pos: usize }
impl<'a> Reader<'a> {
    fn next(&mut self) -> Result<Ext, String> {
        let v = self.msgs.get(self.pos).copied().ok_or_else(|| "proof truncated".to_string())?;
        self.pos += 1;
        Ok(v)
    }
    fn take(&mut self, n: usize) -> Result<Vec<Ext>, String> { (0..n).map(|_| self.next()).collect() }
}

// ---------------------------------------------------------------------------------------------------------------------
// statement-derived shapes
// ---------------------------------------------------------------------------------------------------------------------

fn table_queries<'a>(circ: &'a Circuit, name: &str) -> Vec<&'a Query> { circ.table_queries(name) }

fn hypercube_dim(stmt: &Statement, t: &Table) -> usize {
    let n_wit: usize = stmt.segs.iter().map(|s| s.units * table_queries(s.circ, &t.name).len()).sum();
    log2ceil(n_wit + t.nrows())
}

fn x_layout(stmt: &Statement) -> (Vec<usize>, Vec<usize>, usize) {
    let mut seg_offs = Vec::new();
    let mut l = 0;
    for s in &stmt.segs { seg_offs.push(l); l += s.units * s.ncols(); }
    let mut offs = Vec::new();
    for t in stmt.tables() { offs.push(l); l += t.nrows(); }
    (seg_offs, offs, l)
}

fn n_functionals(stmt: &Statement) -> usize { 2 * stmt.segs.len() + stmt.tables().len() + 1 }

/// The number of extension messages an honest proof of this statement carries (nothing else is accepted).
pub fn expected_msgs(stmt: &Statement) -> usize {
    let mut n = 0;
    for t in stmt.tables() {
        let d = hypercube_dim(stmt, t);
        n += 1 + (0..d).map(|k| 4 * k + 4).sum::<usize>();
    }
    for s in &stmt.segs {
        let b = log2ceil(s.units);
        for layer in s.layers { n += 4 * b + 3 * layer.s_in + 2 + 3 * layer.s_in + 1; }
    }
    n
}

// ---------------------------------------------------------------------------------------------------------------------
// transcript preamble, LogUp, GKR (the cheap, strictly sequential part)
// ---------------------------------------------------------------------------------------------------------------------

struct TableCh { z: Ext, beta: Ext }

fn start_transcript(stmt: &Statement, root: &[u8; 32]) -> (Transcript, Vec<TableCh>, Vec<(Vec<Ext>, Vec<Ext>)>) {
    let mut tr = Transcript::new(DOMAIN);
    for s in &stmt.segs {
        tr.absorb(&s.circ.hash);
        tr.absorb(&(s.units as u64).to_le_bytes());
    }
    tr.absorb(root);
    tr.absorb(&(stmt.chain.steps as u64).to_le_bytes());
    let mut buf = Vec::with_capacity(4 * stmt.chain.public_y16.len());
    for y in &stmt.chain.public_y16 { buf.extend_from_slice(&y.to_le_bytes()); }
    tr.absorb(&buf);
    if let Some(h) = &stmt.chain.spec_hash { tr.absorb(h); }
    let mut chs = Vec::new();
    for t in stmt.tables() {
        let z = tr.challenge();
        let beta = if t.cols > 1 { tr.challenge() } else { Ext::ONE };
        chs.push(TableCh { z, beta });
    }
    let mut pts = Vec::new();
    for s in &stmt.segs {
        let z0 = tr.challenges(s.layers[0].s_out);
        let gamma = tr.challenges(log2ceil(s.units));
        pts.push((z0, gamma));
    }
    (tr, chs, pts)
}

/// `rho` in the top-bit-first convention of `eq_table` / `SplitEq` (`rho[0]` on the leaf index's top bit).
struct LeafClaim { rho: Vec<Ext>, lam: Ext, value: Ext }

/// `--trace-logup`: print every level's point, messages and final-check operands (lane a-protocol-diff).
pub static TRACE_LOGUP: AtomicBool = AtomicBool::new(false);

fn fmt_ext(x: &Ext) -> String { format!("[{}]", x.0.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",")) }
fn fmt_exts(xs: &[Ext]) -> String { format!("[{}]", xs.iter().map(fmt_ext).collect::<Vec<_>>().join(" ")) }

/// One table's fractional-sum GKR, in the **low-bit-first** schedule of the a-gpu2 prover (`gpu/logup.py`,
/// `packed/logup_reference.py`): at level `k` round `i` binds bit `i` of the level index (pairs `(2y, 2y+1)`), and
/// the next level's point is `rho = [mu] ++ x*` -- `mu` on bit 0 (the children pairing), `x*[i]` on bit `i + 1`.
/// `eq(rho, x*)` pairs `rho[j]` with `x*[j]` (both on bit `j`).  The a-gpu prover (v2e) bound the top bit first
/// with `rho = x* ++ [mu]`; the two schedules are the same protocol under bit-reversal of the level index, but the
/// message values differ from level 2 on (the first level whose `eq(rho, x*)` has two factors), which is what this
/// verifier rejected until the switch (`a_protocol_diff.md`).  The leaf claim is returned top-bit-first (reversed).
fn verify_table(n: usize, tr: &mut Transcript, rd: &mut Reader, name: &str) -> Result<LeafClaim, String> {
    let trace = TRACE_LOGUP.load(Ordering::Relaxed);
    let root_q = rd.next()?;
    if root_q.is_zero() { return Err(format!("LogUp {name}: q_root is zero")); }
    tr.absorb_ext(root_q);
    let mut lam = tr.challenge();
    let mut rho: Vec<Ext> = Vec::new();
    let mut value = lam * root_q;
    if trace { println!("trace-logup {name}: n={n} q_root={} lambda_0={}", fmt_ext(&root_q), fmt_ext(&lam)); }
    for k in 0..n {
        let mut running = value;
        let mut x_star = Vec::with_capacity(k);
        for rnd in 0..k {
            let h = rd.take(4)?;
            if trace { println!("trace-logup {name} level {k} round {rnd}: running={} h={}", fmt_ext(&running), fmt_exts(&h)); }
            if h[0] + h[1] != running { return Err(format!("LogUp {name} level {k} round {rnd}: sum mismatch")); }
            tr.absorb_exts(&h);
            let r = tr.challenge();
            running = interpolate(&h, r);
            x_star.push(r);
            if trace { println!("trace-logup {name} level {k} round {rnd}: r={}", fmt_ext(&r)); }
        }
        let vals = rd.take(4)?;
        let (p0, p1, q0, q1) = (vals[0], vals[1], vals[2], vals[3]);
        let ex = eq_point(&rho, &x_star);
        let g = p0 * q1 + p1 * q0 + lam * (q0 * q1);
        if trace {
            let rho_top: Vec<Ext> = rho.iter().rev().copied().collect();
            let ex_top = eq_point(&rho_top, &x_star);
            println!("trace-logup {name} level {k}: rho(low-first)={} x*={} lambda={}", fmt_exts(&rho), fmt_exts(&x_star), fmt_ext(&lam));
            println!("trace-logup {name} level {k}: p0={} p1={} q0={} q1={}", fmt_ext(&p0), fmt_ext(&p1), fmt_ext(&q0), fmt_ext(&q1));
            println!("trace-logup {name} level {k}: g=p0q1+p1q0+lambda q0q1={} eq(rho,x*)={} eq*g={} running={} ok={}",
                     fmt_ext(&g), fmt_ext(&ex), fmt_ext(&(ex * g)), fmt_ext(&running), ex * g == running);
            println!("trace-logup {name} level {k}: [a-gpu top-first pairing] eq(rev rho,x*)*g={} ok={}", fmt_ext(&(ex_top * g)), ex_top * g == running);
        }
        if ex * g != running { return Err(format!("LogUp {name} level {k}: final check")); }
        tr.absorb_exts(&vals);
        let mu = tr.challenge();
        let pv = p0 + mu * (p1 - p0);
        let qv = q0 + mu * (q1 - q0);
        lam = tr.challenge();
        value = pv + lam * qv;
        if trace { println!("trace-logup {name} level {k}: mu={} lambda'={} value'={}", fmt_ext(&mu), fmt_ext(&lam), fmt_ext(&value)); }
        let mut next = Vec::with_capacity(k + 1);
        next.push(mu);
        next.extend_from_slice(&x_star);
        rho = next;
    }
    rho.reverse();
    Ok(LeafClaim { rho, lam, value })
}

struct LayerClaim { w: Vec<Ext>, copy_point: Vec<Ext>, value: Ext }
type FinalClaims = (Vec<Ext>, Ext, Vec<Ext>, Ext);

fn eval_terms_ext(terms: &[crate::circuit::Term], tab: &[Ext]) -> Ext {
    let mut acc = Ext::ZERO;
    for t in terms { acc += tab[t.idx as usize].scale(t.coef); }
    acc
}

fn verify_sumcheck_prod(vars: usize, mut running: Ext, tr: &mut Transcript, rd: &mut Reader, what: &str) -> Result<(Vec<Ext>, Ext), String> {
    let mut point = Vec::with_capacity(vars);
    for round in 0..vars {
        let s = rd.take(3)?;
        if s[0] + s[1] != running { return Err(format!("{what}: round {round} sum mismatch")); }
        tr.absorb_exts(&s);
        let r = tr.challenge();
        running = interpolate(&s, r);
        point.push(r);
    }
    Ok((point, running))
}

fn verify_layer(seg: &str, layer: &Layer, b: usize, claim: &LayerClaim, tr: &mut Transcript, rd: &mut Reader, last: bool) -> Result<(LayerClaim, Option<FinalClaims>), String> {
    let name = format!("{seg}/{}", layer.name);
    let mut running = claim.value;
    let mut c_star = Vec::with_capacity(b);
    for round in 0..b {
        let h = rd.take(4)?;
        if h[0] + h[1] != running { return Err(format!("{name}: phase-1 round {round} sum mismatch")); }
        tr.absorb_exts(&h);
        let r = tr.challenge();
        running = interpolate(&h, r);
        c_star.push(r);
    }
    let e_star = eq_point(&claim.copy_point, &c_star);
    let g_n = layer.gates.len();
    if claim.w.len() < g_n { return Err(format!("{name}: claim width {} < gates {g_n}", claim.w.len())); }
    let wp: Vec<Ext> = claim.w[..g_n].iter().map(|x| *x * e_star).collect();
    let (l_star, after_l) = verify_sumcheck_prod(layer.s_in, running, tr, rd, &format!("{name} phase-2a"))?;
    let vl = rd.next()?;
    let pl = rd.next()?;
    if vl * pl != after_l { return Err(format!("{name}: phase-2a final check")); }
    tr.absorb_exts(&[vl, pl]);
    let (r_star, after_r) = verify_sumcheck_prod(layer.s_in, pl, tr, rd, &format!("{name} phase-2b"))?;
    let vr = rd.next()?;
    tr.absorb_ext(vr);
    let el = eq_table(&l_star);
    let er = eq_table(&r_star);
    let mut rt = Ext::ZERO;
    for (g, gate) in layer.gates.iter().enumerate() {
        let ga: &Gate = gate;
        rt += wp[g] * eval_terms_ext(&ga.a, &el) * eval_terms_ext(&ga.b, &er);
    }
    if vr * rt != after_r { return Err(format!("{name}: phase-2b final check (wiring)")); }
    if last { return Ok((LayerClaim { w: vec![], copy_point: c_star, value: Ext::ZERO }, Some((l_star, vl, r_star, vr)))); }
    let alpha = tr.challenge();
    let nw: Vec<Ext> = el.iter().zip(&er).map(|(x, y)| *x + alpha * *y).collect();
    Ok((LayerClaim { w: nw, copy_point: c_star, value: vl + alpha * vr }, None))
}

// ---------------------------------------------------------------------------------------------------------------------
// the batched functional <a, X> = b, built row-chunk by row-chunk
// ---------------------------------------------------------------------------------------------------------------------

/// `eq(rho, x)` over `2^n` leaves as `hi[x >> lo_bits] * lo[x & mask]`.
struct SplitEq { hi: Vec<Ext>, lo: Vec<Ext>, lo_bits: u32 }
impl SplitEq {
    fn new(rho: &[Ext]) -> SplitEq {
        let n = rho.len();
        let lo_bits = n.min(14);
        SplitEq { hi: eq_table(&rho[..n - lo_bits]), lo: eq_table(&rho[n - lo_bits..]), lo_bits: lo_bits as u32 }
    }
    #[inline(always)]
    fn at(&self, x: usize) -> Ext { self.hi[x >> self.lo_bits] * self.lo[x & ((1usize << self.lo_bits) - 1)] }
    #[inline(always)]
    fn mask(&self) -> usize { (1usize << self.lo_bits) - 1 }
}

/// One table's lookup claim as data for the per-unit scatter.
struct LookupPlan {
    eq: SplitEq,
    /// rho_t * lambda_t (the scale of every column coefficient)
    rl: Ext,
    rho: Ext,
    /// per segment: leaf base, q_t, per query its column terms (idx, -sum_k beta^k coef) -- ext, or base when the
    /// table has a single column (beta^0 = 1, so the coefficient is just -coef)
    segs: Vec<LookupSeg>,
    single: bool,
}
/// `terms_rank`: the single-column (base) terms as `(rank in touched, coef)`; `terms_mat`: the multi-column terms as
/// `(idx, matrix of coef)` so the scatter is one matrix product per term.
struct LookupSeg { base: usize, q_t: usize, terms_mat: Vec<Vec<(usize, [u32; DEG * DEG])>>, /// CSR: query j's terms are `rank_items[rank_offs[j]..rank_offs[j + 1]]`
    rank_offs: Vec<u32>, rank_items: Vec<(u32, u32)> /* (rank, coef in Montgomery form) */, touched: Vec<usize> }

/// Per-thread scratch of the fill: planar accumulators and outputs (`[coord * n + col]`).
pub struct FillScratch { row: Vec<Ext>, planar: Vec<u32>, acc: Vec<u32>, flush: Vec<u32>, /// per-phase nanoseconds (VERITY_GKR_PROFILE=1): input claims, single-column lookups, ext lookups, chain
    pub prof: [u64; 5], profile: bool }
impl FillScratch { pub fn new() -> FillScratch { FillScratch { row: Vec::new(), planar: Vec::new(), acc: Vec::new(), flush: Vec::new(), prof: [0; 5], profile: std::env::var("VERITY_GKR_PROFILE").map(|v| v == "1").unwrap_or(false) } } }

struct Functional<'a> {
    stmt: &'a Statement<'a>,
    tables: Vec<&'a Table>,
    seg_offs: Vec<usize>,
    table_offs: Vec<usize>,
    l: usize,
    /// per segment: eq(c*, u) over the copies, and V[col] = rho_l eq(l*, col) + rho_r eq(r*, col) (as planar matrices)
    eq_c: Vec<Vec<Ext>>,
    v_col: Vec<Vec<Ext>>,
    v_mats: Vec<PlanarMats>,
    lookups: Vec<LookupPlan>,
    /// chain: rho_chain * geo^(v * per_vu) per VU, geo^k for k < per_vu
    chain_v: Vec<Ext>,
    chain_k: Vec<Ext>,
    per_vu: usize,
}

/// The R16-style flush: `out[touched[r]] += (rl * hi[block]) * acc[r]` for every touched column, as one planar
/// matrix product over the ranks (`acc` interleaved `[r * 6 + i]`, canonical; transposed into `flush`, which also
/// receives the products); `acc` is zeroed.
fn flush_single(lp: &LookupPlan, ls: &LookupSeg, h: usize, acc: &mut [u32], flush: &mut [u32], out: &mut [Ext]) {
    let nt = ls.touched.len();
    let g = mul_matrix(lp.rl * lp.eq.hi[h]);
    let (planar, prod) = flush.split_at_mut(DEG * nt);
    for r in 0..nt { for i in 0..DEG { planar[i * nt + r] = acc[r * DEG + i]; } }
    planar_times_matrix(planar, nt, &g, prod);
    for (r, &c) in ls.touched.iter().enumerate() {
        out[c] += Ext(std::array::from_fn(|d| prod[d * nt + r]));
    }
    for a in acc.iter_mut() { *a = 0; }
}

impl<'a> Functional<'a> {
    /// The unit-local part of `a` for unit `u` of segment `si` (every column), into `sc.row` (length ncols).
    fn unit_row(&self, si: usize, u: usize, sc: &mut FillScratch) {
        let seg = &self.stmt.segs[si];
        let ncols = seg.ncols();
        let ec = self.eq_c[si][u];
        let profile = sc.profile;
        let mut tp = if profile { Some(Instant::now()) } else { None };
        let mut lap = |sc_prof: &mut [u64; 5], k: usize| { if let Some(t) = tp.as_mut() { let n = Instant::now(); sc_prof[k] += (n - *t).as_nanos() as u64; *t = n; } };
        let FillScratch { row, planar, acc, flush, prof, .. } = sc;
        // input claims: out[c] = ec * v[c], vectorised over the columns through the planar matrices of v
        planar.resize(DEG * ncols, 0);
        self.v_mats[si].matvec(&ec, planar);
        row.resize(ncols, Ext::ZERO);
        for c in 0..ncols { row[c] = Ext(std::array::from_fn(|d| planar[d * ncols + c])); }
        lap(prof, 0);
        let out = &mut row[..];
        for lp in &self.lookups {
            let ls = &lp.segs[si];
            if ls.q_t == 0 { continue; }
            let x0 = ls.base + u * ls.q_t;
            let mask = lp.eq.mask();
            if lp.single {
                // acc[rank] = sum_j lo[x_j & mask] * coef (base, planar over the touched columns); flushed per hi-block
                // with the matrix of rl * hi[block]
                let nt = ls.touched.len();
                acc.clear();
                acc.resize(DEG * nt, 0);
                flush.resize(2 * DEG * nt, 0);
                let mut hi_cur = x0 >> lp.eq.lo_bits;
                for j in 0..ls.q_t {
                    let x = x0 + j;
                    let h = x >> lp.eq.lo_bits;
                    if h != hi_cur {
                        lap(prof, 1);
                        flush_single(lp, ls, hi_cur, acc, flush, out);
                        lap(prof, 4);
                        hi_cur = h;
                    }
                    let e = &lp.eq.lo[x & mask];
                    for &(rank, coef) in &ls.rank_items[ls.rank_offs[j] as usize..ls.rank_offs[j + 1] as usize] {
                        let rank = rank as usize;
                        let a: &mut [u32; DEG] = (&mut acc[rank * DEG..(rank + 1) * DEG]).try_into().unwrap();
                        // the coefficients are +-1 in practice (v1: 517 / 234 terms); anything else pays a mont_mul
                        if coef == R_MOD_P {
                            crate::lanes::acc6_add(a, &e.0);
                        } else if coef == NEG_R_MOD_P {
                            crate::lanes::acc6_sub(a, &e.0);
                        } else {
                            for i in 0..DEG { a[i] = add(a[i], mont_mul(e.0[i], coef)); }
                        }
                    }
                }
                lap(prof, 1);
                flush_single(lp, ls, hi_cur, acc, flush, out);
                lap(prof, 4);
            } else {
                let mut h_cur = usize::MAX;
                let mut rl_hi = Ext::ZERO;
                for j in 0..ls.q_t {
                    let x = x0 + j;
                    let h = x >> lp.eq.lo_bits;
                    if h != h_cur { rl_hi = lp.rl * lp.eq.hi[h]; h_cur = h; }
                    let e = rl_hi * lp.eq.lo[x & mask];
                    for (idx, m) in &ls.terms_mat[j] { out[*idx] += mat_apply(&e, m); }
                }
                lap(prof, 2);
            }
        }
        // chain (gpu/prover.py::add_chain ordering: k = i * L + j for the incoming-word constraints, then
        // k = steps * L + j for the epilogue links, then per_vu - 1 for the public word)
        let ch = &self.stmt.chain;
        if let Some(links) = &ch.links {
            // v2 (`gpu/prover.py::add_chain_links`): per VU, link by link, the constraint index k runs over
            // [steps incoming constraints | steps-1 zero_nonlast constraints | -], then the epilogue link; then y16
            let steps = ch.steps;
            if si == 0 {
                let (v_, i) = (u / steps, u % steps);
                let gv = self.chain_v[v_];
                let mut o = 0;
                for lk in links {
                    if let Some(cc) = lk.col_c {
                        out[cc] += gv * self.chain_k[o + i];
                        if i + 1 < steps { out[lk.col_y] -= gv * self.chain_k[o + i + 1]; }
                        o += steps;
                    } else if lk.zero_nonlast {
                        if i + 1 < steps { out[lk.col_y] += gv * self.chain_k[o + i]; }
                        o += steps - 1;
                    }
                    if i + 1 == steps { out[lk.col_y] -= gv * self.chain_k[o]; }
                    o += 1;
                }
                debug_assert_eq!(o + 1, self.per_vu);
            } else if si == 1 {
                let gv = self.chain_v[u];
                let mut o = 0;
                for lk in links {
                    o += lk.constraints_per_vu(steps) - 1;
                    out[lk.epi] += gv * self.chain_k[o];
                    o += 1;
                }
                out[ch.epi_y16] += gv * self.chain_k[self.per_vu - 1];
            }
            lap(prof, 3);
            return;
        }
        let lim = ch.limbs();
        if si == 0 {
            let (v_, i) = (u / ch.steps, u % ch.steps);
            let gv = self.chain_v[v_];
            for j in 0..lim {
                out[ch.col_c[j]] += gv * self.chain_k[i * lim + j];
                let k = if i + 1 < ch.steps { (i + 1) * lim + j } else { ch.steps * lim + j };
                out[ch.col_y[j]] -= gv * self.chain_k[k];
            }
        } else if si == 1 {
            let gv = self.chain_v[u];
            for j in 0..lim { out[ch.epi_y32[j]] += gv * self.chain_k[ch.steps * lim + j]; }
            out[ch.epi_y16] += gv * self.chain_k[self.per_vu - 1];
        }
        lap(prof, 3);
    }

    /// `a[lo..hi]` into `buf` (zeroed by the caller).
    fn fill(&self, lo: usize, hi: usize, buf: &mut [Ext], sc: &mut FillScratch) {
        for (si, seg) in self.stmt.segs.iter().enumerate() {
            let (off, ncols, units) = (self.seg_offs[si], seg.ncols(), seg.units);
            let end = off + units * ncols;
            if hi <= off || lo >= end { continue; }
            let u_lo = lo.saturating_sub(off) / ncols;
            let u_hi = ((hi.min(end) - off) + ncols - 1) / ncols;
            for u in u_lo..u_hi.min(units) {
                self.unit_row(si, u, sc);
                let base = off + u * ncols;
                let c_lo = lo.saturating_sub(base).min(ncols);
                let c_hi = hi.saturating_sub(base).min(ncols);
                for c in c_lo..c_hi { buf[base + c - lo] += sc.row[c]; }
            }
        }
        for (ti, lp) in self.lookups.iter().enumerate() {
            let t = self.tables[ti];
            let off = self.table_offs[ti];
            let end = off + t.nrows();
            if hi <= off || lo >= end { continue; }
            let tab_base: usize = lp.segs.iter().zip(&self.stmt.segs).map(|(s, seg)| s.q_t * seg.units).sum();
            for idx in lo.max(off)..hi.min(end) {
                let v = idx - off;
                buf[idx - lo] -= lp.rho * lp.eq.at(tab_base + v);
            }
        }
    }
}

/// `konst_t` and `b` (the value side of the functional), parallel over `eq` high blocks.
fn lookup_konst(stmt: &Statement, lp: &LookupPlan, t: &Table, ch: &TableCh, c: &LeafClaim, threads: usize) -> Ext {
    let lo_bits = lp.eq.lo_bits;
    let mask = lp.eq.mask();
    // witness leaves: sum_{s,q} K_{s,q} S_{s,q},  S_{s,q} = sum over the leaves of query q in segment s of eq(rho, x)
    let mut konst = Ext::ZERO;
    let mut sum_e = Ext::ZERO; // sum of eq over every non-padding leaf
    for (si, ls) in lp.segs.iter().enumerate() {
        if ls.q_t == 0 { continue; }
        let seg = &stmt.segs[si];
        let start = ls.base;
        let end = ls.base + seg.units * ls.q_t;
        let h_lo = start >> lo_bits;
        let h_hi = (end - 1) >> lo_bits;
        let nb = h_hi - h_lo + 1;
        let counter = AtomicUsize::new(0);
        let q_t = ls.q_t;
        let parts: Vec<Vec<Ext>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads).map(|_| s.spawn(|| {
                let mut sq = vec![Ext::ZERO; q_t];
                let mut tq = vec![Ext::ZERO; q_t];
                loop {
                    let bi = counter.fetch_add(1, Ordering::Relaxed);
                    if bi >= nb { break; }
                    let h = h_lo + bi;
                    let xs = start.max(h << lo_bits);
                    let xe = end.min((h + 1) << lo_bits);
                    for x in tq.iter_mut() { *x = Ext::ZERO; }
                    let mut q = (xs - start) % q_t;
                    for x in xs..xe {
                        tq[q] += lp.eq.lo[x & mask];
                        q += 1;
                        if q == q_t { q = 0; }
                    }
                    let eh = lp.eq.hi[h];
                    for q in 0..q_t { if !tq[q].is_zero() { sq[q] += eh * tq[q]; } }
                }
                sq
            })).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let queries = table_queries(seg.circ, &t.name);
        let bs = betas(t, ch);
        for q in 0..q_t {
            let mut s_q = Ext::ZERO;
            for p in &parts { s_q += p[q]; }
            let mut cst = ch.z;
            for (kk, lin) in queries[q].cols.iter().enumerate() { cst -= bs[kk].scale(lin.konst); }
            konst += s_q * (Ext::ONE + c.lam * cst); // p = 1 for every query leaf
            sum_e += s_q;
        }
    }
    let tab_base: usize = lp.segs.iter().zip(&stmt.segs).map(|(s, seg)| s.q_t * seg.units).sum();
    let bs = betas(t, ch);
    let mut r = vec![0u32; t.cols];
    for v in 0..t.nrows() {
        t.row(v, &mut r);
        let e = lp.eq.at(tab_base + v);
        let mut comb = Ext::ZERO;
        for (k, x) in r.iter().enumerate() { comb += bs[k].scale(*x); }
        konst += c.lam * e * (ch.z - comb);
        sum_e += e;
    }
    // padding leaves (0, 1): lambda * sum of eq over the padding = lambda * (1 - sum over the rest)
    konst += c.lam * (Ext::ONE - sum_e);
    konst
}

fn betas(t: &Table, ch: &TableCh) -> Vec<Ext> {
    let mut v = Vec::with_capacity(t.cols);
    let mut b = Ext::ONE;
    for _ in 0..t.cols { v.push(b); b = b * ch.beta; }
    v
}

// ---------------------------------------------------------------------------------------------------------------------
// Ligero
// ---------------------------------------------------------------------------------------------------------------------

fn hash_pair(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] { let mut s = Sha256::new(); s.update(a); s.update(b); s.finalize() }

fn verify_path(root: &[u8; 32], j: usize, col: &Column) -> bool {
    let mut h = Sha256::new();
    let mut buf = Vec::with_capacity(4 * col.values.len());
    for v in &col.values { buf.extend_from_slice(&v.to_le_bytes()); }
    h.update(&buf);
    let mut cur = h.finalize();
    let mut idx = j;
    for sib in &col.path {
        cur = if idx & 1 == 0 { hash_pair(&cur, sib) } else { hash_pair(sib, &cur) };
        idx >>= 1;
    }
    &cur == root
}

fn open_set(tr: &mut Transcript, n: usize, t: usize) -> Vec<usize> {
    let mut set = Vec::with_capacity(t);
    while set.len() < t {
        let c = tr.challenge();
        let j = (c.0[0] as u64 % n as u64) as usize;
        if !set.contains(&j) { set.push(j); }
    }
    set
}

// ---------------------------------------------------------------------------------------------------------------------
// the verifier
// ---------------------------------------------------------------------------------------------------------------------

pub fn verify(stmt: &Statement, proof_bytes: &[u8], p: &Params, threads: usize) -> (Result<(), String>, Stats) {
    let mut st = Stats { threads, ..Default::default() };
    let t_all = Instant::now();
    let r = verify_inner(stmt, proof_bytes, p, threads, &mut st);
    st.t_total = t_all.elapsed().as_secs_f64();
    (r, st)
}

fn verify_inner(stmt: &Statement, proof_bytes: &[u8], p: &Params, threads: usize, st: &mut Stats) -> Result<(), String> {
    // ---- statement shapes
    if stmt.segs.len() != 2 { return Err("statement: two segments (units, epilogue) expected".into()); }
    let ch = &stmt.chain;
    if stmt.segs[0].units != ch.steps * ch.public_y16.len() || stmt.segs[1].units != ch.public_y16.len() { return Err("statement: unit/epilogue/public counts disagree".into()); }
    let (seg_offs, table_offs, l) = x_layout(stmt);
    let rows = l.div_ceil(p.k);
    let depth = p.n.trailing_zeros() as usize;
    let n_msgs = expected_msgs(stmt);
    st.committed_elements = l;
    st.rows = rows;

    // ---- parse (the only reads of the file's counts; all compared to the statement)
    let t0 = Instant::now();
    let proof = Proof::parse(proof_bytes, n_msgs, p.k, p.t, rows, depth)?;
    if proof.msgs.len() != n_msgs { return Err(format!("proof: {} messages, statement needs {n_msgs}", proof.msgs.len())); }
    let op = proof.opening.as_ref().ok_or("missing opening")?;
    if op.w.len() != p.k || op.qc.len() != 2 * p.k - 1 || op.columns.len() != p.t { return Err("ligero: malformed opening".into()); }
    st.bytes_read = proof.bytes_read;
    st.msgs = proof.msgs.len();
    st.t_parse = t0.elapsed().as_secs_f64();

    // ---- transcript replay: LogUp, GKR
    let t1 = Instant::now();
    let (mut tr, chs, pts) = start_transcript(stmt, &proof.root);
    let mut rd = Reader { msgs: &proof.msgs, pos: 0 };
    let mut lookup_claims: Vec<(usize, LeafClaim)> = Vec::new();
    for t in stmt.tables() {
        let n = hypercube_dim(stmt, t);
        let lc = verify_table(n, &mut tr, &mut rd, &t.name)?;
        lookup_claims.push((n, lc));
    }
    let mut finals: Vec<(FinalClaims, Vec<Ext>)> = Vec::new();
    for (si, seg) in stmt.segs.iter().enumerate() {
        let b = log2ceil(seg.units);
        let mut claim = LayerClaim { w: eq_table(&pts[si].0), copy_point: pts[si].1.clone(), value: Ext::ZERO };
        let nl = seg.layers.len();
        for (i, layer) in seg.layers.iter().enumerate() {
            let last = i == nl - 1;
            let (next, fin) = verify_layer(&seg.name, layer, b, &claim, &mut tr, &mut rd, last)?;
            if let Some(f) = fin { finals.push((f, next.copy_point.clone())); }
            claim = next;
        }
    }
    if rd.pos != proof.msgs.len() { return Err("trailing proof data".into()); }
    let rhos = tr.challenges(n_functionals(stmt));
    let geo = tr.challenge();
    st.t_transcript = t1.elapsed().as_secs_f64();

    // ---- the functional's structure and its value b
    let t2 = Instant::now();
    let mut b = Ext::ZERO;
    let mut eq_c = Vec::new();
    let mut v_col = Vec::new();
    let mut ri = 0;
    for (si, seg) in stmt.segs.iter().enumerate() {
        let ((l_star, vl, r_star, vr), c_star) = &finals[si];
        let ec = eq_table(c_star);
        let el = eq_table(l_star);
        let er = eq_table(r_star);
        let s_c = seg.layers[seg.layers.len() - 1].s_in;
        if el.len() != 1 << s_c || seg.ncols() > el.len() || seg.units > ec.len() { return Err(format!("{}: input claim shape", seg.name)); }
        v_col.push((0..seg.ncols()).map(|c| rhos[ri] * el[c] + rhos[ri + 1] * er[c]).collect::<Vec<Ext>>());
        eq_c.push(ec);
        b += rhos[ri] * *vl + rhos[ri + 1] * *vr;
        ri += 2;
    }
    let mut lookups = Vec::new();
    for (ti, t) in stmt.tables().iter().enumerate() {
        let (n, lc) = &lookup_claims[ti];
        if lc.rho.len() != *n { return Err("lookup claim dimension".into()); }
        let rho = rhos[ri];
        ri += 1;
        let eq = SplitEq::new(&lc.rho);
        let bs = betas(t, &chs[ti]);
        let mut segs = Vec::new();
        let mut base = 0;
        for seg in &stmt.segs {
            let qs = table_queries(seg.circ, &t.name);
            let mut terms_ext = Vec::new();
            let mut terms_base = Vec::new();
            let mut touched = std::collections::BTreeSet::new();
            let mut per_query: Vec<std::collections::BTreeMap<usize, Ext>> = Vec::new();
            for qu in &qs {
                let mut m: std::collections::BTreeMap<usize, Ext> = std::collections::BTreeMap::new();
                for (kk, lin) in qu.cols.iter().enumerate() {
                    for tm in &lin.terms { *m.entry(tm.idx as usize).or_insert(Ext::ZERO) -= bs[kk].scale(tm.coef); }
                }
                for k in m.keys() { touched.insert(*k); }
                per_query.push(m);
            }
            let touched: Vec<usize> = touched.into_iter().collect();
            let rank_of: std::collections::HashMap<usize, usize> = touched.iter().enumerate().map(|(r, &c)| (c, r)).collect();
            for m in per_query {
                terms_base.push(m.iter().map(|(&i, e)| (rank_of[&i], e.0[0])).collect::<Vec<_>>());
                terms_ext.push(m.into_iter().map(|(i, e)| (i, mul_matrix(e))).collect::<Vec<_>>());
            }
            let mut rank_offs = vec![0u32];
            let mut rank_items = Vec::new();
            // coefficients in Montgomery form (`R_MOD_P` = 1): the accumulate is a `mont_mul`, not a `% p`
            for tb in &terms_base { for &(r, c) in tb { rank_items.push((r as u32, to_mont(c))); } rank_offs.push(rank_items.len() as u32); }
            segs.push(LookupSeg { base, q_t: qs.len(), terms_mat: terms_ext, rank_offs, rank_items, touched });
            base += seg.units * qs.len();
        }
        let lp = LookupPlan { eq, rl: rho * lc.lam, rho, segs, single: t.cols == 1 };
        let konst = lookup_konst(stmt, &lp, t, &chs[ti], lc, threads);
        b += rho * (lc.value - konst);
        lookups.push(lp);
    }
    // chain
    let rho_chain = rhos[ri];
    let per_vu = ch.constraints_per_vu();
    let mut chain_k = Vec::with_capacity(per_vu);
    let mut g = Ext::ONE;
    for _ in 0..per_vu { chain_k.push(g); g = g * geo; }
    let g_vu = g; // geo^per_vu
    let mut chain_v = Vec::with_capacity(stmt.vus());
    let mut cur = rho_chain;
    for _ in 0..stmt.vus() { chain_v.push(cur); cur = cur * g_vu; }
    for (v, y) in ch.public_y16.iter().enumerate() { b += (chain_v[v] * chain_k[per_vu - 1]).scale(*y); }
    if let Some(links) = &ch.links {
        // the init constants of the incoming links: b += sum_v rho geo^(v per_vu + o) * init
        let mut o = 0;
        for lk in links {
            if lk.col_c.is_some() && lk.init != 0 { for v in 0..stmt.vus() { b += (chain_v[v] * chain_k[o]).scale(lk.init); } }
            o += lk.constraints_per_vu(ch.steps);
        }
    }
    let v_mats: Vec<PlanarMats> = v_col.iter().map(|v| PlanarMats::new(v)).collect();
    let func = Functional { stmt, tables: stmt.tables(), seg_offs, table_offs, l, eq_c, v_col, v_mats, lookups, chain_v, chain_k, per_vu };
    st.t_konst = t2.elapsed().as_secs_f64();

    // ---- Ligero: r, w, qc, the functional value, the opened set
    let r = tr.challenge();
    tr.absorb_exts(&op.w);
    tr.absorb_exts(&op.qc);
    if (op.qc[0] + op.qc[p.k]).scale((p.k as u64 % P64) as u32) != b { return Err("ligero: linear functional value mismatch".into()); }
    let set = open_set(&mut tr, p.n, p.t);
    st.slots = tr.slots;
    st.opened = set.clone();

    // ---- Merkle paths, proximity (parallel over the opened columns)
    let t3 = Instant::now();
    let w_enc = encode(&op.w, p.n);
    let omega = crate::field::root_of_unity(p.n);
    let mut r_pow = Vec::with_capacity(rows);
    let mut cur = Ext::ONE;
    for _ in 0..rows { r_pow.push(cur); cur = cur * r; }
    let col_err: Vec<Option<String>> = {
        let counter = AtomicUsize::new(0);
        let results: Vec<Vec<(usize, Option<String>)>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads).map(|_| s.spawn(|| {
                let mut out = Vec::new();
                loop {
                    let oi = counter.fetch_add(1, Ordering::Relaxed);
                    if oi >= set.len() { break; }
                    let j = set[oi];
                    let col = &op.columns[oi];
                    let e = if col.values.len() != rows || col.path.len() != depth { Some("ligero: opening shape".to_string()) }
                    else if !verify_path(&proof.root, j, col) { Some(format!("ligero: merkle path {j}")) }
                    else {
                        let mut lhs = Ext::ZERO;
                        for i in 0..rows { lhs += r_pow[i].scale(col.values[i]); }
                        if lhs != w_enc[j] { Some(format!("ligero: proximity test at column {j}")) } else { None }
                    };
                    out.push((oi, e));
                }
                out
            })).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut v = vec![None; set.len()];
        for part in results { for (oi, e) in part { v[oi] = e; } }
        v
    };
    for e in col_err.iter().flatten() { return Err(e.clone()); }
    st.t_merkle = t3.elapsed().as_secs_f64();

    // ---- the linear test: rhs_j = sum_i a_i(eta_j) C[i, j], rows built and evaluated chunk by chunk
    let t4 = Instant::now();
    let re = RowEval::new(p.k, p.n, &set);
    const ROWS_PER_BATCH: usize = 8;
    const LANES: usize = ROWS_PER_BATCH * crate::field::DEG;
    let batched = std::env::var("VERITY_GKR_ROWWISE").map(|v| v != "1").unwrap_or(true);
    let chunk_rows = ROWS_PER_BATCH;
    let n_chunks = rows.div_ceil(chunk_rows);
    let counter = AtomicUsize::new(0);
    let (ns_fill, ns_eval) = (std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0));
    let prof_ns: [std::sync::atomic::AtomicU64; 5] = Default::default();
    let parts: Vec<Vec<Ext>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads).map(|_| s.spawn(|| {
            let mut rhs = vec![Ext::ZERO; set.len()];
            let mut buf = vec![Ext::ZERO; chunk_rows * p.k];
            let mut fsc = FillScratch::new();
            let mut ntt_scratch: Vec<u32> = Vec::new();
            let mut out = vec![Ext::ZERO; set.len()];
            let mut outb = vec![Ext::ZERO; chunk_rows * set.len()];
            let (mut tf, mut te) = (0u64, 0u64);
            loop {
                let ci = counter.fetch_add(1, Ordering::Relaxed);
                if ci >= n_chunks { break; }
                let r0 = ci * chunk_rows;
                let r1 = (r0 + chunk_rows).min(rows);
                let lo = r0 * p.k;
                let hi = (r1 * p.k).min(l);
                let ta = Instant::now();
                for x in buf.iter_mut() { *x = Ext::ZERO; }
                func.fill(lo, hi, &mut buf[..hi - lo], &mut fsc);
                let tb = Instant::now();
                if batched {
                    // the whole chunk in one batched evaluation (the tail chunk is zero-padded to `chunk_rows`)
                    re.eval_batch::<ROWS_PER_BATCH, LANES>(&buf, &mut ntt_scratch, &mut outb);
                    for i in r0..r1 {
                        let o = &outb[(i - r0) * set.len()..(i - r0 + 1) * set.len()];
                        for (oi, col) in op.columns.iter().enumerate() { rhs[oi] += o[oi].scale(col.values[i]); }
                    }
                } else {
                    for i in r0..r1 {
                        let row = &buf[(i - r0) * p.k..(i - r0 + 1) * p.k];
                        re.eval(row, &mut ntt_scratch, &mut out);
                        for (oi, col) in op.columns.iter().enumerate() { rhs[oi] += out[oi].scale(col.values[i]); }
                    }
                }
                tf += (tb - ta).as_nanos() as u64;
                te += tb.elapsed().as_nanos() as u64;
            }
            ns_fill.fetch_add(tf, Ordering::Relaxed);
            ns_eval.fetch_add(te, Ordering::Relaxed);
            for k in 0..5 { prof_ns[k].fetch_add(fsc.prof[k], Ordering::Relaxed); }
            rhs
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    if prof_ns[0].load(Ordering::Relaxed) > 0 {
        let g = |k: usize| prof_ns[k].load(Ordering::Relaxed) as f64 * 1e-9;
        eprintln!("fill profile (CPU s): input claims {:.3}, single-column lookups {:.3} (+ flush {:.3}), ext lookups {:.3}, chain {:.3}", g(0), g(1), g(4), g(2), g(3));
    }
    st.t_rows_fill_cpu = ns_fill.load(Ordering::Relaxed) as f64 * 1e-9;
    st.t_rows_eval_cpu = ns_eval.load(Ordering::Relaxed) as f64 * 1e-9;
    let mut rhs = vec![Ext::ZERO; set.len()];
    for part in parts { for (x, y) in rhs.iter_mut().zip(part) { *x += y; } }
    for (oi, &j) in set.iter().enumerate() {
        let eta = crate::field::pow(omega, j as u64);
        let mut qv = Ext::ZERO;
        for c in op.qc.iter().rev() { qv = qv.scale(eta) + *c; }
        if qv != rhs[oi] { return Err(format!("ligero: linear test at column {j}")); }
    }
    st.t_rows = t4.elapsed().as_secs_f64();
    let _ = (mul, sub, eval_lin);
    Ok(())
}
