//! flock-pure: the Flock backend's full committed relation, proved in one binary-field union proof with no link.
//!
//! The statement (`verity/flock-pure/v1`): for every VU `v` of an instance set's range, with x row `x_v` and W column
//! `w_v` (K words, `blake3-keyed/row/v2` row leaves under frame-v3 trees `a` and `b`) and output word `y_v` (tree `y`),
//! `y_v = epilogue(chain(x_v, w_v))`, where `chain` is the subcircuit's `units` tensor-core transition units from
//! c = +0. Flock proves ONE circuit over two gate types:
//! - the keyed-BLAKE3 compression (Flock's `blake3` table): every chunk of every row, chaining values wired block to
//!   block, the chunk's start value wired to the role's public key, every params word (counter, block_len, flags with
//!   KEYED_HASH) wired to a public constant, each chunk's output chaining value public;
//! - the lowered unit (`flock-unit-io/v1`, `verity_flock.lowering`, pinned by its sha256): its x and W input words ARE
//!   the message words of the compression hashing those bytes (unit `j` reads half `j % 2` of block `j / 2`), its
//!   accumulator input is the previous unit's output (the public zero word for unit 0), and the last unit's output word
//!   (the BF16 word for an epilogue relation, else the FP32 accumulator) is public.
//!
//! The verifier holds the instance set. It recomputes the three roots and Σ (relation, lowering digests, counts,
//! instance reference, roots), refuses a `Hello` for another Σ, and in `Commit` checks the public chunk values against
//! its own keyed-BLAKE3 row digests (parent compressions natively) and the public output words against its own words,
//! before any coin. Flock's coins are live (flock-128-r2: Fast100, two reps on one root, R1–R8 of flock-live).
//!
//!   flock-pure serve    --listen ADDR --out DIR --instances FILE --netlist FILE --pin SHA256
//!   flock-pure prove    --verifier ADDR --instances FILE --netlist FILE [--runs N] [--warm W]
//!   flock-pure selftest --instances FILE --netlist FILE [--netlist-other FILE]
//!
//! Output lines are `LIVE\t{json}` / `NEG\t{json}` / `SELFTEST\t{json}` / `SESSION\t{json}`.

use std::{
    array::from_fn,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex},
    time::Instant,
};

use flock_core::{
    circuit::builder::{CircuitShape, GateType, ShapeBuilder, SlotId, SlotWitness, Wire},
    lincheck::LincheckCircuit,
    pcs::{PcsParams, ligerito::{LigeritoProfile, embedded_initial_k_or_default}},
    proof::{LinkClaim, R1csProofCircuitMerged},
    union::publics_digest,
};
use flock_field::F128;
use flock_hash::blake3_compress;
use flock_live::*;
use flock_prover::{
    pcs::Commitment,
    prover::{UnionSlotProverInput, prove_fast_ligerito_union_circuit_linked},
    r1cs::BlockR1cs,
    r1cs_hashes::{
        blake3::{Compression, build_block_r1cs, generate_witness_batch_major_partial_into, io_schema},
        verity_unit::{UnitGate, UnitNet, UnitRow},
    },
    schedule::TableType,
    union::UnionInstance,
    verifier::verify_ligerito_union_circuit_linked,
};
use rayon::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const TABLE: &str = "pure";
const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const PARENT: u32 = 4;
const ROOT: u32 = 8;
const KEYED_HASH: u32 = 16;
const FRAME: &[u8] = b"veritor/protocol/merkle/frame/v3\0";
const SIGMA_TAG: &[u8] = b"verity/flock-pure/v1\0";

fn arg(k: &str, d: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned()).unwrap_or_else(|| d.to_string())
}

fn pack4(w: &[u32]) -> F128 {
    F128::new(w[0] as u64 | ((w[1] as u64) << 32), w[2] as u64 | ((w[3] as u64) << 32))
}
fn unpack4(v: F128) -> [u32; 4] {
    [v.lo as u32, (v.lo >> 32) as u32, v.hi as u32, (v.hi >> 32) as u32]
}
fn pack8(w: &[u32; 8]) -> [F128; 2] {
    [pack4(&w[..4]), pack4(&w[4..])]
}
fn unpack8(a: F128, b: F128) -> [u32; 8] {
    let (x, y) = (unpack4(a), unpack4(b));
    [x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]]
}
fn pack_params(counter: u64, block_len: u32, flags: u32) -> F128 {
    F128::new(counter, block_len as u64 | ((flags as u64) << 32))
}
fn f128_bytes(v: &[F128]) -> Vec<u8> {
    v.iter().flat_map(|x| x.lo.to_le_bytes().into_iter().chain(x.hi.to_le_bytes())).collect()
}
fn bytes_f128(b: &[u8]) -> Vec<F128> {
    b.chunks(16).map(|c| F128::new(u64::from_le_bytes(c[..8].try_into().unwrap()), u64::from_le_bytes(c[8..16].try_into().unwrap()))).collect()
}
fn le_words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn log2ceil(n: usize) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}
fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

/// `blake3-keyed/row/v2`'s key for a role: "verity/blake3-leaf/v1/" ‖ ("x" | "w"), zero padded to 32 bytes.
fn role_key(role: u8) -> [u8; 32] {
    let mut k = [0u8; 32];
    let p = b"verity/blake3-leaf/v1/";
    k[..p.len()].copy_from_slice(p);
    k[p.len()] = if role == 1 { b'x' } else { b'w' };
    k
}
fn key_words(role: u8) -> [u32; 8] {
    from_fn(|i| u32::from_le_bytes(role_key(role)[4 * i..4 * i + 4].try_into().unwrap()))
}

// ------------------------------------------------------------------------------------------------------------
// The instance file (verity_flock.instances) and the frame-v3 statement commitments
// ------------------------------------------------------------------------------------------------------------

#[derive(Clone)]
struct Instances {
    header: Value,
    relation: String,
    vus: usize,
    units: usize,
    row_bytes: usize,
    x: Vec<u8>,
    w: Vec<u8>,
    accs: Vec<u32>,
    out: Vec<u32>,
    y: Vec<u32>,
}

impl Instances {
    fn load(path: &str) -> Self {
        let b = std::fs::read(path).expect("instances");
        let nl = b.iter().position(|&c| c == b'\n').unwrap();
        let header: Value = serde_json::from_slice(&b[..nl]).unwrap();
        assert_eq!(header["format"], "flock-pure-instances/v1");
        let vus = header["vus"].as_u64().unwrap() as usize;
        let units = header["units"].as_u64().unwrap() as usize;
        let row_bytes = header["row_bytes"].as_u64().unwrap() as usize;
        let mut at = nl + 1;
        let mut take = |n: usize| {
            let s = b[at..at + n].to_vec();
            at += n;
            s
        };
        let x = take(vus * row_bytes);
        let w = take(vus * row_bytes);
        let accs = le_words(&take(vus * units * 4));
        let out = le_words(&take(vus * 4));
        let y = le_words(&take(vus * 4));
        assert_eq!(at, b.len(), "instance file length");
        Instances { relation: header["relation"].as_str().unwrap().to_string(), header, vus, units, row_bytes, x, w, accs, out, y }
    }
    fn row(&self, role: u8, v: usize) -> &[u8] {
        let src = if role == 1 { &self.x } else { &self.w };
        &src[v * self.row_bytes..(v + 1) * self.row_bytes]
    }
    fn hex(&self, k: &str, n: &str) -> Vec<u8> {
        unhex(self.header[k][n].as_str().unwrap())
    }
}

fn fv3_hash(tag: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(FRAME);
    h.update((tag.len() as u32).to_be_bytes());
    h.update(tag);
    for p in parts {
        h.update((p.len() as u64).to_be_bytes());
        h.update(p);
    }
    h.finalize().into()
}
fn uint(x: usize) -> Vec<u8> {
    let b = (x as u64).to_be_bytes();
    let z = b.iter().position(|&c| c != 0).unwrap_or(7);
    b[z..].to_vec()
}

/// A frame-v3 root over a range domain (`verity.ml.commitments.merkle.tree_levels`).
fn fv3_root(domain_id: &[u8], leaves: Vec<[u8; 32]>) -> [u8; 32] {
    let n = leaves.len();
    assert!(n > 0);
    let width = n.next_power_of_two();
    let mut level = leaves;
    level.extend((n..width).map(|r| fv3_hash(b"pad", &[domain_id, &uint(r)])));
    let mut depth = 0;
    while level.len() > 1 {
        level = level.par_chunks(2).enumerate().map(|(i, p)| fv3_hash(b"node", &[domain_id, &uint(depth), &uint(i), &p[0], &p[1]])).collect();
        depth += 1;
    }
    level[0]
}

/// The statement's commitments, computed from the values: keyed-BLAKE3 row digests, then the three frame-v3 roots.
struct Committed {
    digests: [Vec<[u8; 32]>; 2],
    roots: [[u8; 32]; 3],
}

fn commit(inst: &Instances) -> Committed {
    let digests: [Vec<[u8; 32]>; 2] = from_fn(|r| {
        let role = r as u8 + 1;
        let key = role_key(role);
        (0..inst.vus).into_par_iter().map(|v| *blake3::keyed_hash(&key, inst.row(role, v)).as_bytes()).collect()
    });
    let sch = |n: &str| inst.header["schemas"][n].as_str().unwrap().to_string();
    let roots: [[u8; 32]; 3] = from_fn(|t| {
        let n = ["a", "b", "y"][t];
        let dom = inst.hex("domain_ids", n);
        let s = sch(n);
        let leaves: Vec<[u8; 32]> = if t < 2 {
            digests[t].par_iter().enumerate().map(|(r, d)| fv3_hash(b"leaf", &[&dom, &uint(r), &uint(r), s.as_bytes(), d])).collect()
        } else {
            let nb = inst.header["y_bytes"].as_u64().unwrap() as usize;
            inst.y.par_iter().enumerate().map(|(r, &y)| fv3_hash(b"leaf", &[&dom, &uint(r), &uint(r), s.as_bytes(), &y.to_be_bytes()[4 - nb..]])).collect()
        };
        fv3_root(&dom, leaves)
    });
    Committed { digests, roots }
}

// ------------------------------------------------------------------------------------------------------------
// The statement shape
// ------------------------------------------------------------------------------------------------------------

/// One keyed-BLAKE3 compression as a circuit gate (schema: cv0, cv1, m0..m3, params → out_lo0, out_lo1, out_hi0, out_hi1).
struct Blake3Gate {
    nu: usize,
}

impl GateType for Blake3Gate {
    type Row = Compression;
    type Hint = ();
    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&build_block_r1cs(self.nu)).with_io_schema(io_schema())
    }
    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Compression {
        let cv = unpack8(inputs[0], inputs[1]);
        let mut m = [0u32; 16];
        for i in 0..4 {
            m[4 * i..4 * i + 4].copy_from_slice(&unpack4(inputs[2 + i]));
        }
        let (counter, block_len, flags) = (inputs[6].lo, inputs[6].hi as u32, (inputs[6].hi >> 32) as u32);
        let out = blake3_compress(&cv, &m, counter, block_len, flags);
        let lo = pack8(out[0..8].try_into().unwrap());
        let hi = pack8(out[8..16].try_into().unwrap());
        outputs.extend_from_slice(&[lo[0], lo[1], hi[0], hi[1]]);
        (cv, m, counter, block_len, flags)
    }
    fn witness(&self, _rows: &[Compression], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

/// Blocks per chunk of a row of `row_bytes` bytes.
fn chunk_blocks(row_bytes: usize) -> Vec<usize> {
    (0..row_bytes.div_ceil(1024)).map(|c| (row_bytes - 1024 * c).min(1024) / 64).collect()
}
fn chunk_flags(blk: usize, nb: usize) -> u32 {
    KEYED_HASH | if blk == 0 { CHUNK_START } else { 0 } | if blk + 1 == nb { CHUNK_END } else { 0 }
}

struct Pure {
    vus: usize,
    units: usize,
    row_bytes: usize,
    chunks: Vec<usize>,
    n_b3: usize,
    nu: usize,
    net: Arc<UnitNet>,
    shape: CircuitShape,
    s_b3: SlotId,
    s_unit: SlotId,
    b3_r1cs: BlockR1cs,
    unit_r1cs: BlockR1cs,
    fixed: Vec<F128>,
}

impl Pure {
    fn new(vus: usize, units: usize, row_bytes: usize, net: Arc<UnitNet>) -> Self {
        assert!(row_bytes % 64 == 0 && row_bytes > 1024, "a row is more than one BLAKE3 chunk of whole blocks (the digest is a parent)");
        assert_eq!(units * 32, row_bytes, "a unit reads 32 bytes of each row");
        let chunks = chunk_blocks(row_bytes);
        let blocks = row_bytes / 64;
        let n_b3 = 2 * vus * blocks;
        let nu = log2ceil(n_b3.max(vus * units)).max(8);
        let mut b = ShapeBuilder::new(nu);
        let s_b3 = b.slot(Blake3Gate { nu });
        let s_unit = b.slot(UnitGate { net: net.clone(), nu });
        let mut fixed = Vec::new();
        let mut fix = |b: &mut ShapeBuilder, v: F128| {
            fixed.push(v);
            b.fixed_public_input(v)
        };
        let keys: Vec<[Wire; 2]> = [1u8, 2].iter().map(|&r| pack8(&key_words(r)).map(|v| fix(&mut b, v))).collect();
        let zero = fix(&mut b, F128::ZERO);
        let params: Vec<Vec<Wire>> = chunks
            .iter()
            .enumerate()
            .map(|(c, &nb)| (0..nb).map(|blk| chunk_flags(blk, nb)).collect::<std::collections::BTreeSet<_>>().into_iter()
                .map(|f| fix(&mut b, pack_params(c as u64, 64, f))).collect())
            .collect();
        let param_of = |c: usize, blk: usize| -> Wire {
            let nb = chunks[c];
            let set: Vec<u32> = (0..nb).map(|k| chunk_flags(k, nb)).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
            params[c][set.iter().position(|&f| f == chunk_flags(blk, nb)).unwrap()]
        };
        // No independence islands (flock-link: an island's value buffer is quadratic in their count).
        for _ in 0..vus {
            let mut msgs: Vec<Vec<[Wire; 4]>> = Vec::new();
            let mut cvs: Vec<[Wire; 2]> = Vec::new();
            for role in 0..2 {
                let mut m_role = Vec::with_capacity(blocks);
                for (c, &nb) in chunks.iter().enumerate() {
                    let mut cv = keys[role];
                    for blk in 0..nb {
                        let m: [Wire; 4] = from_fn(|_| b.input());
                        let o = b.gate(s_b3, &[cv[0], cv[1], m[0], m[1], m[2], m[3], param_of(c, blk)]);
                        cv = [o[0], o[1]];
                        m_role.push(m);
                    }
                    cvs.push(cv);
                }
                msgs.push(m_role);
            }
            let mut acc = zero;
            let mut out = zero;
            for j in 0..units {
                let (blk, h) = (j / 2, 2 * (j % 2));
                let o = b.gate(s_unit, &[msgs[0][blk][h], msgs[0][blk][h + 1], msgs[1][blk][h], msgs[1][blk][h + 1], acc]);
                acc = o[0];
                out = *o.last().unwrap();
            }
            for cv in &cvs {
                b.publish(cv[0]);
                b.publish(cv[1]);
            }
            b.publish(out);
        }
        let shape = b.finish().expect("pure circuit");
        let b3_r1cs = build_block_r1cs(nu);
        let unit_r1cs = net.block_r1cs(nu);
        b3_r1cs.csc_lincheck_circuit();
        unit_r1cs.csc_lincheck_circuit();
        Pure { vus, units, row_bytes, chunks, n_b3, nu, net, shape, s_b3, s_unit, b3_r1cs, unit_r1cs, fixed }
    }

    fn for_instances(inst: &Instances, net: Arc<UnitNet>) -> Self {
        assert_eq!(net.relation, inst.relation, "the netlist lowers another relation than the instance set's");
        assert_eq!(net.out_cols.len() == 2, inst.header["epilogue"].as_bool().unwrap());
        Self::new(inst.vus, inst.units, inst.row_bytes, net)
    }

    fn union(&self) -> UnionInstance<'_> {
        UnionInstance::new(&self.shape.registry, self.shape.counts.clone())
    }

    fn params(&self, lig: LigeritoProfile) -> PcsParams {
        let u = self.union();
        let m = u.dense_m();
        let batch = embedded_initial_k_or_default(m, lig);
        PcsParams { m, log_inv_rate: lig.log_inv_rate(), log_batch_size: batch, profile: lig, num_lanes: u.commit_lanes(batch), merkle_hash: Default::default() }
    }

    fn spec(&self) -> TableSpec {
        let counts_le = self.shape.counts.iter().flat_map(|&n| (n as u64).to_le_bytes()).collect();
        TableSpec { name: TABLE.into(), binding: Binding::Circuit { registry_digest: self.shape.registry.digest(), counts_le, circuit_digest: self.shape.circuit.digest() } }
    }

    /// Σ: the relation, the lowering (netlist sha256, circuit and registry digests, counts), the instance reference and
    /// the three roots.
    fn sigma(&self, inst: &Instances, roots: &[[u8; 32]; 3]) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(SIGMA_TAG);
        h.update(inst.relation.as_bytes());
        h.update([0]);
        for x in [self.vus, self.units, self.row_bytes, self.nu] {
            h.update((x as u64).to_le_bytes());
        }
        h.update(self.net.sha256);
        h.update(self.shape.circuit.digest());
        h.update(self.shape.registry.digest());
        for &c in &self.shape.counts {
            h.update((c as u64).to_le_bytes());
        }
        h.update(serde_json::to_string(&inst.header["instances"]).unwrap().as_bytes());
        for r in roots {
            h.update(r);
        }
        h.finalize().into()
    }

    /// `shape.run`'s inputs (fixed words, then every block's message words).
    fn inputs(&self, inst: &Instances) -> Vec<F128> {
        let mut v = self.fixed.clone();
        for vu in 0..self.vus {
            for role in [1u8, 2] {
                for blk in inst.row(role, vu).chunks(64) {
                    let w = le_words(blk);
                    for j in 0..4 {
                        v.push(pack4(&w[4 * j..4 * j + 4]));
                    }
                }
            }
        }
        v
    }

    /// The honest rows of both slots and the public words, computed natively in the shape's instantiation order.
    fn rows(&self, inst: &Instances) -> (Vec<Compression>, Vec<UnitRow>, Vec<F128>) {
        let per: Vec<(Vec<Compression>, Vec<UnitRow>, Vec<F128>)> = (0..self.vus)
            .into_par_iter()
            .map(|v| {
                let mut b3 = Vec::with_capacity(2 * self.row_bytes / 64);
                let mut pubs = Vec::new();
                for role in [1u8, 2] {
                    let row = inst.row(role, v);
                    let mut off = 0;
                    for (c, &nb) in self.chunks.iter().enumerate() {
                        let mut cv = key_words(role);
                        for blk in 0..nb {
                            let m: [u32; 16] = le_words(&row[off..off + 64]).try_into().unwrap();
                            off += 64;
                            let f = chunk_flags(blk, nb);
                            let o = blake3_compress(&cv, &m, c as u64, 64, f);
                            b3.push((cv, m, c as u64, 64, f));
                            cv = o[..8].try_into().unwrap();
                        }
                        pubs.extend(pack8(&cv));
                    }
                }
                let (xr, wr) = (inst.row(1, v), inst.row(2, v));
                let units: Vec<UnitRow> = (0..self.units)
                    .map(|j| {
                        let x = le_words(&xr[32 * j..32 * j + 32]);
                        let w = le_words(&wr[32 * j..32 * j + 32]);
                        let c = if j == 0 { 0 } else { inst.accs[v * self.units + j - 1] };
                        [pack4(&x[..4]), pack4(&x[4..]), pack4(&w[..4]), pack4(&w[4..]), F128::new(c as u64, 0)]
                    })
                    .collect();
                pubs.push(F128::new(inst.out[v] as u64, 0));
                (b3, units, pubs)
            })
            .collect();
        let mut public = self.fixed.clone();
        let (mut b3, mut units) = (Vec::with_capacity(self.n_b3), Vec::with_capacity(self.vus * self.units));
        for (b, u, p) in per {
            b3.extend(b);
            units.extend(u);
            public.extend(p);
        }
        (b3, units, public)
    }

    fn circuits(&self) -> Vec<&dyn LincheckCircuit> {
        let mut c: Vec<(usize, &dyn LincheckCircuit)> = vec![
            (self.shape.registry_slot(self.s_b3), self.b3_r1cs.csc_lincheck_circuit()),
            (self.shape.registry_slot(self.s_unit), self.unit_r1cs.csc_lincheck_circuit()),
        ];
        c.sort_by_key(|x| x.0);
        c.into_iter().map(|x| x.1).collect()
    }

    fn prove(&self, b3: &[Compression], units: &[UnitRow], public: &[F128], lig: LigeritoProfile, hook: LinkHook<'_>, ch: &mut LiveChallenger) -> Vec<u8> {
        let nu = self.nu;
        let net = self.net.clone();
        let mut slots: Vec<(usize, UnionSlotProverInput<'_>)> = vec![
            (self.shape.registry_slot(self.s_b3), UnionSlotProverInput::in_place(move |dst| generate_witness_batch_major_partial_into(b3, nu, dst), self.b3_r1cs.csc_lincheck_circuit())),
            (self.shape.registry_slot(self.s_unit), UnionSlotProverInput::in_place(move |dst| net.witness_rows_into(units, nu, dst), self.unit_r1cs.csc_lincheck_circuit())),
        ];
        slots.sort_by_key(|x| x.0);
        let slots = slots.into_iter().map(|x| x.1).collect();
        let (proof, commitment, _) =
            prove_fast_ligerito_union_circuit_linked(&self.union(), &self.shape.circuit, public, &self.params(lig), slots, Vec::new(), hook, ch);
        bincode::serialize(&(commitment, proof)).unwrap()
    }

    /// The public words after the fixed ones, per VU: the x chunk values, the W chunk values, the output word.
    fn publics_per_vu(&self) -> usize {
        4 * self.chunks.len() + 1
    }
}

type LinkHook<'a> = &'a mut dyn FnMut(&Commitment) -> Vec<LinkClaim>;

fn parent_cv(key: &[u32; 8], l: &[u32; 8], r: &[u32; 8], root: bool) -> [u32; 8] {
    let mut m = [0u32; 16];
    m[..8].copy_from_slice(l);
    m[8..].copy_from_slice(r);
    blake3_compress(key, &m, 0, 64, KEYED_HASH | PARENT | if root { ROOT } else { 0 })[..8].try_into().unwrap()
}

/// BLAKE3's (keyed) tree over at least two chunk chaining values.
fn tree(key: &[u32; 8], cvs: &[[u32; 8]], root: bool) -> [u32; 8] {
    if cvs.len() == 1 {
        return cvs[0];
    }
    let left = 1usize << (usize::BITS - 1 - (cvs.len() - 1).leading_zeros());
    parent_cv(key, &tree(key, &cvs[..left], false), &tree(key, &cvs[left..], false), root)
}

fn digest_of(key: &[u32; 8], cvs: &[[u32; 8]]) -> [u8; 32] {
    let mut d = [0u8; 32];
    for (i, w) in tree(key, cvs, true).iter().enumerate() {
        d[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    d
}

fn domain(rep: usize) -> Vec<u8> {
    format!("flock-pure/fast100x2/rep{rep}").into_bytes()
}

// ------------------------------------------------------------------------------------------------------------
// The verifier
// ------------------------------------------------------------------------------------------------------------

struct PureVerifier {
    p: Arc<Pure>,
    digests: [Vec<[u8; 32]>; 2],
    out: Vec<u32>,
}

impl PureVerifier {
    fn public(&self, publics: &[u8]) -> Vec<F128> {
        let mut p = self.p.fixed.clone();
        p.extend(bytes_f128(publics));
        p
    }
}

impl StatementVerifier for PureVerifier {
    fn domain(&self, _table: &str, rep: usize) -> Vec<u8> {
        domain(rep)
    }

    fn statement_publics(&self, table: &str, publics: &[u8]) -> Result<Option<[u8; 32]>, String> {
        let p = &self.p;
        if table != TABLE {
            return Err(format!("C4: {table} is not in the statement"));
        }
        let per = p.publics_per_vu();
        if publics.len() != 16 * per * p.vus {
            return Err(format!("C4: {} bytes of public words, want {}", publics.len(), 16 * per * p.vus));
        }
        let w = bytes_f128(publics);
        let nc = p.chunks.len();
        let bad = (0..p.vus).into_par_iter().find_map_first(|v| {
            let s = &w[v * per..(v + 1) * per];
            for r in 0..2 {
                let cvs: Vec<[u32; 8]> = (0..nc).map(|c| unpack8(s[2 * (r * nc + c)], s[2 * (r * nc + c) + 1])).collect();
                if digest_of(&key_words(r as u8 + 1), &cvs) != self.digests[r][v] {
                    return Some(format!("C4: the public chunk values of VU {v}'s {} row do not hash to its leaf digest", ["x", "W"][r]));
                }
            }
            (s[per - 1] != F128::new(self.out[v] as u64, 0)).then(|| format!("C4: VU {v}'s public output word is not its committed output"))
        });
        match bad {
            Some(e) => Err(e),
            None => Ok(Some(publics_digest(&self.public(publics)))),
        }
    }

    fn verify(&self, _table: &str, _rep: usize, proof: &[u8], ch: &mut ReplayChallenger, link: &LinkCtx<'_>) -> Result<Vec<u8>, String> {
        let p = &self.p;
        let (commitment, proof): (Commitment, R1csProofCircuitMerged) = bincode::deserialize(proof).map_err(|e| format!("decode: {e}"))?;
        let pcs = p.params(LigeritoProfile::Fast100);
        if commitment.params != pcs {
            return Err("R7/F2: proof's PCS parameters differ from the pinned Fast100 parameters".into());
        }
        if !link.points.is_empty() || !link.y.is_empty() {
            return Err("this statement has no link claims".into());
        }
        let public = self.public(link.publics);
        let circs = p.circuits();
        let u = p.union();
        catch_unwind(AssertUnwindSafe(|| verify_ligerito_union_circuit_linked(&u, &p.shape.circuit, &public, &circs, &commitment, &proof, &pcs, ch, &[])))
            .map_err(|_| "verifier panicked".to_string())?
            .map_err(|e| format!("{e:?}").chars().take(160).collect::<String>())?;
        Ok(commitment.cap.as_flattened().to_vec())
    }
}

/// The verifier for the instance set it holds: its own commitments (checked against the file's, computed by the
/// Python reference), its own Σ, and the netlist pinned by sha256.
fn server(p: &Arc<Pure>, inst: &Instances) -> Server {
    let c = commit(inst);
    let sigma = p.sigma(inst, &c.roots);
    let mut cfg = SessionConfig::flock_128_r2(vec![p.spec()]);
    cfg.link = Some(LinkSpec { sigma: sigma.to_vec(), m: 0, points: 0 });
    Server::new(cfg, Box::new(PureVerifier { p: p.clone(), digests: c.digests, out: inst.out.clone() }))
}

fn check_roots(inst: &Instances) -> Result<[[u8; 32]; 3], String> {
    let c = commit(inst);
    for (t, n) in ["a", "b", "y"].iter().enumerate() {
        if c.roots[t].to_vec() != inst.hex("roots", n) {
            return Err(format!("root {n}: the native frame-v3 root differs from the reference's"));
        }
    }
    Ok(c.roots)
}

// ------------------------------------------------------------------------------------------------------------
// The prover's session, honest or not
// ------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum CommitMode {
    Honest,
    OtherRootB,
    OtherSigma,
    Skip,
}

#[derive(Clone)]
struct Plan {
    commit: CommitMode,
    publics_flip: Option<usize>,
    unit_tamper: Option<(usize, usize)>,
    forge_middle: bool,
    bad_params: Option<usize>,
    rep1_other_witness: bool,
    profile: [LigeritoProfile; 2],
    substitute: Option<Vec<Vec<u8>>>,
    sigma: Option<[u8; 32]>,
}

impl Plan {
    fn honest() -> Self {
        Plan { commit: CommitMode::Honest, publics_flip: None, unit_tamper: None, forge_middle: false, bad_params: None, rep1_other_witness: false,
               profile: [LigeritoProfile::Fast100; 2], substitute: None, sigma: None }
    }
}

fn call(t: &SharedTransport, r: Req) -> Result<Resp, String> {
    t.lock().unwrap().call(r)
}

fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

/// One session: commit the values (row digests, the three roots, Σ), build the rows, then two live reps.
fn session(p: &Pure, inst: &Instances, t: SharedTransport, plan: &Plan) -> (Value, Vec<Vec<u8>>) {
    let t0 = Instant::now();
    let c = commit(inst);
    let commit_s = t0.elapsed().as_secs_f64();
    let sigma = plan.sigma.unwrap_or_else(|| p.sigma(inst, &c.roots));
    let hello = {
        let mut cfg = SessionConfig::flock_128_r2(vec![p.spec()]);
        cfg.link = Some(LinkSpec { sigma: sigma.to_vec(), m: 0, points: 0 });
        cfg.hello()
    };
    let tw = Instant::now();
    let (mut b3, mut units, mut public) = p.rows(inst);
    if let Some((v, j)) = plan.unit_tamper {
        let i = v * p.units + j;
        units[i][if j == 0 { 0 } else { 4 }].lo ^= if j == 0 { 1 } else { 1 << 3 };
    }
    if plan.forge_middle {
        // Block 5 of VU 0's x row carries another message in its compression row; its output no longer feeds block 6,
        // whose input is still the honest chaining value, and the chunk's public value stays honest.
        let r = &mut b3[5];
        r.1[3] ^= 1;
    }
    if let Some(what) = plan.bad_params {
        let nb = p.chunks[0];
        let r = &mut b3[nb - 1];
        match what {
            0 => r.4 &= !CHUNK_END,
            1 => r.2 = 7,
            2 => r.3 = 32,
            _ => r.4 &= !KEYED_HASH,
        }
    }
    if let Some(k) = plan.publics_flip {
        let at = p.fixed.len() + k;
        public[at].lo ^= 1 << 12;
    }
    let rows_s = tw.elapsed().as_secs_f64();
    let pub_bytes = f128_bytes(&public[p.fixed.len()..]);
    let (b3_1, units_1) = if plan.rep1_other_witness {
        let mut other = inst.clone();
        other.x[3] ^= 0x40;
        let (b, u, _) = p.rows(&other);
        (b, u)
    } else {
        (b3.clone(), units.clone())
    };
    let mut prove_s = Vec::new();
    let mut proofs: Vec<Vec<u8>> = Vec::new();
    let mut committed = false;
    let res: Result<(), String> = (|| {
        match call(&t, Req::Hello(hello))? {
            Resp::Ok => {}
            r => return Err(format!("hello: {r:?}")),
        }
        for rep in 0..2 {
            let mut lch = LiveChallenger::new(t.clone(), TABLE, rep, &domain(rep))?;
            let tt = t.clone();
            let mut hook = |cm: &Commitment| -> Vec<LinkClaim> {
                if committed || plan.commit == CommitMode::Skip {
                    return Vec::new();
                }
                let mut root_b = cm.cap.as_flattened().to_vec();
                if plan.commit == CommitMode::OtherRootB {
                    root_b[0] ^= 1;
                }
                let mut sig = sigma.to_vec();
                if plan.commit == CommitMode::OtherSigma {
                    sig[0] ^= 1;
                }
                match call(&tt, Req::Commit { root_f: sig, roots: vec![(TABLE.into(), root_b)], publics: vec![(TABLE.into(), pub_bytes.clone())] }) {
                    Ok(Resp::Coins(c)) if c.is_empty() => {}
                    other => panic!("LIVE-REFUSED commit: {other:?}"),
                }
                match call(&tt, Req::Link(Vec::new())) {
                    Ok(Resp::Ok) => {}
                    other => panic!("LIVE-REFUSED link: {other:?}"),
                }
                committed = true;
                Vec::new()
            };
            let tp = Instant::now();
            let (b, u) = if rep == 0 { (&b3, &units) } else { (&b3_1, &units_1) };
            let pr = catch_unwind(AssertUnwindSafe(|| p.prove(b, u, &public, plan.profile[rep], &mut hook, &mut lch))).map_err(panic_msg)?;
            prove_s.push(tp.elapsed().as_secs_f64());
            proofs.push(pr);
        }
        Ok(())
    })();
    if let Err(e) = res {
        let v = match call(&t, Req::Finish) {
            Ok(Resp::Verdict(v)) => serde_json::from_str(&v).unwrap(),
            other => json!({"accepted": false, "why": format!("{other:?}")}),
        };
        return (json!({"accepted": v["accepted"], "why": v["why"], "verdict": v, "prover_stopped": e.chars().take(200).collect::<String>()}), proofs);
    }
    let e2e = t0.elapsed().as_secs_f64();
    let sent = plan.substitute.clone().unwrap_or_else(|| proofs.clone());
    for (rep, pr) in sent.iter().enumerate() {
        call(&t, Req::Proof { table: TABLE.into(), rep: rep as u32, bytes: pr.clone() }).unwrap();
    }
    let v: Value = match call(&t, Req::Finish) {
        Ok(Resp::Verdict(v)) => serde_json::from_str(&v).unwrap(),
        other => json!({"accepted": false, "why": format!("{other:?}")}),
    };
    let (ws, wr, calls) = t.lock().unwrap().stats();
    let wait_s = t.lock().unwrap().wait_s();
    let session_s = t0.elapsed().as_secs_f64();
    (
        json!({"accepted": v["accepted"], "why": v["why"], "verdict": v, "relation": inst.relation, "vus": p.vus, "units": p.units,
               "n_blake3": p.n_b3, "n_units": p.vus * p.units, "nu": p.nu, "dense_m": p.params(LigeritoProfile::Fast100).m,
               "sigma": hex(&sigma), "roots": c.roots.iter().map(|r| hex(r)).collect::<Vec<_>>(),
               "commit_s": commit_s, "rows_s": rows_s, "prove_s": prove_s, "prove_total_s": prove_s.iter().sum::<f64>(),
               "e2e_s": e2e, "session_s": session_s, "proof_bytes": proofs.iter().map(|p| p.len()).collect::<Vec<_>>(), "wire_sent": ws, "wire_recv": wr,
               "calls": calls, "wait_s": wait_s}),
        proofs,
    )
}

fn inproc(p: &Arc<Pure>, inst: &Instances) -> (SharedTransport, Arc<Mutex<Server>>) {
    let s = Arc::new(Mutex::new(server(p, inst)));
    (shared(InProc(s.clone())), s)
}

fn neg(p: &Arc<Pure>, pv: &Arc<Pure>, prover: &Instances, verifier: &Instances, name: &str, maps_to: &str, plan: &Plan, want: bool) -> bool {
    let (t, s) = inproc(pv, verifier);
    let (v, _) = session(p, prover, t, plan);
    let acc = v["accepted"].as_bool().unwrap_or(false);
    let rec = s.lock().unwrap().record_json();
    let reps: Vec<Value> = rec["verdict"]["reps"].as_array().cloned().unwrap_or_default().iter().map(|r| json!([r["stream"], r["ok"]])).collect();
    println!(
        "NEG\t{}",
        json!({"case": name, "maps_to": maps_to, "vus": p.vus, "expect": if want {"accept"} else {"reject"}, "accepted": acc,
               "pass": acc == want, "why": v["why"], "prover_stopped": v["prover_stopped"], "reps_ok": reps, "aborted": rec["aborted"],
               "link_sha256": rec["link_sha256"], "require_link": rec["config"]["require_link"]})
    );
    acc == want
}

fn selftest(p: Arc<Pure>, inst: Instances, other_net: Option<Arc<UnitNet>>) {
    let mut all = true;
    // The native rows and public words are the circuit's own (the online phase over the netlist's gate evaluation).
    let (b3, units, public) = p.rows(&inst);
    let w = p.shape.run(&p.inputs(&inst), &[]);
    let same = w.rows::<Blake3Gate>(p.s_b3) == b3.as_slice() && w.rows::<UnitGate>(p.s_unit) == units.as_slice() && w.public == public;
    let unit_ok = units.iter().take(64).all(|r| p.net.eval_one(r).1);
    let roots = check_roots(&inst);
    println!("NEG\t{}", json!({"case": "native_rows_equal_circuit_run", "maps_to": "witness = statement", "pass": same && unit_ok,
        "units_satisfied": unit_ok}));
    println!("NEG\t{}", json!({"case": "native_roots_equal_reference", "maps_to": "frame-v3 conformance", "pass": roots.is_ok(), "why": roots.err()}));
    all &= same && unit_ok && check_roots(&inst).is_ok();
    let h = Plan::honest;
    let mut t = |name: &str, maps: &str, plan: Plan, want: bool, both: Option<&Instances>, verifier: Option<&Instances>| {
        let pi = both.unwrap_or(&inst);
        all &= neg(&p, &p, pi, verifier.unwrap_or(pi), name, maps, &plan, want);
    };
    t("honest_session", "honest", h(), true, None, None);
    t("honest_session_again", "honest (fresh coins)", h(), true, None, None);
    let last = p.vus - 1;
    let per = p.publics_per_vu();
    t("public_output_forged", "C4 (output word)", Plan { publics_flip: Some(last * per + per - 1), ..h() }, false, None, None);
    t("public_chunk_value_forged", "C4 (chunk values)", Plan { publics_flip: Some(1), ..h() }, false, None, None);
    let epi = inst.header["epilogue"].as_bool().unwrap();
    let with_out = |o: u32| -> Instances {
        let mut v = inst.clone();
        v.out[last] = o;
        v.y[last] = if epi { o } else { o >> 10 };
        v
    };
    let wrong_y = with_out(inst.out[last] ^ (1 << 12));
    t("committed_y_not_the_relation_output", "relation: both hold the same wrong y (same Σ); the public output word is the committed one, the last unit's output is not", h(), false, Some(&wrong_y), None);
    t("verifier_y_differs_from_prover_y", "Σ (y root)", h(), false, None, Some(&wrong_y));
    t("unit_operand_differs_from_hashed_bytes", "relation-leaf glue (unit x input vs message word)", Plan { unit_tamper: Some((0, 0)), ..h() }, false, None, None);
    t("accumulator_chain_broken", "chain glue (c_in(j) = c_out(j-1))", Plan { unit_tamper: Some((0, 7)), ..h() }, false, None, None);
    t("forged_middle_block_honest_endpoints", "leaf chain (link neg 13)", Plan { forge_middle: true, ..h() }, false, None, None);
    t("chunk_end_flag_dropped", "leaf params (flag)", Plan { bad_params: Some(0), ..h() }, false, None, None);
    t("wrong_counter", "leaf params (counter)", Plan { bad_params: Some(1), ..h() }, false, None, None);
    t("wrong_block_len", "leaf params (block_len)", Plan { bad_params: Some(2), ..h() }, false, None, None);
    t("keyed_flag_dropped", "leaf params (keyed mode)", Plan { bad_params: Some(3), ..h() }, false, None, None);
    t("root_b_other_than_bound_root", "L1/R1 (the bound root)", Plan { commit: CommitMode::OtherRootB, ..h() }, false, None, None);
    t("commit_names_other_statement", "statement Σ at Commit", Plan { commit: CommitMode::OtherSigma, ..h() }, false, None, None);
    t("coins_before_commit", "R5 (no coin before the statement's publics)", Plan { commit: CommitMode::Skip, ..h() }, false, None, None);
    let mut other_set = inst.clone();
    other_set.header["instances"]["range"] = json!([0, p.vus + 1]);
    t("hello_for_other_instance_set", "Σ (instance reference)", h(), false, None, Some(&other_set));
    let mut other_rows = inst.clone();
    other_rows.x[100] ^= 1;
    t("verifier_holds_other_rows", "Σ (roots)", h(), false, None, Some(&other_rows));
    t("reps_with_different_witness", "R1", Plan { rep1_other_witness: true, ..h() }, false, None, None);
    t("fast_profile_proofs", "R7 / F2", Plan { profile: [LigeritoProfile::Fast; 2], ..h() }, false, None, None);
    let (_, honest_proofs) = session(&p, &inst, inproc(&p, &inst).0, &h());
    t("cross_session_replay_both_reps", "link neg 15", Plan { substitute: Some(honest_proofs), ..h() }, false, None, None);
    if let Some(net2) = other_net {
        // A prover whose lowering is another netlist: its shape has other registry and circuit digests.
        let p2 = Arc::new(Pure::new(p.vus, p.units, p.row_bytes, net2));
        all &= neg(&p2, &p, &inst, &inst, "prover_lowering_other_netlist", "Σ (netlist sha256)", &h(), false);
        let sig = p.sigma(&inst, &commit(&inst).roots);
        all &= neg(&p2, &p, &inst, &inst, "prover_lowering_other_netlist_verifier_sigma", "F2 (registry and circuit digests bound)",
                   &Plan { sigma: Some(sig), ..h() }, false);
    }
    println!("SELFTEST\t{}", json!({"relation": inst.relation, "vus": p.vus, "units": p.units, "nu": p.nu, "n_blake3": p.n_b3,
        "netlist_sha256": hex(&p.net.sha256), "all_pass": all}));
}


// ------------------------------------------------------------------------------------------------------------
// Re-verification of a recorded session (the store's copy: session record, proof files, the instance set)
// ------------------------------------------------------------------------------------------------------------

fn f128_of_hex(s: &str) -> F128 {
    F128::new(u64::from_str_radix(&s[16..32], 16).unwrap(), u64::from_str_radix(&s[..16], 16).unwrap())
}

/// Replay a verifier's session record against its proofs with this machine's own statement (instance set, lowering,
/// commitments): the replayed coins are the recorded ones, so this is not transferable evidence, but every message
/// the proofs carry must hash to what the verifier recorded before issuing each coin.
fn replay(p: &Arc<Pure>, inst: &Instances, rec: &Value, proofs: &[Vec<u8>]) -> Result<Value, String> {
    let c = commit(inst);
    let sigma = p.sigma(inst, &c.roots);
    let mut cfg = SessionConfig::flock_128_r2(vec![p.spec()]);
    cfg.link = Some(LinkSpec { sigma: sigma.to_vec(), m: 0, points: 0 });
    if rec["kind"] != "flock-live-session/v1" {
        return Err("not a flock-live session record".into());
    }
    if rec["hello"].as_str() != Some(cfg.hello().as_str()) {
        return Err("the record's session parameters (Σ) are not this statement's".into());
    }
    if rec["config"]["require_link"] != true || rec["link_sha256"].is_null() || rec["link_mode"] != "exchange" {
        return Err("F1: the record is not a session whose statement was committed before any coin".into());
    }
    if rec["aborted"].is_string() || rec["verdict"]["accepted"] != true {
        return Err("the live verifier did not accept this session".into());
    }
    let (_, _, public) = p.rows(inst);
    let publics = f128_bytes(&public[p.fixed.len()..]);
    if rec["link"]["publics_sha256"][TABLE].as_str() != Some(hex(&sha256(&publics)).as_str()) {
        return Err("C4: the record's public words are not this instance set's chunk values and outputs".into());
    }
    let v = PureVerifier { p: p.clone(), digests: c.digests, out: inst.out.clone() };
    v.statement_publics(TABLE, &publics)?;
    let want_proofs: Vec<String> = rec["proof_sha256"].as_array().ok_or("no proofs recorded")?.iter()
        .map(|x| x["sha256"].as_str().unwrap_or_default().to_string()).collect();
    if proofs.len() != 2 || want_proofs.len() != 2 || proofs.iter().zip(&want_proofs).any(|(b, w)| hex(&sha256(b)) != *w) {
        return Err("the proof files are not the recorded proofs".into());
    }
    let mut streams = std::collections::BTreeMap::new();
    let mut roots = std::collections::BTreeMap::new();
    for s in rec["streams"].as_array().ok_or("no streams")? {
        let name = s["stream"].as_str().unwrap().to_string();
        let rounds = s["rounds"].as_array().unwrap().iter().map(|r| RoundRec {
            sha: unhex(r["msg_sha256"].as_str().unwrap()).try_into().unwrap(),
            len: r["msg_len"].as_u64().unwrap() as usize,
            n: r["n"].as_u64().unwrap() as u32,
            coins: r["coins"].as_array().unwrap().iter().map(|c| f128_of_hex(c.as_str().unwrap())).collect(),
        }).collect();
        roots.insert(name.clone(), s["root_sha256"].as_str().map(|x| x.to_string()));
        streams.insert(name, StreamRec { parent: s["parent"].as_str().unwrap().to_string(), label: s["label"].as_str().unwrap().as_bytes().to_vec(),
                                         at: s["at"].as_u64().unwrap(), rounds, cap: None });
    }
    let record = Arc::new(streams);
    let ctx = LinkCtx { points: &[], y: &[], publics: &publics };
    let t = Instant::now();
    let mut caps = Vec::new();
    for rep in 0..2 {
        let root = rep_stream(TABLE, rep);
        let mut ch = ReplayChallenger::new(record.clone(), &root, &domain(rep));
        let cap = v.verify(TABLE, rep, &proofs[rep], &mut ch, &ctx).map_err(|e| match ch.first_failure() {
            Some(r) => format!("{r} (then the verifier: {e})"),
            None => e,
        })?;
        ch.finish()?;
        let h = hex(&sha256(&cap));
        if roots.get(&root).cloned().flatten() != Some(h.clone()) {
            return Err(format!("{root}: the proof's root is not the root the session committed"));
        }
        caps.push(h);
    }
    if caps[0] != caps[1] {
        return Err("R1: the reps are on different roots".into());
    }
    Ok(json!({"accepted": true, "relation": inst.relation, "vus": p.vus, "sigma": hex(&sigma), "roots": c.roots.iter().map(|r| hex(r)).collect::<Vec<_>>(),
              "netlist_sha256": hex(&p.net.sha256), "root_b_sha256": caps[0], "link_sha256": rec["link_sha256"], "verify_s": t.elapsed().as_secs_f64(),
              "proof_sha256": want_proofs}))
}

fn main() {
    flock_live::init_threads();
    let cmd = std::env::args().nth(1).unwrap_or_default();
    let inst = Instances::load(&arg("--instances", "instances.bin"));
    let net = Arc::new(UnitNet::load(&arg("--netlist", "net.txt")));
    let pin = arg("--pin", "");
    if !pin.is_empty() && pin != hex(&net.sha256) {
        eprintln!("the netlist {} is not the pinned lowering {pin}", hex(&net.sha256));
        std::process::exit(2);
    }
    let tb = Instant::now();
    let p = Arc::new(Pure::for_instances(&inst, net));
    eprintln!("statement {} vus={} units={} blake3={} nu={} m={} netlist={} built in {:.2}s", inst.relation, p.vus, p.units, p.n_b3, p.nu,
              p.params(LigeritoProfile::Fast100).m, hex(&p.net.sha256), tb.elapsed().as_secs_f64());
    match cmd.as_str() {
        "selftest" => {
            let other = std::env::args().any(|a| a == "--netlist-other").then(|| Arc::new(UnitNet::load(&arg("--netlist-other", ""))));
            selftest(p, inst, other)
        }
        "serve" => {
            if let Err(e) = check_roots(&inst) {
                eprintln!("{e}");
                std::process::exit(2);
            }
            serve_loop(move || server(&p, &inst));
        }
        "prove" => {
            let runs: usize = arg("--runs", "1").parse().unwrap();
            let warm: usize = arg("--warm", "0").parse().unwrap();
            let dump = arg("--dump", "");
            for run in 0..warm + runs {
                let mut tcp = TcpTransport::connect(&arg("--verifier", "127.0.0.1:7200")).unwrap();
                let (rtt, rtt_min, rtt_n) = tcp.ping_rtt(30);
                let t = shared(tcp);
                let (mut v, proofs) = session(&p, &inst, t, &Plan::honest());
                v["run"] = json!(run as i64 - warm as i64);
                v["warm"] = json!(run < warm);
                v["open_rtt_ms"] = json!(rtt);
                v["open_rtt_min_ms"] = json!(rtt_min);
                v["open_rtt_n"] = json!(rtt_n);
                if !dump.is_empty() {
                    std::fs::create_dir_all(&dump).unwrap();
                    for (rep, pr) in proofs.iter().enumerate() {
                        let f = format!("{dump}/run{:02}-rep{rep}.proof", run);
                        std::fs::write(&f, pr).unwrap();
                    }
                    v["dump"] = json!(format!("run{:02}-rep{{0,1}}.proof", run));
                }
                println!("LIVE\t{v}");
            }
        }
        "replay" => {
            let rec: Value = serde_json::from_slice(&std::fs::read(arg("--session", "session.json")).unwrap()).unwrap();
            let proofs: Vec<Vec<u8>> = arg("--proofs", "").split(',').map(|f| std::fs::read(f).unwrap()).collect();
            if pin.is_empty() {
                eprintln!("replay needs --pin (the pinned lowering's sha256)");
                std::process::exit(2);
            }
            if let Err(e) = check_roots(&inst) {
                eprintln!("{e}");
                std::process::exit(2);
            }
            match replay(&p, &inst, &rec, &proofs) {
                Ok(v) => println!("REPLAY\t{v}"),
                Err(e) => {
                    println!("REPLAY\t{}", json!({"accepted": false, "why": e}));
                    std::process::exit(1);
                }
            }
        }
        _ => eprintln!("usage: flock-pure serve|prove|selftest --instances FILE --netlist FILE ..."),
    }
}

fn serve_loop(make: impl Fn() -> Server) {
    use std::io::{BufReader, BufWriter, Write};
    let listen = arg("--listen", "127.0.0.1:7200");
    let out = std::path::PathBuf::from(arg("--out", "sessions"));
    std::fs::create_dir_all(&out).unwrap();
    let l = std::net::TcpListener::bind(&listen).unwrap();
    let max: Option<usize> = arg("--sessions", "").parse().ok();
    println!("SERVING {listen}");
    for (i, conn) in l.incoming().enumerate() {
        let conn = conn.unwrap();
        conn.set_nodelay(true).unwrap();
        let mut r = BufReader::new(conn.try_clone().unwrap());
        let mut w = BufWriter::new(conn);
        let mut s = make();
        let id = format!("p{:04}-{}", i, std::process::id());
        loop {
            let Ok(b) = read_frame(&mut r) else { break };
            let resp = match Req::decode(&b) {
                Ok(req) => s.handle(req),
                Err(e) => Resp::Refused(e),
            };
            let done = matches!(resp, Resp::Verdict(_));
            if write_frame(&mut w, &resp.encode()).is_err() || done {
                break;
            }
        }
        let rec = s.record_json();
        let d = out.join(&id);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("session.json"), flock_live::record_text::canonical_json(&rec)).unwrap();
        let line = json!({"session": id, "accepted": rec["verdict"]["accepted"], "why": rec["verdict"]["why"], "aborted": rec["aborted"],
                          "link_mode": rec["link_mode"], "link_sha256": rec["link_sha256"], "require_link": rec["config"]["require_link"],
                          "sigma": rec["hello"], "verify_s": rec["verdict"]["verify_s"], "rounds": rec["verdict"]["rounds"]});
        let mut idx = std::fs::OpenOptions::new().create(true).append(true).open(out.join("index.jsonl")).unwrap();
        writeln!(idx, "{line}").unwrap();
        println!("SESSION\t{line}");
        if Some(i + 1) == max {
            break;
        }
    }
}
