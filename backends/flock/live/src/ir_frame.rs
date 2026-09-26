//! `verity/flock-ir-frame/v1`: an IR template's units bound to their frame-v3 rows (lane flock-ir-lowering;
//! `verity_flock.ir_frame` stages the instance file). One block R1CS for the CPU prover and Flock-CUDA's host-witness mode.
//!
//! Block `b` holds `G` components: their BLAKE3 chunks' compressions at 2^14-bit slots `0 .. C` (chunk slot `q` at slots
//! `q nb .. q nb + nb`), then their units at 2^unit_log slots `unit_pos0 ..`. `A = I ⊗ circuit + Δ` (and B), where Δ makes,
//! inside the block:
//! - each chunk a chain (chaining input of compression `j ≥ 1` = `out_lo` of compression `j − 1`), its first chaining input
//!   the x-row key, block_len 64, flags KEYED_HASH | CHUNK_START (j = 0) | CHUNK_END and, for a row of one chunk, ROOT
//!   (j = nb − 1), all constants;
//! - every leaf input bit of every unit a copy of the message bit it hashes (the plan's wiring);
//! - every sub-circuit's constant a copy of the pinned column (compression slot 0's).
//!
//! Public regions, each opened at two verifier points drawn after the commitment (both reps): every compression's 64
//! counter bits (the chunk's index in its row, verifier-known), every chunk's `out_lo` at its last compression (a
//! prover-committed public: the chunk chaining value, or the row digest for a one-chunk row) and every unit's returned-output
//! words (public, from the verifier's own instance file). The verifier folds each row's chunk values into its digest
//! natively (BLAKE3's keyed parent tree) and checks the digests, and the output words, against the frame-v3 roots it
//! recomputes. Padding (missing components of the last block, blocks up to a power of two) hashes zero messages at counter 0.

use std::{
    array::from_fn,
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

use flock_core::{
    lincheck::{LincheckCircuit, SparseMatrixCircuit, pack_z_lincheck_from_packed},
    proof::eq_skip_weights,
    r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout},
    verifier::ExtraClaim,
};
use flock_field::F128;
use flock_hash::blake3_compress;
use flock_prover::r1cs_hashes::blake3 as b3;
use rayon::prelude::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{
    chunk::{CHUNK_END, CHUNK_START, KEYED_HASH, ROOT, comp, key_words, log2ceil, tree},
    ir_block::IrUnitNet,
};

pub const TAG: &[u8] = b"verity/flock-ir-frame/v1";
pub const PT_LOCAL: usize = 16;
pub const PIN: usize = b3::Z_CONST_POS;
const WORD: usize = 128;
const FRAME: &[u8] = b"veritor/protocol/merkle/frame/v3\0";

fn g_usize(v: &Value, k: &str) -> Result<usize, String> {
    v[k].as_u64().map(|x| x as usize).ok_or(format!("header: {k} missing"))
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|y| y as usize)).collect()).unwrap_or_default()
}

/// The block layout the verifier's own staging wrote (`verity_flock.ir_frame.write`).
#[derive(Clone, Debug)]
pub struct Layout {
    pub comp_slots: usize,
    pub nb: usize,
    pub chunks: usize,
    pub units: usize,
    pub unit_log: usize,
    pub unit_pos0: usize,
    pub k_log: usize,
    pub chunks_per_comp: usize,
    pub units_per_comp: usize,
    pub single_chunk: Vec<bool>,
    pub wiring: Vec<Vec<(usize, usize)>>,
    pub leaf_cols: Vec<usize>,
    pub ret_col: usize,
    pub ret_words: usize,
    pub ret_cols: Vec<usize>,
    pub out_leaf: Vec<Vec<usize>>,
}

/// A chunk slot's contents: (instance, input port, chunk index in the row).
pub type ChunkRef = Option<(usize, usize, usize)>;

/// A `flock-ir-frame/v1` instance file.
pub struct FrameInstances {
    pub header: Value,
    pub unit: String,
    pub n: usize,
    pub upi: usize,
    pub units: usize,
    pub in_ports: Vec<(String, usize)>,
    pub out_ports: Vec<(String, usize)>,
    pub lay: Layout,
    pub blocks: Vec<(Vec<ChunkRef>, Vec<Option<usize>>)>,
    pub rows: Vec<u8>,
    pub digests: Vec<[u8; 32]>,
    pub outputs: Vec<F128>,
    pub cuts: Vec<u32>,
    pub key: [u8; 32],
    /// sha256 of every public byte (header, row digests, output words): what Σ binds; the rows are not in it.
    pub public_sha256: [u8; 32],
}

impl FrameInstances {
    pub fn load(path: &str) -> Result<Self, String> {
        let b = std::fs::read(path).map_err(|e| format!("instances: {e}"))?;
        let nl = b.iter().position(|&c| c == b'\n').ok_or("no header line")?;
        let header: Value = serde_json::from_slice(&b[..nl]).map_err(|e| format!("header: {e}"))?;
        if header["format"] != "flock-ir-frame/v1" {
            return Err("not a flock-ir-frame/v1 instance file".into());
        }
        let (n, upi, units) = (g_usize(&header, "instances")?, g_usize(&header, "units_per_instance")?, g_usize(&header, "units")?);
        if units != n * upi {
            return Err(format!("{units} units, but {n} instances x {upi} units each"));
        }
        let ports = |k: &str| -> Vec<(String, usize)> {
            header[k].as_array().map(|a| a.iter().map(|p| (p[0].as_str().unwrap_or("").to_string(), p[1].as_u64().unwrap_or(0) as usize)).collect()).unwrap_or_default()
        };
        let (in_ports, out_ports) = (ports("in_ports"), ports("out_ports"));
        let l = &header["layout"];
        let lay = Layout {
            comp_slots: g_usize(l, "comp_slots")?,
            nb: g_usize(l, "nb")?,
            chunks: g_usize(l, "chunks_per_block")?,
            units: g_usize(l, "units_per_block")?,
            unit_log: g_usize(l, "unit_log")?,
            unit_pos0: g_usize(l, "unit_pos0")?,
            k_log: g_usize(l, "k_log")?,
            chunks_per_comp: g_usize(l, "chunks_per_comp")?,
            units_per_comp: g_usize(l, "units_per_comp")?,
            single_chunk: l["single_chunk"].as_array().map(|a| a.iter().map(|x| x.as_bool().unwrap_or(false)).collect()).unwrap_or_default(),
            wiring: l["wiring"].as_array().ok_or("no wiring")?.iter().map(|u| {
                u.as_array().map(|a| a.iter().map(|p| (p[0].as_u64().unwrap_or(0) as usize, p[1].as_u64().unwrap_or(0) as usize)).collect()).unwrap_or_default()
            }).collect(),
            leaf_cols: usizes(&l["leaf_cols"]),
            ret_col: l["ret_group"][0].as_u64().ok_or("no ret_group")? as usize,
            ret_words: l["ret_group"][1].as_u64().ok_or("no ret_group")? as usize,
            ret_cols: usizes(&l["ret_cols"]),
            out_leaf: l["out_leaf"].as_array().ok_or("no out_leaf")?.iter().map(usizes).collect(),
        };
        let blocks = header["blocks"].as_array().ok_or("no blocks")?.iter().map(|bl| {
            let chunks = bl[0].as_array().map(|a| a.iter().map(|c| c.as_array().map(|x| (x[0].as_u64().unwrap_or(0) as usize, x[1].as_u64().unwrap_or(0) as usize, x[2].as_u64().unwrap_or(0) as usize))).collect()).unwrap_or_default();
            let us = bl[1].as_array().map(|a| a.iter().map(|u| u.as_u64().map(|x| x as usize)).collect()).unwrap_or_default();
            (chunks, us)
        }).collect();
        let row_bytes: usize = in_ports.iter().map(|p| 2 * p.1).sum();
        let mut at = nl + 1;
        let mut take = |k: usize| -> Result<&[u8], String> {
            let s = b.get(at..at + k).ok_or("instance file too short")?;
            at += k;
            Ok(s)
        };
        let rows = take(n * row_bytes)?.to_vec();
        let pub_start = nl + 1 + n * row_bytes;
        let digests: Vec<[u8; 32]> = take(32 * n * in_ports.len())?.chunks(32).map(|c| c.try_into().unwrap()).collect();
        let outputs: Vec<F128> = take(16 * units * lay.ret_words)?.chunks(16)
            .map(|c| F128::new(u64::from_le_bytes(c[..8].try_into().unwrap()), u64::from_le_bytes(c[8..].try_into().unwrap()))).collect();
        let kc = g_usize(&header, "cut_words").unwrap_or(0);
        let cuts: Vec<u32> = take(4 * n * kc)?.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        if at != b.len() {
            return Err("instance file length".into());
        }
        let key: [u8; 32] = hexb(header["frame_v3"]["key"].as_str().ok_or("no frame_v3 key")?)?.try_into().map_err(|_| "key length")?;
        let mut h = Sha256::new();
        h.update(&b[..nl]);
        h.update(&b[pub_start..pub_start + 32 * n * in_ports.len() + 16 * units * lay.ret_words]);
        let public_sha256: [u8; 32] = h.finalize().into();
        Ok(FrameInstances { unit: header["unit"].as_str().unwrap_or("").to_string(), header, n, upi, units, in_ports, out_ports, lay, blocks, rows,
                            digests, outputs, cuts, key, public_sha256 })
    }

    pub fn row_bytes(&self) -> usize {
        self.in_ports.iter().map(|p| 2 * p.1).sum()
    }

    /// Bytes of input port `p` of instance `i`.
    pub fn row(&self, i: usize, p: usize) -> &[u8] {
        let off: usize = self.in_ports[..p].iter().map(|q| 2 * q.1).sum();
        let at = i * self.row_bytes() + off;
        &self.rows[at..at + 2 * self.in_ports[p].1]
    }

    pub fn output(&self, g: usize) -> &[F128] {
        &self.outputs[g * self.lay.ret_words..(g + 1) * self.lay.ret_words]
    }

    pub fn digest(&self, i: usize, p: usize) -> [u8; 32] {
        self.digests[i * self.in_ports.len() + p]
    }

    pub fn chunks_of(&self, p: usize) -> usize {
        (2 * self.in_ports[p].1).div_ceil(1024)
    }
}

fn hexb(s: &str) -> Result<Vec<u8>, String> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|e| e.to_string())).collect()
}

// --- frame-v3 (verity.commitments.merkle), natively ---------------------------------------------------------------------

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

fn fv3_root(domain_id: &[u8], leaves: Vec<[u8; 32]>) -> [u8; 32] {
    let n = leaves.len();
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

/// Every instance's output words, from the units' returned-output words (the plan's `out_leaf` and `ret_cols`): port-major
/// `[port][instance][word]`.
pub fn instance_outputs(inst: &FrameInstances) -> Result<Vec<Vec<u16>>, String> {
    let widths: Vec<usize> = inst.out_ports.iter().map(|p| p.1).collect();
    let starts: Vec<usize> = widths.iter().scan(0, |a, &w| { let s = *a; *a += w; Some(s) }).collect();
    let mut out: Vec<Vec<Option<u16>>> = widths.iter().map(|&w| vec![None; inst.n * w]).collect();
    for g in 0..inst.units {
        let (i, u) = (g / inst.upi, g % inst.upi);
        let words = inst.output(g);
        for (j, &leaf) in inst.lay.out_leaf[u].iter().enumerate() {
            let c = inst.lay.ret_cols[j];
            let v = ((if c % WORD < 64 { words[c / WORD].lo >> (c % WORD) } else { words[c / WORD].hi >> (c % WORD - 64) }) & 0xFFFF) as u16;
            let p = starts.iter().rposition(|&s| s <= leaf).ok_or("output leaf")?;
            let slot = &mut out[p][i * widths[p] + leaf - starts[p]];
            if slot.replace(v).is_some() {
                return Err(format!("output leaf {leaf} of instance {i} is returned by two units"));
            }
        }
    }
    out.into_iter().map(|v| v.into_iter().collect::<Option<Vec<u16>>>().ok_or("an output word no unit returns".to_string())).collect()
}

/// The frame-v3 roots recomputed from the row digests and the units' output words, checked against the header's.
pub fn check_roots(inst: &FrameInstances) -> Result<(), String> {
    let fv = &inst.header["frame_v3"];
    let get = |k: &str, p: &str| -> Result<Vec<u8>, String> { hexb(fv[k][p].as_str().ok_or(format!("frame_v3.{k}.{p} missing"))?) };
    for (p, (name, _)) in inst.in_ports.iter().enumerate() {
        let dom = get("domain_ids", name)?;
        let schema = fv["schemas"][name].as_str().ok_or("schema")?;
        if schema != "blake3-keyed/row/v2" {
            return Err(format!("input port {name}: schema {schema}, not blake3-keyed/row/v2"));
        }
        let leaves: Vec<[u8; 32]> = (0..inst.n).into_par_iter().map(|i| fv3_hash(b"leaf", &[&dom, &uint(i), &uint(i), schema.as_bytes(), &inst.digest(i, p)])).collect();
        if fv3_root(&dom, leaves).to_vec() != get("roots", name)? {
            return Err(format!("input port {name}: the row digests do not hash to the header's frame-v3 root"));
        }
    }
    let outs = instance_outputs(inst)?;
    for (p, (name, _)) in inst.out_ports.iter().enumerate() {
        let dom = get("domain_ids", name)?;
        let leaves: Vec<[u8; 32]> = outs[p].par_iter().enumerate().map(|(r, &v)| fv3_hash(b"leaf", &[&dom, &uint(r), &uint(r), b"u16", &v.to_be_bytes()])).collect();
        if fv3_root(&dom, leaves).to_vec() != get("roots", name)? {
            return Err(format!("output port {name}: the units' output words do not hash to the header's frame-v3 root"));
        }
    }
    Ok(())
}

/// The block table against the plan: every row chunk in exactly one chunk slot, every unit in exactly one unit slot, and each
/// unit slot's instance the instance of its component's chunks.
pub fn check_blocks(inst: &FrameInstances) -> Result<(), String> {
    let lay = &inst.lay;
    let mut seen_c = BTreeSet::new();
    let mut seen_u = BTreeSet::new();
    for (b, (chunks, units)) in inst.blocks.iter().enumerate() {
        if chunks.len() != lay.chunks || units.len() != lay.units {
            return Err(format!("block {b}: {} chunk slots and {} unit slots, the layout has {} and {}", chunks.len(), units.len(), lay.chunks, lay.units));
        }
        for (u, g) in units.iter().enumerate() {
            let k = u / lay.units_per_comp;
            let insts: BTreeSet<usize> = chunks[k * lay.chunks_per_comp..(k + 1) * lay.chunks_per_comp].iter().map(|c| c.map(|x| x.0)).map(|x| x.unwrap_or(usize::MAX)).collect();
            match g {
                Some(g) => {
                    if !seen_u.insert(*g) || *g >= inst.units || insts != BTreeSet::from([g / inst.upi]) {
                        return Err(format!("block {b}: unit slot {u} holds unit {g}, not of its component's instance"));
                    }
                }
                None if insts != BTreeSet::from([usize::MAX]) => return Err(format!("block {b}: an empty unit slot beside real chunks")),
                None => {}
            }
        }
        for (q, c) in chunks.iter().enumerate() {
            if let Some((i, p, ci)) = c {
                if *p >= inst.in_ports.len() || *ci >= inst.chunks_of(*p) || *i >= inst.n || !seen_c.insert((*i, *p, *ci)) {
                    return Err(format!("block {b}: chunk slot {q} holds a chunk outside the rows or twice"));
                }
                if lay.single_chunk[q] != (inst.chunks_of(*p) == 1) {
                    return Err(format!("block {b}: chunk slot {q}'s ROOT flag does not match its row"));
                }
            }
        }
    }
    let want_c: usize = (0..inst.in_ports.len()).map(|p| inst.chunks_of(p)).sum::<usize>() * inst.n;
    if seen_c.len() != want_c || seen_u.len() != inst.units {
        return Err("the blocks do not hold every row chunk and every unit".into());
    }
    Ok(())
}

// --- the statement ---------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Region {
    Counter,
    Cv,
    Out,
}

pub struct RegionShape {
    pub region: Region,
    pub free: Vec<usize>,
    pub mask: u32,
    pub fixed: u32,
}

pub struct FrameCircuit {
    pub k_log: usize,
    pub comp_slots: usize,
    pub unit_log: usize,
    pub unit_pos0: usize,
    pub units: usize,
    pub unit: SparseMatrixCircuit<'static>,
    pub da: Vec<(u32, u32)>,
    pub db: Vec<(u32, u32)>,
}

impl LincheckCircuit for FrameCircuit {
    fn n_cols(&self) -> usize {
        1 << self.k_log
    }

    fn fold_alpha_batched(&self, alpha: F128, eq: &[F128]) -> Vec<F128> {
        let (cs, us) = (1usize << b3::K_LOG, 1usize << self.unit_log);
        let lo = cs.min(us);
        assert_eq!(eq.len(), 1 << self.k_log);
        let c0 = eq[..lo].iter().position(|e| *e != F128::ZERO).expect("eq tensor vanishes on the first slot");
        let inv = eq[c0].inv();
        let base_c = comp().csc_lincheck_circuit().fold_alpha_batched(alpha, &eq[..cs]);
        let base_u = self.unit.fold_alpha_batched(alpha, &eq[..us]);
        let mut out = vec![F128::ZERO; 1 << self.k_log];
        out.par_chunks_mut(lo).enumerate().for_each(|(k, o)| {
            let start = k * lo;
            let (ratio, base, local) = if start < self.comp_slots << b3::K_LOG {
                let s = start >> b3::K_LOG;
                (eq[(s << b3::K_LOG) | c0] * inv, &base_c, start & (cs - 1))
            } else {
                let p = start >> self.unit_log;
                if !(self.unit_pos0..self.unit_pos0 + self.units).contains(&p) {
                    return;
                }
                (eq[(p << self.unit_log) | c0] * inv, &base_u, start & (us - 1))
            };
            for (x, y) in o.iter_mut().zip(&base[local..local + lo]) {
                *x = ratio * *y;
            }
        });
        for &(i, c) in &self.da {
            out[c as usize] += alpha * eq[i as usize];
        }
        for &(i, c) in &self.db {
            out[c as usize] += eq[i as usize];
        }
        out
    }

    fn const_pin_col(&self) -> Option<usize> {
        Some(PIN)
    }
}

pub fn flags_of(j: usize, nb: usize, single: bool) -> u32 {
    KEYED_HASH | if j == 0 { CHUNK_START } else { 0 } | if j + 1 == nb { CHUNK_END | if single { ROOT } else { 0 } } else { 0 }
}

fn delta(net: &IrUnitNet, lay: &Layout, key: &[u32; 8]) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    let c = comp();
    let (a, b) = (&c.a_0.rows, &c.b_0.rows);
    let z = b3::Z_CONST_POS;
    let input = |r: usize| assert!(a[r] == [r] && b[r] == [z], "compression row {r} is not an input row");
    let (mut da, mut db) = (Vec::new(), Vec::new());
    let both = |da: &mut Vec<(u32, u32)>, db: &mut Vec<(u32, u32)>, i: usize, new: usize| {
        for d in [da, db] {
            d.push((i as u32, i as u32));
            d.push((i as u32, new as u32));
        }
    };
    let constant = |da: &mut Vec<(u32, u32)>, g: &dyn Fn(usize) -> usize, base: usize, val: u32| {
        for bit in 0..32 {
            let r = base + bit;
            input(r);
            da.push((g(r) as u32, g(r) as u32));
            if (val >> bit) & 1 == 1 {
                da.push((g(r) as u32, g(z) as u32));
            }
        }
    };
    for s in 0..lay.comp_slots {
        let (q, j) = (s / lay.nb, s % lay.nb);
        let g = |col: usize| (s << b3::K_LOG) + col;
        if s > 0 {
            both(&mut da, &mut db, g(z), PIN);
        }
        if j > 0 {
            for r in 0..256 {
                input(b3::CV_BASE + r);
                da.push((g(b3::CV_BASE + r) as u32, g(b3::CV_BASE + r) as u32));
                da.push((g(b3::CV_BASE + r) as u32, (((s - 1) << b3::K_LOG) + b3::OUT_LO_BASE + r) as u32));
            }
        } else {
            for (w, &kw) in key.iter().enumerate() {
                constant(&mut da, &g, b3::CV_BASE + 32 * w, kw);
            }
        }
        constant(&mut da, &g, b3::BLEN_BASE, 64);
        constant(&mut da, &g, b3::FLAGS_BASE, flags_of(j, lay.nb, lay.single_chunk[q]));
    }
    for u in 0..lay.units {
        let base = (lay.unit_pos0 + u) << lay.unit_log;
        both(&mut da, &mut db, base + net.const_pos, PIN);
        for (j, &(q, off)) in lay.wiring[u].iter().enumerate() {
            let s = q * lay.nb + off / 64;
            let mb = (s << b3::K_LOG) + b3::M_BASE + 8 * (off % 64);
            for t in 0..16 {
                both(&mut da, &mut db, base + lay.leaf_cols[j] + t, mb + t);
            }
        }
    }
    (da, db)
}

pub struct FrameStmt {
    pub lay: Layout,
    pub net: IrUnitNet,
    pub blocks_real: usize,
    pub nbl: usize,
    pub m: usize,
    pub m_pts: usize,
    pub regions: Vec<RegionShape>,
    pub circuit: FrameCircuit,
    pub r1cs: BlockR1cs,
    pub digest: [u8; 32],
    pub key: [u32; 8],
    /// A padding chunk's value per chunk slot (zero messages, counter 0).
    pub dummy_cv: Vec<[u32; 8]>,
    pub dummy_out: Vec<F128>,
}

impl FrameStmt {
    pub fn new(net: IrUnitNet, inst: &FrameInstances) -> Result<Self, String> {
        let lay = inst.lay.clone();
        let pow2 = |x: usize| x.is_power_of_two();
        if !(pow2(lay.nb) && pow2(lay.chunks) && lay.comp_slots == lay.chunks * lay.nb && pow2(lay.units) && lay.unit_pos0 % lay.units == 0
            && lay.unit_log == net.unit_log && lay.comp_slots << b3::K_LOG <= lay.unit_pos0 << lay.unit_log
            && (lay.unit_pos0 + lay.units) << lay.unit_log <= 1 << lay.k_log && lay.wiring.len() == lay.units
            && lay.wiring.iter().all(|w| w.len() == lay.leaf_cols.len()) && lay.single_chunk.len() == lay.chunks)
        {
            return Err("the block layout is not a valid flock-ir-frame/v1 layout".into());
        }
        if lay.wiring.iter().flatten().any(|&(q, off)| q >= lay.chunks || off % 2 == 1 || off >= 64 * lay.nb) {
            return Err("a unit reads outside its block's chunks".into());
        }
        if net.out_groups.is_empty() || net.out_groups[0].col != lay.ret_col || net.out_groups[0].words != lay.ret_words {
            return Err("the returned-output group is not the netlist's first output group".into());
        }
        let leaf_cols: Vec<usize> = {
            let g = &net.in_groups[0];
            g.bits.iter().scan(g.col * WORD, |a, &w| { let c = *a; *a += w; Some(c) }).collect()
        };
        if leaf_cols != lay.leaf_cols || net.in_groups[0].bits.iter().any(|&w| w != 16) || net.in_groups.len() != 1 {
            return Err("the unit's leaf ports are not the netlist's input group (16-bit ports)".into());
        }
        let blocks_real = inst.blocks.len();
        let nbl = log2ceil(blocks_real).max(3);
        let (kl, m) = (lay.k_log, lay.k_log + nbl);
        let key = key_words(&inst.key);
        let (da, db) = delta(&net, &lay, &key);
        let bits = |lo: usize, n: usize| (lo..lo + n).collect::<Vec<usize>>();
        let lc = lay.comp_slots.trailing_zeros() as usize;
        let lnb = lay.nb.trailing_zeros() as usize;
        let lq = lay.chunks.trailing_zeros() as usize;
        let lu = lay.units.trailing_zeros() as usize;
        let lw = lay.ret_words.trailing_zeros() as usize;
        let mk = |region: Region, free: Vec<usize>, fixed: u32| -> RegionShape {
            let mask = ((1u32 << kl) - 1) & !free.iter().fold(0u32, |mm, &b| mm | (1 << b));
            assert_eq!(fixed & !mask, 0, "{region:?}");
            assert!(free.len() <= PT_LOCAL, "{region:?}: {} free bits", free.len());
            RegionShape { region, free, mask, fixed }
        };
        let regions = vec![
            mk(Region::Counter, [bits(0, 6), bits(b3::K_LOG, lc)].concat(), (b3::T_LO_BASE as u32) & !63),
            mk(Region::Cv, [bits(0, 8), bits(b3::K_LOG + lnb, lq)].concat(), (b3::OUT_LO_BASE as u32) | (((lay.nb - 1) as u32) << b3::K_LOG)),
            mk(Region::Out, [bits(0, 7 + lw), bits(lay.unit_log, lu)].concat(),
               ((lay.ret_col as u32) << 7) | (((lay.unit_pos0 >> lu) as u32) << (lay.unit_log + lu))),
        ];
        assert_eq!(b3::T_HI_BASE, b3::T_LO_BASE + 32);
        let dummy_cv: Vec<[u32; 8]> = (0..lay.chunks).map(|q| {
            let mut cv = key;
            for j in 0..lay.nb {
                cv = blake3_compress(&cv, &[0u32; 16], 0, 64, flags_of(j, lay.nb, lay.single_chunk[q]))[..8].try_into().unwrap();
            }
            cv
        }).collect();
        let dummy_out = {
            let o = net.outputs(&vec![F128::ZERO; net.in_words]);
            o[..lay.ret_words].to_vec()
        };
        let mut h = Sha256::new();
        h.update(TAG);
        h.update(comp().statement_digest());
        h.update(net.sha256);
        h.update(inst.key);
        for x in [blocks_real, nbl, m, kl, lay.comp_slots, lay.nb, lay.chunks, lay.units, lay.unit_log, lay.unit_pos0, lay.ret_col, lay.ret_words] {
            h.update((x as u64).to_le_bytes());
        }
        for d in [&da, &db] {
            h.update((d.len() as u64).to_le_bytes());
            for &(i, c) in d.iter() {
                h.update(i.to_le_bytes());
                h.update(c.to_le_bytes());
            }
        }
        let digest: [u8; 32] = h.finalize().into();
        let kk = 1usize << kl;
        let empty = || SparseBinaryMatrix { num_rows: kk, num_cols: kk, rows: vec![Vec::new(); kk].into() };
        let r1cs = BlockR1cs {
            m,
            k_log: kl,
            k_skip: b3::K_SKIP,
            useful_bits: kk,
            a_0: empty(),
            b_0: empty(),
            c_0: SparseBinaryMatrix { num_rows: kk, num_cols: kk, rows: (0..kk).map(|i| vec![i]).collect::<Vec<_>>().into() },
            layout: WitnessLayout::RowMajor,
            const_pin: Some(PIN),
            digest_cache: OnceLock::from(digest),
            csc_cache: OnceLock::new(),
        };
        let us = 1usize << lay.unit_log;
        let pad = |rows: &Vec<Vec<usize>>| {
            let mut r = rows.clone();
            r.resize(us, Vec::new());
            SparseBinaryMatrix { num_rows: us, num_cols: us, rows: r.into() }
        };
        let (ua, ub): (&'static SparseBinaryMatrix, &'static SparseBinaryMatrix) = (Box::leak(Box::new(pad(&net.a))), Box::leak(Box::new(pad(&net.b))));
        let circuit = FrameCircuit { k_log: kl, comp_slots: lay.comp_slots, unit_log: lay.unit_log, unit_pos0: lay.unit_pos0, units: lay.units,
                                     unit: SparseMatrixCircuit::new(ua, ub), da, db };
        Ok(FrameStmt { lay, net, blocks_real, nbl, m, m_pts: PT_LOCAL + nbl, regions, circuit, r1cs, digest, key, dummy_cv, dummy_out })
    }

    pub fn blocks(&self) -> usize {
        1 << self.nbl
    }

    /// Every compression, block-major (`comp_slots` per block), and each real block's chunk values (the publics).
    pub fn compressions(&self, inst: &FrameInstances) -> (Vec<b3::Compression>, Vec<Vec<[u32; 8]>>) {
        let lay = &self.lay;
        let per: Vec<(Vec<b3::Compression>, Vec<[u32; 8]>)> = (0..self.blocks()).into_par_iter().map(|b| {
            let mut out = Vec::with_capacity(lay.comp_slots);
            let mut cvs = Vec::with_capacity(lay.chunks);
            for q in 0..lay.chunks {
                let chunk = inst.blocks.get(b).and_then(|bl| bl.0[q]);
                let (mut cv, ctr) = (self.key, chunk.map_or(0, |c| c.2 as u64));
                for j in 0..lay.nb {
                    let msg: [u32; 16] = match chunk {
                        Some((i, p, c)) => {
                            let row = inst.row(i, p);
                            let at = 1024 * c + 64 * j;
                            from_fn(|w| u32::from_le_bytes(row[at + 4 * w..at + 4 * w + 4].try_into().unwrap()))
                        }
                        None => [0; 16],
                    };
                    let f = flags_of(j, lay.nb, lay.single_chunk[q]);
                    out.push((cv, msg, ctr, 64, f));
                    cv = blake3_compress(&cv, &msg, ctr, 64, f)[..8].try_into().unwrap();
                }
                cvs.push(cv);
            }
            (out, cvs)
        }).collect();
        let cvs = per.iter().take(self.blocks_real).map(|p| p.1.clone()).collect();
        (per.into_iter().flat_map(|p| p.0).collect(), cvs)
    }

    /// Unit slot `u` of block `b`'s input words: each leaf port the 16-bit word at its wired message bytes.
    pub fn unit_inputs(&self, comps: &[b3::Compression], b: usize, u: usize) -> Vec<F128> {
        let lay = &self.lay;
        let mut w = vec![F128::ZERO; self.net.in_words];
        for (j, &(q, off)) in lay.wiring[u].iter().enumerate() {
            let msg = &comps[b * lay.comp_slots + q * lay.nb + off / 64].1;
            let byte = off % 64;
            let v = ((msg[byte / 4] >> (8 * (byte % 4))) & 0xFFFF) as u64;
            let c = lay.leaf_cols[j];
            if c % WORD < 64 { w[c / WORD].lo |= v << (c % WORD) } else { w[c / WORD].hi |= v << (c % WORD - 64) }
        }
        w
    }

    /// `(z, a, b, z_lincheck)`: the compressions' witness, then every unit evaluated from `inputs(b, u)`.
    #[allow(clippy::type_complexity)]
    pub fn witness(&self, comps: &[b3::Compression], inputs: &(dyn Fn(usize, usize) -> Vec<F128> + Sync)) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let lay = &self.lay;
        let n = 1usize << (self.m - 7);
        let (cz, ca, cb) = b3::generate_witness_with_ab_packed(comps, log2ceil(comps.len()));
        let (mut z, mut a, mut b) = (vec![F128::ZERO; n], vec![F128::ZERO; n], vec![F128::ZERO; n]);
        let wpb = 1usize << (lay.k_log - 7);
        let cw = 1usize << (b3::K_LOG - 7);
        for (dst, src) in [(&mut z, &cz), (&mut a, &ca), (&mut b, &cb)] {
            dst.par_chunks_mut(wpb).enumerate().for_each(|(blk, d)| {
                for s in 0..lay.comp_slots {
                    let ci = blk * lay.comp_slots + s;
                    d[s * cw..(s + 1) * cw].copy_from_slice(&src[ci * cw..(ci + 1) * cw]);
                }
            });
        }
        drop((cz, ca, cb));
        let uw = 1usize << (lay.unit_log - 7);
        let total = self.blocks() * lay.units;
        let groups: Vec<Vec<(usize, Vec<F128>, Vec<F128>, Vec<F128>)>> = (0..total.div_ceil(64)).into_par_iter().map(|gi| {
            let ids: Vec<usize> = (64 * gi..(64 * gi + 64).min(total)).collect();
            let ins: Vec<Vec<F128>> = ids.iter().map(|&x| inputs(x / lay.units, x % lay.units)).collect();
            let (zz, aa, bb) = self.net.eval64(|col| {
                let mut x = 0u64;
                for (l, r) in ins.iter().enumerate() {
                    let w = &r[col / WORD];
                    let t = col % WORD;
                    x |= ((if t < 64 { w.lo >> t } else { w.hi >> (t - 64) }) & 1) << l;
                }
                x
            });
            ids.iter().enumerate().map(|(l, &x)| {
                let pack = |v: &Vec<u64>| -> Vec<F128> {
                    let mut o = vec![F128::ZERO; uw];
                    for (r, &y) in v.iter().enumerate() {
                        if (y >> l) & 1 == 1 {
                            if r % 128 < 64 { o[r / 128].lo |= 1 << (r % 128) } else { o[r / 128].hi |= 1 << (r % 128 - 64) }
                        }
                    }
                    o
                };
                (x, pack(&zz), pack(&aa), pack(&bb))
            }).collect()
        }).collect();
        for (x, uz, ua, ub) in groups.into_iter().flatten() {
            let (blk, u) = (x / lay.units, x % lay.units);
            let at = blk * wpb + (lay.unit_pos0 + u) * uw;
            z[at..at + uw].copy_from_slice(&uz);
            a[at..at + uw].copy_from_slice(&ua);
            b[at..at + uw].copy_from_slice(&ub);
        }
        let zl = pack_z_lincheck_from_packed(&z, self.m, lay.k_log);
        (z, a, b, zl)
    }

    /// Block `b`'s bytes of a region, in free-assignment order.
    fn region_bytes(&self, reg: Region, inst: &FrameInstances, cvs: &[Vec<[u32; 8]>], b: usize) -> Vec<u8> {
        let lay = &self.lay;
        let bl = inst.blocks.get(b);
        match reg {
            Region::Counter => (0..lay.comp_slots).flat_map(|s| {
                let c = bl.and_then(|x| x.0[s / lay.nb]).map_or(0u64, |c| c.2 as u64);
                c.to_le_bytes()
            }).collect(),
            Region::Cv => (0..lay.chunks).flat_map(|q| {
                let cv = match bl.and_then(|x| x.0[q]) {
                    Some(_) => cvs[b][q],
                    None => self.dummy_cv[q],
                };
                cv.into_iter().flat_map(|w| w.to_le_bytes())
            }).collect(),
            Region::Out => (0..lay.units).flat_map(|u| {
                let words: Vec<F128> = match bl.and_then(|x| x.1[u]) {
                    Some(g) => inst.output(g).to_vec(),
                    None => self.dummy_out.clone(),
                };
                words.into_iter().flat_map(|w| [w.lo.to_le_bytes(), w.hi.to_le_bytes()].concat())
            }).collect(),
        }
    }

    pub fn region_point(&self, reg: &RegionShape, r: &[F128]) -> Vec<F128> {
        let kl = self.lay.k_log;
        let mut p = vec![F128::ZERO; self.m];
        for j in 0..kl {
            if (reg.mask >> j) & 1 == 1 && (reg.fixed >> j) & 1 == 1 {
                p[j] = F128::ONE;
            }
        }
        for (t, &j) in reg.free.iter().enumerate() {
            p[j] = r[t];
        }
        p[kl..].copy_from_slice(&r[PT_LOCAL..PT_LOCAL + self.nbl]);
        p
    }

    pub fn region_value(&self, reg: &RegionShape, r: &[F128], inst: &FrameInstances, cvs: &[Vec<[u32; 8]>]) -> F128 {
        let nf = reg.free.len();
        let wt = eq_skip_weights(&r[..nf]);
        let eb = eq_skip_weights(&r[PT_LOCAL..PT_LOCAL + self.nbl]);
        let nbytes = (1usize << nf) / 8;
        let mut t = vec![[F128::ZERO; 256]; nbytes];
        for (i, tb) in t.iter_mut().enumerate() {
            for x in 1..256usize {
                let rest = x & (x - 1);
                tb[x] = tb[rest] + wt[8 * i + (x ^ rest).trailing_zeros() as usize];
            }
        }
        (0..self.blocks()).into_par_iter().map(|b| {
            let bytes = self.region_bytes(reg.region, inst, cvs, b);
            debug_assert_eq!(bytes.len(), nbytes);
            eb[b] * bytes.iter().enumerate().fold(F128::ZERO, |a, (i, &x)| a + t[i][x as usize])
        }).reduce(|| F128::ZERO, |a, b| a + b)
    }

    pub fn extra_points(&self, pts: &[Vec<F128>]) -> Vec<Vec<F128>> {
        self.regions.iter().flat_map(|reg| pts.iter().map(move |r| self.region_point(reg, r))).collect()
    }

    pub fn extra_claims(&self, pts: &[Vec<F128>], inst: &FrameInstances, cvs: &[Vec<[u32; 8]>]) -> Vec<ExtraClaim> {
        let values: Vec<F128> = self.regions.iter().flat_map(|reg| pts.iter().map(move |r| self.region_value(reg, r, inst, cvs))).collect();
        self.extra_points(pts).into_iter().zip(values)
            .map(|(pt, value)| ExtraClaim { value, skip_weights: eq_skip_weights(&pt[..6]), x_full: pt[6..].to_vec() })
            .collect()
    }

    /// The prover's publics: every real block's chunk values, 32 bytes each (8 u32 LE).
    pub fn publics_bytes(cvs: &[Vec<[u32; 8]>]) -> Vec<u8> {
        cvs.iter().flatten().flatten().flat_map(|w| w.to_le_bytes()).collect()
    }

    pub fn parse_publics(&self, b: &[u8]) -> Result<Vec<Vec<[u32; 8]>>, String> {
        let per = 32 * self.lay.chunks;
        if b.len() != per * self.blocks_real {
            return Err(format!("C4: {} bytes of publics, want {}", b.len(), per * self.blocks_real));
        }
        Ok(b.chunks(per).map(|blk| blk.chunks(32).map(|c| from_fn(|w| u32::from_le_bytes(c[4 * w..4 * w + 4].try_into().unwrap()))).collect()).collect())
    }

    /// C4: each row's chunk values fold (natively, BLAKE3's keyed parent tree; a one-chunk row's value is its ROOT output)
    /// into the row digest the verifier's file commits under its frame-v3 root.
    pub fn check_digests(&self, inst: &FrameInstances, cvs: &[Vec<[u32; 8]>]) -> Result<(), String> {
        let mut rows: BTreeMap<(usize, usize), BTreeMap<usize, [u32; 8]>> = BTreeMap::new();
        for (b, (chunks, _)) in inst.blocks.iter().enumerate() {
            for (q, c) in chunks.iter().enumerate() {
                if let Some((i, p, ci)) = c {
                    rows.entry((*i, *p)).or_default().insert(*ci, cvs[b][q]);
                }
            }
        }
        for ((i, p), cs) in rows {
            let v: Vec<[u32; 8]> = cs.into_values().collect();
            let d: [u8; 32] = if v.len() == 1 {
                from_fn(|k| (v[0][k / 4] >> (8 * (k % 4))) as u8)
            } else {
                let t = tree(&self.key, &v, true);
                from_fn(|k| (t[k / 4] >> (8 * (k % 4))) as u8)
            };
            if d != inst.digest(i, p) {
                return Err(format!("C4: instance {i} port {p}: the chunk values do not fold into the committed row digest"));
            }
        }
        Ok(())
    }
}
