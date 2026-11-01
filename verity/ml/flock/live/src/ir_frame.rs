//! `verity/flock-ir-frame/v3`: an IR template's units bound to their frame-v3 rows (lane flock-ir-lowering;
//! `verity_flock.ir_frame` stages the instance file). One block R1CS for the CPU prover and Flock-CUDA's host-witness mode.
//!
//! A run is `nb` consecutive compressions of one BLAKE3 chunk (a chunk's block count, which the last chunk of a row may have
//! short, is a multiple of `nb`): run `r` hashes the chunk's blocks
//! `r nb .. r nb + nb`. Block `b` holds `G` components: their runs at 2^14-bit slots `0 .. C` (run slot `q` at slots
//! `q nb .. q nb + nb`; slots past the components' runs, and holes inside a component, are empty), then their units at
//! 2^unit_log slots `unit_pos0 ..` (an empty unit slot, a hole, is wired only to empty run slots).
//! `A = I ⊗ circuit + Δ` (and B), where Δ makes, inside the block:
//! - each run a chain (chaining input of compression `j ≥ 1` = `out_lo` of compression `j − 1`);
//! - every leaf input bit of every unit a copy of the message bit it hashes (the plan's wiring);
//! - every sub-circuit's constant a copy of the pinned column (compression slot 0's).
//!
//! Public regions, each opened at two verifier points sampled after the commitment (both reps), all values the verifier's:
//! - every compression's counter, block_len and flags (bits 1024..1152: the chunk's index in its row, 64, KEYED_HASH |
//!   CHUNK_START at the chunk's first block | CHUNK_END and, for a row of one chunk, ROOT at its last);
//! - every run's chaining input (the x-row key for a chunk's first run, else the previous run's output value);
//! - every run's `out_lo` at its last compression (a prover-committed public: the run's output value; a chunk's last run's is
//!   the chunk chaining value, or the row digest for a one-chunk row);
//! - every unit's returned-output words, and for a cut template every unit's cut input and cut output words (public, from
//!   the verifier's own instance file, whose cut words the netlist's pinned tail relates).
//!
//! v3 adds, for tensor-core templates (attention): rows whose last chunk is short (a chunk's flags follow its own block
//! count); a public input port, read by the units as cut words and hashed by the verifier into its committed digest instead
//! of by the circuit (`check_public_ports`); zero leaves (a word the IR pads with), wired to an empty run slot, whose all-zero
//! message the verifier-fixed chaining values bind; and outputs the native tail computes, checked against the file's output
//! words (`check_cut_words`) and committed under the output root.
//!
//! So a chunk's runs chain across blocks through their output values, and one circuit serves every block. What the layout
//! means (which leaves each unit slot reads, which output leaves it returns, the row key) is checked against the netlist's
//! pinned leaf maps before anything (`check_leaf_maps`, IR6); the header's copy is never trusted. The verifier folds
//! each row's chunk values into its digest natively (BLAKE3's keyed parent tree) and checks the digests, and the output
//! words, against the frame-v3 roots it recomputes. An empty run slot (padding) hashes `nb` zero blocks from the key at
//! counter 0, flags KEYED_HASH | CHUNK_START (first) | CHUNK_END (last).

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
    chunk::{CHUNK_END, CHUNK_START, KEYED_HASH, ROOT, comp, key_words, log2ceil, row_key, tree},
    ir_block::{CutMap, Group, IrUnitNet, cut_map_of},
    ir_tail::Tables,
};

pub const TAG: &[u8] = b"verity/flock-ir-frame/v3";
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
    /// compressions per run (a chunk's block count, `blocks_of`, is a multiple of it)
    pub nb: usize,
    /// run slots per block
    pub chunks: usize,
    pub units: usize,
    pub unit_log: usize,
    pub unit_pos0: usize,
    pub k_log: usize,
    /// components per block
    pub g: usize,
    pub chunks_per_comp: usize,
    pub units_per_comp: usize,
    pub wiring: Vec<Vec<(usize, usize)>>,
    pub leaf_cols: Vec<usize>,
    /// the units' returned-output group (`ret_words` 0: the units return no output leaf; the tail computes the outputs)
    pub ret_col: usize,
    pub ret_words: usize,
    pub ret_cols: Vec<usize>,
    pub out_leaf: Vec<Vec<usize>>,
}

/// A run slot's contents: (instance, input port, chunk index in the row, run index in the chunk).
pub type ChunkRef = Option<(usize, usize, usize, usize)>;

/// A `flock-ir-frame/v3` instance file.
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
    /// when the tail computes the outputs: every instance's output words (instance-major, ports in order)
    pub inst_outputs: Vec<u16>,
    /// input ports the units read as public cut words: not hashed in the circuit, their digests recomputed from those words
    pub public_ports: Vec<usize>,
    /// every instance's cut words (`cut_words` each)
    pub cuts: Vec<u32>,
    pub key: [u8; 32],
    /// sha256 of every public byte (header, row digests, output words, cut words): what Σ binds; the rows are not in it.
    pub public_sha256: [u8; 32],
}

impl FrameInstances {
    pub fn load(path: &str) -> Result<Self, String> {
        let b = std::fs::read(path).map_err(|e| format!("instances: {e}"))?;
        let nl = b.iter().position(|&c| c == b'\n').ok_or("no header line")?;
        let header: Value = serde_json::from_slice(&b[..nl]).map_err(|e| format!("header: {e}"))?;
        if header["format"] != "flock-ir-frame/v3" {
            return Err("not a flock-ir-frame/v3 instance file".into());
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
            g: g_usize(l, "G")?,
            chunks_per_comp: g_usize(l, "chunks_per_comp")?,
            units_per_comp: g_usize(l, "units_per_comp")?,
            wiring: l["wiring"].as_array().ok_or("no wiring")?.iter().map(|u| {
                u.as_array().map(|a| a.iter().map(|p| (p[0].as_u64().unwrap_or(0) as usize, p[1].as_u64().unwrap_or(0) as usize)).collect()).unwrap_or_default()
            }).collect(),
            leaf_cols: usizes(&l["leaf_cols"]),
            ret_col: l["ret_group"][0].as_u64().unwrap_or(0) as usize,
            ret_words: l["ret_group"][1].as_u64().unwrap_or(0) as usize,
            ret_cols: usizes(&l["ret_cols"]),
            out_leaf: l["out_leaf"].as_array().ok_or("no out_leaf")?.iter().map(usizes).collect(),
        };
        let blocks = header["blocks"].as_array().ok_or("no blocks")?.iter().map(|bl| {
            let chunks = bl[0].as_array().map(|a| a.iter().map(|c| c.as_array().map(|x| {
                let f = |k: usize| x.get(k).and_then(|v| v.as_u64()).unwrap_or(u64::MAX) as usize;
                (f(0), f(1), f(2), f(3))
            })).collect()).unwrap_or_default();
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
        let tail_out = match header["outputs"].as_str() {
            Some("tail") => true,
            Some("units") => false,
            _ => return Err("header: outputs is neither tail nor units".into()),
        };
        if tail_out == (lay.ret_words > 0) || header["layout"]["ret_group"].is_null() != tail_out {
            return Err("the outputs are both, or neither, the units' and the tail's".into());
        }
        let outputs: Vec<F128> = take(16 * units * lay.ret_words)?.chunks(16)
            .map(|c| F128::new(u64::from_le_bytes(c[..8].try_into().unwrap()), u64::from_le_bytes(c[8..].try_into().unwrap()))).collect();
        let out_words: usize = out_ports.iter().map(|p| p.1).sum();
        let inst_outputs: Vec<u16> = take(if tail_out { 2 * n * out_words } else { 0 })?.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let kc = g_usize(&header, "cut_words").unwrap_or(0);
        let cuts: Vec<u32> = take(4 * n * kc)?.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        if at != b.len() {
            return Err("instance file length".into());
        }
        let key: [u8; 32] = hexb(header["frame_v3"]["key"].as_str().ok_or("no frame_v3 key")?)?.try_into().map_err(|_| "key length")?;
        let mut h = Sha256::new();
        h.update(&b[..nl]);
        h.update(&b[pub_start..]);
        let public_sha256: [u8; 32] = h.finalize().into();
        let public_ports = usizes(&header["public_ports"]);
        if public_ports.iter().any(|&p| p >= in_ports.len()) {
            return Err("a public port past the input ports".into());
        }
        Ok(FrameInstances { unit: header["unit"].as_str().unwrap_or("").to_string(), header, n, upi, units, in_ports, out_ports, lay, blocks, rows,
                            digests, outputs, inst_outputs, public_ports, cuts, key, public_sha256 })
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

    /// BLAKE3 blocks of chunk `c` of port `p`'s rows (the last chunk may be short).
    pub fn blocks_of(&self, p: usize, c: usize) -> usize {
        (2 * self.in_ports[p].1 - 1024 * c).min(1024) / 64
    }

    /// Runs of chunk `c` of port `p`.
    pub fn runs_of(&self, p: usize, c: usize) -> usize {
        self.blocks_of(p, c) / self.lay.nb
    }

    pub fn hashed(&self, p: usize) -> bool {
        !self.public_ports.contains(&p)
    }
}

fn hexb(s: &str) -> Result<Vec<u8>, String> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|e| e.to_string())).collect()
}

// --- frame-v3 (verity.ml.commitments.merkle), natively ---------------------------------------------------------------------

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

/// Every instance's output words, port-major `[port][instance][word]`: from the units' returned-output words (the plan's
/// `out_leaf` and `ret_cols`), or the file's own when the tail computes them (`check_cut_words` recomputes those).
pub fn instance_outputs(inst: &FrameInstances) -> Result<Vec<Vec<u16>>, String> {
    let widths: Vec<usize> = inst.out_ports.iter().map(|p| p.1).collect();
    let starts: Vec<usize> = widths.iter().scan(0, |a, &w| { let s = *a; *a += w; Some(s) }).collect();
    if inst.lay.ret_words == 0 {
        let total: usize = widths.iter().sum();
        return Ok(widths.iter().zip(&starts).map(|(&w, &s0)| (0..inst.n).flat_map(|i| inst.inst_outputs[i * total + s0..i * total + s0 + w].to_vec()).collect()).collect());
    }
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

/// The block table against the plan: every hashed port's rows whole BLAKE3 blocks, each chunk's block count a multiple of the
/// run length; every run of every chunk of every hashed port in exactly one run slot of the blocks' components, a public
/// port in none, the slots past the components empty; every unit in exactly one unit slot, beside its component's runs,
/// which are its instance's; an empty unit slot (a hole) wired only to empty run slots, so it reads zeros.
pub fn check_blocks(inst: &FrameInstances) -> Result<(), String> {
    let lay = &inst.lay;
    if lay.nb == 0 || lay.g * lay.chunks_per_comp > lay.chunks || lay.g * lay.units_per_comp != lay.units || lay.wiring.len() != lay.units {
        return Err("the run and component counts are not a flock-ir-frame/v3 layout".into());
    }
    for (p, (name, w)) in inst.in_ports.iter().enumerate() {
        if inst.hashed(p) && ((2 * w) % 64 != 0 || (0..inst.chunks_of(p)).any(|c| inst.blocks_of(p, c) % lay.nb != 0)) {
            return Err(format!("input port {name}: its {}-byte rows are not whole runs of {} blocks", 2 * w, lay.nb));
        }
    }
    let mut seen_c = BTreeSet::new();
    let mut seen_u = BTreeSet::new();
    for (b, (chunks, units)) in inst.blocks.iter().enumerate() {
        if chunks.len() != lay.chunks || units.len() != lay.units {
            return Err(format!("block {b}: {} run slots and {} unit slots, the layout has {} and {}", chunks.len(), units.len(), lay.chunks, lay.units));
        }
        if chunks[lay.g * lay.chunks_per_comp..].iter().any(|c| c.is_some()) {
            return Err(format!("block {b}: a run past its components' slots"));
        }
        for (u, g) in units.iter().enumerate() {
            let k = u / lay.units_per_comp;
            let insts: BTreeSet<usize> = chunks[k * lay.chunks_per_comp..(k + 1) * lay.chunks_per_comp].iter().flatten().map(|x| x.0).collect();
            match g {
                Some(g) => {
                    if !seen_u.insert(*g) || *g >= inst.units || insts != BTreeSet::from([g / inst.upi]) {
                        return Err(format!("block {b}: unit slot {u} holds unit {g}, not of its component's instance"));
                    }
                }
                None if lay.wiring[u].iter().any(|&(q, _)| chunks.get(q).copied().flatten().is_some()) => {
                    return Err(format!("block {b}: empty unit slot {u} is wired to a real run"));
                }
                None => {}
            }
        }
        for (q, c) in chunks.iter().enumerate() {
            if let Some((i, p, ci, r)) = c {
                if *p >= inst.in_ports.len() || !inst.hashed(*p) || *ci >= inst.chunks_of(*p) || *i >= inst.n || *r >= inst.runs_of(*p, *ci)
                    || !seen_c.insert((*i, *p, *ci, *r))
                {
                    return Err(format!("block {b}: run slot {q} holds a run outside the hashed rows or twice"));
                }
            }
        }
    }
    let want_c: usize = (0..inst.in_ports.len()).filter(|&p| inst.hashed(p)).map(|p| (0..inst.chunks_of(p)).map(|c| inst.runs_of(p, c)).sum::<usize>()).sum::<usize>() * inst.n;
    if seen_c.len() != want_c || seen_u.len() != inst.units {
        return Err("the blocks do not hold every run of every hashed row chunk and every unit".into());
    }
    Ok(())
}

/// Red-team-flock-2 IR6: the file's meaning against the netlist's pinned leaf maps (its LEAVES line), not the header's word.
/// The ports are 16-bit words with the pinned leaf counts; the output mapping is the pinned one and the returned-output
/// columns are consecutive 16-bit ports (no returned-output group when the tail computes the outputs); every real unit
/// slot's wiring reads exactly the runs and byte offsets its unit's pinned input leaves occupy in its own instance's hashed
/// rows (so the block assignment is the leaf maps'), and a zero leaf (-1, a word the IR pads with) an empty run slot, whose
/// message the verifier-fixed chaining value binds to zero; the row key is the x-row key.
pub fn check_leaf_maps(inst: &FrameInstances, net: &IrUnitNet) -> Result<(), String> {
    let lv = net.leaves.as_ref().ok_or("the netlist pins no leaf maps (no LEAVES line)")?;
    let ints = |v: &Value| -> Vec<Vec<i64>> { v.as_array().map(|a| a.iter().map(|u| u.as_array().map(|x| x.iter().filter_map(|y| y.as_i64()).collect()).unwrap_or_default()).collect()).unwrap_or_default() };
    let lists = |v: &Value| -> Vec<Vec<usize>> { v.as_array().map(|a| a.iter().map(usizes).collect()).unwrap_or_default() };
    let (lin, lout, pin, pout) = (ints(&lv["in"]), lists(&lv["out"]), lists(&lv["in_ports"]), lists(&lv["out_ports"]));
    if pin.iter().chain(&pout).any(|p| p.len() != 2 || p[1] != 16) {
        return Err("the netlist's ports are not all [leaves, 16] (u16 rows and output words)".into());
    }
    let words = |ports: &[(String, usize)]| ports.iter().map(|p| p.1).collect::<Vec<_>>();
    if words(&inst.in_ports) != pin.iter().map(|p| p[0]).collect::<Vec<_>>() || words(&inst.out_ports) != pout.iter().map(|p| p[0]).collect::<Vec<_>>() {
        return Err("the header's ports are not the netlist's".into());
    }
    let lay = &inst.lay;
    if lin.len() != inst.upi || lout.len() != inst.upi || lay.out_leaf != lout {
        return Err("the output mapping is not the netlist's leaf map".into());
    }
    if lay.ret_words > 0 {
        let g0 = &net.out_groups[0];
        if g0.bits.iter().any(|&b| b != 16) || lay.ret_cols != (0..g0.bits.len()).map(|j| 16 * j).collect::<Vec<_>>() {
            return Err("the returned outputs are not the netlist's 16-bit ports".into());
        }
    } else if lout.iter().any(|o| !o.is_empty()) || !lay.ret_cols.is_empty() {
        return Err("tail-computed outputs, but units that return output leaves".into());
    }
    if inst.key != row_key(b"x") {
        return Err("the row key is not the x-row key (verity/blake3-leaf/v1/x)".into());
    }
    let starts: Vec<usize> = pin.iter().scan(0, |a, p| { let s = *a; *a += p[0]; Some(s) }).collect();
    let total: usize = pin.iter().map(|p| p[0]).sum();
    let run_bytes = 64 * lay.nb;
    for (b, (chunks, units)) in inst.blocks.iter().enumerate() {
        for (u, g) in units.iter().enumerate() {
            let Some(g) = *g else { continue };
            let (i, lu) = (g / inst.upi, g % inst.upi);
            if lay.wiring[u].len() != lin[lu].len() {
                return Err(format!("block {b}: unit slot {u} has {} wired leaf ports, unit {lu} of an instance {}", lay.wiring[u].len(), lin[lu].len()));
            }
            for (j, &pos) in lin[lu].iter().enumerate() {
                let (q, off) = lay.wiring[u][j];
                if pos == -1 {
                    if chunks.get(q).copied().flatten().is_some() || off >= run_bytes {
                        return Err(format!("block {b}: unit slot {u} (unit {g}) reads its zero leaf port {j} from run slot {q}, which is not empty"));
                    }
                    continue;
                }
                let pos = usize::try_from(pos).map_err(|_| format!("unit {lu}: leaf {pos}"))?;
                if pos >= total {
                    return Err(format!("unit {lu}'s leaf {pos} is past its instance's {total} input words"));
                }
                let p = starts.iter().rposition(|&s| s <= pos).unwrap();
                let byte = 2 * (pos - starts[p]);
                if !inst.hashed(p) || chunks.get(q).copied().flatten() != Some((i, p, byte / 1024, byte % 1024 / run_bytes)) || off != byte % run_bytes {
                    return Err(format!("block {b}: unit slot {u} (unit {g}) reads its leaf port {j} from run slot {q} byte {off}, not its input leaf {pos}"));
                }
            }
        }
    }
    Ok(())
}

/// The public ports: exactly the ports every one of whose leaves the netlist's CUT line lists as a public cut word; per
/// instance those words (16-bit) hash, as the port's `blake3-keyed/row/v2` row under the x-row key, to the committed digest.
pub fn check_public_ports(inst: &FrameInstances, cm: &CutMap) -> Result<(), String> {
    let starts: Vec<usize> = inst.in_ports.iter().scan(0, |a, p| { let s = *a; *a += p.1; Some(s) }).collect();
    let by_leaf: BTreeMap<usize, usize> = cm.public.iter().map(|&(k, leaf)| (leaf, k)).collect();
    let covered: Vec<usize> = (0..inst.in_ports.len()).filter(|&p| (starts[p]..starts[p] + inst.in_ports[p].1).all(|l| by_leaf.contains_key(&l))).collect();
    if covered != inst.public_ports || by_leaf.len() != cm.public.len()
        || by_leaf.keys().any(|&l| !covered.iter().any(|&p| (starts[p]..starts[p] + inst.in_ports[p].1).contains(&l)))
    {
        return Err("the header's public ports are not the ports the netlist reads publicly".into());
    }
    for i in 0..inst.n {
        for &p in &covered {
            let mut bytes = Vec::with_capacity(2 * inst.in_ports[p].1);
            for l in starts[p]..starts[p] + inst.in_ports[p].1 {
                let w = inst.cuts[i * cm.words + by_leaf[&l]];
                if w >> 16 != 0 {
                    return Err(format!("instance {i}: public word {l} is not a 16-bit word"));
                }
                bytes.extend_from_slice(&(w as u16).to_le_bytes());
            }
            if *blake3::keyed_hash(&inst.key, &bytes).as_bytes() != inst.digest(i, p) {
                return Err(format!("instance {i}: the public words of port {} do not hash to its committed row digest", inst.in_ports[p].0));
            }
        }
    }
    Ok(())
}

/// Per instance, the netlist's pinned tail evaluated on the cut words the units compute and the public words: every other cut
/// word in the file must be what it gives (the units' cut ports are region claims on these words), a word read by a 16-bit
/// port must fit it, and, when the tail computes the outputs, its output slots must be the file's output words.
pub fn check_cut_words(inst: &FrameInstances, cm: &CutMap, t: &Tables) -> Result<(), String> {
    if cm.words == 0 {
        return Ok(());
    }
    let produced: BTreeSet<usize> = cm.unit_out.iter().flatten().map(|p| p.1).chain(cm.public.iter().map(|p| p.0)).collect();
    let narrow: BTreeSet<usize> = cm.unit_in.iter().flat_map(|u| u.iter().zip(&cm.in_bits).filter(|x| *x.1 == 16).map(|x| x.0 .1)).collect();
    let total: usize = inst.out_ports.iter().map(|p| p.1).sum();
    if !cm.outputs.is_empty() && (cm.outputs.len() != total || inst.lay.ret_words != 0) {
        return Err("the tail's output slots are not the instance's output words".into());
    }
    for i in 0..inst.n {
        let file = &inst.cuts[i * cm.words..(i + 1) * cm.words];
        if let Some(&k) = narrow.iter().find(|&&k| file[k] >> 16 != 0) {
            return Err(format!("instance {i}: cut word {k} feeds a 16-bit port but is not a 16-bit word"));
        }
        let mut slots: Vec<Option<u32>> = (0..cm.words).map(|k| produced.contains(&k).then_some(file[k])).collect();
        crate::ir_tail::eval(&cm.tail, &mut slots, t);
        if let Some(k) = (0..cm.words).find(|&k| slots[k] != Some(file[k])) {
            return Err(format!("instance {i}: cut word {k} is not what the tail computes from the units' words"));
        }
        for (l, &s) in cm.outputs.iter().enumerate() {
            if slots[s] != Some(inst.inst_outputs[i * total + l] as u32) {
                return Err(format!("instance {i}: output word {l} is not what the tail computes"));
            }
        }
    }
    Ok(())
}

// --- the statement ---------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Region {
    /// every compression's counter, block_len and flags
    Params,
    /// every run's chaining input
    CvIn,
    /// every run's output value
    Cv,
    Out,
    CutIn,
    CutOut,
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

/// Flags of block `j` of a chunk of `n` blocks (`single`: the row's only chunk).
pub fn flags_of(j: usize, n: usize, single: bool) -> u32 {
    KEYED_HASH | if j == 0 { CHUNK_START } else { 0 } | if j + 1 == n { CHUNK_END | if single { ROOT } else { 0 } } else { 0 }
}

fn delta(net: &IrUnitNet, lay: &Layout) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    let c = comp();
    let (a, b) = (&c.a_0.rows, &c.b_0.rows);
    let z = b3::Z_CONST_POS;
    let input = |r: usize| assert!(a[r] == [r] && b[r] == [z], "compression row {r} is not an input row");
    for r in (b3::CV_BASE..b3::CV_BASE + 256).chain(b3::T_LO_BASE..b3::FLAGS_BASE + 32) {
        input(r);
    }
    let (mut da, mut db) = (Vec::new(), Vec::new());
    let both = |da: &mut Vec<(u32, u32)>, db: &mut Vec<(u32, u32)>, i: usize, new: usize| {
        for d in [da, db] {
            d.push((i as u32, i as u32));
            d.push((i as u32, new as u32));
        }
    };
    for s in 0..lay.comp_slots {
        let g = |col: usize| (s << b3::K_LOG) + col;
        if s > 0 {
            both(&mut da, &mut db, g(z), PIN);
        }
        if s % lay.nb > 0 {
            for r in 0..256 {
                da.push((g(b3::CV_BASE + r) as u32, g(b3::CV_BASE + r) as u32));
                da.push((g(b3::CV_BASE + r) as u32, (((s - 1) << b3::K_LOG) + b3::OUT_LO_BASE + r) as u32));
            }
        }
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
    /// the netlist's pinned cut structure (no cut words for a template without a cut)
    pub cm: CutMap,
    pub blocks_real: usize,
    pub nbl: usize,
    pub m: usize,
    pub m_pts: usize,
    pub regions: Vec<RegionShape>,
    pub circuit: FrameCircuit,
    pub r1cs: BlockR1cs,
    pub digest: [u8; 32],
    pub key: [u32; 8],
    /// whether each input port's rows are one chunk (ROOT on its last block)
    pub single: Vec<bool>,
    /// where each run sits: (instance, port, chunk, run) -> (block, run slot)
    pub run_at: BTreeMap<(usize, usize, usize, usize), (usize, usize)>,
    /// an empty run slot's output value (`nb` zero blocks from the key, counter 0)
    pub dummy_cv: [u32; 8],
    /// an empty unit slot's returned-output words and cut output words (all-zero inputs)
    pub dummy_out: Vec<F128>,
    pub dummy_cut_out: Vec<F128>,
}

impl FrameStmt {
    pub fn new(net: IrUnitNet, inst: &FrameInstances) -> Result<Self, String> {
        let lay = inst.lay.clone();
        let pow2 = |x: usize| x.is_power_of_two();
        if !(pow2(lay.nb) && pow2(lay.chunks) && lay.comp_slots == lay.chunks * lay.nb && pow2(lay.units) && lay.unit_pos0 % lay.units == 0
            && lay.unit_log == net.unit_log && lay.comp_slots << b3::K_LOG <= lay.unit_pos0 << lay.unit_log
            && (lay.unit_pos0 + lay.units) << lay.unit_log <= 1 << lay.k_log && lay.wiring.len() == lay.units
            && lay.wiring.iter().all(|w| w.len() == lay.leaf_cols.len()))
        {
            return Err("the block layout is not a valid flock-ir-frame/v3 layout".into());
        }
        if lay.wiring.iter().flatten().any(|&(q, off)| q >= lay.chunks || off % 2 == 1 || off >= 64 * lay.nb) {
            return Err("a unit reads outside its block's runs".into());
        }
        let cm = cut_map_of(&net, &inst.header, inst.units)?;
        let ret = lay.ret_words > 0;
        if ret == !cm.outputs.is_empty() || (ret && (net.out_groups[0].col != lay.ret_col || net.out_groups[0].words != lay.ret_words)) {
            return Err("the returned-output group is not the netlist's first output group (or the tail computes the outputs)".into());
        }
        let cut = cm.words > 0;
        let leaf_cols: Vec<usize> = {
            let g = &net.in_groups[0];
            g.bits.iter().scan(g.col * WORD, |a, &w| { let c = *a; *a += w; Some(c) }).collect()
        };
        if leaf_cols != lay.leaf_cols || net.in_groups[0].bits.iter().any(|&w| w != 16) || net.in_groups.len() != 1 + cut as usize
            || net.out_groups.len() != ret as usize + cut as usize
        {
            return Err("the unit's port groups are not a leaf input group of 16-bit ports, a returned-output group and a cut group each".into());
        }
        let blocks_real = inst.blocks.len();
        let nbl = log2ceil(blocks_real).max(3);
        let (kl, m) = (lay.k_log, lay.k_log + nbl);
        let key = key_words(&inst.key);
        let (da, db) = delta(&net, &lay);
        let bits = |lo: usize, n: usize| (lo..lo + n).collect::<Vec<usize>>();
        let lc = lay.comp_slots.trailing_zeros() as usize;
        let lnb = lay.nb.trailing_zeros() as usize;
        let lq = lay.chunks.trailing_zeros() as usize;
        let lu = lay.units.trailing_zeros() as usize;
        let mk = |region: Region, free: Vec<usize>, fixed: u32| -> RegionShape {
            let mask = ((1u32 << kl) - 1) & !free.iter().fold(0u32, |mm, &b| mm | (1 << b));
            assert_eq!(fixed & !mask, 0, "{region:?}");
            assert!(free.len() <= PT_LOCAL, "{region:?}: {} free bits", free.len());
            RegionShape { region, free, mask, fixed }
        };
        let unit_group = |region: Region, g: &Group| {
            mk(region, [bits(0, 7 + g.words.trailing_zeros() as usize), bits(lay.unit_log, lu)].concat(),
               ((g.col as u32) << 7) | (((lay.unit_pos0 >> lu) as u32) << (lay.unit_log + lu)))
        };
        assert!(b3::T_LO_BASE % 128 == 0 && b3::FLAGS_BASE + 32 == b3::T_LO_BASE + 128 && b3::CV_BASE % 256 == 0);
        let mut regions = vec![
            mk(Region::Params, [bits(0, 7), bits(b3::K_LOG, lc)].concat(), b3::T_LO_BASE as u32),
            mk(Region::CvIn, [bits(0, 8), bits(b3::K_LOG + lnb, lq)].concat(), b3::CV_BASE as u32),
            mk(Region::Cv, [bits(0, 8), bits(b3::K_LOG + lnb, lq)].concat(), (b3::OUT_LO_BASE as u32) | (((lay.nb - 1) as u32) << b3::K_LOG)),
        ];
        if ret {
            regions.push(unit_group(Region::Out, &net.out_groups[0]));
        }
        if cut {
            regions.push(unit_group(Region::CutIn, &net.in_groups[1]));
            regions.push(unit_group(Region::CutOut, net.out_groups.last().unwrap()));
        }
        let single: Vec<bool> = (0..inst.in_ports.len()).map(|p| inst.chunks_of(p) == 1).collect();
        let mut run_at = BTreeMap::new();
        for (b, (chunks, _)) in inst.blocks.iter().enumerate() {
            for (q, c) in chunks.iter().enumerate() {
                if let Some(x) = c {
                    run_at.insert(*x, (b, q));
                }
            }
        }
        let mut dummy_cv = key;
        for j in 0..lay.nb {
            dummy_cv = blake3_compress(&dummy_cv, &[0u32; 16], 0, 64, flags_of(j, lay.nb, false))[..8].try_into().unwrap();
        }
        let o = net.outputs(&vec![F128::ZERO; net.in_words]);
        let dummy_out = o[..lay.ret_words].to_vec();
        let dummy_cut_out = match net.out_groups.last() {
            Some(g) if cut => o[g.col - net.out_groups[0].col..g.col - net.out_groups[0].col + g.words].to_vec(),
            _ => vec![],
        };
        let mut h = Sha256::new();
        h.update(TAG);
        h.update(comp().statement_digest());
        h.update(net.sha256);
        h.update(inst.key);
        for x in [blocks_real, nbl, m, kl, lay.comp_slots, lay.nb, lay.chunks, lay.units, lay.unit_log, lay.unit_pos0, lay.ret_col,
                  lay.ret_words, cm.words, regions.len()] {
            h.update((x as u64).to_le_bytes());
        }
        for (p, (_, w)) in inst.in_ports.iter().enumerate() {
            h.update((*w as u64).to_le_bytes());
            h.update([inst.hashed(p) as u8]);
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
        Ok(FrameStmt { lay, net, cm, blocks_real, nbl, m, m_pts: PT_LOCAL + nbl, regions, circuit, r1cs, digest, key, single, run_at, dummy_cv,
                       dummy_out, dummy_cut_out })
    }

    pub fn blocks(&self) -> usize {
        1 << self.nbl
    }

    /// Block `j` of chunk `c` of input port `p` of instance `i`, as 16 message words.
    fn message(inst: &FrameInstances, i: usize, p: usize, c: usize, j: usize) -> [u32; 16] {
        let row = inst.row(i, p);
        let at = 1024 * c + 64 * j;
        from_fn(|w| u32::from_le_bytes(row[at + 4 * w..at + 4 * w + 4].try_into().unwrap()))
    }

    /// A run slot's compressions (chaining input, message, counter, block_len, flags) from chaining input `cv`.
    fn run(&self, inst: &FrameInstances, run: ChunkRef, cv: [u32; 8]) -> Vec<b3::Compression> {
        let lay = &self.lay;
        let mut cv = cv;
        (0..lay.nb).map(|j| {
            let (msg, ctr, f) = match run {
                Some((i, p, c, r)) => (Self::message(inst, i, p, c, r * lay.nb + j), c as u64, flags_of(r * lay.nb + j, inst.blocks_of(p, c), self.single[p])),
                None => ([0; 16], 0, flags_of(j, lay.nb, false)),
            };
            let x = (cv, msg, ctr, 64, f);
            cv = blake3_compress(&cv, &msg, ctr, 64, f)[..8].try_into().unwrap();
            x
        }).collect()
    }

    /// A run's honest chaining input: the key, hashed natively through the chunk's earlier blocks.
    pub fn run_input(&self, inst: &FrameInstances, run: ChunkRef) -> [u32; 8] {
        let mut cv = self.key;
        if let Some((i, p, c, r)) = run {
            for jj in 0..r * self.lay.nb {
                let f = flags_of(jj, inst.blocks_of(p, c), self.single[p]);
                cv = blake3_compress(&cv, &Self::message(inst, i, p, c, jj), c as u64, 64, f)[..8].try_into().unwrap();
            }
        }
        cv
    }

    /// Every compression, block-major (`comp_slots` per block), and each real block's run output values (the publics).
    pub fn compressions(&self, inst: &FrameInstances) -> (Vec<b3::Compression>, Vec<Vec<[u32; 8]>>) {
        let lay = &self.lay;
        let per: Vec<(Vec<b3::Compression>, Vec<[u32; 8]>)> = (0..self.blocks()).into_par_iter().map(|b| {
            let mut out = Vec::with_capacity(lay.comp_slots);
            let mut cvs = Vec::with_capacity(lay.chunks);
            for q in 0..lay.chunks {
                let run = inst.blocks.get(b).and_then(|bl| bl.0[q]);
                let cs = self.run(inst, run, self.run_input(inst, run));
                let last = cs.last().unwrap();
                cvs.push(blake3_compress(&last.0, &last.1, last.2, last.3, last.4)[..8].try_into().unwrap());
                out.extend(cs);
            }
            (out, cvs)
        }).collect();
        let cvs = per.iter().take(self.blocks_real).map(|p| p.1.clone()).collect();
        (per.into_iter().flat_map(|p| p.0).collect(), cvs)
    }

    /// Unit slot `u` of block `b`'s input words: each leaf port the 16-bit word at its wired message bytes, and each cut input
    /// port the instance's cut word (the verifier's file).
    pub fn unit_inputs(&self, inst: &FrameInstances, comps: &[b3::Compression], b: usize, u: usize) -> Vec<F128> {
        let lay = &self.lay;
        let mut w = vec![F128::ZERO; self.net.in_words];
        let put = |w: &mut Vec<F128>, c: usize, v: u64| if c % WORD < 64 { w[c / WORD].lo |= v << (c % WORD) } else { w[c / WORD].hi |= v << (c % WORD - 64) };
        for (j, &(q, off)) in lay.wiring[u].iter().enumerate() {
            let msg = &comps[b * lay.comp_slots + q * lay.nb + off / 64].1;
            let byte = off % 64;
            put(&mut w, lay.leaf_cols[j], ((msg[byte / 4] >> (8 * (byte % 4))) & 0xFFFF) as u64);
        }
        if let Some(g) = inst.blocks.get(b).and_then(|bl| bl.1[u]) {
            let (i, lu) = (g / inst.upi, g % inst.upi);
            for &(c, k) in &self.cm.unit_in[lu] {
                let v = inst.cuts[i * self.cm.words + k] as u64;
                put(&mut w, c, v & 0xFFFF_FFFF);
            }
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

    /// A unit slot's words of a cut port group (`base`: the group's first bit in the ports' column numbering).
    fn cut_group_words(&self, inst: &FrameInstances, g: usize, ports: &[(usize, usize)], words: usize, base: usize) -> Vec<F128> {
        let mut w = vec![F128::ZERO; words];
        let i = g / inst.upi;
        for &(c, k) in ports {
            let (v, t) = (inst.cuts[i * self.cm.words + k] as u64, c - base);
            if t % WORD < 64 { w[t / WORD].lo |= v << (t % WORD) } else { w[t / WORD].hi |= v << (t % WORD - 64) }
        }
        w
    }

    /// Block `b`'s bytes of a region, in free-assignment order.
    fn region_bytes(&self, reg: Region, inst: &FrameInstances, cvs: &[Vec<[u32; 8]>], b: usize) -> Vec<u8> {
        let lay = &self.lay;
        let bl = inst.blocks.get(b);
        let words = |v: Vec<F128>| v.into_iter().flat_map(|w| [w.lo.to_le_bytes(), w.hi.to_le_bytes()].concat()).collect::<Vec<u8>>();
        let cv_bytes = |cv: [u32; 8]| cv.into_iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
        match reg {
            Region::Params => (0..lay.comp_slots).flat_map(|s| {
                let (q, j) = (s / lay.nb, s % lay.nb);
                let (ctr, f) = match bl.and_then(|x| x.0[q]) {
                    Some((_, p, c, r)) => (c as u64, flags_of(r * lay.nb + j, inst.blocks_of(p, c), self.single[p])),
                    None => (0, flags_of(j, lay.nb, false)),
                };
                [ctr.to_le_bytes().to_vec(), 64u32.to_le_bytes().to_vec(), f.to_le_bytes().to_vec()].concat()
            }).collect(),
            Region::CvIn => (0..lay.chunks).flat_map(|q| {
                let cv = match bl.and_then(|x| x.0[q]) {
                    Some((i, p, c, r)) if r > 0 => {
                        let (b0, q0) = self.run_at[&(i, p, c, r - 1)];
                        cvs[b0][q0]
                    }
                    _ => self.key,
                };
                cv_bytes(cv)
            }).collect(),
            Region::Cv => (0..lay.chunks).flat_map(|q| cv_bytes(if bl.and_then(|x| x.0[q]).is_some() { cvs[b][q] } else { self.dummy_cv })).collect(),
            Region::Out => (0..lay.units).flat_map(|u| words(match bl.and_then(|x| x.1[u]) {
                Some(g) => inst.output(g).to_vec(),
                None => self.dummy_out.clone(),
            })).collect(),
            Region::CutIn => {
                let gr = &self.net.in_groups[1];
                (0..lay.units).flat_map(|u| words(match bl.and_then(|x| x.1[u]) {
                    Some(g) => self.cut_group_words(inst, g, &self.cm.unit_in[g % inst.upi], gr.words, gr.col * WORD),
                    None => vec![F128::ZERO; gr.words],
                })).collect()
            }
            Region::CutOut => {
                let gr = self.net.out_groups.last().unwrap();
                let base = (gr.col - self.net.out_groups[0].col) * WORD;
                (0..lay.units).flat_map(|u| words(match bl.and_then(|x| x.1[u]) {
                    Some(g) => self.cut_group_words(inst, g, &self.cm.unit_out[g % inst.upi], gr.words, base),
                    None => self.dummy_cut_out.clone(),
                })).collect()
            }
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

    /// The prover's publics: every real block's run output values, 32 bytes each (8 u32 LE).
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

    /// C4: each row's chunk values (each chunk's last run's output value) fold (natively, BLAKE3's keyed parent tree; a one-chunk
    /// row's value is its ROOT output) into the row digest the verifier's file commits under its frame-v3 root.
    pub fn check_digests(&self, inst: &FrameInstances, cvs: &[Vec<[u32; 8]>]) -> Result<(), String> {
        let mut rows: BTreeMap<(usize, usize), BTreeMap<usize, [u32; 8]>> = BTreeMap::new();
        for (b, (chunks, _)) in inst.blocks.iter().enumerate() {
            for (q, c) in chunks.iter().enumerate() {
                if let Some((i, p, ci, r)) = c
                    && *r + 1 == inst.runs_of(*p, *ci)
                {
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
