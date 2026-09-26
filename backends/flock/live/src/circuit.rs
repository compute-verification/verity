//! `verity/flock-circuit` (lane flock-circuit): any lowered IR template, its frame-v3 rows hashed in the circuit, and
//! nothing public but the frame-v3 roots, the row digests they are built from, the declared outputs and the pinned layout.
//! `verity_flock.circuit` composes the pinned circuit (`flock-circuit`) and stages the instance files.
//!
//! A block holds whole instances (VUs) in aligned ranges of slots: BLAKE3 compressions (2^14 bits; per input port a range of
//! chunk runs and, for a port of several chunks, a range of its parent-tree compressions, the root first), the template's
//! units (2^unit_log bits) and a reserved mask slot of free cells. `A = I ⊗ A_0 + Δ` (and B), where A_0 places each slot
//! type's circuit on its ranges and Δ, derived here from the pinned circuit alone, makes inside the block:
//! - every compression after a chunk's first read the previous compression's output value as its chaining value;
//! - every parent compression's message the output values of its two children;
//! - every unit's leaf input bits copies of the message bits they hash, and a zero leaf a copy of a forced-zero row;
//! - every slot's constant a copy of the pinned column.
//!
//! Public regions, each opened at the verifier's points (drawn after the commitment, both reps), values all the verifier's:
//! every compression's counter, block length and flags; every chain start's chaining value (the x-row key); every row's
//! digest (the output value of its root compression); every unit's returned-output words. So the chaining values, the chunk
//! values and every word between slots stay private. A padding instance is the all-zero instance, whose digests (checked
//! here: keyed BLAKE3 of zero rows) and outputs (forced by its circuit) are pinned.
//!
//! The verifier evaluates no part of the computation (standing rule): it checks the proof, the frame-v3 roots against the
//! public digests and outputs, and nothing else. A circuit carrying a verifier-evaluated tail (a `CUT` line, or a tail,
//! public cut words or native words in META) is refused at parse; a template's tail is a slot type of the circuit or the
//! template has no statement here.

use std::{collections::BTreeMap, sync::OnceLock};

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
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    chunk::{CHUNK_END, CHUNK_START, KEYED_HASH, PARENT, ROOT, comp, key_words, log2ceil},
    ir_block::IrUnitNet,
    lookup::LookupNet,
};

pub const TAG: &[u8] = b"verity/flock-circuit";
pub const FORMAT: &str = "flock-circuit";
pub const INST_FORMAT: &str = "flock-circuit-inputs";
pub const STATEMENT: &str = "verity/flock-circuit";
/// Free in-block coordinates per verifier point (`r[..PT_LOCAL]`); block coordinates follow.
pub const PT_LOCAL: usize = 24;
/// The leaf-commitment scheme this prover and verifier implement for the proof's own Merkle trees, as the composite circuit
/// pins it (`verity_flock.circuit.LEAF_SCHEME`: today's id and the `hm96-sha256/v1` room it reserves).
pub fn leaf_scheme() -> Value {
    json!({"id": "flock-leaf/sha256-unsalted", "salt_bytes": 0,
           "target": {"scheme": "hm96-sha256/v1", "inner": "sha256 column digest", "salt_bytes_per_leaf": 128,
                      "salt_source": "os generator, one draw per leaf", "key_bytes_per_proof": 160,
                      "key": "per proof from the os generator, in the hello before the root"}})
}
const WORD: usize = 128;
const SUB_LOG: usize = b3::K_LOG;
const FRAME: &[u8] = b"veritor/protocol/merkle/frame/v3\0";

pub fn hexb(s: &str) -> Result<Vec<u8>, String> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|e| e.to_string())).collect()
}

fn us(v: &Value, k: &str) -> Result<usize, String> {
    v[k].as_u64().map(|x| x as usize).ok_or(format!("circuit META: {k} missing"))
}

// --- the pinned circuit ------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Comp,
    /// index into `Composite::nets`
    Net(usize),
    Mask,
}

#[derive(Clone, Debug)]
pub enum Role {
    /// chunk runs of input port `p`: run slot `q` holds `(vu, chunk)` or nothing (a dummy run)
    Runs { p: usize, slots: Vec<Option<(usize, usize)>> },
    /// port `p`'s parent-tree compressions: per VU `chunks` slots, heap order (root first), the last a dummy
    Parents { p: usize },
    None,
}

#[derive(Clone, Debug)]
pub struct Range {
    pub kind: Kind,
    pub slot_log: usize,
    pub pos0: usize,
    pub count: usize,
    pub role: Role,
}

impl Range {
    pub fn start(&self) -> usize {
        self.pos0 << self.slot_log
    }
    pub fn slot(&self, s: usize) -> usize {
        (self.pos0 + s) << self.slot_log
    }
}

#[derive(Clone, Debug)]
pub struct Port {
    pub name: String,
    pub words: usize,
    pub chunks: usize,
    pub nb: usize,
}

/// What a compression slot computes (block-uniform).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CompRole {
    Run { g: usize, p: usize, c: usize, j: usize },
    DummyRun { j: usize },
    Parent { g: usize, p: usize, k: usize },
    DummyParent,
}

/// A copy of every bit of output port `src_port` of slot `(src_net, src_idx)` into input port `dst_port` of slot
/// `(dst_net, dst_idx)`, both instance-local (per VU).
#[derive(Clone, Copy, Debug)]
pub struct Wire {
    pub dst: (usize, usize, usize),
    pub src: (usize, usize, usize),
}

pub struct Composite {
    pub name: String,
    pub sha256: [u8; 32],
    pub meta: Value,
    pub nets: Vec<(String, IrUnitNet)>,
    pub ranges: Vec<Range>,
    pub ports: Vec<Port>,
    pub out_words: Vec<usize>,
    pub g: usize,
    pub k_log: usize,
    pub upv: usize,
    pub unit_net: usize,
    pub leaves_in: Vec<Vec<i64>>,
    pub leaves_out: Vec<Vec<usize>>,
    pub key: [u8; 32],
    pub dummy_digests: Vec<[u8; 32]>,
    pub dummy_outputs: Vec<u16>,
    pub leaf_scheme: Value,
    /// slots per VU of each circuit
    pub per_vu: Vec<usize>,
    pub wires: Vec<Wire>,
    /// the lookup slot types' fast witnesses, by circuit index
    pub lookups: BTreeMap<usize, LookupNet>,
}

fn table_of<'a>(t: &'a crate::ir_tail::Tables, name: &str) -> &'a [u32] {
    match name {
        "rsq" => &t.rsq,
        "sqrt" => &t.sqrt,
        "ex2" => &t.ex2,
        _ => &t.rcp,
    }
}

/// Every input (output) port of a circuit: its first column in the slot and its width, groups in order.
pub fn port_cols(groups: &[crate::ir_block::Group]) -> Vec<(usize, usize)> {
    groups.iter().flat_map(|g| g.bits.iter().scan(g.col * WORD, |a, &w| { let c = *a; *a += w; Some((c, w)) }).collect::<Vec<_>>()).collect()
}

impl Composite {
    /// `tables`: the directory of the pinned MUFU tables (`verity_flock.ir_lower.write_tables`), when the circuit has lookups.
    pub fn parse(text: &str, tables: Option<&str>) -> Result<Self, String> {
        let sha256: [u8; 32] = Sha256::digest(text.as_bytes()).into();
        let mut lines = text.split_inclusive('\n');
        let head = lines.next().ok_or("empty circuit")?.trim_end();
        let name = head.strip_prefix("flock-circuit ").ok_or("not a flock-circuit circuit")?.to_string();
        let meta: Value = serde_json::from_str(lines.next().and_then(|l| l.trim_end().strip_prefix("META ")).ok_or("no META line")?)
            .map_err(|e| format!("META: {e}"))?;
        if meta["format"] != FORMAT || meta["statement"] != STATEMENT {
            return Err("META: not verity/flock-circuit".into());
        }
        let mut nets = Vec::new();
        let mut lookups_fast: Vec<(usize, LookupNet)> = Vec::new();
        while let Some(l) = lines.next() {
            let h: Vec<&str> = l.split_whitespace().collect();
            if h.len() != 3 || h[0] != "CIRCUIT" {
                return Err(format!("circuit: expected a CIRCUIT line, found {:?}", l.chars().take(40).collect::<String>()));
            }
            let n: usize = h[2].parse().map_err(|_| "CIRCUIT sections count")?;
            let body: String = (0..n).map(|_| lines.next().unwrap_or("")).collect();
            let net = std::panic::catch_unwind(|| IrUnitNet::parse(&body)).map_err(|_| format!("CIRCUIT {}: not a flock-ir-unit/v2 expanded circuits", h[1]))?;
            if net.cut.is_some() {
                return Err(format!("CIRCUIT {}: a CUT line (a tail the verifier evaluates); the verifier evaluates no part of the computation", h[1]));
            }
            nets.push((h[1].to_string(), net));
        }
        for k in ["tail", "cut", "cut_words", "public_ports", "public", "native", "outputs_by_tail"] {
            if meta.get(k).is_some() {
                return Err(format!("META: {k:?} (a verifier-evaluated part of the computation); the verifier only checks proofs"));
            }
        }
        let lk = meta["lookups"].as_object().cloned().unwrap_or_default();
        if !lk.is_empty() {
            let need: std::collections::BTreeSet<&'static str> = crate::ir_tail::TABLE_SHA256.iter().map(|t| t.0).filter(|t| lk.contains_key(*t)).collect();
            if need.len() != lk.len() {
                return Err("META: a lookup of a table the statement does not pin".into());
            }
            let t = crate::ir_tail::Tables::load(tables.ok_or("the circuit has lookups: pass --tables DIR")?, &need)?;
            for (name, spec) in &lk {
                let pinned = crate::ir_tail::TABLE_SHA256.iter().find(|x| x.0 == name).unwrap();
                if spec["sha256"].as_str() != Some(pinned.1) || spec["table"].as_str() != Some(name.as_str()) || spec["out_bits"] != 31 {
                    return Err(format!("META: lookup {name} is not the pinned table"));
                }
                let n = spec["in_bits"].as_u64().unwrap_or(0) as usize;
                let lo = spec["lo_bits"].as_u64().unwrap_or(0) as usize;
                if 1usize << n != pinned.2 || !(2..n).contains(&lo) {
                    return Err(format!("META: lookup {name}'s index is not its table's"));
                }
                let words: &'static [u32] = Box::leak(table_of(&t, name).to_vec().into_boxed_slice());
                let (net, fast) = LookupNet::build(&format!("lookup:{name}"), words, n, lo, 31);
                nets.push((format!("lookup:{name}"), net));
                lookups_fast.push((nets.len() - 1, fast));
            }
        }
        let ports: Vec<Port> = meta["ports"].as_array().ok_or("META: ports")?.iter().map(|p| Port {
            name: p[0].as_str().unwrap_or("").into(), words: p[1].as_u64().unwrap_or(0) as usize, chunks: p[2].as_u64().unwrap_or(0) as usize,
            nb: p[3].as_u64().unwrap_or(0) as usize,
        }).collect();
        for p in &ports {
            let b = 2 * p.words;
            let ok = p.nb.is_power_of_two() && p.chunks >= 1 && b % 64 == 0 && p.chunks == b.div_ceil(1024) && p.nb == b.min(1024) / 64
                && (p.chunks == 1 || (b % 1024 == 0 && p.chunks.is_power_of_two()));
            if !ok {
                return Err(format!("port {}: not whole power-of-two chunk runs", p.name));
            }
        }
        let g = us(&meta, "vus_per_block")?;
        let upv = us(&meta, "units_per_vu")?;
        let unit_net = nets.iter().position(|n| n.0 == "unit").ok_or("no unit's expanded circuit")?;
        let mut ranges = Vec::new();
        for r in meta["ranges"].as_array().ok_or("META: ranges")? {
            let kind = match r["type"].as_str() {
                Some("comp") => Kind::Comp,
                Some("mask") => Kind::Mask,
                Some("unit") => Kind::Net(nets.iter().position(|n| Some(n.0.as_str()) == r["net"].as_str()).ok_or("a range of an unknown circuit")?),
                _ => return Err("a range of an unknown type".into()),
            };
            let p = r["port"].as_u64().map(|x| x as usize);
            let role = match (r["role"].as_str(), p) {
                (Some("runs"), Some(p)) => Role::Runs { p, slots: r["slots"].as_array().ok_or("runs without slots")?.iter()
                    .map(|x| x.as_array().map(|a| (a[0].as_u64().unwrap_or(0) as usize, a[1].as_u64().unwrap_or(0) as usize))).collect() },
                (Some("parents"), Some(p)) => Role::Parents { p },
                (None, None) => Role::None,
                _ => return Err("a range of an unknown role".into()),
            };
            ranges.push(Range { kind, slot_log: us(r, "slot_log")?, pos0: us(r, "pos0")?, count: us(r, "count")?, role });
        }
        let ints = |v: &Value| -> Vec<Vec<i64>> { v.as_array().map(|a| a.iter().map(|u| u.as_array().map(|x| x.iter().filter_map(|y| y.as_i64()).collect()).unwrap_or_default()).collect()).unwrap_or_default() };
        let leaves_in = ints(&meta["leaves_in"]);
        let leaves_out: Vec<Vec<usize>> = ints(&meta["leaves_out"]).into_iter().map(|u| u.into_iter().map(|x| x as usize).collect()).collect();
        let key: [u8; 32] = hexb(meta["row_key"].as_str().ok_or("META: row_key")?)?.try_into().map_err(|_| "row key length")?;
        let dummy_digests = meta["dummy"]["digests"].as_array().ok_or("META: dummy digests")?.iter()
            .map(|d| hexb(d.as_str().unwrap_or("")).and_then(|b| b.try_into().map_err(|_| "digest length".to_string()))).collect::<Result<Vec<[u8; 32]>, String>>()?;
        let dummy_outputs = meta["dummy"]["outputs"].as_array().ok_or("META: dummy outputs")?.iter().map(|x| x.as_u64().unwrap_or(u64::MAX) as u16).collect();
        let out_words = meta["out_ports"].as_array().ok_or("META: out_ports")?.iter().map(|p| p[1].as_u64().unwrap_or(0) as usize).collect();
        let net_of = |nm: &str| nets.iter().position(|n| n.0 == nm).ok_or(format!("META: unknown circuit {nm}"));
        let mut per_vu = vec![0usize; nets.len()];
        per_vu[unit_net] = upv;
        if let Some(m) = meta["per_vu"].as_object() {
            for (k, v) in m {
                per_vu[net_of(k)?] = v.as_u64().unwrap_or(0) as usize;
            }
        }
        let mut wires = Vec::new();
        for w in meta["wires"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
            let n = |i: usize| w[i].as_str().ok_or("a wire's circuit".to_string()).and_then(net_of);
            let u = |i: usize| w[i].as_u64().map(|x| x as usize).ok_or("a wire's index".to_string());
            wires.push(Wire { dst: (n(0)?, u(1)?, u(2)?), src: (n(3)?, u(4)?, u(5)?) });
        }
        let c = Composite {
            name, sha256, k_log: us(&meta, "k_log")?, leaf_scheme: meta["leaf_scheme"].clone(), meta, nets, ranges, ports,
            out_words, g, upv, unit_net, leaves_in, leaves_out, key, dummy_digests, dummy_outputs, per_vu, wires,
            lookups: lookups_fast.into_iter().collect(),
        };
        c.check()?;
        Ok(c)
    }

    /// The layout's shape: aligned, disjoint ranges inside the block; the comp ranges' roles cover every run and parent of
    /// every VU exactly once; the unit ranges hold `g x units_per_vu` units whose leaf maps are the unit's expanded circuit's ports.
    fn check(&self) -> Result<(), String> {
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for r in &self.ranges {
            let span = r.count.next_power_of_two() << r.slot_log;
            if r.start() % span != 0 || r.start() + span > 1 << self.k_log {
                return Err("a range is not aligned to its span or overruns the block".into());
            }
            if let Kind::Net(i) = r.kind
                && self.nets[i].1.unit_log != r.slot_log
            {
                return Err("a unit range's slot size is not its circuit's".into());
            }
            if (r.kind == Kind::Comp) != (r.slot_log == SUB_LOG && !matches!(r.role, Role::None)) || (r.kind == Kind::Mask && r.count != 1) {
                return Err("a range's type, role and slot size disagree".into());
            }
            spans.push((r.start(), r.start() + span));
        }
        spans.sort();
        if spans.windows(2).any(|w| w[0].1 > w[1].0) {
            return Err("two ranges overlap".into());
        }
        let mut runs = BTreeMap::new();
        let mut parents = BTreeMap::new();
        for r in &self.ranges {
            match &r.role {
                Role::Runs { p, slots } => {
                    let port = self.ports.get(*p).ok_or("runs of an unknown port")?;
                    if slots.len() * port.nb != r.count || !r.count.is_power_of_two() {
                        return Err("a runs range's slot list is not its count".into());
                    }
                    for x in slots.iter().flatten() {
                        if x.0 >= self.g || x.1 >= port.chunks || runs.insert((x.0, *p, x.1), ()).is_some() {
                            return Err("a run outside the block's VUs or listed twice".into());
                        }
                    }
                }
                Role::Parents { p } => {
                    let port = self.ports.get(*p).ok_or("parents of an unknown port")?;
                    if port.chunks < 2 || r.count != self.g * port.chunks || parents.insert(*p, ()).is_some() {
                        return Err("a parents range is not one per multi-chunk port".into());
                    }
                }
                Role::None => {}
            }
        }
        let want: usize = self.ports.iter().map(|p| p.chunks).sum::<usize>() * self.g;
        if runs.len() != want || parents.len() != self.ports.iter().filter(|p| p.chunks > 1).count() {
            return Err("the runs do not cover every chunk of every VU (or a multi-chunk port has no parent tree)".into());
        }
        let unit = &self.nets[self.unit_net].1;
        let ur: Vec<&Range> = self.ranges.iter().filter(|r| r.kind == Kind::Net(self.unit_net)).collect();
        if ur.len() != 1 || ur[0].count != self.g * self.upv || !ur[0].count.is_power_of_two() {
            return Err("not one unit range of vus_per_block x units_per_vu (a power of two) units".into());
        }
        if unit.in_groups.is_empty() || unit.in_groups.len() > 2 || unit.out_groups.is_empty() || unit.out_groups.len() > 2
            || unit.in_groups[0].bits.iter().any(|&b| b != 16) || unit.out_groups[0].bits.iter().any(|&b| b != 16)
        {
            return Err("the unit's ports are not a group of 16-bit leaf inputs (then cut inputs) and one of 16-bit returned outputs (then cut outputs)".into());
        }
        for (i, r) in self.ranges.iter().enumerate() {
            if let Kind::Net(n) = r.kind
                && r.count != self.g * self.per_vu[n]
            {
                return Err(format!("range {i} of {}: {} slots, not {} per VU", self.nets[n].0, r.count, self.per_vu[n]));
            }
        }
        for (n, (nm, _)) in self.nets.iter().enumerate() {
            let ranges = self.ranges.iter().filter(|r| r.kind == Kind::Net(n)).count();
            if ranges != 1 {
                return Err(format!("circuit {nm} has {ranges} ranges, not one"));
            }
        }
        // every input port of every slot is wired exactly once: the unit's leaf ports by its leaf map, the rest by a wire
        let mut wired: BTreeMap<(usize, usize, usize), usize> = BTreeMap::new();
        for w in &self.wires {
            let (dn, di, dp) = w.dst;
            let (sn, si, sp) = w.src;
            let (dc, sc) = (port_cols(&self.nets[dn].1.in_groups), port_cols(&self.nets[sn].1.out_groups));
            if di >= self.per_vu[dn] || si >= self.per_vu[sn] || dp >= dc.len() || sp >= sc.len() || dc[dp].1 != sc[sp].1 {
                return Err(format!("a wire {w:?} outside its slots' ports or of unequal widths"));
            }
            if dn == self.unit_net && dp < unit.in_groups[0].bits.len() {
                return Err("a wire into a unit's leaf port".into());
            }
            *wired.entry(w.dst).or_default() += 1;
        }
        for (n, (nm, net)) in self.nets.iter().enumerate() {
            let first = if n == self.unit_net { net.in_groups[0].bits.len() } else { 0 };
            for i in 0..self.per_vu[n] {
                for p in first..port_cols(&net.in_groups).len() {
                    if wired.get(&(n, i, p)).copied().unwrap_or(0) != 1 {
                        return Err(format!("input port {p} of {nm} slot {i} is not wired exactly once"));
                    }
                }
            }
        }
        let n_in: usize = self.ports.iter().map(|p| p.words).sum();
        let n_out: usize = self.out_words.iter().sum();
        if self.leaves_in.len() != self.upv || self.leaves_out.len() != self.upv {
            return Err("the leaf maps are not one per unit".into());
        }
        let mut seen_out = vec![false; n_out];
        for u in 0..self.upv {
            if self.leaves_in[u].len() != unit.in_groups[0].bits.len() || self.leaves_in[u].iter().any(|&x| x < -1 || x >= n_in as i64)
                || self.leaves_out[u].len() != unit.out_groups[0].bits.len()
            {
                return Err(format!("unit {u}'s leaf maps are not its circuit's ports"));
            }
            for &o in &self.leaves_out[u] {
                if o >= n_out || std::mem::replace(&mut seen_out[o], true) {
                    return Err(format!("output leaf {o} is outside the outputs or returned twice"));
                }
            }
        }
        if seen_out.iter().any(|x| !x) {
            return Err("an output leaf no unit returns".into());
        }
        if unit.useful >= 1 << unit.unit_log {
            return Err("the unit fills its slot: no forced-zero row for a zero leaf".into());
        }
        if self.dummy_digests.len() != self.ports.len() || self.dummy_outputs.len() != n_out {
            return Err("the pinned dummy values are not one per port and output leaf".into());
        }
        if self.leaf_scheme != leaf_scheme() {
            return Err(format!("leaf-commitment scheme {} is not the one this verifier implements", self.leaf_scheme));
        }
        Ok(())
    }

    pub fn range_of(&self, n: usize) -> &Range {
        self.ranges.iter().find(|r| r.kind == Kind::Net(n)).unwrap()
    }

    /// Block-local column of slot `i` (of VU `g`) of circuit `n`.
    pub fn slot_col(&self, n: usize, g: usize, i: usize) -> usize {
        self.range_of(n).slot(g * self.per_vu[n] + i)
    }

    pub fn unit(&self) -> &IrUnitNet {
        &self.nets[self.unit_net].1
    }

    pub fn unit_range(&self) -> &Range {
        self.ranges.iter().find(|r| r.kind == Kind::Net(self.unit_net)).unwrap()
    }

    pub fn row_bytes(&self) -> usize {
        self.ports.iter().map(|p| 2 * p.words).sum()
    }

    /// The role of compression slot `s` of range `r`.
    pub fn comp_role(&self, r: &Range, s: usize) -> CompRole {
        match &r.role {
            Role::Runs { p, slots } => {
                let nb = self.ports[*p].nb;
                match slots[s / nb] {
                    Some((g, c)) => CompRole::Run { g, p: *p, c, j: s % nb },
                    None => CompRole::DummyRun { j: s % nb },
                }
            }
            Role::Parents { p } => {
                let ch = self.ports[*p].chunks;
                if s % ch == ch - 1 { CompRole::DummyParent } else { CompRole::Parent { g: s / ch, p: *p, k: s % ch } }
            }
            Role::None => unreachable!("not a comp range"),
        }
    }

    /// Counter, block length and flags of a compression.
    pub fn params(&self, role: CompRole) -> (u64, u32, u32) {
        match role {
            CompRole::Run { p, c, j, .. } => {
                let port = &self.ports[p];
                let end = j + 1 == port.nb;
                (c as u64, 64, KEYED_HASH | if j == 0 { CHUNK_START } else { 0 } | if end { CHUNK_END | if port.chunks == 1 { ROOT } else { 0 } } else { 0 })
            }
            CompRole::DummyRun { j } => (0, 64, KEYED_HASH | if j == 0 { CHUNK_START } else { 0 }),
            CompRole::Parent { k, .. } => (0, 64, PARENT | KEYED_HASH | if k == 0 { ROOT } else { 0 }),
            CompRole::DummyParent => (0, 64, PARENT | KEYED_HASH),
        }
    }

    /// Block-local column of compression slot `(range, s)` holding the last compression of VU `g`'s chunk `c` of port `p`.
    fn chunk_end(&self, g: usize, p: usize, c: usize) -> usize {
        for r in &self.ranges {
            if let Role::Runs { p: rp, slots } = &r.role
                && *rp == p
                && let Some(q) = slots.iter().position(|x| *x == Some((g, c)))
            {
                return r.slot(q * self.ports[p].nb + self.ports[p].nb - 1);
            }
        }
        unreachable!("every chunk has a run")
    }

    fn parents_range(&self, p: usize) -> &Range {
        self.ranges.iter().find(|r| matches!(r.role, Role::Parents { p: q } if q == p)).unwrap()
    }

    /// Block-local column of VU `g`'s byte `byte` of port `p`: the message bit where it is hashed.
    fn message_col(&self, g: usize, p: usize, byte: usize) -> usize {
        let port = &self.ports[p];
        let (c, rest) = (byte / 1024, byte % 1024);
        let (j, off) = (rest / 64, rest % 64);
        let end = self.chunk_end(g, p, c);
        end - ((port.nb - 1 - j) << SUB_LOG) + b3::M_BASE + 8 * off
    }
}

/// The pinned constant column: the first slot of the first range that has one.
pub fn pin_of(c: &Composite) -> usize {
    let r = c.ranges.iter().find(|r| r.kind != Kind::Mask).unwrap();
    r.start() + match r.kind {
        Kind::Comp => b3::Z_CONST_POS,
        Kind::Net(i) => c.nets[i].1.const_pos,
        Kind::Mask => unreachable!(),
    }
}

/// Δ (A and B), derived from the pinned circuit: chaining, parent messages, unit leaf wiring, constants.
pub fn delta(c: &Composite, pin: usize) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    let cc = comp();
    let (a0, b0) = (&cc.a_0.rows, &cc.b_0.rows);
    let z = b3::Z_CONST_POS;
    for r in (b3::CV_BASE..b3::CV_BASE + 256).chain(b3::M_BASE..b3::M_BASE + 512).chain(b3::T_LO_BASE..b3::FLAGS_BASE + 32) {
        assert!(a0[r] == [r] && b0[r] == [z], "compression row {r} is not an input row");
    }
    let (mut da, mut db) = (Vec::new(), Vec::new());
    let a_only = |da: &mut Vec<(u32, u32)>, i: usize, src: usize| {
        da.push((i as u32, i as u32));
        da.push((i as u32, src as u32));
    };
    let both = |da: &mut Vec<(u32, u32)>, db: &mut Vec<(u32, u32)>, i: usize, src: usize| {
        for d in [da, db] {
            d.push((i as u32, i as u32));
            d.push((i as u32, src as u32));
        }
    };
    for r in &c.ranges {
        for s in 0..r.count {
            let base = r.slot(s);
            let konst = match r.kind {
                Kind::Comp => Some(base + z),
                Kind::Net(i) => Some(base + c.nets[i].1.const_pos),
                Kind::Mask => None,
            };
            if let Some(k) = konst
                && k != pin
            {
                both(&mut da, &mut db, k, pin);
            }
            if r.kind != Kind::Comp {
                continue;
            }
            match c.comp_role(r, s) {
                CompRole::Run { j, .. } | CompRole::DummyRun { j } if j > 0 => {
                    for t in 0..256 {
                        a_only(&mut da, base + b3::CV_BASE + t, base - (1 << SUB_LOG) + b3::OUT_LO_BASE + t);
                    }
                }
                CompRole::Parent { g, p, k } => {
                    let ch = c.ports[p].chunks;
                    let pr = c.parents_range(p);
                    for (side, h) in [2 * k + 1, 2 * k + 2].into_iter().enumerate() {
                        let child = if h < ch - 1 { pr.slot(g * ch + h) } else { c.chunk_end(g, p, h - (ch - 1)) };
                        for t in 0..256 {
                            a_only(&mut da, base + b3::M_BASE + 256 * side + t, child + b3::OUT_LO_BASE + t);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let unit = c.unit();
    let ur = c.unit_range();
    let leaf_cols: Vec<usize> = {
        let g0 = &unit.in_groups[0];
        g0.bits.iter().scan(g0.col * WORD, |a, &w| { let x = *a; *a += w; Some(x) }).collect()
    };
    let starts: Vec<usize> = c.ports.iter().scan(0, |a, p| { let s = *a; *a += p.words; Some(s) }).collect();
    for k in 0..ur.count {
        let (g, u) = (k / c.upv, k % c.upv);
        let base = ur.slot(k);
        for (j, &leaf) in c.leaves_in[u].iter().enumerate() {
            for t in 0..16 {
                let src = if leaf < 0 {
                    base + unit.useful
                } else {
                    let leaf = leaf as usize;
                    let p = starts.iter().rposition(|&s| s <= leaf).unwrap();
                    c.message_col(g, p, 2 * (leaf - starts[p])) + t
                };
                both(&mut da, &mut db, base + leaf_cols[j] + t, src);
            }
        }
    }
    for g in 0..c.g {
        for w in &c.wires {
            let (dc, sc) = (port_cols(&c.nets[w.dst.0].1.in_groups)[w.dst.2], port_cols(&c.nets[w.src.0].1.out_groups)[w.src.2]);
            let (dst, src) = (c.slot_col(w.dst.0, g, w.dst.1) + dc.0, c.slot_col(w.src.0, g, w.src.1) + sc.0);
            for t in 0..dc.1 {
                both(&mut da, &mut db, dst + t, src + t);
            }
        }
    }
    (da, db)
}

// --- the instance files ------------------------------------------------------------------------------------------------

/// A `flock-circuit-inputs` file: the prover's has the rows, the verifier's (`rows: false`) only the public part.
pub struct Instances {
    pub header: Value,
    pub n: usize,
    pub rows: Option<Vec<u8>>,
    pub digests: Vec<[u8; 32]>,
    pub outputs: Vec<u16>,
    /// sha256 of the header line and every public byte after the rows (digests, outputs): what Σ binds
    pub public_sha256: [u8; 32],
}

impl Instances {
    pub fn load(path: &str, c: &Composite) -> Result<Self, String> {
        let b = std::fs::read(path).map_err(|e| format!("instances: {e}"))?;
        let nl = b.iter().position(|&x| x == b'\n').ok_or("no header line")?;
        let mut header: Value = serde_json::from_slice(&b[..nl]).map_err(|e| format!("header: {e}"))?;
        if header["format"] != INST_FORMAT || header["statement"] != STATEMENT {
            return Err("not a flock-circuit-inputs file".into());
        }
        if header["circuit_sha256"].as_str() != Some(crate::hex(&c.sha256).as_str()) {
            return Err("the instance file was staged for another circuit".into());
        }
        let n = header["instances"].as_u64().ok_or("header: instances")? as usize;
        let has_rows = header["rows"].as_bool().ok_or("header: rows")?;
        let rb = c.row_bytes();
        let (n_out, np) = (c.out_words.iter().sum::<usize>(), c.ports.len());
        let body = &b[nl + 1..];
        let rows_len = if has_rows { n * rb } else { 0 };
        if body.len() != rows_len + 32 * n * np + 2 * n * n_out {
            return Err("instance file length".into());
        }
        let rows = has_rows.then(|| body[..rows_len].to_vec());
        let pubs = &body[rows_len..];
        let digests = pubs[..32 * n * np].chunks(32).map(|x| x.try_into().unwrap()).collect();
        let outputs = pubs[32 * n * np..].chunks(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect();
        header["rows"] = Value::Bool(false);
        let mut h = Sha256::new();
        h.update(serde_json::to_vec(&sorted(&header)).unwrap());
        h.update(pubs);
        Ok(Instances { header, n, rows, digests, outputs, public_sha256: h.finalize().into() })
    }

    pub fn digest(&self, i: usize, p: usize, np: usize) -> [u8; 32] {
        self.digests[i * np + p]
    }
}

fn sorted(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.iter().map(|(k, x)| (k.clone(), sorted(x))).collect::<std::collections::BTreeMap<_, _>>().into_iter().collect()),
        Value::Array(a) => Value::Array(a.iter().map(sorted).collect()),
        x => x.clone(),
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

/// A frame-v3 leaf: position `i` of the domain, its schema and its payload.
pub fn fv3_leaf(domain_id: &[u8], i: usize, schema: &[u8], payload: &[u8]) -> [u8; 32] {
    fv3_hash(b"leaf", &[domain_id, &uint(i), &uint(i), schema, payload])
}

pub fn fv3_root(domain_id: &[u8], leaves: Vec<[u8; 32]>) -> [u8; 32] {
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

/// The frame-v3 roots recomputed from the file's row digests and output words, against its header's; the pinned dummy digests
/// against keyed BLAKE3 of zero rows (the commitment scheme's own function; the circuit is never evaluated here); the row key
/// the x-row key.
pub fn check_public(c: &Composite, inst: &Instances) -> Result<(), String> {
    let fv = &inst.header["frame_v3"];
    let get = |k: &str, p: &str| -> Result<Vec<u8>, String> { hexb(fv[k][p].as_str().ok_or(format!("frame_v3.{k}.{p} missing"))?) };
    if hexb(fv["key"].as_str().unwrap_or(""))? != c.key || c.key != crate::chunk::row_key(b"x") {
        return Err("the row key is not the x-row key (verity/blake3-leaf/v1/x)".into());
    }
    let np = c.ports.len();
    for (p, port) in c.ports.iter().enumerate() {
        let dom = get("domain_ids", &port.name)?;
        if fv["schemas"][&port.name] != "blake3-keyed/row/v2" {
            return Err(format!("input port {}: schema is not blake3-keyed/row/v2", port.name));
        }
        let leaves = (0..inst.n).into_par_iter().map(|i| fv3_hash(b"leaf", &[&dom, &uint(i), &uint(i), b"blake3-keyed/row/v2", &inst.digest(i, p, np)])).collect();
        if fv3_root(&dom, leaves).to_vec() != get("roots", &port.name)? {
            return Err(format!("input port {}: the row digests do not hash to the header's frame-v3 root", port.name));
        }
        if *blake3::keyed_hash(&c.key, &vec![0u8; 2 * port.words]).as_bytes() != c.dummy_digests[p] {
            return Err(format!("the pinned dummy digest of port {} is not the zero row's", port.name));
        }
    }
    let n_out: usize = c.out_words.iter().sum();
    let mut at = 0;
    for (q, (name, w)) in c.meta["out_ports"].as_array().unwrap().iter().map(|x| (x[0].as_str().unwrap_or(""), x[1].as_u64().unwrap_or(0) as usize)).enumerate() {
        let _ = q;
        let dom = get("domain_ids", name)?;
        let words: Vec<u16> = (0..inst.n).flat_map(|i| inst.outputs[i * n_out + at..i * n_out + at + w].to_vec()).collect();
        let leaves = words.par_iter().enumerate().map(|(r, v)| fv3_hash(b"leaf", &[&dom, &uint(r), &uint(r), b"u16", &v.to_be_bytes()])).collect();
        if fv3_root(&dom, leaves).to_vec() != get("roots", name)? {
            return Err(format!("output port {name}: the output words do not hash to the header's frame-v3 root"));
        }
        at += w;
    }
    Ok(())
}

// --- the statement -------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Region {
    /// counter, block length and flags of every compression of comp range `r`
    Params(usize),
    /// the chaining value of every chain start of comp range `r` (the key)
    CvIn(usize),
    /// the digest of port `p`: the output value of each VU's root compression in range `r`
    Digest(usize, usize),
    /// every unit slot's returned-output words
    Out,
}

pub struct RegionShape {
    pub region: Region,
    pub free: Vec<usize>,
    pub mask: u32,
    pub fixed: u32,
}

pub struct BlockCircuit {
    pub k_log: usize,
    pub pin: usize,
    /// per slot type: its log size and its circuit (the compression circuit's CSC fold is `comp()`'s)
    pub types: Vec<(usize, Option<SparseMatrixCircuit<'static>>)>,
    /// (type index, first position, count) per range
    pub ranges: Vec<(usize, usize, usize)>,
    pub da: Vec<(u32, u32)>,
    pub db: Vec<(u32, u32)>,
}

impl LincheckCircuit for BlockCircuit {
    fn n_cols(&self) -> usize {
        1 << self.k_log
    }

    fn fold_alpha_batched(&self, alpha: F128, eq: &[F128]) -> Vec<F128> {
        assert_eq!(eq.len(), 1 << self.k_log);
        let lo = self.types.iter().map(|t| 1usize << t.0).min().unwrap();
        let c0 = eq[..lo].iter().position(|e| *e != F128::ZERO).expect("eq tensor vanishes on the first slot");
        let inv = eq[c0].inv();
        let bases: Vec<Vec<F128>> = self.types.iter().map(|(sl, circ)| match circ {
            Some(m) => m.fold_alpha_batched(alpha, &eq[..1 << sl]),
            None => comp().csc_lincheck_circuit().fold_alpha_batched(alpha, &eq[..1 << sl]),
        }).collect();
        let mut out = vec![F128::ZERO; 1 << self.k_log];
        for &(t, pos0, count) in &self.ranges {
            let sl = self.types[t].0;
            let base = &bases[t];
            out[pos0 << sl..(pos0 + count) << sl].par_chunks_mut(1 << sl).enumerate().for_each(|(k, o)| {
                let ratio = eq[((pos0 + k) << sl) | c0] * inv;
                for (x, y) in o.iter_mut().zip(base) {
                    *x = ratio * *y;
                }
            });
        }
        for &(i, c) in &self.da {
            out[c as usize] += alpha * eq[i as usize];
        }
        for &(i, c) in &self.db {
            out[c as usize] += eq[i as usize];
        }
        out
    }

    fn const_pin_col(&self) -> Option<usize> {
        Some(self.pin)
    }
}

/// Everything a session needs, the same on both sides.
pub struct Stmt {
    pub c: Composite,
    pub n: usize,
    pub blocks_real: usize,
    pub nbl: usize,
    pub m: usize,
    pub m_pts: usize,
    pub pin: usize,
    pub regions: Vec<RegionShape>,
    pub circuit: BlockCircuit,
    pub r1cs: BlockR1cs,
    pub digest: [u8; 32],
    pub key: [u32; 8],
    /// a dummy run's output value (`nb` zero blocks from the key), per port
    pub dummy_run_cv: Vec<[u32; 8]>,
    /// how the verifier's identity is pinned: profile, reps, zk, leaf scheme, coin mode (backend identity)
    pub identity: Value,
}

fn bit(w: &F128, t: usize) -> u64 {
    (if t < 64 { w.lo >> t } else { w.hi >> (t - 64) }) & 1
}

fn set_bit(w: &mut F128, t: usize) {
    if t < 64 { w.lo |= 1 << t } else { w.hi |= 1 << (t - 64) }
}

fn flip_bit(w: &mut F128, t: usize) {
    if t < 64 { w.lo ^= 1 << t } else { w.hi ^= 1 << (t - 64) }
}

fn leak<T>(x: T) -> &'static T {
    Box::leak(Box::new(x))
}

impl Stmt {
    pub fn new(c: Composite, n: usize, identity: Value) -> Result<Self, String> {
        let k_log = c.k_log;
        let blocks_real = n.div_ceil(c.g);
        let nbl = log2ceil(blocks_real.max(1)).max(3);
        let m = k_log + nbl;
        let pin = pin_of(&c);
        let (da, db) = delta(&c, pin);
        let mut types: Vec<(usize, Option<SparseMatrixCircuit<'static>>)> = Vec::new();
        let mut type_of: BTreeMap<String, usize> = BTreeMap::new();
        let mut ranges = Vec::new();
        for r in &c.ranges {
            let key = format!("{:?}", r.kind);
            let t = *type_of.entry(key).or_insert_with(|| {
                let circ = match r.kind {
                    Kind::Comp => None,
                    Kind::Net(i) => {
                        let net = &c.nets[i].1;
                        let k = 1usize << net.unit_log;
                        let pad = |rows: &Vec<Vec<usize>>| { let mut x = rows.clone(); x.resize(k, Vec::new()); SparseBinaryMatrix { num_rows: k, num_cols: k, rows: x.into() } };
                        Some(SparseMatrixCircuit::new(leak(pad(&net.a)), leak(pad(&net.b))))
                    }
                    Kind::Mask => {
                        let k = 1usize << r.slot_log;
                        let id = SparseBinaryMatrix { num_rows: k, num_cols: k, rows: (0..k).map(|i| vec![i]).collect::<Vec<_>>().into() };
                        Some(SparseMatrixCircuit::new(leak(id.clone()), leak(id)))
                    }
                };
                types.push((r.slot_log, circ));
                types.len() - 1
            });
            ranges.push((t, r.pos0, r.count));
        }
        let regions = Self::regions(&c)?;
        let key = key_words(&c.key);
        let dummy_run_cv = c.ports.iter().map(|p| {
            let mut cv = key;
            for j in 0..p.nb {
                let (ctr, bl, f) = c.params(CompRole::DummyRun { j });
                cv = blake3_compress(&cv, &[0u32; 16], ctr, bl, f)[..8].try_into().unwrap();
            }
            cv
        }).collect();
        let mut h = Sha256::new();
        h.update(TAG);
        h.update(comp().statement_digest());
        h.update(c.sha256);
        h.update(c.key);
        h.update(serde_json::to_vec(&sorted(&identity)).unwrap());
        for x in [n, blocks_real, nbl, m, k_log, pin, regions.len(), c.g] {
            h.update((x as u64).to_le_bytes());
        }
        for d in [&da, &db] {
            h.update((d.len() as u64).to_le_bytes());
            for &(i, cc) in d.iter() {
                h.update(i.to_le_bytes());
                h.update(cc.to_le_bytes());
            }
        }
        let digest: [u8; 32] = h.finalize().into();
        let kk = 1usize << k_log;
        let empty = || SparseBinaryMatrix { num_rows: kk, num_cols: kk, rows: vec![Vec::new(); kk].into() };
        let r1cs = BlockR1cs {
            m,
            k_log,
            k_skip: b3::K_SKIP,
            useful_bits: kk,
            a_0: empty(),
            b_0: empty(),
            c_0: SparseBinaryMatrix { num_rows: kk, num_cols: kk, rows: (0..kk).map(|i| vec![i]).collect::<Vec<_>>().into() },
            layout: WitnessLayout::RowMajor,
            const_pin: Some(pin),
            digest_cache: OnceLock::from(digest),
            csc_cache: OnceLock::new(),
        };
        let circuit = BlockCircuit { k_log, pin, types, ranges, da, db };
        Ok(Stmt { c, n, blocks_real, nbl, m, m_pts: PT_LOCAL + nbl, pin, regions, circuit, r1cs, digest, key, dummy_run_cv, identity })
    }

    fn regions(c: &Composite) -> Result<Vec<RegionShape>, String> {
        let kl = c.k_log;
        let bits = |lo: usize, n: usize| (lo..lo + n).collect::<Vec<usize>>();
        let mk = |region: Region, free: Vec<usize>, fixed: usize| -> Result<RegionShape, String> {
            let mask = ((1u32 << kl) - 1) & !free.iter().fold(0u32, |mm, &b| mm | (1 << b));
            if fixed as u32 & !mask != 0 || free.len() > PT_LOCAL || free.iter().any(|&b| b >= kl) {
                return Err(format!("{region:?}: not an aligned region of at most {PT_LOCAL} free bits"));
            }
            Ok(RegionShape { region, free, mask, fixed: fixed as u32 })
        };
        let mut out = Vec::new();
        for (ri, r) in c.ranges.iter().enumerate() {
            let lr = r.count.trailing_zeros() as usize;
            match &r.role {
                Role::Runs { p, .. } => {
                    let port = &c.ports[*p];
                    let lnb = port.nb.trailing_zeros() as usize;
                    out.push(mk(Region::Params(ri), [bits(0, 7), bits(SUB_LOG, lr)].concat(), r.start() + b3::T_LO_BASE)?);
                    out.push(mk(Region::CvIn(ri), [bits(0, 8), bits(SUB_LOG + lnb, lr - lnb)].concat(), r.start() + b3::CV_BASE)?);
                    if port.chunks == 1 {
                        out.push(mk(Region::Digest(*p, ri), [bits(0, 8), bits(SUB_LOG + lnb, lr - lnb)].concat(),
                                    r.start() + ((port.nb - 1) << SUB_LOG) + b3::OUT_LO_BASE)?);
                    }
                }
                Role::Parents { p } => {
                    let lc = c.ports[*p].chunks.trailing_zeros() as usize;
                    out.push(mk(Region::Params(ri), [bits(0, 7), bits(SUB_LOG, lr)].concat(), r.start() + b3::T_LO_BASE)?);
                    out.push(mk(Region::CvIn(ri), [bits(0, 8), bits(SUB_LOG, lr)].concat(), r.start() + b3::CV_BASE)?);
                    out.push(mk(Region::Digest(*p, ri), [bits(0, 8), bits(SUB_LOG + lc, lr - lc)].concat(), r.start() + b3::OUT_LO_BASE)?);
                }
                Role::None => {}
            }
        }
        let unit = c.unit();
        let ur = c.unit_range();
        let g0 = &unit.out_groups[0];
        out.push(mk(Region::Out, [bits(0, 7 + g0.words.trailing_zeros() as usize), bits(ur.slot_log, ur.count.trailing_zeros() as usize)].concat(),
                    ur.start() + g0.col * WORD)?);
        Ok(out)
    }

    pub fn blocks(&self) -> usize {
        1 << self.nbl
    }

    fn inst_of(&self, b: usize, g: usize) -> Option<usize> {
        let i = b * self.c.g + g;
        (i < self.n).then_some(i)
    }

    /// 16 message words of block `j` of chunk `c` of port `p`'s row of instance `i` (zeros for a padding instance).
    fn message(&self, rows: Option<&[u8]>, i: Option<usize>, p: usize, c: usize, j: usize) -> [u32; 16] {
        let (Some(rows), Some(i)) = (rows, i) else { return [0; 16] };
        let off: usize = self.c.ports[..p].iter().map(|q| 2 * q.words).sum();
        let at = i * self.c.row_bytes() + off + 1024 * c + 64 * j;
        std::array::from_fn(|w| u32::from_le_bytes(rows[at + 4 * w..at + 4 * w + 4].try_into().unwrap()))
    }

    /// Every compression of block `b` (cv, message, counter, block length, flags) in comp-range order.
    pub fn compressions(&self, rows: Option<&[u8]>, b: usize) -> Vec<b3::Compression> {
        let c = &self.c;
        let mut out: BTreeMap<usize, b3::Compression> = BTreeMap::new();
        let mut cv_at: BTreeMap<usize, [u32; 8]> = BTreeMap::new();
        let run = |role: CompRole, cv: [u32; 8], msg: [u32; 16]| -> (b3::Compression, [u32; 8]) {
            let (ctr, bl, f) = c.params(role);
            let o: [u32; 8] = blake3_compress(&cv, &msg, ctr, bl, f)[..8].try_into().unwrap();
            ((cv, msg, ctr, bl, f), o)
        };
        for r in c.ranges.iter().filter(|r| matches!(r.role, Role::Runs { .. })) {
            let mut cv = self.key;
            for s in 0..r.count {
                let role = c.comp_role(r, s);
                let (msg, j) = match role {
                    CompRole::Run { g, p, c: ch, j } => (self.message(rows, self.inst_of(b, g), p, ch, j), j),
                    CompRole::DummyRun { j } => ([0; 16], j),
                    _ => unreachable!(),
                };
                if j == 0 {
                    cv = self.key;
                }
                let (x, o) = run(role, cv, msg);
                out.insert(r.slot(s), x);
                cv_at.insert(r.slot(s), o);
                cv = o;
            }
        }
        for r in c.ranges.iter().filter(|r| matches!(r.role, Role::Parents { .. })) {
            let Role::Parents { p } = r.role else { unreachable!() };
            let ch = c.ports[p].chunks;
            for s in (0..r.count).rev() {
                let role = c.comp_role(r, s);
                let msg = match role {
                    CompRole::Parent { g, k, .. } => {
                        let kid = |h: usize| if h < ch - 1 { cv_at[&r.slot(g * ch + h)] } else { cv_at[&c.chunk_end(g, p, h - (ch - 1))] };
                        let (l, rr) = (kid(2 * k + 1), kid(2 * k + 2));
                        std::array::from_fn(|w| if w < 8 { l[w] } else { rr[w - 8] })
                    }
                    _ => [0; 16],
                };
                let (x, o) = run(role, self.key, msg);
                out.insert(r.slot(s), x);
                cv_at.insert(r.slot(s), o);
            }
        }
        out.into_values().collect()
    }

    /// Unit slot `k` of block `b`'s input words (its leaf ports, from the rows; zeros for a padding instance or a zero leaf).
    pub fn unit_inputs(&self, rows: Option<&[u8]>, b: usize, k: usize) -> Vec<F128> {
        let c = &self.c;
        let unit = c.unit();
        let (g, u) = (k / c.upv, k % c.upv);
        let mut w = vec![F128::ZERO; unit.in_words];
        let (Some(rows), Some(i)) = (rows, self.inst_of(b, g)) else { return w };
        let rb = c.row_bytes();
        let mut col = unit.in_groups[0].col * WORD;
        for (j, &leaf) in c.leaves_in[u].iter().enumerate() {
            if leaf >= 0 {
                let at = i * rb + 2 * leaf as usize;
                let v = u16::from_le_bytes([rows[at], rows[at + 1]]) as u64;
                if col % WORD < 64 { w[col / WORD].lo |= v << (col % WORD) } else { w[col / WORD].hi |= v << (col % WORD - 64) }
            }
            col += unit.in_groups[0].bits[j];
        }
        w
    }

    /// Every slot's output words (`net.out_cols`) for input words `ins` (a lookup: its table word; else the rows, 64 at once).
    fn eval_outputs(&self, n: usize, net: &IrUnitNet, ins: &[Vec<F128>]) -> Vec<Vec<F128>> {
        if let Some(lk) = self.c.lookups.get(&n) {
            return ins.par_iter().map(|w| {
                let idx = (w[0].lo as usize) & ((1 << lk.n) - 1);
                vec![F128::new((lk.table[idx] & 0x7FFF_FFFF) as u64, 0)]
            }).collect();
        }
        let o0 = net.out_cols[0] * WORD;
        let nw = net.out_cols.len();
        ins.par_chunks(64).flat_map_iter(|grp| {
            let (zz, _, _) = net.eval64(|col| grp.iter().enumerate().fold(0u64, |x, (l, r)| x | (bit(&r[col / WORD], col % WORD) << l)));
            (0..grp.len()).map(move |l| {
                let mut o = vec![F128::ZERO; nw];
                for (r, y) in zz[o0..o0 + nw * WORD].iter().enumerate() {
                    if (y >> l) & 1 == 1 {
                        set_bit(&mut o[r / WORD], r % WORD);
                    }
                }
                o
            }).collect::<Vec<_>>()
        }).collect()
    }

    /// Every slot's input words, converged: the units' leaf words from `unit_inputs(b, k)`, then every wire copied from the
    /// slot that computes it, re-evaluating each slot type whose inputs changed until none do (the tail and the lookups
    /// between the units' two passes). `nets` overrides a circuit's evaluation (a tampered witness).
    pub fn converge(&self, unit_inputs: &(dyn Fn(usize, usize) -> Vec<F128> + Sync), nets: &BTreeMap<usize, &IrUnitNet>) -> Vec<Vec<Vec<F128>>> {
        self.converge_with(unit_inputs, nets, None)
    }

    /// [`Stmt::converge`] with the units' output words given (`unit_outs`, the device's first pass): the units are never
    /// evaluated here, only the tail stages and the lookups.
    pub fn converge_with(&self, unit_inputs: &(dyn Fn(usize, usize) -> Vec<F128> + Sync), nets: &BTreeMap<usize, &IrUnitNet>,
                         unit_outs: Option<Vec<Vec<F128>>>) -> Vec<Vec<Vec<F128>>> {
        let c = &self.c;
        let nb = self.blocks();
        let net_of = |n: usize| -> &IrUnitNet { nets.get(&n).copied().unwrap_or(&c.nets[n].1) };
        let count = |n: usize| c.range_of(n).count;
        let mut ins: Vec<Vec<Vec<F128>>> = (0..c.nets.len()).map(|n| {
            if n == c.unit_net {
                (0..nb * count(n)).into_par_iter().map(|x| unit_inputs(x / count(n), x % count(n))).collect()
            } else {
                vec![vec![F128::ZERO; net_of(n).in_words]; nb * count(n)]
            }
        }).collect();
        if c.wires.is_empty() {
            return ins;
        }
        let mut outs: Vec<Vec<Vec<F128>>> = vec![Vec::new(); c.nets.len()];
        let mut dirty = vec![true; c.nets.len()];
        let fixed = unit_outs.is_some();
        if let Some(u) = unit_outs {
            outs[c.unit_net] = u;
            dirty[c.unit_net] = false;
        }
        for _round in 0..64 {
            for n in 0..c.nets.len() {
                if fixed && n == c.unit_net {
                    dirty[n] = false;
                    continue;
                }
                if dirty[n] {
                    outs[n] = self.eval_outputs(n, net_of(n), &ins[n]);
                    dirty[n] = false;
                }
            }
            let mut changed = false;
            for w in &c.wires {
                let (dn, sn) = (w.dst.0, w.src.0);
                let (dcol, width) = port_cols(&net_of(dn).in_groups)[w.dst.2];
                let scol = port_cols(&net_of(sn).out_groups)[w.src.2].0 - net_of(sn).out_cols[0] * WORD;
                for b in 0..nb {
                    for g in 0..c.g {
                        let xs = b * count(sn) + g * c.per_vu[sn] + w.src.1;
                        let xd = b * count(dn) + g * c.per_vu[dn] + w.dst.1;
                        for t in 0..width {
                            let v = bit(&outs[sn][xs][(scol + t) / WORD], (scol + t) % WORD);
                            let d = &mut ins[dn][xd][(dcol + t) / WORD];
                            if bit(d, (dcol + t) % WORD) != v {
                                flip_bit(d, (dcol + t) % WORD);
                                changed = true;
                                dirty[dn] = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                return ins;
            }
        }
        panic!("the slots' wires did not converge in 64 rounds (a cycle through a slot's own output?)")
    }

    /// `(z, a, b, z_lincheck)`: every block's compressions (`comps(b)`), every slot of every circuit from its input words `ins`
    /// (`converge`; a lookup by its fast witness), the mask slot from `mask`. `nets` overrides a circuit's evaluation.
    #[allow(clippy::type_complexity)]
    pub fn witness(&self, comps: &(dyn Fn(usize) -> Vec<b3::Compression> + Sync), ins: &[Vec<Vec<F128>>],
                   mask: &(dyn Fn(usize) -> Vec<F128> + Sync), nets: &BTreeMap<usize, &IrUnitNet>) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let c = &self.c;
        let kl = c.k_log;
        let n = 1usize << (self.m - 7);
        let wpb = 1usize << (kl - 7);
        let (mut z, mut a, mut bb) = (vec![F128::ZERO; n], vec![F128::ZERO; n], vec![F128::ZERO; n]);
        let comp_slots: Vec<usize> = c.ranges.iter().filter(|r| r.kind == Kind::Comp).flat_map(|r| (0..r.count).map(|s| r.slot(s))).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let cw = 1usize << (SUB_LOG - 7);
        z.par_chunks_mut(wpb).zip(a.par_chunks_mut(wpb)).zip(bb.par_chunks_mut(wpb)).enumerate().for_each(|(blk, ((zb, ab), bbb))| {
            let mut cs = comps(blk);
            let nc = cs.len();
            cs.resize(nc.next_power_of_two(), cs[0]);
            let (cz, ca, cb) = b3::generate_witness_with_ab_packed(&cs, log2ceil(cs.len()));
            for (q, &col) in comp_slots.iter().enumerate() {
                let w = col / WORD;
                zb[w..w + cw].copy_from_slice(&cz[q * cw..(q + 1) * cw]);
                ab[w..w + cw].copy_from_slice(&ca[q * cw..(q + 1) * cw]);
                bbb[w..w + cw].copy_from_slice(&cb[q * cw..(q + 1) * cw]);
            }
            for r in c.ranges.iter().filter(|r| r.kind == Kind::Mask) {
                let mw = mask(blk);
                let w = r.start() / WORD;
                zb[w..w + mw.len()].copy_from_slice(&mw);
                ab[w..w + mw.len()].copy_from_slice(&mw);
                bbb[w..w + mw.len()].copy_from_slice(&mw);
            }
        });
        for (ni, (_, pinned)) in c.nets.iter().enumerate() {
            let r = c.range_of(ni);
            let uw = 1usize << (pinned.unit_log - 7);
            for (x, uz, ua, ub) in self.slot_zab(ni, nets.get(&ni).copied().unwrap_or(pinned), &ins[ni]) {
                let (blk, k) = (x / r.count, x % r.count);
                let at = blk * wpb + r.slot(k) / WORD;
                z[at..at + uw].copy_from_slice(&uz);
                a[at..at + uw].copy_from_slice(&ua);
                bb[at..at + uw].copy_from_slice(&ub);
            }
        }
        let zl = pack_z_lincheck_from_packed(&z, self.m, kl);
        (z, a, bb, zl)
    }

    /// Every slot of circuit `ni`'s `(instance, z, a, b)` words from its input words (a lookup: its fast witness).
    #[allow(clippy::type_complexity)]
    pub fn slot_zab(&self, ni: usize, net: &IrUnitNet, ins: &[Vec<F128>]) -> Vec<(usize, Vec<F128>, Vec<F128>, Vec<F128>)> {
        let c = &self.c;
        let uw = 1usize << (c.nets[ni].1.unit_log - 7);
        {
            let items: Vec<(usize, Vec<F128>, Vec<F128>, Vec<F128>)> = if let Some(lk) = c.lookups.get(&ni) {
                ins.par_iter().enumerate().map(|(x, w)| {
                    let (zz, aa, bq) = lk.witness((w[0].lo as usize) & ((1 << lk.n) - 1));
                    let f = |v: Vec<u64>| v.chunks(2).map(|p| F128::new(p[0], p[1])).collect::<Vec<F128>>();
                    (x, f(zz), f(aa), f(bq))
                }).collect()
            } else {
                ins.par_chunks(64).enumerate().flat_map_iter(|(gi, grp)| {
                    let (zz, aa, bq) = net.eval64(|col| grp.iter().enumerate().fold(0u64, |x, (l, r)| x | (bit(&r[col / WORD], col % WORD) << l)));
                    (0..grp.len()).map(move |l| {
                        let pack = |v: &Vec<u64>| -> Vec<F128> {
                            let mut o = vec![F128::ZERO; uw];
                            for (row, &y) in v.iter().enumerate() {
                                if (y >> l) & 1 == 1 {
                                    set_bit(&mut o[row / WORD], row % WORD);
                                }
                            }
                            o
                        };
                        (64 * gi + l, pack(&zz), pack(&aa), pack(&bq))
                    }).collect::<Vec<_>>()
                }).collect()
            };
            items
        }
    }

    /// The device's first pass handed back (`cut`: per group of 32 units and per cut output row of the unit, the row's bits,
    /// unit l of the group in bit l): the tail stages and lookups converged on the host; returns every unit's late input rows
    /// in the same layout (`late_rows` order) and every host slot's `(absolute F128 word offset, z, a, b)`.
    #[allow(clippy::type_complexity)]
    pub fn host_tail(&self, unit_inputs: &(dyn Fn(usize, usize) -> Vec<F128> + Sync), cut: &[u32], cut_rows: &[usize], late_rows: &[usize])
        -> (Vec<u32>, Vec<(usize, Vec<F128>, Vec<F128>, Vec<F128>)>) {
        let c = &self.c;
        let unit = c.unit();
        let n_units = self.blocks() * c.unit_range().count;
        let groups = n_units.div_ceil(32);
        let o0 = unit.out_cols[0] * WORD;
        let outs: Vec<Vec<F128>> = (0..n_units).map(|u| {
            let mut o = vec![F128::ZERO; unit.out_cols.len()];
            let (g, l) = (u / 32, u % 32);
            for (i, &r) in cut_rows.iter().enumerate() {
                if (cut[g * cut_rows.len() + i] >> l) & 1 == 1 {
                    set_bit(&mut o[(r - o0) / WORD], (r - o0) % WORD);
                }
            }
            o
        }).collect();
        let ins = self.converge_with(unit_inputs, &BTreeMap::new(), Some(outs));
        let mut late = vec![0u32; groups * late_rows.len()];
        for (u, w) in ins[c.unit_net].iter().enumerate() {
            for (k, &r) in late_rows.iter().enumerate() {
                if bit(&w[r / WORD], r % WORD) == 1 {
                    late[(u / 32) * late_rows.len() + k] |= 1 << (u % 32);
                }
            }
        }
        let wpb = 1usize << (c.k_log - 7);
        let mut host = Vec::new();
        for ni in 0..c.nets.len() {
            if ni == c.unit_net {
                continue;
            }
            let r = c.range_of(ni);
            for (x, z, a, b) in self.slot_zab(ni, &c.nets[ni].1, &ins[ni]) {
                host.push(((x / r.count) * wpb + r.slot(x % r.count) / WORD, z, a, b));
            }
        }
        (late, host)
    }

    /// Block `b`'s bytes of a region, in free-assignment order.
    fn region_bytes(&self, reg: &Region, inst: &Instances, b: usize) -> Vec<u8> {
        let c = &self.c;
        let np = c.ports.len();
        let cv_bytes = |cv: &[u32; 8]| cv.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
        match reg {
            Region::Params(ri) => {
                let r = &c.ranges[*ri];
                (0..r.count).flat_map(|s| {
                    let (ctr, bl, f) = c.params(c.comp_role(r, s));
                    [ctr.to_le_bytes().to_vec(), bl.to_le_bytes().to_vec(), f.to_le_bytes().to_vec()].concat()
                }).collect()
            }
            Region::CvIn(ri) => {
                let r = &c.ranges[*ri];
                let starts = match &r.role { Role::Runs { slots, .. } => slots.len(), _ => r.count };
                (0..starts).flat_map(|_| cv_bytes(&self.key)).collect()
            }
            Region::Digest(p, ri) => {
                let r = &c.ranges[*ri];
                let dig = |g: usize| -> Vec<u8> { match self.inst_of(b, g) { Some(i) => inst.digest(i, *p, np).to_vec(), None => c.dummy_digests[*p].to_vec() } };
                match &r.role {
                    Role::Runs { slots, .. } => slots.iter().flat_map(|x| match x { Some((g, _)) => dig(*g), None => cv_bytes(&self.dummy_run_cv[*p]) }).collect(),
                    Role::Parents { .. } => (0..c.g).flat_map(dig).collect(),
                    Role::None => unreachable!(),
                }
            }
            Region::Out => {
                let unit = c.unit();
                let ur = c.unit_range();
                let (g0, n_out) = (&unit.out_groups[0], c.out_words.iter().sum::<usize>());
                (0..ur.count).flat_map(|k| {
                    let (g, u) = (k / c.upv, k % c.upv);
                    let mut w = vec![F128::ZERO; g0.words];
                    let mut col = 0;
                    for (j, &o) in c.leaves_out[u].iter().enumerate() {
                        let v = match self.inst_of(b, g) { Some(i) => inst.outputs[i * n_out + o], None => c.dummy_outputs[o] } as u64;
                        if col % WORD < 64 { w[col / WORD].lo |= v << (col % WORD) } else { w[col / WORD].hi |= v << (col % WORD - 64) }
                        col += g0.bits[j];
                    }
                    w.into_iter().flat_map(|x| [x.lo.to_le_bytes(), x.hi.to_le_bytes()].concat())
                }).collect()
            }
        }
    }

    pub fn region_point(&self, reg: &RegionShape, r: &[F128]) -> Vec<F128> {
        let kl = self.c.k_log;
        let mut p = vec![F128::ZERO; self.m];
        for (j, x) in p.iter_mut().enumerate().take(kl) {
            if (reg.mask >> j) & 1 == 1 && (reg.fixed >> j) & 1 == 1 {
                *x = F128::ONE;
            }
        }
        for (t, &j) in reg.free.iter().enumerate() {
            p[j] = r[t];
        }
        p[kl..].copy_from_slice(&r[PT_LOCAL..PT_LOCAL + self.nbl]);
        p
    }

    pub fn region_value(&self, reg: &RegionShape, r: &[F128], inst: &Instances) -> F128 {
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
            let bytes = self.region_bytes(&reg.region, inst, b);
            debug_assert_eq!(bytes.len(), nbytes, "{:?}", reg.region);
            eb[b] * bytes.iter().enumerate().fold(F128::ZERO, |a, (i, &x)| a + t[i][x as usize])
        }).reduce(|| F128::ZERO, |a, b| a + b)
    }

    pub fn extra_points(&self, pts: &[Vec<F128>]) -> Vec<Vec<F128>> {
        self.regions.iter().flat_map(|reg| pts.iter().map(move |r| self.region_point(reg, r))).collect()
    }

    pub fn extra_claims(&self, pts: &[Vec<F128>], inst: &Instances) -> Vec<ExtraClaim> {
        let values: Vec<F128> = self.regions.iter().flat_map(|reg| pts.iter().map(move |r| self.region_value(reg, r, inst))).collect();
        self.extra_points(pts).into_iter().zip(values)
            .map(|(pt, value)| ExtraClaim { value, skip_weights: eq_skip_weights(&pt[..6]), x_full: pt[6..].to_vec() })
            .collect()
    }

    /// Words per block the device witness needs for the unit slots: every unit's input words, block-major.
    pub fn region_bytes_len(&self, reg: &RegionShape) -> usize {
        (1usize << reg.free.len()) / 8
    }
}
