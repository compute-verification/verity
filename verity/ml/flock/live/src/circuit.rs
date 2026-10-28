//! `verity/flock-circuit` (lane flock-circuit): any lowered IR template, its serving rows committed in the circuit, and
//! nothing public but the frame-v3-sha512 roots, each row's hm96 commit string `b || c` and the pinned layout. The rows are
//! the input rows, then one output row per output port (its words zero-padded to whole SHA-512 blocks): an output is a
//! hidden value like an input. `verity_flock.circuit` composes the pinned circuit (`flock-circuit`) and stages the instance
//! files.
//!
//! A row is an `hm96-sha512/row/v1` leaf: `x = SHA-512(sha512_row_prefix || row)`, `b = x XOR M(key) y`, `c = SHA-512(salt_prefix
//! || y)` for its 192-byte salt `y`; the frame leaf binds `hm96.SHA512.tree_leaf(key, b || c)`. Or (META `row_schemas`, per port)
//! an `hm96-sha512/row/v2` leaf, a bit row of `l` bits: `x = SHA-512(P(l) || ceil(l / 8) bytes)`, the row's bits least
//! significant first, its u16 leaves the same bits (an output row's `l` its words' bits). A block holds whole instances (VUs)
//! in aligned ranges of slots: `sha512x3` (three SHA-512 compressions, 2^18 bits: a row's message blocks after its constant
//! first block but the last), `hm96` (one per row: the last block's compression, then `b || c` from the salt), the template's
//! units (2^unit_log bits), the tail's slots, and a reserved mask slot of free cells. `A = I ⊗ A_0 + Δ` (and B), where A_0
//! places each slot type's circuit on its ranges and Δ, derived here from the pinned circuit alone, makes inside the block:
//! - each row's first compression (its hm96 slot, a row of no other compression) read the prefix's midstate, and every
//!   message bit at or past the row's length (a bit row's fill bits, then SHA-512's padding) its constant;
//! - every later compression read the previous one's output, and the hm96 slot the last one's (the row's wires);
//! - every unit's leaf input bits copies of the message bits that hash them, and a zero leaf (or a bit row's leaf bit past
//!   its length) a copy of a forced-zero row;
//! - every output row's message bits copies of the returned output bits, and its padding words copies of a forced-zero row;
//! - every other wire of the template's slots, and every slot's constant a copy of the pinned column.
//!
//! Public regions, each opened at the verifier's points (drawn after the commitment, both reps), values all the verifier's:
//! every row's `b || c` (its hm96 slot's output). The rows (outputs too), their salts, the chaining values and every word
//! between slots stay private. A padding instance is the all-zero instance with zero salts, whose rows' `b || c` are pinned and
//! checked here natively (a zero salt's mask is zero): its input rows are zero, its output rows its outputs (forced by its
//! circuit, pinned in META). A shared-row file (header `shared_rows`) commits each input port's rows once, as one table, and
//! names the row each instance reads (its refs): the instance's digest values are those rows' `b || c`, so the rows it hashes
//! in the circuit are the committed ones. Its output rows are one per instance (refs `0 … n − 1`): equal outputs stay apart.
//!
//! The verifier evaluates no part of the computation (standing rule): it checks the proof, the frame-v3-sha512 roots against
//! the public `b || c`, and nothing else. A circuit carrying a verifier-evaluated tail (a `CUT` line, or a tail,
//! public cut words or native words in META) is refused at parse; a template's tail is a slot type of the circuit or the
//! template has no statement here.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use flock_core::{
    lincheck::{CscCircuit, LincheckCircuit, pack_z_lincheck_from_packed},
    proof::eq_skip_weights,
    r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout},
    verifier::ExtraClaim,
};
use flock_field::F128;
use flock_prover::r1cs_hashes::blake3 as b3;
use rayon::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha512};

use crate::{chunk::log2ceil, ir_block::IrUnitNet, lookup::LookupNet};

pub const FORMAT: &str = "flock-circuit";
pub const INST_FORMAT: &str = "flock-circuit-inputs";
pub const STATEMENT: &str = "verity/flock-circuit";

/// The strings a circuit statement hashes byte for byte, by its id (PROTOCOL.md §16.5's table, the Lean `Flock.Tags`): the
/// id is also the statement digest's tag, and each id has its own Σ tag and stream domain.
pub struct Tags {
    pub statement: &'static str,
    pub sigma: &'static str,
    /// a rep's domain is this, then the rep's decimal index
    pub domain: &'static str,
}

pub const TAGS: Tags = Tags { statement: STATEMENT, sigma: "verity/flock-circuit/sigma", domain: "flock-circuit/fast100x2/rep" };
pub const TYPED_TAGS: Tags =
    Tags { statement: crate::typed::STATEMENT, sigma: "verity/flock-circuit/types/sigma", domain: "flock-circuit-types/fast100x2/rep" };
/// A templated statement (the Lean `Tags.circuitTmpl`): META `template` and `layout` in place of the unit's rows,
/// which the prover reads from `flock-tmpl-unit`'s file, the unit committed at the layout's positions.
pub const TMPL_FORMAT: &str = "flock-circuit/tmpl";
pub const TMPL_STATEMENT: &str = "verity/flock-circuit/tmpl";
pub const TMPL_TAGS: Tags =
    Tags { statement: TMPL_STATEMENT, sigma: "verity/flock-circuit/tmpl/sigma", domain: "flock-circuit-tmpl/fast100x2/rep" };
pub const SCHEME: &str = "frame-v3-sha512/hm96-sha512";
pub const ROW_SCHEMA: &str = "hm96-sha512/row/v1";
/// a bit row's leaf (`sha512/row/v2`; `verity.ml.commitments.rowleaf.SCHEMA_HM96_SHA512_ROW_V2`)
pub const ROW_SCHEMA_V2: &str = "hm96-sha512/row/v2";
/// a segmented bit row's leaf (`sha512/row-seg/v1`; `verity.ml.commitments.rowleaf.SCHEMA_HM96_SHA512_ROW_SEG`)
pub const ROW_SCHEMA_SEG: &str = "hm96-sha512/row-seg/v1";
/// a statement's identity `row_leaf_in_circuit` when a port is a segmented row
pub const ROW_LEAF_SEG_IDENTITY: &str = "hm96-sha512/row/v2 over sha512/row/v2, hm96-sha512/row-seg/v1 over sha512/row-seg/v1, per port";
pub const SHA_NET: &str = "sha512x3";
pub const HM_NET: &str = "hm96";
/// compressions per `sha512x3` slot
pub const PER_SLOT: usize = 3;
/// META's bindings (what the statement binds about the circuit): a class leaves them out.
pub const BINDING_KEYS: [&str; 3] = ["partition", "program_sha512", "program_digests"];
/// `Stmt.InRange` as the pinned table-soundness theorems prove it (`FlockSoundness/Accounting/Fast100.lean`): `k_log <= 27`,
/// at most 1,024 public regions, at most 64 link-point coordinates; with PROTOCOL.md §16.5's `m <= 35`. A statement outside it
/// is refused, so an accepted statement is one the pins cover.
pub const IN_RANGE_K_LOG: usize = 27;
pub const IN_RANGE_REGIONS: usize = 1024;
pub const IN_RANGE_M_PTS: usize = 64;
pub const M_MAX: usize = 35;

/// A range of a `2^k_log`-bit block, checked before any shift by its `slot_log`: its slots at least one 128-bit word and at
/// most the block, and all of them inside the block (`pos0 + count <= 2^(k_log - slot_log)`, as the Lean check's unbounded
/// arithmetic requires).
pub fn range_in_block(k_log: usize, r: &Range) -> Result<(), String> {
    if !(7..=k_log).contains(&r.slot_log) {
        return Err(format!("slot_log {} is not in [7, k_log = {k_log}]", r.slot_log));
    }
    let slots = 1usize << (k_log - r.slot_log);
    if r.pos0 > slots || r.count > slots - r.pos0 {
        return Err(format!("slots [{}, {} + {}) are not inside the block's {slots}", r.pos0, r.pos0, r.count));
    }
    Ok(())
}

/// A circuit file's block, before its regions are built from it: the block size (`Stmt.InRange`), the hm96 slots' size
/// (at least one 128-bit word, at most the block) and the pinned constant's column inside the block.
pub fn block_in_range(k_log: usize, hm_slot_log: usize, pin: usize) -> Result<(), String> {
    if k_log > IN_RANGE_K_LOG {
        return Err(format!("k_log {k_log}: outside the proved range (Stmt.InRange: k_log <= {IN_RANGE_K_LOG})"));
    }
    if !(7..=k_log).contains(&hm_slot_log) {
        return Err(format!("the hm96 range's slot_log {hm_slot_log} is not in [7, k_log = {k_log}]"));
    }
    if pin >= 1 << k_log {
        return Err(format!("the pinned constant's column {pin} is outside the block (2^{k_log} bits)"));
    }
    Ok(())
}

/// A circuit file's regions, as the pinned table-soundness theorems assume them: at most 1,024 (`Stmt.InRange`), each with
/// distinct free bits (`LinkLayout`: a region is a subcube, one coordinate per free bit).
pub fn regions_in_range(regions: &[RegionShape]) -> Result<(), String> {
    if regions.len() > IN_RANGE_REGIONS {
        return Err(format!("{} public regions: outside the proved range (Stmt.InRange: at most {IN_RANGE_REGIONS})", regions.len()));
    }
    for r in regions {
        let mut free = r.free.clone();
        free.sort_unstable();
        if free.windows(2).any(|p| p[0] == p[1]) {
            return Err(format!("{:?}: its free bits are not distinct (LinkLayout: one coordinate per free bit)", r.region));
        }
    }
    Ok(())
}

/// The instance count's part: the dense size `m = k_log + nbl` (PROTOCOL.md §16.5) and the link points' coordinates.
pub fn shape_in_range(m: usize, m_pts: usize) -> Result<(), String> {
    if m > M_MAX {
        return Err(format!("m = {m}: outside the spec's shape (m <= {M_MAX})"));
    }
    if m_pts > IN_RANGE_M_PTS {
        return Err(format!("{m_pts} link-point coordinates: outside the proved range (Stmt.InRange: at most {IN_RANGE_M_PTS})"));
    }
    Ok(())
}

pub const PARTITION_V1: &str = "verity/partition/v1";
pub const PARTITION_PROVISIONAL: &str = "flock-circuit/unit-cover/v0";
pub const SALT_BYTES: usize = 192;
pub const BC_BYTES: usize = 128;
const SHA512_ROW_TAG: &[u8] = b"verity/sha512-row/v1\0";
const SHA512_ROW_V2_TAG: &[u8] = b"verity/sha512-row/v2\0";
const SHA512_ROW_SEG_TAG: &[u8] = b"verity/sha512-row-seg/v1\0";
const SHA512_ROW_SEG_SEGMENT_TAG: &[u8] = b"verity/sha512-row-seg/v1/segment\0";
/// a `sha512/row-seg/v1` segment, in bits (1 KiB; the last may be shorter)
pub const ROW_SEG_BITS: usize = 8192;
/// a statement's `outputs` identity with runs of output ports (META `out_row_ports`), the Lean verifier's `Tags.outRowsIdentity`
pub const OUT_ROWS_SEG_IDENTITY: &str = "hidden: each output port an hm96-sha512/row/v2 row, or each run of output ports in META out_row_ports one hm96-sha512/row-seg/v1 row of their words, hashed in the circuit; the public file has no output word";
/// the same with a run committed as a `hm96-sha512/row/v2` row, the Lean verifier's `Tags.outRowsV2Identity`
pub const OUT_ROWS_V2_IDENTITY: &str = "hidden: each output port an hm96-sha512/row/v2 row, or each run of output ports in META out_row_ports one hm96-sha512/row/v2 or hm96-sha512/row-seg/v1 row of their words (its row_schemas entry), hashed in the circuit; the public file has no output word";
/// a statement's `outputs` identity with pinned outputs (META `pinned_outputs`), the Lean verifier's `Tags.pinnedIdentity`
pub const PINNED_IDENTITY: &str = "hidden: each output port an hm96-sha512/row/v2 row (each run of output ports in META out_row_ports one bit row of their words), hashed in the circuit, but the last output ports, META pinned_outputs, which have no row: each of their bits a copy of a forced-zero row; the public file has no output word";
/// what `Composite::parse` refuses of META `pinned_outputs`, word for word the Lean verifier's (`HmRow.pinnedOf`)
pub const PINNED_LINEAR: &str = "META: a pinned output's bit is not a linear row (its B the constant alone), which its Δ copy of the forced-zero row pins to zero";
pub const PINNED_REFUSED: &str = "META: pinned_outputs is not the last output ports' names, each once, at least one, of an untyped statement of bit rows (row_schemas) whose padding instance's pinned outputs are zero";
/// what `Composite::check` refuses of the output rows, word for word the Lean verifier's (`HmOut.checkRows`)
pub const OUT_ROWS_REFUSED: &str = "the rows are not the input rows, then one output row per output port (its name, its words zero-padded to whole SHA-512 blocks, all their bits) or per run of output ports (META out_row_ports), a bit row of exactly their words (sha512/row/v2 or sha512/row-seg/v1) named after the first, each name once";
const ROLE_X: u8 = 1;
/// a port's compressions, at most: each reads 1024 message columns of a block of at most 2^IN_RANGE_K_LOG
const MAX_COMPS: usize = 1 << (IN_RANGE_K_LOG - 10);
/// Free in-block coordinates per verifier point (`r[..PT_LOCAL]`); block coordinates follow.
pub const PT_LOCAL: usize = 24;

/// Words as aligned power-of-two blocks `(w, e)` (`w` a multiple of `2^e`), ascending: each maximal run of consecutive words
/// cut from its start, every block the largest aligned one that fits in what is left of the run.
pub fn aligned_blocks(words: &BTreeSet<usize>) -> Vec<(usize, usize)> {
    let (mut out, mut it) = (Vec::new(), words.iter().copied().peekable());
    while let Some(mut a) = it.next() {
        let mut b = a + 1;
        while it.next_if_eq(&b).is_some() {
            b += 1;
        }
        while a < b {
            let mut e = (usize::BITS - 1 - (b - a).leading_zeros()) as usize;
            while a % (1 << e) != 0 {
                e -= 1;
            }
            out.push((a, e));
            a += 1 << e;
        }
    }
    out
}

/// The leaf-commitment scheme this prover and verifier implement for the proof's own Merkle trees, as the composite circuit
/// pins it (`verity_flock.circuit.LEAF_SCHEME`: the current id and the `hm96-sha256/v1` room it reserves). `salt_source` names
/// `zk_hooks::LeafSalts`'s OS reads, as the Lean verifier pins the object (`Flock/Tags.lean`'s `leafSchemeHm96Os`).
pub fn leaf_scheme() -> Value {
    let (id, salt) = if cfg!(feature = "sha512") { ("hm96-sha512/v1", 192) } else { ("flock-leaf/sha256-unsalted", 0) };
    json!({"id": id, "inner": "sha512 column digest", "salt_bytes": salt,
           "salt_source": "os bytes (getrandom), 192 per leaf, each tree's read once, never reused or stored",
           "key": "default_key, a statement constant for every tree",
           "key_sha512": crate::hex(&Sha512::digest(hm96_sha512_default_key()))})
}

/// The leaves of a statement whose trees need no hiding, a recursive session's inner statement (`verity_flock.rec_live`): its
/// firewall commits every tree top under a salt of its own and the openings stay in V*'s witness, so a leaf is the column's
/// SHA-512 digest (`HashKind::Sha512`) and an opening carries no salt. ZK off only (`verity_flock.circuit.LEAF_SCHEME_PLAIN`).
pub fn leaf_scheme_plain() -> Value {
    json!({"id": "flock-leaf/sha512-unsalted", "inner": "sha512 column digest", "salt_bytes": 0,
           "use": "a recursive session's inner statement: the gateway commits every tree top under a salt, ZK off"})
}

/// `hm96-sha512/v1`'s pinned key: `SHA-512("verity/hm96-sha512/key/v1\0" || u32be(i))` for i = 0..3, bit 2047 cleared.
pub fn hm96_sha512_default_key() -> [u8; 256] {
    let mut k = [0u8; 256];
    for (i, c) in k.chunks_mut(64).enumerate() {
        let mut h = Sha512::new();
        h.update(b"verity/hm96-sha512/key/v1\0");
        h.update((i as u32).to_be_bytes());
        c.copy_from_slice(&h.finalize());
    }
    k[255] &= 0x7f;
    k
}
const WORD: usize = 128;
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
    /// index into `Composite::nets`
    Net(usize),
    Mask,
}

#[derive(Clone, Debug)]
pub struct Range {
    pub kind: Kind,
    pub slot_log: usize,
    pub pos0: usize,
    pub count: usize,
    /// aligned to its slot only, spanning its count (no region reads it); else aligned to its power-of-two span
    pub packed: bool,
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
    /// compressions after the constant first block (the last block, the row's last bytes and SHA-512's padding, is the hm96
    /// slot's)
    pub comps: usize,
    /// the row's length in bits: `16 * words` (`sha512/row/v1`), or a bit row's own (`v2`)
    pub bits: usize,
    /// a bit row (`sha512/row/v2`, its leaf `hm96-sha512/row/v2`)
    pub v2: bool,
    /// a segmented bit row (`sha512/row-seg/v1`, its leaf `hm96-sha512/row-seg/v1`): each segment's bits and compressions
    /// (`seg_shape`); `None` for any other row
    pub segs: Option<Vec<(usize, usize)>>,
}

impl Port {
    /// The row bytes its digest hashes after the prefix: `2 * words`, a bit row's `ceil(bits / 8)`.
    pub fn bytes(&self) -> usize {
        if self.v2 || self.segs.is_some() { self.bits.div_ceil(8) } else { 2 * self.words }
    }

    pub fn schema(&self) -> &'static str {
        if self.segs.is_some() { ROW_SCHEMA_SEG } else if self.v2 { ROW_SCHEMA_V2 } else { ROW_SCHEMA }
    }

    /// Compression `k`'s chain start, if it is one: segment `j`'s first compression (`Some(j)`), the outer message's first
    /// (`Some(s)`; a serial row's first, `Some(0)`). Every other compression reads the previous one's output.
    pub fn chain_start(&self, k: usize) -> Option<usize> {
        let Some(segs) = &self.segs else { return (k == 0).then_some(0) };
        let mut first = 0;
        for (j, &(_, n)) in segs.iter().enumerate() {
            if k == first {
                return Some(j);
            }
            first += n;
            if k < first {
                return None;
            }
        }
        (k == first).then_some(segs.len())
    }

    /// Row byte `byte`'s compression and its offset in the block.
    pub fn byte_at(&self, byte: usize) -> (usize, usize) {
        match &self.segs {
            None => (byte / 128, byte % 128),
            Some(segs) => {
                let j = byte / (ROW_SEG_BITS / 8);
                (segs[..j].iter().map(|x| x.1).sum::<usize>() + byte % (ROW_SEG_BITS / 8) / 128, byte % 128)
            }
        }
    }
}

/// A row's constants: its first block, the chaining value after it, and SHA-512's padding after its bytes (a whole block for
/// `sha512/row/v1`).
#[derive(Clone, Debug)]
pub struct RowConsts {
    pub prefix: Vec<u8>,
    pub midstate: [u64; 8],
    pub tail: Vec<u8>,
    /// a segmented row's segments' constants (its own are the outer message's: `Q(l)` and the padding after `64 s` bytes)
    pub segs: Vec<RowConsts>,
}

impl RowConsts {
    /// The message after the prefix: the row's first `bytes` bytes (zeros past `row`), then the padding.
    pub fn message(&self, row: &[u8], bytes: usize) -> Vec<u8> {
        let mut m = row[..bytes.min(row.len())].to_vec();
        m.resize(bytes, 0);
        m.extend_from_slice(&self.tail);
        m
    }
}

/// Column `t` of a message block's 1024 (word `t / 64`, its bit `t % 64`; the block's bytes big-endian per word) as the
/// block's byte and the byte's bit.
pub fn block_bit(t: usize) -> (usize, usize) {
    (8 * (t / 64) + 7 - (t % 64) / 8, t % 8)
}

/// Bit `t` (a block column) of block `k` of port `port`'s message after the prefix: `None` for a row bit (`8 j + b < bits`),
/// else the constant there (a fill bit of the last row byte, zero, or SHA-512's padding).
pub fn const_bit(port: &Port, rc: &RowConsts, k: usize, t: usize) -> Option<bool> {
    let (o, b) = block_bit(t);
    let j = 128 * k + o;
    if 8 * j + b < port.bits {
        return None;
    }
    let bytes = port.bytes();
    Some(j >= bytes && (rc.tail[j - bytes] >> b) & 1 == 1)
}

/// Message bit `t` (a block column) of port `port`'s compression `k` (`k == comps`: its hm96 slot's block).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MsgBit {
    /// the row's bit
    Row(usize),
    Const(bool),
    /// a segment's digest bit: column `.1` of the output of the port's compression `.0` (the segment's last)
    Digest(usize, usize),
}

pub fn msg_bit(port: &Port, rc: &RowConsts, k: usize, t: usize) -> MsgBit {
    let (o, b) = block_bit(t);
    let Some(segs) = &port.segs else {
        return match const_bit(port, rc, k, t) { Some(v) => MsgBit::Const(v), None => MsgBit::Row(8 * (128 * k + o) + b) };
    };
    let mut first = 0;
    for (j, &(m, n)) in segs.iter().enumerate() {
        if k < first + n {
            let (at, bytes) = (128 * (k - first) + o, m.div_ceil(8));
            return if 8 * at + b < m {
                MsgBit::Row(ROW_SEG_BITS * j + 8 * at + b)
            } else {
                MsgBit::Const(at >= bytes && (rc.segs[j].tail[at - bytes] >> b) & 1 == 1)
            };
        }
        first += n;
    }
    let (at, d) = (128 * (k - first) + o, 64 * segs.len());
    if at < d {
        let q = at % 64;
        MsgBit::Digest(segs[..=at / 64].iter().map(|x| x.1).sum::<usize>() - 1, 64 * (q / 8) + 56 - 8 * (q % 8) + b)
    } else {
        MsgBit::Const((rc.tail[at - d] >> b) & 1 == 1)
    }
}

/// A copy of every bit of output port `src_port` of slot `(src_net, src_idx)` into input port `dst_port` of slot
/// `(dst_net, dst_idx)`, both instance-local (per VU).
#[derive(Clone, Copy, Debug)]
pub struct Wire {
    pub dst: (usize, usize, usize),
    pub src: (usize, usize, usize),
}

#[derive(Clone)]
pub struct Composite {
    pub name: String,
    /// SHA-512 of the circuit file: the pin
    pub sha512: [u8; 64],
    /// the class of every instance it proves: SHA-512 of the file without META's bindings (`BINDING_KEYS`), which a
    /// partition names
    pub class_sha512: [u8; 64],
    pub meta: Value,
    pub nets: Vec<(String, IrUnitNet)>,
    pub ranges: Vec<Range>,
    /// every row: the input rows, then one output row per output port (its words zero-padded to whole SHA-512 blocks), or per
    /// run of output ports (`out_row_ports`) a segmented row of exactly their words
    pub ports: Vec<Port>,
    /// the input rows (META `input_ports`): `ports[..n_in]`
    pub n_in: usize,
    /// each output port's words
    pub out_words: Vec<usize>,
    /// each output row's output ports, a run in port order (META `out_row_ports`; absent: one each), covering every output
    /// port but the pinned ones
    pub out_row_ports: Vec<usize>,
    /// the pinned output ports (META `pinned_outputs`), the last ones: no row, each bit Δ's copy of a forced-zero row
    pub pinned: usize,
    pub g: usize,
    pub k_log: usize,
    pub upv: usize,
    pub unit_net: usize,
    pub leaves_in: Vec<Vec<i64>>,
    pub leaves_out: Vec<Vec<usize>>,
    /// the public input ports (META `public_inputs`, PROTOCOL.md §16.10), `(port, words)`: unit inputs that are no row but
    /// public words the verifier recomputes from public data, each naming its recomputation (seed and derivation)
    pub public_inputs: Vec<(String, usize)>,
    /// each unit's leaf ports' public words (META `leaves_public`): the word's index among the public ports' words, or -1
    pub leaves_public: Vec<Vec<i64>>,
    pub rows: Vec<RowConsts>,
    pub sha_net: usize,
    pub hm_net: usize,
    /// the circuit whose output group holds the returned outputs: the unit, or (a tail that returns them) the last stage
    pub out_net: usize,
    /// `out_net`'s `(slot, output port)` of each output word (a stage), in output-leaf order: META `out_net_ports` of ports
    /// (slot 0, the one-slot last stage) or of `[slot, port]` pairs (a repeated last stage, `out_slotted`)
    pub out_ports: Vec<(usize, usize)>,
    pub out_slotted: bool,
    /// a unit input that is a row word (attention's public query): `(unit, input port, leaf)`; low 16 bits the message bits
    pub leaf_cuts: Vec<(usize, usize, usize)>,
    /// `leaf_cuts` by unit: `(input port, leaf)`
    pub leaf_cuts_of: Vec<Vec<(usize, usize)>>,
    /// the all-zero instance's `b || c` per row (its output rows hold `dummy_outputs`)
    pub dummy_bc: Vec<[u8; BC_BYTES]>,
    /// the all-zero instance's output words
    pub dummy_outputs: Vec<u16>,
    pub leaf_scheme: Value,
    /// slots per VU of each circuit
    pub per_vu: Vec<usize>,
    pub wires: Vec<Wire>,
    /// the lookup slot types' fast witnesses, by circuit index
    pub lookups: BTreeMap<usize, LookupNet>,
    /// a typed template's instance (`crate::typed::Instance`): the unit is its root, one slot per VU, beside its parts
    pub typed: Option<TypedBlock>,
}

/// A typed template's block: the instance, and each part's circuit and slot among its layout's slots in the VU.
#[derive(Clone)]
pub struct TypedBlock {
    pub inst: crate::typed::Instance,
    pub parts: Vec<(usize, usize)>,
}

impl TypedBlock {
    /// The block column of unit coordinate `u` in VU `g` (`u` inside the instance, as its parse checks).
    pub fn col(&self, c: &Composite, g: usize, u: usize) -> usize {
        debug_assert!(self.inst.inside(u), "unit coordinate {u} outside the instance");
        if u < self.inst.root.rows.len() {
            return c.slot_col(c.unit_net, g, 0) + u;
        }
        let p = self.inst.parts.partition_point(|&(_, base)| base <= u) - 1;
        let (net, q) = self.parts[p];
        c.slot_col(net, g, q) + u - self.inst.parts[p].1
    }

    /// Whether circuit `n` is the root's or a part's (the instance's evaluation fills its slots).
    pub fn fills(&self, c: &Composite, n: usize) -> bool {
        n == c.unit_net || self.parts.iter().any(|p| p.0 == n)
    }
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

pub(crate) const SHA512_IV: [u64; 8] = [0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
                              0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179];

pub(crate) const SHA512_K: [u64; 80] = [
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
];

/// One SHA-512 compression of a 128-byte block into `cv` (FIPS 180-4 §6.4.2).
pub fn compress(cv: &mut [u64; 8], block: &[u8]) {
    let mut w: [u64; 80] = [0; 80];
    for (t, x) in w.iter_mut().take(16).enumerate() {
        *x = u64::from_be_bytes(block[8 * t..8 * t + 8].try_into().unwrap());
    }
    for t in 16..80 {
        let s0 = w[t - 15].rotate_right(1) ^ w[t - 15].rotate_right(8) ^ (w[t - 15] >> 7);
        let s1 = w[t - 2].rotate_right(19) ^ w[t - 2].rotate_right(61) ^ (w[t - 2] >> 6);
        w[t] = w[t - 16].wrapping_add(s0).wrapping_add(w[t - 7]).wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *cv;
    for t in 0..80 {
        let t1 = h.wrapping_add(e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41)).wrapping_add((e & f) ^ (!e & g))
            .wrapping_add(SHA512_K[t]).wrapping_add(w[t]);
        let t2 = (a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39)).wrapping_add((a & b) ^ (a & c) ^ (b & c));
        (h, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
    }
    for (x, y) in cv.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *x = x.wrapping_add(y);
    }
}

/// `sha512/row/v1`'s first block for an x row of `words` u16 words (`verity.ml.commitments.rowleaf.sha512_row_prefix`).
pub fn row_prefix(words: usize) -> Vec<u8> {
    let mut v = SHA512_ROW_TAG.to_vec();
    v.extend([ROLE_X, 16]);
    v.extend((words as u32).to_be_bytes());
    v.resize(128, 0);
    v
}

/// `sha512/row/v2`'s first block `P(bits)`: `"verity/sha512-row/v2\0" || u64be bits`, zero padded
/// (`verity.ml.commitments.rowleaf.sha512_row_v2_prefix`).
pub fn row_prefix_v2(bits: u64) -> Vec<u8> {
    let mut v = SHA512_ROW_V2_TAG.to_vec();
    v.extend(bits.to_be_bytes());
    v.resize(128, 0);
    v
}

/// `sha512/row-seg/v1`'s outer first block `Q(bits)`: `"verity/sha512-row-seg/v1\0" || u64be bits || u64be s || u32be 1024`,
/// zero padded (`verity.ml.commitments.rowleaf.sha512_row_seg_prefix`).
pub fn row_prefix_seg(bits: u64) -> Vec<u8> {
    let mut v = SHA512_ROW_SEG_TAG.to_vec();
    v.extend(bits.to_be_bytes());
    v.extend(bits.div_ceil(ROW_SEG_BITS as u64).to_be_bytes());
    v.extend(((ROW_SEG_BITS / 8) as u32).to_be_bytes());
    v.resize(128, 0);
    v
}

/// Segment `j` of `s`'s first block `S(j, s, m)`, `m` its bits: `"verity/sha512-row-seg/v1/segment\0" || u64be j || u64be s ||
/// u64be m`, zero padded (`verity.ml.commitments.rowleaf.sha512_row_seg_segment_prefix`).
pub fn segment_prefix(j: usize, s: usize, m: usize) -> Vec<u8> {
    let mut v = SHA512_ROW_SEG_SEGMENT_TAG.to_vec();
    for x in [j, s, m] {
        v.extend((x as u64).to_be_bytes());
    }
    v.resize(128, 0);
    v
}

/// A `sha512/row-seg/v1` row of `bits` bits: each segment's bits and compressions (its message's blocks after its constant
/// first block), then the outer message's (the `64 s` digest bytes, then SHA-512's padding).
pub fn seg_shape(bits: usize) -> (Vec<(usize, usize)>, usize) {
    let s = bits.div_ceil(ROW_SEG_BITS);
    let segs = (0..s).map(|j| {
        let m = (bits - ROW_SEG_BITS * j).min(ROW_SEG_BITS);
        (m, (m.div_ceil(8) + 17).div_ceil(128))
    }).collect();
    (segs, (64 * s + 17).div_ceil(128))
}

/// A row's constants: its prefix, the chaining value after it, the padding after its bytes, and a segmented row's segments'.
pub fn row_consts(p: &Port) -> RowConsts {
    let one = |prefix: Vec<u8>, bytes: usize| {
        let mut midstate = SHA512_IV;
        compress(&mut midstate, &prefix);
        RowConsts { prefix, midstate, tail: pad_tail(128 + bytes), segs: Vec::new() }
    };
    match &p.segs {
        None => one(if p.v2 { row_prefix_v2(p.bits as u64) } else { row_prefix(p.words) }, p.bytes()),
        Some(segs) => {
            let mut rc = one(row_prefix_seg(p.bits as u64), 64 * segs.len());
            rc.segs = segs.iter().enumerate().map(|(j, &(m, _))| one(segment_prefix(j, segs.len(), m), m.div_ceil(8))).collect();
            rc
        }
    }
}

/// `sha512/row-seg/v1`'s digest of a row of `bits` bits, its bytes `row` (`verity.ml.commitments.rowleaf.sha512_row_seg_digest`).
pub fn row_seg_digest(bits: usize, row: &[u8]) -> [u8; 64] {
    let (segs, _) = seg_shape(bits);
    let mut h = Sha512::new();
    h.update(row_prefix_seg(bits as u64));
    for (j, &(m, _)) in segs.iter().enumerate() {
        let at = j * (ROW_SEG_BITS / 8);
        h.update(Sha512::digest([segment_prefix(j, segs.len(), m), row[at..at + m.div_ceil(8)].to_vec()].concat()));
    }
    h.finalize().into()
}

/// The padding block after `total` bytes (a multiple of 128): 0x80, zeros, the 128-bit bit length.
pub fn pad_block(total: usize) -> [u8; 128] {
    let mut b = [0u8; 128];
    b[0] = 0x80;
    b[112..].copy_from_slice(&(8 * total as u128).to_be_bytes());
    b
}

/// SHA-512's padding after a message of `total` bytes: 0x80, zeros to 112 mod 128, the 128-bit bit length.
pub fn pad_tail(total: usize) -> Vec<u8> {
    let mut b = vec![0x80];
    b.resize(1 + (128 - (total + 17) % 128) % 128, 0);
    b.extend((8 * total as u128).to_be_bytes());
    b
}

fn words_be<const N: usize>(b: &[u8]) -> [u64; N] {
    std::array::from_fn(|i| u64::from_be_bytes(b[8 * i..8 * i + 8].try_into().unwrap()))
}

/// Per VU: the `sha512x3` slots, and each port's compressions as `(slot, sub)` (three per slot, ports in order).
pub fn row_slots(ports: &[Port]) -> (usize, Vec<Vec<(usize, usize)>>) {
    let mut at = 0;
    let mut where_ = Vec::new();
    for p in ports {
        where_.push((0..p.comps).map(|k| ((at + k) / PER_SLOT, (at + k) % PER_SLOT)).collect());
        at += p.comps;
    }
    (at.div_ceil(PER_SLOT), where_)
}

/// Each row's chain (compression k's `cv` from compression k-1's `out`) and its hm96 slot's `cv` (from the last; a row of
/// no compression gives its hm96 slot the midstate, a Δ constant).
fn row_wires(ports: &[Port], sha: usize, hm: usize) -> Vec<Wire> {
    let (_, where_) = row_slots(ports);
    let mut out = Vec::new();
    for (p, w) in where_.iter().enumerate() {
        for k in (1..w.len()).filter(|&k| ports[p].chain_start(k).is_none()) {
            out.push(Wire { dst: (sha, w[k].0, 2 * w[k].1), src: (sha, w[k - 1].0, w[k - 1].1) });
        }
        if let Some(last) = w.last()
            && ports[p].chain_start(w.len()).is_none()
        {
            out.push(Wire { dst: (hm, p, 0), src: (sha, last.0, last.1) });
        }
    }
    out
}

impl Composite {
    /// The statement's Ligerito trees have plain SHA-512 leaves ([`leaf_scheme_plain`], the SHA-512 build only).
    pub fn plain_leaves(&self) -> bool {
        cfg!(feature = "sha512") && self.leaf_scheme == leaf_scheme_plain()
    }

    /// `tables`: the directory of the pinned MUFU tables (`verity_flock.ir_lower.write_tables`), when the circuit has lookups.
    pub fn parse(text: &str, tables: Option<&str>) -> Result<Self, String> {
        Self::parse_with(text, tables, None)
    }

    /// [`Composite::parse`], with a templated statement's unit: the rows `flock-tmpl-unit` wrote for its META
    /// `template` and `layout`, at their positions (an `ORDER` line).
    pub fn parse_with(text: &str, tables: Option<&str>, tmpl_unit: Option<&str>) -> Result<Self, String> {
        let sha512: [u8; 64] = Sha512::digest(text.as_bytes()).into();
        let mut lines = text.split_inclusive('\n');
        let head = lines.next().ok_or("empty circuit")?.trim_end();
        // a typed statement (`crate::typed`) carries its unit as types; the prover derives its rows
        let (name, typed, tmpl) = match (head.strip_prefix("flock-circuit "), head.strip_prefix("flock-circuit/types "),
                                         head.strip_prefix("flock-circuit/tmpl ")) {
            (Some(n), _, _) => (n.to_string(), false, false),
            (_, Some(n), _) => (n.to_string(), true, false),
            (_, _, Some(n)) => (n.to_string(), false, true),
            _ => return Err("not a flock-circuit circuit".into()),
        };
        let meta: Value = serde_json::from_str(lines.next().and_then(|l| l.trim_end().strip_prefix("META ")).ok_or("no META line")?)
            .map_err(|e| format!("META: {e}"))?;
        let (format, statement) = if typed {
            (crate::typed::FORMAT, crate::typed::STATEMENT)
        } else if tmpl {
            (TMPL_FORMAT, TMPL_STATEMENT)
        } else {
            (FORMAT, STATEMENT)
        };
        if meta["format"] != format || meta["statement"] != statement {
            return Err(format!("META: not {statement}"));
        }
        let class_sha512: [u8; 64] = {
            let mut m = meta.clone();
            for k in BINDING_KEYS {
                m.as_object_mut().ok_or("META is not an object")?.remove(k);
            }
            let rest = text.splitn(3, '\n').nth(2).unwrap_or("");
            Sha512::digest(format!("{head}\nMETA {}\n{rest}", serde_json::to_string(&sorted(&m)).unwrap()).as_bytes()).into()
        };
        let is_sha512 = |v: &Value| v.as_str().is_some_and(|x| x.len() == 128 && x.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)));
        if !is_sha512(&meta["program_sha512"]) {
            return Err("META: program_sha512 (the Program the partition is over) is not a SHA-512 digest".into());
        }
        let prule = meta["partition"]["rule"].as_str().unwrap_or("");
        if ![PARTITION_V1, PARTITION_PROVISIONAL].contains(&prule) || !is_sha512(&meta["partition"]["digest"]) {
            return Err(format!("META: the partition is not a {PARTITION_V1} (or the provisional {PARTITION_PROVISIONAL}) SHA-512 digest"));
        }
        let mut nets = Vec::new();
        let parse_net = |t: &str| std::panic::catch_unwind(|| IrUnitNet::parse(t)).map_err(|_| "a derived circuit's rows do not parse".to_string());
        let mut instance = None;
        if typed && crate::typed::is_flat(&meta) {
            nets.push(("unit".to_string(), parse_net(&crate::typed::unit_rows(&meta, &name)?)?));
        } else if typed {
            for k in ["unit_log", "units_per_vu", "leaves_in", "leaves_out", "wires", "leaf_cuts", "out_net", "out_net_ports", "lookups"] {
                if meta.get(k).is_some() {
                    return Err(format!("META: {k:?}: a typed template's block wiring is derived from its types, never stated"));
                }
            }
            let (inst, mut generated) = crate::typed::instance(&meta, tables)?;
            nets.push(("root".to_string(), parse_net(&inst.root_text(&name))?));
            for (&j, (d, _, _)) in &inst.layouts {
                nets.push((d.clone(), match generated.remove(&j) { Some(net) => net, None => parse_net(&inst.layout_text(j))? }));
            }
            instance = Some(inst);
        } else if tmpl {
            if !meta["template"].is_array() || !meta["layout"].is_string() {
                return Err("META: a templated statement's template (its lines) and layout (its mode)".into());
            }
            let unit = parse_net(tmpl_unit.ok_or("a templated statement: pass its unit's rows (--tmpl-unit F, from flock-tmpl-unit)")?)?;
            if unit.order.is_none() {
                return Err("the templated unit's rows have no ORDER line (its position map)".into());
            }
            nets.push(("unit".to_string(), unit));
        }
        let mut lookups_fast: Vec<(usize, LookupNet)> = Vec::new();
        while let Some(l) = lines.next() {
            let h: Vec<&str> = l.split_whitespace().collect();
            if h.len() != 3 || h[0] != "CIRCUIT" {
                return Err(format!("circuit: expected a CIRCUIT line, found {:?}", l.chars().take(40).collect::<String>()));
            }
            if typed && h[1] == "unit" {
                return Err("CIRCUIT unit: a typed statement's unit is its types, never rows".into());
            }
            if tmpl && h[1] == "unit" {
                return Err("CIRCUIT unit: a templated statement's unit is its template, never rows".into());
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
        // one sha512/row/v1 schema for every port, or (`row_schemas`) each port's bit row and its length in bits
        let v2 = meta.get("row_schemas").is_some();
        if meta["scheme"] != SCHEME || v2 == meta.get("row_schema").is_some() || (!v2 && meta["row_schema"] != ROW_SCHEMA) {
            return Err(format!("META: rows are not {SCHEME} ({ROW_SCHEMA}, or {ROW_SCHEMA_V2} per port) leaves"));
        }
        if v2 && typed {
            return Err(format!("META: a typed statement's rows are {ROW_SCHEMA} leaves"));
        }
        let n_in = us(&meta, "input_ports")?;
        let listed_ports = meta["ports"].as_array().ok_or("META: ports")?;
        let lengths: Vec<Option<(u64, bool)>> = if v2 {
            let s = meta["row_schemas"].as_array().filter(|s| s.len() == listed_ports.len())
                .ok_or("META: row_schemas is not one [schema, bits] per port")?;
            s.iter().map(|x| match x.as_array().map(|a| a.as_slice()) {
                Some([schema, bits]) if schema == ROW_SCHEMA_V2 || schema == ROW_SCHEMA_SEG => bits.as_u64().map(|l| Some((l, schema == ROW_SCHEMA_SEG)))
                    .ok_or_else(|| "META: a bit row's length is not a u64".to_string()),
                _ => Err(format!("META: row_schemas: each port's is [{ROW_SCHEMA_V2:?} or {ROW_SCHEMA_SEG:?}, bits]")),
            }).collect::<Result<_, _>>()?
        } else {
            vec![None; listed_ports.len()]
        };
        let mut ports = Vec::new();
        for (p, bits) in listed_ports.iter().zip(&lengths) {
            let (name, words, comps) = (p[0].as_str().unwrap_or("").to_string(), p[1].as_u64().unwrap_or(0) as usize, p[2].as_u64().unwrap_or(0) as usize);
            let port = match *bits {
                None => {
                    if words == 0 || (2 * words) % 128 != 0 || comps != 2 * words / 128 {
                        return Err(format!("port {name}: not whole SHA-512 blocks"));
                    }
                    Port { name, words, comps, bits: 16 * words, v2: false, segs: None }
                }
                Some((l, seg)) => {
                    // every count derived from the length: its leaves, and its messages' blocks after their prefixes but the last
                    let bytes = l.div_ceil(8);
                    let serial = (bytes + 17).div_ceil(128) - 1;
                    // a segmented row's are at most a serial row's and 18 more per 2^13 bits: bounded before it is shaped
                    let segs = (seg && serial <= MAX_COMPS as u64).then(|| seg_shape(l as usize));
                    let want = (l.div_ceil(16) as usize, match &segs {
                        Some((s, o)) => s.iter().map(|x| x.1).sum::<usize>() + o - 1,
                        None if seg => MAX_COMPS + 1,
                        None => serial as usize,
                    });
                    if want.1 > MAX_COMPS || (words, comps) != want {
                        return Err(format!("port {name}: a row of {l} bits is {} u16 leaves and {} compressions, not META's {words} and {comps}",
                                           want.0, want.1));
                    }
                    Port { name, words, comps, bits: l as usize, v2: !seg, segs: segs.map(|x| x.0) }
                }
            };
            ports.push(port);
        }
        // each row's constants, recomputed here from its schema and length (never taken from META)
        let rows: Vec<RowConsts> = ports.iter().map(row_consts).collect();
        let listed = meta["rows"].as_array().ok_or("META: rows")?;
        if listed.len() != ports.len() || listed.iter().zip(&rows).any(|(m, r)| {
            m["prefix"].as_str() != Some(crate::hex(&r.prefix).as_str())
                || m["midstate"] != json!(r.midstate.iter().map(|w| format!("{w:016x}")).collect::<Vec<_>>())
                || m["pad"].as_str() != Some(crate::hex(&r.tail).as_str())
        }) {
            return Err(format!("META: a row's constants are not {}'s", if v2 { "sha512/row/v2 or sha512/row-seg/v1" } else { "sha512/row/v1" }));
        }
        if v2 && ports.iter().all(|p| p.comps == 0) {
            return Err(format!("META: no row has a compression before its last block (the {SHA_NET} range, the pin's, holds no slot)"));
        }
        if meta["row_key_sha512"].as_str() != Some(crate::hex(&Sha512::digest(hm96_sha512_default_key())).as_str()) {
            return Err("META: the rows' hm96 key is not hm96-sha512/v1's default key".into());
        }
        let g = us(&meta, "vus_per_block")?;
        // a typed template's unit is its instance's root, one per VU
        let upv = if instance.is_some() { 1 } else { us(&meta, "units_per_vu")? };
        let unit_net = nets.iter().position(|n| n.0 == if instance.is_some() { "root" } else { "unit" }).ok_or("no unit's expanded circuit")?;
        let mut ranges = Vec::new();
        for r in meta["ranges"].as_array().ok_or("META: ranges")? {
            let kind = match r["type"].as_str() {
                Some("mask") => Kind::Mask,
                Some("unit") => Kind::Net(nets.iter().position(|n| Some(n.0.as_str()) == r["net"].as_str()).ok_or("a range of an unknown circuit")?),
                _ => return Err("a range of an unknown type".into()),
            };
            let packed = r["packed"].as_bool().unwrap_or(false);
            ranges.push(Range { kind, slot_log: us(r, "slot_log")?, pos0: us(r, "pos0")?, count: us(r, "count")?, packed });
        }
        let ints = |v: &Value| -> Vec<Vec<i64>> { v.as_array().map(|a| a.iter().map(|u| u.as_array().map(|x| x.iter().filter_map(|y| y.as_i64()).collect()).unwrap_or_default()).collect()).unwrap_or_default() };
        let (leaves_in, leaves_out): (Vec<Vec<i64>>, Vec<Vec<usize>>) = if instance.is_some() {
            (vec![Vec::new()], vec![Vec::new()])
        } else {
            (ints(&meta["leaves_in"]), ints(&meta["leaves_out"]).into_iter().map(|u| u.into_iter().map(|x| x as usize).collect()).collect())
        };
        let public_inputs: Vec<(String, usize)> = match meta.get("public_inputs") {
            None => Vec::new(),
            Some(v) => v.as_array().filter(|a| !a.is_empty()).ok_or("META: public_inputs is not a list of ports")?.iter().map(|p| {
                let s = |k: &str| p[k].as_str().filter(|x| !x.is_empty());
                let keys: Option<Vec<&str>> = p.as_object().map(|m| m.keys().map(|k| k.as_str()).collect());
                match (keys.as_deref(), s("port"), p["words"].as_u64(), s("seed"), s("derivation")) {
                    (Some(["derivation", "port", "seed", "words"]), Some(nm), Some(w), Some(_), Some(_)) if w > 0 => Ok((nm.to_string(), w as usize)),
                    _ => Err("META: a public input names no port, words, or recomputation (a non-empty seed and derivation)".to_string()),
                }
            }).collect::<Result<_, _>>()?,
        };
        if public_inputs.is_empty() != meta.get("leaves_public").is_none() {
            return Err("META: leaves_public without public_inputs, or public_inputs without leaves_public".into());
        }
        let leaves_public = if public_inputs.is_empty() { Vec::new() } else { ints(&meta["leaves_public"]) };
        let dummy_bc = meta["dummy"]["bc"].as_array().ok_or("META: dummy bc")?.iter()
            .map(|d| hexb(d.as_str().unwrap_or("")).and_then(|b| b.try_into().map_err(|_| "b || c length".to_string()))).collect::<Result<Vec<[u8; BC_BYTES]>, String>>()?;
        let dummy_outputs = meta["dummy"]["outputs"].as_array().ok_or("META: dummy outputs")?.iter().map(|x| x.as_u64().unwrap_or(u64::MAX) as u16).collect();
        let out_words: Vec<usize> = meta["out_ports"].as_array().ok_or("META: out_ports")?.iter().map(|p| p[1].as_u64().unwrap_or(0) as usize).collect();
        let pinned = match meta.get("pinned_outputs") {
            None => 0,
            Some(v) => {
                let out_names: Vec<&str> = meta["out_ports"].as_array().map(|a| a.iter().map(|p| p[0].as_str().unwrap_or("")).collect()).unwrap_or_default();
                let names: Option<Vec<&str>> = v.as_array().and_then(|a| a.iter().map(Value::as_str).collect());
                let k = names.as_ref().map_or(0, Vec::len);
                let zero = |o: usize| meta["dummy"]["outputs"].get(o).and_then(Value::as_u64) == Some(0);
                let w0: usize = out_words[..out_words.len().saturating_sub(k)].iter().sum();
                if k == 0 || k > out_names.len() || names.as_deref() != Some(&out_names[out_names.len() - k..]) || !v2 || typed || tmpl
                    || !(w0..out_words.iter().sum()).all(zero)
                {
                    return Err(PINNED_REFUSED.into());
                }
                k
            }
        };
        let rowed = out_words.len() - pinned;
        let out_row_ports: Vec<usize> = match meta.get("out_row_ports") {
            None => vec![1; rowed],
            Some(v) => v.as_array().and_then(|a| a.iter().map(|x| x.as_u64().filter(|&k| k >= 1 && k <= rowed as u64).map(|k| k as usize)).collect())
                .filter(|r: &Vec<usize>| r.iter().sum::<usize>() == rowed)
                .ok_or("META: out_row_ports is not each output row's run of output ports, in order, covering every output port once")?,
        };
        let net_of = |nm: &str| nets.iter().position(|n| n.0 == nm).ok_or(format!("META: unknown circuit {nm}"));
        let mut per_vu = vec![0usize; nets.len()];
        per_vu[unit_net] = upv;
        if let Some(m) = meta["per_vu"].as_object() {
            for (k, v) in m {
                per_vu[net_of(k)?] = v.as_u64().unwrap_or(0) as usize;
            }
        }
        let (sha_net, hm_net) = (net_of(SHA_NET)?, net_of(HM_NET)?);
        let out_net = if instance.is_some() { unit_net } else { net_of(meta["out_net"].as_str().unwrap_or("unit"))? };
        let entry = |x: &Value| -> (usize, usize) {
            let n = |v: &Value| v.as_u64().map_or(usize::MAX, |v| v as usize);
            match x.as_array() {
                Some(p) if p.len() == 2 => (n(&p[0]), n(&p[1])),
                Some(_) => (usize::MAX, usize::MAX),
                None => (0, n(x)),
            }
        };
        let out_ports: Vec<(usize, usize)> = meta["out_net_ports"].as_array().map(|a| a.iter().map(entry).collect()).unwrap_or_default();
        let out_slotted = meta["out_net_ports"].as_array().is_some_and(|a| a.iter().any(Value::is_array));
        let leaf_cuts: Vec<(usize, usize, usize)> = meta["leaf_cuts"].as_array().map(|a| a.iter().map(|w| {
            (w[1].as_u64().unwrap_or(u64::MAX) as usize, w[2].as_u64().unwrap_or(u64::MAX) as usize, w[3].as_u64().unwrap_or(u64::MAX) as usize)
        }).collect()).unwrap_or_default();
        if meta["leaf_cuts"].as_array().is_some_and(|a| a.iter().any(|w| w[0] != "unit")) {
            return Err("META: a leaf cut into anything but a unit".into());
        }
        let mut wires = if instance.is_some() { row_wires(&ports, sha_net, hm_net) } else { Vec::new() };
        for w in meta["wires"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
            let n = |i: usize| w[i].as_str().ok_or("a wire's circuit".to_string()).and_then(net_of);
            let u = |i: usize| w[i].as_u64().map(|x| x as usize).ok_or("a wire's index".to_string());
            wires.push(Wire { dst: (n(0)?, u(1)?, u(2)?), src: (n(3)?, u(4)?, u(5)?) });
        }
        // the rows' wires come first, exactly as the ports imply
        let derived = row_wires(&ports, sha_net, hm_net);
        if wires.len() < derived.len() || wires.iter().zip(&derived).any(|(a, b)| a.dst != b.dst || a.src != b.src)
            || wires[derived.len()..].iter().any(|w| [sha_net, hm_net].contains(&w.dst.0) || [sha_net, hm_net].contains(&w.src.0))
        {
            return Err("META: the rows' wires are not the chains and hm96 inputs the ports imply".into());
        }
        let c = Composite {
            name, sha512, class_sha512, k_log: us(&meta, "k_log")?, leaf_scheme: meta["leaf_scheme"].clone(), meta, nets, ranges, ports, n_in,
            leaf_cuts_of: { let mut v = vec![Vec::new(); upv]; for &(u, j, l) in &leaf_cuts { if u < upv { v[u].push((j, l)); } } v },
            out_words, out_row_ports, pinned, g, upv, unit_net, leaves_in, leaves_out, public_inputs, leaves_public, rows, sha_net, hm_net, out_net, out_ports, out_slotted, leaf_cuts, dummy_bc, dummy_outputs, per_vu, wires,
            lookups: lookups_fast.into_iter().collect(), typed: None,
        };
        let typed = instance.map(|inst| {
            let mut seen: BTreeMap<usize, usize> = BTreeMap::new();
            let parts = inst.parts.iter().map(|&(j, _)| {
                let n = c.nets.iter().position(|x| x.0 == inst.layouts[&j].0).unwrap();
                let q = seen.entry(j).or_default();
                *q += 1;
                (n, *q - 1)
            }).collect();
            TypedBlock { inst, parts }
        });
        let c = Composite { typed, ..c };
        c.check()?;
        block_in_range(c.k_log, c.range_of(c.hm_net).slot_log, pin_of(&c))?;
        regions_in_range(&Stmt::regions(&c)?)?;
        Ok(c)
    }

    /// The layout's shape: aligned, disjoint ranges inside the block; the comp ranges' roles cover every run and parent of
    /// every VU exactly once; the unit ranges hold `g x units_per_vu` units whose leaf maps are the unit's expanded circuit's ports.
    fn check(&self) -> Result<(), String> {
        // first, before anything shifts by them (as HmRow.check): the block and every range's slots bounded, so an empty packed
        // range can't carry an oversized slot_log and no shift or span below overflows
        if !(7..=IN_RANGE_K_LOG).contains(&self.k_log) {
            return Err(format!("k_log {}: not in [7, {IN_RANGE_K_LOG}] (Stmt.InRange)", self.k_log));
        }
        for (i, r) in self.ranges.iter().enumerate() {
            range_in_block(self.k_log, r).map_err(|e| format!("range {i}: {e}"))?;
        }
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for r in &self.ranges {
            let (span, align) = if r.packed { (r.count << r.slot_log, 1 << r.slot_log) } else { (r.count.next_power_of_two() << r.slot_log, r.count.next_power_of_two() << r.slot_log) };
            if r.packed && (r.kind == Kind::Net(self.hm_net) || r.kind == Kind::Net(self.out_net) || r.kind == Kind::Mask) {
                return Err("a packed range that a region reads".into());
            }
            if r.start() % align != 0 || r.start() + span > 1 << self.k_log {
                return Err("a range is not aligned to its span or overruns the block".into());
            }
            if let Kind::Net(i) = r.kind
                && self.nets[i].1.unit_log != r.slot_log
            {
                return Err("a unit range's slot size is not its circuit's".into());
            }
            if r.kind == Kind::Mask && r.count != 1 {
                return Err("not one mask slot".into());
            }
            spans.push((r.start(), r.start() + span));
        }
        spans.sort();
        if spans.windows(2).any(|w| w[0].1 > w[1].0) {
            return Err("two ranges overlap".into());
        }
        let (n_sha, _) = row_slots(&self.ports);
        if self.per_vu[self.sha_net] != n_sha || self.per_vu[self.hm_net] != self.ports.len().next_power_of_two() {
            return Err("the rows' slots per VU are not the ports' compressions (three per slot) and one hm96 slot per port".into());
        }
        let out_names: Vec<&str> = self.meta["out_ports"].as_array().map(|a| a.iter().map(|p| p[0].as_str().unwrap_or("")).collect()).unwrap_or_default();
        let mut names: Vec<&str> = self.ports.iter().take(self.n_in).map(|p| p.name.as_str()).chain(out_names.iter().copied()).collect();
        names.sort_unstable();
        if self.n_in > self.ports.len() || self.ports.len() - self.n_in != self.out_row_ports.len() || names.windows(2).any(|w| w[0] == w[1])
            || self.out_rows().iter().enumerate().any(|(j, p)| {
                let (first, k, _, w) = self.out_group(j);
                let padded = k == 1 && p.words == w.div_ceil(64) * 64 && p.bits == 16 * p.words;
                out_names.get(first) != Some(&p.name.as_str())
                    || !if p.segs.is_some() { p.bits == 16 * w } else if p.v2 { p.bits == 16 * w || padded } else { padded }
            })
        {
            return Err(OUT_ROWS_REFUSED.into());
        }
        let (sha, hm) = (&self.nets[self.sha_net].1, &self.nets[self.hm_net].1);
        if port_cols(&sha.in_groups) != (0..2 * PER_SLOT).map(|i| (port_cols(&sha.in_groups)[i].0, if i % 2 == 0 { 512 } else { 1024 })).collect::<Vec<_>>()
            || port_cols(&sha.out_groups).iter().map(|x| x.1).collect::<Vec<_>>() != vec![512; PER_SLOT]
            || port_cols(&hm.in_groups).iter().map(|x| x.1).collect::<Vec<_>>() != vec![512, 1024, 8 * SALT_BYTES]
            || port_cols(&hm.out_groups).iter().map(|x| x.1).collect::<Vec<_>>() != vec![8 * BC_BYTES]
        {
            return Err("the sha512x3 and hm96 circuits' ports are not (cv, msg) x 3 -> out x 3 and (cv, pad, y) -> bc".into());
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
        if let Some(t) = &self.typed {
            if !self.public_inputs.is_empty() {
                return Err("public inputs: a typed statement's instance reads only rows".into());
            }
            self.check_wiring()?;
            return self.check_typed(t);
        }
        let unit = &self.nets[self.unit_net].1;
        let ur: Vec<&Range> = self.ranges.iter().filter(|r| r.kind == Kind::Net(self.unit_net)).collect();
        if ur.len() != 1 || ur[0].count != self.g * self.upv || (self.out_net == self.unit_net && !ur[0].count.is_power_of_two()) {
            return Err("not one unit range of vus_per_block x units_per_vu (a power of two) units".into());
        }
        let tail_out = self.out_net != self.unit_net;
        if unit.in_groups.is_empty() || unit.in_groups.len() > 2 || unit.out_groups.is_empty() || unit.out_groups.len() > 2
            || unit.in_groups[0].bits.iter().any(|&b| b != 16) || (!tail_out && unit.out_groups[0].bits.iter().any(|&b| b != 16))
        {
            return Err("the unit's ports are not a group of 16-bit leaf inputs (then cut inputs) and one of 16-bit returned outputs (then cut outputs)".into());
        }
        if tail_out {
            let n_out: usize = self.out_words.iter().sum();
            let net = &self.nets[self.out_net].1;
            let oc = port_cols(&net.out_groups);
            if !self.nets[self.out_net].0.starts_with("stage") || self.leaves_out.iter().any(|l| !l.is_empty()) {
                return Err("the returned outputs are not the last stage's only output group: one 16-bit port per output word".into());
            }
            if self.out_slotted {
                let mut seen = BTreeSet::new();
                if self.out_ports.len() != n_out || self.out_ports.iter().any(|&(s, p)| {
                    s >= self.per_vu[self.out_net] || p >= oc.len() || oc[p].1 != 16 || !seen.insert((s, p))
                }) {
                    return Err("the returned outputs are not one 16-bit port of a slot of the last stage per output word, each once".into());
                }
            } else if self.out_ports != (0..oc.len()).map(|p| (0, p)).collect::<Vec<_>>() || oc.len() != n_out
                || oc.iter().any(|x| x.1 != 16) || net.out_groups.len() != 1 || !(16 * n_out).is_power_of_two()
            {
                return Err("the returned outputs are not the last stage's only output group: one 16-bit port per output word".into());
            }
        }
        self.check_wiring()?;
        let n_in = self.in_words();
        let n_out: usize = self.out_words.iter().sum();
        if self.leaves_in.len() != self.upv || self.leaves_out.len() != self.upv {
            return Err("the leaf maps are not one per unit".into());
        }
        // a tail that returns the outputs returns every one of them (the units return none)
        let mut seen_out = vec![tail_out; n_out];
        for u in 0..self.upv {
            if self.leaves_in[u].len() != unit.in_groups[0].bits.len() || self.leaves_in[u].iter().any(|&x| x < -1 || x >= n_in as i64)
                || (!tail_out && self.leaves_out[u].len() != unit.out_groups[0].bits.len())
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
        // Δ copies the forced-zero row onto each pinned bit's A and B rows: only under B the constant alone does that force
        // the bit and its gate's value to zero (elsewhere it would only force A·z · B·z to zero, or free the bit)
        for (n, r) in self.pinned_rows() {
            let net = &self.nets[n].1;
            if (0..16).any(|t| r.checked_add(t).and_then(|i| net.b.get(i)).map(Vec::as_slice) != Some(&[net.const_pos][..])) {
                return Err(PINNED_LINEAR.into());
            }
        }
        if unit.useful >= 1 << unit.unit_log {
            return Err("the unit fills its slot: no forced-zero row for a zero leaf".into());
        }
        // a tail that returns the outputs has no power-of-two count, which alone refuses zero units otherwise
        if self.upv == 0 {
            return Err("units_per_vu is 0: a VU of no unit".into());
        }
        if self.g == 0 {
            return Err("vus_per_block is 0: a block of no VU".into());
        }
        // the Digest region's VU bits are `trailing_zeros(g)` here and `Nat.log2 g` in the Lean verifier, which pin every
        // VU, and agree, only at a power of two
        if !self.g.is_power_of_two() {
            return Err("vus_per_block is not a power of two: the Digest region would pin some VUs and not others".into());
        }
        if self.dummy_bc.len() != self.ports.len() || self.dummy_outputs.len() != n_out {
            return Err("the pinned dummy values are not one per port and output leaf".into());
        }
        if self.leaf_scheme != leaf_scheme() && !self.plain_leaves() {
            return Err(format!("leaf-commitment scheme {} is not one this verifier implements", self.leaf_scheme));
        }
        self.check_public_inputs()
    }

    /// The public input ports (§16.10): each named once, and no row's name; one public leaf map per unit, each public leaf a
    /// zero leaf of `leaves_in`; the unit range unpacked, since the `in` regions read it; their words
    /// ([`Self::public_leaf_words`]) inside the leaf group, each leaf in them a public or a zero leaf in every unit.
    fn check_public_inputs(&self) -> Result<(), String> {
        if self.public_inputs.is_empty() {
            return Ok(());
        }
        let mut names: Vec<&str> = self.public_inputs.iter().map(|p| p.0.as_str()).chain(self.ports.iter().map(|p| p.name.as_str())).collect();
        names.sort_unstable();
        if names.windows(2).any(|w| w[0] == w[1]) {
            return Err("public inputs: a port named twice, or both a row and a public input".into());
        }
        let n_pub = self.public_words() as i64;
        if self.leaves_public.len() != self.upv || self.leaves_public.iter().zip(&self.leaves_in).any(|(lp, li)| {
            lp.len() != li.len() || lp.iter().zip(li).any(|(&q, &x)| q < -1 || q >= n_pub || (q >= 0 && x != -1))
        }) {
            return Err("public inputs: the public leaf maps are not one per unit, each public leaf a public word and a zero leaf of leaves_in".into());
        }
        let ur = self.unit_range();
        if ur.packed {
            return Err("public inputs: a packed unit range (the in region reads it)".into());
        }
        if 7 + log2ceil(ur.count) > PT_LOCAL {
            return Err(format!("public inputs: {} unit slots, too many for an in region of at most {PT_LOCAL} free bits", ur.count));
        }
        let words = self.public_leaf_words();
        if words.is_empty() {
            return Err("public inputs: no unit reads a public word".into());
        }
        let g0 = &self.unit().in_groups[0];
        for (w, e) in aligned_blocks(&words) {
            if w < g0.col || w + (1 << e) > g0.col + g0.words {
                return Err(format!("public inputs: the in region's words {w}..{} leave the unit's leaf group", w + (1 << e)));
            }
        }
        for (lp, li) in self.leaves_public.iter().zip(&self.leaves_in) {
            if let Some(j) = (0..li.len()).find(|&j| words.contains(&(g0.col + j / 8)) && lp[j] < 0 && li[j] >= 0) {
                return Err(format!("public inputs: leaf {j} reads a row word inside word {}, which holds a public word", g0.col + j / 8));
            }
        }
        Ok(())
    }

    /// The public input ports' words (one instance's).
    pub fn public_words(&self) -> usize {
        self.public_inputs.iter().map(|p| p.1).sum()
    }

    /// Whether block column `q` is a bit of a public leaf: in unit slot `k`'s leaf group (unit `k % units_per_vu`), leaf `j`'s
    /// 16 bits from `col 128 + 16 j`, a public word.
    pub fn is_public_col(&self, q: usize) -> bool {
        let ur = self.unit_range();
        let Some(o) = q.checked_sub(ur.start()) else { return false };
        let (k, at) = (o >> ur.slot_log, o & ((1 << ur.slot_log) - 1));
        let Some(x) = at.checked_sub(self.unit().in_groups[0].col * WORD) else { return false };
        k < ur.count && self.leaves_public.get(k % self.upv).and_then(|l| l.get(x / 16)).is_some_and(|&p| p >= 0)
    }

    /// The 128-bit words of a unit slot holding some unit's public leaf (leaf `j` in word `col + j / 8` of the leaf group),
    /// ascending.
    pub fn public_leaf_words(&self) -> BTreeSet<usize> {
        let Some(col) = self.unit().in_groups.first().map(|g| g.col) else { return BTreeSet::new() };
        self.leaves_public.iter().flat_map(|l| l.iter().enumerate().filter(|x| *x.1 >= 0).map(move |(j, _)| col + j / 8)).collect()
    }

    /// The `in` regions (§16.10), as `(w, s)`: `2^s` words of every unit slot from word `w`. The public leaves' words
    /// ([`Self::public_leaf_words`]) in aligned blocks of `2^e` words ([`aligned_blocks`]), each split into `2^(e - s)`
    /// regions, `s = min(e, PT_LOCAL - 7 - log2ceil(units))` (`7 + s` free bits beside the slot bits).
    pub fn in_regions(&self) -> Vec<(usize, usize)> {
        let top = PT_LOCAL.saturating_sub(7 + log2ceil(self.unit_range().count));
        aligned_blocks(&self.public_leaf_words())
            .into_iter()
            .flat_map(|(w, e)| {
                let s = e.min(top);
                (0..1usize << (e - s)).map(move |r| (w + (r << s), s))
            })
            .collect()
    }

    /// Every input port of every slot is wired exactly once: a unit's leaf ports by its leaf map, the rest by a wire. The
    /// row circuits' inputs are the rows' wires and the rows; a typed template's root and parts take no wire (their inputs
    /// are Δ's copies and bindings).
    fn check_wiring(&self) -> Result<(), String> {
        let unit = &self.nets[self.unit_net].1;
        let instance = |n: usize| self.typed.as_ref().is_some_and(|t| t.fills(self, n));
        let leaf_ports = if self.typed.is_some() { 0 } else { unit.in_groups[0].bits.len() };
        let mut wired: BTreeMap<(usize, usize, usize), usize> = BTreeMap::new();
        for w in &self.wires {
            let (dn, di, dp) = w.dst;
            let (sn, si, sp) = w.src;
            let (dc, sc) = (port_cols(&self.nets[dn].1.in_groups), port_cols(&self.nets[sn].1.out_groups));
            if di >= self.per_vu[dn] || si >= self.per_vu[sn] || dp >= dc.len() || sp >= sc.len() || dc[dp].1 != sc[sp].1 {
                return Err(format!("a wire {w:?} outside its slots' ports or of unequal widths"));
            }
            if instance(dn) || instance(sn) {
                return Err("a wire into or out of a typed instance, whose wiring is derived".into());
            }
            if dn == self.unit_net && dp < leaf_ports {
                return Err("a wire into a unit's leaf port".into());
            }
            *wired.entry(w.dst).or_default() += 1;
        }
        for &(u, j, leaf) in &self.leaf_cuts {
            let n_in = self.in_words();
            let pc = port_cols(&unit.in_groups);
            if u >= self.upv || j < leaf_ports || j >= pc.len() || pc[j].1 < 16 || leaf >= n_in {
                return Err("a leaf cut outside the unit's cut inputs or the rows".into());
            }
            *wired.entry((self.unit_net, u, j)).or_default() += 1;
        }
        for (n, (nm, net)) in self.nets.iter().enumerate() {
            if n == self.sha_net || n == self.hm_net || instance(n) {
                continue;
            }
            let first = if n == self.unit_net { leaf_ports } else { 0 };
            for i in 0..self.per_vu[n] {
                for p in first..port_cols(&net.in_groups).len() {
                    if wired.get(&(n, i, p)).copied().unwrap_or(0) != 1 {
                        return Err(format!("input port {p} of {nm} slot {i} is not wired exactly once"));
                    }
                }
            }
        }
        Ok(())
    }

    /// A typed template's block: the root one slot per VU, a power of two of them;
    /// its inputs the rows' words and its outputs the output words (16 bits each, one output group); each placed layout's
    /// slots its parts; a forced-zero row in the root's slot.
    fn check_typed(&self, t: &TypedBlock) -> Result<(), String> {
        let root = &self.nets[self.unit_net].1;
        let n_out: usize = self.out_words.iter().sum();
        if self.per_vu[self.unit_net] != 1 {
            return Err("the instance's root is not one slot per VU".into());
        }
        if !self.range_of(self.unit_net).count.is_power_of_two() {
            return Err("the instance's roots (vus_per_block) are not a power of two".into());
        }
        if t.inst.ins != self.ports[..self.n_in].iter().map(|p| (p.words, 16)).collect::<Vec<_>>()
            || t.inst.outs != self.out_words.iter().map(|&w| (w, 16)).collect::<Vec<_>>()
        {
            return Err("the instance type's ports are not the input rows' words and the output words".into());
        }
        if root.out_groups.len() != 1 || root.out_groups[0].bits.iter().sum::<usize>() != 16 * n_out || !(16 * n_out).is_power_of_two() {
            return Err("the instance's outputs are not one output group of the output words".into());
        }
        for (&j, (d, s, _)) in &t.inst.layouts {
            let n = self.nets.iter().position(|x| &x.0 == d).unwrap();
            if self.per_vu[n] != t.inst.parts.iter().filter(|p| p.0 == j).count() || self.nets[n].1.unit_log != *s {
                return Err(format!("layout {}..: not one slot of its range per part", &d[..16]));
            }
        }
        if root.useful >= 1 << root.unit_log {
            return Err("the instance's root fills its slot: no forced-zero row for a zero bit".into());
        }
        if self.dummy_bc.len() != self.ports.len() || self.dummy_outputs.len() != n_out {
            return Err("the pinned dummy values are not one per port and output leaf".into());
        }
        if self.leaf_scheme != leaf_scheme() && !self.plain_leaves() {
            return Err(format!("leaf-commitment scheme {} is not one this verifier implements", self.leaf_scheme));
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

    /// `verity/flock-circuit`, or `verity/flock-circuit/types` for a typed statement, `verity/flock-circuit/tmpl` for a
    /// templated one
    pub fn statement(&self) -> &str {
        self.meta["statement"].as_str().unwrap_or(STATEMENT)
    }

    /// Its statement's tags.
    pub fn tags(&self) -> &'static Tags {
        match self.statement() {
            crate::typed::STATEMENT => &TYPED_TAGS,
            TMPL_STATEMENT => &TMPL_TAGS,
            _ => &TAGS,
        }
    }

    pub fn unit_range(&self) -> &Range {
        self.ranges.iter().find(|r| r.kind == Kind::Net(self.unit_net)).unwrap()
    }

    pub fn row_bytes(&self) -> usize {
        self.ports.iter().map(|p| 2 * p.words).sum()
    }

    /// the input rows' words (the leaves a unit reads)
    pub fn in_words(&self) -> usize {
        self.ports[..self.n_in].iter().map(|p| p.words).sum()
    }

    pub fn out_rows(&self) -> &[Port] {
        &self.ports[self.n_in.min(self.ports.len())..]
    }

    /// Output row `j`'s run of output ports: its first port, their count, the output word it starts at and their words.
    pub fn out_group(&self, j: usize) -> (usize, usize, usize, usize) {
        let first: usize = self.out_row_ports[..j].iter().sum();
        let k = self.out_row_ports[j];
        (first, k, self.out_words[..first].iter().sum(), self.out_words[first..first + k].iter().sum())
    }

    /// Row `p` of the all-zero instance (u16 LE): zero, or for an output row the zero instance's outputs, zero-padded.
    pub fn dummy_row(&self, p: usize) -> Vec<u8> {
        let mut row = vec![0u8; 2 * self.ports[p].words];
        if p >= self.n_in {
            let (_, _, at, w) = self.out_group(p - self.n_in);
            for (k, v) in self.dummy_outputs[at..at + w].iter().enumerate() {
                row[2 * k..2 * k + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        row
    }

    /// `(g, o)`: the first column of output word `o` of VU `g` (16 bits from it), as the circuit computes it: the unit that
    /// returns it, the last stage's output port, or a typed instance's root's output group.
    pub fn out_col(&self) -> impl Fn(usize, usize) -> usize + '_ {
        let unit = self.unit();
        let ret: Vec<(usize, usize)> = {
            let mut r = vec![(0, 0); self.out_words.iter().sum()];
            for (u, l) in self.leaves_out.iter().enumerate() {
                for (q, &o) in l.iter().enumerate() {
                    r[o] = (u, q);
                }
            }
            r
        };
        let unit_cols = port_cols(unit.out_groups.get(..1).unwrap_or(&[]));
        let tail_cols = port_cols(&self.nets[self.out_net].1.out_groups);
        move |g, o| {
            if self.typed.is_some() {
                self.slot_col(self.unit_net, g, 0) + unit.out_groups[0].col * WORD + 16 * o
            } else if self.out_net != self.unit_net {
                let (s, p) = self.out_ports[o];
                self.slot_col(self.out_net, g, s) + tail_cols[p].0
            } else {
                let (u, q) = ret[o];
                self.slot_col(self.unit_net, g, u) + unit_cols[q].0
            }
        }
    }

    /// The pinned output words' first rows in the circuit that computes them, `(net, row)` per word (16 bits from it): what
    /// [`Self::out_col`] places in a slot, for an untyped statement.
    pub fn pinned_rows(&self) -> Vec<(usize, usize)> {
        let w0: usize = self.out_words[..self.out_words.len() - self.pinned].iter().sum();
        let unit_cols = port_cols(self.unit().out_groups.get(..1).unwrap_or(&[]));
        let tail_cols = port_cols(&self.nets[self.out_net].1.out_groups);
        (w0..self.out_words.iter().sum()).map(|o| if self.out_net != self.unit_net {
            (self.out_net, self.out_ports.get(o).and_then(|&(_, p)| tail_cols.get(p)).map_or(usize::MAX, |x| x.0))
        } else {
            let q = self.leaves_out.iter().find_map(|l| l.iter().position(|&x| x == o));
            (self.unit_net, q.and_then(|q| unit_cols.get(q)).map_or(usize::MAX, |x| x.0))
        }).collect()
    }

    /// `(g, p, byte, t)`: bit `t` of row byte `byte` of VU `g`'s port `p` as a column, a message bit of its compression (the
    /// block's bytes big-endian per word): a `sha512x3` slot's message, or for the last block the hm96 slot's `pad`.
    pub fn msg_col(&self) -> impl Fn(usize, usize, usize, usize) -> usize + '_ {
        let (_, where_) = row_slots(&self.ports);
        let sin = port_cols(&self.nets[self.sha_net].1.in_groups);
        let hin = port_cols(&self.nets[self.hm_net].1.in_groups);
        move |g, p, byte, t| {
            let (k, o) = self.ports[p].byte_at(byte);
            let at = 64 * (o / 8) + 56 - 8 * (o % 8) + t;
            match where_[p].get(k) {
                Some(&(slot, sub)) => self.slot_col(self.sha_net, g, slot) + sin[2 * sub + 1].0 + at,
                None => self.slot_col(self.hm_net, g, p) + hin[1].0 + at,
            }
        }
    }

    /// Committed rows and AND items per VU, over every circuit but the rows' (a typed template: its root and parts, the
    /// ANDs its types'); a flat circuit's ANDs are its rows with an entry outside its port groups and constant.
    pub fn sizes_per_vu(&self) -> (usize, usize) {
        let ours = (0..self.nets.len()).filter(|&n| n != self.sha_net && n != self.hm_net);
        let rows = ours.clone().map(|n| self.per_vu[n] * self.nets[n].1.useful).sum();
        if let Some(t) = &self.typed {
            return (rows, t.inst.ands);
        }
        let ands = ours.map(|n| {
            let net = &self.nets[n].1;
            let port = |r: usize| net.in_groups.iter().chain(&net.out_groups).any(|g| (g.col * WORD..(g.col + g.words) * WORD).contains(&r));
            self.per_vu[n] * (0..net.useful).filter(|&r| r != net.const_pos && !port(r) && !(net.a[r].is_empty() && net.b[r].is_empty())).count()
        }).sum();
        (rows, ands)
    }

}

/// The pinned constant column: the first slot of the first range that has one.
pub fn pin_of(c: &Composite) -> usize {
    let r = c.ranges.iter().find(|r| r.kind != Kind::Mask).unwrap();
    r.start() + match r.kind {
        Kind::Net(i) => c.nets[i].1.const_pos,
        Kind::Mask => unreachable!(),
    }
}

/// Δ (A and B), derived from the pinned circuit: every slot's constant, the rows' constants (midstate, padding block), the unit
/// leaf wiring (each leaf bit a copy of its message bit), every wire (the rows' chains first), and each output row's message
/// bits (copies of the output bits).
pub fn delta(c: &Composite, pin: usize) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    let (mut da, mut db) = (Vec::new(), Vec::new());
    let both = |da: &mut Vec<(u32, u32)>, db: &mut Vec<(u32, u32)>, i: usize, src: usize| {
        for d in [da, db] {
            d.push((i as u32, i as u32));
            d.push((i as u32, src as u32));
        }
    };
    for r in &c.ranges {
        let Kind::Net(i) = r.kind else { continue };
        for s in 0..r.count {
            let k = r.slot(s) + c.nets[i].1.const_pos;
            if k != pin {
                both(&mut da, &mut db, k, pin);
            }
        }
    }
    let (_, where_) = row_slots(&c.ports);
    let (sin, hin) = (port_cols(&c.nets[c.sha_net].1.in_groups), port_cols(&c.nets[c.hm_net].1.in_groups));
    let msg_col = c.msg_col();
    let zero = c.unit_range().slot(0) + c.unit().useful;
    let sout = port_cols(&c.nets[c.sha_net].1.out_groups);
    for g in 0..c.g {
        for (p, w) in where_.iter().enumerate() {
            let (port, rc) = (&c.ports[p], &c.rows[p]);
            // a segmented row: each chain start's midstate (a segment's, the outer message's), every constant message bit,
            // and the outer message's digest bits copies of the segments' last outputs
            if port.segs.is_some() {
                for k in 0..=port.comps {
                    let (cv, msg) = match w.get(k) {
                        Some(&(slot, sub)) => (c.slot_col(c.sha_net, g, slot) + sin[2 * sub].0, c.slot_col(c.sha_net, g, slot) + sin[2 * sub + 1].0),
                        None => (c.slot_col(c.hm_net, g, p) + hin[0].0, c.slot_col(c.hm_net, g, p) + hin[1].0),
                    };
                    if let Some(j) = port.chain_start(k) {
                        let mid = if j < rc.segs.len() { &rc.segs[j].midstate } else { &rc.midstate };
                        for t in 0..512 {
                            both(&mut da, &mut db, cv + t, if (mid[t / 64] >> (t % 64)) & 1 == 1 { pin } else { zero });
                        }
                    }
                    for t in 0..1024 {
                        match msg_bit(port, rc, k, t) {
                            MsgBit::Const(v) => both(&mut da, &mut db, msg + t, if v { pin } else { zero }),
                            MsgBit::Digest(kk, col) => both(&mut da, &mut db, msg + t, c.slot_col(c.sha_net, g, w[kk].0) + sout[w[kk].1].0 + col),
                            MsgBit::Row(_) => {}
                        }
                    }
                }
                continue;
            }
            let cv = match w.first() {
                Some(&(slot, sub)) => c.slot_col(c.sha_net, g, slot) + sin[2 * sub].0,
                None => c.slot_col(c.hm_net, g, p) + hin[0].0,
            };
            for t in 0..512 {
                both(&mut da, &mut db, cv + t, if (rc.midstate[t / 64] >> (t % 64)) & 1 == 1 { pin } else { zero });
            }
            // the last block's constant bits (a sha512/row/v1 row's: all of it), then those of the blocks before it from the
            // first that holds a bit at or past the row's length
            let pad = c.slot_col(c.hm_net, g, p) + hin[1].0;
            for t in 0..1024 {
                if let Some(v) = const_bit(port, rc, port.comps, t) {
                    both(&mut da, &mut db, pad + t, if v { pin } else { zero });
                }
            }
            for k in port.bits / 1024..port.comps {
                let msg = c.slot_col(c.sha_net, g, w[k].0) + sin[2 * w[k].1 + 1].0;
                for t in 0..1024 {
                    if let Some(v) = const_bit(port, rc, k, t) {
                        both(&mut da, &mut db, msg + t, if v { pin } else { zero });
                    }
                }
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
                // a zero leaf, or a bit row's leaf bit past its length: the slot's forced-zero row
                let src = if leaf < 0 {
                    base + unit.useful
                } else {
                    let leaf = leaf as usize;
                    let p = starts.iter().rposition(|&s| s <= leaf).unwrap();
                    let i = 16 * (leaf - starts[p]) + t;
                    if i < c.ports[p].bits { msg_col(g, p, i / 8, i % 8) } else { base + unit.useful }
                };
                both(&mut da, &mut db, base + leaf_cols[j] + t, src);
            }
        }
    }
    let pc = port_cols(&unit.in_groups);
    for g in 0..c.g {
        for &(u, j, leaf) in &c.leaf_cuts {
            let p = starts.iter().rposition(|&s| s <= leaf).unwrap();
            let dst = c.slot_col(c.unit_net, g, u) + pc[j].0;
            for t in 0..pc[j].1 {
                let i = 16 * (leaf - starts[p]) + t;
                both(&mut da, &mut db, dst + t, if t < 16 && i < c.ports[p].bits { msg_col(g, p, i / 8, i % 8) } else { zero });
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
    // a typed template, per VU: each input of the instance a copy of its message bit (input w: bit w % 16 of row word
    // w / 16); each part's bound input rows their forms over the unit's columns, in A and B; the root's rows' entries at
    // its parts' columns
    if let Some(t) = &c.typed {
        let cross = t.inst.cross();
        for g in 0..c.g {
            for (w, &u) in t.inst.root.in_cols.iter().enumerate() {
                let (leaf, bit) = (w / 16, w % 16);
                let p = starts.iter().rposition(|&s| s <= leaf).unwrap();
                both(&mut da, &mut db, t.col(c, g, u), msg_col(g, p, 2 * (leaf - starts[p]) + bit / 8, bit % 8));
            }
            for (r, src) in t.inst.bind.iter().flatten() {
                let i = t.col(c, g, *r) as u32;
                for d in [&mut da, &mut db] {
                    d.push((i, i));
                    d.extend(src.iter().map(|&x| (i, t.col(c, g, x) as u32)));
                }
            }
            for (r, xa, xb) in &cross {
                let i = t.col(c, g, *r) as u32;
                da.extend(xa.iter().map(|&x| (i, t.col(c, g, x) as u32)));
                db.extend(xb.iter().map(|&x| (i, t.col(c, g, x) as u32)));
            }
        }
    }
    // each output row's message bits, per VU: copies of the output bits it commits (word k bit t: bit t % 8 of row byte
    // 2k + t / 8), zero past its port's words
    let out_col = c.out_col();
    for g in 0..c.g {
        for (j, port) in c.out_rows().iter().enumerate() {
            let (_, _, at, w) = c.out_group(j);
            for k in 0..port.words {
                for t in 0..16 {
                    let src = if k < w { out_col(g, at + k) + t } else { zero };
                    both(&mut da, &mut db, msg_col(g, c.n_in + j, 2 * k + t / 8, t % 8), src);
                }
            }
        }
    }
    // each pinned output's bits, per VU: copies of the forced-zero row (its gate's B the constant alone, so its value is 0)
    let w0: usize = c.out_words[..c.out_words.len() - c.pinned].iter().sum();
    for g in 0..c.g {
        for o in w0..c.out_words.iter().sum() {
            for t in 0..16 {
                both(&mut da, &mut db, out_col(g, o) + t, zero);
            }
        }
    }
    // a public leaf's bits are no copy but the in region's: every entry of their rows dropped
    if !c.public_inputs.is_empty() {
        da.retain(|e| !c.is_public_col(e.0 as usize));
        db.retain(|e| !c.is_public_col(e.0 as usize));
    }
    (da, db)
}

// --- the instance files ------------------------------------------------------------------------------------------------

/// A `flock-circuit-inputs` file: the prover's has the rows and their salts, the verifier's (`rows: false`) only the public part.
pub struct Instances {
    pub header: Value,
    pub n: usize,
    pub rows: Option<Vec<u8>>,
    /// every row's salt, instance-major, ports in order (the prover's file only)
    pub salts: Option<Vec<u8>>,
    /// every row's `b || c`, instance-major, ports in order (a shared-row file: its tables' rows, port-major)
    pub bcs: Vec<[u8; BC_BYTES]>,
    /// SHA-512 of the header's compact key-sorted JSON (no newline) and every public byte after the rows and salts
    /// (`b || c`, refs): what Σ binds. A drawn statement's header is its draw's, which no file holds, so this is no file's hash.
    pub public_sha512: [u8; 64],
    /// SHA-512 of the verifier's own public file of the registered population (`public_file`'s bytes, which is the staged
    /// `pub-<n>.bin` when its writer writes the header as serde does: `verity_flock.circuit.public_file`). A drawn statement
    /// keeps its population's.
    pub public_file_sha512: [u8; 64],
    /// the registered population's `b || c` (a drawn statement: the frame-v3 roots are over them); `None`: the statement's own
    pub population: Option<(usize, Vec<[u8; BC_BYTES]>)>,
    /// a shared-row file's row tables (header `shared_rows`); `None`: every instance carries its own rows
    pub tables: Option<RowTables>,
    /// every instance's public input words (the file's last bytes, u16 LE), instance-major, the public ports in order
    pub public: Vec<u16>,
}

/// Header `shared_rows: {port: R_p}`: row `p`'s frame-v3 tree is a table of `R_p` committed rows, and instance `i` reads its
/// row `refs[i * ports + p]`. Rows, salts and `b || c` are the tables', port-major. An output row's table is the instances'
/// own rows, `R_p = n` and refs `0 … n − 1`.
#[derive(Clone)]
pub struct RowTables {
    pub counts: Vec<usize>,
    /// every table's rows (u16 LE), port-major (the prover's file only)
    pub rows: Option<std::sync::Arc<Vec<u8>>>,
    /// every table row's salt, port-major (the prover's file only)
    pub salts: Option<std::sync::Arc<Vec<u8>>>,
    pub refs: Vec<u32>,
}

impl RowTables {
    /// port `p`'s first row among the tables' rows
    pub fn start(&self, p: usize) -> usize {
        self.counts[..p].iter().sum()
    }
}

/// The header's `units`: the instances' global unit indices (the partition's order: ascending), and per block its units.
fn units_entry(c: &Composite, indices: &[usize]) -> Value {
    json!({"vus_per_block": c.g, "units_per_vu": c.upv, "indices": indices, "blocks": indices.chunks(c.g).map(|b| b.to_vec()).collect::<Vec<_>>(),
           "classes": vec![crate::hex(&c.class_sha512); indices.len()]})
}

/// Whether `units` is `units_entry(c, indices)`, checked in place (a population's has millions of entries).
fn units_match(c: &Composite, indices: &[usize], units: &Value) -> bool {
    let class = crate::hex(&c.class_sha512);
    let (Some(u), Some(blocks), Some(classes)) = (units.as_object(), units["blocks"].as_array(), units["classes"].as_array()) else {
        return false;
    };
    let block_ok = |b: &Value, ch: &[usize]| b.as_array().is_some_and(|b| b.len() == ch.len() && b.iter().zip(ch).all(|(x, &i)| x.as_u64() == Some(i as u64)));
    u.len() == 5 && units["vus_per_block"] == json!(c.g) && units["units_per_vu"] == json!(c.upv)
        && blocks.len() == indices.len().div_ceil(c.g) && blocks.iter().zip(indices.chunks(c.g)).all(|(b, ch)| block_ok(b, ch))
        && classes.len() == indices.len() && classes.iter().all(|x| x.as_str() == Some(class.as_str()))
}

impl Instances {
    pub fn load(path: &str, c: &Composite) -> Result<Self, String> {
        let b = std::fs::read(path).map_err(|e| format!("instances: {e}"))?;
        let nl = b.iter().position(|&x| x == b'\n').ok_or("no header line")?;
        let mut header: Value = serde_json::from_slice(&b[..nl]).map_err(|e| format!("header: {e}"))?;
        if header["format"] != INST_FORMAT || header["statement"] != c.statement() {
            return Err(format!("not a flock-circuit-inputs file of a {} statement", c.statement()));
        }
        if header["circuit_sha512"].as_str() != Some(crate::hex(&c.sha512).as_str()) {
            return Err("the instance file was staged for another circuit".into());
        }
        let n = header["instances"].as_u64().ok_or("header: instances")? as usize;
        let has_rows = header["rows"].as_bool().ok_or("header: rows")?;
        let rb = c.row_bytes();
        let np = c.ports.len();
        let body = &b[nl + 1..];
        let counts: Option<Vec<usize>> = match header.get("shared_rows") {
            None => None,
            Some(s) => {
                let m = s.as_object().ok_or("header: shared_rows is not an object")?;
                if m.len() != np || c.ports.iter().any(|p| !m.contains_key(&p.name)) {
                    return Err("header: shared_rows does not name exactly the circuit's rows".into());
                }
                let counts: Vec<usize> = c.ports.iter().map(|p| m[&p.name].as_u64().unwrap_or(0) as usize).collect();
                if counts.contains(&0) {
                    return Err("header: a shared_rows table has no row".into());
                }
                if counts[c.n_in..].iter().any(|&r| r != n) {
                    return Err("header: an output row's table is not one row per instance".into());
                }
                Some(counts)
            }
        };
        // a per-instance file is `n` rows per port, instance-major; a shared-row file `R_p` table rows per port, port-major
        let (rows_len, salts_len, n_bcs, refs_len) = match &counts {
            None => (n * rb, n * np * SALT_BYTES, n * np, 0),
            Some(k) => (k.iter().zip(&c.ports).map(|(r, p)| r * 2 * p.words).sum(), k.iter().sum::<usize>() * SALT_BYTES, k.iter().sum(), 4 * n * np),
        };
        let (rows_len, salts_len) = if has_rows { (rows_len, salts_len) } else { (0, 0) };
        if header.get("public_inputs") != c.meta.get("public_inputs") {
            return Err("header: public_inputs is not the circuit's".into());
        }
        let public_len = 2 * n * c.public_words();
        if body.len() != rows_len + salts_len + BC_BYTES * n_bcs + refs_len + public_len {
            return Err("instance file length".into());
        }
        let private = |a: usize, len: usize| has_rows.then(|| body[a..a + len].to_vec());
        let (rows, salts) = (private(0, rows_len), private(rows_len, salts_len));
        let pubs = &body[rows_len + salts_len..];
        let bcs = pubs[..BC_BYTES * n_bcs].chunks(BC_BYTES).map(|x| x.try_into().unwrap()).collect();
        let tail = &pubs[BC_BYTES * n_bcs..BC_BYTES * n_bcs + refs_len];
        let public = pubs[pubs.len() - public_len..].chunks(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect();
        let (rows, salts, tables) = match counts {
            None => (rows, salts, None),
            Some(counts) => {
                let refs: Vec<u32> = tail.chunks(4).map(|x| u32::from_le_bytes(x.try_into().unwrap())).collect();
                if refs.iter().enumerate().any(|(k, &r)| r as usize >= counts[k % np]) {
                    return Err("shared rows: an instance's ref is past its table".into());
                }
                if refs.iter().enumerate().any(|(k, &r)| k % np >= c.n_in && r as usize != k / np) {
                    return Err("shared rows: an instance's output row is not its own".into());
                }
                (None, None, Some(RowTables { counts, rows: rows.map(std::sync::Arc::new), salts: salts.map(std::sync::Arc::new), refs }))
            }
        };
        let indices: Vec<usize> = header["units"]["indices"].as_array().ok_or("header: units.indices")?.iter()
            .map(|x| x.as_u64().map(|v| v as usize).ok_or("header: a unit index")).collect::<Result<_, _>>()?;
        if indices.len() != n || indices.windows(2).any(|w| w[0] >= w[1]) || !units_match(c, &indices, &header["units"]) {
            return Err("header: the instances' unit indices are not ascending, one per instance, blocks of vus_per_block".into());
        }
        if header.get("unit_draw").is_some() {
            return Err("header: a staged file is a registered population; a draw is made in the session".into());
        }
        header["rows"] = Value::Bool(false);
        let public_sha512 = Self::public_digest(&header, b"", pubs);
        let public_file_sha512 = Self::public_digest(&header, b"\n", pubs);
        Ok(Instances { header, n, rows, salts, bcs, public_sha512, public_file_sha512, population: None, tables, public })
    }

    /// SHA-512 of the header's compact key-sorted JSON, `line_end` and the public bytes. serde_json's maps are sorted (no
    /// `preserve_order`), so the header serializes key-sorted as it is, streamed into the hash (a population's header is about
    /// a gigabyte).
    fn public_digest(header: &Value, line_end: &[u8], pubs: &[u8]) -> [u8; 64] {
        struct Hash<'a>(&'a mut Sha512);
        impl std::io::Write for Hash<'_> {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.update(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut h = Sha512::new();
        serde_json::to_writer(Hash(&mut h), header).unwrap();
        h.update(line_end);
        h.update(pubs);
        h.finalize().into()
    }

    /// The statement of the drawn units (`unit_draw` over this registered population's positions `[0, n)`): instance `i` is the
    /// population's `units[i]`; its header is the population's with the draw, `instances = K` and the drawn units' indices; its
    /// public digest covers that header and the population's public bytes (the frame-v3 roots stay the population's).
    pub fn drawn(&self, c: &Composite, draw: &Value) -> Result<Instances, String> {
        let units = crate::unit_draw::check(draw)?;
        if draw["population"].as_u64() != Some(self.n as u64) || self.population.is_some() {
            return Err("U: the draw's population is not this registered population".into());
        }
        let np = c.ports.len();
        let rb = c.row_bytes();
        let indices: Vec<usize> = units.iter().map(|&u| self.header["units"]["indices"][u].as_u64().unwrap_or(0) as usize).collect();
        // the population's header but its units (millions of entries), which the draw's replace
        let mut header: Value = self.header.as_object().map(|m| m.iter().filter(|(k, _)| k.as_str() != "units").map(|(k, v)| (k.clone(), v.clone()))
            .collect::<serde_json::Map<_, _>>().into()).unwrap_or_default();
        header["unit_draw"] = draw.clone();
        header["instances"] = json!(units.len());
        header["population"] = json!(self.n);
        header["units"] = units_entry(c, &indices);
        let mut pubs: Vec<u8> = self.bcs.iter().flatten().copied().collect();
        if let Some(t) = &self.tables {
            pubs.extend(t.refs.iter().flat_map(|r| r.to_le_bytes()));
        }
        pubs.extend(self.public.iter().flat_map(|w| w.to_le_bytes()));
        let public_sha512 = Self::public_digest(&header, b"", &pubs);
        let public_file_sha512 = self.public_file_sha512;
        let population = Some((self.n, self.bcs.clone()));
        let pw = c.public_words();
        let public: Vec<u16> = units.iter().flat_map(|&u| self.public[u * pw..(u + 1) * pw].to_vec()).collect();
        if let Some(t) = &self.tables {
            // the tables stay the population's; the drawn instances keep their refs into them
            let refs = units.iter().flat_map(|&u| t.refs[u * np..(u + 1) * np].to_vec()).collect();
            return Ok(Instances { header, n: units.len(), rows: None, salts: None, bcs: self.bcs.clone(), public_sha512,
                                  public_file_sha512, population, tables: Some(RowTables { refs, ..t.clone() }), public });
        }
        Ok(Instances {
            header, n: units.len(),
            rows: self.rows.as_ref().map(|r| units.iter().flat_map(|&u| r[u * rb..(u + 1) * rb].to_vec()).collect()),
            salts: self.salts.as_ref().map(|s| units.iter().flat_map(|&u| s[u * np * SALT_BYTES..(u + 1) * np * SALT_BYTES].to_vec()).collect()),
            bcs: units.iter().flat_map(|&u| self.bcs[u * np..(u + 1) * np].to_vec()).collect(),
            public_sha512,
            public_file_sha512,
            population,
            tables: None,
            public,
        })
    }

    /// Where instance `i`'s port-`p` `b || c` sits in `bcs` (a shared-row file: its ref into table `p`).
    pub fn bc_index(&self, i: usize, p: usize, np: usize) -> usize {
        match &self.tables {
            Some(t) => t.start(p) + t.refs[i * np + p] as usize,
            None => i * np + p,
        }
    }

    pub fn bc(&self, i: usize, p: usize, np: usize) -> [u8; BC_BYTES] {
        self.bcs[self.bc_index(i, p, np)]
    }

    /// Whether this is a prover's file (it holds rows and salts).
    pub fn has_rows(&self) -> bool {
        self.rows.is_some() || self.tables.as_ref().is_some_and(|t| t.rows.is_some())
    }

    /// The verifier's own public file of this registered population: the header (`rows` false) and the public bytes after
    /// the rows and salts, which this file drops (`verity_flock.circuit.stage(..., public_only=True)`).
    pub fn public_file(&self) -> Result<Vec<u8>, String> {
        if self.population.is_some() {
            return Err("a drawn statement's public file is its population's".into());
        }
        let mut b = serde_json::to_vec(&self.header).map_err(|e| format!("header: {e}"))?;
        b.push(b'\n');
        b.extend(self.bcs.iter().flatten());
        if let Some(t) = &self.tables {
            b.extend(t.refs.iter().flat_map(|r| r.to_le_bytes()));
        }
        b.extend(self.public.iter().flat_map(|w| w.to_le_bytes()));
        Ok(b)
    }

    /// Every instance's input rows, instance-major, ports side by side (a shared-row file's gathered by its refs), then every
    /// instance's public input words (u16 LE, as [`Stmt::unit_inputs`] reads them); `None`: the verifier's file.
    pub fn instance_rows(&self, c: &Composite) -> Option<Vec<u8>> {
        let public = self.public.iter().flat_map(|w| w.to_le_bytes());
        let Some(t) = &self.tables else {
            return self.rows.as_ref().map(|r| { let mut r = r.clone(); r.extend(public); r });
        };
        let rows = t.rows.as_ref()?;
        let np = c.ports.len();
        let base: Vec<usize> = c.ports.iter().zip(&t.counts).scan(0, |a, (p, &r)| { let s = *a; *a += r * 2 * p.words; Some(s) }).collect();
        let mut out = Vec::with_capacity(self.n * (c.row_bytes() + 2 * c.public_words()));
        for i in 0..self.n {
            for (p, port) in c.ports.iter().enumerate() {
                let (w, r) = (2 * port.words, t.refs[i * np + p] as usize);
                out.extend_from_slice(&rows[base[p] + r * w..base[p] + (r + 1) * w]);
            }
        }
        out.extend(public);
        Some(out)
    }

    /// Every instance's row salts, instance-major, ports in order (a shared-row file's gathered by its refs); `None`: the
    /// verifier's file.
    pub fn instance_salts(&self, np: usize) -> Option<Vec<u8>> {
        let Some(t) = &self.tables else { return self.salts.clone() };
        let salts = t.salts.as_ref()?;
        Some((0..self.n * np).flat_map(|k| {
            let s = t.start(k % np) + t.refs[k] as usize;
            salts[s * SALT_BYTES..(s + 1) * SALT_BYTES].iter().copied()
        }).collect())
    }
}

fn sorted(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.iter().map(|(k, x)| (k.clone(), sorted(x))).collect::<std::collections::BTreeMap<_, _>>().into_iter().collect()),
        Value::Array(a) => Value::Array(a.iter().map(sorted).collect()),
        x => x.clone(),
    }
}

fn fv3_hash(tag: &[u8], parts: &[&[u8]]) -> [u8; 64] {
    let mut h = Sha512::new();
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

/// A frame-v3-sha512 leaf: position `i` of the domain, its schema and its payload.
pub fn fv3_leaf(domain_id: &[u8], i: usize, schema: &[u8], payload: &[u8]) -> [u8; 64] {
    fv3_hash(b"leaf", &[domain_id, &uint(i), &uint(i), schema, payload])
}

pub fn fv3_root(domain_id: &[u8], leaves: Vec<[u8; 64]>) -> [u8; 64] {
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

/// hm96-sha512/v1's salt prefix and the default key's leaf prefix (both one block).
#[cfg(feature = "sha512")]
fn hm96_prefixes() -> (Vec<u8>, Vec<u8>) {
    let k = flock_merkle::hm96::consts();
    (k.salt_prefix.to_vec(), k.leaf_prefix.to_vec())
}

#[cfg(not(feature = "sha512"))]
fn hm96_prefixes() -> (Vec<u8>, Vec<u8>) {
    panic!("verity/flock-circuit's rows are hm96-sha512 leaves: build with the sha512 feature")
}

/// `hm96.SHA512.tree_leaf(key, b || c)` for the default key: SHA-512 of the leaf prefix block and `b || c`.
pub fn row_tree_leaf(bc: &[u8; BC_BYTES]) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(hm96_prefixes().1);
    h.update(bc);
    h.finalize().into()
}

/// Port `port`'s row `row` (its words u16 LE) under a zero salt: `b = x` (a zero salt's mask is zero), `c = SHA-512(salt_prefix
/// || 0^192)`; `x` is the port's row digest (`sha512/row/v1` of its words, or a bit row of its length: the row's first bytes).
pub fn row_bc_zero_salt(port: &Port, row: &[u8]) -> [u8; BC_BYTES] {
    let mut m = row[..port.bytes().min(row.len())].to_vec();
    m.resize(port.bytes(), 0);
    let x: [u8; 64] = if port.segs.is_some() {
        row_seg_digest(port.bits, &m)
    } else {
        let mut h = Sha512::new();
        h.update(if port.v2 { row_prefix_v2(port.bits as u64) } else { row_prefix(port.words) });
        h.update(m);
        h.finalize().into()
    };
    let mut h = Sha512::new();
    h.update(hm96_prefixes().0);
    h.update([0u8; SALT_BYTES]);
    let mut bc = [0u8; BC_BYTES];
    bc[..64].copy_from_slice(&x);
    bc[64..].copy_from_slice(&h.finalize());
    bc
}

/// The frame-v3-sha512 roots recomputed from the file's `b || c`, against its header's; the pinned dummy `b || c` against the
/// padding instance's rows' under zero salts (the commitment scheme's own functions; the circuit is never evaluated here).
pub fn check_public(c: &Composite, inst: &Instances) -> Result<(), String> {
    let fv = &inst.header["frame_v3"];
    let get = |k: &str, p: &str| -> Result<Vec<u8>, String> { hexb(fv[k][p].as_str().ok_or(format!("frame_v3.{k}.{p} missing"))?) };
    if fv["frame"] != "frame-v3-sha512" || fv["row_key_sha512"].as_str() != Some(crate::hex(&Sha512::digest(hm96_sha512_default_key())).as_str()) {
        return Err("the instance file's rows are not frame-v3-sha512 hm96 leaves under the default key".into());
    }
    if let Some(d) = inst.header.get("unit_draw") {
        let units = crate::unit_draw::check(d)?;
        if inst.header["instances"].as_u64() != Some(units.len() as u64) || inst.n != units.len() {
            return Err("U3: the statement's instances are not the drawn units".into());
        }
    }
    for (p, port) in c.ports.iter().enumerate() {
        if fv["schemas"][&port.name] != port.schema() {
            return Err(format!("row {}: schema is not {}", port.name, port.schema()));
        }
        if row_bc_zero_salt(port, &c.dummy_row(p)) != c.dummy_bc[p] {
            return Err(format!("the pinned dummy b || c of row {} is not the padding instance's row under a zero salt", port.name));
        }
    }
    for (name, root) in frame_roots(c, inst)? {
        if root.to_vec() != get("roots", &name)? {
            return Err(format!("row {name}: the rows' b || c do not hash to the header's frame-v3-sha512 root"));
        }
    }
    Ok(())
}

pub const LINKS_FORMAT: &str = "verity/flock-links/v1";

/// Linking (PROTOCOL.md §16.13) on the registered population's public file: the verifier's own links file (`--links`,
/// `verity/flock-links/v1`, derived from its program and query: `verity_flock.class_statement.links_file`) is for this
/// circuit, its partition and META `links`' ports, and every `(reader, port, writer)` it lists has the reader's link row's
/// `b || c` equal to the writer's output row's. A circuit with META `links` needs the file; one without takes none. The links
/// checked. This binds a public topology; it does not hide the topology (the links file is the verifier's view of the circuit).
pub fn check_links(c: &Composite, inst: &Instances, file: Option<&Value>) -> Result<usize, String> {
    let ports: Vec<String> = match c.meta.get("links") {
        None => {
            return match file {
                Some(_) => Err("--links: the circuit declares no links (META links)".into()),
                None => Ok(0),
            };
        }
        Some(ls) => ls.as_array().ok_or("META links is not a list")?.iter()
            .map(|x| x["port"].as_str().map(String::from).ok_or("META links: a link names no port".to_string())).collect::<Result<_, _>>()?,
    };
    let Some(f) = file else {
        return Err("the circuit declares links (META links): pass the verifier's own links file (--links F)".into());
    };
    if f["format"] != LINKS_FORMAT {
        return Err(format!("--links: not a {LINKS_FORMAT} file"));
    }
    if f["circuit_sha512"].as_str() != Some(crate::hex(&c.sha512).as_str()) {
        return Err("--links: derived for another circuit".into());
    }
    if f["partition"] != c.meta["partition"]["digest"] {
        return Err("--links: derived for another partition than the circuit's".into());
    }
    if f["ports"] != json!(ports) {
        return Err("--links: its ports are not the circuit's link rows (META links)".into());
    }
    let np = c.ports.len();
    if c.ports.len() != c.n_in + 1 {
        return Err("META links: a statement with links has one output row".into());
    }
    if inst.population.is_some() {
        return Err("links are checked on the registered population's file, before a draw".into());
    }
    let (n, out) = (inst.n, np - 1);
    let links = f["links"].as_array().ok_or("--links: links is not a list")?;
    for x in links {
        let bad = || format!("--links: {x} is not [reader, port, writer] over the file's {n} instances and the circuit's link rows");
        let a = x.as_array().filter(|a| a.len() == 3).ok_or_else(bad)?;
        let (Some(r), Some(name), Some(w)) = (a[0].as_u64(), a[1].as_str(), a[2].as_u64()) else { return Err(bad()) };
        let (r, w) = (r as usize, w as usize);
        let Some(p) = c.ports[..c.n_in].iter().position(|q| q.name == name).filter(|_| ports.iter().any(|q| q == name)) else {
            return Err(bad());
        };
        if r >= n || w >= n {
            return Err(bad());
        }
        if inst.bc(r, p, np) != inst.bc(w, out, np) {
            return Err(format!("cut wire from instance {w}'s output row to instance {r}'s row {name}: the reader's row is not the writer's committed output row (their b || c differ)"));
        }
    }
    Ok(links.len())
}

/// The frame-v3-sha512 roots of the file's public part, one row tree per row (input rows, then output rows): over the
/// registered population (a drawn statement's instances are some of its units), a shared-row file's row trees over its tables.
pub fn frame_roots(c: &Composite, inst: &Instances) -> Result<Vec<(String, [u8; 64])>, String> {
    let fv = &inst.header["frame_v3"];
    let dom_of = |k: &str| -> Result<Vec<u8>, String> { hexb(fv["domain_ids"][k].as_str().ok_or(format!("frame_v3.domain_ids.{k} missing"))?) };
    let np = c.ports.len();
    let (n, bcs) = match &inst.population { Some((n, b)) => (*n, b), None => (inst.n, &inst.bcs) };
    let mut out = Vec::new();
    for (p, port) in c.ports.iter().enumerate() {
        let dom = dom_of(&port.name)?;
        let (count, row): (usize, Box<dyn Fn(usize) -> usize + Sync>) = match &inst.tables {
            Some(t) => (t.counts[p], Box::new(move |r| t.start(p) + r)),
            None => (n, Box::new(move |i| i * np + p)),
        };
        let leaves = (0..count).into_par_iter().map(|r| fv3_leaf(&dom, r, port.schema().as_bytes(), &row_tree_leaf(&bcs[row(r)]))).collect();
        out.push((port.name.clone(), fv3_root(&dom, leaves)));
    }
    Ok(out)
}

// --- the statement -------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Region {
    /// row `p`'s `b || c` of each VU (its hm96 slot's output)
    Digest(usize),
    /// the public input words of every unit slot's leaf group ([`Composite::in_regions`]), `In(w, s)`: `2^s` words from
    /// word `w`
    In(usize, usize),
}

pub struct RegionShape {
    pub region: Region,
    pub free: Vec<usize>,
    pub mask: u32,
    pub fixed: u32,
}

/// A generated read's table-direct B side (`Lookup.foldB`): product row `prod0 + j nh + h` (`j < k`) reads low minterm
/// `lo_top + l` exactly when bit `j` of word `h 2^lo + l` is set. Its slot type's own B leaves those rows empty.
#[derive(Clone)]
pub struct TableSide {
    pub table: &'static [u32],
    pub lo: usize,
    pub nh: usize,
    pub k: usize,
    pub prod0: usize,
    pub lo_top: usize,
}

impl TableSide {
    pub fn of(ln: &LookupNet) -> Self {
        TableSide { table: ln.table, lo: ln.lo, nh: 1 << (ln.n - ln.lo), k: ln.k, prod0: ln.prod0(), lo_top: ln.lo_top() }
    }

    /// Adds `Bᵀ eq` over the product rows to a slot's fold `out`.
    pub fn fold(&self, eq: &[F128], out: &mut [F128]) {
        let (nlo, mask) = (1usize << self.lo, if self.k >= 32 { u32::MAX } else { (1u32 << self.k) - 1 });
        out[self.lo_top..self.lo_top + nlo].par_iter_mut().enumerate().for_each(|(l, o)| {
            for h in 0..self.nh {
                let mut x = self.table[h * nlo + l] & mask;
                while x != 0 {
                    *o += eq[self.prod0 + x.trailing_zeros() as usize * self.nh + h];
                    x &= x - 1;
                }
            }
        });
    }
}

/// Every column that shares a 128-bit word with a partial region's columns without being one of them, as (region, column): a
/// region is partial when some in-word bit (0 to 6) is not free; its covered words are those whose bits above 6 agree with its
/// fixed bits.
pub fn region_shared_columns(regions: &[RegionShape]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (ri, r) in regions.iter().enumerate().filter(|(_, r)| !(0..7).all(|b| r.free.contains(&b))) {
        let high: Vec<usize> = r.free.iter().copied().filter(|&b| b >= 7).collect();
        for a in 0..1usize << high.len() {
            let w = (r.fixed as usize & !0x7f) | high.iter().enumerate().fold(0, |x, (i, &b)| x | (((a >> i) & 1) << b));
            out.extend((w..w + 128).filter(|&q| q as u32 & r.mask != r.fixed).map(|q| (ri, q)));
        }
    }
    out
}

/// H_reg, the region-word check (docs/region-word-check.md, the red team's C3 on the public-circuit ZK proof): a region claim's
/// ring-switch message carries all 128 bits of every word the region covers, in the clear, so a column sharing a word with a
/// region's columns must be one of them or be forced to zero, its rows in the combined A and B both empty (`0·0 = z_q`). The
/// combined row q is the row of the slot type placed at q, in block columns, plus every Δ entry `(q, col)`, summed over GF(2)
/// (an entry that appears twice cancels). The first offending column, as the refusal.
pub fn region_word_violation(c: &Composite, regions: &[RegionShape], da: &[(u32, u32)], db: &[(u32, u32)]) -> Option<String> {
    let by_row = |d: &[(u32, u32)]| {
        let mut m: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &(q, col) in d {
            m.entry(q as usize).or_default().push(col as usize);
        }
        m
    };
    let (ra, rb) = (by_row(da), by_row(db));
    let empty = |q: usize, delta: &BTreeMap<usize, Vec<usize>>, a: bool| -> bool {
        let mut cols = delta.get(&q).cloned().unwrap_or_default();
        if let Some(r) = c.ranges.iter().find(|r| q >= r.start() && q < r.start() + (r.count << r.slot_log)) {
            let slot0 = q & !((1usize << r.slot_log) - 1);
            match r.kind {
                Kind::Net(i) => {
                    let net = &c.nets[i].1;
                    cols.extend((if a { &net.a } else { &net.b }).get(q - slot0).into_iter().flatten().map(|&k| slot0 + k));
                }
                Kind::Mask => cols.push(q),
            }
        }
        cols.sort_unstable();
        cols.chunk_by(|x, y| x == y).all(|run| run.len() % 2 == 0)
    };
    region_shared_columns(regions).into_iter().find(|&(_, q)| !(empty(q, &ra, true) && empty(q, &rb, false)))
        .map(|(ri, q)| format!("region {:?}: column {q} shares a word with the region's bits and is not forced to zero", regions[ri].region))
}

pub struct BlockCircuit {
    pub k_log: usize,
    pub pin: usize,
    /// per slot type: its log size and its circuit, column-major: the fold gathers each column's rows, where a row scatter
    /// would give each of its ~4 chunks per thread an accumulator as wide as the slot (2^21 to 2^24 columns for a GEMM unit)
    pub types: Vec<(usize, CscCircuit)>,
    /// per slot type: a generated read's table-direct B side
    pub tables: Vec<Option<TableSide>>,
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
        let bases: Vec<Vec<F128>> = self.types.iter().zip(&self.tables).map(|((sl, m), tb)| {
            let mut base = m.fold_alpha_batched(alpha, &eq[..1 << sl]);
            if let Some(tb) = tb {
                tb.fold(&eq[..1 << sl], &mut base);
            }
            base
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

/// A dishonest prover's row witness (VU 0 of block 0): a compression's chaining value, the first chaining value (not the
/// midstate), the last block, a salt, a bit row's first fill bit, or its padding's 0x80 bit in a block it shares with the
/// row's bytes, each followed consistently downstream.
#[derive(Clone, Copy, Debug, Default)]
pub struct RowTamper {
    pub cv_flip: bool,
    pub midstate_flip: bool,
    pub pad_flip: bool,
    pub salt_flip: bool,
    pub fill_flip: bool,
    pub mixed_pad_flip: bool,
}

/// Block `b`'s row slots' input words (`Stmt::row_inputs`), as a witness source hands them over.
pub type RowIns<'a> = &'a (dyn Fn(usize) -> (Vec<Vec<F128>>, Vec<Vec<F128>>) + Sync);

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
    /// SHA-512 over the statement (bound through Σ)
    pub digest: [u8; 64],
    /// its first 32 bytes, for Flock's own 32-byte statement slot (the R1CS digest its transcript absorbs)
    pub flock_digest: [u8; 32],
    /// how the verifier's identity is pinned: profile, reps, zk, leaf scheme, coin mode (backend identity)
    pub identity: Value,
    /// `r1cs.c0_is_identity()`, a function of the statement alone: computed by its first verification, read by the rest
    pub c0_identity: OnceLock<bool>,
}

/// The strongly connected components of the per-VU slot graph (`(net, slot)`, an edge from each wire's source slot to its
/// destination slot, the rows' circuits left out), sources first, each as its levels.
fn slot_components(c: &Composite, wires: &[&Wire]) -> Vec<Vec<Vec<(usize, usize)>>> {
    let nodes: Vec<(usize, usize)> = (0..c.nets.len()).filter(|&n| n != c.sha_net && n != c.hm_net)
        .flat_map(|n| (0..c.per_vu[n]).map(move |s| (n, s))).collect();
    let id: BTreeMap<(usize, usize), usize> = nodes.iter().enumerate().map(|(i, &x)| (x, i)).collect();
    let mut adj = vec![Vec::new(); nodes.len()];
    for w in wires {
        if let (Some(&a), Some(&b)) = (id.get(&(w.src.0, w.src.1)), id.get(&(w.dst.0, w.dst.1))) {
            adj[a].push(b);
        }
    }
    // Tarjan, iterative: components come out sinks first
    let (n, mut index, mut next) = (nodes.len(), vec![usize::MAX; nodes.len()], 0);
    let (mut low, mut on, mut stack, mut comps) = (vec![0; n], vec![false; n], Vec::new(), Vec::new());
    for root in 0..n {
        if index[root] != usize::MAX {
            continue;
        }
        let mut call: Vec<(usize, usize)> = vec![(root, 0)];
        while let Some(&mut (v, ref mut e)) = call.last_mut() {
            if *e == 0 && index[v] == usize::MAX {
                (index[v], low[v]) = (next, next);
                next += 1;
                stack.push(v);
                on[v] = true;
            }
            if let Some(&u) = adj[v].get(*e) {
                *e += 1;
                if index[u] == usize::MAX {
                    call.push((u, 0));
                } else if on[u] {
                    low[v] = low[v].min(index[u]);
                }
                continue;
            }
            call.pop();
            if let Some(&(p, _)) = call.last() {
                low[p] = low[p].min(low[v]);
            }
            if low[v] == index[v] {
                let mut comp = Vec::new();
                while let Some(x) = stack.pop() {
                    on[x] = false;
                    comp.push(x);
                    if x == v {
                        break;
                    }
                }
                comps.push(comp);
            }
        }
    }
    comps.reverse();
    // inside a component, its slots in dependency order, by level: the reverse postorder of a depth-first walk over its own
    // edges, a slot's level one past its forward predecessors' (a cycle's closing wires, backward in that order, are left for a
    // second sweep); a level's slots have no edge between them and are evaluated together
    comps.into_iter().map(|mut comp| {
        comp.sort_unstable();
        let inside: std::collections::BTreeSet<usize> = comp.iter().copied().collect();
        let (mut seen, mut post) = (std::collections::BTreeSet::new(), Vec::with_capacity(comp.len()));
        for &r in &comp {
            if !seen.insert(r) {
                continue;
            }
            let mut walk: Vec<(usize, usize)> = vec![(r, 0)];
            while let Some(&mut (v, ref mut e)) = walk.last_mut() {
                if let Some(&u) = adj[v].get(*e) {
                    *e += 1;
                    if inside.contains(&u) && seen.insert(u) {
                        walk.push((u, 0));
                    }
                    continue;
                }
                post.push(v);
                walk.pop();
            }
        }
        post.reverse();
        let pos: BTreeMap<usize, usize> = post.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let mut level: BTreeMap<usize, usize> = BTreeMap::new();
        for &v in &post {
            let l = *level.entry(v).or_insert(0);
            for &u in &adj[v] {
                if pos.get(&u).is_some_and(|&p| p > pos[&v]) {
                    let e = level.entry(u).or_insert(0);
                    *e = (*e).max(l + 1);
                }
            }
        }
        let mut levels: Vec<Vec<(usize, usize)>> = vec![Vec::new(); level.values().max().map_or(0, |m| m + 1)];
        for &v in &post {
            levels[level[&v]].push(nodes[v]);
        }
        levels
    }).collect()
}

fn bit(w: &F128, t: usize) -> u64 {
    (if t < 64 { w.lo >> t } else { w.hi >> (t - 64) }) & 1
}

fn set_bit(w: &mut F128, t: usize) {
    if t < 64 { w.lo |= 1 << t } else { w.hi |= 1 << (t - 64) }
}

/// A 64 × 64 bit transpose: afterwards bit i of `m[j]` is what bit j of `m[i]` was.
pub(crate) fn transpose64(m: &mut [u64; 64]) {
    let (mut j, mut mask) = (32, 0x0000_0000_ffff_ffffu64);
    while j != 0 {
        for k in (0..64).step_by(2 * j) {
            for i in k..k + j {
                let t = ((m[i] >> j) ^ m[i + j]) & mask;
                m[i] ^= t << j;
                m[i + j] ^= t;
            }
        }
        j >>= 1;
        mask ^= mask << j;
    }
}

/// Bit-sliced rows (`v[row]` bit l is lane l's bit) as the first `n` lanes' packed words, `uw` each.
pub(crate) fn lanes(v: &[u64], n: usize, uw: usize) -> Vec<Vec<F128>> {
    let mut out = vec![vec![F128::ZERO; uw]; n];
    for (blk, rows) in v.chunks(64).enumerate() {
        let mut t = [0u64; 64];
        t[..rows.len()].copy_from_slice(rows);
        transpose64(&mut t);
        for (o, &x) in out.iter_mut().zip(&t) {
            if blk % 2 == 0 { o[blk / 2].lo = x } else { o[blk / 2].hi = x }
        }
    }
    out
}

/// [`lanes`] as u64 words (each F128 word's lo, then hi) into `dst`: lane l's `len` words at `dst[l * stride + off..]`, zero
/// past `v`'s rows. Eight blocks are transposed at a time, so each lane's words land a cache line at a time.
fn lanes_into(v: &[u64], n: usize, dst: &mut [u64], stride: usize, off: usize, len: usize) {
    let blocks = v.len().div_ceil(64);
    assert!(n <= 64 && blocks <= len && (n == 0 || (n - 1) * stride + off + len <= dst.len()), "lanes_into: out of range");
    for (c, rows) in v.chunks(512).enumerate() {
        let mut t = [[0u64; 64]; 8];
        let k = rows.len().div_ceil(64);
        for (tb, r) in t.iter_mut().zip(rows.chunks(64)) {
            tb[..r.len()].copy_from_slice(r);
            transpose64(tb);
        }
        for l in 0..n {
            let at = l * stride + off + 8 * c;
            for (d, tb) in dst[at..at + k].iter_mut().zip(&t) {
                *d = tb[l];
            }
        }
    }
    for l in 0..n {
        dst[l * stride + off + blocks..l * stride + off + len].fill(0);
    }
}

/// Circuit `ni`'s slots (`net`, not a lookup's) from their input words `ins`, as mapped host slots hold them (`HostSlots`
/// with a, b per slot): slot x's a words then its b words, `uw` F128 words each as (lo, hi) u64 pairs, at `out[4 * uw * x..]`,
/// written straight from each lane group's bit-sliced rows. Whether every slot's z is a & b word for word, which that layout
/// requires (if not, `out` holds no layout).
pub fn unit_ab_into(net: &IrUnitNet, ni: usize, uw: usize, ins: &[Vec<F128>], out: &mut [u64]) -> bool {
    assert_eq!(out.len(), 4 * uw * ins.len(), "unit_ab_into: out is not 4 * uw words per slot");
    let t0 = Instant::now();
    let one_pass = unit_flat() && net.one_pass();
    let flat_s = t0.elapsed().as_secs_f64();
    let prof = GroupProf::default();
    let ok = out.par_chunks_mut(64 * 4 * uw).zip(ins.par_chunks(64)).enumerate().all(|(gi, (dst, grp))| {
        let (zz, aa, bq) = prof.eval(net, ni, gi, grp);
        let tw = Instant::now();
        let used = u64::MAX >> (64 - grp.len());
        let ok = zz.iter().zip(&aa).zip(&bq).all(|((z, a), b)| (z ^ (a & b)) & used == 0);
        if ok {
            lanes_into(&aa, grp.len(), dst, 4 * uw, 0, 2 * uw);
            lanes_into(&bq, grp.len(), dst, 4 * uw, 2 * uw, 2 * uw);
        }
        prof.out(tw);
        ok
    });
    prof.print("slot_ab", ni, net.useful, ins.len(), one_pass, flat_s, t0);
    ok
}

/// A lane group's evaluation for [`Stmt::slot_zab`] and [`unit_ab_into`] (`eval64_flat`; with `FC_UNIT_CHECK=1` eval64 beside
/// it, asserted equal), and the times `FC_UNIT_PROFILE` reports.
#[derive(Default)]
struct GroupProf {
    eval_ns: AtomicU64,
    eval_max: AtomicU64,
    ref_ns: AtomicU64,
    ref_max: AtomicU64,
    out_ns: AtomicU64,
    check: OnceLock<bool>,
}

impl GroupProf {
    fn eval(&self, net: &IrUnitNet, ni: usize, gi: usize, grp: &[Vec<F128>]) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let te = Instant::now();
        let lanes_in = |col: usize| grp.iter().enumerate().fold(0u64, |x, (l, r)| x | (bit(&r[col / WORD], col % WORD) << l));
        let got = if unit_flat() {
            let cols = bit_rows(grp, net.in_words);
            net.eval64_flat(|col| cols[col])
        } else {
            net.eval64(lanes_in)
        };
        let e = te.elapsed().as_nanos() as u64;
        self.eval_ns.fetch_add(e, Ordering::Relaxed);
        self.eval_max.fetch_max(e, Ordering::Relaxed);
        if *self.check.get_or_init(|| std::env::var("FC_UNIT_CHECK").as_deref() == Ok("1")) {
            let tr = Instant::now();
            let want = net.eval64(lanes_in);
            let r = tr.elapsed().as_nanos() as u64;
            self.ref_ns.fetch_add(r, Ordering::Relaxed);
            self.ref_max.fetch_max(r, Ordering::Relaxed);
            assert!(want == got, "circuit {ni}, group {gi}: the one-pass evaluation is not eval64's");
        }
        got
    }

    fn out(&self, since: Instant) {
        self.out_ns.fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// `out_sum_s`: the lanes' packed words (`lanes`, or `unit_ab_into`'s writes)
    #[allow(clippy::too_many_arguments)]
    fn print(&self, part: &str, ni: usize, rows: usize, slots: usize, one_pass: bool, flat_s: f64, t0: Instant) {
        if unit_profile() {
            let s = |x: &AtomicU64| x.load(Ordering::Relaxed) as f64 * 1e-9;
            eprintln!("UNITPROF\t{}", serde_json::json!({"part": part, "circuit": ni, "rows": rows, "slots": slots,
                "one_pass": one_pass, "flat_s": flat_s, "groups": slots.div_ceil(64), "wall_s": t0.elapsed().as_secs_f64(),
                "eval_sum_s": s(&self.eval_ns), "eval_max_s": s(&self.eval_max), "out_sum_s": s(&self.out_ns),
                "check": self.check.get().copied().unwrap_or(false), "eval64_sum_s": s(&self.ref_ns), "eval64_max_s": s(&self.ref_max)}));
        }
    }
}

/// Up to 64 lanes' first `words` words as bit-sliced rows (`lanes`' inverse): bit l of row c is bit c of lane l.
fn bit_rows(grp: &[Vec<F128>], words: usize) -> Vec<u64> {
    let mut out = vec![0u64; words * WORD];
    for (w, rows) in out.chunks_exact_mut(WORD).enumerate() {
        for (h, half) in rows.chunks_exact_mut(64).enumerate() {
            let mut t = [0u64; 64];
            for (x, r) in t.iter_mut().zip(grp) {
                *x = r.get(w).map_or(0, |v| if h == 0 { v.lo } else { v.hi });
            }
            transpose64(&mut t);
            half.copy_from_slice(&t);
        }
    }
    out
}

/// The device prover's most nonzeros in one matrix of a slot type: its CSC offsets and counts are 32-bit
/// (`gpu_circuit::CircuitParams`, `prove_circuit.cuh`). The CPU prover's are 64-bit.
pub const GPU_NNZ_MAX: u64 = u32::MAX as u64;

/// Refuses a circuit whose A or B side holds more than `limit` nonzeros.
pub fn nnz_within(name: &str, a: &[Vec<usize>], b: &[Vec<usize>], limit: u64) -> Result<(), String> {
    let count = |m: &[Vec<usize>]| m.iter().map(|r| r.len() as u64).sum::<u64>();
    let (na, nb) = (count(a), count(b));
    if na.max(nb) > limit {
        return Err(format!("{name}: {na} nonzeros in A and {nb} in B, past the GPU prover's {limit} (32-bit offsets): the CPU prover"));
    }
    Ok(())
}

impl Composite {
    /// Every circuit within the device prover's nonzero bound ([`GPU_NNZ_MAX`]).
    pub fn gpu_nnz_ok(&self) -> Result<(), String> {
        self.nets.iter().try_for_each(|(name, net)| nnz_within(name, &net.a, &net.b, GPU_NNZ_MAX))
    }
}

/// Whether host unit slots are evaluated by [`IrUnitNet::eval64_flat`]; `FC_UNIT_FLAT=0`: by `eval64`, the path before it.
pub fn unit_flat() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FC_UNIT_FLAT").as_deref() != Ok("0"))
}

/// `FC_UNIT_PROFILE=1`: the host witness's parts as `UNITPROF` lines on stderr.
pub fn unit_profile() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FC_UNIT_PROFILE").as_deref() == Ok("1"))
}

fn flip_bit(w: &mut F128, t: usize) {
    if t < 64 { w.lo ^= 1 << t } else { w.hi ^= 1 << (t - 64) }
}

impl Stmt {
    pub fn new(c: Composite, n: usize, identity: Value) -> Result<Self, String> {
        let k_log = c.k_log;
        let blocks_real = n.div_ceil(c.g);
        let nbl = log2ceil(blocks_real.max(1)).max(3);
        let m = k_log + nbl;
        shape_in_range(m, PT_LOCAL + nbl)?;
        let pin = pin_of(&c);
        let (da, db) = delta(&c, pin);
        let mut types: Vec<(usize, CscCircuit)> = Vec::new();
        let mut tables: Vec<Option<TableSide>> = Vec::new();
        let mut type_of: BTreeMap<String, usize> = BTreeMap::new();
        let mut ranges = Vec::new();
        // a typed template's generated read layouts, by circuit: their slot types' table-direct B side
        let reads: BTreeMap<usize, TableSide> = c.typed.as_ref().map(|t| t.parts.iter().zip(&t.inst.parts)
            .filter_map(|(&(net, _), &(j, _))| t.inst.reads.get(&j).map(|ln| (net, TableSide::of(ln)))).collect()).unwrap_or_default();
        for r in &c.ranges {
            let key = format!("{:?}", r.kind);
            let t = *type_of.entry(key).or_insert_with(|| {
                let circ = match r.kind {
                    Kind::Net(i) => {
                        let net = &c.nets[i].1;
                        let k = 1usize << net.unit_log;
                        let pad = |rows: &Vec<Vec<usize>>| { let mut x = rows.clone(); x.resize(k, Vec::new()); SparseBinaryMatrix { num_rows: k, num_cols: k, rows: x.into() } };
                        CscCircuit::from_matrices(&pad(&net.a), &pad(&net.b))
                    }
                    Kind::Mask => {
                        let k = 1usize << r.slot_log;
                        let id = SparseBinaryMatrix { num_rows: k, num_cols: k, rows: (0..k).map(|i| vec![i]).collect::<Vec<_>>().into() };
                        CscCircuit::from_matrices(&id, &id)
                    }
                };
                types.push((r.slot_log, circ));
                tables.push(match r.kind { Kind::Net(i) => reads.get(&i).cloned(), Kind::Mask => None });
                types.len() - 1
            });
            ranges.push((t, r.pos0, r.count));
        }
        let regions = Self::regions(&c)?;
        if let Some(e) = region_word_violation(&c, &regions, &da, &db) {
            return Err(e);
        }
        let mut h = Sha512::new();
        h.update(c.tags().statement.as_bytes());
        h.update(c.sha512);
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
        let digest: [u8; 64] = h.finalize().into();
        let flock_digest: [u8; 32] = digest[..32].try_into().unwrap();
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
            digest_cache: OnceLock::from(flock_digest),
            csc_cache: OnceLock::new(),
        };
        let circuit = BlockCircuit { k_log, pin, types, tables, ranges, da, db };
        Ok(Stmt { c, n, blocks_real, nbl, m, m_pts: PT_LOCAL + nbl, pin, regions, circuit, r1cs, digest, flock_digest, identity,
                  c0_identity: OnceLock::new() })
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
        let hr = c.range_of(c.hm_net);
        let (hm, lp, lg) = (&c.nets[c.hm_net].1, c.per_vu[c.hm_net].trailing_zeros() as usize, c.g.trailing_zeros() as usize);
        let bc0 = port_cols(&hm.out_groups)[0].0;
        for p in 0..c.ports.len() {
            out.push(mk(Region::Digest(p), [bits(0, 10), bits(hr.slot_log + lp, lg)].concat(), hr.start() + (p << hr.slot_log) + bc0)?);
        }
        let ur = c.unit_range();
        for (w, s) in c.in_regions() {
            out.push(mk(Region::In(w, s), [bits(0, 7 + s), bits(ur.slot_log, log2ceil(ur.count))].concat(), ur.start() + (w << 7))?);
        }
        Ok(out)
    }

    pub fn blocks(&self) -> usize {
        1 << self.nbl
    }

    fn inst_of(&self, b: usize, g: usize) -> Option<usize> {
        let i = b * self.c.g + g;
        (i < self.n).then_some(i)
    }

    /// Block `b`'s row slots' input words (`sha512x3` slots, then hm96 slots, each VU's in order), computed natively from the rows
    /// and salts (zeros for a padding instance): each compression's chaining value and message block, each hm96 slot's last
    /// chaining value, padding block and salt. `t` tampers VU 0 of block 0, the chain continuing from the tampered value.
    pub fn row_inputs(&self, rows: Option<&[u8]>, salts: Option<&[u8]>, b: usize, t: RowTamper) -> (Vec<Vec<F128>>, Vec<Vec<F128>>) {
        let c = &self.c;
        let (sha, hm) = (&c.nets[c.sha_net].1, &c.nets[c.hm_net].1);
        let (sin, hin) = (port_cols(&sha.in_groups), port_cols(&hm.in_groups));
        let (n_sha, where_) = row_slots(&c.ports);
        let (np, per_hm, rb) = (c.ports.len(), c.per_vu[c.hm_net], c.row_bytes());
        let mut si = vec![vec![F128::ZERO; sha.in_words]; c.g * n_sha];
        let mut hi = vec![vec![F128::ZERO; hm.in_words]; c.g * per_hm];
        let put = |w: &mut [F128], col: usize, words: &[u64]| {
            for (i, &x) in words.iter().enumerate() {
                let at = col + 64 * i;
                debug_assert_eq!(at % 64, 0);
                if at % WORD == 0 { w[at / WORD].lo = x } else { w[at / WORD].hi = x }
            }
        };
        let chained = c.ports.iter().position(|p| p.comps > 1);
        // a bit row's first message bit past its length (its last byte's fill bit), and its padding's first bit in a block with
        // its bytes (bit 7 of the 0x80 byte)
        let filled = c.ports.iter().position(|p| p.v2 && p.bits % 8 != 0);
        let mixed = c.ports.iter().position(|p| p.v2 && p.bytes() % 128 != 0);
        for g in 0..c.g {
            let inst = self.inst_of(b, g);
            let first = b == 0 && g == 0;
            let mut off = 0;
            for (p, port) in c.ports.iter().enumerate() {
                let n = 2 * port.words;
                let row: Vec<u8> = match (rows, inst) { (Some(r), Some(i)) => r[i * rb + off..i * rb + off + n].to_vec(), _ => c.dummy_row(p) };
                off += n;
                let (cv, mut pad) = if let Some(segs) = &port.segs {
                    // each segment's chain from its midstate, then the outer message's over their digests
                    let rc = &c.rows[p];
                    let mut row = row[..port.bytes().min(row.len())].to_vec();
                    row.resize(port.bytes(), 0);
                    let (mut k, mut digests) = (0, Vec::new());
                    let mut chain = |cv: &mut [u64; 8], msg: &[u8], blocks: usize, k: &mut usize| {
                        for blk in msg.chunks(128).take(blocks) {
                            if t.cv_flip && first && *k == 1 && chained == Some(p) {
                                cv[0] ^= 1;
                            }
                            let (slot, sub) = where_[p][*k];
                            let w = &mut si[g * n_sha + slot];
                            put(w, sin[2 * sub].0, &cv[..]);
                            put(w, sin[2 * sub + 1].0, &words_be::<16>(blk));
                            compress(cv, blk);
                            *k += 1;
                        }
                    };
                    for (j, &(m, nb)) in segs.iter().enumerate() {
                        let at = j * (ROW_SEG_BITS / 8);
                        let msg = [&row[at..at + m.div_ceil(8)], &rc.segs[j].tail[..]].concat();
                        let mut cv = rc.segs[j].midstate;
                        if t.midstate_flip && first && p == 0 && j == 0 {
                            cv[0] ^= 1;
                        }
                        chain(&mut cv, &msg, nb, &mut k);
                        digests.extend(cv.iter().flat_map(|x| x.to_be_bytes()));
                    }
                    let msg = [digests, rc.tail.clone()].concat();
                    let mut cv = rc.midstate;
                    chain(&mut cv, &msg, msg.len() / 128 - 1, &mut k);
                    (cv, words_be::<16>(&msg[msg.len() - 128..]))
                } else {
                let mut msg = c.rows[p].message(&row, port.bytes());
                if t.fill_flip && first && filled == Some(p) {
                    msg[port.bits / 8] ^= 1 << (port.bits % 8);
                }
                if t.mixed_pad_flip && first && mixed == Some(p) {
                    msg[port.bytes()] ^= 0x80;
                }
                let mut cv = c.rows[p].midstate;
                if t.midstate_flip && first && p == 0 {
                    cv[0] ^= 1;
                }
                for k in 0..port.comps {
                    if t.cv_flip && first && k == 1 && chained == Some(p) {
                        cv[0] ^= 1;
                    }
                    let (slot, sub) = where_[p][k];
                    let blk = &msg[128 * k..128 * k + 128];
                    let w = &mut si[g * n_sha + slot];
                    put(w, sin[2 * sub].0, &cv);
                    put(w, sin[2 * sub + 1].0, &words_be::<16>(blk));
                    compress(&mut cv, blk);
                }
                (cv, words_be::<16>(&msg[128 * port.comps..]))
                };
                if t.pad_flip && first && p == 0 {
                    pad[0] ^= 1;
                }
                let mut salt = [0u8; SALT_BYTES];
                if let (Some(s), Some(i)) = (salts, inst) {
                    salt.copy_from_slice(&s[(i * np + p) * SALT_BYTES..(i * np + p + 1) * SALT_BYTES]);
                }
                if t.salt_flip && first && p == 0 {
                    salt[0] ^= 1;
                }
                let h = &mut hi[g * per_hm + p];
                put(h, hin[0].0, &cv);
                put(h, hin[1].0, &pad);
                put(h, hin[2].0, &(0..SALT_BYTES / 8).map(|i| u64::from_le_bytes(salt[8 * i..8 * i + 8].try_into().unwrap())).collect::<Vec<_>>());
            }
        }
        (si, hi)
    }

    /// Unit slot `k` of block `b`'s input words (its leaf ports, from the rows; zeros for a padding instance or a zero leaf).
    /// A typed template's: its instance's input bits, the rows' words in order (16 bits each).
    pub fn unit_inputs(&self, rows: Option<&[u8]>, b: usize, k: usize) -> Vec<F128> {
        let c = &self.c;
        if c.typed.is_some() {
            let words = c.in_words();
            let mut w = vec![F128::ZERO; (16 * words).div_ceil(WORD)];
            let (Some(rows), Some(i)) = (rows, self.inst_of(b, k)) else { return w };
            let rb = c.row_bytes();
            for leaf in 0..words {
                let v = u16::from_le_bytes([rows[i * rb + 2 * leaf], rows[i * rb + 2 * leaf + 1]]) as u64;
                let col = 16 * leaf;
                if col % WORD < 64 { w[col / WORD].lo |= v << (col % WORD) } else { w[col / WORD].hi |= v << (col % WORD - 64) }
            }
            return w;
        }
        let unit = c.unit();
        let (g, u) = (k / c.upv, k % c.upv);
        let mut w = vec![F128::ZERO; unit.in_words];
        let (Some(rows), Some(i)) = (rows, self.inst_of(b, g)) else { return w };
        let rb = c.row_bytes();
        let mut col = unit.in_groups[0].col * WORD;
        // a public leaf: instance i's public word q, past every instance's rows (`Instances::instance_rows`)
        let public_at = |q: i64| (q >= 0).then(|| self.n * rb + 2 * (i * c.public_words() + q as usize));
        for (j, &leaf) in c.leaves_in[u].iter().enumerate() {
            let at = if leaf >= 0 { Some(i * rb + 2 * leaf as usize) } else { c.leaves_public.get(u).and_then(|l| public_at(l[j])) };
            if let Some(at) = at {
                let v = u16::from_le_bytes([rows[at], rows[at + 1]]) as u64;
                if col % WORD < 64 { w[col / WORD].lo |= v << (col % WORD) } else { w[col / WORD].hi |= v << (col % WORD - 64) }
            }
            col += unit.in_groups[0].bits[j];
        }
        // a leaf cut (a row word read as a cut input): the word in the port's low 16 bits
        let pc = port_cols(&unit.in_groups);
        for &(j, leaf) in &c.leaf_cuts_of[u] {
            let at = i * rb + 2 * leaf;
            let v = u16::from_le_bytes([rows[at], rows[at + 1]]) as u64;
            let col = pc[j].0;
            if col % WORD < 64 { w[col / WORD].lo |= v << (col % WORD) } else { w[col / WORD].hi |= v << (col % WORD - 64) }
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
    pub fn converge(&self, unit_inputs: &(dyn Fn(usize, usize) -> Vec<F128> + Sync), row_ins: RowIns<'_>, nets: &BTreeMap<usize, &IrUnitNet>) -> Vec<Vec<Vec<F128>>> {
        self.converge_with(unit_inputs, Some(row_ins), nets, None)
    }

    /// [`Stmt::converge`] with the units' output words given (`unit_outs`, the device's first pass): the units are never
    /// evaluated here, only the tail stages and the lookups.
    pub fn converge_with(&self, unit_inputs: &(dyn Fn(usize, usize) -> Vec<F128> + Sync), row_ins: Option<RowIns<'_>>,
                         nets: &BTreeMap<usize, &IrUnitNet>, unit_outs: Option<Vec<Vec<F128>>>) -> Vec<Vec<Vec<F128>>> {
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
        if let Some(f) = row_ins {
            let rv: Vec<_> = (0..nb).into_par_iter().map(f).collect();
            ins[c.sha_net] = rv.iter().flat_map(|x| x.0.iter().cloned()).collect();
            ins[c.hm_net] = rv.into_iter().flat_map(|x| x.1).collect();
        }
        // the rows' wires hold by construction (row_inputs chains natively); the rest converge here
        let row_net = |n: usize| n == c.sha_net || n == c.hm_net;
        let wires: Vec<&Wire> = c.wires.iter().filter(|w| !row_net(w.dst.0)).collect();
        if wires.is_empty() {
            return ins;
        }
        // the slots in the per-VU wire graph's strongly connected components, sources first: a chain of units (GEMM's
        // accumulator, attention's steps) evaluates each slot once; a cycle (a unit that reads its own tail between its two
        // passes) re-evaluates, inside its component, only the slots whose inputs changed
        let mut outs: Vec<Vec<Vec<F128>>> = (0..c.nets.len()).map(|n| vec![Vec::new(); ins[n].len()]).collect();
        let fixed = unit_outs.is_some();
        if let Some(u) = unit_outs {
            outs[c.unit_net] = u;
        }
        let cap = ins.iter().map(|x| x.len()).sum::<usize>() + 64;
        for levels in slot_components(c, &wires) {
            let comp: std::collections::BTreeSet<(usize, usize)> = levels.iter().flatten().copied().collect();
            let from = |n: usize, s: usize| wires.iter().copied().filter(move |w| (w.src.0, w.src.1) == (n, s));
            if comp.iter().all(|&(n, s)| from(n, s).next().is_none()) {
                continue;
            }
            let instances = |n: usize, s: usize| (0..nb).flat_map(move |b| (0..c.g).map(move |g| b * count(n) + g * c.per_vu[n] + s));
            let mut dirty: std::collections::BTreeSet<(usize, usize)> = comp.iter().flat_map(|&(n, s)| instances(n, s).map(move |x| (n, x))).collect();
            // Gauss-Seidel over the component's levels: a level's new outputs reach the slots it feeds before they are evaluated,
            // so a chain settles in one sweep and a cycle's closing wires in one more
            for sweep in 0.. {
                if dirty.is_empty() {
                    break;
                }
                assert!(sweep < cap, "the slots' wires did not converge (a cycle through a slot's own output?)");
                for level in &levels {
                    let pending = &dirty;
                    let todo: Vec<(usize, usize, usize)> = level.iter()
                        .flat_map(|&(n, s)| instances(n, s).filter(move |x| pending.contains(&(n, *x))).map(move |x| (n, s, x))).collect();
                    for &(n, _, x) in &todo {
                        dirty.remove(&(n, x));
                    }
                    for n in todo.iter().map(|t| t.0).collect::<std::collections::BTreeSet<_>>() {
                        if fixed && n == c.unit_net {
                            continue;
                        }
                        let xs: Vec<usize> = todo.iter().filter(|t| t.0 == n).map(|t| t.2).collect();
                        let sub: Vec<Vec<F128>> = xs.iter().map(|&x| ins[n][x].clone()).collect();
                        for (&x, o) in xs.iter().zip(self.eval_outputs(n, net_of(n), &sub)) {
                            outs[n][x] = o;
                        }
                    }
                    for &(n, s, xs) in &todo {
                        for w in from(n, s) {
                            let dn = w.dst.0;
                            let (dcol, width) = port_cols(&net_of(dn).in_groups)[w.dst.2];
                            let scol = port_cols(&net_of(n).out_groups)[w.src.2].0 - net_of(n).out_cols[0] * WORD;
                            let (b, g) = (xs / count(n), xs % count(n) / c.per_vu[n]);
                            let xd = b * count(dn) + g * c.per_vu[dn] + w.dst.1;
                            for t in 0..width {
                                let v = bit(&outs[n][xs][(scol + t) / WORD], (scol + t) % WORD);
                                let d = &mut ins[dn][xd][(dcol + t) / WORD];
                                if bit(d, (dcol + t) % WORD) != v {
                                    flip_bit(d, (dcol + t) % WORD);
                                    if comp.contains(&(dn, w.dst.1)) {
                                        dirty.insert((dn, xd));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        ins
    }

    /// `(z, a, b, z_lincheck)`: every block's compressions (`comps(b)`), every slot of every circuit from its input words `ins`
    /// (`converge`; a lookup by its fast witness), the mask slot from `mask`. `nets` overrides a circuit's evaluation.
    #[allow(clippy::type_complexity)]
    pub fn witness(&self, ins: &[Vec<Vec<F128>>], mask: &(dyn Fn(usize) -> Vec<F128> + Sync), nets: &BTreeMap<usize, &IrUnitNet>)
        -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        self.witness_with(ins, mask, nets, None, None)
    }

    /// [`Stmt::witness`], a typed template's root instance `flip.0` with its bound input row `flip.1` (unit coordinates)
    /// flipped and everything downstream of it evaluated from the flipped bit; or (`forged`) every instance evaluated by
    /// another instance of the same parts (a swapped tail).
    #[allow(clippy::type_complexity)]
    pub fn witness_with(&self, ins: &[Vec<Vec<F128>>], mask: &(dyn Fn(usize) -> Vec<F128> + Sync), nets: &BTreeMap<usize, &IrUnitNet>,
                        flip: Option<(usize, usize)>, forged: Option<&crate::typed::Instance>) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let c = &self.c;
        let kl = c.k_log;
        let n = 1usize << (self.m - 7);
        let wpb = 1usize << (kl - 7);
        let (mut z, mut a, mut bb) = (vec![F128::ZERO; n], vec![F128::ZERO; n], vec![F128::ZERO; n]);
        z.par_chunks_mut(wpb).zip(a.par_chunks_mut(wpb)).zip(bb.par_chunks_mut(wpb)).enumerate().for_each(|(blk, ((zb, ab), bbb))| {
            for r in c.ranges.iter().filter(|r| r.kind == Kind::Mask) {
                let mw = mask(blk);
                let w = r.start() / WORD;
                zb[w..w + mw.len()].copy_from_slice(&mw);
                ab[w..w + mw.len()].copy_from_slice(&mw);
                bbb[w..w + mw.len()].copy_from_slice(&mw);
            }
        });
        for (ni, (_, pinned)) in c.nets.iter().enumerate() {
            if c.typed.as_ref().is_some_and(|t| t.fills(c, ni)) {
                continue;
            }
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
        if let Some(t) = &c.typed {
            for (at, uz, ua, ub) in self.typed_zab(t, &ins[c.unit_net], flip, forged) {
                z[at..at + uz.len()].copy_from_slice(&uz);
                a[at..at + ua.len()].copy_from_slice(&ua);
                bb[at..at + ub.len()].copy_from_slice(&ub);
            }
        }
        let zl = pack_z_lincheck_from_packed(&z, self.m, kl);
        (z, a, bb, zl)
    }

    /// A typed template's root and part slots: `(absolute F128 word offset, z, a, b)`, each VU evaluated from its root's
    /// input words `ins` (`unit_inputs`) by [`crate::typed::Instance::eval64`], 64 at once.
    #[allow(clippy::type_complexity)]
    pub fn typed_zab(&self, t: &TypedBlock, ins: &[Vec<F128>], flip: Option<(usize, usize)>, forged: Option<&crate::typed::Instance>)
        -> Vec<(usize, Vec<F128>, Vec<F128>, Vec<F128>)> {
        let c = &self.c;
        let inst = forged.unwrap_or(&t.inst);
        let (wpb, rr) = (1usize << (c.k_log - 7), c.range_of(c.unit_net));
        let (own, root_rows) = (inst.root.rows.len(), 1usize << c.nets[c.unit_net].1.unit_log);
        let xs: Vec<usize> = (0..ins.len()).collect();
        xs.par_chunks(64).flat_map_iter(|grp| {
            let lanes = flip.filter(|f| grp.contains(&f.0)).map(|f| (f.1, 1u64 << (f.0 % 64)));
            let (zz, aa, bq) = inst.eval64(&|w| grp.iter().enumerate().fold(0u64, |x, (l, &i)| x | (bit(&ins[i][w / WORD], w % WORD) << l)), lanes);
            grp.iter().enumerate().flat_map(|(l, &x)| {
                let (blk, g) = (x / rr.count, x % rr.count);
                // lane l's bits `[lo, lo + used)` into a slot of `rows` rows
                let pack = |v: &Vec<u64>, lo: usize, used: usize, rows: usize| -> Vec<F128> {
                    let mut o = vec![F128::ZERO; rows / WORD];
                    for r in 0..used {
                        if (v[lo + r] >> l) & 1 == 1 {
                            set_bit(&mut o[r / WORD], r % WORD);
                        }
                    }
                    o
                };
                let mut out = vec![(blk * wpb + c.slot_col(c.unit_net, g, 0) / WORD, pack(&zz, 0, own, root_rows), pack(&aa, 0, own, root_rows), pack(&bq, 0, own, root_rows))];
                for (p, &(j, base)) in inst.parts.iter().enumerate() {
                    let (net, q) = t.parts[p];
                    let (span, used) = (1usize << inst.layouts[&j].1, inst.layouts[&j].2.rows.len());
                    out.push((blk * wpb + c.slot_col(net, g, q) / WORD, pack(&zz, base, used, span), pack(&aa, base, used, span), pack(&bq, base, used, span)));
                }
                out
            }).collect::<Vec<_>>()
        }).collect()
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
                let t0 = Instant::now();
                let one_pass = unit_flat() && net.one_pass();
                let flat_s = t0.elapsed().as_secs_f64();
                let prof = GroupProf::default();
                let items = ins.par_chunks(64).enumerate().flat_map_iter(|(gi, grp)| {
                    let (zz, aa, bq) = prof.eval(net, ni, gi, grp);
                    let tl = Instant::now();
                    let (z, a, b) = (lanes(&zz, grp.len(), uw), lanes(&aa, grp.len(), uw), lanes(&bq, grp.len(), uw));
                    prof.out(tl);
                    z.into_iter().zip(a).zip(b).enumerate().map(move |(l, ((z, a), b))| (64 * gi + l, z, a, b))
                }).collect();
                prof.print("slot_zab", ni, net.useful, ins.len(), one_pass, flat_s, t0);
                items
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
        let ins = self.converge_with(unit_inputs, None, &BTreeMap::new(), Some(outs));
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
            if ni == c.unit_net || ni == c.sha_net || ni == c.hm_net {
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
        match reg {
            Region::Digest(p) => (0..c.g).flat_map(|g| match self.inst_of(b, g) { Some(i) => inst.bc(i, *p, np), None => c.dummy_bc[*p] }).collect(),
            // each unit slot's words, 16-bit leaves u16 LE: a public leaf its instance's public word, zero for a zero leaf, a
            // padding instance or a slot past the units
            &Region::In(w, s) => {
                let (ur, np) = (c.unit_range(), c.public_words());
                let j0 = 8 * (w - c.unit().in_groups[0].col);
                let mut out = Vec::with_capacity(16 << (s + log2ceil(ur.count)));
                for k in 0..1usize << log2ceil(ur.count) {
                    let (g, u) = (k / c.upv, k % c.upv);
                    let i = if k < ur.count { self.inst_of(b, g) } else { None };
                    for j in j0..j0 + (8 << s) {
                        let v = match (i, c.leaves_public[u].get(j)) { (Some(i), Some(&q)) if q >= 0 => inst.public[i * np + q as usize], _ => 0 };
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
                out
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

    /// A region claim's ring-switch partial evaluations from the public region data alone: `s[v] = Σ_x eq(p[7..], x) z[128 x + v]`
    /// over the region's cells at its point `p` (M1's audit: a region claim's `s_hat_v` reveals nothing but public data).
    pub fn region_s_hat_v(&self, reg: &RegionShape, r: &[F128], inst: &Instances) -> Vec<F128> {
        let p = self.region_point(reg, r);
        let kl = self.c.k_log;
        let fixed = (reg.fixed & reg.mask) as usize;
        (0..self.blocks()).into_par_iter().map(|b| {
            let bytes = self.region_bytes(&reg.region, inst, b);
            let mut s = vec![F128::ZERO; 128];
            for a in 0..(1usize << reg.free.len()) {
                if (bytes[a / 8] >> (a % 8)) & 1 == 0 {
                    continue;
                }
                let i = reg.free.iter().enumerate().fold(fixed, |i, (t, &j)| i | (((a >> t) & 1) << j)) | (b << kl);
                let x = i >> 7;
                s[i & 127] += (7..self.m).fold(F128::ONE, |acc, j| acc * if (x >> (j - 7)) & 1 == 1 { p[j] } else { F128::ONE + p[j] });
            }
            s
        }).reduce(|| vec![F128::ZERO; 128], |a, b| a.iter().zip(&b).map(|(x, y)| *x + *y).collect())
    }

    /// Sets every public region's bits of the packed witness `z` to the public data (as [`Self::region_s_hat_v`] reads them): a
    /// dummy witness whose region claims are the public ones (M2's extraction run).
    pub fn write_regions(&self, z: &mut [F128], inst: &Instances) {
        let kl = self.c.k_log;
        for reg in &self.regions {
            let fixed = (reg.fixed & reg.mask) as usize;
            for b in 0..self.blocks() {
                let bytes = self.region_bytes(&reg.region, inst, b);
                for a in 0..(1usize << reg.free.len()) {
                    let i = reg.free.iter().enumerate().fold(fixed, |i, (t, &j)| i | (((a >> t) & 1) << j)) | (b << kl);
                    let (w, bit) = (&mut z[i >> 7], i & 127);
                    let on = (bytes[a / 8] >> (a % 8)) & 1 == 1;
                    let m = 1u64 << (bit % 64);
                    let half = if bit < 64 { &mut w.lo } else { &mut w.hi };
                    *half = if on { *half | m } else { *half & !m };
                }
            }
        }
    }

    /// Words per block the device witness needs for the unit slots: every unit's input words, block-major.
    pub fn region_bytes_len(&self, reg: &RegionShape) -> usize {
        (1usize << reg.free.len()) / 8
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    /// PROTOCOL.md §16.5: the statement digest's tag is the statement id. The flat statement keeps its tags; the typed id
    /// has its own.
    #[test]
    fn tags_follow_the_statement_id() {
        assert_eq!((TAGS.statement, TAGS.sigma, TAGS.domain), ("verity/flock-circuit", "verity/flock-circuit/sigma", "flock-circuit/fast100x2/rep"));
        assert_eq!((TYPED_TAGS.statement, TYPED_TAGS.sigma, TYPED_TAGS.domain),
                   ("verity/flock-circuit/types", "verity/flock-circuit/types/sigma", "flock-circuit-types/fast100x2/rep"));
        assert_eq!((TMPL_TAGS.statement, TMPL_TAGS.sigma, TMPL_TAGS.domain),
                   ("verity/flock-circuit/tmpl", "verity/flock-circuit/tmpl/sigma", "flock-circuit-tmpl/fast100x2/rep"));
    }

    #[test]
    fn nonzeros_past_the_gpu_bound_are_refused() {
        let rows = |n: usize| vec![vec![0usize; n], vec![1; n]];
        assert!(nnz_within("u", &rows(3), &rows(2), 6).is_ok());
        assert!(nnz_within("u", &rows(4), &rows(2), 6).unwrap_err().contains("8 nonzeros in A"));
        assert!(nnz_within("u", &rows(1), &rows(4), 6).unwrap_err().contains("8 in B"));
        assert_eq!(GPU_NNZ_MAX, u32::MAX as u64);
    }

    #[test]
    fn k_log_past_in_range_is_refused() {
        assert!(block_in_range(IN_RANGE_K_LOG, 18, 0).is_ok());
        assert!(block_in_range(IN_RANGE_K_LOG + 1, 18, 0).unwrap_err().contains("Stmt.InRange"));
    }

    #[test]
    fn hm96_slot_log_outside_7_to_k_log_is_refused() {
        assert!(block_in_range(20, 7, 0).is_ok() && block_in_range(20, 20, 0).is_ok());
        assert!(block_in_range(20, 6, 0).unwrap_err().contains("hm96"));
        assert!(block_in_range(20, 21, 0).unwrap_err().contains("hm96"));
    }

    #[test]
    fn every_range_slot_log_and_slots_are_bounded_before_any_shift() {
        let r = |slot_log: usize, pos0: usize, count: usize, packed: bool| Range { kind: Kind::Mask, slot_log, pos0, count, packed };
        assert!(range_in_block(20, &r(7, 0, 1 << 13, true)).is_ok() && range_in_block(20, &r(20, 0, 1, false)).is_ok());
        assert!(range_in_block(20, &r(6, 0, 1, false)).unwrap_err().contains("slot_log"));
        // an empty packed range with an oversized slot_log (its span is 0, so the overrun check alone passed it)
        assert!(range_in_block(20, &r(40, 0, 0, true)).unwrap_err().contains("slot_log"));
        // a shift of 64 or more: refused, not a panic (debug) or a wrap (release)
        assert!(range_in_block(20, &r(64, 1, 1, false)).unwrap_err().contains("slot_log"));
        // slots past the block, including a pos0 whose shift would drop bits
        assert!(range_in_block(20, &r(18, 3, 2, false)).unwrap_err().contains("inside the block"));
        assert!(range_in_block(20, &r(18, 1 << 60, 0, true)).unwrap_err().contains("inside the block"));
        assert!(range_in_block(20, &r(18, 4, 0, true)).is_ok());
    }

    #[test]
    fn pin_outside_the_block_is_refused() {
        // the pinned constant past the block: a first range placed at the block's end, as an empty packed range can be
        assert!(block_in_range(20, 18, (1 << 20) - 1).is_ok());
        assert!(block_in_range(20, 18, 1 << 20).unwrap_err().contains("outside the block"));
    }

    #[test]
    fn regions_past_in_range_or_not_subcubes_are_refused() {
        let region = |free: Vec<usize>| RegionShape { region: Region::Digest(0), free, mask: 0, fixed: 0 };
        let many = |n: usize| (0..n).map(|i| region(vec![i % 20, 20])).collect::<Vec<_>>();
        assert!(regions_in_range(&many(IN_RANGE_REGIONS)).is_ok());
        assert!(regions_in_range(&many(IN_RANGE_REGIONS + 1)).unwrap_err().contains("Stmt.InRange"));
        // a region's two bit ranges overlapping: bits 8 and 9 twice (as [bits(0, 10), bits(8, 4)] would be)
        let overlap = region([(0..10).collect::<Vec<_>>(), (8..12).collect()].concat());
        assert!(regions_in_range(&[region((0..14).collect()), overlap]).unwrap_err().contains("LinkLayout"));
    }

    /// §16.10: the in regions' words, run by run, in the largest aligned blocks (the vectors `test_circuit.py` gives Python's).
    #[test]
    fn public_words_are_cut_into_aligned_blocks() {
        let blocks = |w: &[usize]| aligned_blocks(&w.iter().copied().collect());
        assert_eq!(blocks(&(8..137).collect::<Vec<_>>()), [(8, 3), (16, 4), (32, 5), (64, 6), (128, 3), (136, 0)]);
        assert_eq!(blocks(&[0, 1, 2, 3]), [(0, 2)]);
        assert_eq!(blocks(&[3, 5]), [(3, 0), (5, 0)]);
        assert_eq!(blocks(&[1, 2, 3, 4, 5]), [(1, 0), (2, 1), (4, 1)]);
        assert!(blocks(&[]).is_empty());
    }

    #[test]
    fn shapes_past_the_spec_are_refused() {
        assert!(shape_in_range(M_MAX, PT_LOCAL + M_MAX - IN_RANGE_K_LOG).is_ok());
        assert!(shape_in_range(M_MAX + 1, PT_LOCAL).is_err());
        assert!(shape_in_range(30, IN_RANGE_M_PTS + 1).unwrap_err().contains("Stmt.InRange"));
    }

    #[test]
    fn headers_serialize_key_sorted_as_they_are() {
        // public_digest streams the header without sorting it: that holds while serde_json's maps are sorted (no preserve_order)
        let v: Value = serde_json::from_str(r#"{"z": 1, "a": {"y": [2, {"q": 3, "b": 4}], "c": null}, "m": "s"}"#).unwrap();
        assert_eq!(serde_json::to_vec(&v).unwrap(), serde_json::to_vec(&sorted(&v)).unwrap());
    }

    /// The verifier's public file, written as `public_file` writes it, is the file whose SHA-512 the prover's record
    /// carries (`public_file_sha512`), and a drawn statement keeps its population's; Σ's public digest is neither.
    #[test]
    fn the_public_file_digest_is_the_files_own() {
        let c = Composite::parse(&typed_statement(), None).unwrap();
        let n = 2;
        let header = json!({"format": INST_FORMAT, "statement": c.statement(), "circuit_sha512": crate::hex(&c.sha512),
                            "instances": n, "rows": false, "units": units_entry(&c, &[0, 1])});
        let mut b = serde_json::to_vec(&header).unwrap();
        b.push(b'\n');
        b.extend((0..BC_BYTES * n * c.ports.len() + 2 * n * c.public_words()).map(|i| i as u8));
        let p = std::env::temp_dir().join(format!("flock-public-file-{}.bin", std::process::id()));
        std::fs::write(&p, &b).unwrap();
        let inst = Instances::load(p.to_str().unwrap(), &c);
        std::fs::remove_file(&p).unwrap();
        let inst = inst.unwrap();
        let file: [u8; 64] = Sha512::digest(&b).into();
        assert_eq!(inst.public_file().unwrap(), b);
        assert_eq!(inst.public_file_sha512, file);
        assert_ne!(inst.public_sha512, file);
        let d = inst.drawn(&c, &json!({"law": "subset", "population": n, "k": 1, "units": [1]})).unwrap();
        assert_eq!(d.public_file_sha512, file);
        assert_ne!(d.public_sha512, inst.public_sha512);
    }

    #[test]
    fn lanes_are_the_bitwise_pack() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let v: Vec<u64> = (0..1000).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s }).collect();
        let (n, uw) = (37, 1000usize.div_ceil(WORD) + 1);
        for (l, got) in lanes(&v, n, uw).iter().enumerate() {
            let mut want = vec![F128::ZERO; uw];
            for (row, &y) in v.iter().enumerate() {
                if (y >> l) & 1 == 1 {
                    set_bit(&mut want[row / WORD], row % WORD);
                }
            }
            assert!(*got == want, "lane {l}");
        }
    }

    #[test]
    fn bit_rows_are_the_lanes_bits() {
        let mut s = 0x1234_5678_9abc_def1u64;
        let mut r = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let (n, words) = (41, 5);
        let grp: Vec<Vec<F128>> = (0..n).map(|_| (0..words).map(|_| F128::new(r(), r())).collect()).collect();
        let rows = bit_rows(&grp, words);
        assert_eq!(rows.len(), words * WORD);
        for (c, &x) in rows.iter().enumerate() {
            assert_eq!(x >> n, 0, "row {c} has bits past the lanes");
            for (l, v) in grp.iter().enumerate() {
                assert_eq!((x >> l) & 1, bit(&v[c / WORD], c % WORD), "row {c} lane {l}");
            }
        }
    }

    #[test]
    fn unit_ab_into_is_the_lanes_packed() {
        let mut s = 0x0bad_5eed_1234_5677u64;
        let mut r = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        for (rows, n, uw) in [(300, 41, 4), (512, 64, 4), (1000, 1, 8), (64, 7, 1), (4099, 64, 64)] {
            let v: Vec<u64> = (0..rows).map(|_| r()).collect();
            let (stride, off) = (4 * uw + 3, 1);
            let mut dst = vec![!0u64; n * stride];
            lanes_into(&v, n, &mut dst, stride, off, 2 * uw);
            for (l, w) in lanes(&v, n, uw).iter().enumerate() {
                let want: Vec<u64> = w.iter().flat_map(|x| [x.lo, x.hi]).collect();
                assert_eq!(dst[l * stride + off..l * stride + off + 2 * uw], want[..], "rows {rows}, lane {l}");
            }
        }
        let n = 10;
        let table: &'static [u32] = Box::leak((0..1u32 << n).map(|_| r() as u32 & 0x7fff_ffff).collect::<Vec<_>>().into_boxed_slice());
        let (net, _) = crate::lookup::LookupNet::build("t", table, n, 4, 31);
        let uw = 1usize << (net.unit_log - 7);
        // an assertion row (its own column in A, B the constant) that fails: z is not a & b there
        let mut bad = net.clone();
        let i = (0..bad.useful).rev().find(|&i| i != bad.const_pos && !bad.a[i].is_empty() && !bad.b[i].is_empty()).unwrap();
        bad.b[i] = vec![bad.const_pos];
        bad.a[i].push(i);
        for (net, slots, wide) in [(&net, 100, false), (&net, 64, false), (&net, 130, true), (&bad, 100, true)] {
            let ins: Vec<Vec<F128>> = (0..slots).map(|_| (0..net.in_words).map(|w| match (w, wide) {
                (0, false) => F128::new(r() & ((1 << n) - 1), 0),
                (_, false) => F128::ZERO,
                _ => F128::new(r(), r()),
            }).collect()).collect();
            let (mut want, mut ok) = (Vec::new(), true);
            for grp in ins.chunks(64) {
                let (z, a, b) = net.eval64(|col| grp.iter().enumerate().fold(0u64, |x, (l, w)| x | (bit(&w[col / WORD], col % WORD) << l)));
                let (z, a, b) = (lanes(&z, grp.len(), uw), lanes(&a, grp.len(), uw), lanes(&b, grp.len(), uw));
                for ((z, a), b) in z.iter().zip(&a).zip(&b) {
                    ok &= z.iter().zip(a).zip(b).all(|((z, a), b)| z.lo == a.lo & b.lo && z.hi == a.hi & b.hi);
                    want.extend(a.iter().chain(b).flat_map(|x| [x.lo, x.hi]));
                }
            }
            let mut out = vec![!0u64; 4 * uw * slots];
            assert_eq!(unit_ab_into(net, 0, uw, &ins, &mut out), ok, "{slots} slots: z = a & b");
            assert_eq!(ok, !std::ptr::eq(net, &bad), "a lookup's rows hold, the failed assertion does not");
            if ok {
                assert!(out == want, "{slots} slots: the packed a, b words");
            }
        }
    }

    #[test]
    fn native_compression_is_sha512() {
        let msg: Vec<u8> = (0..300u32).map(|i| (i * 7 + 3) as u8).collect();
        let mut padded = msg.clone();
        padded.push(0x80);
        while padded.len() % 128 != 112 {
            padded.push(0);
        }
        padded.extend((8 * msg.len() as u128).to_be_bytes());
        let mut cv = SHA512_IV;
        for blk in padded.chunks(128) {
            compress(&mut cv, blk);
        }
        let want: Vec<u8> = Sha512::digest(&msg).to_vec();
        assert_eq!(cv.iter().flat_map(|w| w.to_be_bytes()).collect::<Vec<u8>>(), want);
        let mut p = row_prefix(64);
        p.extend(vec![0u8; 128]);
        p.extend(pad_block(256));
        let mut cv = SHA512_IV;
        for blk in p.chunks(128) {
            compress(&mut cv, blk);
        }
        let mut h = Sha512::new();
        h.update(row_prefix(64));
        h.update([0u8; 128]);
        assert_eq!(cv.iter().flat_map(|w| w.to_be_bytes()).collect::<Vec<u8>>(), h.finalize().to_vec());
    }

    /// A circuit with these port groups (bits per port) and no rows but the constant.
    fn section(name: &str, ins: &[usize], in_words: usize, outs: &[usize], out_words: usize) -> String {
        let n = (in_words + out_words) * WORD + 1;
        let head = [format!("flock-ir-unit/v2 {name} {n} {} 1 0 {in_words} {}", n - 1, ins.len())].into_iter()
            .chain(ins.iter().map(usize::to_string)).chain([format!("1 {in_words} {out_words} {}", outs.len())])
            .chain(outs.iter().map(usize::to_string)).collect::<Vec<_>>().join(" ");
        let rows: String = (0..n).map(|i| if i == n - 1 { format!("1 {i} 1 {i}\n") } else { "0 0\n".into() }).collect();
        format!("CIRCUIT {name} {}\n{head}\n{rows}", n + 1)
    }

    /// META `rows` entry of a `sha512/row/v1` row of 64 words.
    fn row64() -> Value {
        let prefix = row_prefix(64);
        let mut mid = SHA512_IV;
        compress(&mut mid, &prefix);
        let pad = words_be::<16>(&pad_block(256));
        json!({"prefix": crate::hex(&prefix), "midstate": mid.iter().map(|w| format!("{w:016x}")).collect::<Vec<_>>(),
               "pad": crate::hex(&pad.iter().flat_map(|w| w.to_be_bytes()).collect::<Vec<u8>>())})
    }

    /// A typed template statement small enough to build here: one row port of 64 words, a unit type calling a flat type
    /// twice (placed in two slots of its layout) with an exported, a bound and a constant-form argument, an AND reading
    /// both parts' outputs, and row circuits that have their ports' shapes but no rows.
    fn typed_statement() -> String {
        let callee = json!({"format": "verity/circuit-type/v1", "label": "program", "in": [[1, 2], [1, 1]], "out": [[1, 1]], "calls": [],
                            "tables": [], "items": [[0, [[0], 0], [[1], 0]], [0, [[3], 0], [[2], 1]]], "outputs": [[[4], 0]]});
        let c = crate::typed::type_digest(&callee);
        let mut outputs = vec![json!([[1026], 0]), json!([[1024, 1025], 0]), json!([[5], 0]), json!([[], 1]), json!([[1025], 1])];
        outputs.resize(16, json!([[], 0]));
        let root = json!({"format": "verity/circuit-type/v1", "label": "program", "in": [[64, 16]], "out": [[1, 16]], "calls": [c], "tables": [],
                          "items": [[3, 0, [[[0], 0], [[1], 0], [[2], 1]]], [3, 0, [[[1024], 0], [[3], 0], [[1], 0]]], [0, [[1024], 0], [[1, 1025], 0]]],
                          "outputs": outputs});
        let r = crate::typed::type_digest(&root);
        let with_digest = |mut l: Value| { l["sha512"] = json!(crate::typed::layout_digest(&l)); l };
        let l0 = with_digest(json!({"type": c, "range_log": 9, "in_split": [], "out_split": [], "own": {"derive": "v2"}, "calls": [], "reads": []}));
        let l1 = with_digest(json!({"type": r, "in_split": [], "out_split": [], "own": {"derive": "v2"}, "reads": [],
                                    "calls": [{"item": 0, "layout": 0, "offset": 0}, {"item": 1, "layout": 0, "offset": 1}]}));
        let zeros = "0".repeat(128);
        let row = row64();
        // the input row x, then y's output row (its one word zero-padded to a block): one hm96 slot per row
        let meta = json!({"format": crate::typed::FORMAT, "statement": crate::typed::STATEMENT, "program_sha512": zeros,
            "partition": {"rule": PARTITION_V1, "digest": zeros}, "scheme": SCHEME, "row_schema": ROW_SCHEMA,
            "ports": [["x", 64, 1], ["y", 64, 1]], "input_ports": 1, "rows": [row, row],
            "row_key_sha512": crate::hex(&Sha512::digest(hm96_sha512_default_key())), "vus_per_block": 2, "k_log": 17,
            "ranges": [{"type": "unit", "net": SHA_NET, "slot_log": 14, "pos0": 0, "count": 2, "packed": true},
                       {"type": "unit", "net": HM_NET, "slot_log": 13, "pos0": 4, "count": 4},
                       {"type": "unit", "net": "root", "slot_log": 11, "pos0": 32, "count": 2},
                       {"type": "unit", "net": l0["sha512"], "slot_log": 9, "pos0": 136, "count": 4, "packed": true},
                       {"type": "mask", "net": "mask", "slot_log": 10, "pos0": 70, "count": 1}],
            "per_vu": {SHA_NET: 1, HM_NET: 2, "root": 1, l0["sha512"].as_str().unwrap(): 2},
            "dummy": {"bc": [crate::hex(&[0u8; BC_BYTES]), crate::hex(&[0u8; BC_BYTES])], "outputs": [0]}, "out_ports": [["y", 1]],
            "leaf_scheme": leaf_scheme(),
            "policy": "calls/v1", "unit_in": [], "types": {r.clone(): root, c: callee}, "layouts": [l0, l1], "unit_layout": 1, "unit_type": r});
        format!("{} t\nMETA {}\n{}{}", crate::typed::FORMAT, serde_json::to_string(&meta).unwrap(),
                section(SHA_NET, &[512, 1024, 512, 1024, 512, 1024], 64, &[512, 512, 512], 16),
                section(HM_NET, &[512, 1024, 8 * SALT_BYTES], 32, &[8 * BC_BYTES], 8))
    }

    /// The block's Δ for a typed template, VU 0's entries mapped back to unit coordinates (`TypedBlock::col`): the
    /// constants' pins and the parts' bindings are `delta.txt`'s lines, the message-bit copies are the instance's inputs
    /// at their columns, and the entries without a self entry are the own rows' entries at parts' columns.
    #[test]
    fn typed_block_delta_is_the_derivation() {
        let c = Composite::parse(&typed_statement(), None).unwrap();
        let t = c.typed.as_ref().unwrap();
        assert_eq!(c.tags().statement, crate::typed::STATEMENT);
        let pin = pin_of(&c);
        let (da, db) = delta(&c, pin);
        let mut back: BTreeMap<usize, usize> = (0..t.inst.root.rows.len()).map(|u| (t.col(&c, 0, u), u)).collect();
        for &(j, base) in &t.inst.parts {
            back.extend((base..base + t.inst.layouts[&j].2.rows.len()).map(|u| (t.col(&c, 0, u), u)));
        }
        let msg_col = c.msg_col();
        let msg: BTreeMap<usize, (usize, usize)> = (0..64 * 16).map(|w| (msg_col(0, 0, 2 * (w / 16) + w % 16 / 8, w % 8), (w / 16, w % 16))).collect();
        let (mut lines, mut copies, mut cross) = (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
        for (side, d) in [("A", &da), ("B", &db)] {
            let mut rows: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for &(i, x) in d.iter().filter(|e| back.contains_key(&(e.0 as usize))) {
                rows.entry(i as usize).or_default().push(x as usize);
            }
            for (i, xs) in rows {
                let u = back[&i];
                let rest: Vec<usize> = xs.iter().copied().filter(|&x| x != i).collect();
                if rest.len() == xs.len() {
                    cross.extend(rest.iter().map(|x| (u, side, back[x])));
                } else if rest == [pin] {
                    lines.insert(format!("DELTA {u} 1 one"));
                } else if let [x] = rest[..] && msg.contains_key(&x) {
                    copies.insert((u, msg[&x]));
                } else {
                    let mut src: Vec<usize> = rest.iter().map(|x| back[x]).collect();
                    src.sort_unstable();
                    lines.insert(format!("DELTA {u} {} {}", src.len(), src.iter().map(usize::to_string).collect::<Vec<_>>().join(" ")));
                }
            }
        }
        let staged = crate::typed::staged_rows(&c.meta, &c.name, None).unwrap();
        let delta_txt = &staged.last().unwrap().1;
        assert_eq!(lines, delta_txt.lines().filter(|l| l.starts_with("DELTA")).map(str::to_string).collect::<BTreeSet<_>>());
        for (l, (&(j, base), &(net, q))) in delta_txt.lines().filter(|l| l.starts_with("PART")).zip(t.inst.parts.iter().zip(&t.parts)) {
            assert_eq!(l, format!("PART {} {base}", c.nets[net].0));
            assert_eq!((t.col(&c, 0, base), &c.nets[net].0), (c.slot_col(net, 0, q), &t.inst.layouts[&j].0));
        }
        assert_eq!(copies, t.inst.root.in_cols.iter().enumerate().map(|(w, &u)| (u, (w / 16, w % 16))).collect::<BTreeSet<_>>());
        let want: BTreeSet<(usize, &str, usize)> = t.inst.cross().into_iter()
            .flat_map(|(r, a, b)| a.into_iter().map(move |x| (r, "A", x)).chain(b.into_iter().map(move |x| (r, "B", x)))).collect();
        assert_eq!(cross, want);
        // the instance exercises every kind: two exported inputs, bindings of a part's output, an input and a constant form
        assert_eq!((t.inst.parts.len(), t.inst.bind.iter().map(Vec::len).sum::<usize>(), t.inst.root.in_cols.iter().filter(|&&u| u >= t.inst.root.rows.len()).count()), (2, 4, 2));
        assert!(!cross.is_empty() && lines.len() == 3 + 4);
    }

    /// A generated read's table-direct B side folds as `LookupNet::build_v2`'s explicit product rows do, at random weights.
    #[test]
    fn the_table_side_folds_as_the_explicit_rows() {
        let table: &'static [u32] = Box::leak((0..512u32).map(|i| (i.wrapping_mul(2654435761) >> 7) & 0x7FFF_FFFF).collect::<Vec<_>>().into_boxed_slice());
        let (net, ln) = LookupNet::build_v2("t", &crate::lookup::table_sha512(table), table, 9, 4, 31).unwrap();
        let n = 1usize << net.unit_log;
        let mut s = 7u64;
        let eq: Vec<F128> = (0..n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); F128::new(s, s.rotate_left(29)) }).collect();
        let mut want = vec![F128::ZERO; n];
        for r in ln.prod0()..ln.prod0() + (ln.k << 5) {
            for &c in &net.b[r] {
                want[c] += eq[r];
            }
        }
        let mut got = vec![F128::ZERO; n];
        TableSide::of(&ln).fold(&eq, &mut got);
        assert!(ln.k == 31 && got == want);
    }

    /// Every slot type's column-major fold is upstream's row-scatter fold of the same rows (`SparseMatrixCircuit`), both
    /// sides and α-batched, at random weights: units with rows, units with none but the constant, and the mask.
    #[test]
    fn slot_type_folds_are_the_row_folds() {
        use flock_core::lincheck::SparseMatrixCircuit;
        let c = Composite::parse(&typed_statement(), None).unwrap();
        let st = Stmt::new(c.clone(), 2, json!({})).unwrap();
        let mut s = 11u64;
        let mut rnd = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); F128::new(s, s.rotate_left(29)) };
        let alpha = rnd();
        let mut seen = BTreeSet::new();
        for (r, &(t, _, _)) in c.ranges.iter().zip(&st.circuit.ranges) {
            if !seen.insert(t) {
                continue;
            }
            let k = 1usize << r.slot_log;
            let (a, b) = match r.kind {
                Kind::Net(i) => {
                    let pad = |rows: &Vec<Vec<usize>>| { let mut x = rows.clone(); x.resize(k, Vec::new()); SparseBinaryMatrix { num_rows: k, num_cols: k, rows: x.into() } };
                    (pad(&c.nets[i].1.a), pad(&c.nets[i].1.b))
                }
                Kind::Mask => {
                    let id = SparseBinaryMatrix { num_rows: k, num_cols: k, rows: (0..k).map(|i| vec![i]).collect::<Vec<_>>().into() };
                    (id.clone(), id)
                }
            };
            let eq: Vec<F128> = (0..k).map(|_| rnd()).collect();
            let (rows, cols) = (SparseMatrixCircuit::new(&a, &b), &st.circuit.types[t].1);
            assert_eq!(cols.fold_alpha_batched(alpha, &eq), rows.fold_alpha_batched(alpha, &eq), "{:?}", r.kind);
            assert_eq!(cols.fold_split(&eq), rows.fold_split(&eq), "{:?}", r.kind);
        }
        assert_eq!(seen.len(), st.circuit.types.len());
        assert!(c.ranges.iter().any(|r| r.kind == Kind::Mask) && c.nets.iter().any(|n| n.1.a.iter().filter(|x| !x.is_empty()).count() > 1));
    }

    /// A typed template's refusals at parse: a non-power-of-two count of roots, and the flat block's wiring stated.
    #[test]
    fn typed_template_refusals() {
        let text = typed_statement();
        let meta_of = |f: &dyn Fn(&mut Value)| {
            let mut lines: Vec<String> = text.split('\n').map(str::to_string).collect();
            let mut m: Value = serde_json::from_str(&lines[1][5..]).unwrap();
            f(&mut m);
            lines[1] = format!("META {}", serde_json::to_string(&m).unwrap());
            Composite::parse(&lines.join("\n"), None).err().unwrap_or_default()
        };
        // three VUs a block, every range laid out for them (aligned, disjoint) in 2^18 bits
        assert!(meta_of(&|m| {
            let part = m["ranges"][3]["net"].clone();
            m["vus_per_block"] = json!(3);
            m["k_log"] = json!(18);
            m["ranges"] = json!([{"type": "unit", "net": SHA_NET, "slot_log": 14, "pos0": 0, "count": 3, "packed": true},
                                 {"type": "unit", "net": HM_NET, "slot_log": 13, "pos0": 8, "count": 6},
                                 {"type": "unit", "net": "root", "slot_log": 11, "pos0": 64, "count": 3},
                                 {"type": "unit", "net": part, "slot_log": 9, "pos0": 272, "count": 6, "packed": true},
                                 {"type": "mask", "net": "mask", "slot_log": 10, "pos0": 139, "count": 1}]);
        }).contains("not a power of two"), "three roots");
        assert!(meta_of(&|m| m["wires"] = json!([])).contains("never stated"));
        assert!(Composite::parse(&text, None).is_ok());
    }

    /// A flat statement whose tail returns the outputs: an input row x and y's output row (one word), a unit reading one
    /// leaf of x, and a stage reading the unit's output and returning y's word, with circuits of their ports' shapes only.
    fn flat_tail_statement() -> String {
        let zeros = "0".repeat(128);
        let ports = [Port { name: "x".into(), words: 64, comps: 1, bits: 1024, v2: false, segs: None },
                     Port { name: "y".into(), words: 64, comps: 1, bits: 1024, v2: false, segs: None }];
        let name = |n: usize| ["unit", "stage0", SHA_NET, HM_NET][n];
        let mut wires: Vec<Value> = row_wires(&ports, 2, 3).iter()
            .map(|w| json!([name(w.dst.0), w.dst.1, w.dst.2, name(w.src.0), w.src.1, w.src.2])).collect();
        wires.push(json!(["stage0", 0, 0, "unit", 0, 0]));
        let meta = json!({"format": FORMAT, "statement": STATEMENT, "program_sha512": zeros,
            "partition": {"rule": PARTITION_V1, "digest": zeros}, "scheme": SCHEME, "row_schema": ROW_SCHEMA,
            "ports": [["x", 64, 1], ["y", 64, 1]], "input_ports": 1, "rows": [row64(), row64()],
            "row_key_sha512": crate::hex(&Sha512::digest(hm96_sha512_default_key())), "vus_per_block": 2, "k_log": 17,
            "units_per_vu": 1, "leaves_in": [[0]], "leaves_out": [[]], "leaf_cuts": [], "wires": wires,
            "out_net": "stage0", "out_net_ports": [0],
            "ranges": [{"type": "unit", "net": SHA_NET, "slot_log": 14, "pos0": 0, "count": 2, "packed": true},
                       {"type": "unit", "net": HM_NET, "slot_log": 13, "pos0": 4, "count": 4},
                       {"type": "unit", "net": "unit", "slot_log": 9, "pos0": 128, "count": 2},
                       {"type": "unit", "net": "stage0", "slot_log": 9, "pos0": 130, "count": 2},
                       {"type": "mask", "net": "mask", "slot_log": 10, "pos0": 66, "count": 1}],
            "per_vu": {SHA_NET: 1, HM_NET: 2, "stage0": 1},
            "dummy": {"bc": [crate::hex(&[0u8; BC_BYTES]), crate::hex(&[0u8; BC_BYTES])], "outputs": [0]}, "out_ports": [["y", 1]],
            "leaf_scheme": leaf_scheme()});
        format!("{FORMAT} t\nMETA {}\n{}{}{}{}", serde_json::to_string(&meta).unwrap(),
                section("unit", &[16], 1, &[16], 1), section("stage0", &[16], 1, &[16], 1),
                section(SHA_NET, &[512, 1024, 512, 1024, 512, 1024], 64, &[512, 512, 512], 16),
                section(HM_NET, &[512, 1024, 8 * SALT_BYTES], 32, &[8 * BC_BYTES], 8))
    }

    /// A repeated last stage (two slots per VU, slot `s` reading unit `s`) returns output word `o` from its slot's port (META
    /// `out_net_ports` of `[slot, port]`); an output word named twice or not at all, or a slot or port past the stage's, is refused.
    #[test]
    fn a_repeated_last_stage_returns_each_output_from_its_slot() {
        let text = flat_tail_statement();
        let edit = |f: &dyn Fn(&mut Value)| {
            let mut lines: Vec<String> = text.split('\n').map(str::to_string).collect();
            let mut m: Value = serde_json::from_str(&lines[1][5..]).unwrap();
            m["units_per_vu"] = json!(2);
            m["leaves_in"] = json!([[0], [1]]);
            m["leaves_out"] = json!([[], []]);
            m["ranges"][2]["count"] = json!(4);
            m["ranges"][3]["count"] = json!(4);
            m["ranges"][3]["pos0"] = json!(136);
            m["per_vu"]["stage0"] = json!(2);
            let mut wires: Vec<Value> = m["wires"].as_array().unwrap().iter().filter(|w| w[0] != "stage0").cloned().collect();
            wires.extend((0..2).map(|s| json!(["stage0", s, 0, "unit", s, 0])));
            m["wires"] = json!(wires);
            m["out_ports"] = json!([["y", 2]]);
            m["dummy"]["outputs"] = json!([0, 0]);
            m["out_net_ports"] = json!([[0, 0], [1, 0]]);
            f(&mut m);
            lines[1] = format!("META {}", serde_json::to_string(&m).unwrap());
            lines.join("\n")
        };
        let c = Composite::parse(&edit(&|_| {}), None).unwrap_or_else(|e| panic!("{e}"));
        let s0 = c.nets.iter().position(|n| n.0 == "stage0").unwrap();
        let at = port_cols(&c.nets[s0].1.out_groups)[0].0;
        assert_eq!((c.out_col()(1, 0), c.out_col()(1, 1)), (c.slot_col(s0, 1, 0) + at, c.slot_col(s0, 1, 1) + at));
        for bad in [json!([[0, 0], [0, 0]]), json!([[0, 0], [2, 0]]), json!([[0, 0]]), json!([[0, 0], [1, 1]])] {
            let err = Composite::parse(&edit(&|m| m["out_net_ports"] = bad.clone()), None).err().unwrap_or_default();
            assert!(err.contains("each once"), "{bad}: {err}");
        }
        let flat = Composite::parse(&edit(&|m| m["out_net_ports"] = json!([0, 0])), None).err().unwrap_or_default();
        assert!(flat.contains("one 16-bit port per output word"), "{flat}");
    }

    /// No unit in a VU is refused (PROTOCOL.md §16.10, D8), with the tail that returns the outputs, whose unit count need not
    /// be a power of two, kept consistent: no unit or stage slot, no leaf map and no wire but the rows'.
    #[test]
    fn a_tail_with_no_unit_is_refused() {
        let text = flat_tail_statement();
        assert!(Composite::parse(&text, None).is_ok(), "{:?}", Composite::parse(&text, None).err());
        let mut lines: Vec<String> = text.split('\n').map(str::to_string).collect();
        let mut m: Value = serde_json::from_str(&lines[1][5..]).unwrap();
        m["units_per_vu"] = json!(0);
        m["ranges"][2]["count"] = json!(0);
        m["ranges"][3]["count"] = json!(0);
        m["per_vu"]["stage0"] = json!(0);
        m["leaves_in"] = json!([]);
        m["leaves_out"] = json!([]);
        m["wires"] = json!(m["wires"].as_array().unwrap()[..2]);
        lines[1] = format!("META {}", serde_json::to_string(&m).unwrap());
        let err = Composite::parse(&lines.join("\n"), None).err().unwrap_or_default();
        assert!(err.contains("units_per_vu is 0"), "{err}");
    }

    /// No VU in a block is refused (PROTOCOL.md §16.10, D9), with the tail that returns the outputs, whose unit count need not
    /// be a power of two, kept consistent: every circuit's range has no slot.
    #[test]
    fn a_block_with_no_vu_is_refused() {
        let text = flat_tail_statement();
        let mut lines: Vec<String> = text.split('\n').map(str::to_string).collect();
        let mut m: Value = serde_json::from_str(&lines[1][5..]).unwrap();
        m["vus_per_block"] = json!(0);
        for i in 0..4 {
            m["ranges"][i]["count"] = json!(0);
        }
        lines[1] = format!("META {}", serde_json::to_string(&m).unwrap());
        let err = Composite::parse(&lines.join("\n"), None).err().unwrap_or_default();
        assert!(err.contains("vus_per_block is 0"), "{err}");
    }

    /// Three VUs in a block are refused (PROTOCOL.md §16.10, D9), with the tail that returns the outputs, whose unit count
    /// need not be a power of two, laid out for them (aligned, disjoint) in the same block: the Digest region's VU bits pin
    /// every VU only at a power of two.
    #[test]
    fn a_block_of_three_vus_is_refused() {
        let text = flat_tail_statement();
        let mut lines: Vec<String> = text.split('\n').map(str::to_string).collect();
        let mut m: Value = serde_json::from_str(&lines[1][5..]).unwrap();
        m["vus_per_block"] = json!(3);
        for (i, (pos0, count)) in [(0, 3), (8, 6), (96, 3), (100, 3), (52, 1)].into_iter().enumerate() {
            m["ranges"][i]["pos0"] = json!(pos0);
            m["ranges"][i]["count"] = json!(count);
        }
        lines[1] = format!("META {}", serde_json::to_string(&m).unwrap());
        let err = Composite::parse(&lines.join("\n"), None).err().unwrap_or_default();
        assert!(err.contains("vus_per_block is not a power of two"), "{err}");
    }

    /// A writer with no output row (pinned outputs' form B): one 1024-bit `sha512/row/v2` input row `x`, read as one 16-bit leaf,
    /// and one pinned output word `d = x XOR (x rotated by one)`, each bit a linear row (B the constant alone).
    fn pinned_statement(edit: &dyn Fn(&mut Value)) -> String {
        let x = Port { name: "x".into(), words: 64, comps: 1, bits: 1024, v2: true, segs: None };
        let rc = row_consts(&x);
        let name = |n: usize| ["unit", SHA_NET, HM_NET][n];
        let wires: Vec<Value> = row_wires(std::slice::from_ref(&x), 1, 2).iter()
            .map(|w| json!([name(w.dst.0), w.dst.1, w.dst.2, name(w.src.0), w.src.1, w.src.2])).collect();
        let zeros = "0".repeat(128);
        let mut meta = json!({"format": FORMAT, "statement": STATEMENT, "program_sha512": zeros,
            "partition": {"rule": PARTITION_V1, "digest": zeros}, "scheme": SCHEME, "row_schemas": [[ROW_SCHEMA_V2, 1024]],
            "ports": [["x", 64, 1]], "input_ports": 1,
            "rows": [{"prefix": crate::hex(&rc.prefix), "midstate": rc.midstate.iter().map(|w| format!("{w:016x}")).collect::<Vec<_>>(),
                      "pad": crate::hex(&rc.tail)}],
            "row_key_sha512": crate::hex(&Sha512::digest(hm96_sha512_default_key())), "vus_per_block": 2, "k_log": 17,
            "units_per_vu": 1, "leaves_in": [[0]], "leaves_out": [[0]], "wires": wires,
            "ranges": [{"type": "unit", "net": SHA_NET, "slot_log": 14, "pos0": 0, "count": 2, "packed": true},
                       {"type": "unit", "net": HM_NET, "slot_log": 13, "pos0": 4, "count": 2},
                       {"type": "unit", "net": "unit", "slot_log": 9, "pos0": 128, "count": 2},
                       {"type": "mask", "net": "mask", "slot_log": 10, "pos0": 66, "count": 1}],
            "per_vu": {SHA_NET: 1, HM_NET: 1},
            "dummy": {"bc": [crate::hex(&[0u8; BC_BYTES])], "outputs": [0]}, "out_ports": [["d", 1]], "pinned_outputs": ["d"],
            "leaf_scheme": leaf_scheme()});
        edit(&mut meta);
        let n = 2 * WORD + 1;
        let rows: String = (0..n).map(|i| match i {
            0..16 => format!("1 {i} 1 {i}\n"),
            128..144 => { let (a, b) = (i - 128, (i - 127) % 16); format!("2 {} {} 1 {}\n", a.min(b), a.max(b), n - 1) }
            _ if i == n - 1 => format!("1 {i} 1 {i}\n"),
            _ => "0 0\n".into(),
        }).collect();
        let unit = format!("CIRCUIT unit {}\nflock-ir-unit/v2 unit {n} {} 1 0 1 1 16 1 1 1 1 16\n{rows}", n + 1, n - 1);
        format!("{FORMAT} t\nMETA {}\n{unit}{}{}", serde_json::to_string(&meta).unwrap(),
                section(SHA_NET, &[512, 1024, 512, 1024, 512, 1024], 64, &[512, 512, 512], 16),
                section(HM_NET, &[512, 1024, 8 * SALT_BYTES], 32, &[8 * BC_BYTES], 8))
    }

    /// A pinned output (META `pinned_outputs`) has no row, and Δ ends, per VU, with each of its bits a copy of the unit range's
    /// forced-zero row; the live server refuses the form until a soundness theorem covers it. META naming anything but the
    /// last output ports, of a statement of bit rows whose padding instance's pinned outputs are zero, is refused, and so is
    /// a pinned bit whose row is not linear.
    #[test]
    fn pinned_outputs_are_copies_of_the_forced_zero_row() {
        let c = Composite::parse(&pinned_statement(&|_| {}), None).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((c.pinned, c.out_row_ports.len(), c.ports.len()), (1, 0, 1));
        let (da, db) = delta(&c, pin_of(&c));
        let zero = (c.unit_range().slot(0) + c.unit().useful) as u32;
        let out_col = &c.out_col();
        let want: Vec<(u32, u32)> = (0..c.g).flat_map(|g| (0..16).map(move |t| (out_col(g, 0) + t) as u32))
            .flat_map(|i| [(i, i), (i, zero)]).collect();
        assert_eq!((&da[da.len() - want.len()..], &db[db.len() - want.len()..]), (&want[..], &want[..]));
        assert_eq!(crate::proved_scope::circuit_why(&c, true).as_deref(), Some(crate::proved_scope::PINNED_WHY));
        let refused: [(&str, &dyn Fn(&mut Value)); 5] = [
            ("not an output port", &|m| m["pinned_outputs"] = json!(["e"])),
            ("none", &|m| m["pinned_outputs"] = json!([])),
            ("named twice", &|m| m["pinned_outputs"] = json!(["d", "d"])),
            ("a nonzero padding output", &|m| m["dummy"]["outputs"] = json!([1])),
            ("not a list of names", &|m| m["pinned_outputs"] = json!("d")),
        ];
        for (case, edit) in refused {
            let err = Composite::parse(&pinned_statement(edit), None).err().unwrap_or_default();
            assert!(err.contains(PINNED_REFUSED), "{case}: {err}");
        }
        let text = pinned_statement(&|_| {}).replace("2 0 1 1 256\n", "2 0 1 0\n");
        let err = Composite::parse(&text, None).err().unwrap_or_default();
        assert!(err.contains(PINNED_LINEAR), "{err}");
    }

    #[test]
    fn hm96_sha512_default_key_is_core_s() {
        // hashlib.sha512(verity.ml.commitments.hm96.SHA512.default_key).hexdigest()
        assert_eq!(super::leaf_scheme()["key_sha512"],
                   "afc5a60c85d8aa75cb4c7f726588d299a1712e1c6a621f09ca64809a5fedd744a0f6b0201d05e82b29cd5b274e6c29670cd4e4ab10831911fb1f9e0c7019df22");
    }

    /// flock_merkle::hm96 (verity's Flock patch) against core's verity.ml.commitments.hm96 vectors_sha512.json: the key, both
    /// prefixes, the default-key masks and the leaves over given inner digests.
    #[cfg(feature = "sha512")]
    #[test]
    fn hm96_leaves_match_core_vectors() {
        use flock_merkle::hm96;
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).canonicalize().unwrap().join("../../../..");
        let Ok(text) = std::fs::read_to_string(repo.join("verity/ml/commitments/hm96/vectors_sha512.json")) else {
            return;
        };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let hx = |s: &serde_json::Value| super::hexb(s.as_str().unwrap()).unwrap();
        assert_eq!(hm96::default_key().to_vec(), hx(&v["default_key"]));
        assert_eq!(hm96::consts().salt_prefix.to_vec(), hx(&v["salt_prefix"]));
        assert_eq!(hm96::consts().leaf_prefix.to_vec(), hx(&v["leaf_prefix_default_key"]));
        let mut n = 0;
        for m in v["masks"].as_array().unwrap().iter().filter(|m| m["key"] == "default") {
            assert_eq!(hm96::mask(&hx(&m["salt"])).to_vec(), hx(&m["mask"]), "{}", m["salt_case"]);
            n += 1;
        }
        for l in v["leaves"].as_array().unwrap() {
            let x = flock_hash::digest_from(&hx(&l["inner_digest"]));
            assert_eq!(hm96::leaf(&x, &hx(&l["salt"])).as_flattened().to_vec(), hx(&l["leaf"]));
            n += 1;
        }
        assert!(n >= 8, "{n} vectors");
    }

    /// A bit row's message as the Δ constants leave it (`const_bit`, else the row's bit at `block_bit`'s byte and bit),
    /// compressed from the chaining value after `P(bits)`, is SHA-512 of `P(bits)` and the row's bytes (`sha512/row/v2`) at
    /// every residue of the length: `bits` mod 8 of 0, 1 and 7, and its bytes mod 128 of 0, 1, 111, 112 and 127.
    #[test]
    fn bit_row_constants_complete_sha512_of_the_prefix_and_bytes() {
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut lengths = vec![0usize, 8, 1024, 2048];
        for bytes in [1usize, 111, 112, 127, 128, 129, 239, 240, 255, 256] {
            lengths.extend([8 * bytes - 7, 8 * bytes - 1]);
        }
        for bits in lengths {
            let bytes = bits.div_ceil(8);
            let port = Port { name: "p".into(), words: bits.div_ceil(16), comps: (bytes + 17).div_ceil(128) - 1, bits, v2: true, segs: None };
            let prefix = row_prefix_v2(bits as u64);
            let mut midstate = SHA512_IV;
            compress(&mut midstate, &prefix);
            let rc = RowConsts { prefix: prefix.clone(), midstate, tail: pad_tail(128 + bytes), segs: Vec::new() };
            let mut row: Vec<u8> = (0..bytes).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect();
            if bits % 8 != 0 {
                row[bytes - 1] &= (1u8 << (bits % 8)) - 1;
            }
            let mut cv = rc.midstate;
            for k in 0..=port.comps {
                let mut blk = [0u8; 128];
                for t in 0..1024 {
                    let (o, b) = block_bit(t);
                    let v = const_bit(&port, &rc, k, t).unwrap_or_else(|| (row[128 * k + o] >> b) & 1 == 1);
                    blk[o] |= (v as u8) << b;
                }
                compress(&mut cv, &blk);
            }
            let got: Vec<u8> = cv.iter().flat_map(|w| w.to_be_bytes()).collect();
            assert_eq!(got, Sha512::digest([prefix, row].concat()).to_vec(), "bits {bits}");
        }
    }

    /// A segmented row's message as Δ leaves it (`msg_bit`: the row's bits, constants, and the outer message's digest bits read
    /// from the segments' last outputs), compressed from each chain start's midstate (`chain_start`), is `sha512/row-seg/v1`
    /// of the row (`row_seg_digest`, itself checked against core's vectors here): none, one short, one full and several
    /// segments, a short last one, PoUW's 8 KiB and 16 KiB rows. Its byte map (`byte_at`) places every row bit where `msg_bit`
    /// reads it.
    #[test]
    fn segmented_row_constants_complete_its_digest() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        assert_eq!(crate::hex(&row_seg_digest(0, &[])), crate::hex(&Sha512::digest(row_prefix_seg(0))));
        // core's (`verity.ml.commitments.rowleaf.sha512_row_seg_digest`) of the row of bytes i mod 251 (the last masked to its bits)
        for (bits, want) in [(8193usize, "1009029f51a63cb64c57e4ac641244ecdb437314b19141530f556dc92bf013d2df4479bebd45d7c13c0938ea5e9a8199cf8241638860509ad92986cc0033799e"),
                             (65536, "4269023781ae746c3e03aaac5fc2330ea43cabddc04ea554e6f570ab271605f9d3bec541863e4190e5f87a3cef4a65906c95041588dfcf92c7928bd93193ac43")] {
            let mut row: Vec<u8> = (0..bits.div_ceil(8)).map(|i| (i % 251) as u8).collect();
            if bits % 8 != 0 {
                *row.last_mut().unwrap() &= (1u8 << (bits % 8)) - 1;
            }
            assert_eq!(crate::hex(&row_seg_digest(bits, &row)), want, "bits {bits}");
        }
        for bits in [0usize, 1, 13, 8191, 8192, 8193, 3 * 8192 + 5, 23040, 65536, 131072] {
            let bytes = bits.div_ceil(8);
            let (segs, outer) = seg_shape(bits);
            let comps = segs.iter().map(|x| x.1).sum::<usize>() + outer - 1;
            let port = Port { name: "p".into(), words: bits.div_ceil(16), comps, bits, v2: false, segs: Some(segs.clone()) };
            let rc = row_consts(&port);
            let mut row: Vec<u8> = (0..bytes).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u8 }).collect();
            if bits % 8 != 0 {
                row[bytes - 1] &= (1u8 << (bits % 8)) - 1;
            }
            let mut outs: Vec<[u64; 8]> = Vec::new();
            let mut cv = SHA512_IV;
            for k in 0..=comps {
                if let Some(j) = port.chain_start(k) {
                    cv = if j < segs.len() { rc.segs[j].midstate } else { rc.midstate };
                }
                let mut blk = [0u8; 128];
                for t in 0..1024 {
                    let (o, b) = block_bit(t);
                    let v = match msg_bit(&port, &rc, k, t) {
                        MsgBit::Const(v) => v,
                        MsgBit::Row(i) => {
                            let (kk, oo) = port.byte_at(i / 8);
                            assert_eq!((kk, oo, i % 8), (k, o, b), "bits {bits}: row bit {i}");
                            (row[i / 8] >> (i % 8)) & 1 == 1
                        }
                        MsgBit::Digest(kk, col) => {
                            assert!(kk < k && port.chain_start(kk + 1).is_some(), "bits {bits}: a digest of a segment's last compression");
                            (outs[kk][col / 64] >> (col % 64)) & 1 == 1
                        }
                    };
                    blk[o] |= (v as u8) << b;
                }
                compress(&mut cv, &blk);
                outs.push(cv);
            }
            let got: Vec<u8> = cv.iter().flat_map(|w| w.to_be_bytes()).collect();
            assert_eq!(got, row_seg_digest(bits, &row).to_vec(), "bits {bits}");
            if bits == 65536 {
                assert_eq!((segs.len(), comps + 1, segs[0].1 + outer), (8, 77, 14));
            }
        }
    }
}
