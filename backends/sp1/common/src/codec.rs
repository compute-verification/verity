//! The canonical binary encoding of a batch statement and its witness
//! (`veritor/protocol/proofs/wire.py`), parsed strictly: every index is
//! range-checked, kinds must be sorted and unique, and no byte may be left
//! over.  Integers are big-endian; a `bytes`/`str` is a `u32` length prefix
//! followed by the payload; a list is a `u32` count followed by its items;
//! digests are 32 raw bytes.
//!
//! ```text
//! Statement   = MAGIC "veritor/proofs/statement/v2\0"
//!               str gate_set_id  digest gate_set_digest  u32 width
//!               list<KindProgram> kinds (strictly increasing by kind digest)
//!               list<Obligation>  obligations
//! KindProgram = digest kind  u32 size  list<u32> ports  list<GateOp> gates (len == size)
//!               list<u32> outputs (gate offsets the copy opens, strictly increasing)
//! GateOp      = str op  list<Arg> args
//! Arg         = u8 space (0 = port index, 1 = local gate offset)  u32 value
//! Obligation  = digest session  digest compiled  u64 unit  u64 replay_unit  digest kind
//!               list<CommitmentRef> commitments  list<PositionRef> positions
//!               list<u32> inputs (one per port)  list<u32> outputs (one per opened offset)
//! CommitmentRef = u64 owner_plus_two  digest domain_id  digest root  u64 count
//! PositionRef   = u32 commitment  u64 rank  u64 position  str schema
//!                 u8 has_expected  [bytes expected]
//!
//! StatementV3 = MAGIC "veritor/proofs/statement/v3\0"        (statement format 3)
//!               ... as Statement, with list<PositionEntry> for list<PositionRef>
//! PositionEntry = u8 0  PositionRef
//!               | u8 1  u32 commitment  u64 rank  u64 position  str descriptor
//!                       u64 start  u64 count  u64 stride           (a PositionRange)
//!
//! Witness     = MAGIC "veritor/proofs/witness/v2\0"          (format 1)
//!               list< list< Opening > >    (per obligation, per position)
//! Opening     = bytes value  list<digest> path
//!
//! MultiproofWitness = MAGIC "veritor/proofs/witness/multiproof/v1\0"   (format 2)
//!               list< MultiproofTree >     (per tree, in the statement's tree order)
//! MultiproofTree    = list<bytes> values (one per distinct opened rank, ascending)
//!                     list<digest> siblings (the multiproof, in consumption order)
//!
//! PackedWitness = MAGIC "veritor/proofs/witness/packed/v1\0"       (format 3)
//!               list< list< PackedLeaf > >  (per tree, in the statement's tree order;
//!                                            one leaf per distinct opened rank, ascending)
//! PackedLeaf  = str canonical_id  str dtype  str shape_text  bytes value  list<PathStep> path
//! PathStep    = u8 kind (0 = lift, 1 = sibling)  [digest sibling]
//! ```
//!
//! Statement format 3 (`parse_statement_v3`, `veritor/protocol/proofs/wire.py`
//! `encode_statement_v3`) is format 2 with a tag byte in front of every
//! position entry, so that an entry may be a **range**: `count` consecutive
//! positions reading words `start, start + stride, ...` of one packed leaf,
//! standing for exactly the format-2 positions `PositionRef { commitment,
//! rank, position, schema: "<descriptor>:<index>", expected: None }` in that
//! order (`expand_statement` is that definition; the parser checks the
//! descriptor grammar, `count >= 1`, `stride >= 1`, the last index with
//! checked arithmetic and against the leaf's word count, and that the
//! expanded slot total fits `u32`).  A range never carries a pin.  Format 2
//! is untouched: a format-2 statement cannot hold a range and the two formats
//! have different bytes, hence different digests.  `parse_any_statement`
//! dispatches on the magic.
//!
//! The witness formats are told apart by their magic; `parse_any_witness`
//! dispatches.  Format 2's tree order and the ranks each tree opens are a
//! function of the statement alone (`check::tree_table`); so are format 3's
//! (`check::packed_table`), whose positions all carry a `packed/v1` schema
//! (`packed.rs`): whole leaves under the serving side's framing, one word
//! each read at the statement-fixed index.
//!
//! An optional third input, the **plan**, is untrusted prover advice that
//! spells out that same tree table so the guest can *validate* it in one
//! linear pass instead of computing it (sorting and searching); see
//! `check::validate_plan`.  It is never part of the statement or its digest.
//!
//! ```text
//! Plan = MAGIC "veritor/proofs/plan/tree-table/v1\0"
//!        list<u32> leaf_counts        (one per tree, in the statement's tree order)
//!        list<u32> tree_of            (for every obligation in order, for every
//!                                      commitment it lists: its tree index)
//!        list<u32> leaf_of            (for every obligation in order, for every
//!                                      position: the index of its leaf in its tree)
//! ```

use core::fmt;

pub const STATEMENT_MAGIC: &[u8] = b"veritor/proofs/statement/v2\0";
pub const STATEMENT_V3_MAGIC: &[u8] = b"veritor/proofs/statement/v3\0";
pub const WITNESS_MAGIC: &[u8] = b"veritor/proofs/witness/v2\0";
pub const MULTIPROOF_WITNESS_MAGIC: &[u8] = b"veritor/proofs/witness/multiproof/v1\0";
pub const PACKED_WITNESS_MAGIC: &[u8] = b"veritor/proofs/witness/packed/v1\0";
pub const PLAN_MAGIC: &[u8] = b"veritor/proofs/plan/tree-table/v1\0";

/// A malformed statement or witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodecError(pub String);

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "malformed proof input: {}", self.0)
    }
}

impl std::error::Error for CodecError {}

type Result<T> = core::result::Result<T, CodecError>;

fn err<T>(message: impl Into<String>) -> Result<T> {
    Err(CodecError(message.into()))
}

/// An argument of a gate: a port of the verification unit or an earlier gate of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arg {
    /// Index into `KindProgram::ports` (and so into `Obligation::inputs`).
    Port(u32),
    /// Offset of an earlier gate of the same copy (recomputed, or opened if the
    /// kind lists it in `KindProgram::outputs`).
    Local(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateOp {
    pub op: String,
    pub args: Vec<Arg>,
}

/// The relation of one kind of verification unit, in relative coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KindProgram {
    pub kind: [u8; 32],
    pub size: u32,
    /// The port ordinals the copy reads, ascending.
    pub ports: Vec<u32>,
    pub gates: Vec<GateOp>,
    /// The gate offsets a copy opens (its declared outputs and its source
    /// gates), ascending; every other gate is recomputed by the checker.
    pub outputs: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentRef {
    pub owner_plus_two: u64,
    pub domain_id: [u8; 32],
    pub root: [u8; 32],
    pub count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionRef {
    pub commitment: u32,
    pub rank: u64,
    pub position: u64,
    pub schema: String,
    /// The public input this position must hold (an `in` gate), if any.
    pub expected: Option<Vec<u8>>,
}

/// A statement-format-3 range: `count` consecutive positions reading words
/// `start, start + stride, ...` of the packed leaf at `rank` of `commitment`,
/// whose layout is `descriptor` (`packed/v1:<framing>:u<w><le|be>:<count>`).
/// Its meaning is exactly [`PositionRange::expand`]; the parser has already
/// checked the descriptor, `count >= 1`, `stride >= 1`, and that the last
/// index `start + (count - 1) * stride` exists (checked arithmetic) and is
/// below the leaf's word count.  A range is never pinned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionRange {
    pub commitment: u32,
    pub rank: u64,
    pub position: u64,
    pub descriptor: String,
    pub start: u64,
    pub count: u64,
    pub stride: u64,
}

impl PositionRange {
    /// The word indices the range stands for, in order.
    pub fn indices(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.count).map(move |i| self.start + i * self.stride)
    }

    /// The format-2 positions this range abbreviates: the definition of its meaning.
    pub fn expand(&self) -> Vec<PositionRef> {
        self.indices()
            .map(|index| PositionRef {
                commitment: self.commitment,
                rank: self.rank,
                position: self.position,
                schema: format!("{}:{index}", self.descriptor),
                expected: None,
            })
            .collect()
    }
}

/// One entry of a format-3 obligation's position list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PositionEntry {
    Single(PositionRef),
    Range(PositionRange),
}

/// How many format-2 positions a position entry stands for.
pub trait Slots {
    fn slots(&self) -> u64;
}

impl Slots for PositionRef {
    fn slots(&self) -> u64 {
        1
    }
}

impl Slots for PositionEntry {
    fn slots(&self) -> u64 {
        match self {
            PositionEntry::Single(_) => 1,
            PositionEntry::Range(range) => range.count,
        }
    }
}

/// An obligation, generic in how its positions are spelled: `PositionRef`
/// (format 2, the default) or `PositionEntry` (format 3).  `inputs` and
/// `outputs` index the *expanded* position list in either case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Obligation<P = PositionRef> {
    pub session: [u8; 32],
    pub compiled: [u8; 32],
    pub unit: u64,
    pub replay_unit: u64,
    pub kind: [u8; 32],
    pub commitments: Vec<CommitmentRef>,
    pub positions: Vec<P>,
    /// `inputs[k]`: the slot of the kind's `k`-th read port.
    pub inputs: Vec<u32>,
    /// `outputs[m]`: the slot of the gate at the kind's `m`-th opened offset.
    pub outputs: Vec<u32>,
}

impl<P: Slots> Obligation<P> {
    /// The number of format-2 positions: `positions.len()` once every range is expanded.
    pub fn slot_count(&self) -> u64 {
        self.positions.iter().map(Slots::slots).sum()
    }
}

impl Obligation<PositionEntry> {
    /// The format-2 obligation this one stands for.
    pub fn expand(&self) -> Obligation<PositionRef> {
        let mut positions = Vec::with_capacity(self.slot_count() as usize);
        for entry in &self.positions {
            match entry {
                PositionEntry::Single(position) => positions.push(position.clone()),
                PositionEntry::Range(range) => positions.extend(range.expand()),
            }
        }
        Obligation {
            session: self.session,
            compiled: self.compiled,
            unit: self.unit,
            replay_unit: self.replay_unit,
            kind: self.kind,
            commitments: self.commitments.clone(),
            positions,
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
        }
    }
}

/// A statement, generic in its position spelling like [`Obligation`]:
/// `Statement` is format 2, `Statement<PositionEntry>` format 3.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement<P = PositionRef> {
    pub gate_set_id: String,
    pub gate_set_digest: [u8; 32],
    pub width: u32,
    pub kinds: Vec<KindProgram>,
    pub obligations: Vec<Obligation<P>>,
}

impl<P> Statement<P> {
    /// The program of `kind`, by binary search over the sorted kinds.
    pub fn program(&self, kind: &[u8; 32]) -> Option<&KindProgram> {
        self.kinds
            .binary_search_by(|candidate| candidate.kind.cmp(kind))
            .ok()
            .map(|index| &self.kinds[index])
    }
}

/// The format-2 statement a format-3 statement stands for: every range
/// replaced by its per-word positions.  The one definition of what a range
/// means; `check_openings_ranged` on the format-3 statement and
/// `check_openings_packed` on this expansion decode the same words.
pub fn expand_statement(statement: &Statement<PositionEntry>) -> Statement<PositionRef> {
    Statement {
        gate_set_id: statement.gate_set_id.clone(),
        gate_set_digest: statement.gate_set_digest,
        width: statement.width,
        kinds: statement.kinds.clone(),
        obligations: statement.obligations.iter().map(Obligation::expand).collect(),
    }
}

/// A statement in either format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnyStatement {
    V2(Statement<PositionRef>),
    V3(Statement<PositionEntry>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opening {
    pub value: Vec<u8>,
    pub path: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Witness {
    pub obligations: Vec<Vec<Opening>>,
}

/// One tree of a format-2 witness: the opened values (ascending rank) and the multiproof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiproofTree {
    pub values: Vec<Vec<u8>>,
    pub siblings: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiproofWitness {
    pub trees: Vec<MultiproofTree>,
}

/// One opened packed leaf of a format-3 witness: the parts of its leaf hash
/// under the serving side's framing (`packed::vllm_leaf_hash`) and its path,
/// one step per level, `None` where the node was lifted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedLeaf {
    pub canonical_id: String,
    pub dtype: String,
    pub shape_text: String,
    pub value: Vec<u8>,
    pub path: Vec<Option<[u8; 32]>>,
}

/// Format 3: per tree in the statement's tree order, its opened packed leaves
/// by ascending rank (one per distinct rank the statement's positions name).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedWitness {
    pub trees: Vec<Vec<PackedLeaf>>,
}

/// A witness in any format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnyWitness {
    Openings(Witness),
    Multiproof(MultiproofWitness),
    Packed(PackedWitness),
}

/// An untrusted execution plan: the statement's tree table as the prover
/// proposes it.  Nothing here is trusted; `check::validate_plan` accepts a
/// plan only if it is *the* canonical table of the statement (the one
/// `check::tree_table` computes), so a valid plan changes how the table is
/// obtained, not which openings are authenticated or which values are used.
///
/// The two index lists are flat, in statement order: `tree_of` has one entry
/// per `(obligation, commitment)` and `leaf_of` one per `(obligation,
/// position)`; the statement fixes where each obligation's run starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// `leaf_counts[t]`: how many distinct leaves tree `t` opens.
    pub leaf_counts: Vec<u32>,
    pub tree_of: Vec<u32>,
    pub leaf_of: Vec<u32>,
}

struct Reader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Reader { data, offset: 0 }
    }

    fn take(&mut self, count: usize, what: &str) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| CodecError("overflow".into()))?;
        if end > self.data.len() {
            return err(format!("truncated while reading {what}"));
        }
        let slice = &self.data[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn magic(&mut self, expected: &[u8]) -> Result<()> {
        if self.take(expected.len(), "magic")? != expected {
            return err("wrong magic");
        }
        Ok(())
    }

    fn u8(&mut self, what: &str) -> Result<u8> {
        Ok(self.take(1, what)?[0])
    }

    fn u32(&mut self, what: &str) -> Result<u32> {
        let bytes = self.take(4, what)?;
        Ok(u32::from_be_bytes(bytes.try_into().expect("four bytes")))
    }

    fn u64(&mut self, what: &str) -> Result<u64> {
        let bytes = self.take(8, what)?;
        Ok(u64::from_be_bytes(bytes.try_into().expect("eight bytes")))
    }

    fn digest(&mut self, what: &str) -> Result<[u8; 32]> {
        let bytes = self.take(32, what)?;
        Ok(bytes.try_into().expect("32 bytes"))
    }

    fn bytes(&mut self, what: &str) -> Result<&'a [u8]> {
        let length = self.u32(what)? as usize;
        self.take(length, what)
    }

    fn string(&mut self, what: &str) -> Result<String> {
        let bytes = self.bytes(what)?;
        core::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| CodecError(format!("{what} is not UTF-8")))
    }

    fn count(&mut self, what: &str) -> Result<usize> {
        let count = self.u32(what)? as usize;
        // Every item is at least one byte; refuse counts the buffer cannot hold.
        if count > self.data.len() - self.offset {
            return err(format!("{what} count {count} exceeds the remaining bytes"));
        }
        Ok(count)
    }

    fn finish(self, what: &str) -> Result<()> {
        if self.offset != self.data.len() {
            return err(format!(
                "{} trailing bytes after the {what}",
                self.data.len() - self.offset
            ));
        }
        Ok(())
    }
}

fn parse_gate(reader: &mut Reader<'_>, offset: u32, ports: usize) -> Result<GateOp> {
    let op = reader.string("gate op")?;
    let count = reader.count("gate args")?;
    let mut args = Vec::with_capacity(count);
    for _ in 0..count {
        let space = reader.u8("arg space")?;
        let value = reader.u32("arg value")?;
        let arg = match space {
            0 => {
                if value as usize >= ports {
                    return err(format!("gate {offset} reads port index {value} of {ports}"));
                }
                Arg::Port(value)
            }
            1 => {
                if value >= offset {
                    return err(format!(
                        "gate {offset} reads local offset {value}, not earlier"
                    ));
                }
                Arg::Local(value)
            }
            other => return err(format!("unknown argument space {other}")),
        };
        args.push(arg);
    }
    Ok(GateOp { op, args })
}

fn parse_program(reader: &mut Reader<'_>) -> Result<KindProgram> {
    let kind = reader.digest("kind digest")?;
    let size = reader.u32("kind size")?;
    let port_count = reader.count("ports")?;
    let mut ports = Vec::with_capacity(port_count);
    for index in 0..port_count {
        let ordinal = reader.u32("port ordinal")?;
        if index > 0 && ports[index - 1] >= ordinal {
            return err("ports must be strictly increasing");
        }
        ports.push(ordinal);
    }
    let gate_count = reader.count("gates")?;
    if gate_count != size as usize {
        return err(format!("kind declares {size} gates but lists {gate_count}"));
    }
    let mut gates = Vec::with_capacity(gate_count);
    for offset in 0..gate_count {
        gates.push(parse_gate(reader, offset as u32, port_count)?);
    }
    let output_count = reader.count("outputs")?;
    let mut outputs = Vec::with_capacity(output_count);
    for index in 0..output_count {
        let offset = reader.u32("output offset")?;
        if offset >= size {
            return err(format!("kind opens gate offset {offset} of {size}"));
        }
        if index > 0 && outputs[index - 1] >= offset {
            return err("kind outputs must be strictly increasing");
        }
        outputs.push(offset);
    }
    Ok(KindProgram {
        kind,
        size,
        ports,
        gates,
        outputs,
    })
}

fn parse_indices(reader: &mut Reader<'_>, what: &str, bound: usize) -> Result<Vec<u32>> {
    let count = reader.count(what)?;
    let mut indices = Vec::with_capacity(count);
    for _ in 0..count {
        let index = reader.u32(what)?;
        if index as usize >= bound {
            return err(format!("{what} index {index} is out of range ({bound})"));
        }
        indices.push(index);
    }
    Ok(indices)
}

fn parse_position(reader: &mut Reader<'_>, commitment_count: usize) -> Result<PositionRef> {
    let commitment = reader.u32("position commitment")?;
    if commitment as usize >= commitment_count {
        return err("position names a commitment the obligation lacks");
    }
    let rank = reader.u64("rank")?;
    let position = reader.u64("position")?;
    let schema = reader.string("schema")?;
    let expected = match reader.u8("expected flag")? {
        0 => None,
        1 => Some(reader.bytes("expected value")?.to_vec()),
        other => return err(format!("bad expected flag {other}")),
    };
    Ok(PositionRef {
        commitment,
        rank,
        position,
        schema,
        expected,
    })
}

/// A format-3 range entry, validated: descriptor grammar, `count >= 1`,
/// `stride >= 1`, last index by checked arithmetic and below the leaf's word count.
fn parse_range(reader: &mut Reader<'_>, commitment_count: usize) -> Result<PositionRange> {
    let commitment = reader.u32("range commitment")?;
    if commitment as usize >= commitment_count {
        return err("range names a commitment the obligation lacks");
    }
    let rank = reader.u64("range rank")?;
    let position = reader.u64("range position")?;
    let descriptor = reader.string("range descriptor")?;
    let words = crate::packed::parse_leaf_descriptor(&descriptor)
        .ok_or_else(|| CodecError("range descriptor is not a packed/v1 leaf descriptor".into()))?
        .count;
    let start = reader.u64("range start")?;
    let count = reader.u64("range count")?;
    let stride = reader.u64("range stride")?;
    if count == 0 || stride == 0 {
        return err("a range has at least one word and a stride of at least one");
    }
    let last = (count - 1)
        .checked_mul(stride)
        .and_then(|span| start.checked_add(span))
        .ok_or_else(|| CodecError("range arithmetic overflows".into()))?;
    if last >= words {
        return err(format!("range reaches word {last} of a leaf of {words} words"));
    }
    Ok(PositionRange {
        commitment,
        rank,
        position,
        descriptor,
        start,
        count,
        stride,
    })
}

fn parse_entry(reader: &mut Reader<'_>, commitment_count: usize) -> Result<PositionEntry> {
    match reader.u8("position entry tag")? {
        0 => Ok(PositionEntry::Single(parse_position(reader, commitment_count)?)),
        1 => Ok(PositionEntry::Range(parse_range(reader, commitment_count)?)),
        other => err(format!("bad position entry tag {other}")),
    }
}

fn parse_obligation_with<P: Slots>(
    reader: &mut Reader<'_>,
    statement_kinds: &[KindProgram],
    parse_entry: fn(&mut Reader<'_>, usize) -> Result<P>,
) -> Result<Obligation<P>> {
    let session = reader.digest("session")?;
    let compiled = reader.digest("compiled digest")?;
    let unit = reader.u64("unit")?;
    let replay_unit = reader.u64("replay unit")?;
    let kind = reader.digest("obligation kind")?;
    let program = statement_kinds
        .binary_search_by(|candidate| candidate.kind.cmp(&kind))
        .ok()
        .map(|index| &statement_kinds[index])
        .ok_or_else(|| CodecError("obligation names a kind the statement lacks".into()))?;
    let commitment_count = reader.count("commitments")?;
    let mut commitments = Vec::with_capacity(commitment_count);
    for _ in 0..commitment_count {
        commitments.push(CommitmentRef {
            owner_plus_two: reader.u64("owner")?,
            domain_id: reader.digest("domain id")?,
            root: reader.digest("root")?,
            count: reader.u64("commitment count")?,
        });
    }
    let position_count = reader.count("positions")?;
    let mut positions = Vec::with_capacity(position_count);
    let mut slots: u64 = 0;
    for _ in 0..position_count {
        let entry = parse_entry(reader, commitment_count)?;
        slots = slots
            .checked_add(entry.slots())
            .filter(|&total| total < 1 << 32)
            .ok_or_else(|| CodecError("obligation has too many positions".into()))?;
        positions.push(entry);
    }
    let position_count = slots as usize;
    let inputs = parse_indices(reader, "inputs", position_count)?;
    if inputs.len() != program.ports.len() {
        return err(format!(
            "obligation binds {} inputs but its kind reads {} ports",
            inputs.len(),
            program.ports.len()
        ));
    }
    let outputs = parse_indices(reader, "outputs", position_count)?;
    if outputs.len() != program.outputs.len() {
        return err(format!(
            "obligation binds {} outputs but its kind opens {}",
            outputs.len(),
            program.outputs.len()
        ));
    }
    let mut distinct = outputs.clone();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() != outputs.len() {
        return err("obligation outputs must be distinct positions");
    }
    Ok(Obligation {
        session,
        compiled,
        unit,
        replay_unit,
        kind,
        commitments,
        positions,
        inputs,
        outputs,
    })
}

fn parse_statement_with<P: Slots>(
    data: &[u8],
    magic: &[u8],
    parse_entry: fn(&mut Reader<'_>, usize) -> Result<P>,
) -> Result<Statement<P>> {
    let mut reader = Reader::new(data);
    reader.magic(magic)?;
    let gate_set_id = reader.string("gate set id")?;
    let gate_set_digest = reader.digest("gate set digest")?;
    let width = reader.u32("width")?;
    let kind_count = reader.count("kinds")?;
    let mut kinds: Vec<KindProgram> = Vec::with_capacity(kind_count);
    for index in 0..kind_count {
        let program = parse_program(&mut reader)?;
        if index > 0 && kinds[index - 1].kind >= program.kind {
            return err("kinds must be strictly increasing by digest");
        }
        kinds.push(program);
    }
    let obligation_count = reader.count("obligations")?;
    let mut obligations = Vec::with_capacity(obligation_count);
    for _ in 0..obligation_count {
        obligations.push(parse_obligation_with(&mut reader, &kinds, parse_entry)?);
    }
    reader.finish("statement")?;
    Ok(Statement {
        gate_set_id,
        gate_set_digest,
        width,
        kinds,
        obligations,
    })
}

/// Parse a canonical format-2 statement.
pub fn parse_statement(data: &[u8]) -> Result<Statement> {
    parse_statement_with(data, STATEMENT_MAGIC, parse_position)
}

/// Parse a canonical format-3 statement (ranges validated, not expanded).
pub fn parse_statement_v3(data: &[u8]) -> Result<Statement<PositionEntry>> {
    parse_statement_with(data, STATEMENT_V3_MAGIC, parse_entry)
}

/// Parse a statement in either format, by its magic.
pub fn parse_any_statement(data: &[u8]) -> Result<AnyStatement> {
    if data.starts_with(STATEMENT_MAGIC) {
        parse_statement(data).map(AnyStatement::V2)
    } else if data.starts_with(STATEMENT_V3_MAGIC) {
        parse_statement_v3(data).map(AnyStatement::V3)
    } else {
        err("not a statement of a known format")
    }
}

/// Parse a canonical format-1 witness.
pub fn parse_witness(data: &[u8]) -> Result<Witness> {
    let mut reader = Reader::new(data);
    reader.magic(WITNESS_MAGIC)?;
    let obligation_count = reader.count("witness obligations")?;
    let mut obligations = Vec::with_capacity(obligation_count);
    for _ in 0..obligation_count {
        let opening_count = reader.count("openings")?;
        let mut openings = Vec::with_capacity(opening_count);
        for _ in 0..opening_count {
            let value = reader.bytes("opening value")?.to_vec();
            let depth = reader.count("path")?;
            let mut path = Vec::with_capacity(depth);
            for _ in 0..depth {
                path.push(reader.digest("path digest")?);
            }
            openings.push(Opening { value, path });
        }
        obligations.push(openings);
    }
    reader.finish("witness")?;
    Ok(Witness { obligations })
}

/// Parse a canonical format-2 witness.
pub fn parse_multiproof_witness(data: &[u8]) -> Result<MultiproofWitness> {
    let mut reader = Reader::new(data);
    reader.magic(MULTIPROOF_WITNESS_MAGIC)?;
    let tree_count = reader.count("witness trees")?;
    let mut trees = Vec::with_capacity(tree_count);
    for _ in 0..tree_count {
        let value_count = reader.count("opened values")?;
        let mut values = Vec::with_capacity(value_count);
        for _ in 0..value_count {
            values.push(reader.bytes("opened value")?.to_vec());
        }
        let sibling_count = reader.count("siblings")?;
        let mut siblings = Vec::with_capacity(sibling_count);
        for _ in 0..sibling_count {
            siblings.push(reader.digest("sibling digest")?);
        }
        trees.push(MultiproofTree { values, siblings });
    }
    reader.finish("witness")?;
    Ok(MultiproofWitness { trees })
}

/// Parse a canonical format-3 witness.
pub fn parse_packed_witness(data: &[u8]) -> Result<PackedWitness> {
    let mut reader = Reader::new(data);
    reader.magic(PACKED_WITNESS_MAGIC)?;
    let tree_count = reader.count("witness trees")?;
    let mut trees = Vec::with_capacity(tree_count);
    for _ in 0..tree_count {
        let leaf_count = reader.count("packed leaves")?;
        let mut leaves = Vec::with_capacity(leaf_count);
        for _ in 0..leaf_count {
            let canonical_id = reader.string("leaf id")?;
            let dtype = reader.string("leaf dtype")?;
            let shape_text = reader.string("leaf shape")?;
            let value = reader.bytes("leaf value")?.to_vec();
            let depth = reader.count("path")?;
            let mut path = Vec::with_capacity(depth);
            for _ in 0..depth {
                match reader.u8("path step kind")? {
                    0 => path.push(None),
                    1 => path.push(Some(reader.digest("sibling digest")?)),
                    kind => return err(format!("bad path step kind {kind}")),
                }
            }
            leaves.push(PackedLeaf {
                canonical_id,
                dtype,
                shape_text,
                value,
                path,
            });
        }
        trees.push(leaves);
    }
    reader.finish("witness")?;
    Ok(PackedWitness { trees })
}

/// Parse a witness in whichever format its magic names.
pub fn parse_any_witness(data: &[u8]) -> Result<AnyWitness> {
    if data.starts_with(MULTIPROOF_WITNESS_MAGIC) {
        parse_multiproof_witness(data).map(AnyWitness::Multiproof)
    } else if data.starts_with(PACKED_WITNESS_MAGIC) {
        parse_packed_witness(data).map(AnyWitness::Packed)
    } else {
        parse_witness(data).map(AnyWitness::Openings)
    }
}

/// A `list<u32>` read four bytes at a time (the plan is nothing but these).
fn parse_u32_list(reader: &mut Reader<'_>, what: &str) -> Result<Vec<u32>> {
    let count = reader.count(what)?;
    let bytes = reader.take(count.checked_mul(4).ok_or_else(|| CodecError("overflow".into()))?, what)?;
    let mut items = Vec::with_capacity(count);
    let mut at = 0;
    while at < bytes.len() {
        items.push(u32::from_be_bytes([
            bytes[at],
            bytes[at + 1],
            bytes[at + 2],
            bytes[at + 3],
        ]));
        at += 4;
    }
    Ok(items)
}

/// Parse a plan.  Only its framing is checked here; whether it is the
/// statement's table is `check::validate_plan`'s question.
pub fn parse_plan(data: &[u8]) -> Result<Plan> {
    let mut reader = Reader::new(data);
    reader.magic(PLAN_MAGIC)?;
    let leaf_counts = parse_u32_list(&mut reader, "plan leaf counts")?;
    let tree_of = parse_u32_list(&mut reader, "plan tree indices")?;
    let leaf_of = parse_u32_list(&mut reader, "plan leaf indices")?;
    reader.finish("plan")?;
    Ok(Plan {
        leaf_counts,
        tree_of,
        leaf_of,
    })
}
