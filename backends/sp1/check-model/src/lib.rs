//! The plan validator, the planned operand assembly and the `veritor.int8-q7`
//! interpreter of `zk/sp1/common`, restructured into the subset of Rust that
//! Aeneas translates (lane L3, Target D assessment).
//!
//! Every function here mirrors one in `zk/sp1/common/src/{check,gateset}.rs`
//! (named in its doc comment).  The changes from the production code:
//!
//! * `no_std` + `alloc`; `Vec<T>` and slices stay (Aeneas models them).
//! * No early `return` / `?` inside loops: every loop carries an
//!   `error: Option<E>` accumulator and stops on the first error.  Same first
//!   error, same result.
//! * Index-based `while` loops instead of iterator adapters
//!   (`iter().zip().enumerate()`, `find`, `any`).
//! * Errors are payload-free enums instead of `String`-carrying variants.
//! * Leaf schemas are `u8` codes and commitment digests `u64` identities:
//!   the validator only compares them for equality, so any injective
//!   encoding gives the same verdicts.
//! * Gate ops are the resolved `Op` enum rather than names: the model is of
//!   the program after `GateSet::op` resolved every name (the production
//!   code resolves at every gate; an unknown name is a `CheckError::UnknownOp`
//!   there and is unrepresentable here).
//! * The gate set is fixed to `veritor.int8-q7` at width 32, the only set
//!   the campaign uses; the toy sets are left out.
#![no_std]
#![allow(clippy::needless_range_loop)]

extern crate alloc;
use alloc::vec::Vec;

/// `gateset::Op` restricted to `veritor.int8-q7`.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub enum Op {
    Add32,
    In,
    Mul8,
    RequantS8,
    RequantU7,
    Weight,
}

impl Op {
    /// `Op::is_source`.
    pub fn is_source(self) -> bool {
        matches!(self, Op::In | Op::Weight)
    }

    /// `Op::arity`.
    pub fn arity(self) -> usize {
        match self {
            Op::In | Op::Weight => 0,
            Op::RequantU7 | Op::RequantS8 => 1,
            _ => 2,
        }
    }

    /// `GateSet::arg_widths` for `veritor.int8-q7`.
    pub fn arg_width(self) -> u32 {
        match self {
            Op::Mul8 => 8,
            _ => 32,
        }
    }
}

/// `gateset::mask` (`width <= 64`).
pub fn mask(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// `GateSet::evaluate` for `veritor.int8-q7`: `None` when the argument
/// count or a width is not the gate's.  `a` and `b` are the first two
/// arguments (0 when absent), exactly as the production code reads them.
pub fn evaluate(op: Op, args: &[u64]) -> Option<u64> {
    if args.len() != op.arity() {
        return None;
    }
    let width = op.arg_width();
    let mut i = 0;
    let mut too_wide = false;
    while i < args.len() {
        if args[i] > mask(width) {
            too_wide = true;
        }
        i += 1;
    }
    if too_wide {
        return None;
    }
    let a = if args.is_empty() { 0 } else { args[0] };
    let b = if args.len() < 2 { 0 } else { args[1] };
    match op {
        Op::Mul8 => {
            let x = a as u8 as i8 as i32;
            let y = b as u8 as i8 as i32;
            Some((x * y) as u32 as u64)
        }
        Op::Add32 => Some((a as u32).wrapping_add(b as u32) as u64),
        Op::RequantU7 => {
            let s = (a as u32 as i32) >> 7;
            let c = if s < 0 {
                0
            } else if s > 127 {
                127
            } else {
                s
            };
            Some(c as u64)
        }
        Op::RequantS8 => {
            let s = (a as u32 as i32) >> 7;
            let c = if s < -128 {
                -128
            } else if s > 127 {
                127
            } else {
                s
            };
            Some(c as i8 as u8 as u64)
        }
        Op::In | Op::Weight => None,
    }
}

/// `codec::Arg`.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub enum Arg {
    Port(u32),
    Local(u32),
}

/// `codec::GateOp` with the op resolved.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Gate {
    pub op: Op,
    pub args: Vec<Arg>,
}

/// `codec::KindProgram` (ports are implicit: `Arg::Port(k)` indexes `Obligation::inputs`).
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Program {
    pub gates: Vec<Gate>,
    /// Strictly increasing gate offsets whose values the obligation opens.
    pub outputs: Vec<u32>,
}

/// `codec::CommitmentRef` with the digests as identities.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Commitment {
    pub domain_id: u64,
    pub root: u64,
    pub count: u64,
}

/// `codec::PositionRef` (the pinned `expected` bytes are left to the hashing side).
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Position {
    pub commitment: u32,
    pub rank: u64,
    pub position: u64,
    pub schema: u8,
}

/// `codec::Obligation`.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Obligation {
    /// Index into `Statement::programs` (the parser resolved the kind digest).
    pub program: u32,
    pub commitments: Vec<Commitment>,
    pub positions: Vec<Position>,
    /// Slot (index into `positions`) of every port.
    pub inputs: Vec<u32>,
    /// Slot of every opened gate, aligned with `Program::outputs`.
    pub outputs: Vec<u32>,
}

/// `codec::Statement` after parsing.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Statement {
    pub programs: Vec<Program>,
    pub obligations: Vec<Obligation>,
}

/// `codec::Plan`: the untrusted tree table.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Plan {
    pub leaf_counts: Vec<u32>,
    pub tree_of: Vec<u32>,
    pub leaf_of: Vec<u32>,
}

/// `check::PlanLeaf`.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Leaf {
    pub rank: u64,
    pub position: u64,
    pub schema: u8,
    pub obligation: u32,
    pub slot: u32,
}

/// `check::PlanTree`.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Tree {
    pub commitment: Commitment,
    pub leaves: Vec<Leaf>,
}

/// `check::PlanTable`.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub struct Table {
    pub trees: Vec<Tree>,
}

/// `CheckError::Plan(&'static str)`, one variant per message.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub enum PlanError {
    Shorter,
    Longer,
    TreeOutOfRange,
    NotByFirstAppearance,
    TreeDefinedTwice,
    OtherCommitmentsTree,
    LeafOutOfRange,
    OtherCoordinates,
    TreeNeverNamed,
    LeafNeverNamed,
    NotAscending,
}

/// `check::same_tree`.
pub fn same_tree(a: &Commitment, b: &Commitment) -> bool {
    a.domain_id == b.domain_id && a.root == b.root && a.count == b.count
}

/// Whether one of the first `defined_trees` trees already is `commitment`'s
/// tree (the `u` loop of `check::validate_plan`: no tree is defined twice).
fn defined_before(
    commitments: &[Option<Commitment>],
    defined_trees: usize,
    commitment: &Commitment,
) -> bool {
    let mut found = false;
    let mut u = 0usize;
    while !found && u < defined_trees {
        // (the `match` lives in its own function: Aeneas does not expand an
        // enumeration read inside a loop body)
        found = is_defined_as(commitments[u], commitment);
        u += 1;
    }
    found
}

/// Whether a (possibly still undefined) tree slot is `commitment`'s tree.
fn is_defined_as(existing: Option<Commitment>, commitment: &Commitment) -> bool {
    match existing {
        Some(tree) => same_tree(&tree, commitment),
        None => false,
    }
}

/// One commitment of one obligation against the plan's tree numbering
/// (the inner `c` loop of `check::validate_plan`).
fn plan_commitment(
    statement: &Statement,
    index: usize,
    c: usize,
    tree: usize,
    commitments: &mut Vec<Option<Commitment>>,
    defined_trees: &mut usize,
) -> Option<PlanError> {
    if tree >= commitments.len() {
        return Some(PlanError::TreeOutOfRange);
    }
    let commitment = statement.obligations[index].commitments[c];
    match commitments[tree] {
        None => {
            if tree != *defined_trees {
                Some(PlanError::NotByFirstAppearance)
            } else if defined_before(commitments, *defined_trees, &commitment) {
                Some(PlanError::TreeDefinedTwice)
            } else {
                commitments[tree] = Some(commitment);
                *defined_trees += 1;
                None
            }
        }
        Some(existing) => {
            if same_tree(&existing, &commitment) {
                None
            } else {
                Some(PlanError::OtherCommitmentsTree)
            }
        }
    }
}

/// The commitments of one obligation (`tree_base` is where its row starts in `plan.tree_of`).
fn plan_commitments(
    statement: &Statement,
    index: usize,
    plan: &Plan,
    tree_base: usize,
    commitments: &mut Vec<Option<Commitment>>,
    defined_trees: &mut usize,
) -> Option<PlanError> {
    let count = statement.obligations[index].commitments.len();
    let mut error: Option<PlanError> = None;
    let mut c = 0;
    while c < count && error.is_none() {
        let tree = plan.tree_of[tree_base + c] as usize;
        error = plan_commitment(statement, index, c, tree, commitments, defined_trees);
        c += 1;
    }
    error
}

/// Record or check one leaf of one tree (the body of the inner `slot` loop).
fn plan_leaf(leaves: &mut Vec<Vec<Option<Leaf>>>, tree: usize, k: usize, item: Leaf) -> Option<PlanError> {
    if k >= leaves[tree].len() {
        return Some(PlanError::LeafOutOfRange);
    }
    match leaves[tree][k] {
        None => {
            leaves[tree][k] = Some(item);
            None
        }
        Some(existing) => {
            if existing.rank != item.rank
                || existing.position != item.position
                || existing.schema != item.schema
            {
                Some(PlanError::OtherCoordinates)
            } else {
                None
            }
        }
    }
}

/// The positions of one obligation against the plan's leaf numbering.
fn plan_positions(
    statement: &Statement,
    index: usize,
    plan: &Plan,
    tree_base: usize,
    leaf_base: usize,
    leaves: &mut Vec<Vec<Option<Leaf>>>,
) -> Option<PlanError> {
    let count = statement.obligations[index].positions.len();
    let mut error: Option<PlanError> = None;
    let mut slot = 0;
    while slot < count && error.is_none() {
        let position = statement.obligations[index].positions[slot];
        let tree = plan.tree_of[tree_base + position.commitment as usize] as usize;
        let k = plan.leaf_of[leaf_base + slot] as usize;
        let item = Leaf {
            rank: position.rank,
            position: position.position,
            schema: position.schema,
            obligation: index as u32,
            slot: slot as u32,
        };
        error = plan_leaf(leaves, tree, k, item);
        slot += 1;
    }
    error
}

/// The leaves of one tree in plan order, checked to be all named and strictly ascending by rank.
fn ordered_leaves(row: &Vec<Option<Leaf>>) -> Result<Vec<Leaf>, PlanError> {
    let mut ordered: Vec<Leaf> = Vec::new();
    let mut error: Option<PlanError> = None;
    let mut k = 0;
    while k < row.len() && error.is_none() {
        match row[k] {
            None => error = Some(PlanError::LeafNeverNamed),
            Some(item) => {
                if k > 0 && ordered[k - 1].rank >= item.rank {
                    error = Some(PlanError::NotAscending);
                } else {
                    ordered.push(item);
                }
            }
        }
        k += 1;
    }
    match error {
        Some(e) => Err(e),
        None => Ok(ordered),
    }
}

/// `check::validate_plan`: decide whether `plan` is the canonical tree table
/// of `statement` and return it if so.  Same checks, same order, same first
/// error as the production function.
pub fn validate_plan(statement: &Statement, plan: &Plan) -> Result<Table, PlanError> {
    let tree_count = plan.leaf_counts.len();
    let mut commitments: Vec<Option<Commitment>> = Vec::new();
    let mut leaves: Vec<Vec<Option<Leaf>>> = Vec::new();
    let mut t = 0;
    while t < tree_count {
        commitments.push(None);
        let mut row: Vec<Option<Leaf>> = Vec::new();
        let mut k = 0;
        while k < plan.leaf_counts[t] as usize {
            row.push(None);
            k += 1;
        }
        leaves.push(row);
        t += 1;
    }
    let mut error: Option<PlanError> = None;
    let mut defined_trees = 0usize;
    let mut tree_cursor = 0usize;
    let mut leaf_cursor = 0usize;
    let mut index = 0usize;
    while index < statement.obligations.len() && error.is_none() {
        let commitment_count = statement.obligations[index].commitments.len();
        let position_count = statement.obligations[index].positions.len();
        if plan.tree_of.len() - tree_cursor < commitment_count
            || plan.leaf_of.len() - leaf_cursor < position_count
        {
            error = Some(PlanError::Shorter);
        } else {
            let tree_base = tree_cursor;
            let leaf_base = leaf_cursor;
            tree_cursor += commitment_count;
            leaf_cursor += position_count;
            error = plan_commitments(statement, index, plan, tree_base, &mut commitments, &mut defined_trees);
            if error.is_none() {
                error = plan_positions(statement, index, plan, tree_base, leaf_base, &mut leaves);
            }
        }
        index += 1;
    }
    if let Some(e) = error {
        return Err(e);
    }
    if tree_cursor != plan.tree_of.len() || leaf_cursor != plan.leaf_of.len() {
        return Err(PlanError::Longer);
    }
    if defined_trees != tree_count {
        return Err(PlanError::TreeNeverNamed);
    }
    let mut trees: Vec<Tree> = Vec::new();
    let mut t = 0;
    while t < tree_count && error.is_none() {
        match commitments[t] {
            None => error = Some(PlanError::TreeNeverNamed),
            Some(commitment) => match ordered_leaves(&leaves[t]) {
                Err(e) => error = Some(e),
                Ok(ordered) => trees.push(Tree {
                    commitment,
                    leaves: ordered,
                }),
            },
        }
        t += 1;
    }
    match error {
        Some(e) => Err(e),
        None => Ok(Table { trees }),
    }
}

/// The word of leaf `k` of tree `tree`.
fn word_at(tree_words: &[Vec<u64>], tree: usize, k: usize) -> u64 {
    tree_words[tree][k]
}

/// The operands of one obligation from the plan's leaf numbering.
fn obligation_operands(
    statement: &Statement,
    index: usize,
    plan: &Plan,
    tree_base: usize,
    leaf_base: usize,
    tree_words: &[Vec<u64>],
) -> Vec<u64> {
    let count = statement.obligations[index].positions.len();
    let mut values: Vec<u64> = Vec::new();
    let mut slot = 0;
    while slot < count {
        let position = statement.obligations[index].positions[slot];
        let tree = plan.tree_of[tree_base + position.commitment as usize] as usize;
        let k = plan.leaf_of[leaf_base + slot] as usize;
        values.push(word_at(tree_words, tree, k));
        slot += 1;
    }
    values
}

/// The operand-assembly half of `check::check_openings_planned`: given the
/// validated table and, per tree, the word decoded from each distinct leaf,
/// hand every obligation position the word of the leaf it names.  (Hashing
/// and byte decoding are on the other side of `decode_schema` / `sha2` and
/// are not modelled here.)
pub fn planned_operands(
    statement: &Statement,
    plan: &Plan,
    tree_words: &[Vec<u64>],
) -> Vec<Vec<u64>> {
    let mut decoded: Vec<Vec<u64>> = Vec::new();
    let mut tree_cursor = 0usize;
    let mut leaf_cursor = 0usize;
    let mut index = 0usize;
    while index < statement.obligations.len() {
        let tree_base = tree_cursor;
        let leaf_base = leaf_cursor;
        tree_cursor += statement.obligations[index].commitments.len();
        leaf_cursor += statement.obligations[index].positions.len();
        decoded.push(obligation_operands(statement, index, plan, tree_base, leaf_base, tree_words));
        index += 1;
    }
    decoded
}

/// `check::CheckError`, the variants the relation pass raises.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "debug", derive(Debug))]
pub enum CheckError {
    Arity,
    SourceNotOpened,
    Argument,
    Relation,
}

/// Operand assembly for one gate: the gate's arguments from the opened
/// values (`Arg::Port`) or earlier gates of the copy (`Arg::Local`).
fn gather_args(gate: &Gate, values: &[u64], inputs: &[u32], local: &[u64]) -> Vec<u64> {
    let mut args: Vec<u64> = Vec::new();
    let mut a = 0usize;
    while a < gate.args.len() {
        let word = match gate.args[a] {
            Arg::Port(k) => values[inputs[k as usize] as usize],
            Arg::Local(j) => local[j as usize],
        };
        args.push(word);
        a += 1;
    }
    args
}

/// One gate of one obligation: evaluate it (or take its opened value when it
/// is a source), compare with the opened value when the kind opens it, and
/// append the copy's value to `local`.
fn gate_step(
    gate: &Gate,
    slot: Option<usize>,
    values: &[u64],
    inputs: &[u32],
    local: &mut Vec<u64>,
) -> Option<CheckError> {
    let op = gate.op;
    if gate.args.len() != op.arity() {
        return Some(CheckError::Arity);
    }
    if op.is_source() {
        return match slot {
            None => Some(CheckError::SourceNotOpened),
            Some(s) => {
                local.push(values[s]);
                None
            }
        };
    }
    let args = gather_args(gate, values, inputs, local);
    match evaluate(op, &args) {
        None => Some(CheckError::Argument),
        Some(computed) => match slot {
            None => {
                local.push(computed);
                None
            }
            Some(s) => {
                if values[s] != computed {
                    Some(CheckError::Relation)
                } else {
                    local.push(values[s]);
                    None
                }
            }
        },
    }
}

/// The relation pass for one obligation (the inner loop of `check::check_relations`).
fn check_obligation(program: &Program, obligation: &Obligation, values: &[u64]) -> Option<CheckError> {
    let mut error: Option<CheckError> = None;
    let mut local: Vec<u64> = Vec::new();
    let mut next_output = 0usize;
    let mut offset = 0usize;
    while offset < program.gates.len() && error.is_none() {
        // `program.outputs` is strictly increasing, so one cursor finds the opened slot.
        let mut slot: Option<usize> = None;
        if next_output < program.outputs.len() && program.outputs[next_output] as usize == offset {
            slot = Some(obligation.outputs[next_output] as usize);
            next_output += 1;
        }
        error = gate_step(&program.gates[offset], slot, values, &obligation.inputs, &mut local);
        offset += 1;
    }
    error
}

/// `check::check_relations`: recompute every obligation's gates from its
/// operands; every opened gate must agree with the recomputation.
pub fn check_relations(statement: &Statement, decoded: &[Vec<u64>]) -> Result<(), CheckError> {
    let mut error: Option<CheckError> = None;
    let mut index = 0usize;
    while index < statement.obligations.len() && error.is_none() {
        let program = statement.obligations[index].program as usize;
        error = check_obligation(
            &statement.programs[program],
            &statement.obligations[index],
            &decoded[index],
        );
        index += 1;
    }
    match error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// `CheckFast` restricted to what is modelled: validate the plan, assemble
/// the operands from the per-tree decoded words, run the relation pass.
pub fn check_fast(
    statement: &Statement,
    plan: &Plan,
    tree_words: &[Vec<u64>],
) -> Result<(), CheckError> {
    match validate_plan(statement, plan) {
        Err(_) => Err(CheckError::Argument),
        Ok(_table) => {
            let decoded = planned_operands(statement, plan, tree_words);
            check_relations(statement, &decoded)
        }
    }
}
