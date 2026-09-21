//! Pass 1 of the verification kernel: authenticate and decode every opening
//! a witness supplies for a statement.
//!
//! Mirrors `TransparentBackend.verify` in `veritor/protocol/proofs/transparent.py`:
//! every opening is authenticated under its commitment with the exact leaf
//! and node framing, decoded canonically at the width its own leaf schema
//! names, and compared with the public input where the statement pins one.
//! Only a copy's inputs and outputs are opened; the gates between are never
//! committed, so pass 2's recomputation ([`super::relations`]) is what stands
//! for them.  The two passes are separate so the guest can attribute cycles
//! to each.
//!
//! The opening pass comes in four shapes for the witness formats of
//! `codec`: [`check_openings`] folds one authentication path per position;
//! [`check_openings_multi`] groups the positions by tree ([`tree_table`]),
//! reconstructs every root once from a Merkle multiproof and hashes every
//! distinct leaf once, however many obligations open it.  Both return the same
//! decoded values for the same tree contents.  [`check_openings_packed`] is
//! the pass for positions whose schema is `packed/v1` (`packed.rs`): whole
//! leaves under the serving side's framing `verity-vllm/leaf/v1`,
//! authenticated once per distinct rank, with each position's word read at
//! its statement-fixed index.  [`check_openings_ranged`] is the same packed
//! pass driven by a format-3 statement's position *entries* (a range names
//! many words of one leaf), and [`check_openings_planned`] is the multiproof
//! pass with the tree table validated from untrusted advice
//! ([`validate_plan`]) instead of computed.
//!
//! The three tree tables ([`tree_table`], [`packed_table`], [`ranged_table`])
//! share one grouping discipline, implemented once: trees are numbered by
//! first appearance of their `(domain_id, root, count)` ([`TreeNumbering`]),
//! and each tree's named leaves are sorted and deduplicated by rank with every
//! later naming required to agree with the first on its coordinates
//! ([`dedup_tree_leaves`]).
//!
//! Everything here is a **kernel subroutine**: the acceptance boundary is
//! [`super::verify`], and these functions are exposed for the conformance and
//! rejection-path test suites (and the Lean plan bridge).  Nothing here may be
//! skipped or reordered by a caller that wants a verdict.
//!
//! Under the `unsound-*` diagnostic features (see the crate docs) the
//! authentication steps are skipped; everything else is unchanged.

use std::collections::BTreeMap;

use super::CheckError;
use crate::codec::{
    CommitmentRef, MultiproofWitness, Obligation, PackedWitness, Plan, PositionEntry, PositionRef,
    Statement, Witness,
};
use crate::frame::{fold_multiproof, fold_path, leaf, merkle_depth, MultiproofError};
use crate::gateset::{decode_schema, GateSet};
use crate::packed::{
    extract_word, extract_word_at, parse_leaf_descriptor, parse_packed_schema, vllm_fold_path,
    vllm_leaf_hash, word_be_bytes, LeafDescriptor, PackedSchema,
};
use crate::{scope_end, scope_start};

/// Whether one opening authenticates under its commitment: the leaf hashed
/// with the exact framing, folded along its path to the root.
///
/// In the `unsound-skip-authentication` build this is `true` without hashing;
/// in `unsound-skip-path-fold` the leaf is hashed (and kept from being
/// optimised away) but never folded.
fn authenticate(
    commitment: &CommitmentRef,
    position: &PositionRef,
    value: &[u8],
    path: &[[u8; 32]],
) -> bool {
    if cfg!(feature = "unsound-skip-authentication") {
        return true;
    }
    let digest = leaf(
        &commitment.domain_id,
        position.rank,
        position.position,
        position.schema.as_bytes(),
        value,
    );
    if cfg!(feature = "unsound-skip-path-fold") {
        core::hint::black_box(digest);
        return true;
    }
    fold_path(&commitment.domain_id, position.rank, digest, path) == commitment.root
}

/// Pass 1 for a format-1 witness: authenticate and decode every opening.
/// Returns the decoded values per obligation, indexed like
/// `Obligation::positions`.
///
/// Values decode at the width their own leaf schema names, so the resolved
/// set plays no part here; the parameter keeps the kernel's call shape
/// (resolve once, then both passes) and pins that the set was resolved before
/// any opening is trusted.
pub fn check_openings(
    statement: &Statement,
    witness: &Witness,
    _set: &GateSet,
) -> Result<Vec<Vec<u64>>, CheckError> {
    if witness.obligations.len() != statement.obligations.len() {
        return Err(CheckError::WitnessShape(format!(
            "{} obligations in the witness, {} in the statement",
            witness.obligations.len(),
            statement.obligations.len()
        )));
    }
    let mut decoded = Vec::with_capacity(statement.obligations.len());
    for (index, (obligation, openings)) in statement
        .obligations
        .iter()
        .zip(&witness.obligations)
        .enumerate()
    {
        if openings.len() != obligation.positions.len() {
            return Err(CheckError::WitnessShape(format!(
                "obligation {index} has {} openings for {} positions",
                openings.len(),
                obligation.positions.len()
            )));
        }
        let mut values = Vec::with_capacity(openings.len());
        for (slot, (position, opening)) in obligation.positions.iter().zip(openings).enumerate() {
            let commitment = &obligation.commitments[position.commitment as usize];
            let fail = |reason| CheckError::Opening {
                obligation: index,
                position: slot,
                reason,
            };
            if position.rank >= commitment.count {
                return Err(fail("rank outside the commitment"));
            }
            if opening.path.len() != merkle_depth(commitment.count) {
                return Err(fail("path length differs from the tree depth"));
            }
            if !authenticate(commitment, position, &opening.value, &opening.path) {
                return Err(fail("authentication path does not reach the root"));
            }
            values.push(decode_value(index, slot, position, &opening.value)?);
        }
        decoded.push(values);
    }
    Ok(decoded)
}

/// Compare an opened value with the pinned public input, if any, and decode it.
fn decode_value(
    obligation: usize,
    slot: usize,
    position: &PositionRef,
    value: &[u8],
) -> Result<u64, CheckError> {
    if let Some(expected) = &position.expected {
        if expected.as_slice() != value {
            return Err(CheckError::PublicInput {
                obligation,
                position: slot,
            });
        }
    }
    decode_schema(&position.schema, value).ok_or(CheckError::Value {
        obligation,
        position: slot,
    })
}

// -- the shared tree-grouping discipline ---------------------------------------------

/// One tree of a statement (a distinct `(domain_id, root, count)`) and the
/// leaves the batch opens in it (strictly ascending by rank once
/// deduplicated).  `TreeRef` and `PackedTreeRef` are this shape at the two
/// leaf types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tree<'a, L> {
    pub commitment: &'a CommitmentRef,
    pub leaves: Vec<L>,
}

/// One tree of a statement and the frame-v3 leaves the batch opens in it.
pub type TreeRef<'a> = Tree<'a, TreeLeaf>;
/// One tree of a statement whose positions are all packed, and the leaves it opens.
pub type PackedTreeRef<'a> = Tree<'a, PackedLeafRef>;

/// Trees numbered by first appearance of their `(domain_id, root, count)`
/// scanning the obligations in statement order and each obligation's
/// `commitments` in order -- the order a format-2 or format-3 witness lists
/// them in.  One numbering, shared by the three tree tables.
struct TreeNumbering<'a, L> {
    order: BTreeMap<([u8; 32], [u8; 32], u64), usize>,
    trees: Vec<Tree<'a, L>>,
}

impl<'a, L> TreeNumbering<'a, L> {
    fn new() -> Self {
        TreeNumbering {
            order: BTreeMap::new(),
            trees: Vec::new(),
        }
    }

    /// The tree index of `commitment`, defining a new tree on first sight.
    fn index_of(&mut self, commitment: &'a CommitmentRef) -> usize {
        let trees = &mut self.trees;
        *self
            .order
            .entry((commitment.domain_id, commitment.root, commitment.count))
            .or_insert_with(|| {
                trees.push(Tree {
                    commitment,
                    leaves: Vec::new(),
                });
                trees.len() - 1
            })
    }

    /// The tree index of each of one obligation's commitments, in order.
    fn row<P>(&mut self, obligation: &'a Obligation<P>) -> Vec<usize> {
        obligation
            .commitments
            .iter()
            .map(|commitment| self.index_of(commitment))
            .collect()
    }
}

/// Sort each tree's named leaves by `(rank, obligation, slot)` and keep the
/// first naming of every rank; every later naming must `agree` with it on the
/// coordinates its table compares (a rank two positions open at different
/// coordinates is a statement no tree can serve).  One discipline, shared by
/// the three tree tables.
fn dedup_tree_leaves<L: Copy>(
    trees: &mut [Tree<'_, L>],
    key: impl Fn(&L) -> (u64, usize, usize) + Copy,
    agree: impl Fn(&L, &L) -> bool + Copy,
    conflict: impl Fn(&L) -> CheckError + Copy,
) -> Result<(), CheckError> {
    for tree in trees {
        // stable by (rank, obligation, slot): the first position naming a rank represents it
        tree.leaves.sort_unstable_by_key(key);
        let mut distinct: Vec<L> = Vec::with_capacity(tree.leaves.len());
        for leaf in tree.leaves.drain(..) {
            if let Some(previous) = distinct.last() {
                if key(previous).0 == key(&leaf).0 {
                    if !agree(previous, &leaf) {
                        return Err(conflict(&leaf));
                    }
                    continue;
                }
            }
            distinct.push(leaf);
        }
        tree.leaves = distinct;
    }
    Ok(())
}

/// The index of the deduplicated leaf at `rank` (which the grouping
/// guarantees exists).
fn leaf_of_rank<L>(leaves: &[L], rank: u64, rank_of: impl FnMut(&L) -> u64) -> usize {
    leaves
        .binary_search_by_key(&rank, rank_of)
        .expect("every position's rank is a leaf of its tree")
}

/// The witness lists exactly one tree per tree of the statement's table.
fn expect_witness_trees(actual: usize, expected: usize) -> Result<(), CheckError> {
    if actual != expected {
        return Err(CheckError::WitnessShape(format!(
            "{actual} trees in the witness, {expected} in the statement"
        )));
    }
    Ok(())
}

// -- the multiproof opening pass ------------------------------------------------------

/// One distinct leaf a batch opens: its rank and the first position that names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeLeaf {
    pub rank: u64,
    pub obligation: usize,
    pub slot: usize,
}

/// The trees of a statement in canonical order and where every position lands.
///
/// Trees are numbered by first appearance of their `(domain_id, root, count)`
/// ([`TreeNumbering`]).  `slots[i][j]` is `(tree, leaf)` for position `j` of
/// obligation `i`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeTable<'a> {
    pub trees: Vec<TreeRef<'a>>,
    pub slots: Vec<Vec<(usize, usize)>>,
}

/// Group the positions of a statement by the tree they open (mirrors
/// `veritor.protocol.proofs.statement.opened_trees`).  A rank that two
/// positions open at different coordinates is a statement no tree can serve.
pub fn tree_table(statement: &Statement) -> Result<TreeTable<'_>, CheckError> {
    let mut numbering: TreeNumbering<'_, TreeLeaf> = TreeNumbering::new();
    let mut ranks: Vec<Vec<(usize, u64)>> = Vec::with_capacity(statement.obligations.len());
    for (index, obligation) in statement.obligations.iter().enumerate() {
        let tree_of = numbering.row(obligation);
        let mut row = Vec::with_capacity(obligation.positions.len());
        for (slot, position) in obligation.positions.iter().enumerate() {
            let tree = tree_of[position.commitment as usize];
            numbering.trees[tree].leaves.push(TreeLeaf {
                rank: position.rank,
                obligation: index,
                slot,
            });
            row.push((tree, position.rank));
        }
        ranks.push(row);
    }
    let mut trees = numbering.trees;
    dedup_tree_leaves(
        &mut trees,
        |leaf| (leaf.rank, leaf.obligation, leaf.slot),
        |previous, leaf| {
            let a = &statement.obligations[previous.obligation].positions[previous.slot];
            let b = &statement.obligations[leaf.obligation].positions[leaf.slot];
            a.position == b.position && a.schema == b.schema
        },
        |leaf| CheckError::Opening {
            obligation: leaf.obligation,
            position: leaf.slot,
            reason: "rank opened at two coordinates in one tree",
        },
    )?;
    let slots = ranks
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|(tree, rank)| {
                    (tree, leaf_of_rank(&trees[tree].leaves, rank, |leaf| leaf.rank))
                })
                .collect()
        })
        .collect();
    Ok(TreeTable { trees, slots })
}

/// Authenticate every opened leaf of one tree at once: hash the distinct
/// leaves, fold the multiproof, compare with the root.  Same diagnostic
/// short cuts as `authenticate`.
fn authenticate_tree(
    statement: &Statement,
    tree: &TreeRef<'_>,
    values: &[Vec<u8>],
    siblings: &[[u8; 32]],
) -> Result<(), &'static str> {
    if cfg!(feature = "unsound-skip-authentication") {
        return Ok(());
    }
    let commitment = tree.commitment;
    let digests: Vec<(u64, [u8; 32])> = tree
        .leaves
        .iter()
        .zip(values)
        .map(|(item, value)| {
            let position = &statement.obligations[item.obligation].positions[item.slot];
            (
                item.rank,
                leaf(
                    &commitment.domain_id,
                    item.rank,
                    position.position,
                    position.schema.as_bytes(),
                    value,
                ),
            )
        })
        .collect();
    if cfg!(feature = "unsound-skip-path-fold") {
        core::hint::black_box(digests);
        return Ok(());
    }
    match fold_multiproof(&commitment.domain_id, commitment.count, &digests, siblings) {
        Ok(root) if root == commitment.root => Ok(()),
        Ok(_) => Err("multiproof does not reach the root"),
        Err(MultiproofError::Leaves) => Err("rank outside the commitment"),
        Err(MultiproofError::Short) => Err("multiproof is too short"),
        Err(MultiproofError::Long) => Err("multiproof has digests left over"),
    }
}

/// Pass 1 for a format-2 witness: reconstruct every tree's root once from
/// its multiproof, then decode each position's value from the leaf it names.
/// Returns exactly what `check_openings` returns for the format-1 witness of
/// the same openings.
pub fn check_openings_multi(
    statement: &Statement,
    witness: &MultiproofWitness,
    _set: &GateSet,
) -> Result<Vec<Vec<u64>>, CheckError> {
    scope_start!("merkle.table");
    let table = tree_table(statement);
    scope_end!("merkle.table");
    let table = table?;
    scope_start!("merkle.auth");
    expect_witness_trees(witness.trees.len(), table.trees.len())?;
    for (index, (tree, opened)) in table.trees.iter().zip(&witness.trees).enumerate() {
        if opened.values.len() != tree.leaves.len() {
            return Err(CheckError::WitnessShape(format!(
                "tree {index} has {} values for {} opened leaves",
                opened.values.len(),
                tree.leaves.len()
            )));
        }
        let Some(first) = tree.leaves.first() else {
            if !opened.siblings.is_empty() {
                return Err(CheckError::WitnessShape(format!(
                    "tree {index} opens nothing but has a multiproof"
                )));
            }
            continue;
        };
        let fail = |reason| CheckError::Opening {
            obligation: first.obligation,
            position: first.slot,
            reason,
        };
        if tree
            .leaves
            .last()
            .is_some_and(|leaf| leaf.rank >= tree.commitment.count)
        {
            return Err(fail("rank outside the commitment"));
        }
        authenticate_tree(statement, tree, &opened.values, &opened.siblings).map_err(fail)?;
    }
    scope_end!("merkle.auth");
    scope_start!("merkle.decode");
    let mut decoded = Vec::with_capacity(statement.obligations.len());
    for (index, (obligation, row)) in statement.obligations.iter().zip(&table.slots).enumerate() {
        let mut values = Vec::with_capacity(row.len());
        for (slot, (position, &(tree, leaf))) in obligation.positions.iter().zip(row).enumerate() {
            values.push(decode_value(
                index,
                slot,
                position,
                &witness.trees[tree].values[leaf],
            )?);
        }
        decoded.push(values);
    }
    scope_end!("merkle.decode");
    Ok(decoded)
}

// -- the packed opening pass: whole leaves under the serving side's framing ----------

/// One distinct packed leaf a statement opens: its rank, the `position` and
/// leaf descriptor every position naming it agreed on, and the first position
/// that named it (for error attribution).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedLeafRef {
    pub rank: u64,
    pub position: u64,
    pub descriptor: LeafDescriptor,
    pub obligation: usize,
    pub slot: usize,
}

/// The packed trees of a statement in tree order (the order of [`tree_table`]),
/// where every position lands (`slots[i][j] = (tree, leaf)`), and every
/// position's parsed schema (`schemas[i][j]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedTable<'a> {
    pub trees: Vec<PackedTreeRef<'a>>,
    pub slots: Vec<Vec<(usize, usize)>>,
    pub schemas: Vec<Vec<PackedSchema>>,
}

/// The dedup call the two packed tables share: every position naming one rank
/// of one tree must agree on `position` and on the leaf descriptor (they may
/// differ in word index).
fn dedup_packed_leaves(trees: &mut [PackedTreeRef<'_>]) -> Result<(), CheckError> {
    dedup_tree_leaves(
        trees,
        |leaf| (leaf.rank, leaf.obligation, leaf.slot),
        |previous, leaf| previous.position == leaf.position && previous.descriptor == leaf.descriptor,
        |leaf| CheckError::Opening {
            obligation: leaf.obligation,
            position: leaf.slot,
            reason: "rank opened at two coordinates in one tree",
        },
    )
}

/// Group the positions of a statement by tree and by rank for a format-3
/// witness (mirrors `veritor.protocol.proofs.packed.packed_table`).  Every
/// position must carry a `packed/v1` schema -- this format carries no
/// one-word frame-v3 openings.
pub fn packed_table(statement: &Statement) -> Result<PackedTable<'_>, CheckError> {
    let mut numbering: TreeNumbering<'_, PackedLeafRef> = TreeNumbering::new();
    let mut ranks: Vec<Vec<(usize, u64)>> = Vec::with_capacity(statement.obligations.len());
    let mut schemas: Vec<Vec<PackedSchema>> = Vec::with_capacity(statement.obligations.len());
    for (index, obligation) in statement.obligations.iter().enumerate() {
        let tree_of = numbering.row(obligation);
        let mut row = Vec::with_capacity(obligation.positions.len());
        let mut parsed = Vec::with_capacity(obligation.positions.len());
        for (slot, position) in obligation.positions.iter().enumerate() {
            let schema = parse_packed_schema(&position.schema).ok_or_else(|| {
                CheckError::WitnessShape(format!(
                    "obligation {index} position {slot}: schema is not a packed schema"
                ))
            })?;
            let tree = tree_of[position.commitment as usize];
            numbering.trees[tree].leaves.push(PackedLeafRef {
                rank: position.rank,
                position: position.position,
                descriptor: schema.descriptor(),
                obligation: index,
                slot,
            });
            row.push((tree, position.rank));
            parsed.push(schema);
        }
        ranks.push(row);
        schemas.push(parsed);
    }
    let mut trees = numbering.trees;
    dedup_packed_leaves(&mut trees)?;
    let slots = ranks
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|(tree, rank)| {
                    (tree, leaf_of_rank(&trees[tree].leaves, rank, |leaf| leaf.rank))
                })
                .collect()
        })
        .collect();
    Ok(PackedTable {
        trees,
        slots,
        schemas,
    })
}

/// Whether one packed leaf authenticates: exactly the descriptor's byte
/// length, hashed under `verity-vllm/leaf/v1` with the prover-supplied id,
/// dtype and shape, folded along a path of the right shape to the root.
/// Same diagnostic short cuts as `authenticate`.
fn authenticate_packed(
    commitment: &CommitmentRef,
    item: &PackedLeafRef,
    leaf: &crate::codec::PackedLeaf,
) -> Result<(), &'static str> {
    if item.rank >= commitment.count {
        return Err("rank outside the commitment");
    }
    if item.descriptor.framing != crate::packed::VLLM_LEAF_FRAMING {
        return Err("unknown packed-leaf framing");
    }
    if Some(leaf.value.len()) != item.descriptor.leaf_bytes() {
        return Err("packed leaf length differs from its descriptor");
    }
    if cfg!(feature = "unsound-skip-authentication") {
        return Ok(());
    }
    let digest = vllm_leaf_hash(&leaf.canonical_id, &leaf.dtype, &leaf.shape_text, &leaf.value);
    if cfg!(feature = "unsound-skip-path-fold") {
        core::hint::black_box(digest);
        return Ok(());
    }
    match vllm_fold_path(digest, commitment.count, item.rank, &leaf.path) {
        Some(root) if root == commitment.root => Ok(()),
        Some(_) => Err("authentication path does not reach the root"),
        None => Err("authentication path has the wrong shape"),
    }
}

/// Authenticate every distinct packed leaf of every tree once, under the
/// serving side's framing: the witness must list exactly the table's trees
/// and, per tree, exactly its distinct opened ranks.  Shared by the packed
/// and the ranged pass, whose authentication semantics are the same.
fn authenticate_packed_trees(
    trees: &[PackedTreeRef<'_>],
    witness: &PackedWitness,
) -> Result<(), CheckError> {
    expect_witness_trees(witness.trees.len(), trees.len())?;
    for (index, (tree, leaves)) in trees.iter().zip(&witness.trees).enumerate() {
        if leaves.len() != tree.leaves.len() {
            return Err(CheckError::WitnessShape(format!(
                "tree {index} has {} leaves for {} opened ranks",
                leaves.len(),
                tree.leaves.len()
            )));
        }
        for (item, leaf) in tree.leaves.iter().zip(leaves) {
            authenticate_packed(tree.commitment, item, leaf).map_err(|reason| {
                CheckError::Opening {
                    obligation: item.obligation,
                    position: item.slot,
                    reason,
                }
            })?;
        }
    }
    Ok(())
}

/// Pass 1 for a format-3 witness: authenticate every packed leaf once under
/// the serving side's framing, then read every position's word at its
/// statement-fixed index (`packed::extract_word`), comparing a pinned
/// position's word -- in its canonical big-endian bytes -- with the public
/// input.  Returns the decoded values per obligation, indexed like
/// `Obligation::positions`, as the other opening passes do.
pub fn check_openings_packed(
    statement: &Statement,
    witness: &PackedWitness,
    _set: &GateSet,
) -> Result<Vec<Vec<u64>>, CheckError> {
    scope_start!("merkle.table");
    let table = packed_table(statement);
    scope_end!("merkle.table");
    let table = table?;
    scope_start!("merkle.auth");
    authenticate_packed_trees(&table.trees, witness)?;
    scope_end!("merkle.auth");
    scope_start!("merkle.decode");
    let mut decoded = Vec::with_capacity(statement.obligations.len());
    for (index, ((obligation, row), row_schemas)) in statement
        .obligations
        .iter()
        .zip(&table.slots)
        .zip(&table.schemas)
        .enumerate()
    {
        let mut values = Vec::with_capacity(row.len());
        for (slot, ((position, &(tree, leaf)), schema)) in
            obligation.positions.iter().zip(row).zip(row_schemas).enumerate()
        {
            let word = extract_word(schema, &witness.trees[tree][leaf].value).ok_or(
                CheckError::Value {
                    obligation: index,
                    position: slot,
                },
            )?;
            if let Some(expected) = &position.expected {
                if *expected != word_be_bytes(schema.width, word) {
                    return Err(CheckError::PublicInput {
                        obligation: index,
                        position: slot,
                    });
                }
            }
            values.push(word);
        }
        decoded.push(values);
    }
    scope_end!("merkle.decode");
    Ok(decoded)
}

// -- the ranged opening pass: statement format 3 -----------------------------------------

/// One position entry of a format-3 obligation, resolved to its leaf in the
/// ranged table: which tree, which of that tree's distinct leaves, and how to
/// read its word(s).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RangedSlot<'a> {
    Single {
        tree: usize,
        leaf: usize,
        schema: PackedSchema,
        expected: Option<&'a [u8]>,
    },
    Range {
        tree: usize,
        leaf: usize,
        descriptor: LeafDescriptor,
        start: u64,
        count: u64,
        stride: u64,
    },
}

/// The tree table of a format-3 statement: the same `trees` as
/// [`packed_table`] would give for the expanded statement (same order, same
/// distinct leaves, same first-naming attribution), computed from the
/// *entries* -- one leaf reference per entry, not per word.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangedTable<'a> {
    pub trees: Vec<PackedTreeRef<'a>>,
    pub slots: Vec<Vec<RangedSlot<'a>>>,
}

/// Group the position entries of a format-3 statement by tree and by rank.
/// Every single entry must carry a `packed/v1` schema (as in `packed_table`);
/// a range's descriptor was validated by the parser.  Two entries naming one
/// rank of one tree must agree on `position` and on the leaf descriptor.
///
/// This is the whole of what `merkle.table` does for format 3: with three
/// entries per GEMM coordinate instead of 3,073 positions there is nothing
/// left to organise, so no supplied table (a `Plan`) is needed -- the
/// sort-and-dedup runs over the entries, whose count is the number of
/// distinct (leaf, run) pairs the lowering emitted.
pub fn ranged_table(statement: &Statement<PositionEntry>) -> Result<RangedTable<'_>, CheckError> {
    let mut numbering: TreeNumbering<'_, PackedLeafRef> = TreeNumbering::new();
    // per obligation, per entry: (tree, rank, first expanded slot)
    let mut coordinates: Vec<Vec<(usize, u64, usize)>> =
        Vec::with_capacity(statement.obligations.len());
    for (index, obligation) in statement.obligations.iter().enumerate() {
        let tree_of = numbering.row(obligation);
        let mut row = Vec::with_capacity(obligation.positions.len());
        let mut slot = 0usize;
        for entry in &obligation.positions {
            let (commitment, rank, position, descriptor, width) = match entry {
                PositionEntry::Single(position) => {
                    let schema = parse_packed_schema(&position.schema).ok_or_else(|| {
                        CheckError::WitnessShape(format!(
                            "obligation {index} position {slot}: schema is not a packed schema"
                        ))
                    })?;
                    (position.commitment, position.rank, position.position, schema.descriptor(), 1)
                }
                PositionEntry::Range(range) => {
                    let descriptor = parse_leaf_descriptor(&range.descriptor).ok_or_else(|| {
                        CheckError::WitnessShape(format!(
                            "obligation {index} position {slot}: range descriptor is not a leaf descriptor"
                        ))
                    })?;
                    (range.commitment, range.rank, range.position, descriptor, range.count as usize)
                }
            };
            let tree = tree_of[commitment as usize];
            numbering.trees[tree].leaves.push(PackedLeafRef {
                rank,
                position,
                descriptor,
                obligation: index,
                slot,
            });
            row.push((tree, rank, slot));
            slot += width;
        }
        coordinates.push(row);
    }
    let mut trees = numbering.trees;
    dedup_packed_leaves(&mut trees)?;
    let slots = statement
        .obligations
        .iter()
        .zip(coordinates)
        .map(|(obligation, row)| {
            obligation
                .positions
                .iter()
                .zip(row)
                .map(|(entry, (tree, rank, _))| {
                    let leaf = leaf_of_rank(&trees[tree].leaves, rank, |leaf| leaf.rank);
                    match entry {
                        PositionEntry::Single(position) => RangedSlot::Single {
                            tree,
                            leaf,
                            schema: parse_packed_schema(&position.schema).expect("checked above"),
                            expected: position.expected.as_deref(),
                        },
                        PositionEntry::Range(range) => RangedSlot::Range {
                            tree,
                            leaf,
                            descriptor: parse_leaf_descriptor(&range.descriptor).expect("checked above"),
                            start: range.start,
                            count: range.count,
                            stride: range.stride,
                        },
                    }
                })
                .collect()
        })
        .collect();
    Ok(RangedTable { trees, slots })
}

/// Pass 1 for a format-3 statement with a format-3 witness: authenticate each
/// distinct packed leaf once (exactly as [`check_openings_packed`] does,
/// through the same [`authenticate_packed_trees`]), then read every entry's
/// word(s) by index straight out of the leaf bytes -- a range reads `count`
/// words at `start, start + stride, ...` without building a schema string or
/// a per-word lookup.  Returns the decoded values per obligation indexed like
/// the *expanded* position list, so
/// `check_openings_ranged(s3) == check_openings_packed(expand_statement(s3))`
/// value for value and error for error.
pub fn check_openings_ranged(
    statement: &Statement<PositionEntry>,
    witness: &PackedWitness,
    _set: &GateSet,
) -> Result<Vec<Vec<u64>>, CheckError> {
    scope_start!("merkle.table");
    let table = ranged_table(statement);
    scope_end!("merkle.table");
    let table = table?;
    scope_start!("merkle.auth");
    authenticate_packed_trees(&table.trees, witness)?;
    scope_end!("merkle.auth");
    scope_start!("merkle.decode");
    let mut decoded = Vec::with_capacity(statement.obligations.len());
    for (index, (obligation, row)) in statement.obligations.iter().zip(&table.slots).enumerate() {
        let mut values = Vec::with_capacity(obligation.slot_count() as usize);
        for entry in row {
            match *entry {
                RangedSlot::Single {
                    tree,
                    leaf,
                    ref schema,
                    expected,
                } => {
                    let slot = values.len();
                    let word = extract_word(schema, &witness.trees[tree][leaf].value).ok_or(
                        CheckError::Value {
                            obligation: index,
                            position: slot,
                        },
                    )?;
                    if let Some(expected) = expected {
                        if expected != word_be_bytes(schema.width, word) {
                            return Err(CheckError::PublicInput {
                                obligation: index,
                                position: slot,
                            });
                        }
                    }
                    values.push(word);
                }
                RangedSlot::Range {
                    tree,
                    leaf,
                    ref descriptor,
                    start,
                    count,
                    stride,
                } => {
                    let value = &witness.trees[tree][leaf].value;
                    for i in 0..count {
                        let slot = values.len();
                        let word = extract_word_at(descriptor, start + i * stride, value).ok_or(
                            CheckError::Value {
                                obligation: index,
                                position: slot,
                            },
                        )?;
                        values.push(word);
                    }
                }
            }
        }
        decoded.push(values);
    }
    scope_end!("merkle.decode");
    Ok(decoded)
}

// -- the planned opening pass: an untrusted tree table, validated ---------------------

/// One distinct leaf of a plan's tree: the coordinates every position that
/// names it must agree on, and the first position that defined it (for error
/// attribution and for the leaf hash, exactly as `TreeLeaf` does).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanLeaf<'a> {
    pub rank: u64,
    pub position: u64,
    pub schema: &'a str,
    pub obligation: usize,
    pub slot: usize,
}

/// One tree of a validated plan: its commitment and its distinct leaves,
/// strictly ascending by rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanTree<'a> {
    pub commitment: &'a CommitmentRef,
    pub leaves: Vec<PlanLeaf<'a>>,
}

/// A plan that passed `validate_plan`: the statement's tree table, with the
/// plan's own flat index lists kept for the decode pass (`tree_of` per
/// obligation commitment, `leaf_of` per obligation position, both in
/// statement order and both range-checked).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanTable<'a> {
    pub trees: Vec<PlanTree<'a>>,
    pub tree_of: &'a [u32],
    pub leaf_of: &'a [u32],
}

/// Whether two commitment references name the same tree.
fn same_tree(a: &CommitmentRef, b: &CommitmentRef) -> bool {
    a.domain_id == b.domain_id && a.root == b.root && a.count == b.count
}

/// Decide whether `plan` is the tree table of `statement` -- the one
/// `tree_table` computes -- in one linear pass, and return it if so.
///
/// The plan may claim any tree numbering and any leaf numbering; this
/// accepts exactly the canonical one:
///
/// * trees are numbered by first appearance of a `(domain_id, root, count)`
///   scanning the obligations and their `commitments` in order, and every
///   commitment mapped to a tree names that tree's `(domain_id, root, count)`;
/// * every position maps to a leaf of the tree of its commitment; every
///   position that names a leaf agrees with it on `(rank, position, schema)`;
/// * every declared leaf is named by at least one position, and within a
///   tree the leaves are strictly ascending by rank (so they are the distinct
///   opened ranks, in order, and nothing else);
/// * the plan's index lists have exactly the statement's shape.
///
/// A plan that omits, duplicates, permutes or misdirects anything fails one
/// of these and the batch is rejected; nothing in the plan is used before it
/// is checked against the statement.
pub fn validate_plan<'a>(
    statement: &'a Statement,
    plan: &'a Plan,
) -> Result<PlanTable<'a>, CheckError> {
    let tree_count = plan.leaf_counts.len();
    // Tree `t` is unknown until an obligation commitment first maps to it.
    let mut commitments: Vec<Option<&'a CommitmentRef>> = vec![None; tree_count];
    let mut leaves: Vec<Vec<Option<PlanLeaf<'a>>>> = Vec::with_capacity(tree_count);
    let mut t = 0;
    while t < tree_count {
        leaves.push(vec![None; plan.leaf_counts[t] as usize]);
        t += 1;
    }
    let mut defined_trees = 0usize;
    let mut tree_cursor = 0usize;
    let mut leaf_cursor = 0usize;
    let mut index = 0usize;
    while index < statement.obligations.len() {
        let obligation: &'a Obligation = &statement.obligations[index];
        let commitment_count = obligation.commitments.len();
        let position_count = obligation.positions.len();
        if plan.tree_of.len() - tree_cursor < commitment_count
            || plan.leaf_of.len() - leaf_cursor < position_count
        {
            return Err(CheckError::Plan("plan is shorter than the statement"));
        }
        let tree_row = &plan.tree_of[tree_cursor..tree_cursor + commitment_count];
        let leaf_row = &plan.leaf_of[leaf_cursor..leaf_cursor + position_count];
        tree_cursor += commitment_count;
        leaf_cursor += position_count;
        let mut c = 0;
        while c < commitment_count {
            let tree = tree_row[c] as usize;
            if tree >= tree_count {
                return Err(CheckError::Plan("tree index out of range"));
            }
            match commitments[tree] {
                None => {
                    if tree != defined_trees {
                        return Err(CheckError::Plan(
                            "trees are not numbered by first appearance",
                        ));
                    }
                    // one tree per distinct `(domain_id, root, count)`: a tree
                    // already defined under an earlier index must not be split
                    // into a second one (the canonical table has no duplicates)
                    let mut u = 0;
                    while u < defined_trees {
                        if let Some(existing) = commitments[u] {
                            if same_tree(existing, &obligation.commitments[c]) {
                                return Err(CheckError::Plan("a tree is defined twice"));
                            }
                        }
                        u += 1;
                    }
                    commitments[tree] = Some(&obligation.commitments[c]);
                    defined_trees += 1;
                }
                Some(existing) => {
                    if !same_tree(existing, &obligation.commitments[c]) {
                        return Err(CheckError::Plan(
                            "a commitment is mapped to another commitment's tree",
                        ));
                    }
                }
            }
            c += 1;
        }
        let mut slot = 0;
        while slot < position_count {
            let position: &'a PositionRef = &obligation.positions[slot];
            // `position.commitment` was range-checked by the statement parser.
            let tree = tree_row[position.commitment as usize] as usize;
            let k = leaf_row[slot] as usize;
            let row = &mut leaves[tree];
            if k >= row.len() {
                return Err(CheckError::Plan("leaf index out of range"));
            }
            match row[k] {
                None => {
                    row[k] = Some(PlanLeaf {
                        rank: position.rank,
                        position: position.position,
                        schema: position.schema.as_str(),
                        obligation: index,
                        slot,
                    });
                }
                Some(existing) => {
                    if existing.rank != position.rank
                        || existing.position != position.position
                        || existing.schema != position.schema.as_str()
                    {
                        return Err(CheckError::Plan(
                            "a position is mapped to a leaf with other coordinates",
                        ));
                    }
                }
            }
            slot += 1;
        }
        index += 1;
    }
    if tree_cursor != plan.tree_of.len() || leaf_cursor != plan.leaf_of.len() {
        return Err(CheckError::Plan("plan is longer than the statement"));
    }
    if defined_trees != tree_count {
        return Err(CheckError::Plan("a tree is never named by a commitment"));
    }
    let mut trees: Vec<PlanTree<'a>> = Vec::with_capacity(tree_count);
    let mut t = 0;
    while t < tree_count {
        let commitment = commitments[t].ok_or(CheckError::Plan("tree without a commitment"))?;
        let row = &leaves[t];
        let mut ordered: Vec<PlanLeaf<'a>> = Vec::with_capacity(row.len());
        let mut k = 0;
        while k < row.len() {
            let item = row[k].ok_or(CheckError::Plan("a leaf is named by no position"))?;
            if k > 0 && ordered[k - 1].rank >= item.rank {
                return Err(CheckError::Plan("leaves are not strictly ascending by rank"));
            }
            ordered.push(item);
            k += 1;
        }
        trees.push(PlanTree {
            commitment,
            leaves: ordered,
        });
        t += 1;
    }
    Ok(PlanTable {
        trees,
        tree_of: &plan.tree_of,
        leaf_of: &plan.leaf_of,
    })
}

/// `authenticate_tree` for a validated plan's tree: hash the distinct leaves
/// at the coordinates the plan's positions agreed on, fold the multiproof,
/// compare with the root.  Same diagnostic short cuts.
fn authenticate_plan_tree(
    tree: &PlanTree<'_>,
    values: &[Vec<u8>],
    siblings: &[[u8; 32]],
) -> Result<(), &'static str> {
    if cfg!(feature = "unsound-skip-authentication") {
        return Ok(());
    }
    let commitment = tree.commitment;
    let mut digests: Vec<(u64, [u8; 32])> = Vec::with_capacity(tree.leaves.len());
    let mut k = 0;
    while k < tree.leaves.len() {
        let item = &tree.leaves[k];
        digests.push((
            item.rank,
            leaf(
                &commitment.domain_id,
                item.rank,
                item.position,
                item.schema.as_bytes(),
                &values[k],
            ),
        ));
        k += 1;
    }
    if cfg!(feature = "unsound-skip-path-fold") {
        core::hint::black_box(digests);
        return Ok(());
    }
    match fold_multiproof(&commitment.domain_id, commitment.count, &digests, siblings) {
        Ok(root) if root == commitment.root => Ok(()),
        Ok(_) => Err("multiproof does not reach the root"),
        Err(MultiproofError::Leaves) => Err("rank outside the commitment"),
        Err(MultiproofError::Short) => Err("multiproof is too short"),
        Err(MultiproofError::Long) => Err("multiproof has digests left over"),
    }
}

/// Pass 1 with an untrusted plan: validate the plan against the statement,
/// authenticate every tree once, decode every distinct leaf once, then hand
/// each obligation position the value of the leaf it names.  Returns exactly
/// what `check_openings_multi` returns for the same statement and witness
/// whenever the plan validates (and rejects otherwise); the relation pass is
/// untouched, so the theorem this needs is about this function alone.
pub fn check_openings_planned(
    statement: &Statement,
    witness: &MultiproofWitness,
    plan: &Plan,
    _set: &GateSet,
) -> Result<Vec<Vec<u64>>, CheckError> {
    scope_start!("merkle.plan");
    let table = validate_plan(statement, plan);
    scope_end!("merkle.plan");
    let table = table?;
    check_openings_with_table(statement, witness, &table)
}

/// The rest of `check_openings_planned` once the plan is validated: `table`
/// is the statement's tree table as `validate_plan` returns it (or as the
/// `lean-plan` guest's Lean validator returns it, decoded by its wrapper).
/// `merkle.auth` authenticates every tree once; `merkle.decode` decodes every
/// distinct leaf once and hands each obligation position its leaf's value.
pub fn check_openings_with_table(
    statement: &Statement,
    witness: &MultiproofWitness,
    table: &PlanTable<'_>,
) -> Result<Vec<Vec<u64>>, CheckError> {
    scope_start!("merkle.auth");
    expect_witness_trees(witness.trees.len(), table.trees.len())?;
    let mut t = 0;
    while t < table.trees.len() {
        let tree = &table.trees[t];
        let opened = &witness.trees[t];
        if opened.values.len() != tree.leaves.len() {
            return Err(CheckError::WitnessShape(format!(
                "tree {t} has {} values for {} opened leaves",
                opened.values.len(),
                tree.leaves.len()
            )));
        }
        if tree.leaves.is_empty() {
            if !opened.siblings.is_empty() {
                return Err(CheckError::WitnessShape(format!(
                    "tree {t} opens nothing but has a multiproof"
                )));
            }
            t += 1;
            continue;
        }
        let first = tree.leaves[0];
        let fail = |reason| CheckError::Opening {
            obligation: first.obligation,
            position: first.slot,
            reason,
        };
        if tree.leaves[tree.leaves.len() - 1].rank >= tree.commitment.count {
            return Err(fail("rank outside the commitment"));
        }
        authenticate_plan_tree(tree, &opened.values, &opened.siblings).map_err(fail)?;
        t += 1;
    }
    scope_end!("merkle.auth");
    scope_start!("merkle.decode");
    // Decode each distinct leaf once, at the schema its positions agreed on.
    let mut tree_values: Vec<Vec<u64>> = Vec::with_capacity(table.trees.len());
    let mut t = 0;
    while t < table.trees.len() {
        let tree = &table.trees[t];
        let opened = &witness.trees[t];
        let mut words: Vec<u64> = Vec::with_capacity(tree.leaves.len());
        let mut k = 0;
        while k < tree.leaves.len() {
            let item = &tree.leaves[k];
            let word = decode_schema(item.schema, &opened.values[k]).ok_or(CheckError::Value {
                obligation: item.obligation,
                position: item.slot,
            })?;
            words.push(word);
            k += 1;
        }
        tree_values.push(words);
        t += 1;
    }
    // Hand every position its leaf's word; a pinned public input is compared
    // against the leaf's bytes, exactly as `decode_value` does per position.
    let mut decoded: Vec<Vec<u64>> = Vec::with_capacity(statement.obligations.len());
    let mut tree_cursor = 0usize;
    let mut leaf_cursor = 0usize;
    let mut index = 0usize;
    while index < statement.obligations.len() {
        let obligation = &statement.obligations[index];
        let tree_row = &table.tree_of[tree_cursor..tree_cursor + obligation.commitments.len()];
        let leaf_row = &table.leaf_of[leaf_cursor..leaf_cursor + obligation.positions.len()];
        tree_cursor += obligation.commitments.len();
        leaf_cursor += obligation.positions.len();
        let mut values: Vec<u64> = Vec::with_capacity(obligation.positions.len());
        let mut slot = 0;
        while slot < obligation.positions.len() {
            let position = &obligation.positions[slot];
            let tree = tree_row[position.commitment as usize] as usize;
            let k = leaf_row[slot] as usize;
            if let Some(expected) = &position.expected {
                if expected.as_slice() != witness.trees[tree].values[k].as_slice() {
                    return Err(CheckError::PublicInput {
                        obligation: index,
                        position: slot,
                    });
                }
            }
            values.push(tree_values[tree][k]);
            slot += 1;
        }
        decoded.push(values);
        index += 1;
    }
    scope_end!("merkle.decode");
    Ok(decoded)
}
