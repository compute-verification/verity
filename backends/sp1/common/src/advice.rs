//! Prover-side advice generation: **outside** the verification kernel.
//!
//! Nothing here is needed to reject an invalid statement or witness; the
//! kernel validates whatever advice a prover supplies
//! (`crate::kernel::openings::validate_plan`) and rejects anything that is
//! not the statement's canonical tree table.  This module only *produces*
//! the honest advice, for the Python prover's Rust mirror and the test
//! suites; it is dead code in the guest ELF.

use std::collections::BTreeMap;

use crate::codec::{Plan, Statement};
use crate::kernel::openings::tree_table;
use crate::kernel::CheckError;

/// The honest plan for a statement: its canonical tree table, in the plan's
/// flat encoding.  `validate_plan(statement, &plan_of(statement))` always
/// succeeds (completeness), and this is what the Python prover emits.
pub fn plan_of(statement: &Statement) -> Result<Plan, CheckError> {
    let table = tree_table(statement)?;
    let mut order: BTreeMap<([u8; 32], [u8; 32], u64), u32> = BTreeMap::new();
    for (index, tree) in table.trees.iter().enumerate() {
        let commitment = tree.commitment;
        order.insert(
            (commitment.domain_id, commitment.root, commitment.count),
            index as u32,
        );
    }
    let mut tree_of = Vec::new();
    let mut leaf_of = Vec::new();
    for (obligation, row) in statement.obligations.iter().zip(&table.slots) {
        for commitment in &obligation.commitments {
            tree_of.push(order[&(commitment.domain_id, commitment.root, commitment.count)]);
        }
        for &(_, leaf) in row {
            leaf_of.push(leaf as u32);
        }
    }
    Ok(Plan {
        leaf_counts: table
            .trees
            .iter()
            .map(|tree| tree.leaves.len() as u32)
            .collect(),
        tree_of,
        leaf_of,
    })
}
