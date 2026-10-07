//! Pass 2 of the verification kernel: recompute every obligation's gates from
//! its opened inputs under the pinned semantics of the statement's gate set
//! and compare them with its opened outputs.
//!
//! There is exactly **one** implementation of this pass.  The `attribution`
//! feature (the SP1 sampling profiler's) does not duplicate it: the same body
//! compiles in every build, with its dispatch ([`resolve_op`]), operand
//! assembly ([`gather_args`]) and output comparison ([`output_matches`])
//! kept out of line under `attribution` so the profiler can attribute the
//! gate loop, and inlined otherwise.  (Before the kernel refactor the
//! `attribution` build carried a second copy of this loop that skipped
//! [`check_programs`]; see `docs/vllm-poc/verification-kernel.md`.)
//!
//! These functions are **kernel subroutines**: the acceptance boundary is
//! [`super::verify`].  `check_relations` trusts its `decoded` argument to be
//! the output of an opening pass ([`super::openings`]) on the same statement;
//! the typed path ([`super::AuthenticatedValues`]) makes that unmistakable,
//! and the raw entry point remains only for the conformance and
//! cross-language (Lean) vector suites, which exercise this pass in
//! isolation.

use super::CheckError;
use crate::codec::{Arg, GateOp, Statement};
use crate::gateset::{GateSet, Op};

/// Every program of the statement is well-formed for the gate set: each op is
/// one of the set's, has as many arguments as its arity, and each source gate
/// is one of the program's outputs (Lean `Program.WellFormed`; the structural
/// half -- earlier-only and in-range references, in-range strictly increasing
/// outputs -- is the parser's).
///
/// This holds for *every* program, whether or not an obligation names it,
/// exactly as `checkReference` in `lean/VeritorSpec/Batch.lean` requires.
/// Before the `typed-obligation/v0` conformance work this was only enforced
/// per obligation, so a statement carrying an unused program with a
/// wrong-arity gate was accepted here and rejected by the Lean reference
/// (the `malformed-program-unused-still-rejects` batch of
/// `lean/vectors/tc-ampere-bf16.json`; the `q7.json` batch of the same name
/// uses a forward reference, which the parser catches, and never exposed the
/// gap).  No honest statement has a malformed program, so the change only
/// shrinks the accepted set to the specification's.
pub fn check_programs<P>(statement: &Statement<P>, set: &GateSet) -> Result<(), CheckError> {
    for program in &statement.kinds {
        for (offset, gate) in program.gates.iter().enumerate() {
            let op = set.op(&gate.op).ok_or_else(|| CheckError::UnknownOp {
                kind: program.kind,
                op: gate.op.clone(),
            })?;
            if gate.args.len() != op.arity() {
                return Err(CheckError::Arity {
                    kind: program.kind,
                    offset,
                });
            }
            if op.is_source() && program.outputs.binary_search(&(offset as u32)).is_err() {
                return Err(CheckError::SourceNotOpened {
                    kind: program.kind,
                    offset,
                });
            }
        }
    }
    Ok(())
}

/// Pass 2: every program is well-formed for the gate set
/// ([`check_programs`]), then recompute every obligation's gates from its
/// opened inputs; every opened gate must agree with what was recomputed.
///
/// A source gate has no relation: its opened value *is* its value (the
/// boundary pins an `in` gate to the public input, `kappa_W` a weight), so
/// the kind must open it.  Every other gate is evaluated from what it reads
/// -- an opened input or an earlier gate of the copy -- and, when the kind
/// opens it, compared with the opened value.  Byte-identical in outcome to
/// `TransparentBackend._check_relations`.
pub fn check_relations<P>(
    statement: &Statement<P>,
    decoded: &[Vec<u64>],
    set: &GateSet,
) -> Result<(), CheckError> {
    check_programs(statement, set)?;
    let mut args: Vec<u64> = Vec::with_capacity(2);
    for (index, (obligation, values)) in statement.obligations.iter().zip(decoded).enumerate() {
        let program = statement
            .program(&obligation.kind)
            .expect("the parser checked every obligation's kind");
        let mut local: Vec<u64> = Vec::with_capacity(program.gates.len());
        let mut next_output = 0usize;
        for (offset, gate) in program.gates.iter().enumerate() {
            let op = resolve_op(set, program.kind, offset, gate)?;
            // `program.outputs` is strictly increasing, so one cursor finds the opened slot.
            let slot = if next_output < program.outputs.len()
                && program.outputs[next_output] as usize == offset
            {
                let slot = obligation.outputs[next_output] as usize;
                next_output += 1;
                Some(slot)
            } else {
                None
            };
            if op.is_source() {
                let slot = slot.ok_or(CheckError::SourceNotOpened {
                    kind: program.kind,
                    offset,
                })?;
                local.push(values[slot]);
                continue;
            }
            gather_args(gate, values, &obligation.inputs, &local, &mut args);
            let computed = set
                .evaluate(op, &args)
                .ok_or_else(|| CheckError::Argument {
                    obligation: index,
                    offset,
                    op: gate.op.clone(),
                })?;
            match slot {
                None => local.push(computed),
                Some(slot) => {
                    if !output_matches(values, slot, computed) {
                        return Err(CheckError::Relation {
                            obligation: index,
                            offset,
                            op: gate.op.clone(),
                        });
                    }
                    local.push(values[slot]);
                }
            }
        }
    }
    Ok(())
}

/// Interpreter dispatch: the gate's op in the set, with its arity checked.
/// (Out of line under `attribution` so the profiler can see it; inlined in
/// every other build.)
#[cfg_attr(feature = "attribution", inline(never))]
#[cfg_attr(not(feature = "attribution"), inline(always))]
fn resolve_op(
    set: &GateSet,
    kind: [u8; 32],
    offset: usize,
    gate: &GateOp,
) -> Result<Op, CheckError> {
    let op = set.op(&gate.op).ok_or_else(|| CheckError::UnknownOp {
        kind,
        op: gate.op.clone(),
    })?;
    if gate.args.len() != op.arity() {
        return Err(CheckError::Arity { kind, offset });
    }
    Ok(op)
}

/// Operand assembly: copy the gate's arguments -- opened inputs or earlier
/// gates of the copy -- into the interpreter's argument frame.
#[cfg_attr(feature = "attribution", inline(never))]
#[cfg_attr(not(feature = "attribution"), inline(always))]
fn gather_args(gate: &GateOp, values: &[u64], inputs: &[u32], local: &[u64], args: &mut Vec<u64>) {
    args.clear();
    args.extend(gate.args.iter().map(|arg| match *arg {
        Arg::Port(k) => values[inputs[k as usize] as usize],
        Arg::Local(j) => local[j as usize],
    }));
}

/// Output comparison: the opened value of an output gate against the recomputation.
#[cfg_attr(feature = "attribution", inline(never))]
#[cfg_attr(not(feature = "attribution"), inline(always))]
fn output_matches(values: &[u64], slot: usize, computed: u64) -> bool {
    values[slot] == computed
}
