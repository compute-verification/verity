//! The verification kernel's acceptance boundary.
//!
//! This module is the one place a verdict comes from: a Rust interpreter for
//! the circuit semantics of `veritor.core.gates`; SP1 proves its execution.
//! Untrusted statement, witness and plan bytes go in; a bound public result
//! -- the statement digest and the verdict -- or a rejection comes out.
//! Parsing (`crate::codec`), description validation, authentication
//! (`openings`, `crate::frame`, `crate::packed`) and relation evaluation
//! (`relations`, `crate::gateset` and the numerical modules) all live behind
//! [`verify`]; the SP1 guest (`archive/sp1/guest/src/main.rs`) is a thin wrapper
//! that reads the four input buffers, calls it, commits
//! [`Decision::public_values`] and halts.
//!
//! The pipeline is typed so that no step can be skipped or reordered:
//!
//! ```text
//! raw bytes --parse+seal--> ValidatedStatement --authenticate--> AuthenticatedValues
//!           --check_relations--> verdict     (Decision = digest ++ verdict)
//! ```
//!
//! Only [`ValidatedStatement::validate`] (or [`verify`] itself) constructs a
//! `ValidatedStatement` -- its gate set resolved and pinned by digest -- and
//! only [`ValidatedStatement::authenticate`] constructs
//! [`AuthenticatedValues`], so the relation pass cannot be handed values that
//! were never authenticated.  The per-phase functions under [`openings`] and
//! [`relations`] remain public **as kernel subroutines** for the conformance,
//! rejection-path and cross-language (Lean) vector suites, which exercise
//! each phase in isolation; none of them is an acceptance boundary.
//!
//! Advice is untrusted everywhere: a plan changes how the tree table is
//! obtained, never which openings are authenticated or which values are used
//! ([`openings::validate_plan`]).

pub mod openings;
pub mod relations;

use core::fmt;

use crate::codec::{
    parse_any_statement, parse_any_witness, parse_plan, AnyStatement, AnyWitness,
    MultiproofWitness, PackedWitness, Plan, PositionEntry, Statement, Witness,
};
use crate::frame::sha256;
use crate::gateset::GateSet;
use crate::{
    public_values, public_values_with_object, scope_end, scope_start, PUBLIC_VALUES_LEN,
    PUBLIC_VALUES_WITH_OBJECT_LEN,
};
use openings::{
    check_openings, check_openings_multi, check_openings_packed, check_openings_planned,
    check_openings_ranged,
};
use relations::check_relations;

/// Why the kernel rejects a parsed batch (the wire formats' own rejections
/// are `crate::codec::CodecError`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckError {
    UnknownGateSet {
        id: String,
        width: u32,
    },
    GateSetDigest,
    WitnessShape(String),
    Opening {
        obligation: usize,
        position: usize,
        reason: &'static str,
    },
    PublicInput {
        obligation: usize,
        position: usize,
    },
    Value {
        obligation: usize,
        position: usize,
    },
    UnknownOp {
        kind: [u8; 32],
        op: String,
    },
    Arity {
        kind: [u8; 32],
        offset: usize,
    },
    SourceNotOpened {
        kind: [u8; 32],
        offset: usize,
    },
    /// An argument lies outside the width the gate declares for it (the
    /// Python evaluator raises there, which the transparent backend rejects).
    Argument {
        obligation: usize,
        offset: usize,
        op: String,
    },
    Relation {
        obligation: usize,
        offset: usize,
        op: String,
    },
    /// The plan is not the statement's tree table (`openings::validate_plan`).
    Plan(&'static str),
}

impl fmt::Display for CheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckError::UnknownGateSet { id, width } => {
                write!(f, "unknown gate set {id} at width {width}")
            }
            CheckError::GateSetDigest => {
                f.write_str("the statement's gate set digest is not that of the pinned semantics")
            }
            CheckError::WitnessShape(detail) => write!(f, "witness shape: {detail}"),
            CheckError::Opening {
                obligation,
                position,
                reason,
            } => {
                write!(f, "obligation {obligation} position {position}: {reason}")
            }
            CheckError::PublicInput {
                obligation,
                position,
            } => {
                write!(
                    f,
                    "obligation {obligation} position {position} differs from the public input"
                )
            }
            CheckError::Value {
                obligation,
                position,
            } => {
                write!(
                    f,
                    "obligation {obligation} position {position} is not a canonical value"
                )
            }
            CheckError::UnknownOp { op, .. } => write!(f, "unknown gate {op}"),
            CheckError::Arity { offset, .. } => write!(f, "gate {offset} has the wrong arity"),
            CheckError::SourceNotOpened { offset, .. } => {
                write!(f, "source gate {offset} is not opened")
            }
            CheckError::Argument {
                obligation,
                offset,
                op,
            } => {
                write!(f, "obligation {obligation} gate {offset} feeds {op} an argument outside its width")
            }
            CheckError::Relation {
                obligation,
                offset,
                op,
            } => {
                write!(f, "obligation {obligation} gate {offset} violates {op}")
            }
            CheckError::Plan(reason) => write!(f, "invalid plan: {reason}"),
        }
    }
}

impl std::error::Error for CheckError {}

/// Resolve and pin the statement's gate set.
pub fn gate_set<P>(statement: &Statement<P>) -> Result<GateSet, CheckError> {
    let set = GateSet::resolve(&statement.gate_set_id, statement.width).ok_or_else(|| {
        CheckError::UnknownGateSet {
            id: statement.gate_set_id.clone(),
            width: statement.width,
        }
    })?;
    if set.digest() != statement.gate_set_digest {
        return Err(CheckError::GateSetDigest);
    }
    Ok(set)
}

/// A rejection log line, on the guest's stdout only (the `scopes` feature is
/// the guest's; host builds stay silent on stdout, whose one JSON object is
/// the subprocess protocol).
macro_rules! reject_log {
    ($error:expr) => {{
        #[cfg(feature = "scopes")]
        println!("reject: {}", $error);
        #[cfg(not(feature = "scopes"))]
        let _ = &$error;
    }};
}

// -- the typed pipeline ---------------------------------------------------------------

/// A statement that parsed canonically and whose gate set resolved to the
/// pinned semantics its digest names.  Only [`ValidatedStatement::validate`]
/// and [`verify`] construct one (the fields are private), so holding a
/// `ValidatedStatement` *is* the evidence that description validation ran.
pub struct ValidatedStatement {
    statement: AnyStatement,
    set: GateSet,
}

/// The decoded, authenticated opening values of one batch: exactly one row
/// per obligation, indexed like the (expanded) position list.  Only
/// [`ValidatedStatement::authenticate`] constructs one, so holding an
/// `AuthenticatedValues` is the evidence that every value it carries was
/// opened under its commitment root with the exact framing its schema names
/// and compared with the pinned public inputs.
pub struct AuthenticatedValues<'a> {
    statement: &'a ValidatedStatement,
    values: Vec<Vec<u64>>,
}

impl ValidatedStatement {
    /// Parse `bytes` strictly and pin the gate set: the supported way to
    /// obtain a `ValidatedStatement` outside [`verify`].
    pub fn validate(bytes: &[u8]) -> Result<Self, String> {
        let statement = parse_any_statement(bytes).map_err(|error| error.to_string())?;
        Self::seal(statement).map_err(|error| error.to_string())
    }

    /// Pin the gate set of an already-parsed statement ([`verify`]'s path,
    /// where parsing is instrumented separately).
    fn seal(statement: AnyStatement) -> Result<Self, CheckError> {
        let set = match &statement {
            AnyStatement::V2(statement) => gate_set(statement)?,
            AnyStatement::V3(statement) => gate_set(statement)?,
        };
        Ok(ValidatedStatement { statement, set })
    }

    /// The parsed statement (read-only).
    pub fn statement(&self) -> &AnyStatement {
        &self.statement
    }

    /// The pinned gate set (read-only).
    pub fn gate_set(&self) -> &GateSet {
        &self.set
    }

    /// Pass 1: authenticate the witness against this statement, with the
    /// optional untrusted plan.  Dispatches on (statement format, witness
    /// format, plan) exactly as the guest always has; any pairing the wire
    /// formats do not support is a rejection, never a fallback.
    pub fn authenticate<'a>(
        &'a self,
        witness: &AnyWitness,
        plan: Option<&Plan>,
    ) -> Result<AuthenticatedValues<'a>, CheckError> {
        self.authenticate_with(witness, plan, PlanPath::rust())
    }

    fn authenticate_with(
        &self,
        witness: &AnyWitness,
        plan: Option<&Plan>,
        plan_path: PlanPath<'_>,
    ) -> Result<AuthenticatedValues<'_>, CheckError> {
        let values = match &self.statement {
            // Statement format 3 (ranges): a packed witness and no plan; the leaf table is
            // read off the entries (`openings::ranged_table`), each distinct leaf
            // authenticated once, the words read by index.  Any other pairing is
            // refused before hashing.
            AnyStatement::V3(statement) => match (witness, plan) {
                (AnyWitness::Packed(packed), None) => {
                    check_openings_ranged(statement, packed, &self.set)
                }
                (_, Some(_)) => Err(CheckError::Plan("a format-3 statement takes no plan")),
                (_, None) => Err(CheckError::WitnessShape(
                    "a format-3 statement takes a packed (format-3) witness".to_string(),
                )),
            },
            AnyStatement::V2(statement) => match (witness, plan) {
                (AnyWitness::Openings(opened), None) => {
                    check_openings(statement, opened, &self.set)
                }
                (AnyWitness::Openings(_), Some(_)) => {
                    Err(CheckError::Plan("a plan needs a multiproof witness"))
                }
                (AnyWitness::Multiproof(multi), None) => {
                    check_openings_multi(statement, multi, &self.set)
                }
                (AnyWitness::Packed(packed), None) => {
                    check_openings_packed(statement, packed, &self.set)
                }
                (AnyWitness::Packed(_), Some(_)) => {
                    Err(CheckError::Plan("a plan needs a multiproof witness"))
                }
                (AnyWitness::Multiproof(multi), Some(plan)) => {
                    plan_path.check(statement, multi, plan, &self.set)
                }
            },
        }?;
        Ok(AuthenticatedValues {
            statement: self,
            values,
        })
    }
}

impl AuthenticatedValues<'_> {
    /// The decoded values, one row per obligation (read-only; for tools and
    /// tests that inspect what was opened).
    pub fn values(&self) -> &[Vec<u64>] {
        &self.values
    }

    /// Pass 2 and the verdict: recompute every obligation's gates from these
    /// authenticated values under the statement's pinned gate set.
    pub fn check_relations(self) -> Result<(), CheckError> {
        let AuthenticatedValues { statement, values } = self;
        let result = match &statement.statement {
            AnyStatement::V2(inner) => check_relations(inner, &values, &statement.set),
            AnyStatement::V3(inner) => check_relations(inner, &values, &statement.set),
        };
        // In the zkVM the decoded operands are not needed again and the guest
        // halts right after the commit: skip their frees (L3 measured them at
        // 25.6 M cycles with no effect on the public values).
        #[cfg(target_os = "zkvm")]
        core::mem::forget(values);
        result
    }
}

// -- the decision and the entry point -------------------------------------------------

/// The bound public result of one [`verify`] run: the exact bytes the guest
/// commits -- `sha256(statement bytes) || verdict`, or with a supplied typed
/// object digest carried through unchecked, `h_O || sha256(statement bytes)
/// || verdict`.
pub struct Decision {
    bytes: [u8; PUBLIC_VALUES_WITH_OBJECT_LEN],
    len: usize,
    verdict: bool,
}

impl Decision {
    fn new(object_digest: Option<[u8; 32]>, statement_digest: &[u8; 32], verdict: bool) -> Self {
        let mut bytes = [0u8; PUBLIC_VALUES_WITH_OBJECT_LEN];
        let len = match object_digest {
            None => {
                bytes[..PUBLIC_VALUES_LEN]
                    .copy_from_slice(&public_values(statement_digest, verdict));
                PUBLIC_VALUES_LEN
            }
            Some(h_o) => {
                bytes.copy_from_slice(&public_values_with_object(
                    &h_o,
                    statement_digest,
                    verdict,
                ));
                PUBLIC_VALUES_WITH_OBJECT_LEN
            }
        };
        Decision {
            bytes,
            len,
            verdict,
        }
    }

    /// The public values to commit, in the layout the inputs selected.
    pub fn public_values(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    /// The verdict (`true` = every check passed), as encoded in the last
    /// public byte (a diagnostic build encodes it as `0xEE`/`0xEF`).
    pub fn verdict(&self) -> bool {
        self.verdict
    }
}

/// How the tree table of a planned (multiproof + plan) batch is validated:
/// the kernel's own [`openings::validate_plan`], or -- only in a
/// `lean-plan-bridge` build, which is never the production guest -- an
/// external validator (Lean's `validPlanExec` through the L2 bridge).  In a
/// production build this struct is empty: there is no injection point.
#[derive(Clone, Copy)]
struct PlanPath<'x> {
    #[cfg(feature = "lean-plan-bridge")]
    external: Option<PlanValidator<'x>>,
    _marker: core::marker::PhantomData<&'x ()>,
}

impl<'x> PlanPath<'x> {
    fn rust() -> Self {
        PlanPath {
            #[cfg(feature = "lean-plan-bridge")]
            external: None,
            _marker: core::marker::PhantomData,
        }
    }

    fn check(
        &self,
        statement: &Statement,
        witness: &MultiproofWitness,
        plan: &Plan,
        set: &GateSet,
    ) -> Result<Vec<Vec<u64>>, CheckError> {
        #[cfg(feature = "lean-plan-bridge")]
        if let Some(validator) = self.external {
            return validator(statement, witness, plan, set);
        }
        check_openings_planned(statement, witness, plan, set)
    }
}

/// An external plan validator (see [`PlanPath`]): must accept exactly the
/// canonical tree table and return exactly what
/// [`openings::check_openings_planned`] returns.
#[cfg(feature = "lean-plan-bridge")]
pub type PlanValidator<'x> = &'x dyn Fn(
    &Statement,
    &MultiproofWitness,
    &Plan,
    &GateSet,
) -> Result<Vec<Vec<u64>>, CheckError>;

/// THE acceptance boundary: decide one batch from untrusted bytes.
///
/// * `statement_bytes`, `witness_bytes`: the canonical wire encodings
///   (`crate::codec`), any statement format (v2/v3) and witness format
///   (1/2/3) the codec knows.
/// * `plan_bytes`: untrusted prover advice, empty when absent.  A plan is
///   only accepted with a format-2 witness; it changes neither the statement,
///   its digest, nor the verdict semantics, and an invalid plan rejects the
///   batch exactly like a bad witness.
/// * `object_digest_bytes`: empty, or a 32-byte typed-object digest `h_O`
///   carried through the public values *unchecked* (the guest neither
///   validates the typed object nor proves `h_L` is its lowering; that seam
///   is the verifier's, `crate::binding`).  Any other length is a malformed
///   input: rejected before anything else is examined.
///
/// The verdict is `1` iff the statement parses, its gate-set digest pins the
/// semantics about to be applied, every opening authenticates under its
/// commitment root with the exact framing its schema names, every public
/// input matches, and every obligation's gates, recomputed from its opened
/// inputs, agree with its opened outputs.  Everything else -- malformed
/// bytes, an unknown or mismatched gate set, an unsupported
/// statement/witness/plan pairing, a failed opening, a wrong value, a
/// violated relation -- is verdict `0` over the same statement digest.
pub fn verify(
    statement_bytes: &[u8],
    witness_bytes: &[u8],
    plan_bytes: &[u8],
    object_digest_bytes: &[u8],
) -> Decision {
    verify_with(
        statement_bytes,
        witness_bytes,
        plan_bytes,
        object_digest_bytes,
        PlanPath::rust(),
    )
}

/// [`verify`] with an external plan validator (the `lean-plan` guest's Lean
/// bridge); only compiled under `lean-plan-bridge`, which the production
/// build never enables.  Every other phase is byte-identical to [`verify`].
#[cfg(feature = "lean-plan-bridge")]
pub fn verify_with_plan_validator(
    statement_bytes: &[u8],
    witness_bytes: &[u8],
    plan_bytes: &[u8],
    object_digest_bytes: &[u8],
    validator: PlanValidator<'_>,
) -> Decision {
    verify_with(
        statement_bytes,
        witness_bytes,
        plan_bytes,
        object_digest_bytes,
        PlanPath {
            external: Some(validator),
            _marker: core::marker::PhantomData,
        },
    )
}

fn verify_with(
    statement_bytes: &[u8],
    witness_bytes: &[u8],
    plan_bytes: &[u8],
    object_digest_bytes: &[u8],
    plan_path: PlanPath<'_>,
) -> Decision {
    scope_start!("digest");
    let statement_digest = sha256(statement_bytes);
    scope_end!("digest");

    scope_start!("parse");
    scope_start!("parse.statement");
    let statement = parse_any_statement(statement_bytes);
    scope_end!("parse.statement");
    scope_start!("parse.witness");
    let witness = parse_any_witness(witness_bytes);
    scope_end!("parse.witness");
    scope_start!("parse.plan");
    let plan = if plan_bytes.is_empty() {
        Ok(None)
    } else {
        parse_plan(plan_bytes).map(Some)
    };
    scope_end!("parse.plan");
    scope_end!("parse");

    let parsed = statement.and_then(|statement| {
        witness.and_then(|witness| plan.map(|plan| (statement, witness, plan)))
    });

    // Optional fourth input: a 32-byte typed-object digest `h_O` to carry
    // through the public values unchecked (empty = the 33-byte layout).  Any
    // other length is a malformed input: reject before examining the batch.
    let object_digest: Option<[u8; 32]> = match object_digest_bytes.len() {
        0 => None,
        32 => Some(object_digest_bytes.try_into().unwrap()),
        n => {
            reject_log!(format_args!("typed object digest has {n} bytes, not 0 or 32"));
            return Decision::new(None, &statement_digest, false);
        }
    };

    let verdict = match parsed {
        Err(error) => {
            reject_log!(error);
            false
        }
        Ok((statement, witness, plan)) => {
            let verdict = match ValidatedStatement::seal(statement) {
                Err(error) => {
                    reject_log!(error);
                    false
                }
                Ok(validated) => {
                    scope_start!("merkle");
                    let opened = validated.authenticate_with(&witness, plan.as_ref(), plan_path);
                    scope_end!("merkle");
                    let verdict = match opened {
                        Err(error) => {
                            reject_log!(error);
                            false
                        }
                        Ok(values) => {
                            scope_start!("gates");
                            let result = values.check_relations();
                            scope_end!("gates");
                            match result {
                                Ok(()) => true,
                                Err(error) => {
                                    reject_log!(error);
                                    false
                                }
                            }
                        }
                    };
                    // The guest halts right after the commit: the parsed
                    // statement is never needed again, so skip its frees.
                    #[cfg(target_os = "zkvm")]
                    core::mem::forget(validated);
                    verdict
                }
            };
            #[cfg(target_os = "zkvm")]
            {
                core::mem::forget(witness);
                core::mem::forget(plan);
            }
            verdict
        }
    };

    Decision::new(object_digest, &statement_digest, verdict)
}

// -- whole-batch compositions (test and tool entry points) ----------------------------

/// Both passes: the verdict for one already-parsed batch (format-2 statement,
/// format-1 witness).  A composition of the kernel's phases for the test
/// suites; the byte-level boundary is [`verify`].
pub fn check_batch(statement: &Statement, witness: &Witness) -> Result<(), CheckError> {
    let set = gate_set(statement)?;
    let decoded = check_openings(statement, witness, &set)?;
    check_relations(statement, &decoded, &set)
}

/// Both passes for a format-2 witness.
pub fn check_batch_multi(
    statement: &Statement,
    witness: &MultiproofWitness,
) -> Result<(), CheckError> {
    let set = gate_set(statement)?;
    let decoded = check_openings_multi(statement, witness, &set)?;
    check_relations(statement, &decoded, &set)
}

/// Both passes for a format-3 witness.
pub fn check_batch_packed(
    statement: &Statement,
    witness: &PackedWitness,
) -> Result<(), CheckError> {
    let set = gate_set(statement)?;
    let decoded = check_openings_packed(statement, witness, &set)?;
    check_relations(statement, &decoded, &set)
}

/// Both passes for a witness in any format.
pub fn check_any(statement: &Statement, witness: &AnyWitness) -> Result<(), CheckError> {
    match witness {
        AnyWitness::Openings(witness) => check_batch(statement, witness),
        AnyWitness::Multiproof(witness) => check_batch_multi(statement, witness),
        AnyWitness::Packed(witness) => check_batch_packed(statement, witness),
    }
}

/// Both passes for a format-3 statement, which takes a format-3 witness only.
pub fn check_batch_ranged(
    statement: &Statement<PositionEntry>,
    witness: &PackedWitness,
) -> Result<(), CheckError> {
    let set = gate_set(statement)?;
    let decoded = check_openings_ranged(statement, witness, &set)?;
    check_relations(statement, &decoded, &set)
}

/// Both passes for a statement in either format and a witness in any format:
/// what the guest does without a plan.  A format-3 statement pairs with a
/// format-3 witness and nothing else (its ranges name words of packed leaves;
/// the guest does not expand it to serve the per-position formats).
pub fn check_any_statement(
    statement: &AnyStatement,
    witness: &AnyWitness,
) -> Result<(), CheckError> {
    match (statement, witness) {
        (AnyStatement::V2(statement), witness) => check_any(statement, witness),
        (AnyStatement::V3(statement), AnyWitness::Packed(witness)) => {
            check_batch_ranged(statement, witness)
        }
        (AnyStatement::V3(_), _) => Err(CheckError::WitnessShape(
            "a format-3 statement takes a packed (format-3) witness".to_string(),
        )),
    }
}

/// Both passes for a format-2 witness under an untrusted plan.
pub fn check_batch_planned(
    statement: &Statement,
    witness: &MultiproofWitness,
    plan: &Plan,
) -> Result<(), CheckError> {
    let set = gate_set(statement)?;
    let decoded = check_openings_planned(statement, witness, plan, &set)?;
    check_relations(statement, &decoded, &set)
}
