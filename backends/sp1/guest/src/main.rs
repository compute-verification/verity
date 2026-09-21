//! The SP1 guest: a thin wrapper around the verification kernel
//! (`veritor_zk_common::kernel::verify`), which is the one acceptance
//! boundary -- a Rust interpreter for the circuit semantics; SP1 proves its
//! execution.
//!
//! Input (four `read_vec` buffers): the canonical statement bytes, the
//! canonical witness bytes, in any witness format of `codec` (one path per
//! opening, one Merkle multiproof per tree, or -- format 3 -- whole packed
//! leaves under the serving side's framing with one path each), an optional
//! **plan** (empty when absent), and an optional 32-byte **typed object
//! digest** `h_O` (empty when absent).  Output (public values):
//! `sha256(statement) || verdict` -- or, when `h_O` is supplied,
//! `h_O || sha256(statement) || verdict`, `(h_O, h_L, verdict)`, with `h_O`
//! carried through unchecked (the guest neither validates the typed object
//! nor proves `h_L` is its lowering; IR §I).  What the verdict means, and
//! every parse, authentication and relation check behind it, is the kernel's:
//! this file only reads the buffers, calls `verify`, commits the decision's
//! public values and halts.
//!
//! The plan is untrusted prover advice (`codec::Plan`): the statement's tree
//! table spelled out so the kernel validates it in one linear pass
//! (`kernel::openings::validate_plan`) instead of computing it.  It is only
//! accepted with a format-2 witness, it changes neither the statement, its
//! digest nor the verdict semantics, and an invalid plan rejects the batch
//! exactly like a bad witness.  Without a plan the kernel runs the reference
//! opening pass.
//!
//! (Veritor's `lean-plan` build, which swapped the plan validator for the
//! Lean-compiled one through the kernel's `lean-plan-bridge` seam, is not
//! carried here; the production build never enabled it.)
//!
//! `cycle-tracker-report-*` prints let the host executor attribute cycles to
//! the five phases (`io` here; `digest`, `parse`, `merkle`, `gates` inside
//! the kernel via the `scopes` feature) and, nested under them, to
//! `parse.statement` / `parse.witness` / `parse.plan` and to `merkle.table`
//! (reference) or `merkle.plan` (planned), `merkle.auth`, `merkle.decode`.
//!
//! # Diagnostic builds (never sound, never a result)
//!
//! Built with `--features unsound-skip-authentication` (no leaf hashing, no
//! path folding) or `--features unsound-skip-path-fold` (leaves hashed, paths
//! never folded), the guest measures the cycle floor of the authentication
//! pass and accepts witnesses that prove nothing.  It prints
//! `UNSOUND DIAGNOSTIC BUILD: ...` first and commits the verdict byte `0xEE`
//! (accept) or `0xEF` (reject) instead of `1` / `0`, so the output can never
//! pass for a sound guest's; the host reports it as `unsound`.  See
//! `zk/sp1/README.md`.

#![no_main]
sp1_zkvm::entrypoint!(main);

use veritor_zk_common::DIAGNOSTIC_BANNER;

#[cfg(not(feature = "unsound-relation-only"))] fn main() { // (same line: the sound ELF's panic locations stay put)
    if let Some(banner) = DIAGNOSTIC_BANNER {
        println!("{banner}");
    }

    println!("cycle-tracker-report-start: io");
    let statement_bytes = sp1_zkvm::io::read_vec();
    let witness_bytes = sp1_zkvm::io::read_vec();
    let plan_bytes = sp1_zkvm::io::read_vec();
    let object_digest_bytes = sp1_zkvm::io::read_vec();
    println!("cycle-tracker-report-end: io");

    let decision = veritor_zk_common::kernel::verify(
        &statement_bytes,
        &witness_bytes,
        &plan_bytes,
        &object_digest_bytes,
    );

    sp1_zkvm::io::commit_slice(decision.public_values());

    // Halt right after the commit (`__start` would call `syscall_halt(0)` after
    // `main` returns anyway): the input buffers are never dropped, and the
    // kernel forgets the parsed statement, witness and plan before returning
    // (target_os = "zkvm").  L3 measured those frees at 25.6 M cycles (8.5 %
    // of the stock guest) with no effect on the public values.
    sp1_zkvm::syscalls::syscall_halt(0);
}

/// RELATION-ONLY DIAGNOSTIC BUILD (`--features unsound-relation-only`; never sound, never a result).
///
/// Nothing of the checker runs: one `read_vec` buffer, the **compact input**
/// `K: u32 LE || x[0..K]: u16 LE || W[0..K]: u16 LE` of one `GemmCoordinate<K>` obligation in
/// `veritor.tc-ampere-bf16@1`, goes to `veritor_zk_common::relation_only::decide`, which evaluates
/// the bare gate chain (`tc_dot16_0`, chained `tc_dot16`, `f32_to_bf16`, through the kernel's own
/// `GateSet::evaluate`) and lays out `sha256(input) || sha256(output word) || 0xEE` -- the 65-byte
/// shape of `h_O || h_L || verdict`, so the host is unchanged in shape.  The host's `--witness` is
/// the compact input; the statement is ignored.  Cycle scopes: `relation_only.io` here, then
/// `relation_only.parse`, `gates` and `relation_only.digest` inside `decide`.  This is the
/// arithmetic baseline of the zkVM path; it authenticates nothing and verifies no statement.
/// Declared after the sound `main` so the sound build's source lines (its ELF) are untouched.
#[cfg(feature = "unsound-relation-only")]
fn main() {
    println!("{}", DIAGNOSTIC_BANNER.expect("a relation-only build is a diagnostic build"));

    println!("cycle-tracker-report-start: relation_only.io");
    let input = sp1_zkvm::io::read_vec();
    println!("cycle-tracker-report-end: relation_only.io");

    let public_values = veritor_zk_common::relation_only::decide(&input);
    sp1_zkvm::io::commit_slice(&public_values);
    // halt right after the commit, as the sound guest does (the input buffer is never dropped)
    sp1_zkvm::syscalls::syscall_halt(0);
}
