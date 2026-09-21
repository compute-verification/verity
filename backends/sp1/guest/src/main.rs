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

fn main() {
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
