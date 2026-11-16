# Flock's verifier

Flock is Verity's proof system: a prover shows that one statement about one circuit holds of committed values, and with
`--zk` it shows it in zero knowledge. This directory is Flock's verifier, written in Lean 4. Its definitions are what
the soundness proofs are about, and the same definitions, compiled, are the binary that checks a proof:
`verity-verify-proof`, which `verity verify proof` runs. It replays a recorded proof session offline, on the CPU, and
every coin it reads comes from the session's live record.

The ZK server's Lean is here too (`Flock.ZkServer`), and so is the recursion's (`FlockRec`, `VStar`).

## Try it

~~~sh
bash tools/verity/lean/setup.sh verity/protocols/proofs/flock/verifier
(cd verity/protocols/proofs/flock/verifier && lake build verity-verify-proof)
verity verify proof --help
~~~

The first line installs elan, the toolchain this directory pins and the packages it requires; it fetches nothing else.
The second builds the verifier on the CPU, and the third prints its commands. What each command checks and its flags
are in [the agreement's README](../prover/agreement/README.md), with how the verifier is held to upstream's Rust one.

## What it is proved to do

Its guarantees are stated in `security/Definitions/Flock/Guarantees/` and proved in `security/Proofs/Flock/`; the one
soundness theorem over everything it accepts is
[`FailClosed.flock_verify_sound`](../../../../../security/Proofs/Flock/Soundness/Discharge/FailClosed/One.lean).

The proofs are about the Lean definitions, so the running binary also trusts the pinned toolchain's compiler and the Lean
runtime it links, the C compiler that builds both, and the C library under them, and no declaration here swaps a
definition for code no proof reads (`@[extern]`, `@[implemented_by]`, `unsafe`, `native_decide`), which
`tests/verity/protocols/proofs/flock/prover/test_lean_verifier.py` refuses.

[`DESIGN.md`](DESIGN.md) has the packages it requires and how its libraries are layered.
