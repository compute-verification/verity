# Prove path

Network accounting under arbitrary vLLM usage proves a served run's step units after the fact: a precommitted compiler turns
the run's advice into each step's Programs, the committing vLLM has recorded each step's values, and the proof service draws
units on live coins, proves them with C-Flock `--zk` behind the Lean firewall, and the Lean verifier accepts or refuses them.
`prove_path.py` is that path on a committed TP run: the prover commits each rank's window over the compiler's Programs for the
advice, and the verifier compiles the advice itself and audits each rank through the one-stage audit's per-rank
`service.Stream` on its own coins (`benchmarks/one_stage/a5.py`). `run.sh` builds the prover and the Lean verifier and runs it
on the CPU TP stand-in or on a deferred TP Commit (`PP_TP`), with one forged run that must be refused. The docstrings say how.

## What changes when Programs come from advice

A declared run's audit (`a5.py --tp-commit`) derives each rank's Programs from the workload the record served. Under arbitrary
usage no workload is declared: serving decides the batches, arrivals and lengths, and the advice (I1) records those decisions.
Four things change, and nothing else in the audit does.

1. **Registration binds the compiler and the advice, not a workload.** The rank's window provenance (core's
   `verity/registration/v1` item `window`, which the record's digest covers and the coins commit to before the draw) carries
   `compiler: {name, digest}` and `advice: {digest, bits}` (`pipeline.advice_windows.binding`): the SHA-512 of the compiler
   fixed before the run, and the SHA-512 and bit length of the advice's encoding. The record gains no field. Both are public:
   the compiler is registered, and the advice's bits are the leak the certifier charges.
2. **Each step's Programs are the compiler's output.** The window's `served_programs` are the digests the compiler produced
   for the advice, and the population program and partition the registration is checked against (R1–R7) are over those
   Programs. The Stream's Check names the unit templates' Program digests, as it does for a declared run.
3. **The verifier recompiles; it doesn't take the prover's digests.** It runs its own copy of the registered compiler on the
   advice, derives its own window (`advice_windows.window_of_advice`, from the record, the advice and the compiler alone, reading
   no committed value), and holds the registration to it (`tp.serving_rows.source`: `served_programs`, `compiler` and
   `advice` among its codes, with R2 against the recompiled population). A recompile checks the prover's digests too, so the
   weaker option, checking the prover's digests against a recompile, adds nothing.
4. **What the advice can and can't do.** Advice that names a request the Commit didn't serve, or a served one at another
   arrival, commits no window: the committed store holds no such value (`LookupError` at the prover). A unit committed under
   other advice and spliced in fails R4. A registration naming another compiler, other advice, or none fails `source`
   before any coin is read. The advice can only choose among Programs the compiler emits, and every unit's values are still
   committed under hiding leaves and opened only inside the proof.

What doesn't change: the draw (the service's rule, the partition file's law, on live coins read after the receipt, ε surfaced
at δ by the outcome), the per-member `--zk` sessions behind the firewall, Lean's verdicts, the audit record and the Lean
`Run.run` agreement.

## Status

- **The compiler** is `advice_windows.Traced`, the Build's Programs of record, refused unless of record. On a declared run
  it is exactly what a correct compiler emits. I2's `predict.steps.compile_steps` (v0) compiles a request's Program from φ
  at its advised shape, but not yet a TP rank's Programs, which a rank's window needs; its adapter (`name`, `digest`,
  `load(rank, world, requests)`) replaces `Traced` once it does, with no change to the audit.
- **The advice's encoding** is its canonical JSON in I1's fields (`prove_path.encode`) until I1's encoding lands; the
  digest and bit length are over whichever encoding is registered.
- **Other unit kinds.** N2's head-check units and the I/O tie units are to be drawn on the same service path. N2's units have
  their own registration today (N2-ZK's `n2_zk.py`, on branch `cursor/network-e2e-061a`). The tie unit is a design (the
  certifier's protocol, on branch `cursor/io-tie-061a`).
