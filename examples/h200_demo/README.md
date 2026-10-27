# h200_demo: Demo A on the demo H200

One H200 serves the frozen MQA shape (52 layers, width 16,384) on seeded INT8 weights, committing every op's output as it
runs. After the step and its weights are registered, one live coin from the auditor's challenger draws 8 of the step's
3,124,276,224 proof blocks. Each drawn block is opened against the registered roots, replayed, proved zero-knowledge in
C-Flock `--zk` behind the firewall, and decided by the Lean verifier. The step driver is `benchmarks/e2e_h200`, the bundle
and record interfaces are `benchmarks/e2e_demo`, and the prover is `benchmarks/one_stage/prove_bundle.sh`. The example
calls them in place and moves none of them.

`run.sh` runs one phase at a time (`verity example h200-demo PHASE [--dry-run]`): commit, weights, register, coin, open,
replay, bundle and prove. Each phase is a separate run on its own machine, and its header names the run of record for
each phase and the variables that carry one phase's output to the next. Its standing is `example.toml`.

- **M3 proper (level 1, runs).** DRAWPROVE r20261027-013625-8e25 drew 8 blocks on a live coin, and C-Flock `--zk` proved
  each one and Lean accepted it. The flipped bit on block 239111068 was refused. Demo A's M3 report v3.1 is
  art:0e5de97332019dbd62a74222189ec65787fa52890f2acb8a083dda3244f4ed86 (#4964). At an 8-block draw the guarantee is weak:
  at δ = 1%, at most 43.77% of the step's proof blocks are wrong.
- **Not yet placed.** The Lean verifier ran on vy-nebius-cpu-4 beside the C-Flock prover. Level 2 needs it on a machine the
  prover doesn't control.
- **Demo B (memory-accounting's, M4).** PoUS and the bandwidth audit on the same H200 add Demo B's memory terms. Their run
  of record goes here and in `example.toml`'s `runs`, with `protocol` becoming a list. Nothing here claims them.

M3 proper's jobs are still on branches (`cursor/m3-proper-1de3`, and prove_bundle.sh's `PB_IN_ART` and `PB_COIN` on
`cursor/proofs-m3-proper-92aa`). A phase whose job isn't in the checkout says which branch holds it and starts nothing.
