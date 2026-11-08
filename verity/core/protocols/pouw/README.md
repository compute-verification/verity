# PoUW (`verity.core.protocols.pouw`)

Proof of useful work: certifies that a served model's registered matrix products were computed, by sampled proofs over
committed, noised checked values. An accepted run establishes a credited size S of noised matrix products, at least
(1 − ε)·S of their AND gates computed right, except with probability δ. One protocol carries several named schemes
behind one interface (`protocol.PoUWScheme`, `schemes.get`). The proof system registers each execution phase, samples
its proof blocks with the challenger's coins, proves them in zero knowledge (Flock `--zk`) and verifies them; PoUW
states what each execution phase commits, the Program each sampled proof block must pass, and what the outcome credits.
It carries no policy parameter: it surfaces the ε at δ its sample established. A scheme must leave the useful work at
least half of the honest prover's GPU effort (AGENTS.md, "PoUW is mostly useful work"): the trivial PoUW, which hashes
every product, is a reference, not an answer.

## Start here

- **The verifier** is Lean (What runs, below). `PouwCheck.Period.protocol` (`verity/pouw/period/v1`) is the subprotocol
  the onsite and the remote protocol compose: it decides each execution phase by `PouwCheck.check` and credits a closed
  credit period by `PouwCheck.Credit`. Its executable is `pouw-check`; the Python modules of the same names mirror it,
  held to it by difftests.
- **The guarantees.** Each `G` is stated as `Pouw.Guarantees.G` or `Accounting.Guarantees.G` under
  `Security/Definitions/` and proved as its `SecurityProofs.G` under `Security/Proofs/`; `Security/lean-audit.json`
  records each statement and its premises. W is the credited reference work, t the work-weighted samples, ε_s the share
  of W a sample may miss, and γ = 1 − (1 − γ₀)/Ω_max.
  1. **The audit game**, `Pouw.Guarantees.EndToEnd` (`Pouw/Guarantees/Game.lean`): for every in-domain workload and
     every adversary within its budget and q ≤ 2^64 oracle calls, Pr[the audit accepts ∧ the work done is below
     (1 − γ)(1 − ε_s)·W] ≤ (1 − ε_s)^t + ε(q, N). Assumes TT(γ₀) (`Pouw.Assumptions.TT`), the scheme's conjecture;
     `Pouw.Guarantees.GammaFromTT` is its first step, the scheme's G_γ from TT.
  2. **Per run, for compute accounting**, `Accounting.Guarantees.ComputeOfPouwPerRun`
     (`Accounting/Guarantees/PerRun.lean`): on any device whose compute audit realizes PoUW's game, the compute
     audit's per-run guarantee holds at certificate (1 − γ)(1 − ε_s)·W with error (1 − ε_s)^t + ε(q, N), q the device's
     oracle rate times the run's T ticks. Assumes TT and `Accounting.BudgetAbort` (the cost model truncates).
  3. **A credit period's binding**, `Pouw.Guarantees.PeriodBindingPerProver` (`Accounting/Guarantees/PeriodBinding.lean`):
     binding of a credit period's openings fails in each verification phase with probability at most
     |L|·√(q/2^256 + κ), q the SHA-512 calls of the prover's own fork finder at the period's positions L. Assumes
     `Claims.ExpectedCR` at that finder; κ (accepted openings no path verifies, and positions that split) is stated in
     full and not yet bounded.
  4. **The served credit's spot check**, `Pouw.Guarantees.RelativeSecurity.TheoremSStaged` and `TheoremSOverlap`
     (`Pouw/Guarantees/RelativeSecurity.lean`): a play whose errors are of degree at most 2 is accepted with deficiency
     at least D with probability at most e^{−D/4}. No premise.

  Also proved: the deadline (`WorkBetweenSaltAndDeadline`, `SpareCapacityAtDeadline`, `PearlCSampledByDeadline`) and
  each scheme's γ at its served point (NCP's `NCP.GammaFromTTNCP_U_v1`; Pearl-C's
  `PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_8192` and its sampled twin), under the schemes' TT forms
  (`Pouw.NCP.Assumptions.TTNCP_U`, Pearl-C's `Pouw.PearlC.Assumptions.TTOutPearlCDevRev1W1`, Pearl-C4's
  `Pouw.PearlC.Assumptions.TTOutFp4Sm120W1`), the seeded noise (`Pouw.PearlC.RowSeedNoise`, `Pouw.PearlC.CodeSeedNoise`),
  the receipt clock (`Pouw.Assumptions.HonestClock`), the sampled tile's proof
  (`Pouw.PearlC.Assumptions.TileProofSoundAll`) and the claims `trusted-host`, `live-verifier`, `cr/sha-512` and
  `prf/sha-512` (`verity.core.claims`). Stated and open: an execution phase's ε from its calls'
  (`Pouw.Guarantees.Window.WindowEps`, `Security/Definitions/Pouw/Guarantees/Window.lean`), Dither's
  (`Security/Definitions/Pouw/Guarantees/DitherInstance.lean`), and several of Pearl-C's per-shape and per-model
  instances (`Security/Definitions/Pouw/Guarantees/PearlC.lean`).
- **Where next.** `lean/check/README.md` maps each Lean module to its Python. `Security/README.md` (Start here) gives
  the guarantees PoUW rests on (SPOT's and Flock's) and the composite protocols' soundness, whose PoUW target is
  `PouwSound`. `benchmarks/pouw/` measures it.

## PoUW in the remote protocol

The remote protocol, for an auditor with no devices at the datacenter, composes three subprotocols: sampled proofs'
one-stage audit, PoUW's credit period check, `PouwCheck.Period.protocol` (`verity/pouw/period/v1`), as onsite does, and the
inference-only policy's check (`Core.Remote.components` in `verity/core/protocols/remote/CoreRemote.lean`, id
`verity/remote/v0`). It accepts only when every subprotocol accepts its own case, its commitment binds the
subprotocols' commitments, the policy and the one-stage audit committed to one record, and every execution phase of the credit period
names that record as its Commit record (`Core.Remote.windows`); any one's rejection rejects it
(`Core.Composition.tiedSameB`, `verity/core/protocols/composition/README.md`). In the remote demo, the vLLM demo
(`verity example vllm`), PoUW runs on the served model as H1, under the challenger's development key, so its verdict
credits nothing, and an H1 commitment's served execution blocks aren't yet a `verity/pouw/period/v1` case. The remote
protocol's front door is
`verity/core/protocols/remote/`; PoUW is also a subprotocol of the onsite protocol (`verity/core/protocols/onsite`).

## What runs

The verifier is Lean, the source of truth for what PoUW checks: three Lake packages under `lean/`.

- `lean/check` (`pouw_check`): `PouwCheck.Period.protocol` (`verity/pouw/period/v1`), the `Core.Protocol` both the
  onsite and the remote protocol compose, and `PouwCheck.protocol` (`verity/pouw/v1`), one execution phase's check,
  which neither composes. `PouwCheck.check` decides one execution phase from its registration (the commitment) and the proof
  system's answer to it (the evidence); `PouwCheck.Credit` decides what a served execution phase, and a credit period
  (below), credits. Its executable is `pouw-check` (`id | check | serve`, the core interface's commands).
- `lean/window` (`pouw_window`): what an execution phase registers and makes public: its grid level, its sample size K′, its
  fixed length, its committed entries and side trees, and its round coins (`PouwWindow.Grid`, `PouwWindow.Draw`,
  `PouwWindow.Length`, `PouwWindow.Public`, `PouwWindow.Entry`, `PouwWindow.Side`, `PouwWindow.Rounds`).
- `lean/commit` (`pouw_commit`): PoUW's commitments: tagged bindings, frames, rows, execution-phase coins and samples, all SHA-512.

The Python modules of the same names (`core_check.py`, `credit.py`, `window.py`, `coins.py`, `audit.py`) are their
references, held to them by the difftests (`lean/check/tests`, `lean/window/tests`, `lean/commit/tests`).
`verifier.py` is the entry the Python side calls: `verifier.lean(fn, value)` runs the Lean, and `verifier.coins(n)` is
the challenger's entropy, which the Lean reads from the operating system. `RECORD_VERDICT` (`"lean"`) is the switch, and
setting it to `"python"` is the one line that undoes it.

An **execution phase** is the served steps one commitment covers; the code and the Lean name it a window (`window.py`,
`PouwWindow`, `credit.window`) until a format version. In one execution phase:

1. The weights' root binds every weight's committed rows, and a credit period's open signs it.
2. The salt is a live coin of the challenger, opened once after the root it follows (`PouwCheck.Rounds`,
   `PouwCheck.Pass`), or, for a certified execution phase, the network certifier's salt (`salt.py`).
3. The activation rows and the execution phase's entries are committed under one root, and the execution phase registers its
   level on the grid, its sampling design (a uniform subset of K′ of its N tiles, K′ the least K that misses a wrong
   share ε of the credited work with probability at most δ) and its fixed length before any coin is flipped.
4. The sample's coin is opened once, after the registration's receipt, and samples the proof blocks, one tile each.
5. Each sampled proof block is proved in zero knowledge over the circuit its tile template pins (`PouwCheck.Tile`), and
   opened against the committed entries (`PouwCheck.Opening`).
6. `PouwCheck.check` gives the verdict. A served execution phase is credited only if every step accepts. A credit period
   (the code's credit period, `credit.period_credit`) is the execution phases between an open, which signs the weights' root and
   commits the circuit files, and a close; it counts only closed and complete.

How it is run: the difftests run the Lean against the Python on generated cases
(`uv run tools/verity/check/suites.py verity/core/protocols/pouw/lean/check`, and likewise `lean/window` and
`lean/commit`); the composition's difftest runs `remote serve` on PoUW's cases beside sampled proofs'
(`verity/core/protocols/composition/tests/test_composition_difftest.py`); `benchmarks/pouw/served_zk/of_record.py` runs
each check of a served execution phase through the Lean and its Python reference and records the Lean's verdict.

## Schemes

A scheme turns one registered matrix product's committed operand rows into each tile's checked values, its leaf and its
useful output; everything else (commitments, salt, sample, openings) is the protocol's and the same for every scheme.

- **Pearl-C** (`pearl-c-h100-v0`, `pearl-c-h100-v1`, `pearl-c-sm120-v1`, and their `-h3` forms,
  `verity/ml/protocols/pouw/schemes/pearl_c.py`): Pearl's FP8 (E4M3) proof of useful work, the GEMM-based scheme Pearl
  publishes (`pearl-fp8-v4`, `verity/ml/protocols/pouw/schemes/pearl_kw.py`), adopted as a PoUW scheme, with its noisy
  clean-up pinned bit for bit so that the useful output is a checked value. It runs on the H100 and on the RTX PRO 6000
  Blackwell (sm_120). The `-h3` forms hash with SHA-512 throughout, and h3 is the default wherever a hashing format or
  tree hash goes unnamed. The others (h0, taken by name) keep Pearl's keyed-BLAKE3 seeds, lines and tickets, with
  TurboSHAKE128 leaves and SHA-256 trees.
- **Pearl-C4** (`pearl-c-nvfp4-v0`, `verity/ml/protocols/pouw/schemes/pearl_c4.py`): Pearl-C with the FP8 product
  replaced by sm_120's block-scaled FP4, NVFP4 only.
- **NCP** (`ncp-v1`, `verity/ml/protocols/pouw/schemes/ncp.py`): noise-cancelling polarization over int7 activations, on
  SHA-256 trees (its `commitment_hash`: its tile digests are 32 bytes).
- **Dither**, PoUW's no-fold design over the ACCEL model, is experimental, with its own spec, `dither/PROTOCOL.md`.

Each scheme module's docstring states its steps and roundings, with pinned vectors under `tests/vectors`.

## Core API

~~~python
from verity.core.protocols.pouw import audit, core_check, credit, verifier, window
from verity.ml.protocols.pouw import schemes

scheme = schemes.get("pearl-c-sm120-v1-h3")                 # a configured PoUWScheme
root = audit.weights_root(weight_roots)                    # the weights' root a credit period's open signs
verdict = core_check.check(params, commitment, evidence)   # one execution phase, as `pouw-check` answers
outcome = credit.window(key, ledger, pins, main, ...)      # a served execution phase's credited set and its sampled proof blocks' outcome
period = credit.period_credit(key, ledger, pins, main, name, windows, verify, ...)   # a credit period, closed and complete
work = window.bucket(n_real)                               # the grid level: credited tiles and padded N
k = window.k_prime(eps, delta, work.credited, work.n, tile_work)   # K′, the sample that establishes ε at δ
~~~

## Where things are

- `protocol`: the `PoUWScheme` interface.
- `audit`, `coins`, `salt`, `window`, `core_check`, `credit`, `identifier`, `registration`, `weights`, `operands`,
  `live_reads`: the Python references the Lean is held to, and the reference rules not yet ported.
- `verity/ml/protocols/pouw/`, the prover's, which core doesn't import: the named schemes (`schemes/`) and `circuit/`,
  PoUW as `verity.ml.circuits` Programs: the tile and lane Definitions, the compile pass over a served Program
  (`compile`) and the sampling plan (`plan`). Each Definition has its `circuit-check` sidecar.
- `lean/check`, `lean/commit`, `lean/window`: the Lean verifier, each with its difftest; `lean/check/README.md` and
  `lean/window/README.md` map each Lean module to its Python.
- `dither/`: Dither, experimental, with its `PROTOCOL.md`.
- `tests/`: the package's suite, with the schemes' and the execution phase's pinned vectors under `tests/vectors`.
  `benchmarks/pouw/` is the measuring side; nothing here imports it.
