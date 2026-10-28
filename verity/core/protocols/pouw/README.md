# PoUW (`verity.core.protocols.pouw`)

Proof of useful work: certifies that a served model's registered matrix products were computed, by sampled proofs over
committed, noised checked values. An accepted run establishes a credited size S of noised matrix products, at least
(1 − ε)·S of their AND gates computed right, except with probability δ. One protocol carries several named schemes
behind one interface (`protocol.PoUWScheme`, `schemes.get`). The proof system registers each execution phase, samples its
proof blocks with the challenger's coins, proves them in zero knowledge (Flock `--zk`) and verifies them; PoUW states what each
execution phase commits, the Program each sampled proof block must pass, and what the outcome credits. It carries no policy
parameter: it surfaces the ε at δ its sample established.

## PoUW in the remote protocol

The remote protocol, for an auditor with no devices at the datacenter, composes two subprotocols: sampled proofs'
one-stage audit and PoUW's commit-period check (`Core.Remote.components` in
`verity/core/protocols/remote/CoreRemote.lean`, id `verity/remote/v0`). It accepts only when both components accept
their own cases and its commitment binds both components' commitments; either one's rejection rejects it
(`Core.Composition.product`, `verity/core/protocols/composition/README.md`). PoUW's half of the remote demo has no
runner yet: the remote demo, `verity example vllm`, runs the sampled-proofs half. The remote protocol's front door is
`verity/core/protocols/remote/`; PoUW is also a component of the onsite protocol (`verity/core/protocols/onsite`).

## What runs

The verifier is Lean, three Lake packages under `lean/`:

- `lean/check` (`pouw_check`): `PouwCheck.protocol`, the `Core.Protocol` with id `verity/pouw/v1` that the remote
  protocol composes. `PouwCheck.check` decides one execution phase from its registration (the commitment) and the proof
  system's answer to it (the evidence); `PouwCheck.Credit` decides what a served execution phase, and a credit period
  (below), credits. Its executable is `pouw-check` (`id | check | serve`, the core interface's commands).
- `lean/window` (`pouw_window`): what an execution phase registers and makes public: its grid level, its sample size K′, its
  fixed length, its committed entries and side trees, and its round coins (`PouwWindow.Grid`, `PouwWindow.Draw`,
  `PouwWindow.Length`, `PouwWindow.Public`, `PouwWindow.Entry`, `PouwWindow.Side`, `PouwWindow.Rounds`).
- `lean/commit` (`pouw_commit`): PoUW's commitments: tagged bindings, frames, rows, epoch coins and samples, all SHA-512.

The Python modules of the same names (`core_check.py`, `credit.py`, `window.py`, `coins.py`, `audit.py`) are their
references, held to them by the difftests (`lean/check/tests`, `lean/window/tests`, `lean/commit/tests`).
`verifier.py` is the entry the Python side calls: `verifier.lean(fn, value)` runs the Lean, and `verifier.coins(n)` is
the challenger's entropy, which the Lean reads from the operating system. `RECORD_VERDICT` (`"lean"`) is the switch, and
setting it to `"python"` is the one line that undoes it.

An **execution phase** is the served steps one commitment covers (PoUW's older name, "commit period"). The code and the
Lean call it a window (`window.py`, `PouwWindow`, `credit.window`), and keep the name until a format version. In one
execution phase:

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
   (the code's period, `credit.period_credit`) is the execution phases between an open, which signs the weights' root and
   commits the circuit files, and a close; it counts only closed and complete.

How it is run: the difftests run the Lean against the Python on generated cases
(`uv run tools/verity/check/suites.py verity/core/protocols/pouw/lean/check`, and likewise `lean/window` and
`lean/commit`); the composition's difftest runs `remote serve` on PoUW's cases beside sampled proofs'
(`verity/core/protocols/composition/tests/test_composition_difftest.py`); `benchmarks/pouw/served_zk/of_record.py` runs
each check of a served execution phase through the Lean and its Python reference and records the Lean's verdict.

## Schemes

A scheme turns one registered matrix product's committed operand rows into each tile's checked values, its leaf and its
useful output; everything else (commitments, salt, sample, openings) is the protocol's and the same for every scheme.

- **Pearl-C** (`pearl-c-h100-v0`, `pearl-c-h100-v1`, `pearl-c-sm120-v1`, and their `-h3` forms, `schemes/pearl_c.py`):
  Pearl's FP8 (E4M3) proof of useful work, the GEMM-based scheme Pearl publishes (`pearl-fp8-v4`,
  `schemes/pearl_kw.py`), adopted as a PoUW scheme, with its noisy clean-up pinned bit for bit so that the useful output
  is a checked value. It runs on the H100 and on the RTX PRO 6000 Blackwell (sm_120). The `-h3` forms hash with SHA-512
  throughout; the others keep Pearl's BLAKE3 hashing.
- **Pearl-C4** (`pearl-c-nvfp4-v0`, `schemes/pearl_c4.py`): Pearl-C with the FP8 product replaced by sm_120's
  block-scaled FP4, NVFP4 only.
- **NCP** (`ncp-v1`, `schemes/ncp.py`): noise-cancelling polarization over int7 activations.
- **Dither**, PoUW's no-fold design over the ACCEL model, is experimental, with its own spec, `dither/PROTOCOL.md`.

Each scheme module's docstring states its steps and roundings, with pinned vectors under `tests/vectors`.

## Security assumptions and guarantees

Guarantees are stated in `Security/Definitions/Pouw` and proved in `Security/Proofs/Pouw`, as
`Pouw.SecurityProofs.G : Pouw.Guarantees.G`; `Security/lean-audit.json` lists each proved one with what its statement
reads. Proved:

- **The audit game:** `Pouw.SecurityProofs.Theorem1` (if a scheme's γ holds with error η and the sample misses more than
  a share ε_s of the reference work with probability at most δ_s, then the verifier accepts while the work done is below
  (1 − γ)(1 − ε_s)·W with probability at most δ_s + η), `Pouw.SecurityProofs.WorkWeightedSampling`,
  `Pouw.SecurityProofs.EndToEnd` and `Pouw.SecurityProofs.GammaFromTT`.
- **The deadline:** `Pouw.SecurityProofs.WorkBetweenSaltAndDeadline`, `Pouw.SecurityProofs.SpareCapacityAtDeadline` and
  `Pouw.SecurityProofs.PearlCSampledByDeadline`.
- **The schemes' γ:** NCP's `Pouw.SecurityProofs.NCP.GammaFromTTNCP_U_v1`; Pearl-C's at the served point,
  `Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_8192` and its sampled twin
  `Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_8192`; the row seed's reduction
  `Pouw.SecurityProofs.PearlC.TTOutRowSeedOfTTOut`; the barrier `Pouw.SecurityProofs.Barrier.Generic`.
- **The served credit's relative security:** `Pouw.SecurityProofs.RelativeSecurity.TheoremSStaged` and
  `Pouw.SecurityProofs.RelativeSecurity.TheoremSOverlap`.
- **Compute accounting reads PoUW** through `Accounting.SecurityProofs.ComputeOfPouw`,
  `Accounting.SecurityProofs.ComputeOfPouwPerRun`, `Accounting.SecurityProofs.PouwEndToEndPerRun` and
  `Accounting.SecurityProofs.PouwLocalOff`.

Stated, open: an execution phase's ε from its calls' (`Pouw.Guarantees.WindowEps`, and `Pouw.Guarantees.ServiceEscapes` for
the deployed sample, `Security/Definitions/Pouw/Guarantees/Window.lean`); Dither's guarantees
(`Security/Definitions/Pouw/Guarantees/DitherInstance.lean`); and several of Pearl-C's per-shape and per-model instances
(`Security/Definitions/Pouw/Guarantees/PearlC.lean`). No guarantee over the remote protocol as a whole is stated yet; `Properties.Composition` (`Security/Properties/Composition.lean`) proves the composition's accept
rule against its code.

The guarantees rest on named assumptions, each a hypothesis of the statement: the schemes' conjectures, stated and not
proved (`Pouw.NCP.Assumptions.TTNCP_U`, and Pearl-C's TT_OUT forms such as
`Pouw.PearlC.Assumptions.TTOutPearlCDevRev1W1`; Pearl-C4's `Pouw.PearlC.Assumptions.TTOutFp4Sm120W1`), the seeded noise's forms
(`Pouw.PearlC.RowSeedNoise`, `Pouw.PearlC.CodeSeedNoise`), the receipt clock (`Pouw.Assumptions.HonestClock`), the sampled
tile's proof (`Pouw.PearlC.Assumptions.TileProofSound`), and the claims `trusted-host`, `live-verifier`, `cr/sha-512`
and `prf/sha-512` (`verity.core.claims`).

## Core API

~~~python
from verity.core.protocols.pouw import audit, core_check, credit, schemes, verifier, window

scheme = schemes.get("pearl-c-sm120-v1-h3")                 # a configured PoUWScheme
root = audit.weights_root(weight_roots)                    # the weights' root a credit period's open signs
verdict = core_check.check(params, commitment, evidence)   # one execution phase, as `pouw-check` answers
outcome = credit.window(key, ledger, pins, main, ...)      # a served execution phase's credited set and its sampled proof blocks' outcome
period = credit.period_credit(key, ledger, pins, main, name, windows, verify, ...)   # a credit period, closed and complete
work = window.bucket(n_real)                               # the grid level: credited tiles and padded N
k = window.k_prime(eps, delta, work.credited, work.n, tile_work)   # K′, the sample that establishes ε at δ
~~~

## Where things are

- `protocol`, `schemes/`: the `PoUWScheme` interface and the named schemes.
- `audit`, `coins`, `salt`, `window`, `core_check`, `credit`, `identifier`, `registration`, `weights`, `operands`: the
  Python references the Lean is held to, and the reference rules not yet ported.
- `circuit/`: PoUW as `verity.ml.circuits` Programs: the tile and lane Definitions, the compile pass over a served Program
  (`compile`) and the sampling plan (`plan`). Each Definition has its `circuit-check` sidecar.
- `lean/check`, `lean/commit`, `lean/window`: the Lean verifier, each with its difftest; `lean/check/README.md` and
  `lean/window/README.md` map each Lean module to its Python.
- `PROTOCOL.md`: the experimental spec, until the Lean covers it.
- `dither/`: Dither, experimental, with its `PROTOCOL.md`.
- `tests/`: the package's suite. `benchmarks/pouw/` is the measuring side; nothing here imports it.
