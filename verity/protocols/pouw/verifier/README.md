# PoUW (`verity.protocols.pouw.verifier`)

Proof of useful work: certifies that a served model's registered matrix products were computed, by sampled proofs over
committed, noised checked values. An accepted run establishes a credited size S of noised matrix products, at least
(1 − ε)·S of their AND gates computed right, except with probability δ. One protocol carries several named schemes
behind one interface (`protocol.PoUWScheme`, `schemes.get`). The proof system registers each execution phase, samples
its proof blocks with the challenger's coins, proves them in zero knowledge (Flock `--zk`) and verifies them; PoUW
states what each execution phase commits, the Program each sampled proof block must pass, and what the outcome credits.
It carries no policy parameter: it surfaces the ε at δ its sample established. A scheme must leave the useful work at
least half of the honest developer's GPU effort (AGENTS.md, "PoUW is mostly useful work"): the trivial PoUW, which hashes
every product, is a reference, not an answer.

## Start here

- **The verifier** is Lean (What runs, below): `PouwCheck.Period.protocol` (`verity/pouw/period/v1`), the subprotocol
  the onsite and the remote protocol compose. It decides each execution phase by `PouwCheck.check` and credits a closed
  credit period by `PouwCheck.Credit`; its executable is `pouw-check`, and the Python modules of the same names mirror it.
- **The top-level guarantees**, stated under `security/Definitions/` and proved as `SecurityProofs.G` under
  `security/Proofs/`. W is the credited reference work, t the work-weighted samples and ε_s the share of W a sample may
  miss. Items 1 and 2 hold for every in-domain workload (`Pouw.Workload.InDomain`: at most 2^64 units, a layout in the
  domain D, int7 weight entries) under the Ω_max bound (each unit's reference work at most Ω_max times its matrix
  products'), with at most 2^64 oracle calls. That bound is where TT is conjectured, and the reduction does not use it;
  read past it, TT's error formula gives 2^−48 at 2^80 calls, which no statement claims:
  1. `Pouw.Guarantees.EndToEnd` (`Pouw/Guarantees/Game.lean`): the audit accepts while the work done is below the
     scheme's discounted share of (1 − ε_s)·W, and the run breaks no SHA-512 collision, with probability at most
     (1 − ε_s)^t plus TT's error. Assumes TT (`Pouw.Assumptions.TT`), the scheme's conjecture.
  2. `Accounting.Guarantees.ComputeOfPouwPerRun` (`Accounting/Guarantees/PerRun.lean`): the same certificate and error
     as the compute audit's per-run guarantee, on any device whose compute audit realizes PoUW's game. Assumes TT and
     `Accounting.BudgetAbort`.
  3. `Pouw.Guarantees.PeriodBindingPerProver` (`Accounting/Guarantees/PeriodBinding.lean`): in each verification phase
     of a credit period the openings fail to bind with probability at most |L|·√(q/2^256 + κ), L every position of the
     period. That is one phase's bound, and the period's term is n times it, n its phases. Assumes `Claims.ExpectedCR`;
     κ is not yet bounded, and no deployed |L| is stated.
  4. `Pouw.Guarantees.RelativeSecurity.TheoremSStaged` and `TheoremSOverlap` (`Pouw/Guarantees/RelativeSecurity.lean`):
     a play whose errors are of degree at most 2 is accepted with deficiency at least D with probability at most
     e^{−D/4}. No premise.

  The rest of what is proved, what is open and every named assumption: Security assumptions and guarantees, below.

## PoUW in the remote protocol

The remote protocol, for an auditor with no devices at the datacenter, composes three subprotocols: sampled proofs'
one-stage audit, PoUW as Pearl+'s run verdict, `PouwCheck.PearlPlus.Protocol.protocol` (`verity/pouw/pearl-plus/v1`,
`lean/pearlplus`), and the inference-only policy's check (`Core.Remote.V1.components` in
`verity/protocols/remote/verifier/CoreRemote.lean`, id `verity/remote/v1`). It accepts only when every subprotocol accepts
its own case, its commitment binds the subprotocols' commitments, the policy and the one-stage audit committed to one
record, and Pearl+'s run commitment names that record (`Core.Remote.V1.run`); any one's rejection rejects it
(`Core.Composition.tiedSameB`, `verity/protocols/compose/README.md`). H1's development executable
(`Core.Remote.protocol`, `verity/remote/v0`, which no configuration names) has PoUW's credit period check, `PouwCheck.Period.protocol`
(`verity/pouw/period/v1`), as onsite does, in Pearl+'s place, every execution phase of the credit period naming the record as its
Commit record (`Core.Remote.windows`). In the remote demo, on that development executable, the vLLM demo
(`verity demo vllm`), PoUW runs on the served model as H1, under the challenger's development key, so its verdict
credits nothing, and an H1 commitment's served execution blocks aren't yet a `verity/pouw/period/v1` case. The remote
protocol's front door is
`verity/protocols/remote/verifier/`; PoUW is also a subprotocol of the onsite protocol (`verity/protocols/onsite/verifier`).

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

An **execution phase** is the served steps one commitment covers (PoUW's older name, "commit period"). The code and the
Lean call it an execution phase (`window.py`, `PouwWindow`, `credit.window`), and keep the name until a format version. In one
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
   (the code's credit period, `credit.period_credit`) is the execution phases between an open, which signs the weights' root and
   commits the circuit files, and a close; it counts only closed and complete.

How it is run: the difftests run the Lean against the Python on generated cases
(`uv run tools/verity/check/suites.py verity/protocols/pouw/verifier/lean/check`, and likewise `lean/window` and
`lean/commit`); the composition's difftest runs `remote serve` on PoUW's cases beside sampled proofs'
(`verity/protocols/compose/tests/test_composition_difftest.py`); `benchmarks/pouw/served_zk/of_record.py` runs
each check of a served execution phase through the Lean and its Python reference and records the Lean's verdict.

## Schemes

A scheme turns one registered matrix product's committed operand rows into each tile's checked values, its leaf and its
useful output; everything else (commitments, salt, sample, openings) is the protocol's and the same for every scheme.

- **Pearl-C** (`pearl-c-h100-v0`, `pearl-c-h100-v1`, `pearl-c-sm120-v1`, and their `-h3` forms,
  `verity/protocols/pouw/prover/schemes/pearl_c.py`): Pearl's FP8 (E4M3) proof of useful work, the GEMM-based scheme Pearl
  publishes (`pearl-fp8-v4`, `verity/protocols/pouw/prover/schemes/pearl_kw.py`), adopted as a PoUW scheme, with its noisy
  clean-up pinned bit for bit so that the useful output is a checked value. It runs on the H100 and on the RTX PRO 6000
  Blackwell (sm_120). The `-h3` forms hash with SHA-512 throughout, and h3 is the default wherever a hashing format or
  tree hash goes unnamed. The others (h0, taken by name) keep Pearl's keyed-BLAKE3 seeds, lines and tickets, with
  TurboSHAKE128 leaves and SHA-256 trees.
- **Pearl-C4** (`pearl-c-nvfp4-v0`, `verity/protocols/pouw/prover/schemes/pearl_c4.py`): Pearl-C with the FP8 product
  replaced by sm_120's FP4 with scale groups, NVFP4 only.
- **NCP** (`ncp-v1`, `verity/protocols/pouw/prover/schemes/ncp.py`): noise-cancelling polarization over int7 activations, on
  SHA-256 trees (its `commitment_hash`: its tile digests are 32 bytes).
- **Dither**, a no-fold design over the ACCEL model, is experimental, with its own spec, `dither/PROTOCOL.md`. It rests
  on SAK and is no longer the bar or the plan (below).

Each scheme module's docstring states its steps and roundings, with pinned vectors under `tests/vectors`.

## The bar and the next scheme

PoUW's bar holds on the H200, under an assumption milder than SAK. It supersedes Dither and SAK
(`SAKMatmul`, `SAKHigh`) as the bar.

- **T1**: the gap between honest and adversary is at most 1% of stock, additive, in priced work. The honest program is
  the best possible, hashing is credited, and every matmul takes fresh noise on every call.
- **T2**: honest overhead is at most 100×.
- **T3**: perplexity rises at most 5%.

The next scheme is **the lead**, protocol id `verity/pouw/v2`: NCP-family served
sparse noise with row tags and three certifier-streamed coin lanes, every k32 word committed. Its defaults:

- ShaWrites (rank 4a) counts as milder than SAK, with "honest is best" for hashing's credit;
- the copy model is tile grain;
- hiding is the keyed leaf, d = SHA-512(ks_w ‖ salt_w ‖ M_top);
- the certifier is the coin source, on a trusted host;
- the network term and whole-step coverage are reported beside T1;
- the fold is exact on |A·Bᵀ| ≤ 2^22;
- stock's decode price is reported both ways.

v1 (H1, `verity/pouw/v1`) stays until v2 credits a served window.

## Security assumptions and guarantees

PoUW's security property is stated at the adversary's advantage ε, with failure probability δ: what an adversary
spends is at least the honest work of its correct units over 1 + ε, except with probability δ (the game's and
Pearl-C's, in `security/Definitions/Pouw/Advantage.lean`). The proofs carry it as the gap g = ε/(1 + ε), the same
property: an adversary of cost T wins when its correct units' honest work exceeds T/(1 − g)
(`security/Definitions/Pouw/Game/Defs.lean`), and a lemma in `security/Proofs/Pouw/Advantage.lean` proves the two forms
equivalent. The honest overhead γ is a cost, never a share of the bound: the honest prover's busy SM-time is at most
(1 + γ) times the fastest stock time for the same MACs, capped at γ ≤ 10 at each operating point (`GLOSSARY.md`).

Guarantees are stated in `security/Definitions/Pouw` and proved in `security/Proofs/Pouw`, as
`Pouw.SecurityProofs.G : Pouw.Guarantees.G`; `security/lean-audit.json` lists each proved one with what its statement
reads. Proved:

- **In ε form:** the audit end to end, `Pouw.SecurityProofs.EndToEndEps` (under TT(g₀) and the bound Ω_max, the
  protocol's advantage is ε = Ω_max/(1 − g₀) − 1, and an audit of t work-weighted samples accepts while
  (1 + ε)/(1 − ε_s)·T < W with probability at most (1 − ε_s)^t + δ, for every share ε_s < 1 of the reference work W);
  NCP's `Pouw.SecurityProofs.NCP.EpsFromTTNCP_U_v1`; Pearl-C's at 8192³ on the RTX PRO 6000 (sm_120) at an inert-work
  share of 1/1,000, `Pouw.SecurityProofs.PearlC.EpsSm120v1LoopCast8p72Rev1Cap1000_8192` (ε = 1648180439/444424569561, 0.371%);
  and Pearl-C4's at 8192³, `Pouw.SecurityProofs.PearlC.EpsFp4Sm120_8192` (ε = 9875013901/1366788186099, 0.723%). Each
  follows from its gap form below. Neither Pearl-C figure is the served point's. What is served is Llama 3.2 1B on an
  H200, credited under the H100 record at an inert-work share of 1/400, and its gap is a lemma, not yet a listed
  guarantee: `pearlCGapH100Rev1_llama32_1b` (`security/Proofs/Pouw/PearlC/ServedH100Rev1Gap.lean`), under
  `TTOutPearlCH100Rev1`, g = 69533/10736000 (0.648%) on the model's four linears in one layout, which is `down`'s
  (n = 2048, K = 8192); as an advantage, ε = g/(1 − g) = 69533/10666467 (0.652%). This is the assumption's gap alone:
  sampling's ε_s at δ_s multiplies in as (1 + ε)/(1 − ε_s), and the assumption's own δ adds.
- **The audit game:** `Pouw.SecurityProofs.EndToEnd` and `Pouw.SecurityProofs.GapFromTT`, through two lemmas in
  `security/Proofs/Pouw/Lemmas/Game.lean`: Theorem 1 (if a scheme's gap g holds with failure probability δ, and the
  sample misses more than a share ε_s of the reference work with probability at most δ_s, then the verifier accepts
  while the work done is below (1 − g)(1 − ε_s)·W with probability at most δ_s + δ) and work-weighted sampling.
- **The deadline:** `Pouw.SecurityProofs.WorkBetweenSaltAndDeadline`, `Pouw.SecurityProofs.SpareCapacityAtDeadline` and
  `Pouw.SecurityProofs.PearlCSampledByDeadline`.
- **The schemes' gaps:** NCP's `Pouw.SecurityProofs.NCP.GapFromTTNCP_U_v1`; Pearl-C's at 8192³ on the RTX PRO 6000
  (the served point's is the lemma above),
  `Pouw.SecurityProofs.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_8192` and its sampled twin
  `Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_8192`. The row seed's reduction and the barrier
  are lemmas, in `security/Proofs/Pouw/Lemmas/PearlC.lean` and `security/Proofs/Pouw/Lemmas/Barrier.lean`.
- **The served credit's relative security:** `Pouw.SecurityProofs.RelativeSecurity.TheoremSStaged` and
  `Pouw.SecurityProofs.RelativeSecurity.TheoremSOverlap`.
- **Compute accounting reads PoUW** through `Accounting.SecurityProofs.ComputeOfPouw`,
  `Accounting.SecurityProofs.ComputeOfPouwPerRun`, `Accounting.SecurityProofs.PouwEndToEndPerRun` and
  `Accounting.SecurityProofs.PouwLocalOff`.

Stated, open: an execution phase's failure probability δ from its calls' (`Pouw.Guarantees.Window.WindowDelta`, and
`Pouw.Guarantees.Window.ServiceEscapes` for the deployed sample, `security/Definitions/Pouw/Guarantees/Window.lean`);
Dither's guarantees (`security/Definitions/Pouw/Guarantees/DitherInstance.lean`); and several of Pearl-C's per-shape and
per-model instances (`security/Definitions/Pouw/Guarantees/PearlC.lean`). No guarantee over the remote protocol as a whole
is stated yet; `Properties.Composition` (`security/Properties/Composition.lean`) proves the composition's accept rule
against its code.

The guarantees rest on named assumptions, each a hypothesis of the statement: the schemes' conjectures, stated and not
proved (`Pouw.NCP.Assumptions.TTNCP_U`, and Pearl-C's TT_OUT forms such as
`Pouw.PearlC.Assumptions.TTOutPearlCDevRev1W1`; Pearl-C4's `Pouw.PearlC.Assumptions.TTOutFp4Sm120W1`), the seeded noise's forms
(`Pouw.PearlC.RowSeedNoise`, `Pouw.PearlC.CodeSeedNoise`), the receipt clock (`Pouw.Assumptions.HonestClock`), the sampled
tile's proof (`Pouw.PearlC.Assumptions.TileProofSoundAll`), and the claims `trusted-host`, `live-verifier`, `cr/sha-512`
and `prf/sha-512` (`verity.claims`).

## Core API

~~~python
from verity.protocols.pouw.verifier import audit, core_check, credit, verifier, window
from verity.protocols.pouw.prover import schemes

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
- `verity/protocols/pouw/prover/`, the prover's, which core doesn't import: the named schemes (`schemes/`) and `circuit/`,
  PoUW as `verity.circuits` Programs: the tile and lane Definitions, the compile pass over a served Program
  (`compile`) and the sampling plan (`plan`). Each Definition has its `circuit-check` sidecar.
- `lean/check`, `lean/commit`, `lean/window`: the Lean verifier, each with its difftest; `lean/check/README.md` and
  `lean/window/README.md` map each Lean module to its Python.
- `dither/`: Dither, experimental, with its `PROTOCOL.md`.
- `tests/`: the package's suite, with the schemes' and the execution phase's pinned vectors under `tests/vectors`.
  `benchmarks/pouw/` is the measuring side; nothing here imports it.
