# PoUW (`verity.protocols.accounting.pouw.verifier`)

Proof of useful work: certifies that a served model's registered matrix products were computed, by SPOT over
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
  credit period by `PouwCheck.Credit`; its executable is `pouw-check`, and the Python modules of the same names are
  its twins, never a verdict of their own (Verdicts of record, below).
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
  3. `Pouw.Guarantees.MonitoringPeriodBindingPerProver` (`Accounting/Guarantees/PeriodBinding.lean`): in each verification phase
     of a credit period the openings fail to bind with probability at most |L|·√(q/2^256 + κ), L every position of the
     credit period. That is one phase's bound, and the credit period's term is n times it, n its phases. Assumes `Claims.ExpectedCR`;
     κ is not yet bounded, and no deployed |L| is stated.
  4. `Pouw.Guarantees.RelativeSecurity.TheoremSStaged` and `TheoremSOverlap` (`Pouw/Guarantees/RelativeSecurity.lean`):
     a play whose errors are of degree at most 2 is accepted with deficiency at least D with probability at most
     e^{−D/4}. No premise.

  The rest of what is proved, what is open and every named assumption: Security assumptions and guarantees, below.

## Parties

PoUW, the compute accounting protocol, has three parties (`parties.PROTOCOL`, on the deployment layer
`verity.runtime`). Each execution phase runs the same rounds between them:

- **The prover** is the developer's: its GPU serving and the PoUW prover (`verity/protocols/accounting/pouw/prover/`,
  Pearl-C and Pearl+ among its schemes), proving through Flock `--zk` behind the ZK server. It holds the weights and
  their committed rows, the values it commits under hiding leaves and their salts, and the noise each salt gives. It
  sends the verifier the root the salt follows (the weights' root a credit period's open signs), then the phase's
  registration once it has served on the salt, then the sampled proof blocks' proofs. It trusts nothing: it checks that
  the salt it is sent opens against the challenger's commitment, after the root it sent. Nothing it runs or hosts is
  trusted, and neither is the audited hardware or its vendor.
- **The challenger** is on the auditor's devices, a machine the developer doesn't control. It holds each phase's
  challenges, a salt challenge and a sampling challenge with their hm96 salts (`coins.Challenger`), sampled from its own
  randomness and committed before the phase begins; a served challenger also holds its LMS key and its signed log. It
  sends the verifier the commitment, and opens each challenge once, only after the verifier has recorded what it
  follows: the salt after the prover's root, the sampling challenge after the registration. It trusts nothing it
  receives. It is the one auditor-trusted device PoUW relies on (`trusted-device`); the auditor-party assumptions
  (`Accounting.Assumptions.Parties`) state no PoUW challenger yet.
- **The verifier** is the auditor's own check, the Lean `pouw-check` (`PouwCheck.check`, with the Python of the same
  names as its reference). It holds the phase's commitment, records the prover's root and registration, forwards each
  challenge to the prover once it opens against the commitment (`coins.coin_reason`), and decides the phase on the
  proofs: one output line a phase, which credits only that phase. It trusts its challenger, so an outcome convinces the
  auditor and no third party.

A certified execution phase's salt is the network certifier's instead (`salt.py`), a layer of its own that these parties
don't hold, and a live phase's round challenges aren't in them yet.

## PoUW in the remote protocol

The remote protocol, for an auditor with no devices at the datacenter, composes three subprotocols: SPOT's
one-stage audit, PoUW as Pearl+'s run verdict, `PouwCheck.PearlPlus.Protocol.protocol` (`verity/pouw/pearl-plus/v1`,
`lean/pearlplus`), and the inference-only policy's check (`Core.Remote.V1.components` in
`verity/protocols/compose/verifier/remote/CoreRemote.lean`, id `verity/remote/v1`). It accepts only when every subprotocol accepts
its own case, its commitment binds the subprotocols' commitments, the policy and the one-stage audit committed to one
record, and Pearl+'s run commitment names that record (`Core.Remote.V1.run`); any one's rejection rejects it
(`Core.Composition.tiedSameB`, `verity/protocols/compose/verifier/lean/README.md`). H1's development executable
(`Core.Remote.protocol`, `verity/remote/v0`, which no configuration names) has PoUW's credit period check, `PouwCheck.Period.protocol`
(`verity/pouw/period/v1`), as onsite does, in Pearl+'s place, every execution phase of the credit period naming the record as its
Commit record (`Core.Remote.windows`). In the remote demo, on that development executable, the vLLM demo
(`verity demo vllm`), PoUW runs on the served model as H1, under the challenger's development key, so its verdict
credits nothing, and an H1 commitment's served execution blocks aren't yet a `verity/pouw/period/v1` case. The remote
protocol's front door is
`verity/protocols/compose/verifier/remote/`; PoUW is also a subprotocol of the onsite protocol (`verity/protocols/compose/verifier/onsite`).

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

### Verdicts of record

No function here returns a verdict of its own. Each verdict PoUW acts on comes from one of two places. Some come from
its Lean check: `verifier.of_record` for a `pouw-check-difftest` function, or `verifier.serve_of_record` for a core
protocol's `serve`. The Python computes its answer beside the Lean as the twin (`*_twin`), and a disagreement, or no
answer from the Lean, is a refusal. The rest refuse until their Lean is reached: the public entry raises
`NotImplementedError("compute-accounting/pouw-verdict-from-main")` (`verifier.not_written`), and its twin answers the
difftest only.

| Verdict | Entry | Lean of record | Twin |
| --- | --- | --- | --- |
| An execution phase | `core_check.case`, `answer` | `pouw-check serve` | `case_twin`, `answer_twin`, `check_twin` |
| A credit period | `period_check.case`, `answer` | `pouw-period serve` | `case_twin`, `answer_twin`, `check_twin` |
| A served execution phase's credit | `credit.window` | `Credit.window` (`window`) | `credit._window` |
| A credit period's credit | `credit.period_credit` | `Credit.periodCredit` (`period`) | `credit._period_of` |
| The salt | `core_check.salt_of_record` | `saltCheck` (`salt`) | `salt_twin` |
| A live pass's salt | `core_check.live_salt` | `liveSaltCheck` (`live-salt`) | `live_salt_twin` |
| The round coins | `core_check.rounds` | `roundsCheck` (`rounds`) | `rounds_twin` |
| The sample | `credit.drawn` | `Credit.drawn` (`drawn`) | `drawn_twin` |
| A period's presented roots | `credit.period` | `Credit.periodRoots`, a stub that refuses everything | `period_twin` |
| A Pearl-C period's credit | `credit.pearl_c_period_credit` | `Credit.pearlCPeriodCredit`, a stub that refuses everything | `pearl_c_period_credit_twin` |
| Passes credited together | `credit.passes_distinct` | Pearl+'s `PearlPlus.passesDistinct` (`passes-distinct`) | `passes_distinct_twin` |

Several functions are twins only, because the Lean decides their verdict inside a whole case's (`registered_twin`,
`opening_twin`, `pass_epoch_twin`, `draw_twin`). Others are guards a device runs before a coin opens: `window`'s
`*_epoch_reason` and `rounds_reason`, `coins.coin_reason`, `salt.unit_reason` and `core_check.recorded_reason`. They
only refuse to go on, never credit, and the verdict of record checks the same rules again in Lean.

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
(`uv run tools/verity/check/suites.py verity/protocols/accounting/pouw/verifier/lean/check`, and likewise `lean/window` and
`lean/commit`); the composition's difftest runs `remote serve` on PoUW's cases beside SPOT's
(`verity/protocols/compose/verifier/lean/tests/test_composition_difftest.py`); `verity/protocols/accounting/pouw/prover/kernels/served_zk/of_record.py` runs
each check of a served execution phase through the Lean and its Python reference and records the Lean's verdict.

## Schemes

A scheme turns one registered matrix product's committed operand rows into each tile's checked values, its leaf and its
useful output; everything else (commitments, salt, sample, openings) is the protocol's and the same for every scheme.

- **Pearl-C** (`pearl-c-h100-v0`, `pearl-c-h100-v1`, `pearl-c-sm120-v1`, and their `-h3` forms,
  `verity/protocols/accounting/pouw/prover/schemes/pearl_c.py`): Pearl's FP8 (E4M3) proof of useful work, the GEMM-based scheme Pearl
  publishes (`pearl-fp8-v4`, `verity/protocols/accounting/pouw/prover/schemes/pearl_kw.py`), adopted as a PoUW scheme, with its noisy
  clean-up pinned bit for bit so that the useful output is a checked value. It runs on the H100 and on the RTX PRO 6000
  Blackwell (sm_120). The `-h3` forms hash with SHA-512 throughout, and h3 is the default wherever a hashing format or
  tree hash goes unnamed. The others (h0, taken by name) keep Pearl's keyed-BLAKE3 seeds, lines and tickets, with
  TurboSHAKE128 leaves and SHA-256 trees.
- **Pearl-C4** (`pearl-c-nvfp4-v0`, `verity/protocols/accounting/pouw/prover/schemes/pearl_c4.py`): Pearl-C with the FP8 product
  replaced by sm_120's FP4 with scale groups, NVFP4 only.
- **NCP** (`ncp-v1`, `verity/protocols/accounting/pouw/prover/schemes/ncp.py`): noise-cancelling polarization over int7 activations, on
  SHA-256 trees (its `commitment_hash`: its tile digests are 32 bytes).
- **Dither**, PoUW's no-fold design over the ACCEL model, is experimental, with its own spec, `dither/PROTOCOL.md`.

Each scheme module's docstring states its steps and roundings, with pinned vectors under `tests/vectors`.

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
  NCP's `Pouw.SecurityProofs.NCP.EpsFromTTNCP_U_v1`. Each follows from its gap form. Pearl-C's at 8192³ on the RTX PRO
  6000 (sm_120) at an inert-work share of 1/1,000 is a lemma beside its gap form, not a listed guarantee
  (`PearlC.EpsSm120v1LoopCast8p72Rev1Cap1000_8192`, ε = 1648180439/444424569561, 0.371%, in
  `security/Proofs/Pouw/Lemmas/PearlCRecords.lean`), and its figure is not the served point's. What is served is Llama 3.2 1B on an
  H200, credited under the H100 record at an inert-work share of 1/400, and its gap is a lemma, not yet a listed
  guarantee: `pearlCGapH100Rev1_llama32_1b` (`security/Proofs/Pouw/PearlC/ServedH100Rev1Gap.lean`), under
  `TTOutPearlCH100Rev1`, g = 69533/10736000 (0.648%) on the model's four linears in one layout, which is `down`'s
  (n = 2048, K = 8192); as an advantage, ε = g/(1 − g) = 69533/10666467 (0.652%). This is the assumption's gap alone:
  sampling's ε_s at δ_s multiplies in as (1 + ε)/(1 − ε_s), and the assumption's own δ adds.
- **The two S8 guarantees** (`security/Definitions/Pouw/Guarantees/PearlCS8.lean`) bound the credit of the tensor-clock
  record `devH100Tensor`, the H100 record with its SM pipes priced per second at the 1830 MHz tensor clock beside the
  1980 MHz SM clock: `Pouw.SecurityProofs.PearlC.SampledH100Rev1S8Llama32_1b` at g = 5040227/775520000 (0.650%,
  ε = 0.654%) and `…S8_8192` at 8192³ (0.538%). Their open premises are `w1-complete-h100` (W1 on the H100 with INT8
  at those prices, at zero SHA-512 and fetch prices, matches every adversary within its cost and query bound with
  good-tile credit at least as likely to exceed any threshold; the device reading's FP32 price is refuted on the H200,
  so this premise is falsified there) and `tt-out-tile-pearl-c-h100-rev1` (TT_OUT rev1 per
  audit tile over that W1 setting, falsification test `benchmarks/pouw/tests/test_tt_out_tile_h100_rev1.py`). They do
  not cover what is served: `pearl-c-h100-v1-h3` pays by the per-clock record `devH100`, about 1.5% more, and at the
  two-clock reading no statement of this form covers that credit, since the honest program with its first promotion
  skipped falls about 1% under TT_OUT's threshold for it. TT_OUT rev1 is refuted by proofs' red-team rating, so both
  guarantees are implications from refuted premises.
- **The audit game:** `Pouw.SecurityProofs.EndToEnd` and `Pouw.SecurityProofs.GapFromTT`, through two lemmas in
  `security/Proofs/Pouw/Lemmas/Game.lean`: Theorem 1 (if a scheme's gap g holds with failure probability δ, and the
  sample misses more than a share ε_s of the reference work with probability at most δ_s, then the verifier accepts
  while the work done is below (1 − g)(1 − ε_s)·W with probability at most δ_s + δ) and work-weighted sampling.
- **The deadline:** `Pouw.SecurityProofs.WorkBetweenSaltAndDeadline`, `Pouw.SecurityProofs.SpareCapacityAtDeadline` and
  `Pouw.SecurityProofs.PearlCSampledByDeadline`.
- **The schemes' gaps:** NCP's `Pouw.SecurityProofs.NCP.GapFromTTNCP_U_v1`; Pearl-C's over every device record, per
  unit, per audit tile and under Flock's hidden audit (`Pouw.SecurityProofs.PearlC.GapDevRev1K`, `SampledDevRev1K`,
  `HiddenFlockDevRev1K`), whose gap is `1 − (1 − 1/400)/ω` for a kernel within `ω` of the capped credit on the shapes
  served. Each statement at a record's numbers (`devSm120v1`, `devSm120v2`, `devH100`, `devFp4Sm120`) is a lemma, in
  `security/Proofs/Pouw/Lemmas/PearlCRecords.lean`. The row seed's reduction and the barrier are lemmas, in `security/Proofs/Pouw/Lemmas/PearlC.lean` and `security/Proofs/Pouw/Lemmas/Barrier.lean`.
- **The served credit's relative security:** `Pouw.SecurityProofs.RelativeSecurity.TheoremSStaged` and
  `Pouw.SecurityProofs.RelativeSecurity.TheoremSOverlap`.
- **Compute accounting reads PoUW** through `Accounting.SecurityProofs.ComputeOfPouw`,
  `Accounting.SecurityProofs.ComputeOfPouwPerRun`, `Accounting.SecurityProofs.PouwEndToEndPerRun` and
  `Accounting.SecurityProofs.PouwLocalOff`.

Stated, open: an execution phase's failure probability δ from its calls' (`Pouw.Guarantees.Window.WindowDelta`, and
`Pouw.Guarantees.Window.ServiceEscapes` for the deployed sample, `security/Definitions/Pouw/Guarantees/Window.lean`);
and Dither's guarantees (`security/Definitions/Pouw/Guarantees/DitherInstance.lean`). No guarantee over the remote protocol as a whole
is stated yet; `Properties.Composition` (`security/Properties/Composition.lean`) proves the composition's accept rule
against its code.

The guarantees rest on named assumptions, each a hypothesis of the statement: the schemes' conjectures, stated and not
proved (`Pouw.NCP.Assumptions.TTNCP_U`, and Pearl-C's TT_OUT forms such as
`Pouw.PearlC.Assumptions.TTOutPearlCDevRev1W1` and, at the H100, `Pouw.PearlC.Assumptions.TTOutTilePearlCH100Rev1S8`
beside the device model `Pouw.PearlC.Assumptions.W1CompleteH100`), the receipt clock (`Pouw.Assumptions.HonestClock`), and the claims
`trusted-device`, `live-verifier`, `cr/sha-512` and `prf/sha-512` (`verity.claims`). The lemmas at device records
(`security/Proofs/Pouw/Lemmas/PearlCRecords.lean`) rest on further forms that no listed statement takes: Pearl-C4's
TT_OUT at its sm_120 record (`TTOutFp4Sm120W1`), the seeded noise's forms (`RowSeedNoise`, `CodeSeedNoise`) and the
sampled tile's proof (`TileProofSoundAll`).

## Core API

~~~python
from verity.protocols.accounting.pouw.verifier import audit, core_check, credit, verifier, window
from verity.protocols.accounting.pouw.prover import schemes

scheme = schemes.get("pearl-c-sm120-v1-h3")                 # a configured PoUWScheme
root = audit.weights_root(weight_roots)                    # the weights' root a credit period's open signs
verdict = core_check.case(case)                            # one execution phase, `pouw-check serve`'s answer of record
outcome = credit.window(key, ledger, pins, main, ...)      # a served execution phase's credited set and its sampled proof blocks' outcome
period = credit.period_credit(key, ledger, pins, main, name, windows, verify, ...)   # a credit period, closed and complete
work = window.bucket(n_real)                               # the grid level: credited tiles and padded N
k = window.k_prime(eps, delta, work.credited, work.n, tile_work)   # K′, the sample that establishes ε at δ
~~~

## Where things are

- `protocol`: the `PoUWScheme` interface.
- `parties`: PoUW's prover, challenger and verifier on the deployment layer (`MANIFEST`, `PROTOCOL`), each machine's work
  a capability it is given.
- `audit`, `coins`, `salt`, `window`, `core_check`, `credit`, `identifier`, `registration`, `weights`, `operands`,
  `live_reads`: the Python references the Lean is held to, and the reference rules not yet ported.
- `verity/protocols/accounting/pouw/prover/`, the prover's, which core doesn't import: the named schemes (`schemes/`) and `circuit/`,
  PoUW as `verity.circuits` Programs: the tile and lane Definitions, the compile pass over a served Program
  (`compile`) and the sampling plan (`plan`). Each Definition has its `circuit-check` sidecar.
- `lean/check`, `lean/commit`, `lean/window`: the Lean verifier, each with its difftest; `lean/check/README.md` and
  `lean/window/README.md` map each Lean module to its Python.
- `dither/`: Dither, experimental, with its `PROTOCOL.md`.
- `tests/`: the package's suite, with the schemes' and the execution phase's pinned vectors under `tests/vectors`.
  `benchmarks/pouw/` is the measuring side; nothing here imports it.
