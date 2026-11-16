# PoUW

PoUW, proof of useful work, tells the auditor how much of a developer's GPU time went into the matrix products of the
model the developer registered. The developer serves as usual, but each registered matrix product runs under a scheme
that noises it with a salt the developer learns only once its weights are committed, and commits what a check will need.
After each execution phase the challenger flips coins that sample its tiles, and the developer proves each sampled
tile in zero knowledge.

In the protocol's own words: "An accepted run establishes a credited size S of noised matrix products, at least
(1 − ε)·S of their AND gates computed right, except with probability δ." The statements that carry it, and their
assumptions, are under What is proved, below.

PoUW is one protocol with several schemes. A scheme decides how a matrix product's committed operands become checked
values; the commitments, the salt, the sample and the openings are the protocol's, the same for every scheme. The
verifier is Lean, under `lean/`. The Python in this directory is its reference, held to it by difftests, and returns no
verdict of its own.

## Try it

How many tiles an execution phase must sample depends only on its size and on the ε and δ it is to establish. For an
execution phase of 100,000 real tiles, at ε = 1% and δ = 10⁻³:

~~~sh
uv sync --all-packages --extra torch-cpu
uv run python -c "
from fractions import Fraction
from verity.protocols.resource_accounting.compute.pouw.verifier import window
w = window.bucket(100_000)
print(w.credited, w.n, window.k_prime(Fraction(1, 100), Fraction(1, 1000), w.credited, w.n, 1))
"
~~~

It prints `98304 106496 745`: the execution phase is credited 98,304 tiles, pads to 106,496, and samples 745 of them.
It needs no GPU.

## Parties and schemes

| Party | Where it runs | What it does |
| --- | --- | --- |
| Prover | The developer's machines | Serves, commits, and proves the sampled proof blocks through Flock behind the ZK server. Nothing it runs is trusted. |
| Challenger | The auditor's device | Commits each execution phase's challenges before it begins, and opens each one once. |
| Verifier | The auditor's own check | The Lean `pouw-check`: decides each execution phase on its proofs. |

| Scheme | What it is |
| --- | --- |
| Pearl-C | Pearl's FP8 (E4M3) proof of useful work, on the H100 and the RTX PRO 6000. |
| Pearl-C4 | Pearl-C on sm_120's NVFP4. Experimental. |
| NCP | Noise-cancelling polarization over int7 activations. |
| Dither | A design over the ACCEL model. Experimental, with its own `dither/PROTOCOL.md`. |

[`DESIGN.md`](DESIGN.md) has each party's messages, each scheme in full, one execution phase step by step, the Python
API, and every guarantee and assumption.

## What is proved

- **The top-level guarantees**, stated under `security/Definitions/` and proved as `SecurityProofs.G` under
  `security/Proofs/`. W is the credited reference work, t the work-weighted samples and ε_s the share of W a sample may
  miss. Items 1 and 2 hold for every in-domain workload (`Pouw.Workload.InDomain`: at most 2^64 proof blocks, a layout in the
  domain D, int7 weight entries) under the Ω_max bound (each proof block's reference work at most Ω_max times its matrix
  products'), with at most 2^64 oracle calls. That bound is where TT is conjectured, and the reduction does not use it;
  read past it, TT's error formula gives 2^−48 at 2^80 calls, which no statement claims:
  1. `Pouw.Guarantees.EndToEnd` (`Pouw/Guarantees/Game.lean`): the audit accepts while the work done is below the
     scheme's discounted share of (1 − ε_s)·W, and the run breaks no SHA-512 collision, with probability at most
     (1 − ε_s)^t plus TT's error. Assumes TT (`Pouw.Assumptions.TT`), the scheme's conjecture.
  2. `ResourceAccounting.Guarantees.ComputeOfPouwPerRun` (`Pouw/Guarantees/PerRun.lean`): the same certificate and error
     as the compute audit's per-run guarantee, on any device whose compute audit realizes PoUW's game. Assumes TT and
     `ResourceAccounting.BudgetAbort`.
  3. `Pouw.Guarantees.MonitoringPeriodBindingPerProver` (`Compose/Guarantees/PeriodBinding.lean`): in each verification phase
     of a credit period the openings fail to bind with probability at most |L|·√(q/2^256 + κ), L every position of the
     credit period. That is one phase's bound, and the credit period's term is n times it, n its phases. Assumes `Claims.ExpectedCR`;
     κ is not yet bounded, and no deployed |L| is stated.
  4. `Pouw.Guarantees.RelativeSecurity.TheoremSStaged` and `TheoremSOverlap` (`Pouw/Guarantees/RelativeSecurity.lean`):
     a play whose errors are of degree at most 2 is accepted with deficiency at least D with probability at most
     e^{−D/4}. No assumption.

The rest of what is proved, what is open and every named assumption: [`DESIGN.md`](DESIGN.md)'s "Security
assumptions and guarantees".

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
Commit record (`Core.Remote.windows`). In the vLLM served run, on that development executable
(`adapters/vllm/tools/served_run/`), PoUW runs on the served model as H1, under the challenger's development key, so its verdict
credits nothing, and an H1 commitment's served execution blocks aren't yet a `verity/pouw/period/v1` case. The remote
protocol's front door is
`verity/protocols/compose/verifier/remote/`; PoUW is also a subprotocol of the onsite protocol (`verity/protocols/compose/verifier/onsite`).

## What runs

The verifier is Lean, the source of truth for what PoUW checks: three Lake packages under `lean/`.

- `lean/check` (`pouw_check`): `PouwCheck.Period.protocol` (`verity/pouw/period/v1`), the `Core.Protocol` both the
  onsite and the remote protocol compose, and `PouwCheck.protocol` (`verity/pouw/v1`), one execution phase's check,
  which neither composes. `PouwCheck.check` decides one execution phase from its registration (the commitment) and the proof
  system's answer to it (the evidence); `PouwCheck.Credit` decides what a served execution phase, and a credit period
  ([`DESIGN.md`](DESIGN.md)), credits. Its executable is `pouw-check` (`id | check | serve`, the core interface's commands).
- `lean/window` (`pouw_window`): what an execution phase registers and makes public: its grid level, its sample size K′, its
  fixed length, its committed entries and side trees, and its round coins (`PouwWindow.Grid`, `PouwWindow.Sample`,
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
| A Pearl-C period's credit | `credit.pearl_c_period_credit` | `Credit.pearlCMonitoringPeriodCredit`, a stub that refuses everything | `pearl_c_period_credit_twin` |
| Passes credited together | `credit.passes_distinct` | Pearl+'s `PearlPlus.passesDistinct` (`passes-distinct`) | `passes_distinct_twin` |

Several functions are twins only, because the Lean decides their verdict inside a whole case's (`registered_twin`,
`opening_twin`, `pass_epoch_twin`, `draw_twin`). Others are guards a device runs before a coin opens: `window`'s
`*_epoch_reason` and `rounds_reason`, `coins.coin_reason`, `salt.unit_reason` and `core_check.recorded_reason`. They
only refuse to go on, never credit, and the verdict of record checks the same rules again in Lean.
