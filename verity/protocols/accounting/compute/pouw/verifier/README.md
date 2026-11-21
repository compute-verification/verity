# PoUW

PoUW, proof of useful work, tells the auditor how much of a developer's GPU time went into the matrix products of the
model the developer registered. The developer serves as usual, but each registered matrix product runs under a scheme
that noises it with a salt the developer learns only once its weights are committed, and commits what a check will need.
After each execution phase the challenger flips coins that sample its tiles. On the served scheme, Pearl+, the developer
opens each sampled tile's A and B rows to the Lean check, which recomputes the tile's words from them; the
zero-knowledge path, which would prove the sampled tiles without opening them, is not served.

In the protocol's own words: "An accepted run establishes a credited size S of noised matrix products, at least
(1 − ε)·S of their AND gates computed right, except with probability δ." The statements that carry it, and their
assumptions, are under What is proved, below.

PoUW is one protocol with several schemes. A scheme decides how a matrix product's committed operands become checked
values; the commitments, the salt, the sample and the openings are the protocol's, the same for every scheme. The
verifier is Lean, under `lean/`: Pearl+'s verdict of record is `pouw-pearlplus`'s answer. The Python in this directory
is its reference, held to it by difftests. On the served road it still draws the run's sample and writes the coin record
(`coin.json`), and its own verify still runs, but its verdict decides nothing: `served_passes.py lean` writes the Lean
answer whatever the Python's says and keeps the Python's beside it as a measurement of where they differ.

## Try it

How many tiles an execution phase must sample depends only on its size and on the ε and δ it is to establish. For an
execution phase of 100,000 real tiles, at ε = 1% and δ = 10⁻³:

~~~sh
uv sync --all-packages --extra torch-cpu
uv run python -c "
from fractions import Fraction
from verity.protocols.accounting.compute.pouw.verifier import window
w = window.bucket(100_000)
print(w.credited, w.n, window.k_prime(Fraction(1, 100), Fraction(1, 1000), w.credited, w.n, 1))
"
~~~

It prints `98304 106496 745`: the execution phase is credited 98,304 tiles, pads to 106,496, and samples 745 of them.
It needs no GPU.

## Parties and schemes

| Party | Where it runs | What it does |
| --- | --- | --- |
| Prover | The developer's machines | Serves, commits, and opens the sampled tiles' A and B rows; proving them through Flock behind the ZK server is not served. Nothing it runs is trusted. |
| Challenger | The auditor's device | Commits each execution phase's challenges before it begins, and opens each one once. |
| Verifier | The auditor's own check | The Lean `pouw-pearlplus`: decides each Pearl+ run on its opened tiles. H1's `pouw-verifier` is not served. |

| Scheme | What it is |
| --- | --- |
| Pearl-C | Pearl's FP8 (E4M3) proof of useful work. Served as Pearl+ on the H100 and H200 (`pearl-c-h100-v1-h3`); the RTX PRO 6000's scheme (`pearl-c-sm120-v1-h3`) has kernels and a build, and no Lean verifier or `verity run`. |
| Pearl-C4 | Pearl-C on sm_120's NVFP4. Experimental: its kernels were deleted, and it has no Lean verifier or `verity run`. |
| NCP | Noise-cancelling polarization over int7 activations. A kernel exists (`ncp-v2`); no Lean verifier or `verity run`. |
| Dither | A design over the ACCEL model. Experimental, with its own `verity/protocols/experimental/dither/PROTOCOL.md`: kernels exist; no verifier or `verity run`. |

## What is proved

- **The top-level guarantees**, stated under `security/Definitions/` and proved as `SecurityProofs.G` under
  `security/Proofs/`. Each is listed in `security/lean-audit.json` and none is deployed. W is the credited reference work, t the work-weighted samples and ε_s the share of W a sample may
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
     e^{−D/4}. Its open premise is `Pouw.RelativeSecurity.CubeQuad`.

## PoUW in the remote protocol

The remote protocol, for an auditor with no devices at the datacenter, composes three subprotocols: SPOT's
one-stage audit, PoUW as Pearl+'s run verdict, `PouwVerifier.PearlPlus.Protocol.protocol` (`verity/pouw/pearl-plus/v1`,
`lean/pearlplus`), and the inference-only policy's check (`Core.Remote.V1.components` in
`verity/protocols/run/verifier/remote/CoreRemote.lean`, id `verity/remote/v1`). It accepts only when every subprotocol accepts
its own case, its commitment binds the subprotocols' commitments, the policy and the one-stage audit committed to one
record, and Pearl+'s run commitment names that record (`Core.Remote.V1.run`); any one's rejection rejects it
(`Core.Composition.tiedSameB`, `verity/protocols/run/verifier/lean/README.md`). H1's development executable
(`Core.Remote.protocol`, `verity/remote/v0`, which no configuration names) has PoUW's credit period check, `PouwVerifier.Period.protocol`
(`verity/pouw/period/v1`), as onsite does, in Pearl+'s place, every execution phase of the credit period naming the record as its
Commit record (`Core.Remote.executionPhases`). H1's serving side is deleted, so nothing serves an H1 execution phase
for that executable to check. The remote protocol's front door is
`verity/protocols/run/verifier/remote/`; PoUW is also a subprotocol of the onsite protocol (`verity/protocols/run/verifier/onsite`).

## What runs

The verifier is Lean, the source of truth for what PoUW checks: four Lake packages under `lean/`: the three below and `lean/pearlplus` (`pouw_pearlplus`).

- `lean/check` (`pouw_verifier`): `PouwVerifier.Period.protocol` (`verity/pouw/period/v1`), the `Core.Protocol` both the
  onsite and the remote protocol compose, and `PouwVerifier.protocol` (`verity/pouw/v1`), one execution phase's check,
  which neither composes. `PouwVerifier.check` decides one execution phase from its registration (the commitment) and the proof
  system's answer to it (the evidence); `PouwVerifier.Credit` decides what a served execution phase, and a credit period, credits. Its executable is `pouw-verifier` (`id | check | serve`, the core interface's commands).
- `lean/window` (`pouw_window`): what an execution phase registers and makes public: its grid level, its sample size K′, its
  fixed length, its committed entries and side trees, and its round coins (`PouwWindow.Grid`, `PouwWindow.Sample`,
  `PouwWindow.Length`, `PouwWindow.Public`, `PouwWindow.Entry`, `PouwWindow.Side`, `PouwWindow.Rounds`).
- `lean/commit` (`pouw_commit`): PoUW's commitments: tagged bindings, frames, rows, execution-phase coins and samples, all SHA-512.

The Python modules of the same names (`core_check.py`, `credit.py`, `window.py`, `coins.py`, `audit.py`) are their
references, held to them by the difftests (`lean/check/tests`, `lean/window/tests`, `lean/commit/tests`).
`verifier.py` is the entry the Python side calls: `verifier.lean(fn, value)` runs the Lean, and `verifier.coins(n)` is
the challenger's entropy, which the Lean reads from the operating system. `RECORD_VERDICT` (`"lean"`) switches what
`of_record` doesn't read (the statement runs and the committed circuit files), and setting
it to `"python"` is the one line that undoes those.

### Verdicts of record

No function here returns a verdict of its own. A Pearl+ run's verdict is `pouw-pearlplus`'s, which
`benchmarks/pouw/pearl_c_vllm/served_passes.py lean` runs (the table's last row). Each other verdict PoUW acts on comes
from one of two places. Some come from
its Lean check: `verifier.of_record` for a `pouw-verifier-difftest` function, or `verifier.serve_of_record` for a core
protocol's `serve`. The Lean's answer is the verdict, and no answer from the Lean is the refusal `no-answer`
(`verifier.NO_ANSWER`); the Python twin (`*_twin`) answers the difftest only. `credit.window` and `period_credit` still
run `_window` and `_period_of`, which drive the statement runs whose outcomes the Lean reads; their verdicts are not
read. The rest refuse until their Lean is reached: the public entry raises
`NotImplementedError("compute-accounting/pouw-verdict-from-main")` (`verifier.not_written`), and its twin answers the
difftest only.

| Verdict | Entry | Lean of record | Twin |
| --- | --- | --- | --- |
| An execution phase | `core_check.case`, `answer` | `pouw-verifier serve` | `case_twin`, `answer_twin`, `check_twin` |
| A credit period | `period_check.case`, `answer` | `pouw-period serve` | `case_twin`, `answer_twin`, `check_twin` |
| A served execution phase's credit | `credit.window` | `Credit.executionPhase` (`window`) | `credit._window` |
| A credit period's credit | `credit.period_credit` | `Credit.periodCredit` (`period`) | `credit._period_of` |
| The salt | `core_check.salt_of_record` | `saltCheck` (`salt`) | `salt_twin` |
| A live pass's salt | `core_check.live_salt` | `liveSaltCheck` (`live-salt`) | `live_salt_twin` |
| The round coins | `core_check.rounds` | `roundsCheck` (`rounds`) | `rounds_twin` |
| The sample | `credit.drawn` | `Credit.drawn` (`drawn`) | `drawn_twin` |
| A period's presented roots | `credit.period` | `Credit.period` (`period-roots`) | `period_twin` |
| A Pearl-C period's credit | `credit.pearl_c_period_credit` | `Credit.pearlCMonitoringPeriodCredit`, a stub that refuses everything | `pearl_c_period_credit_twin` |
| Passes credited together | `credit.passes_distinct` | Pearl+'s `PearlPlus.passesDistinct` (`passes-distinct`) | `passes_distinct_twin` |
| A Pearl+ run | `served_passes.py lean`, which writes `pearlplus-lean.json`, read by `served_verdicts.pearlplus_of_record` | `pouw-pearlplus` (`PouwPearlPlus.Main`: `check`; `liveDrawReason` under `--draw-live` and `liveSaltReason` under `--salt-live`, both passed by the served road; under `--of-record`, a live run missing either session is refused) | None: `served_passes.py verify` draws the coin, and its verdict is compared with the Lean's as a measurement (`differ`) |

Several functions are twins only, because the Lean decides their verdict inside a whole case's (`registered_twin`,
`opening_twin`, `pass_epoch_twin`, `draw_twin`). Others are guards a device runs before a coin opens: `window`'s
`*_epoch_reason` and `rounds_reason`, `coins.coin_reason`, `salt.unit_reason` and `core_check.recorded_reason`. They
only refuse to go on, never credit. The verdict of record checks the same rules again in Lean, except `salt.unit_reason`
and `core_check.recorded_reason`: they guard a certified execution phase, which only the prover's `served_commit.Challenger`
class runs, and no command runs one, so no verdict of record rechecks them yet.

### Refusals only Python makes

No Lean verdict makes these refusals. Each runs in the served road's challenger,
`verity/protocols/accounting/compute/pouw/prover/kernels/served_zk/served_commit.py`, and its own test stays until a Lean check makes it:

| Code | What it holds | Where |
| --- | --- | --- |
| `unknown` | A check, sample or outcome names an execution phase that was never registered. | `Challenger._held` |
| `not-drawn` | An execution phase is checked or delivered before its sample. | `Challenger._drawn` |
| `drawn` | An execution phase's sampling coin opens once, and a second sample is refused. | `Challenger.select` |
| `late-registration` | An execution phase registers before its epoch's draw coin is released. | `Challenger.register` |
| `certifier` | A certified execution phase is its own pass's, with no replay and no recorded salt, and the certifier reveals what it committed. | `Challenger._certified` |
| `unsalted` | A certified execution phase's sampled proof block reads its salt (`salt.unit_reason`). | `Challenger._certified` |
| `reregistered` | A certified execution phase enters the registration ledger once per stream and prover. | `Challenger._certified` |
| `level` | A registration's grid level has a population. | `register` |
| `law` | The window's draw follows a subset sampling design. | `draw_bytes` |
| `units` | The outcome's proof blocks are exactly the sampled ones, in order. | `Challenger.outcome` |
| `public` | The outcome's public items are exactly `window.PUBLIC`. | `Challenger.outcome` |
| `pouw-verifier` | An accepted stream outcome is accepted by the Lean answer too. | `Challenger.outcome` |
| `lean-run` | The challenger runs the Lean verifier and gets a documented status from it. | `LEAN_RUN` |
| `public-header` | The Lean verifier's verdict names the challenger's copy of the public file. | `PUBLIC_HEADER` |

`reject`, Python's answer form, leaves with the twins; `lean-python-disagree` went with `of_record.Record`'s Python side.
