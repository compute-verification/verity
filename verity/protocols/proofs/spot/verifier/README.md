# SPOT (`verity-sampled-proofs`)

SPOT spot-checks a committed computation. While serving, the developer commits every value that crosses a
proof-block boundary of a partition of the program and registers the commitment roots. The verifier then samples proof blocks
on the auditor's challenger's coins, the developer proves the sampled proof blocks, and the verifier checks the proofs. It never runs any part
of the computation. An accepted audit yields an integrity profile (`verity.protocols.proofs.spot.verifier.profile.IntegrityProfile`): except with
probability δ, at most so many proof blocks are incorrect, a proof block being incorrect when its committed outputs differ from its gates
applied to its committed inputs. The protocol is stated in Lean (see "Where things are"); this README outlines it,
points to the code, and lists what the Lean does not state yet.

~~~sh
uv run tools/check/suites.py verity/protocols/proofs/spot/verifier        # the Python and its checks
uv run tools/check/suites.py verity/protocols/proofs/spot/verifier/lean   # the Lean verifier's difftests
~~~

SPOT (`verity/sampled-proofs/v1`) is a protocol of both configurations: remote (`verity/remote/v1`,
`verity/protocols/compose/verifier/remote/`) composes it with Pearl+'s run (`verity/pouw/pearl-plus/v1`; H1's development executable
`verity/remote/v0`, with PoUW's period `verity/pouw/period/v1`) and the inference-only policy, and onsite
(`verity/protocols/compose/verifier/onsite/`) with the other protocols. Each configuration's verifier runs
`VStar.Recursive.Sampled`'s Lean check: one-stage v2 (`verity/sampled-proofs/one-stage/v2`, the audit on the proofs).
One-stage v1, whose verdicts are the evidence's, is refused by name (`V1_REFUSED`: a one-stage form whose params or
evidence name no `proofs`), and so is the recursive audit (`verity/sampled-proofs/recursive/v1`, `RECURSIVE_REFUSED` at
its params), which SPOT's soundness doesn't cover. The composite accepts only when every subprotocol accepts.

## SPOT[Flock] on its own

SPOT[Flock] on its own has three roles (`parties.SPOT_PROTOCOL`):

- the **inference server** serves (with vLLM) and commits its values. It sends the verifier its docket and
  commitments, and the proof server the replay inputs, one way;
- the **proof server** replays the sampled requests and proves the sampled proof blocks with `flock-circuit prove --zk`.
  It sends the verifier the proofs and gets back the coins and the sample, nothing else;
- the **verifier**: the challenger's live coins and signed log, and the Lean verifier.

The replay inputs are every value the sampled proof blocks read, with the salts that open their commitments. Each value
is a proof block's output, a certified input (the network certifier's stamp among them), a registered value (weights,
configs, Definitions) or advice charged at its width. A value the served hardware decides that no block computes (an
atomics order, a split choice) is never among them as a free witness: serving makes it canonical so a block computes it,
or it is advice.

Under a two-stage sampling design the same three roles run `parties.SPOT_TWO_PROTOCOL`. The first stage samples replay blocks.
The proof server replays them and sends the verifier their interior docket (`Interior`). Once it is in, the
verifier samples the second stage on fresh coins (`Second`). The proof server then proves the second stage's proof
blocks, one session a template, and the verifier audits both stages.

It has no ZK server and no recursion: the proof server talks only to the verifier. The core protocols (PoUW, PoUS and
network certification, in the remote and onsite configurations) wrap SPOT[Flock] in the recursion layer and the ZK server
(`parties.PROTOCOL`). The network certifier is a layer of its own: it stamps each request's release order and bucket at
ingress, and SPOT[Flock] reads that stamp as a certified input. Neither imports the other (`tests/test_boundaries.py`),
and they compose only above both.

Its guarantee is soundness, and it rests on:

- SHA-512's collision resistance (`TableCRZAt`, `LinkCRZC`);
- uniform verifier randomness (`UniformRandomBytes`);
- an honest verifier: the challenger flips its coins after the docket is recorded and logs them, the verifier
  reads that log and the record the challenger wrote (`ChallengerLogsCoins`, `FlipsAfterRecorded`, `VerifierReadsLog`,
  `RecordCustodyZKJ`), and it runs the Lean verifier;
- where it reads the certifier's stamp, the certifier's own assumptions, which that layer states (`trusted-device` among
  them, `verity/protocols/accounting/certifier/verifier/PROTOCOL.md`).

## Guarantee and assumptions

The guarantee is knowledge soundness of the sampled audit: except with a bounded probability, when the verifier
accepts, at most the profile's count of proof blocks are incorrect. Its top statement is Flock's
`Flock.Guarantees.SampledKSLiveV3`, stated in `security/Proofs/Flock/Bridge/ScopeGate.lean` and proved in
`security/Proofs/Flock/Bridge/KSLive.lean`, and it bounds the probability
that `verity-verify-proof --zk` accepts the files and the run passes the docket's population while the claim fails. It
covers sessions of non-templated statements, as `Flock.Desc.parse` reads a flat file. The V5 guarantees
(`Bridge.SampledKSLiveV5` and those behind it) read the file through `Flock.Desc.parseStmt`, so they also cover templated
statements (`Flock.Tags.templated`, whose acceptance `flock_verify_sound` covers), at the circuit `HmOut.unrolled`
builds. The top statement rests on these named assumptions, each stated by its docstring: `UniformRandomBytes`,
`SHA512CRStrict` (`cr/sha-512`) and `SHA512CRExpected` (`ecr/sha-512`) in `security/Definitions/Assumptions/Flock.lean`,
the last two taken at the collision finders the proof builds from the prover (`TableCRZAt` and `LinkCRZCAt`, which
unfold to them), and `RecordCustodyZKJPLaw` (read as `RecordCustodyZKJ`) in
`security/Proofs/Flock/Soundness/Assumptions/ZkP.lean`.

In the one-stage form, v2's verdicts are its own run of `verity-verify-proof verify --zk` on the sessions' files, which come
in as blobs (under "Where things are"), so a composite accepts there only where that run accepts. No Lean statement yet
carries a composite's acceptance to `SampledKSLiveV3`'s event.

## Outline

**The one-stage audit** is the audit protocol in use. One round goes like this:

1. Serving commits the values that the partition's proof blocks read and write, and sends the roots.
2. The developer registers a record that fixes everything a sampled proof block's statement can depend on: the program, the query
   that partitions it, the sampling design, the commitment scheme and the roots. The verifier checks the record against
   its own copies and logs a receipt before it samples.
3. The verifier samples proof blocks on the challenger's coins under the agreed sampling design and sends the sample in the
   clear, before the session's coin commitment. Nothing is derived from a commitment, and there is no beacon.
4. The sampled proof blocks are proved in one session, whose statement is the sampled proof blocks'.
5. The Lean verifier checks the proofs, and its verdict is the one that counts; the Python beside it is its mirror.
6. The audit record keeps the docket, the receipt, the sample, the proved proof blocks, the statement's digests and every
   verdict, and, when everything accepted, the integrity profile.

**The proof system** (`service`) runs these steps for a protocol, on the challenger's side. A protocol states a `Spec`:
- what it commits;
- its selection rule; a `Law` or a two-stage first stage is a rate, never a fixed count (`rate:`, `stratified-rate:`,
  `work-rate:`), each proof block sampled at least at it however the prover pads `N` (`Spot.Rate`);
- the check, as a public Program;
- its public items, each with its reason.

Everything the auditor observes must be a function of those items. A `Stream` per `Spec` holds each docket to that
rule and then to one-stage's checks, samples once, binds the delivered proof to its outcome, and counts failures against a
public budget.

**The sampling designs.** `subset:K` is a uniform `K`-subset of the `N` proof blocks, and `bernoulli:NUM/DEN` takes each
independently with probability `NUM/DEN`.
`stratified:K` has one stratum per subcircuit, and inside each an independent uniform sample without replacement of
`k_s = max(1, ⌊(2 K n_s + N) / (2N)⌋)` proof blocks, so every subcircuit is exercised in every audit. `work:K` keeps those
strata and that sampler but sizes each stratum by the work its proof blocks are credited: with `W = Σ_s w_s n_s`, stratum `s` samples
`k_s = min(n_s, max(f_s, ⌈K w_s n_s / W⌉))`, where `w_s` and the floor `f_s ≥ 1` come from the verifier's own work table.
Each verifier derives the strata from its own program and query, never from the prover. Both verifiers refuse what the
soundness statement doesn't cover: `bernoulli` at any `p` (`bernoulli`) and a closure map (`closure`). The audit on the
proofs refuses a `stratified` or `work` design in its one-statement form too, by the design's kind
(`OneStageZk.SUBSET_ONLY`), written as a sampling design or as a rate: that form's soundness statement covers `subset:K` alone, which
a rate `rate:NUM/DEN` names over the population. Its per-member form parses every design v1 does.

**What an accepted audit establishes.** Under a stratified sample, a set of incorrect proof blocks has acceptance probability
`Π_s C(n_s − m_s, k_s) / C(n_s, k_s)`, `m_s` being its proof blocks in
stratum `s`, so a subcircuit that is incorrect in every proof block is never missed. Under the work sampling design, incorrect proof blocks
doing a fraction `ε` of the credited work have miss probability at most `(1 − ε)^K`; at `K = 27,713`, a 0.1% share
has at most `2^-40`. The proof system adds, per proof block, its knowledge-soundness and binding errors (ε_ks and ε_bind).

**Two-stage designs** (`TwoStage`). Those above are one-stage. A two-stage design commits only replay-block boundaries
while serving and, once a sampled replay block's interior is committed (`Stream.register_interiors`; `GLOSSARY.md`,
"Two-stage design"), samples `k_t` of its `n_t` instances of each template `t` of the check, one cell (the claims'
`cell`; the rule's `k`, one count or one per template, capped at the cell's size; counts proving every cell whole are
refused). The check names the replay block's templates in the order of their first instance in it and the interior is
template-major, so each cell is a range of the interior's proof blocks and the second stage is one stratum per cell
(`second_law`). A cell of `n` proof blocks and footprint `w` is missed with probability at most
`C(n − w, k) / C(n, k)`, a replay block's cells independently; the profile at the footprint reads a replay block as
`Subset(min k_t)` over `T·n_max` proof blocks, which bounds that product. The verifier samples both stages
(`Select.select`, `Select.selectSecond`), and the two-stage certificate's knowledge soundness at Flock's masked record
holds of that code under the premises `SamplingDeploy.ksTwo_code` states.
A bound of S served bits charges a replay block at most F S, F the grain's fan-out (`partition.fanout`; the capped form at N = F),
1 when every replay block is closed. S is at least the bits one replay block returns, which the verifier counts from its
Definition's return (`second_stage.returned_bits`, Lean `SecondStage.returnedBits`: each returned output field of each
unit on 16-bit words of its own) and a two-stage docket carries (`ExecutionPhaseCommitmentRecord.returned`);
`plancheck.two` refuses a cap below them.

Not built yet: the security function and its review, and composition checks such as PoUW's tiles
per proof block matching this partition (a tile family built with `each` over the proof blocks nests in them by construction).

Serving's record is not yet the scheme the audit proves: the benchmark's first run (`benchmarks/one_stage/a0.py`)
registers a stand-in, a fresh commitment to values captured from serving, and says so (`window.kind = "stand-in"`). Its
served run (`benchmarks/one_stage/a5.py`) registers a stand-in too: a deferred Commit's serving rows, written after serving from its sealed replay bundle opened against the run's record
(`verity-vllm serving-rows`), whose `of` names the record's vllm-v1 run root and the bundle's manifest digest; the link from
those roots to the vllm-v1 roots is the committer's opening, not proof. Rows the in-Commit hook writes before any sample are
`served`: serving's own commitment, with no claim that they hold the values under the Commit's roots. A deferred run's serving becomes
`served` once serving's record is frame-v3-sha512 / hm96-sha512 itself, or once the circuit opens the vllm-v1 SHA-256 leaves.

## Where things are

| Part | Where |
| --- | --- |
| The guarantees' Lean | `security/Proofs/Flock/Soundness/Audit/`, whose `README.md` lists every theorem |
| The executable sample | `verity/commitments/lean/Commitments/Draw.lean` (`verity-verify-proof draw`) |
| The Lean verifier | `lean/` here (`spot-verifier`); its Python twin is `one_stage.verifier`, held to it by `lean/tests/test_difftest.py` |
| The audit on the proofs | `lean/Spot/OneStageZk.lean` (`spot-one-stage-zk`); Python `one_stage.proofs` |
| A monitoring period | `lean/Spot/MonitoringPeriod.lean`, `lean/Spot/MonitoringPeriodAudit.lean` (`spot-monitoring-period`) |
| The challenger | `../device/challenger.py`, Lean of record `lean/Spot/Challenger.lean`; remote's coin path `security/Definitions/Core/Challenger.lean` |
| The stream and the fleet | `service.py`, Lean `lean/Spot/Stream.lean` and `lean/Spot/Fleet.lean`, stated in `security/Properties/Spot/Halt.lean` |
| The sampling planner | `../prover/` (`optimize`, `candidates`, `costfit`), used by `benchmarks/sampling_plan` |
| The parties | `parties.py`, run by `verity.runtime.runner.run` |
| The benchmark | `benchmarks/one_stage`, the one-stage audit end to end |

Each part in full is under "The parts in detail", below.

## Core API

~~~python
from verity.protocols.proofs.spot.verifier import service
from verity.protocols.proofs.spot.verifier.one_stage import audit, draw, partition, docket as registration
from verity.protocols.proofs.spot.device.coins import OwnRandomness, closure_of, draw_first
from verity.protocols.proofs.spot.verifier.consumers import compute_bound, exfiltration_bound, infiltration_bound, location_bits
from verity.protocols.proofs.spot.prover.optimize import Candidate, CostModel, Objective, frontier, for_budget
from verity.protocols.proofs.spot.prover.candidates import Shape, cuts
from verity.protocols.proofs.spot.prover.costfit import fit
~~~

Each module's docstring is its reference: `service` (the proof system: `Spec`, `Stream`, `Fleet`),
`one_stage.docket` (the record and R1 to R8), `draw` (sampling designs), `partition`, `registered` (registered values),
`served`, `audit` (the audit record and the profile), `verdict`, `hello`, `verifier` (each check by the Lean and the
Python), `proofs` (Flock's Lean verifier on a statement's proof files), `parties`, `consumers`, `window_place`, `coins`
(in `../device/`), and the untrusted planners `optimize`, `candidates` and `costfit` (in `../prover/`).

## Round 2 of a two-stage run

Round 2: `second_stage` (Lean `SecondStage`); its `FORM` still refuses two-stage runs. A boundary record may commit a
step schedule (`schedule`, Lean `Schedule`: its replay blocks as runs of its requests' positions), which round 2
refuses unless it covers the replay blocks, and each request's positions, in order (`second-schedule`). What a replay
block returns is read only at the boundary record's port for it: an interior docket that registers a value of
that name is refused (`second-return`). It refuses an interior docket that registers a value the verifier
doesn't hold, unless the value is the opened row of a read against a table round 1's record registers (its root and
domain id those of one port of that record, its leaves the read's rows), at a key that is a port of that record or a
value the verifier holds, its verdict held to 1 (`second-inputs`): any other is the prover's choice after the first
stage's sample. A read is a template the auditor's params record as one (`"read": {"V", "H"}` beside its circuit
file's SHA-512, `template_reads`): the record that `flock-read-tmpl check V H` accepted that file, made where the
params are; round 2 runs no template check.

## The parts in detail

- **The Lean.** Today it sits in Flock's soundness package, `security/Proofs/Flock/Soundness/Audit/`,
  whose `README.md` lists every theorem; `Audit/Flock*.lean` instantiate them for Flock.
  The two-stage design's certificate is `Claims.SamplingGeneric.compilerKSTwo` (`Claims.SamplingTwo`).
- **The executable sample** is the Lean verifier's `verity/commitments/lean/Commitments/Draw.lean` (`verity-verify-proof draw`): the
  samplers, the strata and sizes, the sample object and its checks U1 to U3. Soundness's `ExecSample.lean`,
  `ExecStratified.lean` and `ExecSampleOS.lean` prove that its samples accept incorrect proof blocks no more often than the
  modelled sampling designs do, and `Audit/FlockWork.lean` that its sizes are the model's.
- **The verifier.** `one_stage.verifier` runs the Lean port of the one-stage audit (the Lake package
  `verity/protocols/proofs/spot/verifier/lean`, its `spot-verifier`) and the Python beside it on every check: R1-R8
  (`check`, with the fan-out its verdict publishes), `matches_public`, the challenger's sample (`select`), `check_draw`, `check_proved`, the whole audit record
  (`audit_record`), and the service's public-items rule (`check_public`), hold on an outcome's verdicts
  (`check_verdicts`), stream rules (`stream`) and fleet rules (`fleet`). Its one line
  `RECORD_VERDICT = "lean"` names whose verdict is the record's (`"python"` switches back). The service's `Stream` and
  `benchmarks/one_stage` call it. Python is held to the Lean by
  `verity/protocols/proofs/spot/verifier/lean/tests/test_difftest.py`. The executable answers the challenger's ledger
  and session checks too (`challenger_ledger`, `challenger_head`, `challenger_session`, `challenger_credited` and
  `challenger_pinned`, `Spot/ChallengerSession.lean`), for a caller that asks the verifier for them;
  `lean/tests/test_challenger_difftest.py` holds its answers to the challenger's Python. The same package ports the challenger's sample from the
  selection rule on (`Spot/Select.lean`: `service.selection_law`, `draw.derive` and `service.select_draw`,
  with the number of coin bytes it reads), which the difftest holds to the Python on fixed coins, every rule and random
  partitions. It is the `Stream`'s sample of record.
  The Lean also runs a docket whole, from the docket through the verdict (`Spot/Run.lean`,
  `verifier.run`). `Stream.outcome` holds each outcome on a sample from the coins to it, and an outcome the two don't make alike
  is a reject whose audit record fails `run`. `Spot/Slots.lean` puts that sample in the claims' terms
  (`Claims.Sampling.Partition`, `Deploy.checked`).
- **The audit on the proofs** (`verity/sampled-proofs/one-stage/v2`, `Spot/OneStageZk.lean`, the executable
  `spot-one-stage-zk`): v1 with the verdicts computed rather than read. Its parameters, evidence, refusals
  and check are in the module's docstring; the parameters, one statement's check and a member's statement and files
  are in `Spot/OneStageZkStatement.lean`, which it imports, and the run over the members and the checks in the
  module itself, all in the namespace `Spot.OneStageZk`. It reads the
  sample's coins as v1 does (`OneStage.coinsOf`; `oneStageZkReads` in `security/Properties/Challenger.lean`) and holds
  the session's record to the coin trees the signed release log shows the challenger sampled and opened
  (`Spot/ChallengerTrees.lean`, `challenger.sessions_tree_reason`, `order_why`).
  `one_stage.proofs.check_v2` calls it, and `verdicts_v2` is its verdicts alone, which `tests/test_proofs.py` holds
  to `verify`'s.
  A window of several members has a per-member form of `proofs` (the parameters' `members`, the evidence's
  `statements`; `one_stage.proofs.members_v2` writes it), which the audit's `check` reads beside the one
  statement's and decides by `checkMembersAt`: it accepts where the form's run returns with every session accepted
  and v1's check at the coins accepts on the sessions' verdicts (`one_stage.members.check`, with the Python of the
  run, `members_run`). The pass that reads every member's files (`membersFiles`) is compiled as `membersFilesKept`,
  proved equal to it: it reads the blobs' lookup tables once and a circuit file once for the members that name it,
  and keeps of the file the circuit's own verdicts and the circuit without its rows (`Spot/CircuitKept.lean`), so
  its work on a circuit's rows is by the file and not by the member; and it parses and hashes the program, and
  evaluates the partition's query on it, once for the members that read it under the parameters' partition, so its
  work on the program is by the program and not by the member. The executable's `members` (`Spot/MembersParse.lean`,
  `proofs.members_parsed`) prints what its parsers read of a case, with no verdict, as its module's docstring says. A
  window's sessions run on its challenger's session, each closed there after its evidence
  (`one_stage.closing`, `challenger.members_tree_reason`). The parameters may name the members' wiring beside them
  (`proofs.wiring`, SPOT's `Spot.Wires.Wiring`, refused where `Spot.Wires.check` refuses it), and the check then holds
  every wired row of every unit of the sample to its writer's output row in the members' public files
  (`OneStageZk.wiresHeld`, the `members_wires` piece); `one_stage.members` leaves a wiring `Outside`.
- **A monitoring period's sessions in the auditor's ledger** (`Spot/MonitoringPeriod.lean`, Lean only; the Python it is held
  to is a reference in `lean/tests/test_period_difftest.py`): from the ledger's signed lines, under the auditor's
  challenger key, every session of a lane that the ledger opens on a root during a monitoring period, whatever its
  layout (`sessionsOf`), and of those the windows, the sessions whose layout holds the sample's family (`windowsOf`),
  with the lane's other sessions beside them. The monitoring period is a run of the ledger's entries, from the first
  stamped at or after its start to the first stamped at or after its stop, so monitoring periods that follow one
  another split the entries exactly whatever the stamps say. It refuses a ledger that does not reach the stop, a
  window in no period of the lane, a period left open or listing a session that is no window, a reserve never opened
  and a window that has not ended. It reads the ledger only: no window is audited here.
- **SPOT's monitoring period** (`Spot/MonitoringPeriodAudit.lean`, `verity/sampled-proofs/period/v1`, Lean only, with the same
  reference): the check that every window of a lane's monitoring period passed the one-stage audit on its proofs
  (`OneStageZk`), on the one sample its challenger session gave it. Its commitment is the auditor's ledger; its
  evidence is the host's pins log and, for each window, the session's files and one case. The verifier's parameters
  are the period's alone, this docket's circuit among them, the SHA-512 a period of the lane commits when the
  ledger opens it: a case supplies its docket's public file's digest and its sample's segment, and
  nothing else of what one-stage reads. Only a window of one docket passes: the check reads the roots of every
  family the window's session reserved, and a session whose signed log shows a second commitment in any of them
  (another rank's or a second stage's in a family of its own, or the sample's family's second) is refused until the
  check audits each, as is a public file that shares rows among members. Every window's period commits this
  docket's circuit, and a session of the lane that holds no coin of the family is refused unless a period
  that commits another circuit lists it: a root committed under this docket and never sampled refuses the
  period. It publishes each window with the docket audited and each session of the lane that is another
  docket's, by that docket's circuit and its session alone, of which it holds nothing. Its executable
  is `spot-monitoring-period` (`id`, `check PARAMS COMMITMENT EVIDENCE [BLOB ...]`, `serve`); nothing calls it yet
  but its difftest. What its accept means is stated in `security/Properties/Spot/MonitoringPeriod.lean`, and whose ledger it
  reads in `security/Properties/Spot/MonitoringPeriodLedger.lean`.
- **The challenger** (`challenger`, Lean of record `Spot/Challenger.lean`): every protocol's live coins, by
  label and index, each opened once, after its root is recorded. On the auditor device its coins reach the developer's
  machine only through that machine's certifier coin link (`verity/protocols/compose/device/live_challenger/auditor.py`, the coins' path).
  A remote prover reads them the same way: one TCP stream at its own `127.0.0.1` port (`RemoteChallenger`), deployed
  as the far end of that coin link (`verity/protocols/compose/device/live_challenger/coin_link.py`).
  - **Remote's coin path in Lean** (`security/Definitions/Core/Challenger.lean`, assumptions in its `Assumptions/`, in the
    lock): `ChallengerLogsCoins`, `FlipsAfterRecorded` and `VerifierReadsLog`. Under them
    a logged coin's sample misses a set fixed before it at most as the sampling design allows (`SampleAfterRecorded`, proved).
    None mentions a network certifier.

  Onsite, the challenger rests on two more premises about where it runs, which the network certifier's spec and
  ACCEL's cite by name:
  - `CoinsConfidential` is `Pouw.Assumptions.CoinsConfidential` (`security/Definitions/Pouw/Assumptions/Dither.lean`,
    in the lock), whose docstring is its statement, and ACCEL's pooled guarantee
    `Pouw.Guarantees.Dither.FleetSound` takes it. A path across a host the developer controls would need an encrypted,
    authenticated channel, which the challenger doesn't have. Deployed, one auditor device runs the challenger and the
    certifier, and a run's coins leave it only on the peer's certifier coin link, which the gate refuses to open
    without (`auditor.coin_ports`).
  - `OneClock`, an assumption under the trusted auditor device, with no Lean statement yet: the challenger's release times and the certifier's
    ingress and receipt times are read on one clock. Deployed, that is the auditor device's. A challenger on a device of its own would have to keep its clock within
    1.03 s of the certifier's, the offset ACCEL's timing allows.
- **The Python.** `verity.protocols.proofs.spot.verifier.one_stage` is the one-stage audit's reference and `service` the
  proof system over it; `optimize`, `candidates` and `costfit` plan a sampling design (`benchmarks/sampling_plan`).
  Core API lists each module.
- **The Lean-generated vectors.** `tests/fixtures/exfiltration_lean.json` is `audit_exfiltration` and
  `stratified_miss_eq_greedy` evaluated by Lean, and `exfiltration_two_lean.json` `card_influenceTwo_le_uniform`'s
  count on two-stage designs (`tests/lean/`, by `generate.sh`); `tests/test_consumers.py` holds the consumers to them.
  `tests/test_coins.py` holds the samplers and `draw_first` to the Lean verifier's `sampleOn` on raw bytes,
  `tests/test_one_stage.py` the work sizes to Flock's tests' pins, and `verity/protocols/proofs/flock/prover/tests/stratified_agree.py`
  the Lean verifier's sample to `one_stage.draw`.
- **The parties.** `parties` holds both deployments (Core API) as parties of `verity.runtime`, each machine's work a
  capability it is given, with SPOT[Flock]'s challenger on the verifier's side; `verity.runtime.runner.run` runs them.
- **The benchmark.** `benchmarks/one_stage` runs the one-stage audit end to end with Flock's circuit prover and the Lean
  verifier.
- **The users.** Flock samples and checks the sample in its Lean verifier and proves the sampled proof blocks. The vLLM integration
  commits serving rows with the docket and served domains of `one_stage` (`verity.commitments.periods.rows`, in the
  Commit or from a deferred Commit's bundle by `verity_vllm.pipeline.deferred_rows`); its replay challenge is its own
  (`verity_vllm.commit.challenge`).

### Verdicts of record

No function here returns a verdict of its own. Each verdict SPOT acts on is the Lean's: `one_stage.verifier` asks
`spot-verifier` and computes the Python twin beside it, and a disagreement, or no answer from the Lean, is a
refusal (`DISAGREE`). A verdict whose Lean no interface here reaches yet raises
`NotImplementedError("proofs/spot-two-stage-sound")` (`verifier.not_written`), and its twin answers the difftest only.

| Verdict | Entry | Lean of record | Twin |
| --- | --- | --- | --- |
| R1-R8 on a docket | `verifier.check` | `registration` (`OneStage`) | `registration.check` |
| The audit record | `verifier.audit_record` | `Record.record` | `audit.audit_record` |
| A docket's whole run | `verifier.run` | `Run.run` | the stream's own outcome |
| A two-stage outcome | `Stream.outcome` (`_audit_two`) | `Run.run`'s symbol; it refuses every two-stage run while `SecondStage.FORM` is off | the two records' join |
| Round 2's statements | `second_stage.check` | `SecondStage.checkAny`, which only `Run.run` reaches | `check_twin` |
| A two-stage outcome's completeness | `second_stage.complete` | none yet | `complete_twin` |

A two-stage run whose stages read no coin bytes gives `Run.run` nothing to read, so `_audit_two` raises rather than
join its records.

### Guards

Each takes a verdict of record or files Lean then checks, and can only refuse: none returns an accept a Lean verdict did
not give.

| Guard | Why it can only refuse |
| --- | --- |
| `parties`' `_held` | returns a reject answer unchanged, and turns an accept into a reject when its commitment or interior is not the docket sent; otherwise returns the answer it was given |
| `one_stage.proofs.ks_form_why` | a reason or None for a circuit outside the form `verify --zk` covers (`Flock.KsScope.filesWhy`); a reason refuses a session, None leaves the binary's verdict |
| `one_stage.proofs.template_why` | a reason or None for a circuit the template refuses, read only before `members.files_why`'s other refusals; None decides nothing |
| `one_stage.proofs.population_why` | a reason or None for a public file whose units are not the docket's population in order (`Spot.FilePart.why`); None leaves `verify`'s verdict |

## The docket

### The docket

A docket fixes everything a sampled proof block's statement can depend on before the verifier samples. Its record,
`verity/registration/v1` (`one_stage.docket.record`), names the program, query and partition digests, the
population, the sampling design expanded over the verifier's program and query, the commitment scheme, each port's root
with its leaf count and domain, the leaf layer, the anchors, the run (`window`: its provenance, `served` or `stand-in`, the
stream and prover it is bound to, and its seed salt) and `previous`, the digest of this prover's last record. A record
that registers values is `verity/registration/v2`: the same keys and `registered` (below), so v1's bytes are unchanged.
Its digest is `SHA-512(format ‖ 0 ‖ canon(record))`, canon being canonical JSON (RFC 8785,
`verity.circuits.codec.canonical_json`).

The verifier holds the record to its own copies with R1 to R8 (`check`). The Lean verifier runs the same check
(`Spot.Docket.check`), `Properties.Spot.R1` to `R8` state what each accepts, and
`verity/protocols/proofs/spot/verifier/lean/tests/test_difftest.py` holds the Python to the Lean.

It then writes a receipt (`receipt`, `verity/registration-receipt/v1`), and only then samples. It also holds its own public
file to the record (`matches_public`), which `FailClosed.flock_verify_public_ties` ties to the proved rows. The
challenger's `service.ExecutionPhaseCommitmentRecord` is a record it admitted, with its receipt and its position in the
stream. The format, each check's rule and codes and the window domains (`window_domain`) are in
`one_stage.docket`'s docstring.

Pinned by `tests/test_one_stage.py`, and the window domain also by `adapters/vllm/tests/commit/test_serving_rows.py`.

### Registered values

A program's weights, constants and tables are values: committed once, under hiding leaves, and read only inside a proof
(`one_stage.registered`, the Lean `Spot.Registered`, and Flock's `Flock.Registered`).

- **The commitment.** Row r of a registered value is `words` leaves of `word_bits` bits (16 or 32). It is committed as
  the circuit statement commits that port's row, in the port's row format (`hm96-sha512/row/v1`, or the bit rows
  `hm96-sha512/row/v2` and `hm96-sha512/row-seg/v1`), with a salt sampled once. The root is the frame-v3-sha512 root over
  the rows under the value's registered-value domain (`registered.domain`, on the SHA-512 identity frame of
  `verity/one-stage/registered-domain/v1`). The prover keeps the rows and salts
  (`Registered.commit`). The record's `registered` entry is `{"value", "root", "leaves", "words", "domain"}`, with
  `"word_bits"` when it is not 16 (`Registered.entry`).
- **R7** holds the entries to the verifier's own (`registered_entries`, from its own program). It holds each value to a
  root it already has, when it has one: its pinned root (`pinned`: the weights' root recorded before the salt), the root
  an earlier record of the log registered under the same domain (`held_roots`), or, for an output or reused value, the
  earlier record's port or entry (`outputs`). `Stream.register` and `register_interiors` take `outputs`, keep the record
  each one names in the log R7 reads, and pass them to the run of record: this is how a record reads the state an
  earlier one left, such as a request carried from one run's serving into the next.
- **Per-run values.** Given the run's open root R, from the challenger's signed open of it, every
  value's domain binds R, except the ones the verifier holds across runs (`held`, and its pinned, supplied and output
  values). So a run's values are held to no earlier run's roots. A verifier that names no R refuses its own inputs
  when a value it doesn't hold across runs would be held to an earlier run's root (`unwindowed`).
- **Supplied values.** These are values the verifier fills itself, from the record's roots, positions after the log's,
  its run's seed salt (bound to its stream and prover) and its own bytes. They are committed under public salts, and R7
  recomputes their root (`R7-supplied`). PoUW's seed inputs are such values
  (`verity/protocols/accounting/pouw/verifier/tests/test_pouw_supplied_inputs.py`).
- **Certified values.** R7 holds a registered value no parameter reads to the verifier's `certified` entry of it.
- **Reads.** Every port of a statement is either a registered port of the `--registered` file
  (`Registered.checkComplete`) or bound to the window's committed values: the file's `bound` names its rows' root and
  domain as the docket holds them, and the public header's must be the same (`bind_ports`; `Flock.RegBound`). No
  port that reads a registered input of the verifier's own program is ever bound, and nor is any port the window's
  registered circuit record declares registered for the statement's template (its registered reads, table reads and
  openings' verdicts, outputs included: `RegBound.declaredOf`), which must be a registered port of the file. A file
  that binds ports is verified with that record (`verity-verify-proof --interface`), and refused without it.
- **Served trees.** A served path registers each tree a proof block reads as a registered value. `served.register_served`
  returns a `ServedDocket`: the record's entries, R7's copies and the `--registered` file. That docket is the
  premise `RowsSession.ServedRegistered` of `Flock.Guarantees.ServedRowsLinked`.
- **The audit record** holds the roots the verifier opened the reads against (`audit.session_registered`, Lean
  `Record.sessionRegistered`) to the docket's own (`Record.ownRegistered`), taking a read named by one of its
  own ports at that port's root (`audit.less_own_reads`, `Record.lessOwnReads`) and refusing it at any other. It holds
  a read of a window value its caller names (the parameters' `window_values`, none by default; never a port or a
  registered value of the verifier's) to the root of that value's entry in the record's `window.activations.values`,
  refusing it at any other root, under two roots, or with no such entry (`audit.window_reads`, `Record.windowReads`).

Pinned by `tests/test_registered.py`, `tests/test_served.py` and `verity/protocols/proofs/flock/prover/tests/test_lean_registered_reads.py`.

## Not yet in Lean

The rules below are normative, and no Lean states them yet: the Python is their reference, pinned by the tests and vectors
named with each. A rule leaves this list in the change whose Lean states it.

### The partition

Core's `verity/partition/v1` (`verity.circuits.partition_object`, applied by `one_stage.partition`) is
`{"format": "verity/partition/v1", "program": <SHA-512 of the program>, "query": {"name", "version", "params"}}`, with
digest `SHA-512("verity/partition/v1\0" ‖ canon(object))`.

- The proof blocks are core's evaluation of the query, which refuses a malformed query and a program that is not topological.
  The verifier applies the query to its own copy of the program, and never takes a population or a proof block index on the
  prover's word.
- The one-stage audit applies `Q_template_instance` v0 and `Q_template_instances` v0, and `Q_word` v1 at X = 16 and
  `Q_call` v1 at X <= 16. Under a template query each instance of a listed template in the program's root body is one
  proof block, in canonical order (node, then member). Every proof block's committed bits fit a public size class, a power of two
  from 16 to `SIZE_CLASS_MAX` bits (`size_class`), and a program with a wider proof block is refused. The
  template queries are demo queries for template runs, not the serving partition. Only they define strata; any other
  query refuses the stratified and work sampling designs.
- A record made before the canonical envelope names `{"program", "query"}`, and keeps that object's digest under the same
  rule (`partition.digest`).

The Lean verifier's executable reads the same object (`Flock/HmRow.lean`, `Flock/TemplateQuery.lean`), but no theorem
states it. The guarantees read the circuit file's partition, `Flock.Bridge.partOf`; that the deployed record's is it
is `RelDeployTarget`'s hypothesis, discharged once the layout is built from `FilePart.ofCircuit`. Pinned by
core's `verity/circuits/tests/partition_vectors.json` and `template_instance_vectors.json`, which
`tests/test_one_stage.py` replays, with the object's refusals and digest.

### Sampling designs as text and as registered

- A sampling design's text is `subset:K` or `bernoulli:NUM/DEN` with `NUM ≤ DEN` and `DEN > 0` (`flock-circuit serve
  --draw`'s syntax), or one of the rules `stratified:K` and `work:K` with `K ≥ 1` (`draw.parse_design`), each number in ASCII digits
  (`draw.digits`, as the Lean reads them; a sample's `p` too).
- As objects they are the forms `one_stage.draw`'s docstring gives, the rules expanded by `draw.derive`. The
  docket carries the sampling design expanded.
- `derive` refuses a `K` outside `[1, N]`, a work sampling design without the verifier's work table, a subcircuit the
  table does not name, and a population that does no work.
- Two sampling designs are the same when their canonical JSON bytes are (`draw.same_design`), so `true` is not `1`, as the
  Lean verifier reads them.
- `check_draw` also holds the sample's population to the docket's.

Pinned by `tests/test_one_stage.py`'s checks of each sampling design's text, objects and expansion.

### What a proof block's work may count

The work sampling design's bound on miss probability holds over the work the table credits; what the table may
credit is a rule of its own, stated in `one_stage.draw`'s docstring ("What the table credits"): the rows the Program
declares, nothing for filler, and one work for every proof block of a subcircuit.

`test_derive_expands_the_work_rule_over_the_verifiers_table` covers a subcircuit credited nothing.

### The audit record and the profile

The one-stage profile (`audit.profile`) is core's `IntegrityProfile` of one level `unit` (`Level("unit", <partition
digest>)`) with `sizes =
(population,)`, sampled by the sampling design (`Subset(k)`, `Bernoulli(p)`, or `Stratified` with one stratum per template), with `delta`, the program's digest and the count sampled. It bounds incorrect proof blocks and weights nothing. The work
sampling design's profile is `Stratified` over the same strata at the work sizes, counting the sampled proof blocks only: a
closure proves more, which only helps.

The audit record (`audit.audit_record`) is `{"format": "verity/one-stage/audit/v0", "registration", "registration_digest",
"receipt", "draw", "law", "proved_units", "statement", "checks", "verdicts", "accepted", "complete", "profile",
"integrity_profile", "wrong_units"}`:

- `checks` holds `draw` (`check_draw`) and `proved` (`check_proved`), and a failed `receipt` (it must name the
  docket's digest), `served` (each session was served exactly the sample) or `roots` (each session's root is the
  registered one).
- A verdict is `{"accepted": true | false | null, ...}`; null means that verifier did not run, which leaves the record
  incomplete (`complete` is false).
- The record is accepted only when there is a verifier whose verdict counts (`of_record`), every one of them ran and accepted, every
  other verdict that ran accepted, and every check passed. The profiles are present only then.
- A session that proves one subcircuit's proof blocks is served that subcircuit's share as a `subset:k_s` sample over its `n_s` proof blocks,
  in its own indices; the record keeps the union sample.
- `profile` is the profile's JSON, except that a stratified or work sampling design keeps the v0 form: its JSON with
  `delta` and the count sampled per subcircuit.
- `wrong_units` is `{"delta", "of": population, "bound", "rule", ["per_template"]}`: `bound` is core's exact
  `Stratified.count_bound` at delta for `subset` (one stratum), `stratified` and `work`, and `Bernoulli.count_bound` for
  `bernoulli`; a stratified or work sampling design adds each subcircuit's own bound.
- `integrity_profile` is `verity/integrity-profile/v1`: `{"format", "protocol", "program", "query", "partition",
  "population", "registration": <digest>, "scheme", "roots": [{"port", "root", "leaves"}], "law", "drawn",
  "inputs": {"anchored": <anchors>, "linked": []}, "budget_log2_t": null, "terms": null, "epsilon_log2": null,
  "evidence": {"receipt": <SHA-512 of canon(receipt)>, "draw": <SHA-512 of canon(draw)>, "sessions", "verdict"}}`,
  `sessions` being the session records' SHA-512 and `verdict` the verifier's digest (a list when the sampled
  proof blocks were proved in several sessions). Its ε terms come from the soundness analysis, not from the run, so the run leaves them null.

One-stage's answer of an accepted record (`audit.answer`, as Lean's `OneStage.check` gives it) publishes `{"bound",
"complete", "per_template", "proved"}` and the sample's units under the integrity profile's key for them. `complete`
is the record's own: whether every verdict ran. An accept waits on the verdicts of record and on those that ran, so an
accept can say `complete` is false, and then a verdict that is not of record did not run. A reader that wants a
complete audit reads `audit.answer_complete`, never the verdict alone: it is true only of an accept whose `complete`
is true, so an answer without the field (one written before it, a reject, an undecided) is not complete. A two-stage
outcome says it only under `second_stage.complete`, which raises until its Lean is written (`verifier.not_written`,
goal `proofs/spot-two-stage-sound`); its twin `complete_twin` holds when the two-stage form is allowed, the replay block
is the registered request Definition, and each audit record accepted and is complete. A two-stage outcome's accept is
`Run.run`'s symbol alone: round 2's check (`second_stage.check`) raises the same way, and its twin `check_twin` is the
Python side the run is held to.

Pinned by `tests/test_one_stage.py` (the profile, audit record, answer and incorrect-block bound tests, under each
sampling design).

### The consumers

`consumers` reads any `IntegrityProfile`, whichever protocol produced it, at its
delta, through its weighted bound `harm_bound`; each application counts delta once.

- `infiltration_bound(profile, output_bits)` is `harm_bound(output_bits)`, in bits: the committed values the prover wrote
  in place of what the circuit computes, served or not, so no served output's size caps it. A two-stage design's profile
  is read at the footprint, so there it counts only what the boundary commitment pins down.
- `exfiltration_bound(profile, output_bits)` is `infiltration_bound(profile, output_bits) + location_bits(profile)`, in bits.
  `location_bits` is the location term: with `K = ⌊bound()⌋` of `n` proof blocks, 0 when `K ≤ 0` or `n = 0`, `n` when
  `2K ≥ n`, and otherwise `n H₂(K/n) (1 + MARGIN)` rounded up to the next float, at most `n`. The Lean states the bound
  (`audit_exfiltration`, `card_admissible_le`; for a two-stage design's profile, counted at the footprint,
  `card_influenceTwo_le_uniform`), and the tests hold the Python to be at least it.
- `compute_bound(profile, work)` is `harm_bound(work)`. Policies and PoUW pass their own utility to `worst_case`. It
  refuses a two-stage design's profile (`UNCERTIFIED_COMPUTE`): its interiors are committed after the sample, so its
  relation reads none (`rTwo_congr`) and takes a trace whose interiors were never computed (`deferred_not_credited`).
- A proof block's output bits or work is one number for every proof block, or per stratum a number or `(bits, units)` pairs.

Pinned by `tests/test_consumers.py` and the Lean-generated vectors, each regenerated when its Lean files change (a test compares their sha256).

### The proof system

`service.Stream` holds one protocol's dockets to its `Spec`, on the challenger's side. Each rule (the public-items rule,
the sample, sessions over several rounds, the salt and the docket deadline, the seed salt, the stream's rules, the
fleet's halt, no retries, the failure budget, the log's history and the records), its formats and its refusal codes are
in `service`'s module docstring. Pinned by `tests/test_service.py`, `tests/test_service_history.py`,
`tests/test_service_sessions.py` and `tests/test_service_fleet.py`, and the stream's and the fleet's rules by the
difftest's `stream` and `fleet` pieces.

### The sample's coins

The verifier flips no coins of its own: a sample's coins are the auditor's challenger's, read from its log. The rules,
the byte layout and the evidence are in `service`'s docstring (`Stream.outcome`) and `coins`'; `Run.run` refuses a
sample that read coins without the challenger's evidence or under another `coin_source`. Pinned by
`tests/test_coins.py`, `tests/test_challenger_coins.py` and the difftest's `challenger_read`, `challenger_draw_coins`
and `run`.

### The registered compiler

A verifier that reads the registration log's registered compiler (`compiler`: the log's item as its view publishes it,
with its digest and the public length ℓ of the advice it reads, or null when the log registers none) holds the record's
window to name that compiler by its digest (`window.compiler.digest`, `R6-compiler`) and advice of at most ℓ bits
(`window.advice.bits`, `R6-advice-length`); where the log registers none, a window naming a compiler or advice is
`R6-compiler`. Lean `Docket.compilerOk` and `adviceOk`, in `RunCompiler` of R6; `tests/test_registered.py` and
the difftest's `test_advice_no_registered_compiler_reads_is_refused_by_name`.
