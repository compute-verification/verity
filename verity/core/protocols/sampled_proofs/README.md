# Sampled proofs (`verity-sampled-proofs`)

Sampled proofs spot-check a committed computation. While serving, the developer commits every value that crosses a
proof-block boundary of a partition of the program and registers the commitment roots. The verifier then draws proof blocks
from its own randomness, the developer proves the drawn proof blocks, and the verifier checks the proofs. It never runs any part
of the computation. An accepted audit yields an integrity profile (`verity.core.protocols.profile.IntegrityProfile`): except with
probability δ, at most so many proof blocks are faulty, a proof block being faulty when its committed outputs differ from its gates
applied to its committed inputs. The protocol is stated in Lean (see "Where things are"); this README outlines it,
points to the code, and lists what the Lean does not state yet.

Sampled proofs is a subprotocol of both composite protocols. The remote protocol (`verity/remote/v0`,
`verity/core/protocols/remote/`) composes its one-stage audit (`verity/sampled-proofs/one-stage/v1`) with PoUW
(`verity/pouw/v1`), and onsite (`verity/core/protocols/onsite/`) composes it with the other subprotocols. In each, the
composite's verifier runs the one-stage audit's Lean check as a component, and the composite accepts only when every
component accepts.

## Guarantee and assumptions

The guarantee is knowledge soundness of the sampled audit: except with a bounded probability, when the verifier
accepts, at most the profile's count of proof blocks are faulty. Its top statement is C-Flock's
`Flock.Guarantees.SampledKSLiveV3`, stated in `Security/Proofs/Flock/Bridge/Refused.lean` and proved in
`Security/Proofs/Flock/Bridge/KSLive.lean`, and it bounds the probability
that `flock-verify --zk` accepts the files and the run passes the registration's population while the claim fails. It
rests on these named assumptions:

- `UniformRandomBytes`: the bytes the verifier's draw reads from its operating system are uniform (the CSPRNG
  idealized; its docstring says what that leaves out).
- `RecordCustodyZKJ`: the record `flock-verify --zk` reads is the one the challenger wrote for the live session, and its
  draw file is the one `flock-verify draw` printed for the verifier's own draw.
- `TableCRZAt` (`cr/sha-512`; the statement reads it through its abbreviation `TableCRZ`) and `LinkCRZC`
  (`ecr/sha-512`): SHA-512's collision resistance, for the collision finders the proof builds from the prover.

On `main`, both composites' sampled-proofs component is one-stage v1 (`verity/sampled-proofs/one-stage/v1`), which
takes the sessions' verdicts as evidence: the verifier's run of `flock-verify --zk` produces them, and the link from a
composite's acceptance to that run, and so to `SampledKSLiveV3`, comes with one-stage v2 (under "Where things are").

The draw happens after the registration, so the prover fixes everything a drawn proof block's statement depends on
before any coin. In the remote protocol the coins come from the auditor's challenger (or the verifier's own draw),
which adds two premises about where it runs, `CoinsConfidential` and `OneClock` ("Where things are"). Both are prose
only: no `.lean` file defines either.

## Outline

**The one-stage audit** is the audit protocol in use. One round goes like this:

1. Serving commits the values that the partition's proof blocks read and write, and sends the roots.
2. The developer registers a record that fixes everything a drawn proof block's statement can depend on: the program, the query
   that partitions it, the sampling design, the commitment scheme and the roots. The verifier checks the record against
   its own copies and logs a receipt before it draws.
3. The verifier draws proof blocks from its operating system's randomness under the agreed sampling design and sends the draw
   in the clear, before the session's coin commitment. Nothing is derived from a commitment, and there is no beacon.
4. The drawn proof blocks are proved in one session, whose statement is the drawn proof blocks'.
5. The Lean verifier checks the proofs, and its verdict is the one that counts; the Python beside it is its mirror.
6. The audit record keeps the registration, the receipt, the draw, the proved proof blocks, the statement's digests and every
   verdict, and, when everything accepted, the integrity profile.

**The proof service** (`service`) runs these steps for a protocol, on the challenger's side. A protocol states a `Spec`:
- what it commits;
- its selection rule;
- the check, as a public Program;
- its public items, each with its reason.

Everything the auditor observes must be a function of those items. A `Stream` per `Spec` holds each registration to that
rule and then to one-stage's checks, draws once, binds the delivered proof to its outcome, and counts failures against a
public budget.

**The sampling designs.** `subset:K` is a uniform `K`-subset of the `N` proof blocks, and `bernoulli:NUM/DEN` takes each proof block
independently with probability `NUM/DEN`.
The default is `stratified:K`: one stratum per subcircuit, and inside each an independent uniform draw
without replacement of `k_s = max(1, ⌊(2 K n_s + N) / (2N)⌋)` proof blocks, that is `K n_s / N` rounded half up and never fewer
than one, so every subcircuit is exercised in every audit. `work:K` keeps those strata and that sampler
but sizes each stratum by the work its proof blocks are credited: with `W = Σ_s w_s n_s`, stratum `s` draws
`k_s = min(n_s, max(f_s, ⌈K w_s n_s / W⌉))`, where `w_s` and the floor `f_s ≥ 1` come from the verifier's own work table.
A work draw may add a closure, proof blocks proved with the drawn ones. Each verifier derives the strata from its own program and
query, never from the prover.

**What an accepted audit establishes.** Under a stratified draw, a set of faulty proof blocks has acceptance probability
`Π_s C(n_s − m_s, k_s) / C(n_s, k_s)` (the probability that the draw misses all of them), `m_s` being its proof blocks in
stratum `s`, so a subcircuit that is faulty in every proof block is never missed. Under the work sampling design, faulty proof blocks
doing a fraction `ε` of the credited work have acceptance probability at most `(1 − ε)^K`; at `K = 27,713`, a 0.1% share
has at most `2^-40`. The proof system adds, per proof block, its knowledge-soundness and binding errors (ε_ks and ε_bind).
Applications weight the profile's faulty proof blocks by what each one costs them: exfiltration security by their output bits,
plus a term for which proof blocks are faulty, and compute security by their work.

Everything is committed at serve time: the proof service has no two-stage rule, and the Lean verifier refuses a
registration that names one.
Not built yet: the security function and its review, and composition checks such as PoUW's tiles
per proof block matching this partition (the IR checks refinement, `verity.ml.circuits.partition.validate_refinement`).

Serving's record is not yet the scheme the audit proves: the benchmark's first run (`benchmarks/one_stage/a0.py`)
registers a stand-in, a fresh commitment to values captured from serving, and says so (`window.kind = "stand-in"`). Its
served run (`benchmarks/one_stage/a5.py`) registers a stand-in too: a deferred Commit's serving rows, written after serving from its sealed replay bundle opened against the run's record
(`verity-vllm serving-rows`), whose `of` names the record's vllm-v1 run root and the bundle's manifest digest; the link from
those roots to the vllm-v1 roots is the committer's opening, not proof. Rows the in-Commit hook writes before any draw are
`served`: serving's own commitment, with no claim that they hold the values under the Commit's roots. A deferred window becomes
`served` once serving's record is frame-v3-sha512 / hm96-sha512 itself, or once the circuit opens the vllm-v1 SHA-256 leaves.

## Where things are

- **The Lean.** Today it sits in C-Flock's soundness package, `Security/Proofs/Flock/Soundness/Audit/`,
  whose `README.md` lists every theorem: the one-stage audit (`OneStage`), the sampling designs (`Law`, `Stratified`,
  `StratifiedMiss`, `Work`, `Closure`, `Window`), the registration-dependent draw (`RegDraw`), the retired two stages
  (`TwoStage`, with the tight and insecure cases in `Examples`), exfiltration (`Influence`, `Harm`), the compiled layer
  (`Extraction`) and composition (`Circuit`); `Audit/Flock*.lean` instantiate them for C-Flock. The verifier's coins are
  the named assumption `Assumptions.UniformRandomBytes`.
- **The executable draw** is the Lean verifier's `verity/core/service/flock/Flock/Draw.lean` (`flock-verify draw`): the
  samplers, the strata and sizes, the draw object and its checks U1 to U3, specified in
  `verity/ml/flock/verifier/PROTOCOL.md` §7.3. Soundness's `ExecDraw.lean`, `ExecStratified.lean` and `ExecDrawOS.lean`
  prove that its draws accept faulty proof blocks no more often than the modelled sampling designs do, and
  `Audit/FlockWork.lean` that its sizes are the model's.
- **The verifier.** `one_stage.verifier` runs the Lean port of the one-stage audit (the Lake package
  `verity/core/service/sampled_proofs`, its `sampled-proofs-verifier`) and the Python beside it on every check: R1-R7
  (`check`), `matches_public`, the challenger's draw (`select`), `check_draw`, `check_proved`, the whole audit record
  (`audit_record`), and the service's public-items rule (`check_public`), hold on an outcome's verdicts
  (`check_verdicts`), stream rules (`stream`) and fleet rules (`fleet`). Its one line
  `RECORD_VERDICT = "lean"` names whose verdict is the record's (`"python"` switches back); both outcomes are recorded
  beside it (`verifiers`), and where they disagree the check is refused naming both. The service's `Stream` and
  `benchmarks/one_stage` call it. Python is held to the Lean by
  `verity/core/service/sampled_proofs/tests/test_difftest.py`. The same package ports the challenger's draw from the
  selection rule on (`SampledProofs/Select.lean`: `service.selection_law`, `draw.derive` and `service.select_draw`,
  with the number of coin bytes it reads), which the difftest holds to the Python on fixed coins, every rule and random
  partitions. It is the `Stream`'s draw of record: `Stream.select` hands it the bytes of the challenger's coins the
  Python draw read, and refuses the draw (`disagree`) unless the Lean draws the same object from exactly those bytes.
  The Lean also runs a registration whole, from the registration through the verdict (`SampledProofs/Run.lean`,
  `verifier.run`): the public-items rule, R1-R7, the receipt, the draw from those bytes and its checks, the hold on
  the verdicts and the audit record. `Stream.outcome` holds each outcome on a draw from the coins to it, and an outcome the two don't make alike
  is a reject whose audit record fails `run`. `SampledProofs/Slots.lean` puts that draw in the claims' terms
  (`Claims.Sampling.Partition`, `Deploy.checked`): the partition as two classes, the population's strata and then the
  root's input gates (drawn `all`: the input proof block is always checked), each replay block one proof block, and the draw's
  checked proof blocks as `(slot, index)` in class, stratum and replay-block order.
- **The audit on the proofs** (`verity/sampled-proofs/one-stage/v2`, `SampledProofs/OneStageZk.lean`, the executable
  `sampled-proofs-one-stage-zk`): v1 with the verdicts computed rather than read. Its parameters add the auditor's
  circuit digest and the verifier's registered file, its evidence names the public file and each session's record and
  rep files by SHA-512 in place of `verdicts`, and the files come in as `Core.Blobs`. The check runs
  `flock-verify verify --zk`'s one-stage path (the C-Flock library calls `verifyCmd` makes) on those blobs, binds the
  evidence's `proved_units` to what the accepted sessions proved, and hands the verdicts to v1's `check`.
  `one_stage.proofs.check_v2` calls it, and `verdicts_v2` is its verdicts alone, which `tests/test_proofs.py` holds
  to `verify`'s. Onsite, remote and composition still read v1.
- **The challenger** (`challenger`, Lean of record `SampledProofs/Challenger.lean`): every protocol's live coins, by
  label and index, each opened once, after its root is recorded. On the auditor host its coins reach the prover's
  machine only through that machine's certifier coin link (`benchmarks/live_challenger/auditor.py`, the coins' path).
  A remote prover reads them the same way: one TCP stream at its own `127.0.0.1` port (`RemoteChallenger`), deployed
  as the far end of that coin link (`benchmarks/live_challenger/coin_link.py`); remote has no coin path of its own yet.
  The challenger rests on two premises about where it runs, which the network certifier's spec and ACCEL's cite by
  name; both are prose only:
  - `CoinsConfidential`: the challenger's coin stream reaches the network certifier only over a path on auditor hosts
    (`trusted-host`), or the challenger and the certifier are one device at the boundary. So no machine outside the
    boundary reads a coin before the certifier admits it, and a path across a host the prover controls is not
    allowed: it would need an encrypted, authenticated channel, which the challenger doesn't have. Deployed, one
    auditor host runs both, and a window's coins leave it only on the peer's certifier coin link,
    which the gate refuses to open without (`auditor.coin_ports`). It is not in Lean: no `.lean` file defines it.
    ACCEL's pooled guarantee `Pouw.Guarantees.Dither.FleetSound` builds a form of it into its statement instead (each
    unit's prover is a function of the other units' coins).
  - `OneClock`, a named premise under the trusted auditor host, with no Lean statement yet: the challenger's release times and the certifier's
    ingress and receipt times are read on one clock. Deployed, that is the auditor host's. A challenger on a device of its own would have to keep its clock within
    1.03 s of the certifier's, the offset ACCEL's timing allows.
- **The Python.** `verity.core.protocols.sampled_proofs.one_stage` is the one-stage audit's reference: `registration`,
  `draw` (the draw's checks, not a sampler), `partition`, `audit` and `verdict`. `consumers` reads its profile. `service` is the proof service over it, whose `Stream`
  draws from the challenger's own randomness as `coins.draw_first` does. `coins` is the verifier's randomness, and
  `plan` holds verification plans over IR Programs. `optimize`, `candidates` and
  `costfit` plan a sampling design (`benchmarks/sampling_plan`).
- **The Lean-generated vectors.** `tests/fixtures/exfiltration_lean.json` is `audit_exfiltration` and
  `stratified_miss_eq_greedy` evaluated by Lean (`tests/lean/ExfiltrationVectors.lean`, regenerated by
  `tests/lean/generate.sh`), and `tests/test_consumers.py` holds the consumers to it. `tests/test_coins.py` checks the
  samplers and `draw_first` under each sampling design against the Lean verifier's (`drawOn`) on raw bytes, and
  `tests/test_one_stage.py` the work sizes against the ones C-Flock's
  tests pin. `verity/ml/flock/tests/stratified_agree.py` runs the Lean verifier's draw against `one_stage.draw`.
- **The parties.** `parties` holds the challenger, the firewall, the workers and the verifier as parties of the
  deployment layer (`verity.core.deploy`), with their manifest and table (`PROTOCOL`), each machine's work a
  capability it is given. `verity run CONFIG --party ROLE` runs them.
- **The benchmark.** `benchmarks/one_stage` runs the one-stage audit end to end with C-Flock's circuit prover and the Lean
  verifier.
- **The users.** C-Flock draws and checks the draw in its Lean verifier and proves the drawn proof blocks. The vLLM integration
  commits serving rows with the registration and served domains of `one_stage` (`verity.ml.windows.rows`, in the
  Commit or from a deferred Commit's bundle by `verity_vllm.pipeline.deferred_rows`); its replay challenge is its own
  (`verity_vllm.commit.challenge`).

Tests: `uv run tools/verity/check/suites.py protocols/sampled_proofs`.

## Core API

~~~python
from verity.core.protocols.sampled_proofs import service
from verity.core.protocols.sampled_proofs.one_stage import audit, draw, partition, registration
from verity.core.protocols.sampled_proofs.coins import OwnRandomness, closure_of, draw_first
from verity.core.protocols.sampled_proofs.consumers import compute_bound, exfiltration_bound, location_bits
from verity.core.protocols.sampled_proofs.plan import VerificationPlan
from verity.core.protocols.sampled_proofs.optimize import Candidate, CostModel, Objective, frontier, for_budget
from verity.core.protocols.sampled_proofs.candidates import Shape, cuts
from verity.core.protocols.sampled_proofs.costfit import fit
~~~

- `service`: the types `Spec`, `Value`, `Public`, `Check` and `Outcome`, and the selection rules `All`, `Law` and
  `Sequential`.
  - `check_public` is the public-items rule on a registration, and `check_verdicts` what an outcome holds the verdicts
    to before the audit record (the Python; a `Stream` runs each by the verifier, `verifier.check_public` and
    `verifier.check_verdicts`).
  - `Stream(spec, unit, budget=, delta=, challenger=, rounds=1)` has the calls `register`, `select` (with a work
    sampling design's `closure=`), `session`, `deliver` and `outcome`, which
    raise `Refused` (with its `Verdict`) or `Pending`; `halted` is its halt's cause once its fleet halted it. Its `unit`
    is one subcircuit, or a window's several in order (a TP rank's served window): the check's `program` is then their
    digests and its query `Q_template_instances`, `register(counts=)` gives the challenger's own instances of each,
    and the window is drawn once by `stratified:K`, a stratum per subcircuit.
  - `Fleet()` has `attach(stream, operator)`, before the stream's first registration; `halted` maps each halted
    operator to its halt's cause.
  - `selection_law` is the sampling design a rule names over N proof blocks, and `select_draw` the draw object from the
    challenger's coins (or a sampler) under the expanded design, with a work draw's closure.
- `registration`: `record` builds a `verity/registration/v1` record (v2 with registered values), `check` holds one to
  the verifier's copies (R1 to R7), `matches_public` to the verifier's public file, `receipt` logs it, `digest` names it, and `window_domain` is a
  registered root's commitment domain (`verity.ml.commitments.window`'s, which an integration commits under).
- `draw`: `parse_law` reads a sampling design's text, `derive` expands a rule over the verifier's program and query (and
  a work table), `check_draw` holds a draw to the registration and the verifier's sampling design, and `check_proved`
  holds the proved proof blocks to the draw; `stratum_k`, `work_k`, `work_table` and `proved` are the pieces they use.
- `partition`: `partition`, `digest`, `units` and `strata` over core's `verity/partition/v1` with the template queries,
  and the population programs `population_program` and `mixed_population_program` (`verity.ml.circuits.population`'s,
  which an integration builds its window's program with).
- `registered`: registered values: `Registered.commit` (the prover's rows, salts and tree), `domain`,
  `registered_entries` and `reads` (the verifier's leaf counts, domains and read map, from its own program), `reads_file`
  (the Lean verifier's `--registered`), `verify_read`.
- `served`: `register_served`, the one form in which a served path registers its served trees (`ServedTree`s) as ports
  of the verifier's own `--registered` file: the record's `registered` entries, R7's copies and the file.
- `audit`: `audit_record`, and the profile in its forms: `profile`, `wrong_units`, `integrity_profile`; `core_law` is a
  sampling design as core's `Stratified`; `session_registered` reads the registered roots off the verdicts of record.
- `verdict`: `Verdict`, what every check returns: `ok`, the failure codes in the order found, and a detail per code
  (and `verifiers`: both verifiers' outcomes).
- `hello`: `check_run`, the run's Hello and its one coin tree over `run_streams`.
- `verifier`: the verifier: `check`, `matches_public`, `check_draw`, `check_proved`, `hello`, `audit_record`,
  `check_public`, `check_verdicts` (given the service's Python verdict), `stream` and `fleet`, by the Lean and the
  Python; `RECORD_VERDICT` names whose is the record's.
- `proofs`: the proof service's own run of C-Flock's Lean verifier on a statement's proof files (`Proof`:
  circuit file, public file, sessions): `verify` refuses a circuit file other than the one the auditor committed, runs
  `flock-verify verify --zk --registered` against the verifier's own program, partition and `--registered` file, and
  returns each session's `Finding` (accepted, why, the program its verdict names, its registered finding, the proof blocks it
  proved), so a protocol reads verdicts from that run, never from a record the prover hands in.
- `parties`: `Challenger`, `Firewall`, `Workers` and `Verifier`, `MANIFEST`, `PROTOCOL` (`PROTOCOLS` by id),
  `params_of`, and `IDS`, the configuration's component names and their protocol ids.
- `consumers`: `exfiltration_bound`, `compute_bound` and `location_bits`, over any `IntegrityProfile`.
- `coins`: `OwnRandomness`, the challenger's own randomness; `draw_first`, a one-stage sampling design's draw from it,
  with `closure_of` for a work design's closure; `ChallengerCoins`, a draw's coins from one family of the auditor's
  challenger, and `challenger_stream`, `read_reason` and `coins_reason`, the verifier's replay of them from the record.
- `plan`: `VerificationPlan` of `GateClass`es with its `UnconditionalObligation`s, `Obligation.of_call`, and
  `weight_binding_plan`.
- `optimize`: the untrusted planner. From `Candidate`s, a `CostModel` (per statement, per proof block, commitment, serve-time
  `Commitment`s, replay) and an `Objective` (harm per proof block at δ), `frontier` is the certified Pareto frontier of total
  cost against harm. `for_budget` and `for_target` query it, `weigh` weights weight classes, `overlaps` prices repeated
  commitments, `certify` recomputes a plan's `Certificate`, `sensitivity` and `needs_benchmark` name what moves it,
  and `work_law` registers it as `work:1`.
- `candidates`: `cuts` derives the `Candidate`s from a Definition DAG of `Shape`s by a DP, each proof block's outputs
  in a public size class (`verity.ml.circuits.partition_v2.unit_class`); `pressure` gives the developer hints.
- `costfit`: C-Flock's cost model fitted from measurements, as a cost-model/v1 record.

## Not yet in Lean

The rules below are normative, and no Lean states them yet: the Python is their reference, pinned by the tests and vectors
named with each. A rule leaves this list in the change whose Lean states it.

### The registration record

`verity/registration/v1` (`one_stage.registration`) is what a window is committed to, fixed before the verifier draws:
the program, query and partition digests, the population, the sampling design expanded, the scheme, the roots with their
leaf counts and domains, the leaf layer, the anchors, the window (provenance only: `served` or `stand-in`) and
`previous`. Its digest is `SHA-512("verity/registration/v1\0" ‖ canon(record))`, canon being canonical JSON (RFC 8785,
`verity.ml.circuits.codec.canonical_json`). The verifier checks R1 to R6 (`check`) against its own copies, writes a
receipt with its own clock, and only then draws, and it holds its own public file to the record (`matches_public`),
which the Lean guarantee `FailClosed.flock_verify_public_ties` ties to the proved rows. The format, each check, the
receipt and the window domains (`window_domain`, its tag's bytes still `served-domain`) are in `one_stage.registration`'s
docstrings.

Pinned by `tests/test_one_stage.py` (the registration, receipt, window, served-domain and R1 tests), and the served
domain also by `integrations/vllm/tests/commit/test_serving_rows.py`.

### Registered values

A program's weights, constants and tables are values: committed once, at registration, under hiding leaves, and read
only inside a proof (`one_stage.registered`).

- A registered value is a root parameter of the program, as rows of `words` leaves in the parameter's leaf order, each
  leaf `word_bits` bits: its leaf width rounded up to whole u16 words, 16 or 32 (`width`). Row `r` is committed as the
  circuit statement commits that port's row: `hm96-sha512/row/v1` over `sha512/row/v1` of its u16 words (`row_words`:
  each leaf `word_bits / 16` words, low first, zero-padded to whole 64-word SHA-512 chunks, as a C-Flock port row and
  its `values` row are laid out; u16 LE, role x), with a 192-byte salt drawn once from the operating system and never
  used for another row. Its root is the frame-v3-sha512 root over the rows under `CommitmentDomain(B, -2,
  RangeIndexedDomain(0, rows), hash="sha512")`, `B` the bytes of `identity_digest("verity/one-stage/registered-domain/v0",
  {"program", "value", "schema", "words", "leaves"})`, plus `"word_bits"` when it is not 16. The prover keeps the rows
  and salts. A value a bit-row port reads is committed in that port's format (`Registered.commit`'s `schema`):
  `hm96-sha512/row/v2` over `sha512/row/v2`, or `hm96-sha512/row-seg/v1` over `sha512/row-seg/v1`, of its leaves' bit
  row (`row_words` unpadded), each leaf framed under that schema, under the same domain.
- `verity/registration/v2` is v1 with `"registered": [{"value", "root", "leaves", "words", "domain"}, ...]` (plus
  `"word_bits"` when it is not 16), digest `SHA-512("verity/registration/v2\0" ‖ canon(record))`. A record is v2 exactly
  when it registers values, so v1's bytes are unchanged. R7 (`check`, with the verifier's own `registered = {parameter:
  words}`): the values are exactly the verifier's, each with a SHA-512 hex root and the verifier's integer leaf count,
  words, width (`R7-word_bits`, which also refuses a `"word_bits": 16` written out) and domain (`registered_entries`,
  from its own program), and a value an earlier record of the log registered under the domain it takes now keeps that
  record's root (`held_roots`, keyed by value and domain): the log is the only place the verifier holds roots, so a
  value is committed once a domain whatever the caller passes. 16 is the default, so a 16-bit value's domain, entry and
  root are what they were before `word_bits`.
- Per-window values (`check(..., window_root=, held=)`, `register_served`'s too): a window-independent Program
  registers each window's values under the same names. The verifier takes the window's open root R from the
  challenger's signed open of that window (never from the record), and every registered value's domain binding adds
  `"window": R` (`registered.domain`, Lean `Registered.domainAt`) but the ones it holds across windows: `held` (the
  weights), and its pinned, supplied and output values. So an honest second window's values are held to nothing the
  first registered, and a root of an earlier window opens under no domain R7 takes (`R7-domain` when the entry names
  the earlier domain; under the new domain no registered read of it opens). A verifier that names no R takes every
  domain as before, and fails closed: it refuses its own inputs (ValueError, Lean `OneStage.paramsOfJson`, naming
  `window_root` and `held`) when a value it doesn't hold across windows was registered by an earlier record under the
  domain it takes now (`registration.unwindowed`), so a fixed-name Program's second window needs R, while a Program
  whose per-window values take fresh names (the window-prefixed form) needs none.
- Supplied values (`check(..., supplied=)`, `supplied_root`, the Lean's `Supplied`): a registered value the verifier fills
  itself, so the prover never chooses it. Each row is the bytes of its sources: `{"port": P}` (the record's `roots` entry
  of P), `{"value": V}` (a registered entry's root, never a supplied value's), `{"bytes": hex}`, `{"position": "K"}` (K
  below the record's population, after the log's earlier records' populations, `log_base`, as a big-endian u64, so no
  two records of one log share a position) and `{"window": coin}` (`window_salt`: SHA-512 of the coin bound to the
  verifier's binding, its stream and its prover, and to the record's window; a salt per prover, stream and window since
  R6 takes one record a window, and refused without a binding). The rows are committed under public salts
  (`supplied_salts`), and R7 refuses an entry whose root is not the one it recomputes (`R7-supplied`). PoUW's seed
  inputs (`pouw.roots`, `pouw.salt`, `pouw.positions`) are supplied values
  (`verity/core/protocols/pouw/tests/test_pouw_supplied_inputs.py`).
- Pinned roots (`check(..., pinned={value: root})`): roots of registered values the verifier doesn't fill, held before
  the record (a model's weights recorded before the salt). R7 holds such a value to its pin as to a root the log holds.
- Binding (`check(..., binding={"stream", "prover"})`, lowercase hex: the verifier's stream digest and the identity it
  registered the prover under): R6 refuses a record whose window doesn't name both (`R6-stream`, `R6-prover`), and the
  window salt binds them, so each (prover, stream, window, position) gets its own seed and a tile proved for one stream
  is refused in another, in the same window or not.
- Reads at a public position: the verifier derives every proof block's reads from its own program (`reads`). A proof block's subcircuit
  parameter whose leaves are exactly row `r` of a registered value reads row `r`; a parameter that holds part of a row,
  or reads the value in one proof block and not in another of its subcircuit, is refused. Data-dependent reads are not public
  reads and are not covered here.
- A statement that reads registered rows is a shared-row file whose registered port's table rows carry the
  registration's salts (`Registered.statement`, `table` for each instance's position), so its `b ‖ c` are the
  registered ones, with header `registered: {port: {"value", "positions", "paths"}}`: per table row, its position and
  its frame-v3 path to the registered root (siblings leaf level first, 64 bytes each, hex). The Lean verifier, given the
  verifier's own `--registered` file (`reads_file`: per port, the value, leaf count, row width (`words`, `word_bits`),
  domain and every proof block's position from its own program and query, and the root of the registration it accepted),
  checks each port in its own format (`row_schema`, absent for `hm96-sha512/row/v1`): that the circuit port's rows are
  the registered rows' u16 words, or a bit row's exactly their bits, every table row's path and every instance's
  position, and refuses a statement that names no `registered` entry for such a port or names one it was not given,
  and one whose proof blocks it did not derive (`verity/ml/flock/verifier/PROTOCOL.md` §16.10).
- Served trees (`served.register_served`): a served root is a hiding commitment that binds nothing on its own, so every
  served path registers each tree a proof block reads as a registered value (`ServedTree`: value, root, leaves, words,
  `word_bits`, domain, its row format `row_schema`, for an earlier run's output its `(record digest, port)`, and for an
  earlier record's registered value it reuses, with no tree of its own, its `(record digest, value)`), writes the
  verifier's `--registered` file from it, and verifies with `flock-verify --zk --registered`: the premise
  `RowsSession.ServedRegistered` of `Flock.Guarantees.ServedRowsLinked`. The positions are the path's own public Program
  and `Q_template_instance` query, which the session's `--program` and `--partition` (the returned `partition`) carry;
  rows are fixed slots padded with zeros. A tree's domain is its value's registered-value domain or, for an output or a
  reused value, the earlier port's or entry's (R7 holds the record to it: `registration.check(..., outputs=)`), and `register_served` refuses any other, a value twice, a root that is not 64 bytes of lowercase hex,
  no rows, a row format other than `row/v1` (`row/v2` and `row-seg/v1` are named in `served.SCHEMAS` and refused until
  R7 and `Flock.Registered.checkPort` read them), a root the log registered otherwise (R7-held) and a tree no proof block
  reads. `benchmarks/one_stage/a0.py`'s `hold_registered` is the template.
- The audit record holds `session_registered`, the roots the verifier opened the reads against, to the
  registered roots (check `registered`), and `integrity_profile.inputs.registered` lists them. `session_registered`
  (`audit.session_registered(verdicts, registration)`) reads them off the accepted verdicts of record (the Lean
  verdict's `registered`), never off the registration, for the registration's own values (its roots' ports and
  registered values);
  `verifier.audit_record` reads them so on both sides.

Pinned by `tests/test_registered.py` and `tests/test_served.py` and, for the Lean verifier's check,
`verity/ml/flock/tests/test_lean_registered_reads.py`.

### The partition

Core's `verity/partition/v1` (`verity.ml.circuits.partition_object`, applied by `one_stage.partition`) is
`{"format": "verity/partition/v1", "program": <SHA-512 of the program>, "query": {"name", "version", "params"}}`, with
digest `SHA-512("verity/partition/v1\0" ‖ canon(object))`.

- The proof blocks are core's evaluation of the query, which refuses a malformed query and a program that is not topological.
  The verifier applies the query to its own copy of the program, and never takes a population or a proof block index on the
  prover's word.
- The one-stage audit applies `Q_template_instance` v0 and `Q_template_instances` v0, and `Q_word` v1 at X = 16 and
  `Q_call` v1 at X <= 16. Under a template query each instance of a listed template in the program's root body is one
  proof block, in canonical order (node, then member). Every proof block's committed bits fit a public size class, a power of two
  from 16 to `UNIT_CLASS_MAX` bits (`unit_class`), and a program with a wider proof block is refused. The
  template queries are demo queries for template runs, not the serving partition. Only they define strata; any other
  query refuses the stratified and work sampling designs.
- A record made before the canonical envelope names `{"program", "query"}`, and keeps that object's digest under the same
  rule (`partition.digest`).

The Lean verifier's executable reads the same object (`Flock/HmRow.lean`, `Flock/TemplateQuery.lean`), but no theorem
states it. Pinned by core's `verity/ml/circuits/tests/partition_vectors.json` and `template_instance_vectors.json`,
which `tests/test_one_stage.py` replays (`test_the_partition_is_cores_on_the_pinned_q_word_vector`,
`test_the_partition_is_cores_on_the_pinned_template_vectors`, `test_the_partition_refuses_what_core_refuses`,
`test_a_malformed_query_has_no_units_and_no_digest`, `test_a_program_that_is_not_topological_has_no_units`,
`test_a_pre_canonical_object_keeps_its_digest`).

### Sampling designs as text and as registered

Code and records call a sampling design a `law`; the name stays until a format version renames it.

- A sampling design's text is `subset:K` or `bernoulli:NUM/DEN` with `NUM ≤ DEN` and `DEN > 0` (`flock-circuit serve
  --draw`'s syntax), or one of the rules `stratified:K` and `work:K` with `K ≥ 1` (`draw.parse_law`), each number in ASCII digits
  (`draw.digits`, as the Lean reads them; a draw's `p` too).
- As objects they are `{"law": "subset", "k": K}`, `{"law": "bernoulli", "p": "NUM/DEN"}`, and the rules expanded by
  `draw.derive`: `{"law": "stratified", "k": K, "strata": [{"template", "units", "k"}, ...]}` and
  `{"law": "work", "k": K, "strata": [{"template", "units", "work", "floor", "k"}, ...]}`. The registration carries the
  sampling design expanded.
- `derive` refuses a `K` outside `[1, N]`, a work sampling design without the verifier's work table, a subcircuit the
  table does not name, and a population that does no work.
- Two sampling designs are the same when their canonical JSON bytes are (`draw.same_law`), so `true` is not `1`, as the
  Lean verifier reads them.
- `check_draw` also holds the draw's population to the registration's.

Pinned by `tests/test_one_stage.py` (`test_draw_checks`, `test_derive_expands_the_rule_over_the_verifiers_program`,
`test_derive_expands_the_work_rule_over_the_verifiers_table`, `test_stratified_draw_checks`,
`test_work_draw_checks_and_closures`).

### What a proof block's work may count

The work sampling design's bound on acceptance probability holds over the work the table credits; what the table may
credit is a rule of its own, documented with `draw.work_table`:

- A proof block's work is counted on the rows the Program declares, never on rows a kernel pads it with.
- Filler is credited nothing: a proof block of filler only, or one the prover declares uncredited before the draw, does work 0.
  Its floor still draws it for integrity, and missing it costs no work. A proof block mixing real and filler rows is credited
  its real cells only.
- The work sampling design weights by subcircuit, so every proof block of a subcircuit is credited the same work. A subcircuit whose
  proof blocks are padded differently is split before the sampling design is derived, or credited 0: the bound needs every
  proof block's inclusion probability to be at least `K w_u / W` over the credited `W`, which a per-subcircuit work that
  over-credits a proof block breaks.

`test_derive_expands_the_work_rule_over_the_verifiers_table` covers a subcircuit credited nothing.

### The audit record and the profile

The one-stage profile (`audit.profile`) is core's `IntegrityProfile` of one level `unit` (`Level("unit", <partition
digest>)`) with `sizes =
(population,)`, drawn by the sampling design (`Subset(k)`, `Bernoulli(p)`, or `Stratified` with one stratum per template), with `delta`, the program's digest and the count drawn. It bounds faulty proof blocks and weights nothing. The work
sampling design's profile is `Stratified` over the same strata at the work sizes, counting the drawn proof blocks only: a
closure proves more, which only helps.

The audit record (`audit.audit_record`) is `{"format": "verity/one-stage/audit/v0", "registration", "registration_digest",
"receipt", "draw", "law", "proved_units", "statement", "checks", "verdicts", "accepted", "complete", "profile",
"integrity_profile", "wrong_units"}`:

- `checks` holds `draw` (`check_draw`) and `proved` (`check_proved`), and a failed `receipt` (it must name the
  registration's digest), `served` (each session was served exactly the draw) or `roots` (each session's root is the
  registered one).
- A verdict is `{"accepted": true | false | null, ...}`; null means that verifier did not run, which leaves the record
  incomplete (`complete` is false).
- The record is accepted only when there is a verifier whose verdict counts (`of_record`), every one of them ran and accepted, every
  other verdict that ran accepted, and every check passed. The profiles are present only then.
- A session that proves one subcircuit's proof blocks is served that subcircuit's share as a `subset:k_s` draw over its `n_s` proof blocks,
  in its own indices; the record keeps the union draw.
- `profile` is the profile's JSON, except that a stratified or work sampling design keeps the v0 form: its JSON with
  `delta` and the count drawn per subcircuit.
- `wrong_units` is `{"delta", "of": population, "bound", "rule", ["per_template"]}`: `bound` is core's exact
  `Stratified.count_bound` at delta for `subset` (one stratum), `stratified` and `work`, and `Bernoulli.count_bound` for
  `bernoulli`; a stratified or work sampling design adds each subcircuit's own bound.
- `integrity_profile` is `verity/integrity-profile/v1`: `{"format", "protocol", "program", "query", "partition",
  "population", "registration": <digest>, "scheme", "roots": [{"port", "root", "leaves"}], "law", "drawn",
  "inputs": {"anchored": <anchors>, "linked": []}, "budget_log2_t": null, "terms": null, "epsilon_log2": null,
  "evidence": {"receipt": <SHA-512 of canon(receipt)>, "draw": <SHA-512 of canon(draw)>, "sessions", "verdict"}}`,
  `sessions` being the session records' SHA-512 and `verdict` the verifier's digest (a list when the drawn
  proof blocks were proved in several sessions). Its ε terms come from the soundness analysis, not from the run, so the run leaves them null.

Pinned by `tests/test_one_stage.py` (`test_profile_counts_wrong_units`, `test_the_record_states_the_one_profile_in_its_recorded_bytes`,
`test_audit_record_needs_every_check_and_verdict`, `test_audit_record_holds_the_served_draw_and_the_sessions_roots`,
`test_the_wrong_unit_bound_matches_worst_case_and_scales`, `test_audit_record_under_the_stratified_law`,
`test_the_subset_bound_is_now_the_exact_count`, `test_a4_under_the_stratified_rule`, `test_audit_record_under_the_work_law`).

### The consumers

`consumers` reads any `IntegrityProfile`, whichever protocol produced it, at its
delta, through its weighted bound `harm_bound`; each application counts delta once.

- `exfiltration_bound(profile, output_bits)` is `harm_bound(output_bits) + location_bits(profile)`, in bits.
  `location_bits` is the location term: with `K = ⌊bound()⌋` of `n` proof blocks, 0 when `K ≤ 0` or `n = 0`, `n` when
  `2K ≥ n`, and otherwise `n H₂(K/n) (1 + MARGIN)` rounded up to the next float, at most `n`. The Lean states the bound
  (`audit_exfiltration`, `card_admissible_le`), and the tests hold the Python to be at least it.
- `compute_bound(profile, work)` is `harm_bound(work)`. Policies and PoUW pass their own utility to `worst_case`.
- A proof block's output bits or work is one number for every proof block, or per stratum a number or `(bits, units)` pairs.

Pinned by `tests/test_consumers.py` and `tests/fixtures/exfiltration_lean.json`, whose recorded sha256 of the Lean files it came from fails `test_the_lean_vectors_were_generated_from_this_tree` until it is regenerated.

### The proof service

`service.Stream` holds one protocol's registrations to its `Spec`, on the challenger's side:

- **The public-items rule** (`check_public`): each observable item of a registration record is the service's own, a
  declared Value's one salted root, or an item the Spec declares in `public` with its kind and reason. It runs before
  one-stage's R1 to R7.
- **The draw.** `register` logs the record and its receipt, and `select` draws once from the challenger's coins as the
  Lean verifier's `Flock.Draw.drawWith` does, with no derivation, held to `draw.check_draw`.
- **Sessions over several rounds** (`Stream(rounds=R)`, `Stream.session`), which C-Flock's verifier refuses
  (`verity/ml/flock/verifier/PROTOCOL.md` §16.12).
- **The salt and the registration deadline**: one salt an epoch, and a late registration ends the epoch. This is PoUW's
  `Pouw.AcceptsBy` (`Security/Definitions/Pouw/Game/Deadline.lean`).
- **The stream's rules** (budget, open registrations, salt, deadlines, once-only steps), decided op by op by the verifier
  of record (`verifier.stream`: `SampledProofs/Stream.lean` beside `service._step`) on the counts and entries the op
  reads. A disagreement stops the stream.
- **The fleet's halt (R5-all)** (`service.Fleet`): at the failure that brings any stream of an operator to its budget,
  every stream of the operator halts, each outcome not yet final ends as `halted`, never `accept`, and nothing of the
  operator registers again. Decided by the verifier (`verifier.fleet`: `SampledProofs/Fleet.lean` beside
  `service._fleet_step`, and `Stream.lean`'s `halt`), whose properties are `Security/Properties/SampledProofs/Halt.lean`.
- **No retries**, **the failure budget**, **the log's history** (a check reads a few records, with the whole log's
  verdict), **the records** (`verity/proof-service/stream/v0` and `outcome/v0`) and what a stream refuses until it is
  built.

Each rule, its formats and its refusal codes are in `service`'s module docstring. Pinned by `tests/test_service.py`, the
log's history by `tests/test_service_history.py`, sessions by `tests/test_service_sessions.py`, the fleet's halt by
`tests/test_service_fleet.py`, and the stream's and the fleet's rules by the difftest's `stream` and `fleet` pieces.

### The verifier's randomness

The draw of record is `coins.OwnRandomness` on the Lean verifier's coins (`one_stage.verifier.coins`:
`IO.getRandomBytes`, under `UniformRandomBytes`; `service.Stream`'s default), read with the Lean verifier's samplers (`Flock.Draw`), with no
hash, no derivation and no beacon, after the registration it follows. `draw_first` reads the bytes as
`Flock.Draw.drawWith` does. Nothing derives a draw, the proof service's included, except vLLM's legacy replay challenge
until its re-baseline; `coins`' docstring has the byte layout. Pinned by `tests/test_coins.py`.

A draw from the auditor's challenger (`remote.RemoteChallenger`, or `challenger.LiveChallenger` in process) passes
`coins.ChallengerCoins` as the stream's `coins`. `commit(after)` takes the prover's commitment (the registration's
digest after its receipt, or the run's root) and fixes the family and its next index. Each read then opens
ceil(missing / 64) more of the family's coins after it in one round trip, checks each against the challenger's
commitment, and concatenates them, with no stretch. A rerun's extension opens only new indices, and a commitment is
drawn once. The record is each `Segment` (family, first index, `after` and the openings). The verifier replays it with
`read_reason` against the roots the challenger's signed release log shows (`remote.trusted_session`): the draw's coins
are exactly those released after the commitment, every one read. A stream over `ChallengerCoins` has `coin_source`
`"challenger"` (its digest says so, and it takes no sampler); any other is `"verifier"`. Every outcome's run by the Lean
verifier (`Run.run`, `SampledProofs.ChallengerCoins`) takes the `coin_source`. From the challenger, the check
is mandatory: `Stream.outcome(coins=...)` hands the evidence (`coins_reason`'s `{key, ledger, main, pins, segment,
session}`), and the run takes the commitment and the roots from the session the auditor's ledger opens for the
registration's window (`register`'s `window_root`, never a window the evidence names), run from a commit the host's
pins log records on the verifier's copy of main's first-parent line (`pinned_session`), and refuses the select step (code `coins`) unless the
bytes the draw read are those coins, after the registration's digest. Missing evidence and a registration without its
window root are refused the same way, as is evidence for a draw from the verifier's own coins: the outcome is a reject
failing `run`. Pinned by `tests/test_challenger_coins.py` and the difftest's `challenger_read`,
`challenger_draw_coins` and `run`.

### Verification plans

`plan` builds obligations from a call's declared interface (`args`, `returns`), never from `FunctionCall.out`, which
rescans the circuit on every call. A prescribed input is a sampled gate like any other, and an `UnconditionalObligation`
is proved on every audit, outside the draw, so its gates are the profile's `covered` families. Pinned by `tests/test_plan.py`.
