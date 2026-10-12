# Sampled proofs (`verity-sampled-proofs`)

Sampled proofs spot-check a committed computation. While serving, the developer commits every value that crosses a
proof-unit boundary of a partition of the program and registers the commitment roots. The verifier then draws proof units
from its own randomness, the developer proves the drawn units, and the verifier checks the proofs. It never runs any part
of the computation. An accepted audit yields an integrity profile (`verity.core.protocols.profile.IntegrityProfile`): except with
probability δ, at most so many units are faulty, a unit being faulty when its committed outputs differ from its gates
applied to its committed inputs. The protocol is stated in Lean (see "Where things are"); this README outlines it,
points to the code, and lists what the Lean does not state yet.

## Outline

**The one-stage audit** is the audit protocol in use. One round goes like this:

1. Serving commits the values that the partition's units read and write, and sends the roots.
2. The developer registers a record that fixes everything a drawn unit's statement can depend on: the program, the query
   that partitions it, the sampling design, the commitment scheme and the roots. The verifier checks the record against
   its own copies and logs a receipt before it draws.
3. The verifier draws units from its operating system's randomness under the agreed sampling design and sends the draw
   in the clear, before the session's coin commitment. Nothing is derived from a commitment, and there is no beacon.
4. The drawn units are proved in one session, whose statement is the drawn units'.
5. Each verifier that runs checks the proofs. The Lean verifier is the verifier of record.
6. The audit record keeps the registration, the receipt, the draw, the proved units, the statement's digests and every
   verdict, and, when everything accepted, the integrity profile.

**The proof service** (`service`) runs these steps for a protocol, on the challenger's side. A protocol states a `Spec`:
- what it commits;
- its selection rule;
- the check, as a public Program;
- its public items, each with its reason.

Everything the auditor observes must be a function of those items. A `Stream` per `Spec` holds each registration to that
rule and then to one-stage's checks, draws once, binds the delivered proof to its outcome, and counts failures against a
public budget. The design is `note:proofs/20261006T0307Z-draft-proof-service-architecture`.

**The sampling designs.** `subset:K` is a uniform `K`-subset of the `N` units, and `bernoulli:NUM/DEN` takes each unit
independently with probability `NUM/DEN`.
The default is `stratified:K` (Daniel, Sep 27): one stratum per template, and inside each an independent uniform draw
without replacement of `k_s = max(1, ⌊(2 K n_s + N) / (2N)⌋)` units, that is `K n_s / N` rounded half up and never fewer
than one, so every template is exercised in every audit. `work:K` (Daniel, Sep 28) keeps those strata and that sampler
but sizes each stratum by the work its units are credited: with `W = Σ_s w_s n_s`, stratum `s` draws
`k_s = min(n_s, max(f_s, ⌈K w_s n_s / W⌉))`, where `w_s` and the floor `f_s ≥ 1` come from the verifier's own work table.
A work draw may add a closure, units proved with the drawn ones. Each verifier derives the strata from its own program and
query, never from the prover.

**What an accepted audit establishes.** Under a stratified draw, a set of faulty units has acceptance probability
`Π_s C(n_s − m_s, k_s) / C(n_s, k_s)` (the probability that the draw misses all of them), `m_s` being its units in
stratum `s`, so a template that is faulty in every unit is never missed. Under the work sampling design, faulty units
doing a fraction `ε` of the credited work have acceptance probability at most `(1 − ε)^K`; at `K = 27,713`, a 0.1% share
has at most `2^-40`. The per-unit proof system adds its knowledge-soundness and binding errors (ε_ks and ε_bind).
Applications weight the profile's faulty units by what each one costs them: exfiltration security by their output bits,
plus a term for which units are faulty, and compute security by their work.

**Two-stage sampling** is the alternative for a prover that replays part of the computation to build its witness. The
prover commits the boundary of its replay units (RUs) and registers the root. The verifier draws RUs, per RU class either
each with probability `p` or under any one-stage sampling design over the class's RUs (`subset`, `bernoulli`,
`stratified`, or `work` with the verifier's closure); the prover replays the drawn RUs and commits and registers their
interiors, and the verifier then checks `k` of each drawn RU's proof units, or all of them when it holds fewer. Both
draws come from the verifier's own randomness, each after the registration it follows, and each checked proof unit is
established on its authenticated boundary values. Since the interiors are committed after the first draw, a faulty RU
has acceptance probability at most `1 − p k / n_v` under a Bernoulli first stage, independently across RUs, so an
accepted audit's profile, one per RU class, is over the RUs at that rate, at the class's largest RU. A subset,
stratified or work first stage checks every proof unit of a drawn RU, so there is no second draw, and the profile is
that sampling design's over the RUs.

The proof service runs the two-stage audit whose first stage checks every proof unit of a drawn RU (`service.TwoStage`
with `check="all"`). Not built yet: a second draw (a count `k`), the security function and its red-team review
(decision 41), and composition checks such as PoUW's tiles per proof unit matching this partition (the IR checks
refinement, `verity.ml.circuits.partition.validate_refinement`).
Serving's record is not yet the scheme the audit proves: run A0 registers a stand-in, a fresh commitment to values
captured from serving, and says so (`window.kind = "stand-in"`). Run A5 (`benchmarks/one_stage/a5.py`) registers a served
Commit's own window, its serving rows written from the sealed replay bundle opened against the run's record
(`verity-vllm serving-rows`); the link from those roots to the record's vllm-v1 roots is the committer's opening, not proof.

## Where things are

- **The Lean.** Today it sits in C-Flock's soundness package, `Security/Proofs/Flock/Soundness/Audit/`,
  whose `README.md` lists every theorem: the one-stage audit (`OneStage`), the sampling designs (`Law`, `Stratified`,
  `StratifiedMiss`, `Work`, `Closure`, `Window`), the registration-dependent draw (`RegDraw`), two stages (`TwoStage`,
  with the tight and insecure cases in `Examples`), exfiltration (`Influence`, `Harm`), the compiled layer
  (`Extraction`) and composition (`Circuit`); `Audit/Flock*.lean` instantiate them for C-Flock. The verifier's coins are
  assumption A3 (`Assumptions.UniformRandomBytes`).
- **The executable draw** is the Lean verifier's `verity/core/service/flock/Flock/Draw.lean` (`flock-verify draw`): the
  samplers, the strata and sizes, the draw object and its checks U1 to U3, specified in
  `verity/ml/flock/verifier/PROTOCOL.md` §7.3. Soundness's `ExecDraw.lean`, `ExecStratified.lean` and `ExecDrawOS.lean`
  prove that its draws accept faulty units no more often than the modelled sampling designs do, and
  `Audit/FlockWork.lean` that its sizes are the model's.
- **The verifier of record.** `one_stage.verifier` runs the Lean port of the one-stage audit (the Lake package
  `verity/core/service/sampled_proofs`, its `sampled-proofs-verifier`) and the Python beside it on every check: R1-R7
  (`check`), `matches_public`, `check_draw`, `check_proved`, the whole audit record (`audit_record`), and the service's
  public-items rule (`check_public`) and hold on an outcome's verdicts (`check_verdicts`). Its one line
  `RECORD_VERDICT = "lean"` names whose verdict is the record's (`"python"` switches back); both outcomes are recorded
  beside it (`verifiers`), and where they disagree the check is refused naming both. The service's `Stream` and
  `benchmarks/one_stage` call it. The two are held to each other by
  `verity/core/service/sampled_proofs/tests/test_difftest.py`.
- **The Python.** `verity.core.protocols.sampled_proofs.one_stage` is the one-stage audit's reference: `registration`,
  `draw` (the draw's checks, not a sampler), `partition`, `audit` and `verdict`. `consumers` reads its profile. `service` is the proof service over it, whose `Stream`
  draws from the challenger's own randomness as `two_stage.draw_first` does. `two_stage` is two-stage sampling and the
  verifier's randomness, and
  `verity.core.protocols.sampled_proofs.plan` holds verification plans over IR Programs.
- **The Lean-generated vectors.** `verity/core/protocols/sampled_proofs/tests/fixtures/exfiltration_lean.json` is `audit_exfiltration` and
  `stratified_miss_eq_greedy` evaluated by Lean (`verity/core/protocols/sampled_proofs/tests/lean/ExfiltrationVectors.lean`, regenerated by
  `verity/core/protocols/sampled_proofs/tests/lean/generate.sh`), and `verity/core/protocols/sampled_proofs/tests/test_consumers.py` holds the consumers to it. `tests/test_two_stage.py` checks the
  samplers and stage 1's draw under a one-stage sampling design against the Lean verifier's (`drawOn`) on raw bytes, and
  `tests/test_one_stage.py` the work sizes against the ones C-Flock's
  tests pin. `verity/ml/flock/tests/stratified_agree.py` runs the Lean verifier's draw against `one_stage.draw`.
- **The benchmark.** `benchmarks/one_stage` runs the one-stage audit end to end with C-Flock's circuit prover and the Lean
  verifier.
- **The users.** C-Flock draws and checks the draw in its Lean verifier and proves the drawn units. The vLLM integration
  commits serving rows with the registration and served domains of `one_stage` (`verity_vllm.commit.serving_rows`, in the
  Commit or from a deferred Commit's bundle by `verity_vllm.pipeline.deferred_rows`); its replay challenge is its own
  (`verity_vllm.commit.challenge`).

Tests: `uv run tools/verity/check/suites.py verity-sampled-proofs`.

## Core API

~~~python
from verity.core.protocols.sampled_proofs import service
from verity.core.protocols.sampled_proofs.one_stage import audit, draw, partition, registration
from verity.core.protocols.sampled_proofs.two_stage import OwnRandomness, ReplayUnit, Stage, TwoStageLaw, draw_first
from verity.core.protocols.sampled_proofs.consumers import compute_bound, exfiltration_bound, location_bits
from verity.core.protocols.sampled_proofs.plan import VerificationPlan
~~~

- `service`: the types `Spec`, `Value`, `Public`, `Check` and `Outcome`, and the selection rules `All`, `Law`, `TwoStage`
  and `Sequential`.
  - `check_public` is the public-items rule on a registration, and `check_verdicts` what an outcome holds the verdicts
    to before the audit record (the Python; a `Stream` runs each by the verifier of record, `verifier.check_public` and
    `verifier.check_verdicts`).
  - `Stream(spec, unit, budget=, delta=, challenger=)` has the calls `register`, `select` (with a work sampling design's
    `closure=`), `register_interiors` (under `TwoStage`), `deliver` and `outcome`, which raise `Refused` (with its
    `Verdict`) or `Pending`.
- `registration`: `record` builds a `verity/registration/v1` record (v2 with registered values), `check` holds one to
  the verifier's copies (R1 to R7), `matches_public` to the verifier's public file, `receipt` logs it, `digest` names it, and `served_domain` is a
  served root's commitment domain.
- `draw`: `parse_law` reads a sampling design's text, `derive` expands a rule over the verifier's program and query (and
  a work table), `check_draw` holds a draw to the registration and the verifier's sampling design, and `check_proved`
  holds the proved units to the draw; `stratum_k`, `work_k`, `work_table` and `proved` are the pieces they use.
- `partition`: `partition`, `digest`, `units` and `strata` over core's `verity/partition/v1` with the template queries,
  and the test programs `population_program` and `mixed_population_program`.
- `registered`: registered values: `Registered.commit` (the prover's rows, salts and tree), `domain`,
  `registered_entries` and `reads` (the verifier's leaf counts, domains and read map, from its own program), `reads_file`
  (the Lean verifier's `--registered`), `verify_read`.
- `audit`: `audit_record`, and the profile in its forms: `profile`, `wrong_units`, `integrity_profile`; `core_law` is a
  sampling design as core's `Stratified`; `session_registered` reads the registered roots off the verdicts of record.
- `verdict`: `Verdict`, what every check returns: `ok`, the failure codes in the order found, and a detail per code
  (and `verifiers`, both verifiers' outcomes, from `verifier`).
- `verifier`: the verifier of record, `check`, `matches_public`, `check_draw`, `check_proved`, `audit_record` and
  `check_public` and `check_verdicts` (given the service's Python verdict) by the Lean and the Python, `RECORD_VERDICT`
  naming whose is the record's.
- `consumers`: `exfiltration_bound`, `compute_bound` and `location_bits`, over any `IntegrityProfile`.
- `two_stage`: `TwoStageLaw` of `Stage`s and its `profile`; stage 1's draws `select_replay_units` and `draw_first` (a
  one-stage sampling design's draw object over the RUs, with `closure_of` for a work sampling design's closure), stage
  2's `select_proof_units`, all from `OwnRandomness`; and `check_probability`.
- `plan`: `VerificationPlan` of `GateClass`es with its `UnconditionalObligation`s, `Obligation.of_call`, and
  `weight_binding_plan`.

## Not yet in Lean

The rules below are normative, and no Lean states them yet: the Python is their reference, pinned by the tests and vectors
named with each. A rule leaves this list in the change whose Lean states it.

### The registration record

`verity/registration/v1` (`one_stage.registration`) is what serving is committed to, fixed before the verifier draws:

~~~text
{"format": "verity/registration/v1", "protocol": "verity/sampled-proofs/one-stage/v1",
 "program": "<SHA-512 of the program>", "query": "<SHA-512 of canon(query)>", "partition": "<partition digest>",
 "population": <units the query derives>, "law": <the sampling design's object, expanded>, "scheme": "<leaf and tree scheme>",
 "roots": [{"port": ..., "root": "<hex>", "leaves": ..., "domain": ...}, ...],
 "leaf_layer": "<SHA-512 of the verifier's public file>" | null,
 "anchors": [...], "window": {"kind": "served" | "stand-in", ...provenance}, "previous": null | "<digest>"}
~~~

- The digest is `SHA-512("verity/registration/v1\0" ‖ canon(record))`, canon being canonical JSON (RFC 8785,
  `verity.ml.circuits.codec.canonical_json`).
- Everything a drawn unit's statement can depend on is fixed by the record, or by objects the verifier already holds,
  before the draw. The verifier checks the record against its own copies, writes a receipt with its own clock, and only
  then draws.
- The checks (`check`): the record has exactly these keys, and
  - R1: the format and protocol are these, and the sampling design is the verifier's own, expanded over its program and
    query (`draw.derive`, with its own work table for `work`), as canonical bytes; `window.kind` is `served` or
    `stand-in`;
  - R2: the program, query and partition digests and the population are the verifier's own (an explicit partition
    object must be over the verifier's program);
  - R3: the scheme is the verifier's, and the roots are exactly the ports its layout derives, each exactly
    `{port, root, leaves, domain}` with a hex root and the verifier's leaf count and domain;
  - R4: the leaf layer is the digest of the verifier's copy, so a null one is refused when the verifier holds one;
  - R5: the anchors are the verifier's;
  - R6: `previous` is the digest of this prover's last record (null for the first), and no earlier record has the same
    window.
- The receipt is `{"format": "verity/registration-receipt/v1", "registration": <digest>, "verifier": ..., "received":
  "<UTC, %Y-%m-%dT%H:%M:%SZ>"}`, and the draw names it.
- The verifier holds its own public file (scheme, roots, their domains, its instance count) to the record
  (`matches_public`). That those roots, domains and instance count are the proved rows' is flock-verify's check of the
  public file (`checkPublic`), stated deterministically by the Lean guarantee
  `FailClosed.flock_verify_public_ties`; flock-verify reads no `scheme` from the header, only META's.
- `window` is provenance only: `served` roots are what serving committed, and `stand-in` roots a fresh commitment to
  values captured from serving, whose link to the served root is provenance, not proof.
- A served root's domain (`served_domain`, agreed with vllm-serving-commit) binds the program, the partition, the port,
  its schema, its leaf count and the run: the hex `domain_id` of `CommitmentDomain(B, -1, RangeIndexedDomain(0, leaves),
  hash="sha512")`, `B` being the 64 bytes of `identity_digest_sha512("verity/one-stage/served-domain/v1", {"program",
  "partition", "port", "schema", "leaves", "run"})`.

Pinned by `tests/test_one_stage.py` (`test_registration_v1_passes_r1_to_r6`, `test_registration_refusals`,
`test_receipt_names_the_record`, `test_registration_holds_the_public_file`, `test_the_window_says_served_or_stand_in`,
`test_served_domains_bind_program_partition_port_schema_leaves_and_run`, `test_r1_holds_the_registered_strata_to_the_verifiers_own`,
`test_r1_holds_a_work_law_to_the_verifiers_own_table`, `test_a_record_over_an_explicit_partition_object`), and the served
domain also by `integrations/vllm/tests/commit/test_serving_rows.py`.

### Registered values

A program's weights, constants and tables are values: committed once, at registration, under hiding leaves, and read
only inside a proof (`one_stage.registered`).

- A registered value is a root parameter of the program, as rows of `words` leaves in the parameter's leaf order, each
  leaf `word_bits` bits: its leaf width rounded up to whole u16 words, 16 or 32 (`width`). Row `r` is committed as the
  circuit statement commits that port's row: `hm96-sha512/row/v1` over `sha512/row/v1` of its u16 words (`row_words`:
  each leaf `word_bits / 16` words, low first, zero-padded to whole 64-word SHA-512 blocks, as a C-Flock port row and
  its `values` row are laid out; u16 LE, role x), with a 192-byte salt drawn once from the operating system and never
  used for another row. Its root is the frame-v3-sha512 root over the rows under `CommitmentDomain(B, -2,
  RangeIndexedDomain(0, rows), hash="sha512")`, `B` the bytes of `identity_digest("verity/one-stage/registered-domain/v0",
  {"program", "value", "schema", "words", "leaves"})`, plus `"word_bits"` when it is not 16. The prover keeps the rows
  and salts.
- `verity/registration/v2` is v1 with `"registered": [{"value", "root", "leaves", "words", "domain"}, ...]` (plus
  `"word_bits"` when it is not 16), digest `SHA-512("verity/registration/v2\0" ‖ canon(record))`. A record is v2 exactly
  when it registers values, so v1's bytes are unchanged. R7 (`check`, with the verifier's own `registered = {parameter:
  words}`): the values are exactly the verifier's, each with a SHA-512 hex root and the verifier's integer leaf count,
  words, width (`R7-word_bits`, which also refuses a `"word_bits": 16` written out) and domain (`registered_entries`,
  from its own program), and a value an earlier record of the log registered keeps that record's root (`held_roots`):
  the log is the only place the verifier holds roots, so a value is committed once whatever the caller passes. 16 is the
  default, so a 16-bit value's domain, entry and root are what they were before `word_bits`.
- Reads at a public position: the verifier derives every unit's reads from its own program (`reads`). A unit's template
  parameter whose leaves are exactly row `r` of a registered value reads row `r`; a parameter that holds part of a row,
  or reads the value in one unit and not in another of its template, is refused. Data-dependent reads are not public
  reads and are not covered here.
- A statement that reads registered rows is a shared-row file whose registered port's table rows carry the
  registration's salts (`Registered.statement`, `table` for each instance's position), so its `b ‖ c` are the
  registered ones, with header `registered: {port: {"value", "positions", "paths"}}`: per table row, its position and
  its frame-v3 path to the registered root (siblings leaf level first, 64 bytes each, hex). The Lean verifier, given the
  verifier's own `--registered` file (`reads_file`: per port, the value, leaf count, row width (`words`, `word_bits`),
  domain and every unit's position from its own program and query, and the root of the registration it accepted),
  checks that the circuit port's rows are the registered rows' u16 words, every table row's path and every instance's
  position, and refuses a statement that names no `registered` entry for such a port or names one it was not given,
  and one whose units it did not derive (`verity/ml/flock/verifier/PROTOCOL.md` §16.10).
- The audit record holds `session_registered`, the roots the verifier of record opened the reads against, to the
  registered roots (check `registered`), and `integrity_profile.inputs.registered` lists them. `session_registered`
  (`audit.session_registered(verdicts, registration)`) reads them off the accepted verdicts of record (the Lean
  verdict's `registered`), never off the registration, for the registration's own values (its roots' ports and
  registered values), as the service splits the sessions' roots between a boundary and its interiors;
  `verifier.audit_record` reads them so on both sides.

Pinned by `tests/test_registered.py` and, for the Lean verifier's check, `verity/ml/flock/tests/test_lean_registered_reads.py`.

### The partition

Core's `verity/partition/v1` (`verity.ml.circuits.partition_object`, applied by `one_stage.partition`) is
`{"format": "verity/partition/v1", "program": <SHA-512 of the program>, "query": {"name", "version", "params"}}`, with
digest `SHA-512("verity/partition/v1\0" ‖ canon(object))`.

- The units are core's evaluation of the query, which refuses a malformed query and a program that is not topological.
  The verifier applies the query to its own copy of the program, and never takes a population or a unit index on the
  prover's word.
- The one-stage audit applies only `Q_template_instance` v0 and `Q_template_instances` v0: each instance of a listed
  template in the program's root body is one unit, in canonical order (node, then member), with no width rule. They are
  demo queries for template runs, not the serving partition. Only they define strata; any other query refuses the
  stratified and work sampling designs.
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

- A sampling design's text is `subset:K` or `bernoulli:NUM/DEN` with `NUM ≤ DEN` and `DEN > 0` (M0's `serve --draw`
  syntax), or one of the rules `stratified:K` and `work:K` with `K ≥ 1` (`draw.parse_law`).
- As objects they are `{"law": "subset", "k": K}`, `{"law": "bernoulli", "p": "NUM/DEN"}`, and the rules expanded by
  `draw.derive`: `{"law": "stratified", "k": K, "strata": [{"template", "units", "k"}, ...]}` and
  `{"law": "work", "k": K, "strata": [{"template", "units", "work", "floor", "k"}, ...]}`. The registration carries the
  sampling design expanded.
- `derive` refuses a `K` outside `[1, N]`, a work sampling design without the verifier's work table, a template the
  table does not name, and a population that does no work.
- Two sampling designs are the same when their canonical JSON bytes are (`draw.same_law`), so `true` is not `1`, as the
  Lean verifier reads them.
- `check_draw` also holds the draw's population to the registration's.

Pinned by `tests/test_one_stage.py` (`test_draw_checks`, `test_derive_expands_the_rule_over_the_verifiers_program`,
`test_derive_expands_the_work_rule_over_the_verifiers_table`, `test_stratified_draw_checks`,
`test_work_draw_checks_and_closures`).

### What a unit's work may count

The work sampling design's bound on acceptance probability holds over the work the table credits; what the table may
credit is a rule of its own (the Pearl-C statement review, item 5), documented with `draw.work_table`:

- A unit's work is counted on the rows the Program declares, never on rows a kernel pads it with.
- Filler is credited nothing: a unit of filler only, or one the prover declares uncredited before the draw, does work 0.
  Its floor still draws it for integrity, and missing it costs no work. A unit mixing real and filler rows is credited
  its real cells only.
- The work sampling design weights by template, so every unit of a template is credited the same work. A template whose
  units are padded differently is split before the sampling design is derived, or credited 0: the bound needs every
  unit's inclusion probability to be at least `K w_u / W` over the credited `W`, which a per-template work that
  over-credits a unit breaks.

`test_derive_expands_the_work_rule_over_the_verifiers_table` covers a template credited nothing.

### The audit record and the profile

The one-stage profile (`audit.profile`) is core's `IntegrityProfile` of one level `unit` (`Level("unit", <partition
digest>)`) with `sizes =
(population,)`, drawn by the sampling design (`Subset(k)`, `Bernoulli(p)`, or `Stratified` with one stratum per template), with `delta`, the program's digest and the count drawn. It bounds faulty units and weights nothing. The work
sampling design's profile is `Stratified` over the same strata at the work sizes, counting the drawn units only: a
closure proves more, which only helps.

The audit record (`audit.audit_record`) is `{"format": "verity/one-stage/audit/v0", "registration", "registration_digest",
"receipt", "draw", "law", "proved_units", "statement", "checks", "verdicts", "accepted", "complete", "profile",
"integrity_profile", "wrong_units"}`:

- `checks` holds `draw` (`check_draw`) and `proved` (`check_proved`), and a failed `receipt` (it must name the
  registration's digest), `served` (each session was served exactly the draw) or `roots` (each session's root is the
  registered one).
- A verdict is `{"accepted": true | false | null, ...}`; null means that verifier did not run, which leaves the record
  incomplete (`complete` is false).
- The record is accepted only when there is a verifier of record (`of_record`), every one of them ran and accepted, every
  other verdict that ran accepted, and every check passed. The profiles are present only then.
- A session that proves one template's units is served that template's share as a `subset:k_s` draw over its `n_s` units,
  in its own indices; the record keeps the union draw.
- `profile` is the profile's JSON, except that a stratified or work sampling design keeps the v0 form: its JSON with
  `delta` and the count drawn per template.
- `wrong_units` is `{"delta", "of": population, "bound", "rule", ["per_template"]}`: `bound` is core's exact
  `Stratified.count_bound` at delta for `subset` (one stratum), `stratified` and `work`, and the linear relaxation for
  `bernoulli`; a stratified or work sampling design adds each template's own bound.
- `integrity_profile` is `verity/integrity-profile/v1`: `{"format", "protocol", "program", "query", "partition",
  "population", "registration": <digest>, "scheme", "roots": [{"port", "root", "leaves"}], "law", "drawn",
  "inputs": {"anchored": <anchors>, "linked": []}, "budget_log2_t": null, "terms": null, "epsilon_log2": null,
  "evidence": {"receipt": <SHA-512 of canon(receipt)>, "draw": <SHA-512 of canon(draw)>, "sessions", "verdict"}}`,
  `sessions` being the session records' SHA-512 and `verdict` the verifier of record's digest (a list when the drawn
  units were proved in several sessions). Its ε terms come from the soundness analysis, not from the run, so the run leaves them null.

Pinned by `tests/test_one_stage.py` (`test_profile_counts_wrong_units`, `test_the_record_states_the_one_profile_in_its_recorded_bytes`,
`test_audit_record_needs_every_check_and_verdict`, `test_audit_record_holds_the_served_draw_and_the_sessions_roots`,
`test_the_wrong_unit_bound_matches_worst_case_and_scales`, `test_audit_record_under_the_stratified_law`,
`test_the_subset_bound_is_now_the_exact_count`, `test_a4_under_the_stratified_rule`, `test_audit_record_under_the_work_law`).

### The consumers

`verity.core.protocols.sampled_proofs.consumers` reads any `IntegrityProfile`, whichever protocol produced it, at its
delta, through its weighted bound `harm_bound`; each application counts delta once.

- `exfiltration_bound(profile, output_bits)` is `harm_bound(output_bits) + location_bits(profile)`, in bits.
  `location_bits` is the location term: with `K = ⌊bound()⌋` of `n` units, 0 when `K ≤ 0` or `n = 0`, `n` when
  `2K ≥ n`, and otherwise `n H₂(K/n) (1 + MARGIN)` rounded up to the next float, at most `n`. The Lean states the bound
  (`audit_exfiltration`, `card_admissible_le`), and the tests hold the Python to be at least it.
- `compute_bound(profile, work)` is `harm_bound(work)`. Under the two-stage profile a faulty RU is charged its class's
  `n_v` proof units, the most work it can hold. Policies and PoUW pass their own utility to `worst_case`.
- A unit's output bits or work is one number for every unit, or per stratum a number or `(bits, units)` pairs.

Pinned by `verity/core/protocols/sampled_proofs/tests/test_consumers.py` and
`verity/core/protocols/sampled_proofs/tests/fixtures/exfiltration_lean.json`, whose recorded sha256 of the Lean files it came from fails `test_the_lean_vectors_were_generated_from_this_tree` until it is regenerated.

### The proof service

`service.Stream` holds one protocol's registrations to its `Spec`, on the challenger's side:

- **The public-items rule** (`check_public`) runs before one-stage's R1 to R7. Each field of a registration record is the
  service's own (`format`, `protocol`, `previous`, `law`, `scheme`, which is `SCHEME`, and `query`, which is the
  check's), a declared Value's one salted root (a `roots` entry by port, a `registered` entry by value), or an item the
  Spec declares in `public` with its kind (`structure`, `output`, `coins` or `commitment`) and reason. The declared
  items a record shows are `population` (for `population`, `program` and `partition`), `population:<template>` for each
  stratum, `window`, `leaf_layer` and `anchors` when not empty, and each value's `leaves:<v>` and `domain:<v>`, with
  `words:<v>` and `word_bits:<v>` for a registered value, an entry's fields read in key order. Its codes are `keys`,
  `roots`, `unsalted`, `query` and `undeclared`; the `Stream` runs it by the verifier of record.
- **The order.** A stream holds one open registration at a time. `register` logs the record and its receipt, whose
  `received` is the record's position in the stream's log. `select` then draws once, from the challenger's coins as
  `two_stage.draw_first` reads them (the Lean verifier's `Flock.Draw.drawWith` on the same bytes, with no derivation),
  or from the stream's `sampler`; either draw is held to `draw.check_draw`. Under a work sampling design,
  `select(closure=)` takes the challenger's closure map over `[0, N)`, refused on another sampling design or out of
  range before any coin is read, and the draw carries the drawn units' closure (`two_stage.closure_of`), which the proof
  proves with them (`draw.proved`).
- **Two stages.** `TwoStage(replay, "all")` draws RUs by `replay`, and every proof unit of a drawn RU is checked, so
  there is no second draw; a count `k` is refused. The Spec's values split into the boundary (committed by serve time)
  and the interiors (phase `interior`, never registered values), and each registration holds only its own stage's roots.
  `register` takes the boundary; after `select` and before `deliver`, `register_interiors(draw, record, ports)` takes the
  interiors once: one-stage's record over the `len(draw.proved)` proved RUs (unit `i` is RU `draw.proved[i]`) under
  `subset:len(draw.proved)`, whose `previous` is the boundary's digest and whose `window` differs from it (R6). It is
  logged after the boundary, so the next boundary's `previous` is the interiors' digest. A proof delivered before the
  interiors is refused (`interior`), unless no RU was drawn or the proof is late. The outcome accepts only when both
  registrations' audit records accept, `session_roots` split between them by value, and its profile is the boundary's,
  the replay sampling design's over the RUs. Its refusals are `rule`, `not-drawn`, `final`, `delivered`,
  `interior-registered`, `interior-empty` and `interior-population`, then the public-items rule's and R1 to R7's.
- **No retries.** A second `select`, `register_interiors`, `deliver` or `outcome` is refused. A proof delivered more than
  `Check.proof_deadline` seconds after the draw, or none by then, is the symbol `late`. An outcome is pending until every
  verifier of record has run, and each verdict of record names the delivered sessions and reports the public outputs
  alike (`check_verdicts`: `pending`, `sessions`, `public`, by the verifier of record).
- **The failure budget.** At its `budget`-th `reject` or `late` a stream stops, and it refuses registrations from then on.
- **The records.** The stream's digest is the SHA-512 of `{"format": "verity/proof-service/stream/v0", "spec": <the
  Spec's digest>, "budget", "delta", "challenger"}`. An outcome's record is `{"format":
  "verity/proof-service/outcome/v0", "stream", "order", "registration", "receipt", "draw", "sessions", "symbol",
  "public", "audit", "interior"}`, with the receipt and the draw as the SHA-512 of their canonical JSON, `audit`
  one-stage's audit record (null for `late`), and `interior` null, or under `TwoStage` `{"registration", "receipt",
  "audit"}` for the interiors in the same forms; `Outcome.record` is its SHA-512.

A stream refuses a two-stage count, the `Sequential` rule, a window statement and a timed session until they are built.
Pinned by `tests/test_service.py` (`test_round_trip_register_select_deliver_outcome`,
`test_the_rule_refuses_an_extra_field` and the other `test_the_rule_refuses_*`,
`test_no_retries_nothing_redraws_and_an_outcome_is_final`, `test_a_late_proof_is_a_reject`,
`test_the_stream_stops_at_its_budgeth_failure`, `test_a_call_names_its_own_streams_registration_and_draw`,
`test_two_stage_round_trip_registers_the_interiors_between_the_draw_and_the_proof`,
`test_the_interiors_are_registered_once_after_select_before_deliver_over_the_proved_rus`,
`test_a_two_stage_outcome_accepts_only_when_both_audits_do`, `test_a_work_draw_carries_its_closure_and_the_proof_proves_it`,
`test_a_closure_widens_a_work_draw_only_and_stays_in_range`, `test_the_default_draw_is_draw_first_on_the_challengers_bytes`).

PoUW's example runs on a two-stage `Stream` in `tests/test_service_two_stage.py`, following the architecture note
(`note:verity-root/20261006T0545Z-draft-daniel-one-recursive-architecture-v2`, §4.1): a uniform `subset:K′` over `N`
equal tiles, whose boundaries are A's rows, the tile digests and the per-call records, and whose interiors are A′, the
ok rows, `rowk`'s carries and the cap rows. A `work:K` first stage runs as well, with the challenger's closure, and an
accepted audit's profile is `two_stage.TwoStageLaw.profile` of `Stage(law)` over the RUs. Pinned by
`tests/test_service_two_stage.py` (`test_the_pouw_example_runs_the_four_steps_and_one_record_binds_them`,
`test_each_step_waits_for_the_one_before`, `test_the_interiors_are_the_proved_replay_units_held_to_the_rule`,
`test_a_work_first_stage_draws_with_the_challengers_closure`, `test_a_wrong_interior_or_boundary_rejects_the_audit_once`,
`test_the_stream_stops_at_its_budgeth_failure`, `test_no_retries_nothing_redraws_and_an_outcome_is_final`,
`test_a_late_proof_or_missing_interiors_are_late`,
`test_a_draw_of_no_rus_has_no_interiors_and_its_proof_goes_to_the_boundary`,
`test_what_a_two_stage_spec_and_stream_refuse`, `test_the_draw_is_the_challengers_own_randomness`).

### Two-stage sampling

`two_stage.TwoStageLaw({class: Stage(first, k)})` (decision 43):

- `first` is a probability `0 ≤ p ≤ 1` (each RU of the class independently with `p`), or a one-stage sampling design's
  object over the class's RUs (`draw`'s `subset`, `bernoulli`, or `stratified` and `work` expanded by `draw.derive`),
  its positions the class's RUs in index order. `k ≥ 0`, or `k = None` for every proof unit; an RU of `n_v` proof units
  has `min(k, n_v)` checked. A subset, stratified or work first stage takes `k = None`: thinned by a second draw, its
  acceptance probability is no sampling design a profile states. The Bernoulli sampling design over single units is the
  case where every unit is its own RU with `k = 1` (`TwoStageLaw.bernoulli`), and a gate-level `plan.VerificationPlan`
  is that case, every gate its own RU.
- An RU with `m` faulty proof units passes with probability `1 − p + p C(n_v − m, k) / C(n_v, k)`, but that is not
  the profile's sampling design: the prover picks `m` after the RU draw, and `m = 1` always leaves a skipped RU faulty.
- `TwoStageLaw.profile` is one level `replay` over the class's `n_r` RUs, with `delta`, the exhaustively checked facts
  (`programs`, `prescribed`, `linked`, `covered`) and `drawn`. Under a Bernoulli first stage (a probability, or a
  `bernoulli` sampling design) it is drawn `Bernoulli(p min(k, n_v) / n_v)` with `n_v` the class's largest RU; under a
  subset, stratified or work sampling design it is that sampling design's profile over the RUs, `Subset(k)` or
  `audit.core_law`'s `Stratified`, as `audit.profile` gives it over units, counting the drawn RUs (a closure only
  helps). A caller's `n_r` must equal the class's count, and its `n_v` cannot be smaller than the largest. It counts
  faulty RUs and weights nothing.
- The rate rests on the proof-unit draw being made by the verifier from its own randomness after the interiors are
  committed. A stage that checks every proof unit of a drawn RU draws nothing in stage 2.
- RUs are addressed by their index in the committed order, and the indices are distinct.

The closest Lean is `TwoStage.lean`'s `effEscape_full` and `twoStage_full_profile` (stage 2 checking all: the first
stage's own acceptance probability, for any first-stage sampling design, `Closure.lean`'s `Law.closure` among them),
`effEscape_bernoulli_prod`, and `Examples.lean`'s `effEscape_bernoulli`, `two_stage_b_tight`, `naive_two_level_unsound`
and `late_interior_insecure`; whether they cover `TwoStageLaw.profile` exactly, its largest-RU rule among them, is
unchecked. Pinned by `tests/test_two_stage.py` (`test_detection_matches_the_per_ru_law_and_the_profile_bounds_it`,
`test_toy_attack_is_admitted`, `test_the_profile_is_over_replay_units`,
`test_a_mixed_class_is_profiled_at_its_largest_replay_unit`, `test_bernoulli_is_the_single_unit_law`,
`test_indices_must_be_distinct_and_classes_known`, `test_a_first_stage_is_any_one_stage_law_and_checks_all_unless_bernoulli`,
`test_a_one_stage_first_stage_profiles_as_the_one_stage_law_over_replay_units`, `test_a_subset_first_stage_escapes_at_its_profile`).

### The verifier's randomness

The draw of record is `two_stage.OwnRandomness`: the operating system's generator read directly with the Lean verifier's
samplers (`Flock.Draw`), with no hash, no derivation and no beacon, the RU draw after the boundary is registered and the
proof-unit draw after the interiors are.

- Stage 1 under a one-stage sampling design (`draw_first`) reads the bytes as `Flock.Draw.drawWith` does: `subset` and
  `bernoulli` on one stream (`bernoulli` at its `NUM/DEN` as written), `stratified` and `work` one stream per stratum,
  each of `2 k_s · width(n_s) + 64` bytes, extended by as many again and rerun whenever it runs out. The result is the
  one-stage draw object, which `draw.check_draw` holds to the registration and the sampling design; the verifier's
  closure map widens a work draw only, as `flock-verify draw --closure` does (`closure_of` is Lean's `closureOf`).
- Nothing here derives a draw, the proof service's included (`service.Stream.select` is `draw_first` on the
  challenger's coins). vLLM's legacy replay challenge derives its own (`verity_vllm.commit.challenge.replay_key`,
  under the domain `verity/sampled-proofs/vu-sample/v1`) until its re-baseline.

Pinned by `tests/test_two_stage.py` (`test_own_randomness_is_the_lean_verifiers_samplers_on_raw_bytes`,
`test_draw_first_is_the_lean_verifiers_draw_on_the_same_bytes`, `test_a_stream_that_runs_out_is_extended_as_lean_does`,
`test_a_work_draw_carries_its_closure_and_the_profile_counts_the_drawn_rus`, `test_draw_first_refuses_what_lean_refuses`).

### Verification plans

`plan` builds obligations from a call's declared interface (`args`, `returns`), never from `FunctionCall.out`, which
rescans the circuit on every call. A prescribed input is a sampled gate like any other, and an `UnconditionalObligation`
is proved on every audit, outside both draws, so its gates are the profile's `covered` families. Pinned by
`verity/core/protocols/sampled_proofs/tests/test_plan.py`.
