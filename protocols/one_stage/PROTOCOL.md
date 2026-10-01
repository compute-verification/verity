# One-stage audit

Every proof unit's boundary is committed while serving; the verifier draws units once, from its own randomness, and the
drawn units are proved. The two-stage protocol (replay, then a second draw) is `protocols/sampled_proofs`.

## Lifecycle

1. **Serving commits** the values the partition's units read and write, and sends the roots.
2. **Registration** (`registration.record`, `verity/registration/v1` of `docs/audit-protocols.md` §0.1): program, query
   and partition digests, the population the query derives, the law, the scheme, every root with its leaf count and domain,
   the leaf layer's digest, anchors, the window (provenance, with `kind` `served` or `stand-in`) and the previous record.
   The verifier checks R1-R6 against its own copies (`registration.check`), holds its own public file to it
   (`registration.matches_public`) and writes a receipt (`registration.receipt`) before it draws. Served roots use the
   domain rule agreed with vllm-serving-commit (`registration.served_domain`).
3. **The draw**: the verifier samples units from the operating system's randomness under its law (`stratified:K`, the
   default, `work:K` with its work table, `subset:K` or `bernoulli:NUM/DEN`) and sends the draw in the clear, before the
   session's coin commitment. Nothing is derived from a commitment. `draw.check_draw` holds it to the registration and
   the law.
4. **Proofs** of the drawn units, in one session whose statement is the drawn units' (`draw.check_proved`: its instances
   are the drawn units and a work draw's closure, ascending).
5. **Verification** by each verifier that runs, the Lean verifier of record among them.
6. **The audit record** (`audit.audit_record`): the registration, its receipt, the draw, the law, the proved units, the
   statement's digests, every verdict, and, when every check and verdict that ran accepted, the integrity profile. The
   profile (`audit.profile`) is core's `IntegrityProfile` of one level `unit` with `sizes = (population,)`, drawn by the law
   under every law, with delta: per-stratum bounds on wrong units, nothing weighted. The record states it as recorded
   runs hold it: `profile` (its JSON; a stratified law keeps the v0 form, the law's JSON with delta and the count drawn
   per template), `wrong_units` (its bounds) and `verity/integrity-profile/v1` (§0.2), whose ε terms come from the
   soundness analysis and stay null here.

## The stratified law

The default law (Daniel, Sep 27): one stratum per template, and inside each a uniform draw without replacement of a fixed
size, independently across strata. `docs/sampling-strategies.md` has the analysis.

- **Strata.** Each verifier derives them from its own program and query (`partition.strata`), never from the prover. Apply
  the query and group the units by the template descriptor id of their root node. A stratum's units are ascending,
  disjoint, non-empty half-open ranges `[lo, hi)`, with adjacent ranges merged, and the strata are in the order of their
  first unit. This is defined for `Q_template_instance` and `Q_template_instances`; any other query refuses it until it
  defines its stratum key.
- **Sizes.** For the agreement's `k`, with $1 \le k \le n$, stratum $s$ draws $k_s = \max(1, \lfloor (2 k n_s + n)/(2n) \rfloor)$
  units (`draw.stratum_k`): $k\,n_s/n$ rounded half up, and never fewer than one, so every template is exercised in every
  audit. Always $1 \le k_s \le n_s$, and the total drawn is $\sum_s k_s$.
- **The law object** is the agreement's rule `stratified:K` expanded over the program and query (`draw.derive`):

  ~~~json
  {"law": "stratified", "k": 1024,
   "strata": [{"template": "{descriptor id}", "units": [[0, 287]], "k": 1}, ...]}
  ~~~

  The registration carries it expanded, and R1 holds it to the verifier's own expansion.
- **The draw** carries the law's keys plus `population` and `units`:
  `{"law": "stratified", "population": N, "k": K, "strata": [...], "units": [...]}`. `draw.check_draw` holds its law to the
  verifier's, and checks that the strata tile the population with the rule's `k` and that exactly $k_s$ units fall in
  each stratum. The sampler of record draws each stratum's `subset k_s n_s` on its own stream and maps local index $j$ to
  the stratum's $j$-th unit.
- **A session that proves one template's units** (A4's members) is served that template's share as a `subset:k_s` draw
  over its $n_s$ units, in its own indices. The audit record keeps the union draw.
- **The bound.** A set with $m_s$ wrong units in stratum $s$ escapes with $\prod_s \binom{n_s-m_s}{k_s}/\binom{n_s}{k_s}$. The
  audit record's `wrong_units.bound` is core's exact `Stratified.count_bound` at delta, and `per_template` gives each
  template's own bound. A template wrong in every unit is never missed. `subset:K` is one stratum and gets the same exact
  bound.

## The work law

The draw law's second version (Daniel approved work-proportional sizing on Sep 28). It is the Lean verifier's
`Flock.Draw.Law.work` (`backends/flock/verifier/PROTOCOL.md`, "The work law"), and `draw.py` mirrors it: the stratified
law's strata and sampler, with each stratum's draw sized by the PoUW work its units do.

- **The work table** (`draw.work_table`) is the verifier's own. It maps each template descriptor id to the work one of its
  units is credited, a natural number in any fixed unit, or to `{"work": w, "floor": f}` with `f ≥ 1`; the floor is one
  when the table gives only the work. A template the table does not name refuses the law.
- **Sizes.** With `W = Σ_s w_s·n_s ≥ 1` and `1 ≤ K ≤ N`, stratum `s` draws `k_s = min(n_s, max(f_s, ⌈K·w_s·n_s / W⌉))`
  (`draw.work_k`, Lean `workK`): its work share of `K`, rounded up, never below its floor and never above `n_s`. A work
  stratum is `{"template", "units", "work", "floor", "k"}`, and the law object `{"law": "work", "k": K, "strata": [...]}`
  is the rule `work:K` expanded by `draw.derive` with the table. The registration carries it expanded (`record(...,
  work_table=)`), and R1 holds it to the verifier's own expansion with its own table.
- **The draw** is `{"law": "work", "population": N, "k": K, "strata": [...], "units": [...]}`, with an optional `closure`:
  units proved with the drawn ones (for a tile, the units its credit reads), non-empty, ascending and none of them drawn.
  The session proves the drawn units and the closure.
- **What it guarantees** (Lean `FlockSoundness/Audit/Work.lean`). Wrong units doing a fraction `ε` of the credited work
  escape with probability at most `(1 − ε)^K` (`work_escape_le`), and a stratum with `b_s` of its `n_s` units wrong
  escapes with at most `((n_s − b_s)/n_s)^(f_s)` (`work_escape_floor_le`). The work law is the stratified law at its
  sizes, so the audit's profile is `Stratified` over the same strata at the work sizes, with its exact bounds.
- **What a unit's work may count** (the Pearl-C statement review, item 5). A unit's work is what it is credited, counted
  on the rows the Program declares, never on rows a kernel pads it with.
  - Filler is credited nothing: a unit of filler only, or one the prover declares uncredited before the draw, does work 0.
    Its floor still draws it for integrity, and missing it costs no work.
  - A unit mixing real and filler rows is credited its real cells only.
  - The law weights by template, so every unit of a template must be credited the same work. A template whose units are
    padded differently is split before the law is derived, or credited 0. The escape bound needs every unit's inclusion
    probability to be at least `K·w_u / W` over the credited `W`, which a per-template work that over-credits a unit
    breaks.

## Consumers

Applications read the profile, and it is neither of them (`consumers`). Each weights wrong units by what one costs it,
through the profile's generic weighted bound (`IntegrityProfile.harm_bound`), at the profile's delta:

- **Exfiltration security** (`exfiltration_bound`): wrong units weighted by their output bits, which the prover chose,
  plus which units are wrong (`location_bits`, log₂ of the sets of at most `bound()` units). The soundness package's
  Lean states the bound as `audit_exfiltration`. Lean evaluates it on a grid of uniform laws and one of stratified laws
  (`tests/lean/ExfiltrationVectors.lean`, regenerated by `tests/lean/generate.sh`), and its kernel checks each vector's
  count K: `miss (K + 1) < δ ≤ miss K`. For a stratified law that is `stratified_miss_eq_greedy` on greedy fills of
  `K + 1` and `K` units, and the kernel checks the τ that separates each fill exactly, in integers. The tests check that
  core's `bound()` is K, that its stratified `miss` is those fills' escapes, that `exfiltration_bound` is at least the
  Lean's bound in bits, and that the files the vectors were generated from are unchanged.
- **Compute security** (`compute_bound`): wrong units weighted by their work, which the prover could have skipped.

## Partition

Core's `verity/partition/v1` (`verity.ir.partition_object`): `{"format": "verity/partition/v1", "program": <SHA-512 of the
program>, "query": {"name", "version", "params"}}`, digest `SHA-512("verity/partition/v1\0" ‖ canon(object))`. The units
are core's evaluation of the query, which refuses a malformed query and a program that is not topological. The queries
applied here are `Q_template_instance` v0 and `Q_template_instances` v0: each instance of a listed template in the
program's root body is one unit, in canonical order, with no width rule. They are demo queries for template runs, not the
serving partition. A record made before the canonical envelope names `{"program", "query"}` and keeps that object's
digest under the same rule (`partition.digest`).

## Not here yet

- The serving path's own commitment: run A0 uses a stand-in, a fresh commitment to values captured from serving, and the
  registration says so (`window.kind = "stand-in"`).
- The Lean verifier of record on M0's current statement format, and the partition and program bound inside the statement.
- The protocol's security statement and its proof in Lean.

`benchmarks/one_stage/` runs the lifecycle end to end with M0's prover and the Lean verifier (`pod.sh`, `a0.py`).
