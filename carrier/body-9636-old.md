Lane proofs
Refs PRF-548
Goal demos/remote-demo

**The change.** Three changes to the one-stage SPOT verifier that have to land together, each in the Python check and the Lean check. The first two are also in the locked statement that describes the check, with its proof. The third is proved about the check and is not yet in the locked statement.

1. **The registration check (R1–R8) takes the verifier's certified entries.** One new input, `certified`: the entries the verifier built from registrations it read under the deployment's key, for values that are none of its program's parameters (a link's certified rows). R7 held the record's registered values to exactly the parameters the verifier expects registered. It now holds them to those and the certified values, each once, and holds each certified value's entry to the verifier's on every key, the root included. Without the input the check is today's.
2. **The audit record's `registered` check keeps one direction of its equality.** Each read the accepting verdicts report of one of the record's own values must be of a registered value at its registered root; a registered value that no verdict reports a read of no longer fails the check. These are the commits of #9023, unchanged, merged here; #9023 closes when this lands.
3. **The audit record gains a check, `certified`:** every value the verifier holds a certified entry for has a read that an accepting verdict of record reports. It keeps, for certified values only, the direction that change 2 drops. The check refuses more than the locked statement says of an accept; why the statement does not yet say it is under "The certified read" below.

They are one pull request because 3 needs both others: it reads the entries of 1, and it only bites under 2 (under the equality every registered value already needs a reported read). And 2 without 3 would let a certified port's read go unreported once 1 admits certified values.

The change to what an accept means was approved by the project lead on 12 Nov.

**What the branch holds beside its own change.** The branch holds `main` (`4a7291116e11`), the complete answer's head (#9458, `049c5875989c`, open) and #9023's three commits (`6b782d972ec4`). Against `main` it is 35 files: thirteen that only this change touches, fifteen that only #9458 touches, and seven that both do. If #9458 lands first, `main` is merged in here again and the difference is this change's twenty files alone. This change's own files are twenty:

- Lean code: `Spot/Registration.lean`, `Spot/Record.lean`, `Spot/OneStage.lean`, `Spot/Run.lean`, `Spot/Pieces.lean` under `verity/protocols/spot/verifier/lean/`;
- Python: `one_stage/registration.py`, `one_stage/audit.py`, `one_stage/verifier.py`, `service.py` under `verity/protocols/spot/verifier/`;
- the statement and its proofs: `security/Properties/Spot.lean`, `security/Proofs/Properties/Spot.lean`, `security/Proofs/Properties/Spot/Pinned.lean`, `security/Proofs/Flock/SpotFlock/Code.lean`;
- tests: `lean/tests/test_difftest.py`, `tests/test_registered.py`, `tests/test_service_outputs.py` under the verifier, and two assertions of `benchmarks/one_stage/tests/test_stop_audit.py`;
- the verifier's README, and the two locks (`security/lean-audit.json`, `security/Properties/lean-audit.json`) as the records run wrote them.

## The statement, before and after

### The registration (`Properties.Spot.Registered`)

Before, in `security/Properties/Spot.lean`:

```lean
def ExpectsV2 (v : Registration.Verifier) : Prop := v.registered.isEmpty = false

def R1 (rec : Json) (v : Registration.Verifier) (law : Json) : Prop :=
  keys? rec = some (if v.registered.isEmpty then Registration.KEYS else Registration.KEYS_V2) ∧
  jeq (getD rec "format") (.str (if v.registered.isEmpty then Registration.FORMAT else Registration.FORMAT_V2)) ∧
  …

-- R7's last three conjuncts
    ValuesAre es mine' ∧
    (∀ want ∈ mine', ∀ e, es.reverse.find? (fun e => jeq (getD e "value") (.str want.value)) = some e →
      EntryIs e want (heldFor v.supplied held' want.value)) ∧
    SuppliedIs es mine' v.supplied (Supplied.root …)

def Registered (v : Registration.Verifier) (rec law : Json) : Prop :=
  R1 rec v law ∧ R2 rec v ∧ R3 rec v ∧ R4 rec v ∧ R5 rec v ∧ R6 rec v ∧
    (ExpectsV2 v → R7 rec v ∧ PoolChained rec v ∧ PinnedHeld rec v) ∧ R8 v ∧ RL v
```

After:

```lean
def ExpectsV2 (v : Registration.Verifier) : Prop := (v.registered.isEmpty && v.certified.isEmpty) = false

def R1 (rec : Json) (v : Registration.Verifier) (law : Json) : Prop :=
  keys? rec = some (if v.registered.isEmpty && v.certified.isEmpty then Registration.KEYS else Registration.KEYS_V2) ∧
  jeq (getD rec "format")
    (.str (if v.registered.isEmpty && v.certified.isEmpty then Registration.FORMAT else Registration.FORMAT_V2)) ∧
  …

-- R7's last three conjuncts
    ValuesAre es (mine' ++ v.certified.map (·.1)) ∧
    (∀ want ∈ mine' ++ v.certified.map (·.1), ∀ e,
      es.reverse.find? (fun e => jeq (getD e "value") (.str want.value)) = some e →
        EntryIs e want (heldFor v.supplied held' want.value)) ∧
    SuppliedIs es (mine' ++ v.certified.map (·.1)) v.supplied (Supplied.root …)

def CertifiedHeld (rec : Json) (v : Registration.Verifier) : Prop :=
  ∀ x ∈ v.certified, ∃ e, Registration.entryOf rec x.1.value = some e ∧ jeq (getD e "root") (.str x.2) = true

def Registered (v : Registration.Verifier) (rec law : Json) : Prop :=
  R1 rec v law ∧ R2 rec v ∧ R3 rec v ∧ R4 rec v ∧ R5 rec v ∧ R6 rec v ∧
    (ExpectsV2 v → R7 rec v ∧ PoolChained rec v ∧ PinnedHeld rec v ∧ CertifiedHeld rec v) ∧ R8 v ∧ RL v
```

`v.certified : Array (Registered.Entry × String)` is the new field of `Registration.Verifier`: each certified value's entry (its value, leaves, words, word bits and domain) with the root the verifier holds for it. `mine'` is, as before, the verifier's own entries for the parameters it expects registered, after its output values' steps. `AcceptMeans` itself is unchanged in text: it reads `Registered`.

### The units (`Properties.Spot.UnitsProved`)

Before, its last conjunct and the relation it names:

```lean
def SameRoots (a b : Array (String × Json)) : Prop :=
  a.size = b.size ∧ ∀ x ∈ a, ∃ y ∈ b, x.1 = y.1 ∧ jeq x.2 y.2

def UnitsProved (c : Commitment) (e : Evidence) : Prop :=
  … ∧
  ((field? c.record "registered").isSome ∨ (openedRoots c.record e.verdicts).isSome →
    SameRoots ((openedRoots c.record e.verdicts).getD #[]) (Record.registeredRoots c.record))
```

After:

```lean
def OpenedRegistered (opened registered : Array (String × Json)) : Prop :=
  ∀ x ∈ opened, ∃ y ∈ registered, x.1 = y.1 ∧ jeq x.2 y.2

def UnitsProved (c : Commitment) (e : Evidence) : Prop :=
  … ∧
  OpenedRegistered ((openedRoots c.record e.verdicts).getD #[]) (Record.registeredRoots c.record)
```

`openedRoots` is unchanged: the `{value: root}` that the accepting verdicts of record report (`Record.sessionRegistered`, a value reported under two roots mapped to null), kept to the record's own values (`Record.ownRegistered`). `Record.registeredRoots` is unchanged: each registered value's last entry. `AcceptMeans`' own text is unchanged: it reads `Registered` and `UnitsProved c e`.

### The certified read: proved about the check, not in the locked statement

In the proofs tree (`security/Proofs/Properties/Spot.lean`), new:

```lean
def CertifiedRead (certified : Array String) (opened : Array (String × Json)) : Prop :=
  ∀ v ∈ certified, ∃ y ∈ opened, y.1 = v

def UnitsRead (p : Params) (c : Commitment) (e : Evidence) : Prop :=
  UnitsProved c e ∧
    CertifiedRead (p.verifier.certified.map (·.1.value)) ((openedRoots c.record e.verdicts).getD #[])

theorem acceptMeans_iff (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Core.Integrity) :
    OneStage.check p c e = .accept pub prof ↔
      ∃ law d n, Registered p.verifier c.record law ∧ Drawn e.draw (getD c.record "population") law ∧
        UnitsRead p c e ∧ VerdictsAccept e.verdicts ∧
        Record.profileDraw law = some d ∧ nat? (getD c.record "population") = some n ∧
        pub = publicOf p e law d n ∧ prof = profileOf p c e d n
```

So the compiled check accepts exactly when the locked statement's conjuncts hold and each certified value has a reported read. `Properties.Spot.acceptMeans` is the forward direction without the read.

Why the locked statement does not say it yet. Saying it changes `AcceptMeans`' own text (`UnitsProved` would take the parameters), and a guarantee whose own text changes carries two witness lemmas: that its conclusion fails somewhere, and that the compiled check accepts one built instance. The second is not written for this check. The clause is also the one a later rule replaces: the read asked of each sampled unit at the row the registered Program gives it ("Not established", below). The clause enters the locked statement with that rule and its witnesses.

## What an accept said, and what it says now

### Of the registration

**Before.** An accepted audit's record keeps R1–R8 against the verifier's own copies. Where the verifier expects parameters of its program registered, the record is in the v2 format, its registered values are exactly those parameters, each once, each with the leaves, words, word bits and domain the verifier derives for it, and each held to the root the verifier holds for it where it holds one.

**Now.** The same, with one more kind of registered value. The verifier may hold certified entries: entries for values that are no parameter of its program, each with a root. Then:

- the record is in the v2 format where the verifier expects parameters registered or holds a certified entry;
- the record's registered values are exactly the parameters the verifier expects registered and the values it holds certified entries for, each once, so a value that is both is in no accepted record;
- each certified value's entry in the record has the leaves, words, word bits and domain of the verifier's entry;
- each certified value's entry in the record has the root the verifier holds for it (`CertifiedHeld`).

**It is the same statement whenever the verifier holds no certified entry.** With `v.certified` empty, `v.registered.isEmpty && true` is `v.registered.isEmpty`, `mine' ++ #[]` is `mine'`, and `CertifiedHeld` holds of every record. This is read off the definitions; no Lean lemma states it.

**No named assumption is added, and no premise.** `Properties.Spot.acceptMeans` takes the premises it took. `Properties.Spot.ExpectsV2`, which its statement reads and the lock lists as its one open premise, is the definition above, widened.

**What the statement does not say:** that the entries the verifier holds are the certifier's. The verifier's entries are its input, as its program and its pinned roots are. The caller builds them from the registration it read under the deployment's key.

### Of the reads

**Before.** Where the record registers values or the verifiers report reads of the record's own values, those reads were opened against exactly the record's `{value: root}`: every reported read is of a registered value at its registered root, and every registered value has a reported read.

**Now.**

- Every read the accepting verdicts report of one of the record's own values is of a registered value, opened against the record's root for it. This direction stays, in the check and in the statement.
- A registered value that no verdict reports a read of is not required to have one. This direction is dropped: the old conjunct implies the new one, so every record the old check accepted the new check accepts, and the statement says less.

**What the check asks beyond the statement.** Each value the verifier holds a certified entry for has a read the accepting verdicts report (`CertifiedRead`, proved in `acceptMeans_iff`). With the first bullet that read is at the record's root for it, which `CertifiedHeld` holds to the verifier's. With no certified entry it asks nothing (it quantifies over an empty list).

**No named assumption is added, and no premise,** by any of the three.

## What the checks accepted before, and accept now

### The registration check

| The verifier | The record's registered values | Before | Now |
|---|---|---|---|
| holds no certified entry | any | as today | unchanged |
| expects parameters registered and holds certified entries | those parameters and the certified values, each once, each entry the verifier's on every key | no such input; without it reject, `R7-values` | **accept** |
| expects no parameter registered and holds certified entries | the certified values alone, each entry the verifier's on every key | no such input; without it reject, `keys` | **accept** |
| holds certified entries | a certified value is missing | | reject, `R7-values` and `R7-certified` |
| holds certified entries | a certified value twice, or a value it holds no entry for | | reject, `R7-values` |
| holds certified entries | a certified value at another root | | reject, `R7-certified` |
| holds certified entries | a certified value with other leaves, words, word bits or domain | | reject, `R7-leaves`, `R7-words`, `R7-word_bits`, `R7-domain` |
| holds certified entries | a v1 record | | reject, `keys` |

`R7-certified` is the one new code. A certified value is never supplied, pinned, held or an output: a certified name that is a parameter the verifier expects registered or a value it fills, and an entry in another form, are refused before the record is read (`ValueError` in Python; the Lean refuses the parameters).

### The audit record

| The verdicts of record report | Before | Now |
|---|---|---|
| every registered value, each at its registered root | accept | accept |
| the same, and a value the record does not name | accept | accept |
| some registered values at their roots, the others not at all, none of those certified | reject, `registered` | **accept** |
| no reads, or no report at all, with registered values in the record, none certified | reject, `registered` | **accept** |
| a registered value at another root, by one of its ports or by all | reject, `registered` | reject, `registered` |
| a value under two roots across verdicts | reject, `registered` | reject, `registered` |
| a root port's value as a registered value | reject, `registered` | reject, `registered` |
| no read of a certified value | (no certified value could be registered) | reject, `certified` |
| a certified value's port under a value the record does not list, at another root, and no other read of the value | | reject, `certified` |
| the same, beside another accepted session's read of the value at its root | | accept |

The `certified` check's detail names the values: `{"certified": [...], "unread": [...]}`. The `registered` check's detail keeps its three keys; `wrong` now lists the reported values that are not registered at the root reported.

## The new input

`certified={value: {"root": <128 lowercase hex>, "leaves": int, "words": int, "domain": str}}`, with `"word_bits"` optional (16 or 32; 16 when left out). It is a keyword of `registration.check`, `verifier.check`, `verifier.run` and `service.Stream.register`. The stream keeps it with the registration's other copies and hands it to the run of record at the outcome, so nothing is passed there. In Lean it is the parameters' key `certified`, read by `OneStage.paramsParsed` into `Registration.Verifier.certified`; the key is sent only when there is an entry.

The audit record takes the certified values' names: `audit.audit_record(..., certified=())` and `verifier.audit_record(..., certified=())`, which accepts the same map. `Stream.outcome` passes them from what `Stream.register` kept. In Lean they are `Registration.Verifier.certified`'s values, read by `OneStage.check` and by the run of record; the `record` piece takes an optional key `certified`, a list of strings. A driver that calls `verifier.audit_record` itself passes them, or the `certified` check does not run there.

## What it does not do

- **It does not hold which row a unit reads.** A session's verdict reports each registered port's value and root, and no row. A `--registered` file whose `reads` for a certified port are another order of the same rows (reversed, or shifted by one) is accepted by the session's verifier and reports what the honest file reports, so no check here sees it. Closing it needs the rows derived from the registered Program; it is not in this pull request.
- **It does not refuse every renamed certified port.** A read reported under a name the record does not hold is dropped before the comparison (`Record.ownRegistered`), as before. The `certified` check refuses the audit only when that leaves the certified value with no reported read.
- **It does not derive the verifier's entries, or its `--registered` file, from the record or the Program.** The entries are the verifier's input, as its program and its pinned roots are; the statement does not say they are the certifier's.
- **`register_interiors` takes no certified entries**: a two-stage interior's record that names a certified value stays refused, and `Pieces.run` refuses an interior's parameters that name the key.
- **The sampled audits take none.** `OneStage.paramsOfJson`, which the one-stage audit, its zero-knowledge form and the recursive audit read, refuses parameters that name `certified`. Only the registration check, the run of record and the audit record's `certified` check gain a form.
- **It does not touch** the per-member form (`Spot/OneStageZk.lean`, `classHeld`, `checkMembersAt`), the two-stage audit's `second-reported` check, the sample or its bound.
- **`verity/protocols/spot/verifier/README.md` is over its Markdown cap** without this change: it is 49,340 bytes against 49,152 on `main`. This change's own commits add one line of 114 bytes to it; the rest of the head's 50,290 is the complete answer's paragraph, held here until #9458 lands.

## Not established

- **A member with no unit sampled.** The `certified` check asks, for each certified value, a session of record that reports reading it, as the equality asks of every registered value today. A member with no unit sampled has no session, so a window whose certified rows are read only by such a member is refused, honest or not. It bites only in an audit sampled below every unit; the network certifier's scan samples every unit in its audits of record. The rule that asks the read of sampled members only needs each unit's row from the registered Program, as the rows do, and replaces this check's clause then.
- **The certified read in the locked statement.** `Properties.Spot.acceptMeans` does not state it; `acceptMeans_iff` in the proofs tree does.
- **The rows.** Nothing here holds which row of a certified value a unit reads or writes. A `--registered` file that reverses a certified port's rows (unit k at row T − 1 − k), or shifts an output port's by one (unit k at row k + 1), at the same value and root, is accepted.

## Tests

Existing cases whose verdict changes, each one a case of the dropped direction (change 2):

1. `test_difftest.py::test_a_v2_audit_reads_the_sessions_registered_roots`: a value's reads missing, was reject, now accept.
2. The same test: an empty report (`{}`), was reject, now accept.
3. The same test: no report (`null`), was reject, now accept.
4. `test_registered.py::test_the_audit_record_holds_the_roots_the_reads_were_opened_against`: `session_registered=None`, was a failed `registered` check, now accepted with the same integrity profile.
5. The same test: `session_registered={}`, likewise.
6. `test_registered.py::test_the_opened_roots_come_from_the_verifier_of_records_accepted_verdicts`: a verdict of record with no `registered` field, was reject, now accept.

7. `benchmarks/one_stage/tests/test_stop_audit.py::test_a_record_is_refused_by_each_entry_that_ran_and_did_not_accept`: a stop audit whose session Lean's verdict refuses. Its record is refused as before; its failed checks were `registered` and `roots` and are now `roots` alone, since no verdict reports a read and a registered value needs none.
8. `test_stop_audit.py::test_the_stop_streams_answer_carries_its_audit_records_complete`: the same audit's answer, whose reason names the failed checks: `checks: roots` where it read `checks: registered, roots`. Still a reject.

Every other existing case keeps its verdict. Added:

**The certified entries (change 1).**

- `test_registered.py::test_r7_takes_the_verifiers_certified_entries` (Python's check): the record names a certified value once, its entry the verifier's on every key; a verifier holding no entry refuses (`R7-values`, or by its keys where it expects nothing registered); a record whose registered values are the certified ones alone is accepted with the entry; a certified name the verifier expects, fills, pins, holds or takes from an earlier run, or an entry in another form, is refused before the record is read.
- `test_difftest.py::test_r7_takes_the_verifiers_certified_entries`: Lean and Python agree on each of those, at 16 bits and at 32; the sampled audit refuses parameters that name certified entries; a run of record whose parameters name the entry accepts and one without refuses at its register step; a two-stage run whose interior's parameters name the key is no run's.
- `test_service_outputs.py::test_a_stream_takes_the_verifiers_certified_entries`: a window's record that registers a link's certified rows and no other value is refused through `Stream.register` without the entries (`keys`), with an entry at another root (`R7-certified`) or of other leaves (`R7-leaves`), and accepted with them, through to the outcome; a later window that registers its own state beside the rows is refused without them (`R7-values`).
- `test_service_outputs.py::test_the_run_of_record_reads_the_certified_entries`: a run of record that drops the entries refuses the registration the stream accepted.
- `test_service_outputs.py::test_an_interior_registration_takes_no_certified_entries`: `register_interiors` refuses an interior record that registers certified rows and takes no `certified` keyword.

**The kept direction (change 2).**

- `test_difftest.py::test_the_record_fails_a_read_not_of_a_registered_root_and_no_missing_read`: the record Lean writes equals the one Python writes, accepted with no `registered` check when a registered value has no reported read, and failing `registered` with the value named when its reads are at another root or a root port's value is read as registered.
- in the first test above, all of a value's ports at another root: reject; in `test_registered.py`, a value reported under no single root: reject, and the detail of each failing case.

**The certified read (change 3).** Each asserts the case is accepted when the certified names are not given, so it fails without the check:

- `test_registered.py::test_the_audit_record_holds_a_certified_value_to_a_reported_read` (Python's record): on a record whose registered values are the certified one alone, accepted with a read at the root; failing `certified`, the value named, with no read, an empty report, or the value's port named under a value the record does not list at another root; accepted beside another session's read of the value; a read at another root fails `registered`, not `certified`.
- `test_difftest.py::test_the_record_holds_a_certified_value_to_a_reported_read`, and a generator of certified cases among the `record` piece's inputs: the record Lean writes equals the one Python writes on each of those.
- `test_service_outputs.py::test_a_stream_holds_a_certified_value_to_a_reported_read`: the stream's outcome is refused by the audit record's `certified`, with no `run` or `registered` failure, when the verdict of record reports no read of the rows or reports their port under an unlisted value at another root.

## The proof: four checks

`Properties.Spot.acceptMeans` (`security/Proofs/Properties/Spot.lean`), restated through `Registered` and `UnitsProved`, and `acceptMeans_iff` beside it:

- for the certified entries: `certifiedHeld_iff` (the check's term against `CertifiedHeld`), read by `registration_nil` (the check passes exactly when `Registered` holds), whose R1 and R7 are now over the union;
- for the kept direction: `subDict_iff` replaces `sameDict_iff`;
- for the certified read: `certifiedCheck_ok` (the new check against `CertifiedRead`), with `none_fail_append` and `none_fail_ite_nil`; `recordChecks_nil` takes the parameters and says the record's checks pass exactly when `UnitsRead p c e` holds; `acceptMeans_iff` is over `UnitsRead`, and `acceptMeans` is its forward direction with the read dropped.

`check_eq_checkRef` (`Spot/Registration.lean`) still holds the check to its reference with the new term; `Spot/Pinned.lean` and `SpotFlock/Code.lean` change only where they name the widened R7 and the verifier's new field.

- Elaborates: yes. `verity lean try --siblings` of each of the nine changed Lean files, each lean exit 0 with no sibling declaring a name it does: run `r20261113-030550-5811`, at the head.
- Axioms clean: yes. The same tries' axiom probe: none of the theorems of `Spot/Record.lean` (4), `Spot/Registration.lean` (9), `security/Proofs/Properties/Spot.lean` (83), `security/Proofs/Properties/Spot/Pinned.lean` (4) or `security/Proofs/Flock/SpotFlock/Code.lean` (15) uses an axiom beyond `propext`, `Classical.choice` and `Quot.sound`; `Spot/OneStage.lean`, `Spot/Run.lean`, `Spot/Pieces.lean` and `security/Properties/Spot.lean` hold no theorem.
- Kernel replay: no. The records run is the fast path, which does not replay; the candidate's Lean check does.
- Red-teamed by an agent other than its prover: no.

## The lock

Both locks are committed unedited, as a records run wrote them (`python3 tools/verity/lean/lean_changed.py --records security --update`). They were written by run `r20261113-015226-cc14` before `main` was merged in; the merge left both files as they were, and the records run at the head, `r20261113-030458-e51a`, passes on them without regenerating either.

What moved against `main`, in plain words:

- **No guarantee's own record moves**, in either lock. `Properties.Spot.acceptMeans` keeps its signature and both of its hashes: its own text is what it was. The lists of witnesses, of guarantees without witnesses, of deployed guarantees and their trust gaps, of assumptions and of domain premises are `main`'s.
- **What `Properties.Spot.acceptMeans` reads moves**, which is the statement above. From this change: `Properties.Spot.ExpectsV2`, `R1`, `R7` and `Registered` changed, and `Properties.Spot.CertifiedHeld` is new. From #9023's commits: `Properties.Spot.UnitsProved` changed, with `OpenedRegistered` in place of `SameRoots`.
- **One open premise's digest moves**: `Properties.Spot.ExpectsV2`, the definition widened above. It is the open premise it was, with its owner and its due date.
- **The compiled check's definitions move** in what thirty guarantees read, none of whose statements moves. From this change: `Spot.Registration.Verifier` (with its new field `certified`), `check`, `checkOutputs` and the new `certifiedHeld`; `Spot.OneStage.check`, `paramsOfJson`, `paramsParsed`, `recordChecks` and the new `CERTIFIED_KEYS` and `CERTIFIED_REFUSED`; the new `Spot.Record.certifiedCheck`. From #9023's commits: `Spot.Record.checks` and the new `subDict`. The thirty are `Properties.Spot.acceptMeans`, `bernoulliBounds` and `digestIsRecord`, `Properties.Spot.SecurityProofs.tablesRegistered`, `Properties.Remote.SecurityProofs.acceptComplies` and `acceptInferenceOnly` with its witness, and the sampled game's `Verity.coins_couple`, `remote_soundness_of_subprotocols`, `soundness_at_of_componentBounds`, `soundness_at_of_targets`, `soundness_of_subprotocols`, `soundness_of_upFront`, `target_of_periods` and `wipe_target` with their witnesses.
- **The complete answer's lines** (#9458, held here until it lands): `Properties.Spot.publicOf`, `Spot.OneStage.Public` with its new field `complete`, `publicJson`, and the new `Spot.Record.allRan`.

Nothing else differs from `main`'s locks.

## Runs at the head (`a287d4bf124738d270ac605e153805169f5d73ee`)

- **Records** (`python3 tools/verity/lean/lean_changed.py --records security --update`), run `r20261113-030458-e51a`: `security` `check.py` exit 0 and `security/Properties` `check.py` exit 0, both audits passing on the committed locks, nothing regenerated. It is the fast path, with no kernel replay.
- **Tries** (`python3 tools/verity/lean/lean_changed.py --try FILE --siblings`, the nine files), run `r20261113-030550-5811`: exit 0.
- **Quick tier** (`uv run tools/verity/check/suites.py --changed 4a7291116e11 --quick`), run `r20261113-030533-7f6f`: 73 suites, 72 passed, 1 failed. The failure is the `research` suite, two tests of `tests/test_auto.py` (`test_the_probe_reads_live_claims_and_waiters_and_runs_the_trees_slot_py_on_the_pool` and `test_a_node_2_that_takes_runs_only_through_a_check_slot_is_never_chosen`). The same two fail on that machine when the suite is run alone at `main` (run `r20261113-012506-c977`), and they read no file this change touches. Among the passes: `verity/protocols/spot/verifier` 920, `benchmarks/one_stage` 1,037, `verity-network-traces` 679, `verity-core-sampled-proofs` 325 (2 skipped), `verity/protocols/certifier/verifier/network` 90, `repository` 382.
- **Retired terms** (the sites `tests/lint/test_repo_retired_terms.py` counts, file by file, `main` at `4a7291116e11` against the head, on the 33 files other than the two locks): no file has more sites than on `main`, six have fewer, and the total goes from 3,947 to 3,933. This change's own commits add 20 sites and remove 21.
- **Markdown caps** (`python3 tools/verity/check/gates/markdown_caps.py origin/main HEAD`): one file over its cap, the verifier's README, as said under "What it does not do".
- **A test to know of.** `benchmarks/one_stage/tests/test_kept_compose_cache.py::test_without_v3c_compose_cache_the_warm_and_the_audit_get_the_windows_own_directory_as_before` failed once in an earlier quick tier of this branch (run `r20261113-020205-35be`), where the runner marked it nondeterministic. The file is `main`'s and unchanged here; it passed in the run above.

## Merged tree

`main` (`4a7291116e11`) is merged in, by a merge commit with no conflict; it changed one of this change's files (`lean/tests/test_difftest.py`, where `main` adds a test of its own) and neither lock. `git merge-tree --write-tree` of the head merges clean with `main` as it has moved since (`8ef93f01963c`), with #9458's head `049c5875989c` and with #9023's head `6b782d972ec4`.

