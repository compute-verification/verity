[2026-11-15T09:01Z] 4fec80548673
TERMGB2 to network-accounting (lead): UNSTATED done. New head of `cursor/na-orphan-roots-b51f`: `d00e12bd5ed9`, one commit on top of `ab6ebc361d87`, pushed only there at 1:01 AM PST.

1. The rule [read]. It needs the build's facts (`spec_definitions` from `Facts.lean`), so I read it and applied it by hand to the declarations of our files.
- **The rule** (`check.py`'s `shape`): a name is flagged `shape.unstated` when all of these hold:
  - it is in `spec_definitions` and has `statement` true;
  - its module is under `shape.statements`, which is `["Properties"]` in this lock;
  - it is not lifted by a guarantee's pin, not a listed premise, and has no `shape.frozen` entry.
- **What `spec_definitions` holds** (`Facts.lean`): the definitions declared by audited modules under `spec`. `spec` is the lock's `layers` keys (`spec_of`). `statement` is `ci.type.isProp`: the declaration's type is literally `Prop`, so it must be a closed `def G : Prop`. A predicate with parameters is not a statement.
- **Witnesses:** nothing in the rule treats `*Honest` or `*Nontrivial` differently. A closed `Prop` witness is flagged like any other statement.

2. What the rule flags of ours.

Flagged: four names in `security/Properties/InferenceOnly/PinnedRead.lean`, namespace `InferenceOnlyCheck.Properties.Record`. The module is under the layer key `Properties.InferenceOnly` and under `Properties`.
- **`PinnedReadBinds`** (line 116) is proved [read] by `theorem pinnedReadBinds : PinnedReadBinds` at `security/Proofs/Properties/InferenceOnly/Lemmas/Opening.lean`, line 322, namespace `InferenceOnlyCheck.Properties.Record.SecurityProofs`. The file has no `sorry`.
- **`PinnedReadHonest`** (line 154) is not proved. No theorem has it as its type, and Opening's docstring calls it a "statement not proved".
- **`PinnedReadNontrivial`** (line 133) is not proved, likewise.
- **`OpenedRowReturned`** (line 175) is not proved. No theorem has it as its type.

Not flagged:
- **`PinnedReadAt`, `PinnedRead` and `ReturnsRow`** take parameters, so they are predicates: their type is `… → Prop`, not `Prop`.
- **`RecordForm.form`** has type `Properties.Record.Form`, not `Prop`.
- **`Properties.Certifier.Registered`** (`RegisteredLinkRated`, `RegisteredLinkRatedHonest`, `RegisteredLinkRatedNontrivial`): its module is under no `layers` key [ran] at my head, at lean's newest head `1f1e14dd72a7` or on memory-accounting's `90a78477e756`. So it is outside `spec`, Facts don't list it, and the rule can't flag it. If a `Properties.Certifier` layer key is added later, those three closed `Prop`s will flag. `registeredLinkRated` proves `RegisteredLinkRated`; the two witnesses have no theorem.
- **`Proofs.Properties.Certifier.Registered`** holds only a theorem, and `specDefinition` excludes theorems.

3. The added lines. They sit in `security/Properties/lean-audit.json`, `shape.frozen`, immediately before `InferenceOnlyCheck.Properties.DescCountsCK`, with the other inference-only `unstated` entries:

```
+   "InferenceOnlyCheck.Properties.Record.PinnedReadBinds": "unstated: stated in the properties tree and proved by InferenceOnlyCheck.Properties.Record.SecurityProofs.pinnedReadBinds in Proofs.Properties.InferenceOnly.Lemmas.Opening, listed as a guarantee after the landing, NET-451, Wed 18 Nov",
+   "InferenceOnlyCheck.Properties.Record.PinnedReadHonest": "unstated: PinnedReadBinds's witness that the check's own form has a pinned read, stated and not yet proved, so no guarantee states it; decided after the landing, NET-451, Wed 18 Nov",
+   "InferenceOnlyCheck.Properties.Record.PinnedReadNontrivial": "unstated: PinnedReadBinds's witness that at the verdict bit 0 some pinned read does not return its table's row, stated and not yet proved, so no guarantee states it; decided after the landing, NET-451, Wed 18 Nov",
+   "InferenceOnlyCheck.Properties.Record.OpenedRowReturned": "unstated: that an opening's declared row is the row its reader returns, stated and not yet proved, so no guarantee states it; decided after the landing, NET-451, Wed 18 Nov",
```

4. Rechecks [ran]:
- **Lock:** check.py's `load()` loads it. check.py's `category()` gives `unstated` for all four, and none begins with an `UNLISTED` head. check.py's writer `text()` leaves the four lines unchanged.
- **Stale entries:** each name is a closed `Prop` in a module in `roots`, under `spec` and under `shape.statements`, so `shape.frozen-stale` can't fire. None is a guarantee or a listed premise.
- **Merges:** clean pairwise and stacked with lean's newest head `1f1e14dd72a7`, ours `70863c040628` and memory-accounting's `90a78477e756`. The stacked lock parses and holds our four and memory-accounting's two.
  - `70863c040628` touches the `security/Properties` lock too, not only `security`: it adds ten `Properties.NetworkPeriod.*` `unstated` entries near the top of `shape.frozen` and one `shape.properties` entry. Mine are about 80 lines below, with no key in common.
- **Tests:** `tests/test_lean_packages.py` and `tests/test_repository.py`: 51 passed.

No Lean build, Lean check or records run, so the rule's real output on the built facts is not run. The flag list is the rule applied by hand to the sources.

