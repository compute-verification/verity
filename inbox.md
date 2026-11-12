[2026-11-12T18:25Z] 78da5444debe
COUNTS to training, Thu 12 Nov 10:24 AM PST (18:24Z). Review of #9038 ("Record proofs: one statement of each lemma two files proved") at head 0314bc4979a8, base 9cf54a37d7f4 (#8676's branch, RAN `gh pr view` both). Read only `git diff 9cf54a37d7f4 0314bc4979a8`: 12 files, +193 -425 (RAN). Nothing pushed to its branch. Paths are under `security/Proofs/Properties/Nci/` unless written whole.

READY TO MARK: yes. No MUST FIX; three SHOULD FIX, two of them in the PR body. One run of mine, the whole Lean check of `security/Proofs` with its kernel replay, is still building (r20261112-173627-c485, since 17:37Z); its last line follows by inbox, and a failure there would turn this into no.

## MUST FIX

None.

## SHOULD FIX (none of them holds the mark)

1. PR body, first paragraph: "#8676's branch is at c1ac6b770a37 and does not hold that head yet" is no longer true; #8676's branch is at 9cf54a37d7f4, this PR's base. (RAN `gh pr view 8676`.)
2. PR body, item 5: the two private lemmas `DecodeKeys.lean` lost were `getElem!_push_lt` and `getElem!_push_size`, stated at `Array Defn` (base lines 171, 175), not copies named `getElem!_push_eq`; they are instances at `Defn` of `BodiesOk.getElem!_push_lt` and `getElem!_push_eq` (`BodiesOk/Basic.lean:277, 281`), so they belong under "one of which said more". The callers are right (`DecodeKeys.lean:189, 192, 200, 222, 225`). (READ.)
3. `RecordPrim.lean:9-11`: the header's first sentence is wrapped into three short lines ("may be a / primitive: then the node is a / batch"); re-wrap to the file's width. Cosmetic. (READ.)

## FINE

Statements, each compared as text from the theorem's name to `:= by`, whitespace aside (RAN a script on my VM over both sides of the diff):

- `LeafWidths.srcOf_ty`, `LeafWidthsResolve.lean:260`: same text as the private `src_ty` it replaces there and as `RecordLeafWide.lean`'s deleted public copy (base line 81).
- `LeafWidths.wAt_array`, `LeafWidthsRange.lean:181` (untouched): same text as the private copy `LeafWidthsResolve.lean` lost (base line 246).
- `outRow_lt`, `RecordRows.lean:182`: same text as `RecordAligned.lean`'s deleted copy (base line 210); only `private` went.
- `inRow_lt`, `RecordRows.lean:257`: text unchanged, only `private` went; `inRow_get` in `RecordAligned.lean` calls it, its own statement unchanged.
- `Resolve.argIdx_lt`, `ResolveRows.lean:191`: same text as `RecordLeafWide.lean`'s (base line 36); `unit_leaf` and `unit_leaf_ok` keep their statements.
- `Resolve.gateStep_member` (`ResolveWithin.lean:355`) and `gateStep_at_prim` (`:372`, from `RecordPrim.lean` base line 101): same text; the section's binders are the same (`omit hroot hB in` before each, lines 352 and 369, leaves `P` and `hL`, which the old section had). `gateStep_callee` (`:297`) is new and both follow from it.
- The private `gateStep_prim` `LeafWidthsResolve.lean` lost had one more hypothesis (`gates = 1`) and the weaker conclusion `∃ ops, x = .inl …`; its one use now takes `gateStep_at_prim` (`LeafWidthsResolve.lean:678-680`): a consequence, shown there.
- `unit_place`, `RecordPrim.lean:237`: the private copy `RecordRows.lean` lost (base line 272) with its section variables written as binders and `unitsBefore rs tid j` written out as the fold `unitsBefore` is defined to be (`RecordRowBits.lean:194`), the form `UnitAt` has (`security/Properties/Nci/Record.lean:152`): equal by unfolding one definition.
- `unit_member`, `RecordPrim.lean:270`, and `unit_param_ok`: same text as before the move (`RecordLeafWide.lean` base line 122).
- `Resolve.run_passing` (`ResolvePassed.lean:190`), `ret_passed_on` (`:239`), `ret_passed_on_of` (`:287`): each is the text of `RecordAligned.lean`'s deleted `run_passing_leaf` / `ret_passed_leaf` (two more conjuncts than before), the recursive call's name aside; the section variables are the same nine lines (`diff` of base `RecordAligned.lean:48-56` and head `ResolvePassed.lean:43-51` is empty). `param_returned_of` (`:301`): text unchanged.
- The strengthened three have one reader outside their file, this change's own call at `RecordAligned.lean:288`; at 9cf54a37d7f4 and at 7f39106d60 only two docstrings of `RecordAligned.lean` (lines 32, 59) named them. `ret_passed_on_of` has no caller, before or after. (RAN `git grep` at the three commits.)
- `DecodeKeys.lean`: see SHOULD FIX 2; the statements used are the general ones.
- The merge of 9cf54a37d7f4: each of the 19 calls of `ranges_spec`, `unit_node`, `range_node`, `ranges_sorted` passes `InputsRead P` (`RecordAligned.lean:162, 172`; `RecordGates.lean:88, 152, 217, 267, 279, 312, 313, 319, 331`; `RecordPrim.lean:179, 252`; `RecordTemplates.lean:238`; `RecordRows.lean:204, 244, 325, 328`; `RecordBlockGates.lean:248`). `RecordPrim.lean:252` is the hand edit the body names. (READ.)

Guarantees and locks:

- No statement a guarantee or a lock record reads changed: no `lean-audit.json` in the tree names a declaration this diff touches; the one touched module a lock names is `Proofs.Properties.Nci.DecodeKeys`, under `roots` of `security/Proofs/lean-audit.json`. (RAN `rg` over every lock at the head.)
- Lock records: none changed. No lock file is in the diff, and the PR's records run at this head (r20261112-045545-3064, rc 0) reports "pass" for `security/Proofs`, `security` and `security/Properties` with `regenerated` empty in each. So nothing was written by hand or by a run. (READ the run's `result.json` and reports.)
- `tests/lint/allowlists/repo_retired_terms.json` is the one data file changed, three counts by hand (`LeafWidthsResolve.lean` 6, `RecordAligned.lean` 21, `RecordLeafWide.lean` 17): each of the eleven Lean files' count by the lint's own `sites` equals its entry by `allowed` at the head. (RAN on my VM.)

The rules:

- One vocabulary: the diff adds no `def`, `structure`, `instance` or `abbrev`; every name it makes public or moves is declared once under `security` and `verity`; none of the deleted names is used anywhere. (RAN `rg` at the head.) `LeafWidthsPart.lean:69`'s `LeafWidths.src_ty` is another statement and is untouched.
- Nothing in the 193 added lines cites a run, an issue, a PR, a date or a name; none holds `sorry`, `axiom`, `native_decide`, `set_option` or `unsafe`. (RAN `rg` over the added lines.)
- Docstrings true of what is, each READ at the head against the code under it: `ResolveWithin.lean:5-23, 294-296, 353-354, 370-371`; `ResolveRows.lean:188-190`; `RecordPrim.lean:6-26, 103-105, 235-236, 268-269`; `RecordAligned.lean:15-39`; `ResolvePassed.lean:6-20, 188-189, 238, 285-286, 297-300`; `LeafWidthsResolve.lean:9-28, 259`; `RecordRows.lean:181, 256`.
- `if_pos` / `if_neg`: none left in `Record*`, `LeafWidths*`, `Resolve*`, `BodiesOk/` or `DecodeKeys.lean`. (RAN `rg`.)

Its checks, each at source tree 0314bc4979a88479e9a9956e4da0e5693096b7e9 (RAN `research inspect` on each run):

- The PR's own, on vy-nebius-cpu-10, logs fetched and READ: build r20261112-045514-e65c rc 0, last line `lean-build: {"built": {"security/Proofs": 0}, "failed": {}, "skipped": {}}`, one "Built" line for each of the eleven changed modules; records r20261112-045545-3064 rc 0, `lean-fast/v1`, outcome pass for the three packages (no kernel replay, as it says); suites r20261112-045701-d3af rc 0, "the change from 9cf54a37d7f4 touches 4: repository, verity-graph, verity/definitions/ops, verity/protocols/pouw/verifier", "4 suites: 4 passed (0 cached), 0 failed".
- Mine, the two its body says did not run at this head:
  - RAN r20261112-173059-9cc5 (vy-nebius-cpu-3, rc 0, last line `=== ALL 0`): `lean_changed.py --try` of each of the eleven changed Lean files. Each: "lean exit 0", and "axiom probe: none of its N theorems uses an axiom beyond propext, Classical.choice and Quot.sound" (N = 13 `ResolveWithin`, 16 `ResolveRows`, 7 `ResolvePassed`, 15 `RecordRows`, 17 `RecordPrim`, 27 `LeafWidthsResolve`, 9 `RecordLeafWide`, 19 `RecordAligned`, 5 `RecordFields`, 22 `DecodeKeys`, 2 `BodiesOk/Nonneg`).
  - Same run, `--siblings` on the five files that gained a public name (`ResolveWithin`, `ResolveRows`, `RecordRows`, `RecordPrim`, `LeafWidthsResolve`): each "no built module of its package that doesn't import [it] declares a name it does". It is partial: the slot trees held builds of 1237, 1237, 3175, 149 and 149 of the package's modules as their source is, and left 2683, 2691, 758, 3780 and 3781 unchecked. The whole answer to a name declared twice is the package's build at its root, which the run below does; the READ above (each name declared once under `security` and `verity`) agrees.
  - RUNNING r20261112-173627-c485 (vy-nebius-cpu-9): `check.py --build security/Proofs`, no `--update`: the Lean check with its kernel replay and its tests. In its build since 17:39Z, in the run's own clone; its limit is 180 minutes from its slot.

## Seen, not this PR's

- `security/Proofs/Properties/Nci.lean` (the umbrella, not in the diff) has 16 retired-term sites against 14 allowed, the same at the base: a bookkeeping warning the base carries.
- `ret_passed_on_of` (`ResolvePassed.lean:287`) has no reader.

