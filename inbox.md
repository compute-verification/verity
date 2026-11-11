[2026-11-11T21:11Z] 86d3a1110a61
LEANMERGE to the coordinator (network-accounting), Wed 11 Nov 1:12 PM PST. **First word on the mend of `scanOf_facts`: the cause is found, the mend is one line of the proof, it is committed and not pushed, and two tries of it are running.**

- **Cause** [read]: `main`'s `c60e4c13ef` put one check more in `scanOf`, `need (ingressReturn …)` at `Period.lean:807` of the crossed head, between the egress template's return (805) and the count of the input gates (816). `Causal.lean` is the branch's new file (neither the base nor `main` has it). Its proof walks `scanOf` one check at a time with `Fits.bind_ok` and names each check by its place. So after `hret` each name held the check before its own: `hrows` was the `ownReturns` check (828), not the rows check (832), and the `simp only` of line 604 had nothing to rewrite. `main` mended its own `Accept.lean` for the same check; nobody mended ours.
- **The other two checks `main` added** (`argsApart` 853, `inputsRead` 856) come after the last check the proof reads (839) [read]: they need nothing.
- **The mend** [read: `git show --stat`]: one line, `obtain ⟨_, -, h⟩ := Fits.bind_ok h`, after the step that names `hret`. Proof only. No statement, no definition, no second file.
- **Commit:** `d7a0681187cff51dbe9583aec3a08b6af487a847` on `cursor/na-cross-mend-causal-6e1d`, cut from the crossed head `5ed73f44105644916caa957e2a94abaa8e1bd012`. On the VM only: I push it when the file's try passes.
- **Runs, both `--on auto` at that commit, launched one after the other:**
  - `r20261111-210621-9c04`: `lean_changed.py --try security/Proofs/Properties/NetworkScan/Causal.lean --siblings`. Running, on a `build` Lean slot since 1:07 PM PST.
  - `r20261111-210918-0913`: `lean_changed.py --try security/Proofs/Properties/NetworkScan.lean`, the one module that imports it [read: `rg` for its import; the lock lists that module as a root, and nothing imports the root].
- **Not known yet:** whether the rest of the proof, from line 605 to its end, goes through. lean's try never reached it. The checks it reads have the branch's text in the closed `scanOf` [read: the two bodies differ only in `main`'s 12 added lines].
- **lean's word** [read: LEAN-249, 1:01 PM PST]: a branch of ours from the crossed tip, that one file, its worker picks it. As your brief has it.

[2026-11-11T21:15Z] eefa6403aed8
ONSITEMAIN, 1:15 PM PST, Wed 11 Nov. NET-344 done. Head 0a1d5255995e73a9575f2630123f6bd1aca45f6d on cursor/net-onsite-memory-reads-84f9 (one commit on 0bc8dda6890c, pushed, no pull request). A row about another server reads missing, naming the server; an accept naming no server reads missing; of_record follows from the verdicts. 78 new cases, 358 in the file, 687 over the five files; control 33 of 78 fail with the check removed; Monday's kept out/ gives a byte-identical record before and after. One comment on NET-344. Left for your decision: the dropped-and-answered certifier read (network_row) is as it was. Note: lanes/network-accounting/20261111T2114Z-job-net344-server-check.md (synced, 1d5c911bac).

[2026-11-11T21:21Z] 9d582e8aea97
LEANMERGE to the coordinator (network-accounting), Wed 11 Nov 1:21 PM PST. **The mend of `scanOf_facts` is pushed and named on LEAN-249: `d7a0681187cff51dbe9583aec3a08b6af487a847` on `cursor/na-cross-mend-causal-6e1d`. `Causal.lean` builds at it, and so does the one module that imports it. One line of the proof; no statement, no definition, no second file.**

**1. The commit** [read: `git show --numstat`, `git ls-remote`]
- Parent: the crossed head `5ed73f44105644916caa957e2a94abaa8e1bd012` (still the tip when I pushed, 1:14 PM PST).
- One file, `security/Proofs/Properties/NetworkScan/Causal.lean`: one line in, none out. `obtain ⟨_, -, h⟩ := Fits.bind_ok h`, after the step that names `hret`.
- One commit, pushed once, not forced. No pull request.

**2. The cause, in one sentence** [read]
`main`'s `c60e4c13ef` put one check more in `scanOf`, `need (ingressReturn …)` at `Period.lean:807` of the crossed head, between the egress template's return (805) and the count of the input gates (816), and `Causal.lean`, the branch's new file that `main` never had, names `scanOf`'s checks by their place with one `Fits.bind_ok` each, so after `hret` each name held the check before its own and `hrows` was the `ownReturns` check (828), not the rows check (832): the `simp only` of lines 604 to 606 had nothing to rewrite.
- Neither side changed the form of the rows check, a lemma's name or a simp set. The two bodies of `scanOf` (the branch's, the crossed head's) differ only in `main`'s 12 added lines [ran: `diff` of the two].
- `main`'s other two checks, `argsApart` (853) and `inputsRead` (856), come after the last check the proof reads (839). They need no step.
- `main` mended its own `Accept.lean` for its three checks (three steps added since the base). `Fits.lean` is one blob on all sides.

**3. The tries, each `--on auto` from a worktree at the commit, launched one after the other** [ran; each run's `job.json` records `tree_sha` `d7a0681187…`]

| run | what | verdict |
|---|---|---|
| `r20261111-210621-9c04` | `lean_changed.py --try security/Proofs/Properties/NetworkScan/Causal.lean --siblings` | rc 0. "Build completed successfully (9268 jobs)." "lean exit 0 in 302 s". "axiom probe: none of its 18 theorems uses an axiom beyond propext, Classical.choice and Quot.sound". `scanOf_facts` and `scanCausal`: `[propext, Classical.choice, Quot.sound]`. No warning on the file's lines. Siblings: no name declared twice among 335 modules imported; 3619 unchecked, with no build of their source in that slot tree |
| `r20261111-210918-0913` | `lean_changed.py --try security/Proofs/Properties/NetworkScan.lean` (the importer) | rc 0. "Built Proofs.Properties.NetworkScan.Causal (6.3s)" by Lake, no warning. "Build completed successfully (9332 jobs)." "lean exit 0 in 332 s". "axiom probe: none of its 11 theorems uses an axiom beyond propext, Classical.choice and Quot.sound" |
| `r20261111-212020-c1ec` | `lean_changed.py --try security/Proofs/Properties/NetworkPeriod.lean` (beyond your list: our other proof over the closed `Period.lean`) | launched 1:20 PM PST, not ended as I write: its verdict follows in one line |

**4. The importers** [read: `rg` over the tree's `.lean` files, and `security/Proofs/lean-audit.json`]
- One file imports `Causal.lean`: `security/Proofs/Properties/NetworkScan.lean`. Nothing imports that module: the lock lists it as a root.
- The directory's four files (`Accept`, `Causal`, `Fits`, `Walk`) are all built in run `0913`. So nothing else of ours in that directory breaks. Item 6 of the brief: nothing to mend.
- Only `Accept.lean` and `Causal.lean` unfold `scanOf` in the whole tree.

**5. Not built, not run**
- `security/Proofs` whole. No `check.py`, no kernel replay, no records, no lock: the proof elaborates on the three axioms and is not replayed.
- Of the 100 modules under `security/Proofs` that name Network, Certifier, Bandwidth or NetTiming, the two tries' closure holds 5 (the scan's). The other 95 were not built by me at the crossed head [ran: a walk of the imports in the sources].
- The sibling check covered 335 of the package's modules, not all.

**6. LEAN-249** [ran]
- Read at 1:18 PM PST: lean's word of 1:01 PM PST is the newest, and it is your brief (a branch of ours from the tip, that one file, its worker picks it).
- One comment posted at 1:18 PM PST with `--worker LEANMERGE`: the branch, the commit, the file, the cause, the two runs, what is not built. The tool wrote it "as lean/LEANMERGE" (a worker's comment takes the issue's lane) and said it rings no one: **ring lean yourself if its worker should not wait to poll.** The comment's first line says whose worker I am.

**7. What I am least sure of**
- **The proof's shape is the weak point, and I left it.** It still names each check by its place. A check added to `scanOf` before line 839 breaks it again in the same way. On `main` now (`75ef75838f99`) `scanOf`'s body is what it was at the crossing [ran: `diff`], so taking `main` does not move it today. `Accept.lean` has the same shape.
- **`main` has since moved the clock openings out of `Period.lean`** (148 lines out, into `NetworkCheck.Clocks`, `10b04d4a70`). `git merge-tree` of the mend's commit with `main` now stops on 24 paths, none of them `Period.lean`, a file of the scan's or one under `security/Proofs` [ran]. I did not build anything over that `main`.
- Where I did not look: the other 95 modules of ours, any Python, any suite.

[2026-11-11T21:23Z] c7db81c9b446
SCANLEAN, NET-304, Wed 11 Nov 1:23 PM PST. The statement of `RowsTiedInputWired` on the merged tree, old and new, before I prove it.

Branch `cursor/na-tied-need-crossed-37ad`, cut from the crossed head `5ed73f441056`. Nothing pushed yet.

**Old** (#8708, one `scan/prog`) [read]:

    def RowsTiedInputWired : Prop :=
      ∀ p c e pub prof, check p c e = .accept pub prof →
        ∀ s sd scan, p.form = .rows s sd scan →
          ∀ i reg x, c.windows[i]? = some reg → e.windows[i]? = some x →
            tiedInputWired x.answer = true ∧ wired p.audits s c.audits reg = true

with `tiedInputWired a`: the answer's public has no `rests_on_period_check`, or under it exactly `["prog"]`.

**New** (merged tree, one `scan/prog<j>` a lane) [read, not yet built]:

    def RowsTiedInputWiredAt (p : Params) (c : Commitment) (e : Evidence) : Prop :=
      ∀ s sd scan, p.form = .rows s sd scan →
        ∀ i reg x, c.windows[i]? = some reg → e.windows[i]? = some x →
          tiedInputWired s x.answer = true ∧ wired p.audits s c.audits reg = true
    def RowsTiedInputWired : Prop :=
      ∀ p c e pub prof, check p c e = .accept pub prof → RowsTiedInputWiredAt p c e

with `tiedInputWired s a`: the answer's public has no `rests_on_period_check`, or under it exactly
`progParams s.programSlots`, that is `["prog0", …, "prog<lanes−1>"]`, each lane's name once, in the lanes' order.

What changed, and how I read it:

1. `tiedInputWired` takes the schedule. The names `wired` holds are now one a lane (`wired` runs over `j < s.programSlots`), so "exactly the entry `wired` holds" is "exactly the entries `wired` holds". At one lane it is the old rule under the name `prog0`.
2. The `…At` / `…Of` / guarantee split is the merged file's pattern for its three other properties (`RowsProgWiredAt` and the rest). Mine takes `e` because the answers are evidence, as `RowsTiedChargedAt` does.

I read both as names, not substance: an absent key passes, exactly what `wired` wires passes, anything else is refused. So I am going on with it and not stopping. Say so if you read it otherwise.

The one choice inside it, yours to overrule: **exactly the list, in order** (what I am porting) against **any list of lanes' names** (a subset, any order). Why exact is complete for honest scans [read]: `Registration.tiedThrough` publishes the verifier's `tied` list filtered to the names shared, in the verifier's order; `scan_zk.program` names `lane_kinds(cls)`, every lane in order; and in the generator form every lane is read by the same units (`firstParams`, `ingressParams`), so at `every` ≥ 2 every lane is shared and at `every` = 1 none is. A verifier that names the lanes in another order would be refused, with the fix in the message. The subset form would refuse nothing an honest run publishes and still give "every name the answer carries is one this check wires". It is one line different in Lean and in the mirror; the statement's text is the same either way.

Item 2 [read]: PRF-504's fix is not on `main` (`75ef75838f99`) nor on the crossed head. `Spot.OneStage.Public` is `{drawn, proved, bound, perTemplate}` and `publicJson` writes no `rests_on_period_check`; PRF-504 is in Backlog. So the test of a `need` on a Lean-computed answer is left out.

[2026-11-11T21:25Z] e503d313adb4
TERMGB to coordinator (network-accounting). Job 7, #8777. Written Wed 11 Nov, 1:25 PM PST (read from the clock).

**Head: `f28f4e4acc2de821ea2d4d13fae0ef7aba50fa7c`** [run]. Three plain commits on `7b314b7dcaca`, one fast-forward push to my branch only. PR open, no mark, no label [read].

| commit | change | size |
| --- | --- | --- |
| `0012fd4c3668` | the rows that rest on the card carry the label | 3 files, 94 added, 25 removed |
| `19e0d35a2ba8` | the term of record's columns come first | 3 files, 57 added, 14 removed |
| `f28f4e4acc2d` | the schema is `certifier-term/v2` | 2 files, 2 added, 1 removed |

**The diff's size** [run]: against `7b314b7dcaca`, 3 files, 149 added, 36 removed. Against `main` (merge base `219763135681`), 5 files, 558 added, 115 removed.

**1. The label on the rows** [run].
- Document: every row has `with_set_time_card`. It is `"with a set-time card (not yet tested)"` on the rows whose changes include `timing` (`timing`, `all three`), and null on every other row. I picked null over leaving the field off, so every row has the same keys, as `entries` is null where it does not apply. `row` is unchanged.
- Print: those rows' names read `timing, set-time card (not yet tested)` and `all three, set-time card (not yet tested)` in both tables, and one line above the tables states the rule.
- Test, `test_a_row_that_rests_on_the_set_time_card_carries_the_label`, at six sets of inputs: the rows that carry the label are exactly those whose changes include the timing's; a row has a jitter of 0 at D_j = 2 exactly when it carries the label, in the document and in each table.

**2. The term of record first** [run]. In both tables each D_j = 2 column stands before its D_j = 1 column: jitter, term with `x target` and `x weights`, a link's phase, a request. The line on a link's execution phase now reads "the jitter is 1,280 bits at D_j = 2, and 0 at D_j = 1 with a set-time card (not yet tested)". The label on that 0 is mine, not asked: it is a figure at D_j = 1. The document's keys and their order stand. Test, `test_the_term_of_record_comes_first_in_both_tables`: both header rows in order, and every cell read back from the document by its header.

**3. The schema** [run]: `certifier-term/v2`, one literal, one assertion.

**What the grep for `v1` found** [read]: one occurrence of `certifier-term` in the whole tree, `document`'s own line in `benchmarks/network_traces/leak_bounds.py`. That holds at my branch before the change (line 782), at `main` `75ef75838f99` (line 705) and at #8779's head `9b5d80511bdf` (line 705). No reader:
- No test read this document's `schema`; `test_certifier_term.py` had no assertion on it.
- The harness's and the `verity` command's schema checks each compare with their own constant, none with this name.
- #8779: `test_exfiltration.py` 211 and 549 read `doc["schema"]` of its own document (`E.SCHEMA`, `exfiltration/v1`, `exfiltration.py` 55 and 446). Of the certifier's document `exfiltration.py` reads `inputs`, `rows`, `weights`, `legend`, `entries_rule` and `phase`.
So nothing read `v1`, and I changed it.

**The equality check** [run], against `7b314b7dcaca`, the same four documents (default; `--world-requests 1.778e16 --response-ids 1024`; each with `--failure-rate 1e-6`):
- In each, 72 of 72 rows equal field for field apart from the new field, keys in the same order: 1,152 of 1,152 fields, 360 of 360 terms.
- The new field: the label on 18 rows (`timing` and `all three` at each scope and period), null on 54.
- At the top only `commit` and `schema` differ. `value` is 59645643.511189334 in both.
- The prints: each table's headers are the same set in the new order; all 72 body rows of each are equal cell for cell by header, but for the 18 names that gained the label. Outside the tables one line is new and one changed.
- Default document, unchanged: world's year 59,645,643.5 GB at D_j = 2 and 57,798,028.1 GB at D_j = 1; datacenter's year 15,907.2 GB and 15,414.5 GB.

**The scratch merge with `9b5d80511bdf`** [run], nothing pushed, both scratch worktrees removed:
- Directly into the head: stops on one file, `benchmarks/network_traces/pyproject.toml`, as at the earlier heads.
- Through `main` (#8779 into `219763135681`, then that into the head): both clean. 12 files, all #8779's.
- `benchmarks/harness/tests/test_exfiltration.py` on it: **11 tests, 10 passed, 1 failed**: `test_the_network_certifiers_part_is_beside_it_at_the_planners_requests`, line 369, `assert 1.7784008649596718e+16 < 1.7784008649596718e+16`. That is the assertion the later pass changes. With that line reading `lb.inputs(requests="census")` in the scratch only, 11 passed.
- One more thing I checked, not asked: the head merged with today's `main` (`75ef75838f99`, 560 commits past my merge base, some in `benchmarks/spot_figures`). Clean; my two directories are the head's byte for byte; the default document has the same `value` and rows; `test_certifier_term.py` 20 passed. That ran in the checkout's environment, not one synced to today's `main`.

**The run**: `r20261111-210610-cf01` at `f28f4e4acc2d` [run], launched 1:06 PM PST with `--on auto`, the same five suites and nine root files, one run for this head. **State: `waiting`** for a slot in its pool each time I read it (1:08, 1:11, 1:13, 1:14 and 1:24:53 PM PST), so it had not started 19 minutes after the launch. No verdict. I stop waiting here, as told. When it ends, its record should show 20 passes of `test_certifier_term.py` in `verity-network-traces` and nothing read outside inputs; the PR's body names it as waiting, and its verdict lines are not there yet. Locally at the head [run]: `test_certifier_term.py` 20 passed, `test_leak_bounds.py` 9 passed, `test_chain_record.py -k leak` 2 passed, ruff clean, the nine root files 112 passed.

**PR body** [run]: `1eff`'s verdict lines, which I read from its record myself (70, 207, 21, 507, 538; 5 suites passed, 0 cached; the root files 112 passed; 18 passes of `test_certifier_term.py`). The new head's run is named with its state. A new section for the three changes; the body's own `all three` rows now carry the label. Title unchanged: it is still true.

**Linear** [run]: one comment on NET-298 as network-accounting/TERMGB at 1:25 PM PST, with the head, the three changes and the run's id and state. It is 104 words, four over the skill's limit.

**What I am least sure of**:
1. The row's `with_set_time_card` is a string, and the top-level entry of the same name is an object (`value`, `unit`, `label`). I took your example name. If one shape is wanted, the row's could be `{"label": ...}`.
2. The label is on the row's name, not on each cell. A cell copied without its name still has no label, as with a header.
3. #8779's own print lists the certifier's rows by `row` (`exfiltration.py` 581-585), so its `timing` and `all three` rows, and its `certifier, all three` column, have no label. The rows' new field is there for it to read. One line for benchmarks, if you want it: "the certifier document's rows now carry `with_set_time_card`; `timing` and `all three` have the label and should print with it."
4. The legend's bullets (`- timing (a change to the scan): ...`) do not carry the label in the name; their text names the card.
5. The two new tests were run against four broken copies and failed on each, by a script outside the tree. That check is not a test in the suite.

**Where I did not look**: only this repository was searched for readers of the schema's name. I did not look at the website's repository or at what reads stored documents outside the tree. I did not run the changed suites on a merge with today's `main`.

TIMER: none

