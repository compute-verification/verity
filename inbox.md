[2026-11-12T01:54Z] 6aab19ea649a
CROSSTICK to network-accounting, LEAN-249, job D, Wed 11 Nov 5:54 PM PST. **The runs have ended. On the crossed tip with the four picks, both tests pass and the encodings suite passes with its guard quiet.** The three shas are as in my 5:28 PM PST line; nothing changed since.

Marks: [ran] run by me, [read] read in a file, [argued] reasoned and not run.

## The commits, on `cursor/na-cross-main-closes-84ef` after `33360da0a088`

| commit | file | size |
|---|---|---|
| `cb00f1bc8dc9ea22b737533873a7210097913707` | `verity/protocols/certifier/verifier/tests/test_network_certifier_lengths_row.py` | +2 −3 |
| `b9bfb970b26c3e78f3f3bb1fa9cf18a6e57b1721` | `benchmarks/network_traces/tests/test_scan_zk_challenger.py` | +6 −4 |
| `102668539b17e1eae1e325239b64cf40577abbf3` | `benchmarks/network_e2e/encodings/pyproject.toml` | +5 −1 |

## The pick check [ran]
Scratch worktree at `origin/cursor/lean-remote-merge-crossed-fc8d` (`3ff11d118f93`, still the tip at 5:53 PM PST).
- `git cherry-pick -n` of `33360da0a088`, then the three in order: rc 0 each, no unmerged path.
- Each of the three alone after `33360da0a088`, and each alone on the bare tip: rc 0, no unmerged path.
- No file in common with `a4e1d7e04617`'s four. All three files are ours by the registry.

## The runs, `--on auto`, on the tip with the four picks (tree `f823379a630c`, not pushed) [ran]

| run | what | verdict line |
|---|---|---|
| `r20261112-012920-573f` | suite `benchmarks/network_e2e/encodings`, quick tier | "passed  benchmarks/network_e2e/encodings 451 passed in 92 s"; "1 suites: 1 passed (0 cached), 0 failed"; no guard line |
| `r20261112-012954-bcb6` | the two test files by `pytest`, slow tier included | the slow scan test "PASSED", "1 passed in 273.18s"; the run itself is rc 1 for my command, not for a test (below) |
| `r20261112-013752-05cc` | `test_network_certifier_lengths_row.py` by `pytest` | "PASSED …::test_the_blocks_ands"; "35 passed in 2.35s" |
| `r20261112-013820-a1d5` | suite `verity/protocols/certifier/verifier`, quick tier (not asked; I added it) | "passed  verity/protocols/certifier/verifier 1237 passed, 1 skipped in 627 s"; no guard line |

- **Why `r20261112-012954-bcb6` is rc 1** [read: its stderr]: its first half, the lengths row's file, never started. I ran `uv run pytest` inside the certifier's directory, where `uv` made a fresh environment: "error: Failed to spawn: `pytest`". So I launched that file once more, alone, with the workspace's interpreter: `r20261112-013752-05cc`.

## The two reads for the slow test [read, at the tip]
- `benchmarks/network_traces/scan_zk.py:299` to `301` (`lane_kinds`) and `:1180` (`tied=lane_kinds(cls)`).
- `verity/protocols/certifier/verifier/lean/NetworkCheck/Period.lean:1066` (`progName`), `:464` (`progParams`), `:308` (`programSlots`), `:1131` to `1140` (`wired`).
- The test now has `tied = list(sz.lane_kinds(setup.cls))` at line 89 and compares to it at line 90; the name in scope is `setup.cls`.

## The census read [ran: an import trace at the tip]
`tests/test_arith_vectors.py:24` → `arith_vectors.py:65` → `v1.py:28` → the advice code's `codec.py:94` → `window_code.py:63` → the certifier's `window_charge.py:92` → `arith_encode.py:146` → `head_check.py:67`, which imports `census`.
- It is the move's path and the branch's own read, both [read]: before the move the branch's list named the directory `verity/ml/circuits`, which held `census.py`; the moved list names single files of `verity/definitions/`, the ones `main` reads; `main`'s `head_check` does not import the census.

## The other suites' guards
- **Quiet in my runs** [ran]: the certifier's suite (`r20261112-013820-a1d5`), the period check's suite (`r20261111-210341-7377`, `r20261111-212645-fe5f`), `verity-network-traces` with the renames (`r20261111-212427-7d90`, `r20261111-214538-aaac`), sim and e2e (`r20261111-210326-ca88`, `r20261111-212633-2569`), `prove_path` with the renames (`r20261111-212529-1b46`). Those of 11 Nov are at `5ed73f441056`, one Lean file behind the tip.
- **Not run by me as suites**: `benchmarks/network_e2e/traces`, `verity-datacenter`, `verity-certifier-timing`, `verity-netsim-workload`. The last three depend on the `verity` package [read]. The first reads only the certifier's `release.py`; none of the files it opens at collection is outside its inputs [ran: my own audit hook, not the suite's guard]. lean's worker's run of every suite names the census for the encodings suite alone [read].

## For you, seen on the way and not acted on
- **`main` is at `be9a3d566905`, and the tip's merge with it stops on 49 files, 8 of ours** [ran: `git merge-tree`]. The four of `a4e1d7e04617` (`main` has not changed them since `75ef75838f99`) and four new: `benchmarks/prove_path/tests/test_records_audit.py`, `NetworkCheck/Period.lean`, `verity/protocols/certifier/verifier/network/tests/period_case.py` and the inference-only check's `lake-manifest.json`. The same files stop with and without the picks.
- The cut turn is not in my transcript and its VM state is gone. Nothing of job D was pushed before this turn [read]. I cannot say whether I launched a run in it; I found none to read.
- Least sure: the comment line of commit 3 names today's import chain, which can go stale; and only two lanes exercise the order of `rests_on_period_check`.

Note: `lanes/network-accounting/20261112T0138Z-report-lean249-crosstick-job-d.md`, synced.

[2026-11-12T01:58Z] 5c8ea0349dcb
SCANLEAN to the coordinator (network-accounting), NET-304, full report. Wed 11 Nov 5:57 PM PST.
This is the report that was due at 4:30 PM PST; it is late by the platform's stop and by this resume's work.

Marks: [ran] a command or run I ran, [read] a file or record I read, [argued] my own reasoning.

**Head.** `cursor/na-tied-need-crossed-37ad` at `72caf456be00`, pushed [ran]. No pull request, nothing marked, no
force push, no amended commit. Against the crossed tip `3ff11d118f93` it is the same eight files [ran: `git diff
--stat`]. Two commits since the first report:
- `fcdb903f4c` the certifier verifier's README: my line out, one stale clause rewritten, 49,115 bytes.
- `72caf456be` the mirror: one `prog_params`, not two (below, "Mended").

**Why the runs failed: nothing of ours.** `5b97` and `d465` ran the suite `verity-network-traces` whole and fresh,
`5b97` at the crossed tip itself and `d465` at our `2729cb7daca3`. Both failed on one test, the crossed tree's:
`tests/test_scan_zk_challenger.py::test_the_honest_round_on_a_local_challengers_coins_registers_its_generator_form_and_draws`,
at `benchmarks/network_traces/tests/test_scan_zk_challenger.py:88`, `assert … res["rests_on_period_check"] ==
["prog"]`, which got `['prog0', 'prog1']` [read: each run's `out/logs`]. Our branch does not change that file. Both
logs hold a second cause, also the crossed tree's: the guard reports a read outside the suite's inputs,
`benchmarks/live_challenger` (`beyond_tests: true`), and `suites.py` fails a suite on that alone [read:
`suites.py:2161`]. Its reader is `tests/test_wipe_gate.py:148` [argued: the one line of the suite's tests naming
that path]. CROSSTICK's branch holds a commit for each (`b9bfb970b2`, `33360da0a0`) [read]; neither is on the
crossed branch yet (still `3ff11d118f93`) [ran: `git fetch` at 5:38 PM PST].

**The runs** (all `research run --on auto`; the first try and the six you named).

| run | what it ran | tree | verdict | whose |
|---|---|---|---|---|
| `r20261111-213424-f306` | `lean_changed.py --try --siblings`, the three Lean files | `f77c118904` | done, rc 0; each "lean exit 0"; "none of its 44 theorems uses an axiom beyond propext, Classical.choice and Quot.sound" [ran] | passes |
| `r20261111-214849-bb83` | the same try | `2729cb7daca3` | done, rc 0; each "lean exit 0"; the same axiom line; siblings: no clash, 3502 modules of `security/Proofs` unchecked [ran] | passes |
| `r20261111-214103-94e1` | suite `verity/protocols/certifier/verifier/network`, fresh | `7c53000a04` | done, rc 0; "93 passed in 568 s"; "1 suites: 1 passed (0 cached), 0 failed" [ran] | passes |
| `r20261111-214722-6768` | suite `verity-network-traces`, fresh | `105509ca1b` | failed, rc 1; "551 passed, 1 failed in 533 s" [ran] | the crossed tree's [argued: the same counts as `d465`; its record does not name the test] |
| `r20261111-221401-37dd` | `lean_changed.py --try`, five modules that import the changed files | `2729cb7daca3` | done, rc 0; each "exit 0"; "none of its 11 theorems uses an axiom beyond …" [ran] | passes |
| `r20261111-221939-5b97` | suite `verity-network-traces`, fresh, logs kept | `3ff11d118f93`, the crossed tip, nothing of ours | failed, rc 1; "547 passed, 1 failed in 463 s"; `test_scan_zk_challenger.py:88` [ran] | the crossed tree's |
| `r20261111-221939-d465` | the same | `2729cb7daca3` | failed, rc 1; "551 passed, 1 failed in 475 s"; the same test, the only one failed; our four new tests pass [ran, read: its JSON] | the crossed tree's |

The five modules of `37dd`: `NetworkCheck/PeriodMain.lean`, `NetworkCheck.lean`, `onsite/verifier/CoreOnsite.lean`,
`security/Properties/NetworkScan.lean`, `security/Proofs/Properties/NetworkScan.lean`.
`5b97` and `d465` are mine: launched at 2:19 PM PST in my own command's form, in steps the stop cut from my
transcript [argued]. I found no sign of a second execution of me.

**Mended (ours), and the reruns.** Two things were ours, and neither is a failure of a run.
1. The README was over its cap with my line (below). Mended in `fcdb903f4c`.
2. My port had given the mirror a second `prog_params` (`period_case.py`, a list) below the crossed tree's own (a
   tuple, which the scan's parameters read). Python took the later one for both, with the same names, so no verdict
   moved [argued; `94e1` passed with it]. But it was one thing defined twice, in code held to Lean. `72caf456be`
   removes mine; the tied-input mirror and its tests read the first, and the three places that compare it with a
   JSON list say `list(…)`. No top-level name is defined twice in the three Python files now [ran: an `ast` count;
   ruff's F811 is clean].
Both suites the change touches then ran again, once each, at `72caf456be00`:
- `r20261112-013542-f6ed`, the suite `verity/protocols/certifier/verifier/network`, fresh: done, rc 0. "passed
  verity/protocols/certifier/verifier/network 93 passed in 602 s; 93 outcomes recorded"; "1 suites: 1 passed (0
  cached), 0 failed". No read outside its inputs. The four tied tests are among the passed [ran, read: its JSON].
- `r20261112-013542-5cfd`, the suite `verity-network-traces`, fresh: failed, rc 1. "failed verity-network-traces
  551 passed, 1 failed in 532 s". The one failed test is `test_scan_zk_challenger.py:88` again, and the outside
  read is `benchmarks/live_challenger` again: the crossed tree's two, as before. All 36 tests of `test_scan_zk.py`
  pass, the six the change touches among them [ran, read: its JSON]. I launched it knowing it would end red: the
  commit changes `test_scan_zk.py`, and the suite's own record is the only node evidence that our tests pass on it.
- The Lean was not run again: no Lean file changed after `2729cb7daca3` [ran: `git diff --stat 2729cb7daca3
  HEAD` names the README and three Python files], so `bb83` and `37dd` stand for the head.
- On my VM before the launch, no Lean [ran]: the mirror reaches every new case's verdict (120 tied cases at one
  lane and at two, 28 honest seeds, 194 small periods), and the six `test_scan_zk.py` tests pass by name.

**The form.** The subset form, as you ruled at 1:29 PM PST. `tiedInputWired s a` holds when the answer has no
public outputs, or no `rests_on_period_check` under them, or a list there whose every entry is a string among
`progParams s.programSlots`. Any order; a name twice passes; an empty list passes. A value that is no list (a
string, a number, null) is refused, and so is an entry that is no string. The refusal names the first entry that is
no lane's and the lanes' names: `… its public outputs' rests_on_period_check names "state", none of the scan's
lanes' prog0, prog1` [ran: asserted in both suites].

**The statement, as it stands in `security/Properties/NetworkPeriod.lean`** [read]:

```lean
def RowsTiedInputWiredAt (p : Params) (c : Commitment) (e : Evidence) : Prop :=
  ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (i : ℕ) (reg : Lean.Json) (x : Window), c.windows[i]? = some reg → e.windows[i]? = some x →
      tiedInputWired s x.answer = true ∧ wired p.audits s c.audits reg = true

def RowsTiedInputWired : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → RowsTiedInputWiredAt p c e
```

Proved: `rowsTiedInputWired : RowsTiedInputWired`, and `rowsTiedInputWired_nontrivial : ¬ RowsTiedInputWiredOf
acceptAllRows` [ran: `bb83`, each file "lean exit 0"]. What it does not say, and its docstring says so: that the
answer carries every Input the registration check let through, and that lane j's name `prog<j>` is the Input the
entry `scan/prog<j>` registers. Both are one-stage's registration check's to say.

**What the Lean and the Python each read.**
- Lean, `tiedInputWired s a` in `NetworkCheck/Period.lean`: `a.published`; none gives true. Else `getObjVal?
  "rests_on_period_check"`: an array gives `ts.all (laneTied s)`, any other value false, no key true. `laneTied`
  takes a string that `progParams s.programSlots` contains.
- Python, `tied_input_wired(s, a)` in `period_case.py`: `a["public"]`; no dict or no key gives True. Else a list
  whose every entry is a string in `prog_params(s.lanes)`.
- Held to each other by the difftest of `test_network_period_protocol.py`, on answers its generator writes: five
  kinds that pass, ten that are refused, at one lane and at two, and the honest key where `every > 1` [ran: `94e1`,
  and the rerun].
- **What reads what, plainly:** Lean's period check has only ever read answers the generator wrote. The tests on
  an answer a one-stage verifier computed (`test_scan_zk.py`) run the Python registration check and the Python
  mirror (`PC.reference`, `PC.tied_input_wired`), no Lean [read]. That is item 2, below.

**The README.** `verity/protocols/certifier/verifier/README.md`, cap 49,152 bytes [ran: `wc -c` at each commit].
- 49,158 on the crossed tip: 6 over before anything of ours.
- 49,251 with my line (`105509ca1b`): 99 over. The line that caused it, 93 bytes: "one whose scan's answer names
  a tied Input that is no lane's `prog<j>` (`tiedInputWired`), ".
- 49,115 now (`fcdb903f4c`): 37 under. The cap test's own function returns nothing for it [ran:
  `markdown_over_cap`].
- By your rule the line is out: I found one stale clause and it frees 45 bytes, fewer than the line's 93. Cut:
  "the Programs' row is a compute audit's `SlotEntry_v1` template's entries, that template a pinned Definition
  (its closure digest one of `slotEntryPins`)". In its place, as the crossed tree's `wired` has it [read:
  `entryPins`]: "each lane's entries are a compute audit's pinned `SlotEntry_v1` or `SelectEntry_v1` template's
  (`entryPins`)". The form is said in the Lean docstring, and in `benchmarks/network_traces/README.md`, which has
  no cap trouble (28,731 bytes).
- If you want the line back, a shorter one fits at 49,146 ("names another tied Input (`tiedInputWired`)", the
  template names dropped). Not taken: it says less than either text.

**Not done.**
- Item 2, the `need` on an answer one-stage's Lean verifier computes. PRF-504's fix is on neither tree [read, at
  `main` `75ef75838f99` and again at `be9a3d566905`, and at the crossed tip]: `Spot.OneStage.Public` is `{drawn,
  proved, bound, perTemplate}` and its `publicJson` writes no `rests_on_period_check`. Only the registration piece
  (`Spot/Pieces.lean`) and the Python write the key.
- No lock run: no `--update`, no `check.py`, as told. So no kernel replay of the new theorems beyond the try's
  axiom probe.
- `security/Proofs` was not built whole (the try builds each file with its imports; `bb83` left 3502 of its
  modules unchecked for a clash of names).
- No run of our head with CROSSTICK's two commits: no branch of ours may hold that tree. It is the one run that
  would show the traces suite green with our change. Say when lean has picked them and I launch it.
- Lean's `tiedThrough` was not run on the proper-subset case below.

**Can `shared?` give a proper subset of the lanes?** Yes, in one case [ran, on the Python mirror
`registration._readers`, two lanes, `every` 2 and 4]: with `prog1` expected registered and `prog0` named tied,
`prog0` alone is let through. With no lane registered it is all the lanes or none (`R8-shared-input`). So the
subset form is the one the registration check can produce.

**Lock records the change would add** (not run; from the change's own shape and the same change's lock on `main`'s
tree) [argued]. In `security/Properties/lean-audit.json`: two `guarantees`,
`Properties.NetworkPeriod.rowsTiedInputWired` and `…_nontrivial` (owner `@network-accounting`, no premise); one
`shape.properties` entry (`NetworkCheck.Period.check`, outcome accept); one `witnesses` entry (nontrivial); one
`unwitnessed` entry (the honest half); `reads` records of `NetworkCheck.Period` and `Properties.NetworkPeriod`.
In `security/lean-audit.json`: the `reads` record of `NetworkCheck.Period`. No `open` premise.

**Not ours, left as you said** [read].
- `benchmarks/network_traces/tests/test_scan_zk_challenger.py:88` and its docstring at line 44: CROSSTICK's.
- `benchmarks/network_traces/tests/test_wipe_gate.py:148`: the read outside the suite's inputs.
- `benchmarks/network_traces/chain_record.py:37` and `tests/test_chain_record.py:450-455` say `prog`. They pass:
  the name is opaque there.
- `security/Properties/Spot.lean:134-135`, R8's docstring, names `RowsTiedSampled`, which is gone. Named for its
  owner in my NET-304 comment.
- `benchmarks/network_traces/README.md` has 57 sites of retired terms against 56 allowed, on the crossed tip as on
  ours [ran: the lint's count at both]. A warning, not ours.

**Least sure of.**
- That `6768` failed for the same test: its record holds counts only.
- The lock's list: nothing ran it on this tree.
- Whether the difftest's generated answers stand for what a registration check publishes. They are written by
  the generator, not computed.

**Two corrections to my 5:25 PM PST line.** Its header says 5:27 PM; the clock read 5:25 PM PST when it went. And
it said `6768` "failed on it too" as a fact. That is argued from its counts: its record does not name the test.

