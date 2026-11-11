[2026-11-11T08:29Z] 945128d001df
LOCKORDER to network-accounting, NET-333, report. Times Pacific. Marks: [read] file:line at my head, [run] id and exit (or "local": a command on my VM), [argued].

**State: blocked on one answer about the real run's command (section 10). No pull request open, as you said (12:25 AM).** Branch `cursor/net333-records-pass-1883` (cut from `origin/main` `8201dacff26e`, pushed, three commits), head `95d63ad5d1b0`. The platform gave my run the name `cursor/verity-network-accounting-b51f`; that branch already holds 19 unmerged commits of another task, so I did not push to it (my note of 11:45 PM).

## 1. What is written

- `edc3518c20` `merge.py`: `unwitnessed` and `shape[inline]` join `SORTED` [read merge.py:140-145]; `held` writes its put-back through `keys` [read merge.py:608]. Test `test_a_merge_writes_unwitnessed_and_shape_inline_s_guarantees_in_the_order_check_py_update_writes` [read tests/test_merge.py:1377]. One paragraph in the module docstring and one in the README's "Merging records".
- `ef40762253` + `95d63ad5d1` `lean_changed.py`: `checked` compares what the two say [read lean_changed.py:1653-1675] through `same` and `says` [read :1635-1650]: the JSON of each with every object's keys sorted. Key order at every depth and layout do not count; a list's order, every value and a missing key do (`true` is not `1`). The same means: nothing kept, the committed bytes written back over check.py's rewrite, `None` returned. Tests `test_a_committed_lock_out_of_check_py_s_order_that_says_what_the_build_says_ends_pass_and_keeps_no_record` (slot and dev-cache) and `test_a_committed_lock_that_says_something_else_than_the_build_still_ends_regenerated_with_its_record_kept` (4 differences x 2 paths) [read tests/test_lean_changed.py:487, :546]. Module docstring reworded.
- check.py is not changed. No sleeps or clocks in the tests.

## 2. The readers of the comparison

- `checked`: **changes** (above). It is the only place that compares a record with the committed lock by bytes [run local: grep of lean_changed.py].
- `settled`: code unchanged [read :1044-1069]. It gets `own=False`, so with exit 0 it returns `PASS` (:1069). Its record-difference branch (check.py without `--update` failing only on differences) is untouched: check.py judges records parsed there, never layout [run r20261111-062937-d6f9, exit 0: `security` passed as a requirer, without `--update`, on `main`'s out-of-order lock].
- `verdicts`: code unchanged. It adds `record` and `record_sha256` only for a lock in `new` [read :1107], so a pass carries neither.
- `write_back`: code unchanged, called only when `checked` returned a record [read :1237-1238, :1681], so an unrecorded pass no longer writes the checkout.
- `--dev-cache` path: code unchanged [read :1301-1303], but behaviour changes: check.py `--update` rewrote this checkout's lock there, and `checked` now puts the committed bytes back when the record says the same. Before, the rewrite stayed. I chose this so a pass leaves the lock as committed on both paths. Say if you want the rewrite left.
- `passed`: code unchanged [read :932-945]. With nothing in `new` it keeps the pass on the lock as committed (:938), so the result's `passes` lists the head's own inputs key. Before, a layout-only `regenerated` listed only the rewritten record's key, so the head it ran on had no pass.
- `research queue`'s records gate: code unchanged. It vouches by `passes.<key>` for the head's key [read queue.py:5434-5440] and names a kept record only for `outcome == "regenerated"` [read :5518-5520]. Its two comparisons of merged locks are parsed (`json.loads(..) == json.loads(..)` at queue.py:5485; `got != load_at(..)` at lean_changed.py:907), so the new order in merge.py changes none of their answers. One sentence of its docstring is now loose: "a package whose record its `--update` wrote" should read "kept". I did not edit queue.py (another tool, another suite).

## 3. Cache keys

- Two keys exist. The records key (`--inputs-key`, what `passes` and the gate read) covers the closure's files plus `tool_files` plus `lean_changed.py` itself [read lean_changed.py:691]. The Lean check's audit key (`lean_audit.key`) leaves out `lean_changed.py`, `merge.py`, `tests/`, `*.md`.
- [run local] `--inputs-key security security/Properties security/Proofs`:
  - at `8201dacff26e`: `e3f4156431ab…`, `1e364512bc11…`, `f95f4f408998…`
  - at `edc3518c20` (merge.py only): the same three. **No key moves.**
  - at `95d63ad5d1b0` (head): `4e7560f5cf63…`, `9b83fbe1762a…`, `2bb41999415b…`. **All three move**, as they do for every package: `lean_changed.py` is in every records key.
- [run local] `lean_audit.key` for the three is unchanged at the head (`5087b533ac21…`, `94bc7fd0abe6…`, `6794ba778405…`): no Lean check starts cold.
- So by your rule I open no pull request. The smaller change is `edc3518c20` alone (merge.py, its test, the README paragraph): it moves no key. It does not make a run on today's `main` lock end `pass`, and it cannot fix the order of a guarantee's premises.
- Base rate [run local: `git log --first-parent origin/main`]: landings that moved every package's records key through a tool input: 11, 33, 69 in the last 1, 3, 7 days; through `lean_changed.py` itself 2, 7, 20.

## 4. Where check.py orders each part

- `unwitnessed`: sorted by guarantee [read check.py:2513].
- `shape.inline`, the guarantees: sorted [read check.py:2731].
- `shape.inline`, one guarantee's premises: **not a sort.** They come in the order the statement takes them, or as the old lock had them [read check.py:2704-2708, :2722, :2690]. I left that part to ours' order in merge.py and say so in its comment. Only the parsed comparison makes a premise order harmless.
- Which commits put `main` out of order [run local: replay of `merge.merge` on base, ours, theirs]: `unwitnessed` at `01d153382c`, `shape.inline` at `5e358bead1`. At each, both parents were sorted and `main`'s merge.py reproduces the committed order exactly. With my merge.py both replays give 0 misplaced. Your inference (merges) is right.

## 5. Evidence that item 1 does what you asked, before the real run

[run local] For each of the five `regenerated` runs you named, I fetched the kept `lean-audit/record/lean-audit.json` and the lock at the run's commit: bytes differ, `same` says they say the same, for all five (`0a73`, `dd12`, `cd4a`, `3709`, `6a9d`). In `3709` the key order differs in `unwitnessed`, `shape.inline` and the premises of `EndToEndRegistered` and `EndToEndRegisteredDrawn`, nowhere else. The three `pass` runs ran `--update` on heads whose lock has 0 misplaced keys.

## 6. Your finding 1: the `format` stamp

- Confirmed [run local]: 20 of 32 locks at `8201dacff26e` have no `format`; `security`, `security/Properties`, `security/Proofs` have it. The record `r20261111-062937-d6f9` kept is its lock plus the one line `"format": 1,`, and `same` says they differ. My change does not cover it and I changed nothing about `format`.
- check.py itself calls that rewrite "saying what it said": it compares `new != policy` on the parsed lock before `dump` adds the stamp [read check.py:5698-5713, :493], and nothing but the merge driver reads `format` [read check.py:1042-1046; merge.py:1231-1234, absent reads as 0].
- **Leave `format` out of the comparison.** One place: `same` drops the top-level `format` of both sides. Nothing in the gate changes: it never compares a record with a lock, it only looks up the head's key, and `passed` already keeps the pass on the committed bytes. What the two must agree on is only that: no record kept means the pass is under the committed lock's key. Costs: (i) the 20 locks stay unstamped until their content changes; (ii) at the next bump of `FORMAT`, a lock whose only difference is the old stamp would never get the new one from a records run, which weakens the driver's guard; a narrower rule (leave it out only when the lock has none and the record's is 1) avoids that; (iii) it is one more edit of `lean_changed.py`, so the records key of every package moves, as for item 1.
- **Stamp the 20 locks once.** A data-only commit, no tool change. [run local: simulated in a scratch worktree at `8201dacff26e`] It moves **both** the records key and the Lean check's audit key of **30 of 32 packages**, `security`'s three among them: every key reads each lock in its closure, and `security`'s closure holds 30 packages. Unchanged: `verity/protocols/flock/verifier` and `verity/protocols/certifier/device/program/lean`. So every kept verdict of those 30 misses once, the Lean check's included. That is what a landing that edits check.py costs (6, 24, 45 of them in the last 1, 3, 7 days), and the stamp could ride in one. All 20 are in check.py's layout already: each stamped text is the old text plus one line.

## 7. Your finding 2: #8537 and #8848

- [run local] Both heads hold `security`'s lock with 0 misplaced keys, byte for byte in check.py's layout. Their merge base with `main` is `5c095bb4ab70`, whose lock is `main`'s (53 and 3 misplaced). `main` has not changed `security/lean-audit.json` since that base; #8537 has. #8848 contains #8537's head.
- So when #8537 lands as things stand, git takes its lock whole and the driver does not run [argued: one side unchanged since the base]. `main`'s lock is then sorted, and #8848 after it lands the same way.
- A records run on `main` then ends `pass` with no change of ours [argued], as `r20261111-044443-02b9` did at #8537's head [run, exit 0]. It holds until the next merge in which both sides changed the lock.
- If anything that changes `security`'s lock lands before #8537, the driver runs. [run local replay, base `5c095bb4ab70`, ours `main`, theirs #8537] `main`'s merge.py keeps ours' order: 53 and 3 misplaced remain. Mine gives 0 and 0.
- So `SORTED` still matters after that landing: `01d153382c` made the disorder from two sorted parents, each adding entries, and the next such merge does it again. Item 1 still matters for heads that carry the old lock, for premise order, and for a lock edited by hand.

## 8. Runs

- Suite, on a node [run r20261111-081106-1bee, rc 0, SUCCESS] at head `95d63ad5d1b0`, as printed: `passed  verity-lean-audit            608 passed in 841 s; 608 outcomes recorded` and `1 suites: 1 passed (0 cached), 0 failed; 14.0 min of test time`.
- The real records run: **not launched. I need one answer first (section 10).**

## 9. Not checked, and least sure

- Not run whole: the `research`, `verity-check`, `verity-graph` and `repository` suites, whose inputs my files are. Run locally instead, named files only, all passing: `tools/verity/check/tests/test_lean_changed.py`, `test_lean_records.py`; `tools/research/tests/test_record_driver.py`, `test_queue.py -k "records or lock or merge"`; root `tests/lint/test_repo_silent_excepts.py`, `test_repo_env_reads.py`, `test_repo_retired_terms.py`, `tests/test_repository.py`, `test_doc_paths.py`, `test_no_wall_clock.py`, `test_lean_packages.py`.
- Not read: drive's lock run and how it writes a lock it regenerates; whether the queue lands by a merge commit in every case (I read two landings, both two-parent merges).
- The lean-proofs skill is unchanged: its sentence ("Commit each record it kept") is still true, and the skill is 3 bytes under its 16 KB cap.
- Least sure: (a) restoring the committed bytes on the `--dev-cache` path; (b) that the merged tree after #8537 gives the record #8537's head has, which only the lock run after its landing shows; (c) whether lean wants `says` to follow check.py's own notion (`new == policy`, which also reads old names as today's) rather than plain JSON.

## 10. What I need from you

Your 12:25 AM message is read: both commits, item 1 droppable, `format` out of both, stop before the pull request. Commit `edc3518c20` (merge.py, its test, the README paragraph) depends on nothing in item 1.

What a reader of `pass` can no longer conclude after item 1: that the committed lock is byte for byte what `check.py --update` writes; it says the same, perhaps in another order of keys or layout.

**One question before I launch the real run, since it is launched once.** The brief's command is `--records security --update`, as the skill launches it. At my head `security`'s Lean-check audit key equals `main`'s (section 3), and `kept` answers "reused, nothing built" when the node holds the Lean check's pass on that key [read lean_changed.py:773-774]. I have not read any node's cache, so this is a risk, not a finding [argued]. If it happens the run ends `pass` with `path: reused` and never reaches `checked`. `--rebuild` skips that lookup [read lean_changed.py:1166] and costs nothing else. It is not in the brief or the skill, so I did not add it.

Say one of: "verbatim" or "with --rebuild". I launch that, once, and end my turn while it runs.

TIMER: none

