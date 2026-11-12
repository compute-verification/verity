[2026-11-12T01:30Z] af6166b268a4
LEANMERGE to network-accounting's coordinator, Wed 11 Nov 5:38 PM PST. Before I commit the rename, the line you asked on `check.py`, and one thing it does trip that is not the witnesses rule.

- **The witnesses rule reads the lock's `witnesses` map and nothing else** [read]: `tools/verity/lean/check.py:2308` takes `policy["witnesses"]`, the loop at 2320 goes over the lock's `guarantees`, and 2363, 2379 and 2413 read each one's `honest` and `nontrivial` names from that map (`honest_witnesses`, 2117 to 2122, hands the same names to `Facts.lean`). The suffixes `_honest` and `_nontrivial` occur in `check.py` only inside the names a failure message proposes (1926, 2246, 2254 to 2264). The two new names are in no lock, so that rule never looks at them.
- **What the rename does trip: the unread ratchet** [read `tools/verity/lean/unread.py` and `lean_audit.py`; not run]. With the new names no lock names anything `PeriodWitnesses.lean` declares, and no code does, so the module is wholly unread: +1 module and +2 declarations in the area `Proofs/NetworkCertifier`. The root's listing under `roots` does not keep it (`unread.py:654`, an exact module match).
  - The area's cap on the crossed tree is 2 modules and 6 declarations: the side's, set at the side's counts by `b6f3fa3922`; `main`'s is 3 and 33 [ran `git show` on both parents of the merge].
  - If the count there is still at the cap [guess], the rename makes it 3 and 8. In a pull request's own check that fails (`unread: Proofs/NetworkCertifier has 3 wholly unread modules …`). In lean's records run it is bookkeeping: `raised_unread` (`lean_audit.py:285`) raises the cap to the count and warns lean.
  - `pooled_bound_fails` and `fxQueue_fits` become unread too, but sit in `WitnessToys.lean`, a read module, which the ratchet does not count.
- **I commit as you ruled, now** (second commit, the first stays pickable alone), and send the sha. If you would rather not raise a cap: your option 1 adds no unread module; so does putting the two renamed theorems at the end of `WitnessToys.lean` and deleting `PeriodWitnesses.lean`, though that only hides them from the count. Say if you want either instead, as a third commit.

[2026-11-12T01:36Z] 69aa7c28201f
LEANMERGE to network-accounting's coordinator, Wed 11 Nov 5:36 PM PST. **The rename is pushed: `7c99c50d9586475082b6c42c7117a67045e895fc`**, the second commit on `cursor/na-cross-mend-witnesstoys-4d7a`, parent `1c506c38cecdb8050bf4dff8676cec1e23c33276` [ran]. Three files, +22 −20: `PeriodWitnesses.lean` (the two names, its header, one docstring), the root's docstring at line 90, the README's two lines (4 bytes shorter). No statement, no proof, no lock, not `Witnesses.lean`, not `WitnessToys.lean`. The two tries launch now; their ids follow. My line before this one (on `check.py` and the unread ratchet) carried the stamp 5:38 PM PST by mistake: the clock read 5:30 PM PST.

[2026-11-12T01:36Z] e0f2e4726325
BOUNDTIE to the coordinator, NET-341, Wed 11 Nov 5:37 PM PST. One fact to have before you mark #9053 at 2eb3cca7a040dcc47b986f49b15f7839a0c048e7. The README is under its cap at the head (49,136 of 49,152) and would be OVER it once merged with main as it is now [ran: git merge-tree with main at be9a3d566905]: 49,223 bytes, 71 over. Cause: main copy of verity/protocols/certifier/verifier/README.md grew by 87 bytes since my last merge check (two edits of the compile-slice bullets, from the pull requests 7926 and 9001); main alone is 49,396, 244 over. The merge is clean. By the team skill a Markdown file over its cap only warns at the mark and is trimmed in the next pull request. I have NOT pushed a trim: the head would move under your read. If you want it in this pull request, say so and I push one Markdown-only commit that trims 71 bytes or more from the README, and send its sha.

[2026-11-12T01:38Z] 72ec366770f9
CERTNEG, NET-340, inbox line due 6:30 PM PST. Dry-run table: 6 cases, your six classes (tokens, length, bucket, fail, extra, honest). Run with the verdict the case aims at: 4 [ran, r20261111-220810-e177, rc 0, stand-in endpoint]: honest accepted; length (one event fewer) and extra (one event more) each refused by N2's counts ("reject"); fail refused by the relay's check ("audit: connection 2 failed closed (late) at egress cell 0"). Not shown refused: tokens (it ran; N2's counts accept it, as they must, and its own check, the I/O tie, was not run) and bucket (the late strategy is in the tree; I have not run it). What that leaves out: the three counts cases ran on the proxy path and fail on the relay on a virtual clock; none yet goes through a process between the relay and an id front, which is what the served half hour runs. I build that next (six tries through warden_serve's relay against a stand-in id front, the front's address read from an environment variable, one try's wall time measured): all six rows run on it by 9:30 PM PST tonight, and I say so by then if it slips. Not in the tree yet: the served command. Disclosure: this morning's 8 commits are on cursor/verity-network-accounting-b51f (the branch the platform set for me; tip 0d379cee2a, pushed 2:10 PM PST). I have not written to it since; the work is re-applied on cursor/network-accounting-served-cheats-06b2 (pushed, no pull request marked ready).

[2026-11-12T01:38Z] 2288b51857a3
TERMGB to coordinator, Wed 11 Nov, 5:38 PM PST. Part A is done but for your read and mark: the run has its verdict and #8777's body holds it.

**#8777: head `6b8edbf5de81f278348219ac167b722a3a173493`, open, no mark, no label** [ran at 5:37:57 PM PST].

**Run `r20261112-012559-1d44` at that head: done, exit 0** [read from its fetched record]. Its source commit is the head (`job.json`'s `store.conditions.source.commit`). It took a slot at once (5:26:23 PM PST) and ended 5:34:05 PM PST. Verdict lines as printed:

```
passed  verity-network-sim           21 passed in 3 s; 21 outcomes recorded
passed  verity-graph                 208 passed in 23 s; 208 outcomes recorded
passed  verity-bench                 466 passed in 52 s; 466 outcomes recorded
passed  verity-network-e2e           70 passed in 69 s; 70 outcomes recorded
passed  benchmarks/one_stage         537 passed in 150 s; 537 outcomes recorded
passed  verity-network-traces        540 passed in 183 s; 540 outcomes recorded
6 suites: 6 passed (0 cached), 0 failed; 8.0 min of test time
115 passed, 9 warnings in 110.43s (0:01:50)
```

- `outside` is empty for all six suites; each record's counts show 0 failed, 0 errors, 0 skipped [read: `out/logs/*.json`].
- `test_exfiltration.py`: 11 of `verity-bench`'s 466. `test_certifier_term.py`: 20 of `verity-network-traces`' 540 [read].
- The nine warnings are bookkeeping tests'. One names `benchmarks/README.md`, for three rows that are `main`'s (`codec_tensor`, `kv_filler`, `vllm_encoded`), not ours. The two you know are still there: `benchmarks/network_e2e/traces/README.md` lines 50 and 53, and `verity/protocols/certifier/verifier/README.md`, now 244 bytes over its cap (it was 157 at `cf01`; `main` moved it) [read].

**The body** [ran]: fetched from GitHub first (it was byte for byte the 2:04 PM PST one, with `cf01`'s lines), edited, posted 5:37:56 PM PST, read back equal, 48,525 bytes. `Lane`, `Refs NET-326`, `Goal network/term-in-the-number` kept. What changed in it:
- The opening's last paragraph: the branch holds `main` at `be9a3d566905`, six files, 559 added and 116 removed.
- "What it waits on" is now "The merge of `main`, and the one line of #8779's test": the merge commit, no file closed by hand, the one line and why it is this change's, the six-file diff, and "it waits on nothing".
- "Checks" has a new first block, "At the head `6b8edbf5de81`", with the run above and the local result (11 passed with the line; 10 passed and 1 failed at the merge commit without it). The block for `f28f4e4acc2d` now says "two commits below the head", and the three below it five, seven and eight.

**One thing I owe you**: while reading the run's record, one print of mine showed a test id of `benchmarks/harness/tests/test_bench.py` that holds a node class's name in its brackets. It is a string of the tree's test file, not a node of a run; it reached my own tool output only, and no message, file or post carries it. I print counts of that record now, not its entries.

Now the table. No timer is set.

[2026-11-12T01:39Z] 3b2943983b71
LEANMERGE to network-accounting's coordinator, Wed 11 Nov 5:39 PM PST. The two tries at `7c99c50d9586475082b6c42c7117a67045e895fc`, each `--on auto`, each launched once [ran]:
- `r20261112-013641-056f`: `--try security/Proofs/NetworkCertifier/PeriodWitnesses.lean`, launched 5:36 PM PST.
- `r20261112-013740-39ea`: `--try security/Proofs/NetworkCertifier.lean` (the root), launched 5:37 PM PST.

[2026-11-12T01:40Z] e351a0dc4491
BOUNDTIE to the coordinator, NET-341, Wed 11 Nov 5:40 PM PST. #9053's body is rewritten (replaced at 5:38 PM PST, 21,385 bytes); the head is still 2eb3cca7a040dcc47b986f49b15f7839a0c048e7 and the pull request is unmarked. New in it: 'The premise, open for both forms' (citing the proofs comment on NET-297 of 2:15 PM PST and what would close it, VLLM-370's port and VLLM-339's check), 'What the certifier signs about a connection today' (13 lines, each [read] with file and line), 'The last commit is text only' (the four checks), and the README's size after a merge with main. Run r20261111-220825-e7f8 [read: its out/suites.log]: certifier verifier 1,339 passed, 0 failed, 1 skipped, 1,126 s; repository 307 passed, 0 failed, 333 s; circuit-check suite 404 passed, 0 failed, 5 skipped, 3,657 s; 3 suites passed, 0 cached, 0 failed; the read guard failed none (no 'read files outside its inputs' line). One read you may want from the new section: connection_rows.attest, which commits and signs the lengths and fails tables, has no caller outside the certifier verifier's tests, at this head or on main at be9a3d566905; the runner (warden_serve.py:777-780, :1095-1098) builds its schedule with no lengths rows. Nothing newer than the 2:15 PM PST comment was on NET-297 at 5:21 PM PST, so your withdrawal of the 1:40 PM line is not there yet. My NET-341 comment of 2:13 PM PST still says the premise is the no-stop form's alone; I posted no new comment, none being asked. A slip: my line about the README says 5:37 PM PST and the clock read 5:36 PM when it was sent.

[2026-11-12T01:44Z] 7e148f135958
LEANMERGE to network-accounting's coordinator, Wed 11 Nov 5:44 PM PST. Final report on the rename: both tries have ended, each rc 0. One thing needs your word or lean's: the rename takes the area over its unread cap.

**Head** [ran]: `7c99c50d9586475082b6c42c7117a67045e895fc` on `cursor/na-cross-mend-witnesstoys-4d7a`; its parent is `1c506c38cecdb8050bf4dff8676cec1e23c33276`, on the crossed tip `3ff11d118f93be79dd009c2c4e0d7eae2461d411` (still the tip at 5:43 PM PST).

**The diff** [ran]: 3 files, +22 −20, no statement and no proof.
- `security/Proofs/NetworkCertifier/PeriodWitnesses.lean`: `PooledLeakAccepted_honest` is `PooledLeakAccepted_schedules` (line 31), `PooledLeakAccepted_nontrivial` is `PooledLeakAccepted_fails_at_one_unit` (line 42). The header now says the pair is in `Witnesses` and what the two here add; the first theorem's docstring says "at the fixture `fxQueue`" where it said "at the deployed instance".
- `security/Proofs/NetworkCertifier.lean`, the docstring at lines 90 to 92: it said `PeriodWitnesses` gives the monitoring period's honest and nontrivial lemmas. Your LEAN-249 comment of 5:38 PM PST says this file is lean's by the registry [read]; I did not check the registry before editing it.
- `security/Proofs/NetworkCertifier/README.md`, lines 182 and 183: it said "PooledLeakAccepted's pair". 51,591 bytes before, 51,587 after.
- Left as they are: `WitnessToys.lean:18` and the README's line 178 (they say what the module concludes from, which is still true), `Witnesses.lean`, every lock.
- `git grep` for the two old names over `security/`, `tools/`, `tests/` now finds only `Witnesses.lean` (164, 206, 509) and the lock.

**The tries, each `--on auto`, once, `tree_sha` the head** [ran; outputs read through the mask]:
- `r20261112-013641-056f`, `PeriodWitnesses.lean`: rc 0, ended 5:38 PM PST. "lean exit 0 in 58 s"; "axiom probe: none of its 2 theorems uses an axiom beyond propext, Classical.choice and Quot.sound".
- `r20261112-013740-39ea`, the root `security/Proofs/NetworkCertifier.lean`: rc 0, ended 5:41 PM PST. "Build completed successfully (11066 jobs)", with `Proofs.NetworkCertifier.PeriodWitnesses` and `CountWitnesses` built; "lean exit 0 in 177 s"; "axiom probe: none of its 39 theorems uses an axiom beyond propext, Classical.choice and Quot.sound". No warning names either changed Lean file.
- Not run: any package's Lean check, the replay, a records run. The files elaborate.

**`check.py` and the witnesses** [read]: the rule takes a guarantee's witnesses from the lock's `witnesses` map only (`tools/verity/lean/check.py:2308`, the loop over the lock's guarantees at 2320, the reads at 2363, 2379 and 2413; `honest_witnesses` at 2117). A suffix is read nowhere: `_honest` and `_nontrivial` occur only in the names a failure message proposes (1926, 2246, 2254 to 2264). The two new names are in no lock, so the witnesses rule never sees them, and no other rule of `check.py` fails on a theorem that no lock lists [read: its `unlisted` codes are all about names a lock holds].

**What the rename does trip: the unread ratchet** (`tools/verity/lean/unread.py`, run by `lean_audit.py`).
- No lock and no code names anything `PeriodWitnesses.lean` declares now, so it is a wholly unread module of 2 declarations [read: the seeds; the root under `roots` keeps only its own module, `unread.py:655`; my 5:30 PM PST line said 654].
- The area `Proofs/NetworkCertifier` is capped at 2 modules and 6 declarations on the crossed tree [ran]. The side's own ratchet counted exactly that at `75918e18bbcb`: 2 modules, 6 declarations, 195 lines, one of 120 lines deletable alone [read: `r20261111-034141-2f06`, `out/unread-areas.txt` and `unread.md`]. By their lengths the two are `SampledAuditInstance.lean` (120 lines) and `Warden.lean` (75), three declarations each, and both are byte for byte the same on the crossed tree [ran `sha256sum`; that it is these two is argued, the report does not name them].
- So the area goes to 3 modules and 8 declarations, over its cap [argued; the ratchet reads the audits' facts and I ran none].
- What that does [read `lean_audit.py:285`]: in lean's records run it is bookkeeping, the cap is raised to the count and lean is warned. In a pull request's own check it fails.
- Three ways to close it, none done: lean raises the cap to 3 and 8 in the merge; or the two theorems move to the end of `WitnessToys.lean`, a read module, and `PeriodWitnesses.lean` goes (the count stays 2 and 6, but nothing reads them still); or your option 1.

**For the later issue** [ran `git grep`]: nothing but `PooledLeakAccepted_fails_at_one_unit` reads `pooled_bound_fails` (`PeriodWitnesses.lean:47`; the rest are its declaration and docstrings), and `fxQueue_fits` and `fxRows_wf` are read only by `PooledLeakAccepted_schedules`, `fxQueue_bucket` only by `fxQueue_fits`.

**NET-332** [ran]: my reading of both logs agrees with rc 0, each at `6a285960e2f2`. `r20261111-214832-b9c9`: "lean exit 0 in 34 s", none of its 39 theorems beyond the three axioms. `r20261111-215028-a30c`: 89 passed in 548 s, start to end 11 min 14 s; the slowest test is `test_lean_audit.py::test_service_verify_is_the_same_under_the_adapter` at 72.9 s, the period protocol's slow test 11.3 s.

**Least sure of:** the ratchet's count on the crossed tree, which rests on the side's report and on my reading of the seeds. **Where I did not look:** whether `main`'s merge added a reader of `SampledAuditInstance.lean` or `Warden.lean`; the handle registry.

Everything is pushed and the notes are synced (`tools/na-cross-mend-witnesstoys-leanmerge/`). I posted nothing on Linear.

[2026-11-12T01:47Z] c728ad422140
# NET-345: the crossed head and my branch, reader by reader

TERMGB to coordinator, Wed 11 Nov 2026, 5:47 PM PST. Read only: no commit but this note, no pull request, no remote run. Each claim is marked [ran], [read] or [argued]. The note is `20261112T0147Z-report-net345-table-termgb` (synced): it holds this table, the two sides' figures as run, and the two scripts.

## The answer first

- **Same numbers everywhere but one place.** `terms` and `clock_terms` at a named bound, all of K by links (198 figures) and all of `chain_record.leak` (108 figures) are equal to the last bit on one input [ran].
- **The one disagreement in meaning is d1's jitter.** The crossed head charges every audit the whole chain's steps; mine charges the chain's steps once. One audit: the same K. With n audits the crossed K is higher by (n − 1) · windows · buckets [ran].
- **The Lean period check sides with neither by itself: it charges what its case hands it.** Clock form: ⌈log2 Dj^steps⌉ for each audit, by that audit's handed `steps`. Rows form: ⌈log2 Dj^T⌉ once for each link-window's scan, whatever the audits are [read; ran on the Python mirror].
- **Two names for one bound meet whatever you rule.** `OF_RECORD` is #8777's (line 448 at its head `6b8edbf5de81`), `JITTER_BOUND` is the crossed head's. #8777 and lean's merge together put both in `leak_bounds.py` [read].
- **What the crossed head has and I never built: the scan's period check at the bound 2** (`scan_period.py:106`). On mine and on `main` it is handed 1 [read].

## What was compared, and how

- **Crossed**: lean's `cursor/lean-remote-merge-crossed-fc8d` at `3ff11d118f93`. For `leak_bounds.py`, `tests/test_leak_bounds.py` and `tests/test_chain_record.py` I took CROSSTICK's close, `a4e1d7e04617` on `cursor/na-cross-main-closes-84ef` (head `33360da0a088`).
- **Mine**: `cursor/na-k-two-schedules-b4e6` at `95ba862a3fb4`, four commits on #8777's earlier head `f28f4e4acc2d`.
- **How a figure was run**: each tree in its own process, its own files loaded by path, the same calls with the same arguments, the results compared key by key. The crossed side's `chain_record.py` is git's own merge of the crossed head with `main`, because on CROSSTICK's branch as it stands `chain_record.leak` raises (`leak_bounds` has no `world` until `main` is in) [ran].
- **The period check's numbers** are from each tree's Python mirror (`verity/protocols/certifier/verifier/network/tests/period_case.py`, `public`). The Lean was read, not built.
- Paths below are under `benchmarks/network_traces/` unless they name a directory. Crossed line numbers in `chain_record.py` are the crossed head's own; in the three closed files they are the close's.

## The table

### `leak_bounds`

| Reader | Crossed, file and line | Mine, file and line | Same number on one input | Names that differ |
|---|---|---|---|---|
| `terms` | `leak_bounds.py:182`, `dj: int = 1` | `:199`, `dj` required | Yes at a named bound: jitter 2,400 at 2 and 0 at 1, the other terms 700, 700, 7,952. With no bound: crossed counts at 1, mine raises `TypeError` [ran] | none |
| `clock_terms` | `:204`, `dj: int = 1` | `:224`, `dj` required | Yes at a named bound: jitter 480 at 2, 0 at 1. With no bound: crossed counts at 1, mine raises [ran] | none |
| the bound | `:99` `JITTER_BOUND = 2`, `:100` `JITTER` (the text) | `:126` `OF_RECORD = 2` (#8777's), `:458` `SET_TIME` (#8777's), `chain_record.py:167` `JITTER` | Both 2 [ran] | `JITTER_BOUND` / `OF_RECORD`; `leak_bounds.JITTER` / `chain_record.JITTER` |
| the helper | `:174` `_jitter`, and `:321` `jitter_bits(steps)` beside it | `:191` `_jitter` only | `_jitter(2, 32)` = 32 and `_jitter(2, 96)` = 96 on both; `jitter_bits` exists only on crossed, same values [ran] | `jitter_bits` has no name on mine |
| `LinkLoad` | `:231`; `:266` the property `jitter`; `:293` `terms` passes `dj=JITTER_BOUND`, no parameter; `:300` `bits` | `:251`; no property; `:308` `terms(…, dj=OF_RECORD)`; `:316` `bits(…, dj=OF_RECORD)` | Yes at the default: a week as registered gives jitter 6,048,000 on both; a block gives 32 on both; `bits` 36 of 36 equal. At D_j = 1: only mine can be asked [ran] | `link.jitter` / `link.terms(link.block_s, "registered").jitter` |
| `per_link` | `:374` | `:386`, takes `dj` | 36 of 36 equal [ran] | none |
| `datacenter` | `:386` | `:399`, takes `dj` | 18 of 18 equal [ran] | none |
| `at_link_rate` | `:423` | `:436`, takes `dj` | 108 of 108 equal, at 1, 12,500 and 46,870,000 links [ran] | none |

### `chain_record` and `warden_chain`

| Reader | Crossed, file and line | Mine, file and line | Same number on one input | Names that differ |
|---|---|---|---|---|
| the certificates step | `chain_record.py:220` `certs` writes `window_buckets`; `warden_chain.py:111` `certificates` writes no such field | `chain_record.py:228` `certs` writes `buckets` (`:253`); `warden_chain.py:128` writes `buckets` | Same value from `certs` (`setup.egress_params.buckets`). Each side reads the other's record as K open. On the network certifier's own chain the crossed d1 is open, mine is counted [ran] | `window_buckets` / `buckets` |
| `audit_of` | `:672`, no jitter key | `:679`, `"jitter": 0` (`:716`) | Same units, sampled and free: 4, 2, 96 [ran] | the key `jitter` |
| `network_terms` | `:748`: advice, syncs, cells | `:759`: those and `jitter` (`:770`), `None` where the step names no buckets | Advice 40, syncs 21, cells 0 on both; mine adds jitter 96 [ran] | the key `jitter` |
| d1's terms | `:758` `d1_terms`: `steps = len(rows) * window_buckets` (`:765`), each audit gets `steps` and `LB.jitter_bits(steps)` | `:773` `d1_terms`: `jitter_bound` (`:788`), the jitter in `network_terms` | **One audit: K 290 and 767 on both. Two audits: crossed 484 and 982, mine 388 and 886. Three: crossed 677 and 1,189, mine 485 and 997** (3 windows of 32 buckets, at δ = 1/2 and 2^-20) [ran] | `d1["K"]["audits"][i]["steps"]` only on crossed; `d1["jitter_bound"]` only on mine |
| `pooled_k` | `:729`: `jitter` is the audits' sum, `jitter_term: LB.JITTER` | `:739`: `network["jitter"]` plus the audits' sum (`:752`), `jitter_term: JITTER` | The same keys in `terms`; the sum differs as in the row above [ran] | the text of `jitter_term` |
| `leak` | `:804`; `:867` writes `jitter_per_block`, `jitter_bound`, `jitter_term` | `:815`; `:879` writes `jitter`, `jitter_bound`; the condition names the set-time label | 18 rows, 108 figures, all equal; first row 1,727,208,000 bits a week a link on both [ran] | `terms.jitter_per_block` / `terms.jitter`; the condition's text |
| `leak_markdown` | `:896` | `:911`, and one more line: the K at D_j = 1 under the set-time label | Same figures; the texts differ [ran] | the heading's words |

### `scan_period`, the tests, the text, `capacity.py`

| Reader | Crossed, file and line | Mine, file and line | Same number on one input | Names that differ |
|---|---|---|---|---|
| `scan_period.rows_case` | `scan_period.py:106`: `"jitter": LB.JITTER_BOUND` | `:94`: `"jitter": 1` (`main`: `:97`, the same) | No. The rows form at the bound 2 charges 32 a link-window of 32 buckets: K 384 against 288 on the mirror's input [ran on both mirrors] | none: it is step 3, which I did not build |
| tests, `leak_bounds` | `tests/test_leak_bounds.py:132-139`, `:194-197` | `tests/test_leak_bounds.py:135`, `:138-160` (the default pinned by signature, `dj` selectable), `:216-219` | The pinned figures agree: 9,176 = 9,144 + 32 on both [read] | as the readers' |
| tests, `chain_record` | `tests/test_chain_record.py:111`, `:519`, `:601`, `:620-633` (an audit's `steps` 180, jitter 180), `:759` | `tests/test_chain_record.py:116-119`, `:522-541`, `:615-620`, `:637-653` (jitter 96 once), `:779`; `tests/test_warden_chain.py:65-67` | Each pins its own shape of d1 [read] | as the readers' |
| tests, the scan | `tests/test_scan_zk.py:623`: `p["jitter"] == SP.LB.JITTER_BOUND` | no test pins the scan's handed bound; `main` has none either | not comparable [read] | none |
| README | `README.md:239-245`, `:297`: D_j = 2, `terms.jitter_per_block`, why D_j = 1 is not met, and that the scan's period check passes the same bound | `README.md:210-212`, `:228-230`, `:241-245`, `:298-300`: D_j = 2, `terms.jitter`, the set-time label, d1's jitter once | Both say T bits a sampling block at D_j = 2 [read] | as the readers' |
| `PROTOCOL.md` | `verity/protocols/certifier/verifier/PROTOCOL.md:989-999`, `:1085`, `:1233`: still "under J (D = 1)", "1 under J", "D = 1 is no jitter". Its rows are rewritten for the reference schedule; the rows "The shipped schedule, per window" and "A fixed clock, under J" are gone | the same file, `:987`, `:990-991`, `:1057-1060`, `:1066-1067`, `:1215-1217`: D = 2 the term of record, D = 1 only with a set-time card | [read] The crossed text does not say the term of record is at D = 2 | none |
| the attacks table | `verity/protocols/certifier/verifier/protocol/arbitrary-serving-attacks.md:13`: "Jitter is n·log2 D, which is 0 under J", unchanged from `main` | the same line: a bit a bucket at D = 2, 600 bits a window there, 0 with a set-time card | [read] | none |
| `capacity.accepted_bound_bits` | `verity/protocols/certifier/verifier/capacity.py:44-49` | the same lines, `main`'s | 121.0 at the default and 217.0 at D = 2 over 96 steps, on both [ran] | none. Only the docstring differs, below |

- **`capacity.accepted_bound_bits`, the crossed head's text of it** [read]: "Lean `Properties.NetworkCertifier.EncardAcceptedObservationsLe`, at `steps = W·T`: the check's accepted sequences number at most `#Ac · #Σ_sync^W · D^n`, which is `A_c + W·log2 #Σ_sync + n·log2 D` bits. With one sync value under J it is the content advice's `A_c` alone."
  - Mine, which is `main`'s: "Lean `SecurityProofs.EncardAccSeqsLeSync`: accepted sequences number at most `#Ac · #Sync^W · D^n`, which is `A_c + W·log2 #Sync + n·log2 D` bits. It is 0 for a fixed clock and J (`SecurityProofs.EncardAccSeqsLeFixedClock`)."
  - Both keep `jitter_classes: int = 1, steps: int = 0`, and both of its callers are tests that name no bound (`tests/test_network_certifier_capacity.py`, two lines). `EncardAccSeqsLeFixedClock` is in no file under `security/` or `verity/` of the crossed tree, and in seven of mine.

## d1's jitter: what the Lean period check charges

The file is `verity/protocols/certifier/verifier/lean/NetworkCheck/Period.lean` [read on both trees; the text of these definitions is the same but for the crossed head's extra `faultyEntries` term].

| | Mine, which is `main`'s | Crossed head |
|---|---|---|
| Clock form, an audit's terms: `AuditTerms.of`, `jitter := clog2 (jitter ^ a.steps)` | lines 1198-1200 | 1319-1321 |
| Rows form, a scan's terms: `Scan.terms`, `jitter := clog2 (jitter ^ s.steps)` | 1205-1207 | 1327-1329 |
| `terms`, one unit an audit | 1234-1239 | 1359-1362 |
| `rowsTerms`, one unit a link-window's scan: "The compute audits' own terms are in none" | 1245-1248 | 1371-1373 |
| a scan's `steps := T`, set by the parser | 794 | 862 |
| an audit's `steps` read from the case | 652 | 711 |
| what `steps` is: "`steps`, its monitoring period `k`" | 44-45 | 42-43 |

- **Clock form.** The check charges each audit ⌈log2 Dj^steps⌉ by the `steps` its case hands that audit. Two audits handed the same 96 steps at D_j = 2 are charged 96 each, 192 in all; handed 0 they are charged nothing at any bound [ran on both mirrors: jitter `[96, 96]`, and `[0, 0]`].
- **Rows form.** The check charges each link-window's scan ⌈log2 Dj^T⌉ once. It never reads an audit's `steps`. Three link-windows of 32 buckets at D_j = 2 are charged 32 each, 96 in all, with one audit or with two [ran on both mirrors: `[32, 32, 32]` in both cases].
- **The two designs, two audits over the same three windows of 32 buckets** [ran]:
  - Crossed: 96 for each audit, 192. K 484 at δ = 1/2.
  - Mine: 96 once. K 388 at δ = 1/2.
  - The crossed shape is the clock form's if every audit is handed the whole chain's steps. Mine is the rows form's total.
- **Which is right, as far as I can argue it** [argued]. The guarantee (`security/Definitions/NetworkCertifier/Guarantees/Period.lean:100`, `PeriodLeakAccepted`, the same on both trees) has one factor `Dj ^ k (sw i)` for each window i of the period (line 128), and `Public.units` says a unit is "one a execution phase".
  - So an audit is one execution phase's, with that phase's own steps, and the k of all units sum to the link's steps over the period: windows · buckets.
  - Mine gives that total. The crossed head gives it only with one audit, because it hands every audit all the windows' steps.
  - The crossed count is still an upper bound: it is sound and counts the same steps once for each audit.
  - This is what I am least sure of. If two audits can truly cover one execution phase (two Programs audited over the same wire), the statement as written still has a factor for each, and then the crossed count is the statement's and mine is below it.

## Row: the six docstrings of `security/Definitions/NetworkCertifier/Guarantees.lean` that say J is D = 1

Nothing changed. Mine is `main`'s file. The crossed head rewrites the file for the reference schedule (118 lines added, 143 removed against `main`) [read].

| Mine, line and statement | What mine says | The crossed head's text of it |
|---|---|---|
| 77, `EncardAccSeqsLeFixedClock` | "timing carries nothing beyond the jitter, and nothing at all under J (`D = 1`): the zero-capacity theorem, with content advice" | The statement is gone. No file under `security/` or `verity/` of the crossed tree names it. In its place `EncardAccSeqsLeAdviceBits` (line 67): "at most `2^(E·k) · D^n`, that is `k` bits per execution block plus jitter", with no clause on J |
| 103, `EncardAccSeqsLeConstantRateFixedClock` | "under J (`D = 1`) nothing at all, at most `#Ac` sequences for the content advice" | The statement is gone |
| 146, `EncardAccSeqsLeConstantRatePerWindow` | "`log2 #Σ_sync + T·log2 D` bits per window, which is 7 bits at 16 × 8 sync values under J (`D = 1`)" | Line 125: "`\|a\| + g·T·log2 D` bits per execution block, `\|a\|` under J (`D = 1`), whatever came before" |
| 192, `EncardIngressDecodableLe` | "`n·log2 D` of jitter, none under J's `D = 1`" | Line 173: the same words |
| 233, `EncardIngressObsDecodableLePerWindow` | "each window `D^T` of jitter, 1 under J" | Line 214: the same words |
| 263, the section's note on the calibration's numbers | "Σ_sync of 16 × 8 values, and J (D = 1)" | Line 244: "and J (D = 1)", without the sync set |

- One more on the crossed head that mine words differently: line 110, `EncardDecodableLeConstantRate`, "and under J only the advice's" (mine, line 132: "with a fixed clock under J only the content advice").
- So four of the six survive on the crossed head with J read as D = 1, and two leave with their statements.

## Row: the clock form's `steps: 0`

**Who hands an audit `steps: 0`** [read]:

| Producer | Crossed head | Mine | `main` |
|---|---|---|---|
| `benchmarks/prove_path/records_audit.py`, `period` | line 791 | 693 | 694 |
| `scan_zk.py`, `answers` | 1700 | 1503 | 1503 |
| `scan_period.py`, `rows_case`, which copies each audit's `steps` into the case | 105 | 93 | 96 |

- Tests pin the zero: `benchmarks/prove_path/tests/test_records_audit.py` (crossed 1124, mine 761, `main` 785) and `tests/test_scan_zk.py` (crossed 601 and 617, mine and `main` 539 and 555). The crossed head has one more, `tests/test_certifier_piece.py:579`.
- No other line outside tests hands an audit's `steps` on any of the three trees.

**What the period check then charges**:
- **Clock form: nothing, at any bound.** ⌈log2 Dj^0⌉ = 0. On the mirror an audit of 96 free bits has K 96 at D_j = 1 and at D_j = 2; handed 96 steps it would have K 192 at D_j = 2 [ran on both mirrors].
- **Rows form: the audits' `steps` are not read.** The scan's steps are T, so the jitter is charged by the bound the case hands: 2 on the crossed head, 1 on mine and on `main` [read].
- **Who builds a clock-form case: nothing in the tree.** The only writer of a certifier period case outside tests is `scan_period.rows_case`, the rows form, on both trees [read].
- **Who still takes one**: `chain_record.period_binding` refuses a rows-form answer (crossed line 982, mine 1000), so the block record's certifier entry (`block_entry`, crossed 1006, mine 1024) carries a K only from a clock-form answer. With `steps: 0` that K has no jitter at any bound, while d1's own K beside it charges the jitter on either design [read].
- **The answers of record**, by named fields of `tests/fixtures/certifier-period-answers.json` [read]: the clock-form answer's audit has jitter 0, with K 2,178,452 at δ = 1/100 and 2,180,068 at δ = 1/1,000,000; the rows-form answer's scan has jitter 0 too, with K 12,200,606 and 12,202,528.

## The three short lists

**What my branch has that the crossed head lacks**
1. **No default on `terms` and `clock_terms`.** The crossed head and `main` default to D_j = 1, so a caller that names no bound counts at 1 silently [ran].
2. **A bound the caller can name on K by links**, default the term of record's. The crossed head fixes it inside `LinkLoad.terms`, so no figure at D_j = 1 can be asked [ran].
3. **The set-time label**: the K at D_j = 1 printed only under `SET_TIME`, in `leak`'s condition and one line of `leak_markdown` [ran].
4. **The buckets on the network certifier's own chain** (`warden_chain.certificates`), so d1's K is counted there. On the crossed head it is open on that path [ran].
5. **The text**: `PROTOCOL.md` in six places and the attacks table's row 3. The crossed head's still read J as D = 1 [read].
6. As you expected, all four. The attacks table is one line.

**What the crossed head has that mine lacks**
1. **The scan's period check at the bound 2** (`scan_period.py:106`, pinned by `tests/test_scan_zk.py:623`). This is NET-345's step 3 [read].
2. **Why D_j = 1 is not met, in the constant's text**: "next-bucket host-state channel". Mine says "assumed (J)" and names the set-time card [read].
3. **Each of d1's audits carries its `steps` and its own jitter in the record** [ran].
4. `jitter_bits` and `LinkLoad.jitter`, two helpers that return what mine gets from `clock_terms` and `LinkLoad.terms` [ran].
5. The README's sentence that the scan's period check passes the same bound [read].
6. Not jitter, in the same files: the period check's `faultyEntries` term, `rows_case`'s `v3c` argument, `PROTOCOL.md`'s and `Guarantees.lean`'s reference-schedule rewrite [read].

**Where the two disagree in meaning, not in name**
1. **d1's jitter**: for each audit over all the chain's steps (crossed), or once over the chain's steps (mine). Equal at one audit; (n − 1) · windows · buckets apart at n [ran].
2. **A caller that names no bound**: counted at D_j = 1 (crossed), or refused (mine) [ran].
3. **A K at D_j = 1**: never given (crossed: "not met"), or given under the set-time label (mine). Both say no run has met D_j = 1 [read].
4. **Which field of the certificates step holds the buckets.** It is a name, but each side reads the other's record as K open, so a record written under one design is uncounted under the other [ran].
5. Not a disagreement of designs: the scan's bound, 2 against 1, is step 3 unbuilt on mine.

## Where the two would meet in a merge

My four commits against the crossed head merged with `main`, file by file with `git merge-file`, nothing written to a branch [ran]:
- Against CROSSTICK's close, base `main`: `leak_bounds.py` 2 conflicts, `tests/test_leak_bounds.py` 2, `tests/test_chain_record.py` 5.
- Against the crossed head, where `main` and it merge cleanly: `chain_record.py` 8, `README.md` 2, `PROTOCOL.md` 3.
- None in `warden_chain.py`, `tests/test_warden_chain.py`, `tests/test_certifier_term.py` or the attacks table.
- So six files, 22 places, and each of the 22 names the jitter, its bound or the buckets (checked by keyword).

## Least sure, and where I did not look

- **Least sure**: the argument on which d1 count the guarantee means (above). I read the statement and `Public.units`' docstring; I did not read the proofs, nor lean's proofs commit on the crossed head.
- **The crossed side I ran is assembled**, not a whole merged tree: CROSSTICK's close for three files, git's merge with `main` for `chain_record.py`, the crossed head's own files for the rest. A whole merge of the crossed head with today's `main` stops on 49 files, among them `Period.lean` and `period_case.py`, so the mirror I ran for the crossed side is the crossed head's own, before that merge.
- **The Lean was read, not built or run.** Every number for the period check is the Python mirror's.
- I did not look for what built the clock-form case of the fixture's answer of record.
- I read `PROTOCOL.md` on the crossed head only at the lines that name J or D, not whole.
- I did not read the callers of `records_audit.period` and `scan_zk.answers` past the producing line.
- I did not check the onsite stand-ins or `onsite_record.py` (ONSITEMAIN's branch), as ruled.

