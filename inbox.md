[2026-11-11T07:04Z] 215d1b8de324
PERIODPROOF to coordinator (network-accounting), Tue 10 Nov 11:04 PM PST. NET-293 carry: the proof and the exclusion are carried and pushed, both checks against #8829 come out clean. Three entries and one thing to decide.

**To decide (README over its cap, by 9 and 15 bytes).** `security/Proofs/NetworkCertifier/README.md` is 49,161 bytes at the proof's head and 49,167 at the exclusion's; the cap is 49,152 (`MD_SIZE_CAP`, `tests/test_repository.py:48`) [run: `git cat-file -s`]. Before the move they were 49,143 and 49,149. The 18 bytes are `main`'s, from before the move: `main`'s README is 46,240 at my old base `5482988e3496` and 46,258 at the move's base `2eb7b4aa735f` and since [run]. The statement's is 49,037, under. The cap is bookkeeping: the root suite warns and drive's warn gate tells the owner (`tests/conftest.py:1-8`, `tools/verity/check/tests/test_markdown_caps_gate.py:1-2`) [read]. I did not fix it: a trim changes the branch's own change, and the proof's runs are on its head. If you want it before those two pull requests open: one commit on the proof that cuts 16 bytes or more from its own README lines, merged into the exclusion; only the root suite reruns. Say so and I do it.

**1. Proof, `cursor/net293-period-proof-9631`: head `ff5d7a3d5aec34afc0ed07c062a45c92009c9ec6`.**
- [run] Step 1: `restack.py` over both moves onto `5c095bb4ab7012ba359e0a85f4ecf19892b99629`: `clean`, result `1447f8a0f528bde22b9ed04a5e86e318ff120dd5`, 122 s (a plain merge conflicted in 6 files: the tool's comparison only).
- [run] Step 2: `git merge --no-ff b1f95afc2e02…` (the carried statement, which holds #8537's `74145a920b7f`): tree `676c285ca1c4`, the result's tree, unchanged.
- [run] Step 3: `security/lean-audit.json` is 4,190,437 bytes, under the limit by 3,867, so the script rewrote it: lines naming `Security/` 6, `verity/core/` 0, `verity/ml/` 0, `benchmarks/nci/` 0, `integrations/vllm` 0, as `main`'s. No repair. The Properties and Proofs locks are `main`'s (1,753,345 and 29,034 bytes). By entry: 18 differ from the carried statement's lock (14 added, 1 removed, 3 changed), the same 18 with the same values as `ccc28b59d9..509503d35d`; 37 differ from `main`'s, the same 37 with the same values as before.
- [run] Step 4: against the statement, before `ccc28b59d9..509503d35d` and after `b1f95afc2e..ff5d7a3d5a`: 10 files, +838 −324 both, 10 of 10 at their moved place, same status and counts each, 0 lines rewritten. Against `main` (before `5482988e3496..509503d35d`, after `5c095bb4ab70..ff5d7a3d5a`): 30 files, +2343 −420 both, 30 of 30, 2 lines rewritten (the `UNWITNESSED` keys line of `tests/test_lean_packages.py`, #8537's). No line of its 9 files other than the lock names a retired path.
- [run] Step 5: pushed 10:54 PM PST, a fast-forward `509503d35d..ff5d7a3d5a`. Nothing by hand but the merge.
- [run] Its runs, launched 10:55 to 10:56 PM PST, both running: suites `r20261111-065518-826e`, records `r20261111-065546-b150`.

**2. Exclusion, `cursor/net293-period-exclusion-7c2e`: head `03780e9dc98ca33a528dc014b7a99b9681bf0cd1`. The trap hit; repaired by hand in one commit.**
- [run] Step 1: `restack.py`: `clean`, result `d226ae08890fce521712f3c8d19df88a078df520`, 128 s (plain conflicted in 8). Its lock was 4,197,931 bytes, over the limit: lines naming `Security/` 329, `verity/core/` 7, `verity/ml/` 1, `benchmarks/nci/` 3, `integrations/vllm` 0.
- [run] Step 2: `git merge --no-ff ff5d7a3d5aec…` (the carried proof): `4ec147d54fd5`, tree `f715ab932eea`, the result's tree, unchanged.
- [run] Step 3, the repair, commit `03780e9dc98c`, by FAILFORMS's rule. An entry is the branch's own if its value at `1549d745f012` differs from its value at the old base `5482988e3496`: 41 entries. Every other entry that differed from `main`'s took `main`'s value: 336, of which 326 `shape.frozen` reasons and 10 `shape.conjectures`, the counts #8829's commit gives; the diff is 340 lines each way, as its is. All 336 are in both locks, so I edited in place and each section keeps its order; the file is written through `check.text`, which gives the unrepaired lock back byte for byte, so the commit changes those entries only. None of the 41 names a retired path, and `main` changed none of the 41 since the old base. No `guarantees` or `reads` record changed.
- [run] After it: 4,197,998 bytes; lines 6, 0, 0, 0, 0. `git diff origin/main -- security/lean-audit.json` is +404 −62: 41 entries (19 added, 1 removed, 21 changed), all the branch's own (NET-293's and #8537's), each with its value from before the move. `origin/main` (`d66346efc8f6`) holds the lock blob of `5c095bb4ab70`. The lock is the one file in the tree over the limit. Against the carried proof 6 entries differ, the same 6 with the same values as before.
- [run] Step 4: against the proof, before `509503d35d..1549d745f012` and after `ff5d7a3d5a..03780e9dc9`: 6 files, +711 −16 both, 6 of 6 at their moved place, 0 lines rewritten. Against `main`: 32 files, +3040 −422 both, 32 of 32, the same 2 lines rewritten. No line of its 5 files other than the lock names a retired path. The tree differs from the restack result's in the lock alone.
- [run] Step 5: pushed 11:00 PM PST, a fast-forward `1549d745f0..03780e9dc9`. First-parent history: the restack result, the merge, the repair.
- [run] Its runs, launched 11:00 to 11:01 PM PST, both running: suites `r20261111-070036-6f8d`, records `r20261111-070113-e06f`.
- [argued] The lock is still over the limit, so a branch cut from this head that crossed the moves again would be skipped again. Nothing should: the head is past both moves.

**3. The two checks against #8829 (`1358bfbdd1b0`), read only.**
- [run] `git merge-tree --write-tree 03780e9dc98c 1358bfbdd1b0` exits 0, clean, and its tree is `56375677ff1643e1076d6cb7a64ed3d18f8c9476`, the tree of `1358bfbdd1b0`. No file differs. The other way round gives the same tree. Merge bases: `5c095bb4ab70` and `1549d745f012`. The lock's merge driver is registered in my checkout (`merge.lean-audit.driver` is set), so this is the driver's merge.
- [run] The lock by entry: my 41 entries are all among the 56 where #8829's lock differs from `main`'s; 39 have the same value there; the other 2 (`reads.Definitions.NetworkCertifier.Warden`, `reads.Definitions.NetworkCertifier.Witness`) #8829 changes further itself; 15 more are #8829's alone. That is its body's 17 and 39.
- [argued] So #8829 takes the carried exclusion by a plain merge that leaves its tree as it is.

**Statement's runs:** `r20261111-065107-9f3f` (suites) and `r20261111-065152-f7ba` (records) are running; no verdict yet. Its pull request opens when both are green.

Not checked yet: nothing was built at any of the three heads beyond what the six runs will do. Least sure of: nothing new here; the README's cap is the one open point.

[2026-11-11T07:07Z] 6e106c8c0b98
LEANMERGE to coordinator, NET-325, Tue 10 Nov 11:07 PM PST, to your 10:57 PM. The merge is pushed and the six runs are out.

**The merge.** `cursor/net325-drawn-le-units-22f3` is at `0b12ac8cd69f39214dfa05273a0c7dc7a7093a52`, a merge commit of `750292d4cfe3` and `f5b8707812871f312448b16f039cccea773388f6`, pushed as a fast-forward at 11:01 PM PST [run: `git push`; `git ls-remote` gives that head].
- **No conflict, and the merge touched none of my five files** [run: git]. The base has changed none of them since `2f30e1709f1e`, where my branch was cut and which the base holds; the five are byte for byte as at `750292d4cfe3`.
- **The locks are the base's, whole** [run: `git diff` of the merged head against `f5b870781287` over every `lean-audit.json` and `unread-caps.json`: empty]. I wrote none, and no merge driver ran on one: only the base's side had changed them.
- The merged tree is the base's tree with my five files and nothing else [run: `git diff --name-only f5b870781287 HEAD` lists exactly `Security/Proofs/Properties/NetworkPeriod.lean`, `Security/Properties/NetworkPeriod.lean`, `verity/core/protocols/network/lean/NetworkCheck/Period.lean`, `verity/core/protocols/network/tests/period_case.py`, `verity/core/protocols/network/tests/test_network_period_protocol.py`].
- The base holds no layout move [run: git]: it has `verity/core/protocols/network` and no `verity/protocols`, and its merge base with `origin/main` is `2b869ed35db5`, my old head's own. `main` is not merged.
- **My list at the merged head is the same three Lean files** [run: `git grep -w` for `factsOf`, `auditCheck`, `auditChecks`, `chargedOf` over `*.lean`]: the statement's, the proofs' and `NetworkCheck/Period.lean`. Nothing the base brought names any of the four.

**The runs, each at `0b12ac8cd69f`, `--on auto`, sent with `main`'s tool from a worktree at `origin/main` (`d66346efc8f6`) with my branch's worktree as the source:**

| Run | What | Placed on | State at 11:08 PM PST |
|---|---|---|---|
| `r20261111-070505-89cf` | `verity lean try verity/core/protocols/network/lean/NetworkCheck/Period.lean` | vy-nebius-cpu-7 | ended: "lean exit 0 in 13 s"; "axiom probe: no theorems in it"; "exit 0 in 68 s wall" |
| `r20261111-070350-1452` | `verity lean try Security/Properties/NetworkPeriod.lean` | vy-nebius-cpu-3 | launched |
| `r20261111-070432-b967` | `verity lean try Security/Proofs/Properties/NetworkPeriod.lean` | vy-nebius-cpu-7 | ended: "Build completed successfully (944 jobs)."; "lean exit 0 in 141 s"; "axiom probe: none of its 39 theorems uses an axiom beyond propext, Classical.choice and Quot.sound"; "exit 0 in 182 s wall" |
| `r20261111-070311-18ac` | the period check's Lake package built with its `#guard`s (`lean_changed.py --build`, its 13 files) | vy-nebius-cpu-8 | launched |
| `r20261111-070538-d815` | `suites.py verity/core/protocols/network verity-core-composition verity-network-traces` | vy-nebius-cpu-7 | launched |
| `r20261111-070615-378a` | `lean_changed.py --records Security/Properties --update`, `--cwd clone --declared-output 'lean-audit/*'` | vy-nebius-cpu-1 | launched |

No two of them do the same thing. The records run commits nothing: in a recorded run `--update` leaves each changed lock under the run's `lean-audit/record/`, which I read against the base's lock when it ends.

**Job 2** I start now, by reading only, while the runs go; it goes to you with or after job 1's report and does not hold that report. The note is `lanes/network-accounting/20261111T0217Z-state-net325-leanmerge.md`.

[2026-11-11T07:08Z] c50a7b142557
ONSITEMAIN: NET-331 read 1, done. [read] = the kept `audit/window/summary.json` on vy-nebius-cpu-12; [run] = the fetched run's sampler. Detail appended to the NET-331 note.

- **Compose's share.** `staging.before_draw_s` 16,488.9; 63 circuits, `compose_s` summed 16,305.0 [read]: 98.9% of staging, 96.3% of stage 2. The rest of stage 2: 183.9 s for the members' public files (the difference; no field times them), 218.0 s the verifier's window, 223 s unattributed. A saves 16,305 s and leaves 625 s; the path after A to E is about 6,900 s, 1.9 h [argued].
- **Largest compose.** 840.2 s, `Int8RowQuant_v3`, `k_log` 26. The next five are 716 to 650 s, `AttnBlockFirst_v2`. Median 229.7 s, p90 626.6 s [read].
- **Memory.** The audit process peaked at 61.9 GiB, inside the largest compose, over a floor of 23 to 31 GiB: that compose took 31 to 39 GiB. Eight others took at most 7 to 16 GiB each. The 35 `AttnBlockFirst_v2` composes each added under 1 GiB [run; boundaries placed from `compose_s`, good to 3 min].
- **N: 24.** In submit order the warm takes 1,374 s at 16, 993 s at 24, 932 s at 32; floor 840 s; plus the 218 s window [argued]. Memory at 24: under 200 GiB by the measured needs, since one compose is large, not sixteen [argued].
- **Disk and window.** 487 GB free of 1,590 GB: 35 GB under the 522 GB the circuits take, so the warm does not fit there as it is. Serve dir 56.7 GB; window 11.5 GB with its registration, circuit and window files; audit 28.8 GB; `period.json` present [read]. Not held to its sums.
- **Not found:** bytes per circuit; a timer for the public files; whether `warm` runs from the kept window alone; the disk's write rate with 24 writers.

