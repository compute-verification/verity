Lane circuits

**Not marked: this crosses the weekend's move before its mark.** It is prepared on `main` (`1e4d5c993c08` merged; head `d9b1f18a28fef36e24a3c3d6b6f6031236c943a6`). The chain that takes the served circuits out of `verity/definitions/` rewrites 207 of the list's 239 `file` strings. Once that chain is on `main` the branch is carried across it with `tools/move/restack.py`, `--prune` puts the list back in order, and the lead marks it then.

**What changes.** circuit-check's suite gains one test, `tools/verity/circuit_check/tests/test_superseded_versions.py` (5 s, no torch). It reads the registries the way circuit-check does, in a fresh interpreter, and fails when a Definition version is superseded and `tests/lint/allowlists/superseded_versions.json` doesn't list it. The message gives the three ways through and prints the entry line:

- delete `Name_vN` with the `Name_vN+1` that supersedes it;
- delete another superseded version and give `Name_vN` its entry;
- name, in the list's `held` table, the workload of record that still names `Name_vN`.

The list is exact (a deleted version takes its entry with it; `--prune` does that and adds nothing), and it is one of the tree's ratchets by path, so nothing new holds it: the ratchet gate reports a change that leaves its entries more than it found, the allowlists' merge driver merges it, and the public export drops the entries of the files it doesn't copy.

**What it covers.** Every registered Definition, wherever its module is. The registries circuit-check loads are every module with a sidecar (an adapter's package among them), each module of the served registry and the numerics rows, so a protocol's circuits count as the catalog's: 433 of the 2,249 registered Definitions are defined under `verity/protocols`, and 31 of them are in the list (`CompileSlice_v1` to `_v6`, `CacheStep_v1` and `_v2`, `CompileVerdict_v1`, `CompileWindow_v1` among them); one more is the adapter's (`SeedCheck_v1`). A change that replaces the compile-slice modules with one therefore shrinks the list, with `--prune`. Not covered: a Definition made with `register=False`, which has no id in the registries, and a module or a Program named `_vN`, since the count is of Definition ids and not of file names.

**What it deletes.** Nothing: the freeze. It changes no Definition. **What it lets the lane delete.** The list is the to-do, and it can only get shorter: 239 versions in 64 files at `main` `1e4d5c993c08`, with an empty `held` table.

**The counts, made again on Fri 13 Nov.**

| | Tue 10 Nov | Fri 13 Nov (`main` `1e4d5c993c08`) |
|---|---|---|
| registered Definitions | 1,988 | 2,249, of them 666 Boolean versions |
| superseded and listed (a higher version of the name and kind) | 221 in 56 files | 239 in 64 files |
| names with a listed version | 107 | 122, of them 53 with more than one |
| superseded under the rule as briefed (any higher version of the name) | 731 (Mon 9 Nov) | 832 |

No entry of the 221 is gone. The 18 added are landings of the three days in which nothing held the count, in 17 names. Two names that were listed grew: `CompileSlice` from 5 to 6 and `NvLogf` from 1 to 3. Fifteen are new to the list with one version each: eleven of the DSA top-k mask and norm (`DsaMaskPositions`, `DsaPositionsMerge`, `DsaRMSNorm`, `DsaRMSNormApply`, `DsaRMSNormSumSq`, `DsaRadixPass`, `DsaRadixTies`, `DsaRowValid`, `DsaTopKDigitPick`, `DsaTopKMaskBlock`, `DsaTopKMaskWhole`), `SparseMlaHeadBlock`, the certifier's `CompileSliceRowGroups`, and PoUB's `BlockOpeningCheck` and `LeafMatch`. By place the list holds 207 entries under `verity/definitions`, 31 under `verity/protocols` and 1 under `adapters/vllm`.

The family table classes the 220 it was made on (`main`, Mon 9 Nov): 120 revisions (17 with no caller to move, 64 once their callers move, 41 once their workloads of record are pinned again, of the table's 122), 93 variants under a version number (another kernel, port, input or cut) and 7 undetermined. The 19 listed since are not classed. The variants and the undetermined stay on the list as they are; the retire list says which of them get names of their own.

**What "superseded" means: decided, a higher version of the name and kind.** The lead chose the second row below (`kind`, on the issue). The branch holds both rules as separate commits (`e0c39fd1df` as briefed, `b1ffed34d9` the chosen one). This table is as measured on the landings of 4 to 9 Nov and is not made again; today's counts under each rule are in the table above.

| | superseded when | at `main` on Mon 9 Nov | landings it would have failed, 4 to 9 Nov |
|---|---|---|---|
| as briefed (`e0c39fd1df`) | any higher version of the name is registered | 731 | 21 of the 33 that changed the registry's ids (+51 versions) |
| the same-kind rule (`b1ffed34d9`), chosen | a higher version of the name and kind is registered: a Boolean version registered with the word Definition it computes (`word=`) supersedes no word | 220 | 3 of the 33 (+5 versions) |

18 of the 21 only added a Boolean version beside its word (35 words new in the same landing, 7 older), which is the catalog's stated convention (`GLOSSARY.md`: the word library migrates to Boolean Definitions under new versions of the same names). For those the briefed rule has no fix to print: the word can't be deleted (its Boolean version is held to it), no workload of record names a new word yet, and deleting another version waits on the freeze. The second rule fails the three landings that left a version behind under a higher one of its kind (`Int8RowQuant_v3`, `_v4` and `SelectCollect_v1`; `RMSNormFusedCudaVector8_v1`; then its `_v2`).

**Process pricing** (the process-design skill).

**Failure.** Superseded versions accumulate, and each stays in the registries, its pins and circuit-check's work until someone deletes it. Under the chosen rule the count went from 221 to 239 between Tue 10 and Fri 13 Nov; under the rule as briefed, from 731 of 1,988 registered versions to 832 of 2,249.
**Mechanism.** A code check in the suite that owns the registries' reading, on the tree's existing shrink-only lists. A label alone (the ratchet gate, off the merge gate) warns and reverts nothing, so it would count the growth without stopping it.
**Walkthrough** (as measured on the definitions registry's pinned ids, 4 to 9 Nov, 33 landings that changed them; not made again):

| Change type | Count | What the process adds | Blocks or labels | Who acts |
|---|---|---|---|---|
| adds or removes Definitions, no version left behind | 11 | nothing | – | – |
| adds a Boolean version beside its word | 18 | nothing (as briefed it would have been a failure with no fix that applies) | – | – |
| adds a version and leaves a lower one of its kind | 3 | delete it, swap an entry, or one `held` line | blocks | author |
| deletes a superseded version | 2 (one also among the 18) | `--prune`: one line removed | blocks until pruned | author |

**Price per week.** Human: none. Merge path: 5 s in a suite that already reruns on any registry change; one new suite input, the list, which changes only with the registries. Shared files: the list, edited by the five landings of the last two rows in the days measured; entries merge by the driver, two `held` lines added at once conflict. Always-read words: none.
**Tests.** 4, blocks rather than labels: a label would not hold the count. 6, a shared file: only a change that leaves or deletes a superseded version touches it. 8: a `held` line is the default that proceeds, and it is a statement nothing checks (the index of record holds Programs only by digest).
**Owner and exit.** circuits; it retires when the list is empty, or when variants get names of their own and a version number means a revision only.

**What it touches of other owners.** `tools/verity/circuit_check/pyproject.toml`, three places: the list as the last path of the suite's `inputs` (lines 206 to 208), and the test's name in the two reads groups of the tests that load the registries (lines 230 and 259). It adds no path to either group. `tests/README.md`: one paragraph after "Lint ratchets" (lines 35 to 41).

**Against the queue on Fri 13 Nov.** The test at this head was merged with every pull request the queue listed at 03:24Z (257 open, 58 of them marked), each of the six live candidates, top's staged move branch and `main`: one scratch worktree a head, the test file alone, 264 heads. At 04:17Z the heads that were new or had moved were run the same way, 101 more.

| heads, at 03:24Z | count | merge and pass | merge stops against `main`; pass with their own side of each conflict | would fail | merge stops, no verdict |
|---|---|---|---|---|---|
| marked, admitted | 41 | 41 | 0 | 0 | 0 |
| marked, refused or waiting | 17 | 1 | 16 | 0 | 0 |
| live candidates and `main` | 7 | 7 | 0 | 0 | 0 |
| top's staged move branch | 1 | 0 | 1 | 0 | 0 |
| not marked | 198 | 87 | 106 | 0 | 5 |

- **No marked pull request would fail it, and no candidate does**: 0 of 58 at 03:24Z and 0 of 68 at 04:17Z, when all 58 admitted ones merged and passed.
- The 16 marked ones whose merge stops stop against `main` alone on the same files (ten of them are refused for exactly that); none of those files is one the test reads, so the verdict doesn't depend on the side taken.
- Two controls merged the same way fail as they should: a Definition change that leaves `F32Sqrt_v1` superseded, and the list without `NvLogf_v2`'s entry.
- #9375 is on `main` and in this head: the test names neither renamed class and reads the table through `REGISTRY`.
- #8415 (not marked, on a tree older than the layout's move) retires `CompileSlice_v2`, `CompileSlice_v3` and `CacheStep_v2`, all three listed. Brought up to `main` after this lands, it needs `--prune`.
- The five with no verdict are older trees than the layout's move, whose side of a conflict doesn't import.
- The weekend's move, rehearsed: `r20261113-032839-64dc` ran the four maps of `cursor/top-moves-c-b396` (`680f1541f8c3`) on this head merged with that branch. After the first three and after the fourth, every one of the list's 239 `file` strings names a file, the test is `3 passed`, `--prune` changes only the order of 25 lines, and the test file's graph task passes with no undeclared read. The maps rewrite 207 lines of the list (183 by the served map's `registry/` row, 24 by the kept map's `ops/` row) and no line of the test.

Every pull request by number, lane and state is in `scratch/CIR-208/inflight.md` on `cursor/circuits-cir208-scratch-6f04`, and every path string the test and the list hold, with the map row that moves each, in `scratch/CIR-208/paths.md` there.

**Checked at this head.**

- `r20261113-032650-fda7`: the repository suite, `392 passed in 380 s`.
- `r20261113-032715-ccf0` (48 GB): circuit-check's suite, `439 passed, 5 skipped in 640 s on 5 workers`, 0 cached; it read nothing outside its inputs.
- The test file's graph task: `pass`, `undeclared_reads: []`. Of the 772 tracked files the test opens, none lies outside the suite's inputs; with `main`'s `pyproject.toml` exactly one does, the list.
- The ratchet gate (`origin/main` to this head): `failures: []`. Retired terms: no changed file gains a site.

**Limits.** The list is exact against the registries of the tree it is in. So a pull request that adds or deletes a superseded version and is marked before this lands passes its own check and fails this test in the candidate that holds both. There the failure names this change's files, so by the queue's rule for suspects (the pull requests whose change touches what the failure names) this is the one ejected; once this is on `main`, the other is the one found. That is read from the rule, not run. The ratchet gate, which holds the list shrink-only, is off the merge gate and warns, as for every allowlist; at PR time the test holds the list to the registries, so growth is a visible added line in the diff. A module that moves takes its entries' `file` with it (`--prune`). A new Definition that is another kernel, port or input rather than a revision has a fourth way the message doesn't print, a name of its own; printing it waits on the design for variants' names.

Refs CIR-208
Goal circuits/slice-clean
