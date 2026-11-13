Lane circuits

**Not marked: the mark is the lead's.** Carried across the 28 layout moves of Fri 13 Nov and brought up to `main` `c68135b49d63`; head `bc64a9ad6fb448de241501c93c14fc0eadcfebad`. The lead ran `tools/move/restack.py --kept --no-pins` on the old head `d9b1f18a28fe` onto `4ae17f7eafd7` (`r20261113-155530-28ef`, result `64ba3e81a113`). On that result `main` is merged (`483586d432a2`) and the list is made again from the registries (`bc64a9ad6fb4`). One move still ahead of the list: `tools/move/definitions_served_map.toml` is on `main` and has not run, and it takes `verity/definitions/registry/` to `adapters/vllm/verity_vllm/definitions/registry/`, which rewrites 183 of the list's 236 `file` strings.

**What changes.** circuit-check's suite gains one test, `tools/verity/circuit_check/tests/test_superseded_versions.py` (no torch, 0.3 GB). It reads the registries the way circuit-check does, in a fresh interpreter, and fails when a Definition version is superseded and `tests/lint/allowlists/superseded_versions.json` doesn't list it. The message gives the three ways through and prints the entry line:

- delete `Name_vN` with the `Name_vN+1` that supersedes it;
- delete another superseded version and give `Name_vN` its entry;
- name, in the list's `held` table, the workload of record that still names `Name_vN`.

The list is exact (a deleted version takes its entry with it; `--prune` does that and adds nothing), and it is one of the tree's ratchets by path, so nothing new holds it: the ratchet gate reports a change that leaves its entries more than it found, the allowlists' merge driver merges it, and the public export drops the entries of the files it doesn't copy.

**What it covers.** Every registered Definition, wherever its module is. The registries circuit-check loads are every module with a sidecar (an adapter's package among them), each module of the served registry and the numerics rows, so a protocol's circuits count as the catalog's: 426 of the 2,258 registered Definitions are defined under `verity/protocols`, and 28 of them are in the list (`CompileSlice_v1` and `_v4` to `_v6`, `CacheStep_v1`, `CompileVerdict_v1`, `CompileWindow_v1` among them); one more is the adapter's (`SeedCheck_v1`). A change that retires compile-slice modules therefore shrinks the list, and #9682 did, by three entries. Not covered: a Definition made with `register=False`, which has no id in the registries, and a module or a Program named `_vN`, since the count is of Definition ids and not of file names.

**What it deletes.** Nothing: the freeze. It changes no Definition. **What it lets the lane delete.** The list is the to-do, and it can only get shorter: 236 versions in 61 files at `main` `c68135b49d63`, with an empty `held` table.

**The counts, made again on Fri 13 Nov before and after the moves.**

| | Tue 10 Nov | Fri 13 Nov, old head (`main` `1e4d5c993c08`) | Fri 13 Nov, this head (`main` `c68135b49d63`) |
|---|---|---|---|
| registered Definitions | 1,988 | 2,249, of them 666 Boolean versions | 2,258, of them 673 Boolean versions |
| superseded and listed (a higher version of the name and kind) | 221 in 56 files | 239 in 64 files | 236 in 61 files |
| names with a listed version | 107 | 122, of them 53 with more than one | 122, of them 52 with more than one |
| superseded under the rule as briefed (any higher version of the name) | 731 (Mon 9 Nov) | 832 | 836 |

From the old head to this one, the list made again differs from the list as it was in two ways only:

- **Versions that left: three.** `CacheStep_v2`, `CompileSlice_v2` and `CompileSlice_v3`, deleted with their modules by `3d1ed7adcb0b`, on `main` as `053b8e50fe8a` (#9682). Each had a file to itself, so the files go from 64 to 61. **Versions that entered: none.**
- **Moved paths, same version: 52.** 24 from `verity/definitions/ops/` to `verity/circuits/ops/` and 28 from `verity/protocols/X/` to `verity/protocols/accounting/X/` (pouw 16, certifier 10, poub 2). The other 184 entries are as they were (183 under `verity/definitions/registry/`, the adapter's one).

Made from the registries with no read of the carried list, the text is the carried list less those three lines, byte for byte. The registries themselves gained 18 ids and lost 9 between the two heads, and none of the 18 is superseded: seven `Fa2TrainBwd` words with one Boolean `_v2` each (#9417), `Fp8DsMlaUpconvertRow_v1` (#9317), and `GreedySelect_v1` with `GreedyChunk_v2` and `GreedyCombine_v2`, whose `_v1` left in the same landing (#9075). The other seven that left are the three retired versions and the four `CachePool` Definitions of `CacheStep_v2`'s module (#9682).

From Tue 10 Nov to the old head: no entry of the 221 was gone, and 18 were added, landings of the three days in which nothing held the count, in 17 names. Two names that were listed grew: `CompileSlice` from 5 to 6 and `NvLogf` from 1 to 3. Fifteen were new to the list with one version each: eleven of the DSA top-k mask and norm (`DsaMaskPositions`, `DsaPositionsMerge`, `DsaRMSNorm`, `DsaRMSNormApply`, `DsaRMSNormSumSq`, `DsaRadixPass`, `DsaRadixTies`, `DsaRowValid`, `DsaTopKDigitPick`, `DsaTopKMaskBlock`, `DsaTopKMaskWhole`), `SparseMlaHeadBlock`, the certifier's `CompileSliceRowGroups`, and PoUB's `BlockOpeningCheck` and `LeafMatch`.

By place the list now holds 183 entries under `verity/definitions/registry`, 28 under `verity/protocols/accounting`, 24 under `verity/circuits/ops` and 1 under `adapters/vllm`.

The family table classes the 220 it was made on (`main`, Mon 9 Nov): 120 revisions (17 with no caller to move, 64 once their callers move, 41 once their workloads of record are pinned again, of the table's 122), 93 variants under a version number (another kernel, port, input or cut) and 7 undetermined. Three of the 220 are deleted since (the three above), and the 19 listed after it are not classed. The variants and the undetermined stay on the list as they are; the retire list says which of them get names of their own.

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

**Price per week.** Human: none. Merge path: one load of the registries in a fresh interpreter (3 s on a warm tree, 14 s under the read guard in a fresh clone), in a suite that already reruns on any registry change; one new suite input, the list, which changes only with the registries. Shared files: the list, edited by the five landings of the last two rows in the days measured; entries merge by the driver, two `held` lines added at once conflict. Always-read words: none.
**Tests.** 4, blocks rather than labels: a label would not hold the count. 6, a shared file: only a change that leaves or deletes a superseded version touches it. 8: a `held` line is the default that proceeds, and it is a statement nothing checks (the index of record holds Programs only by digest).
**Owner and exit.** circuits; it retires when the list is empty, or when variants get names of their own and a version number means a revision only.

**What it touches of other owners.** `tools/verity/circuit_check/pyproject.toml`, three places: the list as the last path of the suite's `inputs` (lines 209 to 211), and the test's name in the two reads groups of the tests that load the registries (line 233, and lines 262 and 263). It adds no path to either group. `tests/README.md`: one paragraph after "Lint ratchets" (lines 36 to 42). The carry left these edits as they were, but for one path the moves rewrote inside the `inputs` line the first edit ends (`verity/protocols/pous/` to `verity/protocols/accounting/pous/`), and it changed no line of the test.

**Checked at this head** (`bc64a9ad6fb4`; each run's record names that commit).

- `r20261113-170500-8899`: the repository suite, `361 passed in 456 s`, 0 cached.
- `r20261113-170519-79b6` (48 GB): circuit-check's suite, `451 passed, 5 skipped in 2714 s on 5 workers`, 0 cached; it read nothing outside its inputs. At the old head it was 439 passed, 5 skipped in 640 s; the suite's own sources changed on `main` between the two, where the time went is not looked into, and this test's share is seconds.
- `r20261113-170407-2fdb`, in a clone: `tools/verity/check/pins.py --regen` at the root exits 0 and rewrites no file (6 pins `ok`, none stale, before it and after). The test file's graph task there: `pass`, 3 passed, `undeclared_reads: []`.
- The ratchet gate (`main` `c68135b49d63` to this head): `failures: []`; no retired-term or lane-text site is new in the files it changes.
- `python3 -m research.carry d9b1f18a28fe… bc64a9ad6fb4…`: exit 1, `kind: authored`, `why: bc64a9ad6fb4 changes tests/lint/allowlists/superseded_versions.json`, 70 commits on the first-parent line, no move rerun. So the old head's verdicts don't carry by themselves, and what a reader reads is that one commit: three lines deleted from the list.
- At the merge before the list was made again (`483586d432a2`, the carried list) the test is `1 failed, 2 passed`: the three entries for versions that are gone, nothing new, nothing moved.

**Against the queue, at the old head (`d9b1f18a28fe`, before the moves; not made again).** The test at that head was merged with every pull request the queue listed at 03:24Z on Fri 13 Nov (257 open, 58 of them marked), each of the six live candidates, top's staged move branch and `main`: one scratch worktree a head, the test file alone, 264 heads. At 04:17Z the heads that were new or had moved were run the same way, 101 more.

| heads, at 03:24Z | count | merge and pass | merge stops against `main`; pass with their own side of each conflict | would fail | merge stops, no verdict |
|---|---|---|---|---|---|
| marked, admitted | 41 | 41 | 0 | 0 | 0 |
| marked, refused or waiting | 17 | 1 | 16 | 0 | 0 |
| live candidates and `main` | 7 | 7 | 0 | 0 | 0 |
| top's staged move branch | 1 | 0 | 1 | 0 | 0 |
| not marked | 198 | 87 | 106 | 0 | 5 |

- No marked pull request would have failed it, and no candidate did: 0 of 58 at 03:24Z and 0 of 68 at 04:17Z, when all 58 admitted ones merged and passed.
- The 16 marked ones whose merge stopped did so against `main` alone on the same files; none of those files is one the test reads, so the verdict didn't depend on the side taken.
- Two controls merged the same way failed as they should: a Definition change that leaves `F32Sqrt_v1` superseded, and the list without `NvLogf_v2`'s entry.
- The retirement that #8415 held then (`CompileSlice_v2`, `CompileSlice_v3`, `CacheStep_v2`) has since landed as #9682, ahead of this, so the list made again has no line for them. #8415 is still open.
- The moves were rehearsed before they ran (`r20261113-032839-64dc`, the four maps of top's staged branch). What has run since rewrote 55 of the old list's 239 `file` strings: 24 by the kept map's `ops/` row and 31 by the protocol groups map (28 kept, 3 of the retired versions). The served map's `registry/` row, 183 lines, has not run.

Every pull request of that reading by number, lane and state is in `scratch/CIR-208/inflight.md` on `cursor/circuits-cir208-scratch-6f04`, the path strings with the map row that moves each in `scratch/CIR-208/paths.md`, and this carry's records (the list as carried, what was made again, the registries' ids that entered and left) under `scratch/CIR-208/carry/` there.

**Limits.** The list is exact against the registries of the tree it is in. So a pull request that adds or deletes a superseded version and is marked before this lands passes its own check and fails this test in the candidate that holds both. There the failure names this change's files, so by the queue's rule for suspects (the pull requests whose change touches what the failure names) this is the one ejected; once this is on `main`, the other is the one found. That is read from the rule, not run. The ratchet gate, which holds the list shrink-only, is off the merge gate and warns, as for every allowlist; at PR time the test holds the list to the registries, so growth is a visible added line in the diff. A module that moves takes its entries' `file` with it (`--prune`). A new Definition that is another kernel, port or input rather than a revision has a fourth way the message doesn't print, a name of its own; printing it waits on the design for variants' names.

Refs CIR-208
Goal circuits/slice-clean
