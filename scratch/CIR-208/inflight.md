# CIR-208: the superseded-versions check against the queue, Fri 13 Nov

The base is #8463's head `d9b1f18a28fef36e24a3c3d6b6f6031236c943a6`, which holds `main` `1e4d5c993c08` (and so #9375, on `main` since 03:01Z). The queue is
`research queue status` as read at 03:24Z: 257 open pull requests, 58 of them
marked (41 admitted, 11 refused, 6 waiting on another). Each pull request's
head is `refs/pull/N/head` as fetched right after; 4 had moved between the two reads (#9403, #9533, #9583, #9593), and the fetched head is the one run. A live candidate is a
`refs/heads/queue/*` ref that holds `main`'s tip. #8463 itself is left out.

## Result

**0 of the 58 marked pull requests would fail the check, and no live candidate does.** Of the 198 not marked, 0 would.

| heads | count | merged with the base, the test passes | the merge stops; with the head's side of each conflict the test passes | would fail | the merge stops, no verdict |
|---|---|---|---|---|---|
| marked, admitted | 41 | 41 | 0 | 0 | 0 |
| marked, refused | 11 | 0 | 11 | 0 | 0 |
| marked, waiting | 6 | 1 | 5 | 0 | 0 |
| live candidates | 6 | 6 | 0 | 0 | 0 |
| top's staged move branch | 1 | 0 | 1 | 0 | 0 |
| `main` | 1 | 1 | 0 | 0 | 0 |
| not marked | 198 | 87 | 106 | 0 | 5 |
| all | 264 | 136 | 123 | 0 | 5 |

So the table the brief asks for, one row for each pull request that would fail with the versions it leaves superseded
and its way through, has no row.

## How a head was run, and the controls

One scratch worktree a head (`check.py`): the base, `git merge` of the head, then `pytest` on the test file alone with
every workspace package taken from that worktree, then the test's own functions (`registered`, `superseded`, `listed`,
`findings`) for what the merged tree registers and what the list lacks. "Passes" is `3 passed` and no superseded
version without an entry, no entry for a version that is gone, none filed under another file and none twice.

- Control, merged the same way: a Definition change (`F32Sqrt_v2` registered without the word it computes, so it supersedes `F32Sqrt_v1`): `1 failed, 2 passed in 4.52s`, naming `F32Sqrt_v1`.
- Control, merged the same way: the list with `NvLogf_v2`'s entry taken out: `1 failed, 2 passed in 4.57s`, naming `NvLogf_v2`.
- `main` itself merged with the base: `3 passed in 4.54s`, 239 superseded, 239 entries, 2249 registered.

The test was not idle on these heads: 7 of them register other ids than the base does (1 marked, 6 not marked, 0 of the candidates and branches), and none leaves a version superseded:

| head | lane | state | ids added | ids removed | versioned above `_v1` among the added |
|---|---|---|---|---|---|
| #9075 | proofs | marked, admitted | 3 | 2 | `GreedyChunk_v2`, `GreedyCombine_v2` |
| #9128 | proofs | not marked | 3 | 2 | `GreedyChunk_v2`, `GreedyCombine_v2` |
| #9254 | circuits | not marked | 4 | 0 | `SparseMlaHeadKeysWire_v2`, `SparseMlaHeadScoresCols_v2` |
| #9317 | circuits | not marked | 1 | 0 | none |
| #9417 | training | not marked | 14 | 0 | `Fa2TrainBwdDkDv_v2`, `Fa2TrainBwdDq_v2`, `Fa2TrainBwdHead_v2`, `Fa2TrainBwdKeyFirst_v2`, `Fa2TrainBwdKey_v2`, `Fa2TrainBwdPre_v2` and 1 more |
| #9480 | proofs | not marked | 3 | 2 | `GreedyChunk_v2`, `GreedyCombine_v2` |
| #9504 | vllm | not marked | 3 | 2 | `GreedyChunk_v2`, `GreedyCombine_v2` |

Definitions a head's changed modules register only when something imports them, which the registries' loader does
not (so the check, like circuit-check, does not see them):

- #9364 (vllm): 3, `AllReduceSumBf16Elem_v1`, `AllReduceSumBf16_v1`, `Input32_v1`

## Marked pull requests whose merge with `main` stops

16 of the 58. None stops in a file of this change: each stops against `main` alone on the same files
(`git merge-tree` of `main` and the head; `sweep/conflicts-main-alone.json`), which is the queue's own reason for most of
the refused. For these the test was run on a scratch resolution, the pull request's side of every
conflict, so its verdict is of that tree and not of the merge its owner will make. "Read by the test" counts the
conflicted files among the 772 tracked files the test opens at the base: where it is 0, no file the registries
load is in conflict, and the verdict doesn't depend on the side taken.

| pull request | lane | state | files in conflict | read by the test | with the pull request's side |
|---|---|---|---|---|---|
| #8436 | ci | refused | 2: `tools/research/src/research/vqueue_loop.py`, `tools/research/tests/test_vqueue_loop.py` | 0 | `3 passed in 4.46s`, 239 superseded, 239 entries |
| #8465 | ci | waiting | 3: `tools/research/src/research/vqueue.py`, `tools/research/src/research/vqueue_loop.py`, `tools/research/tests/test_vqueue_loop.py` | 0 | `3 passed in 4.46s`, 239 superseded, 239 entries |
| #8756 | ci | refused | 12: `tools/research/src/research/drive.py`, `tools/research/src/research/vqueue_loop.py`, `tools/research/tests/test_drive_executor.py` and 9 more | 0 | `3 passed in 4.59s`, 239 superseded, 239 entries |
| #8772 | no lane | waiting | 1: `infra/control/post_start.sh` | 0 | `3 passed in 4.43s`, 239 superseded, 239 entries |
| #9080 | proofs | waiting | 4: `benchmarks/one_stage/tests/test_two_stage_second.py`, `benchmarks/one_stage/two_stage_small.py`, `benchmarks/one_stage/two_stage_small.sh` and 1 more | 0 | `3 passed in 4.50s`, 239 superseded, 239 entries |
| #9176 | proofs | refused | 4: `benchmarks/one_stage/tests/test_two_stage_second.py`, `benchmarks/one_stage/two_stage_small.py`, `benchmarks/one_stage/two_stage_small.sh` and 1 more | 0 | `3 passed in 4.52s`, 239 superseded, 239 entries |
| #9212 | proofs | waiting | 4: `benchmarks/one_stage/tests/test_two_stage_second.py`, `benchmarks/one_stage/two_stage_small.py`, `benchmarks/one_stage/two_stage_small.sh` and 1 more | 0 | `3 passed in 4.82s`, 239 superseded, 239 entries |
| #9253 | proofs | refused | 1: `verity/protocols/flock/verifier/FlockVerify.lean` | 0 | `3 passed in 4.67s`, 239 superseded, 239 entries |
| #9363 | lean | refused | 15: `benchmarks/harness/specs/llama32-1b-1gpu-pouw-protocol-served-h200.toml`, `benchmarks/harness/specs/prefill.toml`, `benchmarks/one_stage/composite.py` and 12 more | 0 | `3 passed in 4.60s`, 239 superseded, 239 entries |
| #9396 | ci | refused | 1: `tools/research/src/research/vqueue_loop.py` | 0 | `3 passed in 4.45s`, 239 superseded, 239 entries |
| #9415 | ci | refused | 2: `tools/research/src/research/vqueue.py`, `tools/research/src/research/vqueue_loop.py` | 0 | `3 passed in 4.72s`, 239 superseded, 239 entries |
| #9419 | ci | waiting | 1: `tools/research/src/research/vqueue_loop.py` | 0 | `3 passed in 4.64s`, 239 superseded, 239 entries |
| #9462 | ci | refused | 1: `tools/research/src/research/vqueue_loop.py` | 0 | `3 passed in 4.84s`, 239 superseded, 239 entries |
| #9479 | proofs | refused | 1: `benchmarks/spot_figures/pyproject.toml` | 0 | `3 passed in 4.60s`, 239 superseded, 239 entries |
| #9495 | ci | refused | 1: `tools/research/tests/test_vqueue_loop.py` | 0 | `3 passed in 4.69s`, 239 superseded, 239 entries |
| #9556 | ci | refused | 3: `.agents/skills/team/SKILL.md`, `tools/move/README.md`, `tools/research/src/research/queue.py` | 0 | `3 passed in 4.52s`, 239 superseded, 239 entries |

By lane: ci 9, proofs 5, no lane 1, lean 1.

## The pull requests not marked (a forecast: nobody asked the queue to land these)

| lane | heads | pass | merge stops, passes with its side | would fail | merge stops, no verdict |
|---|---|---|---|---|---|
| proofs | 50 | 18 | 31 | 0 | 1 |
| network-accounting | 21 | 6 | 14 | 0 | 1 |
| vllm | 19 | 6 | 12 | 0 | 1 |
| compute-accounting | 19 | 17 | 0 | 0 | 2 |
| lean | 15 | 7 | 8 | 0 | 0 |
| ci | 13 | 3 | 10 | 0 | 0 |
| training | 11 | 7 | 4 | 0 | 0 |
| top | 10 | 2 | 8 | 0 | 0 |
| circuits | 8 | 2 | 6 | 0 | 0 |
| memory-accounting | 7 | 3 | 4 | 0 | 0 |
| compute-research | 6 | 3 | 3 | 0 | 0 |
| kernels | 6 | 6 | 0 | 0 | 0 |
| soundness | 6 | 4 | 2 | 0 | 0 |
| bc-629a532d-3c1f-447e-b1 | 2 | 0 | 2 | 0 | 0 |
| infra | 2 | 1 | 1 | 0 | 0 |
| MOESERVE-C | 1 | 0 | 1 | 0 | 0 |
| bc-b43da88b-5d25-4463-9b | 1 | 1 | 0 | 0 | 0 |
| benchmarks | 1 | 1 | 0 | 0 | 0 |

111 of them stop against `main`; 20 of those stop in a file the test reads (#7378, #7850, #8139, #8414, #8415, #8460, #8503, #8578, #8580, #8731, #8763, #8869, #9031, #9312, #9333, #9369, #9378, #9457, #9491, #9541), so for them the scratch resolution's verdict is weak: the owner's merge decides what the registries hold.

No verdict, since the head's side of its conflicts gives no tree whose registries load (each is an older tree than
the layout's move, and its side of a conflicted module imports a package that is gone):

- #8415 (network-accounting, not marked): 32 files in conflict; `ModuleNotFoundError: No module named 'verity.ml'`
- #8460 (compute-accounting, not marked): 913 files in conflict; `ImportError: cannot import name 'boolean' from 'verity.ml.circuits' (unknown location)`
- #8503 (compute-accounting, not marked): 913 files in conflict; `ImportError: cannot import name 'boolean' from 'verity.ml.circuits' (unknown location)`
- #8580 (vllm, not marked): 4 files in conflict; `ModuleNotFoundError: No module named 'verity.ml'`
- #8731 (proofs, not marked): 5 files in conflict; `ModuleNotFoundError: No module named 'verity.ml'`

## A second read, at 04:17Z

By 04:17Z `main` was `00479a6c442a` (one of the candidates run above), 288 pull requests besides #8463 were open on
GitHub and 68 were marked. The sweep had already run 192 of them at the very head they still had;
the other 96 (opened or pushed since, or open with no row in the queue's table) were run the same way, at the same base,
with every candidate that then held `main`'s tip. This table is of every pull request open at that read, each at the
head it then had:

| heads | count | of them run in this pass | merged with the base, the test passes | the merge stops; passes with its side | would fail | the merge stops, no verdict |
|---|---|---|---|---|---|---|
| marked, admitted | 58 | 14 | 58 | 0 | 0 | 0 |
| marked, refused | 7 | 0 | 0 | 7 | 0 | 0 |
| marked, waiting | 3 | 0 | 0 | 3 | 0 | 0 |
| not marked | 205 | 67 | 109 | 91 | 0 | 5 |
| open, with no row in the queue's table | 15 | 15 | 2 | 4 | 0 | 9 |
| candidates holding `main`'s tip, and `main` | 5 | 5 | 5 | 0 | 0 | 0 |

**Still none would fail**: 0 of the 68 marked, 0 of the rest.

The 10 marked ones whose merge stops:

- #8436 (ci, refused): 2 files, 0 read by the test; as in the table above; pass with its side
- #8465 (ci, refused): 3 files, 0 read by the test; as in the table above; pass with its side
- #8756 (ci, refused): 12 files, 0 read by the test; as in the table above; pass with its side
- #8772 (-, waiting): 1 files, 0 read by the test; as in the table above; pass with its side
- #9080 (proofs, waiting): 4 files, 0 read by the test; as in the table above; pass with its side
- #9176 (proofs, refused): 4 files, 0 read by the test; as in the table above; pass with its side
- #9212 (proofs, waiting): 4 files, 0 read by the test; as in the table above; pass with its side
- #9363 (lean, refused): 15 files, 0 read by the test; as in the table above; pass with its side
- #9462 (ci, refused): 1 files, 0 read by the test; as in the table above; pass with its side
- #9556 (ci, refused): 3 files, 0 read by the test; as in the table above; pass with its side

Heads of this pass that register other ids than the base, none leaving a version superseded:

- #9417 (training, not marked): 14 added, 0 removed; above `_v1`: `Fa2TrainBwdDkDv_v2`, `Fa2TrainBwdDq_v2`, `Fa2TrainBwdHead_v2`, `Fa2TrainBwdKeyFirst_v2`, `Fa2TrainBwdKey_v2`, `Fa2TrainBwdPre_v2` and 1 more
- #9514 (None, open, with no row in the queue's table): 14 added, 0 removed; above `_v1`: `Fa2TrainBwdDkDv_v2`, `Fa2TrainBwdDq_v2`, `Fa2TrainBwdHead_v2`, `Fa2TrainBwdKeyFirst_v2`, `Fa2TrainBwdKey_v2`, `Fa2TrainBwdPre_v2` and 1 more
- #9605 (compute-accounting, not marked): 0 added, 27 removed

No verdict (the head's side of its conflicts gives no tree whose registries load): #8197, #8415, #8460, #8503, #8580, #8584, #8656, #8705, #8731, #9018, #9030, #9207, #9517, #9610.

## lean's #9375, circuits' #9340 and top's staged move branch

- **#9375** (`34c5cbb3705c`) is on `main` since 03:01Z, and the base holds it: the test on the base is `3 passed`, with the
  renamed `DefinitionTable` and `BodyNode` in the tree. The test names neither class; it reads the table through
  `verity.circuits.defs.REGISTRY`, which kept its name. Before it landed, the earlier head `b1409cffb0` merged with
  `34c5cbb3705c` also gave `3 passed`.
- **#9340** (`3aefb20f1941`, marked): merges with the base, `3 passed`, the same 2,249 ids as the base.
- **top's staged move branch** (`cursor/top-moves-c-b396` at `680f1541f8c3`): its merge stops in one file,
  `tools/verity/check/public_boundary.toml`, against `main` alone too (`main` edited the list the branch edits). With the
  branch's side of it the test is `3 passed`. The branch holds the maps and not the move, so the move itself was run:
  `r20261113-032839-64dc` ran the four maps on this merge. After the first three and again after the fourth, every one of
  the list's 239 `file` strings names a file, the test is `3 passed`, `--prune` changes only the order of 25 lines, and the
  test file's graph task passes with no undeclared read (`paths.md`).

## One pull request that deletes listed versions

**#8415** (network-accounting, not marked, last pushed Tue 10 Nov on a tree older than the layout's move) retires
`CompileSlice_v2`, `CompileSlice_v3` and `CacheStep_v2`. All three are in the list. Its merge with `main` stops in 32
files, so it got no verdict here. When its owner brings it up to `main` after #8463 has landed, the test fails until
`--prune` takes the three entries out: the first way through, used as meant.

## What moved while this was run

Between the first read of the queue (02:52Z, 49 marked) and the one this document is made on (03:24Z, 58 marked), ten
marked pull requests landed, twenty more were marked and one lost its mark; by the second read above (04:17Z) 68 were
marked. An earlier sweep at the head `b1409cffb0` (`main` `8c11a1d5ab`) over the first read's 261 heads found the same:
no head fails, 107 merges stop. A pull request opened, pushed or marked after 04:17Z is not in this document.

## Run, and inferred

- **Run.** Every count in the tables above: 365 merges at this base (264 on the 03:24Z read, 101 more on the 04:17Z
  read), the test file on each tree that merged, the two controls, and `git merge-tree` of `main` with each head whose
  merge stopped on the 03:24Z read (the two with 913 files in conflict left out).
- **Inferred.** That a pull request whose merge stops will pass once its owner merges `main`: the verdict is of the pull
  request's side of each conflict. For the marked ones no conflicted file is one the test reads, so the inference is
  short. For 20 of the 111 not marked on the 03:24Z read it is not.
- **Not run.** The queue's own candidate with #8463 in it; `tools/move/restack.py` on the branch.
