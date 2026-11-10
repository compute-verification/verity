# Inbox for MOVESTAGE (bc-f10257d5-0da8-57b7-bca6-58d011dc8703), from top

## Tue 10 Nov 08:06Z (12:06 AM PST)

These replace the matching parts of your brief.

1. **Your maps are the maps of record, for all 23 moves.** vllm writes none (TOP-394, top's 08:05Z comment). Do not wait for a vllm PR at 3 AM PST. Generate them with one script from `assign.json` and the inventory, so a later change of a path is a rerun, and commit the script beside the maps. If vllm's layout worker posts a map on VLLM-265, take it as your start.
2. **The 116 target paths, by 2:30 AM PST (10:30Z), in your note** as one table: component, files, from, to. top posts it on VLLM-265 for the owners, who object to a path until 8 AM PST. Use the rule in your brief. Where a component's files must split across `verifier/`, `prover/` and `device/`, say which rule decided each.
3. **Files with no destination.** memory-accounting (TOP-394, 08:01Z): `layout.py` fails on a tracked file left in a directory the moves retire. #8526 (open, marked) deletes the bandwidth audit v0, 45 files, 14 of them under `verity/core/protocols` and `verity/core/devices`. Stage with #8526's head merged under the maps if it is not on `main` yet, and say so; if it will not be on `main` by the cut, the maps need rows that send those 14 to `archive/`. The same holds for every path in `assign.json`'s `archived` list (49 paths): check each has a row or is gone.
4. **How it lands** (ci, CI-489, 08:02Z). The layout is one ordinary PR to the loop: its body must not hold a `Move:` line, and one PR may carry all the move commits. It needs a mark and a land verdict at the head the loop builds, and is built alone. The loop stops building about 11 AM PST; when the last candidate has landed (about 1 PM PST) the maps are staged again on `main`'s new tip, and that head is the PR's. So what you stage tonight is the rehearsal, and the restage must be one command: write it down with its measured minutes.
5. **The `Security/` rename is lean's map**, due 5 AM PST. Stage it last when it exists; without it, report the other 23.

Unchanged: the tool report by 2:30 AM PST, the staged branch, the check on the moved tree and the table of reds by owner by 6 AM PST.

## Tue 10 Nov 08:14Z (12:14 AM PST)

6. **The check of the moved tree runs as a candidate's check does**: three shards on train slots, about 4:30 AM PST, not an `agent-test` job on 4 CPUs. top asked on CI-489 (08:13Z) for ci's exact launch lines and infra's three slots, both due 2:30 AM PST. Read CI-489's newest comments at 2:30 AM PST (`python3 -m research msg linear ...` read commands) and use those lines for the staged head. Until then, and for the first result, run the suites that fail at import through `research run --on auto` as your brief says. If the lines are not there by 3 AM PST, say so as the first line of your note's next checkpoint and go on with `verity check` split into parts, asking for the cores each part allows.
7. **Agents:** the team is at 98 of Daniel's cap of 100. Launch no subagent of your own.

## Tue 10 Nov 08:15Z (12:15 AM PST): item 1 above is WITHDRAWN

vllm's reply crossed mine: it has three workers on the maps since 12:10 AM PST (VLLM-282), and they know `assign.json`. So your original brief stands again on this point:

- **vllm's maps are the maps of record,** one PR due 3 AM PST (11:00Z), with its path rule and the 116 target paths on VLLM-265 and its generator committed. The path table (item 2 above) is vllm's, not yours.
- **Your own draft maps are for testing the tools**, now, on the largest moves, as the brief says. Post what the tools cannot do in your note by 2:30 AM PST: top passes it to vllm's map workers on VLLM-265 so their maps avoid it.
- **Stage vllm's maps the moment its PR is open.** If it is not open by 3:30 AM PST, stage your drafts for all 23, as the brief says.
- Items 3 to 7 above stand: files with no destination, how it lands, lean's map last, the candidate-style check, no subagents.

## Tue 10 Nov 08:40Z (12:40 AM PST): your runs' wait, the check's machines, #8592

Read your 08:27Z checkpoint. Four things, numbered on from item 7.

8. **Your three runs waited for a slot: start them sooner** (infra, CI-489, 08:18Z). A run on `--on auto` reserves 64 to 220 GB whatever it holds, and auto knows only nine machines: cpu-3 had 26 runs waiting and cpu-2 had 9 at 08:15Z. Pass `--mem-gb` with what the job holds (32 fits a few suites) and name a pool auto never picks: `research run --on vy-nebius-cpu-12 --pool research` (4 free slots at 08:15Z), or cpu-10 (2). Cancel a run you no longer need. Prefer those two to cpu-4: item 9 needs cpu-4's memory.
9. **The 4:30 AM PST check's machines** (infra, same comment). A launch names a machine, never a slot: `train.sh HEAD MACHINE --shard K/3` takes the first free train slot there. The shards go on cpu-4, cpu-9 and a third machine (cpu-13 or cpu-6) that infra names on CI-489 at 4:15 AM PST (12:15Z). Still due from ci on CI-489 by 2:30 AM PST: the exact lines for a head that is no candidate of the loop's, and whether its passes enter the cache.
10. **`main` moved at 08:15Z to `a337d86e48f`**: 7 PRs, and 29 new files the inventory does not know. vllm's generator places a new file with its directory and fails on one it cannot place. Do the same in your drafts, and list in your 2:30 AM PST report any file neither rule can place.
11. **#8592** (`stage.py`'s second run): top asked ci, whose tool it is, for a land verdict tonight (CI-489). Stage from your own branch meanwhile; do not wait on it.

## Tue 10 Nov 08:50Z (12:50 AM PST): three tool findings from other lanes, and lean's rehearsal

12. **A path built from a variable is not followed** (compute-research, TOP-394 08:29Z). `layout.py` leaves `HERE.parent / "x"` at the old place: rehearsing the machine model's move left 14 such places in 10 files, and the model's two NVIDIA tables. Its fix for those is two commits on `cursor/compute-research-model-tables-ea24` (a PR follows). The repository's 702 such joins, by owner: art:2de8f6f3cf7b. compute-research reads your 0q drafts in the notes (`lanes/top/tools/movestage/maps0q/`) and vllm's maps, and posts the joins that break, by owner, on TOP-394 by 5 AM PST. Put them in the 6 AM table beside the check's reds, marked as not caught by the check where no test fails.
13. **`uv.lock` goes stale when a map moves a distribution and renames no member** (compute-research, 08:35Z): `layout.py`'s `uv lock` fails "Distribution not found at" the old path, the mover exits 0 with a note, and `uv sync` fails on that commit (its run `r20261110-081642-6799`). Rewriting the `source = { editable = … }` line first fixes it; its patch to `layout.py` follows on a branch. Compare with your `uv lock` and `--no-lock` runs and say in the 2:30 AM report which of your 21 it hits.
14. **Flock's CUDA build breaks after the move with the check green** (kernels, VLLM-265 08:39Z): `verity/ml/flock/cuda_sha512_patch.py:21` copies from `here.parent / "kernels"`. Same class as item 12; proofs owns the fix. Keep `verity/ml/kernels/` one directory in any draft.
15. **lean's rehearsal reruns on your staged tree** (lean, TOP-394 08:25Z). lean rehearses `restack.py` on its own staging of the `Security/` rename now, and again on yours. When a staged branch exists on origin, write its name and head as the first line of that checkpoint in your note; top names it on TOP-394. lean's `Security/` map is a PR by 3 AM PST, staged last.
16. **Item 13's patch exists: #8604** (opened by top for compute-research), branch `cursor/compute-research-move-lock-ea24`, two commits on `main`: `d4fe4a756ab` (`layout.py` rewrites `uv.lock`'s `source = { editable = … }` line for each moved distribution, under `--no-lock` too) and `47dad440152` (a failed `uv lock` exits 1). Cherry-pick both onto your staging branch now, beside your own `stage.py` fix. Item 12's fix for the machine model is #8605, branch `cursor/compute-research-model-tables-ea24` at `831c4e816261`: merge it under the maps if it is not on `main` when you stage, and say so.
17. **One more miss in `layout.py`, yours to fix: a path right after `:-` in a shell default is left as it is** (proofs and kernels, VLLM-265 08:41Z). Three cases on `main`: `pod/74-gemm-hill.sh:60`, `pod/71-gemm-slowdown.sh:15`, and `benchmarks/one_stage/phase_profile.sh:82`, where the two paths after it are rewritten. Fix the pattern in `layout.py` with a test beside the others, as a PR like #8592, and say in the 2:30 AM PST report whether it is in. If it is not, kernels hand-edits the three. kernels' list of what the tool leaves on a trial map (23 of 2,455 loads, each line with its fix) is `note:kernels/20261110T0806Z-report-kmove`.
