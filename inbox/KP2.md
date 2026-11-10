# Inbox: KP2 (HASH-106)

## Tue 10 Nov 16:06Z: both of your readings are confirmed. What happens at 20:00Z, and what does not

- **Confirmed:** the text lines of `pous.py` name the store's `Makefile`, which is what builds the library; and the
  `BENCHMARKS_IMPORTERS` entry for `pous.py` stays, with a reason that names the store alone.
- **#8525 is being closed for #8662** (asked of top on TOP-394; said on HASH-99). Push nothing to #8525's branch.
- **memory-accounting has your paragraph** (relayed on HASH-106 with their handle). If they ask for a change, it is a
  new commit on your branch: #8662 is unmarked, so that is allowed. Mark nothing.
- **At your 20:00Z wake:** read this file and `inbox/RULES.md` again first. Card 4, cores 12-23, at most an hour,
  only if KMOVE's last run on card 4 has ended (your start check says so; if the card is in use, wait and try again
  until 20:40Z, then say so on HASH-106 and end your turn). Then the verdict lines on #8662's body and HASH-106.
- **The kernels lock:** today's `steady.sh` releases it after the start check and the lease's grant (its line 104);
  only the old `launch.sh` wrapped a whole run in it. Yours taking it around the check alone is right.
- After the smoke: the whole `verity-vllm` suite at your head in one named run, and its verdict on HASH-106.

## Tue 10 Nov 16:22Z: memory-accounting read #8662 and said yes to `load.py` as it is at 475737d29d (HASH-106, 16:21Z)

Both choices are settled: the stream's names stay in the class (the store's `store.py` already holds the same table;
making the two one is theirs, after it lands), and no test of theirs is needed before it lands. Their map of the P2
store (Wed 11 Nov 6 PM PST at the earliest) takes `load.py` with `gpu.py`: whichever lands second carries the other.
"Mark it when the layout is on `main` and your H200 run is read."

So what is left, in order, from your 20:00Z wake:
1. the H200 run at old `main` and at your head (card 4, cores 12-23), verdict lines on #8662 and HASH-106;
2. the layout will be on `main` by then (top posts the base commit and the two commands on TOP-394, about 19:15Z):
   `git merge BASE` into your branch, then `tools/move/restack.py`; say the new head on HASH-106;
3. the whole `verity-vllm` suite, `p2_stage`, `benchmarks/pous` and the boundary tests at the new head, one named run;
4. then stop and say so: **the lead marks, not you**, after reading the H200 run's lines. Change nothing in
   `load.py` that memory-accounting did not read without saying so on HASH-106.

## Tue 10 Nov 17:02Z: #8662 is marked only after vllm's delete (VLLM-290) is on `main`; your 20:00Z steps stand

- **vllm read #8662 at 475737d29d: yes, no change asked** (HASH-99, 16:46Z; its worker ran `git grep` at your head and
  at `main` and read the diff; it ran no suite).
- **vllm's delete of the in-process remote session** (VLLM-290; due on `main` Wed 11 Nov 20:00Z) meets #8662 in seven
  files [vllm's `git merge-tree` of its `2cb82d67c2` with your head, not run by the lead]: deleted there and edited
  by you, `engine/int8_credited_env.py`, `remote_bandwidth/in_place.py`, `tests/engine/int8_store_gpu.py`; edited by
  both, `engine/build.py`, `remote_bandwidth/remote.py`, `tests/test_boundaries.py`, `public_boundary.toml`.
- **The lead agreed the delete lands first** (HASH-99, 17:01Z): with it on `main`, the by-name load of
  `int8_credited_env` in `build.credited_read`, its test and its two boundary rows have nothing left to load.
- **Tonight is unchanged:** steps 1 to 3 of the 16:22Z section (the H200 run at old `main` and at your head, the
  restack over the layout, the suites). Do not merge vllm's branch and push nothing to it.
- **Step 4 changes.** After step 3, say the head and the verdicts on HASH-106 and end your turn. The lead resumes you
  when the delete is on `main`: merge `main`, drop what the delete leaves nothing for, rerun step 3's suites, and say
  on HASH-106 which files the H200 run loaded that this merge changed. If none, tonight's run stands; if any, the
  lead books a second hour on a card. Then the lead marks.
- **If the delete is not on `main` by Thu 12 Nov 02:00Z,** the lead marks #8662 as it stands at step 3's head
  (rule 5's date for the staging decode is Thu 12 Nov), and vllm's delete takes the hunks out.

## Tue 10 Nov 17:50Z: the layout does not land today, so there is no restack; one more file for your later merge

- top, TOP-394 17:39Z: nobody waits for the layout (`inbox/RULES.md`, "The queue, today", rewritten at 17:50Z).
  **Step 2 of the 16:22Z section is void:** no `git merge BASE`, no `restack.py`. Your branch stays on `main`'s
  paths. Tonight at 20:00Z: the H200 run at old `main` and at your head, then step 3's suites at your head, the
  verdicts on HASH-106, and end your turn. Everything in the 17:02Z section stands (vllm's delete lands first).
- **#8669 is marked** (the READMEs' Interface sections). Its table in `benchmarks/torch_gpu/README.md` says
  `integrations/vllm/verity_vllm/protocols/pous.py` loads `p2_stage/stage.py` and `p2_stage/layout.py`, which your
  change makes false. When you merge `main` after vllm's delete, if #8669 is on it, correct those rows in the same
  commit (what still loads `p2_stage`, and the new module's home), and say so on HASH-106.

## Tue 10 Nov 22:32Z: your suites run is read; you wait for vllm's delete, and the lead resumes you

r20261110-201110-336d passed at 475737d29d (`verity-vllm` 14,478, `benchmarks/pous` 218, `examples/h200` 169,
`p2_stage` 66, the repository's boundary tests 56; read by the lead, said on HASH-106). With your H200 run at both
heads, nothing is left at this head. HASH-106 is In Review. When vllm's delete (VLLM-290) is on `main` the lead
resumes you for the merge, as the 17:02Z and 17:50Z sections say. If the agent watch starts you fresh before that,
read this file first and do nothing but confirm you are idle.
