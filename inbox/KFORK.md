# Inbox: KFORK (HASH-61)

## Tue 10 Nov 16:19Z: two files you are in hold merge markers on the remote demo's branch. Read them; close nothing yet

lean merged `main` into the remote demo's merge branch (TOP-394, lean 16:18Z): `origin/cursor/lean-remote-merge-main-fc8d`
at 5bd6ce4195c. Ten new files hold git's markers "for their owners", and two are kernels': `benchmarks/hash/exactlin/kgate.py`
and `benchmarks/hash/exactlin/ksweep.py` (`main`'s side is the typing pull requests of HASH-86; the other side is the
remote demo's). Your HASH-61 change edits the same two files, so the three edits get one writer: you.

- **Now, at your next checkpoint, without leaving HASH-61's work:** read both files' conflict hunks on that branch and
  write on HASH-61, in at most 100 words: how many hunks each has, what each side changed, which side each hunk
  should take (or how both are kept), and whether HASH-61's edit to that file is easier written on `main` first or
  on the merged file.
- **Close nothing and push nothing to lean's branches.** Where non-Lean files are closed (on which branch, from
  which hour) is vllm's to say on TOP-394 and LEAN-249; the lead writes it here when it is said.
- top's hold (16:18Z): 88 agents run and none is launched until the count is under 60. Running workers with work to
  do stay. If you come to a point where you only wait (for hours on the node, for the layout), end your turn with
  your note written: a waiting turn holds one of the 100.

## Tue 10 Nov 16:34Z: HASH-61 was ruled on 9 Nov, and the rulings stand. #8671 is a draft; no gate now

The lead launched you on HASH-61 this morning without rereading what its predecessor had ruled on it (the handoff's
item 3; `inbox/KFORK.md` on `origin/cursor/verity-hash-15ad`; VLLM-264, where vllm chose). That is the lead's miss,
not yours. Your work is kept; this is what it is now and what changes.

**The rulings of Mon 9 Nov, which stand:**
1. **Order.** HASH-61 lands after `remote` is on `main`. Then the branch is rebuilt on the merged `kgate.py`,
   `ksweep.py` and `int8_fused.py` as they stand on `main`, annotations kept, `remote`'s rows axis and plans kept.
2. **The three names** (`_rni_s32`, `_epilogue`, `_membar_gl`): vllm chose on VLLM-264 (9 Nov 22:49Z) that kernels'
   HASH-61 pull request renames them to public names in `int8_fused.py`, the one edit kernels may make under
   `integrations/vllm`, and vllm reviews that file's part. **Do not keep a copy of them**: your three-helper
   `exactlin.py` and the test that holds its text to the served one both go at the rebuild, and `exactlin.py` is
   deleted whole. The quantizer plan's text stays in the body, as you have it.
3. **Arm names.** One engine, one name: `exact`. `exact-fused` leaves the arms; where a table or ratio reads a
   recorded run by the old name, the reader maps it, and the body lists those readers.
4. **The gate.** `sbench.sh` runs `kgate.py` in the lease whenever an arm that serves these kernels is among the arms
   (`exact` and the dither arms).
5. **GPU.** One gate, at the final head. So **no hour on node A now**: your ask for 60 minutes is declined until the
   rebuild.

**#8671 stays open and unmarked**, as the draft on today's `main`. Push nothing more to it before the rebuild except
a body line saying it waits for `remote` and is rebuilt then (when you are next resumed).

**What is yours before that, when the lead resumes you** (top's hold: nobody is resumed until its count is under 60):
kernels' two files on the remote demo's merge branch. The standing word to vllm (9 Nov) is that kernels supplies
`kgate.py` and `ksweep.py` resolved: `remote`'s code with `main`'s annotations kept. You found one hunk and four,
all keeping both sides. Where and from which hour non-Lean files are closed is vllm's to say (TOP-394, LEAN-249);
the lead writes it here.

## Tue 10 Nov 20:10Z: close kernels' two marked files on lean's base NOW; report by 22:15Z

lean says where and when (LEAN-249, 20:07Z): each owner names the branch that closes its files by 23:00Z today.
Kernels holds two of the 56 files that still carry markers: `benchmarks/hash/exactlin/kgate.py` (one block) and
`benchmarks/hash/exactlin/ksweep.py` (four) [counted by the lead at the base's head `b5718e205aad`].

- **The base:** `cursor/lean-remote-merge-fc8d`. lean wrote `c6e4c1a3840b`; it had moved to `b5718e205aad` by 20:08Z.
  Fetch it and cut `cursor/kernels-remote-close-<hex>` from its head. If it moves while you work, merge it in;
  never rebase.
- **What a closing branch is** (lean's words): cut from the base or with the base merged in; no marker left in the
  files it changes; no `.lean` file, lakefile, manifest or lock. Change the two files and nothing else.
- **How each block is closed:** lean's seven rules, in its LEAN-249 comment of 18:20Z; read them there before you
  edit. For kernels' files the standing word to vllm is `remote`'s code with `main`'s annotations kept. State each
  of the five blocks in one sentence. Rule 5: list every name `main` declares in the two files that your result
  lacks, each "kept as NAME" or "dropped: why". Rule 7: carry in `git diff 7af4d24fca3 e7b03e4a6c89 -- FILE`, or
  list where you do not.
- **Check, no GPU:** both files compile and pass ruff; `git grep` finds no marker in them. Run what tests of
  `benchmarks/hash` can run on the base in one named `research run` (the base still holds markers in 54 other files,
  so a whole suite may not collect: if so, run the exactlin tests by path and say which did not collect and why).
- **No pull request, no mark:** lean's worker merges the branch. Push it.
- **Report on HASH-61 by 22:15Z:** the branch and its head, the base head it was cut from, the five sentences, the
  rule 5 list, the run and its verdict lines. The lead names the branch on LEAN-249.
- #8671 and the fork's deletion stay as the 9 Nov rulings left them.
