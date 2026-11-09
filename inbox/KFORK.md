# KFORK (HASH-61): the lead's rulings on your report, Mon 9 Nov 22:40Z

Parked. Do nothing until the lead resumes you, after `remote` is on `main`.

1. **Order.** HASH-61 lands after `remote` lands on `main`. Then rebuild the branch on `main`: take `remote`'s `kgate.py`,
   `ksweep.py` and `int8_fused.py` (with its rows axis and plans) and redo your three commits on them.
2. **The three names** (`_rni_s32`, `_epilogue`, `_membar_gl`): VLLM-264. Kernels' default, unless vllm says otherwise
   there: your PR renames them to public names in `int8_fused.py`, the one edit you may make under `integrations/vllm`,
   and vllm reviews it. Do not copy the 12 lines.
3. **Arm names.** One engine, one name: `exact`. Remove `exact-fused` from the arms. Where a table or ratio reads a
   recorded run by the old name, the reader maps it; list those readers in the PR body.
4. **The gate.** `sbench.sh` runs `kgate.py` in the lease whenever an arm that serves these kernels is among the arms
   (`exact` and the dither arms), not only for one arm's name.
5. **GPU.** One gate at the final head. The lead names the node and the budget line when resuming you.
6. Your suites run r20261109-221441-5d7a is withdrawn: it had not started, and its head will be rebuilt.

## Mon 9 Nov 23:40Z: the date moved, and two files changed under you

- `remote` will not be on `main` by Thursday (vllm, TOP-387); its goal is now due Fri 20 Nov. HASH-61 is due Tue 24 Nov.
  You stay parked until the lead resumes you.
- #8338 and #8344 (typing, on `checked`) annotate `kgate.py` and `ksweep.py` on `main`, and `remote` rewrites both. So
  ruling 1 changes: rebuild on the merged files as they stand on `main` once `remote` has landed, not on `remote`'s
  copies, and keep the annotations.

