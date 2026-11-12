[2026-11-12T06:35Z] a9016c055f07
training to grade, 2026-11-12T06:36Z (from `date`). Not a stop: your three commits stand as you describe them. One thing the removal of `width` meets, which you need to take in before #9029 leaves draft.

#8910 (route `qat`, QAT's) is marked ready at 4e221ac275bc18fd53bb5bd147169986586e23d7 and lands before #9029. Read by me at that head:

- `routes/qat.py` line 107 has its own autocast, `torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda")`. Your 4.23.0 changes `routes/subset.py` alone, so qat still casts on CUDA alone.
- Its three entries (`s-t20-1.1x-qat-top4`, `-top3`, `-top2`) pass the port check today only with an `expected  width` line each (r20261112-060031-77ea: "3 entries: 3 equal, 0 differ, 0 refused.", 2 expected differences each, width and norm; 14 uses at 32 bits for top2, 21 for top3).
- Your head 01e8bd81627 and #8910's head merge with no textual conflict (`git merge-tree`, ran by me), so nothing stops the merge. After it, with `width` gone, the three qat entries differ on a CPU.

So, after your push:

1. Merge `origin/cursor/training-qat-route-80de` at 4e221ac275bc into your branch. It holds the carrier's head you already stand on, and its fourth commit also changes `capture/trace.py` (`_fake_copy` passes a class only the defaults it takes).
2. One commit, 4.23.0's own: qat's cast on every device, as subset's. Mend qat's tests as you mended the carrier's, where a float32 reference is held against the route.
3. Run the port check on the three qat entries. They must be equal with no `width` line. If they differ for any other reason, stop and tell me what the line says; do not add a rule.
4. Rerun the harness's suite and the grader's with the variable at the new head. #9029's body says it stands on #8910.

My review is still Thursday 6 PM PST, at that head.

