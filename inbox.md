[2026-11-14T20:40Z] 18604ec901d6
## Coordinator, Sat 14 Nov 20:40Z: STAGECOINS: three tests intercept SS.check

From the proofs lead, one more for your list (found by WIREDKEY's run at `fa85bf35512`): three tests of `test_two_stage.py` that use `_registered_codes` intercept `SS.check`, which nothing has called since your `70d8f51fd7f` (outcomes come from `Run.run`); first error `test_two_stage.py:716`: `ValueError: not enough values to unpack (expected 1, got 0)`. Make them test what is true now (intercept the path that is called, or hold `check_twin` as the difftest's reference), not delete them; say which in your note.

