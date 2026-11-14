[2026-11-14T20:19Z] babecb9742c6
model: claude-opus-5-5-high; subscriptions removed: none held
STEP: all three leftovers are on `architecture` (tip `6086d592e83c`), and the comment for vllm is on TOP-424.
1. Imports: map `6e2b27e29b1`, move `e59d35d6ad7` (certified pure), imports `4e4c42e9354`. Both named files collect (38 tests), and the moved `test_train_step.py` passes 276.
2. `DEFAULT_FAMILIES`: `e41237ad180`. No identity changes.
3. Cut: `489372a90cf`.
No merge conflicted with the rename. Three of the five reach-backs are left, all in `derive.py` (`seed_rule`, `stop_rule`, `DEFAULT_RULES`), and none is small.

