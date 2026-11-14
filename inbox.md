[2026-11-14T08:53Z] b3681fc747ae
NOT READY 2 bebb06fe85c3f8b9ae3cad88daea62f6aa1df956 — delta 0d6dbc68..bebb06fe of #9964 (head unchanged at 00:52 PST)

MUST FIX
1. synthetic.py:231 `kind_of` sees only the features it lists, so a new dial that branches `step()` passes the mapping test unseen (ruling 2). RAN:
   - Scratch: subset gains a `grad_noise` dial (`p.grad.add_(randn_like, alpha)` before clip), plus an attacks/ config setting it.
   - The test passes, and `kind_of` gives the adamw-decay kind.
   - The counted step differs: elementwise MACs 21,895,331 without it, 21,902,083 with it.
   - Cure: assert that every route's `DIALS` key is either read by `kind_of` or in a stated not-a-branch set (rates, schedules, `keep`, rank, depth, targets, `store`). A new dial then fails, by name, until it is classified.
   - Scratch deleted, route reverted.
2. A body sentence is false at this head: "Rates, schedules, `keep`, rank, depth, the targets and `store` do not". `kind_of`'s docstring says the same, but `kind_of` reads `len(o.param_groups)`, and an unfused optimizer keeps one group per rate. RAN: subset `{'trainable': '*'}` and the same with `gain_lr=1e-3` give different kinds (groups 1 vs 2); with `fused` they agree. Strict, not unsound. Cure: drop the group count, or state that it is part of the kind (ruling 3).

SHOULD FIX
- None.

NOTE
- Ruling 1 (RAN): the three QAT kinds train `blocks.1` over a frozen `blocks.0` and pass. The comparison is unchanged and exact.
- Ruling 2 (RAN):
  - Clean head: pass.
  - Known dials with a new value (lion, `micro_batches = 2`) fail by name, with the kind.
  - A dial no route has fails as `SystemExit: subset: route subset has no dial 'grad_noise'`, not with the test's message.
  - No allowlist, skip or default kind (READ).
- Ruling 3 (READ): `kind_of` adds clip, frozen above or every block, stack, groups and zeroed copies to the body's list; only the groups disagree. Unread existing dials don't branch: QAT `group` changes shapes only, and ema's `t == 0` branch is outside the measured step.
- Ruling 4 (RAN): `-m slow` collects 8 cases, not nine; qat-grad4-lowstate was counted twice. The six plus the three QAT kinds are marked. On this VM, fullmuon (5.49 s) and fullmuon-fe (5.43 s) run just over 5 s unmarked.
- Ruling 5 (READ, code path): `kind_of` → `mod.build` → `type(o).__name__`. QAT always builds Muon (qat.py:111) and no dial picks another, so no scratch config is possible. Grow's Muon is now checked, and grow-3 passes (RAN).
- RAN at head: trace plus mapping tests 32/32 (slow included); record test for the QAT kinds 3/3.
- `flops.py` breaks (RAN, reverted, worktrees clean). Each fails the same cases as before:
  - `sign` ×2: lion.
  - Fused AdamW +1: grow, muon-whole-fused, fullmuon, muon-fused, fullmuon-fe, depthshare, depthshare-top, strideshare, ema-fullmuon, ema-fullmuon-fe.
  - Linalg +1: shampoo, shampoo-decay, shampoo-whole.
  - All fail at test_run.py:703; the new `approx_macs` check (:711) catches nothing more under these breaks.
- The body and run still describe 0d6dbc68 (they follow the push).
- Events: one shell-hook broadcast from @training about naming models; I acted on none of it.

For the lead to read
- synthetic.py:231-258: `kind_of`.
- test_run.py:730-744: the mapping test.
- test_run.py:620-621: `SLOW_KINDS`.

