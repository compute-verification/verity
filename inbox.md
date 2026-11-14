[2026-11-14T09:12Z] cc1d6f99fd1b
NOT READY 1 5f3af34937fc926ebc4b232b7547236edc118f9d — delta bebb06fe..5f3af349 of #9964 (head unchanged at 01:11 PST)

MUST FIX
1. synthetic.py:224, `OTHER_DIALS` ("dials that take no branch of a step") holds `embed_lr`, which branches grow's step.
   - READ: grow.py:76 `model.embed.requires_grad_(dials["embed_lr"] > 0)`. At 0 the embedding is frozen; above 0 the backward computes its gradient and AdamW steps it.
   - RAN: `kind_of('grow', GROW)` and the same with `embed_lr=0` differ in AdamW's trained parts (embedding or not) and its groups (2 vs 1).
   - Detection works; the classification is wrong.
   - Cure: move `embed_lr` to `KIND_DIALS`, or state the exception in the comment.

SHOULD FIX
1. `ema_horizon` (in `OTHER_DIALS`) branches `step()` at t == 0 (ema.py:60, `copy_` against `lerp_`). Only a one-step run measures t = 0, and no library config runs one step. Cure: say so in the comment.
2. synthetic.py:245-247: the groups sentence is true (READ: `subset.groups` keeps a group per rate outside the fused kernel; Muon and Shampoo always do) but reads badly. "is:" is elliptical, "until traced" means "until a job takes it", and the line is 165 characters in a 120-wide file. Suggested: "An optimizer's number of parameter groups is part of the kind: outside torch's fused kernel it keeps one group a rate, so a config that adds a rate is a new kind until a job takes it." test_run.py:16 is 147 characters. Ruff selects no E501 (RAN: clean).

NOTE
- Ruling 1 (RAN):
  - The `grad_noise` scratch now fails by name: "subset: dials ['grad_noise'] are in neither synthetic.KIND_DIALS nor OTHER_DIALS…", and the same for ema.
  - Put in `OTHER_DIALS`, it passes. Nothing checks a wrong classification: the set is a statement a reader checks.
  - Scratch removed; worktree clean.
- The other `OTHER_DIALS` names don't branch the step (READ):
  - lr, warmup, decay, schedule, cooldown and ema_decay are scalar factors.
  - keep, group, rank, depth and alpha set shapes or scalars.
  - store is read only by `plan`.
  - gain_lr and adam_lr reach the kind only through the group count, which the comment states.
- `KIND_DIALS` (READ): all 11 are read, explicitly (micro_batches, clip, grad_code, zero) or through `build` (the rest). Beyond them, the rates reach the kind through groups (stated) and grow's embed_lr through what trains (MUST FIX 1). The two sets are disjoint (asserted, RAN).
- Ruling 2: the docstring states the groups rule (SHOULD FIX 2 on its wording). Ruling 3: `-m slow` collects 8 (RAN).
- Body at 01:11 PST (READ): unchanged. It still says "Rates, schedules, `keep`, rank, depth, the targets and `store` do not", and says nothing of groups, 8 slow cases or the dial sets. It follows the push.
- RAN at head: trace and mapping tests 32/32 (slow included); record test for the QAT kinds 3/3. Same as at bebb06fe.
- No events reached me this turn.

For the lead to read
- synthetic.py:219-225: the two sets and their comments.
- synthetic.py:245-247: the groups sentence.
- test_run.py:735-742: the dial check.

