[2026-11-14T15:54Z] 858bb88fc8a4
READY 2968ed6eb89d177139bd03189df84b89db2acbc7 — delta 5f3af349..2968ed6e of #9964, and its body (read at 07:58 PST)

MUST FIX
- None.

SHOULD FIX
- None.

NOTE
- Question 1, the must-fix is cured (RAN):
  - Every `DIALS`/`FIXED` key of ema (16), grow (11), lora (9), qat (11) and subset (14) is in exactly one set, and neither set names a dial no route has.
  - Mutations, each reverted: `embed_lr` dropped from `KIND_DIALS` fails, naming ema, grow and subset. `embed_lr` in both sets fails the disjointness assert. A new subset key `grad_noise` fails, naming subset and ema.
  - `embed_lr` moved back into `OTHER_DIALS` passes. As ruled before, a wrong classification is a statement a reader checks; this is no new finding.
- Question 2, both comments are true (READ):
  - `KIND_DIALS`: grow.py:76 freezes the embedding at `embed_lr` 0 and trains it above, and `kind_of` sees that through AdamW's trained parts and groups (RAN at 5f3af349).
  - `OTHER_DIALS`: `ema_horizon` branches only at `t == 0` (ema.py:60). The measured step is the middle of a run's last three, and the library's ema configs run 838 to 5383 steps.
  - The groups sentence is the one suggested, wrapped to 120. RAN: no line over 120 in synthetic.py:219-252; ruff passes.
- Question 3, the body is true at the head:
  - "64 configs … take 25 kinds"; "Two of those kinds had a test job: the LoRA config, and the two grow configs"; "61 configs take 23 kinds". RAN: 64 configs, 25 kinds, old jobs lora and grow covering 3 configs, all 23 `KINDS` used.
  - The `KIND_DIALS` list matches the code. The three failure cases of the mapping test match (READ).
  - 8 slow-marked cases (RAN at 5f3af349; the marks are unchanged).
  - The approx-MACs check and the optimizer read from what the route builds (READ, unchanged since bebb06fe).
  - No test file added; nothing deleted (READ).
  - RAN: run r20261114-154327-6007 has state `running`, rc none, tree_sha 2968ed6eb89d177139bd03189df84b89db2acbc7. `--changed 8e018c9331f` is the merge base. "It is running" is true now.
  - No machine name: the earlier "4-CPU VM" is gone (READ).
  - Loose but true: "the config's depth" means the start's block count (or grow's `depth`).
  - "The 8 new cases over about 5 s": on this VM, fullmuon and fullmuon-fe ran 5.4-5.5 s unmarked and depthshare 4.9 s marked (RAN at bebb06fe). The node's quick run decides, as ruled.
- Question 4 (READ): the delta is one file, synthetic.py. It holds three comment changes and `embed_lr` moved from `OTHER_DIALS` to `KIND_DIALS`; no code path or test changes.
- RAN at head: the mapping test and `[grow-3]` pass (2/2).
- No event addressed to @training reached me.

For the lead to read
- synthetic.py:219-227: the two sets and their comments.
- synthetic.py:247-250: the groups sentence.

[2026-11-14T16:06Z] c2b4c92537c9
training to every worker and reader, Sat 14 Nov 8:06 AM PST. No reply needed.

**The model rule changed (Daniel, TOP-423): only a lead runs Fable 5.2. You run `claude-opus-5-5-high` and stay on it.**

- Any subagent you launch names its model in the call: `claude-opus-5-5-high`, or `claude-sonnet-5-5-high` for a really well-specified task. Never Fable, never `inherit`, never a launch with no model.
- If a picker offers neither model, do not launch: do the work yourself or send it to me.
- A plan, a design or a judgment no check catches comes to me, in your next message, with what you recommend.

