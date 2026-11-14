[2026-11-14T16:07Z] 434118a3fe02
training to every worker and reader, Sat 14 Nov 8:06 AM PST. No reply needed.

**The model rule changed (Daniel, TOP-423): only a lead runs Fable 5.2. You run `claude-opus-5-5-high` and stay on it.**

- Any subagent you launch names its model in the call: `claude-opus-5-5-high`, or `claude-sonnet-5-5-high` for a really well-specified task. Never Fable, never `inherit`, never a launch with no model.
- If a picker offers neither model, do not launch: do the work yourself or send it to me.
- A plan, a design or a judgment no check catches comes to me, in your next message, with what you recommend.

[2026-11-14T16:07Z] b8b3aae78cea
training to STEP, Sat 14 Nov 8:06 AM PST. Read: your 7:46 and 7:55 AM messages.

**#9863's draft flag is off (I cleared it after your mark; GitHub read not a draft at 7:56 AM), so your mark at b5f9048a2a14 stands and the question is closed. READSTEP reads #10136 at c954f564391fdc746f8381dd57147540a9413f8a now; its verdict comes to you and to me.**

- #10136 stays a draft until #9863 lands; then merge `main`, retarget it to `main`, and say the new head.
- The nested `sdpa_kernel` case: agreed, left out, and the body says so.
- On run r20261114-155410-86b8's end and READSTEP's READY you mark, by the rule you have.

[2026-11-14T16:13Z] 458cf3ae0c09
READSTEP, Sat 14 Nov, 08:12 AM PST. First read of draft #10136 at c954f564391fdc746f8381dd57147540a9413f8a, stacked on `cursor/trn144-step-fused-attention-main-28a9` at b5f9048a2a14.

**The refusal does what the body says, before every `gpu` trace and on no other path, and its test catches each switch. But torch 2.14 has two more public settings that change which kernel a call reaches, the backend priority order and the active flash implementation. The refusal reads neither, and the body doesn't say so.**

The delta is one commit in two files, +42 -10 (RAN, `git diff --stat`). The base branch is still at b5f9048a2a14 (RAN, `git ls-remote`).

## Your questions

1. **The code does what the body says, and nothing else.**
   - READ: the whole diff. It adds `_SDP_BACKENDS`, `_sdp_off()` and a refusal at the top of `export_step` under `if gpu`. The rest is docstring text: `_CardSdpa`'s gains one clause and is re-wrapped, and the module docstring and `export_step`'s gain a sentence each.
   - The refusal runs before the trace on every `gpu` path, and on no other. READ: `_CardSdpa` is entered only in `_gpu_branches` (`train_step.py:278`), which only `export_step` enters, under `gpu` (`:334`), after the refusal (`:293`).
   - `step_report(cc=...)` either calls `export_step(..., gpu=gpu)` or takes an `exported` it requires to be a `gpu` export (`:378-381`) (READ).
   - Mutation D in question 3, refusing every export, fails every case, so the export without `gpu` reads no switch (RAN).
2. **The refusal is right for what it reads, but two settings are missing.**
   - The four switches are the four `<name>_sdp_enabled` getters of torch 2.14's `torch.backends.cuda` (RAN, `dir(torch.backends.cuda)`).
   - The message names every switch that is off, with its setter, and says how to fix it. RAN, with flash and mem_efficient off: "the step is exported with torch.backends.cuda.enable_flash_sdp, torch.backends.cuda.enable_mem_efficient_sdp off, ... export it with each on".
   - Missing: the priority order (MUST FIX 1, part a). RAN: under `sdpa_kernel([CUDNN, FLASH, EFFICIENT, MATH], set_priority=True)`, all four switches read on and `torch._C._get_sdp_priority_order()` is `[3, 1, 2, 0, 4]`, cuDNN first. The `gpu` export passes the refusal and records `aten._scaled_dot_product_flash_attention` and its backward.
     - torch's docstring says `set_priority` makes the list's order "their priority order" (READ). The default is `[1, 2, 0, 3, 4]` (RAN), the order `_CardSdpa` assumes.
     - The trace's own `sdpa_kernel(MATH)` keeps the caller's order (RAN), so it doesn't reset it.
     - On a card that takes the call, cuDNN would run it, not flash (INFERRED: no GPU here).
   - Missing: the active flash implementation (MUST FIX 1, part b). `torch.nn.attention.activate_flash_attention_impl("FA3"|"FA4")` is public, opt-in by a call or an environment variable (READ, `torch/nn/attention/_registry.py:61-80`). `current_flash_attention_impl()` reads it back: `None` here, with `['FA3', 'FA4']` registered (RAN).
     - With FA3 or FA4 active, the flash op keeps its name but runs another kernel, while the rules lower it as `fa2_train` (INFERRED from `_fa3.py:45`'s `_set_sdp_use_fa3`).
   - Not missing (INFERRED):
     - `allow_fp16_bf16_reduction_math_sdp` changes the math path's numerics, not the kernel, and the trace runs that path under the exporter's own setting.
     - `overrideable` is the backend for non-CUDA devices.
     - `cudnn` off changes no kernel `_CardSdpa` models on a card where cuDNN is last, since flash, memory-efficient and math take every call first. Refusing it is stricter than needed, but it matches "every backend on".
3. **The tests** (RAN, `/tmp/mut10136.py`, each mutation alone, then undone; the worktree is clean):
   - At the head, the four cases pass.

     | Mutation | Result |
     |---|---|
     | A: refusal skipped | `[cudnn]`, `[flash]`, `[math]` and `[mem_efficient]` fail |
     | B: `flash` dropped from `_SDP_BACKENDS` | only `[flash]` fails |
     | C: `math` dropped | only `[math]` fails |
     | D: refusal on every export (`if gpu and` dropped) | all four fail |
     | E: message names only the first off switch (`off[0]`) | survives: no case has two off (SHOULD FIX 1) |

   - Each case's switches are restored, through `sdpa_kernel`'s exit, even after the raise. RAN: after a refused export inside `sdpa_kernel`, all four read on and the order is the default.
   - Run in file order, the four new cases and the four existing GPU-export tests pass (RAN).
4. **The body** is true as far as it goes (READ, `gh pr view`, draft, head c954f564391f).
   - RAN: the ratchet gate gives "1 change(s), 0 failed".
   - RAN: the one I001 in `train_step.py` under `ruff_vllm.toml` is at b5f9048a2a14 too.
   - It names no machine ("On the VM", "on a node"), and its numbers are test counts in the body, not the tree.
   - "What is not checked" names the `sdpa_kernel` a step's forward enters, and no GPU. It doesn't name the priority order or the flash implementation, so it doesn't say everything that is not covered (MUST FIX 1).
5. **For the lead, beyond the checks:**
   - Whether those two settings are read here or named as not checked is a scope call. Both are one getter each: `torch._C._get_sdp_priority_order()` against the default, and `current_flash_attention_impl() is None`.
   - The first getter is private API, pinned at torch 2.14 (RAN).

## MUST FIX

1. Read the priority order and the flash implementation in the refusal, or name both in the body's "What is not checked".
   - Today a `gpu` export under `sdpa_kernel(order, set_priority=True)` passes and is traced in the default order (RAN), and one with FA3 or FA4 active is traced as FA2 (INFERRED).
   - Fix, smallest: add both to the refusal, with a case each: cuDNN put first with `set_priority=True`, and a monkeypatched `current_flash_attention_impl` returning `"FA3"`.
   - Or, if they are out of scope, add a bullet each under "What is not checked".

## SHOULD FIX

1. Add a case with two switches off, asserting both are named; mutation E, `off[0]`, survives today (RAN).

## NOTE

1. A step's own `sdpa_kernel` inside the trace: `_CardSdpa` could read the switches at each call and would see any set but exactly `[MATH]`. The body's "a check before the trace can't see it" is true as written (INFERRED).
2. STEP's quick tier r20261114-155410-86b8 is STEP's. I didn't read it or launch any run.
3. Events: the shell hook delivered the coordinator's broadcast on the model rule once (16:07Z). It wasn't addressed to me alone and needed no action; I launched no subagent. No PR comment is on #10136 (RAN). Nothing else addressed to @training reached me.

## For the lead to read

1. `adapters/vllm/verity_vllm/program/frontend/train_step.py:171-177, 293-295`: the switch list, `_sdp_off` and the refusal.
2. `train_step.py:192-194`: `_CardSdpa`'s "whose order of backends is torch's default", the assumption nothing reads.
3. `adapters/vllm/tests/program/test_train_sdpa.py:259-270`: the test.

NOT READY 1 c954f564391fdc746f8381dd57147540a9413f8a

