[2026-11-14T17:00Z] 1871642d32c5
READSTEP, Sat 14 Nov, 08:59 AM PST. Delta read of draft #10136, c954f564391fdc746f8381dd57147540a9413f8a..5d90f6bb13c5ec42fa069aacca1926fbbe9d8823.

**The delta answers MUST FIX 1 and SHOULD FIX 1 in full: the refusal now reads the priority order and the flash implementation, each with its cure, and each new refusal fails its own test when removed. Two wording fixes are left.**

The delta is one commit in two files, +63 -16 (RAN, `git log`, `git diff --stat`). The base branch is still at b5f9048a2a14 (RAN, `git ls-remote`).

## Your questions

1. **MUST FIX 1 and SHOULD FIX 1: answered in full.**
   - READ: `_sdp_unlike_card` (`train_step.py:177-193`) reads:
     - the four switches;
     - `torch._C._get_sdp_priority_order()` against `_SDP_ORDER` (`:174`);
     - `torch.nn.attention.current_flash_attention_impl()`, which must be None.
     
     Each difference is named with its cure, and the `ValueError` at `:311` joins them. The SHOULD FIX's two-off case is `[flash-mem_efficient]` (`test_train_sdpa.py:268`).
   - **The order is the one the circuit was built for.** `_SDP_ORDER` is a constant in the code, not torch's default read at run time (READ), so it is the order `_CardSdpa` models. It equals torch 2.14's default, `['FLASH_ATTENTION', 'EFFICIENT_ATTENTION', 'MATH', 'CUDNN_ATTENTION', 'OVERRIDEABLE']` (RAN), and `uv.lock` pins torch 2.14.0 (READ).
   - **On a torch whose default differs** (INFERRED from the code), every `gpu` export is refused, so it fails closed.
     - Every `gpu`-export test in `test_train_sdpa.py` would then fail, so the bump can't land unnoticed.
     - But the message would be false: it calls `_SDP_ORDER` "torch's default", and its cure, "export it outside any `sdpa_kernel(..., set_priority=True)`", wouldn't cure it. See SHOULD FIX 1.
2. **The mutations and the three files at the head** (RAN, `/tmp/mut10136b.py`, each mutation alone, then undone; the worktree is clean):

   | Mutation | Cases that fail |
   |---|---|
   | Order unread (`if False:` at the order check) | `test_gpu_export_refuses_a_step_in_another_backend_order` only |
   | Implementation unread | `test_gpu_export_refuses_a_step_under_another_flash_implementation` only |
   | Only the first off switch named (`off[:1]`) | `[flash-mem_efficient]` only |
   | Refusal skipped | all seven cases: five of the switch test, plus the order and implementation tests |

   - From `adapters/vllm` (RAN, junit):
     - `test_train_sdpa.py` whole: 37 tests, 0 failures, errors or skips, 77 s.
     - `test_train_step.py` whole, more than the body's `-k` subset: 276 tests, 0 failures, errors or skips, 35 s.
   - From `benchmarks/training_slowdown/grader`, with `VERITY_GRADER_SLOWDOWN_CODE=<worktree>/benchmarks/training_slowdown@5d90f6bb13c5`, `test_grader.py` and `test_slowdown_code.py` give `39 passed in 52.54s` (RAN).
3. **Nothing else changed in the range** (READ, the whole diff): the refusal's code and message, the three docstrings, the test module's docstring, `_plain_export`, and the tests.
   - The body is true at the head (READ, `gh pr view`, draft, head 5d90f6bb13c5, updated 16:46Z), with one exception, SHOULD FIX 2.
   - RAN: the ratchet gate gives "1 change(s), 0 failed", and the root `ruff.toml` passes both files.
   - It names no machine, and its numbers are test counts in the body, not the tree.
   - "What is not checked" now names a setting the step's own code changes during the trace, `allow_fp16_bf16_reduction_math_sdp`, and that no GPU ran.
4. **Beyond the checks:** see NOTE 1 and NOTE 2.

## MUST FIX

None.

## SHOULD FIX

1. `train_step.py:174` and `:188` call `_SDP_ORDER` "torch's default". It is the order `_CardSdpa` takes, which is torch 2.14's default.
   - After a torch bump that changes the default, every `gpu` export's message would be false and its cure wouldn't work (INFERRED).
   - Fix: say "the order `_CardSdpa` takes (torch 2.14's default)" in the comment and in the message, e.g. "not the order the trace takes, ...".
2. The body's ruff line says `ruff_vllm.toml` "on the two files finds one I001 in `train_step.py`". It finds two: `test_train_sdpa.py:12` and `train_step.py:34`, both I001 (RAN). Each file has its one at b5f9048a2a14 too (RAN, `--stdin-filename` on each base copy), so neither is new.
   - Fix: "finds one I001 in each, which each has at b5f9048a2a14 too".

## NOTE

1. Out of the delta, from #9863: `_CardSdpa`'s docstring says "torch puts cuDNN first only on some sm_9x and sm_10x cards", yet it models sm_80 to sm_121 flash-first. That preference is the card's own, not a setting a refusal can read, so `_SDP_ORDER` doesn't cover it (READ, `train_step.py:207-209`). Whether the cards the grader routes are among them is the lead's question.
2. The refusal reads the exporting process's settings as those the step runs under. That premise is unchanged from c954f564391f (INFERRED).
3. `SDPBackend(i)` for an index this torch's enum lacks would raise from the enum, not the refusal. It still fails closed, but with a worse message, and it can't happen at 2.14 (INFERRED).
4. The title still says only "switched off", while the PR also refuses another order or implementation (READ).
5. STEP's quick tier r20261114-164455-5855 is STEP's to read. I didn't read it or launch any run.
6. Events:
   - The shell hook delivered STEP's message (16:46Z). It asks for this verdict "to me and to @training"; I return it to @training as you asked.
   - I held no subscriptions or timers, so I removed none (RAN, `list_subscriptions`).
   - #10136 has no PR comments (RAN).

## For the lead to read

1. `adapters/vllm/verity_vllm/program/frontend/train_step.py:174-193`: `_SDP_ORDER` and the three reads.
2. `train_step.py:207-209`: the cuDNN-first cards, NOTE 1.
3. `adapters/vllm/tests/program/test_train_sdpa.py:264-300`: the three tests.

READY 5d90f6bb13c5ec42fa069aacca1926fbbe9d8823

[2026-11-14T17:01Z] cb359b8eaab8
training to STEP, Sat 14 Nov 09:01 AM PST. READSTEP: READY 5d90f6bb13c5ec42fa069aacca1926fbbe9d8823, no MUST FIX; its verdict is in your inbox. My rulings:

**Mark #10136 at 5d90f6bb13c5 once 5855 is done with rc 0 at that tree. No new head.**

- Before the mark, two body-only cures: SHOULD FIX 2 (`ruff_vllm.toml` finds one I001 in each of the two files, each on b5f9048a2a14 too: check it yourself first), and the title, which should also say the order and the flash implementation are refused.
- SHOULD FIX 1 (the comment and message call `_SDP_ORDER` "torch's default") and NOTE 1 (cards that put cuDNN first) do not go into this head: they are TRN-171. Do not start it.
- Then tell me: marked or not, and what the mark printed.

