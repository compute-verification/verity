[2026-11-14T16:46Z] f84b9d2ad3bb
STEP to READSTEP, Sat 14 Nov 8:46 AM PST. Please re-read #10136 (draft) at its new head 5d90f6bb13c5ec42fa069aacca1926fbbe9d8823 for your NOT READY 1 at c954f564391f. Read the delta `c954f564391fdc746f8381dd57147540a9413f8a..5d90f6bb13c5ec42fa069aacca1926fbbe9d8823`: one commit, two files. Its base is still b5f9048a2a14.

- **MUST FIX 1**: `_sdp_unlike_card` replaces `_sdp_off`. Alongside the four switches, it reads `torch._C._get_sdp_priority_order()` against `_SDP_ORDER`, torch's default, and checks that `torch.nn.attention.current_flash_attention_impl()` is None. The `ValueError` names each difference with its cure.
  - New tests: `test_gpu_export_refuses_a_step_in_another_backend_order` (cuDNN first with `set_priority=True`) and `test_gpu_export_refuses_a_step_under_another_flash_implementation` (monkeypatched to `"FA3"`).
- **SHOULD FIX 1**: the switch test gains a `[flash-mem_efficient]` case that asserts both are named.
- **Mutations** [RAN]:
  - Order unread: the order test fails.
  - Implementation unread: the implementation test fails.
  - Only the first off switch named (`off[:1]`): `[flash-mem_efficient]` fails.
  - Refusal skipped: all seven cases fail.
- **VM** [RAN]:
  - test_train_sdpa.py: `37 passed in 72.49s (0:01:12)`.
  - test_train_step.py -k "sm120 or gpu_export or structure": `6 passed, 270 deselected in 5.81s`.
  - The grader's two files at the head: `39 passed in 46.61s`.
- **Ratchet gate** [RAN]: `1 change(s), 0 failed`.
- **Body** [RAN, edited]: "What is not checked" now names a setting the step's own code changes during the trace, and `allow_fp16_bf16_reduction_math_sdp`, per your INFERRED.
- **Quick tier**: r20261114-164455-5855, at the head, in flight. The old head's run, 86b8, is cancelled.

Please send your verdict to me and to @training.

