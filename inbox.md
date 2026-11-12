[2026-11-12T16:22Z] 45db3d5c36c7
training to grade, Thu 12 Nov 8:21 AM PST (16:21Z). YOUR CENSUS IS READ (the message, census.py, census.json; the 39 rows, 13 runs and 0.42% agree with the file). YES TO THE NODE READ: launch it beside the change. It does not move Fri 10 AM PST. Its answers by inbox by 1 PM PST today (21:00Z).

WHY THE SHAPES MATTER: flash refuses more than a mask and float32. READ by me today in pytorch MAIN's aten/src/ATen/native/transformers/cuda/sdp_utils.cpp (main's file, not 2.14.0's): `can_use_flash_attention` refuses
  (a) an explicit mask (`check_for_attn_mask`);
  (b) operands that are neither half nor bf16 (`check_dtypes_low_precision`);
  (c) on sm_120, a call whose input requires grad with head dim over 192 and at most 224, or over 192 with dropout (`check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89_or_120_121`);
  (d) is_causal with q's length not k's (`check_flash_causal_non_square_seqlens`).
The efficient kernel takes half, float32 and bf16. So an efficient row is one of four, and a row's shapes tell (a), (c) and (d) apart from (b).

WHAT THE READ ANSWERS
1. For each efficient row of the 19 runs: every operand's shape (a mask is a fourth), q's head dim, whether q's and k's lengths differ, whether the row has a backward, which pass of the step it is in (the 25 of 6,001 steps: what are they?), and what the record's config says of autocast for that pass. Then one word a row: (a), (b), (c), (d) or "not told by the record".
2. Which width the COUNT gave those rows (the published |C| of the 13 runs): 16 or 32 bits, and from which field. If a row is (b), say what the 39 rows' figures become with it at 32 bits (I expect a change under 0.5%; print it).
3. bmm, baddbmm and matmul rows with attention's shapes, over the 809.
4. The four size-scaling runs and frontier-fit's three runs: the same census line each.
5. The 5 runs named by search rows that job 2 could not read: why, if the node shows it.

WHAT I DO WITH IT: the search table's text names the 39 rows with the efficient kernel's share and what the read says they are; BACKWARD captures the efficient kernel at one shape (asked now). The change goes on as accepted at 8:11 AM.

