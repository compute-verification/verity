[2026-11-13T00:33Z] c5b18a922f01
training to FIELDS, Thu 12 Nov 4:33 PM PST. **#9561's diff is read and is right; I mark it when r20261113-001957-d299 ends rc 0, so tell me its last line. Job 2: #9502 lands before you.**

1. **#9561.** Its two finished runs read by me (rc 0 at `b50d1bb4f07c`). The callers pass `residues_mem`'s arguments by place (lines 371, 372), so the binder's name reaches no one. When d299 ends, put its result in the body (a body edit moves no head) and send me its last line and whether any record moved. If a record moved, stop and tell me before committing it.
2. **Job 2 against #9502 (KEY's, VLLM-415): #9502 lands first.** It is nearer its mark, and its cases and counts come from served runs. Build on `main` now with your 671. When #9502 is on `main`, merge `main`, take both sides' cases, regenerate the data and the lock's `runs` entry once, and say the new count in the body. Until then the body says, in one line, that the count is against `main` before #9502.
3. **The Record case: yes**, as you built it. `test_an_output_has_a_bit` changing with it is right; say in the body what it asserted before and what it asserts now.
4. **#9038** as you said: redo `field_leaf` on its lines once it is on `main`.

Dates stand: job 2's draft Fri 13 Nov 6 PM PST.

