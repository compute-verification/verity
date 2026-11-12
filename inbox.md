[2026-11-12T17:09Z] fd4b5b8651cb
BACKWARD, TRN-140, 9:10 AM PST (17:10Z). Your 9:01 is read; the second capture is running (r20261112-170845-7150): efficient forced in bf16 and in float32 (the draws before rounding, and the bf16 words cast up), B 2 and B 16, default and deterministic, is_causal as model.py calls it, plus mask tensors, what torch selects by itself per dtype, one-operand-grad calls, and causal against non-causal Kineto durations for flash and efficient. Lease ends well before 11:30 AM.

ONE READING YOU NEED BEFORE 1 PM, READ not measured (mem_eff_attention/gemm_kernel_utils.h 130-159 at v2.14.0; cutlass e05f953a, torch's pin, gemm/warp/mma_tensor_op_fast_f32.h 256-265, 353-374): the efficient kernel in FLOAT32 is not float32 multiply-adds. On sm_80 and up its products are tensor-core `OpMultiplyAddFastF32`: each operand is split into a TF32 "big" part (round toward zero) and a TF32 "small" part, and every 16x8x8 step runs three TF32 tensor-core multiply-adds (small x big, big x small, big x big) into a float32 accumulator. So a float32 efficient call is priced neither at 12,637 nor at 1,480.125 a multiply-add: it is three TF32 dot steps a multiply-add. In bf16 it is the plain 16-bit tensor-core step, as flash. The SASS of both comes back with this run; I say MEASURED or not when the words are compared.

Product count, READ (kernel_forward.h, kernel_backward.h): forward two (Q K^T, P V), backward five (K Q^T recomputed, P^T dO, dO V^T, dS K, dS^T Q), dPsum as a float row dot in the kernel. P is rounded to q's dtype before P V, P^T dO AND before dS (flash keeps float32 P for dS); dS is scaled, then rounded.

[2026-11-12T17:11Z] 4c473da6ce6d
BACKWARD, TRN-140, 9:14 AM PST (17:14Z). BLOCKED: THE SECOND CAPTURE CANNOT START BEFORE 11:30 AM. It is not a lease refusal: the node's pool slots refuse it.

r20261112-170845-7150 (launched 9:08 AM, `research run --on vy-nebius-2 --pool research`) is waiting on a slot, and slot.json says why: "every check slot is taken (r0 ... r5: window 2026-11-12T19:30Z (240 min) is open or starts within 180 min)". The research slots on node 2 carry lead=180 on the fill windows file, so they have admitted nothing since 8:30 AM PST and admit nothing until the window ends at 3:30 PM PST. My first capture got in at 8:16 AM, before the lead began.

The job needs one GPU for about 2 minutes and about 5 minutes of CPU (venv, then the SASS off the lease). I have not gone around the gate and will not: the slots file is infra's.

WHAT WOULD UNBLOCK IT, YOUR CALL OR INFRA'S:
1. infra admits this one run (a shorter lead on the r slots, or a slot without windows=) before 11:30 AM; it starts by itself the moment a slot opens, nothing to relaunch.
2. another machine with the same card (sm_120, torch 2.14.0+cu130) that I may use: name it and I relaunch there.
3. it stays queued and starts when the window ends at 3:30 PM PST; with the node stopping at 4 PM (INF-662) that leaves about 30 minutes for a 7-minute job and its custody upload. This is what happens if nobody acts.

Meanwhile, on the CPU and from the source: the efficient kernel's reading (forward and backward, bf16 and float32), its product count, and the twins ready to run on the words when they come.

