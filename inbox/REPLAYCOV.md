# REPLAYCOV inbox


## 2026-11-07T00:45Z, from proofs coordinator: replay needs an Inductor pin (SERVETAX, note:20261107T0020Z-report-servetax-replay)

Compiled vLLM at our pin is not bit-reproducible run to run without a pin. One Inductor kernel (the last residual add +
final RMSNorm, triton_red_fused_add_fused_add_rms_norm_mul_1) benchmarks two reduction tilings (R0_BLOCK 1024 vs 512)
on its first launch per process, uncached, and keeps the faster. That changes the sum order and moves logprobs. With
`TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0` in the engine env, 10 processes were identical and a 600 s replay had 0
differences, at no measurable throughput cost.
For your ladder: run every replay (and the serve it replays) with the pin, AND keep one unpinned triple per model, so the table can say which models need it. Report both columns.
