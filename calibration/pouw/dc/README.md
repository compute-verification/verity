# dc: the epsilon program's measurements on Hopper and datacenter Blackwell

How much can one H200 and one B200 do in each SM each clock? PoUW's ε for a card is computed from a model of that
card, and this directory measures the H200 (sm_90a) and the B200 (sm_100a) for it: their tensor instructions, the
scalar pipes that bound the fold, the uniform datapath, and the honest Dither GEMM with its duo fold, checked word for
word. Every probe's rate is timed only after its SASS gate passes, and the GEMMs' only after they are exact. The ACCEL model's census lines for
both cards are written from these runs.

## Try it

~~~sh
uv run machine-model eps --device sm_100 --credit floor --operands floor --table
~~~

This reads the B200's census lines, which these runs wrote, and prints each PoUW design's ε on them, one line a design.
It needs no GPU; `--device sm_90` gives the H200's.

## Run it

Each job is the command of a `research run` from a pushed commit, on a node with the card, CUDA 13.0
(`/usr/local/cuda-13.0`) and driver ≥ 580. `run.sh` is the job for one family:

~~~text
research run --on H200 --project verity --source /workspace --cwd source --custody-ttl 12h \
  --timeout 60m --env GPU_LEASE_WHO=dc -- bash calibration/pouw/dc/run.sh sm90
research run --on B200 ... -- bash calibration/pouw/dc/run.sh sm100
research run --on H200 ... -- bash calibration/pouw/dc/run.sh sm90jb   # the stress of the H200's rows: probes only
~~~

A run builds and SASS-gates everything before it touches the GPU, and `DC_BUILD_ONLY=1` builds and gates without one.
Its results are not kept here: the analyses print them from a run's `out/`.

## What is not covered

Not covered yet: sparse wgmma and tcgen05 (metadata operands), the N sweep for the FP kinds (s8 only), cta_group::2 (a
CTA pair), M4e beyond fp8acc's binade question (kind::f16/tf32/f8f6f4 accumulation against the silicon models in
general), lanes beside wgmma with more than one lane warp a SMSP (`wgU_`'s lanes stay well under their rate alone,
and under the block-timed points the sm_90 facets rest on), operand streaming (TMA) in `pgemm.cu`, and a pipelined
sm_90 GEMM. On sm_90 CUDA 13.0 has no native FP8 or s4 mma.sync: e4m3 m16n8k32 is 2 HMMA.16816.F32 + 4 FADD (timed as
that), s4 m16n8k64 a CALL (not timed).
