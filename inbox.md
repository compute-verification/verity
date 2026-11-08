[2026-11-08T09:22Z] 75e7c0b57e31
vllm-twostage-int 09:32Z: VLLM-250 comment posted. (1) For (a)/(c): the check template must be TokenSelect_v4{V=154880} (the compiler swaps v1 for dead-free v4; class G spans hold no v1), per_output 1 at rank 0, n_rho = 256. (2) Shared-row interior done for TWOPROVE at 3cab8f4f3f; quick suites running (r20261108-091913-ca38), re-mark #8055 after. (3) Cost, CPU replay, scaled from 4 layers: 6.0 core-h a rank for a 49-position span (48 over 8 ranks); about 500 core-h for a (256,256) span, so not viable for (c). Ask: should (b) build a GPU shape-only replay capturing each step logits row for the TokenSelect units, with node A time after Tue serve?

[2026-11-08T10:34Z] 7a8fedaa2bdc
vllm-twostage-int 10:52Z: answered proofs 10:15Z+10:25Z on VLLM-250. (1) Every template ports at ZKTWO spec do NOT fit Tue: measured 58-71 MiB/pos/rank (8 served class G Programs, full depth, weights by ref; r20261108-102235-e4f7) -> 335 GiB a (256,256) span over 8 ranks, Q=4 1.31 TiB vs 120 GB free; and no capture makes class G instance rows on GPU today. (2) What fits: a layer-boundary capture in the same slot (each layer input hidden+residual and cache rows written, sampled rows only): ~32 GiB at Q=4, minutes. Every template interior then = CPU eval per layer-step, parallel, gated bit-exact by the next layer capture: ~600 core-h a span, Q=2 ~1,200 core-h. YOUR CALL: that CPU budget line; I recommend adding the layer-boundary capture Tue as insurance. (3) One-description interior: ~30 lines for template files + ~150 for wiring on my side; values/salts/roots unchanged; yes to re-agree by Mon 12:00Z. Packaging: #8055 is at 641/800 lines, so the GPU capture (sampler wrapper + replay.py --capture, @proofs file) goes in a follow-up PR on this branch after #8055 lands, or a second branch if you name one. #8055 quick suites on 5969ef08d2 (capture consumer, dry-run path) running: r20261108-101713-a343.

[2026-11-08T11:06Z] c3bbccd0b2c4
vllm-twostage-int 11:07Z: #8055 is ready at 981276e76e: 24 of 24 suites pass remotely, and the body names Goal proofs/spot-two-stage-e2e and Refs VLLM-250. The GPU capture (sampler_capture plus replay.py --tp/--capture, 264 lines, tested locally) is held for a follow-up PR because of the line cap. Chunk ask, answered on VLLM-250 at 11:06Z: (c) becomes the pair, 2,420 TokenSelectChunk_v2{C=64} units and one TokenSelectCombine_v2{V=154880,C=64} unit a token, from the same captured row; both pass Flock's check_proof_block as one-row ports, and TokenSelect_v4 is refused. The capture layout and tests come by Mon 12:00Z; the template-major interior lands on P1's Stream; the PR is marked by Mon 18:00Z. I flagged one gap: round 2's link rule (a template called exactly once) can't tie the chunks' logits to the boundary. That is TWOPROVE's fix, with twostage's BOUNDARY_ROWS naming 2,420 leaves of 64 words a position. The every-template capture estimate comes by 18:00Z, after LIFETIME's rehearsal.

[2026-11-08T11:26Z] 49f50e3388a6
vllm-twostage-int 11:27Z: two asks. (1) A branch for the follow-up PR, stacked on #8055. It is the GPU capture plus the pieces and the template-major interior, about 544 insertions, in 3 local commits, tested; its quick suites are running (r20261108-112221-1935, ETA about 11:45Z). #8055 is admitted, but the queue is long, and on my one branch the follow-up can only open after #8055 lands, which could push it past the Mon 18:00Z dry run. Can I push it as a stacked PR on a second branch (cursor/vllm-twostage-int2-496a)? (2) A GPU slot before Tue for a real-vLLM smoke of replay.py --capture: a small model at TP2 on the pinned vLLM, about 30 min, to check collective_rpc installing the sampler wrapper, which has only been tested on stand-ins. Is node A available Mon, or another sm_120 or H200 box? Also, on VLLM-250 at 11:25Z: P1 counts a replay block's instances only among its root's calls, so twostage's REPLAY_c must call the pieces directly, not batch TokenSelect_v4.

[2026-11-08T11:48Z] 5423c193aeb6
vllm-twostage-int 11:49Z: #8055 is rebased onto checked (d00f68909e) and pushed at 65c15eac42. The one new commit imports verity.core.protocols.spot in interior_rows.py and its test. Locally the 7 tests pass on it, against the Spot Lean verifier rebuilt from checked. Quick suites run r20261108-114812-a371, ETA about 12:10Z; I re-mark when it passes. The follow-up commits are rebased on top, still local.

[2026-11-08T12:12Z] 712c87f08013
vllm-twostage-int 12:15Z: #8055 is re-marked at 65c15eac42 (rebased on checked d00f68909e, imports verity.core.protocols.spot). Quick suites r20261108-114812-a371: 23 of 25 pass. The 2 reds fail identically on checked d00f68909e with none of my commits (I ran both there): verity-vllm tests/program/predict/test_step_needs.py::test_production_sizes_refuse_by_name and benchmarks/route_capture tests/test_gap.py::test_production_steps_name_each_gap. Both expect a no capture refusal for cublas-linear and cublas-router that checked no longer gives, likely since a capture landed for them. step_needs is our lane's suite, so someone should fix it forward. On the follow-up: still waiting on your branch answer.

[2026-11-08T13:07Z] b8e7d943e189
vllm-twostage-int, 13:07Z Sun.
The follow-up is open and unmarked: https://github.com/compute-verification/verity/pull/8111, head ef514aa009, base #8055's branch (cursor/vllm-twostage-int-496a), Lane vllm, Refs VLLM-250, @proofs named, and TWOPROVE's link-rule gap listed as open with owner TWOPROVE. The P1 test stays local until #8066 lands.
e10d (quick suites on 66c42f76f0): 25 of 28 suites passed and 3 failed, with 5 failed tests between them. Two are checked's #7769 reds (test_step_needs, route_capture's test_gap). Three were mine, all fixed in #8111:
- sampler_capture wasn't a public interface for serve_tax's imports; it now states __all__ and is in PUBLIC.
- interior_rows imported the predictor's window_rows, which made a P9 package cycle and broke test_predict's record-path rule. The chunk kernel moved to verity/ml/definitions/numerics/token_select_rows.py. Its first home there, numerics/rows.py, put that file past P10's 800-line cap, which the next quick run (6608) caught, so it got its own module.
The quick suites at ef514aa009 are r20261108-125900-12dd. It is queued behind other runs for a check slot on cpu-2; I expect a result around 13:50Z and will put it in the PR body.
LIFETIME: I answered their questions 1-4 on VLLM-250 and proposed who owns what.
- LIFETIME: the GPU kernel (it needs row/v2, since every token-select piece port is v2), the recorder and the two passes.
- Me: the port map, the token select's GPU taps, interior taking a GPU commitment, and opening the sampled cells.
- vllm-twostage: REPLAY_c. ZKTWO: P1. TWOPROVE: the link gap.
Three of my pieces are already built on the local branch cursor/vllm-twostage-int3-496a at ffa705f1cb:
- interior(device=DeviceCommit(roots, bc, salts)) registers byte for byte what the host commit does under the same salts.
- sampler_capture.piece_rows lays out the pieces in torch, equal word for word to pieces on CPU, NaN/inf/ties included.
- port_map reads which committed word each unit's port word is off read_all itself.
Its quick suites are r20261108-130407-12ea, queued. May I push it as a third stacked branch and open its PR unmarked when 12ea passes?
The smoke test is ready for your go, Mon 16:00-17:30Z on node A: bash ~/.research/notes/lanes/vllm/tools/capture_smoke.sh /tmp/wt-3 . After the capture, it now also checks piece_rows on cuda:0 against the host's pieces (capture_pieces_gpu.py). The CPU half is capture_interior.py, as before.

[2026-11-08T17:50Z] 74ea11c01757
From vllm-certtick (bc-b4ae9664-364a-5558-b1b4-c901de94c9b5), 17:58Z.

1. Please let me post as @vllm on thread 1794083350.111549: `research msg names open --as vllm --poster bc-b4ae9664-364a-5558-b1b4-c901de94c9b5`. `reply --as vllm` refuses me (not a holder). Until then I can't post READY, the certifier hand-off or the report there.

2. Server run r20261108-173912-1d69 is launched on node A (TAKEOVER=1). The FP8 kernel check passed, the GPUs were free at 17:42Z, and it is waiting for 18:45Z. TAIL=300 reached launch.sh (network-accounting's 6e165a8e54).

3. A new risk, bigger than the tail. Please post this to network-accounting if you'd rather not grant me (1):

"@network-accounting TAIL aside, tick-sunday's schedule will fail many connections `late`. `Warden._egress` needs a whole P = 247 B at each slot from s + D. At τ 220 ms a stream makes ~4.5 tok/s (~1.1 KB/s) against egress's 2.47 KB/s, so D's 15 s head start runs dry ~27 s after admission, and any answer past ~110 tokens fails. Seed 0 has 1,611 of 3,000 requests with max_tokens > 110. A fix on your branch, your call: `--cell-bytes 96` (drains ~3.3 tok/s) with TAIL=360, or D ≥ 2,400 with lanes and queue ≥ 512. Can you push one by 18:45Z? vllm's server is r20261108-173912-1d69, as staged."

Without a fix, the run still serves, but about half its connections end in FAIL (late), vLLM aborts those requests, and v1's encoding refuses any trace with an abort. So step 4's advice bits would have no record to encode.

[2026-11-08T18:00Z] 0a5958725426
vllm-certtick, 18:01Z: the late-fail risk is confirmed without a GPU. warden_serve rehearse --fake at tick-sunday's schedule, streaming at vLLM's byte rate under τ 220 ms (~1.1 KB/s): B 256 (as staged) failed 16 of 60 connections `late`, each at egress cells 137-150 (r20261108-175129-0b66); B 96 failed 0 of 60 (r20261108-175218-7402). The fake answers shorter than greedy GLM-5.3, so the real run loses more (up to ~1,600 of 3,000), and each failure aborts its vLLM request, which makes v1 refuse the trace. Please either grant me `research msg names open --as vllm --poster bc-b4ae9664-364a-5558-b1b4-c901de94c9b5` or post to network-accounting: "TAIL aside, tick-sunday's schedule fails answers past ~110 tokens `late` at τ 220 ms (fake check: B 256 failed 16/60, B 96 0/60; r20261108-175129-0b66, r20261108-175218-7402). Please set --cell-bytes 96 and TAIL=360 on your branch by 18:45Z." If nobody on network-accounting answers by 18:50Z, tell me whether I should run their branch as staged (about half fail) or a copy with --cell-bytes 96 and TAIL=360 on a cursor/vllm-certtick branch.

[2026-11-08T18:04Z] 451e5a0d7878
vllm-certtick: the server run is now r20261108-180335-71b2 (relaunched 18:03Z with EXPECT=1000000, otherwise as staged, TAKEOVER=1; r20261108-173912-1d69 cancelled while it held nothing). Why: ticksun's idle stop (3,000 releases, then 300 s quiet + 30 s) lands about 19:36:10Z, before the certifier's last window (about 19:36:30Z at TAIL 300, 19:37:30Z at 360), which would end its watermark stream with error "closed" in warden.json. Now the server serves to its hard 19:42:30Z stop. Still open: @vllm poster rights for me (or you post), and B=256 as staged vs the B=96 variant (local cursor/vllm-certtick-6162 at 872be23818) if network-accounting is silent by 18:50Z.

