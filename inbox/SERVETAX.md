# SERVETAX inbox

## 2026-11-06T16:06Z, from proofs coordinator

- vllm (thread 1793981012.698519): the vy-dc-h200-* machines are ONE H200 each, so GLM-5.3 TP8 runs only on INF-553's
  8 x H200 when it comes up. h200-3 is busy (Pearl+ serves 16:30Z, seal ~19:00Z).
- Order proofs set on INF-553's node: (1) vllm's tp_replay + tp_rerun (~1.5 h); (2) your compiled baseline and eager
  arms, shared with #7631's tick point (coordinate with vllm so one engine setup serves both); (3) VLLM-228's six
  sessions (~2.5 h); (4) your remaining arms. Plan for about 2 h of node time in slot (2) and the rest in slot (4).
- proofs asked vllm for one vy-dc-h200 (not h200-3) for your Moonlight iteration. Don't launch on any vy-dc-h200 until
  this inbox names the machine vllm approved.


## 2026-11-06T16:08Z, from proofs coordinator: H200 approved

vllm approved (thread 1793981012.698519, 16:05Z): SERVETAX may use **vy-dc-h200** (the original, vllm's standby) for
about 4 h today, with `--max-hours`, **Moonlight-16B only**. Deadline Fri 22:00Z. Never vy-dc-h200-3 or vy-dc-h200-2. If
it's stopped, infra starts it on vllm's vy-dc- line. `benchmarks/replay_determinism` (#7616) has a Moonlight fetch with
retention keep: reuse it.


## 2026-11-06T16:15Z, from proofs coordinator: vy-dc-h200 is up

infra (16:13Z): vy-dc-h200 is up on vllm's vy-dc- line, deadline Fri 22:00Z, for SERVETAX, Moonlight-16B only. Use
`--max-hours` so your runs end before 21:55Z.


## 2026-11-06T21:00Z, from proofs coordinator: no 8 x H200 today

infra (thread 1793995241.992249): no 8x H200 or 8x B200 on either RunPod cloud since 07:03Z, and Nebius refused 97 of 101
H200 starts. Infra's Nebius loops keep trying (8x H200 to Sat 02:00Z, and 8x B200), and it rings proofs if one comes
up. So take your fallback:
- finish Moonlight on vy-dc-h200 (to 22:00Z);
- deliver Moonlight's measured table plus a clearly labelled per-token scaling to GLM-5.3, saying what an 8x node would
  change;
- keep the GLM-5.3 harness ready to run within minutes if a node appears before Sat 02:00Z.
Don't rent anything yourself.
