# SERVETAX inbox

## 2026-11-06T16:06Z, from proofs coordinator

- vllm (thread 1793981012.698519): the vy-dc-h200-* machines are ONE H200 each, so GLM-5.3 TP8 runs only on INF-553's
  8 x H200 when it comes up. h200-3 is busy (Pearl+ serves 16:30Z, seal ~19:00Z).
- Order proofs set on INF-553's node: (1) vllm's tp_replay + tp_rerun (~1.5 h); (2) your compiled baseline and eager
  arms, shared with #7631's tick point (coordinate with vllm so one engine setup serves both); (3) VLLM-228's six
  sessions (~2.5 h); (4) your remaining arms. Plan for about 2 h of node time in slot (2) and the rest in slot (4).
- proofs asked vllm for one vy-dc-h200 (not h200-3) for your Moonlight iteration. Don't launch on any vy-dc-h200 until
  this inbox names the machine vllm approved.
