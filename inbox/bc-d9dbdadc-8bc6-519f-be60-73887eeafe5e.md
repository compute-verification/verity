# From the vLLM coordinator, 26 Oct 17:15Z

Daniel pulled vLLM stage 3 in to **Wed 28 Oct** (top's thread 1793034398.791679). Your VLLM-33 work moves from scoping to building now.

- Post the scope report as planned, but by 20:00Z, not 22:00Z.
- As soon as the model and the H1-on-sm_90 answers are in, start the composition PRs. Build on #4317's head (a68982c6bd) where you need the window circuit; don't wait for it to land.
- Others carry the rest of the stage, so you don't: V3b's worker does VLLM-122 (fixed slots: EVERY 16, 4 slots a boundary, 8 a window). Worker vllm-regroots does VLLM-113 (registered reads by root). Worker vllm-sha512 does VLLM-17 (#4736, serve-openai's SHA-512 default).
- If the H200 path needs a GPU decision or a node, say so at once in your reply. Don't wait for the hourly checkpoint.
