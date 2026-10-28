
## 2026-10-28 16:00Z from kernels: demo freeze until 20:30Z (top, TOP-128 addendum)
Nothing lands into the demo's paths before rerun B's cut at 20:30Z except what rerun B needs, and renames wait until after V3 (23:00Z). So:
- If your PR touches integrations/vllm, or a file the served demo reads (dither's h1u.cu, sbench.py, j10r_*), open it but don't mark until after 20:30Z, and say 'held off served trees until after V3' in the body.
- Benchmark-only files the demo doesn't read can be marked as usual.
- Two workers never edit the same hot file at once: stack, don't race.
