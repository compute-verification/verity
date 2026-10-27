# benchmarks/hash

Prover-side GPU code for the served path's commitments and for PoUW's served linear, each held bit for bit to its CPU
reference or Definition before anything is timed. The commitments' hash is SHA-512 with its whole output (hm96 leaves in
frame-v3-sha512 trees, "One hash" in AGENTS.md); nothing here changes a committed format, and nothing a verifier runs
reads this code.

| Directory | What it holds |
|---|---|
| `lib/` | libvc, the served path's SHA-512 commitments on the GPU (`vc.h`, `vclib.py`), held to the CPU references by `conformance.py`; its README has the API, the contract and how to build and run it |
| `dither/` | PoUW's H1 receipts served in compiled vLLM: the one-stream linear (`h1o.cu`) whose noised MMAs serve y, with the peel z = A Bᵀ from the receipted words (H1V3); the A tables (`h1c.cu`); J10-R's per-call-noise units (`h1u.cu`, `j10r*.py`); each kernel gated against the credit's H1V3 reference (`h1ogate.py`), and the served overhead against stock vLLM int8, paired rep by rep (`sbench.py`, run on a leased GPU by `sbench.sh`) |
| `exactlin/` | the int8-exact linear as two Triton kernels (`exactlin.py`: the exact quantizer, and the GEMM with the exact epilogue and per-card configs), gated bit-exact against `int8_exact` (`kgate.py`) and the int8 Definitions' replay evaluators (`replay.py`), timed against stock vLLM int8 at kernel level (`kbench.py`, `ksweep.py`, `zybench.py`) and served (`served.sh`, through `dither/sbench.py`'s exact-fused arm, after the kernel gate), with the served y checked bit for bit by the vLLM integration's `int8_exact_gpu` (`values.py`), the per-step kernel table (`steps.py`) and the share of a served step PoUW doesn't credit (`share.py`) |
| `maxtput/build_job.sh` | a Build of record for a model that has none: the served match alone, on one RTX PRO 6000 |

Each script's header says how to run it, from the source tree's root, and on what hardware. They share one layout: a
run writes its results to `$RESEARCH_RUN_DIR/out` (`OUT` where a header names it) and reads the files it is given, a
build's libraries or a source archive, from `$RESEARCH_RUN_DIR/inputs` or the directory a header's variable names. GPU
work runs under a GPU lease (`gpu-lease`, owned by `GPU_LEASE_WHO`).
