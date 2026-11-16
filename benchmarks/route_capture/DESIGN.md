# Route capture: design

How the capture serves GLM-5.3, what a step's class is, and what the capture keeps. `README.md` says what the
capture is for and how to run it.

## Serving

`run.sh` serves GLM-5.3 on eight H200s at TP8 the way serving does: `vllm serve` at vLLM's defaults (CUDA graphs,
chunked prefill, prefix caching, async scheduling, the model runner vLLM picks), at most 256 sequences and 8,192 tokens a
step, prompts up to 32k tokens. At the integration's vLLM pin, GLM-5.3's architectures are among those vLLM serves with
breakable CUDA graphs (`DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`), which turn torch.compile off: the model runs vLLM's
custom CUDA kernels, captured whole into graphs that break around attention, which runs eagerly. Four `vllm bench serve` clients run for about `DURATION_S`
(600 s) at once:

| Client | Prompts | Sampling | Arrivals |
| --- | --- | --- | --- |
| `sharegpt-greedy` | ShareGPT | greedy | Poisson, ramped linearly from 1 to 24 requests/s |
| `sharegpt-sampled` | ShareGPT | temperature 0.7, top-p 0.95 | Poisson, 2/s |
| `long-context` | random, 8,192 to 32,768 tokens | temperature 0.7, top-p 0.95 | Poisson, 0.1/s |
| `shared-prefix` | 16 shared 4,096-token prefixes, each with a 512-token suffix | greedy | Poisson, 1/s |

`run.sh`'s header lists its variables (`VENV`, `CKPT`, `SHAREGPT`, `DURATION_S`). The engine runs with the Inductor pin
the integration serves with (`TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0`), so for a model vLLM compiles, the fused kernels
captured are the ones a replay runs. vLLM takes FlashInfer's block-FP8 GEMM below 32 rows only when FlashInfer's kernels
are prebuilt (`flashinfer-cubin`, which serving images ship) or `nvcc` is on PATH to compile them, and FlashInfer's
sampler compiles at warmup with that `nvcc`, so `run.sh` puts `CUDA_HOME`'s `nvcc` on PATH, and `engine_env.txt` records
`has_flashinfer()`. The FlashInfer sampler stays on, vLLM's default; if a sampled smoke request fails with it, the server
restarts with it off, and `engine_env.txt` says so.

## A step's class

A step's class (`key`) is its phase mix (decode, prefill or mixed), its token count rounded up to `ROUTECAP_M_EDGES` (which
hold every M where a GLM-5.3 route switches), its request count rounded up to a power of two, the prefills' attention route
(dense while the longest prefill sequence is at most the indexer's top-k, sparse past it), whether a new request hit the
prefix cache, and the sampler's mix. A rank profiles the first `ROUTECAP_PER_CLASS` steps of each class, at most
`ROUTECAP_BUDGET` in all, and past its first `ROUTECAP_WARM` only while profiling has held it for at most `ROUTECAP_DUTY`
of the run. A profiled step synchronizes the device at its start and end, so the steps around it batch a little more
than they would have. Every step gets a row; the kernels of an unprofiled step are those of its class.

