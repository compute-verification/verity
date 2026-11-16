# verity-vllm

vLLM serves the model. This adapter makes that serving something Verity can audit.

Verity knows only circuits and values. It has no notion of a model, a request or a batch. The adapter is the one place
that knows vLLM, and it does two things:

- **It compiles what vLLM serves into circuits.** Before serving, it derives from the model and its serving settings
  the circuits that say, bit for bit, what the GPU will compute.
- **It commits to what vLLM computes.** While vLLM serves, it commits to every value a proof may need, at the point
  where vLLM computes it, and seals the commitments for a later audit.

After serving, an audit samples from what was sealed, and the developer proves the sampled pieces against the circuits.

You serve as you would with vLLM: the same engine, pinned to one version, behind an OpenAI-compatible endpoint or as a
library.

## Quickstart

Serve one request on a machine of your own, with every value committed, and seal it. The machine needs:

- an H100 or H200, with an NVIDIA driver that runs CUDA 12.9;
- the CUDA 12.9 toolkit and a C++ compiler;
- `uv` and `curl`;
- 35 GB of free disk;
- network access to GitHub, PyPI, PyTorch's downloads, Hugging Face and the store's public site. It needs no token or
  key.

~~~sh
git clone --filter=blob:none https://github.com/compute-verification/verity && cd verity
bash adapters/vllm/verity_vllm/ops/serve_fresh.sh --dry-run    # show every path and step, and do nothing
bash adapters/vllm/verity_vllm/ops/serve_fresh.sh
~~~

The script builds the environment, fetches Llama 3.2 1B and makes its INT8 checkpoint, serves one request, and seals
it. Its last line says that it served and sealed, or which step failed.

## Usage

As a library, the class is constructed and called like `vllm.LLM`:

~~~python
from vllm import SamplingParams
from verity_vllm import LLM

llm = LLM("unsloth/Llama-3.2-1B", revision="9535bd9b1d1dea6acafbdc4813b728796aeb28da")
outputs = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=16))
~~~

The model and its revision must be among the adapter's pinned checkpoints. The class's docstring documents each
option it takes.

As a server, `verity-vllm serve-openai` starts the committing endpoint. `verity-vllm --help` lists every other
command.

## What is inside

The package `verity_vllm`, by what each part does:

| Part | What it does |
| --- | --- |
| `engine` | The pinned vLLM engine. |
| `program`, `definitions` | The compiler: the served model as circuits, and the circuits of the kernels it runs. |
| `acquire`, `commit` | Commits values while vLLM serves. |
| `check` | Checks that what was served is what its circuits say. |
| `protocols` | What each protocol needs on the engine: SPOT, PoUW, PoUS and PoUB. |
| `pipeline` | The endpoint and the command line. |

## Development

~~~sh
uv sync --all-packages --extra torch-cpu
~~~

This installs the adapter with CPU-only torch. vLLM itself serves only on a GPU node.

## Read next

- [`DESIGN.md`](DESIGN.md): how serving, the endpoint and replay work, in detail, and the vocabulary the code uses.
- [`tools/served_run/README.md`](tools/served_run/README.md): the served run, from serving to every protocol's verdict.
