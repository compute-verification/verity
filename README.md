# Verity

Verity represents AI datacenter computations exactly as Boolean circuits, proves claims about them, and derives
compute, memory and network accounting from those proofs.

It has two parties. The developer runs computations in its datacenters, keeps private records of them and proves
properties of those records. The auditor checks the proofs, with a challenger online, a verifier offline, and its own
devices at the site.

How far each demo runs and is proved today is in [`demos/README.md`](demos/README.md). What each guarantee states, and
what it rests on, is in [`security/README.md`](security/README.md).

## Quick start

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). The Lean parts need elan (each Lean package pins its
toolchain in `lean-toolchain`); Flock's prover needs Rust.

~~~sh
uv sync --all-packages --extra torch-cpu               # the whole workspace, editable, with CPU-only torch
uv run verity --help                                   # the one command and its subcommands
uv run verity run demos/onsite_full.toml --dry-run     # onsite's parties and channels, resolved without a GPU
uv run tools/check/suites.py --quick                   # the test suites whose inputs changed, slow tests left out
~~~

Build a circuit, evaluate it and name it by its digest:

~~~python
from verity.circuits import Program
from verity.circuits.codec import program_digest
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.circuits.ops.gemm import GEMM_AMPERE, Gemm

gemm = bind(Gemm, K=64, N=8, **GEMM_AMPERE)   # x[64] @ W[8, 64]^T in BF16, as an A100's tensor cores compute it
x, W = [0x3F80] * 64, [0x3F80] * (8 * 64)     # BF16 encodings of 1.0, W flattened row by row
evaluate(gemm, x, W)                          # eight BF16 words 0x4280, that is 64.0
program_digest(Program(gemm))                 # the circuit's name: its SHA-512 digest
~~~

## The repository

| Directory | What it holds |
| --- | --- |
| [`verity/`](verity/README.md) | The distribution `verity`: the circuit language, commitments, the protocols, and the runtime that runs their parties. |
| [`security/`](security/README.md) | The Lean statement and proof of every guarantee. |
| [`adapters/`](adapters/vllm/README.md) | One directory per serving system, holding only what attaches Verity to it. |
| [`hardware/`](hardware/README.md) | One TOML data sheet per GPU, host or server model. Data only. |
| [`demos/`](demos/README.md) | One deployment configuration per demo, which `verity run` reads. A demo holds no code. |
| [`benchmarks/`](benchmarks/README.md) | Measurements, each run by `verity bench NAME`. A benchmark never decides a verdict. |
| `tools/` | The [`verity` command](tools/verity/cli/README.md), [the check](tools/check/README.md), the Lean check and the circuit checker. |
| [`tests/`](tests/README.md) | The invariants of the whole tree. Each package's own tests are in its `tests/`. |
| [`infra/`](infra/README.md) | The team's own machines and services. Nothing in `verity/` reads it. |

## Read next

- [`ARCHITECTURE.md`](ARCHITECTURE.md): the system and its parts. Section 4 is each part's interface, section 6 the
  repository's layout, section 7 where `main` still differs from it, and the Glossary is at its end.
- [`GLOSSARY.md`](GLOSSARY.md): the notation every guarantee is stated in, and the older names of each term.
- [`AGENTS.md`](AGENTS.md): what holds on `main` now, the design rules, and how a change is tested, checked and landed.
  Read it before you change anything.
- The specifications: the [program format](verity/circuits/PROTOCOL.md) and the
  [frame-v3 commitment](verity/commitments/frame_v3/PROTOCOL.md).
