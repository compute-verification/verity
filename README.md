# Verity

Verity is a system by which an AI developer can make verifiable claims to external auditors about how it is using its
datacenters, without revealing its algorithms or data. It represents the datacenter's computations exactly as Boolean
circuits, proves claims about them, and derives compute, memory and network accounting from those proofs.

It has two parties. The developer runs computations in its datacenters, keeps private records of them and proves
properties of those records. The auditor checks the proofs with its verifier, whose online half is the challenger, and
with its attesters, the auditor's own hardware at the site: the network certifier and the memory challenger.

The protocols come in layers, each built on the ones before it: the proofs (Flock, the proof system; SPOT, which samples
what to prove; recursion), resource accounting by resource (compute: PoUW; memory: PoUS and PoUB; network: the network
certifier), workload classification, and the two deployment patterns, remote and onsite. Registration, a deployment's
registration log and its check, runs once before any challenge and sits beside the layers.

## Quick start

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). The Lean parts need elan (each Lean package pins its
toolchain in `lean-toolchain`); Flock's prover needs Rust.

~~~sh
uv sync --all-packages --extra torch-cpu               # the whole workspace, editable, with CPU-only torch
uv run verity --help                                   # the one command and its subcommands
uv run verity run demos/onsite_full.toml --dry-run     # onsite's parties and channels, resolved without a GPU
uv run tools/check/suites.py --quick                   # the test suites whose inputs changed, slow tests left out
~~~

## Read next

- [`ARCHITECTURE.md`](ARCHITECTURE.md): the system, its parts and their interfaces, the repository's layout, and the
  Glossary.
- [`AGENTS.md`](AGENTS.md): what holds on `main` now, the map of the tree, the design rules, and how a change is tested,
  checked and landed.
