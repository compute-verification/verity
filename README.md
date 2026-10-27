# Verity

How to deploy the demos: the hardware each one needs and the command that runs it. Each demo's own README has the rest.

## Setup

~~~sh
uv sync --all-packages --extra torch-cpu    # a CPU machine; a GPU machine takes --extra torch
verity --help
~~~

You also need Rust (C-Flock's prover), Lean through elan (the C-Flock verifier and the protocol checks), and CUDA on a GPU
machine. On our machines every run goes through `research run` (`AGENTS.md`).

## Demos

| Demo | What it serves and audits | Command | Details |
| --- | --- | --- | --- |
| The toy | A two-layer W8A8 MLP on CPU. Every window is audited by each subprotocol, and the verdicts are composed. | `verity bench all-protocols` | `benchmarks/all_protocols/README.md` |
| The H200 demo | The frozen MQA shape on one H200. Every unit's values are committed, and the P2-encoded weights fill the card's memory. PoUS, the bandwidth audit, the network certifier and the wipe run. Live coins draw units, which C-Flock `--zk` proves and the Lean verifier checks. | `verity bench e2e_h200` | `benchmarks/e2e_h200/README.md`, `benchmarks/e2e_demo/README.md` |
| vLLM | Llama-3.2-1B served by vLLM through the committing endpoint. A served window is audited by every subprotocol. | `verity example vllm` | `examples/vllm/README.md` |

`verity example --status` says how far each example has got: runs, placed, stated, proved or measured
(`examples/README.md`).

## Hardware

The prover's side serves and proves. The auditor's side runs the verifier, the challenger that draws the live coins, the
network certifier and the memory challenger. Those must run on machines the prover doesn't control. A demo counts as
placed only when they do.

| Demo | Prover's side | Auditor's side |
| --- | --- | --- |
| The toy | One Linux CPU machine (the run of record used 96 GB of memory) | The same machine (a development run) |
| The H200 demo | One NVIDIA H200 (sm_90a, CUDA 13) to serve, and a CPU machine to prove the drawn units | A CPU host in the H200's region for the challenger and the memory challenger, since PoUS's timed reads need it close; the network certifier on the server's link; the Lean verifier on any CPU machine |
| vLLM | One NVIDIA RTX PRO 6000 (sm_120, CUDA 12.9) with 256 GB of host memory; the prover and the Lean verifier run there today | A CPU host for the challenger |
