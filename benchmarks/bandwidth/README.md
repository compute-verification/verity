# Device ceilings for PoUB

PoUB's timing (`verity/protocols/resource_accounting/memory/poub/verifier/timing.py`) cites a few hardware bounds: how
fast a node's writers can fill a tier with labels, and how much a card holds on chip. These probes measure them. Each
probe is one Attempt on a GPU node, and its records are its result: no code reduces them to one number.

## Try it

`uv run tools/check/suites.py benchmarks/bandwidth` tests the runner, on any machine. On a node, as the command of a
`research run`, `verity bench bandwidth chip|write_gpu|write_host [ARGS]` (`run.sh`) runs one probe's script:

| Probe | What it measures | What it writes under `$OUT` |
| --- | --- | --- |
| `write_gpu` | The write ceilings' GPU half (`timing.GPU_WRITE_BYTES_PER_S`), on one card | `write.jsonl` |
| `write_host` | The write ceilings' host half (`timing.HOST_WRITE_BYTES_PER_S`), on the host's cores | `write_host.jsonl` |
| `chip` | What an RTX PRO 6000 holds on chip (`timing.GPU_CHIP_BYTES`) | `chip.json` and `ur.jsonl` |

Each script's header lists its files, beside the build's and the device's logs. `node.py` reads what a node file
(`hardware/nodes/`) states and no card states: its interfaces off the node, its disks' links, and what each card's
firmware says it gives a program; its docstring has the command line.

[`DESIGN.md`](DESIGN.md) says what each probe measures and how it is gated.
