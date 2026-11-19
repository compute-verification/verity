# Device ceilings for PoUB

PoUB's timing (`verity/protocols/resource_accounting/memory/poub/verifier/timing.py`) cites a few hardware bounds: how
fast a node's writers can fill a tier with labels, and how much a card holds on chip. These probes measure them. Each
probe is one Attempt on a GPU node, and writes its records and the run's `result.json` through the harness's kit
(`result.py`): `write_host` its fastest record as `metrics.toml`'s number, `chip` and `write_gpu` their device program's
check as the validation, with no number.

## Try it

`uv run tools/check/suites.py calibration/bandwidth` tests the runner, on any machine. On a node, as the command of a
`research run`, `verity calibrate bandwidth chip|write_gpu|write_host` (`run.sh`) runs one probe:

| Probe | What it measures | What it writes under `$OUT` |
| --- | --- | --- |
| `write_gpu` | The write ceilings' GPU half (`timing.Figures.gpu_write_bytes_per_s`), on one card | `write.jsonl` |
| `write_host` | The write ceilings' host half (`timing.Figures.host_write_bytes_per_s`), on the host's cores | `write_host.jsonl` |
| `chip` | What an RTX PRO 6000 holds on chip (`timing.Figures.gpu_chip_bytes`, the card's `store.chip`) | `chip.json` and `ur.jsonl` |

`run.sh`'s header lists each probe's files, beside the build's and the device's logs. `node.py` reads what a node file
(`calibration/hardware/nodes/`) states and no card states: its interfaces off the node, its disks' links, and what each card's
firmware says it gives a program; its docstring has the command line.
