# Hardware

The hardware description's capacities: what each GPU, host, server and time base the parties agree on can do, every
number with its source. Every guarantee assumes the real hardware is within these files, and the calibration probes
measure devices against them.

Not yet: an application's file names no hardware description, and four sets of helpers besides the kit wrap `load_card`.

## Pieces

| Piece | What it is | Where |
|---|---|---|
| Card | One SKU's figures (counts, clocks, stores, rates, weights), each an entry `{value, unit, mark, source}` | `cards/<sku>.toml` |
| Node file | One kind of node: its GPU card and count, its host card, its memory leaves and ways out | `nodes/<node>.toml` |
| Architecture file | The compile target a GPU card's `arch` names: capability, virtual architecture, toolkit gates | `architectures/<arch>.toml` |
| Reader | The one loader and checker of all three, and a card's rate at a clock reading | `verity/device.py` |
| Kit | A running device's reported name to its card, its short name and its compile target | `benchmarks/harness/verity_bench/kit.py` |
| Probes | Programs that time a device against its card; a run above a ceiling refutes it | `calibration/tensor_duty`, `calibration/native_peak`, `calibration/pouw` |
| Device model | Prices instruction classes and designs on a card's rows | `calibration/machine_model` |

## What varies

The device. Each cell is served (an application runs on it), exists, or never written.

| GPU card | Card | Node file | Architecture | Device model | Probe point | Application |
|---|---|---|---|---|---|---|
| `h100-sxm-80gb` | exists | exists (one card) | `sm_90a` | never written | exists | served |
| `h200-sxm-141gb` | exists | exists (one card) | `sm_90a` | exists | exists | served |
| `rtx-pro-6000-server` | exists | exists (eight cards) | `sm_120a` | exists | exists | exists |
| `b200-sxm-180gb` | exists | never written | `sm_100a` | exists | never written | never written |

A new device supplies: one card with every section of its kind; an architecture file if its `arch` is new; a node file
for each kind of node it sits in. No code changes.

Fixed for every device: the entry form and its four marks (`verity/device.py`'s docstring defines them); a `ceiling` is
a bound no program on that SKU exceeds, and a figure stated at a clock reading names the reading.

## Interfaces

- From the workload: nothing.
- From `run`: the name a device's driver reports (`gpu_card_reporting`), or a node's kind (`load_node`).
- It gives: to accounting, a device's capacities as entries (`load_card`, `rate(card, fmt, reading)`); to a build, the
  compile target of a card's architecture (`kit.card_gencode`, `load_architecture`); to the device model and the
  probes, a card's rows and ceilings, read at run time and never copied.

## Try it

~~~python
from pathlib import Path
from verity.device import load_card, number, skus
cards = Path("calibration/hardware/cards")
skus("gpu", directory=cards)   # ['b200-sxm-180gb', 'h100-sxm-80gb', 'h200-sxm-141gb', 'rtx-pro-6000-server']
number(load_card("h200-sxm-141gb", directory=cards)["sm"]["count"])   # Fraction(132, 1)
~~~

In `uv run python` from a clone, each line gives what its comment shows.
