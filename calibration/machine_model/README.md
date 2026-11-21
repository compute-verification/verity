# The ACCEL model of each measured card

`machine-model` prices every warp instruction class, and useful work's designs, on the measured ceilings of a card,
and reports a design's ε: an accepted window's node performed at least 1 − ε of the work the window credits
(`data/headline.toml`'s `scope` says what a per-GPU reading needs). It measures nothing and needs no GPU: its inputs
are the card's figures and the probes' censuses, as data.

Not yet: a census arrives by hand, with no command that takes a probe's output into `data/census/`.

## Pieces

| Piece | What it is | Where |
|---|---|---|
| Device | One card's model: the card it names, its rows (ceilings), its sheet lines | `data/model.toml`, `data/devices/<device>/` |
| Census | Where each instruction class lands on the rows, one file an epoch | `data/census/` |
| Vectors | Published price vectors, each re-solved and checked | `data/vectors/` |
| Attacks | Programs priced against a vector | `data/attacks/` |
| Designs | A design's instruction counts, and the honest profile it is compared with | `data/designs/`, `data/honest/` |
| Opcode table | The evidence that every opcode a card runs is priced | `data/cover/` |
| Commands | `solve`, `eps`, `attacks`, `cover`, `headline`, `lean-headline`, `sheet` | `verity_machine_model/commands/` |
| Metric | Each design's ε as one registered point | `run.py`, `metrics.toml` |

## What varies

The device. Each cell is exists or never written; no application reads the model.

| Device | Card | Census and vectors | Designs | Lean sheet | Lean headline |
|---|---|---|---|---|---|
| `sm_120` | `rtx-pro-6000-server` | exists | exists | never written | exists |
| `sm_90` | `h200-sxm-141gb` | exists | exists | exists | exists |
| `sm_100` | `b200-sxm-180gb` | exists | exists | exists | exists |
| none | `h100-sxm-80gb` | never written | never written | never written | never written |

A new device supplies: a directory with a `model.toml` naming its card, its rows and census from the probes, and a
`sheet.toml`. No code changes.

Fixed for every device: the linear program, the commands, and that a card's figure is read from the card when a
command runs (a `card = ` line), never copied into the model's data.

## Interfaces

- From the workload: nothing.
- From `run`: nothing; it takes a card's rates, clocks and stores from the hardware data sheets
  (`verity.device.load_card`) and a census of instruction classes on a device from the probes, as a file.
- It gives: to useful work's accounting, a design's ε (`eps`, `headline`) and the generated Lean that states a card's
  sheet and a design's headline (`sheet --lean`, `lean-headline`), each held current by a test; to the dashboard, one
  metric, each design's ε.

## Try it

~~~sh
uv run machine-model --card rtx-pro-6000-server solve
~~~

It re-solves that card's current price vector and prints its τ, margin and cheapest attacks beside the numbers it was
published with; it exits 1 if the vector fails its check. `uv run machine-model` alone prints every command's usage.
