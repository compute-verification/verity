# Hardware

The data sheets of the hardware Verity accounts for, and nothing else: TOML, no code. Each GPU, host, server model
and time base has one card in `cards/`, the one source of that card's numbers. Each kind of node, a host and the GPUs
it carries, has one file in `nodes/`: what a node's registration states that no card does. The Site's module,
`verity.device`, reads both.

## Try it

~~~python
from verity.device import load_card, number, skus
skus("host")                                    # ['xeon-6776p-2s', 'xeon-platinum-8468-2s']
load_card("xeon-6776p-2s")["rate"]["ops"]       # {'value': 1045299200000000.0, 'unit': 'op/s', 'mark': 'ceiling', ...}
(drift := load_card("nebius-vm-monotonic-raw")["drift"]["max"])   # {'value': 0.0005, 'unit': 's/s', 'mark': 'premise', ...}
number(drift)                                   # Fraction(1, 2000)
~~~

In `uv run python` from a clone, each line gives what its comment shows. A number is an entry
`{ value, unit, mark, source }`:

- `mark` is `ceiling` (no program on that SKU exceeds it, whatever firmware it runs: what the model's premises need),
  `measurement` (a measured value, so only a lower bound on the ceiling), `premise` (no page and no run bounds the
  number; it is assumed under the premise its source states, of whom it names. A row that uses it prints the premise
  for its device's host, and no figure of record uses it) or `placeholder` (none of these: open);
- `source` names a data sheet, a document or an `art:` id (a run's evidence: its record, `research data show RUN`'s
  `outputs.run_record`), and a node by its kind, never its host name.

## The cards

| SKU | kind | file |
|---|---|---|
| RTX PRO 6000 Blackwell Server Edition (board 900-2G153-0000-000) | gpu | `calibration/hardware/cards/rtx-pro-6000-server.toml` |
| H200 SXM 141 GB (board 692-2G520-0280-001) | gpu | `calibration/hardware/cards/h200-sxm-141gb.toml` |
| B200 SXM 180 GB (board 692-2G525-0220-000) | gpu | `calibration/hardware/cards/b200-sxm-180gb.toml` |
| Two Xeon 6776P, the RTX node's host | host | `calibration/hardware/cards/xeon-6776p-2s.toml` |
| Two Xeon Platinum 8468, the H200 nodes' host | host | `calibration/hardware/cards/xeon-platinum-8468-2s.toml` |
| NVIDIA DGX H200, a server none of our nodes is | server | `calibration/hardware/cards/dgx-h200.toml` |
| `CLOCK_MONOTONIC_RAW` in a Nebius virtual machine, a KVM guest on `kvm-clock` or `tsc` | timebase | `calibration/hardware/cards/nebius-vm-monotonic-raw.toml` |

A new SKU is a new card file. Every card of a kind has the same sections and entries.

**The device's own report.** A ceiling is the hardware's, never its firmware's. A few entries marked `ceiling` are
today what the device reports under its vBIOS, with no document behind them: `clock.sm` on the RTX PRO 6000 and the
B200, `clock.memory` on all three GPU cards and `sm.count` on the B200, and with them what a card computes from one
(`clock.tensor` on those two cards, `memory.bandwidth` on all three). `store.dram` on the H200 and the B200 is of that
kind too: what the shipped firmware gives a program, under the board's `store.dram_installed`. Each bounds a card only
while it is the declared board and vBIOS, which the named assumptions `accounting-declared-device` and
`accounting-sheet-clock` state and no mark does. A host card's `store.l2`, `l1d` and `l1i` are
likewise its guest's CPUID.

## The node files

| Node | file | GPU card | host card | the host's own leaves | its ways out |
|---|---|---|---|---|---|
| An RTX PRO 6000 node, eight cards | `calibration/hardware/nodes/rtx-pro-6000-server-x8.toml` | `rtx-pro-6000-server` | `xeon-6776p-2s` | `dram`, and the disks `vda`, `vdb` and `vdc` | `eth0`, at its line rate, and the storage path |
| An H200 node, one card | `calibration/hardware/nodes/h200-sxm-141gb-x1.toml` | `h200-sxm-141gb` | `xeon-platinum-8468-2s` | `dram`, and the disks `vda`, `vdb` and `vdc` | `eth0`, which reads no rate, and the storage path |

A node file names no machine, no address and no provider: it is the kind's, read on one node of it. Its numbers are
entries under the cards' rules.
