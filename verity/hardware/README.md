# The card files (`verity.hardware`)

One TOML file per SKU in `cards/`, the one source of that card's numbers. A card is a GPU's (`kind = "gpu"`) or a
host's (`kind = "host"`: a server's CPUs and DRAM, apart from the GPUs it carries, so the same GPU in another host needs
no second card). Each number is an entry `{ value, unit, mark, source }`:

- `mark` is `ceiling` (no program on that SKU and firmware exceeds it: what the model's premises need), `measurement`
  (a measured value, so only a lower bound on the ceiling) or `placeholder` (neither: open);
- `source` names a data sheet, a document or an `art:` id (a run's evidence: its record, `research data show RUN`'s
  `outputs.run_record`), and a node by its kind, never its host name.

| SKU | kind | file | Lean table |
|---|---|---|---|
| RTX PRO 6000 Blackwell Server Edition (board 900-2G153-0000-000) | gpu | `cards/rtx-pro-6000-server.toml` | `Security/Definitions/Core/Accel/Sm120.lean` |
| H200 SXM 141 GB (board 692-2G520-0280-001) | gpu | `cards/h200-sxm-141gb.toml` | `Security/Definitions/Core/Accel/H200.lean` |
| B200 SXM 180 GB (board 692-2G525-0220-000) | gpu | `cards/b200-sxm-180gb.toml` | none yet: `Security/Definitions/Accounting/Sheets/B200.lean` is held to it |
| Two Xeon 6776P, the RTX node's host | host | `cards/xeon-6776p-2s.toml` | none |
| Two Xeon Platinum 8468, the H200 nodes' host | host | `cards/xeon-platinum-8468-2s.toml` | none |

Every card of a kind has the same sections. A GPU card's are `sm`, `clock`, `power`, `store` (with `chip`, all on-chip
storage), `rate` (each resource's units a SM-clock), `memory` (device memory's bus, transfers, bandwidth and measured
read), `hash` (SHA-512 compressions a second), `squaring` (the squaring ceiling at each codec width), `weight` (the
tensor units a MAC of each format uses) and `model` (`Accel.Card`'s `exact` and code space). A host card's are `cpu`
(sockets and cores), `clock`, `rate` (AMX MACs and the host's op peak), `memory` (DRAM's channels, speed, bandwidth,
measured read and write ceiling), `store` (its caches and allowances, and `chip`, the host's on-chip bytes in S_chip)
and `link` (`window`, what the boundary link lands in host memory at once beyond its rate).

`node.py` reads a node's figures off its cards, the value of the Site's `node h`: `node(devices, host, placeholders)`
sums its devices' GPU cards and its host card (`node(*VY_NEBIUS)`, a vy-nebius node's eight RTX PRO 6000s on two Xeon
6776Ps, is the remote bandwidth audit's). It refuses a node that reads a `placeholder` its caller doesn't name, and
carries the ones it does name: vy-nebius is read with its host DRAM's ceiling open (DDR5-6400, its DIMMs not on
record), and the H200's and B200's reads, chip storage, SHA-512 and squaring rates are not measured, so no node of
them is read yet.

~~~python
from verity.hardware import load, skus
skus("host")                                    # ['xeon-6776p-2s', 'xeon-platinum-8468-2s']
load("xeon-6776p-2s")["rate"]["ops"]            # {'value': 1045299200000000.0, 'unit': 'op/s', 'mark': 'ceiling', ...}
~~~

`load(sku) -> dict` returns the parsed file and refuses a number without its unit, mark and source, and a card without
its kind. `skus(kind=None)` lists the cards, or those of one kind.

**Changing a number:** change the card file, then its Lean table to match. `tests/test_lean.py` evaluates each Lean
`card` arm as an exact rational and fails on any number that differs, and on a store, resource or format of
`Security/Definitions/Core/Accel/Isa.lean` that either side lacks. A new SKU is a new card file and its own Lean table.
