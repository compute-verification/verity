# The card files (`verity.hardware`)

One TOML file per SKU in `cards/`, the one source of that card's numbers. Each number is an entry
`{ value, unit, mark, source }`:

- `mark` is `ceiling` (no program on that SKU and firmware exceeds it: what the model's premises need), `measurement`
  (a measured value, so only a lower bound on the ceiling) or `placeholder` (neither: open);
- `source` names a data sheet, a document, a run id or an `art:` id.

| SKU | file | Lean table |
|---|---|---|
| RTX PRO 6000 Blackwell Server Edition (board 900-2G153-0000-000) | `cards/rtx-pro-6000-server.toml` | `Security/Definitions/Core/Accel/Sm120.lean` |
| H200 SXM 141 GB (board 692-2G520-0280-001) | `cards/h200-sxm-141gb.toml` | `Security/Definitions/Core/Accel/H200.lean` |

Every card has the same sections: `sm`, `clock`, `power`, `store` (with `chip`, all on-chip storage), `rate` (each
resource's units a SM-clock), `memory` (device memory's bus, transfers, bandwidth and measured read), `hash` (SHA-512
compressions a second), `squaring` (the squaring ceiling at each codec width), `weight` (the tensor units a MAC of each
format uses), `model` (`Accel.Card`'s `exact` and code space) and `host` (the node's CPUs, and its DRAM's read and
bandwidth, on-chip storage and link window).

`node.py` reads a node's figures off its cards, the value of the Site's `node h`: `node(devices, host)` sums its
devices' and its host's (`node(*VY_NEBIUS)`, a vy-nebius node's eight RTX PRO 6000s and their host, is the remote
bandwidth audit's). It refuses a node while any figure it reads is a `placeholder`: the H200 card's reads, chip
storage, SHA-512 and squaring rates are not measured, so no H200 node is read yet.

~~~python
from verity.hardware import load, skus
skus()                                          # ['h200-sxm-141gb', 'rtx-pro-6000-server']
load("rtx-pro-6000-server")["host"]["ops"]      # {'value': 1045299200000000.0, 'unit': 'op/s', 'mark': 'ceiling', ...}
~~~

`load(sku) -> dict` returns the parsed file and refuses a number without its unit, mark and source.

**Changing a number:** change the card file, then its Lean table to match. `tests/test_lean.py` evaluates each Lean
`card` arm as an exact rational and fails on any number that differs, and on a store, resource or format of
`Security/Definitions/Core/Accel/Isa.lean` that either side lacks. A new SKU is a new card file and its own Lean table.
