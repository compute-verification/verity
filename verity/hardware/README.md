# The card files (`verity.hardware.cards`)

One TOML file per SKU in `cards/`, beside their loader (`cards/__init__.py`), the one source of that card's numbers. A
card is a GPU's (`kind = "gpu"`), a host's (`kind = "host"`: a server's CPUs and DRAM, apart from the GPUs it carries,
so the same GPU in another host needs no second card) or a server's (`kind = "server"`: what a server model's own
documents state of it apart from those, its network ports). Each number is an entry `{ value, unit, mark, source }`:

- `mark` is `ceiling` (no program on that SKU exceeds it, whatever firmware it runs: what the model's premises need),
  `measurement` (a measured value, so only a lower bound on the ceiling) or `placeholder` (neither: open);
- `source` names a data sheet, a document or an `art:` id (a run's evidence: its record, `research data show RUN`'s
  `outputs.run_record`), and a node by its kind, never its host name.

| SKU | kind | file | Lean table |
|---|---|---|---|
| RTX PRO 6000 Blackwell Server Edition (board 900-2G153-0000-000) | gpu | `cards/rtx-pro-6000-server.toml` | `security/Definitions/Core/Accel/Sm120.lean` |
| H200 SXM 141 GB (board 692-2G520-0280-001) | gpu | `cards/h200-sxm-141gb.toml` | `security/Definitions/Core/Accel/H200.lean` |
| B200 SXM 180 GB (board 692-2G525-0220-000) | gpu | `cards/b200-sxm-180gb.toml` | none yet: `security/Definitions/Accounting/Sheets/B200.lean` is held to it |
| Two Xeon 6776P, the RTX node's host | host | `cards/xeon-6776p-2s.toml` | none |
| Two Xeon Platinum 8468, the H200 nodes' host | host | `cards/xeon-platinum-8468-2s.toml` | none |
| NVIDIA DGX H200, a server none of our nodes is | server | `cards/dgx-h200.toml` | none |

Every card of a kind has the same sections and entries, a host card's `disk.read` apart. A GPU card's are `sm` (`count`, the SMs, and
`warps`, the most warps resident on one at once), `clock` (`tensor` the tensor's, `sm` every other resource's, `memory`
device memory's),
`store` (each store's size; `smem` is the pool of L1 and shared memory, and `smem_carveout` the most of that pool a
kernel can make shared memory, a part of `smem` that no store's size adds; `dram` is the device memory a program gets
and `dram_installed` the memory on the board, a second reading at or above it; and, only where a run found them, the
caches an SM serves code and constants from, `l15`, `l1i`, `l0i` and `const_l0`, which the RTX PRO 6000's card has),
`rate` (each resource's units a SM-clock), `memory` (device memory's bus, transfers and bandwidth, and `read`, the
fastest read of it measured, which stands at the bandwidth as a placeholder where none is), `weight` (the tensor
units a MAC of each format uses), `price` (what an operation costs in tensor units in a timed loop: `fp32_loop`, an FP32
add in a loop of 64 adds and 3 loop instructions, at or above the add's price at the card's ceilings, `rate.tensor` over
32 lanes of `rate.issue`) and `model` (`Accel.Card`'s `exact` and code space). A host card's are `cpu` (the
server's sockets and cores, and `vm_cores`, the cores a GPU that a node's VM gives: a measurement of our nodes, not a
number of the server), `clock`, `rate` (AMX MACs and SHA-512 compressions a cycle a core, and the host's op peak),
`memory` (DRAM's channels, speed, bandwidth, measured read and write ceiling, and `speed_max`, `bandwidth_max` and
`write_max`, a second reading of the speed, the bandwidth and the write ceiling: the channels at the fastest memory
the processor takes, whatever DIMMs the server has), `pcie` (the PCI Express lanes into the sockets: `lanes` a
socket and their `speed`, from the part's page; `other`, what a socket takes from devices apart from them, its
chipset link, or where the part has no chipset the USB ports that may be its own; and `bandwidth`, the bytes a second
all of them carry each way, less the coding), `disk` (`read_max`,
the most the host's disks are read back at, all of them together: the lanes' `bandwidth`, since a disk's bytes are
outside the sockets and enter them over a lane, whatever the disk is; and, only where a run read its node's disks,
`read`, the bytes a second the disks of the node's VM read back: like `vm_cores` a measurement of our node, so the RTX
node's host card has it and the H200 nodes' has no entry rather than an invented one) and `store` (its caches and
allowances, and `chip`, the host's on-chip bytes in S_chip). A server card's is `link`: the line rates of the
server's network ports by its guide's classes (`cluster`, `storage`, `management`), each port in one class, and `rate`,
all of them together, which a boundary link of a host of that model carries at most each way. The DGX H200 is the
machine the inference-only policy's `h200-hgx` location is, and `tests/test_policy.py` holds the policy's `links` to
the card's `link.cluster` and `link.storage`, the twelve ConnectX-7 ports it counts.

**The device's own report.** A ceiling is the hardware's, never its firmware's. A few entries marked `ceiling` are
today what the device reports under its vBIOS, with no document behind them: `clock.sm` on the RTX PRO 6000 and the
B200, `clock.memory` on all three GPU cards and `sm.count` on the B200, and with them what a card computes from one
(`clock.tensor` on those two cards, `memory.bandwidth` on all three). `store.dram` on the H200 and the B200 is of that
kind too: what the shipped firmware gives a program, under the board's `store.dram_installed`. Each bounds a card only
while it is the declared board and vBIOS, which the named assumptions `accounting-declared-device` and
`accounting-sheet-clock` state (`verity.claims`) and no mark does. A host card's `store.l2`, `l1d` and `l1i` are
likewise its guest's CPUID.

~~~python
from verity.hardware.cards import load, skus
skus("host")                                    # ['xeon-6776p-2s', 'xeon-platinum-8468-2s']
load("xeon-6776p-2s")["rate"]["ops"]            # {'value': 1045299200000000.0, 'unit': 'op/s', 'mark': 'ceiling', ...}
~~~

**The interface.** Other code reads a card through `verity.hardware.cards` and imports from nowhere else: the functions
`load`, `skus`, `rate`, `number` (an entry's value as an exact `Fraction`) and `check` (the rule `load` holds a card
to, for another file that states numbers as entries), and the constants `KINDS` (a card's
kinds), `MARKS` (an entry's marks), `ENTRY` (an entry's keys), `WORD` (the bytes in a 32-bit word) and `CARDS` (the
directory the card files are in). The module imports the standard library and `verity.errors` only. The package file,
`verity/hardware/__init__.py`, holds a docstring and nothing else: no import and no re-export. The record
(`verity.hardware.record`), the registration log (`verity.hardware.registration`) and the Site (`verity.hardware.site`)
share the package, so importing one of them, in a verifier too, runs no card-loading code; `tests/test_package.py` holds
both.

`load(sku) -> dict` returns the parsed file and refuses a number without its unit, mark and source, and a card without
its kind. `skus(kind=None)` lists the cards, or those of one kind.

`rate(card, resource="tensor", *, fmt=None, mhz=None) -> Fraction` is the one calculator of a GPU card's peak rates:
Lean's `card.ρ`, the SMs times the resource's rate times its clock (for the tensor `clock.tensor` where the card has one
and `clock.sm` where it doesn't, for the rest `clock.sm`, or the clock `mhz` names), in MACs of operand format `fmt` when
given, and device memory's `memory.bandwidth` in words. A benchmark that quotes a peak calls it rather than multiplying
the card's numbers itself, and `tests/test_lean.py` holds it to each Lean table.

**The Lean tables are generated** (`verity/hardware/lean.py`). A GPU card file that names a Lean table (`lean`) gives
that module: `render(card)` is its text, the card file's numbers (`table(card)`) under the namespace, `clock`,
`tensorClock` and `card` its importers read, and nobody edits it. The text's shape is fixed (the module's docstring
gives it): the Lean check's record of a definition digests the term, so a number written another way, or an arm moved,
is a changed record though no number changed. A division in a field of ℕ is exact, and `render` refuses a card file
whose entry would make Lean's ℕ division truncate.

A card file that also names PoUW's record (`lean_prices`, the RTX PRO 6000's
`security/Definitions/Pouw/PearlC/DevicePricesLoop.lean`) gives that module as well: `render_prices(card)`, the served
record `Prices.sm120Loop` at the card's numbers (`prices(card)`: the add at `price.fp32_loop`, the BF16 MAC and the two
formings by `W1Prices.toPrices`' rule), in the term the hand-written record had.

**Changing a number:** change the card file, then write its table:

~~~sh
uv run python -m verity.hardware.lean --write
~~~

Without `--write` the command names each table its card file no longer gives and exits 1. `tests/test_lean.py` holds
each committed table to `render` byte for byte; it also evaluates each `card` arm as an exact rational, independently
of the generator, and fails on any number that differs from the card file's, and on a store, resource or format of
`security/Definitions/Core/Accel/Isa.lean` that either side lacks. A new SKU is a new card file, whose `lean` names its
table.

## The node files (`verity.hardware.nodes`)

One TOML file per kind of node in `nodes/`, beside their loader (`nodes/__init__.py`): what a node's registration
states that no card does. A node is a host and the GPUs it carries. Its kind is the GPU card and its count, and each
number is an entry under the cards' rules above. A node file names no machine, no address and no provider: it is the
kind's, read on one node of it, and its sources cite what read it by `art:` id.

| Node | file | GPU card | host card | the host's own leaves | its interfaces |
|---|---|---|---|---|---|
| An RTX PRO 6000 node, eight cards | `nodes/rtx-pro-6000-server-x8.toml` | `rtx-pro-6000-server` | `xeon-6776p-2s` | `dram`, and the disks `vda`, `vdb` and `vdc` | `eth0` |
| An H200 node, one card | `nodes/h200-sxm-141gb-x1.toml` | `h200-sxm-141gb` | `xeon-platinum-8468-2s` | `dram`, and the disks `vda`, `vdb` and `vdc` | none on record |

A file has `device` (`sku`, the GPU card, and `count`, how many of it the node carries), `host` (`sku`, the host card)
and, one table a leaf, `memory.NAME`: a memory leaf of the host's own, with its `tier` (`dram` or `disk`), its
`bytes` and its `rate`. DRAM's bytes are the memory the node's VM is given (`lsmem -b`), above `MemTotal`, which leaves
out what the kernel keeps. A disk's leaf is one block device, whole, mounted or not (`lsblk -b`). A leaf with no read
on record has no table. A GPU's own memory is not stated: it is its card's `store.dram`, whole.

A leaf's `rate` is the most its bytes are read back at, in `B/s`, marked `ceiling`: the tightest bound on record that
holds whatever the node runs. The DRAM leaf's is the host card's `memory.bandwidth_max`, the server's channels. A
disk's is its own interface's where the disk states one: a drive of the server's own is behind its own PCI Express
link, whose lanes at their rate, less the coding, bound it (`benchmarks/bandwidth/node.py` reads each disk's driver,
transport and link). A disk that states none takes the host card's `disk.read_max`, the lanes into the server's
sockets, which every byte of a disk enters by. Both nodes' disks are such: `virtio_blk` devices the hypervisor gives
the VM, with no model and no transport. That figure bounds the host's disks together as well as each one, so a reader
that adds the disks' rates takes the smaller of the sum and it. Neither a measured rate nor a limit a provider sets on
a disk is a ceiling: the measured read stays on the host card (`disk.read`), under the ceiling, and a setting is
stated nowhere.

`interface.NAME`, one table an interface, is the node's own way out: an interface that carries traffic off the node,
with its `rate`, the line rate each way in `bit/s`, as the node reports it (`/sys/class/net/NAME/speed`, `ethtool`).
An interface is one when it is a device's and a route of the node's main tables names it; the loopback, a bridge and
a tunnel are the node's own software (`benchmarks/bandwidth/node.py` reads both and says which). An interface that
reports no speed has no table. No latency is stated: nothing reads one, and a link registers at 0.

A node file holds no clock or peak of a card, which are the cards', and no network certifier or memory challenger:
those are the auditor's devices, whose ids and keys are a run's, not a kind of node's.

`nodes.load(kind) -> dict` returns the parsed file and refuses a number that is no entry, and a `sku` that names no
card of its kind. `nodes.kinds()` lists the files.

`nodes.leaves(node, host, card=None) -> (leaves, together)` is the one reader of the leaves' rates. `leaves` is each
`memory.NAME` in the file's order as a `Leaf`: its `id`, `host/NAME`, its `tier`, its `bytes` and `bw`, its rate in
whole bits a second, rounded up. `together` gives, for `dram` and for `disk`, the rate that bounds the host's leaves
of the tier together, in the same unit: the host card's `memory.bandwidth_max` and `disk.read_max` (`card` is that
card's table, loaded by the file's `host.sku` when none is given). It refuses a leaf with no `rate`, a rate that is no
ceiling in `B/s` above 0, a rate above its tier's, and a DRAM rate that is not its tier's.

`Registration.node_entry(node, card, host)`, in the Site
(`site/__init__.py`), gives the entry a node file's table and its GPU card's table register, under the host id its
caller passes: the card's sheet, the site, the host, each device `host/gpuI` with one `hbm` leaf at the sheet's device
memory, the host's leaves `host/NAME`, and for each interface a link `host/NAME` from the host to the outside. The
sheet is built from the card file when the entry is, so no digest is kept in the tree. It refuses a file with no
`dram` leaf or no `disk` leaf: an entry of it would pass the Site's check and state a host with less memory than the
node has. Its message gives the commands that read both on the node.

An interface's link is a boundary link that no certifier splits, at the bits its rate moves in a tick, rounded up.
That is what is so of a node whose interface none of the auditor's hardware sits on, so `onsite` refuses the node by
`uncertified`. Where a network certifier stands in front of such a node without splitting its interface (a process on
another machine that reaches the server through a forward), `node_entry(..., certifier=(id, key))` registers it too:
the certifier under the site and a certified way out, host to certifier and certifier to outside, each at the
interfaces' rates summed. The interfaces' own links stay beside them, so `onsite` still refuses the node, and no
registration says that every bit leaves through a certifier that does not hold the wire.

A file with no `interface` table gives an entry with no link. That entry leaves the node's way out unsaid: `onsite`
finds no boundary link of the node's to refuse, and nothing that reads the site sees how the node reaches the
outside. Read the interfaces on a node of the kind and state them before a registration of it is relied on.

~~~python
from verity.hardware import cards, nodes
from verity.hardware.site import Auditor, Registration, check
node = nodes.load("rtx-pro-6000-server-x8")
gpu = cards.load(node["device"]["sku"])
log = Registration("remote")
site = check(log.append(log.node_entry(node, gpu, host="node0")), Auditor.of_files([gpu]))
~~~

`remote` accepts each node's log against the auditor the GPU card files give, with a stand-in certifier or without
one. `onsite` refuses the eight-card node by `uncertified`, and the H200 node, whose interfaces no run has read, by
`challenger-reach` until a memory challenger is registered with a link to the host (`tests/test_nodes.py`).
