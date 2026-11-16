# Hardware: design

One TOML file per SKU in `hardware/cards/`, read by the Site's module (`verity/device.py`), the one source of that
card's numbers. A card is a GPU's (`kind = "gpu"`), a host's (`kind = "host"`: a server's CPUs and DRAM, apart from the GPUs it carries,
so the same GPU in another host needs no second card), a server's (`kind = "server"`: what a server model's own
documents state of it apart from those, its network ports) or a time base's (`kind = "timebase"`: the clock an
auditor's device reads, apart from the host it runs in). Each number is an entry `{ value, unit, mark, source }`; the README states the marks and the sources, and lists the
cards.

Every card of a kind has the same sections and entries, a host card's `disk.read` apart. A GPU card's are `sm` (`count`, the SMs, and
`warps`, the most warps resident on one at once), `clock` (`tensor` the tensor's, `sm` every other resource's, `memory`
device memory's, and `default`, the name of the reading of `readings(card)`, `sm` or `tensor`, that every figure
is stated at unless it names the other: the higher tensor ceiling, `sm` on every card),
`store` (each store's size; `smem` is the pool of L1 and shared memory, and `smem_carveout` the most of that pool a
kernel can make shared memory, a part of `smem` that no store's size adds; `dram` is the device memory a program gets
and `dram_installed` the memory on the board, a second reading at or above it, which a registered sheet and a Lean
table take as device memory; and, only where a run found them, the caches an SM serves code and constants from, `l15`,
`l1i`, `l0i` and `const_l0`, which the RTX PRO 6000's card has),
`rate` (each resource's units a SM-clock), `memory` (device memory's bus, transfers and bandwidth, and `read`, the
fastest read of it measured, which stands at the bandwidth as a placeholder where none is), `weight` (the tensor
units a MAC of each format uses), `price` (what an operation costs in tensor units, each operation charged in seconds at
its own pipe's data-sheet ceiling: `fp32`, an FP32 add under each reading, keyed by its name: `fp32.sm`,
both pipes at `clock.sm`, `rate.tensor` over 32 lanes of `rate.issue`, the default reading; `fp32.tensor`, the same
with the tensor at `clock.tensor`, which differs only on the H200, 29.58 there and holding only if its tensor pipe
never exceeds 1830 MHz; and `fp32_loop`, an FP32 add in a loop of 64 adds and 3 loop instructions, at or above
`fp32.sm`) and `model` (`Accel.Card`'s `exact` and code space). A host card's are `cpu` (the
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
server of the location `h200-hgx`, defined in `benchmarks/slowdown/data/hardware.toml`, whose `links`
`benchmarks/slowdown/slowdown/params.py`'s `carded` reads from the card: its `link.rate` times the location's hosts,
and its `link.cluster` as the low. A time base's card's is `drift`:
`max`, the most the clock's rate may differ from the SI second's (`s/s`), and `read`, the largest reading of it on
record, a `measurement`, which `max` is never below.

**A device's drift.** The auditor's figure for its device's drift is `drift.max` of the card of the clock that device
reads (`number(load_card(sku)["drift"]["max"])`), and a row is of record only where that entry's mark is `ceiling`. Where
it is `premise`, as on the one time base with a card (the raw clock of a provider's virtual machine, which no page
bounds), the row prints the premise for the device's host, which the registration names and the card does not, and
`../tests/verity/tests/test_cards.py` lists every entry so marked, in a card or a node file.

**The interface.** Other code reads a card through the Site's module, `verity.device`, and imports from nowhere
else: the functions `load_card`, `skus`, `rate`, `number` (an entry's value as an exact `Fraction`) and `check_card`
(the rule `load_card` holds a card to, for another file that states numbers as entries), and the constants
`CARD_KINDS` (a card's kinds), `MARKS` (an entry's marks), `ENTRY` (an entry's keys), `WORD` (the bytes in a 32-bit
word) and `CARDS` (the directory the card files are in, found from the `verity` package's directory). The module
imports the standard library, the wire and `verity.errors` only. Importing the Site or the wire, in a verifier too,
reads no card file and no node file: the readers run only when called (`../tests/verity/tests/test_imports.py`).

`load_card(sku) -> dict` returns the parsed file and refuses a number without its unit, mark and source, and a card without
its kind. `skus(kind=None)` lists the cards, or those of one kind.

`rate(card, resource="tensor", *, fmt=None, mhz=None) -> Fraction` is the one calculator of a GPU card's peak rates:
Lean's `card.ρ`, the SMs times the resource's rate times its clock (for the tensor `clock.tensor` where the card has one
and `clock.sm` where it doesn't, for the rest `clock.sm`, or the clock `mhz` names), in MACs of operand format `fmt` when
given, and device memory's `memory.bandwidth` in words. A benchmark that quotes a peak calls it rather than multiplying
the card's numbers itself. `readings(card, *, fmt=None) -> dict` is the tensor peak under each reading of its clock the
card states (`READINGS`: `clock.tensor`, the card's own, then `clock.sm`), so a figure quoted under every reading names no
clock itself, and the card's default is `readings(card)[card["clock"]["default"]]`.

A new SKU is a new card file.

## The node files (`hardware/nodes/`, read by `verity.device`)

One TOML file per kind of node in `hardware/nodes/`, read by the Site's module: what a node's registration
states that no card does. A node is a host and the GPUs it carries. Its kind is the GPU card and its count, and each
number is an entry under the cards' rules above. A node file names no machine, no address and no provider: it is the
kind's, read on one node of it, and its sources cite what read it by `art:` id.

The README lists the node files.

A file has `device` (`sku`, the GPU card, and `count`, how many of it the node carries), `host` (`sku`, the host card)
and, one table a leaf, `memory.NAME`: a memory leaf of the host's own, with its `tier` (`dram` or `disk`), its
`bytes` and its `rate`. DRAM's bytes are the memory the node's VM is given (`lsmem -b`), above `MemTotal`, which leaves
out what the kernel keeps. A disk's leaf is one block device, whole, mounted or not (`lsblk -b`). A leaf with no read
on record has no table. A GPU's own memory is not stated: it is its card's `store.dram_installed`, whole.

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

A node's ways out are what a byte leaves it by: its interfaces, and its storage path.

`interface.NAME`, one table an interface, is an interface that carries traffic off the node, with its `rate`, the
line rate each way in `bit/s`, as the node reports it (`/sys/class/net/NAME/speed`, `ethtool`). An interface is one
when it is a device's and a route of the node's main tables names it; the loopback, a bridge and a tunnel are the
node's own software (`benchmarks/bandwidth/node.py` reads both and says which). An interface that reports no speed,
as a hypervisor's `virtio_net` device does, states its `driver` and the `source` of that read in place of a rate: a
figure a provider sets for a VM's size is a setting and a measured throughput a lower bound, so neither is written,
and what is not a number is not an entry. No latency is stated: nothing reads one, and a link registers at 0.

`storage`, once a file, is the storage path: the node's way out that is no interface. A disk that is not the
server's own, a network disk, can be attached to another machine or copied through whoever provides it, so what is
written to it has left the node past every interface, where no network attester can split it. The table names the
disk leaves that ride the path (`leaves`) and says why they leave (`source`). It states no rate: the path's rate is
the one that bounds the node's disks together, the host card's `disk.read_max`, written once, there. Both nodes'
disks all ride it. A server whose drives are all its own has no `storage` table and no such way out.

A node file holds no clock or peak of a card, which are the cards', and no network attester or memory attester:
those are the auditor's devices, whose ids and keys are a run's, not a kind of node's.

`load_node(kind) -> dict` returns the parsed file and refuses a number that is no entry, and a `sku` that names no
card of its kind. `node_kinds()` lists the files.

`leaves(node, host, card=None) -> (leaves, together)` is the one reader of the leaves' rates. `leaves` is each
`memory.NAME` in the file's order as a `Leaf`: its `id`, `host/NAME`, its `tier`, its `bytes` and `bw`, its rate in
whole bits a second, rounded up. `together` gives, for `dram` and for `disk`, the rate that bounds the host's leaves
of the tier together, in the same unit: the host card's `memory.bandwidth_max` and `disk.read_max` (`card` is that
card's table, loaded by the file's `host.sku` when none is given). It refuses a leaf with no `rate`, a rate that is no
ceiling in `B/s` above 0, a rate above its tier's, and a DRAM rate that is not its tier's.

`ways_out(node, host, card=None) -> tuple[Way, ...]` is the one reader of the ways out. Each is a `Way`: its
`id`, `host/NAME` for an interface and `host/storage` for the storage path; `bw`, its rate in whole bits a second,
rounded up, or `None` for an interface that states none; and `split`, whether a network attester may split it, which
is so of an interface and not of the storage path. The interfaces come first, in the file's order. The storage path's
`bw` is `leaves`' rate for the disks together, so the file writes no number twice. It refuses an interface that
states neither a rate nor its read, a storage path that does not name disk leaves of the file's, each once, and a
leaf or an interface named `storage`.

`Registration.node_entry(node, card, host, host_card=..., ways=ways_out(node, host), leaves=leaves(node, host))`, in
the Site's module beside the readers, gives the entry a node file's table, its GPU card's and host card's tables, its
ways out and its leaves register, under the host id its caller passes: the card's sheet, the site at its unit of time,
the host with its on-chip state in bits (eight times the host card's `store.chip`) and the rate each tier's leaves
share, each device `host/gpuI` with one `hbm` leaf at the sheet's device memory, the host's leaves `host/NAME` at their
rates, and for each way out a link under its id from the host to the outside. `node_figures` of the same tables gives
the auditor's figures for the host and its leaves, which the Site's check holds the entry's to. The sheet is built from
the card file when the entry is, so no digest is kept in the tree.
It refuses a file with no `dram` leaf or no `disk` leaf: an entry of it would pass the Site's check and state a host
with less memory than the node has. Its message gives the commands that read both on the node. `node_entry` reads
no file itself, so its caller hands it the ways out and the leaves, and it holds them to the file's tables: each way
by name and in order, with a rate exactly where the file states one, and each leaf by name, tier and bytes. A rate's
value is `ways_out`'s or `leaves`' reading.

A way out's link is a boundary link that no network attester splits, at its rate in bits a second. That is what is so
of a node none of whose ways out the auditor's hardware sits on. Two kinds of way say why no network attester splits them
(`Link.unmediated`, from the way's kind): an interface that reads no line rate, whose link states no capacity ("no
line rate stated"), and the storage path ("storage path"). `onsite` takes their reasons, and a site with either is
not mediated (`Site.mediated`). An interface with a rate says none, so `onsite` refuses the node by `uncertified`
until a network attester splits it. Where a network attester stands in front of such a node without splitting its
interface (a process on another machine that reaches the server through a forward),
`node_entry(..., certifier=(id, key, scheme))` registers it too: the network attester under the site (a location of
kind `certifier`) and a certified
way out, host to network attester and network attester to outside, each at the interfaces' rates summed. The ways' own
links stay beside them, so `onsite` still refuses the node, and no registration says that every bit leaves through a
network attester that does not hold the wire. The storage path's rate is not in a network attester's links and no
network attester is linked to the path: one that did split the interface would leave the path's link, and the site
would still not be mediated.

An entry always registers a link out. An interface with no rate takes no network attester, so naming one on a node none of
whose interfaces has a rate is refused. A file that states no interface gives no entry, since the entry would not say
how the node reaches the outside. Each message says what to read or state.

An auditor's device has no node file: `clock_drift(host, load_card(sku))`, beside `node_entry`, reads the card of the
time base its clock runs on (`kind = "timebase"`) and gives, by `drift.max`'s mark, the host's premise
(`ClockPremise`: `ClockRateWithin(<host>, <drift.max>)`, the number, its mark and its source) or, for a `ceiling`, the
figure the host states and `Auditor.drift` holds it to. A number marked `premise` never becomes that figure, so a row
that reads a drift the auditor registered names no premise. A `measurement` or a `placeholder` bounds no clock and is
refused. `auditor_host(host, key, scheme, load_card(sku))` is the host's location (kind `auditor-host`, at the site):
it states the card's SKU as its time base and the figure as its drift, or no drift for a premise.
`node_entry(..., certifier=..., certifier_host=that location)` registers the network attester in the host, so the
clock the network attester reads is the host's (`Site.clock`, `Site.drift`); a network attester under the site reads
none and times nothing.
`Auditor.of_files(..., timebases={host: card})` holds the host's time base and drift, and `ClockPremise.premised()` is
the host's `Account.Premised` (host, drift, premise), the drift of every device of the auditor's in the host.

~~~python
from verity.device import Auditor, Registration, check, leaves, load_card, load_node, node_figures, ways_out
node = load_node("rtx-pro-6000-server-x8")
gpu, host = load_card(node["device"]["sku"]), load_card(node["host"]["sku"])
log = Registration("remote")
out, own = ways_out(node, "node0"), leaves(node, "node0")
entry = log.node_entry(node, gpu, "node0", host_card=host, ways=out, leaves=own)
memory, figures = node_figures(node, host, "node0", ways=out, leaves=own)
site = check(log.append(entry), Auditor.of_files([gpu], memory=memory, hosts=[("node0", figures)]))
~~~

`remote` accepts each node's log against the auditor the GPU card files give, and neither site is mediated. `onsite`
refuses the eight-card node by `uncertified`, by its interface's link, with a stand-in network attester or without
one, and accepts the H200 node, both of whose links say why no network attester splits them, once a memory attester reaches its host
(`../tests/verity/tests/test_nodes.py`).
