# Network simulation

The device graph, and a mock datacenter of Linux network namespaces that runs the real network certifier. Both read the
accounting layer's declarations, `datacenter.Datacenter`, which `datacenter.examples` (`benchmarks/datacenter`) declares:
`rtx_node` is the measured node (8 GPUs on two sockets, each GPU on a PCIe root port of its own, one NIC on socket 1) as
one block, and `leaf_spine` puts such nodes behind a leaf-spine fabric, each through its block's certifier, with hosts
outside every block.

- `graph.py`: the exports, which read any `Datacenter` that validates: the netns plan `mockdc.py` stands up, and
  Shadow's GML. `python graph.py --blocks 2 --netns` prints `leaf_spine(2)`'s plan.
- `records.py`: the certifier's published records as `datacenter.observe.Record`s (port, window, counts, digest), the
  records an honest run declares by port and window, and the certifiers' parameters as the declaration's `Cadence`;
  `datacenter.check_records` audits them.
- `mockdc.py`: a declaration's netns plan stood up on one host, the certifier's proxy on each block's uplink, and both
  audits.
- `workload/`: what runs on the blocks, with its own suite (its modules are listed in `workload/__init__.py`).
  `python -m workload.lab_run run --out DIR`, from this directory in a Python with torch, serves verity_vllm's serving
  Program on the lab, each block in its namespace behind its certifier's proxy and the client on the outside host, then
  checks the certifiers' audit, N3 and the answers.
- `tests/`: the exports' tests, the declarations' values pinned against memory accounting's constants
  (`verity.core.protocols.remote_bandwidth.timing`) and the census, and the records adapter's tests on an injected clock
  (`uv run tools/verity/check/suites.py verity-network-sim`). The mock datacenter is a script, not a test: it runs in real time.

## The mock datacenter

Every namespace is named `na-*`. Each block (`na-b0`, standing in for a node of 8 GPUs) has one interface, to its
certifier's namespace (`na-b0-cert`), whose other interface goes to a leaf switch. By default there are four blocks, two
leaves joined by a spine (`na-spine0`), and one host outside every block (`na-ext0`, on `na-leaf0`). Only the switches
forward IP. A certifier forwards nothing: a block reaches the fabric only through the connections the proxy makes and
accepts, and an iptables rule in the certifier's namespace drops whatever the block opens. Each fabric link has netem,
2 ms ± 0.5 ms per hop at 400 Gbit/s by default. Nothing in the root namespace changes: each veth is created inside its
namespaces, and deleting the namespaces deletes them.

That layout is `datacenter.examples.leaf_spine(4, leaves=2, spines=1)`'s. Any other declaration works whose blocks each have one
boundary link, to a certifier, and which has a device of kind `outside`: `--declaration FILE.py:NAME` names a
`Datacenter`, or a function returning one.

It needs Linux with iproute2, iptables, setpriv and passwordless sudo, and a Python with numpy. Run it on
vy-nebius-cpu-2 (its system `python3` has numpy), not on an agent VM. Every command takes the same `--declaration`, or
`--blocks`, `--leaves` and `--spines`, and `run` refuses a lab that is up in another shape.

~~~sh
python3 benchmarks/network_sim/mockdc.py up       # create the namespaces, then `status`
python3 benchmarks/network_sim/mockdc.py status   # list them, and check what reaches what
python3 benchmarks/network_sim/mockdc.py run --out DIR [--block b1=late ...]
python3 benchmarks/network_sim/mockdc.py down     # stop their processes, and delete every na-* namespace and veth
~~~

As a recorded run, which leaves the lab up:

~~~sh
research run --on vy-nebius-cpu-2 --project verity --source . --cwd source -- \
  bash -c 'M=benchmarks/network_sim/mockdc.py; python3 -B $M down && python3 -B $M up && python3 -B $M run --out "$OUT"'
~~~

## What a run shows

`run` plays `benchmarks/network_traces/tests/test_active_replay.py`'s synthetic trace, three windows of 60 buckets of
100 ms, on every block at once. In each block runs the server (`active_replay.Server`), in its certifier's namespace the
proxy (`verity.core.protocols.certifier.active.Proxy`, unmodified), and on the outside host the peer, whose connection
crosses the fabric. Each block's server plays one strategy: `honest`; `bits`, whose write times carry a random bit per
session; `late`, which holds a session in window 0 past every sync; or `extra`, which writes one event at a position the
Program doesn't have. Each block is audited twice. The certifier's audit is `audit.Audit`'s `check_link` and
`check_ingress_link` on its raw and committed records. The datacenter record check is `datacenter.check_records` on its
records as `datacenter.observe.Record`s against the ones an honest run of the Program publishes: each port (`b0.cert:out`
for what leaves the block, `b0.cert:in` for what enters it) needs a record for each window, with the declared counts
and digest, and each carried window leaves log2 of the sync set's size bits of timing advice. The declaration's cadence
is the certifiers' calibrated parameters: frames of 4,096 bytes, buckets of 100 ticks of 1 ms, windows of 60 buckets and
16 jitter classes (syncs). Each block's wire is also compared with the honest block's. `DIR/summary.json` has the verdicts and the
timing, and `DIR/b0/` a block's records (`records.json` as `Record`s), logs and audit.

Run `r20261018-073637-e25f`, on vy-nebius-cpu-2 at a load of about 120:

| Block | Strategy | Certifier audit | Datacenter record check | Wire against the honest block's |
|---|---|---|---|---|
| b0 | honest | accepted | 6 port-windows carried, 24 bits of advice | |
| b1 | bits | accepted | 6 port-windows carried, 24 bits of advice | identical |
| b2 | late | refused: `responses` window 0, `status:coverage-never-established` | refused: `port b2.cert:out [0, 6000)`, occupancy and content | differs in window 0 |
| b3 | extra | refused: `responses` window 2, `mismatch` | refused: `port b3.cert:out [12000, 18000)`, occupancy and content | differs in window 2 |

`bits` is the covert channel the certifier closes: it is accepted because its wire is the honest one, byte for byte. The
latest tick was 4.2 ms late against the proxy's 25 ms slack. Each bucket reached the peer about 5.4 ms after its tick
from the blocks on `na-leaf0`, and about 9.8 ms after it from those behind the spine. The earlier runs gave the same
certifier verdicts: `r20261018-065536-38c4`, before the declarations, and `r20261018-071731-392a`, on the first
`datacenter` API.

## Limits

- All the processes share one host's monotonic clock, so one origin times every proxy. Blocks on separate hosts would
  need their clocks synchronized to within the slack.
- `datacenter.Cadence` holds one cadence for both directions. The lab's egress and ingress agree on frame size, bucket
  and window, which `records.cadence` checks, but not on the frames per bucket (2 out, 1 in), which `Cadence` doesn't
  hold.

## Serving a workload on the lab

`workload/lab_run.py` serves verity_vllm's serving Program (`workload/served.py`: 4 layers of a truncated SmolLM2,
237,846 gates, a pipeline stage a block) on the lab's declaration. Each block's server runs in the block's namespace,
mockdc's proxy in its certifier's, and the client on `na-ext0`. A block reaches only its certifier, so the client relays
every message between blocks: it leaves its source as server-sent events of a response and enters its destination in a
request (`workload/lab.py`). Each block is checked three ways: the certifier's audit (`audit.Audit` in `active_replay`'s
live mode, on the block's event log), N3 over the certifier's records (`workload.lab.check`), and the answers against
the Program evaluated in one process. Every window declares the calibration's envelope, δ = 12 buckets of 100 ms. The
run needs a Python with torch; the recorded one builds it with `uv` in a scratch directory.

Run `r20261018-084303-dae6`, on vy-nebius-cpu-2 with the lab up in its default shape:

| Run | Certifier audit | N3 | Answers |
|---|---|---|---|
| honest | accepts all four blocks | holds | equal the reference |
| `--tamper b1:flip-bit` | accepts all four blocks | refuses b1's responses (`content`) and b2's request, which carried the flipped message | equal the reference: the bit doesn't change the token |
| `--tamper b2:late` | refuses b2: `responses` window 0, `status:coverage-never-established` | refuses b2 and the blocks it starves (`missing`, `values`) | none |

The latest proxy tick was 4.1 ms late against the 25 ms slack. Each first event left its block at most 186 ms after its
request, against δ's 1.2 s, and each bucket reached the client a median 5.4 ms after its tick from the blocks on
`na-leaf0` and 9.8 ms from those behind the spine.

## N3 in zero knowledge

`workload/n3_unit.py` prototypes N3's proof unit for one port-window: a Boolean Definition, `N3Window_v1{LAYOUT}`,
evaluated in the clear; nothing proves it yet. Its witness is the frames a certifier recorded on one link direction in
one window and the committed rows of the units whose values they carry. Its public input is the certifier's record of
the window, its frames per bucket (occupancy) and its digest (`records.py`'s format), and its static is the window's
layout. It outputs one bit: 1 if and only if the counts are the layout's, the digest is the frames', and each frame bit
is the layout's constant or the bit of a committed row the layout names.

What is committed, by whom and when:

- Each unit commits the values it holds as hm96 rows (hm96-sha512/row/v2, 256 bits a row, under hiding leaves in a
  frame-v3-sha512 tree, `workload/values.py`) and publishes the root and the count. Here it does so when the run ends;
  in the service it must do so before the coins that select windows are drawn.
- Each certifier publishes its record as each window closes. It keeps the frames, which go to the proof service behind
  the firewall as witness and never reach the auditor.
- With NET-18, the ingress certifier also puts its own SALT frame (kind 3 on stream 0xFFFFFFFF; the window and the
  salt's 64-byte commitment) ahead of the arrivals in each window's first bucket, behind any queue carried from the
  window before. It is not a crossing, so the layout holds it as a constant made from the window's salt commitment,
  public once revealed (`lab_windows(..., salts=...)`). `workload/lab.py` and `workload/n3.py` still refuse it as
  malformed.
- The layout is derived from the public statement alone (the Program, the hardware, φ, the placement and the units' row
  offsets): which frame each slot holds, and where each crossing value's bits sit in its source unit's rows.

What the proof service does per window:

1. It decides from public data what needs no unit. A window with nothing declared or recorded passes. One whose record
   holds more frames than the Program declares, or that carries a value of a unit that committed none, is refused
   (`Window.why`).
2. For each other window the draw selects, it runs one `N3Window_v1{layout}` on the certifier's frames and the units'
   rows against the record's digest and counts, and records the ok bit.
3. It binds each row the unit reads to its unit's public root with its hm96 commitment port, as for any unit that reads
   committed values. Opening the rows inside the unit would cost about 28 SHA-512 compressions more per row.

The unit is the match, one AND per compared bit and 22 per base64 character, plus SHA-512 over the record's prefix and
the frames, 57,947 ANDs per compression. `DEMO` (3 frames of 16 bytes, 2 rows) is 58,944 ANDs: 485 for the match, one
compression, 511 to compare the digest and the final AND. The simulator's windows (64-byte frames) take at most 3
compressions. The lab's (4,096-byte frames) take 33 for a window of one frame and up to 193, about 11.2M ANDs over 426
rows, for one of six; about four fifths of the bytes they hash are the frames' zero fill.

The tests (`workload/tests/test_n3_unit.py`) check the circuit against its Python reference on `DEMO`, the simulator's
windows and synthetic lab windows: an honest window gives ok, and a flipped bit, an extra frame, written fill or a wrong
salt gives not-ok, exactly where N3 in the clear refuses. On run `r20261018-084303-dae6`'s records the circuit agrees
with the reference on every window: honest, 15 of 15 ok; `flip-bit`, not-ok at b1's responses and b2's requests in
window 0; `late`, 4 windows refused without a unit, for values b0 and b1 never committed.
`python -m workload.n3_unit circuit-check --out FILE` runs circuit-check's checks on `DEMO`'s binding and its match,
`N3Match`, in about 30 s. It finds no failures and warns twice: N3Match's 485 ANDs are unpinned, and two `Not` gates
inside `Sha512Compress` are recomputed. Its gaps:

- The `circuit-check` command can't find these Definitions. It discovers sidecars only under its `PACKAGES`, and
  `workload` is not an installed package, so there is no sidecar and no pin.
- The window isn't lowered to C-Flock whole. A whole lowering would carry the SHA-512 compression that circuit-check
  records as not fitting in 64 GiB (`checks.UNLOWERED`), which it doesn't apply to a Definition that calls the
  compression. The driver lowers N3Match whole, and the compression and `LmsEqual` have their own entries.
- Only `DEMO` is checked. The simulator's and the lab's layouts are checked only against the reference, by the tests.

Open questions:

- **hm96 or vllm-v1.** The unit reads hm96 rows. Served traffic's values are committed under verity_vllm's vllm-v1
  roots (SHA-256, not hiding), whose link to hm96 rows is the committer's opening, not a proof
  (`verity/core/protocols/sampled_proofs/README.md`). N3 on served traffic needs the crossings bound to the same
  commitment the compute proofs read: serving's record in hm96 itself, or a unit that opens the vllm-v1 leaves.
- **Placement registration.** The layout names each crossing value's source unit, row and offset, and every stream and
  frame position, so it discloses the placement's crossings and the traffic's shape, which the design rules keep
  private. A registered placement would make the layout a committed input. Otherwise only the occupancy stays public
  and the layout becomes witness, in a universal unit per size class (`verity/ml/private_circuits/`).
- **The record digest isn't hiding.** It is a plain SHA-512 of frames whose only unknowns are value bits, so a window
  with few value bits can be inverted by search. It needs a salt only the certifier and the service hold, or the
  certifier's own hm96 rows.
- **The order of frames from several sources.** The lab's order is fixed by the Program (events by position, requests
  by session). Traffic whose interleaving the prover chooses needs that order as witness, checked in the unit.
- **The SALT frame's slot.** The layout takes it from the record, so it discloses how many frames were carried from the
  window before. A fixed slot would not.
- **Zero fill.** A digest over each frame's used bytes, or the certifier's hm96 rows, would cut a lab window's 33 to
  193 compressions to a few.
