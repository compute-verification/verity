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

`workload/n3_unit.py` prototypes N3's proof unit for one port-window: a Boolean Definition, `N3Window_v3{FMT, N, G,
SALTED, REC}`, evaluated in the clear; nothing proves it yet. Its witness is the frames a certifier recorded on one link
direction in one window, the descriptors of the committed layout that frame them, the rows of the units' outboxes that
the descriptors read, and the opening of the window's certified head in N2's commitment. It outputs one bit: 1 if and
only if every descriptor and every outbox row opens in its commitment, every frame is what its descriptor puts on its
rows, and N2's commitment holds a complete head over the frames' Merkle root. The module's docstring states each check.

What is public, each with its reason:

- The size class (protocol structure), which is the unit's statics. `FMT` is the run's descriptor format: the frame size
  B, the widest prefix P and suffix Q, the most segments a frame reads, each unit's outbox depth, and the layout tree's
  depth. N is the window's recorded frame count, G the most segments its port's frames read, SALTED whether the window
  holds the certifier's salt frame, and `REC` the certified record's shape: the UTF-8 length of the link's name and the
  grid's T buckets of r slots, fixed before the run. P, Q, the segments and the depths are maxima and sizes over the
  run, so they are properties of the layout and the outboxes, not their contents.
- The occupancy (protocol structure): the record's frames per bucket, and `at`, where the window's first descriptor sits
  in the layout. `at` is the port's index times the port's capacity, plus the frames the port recorded in earlier
  windows, less their salt frames.
- The port's end of its link, 0 for egress and 1 for ingress (protocol structure). It names the leaf of N2's commitment
  the unit opens.
- Hiding commitments: the layout's root, N2's record root and the units' outbox roots, each with its domain id. A salted
  window also has its window number and its revealed salt commitment, which NET-18 makes public once the salt is
  revealed.
- The output, the ok bit.

What is committed, by whom and when:

- **The outbox.** Each producing unit commits the bytes it sends across a certified link, in φ's send order
  (`Phi.send_order`), as one advice tree beside its values (`register_outbox`). A message is the bytes
  `framing.encode` packs its crossing values into, and the unit's messages are back to back. Row q holds bytes
  qB ... qB + 2B − 1, zero past the end, for q below max(1, ⌈bytes / B⌉), and zero rows follow up to a power of two.
  Rows are hm96-sha512/row/v2 under hiding leaves in a frame-v3-sha512 tree, under the binding
  SHA-512(`verity/netsim/n3-outbox/v1/` ‖ program ‖ unit). The unit publishes the root and the domain id when it
  publishes its value root, before the coins that select windows are drawn; here `Statement.table` registers every
  outbox from the committed value rows when the run ends.
  - A message is one contiguous range of its source's outbox. Consecutive rows overlap by B bytes, so any run of at most
    B bytes lies in one row, the row of its first byte, and a frame reads each run as one segment that opens one row.
  - What it discloses: the domain id carries the row count, 2^depth. That gives the unit's crossing bytes to within a
    factor of two, in rows of B, and nothing else about them. In the lab, b0's 6,016 bytes are two rows and every other
    unit's are one. "What the size class discloses" below says how to hide it.
- **The layout.** The prover registers it before the run (`register_layout`), in the same form, under the binding
  SHA-512(`verity/netsim/n3-layout/v1/` ‖ program ‖ placement root), so the layout names the committed placement.
  Each row is a descriptor, which says how one frame is made from at most G runs of outbox bytes: raw bytes or base64,
  the text's offset and length, a constant prefix and suffix, and for each segment its unit, its row q, the run's offset
  in that row, its length and its destination in the frame's data (or an empty segment). Each port has CAP positions,
  CAP being the power of two above the most frames any port declares. Position port × CAP + j holds the descriptor of
  the port's frame j, and the positions past its last frame hold an invalid one. The salts go to the proof service
  behind the firewall, with the frames.
- **The record commitment is N2's.** `LinkWindows` (`verity/core/protocols/certifier/service.py`) registers each
  link-window's two certified records under one salted root through the proof service, in a frame-v3-sha512 tree under
  one-stage's served domain for the run, the link and the window. Leaves 0 and 1 are each direction's head and
  certificate, which N2's own unit reads. Leaves 2 and 3 are each direction's head alone (`service.HEADS`), and N3's
  unit opens leaf 2 + direction. The head holds N2's Merkle root of the window's frames (`certificate.frames_root`) and
  its status. The unit recomputes the root over its N frames, checks that the status is complete, and opens the head:
  two compressions for its digest at the demo's shape and four at the lab's, where T = 60 buckets make the head about
  400 bytes. The leaf of head and certificate would cost about 272, since the certificate's LMS signature is 34,756
  bytes. N2's accepted outcome on the same root says that leaf holds the certified head of the link, direction and
  window. The firewall keeps the records and salts, and the challenger receives the root alone. Here `Certifier` stands
  in for the certifier and the firewall: a small LMS key signs each head, and the salts come from a seed.
- With NET-18, the ingress certifier also puts its own SALT frame (kind 3 on stream 0xFFFFFFFF; the window and the
  salt's 64-byte commitment) ahead of the arrivals in each window's first bucket. It is not a crossing, so it has no
  descriptor. The unit takes the frame's slot as witness and checks the frame against the revealed commitment.
  `workload/lab.py` and `workload/n3.py` still refuse it as malformed.

The layout is per frame, in each port's send order, not per window. The buckets frames arrive in decide which frames
fall in which window, at run time, so a layout committed before the run can't be cut by window. A window reads the
descriptors at at, at + 1, and so on, and `at` is public occupancy.

What the proof service does per window:

1. It decides from public data what needs no unit. A window with nothing recorded passes. A window is refused if its
   record's status isn't complete, frames or none, or if its frames run past the port's capacity (`Window.why`).
2. For every other window the draw selects, it runs the `N3Window_v3` of the window's size class and records the ok
   bit. A frame past the port's declared frames meets an invalid descriptor, so it gives not-ok. So does a frame that
   carries values of a unit that committed none, since that unit's outbox holds zeros where they would be.

Sizes in ANDs, v2 against v3 on the same windows. Row reads are the outbox rows' openings (v2: the pool of value rows).
Frames are the segments' shifts and each frame's placement (v2: the gather and the compaction). The record is N2's
frames root and the head's opening (v2: the stand-in's digest and opening).

| Window | v2 | v3 | Row reads | Descriptor reads | Frames | Record |
|---|---|---|---|---|---|---|
| `demo()`: 4 frames of 16 bytes, 2 segments | 11,233,195 | 11,326,212 | 5,751,384 | 4,200,320 | 19,932 | 1,354,554 |
| Simulator, 64-byte frames, the smallest: 1 frame | 4,998,983 | 3,359,332 | 613,219 | 1,549,505 | 15,891 | 1,180,713 |
| Simulator, the largest: 4 frames | 12,743,854 | 10,184,935 | 2,452,876 | 6,198,020 | 63,564 | 1,470,448 |
| Lab, 4,096-byte frames: b3's responses, window 1 (1 frame) | 31,074,876 | 10,772,855 | 4,329,426 | 1,383,030 | 1,445,908 | 3,614,487 |
| Lab: b1's responses, window 0 (1 frame) | 684,160,828 | 10,772,855 | 4,329,426 | 1,383,030 | 1,445,908 | 3,614,487 |
| Lab: b3's requests, window 0 (2 frames, 2 segments each) | 1,368,270,969 | 30,221,387 | 17,317,704 | 2,766,060 | 4,726,766 | 5,410,844 |
| Lab: b0's responses, window 0 (6 frames) | 1,483,550,450 | 56,473,643 | 25,976,556 | 8,298,180 | 8,675,448 | 13,523,424 |

The simulator's windows run from 3.4M to 10.2M (v2: 5.0M to 12.7M) and the lab's from 10.7M to 56.5M (v2: 31.1M to
1.48B). At the lab's largest window v3 is 26 times smaller:

- Row reads fall from 1.31B to 26.0M. A 4,096-byte frame reads one segment, one 8,192-byte row, instead of up to 255
  scattered value rows. Each read is 4.33M ANDs, most of it the row's 65 compressions.
- Descriptor reads fall from 31.9M to 8.3M: a descriptor is 383 bits, not v2's 69,920 bits of masks.
- Frames fall from 124M to 8.7M: one shift a segment (0.92M at 4,096 bytes) replaces the compaction.
- The record rises from 11.7M to 13.5M, nearly all of it N2's frames root, which v2's stand-in didn't compute.

The demo stays the size v2 was. At 16 bytes a row read is mostly the hm96 leaf and the path, and one 32-byte outbox row
costs what one 256-bit value row did. Rows of B bytes, two per segment, were measured and set aside: they make the demo
17.1M and the lab's largest window 61.8M, because every segment then opens two rows.

The tests (`workload/tests/test_n3_unit.py`, 44 quick and 6 slow, about 100 s) check the following:

- `N3Frame` against `assemble` on every descriptor of the demo's, the simulator's and the lab's windows, honest and
  tampered, and on random descriptors; `N3Seg` against `segment`;
- every opening against its tree, the outbox's rows, and that each unit's outbox is its messages' payloads back to back
  in φ's send order;
- that N2's commitment is `LinkWindows`'s own and that its head leaf holds N2's frames root, which `N3FramesRoot`
  recomputes;
- the window's reference on the demo, the simulator (whose not-ok ports are exactly N3 in the clear's) and synthetic
  lab windows, with flipped bits, an extra frame, written fill, wrong outbox rows, an incomplete head, a status without
  frames, and right, missing and wrong salts;
- the whole circuit against the reference on the demo (thirteen instances) and on a salted lab window (three), and
  `N3RowRead` on rows past a unit's depth and of another unit.

The evaluations, `python -m workload.n3_unit sim` and `lab`:

- Simulator, 3 blocks, 3 requests, windows of 4 buckets. The circuit equals the reference on all 10 windows of each
  run. The honest run is 10 of 10 ok. `flip-bit` and `fill` on b0 are not-ok at b0's certifier uplink and b1's gpu0
  uplink in window 0. `extra-frame` on b1 is not-ok at b1's certifier uplink in windows 0 and 1, and at b2's gpu0
  uplink in window 1. In every run, those are exactly the ports N3 in the clear refuses.
- Run `r20261018-084303-dae6`'s records, reference only (the circuit at 4,096-byte frames is too large for an agent
  VM). The honest run is 15 of 15 ok. `flip-bit` is not-ok at b1's responses and b2's requests in window 0. `late` is
  not-ok at the four windows that carry values b0 and b1 never committed, and refuses b2's responses on their status.
  Each run takes about 160 s and 5.5 GB, most of it tracing the parts at 4,096 bytes to count their ANDs.

`python -m workload.n3_unit circuit-check --parts --out FILE` runs circuit-check's checks on the demo format's SHA-free
parts: `N3Frame` (12,448 gates; lowered to C-Flock, 4,983 ANDs) and `N3Seg` (3,882 gates, 1,524 ANDs). In 71 s it finds
no failures and no dead or redundant gates, and warns only that neither has pinned counts. Without `--parts` it also
checks the whole window, which wasn't run for v3: v2's window, of the same size, took 42 minutes, nearly all of it
circuit-check's reference evaluator running 67.4M gates on four vectors.

Its gaps:

- The `circuit-check` command can't find these Definitions. It discovers sidecars only under its `PACKAGES`, and
  `workload` is not an installed package, so there is no sidecar and no pin.
- The window isn't lowered to C-Flock whole. A whole lowering would carry the SHA-512 compression that circuit-check
  records as not fitting in 64 GiB (`checks.UNLOWERED`), which it doesn't apply to a Definition that calls the
  compression.
- The window isn't partitioned: it is over the partition budget of 4M gates.
- Only the demo's format is checked. The simulator's and the lab's formats are checked only against the reference, by
  the tests and the evaluations.

Open items:

- **The outbox's agreement with the values (asked of circuits).** Nothing here proves that a unit's outbox holds φ's
  encoding of its committed crossing values, or that each byte is the same in every row that holds it. Either the
  compute proof that commits a unit's values also writes its outbox, or one unit per producing unit opens its outbox
  and the value rows it encodes. The second pays v2's private-position reads, but once per unit per run rather than in
  every window.
- **N2's head leaf in two units (asked of circuits).** N3's unit and N2's unit read leaves of one registered
  commitment. The service must open it in both and hold both to the same root.
- **The layout's binding to the placement (an engineering item).** The binding names the program and the placement
  root, so the prover can't swap layouts after the run. But nothing checks that the descriptors are φ's over the
  committed placement: a prover could register a layout that frames other bytes, and the unit would hold the frames to
  it. One unit per run that derives each descriptor from the opened placement would close this. No claim rests on it
  yet.
- **hm96 or vllm-v1 (asked of circuits).** The outboxes are hm96 rows, and so are the values they must agree with.
  Served traffic's values are committed under verity_vllm's vllm-v1 roots (SHA-256, not hiding), whose link to hm96
  rows is the committer's opening, not a proof (`verity/core/protocols/sampled_proofs/README.md`). N3 on served traffic
  needs the crossings bound to the commitment the compute proofs read: serving's record in hm96 itself, or a unit that
  opens the vllm-v1 leaves.
- **Trailing frames.** A port that stops short of its declared frames passes. Each window it recorded matches its
  descriptors, and the windows after it record nothing, which passes from public data. In the late run, b2's responses
  are refused only because their record's status says coverage was never established. The declared count is private,
  so the closing check needs a unit: one per port that opens the layout at the position after the port's last recorded
  frame and shows that the descriptor there is the invalid one.
- **What the size class discloses.** FMT gives the run's widest prefix and suffix, the most segments any frame reads,
  and each unit's outbox depth; G gives the most segments of the port's frames. A port's occupancy already bounds the
  bytes it carried, but a depth is per unit, which is new where one port carries several units' traffic. Padding every
  outbox to the run's deepest would hide the depths at no cost in the unit, which climbs the deepest tree's levels for
  every unit's row; it costs only larger trees to register.
- **The order of frames from several sources.** The lab's order is fixed by the Program (events by position, requests
  by session). Traffic whose interleaving the prover chooses needs that order as witness, checked in the unit.
- **Further cuts.** A window's descriptors sit at consecutive public positions, so one multiproof over at ... at + n − 1
  would share their paths' upper levels, about n + DL node hashes instead of n·DL. Descriptor reads are 61% of the
  simulator's largest window. N2's frames root is the record's largest part in the lab: 33 compressions a 4,096-byte
  frame, fill included, plus the tree's nodes. It is N2's tree, so the unit can't skip the fill.
