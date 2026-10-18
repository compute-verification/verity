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

`workload/n3_unit.py` prototypes N3's proof unit for one port-window: a Boolean Definition, `N3Window_v2{FMT, N, M,
SALTED}`, evaluated in the clear; nothing proves it yet. Its witness is the frames a certifier recorded on one link
direction in one window, the descriptors of the committed layout that frame them, the committed rows whose values they
carry, and the opening of the record commitment. It outputs one bit: 1 if and only if every descriptor and every row
opens in its commitment, every frame is what its descriptor puts on its rows, and the record commitment holds the
frames' digest. The module's docstring states each check.

What is public, each with its reason:

- The size class (protocol structure), which is the unit's statics. `FMT` is the run's descriptor format: the frame size
  B, the widest prefix P and suffix Q, the most rows a frame reads MF, each unit's value-tree depth, and the layout
  tree's depth. N is the window's recorded frame count, M its pool of rows rounded up to a power of two, and SALTED
  whether the window holds the certifier's salt frame. P, Q and MF are maxima over the run's frames, so they are
  properties of the layout, not the layout.
- The occupancy (protocol structure): the record's frames per bucket, and `at`, where the window's first descriptor sits
  in the layout. `at` is the port's index times the port's capacity, plus the frames the port recorded in earlier
  windows, less their salt frames.
- Hiding commitments: the layout's root, the record commitment's root and the units' value roots, each with its domain
  id. A salted window also has its window number and its revealed salt commitment, which NET-18 makes public once the
  salt is revealed.
- The output, the ok bit.

What is committed, by whom and when:

- Each unit commits the values it holds as hm96 rows (hm96-sha512/row/v2, 256 bits a row, under hiding leaves in a
  frame-v3-sha512 tree, `workload/values.py`) and publishes the root and the count. Here it does so when the run ends;
  in the service it must do so before the coins that select windows are drawn.
- The prover registers the layout before the run (`register_layout`), in the same form: one tree of hiding rows, under
  the binding SHA-512(`verity/netsim/n3-layout/v1/` ‖ program ‖ placement root), so the layout names the committed
  placement. Each row is a descriptor, which says how one frame is made from at most MF committed rows: raw bytes or
  base64, the text's offset and length, a constant prefix and suffix, and for each slot the row's unit and position (or
  a zero row) and a mask of the 256 bits it keeps. Each port has CAP positions, CAP being the power of two above the most
  frames any port declares. Position port × CAP + j holds the descriptor of the port's frame j, and the positions past
  its last frame hold an invalid one. The salts go to the proof service behind the firewall, with the frames.
- The record commitment is a tree of two hiding 512-bit rows per link and window: the link's egress and ingress record
  digests (`record_digest`, SHA-512 of the record's status and frames, which equals `records.py`'s digest on the lab's
  records), under the binding SHA-512(`verity/netsim/n3-record/v1/` ‖ program ‖ link ‖ window). Only its root is
  public, and the salts go to the service. Here `register_record` stands in for its registrant, which the next
  paragraphs name.
- With NET-18, the ingress certifier also puts its own SALT frame (kind 3 on stream 0xFFFFFFFF; the window and the
  salt's 64-byte commitment) ahead of the arrivals in each window's first bucket. It is not a crossing, so it has no
  descriptor. The unit takes the frame's slot as witness and checks the frame against the revealed commitment.
  `workload/lab.py` and `workload/n3.py` still refuse it as malformed.

The outcome asked for per-window layouts under one commitment. But the buckets frames arrive in decide which frames
fall in which window, at run time, so a layout committed before the run can't be cut by window. The layout is therefore
per frame, in each port's send order. A window reads the descriptors at at, at + 1, and so on, and `at` is public
occupancy. It is still one hiding commitment, registered before the run and opened inside the unit.

**The record commitment needs nothing new from the certifier.** On `main`, N2 (`LinkWindows` in
`verity/core/protocols/certifier/service.py`) already registers each link-window's two certified records under one
salted root through the proof service. Each leaf is hm96-sha512/row/v2 over the record head and the certificate, in a
frame-v3-sha512 tree under one-stage's served domain for the run, the link and the window. The firewall keeps the
records and salts, and the challenger receives the root, never a digest or a certificate. That is the commitment this
unit should open, with N2's root and domain as its `record` inputs and the direction as the leaf. Three things differ
from the stand-in:

1. N2's leaf holds the certificate after the head, and its LMS signature is 34,756 bytes. Opening the leaf therefore
   hashes about 272 more compressions, 15.8M ANDs a window. A leaf of the head alone would cost the head's two or
   three compressions. That is a change to `LinkWindows.leaves`, in `verity/`, which is the certifier owner's call.
2. The head carries the Merkle root of the frames (`certificate.frames_root`), not a digest over them. The unit would
   recompute that tree over its N frames: B/128 + 1 compressions for each frame's leaf, about what the digest spends on
   the frame, plus two compressions for each of about N + D nodes, where D = ⌈log2(T·r)⌉ is the tree's depth.
3. N3's unit and N2's unit would then read one registered value. The service must open one commitment in two units
   and hold both to the same root, which is the cross-unit question asked of circuits.

NET-18's commitment in a slot doesn't make a record hiding. Its salt commitment sits in the signed record and is public
once the salt is revealed, so a digest over frames that include it can be inverted as easily as one without. What
hides the record is the salt of N2's leaf, which only the firewall holds.

What the proof service does per window:

1. It decides from public data what needs no unit. A window with nothing recorded passes. A window is refused if its
   record's status isn't complete, frames or none, or if its frames run past the port's capacity (`Window.why`).
2. For every other window the draw selects, it runs the `N3Window_v2` of the window's size class and records the ok
   bit. A frame past the port's declared frames meets an invalid descriptor, so it gives not-ok. So does a frame that
   carries a value of a unit that committed none, since that unit's root is zero.

Sizes in ANDs. v1 is `N3Window_v1` on the same windows. The record column is the digest's SHA-512 (57,947 ANDs a
compression) plus the commitment's opening (550,655). The rest of each window is the gather and the offsets.

| Window | v1 | v2 | Row reads | Descriptor reads | Frames | Record |
|---|---|---|---|---|---|---|
| `demo()`: 4 frames of 16 bytes, pool of 8 | 58,944 (3 frames) | 11,233,195 | 5,751,384 | 4,432,108 | 416,044 | 608,602 |
| Simulator, 64-byte frames, the smallest: 1 frame, pool of 2 | about 0.06M | 4,998,983 | 2,773,758 | 1,549,505 | 65,520 | 608,602 |
| Simulator, the largest: 4 frames, pool of 4 | about 0.18M | 12,743,854 | 5,547,516 | 6,198,020 | 262,080 | 724,496 |
| Lab, 4,096-byte frames: b3's responses, window 1 (1 frame, pool of 1) | 1,957,754 | 31,074,876 | 2,558,637 | 5,323,426 | 20,663,826 | 2,462,906 |
| Lab: b1's responses, window 0 (1 frame, pool of 256) | about 2.0M | 684,160,828 | 655,011,072 | 5,323,426 | 20,663,826 | 2,462,906 |
| Lab: b0's responses, window 0 (6 frames, pool of 512) | 11,534,760 | 1,483,550,450 | 1,310,022,144 | 31,940,556 | 123,982,956 | 11,734,426 |

v2 is 16 to 340 times v1, nearly all of it openings v1 didn't make. v1 read the layout as a static, and it left the
rows to the service's commitment port. A descriptor read hashes the descriptor's LW bits (69,920 at the lab's format)
before its own hm96 leaf and path, about 92 compressions. A frame at 4,096 bytes is 20.7M ANDs, mostly the shifts that
compact up to 255 rows' kept bits into one stream.

The tests (`workload/tests/test_n3_unit.py`, 31 quick and 2 slow, about 100 s) check the following:

- `N3Frame` against `assemble` on every descriptor of the demo's, the simulator's and the lab's windows, honest and
  tampered, and on random descriptors;
- every opening against its tree;
- the window's reference on the demo, the simulator (whose not-ok ports are exactly N3 in the clear's) and synthetic
  lab windows, with flipped bits, an extra frame, written fill, a status without frames, and right, missing and wrong
  salts;
- the whole circuit against the reference on the demo (nine instances) and on a salted lab window (three).

The evaluations, `python -m workload.n3_unit sim` and `lab`:

- Simulator, 3 blocks, 3 requests, windows of 4 buckets. The circuit equals the reference on all 10 windows of each
  run. The honest run is 10 of 10 ok. `flip-bit` and `fill` on b0 are not-ok at b0's certifier uplink and b1's gpu0
  uplink in window 0. `extra-frame` on b1 is not-ok at b1's certifier uplink in windows 0 and 1, and at b2's gpu0
  uplink in window 1. In every run, those are exactly the ports N3 in the clear refuses.
- Run `r20261018-084303-dae6`'s records, reference only (the circuit at 4,096-byte frames is too large for an agent
  VM). The honest run is 15 of 15 ok. `flip-bit` is not-ok at b1's responses and b2's requests in window 0. `late` is
  not-ok at the four windows that carry values b0 and b1 never committed, and refuses b2's responses on their status.
  Each run takes about 150 s and 4.4 GB, most of it tracing the parts at 4,096 bytes to count their ANDs.

`python -m workload.n3_unit circuit-check --out FILE` runs circuit-check's checks on the demo's window and on its
SHA-free parts, `N3Frame`, `N3Gather` and `N3Scatter`. It finds no failures. It takes 42 minutes and 6.8 GB, nearly
all of it circuit-check's reference evaluator running the window's 67.4M gates on four vectors. It warns three ways:

- 23,040 dead gates in the window: 5,760 in each frame's compaction, `N3Step` and `N3Stay` cells whose carried shift
  bits nothing reads, about 3% of `N3Frame`'s gates;
- 84 XORs in `N3AddBase` that recompute a value;
- no pinned counts for `N3Gather` (6,146 ANDs) and `N3Scatter` (110).

Its gaps:

- The `circuit-check` command can't find these Definitions. It discovers sidecars only under its `PACKAGES`, and
  `workload` is not an installed package, so there is no sidecar and no pin.
- The window isn't lowered to C-Flock whole. A whole lowering would carry the SHA-512 compression that circuit-check
  records as not fitting in 64 GiB (`checks.UNLOWERED`), which it doesn't apply to a Definition that calls the
  compression. The driver lowers `N3Gather` and `N3Scatter` whole. `N3Frame` isn't lowered at all: at about 104,000
  ANDs, it is over the whole-Definition budget of 60,000.
- The window isn't partitioned: at 67.4M gates, it is over the partition budget of 4M.
- Only the demo's format is checked. The simulator's and the lab's formats are checked only against the reference, by
  the tests and the evaluations.

v1's three gaps against the design rules are closed: the layout is witness, the record digest enters only under a
hiding commitment, and the salt frame's slot is witness, so it no longer shows how many frames were carried from the
window before. Open questions:

- **Row reads, the cost lever (needs a ruling).** Which rows a frame reads is private now, so each pool row opens inside
  the unit. That is hm96's leaf and the frame-v3 path, about 44 compressions (2.56M ANDs) at the lab's value-tree depth
  of 13, for each of up to 512 pool rows. v1 left the rows to the service's commitment port, which opens public
  positions. There are three levers:
  - The service opens rows from several units' commitments at private positions into one unit's input (asked of
    circuits).
  - Each unit commits its outgoing crossing values in send order, so a 4,096-byte frame reads about 96 contiguous rows
    instead of up to 255 scattered ones. That needs a proof that those rows are the value rows.
  - A finer pool size than a power of two. Up to half the pool is zero rows, each costing a full read.
- **The layout's binding to the placement (needs a ruling: it is what the guarantee rests on).** The binding names the
  program and the placement root, so the prover can't swap layouts after the run. But nothing checks that the
  descriptors are φ's over the committed placement. A prover could register a layout that frames other bits, and the
  unit would hold the frames to it. One unit per run that derives each descriptor from the opened placement would close
  this; until then, it is an assumption.
- **The record commitment** is the stand-in described above, not N2's leaf. Opening N2's leaf needs the leaf's layout
  changed in `verity/` (item 1 above) and the service's cross-unit opening (item 3).
- **hm96 or vllm-v1.** The unit reads hm96 rows. Served traffic's values are committed under verity_vllm's vllm-v1
  roots (SHA-256, not hiding), whose link to hm96 rows is the committer's opening, not a proof
  (`verity/core/protocols/sampled_proofs/README.md`). N3 on served traffic needs the crossings bound to the same
  commitment the compute proofs read: serving's record in hm96 itself, or a unit that opens the vllm-v1 leaves (asked
  of circuits).
- **Trailing frames.** A port that stops short of its declared frames passes. Each window it recorded matches its
  descriptors, and the windows after it record nothing, which passes from public data. In the late run, b2's responses
  are refused only because their record's status says coverage was never established. The declared count is private
  now, so the closing check needs a unit: one per port that opens the layout at the position after the port's last
  recorded frame and shows that the descriptor there is the invalid one.
- **What the size class discloses.** FMT gives the run's widest prefix and suffix and the most rows any frame reads.
  M gives each window's rows read, to within a factor of two. A fixed M per run would hide the second at the cost of
  the largest unit for every window.
- **The order of frames from several sources.** The lab's order is fixed by the Program (events by position, requests
  by session). Traffic whose interleaving the prover chooses needs that order as witness, checked in the unit.
- **Zero fill.** The record digest still hashes each frame's zero fill, about four fifths of a lab window's 33 to 193
  compressions. The digest is now between 0.3% and 6% of a window's ANDs.
