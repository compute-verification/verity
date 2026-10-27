# Network simulation

The device graph, and a mock datacenter of Linux network namespaces that runs the real network certifier. Both read the
accounting layer's declarations, `datacenter.Datacenter`, which `datacenter.examples` (`benchmarks/datacenter`) declares:
`rtx_node` is the measured node (8 GPUs on two sockets, each GPU on a PCIe root port of its own, one NIC on socket 1) as
one block, `leaf_spine` puts such nodes behind a leaf-spine fabric, each through its block's certifier, with hosts
outside every block, and `two_sites` joins two leaf-spines by a WAN between their spines.

- `graph.py`: the exports, which read any `Datacenter` that validates, a fleet's merged one included: the netns plan
  `mockdc.py` stands up, and Shadow's GML. `python graph.py --blocks 2 --netns` prints `leaf_spine(2)`'s plan, and
  `--fleet` `two_sites(2)`'s.
- `records.py`: the certifier's published records as `datacenter.observe.Record`s (port, window, counts, digest), the
  records an honest run declares by port and window, and the certifiers' parameters as the declaration's `Cadence`;
  `datacenter.check_records` audits them.
- `mockdc.py`: a declaration's netns plan stood up on one host, the certifier's proxy on each block's uplink, both
  audits, and the account's charge for each block.
- `workload/`: what runs on the blocks, with its own suite (its modules are listed in `workload/__init__.py`).
  `python -m workload.lab_run run --out DIR`, from this directory in a Python with torch, serves verity_vllm's serving
  Program on the lab, each block in its namespace behind its certifier's proxy and the client on the outside host, then
  checks the certifiers' audit, N3, N2's served verdict and the answers.
- `run.py`: the whole night on the lab host, `verity bench network_sim --out DIR` (below), checked against its expected
  verdicts.
- `tests/`: the exports' tests, the declarations' values pinned against memory accounting's constants
  (`verity.core.protocols.remote_bandwidth.timing`) and the census, the records adapter's tests on an injected clock, and
  `run.py`'s summary and expected verdicts (`uv run tools/verity/check/suites.py verity-network-sim`). The mock
  datacenter is a script, not a test: it runs in real time.

## The mock datacenter

Every namespace is named `na-*`. Each block (`na-b0`, standing in for a node of 8 GPUs) has one interface, to its
certifier's namespace (`na-b0-cert`), whose other interface goes to a leaf switch. By default there are four blocks, two
leaves joined by a spine (`na-spine0`), and one host outside every block (`na-ext0`, on `na-leaf0`). Only the switches
forward IP. A certifier forwards nothing: a block reaches the fabric only through the connections the proxy makes and
accepts, and an iptables rule in the certifier's namespace drops whatever the block opens. Each fabric link has netem, 2 ms ± 0.5 ms per
hop at 400 Gbit/s by default. Nothing in the root namespace changes: each veth is created inside its namespaces, and
deleting the namespaces deletes them.

That layout is `datacenter.examples.leaf_spine(4, leaves=2, spines=1)`'s. Any other declaration works whose blocks each have one
boundary link, to a certifier, and which has a device of kind `outside`: `--declaration FILE.py:NAME` names a
`Datacenter` or a `Fleet`, or a function returning one. A fleet is validated (`Fleet.validate`) and stood up merged
(`Fleet.merged`). `--fleet` is `two_sites` (below).

It needs Linux with iproute2, iptables, setpriv and passwordless sudo, and a Python with numpy. Run it on
vy-nebius-cpu-2 (its system `python3` has numpy), not on an agent VM. Every command takes the same `--declaration`, or
`--fleet`, `--blocks`, `--leaves` and `--spines`, and `run` refuses a lab that is up in another shape.

~~~sh
python3 benchmarks/network_sim/mockdc.py up       # create the namespaces, then `status`
python3 benchmarks/network_sim/mockdc.py status   # list them, and check what reaches what, with each round trip
python3 benchmarks/network_sim/mockdc.py run --out DIR [--salt] [--block b1=late ...]
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
session; `late`, which holds a session in window 0 past every sync; `extra`, which writes one event at a position the
Program doesn't have; or `maintenance`, honest with its certifier down for window 1 (below). Each block is audited three
times. The certifier's audit is `audit.Audit`'s `check_link` and `check_ingress_link` on its raw and committed records.
N3Window_v5's reference checks every raw window against the canonical framing's templates (`workload.n3_canonical`).
The datacenter record check is `datacenter.check_records` on its
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
from the blocks on `na-leaf0`, and about 9.8 ms after it from those behind the spine. On the canonical framing
(`r20261021-083115-dae5`, below) the verdicts are the table's, and v5 refuses only the windows the certifier refuses.

### With the salt

`run --salt` turns each certifier's per-window salt on (`Proxy(salt=active.fresh_salt)`, the certifier's
`PROTOCOL.md`): at each window's first bucket the ingress side puts one salt frame, which carries the window and the
placeholder (64 zero bytes) and never the salt, in the block's ingress record, and sends the salt itself to the block's
server on a connection of its own. Once a block's run is over, `run` makes the verifier's "reveal" call to its proxy
and keeps the salts in `DIR/b0/salts.json`. The record check declares each ingress window it has a salt for with its
salt frame first, and the certifier's audit (`active_replay.audit_salt`) reads each window's salt frame and the
revealed salt's width and counts the salt frames in each ingress window and in egress. The salt takes one of the ingress side's frames in each window's first bucket, so a
request enqueued in that bucket can go out a bucket later.

Run `r20261018-092640-66bf` (`run --salt`, when the frame carried the salt's commitment) gave the table's verdicts on
every block under both audits, one salt frame in each ingress window and none in egress, and each server received
every salt. The night (below) runs it on the placeholder.

### Maintenance

The fifth strategy, `maintenance`, takes a block's certifier down for window 1. The default four blocks take the first
four strategies, so it needs `up --blocks 5` and `run --blocks 5`. The block is drained for that window
(`active_replay.outage_plan`): no request is enqueued in it, and the plan is refused if anything of window 0 would still
be in flight there. Its proxy serves window 0, then closes the link and stops listening, so for window 1 the link is
dead and no record is published. Half a window before window 2, a proxy back from maintenance
(`Proxy(start=2, position=p)`, the certifier's `PROTOCOL.md`) listens on the same port, and the peer connects again. The
proxy process stays up through the outage, but the proxy back from it is a new `Proxy` with fresh certifiers, so it
carries nothing from window 0. The block's records are declared from its own drained plan, the Program its prover
registers.

The record check runs twice on that block, each time with the account's charge for its claims (`mockdc.charge`:
`datacenter.account` on a profile that credits no work and registers no store, under the verifier's plan for that
block):
- **Without a declared maintenance**, the plan is the epoch alone. Both of window 1's port-windows are refused "no
  record", and the account charges each at the uplink's full capacity, 400 Gbit/s for 6 s.
- **With `Plan(maintenance=[Maintenance(b4, [6000, 12000))])`** (`mockdc.verifier_plan`), the declared records leave
  window 1 out (`records.declared(..., verifier=plan)`, as `datacenter.audit` does), and the other four port-windows are
  carried. The account is sound, but it charges everything the block's devices hold. The plan has no wipe, so nothing
  bounds what the block holds once it is back.

The certifier's own audit has no verdict for window 1 on either link (`no-verdict`). It has no notion of maintenance:
excusing a window is the accounting layer's.

Run `r20261018-100310-685b` (`up --blocks 5`, `run --blocks 5`), on vy-nebius-cpu-2 at a load of about 45:

| Block | Strategy | Certifier audit | Datacenter record check | Account's charge (covert budget) |
|---|---|---|---|---|
| b0 | honest | accepted | 6 port-windows carried | sound: 24 bits of timing advice |
| b1 | bits | accepted, wire identical to b0's | 6 port-windows carried | sound: 24 bits |
| b2 | late | refused: `responses` window 0, `status:coverage-never-established` | refused: `port b2.cert:out [0, 6000)`, occupancy and content | not sound: 20 bits + 2.4e12 on the refused window |
| b3 | extra | refused: `responses` window 2, `mismatch` | refused: `port b3.cert:out [12000, 18000)`, occupancy and content | not sound: 20 bits + 2.4e12 |
| b4 | maintenance, no plan | no verdict for window 1 on either link; windows 0 and 2 accepted | refused: `port b4.cert:out` and `port b4.cert:in` `[6000, 12000)`, "no record" | not sound: 16 bits + 4.8e12 on refused windows |
| b4 | maintenance, planned | the same | 4 port-windows carried, nothing refused | sound: 16 bits + 2.16e13 held in maintenance |

The first four blocks' verdicts are the earlier runs'. On b4 the records of windows 0 and 2 are `run_shaper`'s, and
the wire is the records. The latest tick was 4.2 ms late against the 25 ms slack. After the run the lab was brought
back up in its default shape (`mockdc.py down`, then `up`), which `workload/lab_run.py` needs.

### Two sites

`--fleet` stands up `datacenter.examples.two_sites(2)`, many datacenters under one developer
(`Fleet([east, west], [Link(east.spine0, west.spine0, wan)])`), merged (`Fleet.merged`): two leaf-spines, `east` and
`west`, each of two blocks (`east.b0`, …, `west.b1`), and one WAN link between their spines. The WAN's capacity and
one-way latency are declared assumptions, `wan/east-west` (100 Gbit/s) and `wan/east-west-latency` (20 ms), and netem
gives it 20 ms ± 2 ms (`--wan-latency-us`, `--wan-jitter-us`). The outside host is on the west site, so an east block's
link runs through its own certifier, the east fabric, the WAN and the west fabric to `na-ext0`. That makes 15 namespaces
and 14 links.

- The WAN is fabric, as a switch is. Blocks are the only units the account credits, and every block's boundary is
  certified already, so a link between datacenters is no boundary (`Fleet`): a certifier on it would have no port and
  publish no record. The spines route across it, and every certifier still forwards nothing.
- The fleet's ports are its blocks' eight.
- The Program is the same, one outside peer per link (see Limits), so what crosses the WAN is the east blocks' traffic
  with the outside host.

Each block gets every audit, as on one site. Then the record check runs once more, on every block's records together
against every block's declared ones, over the merged fleet's ports, with the account's charge for the whole run
(`mockdc.charge` under the whole plan). `run.py` runs the fleet twice, with the first four strategies swapped between
the sites, so that each is played across the WAN once. Run `r20261018-112635-7b3b` (the whole night, below):

| Run | Block | Strategy | Certifier audit | Datacenter record check | Wire lag (median) | Worst tick |
|---|---|---|---|---|---|---|
| fleet | east.b0 | honest | accepted | 6 port-windows carried | 31.1 ms | 4.3 ms |
| fleet | east.b1 | late | refused: `responses` window 0 | refused: `port east.b1.cert:out [0, 6000)` | 31.1 ms | 3.5 ms |
| fleet | west.b0 | bits | accepted, wire identical to east.b0's | 6 port-windows carried | 5.5 ms | 4.0 ms |
| fleet | west.b1 | extra | refused: `responses` window 2 | refused: `port west.b1.cert:out [12000, 18000)` | 9.8 ms | 3.4 ms |
| fleet | all 8 ports | | | 22 port-windows carried, the two above refused; not sound: 88 bits + 4.8e12 | | |
| fleet-swapped | east.b0 | bits | accepted, wire identical to west.b0's | 6 port-windows carried | 31.1 ms | 4.2 ms |
| fleet-swapped | east.b1 | extra | refused: `responses` window 2 | refused: `port east.b1.cert:out [12000, 18000)` | 31.0 ms | 4.2 ms |
| fleet-swapped | west.b0 | honest | accepted | 6 port-windows carried | 5.4 ms | 4.3 ms |
| fleet-swapped | west.b1 | late | refused: `responses` window 0 | refused: `port west.b1.cert:out [0, 6000)` | 9.8 ms | 3.6 ms |
| fleet-swapped | all 8 ports | | | 22 port-windows carried, the two above refused; not sound: 88 bits + 4.8e12 | | |

The verdicts are the single site's, block for block, and on the canonical framing (`r20261021-083115-dae5`) too, v5's
included. What the WAN adds:
- **To δ, nothing.** A frame's release is counted in buckets from its anchor, the bucket its request entered at the
  block's own certifier, and the frame is handed at that certifier, so both ends of δ are at the block's uplink and the
  WAN is outside it. Every block on both sites declared the same sync, δ = 3 buckets at ρ = 1, and was accepted or
  refused for the same reasons.
- **To what the peer sees, the WAN's 20 ms and the far site's hops.** Each bucket reached `na-ext0` a median 31.1 ms
  after its tick from an east block, against 5.5 ms from the west block on `west.leaf0` and 9.8 ms from the one behind
  the spine: from an east certifier the path is the WAN and four 2 ms fabric hops, 28 ms, against the west ones' 4 ms
  and 8 ms. The fastest of three pings from `na-ext0` took 57.0 ms and 55.0 ms to the east certifiers and back, against
  7.9 ms and 16.1 ms to the west ones.
- **To the requests' lead, most of it.** The Program sends each request half a bucket, 50 ms, before the tick that
  enqueues it. From `na-ext0` to an east certifier is 28 ms one way, plus up to 4 ms of jitter and at most 1.65 ms of
  the peer's own lateness, which leaves about 16 ms. Every request entered its declared bucket on both sites, since
  every honest and `bits` block's ingress records were carried. Past about 35 ms of WAN one way, requests would land a
  bucket late and an honest block's ingress records would be refused, unless the Program sent them earlier.
- **To the proxy's slack, nothing.** The latest tick in the fleet's runs was 4.3 ms late against the 25 ms slack, as on
  one site. The slack bounds the proxy's own scheduling on its host, which the WAN doesn't touch.

## Limits

- All the processes share one host's monotonic clock, so one origin times every proxy. Blocks on separate hosts would
  need their clocks synchronized to within the slack.
- `datacenter.Cadence` holds one cadence for both directions. The lab's egress and ingress agree on frame size, bucket
  and window, which `records.cadence` checks, but not on the frames per bucket (2 out, 1 in), which `Cadence` doesn't
  hold.
- `active_replay`'s Program has one outside peer per link, so a fleet's traffic between sites is a block and the
  outside host, across the WAN. A Program in which a block on one site talks to a block on the other would need the
  far block, behind its own certifier, as a link's peer.

## Serving a workload on the lab

`workload/lab_run.py` serves verity_vllm's serving Program (`workload/served.py`: 4 layers of a truncated SmolLM2,
237,846 gates, a pipeline stage a block) on the lab's declaration. Each block's server runs in the block's namespace,
mockdc's proxy in its certifier's, and the client on `na-ext0`. A block reaches only its certifier, so the client relays
every message between blocks: it leaves its source as server-sent events of a response and enters its destination in a
request (`workload/lab.py`), both on the canonical framing (`active.event`, `active.request_frames`).
Each block is checked five ways: the certifier's audit (`audit.Audit` in `active_replay`'s live mode, on the block's
event log), N3 over the certifier's records (`workload.lab.check`), N3Window_v5's reference on each link-window
(`n3_unit.lab_windows`), N2's served verdict on each link-window (`lab_run.n2_verdicts`), and the answers against the
Program evaluated in one process. Every window declares the calibration's envelope, δ = 12 buckets of 100 ms. The run
needs a Python with torch (below).

Every block's link is served (the certifier's `PROTOCOL.md`, "A served link"): responses padded with empty events to F
(b0 88, b1 and b2 8, b3 2), requests to F_in frames (b3 2, else 1), and a proxy that admits a request only while its
window still ends idle. The proxy signs nothing: a stand-in (`n3_unit.Certifier`) signs N2's heads after the run.

The night `r20261023-075024-27ea`, on vy-nebius-cpu-2 with the lab up in its default shape:

| Run | Certifier audit | N3 | N3 v5 (10 windows a block) | N2 (5 a block) | Answers |
|---|---|---|---|---|---|
| honest | accepts all four blocks | holds | accepts all 40 | accepts all 20 | equal the reference |
| `--tamper b1:flip-bit` | accepts all four blocks | refuses b1's responses (`content`) and b2's request, which carried the flipped message | refuses b1's `responses` window 0 and b2's `requests` window 0 | accepts all 20 | equal the reference: the bit doesn't change the token |
| `--tamper b2:late` | refuses b2: `responses` window 0, `status:coverage-never-established` | refuses b2 and the blocks it starves (`missing`, `values`) | refuses window 0 of b0's `responses` and of b1's and b2's two links | refuses b2's window 0 | none |
| `--tamper b1:extra` | accepts all four blocks | refuses b1 (`unexplained`) | accepts all 40 | refuses b1's window 1 | equal the reference |

Under `late` v5 refuses each window that carried bytes of a block that committed no values, which no committed outbox
holds. Under `extra` v5 takes the empty event past the Program's (open). `r20261021-083115-dae5`'s honest 40-token
prompt passed every check then, v5 on all 68 windows a block
(note:network-accounting/20261021T0615Z-report-net53-canonical-framing).

## The whole night

`verity bench network_sim --out DIR` runs `run.py`, the night on the lab host in one process tree. `verity` isn't on
vy-nebius-cpu-2's path, so there it is `python3 benchmarks/network_sim/run.py --out DIR`. It runs five things in turn:
- the accounting layer's demo (`python -m datacenter.demo`, `DIR/demo.txt`);
- the mock datacenter in five blocks, which take the five strategies, once plain and once with the salt;
- the fleet (`up --fleet --blocks 2`), twice, with the first four strategies swapped between the sites
  (`run.MOCKDC_RUNS`);
- with the lab back in its default shape, lab_run's honest, `b1:flip-bit`, `b2:late` and `b1:extra` runs;
- the summary.

lab_run runs in a Python with torch: the one `--python` names, else `run.py`'s own if it has torch, else one built
with `uv` in a scratch directory. `run.py` refuses to start while a process runs in a lab namespace. However it ends,
it leaves the lab up in its default shape.

`DIR/summary.txt` has a row per block of each run with:
- the certifier's verdicts, and the salt check's;
- the record check's verdicts, or N3's;
- N3Window_v5's, as the windows it accepts of those the block recorded;
- N2's, on lab_run's served links;
- the account's covert budget;
- the median lag from a bucket's tick to its bytes reaching the peer;
- the latest proxy tick against the slack.

Each mockdc run has one more row, `all`, for its record check on every port. The answers and the night's worst tick
follow the rows. `DIR/summary.json` has the same. The exit status is 1 if a verdict or an answer isn't the expected one
(`run.expected_block`, `run.expected_records`, `run.WORKLOAD`), a step exited nonzero, or a tick ran past its slack.

~~~sh
research run --on vy-nebius-cpu-2 --project verity --source . --cwd source --timeout 90m -- \
  bash -c 'python3 -B benchmarks/network_sim/run.py --out "$OUT"'
~~~

Run `r20261023-075024-27ea` (`out/night`, at 51da0f4cb5) exited 0 with "every verdict is the expected one" in 10.3
minutes. v5 accepts every window a block recorded but the one the certifier refuses, and on the lab the table's. Each
mockdc run's check on every port refused exactly what its blocks' own checks refused. With the salt, b4's proxy
revealed the salts of windows 0 and 2 only. Its record check was unchanged, and the salt check, like the audit,
found nothing for window 1. The worst proxy tick was 4.6 ms late, of 25 ms of slack.
`run.py`'s tests render its summary.

## N3 in zero knowledge

`workload/n3_unit.py` prototypes N3's proof block for one port-window: a Boolean Definition, `N3Window_v3{FMT, N, G,
SALTED, REC}`, evaluated in the clear; nothing proves it yet. N3 is a design and the simulator's behaviour, not a
protocol: no proof-service Spec runs it. Its witness is the frames a certifier recorded on one link direction in one
window, the descriptors of the committed layout that frame them, the rows of the units' outboxes that the descriptors
read, and the opening of the window's certified head in N2's commitment. It outputs one bit: 1 if and only if every
descriptor and every outbox row opens in its commitment, every frame is what its descriptor puts on its rows, and N2's
commitment holds a complete head over the frames' Merkle root. The module's docstring states each check.

What is public, each with its reason:

- The size class (protocol structure), which is the unit's statics. `FMT` is the run's descriptor format: the frame size
  B, the widest prefix P and suffix Q, the most segments a frame reads, each unit's outbox depth, and the layout tree's
  depth. N is the window's recorded frame count, G the most segments its port's frames read, SALTED whether the window
  holds the certifier's salt frame, and `REC` the certified record's shape: the UTF-8 length of the link's name and the
  grid's T buckets of r slots, fixed before the run. P, Q, the segments and the depths are maxima and sizes over the
  run, so they are properties of the layout and the outboxes, not their contents.
- The occupancy (protocol structure): the record's frames per bucket, and `at`, where the window's first descriptor sits
  in the layout. `at` is the port's index times the port's capacity, plus the frames the port recorded in earlier
  windows, less their salt frames. The frames per bucket are protocol structure only on a port whose link end declares
  an occupancy profile (`LinkWindows.occupancy`). The Program and the placement fix the profile before the run, the
  spec's context binds it, nothing declared at run time sets it, and N2's head check holds the record's counts to it.
  On a port without one, such as the lab's, they are the record's own.
- The port's end of its link, 0 for egress and 1 for ingress (protocol structure). It names the leaf of N2's commitment
  the unit opens.
- Hiding commitments: the layout's root, N2's record root and the units' outbox roots, each with its domain id. A salted
  window also has its salt frame's payload, its window number and the 64 bytes the live certifier fills with the
  placeholder, protocol structure.
- The output, the ok bit.

What is committed, by whom and when:

- **The outbox.** Each producing unit commits the bytes it sends across a certified link, in φ's send order
  (`Phi.send_order`), as one advice tree beside its values (`register_outbox`). A message is the bytes
  `framing.encode` packs its crossing values into, and the unit's messages are back to back. Row q holds bytes
  qB ... qB + 2B − 1, zero past the end, for q below max(1, ⌈bytes / B⌉), and zero rows follow up to a power of two.
  Rows are hm96-sha512/row/v2 under hiding leaves in a frame-v3-sha512 tree, under the binding
  SHA-512(`verity/netsim/n3-outbox/v1/` ‖ program ‖ unit). The unit publishes the root and the domain id when it
  publishes its value root, before any window is proved; here `Statement.table` registers every
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
  one-stage's served domain for the run, the link and the window. Leaves 0 and 1 are each direction's head, `salt_c`
  and certificate, for N2's unit. Leaves 2 and 3 are each direction's head alone (`service.HEADS`), and N3's unit
  opens leaf 2 + direction. The head holds N2's Merkle root of the window's frames (`certificate.frames_root`) and
  its status. The unit recomputes the root over its N frames, checks that the status is complete, and opens the head:
  two compressions for its digest at the demo's shape and four at the lab's, where T = 60 buckets make the head about
  400 bytes. The leaf of head and certificate would cost about 272, since the certificate's LMS signature is 34,756
  bytes. N2's accepted outcome on the same root says that leaf holds the certified head of the link, direction and
  window. The firewall keeps the records and salts, and the challenger receives the root alone. Here `Certifier` stands
  in for the certifier and firewall: a small LMS key signs each head's salted digest, and every salt is seeded.
- With NET-18, the ingress certifier also puts its own SALT frame (kind 3 on stream 0xFFFFFFFF; the window and the
  placeholder, 64 zero bytes) ahead of the arrivals in each window's first bucket. It is not a crossing, so it has no
  descriptor. The unit takes the frame's slot as witness and checks the frame against the public payload.
  `workload/lab.py` and `workload/n3.py` still refuse it as malformed.

The layout is per frame, in each port's send order, not per window. The buckets frames arrive in decide which frames
fall in which window, at run time, so a layout committed before the run can't be cut by window. A window reads the
descriptors at at, at + 1, and so on, and `at` is public occupancy.

What the proof service would do per window, in N3's design (the simulator does it):

1. It decides from public data what needs no unit. A window with nothing recorded passes from public occupancy. On a
   port with a profile that holds because the Program and the placement fix the profile before the run, the spec's
   context binds it and nothing declared at run time sets it, and a head with frames where the profile says 0 fails
   N2's count check. On a port without one the occupancy is the record's own, and nothing backs the pass. A window is
   refused if its record's status isn't complete, frames or none, or if its frames run past the port's capacity
   (`Window.why`).
2. For every other window it runs the `N3Window_v3` of the window's size class and records the ok bit. N3's design
   gives it Selection All, as N2 has: every port-window of every link-window N2 checks gets its ok bit, so N3 would be
   a structural check. N2 compares no grid (`LinkWindows.verify`, `LinkWindows.clear`). A frame past the port's declared
   frames meets an invalid descriptor, so it gives not-ok. So does a frame that carries values of a unit that
   committed none, since that unit's outbox holds zeros where they would be.

In the simulator a link-window is ok when N2's head checks accept it and N3 is ok on both its port-windows. N2's head
check holds each head's counts to the link's occupancy profile. The egress grid is the head's frames, the leaves of its
frames root, filled in by those counts (`grid.fill`), and N3's design holds those frames to the committed layout's. So
once N3 is a proof-service unit with Selection All, the two would hold the grid N2 used to compare in the clear. Until
then the claim is counts only: timing at the profile. N2 needs the egress end's profile, or a served link's profile,
from which it derives the egress counts (the lab's): with neither, N2 has no verdict (`ValueError`).
The network verdict of record is `LinkWindows.network_verdict`, the AND over a link's link-windows with no ε
or share credit: every window of the egress profile registered in order with an accepted outcome of record. The
simulator runs no proof-service stream, so it doesn't call it. `link_checks` records both bits on every link-window of
the simulator, whose ports each declare φ's counts as their profile.

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
1.48B). At the lab's largest window v3 is 26 times smaller: a 4,096-byte frame reads one 8,192-byte row (4.33M ANDs,
most of it the row's 65 compressions) instead of up to 255 scattered value rows, a descriptor is 383 bits instead of
69,920 bits of masks, and one shift a segment (0.92M) replaces the compaction. The record rises from 11.7M to 13.5M,
nearly all of it N2's frames root, which v2's stand-in didn't compute.

The demo stays the size v2 was. At 16 bytes a row read is mostly the hm96 leaf and the path, and one 32-byte outbox row
costs what one 256-bit value row did. Rows of B bytes, two per segment, were measured and set aside: they make the demo
17.1M and the lab's largest window 61.8M, because every segment then opens two rows.

N3 on every link-window, the price of its design's Selection All, is `workload/n3_cost.py`'s (run
`r20261019-171031-6fa8`, at 8450bbaa73). A port-window small enough is traced whole (`n3_unit.size`); a served one, at
35,000 frames of 4,096 bytes, doesn't fit in memory, so it is composed from the parts its body calls (`composed`). A
link-window that recorded no frames needs no N3 unit, so it costs only N2's two head checks, 6,458,742 ANDs at T = 600
(`N2_LINK_WINDOW`). As built:

| Run | Link-windows | With frames | Port-windows with a unit | N3 a link-window | N3 in all |
|---|---|---|---|---|---|
| Simulator, 64-byte frames, 4 buckets a window | 12 | 6 | 10 | 3.4M to 18.1M | 65.4M |
| Lab run `r20261018-084303-dae6`, 4,096-byte frames | 8 | 8 | 15 | 10.8M to 76.2M | 259.9M |

The simulator's six empty link-windows are its wires'. The lab's links are served: their served counts are 678,795
ANDs at T = 600 and F = 1,024 (`served.size`), in place of the count checks.

At the shape of llama8b-node1's served rows (`SERVED`, from run `r20261006-035423-2cd5`: T = 600 buckets of 100 ms,
slots of 4,096 bytes, 1,440 windows a day, busy egress link-windows of 34,499 and 35,386 frames, and 21,372 ingress
frames for 715,340 egress), each part is traced at its statics and no served value is read: AND counts at the served
shape, not a run on served traffic (run `r20261021-020337-2a6e`, at 17ddf3f34a). Egress port-windows are
unsalted with one segment a frame and ingress ones salted with two, as the lab's are, and an event's JSON is guessed at
256 bytes before its crossing text and 64 after. At the lab's depths the outboxes and the layout are the lab's trees. At
one window's, every outbox has 2^16 rows and the layout two ports of 2^16 positions, the least that hold a busy
window's 35,386 frames. At a day's, 2^26 rows and two ports of 2^26. The lab's layout of 2^6 positions doesn't hold a
busy window, so its rows price a window the service refuses (`Window.why`).

| Depths | Egress frames | Ingress frames | N3 egress | N3 ingress | N3 a link-window | N3 a day, one link | N3 ÷ N2 |
|---|---|---|---|---|---|---|---|
| Lab's | 34,499 | 1,031 | 325.6B | 15.2B | 340.8B | 4.91 × 10^14 | 52,766 |
| Lab's | 35,386 | 1,057 | 334.0B | 15.6B | 349.6B | 5.03 × 10^14 | 54,121 |
| One window's | 34,499 | 1,031 | 475.6B | 22.3B | 497.9B | 7.17 × 10^14 | 77,083 |
| One window's | 35,386 | 1,057 | 487.8B | 22.8B | 510.7B | 7.35 × 10^14 | 79,063 |
| A day's | 34,499 | 1,031 | 592.2B | 27.5B | 619.7B | 8.92 × 10^14 | 95,946 |
| A day's | 35,386 | 1,057 | 607.4B | 28.2B | 635.6B | 9.15 × 10^14 | 98,411 |

N2 is 9.30 × 10^9 ANDs a day on one link. N3 is linear in the frames: a least-squares line through each ladder (7
points from 2 frames, to 35,386 egress and 16,384 ingress) gives 9.44M ANDs an egress frame at the lab's depths, 13.8M
at one window's and 17.2M at a day's, and 14.7M, 21.6M and 26.7M an ingress frame, with the largest residual 3.0%. At
the lab's depths a busy egress window is 46% row reads, 21.5% the frames root, 16% descriptor reads, 10% segments and
7% frames; at a day's, the deeper trees make it 50% row reads and 29% descriptor reads. `N3Window_v4` and `N3Window_v5`
(below) bring a busy link-window to 320.1B and 123.9B.

`composed` equals the whole trace (`checked`) at the served format, at the lab's depths, on unsalted one-segment
windows of 2, 3, 64, 257 and 1,024 frames and salted two-segment ones of 2, 3 and 64 (run `r20261019-171031-6fa8`, at
8450bbaa73). That run's next check, a salted window of 257 frames, was killed at its 32 GiB memory cap, so the busy
windows, and every count at one window's and a day's depths, are composed only. `workload/tests/test_n3_cost.py` holds
`composed` to the trace at the demo's format.

The tests (`workload/tests/test_n3_unit.py`, 46 quick and 6 slow, about 100 s) check the following:

- `N3Frame` against `assemble` on every descriptor of the demo's, the simulator's and the lab's windows, honest and
  tampered, and on random descriptors; `N3Seg` against `segment`;
- every opening against its tree, the outbox's rows, and that each unit's outbox is its messages' payloads back to back
  in φ's send order;
- that N2's commitment is `LinkWindows`'s own and that its head leaf holds N2's frames root, which `N3FramesRoot`
  recomputes;
- the window's reference on the demo, the simulator (whose not-ok ports are exactly N3 in the clear's) and synthetic
  lab windows, with flipped bits, an extra frame, written fill, wrong outbox rows, an incomplete head, a status without
  frames, and right, missing and wrong salts;
- every link-window of the simulator committed under its ports' profiles, with N2's head checks and N3's bit on each
  port-window: frames edited in content alone pass N2 everywhere and N3 refuses exactly N3 in the clear's ports, and
  an extra frame is refused by N2 too;
- N3Window_v5 with the requests' template (`sim_v5`) on every simulator window, refusing exactly what v3 refuses;
- the whole circuit against the reference on the demo (thirteen instances) and on a salted lab window (three), and
  `N3RowRead` on rows past a unit's depth and of another unit.

The evaluations, `python -m workload.n3_unit sim` and `lab`:

- Simulator, 3 blocks, 3 requests, windows of 4 buckets. The circuit equals the reference on all 10 windows of each
  run. The honest run is 10 of 10 ok. `flip-bit` and `fill` on b0 are not-ok at b0's certifier uplink and b1's gpu0
  uplink in window 0. `extra-frame` on b1 is not-ok at b1's certifier uplink in windows 0 and 1, and at b2's gpu0
  uplink in window 1. In every run, those are exactly the ports N3 in the clear refuses. Its 12 link-windows (6 links,
  2 windows) each have N2's head checks under the ports' profiles beside N3's bits (`link_checks`, `summary`'s
  `link_windows_ok` and `n2_not_ok`): the honest run is 12 of 12 ok. `flip-bit` and `fill` pass N2 on every
  link-window, so N3 alone refuses them. `extra-frame` is off the profile at b1's uplink in window 0 and b2's in
  window 1, which N2 refuses too; b1's uplink in window 1 has the profile's count with its frames shifted, which N3
  alone refuses. v5 with the template (`n3_v5`) accepts all 24 port-windows of the honest run and refuses exactly
  those above.
- Run `r20261018-084303-dae6`'s records, reference only (the circuit at 4,096-byte frames is too large for an agent
  VM). The honest run is 15 of 15 ok. `flip-bit` is not-ok at b1's responses and b2's requests in window 0. `late` is
  not-ok at the four windows that carry values b0 and b1 never committed, and refuses b2's responses on their status.
  Each run takes about 160 s and 5.5 GB, most of it tracing the parts at 4,096 bytes to count their ANDs.

`python -m workload.n3_unit circuit-check --parts --out FILE` runs circuit-check's checks on the demo format's SHA-free
parts: `N3Frame` (12,448 gates; lowered to C-Flock, 4,983 ANDs) and `N3Seg` (3,882 gates, 1,524 ANDs). In 71 s it finds
no failures and no dead or redundant gates, and warns only that neither has pinned counts. Without `--parts` it also
checks the whole window. `python -m workload.n3_v4 circuit-check --out FILE [--window]` checks v4's new parts and,
with `--window`, its window, and `workload.n3_v5`'s the same for v5; the gaps below are theirs too.

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
- **N2's head leaf in two units (asked of circuits).** N3's unit and N2's head check both read the head leaf of one
  registered commitment. The service must hold both to the same leaf index (2 + direction) under the same registered
  root, or the two units could be answered with different heads: one whose counts meet the profile, and one whose
  frames root covers other frames.
- **The layout's binding to the placement (an engineering item).** The binding names the program and the placement
  root, so the prover can't swap layouts after the run. But nothing checks that the descriptors are φ's over the
  committed placement: a prover could register a layout that frames other bytes, and the unit would hold the frames to
  it. One unit per run that derives each descriptor from the opened placement would close this. No claim rests on it
  yet.
- **hm96 or vllm-v1 (asked of circuits).** The outboxes are hm96 rows, and so are the values they must agree with.
  Served traffic's values are hm96-sha512 rows by default (rows-sha512). An old record's vllm-v1 roots (SHA-256, not
  hiding) link to hm96 rows by the committer's opening, not proof (`verity/core/protocols/sampled_proofs/README.md`).
  N3 needs the crossings bound to the commitment the compute proofs read: serving's record, hm96 by default, or a
  unit opening vllm-v1's leaves.
- **Trailing frames.** Here a port that stops short of its declared frames passes. Each window it recorded matches
  its descriptors, and the windows after it record nothing, which passes from public data. In the late run, b2's
  responses are refused only because their record's status says coverage was never established. N2 closes it where
  the link declares its occupancy profile (`LinkWindows.occupancy`, public, as the occupancy above is): N2 checks every
  link-window, and its head check (`head_check.NetworkHeadCheck_v2`, a Boolean Definition over the head's hiding read
  of the leaf N3 opens) holds each head's counts to the profile's, so a window with a frame fewer or more than the profile is
  rejected, and N3's frames root would hold the unit's N to those counts' sum. The simulator runs those head checks on
  every link-window, its ports' profiles φ's counts (`link_checks`), but not N2's whole outcome: `Certifier` is an
  honest stand-in, so the certificates' host check would only accept. The lab declares no profile, since its timing
  is the wall clock's, but its links are served, and N2's served counts reject a response a frame short or long
  (`extra`'s run). A link with neither profile needs a unit: one per port that opens the layout at the position after
  the port's last recorded frame and shows that the descriptor there is the invalid one.
- **What the size class discloses.** FMT gives the run's widest prefix and suffix, the most segments any frame reads,
  and each unit's outbox depth; G gives the most segments of the port's frames, and v4's R (below) the most runs its
  reads need. A port's occupancy already bounds the
  bytes it carried, but a depth is per unit, which is new where one port carries several units' traffic. Padding every
  outbox to the run's deepest would hide the depths at no cost in the unit, which climbs the deepest tree's levels for
  every unit's row; it costs only larger trees to register.
- **The order of frames from several sources.** The lab's order is fixed by the Program (events by position, requests
  by session). Traffic whose interleaving the prover chooses needs that order as witness, checked in the unit.
- **Further cuts.** What is left after `N3Window_v5`, below, is under its table. N2's frames root, fill included, is
  the floor: it is N2's tree, so the unit can't skip the fill.

### N3Window_v4 and N3Window_v5

`workload/n3_v4.py` builds `N3Window_v4{FMT, N, G, SALTED, REC, R}`: v3's parts with a window's descriptors opened as
one run of the layout (`N3Range`) and its rows as R runs at private starts, each read fresh or a repeat, R public as G
is (`runs_of`). `workload/n3_v5.py` builds `N3Window_v5{FMT, N, G, SALTED, REC, R, TPL, FB}` on v4's runs, with
NET-52's levers as statics; both stay:

- **Lever 1:** outbox rows of B bytes, each byte in one row; a read hashes the one row it adds to its run's two.
- **Lever 2:** a public template TPL fixes every frame's prefix, text and suffix (`N3FrameFixed`); a descriptor keeps
  only r and the segments.
- **Lever 4:** with a template, frames of FB bytes on rows of fewer (`prepare(setup, RB)`, `resplit`). A served
  egress frame is one token's event, so egress rows of 64 bytes hold its data; ingress keeps rows of B.
- **Lever 3**, each range level's shared first block hashed once, is estimated only: it needs a node part that takes
  a midstate.

A busy served link-window of 35,386 egress and 1,057 ingress frames (run `r20261021-052039-10ee`, composed from the
parts, egress reads in one run and salted ingress ones in two). The floor is N2's frames root, 73.9B, which every
variant keeps.

| Variant | Built | One window's depths | ÷ floor | A day's depths | ÷ floor |
|---|---|---|---|---|---|
| v4 | yes | 320.1B | 4.33 | 320.4B | 4.33 |
| Lever 1 | yes | 253.3B | 3.43 | 253.6B | 3.43 |
| Lever 2, on v4 | yes | 298.2B | 4.03 | 298.4B | 4.04 |
| Levers 1 and 2 | yes | 231.4B | 3.13 | 231.6B | 3.13 |
| Levers 1, 2 and 4 | yes | 123.9B | 1.68 | 124.1B | 1.68 |
| Lever 3 on levers 1 and 2 | estimated | 227.0B | 3.07 | 227.2B | 3.07 |
| Lever 3 on levers 1, 2 and 4 | estimated | 119.5B | 1.62 | 119.7B | 1.62 |
| Frame j reads row j, no descriptors | estimated | 147.5B | 2.00 | 147.6B | 2.00 |

Lever 1 rests on nothing new: v5 accepts exactly where v4 does on every window the tests cover, and refuses what v4
refuses elsewhere by each leaf's binding under SHA-512, as v4 does. Lever 2 rests on a canonical framing the certifier
doesn't have yet, each port's frames one public template; N3 checks the template itself, so its verdict takes no new
assumption, and the framing owes completeness. Lever 4 rests on that framing and on every egress frame's data being at
most 64 bytes, one token's text, a longer one cut over frames. Egress rows of 16, 256 and 512 bytes give 1.67, 1.76 and
1.84 at one window's depths; 512 hold any token of GLM-5.3's vocabulary (the one on the node), so a framing that cuts no token is 1.84. Each
lever's mechanism, what the certifier's framing must change, the tests (`workload/tests/test_n3_v5.py`) and
circuit-check: note:network-accounting/20261021T0520Z-finding-n3-v5-levers.
