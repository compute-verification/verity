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

## What a run checks

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

`honest` and `bits` are accepted by every audit, and `late` and `extra` refused by both the certifier's audit and the
record check (`run.expected_block`). `bits` is the covert channel the certifier closes: it is accepted because its wire
is the honest one, byte for byte.

### With the salt

`run --salt` turns each certifier's per-window salt on (`Proxy(salt=active.fresh_salt)`, the certifier's
`PROTOCOL.md`): at each window's first bucket the ingress side puts one salt frame, which carries the window and the
placeholder (64 zero bytes) and never the salt, in the block's ingress record, and sends the salt itself to the block's
server on a connection of its own. Once a block's run is over, `run` makes the verifier's "reveal" call to its proxy
and keeps the salts in `DIR/b0/salts.json`. The record check declares each ingress window it has a salt for with its
salt frame first, and the certifier's audit (`active_replay.audit_salt`) reads each window's salt frame and the
revealed salt's width and counts the salt frames in each ingress window and in egress. The salt takes one of the ingress side's frames in each window's first bucket, so a
request enqueued in that bucket can go out a bucket later.

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
  record", and the account charges each at the uplink's full capacity.
- **With `Plan(maintenance=[Maintenance(b4, [6000, 12000))])`** (`mockdc.verifier_plan`), the declared records leave
  window 1 out (`records.declared(..., verifier=plan)`, as `datacenter.audit` does), and the other four port-windows are
  carried. The account is sound, but it charges everything the block's devices hold. The plan has no wipe, so nothing
  bounds what the block holds once it is back.

The certifier's own audit has no verdict for window 1 on either link (`no-verdict`). It has no notion of maintenance:
excusing a window is the accounting layer's. A run with `--blocks 5` leaves the lab in that shape; `workload/lab_run.py`
needs it back in its default shape (`mockdc.py down`, then `up`).

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
the sites, so that each is played across the WAN once, and expects the single site's verdicts block for block.

The WAN adds nothing to δ: a frame's release is counted in buckets from its anchor, the bucket its request entered at
the block's own certifier, and the frame is handed at that certifier, so both ends of δ are at the block's uplink and
the WAN is outside it. Nor does it touch the proxy's slack, which bounds the proxy's own scheduling on its host. It
eats into the requests' lead: the Program sends each request half a bucket, 50 ms, before the tick that enqueues it,
so a WAN whose one-way latency, with the fabric's hops, jitter and the peer's own lateness, passes that lead lands
requests a bucket late, and an honest block's ingress records are refused unless the Program sends them earlier.

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

`workload/lab_run.py` serves verity_vllm's serving Program (`workload/served.py`: 4 layers of a truncated SmolLM2, a
pipeline stage a block) on the lab's declaration. Each block's server runs in the block's namespace, mockdc's proxy in
its certifier's, and the client on `na-ext0`. A block reaches only its certifier, so the client relays every message
between blocks: it leaves its source as server-sent events of a response and enters its destination in a request
(`workload/lab.py`), both on the canonical framing (`active.event`, `active.request_frames`).
Each block is checked five ways: the certifier's audit (`audit.Audit` in `active_replay`'s live mode, on the block's
event log), N3 over the certifier's records (`workload.lab.check`), N3Window_v5's reference on each link-window
(`n3_unit.lab_windows`), N2's served verdict on each link-window (`lab_run.n2_verdicts`), and the answers against the
Program evaluated in one process. Every window declares the calibration's envelope, δ = 12 buckets of 100 ms. The run
needs a Python with torch (below). Its tampers are `--tamper b1:flip-bit`, `b2:late` and `b1:extra`, and their
expected verdicts are `run.WORKLOAD`'s.

Every block's link is served (the certifier's `PROTOCOL.md`, "A served link"): responses padded with empty events to F,
requests to F_in frames, and a proxy that admits a request only while its window still ends idle. The proxy signs
nothing: a stand-in (`n3_unit.Certifier`) signs N2's heads after the run.

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
`run.py`'s tests render its summary.

~~~sh
research run --on vy-nebius-cpu-2 --project verity --source . --cwd source --timeout 90m -- \
  bash -c 'python3 -B benchmarks/network_sim/run.py --out "$OUT"'
~~~

`research run` publishes the night's `summary.txt` and `summary.json` with its outputs in the evidence store, where
results are read.

## N3 in zero knowledge

N3, the content check (every frame a certified link carried is a committed crossing), is prototyped in `workload/` as
Boolean window units evaluated in the clear, in no proof-service Spec and not proved: `N3Window_v3`
(`workload/n3_unit.py`), `N3Window_v4` (`workload/n3_v4.py`) and `N3Window_v5` (`workload/n3_v5.py`), with their sizes
at a served link's shape in `workload/n3_cost.py`. Each module's docstring states its checks, what is public and what
is committed.
