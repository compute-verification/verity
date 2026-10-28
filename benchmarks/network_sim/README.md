# Network simulation

The device graph, and a mock datacenter of Linux network namespaces that runs the real network certifier. Both read the
accounting layer's declarations, `datacenter.Datacenter`, which `datacenter.examples` (`benchmarks/datacenter`) declares:
`rtx_node` is the measured node (8 GPUs on two sockets, each GPU on a PCIe root port of its own, one NIC on socket 1) as
one zone, `leaf_spine` puts such nodes behind a leaf-spine fabric, each through its zone's certifier, with hosts
outside every zone, and `two_sites` joins two leaf-spines by a WAN between their spines. A zone is a
`datacenter.physical.Zone`; the rest of the code and its flags still call a zone a block (`Datacenter.blocks`,
`--blocks`).

- `graph.py`: the exports, which read any `Datacenter` that validates, a fleet's merged one included: the netns plan
  `mockdc.py` stands up, and Shadow's GML. `python graph.py --blocks 2 --netns` prints `leaf_spine(2)`'s plan, and
  `--fleet` `two_sites(2)`'s.
- `records.py`: the certifier's published records as `datacenter.observe.PortRecord`s (port, sampling period,
  counts, digest), the records an honest run declares by port and sampling period, and the certifiers' parameters as the
  declaration's `Cadence`; `datacenter.check_records` audits them.
- `mockdc.py`: a declaration's netns plan stood up on one host, the certifier's proxy on each zone's uplink, both
  audits, and the account's charge for each zone.
- `workload/`: what runs on the zones, with its own suite (its modules are listed in `workload/__init__.py`).
  `python -m workload.lab_run run --out DIR`, from this directory in a Python with torch, serves verity_vllm's serving
  Program on the lab, each zone in its namespace behind its certifier's proxy and the client on the outside host, then
  checks the certifiers' audit, N2's served verdict and the answers.
- `run.py`: the whole night on the lab host, `verity bench network_sim --out DIR` (below), checked against its expected
  verdicts.
- `tests/`: the exports' tests, the declared tensor-core rates pinned to the census, the records adapter's tests on an
  injected clock, and `run.py`'s summary and expected verdicts
  (`uv run tools/verity/check/suites.py verity-network-sim`). The mock datacenter is a script, not a test: it runs in
  real time.

## The mock datacenter

Every namespace is named `na-*`. Each zone (`na-b0`, standing in for a node of 8 GPUs) has one interface, to its
certifier's namespace (`na-b0-cert`), whose other interface goes to a leaf switch. By default there are four zones, two
leaves joined by a spine (`na-spine0`), and one host outside every zone (`na-ext0`, on `na-leaf0`). Only the switches
forward IP. A certifier forwards nothing: a zone reaches the fabric only through the connections the proxy makes and
accepts, and an iptables rule in the certifier's namespace drops whatever the zone opens. Each fabric link has netem,
2 ms ± 0.5 ms per hop at 400 Gbit/s by default. Nothing in the root namespace changes: each veth is created inside its
namespaces, and deleting the namespaces deletes them.

That layout is `datacenter.examples.leaf_spine(4, leaves=2, spines=1)`'s. Any other declaration works whose zones each
have one boundary link, to a certifier, and which has a device of kind `outside`: `--declaration FILE.py:NAME` names a
`Datacenter` or a `Fleet`, or a function returning one. A fleet is validated (`Fleet.validate`) and stood up merged
(`Fleet.merged`). `--fleet` is `two_sites` (below).

It needs Linux with iproute2, iptables, setpriv and passwordless sudo, and a Python with numpy. Run it on
a CPU node whose system `python3` has numpy, not on an agent VM. Every command takes the same `--declaration`, or
`--fleet`, `--blocks`, `--leaves` and `--spines`, and `run` refuses a lab that is up in another shape.

~~~sh
python3 benchmarks/network_sim/mockdc.py up       # create the namespaces, then `status`
python3 benchmarks/network_sim/mockdc.py status   # list them, and check what reaches what, with each round trip
python3 benchmarks/network_sim/mockdc.py run --out DIR [--salt] [--block b1=late ...]
python3 benchmarks/network_sim/mockdc.py down     # stop their processes, and delete every na-* namespace and veth
~~~

As a recorded run on that node, its outputs in `$OUT`, which leaves the lab up:

~~~sh
M=benchmarks/network_sim/mockdc.py; python3 -B $M down && python3 -B $M up && python3 -B $M run --out "$OUT"
~~~

## What a run checks

`run` plays `benchmarks/network_traces/tests/test_active_replay.py`'s synthetic trace, three sampling periods of 60
buckets of 100 ms, on every zone at once. In each zone runs the server (`active_replay.Server`), in its certifier's
namespace the proxy (`verity.core.protocols.certifier.active.Proxy`, unmodified), and on the outside host the peer,
whose connection crosses the fabric. Each zone's server plays one strategy: `honest`; `bits`, whose write times carry a
random bit per request; `late`, which holds a request in sampling period 0 past every sync; `extra`, which writes one
event at a position the Program doesn't have; or `maintenance`, honest with its certifier down for sampling period 1
(below). Each zone is audited twice. The certifier's audit is `audit.Audit`'s `check_link` and `check_ingress_link`
on its raw and committed records. The datacenter record check is `datacenter.check_records` on its records as `datacenter.observe.PortRecord`s against the ones an honest run of the Program publishes: each port
(`b0.cert:out` for what leaves the zone, `b0.cert:in` for what enters it) needs a record for each sampling period, with
the declared counts and digest, and each carried sampling period leaves log2 of the sync set's size bits of timing
advice. The declaration's cadence is the certifiers' calibrated parameters: frames of 4,096 bytes, buckets of 100 ticks
of 1 ms, sampling periods of 60 buckets and 16 jitter classes (syncs). Each zone's wire is also compared with the
honest zone's. `DIR/summary.json` has the verdicts and the timing, and `DIR/b0/` a zone's records (`records.json` as
`Record`s), logs and audit.

`honest` and `bits` are accepted by every audit, and `late` and `extra` refused by both the certifier's audit and the
record check (`run.expected_block`). `bits` is the covert channel the certifier closes: it is accepted because its wire
is the honest one, byte for byte.

### With the salt

`run --salt` turns each certifier's per-sampling-period salt on (`Proxy(salt=active.fresh_salt)`): at each sampling
period's first bucket the ingress side puts one salt frame, which carries the sampling period's number and the
placeholder (64 zero bytes) and never the salt, in the zone's ingress record, and sends the salt itself to the zone's
server on a connection of its own. Once a zone's run is over, `run` makes the verifier's "reveal"
call to its proxy and keeps the salts in `DIR/b0/salts.json`. The record check declares each ingress sampling period it
has a salt for with its salt frame first, and the certifier's audit (`active_replay.audit_salt`) reads each sampling
period's salt frame and the revealed salt's width and counts the salt frames in each ingress sampling period and in
egress. The salt takes one of the ingress side's frames in each sampling period's first bucket, so a request enqueued in
that bucket can go out a bucket later.

### Maintenance

The fifth strategy, `maintenance`, takes a zone's certifier down for sampling period 1. The default four zones take the
first four strategies, so it needs `up --blocks 5` and `run --blocks 5`. The zone is drained for that sampling period
(`active_replay.outage_plan`): no request is enqueued in it, and the plan is refused if anything of sampling period 0
would still be in flight there. Its proxy serves sampling period 0, then closes the link and stops listening, so for
sampling period 1 the link is dead and no record is published. Half a sampling period before sampling period 2, a proxy
back from maintenance (`Proxy(start=2, position=p)`) listens on the same port, and the
peer connects again. The proxy process stays up through the outage, but the proxy back from it is a new `Proxy` with
fresh certifiers, so it carries nothing from sampling period 0. The zone's records are declared from its own drained
plan, the Program its prover registers.

The record check runs twice on that zone, each time with the account's charge for its claims (`mockdc.charge`:
`datacenter.account` on a profile that credits no gates and registers no store, under the verifier's plan for that
zone):
- **Without a declared maintenance**, the plan is the epoch alone. Both ports' records for sampling period 1 are refused
  "no record", and the account charges each at the uplink's full capacity.
- **With `Plan(maintenance=[Maintenance(b4, [6000, 12000))])`** (`mockdc.verifier_plan`), the declared records leave
  sampling period 1 out (`records.declared(..., verifier=plan)`, as `datacenter.audit` does), and the other four of the
  zone's records are carried. The account is sound, but it charges everything the zone's devices hold. The plan has no
  wipe, so nothing bounds what the zone holds once it is back.

The certifier's own audit has no verdict for sampling period 1 on either link (`no-verdict`). It has no notion of
maintenance: excusing a sampling period is the accounting layer's. A run with `--blocks 5` leaves the lab in that shape;
`workload/lab_run.py` needs it back in its default shape (`mockdc.py down`, then `up`).

### Two sites

`--fleet` stands up `datacenter.examples.two_sites(2)`, many datacenters under one developer
(`Fleet([east, west], [Link(east.spine0, west.spine0, wan)])`), merged (`Fleet.merged`): two leaf-spines, `east` and
`west`, each of two zones (`east.b0`, …, `west.b1`), and one WAN link between their spines. The WAN's capacity and
one-way latency are declared assumptions, `wan/east-west` (100 Gbit/s) and `wan/east-west-latency` (20 ms), and netem
gives it 20 ms ± 2 ms (`--wan-latency-us`, `--wan-jitter-us`). The outside host is on the west site, so an east zone's
link runs through its own certifier, the east fabric, the WAN and the west fabric to `na-ext0`. That makes 15 namespaces
and 14 links.

- The WAN is fabric, as a switch is. Zones are the only parts of the graph the account credits, and every zone's
  boundary is certified already, so a link between datacenters is no boundary (`Fleet`): a certifier on it would have no
  port and publish no record. The spines route across it, and every certifier still forwards nothing.
- The fleet's ports are its zones' eight.
- The Program is the same, one outside peer per link (see Limits), so what crosses the WAN is the east zones' traffic
  with the outside host.

Each zone gets every audit, as on one site. Then the record check runs once more, on every zone's records together
against every zone's declared ones, over the merged fleet's ports, with the account's charge for the whole run
(`mockdc.charge` under the whole plan). `run.py` runs the fleet twice, with the first four strategies swapped between
the sites, so that each is played across the WAN once, and expects the single site's verdicts zone for zone.

The WAN adds nothing to δ: a frame's release is counted in buckets from its anchor, the bucket its request entered at
the zone's own certifier, and the frame is handed at that certifier, so both ends of δ are at the zone's uplink and the
WAN is outside it. Nor does it touch the proxy's slack, which bounds the proxy's own scheduling on its host. It eats into
the requests' lead: the Program sends each request half a bucket, 50 ms, before the tick that enqueues it, so a WAN
whose one-way latency, with the fabric's hops, jitter and the peer's own lateness, passes that lead lands requests a
bucket late, and an honest zone's ingress records are refused unless the Program sends them earlier.

## Limits

- All the processes share one host's monotonic clock, so one origin times every proxy. Zones on separate hosts would
  need their clocks synchronized to within the slack.
- `datacenter.Cadence` holds one cadence for both directions. The lab's egress and ingress agree on frame size, bucket
  and sampling period, which `records.cadence` checks, but not on the frames per bucket (2 out, 1 in), which `Cadence`
  doesn't hold.
- `active_replay`'s Program has one outside peer per link, so a fleet's traffic between sites is a zone and the outside
  host, across the WAN. A Program in which a zone on one site talks to a zone on the other would need the far zone,
  behind its own certifier, as a link's peer.

## Serving a workload on the lab

`workload/lab_run.py` serves verity_vllm's serving Program (`workload/served.py`: 4 layers of a truncated SmolLM2, split
across the zones in pipeline order) on the lab's declaration. Each zone's server runs in the zone's namespace, mockdc's
proxy in its certifier's, and the client on `na-ext0`. A zone reaches only its certifier, so the client relays every
message between zones: it leaves its source as server-sent events of a response and enters its destination in a request
(`workload/lab.py`), both on the canonical framing (`active.event`, `active.request_frames`). Each zone is checked three
ways: the certifier's audit (`audit.Audit` in `active_replay`'s live mode, on the zone's event log), N2's served verdict
on each sampling block (`lab_run.n2_verdicts`), and the answers against the Program evaluated in one process. Every
sampling block declares the calibration's envelope, δ = 12 buckets of 100 ms. The run needs a Python with torch
(below). Its tampers are `--tamper b2:late` and `b1:extra`, and their expected verdicts are `run.WORKLOAD`'s.

Every zone's link is served (the certifier's `served`): responses padded with empty events to F,
requests to F_in frames, and a proxy that admits a request only while its sampling period still ends idle. The proxy
signs nothing: a stand-in LMS key (`lab_run.n2_verdicts`) signs N2's heads after the run.

## The whole night

`verity bench network_sim --out DIR` runs `run.py`, the night on the lab host in one process tree. `verity` isn't on
the lab host's path, so there it is `python3 benchmarks/network_sim/run.py --out DIR`. It runs five things in turn:
- the accounting layer's demo (`python -m datacenter.demo`, `DIR/demo.txt`);
- the mock datacenter in five zones, which take the five strategies, once plain and once with the salt;
- the fleet (`up --fleet --blocks 2`), twice, with the first four strategies swapped between the sites
  (`run.MOCKDC_RUNS`);
- with the lab back in its default shape, lab_run's honest, `b2:late` and `b1:extra` runs;
- the summary.

lab_run runs in a Python with torch: the one `--python` names, else `run.py`'s own if it has torch, else one built
with `uv` in a scratch directory. `run.py` refuses to start while a process runs in a lab namespace. However it ends,
it leaves the lab up in its default shape.

`DIR/summary.txt` has a row per zone of each run with:
- the certifier's verdicts, and the salt check's;
- the record check's verdicts, on mockdc's runs;
- N2's, on lab_run's served links;
- the account's covert budget;
- the median lag from a bucket's tick to its bytes reaching the peer;
- the latest proxy tick against the slack.

Each mockdc run has one more row, `all`, for its record check on every port. The answers and the night's worst tick
follow the rows. `DIR/summary.json` has the same. The exit status is 1 if a verdict or an answer isn't the expected one
(`run.expected_block`, `run.expected_records`, `run.WORKLOAD`), a step exited nonzero, or a tick ran past its slack.
`run.py`'s tests render its summary.

As a recorded run of at most 90 minutes on the lab host, its outputs in `$OUT`:

~~~sh
python3 -B benchmarks/network_sim/run.py --out "$OUT"
~~~

The run publishes the night's `summary.txt` and `summary.json` with its outputs in the evidence store, where results
are read.

## N3, retired

N3, the content check (every frame a certified link carried is a committed crossing), was prototyped here: in the clear
over the certifiers' records and as Boolean circuits over a sampling block, `N3Window_v3` to `N3Window_v5`, that no Spec
ran. It is retired, superseded by the network certifier's step-indexed scan
(`verity/core/protocols/certifier/ingress_scan.py` and `egress.py`), which `benchmarks/network_traces/scan_zk.py` proves
in zero knowledge.
