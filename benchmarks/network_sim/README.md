# Network simulation

The device graph, and a mock datacenter of Linux network namespaces that runs the real network certifier. Both read the
accounting layer's declarations, `datacenter.Datacenter`, which `datacenter.examples` (`benchmarks/datacenter`) declares:
`rtx_node` is the measured node (8 GPUs on two sockets, each GPU on a PCIe root port of its own, one NIC on socket 1) as
one block, `leaf_spine` puts such nodes behind a leaf-spine fabric, each through its block's certifier, with hosts
outside every block, and `two_sites` joins two leaf-spines by a WAN between a certifier on each.

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
  checks the certifiers' audit, N3 and the answers.
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
forward IP, and a certifier that watches no block, such as one at an end of a WAN (`graph.routers`). A block's
certifier forwards nothing: a block reaches the fabric only through the connections the proxy makes and accepts, and an
iptables rule in the certifier's namespace drops whatever the block opens. Each fabric link has netem, 2 ms ± 0.5 ms per
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
Program doesn't have; or `maintenance`, honest with its certifier down for window 1 (below). Each block is audited twice. The certifier's audit is `audit.Audit`'s `check_link` and
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

### With the salt

`run --salt` turns each certifier's per-window salt on (`Proxy(salt=active.fresh_salt)`, the certifier's
`PROTOCOL.md`): at each window's first bucket the ingress side puts one salt frame, which carries the window and the
salt's commitment, in the block's ingress record, and sends the salt itself to the block's server on a connection of
its own. Once a block's run is over, `run` makes the verifier's "reveal" call to its proxy and keeps the salts in
`DIR/b0/salts.json`. The record check declares each ingress window with its salt frame first, and the certifier's audit
(`active_replay.audit_salt`) reads each commitment against the revealed salt and counts the salt frames in each
ingress window and in egress. The salt takes one of the ingress side's frames in each window's first bucket, so a
request enqueued in that bucket can go out a bucket later.

Run `r20261018-092640-66bf` (`run --salt`), on vy-nebius-cpu-2 at a load of about 35, gave the table's verdicts on
every block, under both audits. On every block, each ingress window held one salt frame and egress none, no egress
frame held a salt or its commitment, the certifier's audit accepted each window's commitment against the revealed
salt, and the block's server received every salt. The latest tick was 4.3 ms late against the 25 ms slack.

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

## Limits

- All the processes share one host's monotonic clock, so one origin times every proxy. Blocks on separate hosts would
  need their clocks synchronized to within the slack.
- `datacenter.Cadence` holds one cadence for both directions. The lab's egress and ingress agree on frame size, bucket
  and window, which `records.cadence` checks, but not on the frames per bucket (2 out, 1 in), which `Cadence` doesn't
  hold.
- `active_replay`'s Program has one outside peer per link, so a fleet's traffic between sites is a block and the
  outside host, across the WAN. A Program in which a block on one site talks to a block on the other would need the
  far block, behind its own certifier, as a link's peer.
- The WAN's certifiers publish no records: `Datacenter.port` gives a port only where a hop crosses a block's boundary,
  and a link between two certifiers crosses none. Records of what crosses the WAN would need a port for a link between
  datacenters, which the model doesn't have.

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

## The whole night

`verity bench network_sim --out DIR` runs `run.py`, the night on the lab host in one process tree. `verity` isn't on
vy-nebius-cpu-2's path, so there it is `python3 benchmarks/network_sim/run.py --out DIR`. It runs four things in turn:
- the accounting layer's demo (`python -m datacenter.demo`, `DIR/demo.txt`);
- the mock datacenter in five blocks, which take the five strategies, once plain and once with the salt;
- with the lab back in its default shape, lab_run's honest, `b1:flip-bit` and `b2:late` runs;
- the summary.

lab_run runs in a Python with torch: the one `--python` names, else `run.py`'s own if it has torch, else one built
with `uv` in a scratch directory. `run.py` refuses to start while a process runs in a lab namespace. However it ends,
it leaves the lab up in its default shape.

`DIR/summary.txt` has a row per block of each run with:
- the certifier's verdicts, and the salt check's;
- the record check's verdicts, or N3's;
- the account's covert budget;
- the latest proxy tick against the slack.

The answers and the night's worst tick follow the rows. `DIR/summary.json` has the same. The exit status is 1 if a
verdict or an answer isn't the expected one (`run.expected_block`, `run.WORKLOAD`), a step exited nonzero, or a tick ran
past its slack.

~~~sh
research run --on vy-nebius-cpu-2 --project verity --source . --cwd source -- \
  bash -c 'python3 -B benchmarks/network_sim/run.py --out "$OUT"'
~~~

Run `r20261018-102319-bb80` exited 0. It took 5.5 minutes on vy-nebius-cpu-2, at a load of about 25, and gave every
expected verdict, the tables above. With the salt, b4's proxy revealed the salts of windows 0 and 2 only. Its record
check under the plan was unchanged, and the certifier's salt check, like its audit, found nothing for window 1. The
latest proxy tick was 4.3 ms late against the 25 ms slack.
