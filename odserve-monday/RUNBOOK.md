# Monday's run of record: the runbook

This branch (`cursor/odserve-monday-kit-061a`) is never merged. It holds the kit for Monday's onsite run of record: every
request and response served through the network certifier on vy-auditor-2, an auditor host the prover doesn't control,
from an H200 (vy-dc-h200-2), with the V3c audit, the network certifier's chain and period, the composite and the onsite
record on a CPU follow-on node. Every script runs from any VM with the team's injected secrets; nothing in it reads a path
on the VM that staged it.

The run's tree is `cursor/onsite-push1b-061a` at `9068336929e0bdd40dc6f60cab2b8975c3809e70` (push 1b). `tree.sh` refuses
another head unless `WANT` names it: a later head only on the coordinator's word.

| What | Where |
|---|---|
| served node | vy-dc-h200-2, 204.12.188.151, one H200: settings, probe, preflight, serve, send, stock control, check, then stopped |
| follow-on node | vy-nebius-cpu-12, 81.85.1.89 (`FOLLOW_NODE`; vy-nebius-cpu-11, 81.85.4.136, is the fallback): 645 GB, one 30 h job |
| network certifier | vy-auditor-2 (`research@vy-auditor-2`, its gate pinned at `701c8c0073a6`); trust in it is `trusted-host` |
| challenger gate | `research@204.12.178.84`, pinned at `f0c4b1355fb7`, lane network-accounting |
| settings | `settings/settings-monday.env`, sha256 prefix `b610e7e37a9bc461`, installed as `/workspace/research/lanes/network-accounting/onsite/settings-monday.env` |

Monday's settings: V3c class `8,32,4,4,16` (CLASSES too), `OF_RECORD=1`, `PRODUCT=remote-network-no-pouw` untied
(`H1_TIE=`) at `V3C_LAW=subset:1`, `POUW_CMD=` (empty), `H1_GATE` on lane network-accounting's key with
`H1_LANE=network-accounting`, `MIN_FREE_INODES=500000`, and the follow-on's caps `V3C_AUDIT_TIMEOUT=13h`,
`CERTIFIER_TIMEOUT=4h`, `CERTIFIER_PERIOD_TIMEOUT=1h`, `COMPOSITE_TIMEOUT=6h`: 24 h together inside its job's
`--timeout 108000` (30 h), which leaves about 6 h for the copy and the record. At 9068 `certifier_piece.py` takes no
`--case-only`, so the period step runs in full, inside its 1 h cap.

## Rules for the run

- Launch only the steps below, in order. Never two GPU jobs on h200-2 at once: the stock control waits for the send.
- Read a run with `runlog.sh` when you check on it; don't poll in a loop. `research run --wait` stops waiting after 20
  minutes (exit 79) and the run goes on.
- Private, never posted: SLOTS, requests, WINDOW_STEPS, MAX_TOKENS, the window's bit, unit, member and file counts,
  WINDOW-COMMIT's registration counts, INFERENCE-ONLY's numbers, α and Γ. `runlog.sh` cuts them from the serve and
  follow-on lines and never prints the ONSITE line, whose serving field holds the counts. The settings job's
  `CAP: unset` line names WINDOW_STEPS and MAX_TOKENS; it is cut too. A failed record check's why can name a member's key
  and n: never post it. `deferred.env` stays on the node.
- Every run id is a store run id (`r<date>-<time>-<4 hex>`), the only form custody publishes.
- h200-2 stops through `stop.sh` alone. If vy-nebius-stop exits 3 or 4, leave the node and post on INF-527. Never run
  shutdown or poweroff: Nebius treats a guest shutdown as a failure, restarts the VM and keeps billing it.
- The serving cost's label is "the certifier's pacing plus H1's commits, against a same-engine stock control". Never set
  it beside run 2's figure.
- Say "served" of the network verdict only once `verity/remote-network/v0` has landed.

## 0. The VM

~~~bash
compgen -e | grep -c '^RUNPOD_SSH_KEY_B64$'          # 1: the research key, which every ssh and research run here uses
git -C /workspace fetch origin cursor/odserve-monday-kit-061a
git -C /workspace worktree add --detach ~/odserve-monday-kit origin/cursor/odserve-monday-kit-061a
cd ~/odserve-monday-kit/odserve-monday
~~~

`env.sh` (sourced by every script) takes `MAIN_REPO` (default `/workspace`: a checkout with origin and its synced
`.venv`, whose `research` and python every script uses), `TREE` (default `~/wt-odserve-monday`), `LOGS` (default
`~/odserve-monday-logs`, every step's log) and `FOLLOW_NODE`. A custody repair also needs the evidence store's keys.

## 1. The tree

~~~bash
bash tree.sh
~~~

Expect:

~~~text
TREE /home/ubuntu/wt-odserve-monday at 9068336929e (= origin's cursor/onsite-push1b-061a), clean: yes
CASE-ONLY no
V2-CASE drawn member, layout from the pin: yes
PIN public-pin.json, layout held: yes
INODES MIN_FREE_INODES: yes
~~~

## 2. The served node is up

infra starts vy-dc-h200-2 for the lease on INF-527. Then:

~~~bash
bash node_probe.sh research@204.12.188.151 'uptime; nvidia-smi -L; df -i /workspace | tail -1'
~~~

Expect `GPU 0: NVIDIA H200 ...` and `EXIT 0`.

## 3. The settings install at 9068

~~~bash
bash settings/settings_job_launch.sh
~~~

Expect `FILE settings-monday.env b610e7e37a9bc461`, `SETTINGS JOB r... exit 0`, `STATE done`, and the job's lines:

~~~text
REMOTE SETTINGS: readable, sets 2 of SLOTS and MAX_TOKENS
COST LOAD: remote's first load (REQUESTS, SLOTS by default) is the front's group's size: yes
settings-monday.env: INSTALLED /workspace/research/lanes/network-accounting/onsite/settings-monday.env (600)
settings-monday.env: FRONT: the tree's front takes its counts (...)
settings-monday.env: OF RECORD: untied remote-network-no-pouw at subset:1, COMPOSITE unset, V3C_TIMEOUT 120m, WC_TIMEOUT 60m, CLASSES holding V3C_CLASS; the follow-on's caps: audit 13h, chain 4h, period 1h, composite 6h
settings-monday.env: V3C: deferred with remote-network-no-pouw's composite to the follow-on on research@vy-nebius-cpu-11 research@vy-nebius-cpu-12: class 8,32,4,4,16, law subset:1 untied, ...
settings-monday.env: CAP: unset …
COPY PATH: rsync to research@vy-nebius-cpu-11 wrote probe-.../ (...)
COPY PATH: rsync to research@vy-nebius-cpu-12 wrote probe-.../ (...)
GATE research@vy-auditor-2 answers with the lane's key, pinned at 701c8c0073a63fb9d3fdc6d9a993e3d5a5039656
GATE KEY: no shell on research@vy-auditor-2 (a command the gate doesn't take runs nothing)
~~~

Any `REFUSED` line stops the run until it is fixed.

## 4. The h200-2 inode cleanup

~~~bash
bash inodes.sh
~~~

It lists every serve dir that still holds a class window (`WINDOW <dir>/v3c-w0-8-32-4-4-16 custody=... sent=...
busy=0`) and ends `FREE INODES n of ...`. The serve refuses under 500,000 free inodes, and its own window takes several
hundred thousand more, so with fewer than 850,000 free, remove windows whose copy and custody are confirmed:

~~~bash
REMOVE="ID ..." bash inodes.sh                                   # custody preserved (or fetched) and sent
REMOVE="r20261106-100208-h2n2" ALLOW="r20261106-100208-h2n2" bash inodes.sh   # Friday's, which predate custody
~~~

Expect `REMOVED ...` for each, then `FREE INODES` at least 850,000. A `KEPT ...: why` line names what is missing.

## 5. The probe and the preflight

~~~bash
bash probe.sh
~~~

Expect `PROBE r... exit 0`, the probe's lines `DIGEST S 1 48b859c923dc29eb...` (rr/circuit.txt's),
`PREDICT-ROW exit 0: ...`, `PREDICT-COMPARE against r20261102-233856-d4d6 exit 0`,
`SETTINGS /workspace/research/lanes/network-accounting/onsite/settings-monday.env b610e7e37a9bc461`, then `PROBE: PASS`
and `PREFLIGHT launch exit 0: r...`. A probe still running at 20 minutes: `PROBE_RID=r... bash probe.sh` once it ends.
When the preflight has ended (a few minutes):

~~~bash
bash runlog.sh h200-2 PREFLIGHT_ID serve
~~~

Expect `STATE done`, `STATED tree: 9068336929e0...`, `GATE pinned at f0c4b1355fb7..., main.txt its first-parent line
(... commits)`, `REGISTERED: ... under the window Program 48b859c923dc29eb..., weights root c3c8a1ba...`, `ROW: ...`,
`CHALLENGER KEY: ...` and `PREFLIGHT_ONLY: every check passed; served_verdicts.sh not run`.

## 6. The serve

`END` is the lease's end less an hour (the publish margin); the serve's timeout is the time to it, at most 8 h, and it
refuses with less than 4 h left.

~~~bash
END=2026-11-09T15:00:00Z bash serve.sh 2>&1 | tee -a ~/odserve-monday-logs/serve.txt
~~~

Expect `TREE 9068336929e (= cursor/onsite-push1b-061a), settings-monday.env b610e7e37a9bc461, main.txt 4887 commits
from f0c4b1355fb, TIMEOUT ...`, the run's id `RID` on a line of its own, and `research: run RID launched on vy-dc-h200-2`.
Check on it (`bash runlog.sh h200-2 RID serve`) every half hour or so. When it has ended, expect:

~~~text
STATE done CUSTODY "preserved": true FETCHED no
GATE pinned at f0c4b1355fb7..., REGISTERED: ..., ROW: ...
(PERIOD and H1 lines: lane network-accounting's period opened and closed, both window reserves joined)
(CERTIFIER-FRONT: the front's exit 0)
SEALED: every closed window's row dir committed under rows-sha512 ...
B-OF-RECORD: every sealed record binds predict row's Build of record ...
SERVED-SEAL VERDICT: PASS ...
WINDOW-CIRCUIT class 8,32,4,4,16: registered …
INFERENCE-ONLY accept …
WINDOW-COMMIT class 8,32,4,4,16: …
WINDOW-CIRCUIT-HELD class 8,32,4,4,16: the committed circuit.json less its run fields is the verifier's own ...
RECORD {"check": {}, "composed": "pending", ...
DEFERRED V3c's audit and the composite: this node has ... GB, under the 290 GB V3c's audit needs ...
SCRUB: 0 withheld into the serve dir's private/withheld/
RUN: served_verdicts.sh exit 0 after ... min
~~~

## 7. The send

While the serve runs, put the follow-on node's memory hold back, so its memory is free when the follow-on launches
(`PLACED ...` or `ALREADY HELD ...`; `follow.sh` removes it as the job starts):

~~~bash
bash place-hold.sh
~~~

Once the serve has ended with exit 0 and custody preserved:

~~~bash
bash send.sh RID
~~~

Expect `SEND r... of RID to vy-nebius-cpu-12: exit 0`, `STATE done` and `REPROOF SEND: PASS: the deferral's inputs and
the serve dir's re-proof inputs it omits copied to the follow-on's node, ..., and the window's sealed row dir, hashed in
D/private/rows.sha256`. Exit 79 means it is still copying: `bash runlog.sh h200-2 SEND_ID all` later.

## 8. The follow-on

On vy-nebius-cpu-12 (or `FOLLOW_NODE=vy-nebius-cpu-11`, which then must be the send's node too), 645 GB, waiting up to
3 h for its memory:

~~~bash
bash follow.sh RID 2>&1 | tee -a ~/odserve-monday-logs/follow.txt
~~~

Expect:

~~~text
CAPS .../settings/settings-monday.env b610e7e37a9bc461
TREE 9068336929e0... (= cursor/onsite-push1b-061a), NODE vy-nebius-cpu-12, D /workspace/incoming/netacct/RID, main.txt ... commits from ... (holds f0c4b1355fb), REPROOF 0, HOLD netacct-inf458-thu.json, caps V3C_AUDIT_TIMEOUT=13h CERTIFIER_TIMEOUT=4h CERTIFIER_PERIOD_TIMEOUT=1h COMPOSITE_TIMEOUT=6h
HOLD on vy-nebius-cpu-12: /workspace/ramlock/netacct-inf458-thu.json: ... removed      (or: absent)
SHIM: the workload syncs the tree's venv before reproof_follow.sh
research: run FRID launched on vy-nebius-cpu-12 ...
~~~

`REFUSED: infra's hold reserves less than MEM_GB` means the hold stays and nothing launched: ask infra on INF-458.
Check on it with `bash runlog.sh cpu-12 FRID follow`. Its first lines: `VENV: synced Python 3.12...`,
`DISK: ... GB free under /workspace/incoming/netacct/RID (FOLLOW_DISK_GB 650)`, `COINS: V3c's audit on
gate:research@204.12.178.84, peer vy-nebius-cpu-12, ...`, `CERTIFIER-KEY: .../certifier-key.json, the network
certifier's key ...`, `SITE-REGISTRATION: ...`, `ROWS: ... PASS`, `HOLD: ...`, `FOLLOW-ON on ..., 755 GB: ...`. At its
end:

~~~text
STATE done CUSTODY "preserved": true
V3C-AUDIT class 8,32,4,4,16 (records_audit.py exit 0): RECORDS {"accepted": true, ...
V3C-PERIOD (records_audit.py period exit 0): PERIOD {...
CERTIFIER-CHAIN (exit 0): CERTIFIER-CHAIN accepted on V3c's window ...
CERTIFIER-PERIOD (exit 0): CERTIFIER {"public": {"K": <about 3.6e12> …
THE NETWORK CERTIFIER {"rc": 0, "verdict": "accept"}
RECORD {"check": {}, "composed": "accept", ...
COMPOSITE CASE verity/sampled-proofs/one-stage/v1 accept
COMPOSITE CASE verity/sampled-proofs/v1 accept
COMPOSITE CASE verity/inference-only/v0 accept
COMPOSITE CASES: served_cases.py exit 0
COMPOSITE remote-network-no-pouw: VERDICT verity/remote-network-no-pouw/v0 accept ...
COMPOSITE: composed on ..., not the served node ...: from a prover-side copy of its serve dir's audit inputs ...
RECEIVED-SCAN KIND-SCAN clean …
PUBLISH: ...
~~~

Each step's wall time against its cap: `bash steps-time.sh follow RID FRID`. Its custody key lives 24 h and the job up to
30 h: a follow-on that outlives its key ends `"preserved": false` ("the custody key expired"); repair it with `research
fetch --all FRID`, then `research data custody FRID --publish`.

## 9. The serving cost

After the send (and the follow-on's launch), on the idle GPU:

~~~bash
bash stock.sh RID                 # the control's id CTL; it reads WINDOW_STEPS from RID's deferred.env on the node
bash runlog.sh h200-2 CTL all     # when done: CONTROL: served_verdicts.sh serve exit 0 ...
bash check.sh RID CTL             # odserve_check.sh of RID against CTL's aggregate
~~~

Post the cost under its label (Rules).

## 10. The node gate and the stop

Take Monday's own class window off h200-2 once the send passed and the serve's custody is preserved
(`REMOVE=RID bash inodes.sh`), then:

~~~bash
CHECK=1 bash stop.sh   # the gate alone
bash stop.sh
~~~

Expect `CUSTODY GATE exit 0 ...: ALLOWED r20261106-100208-h2n2 (done) ALLOWED r20261106-133857-h2n2 (done) ALLOWED
r20261106-134237-h2n2 (running, its pid ... gone) NODE GATE: N run dirs, bad 0`, then `TERMINATED computeinstance-...
(nebius compute instance stop)`, `STOP: vy-nebius-stop exit 0` and `STOP SENT ...`. `NOT STOPPED` names
`$LOGS/node-gate.txt`, whose `NO-CUSTODY` or `RUNNING` lines say which run: repair its custody or wait for it. Post the
stop on NET-190 and INF-527.

## 11. The record

When the follow-on has ended with custody preserved:

~~~bash
bash record.sh RID FRID
~~~

It reads the in-run `out/record/onsite.json` and the copy's pre-draw pin on the follow-on node. Expect:

~~~text
RECORD-VERDICTS: {... "sampled proofs": "accept", "the network certifier": "accept", ...}
RECORD-OF-RECORD: {"composite": "verity/remote-network-no-pouw/v0", "holds": true, ...}
RECORD-PINNED: registered_files {"compared": ["circuit", "public"], "holds": true, "read_by": [...], "why": null} ; auditor-listing holds: True
RECORD-CHECKED: auditor-listing {"holds": true, "why": ""}
RECORD-CHECKED: auditor-record {"holds": true, "why": ""}
RECORD-CHECKED: scan-state-pinned {"holds": true, "why": ""}
CERTIFIER-KEY: not open
K: <about 3.6e12> bits a phase; open under it: [...]
NETWORK-RUN / NETWORK-PERIOD: the network bound over the run and pooled per week, month and year (node, datacenter, world)
SAMPLED-PER-WINDOW / SAMPLED-PERIOD: sampled proofs' ε at δ and its pooled cells
MEMBER-OPEN: stated at .sampled_proofs.member
LAYOUT: the pin states one: yes ; θ names one: yes ; equal: yes
~~~

The checks it reads: **pinned** (`registered_files` holds, comparing circuit and public: the public file is the pin's
staged body completed, and θ's layout is the pin's); **auditor-listing**; **scan-state-pinned**; **auditor-record**;
**certifier-key** not open (where it is, the network certifier's verdict is NOT_ESTABLISHED); and **v2's member layout
from the pre-draw pin** (a5 wrote each member's k and n into `public-pin.json` at staging, before the draw; v2 reads its
layout from that pin, and the record refuses a θ layout that differs: `LAYOUT ... equal: yes`).

## What Monday's result must state

- **Condition 1**, the drawn unit's member, rests on `onsite_record`'s check of θ's layout against the pre-draw pin
  (`layout_pinned`), not on Lean: it is open in Lean (`MEMBER_OPEN`) until v2 derives each member's n from the
  registration, or the registration records each member's key and n.
- **Leakage**: the per-member layout discloses the partition (the member keys, the templates and their member counts,
  every member's n), and it counts in leakage beside the output bits, until the layout is committed under a hiding leaf.
- **The figure of record** is the conservative K, about 3.6e12 bits a phase (the CERTIFIER-PERIOD line and `K:`). It
  tightens to about 2,385 bits a phase only once the cut lemma is proved; until then the conservative figure is the
  result, with each period's pooled total and share stated at it.
- Per AGENTS.md, every figure is given per window and pooled over monitoring periods of a week, a month and a year, for the
  serving node, a datacenter (the census's size, a stated parameter) and the world, in bits and as a share, saying which
  parts are proved and which are scaled from one machine.
- The network verdict is not "served" until `verity/remote-network/v0` lands; the record's composite of record is
  `verity/remote-network-no-pouw/v0`, and the record claims no verified compute.

## What is in the kit

| File | What |
|---|---|
| `env.sh` | paths, nodes and pins every script reads; `tree_ok`, `follow_ip` |
| `tree.sh` | the run's tree at REF's head (WANT) |
| `settings/` | `settings-monday.env`; `settings_job_launch.sh` sends it with `settings_job.sh` and `front_check.py`, which check and install it |
| `inodes.sh` | the class windows on h200-2, and their removal once copied and in custody |
| `probe.sh`, `probe/` | `afa1_probe.sh` and `program_probe.sh`; then the preflight |
| `serve.sh`, `rr/` | `rr/launch.sh` (store run ids) with `remote_run.sh`, `stated.txt`, `circuit.txt`, `scrub.sh`, `scan-out.py`, `retrace-row.sh`, `keys/challenger-key.json` |
| `send.sh`, `follow.sh`, `place-hold.sh`, `follow/` | `reproof_send.sh`; `follow_launch.sh` (`--custody-ttl 24h`) with `reproof_follow.sh`, `reproof_copy.sh`, `reproof_files.py`, `circuits_out.py`, `certifier_key.py`, `challenger-key.json`, `shim/research` |
| `stock.sh`, `check.sh`, `check/` | the same-engine stock control and the cost check |
| `stop.sh` | the node gate (`"preserved": true` or `.fetched`) and vy-nebius-stop |
| `record.sh`, `runlog.sh`, `steps-time.sh`, `node_probe.sh` | read-only: the record's checks, a run's lines, step times, ssh |
