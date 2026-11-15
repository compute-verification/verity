# network_traces

The network certifier (`verity/protocols/accounting/certifier/`) on real served traffic: per-token timing traces for its
calibration, the certifier in front of a served endpoint as a relay of the egress record's one form (`network_certifier`), the
zero-knowledge proofs of their proof blocks, and the network chain's run of record end
to end. Each driver's docstring says what it runs, what it publishes and what its verifier receives.

## Running it

- `verity bench network_traces` runs `run.sh`: one chunk of per-token vLLM timing traces on a leased GPU
  (`record.py`), as a job whose resources are in the script's header.
- Every other `.sh` here is a job on a node, its command and resources in its header (`capture_loopback.sh`, on a
  machine where the job may capture). The proving jobs build Flock's `flock-circuit`, the Lean `verity-verify-proof` and the
  Lean ZK server with `benchmarks/pouw/hidden_zk/job.sh`.
- Every run is published, with its report and outputs, as an Attempt in the evidence store
  (`tools/research/src/research/store/README.md`). Results are read from there, never from this README.
- The tests: `uv run tools/check/suites.py verity-network-traces` (`.agents/skills/testing/SKILL.md` says where).

## Entry points

`../../verity/protocols/accounting/certifier/` is "the certifier" below. Each driver's docstring is its reference.

| Driver | What it is |
| --- | --- |
| `record.py` | Per-token timing of real vLLM serving: the calibration and the honest path |
| the certifier's `device/stall_probe.py` | Finds what makes a tick late |
| `demo_stub.py`, `demo_endpoint.py` | A stand-in for the H200 demo's serving endpoint, and the OpenAI endpoint in front of a decoder that isn't vLLM |
| the certifier's `device/network_certifier_serve.py` | The network certifier as a relay in front of any TCP server, and the offline check of what it published (`serve`, `rehearse`) |
| the certifier's `device/connection_rows_run.py` | A run's connection rows, their check (`check --key-sha512 HEX --session HEX`) and `cost` |
| `network_certifier_chain.py`, the certifier's `prover/network_certifier_zk.py` | The chain's reads of a relay run, and the relay's bucket rows in the proof system |
| the certifier's `prover/` `n2_zk.py`, `n2_tie_zk.py`, `egress_zk.py`, `scan_zk.py`, `scan_period.py`, `leak_scan.py`, and `compile_slice_zk.py` here | Proofs in zero knowledge: each proves its proof blocks with Flock `prove --zk` behind the Lean ZK server and holds them by the Lean `verity-verify-proof verify --zk` |
| `chain_record.py`, `chain_record.sh`, `negatives.py`, `leak_guard.py`; the certifier's `prover/window_chain.py`, `prover/released_keys.py`, `prover/block_certifier.py` | The chain's run of record (below) |
| `leak_bounds.py` | The leak bound (below) |
| the certifier's `device/load_relay.py`, `load_bench.py`, `load_commit.py` | The weights' load (below) |
| `wipe_gate.sh`, `wipe_egress.py` | A memory wipe's egress (below) |
| `served_session.py`, `certifier_piece.py`, `kind_scan.py` | The served session (below) |
| `deviations.py`, `deviations.sh` | A deviating server (below) |
| `capture_flows.py`, `capture_loopback.sh` | A second count from a packet capture (below) |

## The chain's run of record

The run of record takes the network chain end to end on the gate's machines:

1. The network certifier signs every record with its LMS key, on an auditor machine, never on the developer's node.
2. It certifies each execution phase's records.
3. The client receives the certificates and keeps them (`certificates.jsonl`).
4. The consistency proof blocks are proved by sampled Flock `--zk` and decided by the Lean verifier: N2's
   (each execution phase's head checks and counts), and the scan's ingress and egress proof blocks chained by registered
   reads.
5. The verdict is the per-execution-phase acceptance that the certification period's leak bound reads.

**How it runs.** The gate runs on the auditor's machine: `sudo vy-challenger-gate warden-rehearse --scan --peer HOST
--port PORT --minutes 30` (the network certifier as a relay), then `status` and `fetch`. The fetched tarballs go to one
job, `chain_record.sh`, with the endpoint's records (`benchmarks/network_e2e/chain.py`'s `ENDPOINT`:
`advice_trace.jsonl`, `requests.jsonl`, `windows.jsonl` and `serve.json` from the endpoint's serve directory,
`CR_ENDPOINT`). The job runs `chain_record.py`'s steps, network-e2e's
binding and the two provers, and writes `out/report/report.json` and `report.md`:

| step | what it checks | its ε at δ | computed in the clear |
|---|---|---|---|
| custody | each gate run's files against the SHA-512s in its `end.json` | none: a check of files | the files' digests |
| certificates | each execution phase's two certified records, at the client, under the certifier's key and attachment; with `CR_DEPLOYED=1`, a key that isn't LMS at H = 20 is refused | none: signatures | every LMS signature of the client's kept certificates by the Lean (`NetworkCheck.TransferBlocks.refusals`, `audit.certificate_refusals` its reference), and the Lean's audit (`NetworkCheck.Records`, `audit.Audit` its reference) of the run's certified heads, under no sync (the network certifier's link declares none) |
| endpoint | the endpoint's own advice bound to the run (each Commit period opened to its registered root, rebuilt from the certified I/O off the network certifier's cells: `chain_record.py endpoint`) | none: in the clear | all of it |
| N2 | each execution phase's N2 proof block, proved `--zk` and decided by N2's verifier (`n2_zk.py`); not run on the advice profile, whose egress counts no N2 proof block holds | Selection All: no execution phase unsampled; a sample of k of n execution phases hides at most b incorrect ones except with probability δ (the report's `incorrect_at_delta`) | the registration, the sample, the certificates' public fields, the syncs, the pub row opened to the verifier's copy, the output word |
| scan | the scan's ingress and egress proof blocks of an execution phase, one registration, `verify --zk --registered` | the audit's `wrong_units` at the sample's sampling design | each proof block's output bit; the I/O tie stays outside the proofs |
| Lean | every proof of both steps is the verifier's own `verity-verify-proof verify --zk` run | per proof, ε_ks + ε_bind | nothing |

The job's environment picks what it runs: N2's sample on the relay path is `CR_N2_DRAW` (`n2_zk.py prove
--warden-run --draw K`, on the verifier's coins) and any other N2 command is `CR_N2_CMD`, the scan of the gate's own
rows is `CR_WARDEN_RUN` with `CR_SCAN_ARGS="--draw K --challenger gate:…"` (`scan_zk.py prove --warden-run`), and the
client's directory, with its `certificates.jsonl`, is `CR_CLIENT`, by default the network certifier rehearsal's own
`client/`. A step on a stand-in (the scan on its own synthetic run) never makes the report accepted, and a step that
fails or doesn't run says why.

**The record's leak guard** (`leak_guard.py`). Custody publishes the whole run dir, `inputs/` with it, so the job moves
each gate sent under `inputs/` to the quarantine once it is unpacked, writes every output (its own stdout and stderr
too) to a hold outside the run dir (`$Q/held`), and on every exit runs the guard on the run dir and the hold. The guard
moves to the quarantine (`CR_QUARANTINE`, by default `quarantine/RUN` beside the runs root, 0700) every file that holds
a gate run's witness: one under a `witness/` directory or named as `leak_guard.json` declares, one holding an aligned
witness piece raw or as hex at any offset, an archive member that does, or an archive it can't read through. Only then
does it release the rest of the hold to `out/`. A find fails the run (exit 3), and `out/leak-guard.json` names each find
by its path alone. A file equal to one of the gate's public files (by sha256, so a gate's published bundle and its
members aren't scanned) is no find, nor is a piece those files hold, as they are or with their JSON written again. The
false positives are declared data, `leak_guard.json`'s `allow`, each entry written only after its find was read by
counts and digests. Its entries excuse the stall probe's cgroup counters and cgroup path in the runner's own records,
each by the record's file. An entry names its witness file and its records and nothing else: the guard refuses one
that excuses a piece by a key path. A value of the witness that other
files repeat by chance is written so that they don't: the certifier's usage record holds its CPUs as one string of
ranges (`runner_parts.cpu_ranges`), not as a list of integers, which an indented list of consecutive integers anywhere
would repeat. The guard reads pieces of 32 bytes and sees no shorter value of the witness, so what a step may write of
the witness is held by name where it is written: of the network certifier's own check the certificates step writes
the gate's list of published counts and no other key (`network_certifier_chain.COUNTS`). The
guard's cost is linear in the bytes it reads, and past its budget (`--budget-s`) it stops. A signal (before or during
the guard), a guard past its budget, or one that can't run releases nothing and withholds `out/`, `inputs/` and the logs
whole; a SIGKILL leaves the hold where it is, out of the record.

**The keys a record releases** (the certifier's `prover/released_keys.py` and `released_keys.json`). The guard's pieces
find a private value copied into the record; they can't find a value the witness never held as bytes (a count, a length,
a time). So the guard reads each JSON file bound for `out/` by key as well (`--keys`, which `chain_record.sh` passes): a
file holds only keys its list in `released_keys.json` declares public by name, each with its kind, which is one of the
four things that may be public: protocol structure, a protocol output, a challenge or sample of the verifier's, or a
hiding commitment. A list is declared data beside the writers, one for each form of file, wherever a step copies it. A
key is named by its path in the file; `*` stands for a list's index, or, where the list's `maps` says so, for a map's
own keys, which are public ids of the form given there. No entry excuses a key: a key of none of the four kinds is not
declared, and its writer stops writing it. A file with a key its list doesn't name goes to the quarantine with the
guard's other finds and fails the record (exit 3). `out/leak-guard.json` names the file and how many keys; the key's
path and the fix are in the quarantine's report (and in `released_keys.py OUT_DIR`'s output, on a record already made),
since a writer may have made a key's name from a value. The report's `keys` counts the files checked by key, the JSON
files whose lists aren't written yet (`not_yet` in the declared data) by step, and the files with no keys by extension.
A JSON file under no list and no `not_yet` glob fails as an undeclared key does. A file with no keys (a binary, a log,
free text) can't be read this way and is only counted. The check is linear in the bytes it reads.

**The relay path** (`CR_PATH=warden`, `chain_record.NETWORK_CERTIFIER`). Every step reads the relay's own run (the gate's
`warden-rehearse`, its witness under `witness/`; `network_certifier_chain.py`), and a proxy rehearsal's run, whose records came
through the older certifier's proxy in front of the endpoint, is refused by name wherever it is given: that code is
gone, and kept records of its runs are refused by the form that stays. The certificates step reruns
the relay's own offline check on its witness (`relay_check.check`) and holds the public `certificates.jsonl` to the
certified lines. That check requires the session's certified start, so a gate run made by a device that certifies no
start is refused here: such a run is read by the chain of the tree its gate ran. The relay signs no clock certificate,
so none is read. The endpoint step binds the endpoint's own
advice to the certified I/O read off the relay's cells (`chain_record.py endpoint`, with `CR_EOS_IDS` the model's EOS
ids). N2 is the plain proof block on the relay's link (`n2_zk.py prove --warden-run`: both heads at the relay's grid and
parameters digest, no sync declared or charged). The I/O tie is proved on the relay's ingress (`n2_tie_zk.py prove
--class warden`, the report's `tie` step), and the scan is `scan_zk.py prove --warden-run`'s. Each step is held to the
one relay run the certificates step read, and every one is required.

A Commit period that carries a request across its open or close (`verity_vllm.pipeline.window_advice`, schema
`verity-vllm/window-advice/v1`) is bound with the Commit period before it: `chain.served_advice` joins them into one I1
v0 advice (`window_chain.join`). A carried request's record and state are the earlier Commit period's, as its
registered advice closed them (`window_advice.follows`), never the later one's own string's, so the advice's bits, and
N2's K, charge the carry once. A Commit period that carries a request in from one the run doesn't hold, or that leaves a
request with outputs still to keep and no next Commit period, is refused.

## The leak bound

`leak_bounds.py` is the one calculator of K, the bits a covert sender's secrets can number over a monitoring period.
It scales K in two ways. By requests, `verity bench certifier_term` prints the network certifier's own terms at the
load of a node, a datacenter and the world, all a scope serves counted by each check's own rule (`leak_bounds.table`;
its rows, columns and inputs are in `benchmarks/certifier_term/README.md`). By links, the rest of this section: `per_link` gives K for
one link, one network certifier on one uplink, certifying a request rate of one scan class (an `ingress_scan.Class`,
or a window class's `scan`); `datacenter` sums K over a datacenter's links; and `at_link_rate` sums as many links at
one link's rate as a scope has machines. Each comes in two forms, beside its share of one copy of the weights and its
ε. The terms are those of
`NetTiming.Guarantees.MonitoringPeriodLeakAccepted` (`security/Definitions/Certifier/Guarantees/MonitoringPeriod.lean`, proved in
`security/Proofs/Certifier.lean`) as `NetworkCheck.Period`'s rows form counts them, whose execution phases'
failures `NetTiming.monitoringPeriodLeak_le_sum` sums by a union bound (`security/Proofs/Certifier/MonitoringPeriodSum.lean`); the
pooled form is `NetTiming.Guarantees.PooledLeakAccepted`
(`security/Definitions/Certifier/Guarantees/PooledLeak.lean`). Both are about one device on one uplink. The sum
over links is our composition, which no Lean statement covers. It holds except where some link's statement fails, so a
datacenter at δ takes each link at δ / links, and each link pays its own incorrect proof blocks (`datacenter`'s
docstring says why pooling them by weight copy would be unsound).

**K per monitoring period on a run**: `uv run benchmarks/network_traces/chain_record.py leak --warden-run RUN --scan
SUMMARY --out FILE`, with RUN a fetched certifier run and SUMMARY its scan's `summary.json` (its sample). It reads
`leak_bounds` on the run's parameters (`deployment.json`), its traffic (the requests `request.json` asked for and
`end.json`'s execution phases) and the scan's class (`scan_zk.CLASS`), and refuses a scan that left a proof block
unsampled. The end carries no count of the traffic: the gate takes an end as accepted only when its check counted the
request's own number of requests and it ruled every line of the run, all of which are then in `out/`. So `leak`,
which reads the end's `accepted` alone, refuses a run whose end isn't accepted, and a relay session, whose request
asks for no number because its clients are the peer's. FILE holds the widths, each
term, the links and a row for each monitoring period, scope and reading; `FILE.md`, which it prints, holds the tables.
A run's figures are its published Attempt's. The report's `d1` lists, per execution phase, whether its certificates, N2
and the scan accepted.

The terms, in the generator reading of the scan's statement (each proof block's registered rows its boundary values):
- The schedules and the fail cells, ⌈log2(cap_c + 1)⌉ each a connection (`capsTerm`, charged twice, as
  `#S_s ≤ ∏_c (cap_c + 1)`; the record's `terms.cap_per_request`). Here cap_c bounds `CertifierSchedule.cells c`, the
  egress cells of connection c's lane, which is egress_cells. The endpoint's slot count isn't a term of the network
  certifier's schedule: it enters the compute audit's terms, which go into JointPeriodLive.
- The free inputs (`terms.free`): as registered, the tied audit's slot entries, NR·T / EVERY entries of NE bits
  (`ingress_scan.entry_bits`) an execution phase; or an entry's 33 + KO·IDB bits a request if `prog` is an input only at
  its anchored entries.
- No term for incorrect proof blocks with every proof block sampled.
- The jitter term is charged at D_j = 2 (`scan_answer.JITTER_BOUND`): one bit a bucket, T bits an execution phase (the
  record's `terms.jitter_per_block`). J assumed at D_j = 2; at D_j = 1 not met (next-bucket host-state channel). J is
  one bound over every step (`NetTiming.JitterBounded`), so every bucket pays its bit. The certifier hands each bucket
  to a pacer that releases it at a time fixed before the bucket is made (`network_certifier.NetworkCertifier(paced=…)`, `network_certifier_serve.py
  --paced-ms`), and a certifier on the host's clock always paces. Still, the host's state after the server's writes for
  the next bucket moves when a bucket reaches the client. The scan's period check (`scan_period.py`) passes the same
  bound as `NetworkCheck.Period`'s `jitter`.

Each entry carries a read bit (`ingress_scan.entry_layout`: an occupied slot no Program reads is refused). Bucket t
reads slot j's entry at row ⌊t / EVERY⌋ of lane j's entries, `scan/prog<j>` (`NetworkCheck.Period`'s `readsEntries`),
and ignores it outside due and occupied slots, so an execution phase's entries are its NR·nb boundaries'. Unless each
lane's `scan/prog<j>` is chunk j of a compute audit's `SlotEntry_v1` or `SelectEntry_v1` template's entries
(`<descriptor id>/c<j>/o0`, one member a lane), the read bit is the prover's word (an entry of read 1 and no outputs
passes the scan), so `NetworkCheck.Period` refuses the execution phase by name (`wired`). `wired` pins the template's
whole body by its closure digest (`entryPins`, `io_shell.SlotEntry_v1` at its circuit-check rows, and
`SelectEntry_v1`'s), so the block round's `prog<j>`, the o0 of scan_zk's `SlotEntry_v1` stand-in's lane-j member,
whose entries, read bit included, are the prover's, is refused there (scan_period.py). A slot's Program reads the
requests from its own registered `SCAN_STATE`, and the check refuses, next, by name, an execution phase whose scan's
state isn't a wired audit's `SCAN_STATE` by root and domain (`stateTied`). `EVERY` units read each slot entry, which
one-stage's registration check lets through only with each lane's `prog<j>` named tied (`lane_kinds`), and its answer
then names them (`rests_on_period_check`); the check refuses, after that, by name, an execution phase whose scan's
answer names under that key anything that is no lane's `prog<j>` (`tiedInputWired`), since `wired` ties no other
Input. scan_zk's state tree is the window state (`ingress_scan.window_state`: leaf 0 the start's state, leaf t S_{t−1}),
each leaf but the first the o1 of the proof block that computes it and the sin of the next, both registered reads, so
a state tree that skips or alters a transition is refused at that proof block's o1; the block round's registration
names it as `scan/state`.
A scan proof block's public word, o0, is its ok bit. scan_zk commits a member's o0 rows under the verifier's public
salts (`ok_salts`, `scan_ok.salt`: row i's from its index alone), so the o0 root of a member whose every proof block is
ok is a constant of its template and count, and one execution phase's members are a function of the schedule's T
(`scan_ok.members`: `ingress-0`, `last` and `egress-0`; the start unit, `start`, has no ok bit). The check computes those constants once a period
(`scan_ok.constants`, Lean `ScanOk.consts`) and refuses, last, by name, an execution phase whose scan's registration
names a member's `<member>/o0` at another root, leaf count or domain, names none for it, or names another port ending
`/o0` (`held`): comparisons alone, no hash an execution phase. An ok bit is then the verifier's own computation from
public parameters, never a row the prover's registered file states. Two members of one template and one count (at
T = 3, `ingress-0` and `last`) are at one domain and root, and each is held to it. The table is one execution phase a
registration's (`--group 1`, the only group the rows form reads). A schedule of fewer than two ticks has no constants
(its one bucket would be the scan's first and its last, a member no scan is staged with), and the check refuses
every execution phase under one.
Tied (`NetworkCheck.Period`'s `progTied`, scan_zk's block round), the entries are a compute audit's outputs
and the scan's own free term is 0: the bits are that audit's free inputs (the stand-in's `x`), which the rows form's
K charges to the execution phase's term (`tiedAudits`). The egress tie moves the free term from the scan's `prog` to the
tied audit's inputs; K is unchanged in what it bounds, and the registered reading's free term is those slot entries.

The two forms:
- The pooled form (`PooledLeakAccepted`) adds Fmax = ⌈ln(1/δ)⌉ incorrect proof blocks of b = NS + NV bits, and
  Fmax·log2(N + 1) (the record's `terms.fmax` and `terms.b`). The form is conditional: `hesc` and `hleak` await
  `SampledKSLive` and P2.
- The per-period form (`MonitoringPeriodLeakAccepted`) has no δ term with every proof block sampled.
- Both are proved: `NetTiming.SecurityProofs.MonitoringPeriodLeakAccepted` and `NetTiming.SecurityProofs.PooledLeakAccepted`.
- Beside either, ε = m·δ_scan is the scans' δ summed over the period's m execution phases and the links
  (`NetworkCheck.Period`'s `eps`), each scan at its δ (`leak_bounds.SCAN_DELTA`, `scan_zk --delta`'s default).

The scale:
- A week, a month and a year of one link, and of as many links as the census's reference datacenter
  (`benchmarks/census/deployments.json`, one link a machine, `leak_bounds.datacenter_links`) and the world have
  machines: the world is the slowdown calculator's world fleet (`benchmarks/slowdown/data/reference.toml`, `fleet.world-h200`), the census's
  inference (`benchmarks/census/adversary.json`) at the fleet's serving MFU, one link a machine
  (`leak_bounds.world_links`).
- Every link has one network certifier in front of all its machine's compute, serving at the run's rate
  (`leak_bounds.at_link_rate`, the scaling by links). A datacenter's and the world's links are each pooled at
  δ / links. A scope's requests past its links' rate are uncertified, and no row counts them, so these rows are not
  the scope's K: `verity bench certifier_term` counts the network certifier's terms over every request a scope serves.
- The weights are one copy of GLM-5.3 in FP4, its workload card's parameters at 4 bits each (`leak_bounds.WEIGHTS`),
  whatever the links and whatever checkpoint the run served, so every share names the copy it is of.

FILE has both readings of the free inputs. `prog` as registered is the record's K. `prog` only at its anchored entries
is what the generator-form lowering would give if it registers `prog` everywhere but its anchors, and is not the
record's number.

K also rests on these conditions:
- The generator reading of the scan's statement, in which each proof block's registered rows are its boundary values.
  That reading is a claim, not yet a locked guarantee.
- Each connection's cap is its lane's egress_cells, which the network certifier holds a lane to and
  `NetworkCheck.Period` doesn't check.
- The scan proves execution phase 0. A monitoring period's figures scale its per-execution-phase and per-request terms at
  the run's rate.
- The jitter term is T bits an execution phase, one a bucket: J assumed at D_j = 2; at D_j = 1 not met (next-bucket
  host-state channel).
- An untied compute audit's (the loop's) free inputs don't enter K: its δ goes into JointPeriodLive's sum. The tied
  audit's (the `SlotEntry_v1` stand-in's) are the execution phase's free inputs above.

The report names what it stands on beyond the theorem (`chain_record.STUBBED`, `NETWORK_CERTIFIER_STUBBED` and
`POOLED_CONDITION`) and what the theorem doesn't cover (its `uncovered`).

## The demo's load

The H200 demo serves plain INT8 weights generated from the public seed `verity/cmp92/h200/weights/v1`
(`verity.protocols.accounting.pous.prover.served_weights`). They are relayed to the H200 once per serving run and committed as they cross, through
two gate verbs on the auditor's machine:

- `vy-challenger-gate weights-write [--shape h200|tiny]` writes the shape's weights stream onto the auditor's store disk
  (`/var/lib/vy-challenger/store`) once. It keeps the stream only when its tensors' SHA-256 fold to the pinned manifest
  digests (`load_relay.STORE_DIGESTS`).
- `vy-challenger-gate load-relay --peer HOST --port PORT --sink-port PORT --program SHA512 --minutes M` (with
  `warden-rehearse`'s flags) is `warden-rehearse` with the load first. The store goes over 4 TCP streams, an SSH
  tunnel each, to `load_relay.py sink` on the H200's loopback; each value is held to its digest and committed under the
  reading program's domain as it crosses. The certifier's one signer on that link then signs the load's heads before
  its first execution phase, so one LMS key and one attachment carry the load and the execution phases. `load.json`
  (the registrations the server block's record pins) and `load-relay.json` (the relay's counts and the sink's
  acknowledgements) are in the run's `witness/`, which `fetch` gives the run's own lane key; of the load, `out/` holds
  nothing.
- The witness, every row's salts, is the sink's `--witness` file (0600) on the developer's host. The certifier's host
  never writes a salt.

## The wipe's egress

Memory accounting's streamed wipe signs no egress count; its check takes the count as evidence. The count over
its period [t0, t1) comes after, from a relay run that spans it: `wipe_gate.sh` runs that relay through the gate
(`warden-rehearse --scan --prompts` against the toy, `demo_stub.py serve`, on the peer's loopback), at a given UTC time
when asked, and fetches the run's bundle: with `witness/` into a 0700 directory outside the run dir, and to the run's
`$OUT` only the bundle less `witness/`. `wipe_egress.py BUNDLE --t0-ns T0 --t1-ns T1` then writes the `wipe-egress/v1`
record from what the gate published. Each execution phase that, with its verification phase, meets [t0 − 1 s, t1 + 1 s) counts at the
schedule's capacity, 2,621,440 bits an execution phase, with K that count and δ 0. The record is covered only when the
gate's check accepted the run, of which alone the gate publishes certificates, and every one of those execution phases
is certified. The record is all the script writes: what the wipe's reader does with one is
its own function, `wipe_monitoring.counted` (`verity/protocols/accounting/pous/verifier/wipe_monitoring.py`), and is stated there.

## The served session

`served_session.py` is the runbook of the served session of record, the network certifier in front of a served model,
and `certifier_piece.py` is the network certifier's piece of `benchmarks/one_stage/served_verdicts.sh`. Their docstrings
hold each step, file and flag, and that script's header holds `LOAD=session`. What a piece decides, word for word:

- A session's records cover the connections that crossed the relay; that no other route reaches the endpoint is the
  placement's.
- The answer is accept only when the network certifier's records pass `network-rows check` (each record held to its
  certified head, then the certifier program's rows audit, which refuses cells cut, clocked or ordered otherwise than the
  schedule), the chain accepted, every request of the audited window is a certified one inside that execution phase, and
  the Lean check accepts.
- Only the protocol rejecting the served window is a reject: with no fetch kept, a rehearsal the gate's check didn't pass
  (each request late, say) or a step that can't run, the piece refuses (`refused.txt`), which leaves the certifier
  missing.
- An accept's profile names its placement, as labels read from the run: the machine the certifier ran on, and the machine
  that ran each of the piece's steps; the piece establishes no `trusted-device` and gives no bound on that ground.
- It refuses the served node as the sample's peer, since the gate key and the certifier run's witness never go on the
  developer's machine.

## A deviating server

`deviations.py` puts a deviating process between the relay and the server's id front, plants a random string, and tries
six ways to send it out (`negatives.py` is the catalogue of what each check refuses). Its docstring holds the six
classes, the receiver, the checks, when a refusal counts, what a try keeps and why the tries stop. The dry run, against
the stand-in id endpoint:

~~~sh
uv run benchmarks/network_traces/deviations.py try --stand-in --out DIR
~~~

Against a served front, whose `HOST:PORT` is the value of an environment variable the command names:

~~~sh
FRONT=HOST:PORT uv run benchmarks/network_traces/deviations.py try --front-env FRONT --out DIR \
    [--cell-bytes B] [--tail S] [-- --window T --delay D ...]
~~~

And against a served front that listens on its own node's loopback, as a `research run` job on a node that holds a
tunnel key to it (`deviations.sh`: one ssh forward, then the tries, then a guard that fails the job if anything it
keeps holds an address; `research run --env` records none of the three values, and `REACH_ONLY=1` ends the job
once the front is reached, with no try):

~~~sh
PEER_AUTHORITY=USER@HOST[:SSH_PORT] FRONT_AUTHORITY=HOST:PORT TUNNEL_KEY=FILE \
    bash benchmarks/network_traces/deviations.sh [TRY-FLAG ...] [-- REHEARSE-FLAG ...]
~~~

Then the table again, once the server's record of the tries' sampled ids is there:

~~~sh
uv run benchmarks/network_traces/deviations.py table --out DIR --ids REQUESTS.jsonl
~~~


- Two deviations are not tried, and no row says anything of either:
  - a response with an id event more than the server's own. `length` only drops a response's last id event and never
    adds one.
  - a failure part-way through a response. `fail` withholds a response whole, so the relay fails its connection at
    the first egress cell; no try has a server send some of a response's cells and withhold the rest.
- The receiver reads a cell's bucket, never when inside its bucket it arrived. That channel is
  `benchmarks/certifier_timing`'s to measure: `verity bench certifier_timing run --attacks burst`.
- The command exits 0 when honest was accepted and each other class was refused by its check, or wasn't shown to the
  client; 1 when a class did otherwise, honest was refused, or a try wrote no check (it has no row, and the classes
  after it are still tried); and 2 when none of those and a row is shown neither way, as `tokens` and `length` are
  until the tie has the server's record.

## A second count from a packet capture

`capture_flows.py` reads a capture taken beside a run through the relay, at a point that is neither the server's
process nor the network certifier's, and counts the bytes between the two a second time. Its docstring says which bytes
are compared, why they must agree and what it refuses. It says "agrees" or the difference in bytes, and gives no
protocol's verdict.

**Taking the capture**, where every packet between the server and the network certifier passes:

~~~sh
tcpdump -i IFACE -U -s 130 -w run.pcap 'ip host SERVER'
~~~

- The filter keeps the flows with the server's address on one side and nothing else, so a shared link's other traffic
  never enters the file: one `or` clause for each further address (`ip6 host` for an IPv6 one, which `ip host` never
  keeps), and, on an interface captured by its name, `vlan and (…)` where its frames carry a VLAN tag and
  `vlan and vlan and (…)` where they carry two, which `vlan and (…)` doesn't keep. A capture of every interface
  (`-i any`) is a Linux cooked one, where `tcpdump` refuses a filter that holds `vlan`, and the filter without it
  keeps a frame under one tag and none under two. The reader walks the one tag a cooked v1 capture holds after its
  header, where the header's type names it (a v2 capture holds nothing of that tag); a second tag can reach either
  as four bytes before the IP header that no type names, and the reader refuses a cooked frame whose type says IP
  where no IP header starts by that name. It tells by the first byte there, so an unnamed second tag of priority 2
  with a VLAN id of 1280 or more (under IPv4's type), or of priority 3 (under IPv6's), its drop-eligible bit clear,
  is not refused by that name. The reader refuses a file that still holds a flow between two addresses it wasn't
  given; `--skip-foreign` counts and skips them. A flow the filter left out is refused by nothing: the record's bytes
  on it are a difference in each direction, and nothing names the cause.
- `-s 130` keeps at most each packet's first 130 bytes: every header and, where the headers are shorter (a common
  packet's are 54 to 66 bytes), the first bytes of its content, so the file holds the start of each segment. The
  reader needs the 130 for the longest headers it reads, and reads every length from the headers, never from how
  much was kept.
- Start it before the run's first connection and stop it after its last has closed: a flow without its SYN or its FIN
  is only a floor. `tcpdump` gets its packets a little after they pass, so stop it only once the file has stopped
  growing (`-U` writes each packet as it comes), or its last packets are lost.
- The file is pcap; `tcpdump -r IN -w OUT.pcap` rewrites a pcapng one. A capture point that writes each direction of a
  fibre to its own file gives two files: name both.

**On the server's loopback**, where the server listens on its machine's loopback at `PORT` and the network certifier
reaches it through a forward that ends on that machine, their bytes pass in the clear only there, between the
forward's end and the server's port:

~~~sh
tcpdump -i lo -U -s 54 -w run.pcap 'ip host 127.0.0.1 and tcp port PORT'
~~~

- Every flow on a loopback has its address on both sides, so the address alone keeps them all, other programs' among
  them. The port clause keeps the flows of the server's port and nothing else: none of the server's own connections
  to another port is in the file.
- `-s 54` keeps no byte of any segment's content, where `-s 130` keeps each segment's first bytes, a request's start
  and a token id among them. A TCP packet's headers on a Linux loopback are Ethernet's 14 bytes, IPv4's 20 and TCP's
  20 or more, so never under 54, and the reader reads none of TCP's options: it counts the file as it counts one
  kept at 130. A packet with eight bytes of IP options or more is kept too short for its headers, and the reader
  refuses the file by that name.
- The two sides are one address, told apart by the server's port: the reader takes them as
  `--server 127.0.0.1:PORT --certifier 127.0.0.1`.
- The filter and the length are for a server reached over IPv4. Reached over IPv6, the server leaves no packet in
  this file, since `ip host` keeps IPv4 alone, and the reader refuses nothing in a file of no packet: it counts no
  flow, says each direction differs by the record's whole count and pairs no connection, and nothing names the
  cause. That server's capture is `tcpdump -i lo -U -s 74 -w run.pcap 'ip6 host ::1 and tcp port PORT'`, read with
  `--server [::1]:PORT --certifier ::1`. IPv6's header is 40 bytes where IPv4's is 20, so a TCP packet's headers are
  never under 74 and that file holds no byte of content either. A packet under an IPv6 extension header never enters
  it: the filter looks for TCP's header right after IPv6's.
- The count is of what the server's port received and sent. The link under the forward carries the forward's own
  bytes, which this capture doesn't hold and the reader doesn't read.

**On one machine**, `capture_loopback.sh` is the reader's own check, not a served run: the stand-in front, the network
certifier and the load on one machine's loopback (`network_certifier_serve.py rehearse --fake --watermark`), a capture of the
loopback beside them, then the reader. It captures at 54 bytes a packet, as the command above does, so neither its
temporary file of every TCP flow on the loopback nor the file it keeps of the run's two ports holds a byte of
content. The front and the relay share an address there, so the front's is given with its port, and the load's
connections to the relay are among the other flows. The run is in the bytes egress form, or with `EGRESS=id` at its
launch in the id form, id requests each answered an id event a token. It is a `research run` job where the job may
capture, which a pool node's job may not. It holds the reader's count to the relay's on the same bytes; it puts no
link between two machines under the capture.

**Reading it:**

~~~sh
python benchmarks/network_traces/capture_flows.py --capture run.pcap --server SERVER[:PORT] \
    --certifier CERTIFIER --run RUN --out capture_flows.json
~~~

`RUN` is the relay's run directory (`network_certifier_serve.py serve`); without it the capture is summed and listed only. The
result file holds every flow's sums, and the table is:

- The packets read and what was skipped, then the record's connections to the server and how many of its execution
  phases' records (each phase's ingress, egress and reasons) are certified under its key.
- Every other flow with a given address on one side, a line each: what the given side sent (`out`) and received
  (`in`). Machines are named `server`, `certifier` and `other-N`, with their ports: no address is printed or written.

It exits 0 when it read the capture and 2 when it refused it.
