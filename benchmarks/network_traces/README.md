# network_traces

The network attester (`verity/protocols/accounting/network/`) on real served traffic: per-token timing traces for its
calibration, the network attester in front of a served endpoint as a relay of the egress record's one form (`network_attester`), the
zero-knowledge proofs of their proof blocks, and the served session's runbook. Each driver's docstring says what it runs, what it publishes and what its verifier receives.

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

- **Calibration and the honest path.** `record.py` records per-token timing of real vLLM serving, and
  the network attester's `attester/stall_probe.py` finds what makes a tick late.
- **Endpoints besides vLLM.** `demo_stub.py` stands in for the gate's serving endpoint, and `demo_endpoint.py` is
  the OpenAI endpoint in front of a decoder that isn't vLLM, whose scheduler and records the stub serves through.
- **The network attester as a relay.** Its `attester/network_attester_serve.py` runs
  `verity.protocols.accounting.network.verifier.network_attester` in front of any
  TCP server and checks what it published offline: `serve` releases only the egress record's id form by default, and
  `rehearse` names its traffic's (`--egress`: id with `--ids`, else bytes); `connection_rows_run.py`, beside it, is a
  run's connection rows (`--lengths 1|2 [--content P,O[,ROWS] ...]`: the schedule's flags, the publisher's keep of each
  execution phase's tables, or under `--lengths 2` of the network attester's word that it has none, their check, which
  is handed the SHA-512 of the network attester's public key and its session (`check --key-sha512 HEX --session HEX`),
  holds the run's deployment record to them and, with `--registrations FILE` and no problem found, writes what it was
  checked under and then each registration it held, and `cost`, which measures a close); `network_attester_chain.py` is
  the chain's reads of a relay run, and network attestation's `prover/network_attester_zk.py` reads the relay's bucket rows
  in the proof system.
- **Proofs in zero knowledge.** Each driver, in network attestation's `prover/`, proves its
  proof blocks with Flock `prove --zk` behind the Lean ZK server and holds them by the Lean
  `verity-verify-proof verify --zk`: `scan_zk.py` (the step-indexed scan), `scan_period.py` (the
  scan's answers held by `NetworkCheck.Period`). `leak_scan.py`, in
  `prover/` too, is the scan of what the verifier received that every one of them runs.
- **The endpoint step and K on a run.** `chain_record.py` (`endpoint` and `leak`, below), with
  `benchmarks/network_e2e/window_chain.py` (a served endpoint's Commit periods as one I1 v0 advice),
  `leak_guard.py` (the record's leak guard, below) and `prover/released_keys.py` (the keys a record releases, each declared public by name, below).
  `served_session.py` is the served session's runbook (below).
- **The weights' load:** the network attester's `attester/load_relay.py`, `load_bench.py` and `load_commit.py` (below).
- **A memory wipe's egress:** `wipe_gate.sh` and `wipe_egress.py` (below).
- **A second count:** `capture_flows.py` sums a packet capture taken beside a relay run by flow and direction and holds
  it against the relay's record of that run (below).

## The gate's load

The gate's `--shape h200` serves plain INT8 weights generated from the public seed `verity/cmp92/h200/weights/v1`
(`verity.protocols.accounting.memory.pous.prover.served_weights`). They are relayed to the H200 once per serving run and committed as they cross, through
two gate verbs on the auditor's machine:

- `vy-challenger-gate weights-write [--shape h200|tiny]` writes the shape's weights stream onto the auditor's store disk
  (`/var/lib/vy-challenger/store`) once. It keeps the stream only when its tensors' SHA-256 fold to the pinned manifest
  digests (`load_relay.STORE_DIGESTS`).
- `vy-challenger-gate load-relay --peer NAME --port PORT --sink-port PORT --program SHA512 --minutes M` (with
  `warden-rehearse`'s flags) is `warden-rehearse` with the load first. The store goes over 4 TCP streams, an SSH
  tunnel each, to `load_relay.py sink` on the H200's loopback; each value is held to its digest and committed under the
  reading program's domain as it crosses. The network attester's one signer on that link then signs the load's heads before
  its first execution phase, so one LMS key and one attachment carry the load and the execution phases. `load.json`
  (the registrations the server block's record pins) and `load-relay.json` (the relay's counts and the sink's
  acknowledgements) are in the run's `witness/`, which `fetch` gives the run's own lane key; of the load, `out/` holds
  nothing.
- The witness, every row's salts, is the sink's `--witness` file (0600) on the developer's host. The network attester's host
  never writes a salt.

## The wipe's egress

Memory accounting's streamed wipe signs no egress count; its check takes the count as evidence. The count over
its period [t0, t1) comes after, from a relay run that spans it: `wipe_gate.sh` runs that relay through the gate
(`warden-rehearse --scan --prompts` against the toy, `demo_stub.py serve`, on the peer's loopback), at a given UTC time
when asked, and fetches the run's bundle: with `witness/` into a 0700 directory outside the run dir, and to the run's
`$OUT` only the bundle less `witness/`. `wipe_egress.py BUNDLE --t0-ns T0 --t1-ns T1` then writes the `wipe-egress/v1`
record from what the gate published. Each execution phase that, with its verification phase, meets [t0 − 1 s, t1 + 1 s) counts at the
schedule's capacity, 2,621,440 bits an execution phase, with K that count and δ 0. The record is covered only when the
gate's check accepted the run, of which alone the gate publishes attestations, and every one of those execution phases
is attested. The record is all the script writes.

## The served session

`served_session.py` is the runbook of the served session of record, the network attester in front of a served model.
`plan` checks the request with this tree's gate and prints each run's command and the endpoint's flags. The gate is
`site.json`'s `gate`, whose host is the name of the environment variable that holds the challenger host's address
(`VY_CHALLENGER_1_HOST`): the session, the front (`attester_piece.py front`) and
`wipe_gate.sh` read it there, and without it and with no gate given (`--gate`, `CHALLENGER`) each ends naming it.
`session` runs the gate's scan rehearsal (`warden-rehearse --scan --prompts`) against the endpoint on the peer's loopback, by default
the H200 peer's `serve-openai` on :18782, inside the scan's class: prompts of 256 ids from a fresh secret seed (prompts
of the class's 1,024 reach the relay late), each answered greedily with at most 64 ids, ending at the base model's EOS
id 128001. It runs at a given UTC time when asked. The endpoint is started by hand (serve-openai directly, with the
flags `served_session.py`'s `HAND_START` lists), sealed without the attention interiors, with bounded
staging at its default, which that exclusion needs. Its `requests.jsonl` and `advice_trace.jsonl` hold request values
(prompt and streamed ids; prompt lengths, sampling, rows and emitted ids) and stay on the served node, so the
endpoint step runs there: `io` writes the kept network attester run's attested I/O to a 0600 file in the kept fetch's
directory, and `endpoint`, a run on the served node given that file over a private route, checks the endpoint's trace
against it in a 0700 copy whose `requests.jsonl` is rebuilt from `request_users.jsonl` (each engine request's
`engine_id` and its network attester user) and the attested I/O. Its art is one file, `report.json`, the step's public
document: the fields `chain_record.ENDPOINT_PUBLIC` lists, each with its public kind, and no other; the whole report
stays in `--keep`. It writes nothing unless `serve.json` shows the endpoint's flags.
The session's fetch, with `witness/`, stays in a 0700 directory outside the run dir, and `$OUT` gets only the bundle
less `witness/` and the session's outcome. `session --gate local` is the dry rehearsal: this tree's gate on a scratch
host, with `demo_stub.py serve` as the endpoint.

`attester_piece.py` is the network attester's piece of `benchmarks/one_stage/served_verdicts.sh`. Its `front`
(`CERTIFIER_FRONT`) runs before that run's requests: the gate's scan rehearsal sends one group of the execution phase circuit's
class (4 prompts of 24 ids, 4 ids out, `--window-per-group`) to serve-openai, all of it in execution phase 0, the
network attester at the schedule of that class's scan (`--scan-class w63k4`, 5 egress cells), so the engine closes it into the run's earliest execution phase, the one it audits,
and the gate's fetch is kept in the serve dir's `private/certifier`. A request's answer must be whole D·τ after its
admission, 2.7 s at the scan's 100 ms tick; for a slower endpoint `--bucket-ms` runs the scan at a longer tick, up to
4000 ms (108 s). Under that runner's `LOAD=session` the load is the caller's own, and neither the runner nor the gate
sends a request: `front --session` opens the gate's relay session in the rehearsal's place and returns once its relay
listens, the caller's command (`LOAD_CMD`, given the session's `relay.json` and the serve dir) sends the load to the
client address that file names, and `close` (`CERTIFIER_CLOSE`) waits for the session's end and keeps its fetch where a
rehearsal's is. The runner refuses a session without all three, and under any other `LOAD` each of them. A session's
records cover the connections that crossed the relay; that no other route reaches the endpoint is the placement's.
Its `piece` (`CERTIFIER_CMD`) writes `refused.txt` with the one text (`NOT_RUN`) and exits 3 before any step runs, and the
served window reads the network audit as pending: never an accept, and never a reject. The attester launches and serves;
its rows and its Lean program are checked by their own programs (`network-rows`, the network verifier's suites).
`kind_scan.py DIR` reads a piece's files field by field and reports each one that is of none of the four public kinds
and no attestation of the network attester's; `--received OUT` does so for what a verifier on another machine reads of
a served run's OUT. Serving commits no `SCAN_STATE`: V3c's window-commit step
(`benchmarks/one_stage/window_commit.py`) computes it after the serve from the network attester run the serve kept,
its certified ingress rows, and commits it among V3c's `registered`.

## A second count from a packet capture

`capture_flows.py` reads a capture taken beside a run through the relay, at a point that is neither the server's
process nor the network attester's, and counts the bytes between the two a second time. Its docstring says which bytes
are compared, why they must agree and what it refuses. It says "agrees" or the difference in bytes, and gives no
protocol's verdict.

**Taking the capture**, where every packet between the server and the network attester passes:

    tcpdump -i IFACE -U -s 130 -w run.pcap 'ip host SERVER'

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

**On the server's loopback**, where the server listens on its machine's loopback at `PORT` and the network attester
reaches it through a forward that ends on that machine, their bytes pass in the clear only there, between the
forward's end and the server's port:

    tcpdump -i lo -U -s 54 -w run.pcap 'ip host 127.0.0.1 and tcp port PORT'

- Every flow on a loopback has its address on both sides, so the address alone keeps them all, other programs' among
  them. The port clause keeps the flows of the server's port and nothing else: none of the server's own connections
  to another port is in the file.
- `-s 54` keeps no byte of any segment's content, where `-s 130` keeps each segment's first bytes, a request's start
  and a token id among them. A TCP packet's headers on a Linux loopback are Ethernet's 14 bytes, IPv4's 20 and TCP's
  20 or more, so never under 54, and the reader reads none of TCP's options: it counts the file as it counts one
  kept at 130. A packet with eight bytes of IP options or more is kept too short for its headers, and the reader
  refuses the file by that name.
- The two sides are one address, told apart by the server's port: the reader takes them as
  `--server 127.0.0.1:PORT --attester 127.0.0.1`.
- The filter and the length are for a server reached over IPv4. Reached over IPv6, the server leaves no packet in
  this file, since `ip host` keeps IPv4 alone, and the reader refuses nothing in a file of no packet: it counts no
  flow, says each direction differs by the record's whole count and pairs no connection, and nothing names the
  cause. That server's capture is `tcpdump -i lo -U -s 74 -w run.pcap 'ip6 host ::1 and tcp port PORT'`, read with
  `--server [::1]:PORT --attester ::1`. IPv6's header is 40 bytes where IPv4's is 20, so a TCP packet's headers are
  never under 74 and that file holds no byte of content either. A packet under an IPv6 extension header never enters
  it: the filter looks for TCP's header right after IPv6's.
- The count is of what the server's port received and sent. The link under the forward carries the forward's own
  bytes, which this capture doesn't hold and the reader doesn't read.

**On one machine**, `capture_loopback.sh` is the reader's own check, not a served run: the stand-in front, the network
attester and the load on one machine's loopback (`network_attester_serve.py rehearse --fake --watermark`), a capture of the
loopback beside them, then the reader. It captures at 54 bytes a packet, as the command above does, so neither its
temporary file of every TCP flow on the loopback nor the file it keeps of the run's two ports holds a byte of
content. The front and the relay share an address there, so the front's is given with its port, and the load's
connections to the relay are among the other flows. The run is in the bytes egress form, or with `EGRESS=id` at its
launch in the id form, id requests each answered an id event a token. It is a `research run` job where the job may
capture, which a pool node's job may not. It holds the reader's count to the relay's on the same bytes; it puts no
link between two machines under the capture.

**Reading it:**

    python benchmarks/network_traces/capture_flows.py --capture run.pcap --server SERVER[:PORT] \
        --attester ATTESTER --run RUN --out capture_flows.json

`RUN` is the relay's run directory (`network_attester_serve.py serve`); without it the capture is summed and listed only. The
result file holds every flow's sums, and the table is:

- The packets read and what was skipped, then the record's connections to the server and how many of its execution
  phases' records (each phase's ingress, egress and reasons) are attested under its key.
- A line for each direction between the server and the network attester: its flows, `packets` and `wire bytes` (every frame,
  with its headers, ACKs and retransmissions), `payload bytes` (above the TCP header, a retransmitted segment each time
  it was sent), `stream bytes` (each sequence number once, which the record's count is held against), `holes` (bytes
  of sequence numbers the capture missed) and `record bytes`. Then what the direction says: `agrees`; `differs by N
  bytes`, the capture's less the record's, where both counts are exact; `bound: …` with the range the difference lies
  in, where a count is only a bound (a hole, a flow without its SYN or FIN, a connection that failed or whose release
  was lost, the watermark's connection in no pair) and the range holds 0; `differs: …` where it doesn't.
- The pairing. No record holds a port, so connections and flows are paired by their two counts, as multisets: how many
  paired and how many of each stay unpaired. Two connections of equal counts aren't told apart, and "agree" is said of
  the sums and the multisets, never of one connection.
- Each connection whose counts are bounds, and the bytes the server answered on the watermark's connection, which are
  in no record and in no sum. Where that connection pairs with no flow, its answer can't be told from the server's
  other bytes and stays in the sum from the server: that direction is then a bound, lower by as many bytes as the
  capture counts from the server on the largest flow in no pair, and the line says so.
- Every other flow with a given address on one side, a line each: what the given side sent (`out`) and received
  (`in`). Machines are named `server`, `attester` and `other-N`, with their ports: no address is printed or written.

It exits 0 when it read the capture and 2 when it refused it.

## The endpoint step and the record's leak guard

`chain_record.py endpoint` binds the endpoint's own advice to the attested I/O read off the network attester's cells
(each Commit period opened to its registered root and rebuilt from the attested I/O), with the model's EOS ids
(`--eos`); where the endpoint's records stay on the served node, `served_session.py endpoint` runs it there.

**The record's leak guard** (`leak_guard.py`). Custody publishes the whole run dir, so a session's record
(`served_session.py session`) is written to a hold outside it first, and on every exit the guard runs on the run dir
and the hold. The guard moves to the quarantine (0700) every file that holds
a gate run's witness: one under a `witness/` directory or named as `leak_guard.json` declares, one holding an aligned
witness piece raw or as hex at any offset, an archive member that does, or an archive it can't read through. Only then
does it release the rest of the hold to `out/`. A find fails the run (exit 3), and `out/leak-guard.json` names each find
by its path alone. A file equal to one of the gate's public files (by sha256, so a gate's published bundle and its
members aren't scanned) is no find, nor is a piece those files hold, as they are or with their JSON written again. The
false positives are declared data, `leak_guard.json`'s `allow`, each entry written only after its find was read by
counts and digests. Its entries excuse the stall probe's cgroup counters and cgroup path in the runner's own records,
each by the record's file. An entry names its witness file and its records and nothing else: the guard refuses one
that excuses a piece by a key path. A value of the witness that other
files repeat by chance is written so that they don't: the network attester's usage record holds its CPUs as one string of
ranges (`runner_parts.cpu_ranges`), not as a list of integers, which an indented list of consecutive integers anywhere
would repeat. The guard reads pieces of 32 bytes and sees no shorter value of the witness, so what a step may write of
the witness is held by name where it is written: of the network attester's own check the attestations step writes
the gate's list of published counts and no other key (`network_attester_chain.COUNTS`). The
guard's cost is linear in the bytes it reads, and past its budget (`--budget-s`) it stops. A signal (before or during
the guard), a guard past its budget, or one that can't run releases nothing and withholds `out/`, `inputs/` and the logs
whole; a SIGKILL leaves the hold where it is, out of the record.

**The keys a record releases** (network attestation's `prover/released_keys.py` and `released_keys.json`). The guard's pieces
find a private value copied into the record; they can't find a value the witness never held as bytes (a count, a length,
a time). So the guard reads each JSON file bound for `out/` by key as well (`--keys`): a
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

A Commit period that carries a request across its open or close (`verity_vllm.pipeline.window_advice`, schema
`verity-vllm/window-advice/v1`) is bound with the Commit period before it: `chain.served_advice` joins them into one I1
v0 advice (`window_chain.join`). A carried request's record and state are the earlier Commit period's, as its
registered advice closed them (`window_advice.follows`), never the later one's own string's, so the advice's bits, and
N2's K, charge the carry once. A Commit period that carries a request in from one the run doesn't hold, or that leaves a
request with outputs still to keep and no next Commit period, is refused.

## The leak bound

`leak_bounds.py` is the one calculator of K, the bits a covert sender's secrets can number over a monitoring period.
It scales K in two ways. By requests, `verity bench certifier_term` prints the network attester's own terms at the
load of a node, a datacenter and the world, all a scope serves counted by each check's own rule (`leak_bounds.table`;
its rows, columns and inputs are in `benchmarks/certifier_term/README.md`). By links, the rest of this section: `per_link` gives K for
one link, one network attester on one uplink, certifying a request rate of one scan class (an `ingress_scan.Class`,
or a `WindowClass`'s `scan`); `datacenter` sums K over a datacenter's links; and `at_link_rate` sums as many links at
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
SUMMARY --out FILE`, with RUN a fetched network attester run and SUMMARY its scan's `summary.json` (its sample). It reads
`leak_bounds` on the run's parameters (`deployment.json`), its traffic (the requests `request.json` asked for and
`end.json`'s execution phases) and the scan's class (`scan_zk.CLASS`), and refuses a scan that left a proof block
unsampled. The end carries no count of the traffic: the gate takes an end as accepted only when its check counted the
request's own number of requests and it ruled every line of the run, all of which are then in `out/`. So `leak`,
which reads the end's `accepted` alone, refuses a run whose end isn't accepted, and a relay session, whose request
asks for no number because its clients are the peer's. FILE holds the widths, each
term, the links and a row for each monitoring period, scope and reading; `FILE.md`, which it prints, holds the tables.
A run's figures are its published Attempt's. The report's `d1` lists, per execution phase, whether its attestations, N2
and the scan accepted.

The terms, in the generator reading of the scan's statement (each proof block's registered rows its boundary values):
- The schedules and the fail cells, ⌈log2(cap_c + 1)⌉ each a connection (`capsTerm`, charged twice, as
  `#S_s ≤ ∏_c (cap_c + 1)`; the record's `terms.cap_per_request`). Here cap_c bounds `CertifierSchedule.cells c`, the
  egress cells of connection c's lane, which is egress_cells. The endpoint's slot count isn't a term of the network
  attester's schedule: it enters the compute audit's terms, which go into JointPeriodLive.
- The free inputs (`terms.free`): as registered, the tied audit's slot entries, NR·T / EVERY entries of NE bits
  (`ingress_scan.entry_bits`) an execution phase; or an entry's 33 + KO·IDB bits a request if `prog` is an input only at
  its anchored entries.
- No term for incorrect proof blocks with every proof block sampled.
- The jitter term is charged at D_j = 2 (`leak_bounds.JITTER_BOUND`): one bit a bucket, T bits an execution phase (the
  record's `terms.jitter_per_block`). J assumed at D_j = 2; at D_j = 1 not met (next-bucket host-state channel). J is
  one bound over every step (`NetTiming.JitterBounded`), so every bucket pays its bit. The network attester hands each bucket
  to a pacer that releases it at a time fixed before the bucket is made (`network_attester.NetworkAttester(paced=…)`, `network_attester_serve.py
  --paced-ms`), and a network attester on the host's clock always paces. Still, the host's state after the server's writes for
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
state isn't a wired audit's `SCAN_STATE` by root and domain (`stateTied`). `EVERY` proof blocks read each slot entry, which
one-stage's registration check lets through only with each lane's `prog<j>` named tied (`lane_kinds`), and its answer
then names them (`rests_on_period_check`); the check refuses, after that, by name, an execution phase whose scan's
answer names under that key anything that is no lane's `prog<j>` (`tiedInputWired`), since `wired` ties no other
Input. scan_zk's state tree is the execution phase's state (`ingress_scan.window_state`: leaf 0 the start's state, leaf t S_{t−1}),
each leaf but the first the o1 of the proof block that computes it and the sin of the next, both registered reads, so
a state tree that skips or alters a transition is refused at that proof block's o1; the block round's registration
names it as `scan/state`.
A scan proof block's public word, o0, is its ok bit. scan_zk commits a member's o0 rows under the verifier's public
salts (`ok_salts`, `ScanOk.salt`: row i's from its index alone), so the o0 root of a member whose every proof block is
ok is a constant of its template and count, and one execution phase's members are a function of the schedule's T
(`ScanOk.members`: `ingress-0`, `last` and `egress-0`; the start proof block, `start`, has no ok bit). The check computes those constants once a period
(Lean `ScanOk.consts`) and refuses, last, by name, an execution phase whose scan's registration
names a member's `<member>/o0` at another root, leaf count or domain, names none for it, or names another port ending
`/o0` (`held`): comparisons alone, no hash an execution phase. An ok bit is then the verifier's own computation from
public parameters, never a row the prover's registered file states. Two members of one template and one count (at
T = 3, `ingress-0` and `last`) are at one domain and root, and each is held to it. The table is one execution phase a
registration's (`--group 1`, the only group the rows form reads). A schedule of fewer than two ticks has no constants
(its one bucket would be the scan's first and its last, a member no scan is staged with), and the check refuses
every execution phase under one.
Tied (`NetworkCheck.Period`'s `progTied`, scan_zk's block round), the entries are a compute audit's outputs
and the scan's own free term is 0: the bits are that audit's free inputs (the stand-in's `x`), which the rows form's
K charges to the execution phase's term (`tiedAudits`). This tied audit of the slot entries (`read`, n, the egress rows'
output ids) is Lean's egress tie, not `io_tie`'s; K bounds the same; the registered reading's free term is those
entries.

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
  (`calibration/census/deployments.json`, one link a machine, `leak_bounds.datacenter_links`) and the world have
  machines: the world is the slowdown calculator's world fleet (`benchmarks/slowdown/data/reference.toml`, `fleet.world-h200`), the census's
  inference (`calibration/census/adversary.json`) at the fleet's serving MFU, one link a machine
  (`leak_bounds.world_links`).
- Every link has one network attester in front of all its machine's compute, serving at the run's rate
  (`leak_bounds.at_link_rate`, the scaling by links). A datacenter's and the world's links are each pooled at
  δ / links. A scope's requests past its links' rate are unattested, and no row counts them, so these rows are not
  the scope's K: `verity bench certifier_term` counts the network attester's terms over every request a scope serves.
- The weights are one copy of GLM-5.3 in FP4, its workload card's parameters at 4 bits each (`leak_bounds.WEIGHTS`),
  whatever the links and whatever checkpoint the run served, so every share names the copy it is of.

FILE has both readings of the free inputs. `prog` as registered is the record's K. `prog` only at its anchored entries
is what the generator-form lowering would give if it registers `prog` everywhere but its anchors, and is not the
record's number.

K also rests on these conditions:
- The generator reading of the scan's statement, in which each proof block's registered rows are its boundary values.
  That reading is a claim, not yet a locked guarantee.
- Each connection's limit is its lane's egress_cells, which the network attester holds a lane to and
  `NetworkCheck.Period` doesn't check.
- The scan proves execution phase 0. A monitoring period's figures scale its per-execution-phase and per-request terms at
  the run's rate.
- J at D_j = 2, which the jitter term assumes (above).
- An untied compute audit's (the loop's) free inputs don't enter K: its δ goes into JointPeriodLive's sum. The tied
  audit's (the `SlotEntry_v1` stand-in's) are the execution phase's free inputs above.

