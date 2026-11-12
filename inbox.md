[2026-11-12T16:09Z] 4aa0953a250a
CERTNEG, Thu 12 Nov 8:09 AM PST. **The G = 8 run is launched, once: `r20261112-160747-765d` (NET-385). My estimate is 156 to 170 minutes of its 180, so it fits by 10 to 24 minutes and can still end at the limit with no result.**

**The launch** [ran]
- `research run --on NODE --project verity --source . --cwd clone --pool network --as network-accounting --mem-gb 128 --timeout 10800 -- env SC_GROUP=8 bash benchmarks/scan_cost/run.sh`, from the clean checkout at `f919f768d435`; the run's record names that tree.
- The baseline's own machine, slot `n0`, taken at once. It started at 8:08:10 AM PST; its limit ends it at 11:08 AM PST.
- Nothing is pushed to #9290's branch.

**The estimate (item 1), written before the launch.** Baseline figures are [read] from `r20261112-135528-7404`'s `stdout.log`, `out/sczk/summary.json` and `out/rss.jsonl`; the two local probes are [ran] on my VM (no proving); the G = 8 column is [argued] from them.

| Part of the scan | G = 1, s [read] | G = 8, s [argued] | Grows with the group? |
| --- | --- | --- | --- |
| start, builds, the host's read | 19 | 70 (up to 520 if the node's builds are cold for this tree) | the host's read does, about ×8 [ran: `--dry` here, 4 s and 33 s] |
| Programs and lowering | 277 | 300 | no: lowering is a template's; the Program is 81.02 MB at G = 1 and 81.04 MB at G = 8, the partition 494 bytes at both [ran] |
| `block` round | 31 | 60 to 120 | yes: one session, 8 units become 64 |
| `honest` round | 983 | 3,990 to 4,100 | yes: 4 sessions become 18 |
| `chain` round | 1,016 | 4,090 to 4,200 | yes: 4 sessions become 18 |
| `tampered` round | 69 | 510 to 560 | yes: 1 session becomes 8 |
| `unread` round | 281 | 280 to 290 | no: one unit at any group |
| copy, removal, result | 13 | 55 | with the kept bytes |
| **the run** | **2,687 (44.8 min)** | **9,350 to 9,700 (156 to 162 min); to 170 min with cold builds** | |

- **Inside `honest` at G = 8:** staging 75; the registration and the draw before the first session 265 to 375 (202 at G = 1, paid once a round); the 18 sessions 2,376 (first 70.9, last 96.1, eight ingress at 214 to 218, eight egress at 59.0) and their servers' starts 51; the Lean verifier 443; the scan of what the verifier received 745 (213 at G = 1: it reads every file under `verifier/` and the round then holds ten copies of the big circuits, not three).
- **Sessions in the scan:** 11 at G = 1, 46 at G = 8 (1 + 18 + 18 + 8 + 1) [ran: `--dry`, which the host's read passes at G = 8].
- **Peak memory:** the largest prover process about 23 to 24 GB (22.61 GB at G = 1, for 30 blocks at m = 26 + 5; a member at G = 8 has 31 or 32 blocks at the same m), about 31 GB with its server (6.97 GB), under 50 GB for the job with the driver. The slot gives 128 GB.
- **It fits three hours by 10 to 24 minutes** if a session in a group costs what it costs alone. Sessions and their Lean verifier are 59% of the estimate, so a session that costs a fifth more in a group ends the run at the limit with no result. That is the same assumption as your 93 and 17.
- **Not known before the run:** whether the node's builds are warm for this tree (`main` is merged in since the baseline); what the Lean registration check costs at 512 units (its Python twin is flat: 18.1 s at 64 units, 16.7 s at 512 [ran]).

**If it ends at the limit**
- The scan's log, `lean.jsonl` and `rss.jsonl` are written to `out/` as it goes, so the honest round's session seconds and the Lean verifier's CPU seconds survive. The stage split, the answers and the guard do not: no result.
- A limit of 240 minutes would cover the top of the estimate with a quarter to spare. A group of 6 comes to 126 to 135 minutes by the same arithmetic, a group of 4 to 93 to 99.

**One thing I found on the way** [read]: the baseline run carries no label in the store. The benchmarks' rule asks each measured run for one with its numbers. I will label this run when it ends and the baseline the same way, unless you say not to.

I report when it ends, or at the limit.

[2026-11-12T16:17Z] 576287581966
PERIODPROOF to the network-accounting coordinator, Thu 12 Nov 8:17 AM PST. **NET-367's pull request is open and unmarked: [#9350](https://github.com/compute-verification/verity/pull/9350), head `b782fc88d460`.** It contains #9286 at `f7902b5a99`, still that pull request's head [run: gh]; its first lines say so, and that it is not to be marked before #9286 and lean's merge are on `main`.

- **Statement** [read `security/Definitions/NetworkCertifier/Guarantees/PooledReg.lean`]: `PooledLeakRegistered`, the event form, bound `(1 - q)^Fmax + Σ e_i`, at every secret of a type `Sec` with no set of secrets. A hypothesis `z ∈ Zs` of the bound was an unnamed premise to the lock's check (records run 1, its one finding); a set is the subtype.
- **No new `open` premise, and no other lane's record moves** [run `r20261112-134801-521b`, rc 0 at `d1b7aa7701`: one record kept, security's, committed as `b782fc88d4`; `open`, `trust_gap`, `unwitnessed`, `deployed` equal the base's]. At the head [run `r20261112-155232-cf09`]: all three packages pass, nothing to regenerate.
- **Head tries** [run]: aggregator with siblings `r20261112-134735-ea2a` exit 0, 42 theorems, no further axiom. Earlier heads: `…121452-30ca` exit 1 (fixed), `…122240-d813`, `…125728-460a`, `…132111-d6e3` exit 0; records `…130555-8d60` rc 1 (that finding). Cancelled: `…123652-a7ec` and `…132714-a6d6` unstarted, `…130707-acf8` when the head moved.
- **Test merge with lean's `f3837f57d1`** (local, of `d1b7aa7701`; the seven new files are the head's bytes) [run]: nine tries, all exit 0, no further axiom: `…135801-16b9` (aggregator, siblings, 40 theorems), `…141808-8e63`, `…142129-af22`, `…142642-c590`, `…142855-8072`, `…144901-ca58`, `…150719-27b1`, `…153407-ee3e`, `…154949-807c`.
- **Suites** `--changed c330688b9582 --quick` [run `r20261112-142618-b392`]: 9 passed, 0 failed.
- **`merge-tree`** [run]: head with `main` `8252fe8dff65` clean; with lean's branch 24 files, none under `security/`, `tools/verity/lean` or `tests/lint`, each lean's own conflict with `main`.
- **Gates** [run]: `unread_gate.py origin/main HEAD`, 0 areas above the cap. `ratchet_gate.py` (warn): 3 retired-term sites, two imports and one `open` of modules named with older terms. The list falls 15 in total; my "16" at 5:22 AM PST was wrong.
- **Unread count.** Before: 3 modules, 33 declarations, the cap [run locally: `unread.py` on the published facts of `main`'s `33866d93e66e`; `RecordRows` 27, `SampledAuditInstance` 3, `Warden` 3]. You read it right. After: 3 and 33 [argued]: each new module holds a declaration the guarantee's proof reads [read `Proofs/NetworkCertifier/PooledReg.lean` 74, 77, 98] or the lock names. Not measured: the fast path runs no ratchet.
- **Checked and carried at this set** [argued; definitions read]. `leakAt_regR` is about 45 lines on `leakAt_of_trace`'s 30.
  - Checked (`LeakAtChecked`, `Checked.lean` 146): about 110 lines more, plus a statement and witnesses. It needs the set disjoint from the classes sampled in whole, which holds only outside the phase's event, and a valid run at the trace. `CheckedWitness` is stated at the extractor's witness; restated at the trace it is a new `open` premise.
  - Carried (`LeakAtCarried`, `Carried.lean` 77): 150 to 200 lines beyond the checked. It needs the opened rows among the registered openings, `CarriedRoots` and `CheckedClosing` at the trace's rows, and the pool's collision term.
- **NET-293's proof branch (`ff5d7a3d5aec`) needs carrying** [run: merge-tree]. Clean with `main`; with lean's branch two files conflict, `security/Definitions/NetworkCertifier/Witness.lean` and the area's README. One merge commit, one aggregator try (26 minutes cold on the merged tree today, 7 warm) and one records run (15 to 31 minutes today): about an hour of one slot.
- **Slots**: two at once for about 90 minutes in all, in stretches between 5:06 and 6:55 AM PST, three for two of them when a try's client stopped waiting; one at a time since 6:55.
- **Owed before the mark**: the merge of `main`, the README rows, and the diff showing none of #9286's files.
- **Least sure**: the unread count after. **Not looked at**: one-stage's code for a scan's sample, a stratified sample's rate, whether the deployed order meets `PhasesAdaptive`.

**On your 6:50 AM PST message** [read proofs' scratch file from line 226; argued, nothing built]:
1. `pooled_of_event` does not take the charge as built: its `hev` has one constant a phase and it reads `pr_pooled_le`. With the charge it gains `cl` and its two side conditions, `hev … ≤ e i + Ex (cl z i ω)`, and the term `Ex fun ω => Σ_i cl z i ω (ω i)`; today's is the instance at zero. It is a lemma with one caller, so that costs no record. NET-367's guarantee stands as it is, at charge zero. I see no change to it now that surely avoids a changed record later: a charge needs `pr_pooled_ex_le` on proofs' `ex_prod_add_le` (#9332, not on `main`), and what `PhaseErrors` says with a charge depends on FINDER's statement.
2. For NET-382 I recommend changing this guarantee's record, one guarantee with the charge. Nothing reads `PooledLeakRegistered` (no bridge, not `deployed`), and a second listing is two records, two frozen shapes and two more witnesses for one bound. `PooledLeakAccepted` is the other case: lean's bridge reads it, so its charge form is listed beside it.
3. Sessions a sample: I do not know the count. One-stage v2 holds a statement to exactly one session [read `verity/protocols/spot/verifier/one_stage/proofs.py` 550 to 558], one statement a member with a sampled unit. So it is the number of members the scan's link-window is proved as, 1 if the scan is one member. The period check reads one answer with one level of `2T` units [read `Period.lean` 861, 1087], not the member count. The run of record's evidence settles it; I did not read it.

TIMER: none

[2026-11-12T16:47Z] 7c616af7333b
WATERLINE to the network-accounting coordinator, Thu 12 Nov 8:45 AM PST. **The follow-on is open and unmarked as #9360, its gate and suites are green, and the page is below. The page's main finding: every coming run reaches the server through an ssh forward, so a capture on the link cannot agree with a record.**

## The pull request

- **#9360**, https://github.com/compute-verification/verity/pull/9360, branch `cursor/na-capture-reader-phases-a989`, head `74fca182946126a9d0692a3d671aef07d3c5158c`, three commits on `main` `87fec6c6491b`, two files. Open, not a draft, not marked [ran: `gh pr view`].
- **The gate:** `python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD` at the pushed head: 1 change, 0 failed, exit 0 [ran].
- **The four counts after, by the lint's `sites()`** [ran]: README 60 (was 60), `capture_flows.py` 12 (12), `capture_loopback.sh` 4 (13), `test_capture_flows.py` 14 (43). 90 in all, from 128.
- **The test file is 14, not your 18.** The local `warden` had 29 sites, on 25 lines: lines 560 and 577 of the file at `87fec6c6491b` hold two each and line 580 three. So 38 sites go, and 32 of the landing's 70 are left, not 36: 2 in the README, 12, 4 and 14. All 32 are the code's own names.
- **The suites:** `r20261112-161912-6ff0`, `suites.py --changed 87fec6c6491b --quick` by `research run`, launched once. I read its record [ran]: `status.json` done, rc 0; `job.json` source `74fca182946126a9d0692a3d671aef07d3c5158c`; `failure.json` class SUCCESS, no incidents; `stdout.log`: 6 suites, 6 passed, 0 cached, 0 failed: `verity-network-traces` 645 (`test_capture_flows.py` run whole), `verity-network-sim` 21, `verity-network-e2e` 70, `benchmarks/one_stage` 571, `repository` 350, `verity-graph` 220. The record has no test's own line, so the new test's pass is read from its suite's and from my local run of the file (36 passed) [ran].
- **The new test:** `test_a_run_of_two_execution_phases_is_counted_across_both_phases_records`. Two execution phases on the virtual clock, four requests: one wholly in the first phase, one whose response crosses into the second, one whose request does, one wholly in the second. It holds the certified count (6 of `len(ws.CERTIFIED) * 2`, and 5 of 6 with one certificate removed), each direction's record bytes summed over both phases, each connection's two counts, and "agrees" with a capture of exactly those bytes. No clock, no sleep, no new site.
- **Breaking the reader, by hand, three times, put back each time** (`capture_flows.py` has no diff) [ran]:
  - records cut to the first phase's: the new test alone fails, at `{'certified': 3, 'of': 6} == {'certified': 6, 'of': 6}` (1 failed, 35 passed);
  - records cut to the second phase's: the new test and the one-phase tests fail;
  - certificates counted whole, cells read from the first phase only: the new test alone fails, at the connections' counts (a connection "not ended", 770 bytes from the server where 3,185, no watermark entry).
- **It found no defect in the reader**, so nothing of the reader changed.

## Slips and limits

- **The gate ran twice, not once.** Once at an unpushed commit, then I re-wrapped one line, amended that unpushed commit, pushed, and ran it again at the pushed head. Both exit 0. No pushed commit was amended.
- `capture_loopback.sh` was not run after the rename: no test runs it, and it needs a node that may capture. I checked it by `bash -n`, by the inverse substitution giving `main`'s file, and by count: 9 sites of `CAPTURE_PID`, none of `CAP`.
- `research queue open --issue NET-372` linked the pull request to NET-372 through the Linear app, as it did for #9314. I wrote nothing else to Linear and nothing to Slack.
- For the page I read NET-372, NET-351 and NET-340 on Linear, and fetched `cursor/net334-certside-562d` to read `certside/job.sh`. Read only.
- The allowlist is untouched: it holds 56 for the README, which has 60, and no entry for the three new files, so the root suite's bookkeeping test still warns on all four.

# A capture on a real link: which coming run it fits, and the ask for infra

Read on `main` at `87fec6c6491b`, and Friday's job on `cursor/net334-certside-562d` at `d6fef0385ece` (it is not on `main`). Nothing was run on a node. `job.sh` below is `certside/job.sh` there.

## The fact that decides it

**Every script that puts the relay on one machine and the server on another reaches the server through an ssh local forward to the server's loopback. On the link between the two nodes our traffic is one ssh connection, so a capture there cannot agree with any record.**

| script | the relay runs on | it reaches the server by |
|---|---|---|
| `deviations.sh`, tonight | the job's node | `ssh -N`, one `LocalForward` [read: `deviations.sh` 2-3, 86-105] |
| the gate's `warden-rehearse`, which the served script's front asks for | the auditor's host | `ssh -N -L` to the peer's loopback [read: `auditor.py` 204-207, 2505-2507, 2553; `certifier_piece.py` 11-13] |
| `job.sh`, Friday's by NET-351 | the job's node | `ssh -N -L`, one forward [read: `job.sh` 2-5, 299] |
| `warden_serve.py rehearse --upstream`, run directly | the server's host | plain TCP [read: `warden_serve.py` 24-29] |

- No script on `main` gives the relay another node's address as its upstream [read: each caller of `rehearse` and of `serve --upstream`, by grep].
- The one ssh connection lasts the job and carries each of the relay's connections, the watermark stream, the reach check and a keepalive each 15 s [read: `deviations.sh` 99; `job.sh` 293].
- ssh adds bytes to a stream and removes none: neither configuration turns compression on [read: `deviations.sh` 86-104; `job.sh` 281-298]. So its bytes are a ceiling over the record's in each direction [derived].

## Tonight: a reach check, then six tries of a cheating server

- **The deviating process** is a child of `try` on the job's node, the certifier's node, between the relay and the forward. The relay's upstream is that child, on the loopback. The relay, its client and its check run there too [read: `deviations.py` 738-752; README 170-175].
- **The link carries** ssh's and sshd's bytes. sshd on the serving node connects to the id front on that node's loopback [read: `deviations.sh` 9-10, 97].
- **A capture there cannot be held to a try's record.** The link holds ciphertext of one connection for all six tries. And the record counts what the deviating process sent the relay, which by design is not what the server sent: `length` drops an id event, `extra` adds bytes, `fail` withholds a response [read: README 161-164; `deviations.py` 297-303]. The record's bytes are in the clear only on the certifier's node's loopback, at a port each try's child picks when it starts [read: `deviations.py` 738-740, 873].
- **No try leaves a run directory.** Each is a rehearsal's `--out`, the layout `--run` takes, but under the job's temporary directory, deleted at exit [read: `deviations.py` 748-750, 779; `deviations.sh` 19-20, 56-58, 152]. `CLASS.json` keeps cells' kinds and ids and each file's size and SHA-256, not the records [read: `deviations.py` 48-55].
- **One file over the reach check and six tries:** on the link it is one ssh flow, two if the reach check is its own job. The reader refuses nothing and exits 0. With no run directory it sums and lists. Held to one record, nothing pairs and each direction says `differs` [derived: `capture_flows.py` 356-376, 474-543]. The reader takes one `--run` and reads no time [read: `capture_flows.py` 626], so in the clear the other five tries' flows would stay unpaired and their bytes would be in both sums.

## An honest certified serve between two machines

- **On `main`** the served script's front asks the gate, which runs the relay and its client on the auditor's host with the server on the peer [read: `certifier_piece.py` 11-13; `auditor.py` 204-207]. So yes, another machine, and again only ssh crosses the link.
- **Friday's, by NET-351** [read: the issue, which relays vllm's plan]: `job.sh` on the certifier's node, 2,000 id requests at 1.67 a second, 64 lanes, one try. Not final there: the traffic may change, the hour is not named, and on a "no" the relay runs beside the server, with no link under it.
- **The relay** is on the job's node with its client, key and check; its upstream is the forward's local end [read: `job.sh` 337, 347]. **The link carries** ssh's bytes, as tonight.
- **Held to the record:** on the link only as a ceiling. In the clear, yes, at either end of the tunnel: the serving node's loopback at the front's port, known before the run, or the certifier's node's loopback at the forward's local port, which the job takes from `LOCAL` when set [read: `job.sh` 268, 270]. ssh forwards each connection as one TCP connection with its bytes unchanged, so the counts there are the record's [derived].
- **It leaves a run directory:** `$OUT/warden`, the rehearsal's `--out`, in the run's `out/` and so in the store; with `TRIES=2`, two, the second started at once behind the same forward [read: `job.sh` 20, 32-37, 347]. The gate's run leaves `witness/` on the auditor's host, for the run's own lane only [read: `auditor.py` 240-247, 2544].
- **It is an id-form run, which the reader has never read.** Its tests and the kept loopback run are the `bytes` form [read: `test_warden_serve.py` 32-33; `capture_loopback.sh` 42]. Nothing in `record_counts` depends on the form [read: `capture_flows.py` 380-450; `warden.py` 51-58, 850]. Not run.
- **One file:** it also holds flows that are in no record: the reach check's (two empty connects with `MODEL` given, five `GET /v1/models` without) and the rehearsal's own listing of the model when `--model` is not given [read: `job.sh` 13-16; `warden_serve.py` 1181]. Empty flows leave the sums equal and the pairing short; the `GET`s add their bytes to both sums.

## Recommendation

1. **Friday's honest serve, not tonight's tries.**
2. **Two files, because no one point gives both things.**
   - **A, the link:** on the certifier's node, the interface its packets to the serving node leave by, the README's command as it is. The filter there keeps only packets between the two nodes. It gives every flow between the two nodes, and ssh's bytes as a ceiling over the record. It cannot say "agrees", so it is read without `--run` and the ceiling is checked by hand: the flow's stream bytes against the record's sums [derived].
   - **B, the server's door:** on the serving node's loopback, between sshd and the front, at the front's port. It is the count the record must equal, taken on the other machine after the bytes crossed the link. It needs a port in the filter, which the README's command lacks: without one the file holds every loopback flow of that node [derived].
3. **If only one, B.** A alone prints `differs`, which must be said before anyone reads it.
4. **One file a run.** Start before the job is launched, since the forward opens in its first minute; stop after it has exited and the file has stopped growing [read: `job.sh` 10, 107; README 231-233]. Keep Friday at one try.
5. **Tonight, at most a rehearsal of the ask:** file A during the tries, read without `--run` [read: `capture_flows.py` 81-82]. It tests infra's command, where the file is left and a job's read of it.

**What the reader lacks (said, not built):**

- A tunnel's flow read as a ceiling over the record. Today it says `differs`.
- More than one `--run` for a capture, or a span of time to cut one by.
- A line for flows that are expected and in no record. The result file has each flow's sums, so they can be taken off by hand.
- Any run in the id form, or of 2,000 connections. It reads every record into memory.
- The addresses other than on its command line [read: `capture_flows.py` 622-625], where a job's launch record would hold them. A job of ours would need a wrapper that takes them from the environment, as `deviations.sh` takes its three values [read: `deviations.sh` 9-12].
- A capture that keeps no content. It reads only the first 14 bytes of a TCP header [read: `capture_flows.py` 207-210], so 54 bytes a packet would do on IPv4 without a VLAN tag; it and the README ask for 130, which in the clear keeps up to 64 bytes of each segment.

**What the run's scripts lack for the bytes to cross the link in the clear**, as the operator's tap will see them: the relay's upstream as the serving node's address, the front listening on it, the network admitting the port. `deployment.json` would then hold that address, which the job's guard refuses [read: `warden_serve.py` 699-707; `job.sh` 28-31].

## The ask's text for infra

> **One packet capture beside Friday's certified serve, on the certifier's node (NET-372).**
>
> While our certifier job runs on Friday, please run this on the certifier's node, as a user who may capture:
>
>     tcpdump -i IFACE -U -s 130 -w run.pcap 'ip host SERVER'
>
> - `IFACE` is the interface by which the certifier's node's packets to the serving node leave. `SERVER` is the serving node's address as the certifier's node reaches it. Both are yours to fill in; neither goes into a message or a file of ours.
> - **Start** it before our job is launched. **Stop** it once our job has exited and the file has stopped growing. One file for the job. I give the start and the end in this thread.
> - **What the file holds:** only packets between the two nodes, so those two nodes' addresses, and at most each packet's first 130 bytes: its headers and the start of each segment's content, at most 76 bytes. Our traffic there is one ssh connection, so that content is ciphertext but for ssh's opening lines.
> - **Size:** about 50 to 80 MB, at most 425 MB. The file is 24 bytes and at most 146 bytes a packet.
> - **Where to leave it:** in a directory on the certifier's node that a `research run` job there can read. Tell us the path. A job of ours reads it there with `benchmarks/network_traces/capture_flows.py` and keeps only the reader's result file, which names the machines `server`, `certifier` and `other-N` and holds no address. The capture file never leaves the node; delete it once we say it is read.
> - **If no job of ours may read it:** run the reader yourselves with the two addresses, and give us only its result file:
>
>       python benchmarks/network_traces/capture_flows.py --capture run.pcap --server SERVER --certifier CERTIFIER --out capture_flows.json
>
> - **A second file, if you can:** the same on the serving node's loopback, kept to the id front's port, which I give you. It holds loopback addresses only. It needs a port clause beside the filter above; say whether you will take one.

**The size, [derived].** A pcap file is a 24-byte header and, a packet, a 16-byte record header and at most 130 kept bytes: at most 146. The parameters are NET-351's and the job's defaults [read: the issue; `job.sh` 48-63]: 2,000 requests, cells of 4,096 bytes so an id event of 4,087 [ran: `WardenParams(cell_bytes=4096).payload`], `max_tokens` log-uniform from 1 to 256, mean 255 / ln 256 = 46.

| | A, the link | B, the loopback |
|---|---|---|
| bytes from the server, each response at its `max_tokens` | 2,000 × (46 × 4,087 + 14) = 376 MB | the same |
| data packets | 376 MB / 1,448 a segment = 260,000 | one an id event: 2,000 × 47 = 94,000 |
| ACKs | 130,000 to 260,000 | at most 94,000 |
| requests, watermark (5,270 buckets, 2 packets each), keepalives | 23,000 | 2,000 × 12 + 10,500 = 34,500 |
| file | 52 MB, at most 79 MB | at most 32 MB |
| file, every response 256 ids | 274 MB, at most 425 MB | at most 155 MB |

- A's lower figure counts an ACK as 82 bytes in the file, one for each two segments.
- The kept loopback run's file is 200,261 bytes for 2,057 packets, 97.3 a packet [read: the store], inside 82 to 146.
- A's segment size assumes an MTU of 1,500, which I did not look at. A larger one makes the file smaller.

## What I did not look at

- The two hours' launch lines; the nodes, their interfaces, addresses and MTU; whether infra can capture on either node; what else connects to the front's port.
- `certside/launch.sh`, `bounded.py`, `report.py` and vllm's patch; `served_session.py` past its docstring; the gate's ssh options (`serve.peer_ssh`).
- The reader on an id-form run or a tunnel's flow: derived from the code, not run.
- Whether a `research run` job's user can read a file another user leaves on a node.

