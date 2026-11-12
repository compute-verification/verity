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

[2026-11-12T16:56Z] d522d785d604
PERIODPROOF to the network-accounting coordinator: NET-382's statement with the charge. Thu 12 Nov, 8:56 AM PST.

Marks: [ran] with the run, [read] with the file, [argued].

## Where it stands

- Branch `cursor/network-accounting-pooled-charge-3e7a`: #9350's head `b782fc88d460` + a merge of proofs' marked head `a555863cf8d5` (merge commit `1458778a29`, clean) + my commits. Pushed. [ran]
- **The guarantee with the charge elaborates**: statement at `0cdd10bc42f14f688ac57754602f0e9e51d6458b`, try `r20261112-163025-c954`, lean exit 0. [ran] Its doc text as it stands is at `bff4d5738c3362c2004e391baa907be096c6942b` (docstrings only after `0cdd10bc42`); that file was built again as a dependency in `r20261112-164305-27ed`, whose dependency build completed. [ran]
- File: `security/Definitions/NetworkCertifier/Guarantees/PooledReg.lean`, whole, below. Its two premises' definitions are in `security/Definitions/NetworkCertifier/PooledReg.lean` (`ChargeAdaptive`, `PhaseErrors`), also below.
- `main` at 8:35 AM PST was `87fec6c6491b`: none of #9286 (`f7902b5a9958`), lean's merge (`f3837f57d113`), #9332 (`a555863cf8d5`) is on it. [ran]

## What the new term is, in plain words

The bound was `(1 - q)^Fmax + Σ_i e_i`. It is now

    (1 - q)^Fmax + Σ_i e_i + E_ω[ Σ_i cl z ω i (ω i) ]

- The third term is an average over the monitoring period's coins `ω` (every execution phase's prover tape and verifier coins, uniform) of a sum over the phases: each phase's **charge**, taken at the coins that phase drew.
- A phase's charge `cl z ω i` is a function from the phase's own coins to a real number. It is a parameter: whoever applies the theorem names it. The premise `PhaseErrors` now asks that the phase's error (the committed form's `εregX` plus the registered openings' binding term, as on #9350) is at most `e i` **plus the charge's average over the phase's own coins**.
- `e i` is one number a phase, fixed before the period, and the `e i` add. The charge may read the secret and the earlier phases' coins, and the period pays it once: if along every run of the period the charges sum to at most `Q`, the third term is at most `Q` (`pooledLeakRegistered_budget`, an unlisted lemma). [ran: `r20261112-164833-fd7b`]
- What it buys: something a prover can move between phases after it sees the earlier coins (its calls, say) had to be bounded in each phase by its largest value there, and those add over the `m` phases. With the charge it is counted at its total over the period.
- At the charge zero the statement is #9350's, as an unlisted lemma (`pooledLeakRegistered_zero`). [ran: the same run]

## The two side conditions, and of whom they ask

1. **`hcl : ChargeAdaptive cl`**: phase `i`'s charge, as a function of the phase's own coins, is the same at any two histories of the period's coins that agree before phase `i`. It asks this **of the charge, so of whoever names one** (proofs' phase bound, when it lands). It is the condition `PhasesAdaptive` puts on a phase's statement, strategy and openings, and it follows from `PhasesAdaptive` for any charge that reads the period's coins only through those three (`chargeAdaptive_of_phases` [ran: the same run]), a prover's calls at this phase's statement, say. So it asks nothing new of the deployment beyond what `PhasesAdaptive` asks, and like it, it is proved of no check. It is a new `domain` premise in the lock, concluded by the honest witness.
2. **`hcl0 : ∀ z ω i c, 0 ≤ cl z ω i c`**: the charge is never negative. Of the charge, so of whoever names one. A plain range, not a named premise.

One hypothesis changed with them: **`he0 : ∀ i, 0 ≤ e i`** is now written (a plain range). On #9350 it followed from `PhaseErrors`, since an error is not negative; with a charge beside `e i` it no longer follows.

## What the statement does not say

- **It names no charge**, and nothing in the tree shows of any charge but zero that it bounds a phase's error. `PhaseErrors` is a premise. A phase's bound with a charge that is not zero is proofs' to build. No figure is written anywhere and none moves. [read: my diff]
- **The error on `PhaseErrors`'s left is unchanged**: `εregX + Pr[BopsX]`, the square-root form, one number a phase, fixed before the phase's coins. The statement turns no part of it into calls. The fork term is untouched and is still one a phase. [read: `Definitions/NetworkCertifier/PooledReg.lean`, `PhaseErrors`]
- **The third term is an average, not a worst case**, and the statement gives no bound on it: that needs a bound on the charges' sum, along every run or on average, which the one who names the charge must supply.
- **Everything #9350 did not say still holds**: no secrets counted; `RateAt` and `PhasesAdaptive` are hypotheses proved of no check; the five `open` premises are the same five, none new; it is not the deployed system's bound over a monitoring period. [read: `security/lean-audit.json`, the guarantee's `shape` text, which I extended by one sentence on the charge]

## The lock, as edited (the records run is still to come)

- One guarantee, no second listing: `PooledLeakRegistered`'s own record changes. [argued until the records run]
- `NetTiming.PhaseErrors` (`domain`) changes, since it takes the charge; its why is rewritten.
- **One new `domain` premise, `NetTiming.ChargeAdaptive`**, with its one-line why. The honest witness concludes it: there `e` is zero and the charge is the instance's own error at every coins, a charge that is not zero, so the witness reads the new term and not only the old bound.
- The two witnesses' recorded statements change (they are listed).
- **No new `open` premise.**

## Proofs: written and tried, but for the witnesses

- `pr_adaptive_add_ex_le` (in `Proofs/NetworkCertifier/Pooled.lean`) is `ex_prod_add_le` at the phases' accepting events; `pr_adaptive_add_le`, `pr_adaptive_le` and `pr_pooled_le` are its instances at the charge zero; `pr_pooled_ex_le` is the period's bound with the charge. Try `r20261112-163550-991e` on `PooledLeak.lean`: lean exit 0, none of its 11 theorems uses a further axiom; `Pooled.lean` built as its dependency. Commit `d6011c555af2`. [ran]
- **No induction over a period's coins is ours.** The area no longer reads `prCoin_forall_sum_le_pow` or `prCoin_forall_prod_le`; its one step is `ex_prod_add_le`. [read: `rg prCoin security/Proofs/NetworkCertifier/Pooled.lean PooledLeak.lean` finds nothing] Two inductions over the phases' index remain in the area, and neither reads a coin: `prod_min_add_le`, an inequality of real numbers (`∏ min(1, a_i + e_i) ≤ ∏ a_i + Σ e_i`), on `main` today, moved from `PooledLeak.lean` to `Pooled.lean` unchanged, which proofs' scratch calls; and the count half's `encard_chain_le` (`SampledCharge.lean`, a count of records, untouched). If soundness's condition covers the first too, it belongs in proofs' file: tell me.
- `pooledLeak_charge` is `PooledLeakAccepted`'s statement with the charge, an unlisted theorem with its hypotheses written out, and `pooledLeak : Guarantees.PooledLeakAccepted` is it at the charge zero, statement unchanged. This is how I read ruling 2 for `PooledLeakAccepted`: it cannot follow from `PooledLeakRegistered`, which is the narrower statement, so "the statement with the charge" it is proved from is its own form with the charge. No `def` is added for it, so nothing is listed twice. It built as a dependency of the run below. [ran]
- **`pooledLeakRegistered : Guarantees.PooledLeakRegistered` is proved with the charge**: try `r20261112-164833-fd7b` on `Proofs/NetworkCertifier/PooledReg.lean`, lean exit 0, none of its 5 theorems (`pooled_of_event`, `pooledLeakRegistered`, `chargeAdaptive_of_phases`, `pooledLeakRegistered_zero`, `pooledLeakRegistered_budget`) uses an axiom beyond the three. Commit `a608e0c7dc6a`. [ran]
- The witnesses carry the charge. First try of their file: `r20261112-164305-27ed`, exit 1, one error (a lemma's module not imported in the witness's `example`), fixed; its second try is running as I send this. [ran] Not yet run: the aggregator with its siblings, the records run, the suites, the gates.

## The two obligations, for the issue you file and for #9350's body

- **`RateAt`.** At every execution phase, for the statement `s` the phase's verifier parses and the partition `Pt` it reads for `s`, the sampling design the statement carries must miss every set `B` of `Pt`'s proof blocks with probability at most `(1 - q)^|B|` (`s.design.Pa Pt B ≤ (1 - q)^|B|`), at one `q` for the whole period. For a uniform sample of `kd` of the `n` proof blocks that holds at any `q ≤ kd/n` (`NetTiming.rateAt_subset`, proved), so the deployed sample would have to be shown to be that design with `kd/n ≥ q` at every phase; for a scan's stratified sample, `k_s` of the `n_s` proof blocks of each stratum, it would hold at any `q ≤ min_s k_s/n_s`, the worst stratum's rate and not the pooled `Σ k_s / Σ n_s`, and no lemma proves that. Neither is shown of the deployed check: it accepts one level drawn `subset:k` or stratified and reports `drawn` and `units` (`factsOf`, `verity/protocols/certifier/verifier/lean/NetworkCheck/Period.lean` 1057 to 1065), and nothing ties that accepted profile to `s.design` at any `q`. [read: that file; `PooledRegSample.lean` 48 to 69] [argued: the stratified `q`]
- **`PhasesAdaptive`.** For every secret, execution phase `i`'s statement, its prover's strategy and its registered openings must be the same at any two histories of the period's coins (each phase's prover tape and verifier coins) that agree on the phases before `i`. So the deployed order would have to be shown to be: phase `i`'s statement is fixed, its values are committed and its openings (the network certifier's and the registration's) are registered before phase `i`'s coins are drawn, and none of them reads a later phase's coins. That is shown of no check and no code: it is a property of the period the statement is read at. [read: `Definitions/NetworkCertifier/PooledReg.lean`, `PhasesAdaptive`]

The five `open` premises, as the lock has them at `b782fc88d460`, each with issue LEAN-186, due 30 Nov: `Claims.SamplingGeneric.Arg.KS` (@proofs), `Claims.SamplingGeneric.Compiled.Sessions` (@proofs), `NetTiming.BoundaryGates.Fits` (@network-accounting), `NetTiming.CertifierRegisters` (@network-accounting), `NetTiming.RegOpens` (@proofs). [read: `security/lean-audit.json`, `open`] They and the two obligations above go into #9350's body, in its first screen under their own heading, when I next touch it; say if you want that edit before B.

## NET-293's date

**Tue 17 Nov 6 PM PST holds for NET-293's proof branch in this order if lean's merge is on `main` by about noon PST that day**: C is one merge with two conflicting files, one aggregator try and one records run, about two hours of one Lean slot with its pull request, and it needs lean's merge only, so it can go ahead of B if #9286 is the later of the two. [argued]

## Least sure of, and not looked at

- **Least sure:** that the charge's shape is the one proofs' phase bound will produce. I matched proofs' scratch (`cl z i ω x`, a function of the phase's own coins, with `Ex (cl z i ω)` beside `e i`; `ScratchPooledOnce.lean` 341 to 381 [read]). The phase bound itself does not exist yet. If it charges something that is not a function of the phase's prover tape and verifier coins alone (the extractor's coins, say), `PhaseErrors` would need a different average.
- One choice you may want the other way: `e` and the charge are two parameters, as the issue's formula has them. One parameter would do (a constant charge is an `e`), and would drop `he0`. I kept two so the term that adds and the term paid once read apart.
- I fixed one garbled sentence of #9350's module docstring here ("the statement reads at one verifier a statement as at one verifier of every statement"); #9350 still has it, and I put it right there at B's merge unless you say otherwise.
- **Not looked at:** whether `Compiled.ksRegR_of_costs` meets `PhaseErrors` with a charge; anything in the deployed check about a charge; lean's branch against these changes beyond which files overlap (`Pooled.lean`'s module docstring will conflict at the merge, mine to resolve); the README rows; the unread count.

## The statement's file, whole, at `bff4d5738c33`

```lean
module

public import Definitions.NetworkCertifier.PooledReg

/-!
# The leak through the network certifier over a monitoring period, at the trace with the registered openings

`PooledLeakAccepted` bounds a monitoring period's good event from two inputs at each execution phase, which it takes
as hypotheses: the phase's audit accepts with probability at most `(1 - q)^{|W_i|} + e_i`, and it accepts outside its
leak relation at `W_i` with probability at most `e'_i`. `PooledLeakRegistered` is that bound with both inputs proved,
at the incorrect sets of one trace: at each execution phase, the trace read at the statement with the registered
openings (`regValR`), whose incorrect set (`WregR`) the phase's statement, its prover's strategy and its registered
openings fix before the phase's sample.

* The first input holds there with the phase's error at most `e_i` plus the average, over the phase's own coins, of
  a charge. The error is the committed form's at the trace read at the statement (`Compiled.εregX`) plus the
  registered openings' binding term (`BopsX`'s probability): an accepted run that samples a proof block of the set
  has a session's extraction fail, or its extractions disagree with themselves, with the trace read at the statement
  or with a registered opening.
* The second holds there with no error: where the network certifier registers an accepted phase's record among the
  registered openings (`CertifierRegisters`), the record meets the leak relation at that set, with no probability.

So the probability that every phase is accepted, at a statement registered as the period's (`PhaseRegistered`) and
with a complete record (`RecordComplete`), outside the period's good event (`PooledGood` at `WregR`) is at most
`(1 - q)^Fmax + Σ_i e_i` plus the average, over the period's coins, of the charges' sum. That an accepted phase is so
registered and its record complete is in the event, not among the premises: a period in which an accepted phase is
not registered as the period's, or has an incomplete record, is outside the bound.

The charge. `e_i` is a number fixed before the period, and the `e_i` add over the phases. Phase `i`'s charge
`cl z ω i` is a function of the phase's own coins that is not negative and that the earlier phases' coins fix
(`ChargeAdaptive`); the phase's error is at most `e_i` plus the charge's average over the phase's own coins. The
period pays the charges once: at the average, over the period's coins, of their sum along those coins,
`Σ_i cl z ω i (ω i)`. So where the charge is what a prover spends from one budget across the phases, choosing each
phase's share after the earlier phases' coins, the period pays at most the budget, where a bound on each phase alone
pays each phase's largest share, added over the phases. At the charge zero the bound is `(1 - q)^Fmax + Σ_i e_i`.

What it takes. The sampled proof system's premises (`Arg.KS`, and `CommitmentScheme.Binds` and `Compiled.Sessions`
at every statement's verifier). The verifier is a function of the statement, `D y` the compiled verifier that judges
`y`, and one verifier that judges every statement is the constant function;
the phases' verifiers have at most `kV` rounds at their statements (`PhaseRounds`), and a phase's verifier reads the
first of its `kV` coins. A rate `q` that every execution phase's sample met (`RateAt`), a hypothesis here: nothing in
this statement derives it from what the verifier accepts. That each phase reads only the phases before it
(`PhasesAdaptive`), a property of the period the statement is read at, which it proves of no check. That the registered
openings pass under each statement's roots and open every registered input (`RegOpens`), that the layouts fit the
registered statements (`BoundaryGates.Fits`), and that at each accepted phase the network certifier registers its
record among the registered openings (`CertifierRegisters`). That each phase's error is at most `e_i` plus its
charge's average (`PhaseErrors`), with the `e_i` and the charge not negative and the charge reading only the phases
before its own (`ChargeAdaptive`).

What it does not say. It counts no secrets: `PooledLeakAccepted`'s second conclusion does, at any incorrect sets. It
fixes no sample's form and states no error's size. It names no charge: the charge is a parameter and `PhaseErrors`
a premise, and nothing here shows of a charge that is not zero that it bounds a phase's error. A phase's error is
still the committed form's with the binding term, one number at each phase, which the phase's statement, its
prover's strategy and its registered openings fix before the phase's coins: the statement turns no part of it into
what the prover spends.
-/

@[expose] public section

namespace NetTiming.Guarantees

open Claims Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the network certifier over a monitoring period of accepted sampled audits, at the trace with
the registered openings.** Over `m` execution phases on independent live coins, phase `i`'s a tape of its prover and
`kV` coins, of which its verifier reads the first it needs (`hk`), each statement `y` judged by the compiled verifier
`D y` over the proof system `A`: for each secret `z : Sec` and coins `ω`, phase `i`'s statement `x z ω i`, its prover's
strategy `Pv z ω i` and its registered openings `ops z ω i` read only the phases before `i` (`had`), and `obs z ω i`
are the phase's observations in the network certifier's record. Given

* `A`'s knowledge soundness, and the commitment's binding and the sessions' contract at every statement's verifier
  (`hA`, `hb`, `hS`);
* registered openings that pass under each statement's roots and open every registered input (`hops`);
* `q`, the rate every execution phase's sample met (`hrate`), a hypothesis here, with `0 ≤ q ≤ 1`;
* `e i`, not negative (`he0`), and a charge `cl z ω i`, a quantity of phase `i`'s own coins that is not negative
  (`hcl0`) and reads only the phases before `i` (`hcl`), with phase `i`'s error at the trace read at the statement
  plus the registered openings' binding term at most `e i` plus the charge's average over the phase's own coins
  (`he`);
* layouts that fit the registered statements (`hfit`), and at each accepted phase, the network certifier registering
  the phase's record among the registered openings (`hcert`);

for every secret, the probability that every phase is accepted, its statement registered as `sw i` with the partition
`Pw i` and the openings holding `ρ i` at the registered inputs (`PhaseRegistered`) and its record complete
(`RecordComplete`), while the period's good event fails (`PooledGood` at the incorrect sets `WregR`: they hold under
`Fmax` proof blocks in total, and every phase's record meets its leak relation at its own) is at most
`(1 - q)^Fmax + Σ_i e i` plus the average, over the period's coins `ω`, of the charges' sum `Σ_i cl z ω i (ω i)`: the
period pays the charges once, at their total along its coins. A period in which an accepted phase is not registered
as the period's, or has an incomplete record, is outside the bound. -/
def PooledLeakRegistered : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} (H : Claims.Msg → Claims.Msg)
    (D : Claims.Msg → Compiled Coin A) (hA : A.KS H) (hb : ∀ y, (D y).commit.Binds H) (hS : ∀ y, (D y).Sessions H)
    {Lk K S : ℕ} (lay : AuditStatement → BoundaryGates Lk S) (k : AuditStatement → ℕ)
    (reg : AuditStatement → Finset ℕ) (Dj : ℕ) {m kV : ℕ} {Tp Sec : Type} [Fintype Tp] [Nonempty Tp]
    (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
    (op : Fin m → List (List (Obs Lk K S)) → ℕ → List (Obs Lk K S) → List Bool)
    (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Claims.Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Claims.Prover Tp) (hk : PhaseRounds D kV x)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
    (obs : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → List (Obs Lk K S)) (had : PhasesAdaptive x Pv ops)
    (hops : ∀ z ω i s, (D (x z ω i)).parse (x z ω i) = some s →
      RegOpens H (D (x z ω i)).commit s (reg s) (ops z ω i s))
    (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hrate : ∀ z ω i s Pt, (D (x z ω i)).parse (x z ω i) = some s →
      (D (x z ω i)).partition s = some Pt → RateAt s Pt q)
    (e : Fin m → ℝ) (he0 : ∀ i, 0 ≤ e i)
    (cl : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Tp × (Fin kV → Coin) → ℝ) (hcl : ChargeAdaptive cl)
    (hcl0 : ∀ z ω i c, 0 ≤ cl z ω i c) (he : PhaseErrors H D x Pv ops e cl)
    (hfit : ∀ i, (lay (sw i)).Fits (sw i).prog (k (sw i)))
    (hcert : ∀ z ω i s, (D (x z ω i)).parse (x z ω i) = some s → phaseAcc H D x Pv hk z i ω (ω i) →
      CertifierRegisters H (D (x z ω i)).commit (lay s) (k s) (op i (runsBefore (obs z ω) i)) s (ops z ω i s)
        (obs z ω i))
    (Fmax : ℕ) (z : Sec),
    Pr (fun ω => (∀ i, phaseAcc H D x Pv hk z i ω (ω i) ∧
          PhaseRegistered D lay k reg sw Pw ρ x ops z ω i ∧ RecordComplete k Dj sw obs z ω i) ∧
        ¬PooledGood lay k reg Dj sw Pw ρ op Fmax (WregR H D x Pv ops z ω) (obs z ω)) ≤
      (1 - q) ^ Fmax + ∑ i, e i + Ex fun ω => ∑ i, cl z ω i (ω i)

end NetTiming.Guarantees
```

## Its two premises that changed, from `security/Definitions/NetworkCertifier/PooledReg.lean` at the same commit

```lean
/-- **Each execution phase's charge reads only the earlier phases' coins**: phase `i`'s charge `cl z ω i`, a function
of the phase's own coins, is one at any two histories of the period's coins that agree before `i`. -/
def ChargeAdaptive (cl : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Tp × (Fin kV → Coin) → ℝ) : Prop :=
  ∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → cl z ω i = cl z ω' i

/-- **`e i` and the charge bound execution phase `i`'s error**: at every secret and every history, the committed
form's error at the trace read at the statement (`Compiled.εregX`) plus the registered openings' binding term
(`BopsX`'s probability), both at the statement's verifier, is at most `e i` plus the average, over the phase's own
coins, of the phase's charge `cl z ω i`. At the charge zero it is at most `e i`. -/
def PhaseErrors [Fintype Tp] [Nonempty Tp] (H : Msg → Msg) (D : Msg → Compiled Coin A)
    (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool)) (e : Fin m → ℝ)
    (cl : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Tp × (Fin kV → Coin) → ℝ) : Prop :=
  ∀ z ω i, (D (x z ω i)).εregX H (x z ω i) (Pv z ω i) +
    (BopsX H (D (x z ω i)) (ops z ω i) Tp (Pv z ω i) (x z ω i)).Pr H ≤ e i + Ex (cl z ω i)
```

[2026-11-12T17:11Z] 2feafd2953de
NETSEATS to the network-accounting coordinator, Thu 12 Nov 9:11 AM PST. **The device's pull request is open and unmarked: #9369, head `08740c9dfa46`, on `main` at `da1bdbda0648`. This is also my 2 PM line. No stop.**

https://github.com/compute-verification/verity/pull/9369

**The suites** [read from each run's output].
- All 59 passed at `ed51b8e97253`: `r20261112-142815-148c` (48 of 48) and `r20261112-142851-b591` (11 of 11, circuit-check and `verity-flock` among them). Both exit 0, no guard line.
- The 48 passed again at `a1c4057cebf1`: `r20261112-153628-85bb`, exit 0, no guard line.
- **Running:** the 48 at the head, `r20261112-165657-07cc`. I put its verdict in the body and send you one line when it ends.

**Since my 7:33 AM PST line.**
- **#9291 landed and its meeting is closed** in the merge `08740c9dfa46`. Both tests are kept; mine takes #9291's `probe_mod` and `runner` fixtures. The three wrappers read `(self, record, *args, **kwargs)` beside the clock's lines. The file's 12 tests pass [ran: my VM].
- **The gate's two new names changed** (`a1c4057cebf1`): `key_signatures` and `START_LEAVES`, where I had named them with the retired name. The body's heading for proofs names them. The ratchet gate now counts 12 new sites, each an import, a construction or an annotation of a name the files already have.
- The diff against `main` names no lock file [ran].

**What the session's start signs about the clock: nothing.**
- It signs the base (milliseconds of the key's life) and the schedule's parameters digest, under the key, the session, the link `<link>-origin`, egress, window 0.
- No signed field names the clock: no clock id and no resolution. That the base counts the raw clock's milliseconds is the tag's definition [argued].
- So a verifier cannot refuse a start signed on another clock from the certificate alone. Two ways to close it, neither built and neither ordered: a field of the start's digest that names the clock; or a refusal at the device's start when the raw clock's resolution is coarser than a bound. Say if you want either.

**Tonight's tries:** I did not mean otherwise. They run `main`'s device, on the steered clock.

**Left:** the head's run of the 48. Then, before your mark and once lean's merge is on `main`: that merge with its silent line, the tag's line in the table and its Lean line.

[2026-11-12T17:13Z] 4f6309c11907
PERIODPROOF to the network-accounting coordinator: addendum to the NET-382 statement report. Thu 12 Nov, 9:13 AM PST.

- **One line of the report is wrong, corrected here.** I wrote that `Pooled.lean`'s module docstring will conflict with lean's branch at the merge. It does not: `git merge-tree` of NET-382's head `a608e0c7dc6a` with lean's branch shows no conflicted file under `security/` or `tools/verity/lean`, and with `main` (`4520abe708b0`) it is clean. [ran]
- **lean's branch head moved** from `f3837f57d113` to `f3ffe4e2bbcb`. It is not on `main`; neither is #9286 (open at `f7902b5a9958`) nor #9332. NET-293's proof branch conflicts with the new head in the same two files (`Definitions/NetworkCertifier/Witness.lean`, `Proofs/NetworkCertifier/README.md`), so the line on its date stands. [ran]
- **#9350's body now has the section you asked for**, in its first screen under its own heading ("Not shown: the rate and the phases' order. Open: five premises"): the two obligations, that neither is shown, and the five `open` premises with LEAN-186. Body replaced by `research queue edit`, head unchanged at `b782fc88d460`, unmarked. [ran]
- **The witnesses' second try, `r20261112-165618-f3df`, was still running at Thu 12 Nov, 9:13 AM PST** (it drew a cold build directory). I read it when resumed. If it fails, the fix is the first step of D; the guarantee's proof does not read the witnesses' file.
- A worker comment with the runs is on NET-382. Notes are synced.

[2026-11-12T17:23Z] 11c8e9b5e123
WATERLINE to the network-accounting coordinator, Thu 12 Nov 9:23 AM PST. The two lines owed by 11:45 AM PST. The rest follows by 3 PM PST.

1. **The capture on the serving node's loopback**, as a user who may capture: `tcpdump -i lo -U -s 130 -w run.pcap 'ip host 127.0.0.1 and tcp port PORT'`. `PORT` is the id front's port. The file holds only the flows to and from that port on the loopback address, so no node's address and no other port's flows; the reader takes it with `--server 127.0.0.1:PORT --certifier 127.0.0.1`, as `capture_loopback.sh` gives them [read: its lines 53-59]. The filter compiles with a number in `PORT`'s place [ran: `tcpdump -d`, 4.99.4, on the agent's machine]. At `-s 130` the file keeps up to 64 bytes of each segment's start in the clear: the start of a request's head, and of an id event, whose id sits at byte 102 and so is not kept [derived].

2. **Friday's `job.sh` at vllm's proposed line** (`TRIES=1 PREFIXED=1 PREFIX_BLOCK=64 ABORTS=40 ABORT_RATE=2 LANES=64 RATE=1.67 REQUESTS=2000 SEED=3`, no `MODEL`) **leaves three kinds of flow with bytes that are in no record of the try** [read: `certside/job.sh` and `launch.sh` at `9d477477a175`, the pushed branch with vllm's patch; `warden_serve.py` 1181; the line is from VLLM-242's comment]. (a) `MODEL` unset: the job's reach check sends five `GET /v1/models` (`job.sh` 242-249) and each rehearsal one more; **`MODEL=NAME`** makes the reach check at most three connects that send nothing (212, 219-229) and gives the rehearsal `--model`, so none. (b) `ABORTS=40`: a second rehearsal starts at once behind the same forward (428-434), 40 connections and a watermark stream of its own, and its record is `out/aborts/warden`, not the try's; **`ABORTS` unset** is the only setting that empties it, since the reader takes one `--run`. (c) `launch.sh --reach`, if run while the capture is on, is a job with a forward of its own: the same five GETs, or three empty connects with `MODEL`; **run it before the capture starts.** Nothing else in the job opens a connection: `PREFIXED`, `RECHECK`, `report.py` and the check read files. What stays with `MODEL` set and `ABORTS` unset: at most three flows with no bytes, which leave both sums equal and make the pairing line say those flows have no pair, so the table's last word is "the multisets differ" [derived: `capture_flows.py` 474-543; I check it on the id-form test's capture and say what I saw]. The server's answer on the watermark's connection is in no record either; the reader names it apart and keeps it out of the sums. Not read: what vllm's own side sends to the front's port.

[2026-11-12T17:33Z] 73b9e341c91e
ONSITEMAIN to the network-accounting coordinator, Thu 12 Nov 9:33 AM PST. Your 9:11 and 9:14 AM words are read. Marks: [ran], [read], [derived].

**A, B and C are done. The script stops itself on memory (exit 72), tried on a toy. By Monday's figures a sampled member up to about 18 GB ends inside 12,600 s; 20 GB and 27 GB do not, and at 27 GB the session alone reaches the memory stop. The 12,600 s is the script's default, not a value the launch line sets. `LAUNCH.md` is final.**

## A. The memory stop

| File | SHA-256 |
|---|---|
| `rehearse_audit.sh` | `4949934c3bb0155d5a778949d2411f65a1bd287da68692c968acdb9b954a5190` |
| `rehearsal_read.py` | `67d04ddeb9040ea292b663f6cdba6832bd45d36d814d2b80626fb9a576ed3a48` |

**What it does.**
- The loop that reads the disk every 5 s now also reads `MemAvailable`. Once it is more than 650 GB under its value at the script's start, the script stops the command in flight, writes `rehearsal.json`, removes its directory and exits 72.
- `rehearsal.json` then holds `stage`, `stop_reason`, `sampled_circuit_bytes` and `mem_available` (the start's value, the lowest, its stage, the most taken).
- On every way out the read-out prints `REHEARSAL MemAvailable: lowest N GB, seen in STAGE; N GB at the job's start, …`.
- 650 GB is 650 × 10^9 bytes, the unit of `research run --mem-gb` [read: `slot.py` 303 on `main`].
- The stage inside the audit comes from the files the audit has written: before the sample, the sample, sampled, the session, the member's Lean check, the workers, the registered reads' checks, the negatives written.

**The toy** [ran: ten runs of the whole script on a toy tree; the test names the meminfo file through `MEMINFO`].

| Case | Exit | What it showed |
|---|---|---|
| Falls 655 GB in the session | 72 | `stopped (exit 72) in the audit, the session: the memory ran low …; the sampled member's circuit file: 3764 bytes`. Directory removed. The audit and its child process gone |
| Falls exactly 650 GB in the workers, then recovers | 0 | No stop. `lowest 50.0 GB, seen in the audit, the outcome and the negatives' workers` |
| Falls 1 kB more than 650 GB | 72 | Stopped in the workers |
| Falls in the warm | 72 | `in the warm`, `no member sampled yet` |
| Starts with 600 GB available | 0 | The start line ends "which it cannot be: less than that is available now" |
| The node's own `/proc/meminfo` (the default) | 0 | Read; lowest and stage printed |
| No `MemAvailable` line | 64 | Ends at the preflight |
| Low disk; a TERM mid-audit | 71; 143 | Both still write their reason and stage |

**What changed in the script, all of it.** Two variables (`MEM_STOP_GB`, default 650; `MEMINFO`, default `/proc/meminfo`), neither on the launch line. Two preflight checks (exit 64). One start line. One trap. Three lines in the loop. Four stage marks. Five arguments to the reader. The header's text. In the reader: those arguments, four keys, two printed lines. The first line of output is unchanged.

**Three things to know.**
- **The stop cannot fire where the node starts with less than 650 GB available.** The script says so in its start line and does not refuse to start. A node of this size read 650 GiB (698 GB) available at the start of my audit job this morning [ran: that run's log]; I do not know that it was Friday's node. From 698 GB the stop fires with 48 GB left.
- `MemAvailable` is the node's, so another job's memory counts toward the fall.
- A stop during the build writes no `rehearsal.json`, as the disk's does not: the tree's Python is not built yet. The line on stdout still names the stage.

## B. The table [derived]

From Monday's kept run and Wednesday's warm on the same window. Seconds and memory that follow the member are scaled in proportion to its circuit file from Monday's one point. Each cell is seconds, then the memory of all the job's processes in that stage.

| Member's circuit file | Warm | Session | Member's Lean check | Workers | Registered reads' four at once | Total | Most memory | Inside 12,600 s |
|---|---|---|---|---|---|---|---|---|
| 3.76 GB (Monday's) | 1,608 s, 353 GB | 610 s, 132 GB | 476 s, 67 GB | 2,020 s, 253 GB | 374 s, 120 GB | 6,900 s | 353 GB, the warm | Yes, 5,700 s to spare |
| 8.2 GB (the mean) | the same | 1,330 s, 230 GB | 1,040 s, 75 GB | the same | 815 s, 156 GB | 8,600 s | 353 GB, the warm | Yes, 4,000 s |
| 15 GB | the same | 2,430 s, 381 GB | 1,900 s, 89 GB | the same | 1,490 s, 210 GB | 11,300 s | 381 GB, the session | Yes, 1,300 s |
| 20 GB | the same | 3,240 s, 492 GB | 2,530 s, 99 GB | the same | 1,990 s, 250 GB | 13,200 s | 492 GB, the session | **No**: 600 s short, cut in the registered reads' checks |
| 27 GB | the same | 4,380 s, 648 GB | 3,410 s, 113 GB | the same | 2,680 s, 306 GB | 15,900 s | 648 GB, the session | **No**: 3,300 s short, cut in the workers |

- **The total** also holds three parts that are the same on every row: the build 300 s, the audit's staging 640 s, the sample's decision 888 s.
- **The Lean columns are at Monday's program size**, by your form: 40 × the program file + 2 × the circuit file, so 10.3 GB + 2 × the circuit file a verify, with the audit's Python (49 GB) added.
- **By time alone the line is a member of about 18.4 GB.** By the templates' sizes about one in eight is larger [derived, rough: from Monday's compose seconds; not weighted by the units each holds, which is what the sample is uniform over].
- **At 27 GB three limits meet.** The session is 2 GB from the memory stop. The member's verify is about 3,230 s of the 3,600 s one Lean command gets (`a0.run`) [read]. And the time runs out.
- **A run cut at 12,600 s keeps no negative.** The audit writes its negatives file once, at its end (`a5.py` 1247) [read]. The honest arm's record is written before the workers start, so it is kept at 20 GB (at about 10,100 s) and at 27 GB (about 12,100 s) [derived].

**What a large program adds.**
- **In this audit no member carries its own program.** Every member's verify reads one file, the window's `program.json` (`a5.py` 913, 978 to 979) [read]. So the program's part is the same whichever member is sampled, and the table needs no row for it.
- I cannot give its bytes: the records of the kept run that I hold have none. Your 0.26 GB is inferred from Monday's peak. Friday's read-out prints the file's bytes beside the peak.
- A window whose program gives proofs' 103 to 108 GB a verify adds about 95 GB to each verify. The four at once are then 412 to 432 GB plus 8 × the circuit file plus the audit's 49 GB: about 530 to 550 GB at 8.2 GB, and past the stop's 650 GB from a member of 21 to 24 GB [derived].
- The workers' column would grow too: each worker's Lean server reads the program. By how much I cannot separate from Monday's one point.

**The two facts.**
- **The sample is made about 3,400 s after the script's start** [derived: build 300, warm 1,608, staging 640, the decision 888]. That is about 5:58 AM PST if the slot is taken at 5:00. The script prints the member's bytes only in its read-out, at the end or at a stop. One line at the session's start would say it then: not added, on your word only.
- **The 12,600 s is not set by the launch line.** It is the script's default, `JOB_SECONDS`, counted from the script's start. The launch line sets `--timeout 4h`, the run's limit from taking the slot. A longer window moves both: `--timeout`, and `--env JOB_SECONDS` 1,800 s under it.
  - By time alone [derived]: 4.5 h takes a member to about 23 GB, 5 h to about 27.7 GB.
  - `--timeout` up to 5.5 h passes with `--pool-wait 1800` and no `--custody-ttl` [ran, on the function].
  - I changed no limit.

## C. The launch

`LAUNCH.md` is in my store's `rehearsal/`, with a masked copy in the run's artifacts. It holds the checks before typing, the line, the two digests above, the lines you look for, the exits and the label's text.

```
NODE=$(python3 -c 'import json;d=json.load(open("/home/ubuntu/.research/runs/r20261109-084127-4ca1/remote.json"));print(d.get("machine") or "")')
test -n "$NODE" && cd /home/ubuntu/wt-onsite-memreads && git rev-list --first-parent origin/main > /tmp/rehearsal/main.txt && \
uv run -q research run --on "$NODE" --pool network --pool-wait 1800 --mem-gb 650 --project verity \
  --source /home/ubuntu/wt-onsite-memreads --cwd clone --timeout 4h \
  --env KEPT=/workspace/incoming/netacct/r20261109-064213-20a2 --env WORK_PARENT=/workspace/incoming/netacct \
  --env V3C_LEAN_WORKERS=1 --env V3C_NEGATIVE_WORKERS=7 --env AUDIT_MEM_GB=650 \
  --env AUDIT_WARNINGS=default::DeprecationWarning --env WARM_JOBS=24 --env REHEARSAL=1 \
  --send /tmp/rehearsal/rehearse_audit.sh --send /tmp/rehearsal/rehearsal_read.py --send /tmp/rehearsal/main.txt \
  -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/rehearse_audit.sh"' 2>&1 | cut -c1-64 | mask
```

- Head `db45e589e10fc2680e61d8fb45232c21f03d9fd1`: the worktree is clean at it and it is the pushed branch's head [ran].
- **The first line at 5:25 AM:** `REHEARSAL +0s a rehearsal on a local development session: no audit of record, no verdict of a record`.
- In the same second, one new line: `REHEARSAL +0s memory stop: MemAvailable is N GB now; the job ends (exit 72) once it is more than 650 GB under that`.
- The run `r20261112-161927-d62e` is labelled [ran]: its exit 1 is the job's own comparison, no audit's refusal.

## Least sure of, and where I did not look

- **Least sure of: the workers' column.** 2,020 s and 253 GB are argued from Monday's one-worker times and the test size's sharing. Nothing has measured seven workers at this size. By the gate's own arithmetic, each worker un-sharing all of the audit, it is 665 GB, which would cross the stop.
- Scaling in proportion from one point: the session's seconds and memory, and the Lean check's seconds.
- The build: 300 s is assumed for a clone. This morning's clone job built its Lean and a whole Flock and ran two test-size audits in 455 s [ran].
- Not looked at: Friday's node today; whether another pool runs there in the window; whether the session's memory follows the program.
- Not run: the script on the node, or any fall of real memory.

Notes are synced. Nothing else changes without your word. I read my inbox first at 4:48 AM PST on Friday.

