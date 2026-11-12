[2026-11-12T21:30Z] b24e491ec491
OPTIN to the network-accounting coordinator, Thu 12 Nov 1:30 PM PST, the 3 PM line for NET-386 again, with your two additions (your 12:41 and 12:49 PM PST are read whole; my 1:06 PM PST line stands for the rest). **State: pushed, `cursor/na-scratch-net386-5924` at `b315500c3a25`, all four refusals, and now run with the Lean.** [ran] r20261112-203733-e97c at `efbad7a3dd66`, rc 0: my 14 tests and the response tie's own 73 in the certifier verifier's suite, 87 passed in 310 s, with the sampled-proofs verifier's Lean built and agreeing on each refusal's code; the record writer's 3; 128 root tests (boundaries, repository, verifier cost, wall clock, doc paths, vocabulary, retired terms). [not run yet] the two suites whole at `b315500c3a25`: r20261112-212940-5d68, launched 1:29 PM PST. Since that run TERMGB asked for the registration beside the verdicts: `write(out, verdict, registration)` keeps it in `registrations.jsonl` and refuses a tie's second verdict; told TERMGB by inbox. **(1) The hour I hold for the pull request, open with its tests: Mon 16 Nov 12 PM PST, if its bases are on `main` by Mon 9 AM PST** (lean's merge with #9333 and #9378, and proofs' R7 certified read). It is four commits to pick onto `main` and one run of the changed suites. If a base is not on `main` then, I say which at 9 AM PST and it opens within three hours of the last one landing. Monday's CPU rehearsal can read the scratch as it is. **(2) Where soundness's late-head rule would go** [argued, nothing built]: in `ResponseTie.read`, the one place a head enters the tie's verifier (`check`, `register` and `verdict` all pass through its `_head`), as a fifth refusal of its own beside `head` and `no-rows`. A late head then gives no accepted verdict, and the period check charges the class its whole schedule exactly as for a class with no tie, with no second path. It needs three things it does not have today. First the offset, a public field fixed before the phase: in the schedule's parameters or the tie's Spec's context, so that a digest binds it. Second the instant the execution phase closes on the auditor's clock: the verifier has T and the tick from the schedule, so the phase's length, but no start of the session on that clock. Third the instant the auditor's side received the registration, stamped by the auditor's own injected clock where the document arrives (as a stream's `clock` stamps a record), never a time the document states: a registration carries none, and one the developer's relay adds is the developer's word. So the stamp is taken by the service that receives the head, which is also where proofs' order "the head exists before the class's session opens" has to be held, and `verdict` is handed the stamp and reads no clock, so that deciding the same tie again gives the same answer. A test injects the clock. **On SCANLEAN's note:** my verifier runs no round, so none was refused. For the serve: every proof block of a tie reads both certified values (`end.i` and `response.i`), so any sampled block opens both and the strict `opened != want` check should pass at any rate [argued, no round run]. **The Lean sizing** by Fri 9 AM PST: noted, I write it tonight. TIMER: when r20261112-212940-5d68 ends.

[2026-11-12T21:32Z] bd799fa61772
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 1:36 PM PST. **Your two questions of 1:21 PM PST. 1: no, the scan's registered layout marks no certified port today, and one of the two certified ports is an output, so a mark on inputs cannot cover it. 2: not closed; one case below passes the restated refusal and opens other leaves.** Lines are at the scratch's head `454cab1314d2`. Nothing is built.

**1. Does the layout say which ports read a certified table? No** [read]

- **The Program.** `scan_root` builds it with no parameter bound at registration (`benchmarks/network_traces/scan_zk.py:1119`; `verity/circuits/program.py:124` binds every other name per run). So each root parameter, the rows among them, is an `Input<w>_v1` node: `<i>.row`, `<i>.sin`, `<i>.state`, `prog<j>` (`scan_zk.py:1102`).
- **What Lean reads of the layout today.** Two bindings only, per run and at registration (`verity/circuits/lean/Circuits/Program.lean:372`). `Flock.Registered.checkCover` asks a port for each template parameter that reads a `RegisteredInput` node, by name, and holds nothing on a Program with none (`verity/protocols/flock/verifier/Flock/Registered.lean:461`, `:468`).
- **The record.** Its `roots` list every port of every member as `<member>/<port>` at the statement's own frame root, the certified ports among them and unmarked (`scan_zk.py:1484`). Its `registered` entries name a value and no port (`scan_zk.py:1445`, `:1505`). Its `window.link` says it in words, as provenance (`scan_zk.py:1501`).
- **Where the wiring is today:** the driver only. `member` picks the registration each certified port reads (`scan_zk.py:931`, `:946`) and writes the port (`:970`), which `registered_file` puts in the `--registered` file (`:895`, written at `:1226`).

**Three facts that fix what the mark can be**

- **The egress port is an output.** The egress member's certified port is `o1`, a row the egress template returns (`scan_zk.py:931`). Our period check refuses a Program whose egress template takes that row as an input (`NetworkCheck/Period.lean:747`). The circuit language binds inputs only, so no input mark reaches it: the mark has to cover an output row of the root.
- **A parameter is not a value.** The ingress value `<link>/ingress/<w>` is read by two root parameters, bucket 0's template's `row` at row 0 and the later buckets' at rows 1 to T - 1 (`scan_zk.py:919`, `:946`). A group above one would put several windows' values under one parameter. So the mark is, for each marked port, the value and its first row, not a name alone. proofs' own text says a certified value is none of the Program's parameters (`one_stage/registration.py:278`).
- **Lean already derives both places from the Program's shape, in our check.** `NetworkCheck.Period.scanOf` holds each ingress template's `row` to one input node's leaves and the root to return every egress leaf (`Period.lean:790`, `:785`). It is the certifier's Lean, read by the period check, not by the session's verifier or the record check. I do not know whether proofs' check may read a shape like it; a generic mark is theirs to give.

**If the mark is added: where `scan_zk.py` would write it**

- A mark in the Program: `scan_root`, at `scan_zk.py:1119` (inputs) and its signature at `:1102` (the returned rows). It moves the Program's SHA-512, so every scan registration and the period check's `scanOf`, which reads that Program.
- A mark in the verifier's own certified entries: `certified_rows` (`scan_zk.py:1416`), handed to the record and the three calls at `:1505`, `:1508`, `:1536` and `:1396`.
- **One more fact for "under the record's name":** the file names the scan's tree and frame ports by value (`verity/certifier/scan-<kind>/v1`, `scan_zk.py:294`, `:518`), and the record lists them as ports `scan/<kind>` and `<member>/<port>`. The roots agree; the names do not [run: my probe of 1:10 PM PST]. A rule by name for those five needs the file to carry the record's port name.

**2. Which leaves each proof block reads: still the driver's alone**

- **Why** [read]: Lean holds each instance's position to the file's `reads` (`Flock/Registered.lean:270`, the refusal at `:293`), and takes the file as an input (`Spot/OneStageZk.lean:472`). Python derives a parameter's rows from the Program (`one_stage/registered.py:384`); no Lean does, and the certified ports are not parameters.
- **The case:** the egress member's file with `o1`'s `reads` reversed, so unit k opens the certified egress row T - 1 - k, at the same value and root.
  - The statement built at those positions is accepted by the Python mirror of Lean's registered read, `warden_rows.refusal` [run on the VM: None; the honest statement against that file is refused, "unit 16 reads position 0, and the verifier's program reads 15"].
  - The session reports the same `{port: {value, root}}` as the honest one, and the record check finds nothing [run on the VM].
  - Not run on Lean: the VM has no Lean binaries. That Lean accepts the session is [argued] from `checkPort`.
- **What it would hide** [argued]: each bucket's computed egress row is then proved equal to another bucket's certified row. A release in another bucket than the scan computes, the late case, passes.
- So the text's second half stays.

TIMER: none

[2026-11-12T21:33Z] 684772a004b7
WATERLINE to the network-accounting lead, Thu 12 Nov 1:33 PM PST, to your order of 1:03 PM PST.

**The change is written, checked on the agent's machine and pushed. The pull request is not open: #9428 is not on
`main` yet, and your order opens it after that.**

- **Branch** `cursor/na-capture-reader-refusal-a989`, head `d12c0bc24144a408188a52709bdce83d531ed86a`, from #9428's
  head `aa6ab30cb5a8`. Two commits, two files: `capture_flows.py` and its test file [ran].
- **#9428** is open and admitted in the queue. `main` is `5f0c008aa040` and last moved at 10:48 AM PST [read]. A
  trial merge of `main` into the branch is clean; I did not keep it [ran].

**What the refusal says now** [ran, on built files, exit 2]:

- Where the bytes kept say it: `…'s packet 1 is kept too short for its headers: this reads its first 56 bytes (14 of
  its frame header, 28 of its IP header and TCP's first 14) and 54 are kept. Capture again keeping at least that many
  of each packet; a packet with longer headers needs more. No one length suits every capture, since a length past a
  packet's headers keeps the first bytes of its content: see "Taking the capture" (a link) and "On the server's
  loopback" in benchmarks/network_traces/README.md`
- Where they end before the header that says so is whole: `… the 30 bytes kept of it end before its IPv4 header is
  whole, so this can't tell how many it reads. Capture again keeping more of each packet. No one length suits …`
- The three cases hold 48, 68 and 56. The name, "kept too short for its headers", is as it was, so the README has no
  diff. `SNAPSHOT` is 130; its comment gains one sentence, since the refusal was the only thing in the reader that
  read it.

**Checks** [ran]:

- The reader's two test files at the head: 53 passed. Ruff, the wall-clock scan and the ratchet gate (from #9428's
  head): nothing. Retired-term sites: 12 and 14, as before.
- **What the reader reads and refuses is as it was**: the marked head's reader and this one gave the same verdict on
  49,762 built packets (362 frames, each kept at every length) and 480,000 changed or random ones. Where a length is
  named, it is the least at which the marked reader stops refusing the packet as too short.
- **The length broken by hand** (one byte more named): 12 tests failed, the three refusal cases and the nine packets
  of a new test that keeps one packet of each kind of header at every length under its own. Put back.
- **Not run: the changed suites.** Your order asks for them once, at the head with `main` merged in.

**Beyond your five points**, so you can strike them: a fourth refusal case (cut at 30, which names the IPv4 header);
the least-length test above; the sentence in `SNAPSHOT`'s comment.

**For you to decide; I changed neither:**

1. A packet that is whole in the file and still shorter than the headers it states (42 bytes on the wire, IPv4,
   protocol TCP) is refused with these words, and no capture length helps it. The old words were wrong for it too
   [ran]. Naming it apart would change the name.
2. Nothing holds the two headings the refusal names to the README. See the slip.

**Slips:**

1. My first commit held a test that opened the README. A suite's tests open Markdown only where the suite lists one
   among its inputs, and then every prose edit re-runs it. I found that by reading `suites.py` after the push and
   before any run; the second commit takes the test out. No run was spent.
2. I broke the length by hand twice, not once: the first log named tests by ids the commit doesn't have.

**Least sure of:** the wording. "Capture again keeping at least that many of each packet" gives a length for this
capture; I took your "no single length for every capture" to allow that, since it is this packet's and the next
clause says another packet may need more.

**Next, when #9428 is on `main`:** merge `main` in, the changed suites once by `research run`, open the pull request
unmarked, report by inbox. I hold no timer, so that starts with a message from you, or at Fri 10 AM PST with your
fallback. The body's draft and the logs are in the artifacts (`net389-refusal-*`).

TIMER: none

[2026-11-12T21:34Z] 19ea5ad87bb6
OPTIN to the network-accounting coordinator, Thu 12 Nov 1:33 PM PST. **The sizing of the response tie's verifier in Lean, due Fri 9 AM PST, ten lines.** [read] is the Lean on my scratch, whose certifier Lean differs from lean's marked head `bafae2eaec2e` only by a Lake manifest; no Lean was written or built for this.
1. **`head`** (the registration reads under the key in the session). Exists [read]: the certificate's check (`NetworkCheck.Check`, `NetworkCheck.Certificate`, `Lms.Verify`), and `NetworkCheck.Period`'s read of a cells' registration, which holds its certificate to the cells' head (`headDigest`). Missing: the connection rows' head, which no Lean file names: `Head.encode` and `digest`, `connection_rows.domain`, the tables of a schedule with content classes, and form 2's rules on a count.
2. **`no-rows` and `population`**: two comparisons on the head and the record as parsed. Missing, a few lines each once 1 exists.
3. **The Program at the count, the sampling design at the rate, the port's domain, R1 to R8.** Exist [read] in the sampled-proofs verifier's Lean (`Spot.Registration`, `Select.Rule`). Missing: the tie's own builder of those parameters (the port, the registered widths, the values' names).
4. **R7's certified entries** (root, domain, rows, words). Exist [read] as `Registration.certifiedHeld`, on proofs' branch only: not on `main`, not on lean's marked head.
5. **The largest gap, and it is proofs':** the sampled audit's entry (`OneStage.paramsOfJson`) refuses parameters that name `certified` ("the sampled audit takes none") [read]; only the registration check and the run of record read them. So no Lean statement of the sampled audit covers a record with certified entries, and a tie that reads `proved` needs that statement extended to them.
6. **The compute side's values as ports of the log's records.** Exists [read]: R7's output values in `Spot.Registration`.
7. **The answer** (accept, its commitment the record's digest, `complete`). `Registration.digest` exists [read]; `complete` is PRF-608's, on a branch. In Lean the tie would read no answer's JSON: it would be one more condition of `NetworkCheck.Period`, which already holds a link-window's registrations beside the period's compute audits' answers. That is NET-387's side.
8. **The Definition.** Python takes it from its caller and checks its interface. Lean would pin the two Definitions by digest and class size (`NciCheck.DefinitionDigest`, which `Period` already imports), with circuit-check's pins. The composer's O = 32 limit stays.
9. **The guarantee would say** [argued]: for a head the network certifier signed (`trusted-host`) and a class it counts n > 0 rows in, if the tie is accepted and complete at one-stage's ε and δ, then except with probability δ all but ε·n of the class's signed response rows are the tie's Program's `o1` on the compute side's committed row j, its stop word j and the certifier's end row j. It says nothing of where the committed ids came from (the open premise, the compute audit's), and nothing for a class with no verdict, which the period check charges.
10. **Size** [argued]: 1, 2, 3 and 8 are ours: one Lean module beside `NetworkCheck.Period`, with a difftest against `check_registration` and against the tie's parameters, and no new mathematics. 5 is proofs' and is new proof. 7 lands with NET-387. The order that keeps each step true: 1 with its difftest, then 2 and 3, then the guarantee once 5 exists.

[2026-11-12T21:36Z] f2f48ddc191b
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 1:36 PM PST. **Done as you ruled at 1:31 PM PST: the scratch holds proofs' head, the two runs are cancelled, and one run is launched at the merged head.**

- **The merge:** `cursor/na-scratch-net335-7c2e` is at `e44f824ba63b`, pushed: proofs' `8e92d28fe27b` merged once into `454cab1314d2`, no conflict. Over proofs' head it is the same seven files, 351 lines added and 116 removed [run: `git diff --stat 8e92d28fe27b`]. On that tree `test_scan_zk.py` gives 33 passed on the VM [run, on a trial merge of the same tree `cd540f753386`].
- **Cancelled by id** [run: `research cancel`]: run 6 `r20261112-205347-9a48` while it waited (40 minutes, its check never started), and run 4 `r20261112-202915-98ab`, which had run 15 minutes in its slot after waiting 49.
- **Launched once, 1:35 PM PST:** `r20261112-213538-30f1` at `e44f824ba63b`. In order: the Lean try on `NetworkCheck/Period.lean`; the challenger test (the two unlisted cases on both verifiers); the changed suites against `8e92d28fe27b`, each suite's log kept and the failed tests printed by name; the two real rounds (every unit sampled, then a draw of 8 on a local challenger's coins) with the script that reads their registrations and record checks; `scan_period.sh`. The launcher expects a slot in about 13 minutes.
- Job 6's run 2 `r20261112-192107-f9f9` is still running.
- A correction: my line with the two answers is headed 1:36 PM PST and was sent at 1:33 PM PST.

TIMER: none

[2026-11-12T21:37Z] 2c435384e13d
TERMGB to the network-accounting coordinator, Thu 12 Nov 1:37 PM PST, to your 1:16 and 1:31 PM PST messages. **#9454's suites are green at its head: 8 suites, 8 passed. The verdicts' piece is open as #9501, unmarked. None of the four rows moves under lean's head. One thing to weigh before your 3 PM line: `design, no stop block` is not only the whole schedule. It also drops the scan's fail cells, 24,453,012 GB a world-year, and its standing in our document is "a design with no run behind it".**

**1. #9454, yours to read and mark**
- Head `4916fc02307010b555678dc0320ed49f24d1e827`, unchanged since 12:57 PM PST.
- `r20261112-205441-d78a`, the suites `--changed 0e8d16c3b8be --quick` at that head: done, exit 0, 8 passed, 0 failed, none cached [read: its status, and its log through the lane's masked reader].

| suite | verdict |
|---|---|
| `benchmarks/certifier_term` | 15 passed |
| `verity-bench` (the harness) | 563 passed |
| `verity-network-traces` | 652 passed |
| `verity-network-sim` | 21 passed |
| `verity-network-e2e` | 70 passed |
| `benchmarks/one_stage` | 637 passed |
| `repository` | 352 passed |
| `verity-graph` | 220 passed |

- Its body is rewritten: the first screen says the harness passes and names proofs' two commits; the meetings hold the trial merge with lean's head and the site's patch; the runs' table is this run's.

**2. #9501, the verdicts' piece**
- Opened from `cursor/na-stop-verdicts-b4e6` at `a895c57d3120`, unmarked. Its body says it holds #9454's head, names its own three commits, and says the harness's pins wait for proofs' answer.
- It is linked to NET-390. Your order named no issue for the piece. Say if it gets its own and I change the body's last line.

**3. The headline's own figure today** [ran on the VM at #9454's head: `verity bench exfiltration --local-only`]

| in GB | world week | world month | world year | datacenter week | datacenter month | datacenter year |
|---|---|---|---|---|---|---|
| the network certifier's part, "as the scan's check charges it today" (`scan`, D_j = 2) | 973,357 | 4,171,532 | 50,753,639 | 259.6 | 1,113 | 13,536 |
| the compute audit's part | no rows | no rows | 4,887 | no rows | no rows | no rows |
| together, under `scan` | none | none | 50,758,526 | none | none | none |

- The compute audit's record holds rows over the world's year alone, so the other five lines print "no rows of record" and no sum.
- The certifier's part fails with at most ε = 2^-20 over the world's year, at a scan's δ of 2^-63.39. Together fails with at most 0.01.
- It is on the class of at most 1,024 ids a response, modeled.
- The same command's line for `design, no stop block` at D_j = 2: 506,156, 2,169,239 and 26,392,406 GB for the world; 135, 578.5 and 7,039 GB for the datacenter; 26,397,293 GB together over the world's year.

**4. What `design, no stop block` rests on** [ran: the document's terms over the world's year, D_j = 2]

| row | standing | schedules | cells | count | jitter | term |
|---|---|---|---|---|---|---|
| `scan` | on main | 24,453,012 | 24,453,012 | 0 | 1,847,615 | 50,753,639 |
| `design, no stop block` | a design with no run behind it | 24,453,012 | 0 | 91,778 | 1,847,615 | 26,392,406 |

- Both charge the whole schedule, 24,453,012 GB.
- The design's row is lower by the cells. The scan charges a fail cell for each connection. The design pools the failed connections' charge, which rounds to 0 GB here.
- So 26,392,406 GB is supported on Friday only if a run that day is behind the design's pooled fail charge and its head checks. With no such run, the figure that has a check on `main` is the scan's, 50,753,639 GB a world-year and 13,536 GB a datacenter-year. I do not know which of the design's parts Friday's serve runs.

**5. Two lines proofs can act on**
- **The date the ties are due.** The tree holds no date, so it is an option of `verity bench exfiltration` with no default (its options are at `exfiltration.py` lines 926 to 944). `render` (line 738) prints it beside the figures of `doc["stop"]["with_verdicts"]`. With none given, that line prints no date.
- **Which row the headline names.** `render` puts `sums['scan']` in bold at line 771, with the scan's bits a request at line 772 from the row read at line 769, and lists the other rows of record after it at line 773. The names are `TOGETHER`, lines 142 to 146. For Friday the bold figure is the row you rule, and `design` leaves both the bold line and the `together` list at line 791: under #9501 its row equals `design, no stop block`, and its small figure is printed only from `with_verdicts`.
- One more for the same change: under #9501 no row of `TOGETHER` has schedules of 0 [ran: the document at its head], so `at_zero_in` (line 613) is empty and the sentence at line 790, "with a response's output count taken at 0 under ...", names no row.

**6. The four rows under lean's head: none moves** [ran again, 1:25 PM PST, a local branch, deleted, nothing pushed]
- The runner's document at `--response-ids 1024` on the merge of #9454's head with `bafae2eaec2e` equals the head's in every field but `commit`. So `design, no stop block` and `design`, each at D_j = 2 and at D_j = 1, are as before for the week, the month and the year, the world's and the datacenter's.
- `verity bench exfiltration --local-only` prints the same text on the merge, byte for byte.
- The verdicts' head on a merge with `bafae2eaec2e`: clean, its tests pass (6, 63 and 15), and its document equals its head's but for `commit` [ran]. That closes what I had not run at 1:14 PM PST.

**7. TIEVERIFY's answer** (1:30 PM PST) **changes nothing at `a895c57d3120`**
- The registrations are `registrations.jsonl` in a run's `out/`, the name my reader has. The seven names stay. Its writer refuses a second verdict for one class in one run's file [read: its branch's writer].
- Two runs of record that each hold a verdict for one class on one head: my function charges the whole schedule if any of them is refused or not complete, and below it if all are accepted and complete. Both cases are tested.
- TIEVERIFY says its writer checks nothing of a registration and the reader should hold it to the certifier's key. My reader has no key. [argued] A registration that is not checked cannot lower a charge: a class is below its schedule only on four accepted verdicts naming that head's digest, and the tie's verdict names a head only when its registration read under the key. What it can do is leave an execution phase out, which is the coverage gap #9501's body states. Say if the reader is to take the key anyway.

**8. The failed run's line**
- It was in the run's command, not in a file of the tree, so no commit holds it. The notes now hold your citation and the rule for the run of record's command: it ends at pytest and takes pytest's exit as its own.

**Runs** [read: status, 1:37 PM PST]: `r20261112-210949-e63a` (the verdicts' head), `r20261112-202011-ea0c` and `r20261112-204401-0885` are running. 4 PM PST stands for the first.

**A slip:** my first try of the runner, through `uv run`, left an untracked `uv.lock` under `benchmarks/certifier_term/`. I removed it. The tree is clean and nothing of it is in a commit.

**Not established:** the exfiltration command against the store's remote (the read did not end in 150 seconds on the VM; the compute audit's 4,887 GB is from this machine's catalog, record `r20261112-124742-a3fe`); which of the design's parts have a run on Friday.

[2026-11-12T21:40Z] 42796252b9f5
SCANCOUNT to the network-accounting coordinator, Thu 12 Nov 1:45 PM PST. **NET-285 since step 1: the read-back is proved, the view cut has both witnesses, and my files elaborate at lean's mark. NET-401's sizing is below, and its main line reaches NET-285: at the deployed sampled verifier a statement's gates are not its descriptor flattened, so `StmtFlattens` is not a fact of that parser, and `ScanCountBounded`, which asks `Flattens` of each window's statement, speaks of no deployed statement until that is settled.**

Marks: [ran] a run I read, [read] source or a lock I read, [argued] my reasoning from what I read.

## NET-285

Scratch branch `cursor/network-accounting-scan-statements-9e41`, head `07cbbc5bb249`, pushed, no pull request. No lock changed, no `open` entry added.

- **`RowsReadPadded` is proved.** `NetTiming.rowsReadPadded : RowsReadPadded`, in `security/Proofs/NetworkCertifier/RecordRowsRead.lean` (my file), commit `45fac3e427`. Run r20261112-203935-6bf1 at that tree: done, rc 0; lean exit 0 in 117 s; the axiom probe says none of its 10 theorems goes beyond propext, Classical.choice and Quot.sound; no clash [ran]. It takes no hypothesis. It is in no lock, so no check has replayed it through the kernel.
- **`ScanViewCut` compiles at the head and builds on the two files I extended.** `ViewCut.lean`: r20261112-195003-6300 at tree `1c8fa449dcd7`, done, rc 0, axiom probe clean on 6 theorems [ran]. The scan proofs' root file (`security/Proofs/Properties/NetworkScan.lean`, which reads `Accept` and `Fits`): r20261112-200725-65de at the same tree, done, rc 0, clean on 11 theorems [ran]. `scanViewCut` takes `Flattens`, so I do not call it proved.
- **Its two witnesses.** Honest: `Properties.NetworkScan.scan_honest`, the one `scanFits` has [read]. Nontrivial: `Properties.NetworkScan.scanViewCut_nontrivial`, in the new `security/Proofs/Properties/NetworkScan/ViewCutWitness.lean`, commit `17b13e3505`: a Program of one input gate, returned, with no template; its presence gate at step 0 is that input, it has no view gate, and two runs that differ there give two rows. Run r20261112-210850-7796 at that tree: done, rc 0; lean exit 0 in 498 s; clean on 17 theorems; no clash [ran].
- **The nontrivial witness is the model's: no `Flattens` among its binders.** Refuting the conclusion needs an input gate, and `Flattens` then asks `Flock.Program.isInputId` of that gate's key. The kernel does not evaluate it at a string (four forms tried, run r20261112-203040-66b4, failed, rc 1; the string functions it stops on, run r20261112-205104-d136, failed, rc 1, as that probe was built to; both in a local branch, never pushed) and its helper is private to `Circuits.Program` [ran]. The file proves `Flattens` at the instance from that one fact (`oneInput_flattens`) and a `#guard` evaluates the fact. A witness that carries `Flattens` needs one lemma inside `Circuits.Program`, which is not my file.
- **lean's branch, taken once at its mark `bafae2eaec2e`, by merge (`07cbbc5bb2`).** Each of my files elaborates there with a clean axiom probe; four of the five tries exit 1 on the sibling check, which stops on two modules that are not mine [ran]:

| file | run, at tree `07cbbc5bb249` | status | lean | sibling check |
|---|---|---|---|---|
| `security/Properties/NetworkScan/Bound.lean` | r20261112-211232-f83a | done, rc 0 | exit 0 | no clash |
| `security/Proofs/NetworkCertifier/RecordRowsRead.lean` | r20261112-211232-0238 | failed, rc 1 | exit 0, 10 theorems clean | stops |
| `security/Proofs/Properties/NetworkScan/ViewCut.lean` | r20261112-211233-f9bb | failed, rc 1 | exit 0, 6 clean | stops |
| `security/Proofs/Properties/NetworkScan/ViewCutWitness.lean` | r20261112-211232-c5f7 | failed, rc 1 | exit 0, 17 clean | stops |
| `security/Proofs/Properties/NetworkScan.lean` | r20261112-211232-89d3 | failed, rc 1 | exit 0, 11 clean | stops |

  The stop, the same in all four: `Proofs.Pous.FreeBlocks.Zero` and `Proofs.Pous.Sanity` both declare `Pous.seqAccepts.eq_1`. The try names two readings, a clash in the sources or one of the two built against an older build of a module it imports; I did not establish which. `Chain.lean` had no try of its own there: `Bound.lean` imports it, and that try built it [ran].
- **The NET-285 comment of this morning** went as `network-accounting/SCANCOUNT`: `--as network-accounting` is refused to a worker.
- **Held, as you said:** `ScanCountsEntries`, `ScanChainPinned`, `ScanCountBounded`.

## NET-401's sizing (nothing built)

At lean's mark the Properties lock lists the three as `open`, owner network-accounting, LEAN-186, due 30 Nov. `scanFits` takes `Flattens`; `scanScopeAcc` takes `StmtFlattens` and `ScopeMatches`. Neither guarantee is under `deployed` or `trust_gap`, and no other guarantee in the three locks takes any of them [read]. On my scratch branch `ScanChainPinned`, `ScanCountsEntries`, `ScanViewCut` and `ScanCountBounded` take `Flattens` and neither of the other two [read].

**`Flattens P C`**
- What it says: the circuit `C` is the Program `P` in its canonical layout, gate for gate: as many gates as `P`'s root states, an input gate exactly where the primitive is an input, reading exactly the gates its operands name, returning what the root's return resolves to [read].
- Proved or part of the statement: part of what `ScanFits` says. It names the circuits the fit speaks of, as `NciCheck.Properties.Record.Served` does in the same lock under `domain`. The honest witness already concludes it, at the empty Program [read]. So it is a `domain` entry with a one-line why, with nothing to prove; the lock move is lean's and yours to rule.
- With the view cut: `scanViewCut` takes it the same way, and `viewSet` is numbered in `P`'s gates.

**`ScopeMatches D`**
- What it says: the compiled sampled verifier's scope check takes exactly the statements in its sampling scope [read].
- Proved or part of the statement: it can leave. The proof uses one direction, once, to turn `D.scopeCheck x = true` into `D.Scope x` [read, `scanScopeAcc_of_scanFits`]. Stated at a statement in `D`'s sampling scope instead, `ScanScopeAcc` takes neither the scope check nor this premise, and its conclusion is the layout's half alone. Whoever applies it has the sampling scope from soundness: the compiler's contract gives it at an accepted run (`refuses`) [read]. I did not find `Claims.ScopeMatches D.Scope D.scopeCheck` concluded anywhere for a compiled record; the only uses are ours [read, by search].
- With the view cut: no effect.

**`StmtFlattens D`**
- What it says: every statement `D` parses has a descriptor Flock decodes, and the statement's gates are that Program flattened [read].
- Can it be proved of the deployed code: not as stated [argued from what I read; I compiled no counterexample]. The deployed two-stage parser builds a statement as `{ desc := h.desc, prog := bitsProgs h.ms, … }` (`SpotFlock.parse`): the descriptor is the auditor's copy, and the gates are Bits' layout of the members' circuit files [read]. Bits' layout is the constant, the zero, a unit's source gates, then each row lowered to three gates, two XORs and their AND (`Spot.Bits.keyProg`, `lower`) [read]. `Flattens` asks for the descriptor's canonical layout, one gate a primitive instance in the order of the call hierarchy. These are two layouts of one circuit, and nothing in `Held` ties `h.desc` to `h.ms` [read].
- So it is not part of what the guarantee says either: it is the premise that the network check and the sampled verifier read the same gates, and they do not. The network check reads the descriptor (`scanOf`, `scanLayout`, by `Flock.Program.resolve`); the sampled verifier proves statements about the circuit file.
- Two ways it leaves, both invasive:
  1. **A proved bridge**: for an accepted scan, where each boundary gate of the descriptor sits in Bits' layout, and the fit's three halves there. The positions exist as code (`Spot.Bits.inBit`, `outBit`). I found no lemma that relates `Flock.Program.resolve` to them; the nearest work is NCI's record properties (`NciCheck.Properties.Record.*`), which are about a record the scan check does not read [read, by search; not exhaustive]. Causality changes shape: in Bits' layout a unit reads its own gates and source gates, never another unit's computed gate, so "no egress gate depends on a later ingress gate" is a statement about Δ's source keys, not about operands [read for the layout; argued for the consequence].
  2. **The scan check reads the circuit file's layout**: `NetworkCheck.Period.scanOf` and `scanLayout` take positions from what the sampled verifier lays out. That changes the compiled check, its Python and difftest, and `scanFits` is proved again over Bits' layout (`Accept`, `Fits`, `Causal`, `Walk`: about 2,100 lines today). AGENTS.md's rule that Lean holds one description of the circuit, the parsed circuit file the verifier checks, points here.
- With the view cut and the count: `ScanViewCut`, `ScanChainPinned`, `ScanCountsEntries` take `Flattens` of an arbitrary `C` and stand as statements about the descriptor. `ScanCountBounded` asks `Flattens P (sw i).prog` of each window's statement, which is `StmtFlattens` at those statements: under the reading above it holds of no statement the deployed parser builds. Under way 1 my proofs stand and the bridge carries them; under way 2 they are proved again with `scanFits`.

**Can Mon 30 Nov be held, and by whom**
- `Flattens` to `domain` and `ScopeMatches` out of `ScanScopeAcc`: yes, by me, as one change to two statements and their lock entries, once you rule them. Neither needs a new proof.
- `StmtFlattens`: not as a proof of the present statement, by anyone. It needs a decision first, which of the two ways, and that decision is about what the scan's guarantee says and crosses soundness's parser and Bits' bridge. I can write the ask's facts for you. I do not commit either way to 30 Nov without that decision and without reading Δ's source keys for the scan's templates.

## Not checked

- No counterexample to `StmtFlattens` at the deployed code is compiled; the claim rests on reading `SpotFlock/Code.lean`, `Properties/Spot/Members.lean` and `Spot/Bits.lean`.
- I read the two-stage code's parser. I did not read a one-stage code's; `Properties.Spot.Bits` says a statement's program is Bits' layout, which I take to cover it.
- How Δ's source keys carry the scan's state from one bucket's unit to the next: not read.
- Which reading of the sibling stop holds at lean's mark.
- The Site v2 mend is a separate report, by 6 PM PST: its one run (r20261112-213404-d43a) is running.

[2026-11-12T21:41Z] ea6e09008d69
**PERIODPROOF, the 2 PM PST report: B and C are built on lean's marked head in scratch branches, and every check of B's and C's own passed. One thing stops both at that head, and it is not ours: `security/Proofs` does not build there, at `Proofs.Properties.Verity.Sampled.Instance`, so no records run can finish on it. B's records and its unread count are measured at the first take, which builds. C's records run is going at the first take. Nothing of mine is pushed to a pull request's branch.**

Marks: [run] = the run's id and its exit as `research status` prints it; [read] = file and line at the head named; [ran] = a command on my VM; [argued] = neither.

**Lean's branch, taken twice as ordered.** First at `238ad6621c27`, then at its ready line `bafae2eaec2e` (read on LEAN-249 at 12:00 PM PST; still the branch's tip at 12:55 PM PST [ran]). I take it no more.

## B, #9350

- **Scratch:** `cursor/network-accounting-pooled-reg-scratch-8d1f` at **`c7a83e5e9fa5f7233b4cf5a2eee7b3810f30bb33`** = `b782fc88d460` + lean at `238ad6621c27` + my one commit `d17ddb7c2605` (README rows, the docstring) + lean at `bafae2eaec2e`. #9350's branch is untouched at `b782fc88d460`.
- **No stop, at either take.** The allowlist file you expected to stop merges clean under the registered driver. Counted at the merged tree with the lint's own functions [ran]: `Definitions/NetworkCertifier/PooledReg.lean` found 1, allowed 1; `Proofs/NetworkCertifier/Pooled.lean` 4, 4; `PooledRegSample.lean` 2, 2. No count is above the larger side's.
- **Diff against lean's ready head** [ran]: 13 files, +1686 −274: #9350's 12 and the README. None is one of #9286's.
- **The docstring** [read `security/Definitions/NetworkCertifier/Guarantees/PooledReg.lean:30`]: "`y`, and one verifier that judges every statement is the constant function;".
- **README rows** [read `security/Proofs/NetworkCertifier/README.md` 61, 63, 148–152, 187, 295–297, 318, 404]: seven rows, the records paragraph ("the last three", "All 29", "The other 26") and one bullet saying `RateAt` and `PhasesAdaptive` are premises no check derives. The file is 48,875 bytes, 277 under the cap.
- **The lock update, first take** [run `r20261112-175135-7478`, done rc 0 SUCCESS, at `d17ddb7c2605`; outcome `regenerated`; "AUDIT: PASS"; 535 guarantees]:
  - **This guarantee's own records: none moved.** Its 12 entries are regenerated to what the head holds.
  - **Moved, and not this change's:** in `security`'s lock four `reads` entries (`Commitments.FV3`, `Flock.FV3` removed, `NciCheck.Baseline`, `NetworkCheck.Period`); in `security/Properties`' lock the record of `Properties.NetworkPeriod.rowsCapPublished_nontrivial` (signature, type hash) and seven `reads` entries. The scratch changes none of those modules, and its `security/Properties` lock is lean's byte for byte [ran], so they are stale records of lean's branch [argued]. **Put back: I committed none.**
  - Lean named the period check's seven records and that signature on LEAN-249 (1:05 PM PST), and your 1:21 PM PST line says the same. **Beyond those,** my run also regenerated the `reads` entries of `Commitments.FV3`, `Flock.FV3` (gone), `Circuits.Program`, `Flock.Program` (gone) and `Properties.NetworkPeriod`. Lean's locks hold the same text for them at the marked head as at the first take [ran, a compare of the two commits' locks]; `NciCheck.Baseline` alone was put right in between. Lean's own records run writes them all, so this is for its count, not an ask.
- **The lock update, second take** [run `r20261112-200723-13bc` at `c7a83e5e9fa5`, **failed rc 1 UNKNOWN_EXIT**]: **it fails on nothing of B's.** The build of `security/Proofs` stops at `Proofs.Properties.Verity.Sampled.Instance` (ten errors, lines 755 to 823 [read, the run's `lean-audit/audit-Proofs/build.log`]); every other target built, and `security` itself built clean. That file is `main`'s, new since lean's first take and unchanged [ran: one blob at `main` and at the marked head]. It is built there over lean's `security/Properties/Verity/Game.lean`, which adds a field (`macs`) [read, the diff against `main`]. No package got a record; I committed none. compute-accounting's lead reported the same on LEAN-249 at 1:30 PM PST from a run of theirs; mine is a second run that finds it. **B's diff holds neither file,** and B's own lock entries are the same text at both takes [ran], so the first take's run is what B's records rest on until lean's head builds.
- **Suites, once, at `c7a83e5e9fa5`** [run `r20261112-200822-6886`, done rc 0 SUCCESS]: `suites.py --changed bafae2eaec2e --quick`: "9 suites: 9 passed (0 cached), 0 failed" (`repository` 354 passed among them).
- **The Lean check** is the records run: every changed module built, the lock compared and regenerated, no kernel replay. No full replay ran.
- **Gates** against `bafae2eaec2e` [ran]: the unread gate exits 0 ("0 seed file(s) changed, 0 area(s) above the cap … the Lean check's ratchet judges it"). The ratchet gate exits 1 on three sites: `Definitions/NetworkCertifier/PooledReg.lean:4` (an import of `Definitions.NetworkCertifier.Period`), `PooledRegSample.lean:6` and `:40` (the import and `open` of `…Sampling.EndToEndDraw`). All three name modules as `main` names them; the allowlist holds each at its count; the step is off the merge gate. I can't word them away.
- **The unread count, measured** [run `r20261112-193448-e798`, done rc 0 SUCCESS, at the first take `d17ddb7c2605`; `unread.py --build --caps`, the whole of `security/Proofs` built, 14,502 jobs]: **`Proofs/NetworkCertifier` has 4 wholly unread modules holding 34 declarations, at its cap of 4 and 34, not above.** The four are `PeriodWitnesses`, `RecordRows`, `SampledAuditInstance` and `Warden`: none is #9350's. All eight of #9350's modules are read.
  - **Four of #9350's own theorems are reached by no guarantee** [read, the run's `unread.json`]: `Compiled.ksRegR_of_costs` (`PooledRegEvent.lean:178`), `pr_acc_regX_le`, `pr_acc_regTwoX_le` and `phasesMiss_regX` (`PooledRegSample.lean:123`, `:137`, `:247`). They are the event's bound from costs, and the sample's bound at the trace read at the statement (one stage, two stages, and over the period). The cap does not count them (it counts wholly unread modules). NET-382's head reads none of them either [ran, a grep]. Say if you want them out of #9350 at rule 3; I have not removed them.
  - The area's count of unread declarations is 94 of 643; `sampleUniform_val` in `Pooled.lean` is one, and it was unread before #9350.
  - Not measured at the second take: that tree does not build. Lean's change between the takes touches no file under the area's two directories [ran], so I expect the same four [argued].
  - My first run [run `r20261112-175815-219e`, failed rc 1 UNKNOWN_EXIT] gave no count: see Friction.
- **#9350 against `main` with #9286 under it** [ran: `git merge-tree` of `b782fc88d460` with `main` at `0582899c4392`, no stop]: **12 files, +1496 −98.** `Definitions.lean` +1; `Guarantees/PooledReg.lean` +101; `Definitions/NetworkCertifier/PooledReg.lean` +161; the aggregator +9; `Pooled.lean` 139 lines changed; `PooledReg.lean` +115; `PooledRegEvent.lean` +209; `PooledRegLeak.lean` +183; `PooledRegSample.lean` +287; `WitnessesPooledReg.lean` +190; the lock 193; the allowlist 6. The scratch adds the README (22 lines) and the one docstring line.

## C, NET-293's proof

- **Scratch:** `cursor/net293-period-proof-scratch-9631` at **`6602f9212df9ff3e56d0aa14badae48056588785`** = `ff5d7a3d5aec` + lean at `238ad6621c27` + two commits of mine (`9fe161012cd6` a docstring's wording, `9c1cc492678c` two deprecated names) + lean at `bafae2eaec2e`.
- **The first merge stopped in the two files you named;** closed keeping both sides (`Witness.lean`: lean's header and definitions, with `noFormat` [read `:36`, `:55`]; the README: the proof's rows in lean's layout). I took one stale `shape.frozen` entry out of the lock by hand (`NetTiming.Guarantees.PeriodLeakChecked`, "unstated", which the driver kept because lean reworded it). The second merge had no stop.
- **Diff against lean's ready head** [ran]: 10 files, +961 −446 (`PeriodChecked.lean` new, 463 lines).
- **Aggregator try with siblings, at `6602f9212df9`** [run `r20261112-200752-6b81`, done rc 0 SUCCESS]: "lean exit 0 in 1588 s"; no built module outside it declares a name it does (2866 imported, 1185 unchecked); "none of its 40 theorems uses an axiom beyond propext, Classical.choice and Quot.sound". No warning is on a line of mine [read, the run's log]: `Proofs/NetworkCertifier/Checked.lean:309` warns of two deprecated names, and that line is lean's branch's own (its line 294).
- Earlier, at the first take: the same try [run `r20261112-175700-41ff`, done rc 0 SUCCESS] and `PeriodChecked.lean` alone [run `r20261112-190054-a62c`, done rc 0 SUCCESS; 11 theorems, no axiom beyond the three].
- **Suites, once, at `6602f9212df9`** [run `r20261112-205241-f499`, done rc 0 SUCCESS]: `suites.py --changed bafae2eaec2e --quick`: "9 suites: 9 passed (0 cached), 0 failed".
- **The records run: not run at the second take,** where it would stop at the same file of lean's. It is going at C's first-take head `9c1cc492678c` instead (lean at `238ad6621c27`, which builds) [run `r20261112-213657-632c`, launched 1:36 PM PST]. It says whether the proof's own records regenerate to what the scratch holds, and I read every entry that moves. C's lock entries are the same text at both takes [ran].
- **The lock, as the scratch holds it** [read `security/lean-audit.json` at `6602f9212df9`]: `PeriodLeakChecked`'s record lists four `domain` premises (`CommitmentScheme.Binds`, `PeriodChannelReg`, `RecordFormat.ReadsBy`, `WindowsPlaced`) and seven `open` ones, **one of them a new `open` entry: `NetTiming.WindowGoodChecked`** (NET-313, as you ruled for this branch). `PeriodLeakAccepted`'s record and `trust_gap` are lean's.
- **The body** is a file in my note: `lanes/network-accounting/tools/net293carry/pr-proof-body.md`. Its `Goal` line is my guess (`network/period-bound-checked`): confirm it or give me the name.
- **The ratchet gate exits 1 here too** [ran]: 45 sites in five files, nearly all the names `main` gives this guarantee and its modules (`PeriodLeakChecked`, `PeriodChecked`, `PeriodLeakAccepted`, `faultySets`). The new file `PeriodChecked.lean` has 22 and no allowlist entry. Off the gate; I did not touch the allowlist. Tell me if you want entries added at rule 3.

## NET-293's day

**Fri 13 Nov is the earliest I hold for it on `main`, and it is less sure than this morning.** The pull request opens the hour #9375 is on `main`. That day rests on: **lean's head building again** (new today: a fix to `Instance.lean` or `Game.lean`, then lean's own check) and #9375 on `main` by Friday morning; C's records run passing at the first take (going); your read of a 961-line diff and your mark; a land verdict the same day; the queue's own check. Each day #9375 slips moves it a day. The exclusion branch (`03780e9dc98c`) is not in this order and I carried nothing of it.

## What you asked in passing

- **NET-382's two tries** [run]: `r20261112-171739-9616` done rc 0 SUCCESS; `r20261112-165618-f3df` done rc 0 SUCCESS.
- **proofs' #9373** (now at `e11d786b3477` [ran]): it meets B in one paragraph of `security/Proofs/NetworkCertifier/Pooled.lean`'s header; the close keeps my line and its sentence on `pooledIncorrectBlocks_rate`. It also stops in `Guarantees/Pooled.lean` and `verity/protocols/pous/lean/TRUSTED.sha256`, but against lean's branch alone, not against mine [ran, a merge of lean's head with it]. `rateAt_subset` carries `q * P.blocks.length ≤ kd` [read `PooledRegSample.lean:50`]: at rule 3, if #9373 is on `main`, I name `pooledIncorrectBlocks_rate` beside it and restate nothing.
- **NET-382 and `Checked.lean`:** it changes neither `Checked.lean` [ran, an empty diff], so it does not touch `PeriodEscapes`' docstring. C changes the proofs' `Checked.lean` (+26 lines, two lemmas), not the definitions' one.
- **#9332's new name:** B and C import and cite neither name [ran]. NET-382's head imports the old module at `security/Proofs/NetworkCertifier/Pooled.lean:42` and `PooledReg.lean:9`: D changes those two lines. The two heads' files differ in the module docstring only (lines 11–21); the namespace, every statement and every proof are the same text [ran, `diff`].
- **B and C meet each other** in the README's records paragraph, twice [ran]. The second to land keeps both and writes "All 30", "The other 27". Together the file is about 22 bytes over the Markdown cap, a warning: the second trims a row.
- NET-391 goes into #9350's body at rule 3, where the two premises are named.

## Friction

`unread.py --build` fails in a fresh tree: two libraries of one dependency link the same lint plugin at once and one is refused [run `r20261112-175815-219e`, failed rc 1 UNKNOWN_EXIT, after every other target built]. `check.py` builds the plugins alone first; `unread.py` does not. My second run does it by hand. I leave a friction note; the fix is lean's.

## Least sure of, and where I did not look

- **Least sure:** that C's lock is right as I merged it by hand (one frozen entry taken out); its records run is the check, and it has not ended. Next: that the unread count holds at the marked head, which I argued and could not measure.
- I did not build lean's head alone. That its break is not B's rests on the file being `main`'s, on B's diff holding neither file, and on B's change building over lean's first take.
- **Not looked at:** a kernel replay of any package (the records run has none; each try's axiom probe covers its own file); `circuit-check`; the whole check; how #9375 stands in the queue.

**Left, in one line:** C's records run at the first take (`r20261112-213657-632c`), read and sent to you as an addendum when it ends, with the body's lock section filled; then the sizing by 5:30 PM PST. If lean's head changes I take it once more only on your word.

[2026-11-12T21:41Z] e549f1e7627b
**PERIODPROOF, the sizing of the bound in the period check: one function, already in Lean and already read by the properties package; the check's package does not require its package yet. Nothing built. One point needs soundness before it is built: for a stratified sample the bound is over the strata, which `drawn` and `units` alone do not give.**

All [read] at lean's marked head `bafae2eaec2e` unless a line says `main` (`5f0c008aa040`).

1. **The function: `Spot.Bound.boundOf population design δ`** [read `verity/protocols/spot/verifier/lean/Spot/Bound.lean:105`; package `core_spot`]. It is `countBound` over the strata for a `subset` or stratified sample. One-stage publishes exactly it as `bound`, at the profile's own level size, design and δ [read `Spot/OneStage.lean:536`]. So the check recomputes it from the profile it already reads, and an honest answer's `bound` equals it. The check's package `network_check` does not require `core_spot` [read its `lakefile.toml`, same on `main`]: the change adds that one requirement and no second definition, and no cycle (`core_spot` requires `core_protocol`, `flock_verifier`, `lms` and `verity_difftest`). `Spot.Bound` imports only `CoreProtocol.Profile`. On `main` the function is the same with its argument typed `Core.Draw`.
2. **The constant rate is in Lean on `main` and at the marked head, the same text at both** [ran `git diff` of the two files, empty]: `Spot.Select` refuses a fixed count (`FIXED`, `Rule.ofJson`) and parses `rate:`, `stratified-rate:` and `work-rate:`; `Spot.Rate` has `count` and `count_ge`; one-stage applies them at registration (`Spot/Run.lean:263`; line 262 on `main`). All of it is in `core_spot`, the audit's own verifier. **None of it is in the period check:** its parameters name no rate [read `structure Params`, `Audit`], and the profile carries each level's counts, not the rule. So the check can compute the bound at the counts the answer names. It cannot hold those counts to a rate without a rate in its parameters, which changes their format and digest. I read your order as the first only.
3. **Where:** `factsOf`'s last `match` [read `NetworkCheck/Period.lean:1156`], after the three profile refusals: `let need := Spot.Bound.boundOf l.size d prof.delta` with `d` the level's design, then refuse unless `need ≤ b`. Reason: "…: public: bound b: below the sample's bound need at its delta". It covers both callers, `rowsWindowCheck` and `auditCheck` (lines 1179, 1190), so the rows form's scans and the audits of both forms [read `check`, line 1353]. **Mirror:** `_facts` in `verity/protocols/certifier/verifier/network/tests/period_case.py:996`, with `Stratified.count_bound` of `verity/protocols/spot/verifier/profile.py` (exact; held to the Lean by one-stage's own difftest). I would name the mirror's constant `BOUND_LOW`. **Cases that move** [read]: the builder draws each accepted answer's bound at random (`period_case.py:1836`, `:1983`), so it must draw from the sample's bound up; new changes `answer-bound-low` and `audit-bound-low` with their `rows-` twins in `EXPECT`; two `#guard` lines in `Period.lean`. The existing `#guard` at 1485 stays true as written [argued, computed by hand]. **Not read:** which tests of `test_network_period_protocol.py` pin a `wrong` below the sample's bound, and `benchmarks/network_traces/tests/test_certifier_piece.py:163`, a hand-made answer with `"bound": 2`.
4. **The statement:** in `Properties.NetworkPeriod`, in the form its four have (`…At`, `…Of chk`, the `Prop` at `check`): of every period `check` accepts, each audit and each link-window's scan has an answer whose profile has one level `l` with design `d`, whose facts are the public's, and `Spot.Bound.boundOf l.size d prof.delta ≤ f.wrong`. Under it, one lemma of `factsOf` alone. Witnesses: honest at a small accepted period; nontrivial at a check that takes the answer's `bound` as given, which is today's. **Records it moves** [read both locks' sections]: in `security/Properties`' lock a new guarantee with its two witnesses (the `unwitnessed` list only shrinks), `reads[NetworkCheck.Period]` and `reads[Properties.NetworkPeriod]`; in `security`'s lock `reads[NetworkCheck.Period]`; in `network_check`'s lock its `dependencies`. The four proved properties' records should stay, as their statements name `check` and do not unfold it [argued, not run]. No name of it may hold "wrong": that retired name's count is capped.
5. **The day: Mon 16 Nov in review, open and unmarked, if it goes ahead of D**, and D still Wed 18 Nov. After D: Thu 19 Nov. It rests on #9375 on `main` by Fri 13 Nov (the function's type and `factsOf`'s lines are the marked head's) and on line 6.
6. **For soundness, before it is built:** "from `drawn`, `units` and δ" is one-stage's bound only for a `subset` sample. For a stratified one (a tied audit sampled by template, a scan) one-stage's bound is `countBound` over the strata, and the bound of a `subset` of the same totals is another number. With the Python that mirrors `countBound`, over 4,000 random small stratified samples the totals' bound was lower than the strata's in 2,450, equal in 1,550 and never higher [ran on my VM; no proof of "never higher"]. One of them: strata of 9 of 10 and 1 of 10 at δ = 1/100 give 10, the totals 10 of 20 give 5. So a check at the totals would pass a forged `bound` of 5 there, where one-stage's own is 10. I would compute from the profile's design, strata included, which the check already reads and already holds to its level (the three refusals). Then an honest answer is never refused and a `bound` below one-stage's is always refused. Say if the totals alone are meant.
7. **What it does not buy,** as soundness said: `drawn`, `units` and the strata are the same answer's. The statement is about the answer the check read, and `NetworkCertifierSound` stays open under it.

Least sure: line 4's claim that the four proved records stay. Not looked at: the composition difftests that load `period_case` by path, and the benchmarks that feed the check real answers.

[2026-11-12T21:42Z] 02e634be44d0
CERTNEG to the network-accounting coordinator, Thu 12 Nov 1:45 PM PST, to your 1:31 PM and 1:40 PM PST lines. **The dry run's launcher ran at `5f0c008aa040`, and `origin/main` on my VM is still there. Both lines are taken: nothing here fetches `main` until tonight's last launch, and I hold to the posted hours.**

- **The commit** [read]: `origin/main`'s reflog on the VM moved to `5f0c008aa040` at 11:57:36 AM PST and has no later entry; the launcher's cached tree and its `at` file name the same commit; the dry run launched at 12:25 PM PST. It is written in my notes beside tonight's hours (`lanes/network-accounting/`, the plan note of 1:43 PM PST).
- **One fetch of `main` after the dry run, before your rule** [ran]: at about 1:13 PM PST I ran `git fetch origin main` to read the line you named at `main`. `main` was at `5f0c008aa040` then, so `origin/main` did not move. None since.
- **Before each launch tonight:** `git rev-parse origin/main` beside `5f0c008aa040`. If it differs I launch on time and say which launcher ran. `RESEARCH_AT_CHECKOUT=1` stays unset.
- **What I still run, none of which moves `origin/main`** [read: `inbox.py` fetches its own ref only; argued for the rest]: `git ls-remote`, `gh api`, the inbox, `research status` and `research fetch` of my runs, `research notes sync`.
- **NET-396:** the pull request opens after the tries, once #9375 is on `main`. The changed suites' run is still going: 48 of the 52 suites the change reaches have passed, none failed, at 1:26 PM PST [read: its `stdout.log`].

[2026-11-12T21:43Z] dc5d49332501
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 1:43 PM PST. **Job 6: run 2 ended, done with exit 0, and the one run at the merged head is launched. Run 2's review is definition by definition, and it corrects one line of the list I gave you at 11:25 AM PST.** My 1:36 PM PST line said run 2 was still running; its status read done the same minute.

**Run 2** `r20261112-192107-f9f9`, tree `4be378c712b9` (on lean's `ca2e80f1edf7`, not the head you asked the list on) [run: `research status`: done, exit 0]

- **Records with `--update`:** the certifier verifier's package passes as committed. `security` and `security/Properties` fail on record differences alone and pass once regenerated. Nothing is committed from it.
- **Suites** the change from `fcdb33673b07` touches, fresh: 59 of 59 passed.

**What its review says, per definition** [read: the kept `report.json` of each package]

- **Definitions this change changes or adds:** `check` (changed); `okCheck`, `okChecks`, `okChecksAt`, `okRefusal`; `NetworkCheck.ScanOk`'s 23; two of `Identity.Domains` it now reads; `cappedPeriod`, `cappedSchedule` (changed) and six new of `Properties.NetworkPeriod`.
- **Guarantees that read one of them: 33.**
  - soundness, 23 `Verity.*`, `Verity.remote_soundness_of_subprotocols` (deployed) first. Each reads `check`.
  - proofs, 1: `Flock.SecurityProofs.RecursiveAuditC3`. It reads `check`.
  - ours, 9: `rowsCapPublished`, `rowsCapPublished_honest`, `rowsProgWired`, `rowsStateTied`, `rowsTiedCharged` read `check`; `rowsCapPublished_nontrivial` reads `cappedPeriod` and `cappedSchedule` only; and the three new `rowsOkHeld` records.
- **The correction:** I wrote that the 11 `Properties.NetworkPeriod` records each read `cappedSchedule` and `cappedPeriod`. That was the module's readers. By definition, two read them: `rowsCapPublished_honest` and `rowsCapPublished_nontrivial`.
- **Not this change's, at that lock:** the seven `NetworkCheck.Period` names (read by the same 23, the same 1, and 10 of ours), and `NciCheck.Baseline`'s `digest`, `FORMAT`, `FORMAT_V2`. `clockKBoundsCount` and `rowsKBoundsCount` read one of the seven and nothing this change touches. `rowsCapPublished_nontrivial`'s signature moves by the merge (`rowsTerms` gains a field).
- `scanFits`, `scanScopeAcc` and their nontrivial lemmas read the module and none of its changed definitions.

**The one run at `c299bad61230`:** `r20261112-213845-ddd6`, launched 1:38 PM PST, waiting at 1:43 PM PST: the records with `--update`, then the suites changed against `bafae2eaec2e`, each suite's log kept. The exact list comes from its review, the seven set apart.

**NET-335's run** `r20261112-213538-30f1` waits too at 1:43 PM PST.

TIMER: none

[2026-11-12T21:45Z] 8f31db22ce93
SCANCOUNT to the network-accounting coordinator, Thu 12 Nov 2:05 PM PST. **The Site v2 mend is pushed: commit `2bccc178659e` on `cursor/network-accounting-registered-site-v2-7c2e`, one commit of ours on proofs' head `b9ba5acad925`, no pull request. I took proofs' offer whole for the code and changed two things around it; the Python mirror and its vectors needed a mend too. The package's Lean check and the difftest pass. No lock record moves.**

Marks: [ran] a run I read, [read] source I read, [argued] my reasoning from what I read.

## The commit

`2bccc178659e`, "The network certifier's reading of a registered link reads Site v2": three files, 163 lines added, 47 removed.

- `verity/protocols/certifier/verifier/lean/NetworkCheck/Registered.lean`
- `verity/protocols/certifier/verifier/registered.py`
- `verity/protocols/certifier/verifier/tests/test_network_certifier_registered.py`

What it does, as you ruled: a network certifier's link that states no capacity is refused, "link L: it states no capacity; a network certifier's link states its line rate". `Registered.β` stays a `Nat`.

## Against the offer `1005112abb` (the Lean file)

Taken as it is: the body of `registered` (the refusal comes after the key's refusal), the ten literals `β := some N`, "bits a second" for "bits a tick", the docstring of `registered`.

Two places differ:

1. **The module doc's bullet for the new refusal** (the offer's lines 22 to 23; ours 24 to 26). The offer says a valid Site has no such link, "since a link states none only where no certifier splits it". The Site's check does not enforce that: of a link with no β it asks only that `unmediated` is some reason (`Core.Site.Entry.Kinds`, the code's `Site.lean` line 416), and nothing ties that to a link with no certifier at an end [read; every use of a link's β in that file is lines 416, 536, 749, 841]. So a certifier's link with `β := none` and a reason passes the registration's check, and the refusal is what stops it. Ours says that, and sits last, in the order the code refuses.
2. **The guards** (the offer's lines 177 to 180; ours 178 to 186). The offer has one: no β, no reason, the right key. Ours are two: no β is refused with or without a reason; and with a wrong key, the key's refusal comes first. The first pins the case in point 1, the second the order.

## Every other reader of a link's β in the certifier's packages

| reader | against v2 | what I did |
|---|---|---|
| `tests/lean/RegisteredVectors.lean` (the difftest's Lean) | compiles unchanged: it writes `Registered.β`, still a `Nat` [ran, in the difftest] | nothing |
| `registered.py` (the Python mirror) | wrong against v2: it asks v1's keys exactly, so it calls every v2 site malformed, where the Lean reads it [read] | mended: v2's keys for a location, a shared item and a link; the auditor's host as a kind; a figure in the Site's form; β null or a natural; the new refusal |
| `tests/test_network_certifier_registered.py` (its vectors and the difftest) | its sites are v1's, so the difftest fails against v2 [argued; not run before the mend] | mended: sites in v2's form, each new field stated once, the new refusal, v2's malformed forms. 729 vectors, 31 of them the new refusal [ran, locally] |

No other Lean in the certifier's packages reads the Site, and no other Python there reads a link's β [read, by search]. No JSON or TOML fixture holds a registered link [read, by search]. `registered.py` is imported only by its test [read, by search].

**Not ours, listed and left** [read]:
- `verity/hardware/site/__init__.py`, the Python Site: at proofs' head it still writes and reads v1's link (four keys, no `unmediated`). Its difftest (`verity/protocols/spot/verifier/lean/Spot/SiteDifftest.lean`, `tests/test_site_difftest.py` beside it) I did not run.
- `security/{Definitions,Properties,Proofs}/Core/Site.lean` and `security/Proofs/Properties/Core/Site.lean`: the spec's Site and the agreement, proofs'.
- `benchmarks/network_sim/workload/hardware.py`: a benchmark's own links, not the Site's.
- `benchmarks/certifier_term/`: out of the job.

## Shown by

- **The one run, r20261112-213404-d43a, at tree `8e9940c69edb`: done, rc 0** [ran]. It ran two things:
  - `python3 tools/verity/lean/check.py --build verity/protocols/certifier/verifier/lean`: exit 0. "PASS 1557 declarations in 11 modules; axioms propext, Classical.choice, Quot.sound; 4 guarantees". Its build log has `Built NetworkCheck.Registered`.
  - `uv run tools/verity/check/suites.py verity/protocols/certifier/verifier --fresh -- tests/test_network_certifier_registered.py`: exit 0, "105 passed". That is the 104 Python tests and the difftest, which builds `NetworkCheck.Registered` and holds the Python to it on every vector.
- **A try of the one file, r20261112-212227-423c, at tree `f4a9e52809`: done, rc 0**; lean exit 0, so every guard evaluated; no clash; no theorems in it [ran]. Local commit, on proofs' earlier head.
- On my machine: the 104 Python tests of that file pass, ruff is clean [ran].

**The pushed commit is not the tree the run ran at.** proofs' branch moved twice under the work (`e0167418c124`, `d9b3740f8cb1`, now `b9ba5acad925`). The run ran our change on `d9b3740f8cb1`; the pushed commit is the same change, cherry-picked clean, on `b9ba5acad925`. The two trees differ in three files, all under `security/` (`Properties/Core/Site.lean`, `Proofs/Core/Site.lean`, `Proofs/Properties/Core/Site.lean`) [ran: `git diff --stat`]. The certifier's package requires nothing under `security/` and the suite's declared inputs are `verity` [read], so neither part of the run read them. I launched no second run.

## Lock records

None moves [read]. The certifier verifier's lock has four guarantees, reading seven modules, none of them `NetworkCheck.Registered` or `CoreProtocol.Site`; no lock in the tree names `NetworkCheck.Registered`. The package's check passes with its lock as it is [ran]. I generated none.

## Not checked

- The pushed tree itself, as above: the run is one commit of proofs' behind in three files it does not read.
- The whole check at proofs' head, and anything under `security/` there.
- The Site's own difftest, and whether proofs' pull request mends the Python Site.
- Whether a v2 site can be written that the Lean reads and the mirror refuses in a field no vector varies: the vectors state each new field at least once and vary β, `unmediated`, kinds and figures; the others' malformed forms are fixed cases, not seeded.
- If proofs' branch moves again before they pick, the commit cherry-picks onto any head that leaves these three files alone; I do not follow it.

