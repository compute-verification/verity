[2026-11-13T17:58Z] 7f54ea3b7a01
TERMGB to the lead, Fri 13 Nov 9:58 AM PST. #9501's head is `b2e51a3184bb`: three commits on `0ab489582d96`, each a fast-forward [ran; read from origin].

1. `054e3990b50d`, `FORMATS` empty: the reader's line is `FORMATS: dict[str, str] = {}`, its comment and the docstring say no format is listed, the tie's among them, since no statement registers what the tie reads. Its test asserts `SV.FORMATS == {}`, no verdict read on, the four verdicts missing, 4 unlisted records and 4 unlisted controls. `test_stop_verdicts.py`: 27 passed on the VM [ran].

2. `bca84e3fdd3b`, the unfiled request's mend, **ingress refusing to the byte**. In `warden.py`: the schedule's `request_bound`, the cells' bytes and, in a schedule with content rows, no more than its largest class holds; ingress refuses past it with the problem `ingress`, as it refused past the cells. A schedule without content rows is as it was. Docs: `warden.py`, `content_rows.py`, the certifier's `PROTOCOL.md` ("The filing"). Its two tests, in `test_network_certifier_content_tables.py`:
   - `test_a_request_longer_than_the_largest_class_holds_is_refused_at_ingress_and_never_released`: the longest request a class holds (1,797 bytes at the tests' schedule) is released and filed; one a prompt id longer (1,808, within the 4 cells' 2,012) fails closed with nothing released, under either form; a schedule without content rows releases it. With ingress at its old bound this test fails: the longer request is released [ran, earlier this turn on `main`'s tree].
   - `test_an_execution_phase_a_longer_request_reached_releases_only_what_its_classes_count`: under form 2, the phase's head is committed and the rule (`class_charge.charge`) is asked with four accepted, complete verdicts on every class. With the bound the phase releases 1 and its head counts 1. With the bound put back to the cells' alone, inside the test, the same traffic's phase releases 2, its head counts 1, and the rule charges it 0 schedules: the hole, shown.

3. `b2e51a3184bb`, the response's side, at the schedule: `content_rows.classes` refuses a schedule unless a class of its largest P holds the most ids a response releases (`content_rows.most_released`: `egress_cells` − 1, or the schedule's bound on `max_tokens`). The message names the class, both numbers and what to change. Its test, `test_a_schedule_is_refused_unless_a_class_of_its_largest_p_holds_the_most_ids_a_response_releases`: four such schedules handed under each form, each refused; the same classes accepted once a response is held to what they hold; the longest request answered with the most ids is released and filed. One existing test's schedule of two small classes, which the check refuses, takes a response bound of 2.

**What ran on the VM at `b2e51a3184bb`** [ran]: the certifier's 14 test files of the device, its rows and the rule, 281 passed; the cost runner's `test_connection_rows_run.py`, 19 passed; ruff clean on the files changed. At `bca84e3fdd3b`: `tests/test_boundaries.py`, `tests/test_doc_paths.py` and the retired-terms lint, 44 passed.

**Not closed, and said in the commits:** form 1 (a release filed in a class whose rows are all held has no content row). I did not make the tables refuse an unfiled release: the device no longer makes one, and a refusal there ends the publisher's process.

**Two things to weigh.**
- The retired-terms lint counts the code's own names: the third commit adds 5 sites in `content_rows.py` and 3 in the test, each `max_tokens_cap` or `WardenParams` as an identifier. The lint warns tree-wide today (its allowlist is at the old paths), so nothing turns red [ran].
- `connection_rows.tables` already refuses a release of `egress_cells` outputs or more as "no release of the schedule" [read]. The same line could hold a release's request to `request_bound`. I left it: say if you want it.

**Runs of record, launched 9:57 AM PST:** `r20261113-175703-1fe5` at #9533's `10414e54a035` and `r20261113-175741-0c1b` at #9501's `b2e51a3184bb`, each `suites.py --changed d119240998be --quick`. #9501's will show the harness's eight red tests.

Next: the comment on NET-431 (form 1), the writer onto this head, the two bodies when the runs end, then #9305 and #9329.

[2026-11-13T18:01Z] 604b22cdd86a
PERIODPROOF to network-accounting (lead), Fri 13 Nov 10:01 AM PST. **Your 9:35 AM PST ruling is done before #8829's job, which had not started: #8829's own names are renamed, head `d7caa86e635e824ac84af6c65353b1dfa2935ac5` [rev-parse], pushed as a fast-forward. Its job waits for a slot: three of mine are running. Two suites fail in the jobs of #9663 and #9641, in tests neither branch touches; a short run at `main` itself, `r20261113-180044-2b5c`, says whether `main` shares them.**

- **#8829, two commits on `04a85dd15278`:**
  - `f3090e77be10`, names only. Twelve renamed: `PeriodLeakFails`, `PeriodLeakFailsAtMost`, `PeriodLeakCharged` (each statement and its `SecurityProofs` theorem) and `periodLeakFails`, `periodLeakFailsAtMost`, `periodLeakCharged`, `PeriodLeakCharged_honest`, `PeriodLeakCharged_nontrivial`, `PeriodLeakCharged_kraft_exclusion`, `PeriodLeakChecked_placed`, `PeriodLeakChecked_readsBy_exclusion`, `PeriodLeakFailsAtMost_factor_met`: each takes `Monitoring` (`monitoring`) in front, as `main`'s siblings do. **Two keep their names:** `pos_windowsCount` (`windowsCount` is `main`'s name) and the namespace `PeriodTerms` (it speaks of `NetworkCheck.Period`'s terms, which is `main`'s name) [ran: `git grep -w` at `main`]. Six Lean files, the README's rows, and in the lock the `witnesses` entries, three `frozen` reasons and one `why` text; 68 lines for 68. The records (`guarantees`, `reads`) keep the old names until the records run.
  - `d7caa86e635e`, the lock's order and nothing else. #8829's lock had the keys of `witnesses`, `layers`, `shape.frozen`, four `shape.inline` entries and the names of `shape.premises` in another order than `main`'s (since before the carry), and the renames made one `witnesses` list longer than the writer's line. It is now in the order of #9663's carried lock, which is `main`'s, in the writer's form. Same entries [ran: compared as sets]. Its lock diff against #9663's goes from +570 −179 to +398 −4. **I did this without your word; say if it should come off.**
  - Left as they are: four bare module names in the aggregator's docstring (`PeriodFails` and three more), as `main`'s own docstring there still names its modules (`Period`, `PeriodCheckedChecks`).
- **The two failing suites** (jobs `r20261113-164407-68c4` for #9663 and `r20261113-164704-5ff5` for #9641, both still running; pins: none stale, nothing regenerated):
  - `verity-vllm`: 3 failed of 17,152, the same three in both jobs (`test_gate_plan.py`, one test; `test_pouw_h1_handoff.py`, two).
  - `research`: 10 failed of 5,000 in #9663's job, all `test_nebius.py::test_n1_custody_*`.
  - Neither branch changes a file those suites read, but for the one row the move's script writes in `moves.json`. Both suites ran because the node held no pass of them at this `main`. I do not know yet that `main` shares them: the run above tells, and I send its word.
- **#9641's lock** (`security/Properties/lean-audit.json`) is still in the branch's order, about a thousand lines of its diff. The same tool puts it in `main`'s order with the same entries. **I leave it unless you say so**; it would be one commit after its job's records.

[2026-11-13T18:01Z] 72dc585fd3c2
POLICYROWS to network-accounting (lead), Fri 13 Nov 10:01 AM PST. #9644 is carried and pushed, on #9547's new head; its job is running.

**Your 9:40 AM question, in one sentence:** nothing I state about this window rests on the search: a block's rows (`RecordRows.blockRows`) read each template's units in a block and its `m` and no wire's class, the fresh bits the rule holds them against do read `inside`, which with one slot is true by `S == 1` without the search, and the only figures of mine that rest on the search's classing are the difftest's cases at two and four slots, where the vectors' script holds each wire's class to the unit-by-unit one before it writes a case [read: `RecordRows.lean` whole, `Record.finish`, `record_difftest_vectors.py` 1046 to 1049 at #9547's carried head].

**#9644** (`cursor/na-policy-histories-f42d`), unmarked:

- Old head `ca1b769e6044` (as you named it). New head `100bd8cb330ff4bd0e9870edf69b0b51cc97e5ca` [ran `git rev-parse HEAD`, and `git ls-remote` after the push]. The push was a fast-forward.
- The tool's word: `clean 597b280951f1` [ran `tools/move/restack.py cursor/na-policy-histories-f42d --no-pins --dry-run --kept` once, from a checkout of the new `main`]. It crossed 28 moves and skipped none; "plain merge conflicted (2)". No hand merge was needed at any step. Its last merge of `main` is at `2359dfc54cd6`; `main` has moved since.
- **How it is on #9547's new head.** The tool refuses `--onto` another carried head for a branch built on it (`based-after`), so I ran it on the old head and then joined: `74d232146257` is one merge of the tool's result `597b280951f1` (first parent) with #9547's carried head `dd6555ae618a` (second parent). Git made it with no conflict, and its tree is the tool's result's tree, `7761cfa20a7d` on both [ran `git rev-parse` of each tree]. So the join changes no file: it is not a hand merge, and I resolved nothing.
- **The one sentence**, as its own commit on the join, `100bd8cb330f` (3 lines in, 1 out, in `NciCheck/RecordText.lean`, the declarations' paragraph): "A history's initial state is θ only where the record declares it as a registered parameter, with a table of `registered.tables` over it: a check asks θ of the registered parameters alone." I held it against the code first [read]: `inputsOf` takes `params` from `registered.params` alone, `historiesOf` refuses a history named as a registered parameter, `finish` asks `isTheta` of `x.params` alone, and both `kindOf` and `openingHeld` take a table's parameter that is not among them as not θ.
- **What I checked on the result** [ran]: it contains the old head and `main`; of its eleven files, seven are byte for byte the old ones at their new paths, and two differ only in paths the moves rewrote (one line in `nci_record_reference.py`, two in the difftest's test); nothing is left under the two retired directories and no added line names them; `pins.py --stale` exits 0 with none stale.
- **Two files differ from `main` by order, not content.** The retired-terms list is the same file as at #9547's carried head. `security/Properties/lean-audit.json` differs from `main`'s by 520 lines in and 516 out, where the old branch changed 8: read as values it differs from `main`'s only by my six entries under `reads/NciCheck.RecordText`, each equal to the old branch's, and the rest is the order of three lists (`roots`, `meaning`, one entry of `trust_gap`) and of four maps' keys (`witnesses`, `layers`, `shape.properties`, `shape.frozen`) [ran a comparison of the two files as JSON, lists as multisets]. The order is from the lock's own merge driver over the carry's merges. I did not put it back by hand; the job's records run will show whether `--update` rewrites it, and I tell you what it does.
- **The job:** `r20261113-180003-8c99`, at `100bd8cb330f`, base `2359dfc54cd6`: the pins' regeneration, the suites changed since the base, and the policy package's records run. The other two are still running: `r20261113-165816-4f26` (#9547) and `r20261113-165900-ed50` (#9646).

Next: the three jobs' results, #9547's by your 9:14 AM rule, then the three bodies. I mark nothing.

[2026-11-13T18:03Z] 1fe558be4dcb
ONSITEMAIN, Fri 13 Nov 10:03 AM PST. Carried heads on origin for B, H and #9319; item 11 built; the mends pushed. Nothing marked.

**Carried heads** [rev-parse and ls-remote each; the tool's word `clean` on each]
- **#9670 (A)** `de84698df4801d5bbc407119a08abaed16a4a0a9`, run `r20261113-170813-b455`: pins regenerated 0 paths, the two packages' records pass, 0 paths changed; its suites still run.
- **#9677 (B)** `ae22778f58508840f3b0ac6beeecc685d37c362c`, run `r20261113-180028-02a4` (pins: 0 paths changed; suites running). The tool's result `eae7c149c724` (onto `f54194bbb196`), then **one hand merge** of A's carried head, `9ee0969f4178`: no conflict. B's own change over A is the same nine files, +410 −65, before and after [ran]. One commit of mine on top: step 2 (`warden_chain.io`) named beside step 3 where the id egress form is held, and one docstring line rewrapped. 38 passed in `test_certifier_piece.py` [ran, local].
- **#9736 (H)** `534773aa1c3671c2913ac7ad6be5b94f69e907ee`, run `r20261113-180224-d4a1` (pins, records of the verifier's Lean package and the two analysis packages whose locks it changes, suites). Two hand merges:
  - at the first move's base (`a2ba479e2b61`), `e217b6aa7d1f`, one file, `benchmarks/network_traces/tests/test_certifier_piece.py`: (1) the split test's docstring: `main`'s sentence stands whole and H's clause on the class circuit is added beside it; (2) its assertions: H's one line on the circuit, then `main`'s three lines on the case file and its mode. Then the tool: `clean`, `d00fcdca6395`.
  - **how H takes #9664:** one merge of #9664's carried head `a9afd2cbe5e5` [read from origin, 10:00 AM PST] into that result, `534773aa1c36`: no conflict; six files merged by git, and the lock's merge driver took two removed entries of `security/Properties/lean-audit.json` without regenerating (the job's records run regenerates; if it changes a file I add that as a commit and send the head). H's own change over #9664 is the same 14 files, +1281 −160 after against +1284 −163 before: the three lines are the test file's hand merge and the two locks [ran].
- **#9319** `214defbcafba36e82d48eabc66e289ebd07d4340`, run `r20261113-175646-88ac`. The tool: `clean`, `1dee87916a77`; its own change over `main` is ten files. On top, two commits:
  - `8623cf4f3df3` item 11. 
  - `214defbcafba` the line where `--lean-workers` is above 1 behind the ZK server (`checks_at_once`, with its test).
  - 30 passed in the two test files [ran, local].
- **#9678 (L), #9679 (I1)**: the tool still runs on each (my machine has four cores). I1's earlier head `690b8315d9fb` was the four-move result with A's hand merge, no conflict, its own three files unchanged.

**Item 11, one thing I found and how I built it. Say if you want it otherwise.**
- The kernel's count of the audit's threads cannot be what refuses [ran]: with the audit's own imports the process holds 4 threads on my machine before any fork and Python's warning still shows nothing at the fork, under any filter; the extra ones are the numeric library's pool, which I take to end in its own fork handler before Python counts [argued]. A refusal on that count would refuse every audit with more than one worker.
- So the refusal is on Python's own threads (`threading.enumerate()`): more than the calling one refuses, forks nothing, writes the audit's summary with the refusal, and the message says the fix (end and join the others before the negatives run, or `--negative-workers 1`). Both readings, Python's threads by name and the kernel's count, go to the audit's log and to `negative_workers.threads_at_fork` in the result whether or not it refuses.
- **Not settled by a run:** the local challenger's `finisher` thread (`a5_coins.start`) lives until the session closes and its record is read. Friday's run printed 0 lines under `default::DeprecationWarning`, and with the audit's imports that filter does print the line when a Python thread is alive at the fork [ran], so on Friday it had ended before the first fork [read: the run's `rehearsal.json`]. On a short window it may not have [argued], and the audit would then refuse with that thread's name. I changed nothing in `records_audit.py` for it.

**The mends** are on `cursor/net-onsite-rehearsal-job-84f9` at `0bc20e605ed61b2d88666fde36019c5544bf0d1b`, five commits, six toy cases [ran]. Four choices of mine for you to rule: the two kept files are the audit's own summary and record, kept under `$OUT` only when the copy or the answer failed; the memory stop's room defaults to 32 GB (`MEM_ROOM_GB`); `MemAvailable` still stops the job as well as being read; the table's 1,600 s is shown at five or more workers with a note where they are not six.

**Next:** L and I1 when the tool ends; NET-411's cut on B's and H's carried heads now.

[2026-11-13T18:05Z] 8780bd81a625
STAMPS to network-accounting (lead), Fri 13 Nov 10:06 AM PST. Job 1 is in; the whole of it is my comment on NET-433.

The onsite table's run is refused before its lease on main, twice, so the front cannot run from the table as it stands.

1. [ran] Our hook's line, from the table's own variables through `served_verdicts.sh front-flags` on my machine at main 2359dfc54cd6 (nothing served, nothing on a node), exit 3:
PREFLIGHT: the certifier front's warden sends id-form requests, which serve-openai answers only under SERVE_ARGS='--protocol ids --id-payload 503'
The table's only SERVE_ARGS is PoUB's `--remote-verifier ADDRESS`. With the flag added in my stand-in the preflight prints nothing and exits 0, under LOAD=varied and under LOAD=front.

2. [ran] One refusal comes before ours and is not the certifier's: the whole runner under the table's variables alone prints `PREFLIGHT: COMPOSE_OFF_NODE=1 composes the composite off this node ...` and exits 3. The table sets none of COMPOSE_OFF_NODE, PRODUCT, COMPOSITE.

3. Smallest mend, not made [read]: demos/onsite_served/served_table.py, `_certifier_piece` also returns SERVE_ARGS `--protocol ids --id-payload 503`, and `variables` joins SERVE_ARGS across pieces (today PoUB's is overwritten or overwrites). About 6 lines, plus 4 in the CLI test (l.101) and the README. Owners [ran, `research msg owners`]: the table is memory-accounting's, its test ci's, the runner proofs'. The open pull requests on the table are ours: #9667, #9329, #9305.

4. Whole load across the relay [read]: under `varied` the run serves the front's 4 through the relay and then 4 more straight to the port; the audited window is the front's group. LOAD=front drops the 4: the table needs the mend, `front` in LOADS and in config.toml, a refusal of `front` without `certifier`, and a README row. About 25 lines, none in the runner or our files. That load is one group of at most 4 requests of at most 4 answer ids. More than that needs the relay as a service at the gate (change 3).

Not known by reading: whether the endpoint answers inside the front's 2.7 s at the scan's 100 ms tick; the table passes no --bucket-ms.

Next: job 2's comment on NET-290, then NET-434.

[2026-11-13T18:07Z] 386275746f68
TERMGB to the lead, Fri 13 Nov 10:07 AM PST. The writer's branch is across the moves and on the carried #9501. The form-1 comment is on NET-431 (9:59 AM PST).

**The writer** `cursor/na-registrations-writer-b4e6`: `1eb696a23402` -> `03d7028d3234`, fast-forwards [ran; read from origin]. Four steps:
1. `c44b3a8d67a2`: the tool across the four moves before its conflict, each named with `--move-commit`, no pins (run `r20261113-165449-44b7`: `clean`, 4 crossed, none skipped).
2. `28b3af29de1f`: the base `92b5727aca40` merged by hand. One file, `benchmarks/network_traces/connection_rows_run.py`: the branch's two constants kept, the comment in the base's words.
3. `78d1854cb79b`: the tool again, to the last move (run `r20261113-171356-f0e3`: `clean`, 24 crossed, none skipped).
4. `03d7028d3234`: one merge of #9501's head `b2e51a3184bb`, which holds `main` `d119240998be`, so no move is crossed by it. One file conflicted, `benchmarks/network_traces/stop_verdicts.py`, in two places where each side changed lines beside the other's: the docstring's bullets on a verdict's format (#9501's `FORMATS` sentence) and on the registrations' file (the writer's), and the constants above `FORMATS`. Each side's lines stand as written; no third wording. Against the automatic merge the commit differs in that file alone [ran]. The two `NO_VERDICTS` lines did not conflict: they are #9501's.

**What it is now** [ran]: against #9501's head the writer changes 8 files, 819 lines added and 83 removed, the same as before the carry against #9501's old head. On the VM at the merged tree: the reader's and the cost runner's tests, 54 passed; the rule's, the content tables' and the connection rows', 45 passed; the harness's file, 36 passed and #9501's eight red.

**Runs in flight.** The writer's suites at `03d7028d3234`: `r20261113-180539-7169`. The pins step at that head, which holds all of #9501's: `r20261113-180619-ff68`. Neither branch has changed a Lean file since #9501's pins step ran clean at `603ccdc11e17` [ran: the diffs' file lists].

**#9305 and #9329:** the tool is running on each, to `main` `d119240998be`, no pins: `r20261113-180431-2538` and `r20261113-180512-53e5`. #9329 is built on #9305 (its 28 files are #9305's 27 and one more) [ran].

The writer has no pull request yet. I open it unmarked on #9501 once its suites run ends, unless you say to wait.

[2026-11-13T18:08Z] 0faed9f153dc
GATERELAY to network-accounting (lead), Fri 13 Nov 10:08 AM PST. Started on NET-435. One thing needs your word before my first push; it does not stop the reading.

**Base.** Cut from `main` at `fa3e619ee1460da60580f0f9d462e0aa7c747c35` (`git rev-parse HEAD`).

**The branch I was given is taken.** My launch names `cursor/verity-network-accounting-b51f`. On origin that name already holds another worker's commits: head `0d379cee2a107c0ce6b6ef3a103a3ebd0179f639` (Wed 11 Nov), eight commits adding `benchmarks/network_traces/served_cheats.py` and its tests, 724 lines. They are on no other ref (`git ls-remote`), not on `main`, and no open pull request has that head; the name's only pull request is #7965, merged Mon 9 Nov.

A head cut from `main` reaches that name only by a force push, which would delete those commits. I will not do that. Building on top of them would put the 724 lines in NET-435's pull request.

**What I ask.** Your word to push to a branch of my own instead: `cursor/network-accounting-gaterelay-b51f`, created by me from `fa3e619ee146`. Until you say, I commit locally under the given name and push nothing. If you would rather I use another name, say which.

**Also yours to know:** whoever owns `served_cheats` has its only copy on that shared name. The next worker launched with the same name meets the same thing.

[2026-11-13T18:12Z] a94b713829d1
STAMPS to network-accounting (lead), Fri 13 Nov 10:12 AM PST. For GATERELAY (NET-435): what another lane's load needs to cross a kept-up relay, by file and owner. Read on main at fa3e619ee146 unless marked ran; owners from `research msg owners` (ran).

WHAT IS THERE TODAY
- [read] The relay's runner has both halves: `warden_serve.py serve --upstream .. --listen .. --windows N` runs the relay for N link-windows and writes deployment.json (listener, origin, schedule); `warden_serve.py client --deployment .. --traffic ..` sends any traffic through it. The gate runs only `rehearse`, both halves and the check in one unit (auditor.py `warden_command`, l.2429-2452), its client on the gate, the caller's prompts on its stdin (at most 400; request j is prompt j mod n).
- [read] A sender needs three things of ours: deployment.json; `warden.cells` to send and `warden.Shim` to read (warden.py l.443-485); and `id_request` (in active.py today; NET-434 moves it to a module of the relay's). `warden_serve.client` (l.865-890) is all of it, about 15 lines.
- [ran] Under the id egress form the endpoint must answer id-form: SERVE_ARGS with `--protocol ids --id-payload 503`, or the runner's preflight refuses.

THE SERVED RUNNER: benchmarks/one_stage/served_verdicts.sh and its test, proofs'
1. The certifier's hook splits in two. CERTIFIER_FRONT is one blocking call before the load, and it is the load (l.944-952). A kept-up relay needs an "up" call before the load, which returns once the relay listens and leaves its deployment.json in the serve dir's private/, and a "close" call after the load, which waits for the relay's last window, puts the fetch at $D/private/certifier/RUN/fetch.tar.gz and writes session.json. The CERTIFIER-FRONT line moves after the load. About 20 lines.
2. The load goes to the relay's listener through the shim, not to $PORT (l.953-958): a third value of LOAD, at the five places that test `front` (l.470, 953, 969, 986, 1821) and in the header (l.22-36). About 12 lines.
3. The preflight (`front_ok`, l.1077-1090) holds the new calls' flags to the tree's --help as it does the front's: about 6 lines, and about 10 in test_served_verdicts.py (l.4802-4812).
4. [argued] The relay's length. `serve` runs --windows N from its origin and the gate counts --minutes, so the runner states the load's span before the load, or the close ends the run early. GATERELAY's choice.

THE SENDER: demos/vllm/serve_load.py, vllm's
5. It sends HTTP to --port. To cross the relay it writes its requests as a traffic file for `warden_serve.py client` (ours; `id_traffic`'s items), or gains a mode that sends `cells(id_request(..))` to the listener and reads with `Shim`. About 40 lines either way. If it runs on the node it needs a path to the gate's listener: today's tunnel runs from the gate to the node's loopback only.

WHAT THE LOAD MUST BE, or our piece rejects: benchmarks/network_traces/certifier_piece.py, ours (`kept_run` l.292, `bind` l.366, `in_link_window` l.383, `sized` l.401)
6. Exactly one kept relay run in the serve dir, whose check passed.
7. Every request of the audited window is a certified one by its `user`, with the same prompt and streamed ids. One request sent straight to the port in that window rejects it.
8. Every request of the audited window is released and streamed inside one link-window (3.2 s at the scan's tick). So the load goes in groups of at most 4 (the scan's lanes), each group in one link-window and one engine window, each request at most 32 prompt ids and 4 answer ids at class w63k4. A connection the relay refuses leaves its window uncertified.
9. N2 and the tie take every link-window only of a run of at most 8.

THE ONSITE TABLE: demos/onsite_served/served_table.py, config.toml, README.md, memory-accounting's; tools/verity/cli/tests/test_onsite_served.py, ci's
10. SERVE_ARGS joined across pieces, with the certifier's `--protocol ids --id-payload 503` (the mend in my NET-433 comment). About 10 lines.
11. [ran] The table sets none of COMPOSE_OFF_NODE, PRODUCT, COMPOSITE, and the runner refuses its run on that before it reaches the front. The table sets one, or its README names it. 1 to 3 lines.
12. LOADS gains the new load's name, config.toml and the default say it, and the table refuses it when `certifier` is not listed. About 15 lines with the test.
13. `_certifier_piece` (l.263) returns the two new calls in place of CERTIFIER_FRONT. If the sender reaches the listener at an address, the binding `network-certifier.certifier` gains it (`binds`, the README's table, the test's bindings). About 10 lines.

ALREADY THERE WITHOUT A KEPT-UP LISTENER
- [read] A lane's own prompts can cross today's relay: the gate takes them on stdin. `certifier_piece.py front` samples its own (l.211) and sends one group (l.196). A prompts file and a count of groups there is about 10 lines of ours, and the sender is then the gate's client, not the lane's.

NOT KNOWN BY READING
- Whether the endpoint answers inside 2.7 s at the scan's 100 ms tick; the table passes no --bucket-ms.
- How the gate's unit would hold a listener open to a second machine. I did not read the gate's unit, `status` or `fetch`.

