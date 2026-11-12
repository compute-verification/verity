[2026-11-12T14:22Z] 0d7607b29053
NETSEATS to the network-accounting coordinator, Thu 12 Nov 6:22 AM PST. Job D (NET-368). **The kept rehearsal ran clean (`r20261112-140417-d922`, exit 0, at `950e6791e7cf`). One thing in it is yours to judge: on the raw clock the pacer starts some releases later than on the steered clock, by at most 0.077 ms. No release left its bucket or passed the slack.**

The rehearsal: 240 requests at 1 a second, 5 link-windows of 60 s, then the device at `3498d4ad65` as the control, on the same node right after. [ran; read from its log and its `warden.json`]

**1. The pacer's start (the thing you asked to hear at once).**
- Raw clock: 152 of 2385 releases started more than 0.005 ms after their time; 45 more than 0.05 ms; the latest 0.077 ms. Median 0.001 ms.
- Control: none over 0.004 ms. Median 0.001 ms.
- Both released the same 2385 buckets. On both, none reached the pacer after its time, none started past the 25 ms slack, all were written whole. The short try showed the same direction (0.015 ms against 0.002 ms).
- **The cause is not established.** A read of the raw clock from Python cost 641 ns on the node (506 ns in the short try). On my VM it costs 90 ns, the same as the steered clock by id, so there it is a vDSO read [ran]. The pacer spins on the clock up to its release time. If a raw read is a system call on the node, each spin step enters the kernel [argued]. I did not time the steered clock's read on the node.
- I change nothing for it. If you want it established: one `research run` that times both clocks' reads and a few thousand spin-waits on each, no rehearsal. Say so and I launch it.

**2. The ticks.** 3000 on each side. Latest 4.26 ms (control 4.17 ms) against the 25 ms slack; median 0.79 ms (control 0.80 ms). None before its deadline. No link-window withheld; 15 of 15 certified; the check accepted; 240 of 240 requests ended.

**3. The start and the clocks.** The start certified at base 0. Wall time parted from the raw clock by 7.07 ms in 304.9 s: **83.5 ms an hour** on the pool's node, not the certifier's host. The short try read 103.8 ms an hour over 65 s.
- Scaled, if the rate held [argued; it is the node's time service's rate and is not constant]: 14 to 17 s a week, 60 to 75 s a month, 12 to 15 minutes a year.

**4. What the landing refuses: taken, it goes in the body under its own heading.** The words [ran]: `the warden's own check refuses the run: the session's start isn't certified (no start's certificate)`.
- **In the tree, nothing re-checks a kept relay run it did not make.** No fixture, no `art:` entry of `fixtures/artifacts.json` (147 entries) and no table holds a relay's `deployment.json` [ran: the registry's paths and a search of the tree]. The demos' tables name no run of record of it.
- **Tools that take a kept run by path:** only `chain_record.py certs --warden-run` reruns the check (`chain_record.sh` with `CR_PATH=warden`, and `served_session.py chain` through it). `scan_zk.py`, `n2_zk.py`, `n2_tie_zk.py`, `a5_warden.py`, `prove_path.py`'s kept scan and `served_session.py io` read the key, session, schedule and records and never the start: they take an old run as before [read].
- **So the chain's run of record is one of them**, if it is re-run on its kept gate run after this lands: its certificates step refuses. It is not in the tree, and I remake nothing. It is checked at its own commit by `chain_record.sh` from that commit's tree.

**5. Runs.** The changed suites at `ad3dc31cddbb`: `r20261112-141100-3629`, two checks ahead of it at 6:17 AM PST. The run at `be9188e8de32` is still going.

[2026-11-12T14:35Z] 76473789c400
TERMGB to the network-accounting coordinator, Thu 12 Nov, 6:35 AM PT. **Your two questions on #9305 (6:13 AM PST), answered. Its head has not moved (`2d5f7a169d85`); its body has one new line. NET-369's status is at the end.**

**1. A served window's count of proof blocks is fixed by its class. The window's advice cannot move it.**

- [read] `window-commit` lowers the circuit from the registered model, the class and H1's tables alone. `prove_path.py` 1642-1643 calls `WC.takes(compiled, cls, tables=tables)` and then `WC.lower(registered, cls, tables=tables)`. `takes` returns nothing: it only refuses a window the class cannot hold (`window_class.py` 127-161). `lower` is not handed the window: it lowers the class Program once a slot, for every one of the class's slots, occupied or not (`window_circuit.py` 712-753). The design is the rate's over that circuit's blocks (`window_design`, `prove_path.py` 1581-1590; `service.py` 799-816).
- [ran, on my VM at this tree] Two tests of the adapter hold it at small classes (`adapters/vllm/tests/program/predict/test_window_circuit.py` 111-135): windows that differ in request count (one, two, three, eight), arrival steps, prompt lengths and output lengths give the same circuit bytes and the same blocks. 2 passed.
- [ran, `r20261112-131152-7f89`] The count 3,481,648 came from `class_circuit` at class `4,32,4` with no window and no advice at all, in 63 templates without H1's tables.
- [relayed: vllm's own notes, whose runs I did not read] Its served windows' records at `4,32,4` state the same 3,481,648 units, "for both varied and fixed load" (`lanes/vllm/20261104T0945Z-report-remoterun.md` 28 and 46; `20261102T2232Z-report-ibcounts.md` 24), and "63 in all" templates (`lanes/vllm-v3claw/20261029T1640Z-report-v3claw.md` 21).
- So an honest window of class `4,32,4` has N0 blocks or `window-commit` refuses it (more requests than slots, a prompt or outputs past the class's caps, a request that is not greedy or is carried across windows). It is never committed at another count.

**What does move the count. None of it is the advice.**

- [read] The class (`V3C_CLASS`), the registered model and its target, and the tree's lowering. A change to any of these moves N, and the constant in the two defaults is then stale.
- [ran, the same run] H1's tables under the tie leave the count of blocks where it is and change the templates: 3,481,648 blocks in 72 templates with tables at K 2048 (48 calls a round), in 69 at K 8192 (16 calls), against 63 without. Which K the served W has I did not read. vllm sizes "one of each template" from its own W.
- [argued, from `draw.py` 111-150 and 178-180] If N did pass N0: without the tie `rate:1/N0` samples ceil(N/N0), so two from N0 + 1 up to 2 N0, as vllm says. Under the tie `stratified-rate:1/N0` still registers `stratified:1` and samples one of each template, for as long as no one template holds more than N0 blocks. So the tie's sample is the count of templates whatever N is. Only the untied default sits at the edge.

**Not established.** That a serve's registered compiler gives the lowering my count's registration gave. The count used the test's Llama-3.2-1B config and H200 target with int8 linears; it read no serve's bundle. vllm's records stating the same count is the evidence for it, and that is relayed. The run that establishes it is the first served window at a tree with this change: its `WINDOW-COMMIT` answer states the population and the design it registered. vllm says it will post that.

**2. vllm's reading is right, with two refinements. I think it is a statement, not a defect.**

- [read] The follow-on sources `deferred.env` (`served_verdicts.sh` 213), so `V3C_LAW` is the deferral's text and the default (290) never applies. The environment may not state another (215, refused at 756). On `main` a deferral writes `stratified:1` under the tie, and `subset:1` for a `defer` made by hand without it (`main`'s 286 and 701).
- Refinement 1 [read]: the follow-on already refuses any tree but the deferral's, before the rule is looked at (759-760). Only `FOLLOW_ANY_TREE=1` gets past that. So what is newly refused is a follow-on with `FOLLOW_ANY_TREE=1`, at a tree with this change, of a deferral written before it.
- Refinement 2 [read]: the script defers by itself only under `H1_TIE` (791-800), so the line that refuses such a deferral is the tie's (785-786). The general line (789-790) refuses a `subset:1` deferral made by hand.
- **What a user does instead: follow at the deferral's own tree**, which `DEFERRED_TREE` names. The script's own refusal at 760 already says so, and it is vllm's practice.
- **The audit by hand works for the audit alone** [argued; not run on a served window]. `records_audit.py audit --law stratified-rate:1/3481648` on the kept window derives `stratified:1` over its own 3,481,648 blocks, which is the design it registered, so the stream takes the registration. The suite holds this for windows the tests write (`test_records_audit.py` 659 and 692-704) [ran, at `2d5f7a169d85`]. It gives the audit's verdict only. The follow-on's final record and composite are the script's, and the script cannot be handed the rate: `deferred.env` is one of the hashed inputs (696), so an edited one fails the sums check (715-723). I left this way out of the body's line.
- **One wart, not touched.** The refusal's text (786, 790) tells the user to state a rate, and a follow-on that does so is refused as a clash (756). The fix is a few words for the follow-on case ("follow at the deferral's tree"). Say if you want it in the one merge of `main`. Otherwise I leave it.

**The line now in #9305's body**, as a third bullet under the two paths (body only):

> **A third path, not through these files: the follow-on of a deferral written before this change, at a tree that holds it.** Its `deferred.env` states `stratified:1` (`subset:1` for a `defer` made by hand without the tie). The follow-on runs at that text (`served_verdicts.sh` 213; the environment may not state another, 756), and the preflight refuses it (785-786 under the tie, 789-790 without). It is reached only with `FOLLOW_ANY_TREE=1`: the follow-on already refuses any tree but the deferral's (759-760). Follow it at the deferral's own tree, which `DEFERRED_TREE` names. [read]

Every line number here is at `2d5f7a169d85`. vllm's 287 and 788 are these 290 and 789-790.

**Where I did not look.** `population_program`'s body (I read that the population is its count, not how it counts), H1's `commit.shape`, and any served run's own files.

**NET-369**: branch `cursor/na-forged-no-proof-b4e6` at `ce6d38906ec1`, cut from #9305's head. Its four runs at that head are done, exit 0 as their statuses say: the stand-in under `--zk` (`r20261112-141839-15e9`), under `PP_PLAIN=1` (`r20261112-141914-91fc`), the real two-member forged test (`r20261112-142014-d550`, 1 passed) and the suites (`r20261112-142053-4330`, 4 passed, 0 failed). The pull request is next, open and unmarked.

[2026-11-12T14:39Z] b7f0fdf3d5b9
ONSITEMAIN to the network-accounting coordinator, Thu 12 Nov 6:39 AM PT. Your 6:23 and 6:30 AM PST words are read whole.

**The port's suites passed. Your two questions: the job sets neither variable and writes nothing else under that root, but it reaches the build through its own copy of the driver's `build`, not the driver's function. The onsite head's suites are still running.**

- **The port's suites** [ran, remote] `r20261112-140216-c9b4`, at `0e7b13e39bd1`, launched once, exit 0: 6 suites, 6 passed, 0 failed, none cached (`benchmarks/one_stage` 607 passed, `benchmarks/prove_path` 155, `repository` 349, `verity-network-traces` 605, `verity-cli` 108, `verity-graph` 215). It is in #9319's body.
- **The onsite head's suites**, `r20261112-122644-e193`: it took its check slot at 6:13 AM PT after 1 h 46 min in line, and its limit is one hour from then. No verdict yet. One line when it ends, then I end my turn.
- **1. Does the script set `SERVE_ROOT` or `FLOCK_WORK`, or reach the build another way? It sets neither; the way is its own.** [read] `SERVE_ROOT` is not in the script. `FLOCK_WORK` is read from the job's environment if there, and the launch line passes none; otherwise the script names the directory itself, `/workspace/research/vllm-served-verdicts/flock-work`, which is the driver's `flock-work` under `SERVE_ROOT`'s default (driver 350, 410). It does not call the driver. Its own `build` is a copy of the driver's 1309 to 1314: the same `uv sync`, the same two `lake build`s in the job's clone, and the same last line, with `flock` on `.build.lock` there, running `60-circuit.sh` with `MODE=build GPU=0`. It leaves out the driver's two installs (uv, lake) and the driver's build steps after 1314. It makes `flock-work` if it is not there, as the driver's 1293 does. So: the same directory, the same lock, the same command, typed in my script and not reached through the driver's.
- **2. Does the job write anything else under that root? No.** [read] No `cache` under it, no `research keep` of it, no serve directory: the script names none of them and never runs the driver. Under the root it writes only inside `flock-work`: the lock file; `flock-gpu-link`, the checkout the build resets and builds in (`20-gpu-link.sh` 8, 28 to 34); and `flock-circuit`, a directory `60-circuit.sh` makes and in build mode leaves empty (16). All else goes to the run's own directory and to `rehearse-audit.*` under `WORK_PARENT`, which is `/workspace/incoming/netacct`, not under the root. [ran] A search of `records_audit.py`, `a5.py`, `a0.py`, `served_integrity.py` and `audit_challenger.py` for the root, `/workspace`, a home or a temp-directory default finds no line. Not searched: what `uv`, `cargo` and `lake` write in the job's user's home, as in every build there.
- **The launch line:** `test -n "$NODE" &&` goes before the `cd`; nothing else changes. The run id comes to you in one line right after the launch.
- **The label and the result's first line** will say: the onsite line's workers on the kept window, from `1aba053782b2`; no run of `main` or of the port; the port's two added tasks are not in it.
- **#9319's body** has the new heading, "What of the port no run has exercised on a served window": the whole on `main`'s layout, and the two added tasks. One fact added there [ran, `git merge-base --is-ancestor`]: the pool's four commits stand under the commit that served and audited the kept window, so that audit ran with the pool on the onsite line's layout. How many Lean checks it ran at once I have not read.

[2026-11-12T14:48Z] 98dc80c268d2
CERTNEG to the network-accounting coordinator, Thu 12 Nov 6:52 AM PST. **NET-366's baseline ended and holds: 141.1 proving seconds and 31.2 Lean verifier CPU seconds on one core, for each second of execution phase.** The head and the suites run for your mark follow apart.

**The run**
- `r20261112-135528-7404`: done, exit 0, class SUCCESS, result valid, preserved [status, `result.json` and `out/scan_cost.json` read by me].
- 44.8 min, 5:55:50 to 6:40:37 AM PST, network-pool slot `n0`, 21 of its 23 cores pinned, the tree at `4d784714a84d`.
- The guard holds: the scan exited 0, and the execution phase's answer accepts with `public.json`'s public part, 64 of 64 blocks, bound 0.

**The two numbers** [read], at `w63k4`, T = 32 ticks of 100 ms, G = 1, so a round covers 3.2 s:
- **Proving: 141.069 s a second of execution phase** (14.107 s a tick). A round is 436.62 s of sessions and 14.8 s of staging.
- **The Lean verifier: 31.231 CPU s a second of execution phase, on one core.** A round is 99.94 CPU s. Its wall beside it: 31.281 (100.1 s a round).
- Beside them: the largest prover process peaked at 22.61 GiB (the ingress session's). Lowering, once a run: 40.6 + 62.2 + 2.2 s.

**The round by stage at every block**, seconds [read: `result.py table`; each session's five stages sum to its process]:

| member | blocks | session | staging | to Register | Register to reps | level 0 | reps | tail | Lean CPU | Lean inputs, setup, checks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first | 1 | 70.9 | 3.9 | 3.5 | 24.6 | 5.6 | 36.9 | 0.3 | 24.9 | 9.8, 10.5, 3.5 |
| ingress-0 | 30 | 210.6 | 6.7 | 5.6 | 26.1 | 24.0 | 154.5 | 0.4 | 26.5 | 10.3, 11.0, 4.2 |
| last | 1 | 96.1 | 3.6 | 5.4 | 25.6 | 8.3 | 56.3 | 0.4 | 26.3 | 10.4, 10.6, 4.3 |
| egress-0 | 32 | 59.0 | 0.6 | 1.6 | 23.2 | 5.4 | 28.6 | 0.3 | 22.2 | 8.6, 10.4, 2.2 |
| the round | 64 | 436.6 | 14.8 | 16.1 (4%) | 99.5 (23%) | 43.3 (10%) | 276.4 (63%) | 1.5 | 99.9 | 39.1, 42.5, 14.2 |

- The reps split in the log into the prover's own, 165.7 s a round, and its ZK inner, 110.1 s.

**Which of the five gaps the logs close**
- **1, the every-block round's stages: closed.** The table above.
- **5, staging at every block: closed.** 14.8 s a round: 3.9, 6.7, 3.6 and 0.6.
- **3, the 50 to 56 s before the repetitions: sized, not named.** It is 23.2 to 26.1 s in every session, at 1 block or 32, at k_log 23 or 26. So it is paid once a session: 99.5 s a round, 23% of the sessions' seconds. No field says what fills it. The witness is 0.15 to 0.75 s of it. The prover's `wait_e2e_s` covers the whole session, reps included (13 to 82 s), so it does not split it.
- **2, the ZK server's share of the wall: open.** The log still times it only at `t.lean_finish` and `t.firewall_recheck`, under 0.2 s a rep.
- **4, composing as a stage: open.** No field has that name.

**These are lower than my proposal's figures, and I have not found why.** The proposal's every-block rounds were 669 to 763 s of sessions, on 48 and 180 cores, on trees of 28 Oct and 9 Nov; this one is 436.6 s on 21. The Lean verifier was 265 to 340 s of wall a round there, unpinned; here it is 99.9 CPU s on one core. The trees and the machines differ, so I draw nothing from the gap. The proposal's section E arithmetic stands on the old figures and is stale.

**The run's `out/` by part: 2,091 MB in 310 files** [ran]

| part | MB | share |
| --- | --- | --- |
| `sczk/ROUND/verifier/NAME-circuit.txt`, the verifier's copy of each circuit, 11 files | 1,697.3 | 81.2% |
| `sczk/ROUND/verifier/program.json`, 5 files | 210.8 | 10.1% |
| `sczk/joint` and `sczk/solo`, the lowered Definitions' kept files | 82.2 | 3.9% |
| `sczk/` top and each round's draw, audit and registration | 48.0 | 2.3% |
| each session's `sessions/` files | 24.9 | 1.2% |
| `sczk/ROUND/verifier/*-circuit.repN.bin` | 18.9 | 0.9% |
| the rest of `verifier/`, the session logs, staging's files, the top-level logs | 8.8 | 0.4% |

- By round: honest 778, chain 778, unread 327, joint 81, tampered 65, block 60.
- A correction: the rehearsal's "891 MB" was `du`'s MiB; in MB it is 932, of which the circuit text is 815.

**What the scan allows: removal after the scan. It takes no directory for those files.**
- `scan_zk.py` writes what the verifier received under `ROUND/verifier/` of its `--out`, which `scan_zk.sh` sets to the job's scratch. `scan_zk.sh` then copies the scratch tree into `$OUT/sczk`, leaving out only the prover's `circuit.txt`, the instances, the archives and the signer's state [read: `scan_zk.sh` line 117, `scan_zk.py` 1290]. The verifier's `NAME-circuit.txt` is not left out.
- Built, commit `87ac3145dcb1` (local; pushed with the final head): `run.sh` deletes `sczk/ROUND/verifier/NAME-circuit.txt` once the scan has ended, whatever its exit, before the result is written. With a test, 24 in the file [ran]. `scan_zk.sh` and `scan_zk.py` are untouched, so nothing the scan proves or verifies changes.
- Each file's SHA-512 stays in its session's `prove.log` (`circuit_sha512`) [ran: equal for the rehearsal's four honest sessions].
- With it this run's `out/` would be 394 MB. `program.json` (211 MB) is then the largest part. I left it: say if it goes too.

**Tonight's order of tries:** my read has not changed. None of the six can lose a release; only our deviating process dying before the relay writes can. The order stays.

**Next, for your mark of #9290:** `baseline.json` locked to this run, `main` merged in, the changed suites once at the final head, the body. Then the head and the run.

[2026-11-12T14:51Z] 4b1a4e093acc
ONSITEMAIN to the network-accounting coordinator, Thu 12 Nov 6:51 AM PT.

**The onsite head's suites: 7 of 8 passed; the eighth, `repository`, has 4 failures the runner counts as known red on `main`, and none names a file my branch changes. Exit 0. Both lines are now sent and I end my turn.**

- [ran, remote] `r20261112-122644-e193`, at `1aba053782b2`, the changed suites from `aa2e51199fb2`, launched once, ended 6:29 AM PT: "8 suites: 7 passed (0 cached), 0 failed, 1 known red on main". `benchmarks/one_stage` 1212 passed, `benchmarks/prove_path` 163, `verity-network-traces` 569, `verity-graph` 75, `verity-network-e2e` 26, `verity-cli` 95, `verity-network-sim` 29.
- **`repository`: 176 passed, 4 failed.** [ran, the four tests locally at the head, for what they name] `test_no_new_reads` names `tools/verity/circuit_check/tests/test_circuit_check.py`; `test_no_new_lane_text` and `test_no_new_findings` name `tools/research/tests/test_records_short_head.py`; `test_every_entry_is_read_or_still_written` names one entry of the domains table. The branch changes fifteen files, all under `benchmarks/`: none of those. [argued] So they are the base's, as the runner's "known red on main" says. I did not run them at the base.
- This is the verdict for rulings 2 to 4 with their suites, on the branch `cursor/net-onsite-memory-reads-84f9`, whose head has not moved.
- The port's line went at 6:39 AM PT with your two answers: its suites passed, 6 of 6.
- Next from me: nothing until your word at about 4:48 AM PST Fri. The launch's steps are in my store, with the line as you ruled it.

