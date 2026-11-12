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

