[2026-11-14T01:55Z] 8f9d2c27b2ad
TERMGB to network-accounting (lead), Fri 13 Nov 5:55 PM PST. **#9329's run at its mended head ended 0: `r20261114-012103-6071`, exit 0, at 5:52 PM PST, head printed `fe9296fde681d052f4d703d9a3973144e4a116fb`, the head on origin.** [read: the run's record and its files]

- The six lint files: exit 0, 38 passed.
- `suites.py --changed 0a9a1bc70963 --quick`: exit 0. 4 suites, 4 passed: `benchmarks/one_stage` 1,121 passed, with `test_audit_two_members.py` run whole, slow tests included; `repository` 361; `verity-graph` 220; `benchmarks/prove_path` 195.
- What I did not read: the mended test's own line. It is in the pool's log of the suite, which the run's files do not hold. What I read is that its file ran whole and the suite failed nothing; at `aba14110ec6f` the same file run whole gave the one failure.
- **One thing I said at 5:23 PM PST is wrong:** that `benchmarks/one_stage` would be red in `r20261114-001345-b5be` at `0a9a1bc70963`. It passed there, 1,119 passed, because that file is marked slow and `--quick` runs a slow test only where the change reaches its file, which the merge from `aba14110ec6f` does not. So that run does not hold the test; this one does. That run goes on: 122 of 132 passed, 0 failed, 10 left.
- Body of #9329 set at `fe9296fde681` and read back: the tie's arm green at `aba14110ec6f` with each tie's line, the suites' one failure there, the mend, both runs. It says unmarked.
- #9501: 54 of 61 passed, `verity-bench` failed by the eight, 6 left. #9811: 49 passed and 1 kept, `verity-bench` the same, 10 left. [read at 5:53 PM PST]

TIMER: none.

[2026-11-14T01:55Z] f97ae75a588c
GATERELAY to network-accounting (lead), Fri 13 Nov 5:55 PM PST. **Your third word, answered now from reading: what the split's first whole run needs that I do not hold. Four things, and one question that decides how many of them you ask for.** Nothing below was run.

**The run, in one line** [read]: `served_verdicts.sh` from a tree that holds #9880, with a front's fetch kept in the serve dir, `CERTIFIER_CHAIN_CMD` and `CERTIFIER_PERIOD_CMD` set to the piece's `chain` and `period`, `CERTIFIER_CMD` unset, and V3c at its default class 4,32,4. It needs no change to the table: the two variables are set by hand.

**Two ways to make it** [read: the runner's header and its `v3c` step]
- **Whole:** serve, the front, every step. On the served node, under its GPU lease.
- **V3c alone on a kept serve dir** (`served_verdicts.sh v3c`, with `D` and `K`): it serves nothing, so no GPU lease, no model, no vLLM environment. It makes the window commit, V3c's build and audit, the chain beside the audit, then the period check. This is the cheaper one if a fitting serve dir is kept.

**What I do not hold, and who does**
1. **A served window to run it on.**
   - For the kept way: a serve dir, still on a node, of a served run whose front's fetch is in its `private/certifier` and whose audited window is the front's four requests with its advice written. Which one, and whether one is still kept: not known to me. Whoever made the last served run with `CERTIFIER_FRONT` knows; I made none.
   - For the whole way: the served node's GPU lease and what a served run reads there (vLLM's environment, the row dir, the taps, the weights' snapshot). I have never launched one. By the registry `benchmarks/one_stage/` is proofs'; who holds the node's lease this week is yours to say.
2. **Memory for V3c's audit:** 290 GB by the runner's default for a 4,32,4 window (`V3C_AUDIT_MEM_GB`). My runs so far asked the pool for 16 GB. A node with that much free for the run's length is the pool's, so infra's.
3. **V3c's audit's challenges.** The preflight takes a local challenger only under `DEVELOPMENT=1`. A run of record needs the live challenger: its gate, this node's name at it, and two keys (`PP_CHALLENGER`, `PP_PEER`, `PP_CHALLENGER_KEY`, `PP_GATE_KEY`). Those are proofs'.
4. **The network certifier's gate.** I hold `--gate local`: this tree's gate on a scratch host on the same node, the dry rehearsal. The gate on the auditor's device, with its address and key, is the placement's; who holds it is yours to say.

**The question:** does a development run count as "one whole run of the split passes"? That is: `DEVELOPMENT=1`, the local challenger and the local gate.
- If yes, items 3 and 4 fall away, and I need only item 1 and item 2.
- If no, all four.

**Its length** [read, with one measured part]: the window commit, then V3c's build and audit beside the chain, then the period check. The chain alone took 56 minutes at `--draw 8` on a node tonight. V3c's audit's time at this class: not known to me; proofs know it. The `v3c` step's own limits are 120 minutes for the commit and 180 for the audit.

**What this run would be the first to show** [read, never ran]: V3c's window commit reading the kept scan (`kept_scan`), the chain's scan on V3c's trees, and the period check on V3c's `period.json`.

TIMER: none.

