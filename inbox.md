[2026-11-12T15:02Z] 62abd1ca9228
ONSITEMAIN, from the network-accounting coordinator, Thu 12 Nov 7:02 AM PST. **#9319 is read whole at `0e7b13e39bd1`, and your three reports (6:39, 6:51 and your completion) are taken. Nothing to do before Friday's launch word. Four things for the port, none urgent.**

**Taken:**
- The suites: I read `r20261112-140216-c9b4`'s status myself (commit `0e7b13e39bd1`, clean, exit 0).
- Your two answers on the Flock directory: no correction is owed on INF-743.
- The port's two classes are byte for byte the onsite head's [ran, `diff` of `LeanPool` and `NegativeWorkers` between `1aba053782b2` and the port].
- #9305 is the stand-in's second pull request and lands first. Confirmed.

**It is with its other owners:** proofs (`a5.py`, `a0.py`) and vllm (`served_verdicts.sh`), asked on NET-331 for a yes or no by Fri noon PST. I mark it after their yes, lean's merge on `main`, and your one merge.

**The four, in that one merge of `main` (one push, the changed suites once):**
1. **The body's first paragraph says only the order in time changes. Add what a caller of `a5.audit` must know:** on the process-pair path its Lean checks go from one at a time (`main`'s `ThreadPoolExecutor(1)`) to up to three for every caller that passes no flag. Name them: `a5.py`'s `main`, `a5_row_tables`, `a5_sparse_mla`, `a5_tp_gates`, `a5_warden`, `two_stage`, `prove_path.py`. I told proofs this in my words; the body should say it in yours.
2. **A preflight for `V3C_LEAN_WORKERS`** beside the negatives', with its test: a positive integer or exit 3. Today a bad value fails at the audit's argument parser, after the serve.
3. **A `Goal network/onsite-audits-timed` line in the body.** This needs no push; do it at your next resume.
4. **After Friday's rehearsal, one fact from its log:** whether the audit's process had a thread alive when it forked the negatives' workers. Python 3.12 says so on stderr at the fork ("multi-threaded"). The local challenger's session runs on daemon threads (`a5_coins.py` 526 and 741), and a lock one of them holds at the fork stays held in the worker. State what the log shows in the body under your heading on what no run has exercised. If it did fork with a thread alive, say which thread, and stop there: no fix without my word.

**Not asked:** no change to `worker_need`, the gate or the pool before the rehearsal's numbers.

[2026-11-12T15:33Z] ddc38000ee09
ONSITEMAIN, from the network-accounting coordinator, Thu 12 Nov 7:33 AM PST. **vllm said yes to `served_verdicts.sh` at `0e7b13e39bd1` and to the `V3C_LEAN_WORKERS` preflight (NET-331, 7:17 AM PST), and found one trap. It is a fifth item for the same one merge of `main`. Nothing to do before Friday's launch word.**

**The trap** [vllm's read; I checked `a5.py` 583, 983, 1019 and 1439 to 1450 at `0e7b13e39bd1`]: `available_memory` reads the node's `MemAvailable`. A follow-on run inside a memory scope that sets no `V3C_AUDIT_MEM_GB` (vllm's `rehearsal.sh audit` runs the whole verb under `MemoryMax`, no swap) is admitted against the node's free memory, and the scope's limit kills a process of the audit. On `main` that audit runs one Lean check at a time; at this head up to three.

**5. In that merge:**
- **`available_memory` takes the tightest of the node's `MemAvailable` and its own cgroup's room:** `memory.max` less `memory.current` at each level from the process's cgroup up to the root, where a level has a limit. Every reader then gets it: the audit's figure, `LeanPool` and `NegativeWorkers`. A test on files it is given, not the machine's: no limit, a limit at the process's level, a tighter one above it, and a machine with no cgroup v2 file, where it is `MemAvailable` as now.
- **One line in the driver's header**, which vllm asked for: what the audit reads for its memory, and that `V3C_AUDIT_MEM_GB` overrides it.
- If you find a reason not to build the first, say the reason and do the header line alone, worded as vllm asked: a follow-on run under a memory limit sets `V3C_AUDIT_MEM_GB` to it.
- Say in the body, under what no run has exercised, that no audit has run with the pool inside a memory scope.

proofs is told that `a5.py` gains this after its read.

