[2026-11-13T07:56Z] ee914a0c0e3a
CERTNEG, Thu 12 Nov 11:56 PM PST. NET-423's two questions are answered on the issue (11:53 and 11:54 PM PST), each with a control. No branch was written, because the answer to the first is (b).

**1. The clause: (b). A release inside the slack is not covered by what is charged, so the clause stands as built.** [read, and ran for the control]

- **What is charged** is one bit a bucket, 600 bits a window, on the premise that each bucket leaves at a time fixed before it is made (`benchmarks/network_traces/leak_bounds.py:121-126`; J in `security/Definitions/NetworkCertifier/Assumptions.lean:69-72`).
- **A bucket that reached the pacer late leaves when it is handed**, not at its time (`pacer.py:298-304`). Its time of leaving is then the tick's, which the server's writes move. The slack bounds how late a tick may run, not when a bucket leaves (`PROTOCOL.md:1928-1937`).
- **The control** [ran, on this machine, main's pacer, 240 buckets]: a bucket handed 3 to 40 ms before its time left within 0.004 ms of it (79 buckets). One handed 1, 2, 3, 5, 10, 15, 20 or 24 ms late reached the far end 1.4, 2.4, 3.4, 5.4, 10.4, 15.3, 20.5 and 24.4 ms late (161 buckets). Tonight's try agrees: its two late buckets left 5.4 and 4.8 ms after their times, its 5,009 others a median 0.001 ms after.
- **For (a) to hold, the charge is 2,821 bits a window** against 600 today [derived]: 26 offsets a bucket at 1 ms over 600 buckets. A narrower rule, at most two late buckets a window, is 627 bits, and needs a new statement in Lean.
- **The text already states the clause** as built (`PROTOCOL.md:1907`, `:2237-2238`), but not the reason. Two sentences for it are on the issue, for your ruling.

**2. The node: the balancing across memory nodes and compaction both move the certifier's locked pages there; a run can switch off the first for itself, and the second needs infra.** [ran for the settings and the control; argued that the mover left is compaction]

- **The settings** [ran, job `r20261113-074726-8e09`, launched once at 11:47:25 PM PST, exit 0, 169 s]: `kernel.numa_balancing` is 1 and `vm.compact_unevictable_allowed` is 1. The job is not root and can write neither. The node carried a load of 133 on 192 CPUs.
- **The control** [ran]: the ticker alone with the sampler, one window of 150 s, five at once. Stops above 1 ms from the origin on, count and total:

| Setting, by the run itself | Stops | On a page being moved | Its pages the balancing moved |
|---|---|---|---|
| as tonight | 0 | 0 | 463 |
| huge pages off | 1, 2.1 ms | 0 | 151 |
| a memory policy of its own | 5, 96.8 ms | 3, the longest 67.5 ms | 0 |
| one memory node | 1, 2.5 ms | 1 | 0 |
| huge pages off, one node | 1, 2.4 ms | 1 | 0 |

- **A memory policy takes the process out of the balancing** (0 pages moved, against 463). It does not stop the stops: all three with a policy still waited on a moved page.
- **The node compacted hard** [ran]: 449,301 times in 160 s a task stopped to compact memory; 40.7 million pages moved, 2.3 million of them by the balancing. Huge-page collapse made 2 huge pages, and the certifier holds none.
- **The count does not rank the settings**: a ticker without traffic touches few pages, so one window each is chance. What the control shows is which mover remains, not which setting is best.
- **Tonight's try for scale** [ran, its own record]: with traffic, the ticker waited on a moved page 26 times in 8.4 windows, 364 ms together, the longest 71.7 ms.

**The line for infra:** "On the certifier's node, please set `vm.compact_unevictable_allowed=0` (`sysctl -w`, live, no restart) and say when it is set. It is 1 today, so compaction moves locked pages, and our certifier's ticker waited up to 72 ms on its own locked pages, which refused an honest serve. It changes nothing for a job that locks no memory."

**What an honest try at the same load needs:** that setting (or a certifier's node no other job shares), the memory policy in the run (a few lines, not built), and the server up on the serving node with vllm's yes. It takes 24 minutes from launch to exit (1,392 s tonight) plus the wait for memory.

**Least sure of:**

- **That compaction is the mover, and that the setting removes the stops.** The first is argued from the node's counters: the kernel counts no moves by process, and the sampler cannot read a kernel stack without root. The second is read from the kernel's text, not run. This control once more after infra sets it shows both, in 3 minutes.
- **Whether one bit a bucket covers a bucket handed in time whose write the pacer starts late.** Tonight 3 of 5,009 started more than 1 ms after their time, the latest 2.9 ms. The clause faults that only past the slack.

**Events that were not mine** (your 11:38 PM PST rule): your timer `na-served-negatives-thu-2215` reached me at 11:32 PM PST. I launched nothing for it.

**Next:** the I/O tie on NET-340 before Fri 9:30 AM PST, once vllm's cut is there. Notes: `20261113T0755Z-checkpoint-certneg-net-423-answered-wake-plan-for-friday.md`.

TIMER: `certneg-io-tie-fri-0805` (Fri 8:04 AM PST) and `certneg-after-chain-2` (Fri 7:15 AM PST)

