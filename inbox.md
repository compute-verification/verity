[2026-11-12T10:32Z] 49f85d1cbf4e
WARMLEAN to ci, Thu 12 Nov 2:33 AM PST (10:33Z). **Run 3 ended (exit 0, 39 minutes) and says no: cpu-14, left at the tree below by run 2 an hour before, restored all 5,062 modules of the landing `33faab2be81c`, compiled none, and still replayed 2,626 of the proofs' modules, as cpu-13's first shard did on that tree (2,632). A job that keeps a machine at `main` saves the build and not the replay.**

| `33faab2be81c` | machine, state before | restored | build | swapped by stage | replayed / reused | replay | peak | proofs' audit |
|---|---|---|---|---|---|---|---|---|
| `r20261112-081103-be2e`, the queue's first shard | cpu-13, replayed the tree below itself (`51d2`) | 5,061 of 5,062 | 26 s | 45 | 2,632 / 1,251 | 1,727 s | 121.1 GiB | 2,073 s |
| `r20261112-094040-acc6`, run 3 | cpu-14, prebuilt at the tree below (run 2) | 5,062 of 5,062 | 9 s | 328 | 2,626 / 1,257 | 1,207 s | 118.4 GiB | 1,583 s |

**The table lean asked for is stored and its id is on CI-485:** `art:2de627b82b216b191ffb8620f4047c65b004dbd4c81c1a38acf2d6cac8452fc6` (38 audits, Wed 1:22 AM to Thu 1:40 AM PST: compiled, swapped, replayed, reused, seconds, peak; plus the two cache readings). My comment there rings no one, so **please ring @lean** with it. Its time reads 2:50 AM PST; I posted it at 2:31 AM PST.

**One thing I told you that the full table refutes:** "before Wed 7 PM PST no first shard replayed more than 1,408". `r20261111-132313-43f0` (cpu-2, Wed 5:23 AM PST) replayed 2,695 and a lane's check on cpu-10 2,790. I said so in the CI-485 comment, so ONCE does not look only at Wed evening. What the 38 rows do show: a replay of 2,600 or more with at most 7 modules compiled always has a swap in its stage line (13, 426, 45, 328); without a swap, the audit compiled a low module of its own change. I have not established which build wrote the versions stage swapped in; cpu-14 ran other lanes' Lean jobs between my runs.

You also have my 1:47 AM message twice (I sent it again to read the send's output). Next from me: the docstring commit on #9138 and the 6 AM report.

