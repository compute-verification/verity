[2026-11-12T08:14Z] 1f029b030d48
WARMLEAN to ci, Thu 12 Nov 12:16 AM PST (08:16Z). **Run 2 is up at two hours, and I take back the ask for three: `r20261112-081109-a8ad` on cpu-14's prebuild place, slot taken 12:11 AM PST, cut at 2:11 AM PST at the latest. Two corrections to what I told you.**

**1. The cut store kept most of run 1's modules (my 12:02 AM line said none: wrong).** I read cpu-14's artifact index by file time instead of by its size: 6,064 module versions were indexed between 11:52 and 11:57 PM PST, 4,617 of them `security/Proofs`', 766 `security`'s, 323 Flock's verifier's. The store goes package by package in path order, so the cut cost the `verity/` packages after `security/Proofs`, the bundles, the verdict pack and `warm.json`: nothing a later run on cpu-14 needs. So run 2 should restore most of the proofs and take about what a records run takes, not 105 minutes. No more time needed from infra.

**2. Why a warm build still replays 2,700 to 3,200 modules: measured now, and it is not only the key's width.** Two machines hold different bytes for the same module under the same build key:

| pair | module versions both hold | differ in `.olean.server` | of those, also in `.olean.private` | differ in `.olean` |
|---|---|---|---|---|
| cpu-9, cpu-4 | 59,120 | 1,983 | 649 | 1 |
| cpu-9, cpu-13 | 51,806 | 1,837 | 692 | 0 |
| cpu-4, cpu-13 | 43,099 | 1,320 | 462 | 0 |
| cpu-9, cpu-14 | 17,206 | 470 | 408 | 0 |

- One cause I can show: 378 of `security/Proofs`' 3,877 `.olean.server` files on cpu-14 hold the path of the scratch tree they were compiled in, and that path has the audit's pid and a random suffix in it (`lean-audit-scratch-PID-XXXX`). No `.olean` or `.olean.private` holds it; why 3 to 4 in 10 of the differing versions also differ in `.olean.private` I have not found.
- The replay key hashes the raw bytes of all three parts and each import's key (`replay_keys`, `check.py`). So a verdict replayed on cpu-4 answers on cpu-9 only for a module whose whole import closure is the same bytes on both. That is why `r20261112-054616-1f66` (cpu-9) imported `r20261112-043326-caef`'s pack (3,416 entries added) and still replayed 2,843 modules after a 40 s build. [The versions and the key are read by me; that this is what cost `1f66` its replays is my inference, not a test.]
- For lean, beside its replay note: the sampled recompile already compares `.olean.server` as an object graph without those paths (`olean_graph.differs`); the replay key does not. Either the key reads the parts that way, or the sandbox compiles at one fixed path, or a node takes the bundle's version over its own. Not mine to change; I put it in the 6 AM report.

Run 2's target is `19fb19a21e40` read against `main` (`0c0efe136ad8`). Its numbers, and cpu-14's cache after it, by 2:30 AM PST.

