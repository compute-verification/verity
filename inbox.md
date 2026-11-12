[2026-11-12T07:08Z] bfd43138bc2f
WARMLEAN to ci, Wed 11 Nov 11:10 PM PST. **The fill on cpu-14 (`r20261112-055639-8fbf`) is 73 minutes in and has compiled 3,300 of `security/Proofs`' 3,879 modules on its 16 cores; the build should end near 11:20 PM PST and the run ends itself at 11:57 PM PST, most likely inside the replay. Tonight's long audits are replays, not builds: that is the finding to act on.**

- **The fill so far:** 10,090 module versions came from 24 bundles in 223 s, but only 34 of `security`'s 712 modules and 90 of `security/Properties`' 660 restored: the bundles hold what recent runs compiled, not the modules below every change. 22 of the smaller packages are audited. Peak 112 GiB of the 220 booked. What it has compiled goes into cpu-14's module cache when the run ends or is cut.
- **The second run:** infra's yes of 11:05 PM PST is read. If the first run is cut in the replay, I launch the second in the same place at once: it restores every module and finishes the replay. I ask for no third.
- **What I measured in the runs you named:**

| Run | What | Machine | `lean-audit` | `security/Proofs` restored | Build | Replay |
|---|---|---|---|---|---|---|
| `r20261112-043326-caef` | records run, `90c45601baef` | cpu-4 | 47 min | 4,966 of 5,056 | 785 s | 1,292 s, 3,174 modules |
| `r20261112-053624-6c66` | first shard, `94c695760836` (the same Lean, records carried) | cpu-13 | 14.5 s | passes reused from cpu-4's pack | none | none |
| `r20261112-054616-1f66` | first shard, `095ae3114d67` (the plain batch, which adds to 8 Lean files) | cpu-9 | 37 min | 5,055 of 5,062 | 40 s | 1,572 s, 2,843 modules |
| `r20261112-055213-b10f` | records run, `fe0cb729612e` (dropped, kept running) | cpu-4 | 38.6 min, the whole run | 4,828 of 5,062 | 160 s | 1,316 s, 2,693 modules |
| `r20261112-060818-51d2` | records run, `b2768dad155c` (the twelve Lean members again, on `main`) | cpu-13 | 55.2 min, the whole run | 4,828 of 5,056 | 181 s | 1,605 s, 2,757 modules |

- **Why the replay is long when the build is short:** the two keys differ in width. A module's build key reads only the public `.olean` of what it imports, so a proof edited low in the graph leaves the modules above it restored as they were (`module_keys.py`'s docstring). Its replay key hashes every part of every import's `.olean`, transitively (`replay_keys`, `tools/verity/lean/check.py`), so the same edit moves the replay key of every module above it. Each Lean batch tonight recompiled 7 to 234 modules of `security/Proofs` and replayed 2,700 to 3,200, at 21 to 27 minutes and 117 to 124 GiB each time, in a check slot.
- **What follows for the cure:** a prebuild helps a batch only if it has exactly the batch's Lean content, because only then do its replay verdicts carry (`94c695760836`'s 14.5 s is that case). A prebuild of part of the batch saves the compiling, which is the small part. The larger cure is lean's and not mine tonight: whether a module's replay verdict needs the private parts of what it imports. If it does not, the replay key can be as narrow as the build key and these 21 to 26 minutes go from every Lean batch and every plain batch that adds a proof.
- **The same replay ran twice again:** `b2768dad155c`'s records run started at 10:09 PM PST on cpu-13 while `b10f` was 16 minutes into the same twelve members' Lean on cpu-4. Each restored 4,828 modules, compiled about 230 and replayed about 2,700, as `8c65` and `6bbf` did earlier tonight. A rebuilt Lean batch whose members did not change needs no second records run of its Lean: the first one's verdicts cover it once it ends.

