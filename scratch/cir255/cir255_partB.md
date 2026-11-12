**Across the last ten candidates, the circuit_check suite took 957–2,357 s in the nine that ran it, and 73–82% of that was its session building the catalog apart, which it did every time (702–1,941 s). The suite's time is decided by how many xdist workers it gets, because the build runs one binding per worker. At 5 workers, which the suite has had in its own lane since #9259 (57d5, 1cfd), the build alone takes 1,792–1,941 s. circuit-check's step took 2,990–3,636 s when none of the 826 bindings' records counted, 1,601–1,712 s when about 497 did, and 68–311 s when all did, so records reused decide it. The suite's rise after Thu 00:45Z (1,110.6 s at 145a to 1,323.7 s at 62bf, both on cpu-13) is that same build running on cores shared with research-pool runs. The same bindings took 9–13% more CPU, and nothing in the build's inputs changed.**

All ten ran before ci's 12:55Z placement (circuit-check as shard 5 of five, on cpu-7 alone). No candidate placed that way had finished when I wrote this. Times are UTC, 12 Nov.

### The ten

The build column is how long the suite's other xdist workers sat idle, waiting on the one that builds the catalog. It is read from each pytest shard's `resources.jsonl`: one worker forks a child per binding, about 300–420 of them sampled, one of which grows to 31 GiB on 399–522 s of CPU, while every other worker's CPU stays flat until the build ends. Every row is **[read]** this way. The session's own line ("circuit-check's catalog: …") is in no file the run keeps (see below).

| Candidate | Pytest shard, node | Suite s (workers) | Build s | Tests' sum s | Circuit-check shard, node | Step s | Bindings reused / built | Targets reused |
|---|---|---|---|---|---|---|---|---|
| `1cfd02137251` | `r20261112-122119-2989`, cpu-13 | 2,193.2 (5) | 1,792 | 1,597.8 | `r20261112-122608-503c`, cpu-11 | 1,712.3 | 497 / 329 | 15,287 |
| `57d5b5bb7717` | `r20261112-120631-dd39`, cpu-9 | 2,356.9 (5) | 1,941 | 1,669.8 | `r20261112-120503-b03d`, cpu-5 | 1,600.6 | 496 / 330 | 15,287 |
| `71a2afe671aa` | `r20261112-115416-739c`, cpu-13 | 1,071.0 (15) | 850 | 2,386.0 | `r20261112-115843-56a0`, cpu-11 | 68.4 | (no binding needed) | 17,189 |
| `1885cf1fe08c` | `r20261112-112658-d45c`, cpu-13 | 956.5 (15) | 702 | 2,286.4 | `r20261112-112610-49dd`, cpu-7 | 3,299.7 | 0 / 826 | 0 |
| `5456b937e7e5` | `r20261112-111752-423e`, cpu-2 | 1,468.6 (9) | 1,181 | 2,080.0 | `r20261112-112226-42eb`, cpu-11 | 2,990.2 | 0 / 826 | 5,844 |
| `b5fec34665c6` | `r20261112-110811-3b2e`, cpu-9 | 988.7 (15) | 733 | 2,349.9 | `r20261112-111253-9e0b`, cpu-11 | 3,133.9 | 0 / 826 | 522 |
| `8b536f5c1d62` | `r20261112-104832-adbb`, cpu-13 | 1,088.3 (15) | 835 | 2,289.2 | `r20261112-105242-3b1a`, cpu-5 | 3,243.7 | 0 / 826 | 47 |
| `c330688b9582` | `r20261112-104244-d604`, cpu-2 | 1,089.1 (14) | 832 | 2,233.0 | `r20261112-104725-1e24`, cpu-11 | 3,635.9 | 0 / 826 | 0 |
| `e52d53383d7f` | `r20261112-103655-c9e8`, cpu-9 | 1,390.2 (9) | 1,112 | 1,840.9 | `r20261112-104146-92db`, cpu-5 | 3,252.4 | 0 / 826 | 0 |
| `33866d93e66e` | `r20261112-103110-01a1`, cpu-9 | reused | – | – | `r20261112-103548-2b8c`, cpu-11 | 310.7 | 826 / 0 | 17,189 |

The suite's time is set by four things:
- **The session never had records that counted** **[read]**: in all nine it built all 826 bindings apart. The likely reason is **[inferred]**: the conftest reads `~/.cache/verity-check/circuit-check` on the pytest shard's own node, while the candidate's circuit-check shard runs on another node at the same time.
- **Workers.** `_jobs` builds one binding per xdist worker. At 14–15 workers the build took 702–850 s, at 9 workers 1,112–1,181 s, and at 5 workers 1,792–1,941 s. The tests then take 221–416 s of wall time.
- **Whether `heavy.json` lists the served merge** (`DsaPositionsMerge_v1-served`, 31 GiB, added in `9297d64381` at 07:20Z). With the entry, it starts first, 31–74 s into the session, and runs beside the rest. Without it, its default weight queues it among the light bindings, and `cli.admit` holds it until the running set drains. It then starts 436–908 s in and is the build's last child. That was so at be9a, 145a, 62bf, d607, bcaf, 1a3e, 19fb and 33fa, where the workers resumed within 6 s of it ending.
- **Contention on the node**, below.

### Each later shard started before the records below it existed on its node

Each record time is the step's start plus its `prologue_s`. The bindings' records are stored inside the prologue, so it is an upper bound **[computed]**.

| Candidate | Its circuit-check shard took its slot (`slot.json`) | On | Candidate below: its records stored by | On |
|---|---|---|---|---|
| e52d | 10:42:38 | cpu-5 | 33866d93 stored none: it reused 826 already on cpu-11 | cpu-11 |
| c330 | 10:48:17 | cpu-11 | e52d: 11:03:06 (10:47:36 + 930.3 s) | cpu-5 |
| 8b53 | 10:53:28 | cpu-5 | c330: 11:12:55 (10:50:39 + 1,335.9 s) | cpu-11 |
| b5fe | 11:13:37 | cpu-11 | 8b53: 11:11:25 (10:55:24 + 961.0 s) | cpu-5 |

- c330 and 8b53 started 15 and 19 minutes before the records below them existed, and on the other node as well.
- b5fe started 2 minutes after 8b53's records existed, but those were on cpu-5 and b5fe ran on cpu-11.
- e52d ran on a different node from the records below it. It also changed `targets.py` and `heavy.json`, which may on their own have invalidated every binding **[inferred]**.
- One thing I can't explain: cpu-11 held c330's records by 11:12:55, before b5fe's step began there at 11:16:01. None of the paths that changed between c330 and b5fe is under circuit_check, `verity/definitions` or `verity/circuits`, yet 0 of 826 counted. I couldn't read the records' reads to find which read differed.
- The later rows fit the same pattern. 71a2 (cpu-11) reused all 17,189 targets after 5456 and b5fe had run there. 57d5 and 1cfd reused about 497 bindings.

### The rise at 62bf

Both runs were on cpu-13, in slot `t` (cores 96–191).

| | 145a, `r20261112-004428-54eb` | 62bf, `r20261112-025431-e1c4` |
|---|---|---|
| Suite s (workers) | 1,109 (17) | 1,322 (16) |
| Build s | 839 | 1,005 |
| Tests' sum s | 2,653.5 | 3,266.7 |
| CPU of the four largest bindings, s | 410 / 299 / 275 / 337 | 446 / 339 / 302 / 377 |
| CPU of all build children, s | 6,501 | 7,513 |
| Other runs on cpu-13 during the suite | one check shard, cores 0–95 | eight research-pool runs in r0–r5, cores 96–191 |

- **The growth is mostly outside the tests.** The build grew 166 s of the 213 s.
- **The same bindings took 9–13% more CPU.** Between be9a, the base of both, and 62bf, the only change among the catalog's inputs is `verity/circuits/lean/Circuits/Program.lean`, which the build doesn't run.
- **It held while the pool was busy.**
  - d607 (cpu-13, slot `t`, about 15 pool runs): build 1,055 s, the 21.5 GiB binding at 375 s of CPU.
  - bcaf (cpu-2, slot `b` on cores 4–99, pool slots r0–r3 on cores 4–67): build 1,027 s, 360 s.
  - 1a3e (cpu-13, slot `t`, pool r0–r9 busy): build 1,210 s, 397 s.
  - Against these, be9a (cpu-13, slot `c`, no pool runs): build 884 s, 301 s.
- **Cause: CPU contention from research-pool slots placed on the check's cores.** The pool runs and the CPU figures are **[read]**. That they caused the slowdown is **[inferred]**: I started no run to reproduce it.
- **Ruled out:** a code change, the node, the `heavy.json` entry (neither had it), and the worker count (17 against 16).

### Where the session's catalog line goes

The conftest prints it from `pytest_collection_finish` inside each xdist worker, and xdist doesn't relay a worker's stdout. So it reaches no file the run keeps:
- `suites/tools_verity_circuit_check.log` has 17 lines: the header, the progress dots and the summary.
- The suite's json holds counts, durations and reads.
- The manifest goes to the controller's basetemp, which isn't kept.

Writing the line where suites.py collects it, for instance in the suite's json, would make it readable. That change is in the conftest, which isn't mine.
