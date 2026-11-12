# scan_cost

What the network certifier's scan costs to prove and to verify, for each second of execution phase it covers. The
scan is `benchmarks/network_traces/scan_zk.sh`'s: the step-indexed scan of one link, composed in zero knowledge, on a
CPU machine. This benchmark runs it once at a fixed identity, puts two numbers on its honest round with the stages
they are made of, and refuses to write them unless the round's answer is the pinned one.

It states no measured figure. `baseline.json` names the baseline's run, the numbers are in that run's `result.json`
and `out/scan_cost.json` in the evidence store, and `result.py table RUN` prints them.

## The two numbers

A registration holds a group of G execution phases of T ticks each, so a round covers `G · T · tick` seconds of
execution phase. The honest round proves every block of it, one session a member: `first` (tick 0's ingress block),
`ingress-0` … `ingress-(G−1)` (the ingress template's other ticks, a member an execution phase), `last` (the last
tick's ingress block) and `egress-0` … `egress-(G−1)`: 2G + 2 sessions over 2 · G · T blocks.

| Number | What it is |
| --- | --- |
| proving seconds for each second of execution phase | (the round's sessions' seconds + their staging) / (G · T · tick). A session's seconds are its prover process's, start to exit, with the ZK server's work and the waits on the live verifier inside them; staging is each member's statement made ready for its session |
| the Lean verifier's CPU seconds for each second of execution phase, on one core | the CPU seconds, user and system, of the Lean verifier processes that verify the round's sessions / (G · T · tick), each process pinned to one core |

The verifier's wall seconds are written beside its number and are not the number: the core a verifier process is
pinned to, the job's lowest, is one of the 21 the prover's work also runs on, so its wall is contended. The first
number is also written a tick (`proving seconds a tick`). Lowering the class's three Definitions is paid once a
class, whatever the traffic, and is in the result beside the number (`lowering_s`), not in it. The scan's other
rounds (the block's audit and the rounds it expects refused) are run, since the guard reads the whole scan's exit,
and are not counted.

**The stages.** A session's seconds are split by its own `prove.log` (the process's line and the prover's `LIVE`
record) into five stages, in order, that sum to its process:

| Stage | From the log |
| --- | --- |
| to Register | the process's seconds less `e2e_s`: its start, with its statement built, up to its Register |
| Register to the repetitions | `e2e_s` − `tp_s` |
| level 0 | `prove_total_s` less the repetitions |
| the repetitions | the sum of each repetition's `t.prove` |
| the tail | `tp_s` − `prove_total_s` |

The Lean verifier's wall is split by its `lean-verify.log` into reading its inputs, the sampled statement's setup
and the session's checks. The round's sum of each stage is a measurement, the prover's named by the proving number
as its inputs, and `result.py table` prints the round a session a line.

## The identity

Two results compare only at one identity, and `result.py compare` refuses any other pair:

| Field | The benchmark's | Why it is fixed |
| --- | --- | --- |
| `class` | `w63k4` | the Definitions' parameters, so every block's size |
| `sample` | every unit | the honest round proves every block, with no sampling; read from the answers' public parts |
| `ticks`, `tick_ms` | T = 32 ticks of 100 ms: an execution phase of 3.2 s | the class's T and the scan's tick, which the numbers divide by |
| `source` | the synthetic run | the traffic the blocks hold |
| `cores` | 21 | the job is pinned to its first 21 cores, the least a slot of the network pool gives, and its provers size their threads by that |
| `verifier_cores` | 1 | each Lean verifier process is pinned to one core (`scan_zk.sh`'s `SC_LEAN_ONE`) |

The group is a setting, written in the result beside the identity: the numbers are for a second of execution phase
at any G, which is what lets a grouped run be compared with the baseline. The period check's rows form reads one
execution phase a registration (`benchmarks/network_traces/scan_period.py`'s `rows_case` refuses a group of more), so
a result at a group above 1 is the cost of a form the period check does not read today. The machine is not in the
identity; the result's host stamp and `out/host.txt` say what it was.

## The guard

A number stands only where the scan did what it measures:

- `scan_zk.sh` exits 0: every round came out as the scan expects, the honest round accepted and each round it
  expects refused refused at the members it names.
- Each execution phase's answer in `answers.json` accepts, and its public part equals `public.json`'s for the class:
  the bound 0 for each template, every block sampled and proved (over a group, G times the blocks), and the tied value
  the acceptance rests on.

Where the guard does not hold, `result.py write` writes the document with the reasons and a failed validation, no
measurement, and exits 1. `public.json` is a copy of what an accepted scan answers: a change to a class's
Definitions changes its template ids, and the pin with it.

## Running it

`run.sh` is the command of a `research run`, from a clone, in one slot of the network pool, as a job of at most 180
minutes. The slot must give at least 21 cores and 128 GB:

~~~sh
research run --on NODE --project verity --source . --cwd clone --pool network --as LANE --mem-gb 128 \
  --timeout 10800 -- bash benchmarks/scan_cost/run.sh
~~~

It pins the job, clears every `SC_` setting but those below and `SC_MEM_GB`, `SC_CACHE` and `SC_PORT` (so none of
the scan's settings that change what is proved reaches it), runs `scan_zk.sh`, then `result.py write`, which
leaves `out/scan_cost.json` (the document: identity, group, guard, each session with its stages, the round's sums
and the numbers) and the run's `result.json` (the two metrics of `benchmarks/metrics.toml` under their ids, and
their inputs, through the harness kit's writer). The scan's own files stay under `out/` as it keeps them:
`sczk/summary.json`, `sczk/answers.json`, each session's logs, `lean.jsonl`, `rss.jsonl`, `host.txt`. One kind does
not: the verifier's copy of each session's circuit (`sczk/ROUND/verifier/NAME-circuit.txt`), most of a scan's kept
bytes, which no number reads. The scan takes no directory for it, so `run.sh` removes it once the scan has ended;
its SHA-512 is in the session's `prove.log`.

| Setting | What it changes |
| --- | --- |
| `SC_GROUP=G` | G execution phases in one registration (the first target below) |
| `SC_CORES=N` | the cores the job is pinned to; another identity, which compares with no result at 21 |
| `SC_CLASS=test` | the certifier tests' class, to rehearse the runner in minutes; another identity |

~~~text
result.py table DOC                      a result's round, a session a line and a stage a column
result.py compare CANDIDATE [--to DOC]   a candidate beside the baseline, a second of execution phase by stage
result.py baseline [--lock DOC]          the baseline's run, or pin DOC's run as the baseline
~~~

DOC is a `scan_cost.json`, a directory holding one, or a run's id; run them with the workspace's Python (`uv run
python benchmarks/scan_cost/result.py …`). `baseline --lock` refuses a run whose guard does not hold or that lacks
either number.

## Targets

The baseline is the scan as it stands at G = 1: four sessions for each execution phase, each paying its own stages
before and after its repetitions and its own Lean verifier's inputs and setup. Two steps, in order, each a candidate
run compared with the baseline at the identity:

1. **The grouping.** A setting, with no change to the scan: at `SC_GROUP=G` the round has 2G + 2 sessions for G
   execution phases, so `first` and `last` are paid once a group. The target is the proving number below the
   baseline's at G = 8, with the comparison's stage lines saying which stages fell. A group above 1 does not count
   toward the target until the period check reads a group: until then its number says what that change to the
   period check would buy.
2. **One session for a template over the whole group.** A change to the scan's layout: a template's blocks over all G
   execution phases in one session, so the round's sessions no longer grow with G and what a session pays once (to
   Register, Register to the repetitions, the tail, the Lean verifier's inputs and setup) is divided by G. The target
   is both numbers falling with G until the repetitions and the verifier's checks, which are paid a block, are what is
   left. It is a group's form too, and counts on the same condition.

A candidate lands only if the guard holds on its run: a faster scan that answers anything but the pinned public part
has no number.
