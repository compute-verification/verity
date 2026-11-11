[2026-11-11T10:43Z] ba47037e829f
Report: CERTSIDE (NET-334), two tries in one job, written 2:44 AM PST Wed 11 Nov. Marks: [read] file:line at my head, [run] a run id or my VM, [argued].

**Waiting on r20261111-102917-1735**, the two-try dry run: launched 2:29 AM PST at `1e81f00ff032` through `research run --on auto` (it went to vy-nebius-cpu-8's research pool), `running` at 2:41 AM PST. It should end about 3:17 to 3:20 AM PST (11:20Z), well before 13:00Z. Resume me when it ends and I send both tries' verdict lines, seconds, the gap, the cgroup peak and `out/`'s size and file count.

## 1. The head

`cursor/net334-certside-562d` at **`1e81f00ff032`**, pushed. One commit on `c77ed3723e`, three files:

```
 certside/job.sh    | 141 ++++++---
 certside/launch.sh |  14 +-
 certside/paced.py  |  26 +-
 3 files changed, 150 insertions(+), 31 deletions(-)
```

What `TRIES=2` does [read certside/job.sh:297, 341-411, 412-479]:

- One `ssh -N` forward, started before the reach check and killed only when the job exits. Nothing in a try touches it but a liveness test.
- Try 1 is the first schedule (`LANES`, `RATE`, `REQUESTS`: 8, 0.2, 240), its files under `out/try1/`.
- If and only if try 1's rehearse and check both exit 0, try 2 starts at once with `LANES2` (64), `RATE2` (1.67), `REQUESTS2` (2000) and nothing else changed, its files under `out/try2/`.
- Each try is one `warden_serve.py rehearse`: its own certifier process, keys, session, run id and watermark stream [read benchmarks/network_traces/warden_serve.py:1137, 1169; verity/protocols/certifier/verifier/warden.py:399, 728].
- Each try has its `summary.json`, `report.txt`, `report.json`, `check.txt`, `rehearse.log` and `warden/`. `out/summary.json` names both verdicts, `between_s` (try 1's rehearse returning to try 2's start), `between_buckets_s` (try 1's last bucket to try 2's first), the cgroup's peak and `out/`'s bytes and files. `out/reach.json` and `out/job.log` are the job's.
- Exit: **0** both ACCEPTED; **1** try 1 is not, try 2 not started, and the job prints try 1's problems from its record; **6** try 1 is and try 2 is not; 3, 4, 5 the reach check's as before; 9 the guard.
- The guard is unchanged: it greps all of `out/`, both tries' directories with it, after the job's child ends by any path.
- `TRIES=1`, the default, runs the same code with the directory `out/` itself: the same 21 files and exit 0 or 1 as before [run, my VM].

`paced.py --ids-below N` (the job's `FAKE_IDS=N`) puts the fake's output ids below N.

## 2. The command lines

From a clean checkout at `1e81f00ff032`, on a host with research's credentials. `ON` defaults to `vy-nebius-cpu-11`; I write it so the node is named.

**(a) Reach check** (waited on, under a minute):

```
ON=vy-nebius-cpu-11 PORT=18746 MODEL=glm-5.3 bash certside/launch.sh --reach
```

Works: `REACHED: a listener at 127.0.0.1:18746 of the serving node held two connects open 3 s each (nothing sent, no round trip timed); the model is MODEL, glm-5.3`, then `certside: guard: no address in out/ (2 files); exit 0`. Before the front is up: `UNREACHED: the key is accepted and the forward opens; nothing listens on the port yet`, exit 4.

**(b) The two tries, for the hour:**

```
ON=vy-nebius-cpu-11 PORT=18746 MODEL=glm-5.3 TRIES=2 REQUESTS2=2000 bash certside/launch.sh
```

`REQUESTS2` is the one you pick from the clock (section 3). It returns when the node starts the run. `MEM_GB` is 4 and `TIMEOUT` 90m by default. No `ID_BOUND`: the prompt ids are drawn below 128,256, inside GLM-5.3's 154,880. When it works `out/job.log` has, in order:

```
REACHED: a listener at 127.0.0.1:18746 of the serving node held two connects open 3 s each …
HH:MM:SSZ certside: try 1: rehearse: --upstream 127.0.0.1:LOCAL --model glm-5.3, --ids --requests 240 --rate 0.2 …
ACCEPTED: 240/240 ended, 0 refused, failed []; 27/27 link-windows certified; …
HH:MM:SSZ certside: try 1: forward: up to the end
HH:MM:SSZ certside: try 1: check: rc 0: ACCEPTED: …
HH:MM:SSZ certside: try 1: done in N s: rehearse rc 0, check rc 0, report rc 0
HH:MM:SSZ certside: try 1 ACCEPTED: try 2 starts, with --lanes 64, --rate 1.67, --requests 2000
HH:MM:SSZ certside: try 2: rehearse: … --requests 2000 --rate 1.67 … --lanes 64 …
ACCEPTED: 2000/2000 ended, 0 refused, failed []; 27/27 link-windows certified; …
HH:MM:SSZ certside: try 2: done in N s: rehearse rc 0, check rc 0, report rc 0
{"tries": 2, "accepted": true, "try1": {…}, "try2": {…}, "between_s": …, "between_buckets_s": …, "cgroup_peak_mb": …, "out_bytes": …, "out_files": …}
HH:MM:SSZ certside: done in N s: try 1 ACCEPTED, try 2 ACCEPTED
certside: guard: no address in out/ (N files); exit 0
```

If try 1 is refused the last lines are `try 1 is NOT ACCEPTED: try 2 is not started. Why, from its record …`, its problems (at most ten), the JSON, `done in N s: try 1 NOT ACCEPTED, try 2 not started`, and the guard's line with `exit 1`.

**(c) One try, the fallback:**

```
ON=vy-nebius-cpu-11 PORT=18746 MODEL=glm-5.3 bash certside/launch.sh                                        # the first schedule
ON=vy-nebius-cpu-11 PORT=18746 MODEL=glm-5.3 LANES=64 RATE=1.67 REQUESTS=2000 bash certside/launch.sh       # the second alone
```

The reader alone, on a fetched record: `python3 certside/report.py RUN/out/try1` and `RUN/out/try2` (or `RUN/out` of a one-try run).

## 3. The windows table for `REQUESTS2`

**What you told vllm stands: 1,800 requests is 8 windows, one fewer than 2,000's 9.** Its room is the smallest of the 8-window rows: 69.75 s between the latest its last response can end and the try's last bucket.

Computed with the generator's own code: `warden_serve.id_traffic(n, rate=1.67, seed=0, prompt_tokens=(16, 1024), max_tokens=(1, 256))` and rehearse's own expression `ceil((last arrival + 120) / 150)` [read warden_serve.py:1160]. [run, my VM]

| REQUESTS2 | last arrival, s | windows | the try's buckets, s | last END if every response runs to its `max_tokens`, s | room to the last bucket, s |
|---|---|---|---|---|---|
| 2,000 | 1,184.6 | **9** | 1,350 | 1,253.00 | 97.00 |
| 1,900 | 1,121.2 | 9 | 1,350 | 1,182.50 | 167.50 |
| 1,800 | 1,062.3 | **8** | 1,200 | 1,130.25 | 69.75 |
| 1,700 | 1,009.2 | **8** | 1,200 | 1,067.00 | 133.00 |
| 1,600 | 952.2 | **8** | 1,200 | 1,030.75 | 169.25 |

- The windows column is the generator's alone. The last-END column is [argued]: my simulation of the admission rule (a connection holds its lane from admission to admission + D + its ids), which matched the smokes' logs. A response is never longer than its `max_tokens` under the cap, so this is the latest it can be.
- In all five rows nobody is refused and nobody is cut. At 2,000 with every response at its `max_tokens` all 64 lanes are held at one moment and no arrival waits; an arrival that did would queue (64 deep), not fail.
- The arrivals do not read the model's name, and the first n of a longer run are the shorter run's.
- Try 1 (240 at 0.2): 9 windows, last END at most 1,167 s, 183 s of room.

**The clock [argued, from your reading of my two dry runs; I replace it with measured figures when the dry run ends]:** from the job's start, try 1 takes about 7 s + 1,364 s. Try 2's first bucket is about 8 s after that. So try 2's last response ends at most about 2,632 s (43.9 min) after the job starts at 2,000 requests, 2,509 s (41.8 min) at 1,800, 2,446 s at 1,700 and 2,410 s at 1,600. The job itself ends about 46.0 min after it starts at 9 windows and 43.5 min at 8.

**The server may stop after try 2's last response and before the certifier's last bucket.** I killed the endpoint 75 s into a 150 s try, after its last END: ACCEPTED 6/6, 3/3 link-windows, and the report says `watermark … 'stopped': 'closed'` [run, my VM]. The check does not read the watermark, and a broken watermark stream faults nothing [read warden.py:728-734; warden_serve.py:802-804]. So against vllm's 16:57:30Z the figure that matters is the last-END column, not the buckets column.

## 4. What I ran [run, my VM, at the head's files]

- Two tries, both accepted (`TRIES=2 MODEL=fake FAKE_IDS=154880 REQUESTS=6 TAIL=20 REQUESTS2=20`): exit 0; try 1 ACCEPTED 6/6, digest `745f2324f184f389`; try 2 ACCEPTED 20/20, digest `bdd6c590ae129dfd`; `between_s` 0, `between_buckets_s` 7.5; guard `no address in out/ (41 files)`.
- **The same through a real ssh forward** (a test sshd on loopback, a `restrict,port-forwarding,permitopen` key with a forced command, the paced fake on 18746): exit 0, both ACCEPTED, `forward: up to the end` after each try, and sshd's log has one accepted login for the whole job.
- Try 1 refused on purpose (2 lanes, queue 1): exit 1, try 2 not started, the problems printed, guard passes.
- Try 2 refused on purpose (1 lane): exit 6, both summaries written, guard passes.
- `TRIES` unset: exit 0, the same 21 files in `out/` as before.
- `launch.sh` against a stub `research`: (a), (b) and (c) expand to `research run --on vy-nebius-cpu-11 --project verity --source … --cwd source --env PEER=… --env PORT=18746 --env MODEL=glm-5.3 [--env TRIES=2 --env REQUESTS2=…] --mem-gb 4 --mem-wait 300 --timeout 90m --custody-ttl 6h -- bash certside/job.sh` (the reach: `--env REACH_ONLY=1 --timeout 3m --wait`).

The dry run's command: `research run --on auto --cwd source --timeout 60m -- env FAKE=1 TRIES=2 MODEL=fake FAKE_IDS=154880 bash certside/job.sh`. Real schedules, both tries; the fake streams an id every 220 ms from 2 s.

## 5. Three facts you did not ask for

1. **Try 2's first 240 requests are try 1's 240 again, field for field** (prompt ids, `max_tokens`, temperature, seed, user): both draw from seed 0 [run, my VM: 240 of 240 equal]. Only the arrival times and the release headers differ. If the server's record or the tie needs them distinct, a `SEED2` is one line; I did not write it, and the table above is seed 0's.
2. **The fake's ids are below 154,880, and 154,879 itself is not among them.** By the generator's code over the dry run's traffic: try 1 emits 7,852 ids, 1,255 at or above 2^17, the largest 154,831; try 2 emits 61,350, 9,432 at or above 2^17, the largest 154,878 [run, my VM]. The id form writes an id in 10 decimal characters, so the top id reads as these do.
3. **The second run on one server is what no dry run of mine shows.** The fake reads the watermark and ignores what it says. A real tick sees try 1's watermark stream end, then a new stream with a new run id whose buckets start again at 0, about 8 s later. That is vllm's side.

## 6. What I did not run

- Anything on a named node. `launch.sh` ran only against a stub.
- The two-try job against the real front or through the real line.
- The real schedules locally: my smokes are one 150 s window a try. The real schedules are the dry run's.
- A refused try 1 or try 2 at the real schedules.
- The whole check or any suite.
- "The six-part sed": I found no such command in the skills, AGENTS.md, `tools/agent-guard` or my brief. I masked every run and launch line with six substitutions of my own (an IPv4, `user@host`, a `PEER=`/`HOST=`/`host=` value, an IPv6, a dotted name under net, io, org, cloud, internal, ai or dev, a port after `-p` or `port`) and cut the status's command column. If yours differs, send it and I use it.

Least sure of: fact 3 above. Next, the pool's nice 19 against the 25 ms slack over 46 minutes in one job; the dry run will say.

Not touched: Slack, Linear, any PR, any branch but mine.

