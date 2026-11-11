[2026-11-11T09:36Z] 32493809d258
CERTSIDE (NET-334), report 1, written 1:36 AM PST Wed 11 Nov. Marks: [read] file:line at my head, [run] a run id or my VM, [argued].

Two dry runs are running; resume me when they end: **r20261111-093055-db8d** (first try's schedule) and **r20261111-093135-0133** (the second's). Both on vy-nebius-cpu-1's research pool (`--on auto` spilled there: vy-nebius-2's ssh did not answer), launched 1:31 AM PST at `b68a193c5e28`; at 1:36 AM PST the first is `running` and the second still `waiting` (the pool's slot waits for its memory), so the second ends later than the first, each about 25 min after it starts. The timings you asked for from them come when they end; below is what I have without them.

## 1. What would stop the real run (say to vllm before 7:30 AM PST)

Nothing in `warden_serve.py` or `warden.py` at this head stops it by itself. Five things the id front must do, or every connection fails; none is in VLLM-324's Part A as I read it (I read Part A only, 1:00 AM PST):

1. **The watermark stream goes to the same port as the requests.** `serve` opens it with the same `--upstream` [read benchmarks/network_traces/warden_serve.py:777-780]. With `--watermark` the front on 18746 gets, besides id requests, one long `POST /verity/watermark` (chunked, never answered) and must hand it to the tick. If it doesn't, the tick never hears a bucket close. The forward allows one port, so there is no second way in. The fake does read it; a real front I have not seen do it.
2. **The front must close the connection right after the trailer (`io_tie.TRAILER`: `data: [DONE]` and the event's end).** The certifier releases END only once the upstream has ended and less than a whole cell is waiting [read verity/protocols/certifier/verifier/warden.py:693-696]; a front that keeps the connection open after the trailer fails that connection `late` at the END's slot [read warden.py:702-703]. And the bytes after the last id event must be exactly `io_tie.TRAILER`, else `format` [read verity/protocols/certifier/verifier/io_tie.py:98, 151-155, 170-181; warden.py:704-706]. No HTTP head on the response: the first byte is the first id event's.
3. **Each id event is exactly P = 4087 bytes** (`io_tie.id_event(x, 4087)`: the id right-aligned in 10 characters, spaces before the blank line) [read io_tie.py:114-123]; any other byte fails `format`.
4. **The model's name is at most 64 bytes as JSON** (62 characters and its quotes) [read verity/protocols/certifier/verifier/active.py:524, 561-565]. A longer one raises in the client before a byte is sent. The job's reach check exits 5 on it [run, my VM].
5. **Without `MODEL`, both the reach check and `rehearse` call `GET /v1/models`** and need exactly one model listed [read warden_serve.py:1156]. If the front does not answer it, give `MODEL=NAME`; then nothing but id requests and the watermark goes through the forward.

Also for vllm: `id_traffic` draws prompt ids below 128,256 [read warden_serve.py:222]. If GLM-5.3's vocabulary is smaller, set `ID_BOUND=N` (each id x becomes x·N // 128,256; nothing else of the traffic changes [run, my VM: same arrivals, lengths, max_tokens, seeds]). I do not know its vocabulary size.

## 2. The branch

`cursor/net334-certside-562d`, head `c77ed3723e` (pushed; cut from `9d5b8fb7e128`). The directory is **renamed `tick-sunday/` → `certside/`**. Nothing under `verity/` or `benchmarks/` changed.

```
 certside/bounded.py |  38 ++
 certside/job.sh     | 382 ++
 certside/launch.sh  |  64 ++
 certside/paced.py   |  60 ++
 certside/report.py  | 152 ++
 5 files changed, 696 insertions(+)
```

Six commits: Sunday's two files as they were; job.sh; report.py; launch.sh; the timings and the paced fake; `RECHECK` through launch.sh. The dry runs' tree is `b68a193c5e28`; `c77ed3723e` after it changes only launch.sh (one more variable passes through).

Two files you did not ask for: `bounded.py` (the prompt-id bound, since `id_traffic`'s 128,256 is a literal in `benchmarks/`) and `paced.py` (below, section 5).

## 3. The commands

From a clean checkout at the head, on a host with research's credentials:

**Reach check** (under a minute, waited on, exits with the job's code):

```
PORT=18746 bash certside/launch.sh --reach                 # the front answers GET /v1/models
PORT=18746 MODEL=NAME bash certside/launch.sh --reach      # it doesn't: two connects through the forward
```

It prints `launch: the serving node vy-spot-glm53's 127.0.0.1:18746 from vy-nebius-cpu-11, tree c77ed3723e[, model NAME]`, then research's lines. The job's own lines (also `out/job.log`; `out/reach.json` has `reached, exit, class, why, port, by, tunnel_s`, no peer):

- works, no MODEL: `REACHED: <model> at 127.0.0.1:18746 of the serving node, GET round trips [five] ms`, exit 0.
- works, MODEL: `REACHED: a listener at 127.0.0.1:18746 of the serving node held two connects open 3 s each (nothing sent, no round trip timed); the model is MODEL, NAME`, exit 0. It takes 6.5 s.
- tonight, before the front is up: `UNREACHED: the key is accepted and the forward opens; nothing listens on the port yet`, **exit 4**, in 0.2 s, no hang, in both modes [run, my VM, through a real sshd with a `permitopen` key and a closed port: ssh logs `open failed: connect failed: Connection refused`, the same line your r20261111-091407-4f49 had].
- exit 3: `forward forbidden` (`administratively prohibited`, tested ahead of `open failed`), `key refused`, no key file, `ssh refused`, `no tcp` [run, my VM, each but `no tcp`].
- exit 5: reached, and the model's name is too long.
- last line always `certside: guard: no address in out/ (N files); exit RC`.

**First try:**

```
PORT=18746 [MODEL=NAME] [ID_BOUND=N] [MEM_GB=4] bash certside/launch.sh
```

Every schedule and traffic number is the job's default (the schedule you gave; `CAP=256`; `--watermark`; no `--temperature`). It returns when the node starts the run. When it works `out/job.log` ends:

```
ACCEPTED: 240/240 ended, 0 refused, failed []; 27/27 link-windows certified; cells in their bucket N/N; … tick lateness {… 'over_slack': 0}
HH:MM:SSZ certside: forward: up to the end
HH:MM:SSZ certside: rehearse: rc 0
HH:MM:SSZ certside: check: rc 0: ACCEPTED: …
{"accepted": true, "released": 240, …, "seconds": {"reach": …, "venv": …, "rehearse": …, "check": …}, "max_rss_mb": {…}, "cgroup": {…}}
[the report, also out/report.txt and out/report.json]
HH:MM:SSZ certside: done in N s: rehearse rc 0, check rc 0, report rc 0
certside: guard: no address in out/ (N files); exit 0
```

27 = 3 links × 9 windows. The registered digest at the first try's schedule is `745f2324f184f389…de1f4eb1` [run, my VM: `deployment.json`], at the second's `bdd6c590ae129dfd…` [run, my VM, from `WardenParams`].

**Second try:** the same with `LANES=64 RATE=1.67 REQUESTS=2000` in front.

Other variables: `RECHECK=0` skips the job's second `check` (rehearse has already run the same check and written `check.json`); `TAIL`, `REQUESTS`, `DELAY`, `BUCKET_MS`, `QUEUE`, `SEED`, `CAP`, `PROMPT_MAX`, `MAX_TOKENS`; `TIMEOUT` (90m); `KEY` (default Sunday's path); `ON`, `MACHINE`.

The job opens exactly one forward (`ssh -N`, one `-L` to `127.0.0.1:PORT`, no command, no second forward, no agent, no tty) [read certside/job.sh, the ssh_config block and the ssh line]. Exit 9 is the guard: a file of `out/` holds a non-loopback IPv4, an `@` before a name, or the peer's host; it names the file.

**The reader, alone:** `python3 certside/report.py RUN/out` (stdlib only), prints and writes `out/report.json`.

## 4. The local smoke's last lines

[run, my VM, `RESEARCH_RUN_DIR=/tmp/smoke5 FAKE=1 REQUESTS=6 TAIL=20 bash certside/job.sh`, at the head's job.sh, exit 0]

```
ACCEPTED: 6/6 ended, 0 refused, failed []; 3/3 link-windows certified; cells in their bucket 684/684; cells {'ingress': 15, 'egress': 684}; tick lateness {'n': 600, 'p50': 0.701, 'p99': 1.236, 'max': 2.127, 'over_slack': 0}; …
09:29:31Z certside: rehearse: rc 0
09:29:32Z certside: check: rc 0: ACCEPTED: 6/6 ended, …
{"accepted": true, "released": 6, "peak_open": 5, "watermark": {"last": [599, 6], "lost": [], "error": null}, "out_bytes": 655526, "seconds": {"venv": 1, "reach": 1, "rehearse": 158, "check": 159}, "max_rss_mb": {"certifier": 118.98, "publisher": 77.57, "largest_in_rehearse": 118, "check": 80}, …}
connections [warden.json]: arrived 6 (6 with an ordinal, 6 logged), refused 0 (windows []), admitted 6, released 6, ended 6, failed 0, cut by the run's end 0 queued and 0 served
  failed by reason: late 0, limit 0, upstream 0, ingress 0, malformed 0, format 0, cap 0
wait for a lane, buckets (admitted − arrived, the admitted): n 6, min 0, p50 0, p90 0, p99 0, max 0; 0 waited
ingress, buckets (ingress_end − admitted + 1, the released): n 6, min 2, p50 2, p90 4, p99 4, max 4
failed late: 0; the egress cell of each FAIL: []
link-windows [check.json]: 3 of 3 certified, cell rows 2 of 2 registered, withheld windows [], signer refused 0
tick [check.json]: lateness ms {… 'max': 2.127, 'over_slack': 0} against a slack of 25 ms; cells in their bucket at the client [684, 684]; ticker stalls {'episodes': 0, 'max_ms': 0.0}; the certifier ran at nice 0 on 4 cpus
09:29:32Z certside: done in 159 s: rehearse rc 0, check rc 0, report rc 0
certside: guard: no address in out/ (21 files); exit 0
```

Also [run, my VM]: the same through a real ssh forward with a `permitopen` key, `MODEL=fake ID_BOUND=50000 PORT=18746`: exit 0, 6/6, `forward: up to the end`, the watermark through the one port. And an overloaded one on purpose (3 lanes at 0.5 a second, nice 19): REFUSED, and the report says why (`cut by the run's end 26 queued and 3 served`).

## 5. The dry runs' fake: one thing differs from your brief

**`warden_serve`'s fake has no time a token in the id form.** It writes the whole answer at once, `--first-ms` after the request; `--token-ms` is only its text form's [read warden_serve.py:410-414]. So "220 ms a token" cannot be had from it, and a dry run against it never shows the certifier a stream. I wrote `certside/paced.py`: the same fake, its id-form answer one id event every `TOKEN_MS` from `FIRST_MS` after the request, then the trailer and the close. Both dry runs use it (`FIRST_MS=2000 TOKEN_MS=220`). `TOKEN_MS=0` gives `warden_serve`'s own. The second dry run also sets `MODEL=fake`, so it goes the connect-reach way and passes `--model`; the first goes the `/v1/models` way. The traffic is the same either way (the fake's model is `fake`).

The fake's responses are shorter than a model's are likely to be: mean 32.7 ids a request at the first try's traffic against a mean `max_tokens` of 51.9 (`max_tokens` is log-uniform in 1..256) [run, my VM, from `id_traffic` seed 0 and `fake_tokens`].

## 6. How long a try takes, and what to cut

What I can say before the dry runs end:

- **The requests and the tail are one number: 9 windows = 1,350 s (22.5 min), both tries.** `rehearse` runs ⌈(last arrival + TAIL) / 150 s⌉ whole windows [read warden_serve.py:1160]; seed 0's last arrival is at 1,138.9 s (first try) and 1,184.6 s (second) [run, my VM, `id_traffic`].
- Around it [run, my VM, idle, 4 cpus]: venv 1 s warm; reach under 1 s by `/v1/models`, 6.5 s by connect; keys and the 5 s start ≈ 7 s; the job's second check 1 s at 6 requests (at 240 and at 2,000: the dry runs will say); the report under 1 s. So the job is about 1,350 s + 20 s + the check, **≈ 23 min, under 28**, if the check stays small.
- Not in that: the wait for memory (research's, before the job starts, at most `--mem-wait 300` s) and custody's upload after the job ends. `out/` is small: 0.66 MB at 6 requests; I expect about 25 MB at the first try and under 200 MB at the second [argued: ≈ 940 bytes a cell kept, 61 + its ids cells a connection].
- **If you need to cut, the first try: `TAIL=60`.** It drops one window (1,200 s, −150 s) with all 240 requests, and nothing is cut even if every response runs to its `max_tokens`: the last END is then at 1,167 s [argued: my simulation of the admission rule at seed 0, with a lane held D + ids + 1 buckets, which matches the smoke's log]. The margin is 33 s. `REQUESTS=224` with `TAIL=120` is the same 8 windows with a margin of 96 s.
- **The second try needs its 9 windows at 2,000 requests** (the last END up to 1,253 s). `REQUESTS=1800` is 8 windows (last END ≤ 1,130 s).
- `RECHECK=0` saves the second check, whatever that is at 64 lanes.
- I have cut nothing.

**Launch early and wait for a go?** The job has no such mode. What sits inside the 30 minutes besides the run itself is seconds, except the wait for memory. A reach check just before (`--reach`) proves the line and the front; it does not make the venv (the job stops before it), but the venv is a `uv venv` and one cached numpy. So a warm second launch buys nothing; what could eat minutes is cpu-11's memory ledger. If you want that outside the 30 minutes, I can add a wait: the job, launched early, holds its reservation with the forward up and starts the run when the front first listens (it retries the reach while the answer is "nothing listens yet", up to `WAIT_S`), or at a UTC time (`GO_AT`). About ten lines, not written: tell me which go you want, since "the front listens" may come before vllm's READY.

## 7. What the simulation says about the schedule [argued]

At seed 0, with every response at its `max_tokens`: first try, 60 of 240 wait for a lane, the longest 113 buckets (28 s), the queue's peak 6 of 64, none refused, none cut. Second try: nobody waits. A refused connection (a full queue) would leave its whole window uncertified and the check REFUSED [read warden.py:24-30, 440-442, 528; run, my VM, with `QUEUE=1`]; neither try comes near it.

## 8. What I am least sure of

1. **The front and the watermark, and the close after the trailer** (section 1, items 1 and 2). I have only the fake to test against. If the first try fails, look first at `failed by reason` in the report: all `late` at the END's cell is item 2; all `late` at cell 0 with the tick silent is item 1 or a D too short.
2. **A tick past its 25 ms slack at nice 19 in a pool.** It faults that window `monitor-unavailable` for every link [read warden.py:79, 469]. The dry runs will show `over_slack`; I will report it as seen and not tune `--slack`. The real run is not in a pool (launch.sh uses `--on vy-nebius-cpu-11`, as Sunday's).
3. **The check's time and memory at 64 lanes and 2,000 requests.** Unknown until the second dry run ends. `MEM_GB=4` may be under what it takes; `--mem-gb` caps nothing but puts a run over its reservation first for the OOM killer [read tools/research/research/mem.py, its docstring, origin/main].
4. **The guard reads text files for the two patterns and binary files only for the peer's host as given.** An address inside a binary record under another spelling would pass. And it only reports: custody has `out/` either way.
5. **What `research run --on MACHINE` itself prints of `--env PEER=…`** on your terminal and in its launch record: launch.sh prints none, the job keeps none, research's own lines I did not read.
6. **The report's "wait for a lane" is in whole buckets and from the certifier's log**, where I saw `admitted − arrived` of 0 for an unqueued connection; I did not look for a case of −1.

Where I did not look: VLLM-324 beyond Part A; the front's code; `tick` on node A; `research run`'s custody of the launch record; `ingress_scan`.

Not launched: anything on a named node. Not touched: Slack, Linear, any PR.

