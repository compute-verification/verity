[2026-11-11T11:23Z] 1edcfc64a390
Report: CERTSIDE (NET-334), third turn: SEED2, the seeds table, run 1735's figures, the variant's dry run

Written Wed 11 Nov, 3:23 AM PST. Every line from `research` went through your six-part mask; no node is named
below except `ON=...` in the hour's two command lines.

## 0. Short

- New head `d6fef0385ece` on `cursor/net334-certside-562d`, pushed, no force: `SEED2`.
- The seed: **3**, the only one of 1 to 8 that qualifies.
- `r20261111-102917-1735` ended with exit 0, both tries ACCEPTED; its figures are in part 2.
- The variant's dry run: **`r20261111-111911-de73`**, launched 3:19 AM PST at `d6fef0385ece`, `running` at 3:21 AM PST.

## 1. SEED2

**The head and its diff** [read, `git diff --numstat 1e81f00ff032..d6fef0385ece`]: 2 files, 16 insertions, 12 deletions:
`certside/job.sh` +11 −8, `certside/launch.sh` +5 −4.

- `SEED2=${SEED2:-$SEED}` [read, job.sh:92]; try 2 takes it where it takes `LANES2`, `RATE2`, `REQUESTS2`
  [read, job.sh:424]; the line that starts try 2 now ends `, --seed N` [read, job.sh:423].
- Each try's seed is in the job's `summary.json`, `try1.seed` and `try2.seed` [read, job.sh:430 and 459].
- `launch.sh` passes `SEED2` as it passes the others [read, launch.sh:59]. Against a stub of `research` it expands to
  `... --env TRIES=2 --env REQUESTS2=2000 --env SEED2=3 ... -- bash certside/job.sh` [run, my VM].
- Nothing else changed.

**The table**, rate 1.67, 64 lanes, queue 64, D = 60 buckets of 0.25 s, tail 120 s, window 150 s. Last arrival and
windows are the generator's own code (`warden_serve.id_traffic`, and rehearse's own expression for the windows,
warden_serve.py:1160) [run, my VM]. The last END, the room, the lanes and the waits are my simulation with every
response at its `max_tokens` [argued]; its rule, a connection ends at the bucket it was admitted + D + its ids, held in
all 2,240 connections of run 1735 [run, `r20261111-102917-1735`, exit 0]. "Room" is from the last END to the end of
the last bucket. "Equal" counts try 2's requests whose prompt, `max_tokens`, temperature and seed are all those of
one of try 1's 240 (seed 0).

| seed | REQUESTS2 | last arrival, s | windows | last END at max_tokens, s | room, s | most lanes held | arrivals that wait | longest wait, buckets | equal to one of try 1's |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2,000 | 1,184.6 | 9 | 1,253.00 | 97.00 | 64 | 0 | 0 | 240 |
| 0 | 1,800 | 1,062.3 | 8 | 1,130.25 | 69.75 | 64 | 0 | 0 | 240 |
| 1 | 2,000 | 1,207.6 | 9 | 1,273.00 | 77.00 | 63 | 0 | 0 | 0 |
| 1 | 1,800 | 1,089.7 | 9 | 1,152.25 | 197.75 | 63 | 0 | 0 | 0 |
| 2 | 2,000 | 1,198.8 | 9 | 1,243.25 | 106.75 | 60 | 0 | 0 | 0 |
| 2 | 1,800 | 1,083.7 | 9 | 1,134.50 | 215.50 | 60 | 0 | 0 | 0 |
| **3** | 2,000 | 1,197.5 | 9 | 1,252.75 | 97.25 | 61 | 0 | 0 | 0 |
| **3** | 1,800 | 1,073.5 | 8 | 1,142.25 | 57.75 | 60 | 0 | 0 | 0 |
| 4 | 2,000 | 1,212.7 | 9 | 1,283.75 | 66.25 | 64 | 0 | 0 | 0 |
| 4 | 1,800 | 1,083.8 | 9 | 1,157.75 | 192.25 | 64 | 0 | 0 | 0 |
| 5 | 2,000 | 1,216.1 | 9 | 1,260.75 | 89.25 | 60 | 0 | 0 | 0 |
| 5 | 1,800 | 1,083.1 | 9 | 1,135.75 | 214.25 | 60 | 0 | 0 | 0 |
| 6 | 2,000 | 1,154.7 | 9 | 1,207.75 | 142.25 | 64 | 6 | 3 | 0 |
| 6 | 1,800 | 1,039.2 | 8 | 1,090.75 | 109.25 | 64 | 6 | 3 | 0 |
| 7 | 2,000 | 1,167.8 | 9 | 1,227.25 | 122.75 | 64 | 8 | 6 | 0 |
| 7 | 1,800 | 1,051.4 | 8 | 1,097.00 | 103.00 | 64 | 8 | 6 | 0 |
| 8 | 2,000 | 1,177.9 | 9 | 1,244.75 | 105.25 | 64 | 15 | 5 | 0 |
| 8 | 1,800 | 1,074.5 | 8 | 1,141.50 | 58.50 | 64 | 15 | 5 | 0 |

No row refuses or cuts a request. Seed 0's rows are as they were: all 240 of try 1's requests come again.

**The pick: `SEED2=3`.** It is the only seed that meets all four conditions. Why each other fails:

- Seed 1: 9 windows at 1,800; 77.00 s of room at 2,000.
- Seed 2: 9 windows at 1,800.
- Seed 4: 9 windows at 1,800; 66.25 s of room at 2,000.
- Seed 5: 9 windows at 1,800; 89.25 s of room at 2,000.
- Seeds 6, 7 and 8: 6, 8 and 15 arrivals wait for a lane, at both counts.

One thing seed 3 costs: at 1,800 its room is 57.75 s, against seed 0's 69.75 s. You set no floor at 1,800.

**Local smokes** [run, my VM, one 150 s window a try, the paced fake with ids below 154,880]:

- `TRIES=2 SEED2=3`, 6 then 20 requests: exit 0, try 1 ACCEPTED 6/6, try 2 ACCEPTED 20/20; the log says
  `try 2 starts, with --lanes 64, --rate 1.67, --requests 20, --seed 3`; the summary's seeds are 0 and 3.
  Try 1's first request: 536 prompt ids starting 99346, 116686, 55125; `max_tokens` 5; seed 556803596.
  Try 2's first request: 43 prompt ids starting 71333, 17094, 48490; `max_tokens` 27; seed 133194446.
  None of try 2's 20 requests equals one of try 1's.
- `SEED2` unset, at `d6fef0385ece` and at `1e81f00ff032`: exit 0 both, both tries ACCEPTED. Each try's
  `warden/traffic.json` is the same file at the two heads, byte for byte (`cmp`), all 6 and all 20 requests; that file
  is the requests themselves and holds no key. Try 2's first 6 requests are try 1's 6.

## 2. Run `r20261111-102917-1735` (two tries, seed 0 both, at `1e81f00ff032`)

All [run, `r20261111-102917-1735`, exit 0], read from its fetched record (`out/summary.json`, each try's
`check.txt`, `warden/warden.json`, `warden/deployment.json`, `summary.json`, and `out/job.log`).

- **Exit code:** 0 (the run's state `done`, class SUCCESS).
- **Verdict lines:**
  - Try 1: `ACCEPTED: 240/240 ended, 0 refused, failed []; 27/27 link-windows certified; cells in their bucket 22492/22492; ... tick lateness {'n': 5400, 'p50': 0.856, 'p99': 1.966, 'max': 2.224, 'over_slack': 0}`
  - Try 2: `ACCEPTED: 2000/2000 ended, 0 refused, failed []; 27/27 link-windows certified; cells in their bucket 183350/183350; ... tick lateness {'n': 5400, 'p50': 0.852, 'p99': 2.179, 'max': 4.742, 'over_slack': 0}`

| | try 1 | try 2 |
|---|---|---|
| connections ended | 240 of 240 | 2,000 of 2,000 |
| refused, failed, cut | 0, 0, 0 | 0, 0, 0 |
| link-windows certified | 27 of 27 | 27 of 27 |
| ticks past the 25 ms slack | 0 of 5,400 (latest 2.2 ms) | 0 of 5,400 (latest 4.7 ms) |
| `watermark` last | [5399, 240] | [5399, 2000] |
| `watermark` lost | [] | [] |
| `watermark` error | none | none |
| arrivals that waited for a lane | 32, longest 71 buckets | 0 |
| most connections open at once | 8 | 57 |
| rehearse, s | 1,359 | 1,367 |
| check, s | 2 | 9 |
| its `out/` | 7,162,461 bytes | 18,595,965 bytes |

- **Steps, seconds from the job's start:** venv ready at 1, reach done at 8, try 1 starts at 8, its rehearse returns at
  1,367, its check at 1,369; try 2 starts at 1,369, its rehearse returns at 2,736, its check at 2,745.
- **`between_s`** 2 (try 1's rehearse returning to try 2's start). **`between_buckets_s`** 11.4 (try 1's last bucket to
  try 2's first).
- **The clock, seconds from the job's start** (10:30:46Z; the log has whole seconds, so each is good to about 1 s):

| event | second |
|---|---|
| try 1's first bucket | 15.8 |
| try 1's last END | 1,182.8 |
| try 1's last bucket ends | 1,365.8 |
| try 2's first bucket | 1,377.2 |
| try 2's last END | 2,612.7 |
| try 2's last bucket ends | 2,727.2 |
| the job's end | 2,745 (11:16:31Z) |

- **The cgroup's peak:** 1,643 MB (try 1 alone 1,313 MB).
- **`out/`:** 25,782,653 bytes in 41 files as fetched. The job's own summary says 25,780,871 bytes in 40 files: it counts
  before it writes itself.
- **The guard's line:** `certside: guard: no address in out/ (41 files); exit 0`.
- The run waited 66 s for a slot before the job started; its whole wall time is 2,811 s.

**This replaces the [argued] clock** of my last report (try 1 about 7 s + 1,364 s, try 2's first bucket about 8 s
later, try 2's last END by about 2,632 s, the job's end at about 46.0 min):

- A job that starts at second 0 has try 2's first bucket at 1,377 s and ends at 2,745 s (45 min 45 s) [run].
- Try 2's last END was at 2,612.7 s with the fake, which stopped short of `max_tokens` in 1,060 of 2,000 responses
  [run]. With every response at its `max_tokens` it is 1,377.2 + 1,253.00 = 2,630.2 s at seed 0, and
  1,377.2 + 1,252.75 = 2,630.0 s at seed 3 [argued from the two].
- So for try 2's last bucket to end by 16:57:30Z the job must start by 16:12:03Z, and for the job itself to end by
  then, by 16:11:45Z [argued]. The real reach is an ssh forward and not a local connect, so the 8 s before try 1 is the
  fake's, not the hour's: I have not measured the real one.

## 3. The variant's dry run

**`r20261111-111911-de73`**, launched 3:19 AM PST from my worktree at `d6fef0385ece` (clean), after 1735 showed
`done`, exit 0:

`uv run -q research run --on auto --cwd source --timeout 60m -- env FAKE=1 TRIES=2 MODEL=fake FAKE_IDS=154880 SEED2=3 bash certside/job.sh`

One run, placed by `auto`, with preflight. State `running` at 3:21 AM PST. If it takes 1735's 2,745 s it ends near
4:07 AM PST (12:07Z), before 13:00Z [argued]. I have not read anything from it yet.

## 4. The hour's command lines

- With `SEED2`, at `d6fef0385ece`; its two-try dry run is `r20261111-111911-de73` (running):
  `ON=vy-nebius-cpu-11 PORT=18746 MODEL=glm-5.3 TRIES=2 REQUESTS2=2000 SEED2=3 bash certside/launch.sh`
- Without, at `1e81f00ff032`; its two-try dry run is `r20261111-102917-1735` (exit 0):
  `ON=vy-nebius-cpu-11 PORT=18746 MODEL=glm-5.3 TRIES=2 REQUESTS2=2000 bash certside/launch.sh`
  The same line at `d6fef0385ece` sends the same requests (the smoke above), but its full-length dry run was at
  `1e81f00ff032`.

## 5. What I did not run, and what I am least sure of

- Nothing on a named node. `launch.sh` only against a stub of `research`, never for real.
- The real front, the real model and the real ssh forward at these schedules. A refused try at the real schedules.
- Any suite or the check.
- The fake's id statistics for seed 3's try 2 (how many ids at or above 2^17): the variant run will have them.
- 1,800 requests in a full-length run, at either seed: both dry runs use 2,000.
- **Least sure:** the table's END, room, lanes and wait columns are a simulation at `max_tokens`. The real model can
  end responses early, which under the certifier's schedule only adds room and frees lanes sooner; but seed
  3's 61 lanes of 64 at 2,000 is a margin of 3 lanes and rests on the simulation alone until the variant run reports
  its peak (the fake stops early in about half its responses, so its peak will sit below 61).
- **For you to decide:** `certside/launch.sh` has carried the two machines' short names as its defaults for `ON` and
  `MACHINE` since my first commit (lines 7, 16, 28, 30). No commit message names a node. I left the file as it was
  because this commit was to change nothing else; say so if the defaults should go, and I drop them in one commit.
- My two earlier notes named nodes; I masked them and they sync with this turn's note.

