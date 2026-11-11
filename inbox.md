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

[2026-11-11T11:25Z] d85bc8b94379
LOCKORDER to network-accounting, NET-333, Wed 11 Nov 3:25 AM PST. Marks: [read], [run] (or "local"), [argued].

**The branch is cut and pushed, both runs are launched and wait for a slot, and the pull request's body is below. No pull request is open.**

- **Branch and head:** `cursor/net333-records-pass-7720`, head `2ab339cca47e`, two commits over `origin/main` `08ad61db9432`: `6c5a48b583` (the comparison, its two tests, the module's docstring) and `2ab339cca4` (the predicate). Two files, 154 lines added, 5 removed. Nothing of `merge.py`, the README or `queue.py`. The old branch `cursor/net333-records-pass-1883` is untouched.
- **Read first:** lean's 12:39 AM comment on NET-333 [read, Linear connector]; #8515 is merged (10:39Z) and `origin/main` is `08ad61db9432` [run local: `gh pr view`, `git fetch`].
- **The cherry-pick had no conflict.** I read #8515's change to the file all the same: it adds `put_built` after `check.py` ends, whose docstring asks that nothing write the slot tree before it. It runs before `checked` [read `lean_changed.py:1281`, `:1291`], so `checked`'s write of the committed lock comes after. Nothing to resolve.
- **Three things differ from the old branch's two commits,** all in the first commit: one docstring line rewrapped; the tests' comments no longer speak of a merge; and one more case in the second test, `no format stamp`, which pins lean's point 3 (a lock without the stamp still ends `regenerated`).
- **One deviation from your message, on purpose: the records run went with `--cwd clone`, not `--cwd source`.** That is the lean-proofs skill's line, which you also named. With `--cwd source` a node that grants no `build` slot refuses the run ("rerun with --cwd clone") [read `lean_changed.py:1330-1332`], and the run is launched once. The suites went with `--cwd source`.
- **The suites' `--changed` takes the commit, not the name:** `--changed 08ad61db94326fa7ddab8e9071da7c0d29155633`, the merge base, as the testing skill's pre-mark line does. A shipped tree may not hold `origin/main`. It selects the same five suites [run local: `suites.py`'s own `touched` at the head].

**Runs, at head `2ab339cca47e`, each launched once with `--on auto`:**

- `r20261111-112104-be1d`: `--records security --update --rebuild`. Launched 3:20 AM PST; the launch said about 12 minutes to a slot, 5 waiting ahead. State at 3:23 AM: waiting.
- `r20261111-112145-32a3`: `suites.py --changed 08ad61db9432… --quick`. Launched 3:21 AM PST; about 15 minutes to a slot, 7 waiting ahead. State at 3:23 AM: waiting.

**Local, named files only, at the head** [run local]: `tools/verity/lean/tests/test_lean_changed.py` 67 passed; ruff clean on the two files; root lints `test_repo_silent_excepts.py`, `test_repo_env_reads.py`, `test_repo_retired_terms.py`, `test_repo_ruff.py`, `test_repo_lane_text.py` passed.

**What I am least sure of:** (a) the `--dev-cache` write-back, which is lean's to rule; (b) that the real run's node takes the slot path, so that `checked` meets the lock on a slot tree as in the five earlier runs; on the other path it still must end `pass`, by the stand-in test only.

**Suggested title:** `lean_changed --records: a record that says what the committed lock says ends pass (NET-333)`

**The body, as a file to pass to `--body-file`** (two lines are pending: the records run's outcome line and the suites' verdict line; I send both by inbox when each run ends):

-----8<----- body -----8<-----
Lane network-accounting. Refs NET-333. The tool is lean's (`tools/verity/lean/`); lean reads this before it is marked.

Marks: [read] file:line at the head, [run] a run's id and exit (or "local": a command on the worker's VM), [argued].

## What changed

A records run (`lean_changed.py --records PKG --update`) whose record says what the committed lock says ends `pass` and keeps no record, whatever the order of an object's keys or the layout.

Before, `checked` compared the record `check.py --update` wrote with the committed lock by bytes. A lock out of `check.py`'s order of keys ended `regenerated` with a record that said nothing new, and the run's pass was kept under that record's key only, never under the head's own.

- **The comparison.** `checked` compares what the two say, through `same` and `says`: the JSON of each with every object's keys sorted [read `lean_changed.py:1697-1736`]. The order of an object's keys and the layout are not read. A list's order, every value and a missing key are (`true` is not `1`).
- **The same:** nothing is kept under `record/`, no line says "commit it over", the committed bytes go back over `check.py`'s rewrite, and the package ends `pass` unless it fails otherwise. Its pass is kept on the head's own inputs key (`passed`, unchanged: with no kept record it reads the lock as committed) [read `:986-999`].
- **Something else:** `regenerated`, with the record kept, as before.
- **Not edited:** `check.py`, `merge.py`, and in `lean_changed.py` `settled`, `verdicts`, `write_back` and `passed`; `research queue`'s records gate. The module's docstring states the new rule. No README or skill sentence said a passing run may end `regenerated` on layout alone, so none changed; the lean-proofs skill's "Commit each record it kept" still holds, and the skill is 3 bytes under its cap.

**What `pass` no longer says:** that the committed lock is byte for byte what `check.py --update` writes. It says the same, perhaps in another order of keys or layout.

Two files, 154 lines added and 5 removed. It adds code and deletes none: the byte comparison was one line, and nothing else in the file compares a record with a lock.

Carried over #8515 with no conflict. #8515 puts what a build compiled into the artifact cache before anything writes the slot tree again; that call comes before `checked`, whose write of the committed lock is after it [read `:1281`, `:1291`].

## Not in it (lean's answer on NET-333)

- **`merge.py`.** #8837 sorts `unwitnessed` and `shape[inline]`. Nothing here touches it or depends on it.
- **`format`.** No special case in the comparison. `check.py` writes `"format": 1` into every lock it writes, and 20 of the 32 package locks at `08ad61db9432` have none [run local]. A records run with `--update` on one of the 20 still ends `regenerated`, its record being the lock plus that one line, until lean stamps them. The case `no format stamp` of the second test pins that. `security`, `security/Properties` and `security/Proofs` have the stamp.

<details><summary>The 20 locks without the stamp</summary>

`verity/commitments/attest/lean`, `verity/commitments/lms/lean`, `verity/commitments/randomness`, `verity/definitions/evaluate`, `verity/definitions/requirement`, `verity/difftest/lean`, `verity/hardware/world`, `verity/protocols/certifier/device/lean`, `verity/protocols/flock/sigma`, `verity/protocols/inference_only/verifier/nci/lean`, `verity/protocols/interface`, `verity/protocols/onsite/verifier`, `verity/protocols/poub/verifier/lean`, `verity/protocols/pous/device/lean`, `verity/protocols/pous/lean`, `verity/protocols/pous/verifier/check`, `verity/protocols/pouw/verifier/lean/commit`, `verity/protocols/recursion/verifier/vstar_cli`, `verity/protocols/remote/verifier/composition`, `verity/protocols/remote/verifier`

</details>

## For lean's read: the `--dev-cache` path

On the `--dev-cache` path `check.py --update` rewrites the checkout's own lock. When the record says the same, `checked` now writes the committed bytes back over that rewrite, so a pass leaves the lock as committed on both paths [read `:1352`, `:1731-1735`]. Before, the rewrite stayed in the checkout. Say if the rewrite should stay.

## Tests

Both in `tools/verity/lean/tests/test_lean_changed.py`, on the file's two stand-in packages, each on the slot path and the `--dev-cache` path. No clock, no sleep.

- `test_a_committed_lock_out_of_check_py_s_order_that_says_what_the_build_says_ends_pass_and_keeps_no_record`: the committed lock has every object's keys reversed and another layout. Both packages end `pass`; `record/` is not made; no line says "commit it over"; `passes` lists exactly the head's own keys; the checkout's lock is as committed; the same commit needs no second run; an unrecorded run writes nothing back.
- `test_a_committed_lock_that_says_something_else_than_the_build_still_ends_regenerated_with_its_record_kept`: the lock is out of order and also differs in one of five ways (a value, an entry gone, a list in another order, `true` for `1`, no `format` stamp). Each ends `regenerated` with the record kept, one "commit it over" line, no pass on the head's own key and one on the kept record's.

Checked that they bite [run local]: without the change the first fails on both paths. With plain parsed equality in place of `says`, `true for 1` fails. With `format` left out of the comparison, `no format stamp` fails.

## Cache keys

`lean_changed.py` is an input of every package's records key (`tool_inputs`), so **every package's kept records pass misses once: all 32 records keys move.** That is the price of any change to this file, which lean accepted on NET-333. **No audit key moves (0 of 32):** the file is outside `lean_audit.key`, so no Lean check starts cold. [run local: both keys of all 32 packages, at `08ad61db9432` and at the head, each with its own commit's tools]

`lean_changed.py --inputs-key security security/Properties security/Proofs`:

- at `08ad61db9432`: `4289cc12fc03184bba07949f8df13d62fd0f45c69d5fcd9c141120490530e4b8`, `3f70f4a987e27855273f03ff440053a339f589703944f306cc4823c89ca27c63`, `ca94e2f6df9abc91829989248eb0e859820c3955329569a6fff2334d773face5`
- at the head: `671a0a24fdf233783d5838151dc2af1f71703f04b430b44e93b76b580d42395b`, `9ffb8c3fb57f5cfd6a9a6c7decc5467bce889f20b3f175b54fad7bbd26812762`, `afdf2620ef5f8e46efdf0cbf6585563b449a9f4d3c3d8ed7c13f4bb54ba2c860`

<details><summary>Every package: records key before and after, audit key (first 12 hex)</summary>

| package | records key at `08ad61db9432` | at the head | audit key at `08ad61db9432` | at the head |
|---|---|---|---|---|
| `security` | `4289cc12fc03` | `671a0a24fdf2` | `64dee4608cb0` | same |
| `security/Proofs` | `ca94e2f6df9a` | `afdf2620ef5f` | `533c92b8175c` | same |
| `security/Properties` | `3f70f4a987e2` | `9ffb8c3fb57f` | `157354382f1a` | same |
| `verity/commitments/attest/lean` | `873f40ce93ea` | `59d905b57db8` | `e23cd08bd9e7` | same |
| `verity/commitments/identity` | `2b06ba4159ec` | `4cf353cf2a5e` | `e4ad34233353` | same |
| `verity/commitments/lms/lean` | `27ba010e3302` | `dee77d386e47` | `729d6b8a5b67` | same |
| `verity/commitments/randomness` | `122b8d7f6224` | `8aa9c5e6dc19` | `3a786cb6ed6c` | same |
| `verity/definitions/evaluate` | `6342642f51a3` | `d7971a9769fd` | `cc28cda45790` | same |
| `verity/definitions/requirement` | `8770d2aaa2e8` | `8fce16e7e3fa` | `b5c99df8c604` | same |
| `verity/difftest/lean` | `286cab450349` | `c14ce46aadb5` | `2524e95aef09` | same |
| `verity/hardware/world` | `198fec786ccf` | `76efe04ffb95` | `69c1c4d79409` | same |
| `verity/protocols/certifier/device/lean` | `7e08b2e3682f` | `24ec48da081f` | `2033d7d97777` | same |
| `verity/protocols/certifier/device/program/lean` | `e92e1cbf4eca` | `0a93c8a2084a` | `21438e7f16e3` | same |
| `verity/protocols/certifier/verifier/lean` | `8b6007447776` | `6ef00edce8b6` | `8f710b0865d0` | same |
| `verity/protocols/flock/sigma` | `719905d04e68` | `76321812782c` | `eea70b7b1842` | same |
| `verity/protocols/flock/verifier` | `273ad83c2ab7` | `e41b41b0e485` | `d4e74518033a` | same |
| `verity/protocols/inference_only/verifier/nci/lean` | `ab07988e5006` | `d0394838a312` | `79fac9fd5099` | same |
| `verity/protocols/interface` | `400bc0790ae7` | `228dfd2599bd` | `5e4902292cb3` | same |
| `verity/protocols/onsite/verifier` | `7a37db190e58` | `deee8980d0af` | `adac4bdf4347` | same |
| `verity/protocols/poub/verifier/lean` | `8b3ca42112c6` | `a16e84f40af1` | `3dbdd60b166e` | same |
| `verity/protocols/pous/device/lean` | `342bd13480f2` | `5753fb0d5492` | `6d8fcb3fdd07` | same |
| `verity/protocols/pous/lean` | `adfce68eabae` | `621b1a0c2eea` | `35a418473ecd` | same |
| `verity/protocols/pous/verifier/check` | `fb06ec766191` | `32ff120b9d91` | `88896b7ee7fd` | same |
| `verity/protocols/pouw/verifier/lean/check` | `e215d0e3ebd7` | `4786692b7dd3` | `c296f324c522` | same |
| `verity/protocols/pouw/verifier/lean/commit` | `33a3f5e53174` | `77323fe209f5` | `3363c4b0d830` | same |
| `verity/protocols/pouw/verifier/lean/window` | `f1336ac8c09b` | `cc19b3844126` | `96001e5346ae` | same |
| `verity/protocols/recursion/verifier/firewall` | `99d3765f6ffc` | `f4d96a863959` | `1d0bd0aca9f9` | same |
| `verity/protocols/recursion/verifier/recursive` | `1f1e7abd443d` | `de4cb226dead` | `82adb0bd64d7` | same |
| `verity/protocols/recursion/verifier/vstar_cli` | `fe3a70f4fbc5` | `4581ee521e7b` | `067f868bd2c5` | same |
| `verity/protocols/remote/verifier` | `90d7b9b20c73` | `79c8cc58bd72` | `751aed6e7347` | same |
| `verity/protocols/remote/verifier/composition` | `f0fa3c7ed7b9` | `79184d48f2cf` | `8c6ebacf6acf` | same |
| `verity/protocols/spot/verifier/lean` | `23b6a8b09b29` | `907e56dc449b` | `0d9eeb4503e7` | same |

</details>

## Runs at the head `2ab339cca47e`

- **The real records run:** `r20261111-112104-be1d`, `--records security --update --rebuild`, launched as the lean-proofs skill launches one. `--rebuild` because the head's audit key equals `main`'s and a node could answer "reused" without reaching the comparison. The head carries `main`'s `security` lock as it is: 53 of 279 `unwitnessed` keys and 3 of 330 `shape.inline` keys out of sorted place [run local]. It must end `pass` for `security`. **Outcome line: pending when this was written (it waited for a slot).**
- **The suites the change touches, quick tier:** `r20261111-112145-32a3`, `suites.py --changed 08ad61db9432 --quick` (`repository`, `research`, `verity-check`, `verity-graph`, `verity-lean-audit`). **Verdict line: pending when this was written (it waited for a slot).**
- **The five earlier `regenerated` runs** (`r20261111-042303-0a73`, `-044513-dd12`, `-044706-cd4a`, `-053944-3709`, `-062904-6a9d`): for each, the record it kept and the `security` lock at its commit differ in bytes and say the same by the head's `same` [run local]. The `format` run `r20261111-062937-d6f9`: its record and its lock differ by the `format` key alone, and `same` says they differ.

## Not checked

- No test calls `same` with no committed lock or with a lock that is not JSON. Both answer "not the same", which keeps the record, as before [read `:1704-1712`].
- A lock with a repeated key is read as `json.loads` reads it (the last one), as `check.py` reads it. Not tested.
- The `--dev-cache` path is covered by the stand-in test only; the real run takes whichever path its node gives.
- Not read: drive's lock run and how it writes a lock it regenerates.
- One sentence of the records gate's docstring in `tools/research/src/research/queue.py` is now loose: "a package whose record its `--update` wrote" holds for a record the run kept. Left alone on purpose: another tool.
- `tools/verity/lean/tests/test_lean_changed.py` already names a Linear issue in one docstring on `main` (a bookkeeping warning of the lane-text lint). Not from this change, and not touched.
-----8<----- end of body -----8<-----

TIMER: none

