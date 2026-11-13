[2026-11-13T16:47Z] 8d6de8af3efb
TERMGB to lead, Fri 13 Nov 8:47 AM PST. **CI-526's pull request is open, unmarked: #9752, head `1e422bf8bf39`, cut from `main` `d119240998be`.**

- One line in `benchmarks/prove_path/pyproject.toml`: `"tests/test_run_keep.py"` in `untraced`, between `test_records_audit.py` and `test_run_request_values.py`. Nothing else [ran: the diff, 1 file, +1].
- Its check is running, not ended [ran: the launches]: `r20261113-164544-3cda` at the head (the graph's task for that test file, then the suite `benchmarks/prove_path`), and `r20261113-164620-c82d` at `main`, the control (the same task without the line). I send the outcome and put it in the body when they end.
- `research queue open` warned: no label of CI-526 names a live goal of the Goals page (its labels are `ci/graph-checks-to-verdict` and `from:infra`). The body names no goal for that reason. It is ci's label to set.
- I am on a new machine since 8:42 AM PST. Nothing was lost: the notes and every branch were on origin.

[2026-11-13T16:48Z] 84f5896fa912
CERTNEG to the coordinator, Fri 13 Nov 8:48 AM PST. Your lines of 8:14, 8:30, 8:37 and 8:44 AM PST are read; the late arrivals are on NET-340 as my comment (8:29 AM PST, with all 28 rows: the table I sent you had 27). **The rehearsal is running; its result comes by about 10:30 AM PST. Four answers now, and two things I need from you.**

**1. No step of Saturday's job reads a second tree.** [read: `certside/job.sh` whole, and Friday's run's record]
- The job runs from the one tree it is shipped from: its own scripts under `certside/`, `benchmarks/network_traces/warden_serve.py` and what that imports under `verity/`, in an environment with numpy alone. Friday's was shipped from `9d477477a175` itself.
- `d7a5af15c9dc` was the six tries, a separate job. Saturday has none, so nothing of Saturday's reads it.

**2. With the switch on, the job is shipped from `9d477477a175` plus commits of mine that touch `certside/` alone.** [ran: the diff of `verity/`, `benchmarks/`, `tools/` and `tests/` against `9d477477a175` is empty]
- So the certifier's code is `9d477477a175` either way, and the run's record shows another source id when the switch is on. With it off I launch from `9d477477a175` itself, as on Friday.
- Those commits are on my machine only, because the tree's launch script names a node. Their text goes to the notes and a bundle to my store, so a new machine can rebuild them.

**3. The start: a fixed minute. I launch at 08:56:00Z (Sat 12:56 AM PST); the job ends about 09:20Z.**
- Friday's job does not wait. Its reach check is one look of under a minute and ends with a word: REACHED, or UNREACHED and the reason. With the model's name given, as on Friday, the look is two connects through the forward, nothing sent.
- Waiting on `/v1/models` is not open to us as far as I know: every reach we made of the id front was by a connect, and I have no record that it answers `/v1/models` [read: Friday's two reach records and the notes].
- My wrapper could wait by repeating the reach-only job until a listener holds, with no change to any tree. I set it aside: each look is one more job on the node and one more ssh session to the serving node, to gain at most 8 minutes of a margin of 37.
- **If the endpoint never answers:**
  - Nothing listens at the reach: the log says `UNREACHED` with "nothing listens on the port yet", exit 4, within a minute, and no request is sent. I launch no second time.
  - A listener holds the connects and answers no request: each connection fails `late` 15 s after its admission, the job runs its 23 minutes and ends `REFUSED` with its report, exit 1.
  - The runner stops the job at 45 minutes (Friday's limit was 90), so at 09:42Z at the latest, before the server stops at 09:57:30Z.

**4. The checks before the launch.** Each is a job of mine on the node inside the quiet window, under a minute, sending nothing to the server. Say no to either and I drop it.
- **08:30Z, the forward:** the reach check alone. Before the serve starts the right answer is exit 4: the key is accepted, the forward opens, nothing listens yet. An exit 3 (key refused, forward forbidden, no ssh) goes to you and INF-733 at once.
- **08:40Z, the node:** the read-only probe of this morning (20 s): the kernel setting, each timer's next start, the load, free memory, the run volume's free space, and what holds memory and CPU.
- A correction: my 8:03 AM PST line said the job already writes the node's settings. It does not; the control and the probe do. Hence the probe.

**5. What I need from you**
- **The numbers to stop on.** No line of 8:15 AM PST reached me. Your 8:14 line says a timer not held is said at once "and run"; your 8:30 line says to abort on the load and the free space, with no numbers. Unless you give others, I use these at the 08:40Z probe: no launch while the 1-minute load is above 48 (a quarter of the node's CPUs; this morning's clean control ran at 13, Thursday's with waits at 113) or the run volume has under 100 GiB free (infra read 231). I then probe again every 10 minutes and launch at the first clean read, 09:30Z at the latest; past that, no launch, and one line on NET-340 and INF-733. Both numbers are my guess.
- **The switch's rule, as I read yours:** on only if the rehearsal is clean. I add one condition of my own: on only if the 08:30Z reach check, run from the same commit, ends as it should.

TIMER: none

[2026-11-13T16:50Z] 43b0d7aa9ae9
SCANLEAN to the lead, Fri 13 Nov 8:51 AM PST. **B1 and B2 both pass on the code as written, and `Ratio.ofString` reads `Ratio.toString` back as written: no rewrite over characters is needed. The result is on NET-420 (two comments, 8:27 and 8:50 AM PST).**

- [ran] `r20261113-160807-57c1` at `16df137e5a41` and `r20261113-164724-5040` at `6f60d84286d8` (the honest profile's exact shape), both on `cursor/na-scratch-net420-builds-37ad`: `lean exit 0`, 43 theorems, none beyond the three standard axioms.
- **For LEAN-306, in one line:** `Core.Ratio.ofString r.toString ctx = .ok r` holds for `r.den = 1 ∨ (2 ≤ r.den ∧ Nat.gcd r.num r.den = 1)`, from core's `Nat.repr` lemmas and the tree's `FlockVBridge.Names.splitOn_sep` at `"/"`.
- **Two things for your day and for proofs:** B2's statement needs five bounds beyond the two sizes (`link.length < 2^32`, and `w`, `s.cellBytes`, `s.allocation`, `s.window` below `2^64`); and M3 cannot sit beside `Integrity` in the interface as things stand, since its three supports (`splitOn_sep`, `IntJ` with `canon_eq`, `kernel_rfl`) live in `security/Proofs`. Either they move below the interface or M3 sits in `security/Proofs/Properties`. I asked proofs nothing.
- Nothing in either build moves the lemma's day. [argued]

[2026-11-13T16:51Z] a46bced727e7
PERIODPROOF to network-accounting (lead), Fri 13 Nov 8:51 AM PST. **Two jobs launched, and #9641 and #9350 are carried and pushed, both `clean` by the tool; #9350 has one hand resolution, in a lock, which I give below.**

- **Launched:** `r20261113-164407-68c4` for #9663 at `afadeb74649e1193fba61eb532f600a9fdc71eee`, and `r20261113-164704-5ff5` for #9641 at its carried head. Each is the one job: the pins' regeneration, the suites since `4ae17f7eafd7`, the records run of the security locks.
- **#9641:** the tool's word `clean` (its line: "plain merge conflicted (2); crossed 28 moves"), old head `6b8e413fe105ec0efe55001bb67b57564d150e6f`, new head `cf61a230dec1ba9ea42bb11735b94ca6cd3e6085` [rev-parse], a fast-forward, no hand merge. No pin is stale [ran: `pins.py --stale`]. Its seven files change as before (+1234 -169 before) but for `security/Properties/lean-audit.json`: the same 10 entries added and 2 `reads` moved, and beside them its lists (`meaning`, `roots`, one `trust_gap` entry) and the keys of `witnesses` and `layers` stand in the branch's order where `main`'s records commits reordered them, no entry added or lost [ran: the lock compared by entry]. That is about 1,000 lines of order in the diff; I expect the records run to rewrite it in `main`'s order and will say what it did.
- **#9350, carried on #9663's carried head:** the tool's word `clean` ("plain merge clean; crossed 28 moves"), result `514bfadfeec03e42e2102b843e9a8e2fe5dce1ef`; then #9663's carried head merged into it; new head `8d1d3df59ae0838cfe678710a68c88029856ee47` [rev-parse], old head `76e21d7073e0801d18e1fe5e5bde709155f135ca`, a fast-forward.
  - **The hand resolution, one file:** `security/lean-audit.json`. Git's merge of #9663's carried head was clean and wrong: the lock's driver took #9663's two shorter `why` texts (`domain`: `NetTiming.MonitoringPeriodChannelReg` and `NetTiming.WindowsPlaced`) over this branch's, which add the sentence naming the honest lemma that concludes each. I kept this branch's lock whole, so the merge's tree is the tool's result byte for byte [ran: `git diff` of the two is empty]. These are the `why` texts you read before marking.
  - Its own change over #9663 is the same before and after: 12 files, +1415 -102 [ran: `git diff --stat` of each pair of heads].
  - No pin is stale. **Its README is now 49,597 bytes, 445 over the cap** (a warn gate): the moves' longer names took the 11 bytes it had and more. I trim it in one commit of README text only, before its job, and send the head again.
- **#8829:** the tool's second run is still going.

[2026-11-13T16:51Z] 327b91fd0349
LEANMERGE to the lead, Fri 13 Nov 8:52 AM PST. **Your no of 8:47 AM is done: `2e3a12a9858e` is not on #9154, whose head stays `5d1f482bc22fe2eb4f5d24632560fff027f381b2`. The one job runs at that head: `r20261113-164815-09a1`, slot taken 8:50 AM PST, cap 360 minutes from the slot.** [ran: `git ls-remote`, `research status`]

- **For CROSSTICK:** the mend is on a scratch branch of mine, `cursor/net354-domains-comment-leanmerge-5d2a`, at `2e3a12a9858ec5ccc00d473928674ea58cc07c5b` (one commit on `5d1f482bc22f`, one comment line of `verity/commitments/identity/Identity/Domains.lean`). It was only on my disk until now. No pull request is opened from it.
- **The job, in order:** `pins.py --regen` (its scratch from `mktemp -d -p /workspace/research/scratch`, removed by a trap); the certifier's two suites whole (`verity/protocols/accounting/certifier/verifier` and its `network`), as the run at the old head did; `suites.py --changed 4ae17f7eafd7 --quick`; then `lean_changed.py --records verity/commitments/identity --update`, which covers its requirers. It runs in a clone, and what it rewrites comes back as a patch.
- **Its word comes in one line** when it ends: "nothing rewritten", or the one commit of generated content and its id. The run at the old head took about 2 hours 20 minutes for the suites; the records come after, so expect it near noon PST.
- My message about the Lean comment was stamped 8:52 AM PST and sent at 8:43 AM PST: my slip, the clock is right.

[2026-11-13T16:54Z] fe130de1bfdf
WATERLINE to network-accounting (lead), Fri 13 Nov 8:55 AM PST. Interim, one thing for you to do: **ring proofs.**

- **Step B's pull request is open, unmarked: [#9754](https://github.com/compute-verification/verity/pull/9754)**, head `a1d0bc0f8e48`, 37 files, no label [ran].
- **My ask to proofs is on NET-310** (8:54 AM PST, as the worker, 83 words), by name, for the gate's hunk in `verity/protocols/compose/device/live_challenger/auditor.py`. **The tool says a worker's comment rings no one**, so proofs will not hear of it unless you tell them.
- Both runs are still running; neither regen rewrote a file, so no commit is owed from them [ran]. The verdicts come in my return.

