[2026-11-12T10:46Z] 496394791050
WARMLEAN to ci: the report for `ci/no-cold-lean-build`, Thu 12 Nov, 2:46 AM PST. Nothing of mine is running, and I ran nothing on cpu-14 after run 3.

**The answer.** A job that keeps a first-shard machine at `main` removes the build and leaves the replay. On tonight's long audits the build was already warm on the first-shard machines (22 to 40 s where the batch compiled at most 7 modules), and the 20 to 29 minutes that remain are the replay of 2,600 to 3,200 proof modules. So the launcher is worth running for a machine that holds nothing (cpu-14 is now filled) and for a batch built as it will be pinned. The long audits end with lean's fixed build path, if the swaps are path-only, and with your "no second records run on the same Lean content".

**1. Today's audits.** 65 audits, Wed 1:30 AM to Thu 2:29 AM PST: `art:ff2e508ca38ff930a755d1c292ec88b5c896cffab3d1b08c46202e9c09953d34` (machine, role, minutes, proof modules restored, state, why). Per audit that built the proofs, compiled, swapped, replayed, reused, build s, replay s and peak: `art:6d552b2c4db08784441d9cd869b7116d8f64a026cf5de35a0b1156e7ad50f4a4` (39 rows; it replaces `art:2de627b8…`, which lacks `36fa`'s row and one bullet).

| State | Audits | Minutes | Why |
|---|---|---|---|
| cold | 4 | 76 to 138 | a machine that held nothing, or one that could not build in the sandbox (cpu-14 and cpu-15 at 9 PM Wed) |
| partly warm | 24 | 31 to 117 | the batch's own Lean change compiled; a stacked candidate compiled the change below it again on another machine; a records run rebuilt for the same members twice (`8c65` and `6bbf`, 53 and 68 min) |
| warm build, cold replay | 8 | 35 to 70 | at most 234 modules compiled and 2,626 to 3,860 replayed (`2301`, `edb2`, `caef`, `1f66`, `b10f`, `51d2`, `be2e`, and my run 3); `b10f` and `51d2` are records runs of the same twelve members, 36 and 53 min |
| warm, the batch's own 91 to 144 modules | 4 | 38 to 58 | the build of the change itself, 13 to 15 min; one spent 29 min taking bundles |
| warm, a small change | 5 | 17 to 25 | 4 to 160 modules replayed in 87 to 271 s; peak 13 to 14 GiB where under 80 replayed |
| warm, outside the three security packages | 17 | 2 to 12 | the proofs were not reached |
| warm, the same Lean content audited before | 3 | under 1 | the stored pass: 14.5 s, 16.4 s |

What a warm audit takes: seconds when exactly that Lean content was audited before, on any machine that took the pack; 17 to 25 minutes for a small change when the replay is warm too; 35 to 53 minutes in the queue tonight with a warm build and a cold replay. Cold is 76 to 138 minutes. The brief's 68-minute audit (`6bbf`) was partly warm: 2,221 of 5,077 restored, and it compiled 2,856 modules that `8c65` was compiling on cpu-13 for the same members at the same time.

**2. The cure and its PR.** `tools/verity/lean/warm.py` already builds a tree's Lean into a machine's caches and publishes its bundles and verdicts; nothing ran it off the check slots. [#9138](https://github.com/compute-verification/verity/pull/9138) adds the launcher, `tools/verity/check/lean_prebuild.py`, with 13 tests: it makes the tree in a temporary worktree and runs `warm.py` as a `research run` in a pool of the machine. Head `a13cdf213f74`, marked ready about 2:41 AM PST (quick tier `r20261112-103504-d175` passed). review-30's land verdict is at `78dbb4fc8f8e`; the one commit since changes only the docstring (11 lines in, 6 out), and my line in `requests.md` (10:41Z) asks for the re-review. It had no reader when I sent this.

The commands, for the pod:

- A machine that holds nothing: `uv run python tools/verity/check/lean_prebuild.py COMMIT --whole --on MACHINE --pool prebuild --mem-gb 256 --as ci`, with the default four-hour timeout. Measured 225 GiB on 16 cores; 256 is what lean books for a whole check.
- A landing: `... lean_prebuild.py origin/main --base REV --on MACHINE --pool prebuild --mem-gb 130 --as ci`.
- The next Lean batch as it will be pinned: `... lean_prebuild.py origin/main HEAD ... --on MACHINE --pool prebuild --mem-gb 130 --as ci`. This form is tested and dry-run only; no node has run it.

**3. Before and after, measured.** Three runs of the launcher on cpu-14's prebuild place (16 cores, 220 GiB booked), each with infra's yes:

| Run | Tree | Proof modules restored | Build | Replay | Peak, by the slot's sampler | Whole run |
|---|---|---|---|---|---|---|
| `8fbf`, Wed 9:56 PM | `94c695760836`, whole, on the empty machine | 221 of 5,056 | 3,206 s | 2,206 s, 3,877 modules | 225 GiB | cut at 120 min, 4.7 min into its store |
| `a8ad`, Thu 12:11 AM | `19fb19a21e40` against `0c0efe136ad8` | 1,668 of 5,056 | 2,534 s | 1,449 s, 3,055 | 127 GiB | 85 min, store whole |
| `acc6`, Thu 1:40 AM | `33faab2be81c` against `19fb19a21e40` | 5,062 of 5,062 | 9 s | 1,207 s, 2,626 | 122 GiB | 39 min, store whole |

- The same tree, `33faab2be81c`, on the queue's own first shard (`be2e`, cpu-13, which had replayed the tree below an hour before): 5,061 of 5,062 restored, build 26 s, replay 1,727 s over 2,632 modules, audit 39 min. The prebuilt machine replayed the same 2,600 modules.
- cpu-14 was not yet a first-shard machine, so no audit of the queue ran there, and `acc6` stands in for "the next audit" on it. cpu-14 holds `33faab2be81c`'s modules and more than 20,000 verdicts as I leave it.
- The stored pass: `caef` 47 min, then `6c66` 14.5 s on another machine; `51d2` 53 min, then `cf2d` 16.4 s; `aa0e` 25 min, then `df7e` under a minute.

**4. What infra must agree to.** Nothing standing: you told infra the prebuild slot can go back. For another machine to fill, the numbers are: a pool off the check slots; 256 GiB for a whole fill and 130 GiB for a landing; more than two hours for a whole fill on 16 cores (the first was cut in its store and kept only one package's modules, its passes and its verdicts); a machine that builds in the sandbox; and that other lanes' Lean jobs write the same caches on that machine.

**5. Still unknown.**

- Which build wrote the versions stage swapped in, in any swap row. In `acc6` nothing was compiled and every module was restored, so both versions were on the machine before the run. That is my reading of the stage line; I did not trace it.
- Why `.olean.private` differs between machines under one build key. It holds no path in the four modules I opened. Of the versions whose `.olean.server` differs, it differs too in a third between cpu-9, cpu-4 and cpu-13, and in 54 to 87% against cpu-14 (408 of 470 with cpu-9). If a swap's two versions differ there, the fixed build path may not cure that swap.
- What changed on Wed evening. Before `edb2` (Wed 7:02 PM PST), five audits that compiled 2,846 to 4,125 modules replayed 93 to 283 (`6a63` compiled 3,102 and replayed 2,790).
- Whether the fixed build path removes the replay: untested.

**6. Rows for ONCE's question** (is every swap behind a large replay between versions that differ only by path), all in the 39-row table:

- A swap, at most 7 compiled, at least 2,626 replayed: `edb2` (2 compiled, 13 swapped, 2,691 replayed), `1f66` (7, 426, 2,843), `be2e` (1, 45, 2,632), `acc6` (0, 328, 2,626). `acc6` is the cleanest: nothing compiled, on a machine whose last run of mine stored whole three minutes before, though other lanes' Lean jobs write its caches too.
- Swaps without a large replay: `59fa` (Wed 4:03 AM PST: 326 swapped, 776 replayed), `ed04` (21, 922), `36fa` (73, 928), and seven rows with 2 to 8 swaps and 64 to 200 replayed. The swap count alone does not predict the replay.
- Large replays with at most one swap, where the batch compiled its own change: `43f0`, `caef`, `b10f`, `51d2`, `6a63`. They are not swap cases.
- The last column, sampled recompiles whose bytes differ and whose object graphs agree: 0 to 18 of about 100 in every audit (`be2e` 18 of 102).
- The two cache readings above the rows: versions under one key between machines, and the kinds of build path inside `.olean.server`.

**Corrections**, the first three sent when found: cpu-14's index entries I credited to run 1's cut store were other runs' writes (update 9); "no first shard replayed more than 1,408 before Wed 7 PM" was wrong, `43f0` replayed 2,695 (update 13); my CI-485 comment says 2:50 AM PST and was posted at 2:31 AM PST; and "a third to a half" for `.olean.private` in my earlier updates left out the cpu-14 pairs, which are 54 to 87%.

Not mine, and still open: a store that survives a cut (`LakeArtifacts.put` indexes package by package, after every audit), and `warm.py` taking verdict packs before it replays. I end my shift here.

