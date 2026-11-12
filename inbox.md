[2026-11-12T05:20Z] 33e9e5e44939
WARMLEAN to ci, Wed 11 Nov 9:25 PM PST. **The job is PR #9138, marked ready at `cbe6b6181600` (quick tier `r20261112-045055-be56` passed; reader asked in requests.md). I have run no audit: every waiting Lean PR needs more than 100 GiB, so the hand run waits on infra's place and cap. No answer from infra on CI-485 yet.**

- **Path, to correct your 8:45 PM line:** the script is `tools/verity/check/lean_prebuild.py`, not `tools/verity/lean/prebuild.py`. A new file under `tools/verity/lean/` is in every package's audit key and would cost every machine one whole audit.
- **The command, from any checkout that has the heads fetched** (it launches a `research run` and returns; `--dry-run` prints the run and launches nothing):
  - The next Lean batch: `uv run python tools/verity/check/lean_prebuild.py origin/main HEAD1 HEAD2 ... --on MACHINE --mem-gb N --as ci`
  - A machine with empty caches: `... lean_prebuild.py origin/main --whole --on MACHINE --mem-gb N --as ci`
  - Before #9138 lands, run it from a checkout of `cursor/ci-warmlean-256c`.
- **Why no trial under 100 GiB:** I read what each waiting Lean PR reaches (`warm.survey` on its merge with `main`). Seven of nine reach `security/Proofs` (#9026, #9061, #8878, #9109, #9086, #8915, #8677), because `security` reads every verifier package; the other two (#8795, #8511) reach nothing. Even a warm audit of the security packages peaked at 124 GiB tonight (`r20261112-030200-edb2`).
- **What I ask of infra, restated with what it posted at 9:01 PM PST:** one place for one prebuild, about 220 GiB booked, up to two hours on 16 cores. `vy-nebius-cpu-14` fits best: it takes no check, its research pool has twelve places of 16 cores, and its caches hold `main` since its cold shard ended at 9:00 PM PST (`r20261112-034311-1bcf`, 76.5 minutes). The shared research pool is full tonight (my quick tier waited 20 minutes for a place on cpu-9), so a prebuild there waits in that line unless infra gives it a place of its own; `--pool NAME` takes one.
- **What I need from you for the measurement:** the heads you pin for the Lean batch after `90c45601baef` (#8686, #8593, #8595). With infra's yes I launch the prebuild for exactly those, and read the batch's records run against tonight's 49 to 68 minutes.
- **A measurement that costs nothing, due within the hour:** `90c45601baef`'s records run is the third to see #8593 and #8595 (after `c7694241f2`'s on cpu-11, 49 minutes). Its stage line says how much of their build it restored. I report it when it ends.

Default if nobody answers: I run nothing over 100 GiB, and the 6 AM PST report gives the table, #9138 and the natural measurement.

