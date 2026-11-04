
## 18:34Z, proofs coordinator: move queued runs to cpu-5 and cpu-6 research pools now

cpu-9 is refusing new runs: /workspace fell to 24 GB free, under its 50 GB floor (infra told), and its running builds may
run out of space. cpu-4 proofs and cpu-1 are 6-14 deep. At 18:22Z cpu-6 research had 3 free slots and cpu-5 research 10,
with nobody waiting. Relaunch any run of yours that has not started, on `--on vy-nebius-cpu-5 --pool research` or
`--on vy-nebius-cpu-6 --pool research`, and cancel the old copy once the new one registers. This supersedes the earlier
rule to avoid cpu-5 and cpu-6. Leave runs that are already building where they are, unless they are on cpu-9 and short
of disk.

## 19:00Z, proofs coordinator: proofs has dedicated Lean nodes now; use them

infra (INF-453): vy-nebius-cpu-11, cpu-12 and cpu-7 are proofs' Lean nodes, with proofs pools of 8-10 build slots each,
until Fri 13 Nov 6 PM PST; ci's hand shards may share them until about 20:30Z. cpu-1 and cpu-9 stay up until Fri 6 Nov
6 PM PST, and cpu-9 refuses runs while its disk is low. Launch new runs with `--on vy-nebius-cpu-11 --pool proofs`,
`--on vy-nebius-cpu-12 --pool proofs` or `--on vy-nebius-cpu-7 --pool proofs`, spread across the three. Move any run of yours
that hasn't started, and cancel the old copy once the new one registers. The first build on each node is cold, so
put Security/Proofs builds there early.

## 20:01Z, proofs coordinator: cpu-9 is back

cpu-9's disk is fixed (INF-452) and it takes runs again. When proofs' nodes cpu-11, cpu-12 and cpu-7 are full (they were
at 19:52Z), use `--on vy-nebius-cpu-9 --pool proofs`, or `--pool research` there, and move any run of yours that is still
waiting, cancelling the old copy once the new one registers.

## 20:54Z, proofs coordinator: heavy Lean builds go to cpu-11 and cpu-12

cpu-1 OOM-killed a Lean run at 20:45Z (five records builds at 60-100 GB each beside training jobs, under a 700 GB cap).
infra doubled check slots on cpu-11 and cpu-12 (8 each, about 735 GB free each) and cpu-7 (9). Send any Security/Proofs
or Security records build to `--on vy-nebius-cpu-11 --pool proofs` or cpu-12; use cpu-1 and cpu-9 only for small
packages. This replaces "cpu-1 or cpu-9 first".

## 21:08Z, proofs coordinator: more Lean capacity

vy-nebius-cpu-3 is back to Fri 13 Nov 6 PM PST with a 14-slot proofs pool and 5 build trees (`--on vy-nebius-cpu-3
--pool proofs`), shared with network-accounting. cpu-5 and cpu-6 research slots were idle at 20:58Z. Heavy Security
builds: cpu-11, cpu-12 or cpu-3. Move any run of yours that is still waiting.

## 21:54Z, proofs coordinator: every PR names its goal (Daniel, today)

From now on each PR body carries `Goal <id>` naming the Goals-page goal it serves (e.g. `Goal proofs/zero-knowledge`,
`Goal proofs/sampled-audits-sound`, `Goal proofs/recursion-knowledge-sound`, `Goal proofs/sampling-generic`,
`Goal proofs/recursion-generic`, `Goal proofs/terminology-clean`, `Goal proofs/zero-knowledge-any-worker`,
`Goal proofs/zk-leaks-closed`, `Goal proofs/firewall-checks-provable`). Add it to any open PR of yours the next time you
edit its body (a body edit needs no new mark). ci may soon refuse a mark on a PR that names none.

## 22:09Z, proofs coordinator: a tries pool for short Lean tries

infra added a `tries` pool (2 slots each, 40 GB gate) on cpu-11, cpu-12, cpu-7 and cpu-3. Run one-file Lean tries with
`verity lean try FILE --on vy-nebius-cpu-11 --pool tries` (or -12, -7, -3); keep records runs off it. And
`verity lean try --on` exiting 1 can be a false failure (a race infra is fixing): check `research status RUN` before
treating it as one.

## 22:16Z, proofs coordinator: your issue

Your Linear issue is PRF-342 (goal label proofs/recursion-knowledge-sound). Every PR you open carries `Refs PRF-342` and `Goal proofs/recursion-knowledge-sound` in its body: from about Wed 1 PM PST ci gates each mark on the issue's goal label.

## 23:17Z, proofs coordinator: pins.py --merged runs Lean builds locally

`pins.py --merged` runs pinned fixtures' generators, which `lake build` on your VM, and nothing refuses it yet (ci CI-383,
hook refusal Wed). Until then, run it only through `research run --on auto`, or recompute pins by hashing.

## 23:38Z, proofs coordinator: #7133 is marked

KAPPA38 marked #7133 at 458725450f5f6e5fa4d0c4ccb0351a8a0025afac (merges #7144@a717ccc6; `KappaEpsCap` unchanged).
Merge it into #7150 (never next), restate `KappaEpsCap` at your constants or delete it if subsumed, try, run the records
the gate asks (heavy on cpu-11/12/3 `--pool proofs`), commit exactly, recheck, mark #7150. Body: 'Stacked on #7133',
`Refs PRF-342`, `Goal proofs/recursion-knowledge-sound`. FDEVFIX's #7158 marks after it.

## 00:53Z, proofs coordinator: check shape.inline before every mark (until LEAN-172)

lean found that something run AFTER a merge (pins.py --merged, check.py --update, or committing a records run's lock)
can drop `shape.inline` entries from a lean-audit.json: #6676 lost 22 that way, and #6075's committed record may have too.
Until lean's fix (LEAN-172) lands: before marking, diff each lock's `shape.inline` against the merge driver's output for
your merge (`python3 tools/verity/lean/merge.py`) and against next's. Restore any entry whose premise the merged tree still
takes inline, as a normal commit, and say so in the PR body.
