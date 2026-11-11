[2026-11-11T04:51Z] c3783f934316
bounds, 04:52Z. All eight runs are launched and the two old PRs are retired as far as I can.

RUNS (run by me, `research run --on auto`; Lean = `check.py --build security/Proofs security security/Properties`, `--cwd clone`, from nothing; suites = `suites.py --changed 5c095bb4ab70 --quick`):
- #8810 at cb6cd0475a517cd90feda4d345dc57777f90a52b: Lean r20261111-043451-398f (vy-nebius-cpu-11, waiting for a slot), suites x8 r20261111-043522-7fc7 (cpu-12, running)
- #8811 at b51ca39c38411a1f8582e13bbe08aac71a444cde: Lean r20261111-044221-f8e3 (cpu-11, waiting), suites x8 r20261111-044251-f4c1 (cpu-6, running)
- #8593 at 1a3cc0b7a385c6c3ac507b62367551ac055119c2: Lean r20261111-044612-4d76 (cpu-3, running, Lean slot since 04:49Z), suites x52 r20261111-044646-6ba0 (cpu-1, waiting)
- #8595 at e19a95b5ea358769d6e2ba066f6846cbcd93e98c: Lean r20261111-044827-5d47 (cpu-2, waiting, said about 13 min), suites x52 r20261111-044903-3320 (cpu-5, waiting, about 13 min)

RETIRED: #8502 and #8504, through `research queue edit`, as #8561/#8564 were: title `[replaced by #8810] ...` / `[replaced by #8811] ...`, body's first line "Replaced by #88xx: the same change, crossed over the layout move, on a branch its worker can push.", the old body under it. Neither is marked again; both are still OPEN drafts with their branches at be48898185c0 and 3f4c4916dc12. The close is not mine to do: #8561 and #8564 were closed from Daniel's account (read from GitHub's events, both 09:01:19Z 10 Nov) and `research queue` has no close. `research queue status` (run 04:49Z) shows #8502 `refused` (conflicts with main) and #8504 `waiting` on it; #8810, #8811, #8593, #8595 `waiting`, not marked.

NEXT: the bodies of #8593 and #8595 (their crossing, the new base), the checkpoint note, then #8490 after ci's push. Nothing is marked until each head's own runs pass.

