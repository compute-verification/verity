[2026-11-11T09:37Z] 079eeeeba906
CROSS, 1:37 AM PST (09:37Z). CROSS, 1:40 AM PST. THE SEVEN: ALL PUSHED, 1 TO 6 ALL MARKED, rc 0 each. Run by me.
PR | new head | mark ended | the preflight's line
#8551 | 0bb3841abce9d5ae012c5445318da4ffddee6af6 | 09:24:51Z | "merged onto `main` at 0a722c6ec984 (`next` at 15ab284aa5d5 doesn't hold it): nothing refused (161 s; collected 27 suite(s))"
#8680 | 09e3ec29ee7485d020aa0d5ba491aa01a4ca6db2 | 09:26:39Z | same, 105 s, 27 suites
#8712 | 0d75e798b9f07047012d460e3f51ed3ffeefa4b1 | 09:28:39Z | same, 104 s, 27 suites
#8778 | 777bc49fda801fa3ce66eba9a8960c036967e82e | 09:30:27Z | same, 104 s, 27 suites
#8809 | f6dbfd9134bc91958c09a8c2e3f4f7c11b490721 | 09:32:16Z | same, 106 s, 27 suites
#8840 | 062c072e6e7b6968495f08f087dc304f4b4f2662 | 09:34:06Z | same, 106 s, 27 suites
#8854 | 356070ed9c763a1d757591c7e4f5a60972a84e25 | 09:35:57Z | same, 107 s, 27 suites
#8775 | 7d6a71567da5a896055354de4b2e37ddd023948e | not marked by me | (yours)
Each mark's last line: "#N is ready at HEAD". The three things, each branch: #8551's conflict was that one hunk and no other path; 2 to 7 had none. The training_slowdown/ row is the old head's byte for byte in all eight, main's inference_only/ row on the line above. The two tests: 5 passed on each merged tree.

#8606, YOUR RULING (2): main is merged, locally, not pushed: 8e36f2a8b905 (parents 68292c6c93e5 and 0a722c6ec984, no conflict, its tree 2f215584c1c2 is merge-tree's). THE THREE CLAIMS, read by me in the code on that tree:
(a) benchmarks/training_slowdown/training_slowdown/calculator.py puts verity/protocols/inference_only/rho first on sys.path and imports `periods` (the calculator's runner module) and `rho.envelopes`, `rho.hardware`, `rho.params`, and `rho.min_cut` where the package has it; it reads periods.FLEETS and hardware.fleet (fleets), PERIODS, envelopes.policy over params.Registry.load() (the fixed policies), and min_cut.bound.
(b) Registry.load() (rho/params.py) loads rho's data/*.toml and what another file states (`sourced()`, `ranges()`): from verity/protocols/inference_only/nci/calibrations.json the calibrated policy's Gamma and K and K's range (CALIBRATIONS, `_policies`, `_k_range`); the rest is inside `verity` (reporting, profile, partition_v2).
(c) The census's deployments are read by calculator.py ITSELF, not through the calculator: DEPLOYMENTS = benchmarks/census/deployments.json, `window()` takes the reference deployment's window_s. So the body's old words "through it, the census's deployments and adversary" are not what the code does: no adversary file is among the suite's inputs, and rho's package reads no census file (params.py's docstring; its tests do).
THE CLAUSE as I have it in the worktree (uncommitted), after "by path: the lf model.":
  "From the slowdown calculator (`verity/protocols/inference_only/rho`), its directory put first on `sys.path`: its fleets and periods, its registry and its min-cut bound, which `calculator.py` reads, and with the registry the inference-only policy's calibrations (`verity/protocols/inference_only/nci/calibrations.json`), which the registry loads. From the census, by path: its deployments, for the reference window."
TWO MORE SHORT LISTS, found by claim (b): since 9d351bf647 (on main before the crossing's base) rho's params.py reads nci's calibrations.json, so two sentences this PR adds name one file of two:
  nci/README.md: "rho reads data files of this directory by path (the algorithm library, `algorithms.json`)" -> "(the algorithm library, `algorithms.json`, and the calibrations, `calibrations.json`)";
  rho/README.md: "In the other direction, `families.py` reads .../nci/algorithms.json when a family names an algorithm of that library (a family of kind `nci`)" gains ", and `params.py` reads .../nci/calibrations.json for the calibrated policy's Γ and K and the range of K."
By your rule (a list known to be short does not go to main) I put both in the same commit UNLESS YOU SAY NO. I commit, push and launch the suites at 1:50 AM PST, or sooner on your word. Checked on the merged tree and holding: the lock's three conjectures name nci's test_tiers.py and test_fit.py; fixtures/artifacts.json registers nci's sources/; rho's tests load nci's tiers.py and slowdown.py by path; nci's statement.py imports rho.theorem.multiple_bound; circuit_properties imports rho.instances.

