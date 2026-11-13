**The pull request is open and unmarked: [#9683](https://github.com/compute-verification/verity/pull/9683), head `c00bf07b22584705513ad11879f5b3d0a014ab95`, against `main`. Every run I read at that head passed.**

- **circuit-check:** the three wide GEMMs pass without `--as-call`, each partitioned by its callees, and the confirming pass wrote no pins diff.
- **Walk tests:** both pass. With the coordinate's entry removed, the second test fails and names the coordinate.
- **Suites:** the 57 suites `--changed` picks pass (`--quick`), and `repository` and `tools/verity/circuit_check` pass whole.

The body has the reports, each run with what I read in it, and what the change adds to `check`. The circuit-check step's targets and timing are unchanged, apart from the two coordinate targets now checked as Calls. The `tools/verity/circuit_check` suite run on its own rebuilds its catalog after any edit to `targets.py`, as my previous comment explains. Left for the coordinator: the pre-mark checks and the mark.
