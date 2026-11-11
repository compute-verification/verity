# security/Proofs: the proofs

The package `security_proofs` (`security/Proofs/lakefile.toml`): every proof, `<Area>.SecurityProofs.G :
<Area>.Guarantees.G` for each proved guarantee, the lemmas behind them, and under `Properties/` the proofs of
`security/Properties/`. The kernel checks it and nobody reads it for meaning: the locks of `security/` and
`security/Properties/` name its theorems (`proved_in`), and its own lock lists no guarantee.
[`security/README.md`](../README.md) has the argument and what code outside `security/` may do.

## Who may import it

- Nothing requires it. It requires `security`, `security_properties`, the code packages its proofs are about and
  ArkLib (`security/Proofs/lakefile.toml`).
- No module of `Definitions/` or `Properties/` imports a module here, and no guarantee's statement reads a definition
  here (`trees` in `security/lean-audit.json`, which the Lean check, `tools/verity/lean/check.py`, enforces). Flock is
  the exception until its spec is extracted: its guarantees and assumptions are still in `Flock/`, which `trees`'
  `except` lists by module, with the few modules of other areas that read them.
- Outside code may build modules here and run a Lean file that imports them in this package's Lake environment
  (`lake env lean`), or read this package's lock. These name a module here today, so renaming it changes them:
  - `verity/protocols/pous/lean/grade.sh` builds `Proofs.Pous.Statements`, `Proofs.Pous.Sanity` and
    `Proofs.Pous.Accounting.Numbers` for a submission to import, as `submissions/reference.lean` beside it does the
    first; `check.sh` beside it runs the Lean check on `security` with the grader, on scratch submissions that
    import `Proofs.Pous.Statements`.
  - `verity/protocols/spot/verifier/tests/lean/generate.sh` runs `ExfiltrationVectors.lean`
    (`Proofs.Flock.Soundness.Audit.Influence`, `Proofs.Flock.Soundness.Audit.StratifiedMiss`) and
    `ExfiltrationTwoVectors.lean` (`Proofs.Flock.SamplingGeneric.TwoTable`), both beside it.
  - `verity/protocols/flock/prover/tests/lean/generate.sh` runs `RandomnessVectors.lean` beside it
    (`Proofs.Flock.Soundness.PlanDraw`).
  - `verity/protocols/flock/prover/tests/test_lean_verifier.py` builds `Proofs.Flock.Verifier`, which its scratch
    files import; `test_security_inputs.py` beside it checks that the suite's inputs hold that module's imports.
  - `benchmarks/two_stage/tests/guard_node.py` builds `Proofs.Flock.SpotFlock.Code` and
    `Proofs.Flock.SamplingGeneric.TwoAt` and runs `TwoGuard.lean` beside it, which imports them.
  - `verity/protocols/pouw/verifier/tests/test_lean_package.py` reads the lock's `runs` entry of `Proofs.Pouw.Bulk`.

## Areas

Each area but `Properties/` is a root module `<Area>.lean` and the directory `<Area>/`. Its owner is read from the
registry, path by path: `research msg owners PATH`.

| Area | What it holds |
|---|---|
| `Accounting` | The proofs of resource accounting's guarantees; the joint period's bridges |
| `Core` | `Verity.SecurityProofs.G` for core's guarantees, and every lemma behind them |
| `Flock` | Flock's proofs: the verifier's, level3's and soundness's (`Flock/Soundness/README.md`) |
| `Hash` | The hash area's proofs |
| `Nci` | The proofs of the inference-only policy's guarantees |
| `NetworkCertifier` | The network certifier's proofs ([README](NetworkCertifier/README.md)) |
| `Pous` | The proofs of PoUS's guarantees; `Statements` holds the lemmas' statements |
| `Pouw` | PoUW's proofs: every theorem and lemma over `Definitions.Pouw` |
| `Properties/` | The proofs of `security/Properties/`, under each property's area name |
| `Spot` | Populations: which object each policy needs of SPOT's guarantee |
| `WeightSecurity` | Model weight security's proofs: the leakage lemma and its forms at a deployment |
