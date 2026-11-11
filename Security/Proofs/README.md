# Security/Proofs: the proofs

The package `security_proofs` (`Security/Proofs/lakefile.toml`): every proof, `<Area>.SecurityProofs.G :
<Area>.Guarantees.G` for each proved guarantee, the lemmas behind them, and under `Properties/` the proofs of
`Security/Properties/`. The kernel checks it and nobody reads it for meaning: the locks of `Security/` and
`Security/Properties/` name its theorems (`proved_in`), and its own lock lists no guarantee.
[`Security/README.md`](../README.md) has the argument and what code outside `Security/` may do.

## Who may import it

- Nothing requires it. It requires `security`, `security_properties`, the code packages its proofs are about and
  ArkLib (`Security/Proofs/lakefile.toml`).
- No module of `Definitions/` or `Properties/` imports a module here, and no guarantee's statement reads a definition
  here (`trees` in `Security/lean-audit.json`, which the Lean check, `tools/verity/lean/check.py`, enforces). Flock is
  the exception until its spec is extracted: its guarantees and assumptions are still in `Flock/`, which `trees`'
  `except` lists by module, with the few modules of other areas that read them.
- A code package's test may build modules here and run a Lean file that imports them in this package's Lake
  environment (`lake env lean`). These name a module here today, so renaming it changes them:
  - `verity/protocols/pous/lean/grade.sh` builds `Proofs.Pous.Statements`, `Proofs.Pous.Sanity` and
    `Proofs.Pous.Accounting.Numbers` for a submission to import, as `submissions/reference.lean` beside it does the
    first; `check.sh` beside it runs the Lean check on `Security` with the grader.
  - `verity/protocols/spot/verifier/tests/lean/generate.sh` runs `ExfiltrationVectors.lean`
    (`Proofs.Flock.Soundness.Audit.Influence`, `Proofs.Flock.Soundness.Audit.StratifiedMiss`) and
    `ExfiltrationTwoVectors.lean` (`Proofs.Flock.SamplingGeneric.TwoTable`), both beside it.
  - `verity/protocols/flock/prover/tests/lean/generate.sh` runs `RandomnessVectors.lean` beside it
    (`Proofs.Flock.Soundness.PlanDraw`).
  - `verity/protocols/flock/prover/tests/test_lean_verifier.py` builds `Proofs.Flock.Verifier`, which its scratch files import.

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
| `Properties/` | The proofs of `Security/Properties/`, under each property's area name |
| `Spot` | Populations: which object each policy needs of SPOT's guarantee |
| `WeightSecurity` | Model weight security's proofs: the leakage lemma and its forms at a deployment |
