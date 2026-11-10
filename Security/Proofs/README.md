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
  - `verity/core/protocols/pous/lean/grade.sh` builds `Proofs.Pous.Statements`, `Proofs.Pous.Sanity` and
    `Proofs.Pous.Accounting.Numbers` for a submission to import, as `submissions/reference.lean` beside it does the
    first; `check.sh` beside it runs the Lean check on `Security` with the grader.
  - `verity/core/protocols/spot/tests/lean/generate.sh` runs `ExfiltrationVectors.lean`
    (`Proofs.Flock.Soundness.Audit.Influence`, `Proofs.Flock.Soundness.Audit.StratifiedMiss`) and
    `ExfiltrationTwoVectors.lean` (`Proofs.Flock.SamplingGeneric.TwoTable`), both beside it.
  - `verity/ml/flock/tests/lean/generate.sh` runs `RandomnessVectors.lean` beside it
    (`Proofs.Flock.Soundness.PlanDraw`).
  - `verity/ml/flock/tests/test_lean_verifier.py` builds `Proofs.Flock.Verifier`, which its scratch files import.

## Areas

Each area but `Properties/` is a root module `<Area>.lean` and the directory `<Area>/`. The owner is the registry's
(`research msg owners PATH`).

| Area | Owner | What it holds |
|---|---|---|
| `Accounting` | @lean, by file | The proofs of resource accounting's guarantees; the joint period's bridges |
| `Core` | @proofs | `Verity.SecurityProofs.G` for core's guarantees, and every lemma behind them |
| `Flock` | @proofs | Flock's proofs: the verifier's, level3's and soundness's (`Flock/Soundness/README.md`) |
| `Hash` | @proofs | The hash area's proofs |
| `Nci` | @compute-accounting | The proofs of NCI's guarantees |
| `NetworkCertifier` | @network-accounting | The network certifier's proofs ([README](NetworkCertifier/README.md)) |
| `Pous` | @memory-accounting | The proofs of PoUS's guarantees; `Statements` holds the lemmas' statements |
| `Pouw` | @compute-accounting | PoUW's proofs: every theorem and lemma over `Definitions.Pouw` |
| `Properties/` | @lean | The proofs of `Security/Properties/`, under each property's area name |
| `Spot` | @lean | Populations: which object each policy needs of sampled proofs' guarantee |
| `WeightSecurity` | @lean | Model weight security's proofs: the leakage lemma and its forms at a deployment |

Under `Accounting`, the registry gives files to @memory-accounting (the bandwidth, rate and remote audits, memory,
gates) and to @compute-accounting (PoUW's work, coin flow, `TS`).
