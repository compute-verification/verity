# security/Properties: what the code does

The package `security_properties` (`security/Properties/lakefile.toml`): statements about the code's Lean functions,
the checks each protocol runs, each listed in `security/Properties/lean-audit.json` and proved under
`security/Proofs/Properties/` (`proved_in`). [`security/README.md`](../README.md) has the argument and what code outside
`security/` may do.

## Who may import it

- `Proofs.*` imports it, and nothing else: `security_proofs` is the one package that requires it
  (`security/Proofs/lakefile.toml`).
- It requires `security` and, by path, the code packages it states properties of (`security/Properties/lakefile.toml`).
  No module here imports one of `Proofs/` (`trees` in `security/lean-audit.json`).
- `layers` in `security/Properties/lean-audit.json` lists what each module prefix may import: the definitions and the
  code a property is about, never a proof.
- The Lean check (`tools/verity/lean/check.py`) fails on an import outside `layers` or `trees`.

## Areas

An area is a directory, a root module, or both; the code it is about is named by its Lean root. Its owner is read
from the registry, path by path: `research msg owners PATH`.

| Area | What it states |
|---|---|
| `Accounting/` | `RemoteAudit`'s verdicts, the joint period's bridges, a registered circuit's units and space |
| `Core/` | The evaluator's primitives and the Site's fold and check are `Definitions.Core`'s |
| `Flock/` | The circuit a registered Flock circuit file describes; the verifier's two fixed nets build |
| `Hash/` | Core's identity hash is Flock's SHA-512 |
| `Nci/`, `Nci.lean` | What the inference-only check's verdicts mean (`NciCheck`) |
| `NetworkCertifier/` | The certificate and records checks (`NetworkCheck`), the certifier device |
| `Pous/`, `Pous.lean` | PoUS's setup, timed audit, period and wipe (`PousCheck`, `MemoryChallenger`) |
| `Pouw/`, `Pouw.lean` | PoUW's window, its check, commitments, credit and period (`PouwWindow`, `PouwCheck`) |
| `Spot/`, `Spot.lean` | SPOT's one-stage audit: its verdict, sample, plan check and wires (`Spot`) |
| `Verity/` | Soundness at a composite's compiled verifier ([README](Verity/README.md)) |
| `Annot.lean` | The description-level annotation check's property (`Flock.AnnotExpand`) |
| `Attest.lean` | A party's signed period roots and liveness records (`Attest.Period`) |
| `Challenger.lean` | Coins from the device's log: what a sample's coins are (`Spot.OneStageZk`) |
| `Composition.lean` | The composition of core protocols (`CoreComposition`) |
| `Firewall.lean` | The ZK server's record (`Firewall.Record`, `Firewall.Release`) |
| `NetworkPeriod.lean` | The network period check's bound (`NetworkCheck.Period`) |
| `NetworkScan.lean` | The rows form's scan fits its layout (`NetworkCheck.Period.scanOf`) |
| `Remote.lean` | Remote's acceptance and the inference-only policy (`CoreRemote`) |
