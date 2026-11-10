# Security/Properties: what the code does

The package `security_properties` (`Security/Properties/lakefile.toml`): statements about the code's Lean functions,
the checks each protocol runs, each listed in `Security/Properties/lean-audit.json` and proved under
`Security/Proofs/Properties/` (`proved_in`). [`Security/README.md`](../README.md) has the argument and what code outside
`Security/` may do.

## Who may import it

- `Proofs.*` imports it, and nothing else: `security_proofs` is the one package that requires it
  (`Security/Proofs/lakefile.toml`).
- It requires `security` and, by path, the code packages it states properties of (`Security/Properties/lakefile.toml`).
  No module here imports one of `Proofs/` (`trees` in `Security/lean-audit.json`).
- `layers` in `Security/Properties/lean-audit.json` lists what each module prefix may import: the definitions and the
  code a property is about, never a proof.
- The Lean check (`tools/verity/lean/check.py`) fails on an import outside `layers` or `trees`.

## Areas

An area is a directory, a root module, or both; the code it is about is named by its Lean root. The owner is the
registry's (`research msg owners PATH`).

| Area | Owner | What it states |
|---|---|---|
| `Accounting/` | @lean | `RemoteAudit`'s verdicts, the joint period's bridges, a declared circuit's units and space |
| `Core/` | @lean | The evaluator's primitives and the Site's fold and check are `Definitions.Core`'s |
| `Flock/` | @lean | The circuit a registered Flock circuit file describes; the verifier's two fixed nets build |
| `Hash/` | @lean | Core's identity hash is Flock's SHA-512 |
| `Nci/`, `Nci.lean` | @lean | What the inference-only check's verdicts mean (`NciCheck`) |
| `NetworkCertifier/` | @lean | The certificate and records checks (`NetworkCheck`), the certifier device |
| `Pous/`, `Pous.lean` | @lean | PoUS's setup, timed audit, period and wipe (`PousCheck`, `MemoryChallenger`) |
| `Pouw/`, `Pouw.lean` | @lean | PoUW's window, its check, commitments, credit and period (`PouwWindow`, `PouwCheck`) |
| `Spot/`, `Spot.lean` | @lean | Sampled proofs' one-stage audit: its verdict, sample, plan check and wires (`Spot`) |
| `Verity/` | @lean | Soundness at a composite's compiled verifier ([README](Verity/README.md)) |
| `Annot.lean` | @lean | The description-level annotation check's property (`Flock.AnnotExpand`) |
| `Attest.lean` | @lean | A party's signed period roots and liveness records (`Attest.Period`) |
| `Challenger.lean` | @lean | Coins from the device's log: what a sample's coins are (`Spot.OneStageZk`) |
| `Composition.lean` | @lean | The composition of core protocols (`CoreComposition`) |
| `Firewall.lean` | @lean | The ZK server's record (`Firewall.Record`, `Firewall.Release`) |
| `NetworkPeriod.lean` | @lean | The network period check's bound (`NetworkCheck.Period`) |
| `NetworkScan.lean` | @lean | The rows form's scan fits its layout (`NetworkCheck.Period.scanOf`) |
| `Remote.lean` | @lean | Remote's acceptance and the inference-only policy (`CoreRemote`) |
