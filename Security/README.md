# Security

The Lean every protocol states its guarantees in, the properties of its code, and their proofs, in three trees (Daniel,
23 Oct). `AGENTS.md` ("Lean") has the rules, and `.agents/skills/lean-proofs/SKILL.md` how to work here.

| tree | package | holds | may require |
|---|---|---|---|
| `Definitions/` | `security` (`lakefile.toml` here) | each area's vocabulary, and its named assumptions and guarantee statements in `Definitions/<Area>/{Assumptions,Guarantees}` | Mathlib only |
| `Properties/` | `security_properties` (`Properties/lakefile.toml`) | properties of the code, stated against its Lean functions, one `Properties.<Protocol>` per component | `security` and the code it states properties of, never `security_proofs` |
| `Proofs/` | `security_proofs` (`Proofs/lakefile.toml`) | every proof and lemma: `<Area>.SecurityProofs.G` for each proved guarantee, and the properties' proofs in `Proofs/Properties/` | ArkLib, `security`, `security_properties`, the code packages |

The properties' proofs are roots of `security_proofs`, not a package of their own: two Lake packages can't share the
module root `Proofs`.

Areas: `Accounting` (the device model the resource audits share, and their joint guarantee), `Core`, `Flock`, `Hash`,
`Nci`, `NetworkCertifier`, `Pous` and `Pouw`, each a directory in every tree it has Lean in, with `Claims` and
`Population` in `Definitions/` and `Proofs/`. C-Flock's definitions aren't extracted yet: `Definitions/Flock/` holds
only its guarantees, and the rest of its analysis is in `Proofs/Flock/`. `Properties/` still names some modules by
component rather than area (`CertifierDevice`, `CertifierNetwork`, `NetworkRecords`, `NciDesc`, `Identity`,
`Evaluate`, `RemoteAudit`, `SampledProofs`) until they're renamed into their areas.

Each package's lock is its `lean-audit.json`: the guarantees, the definitions they read, the assumptions and the
import layers that the Lean check (`tools/verity/lean/check.py`) holds the package to.
