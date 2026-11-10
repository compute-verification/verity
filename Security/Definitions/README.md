# Security/Definitions: the spec

The one tree of the package `security` (`Security/lakefile.toml`, root `Definitions.lean`): each area's vocabulary, its
premises (`Assumptions`) and its guarantee statements (`Guarantees`), each a `def G : Prop`. It is the part a reader
reads. [`Security/README.md`](../README.md) has the argument, the lock and what code outside `Security/` may do.

## Who may import it

- `Properties.*` and `Proofs.*` import it. It imports neither: `security` requires only Mathlib
  (`tests/test_lean_packages.py`), and no module here imports one of `Proofs/` (`trees` in `Security/lean-audit.json`).
- Inside the tree, `layers` in `Security/lean-audit.json` lists the module prefixes each prefix may import. An area's
  definitions import no assumption, guarantee or proof; its `Assumptions` read the definitions; its `Guarantees` read
  both.
- The Lean check (`tools/verity/lean/check.py`) fails on an import outside `layers` or `trees`, and on a `.lean` file
  that no root imports.
- One test runs Lean in this package's Lake environment: `verity/core/protocols/pouw/tests/ncp_lean.py` writes a
  scratch file importing `Definitions.Pouw.NCP.RouteU`, which its pin's command (`tools/verity/check/pins.py`) runs in
  `Security/` and removes.

## Areas

Each area is a root module `<Area>.lean` and the directory `<Area>/`. The owner is the registry's
(`research msg owners PATH`).

| Area | Owner | What it states |
|---|---|---|
| `Accounting` | @lean, by file | The device model every resource audit is about, and a joint audit's run on it |
| `Assumptions` | @lean | The named assumptions: the few `Prop`s the guarantees take without proof |
| `Claims` | @lean | Flock's claim statements, over Core's interaction model |
| `Core` | @proofs | The model every area states its guarantees in: circuits, partitions, coins |
| `Flock` | @lean | The vocabulary Flock's guarantees are stated in, extracted from `Proofs/Flock` |
| `Hash` | @proofs | Messages, digests and collisions; frame-v3 trees; the served row leaf; the keyed leaf |
| `Nci` | @compute-accounting | The inference-only policy: `Complies c k`, `CompliesRate c α` |
| `NetworkCertifier` | @network-accounting | The network certifier's protocol: grids, the interaction, the Program |
| `Pous` | @memory-accounting | PoUS: its game, its models and the requirement `Pous.Meets` |
| `Pouw` | @compute-accounting | PoUW's protocol: the definitions its statements read |
| `Spot` | @lean | Populations: units over committed value slots, their dataflow, cones and egress |
| `WeightSecurity` | @lean | Model weight security: the setting |

Under `Accounting`, the registry gives files to @memory-accounting (the bandwidth, rate and remote audits, memory,
gates) and to @compute-accounting (PoUW's work, coin flow, `TS`).

## Generated files

A generated file is its generator's: change the generator and run it, never the file.

- `Accounting/Headline/` and `Accounting/HeadlineDc/` are compute-research's. Its headline exporter
  (`benchmarks/pouw/lean_headline.py`, `benchmarks/pouw/lean_headline_dc.py`) writes the instances there, which import
  the `Encoding.lean` beside them.
- `Accounting/Sheets/` is what the machine model prints (`machine-model --device DEVICE sheet --lean`,
  `benchmarks/pouw/machine_model/`).

Each generator's tests hold the files to its output. `Security/README.md` lists every program that writes Lean under
`Security/`.
