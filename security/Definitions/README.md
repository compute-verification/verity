# security/Definitions: the spec

The one tree of the package `security` (`security/lakefile.toml`, root `Definitions.lean`): each area's vocabulary, its
premises (`Assumptions`) and its guarantee statements (`Guarantees`), each a `def G : Prop`. It is the part a reader
reads. [`security/README.md`](../README.md) has the argument, the lock and what code outside `security/` may do.

## Who may import it

- `Properties.*` and `Proofs.*` import it. It imports neither: `security` requires only Mathlib
  (`tests/test_lean_packages.py`), and no module here imports one of `Proofs/` (`trees` in `security/lean-audit.json`).
- Inside the tree, `layers` in `security/lean-audit.json` lists the module prefixes each prefix may import. An area's
  definitions import no assumption, guarantee or proof; its `Assumptions` read the definitions; its `Guarantees` read
  both.
- The Lean check (`tools/verity/lean/check.py`) fails on an import outside `layers` or `trees`, and on a `.lean` file
  that no root imports.
- One pin's command (`tools/verity/check/pins.py`) runs Lean in this package's Lake environment:
  `verity/protocols/pouw/verifier/tests/ncp_lean.py` writes a scratch file importing `Definitions.Pouw.NCP.RouteU` into
  `security/`, and the command runs it there and removes it.

## Areas

Each area is a root module `<Area>.lean` and the directory `<Area>/`. Its owner is read from the registry, path by
path: `research msg owners PATH`.

| Area | What it states |
|---|---|
| `Accounting` | The device model every resource audit is about, and a joint audit's run on it |
| `Assumptions` | The named assumptions: the few `Prop`s the guarantees take without proof |
| `Claims` | Flock's claim statements, over Core's interaction model |
| `Core` | The model every area states its guarantees in: circuits, partitions, coins |
| `Flock` | The vocabulary Flock's guarantees are stated in, extracted from `Proofs/Flock` |
| `Hash` | Messages, digests and collisions; frame-v3 trees; the served row leaf; the keyed leaf |
| `Nci` | The inference-only policy: `Complies c k`, `CompliesRate c α` |
| `NetworkCertifier` | The network certifier's protocol: grids, the interaction, the Program |
| `Pous` | PoUS: its game, its models and the requirement `Pous.Meets` |
| `Pouw` | PoUW's protocol: the definitions its statements read |
| `Spot` | Populations: units over committed value slots, their dataflow, cones and egress |
| `WeightSecurity` | Model weight security: the setting |

## Generated files

A generated file is its generator's: change the generator and run it, never the file.

- `Accounting/Headline/` and `Accounting/HeadlineDc/`: `benchmarks/pouw/lean_headline.py` and
  `benchmarks/pouw/lean_headline_dc.py` write the instances there, which import the `Encoding.lean` beside them.
- `Accounting/Sheets/`: `benchmarks/machine_model/`, the ACCEL model of each measured card, prints each file
  from the card's data sheet (`machine-model --device DEVICE sheet --lean`).

Each generator's tests hold the files to its output. `security/README.md` lists every program that writes Lean under
`security/`.
