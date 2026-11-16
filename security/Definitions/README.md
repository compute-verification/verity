# security/Definitions: the spec

This tree is where Verity's claims are written down. Each area gives its vocabulary, the assumptions its
guarantees may take (`Assumptions`), and each guarantee's statement (`Guarantees`), a `def G : Prop`. It is the part a
reader reads: the proofs are in `security/Proofs/`, which the kernel checks and nobody reads for meaning. The tree is
the package `security` (`security/lakefile.toml`, root `Definitions.lean`). [`security/README.md`](../README.md) has the
argument, the lock and what code outside `security/` may do.

## Try it

~~~sh
git grep -n "^def .* : Prop" -- security/Definitions/Pous/Guarantees.lean
~~~

This lists the statements in PoUS's guarantees file, each with its line, which is where to read one. It needs only the
checkout.

## Who may import it

- `Properties.*` and `Proofs.*` import it. It imports neither: `security` requires only Mathlib
  (`tests/test_lean_packages.py`), and no module here imports one of `Proofs/` (`trees` in `security/lean-audit.json`).
- Inside the tree, `layers` in `security/lean-audit.json` lists the module prefixes each prefix may import. An area's
  definitions import no assumption, guarantee or proof; its `Assumptions` read the definitions; its `Guarantees` read
  both.
- The Lean check (`tools/verity/lean/check.py`) fails on an import outside `layers` or `trees`, and on a `.lean` file
  that no root imports.
- One pin's command (`tools/check/pins.py`) runs Lean in this package's Lake environment:
  `tests/verity/protocols/resource_accounting/compute/pouw/verifier/ncp_lean.py` writes a scratch file importing `Definitions.Pouw.NCP.RouteU` into
  `security/`, and the command runs it there and removes it.

## Areas

Each area is a root module `<Area>.lean` and the directory `<Area>/`. Its owner is read from the registry, path by
path: `research msg owners PATH`.

| Area | What it states |
|---|---|
| `ResourceAccounting` | The device model every resource audit is about, and a joint audit's run on it |
| `Assumptions` | The named assumptions: the few `Prop`s the guarantees take without proof |
| `Certifier` | The network attester's protocol: grids, the interaction, the Program |
| `Claims` | Flock's claim statements, over Core's interaction model |
| `Core` | The model every area states its guarantees in: circuits, partitions, coins |
| `Flock` | The vocabulary Flock's guarantees are stated in, extracted from `Proofs/Flock` |
| `Hash` | Messages, digests and collisions; frame-v3 trees; the served row leaf; the keyed leaf |
| `InferenceOnly` | The inference-only policy: `Complies c k`, `CompliesRate c α` |
| `Pous` | PoUS: its game, its models and the requirement `Pous.Meets` |
| `Pouw` | PoUW's protocol: the definitions its statements read |
| `Spot` | Populations: proof blocks over committed value slots, their dataflow, cones and egress |
| `WeightSecurity` | Model weight security: the setting |

## Generated files

A generated file is its generator's: change the generator and run it, never the file.

- `ResourceAccounting/Headline/` and `ResourceAccounting/HeadlineDc/`: `machine-model lean-headline` writes the instances there, which
  import the `Encoding.lean` beside them.
- `ResourceAccounting/Sheets/`: `benchmarks/machine_model/`, the ACCEL model of each measured card, prints each file
  from the card's data sheet (`machine-model --device DEVICE sheet --lean`).

Each generator's tests hold the files to its output. `security/README.md` lists every program that writes Lean under
`security/`.
