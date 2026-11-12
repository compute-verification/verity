# verity.definitions.private_circuits

A private circuit's gates and wiring are a value. `universal.UniversalUnit_v1{G, N_IN, N_OUT}` is one public Boolean
Definition that runs any private circuit of its size class (G gates over N_IN inputs, N_OUT outputs): it takes the
private circuit's description as an input beside the data, so its own structure depends on the size class alone. It is not
how a private circuit is hidden: circuit hiding is recursion (Flock's `--hide`). The terms and the front doors are
[`verity.circuits`](../../circuits/README.md)'s.

- `Netlist`: a private circuit in SSA form, each gate an XOR, AND, XNOR or NAND of two wires before its own, and any
  other index reads 0. So every bit string describes a private circuit, the Definition is total and no description has a
  cycle.
- `interpret`: the private circuit run gate by gate, the reference the Definition agrees with; `pad` fits a smaller private
  circuit to a class without changing its outputs; `arguments` gives a private circuit and its data as the unit's four inputs.
- `build` and `traced`: the unit over `forms`, and the Boolean Definition traced from it; `and_count` is its AND count in
  closed form, about 2 N_IN + G ANDs per gate of the private circuit.

The module's docstring gives the design and its cost, and the unit's binding file is `universal.circuit-check.py`.

## Tests

`tests/` is this package's suite (`uv run tools/verity/check/suites.py verity/definitions/private_circuits`), and
`benchmarks/private_circuit/` measures the unit.
