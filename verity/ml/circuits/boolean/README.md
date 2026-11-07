# verity.ml.circuits.boolean

The Boolean core every proof works over ([`PROTOCOL.md`](../PROTOCOL.md) §9): three gates on 1-bit wires, `And<n>_v1` and
`Xor<n>_v1` (a member of each family per number of inputs n >= 2; `AND` and `XOR` are the two-input ones) and `Not_v1`,
and the constant bits `Const1[0x0]_v1` and `Const1[0x1]_v1`. The format fixes what each means, and nothing else defines or
re-registers them. A Definition is Boolean (`is_boolean`) when its signature has only 1-bit leaves and every primitive it
reaches is one of these or a 1-bit input.

- `word_type` and `elements`: the word view. An element of a Boolean Definition's port is an innermost array of bits (bit
  j is leaf j, least significant first) or a lone bit, and `word_type` reads each element as one word, so under the word
  view a Boolean Definition takes and returns what its word Definition does. A port's format is metadata the view never
  reads.
- `forms`: GF(2) linear forms and the registered compiler that counts ANDs.
- `trace`: a Boolean Definition from a registered compiler's run over `forms`, written as And/Xor/Not gates;
  `trace.word_of` gives the word Definition a Boolean Definition computes.
- `gather` and `table`: a word, or the entry of a table the program brings, selected by an index, as a multiplexer tree.

The Boolean versions of the primitive circuits are [`verity.ml.circuits.ops`](../ops/README.md)'s (`ops.boolean`), and of
the served kernels the `boolean_*` modules of `verity.ml.definitions.registry`; `verity.ml.definitions.boolean.boolean_version`
finds the one a word Definition has.

## Tests

`tests/test_boolean.py`, in `verity/ml/circuits`'s suite, with its vectors `tests/boolean_vectors.json`.
