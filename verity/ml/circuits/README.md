# verity.ml.circuits

A circuit is gates and the wires between them: each gate produces one value (`Value<w>`, a word of `w` bits) from the
values on its operand wires. This package writes circuits down compactly and computes on them. A **Program** is one
circuit, written as calls of **Definitions**: a primitive Definition is one gate with a total evaluator on bit patterns,
and a composite Definition is a body of calls of other Definitions, possibly batched or scanned. A Program's descriptor
grows with the number of distinct Definitions it uses, not with its gate count, and every gate still has a fixed index
in the canonical layout, so any gate can be located, evaluated and selected without expanding the circuit.

Everything here is on the soundness path (`SOUNDNESS_PATH` in `tests/test_boundaries.py`): the verifiers in
`verity.core` read Programs, their digests and these semantics. The descriptor format, its canonical bytes and its
digests are specified in [`PROTOCOL.md`](PROTOCOL.md), and `tests/format_vectors.json` pins them.

## Example

A half adder as a Boolean Definition, four of them batched, then evaluated, encoded and queried:

```python
from verity.ml.circuits import Array, Program, Tuple, Value, bind, composite
from verity.ml.circuits.boolean import AND, XOR, is_boolean, operands
from verity.ml.circuits.codec import encode_program, program_digest
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.query_ast import evaluate as select, instances
from verity.ml.circuits.refs import tuple_of

BIT = Value(1)


@composite("HalfAdder", 1, [], lambda S: ((("a", BIT), ("b", BIT)), Tuple(BIT, BIT)))
def HalfAdder(B, S, a, b):
    return tuple_of(B.call(XOR, operands(a, b)), B.call(AND, operands(a, b)))  # (sum, carry)


@composite("HalfAdders", 1, ["N"],
           lambda S: ((("a", Array(S.N, BIT)), ("b", Array(S.N, BIT))), Array(S.N, Tuple(BIT, BIT))))
def HalfAdders(B, S, a, b):
    return B.batch(HalfAdder, a, b, axes=(0, 0))


four = bind(HalfAdders, N=4)            # the static N bound: a specialization, id 'HalfAdders_v1{N=4}'
program = Program(four)
program.gates, program.input_gates      # (16, 8): 8 input gates, then an XOR and an AND per half adder
evaluate(four, [1, 1, 0, 0], [1, 0, 1, 0])   # (0, 1, 1, 0, 1, 0, 0, 0): (sum, carry) per pair, flattened
is_boolean(four)                        # True: only 1-bit wires, And/Xor/Not gates and constant bits
program_digest(encode_program(program)) == program_digest(program)   # True: the digest is the descriptor's
calls = select(program, instances("//HalfAdder_v1"))
calls.count(), calls.by_index(2)        # (4, GateSet(2 gates; [12:14]))
```

## Definitions and Programs

- `types` (`Value`, `Array`, `Tuple`, `Record`): a value is the only leaf; arrays, tuples and records are static
  collections of references to values, not structured values. `refs` builds collections of references (`tuple_of`,
  `array_of`, `record_of`).
- `defs`: `primitive(name, version, params, ret)` registers a primitive with its evaluator; `composite(name, version,
  statics, signature)` registers a composite whose body a builder function writes with `B.call`, `B.batch` (one call per
  member, along the given axes) and `B.scan` (a carried state over iterations). `bind(F, **statics)` fixes a composite's
  statics; the result is a specialization, named by an id such as `RMSNormTriton_v1{N=64,EPS=1e-05}`. Every Definition is
  registered in `defs.REGISTRY` under `Name_vN`.
- `program`: `Program(fn)` lowers a fully bound composite to its root circuit: one input gate per leaf of each parameter,
  then the body. A parameter is bound per run (a run input, `Input<w>_v1`: tokens, temperature) unless
  `Program(fn, registered=names)` names it as bound at registration (a registered input, `RegisteredInput<w>_v1`: weights,
  a norm's epsilon). Either way the Program holds no input's value. `theta=names` marks the model's weights.
- `layout`: the canonical layout (`Circuit`): body order, one interval per call, batch members and scan iterations
  consecutive, gate counts cached per specialization, so gate `i` is found by division.
- `constants` and `operations`: what may be a gate and what must be a value. An operation's own definition is gates (a
  format's constants, SHA-512's round constants, the measured GPU tables); everything a program or a request brings is a
  value, and `operations.program_data` lists any data a Program's gates still hold (empty exactly when they hold
  none).
- `ports`: what a Definition declares about its values beyond their widths (intervals, sinks, input kinds; `PROTOCOL.md`
  §12), held at `Program` construction and when a descriptor is decoded.
- `library`: the hardware and protocol objects core owns, each labelled `library`.

## Encoding

`codec` writes a Program as a JSON descriptor and reads it back: `encode_program(program)`, `decode_program(descriptor,
REGISTRY)` (it resolves every Definition through the registry and refuses what the format does not allow),
`program_digest` (SHA-512 of the canonical bytes, RFC 8785) and `descriptor_size`.

## Queries

`query_ast` is the query API: `evaluate(program, query)` returns a `Family`, an ordered lazy sequence of `GateSet`s, each a
set of gate intervals. Results may overlap or leave gates out; whether a family is a partition is checked separately
(`partition`). Queries are built from constructors: `instances(path)` (one gate set per call a path selects, such as
`//HalfAdder_v1`), `members`, `nodes`, `each`, `concat`, `union`, `minus`, `inter`, `where`, `root`; the module's
docstring states each. `parse(text)` reads a query from its text form, and `to_json`, `from_json` and `query_id` (in
`query_codec`) give a query a canonical encoding and a SHA-512 id.

## Evaluation

`evaluation` computes values. `evaluate(target, *args)` runs a Definition on one input of plain integers, gate by gate
(the reference every fast path is checked against). `evaluate_batch` runs it over numpy arrays through a registered
kernel (`register_kernel`) where one covers the specialization, and falls back to the reference where none does or the
kernel declines; `self_check` holds a kernel to the reference. `evaluate_bits` runs a Boolean Definition bit-sliced, 64
instances per machine word.

## The Boolean profile

`boolean` is the profile every proof works over (`PROTOCOL.md` §9): three gates on 1-bit wires, `And<n>_v1`, `Xor<n>_v1`
and `Not_v1`, and the two constant bits. A Boolean Definition uses only those, through every call (`is_boolean`). A
Boolean Definition names the word Definition it computes, and its **word view** reads its bit arrays as that Definition's
words (`elements`). `boolean.forms` is a GF(2) circuit builder that counts ANDs, `boolean.trace` turns a builder's run
into a Boolean Definition, and `boolean.gather` and `boolean.table` select a word, or a table entry, by an index.

Word Definitions (a primitive on 32-bit words with a Python evaluator, say) are the older form: what such a gate computes
is its evaluator's code. Each is being replaced by its Boolean version (`ops.boolean`), whose meaning is its gates.

## Partitions and proof blocks

A proof splits a Program into proof blocks, each proved on its own, which read each other only through committed values.

- `cut`: the partition queries. `Q_word` v1 cuts each call into proof blocks whose committed outputs are at most `X`
  bits, or one value of at most `W` bits, with no value computed in two proof blocks; v2 is a delta on it. `Q_call` v1
  bounds only what a proof block outputs (at most `X` bits); its inputs are unbounded.
- `partition_object`: `verity/partition/v1`, a Program's partition named as a query the verifier evaluates on the
  circuit it holds. `partition_v2`: `verity/partition/v2`, a private partition registered as advice under a hidden
  root, with the public predicates it satisfies.
- `partition`: checks that a family used as a partition is one (every gate in exactly one proof block) and meets the
  width rule.
- `units`: each proof block as a standalone word circuit; `boundary`, `parts` and `intervals`: the boundary sets, part
  types and interval sets the analyses share.

## Accounting views

Pure functions of a Program, read by the resource audits:

- `census`: `verity/census/v1`, what a Program computes, counted by class, and the values its root's calls pass.
- `liveness`: dead gates (`dead_gates(program)`), those no output depends on.
- `requirement` and `requirement_units`: a declared circuit's memory requirement from its wiring and schedule, and its
  two proof blocks (the plan, and the memory map).
- `stacking`: the stacking property over a Program and its census.
- `population`: the population Programs a window registers under a template query.

## Subpackages

- `ops`: the hardware-faithful semantics of ML primitives (FP32 and BF16 arithmetic, MUFU tables, GEMM steps, softmax,
  norms, quantization, MoE routing), registered as Definitions; `ops.boolean` holds their Boolean versions. Each module's
  docstring says what it registers, and each Definition module has a circuit-check sidecar beside it
  (`tools/verity/circuit_check/`).
- `silicon`: the exact semantics of one tensor-core accumulator step per instruction class (`models`: `GroupSum`,
  `BlockScaledAlignAdd`), operand formats (`term`), and `instructions.INSTRUCTIONS`, which says for each instruction which
  model it uses and whether that model is pinned by captures or not yet probed. `benchmarks/tc_probe/` and
  `benchmarks/tc_probe_fp4/` capture the hardware these models are held to.
- `targets`: `target.Target`, a subcircuit with every parameter bound, the exact function a proof system is asked to
  prove; `target.TARGETS` lists them, frozen or draft.

## Tests

`tests/` is this package's suite (`uv run tools/verity/check/suites.py verity/ml/circuits`), and `ops/tests/` is a suite
of its own.
