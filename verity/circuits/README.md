# verity.circuits

This package writes Verity's circuits down and computes on them, and this README is the way in to all the code that builds,
evaluates or checks a circuit: this package and its subpackages, the served workloads' Definitions
(`verity.definitions`), the private circuits (`verity.definitions.private_circuits`) and `circuit-check`. It says what the
words mean, names the one front door for each job, and links each part's README.

Everything in this package is on the soundness path (`SOUNDNESS_PATH` in `tests/test_boundaries.py`): the verifiers in
`verity/protocols/` (each protocol's `verifier/`) read Programs, their digests and these semantics. The descriptor
format, its canonical bytes and its digests are specified in [`PROTOCOL.md`](PROTOCOL.md), and
`tests/format_vectors.json` pins them.

## Terms

The terms are the [Glossary](../../ARCHITECTURE.md#glossary)'s; [`GLOSSARY.md`](../../GLOSSARY.md) holds the rest.

- A **circuit** is gates and the wires between them. A **gate** outputs one bit: the AND or XOR of two or more bits, a NOT,
  an input gate, or the constant 0 or 1 as one of an operation's own bits. A wire carries its source gate's value and none
  of its own.
- A **value** is the bit a gate takes in one run, and the trace is all of them (`evaluation.reference.Trace`). Gates and
  wires are structure; values are data, and everything a program or a request brings is a value, never a gate. A
  *registered input* (`RegisteredInput<w>_v1`: weights, a norm's epsilon, a table) is bound when the Program is
  registered, and a *run input* (`Input<w>_v1`: a request's tokens and parameters) per run, so a Program holds no input's
  value. A value is always hidden: committed under a hiding leaf and touched only inside a zero-knowledge proof.
- A **Definition** is a named, versioned piece of circuit, registered in `defs.REGISTRY` under `Name_vN`. A *primitive*
  Definition is one gate with a total evaluator on bit patterns; a *composite* Definition is a body of calls of other
  Definitions, batched or scanned, with *statics*, the parameters (a shape, a tensor-core step) its body is written over.
  A *Boolean Definition* reaches only the Boolean gates, so its meaning is its gates. A *word Definition*'s gates output
  w-bit words (`Value<w>`) whose meaning is its evaluators' code, and the word library migrates to Boolean Definitions,
  each a new version of the same name that computes the same function under the word view.
- A **binding** gives a composite's statics their values: `bind(F, N=64, EPS=1e-05)` is a *specialization*, named by its
  id (`RMSNormTriton_v1{N=64,EPS=1e-05}`), and only a specialization is a circuit. A **Program** (`Program(spec)`) is one
  circuit stored compressed as Definitions called from a root body: its descriptor grows with the number of distinct
  Definitions it uses, not with its gate count, and every gate keeps a fixed index in the canonical layout, so any gate can
  be located, evaluated and selected without expanding the circuit. Each Definition module's *binding file*,
  `<module>.circuit-check.py` beside it, lists the bindings circuit-check checks its Definitions at, and its
  `<module>.circuit-check.json` pins what their lowerings measure; one unit of a binding file is a *binding unit*.

## Front doors

### Building a circuit: the Definition catalog

[`catalog.load()`](../definitions/registry/catalog.py) (`verity.definitions.registry.catalog`) returns `(REGISTRY,
errors)` with every module of the registry imported, and `catalog.load(("registry", "ops"))` the primitive circuits'
modules too (`SCOPES`). A module that fails to import is named in `errors`, never read as "not registered". Everything that
resolves a Definition by id goes through it, so a new module's Definitions resolve everywhere, not only where someone
imported the module:

```python
from verity.circuits import Program, bind
from verity.definitions.registry import catalog

REGISTRY, errors = catalog.load()                         # errors == []
rms = bind(REGISTRY["RMSNormTriton_v1"], N=64, EPS=1e-05)
program = Program(rms)                                    # the specialization's root circuit
```

A new Definition is written with `defs.primitive` or `defs.composite` in a module of `verity.definitions.ops` (a primitive
circuit) or of `verity.definitions.registry` (a served kernel), with a binding in its module's binding file:
circuit-check's suite fails a registered Definition family no binding reaches.

### Evaluating: `verity.circuits.evaluation`

`evaluate(target, *args)` is the reference and the default: it runs a Definition gate by gate on plain integers, and it
defines what is correct. `evaluate_batch(target, *arrays)` evaluates a batch over numpy arrays through the kernel
registered for the Definition when there is one (`kernel="auto"`), and computes every input the kernel declines, or all of
them when none is registered, with the reference (bit-sliced, for a Boolean Definition); `kernel="reference"` forces the
reference. A kernel is speed only: `register_kernel` registers it beside what it computes, and `self_check` holds it to the
reference bit for bit. `evaluate_bits` evaluates a Boolean Definition bit-sliced, 64 inputs per machine word. The served
Definitions' fast evaluators are `verity.definitions.numerics`'s, registered the same way.

### Checking: `circuit-check`

[`circuit-check`](../../tools/verity/circuit_check/README.md) holds every circuit to the property checks: correspondence
with the reference on every realization, the declared port intervals, redundant and dead gates, and the partition. A
change that adds or modifies a circuit passes `uv run circuit-check <definition|template|row>`, with its report in the PR.

- `check` runs it as its `circuit-check` step (`tools/verity/check/check.py`): `circuit-check --all`, every registered
  Definition and template, each target's pass kept in check's cache (`--cache`, `--catalog-units`) and reused while
  everything it read is unchanged. A shard of the check runs only its shards of the targets (`--shard K/N`), and
  `circuit-check --combine` joins them. A sharded check can first build every catalog binding's record once, in check's
  records step (`--cc-records`: `circuit-check --all --records-only`), and give each shard those records
  (`--cc-records-from`). The step is off the merge gate (`off_gate` in `tools/verity/check/policy.toml`): a candidate's
  check runs it, and its red holds nothing.
- The fine-grained check ([`verity graph`](../../tools/verity/graph/README.md)) has circuit-check as one
  `circuit-catalog` task, which builds those records, one `circuit-binding` task per catalog binding, which checks only that
  binding's targets from them (`circuit-check --all --catalog-units --binding NAME`), and a `circuit-coverage` task, which
  holds the bindings circuit-check makes to the ones the tasks list
  (`tools/verity/graph/verity_graph/families/circuit_binding.py`).

### Regenerating pins and fixtures: `tools/verity/check/pins.py`

`python tools/verity/check/pins.py` lists every tracked file a command regenerates from the tree, with its command, and
`--stale` and `--regen` find and rewrite the stale pins. This package's test vectors (`tests/format_vectors.json`,
`tests/boolean_vectors.json` and the others its `WRITERS` lists) are each rewritten only by
`uv run python tools/verity/check/pins.py --write PATH`, the command a failing test names. A lowering pin in a
`.circuit-check.json` is circuit-check's own: `uv run circuit-check TARGET --update-pins` writes what the target measures
now, and the diff is reviewed.

## The parts

| Part | What it holds |
| --- | --- |
| this package (below) | Definitions and Programs, their encoding, queries, evaluation, partitions and accounting views |
| [`ops`](../definitions/ops/README.md) | the primitive circuits' Definitions, and their Boolean versions (`ops.boolean`) |
| [`silicon`](../definitions/silicon/README.md) | the GPUs' exact semantics: tensor-core steps, the FP32 pipe, casts |
| [`targets`](../definitions/circuits/targets/README.md) | the subcircuits a proof system is asked to prove |
| [`boolean`](boolean/README.md) | the Boolean core: the three gates, the word view, the GF(2) helpers that write gates |
| [`verity.definitions`](../definitions/README.md) | the served workloads' Definitions, their Boolean versions and fast evaluators |
| [`verity.definitions.private_circuits`](../definitions/private_circuits/README.md) | the universal unit, one public Definition that runs any private circuit of a size class |
| [`circuit-check`](../../tools/verity/circuit_check/README.md) | the property checks every circuit passes |

## Example

A half adder as a Boolean Definition, four of them batched, then evaluated, encoded and queried:

```python
from verity.circuits import Array, Program, Tuple, Value, bind, composite
from verity.circuits.boolean import AND, XOR, is_boolean, operands
from verity.circuits.codec import encode_program, program_digest
from verity.circuits.evaluation import evaluate
from verity.circuits.query_ast import evaluate as select, instances
from verity.circuits.refs import tuple_of

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

- `types` (`Value`, `Array`, `Tuple`, `Record`): `Value(w)` is the only leaf, a w-bit word; arrays, tuples and records are
  static collections of references to leaves, not structured values. `refs` builds collections of references
  (`tuple_of`, `array_of`, `record_of`).
- `defs`: `primitive(name, version, params, ret)` registers a primitive with its evaluator; `composite(name, version,
  statics, signature)` registers a composite whose body function writes it with `B.call`, `B.batch` (one call per
  member, along the given axes) and `B.scan` (a carried state over iterations). `bind(F, **statics)` fixes a composite's
  statics.
- `program`: `Program(fn)` lowers a fully bound composite to its root circuit: one input gate per leaf of each parameter,
  then the body. A parameter is a run input unless `Program(fn, registered=names)` names it as a registered input.
  `theta=names` marks the model's weights.
- `layout`: the canonical layout (`Circuit`): body order, one interval per call, batch members and scan iterations
  consecutive, gate counts cached per specialization, so gate `i` is found by division.
- `constants` and `operations`: what may be a gate and what must be a value. An operation's own definition is gates (a
  format's constants, SHA-512's round constants, the measured GPU tables); `operations.program_data` lists any data a
  Program's gates still hold (empty exactly when they hold none).
- `ports`: what a Definition declares about its values beyond their widths (intervals, sinks, input kinds; `PROTOCOL.md`
  §12), held at `Program` construction and when a descriptor is decoded. `metadata`: names, descriptions and formats of a
  Program's parts, outside every digest.
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
- `requirement`: a registered circuit's memory requirement from its wiring and schedule.
- `stacking`: the stacking property over a Program and its census.
- `population`: the population Programs a window registers under a template query.

## Tests

`tests/` is this package's suite (`uv run tools/verity/check/suites.py verity/circuits`), and `../definitions/ops/tests/` is a suite
of its own.
