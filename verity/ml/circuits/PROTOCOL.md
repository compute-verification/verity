# `verity-ir/descriptor/v1`: the program format

This is the byte format of a Program and the rules that turn it into a circuit: from the canonical bytes of a descriptor, a
reader obtains every gate, its primitive and its operand gates, and nothing else is needed. It is written for a second
implementation (the Lean port of `Q_word` first). It describes the bytes the codec already writes and moves no digest.

- Reference: `verity.ml.circuits.codec` (`encode_program`, `decode_program`, `canonical_json`, `jcs_domain_error`, `program_digest`),
  with `types`, `refs`, `defs` (`Node`, `Body`, `Builder`), `program` (the root, `INPUT_ID`), `layout` (`Circuit`, `resolve`),
  `parts.n_members`, for §8, §10 and §11 `cut` and `partition_object`, and for §9 `boolean`.
- Vectors: `verity/ml/circuits/tests/format_vectors.json`, written by
  `python verity/ml/circuits/tests/test_format_spec.py --write`.
- Conformance: `verity/ml/circuits/tests/test_format_spec.py` checks the reference against every vector, and restates
  canonical JSON (as RFC 8785), leaf order, reference sequences and the interning order independently of it. Each section
  names its vector keys. `verity/ml/circuits/tests/partition_vectors.json` pins `Q_word` v1 on top of this format.
- Scope: schema v1. `verity-ir/descriptor/v0` (every body verbatim, types inline, no alias) still decodes; it is not
  specified here.

Indices are 0-based, `[a, b)` is half-open, and a *leaf* is one scalar position of a type.

## 1. Canonical JSON and the program digests

The canonical bytes of a JSON value are its RFC 8785 (JCS) serialization. A descriptor is held to a domain on which JCS is
short to state and on which `codec.canonical_json` (Python's
`json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`) writes exactly the JCS bytes:

- values are objects with string keys, arrays, strings, integers, `true`, `false` and `null`;
- every string, keys included, consists of the characters U+0000..U+007E (ASCII without DEL);
- every integer n has |n| ≤ 2^53 − 1 (`codec.MAX_SAFE_INT`), and there are no other numbers (a float static is written as a
  string, §4.3).

On this domain the bytes are:

- no whitespace; an array is `[`, its elements in order separated by `,`, then `]`;
- an object is `{`, its members `"key":value` sorted by key and separated by `,`, then `}`; keys compare as byte strings, which
  for ASCII is RFC 8785's UTF-16 code-unit order;
- a string is `"`, its characters, `"`: `"` as `\"`, `\` as `\\`, U+0008 `\b`, U+0009 `\t`, U+000A `\n`, U+000C `\f`,
  U+000D `\r`, any other character below U+0020 as `\u00xx` (lowercase hex), every other character (`/` included) as itself;
- an integer is its decimal: `-` for a negative, no leading zeros, `0` for zero.

Outside the domain Python and JCS differ: Python escapes non-ASCII and DEL as `\uXXXX` where JCS writes UTF-8, JCS writes
2^53 + 1 as `9007199254740992` and 10^21 as `1e+21`, `1.0` is `1.0` against `1`, and non-BMP keys sort differently.
`encode_program` and `decode_program` refuse a descriptor whose free-form parts leave the domain (`jcs_domain_error`): every
top-level field but `definitions` and `annotations`, and each definition's key, `id`, `bindings`, `params`, `ret`, and its
body's `params` and `ret_type`. The rest of a descriptor is integers bounded by the program's own table sizes and leaf counts,
and strings of a fixed vocabulary, which the decoder checks (§4.6).

A stored descriptor file need not be canonical (builds write it with spaces). The program digest, `program_digest`, is the hex
SHA-512 of the canonical bytes of the descriptor without `annotations`: the Build's program digest and the `program` field of
`verity/partition/v1`. `annotations` never enter it and carry no meaning in this format. It carries no domain prefix (the
partition object's digests do).

Vectors: `canonical_json.in_domain` (value, bytes), `canonical_json.outside` (value, Python's bytes, the JCS bytes, the
refusal), `program_digest`.

## 2. Types and leaves

~~~text
["Value", w]                  w ≥ 1: one leaf of w bits
["Array", n, T]               n ≥ 0: n elements of type T
["Tuple", [T0, T1, ...]]      fields in order
["Record", [[k0, T0], ...]]   named fields in declared order, names distinct
~~~

leaves(`Value`) = 1, leaves(`Array n T`) = n · leaves(T), and a `Tuple`'s or `Record`'s leaves are the sum of its fields'.
Leaves are numbered row-major: element i of an `Array` occupies leaves [i · leaves(T), (i + 1) · leaves(T)), and field j of a
`Tuple` or `Record` starts after the leaves of fields 0..j−1. A `Record` keeps its declared field order: it is an array of
pairs, which canonical JSON does not reorder. A type is *homogeneous of width w* when all its leaves are w bits wide (§5.1).
`true` and `false` are not integers here or anywhere in the format.

Vectors: `types` (type, leaves, the widths in leaf order).

## 3. Reference sequences

A reference `(s, i, j)` names leaf j of a collection in the enclosing body: s = `"p"` is parameter i of the body, s = `"n"`
is the output of node i. A reference sequence is an ordered list of references, repeats kept, in one of four forms:

| form | JSON | reference k, for 0 ≤ k < count |
|---|---|---|
| affine | `{"affine": [s, i, base, count]}` | `(s, i, base + k)` |
| strided | `{"strided": [s, i, base, inner, outer_stride, inner_stride, count]}`, inner ≥ 1 | `(s, i, base + (k div inner) · outer_stride + (k mod inner) · inner_stride)` |
| explicit | `{"explicit": [[s, i, j], ...]}` | the k-th triple |
| concat | `{"concat": [seq, ...]}` | the parts' references, in order |

A strided sequence with both strides 0 is one leaf referenced `count` times: one produced value broadcast, which is how
writers normalize `[z] * n`. A zero `inner_stride` alone repeats each leaf of a row `inner` times. A sequence passed as a typed
argument has exactly as many references as its type has leaves.

**Members of a batch argument.** For an argument `a` of type T mapped over axis `ax` (§4.4), member m's references are:

- ax = null: all of `a` (the argument is shared by every member);
- ax = 0, T = `Array n E`: references [m · leaves(E), (m + 1) · leaves(E));
- ax = d ≥ 1, T = `Array n E`: the concatenation, over i = 0..n−1, of member m along axis d − 1 of element i.

Equivalently, the references whose index along array axis `ax` is m, in row-major order of the other indices. The member's
type is T without that axis.

Vectors: `refs` (sequence, references), `take` (type, sequence, axis, member, references).

## 4. The descriptor, schema v1

### 4.1 Top level

~~~text
{"schema": "verity-ir/descriptor/v1",
 "root": <id>, "program_fn": <id>,
 "definitions": {<id>: <definition>, ...},
 "callees": [<id>, ...], "types": [<type>, ...],
 "annotations": {...}}
~~~

- `root`: the key in `definitions` of the root (§5).
- `program_fn`: the descriptor id of the program function, which the root inlines. It is part of the identity, and the
  functions its binding record names are definitions too.
- `definitions`: exactly the definitions reachable from `root` (node callees and the functions binding records name,
  transitively) and from `program_fn`'s record.
- `callees`, `types`: the interning tables. A node's `fn` indexes `callees`; a node's `out` and each argument's `type` index
  `types`.

### 4.2 Definitions

~~~text
{"form": "primitive", "id": "Name_vN", "params": [[name, <type>], ...], "ret": ["Value", w]}
{"form": "composite", "id": "Name_vN", "bindings": {<static>: <value>, ...},
 "body": {"params": [[name, <type>], ...], "ret_type": <type>, "nodes": [<node>, ...], "ret": <seq>}}
~~~

A primitive is one gate producing one `Value<w>`. Its evaluator is code, found by id in a registry, and the decoder refuses a
primitive whose `params` or `ret` differ from the registry's; an `Input<w>_v1` or `RegisteredInput<w>_v1` id always denotes
the IR's input primitive of that binding time (§5.2). A body's `params` and `ret_type` are inline types, not interned.

### 4.3 Descriptor ids and binding records

- A primitive, or a composite without static parameters: `Name_vN`, N in decimal without leading zeros. It is also the
  definition's key and its `id`.
- A composite with static parameters: the key is `Name_vN{k1=j1,...,km=jm}`, in the definition's static-parameter order,
  where `ji` is the canonical JSON (§1) of static `ki`'s value; its `id` is `Name_vN`.
- A static value is written as: an integer, string, boolean or null, itself; a list or tuple, an array; a function,
  `{"fn": <descriptor id>}`; a float, `{"f64": <Python float.hex()>}` (1.5 is `"0x1.8000000000000p+0"`); a dict,
  `{"dict": {k: <value>, ...}}`.
- `bindings` holds the same values keyed by static name. Canonical JSON sorts its keys, so the static order is read from the
  key.
- The root's key is `Root[<readable id of the program function>]_v1`. A readable id abbreviates dict statics to a 12-digit hex
  hash (the first 12 hex digits of SHA-512), so a reader treats it as opaque; it is well formed like any static-free composite's.

Example (the descriptor vector): `TRow_v1{N=4,OP={"fn":"TAdd32_v1"},C={"f64":"0x1.8000000000000p+0"},D={"dict":{"a":[1,"x"],"k":2}}}`.

### 4.4 Nodes

~~~text
{"form": "primitive" | "call", "fn": c, "args": [{"type": t, "refs": <seq>}, ...], "out": t}
{"form": "batch", "fn": c, "args": [...], "out": t, "axes": [null | d, ...], "n": n}
{"form": "scan",  "fn": c, "args": [...], "out": t, "n": n, "n_xs": k}
~~~

A node has exactly the keys of its form. Its argument sequences are body-scoped (§3): `p` names the body's parameters and `n`
the body's nodes. Writers name only nodes before the reading node (`defs.Builder` cannot do otherwise), and the canonical layout
relies on it (§6). A descriptor whose node names itself or a later node still decodes, so that checkers can name the defect;
every partition query refuses it (`partition_object.order_violation`, §8). The body's `ret` may name any node.

- **primitive**: the callee is a primitive; the arguments are its parameters, in order; `out` is its return type.
- **call**: the callee is a composite; likewise, with `out` its `ret_type`.
- **batch**: the callee is either. `axes[i]` is null (argument i is shared by every member) or the array axis mapped over
  members, and the member type of argument i (§3) is the callee's parameter i. `n` is the common length of the mapped axes, or
  any n ≥ 0 when none is mapped. `out` = `Array<n, ret>`.
- **scan**: the callee is a composite `Step(carry, x1..xk, s1..sm) -> Tuple(carry, Y)`, k = `n_xs`. The arguments are
  `(init, xs1..xsk, shared1..sharedm)`: `init` has the carry type, `xsj` is `Array<n, type(xj)>` and `sharedj` has the type of
  `sj`. `n` is the common length of the xs, or any n ≥ 0 when k = 0. `out` = `Tuple(carry, Array<n, Y>)`.

### 4.5 Flat concatenation and the alias

Readers: a concat part that is itself a concat is spliced in, so a decoded sequence is flat. `{"alias": [k, a]}`, alone or as
a concat part, stands for the whole reference sequence of argument a of node k of the same body, where node k comes before
the node being read (for `ret`, any node).

Writers (`codec._BodyEncoder`) take a body's sequences in order (node 0's arguments in order, node 1's, …, then `ret`):

1. A sequence that is not a concatenation (affine, strided, explicit) is written as itself.
2. A concatenation is flattened into its parts p1..pN (nested concatenations spliced, depth first).
3. For m = N, N − 1, …, max(2, N − 8) (`ALIAS_WINDOW` = 8): if p1..pm is exactly the part list recorded for an argument
   (k, a) of an earlier node (k below the node being written; for `ret`, any node), write `{"alias": [k, a]}` followed by
   p(m+1)..pN, and stop. Parts compare by form and fields. A list recorded by an earlier argument of the node being written
   counts as not recorded: a reader resolves an alias against the nodes before the one it reads.
4. Otherwise write p1..pN.
5. The result is `{"concat": [...]}`, or its only element when it has one.
6. A node argument with N ≥ 2 parts is then recorded under its full part list, unless that list is already recorded (the
   first argument keeps it). `ret` is not recorded.

Vectors: `alias_writer` (label, the argument or null for `ret`, the JSON written), `alias_reader`. The descriptor vector's
node 12 is `{"concat": [{"alias": [11, 0]}, ...]}`.

### 4.6 The interning tables, and what the decoder refuses

Writers fill `callees` and `types` in first-seen order of a post-order walk from the root. A composite first visits its nodes'
callees, in node order, then the functions its binding record names, in record order. It is then encoded, adding per node
its callee, each argument's type and its output type. A primitive adds nothing when visited. The walk then visits the
functions `program_fn`'s record names. Writers' tables hold distinct entries, types compared by canonical JSON. Readers accept
any tables whose entries are valid; the order matters only to reproduce a writer's bytes.

`decode_program` refuses, with `ValueError` (`KeyError` for an id without a definition, or a primitive the registry does not
have):

- a descriptor outside the canonical JSON domain (§1);
- a primitive keyed other than by its `id`; a composite whose `id` is not `Name_vN`, or whose key is not its `id` and binding
  record (§4.3);
- a primitive id that a family materialised by id matches only in part, or spells otherwise than its member's id (§5.2, §9);
- a root that is not a composite without parameters;
- an input node (§5.2) in a definition other than the root;
- a node form other than the four, or keys other than its form's;
- a primitive node whose callee is a composite, or a call or scan node whose callee is a primitive;
- a batch whose axes are not null or naturals or whose `n` is not a natural; a scan whose `n` or `n_xs` is not a natural, or
  whose `n_xs` is not below its number of arguments;
- a table index outside its table, negative indices included;
- a reference that names no leaf: a space other than `p` and `n`, a parameter or node that does not exist, a leaf outside its
  collection (a node's is its declared `out`; for a strided sequence, the extremes of its pattern), or a non-integer field;
- an alias that does not name an argument of an earlier node;
- a sequence whose length is not its type's leaf count;
- a node whose arguments do not type-check against its callee as in §4.4, or whose declared `out` is not the type §4.4
  computes (the decoder rebuilds every node through `defs.Builder`, with the checks a program is built with);
- a type that is malformed, or whose width or length is a boolean;
- a body whose `ret` does not have its `ret_type`.

Vectors: `decoder_rejects` (label, the refusal), each a one-field mutation of the descriptor vector (`REJECT` in the test).

## 5. Programs and input nodes

### 5.1 The root

A Program's root is a static-free composite without parameters. Its body starts with the input nodes, then inlines the program
function's body, with the input nodes' outputs standing for its parameters, and its `ret` names the program's (designated)
outputs. Each parameter of the program function has a binding time: *run*, bound per request, or *registration*, bound when
the program is registered (`Program(fn, registered=names)` names the registered parameters; every other one is run). The
input nodes are emitted per parameter, in parameter order, by lowering its type T, with `In` = `Input` for a run parameter and
`RegisteredInput` for a registered one:

- T homogeneous of width w, with at least one leaf: one node
  `{"form": "batch", "fn": In<w>_v1, "args": [], "axes": [], "n": leaves(T), "out": Array<leaves(T), Value<w>>}`;
- T without leaves: nothing;
- otherwise a `Record` or `Tuple` lowers each field in order, and an `Array` each element in order.

`Record(h: Array<2, Value<16>>, b: Value<32>)` gives two nodes, `Input16_v1` with n = 2 and `Input32_v1` with n = 1;
`Array<2, Tuple(Value<16>, Value<32>)>` gives four. The binding times change only the input nodes' callees: the layout, the
gates and the Calls are those of the same program with every parameter run, and the descriptor, so both digests, bind which
parameters are registered.

### 5.2 Input nodes are recognized by id

A node is an Input node (a run input) exactly when its callee's id matches `^Input([1-9][0-9]*)_v1$` (`program.INPUT_ID`),
and a registered-input node exactly when it matches `^RegisteredInput([1-9][0-9]*)_v1$` (`program.REGISTERED_ID`); an *input
node* is either (`program.is_input`, `program.input_id`). The whole id matches (`re.fullmatch`), and w is in canonical decimal,
so `Input016_v1`, `Input0_v1`, `Input16_v2`, `Input16` and `Input16_v1\n` are not Input ids, nor are `RegisteredInput016_v1`,
`RegisteredInput16_v2` and `RegisteredInput16_v1\n` registered-input ids.
Each input primitive of width w has no parameters and returns `Value<w>`: a gate whose value is prescribed (asserted equal to
the claimed input), not computed, and never held by the program. The decoder materializes it from the id, ahead of any
registry entry. An input node appears only in the root's body: `Program` and the decoder (§4.6) refuse one in any other
definition, where no parameter would bind it.

The Program's Calls (`partition_object.calls`, the units of `Q_word`) are the root body's nodes that are not input nodes, in
body order.

Vectors: `input_ids` (id, whether it is an Input id), `input_lowering` (type, its Input nodes), `program` (the vector's root
key, Input nodes, Input gate count and Calls); `binding_ids` (id, its binding time and width, or null), `registered_program`
(the vector program with `r` registered: its descriptor, digests, input nodes and Calls).

### 5.3 What a Program holds

An operation's own definition is gates: the constants of the floating-point formats, the round constants of SHA-512 and
Keccak-f, and every measured MUFU table (exp2, rcp, sqrt, rsq and tanh). Everything a program or a request brings is a value: a program's weights, constants and tables (a norm's
epsilon, the attention scale, a softcap, `gelu_tanh_bf16`, `tanh_rn`) are registered inputs, and a request's data and parameters
(temperature, top-p, the Philox positions) are run inputs. A constant primitive (`Const<w>[0x…]_v1`) is therefore an operation's
own bits, and a Program holds no data.

`constants.program_data(root, operation, library)` lists the data a Program still holds, by the Definition that holds it:

- every constant of the root body;
- a constant inside a Definition, unless an enclosing Definition is an operation. A Definition reached first under an
  operation is not visited there, so a later path to it outside any operation lists its constants;
- a table a primitive reads by index (its evaluator, or the circuit that proves it). Such a table is no gate of the program,
  so the primitive declares it (`PrimitiveDefinition.tables`) as a `library` object or a `registered` input that the
  primitive still holds. Outside an operation, a `registered` table is always listed, and a `library` one unless the library
  lists it.

`verity.ml.circuits` names no operations and no library tables, so the guard is `verity.ml.circuits.operations.program_data`. It passes in the
pinned list `verity/operations` v2 (`verity.ml.circuits.operations.OPERATIONS`, `digest()`) and the tables of `verity/library` v2. An
operation is a listed composite as the registry holds it, bound only to functions that hold no data, so a decoded Program's
composites are no operations and their constants are listed.

The guard therefore runs at registration, on the producer's registry. A verifier that receives a descriptor cannot run it: a
decoded Program matches no operation, since the match is the registry's own Definition object, not a descriptor. A
verifier-side check needs operations matched by descriptor digest, under a new `verity/operations` version; that is a later
step.

No statement or table claims that a Program's values are always hidden until the guard returns {} for it.
`Program(fn, strict=guard)` refuses a Program that holds data; it is off by default until programs move their data to inputs.
The format does not refuse a program that holds data: a descriptor's bytes and digests mean what they meant, and programs move
to registered inputs under new ids.

## 6. The canonical layout

Gates are numbered by laying bodies out in node order, recursively.

- **Gate counts.** A primitive is 1 gate, and a composite's gates are its body's last offset. A node's gates: primitive 1,
  call gates(callee), batch and scan n · gates(callee).
- **Member counts** (`parts.n_members`): 1 for a primitive or call node, n for a batch (members) or a scan (iterations).
- **Offsets.** In a body, offsets[0] = 0 and offsets[k + 1] = offsets[k] + gates(node k). Member (or iteration) m of node k
  occupies [offsets[k] + m · gates(callee), offsets[k] + (m + 1) · gates(callee)), members consecutively in index order; a
  batch of a primitive is one gate per member.
- **Activations** (`layout.Scope`): a composite at an absolute gate offset, reached from node k, member m, of its parent
  activation. The root is the root composite at offset 0; child (k, m) of an activation at offset o is at
  o + offsets[k] + m · gates(callee).
- **Locating gate g** (`Circuit.gate`): in an activation at offset o, node k is the one with
  offsets[k] ≤ g − o < offsets[k + 1], and member m = ⌊(g − o − offsets[k]) / gates(callee)⌋ (0 for primitive and call
  nodes). A primitive callee is the gate; otherwise descend into child (k, m).
- **A gate's operands** are its primitive's arguments resolved (§7) in the gate's activation: for a primitive node, every
  reference of every argument in order; for member m of a batch of a primitive, per argument in order, member m's references
  (§3). Order and repeats are kept. Input gates of either binding time and constants have none.
- **Topological order.** When every node names only earlier nodes (§4.4), every gate's operands are gates before it: a callee's
  gates precede the node reading its outputs, a scan iteration reads the iteration before it, and a parameter resolves into the
  parent before the activation. Every partition query checks the node-level rule before it evaluates (§8).

Vectors: `layout.nodes` (each root node: index, form, callee, members, gates, offset), `layout.offsets` (every composite's
offsets), `layout.gates` (every gate: index, primitive, operands, activation path), `layout.root_outputs`,
`layout.gates_total`.

## 7. Scope resolution

`resolve(activation, (s, i, j))` (`layout.resolve_gate`) is the gate a body-scoped reference denotes. It repeats two steps
until it reaches a primitive.

**Up through a parameter** (s = `p`). The activation is child (k, m) of an activation P, and N is node k of P's body:

- N a call: continue in P with reference j of N's argument i;
- N a batch: continue in P with reference j of member m of argument i (§3);
- N a scan, i = 0 (the carry): for m = 0, reference j of `init`, in P; for m > 0, reference j of the step's `ret`, in
  iteration m − 1;
- N a scan, 1 ≤ i ≤ `n_xs`: reference m · leaves(elem) + j of argument i, in P;
- N a scan, i > `n_xs` (shared): reference j of argument i, in P.

The root has no parameters.

**Down through a node output** (s = `n`). Node i of the activation's body starts at absolute offset o:

- a primitive node: the gate is o;
- a call: reference j of the callee's `ret`, in child (i, 0);
- a batch: (m, r) = divmod(j, leaves(ret(callee))); a primitive callee is gate o + m, otherwise reference r of the callee's
  `ret`, in child (i, m);
- a scan, out = `Tuple(carry, Array<n, Y>)`, c = leaves(carry): for j < c the final carry, reference j of `ret` in iteration
  n − 1; otherwise (m, r) = divmod(j − c, leaves(Y)) and reference c + r of `ret` in iteration m.

A scan with n = 0 has no gates, and reading its output is not defined; programs do not.

Vectors: `layout.resolve` (label, the activation as (node, member) steps from the root, reference, gate), one case per rule
above, on the descriptor vector.

## 8. `Q_word` v1 conventions

`Q_word` v1 (`verity.ml.circuits.cut`, `partition_object`) reads a Program through §1–§7 and adds these conventions. They are pinned:
changing any of them is a new version of the query.

- **Applicability**: the program is topological: every node of every body names only its body's parameters and earlier nodes,
  a body's `ret` any of its nodes (§4.4). `partition_object.evaluate` checks it once, before any query (`order_violation`), so
  every query refuses the same programs. The Calls are the root's non-input nodes (§5.2), each of form `call` or `primitive`.
  Otherwise the query is inapplicable: `evaluate` raises `cut.QueryInapplicable` and `partition_object.verify` refuses with
  `query-inapplicable` (the definition, node, argument and reference). `cut.CallGraph` keeps a gate-level check (an operand
  inside the Call comes before its gate) as a backstop.
- **Wiring** (`cut.WIRING`): the primitive ids `Bf16ToF32_v1`, `F32BitsShl23_v1` and `F32Fabs_v1`. Like constants
  (primitives without parameters) their gates are structure: computed in no unit, readable by any, outside the committed set. The list is
  a set of ids, not derived from the descriptor; that each only rearranges bits is checked on the registry by a vLLM test.
- **Wide gates**: a gate whose primitive has more than `cut.WIDE_FAN_IN` = 4096 parameter leaves, so more than 4096
  operands. The layout's operands are the same for every gate (the reference returns a lazy sequence for a wide gate, with the
  same elements). In a Call's graph (`cut.CallGraph`) a wide gate:
  - is external, as a gate reading outside the Call;
  - lists only its operands inside the Call, in operand order; its operands outside the Call are neither listed nor numbered
    as inputs;
  - takes its own recompute token, so it is never a recompute of an earlier gate and no later gate is a recompute of it;
  - is never folded into a constant.

  This is v1 behaviour, not a property of the circuit, and should go when the query is next revised. `cut.WIDE_FAN_IN` is the
  query's own constant, not the layout's lazy-operand threshold (`layout.WIDE_FAN_IN`, which has no effect on operands).

`WIRING` and `WIDE_FAN_IN` are pinned as data in `verity/ml/circuits/tests/qword_vectors.json`, with vectors for every step of the algorithm, every
refusal and a seeded random corpus.

Vectors: `forward_reference` (a root node reading a later node decodes, gate 11 reading gate 39, and `evaluate` refuses it; a
Call whose gate reads a later gate of the Call decodes, and `evaluate` and the `CallGraph` backstop refuse it); `qword_vectors.json`
→ `order_refusals` for `verify`; `wiring`; `wide`: the descriptor of a Call with two identical wide gates, their operands as runs, and the Call's graph
beside the same Call over a 4-leaf primitive (the narrow Call has the recompute pair `[2, 1]` and 5 inputs; the wide one has
none and 1).

## 9. The Boolean profile

A *Boolean* Definition has only `Value<1>` leaves in its signature, and every primitive it reaches is one of these
(`verity.ml.circuits.boolean`, `is_boolean`); a Boolean program is a program whose function is Boolean. It needs no
new syntax: §1–§8 apply unchanged, and its digests are computed the same way.

| id | parameters | returns | value |
|---|---|---|---|
| `And<n>_v1`, n ≥ 2 | `x: Array<n, Value<1>>` | `Value<1>` | x[0] AND … AND x[n−1] |
| `Xor<n>_v1`, n ≥ 2 | `x: Array<n, Value<1>>` | `Value<1>` | x[0] XOR … XOR x[n−1] |
| `Not_v1` | `a: Value<1>` | `Value<1>` | NOT a |
| `Const1[0x0]_v1`, `Const1[0x1]_v1` | none | `Value<1>` | 0, 1 |
| `Input1_v1`, `RegisteredInput1_v1` | none | `Value<1>` | prescribed (§5.2) |

The three gates are fixed by this format: a registry holds exactly these definitions under these ids and refuses another under
the same id. AND and XOR are one family each, with a member per number of inputs n (decimal, no leading zeros), which a decoder
materialises from its id like the constant family; `And2_v1` and `Xor2_v1` are the two-input gates. A gate with more than
`WIDE_FAN_IN` inputs is a wide gate under `Q_word` v1 (§8). The constant bits are the 1-bit members of the constant family (`Const<w>[0x<hex>]_v1`): primitives without
parameters, so structure under `Q_word` v1 (§8), computed in no unit.

**Ids of the families.** A family member has one id, which the decoder materialises (§4.6) and nothing else does: the whole id
matches its family's pattern, and the member it denotes carries exactly that id. `And<n>_v1` and `Xor<n>_v1` match
`^(And|Xor)([2-9]|[1-9][0-9]+)_v1$`. The constant of width w ≥ 1 and value v, 0 ≤ v < 2^w, is
`Const<w>[0x<h>]_v1` (`verity.ml.circuits.ops.prims.const`), with w in decimal without leading zeros and h the lowercase hex of
v zero-padded to ⌊w/4⌋ digits, at least one (Python's `f"{v:0{w // 4}x}"`): `Const32[0x3f800000]_v1`, `Const16[0x3f80]_v1`,
`Const5[0x5]_v1`, `Const5[0x1f]_v1`, `Const1[0x0]_v1`. So `And02_v1`, `And1_v1`, `And2_v1\n`, `Const32[0x0]_v1`,
`Const32[0x03f800000]_v1`, `Const32[0x3F800000]_v1`, `Const5[0x05]_v1`, `Const032[0x3f800000]_v1`, `Const0[0x0]_v1` and
`Const1[0x00]_v1` denote nothing (`KeyError`), and `Const1[0x2]_v1` and `Const8[0x100]_v1`, whose values are 2^w or more,
are refused (`ValueError`).

**Elements.** An *element* of a Boolean type is an innermost `Array<w, Value<1>>`, whose leaf j is bit j of a w-bit word (least
significant first), or a `Value<1>` that is not inside one. `elements(T)` lists the element widths in leaf order;
`word_type(T)` replaces every element by `Value<w>`. The *word view* of a Boolean Definition is its signature under
`word_type`: the signature of the word Definition it computes. A port's number format is an annotation, outside every digest;
the word view does not read it.

Vectors (`verity/ml/circuits/tests/boolean_vectors.json`, written by `python verity/ml/circuits/tests/test_boolean.py --write`): `gates` (id,
kind, parameters, return type, truth table), `constants`, `gate_ids` (id, its kind and inputs, or null), `constant_ids` (id,
its width and value, or null), `word_view` (type, element widths, word type), and `program`: a 2-bit
adder built from two full-adder calls, a batch of `Xor2_v1` over axis 1, a constant, a `Not_v1` and an `Xor3_v1`, with its
descriptor, digest, gate table and truth table on words.

## 10. `Q_word` v2

`Q_word` v2 (`partition_object.QUERIES`, the parameters `X` and `W` checked as v1's) reads a Program as v1 does, with §8's
conventions and v1's Calls, units, owners and committed sets. It differs from v1 in one rule of the cut check: a value computed in
two units (a recomputed pair whose gates have two owners) is reported in `verify`'s detail (`recomputed_across`), not refused
(`gate-recomputed`). Every other code and the width rule are v1's, and `validate_unit_cut` and `cut.check_cut` refuse by default.
v2's units may compute a common value, so a count of v2 units or gates bounds wrong work from above but is not distinct work, and a
consumer that credits work (PoUW) does not take `Q_word` v2.

Vectors: `qword_vectors.json` → `v2`: v1's recompute-free evaluation and every verify refusal at v2, and the program v1 refuses
only for `gate-recomputed` (two members of one Call each computing `x0 + x1`), which v2 accepts and reports.

## 11. `Q_call` v1

`Q_call` v1 (`partition_object.QUERIES`, `call_query`; the algorithm in `verity.ml.circuits.cut`'s docstring, `cut.evaluate_call`) is a
query of its own, `{"name": "Q_call", "version": 1, "params": {"X": X}}`, with one parameter: X, an integer in 1..2^16 (32 is the
agreed value). It reads a Program through §1–§7. A *proof unit* is a set S of gates, with in(S) and out(S) as `verity.ml.circuits.parts`
defines them, counted in bits at any gate width. The one rule is w_out(S) ≤ X; inputs are unbounded. This section is a delta
on §8 and §10.

What carries over from §8 and §10:

- **Applicability.** §8's, with the same refusal: the program is topological, and every Call is a `call` or `primitive` node.
  A root `batch` or `scan` is refused with `why: "root-form"`.
- **Calls.** §8's Calls (the root's non-input nodes, in body order). Each is cut as its Definition's activation in a one-Call
  Program, every returned leaf a designated output, so activations of one Definition share their cut.
- **Recomputes.** §10's report: a value computed in two units is `verify`'s `detail["recomputed_across"]`, never a code. The
  §10 caveat holds: a count of units bounds wrong work but is not distinct work, so a consumer that credits work (PoUW) does
  not take `Q_call` v1.

What changes:

- **Inputs.** A gate whose primitive has no parameters is an input: input gates of either binding time (§5.2) and constants (an
  operation's own bits, §5.3), of any width. It is in no unit and outside the committed set: like an input gate, a constant is
  bound outside the partition, and the cut never reads its value. Every other gate is *computing* and is in exactly one unit.
- **No structure rules.** §8's wiring list (`cut.WIRING`), its constant folding (a gate computed from constants alone) and its
  wide-gate convention (`cut.WIDE_FAN_IN`) do not apply. Every operand of every gate is read.
- **Wide gates.** A computing gate whose declared width exceeds X makes the query inapplicable, with `why: "gate-too-wide"` and
  the Call, Definition, local gate, primitive, width and X. Nothing else is refused for width.
- **Pieces.** A piece is a gate interval of the Call's activation (§6): the Call itself, a body node, or a member or iteration of a
  batch or scan node. Its outputs are its computing gates that a later gate of the Call reads, or that the Call returns. The layout
  is topological, so no gate before the piece reads inside it, and these outputs are out(S) for S the piece's computing gates.
- **The cut**, depth first in layout order from the Call:
  - a piece without computing gates is no unit;
  - a piece whose outputs total at most X bits is one unit;
  - otherwise its children are the pieces: for a composite activation (the Call, a call node's callee, a composite member) its
    body's nodes, and for a batch or scan node its members or iterations.

  A primitive piece is one gate of at most X bits, so it always fits.
- **Units.** Each unit is an interval `[lo, hi)` of the Call's local gates with a path of steps from the Call's activation: `[k]` is
  body node k, and `[k, m]` is member m of node k. Units are numbered by Call, then in layout order. `CallPartition.locate`
  gives the path, and `gates` gives the root interval. Input gates and constants inside an interval are in no unit.
- **The committed set.** A computing gate is committed when a unit other than its own reads it, or when the Call returns it. A
  unit's committed bits are its w_out. `verify` derives it from the reads (`cut.check_call_cut`) and checks the invariant
  (`validate_unit_cut`, whose structure is the constants: gates any unit may read, bound outside the partition) and w_out ≤ X
  (`unit-too-wide`). Served commits are numbered by the
  Call's computing gates in layout order (`cut.CallGates.keep`) and checked as under §8.
- **Recompute keys.** A key is (primitive id, operand tokens). A parameter leaf's token is interned from `("i", leaf)` and a
  constant's from `("c", primitive id)`, so equal constants are equal operands. A program's data reaches a Call only through its
  parameters, from registered or run input gates, so it is compared by leaf, never by value: two gates that read one input
  leaf alike are a pair, and recomputes from a value that was a constant become recomputes from the input that carries it. A computing gate whose key an earlier computing
  gate had is a recomputed pair and takes that gate's token.

Rules the IR left open, pinned here:

- **A pass-through output** is a returned leaf that resolves to a parameter of the Call, or to a constant, rather than to a
  computing gate of it. It is no output of any of the Call's units. The value is the Call's input, or a constant, bound outside
  the partition like an input.
- **A scan's carry** follows from the reads: an iteration's returned carry is an output of that iteration when the next iteration
  reads it, and of the Call when it is the final carry. A carry an iteration passes through unchanged resolves to the scan's
  `init` (§7), so it is a pass-through.
- **A call node of no computing gates** (one that only rearranges its parameters, or returns constants) is no unit.
- **A unit with no outputs** (a dead gate) is still a unit: every computing gate is in exactly one.
- **An unread parameter leaf** is in the input unit and is not committed.

Vectors: `qcall_vectors.json`, written by `python verity/ml/circuits/tests/test_qcall_vectors.py --write`. They cover a word
program and a Boolean program at two values of X. The word program has a Call that fits; one that splits into sub-Calls two levels
down; a batch of a composite and of a primitive; a scan with a chained carry and one with a passed-through carry; pass-through
outputs; a constant read by two units; a constant-derived gate committed across units; a 64-bit input and constant; a recompute
across units and one within a unit; former wiring; a dead gate; and a gate of 4,097 operands. The Boolean program has a 32-bit
Call, two 32-bit sub-Calls, constant bits, a 33-bit ripple-carry adder and a 5,000-input AND. The refusals are `gate-too-wide` (at
X = 16 and for a 64-bit gate), `root-form` (batch, scan), not topological, another program, every served-commit check, and the
objects `validate` refuses.
