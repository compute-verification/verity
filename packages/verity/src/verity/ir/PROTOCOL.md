# `verity-ir/descriptor/v1`: the program format

This is the byte format of a Program and the rules that turn it into a circuit: from the canonical bytes of a descriptor, a
reader obtains every gate, its primitive and its operand gates, and nothing else is needed. It is written for a second
implementation (the Lean port of `Q_word` first). It describes the bytes the codec already writes and moves no digest.

- Reference: `verity.ir.codec` (`encode_program`, `decode_program`, `canonical_json`, `jcs_domain_error`, `program_digest`),
  with `types`, `refs`, `defs` (`Node`, `Body`, `Builder`), `program` (the root, `INPUT_ID`), `layout` (`Circuit`, `resolve`),
  `parts.n_members` and, for §8, `cut`.
- Vectors: `packages/verity/tests/ir/format_vectors.json`, written by
  `python packages/verity/tests/ir/test_format_spec.py --write`.
- Conformance: `packages/verity/tests/ir/test_format_spec.py` checks the reference against every vector, and restates
  canonical JSON (as RFC 8785), leaf order, reference sequences and the interning order independently of it. Each section
  names its vector keys. `tests/ir/partition_vectors.json` pins `Q_word` v1 on top of this format.
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

A stored descriptor file need not be canonical (builds write it with spaces). The digests are over the canonical bytes of the
descriptor without `annotations`:

- `program_digest`: hex SHA-256 of those bytes, the Build's program digest;
- `partition_object.program_sha512`: hex SHA-512 of the same bytes, the `program` field of `verity/partition/v1`.

`annotations` never enter a digest and carry no meaning in this format.

Vectors: `canonical_json.in_domain` (value, bytes), `canonical_json.outside` (value, Python's bytes, the JCS bytes, the
refusal), `program_digest_sha256`, `program_sha512`.

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
primitive whose `params` or `ret` differ from the registry's; an `Input<w>_v1` id always denotes the IR's Input primitive
(§5.2). A body's `params` and `ret_type` are inline types, not interned.

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
  hash, so a reader treats it as opaque; it is well formed like any static-free composite's.

Example (the descriptor vector): `TRow_v1{N=4,OP={"fn":"TAdd32_v1"},C={"f64":"0x1.8000000000000p+0"},D={"dict":{"a":[1,"x"],"k":2}}}`.

### 4.4 Nodes

~~~text
{"form": "primitive" | "call", "fn": c, "args": [{"type": t, "refs": <seq>}, ...], "out": t}
{"form": "batch", "fn": c, "args": [...], "out": t, "axes": [null | d, ...], "n": n}
{"form": "scan",  "fn": c, "args": [...], "out": t, "n": n, "n_xs": k}
~~~

A node has exactly the keys of its form. Its argument sequences are body-scoped (§3): `p` names the body's parameters and `n`
names nodes before it. The body's `ret` may name any node.

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
3. For m = N, N − 1, …, max(2, N − 8) (`ALIAS_WINDOW` = 8): if p1..pm is exactly the part list recorded for an earlier node
   argument (k, a), write `{"alias": [k, a]}` followed by p(m+1)..pN, and stop. Parts compare by form and fields.
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
- a root that is not a composite without parameters;
- a node form other than the four, or keys other than its form's;
- a primitive node whose callee is a composite, or a call or scan node whose callee is a primitive;
- a batch whose axes are not null or naturals or whose `n` is not a natural; a scan whose `n` or `n_xs` is not a natural, or
  whose `n_xs` is not below its number of arguments;
- a table index outside its table, negative indices included;
- a reference that names no leaf: a space other than `p` and `n`, a parameter or node that does not exist or does not come
  before the reading node, a leaf outside its collection (for a strided sequence, the extremes of its pattern), or a
  non-integer field;
- an alias that does not name an argument of an earlier node;
- a sequence whose length is not its type's leaf count;
- a node whose arguments do not type-check against its callee as in §4.4, or whose declared `out` is not the type §4.4
  computes (the decoder rebuilds every node through `defs.Builder`, with the checks a program is built with);
- a type that is malformed, or whose width or length is a boolean;
- a body whose `ret` does not have its `ret_type`.

Vectors: `decoder_rejects` (label, the refusal), each a one-field mutation of the descriptor vector (`REJECT` in the test).

## 5. Programs and Input nodes

### 5.1 The root

A Program's root is a static-free composite without parameters. Its body starts with the Input nodes, then inlines the program
function's body, with the Input nodes' outputs standing for its parameters, and its `ret` names the program's (designated)
outputs. The Input nodes are emitted per parameter of the program function, in parameter order, by lowering its type T:

- T homogeneous of width w, with at least one leaf: one node
  `{"form": "batch", "fn": Input<w>_v1, "args": [], "axes": [], "n": leaves(T), "out": Array<leaves(T), Value<w>>}`;
- T without leaves: nothing;
- otherwise a `Record` or `Tuple` lowers each field in order, and an `Array` each element in order.

`Record(h: Array<2, Value<16>>, b: Value<32>)` gives two nodes, `Input16_v1` with n = 2 and `Input32_v1` with n = 1;
`Array<2, Tuple(Value<16>, Value<32>)>` gives four.

### 5.2 Input nodes are recognized by id

A node is an Input node exactly when its callee's id matches `^Input([1-9][0-9]*)_v1$` (`program.INPUT_ID`,
`program.is_input`): w in canonical decimal, so `Input016_v1`, `Input0_v1`, `Input16_v2` and `Input16` are not Input ids. The
Input primitive of width w has no parameters and returns `Value<w>`: a gate whose value is prescribed (asserted equal to the
claimed input), not computed. The decoder materializes it from the id, ahead of any registry entry.

The Program's Calls (`partition_object.calls`, the units of `Q_word`) are the root body's nodes that are not Input nodes, in
body order.

Vectors: `input_ids` (id, whether it is an Input id), `input_lowering` (type, its Input nodes), `program` (the vector's root
key, Input nodes, Input gate count and Calls).

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
  (§3). Order and repeats are kept. Input gates and constants have none.

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

`Q_word` v1 (`verity.ir.cut`, `partition_object`) reads a Program through §1–§7 and adds these conventions. They are pinned:
changing any of them is a new version of the query.

- **Calls**: the root's non-Input nodes (§5.2), each of form `call` or `primitive`; other forms are refused.
- **Wiring** (`cut.WIRING`): the primitive ids `Bf16ToF32_v1`, `F32BitsShl23_v1` and `F32Fabs_v1`. Like constants
  (primitives without parameters) their gates are structure: computed in no unit, readable by any, never committed. The list is
  a set of ids, not derived from the descriptor; that each only rearranges bits is checked on the registry by a vLLM test.
- **Wide gates**: a gate whose primitive has more than `layout.WIDE_FAN_IN` = 4096 parameter leaves, so more than 4096
  operands. The layout's operands are the same for every gate (the reference returns a lazy sequence for a wide gate, with the
  same elements). In a Call's graph (`cut.CallGraph`) a wide gate:
  - is external, as a gate reading outside the Call;
  - lists only its operands inside the Call, in operand order; its operands outside the Call are neither listed nor numbered
    as inputs;
  - takes its own recompute token, so it is never a recompute of an earlier gate and no later gate is a recompute of it;
  - is never folded into a constant.

  This is v1 behaviour, not a property of the circuit, and should go when the query is next revised.

Vectors: `wiring`; `wide`: the descriptor of a Call with two identical wide gates, their operands as runs, and the Call's graph
beside the same Call over a 4-leaf primitive (the narrow Call has the recompute pair `[2, 1]` and 5 inputs; the wide one has
none and 1).
