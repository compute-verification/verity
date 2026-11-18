# scratch (lean/WIRED): which root node, argument, template and part `row_runs` (Lean `rowRuns`) refuses, and which case
import gzip, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import inference_only_record_reference as R
from verity.circuits import codec
from verity.circuits.program import is_input
from verity.circuits.refs import Affine, Explicit, Grid, Strided

p = Path(sys.argv[2])
text = (gzip.open(p, "rt") if p.suffix == ".gz" else p.open()).read()
program = R.reference.desc_program(text)
nodes = program.root.body.nodes
edge = R.edge_of(nodes)


def name(k):
    fn = nodes[k].fn
    return f"{getattr(fn, 'name', '?')} [{codec.descriptor_id(fn)[:12]}] form={nodes[k].form}"


def src(s, i):
    if s != "n" or not (0 <= i < len(nodes)):
        return f"{s}{i}"
    return f"node {i} ({'input' if is_input(nodes[i].fn) else 'computed'}, {nodes[i].form})"


def case(part):
    if isinstance(part, Grid):
        kinds = {("computed" if (s == "n" and 0 <= i < len(nodes) and not is_input(nodes[i].fn)) else "input") for s, i, *_ in part.lanes}
        return ("a row half its own input and half an edge (grid lanes: " + "/".join(sorted(kinds)) + ")"
                if len(kinds) > 1 else "a grid over input nodes (no run)")
    if isinstance(part, Explicit):
        return "an explicit part over input leaves (no run)"
    if isinstance(part, Strided):
        return ("a strided part that is not a shared run (needs outer_stride 0, inner_stride 1, base>=0, count % inner == 0, "
                "over a primitive or batch input node)")
    if isinstance(part, Affine):
        return "an affine part over something that is not a primitive or batch input node"
    return f"a part of kind {type(part).__name__}"


shown, refusals = 0, 0
runs = []
for k, node in enumerate(nodes):
    if is_input(node.fn):
        continue
    if node.form == "batch" and None in node.axes:
        refusals += 1
        if shown < 8:
            shown += 1
            print(f"REFUSE node {k} {name(k)}: a batch maps an argument over no axis: axes={node.axes}")
    sig = node.fn.signature()[0]
    for a_i, a in enumerate(node.args):
        for part in R._parts(a.refs):
            if edge(part):
                continue
            r = R.part_run(nodes, part)
            if r is None:
                refusals += 1
                if shown < 8:
                    shown += 1
                    where = ", ".join(src(s, i) for s, i, *_ in part.lanes) if isinstance(part, Grid) else (
                        src(part.space, part.idx) if hasattr(part, "idx") else "")
                    print(f"REFUSE node {k} {name(k)} argument {a_i} ({sig[a_i][0] if a_i < len(sig) else '?'}): "
                          f"{case(part)}\n   part as written: {part!r}\n   reads: {where}")
            else:
                runs.append((r, k, a_i, part))
ordered = sorted((x for x in runs if x[0][0][2] > 0), key=lambda x: (x[0][0][0], x[0][0][1]))
for a, b in zip(ordered, ordered[1:]):
    (ra, _), (rb, _) = a[0], b[0]
    if ra != rb and ra[0] == rb[0] and ra[1] + ra[2] > rb[1]:
        refusals += 1
        if shown < 8:
            shown += 1
            print(f"REFUSE overlap, not one shared run, over {src('n', ra[0])}: run {ra} (node {a[1]} {name(a[1])} arg {a[2]}, "
                  f"part {a[3]!r}) and run {rb} (node {b[1]} {name(b[1])} arg {b[2]}, part {b[3]!r})")
print(f"OWN_DIAG {p.name}: {refusals} refusals; row_runs is {'None' if R.row_runs(program) is None else 'ok'}", flush=True)
