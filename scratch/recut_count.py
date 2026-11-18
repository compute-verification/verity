"""The reads at another width (`refs.recut`: an input node read in leaves of 1/q its width) of one wired Program, by
Definition, input node and reading argument, from the Python decoder's `Body.recut`."""
import collections, gzip, json, sys
sys.path.insert(0, ".")
import inference_only_record_reference as R
from verity.circuits.refs import Affine, Concat, Explicit, Grid, Strided
text = (gzip.open if sys.argv[1].endswith(".gz") else open)(sys.argv[1], "rt").read()
program = R.reference.desc_program(text)
def named(r):
    if isinstance(r, Concat):
        for p in r.parts: yield from named(p)
    elif isinstance(r, (Affine, Strided)):
        yield r.space, r.idx
    elif isinstance(r, Explicit):
        for s, i, _ in r.items: yield s, i
    elif isinstance(r, Grid):
        for s, i, *_ in r.lanes: yield s, i
root = program.root
seen = {}
for fn in [*program.defs.values(), root]:
    b = getattr(fn, "body", None)
    if b is None or id(fn) in seen:
        continue
    seen[id(fn)] = 1
    if not b.recut:
        continue
    name = getattr(fn, "name", None) or getattr(fn, "key", None) or str(fn)[:80]
    tag = "ROOT " if fn is root else ""
    print(f"{tag}Definition {str(name)[:140]}: {len(b.recut)} input nodes read recut")
    by = collections.Counter()
    for k, q in sorted(b.recut.items()):
        n = b.nodes[k]
        readers = []
        for j, m in enumerate(b.nodes):
            for a, c in enumerate(m.args):
                if any(s == "n" and i == k for s, i in named(c.refs)):
                    pn = (m.fn.params[a][0] if hasattr(m.fn, "params") and a < len(m.fn.params) else str(a))
                    readers.append((j, getattr(m.fn, "name", str(m.fn))[:60], a, pn))
        if any(s == "n" and i == k for s, i in named(b.ret)):
            readers.append(("ret", "", "", ""))
        by[(getattr(n.fn, "name", str(n.fn))[:60], str(n.out)[:50], q, tuple((r[1], r[3]) for r in readers))] += 1
        if len(by) <= 3 and by[(getattr(n.fn, "name", str(n.fn))[:60], str(n.out)[:50], q, tuple((r[1], r[3]) for r in readers))] == 1:
            print(f"   e.g. input node {k} ({getattr(n.fn, 'name', n.fn)}, out {str(n.out)[:60]}), q={q}, read by {readers[:4]}")
    for (callee, out, q, rd), c in by.most_common():
        print(f"   {c} x input {callee} out {out} at 1/{q} its width, read by {list(rd)[:4]}")
print(f"ROOT body reads recut: {bool(root.body.recut)} ({len(root.body.recut)} input nodes)")
