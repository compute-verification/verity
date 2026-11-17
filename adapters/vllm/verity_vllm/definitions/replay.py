"""Replay of a recorded flat body over new argument collections: the body builder of a derived Program's function (`frontend.derive`)
and of a fold's (`observe.fold`).  Torch-free, so the fold imports it without the frontend."""
from __future__ import annotations

from typing import Any

from verity.circuits.defs import Body, Builder
from verity.circuits.refs import Affine, Coll, Concat, PartLog, Refs, Strided, concat, prefix_runs, runs


def _scattered(p: Refs) -> bool:
    """A `Strided` whose runs are O(count) leaves or O(count / inner) rows: per-leaf iteration (`inner != 1` with a non-unit
    inner stride), single-leaf rows (`inner == 1` with a non-unit outer stride) or one row repeated (`outer_stride == 0`,
    more than one row). A transposed weight read is the common case: at GLM-5.3 scale one such argument is >1e6 runs, so
    substituting it run by run never finishes; an embedding's backward reads its gradient row once per vocabulary row."""
    if not isinstance(p, Strided) or (p.outer_stride == 0 and p.inner_stride == 0):
        return False
    if p.inner == 1:
        return p.outer_stride not in (0, 1)
    return p.inner_stride != 1 or (p.outer_stride == 0 and p.count > p.inner)


def _rebase(p: Strided, env: dict[tuple[str, int], Any]) -> Strided | None:
    """`p` substituted in O(1): when the target's refs over `p`'s leaf span are one contiguous `Affine` run, every leaf
    maps through the same shift, so the result is `p` itself rebased onto the target. None when the span is not one run
    (the caller falls back to run-by-run substitution)."""
    outer = -(-p.count // p.inner)
    last = p.count - (outer - 1) * p.inner
    leaves = [p.base + q * p.outer_stride + r * p.inner_stride
              for q in ({0, max(outer - 2, 0), outer - 1}) for r in (0, (last if q == outer - 1 else p.inner) - 1)]
    lo, hi = min(leaves), max(leaves) + 1
    t = env[(p.space, p.idx)].slice(lo, hi)
    if isinstance(t, Affine) and t.count == hi - lo:
        return Strided(t.space, t.idx, t.base + (p.base - lo), p.inner, p.outer_stride, p.inner_stride, p.count)
    return None


def _substitute_scattered(refs, env: dict[tuple[str, int], Any]):
    """Partwise substitution when some part of `refs` is a scattered `Strided` that rebases in O(1); None otherwise
    (every other shape takes the run path below). A `Concat` cut from a
    `PartLog` keeps the memoised run path: its parts are attention prefixes, never scattered."""
    if isinstance(refs, Concat):
        if refs.prefix is not None:
            return None
        parts = refs.parts
    elif not _scattered(refs):
        return None
    else:
        parts = (refs,)
    sub = [_rebase(p, env) if _scattered(p) else None for p in parts]
    if sub.count(None) == len(sub):
        return None
    out: list[Refs] = []
    for p, r in zip(parts, sub):
        if r is not None:
            out.append(r)
        else:
            out.extend(env[(space, idx)].slice(lo, hi) for space, idx, lo, hi in runs(p))
    return concat(out)


def substitute(refs, env: dict[tuple[str, int], Any], logs: dict[int, tuple[Any, Any]] | None = None):
    """`refs` with every `(space, idx)` run replaced by the corresponding run of `env[(space, idx)]`.

    `logs` (one per `env`) maps a `PartLog` the flat body cut `refs` from to the log of its substituted shared runs, so the
    runs every longer cut shares are substituted once."""
    fast = _substitute_scattered(refs, env)
    if fast is not None:
        return fast
    rs, log, s = prefix_runs(refs)
    if log is None or logs is None or s == 0:
        parts = [env[(space, idx)].slice(lo, hi) for space, idx, lo, hi in rs]
        return concat(parts) if parts else refs
    hit = logs.get(id(log))
    if hit is None or hit[0] is not log:
        hit = logs[id(log)] = (log, PartLog())
    sub = hit[1]
    if len(sub) < s:
        for space, idx, lo, hi in log.shared_runs(len(sub), s):
            sub.append(env[(space, idx)].slice(lo, hi))
    return sub.cut(s, [env[(space, idx)].slice(lo, hi) for space, idx, lo, hi in rs[s:]])


def replayer(body: Body):
    """A body builder that re-emits a recorded flat body over new argument collections (used when the program
    function is inlined into the root: `Program` calls `defn.body(B, S, *inputs)`)."""

    def replay(B: Builder, S, *args: Coll) -> Coll:
        env: dict[tuple[str, int], Any] = {("p", i): a.refs for i, a in enumerate(args)}
        logs: dict[int, tuple[Any, Any]] = {}
        for k, node in enumerate(body.nodes):
            colls = [Coll(a.type, substitute(a.refs, env, logs)) for a in node.args]
            if node.form in ("call", "primitive"):
                out = B.call(node.fn, *colls, name=node.name)
            elif node.form == "batch":
                out = B.batch(node.fn, *colls, axes=node.axes, n=node.n, name=node.name)
            else:
                out = B.scan(node.fn, colls[0], xs=colls[1:1 + node.n_xs], shared=colls[1 + node.n_xs:], n=node.n, name=node.name)
            env[("n", k)] = out.refs
        return Coll(body.ret_type, substitute(body.ret, env))

    return replay
