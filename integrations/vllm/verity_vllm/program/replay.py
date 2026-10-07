"""Replay of a recorded flat body over new argument collections: the body builder of a derived Program's function (`frontend.derive`)
and of a fold's (`observe.fold`).  Torch-free, so the fold imports it without the frontend."""
from __future__ import annotations

from typing import Any

from verity.ml.circuits.defs import Body, Builder
from verity.ml.circuits.refs import Coll, PartLog, concat, prefix_runs


def substitute(refs, env: dict[tuple[str, int], Any], logs: dict[int, tuple[Any, Any]] | None = None):
    """`refs` with every `(space, idx)` run replaced by the corresponding run of `env[(space, idx)]`.

    `logs` (one per `env`) maps a `PartLog` the flat body cut `refs` from to the log of its substituted shared runs, so the
    runs every longer cut shares are substituted once."""
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
