"""A two-stage sampled audit's interior registration, from replay.

The boundary registration's replay blocks are request spans: one request's lifetime in a served window, padded to its size
class's n_rho proof blocks. After the first stage samples Q of them, the prover replays each sampled span and registers its
proof blocks' values, phase ``interior``, as the record `Stream.register_interiors` receives: population Q·n_rho, the
sampled span ``sample[i]``'s units at [i·n_rho, (i+1)·n_rho), chained after the boundary registration.

* `span`: one sampled span's units of the stream's template, read from its `Replay` (`check.replay.request_replay`: one
  request's values replayed on the CPU, a request served from the prefix cache with its hit recomputed) exactly as the
  serving rows read the committed store (`serving_rows.population`, `serving_rows.read_all`).
* `captured`: a span of a token-select template from what a replay at the served step shapes took at the sampler
  (`write_capture`: each sampled token's logits row and id), laid out as `span` lays a token select's units out. A
  replay on the GPU writes it for a served model the CPU `Replay` is too slow for; tests write it from a `Replay`.
* `interior`: the replay sample's spans, each padded to n_rho with copies of its last unit, committed as one window
  (`ml.windows.rows`, every leaf's salt fresh from the OS) of one shared-row member, each input port a table of its own
  Q·n_rho rows and unit i reading row i, so each row a session proves is a registered read of the interior record. It is
  registered after ``previous``, the boundary registration's digest, under the window
  ``{kind, run, row, interior: true[, replay]}``. Its run is `interior_run` of that digest and
  the replay sample, so every port's domain binds which replay block each n_rho units are; the challenger derives the
  same ports from what it holds (`interior_ports`). The spans must share one Program, as a size class's spans share its
  replay Definition.
* `register`: the record on the two-stage stream after its first stage.

A span reads one template's units, one member of word rows; a chunked or shared-row template's span is refused by name.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.spot.one_stage import registration as R
from verity.ml.definitions.numerics.rows import SAMPLED_MEMBER
from verity.ml.windows import rows as CS
from verity_vllm.check.replay.index import SAMPLER_OP
from verity_vllm.check.replay.request_replay import Replay
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.records.logits_capture import LogitsCapture

#: the tag `interior_run` hashes under
RUN_TAG = b"verity-vllm/interior-run/v1\x00"
#: a sampler capture's record (`write_capture`, `captured`)
CAPTURE = "capture.json"


@dataclass(frozen=True)
class Span:
    """One request span's units of one template: its member (``n`` units), the units' rows and outputs as
    `ml.windows.rows.commit` takes them, `serving_rows.read`'s index entries, which say which row each unit is, and the
    digest of the Program it was replayed from."""
    rid: str
    member: CS.Member
    data: dict[str, dict[str, np.ndarray]]
    entries: list[list[Any]]
    program: str
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.member.n

    def padded(self, n_rho: int) -> dict[str, dict[str, np.ndarray]]:
        """Its data at ``n_rho`` units, the last unit repeated after its own."""
        if not 1 <= self.n <= n_rho:
            raise ValueError(f"request {self.rid}'s span holds {self.n} units of {self.member.template}, not 1 to its size "
                             f"class's {n_rho}")
        return {kind: {p: np.concatenate([a, np.repeat(a[-1:], n_rho - self.n, axis=0)]) for p, a in self.data[kind].items()}
                for kind in ("rows", "outs")}


def span(inst: dict[str, Any], replay: Replay, template: str, *, program_dir: str | None = None,
         index: Mapping[str, Any] | None = None) -> Span:
    """Request ``replay.rid``'s units of ``template`` in its Program ``inst``, read from ``replay`` as `serving_rows`
    reads a committed store. ``program_dir`` and ``index``: the ProgramIndex's, as `serving_rows.member_rows` takes
    them; ``replay.pi`` is reused when built with them."""
    rid, t0 = replay.rid, time.perf_counter()
    arrivals = {rid: replay.arrive}
    window, rows = SRW.population({rid: inst}, {"templates": [template], "law": "subset:1"}, "00" * 64,
                                  program_dirs={rid: program_dir} if program_dir else None, index=index, arrivals=arrivals,
                                  reuse=(replay.pi,))
    m = window.members
    if len(m) != 1 or m[0].name or m[0].tables or m[0].links or m[0].wired:
        raise ValueError(f"{template}: request {rid}'s span is {[x.key for x in m]}, not one member of word rows (a chunked or "
                         "shared-row template's interior is not built)")
    t1 = time.perf_counter()
    (data, entries), = SRW.read_all(m, rows, replay, replay.weights, arrivals)
    return Span(rid, m[0], data, entries, str(inst["digest"]),
                {"index_s": round(t1 - t0, 3), "read_s": round(time.perf_counter() - t1, 3)})


def write_capture(directory: Path, template: str, spans: Mapping[str, Mapping[str, Any]]) -> Path:
    """What a replay at the served step shapes took at the sampler for each request ``rid`` of ``spans``, written to
    ``directory`` for `captured`: its ``logits`` ((n, V) u16, row j the bf16 bits its j-th token was selected from),
    ``tokens`` (the n ids selected), ``steps`` (the n engine steps) and ``program`` (what fixes its Program: the
    Program's digest, or its launch's). ``template`` is the token-select row template the rows are units of."""
    directory.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, (rid, s) in enumerate(spans.items()):
        a = np.ascontiguousarray(s["logits"], dtype="<u2")
        (directory / f"{k}.u16").write_bytes(a.tobytes())
        out[rid] = {"file": f"{k}.u16", "sha256": hashlib.sha256(a.tobytes()).hexdigest(), "rows": list(a.shape),
                    "tokens": [int(t) for t in s["tokens"]], "steps": [int(t) for t in s["steps"]],
                    "program": str(s["program"])}
    path = directory / CAPTURE
    doc = LogitsCapture(template, out).dump()
    LogitsCapture.of(doc, str(path))
    path.write_text(json.dumps(doc, indent=1))
    return path


def captured(directory: Path, rid: str, served: Sequence[int] | None = None) -> Span:
    """Request ``rid``'s span of the capture's token-select template (`write_capture`): unit j reads logits row j and
    outputs token j, laid out as `span` reads a token select's units from a Program. ``served``: the ids serving emitted,
    which the captured tokens must be, or the replay didn't reproduce the span and nothing is committed."""
    cap = LogitsCapture.of(json.loads((directory / CAPTURE).read_text()), str(directory / CAPTURE))
    s, t = cap.spans[rid], cap.template
    params, ret = SRW.template(t).fn.signature()
    ins, out = [sum(ty.leaf_widths()) // 16 for _n, ty in params], sum(ret.leaf_widths()) // 16
    if len(ins) != 1 or out != 2 or tuple(s["rows"]) != (len(s["tokens"]), ins[0]):
        raise ValueError(f"{t}: inputs of {ins} words and an output of {out}, captured rows {s['rows']}: a token select reads "
                         "one logits row and outputs one 32-bit id")
    raw = (directory / s["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != s["sha256"]:
        raise ValueError(f"request {rid}: {s['file']} is not the capture's ({s['sha256'][:16]})")
    tokens = np.asarray(s["tokens"], dtype="<u4")
    if served is not None and [int(x) for x in served] != tokens.tolist():
        raise ValueError(f"request {rid}: the replay selected {tokens.tolist()[:8]}..., serving emitted {list(served)[:8]}...: "
                         "it didn't reproduce the span")
    n = len(tokens)
    m = CS.Member(t, (("in0", ins[0]),), ((f"out.{SAMPLED_MEMBER}", 2),), n, 0)
    data = {"rows": {"in0": np.frombuffer(raw, dtype="<u2").reshape(n, ins[0]).astype(np.uint16)},
            "outs": {f"out.{SAMPLED_MEMBER}": tokens.view("<u2").reshape(n, 2).astype(np.uint16)}}
    entries = [[j, j + 1, rid, step, None, SAMPLER_OP, SAMPLED_MEMBER, 0, None] for j, step in enumerate(s["steps"])]
    return Span(rid, m, data, entries, s["program"])


def interior_run(boundary: str, sample: Sequence[int]) -> str:
    """The interior window's run: SHA-512 over the boundary registration's digest and the replay sample, in order."""
    body = json.dumps({"boundary": boundary, "replay_sample": [int(u) for u in sample]}, separators=(",", ":"))
    return hashlib.sha512(RUN_TAG + body.encode()).hexdigest()


def interior_window(member: CS.Member, n_rho: int, boundary: str, sample: Sequence[int]) -> CS.Binding:
    """The window of the replay sample's Q·n_rho units of ``member``'s template, laid out as ``member``'s (its ports and
    their words), a shared-row member of one table row a unit a port, under `interior_run`: what the prover commits and the
    challenger derives the ports of."""
    m = len(sample) * int(n_rho)
    program, query = CS.population_program([(SRW.template(member.template).fn, m)])
    return CS.Binding(program, query, {"law": "subset", "k": m}, interior_run(boundary, sample),
                      (replace(member, n=m, base=0, name="", tables=tuple((p, m) for p, _ in member.rows)),))


def interior_ports(member: CS.Member, n_rho: int, boundary: str, sample: Sequence[int]) -> dict[str, dict[str, Any]]:
    """The challenger's copy of the interior registration's ports: `interior_window`'s, from the template's layout, the
    size class, the boundary registration's digest and its own replay sample."""
    return CS.expected_ports(interior_window(member, n_rho, boundary, sample))


@dataclass(frozen=True)
class Interior:
    """An interior registration: its record, its ports and leaf layer as the challenger derives them from the released
    public bodies, the commitment it opens from, and each span's index entries."""
    record: dict[str, Any]
    ports: dict[str, dict[str, Any]]
    leaf_layer: str
    committed: CS.Committed
    entries: list[list[list[Any]]]


def interior(spans: Sequence[Span], n_rho: int, boundary: str, sample: Sequence[int], *, row: str, replay: str | None = None,
             previous: str | None = None, kind: str = "served", salt_key: bytes | None = None) -> Interior:
    """The replay sample's spans (``spans[i]`` replay block ``sample[i]``'s), each padded to ``n_rho`` units, committed
    and registered after ``previous`` (the boundary registration's digest ``boundary`` by default) under the window
    ``{kind, run, row, interior: true, replay_sample}``: ``row`` the served workload's row, ``replay_sample`` the replay
    sample in order (round 2's ``second-interior-order``), ``replay`` the replay's run when given.
    Every leaf's salt is fresh from the OS; ``salt_key``: tests only, the salts derived from it (`ml.windows.rows.commit`)."""
    if not spans or len(spans) != len(sample):
        raise ValueError(f"{len(spans)} spans for a replay sample of {len(sample)}")
    if len({s.program for s in spans}) != 1:
        raise ValueError(f"the spans were replayed from Programs {sorted({s.program[:16] for s in spans})}, not one: a size "
                         "class's spans share its replay Definition")
    first = spans[0].member
    if any((s.member.template, s.member.rows, s.member.outs, s.member.schemas) !=
           (first.template, first.rows, first.outs, first.schemas) for s in spans):
        raise ValueError("the spans' units are not one template's in one layout")
    window = interior_window(first, n_rho, boundary, sample)
    ports = CS.expected_ports(window)
    padded = [s.padded(n_rho) for s in spans]
    data = {kind_: {p: np.concatenate([d[kind_][p] for d in padded]) for p in padded[0][kind_]} for kind_ in ("rows", "outs")}
    data["refs"] = np.tile(np.arange(window.population, dtype=np.int64)[:, None], (1, len(first.rows)))
    fresh = None if salt_key is not None else {p: os.urandom(v["leaves"] * CS.SALT_BYTES) for p, v in ports.items()}
    c = CS.commit(window, [data], salt_key=salt_key, salts_of=fresh)
    provenance = {"row": row, "interior": True, "replay_sample": [int(u) for u in sample],
                  **({"replay": replay} if replay is not None else {})}
    rec = CS.registration(c, provenance, boundary if previous is None else previous, kind=kind, registrar=R)
    layer = CS.leaf_layer({m.key: c.public_body(k) for k, m in enumerate(window.members)})
    return Interior(rec, ports, layer, c, [s.entries for s in spans])


def register(stream: Any, first: Any, it: Interior, **copies: Any) -> Any:
    """``it`` on the two-stage ``stream`` after its first stage ``first``, with the challenger's ports and leaf layer
    (``copies`` override them, and pass `Stream.register_interiors`' other copies): its logged registration."""
    return stream.register_interiors(first, it.record, copies.pop("ports", it.ports),
                                     leaf_layer=copies.pop("leaf_layer", it.leaf_layer), **copies)


__all__ = ["Interior", "Span", "captured", "interior", "interior_ports", "interior_run", "interior_window", "register",
           "span", "write_capture"]
