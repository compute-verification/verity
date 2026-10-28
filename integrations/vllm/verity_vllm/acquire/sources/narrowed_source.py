"""A narrowed linear's rows (`query.manifest.format.NARROWED_MEMBER`, `<member>@<width>`: a manifest row's `narrowed_of`) as an
acquisition source: the columns `columns` of every token row of the tensor the module returns, committed as `<module>/<member>@<width>`
from the tensor the committer commits whole, with no second capture.

The last layer's fused qkv at a prompt step is the one such site (`program.prune.narrowed`): `dead_free` keeps its last row whole (the
LM head reads it) and of the rows before it only the keys' and values' columns, which later steps' attention reads; nothing reads the
rest.  A forward hook on the module, after the committer's own, takes the column slice of every token row on a step where some
request has rows before its last (more tokens than requests) and acquires it.  The binding map binds a request's rows before its last
to its `<member>@<width>` identity; the slice of its last row is a committed extra, so a batched step's share stays its token rows.
vLLM rotates the keys in place right after the module returns, so the gather collector flushes the slice at once
(`allreduce_norm._flush`), as at the module's own flush point; the arena committer copies it at `_acquire`.

The sites are the manifest's narrowed rows, each with the `columns` its Calls record (`query.required`); a narrowed row without them,
or two rows of one member that disagree, refuses the Commit by name.  Under an external acquisition (the compiled sources, which wrap
no forward) the source does not attach, and the Commit's coverage names the narrowed rows missing.
"""
from __future__ import annotations

import time
from typing import Any, Mapping

from verity_vllm.acquire.sources.allreduce_norm import _flush
from verity_vllm.acquire.sources.vocab_range_source import module_of

SHORT_NAME = "narrowed_rows"
#: bytes per element of a row's dtype; any other dtype is sized at 4 bytes, an upper bound
_BYTES = {"bf16": 2, "f16": 2, "u8": 1, "i8": 1}


def manifest_sites(manifest: Mapping[str, Any] | None) -> dict[str, dict[str, dict[str, Any]]]:
    """{op path: {member: {"columns": (c0, c1), "bytes": element bytes}}} of the manifest's narrowed rows.  ValueError, naming the row,
    for one whose `columns` are missing or not its width, or two rows of one member at different columns."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for row in (manifest or {}).get("identities") or ():
        r = {**row, **(row.get("identity") or {})}
        if not r.get("narrowed_of"):
            continue
        op, member, cols, w = str(r["op_path"]), str(r["output_member"]), r.get("columns"), int(r.get("width") or 0)
        if not cols or len(cols) != 2 or int(cols[1]) - int(cols[0]) != w or int(cols[0]) < 0:
            raise ValueError(f"narrowed rows: {op}/{member} (step {r.get('step')}, request {r.get('request_id')}) records columns {cols} "
                             f"for width {w}; the manifest must name the columns of its rows it is (query.required, Correspondence.columns)")
        site = out.setdefault(op, {}).setdefault(member, {"columns": (int(cols[0]), int(cols[1])), "bytes": _BYTES.get(str(r.get("dtype")), 4)})
        if site["columns"] != (int(cols[0]), int(cols[1])):
            raise ValueError(f"narrowed rows: {op}/{member} at columns {site['columns']} and {tuple(cols)} (step {r.get('step')})")
    return out


class NarrowedRowsSource:
    short_name = SHORT_NAME

    def __init__(self, committer: Any, model: Any, sites: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
        """`sites`: `manifest_sites`."""
        import torch

        if not sites:
            raise ValueError("narrowed rows: no site to tap")
        self.torch, self.c, self.model = torch, committer, model
        self.sites = {str(op): {str(m): dict(v) for m, v in members.items()} for op, members in sites.items()}
        self.own_names = frozenset(f"{op}/{m}" for op, members in self.sites.items() for m in members)
        self._handles: list[Any] = []
        self._held: list[Any] = []
        self.stats: dict[str, Any] = {"tensors": 0, "bytes": 0, "rows": 0, "sites": len(self.sites), "errors": 0}
        self.active = False

    def install(self) -> None:
        mods = dict(self.model.named_modules())
        where = {op: module_of(op, mods) for op in self.sites}
        missing = sorted(op for op, m in where.items() if m is None)
        if missing:
            raise RuntimeError(f"narrowed rows: the manifest's narrowed sites are not modules of the model: {missing[:4]}")
        for op, m in where.items():
            self._handles.append(mods[m].register_forward_hook(self._hook(op)))
        self.active = True

    def uninstall(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._held.clear()
        self.active = False

    def on_step_begin(self) -> None:
        self._held.clear()

    def error(self, msg: str) -> None:
        self.stats["errors"] += 1
        self.c.errors.append(msg[:300])

    def prompt_step(self) -> bool:
        """Whether some request of the committed step has rows before its last: more tokens than requests (the committer's
        `_plan_shape`, (tokens, requests)); every step when the committer records no shape."""
        shape = getattr(self.c, "_plan_shape", None)
        return shape is None or int(shape[0]) > max(1, int(shape[1]))

    def _hook(self, op: str) -> Any:
        members = self.sites[op]

        def post(_module: Any, _args: tuple, out: Any) -> None:
            job = self.c._cur
            if job is None or not self.prompt_step():
                return
            ts = [o for o in out if isinstance(o, self.torch.Tensor)] if isinstance(out, (tuple, list)) else [out]
            if len(ts) != 1 or not isinstance(ts[0], self.torch.Tensor) or ts[0].dim() < 1:
                return self.error(f"narrowed rows: {op} returned {type(out).__name__} with {len(ts)} tensors; its narrowed rows are "
                                  "columns of the one tensor it returns")
            t = ts[0]
            rows = t.reshape(int(t.shape[0]), -1)
            for member, g in members.items():
                c0, c1 = g["columns"]
                if c1 > int(rows.shape[1]):
                    self.error(f"narrowed rows: {op} returned rows of {int(rows.shape[1])} columns; {member} is its columns [{c0}, {c1})")
                    continue
                t0 = time.perf_counter()
                v = rows[:, c0:c1]
                self._held.append(v)
                self.c._acquire(f"{op}/{member}", v, job)
                n = int(v.numel()) * int(v.element_size())
                self.stats["tensors"] += 1
                self.stats["bytes"] += n
                self.stats["rows"] += int(v.shape[0])
                spans = getattr(self.c, "spans", None)
                if spans is not None:
                    spans.add("acquisition", time.perf_counter() - t0, step=job.step, nbytes=n)
            _flush(self.c)

        return post

    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        members = [g for site in self.sites.values() for g in site.values()]
        return {"per_token_bytes": sum((g["columns"][1] - g["columns"][0]) * g["bytes"] for g in members), "per_seq_bytes": 0,
                "tensors_per_step": len(members)}

    def declaration_classes(self) -> list[dict]:
        return [{"class": SHORT_NAME, "how": "bound-as-leaf", "rule": None, "count": self.stats["tensors"], "bytes": self.stats["bytes"],
                 "tensors": self.stats["tensors"], "rows": self.stats["rows"], "errors": self.stats["errors"],
                 "evidence": (f"narrowed-rows source: `<module>/<member>@<width>` of {len(self.sites)} sites, the manifest's columns of every "
                              "token row of the tensor the module returns, at the steps with prompt rows")}]


def applies(manifest: Mapping[str, Any] | None) -> bool:
    """Whether the manifest names a narrowed row (`narrowed_of`)."""
    return any({**r, **(r.get("identity") or {})}.get("narrowed_of") for r in (manifest or {}).get("identities") or ())


def attach(com: Any, manifest: Mapping[str, Any] | None, *, model: Any = None) -> NarrowedRowsSource | None:
    """Attach the source to `com` when the manifest names narrowed rows; None otherwise, and under an external acquisition (the
    compiled sources), which it says.  A manifest whose narrowed rows lack their columns is refused (exit 3)."""
    if not applies(manifest):
        return None
    if getattr(com, "external_acquisition", False):
        print(f"[delta] NOTE: narrowed rows NOT attached: committer {getattr(com, 'name', '?')} acquires every value from compiled graphs; the manifest's "
              "narrowed rows stay unbound (coverage names them)", flush=True)
        return None
    try:
        sites = manifest_sites(manifest)
    except ValueError as e:
        print(f"[delta] REFUSE: {e}", flush=True)
        raise SystemExit(3) from None
    src = NarrowedRowsSource(com, model if model is not None else getattr(com, "model", None), sites)
    com.attach_source(src)
    print(f"[delta] narrowed-rows source attached: {', '.join(f'{op}/{m} {g['columns']}' for op, ms in sites.items() for m, g in ms.items())} "
          f"-> committer {getattr(com, 'name', '?')}", flush=True)
    return src


__all__ = ["SHORT_NAME", "NarrowedRowsSource", "manifest_sites", "applies", "attach"]
