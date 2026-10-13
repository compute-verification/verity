"""Acquire explicitly listed committed-only op outputs.

The required manifest supplies the exact (step, module, op, committed_<k>)
coordinates.  Forward hooks enter a dispatcher mode only while one of those
modules runs; the mode forwards the listed op unchanged and snapshots its output
into the Commit's ordinary step stream.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from verity_vllm.pipeline.committed_units import FAMILY


def manifest_sites(manifest: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    return [dict(r) for r in (manifest or {}).get("identities") or () if r.get("family") == FAMILY]


class CommittedUnitSource:
    short_name = "committed_units"

    def __init__(self, committer: Any, model: Any, sites: list[dict[str, Any]]) -> None:
        if model is None:
            raise ValueError("committed units: a model is required for op-scoped taps")
        if not sites:
            raise ValueError("committed units: the manifest names no committed-only values")
        self.c, self.model, self.sites = committer, model, sites
        rows_by_key: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in sites:
            rows_by_key[(int(row["step"]), str(row["op_path"]), str(row["op"]))].append(row)
        self.by_key: dict[tuple[int, str, str], list[list[dict[str, Any]]]] = {}
        for key, rows in rows_by_key.items():
            units: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                units[str(row["output_member"]).split("/", 1)[0]].append(row)
            self.by_key[key] = [sorted(units[name], key=lambda r: str(r["output_member"]))
                                for name in sorted(units, key=lambda n: int(n.split("_", 1)[-1]))]
        self.modules = {str(r["op_path"]) for r in sites}
        self._handles: list[Any] = []
        self._active: list[tuple[str, Any]] = []
        self._seen: dict[tuple[int, str, str], int] = {}
        self._held: list[Any] = []
        self.stats = {"tensors": 0, "bytes": 0, "sites": len(sites), "modules": len(self.modules), "errors": 0}

    def install(self) -> None:
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode

        source = self

        class _Mode(TorchDispatchMode):
            def __init__(self, module: str) -> None:
                super().__init__()
                self.module = module

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                source._after(self.module, str(func), out, func=func, args=args, kwargs=kwargs or {})
                return out

        named = dict(self.model.named_modules())
        missing = sorted(self.modules - set(named))
        if missing:
            raise RuntimeError(f"committed units: manifest modules not found in model: {missing[:8]}")
        for name in sorted(self.modules):
            module = named[name]

            def pre(_module, _args, _name=name):
                mode = _Mode(_name)
                mode.__enter__()
                self._active.append((_name, mode))

            def post(_module, _args, out, _name=name):
                if not self._active or self._active[-1][0] != _name:
                    self.c.errors.append(f"committed units: unbalanced module tap for {_name}"[:300])
                    self.stats["errors"] += 1
                    return out
                _n, mode = self._active.pop()
                mode.__exit__(None, None, None)
                return out

            self._handles.append(module.register_forward_pre_hook(pre))
            self._handles.append(module.register_forward_hook(post, always_call=True))

    def uninstall(self) -> None:
        while self._active:
            _name, mode = self._active.pop()
            mode.__exit__(None, None, None)
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._held.clear()

    def on_step_begin(self) -> None:
        self._seen.clear()
        self._held.clear()

    def _after(self, module: str, op: str, out: Any, *, func: Any = None, args: tuple[Any, ...] = (),
               kwargs: Mapping[str, Any] | None = None) -> None:
        job = getattr(self.c, "_cur", None)
        if job is None:
            return
        key = (int(job.step), module, op)
        units = self.by_key.get(key)
        if not units:
            return
        occurrence = self._seen.get(key, 0)
        self._seen[key] = occurrence + 1
        if occurrence >= len(units):
            self.c.errors.append(f"committed units: extra {op} occurrence under {module} at step {job.step}"[:300])
            self.stats["errors"] += 1
            return
        rows = units[occurrence]
        tensors = _tensors(out)
        written = _written_values(func, args, kwargs or {}) if func is not None else []
        if written:
            tensors = [t for value in written for t in _tensors(value)]
        if len(tensors) != len(rows):
            self.c.errors.append(
                f"committed units: {op} under {module} returned {len(tensors)} tensors; "
                f"manifest unit {rows[0]['output_member'].split('/', 1)[0]} names {len(rows)}"[:300])
            self.stats["errors"] += 1
            return
        for row, tensor in zip(rows, tensors):
            self._held.append(tensor)
            self.c._acquire(f"{module}/{row['output_member']}".lstrip("/"), tensor, job)
            self.stats["tensors"] += 1
            self.stats["bytes"] += int(tensor.numel()) * int(tensor.element_size())

    def declaration_classes(self) -> list[dict[str, Any]]:
        return [{"class": FAMILY, "how": "bound-as-leaf", "rule": None, "count": self.stats["tensors"],
                 "bytes": self.stats["bytes"], "tensors": self.stats["tensors"], "errors": self.stats["errors"],
                 "evidence": "dispatcher op-output tap at each explicitly listed committed-only Program input"}]


def _tensors(value: Any) -> list[Any]:
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return [value]
    except ImportError:
        return []
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in _tensors(v)]
    if isinstance(value, dict):
        return [t for v in value.values() for t in _tensors(v)]
    return []


def _written_values(func: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> list[Any]:
    schema = getattr(func, "_schema", None)
    if schema is None:
        return []
    return [args[k] if k < len(args) else kwargs.get(arg.name)
            for k, arg in enumerate(schema.arguments)
            if arg.alias_info is not None and arg.alias_info.is_write]


def attach(com: Any, manifest: Mapping[str, Any] | None, required_classes: tuple[str, ...], *,
           explicit_classes: bool = False, model: Any = None) -> tuple[tuple[str, ...], CommittedUnitSource | None]:
    sites = manifest_sites(manifest)
    if not sites:
        return required_classes, None
    source = CommittedUnitSource(com, model if model is not None else getattr(com, "model", None), sites)
    com.attach_source(source)
    if not explicit_classes and FAMILY not in required_classes:
        required_classes = (*required_classes, FAMILY)
    return required_classes, source


__all__ = ["CommittedUnitSource", "attach", "manifest_sites"]
