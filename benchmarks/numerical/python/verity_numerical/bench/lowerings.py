"""Per-template lowering modules: one file per subcircuit template in each backend family, found by a registry.

Each family keeps a ``templates/`` directory beside its code (:data:`DIRECTORIES`) with at most one module per catalogue template
(``bench.templates.TEMPLATES``), named by the template id with ``-`` as ``_`` (``gemm-coordinate`` -> ``gemm_coordinate.py``),
so lanes lowering different templates of one backend edit different files.  A template without a module is ``n/s``, the reason
naming the file to create.  A module needs nothing from this package:

    TEMPLATE = "gemm-coordinate"                          the catalogue id it lowers (required)
    def supports(subcircuit, scheme) -> str | None        None when the family lowers this statement, else why not (required)
    def relation(subcircuit, scheme) -> str               the family's own name of the lowered statement, the one its pod
                                                          scripts take (``--relation``, ``REL=``); required when supports
    def stage(input_set, lo, hi, out_dir) -> Path | None  write ids [lo, hi) of an input set as the family's instance files
                                                          (optional; absent: the scripts regenerate their own frozen set, so
                                                          only that set is accepted)

``subcircuit`` is a ``bench.templates.Subcircuit`` (``.id``, ``.params``, ``.template.id``), ``scheme`` a commitment scheme id
(``frame-v3/blake3-keyed``), ``input_set`` a ``bench.input_sets.InputSet``.  The reason a module returns is what a table prints
for the cell (``n/s``).
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from . import templates as TM

REPO = Path(__file__).resolve().parents[5]
DIRECTORIES = {
    "A-GKR": REPO / "archive" / "gkr" / "templates",
    "B-Ligero": REPO / "archive" / "direct" / "ligero" / "templates",
    "C-Flock": REPO / "verity" / "ml" / "flock" / "python" / "verity_flock" / "templates",
    "D-SP1": REPO / "archive" / "sp1" / "python" / "verity_sp1" / "templates",
}


def module_name(template_id: str) -> str:
    return template_id.replace("-", "_")


@dataclass
class Registry:
    family: str
    directory: Path
    _modules: dict[str, ModuleType] | None = field(default=None, repr=False)

    def _load(self, path: Path) -> ModuleType:
        name = f"verity_lowerings.{self.family.lower().replace('-', '_')}.{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def modules(self) -> dict[str, ModuleType]:
        """Template id -> its module, for every template module in the directory."""
        if self._modules is None:
            self._modules = {}
            for path in sorted(self.directory.glob("*.py")):
                if path.name != "__init__.py":
                    mod = self._load(path)
                    self._modules[str(getattr(mod, "TEMPLATE", path.stem))] = mod
        return self._modules

    def module(self, template_id: str) -> ModuleType | None:
        return self.modules().get(template_id)

    def supports(self, sub: TM.Subcircuit, scheme: str) -> str | None:
        mod = self.module(sub.template.id)
        if mod is None:
            where = self.directory.relative_to(REPO) if self.directory.is_relative_to(REPO) else self.directory
            return f"{self.family}: no lowering module {where}/{module_name(sub.template.id)}.py"
        return mod.supports(sub, scheme)

    def relation(self, sub: TM.Subcircuit, scheme: str) -> str:
        why = self.supports(sub, scheme)
        if why is not None:
            raise ValueError(why)
        return self.modules()[sub.template.id].relation(sub, scheme)

    def stage(self, sub: TM.Subcircuit) -> Any:
        mod = self.module(sub.template.id)
        return getattr(mod, "stage", None) if mod is not None else None

    def missing(self) -> list[str]:
        """The catalogue templates this family has no module for (each ``n/s``, naming the file a lane creates)."""
        stems = {p.stem for p in self.directory.glob("*.py")}
        return [tid for tid in TM.TEMPLATES if module_name(tid) not in stems]

    def problems(self) -> list[str]:
        """Every module is named for a catalogue template and has ``supports``.  A template without a module is not a problem:
        it is ``n/s`` (:meth:`missing`)."""
        out = []
        by_file = {p.stem: p for p in self.directory.glob("*.py") if p.name != "__init__.py"}
        for stem in by_file:
            mod = self._load(by_file[stem])
            tid = getattr(mod, "TEMPLATE", None)
            if tid not in TM.TEMPLATES:
                out.append(f"{self.family}: {stem}.py names TEMPLATE {tid!r}, not a catalogue template")
            elif module_name(tid) != stem:
                out.append(f"{self.family}: {stem}.py lowers {tid!r}; its file is {module_name(tid)}.py")
            if not callable(getattr(mod, "supports", None)):
                out.append(f"{self.family}: {stem}.py has no supports(subcircuit, scheme)")
        return out


_REGISTRIES: dict[str, Registry] = {}


def registry(family: str) -> Registry:
    if family not in DIRECTORIES:
        raise KeyError(f"no lowering directory for backend family {family!r} (one of {sorted(DIRECTORIES)})")
    if family not in _REGISTRIES:
        _REGISTRIES[family] = Registry(family, DIRECTORIES[family])
    return _REGISTRIES[family]


def support_matrix(subcircuits: list[TM.Subcircuit], schemes: list[str]) -> dict[str, dict[str, str | None]]:
    """``{family: {"<subcircuit>+<scheme>": reason or None}}`` over every family's registry."""
    return {fam: {f"{s.id}+{sc}": registry(fam).supports(s, sc) for s in subcircuits for sc in schemes} for fam in DIRECTORIES}


__all__ = ["DIRECTORIES", "Registry", "module_name", "registry", "support_matrix"]
