"""circuit-check's package (its code and its data) names no circuit of the catalog it is handed and imports no integration.

The handed package is the package the modules the catalog's sidecar hands as its enumerators share (`targets.HOOKS`'
``enumerators``): the served Definitions' package.  A name of the handed catalog, in any file of the package and in any file's
path, is

- the id of a Definition the registries register whose module (`sidecars.owner`) is in the handed package, as a whole word
  (``Foo_v1`` in ``Foo_v1{N=4}`` too);
- the label of a template row a sidecar in the handed package binds (a ``TEMPLATE_PARAMS`` entry's subcircuit id);
- the handed package or a module under it, dotted or as a path, which is how a root group is named
  (``<module>.circuit-check#<group>``), and a sidecar file there by its stem (``<stem>.circuit-check``) where no sidecar
  outside the handed package shares that stem;
- a pin in ``pins/`` whose owner (`sidecars.pin_owner`) is in the handed package: it goes in that module's pins file.

An integration is a package of `sidecars.TREES` that registers itself (``adapters/<stack>/<package>`` with an
``__init__.circuit-check`` sidecar, as `sidecars` finds it): the package imports none, by an ``import`` statement or by a
string naming it, and no file names one as a word.  What circuit-check needs of a served circuit, its sidecar
hands it (`sidecars`), and a docstring that needs an example names a made-up one."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import NamedTuple

from circuit_check import sidecars as SC
from circuit_check import targets as TG

PACKAGE = Path(TG.__file__).resolve().parent
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class Found(NamedTuple):
    path: str
    line: int
    kind: str
    name: str


def handed_package() -> str:
    """The package the enumerators' modules share, refused when they share less than a package below a top level."""
    mods = [m.split(".") for m, _fn in TG._handed("enumerators")()]
    common: list[str] = []
    for parts in zip(*mods):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    assert len(common) >= 2, f"the catalog's enumerators share {'.'.join(common) or 'no package'}: no handed package to name"
    return ".".join(common)


def integrations() -> set[str]:
    return {d.name for tree in SC.TREES for d in (SC.REPO / tree).glob("*/*")
            if any((d / f"__init__{SC.SUFFIX}.{kind}").is_file() for kind in ("py", "json"))}


def under(pkg: str, module: str | None) -> bool:
    return module == pkg or (module or "").startswith(pkg + ".")


def handed_names(pkg: str) -> tuple[set[str], set[str], list[str]]:
    """(Definition ids, template row labels, sidecar stems) of the handed package ``pkg``."""
    from verity.circuits.defs import REGISTRY
    from verity_flock import subcircuit_templates as TM

    TG.load_registries()
    ids = {i for i, d in REGISTRY.defs.items() if TG._authored(d) and under(pkg, SC.owner(d))}
    rows = set()
    for entry, (unit, params) in TG.template_params().items():
        if under(pkg, unit.partition(SC.SUFFIX)[0]):
            t = TM.TEMPLATES[TG.template_of(entry)]
            rows |= {t.bind(**p).id for p in params}
    theirs = {m.rpartition(".")[2] for m in SC.discover() if not under(pkg, m)}
    stems = sorted({m.rpartition(".")[2] for m in SC.discover() if under(pkg, m)} - theirs)
    return ids, rows, stems


def _imports(path: Path, text: str, tops: set[str]) -> list[Found]:
    out = []
    for n in ast.walk(ast.parse(text)):
        names = [a.name for a in n.names] if isinstance(n, ast.Import) else \
            [n.module or ""] if isinstance(n, ast.ImportFrom) and not n.level else \
            [n.value] if isinstance(n, ast.Constant) and isinstance(n.value, str) else []
        out += [Found(SC.rel(path), n.lineno, "import", m) for m in names if m.partition(".")[0] in tops]
    return out


def findings() -> list[Found]:
    """Every name of the handed catalog and every integration import in circuit-check's package, by file and line."""
    pkg = handed_package()
    ids, rows, stems = handed_names(pkg)
    tops = integrations()
    dotted = re.compile(rf"(?<![\w.]){re.escape(pkg)}(?:\.[A-Za-z_]\w*)*(?!\w)")
    slashed = re.compile(rf"(?<![\w/.]){re.escape(pkg.replace('.', '/'))}(?:/[\w.-]*\w)*(?![\w-])")
    sidecar = re.compile(rf"(?<![\w.])(?:{'|'.join(map(re.escape, stems))}){re.escape(SC.SUFFIX)}(?![\w-])") if stems else None
    row = re.compile(rf"(?<![\w/.-])(?:{'|'.join(map(re.escape, sorted(rows, key=len, reverse=True)))})(?![\w/.-])") \
        if rows else None
    out: list[Found] = []
    for p in sorted(PACKAGE.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = SC.rel(p)
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            text = ""
        lines = [(0, p.relative_to(PACKAGE).as_posix())] + list(enumerate(text.splitlines(), 1))
        for no, line in lines:
            for w in WORD.findall(line):
                if w in ids:
                    out.append(Found(rel, no, "definition", w))
                elif w in tops:
                    out.append(Found(rel, no, "integration", w))
            for rx, kind in ((dotted, "module"), (slashed, "module"), (sidecar, "sidecar"), (row, "template row")):
                if rx is not None:
                    out += [Found(rel, no, kind, m[0]) for m in rx.finditer(line)]
        if p.suffix == ".py":
            out += _imports(p, text, tops)
    for p in sorted(SC.SHARED_PINS.glob("*/*.json")):
        for kind, pins in json.loads(p.read_text()).items():
            out += [Found(SC.rel(p), 0, "pin", key) for key in pins if under(pkg, SC.pin_owner(kind, key))]
    return sorted(set(out))


def test_the_checker_names_no_circuit_it_is_handed():
    found = findings()
    assert not found, (
        f"{len(found)} name(s) of the handed catalog ({handed_package()}) or of an integration in circuit-check's package; "
        "each goes to the sidecars of the module that defines it (a binding or a table in `<module>.circuit-check.py`, a pin "
        "in `<module>.circuit-check.json`, a peak, a held block or a family's count in `<module>.circuit-check.unkeyed.json`), "
        "and a docstring's example is made up:\n"
        + "\n".join(f"  {f.path}:{f.line}: {f.kind} {f.name}" for f in found))
