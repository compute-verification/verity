"""A `@[csimp]` lemma replaces its definition only in code compiled where the lemma's module is imported: a module that calls
the definition without that import runs the stated form, which may be quadratic or worse in the window. Every module of
the package that calls a definition a `@[csimp]` lemma replaces has the lemma's module in its import closure (or is it).
"""

from __future__ import annotations

import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]

IMPORT = re.compile(r"^(?:public\s+|meta\s+)*import\s+([\w.]+)", re.M)
NAMESPACE = re.compile(r"^namespace\s+([\w.]+)", re.M)
CSIMP = re.compile(r"@\[csimp\]\s*theorem\s+[\w.']+\s*:\s*@([\w.]+)\s*=\s*@[\w.]+")
BLOCK = re.compile(r"/-.*?-/", re.S)
LINE = re.compile(r"--[^\n]*")


def modules(root: Path) -> dict[str, str]:
    return {".".join(p.relative_to(root).with_suffix("").parts): p.read_text() for p in root.rglob("*.lean")
            if ".lake" not in p.parts}


def closure(name: str, imports: dict[str, set[str]]) -> set[str]:
    seen, todo = set(), [name]
    while todo:
        m = todo.pop()
        if m not in seen:
            seen.add(m)
            todo.extend(imports.get(m, ()))
    return seen


def unscoped(mods: dict[str, str]) -> list[str]:
    """Each `caller calls target without csimp_module`: a call compiled without the lemma that replaces it."""
    imports = {m: set(IMPORT.findall(t)) for m, t in mods.items()}
    code = {m: LINE.sub("", BLOCK.sub("", t)) for m, t in mods.items()}
    out = []
    for lemma_mod, text in mods.items():
        spaces = NAMESPACE.findall(text)
        for target in CSIMP.findall(text):
            full = f"{spaces[0]}.{target}" if spaces else target
            ns, name = full.rsplit(".", 1)
            qualified = re.compile(rf"(?<![\w.]){re.escape(ns.rsplit('.', 1)[-1])}\.{re.escape(name)}(?![\w'])")
            bare = re.compile(rf"(?<![\w.]){re.escape(name)}(?![\w'])")
            defines = re.compile(rf"^(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+)?def\s+{re.escape(name)}(?![\w'])", re.M)
            for caller, body in code.items():
                if ns in NAMESPACE.findall(mods[caller]) and defines.search(body):
                    continue
                calls = qualified.search(body) or (ns in NAMESPACE.findall(mods[caller]) and bare.search(body))
                if calls and lemma_mod not in closure(caller, imports):
                    out.append(f"{caller} calls {full} without {lemma_mod}, whose @[csimp] replaces it: import it")
    return sorted(set(out))


def test_csimp_in_scope_wherever_called():
    assert unscoped(modules(PKG)) == []


def test_finds_a_missing_import(tmp_path):
    (tmp_path / "P").mkdir()
    (tmp_path / "P/Slots.lean").write_text("namespace P.Slots\ndef popOk : Nat := 0\nend P.Slots\n")
    (tmp_path / "P/PopOrder.lean").write_text(
        "import P.Slots\nnamespace P.Slots\ndef popOkRanges : Nat := 0\n"
        "@[csimp] theorem popOk_eq_popOkRanges : @popOk = @popOkRanges := rfl\nend P.Slots\n")
    caller = "import P.Slots\nnamespace P.Zk\n-- Slots.popOk in a comment is no call\ndef held : Nat := Slots.popOk\nend P.Zk\n"
    (tmp_path / "P/Zk.lean").write_text(caller)
    assert unscoped(modules(tmp_path)) == ["P.Zk calls P.Slots.popOk without P.PopOrder, whose @[csimp] replaces it: import it"]
    (tmp_path / "P/Zk.lean").write_text("import P.PopOrder\n" + caller)
    assert unscoped(modules(tmp_path)) == []
