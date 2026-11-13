"""The layers of Flock's Lean package (`verity/protocols/flock/verifier`), read from its import lines; no Lean runs.

Low to high: the base packages it requires (`Base`, `Circuits`, `Commitments`, whose own order Lake's requires hold);
Flock alone; the ZK server (`Flock.Firewall` and the modules under it, the library `FlockZkServer`, with the executables
`flock-firewall` and `flock-frame-tape`); and the recursion (`Flock.Rec*` and ``REC_MODULES``, the library `FlockRec`,
with V*'s circuit `VStar` and the `flock-rec-*` and `flock-witness` executables). A module imports its own layer and the
layers below it, so Flock alone reads neither the ZK server nor the recursion, and the ZK server reads nothing of the
recursion. The umbrella `Flock` and the tests (`Flock.Test.*`) read every layer. ``UP`` holds the imports that still
point up, each with what removes it; an entry that matches no import fails, so the list only shrinks.

A module that moved to a base package keeps its old name in Flock's package as a forwarder until every importer names
the base module: it holds `module`, one `public import` of the module it re-exports, and a comment. Anything else there
would be code of the base that no base package holds.
"""
from __future__ import annotations

import re
from pathlib import Path

VERITY = Path(__file__).resolve().parents[4]
LEAN = VERITY / "protocols" / "flock" / "verifier"
#: each base package's root module and its directory
BASE = {"Base": VERITY / "circuits" / "base", "Circuits": VERITY / "circuits" / "lean",
        "Commitments": VERITY / "commitments" / "lean"}

LAYERS = ("the base packages", "Flock alone", "the ZK server", "the recursion", "every layer")
BELOW, FLOCK, ZK, REC, ALL = range(5)
ZK_ROOTS = {"FlockZkServer", "FlockFirewall", "FlockFrameTape"}
REC_ROOTS = {"FlockRec", "FlockRecLinks", "FlockRecOpening", "FlockRecSparse", "FlockWitness"}
#: the recursion's modules that `Flock.Rec…` doesn't spell: the recursive audit's predicates
REC_MODULES = {"Flock.Witness"}
#: `Flock.Rec…` by spelling only: the session's record and its text are Flock alone's
NOT_REC = {"Flock.Record", "Flock.RecordText"}
#: the imports that point up, each with what removes it
UP: dict[tuple[str, str], str] = {
    ("Flock.HmRow", "Flock.RecSparse"):
        "until the `rec-*` pins' checks that `HmRow.parse` runs are a module of the recursion",
    ("Flock.HmRow", "Flock.RecOpening"):
        "until the `rec-*` pins' checks that `HmRow.parse` runs are a module of the recursion",
    ("FlockVerify", "Flock"):
        "until `flock-verify`'s recursive commands are an executable of the recursion: `verifyCmd` names the `Rec*` "
        "modules and `VStar` through the umbrella",
}

#: `/- … -/` and `-- …`
COMMENT = re.compile(r"/-.*?-/|--[^\n]*", re.S)
IMPORT = re.compile(r"^(?:(?:public|private|meta)\s+)*import\s+(?:all\s+)?(\S+)\s*$", re.M)


def layer(module: str) -> int:
    """The layer of a module a file of Flock's package names; Lean's own modules sit below every layer."""
    root = module.split(".")[0]
    if module == "Flock" or module.startswith("Flock.Test."):
        return ALL
    if module == "Flock.Firewall" or module.startswith("Flock.Firewall.") or module in ZK_ROOTS:
        return ZK
    if root == "VStar" or module in REC_ROOTS | REC_MODULES or (
            re.fullmatch(r"Flock\.Rec\w*", module) and module not in NOT_REC):
        return REC
    if root.startswith("Flock"):
        return FLOCK
    return BELOW


def code(text: str) -> str:
    return COMMENT.sub("", text)


def modules(package: Path = LEAN) -> dict[str, Path]:
    """{module: file} of a Lean package's own sources."""
    return {".".join(p.relative_to(package).with_suffix("").parts): p
            for p in sorted(package.rglob("*.lean")) if ".lake" not in p.parts}


def imports(text: str) -> list[str]:
    return IMPORT.findall(code(text))


def upward(sources: dict[str, str]) -> list[tuple[str, str]]:
    """(module, import) for each import of a module of a layer above its own."""
    return [(m, i) for m, text in sorted(sources.items()) for i in dict.fromkeys(imports(text)) if layer(i) > layer(m)]


def test_no_module_imports_a_layer_above_its_own():
    up = upward({m: p.read_text() for m, p in modules().items()})
    new = [f"{m} ({LAYERS[layer(m)]}) imports {i} ({LAYERS[layer(i)]})" for m, i in up if (m, i) not in UP]
    assert not new, ("Flock's Lean is layered: Flock alone, then the ZK server, then the recursion, each reading only "
                     "itself and the layers below. Move what the lower module needs down, or the importing declarations "
                     "up:\n  " + "\n  ".join(new))
    gone = sorted(f"{m} -> {i}" for m, i in set(UP) - set(up))
    assert not gone, "these imports no longer point up: delete their entries of UP:\n  " + "\n  ".join(gone)


def test_an_upward_import_is_found():
    """The scan isn't vacuous: an import in a comment isn't one, `import all` and `meta import` are, and each layer reads
    down and never up."""
    assert imports("module\n\npublic import Flock.Hash\n-- import Flock.RecCoins\n/- import Flock.Firewall -/\n"
                   "import all Init.System.ST\npublic meta import Flock.Canon\n") == ["Flock.Hash", "Init.System.ST", "Flock.Canon"]
    assert [layer(m) for m in ("Init.Data", "Base.Hash", "Flock.Verify", "Flock.Record", "FlockVerify", "Flock.Firewall",
                               "Flock.Firewall.Honest", "FlockFirewall", "Flock.RecCoins", "VStar.Cli", "FlockRecLinks",
                               "Flock.Witness", "Flock.WitnessZk", "Flock", "Flock.Test.RecCap")] == [
        BELOW, BELOW, FLOCK, FLOCK, FLOCK, ZK, ZK, ZK, REC, REC, REC, REC, FLOCK, ALL, ALL]
    sources = {"Flock.Verify": "import Flock.Firewall.Coins\nimport Base.Hash\n", "Flock.Firewall.Coins": "import Flock.RecCoins\n",
               "Flock.RecCoins": "import Flock.Firewall\nimport Flock.Verify\nimport VStar.Cli\n",
               "Flock.Test.RecCap": "import Flock\n", "Flock": "import Flock.RecCoins\n"}
    assert upward(sources) == [("Flock.Firewall.Coins", "Flock.RecCoins"), ("Flock.Verify", "Flock.Firewall.Coins")]


def test_each_library_holds_its_layer():
    """`FlockZkServer` and `FlockRec` import every module of their layer under `Flock`, and nothing else."""
    own = modules()
    for umbrella, rank in (("FlockZkServer", ZK), ("FlockRec", REC)):
        want = sorted(m for m in own if m.startswith("Flock.") and layer(m) == rank)
        assert sorted(imports(own[umbrella].read_text())) == want, (
            f"{umbrella}.lean imports each module of {LAYERS[rank]} under `Flock` once: " + ", ".join(want))


def forwarders() -> dict[str, tuple[Path, str]]:
    """{old name: (its file in Flock's package, the base module it stands for)}."""
    own = modules()
    return {f"Flock.{m.split('.', 1)[1]}": (own[f"Flock.{m.split('.', 1)[1]}"], m)
            for root, package in BASE.items() for m in modules(package)
            if m.startswith(root + ".") and f"Flock.{m.split('.', 1)[1]}" in own}


def test_a_forwarder_only_re_exports():
    assert all((d / "lakefile.toml").is_file() for d in BASE.values())
    bad = [f"{old}: {path.relative_to(LEAN)} must hold only `module` and `public import {new}`"
           for old, (path, new) in sorted(forwarders().items())
           if code(path.read_text()).split() != ["module", "public", "import", new]]
    assert not bad, ("a module that moved to a base package keeps its old name only to re-export it; what it declares "
                     "belongs in the base module:\n  " + "\n  ".join(bad))
