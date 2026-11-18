"""Second paths around the shared interfaces, each a list in `tests/interfaces_ratchet.json` that only shrinks.

Each ratchet computes its offenders from the tracked tree and fails on one its list doesn't hold, or on a file whose count
of offending lines grew; an entry that is no longer an offender fails nothing, so a cleanup lands without touching the
list. The lists are written only by this file: ``--prune`` drops stale entries and lowers counts, adding nothing (the
integrator runs it at each merge sweep), ``--init`` rewrites them from the tree (top's alone), ``--report`` prints each
ratchet's count. Tests (``tests/``, any ``/tests/``), ``archive/`` and ``fixtures/`` are never offenders.

- ``scheme_name_tables``: a Python or shell file outside `verity/protocols/compose/schemes.py` and every ``scheme.py``
  that compares against a scheme's name (``DECLARATIONS``' keys and the old ``pouw``, ``certifier``,
  ``network-certifier``), keys a mapping by it, indexes by it, or reads it with ``.get``, ``.pop`` or ``.setdefault``;
  a membership test counts every name in its literal. Python is read by token, so a docstring or comment never counts.
- ``scheme_package_imports``: an import of a scheme's package under ``benchmarks/``, ``calibration/``, ``tools/`` or
  ``applications/``. The vLLM adapter's are `tests/test_boundaries.py`'s ``RESOURCE_ACCOUNTING_IMPORTERS``
  (``test_the_vllm_adapter_imports_no_resource_accounting_protocol``), not repeated here.
- ``messages_one_sided``: a class of `verity/hooks.py`, Protocol types included, that code (an import from
  ``verity.hooks`` or an attribute of the module, read by ``ast``) names in fewer than two areas (``AREAS``).
- ``toolkit_paths``: a tracked text file naming ``/usr/local/cuda`` outside `calibration/hardware/` and the kit (``KIT``).
- ``card_names_in_code``: a Python or shell file outside `calibration/hardware/` and every ``/cards/`` that compares
  against or keys a mapping by a card, node or architecture name, case-insensitively.
- ``lean_reexports``: a Lean ``export Verity ...`` or ``export Hash ...``, or a one-line ``abbrev N ... := Q.N x y``
  whose body is another namespace's ``N`` applied to bare names only (a specialization, ``Verity.Coins (Pub N)``, is not).
- ``lean_old_names``: ceilings on the number of Lean files naming each old name, and of `security/` files defining a
  ``structure Scheme`` or ``structure Partition``."""

from __future__ import annotations

import ast
import io
import json
import keyword
import re
import subprocess
import sys
import tokenize
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "tests" / "interfaces_ratchet.json"
ME = "python3 tests/test_interfaces_ratchet.py"
SCHEMES = "verity/protocols/compose/schemes.py"
HOOKS = "verity/hooks.py"
HARDWARE = "calibration/hardware/"
OLD_SCHEMES = ("pouw", "certifier", "network-certifier")
CARD_NAMES = ("H200", "H100", "B200", "RTX PRO 6000", "sm_90", "sm_100", "sm_120", "sm90", "sm100", "sm120")
SCRIPT = (".py", ".sh")
SCHEME_IMPORT = re.compile(r"^\s*(?:from|import)\s+verity\.protocols\.(?:resource_accounting\.(?:compute|memory|network)"
                           r"|experimental\.dither)\b")
SCHEME_IMPORTERS = ("benchmarks/", "calibration/", "tools/", "applications/")
CUDA = "/usr/local/cuda"
#: the machine record's reader and the toolkit fetcher it points at: where a toolkit's directory is written down
KIT = ("benchmarks/harness/verity_bench/kit.py", "benchmarks/harness/verity_bench/kit_report.py",
       "benchmarks/kernels/machine.py", "verity/protocols/proofs/flock/prover/pod/cuda-13.3.sh")
#: where a class of `verity.hooks` can be named, the first that holds a path; `verity` is the rest of `verity/`
AREAS = (("adapter", "adapters/"),
         ("compute", "verity/protocols/resource_accounting/compute/"),
         ("poub", "verity/protocols/resource_accounting/memory/poub/"),
         ("pous", "verity/protocols/resource_accounting/memory/pous/"),
         ("network", "verity/protocols/resource_accounting/network/"),
         ("dither", "verity/protocols/experimental/dither/"),
         ("verity", "verity/"),
         ("benchmarks", "benchmarks/"))
LEAN_EXPORT = re.compile(r"^\s*export\s+(?:Verity|Hash)(?:\.[\w.]+)?\s*\(")
LEAN_ABBREV = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable)\s+)*abbrev\s+(?P<n>[^\s:(){}\[\]]+)"
                         r"(?=[\s:({\[])[^=]*:=\s*(?:_root_\.)?(?:[A-Z][\w']*\.)+(?P=n)(?:\s+[A-Za-z_][\w'₀-₉]*)*\s*$")
#: {ceiling's name: (the files it counts in, the pattern)}
OLD_LEAN = {"Claims.Msg": ("", re.compile(r"\bClaims\.Msg\b")),
            "Claims.Collision": ("", re.compile(r"\bClaims\.Collision\b")),
            "FlockSoundness.Game": ("", re.compile(r"\bFlockSoundness\.Game\b")),
            "structure Scheme": ("security/", re.compile(r"^\s*structure\s+Scheme\b", re.M)),
            "structure Partition": ("security/", re.compile(r"^\s*structure\s+Partition\b", re.M))}
TOOLKIT = ("the machine's record of its toolkits, read by one function: `verity_bench.kit.toolkit(version)` "
           "(`benchmarks/kernels/machine.py toolkit 13.0` from a shell); under `verity/` and `adapters/` a script takes "
           "`CUDA_HOME` as given and refuses without it")
#: {ratchet: the interface its offenders go through}
INTERFACE = {
    "scheme_name_tables": "`verity.protocols.compose.schemes.get(name)` and the scheme's declared fields (`Scheme`)",
    "scheme_package_imports": "the scheme by its name from the table (`verity.protocols.compose.schemes.get`), attached "
                              "through the exchange",
    "messages_one_sided": "a message of `verity.hooks` exists because two sides use it; a scheme's own need is a declared "
                          "field or a port of its `Scheme`",
    "toolkit_paths": TOOLKIT,
    "card_names_in_code": "`verity.device.load_card` and the card, node and architecture files under `calibration/hardware/`",
    "lean_reexports": "the core's own name (`Verity.X`, `Hash.X`), read where it is defined",
    "lean_old_names": "the core's own name (`Hash.Msg`, `Hash.Collision`, `Verity.Game`, `Verity.Scheme`, the core's "
                      "partition)",
}
RATCHETS = tuple(INTERFACE)


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    return [p.decode() for p in out.split(b"\0") if p]


def _excluded(path: str) -> bool:
    return path.startswith(("tests/", "archive/", "fixtures/")) or any(f"/{d}/" in path for d in ("tests", "archive", "fixtures"))


def _scheme_names(text: str) -> tuple[str, ...]:
    """``DECLARATIONS``' keys, read from `verity/protocols/compose/schemes.py`'s text, and the old names."""
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and "DECLARATIONS" in {
                getattr(t, "id", None) for t in (node.targets if isinstance(node, ast.Assign) else [node.target])}:
            table = next(n for n in ast.walk(node.value) if isinstance(n, ast.Dict))
            return tuple(sorted({ast.literal_eval(k) for k in table.keys if k is not None} | set(OLD_SCHEMES)))
    raise AssertionError(f"{SCHEMES} has no DECLARATIONS table")


def _card_names(files: dict[str, str]) -> frozenset[str]:
    """Every card's, node's and architecture's id (its file's name, its ``sku``, ``node`` or an architecture's ``name``)
    and every ``arch``, from the hardware files, and `CARD_NAMES`, lower-cased."""
    names = {n.lower() for n in CARD_NAMES}
    for path, text in files.items():
        keys = "sku|node|arch|name" if path.startswith(f"{HARDWARE}architectures/") else "sku|node|arch"
        names.add(Path(path).stem.lower())
        names |= {m.lower() for m in re.findall(rf'^\s*(?:{keys})\s*=\s*"([^"]+)"', text, re.M)}
    return frozenset(names)


def _quoted(names: Iterable[str]) -> str:
    return "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))


def _line_patterns(names: Iterable[str], flags: int = 0) -> list[re.Pattern[str]]:
    """The comparison and key-or-index shapes on one line, for shell and Python that doesn't tokenize."""
    q = _quoted(names)
    return [re.compile(rf"""(?:==|!=|\bin\b|\bis\b|startswith\(|\bcase)\s*\(?\s*["']({q})["']""", flags),
            re.compile(rf"""["']({q})["']\s*:|\[\s*["']({q})["']\s*\]""", flags)]


def _shell_hits(text: str, names: Iterable[str], fold: bool = False) -> set[int]:
    patterns = _line_patterns(names, re.I if fold else 0)
    return {i for i, line in enumerate(text.splitlines(), 1)
            if not line.lstrip().startswith(("#", "for ")) and any(p.search(line) for p in patterns)}


_OPEN = {")": "(", "]": "[", "}": "{"}
_GETTERS = {"get", "pop", "setdefault"}


def _python_hits(text: str, names: frozenset[str], fold: bool = False) -> set[int]:
    """The lines where a string literal equal to one of ``names`` is compared against (``==``, ``!=``, ``in``, ``is``,
    ``startswith(``, ``case``, or an element of a literal a membership test reads), keys a mapping, indexes one or is
    ``.get``, ``.pop`` or ``.setdefault``'s key. Falls back to `_line_patterns` where the text doesn't tokenize."""
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(text).readline)
                if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)]
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return _shell_hits(text, names, fold)
    out = set()
    for i, t in enumerate(toks):
        if t.type != tokenize.STRING:
            continue
        try:
            value = ast.literal_eval(t.string)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(value, str) or (value.lower() if fold else value) not in names:
            continue
        prev = toks[i - 1].string if i else ""
        nxt = toks[i + 1].string if i + 1 < len(toks) else ""
        if (prev in ("==", "!=", "is", "case") or nxt in ("==", "!=") or prev == "in" and not _iterates(toks, i - 1)
                or prev in ("{", ",") and nxt == ":" or prev == "[" and nxt == "]"
                or prev == "(" and i >= 2 and toks[i - 2].string == "startswith"
                or prev == "(" and i >= 3 and toks[i - 2].string in _GETTERS and toks[i - 3].string == "."
                and not (i >= 4 and toks[i - 4].string == "schemes")  # the one table's own reader
                or prev in ("(", "[", "{", ",") and _in_membership(toks, i)):
            out.add(t.start[0])
    return out


def _in_membership(toks: list[tokenize.TokenInfo], i: int) -> bool:
    """Whether token ``i`` is an element of a bracketed literal right after a membership test's ``in``."""
    depth = 0
    for j in range(i - 1, -1, -1):
        s = toks[j].string
        if s in _OPEN:
            depth += 1
        elif s in ("(", "[", "{"):
            if depth == 0:
                return j > 0 and toks[j - 1].string == "in" and not _iterates(toks, j - 1)
            depth -= 1
        elif depth == 0 and s not in (",",) and toks[j].type != tokenize.STRING:
            return False
    return False


def _iterates(toks: list[tokenize.TokenInfo], k: int) -> bool:
    """Whether the ``in`` at token ``k`` is a ``for``'s, back over its targets."""
    j = k - 1
    while j >= 0 and (toks[j].type == tokenize.NAME and not keyword.iskeyword(toks[j].string) or toks[j].string in ",()[]"):
        j -= 1
    return j >= 0 and toks[j].string == "for"


def _hooks_classes(text: str) -> dict[str, int]:
    """{class defined at the top of `verity/hooks.py`: its line}."""
    return {n.name: n.lineno for n in ast.parse(text).body if isinstance(n, ast.ClassDef)}


def _hooks_names(text: str, path: str, classes: Iterable[str]) -> set[str]:
    """The classes of `verity.hooks` a module names in code: imported from it, or an attribute of the module under the
    name it is imported as (``H.Name``, ``hooks.Name``, ``verity.hooks.Name``)."""
    try:
        tree = ast.parse(text, path)
    except SyntaxError:
        return set()
    classes = set(classes)
    top = path.count("/") == 1 and path.startswith("verity/")
    aliases, found, dotted = set(), set(), False
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == "verity.hooks" and a.asname:
                    aliases.add(a.asname)
                elif a.name == "verity.hooks":
                    dotted = True
        elif isinstance(n, ast.ImportFrom):
            if n.level == 0 and n.module == "verity.hooks" or top and n.level == 1 and n.module == "hooks":
                found |= {a.name for a in n.names if a.name in classes}
            elif n.level == 0 and n.module == "verity" or top and n.level == 1 and n.module is None:
                aliases |= {a.asname or a.name for a in n.names if a.name == "hooks"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in classes:
            v = n.value
            if isinstance(v, ast.Name) and v.id in aliases or dotted and isinstance(v, ast.Attribute) \
                    and v.attr == "hooks" and isinstance(v.value, ast.Name) and v.value.id == "verity":
                found.add(n.attr)
    return found


def _area(path: str) -> str | None:
    return next((name for name, prefix in AREAS if path.startswith(prefix)), None)


def _lean_reexport_lines(text: str) -> set[int]:
    if "export" not in text and "abbrev" not in text:
        return set()
    return {i for i, line in enumerate(text.splitlines(), 1) if LEAN_EXPORT.match(line) or LEAN_ABBREV.match(line)}


@dataclass
class Scan:
    """Each ratchet's offenders now: {path or name: the lines or files behind it}."""

    found: dict[str, dict[str, list]] = field(default_factory=lambda: {r: {} for r in RATCHETS})
    text: dict[str, list[str]] = field(default_factory=dict)
    where: dict[str, int] = field(default_factory=dict)

    def counts(self) -> dict[str, dict[str, int]]:
        """What the list holds for this scan: a count of lines or files per entry, a count of areas per class."""
        return {r: {k: v[0] if r == "messages_one_sided" else len(v) for k, v in self.found[r].items()}
                for r in RATCHETS}

    def add(self, ratchet: str, path: str, lines: Iterable[int], text: str) -> None:
        if lines := sorted(set(lines)):
            self.found[ratchet][path] = lines
            self.text[path] = text.splitlines()


def scan(files: dict[str, str]) -> Scan:
    """Every ratchet over ``files``, {tracked path: its text}."""
    s = Scan()
    schemes = frozenset(_scheme_names(files[SCHEMES])) if SCHEMES in files else frozenset(OLD_SCHEMES)
    cards = _card_names({p: t for p, t in files.items() if p.startswith(HARDWARE) and p.endswith(".toml")})
    scheme_pre = re.compile(rf"""["']({_quoted(schemes)})["']""")
    card_pre = re.compile(rf"""["']({_quoted(cards)})["']""")
    classes = _hooks_classes(files[HOOKS]) if HOOKS in files else {}
    named: dict[str, set[str]] = {c: set() for c in classes}
    for path, text in files.items():
        if _excluded(path):
            continue
        script = path.endswith(SCRIPT)
        if script and path != SCHEMES and Path(path).name != "scheme.py" and scheme_pre.search(text):
            s.add("scheme_name_tables", path, (_python_hits(text, schemes) if path.endswith(".py")
                                               else _shell_hits(text, schemes)), text)
        if script and not path.startswith(HARDWARE) and "/cards/" not in f"/{path}" \
                and card_pre.search(text.lower()):
            s.add("card_names_in_code", path, (_python_hits(text, cards, fold=True) if path.endswith(".py")
                                               else _shell_hits(text, cards, fold=True)), text)
        if path.endswith(".py") and path.startswith(SCHEME_IMPORTERS) and "verity.protocols." in text:
            s.add("scheme_package_imports", path,
                  (i for i, line in enumerate(text.splitlines(), 1) if SCHEME_IMPORT.match(line)), text)
        if CUDA in text and not path.startswith(HARDWARE) and path not in KIT:
            s.add("toolkit_paths", path, (i for i, line in enumerate(text.splitlines(), 1) if CUDA in line), text)
        if path.endswith(".py") and path != HOOKS and ("verity.hooks" in text or "import hooks" in text) \
                and (area := _area(path)):
            for c in _hooks_names(text, path, classes):
                named[c].add(area)
        if path.endswith(".lean"):
            s.add("lean_reexports", path, _lean_reexport_lines(text), text)
            for name, (under, pattern) in OLD_LEAN.items():
                if path.startswith(under) and name.split()[-1] in text and pattern.search(text):
                    s.found["lean_old_names"].setdefault(name, []).append(path)
    for c, areas in named.items():
        if len(areas) < 2:
            s.found["messages_one_sided"][c] = [len(areas), *sorted(areas)]
            s.where[c] = classes[c] or 1
    return s


def _read(paths: Iterable[str]) -> dict[str, str]:
    out = {}
    for path in paths:
        file = ROOT / path
        if _excluded(path) or not file.is_file() or file.is_symlink():
            continue
        data = file.read_bytes()
        if b"\0" not in data:
            out[path] = data.decode("utf-8", errors="replace")
    return out


@cache
def tree() -> Scan:
    return scan(_read(_tracked()))


def listed() -> dict[str, dict[str, int]]:
    return json.loads(LIST.read_text()) if LIST.is_file() else {}


def dump(lists: dict[str, dict[str, int]]) -> str:
    return json.dumps({r: dict(sorted(lists.get(r, {}).items())) for r in RATCHETS}, indent=1, sort_keys=True,
                      ensure_ascii=False) + "\n"


def grown(ratchet: str, now: dict[str, int], held: dict[str, int]) -> list[str]:
    """The entries of ``now`` the list doesn't hold, or holds at a lower count (file lists and ceilings)."""
    if ratchet == "messages_one_sided":
        return sorted(k for k in now if k not in held)
    return sorted(k for k, n in now.items() if n > held.get(k, 0))


def pruned(lists: dict[str, dict[str, int]], now: dict[str, dict[str, int]]) -> tuple[dict, list[str]]:
    """``lists`` without its stale entries and with each count lowered to today's, and what changed; nothing is added. A
    ceiling (``lean_old_names``) stays, lowered, down to 0."""
    out, said = {}, []
    for r in RATCHETS:
        out[r] = {}
        for k, n in sorted(lists.get(r, {}).items()):
            cur = now[r].get(k, 0 if r == "lean_old_names" else None)
            if cur is None:
                said.append(f"{r}: {k} dropped")
                continue
            if r != "messages_one_sided" and cur < n:
                said.append(f"{r}: {k} {n} -> {cur}")
                n = cur
            out[r][k] = n
    return out, said


def failure(ratchet: str, s: Scan, keys: list[str], held: dict[str, int]) -> str:
    head = (f"{ratchet}: new second paths. Go through {INTERFACE[ratchet]}; or, only if it truly cannot, have top "
            f"regenerate the list (`{ME} --init`) with the reason in the pull request. A cleanup needs no edit to "
            f"tests/interfaces_ratchet.json (`{ME} --prune` drops stale entries).")
    rows = []
    for k in keys:
        got = s.found[ratchet][k]
        if ratchet == "messages_one_sided":
            rows.append(f"  {HOOKS}:{s.where[k]}: class {k} is named in code only in "
                        f"{', '.join(got[1:]) or 'no area'} (areas: {', '.join(a for a, _ in AREAS)})")
        elif ratchet == "lean_old_names":
            rows.append(f"  {k}: {len(got)} files, ceiling {held.get(k, 0)}: {', '.join(got)}")
        else:
            rows.append(f"  {k}: {len(got)} lines, listed {held.get(k, 0)}:")
            rows += [f"    {k}:{i}: {s.text[k][i - 1].strip()[:160]}" for i in got]
    return "\n".join([head, *rows])


@pytest.mark.parametrize("ratchet", RATCHETS)
def test_no_second_path_grows(ratchet: str) -> None:
    s = tree()
    held = listed().get(ratchet, {})
    new = grown(ratchet, s.counts()[ratchet], held)
    assert not new, failure(ratchet, s, new, held)


def test_the_lists_are_this_files_output() -> None:
    assert LIST.read_text() == dump(listed()), (f"tests/interfaces_ratchet.json is written only by `{ME} --prune` or "
                                                "`--init`: run --prune")


# The self-test: each ratchet over made-up files, one that must be flagged and one that must not.

HOOKS_TEXT = ("from dataclasses import dataclass\n\n@dataclass\nclass Both:\n    x: int\n\n"
              "class OneSide:\n    pass\n\nclass Nowhere:\n    pass\n")
SCHEMES_TEXT = 'from types import MappingProxyType\nDECLARATIONS = MappingProxyType({\n    "pous": "m",\n    "dither": "d",\n})\n'
BASE = {SCHEMES: SCHEMES_TEXT, HOOKS: HOOKS_TEXT,
        f"{HARDWARE}cards/h200-sxm-141gb.toml": 'sku = "h200-sxm-141gb"\nname = "NVIDIA H200"\narch = "sm_90a"\n',
        f"{HARDWARE}architectures/sm_90a.toml": 'name = "sm_90a"\n'}


def _flagged(ratchet: str, files: dict[str, str]) -> dict[str, int]:
    return scan({**BASE, **files}).counts()[ratchet]


@pytest.mark.parametrize(("ratchet", "path", "text", "count"), [
    ("scheme_name_tables", "benchmarks/a.py", 'if s == "pous":\n    pass\n', 1),
    ("scheme_name_tables", "benchmarks/a.py", 'T = {\n    "pouw": 1,\n    "dither": 2,\n}\nx = c["certifier"]\n', 3),
    ("scheme_name_tables", "benchmarks/a.py", 'y = cfg.get("pous", {})\nok = name in ("pouw", "pous")\n', 2),
    ("scheme_name_tables", "benchmarks/a.py", 'match s:\n    case "network-certifier":\n        pass\n', 1),
    ("scheme_name_tables", "benchmarks/a.sh", 'if [ "$p" == "pous" ]; then :; fi\n', 1),
    ("scheme_name_tables", "benchmarks/a.py", '"""keys: {"pous": 1}, s == "pous"."""\n# s == "pous"\nNAME = "pous"\n'
                                              'f(protocol="pous")\nx = ["pous", "dither"]\n'
                                              'for p in ("pous", "dither"):\n    pass\n', 0),
    ("scheme_name_tables", "benchmarks/a.py", 'P = schemes.get("pous").protocol\nq = compose.schemes.get("dither")\n', 0),
    ("scheme_name_tables", "verity/protocols/x/scheme.py", 'if s == "pous":\n    pass\n', 0),
    ("scheme_name_tables", "benchmarks/tests/a.py", 'if s == "pous":\n    pass\n', 0),
    ("scheme_name_tables", "benchmarks/a.sh", '# [ "$p" == "pous" ]\necho "pous"\n', 0),
    ("scheme_package_imports", "tools/a.py",
     "from verity.protocols.resource_accounting.memory.poub import scheme\nimport verity.protocols.experimental.dither\n", 2),
    ("scheme_package_imports", "tools/a.py",
     "from verity.protocols.resource_accounting import scheme\nfrom verity.protocols.compose import schemes\n", 0),
    ("scheme_package_imports", "verity/a.py", "from verity.protocols.resource_accounting.compute import pouw\n", 0),
    ("toolkit_paths", "benchmarks/run.sh", "export PATH=/usr/local/cuda/bin:$PATH\n", 1),
    ("toolkit_paths", "benchmarks/run.sh", 'nvcc="$(python3 benchmarks/kernels/machine.py toolkit 13.0)/bin/nvcc"\n', 0),
    ("toolkit_paths", f"{HARDWARE}nodes/x.toml", 'toolkit = "/usr/local/cuda-13.0"\n', 0),
    ("toolkit_paths", "benchmarks/harness/verity_bench/kit.py", 'D = "/usr/local/cuda"\n', 0),
    ("card_names_in_code", "benchmarks/a.py", 'if card == "h200":\n    pass\nP = {\n    "SM_90A": 1,\n}\n', 2),
    ("card_names_in_code", "benchmarks/a.py", 'x = peak["RTX PRO 6000"]\nif arch.startswith("sm_120"):\n    pass\n', 2),
    ("card_names_in_code", "benchmarks/a.py", 'CARD = "h200-sxm-141gb"\nc = load_card(CARD)\n"""card == "h200"."""\n', 0),
    ("card_names_in_code", "benchmarks/cards/a.py", 'if card == "h200":\n    pass\n', 0),
    ("lean_reexports", "security/A.lean", "export Verity (Game)\nabbrev Msg := Hash.Msg\n"
                                          "abbrev Collision (H : Msg → Msg) (u v : Msg) : Prop := Hash.Collision H u v\n", 3),
    ("lean_reexports", "security/A.lean", "export Claims.Json (Same)\nabbrev Coins (N : ℕ) : Type := Verity.Coins (Pub N) N\n"
                                          "abbrev Collision (H : List UInt8 → D) : Prop := Hash.HasCollision H\n"
                                          "abbrev Loc : Type := m.card.Loc\nabbrev digBytes (d : Dig) : List UInt8 := enc.dig d\n",
     0),
])
def test_each_ratchet_reads_each_form(ratchet: str, path: str, text: str, count: int) -> None:
    assert _flagged(ratchet, {path: text}).get(path, 0) == count


def test_a_one_sided_message_is_found() -> None:
    files = {"adapters/x/a.py": "from verity import hooks as H\nH.Both(1)\nH.OneSide()\n",
             "verity/protocols/resource_accounting/network/b.py": "from verity.hooks import Both\n",
             "benchmarks/c.py": '"""H.Nowhere and OneSide."""\nimport verity.hooks\n# hooks.Nowhere\n'
                                "x = verity.hooks.EVENTS\n"}
    assert _flagged("messages_one_sided", files) == {"OneSide": 1, "Nowhere": 0}


def test_an_old_lean_name_is_counted() -> None:
    files = {"security/A.lean": "def f : Claims.Msg := x\nstructure Scheme where\n",
             "verity/B.lean": "theorem t : FlockSoundness.Game := x\nstructure Scheme where\n",
             "security/C.lean": "-- Claims.MsgX\nstructure Partitions where\n"}
    assert _flagged("lean_old_names", files) == {"Claims.Msg": 1, "FlockSoundness.Game": 1, "structure Scheme": 1}


def test_a_list_only_shrinks() -> None:
    held = {r: {} for r in RATCHETS} | {"toolkit_paths": {"a.sh": 2, "b.sh": 1}, "lean_old_names": {"Claims.Msg": 3},
                                        "messages_one_sided": {"Gone": 1, "Still": 0}}
    now = {r: {} for r in RATCHETS} | {"toolkit_paths": {"a.sh": 1, "c.sh": 1}, "lean_old_names": {"Claims.Msg": 4},
                                       "messages_one_sided": {"Still": 1, "New": 0}}
    assert grown("toolkit_paths", now["toolkit_paths"], held["toolkit_paths"]) == ["c.sh"]
    assert grown("lean_old_names", now["lean_old_names"], held["lean_old_names"]) == ["Claims.Msg"]
    assert grown("messages_one_sided", now["messages_one_sided"], held["messages_one_sided"]) == ["New"]
    out, said = pruned(held, now)
    assert out["toolkit_paths"] == {"a.sh": 1} and out["messages_one_sided"] == {"Still": 0}
    assert out["lean_old_names"] == {"Claims.Msg": 3}
    assert said == ["messages_one_sided: Gone dropped", "toolkit_paths: a.sh 2 -> 1", "toolkit_paths: b.sh dropped"]
    assert dump(out).endswith("}\n") and dump(out) == dump(json.loads(dump(out)))


def main(argv: list[str]) -> int:
    if argv not in (["--init"], ["--prune"], ["--report"]):
        print(f"usage: {ME} --init|--prune|--report", file=sys.stderr)
        return 2
    now = tree().counts()
    if argv == ["--report"]:
        for r in RATCHETS:
            print(r, ", ".join(f"{k} {n}" for k, n in sorted(now[r].items())) if r == "lean_old_names" else len(now[r]))
        return 0
    if argv == ["--init"]:
        LIST.write_text(dump(now))
        return 0
    out, said = pruned(listed(), now)
    LIST.write_text(dump(out))
    print("\n".join(said) or "nothing stale")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
