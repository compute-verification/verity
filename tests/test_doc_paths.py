"""Every repository path a maintained document names exists, so a move or a deletion that misses a README fails here instead
of leaving the next reader a path to nowhere.

A document is a tracked `*.md`, `CHANGELOG.md` aside (history names the paths things had then). A path is a relative markdown link target, or a backticked token with a `/` in it; neither may
carry a URL scheme or a glob or template character (`* < > { } $ | @ \\`), and a trailing `#anchor`, `::symbol`, `:LINE`
or punctuation is dropped. It is checked when its first segment is a top-level entry of the tree or a retired one
(`RETIRED`). A link target is resolved from its document's directory; a backticked path may be written from the root or
from the document's directory or one of its ancestors (a package README's `tests/test_x.py`). It exists when it is a
tracked file, a directory holding one, or a fixture `fixtures/artifacts.json` registers for fetching (or a path inside
one). Under `verity/`, a token whose second segment is no entry of `verity/` is an identifier (`verity/partition/v1`),
not a path; any other token that reads as a repository path but isn't one is in `NOT_PATHS` with its reason.

A backticked bare file name of source code (`probe.py`, `run.sh`, `hash.c`: an extension in `_NAME`) is checked too.
Prose names a file by its bare name wherever it lives (a README names its own directory's files and its neighbours'),
so the name exists when some tracked file has it; a script that was deleted, or never landed, has none.
"""
from __future__ import annotations

import functools
import json
import posixpath
import re

from test_repository import REGISTRY, ROOT, _tracked

#: Top-level names the tree had before a move, so a mention of one is still read as a path.
RETIRED = frozenset({"packages", "protocols", "pods", "census", "security_proofs", "frontends", "archive"})
#: (document, token as written) that reads as a repository path but names something else, each with why.
NOT_PATHS = {
    ("verity/ml/circuits/ops/tests/fixtures/README.md", "tests/veritor/core/golden/"): "veritor's path, where the captures were copied from",
    ("tools/research/README.md", ".cursor/mcp.json"): "the Cursor config the reader writes in their own checkout",
    (".agents/skills/kernel-engineering/SKILL.md", "panel.py"): "the kernel panel, kept in an agent store",
    ("benchmarks/pouw/harness/README.md", "panel.py"): "the kernel panel, kept in an agent store",
    ("Security/Proofs/Flock/Soundness/assumption-notes/a3-uniform-bytes.md", "io.cpp"): "the Lean runtime's source",
    ("Security/Proofs/Flock/Soundness/assumption-notes/a3-uniform-bytes.md", "random.c"): "the Lean runtime's source",
    ("benchmarks/numerical/README.md", "tc.rs"): "SP1's guest, deleted with the archive",
    ("benchmarks/numerical/README.md", "TCB.lean"): "the archived tables' Lean, outside this tree",
    ("benchmarks/numerical/README.md", "TCTotal.lean"): "the archived tables' Lean, outside this tree",
    ("benchmarks/numerical/README.md", "utils.cpp::bfloat16_to_gfloat_tensor"): "the GPU simulator's source",
    ("benchmarks/pous/README.md", "gmp.h"): "the system's GMP header",
    ("benchmarks/ppl/README.md", "linears_layerwise.py"): "a harness the bench replaced",
    ("fixtures/proof-format-v3/README.md", "test_format3_equivalence.py::test_write_vectors"): "veritor's test",
    ("fixtures/proof-format-v3/README.md", "format3_equivalence.rs"): "D-SP1's guest, deleted with the archive",
    ("tools/verity/lean/README.md", "lakefile.lean"): "the file the package deliberately lacks",
    ("verity/core/protocols/pous/DISCREPANCIES.md", "BandPubEncoder.lean"): "the band chain's Lean, outside this tree",
    ("verity/core/protocols/pous/PROTOCOL.md", "BandPubEncoder.lean"): "the band chain's Lean, outside this tree",
    ("verity/core/protocols/pous/PROTOCOL.md", "BandEncDec.lean"): "the band chain's Lean, outside this tree",
    ("verity/core/protocols/pous/PROTOCOL.md", "regate.py"): "the timing red team's script",
}

_TOKEN = re.compile(r"`([^`\s]+)`|\]\(([^)\s]+)\)")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+-]*:")
_SKIP = re.compile(r"[*<>{}$|@\\]")
_LINE = re.compile(r":\d+(-\d+)?$")
_NAME = re.compile(r"^[\w.+-]+\.(py|sh|c|h|cc|cpp|cu|cuh|rs|lean)$")


@functools.cache
def _tree() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """(tracked files, the directories holding them, the fixtures registered for fetching)."""
    files = frozenset(_tracked())
    dirs = frozenset(f.rsplit("/", i)[0] for f in files for i in range(1, f.count("/") + 1))
    registry = json.loads((ROOT / REGISTRY).read_text(encoding="utf-8"))
    fetched = frozenset(v.rstrip("/") for k, v in registry.items() if k.startswith("art:") and isinstance(v, str))
    return files, dirs, fetched


def documents(files: frozenset[str]) -> list[str]:
    return sorted(f for f in files if f.endswith(".md") and posixpath.basename(f) != "CHANGELOG.md")


def mentions(line: str) -> list[tuple[str, str, bool]]:
    """(token as written, the path it reads as, whether it is a link target) for each path-shaped token of ``line``: a
    link target, a backticked path, or a backticked bare source file name."""
    out = []
    for m in _TOKEN.finditer(line):
        raw, link = m[1] or m[2], m[2] is not None
        if "://" in raw or _SCHEME.match(raw) or _SKIP.search(raw):
            continue
        path = _LINE.sub("", raw.split("#")[0].split("::")[0].rstrip(".,:;)"))
        if path and (link or "/" in path or _NAME.match(path)):
            out.append((raw, path, link))
    return out


@functools.cache
def _names(files: frozenset[str]) -> frozenset[str]:
    return frozenset(posixpath.basename(f) for f in files)


def _exists(rel: str, tree) -> bool:
    """A tracked file, a directory holding one, or a registered fixture or a path inside one."""
    files, dirs, fetched = tree
    segments = rel.split("/")
    return rel in files or rel in dirs or any("/".join(segments[:i]) in fetched for i in range(1, len(segments) + 1))


def dead(doc: str, path: str, link: bool, tree) -> bool:
    """Whether ``path``, named in ``doc``, reads as a repository path that ``tree`` (see `_tree`) lacks."""
    here = posixpath.dirname(doc)
    if not link and "/" not in path:
        return path not in _names(tree[0])
    if link:
        path = posixpath.normpath(path.lstrip("/") if path.startswith("/") else posixpath.join(here, path))
        if path.startswith("../"):
            return False
    path = path.removeprefix("./").rstrip("/")
    top, _, rest = path.partition("/")
    if top not in RETIRED and not _exists(top, tree):
        return False
    if top == "verity" and rest and not _exists("verity/" + rest.split("/")[0], tree):
        return False
    bases = [""] if link or not here else [""] + [here.rsplit("/", i)[0] + "/" for i in range(here.count("/") + 1)]
    return not any(_exists(base + path, tree) for base in bases)


def dead_paths(tree, read=lambda d: (ROOT / d).read_text(errors="replace")) -> list[tuple[str, int, str]]:
    """(document, line, token as written) for each path a document of ``tree`` names that the tree lacks."""
    return [(doc, n, raw) for doc in documents(tree[0]) for n, line in enumerate(read(doc).splitlines(), 1)
            for raw, path, link in mentions(line) if (doc, raw) not in NOT_PATHS and dead(doc, path, link, tree)]


def test_every_path_a_document_names_exists():
    bad = dead_paths(_tree())
    assert not bad, (
        "documents name repository paths that aren't in the tree. Write each path's current location (where it moved: "
        "`git log --follow -- PATH`, `tools/move/layout_map.toml`) or drop the mention; an identifier that only looks like a "
        "path goes in this test's NOT_PATHS with a reason:\n" + "\n".join(f"  {d}:{n}: `{t}`" for d, n, t in bad))


def test_not_paths_are_current():
    """Each entry is still written in its document and still reads as a missing path; one a fix resolved, or whose document
    dropped it, leaves the list."""
    tree = _tree()
    stale = [(doc, tok) for doc, tok in NOT_PATHS
             if doc not in tree[0] or not any(raw == tok and dead(doc, path, link, tree)
                                              for line in (ROOT / doc).read_text().splitlines()
                                              for raw, path, link in mentions(line))]
    assert not stale, f"NOT_PATHS entries that no longer apply; remove them: {stale}"


def test_the_scan_reads_each_form_and_catches_a_dead_path():
    """The rules above, on a toy tree."""
    files = frozenset({"README.md", "tools/x/README.md", "tools/x/tests/test_a.py", "verity/primitives/b.py",
                       "tools/x/CHANGELOG.md", "fixtures/artifacts.json", "tests/test_r.py"})
    dirs = frozenset({"tools", "tools/x", "tools/x/tests", "verity", "verity/primitives", "fixtures", "tests"})
    tree = (files, dirs, frozenset({"tools/x/tests/fixtures/big"}))
    docs = {
        "README.md": ("`tools/x/tests/test_a.py:12` `verity/primitives/b.py::f` `tools/x/`, `verity/partition/v1` "
                      "`tools/gone.py`. [a](tools/x/README.md#top) [b](tools/x/gone.md) [c](https://x.org/tools/y) "
                      "`tools/*.py` `pods/nebius/` `verity/primitives/gone.py` `art:abc/def` [d](#anchor) `notatop/x`"),
        "tools/x/README.md": ("`tests/test_a.py` `tests/test_b.py` [e](tests/test_a.py) [f](../../README.md) [g](gone.md) "
                              "`tools/x/tests/fixtures/big/a.bin` `test_a.py:3` `b.py::f` `gone.sh`, `out.json` `x`"),
        "tools/x/CHANGELOG.md": "`tools/gone.py`",
    }
    assert documents(files) == ["README.md", "tools/x/README.md"]
    assert dead_paths(tree, read=docs.__getitem__) == [
        ("README.md", 1, "tools/gone.py"), ("README.md", 1, "tools/x/gone.md"), ("README.md", 1, "pods/nebius/"),
        ("README.md", 1, "verity/primitives/gone.py"), ("tools/x/README.md", 1, "tests/test_b.py"),
        ("tools/x/README.md", 1, "gone.md"),
        ("tools/x/README.md", 1, "gone.sh")]
