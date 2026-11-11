"""Every repository path a maintained document names exists, so a move or a deletion that misses a README fails here instead
of leaving the next reader a path to nowhere.

A document is a tracked `*.md`, `CHANGELOG.md` aside (history names the paths things had then), or a `.lean` file under
`security/`, of which only the comments and docstrings are read (`--`, `/- -/`, `/-- -/`, as the lane-text lint reads
them). A path is a relative markdown link target, or a backticked token with a `/` in it; neither may
carry a URL scheme or a glob or template character (`* < > { } $ | @ \\`), and a trailing `#anchor`, `::symbol`, `:LINE`
or punctuation is dropped. It is checked when its first segment is a top-level entry of the tree or a retired one
(`RETIRED`). A link target is resolved from its document's directory; a backticked path may be written from the root or
from the document's directory or one of its ancestors (a package README's `tests/test_x.py`). It exists when it is a
tracked file, a directory holding one, or a fixture `fixtures/artifacts.json` registers for fetching (or a path inside
one). Under `verity/`, a token whose second segment is no entry of `verity/` is an identifier (`verity/partition/v1`),
not a path, unless it names a file (its last segment has an extension: `verity/protocols/cap.py`, from before a move)
or its second segment is a directory `verity/` had before a move (`VERITY_RETIRED`: `verity/core/`);
any other token that reads as a repository path but isn't one is in `NOT_PATHS` with its reason.

A backticked bare file name of source code (`probe.py`, `run.sh`, `hash.c`: an extension in `_NAME`) is checked too.
Prose names a file by its bare name wherever it lives (a README names its own directory's files and its neighbours'),
so the name exists when some tracked file has it; a script that was deleted, or never landed, has none. A name of
three or more dotted segments, one before the last capitalized (`Loop.RepSrc.h`, or `coin_tree.Rules.h`: a namespace,
a structure and its field), is a dotted Lean name, not a file.
"""
from __future__ import annotations

import functools
import json
import posixpath
import re

from lint.test_repo_lane_text import lean_prose
from test_repository import REGISTRY, ROOT, _tracked

#: The Lean whose comments are documents.
LEAN_DOCS = "security/"
#: Top-level names the tree had before a move, so a mention of one is still read as a path.
RETIRED = frozenset({"packages", "protocols", "pods", "census", "security_proofs", "frontends", "archive", "glossary"})
#: The same one level down: directories `verity/` had before a move, so `verity/core/` reads as a path and not as an
#: identifier.
VERITY_RETIRED = frozenset({"core", "ml"})
#: (document, token as written) that reads as a repository path but names something else, each with why.
NOT_PATHS = {
    ("verity/definitions/ops/tests/fixtures/README.md", "tests/veritor/core/golden/"): "veritor's path, where the captures were copied from",
    ("tools/research/README.md", ".cursor/mcp.json"): "the Cursor config the reader writes in their own checkout",
    (".agents/skills/kernel-engineering/SKILL.md", "panel.py"): "the kernel panel, kept in an agent store",
    ("benchmarks/pouw/harness/README.md", "panel.py"): "the kernel panel, kept in an agent store",
    ("security/Proofs/Flock/Soundness/assumption-notes/a3-uniform-bytes.md", "io.cpp"): "the Lean runtime's source",
    ("security/Proofs/Flock/Soundness/assumption-notes/a3-uniform-bytes.md", "random.c"): "the Lean runtime's source",
    ("benchmarks/numerical/README.md", "tc.rs"): "SP1's guest, deleted with the archive",
    ("benchmarks/numerical/README.md", "TCB.lean"): "the archived tables' Lean, outside this tree",
    ("benchmarks/numerical/README.md", "TCTotal.lean"): "the archived tables' Lean, outside this tree",
    ("benchmarks/numerical/README.md", "utils.cpp::bfloat16_to_gfloat_tensor"): "the GPU simulator's source",
    ("benchmarks/pous/README.md", "gmp.h"): "the system's GMP header",
    ("benchmarks/ppl/README.md", "linears_layerwise.py"): "a harness the bench replaced",
    ("fixtures/proof-format-v3/README.md", "test_format3_equivalence.py::test_write_vectors"): "veritor's test",
    ("fixtures/proof-format-v3/README.md", "format3_equivalence.rs"): "D-SP1's guest, deleted with the archive",
    ("tools/verity/lean/README.md", "lakefile.lean"): "the file the package deliberately lacks",
    ("verity/protocols/pous/verifier/DISCREPANCIES.md", "BandPubEncoder.lean"): "the band chain's Lean, outside this tree",
    ("verity/protocols/pous/verifier/PROTOCOL.md", "BandPubEncoder.lean"): "the band chain's Lean, outside this tree",
    ("verity/protocols/pous/verifier/PROTOCOL.md", "BandEncDec.lean"): "the band chain's Lean, outside this tree",
    ("verity/protocols/pous/verifier/PROTOCOL.md", "regate.py"): "the timing red team's script",
    ("security/Definitions/Pous/Model/BandChain.lean", "SpongeDense.lean"): "the band chain's Lean, outside this tree",
    ("security/Definitions/Pous/Model/BandChain.lean", "BandChainModel.lean"): "the band chain's Lean, outside this tree",
    ("security/Definitions/Pous/Model/BandChain.lean", "BandMultiModel.lean"): "the band chain's Lean, outside this tree",
    ("security/Proofs/Pous/Chain/Family.lean", "BandMultiFamilyFinal.lean"): "the band chain's Lean, outside this tree",
    ("security/Proofs/Pous/Chain/Family.lean", "flatten_family_chain.py"): "the band chain's flattener, outside this tree",
    ("security/Proofs/Pous/Chain/Family.lean", "flatten_chain.py"): "the band chain's flattener, outside this tree",
    ("security/Proofs/Pous/Chain/Flat.lean", "flatten_chain.py"): "the band chain's flattener, outside this tree",
    ("security/Proofs/Pous/Chain/Flat.lean", "Birthday.lean"): "the band chain's Lean, outside this tree",
    ("security/Proofs/Pous/Chain/Flat.lean", "P3MeetsLucky.lean"): "the band chain's Lean, outside this tree",
    ("security/Proofs/Pous/Chain/Flat.lean", "P3MeetsFinal.lean"): "the band chain's Lean, outside this tree",
    ("security/Definitions/Hash/Guarantees.lean", "EmbeddingRead.h"): "a Lean name, the read's row `h`",
    ("security/Proofs/Flock/Claims/Recursion/DecodeTies.lean", "DecAt.sh"): "a Lean name, `DecAt`'s shape `sh`",
    ("security/Proofs/Flock/Claims/Recursion/VerdictRowRDeployed.lean", "DecAt.sh"): "a Lean name, `DecAt`'s shape `sh`",
    ("security/Proofs/Flock/Claims/ZK/FrameOpenLive.lean", "d.h"): "a Lean name, the frame's pads",
    ("security/Proofs/Flock/Recursive/HPortTie.lean", "lc.h"): "a Lean name, a field of `holoRep_refines`' output",
    ("security/Proofs/Flock/Soundness/Discharge/Composed/Inner.lean", "l.c"): "a Lean name, `verifyInner`'s target",
}
#: (document, token as written): a dead path left as `main` writes it because a demo branch rewrites those lines, and a
#: file a demo branch edits isn't rewritten until it lands. Unlike `NOT_PATHS` it has no currency check, so the demo
#: branch's landing, which drops the mention, needs no edit here; the entry leaves after it.
WAITING = {
    ("security/Definitions/NetworkCertifier/Schedule.lean", "verity/protocols/accounting/communication/certifier/schedule.py"):
        "the `remote` demo branch rewrites this docstring",
    ("security/Definitions/Accounting/Guarantees/Headline.lean", "fixtures/fold_bulk_loops.json"): "both demo branches edit the file",
    ("security/Definitions/Accounting/Guarantees/Headline.lean", "fold_bulk.cu"): "both demo branches edit the file",
    ("security/Definitions/Accounting/Sheets/B200.lean", "census/DS-100.csv"): "both demo branches edit the file",
    ("security/Definitions/Accounting/Sheets/H200.lean", "census/DS-90.csv"): "both demo branches edit the file",
    ("security/Definitions/Core/TC/Pipeline.lean", "relation.py"): "both demo branches edit the file",
    ("security/Definitions/Core/TC/Relation.lean", "relation.py"): "both demo branches edit the file",
    ("security/Definitions/Pouw.lean", "verity/protocols/accounting/work/pouw/lean/scripts/fp8atom_vectors.py"):
        "both demo branches edit the file",
    ("security/Definitions/Pouw.lean", "verity/protocols/accounting/work/pouw/lean/scripts/h1t_vectors.py"):
        "both demo branches edit the file",
    ("security/Definitions/Pouw/DitherInstance.lean", "benchmarks/pouw/theory/d5_v2.py"): "both demo branches edit the file",
    ("security/Definitions/Pouw/PearlC/RowCircuit.lean", "tests/test_circuit_row_lean.py"): "both demo branches edit the file",
    ("security/Definitions/Pouw/PearlC/Tile4Circuit.lean", "tests/test_circuit_tile_lean.py"): "both demo branches edit the file",
    ("security/Definitions/Pouw/PearlC/TileCircuit.lean", "tests/test_circuit_tile_lean.py"): "both demo branches edit the file",
    ("security/Definitions/Pouw/PearlC/WholeCircuit.lean", "tests/test_circuit_whole_lean.py"): "both demo branches edit the file",
    ("security/Proofs/Pouw/Period/Accept.lean", "benchmarks/all_protocols/legs/pouw.py"): "both demo branches edit the file",
    ("security/Proofs/Pouw/TileBound/WinogradStrassen.lean", "winograd_strassen_check.py"): "both demo branches edit the file",
}

_TOKEN = re.compile(r"`([^`\s]+)`|\]\(([^)\s]+)\)")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+-]*:")
_SKIP = re.compile(r"[*<>{}$|@\\]")
_LINE = re.compile(r":\d+(-\d+)?$")
#: ASCII, as every tracked file's name is, so a Lean field of a Greek binder (`σ.c`) doesn't read as a C file.
_NAME = re.compile(r"^[\w.+-]+\.(py|sh|c|h|cc|cpp|cu|cuh|rs|lean)$", re.ASCII)
_LEAN_NAME = re.compile(r"^(?=(?:\w+\.){2,}\w+$)(?:\w+\.)*[A-Z]\w*\.")


@functools.cache
def _tree() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """(tracked files, the directories holding them, the fixtures registered for fetching)."""
    files = frozenset(_tracked())
    dirs = frozenset(f.rsplit("/", i)[0] for f in files for i in range(1, f.count("/") + 1))
    registry = json.loads((ROOT / REGISTRY).read_text(encoding="utf-8"))
    fetched = frozenset(v.rstrip("/") for k, v in registry.items() if k.startswith("art:") and isinstance(v, str))
    return files, dirs, fetched


def documents(files: frozenset[str]) -> list[str]:
    return sorted(f for f in files if f.endswith(".md") and posixpath.basename(f) != "CHANGELOG.md"
                  or f.startswith(LEAN_DOCS) and f.endswith(".lean"))


def lines(doc: str, text: str) -> list[tuple[int, str]]:
    """(line, text) of each line of ``doc`` that is prose: every line of Markdown, and of Lean only its comments."""
    if not doc.endswith(".lean"):
        return list(enumerate(text.splitlines(), 1))
    return [(first + i, line) for first, chunk in lean_prose(text) for i, line in enumerate(chunk.splitlines())]


def mentions(line: str) -> list[tuple[str, str, bool]]:
    """(token as written, the path it reads as, whether it is a link target) for each path-shaped token of ``line``: a
    link target, a backticked path, or a backticked bare source file name."""
    out = []
    for m in _TOKEN.finditer(line):
        raw, link = m[1] or m[2], m[2] is not None
        if "://" in raw or _SCHEME.match(raw) or _SKIP.search(raw):
            continue
        path = _LINE.sub("", raw.split("#")[0].split("::")[0].rstrip(".,:;)"))
        if path and (link or "/" in path or _NAME.match(path) and not _LEAN_NAME.match(path)):
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
    named_file = "." in posixpath.basename(path)
    under = rest.split("/")[0]
    if top == "verity" and rest and not named_file and under not in VERITY_RETIRED and not _exists("verity/" + under, tree):
        return False
    bases = [""] if link or not here else [""] + [here.rsplit("/", i)[0] + "/" for i in range(here.count("/") + 1)]
    return not any(_exists(base + path, tree) for base in bases)


def dead_paths(tree, read=lambda d: (ROOT / d).read_text(errors="replace")) -> list[tuple[str, int, str]]:
    """(document, line, token as written) for each path a document of ``tree`` names that the tree lacks."""
    return [(doc, n, raw) for doc in documents(tree[0]) for n, line in lines(doc, read(doc))
            for raw, path, link in mentions(line)
            if (doc, raw) not in NOT_PATHS and (doc, raw) not in WAITING and dead(doc, path, link, tree)]


def test_every_path_a_document_names_exists():
    bad = dead_paths(_tree())
    assert not bad, (
        "documents name repository paths that aren't in the tree. Write each path's current location (where it moved: "
        "`git log --follow -- PATH`) or drop the mention; an identifier that only looks like a "
        "path goes in this test's NOT_PATHS with a reason:\n" + "\n".join(f"  {d}:{n}: `{t}`" for d, n, t in bad))


def test_not_paths_are_current():
    """Each entry is still written in its document and still reads as a missing path; one a fix resolved, or whose document
    dropped it, leaves the list."""
    tree = _tree()
    stale = [(doc, tok) for doc, tok in NOT_PATHS
             if doc not in tree[0] or not any(raw == tok and dead(doc, path, link, tree)
                                              for _, line in lines(doc, (ROOT / doc).read_text())
                                              for raw, path, link in mentions(line))]
    assert not stale, f"NOT_PATHS entries that no longer apply; remove them: {stale}"


def test_a_directory_verity_had_before_a_move_reads_as_a_path():
    """`verity/core/` names a directory the tree lacks, where `verity/partition/v1` names a format: the first is a dead
    path, with or without a file under it, and the second is no path."""
    tree = (frozenset({"README.md", "verity/protocols/a.py"}), frozenset({"verity", "verity/protocols"}), frozenset())
    docs = {"README.md": "`verity/core/` `verity/ml/circuits` `verity/core/protocols/a.py` `verity/protocols/` "
                         "`verity/partition/v1` `verity/runtime/`"}
    assert dead_paths(tree, read=docs.__getitem__) == [
        ("README.md", 1, "verity/core/"), ("README.md", 1, "verity/ml/circuits"),
        ("README.md", 1, "verity/core/protocols/a.py")]


def test_the_scan_reads_each_form_and_catches_a_dead_path():
    """The rules above, on a toy tree."""
    files = frozenset({"README.md", "tools/x/README.md", "tools/x/tests/test_a.py", "verity/primitives/b.py",
                       "tools/x/CHANGELOG.md", "fixtures/artifacts.json", "tests/test_r.py", "security/A.lean",
                       "verity/primitives/B.lean"})
    dirs = frozenset({"tools", "tools/x", "tools/x/tests", "verity", "verity/primitives", "fixtures", "tests"})
    tree = (files, dirs, frozenset({"tools/x/tests/fixtures/big"}))
    docs = {
        "README.md": ("`tools/x/tests/test_a.py:12` `verity/primitives/b.py::f` `tools/x/`, `verity/partition/v1` "
                      "`verity/moved/c.py` "
                      "`tools/gone.py`. [a](tools/x/README.md#top) [b](tools/x/gone.md) [c](https://x.org/tools/y) "
                      "`tools/*.py` `pods/nebius/` `verity/primitives/gone.py` `art:abc/def` [d](#anchor) `notatop/x`"),
        "tools/x/README.md": ("`tests/test_a.py` `tests/test_b.py` [e](tests/test_a.py) [f](../../README.md) [g](gone.md) "
                              "`tools/x/tests/fixtures/big/a.bin` `test_a.py:3` `b.py::f` `gone.sh`, `out.json` `x`"),
        "tools/x/CHANGELOG.md": "`tools/gone.py`",
        "security/A.lean": ('def s := "`tools/gone.py`"\n/-- `tools/x/` and\n`tools/lean_gone.py` -/\n'
                            "def t := 1 -- `x.lean`, `Loop.RepSrc.h`, `coin_tree.Rules.h`, `Gone.lean`, `a.b.h`, `σ.c`\n"),
    }
    assert documents(files) == ["README.md", "security/A.lean", "tools/x/README.md"]
    assert dead_paths(tree, read=docs.__getitem__) == [
        ("README.md", 1, "verity/moved/c.py"),
        ("README.md", 1, "tools/gone.py"), ("README.md", 1, "tools/x/gone.md"), ("README.md", 1, "pods/nebius/"),
        ("README.md", 1, "verity/primitives/gone.py"), ("security/A.lean", 3, "tools/lean_gone.py"),
        ("security/A.lean", 4, "x.lean"), ("security/A.lean", 4, "Gone.lean"), ("security/A.lean", 4, "a.b.h"),
        ("tools/x/README.md", 1, "tests/test_b.py"),
        ("tools/x/README.md", 1, "gone.md"),
        ("tools/x/README.md", 1, "gone.sh")]
