"""P12: each shared key has one owner.

Checked today, against ``allowlists/p12_shared_keys.json``:

* ``row-id-regex``: a string literal that is a regex over the row-id grammar (``__tp(\\d+)__``, ``__b\\d+``,
  ``__[^_]+__``).
* ``row-id-split``: splitting a string on ``"__"`` (``split``/``rsplit``/``partition``/``rpartition``), the other way
  the row id gets parsed.
* ``root-list``: a hand-written list, tuple or set of string literals naming code roots (``verity_vllm``, ``verity``,
  ``research``, ``integrations/vllm``, ``verity/...``, ``verity/ml/circuits/...`` (its census aside), ``verity/ml/kernels/...``,
  ``verity/core/...``, ``tools/research/...`` or paths under them), which is
  how code identity gets defined outside the one ``code_identity()``.

The owners, ``config.py`` (row-id grammar) and ``pipeline/identity.py`` (code identity), do not exist yet; the modules
that become them (``target_family.py`` into ``config.py``; ``pipeline/source_identity.py`` and
``pipeline/release_json.py`` into ``identity``) are already exempt.  Not linted: the dtype vocabulary and ``RowSpec``
(no single literal shape to scan for), and the rule's test that editing a core file in a temp tree changes the identity.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache

from . import _ratchet as R

NAME = "p12_shared_keys"
RULE = "P12 (each shared key has one owner)"
OWNERS = frozenset({"verity_vllm/config.py", "verity_vllm/pipeline/identity.py", "verity_vllm/target_family.py",
                    "verity_vllm/pipeline/source_identity.py", "verity_vllm/pipeline/release_json.py"})
ROW_REGEX = re.compile(r"__\(?(?:\?:)?(?:tp|bi|b|i|o|c)?\(?\\d|__\[\^_\]|\[\^_\]\+__")
CODE_ROOT = re.compile(r"^(?:\.\./)*(?:(verity_vllm|verity|verity.ml.circuits|verity.ml.kernels|verity.core|research|integrations/vllm|tools/research)(?:/|$)"
                       r"|verity/ml/circuits/(?!census)|verity/ml/kernels/|verity/core/)")


def file_violations(src: str, rel: str) -> list[R.Violation]:
    if rel in OWNERS:
        return []
    tree = R.parse_src(src)
    docs = R.docstring_ids(tree)
    out: list[R.Violation] = []
    for node, symbol in R.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs \
                and ROW_REGEX.search(node.value):
            out.append(R.Violation(rel, "row-id-regex", symbol, node.value[:80], node.lineno))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("split", "rsplit", "partition", "rpartition") and node.args \
                and R.is_str(node.args[0]) and node.args[0].value == "__":
            out.append(R.Violation(rel, "row-id-split", symbol, f"{node.func.attr}('__')", node.lineno))
        elif isinstance(node, (ast.Tuple, ast.List, ast.Set)) and len(node.elts) >= 2 \
                and all(R.is_str(e) for e in node.elts):
            roots = [e.value for e in node.elts if CODE_ROOT.match(e.value)]
            if roots:
                out.append(R.Violation(rel, "root-list", symbol, ", ".join(roots)[:120], node.lineno))
    return out


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    out: list[R.Violation] = []
    for p in R.library_files():
        out += file_violations(R.source(p), R.rel(p))
    return tuple(out)


def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "Parse row ids with the one row-id grammar (config) and take code identity from the one "
                    "code_identity() over the import closure.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_detectors_on_examples() -> None:
    bad = (
        "import re\n"
        "CODE_ROOTS = ('verity_vllm', 'verity_vllm_sampler', 'e2e')\n"
        "TP = re.compile(r'__tp(\\d+)__')\n"
        "def batch(row):\n"
        "    m = re.search(r'__b(\\d+)__', row)\n"
        "    return row.split('__')[4], ['git', 'status', '--', 'verity_vllm', '../../verity']\n"
    )
    got = sorted((v.kind, v.detail) for v in file_violations(bad, "verity_vllm/pipeline/x.py"))
    assert got == [("root-list", "verity_vllm"), ("root-list", "verity_vllm, ../../verity"),
                   ("row-id-regex", "__b(\\d+)__"), ("row-id-regex", "__tp(\\d+)__"), ("row-id-split", "split('__')")], got
    good = "MODS = ('verity_vllm.program', 'verity_vllm.check')\nKEY = 'a__b'\nx = name.split('_')\n"
    assert file_violations(good, "verity_vllm/pipeline/x.py") == []
    assert file_violations(bad, "verity_vllm/target_family.py") == []
