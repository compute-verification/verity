"""P14: one way to do each thing: build a vLLM engine, reach into it, refuse, encode canonical JSON and draw ChaCha20.

Checked against ``allowlists/p14_one_way.json``, keyed ``(file, kind, symbol, detail)``:

* ``vllm-LLM``: a call of ``vllm.LLM`` (however it was imported) outside `BUILD`, the one place that builds a vLLM
  engine; detail ``vllm.LLM``.
* ``vllm-internals``: a read or write of vLLM's engine internals, the attributes in `INTERNALS` (as ``x.llm_engine`` or as
  the name a ``getattr``/``hasattr``/``setattr``/``delattr`` call is given), outside `HANDLE`, the one module that walks
  from an ``LLM`` to its scheduler, model runner, config and workers; detail the attribute.
* ``refusal-class``: a class named for a refusal (``...Refused``, ``...Refuse``, ``...Refusal``) outside `ERRORS`, whose
  one ``Refused`` every refusal raises; detail the class's name.
* ``canonical-json``: a function whose body is one ``return json.dumps(..., sort_keys=True, separators=(",", ":"), ...)``
  (``.encode()``d or not) outside `HASHING`, another canonical JSON encoder beside `CANONICAL_JSON` and `HASHING`'s
  NaN-refusing ``canonical_json_bytes``; the symbol is the function, detail `CANONICAL_JSON`.
* ``chacha20``: ChaCha20's constant (``b"expand 32-byte k"``, or its first word ``0x61707865``), another ChaCha20 beside
  `CHACHA20`; detail `CHACHA20`.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache

from . import _ratchet as R

NAME = "p14_one_way"
RULE = "P14 (one way to build and reach the vLLM engine, to refuse, to encode canonical JSON and to draw ChaCha20)"
BUILD = "verity_vllm/engine/build.py"
HANDLE = "verity_vllm/engine/handle.py"
ERRORS = "verity_vllm/errors.py"
HASHING = "verity_vllm/commit/hashing.py"
CANONICAL_JSON = "verity.ml.circuits.codec.canonical_json"
CHACHA20 = "verity.ml.crypto.chacha20"
VLLM_LLM = frozenset({"vllm.LLM", "vllm.entrypoints.llm.LLM"})
INTERNALS = frozenset({"llm_engine", "engine_core", "model_executor"})
REFUSAL = re.compile(r"(?:Refused|Refuse|Refusal)$")
SIGMA = frozenset({b"expand 32-byte k", "expand 32-byte k", 0x61707865})
_ATTR_FUNCS = frozenset({"getattr", "hasattr", "setattr", "delattr"})


def _canonical_encoder(fn: ast.FunctionDef | ast.AsyncFunctionDef, names: R.Aliases) -> bool:
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and R.is_str(s.value))]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    v = body[0].value
    if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "encode":
        v = v.func.value
    if not (isinstance(v, ast.Call) and names.resolve(v.func) == "json.dumps"):
        return False
    kw = {k.arg: k.value for k in v.keywords}
    sep = kw.get("separators")
    return isinstance(kw.get("sort_keys"), ast.Constant) and kw["sort_keys"].value is True \
        and isinstance(sep, (ast.Tuple, ast.List)) and [getattr(e, "value", None) for e in sep.elts] == [",", ":"]


def file_violations(src: str, rel: str) -> list[R.Violation]:
    tree = R.parse_src(src)
    names = R.aliases(tree)
    docs = R.docstring_ids(tree)
    out: list[R.Violation] = []
    for node, symbol in R.walk(tree):
        if rel != BUILD and isinstance(node, ast.Call) and names.resolve(node.func) in VLLM_LLM:
            out.append(R.Violation(rel, "vllm-LLM", symbol, "vllm.LLM", node.lineno))
        if rel != ERRORS and isinstance(node, ast.ClassDef) and REFUSAL.search(node.name):
            out.append(R.Violation(rel, "refusal-class", symbol, node.name, node.lineno))
        if rel != HASHING and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _canonical_encoder(node, names):
            qual = node.name if symbol == "<module>" else f"{symbol}.{node.name}"
            out.append(R.Violation(rel, "canonical-json", qual, CANONICAL_JSON, node.lineno))
        if isinstance(node, ast.Constant) and id(node) not in docs and type(node.value) in (bytes, str, int) \
                and node.value in SIGMA:
            out.append(R.Violation(rel, "chacha20", symbol, CHACHA20, node.lineno))
        if rel == HANDLE:
            continue
        if isinstance(node, ast.Attribute) and node.attr in INTERNALS:
            out.append(R.Violation(rel, "vllm-internals", symbol, node.attr, node.lineno))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ATTR_FUNCS \
                and len(node.args) >= 2 and R.is_str(node.args[1]) and node.args[1].value in INTERNALS:
            out.append(R.Violation(rel, "vllm-internals", symbol, node.args[1].value, node.lineno))
    return out


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    out: list[R.Violation] = []
    for p in R.library_files():
        out += file_violations(R.source(p), R.rel(p))
    return tuple(out)


def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    f"Build a vLLM engine only in {BUILD} (`build_engine`, or `verity_vllm.LLM` on top of it); reach "
                    f"its scheduler, model runner, config or workers only through {HANDLE}'s accessors (add the accessor "
                    "there if it is missing, rather than walking `llm_engine`/`engine_core`/`model_executor` here); and "
                    f"raise {ERRORS}'s `Refused` rather than declaring another refusal class; encode canonical JSON with "
                    f"`{CANONICAL_JSON}` (or {HASHING}'s `canonical_json_bytes` where NaN must be refused) rather than "
                    "another `json.dumps(sort_keys=True)` wrapper; and draw ChaCha20 with "
                    f"`{CHACHA20}.keystream` (RFC 8439's 32-bit counter and 96-bit nonce are its 64-bit counter and nonce "
                    "read another way) rather than another implementation.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_detectors_on_examples() -> None:
    bad = (
        "import vllm\n"
        "from vllm import LLM\n"
        "def build(kw):\n"
        "    a = LLM(**kw)\n"
        "    b = vllm.LLM(model='m')\n"
        "    core = getattr(a.llm_engine, 'engine_core', None)\n"
        "    return hasattr(core, 'model_executor'), getattr(b, 'other', None), {'engine_core': 1}\n"
    )
    got = sorted((v.kind, v.symbol, v.detail) for v in file_violations(bad, "verity_vllm/pipeline/x.py"))
    assert got == [("vllm-LLM", "build", "vllm.LLM"), ("vllm-LLM", "build", "vllm.LLM"),
                   ("vllm-internals", "build", "engine_core"), ("vllm-internals", "build", "llm_engine"),
                   ("vllm-internals", "build", "model_executor")], got
    assert [v.kind for v in file_violations(bad, BUILD)] == ["vllm-internals"] * 3
    assert [v.kind for v in file_violations(bad, HANDLE)] == ["vllm-LLM"] * 2
    good = "from verity_vllm.llm import LLM\ndef f(m):\n    return LLM(m).generate(['x'])\n"
    assert file_violations(good, "verity_vllm/pipeline/x.py") == []
    refusals = "class Refused(ValueError):\n    pass\nclass _SkipRefused(Exception):\n    pass\nclass Refusals:\n    pass\n"
    assert [v.detail for v in file_violations(refusals, "verity_vllm/pipeline/x.py")] == ["Refused", "_SkipRefused"]
    assert file_violations(refusals, ERRORS) == []
    encoders = (
        "import json as j\n"
        "from json import dumps\n"
        "def cj(o):\n"
        "    '''canonical.'''\n"
        "    return j.dumps(o, sort_keys=True, separators=(',', ':')).encode()\n"
        "class C:\n"
        "    def raw(self, o):\n"
        "        return dumps(o, separators=[',', ':'], sort_keys=True, default=str)\n"
        "def key(o):\n"
        "    return j.dumps(o, sort_keys=True)\n"
        "def unsorted(o):\n"
        "    return j.dumps(o, separators=(',', ':'))\n"
        "def two(o):\n"
        "    s = j.dumps(o, sort_keys=True, separators=(',', ':'))\n"
        "    return s.encode()\n"
    )
    assert sorted((v.kind, v.symbol) for v in file_violations(encoders, "verity_vllm/x.py")) == [
        ("canonical-json", "C.raw"), ("canonical-json", "cj")]
    assert file_violations(encoders, HASHING) == []
    chacha = "'''expand 32-byte k'''\nSIGMA = b'expand 32-byte k'\ndef qr(s):\n    s[0] = 0x61707865\n    return 1\n"
    assert [(v.kind, v.symbol) for v in file_violations(chacha, "verity_vllm/x.py")] == [
        ("chacha20", "<module>"), ("chacha20", "qr")]
