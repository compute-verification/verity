"""One source per constant (Daniel's rule for the parameters cluster: every constant is read from one source, and a new
restatement fails at PR time). Each number in ``CONCEPTS`` is stated as a literal once, at its source in ``CANONICAL``, and
every other module reads it from there: an import, an attribute of the module, or for SHA-512's width
`hashlib.sha512().digest_size` where the module may not import `verity.ml.commitments`.

A restatement is a binding whose name fits a concept (its stem, after an optional `PREFIX_`, in any case) and whose value
is a literal the concept's test accepts, so `DELTA = 0.01` and a 32-byte `DIGEST_BYTES` stay out:

- the bindings: an assignment in a module's or a class's body (annotated or not, each target of a chain or a tuple
  unpacking, inside `if`, `try` and `with` too), a string key of the dict such an assignment builds
  (`DEFAULTS = {"p_evade": …}`), and a parameter's default in a module's function or a class's method. A function's
  locals and a call's arguments are not scanned.
- a literal: numbers, strings, and tuples, lists, sets and dicts of literals; arithmetic on numbers, whose operands may be
  the module's or the class's own literal names (so a formula restates its result); `Fraction`, `float` or `int` of
  literals. A name alone, an attribute, a subscript or any other call is a read.

The scan reads tracked Python under ``SCANNED`` but not tests (a `tests/` or `test/` directory, `test_*.py`,
`conftest.py`), fixtures or vector generators (`*vectors*.py`), which pin their own copies on purpose. A text prefilter
picks the files that bind a concept's name to something that could be a literal, and only those are parsed.

Today's restatements are ``KNOWN``, ``"path::Name": "reason"``, naming the PR that removes it, or "no PR yet". A line
that no longer matches a restatement is stale and doesn't fail, so a PR that removes one needn't touch this file.
`python tests/test_python_constants.py` lists every hit and the stale lines, and `--prune` deletes those and sets
``KNOWN_MAX``. ``KNOWN`` only shrinks: it holds ``KNOWN_MAX`` lines, and a new line raises ``KNOWN_MAX`` in the same PR,
where review sees it."""
from __future__ import annotations

import ast
import math
import operator
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path

from test_repository import ROOT, _tracked

SCANNED = ("verity/", "integrations/", "benchmarks/", "tools/", "examples/")


def _numbers(v: object) -> list:
    """Every number in a literal, through tuples, lists, sets and dict values."""
    if isinstance(v, bool) or v is None:
        return []
    if isinstance(v, (int, Fraction)) or isinstance(v, float) and math.isfinite(v):
        return [Fraction(v)]
    items = v.values() if isinstance(v, dict) else v if isinstance(v, (tuple, list, set, frozenset)) else ()
    return [n for x in items for n in _numbers(x)]


def _strings(v: object) -> set[str]:
    """Every string in a literal, dict keys included."""
    if isinstance(v, str):
        return {v.lower()}
    items = [*v.keys(), *v.values()] if isinstance(v, dict) else v if isinstance(v, (tuple, list, set, frozenset)) else ()
    return {s for x in items for s in _strings(x)}


def _is(*values) -> Callable[[str, object], bool]:
    return lambda name, v: any(n in values for n in _numbers(v))


def _periods(name: str, v: object) -> bool:
    """A week, a month and a year, by name or in any one unit (30 and 365 days)."""
    nums = {n for n in _numbers(v) if n > 0}
    return {"week", "month", "year"} <= _strings(v) or any(w * 30 / 7 in nums and w * 365 / 7 in nums for w in nums)


def _deltas(name: str, v: object) -> bool:
    nums = _numbers(v)
    return all(any(abs(n - d) <= d * Fraction(1, 10 ** 9) for n in nums) for d in (Fraction(1, 100), Fraction(1, 10 ** 6)))


def _window_delta(name: str, v: object) -> bool:
    """2^-40, or its 40 bits under a `_BITS` or `_LOG2` name."""
    if re.search(r"_(bits|log2)$", name, re.I):
        return any(abs(n) == 40 for n in _numbers(v))
    return Fraction(1, 2 ** 40) in _numbers(v)


def _width(name: str, v: object) -> bool:
    """64 bytes, or 512 under a `_BITS` name."""
    return (512 if name.lower().endswith("_bits") else 64) in _numbers(v)


#: Each concept: the stems a binding of it may be named (a regex over the name's end, lower case) and the test its value
#: passes.
CONCEPTS: dict[str, tuple[str, Callable[[str, object], bool]]] = {
    "the advice cap ℓ = 2^23 bits": ("cap_bits", _is(2 ** 23)),
    "the monitoring periods (a week, a month, a year)": ("periods", _periods),
    "the reporting δ pair (1/100, 10^-6)": ("deltas", _deltas),
    "the per-window δ = 2^-40": ("delta|delta(?:_s)?_(?:bits|log2)|p_evade", _window_delta),
    "SHA-512's width, 64 bytes (512 bits)": ("(?:digest|root)(?:_bytes)?|(?:node|label)_bytes|(?:digest|root|sha|sha512)_bits",
                                            _width),
    "SHA-512's AND count, 58,120": ("sha512(?:_ands)?", _is(58_120)),
    "PoUS's Δ, 500,000 ns": ("(?:delta|deadline)_ns", _is(500_000)),
}
#: Each concept's one source, ``path::Name``. Where it is moving, both places are listed; exactly one states the literal.
CANONICAL: dict[str, tuple[str, ...]] = {
    "the advice cap ℓ = 2^23 bits": ("integrations/vllm/verity_vllm/pipeline/advice_windows.py::CAP_BITS",),
    "the monitoring periods (a week, a month, a year)": ("verity/core/protocols/pous/wipe_monitoring.py::PERIODS",
                                                         "verity/core/protocols/profile.py::PERIODS"),
    "the reporting δ pair (1/100, 10^-6)": ("verity/core/protocols/pous/wipe_monitoring.py::DELTAS",
                                            "verity/core/protocols/profile.py::DELTAS"),
    "the per-window δ = 2^-40": ("verity/core/protocols/profile.py::WINDOW_DELTA",
                                 "verity/core/protocols/pouw/credit.py::DELTA"),
    "SHA-512's width, 64 bytes (512 bits)": ("verity/ml/commitments/scheme.py::DIGEST_BYTES",),
    "SHA-512's AND count, 58,120": ("verity/core/protocols/sampled_proofs/optimize.py::SHA512_ANDS",),
    "PoUS's Δ, 500,000 ns": ("verity/core/protocols/pous/audit.py::DEADLINE_NS",),
}
#: What to write instead, beyond importing the canonical name.
HINT = {"SHA-512's width, 64 bytes (512 bits)": ' and take ["sha512"], or `hashlib.sha512().digest_size` where the module '
                                                "may not import verity.ml.commitments"}

KNOWN_MAX = 58

#: Today's restatements, ``"path::Name": "reason"``, the reason naming the PR that removes it (or "no PR yet").
KNOWN: dict[str, str] = {
    # the advice cap
    "benchmarks/network_e2e/encodings/v1.py::Registered.cap_bits": "#5714 reads advice_windows.CAP_BITS",
    "benchmarks/prove_path/prove_path.py::CAP_BITS": "#5714 reads advice_windows.CAP_BITS",
    "integrations/vllm/verity_vllm/program/predict/steps.py::CAP_BITS": "#5714 reads advice_windows.CAP_BITS",
    # the monitoring periods
    "benchmarks/network_traces/chain_record.py::PERIODS": "#5877 derives its minutes from core's seconds",
    "benchmarks/prove_path/window_cost.py::PERIODS": "#5877 reads core's PERIODS",
    "benchmarks/rho/periods.py::PERIODS": "#5879 reads core's names through rho.params",
    "benchmarks/sampling_plan/cpu_budget.py::PERIODS": "#5878 reads core's PERIODS",
    "benchmarks/sampling_plan/transformer.py::PERIODS": "#5878 reads core's PERIODS",
    "examples/h200/interfaces/verity_e2e_demo/block_record.py::PERIODS": "#5854 reads core's PERIODS",
    "examples/lifecycle/lifecycle.py::PERIODS": "no PR yet: the lifecycle example's own table",
    # the reporting δ pair
    "benchmarks/network_traces/block_certifier.py::BLOCK_DELTAS": "no PR yet: the block record's δs, core's DELTAS",
    "benchmarks/network_traces/chain_record.py::LEAK_DELTAS": "#5877 reads core's DELTAS",
    "benchmarks/sampling_plan/cpu_budget.py::DELTAS": "#5878 reads core's DELTAS",
    "benchmarks/sampling_plan/h200.py::DELTAS": "#5878 reads core's DELTAS",
    "benchmarks/sampling_plan/transformer.py::DELTAS": "#5878 reads core's DELTAS",
    "examples/h200/interfaces/verity_e2e_demo/block_record.py::DELTAS": "#5854 reads core's DELTAS",
    # the per-window δ
    "benchmarks/nci/tiers.py::DEFAULTS[p_evade]": "#5868 reads profile.WINDOW_DELTA",
    "benchmarks/network_traces/certifier_serve.py::DELTA": "#5868 reads profile.WINDOW_DELTA",
    "benchmarks/numerical/python/verity_numerical/bench/unit_costs.py::DELTA":
        "no PR yet: prices the archived proof systems' tables, which the census leaves",
    "benchmarks/numerical/python/verity_numerical/bench/unit_costs_padding.py::DELTA":
        "no PR yet: prices the archived proof systems' tables, which the census leaves",
    "benchmarks/pouw/period_cost.py::DELTA_S_LOG2": "#5868 reads profile.WINDOW_DELTA",
    "benchmarks/pouw/theory/attn_noise.py::grind_eps(delta_log2=)": "no PR yet: the grind's δ as a default",
    "benchmarks/resource_accounting/certified.py::DELTA": "#5868 reads profile.WINDOW_DELTA",
    "benchmarks/sampling_plan/int8_llama.py::DELTA": "#5868 reads profile.WINDOW_DELTA",
    "verity/core/protocols/pouw/grind.py::DELTA_BITS": "#5867 reads profile.WINDOW_DELTA",
    "verity/core/protocols/remote_bandwidth/audit.py::DELTA": "#5867 reads profile.WINDOW_DELTA",
    "verity/core/protocols/remote_bandwidth/period.py::DELTA": "#5867 reads profile.WINDOW_DELTA",
    "verity/core/protocols/remote_bandwidth/timing.py::stated(delta=)": "no PR yet: the remote audit's default δ",
    # SHA-512's width
    "benchmarks/pous/p2_v1/reference.py::DIGEST_BYTES": "#5891 reads hashlib",
    "benchmarks/pous/p2_v1/scale.py::DIGEST": "#5891 reads hashlib",
    "benchmarks/pouw/theory/attn_noise.py::SHA_BITS": "#5891 reads hashlib",
    "benchmarks/pouw/theory/scope_offload.py::DIGEST_BYTES": "#5891 reads hashlib",
    "benchmarks/served_commit/tp_decode.py::ROOT_BYTES": "#5891 reads hashlib",
    "benchmarks/word_codec/kv.py::DIGEST_BYTES": "#5891 reads hashlib",
    "integrations/vllm/verity_vllm/commit/step_rows.py::ROOT_BYTES": "no PR yet: a frame-v3-sha512 root",
    "integrations/vllm/verity_vllm/pipeline/deferred_rows.py::SERVED_ROOT": "no PR yet: a frame-v3-sha512 root",
    "integrations/vllm/verity_vllm/pipeline/tp/run_root.py::RANK_ROOT_BYTES": "no PR yet: a rank's SHA-512 root",
    "verity/core/protocols/bandwidth/openings.py::NODE_BYTES": "#5891 reads tags.DIGEST_BYTES",
    "verity/core/protocols/bandwidth/tags.py::DIGEST_BYTES": "#5891 reads hashlib",
    "verity/core/protocols/certifier/active.py::WEIGHTS_ROOT_BYTES": "#5891 reads scheme.DIGEST_BYTES",
    "verity/core/protocols/certifier/certificate.py::DIGEST_BYTES": "#5891 reads scheme.DIGEST_BYTES",
    "verity/core/protocols/pous/streamed_wipe.py::LABEL_BYTES": "#5891 reads scheme.DIGEST_BYTES",
    "verity/core/protocols/pous/verdict_circuit.py::DIGEST_BITS": "#5891 reads scheme.DIGEST_BYTES",
    "verity/core/protocols/pous/verifier.py::DIGEST_BYTES": "#5891 reads scheme.DIGEST_BYTES",
    "verity/core/protocols/pouw/salt.py::ANCHORS_ROOT_BYTES": "#5891 reads scheme.DIGEST_BYTES",
    "verity/core/protocols/pouw/salt.py::REQUESTS_ROOT_BYTES": "#5891 reads scheme.DIGEST_BYTES",
    "verity/core/protocols/pouw/salt.py::WEIGHTS_ROOT_BYTES": "#5891 reads scheme.DIGEST_BYTES",
    "verity/core/protocols/pouw/window.py::_DIGEST": "#5891 reads scheme.DIGEST_BYTES",
    "verity/core/protocols/remote_bandwidth/audit.py::NODE_BYTES": "no PR yet: a tree node, one digest",
    "verity/core/protocols/remote_bandwidth/codec.py::DIGEST_BYTES": "#5891 reads scheme.DIGEST_BYTES",
    "verity/core/protocols/remote_bandwidth/receipts.py::DIGEST_BYTES": "#5891 reads scheme.DIGEST_BYTES",
    "verity/core/protocols/sampled_proofs/remote.py::ROOT_BYTES": "no PR yet: a frame-v3-sha512 root",
    "verity/ml/flock/python/verity_flock/rec_live.py::CAP_NODE_BYTES": "no PR yet: an hm96-sha512 node",
    # SHA-512's AND count
    "benchmarks/commitments/hiding_leaf_cost.py::SHA512_ANDS":
        "no PR yet: the formula, to keep only as a test that it equals optimize.SHA512_ANDS",
    "benchmarks/one_stage/zk_cost.py::SHA512": "no PR yet",
    "benchmarks/sampling_plan/inputs.py::SHA512": "no PR yet",
    # PoUS's Δ
    "benchmarks/pous/fast_decode/model.py::DELTA_NS": "no PR yet: the census's PR 7",
    "benchmarks/pous/p2_v1/reference.py::DELTA_NS": "no PR yet: the census's PR 7",
}

_ARITH = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
          ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.LShift: operator.lshift,
          ast.RShift: operator.rshift, ast.BitOr: operator.or_, ast.BitAnd: operator.and_, ast.BitXor: operator.xor}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: operator.invert}
_NUMBER = {"Fraction": Fraction, "float": float, "int": int}
_STEMS = "|".join(stem for stem, _ in CONCEPTS.values())
_NAME = {c: re.compile(rf"(?:\w*_)?(?:{stem})", re.I) for c, (stem, _) in CONCEPTS.items()}
_BINDS = rf"(?:{_STEMS})\b(?:['\"][ \t]*:|(?:[ \t]*,[ \t]*\w+)*[ \t]*\)?[ \t]*(?::[^=\n]*)?=(?!=))[ \t]*"
#: A file worth parsing: a concept's name bound to what may be a literal (a number, a sign, a display, `Fraction`,
#: `float` or `int`, or arithmetic starting from a name); a file that renames `Fraction` is parsed whenever it binds one.
_CANDIDATE = re.compile(_BINDS + r"(?:[-+~.(\[{\d]|(?:fractions?|float|int)\b|[a-z_]\w*[ \t]*(?:[-+*/%&|^]|<<|>>))")
_CANDIDATE_ALIASED = re.compile(_BINDS)


class _NotLiteral(Exception):
    pass


@dataclass(frozen=True)
class Hit:
    key: str        # "path::Name", "path::Class.Name", "path::DICT[key]", "path::function(param=)"
    concept: str
    line: int
    text: str       # the binding, as written


class _Module:
    """One module's bindings of a concept's name to a literal."""

    def __init__(self, path: str, text: str) -> None:
        self.path, self.text, self.hits, self.calls = path, text, [], dict(_NUMBER)
        self._body(ast.parse(text, filename=path).body, "", {})

    def literal(self, node: ast.expr, env: dict, operand: bool = False) -> object:
        """`node`'s value if it is a literal; a name counts only as an operand of arithmetic or a call."""
        match node:
            case ast.Constant(value=v) if not isinstance(v, (bytes, complex)) and v is not Ellipsis:
                return v
            case ast.Name(id=n) if operand and n in env:
                return env[n]
            case ast.UnaryOp(op=op, operand=x) if type(op) in _UNARY:
                return self._number(_UNARY[type(op)], self.literal(x, env, True))
            case ast.BinOp(left=a, op=op, right=b) if type(op) in _ARITH:
                return self._number(_ARITH[type(op)], self.literal(a, env, True), self.literal(b, env, True))
            case ast.Tuple(elts=xs) | ast.List(elts=xs) | ast.Set(elts=xs):
                return tuple(self.literal(x, env) for x in xs)
            case ast.Dict(keys=ks, values=vs) if None not in ks:
                try:
                    return {self.literal(k, env): self.literal(v, env) for k, v in zip(ks, vs)}
                except TypeError as e:
                    raise _NotLiteral from e
            case ast.Call(func=f, args=args, keywords=[]) if (fn := self._callee(f)) is not None:
                try:
                    return fn(*(self.literal(a, env, True) for a in args))
                except (ArithmeticError, TypeError, ValueError) as e:
                    raise _NotLiteral from e
        raise _NotLiteral

    def _callee(self, f: ast.expr) -> Callable | None:
        match f:
            case ast.Name(id=n):
                return self.calls.get(n)
            case ast.Attribute(value=ast.Name(id=m), attr="Fraction") if self.calls.get(m + ".") is Fraction:
                return Fraction
        return None

    @staticmethod
    def _number(op: Callable, *xs: object) -> object:
        if not all(isinstance(x, (int, float, Fraction)) and not isinstance(x, bool) for x in xs):
            raise _NotLiteral
        if op in (operator.pow, operator.lshift) and abs(xs[-1]) > 4096:
            raise _NotLiteral
        try:
            return op(*xs)
        except (ArithmeticError, TypeError, ValueError) as e:
            raise _NotLiteral from e

    def _body(self, body: list[ast.stmt], prefix: str, env: dict) -> None:
        for s in body:
            match s:
                case ast.Import(names=names) | ast.ImportFrom(names=names):
                    for a in names:
                        bound = a.asname or a.name.split(".")[0]
                        env.pop(bound, None)
                        if isinstance(s, ast.ImportFrom) and s.module == "fractions" and a.name == "Fraction":
                            self.calls[bound] = Fraction
                        elif isinstance(s, ast.Import) and a.name == "fractions":
                            self.calls[bound + "."] = Fraction
                case ast.Assign(targets=targets, value=value):
                    for t in targets:
                        self._bind(t, value, prefix, env, s)
                case ast.AnnAssign(target=t, value=value) if value is not None:
                    self._bind(t, value, prefix, env, s)
                case ast.AugAssign(target=ast.Name(id=n)):
                    env.pop(n, None)
                case ast.ClassDef(name=n, body=b):
                    env.pop(n, None)
                    self._body(b, f"{prefix}{n}.", dict(env))
                case ast.FunctionDef() | ast.AsyncFunctionDef():
                    env.pop(s.name, None)
                    a = s.args
                    positional = [*a.posonlyargs, *a.args]
                    for arg, default in [*zip(positional[len(positional) - len(a.defaults):], a.defaults),
                                         *zip(a.kwonlyargs, a.kw_defaults)]:
                        if default is not None:
                            self._emit(arg.arg, f"{prefix}{s.name}({arg.arg}=)", default, env, default)
                case ast.If() | ast.Try() | ast.With() | ast.AsyncWith():
                    for field in ("body", "orelse", "finalbody"):
                        self._body(getattr(s, field, []), prefix, env)
                    for h in getattr(s, "handlers", []):
                        self._body(h.body, prefix, env)
                case _ if type(s).__name__ == "TryStar":
                    for b in (s.body, s.orelse, s.finalbody, *(h.body for h in s.handlers)):
                        self._body(b, prefix, env)

    def _bind(self, target: ast.expr, value: ast.expr, prefix: str, env: dict, stmt: ast.stmt) -> None:
        match target, value:
            case ast.Name(id=n), _:
                v = self._emit(n, prefix + n, value, env, stmt)
                if isinstance(v, _NotLiteral):
                    env.pop(n, None)
                else:
                    env[n] = v
                if isinstance(value, ast.Dict):
                    for k, x in zip(value.keys, value.values):
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            self._emit(k.value, f"{prefix}{n}[{k.value}]", x, env, k)
            case (ast.Tuple(elts=ts) | ast.List(elts=ts)), (ast.Tuple(elts=vs) | ast.List(elts=vs)) if len(ts) == len(vs) \
                    and not any(isinstance(x, ast.Starred) for x in (*ts, *vs)):
                for t, v in zip(ts, vs):
                    self._bind(t, v, prefix, env, stmt)
            case _:
                for node in ast.walk(target):
                    if isinstance(node, ast.Name):
                        env.pop(node.id, None)

    def _emit(self, name: str, key: str, value: ast.expr, env: dict, at: ast.AST) -> object:
        """Record `key` for each concept `name` and `value` restate; the literal's value, or the `_NotLiteral` it isn't."""
        try:
            v = self.literal(value, env)
        except _NotLiteral as e:
            return e
        for concept, (_, test) in CONCEPTS.items():
            if _NAME[concept].fullmatch(name) and test(name, v):
                src = " ".join((ast.get_source_segment(self.text, value) or "").split())
                self.hits.append(Hit(f"{self.path}::{key}", concept, at.lineno,
                                     f"{name} = {src[:80] + '…' if len(src) > 80 else src}"))
        return v


def module_hits(path: str, text: str) -> list[Hit]:
    """Every restatement in the module `text` at `path`, its sources' included."""
    low = text.lower()
    if not _CANDIDATE.search(low) and not ("fraction as" in low and _CANDIDATE_ALIASED.search(low)):
        return []
    return _Module(path, text).hits


def scanned(files: list[str]) -> list[str]:
    out = []
    for f in files:
        parts = f.split("/")
        if (f.endswith(".py") and f.startswith(SCANNED) and not {"tests", "test", "fixtures"} & set(parts[:-1])
                and not parts[-1].startswith("test_") and parts[-1] != "conftest.py" and "vectors" not in parts[-1]):
            out.append(f)
    return out


def _read(f: str) -> str | None:
    """A tracked file's text; None for one deleted from the working tree."""
    return (ROOT / f).read_text(encoding="utf-8") if (ROOT / f).is_file() else None


def scan(files: list[str], read: Callable[[str], str | None] = _read) -> list[Hit]:
    return [h for f in scanned(files) if (text := read(f)) is not None for h in module_hits(f, text)]


@cache
def _tree() -> tuple[tuple[Hit, ...], frozenset[str]]:
    files = _tracked()
    return tuple(scan(files)), frozenset(files)


def module_of(path: str, tracked: frozenset[str]) -> str:
    """The import name of the module at `path`: its packages are the directories above it with an `__init__.py`."""
    parts = path.removesuffix(".py").split("/")
    i = len(parts) - 1
    while i > 0 and "/".join(parts[:i]) + "/__init__.py" in tracked:
        i -= 1
    return ".".join(parts[i:])


def sources(hits: list[Hit] | tuple[Hit, ...]) -> dict[str, list[str]]:
    """Each concept's canonical keys that state its literal: exactly one, for every concept."""
    keys = {h.key for h in hits}
    return {c: [k for k in CANONICAL[c] if k in keys] for c in CONCEPTS}


def restatements(hits, known: dict[str, str], tracked: frozenset[str]) -> list[str]:
    """For each hit neither canonical nor known: where it is, and the import that replaces it."""
    canon, srcs = {k for ks in CANONICAL.values() for k in ks}, sources(hits)
    out = []
    for h in hits:
        if h.key in canon or h.key in known:
            continue
        src = (srcs[h.concept] or list(CANONICAL[h.concept]))[0]
        path, name = src.split("::")
        out.append(f"{h.key.partition('::')[0]}:{h.line}: {h.text} restates {h.concept}: "
                   f"from {module_of(path, tracked)} import {name}{HINT.get(h.concept, '')} ({path})")
    return out


def test_no_constant_is_restated():
    hits, tracked = _tree()
    bad = restatements(hits, KNOWN, tracked)
    assert not bad, ("a constant restated as a literal: every constant is read from one source. Read it instead:\n  "
                     + "\n  ".join(bad) + f"\n(a copy that must stay goes in KNOWN in {Path(__file__).name} with its "
                     "reason, raising KNOWN_MAX in the same PR)")


def test_each_constant_has_one_source():
    bad = []
    for c, ks in sources(_tree()[0]).items():
        if not ks:
            bad.append(f"{c}: none of {', '.join(CANONICAL[c])} states it: name where it went in CANONICAL")
        elif len(ks) > 1:
            bad.append(f"{c}: {', '.join(ks)} all state it: one reads the other")
    assert not bad, "each constant is stated as a literal at exactly one of its CANONICAL sources:\n  " + "\n  ".join(bad)


def test_known_only_shrinks():
    assert len(KNOWN) == KNOWN_MAX, (
        f"KNOWN has {len(KNOWN)} lines and KNOWN_MAX is {KNOWN_MAX}: KNOWN only shrinks. Read the canonical source "
        f"instead of adding a line; after deleting lines, set KNOWN_MAX to {len(KNOWN)} "
        f"(`python {Path(__file__).relative_to(ROOT)} --prune` deletes the stale ones and does both)")


def test_known_reasons_name_their_pr():
    bad = [k for k, why in KNOWN.items() if "\n" in why or not re.search(r"#\d{3,}\b|\bno PR yet\b", why)]
    assert not bad, f'each KNOWN reason is one line naming the PR that removes it ("#NNNN") or saying "no PR yet": {bad}'


def test_the_scan_reads_each_form():
    """A literal restatement is found in each binding form; a read, another δ and a function's local are not."""
    src = '''
import hashlib
from fractions import Fraction as Q
from verity.core.protocols.pouw.credit import DELTA as WINDOW
from verity.ml.commitments import scheme

DELTA = 2.0 ** -40
OTHER_DELTA = 0.01
DELTA_BITS: int = 40
X_DELTA = Q(1, 1 << 40)
DIGEST_BYTES = hashlib.sha512().digest_size
ROOT_BYTES = scheme.DIGEST_BYTES["sha512"]
NODE_BYTES = DIGEST_BYTES
SHORT_DIGEST_BYTES = 32
DIGEST_BITS = 8 * 64
ADD, ADD64 = 31, 63
SHA512_ANDS = 80 * (7 * ADD64 + 64 + 64) + 64 * 3 * ADD64 + 8 * ADD64
STEP, SHA512 = 4_190, 58_120
PERIODS = {"week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}
MINUTE_PERIODS = (10_080, 43_200, 525_600)
NAMED_PERIODS = ("week", "month", "year")
CORE_PERIODS = dict(scheme.PERIODS)
DELTAS = (1e-2, 1e-6)
OWN_DELTAS = (0.5, 2.0 ** -20)
CAP_BITS = 1 << 23
DEFAULTS = {"p_evade": 2.0 ** -40, "share": 0.01}
if True:
    DEADLINE_NS = 500_000


class Encoding:
    cap_bits: int = 1 << 23
    delta = WINDOW

    def stated(self, store, *, delta: float = 2.0 ** -40, rounds=3):
        DELTA_NS = 500_000
        return DELTA_NS
'''
    assert [(h.key.partition("::")[2], h.line, h.text) for h in module_hits("benchmarks/x.py", src)] == [
        ("DELTA", 7, "DELTA = 2.0 ** -40"), ("DELTA_BITS", 9, "DELTA_BITS = 40"),
        ("X_DELTA", 10, "X_DELTA = Q(1, 1 << 40)"), ("DIGEST_BITS", 15, "DIGEST_BITS = 8 * 64"),
        ("SHA512_ANDS", 17, "SHA512_ANDS = 80 * (7 * ADD64 + 64 + 64) + 64 * 3 * ADD64 + 8 * ADD64"),
        ("SHA512", 18, "SHA512 = 58_120"),
        ("PERIODS", 19, 'PERIODS = {"week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}'),
        ("MINUTE_PERIODS", 20, "MINUTE_PERIODS = (10_080, 43_200, 525_600)"),
        ("NAMED_PERIODS", 21, 'NAMED_PERIODS = ("week", "month", "year")'), ("DELTAS", 23, "DELTAS = (1e-2, 1e-6)"),
        ("CAP_BITS", 25, "CAP_BITS = 1 << 23"), ("DEFAULTS[p_evade]", 26, "p_evade = 2.0 ** -40"),
        ("DEADLINE_NS", 28, "DEADLINE_NS = 500_000"), ("Encoding.cap_bits", 32, "cap_bits = 1 << 23"),
        ("Encoding.stated(delta=)", 35, "delta = 2.0 ** -40")]


def test_a_restatement_fails_and_an_import_passes():
    """On a toy tree: the copy fails, naming the import that replaces it; the read, the source and a test pass."""
    files = {
        "verity/__init__.py": "", "verity/core/__init__.py": "", "verity/core/protocols/__init__.py": "",
        "verity/core/protocols/pous/__init__.py": "", "verity/core/protocols/pous/audit.py": "DEADLINE_NS = 500_000\n",
        "benchmarks/pous/copy.py": "import math\n\nDELTA_NS = 500_000\n",
        "benchmarks/pous/read.py": "from verity.core.protocols.pous.audit import DEADLINE_NS\n\nLIMIT_NS = DEADLINE_NS\n",
        "benchmarks/pous/tests/test_copy.py": "DELTA_NS = 500_000\n",
        "verity/core/protocols/pous/lean/scripts/difftest_vectors.py": "DELTA_NS = 500_000\n",
        "archive/pous/copy.py": "DELTA_NS = 500_000\n",
    }
    hits = scan(list(files), files.get)
    assert [h.key for h in hits] == ["verity/core/protocols/pous/audit.py::DEADLINE_NS", "benchmarks/pous/copy.py::DELTA_NS"]
    assert restatements(hits, {}, frozenset(files)) == [
        "benchmarks/pous/copy.py:3: DELTA_NS = 500_000 restates PoUS's Δ, 500,000 ns: "
        "from verity.core.protocols.pous.audit import DEADLINE_NS (verity/core/protocols/pous/audit.py)"]
    assert restatements(hits, {"benchmarks/pous/copy.py::DELTA_NS": "no PR yet"}, frozenset(files)) == []
    assert sources(hits)["PoUS's Δ, 500,000 ns"] == ["verity/core/protocols/pous/audit.py::DEADLINE_NS"]


_KNOWN_OPEN, _KNOWN_CLOSE = "KNOWN: dict[str, str] = {\n", "\n}\n"
_ENTRY = re.compile(r'^    "([^"\n]+)":[ \n]+"(?:[^"\\\n]|\\.)*",?\n', re.M)
_EMPTY_HEADING = re.compile(r"^    #[^\n]*\n(?=    #|\Z)", re.M)
_KNOWN_MAX = re.compile(r"^KNOWN_MAX = \d+$", re.M)


def pruned(source: str, drop: set[str]) -> str:
    """This file's `source` with the ``KNOWN`` lines in `drop` (and headings left empty) deleted, and ``KNOWN_MAX`` set to
    the lines left."""
    head, _, rest = source.partition(_KNOWN_OPEN)
    block, _, tail = rest.partition(_KNOWN_CLOSE)
    block = _EMPTY_HEADING.sub("", _ENTRY.sub(lambda m: "" if m.group(1) in drop else m.group(0), block + "\n"))
    head = _KNOWN_MAX.sub(f"KNOWN_MAX = {len(_ENTRY.findall(block))}", head)
    return head + _KNOWN_OPEN + block.removesuffix("\n") + _KNOWN_CLOSE + tail


def test_prune_deletes_stale_lines():
    src = ('KNOWN_MAX = 3\nKNOWN: dict[str, str] = {\n    # a\n    "a::A": "#1234",\n    # b\n'
           '    "b::B":\n        "no PR yet",\n    "b::C": "#5678",\n}\n\ny = 2\n')
    assert pruned(src, set()) == src
    assert pruned(src, {"a::A", "b::B"}) == ('KNOWN_MAX = 1\nKNOWN: dict[str, str] = {\n    # b\n'
                                             '    "b::C": "#5678",\n}\n\ny = 2\n')


if __name__ == "__main__":
    hits, tracked = _tree()
    canon = {k for ks in CANONICAL.values() for k in ks}
    for h in sorted(hits, key=lambda h: (h.concept, h.key)):
        state = "canonical" if h.key in canon else "known" if h.key in KNOWN else "NEW"
        print(f"{state:9} {h.concept:44} {h.key.partition('::')[0]}:{h.line}  {h.key.partition('::')[2]}  [{h.text}]")
    stale = sorted(set(KNOWN) - {h.key for h in hits})
    print(f"\n{len(KNOWN)} known, KNOWN_MAX {KNOWN_MAX}; stale (delete with --prune): {len(stale)}")
    for k in stale:
        print(f"  {k}")
    if "--prune" in sys.argv[1:]:
        me = Path(__file__)
        me.write_text(pruned(me.read_text(encoding="utf-8"), set(stale)), encoding="utf-8")
        print(f"pruned {len(stale)} lines from {me.relative_to(ROOT)}")
