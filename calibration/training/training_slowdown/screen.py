"""The screen of a route's source: the names a route may not contain, refused before the route is imported
(`config.route`) and, when the job ends, in every module it imported outside the fixed side (`run.Imports`). It
reads source text and imports the standard library alone."""
from __future__ import annotations

import io
import re
import tokenize

#: names a route may not contain: the harness's internals, dispatch-mode control, ways to GPU or host kernels the
#: counter doesn't see, threads and processes, signals (a job told to stop is run again from its kept state, so a
#: route that could stop its own would run steps again that the record counts once), and code built from strings.
#: `torch.compile` is allowed: TORCHDYNAMO_DISABLE, set before torch is imported, makes it return the eager module,
#: and `_dynamo` (which could turn it back on) is refused. A screen of source text is a first filter only: the ways
#: it names are caught as they run where they can be (flops.py).
FORBIDDEN = (r"\.counter\b", r"\.pause\b", r"\buncounted\b", r"\btraining_slowdown\.(?!(codecs|model)\b)",
             r"(?m)from\s+training_slowdown\s+import\s+(?!(codecs|model)\s*$)", r"\bAdapter\b", "_python_dispatch", "_pop_mode", "_disable_current_modes", "TorchDispatchMode",
             "_set_dispatch_mode", "CUDAGraph", r"cuda\.graph", "_dynamo", "TORCHDYNAMO", "triton", "ctypes", "cupy",
             "numba", "jax", r"torch\.utils\.cpp_extension", "load_inline", "subprocess", "multiprocessing",
             r"os\.system", r"os\.exec", r"os\.spawn", r"os\.posix_spawn", r"\bos\.fork", r"\bos\.popen\b", r"\bpty\b",
             "_posixsubprocess", r"\bthreading\b", r"\b_thread\b", r"\bconcurrent\b", r"\btorch\.library\b",
             "TensorBase", "addaudithook", r"sys\.modules", r"\bgc\b\.", "__import__", r"importlib", r"\bbuiltins\b",
             "__builtins__", "__subclasses__", r"(?<![.\w])(exec|eval|compile)\s*\(", r"\bsys\.monitoring\b",
             r"(?m)^\s*from\s+sys\s+import\b.*\bmonitoring\b", r"\btorch\._C\b", r"(?m)^\s*from\s+torch\s+import\b.*\b_C\b",
             r"__bases?__", "__mro__", "__closure__", "cell_contents", r"\bpyarrow\b", r"\bpolars\b", r"\bduckdb\b",
             r"(?m)^\s*(import|from)\s+datasets\b", r"\brunpy\b", r"\bmarshal\b", r"\b_?c?pickle\b", r"\bdill\b",
             r"\bshelve\b", r"\bzipimport\b", "FunctionType", "CodeType", "__code__", r"\bmeta_path\b",
             r"\bpath_hooks\b", r"weights_only\s*=\s*False", r"\bos\.kill", r"\bsignal\b")
#: names that run source or code built at run time: refused wherever a route names them, called or not (`f = exec`),
#: except as another object's attribute (`re.compile`, `torch.compile`, `model.eval`)
FORBIDDEN_NAMES = frozenset({"exec", "eval", "compile", "execfile", "breakpoint"})


def _code(source: str) -> str:
    """`source` without its comments and docstrings (statements that are only a string), which never run."""
    toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    starts = (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT)
    kept = [t for i, t in enumerate(toks) if t.type != tokenize.COMMENT and not (
        t.type == tokenize.STRING and (i == 0 or toks[i - 1].type in starts)
        and toks[i + 1].type in (tokenize.NEWLINE, tokenize.ENDMARKER))]
    return tokenize.untokenize(kept)


def _names(code: str) -> set[str]:
    """The bare names `code` uses: NAME tokens not after a `.`."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, SyntaxError):
        return set(re.findall(r"(?<![.\w])\w+", code))
    return {t.string for i, t in enumerate(toks) if t.type == tokenize.NAME and not (i and toks[i - 1].string == ".")}


def screen(source: str) -> list[str]:
    try:
        code = _code(source)
    except (tokenize.TokenError, SyntaxError, ValueError) as e:
        return [f"source that doesn't tokenize ({type(e).__name__})"]
    return [pat for pat in FORBIDDEN if re.search(pat, code)] + sorted(FORBIDDEN_NAMES & _names(code))
