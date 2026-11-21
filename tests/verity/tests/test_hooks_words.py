"""`verity.hooks` speaks no serving stack's words: its identifiers, string constants and the ``literals`` of its docstrings
hold none of `REFUSED` but the ones `KEPT` names, a list that only shrinks."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from verity import hooks as H

#: a serving stack's words (vLLM, an HTTP server, PyTorch, CUDA), matched as whole parts of a snake_case name, plural too
REFUSED = ("http", "route", "header", "first_line", "body", "server", "path", "vllm", "torch", "cuda", "gpu", "kv",
           "scheduler", "token", "prompt", "slot", "linear", "async", "in_process", "eager", "stream", "request")
#: the hooks' own Request (the type of a scheme's requests, and `unmet`'s argument of them), not an HTTP request
HOOKS_REQUEST = frozenset({"Request", "requests"})

#: each word kept today, by the name or literal that holds it: where it is, who reads it, and the neutral word for it
KEPT = {
    "ServeRoute": "a request that the server answer a path; network's Admitter files it for its watermark, the vLLM "
                  "frontend serves it; neutral: the scheme names an external input port, the adapter maps it to a path",
    "Route": "ServeRoute's handler (first line, headers, body -> status, text); neutral: a port's handler, bytes in, "
             "a status and text out",
    "path": "ServeRoute.path, the URL path; neutral: the port's name",
    "first_line": "InputArrived.first_line and Route's argument, the HTTP request line; network's Admitter reads it; "
                  "neutral: drop it, or an opaque byte string arriving with the input",
    "headers": "InputArrived.headers and Route's argument; network's Admitter reads the release headers (release_of); "
               "neutral: the input's release, an opaque byte string a named port carries",
    "body": "Route's argument, the request body as it streams in; neutral: the port's bytes",
    "slots": "StepBegan.slots (the scheduler's running slots; vLLM's admission fills it, network's TickGate reads it) "
             "and RangesDeclared.slots (a KV block's token slots, PoUB's binder); neutral: capacity, and a range's words",
    "slot_bytes": "RangesDeclared.slot_bytes, a KV slot's bytes (PoUB's binder); neutral: word_bytes",
    "slot": "ValueProduced.slot, the value's position in its port (the Commit's taps); neutral: index",
    "eager": "a Declared property (no compiled graph); PoUW and PoUS require it (their adapter wraps aren't "
             "graph-capturable); neutral: a property a scheme names, its value the adapter's",
    "async_steps": "a Declared property, a step scheduled before the last one's outputs are read (vLLM's async "
                   "scheduling; PoUB requires it off); neutral: overlapped_steps",
}


def _snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).replace("-", "_").lower()


def _names(source: str) -> set[str]:
    """Every identifier, string constant and docstring ``literal`` of ``source``; a docstring's prose is no name."""
    tree, docs, out = ast.parse(source), set(), set()
    for node in ast.walk(tree):
        for stmt in getattr(node, "body", None) if isinstance(getattr(node, "body", None), list) else ():
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                docs.add(id(stmt.value))
                out |= set(re.findall(r"``([^`]+)``", stmt.value.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.arg, ast.keyword)) and node.arg:
            out.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            out.add(node.value)
    return out


def stack_words(source: str) -> dict[str, list[str]]:
    """Each name of ``source`` that holds a refused word, with the words it holds."""
    found = {}
    for name in _names(source) - HOOKS_REQUEST:
        if words := [w for w in REFUSED if re.search(rf"(^|_){w}s?(_|$)", _snake(name))]:
            found[name] = words
    return found


def test_hooks_hold_no_serving_stack_word_but_the_kept() -> None:
    found = stack_words(Path(H.__file__).read_text())
    new = {n: w for n, w in found.items() if n not in KEPT}
    assert not new, (f"verity/hooks.py names a serving stack's words {new}: say it in the circuit's words (a workload, "
                     f"an input, a step, a value, a block, a range, an output, a property, a scheme) and leave the "
                     f"stack's to the adapter")
    gone = sorted(set(KEPT) - set(found))
    assert not gone, f"verity/hooks.py no longer holds {gone}: delete them from KEPT, which only shrinks"


def test_the_scan_tells_a_hooks_request_from_an_http_one() -> None:
    source = 'type Request = int\ndef unmet(requests): """``server_request`` prose about a server."""\nX = "http-body"'
    assert stack_words(source) == {"server_request": ["server", "request"], "http-body": ["http", "body"]}
