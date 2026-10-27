"""Ruff's findings on every tracked Python file outside ``archive/`` that the vLLM integration's ruff lint doesn't hold, held to
``allowlists/repo_ruff.json``: ``ruff check`` under the root ``ruff.toml``, with the ruff the root ``dev`` group pins.

The integration's lint (``integrations/vllm/tests/lint/test_ruff.py``) holds ``integrations/vllm`` and ``verity/ml/definitions``
to its own list, and ``archive/`` is kept but not developed (``archive/AGENTS.md``), so neither is held here (``LEFT_OUT``); this
lint holds the rest of ``git ls-files``' ``*.py`` and ``*.pyi``.  Findings are counted per ``(file, rule)``: the file relative
to the repository and the rule's code, the key the ratchet gate counts by too, so a refactor that moves a finding within its
file changes nothing here.  An entry ``{"file", "rule", "count"}`` allows ``count`` findings of its pair.  A pair found more
times than its entry allows, or found with no entry, fails.  An entry that allows more than is found doesn't: the list can run
ahead of a tree, holding the findings of PRs queued below it on ``next`` that a fix, an ejection or a reorder leaves out, and
``--prune`` deletes and lowers what this tree no longer finds.  ``ruff check --fix PATH`` fixes some (F401, F541, RUF100's
stale ``# noqa``).

A file that moves takes its entries with it, in the same change (``--move``): this lint reads no history, so a renamed file's
findings count under its new path.  The ratchet gate (``integrations/vllm/tools/ratchet_gate.py``) refuses a rename that
leaves its entries behind, and a change that grows the list or the findings of the Python it changes, and judges a change to
this module, to ``ruff.toml`` or to the ruff pin by running this lint as the change's base has it on the change's tree.  So
it imports only the standard library, ``test_no_new_findings`` takes no fixture, and the gate reads ``LEFT_OUT`` from the
base's copy of this module.

    python3 tests/lint/test_repo_ruff.py --move OLD NEW   # a file (or directory) moved: its entries follow it
    python3 tests/lint/test_repo_ruff.py --prune          # delete and lower what this tree no longer finds: the list only shrinks
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

NAME = "repo_ruff"
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ruff.toml"
LIST = Path(__file__).resolve().parent / "allowlists" / f"{NAME}.json"
#: what the integration's ruff lint holds to its own list (its `ROOT` and `DEFINITIONS`), and the archive
LEFT_OUT = ("integrations/vllm/", "verity/ml/definitions/", "archive/")
FIELDS = ("file", "rule", "count")
FIX = ("Fix the code (`ruff check --fix PATH` fixes some), or, where the rule is wrong for that line, say why in a "
       "`# noqa: CODE` on it. If the file moved, move its entries in the same change: "
       "`python3 tests/lint/test_repo_ruff.py --move OLD NEW`.")


def files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", "*.py", "*.pyi"], cwd=ROOT, capture_output=True, check=True).stdout
    return sorted(p for p in out.decode().split("\0") if p and not p.startswith(LEFT_OUT))


@lru_cache(maxsize=None)
def scan() -> tuple[tuple[tuple[str, str], int, str], ...]:
    """Each finding's (file, rule), line and message."""
    cmd = [sys.executable, "-m", "ruff", "check", "--config", str(CONFIG), "--no-cache", "--exit-zero", "--output-format",
           "json", "--", *files()]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, (f"ruff check exited {r.returncode} (no ruff? `uv sync --all-packages`: the root `dev` group pins "
                               f"it):\n{r.stderr[-2000:]}")
    return tuple(((Path(f["filename"]).relative_to(ROOT).as_posix(), f["code"] or "syntax-error"), f["location"]["row"],
                  f["message"]) for f in json.loads(r.stdout))


def allowed() -> Counter:
    doc = json.loads(LIST.read_text(encoding="utf-8"))
    assert isinstance(doc, dict) and isinstance(doc.get("entries"), list), f"{LIST.name}: {{'entries': [...]}} expected"
    out: Counter = Counter()
    for e in doc["entries"]:
        assert isinstance(e, dict) and set(e) == set(FIELDS), f"{LIST.name}: an entry is {{{', '.join(FIELDS)}}}: {e}"
        assert isinstance(e["count"], int) and e["count"] >= 1, f"{LIST.name}: count must be a positive int: {e}"
        k = (e["file"], e["rule"])
        assert k not in out, f"{LIST.name}: duplicate entry {e}"
        out[k] = e["count"]
    return out


def test_no_new_findings() -> None:
    found, have, sites = Counter(k for k, *_ in scan()), allowed(), {}
    for k, line, message in scan():
        sites.setdefault(k, []).append((line, message))
    new = [f"  {k[0]} [{k[1]}]: {n} found, {have[k]} allowed\n"
           + "\n".join(f"    {k[0]}:{line} {message}" for line, message in sorted(sites[k]))
           for k, n in sorted(found.items()) if n > have[k]]
    assert not new, (f"ruff (ruff.toml): {len(new)} (file, rule) pair(s) found more often than tests/lint/allowlists/{NAME}.json "
                     f"allows. {FIX}\n" + "\n".join(new))


def moved(entries: Counter, old: str, new: str) -> Counter:
    """``entries`` with the file ``old``'s, or the directory ``old``'s files', under ``new``: a pair already there adds."""
    old, new = old.rstrip("/"), new.rstrip("/")
    out: Counter = Counter()
    for (f, rule), n in entries.items():
        out[(new + f[len(old):] if f == old or f.startswith(f"{old}/") else f, rule)] += n
    return out


def dump(entries: Counter) -> None:
    lines = [f"    {json.dumps(dict(zip(FIELDS, (*k, n))), ensure_ascii=False)}" for k, n in sorted(entries.items()) if n > 0]
    LIST.write_text('{\n  "rule": "ruff (ruff.toml)",\n  "entries": [\n' + ",\n".join(lines) + "\n  ]\n}\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--move"] and len(sys.argv) == 4:
        have = allowed()
        dump(after := moved(have, *sys.argv[2:]))
        print(f"{sum(1 for k in have if k not in after)} entries moved from {sys.argv[2]} to {sys.argv[3]}")
    elif sys.argv[1:] == ["--prune"]:
        dump(allowed() & Counter(k for k, *_ in scan()))
    else:
        sys.exit(__doc__.rsplit("\n\n", 1)[-1])
