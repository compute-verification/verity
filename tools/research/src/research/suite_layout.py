"""Where a suite's tests are, which suite a path under `tests/` belongs to and what a suite is called: the one rule that
`tools/check/suites.py` (`discover`, `inputs_of`, `select`), the queue (`landing.suite_name`, `vqueue_slim.suite_reads`,
`queue.lowrisk`), `suite_size` and the graph's test-file family read.

The root's `tests/` mirrors the tree by path (`MIRROR`): the tests of `X` are at `tests/X`.  A suite there is `X`'s: it keeps
the name `X` gave it (`suite_name`), and an input or dependency directory `X` covers `tests/X` too (`mirrors`).  The rest of
`tests/` is the root suite's own (`root_owns`): its files there and its directories `ROOT_OWN`, which mirror no directory of
the tree.  The root suite is `repository`, and `tests` names it too (`names`).

Stdlib only, and no I/O: each caller hands it what it read, from a checkout or from a commit it never checked out.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping

#: the root's directory that mirrors the tree by path: the tests of `X` are at `tests/X`
MIRROR = "tests"
#: the root suite's name
ROOT_SUITE = "repository"
#: the root suite's own directories under `MIRROR`, beside its files there: each mirrors no directory of the tree
#: (`tests/test_lowrisk_excludes.py` holds `tests/` to it)
ROOT_OWN = ("lint",)


def norm(name: str) -> str:
    """A distribution's name as the suites name it: runs of `-`, `_` and `.` as one `-`, lower case."""
    return re.sub(r"[-_.]+", "-", name).lower()


def mirror_of(path: str) -> str | None:
    """The path a path under the root's `tests/` mirrors (`tests/X/...` -> `X/...`), or None."""
    return path[len(MIRROR) + 1:] if path.startswith(MIRROR + "/") else None


def root_owns(path: str) -> bool:
    """Whether ``path`` is the root suite's own under `tests/`: a file of `tests/` itself or under a `ROOT_OWN` directory, or
    under `tests/tests/`, which would mirror `tests/` itself and which the root suite collects."""
    rest = mirror_of(path)
    return rest is not None and ("/" not in rest or rest.split("/", 1)[0] in (*ROOT_OWN, MIRROR))


def stands_for(path: str) -> str | None:
    """The path of the tree whose tests ``path`` is (`tests/X/...` -> `X/...`), or None for the root suite's own and for a
    path outside `tests/`."""
    return None if root_owns(path) else mirror_of(path)


def tested(path: str) -> tuple[str, ...]:
    """The directories a suite at ``path`` under `tests/` holds the tests of, as `select` and the check's tables name it:
    `tests/X` and `tests/X/tests` hold `X`'s. None elsewhere."""
    src = mirror_of(path)
    return () if src is None else tuple(dict.fromkeys((src, src.removesuffix("/tests"))))


def member_dirs(pyproject: Callable[[str], Mapping]) -> dict[str, str]:
    """{directory: normalized distribution name} of the workspace's members but the root, from ``pyproject`` (a directory's
    pyproject.toml, read as a mapping, {} where it has none)."""
    members = pyproject(".").get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])
    return {d.strip("/"): norm(n) for d in members if isinstance(n := pyproject(d.strip("/")).get("project", {}).get("name"), str)}


def suite_name(path: str, project: Mapping, dirs: Mapping[str, str]) -> str:
    """The name of the suite at ``path`` whose pyproject.toml's `[project]` is ``project``: `ROOT_SUITE` for the root's, else its
    distribution's, else its path.  A suite at `tests/X` is named as `X` named it: by its path `X`; and the virtual member
    `N-tests` holding the tests of the member N, at `X/tests` or (mirrored) `tests/X` or `tests/X/tests`, where `X` is N's
    directory (``dirs``: `member_dirs`), by N."""
    path = path.strip("/") or "."
    if path == ".":
        return ROOT_SUITE
    if not isinstance(raw := project.get("name"), str):
        return mirror_of(path) or path
    name = norm(raw)
    if name.endswith("-tests"):
        for d in tested(path) or (path.removesuffix("/tests"),):
            if d != path and dirs.get(d) == name.removesuffix("-tests"):
                return dirs[d]
    return name


def names(path: str, name: str) -> tuple[str, ...]:
    """What names the suite at ``path`` called ``name`` exactly, as `select` reads a name: its name, its path, the directories
    it holds the tests of (`tested`), and for the root's, `tests` and `tests/`."""
    return (name, path, path.rstrip("/"), *tested(path), *((MIRROR, MIRROR + "/") if path == "." else ()))


def mirrors(paths: Iterable[str], exists: Callable[[str], bool]) -> set[str]:
    """What the ``paths`` (from the root) cover beside themselves: `tests/X` for each `X` but the root and the paths already
    under `tests/`, where it ``exists``."""
    tops = {p.strip("/") for p in paths} - {"", "."}
    return {m for p in tops if mirror_of(p) is None and exists(m := f"{MIRROR}/{p}")}


def held(path: str, dirs: tuple[str, ...]) -> bool:
    """Whether one of the directories ``dirs`` (each from the root, ending in `/`) holds ``path``, the mirror read in: `X/`
    holds `tests/X/` too (`stands_for`), and `tests/` holds only the root suite's own (`root_owns`)."""
    if path.startswith(rest := tuple(d for d in dirs if d != MIRROR + "/")):
        return True
    if root_owns(path):
        return MIRROR + "/" in dirs
    return (src := mirror_of(path)) is not None and src.startswith(rest)
