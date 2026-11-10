"""The tree a list of `layout.py` maps leaves, computed from the maps' rows and a commit's tree without moving anything:
what `suite_check.py` and `gone_dirs.py` read. The maps apply in the order given, and each names the paths of the commit
(as `layout_maps.py`'s maps do), so a file goes where the first map with a row for it sends it."""
from __future__ import annotations

import posixpath
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layout as L  # noqa: E402


class Moved:
    def __init__(self, maps: list[str], ref: str = "origin/main", repo: Path | None = None):
        self.repo = repo or HERE.parent.parent
        self.ref = self._git("rev-parse", "--verify", f"{ref}^{{commit}}").strip()
        self.files = [p for p in self._git("ls-tree", "-r", "--name-only", "-z", self.ref).split("\0") if p]
        self.tracked = set(self.files)
        self.layouts = [L.Layout(tomllib.loads(Path(m).read_text())) for m in maps]
        self.new = {p: self._file(p) for p in self.files}
        self.old_dirs = _dirs(self.files)
        self.new_dirs = _dirs(self.new.values())
        self._cat = None

    def _git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=self.repo, check=True, capture_output=True, text=True).stdout

    def _file(self, p: str) -> str:
        return next((n for lay in self.layouts if (n := lay.file(p)) is not None), p)

    def directory(self, d: str) -> str | None:
        """Where a reference to old directory `d` reads after the maps: None when no map has a row or a `[tokens]` row for it."""
        return next((n for lay in self.layouts if (n := lay.directory(d)[0]) is not None), None)

    def excluded(self, p: str) -> bool:
        """Whether no map rewrites file `p` (`[rewrite] exclude`)."""
        return all(lay.excluded(p) for lay in self.layouts) if self.layouts else False

    def read(self, p: str) -> str | None:
        """The commit's text of `p`; None for a file that is not UTF-8 text."""
        if self._cat is None:
            self._cat = subprocess.Popen(["git", "cat-file", "--batch"], cwd=self.repo, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE)
        self._cat.stdin.write(f"{self.ref}:{p}\n".encode())
        self._cat.stdin.flush()
        head = self._cat.stdout.readline().split()
        if len(head) < 3:
            return None
        data = self._cat.stdout.read(int(head[2]) + 1)[:-1]
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def close(self) -> None:
        if self._cat is not None:
            self._cat.stdin.close()
            self._cat.wait()
            self._cat = None


def _dirs(paths) -> set[str]:
    out: set[str] = set()
    for p in paths:
        d = posixpath.dirname(p)
        while d and d not in out:
            out.add(d)
            d = posixpath.dirname(d)
    return out


def under(p: str, d: str) -> bool:
    return d in ("", ".") or p == d or p.startswith(d.rstrip("/") + "/")
