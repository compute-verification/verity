"""Every `ready` and `reviewed` label on every head of the given PRs, from the store's remote, one listing per PR.

    uv run --no-sync python labels.py OUT_JSON PR...

Writes {pr: [label doc, ...]} (each doc as stored: target pr:N@SHA, key, value, by, ts, ref).  Read only.
"""
from __future__ import annotations

import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from research.store.local import LocalStore


def main(out: str, *prs: str) -> None:
    remote = LocalStore().remote
    tmp = Path(tempfile.mkdtemp())

    def one(n: str) -> tuple[int, list[dict]]:
        docs = []
        for i, k in enumerate(remote.list(f"labels/pr_{n}@")):
            p = tmp / f"{n}-{i}.json"
            remote.get(k, p)
            docs.append(json.loads(p.read_text()))
            p.unlink()
        return int(n), docs

    with ThreadPoolExecutor(32) as ex:
        got = dict(ex.map(one, prs))
    Path(out).write_text(json.dumps(got))
    print(f"{len(got)} PRs, {sum(map(len, got.values()))} labels")


if __name__ == "__main__":
    main(*sys.argv[1:])
