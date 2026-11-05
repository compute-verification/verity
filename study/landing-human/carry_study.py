"""Replay `research.carry` over every re-verdict of a window: each land verdict at a PR's new head when an older head of that PR
already had one.  For each, the verdict `carry` would have given with no reviewer, judged as of the verdict's time: a merged
parent is trusted when it is an ancestor of `main`'s or `checked`'s tip then (GitHub's push activity) or of another PR's head
that had a land verdict by then.

    uv run --no-sync python study/landing-human/carry_study.py LABELS.json ACT_DIR OUT.json [--jobs N]

LABELS.json: every `pr:*` label (`labels.py`); ACT_DIR: `act_main.tsv` and `act_checked.tsv` (`gh api .../activity`).  The
clone needs every head (`git fetch origin '+refs/pull/*/head:refs/pr/*'`), a ref `refs/rv/PR/HEAD` per reviewed head, and a
commit-graph (`git commit-graph write --reachable`) for `for-each-ref --contains` to answer in milliseconds."""

from __future__ import annotations

import bisect
import collections
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from research import carry

REPO = Path(__file__).resolve().parents[2]
W0 = "2026-10-22T17:00:00Z"


def git(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)


def tips(act_dir: Path) -> dict[str, tuple[list[str], list[str]]]:
    out = {}
    for ref in ("main", "checked"):
        rows = sorted(ln.split("\t") for ln in (act_dir / f"act_{ref}.tsv").read_text().splitlines() if ln)
        out[ref] = ([r[0] for r in rows], [r[3] for r in rows])
    return out


def tip_at(t: dict, ref: str, ts: str) -> str | None:
    times, afters = t[ref]
    i = bisect.bisect_right(times, ts) - 1
    return afters[i] if i >= 0 else None


def pairs(labels: list[dict]) -> list[dict]:
    land = collections.defaultdict(list)
    marks = collections.defaultdict(list)
    for r in labels:
        if r["key"] == "reviewed" and r["value"] == "land":
            land[r["pr"]].append(r)
        if r["key"] == "ready" and r["value"] is True:
            marks[(r["pr"], r["head"])].append(r["ts"])
    out = []
    for pr, vs in land.items():
        vs.sort(key=lambda r: r["ts"])
        seen: list[tuple[str, str]] = []
        for r in vs:
            if r["head"] in (h for h, _ in seen):
                continue
            if seen and r["ts"] >= W0:
                m = [x for x in marks.get((pr, r["head"]), []) if x <= r["ts"]]
                out.append({"pr": pr, "old": seen[-1][0], "old_ts": seen[-1][1], "new": r["head"], "ts": r["ts"], "by": r["by"],
                            "ref": r["ref"], "mark": min(m) if m else None})
            seen.append((r["head"], r["ts"]))
    return out


_T: dict = {}
_FIRST: dict = {}
_EXCLUDES: list = []


def _init(t: dict, first: dict, excludes: list) -> None:
    _T.update(t)
    _FIRST.update(first)
    _EXCLUDES.extend(excludes)


def trusted_at(ts: str, pr: int):
    landed = [x for x in (tip_at(_T, "main", ts), tip_at(_T, "checked", ts)) if x]

    def trusted(c: str) -> bool:
        if any(c == x or git("merge-base", "--is-ancestor", c, x).returncode == 0 for x in landed):
            return True
        for ref in git("for-each-ref", "--contains", c, "refs/rv/", "--format=%(refname)").stdout.split():
            _, _, n, h = ref.split("/")
            if int(n) != pr and _FIRST.get(f"{n}@{h}", "~") <= ts:
                return True
        return False

    return trusted


def judge(p: dict) -> dict:
    from research.queue import lowrisk

    try:
        strict = carry.delta(REPO, p["old"], p["new"], trusted_at(p["ts"], p["pr"]))
        wide = carry.delta(REPO, p["old"], p["new"], trusted_at(p["ts"], p["pr"]), lowrisk=lambda fs: lowrisk(fs, _EXCLUDES))
        out = {**p, **strict.as_dict(), "wide": wide.as_dict()}
        if strict.kind == "untrusted-parent":
            out["later"] = carry.delta(REPO, p["old"], p["new"], trusted_at("2026-11-05T17:30:00Z", p["pr"])).as_dict()
        return out
    except Exception as e:  # noqa: BLE001
        return {**p, "kind": "error", "carries": False, "why": f"{type(e).__name__}: {e}"[:300]}


def main() -> None:
    labels_path, act_dir, out_path = sys.argv[1:4]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    labels = json.load(open(labels_path))
    first: dict[str, str] = {}
    for r in labels:
        if r["key"] == "reviewed" and r["value"] == "land":
            k = f"{r['pr']}@{r['head']}"
            first[k] = min(first.get(k, "~"), r["ts"])
    ps = pairs(labels)
    import tomllib
    excludes = tomllib.loads(git("show", "origin/main:tools/verity/check/drive.toml").stdout)["lowrisk_excludes"]
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(tips(Path(act_dir)), first, excludes)) as ex:
        res = list(ex.map(judge, ps, chunksize=8))
    json.dump(res, open(out_path, "w"), indent=0)
    c = collections.Counter(r["kind"] for r in res)
    w = collections.Counter(r.get("wide", {}).get("kind") for r in res)
    print(len(res), "re-verdicts;", sum(r["carries"] for r in res), "carry (strict);", dict(c.most_common()))
    print(sum(r.get("wide", {}).get("carries", False) for r in res), "carry (with low-risk deltas);", dict(w.most_common()))
    print(sum(r.get("later", {}).get("carries", False) for r in res), "untrusted-parent ones that carry once their parents are reviewed")


if __name__ == "__main__":
    main()
