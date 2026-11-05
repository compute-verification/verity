"""Replay `research.carry` over every re-verdict of a window: each land verdict at a PR's new head when an older head of that PR
already had one.  For each, the verdict `carry` would have given with no reviewer, judged as of the verdict's time: a merged
parent is trusted when it is an ancestor of `main`'s or `checked`'s tip then (GitHub's push activity) or of another PR's
current marked head whose verdicts then landed it (each reviewer's newest; a hold by anyone keeps it).  ``loose`` repeats it
trusting any head the verdicts land, marked or not, to show what the mark requirement removes.

    uv run --no-sync python study/landing-human/carry_study.py LABELS.json ACT_DIR OUT.json [--jobs N] [--marks | --holds]

--marks judges every re-head instead (each PR's consecutive marked heads, as of the newer one's mark); --holds each head whose
first verdict was a hold, over the PR's last land-verdict head (the false-positive test).

LABELS.json: every `pr:*` label (`labels.py`); ACT_DIR: `act_main.tsv` and `act_checked.tsv` (`gh api .../activity`).  The
clone needs every head (`git fetch origin '+refs/pull/*/head:refs/pr/*'`), a ref `refs/rv/PR/HEAD` per reviewed head, and a
commit-graph (`git commit-graph write --reachable`) for `for-each-ref --contains` to answer in milliseconds."""

from __future__ import annotations

import bisect
import collections
import functools
import json
import subprocess
import sys
import tomllib
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


def mark_pairs(labels: list[dict]) -> list[dict]:
    """Each PR's consecutive marked heads (by first mark), the newer marked in the window: every re-head, reviewed or not,
    judged as of the newer head's mark."""
    marks: dict[int, dict[str, str]] = collections.defaultdict(dict)
    for r in labels:
        if r["key"] == "ready" and r["value"] is True and r["ts"] >= "2026-10-12T06:00:00Z":
            h = marks[r["pr"]]
            h[r["head"]] = min(h.get(r["head"], "~"), r["ts"])
    out = []
    for pr, hs in marks.items():
        seq = sorted(hs.items(), key=lambda kv: kv[1])
        for (old, ots), (new, nts) in zip(seq, seq[1:]):
            if nts >= W0:
                out.append({"pr": pr, "old": old, "old_ts": ots, "new": new, "ts": nts, "by": None, "ref": None, "mark": nts})
    return out


def hold_pairs(labels: list[dict]) -> list[dict]:
    """The false-positive test: each head whose first verdict was a hold, of a PR with a land verdict at an older head, judged
    as of the hold.  One that carries is a head `carry` would have landed that a reviewer held."""
    verdicts = collections.defaultdict(list)
    for r in labels:
        if r["key"] == "reviewed" and r["value"] in ("land", "hold"):
            verdicts[r["pr"]].append(r)
    out = []
    for pr, vs in verdicts.items():
        vs.sort(key=lambda r: r["ts"])
        landed: list[str] = []
        seen: set[str] = set()
        for r in vs:
            if r["head"] in seen:
                continue
            seen.add(r["head"])
            if r["value"] == "hold" and landed and r["ts"] >= W0:
                out.append({"pr": pr, "old": landed[-1], "new": r["head"], "ts": r["ts"], "by": r["by"], "ref": r["ref"],
                            "mark": None})
            if r["value"] == "land":
                landed.append(r["head"])
    return out


_T: dict = {}
_VERDICTS: dict = {}
_MARKS: dict = {}
_EXCLUDES: list = []


def _init(t: dict, verdicts: dict, marks: dict, excludes: list) -> None:
    _T.update(t)
    _VERDICTS.update(verdicts)
    _MARKS.update(marks)
    _EXCLUDES.extend(excludes)


@functools.lru_cache(maxsize=None)
def _contains(c: str) -> tuple[str, ...]:
    return tuple(git("for-each-ref", "--contains", c, "refs/rv/", "--format=%(refname)").stdout.split())


@functools.lru_cache(maxsize=None)
def _ancestor(c: str, x: str) -> bool:
    return c == x or git("merge-base", "--is-ancestor", c, x).returncode == 0


def landable_at(key: str, ts: str) -> bool:
    """Whether a head's verdicts as of ``ts`` land it: each reviewer's newest counts, a hold by anyone keeps it."""
    newest: dict[str, str] = {}
    for vts, by, value in _VERDICTS.get(key, ()):
        if vts <= ts:
            newest[by] = value
    return "land" in newest.values() and "hold" not in newest.values()


def marked_at(pr: int, head: str, ts: str) -> bool:
    """Whether ``head`` is, as of ``ts``, the PR's head drive would land: its newest ready label is a mark, and no other head
    of the PR was marked after it."""
    state: dict[str, tuple[str, bool]] = {}
    for mts, h, value in _MARKS.get(pr, ()):
        if mts <= ts:
            state[h] = (mts, value)
    if head not in state or not state[head][1]:
        return False
    return all(not v or t <= state[head][0] for h, (t, v) in state.items() if h != head)


def trusted_at(ts: str, pr: int, loose: bool = False):
    """A commit is trusted when it is landed (an ancestor of `main`'s or `checked`'s tip) or an ancestor of another PR's
    current marked head that its verdicts land; ``loose`` takes any head the verdicts land, marked or not."""
    landed = [x for x in (tip_at(_T, "main", ts), tip_at(_T, "checked", ts)) if x]

    def trusted(c: str) -> bool:
        if any(_ancestor(c, x) for x in landed):
            return True
        for ref in _contains(c):
            _, _, n, h = ref.split("/")
            if int(n) != pr and landable_at(f"{n}@{h}", ts) and (loose or marked_at(int(n), h, ts)):
                return True
        return False

    return trusted


def judge(p: dict) -> dict:
    from research.queue import lowrisk

    try:
        strict = carry.delta(REPO, p["old"], p["new"], trusted_at(p["ts"], p["pr"]))
        out = {**p, **strict.as_dict(), "wide": strict.as_dict(), "loose": strict.as_dict()}
        if strict.carries:
            return out
        out["wide"] = carry.delta(REPO, p["old"], p["new"], trusted_at(p["ts"], p["pr"]),
                                  lowrisk=lambda fs: lowrisk(fs, _EXCLUDES)).as_dict()
        if strict.kind == "untrusted-parent":
            out["loose"] = carry.delta(REPO, p["old"], p["new"], trusted_at(p["ts"], p["pr"], loose=True)).as_dict()
            out["later"] = carry.delta(REPO, p["old"], p["new"], trusted_at("2026-11-05T17:30:00Z", p["pr"])).as_dict()
        return out
    except Exception as e:  # noqa: BLE001
        return {**p, "kind": "error", "carries": False, "why": f"{type(e).__name__}: {e}"[:300]}


def main() -> None:
    labels_path, act_dir, out_path = sys.argv[1:4]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    labels = json.load(open(labels_path))
    verdicts: dict[str, list] = collections.defaultdict(list)
    for r in sorted(labels, key=lambda r: r["ts"]):
        if r["key"] == "reviewed" and r["value"] in ("land", "hold"):
            verdicts[f"{r['pr']}@{r['head']}"].append((r["ts"], r["by"], r["value"]))
    marks: dict[int, list] = collections.defaultdict(list)
    for r in sorted(labels, key=lambda r: r["ts"]):
        if r["key"] == "ready":
            marks[r["pr"]].append((r["ts"], r["head"], r["value"] is True))
    ps = mark_pairs(labels) if "--marks" in sys.argv else hold_pairs(labels) if "--holds" in sys.argv else pairs(labels)
    excludes = tomllib.loads(git("show", "origin/main:tools/verity/check/drive.toml").stdout)["lowrisk_excludes"]
    init = (tips(Path(act_dir)), dict(verdicts), dict(marks), excludes)
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=init) as ex:
        res = list(ex.map(judge, ps, chunksize=8))
    json.dump(res, open(out_path, "w"), indent=0)
    c = collections.Counter(r["kind"] for r in res)
    w = collections.Counter(r.get("wide", {}).get("kind") for r in res)
    print(len(res), "re-verdicts;", sum(r["carries"] for r in res), "carry (strict);", dict(c.most_common()))
    print(sum(r.get("wide", {}).get("carries", False) for r in res), "carry (with low-risk deltas);", dict(w.most_common()))
    print(sum(r.get("loose", {}).get("carries", False) for r in res), "carry trusting unmarked or superseded heads (loose)")
    print(sum(r.get("later", {}).get("carries", False) for r in res), "untrusted-parent ones that carry once their parents are reviewed")


if __name__ == "__main__":
    main()
