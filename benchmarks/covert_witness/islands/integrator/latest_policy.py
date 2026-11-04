"""Print the scorer's newest stored algorithm-policy table art at te_core 4.13 or later, by its refresh run id.

    python islands/integrator/latest_policy.py

It takes both forms of the scorer's evidence/v1 meta: `what` naming the table (`WHAT`), with the version in `source`
and the run in `refresh`, and a refresh's own, whose `what` reads "... uncut tables ... refresh RUN ... te_core X.Y".
An art qualifies with te_core 4.13 or later, a refresh run and both algorithm-policy.json and frontier-fit.json; the
newest refresh run wins, an art whose manifest lacks those fields is skipped, and with none (or no store) it prints
`FALLBACK`."""
import json
import re
import subprocess
import sys

WHAT = "training/algorithm-policy uncut, its coverage and training/frontier-fit (te_core.algorithm_policy)"
FALLBACK = "art:377ab72403325bed75de054d7e6cfabc82e14c3ccfcf0bdec73a599e816bbfc6"
RUN = re.compile(r"(r\d{8}-\d{6}-[0-9a-f]{4})")
VERSION = re.compile(r"te_core (\d+)\.(\d+)")


def select(*where):
    r = subprocess.run(["research", "data", "select", "--kind", "evidence/v1", *where, "--json"],
                       capture_output=True, text=True, timeout=300)
    return json.loads(r.stdout)


def qualifies(a: dict) -> str | None:
    """The art's refresh run if it is a 4.13+ table with both files, else None."""
    meta = a["manifest"]["meta"]
    what = meta.get("what", "")
    v = VERSION.search(meta.get("source", "")) or VERSION.search(what)
    run = RUN.match(meta.get("refresh", "")) or re.search(r"refresh " + RUN.pattern, what)
    files = {f["path"] for f in a["manifest"]["payload"]["files"]}
    if v and (int(v[1]), int(v[2])) >= (4, 13) and run and {"algorithm-policy.json", "frontier-fit.json"} <= files:
        return run[1]
    return None


def pick(arts: list[dict]) -> str:
    best = (None, FALLBACK)
    for a in arts:
        try:
            run = qualifies(a)
        except (KeyError, TypeError) as e:
            print(f"latest_policy: skipping {a.get('id') if isinstance(a, dict) else a!r}: {e!r}", file=sys.stderr)
            continue
        if run and (best[0] is None or run > best[0]):
            best = (run, a["id"])
    return best[1]


def main() -> int:
    try:
        arts = select("--where", f"meta.what={WHAT}")
        try:
            arts += [a for a in select() if "uncut tables" in a["manifest"]["meta"].get("what", "")]
        except Exception as e:  # noqa: BLE001
            print(f"latest_policy: the refresh-form select failed: {e}", file=sys.stderr)
        print(pick(arts))
    except Exception as e:  # noqa: BLE001
        print(f"latest_policy: {e}; using {FALLBACK}", file=sys.stderr)
        print(FALLBACK)
    return 0


if __name__ == "__main__":
    sys.exit(main())
