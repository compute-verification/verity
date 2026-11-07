"""NET-190's serving cost, on the served node: the run served through the network certifier against the remote run it
mirrors (same card, model and build), with `served_overhead`'s rulers in both arms.

    odserve_cost.py SERVE_DIR PRIVATE_DIR [STOCK]

The network certifier's client is the front's load's client: its log (the fetch's witness client.json, unpacked by odserve_check.py
in PRIVATE_DIR/fetch) is written as the load's timings document (`served_overhead.client_metrics`' form: each
connection's send, first data cell and last cell, on the deployment's origin), in PRIVATE_DIR/serve beside links to the
serve dir's step trace, windows and serve.json. `served_overhead` then reads the busy periods the network certifier paces as the one
load whose span the client's is, as it reads remote's load with serve_load's timings, and PRIVATE_DIR/overhead.json gets
its aggregate. With STOCK (the remote run's public out/serve/overhead.json), `served_overhead.table` holds the two to
one engine (`identity`) and one load size, and `public` of it is printed: each metric's ratio, engine and client, with
no count, time or rate of either arm. The network certifier holds each response D·τ from its admission, so the client's ratios are
its schedule's. Exit 0 when the table ran.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TREE = Path.cwd()
sys.path[:0] = [str(TREE / "benchmarks" / "one_stage")]
import served_overhead as SO  # noqa: E402

FRONT_TIMINGS = "front.timings.json"


def timings(private: Path) -> dict:
    """The network certifier's client's log as a timings document; `ValueError` when no fetch is unpacked or a connection failed."""
    hit = next((private / "fetch").rglob("client.json"), None)
    dep = next((private / "fetch").rglob("deployment.json"), None)
    if hit is None or dep is None:
        raise ValueError("no unpacked fetch with client.json and deployment.json (odserve_check.py runs first)")
    reqs = []
    for c in json.loads(hit.read_text())["connections"]:
        if "error" in c or not c.get("arrivals_ms"):
            raise ValueError(f"the network certifier's client's connection {c.get('index')} failed or got no cell")
        r = c.get("response") or {}
        first = next((t for t, k in zip(c["arrivals_ms"], c["kinds"]) if k == "d"), None)
        reqs.append({"user": f"connection-{c['index']}", "ok": bool(r.get("complete")) and c.get("outcome") == "end",
                     "send_s": c["sent_ms"] / 1e3, "first_token_s": None if first is None else first / 1e3,
                     "done_s": c["arrivals_ms"][-1] / 1e3, "tokens": int(r.get("completion_tokens") or 0)})
    return {"start_unix": float(json.loads(dep.read_text())["origin_unix"]), "requests": reqs}


def shadow(serve: Path, private: Path) -> Path:
    d = private / "serve"
    d.mkdir(exist_ok=True)
    for name in (SO.TRACE, SO.WINDOWS, SO.SERVE):
        if (serve / name).exists() and not (d / name).exists():
            (d / name).symlink_to(serve / name)
    (d / FRONT_TIMINGS).write_text(json.dumps(timings(private)))
    return d


def reason(e: Exception) -> str:
    s = str(e)
    if "loads of different sizes" in s:
        return "loads of different sizes"
    if "engine differs" in s:
        return "the engines differ (the keys in the serve dir's private/odserve/cost.json)"
    return f"{type(e).__name__} (its reason in the serve dir's private/odserve/cost.json)"


def main(argv: list[str]) -> int:
    serve, private = Path(argv[1]), Path(argv[2])
    stock = Path(argv[3]) if len(argv) > 3 and argv[3] else None
    full: dict = {}
    pub: dict = {"schema": "odserve/cost/v0"}
    try:
        d = shadow(serve, private)
        full["composed"] = SO.aggregate(d)
        (private / "overhead.json").write_text(json.dumps(full["composed"]))
    except (ValueError, OSError, KeyError) as e:
        full["refused"] = f"{type(e).__name__}: {e}"
        pub["ratios"] = f"no aggregate: {reason(e)}"
        d = None
    if d is not None and stock is None:
        pub["ratios"] = "no stock aggregate given"
    elif d is not None:
        try:
            t = SO.table([d], [stock])
            full["table"] = t
            pub["ratios"] = SO.public(t)["ratios"]
            pub["stock_run"] = t["stock"][0]["run_id"]
            pub["allowed"] = sorted({k for x in t["allowed"].values() for k in x})
        except (ValueError, OSError, KeyError) as e:
            full["refused"] = f"{type(e).__name__}: {e}"
            pub["ratios"] = f"refused: {reason(e)}"
    (private / "cost.json").write_text(json.dumps(full, indent=1, default=str))
    print(json.dumps(pub))
    return 0 if isinstance(pub.get("ratios"), dict) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
