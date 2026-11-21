"""chain-record: the network attester path's endpoint step and its K per monitoring period, each read off the attester's
fetched gate run.

    python benchmarks/network_traces/chain_record.py endpoint --warden-run DIR --endpoint DIR --eos IDS --out FILE
    python benchmarks/network_traces/chain_record.py leak --warden-run DIR [--scan SUMMARY] --out FILE

* **endpoint**: binds the endpoint's own advice to the attested I/O read off the network attester's cells
  (`network_attester_endpoint`); where the endpoint's records stay on the served node, that node runs it
  (`served_endpoint`, `served_session.py endpoint`).
* **leak**: K per monitoring period (week, month, year) on the attester's gate run, pooled at δ = 1% and 10⁻⁶ and in the
  per-period form, each beside its share of one copy of the weights (`leak`), computed by `leak_bounds`' scaling by
  links: one link at the run's rate, and as many of it summed as the census's reference datacenter and the world
  have machines. The network attester's term at a scope's own load is `verity bench certifier_term`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

from verity.protocols.accounting.network.prover import released_keys as RK

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(Path(__file__).resolve().parents[2]))  # the root, for benchmarks.network_traces

DELTAS = (0.5, 2.0 ** -20)
#: the `tie` step's statement: it holds no id's value
INGRESS_TIE = "the ingress tie (IoTieRequest): prompt lengths and arrival order"

CLEAR = {
    "custody": "the SHA-512 of each fetched file against the gate's end.json (public files)",
    "endpoint": "in the clear, where the endpoint step runs, on the endpoint's records and the run's certified I/O: "
                "each Commit window's advice opened to its registered root and rebuilt from the certified I/O; "
                "no I/O tie runs on this path; token content is tied by nothing here",
    "n2": "at the verifier: the registration records, the sample, the certificates' public fields, the clock syncs, the "
          "proof's pub row opened to the verifier's own copy and the o0 output word; never a record, head, leaf "
          "salt, sibling or the network attester's salt (n2_zk's RECEIVED, scanned for every piece)",
    "scan": f"at the verifier: each unit's o0 output opened from the salts the prover sends; {INGRESS_TIE}, proved "
            "in a step of its own (`tie`) on the network attester path, run on no other",
    "lean": "nothing: each proof is decided by the verifier's own Lean `verity-verify-proof verify --zk` run",
    "warden_certificates": "in the clear, in the chain's job (chain_record.py certs), on the network certifier's "
                           "witness: the network certifier's own offline check "
                           "(network_attester_serve.check: each record's certificate and its rows' registration under the "
                           "network certifier's key and session, the audit and the client's view), and the public "
                           "certificates.jsonl held to the certified lines",
    "warden_endpoint": "in the clear, where the endpoint step runs (chain_record.py endpoint, in the chain's job), on "
                       "the endpoint's records and the network certifier's certified I/O (network_attester_chain.io, off "
                       "the records' cells): each Commit window's advice opened to its registered root and rebuilt "
                       "from the certified I/O",
    "warden_endpoint_served": "on the served node, the prover's, whose records hold request values and stay there: "
                              "each Commit window's advice opened to its registered root and rebuilt from the network certifier's "
                              "certified I/O sent to it (served_session.py endpoint); the chain reads its report",
    "tie": "at the verifier: each ingress bucket unit's pub row opened to the verifier's own copy (the advice's prompt "
           "lengths and the network certifier's release) and its o0 verdict bit; never a cell or its salt (n2_tie_zk's RECEIVED, "
           "scanned for every piece). It holds each request's prompt length and arrival order, not the ids' values",
}
#: an endpoint report's ``at`` when its step ran on the served node (`served_session.py endpoint`)
SERVED_NODE = "the served node"


def _json(path) -> dict | None:
    p = Path(path) if path else None
    return json.loads(p.read_text()) if p and p.is_file() else None


# --- custody ----------------------------------------------------------------------------------------------------------


# --- attestations -----------------------------------------------------------------------------------------------------

# --- report -----------------------------------------------------------------------------------------------------------


def endpoint_advice(b: dict) -> dict:
    """What the record keeps of the endpoint's advice (`chain.bits`): whether it is bound and rebuilt. Its Commit windows'
    count, whys and rows, which name requests and count each window's, stay in the endpoint's report."""
    return {"bound": b.get("bound"), "rebuilt": b.get("rebuilt")}


def advice_bits(b: dict) -> dict:
    """What the record keeps of the advice's bits (`chain.bits`): the egress window's T buckets alone. Its lengths, v0's
    and the residual's, are functions of what was served: private, in the endpoint's report."""
    return {"T": b.get("T")}


#: `verity-vllm advice`, as network-e2e runs it
ADVICE_CLI = [sys.executable, "-m", "verity_vllm.pipeline.cli", "advice"]


def network_attester_endpoint(run: Path, ep: Path, eos: list[int]) -> dict:
    """The endpoint's own advice for a network attester run (`served_endpoint`), on the attested I/O read off its
    cells (`network_attester_chain.io`)."""
    from benchmarks.network_traces import network_attester_chain as WC

    if not (ep / "advice_trace.jsonl").is_file():
        raise SystemExit(f"{ep} holds no advice_trace.jsonl: the endpoint step reads the endpoint's private records (its "
                         f"serve dir, or served_session.py endpoint's --keep records/), not its art, whose report.json "
                         f"is public (CR_ENDPOINT=<the art> reads that report alone)")
    return served_endpoint(WC.deployment(run)[0], WC.io(run), ep, eos)


def served_endpoint(dep: dict, io: dict, ep: Path, eos: list[int]) -> dict:
    """The endpoint's own advice for the network attester run of ``dep`` (its ``deployment.json``), as network-e2e's
    `endpoint_advice` makes it for a network attester rehearsal's: `verity-vllm advice` over the endpoint's records ``ep``
    (network-e2e's ``chain.ENDPOINT``) into ``ep/advice`` at the network attester's bucket unless an index is there,
    then ``chain.served_advice`` on the attested I/O ``io`` (`network_attester_chain.io`'s) over the network attester's
    execution phases, with the model's EOS ids ``eos`` (a network attester run records none)."""
    import subprocess

    sys.path.insert(0, str(HERE.parent / "network_e2e"))
    import chain

    T = dep["params"]["window"]
    window = {"origin_unix": dep["origin_unix"], "windows": dep["windows"],
              "window_s": T * dep["params"]["bucket_ms"] / 1000}
    out = {"warden_run": dep["run_id"], "endpoint": str(ep), "io": {k: v for k, v in io.items() if k != "records"}}
    adv = ep / "advice"
    if not (adv / "windows.json").exists():
        p = subprocess.run([*ADVICE_CLI, str(ep), "--out", str(adv), "--tau", repr(window["window_s"] / T)],
                           capture_output=True, text=True, check=False)
        (ep / "advice.log").write_text(p.stdout + p.stderr)
        if not (adv / "windows.json").exists():
            return out | {"chain": {"bits": {"bound": False, "bound_why": f"verity-vllm advice made no index (exit "
                                             f"{p.returncode}: {(p.stdout + p.stderr)[-300:].strip()})"}}}
    try:
        info, bits = chain.served_advice(ep, adv, window, io["records"], T, eos)
    except (OSError, KeyError, ValueError) as err:
        return out | {"chain": {"bits": {"bound": False, "bound_why": f"the endpoint's advice: {err!r}"}}}
    return out | {"advice": info, "chain": {"bits": bits}}


ENDPOINT_PUBLIC_FORMAT = "network-attester-endpoint-step/1"

#: The endpoint step's public document (`served_session.py endpoint`'s art): each field it may hold, by its path, with
#: its public kind, as released_keys.json's list of that name declares it. `endpoint_public` writes these and nothing
#: else, so a field not there is never published. ``io/accepted``: the attested I/O is well formed and in order; no id
#: is compared with the Commit. ``io/violations``: how many of those checks failed.
ENDPOINT_LIST = "the endpoint step's public document"
ENDPOINT_PUBLIC = next(x for x in RK.load().lists if x.name == ENDPOINT_LIST).keys


def endpoint_public(step: dict) -> dict:
    """`ENDPOINT_PUBLIC`'s fields of ``step`` (`network_attester_endpoint_step`'s), each a scalar; refuses any other value."""
    doc: dict = {}
    for path in ENDPOINT_PUBLIC:
        *up, name = path.split("/")
        src, dst = step | {"format": ENDPOINT_PUBLIC_FORMAT}, doc
        for k in up:
            src, dst = src.get(k) if isinstance(src.get(k), dict) else {}, dst.setdefault(k, {})
        if name in src:
            if src[name] is not None and not isinstance(src[name], (bool, int, float, str)):
                raise ValueError(f"the endpoint step's {path} is a {type(src[name]).__name__}, not a scalar: "
                                 f"ENDPOINT_PUBLIC publishes scalars only")
            dst[name] = src[name]
    return doc


def network_attester_endpoint_step(e: dict | None) -> dict:
    """`network_attester_endpoint`'s report, or its public document (`endpoint_public`): accepted when the network
    attester's attested I/O reads whole off its cells and the endpoint's own advice is bound to it and rebuilt from it.
    ``io``'s ``accepted``: the attested I/O is well formed and in order; no id is compared with the Commit."""
    if e is None:
        return {"verdict": "not run", "why": "no endpoint records (CR_ENDPOINT)",
                "in_the_clear": CLEAR["warden_endpoint"]}
    if e.get("format") == ENDPOINT_PUBLIC_FORMAT:
        p = endpoint_public(e)
        io, advice, bits = p.get("io") or {}, p.get("advice") or {}, p.get("advice_bits") or {}
        if missing := [k for k in ENDPOINT_PUBLIC if k != "why" and k.split("/")[-1] not in
                       (p if "/" not in k else p.get(k.split("/")[0]) or {})]:
            raise ValueError(f"the endpoint's public document lacks {', '.join(missing)}: rerun `served_session.py "
                             f"endpoint`, or pass the private keep's report.json (--keep)")
    else:
        b, io = (e.get("chain") or {}).get("bits") or {}, e.get("io") or {}
        advice, bits = endpoint_advice(b), advice_bits(b)
        io = {"accepted": io.get("accepted"), "violations": len(io.get("violations") or [])}
    why = (f"the network certifier's certified I/O doesn't read whole off its cells: {io['violations']} violation(s)"
           if not io["accepted"] else None if advice["bound"] and advice["rebuilt"]
           else "the endpoint's advice isn't bound to the certified I/O or rebuilt from it")
    return {"verdict": "refused" if why else "accepted", "why": why, "warden_run": e.get("warden_run"), "io": io,
            "advice": advice, "advice_bits": bits, "at": e.get("at"),
            "in_the_clear": CLEAR["warden_endpoint_served" if e.get("at") == SERVED_NODE else "warden_endpoint"]}


# --- leak: K per monitoring period, on the network attester run ------------------------------------------------------

#: what a monitoring period's count runs over, each machine one link at the run's rate: one link
#: (`leak_bounds.per_link`), and as many of it as the census's reference datacenter and the world have machines
#: (`leak_bounds.at_link_rate`)
SCOPES = ("link", "datacenter", "world")


def leak(run: Path, scan: dict | None = None, weights=None) -> dict:
    """K, the bits the network attester's accepted transcripts can carry about the secret, per monitoring period on
    its gate run ``run`` (a fetched copy: its ``request.json``, ``end.json`` and ``deployment.json``), for the scan's class
    (`ingress_scan.W63`) and its sample of every proof block, by `leak_bounds`' scaling by links (`SCOPES`): one link
    certifying the run's own request rate (`leak_bounds.per_link`), and as many identical links summed
    (`leak_bounds.at_link_rate`) as the census's reference datacenter has machines (`leak_bounds.datacenter_links`)
    and as the world fleet has (`leak_bounds.world_links`), whatever those scopes serve. Each count is in both forms,
    the per-period form (`MonitoringPeriodLeakAccepted`'s count) and the pooled one (`PooledLeakAccepted`'s at q = 1), in both
    readings of the sampling block's free inputs (`leak_bounds.READINGS`), and beside ε, its scans' δ summed, and its
    share of ``weights`` (`leak_bounds.WEIGHTS`). The run's request rate is the requests its request asked for over
    its execution phases: the end carries no count of the traffic, and the gate takes an end as accepted only when
    its check counted the request's own number (`auditor.network_attester_outcome`). So K is counted only on an accepted run
    of the device's own client; a relay session's request asks for no number, its clients being the peer's, and
    nothing holds its requests to one."""
    import leak_bounds as LB
    from verity.protocols.accounting.network.prover import ingress_scan as IS
    from verity.protocols.accounting.network.verifier import network_attester as W
    weights = LB.WEIGHTS if weights is None else weights
    dep = _json(next((p for p in (run / "out" / "deployment.json", run / "deployment.json") if p.is_file()), None))
    v = (_json(run / "end.json") or {}).get("verdict") or {}
    if not dep or not v.get("windows"):
        raise SystemExit(f"leak: {run} has no deployment.json, or no end.json verdict with its windows")
    asked = (_json(run / "request.json") or {}).get("requests")
    if type(asked) is not int or asked < 1:
        raise SystemExit(f"leak: {run}'s request.json asks for no number of requests (a relay session's clients are "
                         "the peer's), so nothing holds its requests to a count and K has no rate")
    if v.get("accepted") is not True:
        raise SystemExit(f"leak: {run}'s end isn't accepted, and the gate holds only an accepted end's requests to "
                         "the number its request asked for")
    params, cls = W.NetworkAttesterParams(**dep["params"]), IS.W63
    cls.serves(params)
    T, units = params.window, 2 * params.window + 1
    sampled = (scan or {}).get("draw", {}).get("honest") if isinstance((scan or {}).get("draw"), dict) else units
    if sampled != units:
        raise SystemExit(f"leak: the scan sampled {sampled} of {units} proof blocks; K's terms for incorrect proof blocks "
                         f"are 0 only with every proof block sampled (scan_zk --draw {units})")
    rate = asked / v["windows"]
    window_s = T * params.bucket_ms / 1000
    at = {"T": T, "tau": Fraction(params.bucket_ms, 1000), "egress_cells": params.egress_cells}
    per_s = Fraction(asked, v["windows"]) / (T * at["tau"])
    link = LB.link(cls, per_s, **at)
    links = {"link": 1, "datacenter": LB.datacenter_links(), "world": LB.world_links()}

    def bound(scope: str, period: str, reading: str, delta: Fraction | None = None) -> LB.Bound:
        if scope == "link":
            return LB.per_link(cls, per_s, period, reading, delta, **at, weights=weights)
        return LB.at_link_rate(link, links[scope], period, reading, delta, weights=weights)

    rows = []
    for period, s in LB.PERIODS.items():
        for scope in SCOPES:
            lw = links[scope] * s / window_s
            for reading in LB.READINGS:
                pooled = {f"{float(d):g}": bound(scope, period, reading, d) for d in LB.DELTAS}
                per = bound(scope, period, reading)
                rows.append({"period": period, "scope": scope, "links": links[scope], "link_windows": lw,
                             "requests": rate * lw, "prog": reading, "pooled": {d: b.bits for d, b in pooled.items()},
                             "pooled_share": {d: b.share for d, b in pooled.items()},
                             "per_period": per.bits, "per_period_share": per.share, "eps": str(per.eps),
                             "per_link_window": link.bits(link.block_s, reading)})
    free = {reading: {"link_window": f["block"], "request": f["request"]}
            for reading, f in ((r, link.free(r)) for r in LB.READINGS)}
    split = "; ".join(f"the {scope}'s F = {LB.fmax(d / links[scope])} at δ = {float(d):g}"
                      for scope in SCOPES[1:] for d in LB.DELTAS)
    return {"warden_run": (_json(run / "end.json") or {}).get("run") or run.name, "params": dep["params"],
            "class": {k: getattr(cls, k) for k in IS.STATICS},
            "widths": link.widths | {"entries": link.entries, "egress_row": 8 * cls.egress.R * cls.B},
            "draw": {"units": units, "drawn": sampled},
            "run": {"link_windows": v["windows"], "requests": asked, "requests_per_link_window": rate,
                    "link_window_s": window_s, "accepted": v.get("accepted")},
            "terms": {"schedules_per_request": link.cell, "cells_per_request": link.cell,
                      "cap_per_request": link.cap, "free": free, "faulty": 0,
                      "jitter_per_block": link.jitter, "jitter_bound": LB.JITTER_BOUND, "jitter_term": LB.JITTER,
                      "b": link.b, "fmax": {f"{float(d):g}": LB.fmax(d) for d in LB.DELTAS},
                      "scan_delta": str(link.scan_delta)},
            "links": links, "weights": weights._asdict(), "rows": rows, "theorems": dict(LB.STATEMENTS),
            "conditions": [
                "the generator reading of the scan's statement (each proof block's registered rows its boundary "
                "values): a claim, not yet a locked guarantee",
                "each connection's cap is its lane's egress_cells, which the network certifier holds a lane to and "
                "NetworkVerifier.Period doesn't check; the schedules and the fail cells are charged at that cap, "
                "⌈log2(cap + 1)⌉ each a request, as its rows form charges them",
                f"the jitter term is T bits a sampling block, one a bucket: {LB.JITTER}",
                "the scan proves sampling block 0 of the network certifier run; a monitoring period's terms are its "
                "per-sampling-block and per-request terms at the run's request rate",
                "the datacenter's rows sum as many identical links as the census's reference datacenter has machines, "
                "and the world's as many as the world fleet has (leak_bounds.at_link_rate, its scaling by links), "
                f"each link pooled at δ / links ({split}) and paying its own incorrect proof blocks: our "
                "composition, which no Lean statement covers. Each link certifies the run's rate; a scope's requests "
                "past its links' rate are uncertified, and no row counts them, so these rows are not the scope's K "
                "(leak_bounds.table counts the network certifier's terms over every request a scope serves)",
                "ε is the scans' δ summed over the period's sampling blocks and the links (NetworkVerifier.Period's "
                "eps, m·δ), each scan at the scan's δ",
                "the loop's (an untied compute audit's) free inputs don't enter K: its δ goes into JointPeriodLive's "
                "sum",
                "the scan's prog is a compute audit's output (NetworkVerifier.Period's progTied), so its slot entries' "
                "bits are the tied audit's free inputs, which Lean's rows-form K charges to the sampling block's term "
                "(tiedAudits), as this K does",
                f"a share is of one copy of the weights, {weights.label}, whatever the links"]}


def leak_markdown(doc: dict) -> str:
    import leak_bounds as LB
    d1, d2 = (f"{float(d):g}" for d in LB.DELTAS)
    bits = lambda x: f"{x / 10 ** (e := math.floor(math.log10(x))):.3g}·10^{e}" if x >= 1e6 else f"{x:,.0f}"
    share = lambda x: f"{100 * x:.3g}%" if x < 1 else f"{x:.3g}×" if x < 100 else f"{x:,.0f}×"
    lines = []
    for reading, title in LB.READINGS.items():
        lines += [f"**{title}**, bits, the share of one copy of the weights ({doc['weights']['label']}) and ε, the "
                  f"scans' δ summed (the jitter term one bit a bucket, {LB.JITTER}):", "",
                  "| period | links | K pooled, δ = 1% | K pooled, δ = 1e-6 | K per period | ε | a sampling block |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['period']} | {r['scope']}, {r['links']:,} | {bits(r['pooled'][d1])} "
                  f"({share(r['pooled_share'][d1])}) | {bits(r['pooled'][d2])} ({share(r['pooled_share'][d2])}) | "
                  f"{bits(r['per_period'])} ({share(r['per_period_share'])}) | {float(Fraction(r['eps'])):.3g} | "
                  f"{r['per_link_window']:,.0f} |"
                  for r in doc["rows"] if r["prog"] == reading]
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("endpoint",
                       help="the network attester path: the endpoint's own advice bound to the network attester's certified I/O")
    e.add_argument("--warden-run", dest="network_attester_run", required=True, type=Path)
    e.add_argument("--endpoint", required=True, type=Path, help="the endpoint's records (network-e2e chain.ENDPOINT)")
    e.add_argument("--eos", required=True,
                   help="the model's EOS token ids, comma-separated (a network attester run records none)")
    e.add_argument("--out", required=True, type=Path)
    k = sub.add_parser("leak", help="K per monitoring period on the network attester run, for the scan's class and sample")
    k.add_argument("--warden-run", dest="network_attester_run", required=True, type=Path,
                   help="the fetched network attester gate run (its end.json)")
    k.add_argument("--scan", help="the scan's summary on that run (its sample); default every proof block sampled")
    k.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.command == "leak":
        doc = leak(a.network_attester_run, _json(a.scan))
        a.out.write_text(json.dumps(doc, indent=1, sort_keys=True))
        a.out.with_suffix(".md").write_text(leak_markdown(doc))
        print(leak_markdown(doc))
        return 0
    doc = network_attester_endpoint(a.network_attester_run, a.endpoint, [int(x) for x in a.eos.split(",")])
    doc["verdict"] = network_attester_endpoint_step(doc)["verdict"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: doc.get(k) for k in ("verdict", "why") if k in doc}))
    return 0 if doc.get("verdict") == "accepted" else 1


if __name__ == "__main__":
    sys.exit(main())
