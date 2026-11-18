"""Today's rule and refresh side by side: the sessions of a run of `run.sh` (its `out/verifier.json` and
`out/prover.json`), one row a session, as one document and one result through the harness kit.

    PLAN=kv:4,kv:4 REFRESH=0,1 ... bash benchmarks/poub_session/run.sh
    python benchmarks/poub_session/refresh_result.py OUT            one run: one registration, a session a rule
    python benchmarks/poub_session/refresh_result.py OUT OUT2       two runs, and the result says so

A session is under one rule (`audit.Params.refresh`), so the two rules are two sessions: on one registered store, by
the same plan from the same prover seed, each drawing its own live coins, which nothing seeds or shares. The document
(`OUT/refresh.json`, `SCHEMA`) says which sessions those are, the commit, the codec and the store's depth, and for each rule
(`row`) the verifier's verdict with its failure's name, γ and its error, the receipts and bytes credited under that
rule, the share of what was written that was held but not yet creditable (`probation`), and the re-key work with its
share of the session's encodes (`rekeys`).  With `REFRESH` unset every row is today's rule.  A session on a card
adds its `card` object (`card`); the CPU reference's has none, and the document then says so (`size`): its codec ran
on a processor, and nothing in it is a rate.

The result is a trial, always: these are a branch's or a loopback's sessions and reach no metric's line, so each row's
γ is a kit measurement in `out/trial-measurements.json` (member `refresh-off` or `refresh-on`, the row beside it), and
`result.json` says `trial: true` and names `refresh.json`.

Read from the verifier's record alone, but for the re-key work, which only the prover counts (`prover.json`).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from verity_bench import kit

SCHEMA = "verity/poub-refresh-rules/v0"
UNIT = "share of the node's memory read rate left uncertified, at the data sheet's read bandwidth"
COINS = "each session drew its own live coins from the {challenger} challenger: none is seeded, none shared"
PROCESSOR = ("the reference size runs {codec} on a processor (the driver's CPU reference prover) and is not a rate: "
             "kernels measured on an H200 re-key the deployed codec at 0.52 GB/s and p512/k2 at 2.08 GB/s (the "
             "lane's measurements of 18 Nov 2026), and no session runs at 512 bits on a card today, where there is "
             "no receipt-sketch path")
NO_BOUND = ("γ is above 1 at this size ({blocks:,} blocks of {block_bytes:,} bytes): the rows show each rule's "
            "verdict and its ledger's lines, not a bound")


def _at(d, *keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def probation(session: dict, block_bytes: int) -> dict | None:
    """What a refresh session held on probation, from its record's windows (`"probation"`, an integer a pass, the
    blocks written that no plan may hold yet): those block-passes in bytes, beside the bytes of written blocks its
    passes read (`post_seed`), and the first over their sum. None for a session whose record names none: today's rule."""
    windows = _at(session, "record", "profile", "windows") or []
    if not any("probation" in w for w in windows):
        return None
    waiting = sum(sum(w.get("probation") or ()) for w in windows) * block_bytes
    read = sum(sum(w.get("post_seed") or ()) for w in windows) * block_bytes
    return {"waiting_bytes": waiting, "written_bytes": waiting + read,
            "share": waiting / (waiting + read) if waiting + read else None,
            "term": _at(session, "ledger", "spec", "probation")}


def rekeys(proved: dict | None, block_bytes: int) -> dict | None:
    """The session's re-key work as its prover counted it (`prover.json`'s session): the re-key appends, their blocks
    and bytes, and those blocks over every block the session encoded. A refresh session's is its `refresh` record
    (`appends`, `blocks`, `encoded`); today's rule's is the serving trace's (`kv`: `rekeyed` of `appended`, which
    counts no appends). None where the prover recorded neither."""
    held = (proved or {}).get("refresh") or (proved or {}).get("kv")
    if not isinstance(held, dict):
        return None
    blocks = held.get("blocks", held.get("rekeyed"))
    encoded = held.get("encoded", held.get("appended"))
    if not isinstance(blocks, int):
        return None
    return {"appends": held.get("appends"), "blocks": blocks, "bytes": blocks * block_bytes,
            "share_of_encodes": blocks / encoded if isinstance(encoded, int) and encoded else None}


def number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def card(calibration: dict, session: dict, proved: dict | None, store_bytes: int, block_bytes: int) -> dict | None:
    """What a session on a card adds to its row, each from the records and None where they hold none: the median
    pass (`calibration.pass_ns.p50`), the card's bare read rate (`read_tiers.hbm.bytes_per_s`), the registration's
    encode rate, the re-keys' rate (their bytes over their encode time, the prover's `refresh.encode_ns`) and that
    time over the session's span, the worst window's probation over written blocks, and the store's bytes. None, the
    whole object, for a session the CPU reference proved: its calibration times no pass (`pass_ns`)."""
    if not isinstance(calibration.get("pass_ns"), dict):
        return None
    work = (proved or {}).get("refresh") or {}
    blocks, encode_ns = work.get("blocks"), work.get("encode_ns")
    span = _at(session, "record", "public", "span_ns")
    timed = number(blocks) and number(encode_ns) and encode_ns > 0
    shares = []
    for w in _at(session, "record", "profile", "windows") or ():
        waiting, read = sum(w.get("probation") or ()), sum(w.get("post_seed") or ())
        if "probation" in w and waiting + read:
            shares.append(waiting / (waiting + read))
    p50, bare = calibration["pass_ns"].get("p50"), _at(calibration, "read_tiers", "hbm", "bytes_per_s")
    return {"pass_ms_p50": p50 / 1e6 if number(p50) else None,
            "bare_read_TBps": bare / 1e12 if number(bare) else None,
            "encode_Bps": calibration.get("encode_bytes_per_s"),
            "rekey_Bps": blocks * block_bytes / (encode_ns / 1e9) if timed else None,
            "rekey_share": encode_ns / span if timed and number(span) and span else None,
            "probation_share": max(shares) if shares else None,
            "store_bytes": store_bytes}


def row(session: dict, proved: dict | None, registration: dict, block_bytes: int, run: int,
        calibration: dict | None = None) -> dict:
    """One session as its rule's row. Its verdict is the verifier's: `accepted`, by which verifier (`verdict`), and a
    refused session's failure by name with its window, and then no γ; where the two verifiers disagreed, on what and
    what each said (`disagreement`). `card` is `card`'s object, None off a card."""
    accepted = session.get("accepted") is True
    inputs, spec = _at(session, "ledger", "inputs") or {}, _at(session, "ledger", "spec") or {}
    receipts, post = inputs.get("receipts"), inputs.get("post_seed_receipts")
    credited = None
    if isinstance(receipts, int) and isinstance(post, int):
        credited = receipts if inputs.get("post_seed_credited") else receipts - post
    failure = session.get("failure") or {}
    lean_ns = [v for v in (session.get("lean_check_ns") or {}).values() if isinstance(v, int)]
    blocks = registration["blocks"]
    differ = session.get("disagreement") or None
    return {
        "refresh": session.get("refresh") is True, "run": run, "session": session["k"],
        "accepted": accepted, "verdict": session.get("verdict"),
        "failure": failure.get("failure"), "failure_window": failure.get("t"),
        "disagreement": differ and {k: differ.get(k) for k in ("t", "event", "lean", "python")},
        "codec": registration.get("codec"), "store_depth": max(blocks - 1, 0).bit_length(),
        "windows": session.get("windows"), "lifetime": session.get("lifetime"), "samples": session.get("samples"),
        "gamma": spec.get("gamma") if accepted else None,
        "gamma_decided_by": spec.get("gamma_decided_by") if accepted else None,
        "error": _at(session, "ledger", "declared", "gamma", "spec", "error", "windows") if accepted else None,
        "certified_bytes": spec.get("certified_bytes") if accepted else None,
        "passes": inputs.get("passes"), "receipts": receipts, "post_seed_receipts": post,
        "post_seed_credited": inputs.get("post_seed_credited"),
        "credited_receipts": credited, "credited_bytes": None if credited is None else credited * block_bytes,
        "appended": session.get("appended"), "released": session.get("released"),
        "probation": probation(session, block_bytes) if accepted else None,
        "rekeys": rekeys(proved, block_bytes),
        "lean_check_ms": [min(lean_ns) / 1e6, max(lean_ns) / 1e6] if lean_ns else None,
        "card": card(calibration or {}, session, proved, blocks * block_bytes, block_bytes),
    }


def read(out: Path) -> tuple[dict, dict]:
    v = json.loads((out / "verifier.json").read_text())
    if not v.get("sessions"):
        raise ValueError(f"{out}/verifier.json records no session")
    p = out / "prover.json"
    return v, json.loads(p.read_text()) if p.is_file() else {}


def document(outs: list[Path]) -> dict:
    """The comparison of the runs at `outs`: what the rows share, said once, and `rules`, a row a session in the
    order run. Refused where the runs' stores differ in size or codec, which no row could be compared across."""
    rows, registrations, challengers, plans, block_bytes = [], [], set(), [], None
    for n, out in enumerate(outs):
        v, p = read(out)
        reg = v["registration"]
        registrations.append(reg)
        proved = {s.get("k"): s for s in p.get("sessions") or ()}
        for s in v["sessions"]:
            size = _at(s, "record", "public", "block_bytes")
            block_bytes = block_bytes or size
            challengers.add(_at(s, "challenger", "challenger"))
            ps = proved.get(s["k"])
            plans.append(None if ps is None else f"{ps.get('mode')}:{ps.get('fraction')}")
            rows.append((s, ps, reg, n, v.get("calibration") or {}))
    if len({(r["blocks"], r.get("codec")) for r in registrations}) != 1:
        raise ValueError("the runs' stores differ in blocks or codec: their sessions are not one comparison")
    if block_bytes is None:
        raise ValueError("no session of these runs was accepted, so none states its record's block size")
    rules = [row(s, ps, reg, block_bytes, n, cal) for s, ps, reg, n, cal in rows]
    reg = registrations[0]
    digests = sorted({r["digest"] for r in registrations})
    gammas = [r["gamma"] for r in rules if r["gamma"] is not None]
    return {
        "schema": SCHEMA, "trial": True,
        "runs": len(outs), "same_registration": len(digests) == 1, "registrations": digests,
        "codec": reg.get("codec"),
        "store": {"blocks": reg["blocks"], "block_bytes": block_bytes, "bytes": reg["blocks"] * block_bytes,
                  "depth": max(reg["blocks"] - 1, 0).bit_length(), "rows": reg.get("rows")},
        "coins": COINS.format(challenger="/".join(sorted(c or "unnamed" for c in challengers))),
        "plans": plans,
        "reference": NO_BOUND.format(blocks=reg["blocks"], block_bytes=block_bytes)
        if gammas and min(gammas) >= 1 else None,
        "size": PROCESSOR.format(codec=reg.get("codec")) if all(r["card"] is None for r in rules) else None,
        "rules": rules,
    }


def write(doc: dict, out: Path) -> None:
    """`out/refresh.json`, and through the kit a trial's result naming it, with each accepted rule's γ a measurement."""
    w = dataclasses.replace(kit.writer(out), trial=True)
    doc |= {"commit": w.stamp["commit"], "run_id": w.run}
    (out / "refresh.json").write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")
    w.fingerprint(script="benchmarks/poub_session/refresh_result.py", codec=doc["codec"],
                  store_depth=doc["store"]["depth"], plans=doc["plans"], runs=doc["runs"])
    rel = (out / "refresh.json").resolve().relative_to(w.out.resolve()).as_posix()
    w.measurement_file(rel)
    for r in doc["rules"]:
        w.measurement("uncertified share by rule", r["gamma"], UNIT, mark="measured",
                      member=f"{'refresh-on' if r['refresh'] else 'refresh-off'}-run-{r['run']}-session-{r['session']}",
                      why=None if r["gamma"] is not None else f"the session was not accepted ({r['failure']})",
                      extra={"rule": r})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, nargs="+", metavar="OUT", help="a run's out/ directory; the first is written")
    a = ap.parse_args(argv)
    try:
        doc = document(a.out)
    except (OSError, KeyError, ValueError) as e:
        print(f"refresh_result: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    write(doc, a.out[0])
    print(json.dumps(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
