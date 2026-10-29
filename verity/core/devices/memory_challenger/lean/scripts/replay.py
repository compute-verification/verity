"""Replay recorded memory-challenger audits through the Lean verdict (`verdict.py`) and `memory_challenger.verdict`, and
compare both with the verdict the run recorded.

    uv run python verity/core/devices/memory_challenger/lean/scripts/replay.py BENCH_RESULT.json... [--json OUT]

Each file is a `bench-result.json` of `benchmarks/pous/pod.sh IMPL=p2-v3-mc-gpu`, whose `metrics` hold the timed audit
(`audit`) and its controls (`value_control`, `late_control`, `recompute_control`, `tampered_control`), each the output of
`benchmarks/pous/memory_challenger/live.py`.  Every record is judged as `live.judge` judged it: under
`memory_challenger.SCHEMES` (its `accepted`, `witness.reasons`) and under the signer's own mapping (`accepted_under_signer`,
`witness.reasons_under_signer`; `..._under_test_signer` in older records), the latter only where the signer's scheme is one
today's code verifies.  The timed audit's sessions are also judged under `live.signature_control`'s two edits (a signature
bit flipped, a receipt time moved after signing), as that gate judges them.

A record from before `live.py` kept C's digests at the revealed labels (`witness.expected`) holds no table of them: `live.py`
read `DIGESTS.bin` and kept, per answer, only whether its digest was that label's (`answers[j].correct`, `r.digest ==
expected(r.index)`).  `verdict` reads `expected` only through that equality, so the replay rebuilds `expected` at the
revealed labels from those bits: a correct round's digest is the label's, and a label whose rounds were all wrong gets a
digest none of them has.  Every other input is the record's own:
the signed session, the commitment, the openings and the signer's public key, as `live.py` passed them."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from verity.core.protocols.pous import memory_challenger as mc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
SECTIONS = ("audit", "value_control", "late_control", "recompute_control", "tampered_control")


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load("memory_challenger_lean_verdict", HERE.parent / "verdict.py")
TS = _load("pous_test_signer", ROOT / "verity/core/protocols/pous/tests/pous_test_signer.py")


def expected_of(rec: dict[str, Any], signed: mc.SignedSession):
    """C's digests at the revealed labels: the record's own, else rebuilt from its per-answer `correct` bits (module
    docstring)."""
    if "expected" in rec["witness"]:
        return {int(i): bytes.fromhex(h) for i, h in rec["witness"]["expected"].items()}.get
    answers = rec["answers"]
    if len(answers) != len(signed.rounds) or any(a["index"] != r.index for a, r in zip(answers, signed.rounds, strict=True)):
        raise ValueError("the record's answers are not its signed rounds")
    right: dict[int, bytes] = {}
    wrong: dict[int, set[bytes]] = {}
    for a, r in zip(answers, signed.rounds, strict=True):
        if a["correct"]:
            if right.setdefault(r.index, r.digest) != r.digest:
                raise ValueError(f"two different digests of block {r.index} are both correct")
        else:
            wrong.setdefault(r.index, set()).add(r.digest)
    for i, ds in wrong.items():
        if i in right and right[i] in ds:
            raise ValueError(f"block {i}'s digest is both correct and wrong")
        right.setdefault(i, hashlib.sha512(b"none of these: " + b"".join(sorted(ds))).digest())
    return right.get


def _stored(rec: dict[str, Any], under_signer: bool) -> tuple[bool, list[str]] | None:
    if not under_signer:
        return rec["accepted"], rec["witness"]["reasons"]
    for suffix in ("signer", "test_signer"):
        if f"accepted_under_{suffix}" in rec:
            return rec[f"accepted_under_{suffix}"], rec["witness"].get(f"reasons_under_{suffix}")
    return None


def judge(signed, commitment, openings, expected, trusted, schemes) -> dict[str, Any]:
    python = mc.verdict(signed, commitment, openings, expected, trusted=trusted, schemes=schemes)
    lean = V.verdict(signed, commitment, openings, expected, trusted=trusted, schemes=schemes)
    return {"python": python, "lean": lean, "same": lean == python}


def replay(doc: dict[str, Any]) -> dict[str, Any]:
    """Every verdict of one run, each as the Lean and today's Python give it and as the run recorded it."""
    calls = []
    for section in SECTIONS:
        out = (doc.get("metrics") or {}).get(section)
        if not isinstance(out, dict) or not out.get("records"):
            continue
        signer = out["signer"]
        trusted = {bytes.fromhex(signer["public_key"])}
        own = mc.SCHEMES if signer["scheme"] in mc.SCHEMES else TS.SCHEMES if signer["scheme"] in TS.SCHEMES else None
        for n, rec in enumerate(out["records"]):
            signed = mc.SignedSession.from_json(rec["witness"]["signed"])
            commitment = mc.SessionCommitment.from_json(rec["public"]["commitment"])
            openings = [mc.CoinOpening.from_json(o) for o in rec["public"]["openings"]]
            expected = expected_of(rec, signed)
            mappings = [("SCHEMES", mc.SCHEMES, False)] + ([("signer", own, True)] if own is not None else [])
            for name, schemes, under_signer in mappings:
                stored = _stored(rec, under_signer)
                j = judge(signed, commitment, openings, expected, trusted, schemes)
                calls.append({"section": section, "record": n, "schemes": name, "edit": None, **j,
                              "stored": stored, "control": section != "audit", "scheme": signed.scheme})
            if section == "audit" and own is not None:
                sig = bytearray(signed.signature)
                sig[len(sig) // 2] ^= 1
                r = signed.rounds[len(signed.rounds) // 2]
                rounds = list(signed.rounds)
                rounds[r.j] = mc.Round(r.j, r.index, r.send_ns, r.receipt_ns - 1, r.digest)
                for edit, s in (("signature bit flipped", dataclasses.replace(signed, signature=bytes(sig))),
                                ("receipt moved after signing", dataclasses.replace(signed, rounds=tuple(rounds)))):
                    j = judge(s, commitment, openings, expected, trusted, own)
                    calls.append({"section": section, "record": n, "schemes": "signer", "edit": edit, **j,
                                  "stored": None, "control": True, "scheme": s.scheme})
    return {"run_id": doc.get("run_id"), "calls": calls}


def _agrees_with_record(c: dict[str, Any]) -> bool | str | None:
    """True when today's Python gives the recorded verdict; `"retired"` when it differs only in refusing the signer's scheme,
    which `SCHEMES` no longer accepts (the record's version is retired); None when nothing was recorded to compare."""
    if c["stored"] is None:
        return None if c["edit"] is None else "the signature does not verify" in c["python"].reasons
    accepted, reasons = c["stored"]
    now = list(c["python"].reasons)
    if c["python"].accepted == accepted and (reasons is None or now == list(reasons)):
        return True
    retired = f"signature scheme {c['scheme']!r} is not accepted"
    return "retired" if reasons is not None and now == [retired, *reasons] and c["scheme"] not in mc.SCHEMES else False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", default=None, help="write each run's summary and every disagreement here")
    a = ap.parse_args(argv)
    runs = []
    for f in a.files:
        doc = json.loads(Path(f).read_text())
        r = replay(doc)
        calls = r["calls"]
        metrics = doc.get("metrics") or {}
        signers = sorted({metrics[s]["signer"]["scheme"] for s in SECTIONS
                          if isinstance(metrics.get(s), dict) and metrics[s].get("records")})
        summary = {"run_id": r["run_id"], "file": f, "signers": signers, "calls": len(calls),
                   "sessions": len({(c["section"], c["record"]) for c in calls}),
                   "same": sum(c["same"] for c in calls), "controls": sum(c["control"] for c in calls),
                   "accepted": sum(c["lean"].accepted for c in calls),
                   "matches_record": sum(_agrees_with_record(c) is True for c in calls),
                   "retired_scheme": sum(_agrees_with_record(c) == "retired" for c in calls),
                   "unrecorded": sum(_agrees_with_record(c) is None for c in calls),
                   "differences": [{k: (dataclasses.asdict(v) if isinstance(v, mc.Verdict) else v) for k, v in c.items()}
                                   for c in calls if not c["same"] or _agrees_with_record(c) is False]}
        summary["as_recorded"] = summary["matches_record"] + summary["retired_scheme"] + summary["unrecorded"] == len(calls)
        runs.append(summary)
        print(f"{summary['run_id']}: {summary['calls']} verdicts over {summary['sessions']} sessions ({', '.join(signers)}), "
              f"{summary['same']} same verdict, {summary['controls']} negative controls, {summary['accepted']} accepted, "
              f"{summary['matches_record']} as recorded, {summary['retired_scheme']} as recorded but for a retired scheme, "
              f"{len(summary['differences'])} differences", flush=True)
    judged = [r for r in runs if r["calls"]]
    calls = sum(r["calls"] for r in judged)
    same_runs = sum(r["same"] == r["calls"] for r in judged)
    line = (f"{len(judged)} runs, {same_runs} same verdict, {sum(r['controls'] for r in judged)} negative controls "
            f"({calls} verdicts, {sum(r['same'] for r in judged)} same; {sum(r['as_recorded'] for r in judged)} runs as "
            f"recorded; {len(runs) - len(judged)} files hold no session)")
    print(line)
    if a.json:
        Path(a.json).write_text(json.dumps({"line": line, "runs": runs}, indent=1) + "\n")
    return 0 if same_runs == len(judged) and all(r["as_recorded"] for r in judged) else 1


if __name__ == "__main__":
    sys.exit(main())
