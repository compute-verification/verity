"""``python rec_private_summary.py OUT``: OUT/summary.json from the records 87-rec-private.sh and 89-rec-private-direct.sh
leave under OUT (whichever of the staging, inner sessions, V*'s folds and stagings, outer sessions and public views are
there), a line of each outer session's verdicts, and a last line of the Lean verifier of record's.

Each verdict fails closed: ``accepted`` only when the verifier said so, and a refusal of a statement whose form the soundness
proof does not cover (`ProvedScope.check`) is ``Lean refused: form not covered``, a refusal like any other. The live
build's ``without_refusal`` (the verdict it would give but for that refusal) is reported and counts for nothing.
``lean_of_record`` accepts the run only when Lean accepted every part cap.json names in its session of the recursive
session (NAME, never NAME-direct); a control is ``caught`` only when Lean refused it for a reason other than its form, as the
form's refusal comes after every other check. ``outside-alg-*``, a description outside the class proved as its own
registered statement, is expected not to be caught: every description of the class's width is a member of it. The firewall's
contract on the run's own inner transcript with a coin the challenger did not commit to (``other-coins-contract``) is a
control of Lean's contract. ``salts`` counts the salts in the firewall's transcripts (the inner phase's, every outer
session's): the tape's salts, the challenger's openings of its coin commitments and the tape's pads, each distinct out of
total."""
import json
import re
import sys
from pathlib import Path

FORM = ": a statement form the soundness proof does not cover"
CONTROLS = ("forged-top-", "flipped-top-", "other-alg-", "faulty-alg-", "other-coins-alg-", "outside-alg-")
#: the controls Lean must accept, each with why
ACCEPTED = {"outside-alg-": "a description outside the class is a member of it (every description of its width is)"}


def verdict(who: str, accepted, why) -> str:
    if accepted is True:
        return "accepted"
    if accepted is None:
        return f"{who}: no verdict"
    w = str(why or "")
    return f"{who} refused: form not covered" if w.endswith(FORM) else f"{who} refused: {w.removeprefix('refused: ')}"


def text(p: Path) -> str:
    return p.read_text() if p.exists() else ""


def tv(p: Path, key: str) -> str | None:
    m = re.search(key + r".*: (\S+)", text(p))
    return m.group(1) if m else None


def gb(p: Path) -> float | None:
    rss = tv(p, r"Maximum resident set size \(kbytes\)")
    return round(int(rss) / 2**20, 3) if rss else None


def lean(p: Path) -> dict:
    t = text(p)
    at = t.find("VERDICT ")
    try:
        v = json.JSONDecoder(strict=False).raw_decode(t[at + 8:])[0] if at >= 0 else None
    except ValueError:
        v = None
    if not isinstance(v, dict):
        return {"accepted": False, "verdict": verdict("Lean", None, None), "why": (t + text(p.with_suffix(".err")))[-500:]}
    ok = v.get("accepted") is True
    return {"accepted": ok, "verdict": verdict("Lean", ok, v.get("why")),
            **{k: v.get(k) for k in ("why", "registered", "partition")}}


def up(p: Path):
    return next((json.loads(ln.split("\t", 1)[1]) for ln in text(p).splitlines() if ln.startswith("UPSTREAM")), text(p)[-400:] or None)


def live(d: Path) -> dict:
    rows = [json.loads(ln[5:]) for ln in text(d / "prove.txt").splitlines() if ln.startswith("LIVE")]
    served = [json.loads(ln.split("\t", 1)[1]) for ln in text(d / "serve.log").splitlines() if ln.startswith("SESSION\t")]
    r = rows[0] if rows else {}
    ok = r.get("accepted") is True
    return {"accepted": ok, "verdict": verdict("live", ok if r else None, r.get("why")), "why": r.get("why"),
            "without_refusal": (r.get("verdict") or {}).get("without_refusal"), "firewall": r.get("firewall"),
            "prove_s": r.get("prove_s"), "prove_total_s": r.get("prove_total_s"),
            "e2e_s": r.get("e2e_s"), "proof_bytes": r.get("proof_bytes"), "k_log": r.get("k_log"), "instances": r.get("instances"),
            "serve": [{k: s.get(k) for k in ("accepted", "why", "verify_s")} for s in served],
            "stops": [ln for ln in text(d / "prove.err").splitlines() if ln.startswith(("FIREWALL-REFUSED", "PROVER-STOPPED"))][:4],
            "prove_wall": tv(d / "time-prove.txt", r"Elapsed \(wall clock\) time"), "prover_peak_gb": gb(d / "time-prove.txt")}


def dirs(p: Path) -> list[Path]:
    return sorted(q for q in p.iterdir() if q.is_dir()) if p.is_dir() else []


def of_record(s: dict) -> dict:
    parts = (s["cap"] if isinstance(s["cap"], dict) else {}).get("parts") or []
    seen = {n: v["lean"]["verdict"] for n, v in s["outer"].items()}
    honest = {p: seen.get(p, "no session in the recursive session") for p in parts}
    refused = {p: v for p, v in honest.items() if v != "accepted"} if parts else {"cap.json": "no parts: nothing to accept"}
    return {"accepted": not refused, "verdict": "accepted" if not refused else "not accepted", "parts": honest, "refused": refused,
            "direct": {p: seen[p + "-direct"] for p in parts if p + "-direct" in seen},
            "controls": {n: {"verdict": v, "caught": v.startswith("Lean refused: ") and v != "Lean refused: form not covered"}
                         | {"expect": next(("accepted: " + why for a, why in ACCEPTED.items() if n.startswith(a)), "caught")}
                         for n, v in seen.items() if n.startswith(CONTROLS)} | contract_controls(s)}


def contract_controls(s: dict) -> dict:
    """The run's own transcript with a coin the challenger did not commit to, as the firewall's contract decided it."""
    v = ((s["inner"].get("honest") or {}).get("coins_contract") or {}).get("run-other-coins")
    if not isinstance(v, dict):
        return {}
    got = v["verdict"]
    verdict_ = verdict("Lean", got.get("accepted"), got.get("why"))
    return {"other-coins-contract": {"verdict": verdict_, "caught": got.get("accepted") is False and v["ok"], "expect": "caught",
                                     "why_expected": v["expect"]["why"]}}


def salts(o: Path) -> dict:
    """Every salt in the firewall's transcripts: the inner phase's (`firewall_agree`'s replay of the honest session) and each
    outer session's (FC_FIREWALL_TRANSCRIPT), distinct out of total, by kind and in all."""
    files = sorted((o / "inner" / "firewall-contract").glob("session-*.json")) + sorted((o / "outer").glob("*/transcript/session-*.json"))
    seen = {"tape_salts": [], "coin_openings": [], "pads": []}
    for f in files:
        t = json.loads(f.read_text())
        seen["tape_salts"] += (t.get("tape") or {}).get("salts") or []
        seen["pads"] += (t.get("tape") or {}).get("pads") or []
        seen["coin_openings"] += [it["coins"]["opening"] for it in t.get("items", []) if "coins" in it and it["coins"].get("opening")]
    every = seen["tape_salts"] + seen["coin_openings"]
    return {"transcripts": [str(f.relative_to(o)) for f in files],
            **{k: {"distinct": len(set(v)), "total": len(v)} for k, v in seen.items()},
            "salts_and_openings": {"distinct": len(set(every)), "total": len(every)}}


def main(o: Path) -> dict:
    s = {"stage": {k: json.loads(text(o / f"stage-{k}.json") or "null") for k in ("universal", "wired", "outside", "faulty")},
         "cap": json.loads(text(o / "cap.json") or "null"), "inner": {}, "fold": {}, "vstar": {}, "outer": {},
         "public": json.loads(text(o / "public" / "public.json") or "null")}
    for d in dirs(o / "inner"):
        if (d / "prove.txt").exists():
            s["inner"][d.name] = {**live(d), "replay": up(d / "replay.txt"), "sizes": text(d / "sizes.txt")}
    if "honest" in s["inner"]:
        h = o / "inner" / "honest"
        s["inner"]["honest"]["lean_plain_leaves"] = lean(h / "lean.log") if text(h / "lean.log") else text(h / "lean.err")[-400:]
        s["inner"]["honest"]["firewall_contract"] = {
            "result": json.loads(text(o / "inner" / "firewall-contract" / "result.json") or "null"),
            "agreement": text(o / "inner" / "firewall-contract" / "agreement.tsv")}
        s["inner"]["honest"]["firewall_cases"] = {
            "result": json.loads(text(o / "inner" / "firewall-cases" / "result.json") or "null"),
            "challenger": [ln for ln in text(o / "inner" / "firewall-cases" / "agreement.tsv").splitlines()
                           if ln.startswith(("case", "challenger-", "tamper-uncommitted", "tamper-wrong-opening", "tamper-no-opening"))]}
        s["inner"]["honest"]["coins_contract"] = json.loads(text(o / "inner" / "coins-contract" / "result.json") or "null")
    if "made-up" in s["inner"]:
        own = text(o / "inner" / "made-up" / "own-links.txt")
        s["inner"]["made-up"]["under_the_verifiers_links"] = next(
            (json.loads(ln.split("\t", 1)[1]) for ln in own.splitlines() if ln.startswith("REFUSED\t")), own[-400:])
    for f in sorted((o / "vstar").glob("fold-*.exit")):
        n = f.stem[5:]
        s["fold"][n] = {"exit": int(text(f).strip() or -1), "err": text(o / "vstar" / f"fold-{n}.err")[-600:],
                        "wall": tv(o / "vstar" / f"time-fold-{n}.txt", r"Elapsed \(wall clock\) time")}
    for n in ("honest", "forged-top", "other", "flipped-checked", "flipped", "faulty", "other-coins", "outside"):
        t = text(o / "vstar" / f"{n}.json")
        if t or text(o / "vstar" / f"{n}.err"):
            s["vstar"][n] = {"stage": json.loads(t.splitlines()[0]) if t.strip() else text(o / "vstar" / f"{n}.err")[-800:],
                             "wall": tv(o / "vstar" / f"time-{n}.txt", r"Elapsed \(wall clock\) time")}
    for n in ("vstar-honest", "vstar-flipped"):
        if (o / "vstar" / f"{n}.json").exists():
            s["vstar"][n] = (text(o / "vstar" / f"{n}.json") + text(o / "vstar" / f"{n}.err"))[-800:]
    if (o / "vstar" / "forge-top.json").exists():
        s["vstar"]["forge_top"] = json.loads(text(o / "vstar" / "forge-top.json") or "null")
    for d in dirs(o / "outer"):
        s["outer"][d.name] = {"live": live(d), "lean": lean(d / "lean.log"), "lean_wall": tv(d / "time-lean.txt", r"Elapsed \(wall clock\) time"),
                              "replay": up(d / "replay.txt"),
                              "core_firewall": text(d / "core-firewall.txt").strip() or text(d / "core-firewall.err")[-300:] or None,
                              "sizes": text(d / "sizes.txt")}
    s["salts"] = salts(o)
    s["lean_of_record"] = of_record(s)
    (o / "summary.json").write_text(json.dumps(s, indent=1))
    return s


if __name__ == "__main__":
    s = main(Path(sys.argv[1]))
    for n, v in s["outer"].items():
        print(n, json.dumps({"live": v["live"]["verdict"], "lean": v["lean"]["verdict"],
                             "core_firewall": (v["core_firewall"] or "")[:200]}))
    r = s["lean_of_record"]
    print("CONTROLS", json.dumps({n: [c["verdict"], c["caught"]] for n, c in r["controls"].items()}))
    print("SALTS", json.dumps({k: v for k, v in s["salts"].items() if k != "transcripts"}))
    print("LEAN-OF-RECORD", r["verdict"], json.dumps(r["refused"]))
