"""``python rec_private_summary.py OUT``: OUT/summary.json from the records 87-rec-private.sh and 89-rec-private-direct.sh
leave under OUT (whichever of the staging, inner sessions, V*'s folds and stagings, outer sessions and public views are
there), a line of each outer session's verdicts, and a last line of the Lean verifier of record's.

Each verdict fails closed: ``accepted`` only when the verifier said so, and a refusal of a statement whose form the soundness
proof does not cover (`ProvedScope.check`) is ``Lean refused: form not covered``, a refusal like any other. The live
build's ``without_refusal`` (the verdict it would give but for that refusal) is reported and counts for nothing.
``lean_of_record`` accepts the run only when Lean accepted every part cap.json names in its session of the recursive
session (NAME, never NAME-direct); a control is ``caught`` only when Lean refused it for a reason other than its form, as the
form's refusal comes after every other check."""
import json
import re
import sys
from pathlib import Path

FORM = ": a statement form the soundness proof does not cover"
CONTROLS = ("forged-top-", "flipped-top-", "other-alg-")


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
                         for n, v in seen.items() if n.startswith(CONTROLS)}}


def main(o: Path) -> dict:
    s = {"stage": {k: json.loads(text(o / f"stage-{k}.json") or "null") for k in ("universal", "wired")},
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
    if "made-up" in s["inner"]:
        own = text(o / "inner" / "made-up" / "own-links.txt")
        s["inner"]["made-up"]["under_the_verifiers_links"] = next(
            (json.loads(ln.split("\t", 1)[1]) for ln in own.splitlines() if ln.startswith("REFUSED\t")), own[-400:])
    for f in sorted((o / "vstar").glob("fold-*.exit")):
        n = f.stem[5:]
        s["fold"][n] = {"exit": int(text(f).strip() or -1), "err": text(o / "vstar" / f"fold-{n}.err")[-600:],
                        "wall": tv(o / "vstar" / f"time-fold-{n}.txt", r"Elapsed \(wall clock\) time")}
    for n in ("honest", "forged-top", "other", "flipped-checked", "flipped"):
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
    s["lean_of_record"] = of_record(s)
    (o / "summary.json").write_text(json.dumps(s, indent=1))
    return s


if __name__ == "__main__":
    s = main(Path(sys.argv[1]))
    for n, v in s["outer"].items():
        print(n, json.dumps({"live": v["live"]["verdict"], "lean": v["lean"]["verdict"],
                             "core_firewall": (v["core_firewall"] or "")[:200]}))
    r = s["lean_of_record"]
    print("LEAN-OF-RECORD", r["verdict"], json.dumps(r["refused"]))
