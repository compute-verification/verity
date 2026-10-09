"""``python rec_private_summary.py OUT``: OUT/summary.json from the records 87-rec-private.sh and 89-rec-private-direct.sh
leave under OUT (whichever of the staging, inner sessions, V*'s folds and stagings, outer sessions and public views are
there), and a line of each outer session's verdicts.

Each verifier's verdict is reported with ``every_check_passed``: Lean's verify refuses a statement whose form the soundness
proof does not cover (`ProvedScope.check`) last, so its reason is that refusal only when every check passed; the live build
gives that refusal whatever the session holds, and the verdict it would otherwise give as ``without_refusal``."""
import json
import re
import sys
from pathlib import Path

FORM = ": a statement form the soundness proof does not cover"


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
    if at < 0:
        return {"accepted": None, "why": (t + text(p.with_suffix(".err")))[-500:]}
    v = json.JSONDecoder(strict=False).raw_decode(t[at + 8:])[0]
    return {**{k: v.get(k) for k in ("accepted", "why", "registered", "partition")},
            "every_check_passed": bool(v.get("accepted")) or str(v.get("why", "")).endswith(FORM)}


def up(p: Path):
    return next((json.loads(ln.split("\t", 1)[1]) for ln in text(p).splitlines() if ln.startswith("UPSTREAM")), text(p)[-400:] or None)


def live(d: Path) -> dict:
    rows = [json.loads(ln[5:]) for ln in text(d / "prove.txt").splitlines() if ln.startswith("LIVE")]
    served = [json.loads(ln.split("\t", 1)[1]) for ln in text(d / "serve.log").splitlines() if ln.startswith("SESSION\t")]
    r = rows[0] if rows else {}
    wr = (r.get("verdict") or {}).get("without_refusal")
    return {"accepted": r.get("accepted"), "why": r.get("why"), "without_refusal": wr,
            "every_check_passed": (wr or {}).get("accepted", r.get("accepted")) if r else None, "firewall": r.get("firewall"),
            "prove_s": r.get("prove_s"), "prove_total_s": r.get("prove_total_s"),
            "e2e_s": r.get("e2e_s"), "proof_bytes": r.get("proof_bytes"), "k_log": r.get("k_log"), "instances": r.get("instances"),
            "serve": [{k: s.get(k) for k in ("accepted", "why", "verify_s")} for s in served],
            "stops": [ln for ln in text(d / "prove.err").splitlines() if ln.startswith(("FIREWALL-REFUSED", "PROVER-STOPPED"))][:4],
            "prove_wall": tv(d / "time-prove.txt", r"Elapsed \(wall clock\) time"), "prover_peak_gb": gb(d / "time-prove.txt")}


def dirs(p: Path) -> list[Path]:
    return sorted(q for q in p.iterdir() if q.is_dir()) if p.is_dir() else []


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
    (o / "summary.json").write_text(json.dumps(s, indent=1))
    return s


if __name__ == "__main__":
    s = main(Path(sys.argv[1]))
    for n, v in s["outer"].items():
        print(n, json.dumps({"live": v["live"]["every_check_passed"], "lean": (v["lean"].get("accepted"), v["lean"].get("why")),
                             "core_firewall": (v["core_firewall"] or "")[:200]}))
