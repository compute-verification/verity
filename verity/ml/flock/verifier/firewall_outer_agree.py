"""The outer firewall's contract check: `flock-circuit prove --zk` gated sessions (`FC_FIREWALL=1 FC_HOST_WITNESS=1`, the CPU
prover) with `FC_FIREWALL_TRANSCRIPT` recording what the firewall lets out (`live/src/firewall_transcript.rs`), and the contract
(`flock-firewall contract`, `Flock/Firewall/Contract.lean`) on each transcript. `firewall_agree.py` is the inner phase's.

    python firewall_outer_agree.py --lean BIN --out DIR CASE=DIR...

`CASE` is `honest` or an `FC_FIREWALL_TAMPER` control, `DIR` the directory its sessions' transcripts went to (``session-<k>.json``)
with the public input each recorded (``public-<k>.json``). The contract holds every transcript to one public input
(`public_input`, written to DIR/public.json): the honest sessions', which must be one and the same in every honest session,
and with which every control's agrees wherever both name a value. The outer public input has no schedule, so the contract
decides each outer item alone and holds them to no order or number; here every transcript is held to the honest sessions'
item sequence instead (`sequence`: each item's kind and `place`, one and the same in every honest session, and each
control's the same up to its cut, its held-back candidate where those items leave in an honest session). Each transcript must
meet its case's expectation (`EXPECT`): an honest session passes the firewall, the prover does not stop, and the contract
accepts; a control's session is refused by the firewall (or, for `algebra`, stopped by the prover's self-check), and the contract
refuses it for the control's own reason at the item where the firewall stopped (`firewall.gate.at`; for `algebra`, the first
failing check). Every check's outcome is recomputed here from its affine form and the tape's pads (`c + Σ a_i h_i = 0` in
GF(2^128)); the form is the verifier's replay of the masked values and the coins, on the transcript's word. Writes
DIR/agreement.tsv and DIR/result.json; exits 1 if any transcript misses its expectation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

MOD = (1 << 128) | 0x87

#: per case: the firewall's refusal (a prefix) or the prover's stop, and the contract's reason (a prefix and a suffix)
EXPECT = {
    "honest": {"firewall": None, "lean": None},
    "root": {"firewall": "FIREWALL-REFUSED: level-0 root",
             "lean": ("the session's Commit: the level-0 roots", "not the value the firewall's recheck fixes")},
    "witness": {"firewall": "FIREWALL-REFUSED: level-0 root",
                "lean": ("the session's Commit: the level-0 roots", "not the value the firewall's recheck fixes")},
    "label": {"firewall": "FIREWALL-REFUSED: algebraic phase, round 0, frame 1",
              "lean": ("rep 0 algebraic phase round 0 frame 1: label", "not the value the statement fixes")},
    "opening": {"firewall": "FIREWALL-REFUSED: opening round 0, frame 2",
                "lean": ("rep 0 opening round 0 frame 2: ", "not the value the firewall's opening fixes")},
    "algebra": {"firewall": "PROVER-STOPPED: the batched constraint fails", "lean": ("the relation ", " fails")},
    "algebra-proof": {"firewall": "FIREWALL-REFUSED: the proof's exposed value 0",
                      "lean": ("rep 0 proof: the masked zerocheck and lincheck", "not the value the firewall's own values fixes")},
    "shape": {"firewall": "FIREWALL-REFUSED: the proof's shape",
              "lean": ("rep 0 proof: shape", "not the value the statement fixes")},
}


def gmul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> 128:
            a ^= MOD
    return r


def recheck(t: dict) -> tuple[int, int]:
    """Each check's outcome from its form: (agreeing, total)."""
    pads = [int(h, 16) for h in t["tape"]["pads"]]
    agree = total = 0
    for it in t["items"]:
        if "check" not in it:
            continue
        c = it["check"]
        v = int(c["form"]["c"], 16)
        for i, a in c["form"]["a"]:
            v ^= gmul(int(a, 16), pads[i])
        total += 1
        agree += (v == 0) == c["ok"]
    return agree, total


def first_failing_check(t: dict) -> int | None:
    return next((k for k, it in enumerate(t["items"]) if "check" in it and not it["check"]["ok"]), None)


def judge(case: str, t: dict, v: dict) -> tuple[bool, str]:
    """Whether the transcript and the contract's verdict meet the case's expectation, and why not."""
    e, fw = EXPECT[case], t["firewall"]
    if fw["recorder_errors"]:
        return False, f"recorder: {fw['recorder_errors'][0]}"
    why_fw = fw["gate"]["refused"] or fw["stopped"]
    if e["firewall"] is None:
        if why_fw:
            return False, f"the firewall stopped an honest session: {why_fw}"
        return (v["accepted"], "" if v["accepted"] else f"the contract refuses an honest session: {v['why']}")
    if not (why_fw or "").startswith(e["firewall"]):
        return False, f"the firewall's outcome {why_fw!r} is not {e['firewall']!r}"
    if v["accepted"]:
        return False, "the contract accepts a control"
    pre, suf = e["lean"]
    if not (v["why"].startswith(pre) and v["why"].endswith(suf)):
        return False, f"the contract's reason {v['why']!r} is not {pre!r}…{suf!r}"
    at = fw["gate"]["at"] if fw["gate"]["refused"] else first_failing_check(t)
    if v["released"] != at:
        return False, f"the contract cuts at item {v['released']}, the firewall at {at}"
    return True, ""


def public_input(cases: list[tuple[str, list[Path]]]) -> tuple[dict, dict[str, str]]:
    """The public input every transcript is held to: the honest sessions' (each ``public-<k>.json`` beside its transcript),
    which must all be equal; and per transcript of any case, why its own disagrees (a statement value under a name the
    honest one gives another, or a relation it lacks)."""
    pub_of = lambda f: json.loads(f.with_name(f.name.replace("session-", "public-", 1)).read_text())
    honest = [(f, pub_of(f)) for case, files in cases if case == "honest" for f in files]
    if not honest:
        raise SystemExit("no honest session: nothing fixes the public input")
    pub = honest[0][1]
    bad = {}
    for case, files in cases:
        for f in files:
            mine = pub_of(f)
            if case == "honest" and mine != pub:
                bad[str(f)] = "its public input is not the other honest sessions'"
                continue
            clash = next((k for k, v in mine["statement"].items() if pub["statement"].get(k, v) != v), None)
            extra = next((r for r in mine["relations"] if r not in pub["relations"]), None)
            if clash is not None:
                bad[str(f)] = f"its statement value for {clash!r} is not the honest sessions'"
            elif extra is not None:
                bad[str(f)] = f"its relation {extra!r} is not the honest sessions'"
    return pub, bad


def place(it: dict) -> tuple:
    """What fixes an item's place in a session: its kind and the fields that are no value (a fixed item's name, source and
    pads, a masked value's pad, a check's relation and pads, a coins item's stream and round, a commitment's salt)."""
    ((k, v),) = it.items()
    if k == "fixed":
        return (k, v["what"], v["by"], tuple(v["pads"]))
    if k == "masked":
        return (k, v["pad"])
    if k == "check":
        return (k, v["relation"], tuple(v["pads"]))
    if k == "coins":
        return (k, v["stream"], v["round"])
    if k == "commit":
        return (k, v["salt"])
    return (k,)


def _named(p: tuple | None) -> str:
    if p is None:
        return "the end"
    k = p[0]
    if k == "fixed":
        return f"fixed {p[1]!r} by {p[2]}"
    if k in ("masked", "commit"):
        return f"{k} under {'pad' if k == 'masked' else 'salt'} {p[1]}"
    if k == "check":
        return f"check of {p[1]!r}"
    if k == "coins":
        return f"coins of {p[1]} round {p[2]}"
    return k


def _fits(c: tuple, h: tuple) -> bool:
    """A held-back candidate's item at an honest item's place: the same, except that a fixed item's or a check's pads may be
    fewer (its dependencies as of where the firewall checked it, before the honest session's later items used more pads)."""
    if c == h:
        return True
    return c[0] in ("fixed", "check") and c[0] == h[0] and c[1:-1] == h[1:-1] and set(c[-1]) <= set(h[-1])


def sequence_miss(t: dict, honest: list[tuple], whole: bool) -> str:
    """Why a transcript's items are not in the honest sessions' order and number ('' when they are). What the session let out
    must be the honest sequence's beginning: the whole transcript, or, where the firewall refused, the items before the
    candidate it held back (`firewall.gate.from`); with `whole` (an honest session) it must be all of it. The held-back
    candidate's items, up to the one the contract must refuse, must be the honest sequence's run where they leave in an honest
    session (in that order and number, at or after `from`, `_fits`); a stand-in for the refusal (`stand_in`) is no item of it."""
    seq = [place(it) for it in t["items"]]
    g = t["firewall"]["gate"]
    sent = len(seq)
    if g["refused"]:
        sent = g.get("from")
        if not isinstance(sent, int) or not isinstance(g["at"], int) or not 0 <= sent <= g["at"] < len(seq):
            return "the transcript does not say where the firewall's held-back candidate begins (firewall.gate.from)"
    at = lambda s, i: s[i] if i < len(s) else None
    end = max(sent, len(honest)) if whole else sent
    d = next((i for i in range(end) if at(seq, i) != at(honest, i)), None)
    if d is not None:
        return f"its item {d} is {_named(at(seq, d))}, where the honest sessions have {_named(at(honest, d))}"
    cand = seq[sent:]
    if cand and t["items"][-1].get("fixed", {}).get("stand_in"):
        cand = cand[:-1]
    if not cand:
        return ""
    q = next((i for i in range(sent, len(honest)) if _fits(cand[0], honest[i])), None)
    if q is None:
        return f"its held-back candidate begins (item {sent}) with {_named(cand[0])}, which no honest session sends after item {sent}"
    d = next((i for i in range(len(cand)) if q + i >= len(honest) or not _fits(cand[i], honest[q + i])), None)
    if d is not None:
        return (f"its held-back candidate's item {sent + d} is {_named(cand[d])}, where the honest sessions have "
                f"{_named(at(honest, q + d))} (their item {q + d})")
    return ""


def sequence(cases: list[tuple[str, list[Path]]]) -> tuple[list[tuple], dict[str, str]]:
    """The order and number of items every transcript is held to: the honest sessions' (each item's `place`), which must be
    one and the same in every honest session; and per transcript of any case, why its items are not in it
    (`sequence_miss`)."""
    honest = [f for case, files in cases if case == "honest" for f in files]
    if not honest:
        raise SystemExit("no honest session: nothing fixes the item sequence")
    seq = [place(it) for it in json.loads(honest[0].read_text())["items"]]
    bad = {}
    for case, files in cases:
        for f in files:
            why = sequence_miss(json.loads(f.read_text()), seq, case == "honest")
            if why:
                bad[str(f)] = why
    return seq, bad


def contract(lean: str, files: list[Path], public: Path) -> dict[str, dict]:
    p = subprocess.run([lean, "contract", "--public", str(public), *map(str, files)], capture_output=True, text=True, check=False)
    out = {}
    for line in p.stdout.splitlines():
        d = json.loads(line)
        out[d["file"]] = d
    if p.returncode not in (0, 2) or len(out) != len(files):
        raise SystemExit(f"flock-firewall contract: rc {p.returncode}: {p.stderr[-2000:]}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("cases", nargs="+", help="CASE=DIR: a case of EXPECT and its transcripts' directory")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rows, ok_all, cases = [], True, []
    for spec in a.cases:
        case, _, d = spec.partition("=")
        if case not in EXPECT:
            ap.error(f"{case}: not a case of {sorted(EXPECT)}")
        files = sorted(Path(d).glob("session-*.json"), key=lambda p: int(p.stem.split("-")[1]))
        if not files:
            ap.error(f"{d}: no session-*.json")
        cases.append((case, files))
    pub, disagree = public_input(cases)
    seq, out_of_order = sequence(cases)
    (a.out / "public.json").write_text(json.dumps(pub))
    for case, files in cases:
        verdicts = contract(a.lean, files, a.out / "public.json")
        for f in files:
            t, v = json.loads(f.read_text()), verdicts[str(f)]
            if "error" in v:
                good, why = False, f"not a transcript: {v['error']}"
            elif str(f) in disagree:
                good, why = False, disagree[str(f)]
            elif str(f) in out_of_order:
                good, why = False, out_of_order[str(f)]
            else:
                good, why = judge(case, t, v)
            agree, total = recheck(t)
            if agree != total:
                good, why = False, f"{total - agree} of {total} checks' outcomes are not their forms'"
            kinds: dict[str, int] = {}
            for it in t["items"]:
                (k,) = it
                kinds[k] = kinds.get(k, 0) + 1
            fw = t["firewall"]
            rows.append({"case": case, "file": f.name, "firewall": fw["gate"]["refused"] or fw["stopped"], "firewall_at": fw["gate"]["at"],
                         "accepted": v.get("accepted"), "released": v.get("released"), "why": v.get("why"), "items": len(t["items"]),
                         "kinds": kinds, "checks_recomputed": f"{agree}/{total}",
                         "on_its_word": sorted({it["fixed"]["what"] for it in t["items"] if it.get("fixed", {}).get("on_its_word")}),
                         "stand_in": sum(1 for it in t["items"] if it.get("fixed", {}).get("stand_in")), "ok": good, "miss": why})
            ok_all &= good
    cols = ["case", "file", "firewall", "firewall_at", "accepted", "released", "why", "items", "checks_recomputed", "stand_in", "ok", "miss"]
    with open(a.out / "agreement.tsv", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join("" if r[c] is None else str(r[c]).replace("\t", " ") for c in cols) + "\n")
    summary = {"transcripts": len(rows), "ok": sum(r["ok"] for r in rows), "cases": sorted({r["case"] for r in rows}),
               "public": {"relations": len(pub["relations"]), "statement": len(pub["statement"]), "computes": len(pub["computes"])},
               "sequence": len(seq), "rows": rows}
    (a.out / "result.json").write_text(json.dumps(summary, indent=1))
    for r in rows:
        if not r["ok"]:
            print(f"MISS {r['case']} {r['file']}: {r['miss']}", file=sys.stderr)
    print(json.dumps({k: summary[k] for k in ("transcripts", "ok", "cases", "public", "sequence")}))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
