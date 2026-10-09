"""``python rec_private_coins.py --lean BIN --session DIR --out D``: the firewall's contract (`flock-firewall contract`) on the
run's own inner session ``DIR`` (`firewall_agree.replay`: the recorded session through today's firewall again, with its own
coins, salts and the challenger's commitments and openings, ``coins.json``) under the verifier's public input, which holds
the challenger's commitments: as it is (``run-honest``, which the contract accepts), and with the first coin of rep 0's
challenge round, the coin ``rec_inner.py forge --what challenge`` flips, one bit off as the workers received and relayed it
(``run-other-coins``: an inner transcript against coins other than the ones the challenger committed to), which the
contract refuses at that round. Writes D/<case>.json, D/public-0.json and D/result.json (each case's verdict, its
expectation and whether it met it); exits 1 if one missed."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verifier"))
import firewall_agree as FA

STREAM = "circuit/rep0"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", required=True)
    ap.add_argument("--session", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    s, bad = FA.replay(a.session)
    if bad:
        raise SystemExit(f"{a.session}: {bad}")
    res = decide(a.lean, a.out, s)
    print(json.dumps(res))
    return 0 if all(v["ok"] for n, v in res.items() if n != "swapped") else 1


def decide(lean: str, out: Path, s) -> dict:
    """``run-honest`` and ``run-other-coins`` of the replayed session ``s`` (a `firewall_transcript.Session`), decided by the
    contract ``lean`` under ``s``'s public input, into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    honest, pub = s.transcript(), s.public()
    k = next(r["k"] for r in s.coins.record()["rounds"] if r["stream"] == STREAM and r["k"] > 0 and r["n"])
    other = copy.deepcopy(honest)
    item = next(it["coins"] for it in other["items"] if "coins" in it and (it["coins"]["stream"], it["coins"]["round"]) == (STREAM, k))
    c = item["coins"][0]
    item["coins"] = item["relayed"] = [c[:-1] + format(int(c[-1], 16) ^ 1, "x")] + item["coins"][1:]
    want = {"run-honest": {"accepted": True, "why": ""},
            "run-other-coins": {"accepted": False, "why": f"{STREAM} round {k}: coins that do not open the challenger's commitment"}}
    got = FA.decide(lean, out, [("run-honest", honest, pub), ("run-other-coins", other, pub)])
    res = {name: {"verdict": v, "expect": want[name],
                  "ok": v.get("accepted") == want[name]["accepted"] and want[name]["why"] in (v.get("why") or "")}
           for name, v in zip(want, got)}
    res["swapped"] = {"stream": STREAM, "round": k, "coin": 0, "rounds_committed": sum(len(v) for v in pub["coins"].values())}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    sys.exit(main())
