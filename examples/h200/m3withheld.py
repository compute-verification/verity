"""M3's coin under a withheld session: the challenger's coin-commitment root reaches no one before the prover's
registration is in the ledger, and the check refuses a coin whose session released it sooner.

    python examples/h200/m3withheld.py --registration REG.json --main MAIN.txt --pin COMMIT --out DIR

Each case runs against a gate of its own on loopback (`m3gate.LocalGate`, the auditor host's gate as it is to open a
session withheld; plain, as it opens one today), on the registration ``REG.json`` (m3reg's record or a step
registration, as `m3coin` reads it), its host pinned to ``COMMIT`` (on main's first-parent line ``MAIN.txt``), and is
decided by the verifier's side, `m3check.session_audit`, under the gate's key record:

* ``withheld``: `m3coin.coin` on a withheld session; the open's answer carries no root, the hello answers the layout alone,
  and the root comes with the coin's opening at the signed close. The check accepts.
* ``plain``: `m3coin.coin` on a gate that reserves it plain: the client refuses it before its open (the reserve's
  answer isn't ``withheld``). A prover's client that opened it and took the coin anyway, after its registration, saw
  the challenger's root at the hello and is refused by the check: the ledger's open carries no ``withheld: true``.
* ``late``: a prover that commits after seeing the root: it opens a plain session on a placeholder root, reads the
  challenger's root at the hello, and only then fixes its registration (bound to that root) and takes the coin after it.
  The check refuses it. On a withheld gate the same prover sees no root before its registration: the hello carries none.

``DIR/<case>/`` keeps each case's ``coin.json``, the session's tree, ledger and pins log; ``DIR/summary.json`` each case's
verdict. Exit 0 when the honest case is accepted and every other is refused, else 4.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3check as CHECK
import m3coin as COIN
import m3gate as GATE
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import remote as R


def _gate(out: Path, pin: str, utc: str, withholds: bool, name: str) -> GATE.LocalGate:
    seed = hashlib.sha512(f"m3withheld gate {name}".encode()).digest()
    return GATE.LocalGate(out / "gate", seed, pin, utc, withholds=withholds, entropy=f"m3withheld {name}".encode())


def _audit(gate: GATE.LocalGate, out: Path, coin_doc: dict, digest: str, main: Path) -> dict:
    (out / "key-record.json").write_text(json.dumps(gate.key_record()) + "\n")
    return CHECK.session_audit(coin_doc, digest, out, out / "key-record.json", main)


def _keep(gate: GATE.LocalGate, out: Path, sid: str) -> None:
    """The session's tree, the ledger and the pins log as `m3coin.coin` keeps them."""
    gate.status(sid)
    with tarfile.open(fileobj=io.BytesIO(gate(["fetch", sid])), mode="r:gz") as tar:
        tar.extractall(out, filter="data")
    (out / f"{sid}.ledger.jsonl").write_bytes(gate(["ledger"]))
    (out / f"{sid}.pins.jsonl").write_bytes(gate(["pins"]))


def _taken(gate: GATE.LocalGate, peer: str, opened_on: str, after: str) -> tuple[dict, str | None]:
    """A prover's own client: a session reserved and opened on `opened_on`, the root its hello shows (None when
    withheld), and one coin taken after `after`; its ``coin.json`` fields `m3check.session_audit` reads."""
    r = json.loads(gate(["reserve", "--peer", peer, "--scheme", COIN.SCHEME, "--families", json.dumps([[COIN.LABEL, 1]]),
                         "--epoch", "0", "--minutes", "20", "--withhold"]))
    o = json.loads(gate(["open", "--session", r["session"], "--root", opened_on]))
    client = R.RemoteChallenger.connect(f"127.0.0.1:{o['port']}", bytes.fromhex(o["secret"]))
    seen = None if client.withheld else client.commitment.root.hex()
    return {"reserve": r, "open": o, "client": client, "seen": seen}, seen


def _finish(t: dict, after: str) -> dict:
    client = t["client"]
    opened = client.open(COIN.LABEL, 0, bytes.fromhex(after))
    client.close()
    opening = client.opened[(COIN.LABEL, 0)] if client.withheld else opened
    return {"reserve": t["reserve"], "commitment": client.commitment.root.hex(), "opening": opening.to_json()}


def withheld(reg: dict, out: Path, pin: str, main: Path, utc: str, peer: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    gate = _gate(out, pin, utc, True, "withheld")
    rc = COIN.coin(gate, reg, out, gate_name="m3gate (loopback, withholds)", peer=peer, minutes=20,
                   main=R.main_lines(main))
    gate.join()
    doc = json.loads((out / "coin.json").read_text())
    digest = doc["registration"]
    audit = _audit(gate, out, doc, digest, main) if rc == 0 else {"ok": False, "why": doc.get("failure")}
    return {"case": "withheld", "want": "accept", "rc": rc, "root_at_open": doc["open"].get("commitment"),
            "withheld_on": doc.get("withheld_on"), "audit": audit, "accepted": bool(audit.get("ok"))}


def plain(reg: dict, out: Path, pin: str, main: Path, utc: str, peer: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    gate = _gate(out / "client", pin, utc, False, "plain")
    (out / "client").mkdir(parents=True, exist_ok=True)
    rc = COIN.coin(gate, reg, out / "client", gate_name="m3gate (loopback, plain)", peer=peer, minutes=20,
                   main=R.main_lines(main))
    gate.join()
    client = json.loads((out / "client" / "coin.json").read_text())
    digest = client["registration"]
    gate = _gate(out, pin, utc, False, "plain-taken")
    t, seen = _taken(gate, peer, digest, digest)
    doc = _finish(t, digest)
    gate.join()
    _keep(gate, out, doc["reserve"]["session"])
    (out / "coin.json").write_text(json.dumps(doc, indent=1) + "\n")
    audit = _audit(gate, out, doc, digest, main)
    return {"case": "plain", "want": "refuse", "client_rc": rc, "client_failure": client.get("failure"),
            "root_at_hello": seen, "audit": audit, "accepted": bool(audit.get("ok"))}


def late(reg: dict, out: Path, pin: str, main: Path, utc: str, peer: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    digest = COIN.registration_of(reg)[0]
    placeholder = hashlib.sha512(b"m3withheld placeholder " + bytes.fromhex(digest)).hexdigest()
    gate = _gate(out, pin, utc, False, "late")
    t, seen = _taken(gate, peer, placeholder, placeholder)
    chosen = hashlib.sha512(bytes.fromhex(digest) + bytes.fromhex(seen)).hexdigest()
    doc = _finish(t, chosen)
    gate.join()
    _keep(gate, out, doc["reserve"]["session"])
    (out / "coin.json").write_text(json.dumps(doc, indent=1) + "\n")
    audit = _audit(gate, out, doc, chosen, main)
    blind = _gate(out / "withheld-gate", pin, utc, True, "late-withheld")
    tw, seen_withheld = _taken(blind, peer, placeholder, placeholder)
    tw["client"].close()
    blind.join()
    return {"case": "late", "want": "refuse", "root_at_hello": seen, "registration_after_root": chosen,
            "root_at_hello_when_withheld": seen_withheld, "audit": audit, "accepted": bool(audit.get("ok"))}


CASES = {"withheld": withheld, "plain": plain, "late": late}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--main", required=True, help="main's first-parent history (git rev-list --first-parent origin/main)")
    ap.add_argument("--pin", required=True, help="the commit the loopback host pins, on main's first-parent line")
    ap.add_argument("--utc", default="2026-10-27T00:00:00Z", help="the time the loopback gate's entries record")
    ap.add_argument("--peer", default="vy-nebius-cpu-4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out, reg = Path(a.out), json.loads(Path(a.registration).read_text())
    rows = []
    for name, case in CASES.items():
        try:
            row = case(reg, out / name, a.pin, Path(a.main), a.utc, a.peer)
        except (Ch.CoinError, OSError, ValueError, KeyError) as e:
            row = {"case": name, "want": "accept" if name == "withheld" else "refuse", "error": f"{type(e).__name__}: {e}",
                   "accepted": False}
        row["ok"] = row["accepted"] == (row["want"] == "accept") and "error" not in row
        print("CASE " + json.dumps({k: row.get(k) for k in ("case", "want", "accepted", "ok")}), flush=True)
        rows.append(row)
    summary = {"format": "verity/e2e-h200/m3-withheld/v1", "registration": COIN.registration_of(reg)[0], "cases": rows,
               "ok": all(r["ok"] for r in rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("VERDICT " + json.dumps({"ok": summary["ok"], "cases": {r["case"]: r["ok"] for r in rows}}), flush=True)
    return 0 if summary["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
