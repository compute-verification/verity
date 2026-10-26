"""M3's live coin (PRF-172): one session of the auditor host's challenger (vy-challenger-1, `benchmarks/live_challenger/
auditor.py`) under the lane's own gate key, reserved for this machine, opened at the registration's digest, and one coin
opened after that root; the drawn units are a `subset:K` of the registered population derived from that coin alone.

    python benchmarks/e2e_h200/m3coin.py --registration REG.json --gate-key KEY --gate research@HOST --peer NAME --out DIR

* The gate's ledger signs the session's `open` entry, which binds the registration's digest as the window root, before
  `serve.py` releases any coin; the client records that same digest as the coin's `after`.
* `units = derive(coin, DOMAIN, {"registration": digest}).subset(k, population, "m3")` (`verity.ml.randomness`): the
  prover can't compute it before the coin opens, and the coin opens only after the registration is in the ledger.
* One draw. Any failure after the open is written to ``coin.json`` as the visible failure, and nothing is retried: a
  second session over the same root is refused by the ledger, and this script never asks for another.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import remote
from verity.ml import randomness as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3reg as REG  # noqa: E402

SCHEME = "verity/e2e-h200-m3/coins/v1"
LABEL = "m3-draw"
DOMAIN = "verity/e2e-h200-m3/draw/v1"


def gate(ssh: list[str], words: list[str]) -> tuple[int, bytes]:
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", *ssh, shlex.join(words)],
                       capture_output=True, stdin=subprocess.DEVNULL)
    return p.returncode, p.stdout if p.returncode == 0 else p.stdout + p.stderr


def draw(coin: bytes, registration: str, k: int, population: int) -> list[int]:
    return list(R.derive(coin, DOMAIN, {"registration": registration}).subset(k, population, "m3"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--gate-key", required=True)
    ap.add_argument("--gate", default="research@204.12.178.84")
    ap.add_argument("--peer", required=True, help="this machine's name in the auditor host's /etc/vy/peers")
    ap.add_argument("--minutes", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--main", required=True, help="main's first-parent history (git rev-list --first-parent origin/main)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = json.loads(Path(a.registration).read_text())
    digest = reg["digest"]
    if REG.digest(REG.FORMAT, reg["record"]) != digest:
        raise SystemExit("the registration's digest is not its record's")
    law, population = reg["record"]["law"], int(reg["record"]["population"])
    if law != {"law": "subset", "k": law.get("k")} or not isinstance(law["k"], int):
        raise SystemExit(f"the registered law is {law}, not subset:K")
    ssh = ["-i", a.gate_key, "-o", "StrictHostKeyChecking=accept-new", a.gate]
    doc: dict = {"format": "verity/e2e-h200-m3/coin/v1", "registration": digest, "gate": a.gate, "peer": a.peer,
                 "scheme": SCHEME, "families": [[LABEL, 1]], "domain": DOMAIN, "law": law, "population": population}

    def write(**kw) -> None:
        doc.update(kw)
        (out / "coin.json").write_text(json.dumps(doc, indent=1) + "\n")

    def ask(words: list[str]) -> bytes:
        rc, got = gate(ssh, words)
        if rc != 0:
            raise Ch.CoinError(f"the gate answered {words[0]} with exit {rc}: {got.decode(errors='replace').strip()}")
        return got

    pins = [x for x in ask(["pins"]).decode().split("\n") if x]
    pin = json.loads(pins[-1]).get("commit") if pins else None
    doc["host_pin"] = pin
    if pin not in remote.main_lines(a.main):
        write(failure=f"refused before reserve: the host's last pin {pin} is not on main's first-parent line")
        raise SystemExit(f"the challenger host pins {pin} last, not a commit on main's first-parent line ({a.main}): its "
                         "session's coins would be refused, so the one session isn't spent")
    pub = json.loads(ask(["pubkey"]))
    doc["challenger_key"] = {"W": pub["W"], "H": pub["H"], "public_sha512": hashlib.sha512(bytes.fromhex(pub["public"])).hexdigest()}
    r = json.loads(ask(["reserve", "--peer", a.peer, "--scheme", SCHEME, "--families", json.dumps([[LABEL, 1]]),
                        "--epoch", "0", "--minutes", str(a.minutes)]))
    write(reserve=r)
    print("RESERVED " + json.dumps(r), flush=True)
    try:
        o = json.loads(ask(["open", "--session", r["session"], "--root", digest]))
    except Ch.CoinError as e:
        write(failure=f"open: {e}")
        print(f"FAILED open: {e}", flush=True)
        return 2
    write(open={k: v for k, v in o.items() if k != "secret"})
    print("OPENED " + json.dumps(doc["open"]), flush=True)
    try:
        commitment = o.get("commitment")
        for _ in range(120):
            if commitment:
                break
            time.sleep(1)
            commitment = json.loads(ask(["status", r["session"]])).get("commitment")
        if not commitment:
            raise Ch.CoinError(f"session {r['session']} published no coin commitment")
        client = None
        for _ in range(600):
            try:
                client = remote.RemoteChallenger.connect(f"127.0.0.1:{o['port']}", bytes.fromhex(o["secret"]))
                break
            except ConnectionRefusedError:
                time.sleep(1)
        if client is None:
            raise Ch.CoinError(f"session {r['session']} never listened at 127.0.0.1:{o['port']}")
        if client.commitment.root.hex() != commitment:
            client.close()
            raise Ch.CoinError(f"the challenger commits to {client.commitment.root.hex()[:16]}…, not the open's {commitment[:16]}…")
        after = bytes.fromhex(digest)
        opening = client.open(LABEL, 0, after)
        why = Ch.live_reason(client.commitment, opening, LABEL, 0, after)
        if why is not None:
            raise Ch.CoinError(f"the opening doesn't verify: {why}")
        client.close()
    except (Ch.CoinError, OSError, ValueError, KeyError) as e:
        write(failure=f"after open: {e}")
        print(f"FAILED after open (no redraw): {e}", flush=True)
        return 3
    units = draw(opening.coin, digest, law["k"], population)
    write(commitment=commitment, opening=opening.to_json(), units=units,
          derivation=f"verity.ml.randomness.derive(coin, {DOMAIN!r}, {{'registration': digest}}).subset({law['k']}, {population}, 'm3')")
    print("DRAW " + json.dumps({"units": units}), flush=True)
    sid = r["session"]
    end = None
    for _ in range(120):
        if (end := json.loads(ask(["status", sid])).get("end")) is not None:
            break
        time.sleep(2)
    try:
        with tarfile.open(fileobj=io.BytesIO(ask(["fetch", sid])), mode="r:gz") as tar:
            tar.extractall(out, filter="data")
        (out / f"{sid}.ledger.jsonl").write_bytes(ask(["ledger"]))
        (out / f"{sid}.pins.jsonl").write_bytes(ask(["pins"]))
    except (Ch.CoinError, OSError, tarfile.TarError) as e:
        write(fetch_failure=str(e))
    write(end=end and end.get("body"))
    print("END " + json.dumps(doc["end"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
