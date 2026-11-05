"""M3's live coin: one withheld session of the auditor host's challenger (`benchmarks/live_challenger/auditor.py`) under
the lane's own gate key, reserved for this machine, opened at the registration's digest, and one coin opened after that
root; the sampled proof blocks are a `subset:K` of the registered population derived from that coin alone.

    python examples/h200/m3coin.py --registration REG.json --gate-key KEY --gate research@HOST --peer NAME --out DIR

``REG.json`` is m3reg's record, or a step registration (`replay_bundle.STEP_FORMAT`, M3 proper), whose digest is
`replay_bundle.step_digest`'s; for one, ``receipt.json`` (the step's receipt, as `prove_bundle.py --step` reads it) is written
before the session is reserved and ``draw.json`` (``{law, k, population, units}``, sorted) after the coin.

* The gate's ledger signs the session's `open` entry, which binds the registration's digest as the window root, before
  `serve.py` releases any coin; the client records that same digest as the coin's `after`.
* The session is withheld (``reserve --withhold``; the gate serves it with `remote.open_session(..., withhold=True)`):
  the gate's open answers no root (``commitment`` null until the session's end, ``withheld: true``, ``signed: true``),
  the challenger answers the hello with its layout alone and the coin with its raw bytes, and releases its coin
  commitment's root and the coin's salt and path only at its signed close, which the client checks against the coin it
  was given (`RemoteChallenger.close`). So the challenger's root reaches no one before the prover's registration is in
  the ledger, and soundness needs no hiding premise on it; the ledger's open carries ``withheld: true``, which the check
  requires (`m3check.session_audit`, `challenger.withheld_on`). A reserve the gate doesn't mark withheld is refused
  before its open, an open answer that hands out the root or isn't withheld is refused, and so is a session whose
  challenger answers the hello with its root, each before any coin.
* `units = derive(coin, DOMAIN, {"registration": digest}).subset(k, population, "m3")` (`verity.ml.randomness`): the
  prover can't compute it before the coin opens, and the coin opens only after the registration is in the ledger.
* One sample. Any failure after the open is written to ``coin.json`` as the visible failure, and nothing is retried: a
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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import remote
from verity.ml import randomness as R

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent / "interfaces")]
import m3reg as REG  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registration as OR  # noqa: E402
from verity_e2e_demo import replay_bundle as RB  # noqa: E402

SCHEME = "verity/e2e-h200-m3/coins/v1"
LABEL = "m3-draw"
DOMAIN = "verity/e2e-h200-m3/draw/v1"

#: the gate as the client asks it: its verb's words, its answer's bytes (`CoinError` when it refuses)
Ask = Callable[[list[str]], bytes]


def gate(ssh: list[str], words: list[str]) -> tuple[int, bytes]:
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", *ssh, shlex.join(words)],
                       capture_output=True, stdin=subprocess.DEVNULL)
    return p.returncode, p.stdout if p.returncode == 0 else p.stdout + p.stderr


def ssh_ask(ssh: list[str]) -> Ask:
    """The gate at `ssh` as an `Ask`."""
    def ask(words: list[str]) -> bytes:
        rc, got = gate(ssh, words)
        if rc != 0:
            raise Ch.CoinError(f"the gate answered {words[0]} with exit {rc}: {got.decode(errors='replace').strip()}")
        return got
    return ask


def draw(coin: bytes, registration: str, k: int, population: int) -> list[int]:
    return list(R.derive(coin, DOMAIN, {"registration": registration}).subset(k, population, "m3"))


def registration_of(reg: dict) -> tuple[str, dict, int, bool]:
    """A registration's digest, law, population and whether it is a step registration; SystemExit when it is neither
    form or its law is not subset:K."""
    step = reg.get("format") == RB.STEP_FORMAT
    if step:
        if set(reg) - {RB.STEPS_KEY} != RB.STEP_KEYS or reg["population"] != reg["slots"] * reg["units_per_slot"]:
            raise SystemExit(f"not a {RB.STEP_FORMAT} with exactly {sorted(RB.STEP_KEYS)} and optionally {RB.STEPS_KEY}")
        digest, law, population = RB.step_digest(reg), reg["law"], int(reg["population"])
    else:
        digest = reg["digest"]
        if REG.digest(REG.FORMAT, reg["record"]) != digest:
            raise SystemExit("the registration's digest is not its record's")
        law, population = reg["record"]["law"], int(reg["record"]["population"])
    if law != {"law": "subset", "k": law.get("k")} or not isinstance(law["k"], int):
        raise SystemExit(f"the registered law is {law}, not subset:K")
    return digest, law, population, step


def coin(ask: Ask, reg: dict, out: Path, *, gate_name: str, peer: str, minutes: int, main: list[str],
         connect: Callable[..., Any] | None = None, sleep: Callable[[float], None] = time.sleep) -> int:
    """The one coin for registration `reg` from the gate `ask` (module docstring), its record ``coin.json`` and, for a
    step registration, ``receipt.json`` and ``draw.json`` in `out`, with the session's tree, ledger and pins log as the
    gate gives them once it ended; `main` is main's first-parent history, which the host's last pin must be on. Exit
    0, or 2 when the open failed and 3 when anything after it did (no redraw)."""
    digest, law, population, step = registration_of(reg)
    doc: dict = {"format": "verity/e2e-h200-m3/coin/v1", "registration": digest, "gate": gate_name, "peer": peer,
                 "scheme": SCHEME, "families": [[LABEL, 1]], "domain": DOMAIN, "law": law, "population": population,
                 "withhold": True}

    def write(**kw) -> None:
        doc.update(kw)
        (out / "coin.json").write_text(json.dumps(doc, indent=1) + "\n")

    if step:
        receipt = {**OR.receipt(reg, f"vy-challenger-1's gate ({gate_name}), session root the step registration's digest"),
                   "registration": digest}
        (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    pins = [x for x in ask(["pins"]).decode().split("\n") if x]
    pin = json.loads(pins[-1]).get("commit") if pins else None
    doc["host_pin"] = pin
    if pin not in main:
        write(failure=f"refused before reserve: the host's last pin {pin} is not on main's first-parent line")
        raise SystemExit(f"the challenger host pins {pin} last, not a commit on main's first-parent line: its "
                         "session's coins would be refused, so the one session isn't spent")
    pub = json.loads(ask(["pubkey"]))
    doc["challenger_key"] = {"W": pub["W"], "H": pub["H"], "public_sha512": hashlib.sha512(bytes.fromhex(pub["public"])).hexdigest()}
    r = json.loads(ask(["reserve", "--peer", peer, "--scheme", SCHEME, "--families", json.dumps([[LABEL, 1]]),
                        "--epoch", "0", "--minutes", str(minutes), "--withhold"]))
    write(reserve=r)
    print("RESERVED " + json.dumps(r), flush=True)
    sid = r["session"]
    if r.get("withheld") is not True:
        write(failure=f"refused before open: the gate reserved session {sid} plain, so it is not withheld: its "
                      "challenger would hand out its coin commitment's root before the registration is in")
        print(f"FAILED reserve: {doc['failure']}", flush=True)
        return 2
    try:
        o = json.loads(ask(["open", "--session", sid, "--root", digest]))
        if o.get("withheld") is not True or o.get("commitment") is not None or o.get("signed") is not True:
            raise Ch.CoinError(f"the gate's open of {sid} is not withheld: its answer must say `withheld: true` and "
                               f"`signed: true` and carry no coin commitment root, not "
                               f"{json.dumps({k: o.get(k) for k in ('withheld', 'signed', 'commitment')})}")
    except Ch.CoinError as e:
        write(failure=f"open: {e}")
        print(f"FAILED open: {e}", flush=True)
        return 2
    write(open={k: v for k, v in o.items() if k != "secret"})
    print("OPENED " + json.dumps(doc["open"]), flush=True)
    try:
        client = remote.connect_when_listening(f"127.0.0.1:{o['port']}", bytes.fromhex(o["secret"]), connect, sleep=sleep)
        if not client.withheld:
            client.close()
            raise Ch.CoinError(f"session {sid} is not withheld: its challenger answered the hello with its coin "
                               "commitment's root, so the gate opened it plain")
        after = bytes.fromhex(digest)
        client.open(LABEL, 0, after)
        client.close()
        opening, commitment = client.opened[(LABEL, 0)], client.commitment.root.hex()
    except (Ch.CoinError, OSError, ValueError, KeyError) as e:
        write(failure=f"after open: {e}")
        print(f"FAILED after open (no redraw): {e}", flush=True)
        return 3
    units = draw(opening.coin, digest, law["k"], population)
    write(commitment=commitment, opening=opening.to_json(), units=units,
          derivation=f"verity.ml.randomness.derive(coin, {DOMAIN!r}, {{'registration': digest}}).subset({law['k']}, {population}, 'm3')")
    if step:
        (out / "draw.json").write_text(json.dumps({"law": "subset", "k": law["k"], "population": population,
                                                    "units": sorted(units)}, sort_keys=True) + "\n")
    print("DRAW " + json.dumps({"units": units}), flush=True)
    end = None
    for _ in range(120):
        if (end := json.loads(ask(["status", sid])).get("end")) is not None:
            break
        sleep(2)
    try:
        with tarfile.open(fileobj=io.BytesIO(ask(["fetch", sid])), mode="r:gz") as tar:
            tar.extractall(out, filter="data")
        ledger = ask(["ledger"])
        (out / f"{sid}.ledger.jsonl").write_bytes(ledger)
        (out / f"{sid}.pins.jsonl").write_bytes(ask(["pins"]))
        write(withheld_on=Ch.withheld_on([x for x in ledger.decode().split("\n") if x], digest))
    except (Ch.CoinError, OSError, tarfile.TarError) as e:
        write(fetch_failure=str(e))
    write(end=end and end.get("body"))
    print("END " + json.dumps(doc["end"]), flush=True)
    return 0


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
    ssh = ["-i", a.gate_key, "-o", "StrictHostKeyChecking=accept-new", a.gate]
    return coin(ssh_ask(ssh), json.loads(Path(a.registration).read_text()), out, gate_name=a.gate, peer=a.peer,
                minutes=a.minutes, main=remote.main_lines(a.main))


if __name__ == "__main__":
    sys.exit(main())
