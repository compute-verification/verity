"""A served window's live coins: where they come from, and the verifier's check of them from the challenger's own run.

* **Off the node** (`Session`, its id and HOST:PORT): the proof service's challenger on its auditor host
  (`sampled_proofs.remote.RemoteChallenger`), one session a window. A session comes from infra's gate (`request`:
  ``ssh GATE open`` with the window's period and Commit, the peer node and the window's layout), which opens at most
  one session per (period, Commit), ever, and returns the session's id and the port its challenger listens on at the
  peer's 127.0.0.1; lanes don't start challenger sessions themselves. A window requests its own once its Commit is
  fixed (serve.py ``--gate``); ``coins.py open`` requests one by hand for development, passed as ``--challenger
  HOST:PORT`` and ``--session ID`` (``$POUW_CHALLENGER``, ``$POUW_CHALLENGER_SESSION``). An address without its session
  id, or an id without its address, is refused. The session's layout is the window's live epoch (`layout`:
  `window.LIVE_SCHEME`, the window's epoch, `window.live_families(rounds)`), and the client refuses a commitment of
  any other.
  Each coin opens there, after the root the serving loop sends, and the challenger logs both before the coin leaves its
  host. The prover's record says only ``coins=off-node session ID at HOST:PORT (unchecked)``: what it writes about its
  coins is the prover's.
* **In-process** (no address): `window.live_challenger` in the serving process, on the prover's node. It stays for CPU
  tests, and every run that uses it carries `IN_PROCESS`, ``coins=in-process (not certified)``.

`check` is the verifier's side. It reads the session as infra's gate publishes it (``fetch SESSION``: request.json beside
the challenger's out/ with record.json, key.json and releases.jsonl), never the prover's bundle, through
`remote.trusted_session` under the challenger's key from its deployment record (``--key-record``, infra's
challenger-key/v1, never the gate ledger's key_sha512); no key is pinned in code. It holds the window's manifest to that
record: the gate's request and the signed record's host name the manifest's session, the request's layout is the signed
commitment's and its window the manifest's (session, period, Commit), the Commit being the registration root; the salt
coin after the registration root, round s's coin after round s's root for every round the challenger logged, and the
draw coin after the root the manifest names. The key record and the window's period are the verifier's to name; while
it names either not, the coins are ``off-node HOST (checked; key, period unanchored)`` (under key.json's own key), and
only with both are they ``off-node HOST (checked)``.

    python3 coins.py families --rounds S                        # the challenger's FAMILIES for a window of S rounds
    python3 coins.py open --gate USER@HOST --period P --commit HEX --peer NODE --epoch E --rounds S [--minutes M]
    python3 coins.py check --challenger-out SESSION_DIR --manifest OUT/manifest.json [--key-record KEY] [--period P]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verity.core.protocols.pouw import window as Wd  # noqa: E402
from verity.core.protocols.sampled_proofs import challenger as Ch  # noqa: E402
from verity.core.protocols.sampled_proofs import remote as R  # noqa: E402

CHALLENGER_ENV, SESSION_ENV = "POUW_CHALLENGER", "POUW_CHALLENGER_SESSION"
GATE_ENV, PERIOD_ENV, PEER_ENV = "POUW_CHALLENGER_GATE", "POUW_WINDOW_PERIOD", "POUW_PEER"
IN_PROCESS = "in-process (not certified)"


@dataclass(frozen=True)
class Session:
    """A challenger session as infra's gate returns it: its id and the HOST:PORT its challenger listens on here."""
    id: str
    address: str


def session(address: str | None = None, sid: str | None = None) -> Session | None:
    """The window's challenger session: `address` and `sid`, else ``$POUW_CHALLENGER`` and ``$POUW_CHALLENGER_SESSION``;
    None (neither) for the in-process challenger.  One without the other is refused."""
    at = address or os.environ.get(CHALLENGER_ENV) or None
    sid = sid or os.environ.get(SESSION_ENV) or None
    if (at is None) != (sid is None):
        raise Ch.CoinError(f"a challenger session is an id and an address from infra's gate, not {sid or at!r} "
                           f"alone (--challenger and --session, or ${CHALLENGER_ENV} and ${SESSION_ENV})")
    return None if at is None else Session(sid, at)


def layout(epoch: int, rounds: int) -> dict:
    """A live window's coin layout, the challenger session's SCHEME, EPOCH and FAMILIES."""
    return {"scheme": Wd.LIVE_SCHEME, "epoch": epoch, "families": [list(f) for f in Wd.live_families(rounds)]}


def ssh_args(key: str | None = None) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", *(["-o", "IdentitiesOnly=yes", "-i", key] if key else [])]


def request(gate: str, period: str, commit: bytes, peer: str, epoch: int, rounds: int, minutes: int | None = None,
            lane: str | None = None, ssh: Sequence[str] = ("ssh", "-o", "BatchMode=yes"),
            run: Callable[..., Any] = subprocess.run) -> tuple[Session, dict]:
    """A session for the window (`period`, `commit`, the window's served Commit root) from infra's gate at `gate`
    (``ssh GATE open``), its layout the window's (`layout` of `epoch` and `rounds`) and its challenger listening on
    `peer`'s 127.0.0.1:PORT, where the window's client runs.  The session and the gate's reply; a refusal (a second
    session for the window among them) raises."""
    lay = layout(epoch, rounds)
    args = ["open", "--period", period, "--commit", commit.hex(), "--peer", peer, "--scheme", lay["scheme"],
            "--families", json.dumps(lay["families"], separators=(",", ":")), "--epoch", str(epoch)]
    args += ["--minutes", str(minutes)] if minutes else []
    args += ["--lane", lane] if lane else []
    p = run([*ssh, gate, shlex.join(args)], capture_output=True, text=True)
    try:
        got = json.loads(p.stdout)
    except ValueError:
        got = None
    if p.returncode or not isinstance(got, dict) or "refused" in got or not {"session", "port"} <= got.keys():
        why = got.get("refused") if isinstance(got, dict) else None
        raise Ch.CoinError(f"infra's gate opened no session for {period}: {why or p.stderr.strip() or p.stdout.strip()}")
    return Session(got["session"], f"127.0.0.1:{got['port']}"), got


def wait_listening(at: Session, seconds: float = 900.0) -> None:
    """Until session `at`'s challenger listens here (``ss -ltn``, never a probe connect: it serves the first connection
    only, so the window connects the moment it appears)."""
    import time
    port, deadline = at.address.rpartition(":")[2], time.monotonic() + seconds
    while time.monotonic() < deadline:
        if subprocess.run(["ss", "-Hltn", f"sport = :{port}"], capture_output=True, text=True, check=False).stdout.strip():
            return
        time.sleep(0.05)
    raise Ch.CoinError(f"session {at.id}'s challenger never listened on {at.address}")


def layout_reason(commitment: Ch.LiveCommitment, epoch: int, rounds: int) -> str | None:
    """Why `commitment` is not the coin commitment of a live window of `rounds` rounds at `epoch`; None when it is."""
    want = layout(epoch, rounds)
    got = {"scheme": commitment.scheme, "epoch": commitment.epoch, "families": [list(f) for f in commitment.families]}
    if got != want:
        return f"the challenger committed to {json.dumps(got)}, not this window's {json.dumps(want)}"
    return None


class Coins:
    """A window's live coins from `challenger` (`LiveChallenger`'s commitment and open: a `RemoteChallenger`, or the
    in-process `LiveChallenger`), `source` "off-node" or "in-process": its salt and draw coins, and its rounds through
    `LiveRecord` (`rounds`)."""

    def __init__(self, challenger: Any, source: str, session: Session | None = None) -> None:
        self.challenger, self.source, self.session = challenger, source, session
        self.commitment: Ch.LiveCommitment = challenger.commitment
        self.rounds = Ch.LiveRecord(challenger, Wd.ROUND)
        self.label = (f"off-node session {session.id} at {session.address} (unchecked)" if session is not None
                      else IN_PROCESS)

    def salt(self, after: bytes) -> Ch.LiveOpening:
        return self.challenger.open(Wd.SALT, 0, after)

    def draw(self, after: bytes) -> Ch.LiveOpening:
        return self.challenger.open(Wd.DRAW, 0, after)

    def close(self) -> None:
        """End an off-node session: the challenger answers closed and writes its summary."""
        if self.source == "off-node":
            self.challenger.close()

    def record(self) -> dict:
        s = self.session
        return {"source": self.source, "label": self.label, "challenger": s and s.address, "session": s and s.id}


def open_coins(at: Session | None, epoch: int, rounds: int, entropy: Callable[[int], bytes] | None = None,
               connect: Callable[[str], Any] = R.RemoteChallenger.connect) -> Coins:
    """The window's coins: the challenger of session `at`, its commitment held to `layout`, or with no session the
    in-process challenger over `entropy` (the Lean verifier of record's coins by default)."""
    if at is None:
        if entropy is None:
            from verity.core.protocols.pouw import verifier
            entropy = verifier.coins
        return Coins(Wd.live_challenger(epoch, rounds, entropy), "in-process")
    client = connect(at.address)
    why = layout_reason(client.commitment, epoch, rounds)
    if why is not None:
        client.close()
        raise Ch.CoinError(why)
    return Coins(client, "off-node", at)


def lms_verify(key: Ch.ChallengerKey, message: bytes, signature: bytes) -> bool:
    from verity.ml.crypto import lms
    p = lms.Params(64, key.W, key.H)
    try:
        return lms.verify(p, key.public, message, lms.Signature.decode(p, signature))
    except ValueError:
        return False


def _openings(docs: list[dict]) -> list[Ch.LiveOpening]:
    return [Ch.LiveOpening.from_json(d) for d in docs]


def published(directory: str | os.PathLike) -> tuple[Path, Path]:
    """A session as infra's gate publishes it: `directory` holds request.json and the challenger's out/, or is that
    out/.  (out/, request.json)."""
    d = Path(directory)
    return (d / "out", d / "request.json") if (d / "out").is_dir() else (d, d.parent / "request.json")


def request_reason(req: dict, host: str, sid: str | None, commitment: Ch.LiveCommitment, window: dict | None,
                   registration: bytes, period: str | None) -> str | None:
    """Why the published session (the gate's request `req`, the signed record's `host`) is not session `sid`'s for this
    layout and the manifest's `window` (its period, and its Commit the registration root), or the window isn't
    `period` when the verifier names one; None when it is."""
    lay = {"scheme": commitment.scheme, "epoch": commitment.epoch, "families": [list(f) for f in commitment.families]}
    asked = {k: req.get(k) for k in lay}
    if req.get("session") != sid or not host.endswith(f".{sid}"):
        return f"the gate's request and signed record are session {req.get('session')!r} on {host}, not the window's {sid!r}"
    if asked != lay:
        return f"the gate's request asked for {json.dumps(asked)}, not the signed commitment's {json.dumps(lay)}"
    if window is None or window.get("commit") != registration.hex():
        return "the manifest names no window whose Commit is its registration root"
    if req.get("window") != window:
        return f"the session was requested for window {json.dumps(req.get('window'))}, not {json.dumps(window)}"
    if period is not None and window.get("period") != period:
        return f"the window is period {window.get('period')!r}, not {period!r}"
    return None


def key_record(path: str | os.PathLike) -> Ch.ChallengerKey:
    """The challenger's key from its deployment record (infra's challenger-key/v1: H, W, pubkey, pubkey_sha512), the key
    a verifier holds a session to, never the session's own key.json nor the gate ledger's key_sha512."""
    doc = json.loads(Path(path).read_text())
    if not str(doc.get("schema", "")).startswith("challenger-key/"):
        raise ValueError(f"{path} is not a challenger-key record")
    key = Ch.ChallengerKey.from_json({"H": doc["H"], "W": doc["W"], "public": doc["pubkey"]})
    if hashlib.sha512(key.public).hexdigest() != doc["pubkey_sha512"]:
        raise ValueError(f"{path}'s pubkey is not its pubkey_sha512's")
    return key


def check(directory: str | os.PathLike, coins: dict, registration: bytes, rounds: list[bytes],
          salts: dict[int, str] | None = None, draw_after: bytes | None = None, key: Ch.ChallengerKey | None = None,
          period: str | None = None) -> dict:
    """The verifier's check of a window's coins against the session `directory` as infra's gate publishes it
    (`published`), read from the challenger's own run.  `coins` is the manifest's ``coins`` item (the openings the
    prover received, the session id and the window's period and Commit), and the rest is what the verifier recomputes
    from the window's committed roots: `registration`, the round roots `rounds` in order, round r's unit salt `salts[r]`
    (hex) where it holds them, and `draw_after`, the root the draw coin follows (None: the window opened no draw); `key`
    is the challenger's key from its deployment record (`key_record`) and `period` the window's period, each None while
    the verifier names none.  It holds: key.json is `key`; the record verifies under it (`trusted_session`); the gate's
    request and the signed record are this session's, for the signed layout and the manifest's window, whose Commit is
    the registration root (`request_reason`); the window's commitment is the record's, a live epoch's; the salt coin
    opened after the registration root; the challenger received exactly `rounds`, and round r's coin opened once after
    round r's root (`rounds_reason` over the challenger's roots); each salt is `h1tile.live_salt` of the opened coin;
    the draw coin opened after `draw_after`, or none opened.  ``ok`` only when all hold."""
    from verity.core.protocols.pouw.circuit import h1tile as HT
    d, req_path = published(directory)
    out: dict[str, Any] = {"ok": False, "coins": "refused", "challenger_out": str(d), "session": coins.get("session")}
    try:
        theirs = Ch.ChallengerKey.from_json(json.loads((d / R.KEY).read_text()))
        host = Ch.ChallengerRecord.from_json(json.loads((d / R.RECORD).read_text())).host
        req = json.loads(req_path.read_text())
    except (OSError, ValueError) as e:
        return {**out, "reason": f"the session's request, key or record is unreadable: {e}"}
    out.update(host=host, key_sha512=hashlib.sha512(theirs.public).hexdigest(), window=req.get("window"))
    if key is not None and theirs != key:
        return {**out, "reason": f"the session's key is sha512 {out['key_sha512'][:16]}..., not the deployment record's "
                                 f"{hashlib.sha512(key.public).hexdigest()[:16]}..."}
    loose = [name for name, anchor in (("key", key), ("period", period)) if anchor is None]
    key = key or theirs
    try:
        commitment, roots = R.trusted_session(d, key, lms_verify)
        mine = Ch.LiveCommitment.from_json(coins.get("commitment"))
        salt = _openings([coins["salt_coin"]])
        openings = _openings([r["opening"] for r in coins.get("rounds", [])])
        draw = _openings([coins["draw_coin"]]) if coins.get("draw_coin") else []
    except (KeyError, TypeError, Ch.CoinError) as e:
        return {**out, "reason": f"{type(e).__name__}: {e}"}
    got = roots[Wd.ROUND]
    bad_salts = sorted(r for r, s in (salts or {}).items()
                       if not (0 <= r < min(len(got), len(openings)))
                       or HT.live_salt(commitment, openings[r].coin, r, got[r]).hex() != s)
    reasons = {
        "request": request_reason(req, host, coins.get("session"), commitment, coins.get("window"), registration, period),
        "commitment": None if mine == commitment else "the window's coin commitment is not the one the challenger signed",
        "epoch": Wd.live_epoch_reason(commitment),
        "salt": (None if roots[Wd.SALT] == (registration,) else "the challenger's salt root is not the registration root")
        or Ch.order_reason(commitment, Wd.SALT, roots[Wd.SALT], salt),
        "rounds": (None if got == tuple(rounds) else
                   f"the challenger received {len(got)} round roots, not the window's {len(rounds)} committed ones")
        or Wd.rounds_reason(commitment, got, openings),
        "salts": None if not bad_salts else f"rounds {bad_salts[:8]}'s unit salts are not their opened coins' live_salt",
        "draw": (None if roots[Wd.DRAW] == (() if draw_after is None else (draw_after,)) else
                 "the challenger's draw roots are not the window's")
        or (Ch.order_reason(commitment, Wd.DRAW, roots[Wd.DRAW], draw) if draw_after is not None else None),
    }
    out.update(reasons=reasons, rounds=len(got), draws=len(roots[Wd.DRAW]), commitment=commitment.root.hex())
    out["ok"] = all(v is None for v in reasons.values())
    out["coins"] = (f"off-node {host} (checked{'; ' + ', '.join(loose) + ' unanchored' if loose else ''})"
                    if out["ok"] else "refused")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("families", help="the challenger session's FAMILIES for a window of --rounds rounds")
    f.add_argument("--rounds", type=int, required=True)
    o = sub.add_parser("open", help="request the window's session from infra's gate; prints its reply and the client's env")
    o.add_argument("--gate", default=os.environ.get(GATE_ENV), help=f"infra's gate, USER@HOST (default ${GATE_ENV})")
    o.add_argument("--ssh-key", help="the ssh key the gate takes (default: ssh's own)")
    for name in ("--period", "--commit", "--peer"):
        o.add_argument(name, required=True)
    o.add_argument("--epoch", type=int, required=True)
    o.add_argument("--rounds", type=int, required=True)
    o.add_argument("--minutes", type=int)
    o.add_argument("--lane")
    c = sub.add_parser("check", help="a window's coins against the challenger's own run")
    c.add_argument("--challenger-out", required=True, help="the session as infra's gate publishes it (request.json "
                   "and out/), or its out/")
    c.add_argument("--manifest", required=True, help="the window's manifest.json")
    c.add_argument("--key-record", help="the challenger's key record from its deployment (challenger-key/v1, an art: "
                   "fetched with `research data fetch`)")
    c.add_argument("--period", help="the window's period, as the verifier knows it")
    a = ap.parse_args(argv)
    if a.cmd == "families":
        print(json.dumps(layout(0, a.rounds)["families"], separators=(",", ":")))
        return 0
    if a.cmd == "open":
        if not a.gate:
            ap.error(f"open needs --gate or ${GATE_ENV}")
        at, got = request(a.gate, a.period, bytes.fromhex(a.commit), a.peer, a.epoch, a.rounds, a.minutes, a.lane,
                          ssh_args(a.ssh_key))
        print(json.dumps(got))
        print(f"{CHALLENGER_ENV}={at.address} {SESSION_ENV}={at.id}")
        return 0
    import importlib.util
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    spec = importlib.util.spec_from_file_location("served_h1_serve", here / "serve.py")
    SV = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(SV)
    m = json.loads(Path(a.manifest).read_text())
    key = key_record(a.key_record) if a.key_record else None
    got = check(a.challenger_out, m.get("coins", {}), bytes.fromhex(m["roots"]["registration"]), **SV.coins_expect(m),
                key=key, period=a.period)
    print(json.dumps(got, indent=1))
    return 0 if got["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
