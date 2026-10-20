"""A served window's live coins: where they come from, and the verifier's check of them from the challenger's own run.

* **Off the node** (`Session`: ``--challenger HOST:PORT`` and ``--session ID``, else ``$POUW_CHALLENGER`` and
  ``$POUW_CHALLENGER_SESSION``): the proof service's challenger on its auditor host (`sampled_proofs.remote.
  RemoteChallenger`), one session a window. A session comes from infra's request call, which takes the window's Commit
  and period and returns the session's id and the port its tunnel opens on the prover's node; lanes don't start
  challenger sessions themselves. An address without its session id, or an id without its address, is refused. The
  session's layout is the window's live epoch (`layout`: `window.LIVE_SCHEME`, the window's epoch,
  `window.live_families(rounds)`, the FAMILIES the request names), and the client refuses a commitment of any other.
  Each coin opens there, after the root the serving loop sends, and the challenger logs both before the coin leaves its
  host. The prover's record says only ``coins=off-node session ID at HOST:PORT (unchecked)``: what it writes about its
  coins is the prover's.
* **In-process** (no address): `window.live_challenger` in the serving process, on the prover's node. It stays for CPU
  tests, and every run that uses it carries `IN_PROCESS`, ``coins=in-process (not certified)``.

`check` is the verifier's side. It reads the session's directory as the challenger published it (record.json, key.json,
releases.jsonl), never the prover's bundle, through `remote.trusted_session` under the key that directory's key.json
holds; no key is pinned in code. It holds the window's manifest to that record: the commitment the challenger signed, the salt coin after the registration root, round s's coin
after round s's root for every round the challenger logged, and the draw coin after the root the manifest names. Only
then are the coins ``off-node (checked)``.

    python3 coins.py families --rounds S                        # the challenger's FAMILIES for a window of S rounds
    python3 coins.py check --challenger-out SESSION_DIR --manifest OUT/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verity.core.protocols.pouw import window as Wd  # noqa: E402
from verity.core.protocols.sampled_proofs import challenger as Ch  # noqa: E402
from verity.core.protocols.sampled_proofs import remote as R  # noqa: E402

CHALLENGER_ENV, SESSION_ENV = "POUW_CHALLENGER", "POUW_CHALLENGER_SESSION"
IN_PROCESS = "in-process (not certified)"


@dataclass(frozen=True)
class Session:
    """A challenger session as infra's request call returns it: its id and the HOST:PORT its tunnel opens here."""
    id: str
    address: str


def session(address: str | None = None, sid: str | None = None) -> Session | None:
    """The window's challenger session: `address` and `sid`, else ``$POUW_CHALLENGER`` and ``$POUW_CHALLENGER_SESSION``;
    None (neither) for the in-process challenger.  One without the other is refused."""
    at = address or os.environ.get(CHALLENGER_ENV) or None
    sid = sid or os.environ.get(SESSION_ENV) or None
    if (at is None) != (sid is None):
        raise Ch.CoinError(f"a challenger session is an id and an address from infra's request call, not {sid or at!r} "
                           f"alone (--challenger and --session, or ${CHALLENGER_ENV} and ${SESSION_ENV})")
    return None if at is None else Session(sid, at)


def layout(epoch: int, rounds: int) -> dict:
    """A live window's coin layout, the challenger session's SCHEME, EPOCH and FAMILIES."""
    return {"scheme": Wd.LIVE_SCHEME, "epoch": epoch, "families": [list(f) for f in Wd.live_families(rounds)]}


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


def check(directory: str | os.PathLike, coins: dict, registration: bytes, rounds: list[bytes],
          salts: dict[int, str] | None = None, draw_after: bytes | None = None) -> dict:
    """The verifier's check of a window's coins against the challenger's directory `directory`, read from the
    challenger's own run.  `coins` is the manifest's ``coins`` item (the openings the prover received), and the rest is
    what the verifier recomputes from the window's committed roots: `registration`, the round roots `rounds` in order,
    round r's unit salt `salts[r]` (hex) where it holds them, and `draw_after`, the root the draw coin follows (None: the
    window opened no draw).  It holds: the record verifies under the key the directory's key.json holds
    (`trusted_session`); the window's commitment is the
    record's, a live epoch's; the salt coin opened after the registration root; the challenger received exactly `rounds`,
    and round r's coin opened once after round r's root (`rounds_reason` over the challenger's roots); each salt is
    `h1tile.live_salt` of the opened coin; the draw coin opened after `draw_after`, or none opened.  ``ok`` only when
    all hold."""
    from verity.core.protocols.pouw.circuit import h1tile as HT
    d = Path(directory)
    out: dict[str, Any] = {"ok": False, "coins": "refused", "challenger_out": str(d), "session": coins.get("session")}
    try:
        key = Ch.ChallengerKey.from_json(json.loads((d / R.KEY).read_text()))
        host = Ch.ChallengerRecord.from_json(json.loads((d / R.RECORD).read_text())).host
    except (OSError, ValueError) as e:
        return {**out, "reason": f"the challenger's key or record is unreadable: {e}"}
    out.update(host=host, key_sha512=hashlib.sha512(key.public).hexdigest())
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
    out["coins"] = f"off-node {host} session {out['session']} (checked)" if out["ok"] else "refused"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("families", help="the challenger session's FAMILIES for a window of --rounds rounds")
    f.add_argument("--rounds", type=int, required=True)
    c = sub.add_parser("check", help="a window's coins against the challenger's own run")
    c.add_argument("--challenger-out", required=True, help="the session's directory as the challenger published it")
    c.add_argument("--manifest", required=True, help="the window's manifest.json")
    a = ap.parse_args(argv)
    if a.cmd == "families":
        print(json.dumps(layout(0, a.rounds)["families"], separators=(",", ":")))
        return 0
    import importlib.util
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    spec = importlib.util.spec_from_file_location("served_h1_serve", here / "serve.py")
    SV = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(SV)
    m = json.loads(Path(a.manifest).read_text())
    got = check(a.challenger_out, m.get("coins", {}), bytes.fromhex(m["roots"]["registration"]), **SV.coins_expect(m))
    print(json.dumps(got, indent=1))
    return 0 if got["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
