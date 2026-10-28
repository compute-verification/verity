"""Each drawing round's coins from the auditor's challenger: a round of the network chain that draws (a scan round's
`service.Stream`, a sampled N2 period's draw) reads its coins from a challenger session of its own and from nothing else,
never the verifier's own randomness.

* **The challenger** (``--challenger``): ``local``, a development session on this machine (`a5_coins.Local`: the auditor
  host's own service on a scratch root, under an LMS key of its own, so it shows the path, not an auditor's coins: the
  dry runs'); ``loopback:HOST:PORT``, the same on the challenger party's host; ``gate:USER@HOST``, the auditor's
  challenger through infra's gate, on the network certifier's machine, with ``--peer`` (this node as the gate's /etc/vy/peers names
  it, whose 127.0.0.1 the gate's coin link serves), ``--challenger-key`` (the deployment's challenger-key/v1 record) and
  ``--gate-key`` (the lane's gate key). ``--main`` is the verifier's own copy of main's first-parent history, on which
  the session's pin must be. Tests run ``local`` on live_challenger's stub of the Lean's coins
  (``benchmarks/live_challenger/tests/stub_serve.py``): the fake challenger.
* **The root** (`root`): the SHA-512 of the run's base (what the auditor's devices signed, which the prover can't
  vary: the scan's the warden's session and signed registrations, a sampled N2 period's its certificates), the step
  and the round. The gate opens a root once, ever, so each round of a run draws once, and a rerun of the run can't
  draw it again.
* **A round's session** (`Session`: `a5_coins.start` for one rank). The round plans its need (`RankCoins.plan`, which
  opens the session), commits to its registration (`RankCoins.commit`), reads its coins and ends its turn
  (`RankCoins.drawn`: the segment the session released). `close` publishes the session's evidence, kept under its
  directory, and `held` holds what the round read to it: `coins.coins_reason` on the root and the round's commitment,
  each of `a5_coins.negatives` refused, and the bytes read the ones released. A session that can't open, or whose
  evidence doesn't hold, fails the round: nothing redraws and nothing falls back to other coins.
* **Not the challenger's** (`FLOCK`): each Flock session's coins, its coin tree, which ``flock-circuit serve --zk``
  draws itself at Hello and opens round by round.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from verity.core.protocols.sampled_proofs import coins as CC, service as S

HERE = Path(__file__).resolve().parent
ROOT = "verity/certifier/round-coins/v1"
KINDS = "local, loopback:HOST:PORT or gate:USER@HOST"
#: whose coins every Flock session (scan, N2, tie) runs on: not the challenger's
FLOCK = {"source": "flock-circuit serve --zk", "challenger": False,
         "by": "the session's coin tree, drawn by the Flock server itself (the OS) at Hello and opened round by round, "
               "on the node the chain runs on: not the auditor's challenger's"}


def a5_coins():
    if str(HERE.parent / "one_stage") not in sys.path:
        sys.path.append(str(HERE.parent / "one_stage"))
    import a5_coins as AC
    return AC


def digest(doc) -> str:
    """The SHA-512 of `doc` in canonical JSON (128 hex digits)."""
    return hashlib.sha512(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def root(base: str, step: str, name: str) -> str:
    """Round `name` of `step`'s session root (128 hex digits): the SHA-512 of the run's `base`, the step and the round."""
    return digest({"format": ROOT, "base": base, "step": step, "round": name})


def add_args(p, what: str) -> None:
    """The challenger's flags on parser `p`; `what` names the draws they take."""
    p.add_argument("--challenger", help=f"{what} on the auditor's challenger's coins, a session a round: {KINDS} "
                   "(the auditor's challenger through its gate, with --peer and --challenger-key)")
    p.add_argument("--peer", help="with --challenger gate:USER@HOST: this node as the gate's /etc/vy/peers names it, whose "
                   "127.0.0.1 the session's coin link serves")
    p.add_argument("--challenger-key", help="with --challenger gate:USER@HOST: the challenger's key record of its "
                   "deployment (challenger-key/v1), never the session's key.json")
    p.add_argument("--gate-key", help="with --challenger gate:USER@HOST: the lane's own gate key (default ssh's own)")
    p.add_argument("--main", help="with --challenger: the verifier's own copy of main's first-parent history (git "
                   "rev-list --first-parent origin/main), which must hold the commit the session's host pins")


def refusal(a) -> str | None:
    """Why the flags can't draw on a challenger, or None when they can."""
    c = a.challenger
    if not c:
        return f"--challenger names the auditor's challenger every drawing round reads its coins from ({KINDS}): " \
               "nothing draws on the verifier's own coins"
    if c != "local" and not c.startswith(("gate:", "loopback:")):
        return f"--challenger is {KINDS}, not {c!r}"
    if c.startswith("gate:") and not (a.peer and a.challenger_key):
        return "--challenger gate:USER@HOST needs --peer and --challenger-key"
    if not a.main:
        return "--challenger needs --main: the verifier's git rev-list --first-parent origin/main, which must hold the " \
               "commit the session's host pins"
    return None


def subset_need(k: int, n: int) -> int:
    """The most coins a draw of `subset:K` of n reads (`a5_coins.need`'s rule): the stream's sampler on bytes that reject
    no word, what it reads plus its longest stream once more."""
    reads: list[int] = []

    def zeros(m: int) -> bytes:
        reads.append(m)
        return bytes(m)
    rule = S.Law(f"subset:{k}")
    S.select_draw(rule, S.selection_law(rule, n), n, zeros)
    return -(-(sum(reads) + max(reads, default=0)) // a5_coins().Ch.COIN_BYTES)


class Session:
    """Round `name`'s challenger session on `root`, its evidence kept under `dest` (module docstring). `rank` is the
    round's coins (`a5_coins.RankCoins`), which its draw plans, commits and reads."""

    def __init__(self, a, root: str, dest: Path) -> None:
        self.AC, self.root, self.challenger, self.dest = a5_coins(), root, a.challenger, dest
        self.manager, self.draws, self.finisher = self.AC.start(
            a.challenger, root, [0], dest, a.main, ssh_key=a.gate_key, peer=a.peer, key_record=a.challenger_key)
        self.rank = self.AC.RankCoins(self.draws, 0)
        self.published: dict | None = None
        self.pre: dict | None = None

    def close(self) -> dict:
        """The round has drawn or won't: the session's evidence once it has ended (`a5_coins.Draws.evidence`)."""
        if self.published is None:
            self.draws.skip(0)
            self.finisher.join()
            self.published, self.pre = self.draws.evidence(), self.draws.pre()
            self.manager.shutdown()
        return self.published

    def evidence(self, segment: dict | None) -> dict | None:
        """The round's draw's evidence (`coins.coins_reason`'s), None when the session gave none."""
        published = self.close()
        return None if segment is None else self.AC.evidence(published, segment)

    def held(self, after: bytes, read: bytes, segment: dict | None) -> dict:
        """The round's coins held to the session's evidence: the bytes `read` released after the round's commitment
        `after` in this session (`coins.coins_reason` on `root`), the bytes it read its rank's, and each of
        `a5_coins.negatives` refused. The summary's record of the round's session."""
        published = self.close()
        ev = self.evidence(segment)

        def reason(e) -> str | None:
            return CC.coins_reason(e, self.root, after, read, self.AC.lms_verify)
        why = (published.get("error") or "the draw released no segment") if ev is None else reason(ev)
        negatives = ({} if ev is None or self.pre is None else
                     {k: reason(v) is not None for k, v in self.AC.negatives(ev, self.pre).items()})
        same = read == bytes(self.rank.read)
        trusted = published.get("trusted") or {}
        ok = (why is None and same and bool(negatives) and all(negatives.values())
              and trusted.get("pinned_session") is None)
        if ev is not None:
            self.dest.mkdir(parents=True, exist_ok=True)
            (self.dest / "coins.json").write_text(json.dumps(ev, sort_keys=True))
        return {"source": f"challenger:{self.challenger}", "root": self.root,
                "session": (published.get("plan") or {}).get("session"), "trusted": trusted or None,
                "plan": published.get("plan"), "error": published.get("error"), "coins_reason": why,
                "read_bytes": len(read), "read_is_released": same, "negatives_refused": negatives, "ok": ok}
