"""A served window's two-stage audit, its first stage, as the challenger runs it: each size class's boundary registration
(`verity_vllm.pipeline.boundary_rows`, committed inside the engine core while the window was served) on a two-stage
`service.Stream` of its own (`stream_of`), registered, then sampled once on the auditor's challenger's coins (`first`):
the replay blocks, which are the class's spans the replay side replays. `resume` gives the replay side the same stream at
the same point, for its interior registration and the second stage (`First.second`).

* **The Spec** (`spec_of`): the values ``b.ingress`` and ``b.egress`` (phase ``serve``) and the interior values (phase
  ``interior``: the check template's ports as the replay side commits them, `interior_values`); the selection
  `TwoStage(grain, replay, k)` over the class's replay block; the check, its templates' programs and their query
  (`Layout.templates`, `Layout.query`: the check template's instances, or, inlined, the templates its body calls) with
  its proof deadline; and the public items both registrations show.
* **The coins.** `first` takes a class's `ChallengerCoins` (a test's in-process challenger). The command opens one
  challenger session on the window's run (`a5_coins.start`: ``local``, ``loopback:HOST:PORT`` or ``gate:USER@HOST``),
  each class a rank of it, sampling on a thread of its own in its turn; the session's evidence goes into every class's
  ``stage1.json``.
* **The hand-off** (``stage1.json``, a class's: its ``commit.json``, the rule and the Spec's digest, the record, its
  registration's digest and receipt, the sample, the bytes it read and the coins' segment and evidence). A `Stream` holds
  code and doesn't persist, so `resume` builds the class's stream again from them, registers the record and samples it
  again on the bytes read (`Staged`, which then reads the second stage's coins from a live session), holding each to
  ``stage1.json``: the same Spec digest, registration digest and sample.

Until the class's request Definition lands, the replay block is `boundary_rows`' stand-in (one check template's
instances, or their bodies inlined, as the class's ``commit.json`` says), and so is every sample this draws. An inlined
check whose body calls several templates needs core's two-stage over a window of templates.

Usage::

    two_stage.py first --boundary DIR [DIR ...] --check TEMPLATE --replay rate:NUM/DEN --k K --interior FILE.json \\
        --challenger local --main FILE --out OUT [--advice REGISTRATION.json] [--proof-deadline S]

Each DIR a class's (``w<window>/p<P>-o<O>/``), every one of one window; FILE.json the interior values,
``[[name, schema], ...]``; REGISTRATION.json the window's advice registration, whose ``binding`` the records' windows
name. A ``local`` session's challenger reads its coins through ``sampled-proofs-verifier``, which must be built first
(``lake build sampled-proofs-verifier`` in ``verity/protocols/spot/verifier/lean``): building it on first use outlasts the
local gate's start.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity.protocols.spot.device import challenger as Ch, coins as CC
from verity.protocols.spot.verifier import service as S
from verity.protocols.spot.verifier.one_stage import draw as D, partition as P
from verity.circuits.defs import SpecializedDefinition
from verity.circuits.program import Program
from verity.commitments.windows import rows as CS
from verity_vllm.pipeline import boundary_rows as BR

FORMAT = "verity-vllm/two-stage/stage1/v0"
CHALLENGER = "two_stage.py on the auditor's challenger's coins"
STAGE1 = "stage1.json"
#: seconds on the challenger's clock from the sample to the proofs: the replay, the interior registration and the second
#: stage's proofs all come inside it
PROOF_DEADLINE_S = 86400.0


def interior_values(member: CS.Member) -> tuple[tuple[str, str], ...]:
    """The interior values of a replay side that commits the check template's units as one window member
    (`ml.windows.rows`): each port's name and schema, in the member's order."""
    return tuple((member.port(p), member.schema(p)) for p in CS.port_leaves(member))


@dataclass(frozen=True)
class Rule:
    """A class's two-stage rule: the first stage's law over its spans (``replay``, a rate law ``rate:NUM/DEN[:T]``), the
    proof blocks ``k`` the second stage samples of each check template in each sampled span, the interior values the replay
    side commits, and the proof deadline. A fixed count ``subset:Q`` is refused: the serve sizes a class's N, and Q/N falls as it pads N."""

    replay: str
    k: int
    interior: tuple[tuple[str, str], ...]
    proof_deadline: float = PROOF_DEADLINE_S

    def __post_init__(self) -> None:
        if D.parse_rate(self.replay) is None:
            raise ValueError("a two-stage rule's replay law is a rate law, rate:NUM/DEN or rate:NUM/DEN:T, "
                             f"not {self.replay!r}")

    def to_json(self) -> dict[str, Any]:
        return {"replay": self.replay, "k": self.k, "interior": [list(v) for v in self.interior],
                "proof_deadline": self.proof_deadline}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Rule:
        return cls(d["replay"], int(d["k"]), tuple((str(n), str(s)) for n, s in d["interior"]), float(d["proof_deadline"]))


def spec_of(layout: BR.Layout, rule: Rule) -> S.Spec:
    names = [*BR.PORTS, *(n for n, _ in rule.interior)]
    values = (tuple(S.Value(p, BR.SCHEMA, "serve") for p in BR.PORTS)
              + tuple(S.Value(n, s, "interior") for n, s in rule.interior))
    public = (S.Public("population", "structure", "the class's spans N; then the sampled spans' proof blocks Q n_rho"),
              S.Public("window", "structure", "provenance: the serve's run, the window, the class, the check, the index's leaf"),
              S.Public("leaf_layer", "structure", "the interior registration's public bodies' digest"),
              *(S.Public(f"{k}:{v}", "structure", f"{v}'s {k}, from the class and its spans") for v in names
                for k in ("leaves", "domain")))
    digests = tuple(P.program_digest(Program(t)) for t in layout.templates)
    check = S.Check(digests[0] if len(digests) == 1 else digests, layout.query, proof_deadline=rule.proof_deadline)
    return S.Spec(values, S.TwoStage(layout.grain, rule.replay, rule.k), check, public, f"{BR.FORMAT}\0{layout.run}".encode())


def stream_of(layout: BR.Layout, rule: Rule, coins: CC.ChallengerCoins, *, delta: float = 0.5, **kwargs: Any) -> S.Stream:
    """The class's two-stage stream on ``coins``: its units the check's templates (`Layout.templates`), its replay block
    the layout's."""
    ts = layout.templates
    return S.Stream(spec_of(layout, rule), ts[0] if len(ts) == 1 else ts, budget=1, delta=delta, challenger=CHALLENGER,
                    coins=coins, replay=layout.replay, **kwargs)


class Recorded(CC.ChallengerCoins):
    """``coins`` (a `ChallengerCoins`) with every byte read kept (`read`): what `resume` samples again."""

    def __init__(self, coins: CC.ChallengerCoins) -> None:
        self.coins, self.read = coins, bytearray()

    def commit(self, after: bytes) -> Any:
        return self.coins.commit(after)

    def __call__(self, n: int) -> bytes:
        b = self.coins(n)
        self.read.extend(b)
        return b


class Staged(CC.ChallengerCoins):
    """The first stage's bytes read again, in order; then, once committed (`commit`, to the interior registration's
    digest), ``then``'s: the second stage's live coins."""

    def __init__(self, read: bytes, then: CC.ChallengerCoins | None = None) -> None:
        self._read, self._at, self.then, self.live = bytes(read), 0, then, False

    def commit(self, after: bytes) -> Any:
        if self._at != len(self._read):
            raise Ch.CoinError(f"the second stage's coins follow the first stage's {len(self._read)} bytes, of which "
                               f"{self._at} were read again")
        if self.then is None:
            raise Ch.CoinError("no live coins for the second stage")
        self.live = True
        return self.then.commit(after)

    def __call__(self, n: int) -> bytes:
        if self.live:
            return self.then(n)
        if self._at + n > len(self._read):
            raise Ch.CoinError("the first stage read again reads past its bytes")
        self._at += n
        return self._read[self._at - n:self._at]


@dataclass
class First:
    """A class's first stage: its layout and rule, the stream, the record, its registration and the sample."""

    pub: dict[str, Any]
    rule: Rule
    layout: BR.Layout
    stream: S.Stream
    record: dict[str, Any]
    registration: S.WindowCommitmentRecord
    draw: S.Draw
    coins: CC.ChallengerCoins
    read: bytes = b""
    extra: dict[str, Any] = field(default_factory=dict)

    def stage1(self) -> dict[str, Any]:
        """``stage1.json``: what `resume` and the replay side read."""
        return {"format": FORMAT, "commit": self.pub, "rule": self.rule.to_json(), "spec": self.stream.spec.digest(),
                "record": self.record, "registration": self.registration.digest, "receipt": dict(self.registration.receipt),
                "draw": self.draw.draw, "units": list(self.draw.units), "read": self.read.hex(), **self.extra}

    def second(self, record: Mapping[str, Any], ports: Mapping[str, Mapping[str, Any]],
               **kwargs: Any) -> tuple[S.WindowCommitmentRecord, S.Draw]:
        """The replay side's interior registration over the sampled spans' proof blocks (`Stream.register_interiors`,
        taking its copies as it does), the coins committed to its digest, and the second stage's sample: ``k`` proof
        blocks in each sampled span."""
        ireg = self.stream.register_interiors(self.draw, record, ports, **kwargs)
        self.coins.commit(bytes.fromhex(ireg.digest))
        return ireg, self.stream.select(ireg)


def first(pub: Mapping[str, Any], check: SpecializedDefinition, rule: Rule, coins: CC.ChallengerCoins, *,
          previous: str | None = None, window: Mapping[str, Any] | None = None, delta: float = 0.5,
          window_root: str | None = None, **kwargs: Any) -> First:
    """One class's first stage on ``coins``: its stream registers the boundary record (`boundary_rows.record`, at the
    rule's first-stage law over its spans, chained after ``previous``, its window naming ``window``'s items), the coins
    commit to the registration's digest, and the stream samples once. Raises `service.Refused` as the stream does."""
    layout = BR.layout_of(pub, check)
    coins = coins if isinstance(coins, Recorded) else Recorded(coins)
    stream = stream_of(layout, rule, coins, delta=delta, **kwargs)
    rec = BR.record(pub, check, S.selection_law(stream.spec.selection, layout.spans), previous, window)
    reg = stream.register(rec, layout.ports(), **({} if window_root is None else {"window_root": window_root}))
    coins.commit(bytes.fromhex(reg.digest))
    draw = stream.select(reg)
    return First(dict(pub), rule, layout, stream, rec, reg, draw, coins, bytes(coins.read))


def resume(stage1: Mapping[str, Any], check: SpecializedDefinition, then: CC.ChallengerCoins | None = None, *,
           delta: float = 0.5, window_root: str | None = None, **kwargs: Any) -> First:
    """A class's stream again at the end of its first stage, from its ``stage1.json``: the record registered again and
    sampled again on the bytes the sample read, then on ``then`` once the interior registration's digest is committed
    (`Staged.commit`). ValueError when the Spec, the registration or the sample isn't ``stage1.json``'s."""
    if stage1.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} hand-off")
    pub, rule = stage1["commit"], Rule.from_json(stage1["rule"])
    layout = BR.layout_of(pub, check)
    coins = Staged(bytes.fromhex(stage1["read"]), then)
    stream = stream_of(layout, rule, coins, delta=delta, **kwargs)
    if stream.spec.digest() != stage1["spec"]:
        raise ValueError("the class's Spec isn't the one its first stage ran under")
    reg = stream.register(stage1["record"], layout.ports(), **({} if window_root is None else {"window_root": window_root}))
    if reg.digest != stage1["registration"]:
        raise ValueError("the record registers to another digest than its first stage's")
    draw = stream.select(reg)
    if list(draw.units) != list(stage1["units"]) or draw.draw != stage1["draw"]:
        raise ValueError("the first stage samples other spans read again than it did")
    return First(dict(pub), rule, layout, stream, dict(stage1["record"]), reg, draw, coins, bytes.fromhex(stage1["read"]))


def template_fn(t: str) -> SpecializedDefinition:
    return BR.check_template(t)


def _first_all(a: argparse.Namespace, pubs: list[dict], check: SpecializedDefinition, rule: Rule,
               window: Mapping[str, Any] | None) -> list[dict]:
    """Every class's first stage on one challenger session over the window's run, each class a rank on a thread of its own
    (`a5_coins.Draws`: the session opens once every rank has planned): each class's ``stage1.json`` contents, or why it
    sampled nothing."""
    import a5_coins as AC

    root = pubs[0]["run"]
    manager, draws, finisher = AC.start(a.challenger, root, list(range(len(pubs))), Path(a.out) / "challenger", a.main,
                                        ssh_key=a.ssh_key, peer=a.peer, key_record=a.challenger_key)
    out: list[dict] = [{} for _ in pubs]

    def run(k: int) -> None:
        coins = AC.RankCoins(draws, k)
        try:
            layout = BR.layout_of(pubs[k], check)
            coins.plan(AC.need(rule.replay, layout.program, layout.grain, layout.spans))
            got = first(pubs[k], check, rule, coins, window=window, delta=a.delta, window_root=root)
            out[k] = got.stage1()
        except (S.Refused, ValueError, KeyError, TypeError, OSError) as e:  # its stage1.json says why; the others go on
            out[k] = {"format": FORMAT, "commit": pubs[k], "refused": getattr(getattr(e, "verdict", None), "codes", None),
                      "error": f"{type(e).__name__}: {e}"}
        finally:
            out[k]["segment"] = coins.drawn()
            draws.skip(k)

    threads = [threading.Thread(target=run, args=(k,), name=f"class-{k}") for k in range(len(pubs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    finisher.join()
    published = draws.evidence()
    for o in out:
        o["evidence"] = AC.evidence(published, o["segment"]) if o.get("segment") else None
        o["session"] = {k: published.get(k) for k in ("plan", "trusted", "error") if k in published}
    manager.shutdown()
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("first", help="every class's boundary registration and first stage")
    f.add_argument("--boundary", nargs="+", required=True, help="each class's directory (w<window>/p<P>-o<O>/), one window's")
    f.add_argument("--check", required=True, help="the check template's descriptor id (serving_templates.template)")
    f.add_argument("--replay", required=True, help="the first stage's law over a class's spans, a rate law: rate:NUM/DEN or rate:NUM/DEN:T")
    f.add_argument("--k", type=int, required=True, help="proof blocks the second stage samples of each check template in each sampled span")
    f.add_argument("--interior", required=True, help="the interior values, a JSON list of [name, schema]")
    f.add_argument("--advice", help="the window's advice registration, whose binding the records' windows name")
    f.add_argument("--proof-deadline", type=float, default=PROOF_DEADLINE_S)
    f.add_argument("--delta", type=float, default=0.5)
    f.add_argument("--challenger", required=True, help="local, loopback:HOST:PORT or gate:USER@HOST (a5_coins.start)")
    f.add_argument("--main", required=True, help="the verifier's own `git rev-list --first-parent origin/main`, a file")
    f.add_argument("--peer")
    f.add_argument("--challenger-key")
    f.add_argument("--ssh-key")
    f.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pubs = [json.loads((Path(d) / BR.COMMIT).read_text()) for d in a.boundary]
    if len({(p["run"], p["window"]) for p in pubs}) != 1:
        ap.error("--boundary: every class of one window (one run and window index)")
    window = {"binding": json.loads(Path(a.advice).read_text())["binding"]} if a.advice else None
    check = template_fn(a.check)
    rule = Rule(a.replay, a.k, tuple((str(n), str(s)) for n, s in json.loads(Path(a.interior).read_text())),
                a.proof_deadline)
    out = Path(a.out)
    for d, s in zip(a.boundary, _first_all(a, pubs, check, rule, window), strict=True):
        dest = out / Path(d).name
        dest.mkdir(parents=True, exist_ok=True)
        (dest / STAGE1).write_text(json.dumps(s, indent=1) + "\n")
        print(json.dumps({"class": Path(d).name, "spans": s["commit"]["spans"], "units": s.get("units"),
                          "error": s.get("error")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
