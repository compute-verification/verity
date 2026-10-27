"""The onsite composite (`verity/onsite/v0`: the network certifier, PoUS, PoUW, the bandwidth audit and sampled proofs) on
the lifecycle, as a deployment of four parties: the developer and three of the auditor's, the verifier, the network
certifier on the site's uplink and the memory challenger.

- The network certifier is the outside world's only way in: it carries each request to the developer and its response
  back, commits to every frame it sees under its own salts, and signs each segment's view (`lms.certificate_message`, one
  signature a direction). The developer gets the frames' openings, to prove its I/O gates against; the verifier gets the
  signed commitments and the count.
- The memory challenger opens a round once a segment's values are committed: on its own coins it picks theta rows, asks
  the developer for their openings, and times the answer on its own clock. PoUS holds the openings to theta's registered
  commitment; the bandwidth audit holds the time to the link's latency and the data sheet's rate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lifecycle import (CERTIFIER, CHALLENGER, DEVELOPER, SHARED, SKUS, THETA_ROWS, VERIFIER, Developer, Device, Hardware,
                       Tape, Verifier, answer, body, digest, frame, frame_commitment, load_of, msg, opens, products,
                       schema, theta_of)
from verity.core.deploy import runtime
from verity.core.deploy.manifest import Channel, Manifest, Role
from verity.core.deploy.party import Finish, Identity, Received, Send
from verity.core.protocols.sampled_proofs.parties import IDS
from verity.core.wire import Message
from verity.ml.circuits import codec
from verity.ml.crypto import lms

ID = "lifecycle-onsite"
SIZES = (3, 4, 2)
"""Requests in each segment of the example's period."""
LINK, CERT = "uplink0", "cert0"
HARDWARE = Hardware((Device("gpu0", "sim-int8-a", LINK), Device("gpu1", "sim-int8-b", LINK)), ((LINK, CERT),))
LINK_MS = 1
"""The latency between the memory challenger and the developer's node, each way."""
LMS = lms.Params(64, 4, 4)
"""The certifier's signing key: 16 one-time keys, two a segment."""
ROWS = 2
"""Theta rows the memory challenger asks for a round."""

ONSITE = (schema("Go", "structure"), schema("Ingress", "private"), schema("Egress", "private"), schema("Mark", "private"),
          schema("View", "commitment"), schema("Open", "structure"), schema("Challenge", "coins"),
          schema("Answer", "private"), schema("Report", "output"))


class Certifier:
    """The network certifier on the uplink. Capabilities: `load` (each segment's requests: the outside world) and `seed`
    (its signing key's)."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        c = identity.capabilities
        self.load, self.key = [list(s) for s in c["load"]], lms.keygen(LMS, c["seed"])
        self.k, self.i, self.frames, self.openings = 0, 0, [], {}
        return [Send("coins", Message("CoinsDraw", {"n": 64}))]

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if r.channel == "coins":
            self.tape = Tape(m.body)
            self.session = self.tape.bytes("session", lms.SESSION_BYTES)
            return []
        if m.type in ("Done", "Closed", "Refused"):
            return [Finish({"segments": self.k})]
        if m.type == "Go":
            return self._next()
        if m.type == "Egress":
            x, y = self.load[self.k - 1][self.i], body(m)["y"]
            sin, sout = self.tape.bytes(f"{self.k}/{self.i}/in", 64), self.tape.bytes(f"{self.k}/{self.i}/out", 64)
            self.frames.append({"in": frame_commitment(frame(x), sin), "out": frame_commitment(frame(y), sout)})
            self.openings[str(self.i)] = {"in": frame(x).hex(), "in_salt": sin.hex(), "out": frame(y).hex(),
                                          "out_salt": sout.hex()}
            self.i += 1
            return self._next()
        return []

    def _next(self) -> list[Any]:
        if self.k and self.i < len(self.load[self.k - 1]):
            return [Send(DEVELOPER, msg("Ingress", self.k, {"request": self.i, "x": self.load[self.k - 1][self.i]}))]
        out = []
        if self.k:
            view = {"segment": self.k, "link": LINK, "session": self.session.hex(), "count": len(self.frames),
                    "frames": self.frames}
            sigs = {d: lms.sign_at(self.key, 2 * (self.k - 1) + j, self._message(view, d)).encode().hex()
                    for j, d in enumerate(lms.DIRECTIONS)}
            out = [Send(DEVELOPER, msg("Mark", self.k, {"frames": self.openings})),
                   Send(VERIFIER, msg("View", self.k, {**view, "signatures": sigs}))]
        if self.k == len(self.load):
            return out
        self.k, self.i, self.frames, self.openings = self.k + 1, 0, [], {}
        return out + self._next()

    def _message(self, view: Mapping[str, Any], direction: str) -> bytes:
        return certified(self.key.I, view, direction)


def certified(I: bytes, view: Mapping[str, Any], direction: str) -> bytes:
    """What the certifier signs for a direction of a segment's view: the frames' commitments that way, and the count."""
    way = "in" if direction == "ingress" else "out"
    record = codec.canonical_json({"count": view["count"], "frames": [f[way] for f in view["frames"]]})
    return lms.certificate_message(I, bytes.fromhex(view["session"]), view["link"], direction, view["segment"], record)


class Challenger:
    """The memory challenger: one round a committed segment, on its own coins and clock."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        self.queue: list[dict[str, Any]] = []
        self.round: dict[str, Any] | None = None
        self.rounds = 0
        return []

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if m.type in ("Done", "Closed", "Refused"):
            return [Finish({"rounds": self.rounds})]
        if m.type == "Open":
            self.queue.append({"k": m.fields["k"], **body(m)})
            return self._begin() if self.round is None else []
        if r.channel == "coins":
            picks = list(range(THETA_ROWS))
            rows = []
            for b in m.body:
                if len(rows) == ROWS:
                    break
                rows.append(picks.pop(b % len(picks)))
            self.round["rows"] = rows
            return [Send("clock", Message("ClockRead"))]
        if r.channel == "clock":
            self.round["sent"] = m.fields["ns"]
            return [Send(DEVELOPER, msg("Challenge", self.round["k"], {"rows": self.round["rows"]}))]
        if m.type == "Answer":
            return self._report(r.at, body(m))
        return []

    def _begin(self) -> list[Any]:
        self.round = self.queue.pop(0)
        return [Send("coins", Message("CoinsDraw", {"n": 16}))]

    def _report(self, at: int, a: Mapping[str, Any]) -> list[Any]:
        rd, self.round = self.round, None
        self.rounds += 1
        got = a.get("rows", [])
        ok = [o.get("position") for o in got] == rd["rows"] and all(opens(rd["theta"], o) for o in got)
        elapsed = at - rd["sent"]
        bound = 2 * rd["latency_ns"] + -(-ROWS * 4 // rd["bytes_per_ns"])
        report = {"rows": rd["rows"], "pous": None if ok else "an answer doesn't open theta's registered commitment",
                  "elapsed_ns": elapsed, "bound_ns": bound,
                  "bandwidth": None if elapsed <= bound else f"answered in {elapsed} ns, over the link's {bound} ns"}
        return [Send(VERIFIER, msg("Report", rd["k"], report)), *(self._begin() if self.queue else ())]


class OnsiteVerifier(Verifier):
    """The verifier, with the onsite components: the certifier's signed view of each segment, and the memory
    challenger's report."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        self.keys = {c: bytes.fromhex(pk) for c, pk in params["certifiers"].items()}
        self.views: dict[int, dict[str, Any]] = {}
        self.reports: dict[int, dict[str, Any]] = {}
        return super().start(params, identity)

    def registered(self) -> list[Any]:
        return [Send(CERTIFIER, msg("Go", 0))]

    def closed(self, k: int) -> list[Any]:
        rates = [SKUS[d.sku]["mem_bytes_per_ns"] for d in self.registration.hardware.devices]
        return [Send(CHALLENGER, msg("Open", k, {"theta": self.registration.theta, "latency_ns": LINK_MS * 1_000_000,
                                                 "bytes_per_ns": min(rates)}))]

    def component(self, m: Message, k: int, j: Any) -> None:
        if m.type == "View":
            cert = dict(self.registration.hardware.certifiers).get(j.get("link"))
            pk = self.keys.get(cert)
            signed = pk is not None and all(
                lms.verify(LMS, pk, certified(pk[:16], j, d), lms.Signature.decode(LMS, bytes.fromhex(j["signatures"][d])))
                for d in lms.DIRECTIONS)
            self.views[k] = {"view": j, "signed": signed}
        elif m.type == "Report":
            self.reports[k] = j
        else:
            super().component(m, k, j)

    def ready(self, k: int) -> bool:
        return super().ready(k) and k in self.views and k in self.reports

    def frames(self, k: int, u: int) -> dict[str, str] | None:
        fs = self.views[k]["view"]["frames"]
        return fs[u] if u < len(fs) else {"in": "", "out": ""}

    def answers(self, k: int, checks: Mapping[int, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        v, rp, seg = self.views[k], self.reports[k], self.circuit[k - 1]
        view = v["view"]
        bad = ("the view's signatures don't verify under the registered certifier's key" if not v["signed"] else
               f"the certifier saw {view['count']} requests, the circuit has {seg.real} real units"
               if view["count"] != seg.real else
               "; ".join(f"unit {u}: {c['io']}" for u, c in sorted(checks.items()) if c["io"]) or None)
        vd, rd = digest("lifecycle/certifier-view/v1", view), digest("lifecycle/challenger-report/v1", rp)
        return {IDS["certifier"]: answer(IDS["certifier"], "reject" if bad else "accept", vd, bad,
                                         {"count": view["count"]}),
                IDS["pous"]: answer(IDS["pous"], "reject" if rp["pous"] else "accept", rd, rp["pous"],
                                    {"rows": rp["rows"]}),
                IDS["bandwidth"]: answer(IDS["bandwidth"], "reject" if rp["bandwidth"] else "accept", rd, rp["bandwidth"],
                                         {"elapsed_ns": rp["elapsed_ns"], "bound_ns": rp["bound_ns"]})}


MANIFEST = Manifest(
    ID, 1, "examples/lifecycle/onsite.py:MANIFEST",
    parties=(Role(DEVELOPER, "developer", needs=("coins", "clock")), Role(VERIFIER, "auditor", needs=("coins",)),
             Role(CERTIFIER, "auditor", per="link", needs=("coins",)),
             Role(CHALLENGER, "auditor", needs=("coins", "clock"))),
    channels=(Channel(DEVELOPER, VERIFIER, ("Register", "Close", "Proofs")),
              Channel(VERIFIER, DEVELOPER, ("Draw", "Done"), carries_coins=True),
              Channel(CERTIFIER, DEVELOPER, ("Ingress", "Mark")),
              Channel(DEVELOPER, CERTIFIER, ("Egress",)),
              Channel(CERTIFIER, VERIFIER, ("View",)),
              Channel(VERIFIER, CERTIFIER, ("Go", "Done")),
              Channel(VERIFIER, CHALLENGER, ("Open", "Done")),
              Channel(CHALLENGER, VERIFIER, ("Report",)),
              Channel(CHALLENGER, DEVELOPER, ("Challenge",), LINK_MS, carries_coins=True),
              Channel(DEVELOPER, CHALLENGER, ("Answer",), LINK_MS)),
    messages=SHARED + ONSITE)
PROTOCOL = runtime.ProtocolDef(MANIFEST, {DEVELOPER: Developer, VERIFIER: OnsiteVerifier, CERTIFIER: Certifier,
                                          CHALLENGER: Challenger})


def run(cheat: str | None = None, seed: bytes = b"lifecycle/onsite") -> dict[str, Any]:
    """One monitoring period of the onsite composite, simulated: the verifier's outcome, and each party's."""
    cert_seed = Tape(seed).bytes("certifier-key", LMS.N)
    pk = lms.public_key(lms.keygen(LMS, cert_seed))
    config = {"protocol": ID, "params": {"protocol": "onsite", "segments": len(SIZES), "certifiers": {CERT: pk.hex()}},
              "party": [{"name": p} for p in (DEVELOPER, VERIFIER, CERTIFIER, CHALLENGER)]}
    caps = {DEVELOPER: {"theta": theta_of(1), "state0": 0, "hardware": HARDWARE, "product": products()["onsite"][0],
                        "cheat": cheat},
            CERTIFIER: {"load": load_of(2, SIZES), "seed": cert_seed}}
    result = runtime.run(config, [DEVELOPER, VERIFIER, CERTIFIER, CHALLENGER], "simulated", None,
                         protocols={ID: PROTOCOL}, seed=seed, authenticate=False, capabilities=caps)
    return {"status": result.status, "outcomes": result.outcomes, **(result.outcomes[VERIFIER]["outcome"] or {})}
