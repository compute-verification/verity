"""The proof service's first calls as the joint note's §2 gives them (proofs' P1), over one-stage, for served_zk until P1
lands in `sampled_proofs`: `served_commit.py` imports this module as `S`, and switching to P1 is that import.

* **`Spec`**: the protocol's committed values, its selection (a one-stage law), its check (the per-unit Program and
  query, and the commitment scheme), its public items and its statement context.
* **`register(spec, items, roots)`**: one-stage's `verity/registration/v1` record of the spec's check and law, its window
  `{"kind": "served", "context", "public"}` holding the registered public items.  The public-items rule (`public_problem`):
  every field is one-stage's structure derived from the spec, a declared public item, or a salted root of a declared
  value; a commitment item is one of those roots.  Anything else is refused (`Refused`), at `register` and at `receive`.
* **`receive(spec, rec, ports, expect)`**: the rule, the protocol's own recomputation of its items (`expect`: the
  verifier's values, `None` for a commitment), and one-stage's R1-R6 against the verifier's own copies; the receipt.
* **`select(spec, rec, receipt, sample, log)`**: the verifier's draw from its own randomness (`sample(population, k)`,
  Lean `flock-verify draw`), after the receipt and once per receipt: a second select of a receipt is refused, so no retry
  redraws.
* **`outcome(spec, rec, receipt, draw, units, log)`**: one-stage's audit record over every drawn unit, accepted only when
  each unit's verdicts accept, and the `Outcome` whose `public` is exactly the spec's public items.  Final: a second
  outcome of a draw is refused.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from verity.primitives.circuits.codec import canonical_json
from verity.primitives.circuits.program import Program
from verity.protocols.verification.sampled_proofs.one_stage import audit as A
from verity.protocols.verification.sampled_proofs.one_stage import draw as D
from verity.protocols.verification.sampled_proofs.one_stage import registration as R
from verity.protocols.verification.sampled_proofs.one_stage.verdict import _Collect

PUBLIC_KINDS = ("structure", "output", "coins", "commitment")
SOURCES = ("parameter", "window", "outcome")
_ROOT = re.compile(r"^[0-9a-f]{128}$")


class Refused(ValueError):
    """A registration, draw or outcome the service will not make."""


@dataclass(frozen=True)
class Public:
    item: str
    kind: str
    reason: str
    source: str


@dataclass(frozen=True)
class Value:
    name: str
    schema: str | None
    phase: str
    registered: bool = False


@dataclass(frozen=True)
class Check:
    program: Program
    query: Mapping[str, Any]
    scheme: str
    public_outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Spec:
    values: tuple[Value, ...]
    selection: Mapping[str, Any]
    check: Check
    public: tuple[Public, ...]
    context: str

    def registered_items(self) -> set[str]:
        return {p.item for p in self.public if p.source != "outcome"}


@dataclass(frozen=True)
class Outcome:
    accepted: bool
    verdict: dict[str, Any]
    profile: dict[str, Any] | None
    terms: dict[str, Any]
    public: dict[str, Any]
    record: str


def public_problem(spec: Spec, rec: Mapping[str, Any]) -> dict[str, Any]:
    """The public-items rule on a registration record, as a one-stage verdict's JSON."""
    c = _Collect()
    window = rec.get("window")
    if not isinstance(window, Mapping) or set(window) != {"kind", "context", "public"}:
        c.fail("public-window", sorted(window) if isinstance(window, Mapping) else window)
        return c.verdict().to_json()
    if window["kind"] != "served" or window["context"] != spec.context:
        c.fail("public-context", {"registered": [window["kind"], window["context"]], "spec": ["served", spec.context]})
    items = window["public"]
    if not isinstance(items, Mapping) or set(items) != spec.registered_items():
        c.fail("public-items", {"registered": sorted(items) if isinstance(items, Mapping) else items,
                                "declared": sorted(spec.registered_items())})
    ports = {v.name for v in spec.values if v.phase in ("epoch", "serve")}
    roots = rec.get("roots") if isinstance(rec.get("roots"), list) else []
    if sorted(r.get("port") for r in roots if isinstance(r, Mapping)) != sorted(ports) or len(roots) != len(ports):
        c.fail("public-roots", {"registered": [r.get("port") for r in roots if isinstance(r, Mapping)], "values": sorted(ports)})
    if not all(isinstance(r, Mapping) and isinstance(r.get("root"), str) and _ROOT.match(r["root"]) for r in roots):
        c.fail("public-salted", "a root is not a SHA-512 root")
    held = {r.get("root") for r in roots if isinstance(r, Mapping)}
    for p in spec.public:
        if p.kind == "commitment" and isinstance(items, Mapping) and p.item in items and items[p.item] not in held:
            c.fail("public-commitment", {"item": p.item, "registered": items[p.item]})
    if rec.get("leaf_layer") is not None or rec.get("anchors") != []:
        c.fail("public-extra", {"leaf_layer": rec.get("leaf_layer"), "anchors": rec.get("anchors")})
    return c.verdict().to_json()


def register(spec: Spec, items: Mapping[str, Any], roots: Sequence[Mapping[str, Any]], previous: str | None = None) -> dict:
    """The registration of `roots` with the public `items`; raises `Refused` under the public-items rule."""
    window = {"kind": "served", "context": spec.context, "public": dict(items)}
    rec = R.record(spec.check.program, spec.check.query, spec.selection, spec.check.scheme, roots, window, previous=previous)
    v = public_problem(spec, rec)
    if not v["ok"]:
        raise Refused(json.dumps(v))
    return rec


def receive(spec: Spec, rec: Mapping[str, Any], ports: Mapping[str, Mapping[str, Any]], expect: Mapping[str, Any],
            log: Sequence[Mapping[str, Any]] = (), verifier: str = "served-zk/lean-flock-verify") -> tuple[dict, dict | None]:
    """The verifier's checks of `rec` and its receipt when they hold: the public-items rule, each registered item against
    the verifier's own value (`expect[item]`, skipped where it is None), and one-stage's R1-R6 against the spec."""
    v = public_problem(spec, rec)
    if v["ok"]:
        items = rec["window"]["public"]
        wrong = {k: {"registered": items.get(k), "verifier": want} for k, want in expect.items()
                 if want is not None and items.get(k) != want}
        mine = R.check(rec, spec.check.program, spec.check.query, spec.selection, spec.check.scheme, ports, log=log).to_json()
        codes = (["public-value"] if wrong else []) + mine["codes"]
        v = {"ok": not codes, "codes": codes, "detail": {**({"public-value": wrong} if wrong else {}), **mine["detail"]}}
    return v, (R.receipt(rec, verifier) if v["ok"] else None)


def _once(log: Path, name: str) -> Path:
    log.mkdir(parents=True, exist_ok=True)
    p = log / name
    if p.exists():
        raise Refused(f"{name} exists: the service makes it once, and a retry never redraws")
    return p


def select(spec: Spec, rec: Mapping[str, Any], receipt: Mapping[str, Any], sample: Callable[[int, int], dict],
           log: Path) -> dict:
    """The draw under the spec's law over the registered population, after `receipt`, once."""
    if receipt.get("registration") != R.digest(rec):
        raise Refused("the receipt is not this registration's")
    if spec.selection.get("law") != "subset":
        raise Refused(f"the stand-in selects subset laws only, not {spec.selection}")
    out = _once(log, f"draw-{receipt['registration'][:32]}.json")
    draw = sample(rec["population"], spec.selection["k"])
    v = D.check_draw(draw, rec, spec.selection)
    if not v.ok:
        raise Refused(f"the sampler's draw is not the law's: {v.to_json()}")
    out.write_text(json.dumps(draw, sort_keys=True))
    return draw


def _all(units: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    vs = [u["verdicts"].get(name) for u in units]
    if all(v is None for v in vs):
        return {}
    ran = [v for v in vs if v is not None and v.get("accepted") is not None]
    accepted = None if len(ran) < len(vs) else all(v["accepted"] for v in ran)
    return {name: {"of_record": any(bool(v and v.get("of_record")) for v in vs), "accepted": accepted,
                   "why": next((v.get("why", "") for v in ran if not v["accepted"]), ""),
                   "digest": hashlib.sha512(canonical_json([(v or {}).get("digest") for v in vs])).hexdigest(),
                   "units": [(v or {}).get("accepted") for v in vs]}}


def outcome(spec: Spec, rec: Mapping[str, Any], receipt: Mapping[str, Any], draw: Mapping[str, Any],
            units: Sequence[Mapping[str, Any]], delta: float, log: Path, public: Mapping[str, Any]) -> tuple[Outcome, dict]:
    """The window's outcome from each proved unit's `{"unit", "statement", "verdicts"}`, ascending (`draw.proved`):
    one-stage's audit record over them, and the `Outcome`, whose public items are the registered ones with the verdict
    and the draw.  `public` holds the protocol's own outcome-time items besides those two (none for PoUW)."""
    proved = [u["unit"] for u in units]
    if proved != D.proved(draw):
        raise Refused("the units are not the draw's, ascending")
    out = _once(log, f"outcome-{receipt['registration'][:32]}.json")
    names = sorted({n for u in units for n in u["verdicts"]})
    verdicts = {k: v for n in names for k, v in _all(units, n).items()}
    statement = {"units": [{"unit": u["unit"], **u["statement"]} for u in units]}
    rec_a = A.audit_record(rec, receipt, draw, spec.selection, proved, len(units), statement, verdicts, delta)
    items = {**rec["window"]["public"], **public, "verdict": rec_a["accepted"], "drawn": list(draw["units"])}
    declared = {p.item for p in spec.public}
    if set(items) != declared:
        raise Refused(f"the outcome's public items {sorted(items)} are not the declared {sorted(declared)}")
    o = Outcome(rec_a["accepted"], {"accepted": rec_a["accepted"], "checks": rec_a["checks"], "verdicts": rec_a["verdicts"]},
                rec_a["profile"], {"wrong_units": rec_a["wrong_units"]}, items,
                hashlib.sha512(canonical_json(rec_a)).hexdigest())
    out.write_text(json.dumps({"public": o.public, "record": o.record}, sort_keys=True))
    return o, rec_a


__all__ = ["Check", "Outcome", "Public", "Refused", "Spec", "Value", "outcome", "public_problem", "receive", "register",
           "select"]
