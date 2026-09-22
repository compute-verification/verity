"""Bernoulli baseline security sampler with commitment -> challenge chronology (brief A.4/A.5; lane PROTO-A, A1).

LEGACY STAGES (DIRECTIVE v4) -- [A10 2026-09-10, WR-8].  ``stage1`` / ``stage2`` / ``reveal`` are the pre-contract sampler: no epoch
declaration, no exemption request (``B_ex = 0`` hard-wired), and by default the challenge seed is written to ``seed.hex`` in the record
directory on the prover's host (docs/PROTOCOL-CONFORMANCE.md R7).  Those three CLI subcommands refuse to run without
``--legacy-directive-v4``; ``stage1(..., seed_file=False)`` keeps the seed in this process (``_HELD_SEEDS``) for callers that need it.
The pure helpers (``phase_digest_q/s``, ``draw_J``, ``draw_Q``, ``challenge_digest``, ``_event``) and the re-derivation entry points
(``verify``, ``from_transcript``, ``verify_adapter_transcript``) are NOT legacy: ``e2e.partition_sample`` imports the helpers and the
contract path is ``e2e.partition_sample`` (stages with barriers + F) driven by ``e2e.verifier_v3.driver`` / ``record_v5``.

The paper's Protocol 2 with ``B_ex = 0`` over the population of a protocol instance (``e2e.protocol_instance``): every replay unit
of the validated partition independently with probability ``q`` (``J``), then -- after the retained interiors are committed --
every verification unit inside ``J``'s RUs independently with probability ``s`` (``Q``).  Draws reuse
``veritor.protocol.challenge.bernoulli_subset`` (exact rational ``p``; HMAC-SHA256 keyed by the verifier seed over a phase digest;
work ``O(K log N)``), so the joint law is the product of independent coins and both parties re-derive the same selection.

Chronology (``chronology.json``): an append-only event list ``{order, kind, name, digest, utc, derived_from}`` in the instance's
``sampler.chronology`` format -- the instance declaration, the verifier's seed commitment, the boundary commitment, ``r_q`` (derived
from instance digest + boundary digest), the interior commitment, ``r_s`` (derived from instance digest + ``r_q`` + interior digest),
the seed reveal.  ``challenges_drawn`` lists every challenge ever produced for this instance; a second stage-1 or stage-2 draw against an
existing chronology is REFUSED (no rerolls), and realized sizes are written to ``realized`` only -- never into the law.

    python -m e2e.security_sample stage1 --instance I.json --population POP.json --boundary-digest HEX --out DIR [--seed HEX]
    python -m e2e.security_sample stage2 --instance I.json --population POP.json --interior-digest HEX --out DIR
    python -m e2e.security_sample reveal --out DIR
    python -m e2e.security_sample verify --instance I.json --population POP.json --out DIR    # re-derive J, Q from the revealed seed

``POP.json`` is a population file ``{"vu_count", "ru_count", "ru_of": [ru per vu], "plan_digest", "program_digest"}``
(``e2e.population`` writes one: ``population.json`` next to the maps).  The fixed-size path (``run_composed --k``) is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
from fractions import Fraction
from itertools import accumulate
from bisect import bisect_right
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from e2e.protocol_instance import InstanceError, instance_digest, load_instance  # noqa: E402

SCHEMA = "veritor/security-sample-chronology/v1"
SEED_DOMAIN = b"veritor/proto-a/seed-commitment/v1"
PHASE_DOMAIN = b"veritor/proto-a/phase/v1"
Q_STAGE = b"q/replay-unit"
S_STAGE = b"s/verification-unit"

# [A10 2026-09-10, WR-8] the stage1/stage2/reveal path is DIRECTIVE-v4 legacy (seed custody R7; no F / epoch).  Marker + CLI gate.
LEGACY_STAGES = True
LEGACY_FLAG = "--legacy-directive-v4"
LEGACY_CLI_COMMANDS = ("stage1", "stage2", "reveal")
LEGACY_POINTER = (
    f"e2e.security_sample: LEGACY (DIRECTIVE v4) sampler stages; refusing `stage1`/`stage2`/`reveal` without {LEGACY_FLAG}.\n"
    "  The seed is written to seed.hex on the prover's host (R7) and there is no exemption request / epoch (R8, R16).\n"
    "  Contract path: e2e.partition_sample stages driven by e2e.verifier_v3.driver (seed held in the verifier process; F, B_ex as RUs);\n"
    f"  runner: record_v5/.  Historical reproduction: re-run with {LEGACY_FLAG} (behaviour is otherwise unchanged; add --no-seed-file to hold the seed in process)."
)
_HELD_SEEDS: dict[str, bytes] = {}  # stage1(seed_file=False): the seed never touches the record directory; stage2/reveal read it here


def _seed_for(out: Path) -> bytes:
    """The seed of this chronology: held in this process when stage 1 kept it, else the legacy ``seed.hex``."""
    held = _HELD_SEEDS.get(str(out.resolve()))
    if held is not None:
        return held
    f = out / "seed.hex"
    if not f.exists():
        raise ChronologyError("seed is neither held by this process nor in seed.hex: stage 2 / reveal must run in the process that ran stage 1 with seed_file=False")
    return bytes.fromhex(f.read_text().strip())


class ChronologyError(ValueError):
    """A draw that the chronology forbids (reroll, stage out of order, seed mismatch)."""


def _h(*parts: bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return h.hexdigest()


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _limits():
    from veritor.core import VerificationLimits

    return VerificationLimits()


def phase_digest_q(instance_digest_hex: str, boundary_digest_hex: str) -> str:
    return _h(PHASE_DOMAIN, b"q", bytes.fromhex(instance_digest_hex), bytes.fromhex(boundary_digest_hex))


def phase_digest_s(instance_digest_hex: str, r_q_hex: str, interior_digest_hex: str) -> str:
    return _h(PHASE_DOMAIN, b"s", bytes.fromhex(instance_digest_hex), bytes.fromhex(r_q_hex), bytes.fromhex(interior_digest_hex))


def draw_J(seed: bytes, phase: str, ru_count: int, q: Fraction) -> tuple[int, ...]:
    from veritor.protocol.challenge import bernoulli_subset

    return bernoulli_subset(seed, Q_STAGE, bytes.fromhex(phase), ru_count, q, _limits())


def draw_Q(seed: bytes, phase: str, J: tuple[int, ...], vus_by_ru: list[list[int]], s: Fraction) -> tuple[int, ...]:
    """Bernoulli(s) over the VUs of the selected RUs (ranked block by block in J order), mapped back to VU indices."""

    from veritor.protocol.challenge import bernoulli_subset

    blocks = [vus_by_ru[r] for r in J]
    ends = list(accumulate(len(b) for b in blocks))
    count = ends[-1] if ends else 0
    ranks = bernoulli_subset(seed, S_STAGE, bytes.fromhex(phase), count, s, _limits())

    def vu_at(rank: int) -> int:
        b = bisect_right(ends, rank)
        start = ends[b] - len(blocks[b])
        return blocks[b][rank - start]

    return tuple(sorted(vu_at(r) for r in ranks))


def challenge_digest(stage: str, phase: str, count: int, p: Fraction, selection: tuple[int, ...]) -> str:
    doc = {"stage": stage, "phase_digest": phase, "count": count, "p": str(p), "selection": list(selection)}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# -- the chronology file ---------------------------------------------------------------------------------------------------


def _load_pop(path: Path) -> dict[str, Any]:
    pop = json.loads(Path(path).read_text())
    for k in ("vu_count", "ru_count", "ru_of", "plan_digest", "program_digest"):
        if k not in pop:
            raise ChronologyError(f"population file missing {k!r}")
    if len(pop["ru_of"]) != pop["vu_count"]:
        raise ChronologyError("population ru_of length != vu_count")
    return pop


def _check_instance(inst, pop: dict[str, Any]) -> None:
    if inst.sampler.law.kind != "bernoulli":
        raise ChronologyError(f"instance law is {inst.sampler.law.kind!r}; this sampler implements 'bernoulli' only (fixed-size stays on run_composed --k)")
    n = inst.partition.instances
    if inst.partition.vu_count.value * n != pop["vu_count"] or inst.partition.ru_count.value * n != pop["ru_count"]:
        raise ChronologyError(f"population {pop['vu_count']}/{pop['ru_count']} != instance partition {n} x {inst.partition.vu_count.value}/{inst.partition.ru_count.value}")
    if pop.get("instances", 1) != n:
        raise ChronologyError(f"population file instances {pop.get('instances', 1)} != instance partition.instances {n}")
    if inst.partition.plan_descriptor_digest != pop["plan_digest"] or inst.program.digest != pop["program_digest"]:
        raise ChronologyError("population file's plan/program digests differ from the instance's")
    if pop.get("partition_digest") != inst.partition.partition_digest:
        raise ChronologyError("population file's partition_digest differs from the instance's (draft 2: the partition itself is pinned)")


def _read(out: Path) -> dict[str, Any] | None:
    f = out / "chronology.json"
    return json.loads(f.read_text()) if f.exists() else None


def _write(out: Path, doc: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "chronology.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def _event(doc: dict[str, Any], kind: str, name: str, digest: str | None, derived_from: list[str], **extra: Any) -> None:
    doc["events"].append({"order": len(doc["events"]), "kind": kind, "name": name, "digest": digest, "utc": _utc(),
                          "derived_from": derived_from, **extra})


def stage1(instance: Path, population: Path, boundary_digest: str, out: Path, seed_hex: str | None = None, *, seed_file: bool = True) -> dict[str, Any]:
    """LEGACY stage (see module banner).  ``seed_file=True`` (the v4 default) writes ``seed.hex`` into ``out``; ``False`` holds the seed
    in this process (``_HELD_SEEDS``) so ``stage2``/``reveal`` in the same process can use it and nothing prover-readable is written."""
    inst = load_instance(instance)
    pop = _load_pop(population)
    _check_instance(inst, pop)
    idig = instance_digest(inst)
    existing = _read(out)
    if existing is not None:
        if any(e["kind"] == "challenge" for e in existing["events"]):
            raise ChronologyError("chronology already holds a challenge for this instance: a second stage-1 draw is a reroll and is refused")
        raise ChronologyError("chronology exists without a challenge: remove it explicitly (an aborted audit's chronology is evidence, not scratch)")
    if len(boundary_digest) != 64:
        raise ChronologyError("boundary commitment digest must be a sha256 hex")
    seed = bytes.fromhex(seed_hex) if seed_hex else secrets.token_bytes(32)
    if len(seed) != 32:
        raise ChronologyError("seed must be 32 bytes")
    q = Fraction(inst.sampler.law.q)
    doc: dict[str, Any] = {"schema": SCHEMA, "instance_digest": idig, "instance_path": str(instance), "law": {"kind": "bernoulli", "q": str(q), "s": inst.sampler.law.s},
                           "population": {"vu_count": pop["vu_count"], "ru_count": pop["ru_count"], "plan_digest": pop["plan_digest"], "program_digest": pop["program_digest"]},
                           "seed_derivation": "seed = 32 random bytes (secrets.token_bytes) held by the verifier; seed_commitment = H(domain, seed); "
                                              "r_q = bernoulli_subset(seed, 'q/replay-unit', phase_q, N_R, q) with phase_q = H(PHASE_DOMAIN, 'q', instance_digest, boundary_digest); "
                                              "r_s = bernoulli_subset(seed, 's/verification-unit', phase_s, |VUs of J|, s) with phase_s = H(PHASE_DOMAIN, 's', instance_digest, r_q_digest, interior_digest)",
                           "events": [], "challenges_drawn": [], "rerolls": 0, "realized": {}}
    _event(doc, "declaration", "protocol_instance", idig, [], note="status = " + inst.status)
    seed_commitment = _h(SEED_DOMAIN, seed)
    _event(doc, "commitment", "seed_commitment", seed_commitment, ["protocol_instance"])
    _event(doc, "commitment", "boundary", boundary_digest, ["protocol_instance"], note="prover's first-boundary commitment (all RUs' boundaries + designated outputs)")
    phase = phase_digest_q(idig, boundary_digest)
    J = draw_J(seed, phase, pop["ru_count"], q)
    cd = challenge_digest("q", phase, pop["ru_count"], q, J)
    _event(doc, "challenge", "r_q", cd, ["seed_commitment", "boundary"], phase_digest=phase, selection=list(J))
    doc["challenges_drawn"].append(cd)
    doc["realized"]["J_size"] = {"value": len(J), "label": "MEASURED", "source": "chronology.json events[r_q].selection (realized, not the rate q)"}
    if not seed_file:
        doc["seed_custody"] = "verifier_process_memory"  # the legacy (seed_file=True) chronology is unchanged byte for byte
    _write(out, doc)
    if seed_file:
        (out / "seed.hex").write_text(seed.hex() + "\n")  # LEGACY: verifier-private only if the prover cannot read the record dir (R7)
    else:
        _HELD_SEEDS[str(out.resolve())] = seed
    return doc


def stage2(instance: Path, population: Path, interior_digest: str, out: Path) -> dict[str, Any]:
    inst = load_instance(instance)
    pop = _load_pop(population)
    _check_instance(inst, pop)
    doc = _read(out)
    if doc is None:
        raise ChronologyError("no chronology: stage 1 has not run")
    if doc["instance_digest"] != instance_digest(inst):
        raise ChronologyError("chronology belongs to a different instance")
    if any(e["name"] == "r_s" for e in doc["events"]):
        raise ChronologyError("r_s already drawn for this instance: a second inner draw is a reroll and is refused")
    rq = next((e for e in doc["events"] if e["name"] == "r_q"), None)
    if rq is None:
        raise ChronologyError("no r_q in the chronology")
    if len(interior_digest) != 64:
        raise ChronologyError("interior commitment digest must be a sha256 hex")
    seed = _seed_for(out)
    if _h(SEED_DOMAIN, seed) != next(e["digest"] for e in doc["events"] if e["name"] == "seed_commitment"):
        raise ChronologyError("seed does not open the recorded seed commitment")
    s = Fraction(inst.sampler.law.s)
    _event(doc, "commitment", "interior", interior_digest, ["r_q"], note="prover's commitment to the retained interiors of J (no exemptions: B_ex = 0)")
    from e2e.population import vus_by_ru

    J = tuple(rq["selection"])
    phase = phase_digest_s(doc["instance_digest"], rq["digest"], interior_digest)
    groups = vus_by_ru(pop["ru_of"], pop["ru_count"])
    Q = draw_Q(seed, phase, J, groups, s)
    count = sum(len(groups[r]) for r in J)
    cd = challenge_digest("s", phase, count, s, Q)
    _event(doc, "challenge", "r_s", cd, ["r_q", "interior"], phase_digest=phase, selection=list(Q), candidates=count)
    doc["challenges_drawn"].append(cd)
    doc["realized"]["Q_size"] = {"value": len(Q), "label": "MEASURED", "source": "chronology.json events[r_s].selection (realized, not the rate s)"}
    doc["realized"]["Q_candidates"] = {"value": count, "label": "MEASURED", "source": "sum of |VUs of r| over r in J"}
    _write(out, doc)
    return doc


def reveal(out: Path) -> dict[str, Any]:
    doc = _read(out)
    if doc is None or not any(e["name"] == "r_s" for e in doc["events"]):
        raise ChronologyError("seed is revealed only after r_s (the second challenge) exists")
    if any(e["name"] == "seed_reveal" for e in doc["events"]):
        return doc
    seed_hex = _seed_for(out).hex()
    _event(doc, "message", "seed_reveal", None, ["r_s"], seed=seed_hex)
    _write(out, doc)
    return doc


def verify(instance: Path, population: Path, out: Path) -> list[str]:
    """Re-derive J and Q from the revealed seed and the recorded commitments; list every disagreement."""

    problems: list[str] = []
    inst = load_instance(instance)
    pop = _load_pop(population)
    _check_instance(inst, pop)
    doc = _read(out)
    if doc is None:
        return ["no chronology.json"]
    if doc["instance_digest"] != instance_digest(inst):
        problems.append("chronology instance digest != instance")
    ev = {e["name"]: e for e in doc["events"]}
    for name in ("protocol_instance", "seed_commitment", "boundary", "r_q", "interior", "r_s", "seed_reveal"):
        if name not in ev:
            problems.append(f"chronology lacks event {name!r}")
    if problems:
        return problems
    orders = [e["order"] for e in doc["events"]]
    if orders != sorted(orders):
        problems.append("events out of order")
    for a, b in (("seed_commitment", "r_q"), ("boundary", "r_q"), ("r_q", "interior"), ("interior", "r_s"), ("r_s", "seed_reveal")):
        if ev[a]["order"] >= ev[b]["order"]:
            problems.append(f"{a} does not precede {b}")
    seed = bytes.fromhex(ev["seed_reveal"]["seed"])
    if _h(SEED_DOMAIN, seed) != ev["seed_commitment"]["digest"]:
        problems.append("revealed seed does not open the seed commitment")
    q, s = Fraction(inst.sampler.law.q), Fraction(inst.sampler.law.s)
    phase_q = phase_digest_q(doc["instance_digest"], ev["boundary"]["digest"])
    if phase_q != ev["r_q"]["phase_digest"]:
        problems.append("r_q phase digest != H(instance, boundary)")
    J = draw_J(seed, phase_q, pop["ru_count"], q)
    if list(J) != ev["r_q"]["selection"]:
        problems.append("r_q selection does not re-derive")
    if challenge_digest("q", phase_q, pop["ru_count"], q, J) != ev["r_q"]["digest"]:
        problems.append("r_q digest does not re-derive")
    from e2e.population import vus_by_ru

    phase_s = phase_digest_s(doc["instance_digest"], ev["r_q"]["digest"], ev["interior"]["digest"])
    if phase_s != ev["r_s"]["phase_digest"]:
        problems.append("r_s phase digest != H(instance, r_q, interior)")
    groups = vus_by_ru(pop["ru_of"], pop["ru_count"])
    Q = draw_Q(seed, phase_s, J, groups, s)
    if list(Q) != ev["r_s"]["selection"]:
        problems.append("r_s selection does not re-derive")
    for vu in ev["r_s"]["selection"]:
        if not 0 <= vu < pop["vu_count"]:
            problems.append(f"sampled VU {vu} outside the population [0, {pop['vu_count']})")
        elif pop["ru_of"][vu] not in J:
            problems.append(f"sampled VU {vu} lies in RU {pop['ru_of'][vu]} which is not in J")
    drawn = [e["digest"] for e in doc["events"] if e["kind"] == "challenge"]
    if drawn != doc["challenges_drawn"]:
        problems.append("challenges_drawn != the challenge events (a draw is missing from the record)")
    if doc.get("rerolls", 0) != 0:
        problems.append("rerolls != 0")
    return problems


ADAPTER_KIND = "adapter-transcript/v1"
ADAPTER_SEED_DERIVATION = ("verity_vllm.challenge.InteractiveVerifierSource: seed = 32 OS bytes drawn by the verifier before the statement; seed_commitment = "
                           "H('verity-vllm/challenge-seed-commitment/v1', seed) is a field of T_0; r_q = HMAC-SHA256(seed, frame(domain, T_0, 'q/replay-unit')), "
                           "J = core bernoulli_subset(r_q, N_requests, q); T_1 = H(T_0, replay_challenge); T_2 = H(T_1, committed interiors of J); "
                           "r_s = HMAC-SHA256(seed, frame(domain, T_2, 's/verification-unit')), Q = core bernoulli_subset(r_s, |units of J|, s) over the committed "
                           "relation units of J (block order); T_3 = H(T_2, sample_challenge); seed revealed after T_3")


def from_transcript(record: Path, out: Path) -> dict[str, Any]:
    """[REC] ``chronology.json`` (kind ``adapter-transcript/v1``) from a ``run_composed`` selective record: the vLLM adapter's
    ``verify/challenge_transcript.json`` + ``verify/statement.json`` + ``handshake/events.json`` (UTC stamps of the relayed
    messages).  Every digest is copied from the transcript; nothing is drawn here.  ``e2e.instance_check`` re-derives it."""

    record = Path(record)
    tr = json.loads((record / "verify" / "challenge_transcript.json").read_text())
    st = json.loads((record / "verify" / "statement.json").read_text())
    hs = json.loads((record / "handshake" / "events.json").read_text()) if (record / "handshake" / "events.json").exists() else {"events": []}
    manifest = json.loads((record / "manifest.json").read_text())
    pin = (manifest.get("inputs") or {}).get("protocol_instance") or {}
    stamps = {e.get("file"): e.get("utc") for e in hs.get("events", []) if e.get("relayed")}
    phases = {p["phase"]: p["digest"] for p in tr.get("phases", [])}
    rc, sc, rv = tr["replay_challenge"], tr["sample_challenge"], tr.get("reveal") or {}
    n_req = len(st.get("boundary_roots") or {})
    doc: dict[str, Any] = {
        "schema": SCHEMA, "chronology_kind": ADAPTER_KIND, "instance_digest": pin.get("instance_digest"), "instance_path": str(record / "protocol_instance.json"),
        "law": {"kind": "bernoulli", "q": str(Fraction(tr["policy"]["q"])), "s": str(Fraction(tr["policy"]["s"]))},
        "population": {"note": "the adapter's inner population is its committed relation units of J (sample_challenge.blocks), NOT the instance's validated VUs; "
                               "e2e.instance_check reports the difference", "requests": n_req, "units_of_J": sc.get("population"), "blocks": len(sc.get("blocks") or [])},
        "seed_derivation": ADAPTER_SEED_DERIVATION, "events": [], "challenges_drawn": [], "rerolls": 0, "realized": {},
        "transcript": {"path": "verify/challenge_transcript.json", "sha256": hashlib.sha256((record / "verify" / "challenge_transcript.json").read_bytes()).hexdigest(),
                       "statement": "verify/statement.json", "statement_sha256": hashlib.sha256((record / "verify" / "statement.json").read_bytes()).hexdigest()},
    }

    def ev(kind: str, name: str, digest: str | None, derived_from: list[str], utc: str | None, **extra: Any) -> None:
        doc["events"].append({"order": len(doc["events"]), "kind": kind, "name": name, "digest": digest, "utc": utc, "derived_from": derived_from, **extra})

    ev("declaration", "protocol_instance", pin.get("instance_digest"), [], pin.get("pinned_utc"), note="pinned in manifest.inputs.protocol_instance before READY_PHASE0")
    ev("commitment", "seed_commitment", tr["source"]["seed_commitment"], ["protocol_instance"], stamps.get("READY_PHASE0"),
       note="verifier's seed commitment; a field of T_0 (statement.source.seed_commitment)")
    ev("commitment", "statement", phases.get("statement"), ["protocol_instance", "seed_commitment"], stamps.get("READY_PHASE0"),
       note=f"T_0: phase-0 commitments of every request (boundary trees = designated outputs + prompt tokens + step records of {n_req} requests), weights root, "
            "description / required digests, bindings, policy, seed commitment -- the first boundary of the whole-request partition, before r_q",
       boundary_roots={k: v.get("root") for k, v in (st.get("boundary_roots") or {}).items()}, weights_root=(st.get("weights_root") or {}).get("root"))
    ev("challenge", "r_q", rc["r_q"], ["seed_commitment", "statement"], stamps.get("replay_challenge.json"), phase_digest=phases.get("statement"),
       selection=list(rc.get("selected_indices") or []), selected=list(rc.get("selected") or []), candidates=rc.get("candidates"), selector=rc.get("selector"))
    doc["challenges_drawn"].append(rc["r_q"])
    ev("commitment", "replay_phase", phases.get("replay"), ["r_q"], stamps.get("replay_challenge.json"), note="T_1 = H(T_0, replay_challenge)")
    ev("commitment", "interior", phases.get("interior"), ["replay_phase"], stamps.get("READY_PHASE2"),
       note=f"T_2 = H(T_1, committed interiors of J): {tr.get('interiors', {}).get('trees')} replay-unit trees, digest {tr.get('interiors', {}).get('digest')}",
       interiors_digest=tr.get("interiors", {}).get("digest"))
    ev("challenge", "r_s", sc["r_s"], ["r_q", "interior"], stamps.get("READY_PHASE2"), phase_digest=phases.get("interior"), selection=list(sc.get("selected_ranks") or []),
       candidates=sc.get("population"), certified_sample=list(sc.get("certified_sample") or []))
    doc["challenges_drawn"].append(sc["r_s"])
    ev("commitment", "sample_phase", phases.get("sample"), ["r_s"], stamps.get("READY_PHASE2"), note="T_3 = H(T_2, sample_challenge)")
    ev("message", "seed_reveal", None, ["sample_phase"], None, seed=rv.get("seed"))
    doc["realized"] = {"J_size": {"value": len(rc.get("selected") or []), "label": "MEASURED", "source": "challenge_transcript.json replay_challenge.selected"},
                       "J_candidates": {"value": rc.get("candidates"), "label": "MEASURED", "source": "replay_challenge.candidates (requests)"},
                       "Q_size": {"value": len(sc.get("selected_ranks") or []), "label": "MEASURED", "source": "sample_challenge.selected_ranks"},
                       "Q_candidates": {"value": sc.get("population"), "label": "MEASURED", "source": "sample_challenge.population (committed relation units of J)"}}
    out.mkdir(parents=True, exist_ok=True)
    _write(out, doc)
    return doc


def verify_adapter_transcript(record: Path, out: Path) -> list[str]:
    """Re-derive an ``adapter-transcript/v1`` chronology from the record's transcript with the core (CPU-only) functions:
    T_0 from statement.json, the seed commitment, r_q and J, T_1, T_2 (when the committed interiors are at hand), r_s and the
    certified ranks, T_3; and the chronology events' consistency with the transcript.  Certified-sample *ids* need the RU indices
    (verify_composed's bundle row re-derives them on a machine holding the bundle)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from verity_vllm.challenge import CORE_STAGE_Q, CORE_STAGE_S, PURPOSE_Q, PURPOSE_S, Policy, RankedUnits, _HmacSource, core_bernoulli_subset, interior_phase, \
        replay_phase, sample_phase, seed_commitment, statement_digest

    record = Path(record)
    problems: list[str] = []
    doc = _read(out)
    if doc is None:
        return ["no chronology.json"]
    if doc.get("chronology_kind") != ADAPTER_KIND:
        return [f"chronology_kind {doc.get('chronology_kind')!r} != {ADAPTER_KIND!r}"]
    tpath, spath = record / "verify" / "challenge_transcript.json", record / "verify" / "statement.json"
    for p in (tpath, spath):
        if not p.exists():
            return [f"{p.relative_to(record)} missing"]
    if hashlib.sha256(tpath.read_bytes()).hexdigest() != doc["transcript"]["sha256"] or hashlib.sha256(spath.read_bytes()).hexdigest() != doc["transcript"]["statement_sha256"]:
        problems.append("chronology.json was built from a different transcript / statement (sha mismatch)")
    tr, st = json.loads(tpath.read_text()), json.loads(spath.read_text())
    ev = {e["name"]: e for e in doc["events"]}
    for name in ("protocol_instance", "seed_commitment", "statement", "r_q", "replay_phase", "interior", "r_s", "sample_phase", "seed_reveal"):
        if name not in ev:
            problems.append(f"chronology lacks event {name!r}")
    if problems:
        return problems
    for a, b in (("seed_commitment", "r_q"), ("statement", "r_q"), ("r_q", "interior"), ("interior", "r_s"), ("r_s", "seed_reveal")):
        if ev[a]["order"] >= ev[b]["order"]:
            problems.append(f"{a} does not precede {b}")
    phases = {p["phase"]: p["digest"] for p in tr.get("phases", [])}
    t0 = statement_digest(st)
    if t0.hex() != phases.get("statement") or ev["statement"]["digest"] != t0.hex():
        problems.append("T_0 does not re-derive from statement.json")
    if (st.get("source") or {}).get("seed_commitment") != ev["seed_commitment"]["digest"]:
        problems.append("seed commitment in the chronology != the one inside T_0 (statement.source)")
    seed_hex = ev["seed_reveal"].get("seed")
    if not seed_hex:
        return problems + ["no seed revealed"]
    seed = bytes.fromhex(seed_hex)
    if seed_commitment(seed) != ev["seed_commitment"]["digest"]:
        problems.append("revealed seed does not open the seed commitment")
    rc, sc = tr["replay_challenge"], tr["sample_challenge"]
    r_q = _HmacSource.derive(seed, t0, PURPOSE_Q)
    if r_q.hex() != rc["r_q"] or ev["r_q"]["digest"] != rc["r_q"]:
        problems.append("r_q does not re-derive from the seed and T_0")
    policy = Policy.parse(tr["policy"]["q"], tr["policy"]["s"])
    if str(policy.q) != doc["law"]["q"] or str(policy.s) != doc["law"]["s"]:
        problems.append("chronology law != transcript policy")
    candidates = sorted(k.split("/", 1)[1] for k in (st.get("boundary_roots") or {}))
    sel = rc.get("selector") or {}
    if sel.get("kind") == "fixed_size":
        problems.append("transcript selector is fixed_size: not a Bernoulli chronology")
    else:
        picked = core_bernoulli_subset(r_q, CORE_STAGE_Q, t0, len(candidates), policy.q)
        if [candidates[i] for i in picked] != list(rc["selected"]) or list(picked) != ev["r_q"]["selection"]:
            problems.append("J does not re-derive from r_q over the statement's requests")
    t1 = replay_phase(t0, rc)
    if t1.hex() != phases.get("replay") or ev["replay_phase"]["digest"] != t1.hex():
        problems.append("T_1 does not re-derive")
    cpath = record / "adapter" / "selective" / "commitments.json"
    t2_hex = phases.get("interior")
    if cpath.exists():
        trees = (json.loads(cpath.read_text()).get("trees") or {})
        interiors = {}
        for tid, _n in sc.get("blocks") or []:
            rec = trees.get(tid)
            if rec is None:
                problems.append(f"committed interiors lack {tid}")
                break
            interiors[tid] = {"root": rec["root"], "leaf_count": rec["leaf_count"], "ids_digest": rec.get("ids_digest"), "phase": rec.get("phase"), "positions": rec.get("positions")}
        t2 = interior_phase(t1, interiors)
        if t2.hex() != t2_hex:
            problems.append("T_2 does not re-derive from the committed interiors of J")
    else:
        doc.setdefault("rederive_notes", []).append("T_2 recorded (adapter/selective/commitments.json not beside the record); verify_composed's bundle row re-derives it")
    if ev["interior"]["digest"] != t2_hex:
        problems.append("interior event digest != T_2")
    t2b = bytes.fromhex(t2_hex)
    r_s = _HmacSource.derive(seed, t2b, PURPOSE_S)
    if r_s.hex() != sc["r_s"] or ev["r_s"]["digest"] != sc["r_s"]:
        problems.append("r_s does not re-derive from the seed and T_2")
    units = RankedUnits([(tid, int(n)) for tid, n in sc.get("blocks") or []])
    ranks = core_bernoulli_subset(r_s, CORE_STAGE_S, t2b, units.count, policy.s)
    if list(ranks) != list(sc["selected_ranks"]) or list(ranks) != ev["r_s"]["selection"]:
        problems.append("the certified ranks do not re-derive from r_s")
    if units.count != sc.get("population") or ev["r_s"].get("candidates") != units.count:
        problems.append("inner population != sum of the committed blocks of J")
    t3 = sample_phase(t2b, sc)
    if t3.hex() != phases.get("sample") or ev["sample_phase"]["digest"] != t3.hex():
        problems.append("T_3 does not re-derive")
    if doc["challenges_drawn"] != [rc["r_q"], sc["r_s"]]:
        problems.append("challenges_drawn != [r_q, r_s]")
    if doc.get("rerolls", 0) != 0:
        problems.append("rerolls != 0")
    return problems


def chronology_for_instance(doc: dict[str, Any]) -> dict[str, Any]:
    """The ``sampler.chronology`` section of a protocol instance, from a chronology.json (for the corrected record's instance)."""

    return {"seed_derivation": doc["seed_derivation"], "seed_commitment": next(e["digest"] for e in doc["events"] if e["name"] == "seed_commitment"),
            "events": [{"order": e["order"], "kind": e["kind"], "name": e["name"], "digest": e["digest"], "utc": e["utc"], "derived_from": e["derived_from"],
                        "source": "chronology.json"} for e in doc["events"]],
            "challenges_drawn": list(doc["challenges_drawn"]), "rerolls": int(doc["rerolls"]), "source": "e2e.security_sample chronology.json"}


def main(argv: list[str] | None = None) -> int:
    # [A10 WR-8] R14 legacy gate on the seed-producing subcommands only; verify / from-transcript / verify-transcript are re-derivations.
    args = list(sys.argv[1:] if argv is None else argv)
    legacy_ok = LEGACY_FLAG in args
    args = [x for x in args if x != LEGACY_FLAG]
    if any(c in args for c in LEGACY_CLI_COMMANDS) and not legacy_ok and not any(x in ("-h", "--help") for x in args):
        print(LEGACY_POINTER, file=sys.stderr)
        return 2
    argv = args
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(LEGACY_FLAG, action="store_true", help="REQUIRED for stage1/stage2/reveal: acknowledge the DIRECTIVE-v4 legacy sampler stages (see module banner)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stage1", "stage2", "verify"):
        sp = sub.add_parser(name)
        sp.add_argument("--instance", required=True)
        sp.add_argument("--population", required=True)
        sp.add_argument("--out", required=True)
        if name == "stage1":
            sp.add_argument("--boundary-digest", required=True)
            sp.add_argument("--seed", default=None, help="32-byte hex (tests only; default: fresh OS randomness)")
            sp.add_argument("--no-seed-file", action="store_true", help="hold the seed in this process instead of writing seed.hex (R7); stage2/reveal must then run in the same process")
        if name == "stage2":
            sp.add_argument("--interior-digest", required=True)
    rv = sub.add_parser("reveal")
    rv.add_argument("--out", required=True)
    for name in ("from-transcript", "verify-transcript"):
        ft = sub.add_parser(name, help="[REC] chronology.json of kind adapter-transcript/v1 from / against a run_composed selective record")
        ft.add_argument("--record", required=True)
        ft.add_argument("--out", required=True, help="directory holding chronology.json (normally the record root)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "stage1":
            doc = stage1(Path(a.instance), Path(a.population), a.boundary_digest, Path(a.out), a.seed, seed_file=not a.no_seed_file)
            print(f"[security_sample] r_q drawn: |J| = {doc['realized']['J_size']['value']} of N_R = {doc['population']['ru_count']} (q = {doc['law']['q']})")
        elif a.cmd == "stage2":
            doc = stage2(Path(a.instance), Path(a.population), a.interior_digest, Path(a.out))
            print(f"[security_sample] r_s drawn: |Q| = {doc['realized']['Q_size']['value']} of {doc['realized']['Q_candidates']['value']} candidates (s = {doc['law']['s']})")
        elif a.cmd == "reveal":
            reveal(Path(a.out))
            print("[security_sample] seed revealed")
        elif a.cmd == "from-transcript":
            doc = from_transcript(Path(a.record), Path(a.out))
            print(f"[security_sample] chronology ({ADAPTER_KIND}) written: {len(doc['events'])} events; |J| = {doc['realized']['J_size']['value']}/"
                  f"{doc['realized']['J_candidates']['value']} requests, |Q| = {doc['realized']['Q_size']['value']}/{doc['realized']['Q_candidates']['value']} units")
        elif a.cmd == "verify-transcript":
            problems = verify_adapter_transcript(Path(a.record), Path(a.out))
            for pr in problems:
                print(f"[security_sample]   problem: {pr}")
            print(f"[security_sample] {'REDERIVED' if not problems else 'MISMATCH'} ({len(problems)} problem(s))")
            return 0 if not problems else 1
        else:
            problems = verify(Path(a.instance), Path(a.population), Path(a.out))
            for pr in problems:
                print(f"[security_sample]   problem: {pr}")
            print(f"[security_sample] {'REDERIVED' if not problems else 'MISMATCH'} ({len(problems)} problem(s))")
            return 0 if not problems else 1
    except (ChronologyError, InstanceError) as e:
        print(f"[security_sample] REFUSED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
