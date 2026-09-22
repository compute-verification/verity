"""[A9 contract 2026-09-10] Run configuration (``veritor/a9/run-config/v1``) and the LEGACY GUARD (R14).

A run is described by one JSON file.  :func:`load` parses it and REFUSES, with a named error, every legacy configuration the contract
rules out (docs/PROTOCOL.md; docs/PROTOCOL-CONFORMANCE.md R2/R6/R8/R11/R14/R16):

    E_LEGACY_BEX_UNIT        ``epoch.B_ex`` is not an RU count (``B_ex_unit != "RU"``, or the legacy keys ``budget_vus`` /
                             ``B_ex_vus`` / ``exempt_budget_vu`` / ``exemption_budget_vu`` are present)
    E_LEGACY_NO_EPOCH        no ``epoch`` section (the legacy "per-request budget" shape); or ``per_request_budget`` present
    E_LEGACY_FAMILY_EXEMPT   ``f_policy`` names a family exemption / "exempt (charged)" (``exempt-family``, ``exempt-charged``,
                             ``exempt_charged``, ``charged``) or ``exemption_policy`` is present
    E_LEGACY_INPUTS_AS_VUS   ``population.inputs_as_vus`` true, or ``population.source == "legacy"`` (the legacy population with
                             Input kinds as VUs) -- only ``legacy-minus-inputs`` (labelled) or ``a1`` are accepted
    E_LEGACY_CAPTURE         ``capture.fields`` beyond {inputs, outputs, seed} (decision 1: serve-time capture is the three fields)
    E_LEGACY_DRIVER          ``driver`` / ``verifier`` names a legacy module (``rec_p1_protocol``, ``protocol_6k``, ``verifier_v2``
                             as the verdict, ``partition_sample.stage2`` without exemptions)
    E_LEGACY_INSTANCE        ``instance.schema`` is ``veritor/protocol-instance/v1`` without ``epoch``
    E_CONFIG                 anything else malformed (unknown F policy / randomness mode, q/s not fractions in (0,1], ...)

Nothing here is "auto-migrated": a refused file has to be rewritten by a person, and the error says which key and why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA = "veritor/a9/run-config/v1"

F_POLICIES = ("exempt-on-inconsistency", "exempt-unsupported")
# A3's driver / prover name the same two policies `gamble` / `exempt-unsupported` (--exemption-policy); `exempt-on-inconsistency` is the
# run config's name for "no coverage exemption": A3's prover exempts an inconsistent / failed replay in either policy.
A3_POLICY = {"exempt-on-inconsistency": "gamble", "exempt-unsupported": "exempt-unsupported"}
FAMILY = "veritor.tc-ampere-bf16@2"  # the interface family of the fixed circuit's relations (A5/6 coverage_v1; A3 epoch.family)


def approved_elf_sha256(family: str = FAMILY) -> list[str]:
    """The approved guest set for the epoch: the `reproduced` entries of e2e/approved_guests.json whose `families` list the epoch's family
    (A3 ckpt 5: V4.guest ties family <-> ELF; a guest without the family is a refusal, so it is not declared)."""
    p = Path(__file__).resolve().parents[1] / "e2e" / "approved_guests.json"
    doc = json.loads(p.read_text())
    return sorted(g["elf_sha256"] for g in doc["guests"] if g.get("approval") == "reproduced" and family in (g.get("families") or {}))
RANDOMNESS_MODES = ("production", "reproducibility", "forced-test")
CAPTURE_FIELDS = ("inputs", "outputs", "seed")
POPULATION_SOURCES = ("legacy-minus-inputs", "a1")
WORKLOADS = {
    # name -> (model, prompt_len, max_tokens (= STEPS+1 served tokens per request), requests)
    "TINY-B0-4x8": ("HuggingFaceTB/SmolLM2-135M", 4, 9, 4),
    # [E10 EOD 2026-09-11] the ONE-request TINY epoch (common brief: N_R = 1, q = 1 so |J| = 1; same program Serve_v1{C={6f196795e564},LP=4,STEPS=8},
    # same operations and numerical semantics as attempt 6; `workload.request_indices` names WHICH of the B0 workload's requests it is)
    "TINY-B0-4x8-1req": ("HuggingFaceTB/SmolLM2-135M", 4, 9, 1),
    # [E1 NIGHT 2026-09-12, G3; PROPOSED] the second-model one-request epochs at the TINY shape (workload.registry_case SMOL360 / QWEN05,
    # workload.semantic_profile llama-v3 / qwen2-v2; run configs V5-SMOL360-4x9-bex0-inc.json, V5-QWEN05-4x9-bex0-inc.json)
    "SMOL360-SmolLM2-360M-4x8-1req": ("HuggingFaceTB/SmolLM2-360M", 4, 9, 1),
    "QWEN05-Qwen2.5-0.5B-4x8-1req": ("Qwen/Qwen2.5-0.5B", 4, 9, 1),
    "L1-32x15": ("HuggingFaceTB/SmolLM2-135M", 32, 16, 4),
    # [E1 SCALE 2026-09-12] the one-request SmolLM2-135M curve points (E8's NIGHT naming and run configs, verbatim: the same Serve_v1 program at
    # LP=P, STEPS=O-1; request = the shipped workload rule over B0-corpus prompt 0 truncated to LP); E8 carried these as an uncommitted dirty
    # patch on eod/e8 -- committed here so the SCALE records freeze a clean tree.  Owner (E10) may re-home them.
    "NIGHT-B0-128x63-1req": ("HuggingFaceTB/SmolLM2-135M", 128, 64, 1),
    # [E10 CURVE 2026-09-13, D56 item 3] the six TRANSITION records around the first two 64-key block boundaries: one request, prompt P,
    # 4 output tokens (STEPS = 3), the same workload rule (request index 0 -> random.Random(1) prompt); numerical comparisons, not protocol runs
    **{f"CURVE-B0-{p}x3-1req": ("HuggingFaceTB/SmolLM2-135M", p, 4, 1) for p in (63, 64, 65, 127, 128, 129)},
    "NIGHT-B0-256x63-1req": ("HuggingFaceTB/SmolLM2-135M", 256, 64, 1),  # [E1 SCALE D47] the third curve point (E6 pins 256x64; 512x128 not pinnable tonight)
    "NIGHT-B0-512x127-1req": ("HuggingFaceTB/SmolLM2-135M", 512, 128, 1),
    "SMALL-6500x1600": ("HuggingFaceTB/SmolLM2-135M", 6500, 1600, 4),
}

# [WR-7, A10] the legacy key vocabulary is A7's certificate_v2.LEGACY_KEYS (one list, refused everywhere), plus the run-config-only shapes
try:
    from e2e.certificate_v2 import LEGACY_KEYS as _CV2_LEGACY_KEYS
except ImportError:  # record_v5 imported without vllm-poc on sys.path yet
    from record_v5 import add_paths

    add_paths()
    from e2e.certificate_v2 import LEGACY_KEYS as _CV2_LEGACY_KEYS

_LEGACY_BEX_KEYS = tuple(sorted(_CV2_LEGACY_KEYS | {"budget_vus", "B_ex_vus", "exempt_budget_vu", "exemption_budget_vu", "budget_vu"}))
_LEGACY_TOP_KEYS = tuple(sorted(_CV2_LEGACY_KEYS | {"per_request_budget", "exemption_policy", "exempt_charged", "charged_exemptions", "family_exemption"}))
_LEGACY_F = tuple(sorted(set(_CV2_LEGACY_KEYS) | {"exempt-family", "exempt-charged", "exempt_charged", "charged", "family", "exempt (charged)"}))
_LEGACY_DRIVERS = ("rec_p1_protocol", "protocol_6k", "verifier_v2", "e2e.protocol", "verify_6k")


class ConfigError(ValueError):
    def __init__(self, code: str, msg: str) -> None:
        super().__init__(f"{code}: {msg}")
        self.code = code


@dataclass(frozen=True)
class Epoch:
    requests: int
    B_ex: int
    label: str  # "DEFAULT TAKEN" for the integrator (both B_ex = 0 and B_ex = 1 are DEFAULT TAKEN)

    def as_json(self) -> dict[str, Any]:
        return {"requests": self.requests, "B_ex": self.B_ex, "B_ex_unit": "RU", "label": self.label}


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    workload: str
    model: str
    prompt_len: int
    max_tokens: int
    requests: int
    epoch: Epoch
    q: Fraction
    s: Fraction
    delta: Fraction
    randomness_mode: str
    f_policy: str
    population_source: str
    plan: str
    guest_approved_only: bool
    family: str = FAMILY
    approved_elf_sha256: tuple[str, ...] = ()
    record_label: str | None = None
    s_source: str = "UNAVAILABLE (run config without sampling.s_source)"  # e.g. "as-built diagnostic -- coverage gap known" (USER DIRECTIVE 20:10Z §4); carried into the record and the report
    # [E10] which rows of the B0 workload (e2e/workload-B0-small.json) the epoch's requests are, in epoch order; default 0..requests-1.
    # The request ids (uuid5 over model | prompt ids | max_tokens) do not depend on the position, so a 1-request epoch over row 2 names
    # exactly the request attempt 6 drew as J = [1] (35285052-1526-5943-b9a6-3a2c8bbecc4a).
    request_indices: tuple[int, ...] = ()
    path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def is_production(self) -> bool:
        return self.randomness_mode == "production"

    @property
    def workload_short(self) -> str:
        return f"{self.prompt_len}x{self.max_tokens - 1}x{self.requests}"

    def as_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "run_id": self.run_id, "workload": {"name": self.workload, "model": self.model, "prompt_len": self.prompt_len,
                                                                   "max_tokens": self.max_tokens, "requests": self.requests, "greedy": True, "prefix_caching": False, "batch_invariant": True,
                                                                   **({"request_indices": list(self.request_indices)} if self.request_indices else {})},
            "epoch": self.epoch.as_json(), "sampling": {"q": str(self.q), "s": str(self.s), "delta": str(self.delta), "law": "bernoulli-per-unit", "s_source": self.s_source},
            "randomness": {"mode": self.randomness_mode, "verifier_seed_custody": "verifier-held" if self.is_production else "recorded"},
            "f_policy": self.f_policy, "capture": {"fields": list(CAPTURE_FIELDS)},
            "population": {"source": self.population_source, "plan": self.plan}, "guest": {"approved_elf_only": self.guest_approved_only, "family": self.family, "approved_elf_sha256": list(self.approved_elf_sha256)},
            **({"record_label": self.record_label} if self.record_label else {}),
        }


def _frac(v: Any, key: str) -> Fraction:
    try:
        f = Fraction(str(v))
    except (ValueError, ZeroDivisionError) as e:
        raise ConfigError("E_CONFIG", f"{key}={v!r} is not a fraction") from e
    if not 0 < f <= 1:
        raise ConfigError("E_CONFIG", f"{key}={v!r} outside (0, 1]")
    return f


def guard(doc: dict[str, Any]) -> None:
    """The legacy guard: raise :class:`ConfigError` for every legacy configuration; return silently for a contract-shaped one."""

    if doc.get("schema") != SCHEMA:
        raise ConfigError("E_CONFIG", f"schema {doc.get('schema')!r} != {SCHEMA!r}")
    for k in _LEGACY_TOP_KEYS:
        if k in doc:
            raise ConfigError("E_LEGACY_NO_EPOCH" if k == "per_request_budget" else "E_LEGACY_FAMILY_EXEMPT",
                              f"key {k!r} is a legacy (DIRECTIVE v4) configuration; the contract has one epoch with an RU-count B_ex and "
                              f"per-RU exemptions only (docs/PROTOCOL.md §3)")
    ep = doc.get("epoch")
    if not isinstance(ep, dict):
        raise ConfigError("E_LEGACY_NO_EPOCH", "no `epoch` section: the contract requires the epoch declaration (requests, B_ex as an RU count)")
    for k in _LEGACY_BEX_KEYS:
        if k in ep or k in doc:
            raise ConfigError("E_LEGACY_BEX_UNIT", f"key {k!r}: B_ex is a count of exempted RUs, never of VUs (PROTOCOL-CONFORMANCE R11)")
    if "B_ex_unit" not in ep:
        raise ConfigError("E_LEGACY_BEX_UNIT", "epoch.B_ex_unit is absent: the unit is never defaulted (WR-7); write \"RU\" explicitly")
    if ep.get("B_ex_unit") != "RU":
        raise ConfigError("E_LEGACY_BEX_UNIT", f"epoch.B_ex_unit={ep.get('B_ex_unit')!r}: B_ex is an RU count")
    if not isinstance(ep.get("B_ex"), int) or ep["B_ex"] < 0:
        raise ConfigError("E_CONFIG", f"epoch.B_ex={ep.get('B_ex')!r} must be a non-negative integer (RU count)")
    if not isinstance(ep.get("requests"), int) or ep["requests"] < 1:
        raise ConfigError("E_CONFIG", "epoch.requests must be a positive integer")
    fp = doc.get("f_policy")
    if isinstance(fp, str) and fp.strip().lower() in _LEGACY_F:
        raise ConfigError("E_LEGACY_FAMILY_EXEMPT", f"f_policy={fp!r}: family exemptions / 'exempt (charged)' are not in the contract; "
                                                    f"F is a set of RUs (settings: {F_POLICIES})")
    if fp not in F_POLICIES:
        raise ConfigError("E_CONFIG", f"f_policy={fp!r} not in {F_POLICIES}")
    pop = doc.get("population") or {}
    if pop.get("inputs_as_vus") is True:
        raise ConfigError("E_LEGACY_INPUTS_AS_VUS", "population.inputs_as_vus=true: inputs are not VUs (PROTOCOL-CONFORMANCE R2)")
    if pop.get("source") == "legacy":
        raise ConfigError("E_LEGACY_INPUTS_AS_VUS", "population.source='legacy' has Input kinds as VUs; use 'legacy-minus-inputs' (labelled) or 'a1'")
    if pop.get("source") not in POPULATION_SOURCES:
        raise ConfigError("E_CONFIG", f"population.source={pop.get('source')!r} not in {POPULATION_SOURCES}")
    cap = doc.get("capture") or {}
    fields = cap.get("fields", list(CAPTURE_FIELDS))
    extra = [f for f in fields if f not in CAPTURE_FIELDS]
    if extra:
        raise ConfigError("E_LEGACY_CAPTURE", f"capture.fields {extra} beyond {CAPTURE_FIELDS} (decision 1: serve-time capture is inputs/outputs/seed only)")
    for key in ("driver", "verifier", "prover"):
        v = doc.get(key)
        if isinstance(v, str) and any(v.startswith(l) or v == l for l in _LEGACY_DRIVERS):
            raise ConfigError("E_LEGACY_DRIVER", f"{key}={v!r} is a legacy module; the runner composes A3's verifier_v3 / selective_prover")
    inst = doc.get("instance") or {}
    if inst.get("schema") == "veritor/protocol-instance/v1" and "epoch" not in inst:
        raise ConfigError("E_LEGACY_INSTANCE", "instance without `epoch` (legacy protocol-instance/v1)")
    rnd = doc.get("randomness") or {}
    if rnd.get("mode", "production") not in RANDOMNESS_MODES:
        raise ConfigError("E_CONFIG", f"randomness.mode={rnd.get('mode')!r} not in {RANDOMNESS_MODES}")
    wl = doc.get("workload") or {}
    if wl.get("name") not in WORKLOADS:
        raise ConfigError("E_CONFIG", f"workload.name={wl.get('name')!r} not in {tuple(WORKLOADS)}")
    if wl.get("greedy", True) is not True or wl.get("prefix_caching", False) is not False or wl.get("batch_invariant", True) is not True:
        raise ConfigError("E_CONFIG", "workload must be greedy, prefix caching off, batch-invariant (DIRECTIVE §1.5.10)")
    ri = wl.get("request_indices")
    if ri is not None:
        nreq = WORKLOADS[wl["name"]][3]
        if not isinstance(ri, list) or len(ri) != nreq or any(not isinstance(i, int) or i < 0 for i in ri) or len(set(ri)) != len(ri):
            raise ConfigError("E_CONFIG", f"workload.request_indices={ri!r} must be {nreq} distinct non-negative B0-workload row indices (one per epoch request)")


def parse(doc: dict[str, Any], *, path: Path | None = None) -> RunConfig:
    guard(doc)
    wl = doc["workload"]
    model, plen, mtok, nreq = WORKLOADS[wl["name"]]
    ep = doc["epoch"]
    if ep["requests"] != nreq:
        raise ConfigError("E_CONFIG", f"epoch.requests={ep['requests']} != workload requests {nreq} (one epoch = the workload's requests)")
    smp = doc.get("sampling") or {}
    request_indices = tuple(int(i) for i in (wl.get("request_indices") or ()))
    return RunConfig(
        run_id=str(doc.get("run_id") or "V5"), workload=wl["name"], model=wl.get("model", model), prompt_len=int(wl.get("prompt_len", plen)),
        max_tokens=int(wl.get("max_tokens", mtok)), requests=nreq, epoch=Epoch(requests=nreq, B_ex=int(ep["B_ex"]), label=str(ep.get("label", "DEFAULT TAKEN"))),
        q=_frac(smp.get("q", "1/2"), "sampling.q"), s=_frac(smp.get("s", "1/16384"), "sampling.s"), delta=_frac(smp.get("delta", "1/1000000000000"), "sampling.delta"),
        randomness_mode=(doc.get("randomness") or {}).get("mode", "production"), f_policy=doc["f_policy"],
        population_source=doc["population"]["source"], plan=str(doc["population"].get("plan", "P1/cu=whole")),
        guest_approved_only=bool((doc.get("guest") or {}).get("approved_elf_only", True)),
        record_label=doc.get("record_label"), s_source=str(smp.get("s_source") or "UNAVAILABLE (run config without sampling.s_source)"), family=str((doc.get("guest") or {}).get("family") or FAMILY), approved_elf_sha256=tuple((doc.get("guest") or {}).get("approved_elf_sha256") or approved_elf_sha256()),
        request_indices=request_indices, path=path, raw=doc,
    )


def load(path: str | Path) -> RunConfig:
    p = Path(path)
    return parse(json.loads(p.read_text()), path=p)


DEFAULT_LABELS = {
    "TINY-B0-4x8": "TINY end-to-end dry run -- the first fresh integrated record from a fixed snapshot (USER DIRECTIVE 20:10Z §5); computed-only population + epoch.inputs pinned",
    "TINY-B0-4x8-1req": "TINY one-request epoch (EOD 2026-09-11 common brief: N_R = 1, q = 1 => |J| = 1; interior values from the reference interpreter (E1) under E2's pinned interior layout); "
                        "same program, operations and numerical semantics as attempt 6; new epoch identity",
    "SMOL360-SmolLM2-360M-4x8-1req": "[E1 NIGHT 2026-09-12 G3] same-architecture second checkpoint (SmolLM2-360M, Llama) one-request epoch at the TINY shape: Serve_v1 at b1.B0L_CONFIG; does it expose different operations or numerics?",
    "QWEN05-Qwen2.5-0.5B-4x8-1req": "[E1 NIGHT 2026-09-12 G3] different-architecture checkpoint (Qwen2.5-0.5B, Qwen2: QKV bias, eps 1e-6, vocab 151936) one-request epoch at the TINY shape: Serve_v1 at b1.B1S_CONFIG",
    "L1-32x15": "32x15 conformance-scale dry run",
    "SMALL-6500x1600": "as-built diagnostic -- coverage gap known (USER DIRECTIVE 20:10Z §4: exactly one; the component timings are the output)",
}


# s per workload -- DIRECTIVE §9.1 decision 2: s is determined by the expected work.  TINY / 32x15: L1's 2^-14 (out/record/expected_work.json,
# E|Q| = 20,288 at 32x15).  SMALL: D4's expected-work rule (out/record-6k/expected_work.json on lane/record-6k, `s_rule`): the largest 2^-k with
# expected priced proving work <= 4 GPU-h at the conservative price -> 2^-19 (E[GemmCoordinate] = 5,105 -> 3.47 GPU-h; at 2^-14 it is 111 GPU-h and
# A3's sampler draws |Q| = 18.8M at 6.5K -- measured on the CPU pod, 308 s stage 2).
DEFAULT_S = {"TINY-B0-4x8": ("1/16384", "L1's s = 2^-14 (out/record/expected_work.json), the conformance-scale value"),
             "TINY-B0-4x8-1req": ("1/16384", "L1's s = 2^-14 (out/record/expected_work.json), the conformance-scale value; E|Q| = 7,922,613 x 2^-14 = 483.6 VUs for the one retained RU"),
             "SMOL360-SmolLM2-360M-4x8-1req": ("1/16384", "L1's s = 2^-14, the conformance-scale value; E|Q| = N_V(SmolLM2-360M, E6's population) x 2^-14 for the one retained RU"),
             "QWEN05-Qwen2.5-0.5B-4x8-1req": ("1/16384", "L1's s = 2^-14, the conformance-scale value; E|Q| = N_V(Qwen2.5-0.5B, E6's population) x 2^-14 for the one retained RU"),
             "L1-32x15": ("1/16384", "L1's s = 2^-14 (out/record/expected_work.json: E|Q| = 20,288 at q = 1/2, 4 instances)"),
             "SMALL-6500x1600": ("1/524288", "D4's expected-work rule (lane/record-6k out/record-6k/expected_work.json#s_rule): largest 2^-k with expected priced proving work "
                                             "<= 4 GPU-h conservative -> 2^-19 (E[GemmCoordinate] = 5,105; E[unsupported selected] = 387,408 per E|J| = 2)")}


def default_doc(workload: str, *, B_ex: int, f_policy: str, mode: str = "production", s: str | None = None, q: str = "1/2", record_label: str | None = None) -> dict[str, Any]:
    """The run-config document for one (workload, B_ex, F policy) cell; both B_ex values are DEFAULT TAKEN for the integrator."""

    model, plen, mtok, nreq = WORKLOADS[workload]
    s_default, s_source = DEFAULT_S[workload]
    s_source = s_source if s is None else f"given by the caller ({s})"
    s = s or s_default
    short = {"TINY-B0-4x8": "TINY", "TINY-B0-4x8-1req": "TINY1", "SMOL360-SmolLM2-360M-4x8-1req": "SMOL360-4x9", "QWEN05-Qwen2.5-0.5B-4x8-1req": "QWEN05-4x9",
             "L1-32x15": "32x15", "SMALL-6500x1600": "6500x1600"}[workload]
    fp = {"exempt-on-inconsistency": "inc", "exempt-unsupported": "uns"}[f_policy]
    return {
        "schema": SCHEMA, "run_id": f"V5-{short}-bex{B_ex}-{fp}",
        "workload": {"name": workload, "model": model, "prompt_len": plen, "max_tokens": mtok, "requests": nreq, "greedy": True, "prefix_caching": False, "batch_invariant": True},
        "epoch": {"requests": nreq, "B_ex": B_ex, "B_ex_unit": "RU", "label": "DEFAULT TAKEN"},
        "sampling": {"q": q, "s": s, "delta": "1/1000000000000", "law": "bernoulli-per-unit", "s_source": s_source},
        "randomness": {"mode": mode}, "f_policy": f_policy, "capture": {"fields": list(CAPTURE_FIELDS)},
        "population": {"source": "legacy-minus-inputs", "plan": "P1/cu=whole"}, "record_label": record_label or DEFAULT_LABELS[workload],
        "guest": {"approved_elf_only": True, "family": FAMILY, "approved_elf_sha256": approved_elf_sha256()},
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="run configs: `check FILE` runs the legacy guard; `gen DIR` writes the default cells")
    ap.add_argument("cmd", choices=("check", "gen"))
    ap.add_argument("path")
    a = ap.parse_args(argv)
    if a.cmd == "gen":
        d = Path(a.path)
        d.mkdir(parents=True, exist_ok=True)
        for wl in WORKLOADS:
            for bex in (0, 1):
                for fp in F_POLICIES:
                    doc = default_doc(wl, B_ex=bex, f_policy=fp)
                    (d / f"{doc['run_id']}.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(f"[config] wrote {len(WORKLOADS) * 4} configs to {d}")
        return 0
    try:
        c = load(a.path)
        print(f"[config] OK {c.run_id}: {c.workload} epoch={c.epoch.requests} req, B_ex={c.epoch.B_ex} RU ({c.epoch.label}), q={c.q} s={c.s} F={c.f_policy} randomness={c.randomness_mode}")
        return 0
    except ConfigError as e:
        print(f"[config] REFUSED {a.path}: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "F_POLICIES", "RANDOMNESS_MODES", "CAPTURE_FIELDS", "POPULATION_SOURCES", "WORKLOADS", "ConfigError", "Epoch", "RunConfig", "guard", "parse", "load"]
