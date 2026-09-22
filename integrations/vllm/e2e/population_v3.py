#!/usr/bin/env python3
"""[A1 CIRCUIT 2026-09-10, contract R2 / R11] The **computed-only** population of ``P1/cu=whole/inputs=prescribed`` --
input gates in no VU and no RU -- as a census per SMALL shape, and the tools that establish it:

* ``rules``   -- the rule-generated kind table at any (LP, STEPS) of ``Serve_v1{C=B0}`` (MODELED): L5's exact structural rules
  (``data/census/census_small_6500x1600.json#rules``, read as ``record_6k/rules6k.py`` reads them -- same source file, same refusals;
  not imported), every ``Input`` kind taken out and reported as the prescribed input tuple.  Holdout: the 32x15 computed counts
  are re-asserted against the validated artifact before any use.
* ``widths``  -- the exhaustive per-kind interface widths of a plan (MEASURED): every VU's boundary (``veritor.plans.boundary
  .boundary_of``, the validator's) folded into a per-kind histogram of ``(|S.out|, out bits)``; declared width beside it
  (``population_v2._callee_out_bits``: the certificate's kind table).  TINY on the laptop; 32x15 on a CPU pod (forked workers).
* ``census``  -- ``veritor/a1/population-computed/v1``: per kind count, offset, declared width, actual widths (min / max /
  histogram), the partition and plan digests, the prescribed inputs, one label per row -- MEASURED (validated unit by unit:
  ``compact_p1_validate.py --inputs prescribed`` record + ``widths`` record) or MODELED (rule) -- and INCONSISTENT with the cause
  wherever a measured count disagrees with the rule.  Consumers: :func:`rule_view_kinds` (A9's ``RuleView`` shape),
  :func:`certificate_families` (A7's ``certificate_v2.population_from_families(source="a1")`` arguments).

Digests.  A MEASURED census carries the plan's ``partition-intervals/v1`` digest (never the with-inputs digest ``56ebcdf5...`` /
``3646488c...``).  A MODELED census carries the identity digest of its kind table + prescribed-inputs block
(``veritor/a1/rule-population-computed-kind-table/v1``); the two input treatments of one shape never share an identity.

    PYTHONPATH=. .venv/bin/python e2e/population_v3.py rules --lp 6500 --steps 1599 --out data/contract/a1-circuit/pop-rules-computed-6500x1599.json
    PYTHONPATH=. .venv/bin/python e2e/population_v3.py widths --case TINY --out /tmp/widths-tiny.json
    PYTHONPATH=. .venv/bin/python e2e/population_v3.py widths --case B0 --lp 32 --steps 15 --workers 16 --out ...   # pod
    PYTHONPATH=. .venv/bin/python e2e/population_v3.py census --lp 32 --steps 15 --validation <compact_p1_validate .json> \\
        --widths <widths .json> --out data/contract/a1-circuit/population-computed-32x15.json
    PYTHONPATH=. .venv/bin/python e2e/population_v3.py census --lp 6500 --steps 1599 --widths-from data/contract/a1-circuit/population-computed-32x15.json \\
        --out data/contract/a1-circuit/population-computed-6500x1599.json
"""
from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import multiprocessing as mp
import os
import platform
import re
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(ROOT), str(POC), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

RULES_SCHEMA = "veritor/a1/population-rules-computed/v1"
WIDTHS_SCHEMA = "veritor/a1/kind-widths-measured/v1"
CENSUS_SCHEMA = "veritor/a1/population-computed/v1"
KIND_TABLE_DIGEST = "veritor/a1/rule-population-computed-kind-table/v1"
PROGRAM_DIGEST_KIND = "rec6k/program-identity/v1 (identity digest of the program's name; the program was not built at this shape)"
PLAN_NAME = "P1/cu=whole/inputs=prescribed"
CONSTRUCTION = "input gates prescribed (in no VU and no RU); the population is the computed gates of P1/cu=whole"
CENSUS_6500 = ROOT / "data" / "census" / "census_small_6500x1600.json"
CENSUS_6500_SHA256_PREFIX = "133934a8"  # the census L5 published (D4 asserts the same prefix)
VALIDATED_32_15 = ROOT / "data" / "rec" / "p1-b0-lp32s15-validated.json"
B0_CONFIG_SHA12 = "6f196795e564"
M_REQUESTS = 130_000_000_000  # the directive's M (census convention for A_paper)
LABELS = ("MEASURED", "MODELED", "UNAVAILABLE", "INCONSISTENT")

# [DIRECTIVE Amendment A2, decision 4] token selection is ONE VU per decode step whose output is the 18-bit index unit (SMALL: V = 49,152,
# the index fits in 16 bits; the I32 container is 32); the ArgmaxStep scan-step units (80-bit carries) leave the partition.  The kind is
# named as D2's P1v2 names it so A7's widths and CK-ELEM's builder refer to the same unit.
ARGMAX_KIND = "ArgmaxStep_v1|scan"
TOKEN_SELECT_KIND = "TokenSelect_v1|whole-scan"
TOKEN_SELECT_FORMS = ("whole-scan", "argmax-scan")  # ruled form (default) | the P1 form the validated plans carry
TOKEN_SELECT_INDEX_BITS = 18  # the §1.5.9 index unit as declared
TOKEN_SELECT_CONTAINER_BITS = 32  # the I32 gate ``ni`` the validator sees as the unit's S.out
SMALL_VOCAB = 49_152
TOKEN_SELECT_DOMAIN_BITS = (SMALL_VOCAB - 1).bit_length()  # ceil(log2 V) = 16
TOKEN_SELECT_RULE = ("Amendment A2 decision 4: ArgmaxStep -> one VU per step (the whole scan of a decode step = one unit), 18-bit index; "
                     "the scan-step units' 80-bit carries are interior to the unit")

_SUBFAMILY = re.compile(r"^(?P<base>.+?)\|(?:out=\d+|prompt|weights)(?:\|out=\d+)?$")


def _digest(doc: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# -- rules (MODELED) -------------------------------------------------------------------------------------------------------------


def program_identity_digest(lp: int, steps: int, config_sha12: str = B0_CONFIG_SHA12, family: str = "Serve_v1", attention: dict[str, Any] | None = None) -> str:
    """D4's identity digest of the program's *name* (the same program with either input treatment).  [E6 CURVE, D57] a declared attention
    geometry is part of the name; ``None`` (legacy) leaves every pre-D57 digest unchanged."""

    doc: dict[str, Any] = {"schema": "veritor/rec6k/program-identity/v1", "family": family, "config_sha12": config_sha12, "LP": lp, "STEPS": steps}
    if attention is not None:  # the geometry statics (E7's Serve_v4 binds BN = kBlockN; kernel / kBlockM / num_splits are the model's admission conditions)
        doc["attention"] = {"kernel": attention["kernel"], "kBlockM": int(attention["kBlockM"]), "kBlockN": int(attention["kBlockN"]), "num_splits": int(attention.get("num_splits", 1))}
    return _digest(doc)


def identity_function(pin: dict[str, Any], lp: int, steps: int) -> str:
    """The pinned identity's bound id: the registry's own (``pin["function"]``) when the pin carries it -- required for a geometry pin, whose id
    E7's static may extend -- else the legacy composition ``Serve_v1{C={sha12},LP,STEPS}``."""

    if pin.get("function"):
        return str(pin["function"])
    if pin.get("attention"):  # the registry's bound id through E7's hook (a named refusal when the tree has no geometry static)
        from record_v5 import instance_v5 as i5

        return str(i5.program_identity(pin["case"], lp, steps, pin["attention"])["function"])
    return f"{pin['family']}{{C={{{pin['config_sha12']}}},LP={lp},STEPS={steps}}}"


def load_rules(path: Path = CENSUS_6500) -> tuple[dict[str, dict[str, Fraction]], str]:
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if not sha.startswith(CENSUS_6500_SHA256_PREFIX):
        raise SystemExit(f"REFUSED: {path} sha256 {sha[:16]} does not start with {CENSUS_6500_SHA256_PREFIX}")
    doc = json.loads(raw)
    if doc.get("unavailable_kinds") or doc.get("fit_failures"):
        raise SystemExit(f"REFUSED: census rules carry unavailable kinds / fit failures: {doc.get('unavailable_kinds')} {doc.get('fit_failures')}")
    rules: dict[str, dict[str, Fraction]] = {}
    for kind, r in doc["rules"].items():
        if r.get("fit") != "OK" or not r.get("design_full_rank") or not r.get("holdout", {}).get("equal"):
            raise SystemExit(f"REFUSED: rule for {kind} is not an exact full-rank fit reproducing the holdout: {r}")
        rules[kind] = {t: Fraction(c) for t, c in r["coefficients"].items()}
    return rules, sha


def evaluate(rules: dict[str, dict[str, Fraction]], lp: int, steps: int, bn: int | None = None) -> dict[str, int]:
    """Every rule (sub-families included) at (LP, STEPS): exact non-negative integers, else refused.  [E6 CURVE, D57] ``bn`` = the pinned
    identity's FA2 kBlockN (``None`` = the legacy 64 the rules were fitted on); the basis terms follow the block, the coefficients do not."""

    from census_v2.small_scaling import TERMS, terms  # L5's basis, read-only

    basis = dict(zip(TERMS, terms(lp, steps, bn), strict=True))
    out: dict[str, int] = {}
    for kind, coef in rules.items():
        v = sum(coef.get(t, Fraction(0)) * basis[t] for t in TERMS)
        if v.denominator != 1 or v < 0:
            raise SystemExit(f"REFUSED: rule {kind} at ({lp},{steps}) gives {v} (not a non-negative integer)")
        out[kind] = int(v)
    return out


def population_kinds(rules: dict[str, dict[str, Fraction]]) -> list[str]:
    """The population kind keys (``fn|form``) of the computed gates: every rule key that is not a per-width / per-source sub-family,
    ``Input`` excluded."""

    kinds = set()
    for k in rules:
        m = _SUBFAMILY.match(k)
        base = m.group("base") if m else k
        if base != "Input|batch":
            kinds.add(base)
    return sorted(kinds)


def regroup_token_selection(counts: dict[str, int], nout: int, *, source: str, vocab: int = SMALL_VOCAB) -> dict[str, Any]:
    """Apply Amendment A2 decision 4 to a per-kind count table **in place**: the ``ArgmaxStep_v1|scan`` scan-step units (``V - 1`` per
    decode step: 49,151 at SMALL) become ``NOUT`` ``TokenSelect_v1|whole-scan`` units.  Refuses unless the scan-step count is exactly
    ``NOUT * (V - 1)`` for the program's vocabulary ``vocab`` (another checkpoint's V changes the whole-scan width, never the unit count).
    Returns the ``token_selection`` record (what was regrouped, by which rule, from which source)."""

    n = counts.pop(ARGMAX_KIND, None)
    if n is None:
        raise SystemExit(f"REFUSED: no {ARGMAX_KIND} count to regroup ({source})")
    if nout < 1 or n % nout:
        raise SystemExit(f"REFUSED: {ARGMAX_KIND} = {n} scan-step units is not a multiple of NOUT = {nout} ({source})")
    if n // nout != vocab - 1:
        raise SystemExit(f"REFUSED: {ARGMAX_KIND} = {n} scan-step units / NOUT {nout} = {n // nout} != V - 1 = {vocab - 1} ({source})")
    if (vocab - 1).bit_length() > TOKEN_SELECT_INDEX_BITS:
        raise SystemExit(f"REFUSED: vocabulary {vocab} needs {(vocab - 1).bit_length()} index bits > the declared {TOKEN_SELECT_INDEX_BITS} ({source})")
    if TOKEN_SELECT_KIND in counts:
        raise SystemExit(f"REFUSED: {TOKEN_SELECT_KIND} already present ({source})")
    counts[TOKEN_SELECT_KIND] = nout
    return {"form": "whole-scan", "kind": TOKEN_SELECT_KIND, "units": nout, "replaces": {"kind": ARGMAX_KIND, "units": n, "scan_steps_per_unit": n // nout},
            "rule": TOKEN_SELECT_RULE, "index_bits_declared": TOKEN_SELECT_INDEX_BITS, "domain_bits_small": TOKEN_SELECT_DOMAIN_BITS,
            "container_bits": TOKEN_SELECT_CONTAINER_BITS, "vocab_small": SMALL_VOCAB, "vocab": vocab, "domain_bits": (vocab - 1).bit_length(), "source": source,
            "note": "D2's P1v2 regroups the same scan-step units under the same kind name (D2X report §3); the regrouped unit's S.out is its last "
                    "step's S.out (the I32 gate ni)"}


def computed_kind_counts(rules: dict[str, dict[str, Fraction]], lp: int, steps: int, *, token_select: str = "whole-scan", bn: int | None = None
                         ) -> tuple[dict[str, int], dict[str, Any], dict[str, int]]:
    """``(computed kind -> count per request, prescribed-inputs block, every evaluated rule)`` at (LP, STEPS).  Sub-families collapse
    onto their base kind and must sum to it (``ArgmaxStep_v1|scan`` = ``out=32`` + ``out=80``).  ``token_select="whole-scan"`` (default,
    Amendment A2 decision 4) regroups the scan-step units into NOUT ``TokenSelect_v1|whole-scan`` units; ``"argmax-scan"`` keeps the P1 form
    (the form the validated 32x15 / TINY plans carry)."""

    if token_select not in TOKEN_SELECT_FORMS:
        raise SystemExit(f"REFUSED: token_select {token_select!r} not in {TOKEN_SELECT_FORMS}")
    ev = evaluate(rules, lp, steps, bn)
    counts: dict[str, int] = {}
    for kind in population_kinds(rules):
        if kind not in ev:
            raise SystemExit(f"REFUSED: kind {kind} has no base rule")
        counts[kind] = ev[kind]
        subs = [k for k in ev if k != kind and _SUBFAMILY.match(k) and _SUBFAMILY.match(k).group("base") == kind]
        if subs and sum(ev[s] for s in subs) != ev[kind]:
            raise SystemExit(f"REFUSED: sub-families of {kind} sum to {sum(ev[s] for s in subs)} != base rule {ev[kind]} at ({lp},{steps})")
    token_selection = (regroup_token_selection(counts, steps + 1, source=f"rules at ({lp},{steps})") if token_select == "whole-scan"
                       else {"form": "argmax-scan", "kind": ARGMAX_KIND, "units": counts[ARGMAX_KIND],
                             "note": "P1 form: one VU per scan step (80-bit carries); NOT the ruled form (Amendment A2 decision 4)"})
    prompt, weights = ev["Input|batch|prompt"], ev["Input|batch|weights"]
    w0, w16 = ev["Input|batch|weights|out=0"], ev["Input|batch|weights|out=16"]
    if w0 + w16 != weights:
        raise SystemExit(f"REFUSED: Input weights per-width sub-families {w0 + w16} != weights rule {weights}")
    if prompt != lp:
        raise SystemExit(f"REFUSED: Input|batch|prompt rule gives {prompt} != LP {lp}")
    prescribed = {
        "gates": prompt + weights,
        "prompt_token_ids": {"gates": prompt, "container_bits": 32, "domain_bits": 16, "domain": "vocabulary index, V = 49,152 (SMALL)",
                             "rule": "Input|batch|prompt = LP"},
        "weights": {"gates": weights, "bits": 16, "consumed": w16, "unconsumed_rope_rows": w0,
                    "rule": "Input|batch|weights = Input|batch|weights|out=16 (consumed, bf16) + Input|batch|weights|out=0 (rope cos/sin rows at positions >= P: read by nothing)"},
        "seed": {"gates": 0, "note": "greedy decoding: TokenSelect is deterministic, no seed gate exists in Serve_v1"},
        "treatment": "prescribed: in no VU and no RU; values checked against the declared input unconditionally (contract step 2), never sampled",
    }
    prescribed["token_selection"] = token_selection  # the form of the token-selection unit is part of the population's identity
    return counts, prescribed, ev


def rule_widths(ev: dict[str, int], token_selection: dict[str, Any] | None = None) -> dict[str, list[list[int]]]:
    """Per-width sub-family counts the rules carry, per computed kind: ``{kind: [[out_bits, units], ...]}`` (only ``ArgmaxStep_v1|scan``
    at SMALL: ``out=32`` / ``out=80``).  Every other kind's per-width split is not a rule.  Under the whole-scan form the ArgmaxStep split
    is replaced by ``TokenSelect_v1|whole-scan: [[32, NOUT]]`` (its S.out is the last step's I32 ``ni``)."""

    out: dict[str, dict[int, int]] = {}
    for k, n in ev.items():
        m = re.match(r"^(?P<base>.+?)\|out=(?P<bits>\d+)$", k)
        if m and not m.group("base").startswith("Input"):
            out.setdefault(m.group("base"), {})[int(m.group("bits"))] = n
    res = {k: [[b, n] for b, n in sorted(v.items())] for k, v in out.items()}
    if token_selection and token_selection.get("form") == "whole-scan":
        res.pop(ARGMAX_KIND, None)
        res[TOKEN_SELECT_KIND] = [[TOKEN_SELECT_CONTAINER_BITS, token_selection["units"]]]
    return res


def check_holdout(rules: dict[str, dict[str, Fraction]], validated: Path = VALIDATED_32_15) -> dict[str, Any]:
    """Re-assert every computed kind count at (32, 15) against the validated artifact (exact; the P1 ``argmax-scan`` form, which is what
    the validated plan carries) and the input count against its ``input_units``."""

    val = json.loads(validated.read_text())
    counts, prescribed, _ = computed_kind_counts(rules, 32, 15, token_select="argmax-scan")
    exact = {k: int(v) for k, v in val["unit_counts"]["per_kind_fn_form"].items() if not k.startswith("Input")}
    missing = sorted(set(exact) - set(counts))
    extra = {k: v for k, v in counts.items() if k not in exact and v != 0}
    mism = {k: (counts.get(k), exact[k]) for k in exact if counts.get(k) != exact[k]}
    if missing or extra or mism:
        raise SystemExit(f"REFUSED: rules do not reproduce the validated 32x15 computed counts: missing {missing}, extra nonzero {extra}, mismatched {mism}")
    if prescribed["gates"] != int(val["unit_counts"]["input_units"]):
        raise SystemExit(f"REFUSED: prescribed inputs {prescribed['gates']} != validated input_units {val['unit_counts']['input_units']}")
    return {"validated": str(validated.relative_to(ROOT)), "kinds_checked": len(exact), "computed_total": sum(exact.values()),
            "input_units": prescribed["gates"], "equal": True, "partition_digest_32x15_with_inputs": val["plan"]["partition_digest"],
            "kinds_zero_at_32x15_new_at_shape": sorted(k for k, v in counts.items() if v == 0)}


def kind_table_digest(counts: dict[str, int], prescribed: dict[str, Any], lp: int, steps: int, *, literals: str = "retained") -> str:
    """The identity digest of the rule-generated computed-only population: sha256 over the canonical JSON of the kind table AND the
    prescribed-inputs block (the input treatment is part of the identity).  Not ``veritor.plans.summary.partition_digest`` (no gate
    intervals exist here) and never the with-inputs digest.  [E6 2026-09-11] ``literals="compiled"`` enters the digest document (the
    literal treatment is part of the identity); ``retained`` leaves the document -- and every digest written before today -- unchanged."""

    doc = {"schema": KIND_TABLE_DIGEST, "LP": lp, "STEPS": steps, "P": lp + steps, "NOUT": steps + 1,
           "kinds": {k: counts[k] for k in sorted(counts)},
           "prescribed_inputs": {"prompt": prescribed["prompt_token_ids"]["gates"], "weights": prescribed["weights"]["gates"],
                                 "weights_unconsumed": prescribed["weights"]["unconsumed_rope_rows"]}}
    if literals != "retained":
        doc["literals"] = literals
    return _digest(doc)


def a_paper(max_cut: int, v_r: int, v_global: int = 0, m: int = M_REQUESTS) -> int:
    """The census's A_paper form: ``max_cut + ceil(log2(4 * (V_global + M * V_r)))`` (ANALYTICAL); with the inputs prescribed
    ``V_global = 0``."""

    import math

    return max_cut + math.ceil(math.log2(4 * (v_global + m * v_r)))


def build_rules(lp: int, steps: int, *, token_select: str = "whole-scan") -> dict[str, Any]:
    """The rule-generated computed-only population at (LP, STEPS) (MODELED).  ``token_select``: ``"whole-scan"`` (Amendment A2
    decision 4, default) or ``"argmax-scan"`` (the P1 form of the validated plans; what ``pop-rules-computed-6500x1599.json`` carries)."""

    rules, sha = load_rules()
    holdout = check_holdout(rules)
    counts, prescribed, ev = computed_kind_counts(rules, lp, steps, token_select=token_select)
    token_selection = prescribed.pop("token_selection")
    kinds = sorted(counts)
    offsets, acc = {}, 0
    for k in kinds:
        offsets[k] = acc
        acc += counts[k]
    ktd = kind_table_digest(counts, prescribed, lp, steps)
    with_inputs = acc + prescribed["gates"]
    rw = rule_widths(ev, token_selection)
    argmax80 = ev.get("ArgmaxStep_v1|scan|out=80", 0) if token_select == "argmax-scan" else 0
    max_cut = 80 if argmax80 else 32
    ts_width = ({"kind": TOKEN_SELECT_KIND, "declared_index_bits": TOKEN_SELECT_INDEX_BITS, "domain_bits_small": TOKEN_SELECT_DOMAIN_BITS,
                 "container_bits": TOKEN_SELECT_CONTAINER_BITS, "actual_bits": {str(TOKEN_SELECT_CONTAINER_BITS): steps + 1},
                 "note": "declared 18 (index unit, §1.5.9) printed beside the actual container 32 and the SMALL domain 16 -- never clamped"}
                if token_select == "whole-scan" else
                {"kind": ARGMAX_KIND, "declared_bits": 112, "actual_bits": {"80": argmax80, "32": ev.get("ArgmaxStep_v1|scan|out=32", 0)}})
    doc = {
        "schema": RULES_SCHEMA, "construction": "rule-generated (MODELED), not validated at this shape; " + CONSTRUCTION
                  + f"; token selection {token_selection['form']}",
        "instances": 1, "LP": lp, "STEPS": steps, "P": lp + steps, "NOUT": steps + 1,
        "vu_count": acc, "ru_count": 1,
        "ru_rule": "one RU per request (P1/cu=whole, computed gates only; in(R) = every consumed input gate, out(R) = the NOUT token-id gates); "
                   "VU rank = kind offset + index within kind, kinds in sorted key order",
        "kinds": [{"kind": k, "fn": k.split("|")[0], "form": k.split("|", 1)[1] if "|" in k else "", "count": counts[k], "offset": offsets[k]} for k in kinds],
        "rule_widths": rw,
        "token_selection": token_selection,
        "prescribed_inputs": prescribed,
        "kind_table_digest": ktd,
        "program_digest": program_identity_digest(lp, steps), "program_digest_kind": PROGRAM_DIGEST_KIND,
        "plan_name": PLAN_NAME,
        "plan_digest": _digest({"schema": "veritor/a1/plan-rule-modeled/v1", "plan": PLAN_NAME, "kind_table_digest": ktd}),
        "plan_digest_kind_note": "a1/plan-rule-modeled/v1 (P1/cu=whole/inputs=prescribed applied to the rule-generated kind table; not veritor.plans.summary.plan_digest)",
        "partition_digest": ktd, "partition_digest_kind": KIND_TABLE_DIGEST,
        "with_inputs_for_comparison": {"vu_count": with_inputs, "input_units": prescribed["gates"], "ratio_with_over_without": round(with_inputs / acc, 6),
                                       "note": "D4's population-rules/v1 at the same shape counts these input units as VUs (per-RU-copy formulation)"},
        "widths": {"max_cut_bits": max_cut, "max_cut_bits_excluding_ArgmaxStep": 32,
                   "token_selection": ts_width,
                   "A_paper": a_paper(max_cut, acc), "A_paper_excluding_ArgmaxStep": a_paper(32, acc),
                   "A_paper_with_inputs_D4_convention": a_paper(max_cut, acc, prescribed["weights"]["gates"]),
                   "A_paper_form": "max_cut + ceil(log2(4 * (V_global + M * V_r))), M = 1.3e11; V_global = 0 with the inputs prescribed (ANALYTICAL)"},
        "rules": {"source": str(CENSUS_6500.relative_to(ROOT)), "sha256": sha, "basis": "census_v2.small_scaling.terms(LP, STEPS)", "holdout_32x15": holdout},
        "label": "MODELED (counts by exact structural rules fitted by L5 on 26 census points + validated 32x15 holdout; no plan exists at this shape); "
                 "digests are identity digests of this document, not partition-intervals/v1",
    }
    return doc


# -- widths (MEASURED, exhaustive) -----------------------------------------------------------------------------------------------


def kind_key(node, run) -> str:
    """The census key of a run's kind: ``callee|form`` (``batch`` for an adapter-unrolled batch) -- ``compact_p1_validate.kind_counts``."""

    return f"{node.fn.name}|{'batch' if run.unrolled else node.form}"


_SHARED: dict[str, Any] = {}


class WidthCensus:
    """Per-kind histogram of unit interfaces: ``add(kind, declared_bits, boundary)`` per verification unit; ``to_json()`` is one
    part (mergeable with :func:`merge_widths`).  Used by the standalone ``widths`` pass and, through ``validate_plan(on_boundary=...)``,
    by ``compact_p1_validate.py --widths`` so one streamed validation yields the verdict AND the exhaustive widths."""

    __slots__ = ("kinds", "units", "_kind_of")

    def __init__(self, plan=None, circuit=None) -> None:
        self.kinds: dict[str, dict[str, Any]] = {}
        self.units = 0
        self._kind_of = None
        if plan is not None:
            from e2e.population_v2 import _callee_out_bits

            runs = plan.verification_units.runs
            cache: dict[int, tuple[str, int]] = {}

            def kind_of(u: int) -> tuple[str, int]:
                r, _ = plan.verification_units.locate(u)
                hit = cache.get(r)
                if hit is None:
                    node = circuit.scope_at(runs[r].path).fn.nodes[runs[r].node_idx]
                    hit = cache[r] = (kind_key(node, runs[r]), _callee_out_bits(node.fn))
                return hit

            self._kind_of = kind_of

    def on_boundary(self, side: str, u: int, unit, boundary) -> None:
        """The ``validate_plan(on_boundary=...)`` observer (verification side only)."""

        if side != "verification":
            return
        key, declared = self._kind_of(u)
        self.add(key, declared, boundary)

    def add(self, key: str, declared_bits: int, b) -> None:
        a = self.kinds.get(key)
        if a is None:
            a = self.kinds[key] = {"count": 0, "declared_out_bits": declared_bits, "widths": Counter(), "in_bits_min": None, "in_bits_max": 0,
                                   "in_bits_total": 0, "out_bits_total": 0}
        a["count"] += 1
        a["widths"][(len(b.outputs), b.out_bits)] += 1
        ib = b.in_bits
        a["in_bits_total"] += ib
        a["out_bits_total"] += b.out_bits
        a["in_bits_min"] = ib if a["in_bits_min"] is None else min(a["in_bits_min"], ib)
        a["in_bits_max"] = max(a["in_bits_max"], ib)
        self.units += 1

    def to_json(self) -> dict[str, Any]:
        return {"units": self.units, "kinds": {k: {**a, "widths": [[sc, bits, n] for (sc, bits), n in sorted(a["widths"].items())]} for k, a in self.kinds.items()}}


def _widths_range(args: tuple[int, int, int]) -> dict[str, Any]:
    """Per-kind width histogram of the runs ``[r_lo, r_hi)`` of the shared plan (one forked worker)."""

    from e2e.population_v2 import _callee_out_bits
    from veritor.plans.boundary import boundary_of

    w, r_lo, r_hi = args
    circuit, plan = _SHARED["circuit"], _SHARED["plan"]
    runs = plan.verification_units.runs
    wc = WidthCensus()
    steps = 0
    t = time.perf_counter()
    for r in range(r_lo, r_hi):
        run = runs[r]
        node = circuit.scope_at(run.path).fn.nodes[run.node_idx]
        key, declared = kind_key(node, run), _callee_out_bits(node.fn)
        for j in range(run.count):
            b, esc = boundary_of(run.unit(circuit, j))
            steps += esc
            wc.add(key, declared, b)
    return {"worker": w, "runs": [r_lo, r_hi], "escape_steps": steps, "seconds": round(time.perf_counter() - t, 3), **wc.to_json()}


def merge_widths(parts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Exact merge of :class:`WidthCensus` parts (``{"kinds": {...}}`` each) into the per-kind widths table."""

    merged: dict[str, dict[str, Any]] = {}
    for p in parts:
        for key, a in p["kinds"].items():
            m = merged.get(key)
            if m is None:
                m = merged[key] = {"count": 0, "declared_out_bits": a["declared_out_bits"], "widths": Counter(), "in_bits_min": None, "in_bits_max": 0,
                                   "in_bits_total": 0, "out_bits_total": 0}
            if m["declared_out_bits"] != a["declared_out_bits"]:
                raise SystemExit(f"REFUSED: kind {key} has two declared widths {m['declared_out_bits']} / {a['declared_out_bits']}")
            m["count"] += a["count"]
            for sc, bits, n in a["widths"]:
                m["widths"][(sc, bits)] += n
            m["in_bits_total"] += a["in_bits_total"]
            m["out_bits_total"] += a["out_bits_total"]
            m["in_bits_min"] = a["in_bits_min"] if m["in_bits_min"] is None else min(m["in_bits_min"], a["in_bits_min"])
            m["in_bits_max"] = max(m["in_bits_max"], a["in_bits_max"])
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(merged):
        m = merged[key]
        hist = [[sc, bits, n] for (sc, bits), n in sorted(m["widths"].items())]
        bits_seen = [bits for _, bits, _ in hist]
        out[key] = {"count": m["count"], "declared_out_bits": m["declared_out_bits"],
                    "actual_out_bits": {"min": min(bits_seen), "max": max(bits_seen), "histogram": hist,
                                        "histogram_fields": ["out_scalars", "out_bits", "units"]},
                    "declared_equals_actual": bits_seen == [m["declared_out_bits"]],
                    "in_bits": {"min": m["in_bits_min"], "max": m["in_bits_max"], "total": m["in_bits_total"]},
                    "out_bits_total": m["out_bits_total"]}
    return out


def measured_widths(plan, circuit, *, workers: int = 1, chunks: int | None = None, log=None) -> dict[str, Any]:
    """Exhaustive per-kind interface widths of ``plan`` (a ``UnitRuns`` verification side): every unit's boundary computed with the
    validator's :func:`boundary_of`.  ``workers > 1`` forks (the circuit and plan are shared copy-on-write, as in
    ``compact_p1_validate.sharded_validate``); the merge is exact (a histogram sum)."""

    from veritor.plans.runs import UnitRuns

    vus = plan.verification_units
    if not isinstance(vus, UnitRuns):
        raise SystemExit("measured_widths needs a compact (UnitRuns) verification side")
    n_runs = len(vus.runs)
    chunks = chunks or max(1, workers * 4)
    # cut the runs into ``chunks`` ranges of about equal unit count (relation units cost about the same; ragged scans are few)
    total = len(vus)
    bounds = [0]
    acc = 0
    for r, run in enumerate(vus.runs):
        acc += run.count
        if len(bounds) < chunks and acc >= len(bounds) * total / chunks:
            bounds.append(r + 1)
    while len(bounds) < chunks + 1:
        bounds.append(n_runs)
    bounds[-1] = n_runs
    tasks = [(i, bounds[i], bounds[i + 1]) for i in range(chunks) if bounds[i] < bounds[i + 1]]
    _SHARED["circuit"], _SHARED["plan"] = circuit, plan
    t = time.perf_counter()
    parts: list[dict[str, Any]] = []
    if workers <= 1:
        for task in tasks:
            parts.append(_widths_range(task))
            if log:
                log(f"    part {task[0]} runs [{task[1]}, {task[2]}) {parts[-1]['units']} units in {parts[-1]['seconds']:.0f}s")
    else:
        gc.collect()
        gc.freeze()
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            for part in pool.imap_unordered(_widths_range, tasks, chunksize=1):
                parts.append(part)
                if log:
                    log(f"    part {part['worker']} runs {part['runs']} {part['units']} units in {part['seconds']:.0f}s  ({len(parts)}/{len(tasks)} done, {time.perf_counter() - t:.0f}s)")
        gc.unfreeze()
    kinds = merge_widths(parts)
    units = sum(p["units"] for p in parts)
    if units != total or sum(k["count"] for k in kinds.values()) != total:
        raise SystemExit(f"REFUSED: widths pass covered {units} of {total} verification units")
    return {"kinds": kinds, "units": total, "runs": n_runs, "escape_steps": sum(p["escape_steps"] for p in parts), "workers": workers, "chunks": len(tasks),
            "seconds": round(time.perf_counter() - t, 3), "worker_seconds_sum": round(sum(p["seconds"] for p in parts), 3)}


def build_widths(case: str, lp: int, steps: int, *, workers: int, chunks: int | None, inputs: str = "prescribed", log=print) -> dict[str, Any]:
    """Build ``Serve_v1{C=case, LP, STEPS}`` and its ``P1/cu=whole`` plan with ``inputs`` and measure every kind's widths."""

    import resource

    from e2e import population as popmod
    from veritor.plans.canonical import PRESCRIBED_SUFFIX
    from veritor.plans.summary import PARTITION_DIGEST, partition_digest

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
    if case == "TINY":
        config, lp, steps = dict(popmod.TINY), popmod.TINY_LP, popmod.TINY_STEPS
    else:
        from verity_vllm.registry import b1

        config = dict(getattr(b1, f"{case}_CONFIG"))
    t0 = time.perf_counter()
    program, adapted, circuit, pdigest = popmod.build(config, lp, steps)
    spec = "P1/cu=whole" + (PRESCRIBED_SUFFIX if inputs == "prescribed" else "")
    plan, meta = popmod.make_plan(spec, adapted, circuit)
    t1 = time.perf_counter()
    log(f"  program {program.gates} gates ({program.input_gates} input), plan {plan.name}: {len(plan.verification_units)} VUs in {len(plan.verification_units.runs)} runs ({t1 - t0:.1f}s)")
    pdig = partition_digest(plan)
    res = measured_widths(plan, circuit, workers=workers, chunks=chunks, log=log)
    rss = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    return {"schema": WIDTHS_SCHEMA, "label": "MEASURED (exhaustive: every verification unit's boundary, veritor.plans.boundary.boundary_of)",
            "utc": _utc(), "host": platform.node(), "platform": platform.platform(), "cpus": os.cpu_count(), "maxrss_mb": rss,
            "case": case, "LP": lp, "STEPS": steps, "inputs": inputs, "plan_name": plan.name,
            "program": {"name": adapted.program.name, "digest": pdigest, "gates": program.gates, "input_gates": program.input_gates},
            "partition_digest": pdig, "partition_digest_kind": PARTITION_DIGEST,
            "prescribed_outside": meta.get("prescribed_outside"), "prescribed_outside_sides": meta.get("prescribed_outside_sides"),
            "vu_count": len(plan.verification_units), "ru_count": len(plan.commitment_units),
            "declared_width_rule": "population_v2._callee_out_bits: a primitive's width, or the sum of a definition's declared return widths (the certificate's kind table)",
            "actual_width_rule": "|S.out| and Σ widths of the unit's outgoing boundary (gates read outside the unit or protected), the validator's boundary_of",
            **res}


# -- census -------------------------------------------------------------------------------------------------------------------


def _load(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text())


def _validation_view(val: dict[str, Any]) -> dict[str, Any]:
    """What the census takes from a ``compact_p1_validate.py --inputs prescribed`` record (``veritor/compact-p1-validation/v1``)."""

    if val.get("schema") != "veritor/compact-p1-validation/v1":
        raise SystemExit(f"REFUSED: validation record schema {val.get('schema')!r} is not compact-p1-validation/v1")
    plan = val["plan"]
    if plan.get("inputs") != "prescribed" or not plan["name"].endswith("/inputs=prescribed"):
        raise SystemExit(f"REFUSED: validation record is of plan {plan.get('name')!r} with inputs={plan.get('inputs')!r}, not the prescribed construction")
    counts = {k: int(v) for k, v in plan["kind_counts"].items()}
    if any(k.startswith("Input") for k in counts):
        raise SystemExit("REFUSED: the prescribed plan's kind counts contain an Input kind")
    v = val.get("validation") or {}
    return {"source": "scripts/compact_p1_validate.py (veritor/compact-p1-validation/v1)", "utc": val.get("utc"), "host": val.get("host"),
            "platform": val.get("platform"), "cpu": val.get("cpu"), "cpus": val.get("cpus"), "args": val.get("args"),
            "program": val.get("program"), "config": val.get("config"),
            "vus": int(plan["vus"]), "cus": int(plan["cus"]), "vu_runs": plan.get("vu_runs"),
            "partition_digest": plan["partition_digest"], "partition_digest_kind": plan["partition_digest_kind"],
            "plan_digest": plan.get("plan_digest"), "prescribed_outside": plan.get("prescribed_outside"), "prescribed_outside_gates": plan.get("prescribed_outside_gates"),
            "kind_counts": counts,
            "verdict": {"status": v.get("status"), "validated": v.get("validated"), "mode": v.get("mode"), "vu_edges": v.get("vu_edges"), "cu_edges": v.get("cu_edges"),
                        "escape_steps": v.get("escape_steps"), "vu_stats": v.get("vu_stats"), "cu_stats": v.get("cu_stats"), "reasons": v.get("reasons"),
                        "seconds": v.get("seconds"), "worker_maxrss_mb": v.get("worker_maxrss_mb")} if v else None,
            "maxrss_mb": val.get("maxrss_mb"), "seconds_total": val.get("seconds_total"),
            "plan_materialized": {"runs": plan.get("vu_runs"), "maxrss_mb": val.get("maxrss_mb"), "worker_maxrss_mb": v.get("worker_maxrss_mb"),
                                  "note": "compact_p1_validate.py builds the canonical plan in memory at this shape (UnitRuns; D2X: ~403 B/run) and validates it "
                                          "streamed; the MODELED shape (6500x1599) materializes nothing -- counts by rule, rows in the rank layout"}}


def _prescribed_from_validation(vv: dict[str, Any], rules_prescribed: dict[str, Any] | None) -> dict[str, Any]:
    ivs = vv.get("prescribed_outside") or []
    gates = sum(hi - lo for lo, hi in ivs)
    if vv.get("prescribed_outside_gates") not in (None, gates):
        raise SystemExit(f"REFUSED: prescribed_outside intervals {gates} gates != recorded {vv.get('prescribed_outside_gates')}")
    prompt = ivs[0][1] - ivs[0][0] if ivs else 0
    weights = gates - prompt
    doc = {"gates": gates, "gate_intervals": [list(iv) for iv in ivs], "sides": ["verification", "commitment"],
           "prompt_token_ids": {"gates": prompt, "container_bits": 32, "interval": list(ivs[0]) if ivs else None},
           "weights": {"gates": weights, "bits": 16, "intervals": [list(iv) for iv in ivs[1:]]},
           "seed": {"gates": 0, "note": "greedy decoding: TokenSelect is deterministic, no seed gate exists in Serve_v1"},
           "treatment": "prescribed: in no VU and no RU; values checked against the declared input unconditionally (contract step 2), never sampled",
           "label": "MEASURED (the plan's root Input batch intervals)"}
    if rules_prescribed is not None:
        if rules_prescribed["gates"] != gates or rules_prescribed["prompt_token_ids"]["gates"] != prompt:
            raise SystemExit(f"REFUSED: rule prescribed inputs {rules_prescribed['gates']}/{rules_prescribed['prompt_token_ids']['gates']} != plan {gates}/{prompt}")
        doc["weights"]["consumed"] = rules_prescribed["weights"]["consumed"]
        doc["weights"]["unconsumed_rope_rows"] = rules_prescribed["weights"]["unconsumed_rope_rows"]
        doc["weights"]["split_label"] = "MODELED (rule Input|batch|weights|out=16 / out=0; exact at 32x15: census_v2 whole-node escape set)"
    return doc


def declared_primitive_width(kind: str) -> int | None:
    """The declared return width of a registered scalar primitive kind (``Name_vN|primitive``), else None."""

    m = re.match(r"^(?P<name>[A-Za-z0-9]+)_v(?P<ver>\d+)\|primitive$", kind)
    if not m:
        return None
    from verity_vllm.registry import prims

    prim = getattr(prims, m.group("name"), None)
    ret = getattr(prim, "ret", None)
    if prim is None or ret is None or getattr(prim, "version", None) != int(m.group("ver")):
        return None
    hw = ret.homogeneous_width() if callable(getattr(ret, "homogeneous_width", None)) else None
    return int(hw) if hw is not None else None


def vocab_of(case: str) -> int:
    if case == "TINY":
        from e2e import population as popmod

        return int(popmod.TINY["V"])
    from verity_vllm.registry import b1

    return int(getattr(b1, f"{case}_CONFIG")["V"])


def ru_interfaces(case: str, lp: int, steps: int, vv: dict[str, Any] | None) -> dict[str, Any]:
    """[A7 W-A7-2] The per-RU interfaces the certificate needs and no population file carried: one RU per request (P1/cu=whole), its
    designated-output width per RU (``exit_bits`` = NOUT tokens x ceil(log2 V) bits, the declared output domain the record prices; the
    §1.5.9 18-bit index unit and the I32 container are printed beside it, never clamped) and the RU -> request map.  ``in_bits`` and the
    RU's S.out are MEASURED when a validation record is present."""

    nout = steps + 1
    v = vocab_of(case)
    domain_bits = (v - 1).bit_length()
    per_ru = {"ru": 0, "request": 0, "out_gates": nout, "out_bits_container": nout * TOKEN_SELECT_CONTAINER_BITS,
              "exit_bits": nout * domain_bits, "exit_bits_index_unit_18": nout * TOKEN_SELECT_INDEX_BITS,
              "exit_bits_basis": f"NOUT = {nout} designated token-id gates x ceil(log2 V = {v}) = {domain_bits} bits (the declared output domain "
                                 f"W_declared / N the record prices); container I32 x NOUT = {nout * 32}; §1.5.9 index unit 18 x NOUT = {nout * 18}",
              "in_bits": None, "label": "MODELED (NOUT and V from the workload; no validation record)"}
    if vv and vv.get("verdict") and vv["verdict"].get("cu_stats"):
        cs = vv["verdict"]["cu_stats"]
        if cs.get("outputs") != nout or cs.get("out_bits") != nout * 32:
            raise SystemExit(f"REFUSED: RU S.out {cs.get('outputs')} gates / {cs.get('out_bits')} bits != NOUT {nout} x 32")
        per_ru.update({"in_bits": cs.get("in_bits"), "out_gates_measured": cs.get("outputs"), "out_bits_container_measured": cs.get("out_bits"),
                       "label": "MEASURED (the validated plan's CU boundary: |S.out| gates and bits; exit_bits = out_gates x domain bits)"})
    return {"ru_count": 1, "rus_per_request": 1, "per_ru": [per_ru],
            "map": {"rule": "per-instance file: RU 0 serves the instance's one request; the sampler's ReplicatedView makes global RU r * ru_count + j of "
                            "instance (request) r -- every RU serves exactly one request (P1/cu=whole)", "per_ru_request": [0]},
            "certificate": "certificate_v2.population_from_population_file(path, instances=N, ru_exit_bits=[r['exit_bits'] for r in per_ru]) -- one exit "
                           "width per RU of the per-instance file; replay_units == N x ru_count"}


def build_census(lp: int, steps: int, *, case: str = "B0", validation: dict[str, Any] | None, widths: dict[str, Any] | None,
                 widths_from: dict[str, Any] | None = None, rules: bool = True, token_select: str = "whole-scan") -> dict[str, Any]:
    """Assemble the computed-only census of (LP, STEPS).

    ``validation``: a ``compact_p1_validate.py --inputs prescribed`` record at this shape (MEASURED counts, partition/plan digests,
    streamed acyclicity).  ``widths``: a :func:`build_widths` record at this shape (MEASURED widths).  ``widths_from``: another census
    whose *measured* width sets are carried over as the kind's widths (a property of the kind's relation, not of the shape; the
    per-width split at this shape is then MODELED where a rule exists, else UNAVAILABLE).  ``rules``: evaluate L5's rules (B0 only).
    ``token_select``: ``"whole-scan"`` (Amendment A2 decision 4, default) regroups the validated plan's ``ArgmaxStep_v1|scan`` units into
    NOUT ``TokenSelect_v1|whole-scan`` units -- that row and the partition identity are then MODELED (the regrouped plan is not validated
    here; every other kind's units are unchanged and stay MEASURED); ``"argmax-scan"`` is the validated P1 form as measured.
    """

    if token_select not in TOKEN_SELECT_FORMS:
        raise SystemExit(f"REFUSED: token_select {token_select!r} not in {TOKEN_SELECT_FORMS}")
    rules_doc = build_rules(lp, steps, token_select=token_select) if rules and case == "B0" else None
    vv = _validation_view(validation) if validation else None
    nout = steps + 1
    measured_token_selection: dict[str, Any] | None = None
    partial: dict[str, Any] | None = None  # a sharded verdict whose finished ranges do not tile [0, n): every finished range valid, gaps UNAVAILABLE
    if vv is not None:
        if vv["args"]["case"] != case or int(vv["args"]["lp"]) != lp or int(vv["args"]["steps"]) != steps:
            raise SystemExit(f"REFUSED: validation record is {vv['args']['case']} {vv['args']['lp']}x{vv['args']['steps']}, census is {case} {lp}x{steps}")
        verdict = validation.get("validation") or {}
        if verdict and verdict.get("status") != "valid":
            reasons = verdict.get("reasons") or []
            parts = verdict.get("workers") or []
            if reasons and all(r[0] == "uncovered" for r in reasons) and parts and all(p.get("status") == "valid" for p in parts):
                gaps = verdict.get("gaps") or []
                partial = {"gaps": gaps, "units_validated": verdict["vu_stats"]["units"], "units": vv["vus"],
                           "ranges_valid": len(parts), "units_unvalidated": sum(hi - lo for lo, hi in gaps),
                           "label": f"PARTIAL: {len(parts)} finished ranges all valid ({verdict['vu_stats']['units']} of {vv['vus']} verification units); "
                                    f"acyclicity of the units in {gaps} UNAVAILABLE (never finished on the pod)"}
                vv["verdict"]["status"] = "partial"
                vv["verdict"]["partial"] = partial
            else:
                raise SystemExit(f"REFUSED: validation record verdict {verdict.get('status')!r}: {reasons[:3]}")
        embedded = verdict.get("widths")
        if embedded and widths is None:  # compact_p1_validate.py --widths: the widths observed on the validator's own boundaries
            if not embedded.get("complete") and partial is None:
                raise SystemExit("REFUSED: the validation record's embedded widths do not cover every verification unit")
            widths = {"schema": WIDTHS_SCHEMA, "case": case, "LP": lp, "STEPS": steps, "inputs": "prescribed", "partition_digest": vv["partition_digest"],
                      "kinds": embedded["kinds"], "units": embedded["units"], "utc": vv["utc"], "host": vv["host"], "workers": (vv["args"] or {}).get("workers"),
                      "seconds": (vv["verdict"] or {}).get("seconds"), "escape_steps": None, "maxrss_mb": (vv["verdict"] or {}).get("worker_maxrss_mb"),
                      "label": embedded["label"], "program": vv.get("program") or {}, "source": "embedded in the validation record (validate_plan on_boundary observer)",
                      "complete": bool(embedded.get("complete"))}
            if partial is not None:
                widths["label"] = f"MEASURED on {embedded['units']} of {vv['vus']} units (validated ranges only; units in {partial['gaps']} unobserved)"
    if widths is not None:
        if widths.get("schema") != WIDTHS_SCHEMA or widths["case"] != case or int(widths["LP"]) != lp or int(widths["STEPS"]) != steps or widths["inputs"] != "prescribed":
            raise SystemExit(f"REFUSED: widths record is not a {WIDTHS_SCHEMA} of {case} {lp}x{steps} inputs=prescribed")
        if vv is not None and widths["partition_digest"] != vv["partition_digest"]:
            raise SystemExit(f"REFUSED: widths record partition digest {widths['partition_digest'][:16]} != validation {vv['partition_digest'][:16]}")
        if vv is not None and vv["verdict"] and widths.get("escape_steps") is not None and vv["verdict"].get("escape_steps") is not None:
            # the standalone pass and the validator visited the same boundaries iff their VU-side escape steps agree
            vu_steps = vv["verdict"]["escape_steps"] - (vv["verdict"].get("cu_escape_steps") or 0)
            widths["escape_steps_equal_validation_vu_side"] = widths["escape_steps"] == vu_steps
    if widths_from is not None and widths_from.get("schema") != CENSUS_SCHEMA:
        raise SystemExit(f"REFUSED: --widths-from is not a {CENSUS_SCHEMA} census")

    rule_counts = {k["kind"]: k["count"] for k in rules_doc["kinds"]} if rules_doc else {}
    rule_w = rules_doc["rule_widths"] if rules_doc else {}
    measured_counts = dict(vv["kind_counts"]) if vv else (
        {k: v["count"] for k, v in widths["kinds"].items()} if widths else {})
    if vv and widths:
        wc = {k: v["count"] for k, v in widths["kinds"].items()}
        if partial is None and wc != measured_counts:
            raise SystemExit(f"REFUSED: widths record counts differ from the validation record's kind counts: {sorted(set(wc.items()) ^ set(measured_counts.items()))[:6]}")
        if partial is not None and any(wc[k] > measured_counts.get(k, 0) for k in wc):
            raise SystemExit("REFUSED: widths record observed more units of a kind than the plan has")
    measured_vus = vv["vus"] if vv else None
    if measured_counts and token_select == "whole-scan":
        # the validated plan carries the P1 form; the ruled form is its regrouping (counts exact by division, the regrouped plan not validated here)
        src = f"validated plan {vv['partition_digest'][:16]} (P1 form)" if vv else "widths record (P1 form)"
        measured_token_selection = regroup_token_selection(measured_counts, nout, source=src)
        measured_vus = measured_vus - measured_token_selection["replaces"]["units"] + nout if measured_vus is not None else None
        if widths is not None and ARGMAX_KIND in widths["kinds"] and widths.get("complete", True):
            wk = widths["kinds"][ARGMAX_KIND]
            last_step_32 = sum(n for _, bits, n in wk["actual_out_bits"]["histogram"] if bits == TOKEN_SELECT_CONTAINER_BITS)
            if last_step_32 != nout:
                raise SystemExit(f"REFUSED: {last_step_32} ArgmaxStep units with a 32-bit S.out != NOUT {nout}: the whole-scan unit's S.out is not one I32 per step")
            measured_token_selection["replaces"]["width_measured"] = {"declared_bits": wk["declared_out_bits"], "actual_out_bits": wk["actual_out_bits"],
                                                                      "in_bits": wk["in_bits"], "label": "MEASURED (P1 form, exhaustive)"}
            measured_token_selection["whole_scan_s_out_measured"] = {"units_with_32_bit_s_out": last_step_32, "equals_NOUT": True,
                                                                     "basis": "the last scan step of each decode step is the only ArgmaxStep unit whose S.out is the I32 ni"}
    keys = sorted(set(rule_counts) | set(measured_counts))
    carried = {k["kind"]: k for k in (widths_from or {}).get("kinds", []) if k.get("width", {}).get("label") == "MEASURED"}

    rows: list[dict[str, Any]] = []
    inconsistent: list[dict[str, Any]] = []
    offset = 0
    for k in keys:
        rc, mc = rule_counts.get(k), measured_counts.get(k)
        regrouped = k == TOKEN_SELECT_KIND and measured_token_selection is not None
        if mc is not None and rc is not None:
            if mc == rc:
                count, clabel, cause = mc, ("MODELED" if regrouped else "MEASURED"), None
            else:
                count, clabel, cause = mc, "INCONSISTENT", f"rule {rc} != measured {mc} (unit-by-unit count of the validated plan)"
        elif mc is not None:
            count, clabel, cause = mc, ("MODELED" if regrouped else "MEASURED"), None
        else:
            count, clabel, cause = rc, "MODELED", None
        if count == 0 and mc is None:
            # a kind the rules know but that is empty at this shape (e.g. F32FmaFtz below the FA2 rescale horizon): recorded, offset unchanged
            pass
        fn, _, form = k.partition("|")
        row: dict[str, Any] = {"kind": k, "fn": fn, "form": form, "count": count, "offset": offset, "count_label": clabel}
        if rc is not None:
            row["count_rule"] = rc
        if mc is not None:
            row["count_measured"] = mc
        if cause:
            row["cause"] = cause
            inconsistent.append({"kind": k, "cause": cause})
        if regrouped:
            row["basis"] = (f"regrouping of the validated plan's {measured_token_selection['replaces']['units']} {ARGMAX_KIND} units "
                            f"({measured_token_selection['replaces']['scan_steps_per_unit']} per decode step) into NOUT = {nout} whole-scan units "
                            f"({TOKEN_SELECT_RULE}); exact by division, the regrouped plan itself is not validated here")
        # widths
        w: dict[str, Any]
        if k == TOKEN_SELECT_KIND:
            w = {"declared_bits": TOKEN_SELECT_INDEX_BITS, "declared_basis": "§1.5.9 token-selection index unit (18 bits as declared)",
                 "container_bits": TOKEN_SELECT_CONTAINER_BITS, "domain_bits": (vocab_of(case) - 1).bit_length(), "vocab": vocab_of(case),
                 "actual_min_bits": TOKEN_SELECT_CONTAINER_BITS, "actual_max_bits": TOKEN_SELECT_CONTAINER_BITS,
                 "histogram": [[1, TOKEN_SELECT_CONTAINER_BITS, count]], "histogram_fields": ["out_scalars", "out_bits", "units"],
                 "declared_equals_actual": False, "label": "MODELED",
                 "basis": "S.out of the whole-scan unit = S.out of its last scan step (the I32 gate ni)"
                          + (": MEASURED 32 bits on exactly NOUT ArgmaxStep units at this shape" if measured_token_selection and
                             measured_token_selection.get("whole_scan_s_out_measured") else "; the same relation at every shape (D2 P1v2: out_bits 32)")
                          + "; declared 18 / domain 16 (SMALL) / actual 32 printed side by side, never clamped"}
        elif widths is not None and k in widths["kinds"]:
            wk = widths["kinds"][k]
            w = {"declared_bits": wk["declared_out_bits"], "actual_min_bits": wk["actual_out_bits"]["min"], "actual_max_bits": wk["actual_out_bits"]["max"],
                 "histogram": wk["actual_out_bits"]["histogram"], "histogram_fields": ["out_scalars", "out_bits", "units"],
                 "declared_equals_actual": wk["declared_equals_actual"], "in_bits": wk["in_bits"], "label": "MEASURED",
                 "basis": "exhaustive: every unit's boundary at this shape"}
            if partial is not None and wk["count"] != count:
                w["units_observed"] = wk["count"]
                w["basis"] = f"MEASURED on {wk['count']} of {count} units (the validated ranges); the units in {partial['gaps']} were not observed"
                w["complete"] = False
            if k in rule_w and (partial is None or wk["count"] == count):
                by_bits = Counter()
                for _, bits, n in wk["actual_out_bits"]["histogram"]:
                    by_bits[bits] += n
                w["rule_split_agrees"] = [[b, n] for b, n in sorted(by_bits.items())] == rule_w[k]
                if not w["rule_split_agrees"]:
                    row["cause"] = (row.get("cause", "") + f"; rule per-width split {rule_w[k]} != measured {sorted(by_bits.items())}").lstrip("; ")
                    row["count_label"] = "INCONSISTENT"
                    inconsistent.append({"kind": k, "cause": row["cause"]})
        elif k in carried:
            cw = carried[k]["width"]
            w = {"declared_bits": cw["declared_bits"], "actual_min_bits": cw["actual_min_bits"], "actual_max_bits": cw["actual_max_bits"],
                 "declared_equals_actual": cw["declared_equals_actual"], "label": "MODELED",
                     "basis": f"width set MEASURED at {widths_from['case']} {widths_from['shape']['LP']}x{widths_from['shape']['STEPS']} (exhaustive; a property of the "
                              f"kind's relation); assumed the same set at this shape"}
            if k in rule_w:
                w["histogram_by_bits"] = rule_w[k]
                w["split_label"] = "MODELED (rule per-width sub-families)"
            elif cw["actual_min_bits"] != cw["actual_max_bits"]:
                w["split_label"] = "UNAVAILABLE (no per-width rule; only the width set is known at this shape)"
            else:
                w["histogram_by_bits"] = [[cw["actual_max_bits"], count]]
                w["split_label"] = "MODELED (single width observed)"
        else:
            w = {"declared_bits": None, "actual_min_bits": None, "actual_max_bits": None, "label": "UNAVAILABLE",
                 "basis": "no widths record at this shape and no census to carry a measured width set from"}
            declared = declared_primitive_width(k)
            if declared is not None:  # a registered scalar primitive no measured shape instantiates (F32FmaFtz / F32SubFtz below the FA2 rescale horizon)
                w.update({"declared_bits": declared, "actual_max_bits": declared, "actual_min_bits": None, "label": "MODELED",
                          "basis": "declared return width of the registered primitive (verity_vllm.registry.prims); no unit of this kind exists at any "
                                   "measured shape, so the actual set is UNAVAILABLE -- actual_max is the declared bound (a dead 0-bit unit would be below it)"})
            if k in rule_w:
                w["histogram_by_bits"] = rule_w[k]
                w["actual_min_bits"], w["actual_max_bits"] = rule_w[k][0][0], rule_w[k][-1][0]
                w["split_label"] = "MODELED (rule per-width sub-families)"
                w["label"] = "MODELED"
        row["width"] = w
        rows.append(row)
        offset += count
    vu_count = offset
    if measured_vus is not None and vu_count != measured_vus:
        raise SystemExit(f"REFUSED: kind counts sum to {vu_count} but the validated plan has {measured_vus} VUs"
                         + (" after regrouping" if measured_token_selection else ""))
    if rules_doc and not measured_counts and vu_count != rules_doc["vu_count"]:
        raise SystemExit("REFUSED: census kinds do not sum to the rules' vu_count")

    measured = vv is not None
    max_cut = max((r["width"]["actual_max_bits"] or 0) for r in rows) if rows else 0
    # identities
    if measured and measured_token_selection is not None:
        validated = {"digest": vv["partition_digest"], "kind": vv["partition_digest_kind"], "label": "MEASURED", "form": "argmax-scan (P1)",
                     "vus": vv["vus"], "source": f"{vv['source']} on {vv['host']} ({vv['utc']})"}
        partition = {"digest": None, "kind": None, "label": "UNAVAILABLE",
                     "note": "the regrouped (whole-scan) partition's partition-intervals/v1 digest is re-derived from the regrouped rows of the "
                             "validated plan by `population_v3.py publish --rows` (MODELED: the regrouped plan is not validated by validate_plan)",
                     "validated_p1_form": validated}
        plan_digest = {"digest": _digest({"schema": "veritor/a1/plan-regrouped/v1", "from": vv["plan_digest"], "from_partition": vv["partition_digest"],
                                          "regrouping": TOKEN_SELECT_RULE, "kind": TOKEN_SELECT_KIND}),
                       "kind": "a1/plan-regrouped/v1 (identity digest: the validated P1 plan digest + the regrouping rule; not veritor.plans.summary.plan_digest)",
                       "label": "MODELED", "from": vv["plan_digest"]}
    elif measured:
        partition = {"digest": vv["partition_digest"], "kind": vv["partition_digest_kind"], "label": "MEASURED",
                     "source": f"{vv['source']} on {vv['host']} ({vv['utc']})"}
        plan_d = vv["plan_digest"] or {"kind": None, "digest": None}
        plan_digest = {"digest": plan_d.get("digest"), "kind": plan_d.get("kind"), "label": "MEASURED" if plan_d.get("digest") else "UNAVAILABLE"}
    if measured:
        program = {"name": (vv.get("program") or {}).get("name") or (widths or {}).get("program", {}).get("name"),
                   "digest": (vv.get("program") or {}).get("digest") or (widths or {}).get("program", {}).get("digest"),
                   "digest_kind": "verity-ir/program_digest", "gates": (vv.get("program") or {}).get("gates"),
                   "input_gates": (vv.get("program") or {}).get("input_gates"), "label": "MEASURED"}
        prescribed = _prescribed_from_validation(vv, rules_doc["prescribed_inputs"] if rules_doc else None)
    else:
        partition = {"digest": rules_doc["partition_digest"], "kind": rules_doc["partition_digest_kind"], "label": "MODELED",
                     "source": "identity digest of the rule-generated kind table + prescribed-inputs block (no plan exists at this shape)"}
        plan_digest = {"digest": rules_doc["plan_digest"], "kind": "a1/plan-rule-modeled/v1", "label": "MODELED"}
        program = {"name": f"Serve_v1{{C={{{B0_CONFIG_SHA12}}},LP={lp},STEPS={steps}}}", "digest": rules_doc["program_digest"],
                   "digest_kind": rules_doc["program_digest_kind"], "gates": None, "input_gates": rules_doc["prescribed_inputs"]["gates"], "label": "MODELED"}
        prescribed = dict(rules_doc["prescribed_inputs"], label="MODELED (rules Input|batch|prompt, Input|batch|weights|out=16/out=0)")
    prescribed.pop("token_selection", None)
    token_selection = measured_token_selection or (rules_doc["token_selection"] if rules_doc else
                                                   {"form": token_select, "kind": TOKEN_SELECT_KIND if token_select == "whole-scan" else ARGMAX_KIND})
    if rules_doc and measured_token_selection and rules_doc["token_selection"]["units"] != measured_token_selection["units"]:
        raise SystemExit("REFUSED: rules and measured token-selection unit counts differ")
    interfaces = ru_interfaces(case, lp, steps, vv)
    ru = {"count_per_instance": 1, "rule": "one RU per request: P1/cu=whole over the computed gates; in(R) = every consumed input gate (prescribed), "
                                          "out(R) = the NOUT designated token-id gates (the token-selection unit's ni per decode position)",
          "out_gates": nout, "out_bits_container": nout * 32, "out_domain_bits_declared": interfaces["per_ru"][0]["exit_bits"],
          "out_domain_note": f"container I32 per token gate; declared domain = vocabulary index ({interfaces['per_ru'][0]['exit_bits'] // nout} bits at V = "
                             f"{vocab_of(case)}); A2/A7 own the domain check"}
    if vv and vv["verdict"] and vv["verdict"].get("cu_stats"):
        cs = vv["verdict"]["cu_stats"]
        in_bits = cs.get("in_bits")
        consumed = None
        if in_bits is not None and (in_bits - lp * 32) % 16 == 0:
            consumed = {"prompt": lp, "weights": (in_bits - lp * 32) // 16, "derivation": "in_bits = LP x 32 (prompt I32) + consumed weight gates x 16 (bf16)"}
        ru.update({"out_gates_measured": cs.get("outputs"), "out_bits_container_measured": cs.get("out_bits"), "in_bits_measured": in_bits,
                   "in_gates_consumed_derived": consumed, "cu_stats": cs})
        if cs.get("outputs") != nout or cs.get("out_bits") != nout * 32:
            raise SystemExit(f"REFUSED: RU S.out {cs.get('outputs')} gates / {cs.get('out_bits')} bits != NOUT {nout} x 32")
        if consumed is not None and prescribed.get("weights", {}).get("consumed") not in (None, consumed["weights"]):
            raise SystemExit(f"REFUSED: RU in_bits imply {consumed['weights']} consumed weight gates != rule {prescribed['weights']['consumed']}")
    with_inputs = vu_count + prescribed["gates"]
    agreement = {"rules_vs_measured": {"checked": bool(rule_counts and measured_counts), "kinds_compared": len(set(rule_counts) & set(measured_counts)),
                                       "equal": not inconsistent and bool(rule_counts and measured_counts),
                                       "inconsistent": inconsistent,
                                       "rule_kinds_absent_from_plan": sorted(k for k in rule_counts if k not in measured_counts and rule_counts[k] != 0) if measured_counts else None,
                                       "plan_kinds_absent_from_rules": sorted(k for k in measured_counts if k not in rule_counts) if rule_counts else None}}
    if rules_doc and measured_counts:
        agreement["rules_vs_measured"]["zero_rule_kinds_absent_from_plan_as_expected"] = sorted(k for k in rule_counts if rule_counts[k] == 0 and k not in measured_counts)
        agreement["rules_vs_measured"]["prescribed_inputs_equal"] = rules_doc["prescribed_inputs"]["gates"] == prescribed["gates"]
    label = ("MEASURED (validated unit by unit: counts, digests and acyclicity from compact_p1_validate --inputs prescribed"
             + ("; widths exhaustive" if widths else "; widths " + ("carried" if widths_from else "UNAVAILABLE"))
             + (f"; except {TOKEN_SELECT_KIND}: MODELED regrouping of the validated ArgmaxStep scan-step units (Amendment A2 decision 4), partition identity "
                "re-derived from the regrouped rows" if measured_token_selection else "") + ")") if measured else \
        "MODELED (rule-generated counts; identity digests of this kind table; no plan exists at this shape)"
    if partial is not None:
        label = (f"{partial['label']}; counts MEASURED for every kind (the plan's kind_counts, independent of the verdict); widths MEASURED on the "
                 f"validated ranges only -- " + label)
    if inconsistent:
        label = "INCONSISTENT: " + "; ".join(f"{x['kind']}: {x['cause']}" for x in inconsistent)
    plan_name = PLAN_NAME + ("/token-select=whole-scan" if token_select == "whole-scan" else "")
    doc = {
        "schema": CENSUS_SCHEMA, "utc": _utc(), "label": label, "construction": CONSTRUCTION + f"; token selection {token_selection['form']}", "case": case,
        "shape": {"LP": lp, "STEPS": steps, "P": lp + steps, "NOUT": nout,
                  "note": "STEPS = decode steps; NOUT = STEPS + 1 = output tokens (the workload's max_tokens); P = LP + STEPS positions"},
        "plan_name": plan_name, "instances": 1, "vu_count": vu_count, "ru_count": 1,
        "program": program, "partition": partition, "plan_digest": plan_digest,
        "ru": ru, "ru_interfaces": interfaces, "token_selection": token_selection,
        "kinds": rows,
        "kinds_fields_note": "kind/fn/form/count/offset as D4's population-rules/v1 (VU rank = offset + index within kind, kinds in sorted key order; "
                             "a RuleView-style sampler consumes this table) + count_label/count_rule/count_measured/width per row",
        "prescribed_inputs": prescribed,
        "widths_summary": {"max_cut_bits": max_cut, "kinds_declared_ne_actual": sorted(r["kind"] for r in rows if r["width"].get("declared_equals_actual") is False),
                           "kinds_with_zero_bit_units": sorted(r["kind"] for r in rows if r["width"].get("actual_min_bits") == 0),
                           "note": "declared = certificate kind-table width (population_v2._callee_out_bits); actual = the validator's S.out per unit"},
        "validation": vv,
        "widths_record": {"utc": widths["utc"], "host": widths["host"], "seconds": widths["seconds"], "workers": widths["workers"], "escape_steps": widths["escape_steps"],
                          "maxrss_mb": widths["maxrss_mb"], "label": widths["label"], "source": widths.get("source", "population_v3.py widths (standalone pass)"),
                          "escape_steps_equal_validation_vu_side": widths.get("escape_steps_equal_validation_vu_side")} if widths else None,
        "rules": {"vu_count": rules_doc["vu_count"], "kind_table_digest": rules_doc["kind_table_digest"], "plan_digest": rules_doc["plan_digest"],
                  "program_identity_digest": rules_doc["program_digest"], "source": rules_doc["rules"], "widths": rules_doc["widths"],
                  "note": "the same kind table as a rule-generated population; when this census is MEASURED the identity digests here and the plan digests above name the same population"} if rules_doc else None,
        "agreement": agreement,
        "with_inputs_for_comparison": {"vu_count": with_inputs, "input_units": prescribed["gates"], "ratio_with_over_without": round(with_inputs / vu_count, 6) if vu_count else None,
                                       "legacy_partition_digest_note": "the with-inputs (units) construction has its own partition digest (32x15: 56ebcdf5..., TINY: 3646488c...); "
                                                                       "it never names this population"},
        "consumers": {
            "a9_runner": "e2e.population_v3.rule_view_kinds(census) -> D4 RuleView-shaped kind table (replaces population_v5 legacy-minus-inputs); at a shape with a plan, "
                         "the population/v2 rows of the prescribed plan: e2e.population maps --plan P1/cu=whole/inputs=prescribed --stream --population-v2 (pod)",
            "a7_certificate": "e2e.population_v3.certificate_families(census, instances, width='actual_max'|'declared') -> (families, banks, out_width) for "
                              "certificate_v2.population_from_families(..., source='a1'); identity by partition.digest (+ kind)",
            "d2x_census": "agreement.rules_vs_measured; kinds[].count_rule vs count_measured",
        },
    }
    doc["content_sha256"] = _digest({k: v for k, v in doc.items() if k not in ("content_sha256", "utc")})
    return doc


# -- consumer adapters ----------------------------------------------------------------------------------------------------------


def rule_view_kinds(census: dict[str, Any]) -> list[dict[str, Any]]:
    """The census's kind table in D4's ``population-rules/v1`` shape (``kind/fn/form/count/offset``, sorted key order, one RU per
    instance) -- what a ``RuleView``-style sampler (A9 ``V5RulesView``, D4 ``RuleView``) consumes; no ``Input`` kind is present."""

    if census.get("schema") != CENSUS_SCHEMA:
        raise ValueError(f"not a {CENSUS_SCHEMA} census")
    out = []
    acc = 0
    for r in census["kinds"]:
        if r["fn"] == "Input":
            raise ValueError("census contains an Input kind")
        if r["offset"] != acc:
            raise ValueError(f"kind {r['kind']} offset {r['offset']} != running total {acc}")
        out.append({"kind": r["kind"], "fn": r["fn"], "form": r["form"], "count": int(r["count"]), "offset": int(r["offset"])})
        acc += int(r["count"])
    if acc != census["vu_count"]:
        raise ValueError(f"kinds sum to {acc} != vu_count {census['vu_count']}")
    return out


def certificate_families(census: dict[str, Any], instances: int, *, width: str = "actual_max") -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """``(families, banks, out_width)`` for ``certificate_v2.population_from_families(families, banks, out_width, source="a1", ...)``:
    one family per computed kind with ``count`` = units over all ``instances`` RUs and ``width`` = the kind's ``actual_max`` (default) or
    ``declared`` bits (a census row whose width is UNAVAILABLE refuses); one bank ``P1/cu=whole RU`` with ``copies = instances``,
    ``units`` = VUs per RU and ``exit_width`` = the RU's S.out container bits (NOUT x 32).  ``N_V == Σ B_a v_a`` by construction."""

    if census.get("schema") != CENSUS_SCHEMA:
        raise ValueError(f"not a {CENSUS_SCHEMA} census")
    if instances < 1:
        raise ValueError("instances must be >= 1")
    key = {"actual_max": "actual_max_bits", "declared": "declared_bits"}[width]
    fams = []
    for r in census["kinds"]:
        if r["count"] == 0:
            continue
        w = r["width"].get(key)
        if w is None:
            raise ValueError(f"kind {r['kind']}: {key} is UNAVAILABLE in this census ({r['width'].get('label')})")
        fams.append({"kind": r["kind"], "count": int(r["count"]) * instances, "width": int(w), "role": "computed",
                     "provenance": f"a1 census {census['shape']['LP']}x{census['shape']['STEPS']}: count {r['count_label']}, width {r['width']['label']} ({key})"})
    exits = [int(r["exit_bits"]) for r in census["ru_interfaces"]["per_ru"]]
    pd = census["partition"].get("digest")
    banks = [{"kind": f"P1/cu=whole RU{i} (computed gates)", "copies": instances, "units": int(census["vu_count"]), "exit_width": e,
              "provenance": f"a1 census: one RU per request; exit = NOUT x declared domain bits ({census['partition']['label']} partition "
                            f"{pd[:16] if pd else 'UNAVAILABLE'})"} for i, e in enumerate(exits)]
    return fams, banks, sum(exits)


# -- the published population file (one file for the sampler, the verifier's locator and the certificate) ---------------------------
#
# [Integrator 21:15Z / A7 W-A7-2, W-A7-3]  The published census is a ``veritor/population/v2`` document -- the schema
# ``e2e.partition_sample.load_population`` (CompactPopulation) and ``certificate_v2.population_from_population_file`` already read -- with
# the census embedded under ``census`` and the per-RU interfaces under ``ru_interfaces``.  MEASURED shapes carry the validated plan's rows
# (regrouped to the ruled token-selection form); the MODELED shape carries one row per kind in the rank layout (no circuit gates exist
# there: ``gate_layout = "rank"``), so the same loaders and the same ``RankLayout`` inverse serve every shape.

RANK_LAYOUT_SCHEMA = "veritor/a1/rank-layout/v1"
GATE_LAYOUT_RANK = ("rank", "no circuit at this shape: unit k of kind i occupies rank offset_i + k; the row's lo/stride/unit_size are ranks, "
                            "NOT circuit gates (gate_interval(vu) is the VU's own rank interval); gates = vu_count")


def regroup_rows(rows_doc: dict[str, Any], nout: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply Amendment A2 decision 4 to a ``population/v2`` document of the P1 form: every ``ArgmaxStep_v1|scan`` row (one row per
    decode step's scan: ``L`` units of 5 gates, stride 5 -- refused otherwise) becomes ONE ``TokenSelect_v1|whole-scan`` unit over the
    same gates; VU ranks are renumbered contiguously; the kind entry is replaced under the same id (``out_bits`` 32 = the I32 ``ni``
    the unit exposes, ``size`` = the scan's gates).  Returns ``(new document without content_sha256, regrouping record)``."""

    from e2e.population_v2 import POPULATION_V2, ROW_FIELDS

    if rows_doc.get("schema") != POPULATION_V2 or rows_doc.get("row_fields") != list(ROW_FIELDS):
        raise SystemExit("REFUSED: rows document is not population/v2")
    entries = [dict(k) for k in rows_doc["kinds"]["entries"]]
    argmax = [k for k in entries if f"{k['fn']}|{k['form']}" == ARGMAX_KIND]
    if len(argmax) != 1:
        raise SystemExit(f"REFUSED: {len(argmax)} {ARGMAX_KIND} kind entries (expected exactly one)")
    if any(f"{k['fn']}|{k['form']}" == TOKEN_SELECT_KIND for k in entries):
        raise SystemExit(f"REFUSED: rows document already carries {TOKEN_SELECT_KIND}")
    kid, size = argmax[0]["id"], int(argmax[0]["size"])
    scans = [r for r in rows_doc["rows"] if r[3] == kid]
    units = sum(r[1] for r in scans)
    if len(scans) != nout or units % nout:
        raise SystemExit(f"REFUSED: {len(scans)} {ARGMAX_KIND} rows / {units} units for NOUT = {nout}: not one scan row per decode step")
    steps_per_scan = units // nout
    for r in scans:
        if r[1] != steps_per_scan or r[5] != size or r[6] != size:
            raise SystemExit(f"REFUSED: scan row {r} is not {steps_per_scan} contiguous units of {size} gates (stride == size)")
    new_rows, nxt, gate_ivs = [], 0, []
    for r in rows_doc["rows"]:
        first, count, ru, k, lo, stride, usz = r
        if k == kid:
            span = count * size
            new_rows.append([nxt, 1, ru, k, lo, span, span])
            gate_ivs.append([lo, lo + span])
            nxt += 1
        else:
            new_rows.append([nxt, count, ru, k, lo, stride, usz])
            nxt += count
    if nxt != int(rows_doc["vu_count"]) - units + nout:
        raise SystemExit("REFUSED: regrouped rows do not renumber to vu_count - scan-step units + NOUT")
    entries[kid] = {"id": kid, "fn": TOKEN_SELECT_KIND.split("|")[0], "form": TOKEN_SELECT_KIND.split("|")[1], "primitive": False,
                    "out_bits": TOKEN_SELECT_CONTAINER_BITS, "size": steps_per_scan * size}
    doc = {k: v for k, v in rows_doc.items() if k not in ("content_sha256", "seconds")}
    doc["rows"] = new_rows
    doc["kinds"] = dict(rows_doc["kinds"], entries=entries)
    doc["vu_count"] = nxt
    doc["per_instance"] = {"vu_count": nxt, "ru_count": int(rows_doc["ru_count"])}
    record = {"form": "whole-scan", "kind": TOKEN_SELECT_KIND, "units": nout, "kind_id": kid,
              "replaces": {"kind": ARGMAX_KIND, "units": units, "scan_steps_per_unit": steps_per_scan, "gates_per_scan_step": size, "rows": len(scans)},
              "whole_scan_gate_intervals": gate_ivs, "rule": TOKEN_SELECT_RULE,
              "from": {"partition_digest": rows_doc["partition_digest"], "partition_digest_kind": rows_doc.get("partition_digest_kind"),
                       "plan_digest": rows_doc["plan_digest"], "vu_count": int(rows_doc["vu_count"]), "content_sha256": rows_doc.get("content_sha256")},
              "label": "MODELED (rows of the validated P1 plan regrouped by rule; the regrouped plan was not validated by validate_plan)"}
    return doc, record


def _rank_layout_rows(census: dict[str, Any]) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """One ``population/v2`` row per non-empty kind in the rank layout (``lo`` = kind offset, stride = unit_size = 1) and the kind entries
    (``out_bits`` = the certificate's declared width: the census's declared bits, or the container for the token-selection unit)."""

    rows, entries = [], []
    for r in census["kinds"]:
        w = r["width"]
        declared = TOKEN_SELECT_CONTAINER_BITS if r["kind"] == TOKEN_SELECT_KIND else w.get("declared_bits")
        if declared is None:
            declared = w.get("actual_max_bits")
        if declared is None:
            raise SystemExit(f"REFUSED: kind {r['kind']} has no declared or actual width to publish")
        kid = len(entries)
        entries.append({"id": kid, "fn": r["fn"], "form": r["form"], "primitive": r["form"] == "primitive", "out_bits": int(declared), "size": 1})
        if r["count"] > 0:
            rows.append([int(r["offset"]), int(r["count"]), 0, kid, int(r["offset"]), 1, 1])
    return rows, entries


def _augment_entries(entries: list[dict[str, Any]], census: dict[str, Any]) -> list[dict[str, Any]]:
    by_kind = {r["kind"]: r for r in census["kinds"]}
    out = []
    for k in entries:
        name = f"{k['fn']}|{k['form']}"
        r = by_kind.get(name)
        if r is None:
            raise SystemExit(f"REFUSED: kind entry {name} has no census row")
        w = r["width"]
        e = dict(k, name=name, count=int(r["count"]), count_label=r["count_label"], out_bits_declared=w.get("declared_bits"),
                 out_bits_actual_min=w.get("actual_min_bits"), out_bits_actual_max=w.get("actual_max_bits"), width_label=w.get("label"))
        if name == TOKEN_SELECT_KIND:
            e.update(index_bits_declared=TOKEN_SELECT_INDEX_BITS, domain_bits=w.get("domain_bits"), container_bits=TOKEN_SELECT_CONTAINER_BITS)
        out.append(e)
    return out


def publish(census: dict[str, Any], rows_doc: dict[str, Any] | None = None) -> dict[str, Any]:
    """The published ``population/v2`` document of a census: rows (validated plan's rows regrouped to the census's token-selection
    form, or the rank layout when the census is MODELED), kind entries with count and widths, ``ru_interfaces``, the census itself.
    Self-checked with ``CompactPopulation.check`` (partition digest re-derived from the rows for MEASURED shapes; ``check="fast"``
    only for the rank layout -- its digest is the kind-table identity digest, O(kinds), not re-derivable by the O(N_V) stream)."""

    from e2e.population_v2 import POPULATION_V2, ROW_FIELDS, CompactPopulation, content_sha256
    from veritor.plans.summary import PARTITION_DIGEST

    if census.get("schema") != CENSUS_SCHEMA:
        raise SystemExit(f"REFUSED: not a {CENSUS_SCHEMA} census")
    census = json.loads(json.dumps(census))
    form = census["token_selection"]["form"]
    nout = census["shape"]["NOUT"]
    measured = census["validation"] is not None
    if measured:
        if rows_doc is None:
            raise SystemExit("REFUSED: a MEASURED census is published from the validated plan's population/v2 rows (--rows)")
        vv = census["validation"]
        if rows_doc.get("partition_digest") != vv["partition_digest"]:
            raise SystemExit(f"REFUSED: rows partition digest {str(rows_doc.get('partition_digest'))[:16]} != the census's validated plan {vv['partition_digest'][:16]}")
        if rows_doc.get("plan_digest") != (vv["plan_digest"] or {}).get("digest"):
            raise SystemExit("REFUSED: rows plan digest != the census's validated plan digest")
        p1_counts = CompactPopulation(rows_doc).kind_counts()
        if p1_counts != vv["kind_counts"]:
            raise SystemExit("REFUSED: rows per-kind counts != the validation record's kind counts")
        if form == "whole-scan":
            doc, regroup = regroup_rows(rows_doc, nout)
        else:
            doc, regroup = {k: v for k, v in rows_doc.items() if k not in ("content_sha256", "seconds")}, None
        doc["label"] = census["label"]
        doc["plan_name"] = census["plan_name"]
        doc["construction"] = census["construction"]
    else:
        rows, entries = _rank_layout_rows(census)
        n = int(census["vu_count"])
        doc = {"schema": POPULATION_V2, "plan_name": census["plan_name"], "instances": 1, "vu_count": n, "ru_count": 1,
               "per_instance": {"vu_count": n, "ru_count": 1}, "gates": n, "gate_layout": GATE_LAYOUT_RANK[0], "gate_layout_note": GATE_LAYOUT_RANK[1],
               "program_digest": census["program"]["digest"], "program_digest_kind": census["program"]["digest_kind"],
               "plan_digest": census["plan_digest"]["digest"], "plan_digest_kind": census["plan_digest"]["kind"],
               "partition_digest": census["partition"]["digest"], "partition_digest_kind": census["partition"]["kind"],
               "partition_digest_note": "identity digest of the kind table + prescribed-inputs block (O(kinds)); NOT re-derivable by "
                                        "CompactPopulation.recompute_partition_digest (that stream is O(N_V) = 1e11 here) -- load with check='fast'",
               "kinds": {"schema": "veritor/population-kinds/v1", "key": "(kind program, node form) as population/v2; rank layout", "entries": entries},
               "row_fields": list(ROW_FIELDS), "rows": rows, "cu_intervals": [[[0, n]]],
               "prescribed_inputs": census["prescribed_inputs"],
               "label": census["label"], "construction": census["construction"]}
        regroup = None
    doc["kinds"] = dict(doc["kinds"], entries=_augment_entries(doc["kinds"]["entries"], census))
    doc["ru_interfaces"] = census["ru_interfaces"]
    doc["token_selection"] = census["token_selection"]
    doc["rank_layout"] = {"schema": RANK_LAYOUT_SCHEMA, "inverse": "e2e.population_v3.RankLayout(doc).locate(rank) -> kind, index within kind, RU, gate interval "
                                                                   "(MEASURED shapes) or rank interval (rank layout); rank_of(kind, index) is its inverse",
                          "order": "gate order of the validated plan's rows (MEASURED shapes) | kind offset + index within kind, kinds in sorted key order (rank layout)"}
    if regroup is not None:
        doc["regrouping"] = regroup
        doc["content_sha256"] = content_sha256(doc)
        d = CompactPopulation(doc).recompute_partition_digest()  # O(N_V) sha stream over the regrouped rows + CU intervals
        doc["partition_digest"], doc["partition_digest_kind"] = d, PARTITION_DIGEST
        doc["partition_digest_note"] = (f"re-derived from the regrouped rows (partition-intervals/v1 stream); the validated P1 form is "
                                        f"{regroup['from']['partition_digest']} -- MODELED: the regrouped plan was not validated by validate_plan")
        doc["plan_digest"], doc["plan_digest_kind"] = census["plan_digest"]["digest"], census["plan_digest"]["kind"]
        census["partition"] = {"digest": d, "kind": PARTITION_DIGEST, "label": "MODELED",
                               "source": "population_v3.publish: partition-intervals/v1 stream over the regrouped rows of the validated plan",
                               "validated_p1_form": census["partition"]["validated_p1_form"]}
        census["content_sha256"] = _digest({k: v for k, v in census.items() if k not in ("content_sha256", "utc")})
    doc["content_sha256"] = content_sha256(doc)
    pop = CompactPopulation(doc)
    res = pop.check(digest=measured)
    counts = pop.kind_counts()
    for r in census["kinds"]:
        if counts.get(r["kind"], 0) != r["count"]:
            raise SystemExit(f"REFUSED: published rows carry {counts.get(r['kind'], 0)} {r['kind']} units, census says {r['count']}")
    doc["census"] = census
    doc["self_check"] = {k: v for k, v in res.items() if not k.endswith("seconds")}  # no wall time in the file: regenerations are byte-identical
    doc["content_sha256"] = content_sha256(doc)
    return doc


class RankLayout:
    """Rank -> (kind, index within kind, RU, interval) and back, by rule over the rows of a published file -- ``O(rows)`` state, ``O(log rows)``
    queries; no plan object.  On a MEASURED file the interval is the unit's circuit gate interval; on the rank layout it is the rank itself.
    NOT DONE here: kind index -> circuit coordinates (request, t, layer, head, position): that inverse needs a per-kind enumeration rule
    (the canonical traversal order of each kind's units), which no rule file carries yet."""

    def __init__(self, doc: dict[str, Any]) -> None:
        from e2e.population_v2 import CompactPopulation

        self.pop = CompactPopulation(doc)
        self.rank_layout = doc.get("gate_layout") == GATE_LAYOUT_RANK[0]
        self._kind_rows: dict[int, list[int]] = {}
        self._kind_prefix: dict[int, list[int]] = {}
        for i, r in enumerate(self.pop.rows):
            self._kind_rows.setdefault(r[3], []).append(i)
        for kid, idx in self._kind_rows.items():
            pref = [0]
            for i in idx:
                pref.append(pref[-1] + self.pop.rows[i][1])
            self._kind_prefix[kid] = pref
        self.kind_ids = {k.name: k.id for k in self.pop.kinds}

    def count(self, kind: str) -> int:
        return self._kind_prefix.get(self.kind_ids[kind], [0])[-1]

    def locate(self, rank: int) -> dict[str, Any]:
        import bisect

        row, j = self.pop._row(rank)
        kid = row[3]
        i = bisect.bisect_right(self._row_firsts(kid), row[0]) - 1  # which of the kind's rows (in rank order) holds this unit
        index = self._kind_prefix[kid][i] + j
        lo = row[4] + j * row[5]
        return {"rank": rank, "kind": self.pop.kinds[kid].name, "kind_id": kid, "index_within_kind": index, "ru": row[2],
                ("rank_interval" if self.rank_layout else "gate_interval"): [lo, lo + row[6]]}

    def _row_firsts(self, kid: int) -> list[int]:
        cache = getattr(self, "_firsts", None)
        if cache is None:
            cache = self._firsts = {}
        if kid not in cache:
            cache[kid] = [self.pop.rows[i][0] for i in self._kind_rows[kid]]
        return cache[kid]

    def rank_of(self, kind: str, index: int) -> int:
        import bisect

        kid = self.kind_ids[kind]
        pref = self._kind_prefix[kid]
        if not 0 <= index < pref[-1]:
            raise IndexError(f"{kind} has {pref[-1]} units")
        i = bisect.bisect_right(pref, index) - 1
        return self.pop.rows[self._kind_rows[kid][i]][0] + (index - pref[i])


# -- CLI ------------------------------------------------------------------------------------------------------------------------


# -- the verifier's derivation (V2.population_derived) ------------------------------------------------------------------------------
#
# Directive 2026-09-10 16:04 PT: "matching is necessary but insufficient: the verifier must derive the expected population from the
# pinned circuit and partition."  ``expected_population`` derives, from the pinned instance alone (program identity = workload shape,
# plan name = token-selection form and input treatment), the kind table (kind -> count, declared width), N_V per RU, the RU count and the
# kind-table digest -- by rule: no plan is built, no prover file is read.  ``compare`` names every field of a population file that
# disagrees with it.  What is derived and what can only be compared against the file's own content is stated in ``derivable`` /
# ``not_derivable`` (and in data/contract/a1-circuit/for-a3.md).

EXPECTED_SCHEMA = "veritor/a1/expected-population/v1"
POPULATION_V2_SCHEMA = "veritor/population/v2"
A9_POPULATION_V5 = "veritor/a9/population-v5/v1"
A9_PARTITION_DIGEST_KIND = "a9/legacy-minus-inputs/v1"
A9_PARTITION_DIGEST_KIND_V2 = "a9/legacy-minus-inputs/v2"  # [E6 2026-09-11] e2e.partitions.legacy_minus_inputs_v2
PLAN_NAME_P1 = "P1/cu=whole"
# Declared return widths of the registered *functions* the population's callee kinds stop at (``population_v2._callee_out_bits``: the
# callee's returns summed; a primitive's width).  Function identities do not depend on the shape, so the table is pinned here; it is
# MEASURED equal to the kind entries of the validated TINY (4x8 SMALL) and 32x15 plans (tests/contract/a1_expected_population_test.py).
DECLARED_CALLEE_BITS: dict[str, int] = {
    "AmpereBF16TcDot16_v1|batch": 32, "ArgmaxStep_v1|scan": 112, "Bf16AddF2fp_v1|batch": 16, "Bf16ToF32_v1|batch": 32,
    "DotBf16_v1{K=64}|call": 32, "F2fpBf16_v1|batch": 16, "F32MulFtz_v1|batch": 32, "F32Mul_v1|batch": 32, "GatherBf16x49152_v1|batch": 16,
    "GemmCoordinate_v1{K=1536}|batch": 16, "GemmCoordinate_v1{K=576}|batch": 16, "RopeOutAdd_v1|batch": 16, "RopeOut_v1|batch": 16,
    "SiluMulBf16_v1|batch": 16, TOKEN_SELECT_KIND: TOKEN_SELECT_CONTAINER_BITS,
}
DECLARED_CALLEE_BITS_SHA256 = _digest(DECLARED_CALLEE_BITS)


def declared_width(kind: str) -> int:
    """The declared width of a population kind, by rule: a scalar primitive from the registry (``Name_vN|primitive``, ``Const<bits>[..]``),
    a callee kind from :data:`DECLARED_CALLEE_BITS`.  Refuses a kind with no declared width (it is not a kind of this program)."""

    m = re.match(r"^Const(?P<bits>\d+)\[0x[0-9a-fA-F]+\]_v\d+\|primitive$", kind)
    if m:
        return int(m.group("bits"))
    w = declared_primitive_width(kind)
    if w is not None:
        return w
    if kind in DECLARED_CALLEE_BITS:
        return DECLARED_CALLEE_BITS[kind]
    # [E6 NIGHT 2026-09-12] shape-parametric callee kinds of Serve_v1 (another checkpoint changes K, V -- not the function): the declared width is
    # the function's return width for ANY parameter value (GemmCoordinate<K>: one BF16 coordinate, K a multiple of 16 -- programs.py:92; Gather<V>:
    # one BF16 row word); a `Name_vN|batch` of a registered scalar primitive (Bf16Add: the QKV bias add) has the primitive's width.
    m = _GEMM_COORD_KIND.match(kind)
    if m:
        if int(m.group("k")) % 16:
            raise SystemExit(f"REFUSED: {kind}: K = {m.group('k')} is not a multiple of 16 (the tensor-core coordinate's step)")
        return 16
    if _GATHER_KIND.match(kind):
        return 16
    m = _BATCH_PRIM_KIND.match(kind)
    if m:
        w = declared_primitive_width(f"{m.group('name')}_v{m.group('ver')}|primitive")
        if w is not None:
            return w
    raise SystemExit(f"REFUSED: no declared width by rule for kind {kind!r}")


_GEMM_COORD_KIND = re.compile(r"^GemmCoordinate_v1\{K=(?P<k>\d+)\}\|batch$")
_GATHER_KIND = re.compile(r"^GatherBf16x(?P<v>\d+)_v1\|batch$")
_BATCH_PRIM_KIND = re.compile(r"^(?P<name>[A-Za-z0-9]+)_v(?P<ver>\d+)\|batch$")


class ExpectedKind(dict):
    """``{"kind","fn","form","count","declared_bits","offset"}`` (+ ``index_bits`` for the token-selection unit)."""


class ExpectedPopulation(dict):
    """The population the verifier expects, derived by rule from the pinned instance (see module section).  A plain JSON-able dict with
    attribute access; ``kinds`` are sorted by kind name and carry only kinds with ``count > 0``."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    @property
    def kind_by_name(self) -> dict[str, ExpectedKind]:
        return {k["kind"]: k for k in self["kinds"]}

    def to_json(self) -> dict[str, Any]:
        return json.loads(json.dumps(self))


class Disagreement(dict):
    """``{"field","expected","actual","note"}``: one field of a population file that is not the derived one."""

    def __init__(self, field: str, expected: Any, actual: Any, note: str) -> None:
        super().__init__(field=field, expected=expected, actual=actual, note=note)

    @property
    def field(self) -> str:
        return self["field"]


def pinned_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Normalize a pinned instance: ``veritor/protocol-instance/v1`` (A3/A9's ``protocol_instance.json``: ``program.identity.{case, LP,
    STEPS, config_sha12}``, ``partition.plan_name``, ``partition.instances``) or a flat ``{"case","lp","steps","plan_name"[,"instances"]}``."""

    if "program" in instance and isinstance(instance["program"], dict) and "identity" in instance["program"]:
        ident = instance["program"]["identity"]
        part = instance.get("partition") or {}
        return {"case": ident.get("case", "B0"), "lp": int(ident["LP"]), "steps": int(ident["STEPS"]), "config_sha12": ident.get("config_sha12", B0_CONFIG_SHA12),
                "family": ident.get("family", "Serve_v1"), "plan_name": part.get("plan_name", PLAN_NAME_P1), "instances": part.get("instances"),
                "count_basis": part.get("count_basis"), "partition_digest_kind": part.get("partition_digest_kind"),
                "attention": ident.get("attention"), "function": ident.get("function"),  # [D57] declared geometry (None = legacy) + the bound id
                "source": instance.get("schema", "protocol-instance")}
    for k in ("case", "lp", "steps"):
        if k not in instance:
            raise SystemExit(f"REFUSED: pinned instance lacks {k!r} (program.identity or flat case/lp/steps)")
    return {"case": instance["case"], "lp": int(instance["lp"]), "steps": int(instance["steps"]), "config_sha12": instance.get("config_sha12", B0_CONFIG_SHA12),
            "family": instance.get("family", "Serve_v1"), "plan_name": instance.get("plan_name", PLAN_NAME_P1), "instances": instance.get("instances"),
            "count_basis": instance.get("count_basis"), "partition_digest_kind": instance.get("partition_digest_kind"),
            "attention": instance.get("attention"), "function": instance.get("function"), "source": "flat"}


# [E6 2026-09-11] literal (Const32) treatment -- handoff fixed decision, E5 decides compile-vs-retain per the rule and hands E6 the consequence:
#   "retained": every Const<bits>[..] gate is a VU of its own kind (today's rule; A9 attempt 6; the P1 plans)
#   "compiled": the literal is folded into its consumer's prescribed function (no VU, no prescribed input; the value is pinned in the
#               consumer's checker) -- the Const kinds leave the population and the identity changes ONCE, coherently
LITERALS_FORMS = ("retained", "compiled")
_CONST_KIND = re.compile(r"^Const(?P<bits>\d+)\[(?P<hex>0x[0-9a-fA-F]+)\]_v\d+\|primitive$")


def is_literal_kind(kind: str) -> bool:
    return _CONST_KIND.match(kind) is not None


def apply_literals(counts: dict[str, int], literals: str) -> dict[str, Any]:
    """Apply the literal treatment to a computed kind table **in place**; returns the ``literals`` record.  ``retained`` changes nothing;
    ``compiled`` removes every Const kind and records what left (kind -> units, total)."""

    if literals not in LITERALS_FORMS:
        raise SystemExit(f"REFUSED: literals {literals!r} not in {LITERALS_FORMS}")
    consts = {k: n for k, n in counts.items() if is_literal_kind(k)}
    if literals == "retained":
        return {"treatment": "retained", "kinds": consts, "vus": sum(consts.values()),
                "rule": "every Const<bits>[value] gate is a VU of its own kind (PROTOCOL-CONFORMANCE OR1: a literal IS a VU until E5 rules compile)"}
    for k in consts:
        del counts[k]
    return {"treatment": "compiled", "kinds": consts, "vus": 0, "removed_vus": sum(consts.values()),
            "rule": "the literal is part of its consumer's prescribed function: no VU, no prescribed input; its value is pinned by the consumer's "
                    "checker (E5 ck_select / ck_ftz) -- decided by E5 per the handoff rule, applied here once"}


def expected_population(instance: dict[str, Any], *, token_select: str | None = None, literals: str | None = None) -> ExpectedPopulation:
    """Derive the expected population **by rule** from the pinned instance alone.  Builds no plan, reads no prover file; reads only the
    committed census rules (sha pinned) and the registry.  ``token_select`` defaults from the plan name (``/token-select=whole-scan`` ->
    the ruled form; a bare ``P1/cu=whole`` -> the P1 ``argmax-scan`` form the validated plans and A9's records carry); ``literals``
    (``retained`` | ``compiled``) likewise defaults from the plan name (``/literals=compiled``), else ``retained``.  Refuses a case
    without rules (only B0 has L5's rules) or a plan that is not ``P1/cu=whole``.  [E6 2026-09-11] a non-Serve_v1 identity is routed to
    :func:`expected_population_generic` (any registered ``verity.ir`` composite; no vLLM assumption)."""

    if _is_generic_identity(instance):
        return expected_population_generic(instance)
    pin = pinned_instance(instance)
    plan_name = str(pin["plan_name"])
    if not plan_name.startswith(PLAN_NAME_P1):
        raise SystemExit(f"REFUSED: plan {plan_name!r} is not {PLAN_NAME_P1}: the rules describe the P1/cu=whole partition only")
    if token_select is None:
        token_select = "whole-scan" if "token-select=whole-scan" in plan_name else "argmax-scan"
    if literals is None:
        literals = "compiled" if "literals=compiled" in plan_name else "retained"
    if pin["case"] != "B0":
        # [E6 NIGHT 2026-09-12] no L5 rules for this checkpoint: the kind table comes from the P1/cu=whole plan of the pinned program itself -- which
        # exists only for a registry case (b1.<case>_CONFIG); anything else is refused by name
        from verity_vllm.registry import b1

        if not hasattr(b1, f"{pin['case']}_CONFIG"):
            raise SystemExit(f"REFUSED: no census rules for case {pin['case']!r} and no registry config b1.{pin['case']}_CONFIG to derive its P1 kind table from")
        return expected_population_serve_plan(pin, plan_name=plan_name, token_select=token_select, literals=literals)
    lp, steps = pin["lp"], pin["steps"]
    rules, sha = load_rules()
    from e2e.population import attention_kblock_n

    bn = attention_kblock_n(pin.get("attention"))  # [D57] the identity's geometry; legacy 64 when it carries none
    counts, prescribed, _ = computed_kind_counts(rules, lp, steps, token_select=token_select, bn=bn)
    literals_rec = apply_literals(counts, literals)
    kinds: list[ExpectedKind] = []
    offset = 0
    for name in sorted(counts):
        n = counts[name]
        if n == 0:
            continue
        fn, form = name.split("|", 1)
        k = ExpectedKind(kind=name, fn=fn, form=form, count=n, declared_bits=declared_width(name), offset=offset)
        if name == TOKEN_SELECT_KIND:
            k["index_bits"] = TOKEN_SELECT_INDEX_BITS
        kinds.append(k)
        offset += n
    ktd = kind_table_digest(counts, prescribed, lp, steps, literals=literals)
    identity = {"family": pin["family"], "config_sha12": pin["config_sha12"], "LP": lp, "STEPS": steps,
                "function": identity_function(pin, lp, steps), "identity_digest": program_identity_digest(lp, steps, pin["config_sha12"], pin["family"], pin.get("attention"))}
    if pin.get("attention"):
        identity["attention"] = dict(pin["attention"]); identity["rule_basis_kblock_n"] = bn
    return ExpectedPopulation(
        schema=EXPECTED_SCHEMA, case=pin["case"], lp=lp, steps=steps, nout=steps + 1, program_identity=identity, plan_name=plan_name,
        inputs="excluded", inputs_rule="Input gates are in no VU and no RU: prescribed (A1) or legacy-minus-inputs (A9) -- the same computed kind table",
        token_select=token_select, literals=literals_rec, kinds=kinds, vu_count_per_ru=offset, ru_count_per_request=1, instances=pin["instances"],
        prescribed_inputs={"gates": prescribed["gates"], "prompt": prescribed["prompt_token_ids"]["gates"], "weights": prescribed["weights"]["gates"],
                           "weights_consumed": prescribed["weights"]["consumed"], "weights_unconsumed_rope_rows": prescribed["weights"]["unconsumed_rope_rows"]},
        kind_table_digest=ktd, kind_table_digest_kind=KIND_TABLE_DIGEST,
        derivable={"kind_table": "kind -> count, declared width (rules at (LP, STEPS) + registry / DECLARED_CALLEE_BITS)",
                   "vu_count_per_ru": offset, "ru_count_per_request": 1, "prescribed_inputs": "prompt = LP, weights = consumed + unconsumed rope rows",
                   "literals": literals, KIND_TABLE_DIGEST: ktd},
        not_derivable={"partition-intervals/v1": "hashes every VU's gate interval: deriving it materializes the plan; the verifier checks that the file's "
                                                 "own rows re-derive it (O(N_V) sha updates; compare(self_check='full')) and that the rows' kind table is the derived one",
                       A9_PARTITION_DIGEST_KIND: "sha256(basis P1 digest, excluded kinds, the retained rows): both need the plan; the verifier checks that the "
                                                 "file re-derives it from its own basis digest + rows (record_v5.population_v5._partition_digest) and the derived kind table",
                       A9_PARTITION_DIGEST_KIND_V2: "sha256(rows' v1 digest, DERIVED kind-table digest, frozen coverage-table sha, literals, token_select): the rows member is the "
                                                    "file's own re-derivation, every other member is pinned or derived (e2e.partitions.legacy_minus_inputs_v2)",
                       "content_sha256": "the file's bytes; self-consistency only", "plan_digest": "plan-rule/v1 over the plan's runs; compared only",
                       "gates": "no rule for the program's total gate count in L5's census; compared only",
                       "kinds[*].size": "gates per unit is a plan fact; compared only"},
        rules={"source": str(CENSUS_6500.relative_to(ROOT)), "sha256": sha, "basis": "census_v2.small_scaling.terms (L5, read-only); 26 census points + validated 32x15 holdout",
               "declared_callee_bits_sha256": DECLARED_CALLEE_BITS_SHA256},
        label="MODELED (rule-generated from the pinned instance; no plan built, no prover file read)")


# -- [E6 NIGHT 2026-09-12, G3] a Serve_v1 case WITHOUT L5's census rules (another checkpoint): the kind table from the program's own plan ---------
#
# L5's rules are fitted to B0 (SmolLM2-135M) only.  For any other ``b1.<case>_CONFIG`` the population is derived from the pinned program itself:
# the P1/cu=whole plan is built as UnitRuns (``e2e.population.build`` + ``make_plan``: the constructors A9/E10 built every TINY population with) and
# the kind table is read off the stopped nodes, O(runs), never per unit (``scripts/compact_p1_validate.kind_counts``).  Laptop cost MEASURED
# 2026-09-12: B0 4x8 5.9 s / 0.36 GB, SmolLM2-360M 4x8 ~7 s / 0.6 GB, Qwen2.5-0.5B 4x8 ~6 s / 0.45 GB.  Control: on B0 the plan's kind table
# equals the rule's kind for kind (7,922,613 computed units; tests/contract/e6_population_test.py).  The same transforms then apply (inputs
# prescribed, literals per E5's decision, token selection per Amendment A2 decision 4), so the digest members are the same functions.

SERVE_PLAN_RULE = ("kind table = units per (stopped-node callee, form) of the P1/cu=whole UnitRuns plan of bind(b1.Serve, C=b1.<case>_CONFIG, LP, STEPS) "
                   "(e2e.population.make_plan; compact_p1_validate.kind_counts), Input units prescribed (in no VU), literals and token selection as for B0")


@functools.lru_cache(maxsize=8)
def serve_plan_kind_table(case: str, lp: int, steps: int, family: str | None = None, attention_json: str | None = None) -> dict[str, Any]:
    """The P1/cu=whole plan's kind table of ``Serve_v1{C=b1.<case>_CONFIG, LP, STEPS}``: ``{"counts": kind -> units (Input and literal kinds
    included), "widths": kind -> declared out bits (population_v2._callee_out_bits of the stopped node), "program_digest", "gates", "input_gates",
    "plan_digest", "plan_digest_kind", "function", "config", "units_total", "runs", "seconds", "peak_rss_gb"}``.  MEASURED: the plan is built
    in-process (no validate_plan here; the rows pipeline -- freeze_serve -- carries the plan's own partition digest)."""

    import resource
    import time as _time

    from e2e import population as popmod
    from e2e.population_v2 import _callee_out_bits
    from verity_vllm.registry import b1
    from veritor.plans.runs import UnitRuns

    if not hasattr(b1, f"{case}_CONFIG"):
        raise SystemExit(f"REFUSED: registry has no {case}_CONFIG (verity_vllm/registry/b1.py): the pinned case must be a registered static configuration")
    config = dict(getattr(b1, f"{case}_CONFIG"))
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
    t0 = _time.perf_counter()
    # [E7 D54/D57] family: Serve_v1 (default) | Serve_v4 with the identity's declared geometry (JSON string: the lru_cache key must be hashable)
    program, adapted, circuit, pdigest = popmod.build(config, lp, steps, family=family, attention=json.loads(attention_json) if attention_json else None)
    plan, meta = popmod.make_plan(PLAN_NAME_P1, adapted, circuit)
    vus = plan.verification_units
    if not isinstance(vus, UnitRuns):
        raise SystemExit("REFUSED: the P1/cu=whole plan is not UnitRuns (per-unit materialisation is the laptop rule's forbidden path)")
    counts: dict[str, int] = {}
    widths: dict[str, int] = {}
    for run in vus.runs:
        node = circuit.scope_at(run.path).fn.nodes[run.node_idx]
        kind = f"{node.fn.name}|{'batch' if run.unrolled else node.form}"
        counts[kind] = counts.get(kind, 0) + run.count
        w = int(_callee_out_bits(node.fn))
        if widths.setdefault(kind, w) != w and node.fn.name != "Input":  # Input units: prompt tokens I32, weights BF16 -- prescribed, never VUs
            raise SystemExit(f"REFUSED: kind {kind} has two declared widths in the plan ({widths[kind]}, {w})")
    if len(plan.commitment_units) != 1:
        raise SystemExit(f"REFUSED: P1/cu=whole has {len(plan.commitment_units)} commitment units, not one RU per request")
    # no plan_digest here: ``plan.to_json()`` over ~2 M runs costs > 1 GB (MEASURED 2026-09-12: SmolLM2-360M 4x8 +1.1 GB); the rows pipeline
    # (``freeze_serve`` / the two-phase build) derives plan-rule/v1 in its own process and the instance pins it from there
    return {"case": case, "LP": lp, "STEPS": steps, "config": config, "function": getattr(getattr(program, "spec", None), "id", None),
            "program_digest": pdigest, "gates": int(program.gates), "input_gates": int(program.input_gates),
            "plan_digest": None, "plan_digest_kind": None, "plan_name": PLAN_NAME_P1, "meta": meta,
            "counts": dict(sorted(counts.items())), "widths": widths, "units_total": len(vus), "runs": len(vus.runs),
            "seconds": round(_time.perf_counter() - t0, 2), "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2),
            "label": "MEASURED (P1/cu=whole UnitRuns built from the pinned program in-process; kind table O(runs); plan NOT validated here)"}


def expected_population_serve_plan(pin: dict[str, Any], *, plan_name: str, token_select: str, literals: str) -> ExpectedPopulation:
    """:func:`expected_population` for a Serve_v1 case without L5 rules -- the kind table from :func:`serve_plan_kind_table`, then the SAME
    transforms as the B0 rule: ``Input|batch`` prescribed (prompt = LP, weights = the rest; unconsumed rope rows = (MAX_POS - P) * D), literals
    per E5's decision, token selection regrouped to NOUT whole-scan units over the program's own V.  Every kind's declared width (``declared_width``,
    shape-parametric) must equal the plan's; the kind-table digest is the same function as B0's."""

    case, lp, steps = pin["case"], pin["lp"], pin["steps"]
    kt = serve_plan_kind_table(case, lp, steps, pin.get("family"),  # [E7 D54/D57] the pin's family + declared geometry
                               json.dumps(pin["attention"], sort_keys=True) if pin.get("attention") else None)
    cfg = kt["config"]
    V, D, max_pos = int(cfg["V"]), int(cfg["D"]), int(cfg["MAX_POS"])
    counts = {k: n for k, n in kt["counts"].items() if not k.startswith("Input")}
    inputs = sum(n for k, n in kt["counts"].items() if k.startswith("Input"))
    if kt["counts"].get("Input|batch", 0) != inputs or inputs != kt["input_gates"]:
        raise SystemExit(f"REFUSED: Input units {inputs} != the program's input gates {kt['input_gates']} (every input gate is one prescribed unit under P1)")
    if inputs <= lp:
        raise SystemExit(f"REFUSED: {inputs} input units cannot hold LP = {lp} prompt tokens plus weights")
    weights = inputs - lp
    unconsumed = (max_pos - (lp + steps)) * D
    if unconsumed < 0 or unconsumed > weights:
        raise SystemExit(f"REFUSED: unconsumed rope rows (MAX_POS - P) * D = {unconsumed} outside [0, {weights}]")
    prescribed = {
        "gates": inputs,
        "prompt_token_ids": {"gates": lp, "container_bits": 32, "domain_bits": (V - 1).bit_length(), "domain": f"vocabulary index, V = {V:,} ({cfg['model']})",
                             "rule": "Input|batch|prompt = LP"},
        "weights": {"gates": weights, "bits": 16, "consumed": weights - unconsumed, "unconsumed_rope_rows": unconsumed,
                    "rule": "Input|batch units - LP; unconsumed = the rope cos/sin rows at positions >= P = (MAX_POS - P) * D (read by nothing; = L5's rule on B0)"},
        "seed": {"gates": 0, "note": "greedy decoding: TokenSelect is deterministic, no seed gate exists in Serve_v1"},
        "treatment": "prescribed: in no VU and no RU; values checked against the declared input unconditionally (contract step 2), never sampled",
    }
    src = f"P1/cu=whole plan of {kt['function']} ({case} {cfg['model']})"
    if token_select == "whole-scan":
        prescribed["token_selection"] = regroup_token_selection(counts, steps + 1, source=src, vocab=V)
    elif token_select == "argmax-scan":
        prescribed["token_selection"] = {"form": "argmax-scan", "kind": ARGMAX_KIND, "units": counts.get(ARGMAX_KIND, 0),
                                         "note": "P1 form: one VU per scan step (80-bit carries); NOT the ruled form (Amendment A2 decision 4)"}
    else:
        raise SystemExit(f"REFUSED: token_select {token_select!r} not in {TOKEN_SELECT_FORMS}")
    literals_rec = apply_literals(counts, literals)
    kinds: list[ExpectedKind] = []
    offset = 0
    for name in sorted(counts):
        n = counts[name]
        if n == 0:
            continue
        fn, form = name.split("|", 1)
        declared = declared_width(name)
        measured = TOKEN_SELECT_CONTAINER_BITS if name == TOKEN_SELECT_KIND else kt["widths"].get(name)
        if measured is not None and measured != declared:
            raise SystemExit(f"REFUSED: kind {name}: declared width by rule {declared} != the plan's callee width {measured}")
        k = ExpectedKind(kind=name, fn=fn, form=form, count=n, declared_bits=declared, offset=offset)
        if name == TOKEN_SELECT_KIND:
            k["index_bits"] = TOKEN_SELECT_INDEX_BITS
        kinds.append(k)
        offset += n
    ktd = kind_table_digest(counts, prescribed, lp, steps, literals=literals)
    identity = {"family": pin["family"], "config_sha12": pin["config_sha12"], "case": case, "LP": lp, "STEPS": steps,
                "function": kt["function"] if pin.get("attention") and not pin.get("function") else identity_function(pin, lp, steps),
                "identity_digest": program_identity_digest(lp, steps, pin["config_sha12"], pin["family"], pin.get("attention"))}
    if pin.get("attention"):
        identity["attention"] = dict(pin["attention"])
    if kt["function"] and kt["function"] != identity["function"]:
        raise SystemExit(f"REFUSED: pinned identity {identity['function']!r} != the registry's bound id {kt['function']!r} (config_sha12 names a different static configuration)")
    return ExpectedPopulation(
        schema=EXPECTED_SCHEMA, case=case, lp=lp, steps=steps, nout=steps + 1, program_identity=identity, plan_name=plan_name,
        inputs="excluded", inputs_rule="Input gates are in no VU and no RU: prescribed (A1) or legacy-minus-inputs (A9) -- the same computed kind table",
        token_select=token_select, literals=literals_rec, kinds=kinds, vu_count_per_ru=offset, ru_count_per_request=1, instances=pin["instances"],
        prescribed_inputs={"gates": prescribed["gates"], "prompt": lp, "weights": weights, "weights_consumed": weights - unconsumed, "weights_unconsumed_rope_rows": unconsumed},
        kind_table_digest=ktd, kind_table_digest_kind=KIND_TABLE_DIGEST,
        program={"digest": kt["program_digest"], "gates": kt["gates"], "input_gates": kt["input_gates"],
                 "model": cfg["model"], "revision": cfg.get("revision"), "label": "MEASURED (built in-process from the pinned identity)"},
        derivable={"kind_table": "kind -> count, declared width (P1/cu=whole UnitRuns of the pinned program + declared_width)", "vu_count_per_ru": offset,
                   "ru_count_per_request": 1, "prescribed_inputs": "prompt = LP, weights = Input units - LP; unconsumed rope rows = (MAX_POS - P) * D",
                   "literals": literals, KIND_TABLE_DIGEST: ktd, "program_digest": kt["program_digest"], "gates": kt["gates"]},
        not_derivable={"partition-intervals/v1": "hashes every VU's gate interval (O(N_V)); the rows pipeline (freeze_serve) derives it once; compare(self_check='full') re-derives",
                       "plan_digest": "plan-rule/v1 over the plan's runs (plan.to_json, > 1 GB at 2 M runs): derived once by the rows pipeline, pinned, compared only",
                       A9_PARTITION_DIGEST_KIND: "sha256(basis P1 digest, excluded kinds, the retained rows): needs the rows",
                       A9_PARTITION_DIGEST_KIND_V2: "sha256(rows' v1 digest, DERIVED kind-table digest, frozen coverage-table sha, literals, token_select): the rows member is the "
                                                    "file's own re-derivation, every other member is pinned or derived (e2e.partitions.legacy_minus_inputs_v2)",
                       "content_sha256": "the file's bytes; self-consistency only", "kinds[*].size": "gates per unit is a plan fact; compared only"},
        rules={"source": SERVE_PLAN_RULE, "sha256": kt["program_digest"], "basis": f"{src}: {kt['units_total']:,} units in {kt['runs']:,} runs, {kt['seconds']} s, "
                                                                                    f"{kt['peak_rss_gb']} GB peak RSS (no L5 rules for this case)",
               "declared_callee_bits_sha256": DECLARED_CALLEE_BITS_SHA256},
        label=f"MEASURED (plan-derived from the pinned program: {kt['label']})")


# -- [E6 2026-09-11] generic registered programs (no vLLM assumption) -----------------------------------------------------------------
#
# A pinned instance whose ``program.identity.family`` is not ``Serve_v1`` names a registered ``verity.ir`` composite (``family`` =
# ``REGISTRY`` id, ``params`` = static bindings, ``module`` = the module whose import registers it; e2e.protocol_instance.ProgramIdentity).
# The population is derived by walking the bound program's ROOT BODY BY NODE (never per gate):
#   * every root ``Input*`` node is a prescribed input collection (in no VU and no RU; checked against the declaration, never sampled);
#   * every root ``call`` node is one RU (S_R = one RU per root call), its gate interval the call's;
#   * inside an RU every computed gate is one VU (S_V = one VU per computed gate); its kind is ``<primitive id>|<form of the innermost
#     body node that emits it>`` (``batch`` / ``primitive`` / ``scan``), composites recursed by node -- so the count of a kind is a sum
#     of node sizes, O(nodes), not O(gates);
#   * a computed root node that is not a call (a gate in no RU) refuses the program: the rule has no RU for it.
# This reproduces E8's ``demo_core.partition`` (``DemoCore_v1{N,R}``: N_V = 2NR, N_R = R, kinds SiluMulBf16_v1|batch, Bf16AddF2fp_v1|batch)
# and is the rule any other small registered composite is checked by.  The program is BUILT (the identity is a registry binding, cheap
# for a small program) but never its plan or population; ``GENERIC_MAX_GATES`` refuses a program too large to carry a per-gate
# population file on the laptop (the vLLM programs never come here: they are Serve_v1 and derived from the census rules).

GENERIC_EXPECTED_SCHEMA = "veritor/e6/expected-population-generic/v1"
GENERIC_RULE_KIND = "e6/generic/gate-vu-call-ru/v1"
GENERIC_RULE = "S_R = one RU per root call node; S_V = one VU per computed gate of the RU; kind = <primitive id>|<innermost node form>; root Input nodes prescribed"
GENERIC_MAX_GATES = 1 << 24


def _is_generic_identity(instance: dict[str, Any]) -> bool:
    ident = (instance.get("program") or {}).get("identity") if isinstance(instance.get("program"), dict) else None
    ident = ident if ident is not None else instance
    fam = ident.get("family") if isinstance(ident, dict) else None
    return isinstance(fam, str) and fam != "Serve_v1" and ("params" in ident or "module" in ident)


def generic_identity(instance: dict[str, Any]) -> dict[str, Any]:
    """``{"family","params","module","function","case","config_sha12"}`` from a protocol instance or a flat identity dict."""

    ident = instance["program"]["identity"] if isinstance(instance.get("program"), dict) and "identity" in instance["program"] else instance
    for k in ("family", "params", "module"):
        if not ident.get(k):
            raise SystemExit(f"REFUSED: generic program identity lacks {k!r} (family = registry id, params = static bindings, module = registering module)")
    return {"family": ident["family"], "params": dict(ident["params"]), "module": ident["module"], "function": ident.get("function"),
            "case": ident.get("case"), "config_sha12": ident.get("config_sha12")}


def bind_generic(ident: dict[str, Any]):
    """``(SpecializedDefinition, Program, program_digest)`` of the identity: import the registering module, look the family up in the
    registry, bind the params.  Refuses when the bound id is not the pinned ``function`` or the program exceeds ``GENERIC_MAX_GATES``."""

    import importlib

    from verity.ir import Program, bind, encode_program, program_digest
    from verity.ir.defs import REGISTRY

    importlib.import_module(ident["module"])
    if ident["family"] not in REGISTRY:
        raise SystemExit(f"REFUSED: {ident['family']!r} is not registered after importing {ident['module']!r}")
    sp = bind(REGISTRY[ident["family"]], **ident["params"])
    if ident.get("function") and sp.id != ident["function"]:
        raise SystemExit(f"REFUSED: bind({ident['family']}, **{ident['params']}).id = {sp.id!r} != pinned function {ident['function']!r}")
    if sp.gates > GENERIC_MAX_GATES:
        raise SystemExit(f"REFUSED: {sp.id} has {sp.gates:,} gates > GENERIC_MAX_GATES {GENERIC_MAX_GATES:,} (laptop rule: no per-gate population of this size)")
    prog = Program(sp)
    return sp, prog, program_digest(encode_program(prog))


def _node_kind_counts(node, into: dict[str, int], widths: dict[str, int]) -> None:
    """Add the computed gates emitted by ``node`` to ``into`` (kind -> count), by node -- composites recursed, never per gate."""

    from verity.ir.defs import PrimitiveDefinition

    fn = node.fn
    if isinstance(fn, PrimitiveDefinition):
        kind = f"{fn.id}|{node.form}"
        into[kind] = into.get(kind, 0) + (node.n if node.form in ("batch", "scan") else 1)
        widths.setdefault(kind, int(fn.ret.w))
        return
    reps = node.n if node.form in ("batch", "scan") else 1
    sub: dict[str, int] = {}
    for inner in fn.body.nodes:
        _node_kind_counts(inner, sub, widths)
    for k, n in sub.items():
        into[k] = into.get(k, 0) + n * reps


def expected_population_generic(instance: dict[str, Any]) -> ExpectedPopulation:
    """The population of a generic registered program by the rule above (module section).

    **Count rule (INTEGRATOR D17, E8 21:56Z):** ``kinds[*].count`` is the kind's VU count **PER RU** -- the one semantics every reader of an
    ``ExpectedPopulation`` has (the Serve_v1 rule, ``population_derived_check`` / E9's ``V2.population_derived``, ``kind_table.inventory``):
    ``count = count_per_instance / ru_count_per_instance`` when the RUs are homogeneous (every RU has the same kind table -- then the division
    is exact by construction); ``count_per_instance`` (all RUs) is carried beside it.  A program whose RUs are NOT homogeneous has no single
    per-RU number: ``count`` is then the per-instance total, ``homogeneous_rus`` is False, and ``population_derived_check`` refuses the per-RU
    comparison by name (``compare_generic`` -- which compares per-RU kind tables from ``rus[*].kinds`` -- is the comparator for that program).
    ``rus`` lists every RU with its gate interval and kind table; ``prescribed_inputs`` the root Input collections by width."""

    ident = generic_identity(instance)
    sp, prog, pdig = bind_generic(ident)
    builder = (instance.get("program") or {}).get("builder") if isinstance(instance.get("program"), dict) else None
    if builder:  # [D1] the instance's builder pin must rebuild the same program the registry bind gives (fail-closed on either)
        from e2e import protocol_instance as pi

        try:
            prog, pdig_b = pi.build_pinned_program(builder, expected_digest=pdig)
        except pi.InstanceError as e:
            raise SystemExit(f"REFUSED: {e}") from e
        if pdig_b != pdig:
            raise SystemExit(f"REFUSED: program.builder rebuilt {pdig_b}, registry bind gives {pdig}")
    root = prog.circuit.root
    body = root.body
    inputs: list[dict[str, Any]] = []
    rus: list[dict[str, Any]] = []
    widths: dict[str, int] = {}
    totals: dict[str, int] = {}
    for k, node in enumerate(body.nodes):
        lo, hi = body.offsets[k], body.offsets[k + 1]
        if node.fn.id.startswith("Input"):
            w = int(node.fn.ret.w)
            inputs.append({"name": node.name, "kind": f"{node.fn.id}|{node.form}", "gates": hi - lo, "width": w, "gate_interval": [lo, hi]})
            continue
        if node.form != "call":
            raise SystemExit(f"REFUSED: root node {k} ({node.fn.id}|{node.form}, {node.name}) is a computed gate outside any call: the rule has no RU for it")
        per: dict[str, int] = {}
        _node_kind_counts(node, per, widths)
        if sum(per.values()) != hi - lo:
            raise SystemExit(f"REFUSED: RU {len(rus)} kind counts {sum(per.values())} != its gate interval size {hi - lo}")
        rus.append({"index": len(rus), "name": node.name or f"call{len(rus)}", "callee": node.fn.id, "gate_interval": [lo, hi], "gates": hi - lo,
                    "kinds": {kk: per[kk] for kk in sorted(per)}})
        for kk, n in per.items():
            totals[kk] = totals.get(kk, 0) + n
    if not rus:
        raise SystemExit("REFUSED: the program has no root call: no RU exists under the rule")
    # the RU's declared output tuple (committed at step 1 by the program's record): its gates' kinds are the boundary-producer kinds;
    # every other kind of the RU is interior (|returns| gates per RU -- small; the innermost node's form names the kind as above)
    circuit = prog.circuit
    by_lo = {int(c.lo): c for c in circuit.root_calls()}
    for r in rus:
        call = by_lo.get(r["gate_interval"][0])
        outs: dict[str, int] = {}
        for g in (call.returns if call is not None else ()):
            gate = circuit.gate(int(g))
            sc = gate.scope
            body_ = sc.spec.body
            form = body_.nodes[body_.node_at(int(g) - sc.offset)].form
            kk = f"{gate.prim.id}|{form}"
            outs[kk] = outs.get(kk, 0) + 1
        r["output_kinds"] = {kk: outs[kk] for kk in sorted(outs)}
        r["outputs"] = sum(outs.values())
    ru_output_kinds = sorted({kk for r in rus for kk in r["output_kinds"]})
    kinds: list[ExpectedKind] = []
    offset = 0
    for name in sorted(totals):
        fn, form = name.split("|", 1)
        kinds.append(ExpectedKind(kind=name, fn=fn, form=form, count=totals[name], declared_bits=widths[name], offset=offset,
                                  ru_output=(name in ru_output_kinds)))
        offset += totals[name]
    n_in = sum(i["gates"] for i in inputs)
    if n_in != prog.input_gates or offset + n_in != prog.gates:
        raise SystemExit(f"REFUSED: inputs {n_in} + computed {offset} != program gates {prog.gates} (input_gates {prog.input_gates})")
    homogeneous = len({json.dumps(r["kinds"], sort_keys=True) for r in rus}) == 1
    if homogeneous:  # D17: count PER RU (exact: every RU carries rus[0]["kinds"][kind] gates of the kind); the per-instance total beside it
        per_ru_kinds = rus[0]["kinds"]
        out_kinds: list[ExpectedKind] = []
        off = 0
        for k in kinds:
            out_kinds.append(ExpectedKind(kind=k["kind"], fn=k["fn"], form=k["form"], count=per_ru_kinds[k["kind"]], declared_bits=k["declared_bits"], offset=off,
                                          ru_output=k["ru_output"], count_per_instance=totals[k["kind"]],
                                          count_label="per RU (homogeneous RUs; x ru_count_per_instance = per instance)"))
            off += per_ru_kinds[k["kind"]]
        kinds = out_kinds
        assert all(k["count"] * len(rus) == k["count_per_instance"] for k in kinds) and off == rus[0]["gates"]
    else:
        kinds = [ExpectedKind(**{**k, "count_per_instance": totals[k["kind"]], "count_label": "per INSTANCE (RUs not homogeneous: no per-RU number; compare_generic per RU)"})
                 for k in kinds]
    # the rule digest is over the per-INSTANCE kind totals (independent of the count semantics chosen for `count`)
    rule_digest = _digest({"schema": GENERIC_RULE_KIND, "function": sp.id, "program_digest": pdig, "kinds": {k["kind"]: totals[k["kind"]] for k in kinds},
                           "rus": [[r["gate_interval"][0], r["gate_interval"][1], r["callee"]] for r in rus], "inputs": [[i["gate_interval"][0], i["gate_interval"][1], i["width"]] for i in inputs]})
    identity = {"family": ident["family"], "params": ident["params"], "module": ident["module"], "function": sp.id, "case": ident.get("case"),
                "config_sha12": ident.get("config_sha12"), "program_digest": pdig, "program_digest_kind": "verity.ir.codec.program_digest(encode_program(Program(bind(...))))"}
    return ExpectedPopulation(
        schema=GENERIC_EXPECTED_SCHEMA, program_identity=identity, plan_name=GENERIC_RULE_KIND, rule=GENERIC_RULE,
        inputs="excluded", inputs_rule="root Input collections are prescribed: in no VU and no RU; checked against the declared inputs, never sampled",
        kinds=kinds, rus=rus, ru_output_kinds=ru_output_kinds, vu_count_per_instance=offset, ru_count_per_instance=len(rus),
        vu_count_per_ru=(rus[0]["gates"] if homogeneous else None), ru_count_per_request=len(rus), homogeneous_rus=homogeneous, instances=None,
        gates=int(prog.gates), input_gates=int(prog.input_gates),
        prescribed_inputs={"gates": n_in, "collections": inputs, "treatment": "prescribed: in no VU and no RU; values checked against the declaration unconditionally, never sampled"},
        kind_table_digest=rule_digest, kind_table_digest_kind=GENERIC_RULE_KIND,
        derivable={"kind_table": "kind -> count PER RU when the RUs are homogeneous (D17; count_per_instance beside it), declared width (registry primitive return widths)",
                   "rus": "one per root call: name, callee, gate interval, kind table",
                   "prescribed_inputs": "root Input collections by width", "program_digest": pdig, GENERIC_RULE_KIND: rule_digest},
        not_derivable={"partition_digest (the program's own rule, e.g. demo-core/stage-ru-gate-vu/v1)": "sha256 over the producer's partition document; compared against "
                                                                                                        "the rule on N_V, N_R, kind counts, RU intervals (compare_generic)"},
        label="DERIVED (by rule from the registered program's body nodes; the program was built, no plan and no prover file read)")


def compare_generic(expected: ExpectedPopulation, partition_doc: dict[str, Any] | str | Path) -> list[Disagreement]:
    """Every field of a generic program's partition document that is not the rule's.  The document must carry ``program_digest``,
    ``N_V``, ``N_R``, ``kinds`` (kind -> count) and ``rus[*].gate_interval``; when it also carries ``vus[*]`` (``ru_index``, ``kind``)
    their per-RU kind tables are checked too (O(N_V), small by construction).  ``[]`` iff the partition IS the rule's population."""

    doc, _ = _load_population_doc(partition_doc)
    out: list[Disagreement] = []
    if doc.get("program_digest") != expected["program_identity"]["program_digest"]:
        out.append(Disagreement("program_digest", expected["program_identity"]["program_digest"], doc.get("program_digest"), "not a partition of THIS program"))
    if int(doc.get("N_V", -1)) != expected["vu_count_per_instance"]:
        out.append(Disagreement("N_V", expected["vu_count_per_instance"], doc.get("N_V"), "one VU per computed gate"))
    if int(doc.get("N_R", -1)) != expected["ru_count_per_instance"]:
        out.append(Disagreement("N_R", expected["ru_count_per_instance"], doc.get("N_R"), "one RU per root call"))
    exp = {k["kind"]: int(k.get("count_per_instance", k["count"])) for k in expected["kinds"]}  # E8's document counts per instance (D17: `count` is per RU)
    got = {str(k): int(v) for k, v in (doc.get("kinds") or {}).items()}
    for name in sorted(set(exp) | set(got)):
        if exp.get(name) != got.get(name):
            out.append(Disagreement(f"kinds[{name}]", exp.get(name), got.get(name), "kind count (per instance) differs from the rule"))
    rus = doc.get("rus") or []
    if len(rus) != len(expected["rus"]):
        out.append(Disagreement("rus", len(expected["rus"]), len(rus), "RU count differs"))
    for e, r in zip(expected["rus"], rus):
        if list(r.get("gate_interval") or []) != e["gate_interval"]:
            out.append(Disagreement(f"rus[{e['index']}].gate_interval", e["gate_interval"], r.get("gate_interval"), "the RU is not the root call's interval"))
    vus = doc.get("vus")
    if isinstance(vus, list) and vus:
        per: dict[int, dict[str, int]] = {}
        for v in vus:
            d = per.setdefault(int(v["ru_index"]), {})
            d[v["kind"]] = d.get(v["kind"], 0) + 1
        for e in expected["rus"]:
            if per.get(e["index"], {}) != e["kinds"]:
                out.append(Disagreement(f"rus[{e['index']}].kinds", e["kinds"], per.get(e["index"]), "this RU's VU kind table != the rule's"))
    return out


# -- [E6 2026-09-11] freezing an instance: TINY (Serve_v1 B0 4x8, one request) and CORE (E8's DemoCore_v1{N,R}) ----------------------------
#
# ``freeze_tiny`` / ``freeze_core`` derive the population by rule, build the family inventory and the coverage table (e2e.coverage_v1.kind_table),
# and write the pre-challenge ``protocol_instance.json`` with every identity pinned (e2e.protocol_instance.frozen_instance): program digest,
# partition digest (+ kind), plan digest, kind-table digest, the coverage-table sha, the interior-layout pin, the declared request(s).  Nothing
# here reads a prover file; the TINY numbers that are not derivable by rule (program digest and gate counts of the built program, the plan-rule
# digest, the v1 rows digest) are the MEASURED values of attempt 6 (out/records/V5-TINY-bex0-inc-elem-b8177939.md §2-3: the same program, the
# same plan, the same rows) and are labelled with that source.

TINY_PINS: dict[str, Any] = {
    "case": "B0", "lp": 4, "steps": 8, "model": "HuggingFaceTB/SmolLM2-135M", "max_tokens": 9,
    "program_digest": "b33a0463336bc2224225e2e43bb711aec1f2af63f831742a21d0d458de754cae",
    "gates": 242_675_685, "input_gates": 135_039_300,
    "plan_digest": "c4e04cd6134dec5a086e4883addd8204c2c321736cbf491a4382bd6dbd5685bd", "plan_digest_kind": "plan-rule/v1",
    "basis_partition_digest_with_inputs_prefix": "5d1c27cbf654a405",  # partition-intervals/v1 of the validated P1/cu=whole plan WITH inputs (population.compact.json)
    "rows_partition_digest_v1": "b956ffa121c4d2fdf6b41e595351f5bbbb400fa138e0eb80422397a7df9583a7",  # a9/legacy-minus-inputs/v1 over the ORIGINAL (retained, argmax-scan) rows
    "population_v1_content_sha256": "cfb2a1a5cd2443d0aa272fbff6f8c597c145af5b887eb34a0da6735bbf0704fc",  # /workspace/a9/pops/b0-4x8/population.v5.json
    # [D19 2026-09-11] the v1 digest of the rows AS THE SHIPPED FILE CARRIES THEM after legacy_minus_inputs_v2.from_v5 (Const rows dropped, ArgmaxStep scans
    # regrouped): v1_rows_digest(from_v5(original)) -- MEASURED by E10 on the pod and reproduced here on E10's file (sha256 78e59755...); keyed by the fixed
    # decisions.  `rederive`/`check` (the shipped verifier's reading, record_v5 load check="full") hash THESE rows, so the pin must too.
    "rows_partition_digest_v1_transformed": {"compiled/whole-scan": "c62a5d87424350625840d85aac59ac7912ba34cdffdefbad5dff22ba6630aaac"},
    "source": "out/records/V5-TINY-bex0-inc-elem-b8177939.md §2 (identities) / §3 (population) -- MEASURED on the pod, attempt 6 (same program, plan, rows)",
    "vocabulary": {"digest": "33ee156d86a1bb72a5f0918413baa9295b15a3d105553f553133007fc1d00623", "profile_name": "llama-eager-batch-invariant-v3",
                   "profile_sha256": "847913c5f8bd2a546e46b86a4b5579dd4eb596a9f4733d3fd687f74fc0260474", "schema": "verity-ir/registry-version/v0",
                   "source": "manifests/semantic-profiles/llama-eager-batch-invariant-v3.json (sha256 of the profile file; DERIVE-CORR s4/binding.json vocabulary digest); "
                             "as attempt 6 (out/records/V5-TINY-bex0-inc-elem-b8177939.md)"},
    "approved_elf_sha256": ["9fd00e60fb6a74762de987cde211aca9de4576de844569da779acac6bf3863db"],
    "family": "veritor.tc-ampere-bf16@2",
    "weights": {"expected_root_prefix": "f73fa155", "ids_digest": "0cd4977c75a7da4e18b88f9adb84702195b5308ea74e80abf396051b59a291ec", "leaf_count": 213_045,
                "checkpoint": "SmolLM2-135M @ 93efa2f0 (weights_authorized.json, authorized auditor-side on the pod before the audit; E10 re-authorizes and must reproduce root + ids_digest)"},
    "s": "1/16384", "s_source": "record_v5/config.py DEFAULT_S[TINY-B0-4x8] (L1's s = 2^-14)", "q": "1", "delta": "1/1000000000000",
}


def tiny_request(index: int = 0) -> dict[str, Any]:
    """The declared TINY request: row ``index`` of record_v5's deterministic workload (``base_request_id(model, prompt ids, max_tokens)``)."""

    from record_v5 import config as cfgmod, workload_v5 as wl5  # A9/E10's workload rule, read-only

    cfg = cfgmod.parse(cfgmod.default_doc("TINY-B0-4x8", B_ex=0, f_policy=cfgmod.F_POLICIES[0]))
    rows = wl5.request_rows(wl5.make_workload(cfg))
    return rows[index]


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_json(path: Path, doc: Any, *, canonical: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n" if canonical else json.dumps(doc, indent=1, sort_keys=True) + "\n"
    path.write_bytes(text.encode())
    return _sha256_hex(text.encode())


LITERALS_PIN_MODES = ("auto", "required", "off")


def literals_commitment(literal_kinds: dict[str, int], *, mode: str = "auto") -> tuple[dict[str, Any] | None, str]:
    """[D36] ``(commitments.literals, note)`` of the pinned program from the population's OWN literal kinds (``apply_literals`` record ``kinds``:
    ``Const<w>[0x…]_v1|primitive -> units``, derived from the same P1/cu=whole kind table as everything else): the literal ids are the kinds' callee names,
    the block is E9's ``LiteralTable(ids).pin()`` through :func:`protocol_instance.literals_pin`.  NOT ``LiteralTable.from_program`` here: that walks
    every gate through ``locator.literal_of`` (MEASURED 2026-09-12: 5.2 us/gate = 746 s on B0 1x0's 144.5 M gates; > 16 min, killed, on Qwen2.5-0.5B
    4x8's 510 M) -- the kind table already IS the set of Const primitives the program instantiates (every Const gate is one ``|primitive`` unit of the
    plan; Inputs are the only gates outside it), so the two derivations are equal by construction and MEASURED equal: on B0 1x0 ``LiteralTable.from_program
    (program).pin() == literals_pin(kind-table ids)`` (same 7 ids, same two roots; /tmp/e6pop/lit_eq.log 07:28Z).
    ``mode``: ``auto`` = pin when E9's class is on this checkout, else ``(None, why)`` by name; ``required`` = refuse instead; ``off`` = not requested."""

    from e2e import protocol_instance as pi

    if mode not in LITERALS_PIN_MODES:
        raise SystemExit(f"REFUSED: literals pin mode {mode!r} not in {LITERALS_PIN_MODES}")
    if mode == "off":
        return None, "commitments.literals not requested (--literals-pin off)"
    ids = literal_ids_of_kinds(literal_kinds)
    if not ids:
        raise SystemExit("REFUSED: the population's kind table has no Const<w>[0x…]_v1 literal kind; a Serve_v1 program always has (rmsnorm eps / N, the sampler's zero)")
    try:
        pin = pi.literals_pin(ids)
    except pi.InstanceError as e:
        if mode == "required":
            raise SystemExit(f"REFUSED: {e}")
        return None, f"commitments.literals NOT PINNED: {e}; re-pin on the merged tree (D36)"
    return pin, (f"commitments.literals pinned: {len(pin['literals'])} literal ids, {len(pin['trees'])} tree(s) {sorted(pin['trees'])} -- "
                 f"verifier_v3.steps.LiteralTable(<the kind table's Const ids>).pin() [D36]; == LiteralTable.from_program(program).pin() by construction")


def literal_ids_of_kinds(literal_kinds: dict[str, int]) -> list[str]:
    """The sorted, distinct ``Const<w>[0x…]_v<n>`` primitive ids named by the literal kinds ``Const…|primitive`` (the kind key's callee-definition part)."""

    ids = sorted({k.rsplit("|", 1)[0] for k in literal_kinds if is_literal_kind(k)})
    bad = [i for i in ids if not re.match(r"^Const\d+\[0x[0-9a-fA-F]+\]_v\d+$", i)]
    if bad:
        raise SystemExit(f"REFUSED: literal kinds that are not Const<w>[0x…]_v<n> primitives: {bad}")
    return ids


def freeze_tiny(out_dir: Path, *, literals: str = "retained", token_select: str = "argmax-scan", layout_digest: str | None = None, request_index: int = 0,
                inventory_dir: Path | None = None, measured_positions: Path | None = None, layout_rule: str | None = None,
                population_file: Path | None = None, coverage_table: Path | None = None, tables: bool = True, literals_pin: str = "auto") -> dict[str, Any]:
    """Freeze the TINY instance of the day: SMALL B0 at 4x8, ONE declared request, q = 1 for the principal run; population by rule; the
    coverage table and the interior-layout pin in ``commitments``; partition identity ``a9/legacy-minus-inputs/v2``.  Writes
    ``out_dir/{expected_population,inventory,coverage_table,protocol_instance,pins}.json``, ``inventory.md`` (+ copies to ``inventory_dir``)."""

    from e2e import protocol_instance as pi
    from e2e.coverage_v1 import kind_table as kt
    from e2e.partitions import legacy_minus_inputs_v2 as lmi2
    from record_v5 import config as cfgmod, instance_v5 as i5  # program identity by bind (A9's rule, read-only); the approved-guest rule

    T = TINY_PINS
    out_dir = Path(out_dir)
    plan_name = PLAN_NAME_P1 + ("/token-select=whole-scan" if token_select == "whole-scan" else "") + ("/literals=compiled" if literals == "compiled" else "")
    identity = i5.program_identity(T["case"], T["lp"], T["steps"])
    exp = expected_population({"program": {"identity": identity}, "partition": {"plan_name": plan_name, "instances": 1}}, token_select=token_select, literals=literals)
    layout_pin = pi.interior_layout_pin(digest=layout_digest, rule=layout_rule)  # E2's layout_pin() verbatim when e2e.interior.layout is importable, else placeholder
    layout_status = "pinned" if layout_pin["digest"] else "placeholder"
    # E3's streamed inventory of the record's positions (verifier_v3.locator.inventory), when given: folded per family + reconciled (D14)
    measured = json.loads(Path(measured_positions).read_text()) if measured_positions else None
    inv = kt.inventory(exp, population="TINY", layout_pin=layout_status, n_r=1, measured_positions=measured, tables_pinned=tables)
    live_table = kt.coverage_table(inv)
    if coverage_table is None:
        table, table_sha = live_table, _write_json(out_dir / "coverage_table.json", live_table, canonical=True)
        live_sha = table_sha
    else:
        # [D19] a coverage table FROZEN with an earlier freeze of this instance is kept as the pin (the population file the record ships names its sha);
        # it must describe exactly this population's kinds; the live table of the checkout is written beside it for the record (owners/approvals move later)
        raw = Path(coverage_table).read_bytes()
        table = json.loads(raw)
        if set(table.get("kinds") or {}) != set(live_table["kinds"]) or table.get("population") != "TINY" or table.get("schema") != live_table["schema"]:
            raise SystemExit(f"REFUSED: {coverage_table} does not describe this population's kinds ({len(table.get('kinds') or {})} vs {len(live_table['kinds'])})")
        table_sha = _write_json(out_dir / "coverage_table.json", table, canonical=True)
        if table_sha != _sha256_hex(raw):
            raise SystemExit(f"REFUSED: {coverage_table} is not in canonical form (sha {_sha256_hex(raw)[:16]} != canonical {table_sha[:16]})")
        live_sha = _sha256_hex((json.dumps(live_table, sort_keys=True, separators=(",", ":")) + "\n").encode())
        _write_json(out_dir / f"coverage_table.live-{live_sha[:8]}.json", live_table, canonical=True)
    ct = {"path": str(out_dir / "coverage_table.json"), "sha256": table_sha, "kinds": len(table["kinds"])}
    ct_prov = {} if live_sha == table_sha else {"frozen_from": str(coverage_table), "live_table_sha256_of_this_checkout": live_sha,
                                                "note": "pinned before the challenge; later approval/builder flips do not re-identify a frozen instance [D19]"}
    rows_v1 = _tiny_rows_digest_v1(exp, literals, token_select, population_file, table_sha)
    pdig = lmi2.partition_digest(rows_partition_digest_v1=rows_v1, kind_table_digest=exp["kind_table_digest"], coverage_table_sha256=table_sha,
                                 literals=literals, token_select=token_select)
    req = tiny_request(request_index)
    declared = pi.declared_inputs_section([req], source=f"record_v5/workload_v5 (TINY-B0-4x8) row {request_index}: the auditor's one request; greedy declared; seed / sampling params not pinned",
                                          pin_sampling_params=False, LP=T["lp"])
    n_v = exp["vu_count_per_ru"]
    src = T["source"]
    program = {"digest": T["program_digest"], "identity": identity, "input_inclusion": "gates_include_input_gates", "kind": "serve",
               "gates": {"value": T["gates"], "label": "MEASURED", "source": f"{src}: built program circuit size (input gates included)"},
               "input_gates": {"value": T["input_gates"], "label": "MEASURED", "source": f"{src}; = the rule's prescribed inputs {exp['prescribed_inputs']['gates']:,} (population_v3)"},
               "source": f"identity = bind(b1.Serve, C=b1.B0_CONFIG, LP=4, STEPS=8).id (record_v5/instance_v5.program_identity); digest {src}"}
    partition = {
        "plan_name": plan_name, "plan_descriptor_digest": T["plan_digest"], "plan_digest_kind": T["plan_digest_kind"], "vu_level": "coordinate", "cu_strategy": "whole",
        "validation": {"artifact": str(out_dir / "expected_population.json"), "program_digest": T["program_digest"], "plan_digest": T["plan_digest"], "partition_digest": pdig,
                       "status": "valid",
                       "source": f"rows: the BASIS plan's verdict (e2e.population_v2 build validated in-process, basis partition {T['basis_partition_digest_with_inputs_prefix']}, {src}); "
                                 f"kind table: DERIVED by rule from the pinned instance (population_v3.expected_population); the population is the rows minus the Input kinds "
                                 f"{'minus the Const kinds ' if literals == 'compiled' else ''}(a9/legacy-minus-inputs/v2, e2e.partitions.legacy_minus_inputs_v2)"},
        "vu_count": {"value": n_v, "label": "MEASURED", "source": f"population_v3.expected_population: rule kind table at (4, 8), literals={literals}, token_select={token_select}; "
                                                                  f"= attempt 6's rows count 7,922,613 when retained/argmax-scan ({src})"},
        "ru_count": {"value": 1, "label": "MEASURED", "source": "P1/cu=whole: one RU per request (population_v3.expected_population ru_count_per_request)"},
        "count_basis": "validated_plan_minus_inputs", "address_map": None, "instances": 1,
        "partition_digest": pdig, "partition_digest_kind": lmi2.KIND,
        "source": f"{lmi2.KIND} = sha256(rows' v1 digest {rows_v1[:16]} (v1_rows_digest over the rows the shipped file carries after from_v5 [D19]; original v1 file "
                  f"{T['rows_partition_digest_v1'][:16]}), kind-table digest {exp['kind_table_digest'][:16]} ({KIND_TABLE_DIGEST}), coverage table sha {table_sha[:16]} "
                  f"({ct['path']}), literals={literals}, token_select={token_select}) -- e2e.partitions.legacy_minus_inputs_v2; a table row flip or a fixed-decision change is a new identity",
    }
    outputs = [{"name": "token_slot", "multiplicity": T["steps"] + 1, "domain_bits_declared": 16, "wire_bits": 32,
                "enforcement": "declared_domain 16 on sampler.select; check_output_domains on every request",
                "source": f"Serve_v1 STEPS+1 = {T['steps'] + 1} root output slots per request; OUT16 (GUIDANCE §32)"}]
    # [D23] the four MUFU table roots (E2 table-layout/v1): rebuilt here, check_tables == [] (pi.table_pins raises otherwise)
    table_pins = pi.table_pins() if tables else None
    lit_pin, lit_note = literals_commitment(exp["literals"]["kinds"], mode=literals_pin)
    commitments = {"interior_layout": layout_pin,
                   **({"tables": table_pins} if table_pins else {}),
                   **({"literals": lit_pin} if lit_pin else {}),
                   "coverage_table": {**ct, "source": "e2e.coverage_v1.kind_table.coverage_table (support / producer / position per kind; E6 inventory) -- frozen with the instance"
                                                     + (f"; kept from the earlier freeze {Path(coverage_table).name} (D19: pinned before the challenge; this checkout's live table sha "
                                                        f"{live_sha[:16]} written beside it)" if ct_prov else "")},
                   "source": "E6 2026-09-11: pins fixed before the first commitment; V3.interior_layout fails closed while the layout pin is a placeholder; "
                             "[D36] V3.literal_table fails closed (LITERAL_UNPINNED) on a selected literal VU while commitments.literals is absent"}
    notes = [f"pre-challenge instance written by e2e.population_v3.freeze_tiny; {i5.CONSTRUCTION_NOTE}", "basis=rows",
             "loaded by record_v5/instance_v5.load_instance_v5: validate_instance's rules are all applied; exactly the enumerated accepted violations are admitted",
             f"TINY of EOD 2026-09-11: ONE declared request ({req['request_id']}, prompt {req['prompt_token_ids']}); q = 1 for the principal run (|J| = 1); "
             f"literals={literals}; token_select={token_select}; same operations and numerical semantics as attempt 6 (b8177939)",
             "F policy: gamble (B_ex = 0 principal run); randomness mode: production; serving: batch-invariant, greedy, prefix caching off",
             f"interior layout pin: {('pinned ' + str(layout_pin['digest']) + ' (' + str(layout_pin.get('rule') or 'rule unnamed') + ')') if layout_pin['digest'] else 'PLACEHOLDER (awaiting E2 layout_digest)'}",
             lit_note]
    tables_id = ("-t" + _digest({k: v["root"] for k, v in table_pins.items()})[:8]) if table_pins else ""
    tables_id += ("-l" + _digest({k: v["root"] for k, v in lit_pin["trees"].items()})[:8]) if lit_pin else ""
    # [E6 NIGHT 2026-09-12, E10 05:40Z blocker] the approved guest set is record_v5.config's RULE (every `reproduced` guest of e2e/approved_guests.json listing the epoch's
    # family), not attempt 6's literal ["9fd00e60"]: the night host embeds 666e937f (the same eight families + veritor.ftz-f32@1) and steps.guest_acceptance refuses a
    # guest outside the pinned set; the rule yields both (and 73081b7b) -- MEASURED: ['666e937f', '73081b7b', '9fd00e60']
    elfs = cfgmod.approved_elf_sha256(T["family"])
    if not set(T["approved_elf_sha256"]) <= set(elfs):
        raise SystemExit(f"REFUSED: the approved-guest rule {[e[:8] for e in elfs]} does not contain attempt 6's pinned guest {[e[:8] for e in T['approved_elf_sha256']]}")
    notes.append(f"approved_elf_sha256 = record_v5.config.approved_elf_sha256({T['family']!r}) = {[e[:8] + '…' for e in elfs]} (every reproduced guest of e2e/approved_guests.json "
                 "with the epoch's family; attempt 6 pinned 9fd00e60 only -- E10 05:40Z: the night host is 666e937f)")
    inst = pi.frozen_instance(instance_id=f"E6-TINY-4x8-r1-{pdig[:8]}{tables_id}", program=program, partition=partition, vocabulary=dict(T["vocabulary"]), outputs=outputs,
                              q=T["q"], s=T["s"], delta=T["delta"], request_ids=[req["request_id"]], B_ex=0, B_ex_provenance="E6 freeze: B_ex = 0 principal run (brief)",
                              family=T["family"], approved_elf_sha256=elfs, declared_inputs=declared, commitments=commitments, notes=notes,
                              chronology_kind="partition-sample/v1", seed_custody="verifier_process_memory", vocabulary_V=SMALL_VOCAB)
    inst["sampler"]["law"]["delta_note"] = T["s_source"]
    inst["partition"]["weight_input_check"] = dict(pi.DEFAULT_WEIGHT_CHECK)
    decoded = pi._decode(pi.ProtocolInstance, inst, "instance")
    inst = pi._encode(decoded)  # canonical form: the written digest survives load -> encode
    accepted, refused = i5.classify_violations(pi.validate_instance(decoded), i5.ACCEPTED_ROWS)
    if refused:
        raise SystemExit("REFUSED: frozen TINY instance violates validate_instance beyond record_v5's allowlist: " + "; ".join(refused))
    idig = pi.instance_digest(inst)
    _write_json(out_dir / "expected_population.json", exp.to_json())
    _write_json(out_dir / "inventory.json", inv)
    (out_dir / "inventory.md").write_text(kt.inventory_markdown(inv))
    if inventory_dir is not None:
        Path(inventory_dir).mkdir(parents=True, exist_ok=True)
        _write_json(Path(inventory_dir) / "inventory-tiny.json", inv)
        (Path(inventory_dir) / "inventory-tiny.md").write_text(kt.inventory_markdown(inv))
    _write_json(out_dir / "protocol_instance.json", inst)
    pins = {"schema": "veritor/e6/frozen-pins/v1", "population": "TINY", "utc": _utc(), "instance_path": str(out_dir / "protocol_instance.json"), "instance_digest": idig,
            "instance_id": inst["instance_id"], "program": {"function": identity["function"], "digest": T["program_digest"], "gates": T["gates"], "input_gates": T["input_gates"]},
            "partition": {"digest": pdig, "kind": lmi2.KIND, "members": lmi2.preimage(rows_partition_digest_v1=rows_v1, kind_table_digest=exp["kind_table_digest"],
                                                                                        coverage_table_sha256=table_sha, literals=literals, token_select=token_select),
                          "rows_partition_digest_v1_rule": "legacy_minus_inputs_v2.v1_rows_digest over the rows the shipped population file carries (after from_v5's transform), "
                                                           "as rederive/check re-derive them [D19]"},
            "plan": {"digest": T["plan_digest"], "kind": T["plan_digest_kind"], "name": plan_name},
            "kind_table_digest": exp["kind_table_digest"], "coverage_table": {**ct, **ct_prov}, "N_V": n_v, "N_R": 1, "kinds": len(exp["kinds"]), "prescribed_inputs": exp["prescribed_inputs"],
            "literals": exp["literals"], "token_select": token_select,
            "epoch": {"request_ids": [req["request_id"]], "prompt_token_ids": req["prompt_token_ids"], "q": T["q"], "s": T["s"], "delta": T["delta"], "B_ex": 0},
            "interior_layout": inst["commitments"]["interior_layout"], "tables": {k: {"tree": v["tree"], "root": v["root"], "leaf_count": v["leaf_count"], "rows": v["rows"]} for k, v in (table_pins or {}).items()}, "weights": T["weights"],
            "literals_commitment": lit_pin or lit_note,
            "population_file": {"expected": "E10 builds population.v5.json on the pod as attempt 6 and transforms it with e2e.partitions.legacy_minus_inputs_v2 from-v5 "
                                            f"--instance {out_dir / 'protocol_instance.json'} --coverage-table {ct['path']} --literals {literals} --token-select {token_select}",
                                "v1_content_sha256": T["population_v1_content_sha256"], "v1_rows_partition_digest_original": T["rows_partition_digest_v1"],
                                "rows_partition_digest_v1_pinned": rows_v1, "checked_against": (str(population_file) if population_file else None)},
            "accepted_violations": accepted, "inventory": inv["summary"]}
    _write_json(out_dir / "pins.json", pins)
    return pins


def _tiny_rows_digest_v1(exp: ExpectedPopulation, literals: str, token_select: str, population_file: Path | None, coverage_table_sha256: str) -> str:
    """[D19] ``rows_partition_digest_v1`` of the TINY pin = the v1 digest of the rows AS THE SHIPPED POPULATION FILE CARRIES THEM (after
    ``legacy_minus_inputs_v2.from_v5``: Const rows dropped when literals are compiled, ArgmaxStep scans regrouped when whole-scan) -- because the
    shipped verifier (``rederive``/``check`` under record_v5's ``check="full"``, ``partition_sample.stage1``, ``reverify.V1.partition``) re-derives that
    digest from the file's own rows and refuses any file whose ``v2.rows_partition_digest_v1`` differs.  With no transform (retained, argmax-scan) it is
    the original file's digest.  When ``population_file`` (E10's transformed v2 file) is given, the digest is RE-DERIVED from its rows with the shipped
    function and must agree with the pinned constant, the file's provenance (``v2.from_v1``) with the attempt-6 identities and its rows with the rule."""

    from e2e.partitions import legacy_minus_inputs_v2 as lmi2

    T = TINY_PINS
    key = f"{literals}/{token_select}"
    if literals == "retained" and token_select == "argmax-scan":
        pinned = T["rows_partition_digest_v1"]
    elif key in T["rows_partition_digest_v1_transformed"]:
        pinned = T["rows_partition_digest_v1_transformed"][key]
    elif population_file is None:
        raise SystemExit(f"REFUSED: no transformed-rows v1 digest is pinned for literals={literals}, token_select={token_select}; pass --population-file <from_v5 output> to derive it")
    else:
        pinned = None
    if population_file is None:
        return pinned
    doc, raw = _load_population_doc(population_file)
    if doc.get("partition_digest_kind") != lmi2.KIND:
        raise SystemExit(f"REFUSED: {population_file} is not a {lmi2.KIND} population file (kind {doc.get('partition_digest_kind')!r})")
    v2 = doc.get("v2") or {}
    prov = v2.get("from_v1") or {}
    if prov.get("content_sha256") != T["population_v1_content_sha256"] or prov.get("partition_digest") != T["rows_partition_digest_v1"]:
        raise SystemExit(f"REFUSED: {population_file} was not transformed from the attempt-6 v1 file (from_v1 {prov} != pinned {T['population_v1_content_sha256'][:16]}/{T['rows_partition_digest_v1'][:16]})")
    if v2.get("literals") != literals or v2.get("token_select") != token_select:
        raise SystemExit(f"REFUSED: {population_file} carries literals={v2.get('literals')}, token_select={v2.get('token_select')}, the freeze asks {literals}/{token_select}")
    derived = lmi2.v1_rows_digest(doc)  # the shipped function over the file's OWN rows
    if derived != v2.get("rows_partition_digest_v1") or (pinned is not None and derived != pinned):
        raise SystemExit(f"REFUSED: rows' v1 digest {derived[:16]} != file's v2.rows_partition_digest_v1 {str(v2.get('rows_partition_digest_v1'))[:16]} / pinned {str(pinned)[:16]}")
    if int(doc.get("vu_count", -1)) != exp["vu_count_per_ru"] or int(doc.get("ru_count", -1)) != 1:
        raise SystemExit(f"REFUSED: {population_file} N_V {doc.get('vu_count')} / N_R {doc.get('ru_count')} != the rule's {exp['vu_count_per_ru']} / 1")
    if lmi2.check(doc):
        raise SystemExit(f"REFUSED: {population_file} fails legacy_minus_inputs_v2.check: {lmi2.check(doc)}")
    return derived


# -- [E6 NIGHT 2026-09-12, G3] other Serve_v1 checkpoints: the frozen instance of a registry case ----------------------------------------
#
# What is FIXED per case (data, beside the registry config the identity binds): the checkpoint, the semantic profile the recorder must bind
# (name + file; its sha256 is the instance's `vocabulary.profile_sha256`, compared with the record's `semantic_profile` sha by
# protocol_instance.check_record), and the sampler's declared token domain.  Everything else is DERIVED: the kind table from the program
# (expected_population_serve_plan), the rows / plan digests by the two-phase rows pipeline (population.v5.json + plan_digest.json, MEASURED on
# the laptop), the partition identity a9/legacy-minus-inputs/v2 from the shipped rows + the derived kind table + the frozen coverage table --
# and the instance is refused unless `compare(expected, shipped_file, self_check="full")` is EMPTY: the digest the instance pins is the one the
# verifier derives (the 20a9090d lesson).
SERVE_CASES: dict[str, dict[str, Any]] = {
    # [E6 NIGHT 2026-09-12] TINY's checkpoint at the G4 lengths (32x15, 128x63, ...): the population is L5's census rule (B0 has rules; no plan build), the rows
    # are the two-phase pipeline's; the same profile / domain as TINY (record_v5 TINY_PINS)
    "B0": {"model": "HuggingFaceTB/SmolLM2-135M", "label": "B0",
           "profile": "llama-eager-batch-invariant-v3",
           "profile_why": "TINY's profile (E6-TINY-4x8-r1 pins it; LlamaForCausalLM, eager batch-invariant kernels, OUT16 sampler domain); the checkpoint and its "
                          "kernels do not change with the prompt / decode length",
           "domain_bits": 16, "domain_why": "OUT16 (GUIDANCE §32): V = 49,152 <= 2^16"},
    "SMOL360": {"model": "HuggingFaceTB/SmolLM2-360M", "label": "SMOL360",
                "profile": "llama-eager-batch-invariant-v3",
                "profile_why": "same architecture (LlamaForCausalLM) and the same eager batch-invariant kernels as TINY's SmolLM2-135M; v3 = v2 + the sampler's "
                               "declared 16-bit token domain (OUT16), which V = 49,152 fits (16 index bits); nothing in the profile names the checkpoint's dims",
                "domain_bits": 16, "domain_why": "OUT16 (GUIDANCE §32): V = 49,152 <= 2^16"},
    "QWEN05": {"model": "Qwen/Qwen2.5-0.5B", "label": "QWEN05",
               "profile": "qwen2-eager-batch-invariant-v2",
               "profile_why": "the only Qwen2ForCausalLM profile of the eager batch-invariant recorder (qkv bias node, eps / rope theta from config.json); it declares NO "
                              "sampler token domain, so the instance's 18-bit declaration is the enforcement (a qwen2 v3 with \"declared_domain\": 18 is the OUT16 "
                              "analogue -- E1/E3; OUT16 itself would REJECT every token id >= 65,536 of V = 151,936)",
               "domain_bits": 18, "domain_why": "V = 151,936 needs 18 index bits (2^17 = 131,072 < V <= 2^18); OUT16 is Llama-vocabulary-specific"},
}
PROFILES_DIR = POC / "manifests" / "semantic-profiles"


def serve_request(case: str, steps: int, *, prompt_token_ids: list[int] | None = None, index: int = 0, source: str | None = None) -> dict[str, Any]:
    """A declared request for a Serve case: E1's (``prompt_token_ids`` given) or the PROVISIONAL one -- TINY's prompt ids (record_v5's workload
    row ``index``, 4 SmolLM2 token ids; valid indices of any V >= 49,152) under the case's model name, ``max_tokens = STEPS + 1``; the id is
    the serving lane's ``base_request_id(model, ids, max_tokens)`` rule, so the instance's request id IS the one the record will carry."""

    from verity_vllm_sampler.workload import base_request_id  # the serving lane's id rule, read-only

    ids = list(prompt_token_ids) if prompt_token_ids is not None else list(tiny_request(index)["prompt_token_ids"])
    model = SERVE_CASES[case]["model"]
    return {"request_id": base_request_id(model, ids, steps + 1), "prompt_token_ids": ids, "model": model, "max_tokens": steps + 1,
            "provenance": ((source or "declared by E1 (prompt ids given)") if prompt_token_ids is not None else f"PROVISIONAL: TINY workload row {index}'s prompt ids under {model}")}


def freeze_serve(case: str, lp: int, steps: int, out_dir: Path, *, pipeline_dir: Path, literals: str = "compiled", token_select: str = "whole-scan",
                 prompt_token_ids: list[int] | None = None, request_source: str | None = None, request_index: int = 0, layout_digest: str | None = None, layout_rule: str | None = None,
                 tables: bool = True, revision: int = 1, s: str = "1/16384", s_source: str = "TINY's s = 2^-14 (record_v5/config.py DEFAULT_S[TINY-B0-4x8]); no DEFAULT_S row for this workload yet",
                 weights: dict[str, Any] | None = None, literals_pin: str = "auto", attention: dict[str, Any] | None = None,
                 attention_source: str | None = None) -> dict[str, Any]:
    """Freeze ``E6-<case>-<lp>x<steps>-r<revision>-<digest>`` for a registry Serve case (``b1.<case>_CONFIG``; :data:`SERVE_CASES`) from the two-phase
    rows pipeline's outputs in ``pipeline_dir`` (``plan_digest.json``: plan-rule/v1 MEASURED once; ``population.v5.json``: A9's v1 file of the same
    plan).  Population by rule (plan-derived kind table), the shipped file = ``legacy_minus_inputs_v2.from_v5`` of the v1 file under the frozen
    coverage table, ``partition_digest`` = the file's v2 digest AND ``compare(expected, file, self_check="full") == []`` (refused otherwise).
    Writes ``out_dir/{expected_population,inventory,coverage_table,population.v2,protocol_instance,pins}.json`` + ``inventory.md``."""

    from e2e import protocol_instance as pi
    from e2e.coverage_v1 import kind_table as kt
    from e2e.partitions import legacy_minus_inputs_v2 as lmi2
    from record_v5 import config as cfgmod, instance_v5 as i5
    from verity_vllm.correspondence.program import vocabulary_version

    if case not in SERVE_CASES:
        raise SystemExit(f"REFUSED: no SERVE_CASES entry for {case!r} (profile / declared token domain are fixed per case, not derived)")
    S = SERVE_CASES[case]
    out_dir, pipeline_dir = Path(out_dir), Path(pipeline_dir)
    plan_name = PLAN_NAME_P1 + ("/token-select=whole-scan" if token_select == "whole-scan" else "") + ("/literals=compiled" if literals == "compiled" else "")
    # [E6 CURVE 2026-09-13, D57] `attention` = the declared kernel geometry {"kernel","kBlockM","kBlockN"} pinned in program.identity -> E7's `Serve_v4{C,LP,STEPS,BN}` through
    # `b1.serve_program`; the rule's basis reads its kBlockN); None = the legacy program, byte-for-byte the pre-D57 identity and digests.
    # `attention_source` = the provenance of the VALUE (E10's MEASURED kernel geometry, D57) -- required with a geometry; recorded beside it.
    if attention is not None and not attention_source:
        raise SystemExit("REFUSED: an explicit attention geometry needs --attention-source (the measurement it is pinned from, D57)")
    identity = i5.program_identity(case, lp, steps, attention)
    exp = expected_population({"program": {"identity": identity}, "partition": {"plan_name": plan_name, "instances": 1}}, token_select=token_select, literals=literals)
    from verity_vllm.registry import b1

    cfg = dict(getattr(b1, f"{case}_CONFIG"))  # the registry's data (no build here: B0 at 32x15 is 519 M gates)
    V = int(cfg["V"])
    if cfg["model"] != S["model"]:
        raise SystemExit(f"REFUSED: b1.{case}_CONFIG names {cfg['model']!r}, SERVE_CASES {S['model']!r}")
    if (V - 1).bit_length() > S["domain_bits"]:
        raise SystemExit(f"REFUSED: V = {V} needs {(V - 1).bit_length()} index bits > the case's declared token domain {S['domain_bits']}")
    # the pipeline's MEASURED identities of the same program / plan
    pdj = json.loads((pipeline_dir / "plan_digest.json").read_text())
    if "program" not in exp:  # B0: L5's census rule derives counts, not the program's digest / size -- those are the pipeline's MEASURED build facts
        v1_head, _ = _load_population_doc(pipeline_dir / "population.v5.json")
        if not v1_head.get("gates"):
            raise SystemExit(f"REFUSED: {pipeline_dir / 'population.v5.json'} carries no program gate count; the rule (case {case}) does not derive it")
        exp["program"] = {"digest": pdj["program_digest"], "gates": int(v1_head["gates"]), "input_gates": int(exp["prescribed_inputs"]["gates"]),
                          "source": "MEASURED by the rows pipeline (plan_digest.json program_digest; population.v5.json gates = circuit.size); input_gates = the rule's prescribed inputs"}
    if pdj["program_digest"] != exp["program"]["digest"] or pdj.get("case") != case or int(pdj["LP"]) != lp or int(pdj["STEPS"]) != steps or pdj.get("plan_name") != PLAN_NAME_P1:
        raise SystemExit(f"REFUSED: {pipeline_dir / 'plan_digest.json'} is not this program's ({pdj.get('program_digest', '')[:16]} vs {exp['program']['digest'][:16]}, {pdj.get('case')} {pdj.get('LP')}x{pdj.get('STEPS')})")
    if pdj.get("attention") != (identity.get("attention") if attention is not None else None):
        raise SystemExit(f"REFUSED: {pipeline_dir / 'plan_digest.json'} was built with attention {pdj.get('attention')}, this freeze declares {identity.get('attention')} (D57: the pipeline and the pin must name the same geometry)")
    v1_doc, v1_raw = _load_population_doc(pipeline_dir / "population.v5.json")
    if v1_doc.get("program_digest") != exp["program"]["digest"] or v1_doc.get("plan_digest") != pdj["plan_digest"]:
        raise SystemExit(f"REFUSED: population.v5.json program {str(v1_doc.get('program_digest'))[:16]} / plan {str(v1_doc.get('plan_digest'))[:16]} != the pins")
    # inventory + coverage table (frozen with the instance; its sha is a member of the partition identity)
    layout_pin = pi.interior_layout_pin(digest=layout_digest, rule=layout_rule)
    inv = kt.inventory(exp, population=S["label"], layout_pin="pinned" if layout_pin["digest"] else "placeholder", n_r=1, tables_pinned=tables)
    table = kt.coverage_table(inv)
    table_sha = _write_json(out_dir / "coverage_table.json", table, canonical=True)
    ct = {"path": str(out_dir / "coverage_table.json"), "sha256": table_sha, "kinds": len(table["kinds"])}
    # the shipped population file: the pipeline's v1 rows under the fixed decisions; its v2 digest must be the one the verifier derives
    shipped = lmi2.from_v5(v1_doc, expected=exp, coverage_table=ct)
    shipped_path = out_dir / "population.v2.json"
    shipped_sha = _write_json(shipped_path, shipped, canonical=True)
    disagreements = compare(exp, shipped_path, instances=1, self_check="full")
    if disagreements:
        raise SystemExit("REFUSED: the shipped population file is not the derived population (the instance would pin a digest the verifier does not derive): "
                         + "; ".join(f"{d['field']}: expected {d['expected']!r}, actual {d['actual']!r}" for d in disagreements[:8]))
    problems = lmi2.check(shipped, exp)
    if problems:
        raise SystemExit("REFUSED: legacy_minus_inputs_v2.check: " + "; ".join(problems))
    rows_v1 = shipped["v2"]["rows_partition_digest_v1"]
    pdig = lmi2.partition_digest(rows_partition_digest_v1=rows_v1, kind_table_digest=exp["kind_table_digest"], coverage_table_sha256=table_sha, literals=literals, token_select=token_select)
    if pdig != shipped["partition_digest"]:
        raise SystemExit(f"REFUSED: partition digest {pdig[:16]} != the shipped file's {shipped['partition_digest'][:16]}")
    n_v = exp["vu_count_per_ru"]
    if int(shipped["vu_count"]) != n_v or int(shipped["ru_count"]) != 1:
        raise SystemExit(f"REFUSED: shipped N_V {shipped['vu_count']} / N_R {shipped['ru_count']} != the rule's {n_v} / 1")
    # the declared request, the profile, the vocabulary
    req = serve_request(case, steps, prompt_token_ids=prompt_token_ids, index=request_index, source=request_source)
    declared = pi.declared_inputs_section([{"request_id": req["request_id"], "prompt_token_ids": req["prompt_token_ids"]}],
                                          source=f"{req['provenance']}; id = verity_vllm_sampler.workload.base_request_id(model, prompt ids, max_tokens = STEPS + 1); greedy declared; "
                                                 "seed / sampling params not pinned", pin_sampling_params=False, LP=lp)
    profile_path = PROFILES_DIR / f"{S['profile']}.json"
    if not profile_path.exists():
        raise SystemExit(f"REFUSED: semantic profile {profile_path} does not exist")
    vv = vocabulary_version()
    vocabulary = {"digest": vv["digest"], "schema": vv["schema"], "profile_name": S["profile"], "profile_sha256": _sha256_hex(profile_path.read_bytes()),
                  "source": f"{profile_path.relative_to(ROOT)} (sha256 of the profile file); vocabulary = "
                            f"verity_vllm.correspondence.program.vocabulary_version() of THIS checkout (its `sources` member hashes registry/b1.py, so it moved when the "
                            f"SMOL360 / QWEN05 configs were added); why this profile: {S['profile_why']}"}
    src = f"two-phase rows pipeline on the laptop (E6 NIGHT 2026-09-12: {pipeline_dir}; plan_digest.json method: {pdj.get('method', '?')})"
    program = {"digest": exp["program"]["digest"], "identity": identity, "input_inclusion": "gates_include_input_gates", "kind": "serve",
               **({"attention_provenance": {"label": "MEASURED", "value": identity["attention"], "source": attention_source,
                                            "rule": "D57: the kernel geometry is a declared static of the pinned identity; absent = legacy kBlockN 64"}} if attention is not None else {}),
               "gates": {"value": exp["program"]["gates"], "label": "MEASURED", "source": "population_v3.serve_plan_kind_table: the built program's circuit size (input gates included)"},
               "input_gates": {"value": exp["program"]["input_gates"], "label": "MEASURED", "source": f"built program; = the rule's prescribed inputs {exp['prescribed_inputs']['gates']:,}"},
               "source": f"identity = bind(b1.Serve, C=b1.{case}_CONFIG, LP={lp}, STEPS={steps}).id (record_v5/instance_v5.program_identity); {S['model']} @ {cfg.get('revision')}"}
    partition = {
        "plan_name": plan_name, "plan_descriptor_digest": pdj["plan_digest"], "plan_digest_kind": pdj["plan_digest_kind"], "vu_level": "coordinate", "cu_strategy": "whole",
        "validation": {"artifact": str(out_dir / "expected_population.json"), "program_digest": exp["program"]["digest"], "plan_digest": pdj["plan_digest"], "partition_digest": pdig,
                       "status": "valid",
                       "source": f"rows: the P1/cu=whole UnitRuns of the built program ({src}; basis partition {str(v1_doc.get('basis', {}).get('partition_digest', ''))[:16]}); kind table: DERIVED "
                                 f"by rule from the pinned instance (population_v3.expected_population -> serve plan rule); population = rows minus Input kinds"
                                 f"{' minus Const kinds' if literals == 'compiled' else ''} ({lmi2.KIND}); compare(self_check='full') == [] at freeze"},
        "vu_count": {"value": n_v, "label": "MEASURED", "source": f"population_v3.expected_population (plan-derived kind table at ({lp}, {steps}), literals={literals}, token_select={token_select}) "
                                                                  f"== the shipped file's vu_count"},
        "ru_count": {"value": 1, "label": "MEASURED", "source": "P1/cu=whole: one RU per request"},
        "count_basis": "validated_plan_minus_inputs", "address_map": None, "instances": 1,
        "partition_digest": pdig, "partition_digest_kind": lmi2.KIND,
        "source": f"{lmi2.KIND} = sha256(rows' v1 digest {rows_v1[:16]}, kind-table digest {exp['kind_table_digest'][:16]} ({KIND_TABLE_DIGEST}), coverage table sha {table_sha[:16]}, "
                  f"literals={literals}, token_select={token_select}) -- e2e.partitions.legacy_minus_inputs_v2; verifier-derived: population_v3.compare(self_check='full') == [] on {shipped_path.name}",
    }
    outputs = [{"name": "token_slot", "multiplicity": steps + 1, "domain_bits_declared": S["domain_bits"], "wire_bits": 32,
                "enforcement": f"declared token domain {S['domain_bits']} bits on the sampler's select; check_output_domains on every request (value >> {S['domain_bits']} == 0)",
                "source": f"Serve_v1 STEPS+1 = {steps + 1} root output slots per request; {S['domain_why']}"}]
    table_pins = pi.table_pins() if tables else None
    lit_pin, lit_note = literals_commitment(exp["literals"]["kinds"], mode=literals_pin)
    commitments = {"interior_layout": layout_pin, **({"tables": table_pins} if table_pins else {}), **({"literals": lit_pin} if lit_pin else {}),
                   "coverage_table": {**ct, "source": "e2e.coverage_v1.kind_table.coverage_table (support / producer / position per kind; E6 inventory) -- frozen with the instance"},
                   "source": "E6 NIGHT 2026-09-12: pins fixed before the first commitment; V3.interior_layout fails closed while the layout pin is a placeholder; "
                             "[D36] V3.literal_table fails closed (LITERAL_UNPINNED) on a selected literal VU while commitments.literals is absent"}
    notes = [f"pre-challenge instance written by e2e.population_v3.freeze_serve ({case}); {i5.CONSTRUCTION_NOTE}", "basis=rows",
             "loaded by record_v5/instance_v5.load_instance_v5: validate_instance's rules are all applied; exactly the enumerated accepted violations are admitted",
             f"{S['model']} @ {cfg.get('revision')} at {lp}x{steps}: ONE declared request ({req['request_id']}, prompt {req['prompt_token_ids']}; {req['provenance']}); q = 1 (|J| = 1); "
             f"literals={literals}; token_select={token_select}",
             f"semantic profile {S['profile']}: {S['profile_why']}",
             f"declared token domain {S['domain_bits']} bits: {S['domain_why']}",
             "F policy: gamble (B_ex = 0 principal run); randomness mode: production; serving: batch-invariant, greedy, prefix caching off",
             f"interior layout pin: {('pinned ' + str(layout_pin['digest']) + ' (' + str(layout_pin.get('rule') or 'rule unnamed') + ')') if layout_pin['digest'] else 'PLACEHOLDER (awaiting E2 layout_digest)'}",
             "weights: " + (f"{weights}" if weights else "NOT YET AUTHORIZED (E1 authorizes the checkpoint auditor-side before the audit: weights_authorized.json root + ids_digest; a later revision pins them)"),
             lit_note]
    tables_id = ("-t" + _digest({k: v["root"] for k, v in table_pins.items()})[:8]) if table_pins else ""
    tables_id += ("-l" + _digest({k: v["root"] for k, v in lit_pin["trees"].items()})[:8]) if lit_pin else ""
    elfs = cfgmod.approved_elf_sha256()
    inst = pi.frozen_instance(instance_id=f"E6-{case}-{lp}x{steps}-r{revision}-{pdig[:8]}{tables_id}", program=program, partition=partition, vocabulary=vocabulary, outputs=outputs,
                              q="1", s=s, delta="1/1000000000000", request_ids=[req["request_id"]], B_ex=0, B_ex_provenance="E6 freeze: B_ex = 0 principal run (brief)",
                              family=cfgmod.FAMILY, approved_elf_sha256=elfs, declared_inputs=declared, commitments=commitments, notes=notes,
                              chronology_kind="partition-sample/v1", seed_custody="verifier_process_memory", vocabulary_V=V)
    inst["sampler"]["law"]["delta_note"] = s_source
    inst["partition"]["weight_input_check"] = dict(pi.DEFAULT_WEIGHT_CHECK)
    decoded = pi._decode(pi.ProtocolInstance, inst, "instance")
    inst = pi._encode(decoded)
    accepted, refused = i5.classify_violations(pi.validate_instance(decoded), i5.ACCEPTED_ROWS)
    if refused:
        raise SystemExit(f"REFUSED: frozen {case} instance violates validate_instance beyond record_v5's allowlist: " + "; ".join(refused))
    idig = pi.instance_digest(inst)
    _write_json(out_dir / "expected_population.json", exp.to_json())
    _write_json(out_dir / "inventory.json", inv)
    (out_dir / "inventory.md").write_text(kt.inventory_markdown(inv))
    _write_json(out_dir / "protocol_instance.json", inst)
    pins = {"schema": "veritor/e6/frozen-pins/v1", "population": S["label"], "utc": _utc(), "instance_path": str(out_dir / "protocol_instance.json"), "instance_digest": idig,
            "instance_id": inst["instance_id"], "model": S["model"], "revision_hf": cfg.get("revision"), "config": {k: v for k, v in cfg.items() if k != "profile"},
            "program": {"function": identity["function"], "digest": exp["program"]["digest"], "gates": exp["program"]["gates"], "input_gates": exp["program"]["input_gates"]},
            "partition": {"digest": pdig, "kind": lmi2.KIND, "members": lmi2.preimage(rows_partition_digest_v1=rows_v1, kind_table_digest=exp["kind_table_digest"],
                                                                                        coverage_table_sha256=table_sha, literals=literals, token_select=token_select),
                          "verifier_derived": f"population_v3.compare(expected, {shipped_path.name}, self_check='full') == [] at freeze; legacy_minus_inputs_v2.check == []"},
            "plan": {"digest": pdj["plan_digest"], "kind": pdj["plan_digest_kind"], "name": plan_name, "method": pdj.get("method")},
            "kind_table_digest": exp["kind_table_digest"], "coverage_table": ct, "N_V": n_v, "N_R": 1, "kinds": len(exp["kinds"]), "prescribed_inputs": exp["prescribed_inputs"],
            "literals": exp["literals"], "token_select": token_select, "vocabulary": vocabulary, "outputs": outputs,
            "epoch": {"request_ids": [req["request_id"]], "prompt_token_ids": req["prompt_token_ids"], "request_provenance": req["provenance"], "q": "1", "s": s, "delta": "1/1000000000000", "B_ex": 0},
            "interior_layout": inst["commitments"]["interior_layout"],
            "tables": {k: {"tree": v["tree"], "root": v["root"], "leaf_count": v["leaf_count"], "rows": v["rows"]} for k, v in (table_pins or {}).items()},
            "literals_commitment": lit_pin or lit_note,
            "weights": weights or {"status": "NOT YET AUTHORIZED", "who": "E1 (auditor-side authorization of the checkpoint on the pod)"},
            "population_file": {"path": str(shipped_path), "content_sha256": shipped_sha, "v1_content_sha256": v1_doc.get("content_sha256"), "v1_rows_partition_digest_original": v1_doc.get("partition_digest"),
                                "rows_partition_digest_v1_pinned": rows_v1, "transform": shipped["v2"]["transform"]},
            "accepted_violations": accepted, "inventory": inv["summary"], "label": "MEASURED (program built, plan built, rows built on the laptop; nothing argued)"}
    _write_json(out_dir / "pins.json", pins)
    return pins


def night_table_row(case: str, prompt_len: int, max_tokens: int, out_dir: Path, *, model: str | None = None, instance_path: str | None = None,
                    literals: str = "compiled", token_select: str = "whole-scan") -> dict[str, Any]:
    """[NIGHT 2026-09-12, target 3] The population fields of the measured table's row ``out/night/table/<model>__<p>x<d>.json`` (brief §"The
    measured table"): ``N_R`` (one RU per request under P1/cu=whole), ``N_V`` (the derived population at (LP = p, STEPS = d - 1) under the fixed
    decisions), each ``{value, label, where}``; ``instance`` = the pinned instance path or the reason there is none yet.  Other groups' fields
    in an existing file are kept (the file is merged by key, never rewritten)."""

    from verity_vllm.registry import b1

    steps = max_tokens - 1
    cfg = dict(getattr(b1, f"{case}_CONFIG"))
    model = model or cfg["model"]
    pn = PLAN_NAME_P1 + ("/token-select=whole-scan" if token_select == "whole-scan" else "") + ("/literals=compiled" if literals == "compiled" else "")
    exp = expected_population({"case": case, "lp": prompt_len, "steps": steps, "plan_name": pn, "instances": 1})
    by_rule = case == "B0"
    if by_rule:
        label = "MODELED" if (prompt_len, steps) not in ((32, 15), (6500, 1599)) else "MEASURED"
        where = (f"population_v3.expected_population: L5's census rules (data/census/census_small_6500x1600.json, MEASURED on the built 6500x1600 plan; hold-out "
                 f"validated on the built 32x15 plan, data/rec/p1-b0-lp32s15-validated.json) evaluated at (LP={prompt_len}, STEPS={steps}); literals={literals}, "
                 f"token_select={token_select}. Exact by rule; the pinned instance's compare(self_check='full') re-derives it against the built rows -> MEASURED when the rows exist")
        # [E6 SCALE 2026-09-12] a pinned instance of this shape IS the built rows: its partition.vu_count was compared against the rows' v1 digest
        # at the freeze (compare(self_check='full') == []), so the rule's count is MEASURED for the shape the moment the instance names it.
        if instance_path and (ROOT / instance_path).exists():
            pinned = json.loads((ROOT / instance_path).read_text())
            if int(pinned["partition"]["vu_count"]["value"]) == int(exp["vu_count_per_ru"]) and pinned["program"]["identity"].get("LP") == prompt_len and pinned["program"]["identity"].get("STEPS") == steps:
                label = "MEASURED"
                where = (f"pinned instance {pinned['instance_id']} ({instance_path}): partition.vu_count == the rows' count at the freeze (compare(self_check='full') == [] "
                         f"over the built P1/cu=whole rows of Serve_v1 at (LP={prompt_len}, STEPS={steps})); == L5's census rule evaluated at the same shape (literals={literals}, token_select={token_select})")
            else:
                raise SystemExit(f"REFUSED: {instance_path} is not a pinned instance of ({case}, LP={prompt_len}, STEPS={steps}) with the rule's count {exp['vu_count_per_ru']}")
    else:
        label, where = "MEASURED", f"population_v3.expected_population -> serve plan rule: P1/cu=whole UnitRuns of the built program ({exp['label']})"
    path = Path(out_dir) / f"{model.split('/')[-1]}__{prompt_len}x{max_tokens}.json"
    row = json.loads(path.read_text()) if path.exists() else {}
    row.update({
        "model": {"value": model, "label": "MEASURED", "where": f"verity_vllm.registry.b1.{case}_CONFIG"},
        "architecture": {"value": "Qwen2ForCausalLM" if case.startswith("QWEN") else "LlamaForCausalLM", "label": "MEASURED", "where": "config.json architectures[0]"},
        "prompt_len": {"value": prompt_len, "label": "MEASURED", "where": "LP of the bound program"},
        "max_tokens": {"value": max_tokens, "label": "MEASURED", "where": "STEPS + 1 output slots of the bound program"},
        # per-field ``utc`` (integrator 07:25Z: the field-wise union takes the latest utc per field; N_R / N_V / instance are E6's fields -- the pinned instance's form:
        # ONE declared request per instance, compiled literals, whole-scan token selection)
        "N_R": {"value": exp["ru_count_per_request"], "label": "MEASURED", "where": "P1/cu=whole: one RU per request (population_v3.expected_population ru_count_per_request); "
                                                                                  "the instance declares ONE request", "utc": _utc()},
        "N_V": {"value": exp["vu_count_per_ru"], "label": label, "where": where, "kinds": len(exp["kinds"]), "kind_table_digest": exp["kind_table_digest"],
                "literals": literals, "token_select": token_select, "prescribed_inputs": exp["prescribed_inputs"], "utc": _utc()},
        "instance": ({"value": instance_path, "label": "MEASURED", "where": "E6 population_v3.freeze_serve: compare(self_check='full') == [] at the freeze", "utc": _utc()}
                     if instance_path else row.get("instance") or
                     "NOT PINNED (E6 pins E6-<case>-<p>x<d> the moment E2/E8 declare the request; the population fields above do not depend on it)"),
        "population_utc": _utc(), "population_source": "E6 (G3) population_v3.night_table_row",
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=1, sort_keys=True) + "\n")
    return row


def freeze_core(out_dir: Path, *, family: str = "DemoCore_v1", params: dict[str, int] | None = None, module: str = "e2e.demo_core.circuit", builder_symbol: str = "build",
                layout_digest: str | None = None, layout_rule: str | None = None,
                partition_doc: dict[str, Any] | None = None, inventory_dir: Path | None = None, population_doc: dict[str, Any] | None = None,
                declared_inputs: Path | None = None, request_id: str = "core-0") -> dict[str, Any]:
    """Freeze E8's CORE instance: the registered ``DemoCore_v1{N,R}`` at ``params`` (default N=8, R=8), one program instance, q = s = 1
    (exhaustive first record).  The partition identity is the program's own rule (E8's ``demo_core.partition``: kind ``demo-core/stage-ru-gate-vu/v1``),
    checked here against :func:`expected_population_generic` (``compare_generic`` must be empty); ``partition_doc`` = E8's ``Partition.to_json()``
    (computed by importing E8's module when None)."""

    from e2e import protocol_instance as pi
    from e2e.coverage_v1 import kind_table as kt

    params = dict(params or {"N": 8, "R": 8})
    out_dir = Path(out_dir)
    identity = pi.generic_program_identity(family, params, module=module, case="".join(f"{k}{v}" for k, v in sorted(params.items())))
    exp0 = expected_population({"program": {"identity": identity}})
    pdig_prog = exp0["program_identity"]["program_digest"]
    builder = pi.program_builder_pin(module=module, symbol=builder_symbol, params=params, program_digest=pdig_prog)
    exp = expected_population({"program": {"identity": identity, "builder": builder}})  # [D1] builder route must agree with the registry bind
    import importlib

    mod = importlib.import_module(module)
    part_obj = mod.partition(**params)
    if partition_doc is None:
        partition_doc = part_obj.to_json()
    dis = compare_generic(exp, partition_doc)
    if dis:
        raise SystemExit("REFUSED: E8's partition disagrees with the rule: " + json.dumps([dict(d) for d in dis]))
    part_kind = str(partition_doc.get("kind") or "partition-of-the-program/v1")
    plan_digest = _digest(partition_doc)  # E8's Partition.digest = sha256(canonical partition document): the PLAN (constructor rule) digest, kind plan-rule/v1
    if plan_digest != part_obj.digest:
        raise SystemExit(f"REFUSED: partition document digest {plan_digest} != {module}.partition(**params).digest {part_obj.digest}")
    # [D17] the POPULATION file the shipped driver pins (E9 steps.population_derived -> population_derived_check): E8's population/v2 document
    # (partition-intervals/v1 digest re-derived from its rows by the shipped CompactPopulation.check) -- checked against the rule here, then its
    # partition_digest is what partition.partition_digest pins (the plan digest goes to plan_descriptor_digest / validation.plan_digest)
    if population_doc is None:
        pop_mod = importlib.import_module(module.rsplit(".", 1)[0] + ".instance")
        population_doc = pop_mod.population_doc(part_obj)
    pop_check = population_derived_check({"program": {"identity": identity, "builder": builder}, "partition": {"instances": 1}}, population_doc, instances=1, self_check="full")
    if pop_check["status"] != "matches":
        raise SystemExit("REFUSED: the population file disagrees with the rule: " + json.dumps(pop_check["disagreements"]))
    if population_doc.get("plan_digest") != plan_digest or population_doc.get("program_digest") != pdig_prog:
        raise SystemExit(f"REFUSED: population file plan_digest {population_doc.get('plan_digest')} / program_digest {population_doc.get('program_digest')} != {plan_digest} / {pdig_prog}")
    part_digest, part_digest_kind = str(population_doc["partition_digest"]), str(population_doc["partition_digest_kind"])
    pop_sha = _write_json(out_dir / "population.compact.json", population_doc)
    # [D17] the auditor's declared x: E8's declared_inputs.json (input seed of the run) -> epoch.inputs, digest re-derived here, never copied
    inputs_section = None
    if declared_inputs is not None:
        din = json.loads(Path(declared_inputs).read_text())
        words = [int(w) for w in din["words_in_gate_order"]]
        cols = exp["prescribed_inputs"]["collections"]
        if [c["name"] for c in cols] != list(din["collections"]) or any(list(range(*c["gate_interval"])) != list(din["collections"][c["name"]]["gates"]) for c in cols):
            raise SystemExit("REFUSED: declared_inputs.json collections/gates are not the program's root Input collections")
        if words != [int(w) for name in din["collections"] for w in din["collections"][name]["words"]] or len(words) != exp["input_gates"]:
            raise SystemExit("REFUSED: declared_inputs.json words_in_gate_order != the collections' words in gate order")
        ddig = pi.declared_input_digest(words)
        if ddig != din.get("digest"):
            raise SystemExit(f"REFUSED: declared_inputs.json digest {din.get('digest')} != declared_input_digest(words) {ddig}")
        inputs_section = {"kind": pi.DECLARED_INPUTS_KIND, "layout": None,
                          "per_request": {request_id: {"prompt_len": len(words), "prompt_sha256": ddig, "sampling_params_sha256": None, "seed": None}},
                          "source": f"the auditor's declared x: declared_input_digest({len(words)} input words in gate order; e2e.protocol_instance.declared_input_digest) "
                                    f"from {Path(declared_inputs).name} (E8's fixture; verified against the program's Input collections here); layout not pinned (int32-token rule does not apply)"}
        _write_json(out_dir / "declared_inputs.json", din)
    layout_pin = pi.interior_layout_pin(digest=layout_digest, rule=layout_rule)  # E2's layout_pin() verbatim when e2e.interior.layout is importable, else placeholder
    layout_status = "pinned" if layout_pin["digest"] else "placeholder"
    inv = kt.inventory(exp, population="CORE", layout_pin=layout_status, ru_output_kinds=set(exp["ru_output_kinds"]), n_r=int(exp["ru_count_per_instance"]),
                       count_basis=("per RU" if exp["homogeneous_rus"] else "per instance"),  # D17: kinds[].count is per RU for homogeneous RUs
                       interior_committed=True)  # E8's record commits interior/core-ru<r>/000000 (E8 20:17Z: 128/128 PROVEN+VERIFIED under 9fd00e60)
    table = kt.coverage_table(inv)
    table_sha = _write_json(out_dir / "coverage_table.json", table, canonical=True)
    ct = {"path": str(out_dir / "coverage_table.json"), "sha256": table_sha, "kinds": len(table["kinds"])}
    n_v, n_r = exp["vu_count_per_instance"], exp["ru_count_per_instance"]
    prims = sorted({k["fn"] for k in exp["kinds"]} | {c["kind"].split("|")[0] for c in exp["prescribed_inputs"]["collections"]})
    vocab_digest = _digest({"primitives": prims})
    program = {"digest": pdig_prog, "identity": identity, "input_inclusion": "gates_include_input_gates", "kind": "registered", "builder": builder,
               "gates": {"value": exp["gates"], "label": "MEASURED", "source": "Program(bind(REGISTRY[family], **params)).gates (population_v3.expected_population_generic)"},
               "input_gates": {"value": exp["input_gates"], "label": "MEASURED", "source": "Program.input_gates = the root Input collections (prescribed)"},
               "source": "verity.ir.codec.program_digest(encode_program(Program(bind(...)))) -- rebuilt by the verifier from the identity (family + params + module) "
                         f"and, identically, from program.builder ({module}.{builder_symbol}(**params); INTEGRATOR D1; e2e.protocol_instance.build_pinned_program)"}
    partition = {
        "plan_name": part_kind, "plan_descriptor_digest": plan_digest, "plan_digest_kind": str(population_doc.get("plan_digest_kind") or "plan-rule/v1"), "vu_level": "gate",
        "cu_strategy": "root-call",
        "validation": {"artifact": str(out_dir / "population.compact.json"), "program_digest": pdig_prog, "plan_digest": plan_digest, "partition_digest": part_digest,
                       "status": "valid", "source": f"population_v3.expected_population_generic re-derives the partition by rule ({GENERIC_RULE_KIND}); compare_generic(E8 partition) = [] "
                                                    "(N_V, N_R, kind counts, RU intervals, per-RU kind tables); population_derived_check(population.compact.json) = matches "
                                                    f"(rows per RU, tiling, {part_digest_kind} re-derived by CompactPopulation); population.compact.json sha256 {pop_sha}"},
        "vu_count": {"value": n_v, "label": "MEASURED", "source": "one VU per computed gate of every root call (population_v3.expected_population_generic)"},
        "ru_count": {"value": n_r, "label": "MEASURED", "source": "one RU per root call node (population_v3.expected_population_generic)"},
        "count_basis": "validated_plan", "address_map": None, "instances": 1,
        "partition_digest": part_digest, "partition_digest_kind": part_digest_kind,  # [D17] the POPULATION's digest (partition-intervals/v1), not the plan's
        "first_boundary": {"rule": "outgoing boundary runs of every CU", "rows": {"value": sum(r["outputs"] for r in exp["rus"]), "label": "MEASURED",
                                                                                  "source": "sum over RUs of |returns| (expected_population_generic rus[*].outputs)"},
                           "bytes": None, "committed_in": "boundary", "source": "every RU's declared output tuple is committed before r_q (E8 CoreRecord trees)"},
        "source": f"partition_digest = the population file's {part_digest_kind} (rows + CU intervals; e2e.population_v2.CompactPopulation); plan_descriptor_digest = "
                  f"{part_kind} = sha256(canonical JSON of the program's partition document, E8 demo_core.partition) [D17]; rule-derived identity {GENERIC_RULE_KIND} "
                  f"{exp['kind_table_digest'][:16]} (population_v3); coverage table sha {table_sha[:16]} pinned in commitments.coverage_table",
    }
    N = int(params.get("N", exp["rus"][-1]["outputs"]))
    outputs = [{"name": "h_R", "multiplicity": N, "domain_bits_declared": 16, "wire_bits": 16, "enforcement": "none needed: bf16 words, the declared domain is the wire",
                "source": f"{identity['function']} root output: the last stage's N = {N} bf16 words"}]
    commitments = {"interior_layout": layout_pin,
                   "coverage_table": {**ct, "source": "e2e.coverage_v1.kind_table.coverage_table (E6 inventory) -- frozen with the instance"},
                   "source": "E6 2026-09-11: pins fixed before the first commitment"}
    vocabulary = {"digest": vocab_digest, "schema": "veritor/e6/program-primitives/v1", "profile_name": "none (generic verity.ir program: its vocabulary is its primitive set)",
                  "profile_sha256": vocab_digest, "source": f"sha256(canonical JSON of the sorted primitive ids the program uses: {prims})"}
    notes = ["pre-challenge instance written by e2e.population_v3.freeze_core (generic registered verity.ir program; no vLLM assumption)",
             f"CORE of EOD 2026-09-11: {identity['function']} ({partition_doc.get('program', {}).get('label', 'generic protocol demonstration')}); one program instance; "
             "q = s = 1 for the exhaustive first record (E8), then the predeclared sampled sequence",
             (f"declared inputs: the root Input collections {[c['name'] for c in exp['prescribed_inputs']['collections']]} ({exp['input_gates']} bf16 words) pinned in epoch.inputs "
              f"(prompt_sha256 = declared_input_digest of the words in gate order, re-derived from {Path(declared_inputs).name}; request {request_id}) [D17]"
              if inputs_section is not None else
              f"declared inputs: the root Input collections {[c['name'] for c in exp['prescribed_inputs']['collections']]} ({exp['input_gates']} bf16 words) -- NOT pinned "
              "(no --declared-inputs given): the shipped driver's step2_check_registered refuses such an instance by name"),
             f"interior layout pin: {('pinned ' + str(layout_pin['digest']) + ' (' + str(layout_pin.get('rule') or 'rule unnamed') + ')') if layout_pin['digest'] else 'PLACEHOLDER (awaiting E2 layout_digest)'}"]
    inst = pi.frozen_instance(instance_id=f"E6-CORE-{identity['case']}-{part_digest[:8]}", program=program, partition=partition, vocabulary=vocabulary, outputs=outputs,
                              q="1", s="1", delta="1/1000000000000", request_ids=[request_id], B_ex=0, B_ex_provenance="E6 freeze: B_ex = 0 principal run (brief)",
                              family="veritor.elem-bf16@1", approved_elf_sha256=list(TINY_PINS["approved_elf_sha256"]), declared_inputs=inputs_section, commitments=commitments,
                              notes=notes, chronology_kind="partition-sample/v1", seed_custody="verifier_process_memory")
    decoded = pi._decode(pi.ProtocolInstance, inst, "instance")
    problems = pi.validate_instance(decoded)
    if problems:
        raise SystemExit("REFUSED: frozen CORE instance violates validate_instance: " + "; ".join(problems))
    idig = pi.instance_digest(inst)
    _write_json(out_dir / "expected_population.json", exp.to_json())
    _write_json(out_dir / "partition.json", partition_doc)
    _write_json(out_dir / "inventory.json", inv)
    (out_dir / "inventory.md").write_text(kt.inventory_markdown(inv))
    if inventory_dir is not None:
        Path(inventory_dir).mkdir(parents=True, exist_ok=True)
        _write_json(Path(inventory_dir) / "inventory-core.json", inv)
        (Path(inventory_dir) / "inventory-core.md").write_text(kt.inventory_markdown(inv))
    _write_json(out_dir / "protocol_instance.json", inst)
    pins = {"schema": "veritor/e6/frozen-pins/v1", "population": "CORE", "utc": _utc(), "instance_path": str(out_dir / "protocol_instance.json"), "instance_digest": idig,
            "instance_id": inst["instance_id"], "program": {"function": identity["function"], "digest": pdig_prog, "gates": exp["gates"], "input_gates": exp["input_gates"], "identity": identity},
            "partition": {"digest": part_digest, "kind": part_digest_kind, "rule_digest": exp["kind_table_digest"], "rule_digest_kind": GENERIC_RULE_KIND,
                          "population_file": {"path": str(out_dir / "population.compact.json"), "sha256": pop_sha, "content_sha256": population_doc.get("content_sha256")}},
            "plan": {"digest": plan_digest, "kind": str(population_doc.get("plan_digest_kind") or "plan-rule/v1"), "name": part_kind}, "coverage_table": ct, "N_V": n_v, "N_R": n_r,
            "kinds": {k["kind"]: {"count_per_ru": k["count"], "count_per_instance": k.get("count_per_instance", k["count"])} for k in exp["kinds"]},
            "ru_output_kinds": exp["ru_output_kinds"], "prescribed_inputs": exp["prescribed_inputs"]["gates"],
            "epoch": {"request_ids": [request_id], "q": "1", "s": "1", "delta": "1/1000000000000", "B_ex": 0, "inputs": inputs_section}, "interior_layout": inst["commitments"]["interior_layout"],
            "inventory": inv["summary"]}
    _write_json(out_dir / "pins.json", pins)
    return pins


def _load_population_doc(population_file: Path | str | dict[str, Any]) -> tuple[dict[str, Any], bytes | None]:
    if isinstance(population_file, dict):
        return population_file, None
    raw = Path(population_file).read_bytes()
    return json.loads(raw), raw


def _content_sha_rederives(doc: dict[str, Any]) -> bool:
    body = {k: v for k, v in doc.items() if k not in ("content_sha256", "seconds")}
    return _digest(body) == doc.get("content_sha256")


def compare(expected: ExpectedPopulation, population_file: Path | str | dict[str, Any], *, instances: int | None = None,
            self_check: str = "fast") -> list[Disagreement]:
    """Every field of ``population_file`` (``veritor/population/v2`` or ``veritor/a9/population-v5/v1``) that is not the expected one.
    ``[]`` iff the file's population IS the derived one on every derivable field and its own row-level digests re-derive.
    ``self_check``: ``"fast"`` (content sha + structure; the row-level partition digests are the file's own claim, listed in
    ``expected.not_derivable``) or ``"full"`` (also re-derive the partition digest from the rows: O(N_V) for ``partition-intervals/v1``).
    Never raises on a disagreement; raises only on an unreadable file."""

    doc, _ = _load_population_doc(population_file)
    out: list[Disagreement] = []
    schema = doc.get("schema")
    if schema not in (POPULATION_V2_SCHEMA, A9_POPULATION_V5):
        return [Disagreement("schema", f"{POPULATION_V2_SCHEMA} | {A9_POPULATION_V5}", schema, "not a population file this verifier reads")]
    exp = expected.kind_by_name
    entries = list((doc.get("kinds") or {}).get("entries") or [])
    names: dict[int, str] = {}
    for e in entries:
        names[int(e["id"])] = f"{e['fn']}|{e['form']}"
    # 1. kind table: names and declared widths
    file_names = set(names.values())
    # [E6 NIGHT 2026-09-12, E9's DRV-B finding] a WITH-INPUTS P1 file (the archives' population.compact.json: `inputs = units`, one unit per input gate) carries the
    # prescribed inputs as `Input*` rows.  They are not VUs of the derived population (Input gates are in no VU); the file is admitted by name iff those rows number
    # exactly the program's prescribed input gates (x instances) -- then the population is the file's rows MINUS them, and the N_V / per-RU counts are compared net of
    # them.  Any other count is a disagreement (the rows are not this program's inputs).  A transformed file (legacy_minus_inputs_v2) has no Input rows: unchanged.
    with_inputs_file = schema == POPULATION_V2_SCHEMA and "basis" not in doc and "v2" not in doc  # the plan's own rows, no exclusion applied
    input_names = {n for n in file_names if n.split("|", 1)[0].startswith("Input")} if with_inputs_file else set()
    for name in sorted(file_names - set(exp) - input_names):
        if name.split("|", 1)[0].startswith("Input"):
            out.append(Disagreement(f"kinds.extra[{name}]", None, name, "an Input kind: input gates are prescribed / excluded -- in no VU of the expected population "
                                                                       "(this file's construction already excludes them; an Input row here is not the plan's)"))
        else:
            out.append(Disagreement(f"kinds.extra[{name}]", None, name, "a kind the rules do not derive at this shape (renamed, or not of this program)"))
    for name in sorted(set(exp) - file_names):
        out.append(Disagreement(f"kinds.missing[{name}]", exp[name]["count"], None, "an expected kind with no entry in the file (renamed or dropped)"))
    for e in entries:
        name = f"{e['fn']}|{e['form']}"
        if name in exp and int(e.get("out_bits", -1)) != exp[name]["declared_bits"]:
            out.append(Disagreement(f"kinds[{name}].out_bits", exp[name]["declared_bits"], e.get("out_bits"), "declared width (the certificate's family width) differs"))
    # 2. rows: per-kind unit counts, per-RU unit counts, tiling -- the file's own content, scanned once (O(rows))
    rows = doc.get("rows") or []
    per_kind: dict[str, int] = {}
    per_ru_kind: dict[tuple[int, str], int] = {}
    per_ru: dict[int, int] = {}
    tiled, nxt = True, 0
    inputs_total = 0
    inputs_per_ru: dict[int, int] = {}
    for r in rows:
        if not (isinstance(r, list) and len(r) == 7):
            out.append(Disagreement("rows.fields", 7, len(r) if isinstance(r, list) else type(r).__name__, "row is not first_vu,count,ru,kind,lo,stride,unit_size"))
            break
        first, count, ru, kid = r[0], r[1], r[2], r[3]
        name = names.get(kid, f"kind#{kid}")
        if first != nxt:
            tiled = False
        nxt = first + count
        if name in input_names:  # prescribed inputs carried as units: counted apart, never as VUs
            inputs_total += count
            inputs_per_ru[ru] = inputs_per_ru.get(ru, 0) + count
            continue
        per_kind[name] = per_kind.get(name, 0) + count
        per_ru_kind[ru, name] = per_ru_kind.get((ru, name), 0) + count
        per_ru[ru] = per_ru.get(ru, 0) + count
    if not tiled:
        out.append(Disagreement("rows.tiling", "rows tile [0, vu_count) in VU order", "gap or overlap", "rows do not tile the population"))
    n_inst = int(doc.get("instances", 1))
    exp_ru = expected["ru_count_per_request"] * n_inst
    ru_count = int(doc.get("ru_count", -1))
    if input_names:
        prescribed = int((expected.get("prescribed_inputs") or {}).get("gates") or 0) * n_inst
        if inputs_total != prescribed:
            out.append(Disagreement("rows.inputs", prescribed, inputs_total, f"the file's Input rows ({sorted(input_names)}) are not the program's prescribed input gates x {n_inst} instance(s); "
                                                                            "a with-inputs P1 file is admitted only when they are exactly those"))
        for ru, n in sorted(inputs_per_ru.items()):
            per_ru[ru] = per_ru.get(ru, 0)  # an RU that holds only inputs still exists for the per-RU comparison
    if expected.get("homogeneous_rus") is False:
        # [D17] a generic program whose RUs differ has no per-RU kind count: `count` is per instance and this per-RU comparison is undefined --
        # refused by name (compare_generic checks that program's per-RU kind tables from rus[*].kinds)
        out.append(Disagreement("rows.rule", "homogeneous RUs (kinds[*].count per RU)", "RUs not homogeneous",
                                "population_derived_check compares rows per RU; use compare_generic for a program whose RUs differ"))
        return out
    for name in sorted(set(per_kind) | set(exp)):
        if name in exp and name in per_kind:
            if per_kind[name] != exp[name]["count"] * exp_ru:
                out.append(Disagreement(f"rows.kind_counts[{name}]", exp[name]["count"] * exp_ru, per_kind[name],
                                        f"per-kind unit count of the rows != the rule's count per RU x {exp_ru} RU(s)"))
            else:
                for ru in range(max(ru_count, 0)):
                    if per_ru_kind.get((ru, name), 0) != exp[name]["count"]:
                        out.append(Disagreement(f"rows.ru[{ru}].kind_counts[{name}]", exp[name]["count"], per_ru_kind.get((ru, name), 0),
                                                "this RU's units of the kind != the rule (every RU serves one request)"))
        elif name in exp and name not in per_kind and name in file_names:
            out.append(Disagreement(f"rows.kind_counts[{name}]", exp[name]["count"] * exp_ru, 0, "kind entry present but no row carries it"))
        elif name not in exp and name in per_kind and name not in file_names:
            out.append(Disagreement(f"rows.kind_counts[{name}]", None, per_kind[name], "rows reference a kind id with no entry"))
    # 3. N_V, N_R
    if instances is not None and n_inst not in (1, instances):
        out.append(Disagreement("instances", instances, n_inst, "the file is neither per-instance (1) nor the epoch's instance count"))
    if ru_count != exp_ru:
        out.append(Disagreement("ru_count", exp_ru, ru_count, f"{expected['ru_count_per_request']} RU per request x {n_inst} instance(s)"))
    if len(per_ru) != ru_count:
        out.append(Disagreement("rows.ru_count", ru_count, len(per_ru), "the rows carry a different number of RUs than ru_count (an RU dropped or added)"))
    for ru in range(max(ru_count, 0)):
        if per_ru.get(ru, 0) != expected["vu_count_per_ru"]:
            out.append(Disagreement(f"rows.ru[{ru}].vu_count", expected["vu_count_per_ru"], per_ru.get(ru, 0), "N_V of this RU != the rule's N_V per RU"))
    exp_vu = expected["vu_count_per_ru"] * exp_ru
    if int(doc.get("vu_count", -1)) - inputs_total != exp_vu:
        out.append(Disagreement("vu_count", exp_vu + inputs_total, doc.get("vu_count"), "N_V per RU x N_R" + (f" + the {inputs_total} prescribed Input units the with-inputs file carries" if inputs_total else "")))
    if nxt != int(doc.get("vu_count", -1)):
        out.append(Disagreement("rows.vu_count", doc.get("vu_count"), nxt, "rows cover a different number of VUs than vu_count"))
    # 4. identity: derived where the digest is the kind table's; else the file's own re-derivation
    pdk = doc.get("partition_digest_kind")
    pd = doc.get("partition_digest")
    if pdk == KIND_TABLE_DIGEST:
        if pd != expected["kind_table_digest"]:
            out.append(Disagreement("partition_digest", expected["kind_table_digest"], pd, f"{KIND_TABLE_DIGEST}: DERIVED by rule; the file's differs"))
    elif pdk == A9_PARTITION_DIGEST_KIND:
        basis = doc.get("basis") or {}
        try:
            from record_v5 import population_v5 as a9pop  # A9's definition, read-only

            rederived = a9pop._partition_digest(basis.get("partition_digest"), (basis.get("excluded") or {}).get("kinds"), rows)
        except Exception as e:  # pragma: no cover - A9's module missing
            rederived = f"UNAVAILABLE: {e}"
        if rederived != pd:
            out.append(Disagreement("partition_digest", rederived, pd, f"{A9_PARTITION_DIGEST_KIND} does not re-derive from the file's own basis digest + rows (not derivable by rule)"))
    elif pdk == A9_PARTITION_DIGEST_KIND_V2:
        # [E6 2026-09-11] v2 = H(v1 digest of the rows, DERIVED kind-table digest, frozen coverage-table sha, literals, token_select):
        # the kind-table member IS derivable (must equal ours); the rows member is the file's own re-derivation
        from e2e.partitions import legacy_minus_inputs_v2 as lmi2

        v2 = doc.get("v2") or {}
        if v2.get("kind_table_digest") != expected["kind_table_digest"]:
            out.append(Disagreement("v2.kind_table_digest", expected["kind_table_digest"], v2.get("kind_table_digest"), f"{KIND_TABLE_DIGEST}: DERIVED by rule; the file's differs"))
        if v2.get("literals") != (expected.get("literals") or {}).get("treatment") or v2.get("token_select") != expected.get("token_select"):
            out.append(Disagreement("v2.literals/token_select", [(expected.get("literals") or {}).get("treatment"), expected.get("token_select")],
                                    [v2.get("literals"), v2.get("token_select")], "the fixed decisions as applied differ from the derivation's"))
        try:
            rederived = lmi2.rederive(doc)
        except Exception as e:  # noqa: BLE001
            rederived = f"UNAVAILABLE: {e}"
        if rederived != pd:
            out.append(Disagreement("partition_digest", rederived, pd, f"{A9_PARTITION_DIGEST_KIND_V2} does not re-derive from the rows' v1 digest + the v2 block"))
    elif self_check == "full" and schema == POPULATION_V2_SCHEMA:
        from e2e.population_v2 import CompactPopulation

        try:
            rederived = CompactPopulation(doc).recompute_partition_digest()
        except Exception as e:
            rederived = f"UNAVAILABLE: {e}"
        if rederived != pd:
            out.append(Disagreement("partition_digest", rederived, pd, f"{pdk} does not re-derive from the file's own rows (not derivable by rule)"))
    # 5. content sha (self-consistency)
    if not _content_sha_rederives(doc):
        out.append(Disagreement("content_sha256", "re-derives from the document", doc.get("content_sha256"), "edited after it was written (never derivable by rule)"))
    return out


def population_derived_check(instance: dict[str, Any], population_file: Path | str | dict[str, Any], *, instances: int | None = None,
                             token_select: str | None = None, self_check: str = "fast") -> dict[str, Any]:
    """The verifier's ``V2.population_derived`` check in one call: ``{"status": "matches" | "disagrees", "expected", "disagreements",
    "derived", "compared_only"}``."""

    exp = expected_population(instance, token_select=token_select)
    dis = compare(exp, population_file, instances=instances, self_check=self_check)
    doc, raw = _load_population_doc(population_file)
    return {"schema": "veritor/a1/population-derived-check/v1", "status": "matches" if not dis else "disagrees", "expected": exp.to_json(),
            "disagreements": [dict(d) for d in dis], "derived": sorted(exp["derivable"]), "compared_only": sorted(exp["not_derivable"]),
            "population_file": {"schema": doc.get("schema"), "sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
                                "partition_digest": doc.get("partition_digest"), "partition_digest_kind": doc.get("partition_digest_kind"),
                                "vu_count": doc.get("vu_count"), "ru_count": doc.get("ru_count")},
            "self_check": self_check, "utc": _utc()}


def _write(doc: dict[str, Any], out: str) -> None:
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {p} ({p.stat().st_size} bytes)")


def _attention_arg(arg: str | None, case: str) -> dict[str, Any] | None:
    """[E7 D57] `--attention`: None -> None (Serve_v1); "profile" -> `b1.fa2_kernel_geometry(profile attention consts, D)` of the case's
    SERVE_CASES profile (the derivation E6 uses to FILL the pin; the verifier later trusts the pinned value); else the JSON geometry."""
    if not arg:
        return None
    from verity_vllm.registry import b1
    if arg == "profile":
        prof = ROOT / "manifests" / "semantic-profiles" / f"{SERVE_CASES[case]['profile']}.json"
        consts = b1.profile_attention_consts(json.loads(prof.read_text()))
        g = b1.fa2_kernel_geometry(consts, int(getattr(b1, f"{case}_CONFIG")["D"]))
        return {"kernel": g["kernel"], "kBlockM": g["kBlockM"], "kBlockN": g["kBlockN"], "num_splits": g["num_splits"],
                "traits": g["traits"], "source": g["source"], "profile": SERVE_CASES[case]["profile"]}
    return json.loads(arg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rules", help="rule-generated computed-only kind table at (LP, STEPS) (MODELED)")
    r.add_argument("--lp", type=int, default=6500)
    r.add_argument("--steps", type=int, default=1599)
    r.add_argument("--out", default="")
    r.add_argument("--token-select", default="whole-scan", choices=TOKEN_SELECT_FORMS)
    w = sub.add_parser("widths", help="exhaustive per-kind widths of the prescribed plan (MEASURED; pod at 32x15)")
    w.add_argument("--case", default="TINY")
    w.add_argument("--lp", type=int, default=4)
    w.add_argument("--steps", type=int, default=8)
    w.add_argument("--inputs", default="prescribed", choices=("prescribed", "units"))
    w.add_argument("--workers", type=int, default=1)
    w.add_argument("--chunks", type=int, default=0)
    w.add_argument("--recursion-limit", type=int, default=100_000)
    w.add_argument("--out", required=True)
    c = sub.add_parser("census", help="assemble population-computed-<shape>.json")
    c.add_argument("--case", default="B0")
    c.add_argument("--lp", type=int, required=True)
    c.add_argument("--steps", type=int, required=True)
    c.add_argument("--validation", default="", help="compact_p1_validate.py --inputs prescribed record at this shape")
    c.add_argument("--widths", default="", help="population_v3.py widths record at this shape")
    c.add_argument("--widths-from", default="", help="a population-computed census whose measured width sets are carried over")
    c.add_argument("--no-rules", action="store_true", help="do not evaluate L5's rules (a non-B0 case)")
    c.add_argument("--token-select", default="whole-scan", choices=TOKEN_SELECT_FORMS)
    c.add_argument("--out", required=True)
    p = sub.add_parser("publish", help="the population/v2-shaped file of a census (sampler + certificate + RankLayout read it)")
    p.add_argument("--census", required=True)
    p.add_argument("--rows", default="", help="the validated plan's population.compact.json (MEASURED census; population_v2 build --no-validate ...)")
    p.add_argument("--out", required=True)
    d = sub.add_parser("derive", help="V2.population_derived: derive the expected population from the pinned instance and compare a population file")
    d.add_argument("--instance", required=True, help="protocol_instance.json (veritor/protocol-instance/v1) or a flat {case, lp, steps, plan_name} JSON")
    d.add_argument("--population", default="", help="population file (population/v2 or a9/population-v5); omitted: print the derivation only")
    d.add_argument("--instances", type=int, default=None, help="the epoch's instance (request) count")
    d.add_argument("--token-select", choices=TOKEN_SELECT_FORMS, default=None, help="override the form the plan name implies")
    d.add_argument("--self-check", choices=("fast", "full"), default="fast")
    d.add_argument("--out", default="")
    f = sub.add_parser("freeze", help="[E6 2026-09-11] freeze the TINY or CORE instance: population by rule, inventory, coverage table, protocol_instance.json, pins")
    f.add_argument("population", choices=("tiny", "core", "serve"))
    f.add_argument("--case", default="", help="SERVE: the registry case (b1.<case>_CONFIG; population_v3.SERVE_CASES), e.g. SMOL360 / QWEN05")
    f.add_argument("--lp", type=int, default=4, help="SERVE: prompt length LP")
    f.add_argument("--steps", type=int, default=8, help="SERVE: decode steps (NOUT = STEPS + 1)")
    f.add_argument("--pipeline-dir", default="", help="SERVE: the two-phase rows pipeline's output dir (plan_digest.json + population.v5.json)")
    f.add_argument("--prompt-token-ids", default="", help="SERVE: E1's declared prompt token ids (JSON list); omitted = the PROVISIONAL request (TINY's ids under the model)")
    f.add_argument("--request-source", default="", help="SERVE: where the declared prompt ids come from (E1's run config / workload row), recorded in epoch.inputs")
    f.add_argument("--revision", type=int, default=1, help="SERVE: the r<N> of the instance id")
    f.add_argument("--attention", default=None, help='SERVE [D57]: the declared attention kernel geometry JSON, e.g. {"kernel":"fa2-splitkv-align","kBlockM":64,"kBlockN":128,"num_splits":1} '
                                                       '-> Serve_v4; omitted -> Serve_v1 (legacy kBlockN 64). "profile" = derive it from the case profile via b1.fa2_kernel_geometry')
    f.add_argument("--out", required=True, help="output directory (out/eod/e6/tiny | out/eod/e6/core)")
    f.add_argument("--inventory-dir", default="", help="also write inventory-<pop>.{json,md} here (out/eod/e6)")
    f.add_argument("--literals", choices=LITERALS_FORMS, default="retained", help="TINY: E5's Const decision as applied")
    f.add_argument("--token-select", choices=TOKEN_SELECT_FORMS, default="argmax-scan", help="TINY: the token-selection form as applied")
    f.add_argument("--layout-digest", default="", help="E2's interior layout_digest() (64 hex); omitted = E2's module when importable, else named placeholder")
    f.add_argument("--layout-rule", default="", help="E2's rule name for --layout-digest (e.g. interior-layout/chunks-4096/v2, D13)")
    f.add_argument("--request-index", type=int, default=0, help="TINY: which row of record_v5's deterministic workload is the one declared request")
    f.add_argument("--params", default="", help="CORE: JSON static bindings, default {\"N\": 8, \"R\": 8}")
    f.add_argument("--partition-doc", default="", help="CORE: E8's partition document (Partition.to_json()); omitted = computed by importing e2e.demo_core.circuit")
    f.add_argument("--measured-positions", default="", help="TINY: E3's verifier_v3.locator.inventory(resolver) JSON of the record (streamed); folded + reconciled in the inventory")
    f.add_argument("--declared-inputs", default="", help="CORE: E8's declared_inputs.json (words_in_gate_order + collections) -> epoch.inputs (digest re-derived) [D17]")
    f.add_argument("--population-file", default="", help="TINY: the transformed (from_v5) a9/legacy-minus-inputs/v2 population file the record ships; its rows' v1 digest is re-derived and pinned [D19]")
    f.add_argument("--coverage-table", default="", help="TINY: keep an earlier freeze's coverage table (canonical JSON) as the pin instead of regenerating it from this checkout [D19]")
    f.add_argument("--no-tables", action="store_true", help="TINY: do not pin commitments.tables (the MUFU table roots, E2 table-layout/v1) [D23]")
    f.add_argument("--literals-pin", choices=LITERALS_PIN_MODES, default="auto", help="[D36] commitments.literals (prescribed-literal/v1): auto = pin when E9's LiteralTable is on this checkout, else say so by name; required = refuse without it")
    f.add_argument("--population-doc", default="", help="CORE: the population/v2 file to pin (default: e2e.demo_core.instance.population_doc(partition(**params)))")
    f.add_argument("--request-id", default="core-0", help="CORE: the one request id of the epoch (RU ids become <request>-ru<r>)")
    f.add_argument("--attention-source", default="", help="SERVE [D57]: provenance of the geometry value (E10's MEASURED kernel names + block config); required with --attention")
    tr = sub.add_parser("table-row", help="[NIGHT] the population fields (N_R, N_V, instance) of a measured-table row out/night/table/<model>__<p>x<d>.json")
    tr.add_argument("--case", default="B0")
    tr.add_argument("--prompt-len", type=int, required=True)
    tr.add_argument("--max-tokens", type=int, required=True)
    tr.add_argument("--out-dir", default=str(ROOT / "out" / "night" / "table"))
    tr.add_argument("--instance", default="")
    tr.add_argument("--literals", choices=LITERALS_FORMS, default="compiled")
    tr.add_argument("--token-select", choices=TOKEN_SELECT_FORMS, default="whole-scan")
    a = ap.parse_args(argv)
    if a.cmd == "table-row":
        row = night_table_row(a.case, a.prompt_len, a.max_tokens, Path(a.out_dir), instance_path=a.instance or None, literals=a.literals, token_select=a.token_select)
        print(json.dumps({k: row[k] for k in ("model", "prompt_len", "max_tokens", "N_R", "N_V", "instance")}, indent=1))
        return 0
    if a.cmd == "freeze":
        inv_dir = Path(a.inventory_dir) if a.inventory_dir else None
        if a.population == "tiny":
            pins = freeze_tiny(Path(a.out), literals=a.literals, token_select=a.token_select, layout_digest=a.layout_digest or None, layout_rule=a.layout_rule or None,
                               request_index=a.request_index, inventory_dir=inv_dir, measured_positions=Path(a.measured_positions) if a.measured_positions else None,
                               population_file=Path(a.population_file) if a.population_file else None, coverage_table=Path(a.coverage_table) if a.coverage_table else None,
                               tables=not a.no_tables, literals_pin=a.literals_pin)
        elif a.population == "serve":
            pins = freeze_serve(a.case, a.lp, a.steps, Path(a.out), pipeline_dir=Path(a.pipeline_dir), literals=a.literals, token_select=a.token_select,
                                prompt_token_ids=json.loads(a.prompt_token_ids) if a.prompt_token_ids else None, request_source=a.request_source or None, request_index=a.request_index,
                                layout_digest=a.layout_digest or None, layout_rule=a.layout_rule or None, tables=not a.no_tables, revision=a.revision, literals_pin=a.literals_pin,
                                attention=_attention_arg(a.attention, a.case), attention_source=a.attention_source or None)
        else:
            pins = freeze_core(Path(a.out), params=json.loads(a.params) if a.params else None, layout_digest=a.layout_digest or None, layout_rule=a.layout_rule or None,
                               partition_doc=_load(a.partition_doc) if a.partition_doc else None, inventory_dir=inv_dir,
                               population_doc=_load(a.population_doc) if a.population_doc else None, declared_inputs=Path(a.declared_inputs) if a.declared_inputs else None,
                               request_id=a.request_id)
        print(json.dumps({k: pins[k] for k in ("population", "instance_path", "instance_digest", "N_V", "N_R", "program", "partition", "plan", "interior_layout", "inventory")}, indent=1, default=str))
        return 0
    if a.cmd == "derive":
        inst = _load(a.instance)
        if a.population:
            res = population_derived_check(inst, a.population, instances=a.instances, token_select=a.token_select, self_check=a.self_check)
        else:
            res = {"schema": "veritor/a1/population-derived-check/v1", "status": "derived", "expected": expected_population(inst, token_select=a.token_select).to_json(),
                   "disagreements": None, "utc": _utc()}
        if a.out:
            _write(res, a.out)
        exp = res["expected"]
        print(json.dumps({"status": res["status"], "shape": [exp["lp"], exp["steps"]], "token_select": exp["token_select"], "kinds": len(exp["kinds"]),
                          "vu_count_per_ru": exp["vu_count_per_ru"], "kind_table_digest": exp["kind_table_digest"],
                          "disagreements": res["disagreements"], "population_file": res.get("population_file")}, indent=1))
        return 0 if res["status"] != "disagrees" else 2
    if a.cmd == "publish":
        doc = publish(_load(a.census), _load(a.rows))
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode() + b"\n")  # canonical bytes: a regenerated file is byte-identical
        if doc["census"] != _load(a.census):  # the regrouped partition digest is known only once the rows exist: write it back into the census
            _write(doc["census"], a.census)
        sha = hashlib.sha256(out.read_bytes()).hexdigest()
        print(json.dumps({"out": str(out), "bytes": out.stat().st_size, "file_sha256": sha, "content_sha256": doc["content_sha256"], "vu_count": doc["vu_count"],
                          "rows": len(doc["rows"]), "kinds": len(doc["kinds"]["entries"]), "partition_digest": doc["partition_digest"],
                          "partition_digest_kind": doc["partition_digest_kind"], "self_check": doc["self_check"], "label": doc["label"]}, indent=1))
        return 0
    if a.cmd == "rules":
        doc = build_rules(a.lp, a.steps, token_select=a.token_select)
        summary = {k: doc[k] for k in ("schema", "LP", "STEPS", "vu_count", "ru_count", "kind_table_digest", "plan_digest", "program_digest")}
        summary["kinds"] = len(doc["kinds"])
        summary["prescribed_inputs"] = {k: doc["prescribed_inputs"][k]["gates"] if isinstance(doc["prescribed_inputs"][k], dict) else doc["prescribed_inputs"][k]
                                        for k in ("gates", "prompt_token_ids", "weights")}
        summary["with_inputs"] = doc["with_inputs_for_comparison"]["vu_count"]
        summary["widths"] = {k: doc["widths"][k] for k in ("max_cut_bits", "A_paper", "A_paper_excluding_ArgmaxStep", "A_paper_with_inputs_D4_convention")}
        print(json.dumps(summary, indent=1))
        _write(doc, a.out)
        return 0
    if a.cmd == "widths":
        sys.setrecursionlimit(a.recursion_limit)
        doc = build_widths(a.case, a.lp, a.steps, workers=a.workers, chunks=a.chunks or None, inputs=a.inputs, log=lambda s: print(s, flush=True))
        ne = {k: (v["declared_out_bits"], v["actual_out_bits"]["histogram"]) for k, v in doc["kinds"].items() if not v["declared_equals_actual"]}
        print(f"  {doc['units']} units / {len(doc['kinds'])} kinds in {doc['seconds']}s; escape steps {doc['escape_steps']}; declared != actual: {json.dumps(ne)}")
        _write(doc, a.out)
        return 0
    doc = build_census(a.lp, a.steps, case=a.case, validation=_load(a.validation), widths=_load(a.widths), widths_from=_load(a.widths_from), rules=not a.no_rules,
                       token_select=a.token_select)
    print(json.dumps({"label": doc["label"], "vu_count": doc["vu_count"], "kinds": len(doc["kinds"]), "partition": doc["partition"], "plan_digest": doc["plan_digest"],
                      "agreement": doc["agreement"], "widths_summary": doc["widths_summary"]}, indent=1))
    _write(doc, a.out)
    return 0 if not doc["label"].startswith("INCONSISTENT") else 2


if __name__ == "__main__":
    raise SystemExit(main())
