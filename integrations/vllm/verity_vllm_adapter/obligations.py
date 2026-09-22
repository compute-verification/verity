"""From the verifier's challenge to a composed ``typed-obligation/v0`` object (INT; GUIDANCE §18.6).

The security-mode challenge (:mod:`verity_vllm_adapter.verify`, ``challenge.json``) samples row VUs of
the bundle.  The generic backend proves one family today -- the BF16 GEMM coordinate
(``veritor.tc-ampere-bf16@1``, PW's path) -- so this module does three things, none of which changes
the sampling policy:

1. :func:`sampled_gemm_rows`: the sampled VUs whose node is a GEMM (``op_type`` ``gemm``/``lm_head``),
   with the sample they came from (certified / coverage / adjacent) and the same-runtime verdict the
   verifier recorded for the whole row;
2. :func:`select_coordinate_units`: a **deterministic sub-selection** of coordinates of those rows to
   prove -- seeded by the challenge digest, so it is a function of the verifier's randomness and not
   the prover's choice -- arranged so the exported obligations *share authenticated values*: two
   coordinates of one committed output row share its ``x`` row; two sampled rows of one GEMM node get
   the same coordinate ``j`` and share the weight row ``W[j, :]``;
3. :func:`input_equations`: sampled ``Input`` equations (IR §E) -- weight words of the opened ``W[j, :]``
   rows pinned to the registered checkpoint's words (read from the safetensors file without torch,
   :func:`verity_vllm_adapter.weight_sources.checkpoint_row_words`), and the request's boundary leaves
   (prompt tokens and the bound output tokens of WB2's format) pinned to the production record --
   one ``Input<32>`` gate per token leaf, authenticated against the boundary root.

:func:`build` assembles the object with :func:`verity_vllm.typed_obligation_export.export_composed` and
returns the record plus the composition report.  Everything here runs on CPU from the bundle alone.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm import identity
from verity_vllm import relations as R
from verity_vllm.capture import CaptureReader, boundary_tree_id, ru_tree_id
from verity_vllm.commit import Opening, TreeCache, verify_opening_in
from verity_vllm.description import Description
from verity_vllm.typed_obligation_export import (
    ComposedExport,
    CoordinateUnit,
    InputEquation,
    composed_record,
    export_composed,
)
from verity_vllm_adapter.coordinate_check import GEMM_TYPES, CoordinateOpenings, coordinate_openings, family_of
from verity_vllm_adapter.weight_sources import checkpoint_row_words

__all__ = ["SampledRow", "sampled_gemm_rows", "sampled_producers", "select_coordinate_units", "chain_record", "input_equations", "boundary_input_equations",
           "output_copy_checks", "build"]


@dataclass(frozen=True)
class SampledRow:
    canonical_id: str
    op: str
    op_type: str
    family: str
    phase: str
    layer: str
    request_id: str
    N: int
    samples: tuple[str, ...]  # certified | coverage | adjacent
    verifier_status: str | None  # the same-runtime whole-row verdict from verify.json, if given


def sampled_gemm_rows(challenge: dict[str, Any], description: Description, reader: CaptureReader,
                      verify_results: list[dict[str, Any]] | None = None) -> list[SampledRow]:
    """The challenge's sampled VUs that are GEMM rows, in the challenge's sample order (certified first)."""
    status = {r["canonical_id"]: r.get("status") for r in (verify_results or [])}
    membership: dict[str, list[str]] = {}
    for key, name in (("certified_sample", "certified"), ("coverage_sample", "coverage"), ("adjacent_vus", "adjacent")):
        for cid in challenge.get(key) or []:
            membership.setdefault(cid, []).append(name)
    if not membership:
        for cid in challenge.get("sample") or []:
            membership.setdefault(cid, []).append("sample")
    rows: list[SampledRow] = []
    for cid, samples in membership.items():
        parsed = identity.parse(cid)
        if parsed.op is None or parsed.request_id is None:
            continue
        node = description.node(parsed.op)
        if node.op_type not in GEMM_TYPES:
            continue
        inst = description.instance_of(cid, reader.boundary(parsed.request_id))
        N = int(description.resolve_shape(node.port_shape(inst.port))[0])
        rows.append(SampledRow(cid, node.op, node.op_type, family_of(node), inst.phase, node.layer, parsed.request_id, N,
                               tuple(samples), status.get(cid)))
    return rows


def _rng(challenge: dict[str, Any], tag: str) -> random.Random:
    seed = f"{challenge.get('digest') or challenge.get('challenge_digest') or ''}/{tag}"
    return random.Random(seed)


@dataclass
class Selection:
    units: list[tuple[str, int]]  # (canonical_id, j)
    shared_x: list[dict[str, Any]] = field(default_factory=list)
    shared_w: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"units": [{"canonical_id": c, "j": j} for c, j in self.units], "shared_x_rows": self.shared_x, "shared_w_rows": self.shared_w, "note": self.note}


def sampled_producers(rows: list[SampledRow], description: Description, reader: CaptureReader, challenge: dict[str, Any],
                      verify_results: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """The sampled GEMM rows whose ``x`` row is *itself the output of a sampled VU* (the output-feeds-input
    chain; INT3).  ``x`` is resolved as the coordinate check resolves it (``resolve_args(inst)["x"]``, the
    producer's row at the same position); the producer is "sampled" when the challenge's certified,
    coverage or adjacent sample names it.  Keyed by the consuming row's canonical id."""
    sampled: dict[str, list[str]] = {}
    for key, name in (("certified_sample", "certified"), ("coverage_sample", "coverage"), ("adjacent_vus", "adjacent")):
        for cid in challenge.get(key) or []:
            sampled.setdefault(cid, []).append(name)
    status = {r["canonical_id"]: r.get("status") for r in (verify_results or [])}
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        try:
            inst = description.instance_of(r.canonical_id, reader.boundary(r.request_id))
            x = description.resolve_args(inst, reader.boundary(r.request_id), reader.profile_id).get("x")
        except Exception:  # noqa: BLE001 -- a row whose x does not resolve is simply not a chain candidate
            continue
        if not isinstance(x, str) or x not in sampled:
            continue
        px = identity.parse(x)
        out[r.canonical_id] = {"consumer": r.canonical_id, "consumer_op": r.op, "consumer_family": r.family, "producer": x, "producer_op": px.op,
                               "producer_samples": sampled[x], "producer_verifier_status": status.get(x), "consumer_samples": list(r.samples),
                               "consumer_verifier_status": r.verifier_status}
    return out


def select_coordinate_units(rows: list[SampledRow], challenge: dict[str, Any], *, rows_sharing_x: int = 2, coords_per_row: int = 2,
                            ops_sharing_w: int = 1, prefer_status: str = "exact", chain_candidates: dict[str, dict[str, Any]] | None = None) -> Selection:
    """A deterministic sub-selection of GEMM coordinates of the sampled rows to prove.

    ``rows_sharing_x`` rows are drawn (certified sample first, then coverage); each contributes
    ``coords_per_row`` distinct coordinates ``j`` -- those obligations share the row's authenticated
    ``x``.  Then ``ops_sharing_w`` GEMM nodes with at least two sampled rows are drawn; both rows get
    one common ``j`` -- those obligations share the authenticated weight row ``W[j, :]``.  Rows the
    verifier did not mark ``prefer_status`` are avoided when any other row exists (a proof of a row
    the verifier already rejected proves nothing new about the run).  The draw is seeded by the
    challenge digest: the same challenge yields the same units.

    ``chain_candidates`` (INT3, :func:`sampled_producers`): when given and non-empty, the first of the
    ``rows_sharing_x`` rows is drawn from the rows whose producer VU is itself sampled (and, when any is,
    verified ``prefer_status``), so the exported object realises one output-feeds-input chain when the
    verifier's sample offers one.  The sampling policy is untouched: only which sampled rows are proved."""
    if not rows:
        return Selection([], note="no GEMM row in the sampled VUs")
    rng = _rng(challenge, "coordinate-units")
    good = [r for r in rows if r.verifier_status in (None, prefer_status)] or rows
    ranked = sorted(good, key=lambda r: (0 if "certified" in r.samples else 1, r.canonical_id))
    certified = [r for r in ranked if "certified" in r.samples] or ranked
    units: list[tuple[str, int]] = []
    shared_x: list[dict[str, Any]] = []
    chain_note = ""
    chosen_rows: list[SampledRow] = []
    if chain_candidates:
        chain_rows = [r for r in ranked if r.canonical_id in chain_candidates]
        exact_chain = [r for r in chain_rows if chain_candidates[r.canonical_id].get("producer_verifier_status") in (None, prefer_status)] or chain_rows
        if exact_chain:
            first = rng.choice(sorted(exact_chain, key=lambda r: r.canonical_id))
            chosen_rows.append(first)
            chain_note = f"; chain row {first.canonical_id} drawn first from {len(exact_chain)} rows whose producer VU is sampled"
    pool = [r for r in certified if r not in chosen_rows]
    chosen_rows += rng.sample(pool, min(max(rows_sharing_x - len(chosen_rows), 0), len(pool)))
    for r in chosen_rows:
        js = sorted(rng.sample(range(r.N), min(coords_per_row, r.N)))
        units += [(r.canonical_id, j) for j in js]
        shared_x.append({"row": r.canonical_id, "j": js, "samples": list(r.samples)})
    by_op: dict[str, list[SampledRow]] = {}
    for r in ranked:
        by_op.setdefault(r.op, []).append(r)
    multi = sorted(op for op, rs in by_op.items() if len({x.canonical_id for x in rs}) >= 2)
    shared_w: list[dict[str, Any]] = []
    for op in rng.sample(multi, min(ops_sharing_w, len(multi))):
        pair = rng.sample(by_op[op], 2)
        j = rng.randrange(pair[0].N)
        for r in pair:
            if (r.canonical_id, j) not in units:
                units.append((r.canonical_id, j))
        shared_w.append({"op": op, "j": j, "rows": [r.canonical_id for r in pair], "samples": [list(r.samples) for r in pair]})
    note = (f"{len(units)} coordinates of {len({c for c, _ in units})} sampled GEMM rows; seeded by the challenge digest; "
            f"{len(multi)} GEMM nodes had >= 2 sampled rows" + chain_note)
    return Selection(units, shared_x, shared_w, note)


def chain_record(selection: Selection, units: list[CoordinateOpenings], producers: dict[str, dict[str, Any]],
                 verify_results: list[dict[str, Any]] | None, challenge: dict[str, Any]) -> dict[str, Any]:
    """What the exported object realises of the output-feeds-input chain: every opened coordinate unit whose
    ``x`` row is a sampled VU's output, with the producer's sample membership and the verifier's verdict on
    it.  ``realised`` is True when at least one such unit was opened.  ``candidates`` counts the sampled
    GEMM rows that could have offered one (0 = the sample offered none: a scope statement, not a failure)."""
    status = {r["canonical_id"]: r.get("status") for r in (verify_results or [])}
    links: list[dict[str, Any]] = []
    for u in units:
        p = producers.get(u.canonical_id)
        if p is None or p["producer"] != u.x_id:
            continue
        links.append({"consumer": u.canonical_id, "consumer_op": u.op, "family": u.family, "j": u.j, "x_id": u.x_id, "producer_op": p["producer_op"],
                      "producer_samples": p["producer_samples"], "producer_verifier_status": status.get(u.x_id, p.get("producer_verifier_status")),
                      "x_cols": None if u.x_cols is None else [u.x_cols[0], u.x_cols[-1] + 1],
                      "meaning": "the obligation's x operand is the authenticated leaf the verifier checked as the producer's output: "
                                 "the same committed interior value is the producer's checked output and this proved coordinate's input"})
    return {"realised": bool(links), "links": links, "candidates": len(producers),
            "units_selected": len(selection.units), "note": ("realised: " + "; ".join(f"{l['producer_op']} -> {l['consumer_op']}[j={l['j']}]" for l in links[:4])) if links
            else (f"not realised: {len(producers)} sampled GEMM row(s) had a sampled producer but none was opened" if producers
                  else "not realised: no sampled GEMM row's producer VU was in the verifier's sample (certified + coverage + adjacent)")}


def open_units(description: Description, reader: CaptureReader, commitments: dict[str, Any], cache: TreeCache,
               selection: Selection) -> tuple[list[CoordinateOpenings], list[dict[str, Any]]]:
    """Authenticated openings for every selected coordinate; the ones that cannot be opened are reported, not dropped silently."""
    opened: list[CoordinateOpenings] = []
    problems: list[dict[str, Any]] = []
    for cid, j in selection.units:
        try:
            opened.append(coordinate_openings(description, reader, commitments, cache, cid, j))
        except (R.RelationError, KeyError) as e:
            problems.append({"canonical_id": cid, "j": j, "problem": f"{type(e).__name__}: {e}"})
    return opened, problems


def input_equations(units: list[CoordinateOpenings], challenge: dict[str, Any], *, model_dir: str | Path, expected_file_sha256: dict[str, str] | None,
                    per_unit: int = 2) -> tuple[list[InputEquation], list[dict[str, Any]]]:
    """``per_unit`` weight words of each opened ``W[j, :]`` row pinned to the checkpoint's words (one equation per
    distinct word; the same weight row opened by two units is pinned once).  Every pin's expected value is the
    *checkpoint's* word -- if the committed word differs, the relation (and the proof) rejects."""
    rng = _rng(challenge, "input-equations/weights")
    eqs: list[InputEquation] = []
    seen: set[tuple[str, int]] = set()
    report: list[dict[str, Any]] = []
    rows_cache: dict[tuple[str, int], tuple[bytes, dict[str, Any]]] = {}
    for u in units:
        key = (u.param, u.j)
        if key not in rows_cache:
            rows_cache[key] = checkpoint_row_words(model_dir, u.param, u.j, expected_file_sha256=expected_file_sha256)
        data, source = rows_cache[key]
        width = 2
        for k in sorted(rng.sample(range(u.K), min(per_unit, u.K))):
            if (u.w_id, k) in seen:
                continue
            seen.add((u.w_id, k))
            expected = int.from_bytes(data[k * width:(k + 1) * width], "little")
            prefix: tuple[int, ...] = (0,) if tuple(u.w_row.shape) == (1, u.K) else ()
            eqs.append(InputEquation(u.w_row, (*prefix, k), expected, f"checkpoint:{source['file']}@{source['file_sha256'][:12]}:{source['tensor']}[{source['local_row']}][{k}]"))
            report.append({"leaf": u.w_id, "k": k, "expected": expected, "source": source})
    return eqs, report


def boundary_input_equations(reader: CaptureReader, commitments: dict[str, Any], cache: TreeCache, production_rows: list[dict[str, Any]],
                             challenge: dict[str, Any], *, request_ids: list[str], prompt_per_request: int = 1, outputs_per_request: int = 1,
                             ) -> tuple[list[InputEquation], list[dict[str, Any]]]:
    """Boundary leaves pinned to the production record: ``prompt_per_request`` prompt tokens and ``outputs_per_request``
    bound output tokens per request, each an ``Input<32>`` equation whose expected value is the token the API saw
    (``production.jsonl``) and whose committed position is the boundary tree's leaf (WB2's format, ``int32``)."""
    rng = _rng(challenge, "input-equations/boundary")
    by_rid = {r["request_id"]: r for r in production_rows}
    eqs: list[InputEquation] = []
    report: list[dict[str, Any]] = []
    for rid in request_ids:
        rec = by_rid.get(rid)
        if rec is None:
            report.append({"request_id": rid, "problem": "no production record"})
            continue
        b = reader.boundary(rid)
        prompt = list(rec.get("prompt_token_ids") or b.prompt_token_ids)
        outputs = list(rec.get("output_token_ids") or [])
        picks = [("prompt_token", t, int(prompt[t])) for t in sorted(rng.sample(range(len(prompt)), min(prompt_per_request, len(prompt))))]
        present = [(k, r) for k, r in enumerate(b.output_records) if r.present]  # the k-th record is the API's k-th output token
        for k, r in rng.sample(present, min(outputs_per_request, len(present))):
            api_tok = outputs[k] if k < len(outputs) else int(r.token_id)
            picks.append(("output_token", r.t, int(api_tok)))
        for fld, t, expected in sorted(picks):
            cid = str(identity.boundary_id(reader.profile_id, rid, fld, t))
            o = cache.open(cid)
            assert isinstance(o, Opening)
            u32 = expected & 0xFFFFFFFF
            # an int32 leaf is two Value<16> words (low half first); pinning both fixes the token exactly
            eqs.append(InputEquation(o, (0,), u32 & 0xFFFF, f"production.jsonl:{rid}:{fld}[{t}] low half"))
            eqs.append(InputEquation(o, (1,), u32 >> 16, f"production.jsonl:{rid}:{fld}[{t}] high half"))
            report.append({"leaf": cid, "field": fld, "t": t, "expected": expected, "committed": int.from_bytes(o.value, "little", signed=True),
                           "words": [u32 & 0xFFFF, u32 >> 16]})
    return eqs, report


#: The interior copy of a served output token: the ``sampler.select`` node's one port, written by the
#: replay at the request's sampled position of each step (``verity_vllm_adapter.capture.Capture._write_ru``).
OUTPUT_COPY_OP = "sampler.select"


def output_copy_checks(reader: CaptureReader, commitments: dict[str, Any], cache: TreeCache, *,
                       request_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """PROTOCOL.md R4, the served output tokens: the step-1 boundary leaf ``output_token/t`` and the
    step-3 interior leaf ``…/l=sampler/op=sampler.select/port=0`` of the RU that sampled position ``t``
    are two physical representations of one gate (the TokenSelect/ArgmaxStep output of step ``t`` =
    the circuit output ``y_t``).  The replay writes the interior copy from the *teacher-forced* token,
    so an honest prover's two copies agree by construction -- but nothing checked it.  This is the
    explicit agreement check: one row per interior copy of every committed RU tree of the requests,
    both openings authenticated against their committed roots (``commitments.json``: the phase-2
    record carries the boundary trees and the RU trees), the two ``int32`` words compared exactly.

    A row is ``ok`` iff both openings authenticate and the words are equal; an interior copy at a
    position with no present boundary record, or a present boundary record whose position has no
    interior copy in any committed RU tree of the request, is a row with ``ok: False`` (a value
    committed once is not a value committed twice consistently).  A request none of whose RU trees
    is committed (``R`` not in ``J``) has no second representation and no rows.  No layout, digest
    or statement changes: a verifier-side check over the record as it is (rows are what
    ``rec_p1_verify`` would log).
    """

    rows: list[dict[str, Any]] = []
    for rid in request_ids if request_ids is not None else reader.request_ids():
        b = reader.boundary(rid)
        boundary_tid = boundary_tree_id(rid)
        served = {r.t: int(r.token_id) for r in b.output_records if r.present}
        interior: dict[int, tuple[str, str]] = {}
        committed_rus = [ru_tree_id(rid, step) for _, step in reader.ru_keys(rid) if ru_tree_id(rid, step) in commitments.get("trees", {})]
        if not committed_rus:
            continue  # R not in J: step 3 committed nothing for it, there is no second representation to compare
        for tid in committed_rus:
            for cid in cache.tree(tid).ids:
                parsed = identity.parse(cid)
                if parsed.op == OUTPUT_COPY_OP and parsed.port == 0 and parsed.t is not None:
                    if parsed.t in interior:
                        rows.append({"request_id": rid, "t": parsed.t, "kind": "output_copy", "ok": False,
                                     "problem": f"two interior copies: {interior[parsed.t][1]} and {cid}"})
                    interior[parsed.t] = (tid, cid)
        for t in sorted(set(served) | set(interior)):
            row: dict[str, Any] = {"request_id": rid, "t": t, "kind": "output_copy", "boundary_tree": boundary_tid,
                                   "boundary_leaf": str(identity.boundary_id(reader.profile_id, rid, "output_token", t)),
                                   "interior_tree": interior.get(t, (None, None))[0], "interior_leaf": interior.get(t, (None, None))[1]}
            if t not in served:
                row.update(ok=False, problem="interior copy at a position with no present output record")
            elif t not in interior:
                row.update(ok=False, problem="served output with no interior copy in any committed RU tree of the request")
            else:
                ob = cache.open(row["boundary_leaf"])
                oi = cache.open(interior[t][1])
                assert isinstance(ob, Opening) and isinstance(oi, Opening)
                b_ok, i_ok = verify_opening_in(commitments, ob), verify_opening_in(commitments, oi)
                b_tok = int.from_bytes(ob.value, "little", signed=True)
                i_tok = int.from_bytes(oi.value, "little", signed=True)
                row.update(boundary_authenticates=b_ok, interior_authenticates=i_ok, boundary_token=b_tok, interior_token=i_tok,
                           served_token=served[t], ok=bool(b_ok and i_ok and b_tok == i_tok == served[t]))
                if not row["ok"]:
                    row["problem"] = "openings do not authenticate" if not (b_ok and i_ok) else "interior copy differs from the step-1 output token"
            rows.append(row)
    return rows


def build(commitments: dict[str, Any], units: list[CoordinateOpenings], equations: list[InputEquation], *, profile_id: str,
          name: str, description: str, source: str) -> tuple[ComposedExport, dict[str, Any]]:
    """The composed object and its record (statement, witness, verdicts)."""
    cus = [CoordinateUnit(u.canonical_id, u.j, u.x, u.w_row, u.out_row, u.x_cols) for u in units]
    export = export_composed(commitments, cus, equations, profile_id=profile_id)
    record = composed_record(export, name=name, description=description, source=source)
    record["provenance"]["units"] = [u.meta() for u in units]
    return export, record


def load_challenge(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
