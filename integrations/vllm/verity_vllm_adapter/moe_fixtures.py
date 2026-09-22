"""Authenticated MoE operation fixtures (contract §16.8, W9 lane).

``extract`` takes a replay+commit run and one (request, layer, position) and writes a small,
self-contained directory holding *authenticated openings* (value bytes + Merkle path, one per
canonical id) for the MoE block at that token — router logits, top-k ids and weights, the block
input row, the active expert slots' outputs, every inactive slot (derived zeros), the combination,
the shared-expert path and the block's residual add — plus everything a fresh process needs to
re-run the registered relations on them: the roots of the trees they open under, the request's
boundary, the description, the profile manifest. The active expert's per-expert weight leaves
(``w=<param>.e<slot>``) are included as openings under the weights root so the expert relation is
re-runnable from the fixture alone (~17 MB for Qwen1.5-MoE); the rest of the checkpoint is
referenced by sha256 (``manifest.model.weights_sha256``).

``verify`` re-authenticates every opening against the recorded roots and runs
``verity_vllm.relations.check`` on the fixture's VUs with a cache that serves the stored openings
(never the run directory): the independent relations (router ids, combination, cast) run on CPU;
the same-runtime ones (router weights, expert slot, shared-expert sigmoid/mul, residual add) run
when ``verity_vllm_adapter.relations_gpu`` imports (torch + vLLM + GPU) and are reported as
``skipped`` otherwise.

    python -m verity_vllm_adapter.moe_fixtures extract RUN_DIR --out DIR [--request N|ID] [--layer L] [--t T]
    python -m verity_vllm_adapter.moe_fixtures verify DIR [--gpu]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from verity_vllm import identity, relations
from verity_vllm import commit as vcommit
from verity_vllm.capture import Boundary, CaptureReader
from verity_vllm.commit import Opening, TreeCache, verify_opening_in
from verity_vllm.description import Description

SCHEMA = "verity-vllm/moe-fixtures/v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- extract -------------------------------------------------------------------------------------


def moe_ids(description: Description, reader: CaptureReader, profile_id: str, request_id: str, layer: str, t: int) -> dict[str, Any]:
    """The MoE block's value ids at (request, layer, t): VUs to check, and their inputs."""
    boundary = reader.boundary(request_id)
    phase = "prefill" if t < len(boundary.prompt_token_ids) else "decode"
    pre = f"layers.{layer}."
    names = {
        "router_logits": pre + "mlp.gate.gemm",
        "topk_ids": pre + "mlp.experts.topk",
        "topk_weights": pre + "mlp.experts.topk_weights",
        "expert_out": pre + "mlp.experts.expert_out",
        "combine": pre + "mlp.experts.combine",
        "block_input": pre + "post_attention_layernorm.rmsnorm",
        "shared_gate": pre + "mlp.shared_expert_gate.gemm",
        "shared_sigmoid": pre + "mlp.shared_expert.sigmoid",
        "shared_mul": pre + "mlp.shared_expert.mul",
        "shared_down": pre + "mlp.shared_expert.down_proj.gemm",
        "add": pre + "mlp.experts.add",
    }
    present = {k: v for k, v in names.items() if v in description}

    def vid(op: str, e: int | None = None) -> str:
        node = description.node(op)
        return str(identity.CanonicalId(kind="value", profile_id=profile_id, request_id=request_id, t=t, phase=phase,
                                        layer=node.layer, op=op, port=0, expert=e))

    out: dict[str, Any] = {"phase": phase, "nodes": present, "vus": {}, "active": [], "inactive": []}
    for k, op in present.items():
        if k == "expert_out":
            continue
        out["vus"][k] = vid(op)
    if "expert_out" in present:
        node = description.node(present["expert_out"])
        ids_val = reader.value(vid(present["topk_ids"]))
        active = [int(x) for x in ids_val.array().reshape(-1)]
        out["active"] = active
        out["inactive"] = [e for e in range(node.experts or 0) if e not in active]
        out["vus"]["expert_out_active"] = [vid(present["expert_out"], e) for e in active]
        out["vus"]["expert_out_inactive_sample"] = vid(present["expert_out"], out["inactive"][0]) if out["inactive"] else None
        out["slot_family"] = vid(present["expert_out"]).rsplit("/port=", 1)[0] + "/port=0"
    return out


def extract(run_dir: Path, out: Path, request: str | None, layer: str, t: int | None) -> dict[str, Any]:
    reader = CaptureReader(run_dir)
    description = Description(reader.description)
    commitments = vcommit.load_commitments(reader)
    profile_id = reader.profile_id
    rids = reader.request_ids()
    if request is None:
        request_id = rids[0]
    elif request.isdigit() and request not in rids:
        request_id = rids[int(request)]
    else:
        request_id = request
    boundary = reader.boundary(request_id)
    if t is None:
        t = len(boundary.prompt_token_ids)  # first decode position: its step's tree is small
    sel = moe_ids(description, reader, profile_id, request_id, layer, t)
    cache = TreeCache(reader)

    openings: dict[str, dict[str, Any]] = {}
    roles: dict[str, Any] = {}

    def add(cid: str, role: str) -> None:
        if cid in openings:
            return
        o = cache.open(cid)
        assert verify_opening_in(commitments, o), cid
        openings[cid] = o.to_json()
        roles.setdefault(role, []).append(cid)

    # the VUs and, through the description, every input they open
    vus: list[str] = []
    weights_for = (sel["vus"].get("expert_out_active") or [None])[0]  # one active expert's weights (~17 MB)
    for role, v in sel["vus"].items():
        for cid in (v if isinstance(v, list) else [v]):
            if cid is None:
                continue
            vus.append(cid)
            add(cid, role)
            inst = description.instance_of(cid, boundary)
            for name, ref in description.resolve_args(inst, boundary, profile_id).items():
                if isinstance(ref, dict):
                    continue
                for r in (ref if isinstance(ref, list) else [ref]):
                    if identity.parse(r).kind == "weight":
                        # per-expert weight leaves for ONE active slot (the fixture's re-runnable
                        # expert) and small dense weights (router gate, norms, shared-expert gate,
                        # <= 1 MB); every other weight is referenced by id + checkpoint sha, not copied
                        small = reader.weights.entry(r).nbytes <= (1 << 20) if r in reader.weights else False
                        if (cid == weights_for and inst.expert is not None) or small:
                            add(r, f"weights:{name}")
                        elif r not in roles.setdefault("weights_referenced", []):
                            roles["weights_referenced"].append(r)
                        continue
                    add(r, f"input:{role}:{name}")
    trees = sorted({o["tree"] for o in openings.values()})
    doc = {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(run_dir),
        "profile_id": profile_id,
        "model": reader.manifest.get("model"),
        "adapter": reader.manifest.get("adapter"),
        "selection": {"request_id": request_id, "layer": layer, "t": t, "phase": sel["phase"],
                      "router_active_slots": sel["active"], "inactive_slot_in_fixture": sel["inactive"][:1],
                      "inactive_slots_total": len(sel["inactive"]), "slot_family": sel.get("slot_family")},
        "nodes": sel["nodes"],
        "vus": vus,
        "roles": roles,
        "roots": {tid: commitments["trees"][tid] for tid in trees},
        "commitments_digest": hashlib.sha256(json.dumps(commitments, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "boundary": boundary.to_json(),
        "openings": openings,
        "description_sha256": hashlib.sha256(json.dumps(reader.description, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "note": "openings are (canonical id, dtype, shape, value bytes, Merkle path) under the recorded root of their tree; "
                "inactive expert slots are derived zeros (never captured); the active expert's w13/w2 per-expert leaves are "
                "included so moe.expert_slot.same_runtime.v1 is re-runnable; other weights referenced by manifest.model.weights_sha256",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "fixture.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    (out / "description.json").write_text(json.dumps(reader.description, indent=1, sort_keys=True) + "\n")
    (out / "manifest.json").write_bytes((run_dir / "manifest.json").read_bytes())
    shas = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.name != "SHA256SUMS.json"}
    (out / "SHA256SUMS.json").write_text(json.dumps(shas, indent=1, sort_keys=True) + "\n")
    return {"vus": len(vus), "openings": len(openings), "bytes": sum(v["bytes"] for v in shas.values()), "files": shas}


# --- verify --------------------------------------------------------------------------------------


class FixtureCache:
    """Serves stored openings; ``relations.check`` never touches a run directory."""

    def __init__(self, openings: dict[str, Opening]) -> None:
        self._o = openings

    def open(self, canonical_id: str) -> Opening:
        try:
            return self._o[canonical_id]
        except KeyError as error:
            raise KeyError(f"{canonical_id} is not in the fixture") from error


class FixtureReader(CaptureReader):
    """The two things ``relations.check`` asks a reader for (profile id, the request's boundary),
    without a run directory; every value comes from the openings via ``FixtureCache``."""

    def __init__(self, boundary: Boundary, profile_id: str) -> None:  # noqa: D107 (no super: no directory)
        self._b = boundary
        self._pid = profile_id

    @property
    def profile_id(self) -> str:  # type: ignore[override]
        return self._pid

    def boundary(self, request_id: str) -> Boundary:
        if request_id != self._b.request_id:
            raise KeyError(request_id)
        return self._b


def verify(fixture_dir: Path, gpu: bool) -> dict[str, Any]:
    shas = json.loads((fixture_dir / "SHA256SUMS.json").read_text())
    bad = [n for n, e in shas.items() if sha256_file(fixture_dir / n) != e["sha256"]]
    if bad:
        raise SystemExit(f"sha256 mismatch: {bad}")
    doc = json.loads((fixture_dir / "fixture.json").read_text())
    description = Description(json.loads((fixture_dir / "description.json").read_text()))
    openings = {cid: Opening.from_json(o) for cid, o in doc["openings"].items()}
    commitments = {"trees": doc["roots"]}
    auth = {cid: verify_opening_in(commitments, o) for cid, o in openings.items()}
    if gpu:
        import verity_vllm_adapter.relations_gpu  # noqa: F401  (registers the same-runtime checkers)
    else:
        try:
            import verity_vllm_sampler.registry as sampler_registry

            sampler_registry.install()
        except Exception:  # noqa: BLE001
            pass
    boundary = Boundary.from_json(doc["boundary"])
    report = relations.check(description, FixtureReader(boundary, doc["profile_id"]), commitments, doc["vus"],
                             cache=FixtureCache(openings))  # type: ignore[arg-type]
    results = []
    referenced = set(doc["roles"].get("weights_referenced", []))
    for r in report.results:
        status = r.status
        if status == "unknown_relation" and not gpu and r.relation_id and r.relation_id.endswith("same_runtime.v1"):
            status = "skipped (same-runtime checker needs torch + vLLM + GPU: run with --gpu)"
        if status == "missing" and any(w in (r.detail or "") for w in referenced):
            status = "skipped (input weight referenced by checkpoint sha, not copied into the fixture)"
        results.append({"canonical_id": r.canonical_id, "relation": r.relation_id, "status": status, "detail": r.detail,
                        "inputs_opened": r.inputs_opened, "input_bytes": r.input_bytes})
    out = {
        "fixture": str(fixture_dir),
        "openings": len(openings), "authenticated": sum(auth.values()),
        "vus": len(doc["vus"]),
        "results": results,
        "summary": {s: sum(1 for r in results if r["status"].split(" ")[0] == s) for s in sorted({r["status"].split(" ")[0] for r in results})},
        "openings_verified_by_check": report.openings_verified, "openings_attempted_by_check": report.openings_attempted,
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("run_dir")
    ex.add_argument("--out", required=True)
    ex.add_argument("--request", default=None)
    ex.add_argument("--layer", default="0")
    ex.add_argument("--t", type=int, default=None)
    ve = sub.add_parser("verify")
    ve.add_argument("fixture_dir")
    ve.add_argument("--gpu", action="store_true")
    ve.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "extract":
        res = extract(Path(args.run_dir), Path(args.out), args.request, args.layer, args.t)
        print(json.dumps({k: v for k, v in res.items() if k != "files"}))
        for n, e in res["files"].items():
            print(f"  {e['sha256']}  {e['bytes']:>10}  {n}")
        return 0
    res = verify(Path(args.fixture_dir), args.gpu)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "results"}))
    for r in res["results"]:
        print(f"  {r['status']:<10} {r['relation'] or '-':<40} {r['canonical_id'].split('/r=')[1][37:]}" + (f"  {r['detail']}" if r["detail"] else ""))
    return 0 if all(r["status"].split(" ")[0] in ("exact", "skipped") for r in res["results"]) else 1


if __name__ == "__main__":
    sys.exit(main())
