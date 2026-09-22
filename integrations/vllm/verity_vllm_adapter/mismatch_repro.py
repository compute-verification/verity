"""Compiled/eager mismatch reproducer (CONTRACT §16.8): for one (request, position) pair, the production
record, the full-vocab logits of a compiled-preserving replay and of an eager replay over the *same
production prefix* (both replays follow the production trajectory: the logits are the two profiles'
functions of the same input), the two argmaxes, the resolved kernel paths and the generated-kernel
identity of both profiles.

Inputs are finished replay run directories with stored values (``--mode commit`` or
``--store-values``): the logits row is the committed ``lm_head.gemm`` value of the position (bf16,
``[V]``), read through :class:`verity_vllm.capture.CaptureReader`; nothing is recomputed here.

    python -m verity_vllm_adapter.mismatch_repro --production P --compiled-run C --eager-run E \
        --request 784c37e1 --t 511 --t 529 --out bundles/mismatch-B0-784c37e1-t511
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm import identity
from verity_vllm.capture import CaptureReader


def _bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _find_request(reader: CaptureReader, prefix: str) -> str:
    hits = [r for r in reader.request_ids() if r.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"request prefix {prefix!r} matches {hits} in {reader.run_dir}")
    return hits[0]


def _logits(reader: CaptureReader, rid: str, t: int, profile_id: str) -> tuple[np.ndarray, str, dict[str, Any]]:
    """(float32 logits, canonical id, entry meta) of the lm_head.gemm value at position t."""
    step = reader.boundary(rid).step_of(t)
    if step is None:
        raise SystemExit(f"{rid} has no step for t={t}")
    cid = str(identity.value_id(profile_id, rid, t, step.phase, "lm_head", "lm_head.gemm", 0))
    v = reader.value(cid)
    if v.dtype != "bfloat16":
        raise SystemExit(f"{cid}: dtype {v.dtype}, expected bfloat16")
    return _bf16_to_f32(v.data), cid, {"dtype": v.dtype, "shape": list(v.shape), "sha256": hashlib.sha256(v.data).hexdigest()}


def _production_record(prod: Path, rid: str) -> dict[str, Any]:
    for line in (prod / "production.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec.get("request_id") == rid:
            return rec
    raise SystemExit(f"{rid} not in {prod / 'production.jsonl'}")


def _kernel_evidence(run: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rj = run / "replay.json"
    if rj.exists():
        d = json.loads(rj.read_text())
        out["replay_exec"] = d.get("replay_exec", "eager")
        out["effective_execution"] = d.get("effective_execution")
        out["kernel_identity"] = d.get("kernel_identity")
        out["patches"] = d.get("patches")
        out["replay_restrictions"] = d.get("replay_restrictions")
        out["profile_id"] = d.get("profile_id")
        out["adapter_revision"] = d.get("adapter_revision")
    kj = run / "kernels.json"
    if kj.exists():
        k = json.loads(kj.read_text())
        out["generated_kernels"] = {"summary": k["summary"], "kernels": k["kernels"]}
    mf = run / "manifest.pretty.json"
    if not mf.exists():
        mf = run / "manifest.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        out["manifest_backends"] = m.get("backends")
        out["manifest_compilation"] = m.get("compilation")
        out["manifest_adapter"] = m.get("adapter")
    return out


def _resolved_kernels(run: Path) -> list[dict[str, Any]]:
    """One line per description node: the op, its relation and the kernel that computed it in this profile
    (eager: the resolved runtime implementation path; compiled: the generated kernel and its source hash)."""
    d = json.loads((run / "description.json").read_text())
    nodes = d["nodes"]
    if isinstance(nodes, dict):
        nodes = list(nodes.values())
    rows = []
    for n in nodes:
        row = {k: n.get(k) for k in ("op", "family", "op_type", "layer", "module_path", "relation", "runtime_impl", "kernel_name", "kernel_fn", "kernel_source_sha256", "launch_config") if n.get(k) is not None}
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--production", required=True)
    ap.add_argument("--compiled-run", required=True, help="compiled-nograph replay dir with stored values")
    ap.add_argument("--eager-run", required=True, help="eager replay dir with stored values (same production)")
    ap.add_argument("--request", required=True, help="request id prefix")
    ap.add_argument("--t", type=int, action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prod = Path(args.production)
    comp = CaptureReader(args.compiled_run)
    eag = CaptureReader(args.eager_run)
    rid = _find_request(comp, args.request)
    if _find_request(eag, args.request) != rid:
        raise SystemExit("request id differs between the two runs")
    rec = _production_record(prod, rid)
    (out / "production_record.json").write_text(json.dumps(rec, indent=1) + "\n")
    plen = len(rec.get("prompt_token_ids", []))
    positions = []
    for t in args.t:
        lc, cid_c, meta_c = _logits(comp, rid, t, comp.profile_id)
        le, cid_e, meta_e = _logits(eag, rid, t, eag.profile_id)
        np.save(out / f"logits_compiled_t{t}.npy", lc)
        np.save(out / f"logits_eager_t{t}.npy", le)
        # the token production emitted from position t's logits: generated index t - plen
        # position t's logits produce generated token index t - (plen - 1): the last prompt position emits the first token
        gen_idx = t - (plen - 1)
        outs = rec.get("output_token_ids", [])
        prod_tok = int(outs[gen_idx]) if 0 <= gen_idx < len(outs) else None
        ac, ae = int(lc.argmax()), int(le.argmax())
        top_c = np.argsort(-lc)[:5].tolist()
        top_e = np.argsort(-le)[:5].tolist()
        diff = lc - le
        positions.append({
            "t": t,
            "phase": comp.boundary(rid).step_of(t).phase,
            "production_token": prod_tok,
            "compiled_argmax": ac, "eager_argmax": ae,
            "compiled_top5": [(int(i), float(lc[i])) for i in top_c],
            "eager_top5": [(int(i), float(le[i])) for i in top_e],
            "compiled_logit_of_each": {"production_token": float(lc[prod_tok]) if prod_tok is not None else None, "eager_argmax": float(lc[ae])},
            "eager_logit_of_each": {"production_token": float(le[prod_tok]) if prod_tok is not None else None, "compiled_argmax": float(le[ac])},
            "n_differing_logits": int((diff != 0).sum()), "vocab": int(lc.size),
            "max_abs_diff": float(np.abs(diff).max()), "l2_diff": float(np.linalg.norm(diff)),
            "compiled": {"canonical_id": cid_c, **meta_c, "file": f"logits_compiled_t{t}.npy"},
            "eager": {"canonical_id": cid_e, **meta_e, "file": f"logits_eager_t{t}.npy"},
        })
    doc = {
        "schema": "verity-vllm/mismatch-repro/v1",
        "request_id": rid,
        "prompt_len": plen,
        "production": {"dir": str(prod), **({k: v for k, v in json.loads((prod / "run.json").read_text()).items() if k in ("profile_id", "utc", "stock_vs_stock_identical")} if (prod / "run.json").exists() else {})},
        "positions": positions,
        "compiled_profile": _kernel_evidence(Path(args.compiled_run)),
        "eager_profile": _kernel_evidence(Path(args.eager_run)),
        "note": "logits = committed lm_head.gemm value (bf16 -> float32) of the replay that followed the production trajectory; "
                "both replays saw the identical prefix (production tokens forced), so the two rows are the two profiles' numerical functions of the same input.",
    }
    (out / "mismatch.json").write_text(json.dumps(doc, indent=1) + "\n")
    (out / "kernel_paths.json").write_text(json.dumps({
        "compiled": {"interception": json.loads((Path(args.compiled_run) / "description.json").read_text()).get("interception"), "nodes": _resolved_kernels(Path(args.compiled_run))},
        "eager": {"interception": json.loads((Path(args.eager_run) / "description.json").read_text()).get("interception"), "nodes": _resolved_kernels(Path(args.eager_run))},
    }, indent=1) + "\n")
    # the production profile's own generated-kernel inventory (the compiled replay's kernel_identity compares against it)
    for src, dst in (("kernels.json", "production_kernels.json"), ("kernels.autotune.json", "production_kernels.autotune.json"), ("manifest.pretty.json", "production_manifest.json")):
        if (prod / src).exists():
            (out / dst).write_bytes((prod / src).read_bytes())
    # the two replays' own kernel inventories, identity comparisons against production and manifests
    for label, run in (("compiled", Path(args.compiled_run)), ("eager", Path(args.eager_run))):
        for src in ("kernels.json", "kernel_identity.json", "kernel_identity.autotune.json", "manifest.json"):
            if (run / src).exists():
                (out / f"{label}_replay_{src}").write_bytes((run / src).read_bytes())
    for p in positions:
        print(f"t={p['t']} production={p['production_token']} compiled_argmax={p['compiled_argmax']} eager_argmax={p['eager_argmax']} "
              f"differing_logits={p['n_differing_logits']}/{p['vocab']} max_abs_diff={p['max_abs_diff']:.4g}")


if __name__ == "__main__":
    main()
