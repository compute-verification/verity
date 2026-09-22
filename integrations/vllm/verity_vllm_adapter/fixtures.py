"""Persist authenticated GEMM operation fixtures for the numerical lanes (contract section 16.8).

For each GEMM family a few coordinate checks are turned into self-contained fixtures: the opened
input row ``x`` (after W1's ``arg_cols`` slice), the opened weight row ``W[j, :]`` (= column ``j``
of ``W^T``), the committed output row and coordinate ``j``, the Merkle openings themselves
(JSON, verifiable against the bundle's ``commitments.json`` which is copied alongside), the
profile id, the kernel identity and the checker's verdict.  Everything is little-endian BF16 words.

    PYTHONPATH=. python -m verity_vllm_adapter.fixtures <bundle> <coordinate_checks.jsonl> --out <dir> --per-family 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from verity_vllm import identity
from verity_vllm.capture import COMMITMENTS_FILE, CaptureReader
from verity_vllm.commit import TreeCache, load_commitments, verify_opening_in
from verity_vllm.description import Description


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bundle")
    ap.add_argument("checks_jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-family", type=int, default=3)
    ap.add_argument("--experiment-id", default=None)
    args = ap.parse_args(argv)
    reader = CaptureReader(args.bundle)
    description = Description(reader.description)
    commitments = load_commitments(reader)
    cache = TreeCache(reader)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    picked: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(args.checks_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") != "exact" or not r.get("w_id"):
                continue
            key = f"{r['family']}/{r['phase']}"
            if len(picked[key]) < args.per_family:
                picked[key].append(r)
    manifest: dict[str, Any] = {
        "schema": "verity-vllm/gemm-fixtures/v1",
        "bundle": str(args.bundle),
        "experiment_id": args.experiment_id,
        "profile_id": reader.profile_id,
        "weights_format_version": reader.weights.index.format_version,
        "word_format": "little-endian BF16 (uint16) words",
        "fixtures": [],
    }
    shutil.copy2(Path(args.bundle) / COMMITMENTS_FILE, out / COMMITMENTS_FILE)
    for key, rows in sorted(picked.items()):
        for n, r in enumerate(rows):
            name = f"{key.replace('/', '_')}_{n}"
            d = out / name
            d.mkdir(exist_ok=True)
            cid = r["canonical_id"]
            node = description.node(r["op"])
            openings = {}
            for tag, vid in (("x", r["x_id"]), ("w_row", r["w_id"]), ("out_row", cid)):
                o = cache.open(vid)
                assert verify_opening_in(commitments, o), vid
                openings[tag] = o.to_json()
                (d / f"{tag}.u16").write_bytes(o.value)
            cols = (node.extra.get("arg_cols") or {}).get("x")
            meta = {
                "family": r["family"], "phase": r["phase"], "layer": r["layer"], "op": r["op"],
                "canonical_id": cid, "j": r["j"], "K": r["K"], "N": r["N"],
                "x_id": r["x_id"], "x_arg_cols": cols, "w_id": r["w_id"], "weight_param": node.arg("w").param,
                "committed_word": r["committed_word"], "predicted_word": r["predicted_word"],
                "kernel": {"runtime_impl": node.runtime_impl, "runtime_op": node.extra.get("runtime_op"), "kernel_path": r.get("kernel_path"), "coverage": r.get("coverage")},
                "profile_id": reader.profile_id,
                "semantics": "out[j] = bf16_rne(mma.sync m16n8k16 bf16 chain over ascending 16-wide K slices of x[arg_cols] . W[j, :]); openings authenticate against commitments.json (RU tree for x/out rows, weights tree row leaf for W[j, :])",
                "files": {},
            }
            (d / "openings.json").write_text(json.dumps(openings, indent=1, sort_keys=True) + "\n")
            for p in sorted(d.iterdir()):
                meta["files"][p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            (d / "fixture.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
            manifest["fixtures"].append({"name": name, **{k: meta[k] for k in ("family", "phase", "layer", "op", "j", "K", "N", "committed_word")}, "sha256_fixture_json": sha256_file(d / "fixture.json")})
    manifest["commitments_sha256"] = sha256_file(out / COMMITMENTS_FILE)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    manifest["total_bytes"] = total
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    shas = {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name not in ("sha256s.json", "README.md")}
    (out / "sha256s.json").write_text(json.dumps(shas, indent=1, sort_keys=True) + "\n")
    families = sorted({r["family"] for rows in picked.values() for r in rows})
    readme = f"""# Authenticated GEMM operation fixtures ({args.experiment_id or 'experiment'})

Contract section 16.8.  {len(manifest['fixtures'])} fixtures, {total:,} bytes, from bundle profile
`{reader.profile_id}` (capture format weights v{reader.weights.index.format_version}); families {', '.join(families)},
phases prefill and decode, {args.per_family} per (family, phase).  Every fixture was checked
`exact` by the independent GEMM relation (`verity_vllm_numerics.relations.independent_gemm`) on this
pod; `fixture.json` records the committed and predicted words.

Per fixture directory `<family>_<phase>_<n>/`:

- `x.u16` -- the opened input row (all K' words of the producer's row; `x_arg_cols` in
  `fixture.json` says which slice the GEMM consumed), little-endian BF16.
- `w_row.u16` -- the opened weight row `W[j, :]` (= column j of W^T), K words.
- `out_row.u16` -- the committed output row; the checked coordinate is word `j`.
- `openings.json` -- the three Merkle openings (`x`, `w_row`, `out_row`) as
  `verity_vllm.commit.Opening.to_json()`; verify with `verify_opening_in(commitments, opening_from_json(o))`
  against `commitments.json` (copied from the bundle; sha256 in `manifest.json`).
- `fixture.json` -- ids, shape, kernel identity (`runtime_impl`, `runtime_op`, `kernel_path`, coverage
  family), profile id, semantics, per-file sha256.

`sha256s.json` lists every file; `manifest.json` the index.  Use: feed `x[arg_cols]` and `w_row` to
any implementation of the relation and compare with `out_row[j]`.
"""
    (out / "README.md").write_text(readme)
    print(json.dumps({"fixtures": len(manifest["fixtures"]), "total_bytes": total}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
