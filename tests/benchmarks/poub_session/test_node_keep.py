"""The node form (SERVE=node) keeps neither the probe's served tokens nor the arena's CUDA IPC handle under the run's
output. Its engines write their full export to the prover's scratch (`engine.<g>.json` and `status.<g>.json` in
--engines-dir, run.sh's `$B`); `engine.record` makes what each engine writes to --out (`$O/engine.<g>.json`), and
`node.prover_engines` what prover.json keeps of the exports it read from the scratch. Here the scratch holds stand-ins
of both, and no file under the output holds either value."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "poub_session"))

import engine  # noqa: E402
import node  # noqa: E402

HANDLE = "5e" * 64
PLAIN = [[902117, 441733, 118003, 229051], [771399, 880461, 990137, 313379]]


def _engine_files(scratch: Path, g: int) -> tuple[dict, dict]:
    """An engine's export and decode checks as engine.main makes them, written where it writes them in the scratch."""
    info = {"gpu": g, "uuid": f"GPU-{g}", "bytes": 1 << 30, "weights_bytes": 1 << 29, "weights_blocks": 131072,
            "outside_bytes": 0, "params": 2, "model": "m", "staged": 0, "pack_s": 0.1, "entries": [["a.w", 0, 4096]],
            "handle": HANDLE, "offset": 0, "allocation": 1 << 30, "plain_sha256": "c" * 64}
    checks = {"arena_decodes": True, "probe_tokens_equal": True, "first_block": 12, "hooked_modules": 3,
              "decode_calls": 5, "probe": {"plain": PLAIN, "decoded": PLAIN}}
    engine.write_json(scratch / f"engine.{g}.json", info)
    engine.write_json(scratch / f"status.{g}.json", {"gpu": g, "state": "decoding", "pid": 1, **checks})
    return info, checks


def test_the_node_forms_output_holds_no_served_token_and_no_ipc_handle(tmp_path):
    scratch, out = tmp_path / "B", tmp_path / "run" / "out"
    scratch.mkdir()
    out.mkdir(parents=True)
    for g in (0, 1):
        info, checks = _engine_files(scratch, g)
        a = argparse.Namespace(out=out, model="m", seed=0)
        steps = [{"t": 0.0, "ms": 1.0, "tokens": 4, "running": 1, "phase": "plain"}]
        engine.write_json(out / f"engine.{g}.json", engine.record(info, checks, a, scratch, 5, steps))
    exports = [json.loads((scratch / f"engine.{g}.json").read_text()) for g in (0, 1)]
    (out / "prover.json").write_text(json.dumps({"engines": node.prover_engines(exports)}, indent=1))

    secret = [HANDLE] + [str(t) for ids in PLAIN for t in ids]
    assert all(any(s in p.read_text() for s in secret) for p in scratch.iterdir()), "the stand-ins are in the scratch"
    kept = {p.relative_to(out).as_posix(): p.read_text() for p in out.rglob("*") if p.is_file()}
    assert sorted(kept) == ["engine.0.json", "engine.1.json", "prover.json"]
    assert [(f, s) for f, text in kept.items() for s in secret if s in text] == []
    rec = json.loads(kept["engine.0.json"])
    assert "handle" not in rec and rec["checks"]["probe_tokens_equal"] is True
    assert rec["checks"]["probe"] == {"plain": [4, 4], "decoded": [4, 4]} and rec["weights_blocks"] == 131072
    assert [e["uuid"] for e in json.loads(kept["prover.json"])["engines"]] == ["GPU-0", "GPU-1"]
