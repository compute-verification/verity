"""run.sh publishes no request value into $OUT, which custody uploads: it refuses an art: row dir or serve run (a published
one's sealed bundle or requests.jsonl holds them) before it writes anything, and a node's serve dir, with a prompt planted
in its requests.jsonl, its scheduler and load traces and its advice's opening, reaches $OUT only as its public records."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

RUN_SH = Path(__file__).resolve().parents[1] / "run.sh"
PROMPT = [918273, 827364, 736455]
NEEDLES = (b"918273", b"prompt_ids", b"streamed_ids", b"prompt_token_ids")


def block(first: str, last: str) -> str:
    """run.sh's lines from the one starting with ``first`` through the next one starting with ``last``."""
    lines = RUN_SH.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith(first))
    j = next(k for k in range(i, len(lines)) if lines[k].startswith(last))
    return "\n".join(lines[i:j + 1]) + "\n"


def put(root: Path, files: dict[str, object]) -> None:
    for name, doc in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("".join(json.dumps(d) + "\n" for d in doc) if name.endswith(".jsonl") else json.dumps(doc))


def published(out: Path) -> list[str]:
    """$OUT's files by name, after holding that none carries a planted request value (named, never printed)."""
    files = {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    leaked = sorted(name for name, data in files.items() if any(n in data for n in NEEDLES))
    assert not leaked, f"request values reached $OUT in {leaked}"
    return sorted(files)


@pytest.mark.parametrize("var", ["PP_LLAMA", "PP_SERVE_RUN", "PP_SERVE_ROWS"])
def test_an_art_row_dir_or_serve_run_is_refused_before_anything_is_written(tmp_path, var):
    unset = [a for v in ("PP_LLAMA", "PP_SERVE_RUN", "PP_SERVE_ROWS", "OUT") for a in ("-u", v)]
    got = subprocess.run(["env", *unset, f"RESEARCH_RUN_DIR={tmp_path}", f"{var}=art:{'0' * 64}", "bash", str(RUN_SH)],
                         cwd=tmp_path, capture_output=True, text=True)
    assert got.returncode == 2
    assert f"give {var} the serve directory on this node" in got.stdout
    assert list(tmp_path.iterdir()) == []


def test_a_serve_dirs_request_valued_records_never_reach_out(tmp_path):
    serve, out = tmp_path / "serve", tmp_path / "out"
    public = {"serve.json": {"model": "m", "run_id": "r"}, "windows.jsonl": [{"window": 0, "run_id": "r/w0", "run_root": "ab"}],
              "advice/w0/registration.json": {"window": 0, "rows": 4}}
    put(serve, {**public,
                "requests.jsonl": [{"engine_id": "e0", "user": "u0", "prompt_ids": PROMPT, "streamed_ids": [7], "windows": [0]}],
                "request_users.jsonl": [{"engine_id": "e0", "user": "u0"}],
                "advice_trace.jsonl": [{"kind": "arrive", "request": "e0", "prompt_ids": PROMPT}],
                "step_trace_r0.jsonl": [{"kind": "step", "rows": [["e0", 0, 3]], "prompt_ids": PROMPT}],
                "load.jsonl": [{"user": "u0", "prompt_ids": PROMPT}],
                "advice/w0/private/opening.json": {"rows": [PROMPT]}})
    served = block("  served() {", "  }")
    code = f'O={out}; PP=true; TG=(); run() {{ :; }}\n{served}served served {serve} 0\n'
    got = subprocess.run(["bash", "-c", code], capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    assert published(out) == sorted(f"served-records/{f}" for f in (
        "serve.json", "windows.jsonl", "registration.json", "compiler.json"))


def test_the_standins_workload_prompts_never_reach_out(tmp_path):
    s, out = tmp_path / "served", tmp_path / "out"
    put(s, {"workload.json": {"requests": [{"request_id": "q0", "prompt_token_ids": PROMPT}]},
            "commit/runs.jsonl": [{"rank": 0, "root": "ab"}], "commit/versions.json": {"root_tag": "t"},
            "commit/rank0/rows.json": {"rows": [PROMPT]}})
    code = f"O={out}; S={s}\n" + block("  mkdir -p $O/standin", "  mkdir -p $O/standin")
    got = subprocess.run(["bash", "-c", code], capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    assert published(out) == ["standin/runs.jsonl", "standin/versions.json"]
