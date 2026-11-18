"""run.sh publishes no request value into $OUT, which custody uploads: it refuses an art: row dir or serve run (a published
one's sealed bundle or requests.jsonl holds them) before it writes anything, and a node's serve dir, with a prompt planted
in its requests.jsonl, its scheduler and load traces and its advice's opening, reaches $OUT only as its registration's
public form and its compiler, while records() still finds every file it opens."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

RUN_SH = Path(__file__).resolve().parents[3] / "benchmarks" / "prove_path" / "run.sh"
PUBF = RUN_SH.parents[1] / "one_stage" / "public_files.py"
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


#: what a serve dir's records hold that is not public, each planted where its writer puts it: a user, request ids, the
#: seed coin, timings, the advice's own length and the request count, and seven whole dumped dictionaries
SECRETS = {"user": "planted-user-5d1", "request": "planted-req-8c4", "seed_coin": 271828182845, "time": 1731903119.25,
           "bits": 39113, "requests": 5813, "dict": "planted-dict-key"}
CAP = 8388608


def serve_dir(serve: Path) -> None:
    s = SECRETS
    dicts = {f"dumped_{i}": {s["dict"]: i, "user": s["user"]} for i in range(7)}
    put(serve, {
        "serve.json": {"model": "m", "seed_coin": s["seed_coin"], "started": s["time"], "users": [s["user"]],
                       "requests": [s["request"]], **dicts},
        "windows.jsonl": [{"window": 0, "run_id": "r/w0", "run_root": "ab" * 64, "requests": [s["request"]],
                           "users": [s["user"]], "t0": s["time"]}],
        "advice/w0/registration.json": {
            "schema": "verity-vllm/advice-registration/v1", "row_bits": 256, "rows": 153, "T": 600, "window": 0,
            "binding": {"compiler": {"name": "c", "digest": "d"}, "advice": {"root": "ab" * 64, "bits": s["bits"]}},
            "commit": {"run_id": "r/w0", "run_root": "ab" * 64}, "compiler": {"name": "c", "digest": "d", "cap_bits": CAP},
            "registered_roots": {"weights/w@16": "cd" * 64}, "requests": s["requests"], "compiled": [s["request"]],
            "refused": {s["request"]: s["user"]}},
        "requests.jsonl": [{"engine_id": s["request"], "user": s["user"], "prompt_ids": PROMPT, "streamed_ids": [7],
                            "windows": [0]}],
        "request_users.jsonl": [{"engine_id": s["request"], "user": s["user"]}],
        "advice_trace.jsonl": [{"kind": "arrive", "request": s["request"], "prompt_ids": PROMPT, "t": s["time"]}],
        "step_trace_r0.jsonl": [{"kind": "step", "rows": [[s["request"], 0, 3]], "prompt_ids": PROMPT}],
        "load.jsonl": [{"user": s["user"], "prompt_ids": PROMPT}],
        "advice/w0/private/opening.json": {"rows": [PROMPT]}})


def no_secret(out: Path) -> list[str]:
    """`published`, after holding that no byte under ``out`` carries a planted secret (named, never printed)."""
    needles = [str(v).encode() for v in SECRETS.values()]
    hit = sorted(f"{p.relative_to(out)}: {k}" for p in out.rglob("*") if p.is_file()
                 for k, n in zip(SECRETS, needles) if n in p.read_bytes())
    assert not hit, f"a serve dir's private values reached $OUT: {hit}"
    return published(out)


#: the compiler record `prove_path.py compiler` prints, which records() reads back from $OUT
STUB_PP = "#!/bin/sh\necho '{\"name\": \"c\", \"digest\": \"d\"}'\n"


def served_code(out: Path, serve: Path, run: str) -> str:
    return (f'O={out}; PUBF={PUBF}; PP={out.parent / "pp"}; TG=(); CH=()\nrun() {{ {run} }}\n'
            f'{block("records() {", "}")}{block("  served() {", "  }")}served served {serve} 0\n')


def test_a_serve_dirs_private_values_never_reach_out(tmp_path):
    serve, out = tmp_path / "serve", tmp_path / "out"
    serve_dir(serve)
    (tmp_path / "pp").write_text(STUB_PP)
    (tmp_path / "pp").chmod(0o755)
    got = subprocess.run(["bash", "-c", served_code(out, serve, ":;")], capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    assert no_secret(out) == ["served-records/compiler.json", "served-records/registration.json"]
    reg = json.loads((out / "served-records/registration.json").read_text())
    assert reg["binding"]["advice"]["bits"] == CAP and "requests" not in reg and "refused" not in reg


def test_records_finds_every_file_it_opens_after_served(tmp_path):
    """A stub run of served() and records(): each path the records audit is handed under $OUT or the serve dir exists."""
    serve, out, log = tmp_path / "serve", tmp_path / "out", tmp_path / "py.log"
    serve_dir(serve)
    (tmp_path / "pp").write_text(STUB_PP)
    (tmp_path / "pp").chmod(0o755)
    (tmp_path / "part.json").write_text(json.dumps({"law": "work-rate:1/1"}))
    (tmp_path / "py").write_text(
        f'#!/bin/bash\n[ "$1" = -c ] && exec {sys.executable} "$@"\necho "$*" >> {log}\n'
        f'for a in "$@"; do case "$a" in {out}/*|{serve}/*) [ -e "$a" ] || {{ echo "missing $a" >> {log}; exit 3; }};; esac; done\n')
    (tmp_path / "py").chmod(0o755)
    run = f'mkdir -p $O/$1; REC_NAME=$1; records $1 {tmp_path / "part.json"} || return 1;'
    code = f"PY={tmp_path / 'py'}; REPO={RUN_SH.parents[2]}; W={tmp_path / 'w'}\n" + served_code(out, serve, run)
    got = subprocess.run(["bash", "-c", code], capture_output=True, text=True)
    assert got.returncode == 0, (got.stderr, log.read_text() if log.exists() else "")
    calls = log.read_text()
    assert "missing" not in calls and "records_audit.py audit" in calls and "--run " + "ab" * 64 in calls
    no_secret(out)


def test_the_standins_workload_prompts_never_reach_out(tmp_path):
    s, out = tmp_path / "served", tmp_path / "out"
    put(s, {"workload.json": {"requests": [{"request_id": "q0", "prompt_token_ids": PROMPT}]},
            "commit/runs.jsonl": [{"rank": 0, "root": "ab"}], "commit/versions.json": {"root_tag": "t"},
            "commit/rank0/rows.json": {"rows": [PROMPT]}})
    code = f"O={out}; S={s}; PUBF={PUBF}\n" + block("  mkdir -p $O/standin", "  mkdir -p $O/standin")
    got = subprocess.run(["bash", "-c", code], capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    assert published(out) == ["standin/runs.jsonl", "standin/versions.json"]
