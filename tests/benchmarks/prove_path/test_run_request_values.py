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
PL = RUN_SH.parent / "public_log.py"
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
    return (f'O={out}; K={out.parent / "keep"}; PUBF={PUBF}; PL={PL}; PP={out.parent / "pp"}; TG=(); CH=()\nrun() {{ {run} }}\n'
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
        'case "$*" in *records_audit.py*) echo \'RECORDS {"ok": true, "leak": {"advice_bits": 4242424}}\';; esac\n'
        f'for a in "$@"; do case "$a" in {out}/*|{serve}/*|{tmp_path / "keep"}/*) [ -e "$a" ] || {{ echo "missing $a" >> {log}; exit 3; }};; esac; done\n')
    (tmp_path / "py").chmod(0o755)
    run = f'mkdir -p $O/$1 $K/$1; REC_NAME=$1; records $1 {tmp_path / "part.json"} || return 1;'
    code = f"PY={tmp_path / 'py'}; REPO={RUN_SH.parents[2]}; W={tmp_path / 'w'}\n" + served_code(out, serve, run)
    got = subprocess.run(["bash", "-c", code], capture_output=True, text=True)
    assert got.returncode == 0, (got.stderr, log.read_text() if log.exists() else "")
    calls = log.read_text()
    assert "missing" not in calls and "records_audit.py audit" in calls and "--run " + "ab" * 64 in calls
    no_secret(out)
    assert (tmp_path / "keep/served/compilers.json").is_file() and not list(out.rglob("compilers.json"))
    assert b"4242424" not in (out / "served/audit.log").read_bytes() and "4242424" not in got.stdout
    assert b"4242424" in (tmp_path / "keep/served/audit.log").read_bytes()


def test_the_standins_workload_prompts_never_reach_out(tmp_path):
    s, out = tmp_path / "served", tmp_path / "out"
    put(s, {"workload.json": {"requests": [{"request_id": "q0", "prompt_token_ids": PROMPT}]},
            "commit/runs.jsonl": [{"rank": 0, "root": "ab"}], "commit/versions.json": {"root_tag": "t"},
            "commit/rank0/rows.json": {"rows": [PROMPT]}})
    code = f"O={out}; S={s}; PUBF={PUBF}\n" + block("  mkdir -p $O/standin", "  mkdir -p $O/standin")
    got = subprocess.run(["bash", "-c", code], capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    assert published(out) == ["standin/runs.jsonl", "standin/versions.json"]


#: the files a run keeps in PP_KEEP and never writes under $O: a window's partition and the files beside it, the advice's
#: length, the tools' error output and logs no reader takes from $OUT, the compilers' argv (local paths), the machine's shape
KEPT = ("partition.json", "index.json", "window.json", "m*-public.bin", "audit.err", "compilers.json", "advice.log",
        "sync.log", "build-lean.log", "build-sampled-proofs.log", "m0-build", "sets-$SET.log", "fetch-main.log", "keep.log",
        "tie-fixture.err", "standin.err")


def test_no_kept_file_is_written_under_out():
    text = RUN_SH.read_text()
    lines = [line for line in text.splitlines() if "$O/" in line and not line.lstrip().startswith("#")]
    bare = [k.replace("*", "") for k in KEPT]
    hit = sorted({k for line in lines for k in bare if f"$O/{k}" in line or ("$O/" in line and f"/{k}" in line
                                                                              and f"$K/" not in line)})
    assert not hit, f"run.sh writes {hit} under $OUT; they go to $K (PP_KEEP)"
    assert 'K=${PP_KEEP:-$W/keep}' in text and "chmod 700" in text


#: `prove_path.py`'s commit and audit as run() calls them: a window whose partition and siblings hold a planted value, and
#: an audit whose error output does; registration.json, commit.log and the PROVE-PATH line are what $OUT keeps
STUB_RUN = r"""#!/bin/bash
cmd=$1; shift; out=
while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done
case $cmd in
  commit) mkdir -p $out/rank0 && for f in window partition index; do echo "{\"planted\": \"planted-$f\"}" > $out/rank0/$f.json; done
          echo '{"window": "public"}' > $out/rank0/registration.json; echo planted-bin > $out/rank0/m0-public.bin
          echo '{}' > $out/commit.json; echo '[prove-path commit] rank 0: {"bits": 4242423}';;
  audit) mkdir -p $out && echo '{"ok": true}' > $out/summary.json
         echo 'PROVE-PATH {"advice_bits": 4242421, "forged": false, "ok": true, "ranks": {"0": {"accepted": true, "charge": {"c": 4242422}}}}'
         echo 'a5 progress planted-progress'; echo planted-auditerr >&2;;
  compiler) echo '{"name": "c", "digest": "d"}';;
esac
"""
PLANTED_RUN = (b"planted-partition", b"planted-window", b"planted-index", b"planted-bin", b"planted-auditerr",
               b"4242421", b"4242422", b"4242423", b"planted-progress")


def _run_arm(tmp_path: Path, arm: str) -> tuple[Path, Path]:
    out, keep, w = tmp_path / "out", tmp_path / "keep", tmp_path / "w"
    (tmp_path / "pp").write_text(STUB_RUN)
    (tmp_path / "pp").chmod(0o755)
    for d in (out, keep, w):
        d.mkdir()
    code = (f"O={out}; K={keep}; W={w}; PUBF={PUBF}; PL={PL}; PP={tmp_path / 'pp'}; TG=(); CH=(); PLAIN=(); TIME=\n"
            f"{block('keep() {', '}')}{block('run() {', '}')}{arm}\n")
    got = subprocess.run(["bash", "-c", code], capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    assert not [n for n in PLANTED_RUN if n in got.stdout.encode()], "the run's stdout carries a planted value"
    return out, keep


@pytest.mark.parametrize("arm", ["run glm '--tp-commit T --build-root B' A part.json ''",
                                 "run llama '--bundle B --row-dir R --workload L' A part.json ''",
                                 "run standin '--tp-commit S --build-root B' A part.json '' --forge 0:1"])
def test_a_runs_partition_its_siblings_and_error_output_stay_in_the_keep(tmp_path, arm):
    """The TP, single-GPU and stand-in arms each go through run(): $OUT gets the window's registration, commit.log and the
    audit's log, whose PROVE-PATH line `verity bench prove layers` reads, and none of the planted values; the keep holds them."""
    out, keep = _run_arm(tmp_path, arm)
    name = arm.split()[1]
    files = {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert not [f for f, b in files.items() if any(n in b for n in PLANTED_RUN)], sorted(files)
    assert f"{name}/windows/rank0/registration.json" in files and b"PROVE-PATH" in files[f"{name}/audit.log"]
    assert f"{name}/commit.log" in files and f"{name}/summary.json" in files
    kept = {p.relative_to(keep).as_posix() for p in keep.rglob("*") if p.is_file()}
    assert {f"{name}/windows/rank0/{f}" for f in ("partition.json", "index.json", "window.json", "m0-public.bin")} <= kept
    assert f"{name}/audit.err" in kept and b"4242421" in (keep / name / "audit.log").read_bytes()
    line = json.loads(files[f"{name}/audit.log"].decode().split("PROVE-PATH ", 1)[1].splitlines()[0])
    assert line == {"forged": False, "ok": True, "ranks": {"0": {"accepted": True}}}


def test_pp_keep_under_out_or_the_run_dir_is_refused_before_anything_is_written(tmp_path):
    lines = RUN_SH.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("K=${PP_KEEP"))
    code = "\n".join(lines[i:i + 6]) + "\n"
    for keep in (tmp_path / "run" / "out" / "k", tmp_path / "run" / "k"):
        got = subprocess.run(["bash", "-c", f"O={tmp_path / 'run' / 'out'}; RESEARCH_RUN_DIR={tmp_path / 'run'}; "
                              f"PP_KEEP={keep}; W={tmp_path}\n{code}"], capture_output=True, text=True)
        assert got.returncode == 2 and "which custody publishes" in got.stdout and not keep.exists()
    keep = tmp_path / "elsewhere"
    got = subprocess.run(["bash", "-c", f"O={tmp_path / 'run' / 'out'}; RESEARCH_RUN_DIR={tmp_path / 'run'}; "
                          f"PP_KEEP={keep}; W={tmp_path}\n{code}"], capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    assert keep.stat().st_mode & 0o777 == 0o700


def test_public_log_lets_no_advice_length_or_charge_go():
    import importlib.util
    spec = importlib.util.spec_from_file_location("public_log", PL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.public_line('RECORDS {"ok": true, "draw_counts": {"t": 2}, "leak": {"advice_bits": 9, "bound": "vacuous"}}') \
        == 'RECORDS {"draw_counts": {"t": 2}, "ok": true}'
    assert m.public_line('PROVE-PATH {"advice_bits": 9, "proof": "zk", "ranks": {"0": {"charge": {"c": 9}, "ok": true}}}') \
        == 'PROVE-PATH {"proof": "zk", "ranks": {"0": {"ok": true}}}'
    assert m.public_line("ValueError: boom") == "ValueError: boom"
    assert m.public_line("PROVE-PATH not json 9") is None and m.public_line("[prove-path commit] rank 0: {}") is None
