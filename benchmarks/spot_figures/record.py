"""The configuration the SPOT post claims (`record.json`), and a serve's stamp against it.

`record.json` is one row a switch: its number, its name, the run that sets it (`serve`, `proving` or `planner`), the
value the post claims (`post`) and the reader that finds a serve's value of it (`read`, in `READERS`; null where
nothing a serve records gives it). A number in the post holds at the post's configuration only as far as the runs it
came from had each switch at the post's value, so each such run carries a stamp:

    {"digest": SHA-512 of record.json, "observed": {"N": the serve's value of switch N, ...}, "off": [N, ...]}

    uv run python benchmarks/spot_figures/record.py stamp META.json [--out DIR]   # prints the off list; DIR/record.json

`stamp` reads a serve's `meta.json` (the serve tax's, or any harness's that records `server_argv` and `versions`) and
nothing else, so the serve does not change and a stored serve is stamped afterwards. `off` lists every switch a serve
sets that its `meta.json` does not show at the post's value: one it shows at another value, one it shows only in part
(the reader's value lacks a field `post` has), and one with no reader. A stamp is as good as the `meta.json` it read:
what a run changed without recording it there, a stamp does not see. It holds configuration only, no request, count or
time, so it can be published where the serve's own record is private.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE / "record.json"
RUNS = ("serve", "proving", "planner")
#: vLLM's OpenAI server, which every serve launches: a `meta.json` whose command line runs anything else is no serve's
SERVER = "vllm.entrypoints.openai.api_server"
#: the serve tax's names for sampled decoding (`serve_tax.Arm.sampling`)
SAMPLED = ("seeded", "unseeded")
#: the serve tax's plugin that takes the two-stage boundary commitment
TWO_STAGE = "boundary_rows"


def option(meta: dict, name: str) -> str | None:
    """The value the server's command line gives option ``name``, the last one as the server reads it; None where it
    does not give it."""
    argv, value = meta["server_argv"], None
    for i, word in enumerate(argv):
        if word == name and i + 1 < len(argv):
            value = argv[i + 1]
        elif word.startswith(name + "="):
            value = word.split("=", 1)[1]
    return value


def checkpoint(meta: dict) -> str | None:
    """The checkpoint's revision: of `model` (REPO@REVISION), else the last part of the `--model` path where it is one."""
    model = meta.get("model")
    if isinstance(model, str) and "@" in model:
        return model.rsplit("@", 1)[1]
    last = (option(meta, "--model") or "").rstrip("/").rsplit("/", 1)[-1]
    return last if re.fullmatch(r"[0-9a-f]{40}", last) else None


def precision(meta: dict) -> str:
    return option(meta, "--quantization") or option(meta, "--dtype") or "checkpoint"


def whole(meta: dict, name: str) -> int | None:
    value = option(meta, name)
    return int(value) if value is not None and value.isdigit() else None


def engine(meta: dict) -> dict | None:
    """The commit vLLM's version names. Nothing here says the server is otherwise unmodified."""
    found = re.search(r"\+g([0-9a-f]{7,40})", str((meta.get("versions") or {}).get("vllm", "")))
    return {"commit": found.group(1)} if found else None


def prefix_caching(meta: dict) -> str:
    """vLLM's default, on, unless the last of the option's two forms turns it off."""
    forms = [word for word in meta["server_argv"] if word in ("--enable-prefix-caching", "--no-enable-prefix-caching")]
    return "off" if forms and forms[-1].startswith("--no-") else "on"


def decoding(meta: dict) -> str | None:
    sampling = meta.get("sampling")
    return "sampled" if sampling in SAMPLED else sampling if isinstance(sampling, str) else None


def execution_phase(meta: dict) -> float | None:
    seconds = meta.get("window_s")
    return seconds if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) else None


def sampling_design(meta: dict) -> str | None:
    return "two-stage" if TWO_STAGE in (meta.get("plugin") or ()) else None


def certifier(meta: dict) -> dict | None:
    """Whether the serve ran under the certified tick. Nothing here says which certifier stood in front of it."""
    return {"tick": meta["tick"] is not None} if "tick" in meta else None


READERS: dict[str, Callable[[dict], object]] = {
    "checkpoint": checkpoint,
    "precision": precision,
    "context": lambda meta: whole(meta, "--max-model-len"),
    "tensor_parallel": lambda meta: whole(meta, "--tensor-parallel-size"),
    "engine": engine,
    "prefix_caching": prefix_caching,
    "decoding": decoding,
    "execution_phase": execution_phase,
    "sampling_design": sampling_design,
    "certifier": certifier,
}


def switches(record: bytes | None = None) -> list[dict]:
    """The record's rows, checked: numbered from 1 in order, each with a name, a run, the post's value and a known reader."""
    rows = json.loads(RECORD.read_bytes() if record is None else record)["switches"]
    for i, row in enumerate(rows, 1):
        if row.get("n") != i or not row.get("name") or row.get("run") not in RUNS or row.get("post") is None:
            raise ValueError(f"record.json: row {i} needs n = {i}, a name, a run of {RUNS} and the post's value: {row}")
        if row.get("read") is not None and (row["read"] not in READERS or row["run"] != "serve"):
            raise ValueError(f"record.json: switch {i} reads {row['read']!r}, but a reader is one of {sorted(READERS)} "
                             "and reads a serve")
    return rows


def stamp(meta: dict) -> dict:
    """A serve's stamp, from its `meta.json`: the record's digest, the serve's value of each switch it shows, and the
    switches a serve sets that it does not show at the post's value."""
    argv = meta.get("server_argv") if isinstance(meta, dict) else None
    if not isinstance(argv, list) or SERVER not in argv:
        raise ValueError(f"not a serve's meta.json: its `server_argv` does not run {SERVER}")
    record = RECORD.read_bytes()
    observed, off = {}, []
    for row in switches(record):
        if row["run"] != "serve":
            continue
        value = READERS[row["read"]](meta) if row["read"] else None
        if value is not None:
            observed[str(row["n"])] = value
        if value != row["post"]:
            off.append(row["n"])
    return {"digest": hashlib.sha512(record).hexdigest(), "observed": observed, "off": off}


def listed(numbers: list[int]) -> str:
    return ", ".join(map(str, numbers)) or "none"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("stamp", help="a serve's stamp, from its meta.json")
    s.add_argument("meta", type=Path, help="the serve's meta.json")
    s.add_argument("--out", type=Path, help="the directory to write record.json, the stamp, in")
    a = ap.parse_args(argv)
    if a.out and (a.out / RECORD.name).resolve() == RECORD:
        sys.exit(f"--out {a.out}: the stamp would replace {RECORD.name}, the configuration; give the run's directory")
    try:
        made = stamp(json.loads(a.meta.read_text()))
    except ValueError as e:
        sys.exit(f"{a.meta}: {e}")
    serve = [row["n"] for row in switches() if row["run"] == "serve"]
    print(f"record {made['digest'][:16]}: a serve sets {len(serve)} of its {len(switches())} switches")
    print(f"at the post's value: {listed([n for n in serve if n not in made['off']])}")
    print(f"off: {listed(made['off'])}")
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / RECORD.name).write_text(json.dumps(made, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {a.out / RECORD.name}")


if __name__ == "__main__":
    main()
