"""What this suite's tests share: the paths, the references the Lean is compared with (Rust's `flock_live::release` through
`tests/rust`, Python's `verity_flock.firewall`, and `flock-firewall contract` for the send check's verdict), and the
strategies that draw records, ops and outer sessions.

Every record drawn is one the firewall wrote (Python's, from a drawn history of ops), or one torn by a crash (its last line
cut short), with blank lines in it or without its final newline. Strings are any Unicode but lone surrogates, integers are
below 2^40, and a schedule is any JSON but floats: outside that domain Python and Rust already differ from each other (a
budget that isn't an integer, statements that aren't strings, integers past u64, lines that are JSON but not the firewall's,
`\\r\\n`), and the Lean reads only the firewall's own lines (`test_the_lean_reads_only_the_firewalls_lines`).
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from hypothesis import strategies as st

import verity_difftest as D
from verity_flock import firewall as FW

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parents[3]
FLOCK = ROOT / "verity" / "ml" / "flock"
LEAN_FLOCK = ROOT / "verity" / "core" / "service" / "flock"
VECTORS = FLOCK / "live" / "src" / "firewall_record_vectors.json"
RELEASE_RS = FLOCK / "live" / "src" / "release.rs"
CRATE = HERE / "rust"
REFUSED = "FIREWALL-REFUSED: "


def _default(name: str) -> str:
    """The default `check_build.sh` gives `name` (`name=${VAR:-default}`), with `$HOME` expanded."""
    m = re.search(rf"^{name}=\$\{{[A-Z_]+:-([^}}]+)\}}$", (FLOCK / "check_build.sh").read_text(), re.M)
    assert m, f"check_build.sh sets no default for {name}"
    return m.group(1).replace("$HOME", str(Path.home()))


def rust_reference() -> Path:
    """`tests/rust` built at the toolchain `check_build.sh` pins, in a target directory outside the tree named by its
    sources' digest (so two checkouts on one machine never run each other's build): the `release-reference` binary. A
    missing toolchain is installed through rustup under `check_build.sh`'s lock, so the two never install it at once."""
    toolchain = os.environ.get("RUST_TOOLCHAIN") or _default("RUST_TOOLCHAIN")
    env = {**os.environ, "PATH": f"{Path.home() / '.cargo' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
           "RUSTUP_TOOLCHAIN": toolchain, "RUSTUP_AUTO_INSTALL": "0"}
    listed = subprocess.run(["rustup", "toolchain", "list"], env=env, capture_output=True, text=True, check=True).stdout
    if not any(line.startswith(f"{toolchain}-") for line in listed.splitlines()):
        lock = Path((os.environ.get("FLOCK_CHECKOUT") or _default("F")) + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        with open(lock, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            subprocess.run(["rustup", "toolchain", "install", toolchain, "--profile", "minimal"], env=env, check=True,
                           capture_output=True)
    sources = [CRATE / "Cargo.toml", CRATE / "Cargo.lock", CRATE / "src" / "main.rs", RELEASE_RS]
    digest = hashlib.sha256(b"".join(hashlib.sha256(p.read_bytes()).digest() for p in sources)).hexdigest()[:16]
    target = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "verity" / "core-firewall-rust" / digest
    p = subprocess.run(["cargo", "build", "--release", "--locked", "--quiet"], cwd=CRATE, env={**env, "CARGO_TARGET_DIR": str(target)},
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cargo build of {CRATE} failed ({p.returncode}):\n{p.stdout}{p.stderr}")
    return target / "release" / "release-reference"


def through(server: D.Server, fn: str):
    """A reference that asks `server` (a `Difftest.serve`-framed process: the Rust reference) for `fn`."""
    def reference(value):
        kind, text = server.call(fn, value)
        if kind == "reject":
            raise D.Reject(text)
        return json.loads(text)

    return reference


def _op(fw: FW.Firewall, op: list):
    """One op of the shared vectors on `fw`, its result as they write it (`test_firewall_release`'s `_vector_op`)."""
    name, *args = op
    try:
        if name == "admit":
            return {"release_ms": fw.admit(*args, timed=True)}
        if name == "slot":
            with fw.slot(*args):
                return None
        return getattr(fw, name)(*args)
    except FW.Refused as e:
        return {"refused": str(e).removeprefix(REFUSED)}
    except json.JSONDecodeError:
        raise
    except ValueError as e:
        if name == "open":
            return {"error": str(e)}
        raise


def python_op(value: dict) -> dict:
    """`{"op", "record"}` through `verity_flock.firewall` on a state directory holding the record: `{"record", "result"}`.
    A record it can't read is a rejection (json's error is a ValueError, which an open would report as its own)."""
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "record.jsonl"
        log.write_bytes(value["record"].encode())
        try:
            result = _op(FW.Firewall(Path(d)), value["op"])
        except json.JSONDecodeError as e:
            raise D.Reject(f"the record: {e}") from None
        return {"record": log.read_bytes().decode(), "result": result}


def record_of(ops: list[list]) -> str:
    """The record Python's firewall writes running `ops` from an empty state."""
    with tempfile.TemporaryDirectory() as d:
        fw = FW.Firewall(Path(d))
        for op in ops:
            _op(fw, op)
        log = Path(d) / "record.jsonl"
        return log.read_bytes().decode() if log.exists() else ""


@functools.cache
def firewall_agree():
    """`verity/ml/flock/verifier/firewall_agree.py`, the contract check's sessions."""
    spec = importlib.util.spec_from_file_location("firewall_agree", FLOCK / "verifier" / "firewall_agree.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TEXT = st.text(st.characters(exclude_categories=("Cs",)), max_size=4)
NAMES = ("L0", "L1", "alg")
SESSION = st.one_of(st.sampled_from(["r0000", "r0001"]), st.sampled_from(["r0000", "r0001", "r0002", "r9999"]), TEXT)
PART = st.one_of(st.sampled_from(NAMES), st.sampled_from(NAMES), st.sampled_from(["inner", ""]), TEXT)
STREAM = st.one_of(st.sampled_from(["s", "t"]), st.sampled_from(["s", "t"]), TEXT)
VALID_PARTS = st.lists(st.sampled_from(NAMES), min_size=1, max_size=3, unique=True)
PARTS = st.one_of(VALID_PARTS, VALID_PARTS, VALID_PARTS, st.lists(PART, max_size=4))
BUDGET = st.one_of(st.none(), st.none(), st.integers(0, 3))
NAT = st.integers(0, 2**40)
JSON = st.recursive(st.none() | st.booleans() | st.integers(-3, 2**40) | TEXT,
                    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(TEXT, inner, max_size=3), max_leaves=6)


def valid_schedule(n: int):
    return st.builds(lambda s, w: {"start_ms": s, "windows_ms": w}, st.integers(0, 60),
                     st.lists(st.integers(1, 30), min_size=n, max_size=n))


def schedule(n: int):
    """A release schedule for `n` statements: none, one, or one off by a width, a key or its form."""
    near = st.builds(lambda s, w: {"start_ms": s, "windows_ms": w}, st.integers(-2, 60) | NAT,
                     st.lists(st.integers(-1, 30) | NAT | st.booleans(), max_size=4))
    extra = st.builds(lambda r, k, v: r | {k: v}, valid_schedule(n), TEXT, JSON)
    return st.one_of(st.none(), st.none(), valid_schedule(n), valid_schedule(n), near, extra, JSON)


@st.composite
def open_op(draw):
    parts = draw(PARTS)
    return ["open", draw(STREAM), parts, draw(BUDGET), draw(schedule(len(parts)))]


OP = st.one_of(
    open_op(),
    st.builds(lambda s: ["finish", s], SESSION),
    st.builds(lambda s, p, w: ["stop", s, p, w], SESSION, PART, TEXT),
    st.builds(lambda s, p: ["release", s, p], SESSION, PART),
    st.builds(lambda s, p: ["admit", s, p], SESSION, PART),
    st.builds(lambda s: ["failures", s], STREAM),
    st.builds(lambda s, p: ["slot", s, p], SESSION, PART),
)


@st.composite
def torn(draw, text: str) -> str:
    """`text` as it is (most often), cut short inside its last line, with a blank line after each, or without its final
    newline."""
    how = draw(st.sampled_from([0, 0, 0, 0, 0, 1, 2, 3])) if text else 0
    lines = text.splitlines(keepends=True)
    if how == 1:
        keep = draw(st.integers(1, len(lines[-1]) - 2))
        return "".join(lines[:-1]) + lines[-1][:keep]
    if how == 2:
        return text.replace("\n", "\n\n")
    if how == 3:
        return text[:-1]
    return text


@st.composite
def scene(draw, kinds=("served", "budget", "served", "budget", "none")) -> tuple[str, str, str, list[list]]:
    """A record, an outer session to ask of it, and the ops aimed at the two, `(record, session, part, aimed)`. Most often
    a recursive session opened and finished, its first k statements released and the next one asked (sometimes after a
    slot of it whose holder died); or a stream's sessions stopped up to its budget, or one short of it, and the stream
    opened again; or nothing; then a few ops of any kind."""
    ops, ask, again = [], None, None
    kind = draw(st.sampled_from(kinds))
    parts, stream = draw(VALID_PARTS), draw(STREAM)
    if kind == "served":
        again = ["open", stream, parts, draw(st.one_of(st.none(), st.integers(1, 3))),
                 draw(st.one_of(st.none(), valid_schedule(len(parts))))]
        k = draw(st.integers(0, len(parts)))
        ops += [again, ["finish", "r0000"]] + [["release", "r0000", p] for p in parts[:k]]
        ask = ("r0000", parts[k] if k < len(parts) else parts[0])
        if draw(st.sampled_from([False, False, False, True])):
            ops.append(["slot", *ask])
    elif kind == "budget":
        budget = draw(st.integers(1, 2))
        again = ["open", stream, parts, budget, None]
        n = draw(st.integers(budget - 1, budget))
        for i in range(n):
            ops += [again, ["finish", f"r{i:04d}"], ["stop", f"r{i:04d}", parts[0], "cut"]]
        ask = (f"r{max(n - 1, 0):04d}", parts[0])
    ops += draw(st.lists(OP, max_size=draw(st.sampled_from([0, 0, 0, 2, 5]))))
    s, p = ask if ask is not None and draw(st.sampled_from([True, True, True, False])) else (draw(SESSION), draw(PART))
    aimed = [["admit", s, p], ["slot", s, p], ["release", s, p], ["finish", s], ["stop", s, p, "cut"]]
    if kind == "budget":
        aimed = [again, ["admit", s, p], ["failures", stream]]
    elif again is not None:
        aimed = [["admit", s, p], *aimed, again, ["failures", stream]]
    return draw(torn(record_of(ops))), s, p, aimed


@st.composite
def op_input(draw) -> dict:
    """`{"op", "record"}`: one of the ops aimed at the scene, or any op."""
    text, _, _, aimed = draw(scene())
    return {"op": draw(st.one_of(st.sampled_from(aimed), OP)), "record": text}


PANIC = st.one_of(TEXT, st.builds(lambda c, n: c * n, st.characters(exclude_categories=("Cs",)), st.integers(195, 205)))
STEP = st.one_of(st.just("ok"), st.just("ok"), st.just("ok"), st.builds(lambda w: {"err": w}, TEXT),
                 st.builds(lambda m: {"panic": m}, PANIC))
SEND = st.one_of(st.just(True), st.just(True), st.just(False), st.builds(lambda m: {"panic": m}, PANIC))


@st.composite
def outer_input(draw) -> dict:
    """`{"now_ms", "part", "prove", "ready", "record", "send", "session"}`: one outer session of a scene."""
    text, s, p, _ = draw(scene(("served", "served", "served", "budget", "none")))
    return {"now_ms": draw(st.integers(0, 150)), "part": p, "prove": draw(STEP), "ready": draw(STEP), "record": text,
            "send": draw(SEND), "session": s}
