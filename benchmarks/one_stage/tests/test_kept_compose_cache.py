"""`served_verdicts.sh`'s kept compose cache (V3C_COMPOSE_CACHE, with V3C_WARM_JOBS): the script on toy windows, its
records_audit.py a fake whose `warm` and `audit` are a5's own (`a5.warm`, `a5.own_composed`, so `a5.compose_key` and
`compose_circuit.cached`) over toy templates, each a Definition and a composer that writes a few lines and notes each
compose it makes. A cache one window's warm made is taken whole by a later window's warm and audit of the same
templates once the first window's serve dir is gone, as links under the first's partition and program and as the
circuits composed bound to its own under another's; a window whose templates differ in their Definition, statement or
id takes none, composes its own and keeps them beside the first's, which stay as they were; unset or empty, the warm
and the audit get the window's own directory, the script's steps, limits and lines the same; without V3C_WARM_JOBS no
step reads it; and a relative path is refused before any step runs."""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity_flock import circuit as FC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_served_verdicts as SV  # noqa: E402

B1 = {"partition": "a1" * 64, "program": "b2" * 64}
B2 = {"partition": "c3" * 64, "program": "d4" * 64}
#: a window's templates, each by the width its toy Definition and statement are made at
SERVED = {"Toy{N=1}": 16, "Toy{N=2}": 32}
#: another window's: SERVED's first id at another Definition and statement, and its first Definition and statement at
#: another id
OTHER = {"Toy{N=1}": 64, "Toy{N=3}": 16}


def _toy_circuit(width: int, partition: str, program: str, rule: str) -> bytes:
    """A toy template's circuit bound to a partition and a program: its META holds the binding as `circuit.compose` writes
    it, once each."""
    meta = json.dumps({"k_log": width, "partition": {"digest": partition, "rule": rule}, "program_sha512": program},
                      sort_keys=True, separators=(",", ":"))
    return f"toy circuit {width}\nMETA {meta}\n".encode() + b"g\n" * width


#: records_audit.py's `warm` and `audit` over the toy window TOY_WINDOW names (its partition, program and templates),
#: through a5's own warm and take: each compose a composer makes is a line of $ORDER.composed
PRELUDE = '''import hashlib, importlib, json, os, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, os.environ["ONE_STAGE"])
import a5
from verity_flock import circuit as FC

circuits = a5.Program.__module__.rsplit(".", 1)[0]  # the circuits package, by the name a5 imports it under
BIT = importlib.import_module(circuits + ".boolean").BIT
definition = importlib.import_module(circuits + ".boolean.trace").definition
bind = importlib.import_module(circuits + ".defs").bind
Array = importlib.import_module(circuits + ".types").Array


''' + inspect.getsource(_toy_circuit) + '''

def toy(width):
    defn = bind(definition(f"KeptCacheToy{width}", 1, lambda C, a: [C.AND(a[0], a[1])], (("a", Array(width, BIT)),),
                           Array(1, BIT), register=False))

    def compose(out, partition, program):
        with open(os.environ["ORDER"] + ".composed", "a") as f:
            f.write(f"{width}\\n")
        body = _toy_circuit(width, partition, program, FC.PARTITION_V1)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(body)
        return {"pin": hashlib.sha512(body).hexdigest(), "class_sha512": hashlib.sha512(b"toy %d" % width).hexdigest(),
                "k_log": width}
    compose.inputs = ["unit/window", {"width": width}]
    return compose, defn


W = json.loads(os.environ["TOY_WINDOW"])
SETS = {t: toy(width) for t, width in W["templates"].items()}
MINE = SimpleNamespace(members=[SimpleNamespace(template=t) for t in W["templates"]], partition_digest=W["partition"],
                       partition={"program": W["program"]})
CACHE = SimpleNamespace(compose_cache=sys.argv[sys.argv.index("--compose-cache") + 1] if "--compose-cache" in sys.argv
                        else None)
'''
WARM = '''    got = a5.warm(CACHE, SETS, MINE, int(a[a.index("--jobs") + 1]))
    print("WARM " + json.dumps({"templates": len(got), "kept": sum(g["kept"] is True for g in got)}, sort_keys=True))
'''
AUDIT = '''    circuits = {}
    for i, t in enumerate(W["templates"]):
        made, kept = a5.own_composed(CACHE, *SETS[t], t, out / "window" / f"m{i}" / "circuit.txt", W["partition"],
                                     W["program"])
        circuits[t] = {"pin": made["pin"], "cached": kept, "at": str(out / "window" / f"m{i}" / "circuit.txt")}
    (out / "summary.json").write_text(json.dumps({"circuits": circuits}))
'''
#: what `test_served_verdicts.FAKE_RECORDS_AUDIT` prints for a warm and writes for an audit, which a5's own replace here
STUBS = ('    print(json.dumps({"template": "a@0", "seconds": 1.5, "kept": False, "pin": "ab"}))\n'
         '    print("WARM " + json.dumps({"templates": 3, "kept": 0, "seconds": 2.0}, sort_keys=True))\n',
         '    (out / "summary.json").write_text("{}")\n')
assert all(SV.FAKE_RECORDS_AUDIT.count(s) == 1 for s in STUBS), "FAKE_RECORDS_AUDIT's warm and audit are no longer these lines"
KEPT_AUDIT = PRELUDE + SV.FAKE_RECORDS_AUDIT.replace(STUBS[0], WARM).replace(STUBS[1], AUDIT)


def _run(root: Path, templates: dict, binding: dict, kept: Path | None) -> SimpleNamespace:
    """The script's V3c step on a toy window of ``templates`` bound to ``binding``, its serve dir and tree under ``root``,
    warmed two at a time, with V3C_COMPOSE_CACHE ``kept`` (None: unset): the widths its composers composed at, what its
    V3C-WARM line counts, and each template's circuit as its audit took it (its pin, whether the cache held it, its
    bytes and file)."""
    window = json.dumps({**binding, "templates": templates})
    rc, lines, order, _limits = SV._v3c(root, records_audit=KEPT_AUDIT, V3C_WARM_JOBS="2", TOY_WINDOW=window,
                                       ONE_STAGE=str(HERE.parent), V3C_COMPOSE_CACHE=str(kept) if kept else None)
    assert rc == 0 and order.index("warm") < order.index("audit"), (lines, order)
    vw = root / "serve" / f"v3c-w7-{SV.VTAG}"
    composed = root / "order.composed"
    circuits = json.loads((vw / "audit" / "summary.json").read_text())["circuits"]
    for c in circuits.values():
        c["bytes"] = Path(c["at"]).read_bytes()
    said = next(x for x in lines if x.startswith("V3C-WARM"))
    return SimpleNamespace(vw=vw, composed=sorted(map(int, composed.read_text().split())) if composed.exists() else [],
                           warm=json.loads(said.split(": WARM ")[1]), circuits=circuits)


def _entries(kept: Path) -> dict[str, tuple[bytes, bytes]]:
    """The cache's entries: each one's line and circuit, by its key."""
    return {p.parent.name: (p.read_bytes(), (p.parent / "circuit.txt").read_bytes()) for p in sorted(kept.glob("*/made.json"))}


def _fresh(templates: dict, binding: dict) -> dict[str, bytes]:
    return {t: _toy_circuit(w, binding["partition"], binding["program"], FC.PARTITION_V1) for t, w in templates.items()}


@pytest.mark.parametrize("binding", [B1, B2], ids=["the-first-windows-binding", "another-binding"])
def test_a_kept_cache_one_windows_warm_made_is_taken_whole_by_a_later_windows_audit_of_the_same_templates(tmp_path, binding):
    """With V3C_COMPOSE_CACHE, the first window's warm composes each template once into that directory, none into the
    window's own, and its audit takes them. With the first window's serve dir and tree gone, a second window of the same
    templates composes none: its warm finds both kept, and its audit takes both, each the circuit a compose bound to the
    second window would write (under the first's binding the kept file itself, linked)."""
    kept = tmp_path / "kept"
    one = _run(tmp_path / "one", SERVED, B1, kept)
    assert one.composed == [16, 32] and one.warm == {"templates": 2, "kept": 0}
    assert all(c["cached"] is True for c in one.circuits.values()) and not (one.vw / "compose-cache").exists()
    entries = _entries(kept)
    assert sorted(circuit for _line, circuit in entries.values()) == sorted(_fresh(SERVED, B1).values())
    shutil.rmtree(tmp_path / "one")
    assert _entries(kept) == entries
    two = _run(tmp_path / "two", SERVED, binding, kept)
    assert two.composed == [] and two.warm == {"templates": 2, "kept": 2}
    assert {t: c["cached"] for t, c in two.circuits.items()} == dict.fromkeys(SERVED, True)
    assert {t: c["bytes"] for t, c in two.circuits.items()} == _fresh(SERVED, binding)
    assert {t: c["pin"] for t, c in two.circuits.items()} == {t: hashlib.sha512(b).hexdigest()
                                                              for t, b in _fresh(SERVED, binding).items()}
    linked = [any(Path(c["at"]).samefile(kept / k / "circuit.txt") for k in entries) for c in two.circuits.values()]
    assert linked == [binding == B1] * 2
    assert _entries(kept) == entries and not (two.vw / "compose-cache").exists()


def test_a_window_whose_templates_differ_takes_none_of_a_kept_cache_and_composes_and_keeps_its_own(tmp_path):
    """A window whose templates differ from the kept ones (one id at another Definition and statement, one Definition and
    statement at another id) is offered the first window's cache: its warm finds none kept and composes both, its audit
    gets its own circuits, and the cache then holds both windows' entries, the first's as they were."""
    kept = tmp_path / "kept"
    _run(tmp_path / "one", SERVED, B1, kept)
    entries = _entries(kept)
    shutil.rmtree(tmp_path / "one")
    other = _run(tmp_path / "other", OTHER, B1, kept)
    assert other.composed == [16, 64] and other.warm == {"templates": 2, "kept": 0}
    assert {t: c["bytes"] for t, c in other.circuits.items()} == _fresh(OTHER, B1)
    after = _entries(kept)
    assert len(after) == 4 and {k: after[k] for k in entries} == entries
    assert sorted(c for k, (_line, c) in after.items() if k not in entries) == sorted(_fresh(OTHER, B1).values())


def test_without_v3c_compose_cache_the_warm_and_the_audit_get_the_windows_own_directory_as_before(tmp_path):
    """Unset or empty, the warm's and the audit's arguments name $VW/compose-cache, the warm's whole line as it was; set,
    the same arguments but for that one value at each, with the same steps in the same order, limits and verdict lines."""
    kept, got = tmp_path / "kept", {}
    for name, given in (("unset", None), ("empty", ""), ("set", str(kept))):
        root = tmp_path / name
        rc, lines, order, limits = SV._v3c(root, V3C_WARM_JOBS="3", V3C_COMPOSE_CACHE=given)
        assert rc == 0, lines
        here = lambda x, root=root: x.replace(str(root), "ROOT")  # noqa: E731
        argv = {s: [here(x) for x in json.loads((root / f"order.{s}").read_text())] for s in ("warm", "audit")}
        got[name] = (argv, order, limits, [here(x) for x in lines])
    argv, order, _limits, _lines = got["unset"]
    own = f"ROOT/serve/v3c-w7-{SV.VTAG}/compose-cache"
    assert argv["warm"] == ["warm", "--window", f"ROOT/serve/v3c-w7-{SV.VTAG}/window", "--law", "subset:1",
                            "--compose-cache", own, "--jobs", "3"]
    assert argv["audit"][argv["audit"].index("--compose-cache") + 1] == own and argv["audit"].count(own) == 1
    assert order.index("warm") < order.index("audit")
    assert got["empty"] == got["unset"]
    named = {s: [str(kept) if x == own else x for x in a] for s, a in argv.items()}
    assert got["set"] == (named, *got["unset"][1:]) and named != argv
    assert not kept.exists()


def test_v3c_compose_cache_without_v3c_warm_jobs_starts_no_warm_and_gives_the_audit_no_cache(tmp_path):
    """The setting names where the warm's composes go and the audit takes them from: with no warm (V3C_WARM_JOBS unset),
    the audit gets no cache, composes every template itself as before, and the directory is not made."""
    kept = tmp_path / "kept"
    rc, lines, order, _limits = SV._v3c(tmp_path / "run", V3C_COMPOSE_CACHE=str(kept))
    assert rc == 0 and "warm" not in order and "audit" in order, (lines, order)
    assert "--compose-cache" not in json.loads((tmp_path / "run" / "order.audit").read_text())
    assert not any(x.startswith("V3C-WARM") for x in lines) and not kept.exists()


@pytest.mark.parametrize("given", ["kept", "./kept", "../kept"])
@pytest.mark.parametrize("jobs", ["3", None])
def test_the_preflight_refuses_a_relative_v3c_compose_cache_and_no_absolute_one(tmp_path, given, jobs):
    """The warm and the audit run in the tree, where a relative path would put a kept cache: the preflight refuses one,
    with or without a warm, before any step runs and with nothing made at it; the same name from / is not refused."""
    root = tmp_path / "run"
    rc, lines, order, limits = SV._v3c(root, V3C_WARM_JOBS=jobs, V3C_COMPOSE_CACHE=given)
    assert (rc, lines, order, limits) == (3, [
        "PREFLIGHT: V3C_COMPOSE_CACHE is the kept compose cache's directory, an absolute path (the warm and the audit "
        f"run in the tree, where a relative one would put it), not {given}: name it from /"], [], [])
    assert not list(root.rglob("kept")) and not (tmp_path / "kept").exists()
    rc, lines, order, _limits = SV._v3c(tmp_path / "absolute", V3C_WARM_JOBS=jobs, V3C_COMPOSE_CACHE=f"/{given}")
    assert rc == 0 and "audit" in order and not any(x.startswith("PREFLIGHT") for x in lines), (lines, order)
