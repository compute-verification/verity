"""The registered starts on every node, from one preserved artifact in the evidence store (`starts_store.json`).

    python starts_store.py export --from DIR --out OUT    # DIR's starts into OUT/starts/<name>/, their manifest OUT/starts_store.json
    python3 starts_store.py install [--to DIR] [--keep]   # on a node: fetch the artifact, check each file, install under DIR

`STARTS` (`$CW_STARTS`, default /workspace/te/starts) is the one directory `prepare` reads starts from; nothing reads a
run's directory. `install` fetches `starts_store.json`'s artifact (`research data fetch`), checks every file's sha256 and
each start.json's digest against the manifest, and moves each start into place whole, through a directory beside it. It
leaves a start whose files already match and refuses one that differs (remove it with `research retention rm`). `--keep`
records the directory's retention (`research keep`, owner @training, the artifact as its copy). `export` checks each θ's
digest (`prepare.digest`, so it needs torch); `install` is stdlib only, with research from this tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STARTS = Path(os.environ.get("CW_STARTS", "/workspace/te/starts"))
MANIFEST = HERE / "starts_store.json"
NAMES = ("s50", "s100", "m50", "m100")
FILES = ("start.json", "theta.safetensors")
RESEARCH_SRC = HERE.parents[1] / "tools" / "research" / "src"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def entry(d: Path) -> dict:
    return {"digest": json.loads((d / "start.json").read_text())["digest"],
            "files": {f: sha256(d / f) for f in FILES}, "bytes": sum((d / f).stat().st_size for f in FILES)}


def export(src: Path, out: Path, names=NAMES, check: bool = True) -> dict:
    starts = {}
    for name in names:
        d = out / "starts" / name
        d.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            shutil.copyfile(src / name / f, d / f)
        if check:
            import prepare
            from safetensors.torch import load_file

            got = prepare.digest(load_file(str(d / "theta.safetensors")))
            want = json.loads((d / "start.json").read_text())["digest"]
            if got != want:
                raise SystemExit(f"starts_store: {src / name}'s θ has digest {got}, its start.json says {want}")
        starts[name] = entry(d)
    man = {"schema": "covert-witness/starts-store/v1", "art": None, "path": "starts", "starts": starts}
    (out / "starts_store.json").write_text(json.dumps(man, indent=1) + "\n")
    return man


def _research(*args: str) -> str:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(RESEARCH_SRC), *filter(None, [os.environ.get("PYTHONPATH")])]))
    res = subprocess.run([sys.executable, "-m", "research", *args], env=env, capture_output=True, text=True)
    if res.returncode:
        raise SystemExit(f"starts_store: research {' '.join(args)} exited {res.returncode}:\n{res.stdout}{res.stderr}")
    return res.stdout


def matches(d: Path, want: dict) -> bool:
    return all((d / f).is_file() and sha256(d / f) == h for f, h in want["files"].items())


def install(to: Path, man: dict, fetch=None) -> list[str]:
    """Install every start the manifest lists under `to`; the names installed now."""
    if not (man.get("art") or "").startswith("art:"):
        raise SystemExit("starts_store: the manifest names no artifact (export, then `research data put --tree`)")
    to.mkdir(parents=True, exist_ok=True)
    todo = []
    for name, want in man["starts"].items():
        if matches(to / name, want):
            continue
        if (to / name).exists():
            raise SystemExit(f"starts_store: {to / name} differs from the manifest; remove it with `research retention rm`")
        todo.append(name)
    if not todo:
        return []
    tmp = Path(tempfile.mkdtemp(prefix=".fetch-", dir=to))
    try:
        got = Path(fetch(man["art"], tmp) if fetch else _research("data", "fetch", man["art"], "--to", str(tmp))
                   .strip().splitlines()[-1])
        root = got / man.get("path", "starts")
        for name in todo:
            want, src = man["starts"][name], root / name
            for f, h in want["files"].items():
                if not (src / f).is_file() or sha256(src / f) != h:
                    raise SystemExit(f"starts_store: the artifact's {name}/{f} is not the manifest's (sha256 {h})")
            if json.loads((src / "start.json").read_text())["digest"] != want["digest"]:
                raise SystemExit(f"starts_store: the artifact's {name}/start.json names another digest")
            stage = tmp / f"stage-{name}"
            stage.mkdir()
            for f in want["files"]:
                shutil.move(str(src / f), stage / f)
            os.rename(stage, to / name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return todo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--from", dest="src", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--names", nargs="+", default=list(NAMES))
    i = sub.add_parser("install")
    i.add_argument("--to", type=Path, default=STARTS)
    i.add_argument("--manifest", type=Path, default=MANIFEST)
    i.add_argument("--keep", action="store_true", help="record the directory's retention (research keep)")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        print(json.dumps(export(a.src, a.out, a.names)))
        return 0
    man = json.loads(a.manifest.read_text())
    done = install(a.to, man)
    if a.keep:
        _research("keep", str(a.to), "--owner", "@training", "--expires", "never", "--priority", "keep", "--art", man["art"],
                  "--regenerate", "python3 benchmarks/covert_witness/starts_store.py install --keep",
                  "--why", "covert_witness's registered starts, the one directory prepare reads them from")
    print(json.dumps({"installed": done, "to": str(a.to), "art": man["art"],
                      "starts": {n: s["digest"] for n, s in man["starts"].items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
