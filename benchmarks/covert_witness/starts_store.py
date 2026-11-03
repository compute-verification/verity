"""The registered starts on every node, from preserved artifacts in the evidence store (`starts_store.json`).

    python starts_store.py export --from DIR --out OUT   # DIR's starts to OUT/starts/<name>/, the manifest to OUT
    python3 starts_store.py pack --out starts.tar        # where the store's remote is: the artifacts as one checked tar
    python3 starts_store.py install [--to DIR] [--tar starts.tar] [--keep]   # on a node: check each file, install in DIR

`STARTS` (`$CW_STARTS`, default /workspace/research/te/starts, which the run user owns on every node) is the one directory
`prepare` reads starts from; nothing reads a run's directory. A start is in the manifest's `art` under its `path`, or in
its entry's own `art` and `path` (one another export published, as the v3 ladder's starts). `install` fetches each start's
artifact (`research data fetch`; a node holds no store key, so there it unpacks `pack`'s tar, sent with the run), checks
every file's sha256
and each start.json's digest against the manifest, and moves each start into place whole, through a directory beside
it. It leaves a start whose files already match and refuses one that differs (remove it with `research retention rm`).
`--keep` records the directory's retention (`research keep`, owned by the training lane, the artifact as its copy). `export` checks
each θ's digest (`prepare.digest`, so it needs torch) and records each start's shape (`shape`, what te_core.policy's
reference shapes read); `pack` and `install` are stdlib only, with research from this tree.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STARTS = Path(os.environ.get("CW_STARTS", "/workspace/research/te/starts"))
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


def shape(rec: dict) -> dict:
    """What te_core.policy sets a start's reference shapes from: its architecture and tokenizer (which data it is on:
    data v3's, else the pilot's, whose entries predate the field), its training tokens and C0 when its provenance
    counted it, and its forward MACs per token at full context (needs torch). torch's FLOP formulas have no entry for
    CPU's fused attention kernel, so on CPU attention runs as its math form, whose matmuls are counted."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    import prepare
    from te_core.flops import forward_macs_per_token

    cfg, prov = rec["config"], rec["provenance"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.device(dev):
        net = prepare.Net(cfg)
    with sdpa_kernel(SDPBackend.MATH) if dev == "cpu" else contextlib.nullcontext():
        per_token = forward_macs_per_token(net, cfg["ctx"], cfg["vocab"], dev)
    out = {k: cfg[k] for k in ("family", "d_model", "n_layer", "vocab", "ctx")}
    out.update(tokenizer=rec["tokenizer"], parameters=rec["parameters"], tokens=prov["tokens"],
               tokens_per_param=round(prov["tokens"] / rec["parameters"], 2), forward_macs_per_token=per_token)
    if (prov.get("c0") or {}).get("macs"):
        out["c0_macs"] = prov["c0"]["macs"]
    return out


def entry(d: Path, with_shape: bool = False) -> dict:
    rec = json.loads((d / "start.json").read_text())
    return {"digest": rec["digest"], "files": {f: sha256(d / f) for f in FILES},
            "bytes": sum((d / f).stat().st_size for f in FILES), **({"shape": shape(rec)} if with_shape else {})}


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
        starts[name] = entry(d, with_shape=check)
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


def store_fetch(art: str, tmp: Path) -> Path:
    """The artifact's tree, fetched from the evidence store (where this machine has its remote)."""
    return Path(_research("data", "fetch", art, "--to", str(tmp)).strip().splitlines()[-1])


def from_tar(tar: Path):
    """A fetch that unpacks `pack`'s tar instead, once, for every artifact (the tar holds every start): for a node with
    no store remote, which gets the tar with its run."""
    root: list[Path] = []

    def fetch(art: str, tmp: Path) -> Path:
        if not root:
            with tarfile.open(tar) as tf:
                tf.extractall(tmp / "tar", filter="data")
            root.append(tmp / "tar")
        return root[0]
    return fetch


def sources(man: dict, names) -> dict[tuple[str, str], list[str]]:
    """`names` by the artifact and path holding each: its entry's own `art` and `path`, else the manifest's."""
    by: dict[tuple[str, str], list[str]] = {}
    for name in names:
        want = man["starts"][name]
        art = want.get("art") or man.get("art") or ""
        if not art.startswith("art:"):
            raise SystemExit(f"starts_store: the manifest names no artifact for {name} (export, then `research data put "
                             "--tree`)")
        by.setdefault((art, want.get("path") or man.get("path", "starts")), []).append(name)
    return by


def checked(man: dict, names, tmp: Path, fetch) -> dict[str, Path]:
    """Each of `names`' directory in its fetched artifact, every file checked against the manifest."""
    dirs = {}
    for i, ((art, path), group) in enumerate(sources(man, names).items()):
        (tmp / str(i)).mkdir()
        root = fetch(art, tmp / str(i)) / path
        for name in group:
            want, src = man["starts"][name], root / name
            for f, h in want["files"].items():
                if not (src / f).is_file() or sha256(src / f) != h:
                    raise SystemExit(f"starts_store: {art}'s {name}/{f} is not the manifest's (sha256 {h})")
            if json.loads((src / "start.json").read_text())["digest"] != want["digest"]:
                raise SystemExit(f"starts_store: {art}'s {name}/start.json names another digest")
            dirs[name] = src
    return dirs


def pack(man: dict, out: Path, fetch=store_fetch) -> Path:
    """Every start the manifest lists as one tar (`PATH/<name>/<file>`, each under its artifact's path), checked against
    the manifest."""
    tmp = Path(tempfile.mkdtemp(prefix="starts-pack-"))
    try:
        dirs = checked(man, man["starts"], tmp, fetch)
        paths = {name: path for (_, path), group in sources(man, man["starts"]).items() for name in group}
        with tarfile.open(out, "w") as tf:
            for name, want in man["starts"].items():
                for f in want["files"]:
                    tf.add(dirs[name] / f, arcname=f"{paths[name]}/{name}/{f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def install(to: Path, man: dict, fetch=store_fetch) -> list[str]:
    """Install every start the manifest lists under `to`; the names installed now."""
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
        dirs = checked(man, todo, tmp, fetch)
        for name in todo:
            stage = tmp / f"stage-{name}"
            stage.mkdir()
            for f in man["starts"][name]["files"]:
                shutil.move(str(dirs[name] / f), stage / f)
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
    p = sub.add_parser("pack")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    i = sub.add_parser("install")
    i.add_argument("--to", type=Path, default=STARTS)
    i.add_argument("--manifest", type=Path, default=MANIFEST)
    i.add_argument("--tar", type=Path, default=None, help="pack's tar, sent with the run, instead of the store")
    i.add_argument("--keep", action="store_true", help="record the directory's retention (research keep)")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        print(json.dumps(export(a.src, a.out, a.names)))
        return 0
    man = json.loads(a.manifest.read_text())
    if a.cmd == "pack":
        print(pack(man, a.out))
        return 0
    done = install(a.to, man, from_tar(a.tar) if a.tar else store_fetch)
    if a.keep:
        _research("keep", str(a.to), "--owner", "@training", "--expires", "never", "--priority", "keep", "--art", man["art"],
                  "--regenerate", "python3 benchmarks/covert_witness/starts_store.py install --keep [--tar pack's tar]",
                  "--why", "covert_witness's registered starts, the one directory prepare reads them from")
    print(json.dumps({"installed": done, "to": str(a.to), "art": man["art"],
                      "starts": {n: s["digest"] for n, s in man["starts"].items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
