"""The guest build pin: what a verifier needs to REBUILD the guest ELF a record's proofs were made with.

GUIDANCE §57.10 (S12 caveat): metadata is not build safety.  ``guest_build`` pins the source (commit + tree
hash of ``zk/sp1``, which contains ``Cargo.lock``), the toolchain (``cargo prove``, the succinct ``rustc``, the
``sp1-sdk`` / ``sp1-zkvm`` crate versions from the lock file), the RESOLVED feature set of the guest crate, and
the result (``elf_sha256``, ``vk_hash``), with ``built_by`` / ``built_at``.  It is written by whoever runs
``cargo prove build`` (``pin``), stored beside the host binary as ``<host>.guest_build.json`` so
``run_composed`` copies it into ``manifest.inputs.guest_build``, and it is what an ``approved_guests.json``
entry with ``approval = "reproduced"`` carries.

    python -m e2e.guest_build pin   --repo /path/to/veritor --elf zk/sp1/target/elf-compilation/.../veritor-zk-guest \\
                                    [--host zk/sp1/target/release/veritor-zk-host] [--features a,b] --out pin.json
    python -m e2e.guest_build check --pin pin.json --elf <freshly built ELF> [--host <host>]

``check`` is the reproducibility test: the ELF sha (and vk, when a host is given) of a fresh build must equal
the pin's.  The pin itself never proves anything to a verifier: only a verifier's own rebuild (``check``
passing) or its explicit approval does.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "veritor/guest-build/v1"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str | None:
    try:
        return subprocess.run(cmd, cwd=cwd, env=env, check=True, capture_output=True, text=True, timeout=600).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def source_state(repo: Path, subdir: str = "zk/sp1") -> dict[str, Any]:
    """Commit, git tree id of ``subdir`` at HEAD, and a content hash of the WORKING copy of every tracked file
    under ``subdir`` (so a dirty tree is visible: ``working_tree_sha256 != tree_sha_of_head`` only tells you
    it differs; ``dirty_files`` names the files)."""

    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    tree = _run(["git", "rev-parse", f"HEAD:{subdir}"], cwd=repo)
    files = (_run(["git", "ls-files", "-z", subdir], cwd=repo) or "").split("\0")
    files = sorted(f for f in files if f)
    h = hashlib.sha256()
    for rel in files:
        p = repo / rel
        h.update(rel.encode() + b"\0")
        h.update((_sha256(p) if p.is_file() else "missing").encode() + b"\n")
    status = _run(["git", "status", "--porcelain", "--", subdir], cwd=repo) or ""
    dirty = [line[3:] for line in status.splitlines() if line.strip()]
    return {"source_commit": commit, "source_tree_sha": tree, "source_subdir": subdir, "tracked_files": len(files),
            "working_tree_sha256": h.hexdigest(), "dirty_files": dirty,
            "describe": _run(["git", "describe", "--always", "--dirty", "--abbrev=12"], cwd=repo)}


def toolchain(repo: Path) -> dict[str, Any]:
    """``cargo prove`` / succinct ``rustc`` / stable ``rustc`` / ``cargo`` versions and the pinned sp1 crate versions."""

    lock = (repo / "zk" / "sp1" / "Cargo.lock").read_text() if (repo / "zk" / "sp1" / "Cargo.lock").exists() else ""
    crates: dict[str, str] = {}
    for name in ("sp1-sdk", "sp1-zkvm", "sp1-build", "sp1-core-machine", "sp1-prover"):
        m = re.search(rf'name = "{re.escape(name)}"\nversion = "([^"]+)"', lock)
        if m:
            crates[name] = m.group(1)
    return {
        "cargo_prove": _run(["cargo", "prove", "--version"]),
        "rustc": _run(["rustup", "run", "succinct", "rustc", "--version"]) or _run(["rustc", "+succinct", "--version"]),
        "rustc_stable": _run(["rustc", "--version"]),
        "cargo": _run(["cargo", "--version"]),
        "sp1_sdk": crates.get("sp1-sdk"),
        "sp1_zkvm": crates.get("sp1-zkvm"),
        "sp1_crates": crates,
        "cargo_lock_sha256": hashlib.sha256(lock.encode()).hexdigest() if lock else None,
        "rust_toolchain_toml": (repo / "zk" / "sp1" / "rust-toolchain.toml").read_text().strip() if (repo / "zk" / "sp1" / "rust-toolchain.toml").exists() else None,
    }


def resolved_features(repo: Path, requested: list[str]) -> dict[str, Any]:
    """The guest crate's RESOLVED features (``cargo tree -e features``), not just what was asked for."""

    manifest = repo / "zk" / "sp1" / "guest" / "Cargo.toml"
    cmd = ["cargo", "tree", "--manifest-path", str(manifest), "-e", "features", "--prefix", "none", "--no-dedupe"]
    if requested:
        cmd += ["--features", ",".join(requested)]
    out = _run(cmd, cwd=repo / "zk" / "sp1")
    feats: dict[str, list[str]] = {}
    if out is not None:
        for line in out.splitlines():
            m = re.match(r"^(veritor-zk-guest|veritor-zk-common) feature \"([^\"]+)\"", line.strip())
            if m:
                feats.setdefault(m.group(1), [])
                if m.group(2) not in feats[m.group(1)]:
                    feats[m.group(1)].append(m.group(2))
    unsound = sorted({f for fs in feats.values() for f in fs if f.startswith("unsound")})
    return {"requested": list(requested), "resolved": {k: sorted(v) for k, v in feats.items()} if out is not None else None,
            "resolved_by": "cargo tree -e features (guest crate)" if out is not None else "UNAVAILABLE (cargo tree failed)",
            "unsound_features_resolved": unsound}


def host_info(host: Path | None) -> dict[str, Any] | None:
    if not host or not Path(host).exists():
        return None
    out = _run([str(host), "info"], env={**os.environ, "SP1_PROVER": "cpu"})
    if out is None:
        return None
    try:
        info = json.loads(out.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None
    return {**info, "host_sha256": _sha256(Path(host)), "host_path": str(host)}


def pin(repo: Path, elf: Path, *, host: Path | None, features: list[str], vk_hash: str | None, build_argv: str | None,
        source_commit: str | None = None, expect_tree_sha: str | None = None) -> dict[str, Any]:
    src = source_state(repo)
    tc = toolchain(repo)
    fr = resolved_features(repo, features)
    hi = host_info(host)
    elf_sha = _sha256(elf)
    problems: list[str] = []
    if source_commit:
        # a tree-imported checkout (the pod gets the files, not the history): the caller names the real commit and the
        # zk/sp1 tree id it must have; the tree id is content-derived, so equality is the check that the source is that commit's
        src["source_commit_local"] = src["source_commit"]
        src["source_commit"] = source_commit
        src["source_commit_from"] = "--source-commit (tree-imported checkout; verified by the zk/sp1 tree id)"
    if expect_tree_sha and src["source_tree_sha"] != expect_tree_sha:
        problems.append(f"zk/sp1 tree id {src['source_tree_sha']} != expected {expect_tree_sha}: this is NOT the named commit's source")
    if hi is not None:
        if hi.get("elf_sha256") != elf_sha:
            problems.append(f"the host embeds ELF {str(hi.get('elf_sha256'))[:12]}, the file is {elf_sha[:12]}")
        vk_hash = vk_hash or hi.get("vk_hash")
        if hi.get("guest_features") not in (None, features):
            problems.append(f"the host reports guest_features {hi.get('guest_features')}, the build requested {features}")
    if fr["unsound_features_resolved"]:
        problems.append(f"UNSOUND features resolved into the guest: {fr['unsound_features_resolved']}")
    if src["dirty_files"]:
        problems.append(f"zk/sp1 working tree is DIRTY ({len(src['dirty_files'])} file(s)): the commit does not name this source")
    return {
        "schema": SCHEMA,
        "source_commit": src["source_commit"], "source_tree_sha": src["source_tree_sha"], "source": src,
        "toolchain": {"cargo_prove": tc["cargo_prove"], "rustc": tc["rustc"], "sp1_sdk": tc["sp1_sdk"], **{k: v for k, v in tc.items() if k not in ("cargo_prove", "rustc", "sp1_sdk")}},
        "features_resolved": (fr["resolved"] or {}).get("veritor-zk-guest", [] if fr["resolved"] is not None else None), "features": fr,
        "elf_sha256": elf_sha, "elf_path": str(elf), "elf_bytes": elf.stat().st_size, "vk_hash": vk_hash,
        "host": hi,
        "built_by": f"{getpass.getuser()}@{socket.gethostname()}", "built_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": platform.platform(), "build_argv": build_argv,
        "problems": problems,
    }


def check(pin_doc: dict[str, Any], elf: Path, host: Path | None) -> dict[str, Any]:
    """Reproducibility: a fresh ELF (and host vk) against a pin."""

    elf_sha = _sha256(elf)
    hi = host_info(host)
    same_elf = elf_sha == pin_doc.get("elf_sha256")
    same_vk = (hi is None) or (str(hi.get("vk_hash", "")).lower() == str(pin_doc.get("vk_hash", "")).lower())
    return {"pin_elf_sha256": pin_doc.get("elf_sha256"), "fresh_elf_sha256": elf_sha, "elf_bit_identical": same_elf,
            "pin_vk_hash": pin_doc.get("vk_hash"), "fresh_vk_hash": (hi or {}).get("vk_hash"), "vk_identical": same_vk if hi is not None else None,
            "reproduced": same_elf and same_vk}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pin", help="write the build pin of an ELF just built from this repo")
    p.add_argument("--repo", required=True)
    p.add_argument("--elf", required=True)
    p.add_argument("--host", default=None, help="a host binary embedding the ELF (its `info` gives vk_hash; checked against the ELF file)")
    p.add_argument("--features", default="", help="comma list of cargo features passed to `cargo prove build` (default none)")
    p.add_argument("--vk-hash", default=None)
    p.add_argument("--build-argv", default=None, help="the exact build command line, for the record")
    p.add_argument("--source-commit", default=None, help="the real commit when the checkout is a tree import (verified by --expect-tree-sha)")
    p.add_argument("--expect-tree-sha", default=None, help="the git tree id `git rev-parse <commit>:zk/sp1` the checkout must have")
    p.add_argument("--out", required=True)
    c = sub.add_parser("check", help="compare a fresh build with a pin (reproducibility)")
    c.add_argument("--pin", required=True)
    c.add_argument("--elf", required=True)
    c.add_argument("--host", default=None)
    c.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "pin":
        doc = pin(Path(a.repo).resolve(), Path(a.elf).resolve(), host=Path(a.host).resolve() if a.host else None,
                  features=[f for f in a.features.split(",") if f], vk_hash=a.vk_hash, build_argv=a.build_argv,
                  source_commit=a.source_commit, expect_tree_sha=a.expect_tree_sha)
        Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"[guest_build] pin -> {a.out}: commit {str(doc['source_commit'])[:12]} tree {str(doc['source_tree_sha'])[:12]} elf {doc['elf_sha256'][:12]} vk {str(doc['vk_hash'])[:14]} "
              f"features {doc['features_resolved']}" + (f"; PROBLEMS: {doc['problems']}" if doc["problems"] else ""))
        return 0 if not doc["problems"] else 1
    doc = check(json.loads(Path(a.pin).read_text()), Path(a.elf).resolve(), Path(a.host).resolve() if a.host else None)
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"[guest_build] {'REPRODUCED' if doc['reproduced'] else 'NOT REPRODUCED'}: elf {doc['fresh_elf_sha256'][:12]} vs pin {str(doc['pin_elf_sha256'])[:12]}; vk identical {doc['vk_identical']}")
    return 0 if doc["reproduced"] else 1


if __name__ == "__main__":
    sys.exit(main())
