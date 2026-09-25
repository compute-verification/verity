"""Independent re-verification of stored B-Ligero results, from the evidence store alone (the coordinator's / an auditor's check).

    uv run python -m backends.direct.ligero.reverify RESULT... --by WHO [--verifier BIN | --build] [--relation R]
                                                     [--jobs N] [--work DIR] [--keep] [--dry-run] [--json]

RESULT is a bench-result/v1 artifact id (abbreviated ids welcome).  For each one:

  1. the dump tree is the result's `run_files` ref and the run id its producing attempt; the tree is fetched into --work after
     making room (blobs the remote verifiably holds are evicted when the disk is short);
  2. custody: every proof's and coins file's sha256 and every statement's size match `proofs/manifest.json`; system.bin its sha256;
  3. `ligero-verify system-digest` must PIN the system to a registered relation whose base name is the one the manifest claims
     (and the attempt's `relation` param, and --relation when given);
  4. `ligero-verify batch --target-bits 128` on every rep dir: batch ACCEPT, zero rejections, Rust/Python verdicts agree, and
     the rep holds as many proofs as the manifest lists for it;
     before it, a dump whose statements carry hash authentication (included-hash) must commit to the claimed instance set: the
     a/b/y bindings and roots are recomputed from the set (--instances-root, default $BENCH_INSTANCES) and every statement's
     trees must equal them, and each rep's statements must cover every VU of [0, total) exactly once;
  5. PASS -> a PRESERVED verification-verdict/v1 artifact (refs proof=<run-files>, result=RESULT; the per-rep batch JSON as its
     payload) and labels on RESULT by --by: verified=accepted, verifier, verifier_seconds, note (pushed with the labels sync).
     FAIL (custody / pin / verifier rejection) -> nothing is written; the per-rep JSON stays in --work, exit 1.
     ERROR (fetch, disk, a verifier that cannot read the format) -> not a rejection, never labelled as one.

The verifier's identity is computed, never typed: the binary's sha256, the last commit that touched backends/ligero-verify, and
a refusal when that source tree is newer than the binary (rebuild with --build) or dirty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CRATE = REPO / "backends" / "ligero-verify"
TARGET_BITS = 128
VERDICT_KIND = "verification-verdict/v1"
DISK_MARGIN = 512 << 20


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()


# ---- the verifier ----------------------------------------------------------------------------------------------------------------------

@dataclass
class Verifier:
    path: Path
    sha256: str
    src_commit: str | None          # last commit touching backends/ligero-verify (None: a binary from outside this repo)

    @property
    def tag(self) -> str:
        return f"ligero-verify@{self.src_commit[:7]}" if self.src_commit else f"ligero-verify(sha256 {self.sha256[:16]})"

    def run(self, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.path), *args], cwd=cwd, capture_output=True, text=True)


def default_binary() -> Path:
    return Path(os.environ.get("CARGO_TARGET_DIR") or CRATE / "target") / "release" / "ligero-verify"


def build() -> Path:
    r = subprocess.run(["cargo", "build", "--release", "--manifest-path", str(CRATE / "Cargo.toml")], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"reverify: cargo build failed:\n{r.stderr[-2000:]}")
    return default_binary()


def crate_of(binary: Path, default: bool) -> Path | None:
    """The ligero-verify crate a binary was built in: the `backends/ligero-verify` directory above it (any worktree), or this
    repo's crate for the default path (CARGO_TARGET_DIR may be elsewhere).  None: a binary from outside any checkout."""
    for d in binary.parents:
        if d.name == "ligero-verify" and (d / "Cargo.toml").is_file():
            return d
    return CRATE if default else None


def load_verifier(path: Path | None, *, allow_stale: bool = False) -> Verifier:
    """The binary plus its provenance.  A binary built in a checkout must be at least as new as every verifier source file there
    and those sources must be committed: otherwise the label would name a commit the binary was not built from."""
    p = (path or default_binary()).expanduser().resolve()
    if not p.is_file():
        raise SystemExit(f"reverify: no verifier binary at {p} (build it with --build, or give --verifier BIN)")
    crate = crate_of(p, default=path is None)
    src_commit = None
    if crate is not None:
        sources = [q for q in crate.rglob("*") if q.is_file() and "target" not in q.relative_to(crate).parts]
        newest = max(sources, key=lambda q: q.stat().st_mtime)
        dirty = _git(crate, "status", "--porcelain", "--", ".")
        problems = ([f"{newest.relative_to(crate)} is newer than the binary"] if newest.stat().st_mtime > p.stat().st_mtime else []) + \
                   ([f"uncommitted changes under {crate}"] if dirty else [])
        if problems and not allow_stale:
            raise SystemExit(f"reverify: {p} does not correspond to a commit: {'; '.join(problems)} (rebuild with --build, commit, "
                             "or --allow-stale to verify anyway with an untagged binary)")
        src_commit = None if problems else (_git(crate, "log", "-1", "--format=%H", "--", ".") or None)
    return Verifier(p, _sha256(p), src_commit)


# ---- one result -------------------------------------------------------------------------------------------------------------------------

@dataclass
class Report:
    result: str
    status: str = "ERROR"                   # PASS | FAIL | ERROR | SKIP
    why: list[str] = field(default_factory=list)
    run_files: str | None = None
    run: str | None = None
    relation: str | None = None
    mode: str | None = None
    custody: dict[str, int] = field(default_factory=dict)
    reps: dict[str, dict] = field(default_factory=dict)
    verdict: str | None = None
    labels: list[str] = field(default_factory=list)
    hashed: bool = False                    # included-hash statements: trees recomputed from the instance set

    def line(self) -> str:
        acc = sum(v.get("accepted", 0) for v in self.reps.values())
        tot = sum(v.get("n", 0) for v in self.reps.values())
        bits = min((v.get("batch_bits") or 0.0 for v in self.reps.values()), default=0.0)
        facts = (f"custody {self.custody.get('files', 0) - self.custody.get('mismatched', 0)}/{self.custody.get('files', 0)}, pinned={self.relation}, "
                 f"{acc}/{tot}, 2^-{bits:.2f}") if self.reps else ""
        tail = f" -> verdict {self.verdict}" if self.verdict else ""
        return f"{self.result[:16]}… {self.run or '?'}: {facts}{' ' if facts else ''}{self.status}{': ' + '; '.join(self.why) if self.why else ''}{tail}"


def custody(pdir: Path, man: dict) -> tuple[int, list[str], list[str]]:
    """(files the manifest lists, mismatched, missing): sha256 where the manifest records one, size for statements.  Missing files
    are absent evidence (a dump that never left the pod), not a broken chain of custody."""
    bad: list[str] = []
    missing: list[str] = []
    n = 0
    sysf = man.get("system_file") or {}
    listed = [(rel, f.get(f"{key}_sha256"), f.get(f"{key}_bytes")) for f in man.get("files", []) for key in ("proof", "coins", "stmt") if (rel := f.get(key))]
    for rel, sha, size in [*listed, (sysf.get("path", "system.bin"), sysf.get("sha256"), sysf.get("bytes"))]:
        n += 1
        p = pdir / rel
        if not p.is_file():
            missing.append(rel)
        elif sha and _sha256(p) != sha:
            bad.append(f"{rel} sha256 mismatch")
        elif size is not None and p.stat().st_size != size:
            bad.append(f"{rel} size {p.stat().st_size} != {size}")
    return n, bad, missing


def expected_relations(man: dict, params: dict, asserted: str | None) -> dict[str, str]:
    out = {}
    mrel = man.get("relation")
    if isinstance(mrel, str) and mrel:
        out["manifest"] = mrel                        # fp4/chain.py writes the relation's name, relchain.py a {name, ...} block
    elif isinstance(mrel, dict) and mrel.get("name"):
        out["manifest"] = mrel["name"]
    if isinstance(params.get("relation"), str) and params["relation"] != "<redacted>":
        out["attempt"] = params["relation"]
    if asserted:
        out["--relation"] = asserted
    return out


# ---- the committed set (authentication = included-hash) ---------------------------------------------------------------------------------

TreeKey = tuple[bytes, int, int, bytes]          # (binding, owner, count, root)


def _row_digests(rows, word_bits: int, role: int) -> list[bytes]:
    """Keyed-BLAKE3 row digests (``blake3-keyed/row/v2``): the ``blake3`` package when installed, else core's reference."""
    from verity.commitments import rowleaf
    key = rowleaf.blake3_row_key(role)
    try:
        import blake3
    except ImportError:
        return [rowleaf.blake3_row_digest(r, word_bits, role) for r in rows]
    return [blake3.blake3(rowleaf.row_bytes(r, word_bits), key=key).digest() for r in rows]


def committed_trees(pinned: str, man: dict, *, instances_root: str | None = None) -> dict[str, TreeKey]:
    """The x / W / y tree references of the instance set the dump names, recomputed from the set itself: the pinned relation
    (``system-digest``) fixes the relation and leaf scheme, the manifest's ``set`` block the range ``[0, total_vus)`` and the
    set's manifest digest, which the regenerated set must reproduce.  Bindings are ``hashauth.binding_digest`` of (dataset,
    tier, manifest digest, range, K, tree, schema); keyed-BLAKE3 row trees are framed by core ``frame-v3`` over digests of the
    raw rows (independent of the prover's leaf code); other leaf schemes use the backend's reference tree builder.
    ``ValueError`` when the set cannot be recomputed (no ``set`` block, a tile, a digest mismatch, an operand format with
    its own lanes)."""
    from verity.commitments import CommitmentDomain, RangeIndexedDomain
    from verity.commitments.frame_v3 import FrameV3, Port, RowLeaf, WordLeaf
    from verity.commitments.rowleaf import ROLE_W, ROLE_X

    from . import hashauth
    from .auth import OWNER
    from .leaf import registry as leaf_registry
    from .relations import relation
    from .relchain import instances, instances_digest

    base, leaf_name = leaf_registry.split_relation_name(pinned)
    s = man.get("set") or {}
    total, claimed = s.get("total_vus"), s.get("instances")
    if not isinstance(total, int) or total <= 0 or not isinstance(claimed, str):
        raise ValueError("the dump names no instance set (manifest `set`: total_vus, instances): the commitments cannot be recomputed")
    if s.get("tile") or s.get("sharing") not in (None, "none"):
        raise ValueError(f"shared-row layouts (set {s.get('tile') or s.get('sharing')}) are not recomputed")
    rel = relation(base)
    if getattr(rel, "hash_format", None) is not None or rel.word_dtype not in ("<u1", "<u2"):
        raise ValueError(f"{rel.name}: operands of dtype {rel.word_dtype} / a lane format are not recomputed")
    digest = instances_digest(rel, total, root=instances_root)
    if digest != claimed:
        raise ValueError(f"the regenerated {rel.name} set of {total} VUs has manifest digest {digest[:16]}…, the dump claims {claimed[:16]}…")
    data = instances(rel, total, root=instances_root)
    word_bits = 16 if rel.word_dtype == "<u2" else 8
    K = int(getattr(rel, "vu_words", 1536))
    scheme = leaf_registry.get(leaf_name)
    ysch, ybytes = hashauth.word_schema(rel.y_bits)
    x_rows = [[int(w) for w in v[0]] for v in data]
    w_cols = [[int(w) for w in v[1]] for v in data]
    y_words = [int(rel.y_public(int(v[3]))) for v in data]
    if any(len(r) != K for r in x_rows + w_cols):
        raise ValueError(f"{rel.name}: instance rows are not K = {K} words")
    bindings = {n: hashauth.binding_digest(dataset=rel.instances_dataset, tier=rel.instances_tier, manifest_sha256=claimed, lo=0,
                                           hi=total, K=K, tree=n, schema=(scheme.schema if n != "y" else ysch)) for n in ("a", "b", "y")}
    if scheme.schema == "blake3-keyed/row/v2":
        dom = {n: CommitmentDomain(bindings[n], OWNER[n], RangeIndexedDomain(0, total)) for n in ("a", "b", "y")}
        fv = FrameV3({"a": Port(dom["a"], RowLeaf("blake3", ROLE_X, word_bits, K)), "b": Port(dom["b"], RowLeaf("blake3", ROLE_W, word_bits, K)),
                      "y": Port(dom["y"], WordLeaf(ysch, ybytes))})
        leaves = {"a": [fv.tree_leaf("a", r, d) for r, d in enumerate(_row_digests(x_rows, word_bits, ROLE_X))],
                  "b": [fv.tree_leaf("b", r, d) for r, d in enumerate(_row_digests(w_cols, word_bits, ROLE_W))],
                  "y": [fv.tree_leaf("y", r, fv.leaf_layout("y", r).digest(w.to_bytes(ybytes, "big"))) for r, w in enumerate(y_words)]}
        return {n: (bindings[n], OWNER[n], total, fv.root(n, leaves[n])) for n in ("a", "b", "y")}
    import numpy as np
    trees, _ = hashauth.build_trees(x_rows=np.asarray(x_rows), w_cols=np.asarray(w_cols), y_words=np.asarray(y_words),
                                    word_bits=word_bits, y_bits=rel.y_bits, bindings=bindings, leaf=scheme)
    return {n: (t.ref.binding, t.ref.owner, t.ref.count, t.ref.root) for n, t in trees.items()}


def commitment_problems(pdir: Path, man: dict, pinned: str, reps: list[str], *, instances_root: str | None = None) -> tuple[bool, list[str]]:
    """``(hashed, problems)``.  A dump whose pinned relation names a leaf (``+hash``, ``+blake3``, ...: the systems that take a
    hash-authentication block) passes only when every statement carries the block, its three tree references equal
    :func:`committed_trees` and each rep's statements' ``vu_index`` together cover ``[0, total_vus)`` exactly once
    (red-team-standard-hash R2); a bare relation's statements are not read here.  Coverage counts only what ``batch``
    verifies, so in every dump each rep's ``.stmt`` stems must equal its ``.proof`` stems and the manifest's entries for it
    (red-team-standard-hash R4: a statement without a proof claims VUs nobody proved)."""
    from .serialize import read_statement

    orphans: list[str] = []
    for r in reps:
        stmts, proofs = ({f.stem for f in (pdir / r).glob(f"*.{s}")} for s in ("stmt", "proof"))
        listed = [f for f in man.get("files", []) if isinstance(f, dict) and str(f.get("proof") or f.get("stmt") or "").split("/", 1)[0] == r]
        if stmts != proofs:
            orphans.append(f"{r}: {len(stmts - proofs)} statement(s) without a proof, {len(proofs - stmts)} proof(s) without a statement")
        if any(not f.get("proof") for f in listed):
            orphans.append(f"{r}: the manifest lists a statement without a proof")
        listed_proofs = [Path(f["proof"]).stem for f in listed if f.get("proof")]
        if listed and (set(listed_proofs) != proofs or len(listed_proofs) != len(set(listed_proofs))):
            orphans.append(f"{r}: the manifest's proofs are not the rep's proofs")
        if any(f.get("stmt") and Path(f["stmt"]).stem != Path(f["proof"]).stem for f in listed if f.get("proof")):
            orphans.append(f"{r}: the manifest pairs a proof with another sub-batch's statement")
    hashed = "+" in (pinned or "")
    if orphans or not hashed:
        return hashed, orphans
    per_rep: dict[str, list] = {}
    unreadable: list[str] = []
    for r in reps:
        for f in sorted((pdir / r).glob("*.stmt")):
            try:
                per_rep.setdefault(r, []).append((f.name, read_statement(f.read_bytes())))
            except Exception as e:  # noqa: BLE001  (a statement batch would reject cannot be counted either)
                unreadable.append(f"{r}/{f.name}: unreadable statement ({type(e).__name__}: {e})")
    if unreadable:
        return True, unreadable[:8] + ([f"… {len(unreadable) - 8} more"] if len(unreadable) > 8 else [])
    if n_v6 := sum(bool(getattr(st, "v6", False)) for sts in per_rep.values() for _, st in sts):
        return True, [f"commitment: {n_v6} row-sharing (v6) statement(s): shared-row layouts are not recomputed"]
    try:
        want = committed_trees(pinned, man, instances_root=instances_root)
    except ValueError as e:
        return True, [f"commitment: {e}"]
    total = want["y"][2]
    problems: list[str] = []
    for r in reps:
        sts = per_rep.get(r, [])
        covered: list[int] = []
        for name, st in sts:
            ha = getattr(st, "hash_auth", None)
            if ha is None:
                problems.append(f"{r}/{name}: no hash-authentication block beside hashed statements")
                continue
            if sorted(t.name for t in ha.trees) != sorted(want):
                problems.append(f"{r}/{name}: trees {sorted(t.name for t in ha.trees)}, the instance set commits {sorted(want)}")
                continue
            for t in ha.trees:
                got = (bytes(t.binding), int(t.owner), int(t.count), bytes(t.root))
                if got != want[t.name]:
                    what = [k for k, a, b in zip(("binding", "owner", "count", "root"), got, want[t.name]) if a != b]
                    problems.append(f"{r}/{name}: tree {t.name} {'/'.join(what)} differ from the instance set's")
            covered += list(ha.vu_index)
        if sorted(covered) != list(range(total)):
            dup = len(covered) - len(set(covered))
            problems.append(f"{r}: statements cover {len(set(covered))} distinct VUs of [0, {total}) ({dup} claimed twice)")
    return True, problems[:8] + ([f"… {len(problems) - 8} more"] if len(problems) > 8 else [])


def verify_tree(pdir: Path, man: dict, v: Verifier, rep: Report, *, jobs: int, params: dict, asserted: str | None, out_dir: Path,
                instances_root: str | None = None) -> None:
    n, bad, missing = custody(pdir, man)
    rep.custody = {"files": n, "mismatched": len(bad), "missing": len(missing)}
    if bad:
        rep.status, rep.why = "FAIL", [f"custody: {', '.join(bad[:5])}{' …' if len(bad) > 5 else ''}"]
        return
    if missing:
        rep.status, rep.why = "ERROR", [f"incomplete dump: {len(missing)} of {n} files the manifest lists are not in the tree (never uploaded?), "
                                        f"e.g. {', '.join(missing[:3])}"]
        return
    sysrel = (man.get("system_file") or {}).get("path", "system.bin")
    d = v.run(["system-digest", "--system", sysrel], pdir)
    if d.returncode != 0:
        rep.status, rep.why = "ERROR", [f"system-digest failed (a verifier older than this system format?): {d.stderr.strip()[-300:]}"]
        return
    dig = json.loads(d.stdout)
    rep.relation = dig.get("pinned_relation")
    exp = expected_relations(man, params, asserted)
    if not rep.relation:
        rep.status, rep.why = "FAIL", ["system is not pinned to any registered relation"]
        return
    base = rep.relation.split("+")[0]
    wrong = [f"{src} says {r}" for src, r in exp.items() if r not in (rep.relation, base)]
    if wrong:
        rep.status, rep.why = "FAIL", [f"pinned {rep.relation} but {', '.join(wrong)}"]
        return
    per_rep_n: dict[str, int] = {}
    for f in man.get("files", []):
        if f.get("proof"):
            k = f["proof"].split("/", 1)[0]
            per_rep_n[k] = per_rep_n.get(k, 0) + 1
    reps = sorted(p.name for p in pdir.iterdir() if p.is_dir() and p.name.startswith("rep"))
    if not reps:
        rep.status, rep.why = "FAIL", ["no rep*/ directory in the dump"]
        return
    rep.hashed, bad = commitment_problems(pdir, man, rep.relation, reps, instances_root=instances_root)
    if bad:
        rep.status, rep.why = "FAIL", bad
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in reps:
        out = out_dir / f"{r}.json"
        b = v.run(["batch", "--system", sysrel, "--dir", r, "--jobs", str(jobs), "--threads", "1", "--target-bits", str(TARGET_BITS), "--json", str(out)], pdir)
        if not out.is_file():
            rep.status, rep.why = "ERROR", [f"{r}: verifier wrote no JSON (rc {b.returncode}): {b.stderr.strip()[-300:]}"]
            return
        rep.reps[r] = json.loads(out.read_text())
    problems = []
    for r, x in rep.reps.items():
        if not (x.get("batch_accepted") and x.get("system_pinned")):
            problems.append(f"{r}: batch {'REJECT' if not x.get('batch_accepted') else 'unpinned'} ({x.get('batch_reason')})")
        if x.get("rejected"):
            problems.append(f"{r}: {x['rejected']} proof(s) rejected")
        if x.get("python_disagree"):
            problems.append(f"{r}: Rust/Python disagree on {x['python_disagree']}")
        if per_rep_n.get(r) not in (None, x.get("n")):
            problems.append(f"{r}: {x.get('n')} proofs verified, manifest lists {per_rep_n[r]}")
        if x.get("n") != (n_stmt := len(list((pdir / r).glob("*.stmt")))):
            problems.append(f"{r}: {x.get('n')} proofs verified, the rep holds {n_stmt} statements")
    rep.status, rep.why = ("FAIL", problems) if problems else ("PASS", [])


def detail(rep: Report, v: Verifier) -> str:
    acc = sum(x["accepted"] for x in rep.reps.values())
    tot = sum(x["n"] for x in rep.reps.values())
    bits = min(x["batch_bits"] for x in rep.reps.values())
    s = (f"{v.tag} (sha256 {v.sha256[:16]}) batch --target-bits {TARGET_BITS} on every rep of the run-files tree {rep.run_files} "
         f"(custody: {rep.custody['files']}/{rep.custody['files']} files match proofs/manifest.json): {acc}/{tot} accepted over {len(rep.reps)} rep(s), "
         f"every batch ACCEPT, min union bound 2^-{bits:.2f}, system PINNED ({rep.relation}), Rust/Python verdict agreement {tot}/{tot}; "
         f"host {os.uname().nodename}, 1 thread per proof; mode={rep.mode}")
    if rep.hashed:
        s += "; hash commitments recomputed from the instance set (a/b/y bindings + roots equal, every VU covered once per rep)"
    if rep.mode == "interactive":
        s += (" -- each sub-batch verified against its recorded step-0 coins (the live verifier's session coins): consistency with that "
              "session, not transferable evidence (D7)")
    return s


def free_bytes(path: Path) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def reverify(st: Any, result: str, v: Verifier, *, by: str | None, jobs: int, work: Path, asserted: str | None, dry_run: bool, keep: bool,
             instances_root: str | None = None) -> Report:
    from research.store import evict as EV
    from research.store.ids import is_run_id

    rep = Report(result)
    m = st.get_manifest(result)
    rep.run_files = (m.refs or {}).get("run_files")
    if not isinstance(rep.run_files, str):
        rep.why = [f"{m.kind} has no run_files ref: nothing was dumped for this result"]
        return rep
    producers = st.index.attempts_by_output(result)
    rep.run = producers[0] if producers else (m.meta or {}).get("run_id")
    # a result put by hand from a pod run carries the bench's own run_id (b-ligero-...), which names no attempt
    att = st.get_attempt(rep.run) if is_run_id(rep.run) else None
    params = (att or {}).get("params") or {}
    tree = st.get_manifest(rep.run_files)
    need = 2 * (tree.payload.bytes if tree.payload else 0) + DISK_MARGIN
    work.mkdir(parents=True, exist_ok=True)
    if free_bytes(work) < need:
        EV.evict_store(st, target_free_bytes=need, dry_run=False, min_bytes=EV.MIN_BYTES, keep_trees=False, log=None)
    if free_bytes(work) < need:
        rep.status, rep.why = "SKIP", [f"needs {need / 2**30:.1f} GiB free under {work}, have {free_bytes(work) / 2**30:.1f} after eviction"]
        return rep
    d = work / result.removeprefix("art:")[:16]
    try:
        st.fetch(rep.run_files, d)
        pdir = next((d / n for n in ("proofs", "dumps") if (d / n / "manifest.json").is_file()), None)
        if pdir is None:
            rep.why = [f"{rep.run_files} has no proofs/ or dumps/ manifest.json"]
            return rep
        man = json.loads((pdir / "manifest.json").read_text())
        rep.mode = man.get("mode") or params.get("mode")
        per_rep = work / f"{d.name}.reps"
        verify_tree(pdir, man, v, rep, jobs=jobs, params=params, asserted=asserted, out_dir=per_rep, instances_root=instances_root)
        if rep.status != "PASS":
            if per_rep.is_dir():
                print(f"reverify: {result}: per-rep verifier JSON kept in {per_rep}", file=sys.stderr)
            return rep
        shutil.rmtree(per_rep, ignore_errors=True)
        if dry_run:
            return rep
        agg = work / f"{d.name}.verdict.json"
        agg.write_text(json.dumps({"reps": rep.reps, "custody": rep.custody, "verifier": {"path": str(v.path), "sha256": v.sha256, "src_commit": v.src_commit}}))
        det = detail(rep, v)
        vsum = round(sum(x["verify_seconds_sum"] for x in rep.reps.values()), 3)
        meta = {"result": "PASS", "verifier": v.tag, "verifier_sha256": v.sha256, "detail": det, "run": rep.run, "relation": rep.relation, "mode": rep.mode,
                "accepted": sum(x["accepted"] for x in rep.reps.values()), "proofs": sum(x["n"] for x in rep.reps.values()), "reps": len(rep.reps),
                "batch_bits_min": min(x["batch_bits"] for x in rep.reps.values()), "verify_sum_seconds": vsum, "custody": rep.custody}
        vid = st.put_artifact(VERDICT_KIND, meta, {"proof": rep.run_files, "result": result}, file=agg)
        pushed = st.push(vid, preserve=True)
        if not pushed["preserved"]:
            rep.status, rep.why = "ERROR", [f"verdict {vid} not preserved: {'; '.join(pushed['errors'])} (no labels written)"]
            return rep
        rep.verdict = vid
        _label(st, result, by, vid, rep, v, det, vsum)
        return rep
    finally:
        if not keep:
            shutil.rmtree(d, ignore_errors=True)


def _label(st: Any, result: str, by: str, vid: str, rep: Report, v: Verifier, det: str, vsum: float) -> None:
    from research.store.cli import main as data

    who = f"{v.tag} (pinned {rep.relation}; independent of the producing lane; host {os.uname().nodename})"
    for key, val, extra in (("verified", "accepted", []), ("verifier", who, []), ("verifier_seconds", str(vsum), ["--value-json"]), ("note", det, [])):
        if data(["label", result, key, val, "--by", by, "--ref", vid, "--store", str(st.root), *extra]) != 0:
            raise RuntimeError(f"writing label {key} on {result} failed")
        rep.labels.append(key)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backends.direct.ligero.reverify", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", metavar="RESULT", help="bench-result/v1 artifact ids (abbreviated ok)")
    ap.add_argument("--by", default=None, help="the asserter written on the labels (required unless --dry-run)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--verifier", type=Path, default=None, help="ligero-verify binary (default: this repo's target/release build)")
    g.add_argument("--build", action="store_true", help="cargo build --release this repo's verifier first and use it")
    ap.add_argument("--allow-stale", action="store_true", help="verify with a binary that does not match a commit (verdict untagged)")
    ap.add_argument("--relation", default=None, help="also require the pinned relation to be this one")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="parallel proofs per batch (default: half the cores)")
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "reverify", help="where trees are fetched (removed after)")
    ap.add_argument("--keep", action="store_true", help="keep the fetched trees")
    ap.add_argument("--dry-run", action="store_true", help="fetch, check and verify; write nothing to the store")
    ap.add_argument("--store", default=None, help="store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--instances-root", default=os.environ.get("BENCH_INSTANCES"),
                    help="where frozen instance sets live, to recompute hash commitments (default $BENCH_INSTANCES)")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    if not ns.by and not ns.dry_run:
        ap.error("--by is required (who asserts the labels), or --dry-run")

    from research.store import labels_sync as LS
    from research.store.cli import expand_short_ids
    from research.store.local import LocalStore

    st = LocalStore(ns.store)
    v = load_verifier(build() if ns.build else ns.verifier, allow_stale=ns.allow_stale)
    reports = []
    for res in expand_short_ids("show", [*ns.results, *(["--store", ns.store] if ns.store else [])])[:len(ns.results)]:
        t0 = time.monotonic()
        try:
            rep = reverify(st, res, v, by=ns.by, jobs=ns.jobs, work=ns.work, asserted=ns.relation, dry_run=ns.dry_run, keep=ns.keep,
                           instances_root=ns.instances_root)
        except Exception as e:  # noqa: BLE001  (one result's environment failure must not abandon the batch)
            rep = Report(res, "ERROR", [f"{type(e).__name__}: {e}"])
        reports.append(rep)
        if not ns.json:
            print(f"{rep.line()}  ({time.monotonic() - t0:.1f}s)", flush=True)
    if any(r.labels for r in reports) and st.remote is not None:
        LS.sync(st, push=True, pull=False)
    if ns.json:
        print(json.dumps([r.__dict__ for r in reports], indent=2, default=str))
    else:
        n = {s: sum(r.status == s for r in reports) for s in ("PASS", "FAIL", "ERROR", "SKIP")}
        print(f"{len(reports)} result(s): " + ", ".join(f"{k} {c}" for k, c in n.items() if c) + (" (dry run: nothing written)" if ns.dry_run else "")
              + f"; verifier {v.tag}")
    return 0 if all(r.status == "PASS" for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
