"""The machine side of `remote.ship_source`: runs as `<python> -c <this file's text> plan|verify ...` on the machine, before any research
package is guaranteed to be there, so it is stdlib-only and self-contained.

  plan <root> <partial>              stdin: gzip'd JSON {"files": {path: [sha256, executable]}}, the launcher's archive manifest.  Creates
                                     <partial>/ and copies into it every wanted file that a tree under <root>/src/ already holds by content.
                                     Each source is hashed as it is read; one whose bytes no longer match its index entry is not used and
                                     its tree's index is dropped.  Saves the manifest as <partial>/.want.json.gz and prints JSON
                                     {missing: [paths], reused, reused_bytes, indexed, stale}.
  verify <root> <partial> <identity> every regular file under <partial>/ against the saved manifest: the same path set, the same sha256,
                                     the same executable bit.  Prints JSON {ok, files, missing, extra, differs, mode} (counts + examples);
                                     when ok, writes the index of <identity> so later plans reuse it without hashing.

Index: <root>/cache/src-index/<tree name>.sha256 (`<sha256>  <path>` lines), one per directory under <root>/src/ (not dot-names, not
`*.partial-*`), built by hashing the tree the first time a plan needs it -- newest trees first, stopping once every wanted hash is found.
It is derived data: deleting it costs a re-hash, a stale entry is caught by the copy-time hash, and nothing is READY without `verify`.
"""
import gzip
import hashlib
import json
import os
import sys

MARKERS = ("READY.json", ".complete")      # a tree's own markers, not archive content
WANT = ".want.json.gz"
EXAMPLES = 10


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _files(top, skip=()):
    """(posix path relative to top, absolute path) of every regular file; symlinks are neither followed nor listed."""
    for d, dirs, names in os.walk(top):
        dirs.sort()
        for n in sorted(names):
            p = os.path.join(d, n)
            rel = os.path.relpath(p, top).replace(os.sep, "/")
            if rel not in skip and not os.path.islink(p) and os.path.isfile(p):
                yield rel, p


def _index_path(root, name):
    return os.path.join(root, "cache", "src-index", name + ".sha256")


def _read_index(root, name):
    try:
        with open(_index_path(root, name), encoding="utf-8") as f:
            return {path: sha for sha, _, path in (line.rstrip("\n").partition("  ") for line in f) if path}
    except OSError:
        return None


def _write_index(root, name, shas):
    p = _index_path(root, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = f"{p}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(f"{sha}  {path}\n" for path, sha in sorted(shas.items()))
    os.replace(tmp, p)


def _trees(root, partial):
    """(name, path) of every source tree under <root>/src/, newest first."""
    src = os.path.join(root, "src")
    try:
        names = os.listdir(src)
    except OSError:
        return []
    out = []
    for n in names:
        p = os.path.join(src, n)
        if n.startswith(".") or ".partial-" in n or not os.path.isdir(p) or os.path.abspath(p) == os.path.abspath(partial):
            continue
        out.append((os.stat(p).st_mtime, n, p))
    return [(n, p) for _, n, p in sorted(out, reverse=True)]


def plan(root, partial):
    want = json.loads(gzip.decompress(sys.stdin.buffer.read()))["files"]
    os.makedirs(partial)
    with open(os.path.join(partial, WANT), "wb") as f:
        f.write(gzip.compress(json.dumps(want).encode(), 1))
    need = {sha for sha, _ in want.values()}
    have, indexed = {}, []                  # sha -> (tree name, absolute path)
    for name, top in _trees(root, partial):
        if need <= have.keys():
            break
        idx = _read_index(root, name)
        if idx is None:
            idx = {rel: _sha(p) for rel, p in _files(top, MARKERS)}
            _write_index(root, name, idx)
            indexed.append(name)
        for rel, sha in idx.items():
            if sha in need and sha not in have:
                have[sha] = (name, os.path.join(top, rel))
    missing, reused, nbytes, stale = [], 0, 0, set()
    for rel, (sha, executable) in sorted(want.items()):
        data = None
        if sha in have:
            name, src = have[sha]
            try:
                with open(src, "rb") as f:
                    data = f.read()
            except OSError:
                pass
            if data is None or hashlib.sha256(data).hexdigest() != sha:
                stale.add(name)
                data = None
        if data is None:
            missing.append(rel)
            continue
        dst = os.path.join(partial, *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(data)
        os.chmod(dst, 0o755 if executable else 0o644)
        reused += 1
        nbytes += len(data)
    for name in stale:
        try:
            os.remove(_index_path(root, name))
        except OSError:
            pass
    print(json.dumps({"missing": missing, "reused": reused, "reused_bytes": nbytes, "indexed": indexed, "stale": sorted(stale)}))


def verify(root, partial, identity):
    p = os.path.join(partial, WANT)
    with open(p, "rb") as f:
        want = json.loads(gzip.decompress(f.read()))
    os.remove(p)
    got = {rel: (_sha(ap), bool(os.stat(ap).st_mode & 0o100)) for rel, ap in _files(partial)}
    both = set(want) & set(got)
    bad = {"missing": sorted(set(want) - set(got)), "extra": sorted(set(got) - set(want)),
           "differs": sorted(r for r in both if got[r][0] != want[r][0]),
           "mode": sorted(r for r in both if got[r][0] == want[r][0] and got[r][1] != bool(want[r][1]))}
    ok = not any(bad.values())
    if ok:
        _write_index(root, identity, {rel: sha for rel, (sha, _) in got.items()})
    print(json.dumps({"ok": ok, "files": len(got), **{k: {"n": len(v), "e.g.": v[:EXAMPLES]} for k, v in bad.items()}}))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"plan": plan, "verify": verify}[cmd](*args)
