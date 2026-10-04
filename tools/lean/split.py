"""Split a Lean package into its spec and its proofs, as a pure move.

    python tools/lean/split.py PKG --area NAME [--keep MODULE ...] [--apply] [--json]

The spec is what the lock trusts: the modules under `<root>.Protocol`, `<root>.Assumptions` and `<root>.Guarantees`, every
module the lock's `reads` names, and the modules `--keep` names (a spec module under a proof's name, such as a module a
grader trusts), with the package's tools that import only those (its exempt modules and its other libraries). Everything
else is the proofs, umbrella roots included, and moves to a new Lake package `security_proofs/NAME/` that requires the spec
package by path, with the same toolchain and the same Mathlib. A proof module whose root namespace is also a spec module's
moves under a root of its own (`P.SecurityProofs.X` becomes the module `PProofs.SecurityProofs.X`), since Lean's search path
gives every module under a root to the first package holding it; declarations keep their names, so the lock's `guarantees`
and `reads` stay byte for byte. Only its package-level sections are divided between the two packages: `roots`, `proved_in`
(the spec's lock names the new package), and the `exempt`, `runs`, `layers` and `upstream` entries about moved modules,
under their new names.

It refuses, printing the edges, when a spec module imports a proof module: either the import goes, or the imported module is
the spec's too (`--keep`). Without `--apply` it prints the plan and changes nothing. With `--apply` it `git mv`s the proofs,
writes the new package's `lakefile.toml`, `lake-manifest.json`, `lean-toolchain` and `lean-audit.json`, gives the spec
package its roots, gives every package that requires the spec and imports a moved module a require of the new package,
rewrites every Lean file's imports of a renamed module and every other lock's `layers` that let a module import one,
re-keys `tools/lean/lean-deps.json` for each manifest it changes (the bundles stay the same: only path dependencies differ),
rewrites the repository-relative paths of the moved files wherever a tracked file names them, and lists the mentions it
could not rewrite (a path built from parts, prose) for a person to read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC_NAMES = ("Protocol", "Assumptions", "Guarantees")
LEAN_DEPS = "tools/lean/lean-deps.json"
IMPORT = re.compile(r"^(?:public\s+)?(?:meta\s+)?import\s+(.*)$")


def _audit():
    import importlib.util

    spec = importlib.util.spec_from_file_location("lean_audit_for_split", HERE / "audit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = _audit()


def imports(path: Path) -> list[str]:
    """The modules a Lean file's header imports, comments aside."""
    out = []
    for line in A.strip_comments(path.read_text(errors="replace")).splitlines():
        s = line.strip()
        if not s or s in ("module", "prelude"):
            continue
        if not (m := IMPORT.match(s)):
            break
        out += m.group(1).split()
    return out


def lakefile(pkg: Path) -> dict:
    return tomllib.loads((pkg / "lakefile.toml").read_text())


def deps_key(pkg: Path) -> str | None:
    """`lean_audit.deps_key`: what a package's fetched dependencies are keyed by in `lean-deps.json` (restated, since
    `lean_audit` loads `tools/check/common.py`; a test holds the two equal)."""
    m = pkg / "lake-manifest.json"
    if not m.exists() or not [d for d in json.loads(m.read_text()).get("packages", []) if d.get("type") != "path"]:
        return None
    return hashlib.sha256(m.read_bytes() + b"\0" + (pkg / "lean-toolchain").read_bytes()).hexdigest()


def rel(frm: Path, to: Path) -> str:
    return os.path.relpath(to.resolve(), frm.resolve())


def lake_packages(root: Path) -> list[Path]:
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in A.SKIP_DIRS)
        if "lakefile.toml" in files and not Path(d).resolve().is_relative_to(A.FIXTURES.resolve()):
            out.append(Path(d))
    return out


def path_requires(pkg: Path) -> list[tuple[str, Path]]:
    return [(r["name"], (pkg / r["path"]).resolve()) for r in lakefile(pkg).get("require", []) if "path" in r]


class Refused(Exception):
    pass


def plan(pkg: Path, area: str, keep: list[str], root: Path = ROOT) -> dict:
    """The split of ``pkg``, computed from its sources, lakefile and lock; raises `Refused` with the reason."""
    pkg, root = pkg.resolve(), root.resolve()
    policy = json.loads((pkg / A.POLICY).read_text())
    files = A.lean_modules(pkg)
    imp = {m: imports(pkg / f) for m, f in files.items()}
    roots = policy["roots"]
    libs = lakefile(pkg).get("lean_lib", [])
    main_libs = [lib for lib in libs if set(lib.get("roots", [lib["name"]])) & set(roots)]
    aux_roots = [r for lib in libs if lib not in main_libs for r in lib.get("roots", [lib["name"]])]
    if missing := sorted(set(keep) - set(files)):
        raise Refused(f"--keep names modules {pkg} does not have: {', '.join(missing)}")
    prefixes = [f"{r}.{n}" for r in roots for n in SPEC_NAMES]
    seed = {m for m in files if A.under(prefixes, m)} | (set(policy.get("reads", {})) & set(files)) | set(keep)
    if bad := sorted(set(roots) & seed):
        raise Refused(f"a root is a spec module ({', '.join(bad)}): nothing separates the proofs from it")
    edges = sorted((a, b) for a in seed for b in imp[a] if b in files and b not in seed)
    if edges:
        raise Refused("spec modules import proof modules; drop the import, or keep the imported module in the spec "
                      "(--keep, its name kept):\n" + "\n".join(f"  {a} imports {b}" for a, b in edges))
    exempt = policy.get("exempt", {})
    spec_side = set(seed)
    changed = True
    while changed:
        changed = False
        for m in sorted(set(files) - spec_side):
            if (A.under(exempt, m) or A.under(aux_roots, m)) and all(i in spec_side for i in imp[m] if i in files):
                spec_side.add(m)
                changed = True
    proofs = sorted(set(files) - spec_side)
    if not proofs:
        raise Refused(f"{pkg} has no proofs to move")
    spec_tops = {m.split(".")[0] for m in spec_side}
    rename = {m: proof_name(m, spec_tops) for m in proofs if proof_name(m, spec_tops) != m}
    taken = set(files) | {m for _, d in path_requires(pkg) for m in A.lean_modules(d)}
    if clash := sorted({n.split(".")[0] for n in rename.values()} & {m.split(".")[0] for m in taken}):
        raise Refused(f"the proofs' root {', '.join(clash)} is already a root of {pkg} or of a package it requires")
    tops = sorted(m for m in seed if not any(p != m and A.under([p], m) for p in seed))
    reach, todo = set(), deque(tops)
    while todo:
        m = todo.popleft()
        if m not in reach:
            reach.add(m)
            todo.extend(i for i in imp[m] if i in files)
    spec_roots = tops + sorted(seed - reach)
    dest = root / "security_proofs" / area
    if dest.exists():
        raise Refused(f"{dest.relative_to(root)} exists")

    def side(mods: list[str]) -> set[str]:
        return {"spec" if m in spec_side else "proofs" for m in mods}

    proof_roots = [rename.get(r, r) for r in roots]
    lock_spec, lock_proofs = dict(policy), {"roots": proof_roots}
    lock_spec["roots"] = spec_roots
    lock_spec["proved_in"] = [rel(pkg, dest)]
    for key in ("exempt", "runs", "compile_time"):
        mine, theirs = {}, {}
        for k, v in policy.get(key, {}).items():
            s = side([m for m in files if A.under([k], m)] or [k])
            (mine if "spec" in s else {})[k] = v
            (theirs if "proofs" in s else {})[proof_name(k, spec_tops)] = v
        if key in policy:
            if mine:
                lock_spec[key] = mine
            else:
                del lock_spec[key]
        if theirs:
            lock_proofs[key] = theirs
    for key in ("axioms", "escapes", "dependency_unchecked"):
        if key in policy:
            lock_proofs[key] = policy[key]
    layers_spec, layers_proofs = {}, {}
    for k, v in policy.get("layers", {}).items():
        inside = [m for m in files if A.under([k], m)]
        if not inside or set(inside) & spec_side:
            layers_spec[k] = v
        if set(inside) - spec_side:
            layers_proofs[proof_name(k, spec_tops)] = v | {
                "imports": renamed_prefixes(v.get("imports", []), proofs, spec_side, spec_tops),
                "except": [rename.get(m, m) for m in v.get("except", [])]}
    if "layers" in policy:
        lock_spec["layers"] = layers_spec
        if layers_proofs:
            lock_proofs["layers"] = layers_proofs
    if up := policy.get("upstream"):
        mine, theirs = {}, {}
        for k, w in up.get("watch", {}).items():
            about = w.get("about", "").split(":")[0].split()[0] if w.get("about") else ""
            (theirs if A.under(proofs, about) or any(A.under([m], about) for m in proofs) else mine)[k] = w
        if mine:
            lock_spec["upstream"] = up | {"watch": mine}
        else:
            del lock_spec["upstream"]
        if theirs:
            lock_proofs["upstream"] = up | {"watch": theirs}
    if "dependencies" in policy:
        lock_proofs["dependencies"] = policy["dependencies"]
    lock_proofs["guarantees"] = {}

    moved_runs = [m for m in lock_proofs.get("runs", {})]
    extra = sorted({a for m in moved_runs for a in policy["runs"][m].get("generate", [])
                    if not a.startswith("{") and (pkg / a).is_file()})
    tracked = git_files(root)
    pkg_rel = pkg.relative_to(root).as_posix()
    moves = {f"{pkg_rel}/{files[m]}": f"{dest.relative_to(root).as_posix()}/"
             + ("/".join(rename[m].split(".")) + ".lean" if m in rename else files[m]) for m in proofs}
    moves |= {f"{pkg_rel}/{e}": f"{dest.relative_to(root).as_posix()}/{e}" for e in extra}
    moves = {a: b for a, b in sorted(moves.items()) if a in tracked}

    needs = {name: d for name, d in path_requires(pkg)}
    provided = {name: set(A.lean_modules(d)) for name, d in needs.items()}

    def required(mods) -> list[str]:
        return [n for n in needs if any(i in provided[n] for m in mods for i in imp[m])]

    spec_needs, proofs_needs = required(spec_side), required(proofs)
    dependents = []
    for d in lake_packages(root):
        if d.resolve() == pkg or not any(t == pkg for _, t in path_requires(d)):
            continue
        uses = sorted({i for f in A.lean_modules(d).values() for i in imports(d / f) if i in set(proofs)})
        if uses:
            dependents.append({"package": d.resolve().relative_to(root).as_posix(), "imports": uses})
    main_lib = main_libs[0]["name"] if main_libs else roots[0]
    return {"package": pkg_rel, "area": area, "dest": dest.relative_to(root).as_posix(),
            "name": f"security_proofs_{area}", "spec": sorted(spec_side), "seed": sorted(seed), "proofs": proofs,
            "keep": sorted(keep), "spec_roots": spec_roots, "proof_roots": proof_roots, "rename": rename, "moves": moves,
            "main_lib": main_lib, "proofs_lib": proof_name(main_lib, spec_tops | {main_lib}),
            "spec_tops": sorted(spec_tops), "spec_requires": spec_needs, "proofs_requires": proofs_needs,
            "dependents": dependents, "lock_spec": lock_spec, "lock_proofs": lock_proofs}


def proof_name(m: str, tops: set[str]) -> str:
    """A proof module's name in the proofs package: under a root namespace of its own (`P.SecurityProofs.X` as
    `PProofs.SecurityProofs.X`) when its root is also a root of the spec's modules, since Lean's search path gives every
    module under a root to the first package holding it. Its declarations keep their names."""
    r, dot, rest = m.partition(".")
    return f"{r}Proofs{dot}{rest}" if r in tops else m


def renamed_prefixes(prefixes: list[str], proofs, spec_side, tops: set[str]) -> list[str]:
    """A layer's `imports` for the modules after a split: a prefix that names proof modules also under their new root; one
    that names spec modules, or no module of the package, as it is."""
    out = []
    for p in prefixes:
        if any(A.under([p], m) for m in spec_side) or not any(A.under([p], m) for m in proofs):
            out.append(p)
        if any(A.under([p], m) for m in proofs) and proof_name(p, tops) not in out:
            out.append(proof_name(p, tops))
    return out


def git_files(root: Path) -> set[str]:
    r = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True)
    return set(r.stdout.split("\0")) - {""} if r.returncode == 0 else set()


TOML_BLOCK = re.compile(r"(?ms)^\[\[(\w+)\]\]\n.*?(?=^\[|\Z)")


def blocks(text: str) -> list[tuple[str, str, dict]]:
    """``text``'s `[[table]]` blocks as (table, text, parsed), in order (comments above a block belong to the one before)."""
    return [(m.group(1), m.group(0), tomllib.loads(m.group(0).replace(f"[[{m.group(1)}]]", "", 1))) for m in TOML_BLOCK.finditer(text)]


def spec_lakefile(text: str, p: dict) -> str:
    """The spec's lakefile: its main library given the spec's roots, and the path requires only the proofs used dropped."""
    out = text
    for table, block, cfg in blocks(text):
        if table == "lean_lib" and cfg.get("name") == p["main_lib"]:
            if "roots" in cfg:
                new = re.sub(r"(?m)^roots\s*=.*$", "roots = " + json.dumps(p["spec_roots"]), block)
            else:
                new = re.sub(r'(?m)^(name\s*=.*)$', lambda m: m.group(1) + "\nroots = " + json.dumps(p["spec_roots"]), block, 1)
            out = out.replace(block, new)
        if table == "require" and "path" in cfg and cfg["name"] not in p["spec_requires"]:
            below = re.search(r"(?:\n[ \t]*(?:#[^\n]*)?)*\Z", block.rstrip("\n"))
            out = out.replace(block, below.group(0).lstrip("\n") + "\n" if below.group(0).strip() else "")
    return out


def proofs_lakefile(spec_pkg: Path, dest: Path, p: dict) -> str:
    text = (spec_pkg / "lakefile.toml").read_text()
    cfg = tomllib.loads(text)
    head = [f'name = "{p["name"]}"', f'defaultTargets = {json.dumps([p["proofs_lib"]])}', ""]
    if opts := re.search(r"(?ms)^\[leanOptions\]\n.*?(?=^\[|\Z)", text):
        head.append(opts.group(0).rstrip() + "\n")
    reqs = [re.sub(r"(?:\n[ \t]*(?:#[^\n]*)?)+\Z", "", b.rstrip()) + "\n" for t, b, c in blocks(text)
            if t == "require" and "path" not in c]
    reqs.append(f'[[require]]\nname = "{cfg["name"]}"\npath = "{rel(dest, spec_pkg)}"\n')
    for name, d in path_requires(spec_pkg):
        if name in p["proofs_requires"] and name not in p["spec_requires"]:
            reqs.append(f'[[require]]\nname = "{name}"\npath = "{rel(dest, d)}"\n')
    libs = (f'# The proofs of `{p["package"]}`\'s guarantees and every lemma behind them, untrusted: the spec\'s lock says\n'
            f'# what they prove.\n[[lean_lib]]\nname = "{p["proofs_lib"]}"\nroots = {json.dumps(p["proof_roots"])}\n')
    return "\n".join(head) + "\n" + "\n".join(reqs) + "\n" + libs


def entry(name: str, d: str, inherited: bool) -> dict:
    return {"type": "path", "scope": "", "name": name, "manifestFile": "lake-manifest.json", "inherited": inherited, "dir": d,
            "configFile": "lakefile.toml"}


def with_path_deps(manifest: dict, direct: list[tuple[str, str, Path]], name: str | None = None) -> dict:
    """``manifest`` with its path packages exactly ``direct`` ([(name, dir from the package, its directory)]) and what they
    require by path, as Lake records them: the direct ones first, then each one's inherited, its dir composed through its
    parent's; the fetched packages as they were."""
    path = [entry(n, d, False) for n, d, _ in direct]
    for _, d, src in direct:
        sub = json.loads((src / "lake-manifest.json").read_text()) if (src / "lake-manifest.json").exists() else {"packages": []}
        path += [entry(e["name"], f"{d}/{e['dir']}", True) for e in sub["packages"] if e["type"] == "path"]
    seen, out = set(), []
    for e in [*path, *(e for e in manifest["packages"] if e["type"] != "path")]:
        if e["name"] not in seen:
            seen.add(e["name"])
            out.append(e)
    return manifest | ({"name": name} if name else {}) | {"packages": out}


def lake_json(v, indent: int = 0, col: int = 0, trail: int = 0, width: int = 120) -> str:
    """``v`` as Lean's `Json.pretty` lays it out (how Lake writes a manifest): a bracket on one line when it fits, else one
    item per line nested one deeper than its parent; a key whose value does not fit puts the value on the next line."""
    flat = json.dumps(v, ensure_ascii=False, separators=(", ", ": "))
    if not isinstance(v, (dict, list)) or not v or col + len(flat) + trail <= width:
        return flat
    pad, items = "\n" + " " * (indent + 1), list(v.items()) if isinstance(v, dict) else list(v)
    out, c = [], col + 1
    for i, it in enumerate(items):
        t = 1 if i < len(items) - 1 else 1 + trail
        if isinstance(v, dict):
            k = json.dumps(it[0], ensure_ascii=False) + ":"
            one = json.dumps(it[1], ensure_ascii=False, separators=(", ", ": "))
            if c + len(k) + 1 + len(one) + t <= width:
                s = f"{k} {one}"
            else:
                s = k + pad + lake_json(it[1], indent + 1, indent + 1, t, width)
        else:
            s = lake_json(it, indent + 1, c, t, width)
        out.append(s)
        c = indent + 1
    return ("{" if isinstance(v, dict) else "[") + ("," + pad).join(out) + ("}" if isinstance(v, dict) else "]")


def write_manifest(path: Path, doc: dict) -> None:
    path.write_text(lake_json(doc) + "\n")


def write_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def set_path_deps(pkg: Path) -> None:
    """Rewrite ``pkg``'s manifest for the path requires its lakefile now has (unchanged when they are the same)."""
    m = json.loads((pkg / "lake-manifest.json").read_text())
    new = with_path_deps(m, [(n, rel(pkg, d), d) for n, d in path_requires(pkg)])
    if new != m:
        write_manifest(pkg / "lake-manifest.json", new)


def rekey(pins: dict, old: str | None, new: str | None, package: str, keep_old: bool) -> list[str]:
    """Pin ``package``'s new manifest (``new``) to the bundle ``old`` pinned: the fetched packages are the same."""
    if old == new:
        return []
    if old is None or new is None or old not in pins:
        return [f"lean-deps: no bundle pinned for {package}'s manifest; its first check builds its dependencies cold"]
    pins[new] = pins[old] | {"package": package}
    if not keep_old:
        del pins[old]
    return [f"lean-deps: {package}: {new[:12]} pinned to the bundle of {old[:12]} ({pins[new]['sha256'][:12]})"]


def apply(p: dict, root: Path = ROOT) -> list[str]:
    """Carry out ``p`` in the checkout at ``root``; returns what it did and what a person still reads."""
    root = root.resolve()
    pkg, dest = root / p["package"], root / p["dest"]
    notes = []
    dependents = [root / d["package"] for d in p["dependents"]]
    old_keys = {d: deps_key(d) for d in [pkg, *dependents]}
    spec_name = lakefile(pkg)["name"]
    for a, b in p["moves"].items():
        (root / b).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "mv", a, b], cwd=root, check=True)
    (dest / "lean-toolchain").write_text((pkg / "lean-toolchain").read_text())
    (dest / "lakefile.toml").write_text(proofs_lakefile(pkg, dest, p))
    direct = [(spec_name, rel(dest, pkg), pkg)]
    direct += [(n, rel(dest, d), d) for n, d in path_requires(pkg) if n in p["proofs_requires"] and n not in p["spec_requires"]]
    write_manifest(dest / "lake-manifest.json",
                   with_path_deps(json.loads((pkg / "lake-manifest.json").read_text()), direct, p["name"]))
    A.dump(dest, p["lock_proofs"])
    (pkg / "lakefile.toml").write_text(spec_lakefile((pkg / "lakefile.toml").read_text(), p))
    set_path_deps(pkg)
    A.dump(pkg, p["lock_spec"])
    for dep, d in zip(dependents, p["dependents"]):
        lf = (dep / "lakefile.toml").read_text()
        block = next(b for t, b, c in blocks(lf) if t == "require" and c.get("name") == spec_name)
        own = re.match(r"(?s)(.*?\n)((?:[ \t]*(?:#[^\n]*)?\n)*)\Z", block)
        lf = lf.replace(block, own[1] + f'\n[[require]]\nname = "{p["name"]}"\npath = "{rel(dep, dest)}"\n' + own[2], 1)
        (dep / "lakefile.toml").write_text(re.sub(r"\n{3,}", "\n\n", lf))
        set_path_deps(dep)
        notes.append(f"dependent: {d['package']} requires {p['dest']} (it imports {', '.join(d['imports'])})")
    pins_path = root / LEAN_DEPS
    if pins_path.exists():
        doc = json.loads(pins_path.read_text())
        pins = doc["deps"]
        notes += rekey(pins, old_keys[pkg], deps_key(dest), p["dest"], keep_old=True)
        notes += rekey(pins, old_keys[pkg], deps_key(pkg), p["package"], keep_old=False)
        for dep, d in zip(dependents, p["dependents"]):
            notes += rekey(pins, old_keys[dep], deps_key(dep), d["package"], keep_old=False)
        doc["deps"] = dict(sorted(pins.items()))
        write_json(pins_path, doc)
    notes += rewrite_imports(p, root)
    notes += relayer(p, root)
    notes += rewrite_paths(p, root)
    notes += stale_hashes(pkg, root)
    rewritten = [n.split()[1] for n in notes if n.startswith("rewrote: ")]
    subprocess.run(["git", "add", "-A", "--", p["dest"], p["package"], *(d["package"] for d in p["dependents"]), *rewritten] +
                   ([LEAN_DEPS] if pins_path.exists() else []), cwd=root, check=True, capture_output=True)
    return notes


def rewrite_imports(p: dict, root: Path) -> list[str]:
    """Every tracked Lean file's imports of a renamed proof module, under its new name (the header only)."""
    if not p["rename"]:
        return []
    notes = []
    for f in sorted(git_files(root)):
        path = root / f
        if not f.endswith(".lean") or not path.is_file():
            continue
        lines, count = path.read_text().split("\n"), 0
        code = A.strip_comments("\n".join(lines)).split("\n")
        for i, s in enumerate(code):
            s = s.strip()
            if not s or s in ("module", "prelude"):
                continue
            if not IMPORT.match(s):
                break
            new = re.sub(r"(?<![\w.'])([\w.'\u0080-\uffff]+)", lambda m: p["rename"].get(m.group(1), m.group(1)), lines[i])
            count += new != lines[i]
            lines[i] = new
        if count:
            path.write_text("\n".join(lines))
            notes.append(f"rewrote: {f} ({count} import(s))")
    return notes


def relayer(p: dict, root: Path) -> list[str]:
    """Each other lock's `layers` that let a module import the moved proofs, also under their new root."""
    pkg, dest = (root / p["package"]).resolve(), (root / p["dest"]).resolve()
    spec, proofs, tops = set(p["spec"]), set(p["proofs"]), set(p["spec_tops"])
    notes = []
    for d in lake_packages(root):
        if d.resolve() in (pkg, dest) or not (d / A.POLICY).is_file():
            continue
        policy = json.loads((d / A.POLICY).read_text())
        layers = {k: v | {"imports": renamed_prefixes(v.get("imports", []), proofs, spec, tops)}
                  for k, v in policy.get("layers", {}).items()}
        if layers != policy.get("layers", {}):
            A.dump(d, policy | {"layers": layers})
            notes.append(f"rewrote: {(d / A.POLICY).relative_to(root).as_posix()} (layers that import the moved proofs)")
    return notes


def moved_dirs(p: dict, tracked: set[str]) -> dict[str, str]:
    """{directory: its new path} for each directory under the package all of whose tracked files move."""
    out = {}
    for a, b in p["moves"].items():
        d = a.rsplit("/", 1)[0]
        while d != p["package"] and d.startswith(p["package"] + "/"):
            if all(f in p["moves"] for f in tracked if f.startswith(d + "/")):
                out[d] = b[:len(b) - len(a) + len(d)]
            d = d.rsplit("/", 1)[0]
    return out


def rewrite_paths(p: dict, root: Path) -> list[str]:
    """Every tracked text file's mention of a moved file or directory by its repository path, or by its path from a
    directory above the package that holds the file (`lean/P/SecurityProofs/X.lean` in `protocols/p/docs/A.md`), rewritten
    to the repository path; and, in the package's area (the package's parent directory), mentions of one by its path inside
    the package, listed for a person (a layout table, a path built from parts)."""
    tracked = git_files(root)
    dirs = moved_dirs(p, tracked)
    subs = dict(sorted({**p["moves"], **dirs}.items(), key=lambda kv: -len(kv[0])))
    pkg, dest = p["package"], p["dest"]
    area = pkg.rsplit("/", 1)[0] if "/" in pkg else ""

    def mention(prefix: str) -> re.Pattern | None:
        n = len(prefix)
        alts = [re.escape(a[n:]) for a in subs if a.startswith(prefix)]
        before = r"(?<![\w./-])" if prefix else r"(?<![\w.-])"
        return re.compile(before + "(?:" + "|".join(alts) + r")(?![\w.-])") if alts else None

    pats = {"": mention("")}
    above = [pkg[:i + 1] for i in range(len(pkg) - 1, 0, -1) if pkg[i] == "/"]
    n = len(pkg) + 1
    inner = sorted({a[n:] for a in subs}, key=len, reverse=True)
    parts = [r"""["']\s*/\s*["']""".join(re.escape(x) for x in i.split("/")) for i in inner if "/" in i]
    loose = re.compile("|".join([*(re.escape(i) for i in inner), *parts]))
    notes, residual = [], []
    for f in sorted(tracked):
        path = root / f
        if not path.is_file() or f.endswith("lean-audit.json") or f == LEAN_DEPS or path.stat().st_size > 2 << 20:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        prefixes = [""] + ([] if f.startswith((pkg + "/", dest + "/")) else
                           [a for a in above if f.startswith(a)])
        new, count = text, 0
        for prefix in prefixes:
            pat = pats.setdefault(prefix, mention(prefix))
            if pat:
                new, k = pat.subn(lambda m: subs[prefix + m.group(0)], new)
                count += k
        if new != text:
            path.write_text(new)
            notes.append(f"rewrote: {f} ({count} path(s))")
        if (area and not f.startswith(area + "/")) or f.startswith(dest + "/") or f.endswith(".lean"):
            continue
        for i, line in enumerate(new.splitlines(), 1):
            if loose.search(line) and not any(b in line for b in subs.values()):
                residual.append(f"read: {f}:{i}: {line.strip()[:150]}")
    return notes + residual


HASHED = re.compile(r"^([0-9a-f]{64}) [ *](.+)$", re.M)


def stale_hashes(pkg: Path, root: Path) -> list[str]:
    """Each line of a `sha256sum` list in the package (`TRUSTED.sha256`) whose file the split moved or changed: a trust pin,
    regenerated by whoever owns it, never by the split."""
    out = []
    for f in sorted(pkg.rglob("*.sha256")):
        if ".lake" in f.relative_to(pkg).parts:
            continue
        for digest, name in HASHED.findall(f.read_text()):
            target = f.parent / name
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                out.append(f"read: {rel(root, f)}: {name} {'moved' if not target.is_file() else 'changed'}; regenerate the list")
    return out


def describe(p: dict) -> str:
    lines = [f"split {p['package']} -> spec (stays) + {p['dest']} (package {p['name']})",
             f"  spec: {len(p['spec'])} modules ({len(p['seed'])} the lock trusts, {len(p['spec']) - len(p['seed'])} tools "
             f"that import only them){'; kept under proof names: ' + ', '.join(p['keep']) if p['keep'] else ''}",
             f"  spec roots: {', '.join(p['spec_roots'])}",
             f"  proofs: {len(p['proofs'])} modules, roots {', '.join(p['proof_roots'])}, library {p['proofs_lib']}; "
             f"{len(p['rename'])} renamed under their own root; {len(p['moves'])} files move",
             f"  requires: spec {p['spec_requires'] or '(no path packages)'}; proofs: "
             + ", ".join(["the spec", *(r for r in p["proofs_requires"] if r not in p["spec_requires"])]),
             f"  lock: guarantees {len(p['lock_spec'].get('guarantees', p['lock_spec'].get('pins', {})))} and reads {len(p['lock_spec'].get('reads', {}))} stay byte for "
             f"byte; to the proofs: " + (", ".join(f"{k} {list(v) if isinstance(v, dict) and k != 'dependencies' else ''}".strip()
                                                    for k, v in p["lock_proofs"].items() if k != "guarantees") or "nothing")]
    lines += [f"  dependent: {d['package']} imports {', '.join(d['imports'])}: gains a require of {p['dest']}"
              for d in p["dependents"]]
    lines += [f"  mv {a} -> {b}" for a, b in p["moves"].items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", type=Path)
    ap.add_argument("--area", required=True, help="the proofs package's directory under security_proofs/")
    ap.add_argument("--keep", nargs="*", default=[], help="modules under a proof's name that are the spec's (names kept)")
    ap.add_argument("--apply", action="store_true", help="move the files and write the configuration (else print the plan)")
    ap.add_argument("--json", action="store_true", help="print the plan as JSON")
    ap.add_argument("--root", type=Path, default=ROOT, help="the checkout (default: this one)")
    a = ap.parse_args(argv)
    pkg = a.package if a.package.is_absolute() else (Path.cwd() / a.package)
    try:
        p = plan(pkg, a.area, a.keep, a.root)
    except Refused as e:
        print(f"split: refused: {e}", file=sys.stderr)
        return 1
    print(json.dumps(p, indent=1, ensure_ascii=False) if a.json else describe(p))
    if a.apply:
        for n in apply(p, a.root):
            print(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
