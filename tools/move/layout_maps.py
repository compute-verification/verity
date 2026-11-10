#!/usr/bin/env python3
"""`tools/move/layout_maps.py [--ref REF] [--targets FILE]`: write the layout's maps, one `layout.py` map a move.

    python3 tools/move/layout_maps.py            # regenerates every tools/move/layout_NN_NAME_map.toml on HEAD

It reads `layout_maps.json` beside it (the layout's assignment at one commit: each moved component's new directory and
files, the files archived and the files that stay) and the files REF tracks, and writes one map a move, numbered in
the order they run. It writes nothing unless every check below passes, so it is the one rerun when `main` moves: a
tracked file it cannot place stops it, by name. Then each map runs in the order of its name:

    for m in tools/move/layout_??_*_map.toml; do python3 tools/move/layout.py --map $m --dry-run; done

THE RULE for a path inside a directory. The layout gives each component a directory, which ends in `verifier/`,
`prover/` or `device/` where the component is that part. A moved file's new path is

    its component's new directory / the name of the directory it is in today / its path below that directory

where "the directory it is in today" is the one a person names it by (`named`: the Python package, Lake package or
benchmark directory), and that name is left out where the new directory, less its part, already ends in it
(`verity/core/protocols/pouw/audit.py` goes to `verity/protocols/pouw/verifier/audit.py`, not
`.../verifier/pouw/audit.py`). So a component keeps its own subpath, nothing is renamed, and no new path keeps a `core`
or `ml` segment. An archived file goes to `archive/` under its whole path today.

A module's circuit-check sidecar and its pins (`X.circuit-check.py`, `X.circuit-check.json`) go where `X.py` goes,
whichever component they are in: circuit-check finds a module's bindings beside it.

EXCEPTIONS, each a row of `EXCEPT` or `KEEP` with its reason: where the rule gives two files one path, puts a file
nobody trusts in a trusted directory, takes a module out of its import package, or gives a path nobody would look for.
A row the rule itself now gives is reported as doing nothing.

A FILE THE DATA DOESN'T KNOW (tracked at REF, added since the data was cut) goes with the files beside it: a directory
that goes one way takes it along; in a directory several components share it goes with the one component that holds
the other files of that directory, and where that isn't one component the script stops and names it, for a row in
`NEW`. A path of the data that REF no longer tracks is dropped and reported.

KINDS, each treated one way:
- Python packages. A whole directory is one directory row; a directory several moves share gets one row a file (or a
  subdirectory). `[modules]` is what `layout.py` derives from `[paths]` and `[roots]`, so imports, dotted names and
  `-m` names follow the files. `layout.py` adds an empty `__init__.py` for each new package level a moved module needs
  and no moved file brings (its "files created").
- Lake packages. Each moves whole, by a directory row at or above its root or a `[tokens]` row for it, so a lakefile's
  or a manifest's relative `path` to it follows. Lean module names don't change.
- Distributions with their own `pyproject.toml`. The pyproject moves with its component and the distribution keeps its
  name and import name (`ROOTS`). A directory several maps share is named in `[tokens]` by the one map that takes its
  `pyproject.toml` (else its `__init__.py`, its `lakefile.toml`, else most of its files), so a reference to the
  directory follows that map.
- Tests go with the component they belong to, by the same rule; plain files likewise.

ORDER. Each map names today's paths only, no map has a row at or under another map's row, and no destination lies
under any map's row, so the files land the same in any order. Module names do not: a map that moves a package's
`__init__.py` renames the package, and would rename with it the modules another map takes out of that package. So that
other map runs first: the order is derived from the maps, their names carry it, and each map's header names the maps
that run before it. `[retired]` is empty in every map: a directory two maps share can be retired by neither while
each map passes alone, and this script's own check does that job.

CHECKS (the script fails on any): every tracked file is one the data places or `NEW` names; every file the layout
moves or archives, but for `LEFT`'s, has exactly one destination over all the maps, and each map moves exactly its own
files; no destination is claimed twice or is a file that stays; no new path outside `archive/` keeps a `core` or `ml`
segment; every tracked file under `verity/core` or `verity/ml` moves, but for `LEFT`'s; no file of a component the
auditor doesn't trust lands under a trusted directory, but for `UNTRUSTED`'s; every moved Lake package moves whole;
`[modules]` gives every moved module its new name; the order has no cycle.
"""
import argparse
import collections
import json
import os
import posixpath
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import layout as L  # noqa: E402

#: (today's path: a file, or a directory ending in "/"; its new path; why the rule is not followed)
EXCEPT = [
    ("verity/ml/flock/verifier/", "verity/flock/agreement/",
     "the agreement with upstream's verifier, which nobody trusts: the rule puts its scripts in "
     "`verity/flock/verifier/`, the trusted root of the Lean verifier's Lake package, and its README on that package's"),
    ("verity/ml/flock/python/verity_flock/", "verity/flock/prover/python/verity_flock/",
     "a module goes with its import package: eleven bench drivers are modules of `verity_flock`, which the rule would "
     "leave in a second `verity/flock/python/verity_flock/` with no `__init__.py`, outside the distribution that keeps "
     "the name"),
    ("verity/core/protocols/certifier/lean/", "verity/protocols/certifier/device/program/lean/",
     "the certifier program's Lake package: the rule puts it on `device/lean/`, where the certifier device's Lake "
     "package goes (two lakefiles, manifests, locks and toolchains on one path)"),
    ("verity/core/service/spot/", "verity/spot/verifier/lean/",
     "SPOT's Lake package: the rule puts its `pyproject.toml` on the Python verifier's in `verity/spot/verifier/`; the "
     "Lean takes `lean/`, as each protocol's Lean is its `lean/`"),
    ("verity/core/service/vstar/recursive/", "verity/spot/verifier/recursive/",
     "the recursive audit's Lake package requires SPOT's: under Flock's verifier, where the rule leaves it, it reads "
     "against the order of the layout"),
    ("verity/core/service/vstar/tests/test_vstar_recursive.py", "verity/spot/verifier/recursive/tests/test_vstar_recursive.py",
     "the recursive audit's test goes with its package"),
    ("verity/core/service/vstar/tests/recursive_data.json", "verity/spot/verifier/recursive/tests/recursive_data.json",
     "the recursive audit's test data goes with its test"),
    ("benchmarks/pouw/machine_model/", "verity/hardware/machine_model/",
     "the machine model: the rule gives `verity/hardware/pouw/machine_model/`, where `pouw/` names only the benchmark "
     "directory it sits in today"),
    ("verity/core/protocols/record.py", "verity/hardware/record.py",
     "the rule gives `verity/hardware/protocols/record.py`: a `protocols/` segment that names nothing, and a package "
     "whose `__init__.py` no moved file brings"),
    ("verity/core/protocols/registration.py", "verity/hardware/registration.py",
     "as `record.py`: the rule gives `verity/hardware/protocols/registration.py`"),
    ("verity/core/protocols/profile.py", "verity/spot/verifier/profile.py",
     "the integrity profile: the rule gives `verity/spot/verifier/protocols/profile.py`, a `protocols/` segment that "
     "names nothing"),
    ("verity/core/protocols/pyproject.toml", "verity/protocols/pyproject.toml",
     "the suite of the protocols' own modules goes where the one module left of them, `reporting.py`, goes; the rule "
     "gives `verity/spot/verifier/protocols/pyproject.toml`"),
    ("verity/ml/commitments/leaves.py", "integrations/vllm/verity_vllm/commitments/leaves.py",
     "a module goes into an import package: the rule gives `integrations/vllm/commitments/`, beside `verity_vllm` and "
     "in no package, where it would be imported as `integrations.vllm.commitments.leaves`"),
    ("verity/ml/commitments/vllm_v1/", "integrations/vllm/verity_vllm/commitments/vllm_v1/",
     "as `leaves.py`: the commitment scheme named for vLLM goes into `verity_vllm`"),
    ("verity/ml/commitments/tests/test_leaves.py", "integrations/vllm/tests/commitments/test_leaves.py",
     "the test of `leaves.py` goes into the integration's suite"),
    ("verity/ml/commitments/tests/test_vllm_v1.py", "integrations/vllm/tests/commitments/test_vllm_v1.py",
     "the test of `vllm_v1` goes into the integration's suite"),
    ("verity/core/devices/memory_challenger/lean/scripts/replay.py",
     "verity/protocols/pous/device/memory_challenger/lean/scripts/replay.py",
     "an archived file inside the memory challenger's Lake package, which moves whole by one row"),
    ("examples/README.md", "workloads/README.md",
     "the demos' index: the rule gives `workloads/dither_full_util/examples/README.md`, inside one demo"),
    ("verity/ml/circuits/tests/test_settings.py", "verity/circuits/tests/test_settings.py",
     "the one file the layout keeps (it tests `verity/settings.py`) whose directory moves away from under it: it goes "
     "with the suite that runs it"),
]
#: an EXCEPT row whose files go with another move than their component's: {the row's path today: the move}
EXCEPT_MOVE = {
    "verity/ml/flock/python/verity_flock/": "flock_prover",
    "verity/core/service/vstar/recursive/": "spot",
    "verity/core/service/vstar/tests/test_vstar_recursive.py": "spot",
    "verity/core/service/vstar/tests/recursive_data.json": "spot",
    "verity/core/protocols/pyproject.toml": "verity",
    "verity/core/devices/memory_challenger/lean/scripts/replay.py": "pous",
    "verity/ml/circuits/tests/test_settings.py": "circuits",
}
#: components the layout gives a directory that stay where they are: {component id: why}
KEEP = {
    "bench:31": "the sampling-plan calculators and figures stay in `benchmarks/`: moved under `verity/`, two of their "
                "scripts would read `benchmarks/census` by path",
}
#: tracked files under `verity/core` or `verity/ml` that no map moves, each with why. Deleting them is an edit, which a
#: move cannot make: one commit after the moves.
LEFT = {
    "verity/core/__init__.py": "the `__init__.py` of a package level the moves empty (the layout keeps it)",
    "verity/ml/__init__.py": "the `__init__.py` of a package level the moves empty (the layout keeps it)",
    "verity/ml/protocols/__init__.py": "the `__init__.py` of a package level the moves empty (the layout keeps it)",
    "verity/core/protocols/__init__.py":
        "the `__init__.py` of a package level the moves empty. A row for it renames the package `verity.core.protocols`, "
        "and with it every protocol's modules in every map that runs later; in a map that runs last it lands on the "
        "`verity/protocols/__init__.py` that `layout.py` created for the first protocol to move",
    "verity/core/devices/__init__.py":
        "the `__init__.py` of a package level the moves empty: by the rule it goes to "
        "`verity/protocols/pous/device/devices/`, which renames the package `verity.core.devices` for the certifier's too",
}
#: files of a component the auditor doesn't trust that land under a trusted directory: {component id: why it may}
UNTRUSTED = {
    "core:zk-server": "the ZK server's release side, which the developer trusts and the auditor doesn't: the layout puts it "
                      "beside Flock's verifier, whose Lake package holds the ZK server's other half (`Flock.Firewall`)",
    "edge:tools.config": "the settings package (`verity_config`), which the layout puts with the Configuration it reads",
}
#: tracked files the data doesn't know, in a directory several components share: {path, or directory ending in "/":
#: the component id it goes with, "stays" or "archived"}
NEW = {
    "verity/ml/commitments/history.py": "ml:commitment-schemes",
    "verity/ml/commitments/tests/test_history.py": "ml:commitment-schemes",
}
#: references to a directory a map splits that stay right as they are: (the move, "old file: old directory", why)
SPLIT = [
    ("flock_prover", f"verity/ml/flock/live/src/{name}.rs: verity",
     "`../../../..` from the crate's directory (`CARGO_MANIFEST_DIR`) is the repository's root, before the move and "
     "after it: the crate is as deep in `verity/flock/prover/live` as in `verity/ml/flock/live`")
    for name in ("circuit", "firewall", "lookup", "sha512_native")
]
#: import name -> its package directory today; a module outside every root is named by its path from the root
ROOTS = {
    "verity": "verity", "verity_difftest": "verity/core/difftest/python/verity_difftest",
    "verity_flock": "verity/ml/flock/python/verity_flock", "verity_numerical": "benchmarks/numerical/python/verity_numerical",
    "verity_vllm": "integrations/vllm/verity_vllm", "research": "tools/research/src/research",
    "circuit_check": "tools/verity/circuit_check/src/circuit_check", "cluster": "infra/cluster/src/cluster",
    "verity_cli": "tools/verity/cli/verity_cli", "verity_config": "tools/config/verity_config",
    "verity_machine_model": "benchmarks/pouw/machine_model/verity_machine_model", "datacenter": "benchmarks/datacenter/datacenter",
}
#: the moves, in the order asked for where nothing else decides: (map name, what it is)
MOVES = [
    ("circuits", "the circuit language, to `verity/circuits`"),
    ("definitions", "the Definitions and their host kernels, to `verity/definitions`"),
    ("flock_verifier", "Flock's Lean package, to `verity/flock/verifier`"),
    ("flock_prover", "Flock's prover, to `verity/flock/prover`"),
    ("flock", "Flock's tests, pod scripts and agreement with upstream, to `verity/flock`"),
    ("spot", "sampled proofs, to `verity/spot`"),
    ("pouw", "PoUW, to `verity/protocols/pouw`"),
    ("pous", "PoUS, to `verity/protocols/pous`"),
    ("poub", "PoUB, to `verity/protocols/poub`"),
    ("certifier", "network certification, to `verity/protocols/certifier`"),
    ("inference_only", "the inference-only policy, to `verity/protocols/inference_only`"),
    ("composite", "the composite protocols, to `verity/composite`"),
    ("commitments", "commitments, hashes, signatures and the challenge derivation, to `verity/commitments`"),
    ("hardware", "the machine model, Site and registration, to `verity/hardware`"),
    ("configuration", "the Configuration, the wire and the settings package, to `verity/configuration`"),
    ("windows", "Window, to `verity/windows`"),
    ("difftest", "the difftest harness, to `verity/difftest`"),
    ("verity", "the claims' ids and the reporting grid, to `verity`"),
    ("vllm", "the commitment scheme named for vLLM and the window-advice compiler, to `integrations/vllm`"),
    ("workloads", "the demos and the recursion end-to-end run, to `workloads`"),
    ("benchmarks", "the GPU probes that are Definitions today, to `benchmarks`"),
    ("archive", "what the layout archives, to `archive`"),
]
EXCLUDE = [
    "fixtures/**", "**/fixtures/**", "**/*vectors*.json", "**/*vectors*.txt.gz", "**/CHANGELOG.md", "*.patch",
    "**/TRUSTED.sha256", "**/lake-manifest.json.lock", "uv.lock", "tools/move/**", "archive/**",
    "benchmarks/numerical/reports/**", "benchmarks/numerical/tests/bench/data/**", "**/price_twins.json",
    "verity/ml/circuits/silicon/captures/**", "verity/ml/circuits/silicon/hawkeye/**",
    "verity/definitions/silicon/captures/**", "verity/definitions/silicon/hawkeye/**",
    "integrations/vllm/manifests/**", "integrations/vllm/workloads/**", "integrations/vllm/data/**",
    "integrations/vllm/verity_vllm/pins/**", "tools/verity/check/tests/test_suites.py",
    "tools/verity/check/tests/test_move_check.py", "tools/research/src/research/store/moves.json",
    "benchmarks/tc_probe_fp4/fp4_emulation_1ad1aaa2.py", "benchmarks/network_e2e/tests/data/**",
    "benchmarks/hash/dither/j10b_ref.py", "benchmarks/hash/dither/j10ca_ref.py", "benchmarks/hash/dither/j10hb.py",
    "benchmarks/hash/dither/j10rtx.py", "benchmarks/hash/dither/j10served.py",
    "integrations/vllm/tools/migrate/superseded_quarantine.py", "integrations/vllm/tests/migrate/test_superseded_quarantine.py",
]
INCLUDE = ["fixtures/artifacts.json", "**/fixtures/README.md", "**/fixtures/**/README.md", "**/lake-manifest.json",
           "integrations/vllm/manifests/definitions.json"]
NAMESAKE = ("pyproject.toml", "__init__.py", "lakefile.toml")
MAP_NAME = re.compile(r"layout_\d\d_\w+_map\.toml")
SIDECAR = re.compile(r"(.+)\.circuit-check\.(?:py|json)")


def die(msg):
    raise SystemExit("layout_maps: " + msg)


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if r.returncode:
        die(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def named(p):
    """The directory a person names `p` by today: its Python package, Lake package or benchmark directory."""
    parts = p.split("/")
    stem, parts = parts[-1], parts[:-1]
    if not parts:
        return "."
    t = parts[0]

    def take(n):
        return "/".join(parts[:n])
    if t == "Security":
        if len(parts) == 1:
            return "Security"
        if parts[1] in ("Definitions", "Proofs", "Properties"):
            if len(parts) >= 3 and parts[1] == "Proofs" and parts[2] == "Properties":
                if len(parts) >= 4:
                    return take(4)
                return take(3) + "/" + stem[:-5] if stem.endswith(".lean") else take(3)
            if len(parts) >= 3:
                return take(3)
            return take(2) + "/" + stem[:-5] if stem.endswith(".lean") else take(2)
        return take(2)
    if t == "verity":
        if len(parts) >= 3 and parts[1] == "core" and parts[2] in ("protocols", "service", "primitives", "devices"):
            if len(parts) >= 5 and parts[2] == "protocols" and parts[3] == "network" and parts[4] in ("nci", "lean"):
                return take(5)
            return take(4) if len(parts) >= 4 else take(3)
        if len(parts) >= 3 and parts[1] == "ml" and parts[2] == "protocols":
            return take(4) if len(parts) >= 4 else take(3)
        if len(parts) >= 4 and parts[1] == "ml" and parts[2] == "circuits" and parts[3] in ("ops", "silicon"):
            return take(4)
        return take(3) if len(parts) >= 3 else take(len(parts))
    if t == "integrations":
        if len(parts) >= 4 and parts[2] == "verity_vllm":
            return take(4)
        return take(3) if len(parts) >= 3 else take(len(parts))
    if t == "tools":
        if len(parts) >= 3 and parts[1] == "verity":
            return take(3)
        return take(2) if len(parts) >= 2 else "tools"
    if t in ("benchmarks", "examples", "infra", "fixtures"):
        return take(2) if len(parts) >= 2 else t
    return t


def common_dir(paths):
    """The deepest directory that holds every one of `paths`."""
    out = []
    for segs in zip(*(p.split("/")[:-1] for p in paths)):
        if len(set(segs)) != 1:
            break
        out.append(segs[0])
    return "/".join(out)


def move_of(cdir):
    """The move (map) that fills a component's new directory."""
    s = cdir.strip("/").split("/")
    if s[0] == "verity":
        if len(s) == 1:
            return "verity"
        if s[1] == "protocols":
            return s[2]
        if s[1] == "flock":
            return "flock" if len(s) == 2 else f"flock_{s[2]}"
        return s[1]
    return s[1] if s[0] == "integrations" else s[0]


def holds(row, p):
    return p == row or (row.endswith("/") and p.startswith(row))


def excepted(p):
    """`p`'s EXCEPT row (the longest that holds it), its new path by it and that row's move if it names one, or Nones."""
    best = None
    for old, new, _ in EXCEPT:
        if holds(old, p) and (best is None or len(old) > len(best[0])):
            best = (old, new)
    return (None, None, None) if best is None else (best[0], best[1] + p[len(best[0]):], EXCEPT_MOVE.get(best[0]))


def place(data, tracked, say):
    """{tracked file: a component's id, "stays" or "archived"}: by the data's paths, `NEW`, or the files beside it."""
    entry = {}
    for key, paths in [(c["id"], c["paths"]) for c in data["components"]] + [("archived", data["archived"]), ("stays", data["stays"])]:
        for p in paths:
            if p in entry:
                die(f"layout_maps.json names {p} twice: for {entry[p]} and for {key}")
            entry[p] = key
    known = {c["id"] for c in data["components"]} | {"stays", "archived"}
    for p, key in NEW.items():
        if key not in known:
            die(f"NEW: {p} goes with {key}, which is no component of layout_maps.json")
        entry[p] = key
    used, owner, unknown = set(), {}, []
    for p in sorted(tracked):
        e, d = (p if p in entry else None), posixpath.dirname(p)
        while e is None and d:
            e, d = (d + "/" if d + "/" in entry else None), posixpath.dirname(d)
        if e is None:
            unknown.append(p)
        else:
            used.add(e)
            owner[p] = entry[e]
    beside = collections.defaultdict(set)              # directory -> who holds the data's files directly in it
    for p, key in owner.items():
        if p in entry and p not in NEW:
            beside[posixpath.dirname(p)].add(key)
    stuck = []
    for p in unknown:
        keys = beside.get(posixpath.dirname(p), set())
        if len(keys) == 1:
            owner[p] = next(iter(keys))
            say(f"new since the data was cut: {p} goes with the other files of its directory ({owner[p]})")
        else:
            stuck.append(f"{p}: " + ("its directory holds no file the data knows" if not keys else
                                     "its directory's files are " + ", ".join(sorted(keys))))
    if stuck:
        die("tracked files the data doesn't place, each in a directory several components share: give each a row in NEW "
            "(a component's id, \"stays\" or \"archived\"), or one row for its directory\n  " + "\n  ".join(stuck))
    for e in sorted(set(entry) - used):
        say(f"dropped, {e} is no longer tracked: a path of {entry[e]}")
    return owner


def plan(data, owner, say):
    """{today's path: (new path, component id, move)} for every file a map moves."""
    comps = {c["id"]: c for c in data["components"]}
    parts = set(data["parts"])
    out, plain = {}, {}
    for p, key in owner.items():
        row, new, mv = excepted(p)
        if key == "stays" or key in KEEP:
            rule, move = None, None
        elif key == "archived":
            rule, move = "archive/" + p, "archive"
        else:
            c = comps[key]
            cdir = c["to"]
            if c["part"] and not cdir.endswith("/" + c["part"]):
                die(f"{key}: its part is {c['part']} and its directory {cdir} does not end in it")
            segs = cdir.split("/")
            while len(segs) > 1 and segs[-1] in parts:
                segs = segs[:-1]
            nd = named(p)
            if not (nd == "." or p.startswith(nd + "/")):
                die(f"{p}: not under the directory it is named by, {nd}")
            below = p if nd == "." else p[len(nd) + 1:]
            base = nd.split("/")[-1]
            rule, move = f"{cdir}/{below if nd == '.' or base == segs[-1] else base + '/' + below}", move_of(cdir)
        if row is not None:
            plain.setdefault(row, []).append(rule == new)
            if (mv or move) is None:
                die(f"{p}: the layout keeps it, so its EXCEPT row needs a move in EXCEPT_MOVE")
            out[p] = (new, key, mv or move)
        elif rule is not None and p not in LEFT:
            out[p] = (rule, key, move)
    for p in sorted(owner):                           # a module's circuit-check sidecar and its pins go where the module goes
        if (m := SIDECAR.fullmatch(p)) and (mod := m.group(1) + ".py") in owner and p not in LEFT and excepted(p)[0] is None:
            there = out.get(mod)
            new = None if there is None else f"{posixpath.dirname(there[0])}/{posixpath.basename(p)}"
            if (out.get(p) or (None,))[0] != new:
                say(f"with its module: {p} goes to {new or 'nowhere (its module stays)'}, not {(out.get(p) or ('nowhere',))[0]}")
                if new is None:
                    del out[p]
                else:
                    out[p] = (new, owner[p], there[2])
    for old, new, _ in EXCEPT:
        if old not in plain:
            say(f"dropped, EXCEPT's row {old} holds no tracked file")
        elif all(plain[old]):
            say(f"EXCEPT's row {old} does nothing now: the rule gives {new}")
    for key in KEEP:
        if key not in comps:
            say(f"KEEP's row {key} does nothing now: the layout no longer moves it")
    return out


def compress(mine, tracked_dirs):
    """The rows of one map: `mine` is {old file: new file}; `tracked_dirs` is {directory: (files in it, subdirectories)}
    over every tracked file. A directory is one row where every tracked file under it is the map's and keeps its path
    below it; otherwise its files are rows and its subdirectories are looked at in turn."""
    whole = {}

    def walk(d):
        files, subs = tracked_dirs[d]
        target = set()
        for f in files:
            p = f"{d}/{f}" if d else f
            new = mine.get(p)
            target.add(None if new is None or posixpath.basename(new) != f else posixpath.dirname(new))
        for s in subs:
            n = walk(f"{d}/{s}" if d else s)
            target.add(None if n is None or posixpath.basename(n) != s else posixpath.dirname(n))
        whole[d] = target.pop() if len(target) == 1 else None
        return whole[d]
    walk("")
    rows = {}

    def emit(d):
        if d and whole[d] is not None:
            rows[d + "/"] = whole[d] + "/"
            return
        files, subs = tracked_dirs[d]
        for f in files:
            p = f"{d}/{f}" if d else f
            if p in mine:
                rows[p] = mine[p]
        for s in subs:
            emit(f"{d}/{s}" if d else s)
    emit("")
    return rows


def tokens_for(move, dest, tracked_dirs, rows):
    """The directories several maps share that this map speaks for: {old directory: new directory}."""
    out = {}
    under = collections.defaultdict(list)              # directory -> tracked files below it
    for d, (files, _) in tracked_dirs.items():
        for f in files:
            p = f"{d}/{f}" if d else f
            a = d
            while a:
                under[a].append(p)
                a = posixpath.dirname(a)
    dir_rows = [k.rstrip("/") for k in rows if k.endswith("/")]
    for d, ps in under.items():
        if any(d == r or d.startswith(r + "/") for r in dir_rows):
            continue
        mine = [p for p in ps if p in dest and dest[p][2] == move]
        if not mine:
            continue
        files = tracked_dirs[d][0]
        name = next((n for n in NAMESAKE if n in files), None)
        if name is not None:
            if dest.get(f"{d}/{name}", (None, None, None))[2] != move:
                continue
        elif 2 * len(mine) <= len(ps):
            continue
        new = {dest[p][0][: -len(p[len(d):])] if dest[p][0].endswith(p[len(d):]) else None for p in mine}
        if len(new) == 1 and (n := new.pop()) is not None:
            out[d + "/"] = n + "/"
    return out


def toml_str(s):
    return json.dumps(s, ensure_ascii=False)


def wrap(prefix, text, width=118):
    out, line = [], prefix
    for w in text.split():
        if len(line) + len(w) + 1 > width and line.strip("# "):
            out.append(line.rstrip())
            line = prefix
        line += w + " "
    out.append(line.rstrip())
    return "\n".join(out)


def render(fname, v, before):
    rows = v["rows"]
    lines = [wrap("# ", f"The layout move of {v['what']}: what `tools/move/layout.py --map tools/move/{fname}` moves and "
                        f"rewrites. {len(v['mine'])} files in {len(rows)} rows. Written by `tools/move/layout_maps.py`, "
                        "which states the rule: edit that, not this.")]
    lines.append(wrap("# ", "A moved file's new path is its component's new directory, then the name of the directory it is in "
                            "today, then its path below that; the name is left out where the new directory, less `verifier/`, "
                            "`prover/` or `device/`, already ends in it. A directory is one row where all of it goes one way; a "
                            "directory this move shares with another is one row a file. Nothing is edited or deleted."))
    if v["reasons"]:
        lines.append("# Rows that depart from that rule:")
        for old, new, why in v["reasons"]:
            lines.append(wrap("#   ", f"`{old}` to `{new}`: {why}.").replace("#   ", "# - ", 1))
    lines.append(wrap("# ", "The layout's maps run in the order of their names. " +
                      ("This one runs after " + ", ".join(f"`{b}`" for b in before) + ": it renames a package that those "
                       "maps take modules out of, and would rename their modules with it."
                       if before else "This one renames no package another map takes modules out of, so its place among them "
                       "is free.") + " The format is `layout_map.toml`'s."))
    fmt = lambda d: "{ " + ", ".join(f"{k} = {toml_str(x)}" for k, x in d.items()) + " }"   # noqa: E731
    lines += ["", "[roots]", f"old = {fmt(ROOTS)}", f"new = {fmt(v['roots_new'])}", "", "[paths]"]
    lines += [f"{toml_str(k)} = {toml_str(x)}" for k, x in sorted(rows.items())]
    if v["tokens"]:
        lines += ["", "[tokens]", "# directories this move shares with other moves, as a reference to one reads after it"]
        lines += [f"{toml_str(k)} = {toml_str(x)}" for k, x in sorted(v["tokens"].items())]
    if v["split"]:
        lines += ["", "[split]", "# references to a directory this move splits that stay right: \"old file: old directory\" = why"]
        lines += [f"{toml_str(k)} = {toml_str(x)}" for k, x in sorted(v["split"].items())]
    lines += ["", "[retired]", "paths = []", "", "[drop]", "paths = []", "", "[rewrite]",
              "# Never rewritten: records of what was (the earlier moves and their maps, changelogs, research's recorded moves, the",
              "# archive), and files whose bytes a digest pins.",
              "exclude = [" + ",\n           ".join(", ".join(toml_str(x) for x in EXCLUDE[i:i + 4]) for i in range(0, len(EXCLUDE), 4)) + "]",
              "include = [" + ", ".join(toml_str(x) for x in INCLUDE) + "]", "", "[modules]",
              "# the renames [paths] and [roots] imply (`layout.py --modules`)"]
    lines += [f"{toml_str(k)} = {toml_str(x)}" for k, x in sorted(v["modules"].items())]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default="HEAD", help="the commit whose files the maps move (default: HEAD)")
    ap.add_argument("--data", default=os.path.join(HERE, "layout_maps.json"))
    ap.add_argument("--out", default=HERE, help="where the maps go (default: tools/move)")
    ap.add_argument("--targets", help="write the table of where each moved component goes here, tab-separated")
    ap.add_argument("--created", help="write here the `__init__.py`s `layout.py` creates over all the maps, one a line")
    a = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(HERE))
    data = json.load(open(a.data))
    head = git(repo, "rev-parse", a.ref).strip()
    tracked = {p for p in git(repo, "ls-tree", "-r", "--name-only", "-z", a.ref).split("\0") if p}
    notes = []
    owner = place(data, tracked, notes.append)
    dest = plan(data, owner, notes.append)
    comps = {c["id"]: c for c in data["components"]}

    # ---- checks on the plan
    by_new = collections.defaultdict(list)
    for p, (new, _, _) in dest.items():
        by_new[new].append(p)
    twice = {n: ps for n, ps in by_new.items() if len(ps) > 1}
    if twice:
        die("two files, one destination (add a row to EXCEPT):\n  " + "\n  ".join(f"{n}: {ps}" for n, ps in sorted(twice.items())[:40]))
    onto = sorted(n for n in by_new if n in tracked and n not in dest)
    if onto:
        die("a destination is a file that stays:\n  " + "\n  ".join(onto[:40]))
    bad = sorted(n for n in by_new if not n.startswith("archive/") and {"core", "ml"} & set(n.split("/")[:-1]))
    if bad:
        die("a new path keeps a `core` or `ml` segment:\n  " + "\n  ".join(bad[:40]))
    same = sorted(p for p, (n, _, _) in dest.items() if p == n)
    if same:
        die("a row that moves nothing:\n  " + "\n  ".join(same[:40]))
    stay = sorted(p for p in tracked if p.startswith(("verity/core/", "verity/ml/")) and p not in dest and p not in LEFT)
    if stay or set(LEFT) - tracked:
        die("under verity/core or verity/ml with no destination (or LEFT names a file that is gone):\n  "
            + "\n  ".join(stay[:40] + sorted(set(LEFT) - tracked)))
    trusted_dirs = [t.strip("/") for t in data["trusted_dirs"]]
    untrusted = collections.defaultdict(list)
    for p, (n, key, _) in dest.items():
        if key in comps and not comps[key]["trusted"] and any(n == t or n.startswith(t + "/") for t in trusted_dirs):
            untrusted[key].append(f"{p} -> {n}")
    if set(untrusted) - set(UNTRUSTED):
        die("files of a component the auditor doesn't trust land under a trusted directory (a row in EXCEPT, or in "
            "UNTRUSTED with why it may):\n  " + "\n  ".join(f"{k} ({comps[k]['name']}): {len(v)} files, {v[0]}"
                                                            for k, v in sorted(untrusted.items()) if k not in UNTRUSTED))
    names = [m for m, _ in MOVES]
    unknown = sorted({m for _, _, m in dest.values()} - set(names))
    if unknown:
        die(f"moves with no entry in MOVES: {unknown}")

    # ---- the maps
    tracked_dirs = collections.defaultdict(lambda: ([], set()))
    for p in sorted(tracked):
        d = posixpath.dirname(p)
        tracked_dirs[d][0].append(posixpath.basename(p))
        while d:
            tracked_dirs[posixpath.dirname(d)][1].add(posixpath.basename(d))
            d = posixpath.dirname(d)
    tracked_dirs = {d: (fs, sorted(ss)) for d, (fs, ss) in tracked_dirs.items()}
    maps = {}
    for move, what in MOVES:
        mine = {p: n for p, (n, _, m) in dest.items() if m == move}
        if not mine:
            die(f"{move}: no file")
        rows = compress(mine, tracked_dirs)
        tokens = {} if move == "archive" else tokens_for(move, dest, tracked_dirs, rows)   # the archive speaks for no directory
        roots_new = {}
        for name, d in ROOTS.items():
            if not any(p.startswith(d + "/") for p in tracked):
                die(f"ROOTS: {name}'s directory {d} holds no tracked file")
            if f"{d}/__init__.py" not in mine:         # the package stays, whatever leaves it
                roots_new[name] = d
                continue
            inside = [p for p in tracked if p.startswith(d + "/")]
            new = {mine[p][: -len(p[len(d):])] if p in mine and mine[p].endswith(p[len(d):]) else None for p in inside}
            if len(new) != 1 or None in new:
                die(f"{move}: it moves the package {name} ({d}) in part, or to two places: {sorted(map(str, new))}")
            roots_new[name] = new.pop().rstrip("/")
        split = {k: why for m, k, why in SPLIT if m == move and k.split(": ")[0] in tracked}
        lay_data = {"roots": {"old": dict(ROOTS), "new": roots_new}, "paths": rows, "tokens": tokens, "split": split}
        modules = L.derived_modules(L.Layout(lay_data))
        lay_data["modules"] = modules
        lay = L.Layout(lay_data)
        got = {p: lay.file(p) for p in tracked if lay.file(p) is not None}
        if got != mine:
            diff = sorted(set(got.items()) ^ set(mine.items()))
            die(f"{move}: its rows move other than its files: {diff[:10]}")
        for p, n in mine.items():
            if p.endswith(".py") and (mo := L.module_name(p, ROOTS)) and (mn := L.module_name(n, roots_new)) and lay.module(mo) != mn:
                die(f"{move}: {p} -> {n}: its module {mo} would be renamed {lay.module(mo)}, not {mn}")
        reasons = [(o, n, w) for o, n, w in EXCEPT if any(holds(o, p) for p in mine)]
        maps[move] = {"what": what, "rows": rows, "tokens": tokens, "split": split, "modules": modules,
                      "roots_new": roots_new, "lay": lay, "mine": mine, "reasons": reasons}

    # ---- across the maps: no row over another map's files or over a destination, Lake packages whole
    keys = [(m, k) for m in maps for k in maps[m]["rows"]]
    for m, k in keys:
        for m2, k2 in keys:
            if m != m2 and (k == k2 or (k.endswith("/") and k2.startswith(k))):
                die(f"{m}'s row {k} holds {m2}'s row {k2}")
    for m, v in maps.items():
        hit = sorted(n for n in by_new if v["lay"].file(n) is not None)
        if hit:
            die(f"{m}: a row of it would move a destination again: {hit[:5]}")
    for p in sorted(dest):
        if posixpath.basename(p) in ("lakefile.toml", "lakefile.lean"):
            root, m = posixpath.dirname(p), dest[p][2]
            new_root = posixpath.dirname(dest[p][0])
            there = maps[m]["lay"].directory(root)[0]
            if there != new_root or any(n != new_root + q[len(root):] for q, n in maps[m]["mine"].items() if q.startswith(root + "/")):
                die(f"the Lake package at {root} does not move whole to {new_root} by a row or a token of {m} (it gives {there})")

    # ---- the order: B runs after A when B's renames would rename a module A moves
    after = {m: set() for m in maps}
    for b, vb in maps.items():
        for a_, va in maps.items():
            if a_ == b:
                continue
            for p in va["mine"]:
                if p.endswith(".py") and (mo := L.module_name(p, ROOTS)) and vb["lay"].module(mo) != mo:
                    after[b].add(a_)
                    break
            for n in va["mine"].values():
                if n.endswith(".py") and (mn := L.module_name(n, va["roots_new"])) and vb["lay"].module(mn) != mn:
                    die(f"{b}'s [modules] would rename {mn}, the new name of a module {a_} moves")
    order, left = [], list(names)
    while left:
        ready = [m for m in left if not (after[m] - set(order))]
        if not ready:
            die(f"the order has a cycle among {left}: " + "; ".join(f"{m} after {sorted(after[m] - set(order))}" for m in left))
        order.append(ready[0])
        left.remove(ready[0])

    # ---- write: the maps' names carry the order, so `layout_??_*_map.toml` sorted is the order they run in
    fname = {m: f"layout_{i:02d}_{m}_map.toml" for i, m in enumerate(order, 1)}
    os.makedirs(a.out, exist_ok=True)
    for old in os.listdir(a.out):
        if MAP_NAME.fullmatch(old) and old not in fname.values():
            os.remove(os.path.join(a.out, old))
    for move in order:
        with open(os.path.join(a.out, fname[move]), "w") as f:
            f.write(render(fname[move], maps[move], [fname[b] for b in order if b in after[move]]))
    if a.targets:
        by_comp = collections.defaultdict(list)
        for p, (n, key, m) in dest.items():
            by_comp[(m, key)].append((p, n))
        with open(a.targets, "w") as f:
            f.write("\t".join(["map", "component", "name", "files", "where its files are today", "target path",
                               "by directory: today -> target (files)"]) + "\n")
            for (m, key), pairs in sorted(by_comp.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])):
                by = collections.Counter()
                for p, n in pairs:
                    i = 0
                    while i < min(len(p), len(n)) and p[-1 - i] == n[-1 - i]:
                        i += 1
                    tail = p[len(p) - i:]
                    tail = tail[tail.find("/") + 1:] if "/" in tail else posixpath.basename(p)
                    by[(p[: len(p) - len(tail)].rstrip("/") or ".", n[: len(n) - len(tail)].rstrip("/") or ".")] += 1
                name = comps[key]["name"] if key in comps else {"archived": "what the layout archives"}.get(key, "a file the layout keeps")
                f.write("\t".join([fname[m], key, name.replace("\t", " "), str(len(pairs)), common_dir([p for p, _ in pairs]) or ".",
                                   common_dir([n for _, n in pairs]) or ".",
                                   "; ".join(f"{o} -> {n} ({k})" for (o, n), k in sorted(by.items()))]) + "\n")
    after_all = (tracked - set(dest)) | set(by_new)
    created = set()                                    # as `layout.py`'s `create` does, over all the maps
    for move in order:
        v = maps[move]
        for p, n in v["mine"].items():
            if n.endswith(".py") and f"{posixpath.dirname(p)}/__init__.py" in tracked:
                root = next((d for d in sorted(v["roots_new"].values(), key=len, reverse=True) if n.startswith(d + "/")), None)
                d = posixpath.dirname(n)
                while root is not None and d and (d == root or d.startswith(root + "/")):
                    if f"{d}/__init__.py" not in after_all:
                        created.add(f"{d}/__init__.py")
                    d = posixpath.dirname(d)
    if a.created:
        with open(a.created, "w") as f:
            f.write("".join(p + "\n" for p in sorted(created)))
    moved = {key for _, key, _ in dest.values() if key in comps}
    print(f"{len(order)} maps at {a.ref} {head[:12]} (the data's cut: {data['commit'][:12]}): {len(dest)} files "
          f"({sum(1 for _, k, _ in dest.values() if k in comps)} in {len(moved)} components that move, "
          f"{sum(1 for _, k, _ in dest.values() if k == 'archived')} archived), "
          f"{sum(len(v['rows']) for v in maps.values())} rows, {sum(len(v['tokens']) for v in maps.values())} tokens")
    for move in order:
        v = maps[move]
        print(f"  {fname[move]}: {len(v['mine'])} files, {len(v['rows'])} rows "
              f"({sum(1 for k in v['rows'] if k.endswith('/'))} directories), {len(v['tokens'])} tokens, {len(v['split'])} "
              f"split rows, {len(v['modules'])} module renames" + (f"; after {', '.join(sorted(after[move]))}" if after[move] else ""))
    for key, v in sorted(untrusted.items()):
        print(f"  untrusted under a trusted directory, allowed by UNTRUSTED: {key}, {len(v)} files ({v[0]})")
    print(f"  `layout.py` creates {len(created)} `__init__.py` over the maps (--created FILE lists them)")
    for p in LEFT:
        print(f"  left in place: {p}")
    for n in notes:
        print("  " + n)


if __name__ == "__main__":
    main()
