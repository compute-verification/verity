"""The Lean packages in the tree (AGENTS.md "Lean"): each is audited (a `lean-audit.json` beside its lakefile, which `check`'s
`lean-audit` step reads), and all of them build with one toolchain and one Mathlib, so a bump is one change to all of them."""
from __future__ import annotations

import functools
import importlib.util
import itertools
import json
import os
import re
import tomllib
import warnings
from collections import Counter
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".lake", ".git", ".venv", "node_modules", "__pycache__"}
#: the Lean tools' test fixtures, which no audit reads (`modules.FIXTURES`): trees laid out as this one's, roots and all
FIXTURES = ROOT / "tools/verity/lean/tests/fixtures"


def lake_packages() -> list[Path]:
    out = []
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(x for x in dirs if x not in SKIP)
        if ("lakefile.toml" in files or "lakefile.lean" in files) and not Path(d).is_relative_to(FIXTURES):
            out.append(Path(d))
    return out


def test_every_lake_package_is_audited():
    pkgs = lake_packages()
    assert len(pkgs) >= 3
    missing = [str(p.relative_to(ROOT)) for p in pkgs if not (p / "lean-audit.json").exists()]
    assert not missing, f"Lake packages without a lean-audit.json (tools/verity/lean/README.md): {missing}"


def test_every_package_module_is_a_module():
    """Every in-tree package's Lean is on Lean's module system (`tools/verity/lean/modules.py`), so an importer rebuilds
    only when what a module exports changes, never for a proof's body."""
    spec = importlib.util.spec_from_file_location("lean_modules", ROOT / "tools" / "verity" / "lean" / "modules.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    tracked = m.tracked()
    files = m.modules([], tracked)
    assert len(files) > 1000, len(files)
    left = [f for f in files if not m.is_module((ROOT / f).read_text())]
    assert not left, "not on Lean's module system; port each with `python3 tools/verity/lean/modules.py FILE`:\n  " + \
        "\n  ".join(left)
    lakefiles = m.lake_option_missing(tracked)
    assert not lakefiles, f"without `{m.LAKE_OPTION}` under [leanOptions], so a public statement's proofs become auxiliary " \
        "theorems and its lock records move; add it with `python3 tools/verity/lean/modules.py PACKAGE`:\n  " + \
        "\n  ".join(lakefiles)


def test_one_toolchain():
    """The jobs' base image installs the same one."""
    pins = {str(p.relative_to(ROOT)): (p / "lean-toolchain").read_text().strip() for p in lake_packages()}
    image = ROOT / "tools/research/src/research/pods/nebius/image/Dockerfile"
    pins[str(image.relative_to(ROOT))] = re.search(r"^ARG LEAN_TOOLCHAIN=(\S+)", image.read_text(), re.MULTILINE).group(1)
    assert len(set(pins.values())) == 1, pins


def test_one_mathlib():
    revs = {}
    for p in lake_packages():
        manifest = p / "lake-manifest.json"
        if manifest.exists():
            for dep in json.loads(manifest.read_text())["packages"]:
                if dep["name"] == "mathlib":
                    revs[str(p.relative_to(ROOT))] = dep["rev"]
    assert revs and len(set(revs.values())) == 1, revs


#: the size of each lock's `unwitnessed`, as the regeneration that recorded every guarantee's premises left it: the
#: guarantees stating what an acceptance means, or an exclusion's bound, that have no `honest` and `nontrivial` witnesses
#: (`check.py`'s `witness_failures`)
# Security's 279: Flock.SecurityProofs.SpotExfilTwo, NetTiming.SecurityProofs.SampledLeakChecked and PooledLeakChecked carry
# their nontrivial witnesses; their honest ones are due with the two-stage layout and the deployed check pinning CheckedUnique
UNWITNESSED = {"Security": 303, "Security/Properties": 191, "verity/protocols/flock/verifier": 7}


def test_the_unwitnessed_list_only_shrinks():
    """A lock's `unwitnessed` never grows. A new guarantee of what an acceptance means, and one whose statement
    changed, carries its witnesses instead; when a guarantee gets them, its entry goes, and its lock's number here comes
    down with it (`_held`)."""
    _held("UNWITNESSED", UNWITNESSED, {_rel(p): len(_lock(p).get("unwitnessed", {})) for p in lake_packages()},
          "`unwitnessed`", "it only shrinks (LEAN-66); give the guarantee you added its `honest` and `nontrivial` "
          "witnesses under `witnesses` (check.py's failure names both) instead")


def test_the_trust_base_is_one_list():
    """Every lock that names a trust base (`trust.base`, tools/verity/lean/README.md) names the same one, each premise
    with the same trust assumption, so no package's deployed guarantees count a premise as trusted that another's count
    in their gap; a premise one guarantee alone may rest on is its own (`trust.extra`)."""
    bases = {_rel(p): _lock(p)["trust"]["base"] for p in lake_packages() if "trust" in _lock(p)}
    assert "Security" in bases and len({json.dumps(b, sort_keys=True) for b in bases.values()}) == 1, bases


def test_the_randomness_assumption_cites_this_toolchain():
    """A3 (Flock soundness's `ASSUMPTIONS.md`, `Assumptions.UniformRandomBytes` in `Definitions/Assumptions/Flock.lean`)
    cites what the toolchain's runtime reads for `IO.getRandomBytes` (`lean_io_get_random_bytes`), so a toolchain bump
    fails here until someone re-reads it and updates the citation."""
    security = ROOT / "Security"
    version = (security / "Proofs/lean-toolchain").read_text().strip().rpartition(":")[2]
    for doc in ("Proofs/Flock/Soundness/ASSUMPTIONS.md", "Definitions/Assumptions/Flock.lean"):
        assert f"Lean {version}'s runtime" in (security / doc).read_text(), doc


def _module_roots(pkg: Path):
    """(source directory, root module) for each library and executable of `pkg`, as Lake resolves them."""
    cfg = tomllib.loads((pkg / "lakefile.toml").read_text())
    src = pkg / cfg.get("srcDir", ".")
    for lib in cfg.get("lean_lib", []):
        for r in lib.get("roots", [lib["name"]]) + [g.rstrip("+*").rstrip(".") for g in lib.get("globs", [])]:
            yield (src / lib.get("srcDir", ".")).resolve(), r
    for exe in cfg.get("lean_exe", []):
        yield (src / exe.get("srcDir", ".")).resolve(), exe.get("root", exe["name"])


def test_lean_module_roots_are_disjoint():
    """No module name is claimed from two source directories, by two packages or by one. Lake resolves an import by name
    across every library in the workspace and doesn't report a duplicate: a library whose roots overlap another's builds
    the other's module from its own directory, without a warning. Within one package, roots in different source
    directories may share a namespace but not nest (`Pouw.Protocol` and `Pouw.Guarantees` may; `Pouw` and
    `Pouw.Guarantees` may not). Across packages they may not even share a first component: with `Pk.A` in one package
    and `Pk.B` in another, `lake build` passes but `lake env lean` looks for every `Pk.*` under the first package's
    build directory and fails to find `Pk.B`."""
    claims = []
    for pkg in lake_packages():
        assert (pkg / "lakefile.toml").exists(), f"{pkg.relative_to(ROOT)}: only a lakefile.toml can be read here"
        claims += [(pkg, src, root) for src, root in _module_roots(pkg)]

    def nested(a: str, b: str) -> bool:
        a, b = a.split("."), b.split(".")
        return a[: len(b)] == b or b[: len(a)] == a

    def clash(x, y) -> bool:
        if x[0] != y[0]:
            return x[2].split(".")[0] == y[2].split(".")[0]
        return x[1] != y[1] and nested(x[2], y[2])

    clashes = [f"{x[2]} ({x[0].relative_to(ROOT)}, {x[1].relative_to(ROOT)}) and "
               f"{y[2]} ({y[0].relative_to(ROOT)}, {y[1].relative_to(ROOT)})"
               for x, y in itertools.combinations(claims, 2) if clash(x, y)]
    assert len(claims) >= 10
    assert not clashes, clashes


def test_every_record_merges_with_the_driver():
    """`.gitattributes` sends every `lean-audit.json` to `tools/verity/lean/merge.py`, and each record's `reads` digests are the ones
    the driver recomputes from their definitions."""
    spec = importlib.util.spec_from_file_location("lean_merge", ROOT / "tools" / "verity" / "lean" / "merge.py")
    merge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge)
    assert f"lean-audit.json merge={merge.NAME}" in (ROOT / ".gitattributes").read_text().splitlines()
    stale = [f"{p.relative_to(ROOT)}: {m}" for p in lake_packages()
             for m, g in json.loads((p / "lean-audit.json").read_text()).get("reads", {}).items()
             if g["digest"] != merge.A.module_digest(merge.A.definitions(g))]
    assert not stale, (f"reads digests that are not the SHA-256 of their definitions (`python tools/verity/lean/merge.py --normalize "
                       f"FILE` recomputes them): {stale}")


def test_every_fetching_package_has_a_pinned_bundle():
    """A Lake package that fetches dependencies has a bundle in `lean-deps.json` for its manifest's fetched entries and its
    toolchain (`deps_key`), which is what `check`'s preflight on node 1 restores; without one main's full check can't start.
    A change to a fetched entry needs its pin: a cold `lean_audit.py --export` run, then `--pin RUN`; a
    `path` entry changes no key."""
    spec = importlib.util.spec_from_file_location("lean_audit", ROOT / "tools" / "verity" / "lean" / "lean_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    pinned = json.loads((ROOT / audit.PIN).read_text())["deps"]
    missing = [f"{p.relative_to(ROOT)} ({k[:12]})" for p in lake_packages() if (k := audit.deps_key(p)) and k not in pinned]
    assert not missing, f"Lake packages whose dependencies {audit.PIN} pins no bundle for: {missing}"


#: a fresh pin of PKG's bundle: an audit of this tree bundles the dependencies its build left, then `--pin` pins that run's
REPIN = ("research run --on auto --mem-gb 32 --project verity --source . --cwd clone -- bash -c 'python3 "
         "tools/verity/lean/lean_audit.py --out \"$OUT/audit\" --export \"$OUT/lean-deps\" --package PKG', then "
         "`python3 tools/verity/lean/lean_audit.py --pin RUN` where the store's remote is configured")


def unbundled(imports: dict[str, list[str]], listed: dict[str, dict[str, list[str]]], local: set[str]) -> list[str]:
    """`FILE: MODULE (PACKAGE)` for each import of ``imports`` ({file: the modules it imports}) from a fetched package of
    ``listed`` (a pin's `modules`) that its bundle doesn't hold built. A root the tree's own modules have (``local``) is the
    tree's, whatever a fetched package's checkout has at its top."""
    owner = {r: n for n, m in listed.items() for r in m["roots"] if r not in local}
    built = {m for v in listed.values() for m in v["built"]}
    return sorted(f"{f}: {m} ({owner[m.split('.')[0]]})" for f, ms in imports.items() for m in ms
                  if m.split(".")[0] in owner and m not in built)


def test_every_import_from_a_fetched_package_is_in_its_pinned_bundle():
    """A bundle holds a fetched package outside Mathlib's cache built only as far as the tree imported it when it was
    exported (Lake builds only what a build imports), and a node without a warm copy has the bundle alone: the Lean check's
    module runner mounts it read-only, so an import beyond it is an error there, not a build. Each import of a package's
    sources, and of those of the packages it requires by path, from such a package is one its pin lists built (`modules`,
    from `lean_audit.py --export`). Direct imports suffice: Lake built each one's own imports with it."""
    audit = _load("lean_audit", ROOT / "tools" / "verity" / "lean" / "lean_audit.py")
    headers = _load("lean_headers", ROOT / "tools" / "verity" / "lean" / "headers.py")
    pinned = json.loads((ROOT / audit.PIN).read_text())["deps"]
    files = _lean_files()
    local = {headers.components(r)[0] for p in lake_packages() for _, r in _module_roots(p)}
    imports = {f: [headers.name(i.module) for i in h.imports] if (h := headers.header(_text(f))) else [] for f in files}
    bad = []
    for pkg in lake_packages():
        if (k := audit.deps_key(pkg)) is None or k not in pinned:
            continue
        if "modules" not in pinned[k]:
            bad.append(f"{_rel(pkg)}: its bundle {k[:12]} lists no modules")
            continue
        builds = {pkg, *(ROOT / d for d in audit.path_deps(_rel(pkg)))}
        bad += [f"{_rel(pkg)}: {u}" for u in unbundled({_rel(f): ms for f, ms in imports.items() if files[f] in builds},
                                                     pinned[k]["modules"], local)]
    assert not bad, (f"imports that {audit.PIN}'s bundles don't hold built, which a node without a warm copy can't build; "
                     f"re-pin each PACKAGE named: {REPIN}:\n  " + "\n  ".join(bad))


def test_an_import_its_bundle_does_not_hold_built_is_named():
    """`unbundled` on a toy pin: an import of a fetched package's module that isn't built is named with its package, also
    of a package nothing was built of; a built one, a module of Mathlib's cache (no listed root) and a root the tree's own
    modules have are not."""
    listed = {"Arklib": {"roots": ["ArkLib"], "built": ["ArkLib.A"]}, "VCVio": {"roots": ["VCVio", "Proofs"], "built": []}}
    imports = {"P/X.lean": ["ArkLib.A", "ArkLib.B", "Mathlib.Data.Nat", "Proofs.Y"], "P/Z.lean": ["VCVio.C"]}
    assert unbundled(imports, listed, {"Proofs"}) == ["P/X.lean: ArkLib.B (Arklib)", "P/Z.lean: VCVio.C (VCVio)"]


SECURITY = ROOT / "Security"
PROOFS = SECURITY / "Proofs"


def _requires(pkg: Path) -> list[dict]:
    return tomllib.loads((pkg / "lakefile.toml").read_text()).get("require", []) if (pkg / "lakefile.toml").exists() else []


def test_security_package_requires_only_mathlib():
    """The security analysis's trusted text (`Security`: the spec, Definitions) is what every guarantee is stated
    in, so it requires nothing of the repository and no ArkLib: a path dependency here would put a component or a proof
    under every guarantee."""
    assert SECURITY in lake_packages()
    assert [r["name"] for r in _requires(SECURITY)] == ["mathlib"]


def test_proofs_require_the_security_package():
    """The proofs (`Security/Proofs`, srcDir `..`) prove what `security` states, so they require it by path."""
    assert PROOFS in lake_packages()
    paths = {(PROOFS / r["path"]).resolve() for r in _requires(PROOFS) if "path" in r}
    assert SECURITY in paths
    assert tomllib.loads((PROOFS / "lakefile.toml").read_text()).get("srcDir") == ".."


def test_code_never_requires_the_security_analysis():
    """Code (Flock's verifier, the PoUS grader, the certifier's program) stays with its component and never imports
    `Security/`: the analysis reads the code, never the reverse."""
    code = [pkg for pkg in lake_packages() if not pkg.is_relative_to(SECURITY) and not pkg.is_relative_to(ROOT / "tools")]
    assert code, "no code packages outside Security"
    bad = [f"{pkg.relative_to(ROOT)} requires {(pkg / r['path']).resolve().relative_to(ROOT)}"
           for pkg in code for r in _requires(pkg) if "path" in r and (pkg / r["path"]).resolve().is_relative_to(SECURITY)]
    assert not bad, bad


def test_every_upstream_watch_is_scanned():
    """The audit's upstream pass reads only a lock's `scan` packages, so a `watch` list with nothing to scan checks none of
    its entries and passes (a lock that loses its `scan` silently stops watching). Each lock that
    watches a dependency for its assumptions scans one, and the proofs' lock scans ArkLib."""
    bare = [str(p.relative_to(ROOT)) for p in sorted(ROOT.rglob("lean-audit.json")) if ".lake" not in p.parts
            and (up := json.loads(p.read_text()).get("upstream", {})).get("watch") and not up.get("scan")]
    assert not bare, bare
    assert "Arklib" in json.loads((PROOFS / "lean-audit.json").read_text())["upstream"]["scan"]


def test_every_open_conjecture_names_a_test_that_exists():
    """An open entry of a lock's `shape.conjectures` names its falsification test as a pytest node id, `PATH::NAME` from
    the repository's root, and that file defines the test, so each conjecture's test runs."""
    bad = []
    for p in sorted(ROOT.rglob("lean-audit.json")):
        if ".lake" in p.parts:
            continue
        for name, c in json.loads(p.read_text()).get("shape", {}).get("conjectures", {}).items():
            if c.get("status") != "open":
                continue
            path, _, test = str(c.get("test", "")).partition("::")
            node = test.split("[")[0]
            if not (ROOT / path).is_file() or not re.search(rf"^def {re.escape(node)}\(", (ROOT / path).read_text(), re.M):
                bad.append(f"{p.relative_to(ROOT)}: {name}'s test {c.get('test')!r} is not PATH::NAME of a test in the tree")
    assert not bad, bad


#: the size of each lock's `shape.inline`, in premises: the inline hypotheses lock-listed guarantees took that are neither
#: a named assumption, another listed guarantee's statement nor a plain range when the inline rule came (`check.py`'s
#: `inline_failures`), with those the PRs on `next` then brought and those `main` landed before the rule reached them, less
#: the 398 premises of the lemmas and completeness theorems the lock stopped listing and the guarantees deleted with the
#: falsified conjectures they rest on; a renamed or restated premise counts twice, under both texts, until `--update`
#: drops the one no tree prints
INLINE = {"Security": 915, "Security/Properties": 132, "verity/protocols/flock/verifier": 6}


def test_the_inline_list_only_shrinks():
    """A lock's `shape.inline` never grows: a guarantee's new inline hypothesis is named as an assumption or stated as
    another listed guarantee's conclusion instead; when a grandfathered one is named, or a renamed one's old text is no
    longer printed, `--update` drops its entry, and its lock's number here comes down with it (`_held`)."""
    _held("INLINE", INLINE, {_rel(p): sum(map(len, _lock(p).get("shape", {}).get("inline", {}).values()))
                             for p in lake_packages()},
          "`shape.inline`", "it only shrinks; name the hypothesis as a Prop and list it in the lock (under `open`, with "
          "its issue and due date), or state it as a lock-listed guarantee's conclusion")


#: how far a cap here may stand above its count ({"CAP key": room}): changes queued on `next` when the caps came to equal
#: their counts moved the counts and not the caps, so each room is the gap between `next`'s count and `main`'s, and each
#: cap holds on both. Once those changes have landed, a cap and its room come down to the count together.
ROOM = {"INLINE Security": 3, "INLINE Security/Properties": 20, "UNWITNESSED Security": 2, "RETIRED_NAMES draw": 59, "RETIRED_NAMES wrong": 10, "OVERLONG Security/Proofs": 275, "RETIRED_NAMES escape": 9, "RETIRED_NAMES law": 9, "TREES_EXCEPT Security": 1, "UNWITNESSED Security/Properties": 3, "OVERLONG verity/protocols/pouw/verifier/lean/check": 3, "OVERLONG verity/protocols/flock/verifier": 4, "IMPORT_ALL_USES Security/Proofs": 1}


def test_the_caps_the_lock_run_repairs_are_the_files_drive_commits():
    """Drive's lock run (`lean_audit.py --bookkeeping warn`) repairs the unread caps and this file's caps on what the locks
    list, and drive commits them beside the locks (`research.merge.REPAIRED`, which the merge gate holds to the run's copies):
    the two name the same files, and each cap here is the one-line dict `lean_audit.held_caps` rewrites."""
    import ast

    def assigned(path: str) -> dict[str, ast.expr]:
        return {t.id: n.value for n in ast.parse((ROOT / path).read_text()).body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    audit, merge = assigned("tools/verity/lean/lean_audit.py"), assigned("tools/research/src/research/merge.py")
    repaired = {ast.literal_eval(e) for e in merge["REPAIRED"].args[0].elts}
    assert repaired == {ast.literal_eval(audit["UNREAD_CAPS"]), ast.literal_eval(audit["CAPS_TEST"])}
    assert ast.literal_eval(audit["CAPS_TEST"]) == _rel(Path(__file__))
    text = Path(__file__).read_text()
    for name in ("ROOM", *(k.value for k in audit["LOCK_CAPS"].keys)):
        assert re.search(rf"^{name} = \{{.*\}}$", text, re.M), f"{name} is no longer one line here: `held_caps` can't hold it"


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def _lock(pkg: Path) -> dict:
    return json.loads((pkg / "lean-audit.json").read_text())


def _held(name: str, caps: dict[str, int], got: dict[str, int], what: str, fix: str) -> None:
    """``got``, a count by key, held to the cap ``caps`` (this file's ``name``), which equals it: a count above its cap
    fails with ``fix``, and a cap above its count by more than its `ROOM` warns, naming the value to lower it to, so a
    change that lowers a count never fails the check of a commit it lands on; the next change, or drive's lock run, lowers
    the cap (`lean_audit.held_caps`), so the next offender, or an entry added by hand, is seen. The `lean-caps` merge driver
    (`tools/verity/lean/merge.py --caps`) merges two changes to a cap as the larger of the two sides' caps."""
    over = {k: f"{n} > {caps.get(k, 0)}" for k, n in got.items() if n > caps.get(k, 0)}
    assert not over, (f"{what} grew in {over}: {fix} (where two merged changes each added entries, the merge kept the "
                      f"larger of their caps while the count took both: raise it to the count)")
    loose = {k: got.get(k, 0) + ROOM.get(f"{name} {k}", 0) for k in caps
             if caps[k] > got.get(k, 0) + ROOM.get(f"{name} {k}", 0)}
    for k, n in loose.items():
        warnings.warn(f"{what} in {k} is {got.get(k, 0)}, under its cap of {caps[k]}: lower {name}[{k!r}] to {n}", stacklevel=2)


#: the size of each lock's `trees.except`: the modules whose statements may read a definition of the proofs tree, almost
#: all Flock's until its spec is extracted
TREES_EXCEPT = {"Security": 41}


def test_the_trees_except_list_only_shrinks():
    """A lock's `trees.except` never grows: a statement reads only the spec, and `check.py` fails an entry that excepts
    nothing, which goes with its cap here."""
    _held("TREES_EXCEPT", TREES_EXCEPT, {_rel(p): len(_lock(p).get("trees", {}).get("except", {})) for p in lake_packages()},
          "`trees.except`", "move the definition the statement reads from `Proofs/` into `Definitions/` instead")


#: the size of each lock's `four_names_except`: the guarantees under a root `four_names` lists that weren't in the four-name
#: form, frozen or a witness when the rule came to every root
FOUR_NAMES_EXCEPT = {"Security": 3}


def test_the_four_names_exceptions_only_shrink():
    """A lock's `four_names_except` never grows: a new guarantee is stated in the four-name form, and `check.py` fails an
    entry that is in it, which goes with its cap here."""
    _held("FOUR_NAMES_EXCEPT", FOUR_NAMES_EXCEPT, {_rel(p): len(_lock(p).get("four_names_except", {})) for p in lake_packages()},
          "`four_names_except`", "state the guarantee as `X.SecurityProofs.G : X.Guarantees.G` (`check.py --lift` writes it)")


def _blanked(src: str) -> tuple[str, dict[int, bool]]:
    """``src`` with its comments and string literals blanked to spaces (line breaks kept, so offsets and lines stay), and
    {where each comment ends: whether it is a doc comment (`/--`)}."""
    out, ends, i = [], {}, 0
    opener = re.compile(r'--|/-|"|\'"\'')
    while (m := opener.search(src, i)) is not None:
        out.append(src[i:m.start()])
        j = m.end()
        if m[0] == "--":
            j = src.find("\n", j)
            j = len(src) if j < 0 else j
        elif m[0] == "/-":
            depth = 1
            while j < len(src) and depth:
                if src.startswith("/-", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("-/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            ends[j] = src.startswith("/--", m.start())
        elif m[0] == '"':
            while j < len(src) and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, len(src))
        out.append(re.sub(r"[^\n]", " ", src[m.start():j]))
        i = j
    out.append(src[i:])
    return "".join(out), ends


#: a scope's opening or closing, or a declaration, at the start of a line of blanked source (`_blanked`)
_SCOPED = re.compile(r"^[ \t]*(?:(?P<namespace>namespace)[ \t]+(?P<ns>\S+)"
                     r"|(?:@\[[^\]\n]*\][ \t]*)?(?:(?:public|private|noncomputable)[ \t]+)*(?P<section>section|mutual)\b"
                     r"[ \t]*(?P<sec>[^\s]*)|(?P<end>end)\b[ \t]*(?P<closes>[^\s]*)"
                     r"|(?:@\[[^\]\n]*\][ \t]*)*(?P<mods>(?:(?:private|protected|noncomputable|partial|unsafe|nonrec|public"
                     r"|meta)[ \t]+)*)(?P<kind>theorem|lemma|def|abbrev|structure|class[ \t]+inductive|inductive|class"
                     r"|instance|opaque)[ \t]+(?P<name>[^\s(:{\[]+))", re.MULTILINE)


class Declaration(NamedTuple):
    #: the full name, under the namespaces open where it is declared
    name: str
    #: the name as the declaration writes it
    written: str
    kind: str
    line: int
    #: whether a doc comment (`/-- … -/`) heads it, attributes between them aside
    documented: bool
    private: bool


def declarations(src: str) -> list[Declaration]:
    """Each declaration of Lean source ``src`` that names itself, outside comments."""
    code, ends = _blanked(src)
    scopes: list[tuple[str, int]] = []
    out, at, line = [], 0, 1
    for m in _SCOPED.finditer(code):
        if m["namespace"]:
            scopes.append((m["ns"], len(m["ns"].split("."))))
        elif m["section"]:
            scopes.append(("", len(m["sec"].split(".")) if m["sec"] else 1))
        elif m["end"]:
            n = len(m["closes"].split(".")) if m["closes"] else 1
            while scopes and n > 0:
                n -= scopes.pop()[1]
        else:
            ns = ".".join(s for s, _ in scopes if s)
            written = m["name"].rstrip(".")
            name = written.removeprefix("_root_.") if written.startswith("_root_.") or not ns else f"{ns}.{written}"
            j = m.start() + len(m[0]) - len(m[0].lstrip())
            while True:
                while j and src[j - 1].isspace():
                    j -= 1
                k = src.rfind("\n", 0, j) + 1
                if not (src[j - 1:j] == "]" and src[k:j].lstrip().startswith("@[")):
                    break
                j = k
            line, at = line + code.count("\n", at, m.start()), m.start()
            out.append(Declaration(name, written, m["kind"].split()[-1], line, ends.get(j, False),
                                   "private" in m["mods"].split()))
    return out


def _module_file(pkg: Path, module: str) -> Path | None:
    """The file of ``module`` among `pkg`'s libraries and executables, or None when it isn't one of them."""
    for src, root in _module_roots(pkg):
        if (module == root or module.startswith(root + ".")) and (f := src / (module.replace(".", "/") + ".lean")).is_file():
            return f
    return None


@functools.cache
def _lean_files() -> dict[Path, Path]:
    """Each `.lean` file of the tree, with the package it builds in: the deepest whose directory holds it."""
    pkgs = sorted(lake_packages(), key=lambda p: len(p.parts), reverse=True)
    out = {}
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(x for x in dirs if x not in SKIP)
        for f in sorted(files):
            if f.endswith(".lean") and (pkg := next((k for k in pkgs if Path(d).is_relative_to(k)), None)) is not None:
                out[Path(d) / f] = pkg
    return out


@functools.cache
def _text(f: Path) -> str:
    return f.read_text()


def undocumented(read: dict[str, set[str]], sources: dict[str, tuple[str, str]]) -> list[str]:
    """`FILE:LINE NAME` for each definition no doc comment heads that a statement reads: a name of ``read`` ({module:
    names}) declared in that module's source (``sources``, {module: (its file, its text)}), or the declaration it belongs
    to (a structure's field or constructor, a definition's generated part). An instance means what its class says, and a
    theorem is no definition."""
    out = set()
    for m, names in read.items():
        if m not in sources:
            continue
        rel, text = sources[m]
        decls = {d.name: d for d in declarations(text) if d.kind not in ("theorem", "lemma", "instance")}
        for n in names:
            parts = n.split(".")
            d = next((decls[k] for i in range(len(parts), 0, -1) if (k := ".".join(parts[:i])) in decls), None)
            if d is not None and not d.documented:
                out.add(f"{rel}:{d.line} {d.name}")
    return sorted(out)


#: the definitions of the spec (`Security`) that a lock-listed guarantee's statement reads (a lock's `reads`) and no doc
#: comment heads
UNDOCUMENTED = {"Security": 61}


def test_every_definition_a_guarantee_reads_says_what_it_means():
    """A statement means what the definitions it reads say, and the landing DM shows each one with its doc comment: a body
    that says less than intended is an error only its reader catches, so every spec definition a listed guarantee reads
    carries a `/-- … -/` saying what it means."""
    read: dict[str, set[str]] = {}
    for p in lake_packages():
        for m, r in _lock(p).get("reads", {}).items():
            read.setdefault(m, set()).update(r.get("definitions", {}))
    sources = {m: (_rel(f), _text(f)) for m in read if (f := _module_file(SECURITY, m)) is not None}
    bad = undocumented(read, sources)
    _held("UNDOCUMENTED", UNDOCUMENTED, {"Security": len(bad)}, "the read definitions without a doc comment",
          "add a `/-- … -/` above each new one saying what it means; the landing DM shows it:\n  " + "\n  ".join(bad))


#: the theorems the spec (`Security`) states: `theorem` and `lemma` declarations in its files
SPEC_THEOREMS = {"Security": 156}


def test_the_spec_states_no_new_lemma():
    """The spec holds definitions, named assumptions and guarantee statements; a lemma about them goes in `Proofs/`, which
    may change without a statement reviewer, except a theorem a definition's proof field needs."""
    bad = sorted(f"{_rel(f)}:{d.line} {d.name}" for f, pkg in _lean_files().items() if pkg == SECURITY
                 for d in declarations(_text(f)) if d.kind in ("theorem", "lemma"))
    _held("SPEC_THEOREMS", SPEC_THEOREMS, {"Security": len(bad)}, "the spec's theorems",
          "move the new one to `Proofs/<Area>/`; a theorem a definition's proof field needs stays, and raises the cap in "
          "this change:\n  " + "\n  ".join(bad))


#: the lines a module may run to: one module is one unit of rebuild, so a long one costs every check and train its build
LONG = 2000
#: the lines past `LONG` of each package's modules, summed
OVERLONG = {"Security": 963, "Security/Proofs": 54185, "verity/protocols/pouw/verifier/lean/check": 281, "verity/protocols/flock/verifier": 700}


def overlong(lines: dict[str, int]) -> dict[str, int]:
    """{file: its lines} for each file of ``lines`` ({file: its lines}) past `LONG`."""
    return {f: n for f, n in sorted(lines.items()) if n > LONG}


def test_no_long_module_grows():
    by: dict[str, dict[str, int]] = {}
    for f, pkg in _lean_files().items():
        by.setdefault(_rel(pkg), {})[_rel(f)] = len(_text(f).splitlines())
    long = {p: overlong(fs) for p, fs in by.items() if overlong(fs)}
    _held("OVERLONG", OVERLONG, {p: sum(n - LONG for n in fs.values()) for p, fs in long.items()},
          f"the lines past {LONG} of a package's modules",
          "put the new declarations in a module beside the long one and import it; a generator splits its output:\n  "
          + "\n  ".join(f"{f}: {n} lines" for fs in long.values() for f, n in fs.items()))


#: the Lean declarations whose own names carry each of the Glossary's older names (`test_retired_terms.py`'s `LEAN_WORDS`)
RETIRED_NAMES = {"draw": 1020, "law": 133, "escape": 98, "wrong": 130, "dampener": 37, "firewall": 26, "horizon": 31, "warden": 10, "tap": 19, "leg": 3}


#: each `import all` of an in-tree module ({"FILE: MODULE": why}), those in the tree or on `next` when the rule came; an
#: entry whose FILE the tree doesn't have waits for the change on `next` that adds it
IMPORT_ALL: dict[str, str] = {
    "Security/Proofs/Flock/Claims/Sampling/Select/Second.lean: Spot.Draw":
        "its proofs open `Spot.Draw`'s private `stratum?` and `natRanges` to read the stratified law's strata",
}
#: the entries of `IMPORT_ALL` whose FILE the tree has, per package
IMPORT_ALL_USES = {"Security/Proofs": 1}


def imports_all(files: dict[str, str], in_tree) -> list[str]:
    """`FILE: MODULE` for each `import all MODULE` of ``files`` ({file: its source}) that ``in_tree`` (a module's name: bool)
    says is the tree's."""
    headers = _load("lean_headers", ROOT / "tools" / "verity" / "lean" / "headers.py")
    return sorted(f"{f}: {i.module}" for f, src in files.items() if (h := headers.header(src))
                  for i in h.imports if i.all and in_tree(i.module))


def import_all_failures(found: list[str], allowed: dict[str, str], files: set[str]) -> list[str]:
    """What fails of the `import all` uses ``found`` (`imports_all`) against ``allowed`` (`IMPORT_ALL`), in a tree of
    ``files``: a use it doesn't list, an entry whose file no longer has it, and an entry without its reason."""
    return ([f"`import all` of an in-tree module, {u}: import it publicly, and move the proof that needs its private part "
             f"into it or `@[expose]` the definition that proof reads" for u in found if u not in allowed]
            + [f"IMPORT_ALL lists {u}, which no file has any more: remove the entry" for u in sorted(set(allowed) - set(found))
               if u.split(": ")[0] in files]
            + [f"IMPORT_ALL[{u!r}] gives no reason" for u, why in sorted(allowed.items()) if not why.strip()])


def test_no_import_all_of_an_in_tree_module():
    """`import all M` reads `M`'s private part, so every edit of `M` rebuilds the importer as if `M` exposed everything,
    which undoes what the module system buys; `import all` of the toolchain's own modules (`Init`) is not the tree's."""
    homes = [h for p in lake_packages() for h in _module_roots(p)]
    files = _lean_files()

    def in_tree(m: str) -> bool:
        return any((m == r or m.startswith(r + ".")) and (s / (m.replace(".", "/") + ".lean")).is_file() for s, r in homes)
    pkg = {_rel(f): _rel(p) for f, p in files.items()}
    bad = import_all_failures(imports_all({_rel(f): _text(f) for f in files}, in_tree), IMPORT_ALL, set(pkg))
    assert not bad, "\n".join(bad)
    _held("IMPORT_ALL_USES", IMPORT_ALL_USES, dict(Counter(pkg[f] for u in IMPORT_ALL if (f := u.split(": ")[0]) in pkg)),
          "`IMPORT_ALL`", "it only shrinks; import the module publicly instead")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_cap_merges_as_the_larger_of_its_sides():
    """Each cap here is one line the `lean-caps` merge driver reads (`merge.py`'s `CAP_LINE`), so two changes to it merge
    as the larger of the two sides' caps instead of conflicting."""
    merge = _load("lean_merge", ROOT / "tools" / "verity" / "lean" / "merge.py")
    caps = [s for s in Path(__file__).read_text().splitlines() if re.match(r"[A-Z][A-Z_]* = \{", s)
            and isinstance(v := globals()[s.split(" = ")[0]], dict) and all(isinstance(n, int) for n in v.values())]
    assert len(caps) == len(merge.CAP_NAMES), (caps, merge.CAP_NAMES)
    assert all(merge.CAP_LINE.match(s) for s in caps), [s for s in caps if not merge.CAP_LINE.match(s)]


def test_a_cap_fails_above_its_count_and_warns_below_it_beyond_its_room():
    """`_held` on toy counts: a count above its cap fails, naming the check's fix, and a cap above its count by more than
    its room only warns, naming the value to lower it to."""
    import pytest

    with pytest.raises(AssertionError, match=r"`x` grew in \{'A': '3 > 2'\}: say less"):
        _held("CAP", {"A": 2}, {"A": 3}, "`x`", "say less")
    with pytest.warns(UserWarning, match=r"`x` in A is 1, under its cap of 2: lower CAP\['A'\] to 1"):
        _held("CAP", {"A": 2}, {"A": 1}, "`x`", "say less")
    with pytest.warns(UserWarning, match=r"`x` in B is 0, under its cap of 1"):
        _held("CAP", {"A": 2, "B": 1}, {"A": 2}, "`x`", "say less")
    ROOM["CAP A"] = 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _held("CAP", {"A": 2}, {"A": 1}, "`x`", "say less")
        with pytest.warns(UserWarning, match=r"lower CAP\['A'\] to 1"):
            _held("CAP", {"A": 2}, {"A": 0}, "`x`", "say less")
    finally:
        del ROOM["CAP A"]


def test_declarations_reads_names_scopes_and_doc_comments():
    """The source reading the checks share, on a toy file: names under nested and dotted namespaces (`_root_.` aside), a
    section or `mutual` closing without one, a doc comment heading its declaration across attributes but not across
    another declaration, and nothing inside a comment or a string."""
    src = ('namespace A.B\n/-- doc -/\n@[simp]\ndef f : Nat := 1\ndef g : Nat := 2 -- def h\nsection S\n'
           '/- def i -/\nprivate theorem t : True := trivial\nend S\nnamespace C\nstructure D where\n  x : Nat\nend C\n'
           'def s : String := "\\ndef j"\nend A.B\nmutual\n/-- doc -/ inductive E | a\nend\ntheorem _root_.k.l : True := trivial\n'
           '/-! module doc -/\nabbrev m := 0\n')
    got = [(d.name, d.kind, d.line, d.documented, d.private) for d in declarations(src)]
    assert got == [("A.B.f", "def", 4, True, False), ("A.B.g", "def", 5, False, False), ("A.B.t", "theorem", 8, False, True),
                   ("A.B.C.D", "structure", 11, False, False), ("A.B.s", "def", 14, False, False),
                   ("E", "inductive", 17, True, False), ("k.l", "theorem", 19, False, False), ("m", "abbrev", 21, False, False)]


def test_a_read_definition_without_a_doc_comment_is_named():
    """`undocumented` on toy sources: a read definition with no doc comment is named at its line, a field or constructor
    by its structure, and a documented one, an instance, a theorem and a module nothing reads are not."""
    src = ("namespace P\n/-- a -/\ndef a : Nat := 1\ndef b : Nat := 2\nstructure S where\n  x : Nat\n"
           "instance i : Inhabited S := ⟨⟨0⟩⟩\ntheorem t : True := trivial\nend P\n")
    read = {"P.Defs": {"P.a", "P.b", "P.S.x", "P.S.mk", "P.i", "P.t", "P.gone"}, "Q": {"Q.q"}}
    assert undocumented(read, {"P.Defs": ("P/Defs.lean", src)}) == ["P/Defs.lean:4 P.b", "P/Defs.lean:5 P.S"]


def test_a_module_is_long_past_its_2000th_line():
    assert overlong({"a.lean": LONG, "b.lean": LONG + 1, "c.lean": 3}) == {"b.lean": LONG + 1}


def test_an_in_tree_import_all_is_refused_unless_listed_with_its_reason():
    """`imports_all` and `import_all_failures` on toy files: an `import all` the tree resolves fails unless `IMPORT_ALL`
    lists it, with a reason; the toolchain's (`Init`) is no use; an entry its file no longer has fails too, and one whose
    file the tree doesn't have waits."""
    files = {"A.lean": "module\npublic import B\nimport all C.D\nimport all Init.Data.Repr\n", "E.lean": "import all C.D\n"}
    found = imports_all(files, lambda m: m.split(".")[0] in ("B", "C"))
    assert found == ["A.lean: C.D", "E.lean: C.D"]
    assert import_all_failures(found, {"A.lean: C.D": "a by-decide example reads `C.D`'s private table"}, set(files)) == [
        "`import all` of an in-tree module, E.lean: C.D: import it publicly, and move the proof that needs its private part "
        "into it or `@[expose]` the definition that proof reads"]
    assert import_all_failures([], {"A.lean: C.D": "why", "F.lean: G": " ", "H.lean: C.D": "on next"}, {"A.lean", "F.lean"}) == [
        "IMPORT_ALL lists A.lean: C.D, which no file has any more: remove the entry",
        "IMPORT_ALL lists F.lean: G, which no file has any more: remove the entry", "IMPORT_ALL['F.lean: G'] gives no reason"]
