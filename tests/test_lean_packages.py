"""The Lean packages in the tree (AGENTS.md "Lean"): each is audited (a `lean-audit.json` beside its lakefile, which `check`'s
`lean-audit` step reads), and all of them build with one toolchain and one Mathlib, so a bump is one change to all of them."""
from __future__ import annotations

import functools
import importlib.util
import itertools
import json
import os
import re
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".lake", ".git", ".venv", "node_modules", "__pycache__"}


def lake_packages() -> list[Path]:
    out = []
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(x for x in dirs if x not in SKIP)
        if "lakefile.toml" in files or "lakefile.lean" in files:
            out.append(Path(d))
    return out


def test_every_lake_package_is_audited():
    pkgs = lake_packages()
    assert len(pkgs) >= 3
    missing = [str(p.relative_to(ROOT)) for p in pkgs if not (p / "lean-audit.json").exists()]
    assert not missing, f"Lake packages without a lean-audit.json (tools/lean/README.md): {missing}"


def test_one_toolchain():
    pins = {str(p.relative_to(ROOT)): (p / "lean-toolchain").read_text().strip() for p in lake_packages()}
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


def test_the_randomness_assumption_cites_this_toolchain():
    """A3 (C-Flock soundness's `ASSUMPTIONS.md`, `Assumptions.UniformRandomBytes`) cites what the toolchain's runtime reads
    for `IO.getRandomBytes` (`lean_io_get_random_bytes`), so a toolchain bump fails here until someone re-reads it and
    updates the citation."""
    proofs = ROOT / "verity/Security/Proofs"
    version = (proofs / "lean-toolchain").read_text().strip().rpartition(":")[2]
    for doc in ("ASSUMPTIONS.md", "Assumptions.lean"):
        assert f"Lean {version}'s runtime" in (proofs / "Flock/Soundness" / doc).read_text(), doc


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
    """`.gitattributes` sends every `lean-audit.json` to `tools/lean/merge.py`, and each record's `reads` digests are the ones
    the driver recomputes from their definitions."""
    spec = importlib.util.spec_from_file_location("lean_merge", ROOT / "tools" / "lean" / "merge.py")
    merge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge)
    assert f"lean-audit.json merge={merge.NAME}" in (ROOT / ".gitattributes").read_text().splitlines()
    stale = [f"{p.relative_to(ROOT)}: {m}" for p in lake_packages()
             for m, g in json.loads((p / "lean-audit.json").read_text()).get("reads", {}).items()
             if g["digest"] != merge.A.module_digest(merge.A.definitions(g))]
    assert not stale, (f"reads digests that are not the SHA-256 of their definitions (`python tools/lean/merge.py --normalize "
                       f"FILE` recomputes them): {stale}")


SECURITY = ROOT / "verity" / "Security"
PROOFS = SECURITY / "Proofs"


def _requires(pkg: Path) -> list[dict]:
    return tomllib.loads((pkg / "lakefile.toml").read_text()).get("require", []) if (pkg / "lakefile.toml").exists() else []


def test_security_package_requires_only_mathlib():
    """The security analysis's trusted text (`verity/Security`: Definitions and Specs) is what every guarantee is stated
    in, so it requires nothing of the repository and no ArkLib: a path dependency here would put a component or a proof
    under every guarantee."""
    assert SECURITY in lake_packages()
    assert [r["name"] for r in _requires(SECURITY)] == ["mathlib"]


def test_proofs_require_the_security_package():
    """The proofs (`verity/Security/Proofs`, srcDir `..`) prove what `security` states, so they require it by path."""
    assert PROOFS in lake_packages()
    paths = {(PROOFS / r["path"]).resolve() for r in _requires(PROOFS) if "path" in r}
    assert SECURITY in paths
    assert tomllib.loads((PROOFS / "lakefile.toml").read_text()).get("srcDir") == ".."


def test_code_never_requires_the_security_analysis():
    """Code (C-Flock's verifier, the PoUS grader, the warden's program) stays with its component and never imports
    `verity/Security/`: the analysis reads the code, never the reverse (Daniel's layout ruling of Oct 5)."""
    code = [pkg for pkg in lake_packages() if not pkg.is_relative_to(SECURITY) and not pkg.is_relative_to(ROOT / "tools")]
    assert code, "no code packages outside verity/Security"
    bad = [f"{pkg.relative_to(ROOT)} requires {(pkg / r['path']).resolve().relative_to(ROOT)}"
           for pkg in code for r in _requires(pkg) if "path" in r and (pkg / r["path"]).resolve().is_relative_to(SECURITY)]
    assert not bad, bad


def test_every_upstream_watch_is_scanned():
    """The audit's upstream pass reads only a lock's `scan` packages, so a `watch` list with nothing to scan checks none of
    its entries and passes (red-team's B1 on the Lean move: the proofs' lock lost soundness's ArkLib scan). Each lock that
    watches a dependency for its assumptions scans one, and the proofs' lock scans ArkLib."""
    bare = [str(p.relative_to(ROOT)) for p in sorted(ROOT.rglob("lean-audit.json")) if ".lake" not in p.parts
            and (up := json.loads(p.read_text()).get("upstream", {})).get("watch") and not up.get("scan")]
    assert not bare, bare
    assert "Arklib" in json.loads((PROOFS / "lean-audit.json").read_text())["upstream"]["scan"]


#: Libraries only proofs may use: the trusted text (a guarantee's statement, a named assumption) and the zero-knowledge
#: simulator and extractor build on Mathlib alone, so what they say doesn't rest on a proof library's definitions.
PROOF_ONLY = ("ArkLib", "VCVio", "CompPoly")
SOUNDNESS = "Proofs.Flock.Soundness"

#: C-Flock soundness modules that hold trusted text and reach a proof-only library today, without the
#: `Proofs.Flock.Soundness.` prefix. C-Flock's spec is not yet extracted (the record's `reads_exempt`), so its guarantees
#: still read definitions that sit beside ArkLib-based proofs. The list only shrinks: a module that no longer reaches one, or
#: holds no trusted text, fails until its entry goes.
ARKLIB_PENDING = frozenset("""
Assumptions.Zk Audit.Flock Audit.FlockBatched Audit.FlockBatchedAcc Audit.FlockLink Audit.FlockLinked
Audit.FlockPublic Audit.TwoStageExfil Binding.Layout Binding.Leaf Binding.Rows CROnly.Defs CompiledLock
CompiledSound CompiledTable Discharge.Aliased.Copies Discharge.Aliased.Keys Discharge.Composed.Base
Discharge.Composed.Coined Discharge.Composed.Defs Discharge.Composed.DefsJ Discharge.Composed.Inner
Discharge.Composed.Inner.Lincheck Discharge.Composed.Inner.LincheckReplay Discharge.DecodeZL.LIG
Discharge.DecodeZL.OP Discharge.DecodeZL.Statements Discharge.DecodeZL.Wire Discharge.DecodeZLJ.Statements
Discharge.DrawWords.Statements Discharge.Exec.CompiledL Discharge.Exec.Decode Discharge.Exec.Emit
Discharge.Exec.FlockBatchedAccL Discharge.Exec.FlockBatchedL Discharge.Exec.FlockLinkL Discharge.Exec.FramesLigerito
Discharge.Exec.FramesRep Discharge.Exec.Live Discharge.Exec.LiveOn Discharge.Exec.Pts Discharge.Exec.Rounds
Discharge.Exec.SessionBatchAccL Discharge.Exec.SessionBatchL Discharge.Exec.SessionL Discharge.Exec.SessionLDecisive
Discharge.Exec.SessionSoundL Discharge.Exec.Setup Discharge.Exec.Shape Discharge.Exec.StrictCRL
Discharge.Exec.Verdict Discharge.Hidden.Defs Discharge.Hidden.Flat Discharge.Hidden.Wide.Checked
Discharge.Hidden.Wide.Defs Discharge.HiddenDec.Statements Discharge.HiddenExec.Defs Discharge.HiddenExec.Flat
Discharge.HiddenExec.HeadlineExec Discharge.HiddenExec.KeyH Discharge.HiddenExec.Placed Discharge.HiddenExec.Ret
Discharge.HiddenExec.SiteH Discharge.HiddenExec.Sites Discharge.HiddenExec.Statements Discharge.HiddenExec.Tables
Discharge.Hm.Delta Discharge.Hm.Place Discharge.Hm.Port Discharge.Hm.Row Discharge.Integrate.Headline
Discharge.Integrate.Sites Discharge.Layout.HmRegion Discharge.Layout.KeyProg Discharge.Layout.Keyed
Discharge.Layout.KeysRow Discharge.Layout.OfRows Discharge.Layout.OutPin Discharge.Layout.OutRegion
Discharge.Layout.OutShape Discharge.Layout.Outs Discharge.Layout.Pins Discharge.Layout.Readers
Discharge.Layout.RowEnc Discharge.Layout.Scope Discharge.Layout.Site Discharge.Placed.Keyed Discharge.Placed.Setup
Discharge.Placed.Typed Discharge.Placed.Wiring Discharge.PlaysCoin.Statements Discharge.PrivateCircuit.Defs
Discharge.PrivateCircuit.Statements Discharge.Tables.Session Discharge.Tables.Sim Discharge.Tables.Statements
Discharge.Tables.Steps Discharge.Zero.Delta Discharge.Zero.Stack Discharge.ZkBind.At Discharge.ZkBind.Bind
Discharge.ZkBind.Place Discharge.ZkBind.Shadow Discharge.ZkCoins.SessionDefs Discharge.ZkCoins.Statements
Discharge.ZkEveryCoin.Defs Discharge.ZkExec.Coin Discharge.ZkExec.Compiled Discharge.ZkExec.Live
Discharge.ZkExec.LiveStatements Discharge.ZkExec.PlaysStatements Discharge.ZkExec.RunsStatements
Discharge.ZkExec.Shape Discharge.ZkExec.Statements Discharge.ZkExec.VerdictLig Discharge.ZkExec.VerdictStatements
Discharge.ZkHidden.Defs Discharge.ZkHidden.DefsJ Discharge.ZkHidden.SoundBind Discharge.ZkHidden.SoundDefs
Discharge.ZkHidden.Statements Discharge.ZkHidden.View Discharge.ZkLink.Analysis Discharge.ZkLink.BadZ
Discharge.ZkLink.Compiled Discharge.ZkLink.CompiledLink Discharge.ZkLink.DrawOk Discharge.ZkLink.Link
Discharge.ZkLink.LockLevel Discharge.ZkLink.LockRep Discharge.ZkLink.OracleP Discharge.ZkLink.Push
Discharge.ZkLink.SessionZ Discharge.ZkLink.StrictZ Discharge.ZkLink.TablesAt Discharge.ZkReg.Bind
Discharge.ZkReg.BindH Discharge.ZkReg.BindHJ Discharge.ZkReg.Defs Discharge.ZkReg.DefsH Discharge.ZkReg.DefsHJ
Discharge.ZkSession.Algebraic Discharge.ZkSession.Compose Discharge.ZkSession.Inner Discharge.ZkSession.Level0
Discharge.ZkSession.Numbers Discharge.ZkSession.Session Discharge.ZkSession.Table Discharge.ZkView.Defs
Discharge.ZkView.Statements E2E ExecDrawOS ExecFlatCopies ExecStratified Headline Instance KeyedDraw Knowledge
LigeritoPadLevel0 LigeritoPhase LinkSound ListSize PadCode Refine.ArithK Refine.Final Refine.Frames Refine.Ligerito
Refine.LigeritoRun Refine.Live Refine.LiveCompiled Refine.LiveSim Refine.Paths Refine.Realizes Refine.Regions
Refine.Rep Refine.Salted Refine.StmtOf Refine.Table Refine.Tapes Registered.Defs Session SessionBatch
SessionBatchAcc SoundnessPad StrictCR Types.Program Types.ProgramPlaces Types.Units ZK.Masking ZerocheckPhase
""".split())

_IMPORT = re.compile(r"(?:public\s+)?(?:meta\s+)?import\s+(?:all\s+)?(\S+)")
_DEFINES = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable|partial)\s+)*"
                      r"(?:def|abbrev|structure|inductive|class|instance|opaque)\b", re.M)


@functools.cache
def _claims() -> tuple[tuple[Path, str], ...]:
    return tuple(c for pkg in lake_packages() for c in _module_roots(pkg))


def _source(module: str) -> Path | None:
    for src, root in _claims():
        if module == root or module.startswith(root + "."):
            if (path := src / (module.replace(".", "/") + ".lean")).exists():
                return path
    return None


def _modules_under(prefix: str) -> list[str]:
    out = {prefix} if _source(prefix) else set()
    for src, root in _claims():
        if (d := src / prefix.replace(".", "/")).is_dir() and (prefix.startswith(root + ".") or prefix == root):
            out |= {".".join(p.relative_to(src).with_suffix("").parts) for p in d.rglob("*.lean")}
    return sorted(out)


@functools.cache
def _imports(module: str) -> tuple[str, ...]:
    path = _source(module)
    if path is None:
        return ()
    out = []
    for line in re.sub(r"/-.*?-/", "", path.read_text(), flags=re.S).splitlines():
        if not (s := line.strip()) or s.startswith("--") or s in ("module", "prelude"):
            continue
        if not (m := _IMPORT.fullmatch(s)):
            break
        out.append(m.group(1))
    return tuple(out)


@functools.cache
def _proof_only_chain(module: str) -> tuple[str, ...] | None:
    """An import chain from `module` to a proof-only library, read from the `.lean` imports (no build), or None."""
    if module.split(".")[0] in PROOF_ONLY:
        return (module,)
    for i in _imports(module):
        if (chain := _proof_only_chain(i)) is not None:
            return (module, *chain)
    return None


def _chains(modules) -> list[str]:
    return [f"{m}: " + " <- ".join(reversed(_proof_only_chain(m))) for m in sorted(modules)]


def test_zk_simulator_and_extractor_build_on_mathlib():
    """The Goldreich–Kahan simulator, its extractor and the model Theorem GK reads (`ZK/GK/`): every module there that
    declares a definition reaches no proof-only library. The bound's proof (`GK/Theorem.lean`) may."""
    gk = [m for m in _modules_under(f"{SOUNDNESS}.ZK.GK") if _DEFINES.search(_source(m).read_text())]
    assert {f"{SOUNDNESS}.ZK.GK.{m}" for m in ("Rand", "Extract", "Sim", "Model")} <= set(gk), gk
    reach = [m for m in gk if _proof_only_chain(m)]
    assert not reach, "ZK/GK definitions that reach a proof-only library:\n" + "\n".join(_chains(reach))


def test_trusted_text_builds_on_mathlib():
    """Every module whose definitions a guarantee's statement reads (the record's `reads`) or that states a named
    assumption (under its `assumptions`) reaches no proof-only library, but for `ARKLIB_PENDING`, which only shrinks."""
    record = json.loads((SECURITY / "lean-audit.json").read_text())
    trusted = set(record["reads"]) | {m for p in record["assumptions"] for m in _modules_under(p)}
    missing = sorted(m for m in trusted if _source(m) is None)
    assert not missing, f"trusted modules with no source in the tree: {missing}"
    reach = {m for m in trusted if _proof_only_chain(m)}
    pending = {f"{SOUNDNESS}.{m}" for m in ARKLIB_PENDING}
    new = reach - pending
    assert not new, ("trusted text that reaches a proof-only library; move the definitions it reads into a module that "
                     "builds on Mathlib alone (as `ZK/GK/Model.lean` holds Theorem GK's):\n" + "\n".join(_chains(new)))
    stale = sorted(m.removeprefix(SOUNDNESS + ".") for m in pending - reach)
    assert not stale, ("ARKLIB_PENDING entries that no longer reach a proof-only library or hold no trusted text; remove "
                       "them:\n" + textwrap.fill(" ".join(stale), 116))
