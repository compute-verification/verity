"""window.py's `name()` and `key()` against prove_time.sh's `name()` and `KEY`, which they mirror: a statement window.py's
`prove` looks for under FLOCK_WORK/data/NAME-KEY is the directory prove_time.sh staged it into only while the two agree. The
names are compared on every block type a served card's plan stages, each at the sizes it takes, and on the cases the rule
treats apart (case, characters outside the name's, a trailing dash, the 96-character cut); the key on a stand-in tree, as
files under its hashed paths and outside them change. The paths each key hashes (its KEY_PATHS) are held to this tree, and
on a stand-in tree without one of them, or with a file its pipeline fails on, each key refuses: a digest of what is left
would keep its value across a change to what is missing."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import HERE, load

PT = (HERE / "prove_time.sh").read_text()
ROOT = HERE.parent.parent
#: what both refusals of a tree without one of KEY_PATHS say to do
FIX = "correct KEY_PATHS in benchmarks/served_units/prove_time.sh and window.py together"

_EPS = '{"f64":"0x1.4f8b588e368f1p-17"}'
_MLA = 'DOT={"fn":"HopperBF16WgmmaDot16_v1"},SCALE={"f64":"0x1.0000000000000p-4"}'
_TREE = "[[[[[[0,[1,2]],3],5],6],4],7]"
#: every statement type of a served card's plan, as its UNITS_FILE writes it: stage.py's types (with statics), census
#: names verbatim (the `_v2` forms and the slots forms beside the `_v1`), and the arms' flags
TYPES = [
    "mla-split-part", 'mla-split-part:T=64', "gq:K=6144", "gq:K=2048", "gq:K=1536", "gq:K=256", "fi:K=256", "fi:K=1536",
    "fi:K=2048", "fi:K=6144", "fi-quant", "fi-quant:K=2048", "router-row", "router-slots", "dsa-q-row", "rmsnorm-v8",
    "fp8-coord:K=6144", "fp8-coord:K=6144+public-weights", "fp8-coord:K=6144+packed-weights", "s8-coord:K=2048",
    *(f"census:SparseMlaHeadBlock_{v}{{DR=64,DV=512,NVIS={nv},FIRST={f},BN=64,{_MLA.replace('_v1', '_' + v)}}}"
      for v in ("v1", "v2") for nv, f in ((64, "false"), (63, "false"), (48, "false"), (32, "false"), (1, "true"))),
    'census:SparseMlaHeadEnd_v1{DV=512,INV={"fn":"Fa3InvSum_v1"}}',
    *(f'census:SparseMlaCombine_v1{{NSPLIT={k},DV=512,INV={{"fn":"Fa3InvSum_v1"}}}}' for k in (5, 8, 16)),
    'census:SparseMlaSplitPart_v2{T=64,DR=64,DV=512,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v2"},INV={"fn":"Fa3InvSum_v1"},'
    'SCALE={"f64":"0x1.0000000000000p-4"}}',
    'census:MoeExpertFp8CoordinateChunk_v1{E=256,K=512,G=128,KB=128,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v1"},'
    'FIRST=false,LAST=false}',
    'census:MoeExpertFp8GemmW_v1{E=256,K=256,N=1,G=128,KB=128,SWAP=true,DOT={"fn":"HopperE4m3QgmmaDot32_v1"}}',
    *(f'census:BatchedGemm_v1{{B=1,K={k},N=1,DOT={{"fn":"HopperBF16WgmmaDot16_v1"}}}}' for k in (512, 192)),
    'census:CublasSplitKGemm_v1{K=6144,N=1,SPLIT_K=4,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}',
    'census:CublasSplitKGemmF32Out_v1{K=6144,N=1,SPLIT_K=3,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}',
    *(f"census:DsaRMSNorm_{v}{{N={n},EPS={_EPS}}}" for v in ("v1", "v2") for n in (2048, 512)),
    f"census:RmsNormMnnvl_v1{{N=6144,SHOT=1,CLUSTER=8,EPS={_EPS},FMA=true}}",
    "census:AllReduceMnnvl_v1{WORLD=8,N=3072,SHOT=1}", "census:AllReduceResidualMnnvl_v1{WORLD=8,N=3072,SHOT=1}",
    f"census:AllReduceMultimem_v1{{WORLD=8,SIZE=3072,TREES=[{','.join([_TREE] * 8)}],FTZ=false}}",
    "census:EmbeddingShard_v1{VS=19360,H=1,START=0}", "census:LlBf16DotprodGemm_v1{N=1,K=6144,BS=128,VEC=8,FMA=1}",
    'census:Gemm_v2{K=6144,N=1,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}',
    *(f"census:FlashInferFp8Linear_v2{{K={k},N=1}}" for k in (2048, 6144)),
    *(f"census:ScaledMmFp8BlockSharedScale_v1{{K={k},N=1,G=128}}" for k in (1536, 256)),
    *(f"census:DsaMqaLogits_v3{{N={n},H=32,D=128}}" for n in (1, 30)),
    'census:DsaIndexerKRow_v1{D=128,ROPE=64,EPS={"f64":"0x1.0c6f7a0b5ed8dp-20"}}', "census:DsaTopKShortRow_v1{TOPK=2048}",
    "census:MoeSum_v1{TOPK=8,H=3072}", f"census:RMSNormCudaVector8_v1{{N=6144,EPS={_EPS},BLOCK=256}}",
    "census:SiluMul_v2{I=256}", "census:SiluMul_v2{I=1536}", "census:RoPEHeadGptJ_v2{D=64}",
    "census:TopPStep_v1{V=154880,S=4,N=2048,FIRST=false,COMBINE=-1}", "census:TopPStatsHead_v1{N=5952,FIRST=false}",
    "census:TopPStatsExp_v1{N=2976,SPT=4,AT=0,LAST=true,FIRST=false}",
    "census:GumbelTopPSelect_v1{V=154880,N=256,FIRST=false,LAST=false}",
]
#: the rule's own cases: a trailing dash, runs of characters outside the name's, upper case, and names of 96 and 97
#: characters (cut to 80 and a digest past 96)
EDGES = ["fi:K=", "a::{}[]\"\"b", "ABC_def", "x" * 96, "x" * 97, "census:" + "y" * 89, "census:" + "y" * 90, "-lead", "."]
NS = ["full", "full/4", "full/16", "2", "16"]


def bash_names(pairs: list[tuple[str, str]]) -> list[str]:
    fn = re.search(r"^name\(\) \{\n.*?\n\}\n", PT, re.M | re.S)
    assert fn, "prove_time.sh's name() is not where this test reads it"
    script = fn[0] + 'while IFS=$\'\\t\' read -r u n; do name "$u" $n; done\n'
    feed = "".join(f"{u}\t{n}\n" for u, n in pairs)
    return subprocess.run(["bash", "-c", script], input=feed, capture_output=True, text=True, check=True).stdout.splitlines()


def test_names_agree(monkeypatch):
    w = load(monkeypatch, "window.py")
    pairs = [(u, n) for u in TYPES + EDGES for n in NS]
    assert bash_names(pairs) == [w.name(u, n) for u, n in pairs]
    long = [u for u in TYPES if len(w.name(u, "full")) > 96 + len("-nfull")]
    assert not long, f"names past the cut: {long}"
    assert len({w.name(u, n) for u, n in pairs if u in TYPES}) == len(TYPES) * len(NS), "two statements share a directory"


def test_arms():
    """prove_time.sh's stage() hands stage.py each arm's flags and the type without the arm's suffix; a `+` inside a type's
    statics is no arm."""
    arm = re.search(r"^  local u=\$\{1%\+\*\} flag= [^\n]*\n  case \$1 in .*? esac\n", PT, re.M | re.S)
    assert arm, "prove_time.sh's stage() does not read its arms where this test reads them"
    script = "name() { :; }\narm() {\n" + arm[0] + '  echo "$u|$flag"\n}\nwhile IFS= read -r u; do arm "$u" 2; done\n'
    plus = 'census:Scaled_v1{S={"f64":"0x1.0000000000000p+4"}}'
    want = {"gq:K=6144": "gq:K=6144|", "fp8-coord:K=6144+public-weights": "fp8-coord:K=6144|--public-weights",
            "fp8-coord:K=6144+packed-weights": "fp8-coord:K=6144|--public-weights --packed",
            plus: f"{plus}|", f"{plus}+public-weights": f"{plus}|--public-weights"}
    got = subprocess.run(["bash", "-c", script], input="".join(f"{u}\n" for u in want), capture_output=True, text=True,
                         check=True).stdout.splitlines()
    assert got == list(want.values())


def test_statement_dir():
    """prove_time.sh stages and proves a statement in $FW/data/NAME-KEY, which window.py reads as data/{name}-{key}."""
    assert PT.count("local d=$FW/data/$nm-$KEY") == 2
    assert len(re.findall(r'nm=\$\(name "\$1" \$2\)', PT)) == 2


def tree(root: Path, files: dict[str, str]) -> None:
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)


#: the stand-in tree: files under each of the key's paths, and outside them
FILES = {"benchmarks/served_units/stage.py": "s", "benchmarks/served_units/window.py": "w",
         "verity/circuits/a.py": "a", "verity/definitions/sub/b.py": "b", "verity/commitments/c.txt": "c",
         "verity/protocols/p.py": "p",
         "adapters/vllm/verity_vllm/d.py": "d", "adapters/vllm/tests/e.py": "e",
         "tools/verity/circuit_check/src/circuit_check/f.py": "f", "verity/core/g.py": "g"}


@pytest.fixture
def repo(tmp_path):
    tree(tmp_path, FILES)
    return tmp_path


def sh_key_paths() -> list[str]:
    paths = re.search(r'^KEY_PATHS="([^"\n]+)"$', PT, re.M)
    assert paths, "prove_time.sh's KEY_PATHS is not where this test reads it"
    return paths[1].split()


def sh_key(repo: Path) -> subprocess.CompletedProcess:
    """prove_time.sh's own lines, from KEY_PATHS to the end of KEY's assignment, run in `repo` under the script's shell
    options: the key is its output's last line where they compute one."""
    opts = re.search(r"^set -\w+ pipefail$", PT, re.M)
    lines = re.search(r"^KEY_PATHS=.*?^KEY=\$\(.*?\n(?=\n)", PT, re.M | re.S)
    assert opts and lines, "prove_time.sh's KEY is not where this test reads it"
    return subprocess.run(["bash", "-c", f'{opts[0]}\nREPO="{repo}"\n{lines[0]}echo $KEY'], capture_output=True, text=True)


def test_key_agrees(monkeypatch, repo):
    w = load(monkeypatch, "window.py")
    monkeypatch.setattr(w, "REPO", repo)

    def both() -> str:
        r = sh_key(repo)
        assert r.returncode == 0, r.stdout + r.stderr
        sh = r.stdout.strip()
        assert re.fullmatch(r"[0-9a-f]{16}", sh) and sh == w.key()
        return sh

    k0 = both()
    for f in ("verity/commitments/c.txt", "verity/core/g.py", "adapters/vllm/tests/e.py", "benchmarks/served_units/window.py"):
        (repo / f).write_text("changed")
        assert both() == k0, f"{f} is outside the staging tree's key"
    keys = {k0}
    for f in ("benchmarks/served_units/stage.py", "verity/definitions/sub/b.py", "adapters/vllm/verity_vllm/d.py",
              "tools/verity/circuit_check/src/circuit_check/f.py"):
        (repo / f).write_text("changed")
        keys.add(both())
    tree(repo, {"verity/protocols/new.py": ""})
    keys.add(both())
    assert len(keys) == 6, "a change to a staged source left the key as it was"


def test_key_paths_in_tree(monkeypatch):
    """Each key's paths, read from prove_time.sh's KEY_PATHS and window.py's, are one list, and this tree holds each. A
    rename that leaves an old name in them would take what was under it out of the key."""
    sh, py = sh_key_paths(), load(monkeypatch, "window.py").KEY_PATHS.split()
    assert sh == py, f"prove_time.sh's KEY_PATHS {sh} and window.py's {py} differ: {FIX}"
    gone = [p for p in sh if not (ROOT / p).exists()]
    assert not gone, (f"KEY_PATHS names {gone}, which this tree does not hold: {FIX}, naming every path that now holds the "
                      "Python that was under it, and this test's FILES with them")


def test_key_refuses_a_missing_path(monkeypatch, tmp_path):
    """On a stand-in tree without one of the key's paths, each of them in turn, prove_time.sh's lines stop before any key and
    window.py's key() raises, each naming the path and the fix."""
    w = load(monkeypatch, "window.py")
    for i, gone in enumerate(sh_key_paths()):
        repo = tmp_path / str(i)
        tree(repo, FILES)
        monkeypatch.setattr(w, "REPO", repo)
        whole = sh_key(repo)
        assert whole.returncode == 0 and whole.stdout.strip() == w.key(), f"FILES, whole: {whole.stdout}{whole.stderr}"
        if (repo / gone).is_dir():
            shutil.rmtree(repo / gone)
        else:
            (repo / gone).unlink()
        why = f"no {gone} in {repo}, a path of the staging key: {FIX}"
        r = sh_key(repo)
        assert r.returncode == 1 and r.stdout.strip() == f"prove_time.sh: {why}", r.stdout + r.stderr
        with pytest.raises(SystemExit) as stop:
            w.key()
        assert str(stop.value) == f"window.py: {why}"


def test_key_refuses_a_failed_pipeline(monkeypatch, repo):
    """Each key reads its pipeline's status: where a step before the last fails (here xargs, at the unmatched quote in a
    file's name), prove_time.sh's lines stop and window.py's key() raises, rather than keep the digest of what got through."""
    w = load(monkeypatch, "window.py")
    monkeypatch.setattr(w, "REPO", repo)
    tree(repo, {"verity/circuits/it's.py": ""})
    r = sh_key(repo)
    assert r.returncode == 1 and r.stdout.startswith("prove_time.sh: no staging key: "), r.stdout + r.stderr
    with pytest.raises(subprocess.CalledProcessError):
        w.key()
