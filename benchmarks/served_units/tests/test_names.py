"""window.py's `name()` and `key()` against prove_time.sh's `name()` and `KEY`, which they mirror: a statement window.py's
`prove` looks for under FLOCK_WORK/data/NAME-KEY is the directory prove_time.sh staged it into only while the two agree. The
names are compared on every block type a served card's plan stages, each at the sizes it takes, and on the cases the rule
treats apart (case, characters outside the name's, a trailing dash, the 96-character cut); the key on a stand-in tree, as
files under its hashed paths and outside them change."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from conftest import HERE, load

PT = (HERE / "prove_time.sh").read_text()

_EPS = '{"f64":"0x1.4f8b588e368f1p-17"}'
_MLA = 'DOT={"fn":"HopperBF16WgmmaDot16_v1"},SCALE={"f64":"0x1.0000000000000p-4"}'
_TREE = "[[[[[[0,[1,2]],3],5],6],4],7]"
#: every statement type of a served card's plan, as its UNITS_FILE writes it: stage.py's types (with statics), census
#: names verbatim (the `_v2` forms and the slots forms beside the `_v1`), and the arms' flags
TYPES = [
    "mla-split-part", 'mla-split-part:T=64', "gq:K=6144", "gq:K=2048", "gq:K=1536", "gq:K=256", "fi:K=256", "fi:K=1536",
    "fi:K=2048", "fi:K=6144", "fi-quant", "fi-quant:K=2048", "router-row", "router-slots", "dsa-q-row", "rmsnorm-v8",
    "fp8-coord:K=6144", "fp8-coord:K=6144+public-weights", "fp8-coord:K=6144+packed-weights", "s8-linear:K=2048,N=2",
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


def test_statement_dir():
    """prove_time.sh stages and proves a statement in $FW/data/NAME-KEY, which window.py reads as data/{name}-{key}."""
    assert PT.count("local d=$FW/data/$nm-$KEY") == 2
    assert len(re.findall(r'nm=\$\(name "\$1" \$2\)', PT)) == 2


def tree(root: Path, files: dict[str, str]) -> None:
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)


@pytest.fixture
def repo(tmp_path):
    tree(tmp_path, {"benchmarks/served_units/stage.py": "s", "benchmarks/served_units/window.py": "w",
                    "verity/ml/a.py": "a", "verity/ml/sub/b.py": "b", "verity/ml/c.txt": "c",
                    "integrations/vllm/verity_vllm/d.py": "d", "integrations/vllm/tests/e.py": "e",
                    "tools/verity/circuit_check/src/circuit_check/f.py": "f", "verity/core/g.py": "g"})
    return tmp_path


def test_key_agrees(monkeypatch, repo):
    assign = re.search(r"^KEY=\$\(.*?\)$", PT, re.M | re.S)
    assert assign, "prove_time.sh's KEY is not where this test reads it"
    w = load(monkeypatch, "window.py")
    monkeypatch.setattr(w, "REPO", repo)

    def both() -> str:
        sh = subprocess.run(["bash", "-c", f'REPO="{repo}"\n{assign[0]}\necho $KEY'], capture_output=True, text=True,
                            check=True).stdout.strip()
        assert re.fullmatch(r"[0-9a-f]{16}", sh) and sh == w.key()
        return sh

    k0 = both()
    for f in ("verity/ml/c.txt", "verity/core/g.py", "integrations/vllm/tests/e.py", "benchmarks/served_units/window.py"):
        (repo / f).write_text("changed")
        assert both() == k0, f"{f} is outside the staging tree's key"
    keys = {k0}
    for f in ("benchmarks/served_units/stage.py", "verity/ml/sub/b.py", "integrations/vllm/verity_vllm/d.py",
              "tools/verity/circuit_check/src/circuit_check/f.py"):
        (repo / f).write_text("changed")
        keys.add(both())
    tree(repo, {"verity/ml/new.py": ""})
    keys.add(both())
    assert len(keys) == 6, "a change to a staged source left the key as it was"
