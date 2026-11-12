"""ship_nvcc.sh (the nvcc this directory's build.sh and the other ship builds compile with), without nvcc: a stand-in
toolkit whose nvcc prints a dry run of nvcc's shape and whose gcc, cicc and ptxas record what they are given. What cicc is
handed for one source at two places in a tree is the same to the byte, and a dry run the wrapper can't follow is refused
before cicc runs."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SHIP_NVCC = Path(__file__).resolve().parents[2] / "ship_nvcc.sh"
TEMP = "tmpxft_0000aaaa_00000000"

#: nvcc's dry run for a cubin: its variables, then the preprocessor, cicc, ptxas and an rm of a file it never wrote
DRY = """#$ _NVVM_BRANCH_=nvvm
#$ _SPACE_={s}
#$ CICC_PATH={kit}/nvvm/bin
#$ NVVMIR_LIBRARY_DIR={kit}/nvvm/libdevice
#$ LD_LIBRARY_PATH={kit}/lib:
#$ PATH={kit}/nvvm/bin:{kit}/bin:/usr/bin:/bin
#$ INCLUDES="-I{kit}/include"{s}{s}
#$ gcc -std=c++17 -E -x c++ "-I{kit}/include" -m64 "{src}" -o "{ii}"{s}
#$ "$CICC_PATH/cicc" --c++17 --orig_src_path_name "{src}" -arch compute_90a {more}"{ii}" -o "{tmp}/{temp}-6_t.ptx"
#$ ptxas -arch=sm_90a -m64 -v "{tmp}/{temp}-6_t.ptx" -o "{out}"{s}
#$ rm {tmp}/{temp}-3_t.fatbin
"""

#: the preprocessor's text for the source at SRC including a header spelled INC: line markers, and one line of the
#: program that holds a file's name itself, as __FILE__ leaves it
TEXT = """# 0 "{src}"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 0 "<command-line>" 2
# 1 "{src}"
# 1 "{inc}" 1
# 22 "{inc}"
int f(int);
# 2 "{src}" 2
const char *name = "kept/as/written.cuh";
"""

#: a tool that records its arguments, directory, environment's names and input, and writes its -o
TOOL = """#!/bin/bash
rec=@REC@.$(basename "$0"); mkdir -p "$rec"
printf '%s\\n' "$@" > "$rec/argv"; pwd > "$rec/cwd"; compgen -e | LC_ALL=C sort > "$rec/env"
out=; in=
while [ $# -gt 0 ]; do case $1 in -o) out=$2; shift ;; *.cu|*.ii|*.ptx) in=$1 ;; esac; shift; done
cp "$in" "$rec/input"; cp "$in" "$out"
"""


def toolkit(tmp: Path) -> Path:
    kit = tmp / "kit"
    for tool in ("bin/gcc", "bin/ptxas", "nvvm/bin/cicc"):
        (kit / tool).parent.mkdir(parents=True, exist_ok=True)
        (kit / tool).write_text(TOOL.replace("@REC@", str(tmp / "rec")))
        (kit / tool).chmod(0o755)
    (kit / "bin/nvcc").write_text('#!/bin/bash\ncat "$(dirname "$0")/dry.txt" >&2\n')
    (kit / "bin/nvcc").chmod(0o755)
    (tmp / "tmp").mkdir()
    return kit


def run(tmp: Path, place: str, inc: str, more: str = "", dry: str = DRY) -> subprocess.CompletedProcess:
    """The wrapper on a source at `place` (a directory under tmp) whose header is spelled `inc`."""
    kit = tmp / "kit"
    src = tmp / place / "t.cu"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(TEXT.format(src=src, inc=inc))
    (kit / "bin/dry.txt").write_text(dry.format(kit=kit, tmp=tmp / "tmp", temp=TEMP, ii=tmp / "tmp" / f"{TEMP}-5_t.cpp1.ii",
                                                src=src, out=tmp / "out.cubin", more=more, s=" "))
    env = dict(os.environ, CUDA=str(kit), TMPDIR=str(tmp / "tmp"), NOT_CICCS="1")
    return subprocess.run(["bash", str(SHIP_NVCC), "-cubin", "-o", str(tmp / "out.cubin"), str(src)], env=env,
                          capture_output=True, text=True, check=False)


def given(tmp: Path, tool: str) -> dict[str, str]:
    return {f.name: f.read_text() for f in sorted((tmp / f"rec.{tool}").iterdir())}


@pytest.fixture()
def tmp(tmp_path: Path) -> Path:
    toolkit(tmp_path)
    return tmp_path


def test_cicc_is_handed_the_same_at_two_places_of_a_tree(tmp: Path) -> None:
    seen = []
    for place, inc in (("a/b", "../../verity/definitions/kernels/h.cuh"), ("a/much/longer/place/of/it", "../h.cuh")):
        r = run(tmp, place, inc)
        assert r.returncode == 0, r.stderr
        assert (tmp / "out.cubin").read_text() == given(tmp, "cicc")["input"]
        assert not list((tmp / "tmp").iterdir()), "nvcc's temporaries are removed"
        seen.append(given(tmp, "cicc"))
    assert seen[0] == seen[1]
    cicc = seen[0]
    argv = cicc["argv"].splitlines()
    assert argv[argv.index("--orig_src_path_name") + 1] == "/k/t.cu"
    assert Path(cicc["cwd"].strip()).resolve() == (tmp / "tmp").resolve()
    assert set(cicc["env"].split()) <= {"PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "TMPDIR", "PWD", "SHLVL", "OLDPWD", "_"}
    assert cicc["input"].splitlines() == [
        '# 0 "t.cu"', '# 0 "<built-in>"', '# 0 "<command-line>"', '# 1 "stdc-predef.h" 1 3 4', '# 0 "<command-line>" 2',
        '# 1 "t.cu"', '# 1 "h.cuh" 1', '# 22 "h.cuh"', "int f(int);", '# 2 "t.cu" 2',
        'const char *name = "kept/as/written.cuh";']


def test_the_other_steps_run_as_nvcc_wrote_them(tmp: Path) -> None:
    assert run(tmp, "a/b", "../h.cuh").returncode == 0
    gcc = given(tmp, "gcc")
    assert str(tmp / "a/b/t.cu") in gcc["argv"].splitlines() and "NOT_CICCS" in gcc["env"].split()
    assert gcc["input"] == (tmp / "a/b/t.cu").read_text()


def test_a_cicc_step_naming_another_variable_is_refused_before_cicc_runs(tmp: Path) -> None:
    for more in ('"$NVVMIR_LIBRARY_DIR/libdevice.10.bc" ', '"${NVVMIR_LIBRARY_DIR}/libdevice.10.bc" ', '"$(pwd)" '):
        r = run(tmp, "a/b", "../h.cuh", more=more)
        assert r.returncode != 0
        assert "cicc's step names a variable besides CICC_PATH" in r.stderr and more.strip('" ')[:5] in r.stderr
        assert not (tmp / "rec.cicc").exists() and not (tmp / "out.cubin").exists()


def test_a_dry_run_with_no_cicc_step_is_refused(tmp: Path) -> None:
    r = run(tmp, "a/b", "../h.cuh", dry="".join(x for x in DRY.splitlines(keepends=True) if '/cicc" ' not in x))
    assert r.returncode != 0 and "has no cicc step" in r.stderr
    assert not (tmp / "rec.gcc").exists()
