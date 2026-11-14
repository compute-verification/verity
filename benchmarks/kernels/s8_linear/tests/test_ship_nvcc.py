"""ship_nvcc.sh (the nvcc this directory's build.sh and the other ship builds compile with), without nvcc: a stand-in
toolkit whose nvcc prints a dry run of nvcc's shape and whose gcc, cicc and ptxas record what they are given. What cicc is
handed for one source at two places in a tree is the same to the byte, and a dry run the wrapper can't follow, or a text
that names the file it is in, is refused before cicc runs. job_nvcc.sh (the nvcc a job's own libraries compile with) is that
wrapper in a directory of the call's own. And ship_probe.sh, which a ship build runs before it compiles: with a stand-in
nvcc whose cubin follows its source's path it passes, and it refuses, saying which comparison failed, an nvcc whose cubin
follows no path and a toolkit that hands cicc something of a file's place the wrapper doesn't fix."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SHIP_NVCC = Path(__file__).resolve().parents[2] / "ship_nvcc.sh"
JOB_NVCC = SHIP_NVCC.with_name("job_nvcc.sh")
SHIP_PROBE = SHIP_NVCC.with_name("ship_probe.sh")
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
#: program that holds a string of its own, which names no file of the markers
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


def run(tmp: Path, place: str, inc: str, more: str = "", dry: str = DRY, text: str = TEXT) -> subprocess.CompletedProcess:
    """The wrapper on a source at `place` (a directory under tmp) whose header is spelled `inc` and whose preprocessed text
    is `text`."""
    kit = tmp / "kit"
    src = tmp / place / "t.cu"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(text.format(src=src, inc=inc))
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


def test_a_text_that_names_a_file_of_the_tree_is_refused_before_cicc_runs(tmp: Path) -> None:
    """What __FILE__ leaves in a text: the name its line markers give the file it stands in, or the source's."""
    src = tmp / "a/b/t.cu"
    in_header = TEXT.replace("int f(int);\n", 'int f(int);\nconst char *at = "{inc}";\n')
    in_source = TEXT.replace('"kept/as/written.cuh"', '"{src}"')
    of_source = TEXT.replace("int f(int);\n", 'const char *of = "{src}";\n')
    assert len({TEXT, in_header, in_source, of_source}) == 4
    for text, where in ((in_header, "../h.cuh line 23: "), (in_source, f"{src} line 2: "), (of_source, "../h.cuh line 22: ")):
        r = run(tmp, "a/b", "../h.cuh", text=text)
        assert r.returncode != 0
        assert "the text cicc compiles names a file" in r.stderr and "take it out of " + where in r.stderr
        assert not (tmp / "rec.cicc").exists() and not (tmp / "out.cubin").exists()


def test_a_system_header_may_name_itself(tmp: Path) -> None:
    text = TEXT.replace('# 1 "{inc}" 1\n', '# 1 "/usr/include/x.h" 1 3\nconst char *x = "/usr/include/x.h";\n# 1 "{inc}" 1\n')
    assert text != TEXT
    assert run(tmp, "a/b", "../h.cuh", text=text).returncode == 0
    assert 'const char *x = "/usr/include/x.h";' in given(tmp, "cicc")["input"].splitlines()


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


def test_job_nvcc_is_the_wrapper_in_a_directory_of_the_calls_own(tmp: Path) -> None:
    """A real nvcc names its temporaries under TMPDIR, so the stand-in's dry run here does too."""
    kit = tmp / "kit"
    (kit / "bin/nvcc").write_text('#!/bin/bash\nsed "s|@TMP@|$TMPDIR|g" "$(dirname "$0")/dry.txt" >&2\n')
    seen = []
    for place in ("a/b", "a/much/longer/place/of/it"):
        src = tmp / place / "t.cu"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(TEXT.format(src=src, inc="../h.cuh"))
        (kit / "bin/dry.txt").write_text(DRY.format(kit=kit, tmp="@TMP@", temp=TEMP, ii=f"@TMP@/{TEMP}-5_t.cpp1.ii", src=src,
                                                    out=tmp / "out.cubin", more="", s=" "))
        r = subprocess.run(["bash", str(JOB_NVCC), "-cubin", "-o", str(tmp / "out.cubin"), str(src)],
                           env=dict(os.environ, CUDA=str(kit), TMPDIR=str(tmp / "tmp")), capture_output=True, text=True,
                           check=False)
        assert r.returncode == 0, r.stderr
        cicc = given(tmp, "cicc")
        own = cicc["cwd"].strip()
        assert own.startswith("/tmp/job_nvcc.") and len(own) == len("/tmp/job_nvcc.XXXXXXXX") and not Path(own).exists()
        seen.append((own, cicc["argv"].replace(own, "TMP"), cicc["input"]))
    assert seen[0][0] != seen[1][0] and seen[0][1:] == seen[1][1:]
    assert not list((tmp / "tmp").iterdir())
    r = subprocess.run(["bash", str(JOB_NVCC), "-cubin", "t.cu"], env=dict(os.environ, CUDA=""), capture_output=True, text=True,
                       check=False)
    assert r.returncode != 0 and "the toolkit whose nvcc builds" in r.stderr


#: an nvcc for the probe. Asked for a dry run, it prints DRY for its source and its -o; else it writes a cubin that is
#: @PLAIN@, or fails where that is empty
PROBE_NVCC = """#!/bin/bash
src=; out=; dry=
while [ $# -gt 0 ]; do case $1 in -o) out=$2; shift ;; -dryrun) dry=1 ;; *.cu) src=$1 ;; esac; shift; done
if [ -n "$dry" ]; then sed -e "s|@SRC@|$src|g" -e "s|@OUT@|$out|g" "$(dirname "$0")/dry.txt" >&2; exit 0; fi
[ -n "@PLAIN@" ] || { echo "no such architecture" >&2; exit 1; }
echo "@PLAIN@" > "$out"
"""

#: a preprocessor for the probe: a line marker that names the source by its path, the program, and @MORE@
PROBE_GCC = """#!/bin/bash
src=; out=
while [ $# -gt 0 ]; do case $1 in -o) out=$2; shift ;; *.cu) src=$1 ;; esac; shift; done
{ echo "# 1 \\"$src\\""; echo "int probe;"; @MORE@; } > "$out"
"""
#: what nvcc hashes a source's path into, in the program's own text, where no line marker holds it
OF_THE_PLACE = 'echo "int at_$(dirname "$src" | cksum | cut -d " " -f 1);"'


def probe(tmp: Path, plain: str = "$src", more: str = "true") -> subprocess.CompletedProcess:
    """ship_probe.sh with a stand-in toolkit whose nvcc's own cubin is `plain` and whose preprocessor adds `more`."""
    kit = tmp / "kit"
    for tool, text in (("nvcc", PROBE_NVCC.replace("@PLAIN@", plain)), ("gcc", PROBE_GCC.replace("@MORE@", more))):
        (kit / "bin" / tool).write_text(text)
    (kit / "bin/dry.txt").write_text(DRY.format(kit=kit, tmp=tmp / "tmp", temp=TEMP, ii=tmp / "tmp" / f"{TEMP}-5_t.cpp1.ii",
                                                src="@SRC@", out="@OUT@", more="", s=" "))
    env = dict(os.environ, CUDA=str(kit), TMPDIR=str(tmp / "tmp"))
    return subprocess.run(["bash", str(SHIP_PROBE)], env=env, capture_output=True, text=True, check=False)


def test_the_probe_passes_where_nvcc_follows_the_path_and_the_wrapper_does_not(tmp: Path) -> None:
    r = probe(tmp)
    assert (r.returncode, r.stderr) == (0, "")
    assert given(tmp, "cicc")["input"] == '# 1 "probe.cu"\nint probe;\n'
    assert not list((tmp / "tmp").iterdir()), "the probe's two places are removed"


def test_the_probe_refuses_an_nvcc_whose_cubin_follows_no_path(tmp: Path) -> None:
    r = probe(tmp, plain="one cubin wherever")
    assert r.returncode != 0 and "the first comparison failed" in r.stderr and "second" not in r.stderr
    assert "nvcc itself gives the probe one cubin at two places" in r.stderr
    assert not list((tmp / "tmp").iterdir())


def test_the_probe_refuses_a_wrapper_whose_cubin_follows_the_path(tmp: Path) -> None:
    r = probe(tmp, more=OF_THE_PLACE)
    assert r.returncode != 0 and "the second comparison failed" in r.stderr and "first" not in r.stderr
    assert "ship_nvcc.sh gives the probe two cubins at two places" in r.stderr
    assert not list((tmp / "tmp").iterdir())


def test_the_probe_refuses_a_toolkit_that_does_not_build_it(tmp: Path) -> None:
    r = probe(tmp, plain="")
    assert r.returncode != 0 and "no such architecture" in r.stderr
    assert "nvcc doesn't build the probe" in r.stderr and "comparison" not in r.stderr
