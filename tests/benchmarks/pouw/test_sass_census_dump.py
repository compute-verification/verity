"""sass_census's dump.sh end to end over a stub cuobjdump that serves the sample listing: a binary dumped one run of its
ELFs at a time counts as when dumped whole, and a second run reads its count from the cache and says so."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import sass_count

ROOT = Path(__file__).resolve().parents[3]
DUMP = ROOT / "calibration" / "pouw" / "sass_census" / "dump.sh"
FIX = Path(__file__).resolve().parent / "fixtures" / "sass_census_sample.sass"

#: a cuobjdump whose binary holds three ELFs, the sample's two architectures and its first again, each named, as
#: cuobjdump names them, after the path it was given
STUB = """#!{python}
import pathlib, sys
a = sys.argv[1:]
with open({log!r}, "a") as f:
    f.write(" ".join(a[:-1] + [pathlib.Path(a[-1]).name]) + "\\n")
head, *secs = pathlib.Path({fix!r}).read_text().split("Fatbin elf code:")
elf = [head + "Fatbin elf code:" + secs[0]] + ["Fatbin elf code:" + s for s in (secs[1], secs[0])]
names = [f"{{pathlib.Path(a[-1]).name}}.{{i}}.{{arch}}.cubin" for i, arch in enumerate(["sm_90a", "sm_100a", "sm_90a"], 1)]
if a[0] == "--version":
    print("stub cuobjdump 1")
elif a[0] == "-lelf":
    print("".join(f"ELF file    {{i}}: {{n}}\\n" for i, n in enumerate(names, 1)), end="")
elif a[0] == "-xelf":
    for i, (n, t) in enumerate(zip(names, elf), 1):
        print(f"Extracting ELF file    {{i}}: {{n}}")
        pathlib.Path(n).write_text(t)
elif a[-1].endswith(".cubin"):
    sys.stdout.write(pathlib.Path(a[-1]).read_text())
else:
    sys.stdout.write("".join(elf))
"""


def dump(tmp_path: Path, out: Path, cache: Path, stub: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("RESEARCH_RUN")}
    env.update(OUT=str(out), NVCC=str(stub), CUOBJDUMP=str(stub), SASS_CENSUS_CACHE=str(cache), JOBS="2")
    return subprocess.run(["bash", str(DUMP), str(tmp_path / "lib")], env=env, capture_output=True, text=True,
                          check=False)


def test_dump_splits_a_binary_into_its_elfs_and_reads_a_cached_count_back(tmp_path):
    log = tmp_path / "calls.log"
    stub = tmp_path / "cuobjdump"
    stub.write_text(STUB.format(python=sys.executable, log=str(log), fix=str(FIX)))
    stub.chmod(0o755)
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "libtest.so").write_bytes(b"\0" * 200_000)
    # a second path to the same binary, whose ELFs cuobjdump would name otherwise
    (tmp_path / "lib" / "libtest.so.1").symlink_to("libtest.so")
    sha = subprocess.run(["sha256sum", str(tmp_path / "lib" / "libtest.so")], capture_output=True, text=True,
                         check=True).stdout[:16]

    cold = dump(tmp_path, tmp_path / "cold", tmp_path / "cache", stub)
    assert cold.returncode == 0, cold.stderr
    sass = [ln for ln in log.read_text().splitlines() if ln.startswith("-sass")]
    assert sorted(sass) == ["-sass libtest.so.1.sm_90a.cubin", "-sass libtest.so.2.sm_100a.cubin",
                            "-sass libtest.so.3.sm_90a.cubin"], sass
    assert "dumped whole" not in cold.stdout + cold.stderr
    head, *secs = FIX.read_text().split("Fatbin elf code:")
    text = head + "".join("Fatbin elf code:" + s for s in (secs[0], secs[1], secs[0]))
    whole = json.dumps(sass_count.count(text.splitlines(keepends=True), f"libtest.so@{sha}"))
    assert (tmp_path / "cold" / "counts" / f"{sha}.json").read_text() == whole
    assert not (tmp_path / "cold" / "cached.tsv").exists() and "read from the cache" not in cold.stdout

    log.write_text("")
    warm = dump(tmp_path, tmp_path / "warm", tmp_path / "cache", stub)
    assert warm.returncode == 0, warm.stderr
    assert not [ln for ln in log.read_text().splitlines() if ln.startswith("-sass")]
    assert f"libtest.so@{sha} read from the cache" in warm.stdout
    assert (tmp_path / "warm" / "cached.tsv").read_text().split("\t")[0] == sha
    assert (tmp_path / "warm" / "census.json").read_bytes() == (tmp_path / "cold" / "census.json").read_bytes()
    [m] = json.loads((tmp_path / "warm" / "result.json").read_text())["measurements"]
    assert m["from_cache"] == [f"libtest.so@{sha}"]
