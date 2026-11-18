"""sass_census's dump.sh end to end over a stub cuobjdump that serves the sample listing: a binary dumped one architecture
at a time counts as when dumped whole, and a second run reads its count from the cache and says so."""
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

STUB = """#!{python}
import pathlib, sys
a = sys.argv[1:]
with open({log!r}, "a") as f:
    f.write(" ".join(a[:-1]) + "\\n")
if a[0] in ("--version", "--help"):
    print("stub cuobjdump 1" if a[0] == "--version" else "'sm_90a','sm_100a'")
elif a[0] == "-lelf":
    print("ELF file    1: lib.1.sm_90a.cubin\\nELF file    2: lib.2.sm_100a.cubin")
else:
    text = pathlib.Path({fix!r}).read_text()
    if "-arch" in a:
        arch = a[a.index("-arch") + 1]
        text = "".join("Fatbin elf code:" + s for s in text.split("Fatbin elf code:")[1:] if "arch = " + arch + "\\n" in s)
    sys.stdout.write(text)
"""


def dump(tmp_path: Path, out: Path, cache: Path, stub: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("RESEARCH_RUN")}
    env.update(OUT=str(out), NVCC=str(stub), CUOBJDUMP=str(stub), SASS_CENSUS_CACHE=str(cache), JOBS="2")
    return subprocess.run(["bash", str(DUMP), str(tmp_path / "lib")], env=env, capture_output=True, text=True,
                          check=False)


def test_dump_splits_by_architecture_and_reads_a_cached_count_back(tmp_path):
    log = tmp_path / "calls.log"
    stub = tmp_path / "cuobjdump"
    stub.write_text(STUB.format(python=sys.executable, log=str(log), fix=str(FIX)))
    stub.chmod(0o755)
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "libtest.so").write_bytes(b"\0" * 200_000)
    sha = subprocess.run(["sha256sum", str(tmp_path / "lib" / "libtest.so")], capture_output=True, text=True,
                         check=True).stdout[:16]

    cold = dump(tmp_path, tmp_path / "cold", tmp_path / "cache", stub)
    assert cold.returncode == 0, cold.stderr
    sass = [ln for ln in log.read_text().splitlines() if ln.startswith("-sass")]
    assert sorted(sass) == ["-sass -arch sm_100a", "-sass -arch sm_90a"]
    with FIX.open() as f:
        whole = json.dumps(sass_count.count(f, f"libtest.so@{sha}"))
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
