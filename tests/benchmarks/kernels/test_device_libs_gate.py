"""device_libs_gate.py's choice of what the smoke gates, with no toolkit: the shared objects the loader's record says a
device step mapped, never another left in the cache; a host-only module passed over and named; an unreadable one a
failure."""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("device_libs_gate", ROOT / "benchmarks/kernels/device_libs_gate.py")
G = importlib.util.module_from_spec(spec)
sys.modules["device_libs_gate"] = G
spec.loader.exec_module(G)


def elf(path: Path, sections: tuple[str, ...]) -> Path:
    """A 64-bit ELF holding only section headers: a null one, the names, and one of 16 bytes for each of `sections`."""
    names = b"\0" + b"".join(s.encode() + b"\0" for s in (".shstrtab", *sections))
    offs = [names.index(b"\0" + s.encode() + b"\0") + 1 for s in (".shstrtab", *sections)]
    shoff, n = 64 + len(names) + 16, 2 + len(sections)
    heads = [struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
             struct.pack("<IIQQQQIIQQ", offs[0], 3, 0, 0, 64, len(names), 0, 0, 1, 0)]
    heads += [struct.pack("<IIQQQQIIQQ", o, 1, 0, 0, 64 + len(names), 16, 0, 0, 1, 0) for o in offs[1:]]
    head = b"\x7fELF\x02\x01\x01" + bytes(9) + struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, shoff, 0, 64, 0, 0, 64, n, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(head + names + bytes(16) + b"".join(heads))
    return path


def record(dir: Path, *paths: Path) -> Path:
    dir.mkdir(parents=True, exist_ok=True)
    (dir / "ld.4242").write_text("".join(f"     4242:\tfile={p} [0];  generating link map\n" for p in paths))
    return dir


def run(tmp_path: Path, capsys, home: Path, rec: Path) -> tuple[int, list[str]]:
    rc = G.main(["--home", str(home), "--record", str(rec), "--cuda-bin", str(tmp_path / "bin"),
                 "--json", str(tmp_path / "gate.json")])
    return rc, capsys.readouterr().out.splitlines()


def test_only_what_a_step_mapped_is_chosen_and_a_host_module_is_passed_over(tmp_path, capsys):
    home = tmp_path / "home"
    stale = elf(home / ".cache/verity/row_table/old/librow_table_cuda.so", (".nv_fatbin",))
    host = elf(home / ".triton/cache/X/cuda_utils.so", (".text",))
    outside = elf(tmp_path / "lib/libtorch.so", (".nv_fatbin",))
    rec = record(tmp_path / "loaded", host, outside)
    assert G.loaded(rec, home) == [host.resolve()]
    rc, out = run(tmp_path, capsys, home, rec)
    assert rc == 1
    assert f"not gated, no device step mapped it: {stale.resolve()}" in out
    assert f"passed over: {host.resolve()}: no .nv_fatbin section, a host-only module" in out
    assert out[-1] == f"the device steps mapped no device library under {home}"


def test_an_unreadable_library_and_a_missing_record_fail(tmp_path, capsys):
    home = tmp_path / "home"
    rc, out = run(tmp_path, capsys, home, tmp_path / "none")
    assert rc == 1 and out[-1].startswith("no loader record under")
    gone = home / ".cache/verity/row_table/k/libgone_cuda.so"
    rc, out = run(tmp_path, capsys, home, record(tmp_path / "loaded", gone))
    assert rc == 1 and out[-1] == f"unreadable, or not a 64-bit ELF: {gone}"


def test_the_summary_says_pinned_or_unpinned():
    def b(toolkit, unpinned, flagged, covered):
        return {"path": "/h/.cache/verity/row_table/7f3322a00ddf/librow_table_cuda.so", "flagged": flagged, "covered": covered,
                "cubins": [{"tool_notes": [{"tool": "nvlink"}], "toolkit": None, "unpinned": None},
                           {"tool_notes": [{"tool": "ptxas"}], "toolkit": toolkit, "unpinned": unpinned}]}
    assert G.summary(b("13.0.1-sm_90a", None, 6, {"int-seed": 6})) == \
        "librow_table_cuda.so (row_table/7f3322a00ddf) pinned 13.0.1-sm_90a, 6 flagged, covered int-seed 6"
    assert G.summary(b(None, "ptxas ... with unpinned arguments", 0, {})) == \
        "librow_table_cuda.so (row_table/7f3322a00ddf) unpinned, 0 flagged"
