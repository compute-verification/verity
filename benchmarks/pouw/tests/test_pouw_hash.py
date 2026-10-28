"""The shared hashing module off the device (benchmarks/pouw/pouw_hash): the host build of pouw_hash.cuh against every
reference (check.py: hashlib, the `blake3` package, verity.ml.crypto.turboshake, Pearl-C's `leaf` and Pearl's ticket), the
checker's rejections by name, the header's domain bytes against the scheme's, and sass.py's opcode census on SASS lines."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "pouw_hash"
sys.path.insert(0, str(HERE))

import check as K  # noqa: E402
import sass as S  # noqa: E402
from verity.ml.protocols.pouw.schemes import pearl_c as C  # noqa: E402


@pytest.fixture(scope="module")
def host_rows(tmp_path_factory) -> list[dict]:
    cxx = shutil.which("g++") or shutil.which("c++")
    if not cxx:
        pytest.skip("no C++ compiler")
    exe = tmp_path_factory.mktemp("pouw_hash") / "pouw_hash_host"
    subprocess.run([cxx, "-O1", "-std=c++17", "-ffp-contract=off", "-x", "c++", "-o", str(exe), str(HERE / "pouw_hash_check.cu")],
                   check=True)
    out = subprocess.run([str(exe), "check"], check=True, capture_output=True, text=True).stdout
    return [json.loads(line) for line in out.splitlines()]


def test_host_build_matches_every_reference(host_rows):
    [summary] = K.check(host_rows)
    assert summary["ok"], summary
    assert summary["rows"] == summary["expected"] == len(K.expected())


def test_check_rejects_a_wrong_missing_or_unknown_row(host_rows):
    rows = [dict(r) for r in host_rows if r["what"] != "quad lane 7"]
    leaf = next(r for r in rows if r["what"] == "pearl-c leaf quad")
    leaf["out"] = ("0" if leaf["out"][0] != "0" else "1") + leaf["out"][1:]
    rows.append({**rows[0], "what": "turboshake128 bytes", "len": 3})
    [summary] = K.check(rows)
    assert not summary["ok"]
    assert summary["bad"] == [["pearl-c leaf quad", 4096, 0x22]]
    assert summary["missing"] == [["quad lane 7", 256, 0x21]]
    assert summary["extra"] == [["turboshake128 bytes", 3, rows[0]["d"], len(rows[0]["out"]) // 2]]


def test_against_holds_device_rows_to_the_host_rows(host_rows):
    want = {K.row_key(r): r["out"] for r in host_rows}
    device = [{**r, "where": "device"} for r in host_rows]
    assert K.check(device, want)[0]["ok"]
    device[5] = {**device[5], "out": "00" * 32}
    assert K.check(device, want)[0]["bad"] == [list(K.row_key(device[5]))]


def test_domain_bytes_are_the_schemes():
    src = (HERE / "pouw_hash.cuh").read_text()
    m = re.search(r"constexpr uint8_t D_MESSAGE = (0x[0-9A-F]+), D_LEAF = (0x[0-9A-F]+), D_ROWS = (0x[0-9A-F]+);", src)
    d_rows = 0x23                     # h0's rows domain byte: -h1 commits A's rows with BLAKE3 and has none
    assert [int(x, 16) for x in m.groups()] == [C.D_MESSAGE, C.D_LEAF, d_rows]
    assert re.search(r"constexpr int TILE = 64, CHUNK = (\d+)", src).group(1) == str(C.CHUNK)


def test_sass_census_splits_iadd3_and_gates_the_op_kernels():
    sass = """
        Function : _Z4k_opILi0EEvxP3RecPm
        /*0160*/                   IADD3 R4, PT, PT, R2.reuse, R4, RZ ;               /* 0x0000000402047210 */
        /*0170*/                   IADD3 R3, PT, PT, R5, R8, R4 ;                     /* 0x0000000f02067210 */
        /*0180*/               @P0 LOP3.LUT R2, R3, R0, R5, 0x96, !PT ;               /* 0x0000000d02087210 */
        /*0190*/                   SHF.L.W.U32.HI R3, R6, 0x7, R6 ;                   /* 0x00000009020a7210 */
    """
    [(name, ops)] = S.census(sass).items()
    assert name == "_Z4k_opILi0EEvxP3RecPm"
    assert ops == {"IADD3.RZ": 1, "IADD3": 1, "LOP3": 1, "SHF": 1}
    kernels = {name: {"ops": dict(ops)}}
    assert S.gate(kernels) == [f"{name}: 1 IADD3.RZ, want at least {S.OP_PER_ITER}"]
    kernels[name]["ops"]["IADD3.RZ"] = S.OP_PER_ITER
    kernels["_Z5k_msgxP3RecPm"] = {"ops": {"LDL": 2}, "stack": 16}
    assert S.gate(kernels) == ["_Z5k_msgxP3RecPm: local memory (stack 16, LDL 2, STL 0)"]
    del kernels["_Z5k_msgxP3RecPm"]
    kernels["_Z5k_mmaxP3RecPm"] = {"ops": {"QMMA": S.MMA_PER_ITER - 1}}
    assert S.gate(kernels) == [f"_Z5k_mmaxP3RecPm: {S.MMA_PER_ITER - 1} QMMA, want at least {S.MMA_PER_ITER}"]
