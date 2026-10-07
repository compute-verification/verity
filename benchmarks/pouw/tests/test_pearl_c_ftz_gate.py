"""build.sh's `.FTZ` gate (pearl_c/ftz_gate.py) on nvdisasm text: nvcc's correctly rounded sequences pass whatever registers
ptxas picks; a stray `.FTZ` op, a fast path without its slow-path call, an altered slow path, fast math's approximate forms and an
unpinned nvcc fail.  The pins themselves come from the test kernels (ftz_pins.json; build.sh runs the gate on the real cubins
and a --use_fast_math control), and the fixture's words for those kernels are correctly rounded on subnormals."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "pearl_c"
sys.path.insert(0, str(HERE))

import fixture as F  # noqa: E402
import ftz_gate as G  # noqa: E402
from verity.ml.circuits.silicon import fp32  # noqa: E402

RCP = "$__internal_0_$__cuda_sm20_rcp_rn_f32_slowpath"


def rcp_slow(r: int = 0, extra: tuple[str, ...] = ()) -> list[str]:
    return [f"FFMA R{r + 2}, R{r + 3}, R{r + 4}, -1", f"FADD.FTZ R{r + 2}, -R{r + 2}, -RZ", *extra, f"IMAD.MOV.U32 R{r + 5}, RZ, RZ, R{r + 2}",
            "RET.REL.NODEC R6 `(k)"]


def rcp_inline(r: int = 0, call: bool = True, stray: bool = False) -> list[str]:
    body = ["BSSY B0, `(.L_x_1)", "@P0 BRA `(.L_x_0)", f"MOV R{r + 6}, 0x80"]
    body += [f"CALL.REL.NOINC `({RCP})"] if call else [f"LOP3.LUT R{r + 7}, R{r + 1}, 0x7f, RZ, 0xc0, !PT"]
    body += ["BRA `(.L_x_1)", ".L_x_0:", f"MUFU.RCP R{r + 9}, R{r}", f"FFMA R{r + 2}, R{r + 9}, R{r}, -1",
             f"FADD.FTZ R{r + 2}, -R{r + 2}, -RZ", f"FFMA R{r + 9}, R{r + 9}, R{r + 2}, R{r + 9}", ".L_x_1:", "BSYNC B0"]
    return body + ([f"FADD.FTZ R{r + 3}, R{r + 9}, R{r + 1}"] if stray else []) + ["EXIT"]


def nvdisasm(kernels: dict[str, dict[str, list[str]]]) -> str:
    lines = []
    for k, fns in kernels.items():
        lines.append(f'\t.section\t.text.{k},"ax",@progbits')
        off = 0
        for label, body in fns.items():
            lines.append(f"{label}:")
            for s in body:
                if s.endswith(":"):
                    lines.append(s)
                else:
                    lines.append(f"        /*{off:04x}*/                   {s.replace('`(k)', f'`({k})')} ;")
                    off += 16
    return "\n".join(lines) + "\n"


def cubin(**kernels) -> str:
    base = {"ieee_ops": {"ieee_ops": rcp_inline(), RCP: rcp_slow()}, "ieee_q": {"ieee_q": rcp_inline(), RCP: rcp_slow()}}
    return nvdisasm(base | kernels)


PINS = {"V12.9.86/sm_90a": G.pin_entry(cubin())}


def test_nvcc_s_sequences_pass_whatever_registers_ptxas_picks():
    shipped = {"stats_s5": {"stats_s5": rcp_inline(r=10), RCP: rcp_slow(r=20)}}
    assert G.check(cubin(**shipped), "V12.9.86/sm_90a", PINS) == []
    assert PINS["V12.9.86/sm_90a"]["fast"] == {RCP.split("$")[2]: [list(G.skeleton(rcp_inline()[6:10]))]}


def test_a_stray_ftz_op_fails_even_beside_a_divide():
    problems = G.check(cubin(plainq={"plainq": rcp_inline(stray=True), RCP: rcp_slow()}), "V12.9.86/sm_90a", PINS)
    assert len(problems) == 1 and problems[0].startswith("plainq: `FADD.FTZ R3, R9, R1`")


def test_a_fast_path_without_its_slow_path_call_fails():
    problems = G.check(cubin(norms={"norms": rcp_inline(call=False)}), "V12.9.86/sm_90a", PINS)
    assert problems and all("outside nvcc's correctly rounded" in p for p in problems)


def test_an_altered_slow_path_fails():
    shipped = {"lines": {"lines": rcp_inline(), RCP: rcp_slow(extra=("FMUL.FTZ R9, R9, 0.5",))}}
    problems = G.check(cubin(**shipped), "V12.9.86/sm_90a", PINS)
    assert problems == ["lines: a __cuda_sm20_rcp_rn_f32_slowpath slow path with .FTZ ops that no test kernel pins (4 ops)"]


def test_fast_math_s_approximate_forms_fail():
    fast = ["MUFU.RCP R3, R2", "FMUL.FTZ R4, R1, R3", "FFMA.FTZ R5, R4, R2, R1", "EXIT"]
    problems = G.check(cubin(plainq={"plainq": fast}), "V12.9.86/sm_90a", PINS)
    assert len(problems) == 2 and all(p.startswith("plainq: ") for p in problems)
    problems = G.check(nvdisasm({"ieee_ops": {"ieee_ops": fast}, "ieee_q": {"ieee_q": fast}}), "V12.9.86/sm_90a", PINS)
    assert any(p.startswith("ieee_ops: ") for p in problems) and any("not ftz_pins.json's" in p for p in problems)


def test_an_unpinned_nvcc_fails_closed_and_ftz_free_code_needs_no_pin():
    problems = G.check(cubin(), "V13.0.88/sm_90a", PINS)
    assert problems == ["nvcc V13.0.88/sm_90a: no pinned IEEE sequences, so no .FTZ op is admitted (pinned: ['V12.9.86/sm_90a'])"]
    assert G.check(nvdisasm({"hash_rows": {"hash_rows": ["F2I.FTZ.U32.TRUNC.NTZ R1, R2", "EXIT"]}}), "V13.0.88/sm_120a", PINS) == []


def test_the_pins_are_the_test_kernels_under_cuda_12_9():
    pins, build = G.load_pins(), (HERE / "build.sh").read_text()
    flags = re.search(r'^FLAGS="([^"]+)"$', build, re.M).group(1)
    assert list(pins) == [f"V12.9.86/sm_90a {flags}"] and G.TEST_KERNELS == ("ieee_ops", "ieee_q")
    assert "ftz" not in flags and "fast" not in flags and 'release 12.9' in build and '"$NV/sm_90a $FLAGS"' in build
    entry = pins[f"V12.9.86/sm_90a {flags}"]
    assert set(entry["slow"]) == {"__cuda_sm20_rcp_rn_f32_slowpath", "__cuda_sm20_sqrt_rn_f32_slowpath",
                                  "__cuda_sm3x_div_rn_noftz_f32_slowpath"}
    assert set(entry["fast"]) == {"__cuda_sm20_rcp_rn_f32_slowpath", "__cuda_sm20_sqrt_rn_f32_slowpath"}
    assert len(entry["slow"]["__cuda_sm3x_div_rn_noftz_f32_slowpath"]) == 2       # general, and specialised to 448 / x
    for d in (entry["slow"], entry["fast"]):
        for seqs in d.values():
            assert all(any(G.gated_ftz(re.sub(r"^@!?p ", "", s)) for s in seq) for seq in seqs)
    source = (HERE / "pearl_c.cu").read_text()
    assert all(f"__global__ void __launch_bounds__(128) {k}(const Args a)" in source for k in G.TEST_KERNELS)


def test_the_test_kernels_words_are_correctly_rounded_on_subnormals():
    x, y = F.ieee_inputs()
    ops, q = F.ieee_reference(x, y)
    cov = F.ieee_coverage(x, y, ops)
    assert min(cov["subnormal x"], cov["subnormal y"], cov["subnormal x / y"], cov["subnormal x / 448"],
               cov["subnormal 1 / x"], cov["subnormal bf16 quotient"]) > 50
    for i in range(0, x.size, 37):
        a, b = int(x[i]), int(y[i])
        assert list(ops[i, :5]) == [fp32.div(a, b), fp32.div(F.F448, a), fp32.div(a, F.F448), fp32.div(F.ONE, a), fp32.sqrt(a)]
    assert (q == ops[:, 1]).all() and json.loads(json.dumps(cov)) == cov and np.all((x >> 23) & 0xFF < 0xFF)
