"""The decoder (`transformer`): every unit it records is its template's function, a tampering decoder's are not, its KV
cache carries exactly the K and V coordinates earlier positions computed, it tracks its float32 reference, its unit counts
are the plan's, core's registration check accepts its written window, the circuit it registers before running is the one
it writes, and it loads no vLLM."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
from verity.ml.circuits.codec import descriptor_id
from verity.ml.windows import interface as IF, rows as CS
from verity_flock import circuit as FC
from verity_torch_loop import serve_transformer as S, transformer as T

LAW = D.parse_law("subset:2")
SH, SF, BF = (descriptor_id(t) for t in T.TEMPLATES)


def served(tamper: bool = False, prompts: tuple[bytes, ...] = (b"Hi",), generate: int = 3):
    loop = T.DecoderLoop(T.Model(), list(prompts), generate, tamper=tamper)
    return loop, loop.window(LAW, cap=16)


def recorded(win: IF.Window, tid: str, op: str) -> list[tuple[dict, list[np.ndarray], np.ndarray]]:
    """The runs of template ``tid``'s units whose op path ends in ``op``: (label, inputs, the one output)."""
    return [(b.label, b.inputs, b.outputs[0]) for b in win._recorded[tid] if b.label["op"].endswith(op)]


@pytest.mark.slow
def test_every_unit_it_records_is_its_template_s_function():
    """~6 s: every unit of one request decoded 3 bytes past a 2-byte prompt (11,048 units), through the reference."""
    loop, win = served()
    assert win.window.population == sum(loop.units.values()) == 11_048
    assert all(win.count(t) == n for t, n in loop.units.items())
    assert win.mismatches() == {}


def test_a_tampering_decoder_s_outputs_are_not():
    _, win = served(tamper=True, generate=1)
    bad = win.mismatches()
    assert {t: len(v) for t, v in bad.items()} == {t: win.count(t) for t in (SH, SF, BF)}


def test_the_kv_cache_carries_the_k_and_v_coordinates_of_earlier_positions():
    """Score unit (t, h, s) reads position s's K coordinates of head h, and PV unit (t, h, d) reads the V coordinate d of
    head h at every position s <= t and +0 past t, then the +0 padding to `PV_K`: the rows a cut wire will read from
    those units."""
    loop, win = served(generate=4)
    qkv = {(lab["pos"], lab["layer"]): out for lab, _, out in recorded(win, SH, "qkv_proj")}
    for lab, (x, w), _ in recorded(win, BF, "qk.h0") + recorded(win, BF, "qk.h1"):
        t, layer, h = lab["pos"], lab["layer"], int(lab["op"][-1])
        cols = slice(T.H + h * T.D, T.H + (h + 1) * T.D)
        assert w.shape == (t + 1, T.D)
        assert all((w[s] == qkv[s, layer][cols]).all() for s in range(t + 1))
        assert (x == qkv[t, layer][h * T.D:(h + 1) * T.D]).all()
    for lab, (_, w), _ in recorded(win, BF, "pv.h0") + recorded(win, BF, "pv.h1"):
        t, layer, h = lab["pos"], lab["layer"], int(lab["op"][-1])
        v = np.stack([qkv[s, layer][2 * T.H + h * T.D:2 * T.H + (h + 1) * T.D] for s in range(t + 1)], axis=1)
        assert w.shape == (T.D, T.PV_K) and (w[:, :t + 1] == v).all() and not w[:, t + 1:].any()


def test_it_tracks_its_float32_reference():
    """Every logits row's cosine with the float32 network's on the same bytes is at least 0.999. Sweep: seeds 0-49, two
    8-byte prompts decoded 8 bytes each (800 rows), worst 0.99985."""
    loop = T.DecoderLoop(T.Model(), [b"Hello, w", b"verity::"], 8)
    loop.window(LAW, cap=16)
    for r, out in enumerate(loop.outputs):
        assert len(out) == 16 and out[:8] == loop.prompts[r]
        ref = T.reference_logits(loop.model, out[:-1])[loop.p - 1:]
        assert torch.nn.functional.cosine_similarity(loop.logits[r].to(torch.float32), ref, dim=-1).min() >= 0.999
        assert list(out[8:]) == loop.logits[r].to(torch.float32).argmax(-1).tolist()


def test_its_unit_counts_are_the_plan_s():
    """note:compute-accounting/20261025T2154Z-report-allp-xfmr: 88,273 units a request at P = 8, G = 24; of them 77,568
    static coordinates (K = 128 and 512), 1,984 scores and 7,936 PV coordinates; 353,092 row and coordinate units a
    window of 4 requests less the 3,140 row units (11 a layer and position, 3 more at a logits position) not yet units."""
    n = T.units(4, 8, 24)
    assert n == {SH: 278_528, SF: 31_744, BF: 7_936 + 31_744}
    rows = 4 * (31 * (1 + 2 * 11) + 24 * 3)
    assert sum(n.values()) + rows == 353_092


def test_core_accepts_its_written_window_and_the_circuit_it_registers_first_is_the_one_it_writes(tmp_path, capsys):
    shape = ["--law", "subset:2", "--prompt", "A", "--generate", "1", "--cap", "64"]
    assert S.main(["--circuit-only", *shape]) == 0
    before = json.loads(capsys.readouterr().out)["circuit"]
    assert S.main(["--out", str(tmp_path), *shape]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["circuit"] == before == hashlib.sha512((tmp_path / "circuit.json").read_bytes()).hexdigest()
    assert got["mismatched"] == {} and got["population"] == sum(T.units(1, 1, 1).values())
    rec = json.loads((tmp_path / "registration.json").read_text())
    w, doc = IF.of_records(tmp_path, LAW, IF.resolve)
    layer = CS.leaf_layer({m.key: (tmp_path / f"m{k}-public.bin").read_bytes() for k, m in enumerate(w.members)})
    v = R.check(rec, w.program, w.query, w.law, CS.SCHEME, CS.expected_ports(w), leaf_layer=layer)
    assert v.ok, v.to_json()
    assert rec["window"]["run"] == got["run"] and [t for t, _ in IF.templates_of(w)] == [SH, SF, BF]


def test_every_template_has_a_row_c_flock_pins_it_by():
    """C-Flock stages a unit only if one of its rows takes a SHA-512 compression before its last block (`circuit.compose`):
    the audit of r20261025-225017-fa7f refused every member while PV's K was 32."""
    for fn in {t for dot in T.BF16_DOTS for t in T.templates(dot)}:
        ins, _ = IF.unit_rows(fn)
        assert any(FC.Port(name, -(-n // 16), n).comps for name, _, n, _ in ins), descriptor_id(fn)
    assert FC.Port("p", 32, 512).comps == 0


def test_its_bf16_gemms_bind_hopper_unless_ampere_is_chosen():
    """Circuits' ruling: the BF16 GEMMs bind `GEMM_HOPPER` (the H200 target). `GEMM_AMPERE` stays selectable, and a decoder
    under it records its own template's function."""
    assert "HopperBF16WgmmaDot16" in BF and T.DecoderLoop(T.Model(), [b"A"], 1).circuit().ids == [SH, SF, BF]
    loop = T.DecoderLoop(T.Model(), [b"A"], 1, dot="ampere")
    amp = descriptor_id(T.dynamic(T.D, "ampere"))
    assert "AmpereBF16TcDot16" in amp and loop.circuit().ids == [SH, SF, amp]
    assert loop.window(LAW).mismatches() == {}
    with pytest.raises(ValueError, match="BF16 step"):
        T.DecoderLoop(T.Model(), [b"A"], 1, dot="sm120")


def test_it_loads_no_vllm():
    code = ("import sys, verity_torch_loop.transformer, verity_torch_loop.serve_transformer; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('verity_vllm', 'vllm')))")
    got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=Path(T.__file__).parents[1])
    assert got.stdout.strip() == "[]"
