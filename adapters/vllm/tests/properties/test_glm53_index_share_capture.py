"""GLM-5.3's IndexShare capture without a GPU (`glm53_index_share_capture_gpu`): it is a registered property; and its `main` end to
end with the GPU stood in for by the twin's words, through the spawned worker pool to the sealed record, the GO / NO-GO line and
the exit status, so a slip between the comparisons and the record shows here and not in the card's lease."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import json
import sys
import types

import numpy as np
import pytest
from verity.definitions.numerics import dsa_norm_rows as NR

from tests.properties import glm53_index_share_capture_gpu as IS
from verity_vllm.records.property_record import REGISTRY, digest_of, seal

FR = IS.FR
FAKE_PTX = ".version 8.7\n.target sm_90a\n.visible .entry {k}(\n)\n.maxntid 128\n{{\n\tret;\n}}\n"


def test_the_capture_is_a_registered_property() -> None:
    assert REGISTRY["glm53_index_share_capture"].schema == IS.SCHEMA and IS.RECORD == "glm53_index_share_capture.json"
    doc = seal({"schema": IS.SCHEMA, "gate": {"ok": False}}, False)
    assert doc["ok"] is False and doc["digest"] == digest_of(doc)


def _stub(monkeypatch, flip: bool, rope_definition: str) -> None:
    monkeypatch.setattr(IS.C, "gate", lambda *a: {"ok": True})
    monkeypatch.setattr(IS.G, "_environment", lambda: {})
    monkeypatch.setattr(IS.DS, "environment", lambda work: ({"TRITON_CACHE_DIR": str(work / "triton_cache")}, {}))
    monkeypatch.setattr(IS.DS, "served", lambda: {"ok": True})
    monkeypatch.setattr(FR, "SAMPLE", (0,))
    monkeypatch.setattr(FR, "SPECIAL", ())

    def gpu(work, doc):
        rng = np.random.default_rng(5)
        doc["triton"], doc["selection"] = "stub", {"ok": True}
        probes = {norm: FR.probe(N, FR.EPS, 20261013 + N) for norm, N, _, _ in FR.NORMS}
        doc["probes"] = {norm: {"rows": len(p["x"]), "targets": p["targets"]} for norm, p in probes.items()}
        jobs = []
        for case in FR.CASES:
            saved, probe_rows = {}, {}
            for norm, N, _, _ in FR.NORMS:
                if case == "served":
                    x, w = np.stack([FR.bf16_row(rng, N, 1.0) for _ in range(4)]), FR.weight_row(rng, N)
                else:
                    x, w = probes[norm]["x"], probes[norm]["w"]
                    probe_rows[norm] = len(x)
                out = NR.dsa_rmsnorm_rows(x, w, FR.EPS)["out"]
                if flip and case == "served" and norm == "kv_c":
                    out[1, 3] ^= 1
                saved[norm] = {"x": x, "w": w, "out": out}
            jobs += FR.jobs_for(work, case, saved, probe_rows)
            doc["launches"].append({"case": case, "tokens": 4, "checks": {"deterministic": True, "topk_untouched": True}})
        doc["launches"] += [{"case": c, "tokens": 3, "checks": {"deterministic": True}} for c in ("q_served", "rope_probe")]
        doc["rope"] = {k: {"definition": rope_definition, "words": 64} for k in ("k_pe", "q_pe")}
        doc["rope"]["ok"] = all(doc["rope"][k]["definition"] == IS.ROPE_DEFINITION for k in ("k_pe", "q_pe"))
        cache = work / "triton_cache" / "k"
        cache.mkdir(parents=True)
        for k in IS.DS.KERNELS:
            (cache / f"{k}.ptx").write_text(FAKE_PTX.format(k=k))
        return jobs

    monkeypatch.setattr(IS, "gpu", gpu)
    monkeypatch.setitem(sys.modules, "triton", types.SimpleNamespace(__file__="/nonexistent/triton/__init__.py"))


@pytest.mark.slow
@pytest.mark.parametrize(("flip", "rope_definition", "go"), [(False, "RoPEHeadGptJ_v2", True), (True, "RoPEHeadGptJ_v2", False),
                                                            (False, "RoPEHeadGptJ_v1", False)])
def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys, flip: bool, rope_definition: str, go: bool) -> None:
    """`main` with the GPU stood in for (the twin's words as the kernel's on four served rows a norm and the real RMSNorm probes,
    fake PTX): the jobs through the spawned worker pool, the summary, the sealed record, the verdict line and the exit status; a
    flipped kv_c word and a RoPE settled on another Definition each make it NO-GO.  Slow (the probes' search and the reference
    evaluator of `DsaRMSNorm_v1` at N 2048 on the probe rows): it is the only test that seals this capture's own document, which
    the card's run otherwise does first."""
    _stub(monkeypatch, flip, rope_definition)
    out = tmp_path / "out"
    assert IS.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]) == (0 if go else 1)
    doc = json.loads((out / IS.RECORD).read_text())
    assert doc["schema"] == IS.SCHEMA and doc["digest"] == digest_of(doc) and doc["ok"] is go
    assert doc["ptx"]["ok"] and doc["ptx"]["e4m3_cvt"] == {k: [0] for k in IS.DS.KERNELS} and doc["bind"]["ok"]
    cases = doc["summary"]["cases"]
    assert {k: t["rows"] for k, t in cases.items() if k.startswith("served/")} == {"served/q_c": 4, "served/kv_c": 4}
    assert cases["served/kv_c"]["mismatch"] == int(flip) and cases["served/q_c"]["mismatch"] == 0
    assert all(n >= FR.PER_ALT for alts in doc["summary"]["refuted"].values() for n in alts.values())
    printed = capsys.readouterr().out
    assert f"GLM53-INDEX-SHARE {'GO' if go else 'NO-GO'} {doc['digest']}" in printed
    assert f"ROPE k_pe={rope_definition} q_pe={rope_definition}" in printed
