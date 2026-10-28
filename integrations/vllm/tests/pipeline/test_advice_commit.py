"""`verity-vllm advice` commits the advice and publishes only its binding, never the advice, on run A's
advice (`fixtures/advice_run_a`: a recorded step trace and Commit record, Qwen3-8B on one GPU): the root is the one
`advice_windows.commit_advice` and `verity.ml.commitments.advice` give for I1's encoding, the registration carries the root and the
bit count and no byte of the advice, and one changed advice byte changes the root."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from verity.core.protocols.certifier import advice as I1
from verity.ml.commitments import advice as AC
from verity_vllm.pipeline import advice as ADV
from verity_vllm.pipeline import advice_windows as AW

RUN_A = Path(__file__).parent / "fixtures" / "advice_run_a"
RUN_A_BITS = 71801
C = ADV.registered_compiler("test-compiler", "ab" * 64)


@pytest.fixture(scope="module")
def run_a(tmp_path_factory):
    run = tmp_path_factory.mktemp("run_a")
    for name in ("step_trace.jsonl", "requests.jsonl"):
        (run / name).write_bytes(gzip.decompress((RUN_A / f"{name}.gz").read_bytes()))
    rep = ADV.write(run, compiler=C)
    out = run / "advice"
    return {"report": rep, "registration": json.loads((out / "registration.json").read_text()),
            "registration_text": (out / "registration.json").read_text(), "data": (out / "private" / "advice.bin").read_bytes(),
            "opening": json.loads((out / "private" / "opening.json").read_text())}


def _salts(opening):
    return [bytes.fromhex(s) for s in opening["salts"]]


def test_the_root_is_the_advice_commitments_reference_on_run_a(run_a):
    data, salts, root = run_a["data"], _salts(run_a["opening"]), run_a["registration"]["binding"]["advice"]["root"]
    assert I1.encode(I1.decode(data)) == data and I1.total_bits(I1.decode(data)) == RUN_A_BITS
    assert run_a["opening"]["advice"] == data.hex() and len(salts) == run_a["registration"]["rows"] == len(AW.advice_rows(data))
    assert root == AW.commit_advice(C, data, salts).root.hex()
    ref = AC.commit(AW.advice_rows(data), AW.ROW_BITS, AW.commitment_binding(C), salts)
    assert root == ref.root.hex()
    for i in (0, len(salts) - 1):
        assert AC.verify(bytes.fromhex(root), len(salts), AW.ROW_BITS, AW.commitment_binding(C), ref.open(i))


def test_the_registration_carries_the_root_and_bit_count_and_not_the_advice(run_a):
    reg, data = run_a["registration"], run_a["data"]
    assert reg["binding"] == AW.binding(C, reg["binding"]["advice"]["root"], RUN_A_BITS)
    assert set(reg) == {"schema", "scheme", "binding", "rows", "row_bits", "T", "bits_by_window", "compiler"}
    assert reg["compiler"] == {"name": AW.CAPPED, "digest": C.digest, "cap_bits": ADV.CAP_BITS,
                               "inner": {"name": "test-compiler", "digest": "ab" * 64}}
    assert sum(reg["bits_by_window"].values()) == RUN_A_BITS
    assert reg["bits_by_window"] == {str(w): n for w, n in I1.bits_by_window(I1.decode(data), reg["T"]).items()}
    text = run_a["registration_text"]
    assert data.hex() not in text and data[:32].hex() not in text and data[-32:].hex() not in text
    assert not any(s in text for s in run_a["opening"]["salts"])
    for digest in (hashlib.sha512(data).hexdigest(), hashlib.sha256(data).hexdigest()):
        assert digest not in text
    assert run_a["report"]["registration"] == reg


def test_a_tampered_advice_byte_changes_the_root(run_a):
    data, salts, root = run_a["data"], _salts(run_a["opening"]), run_a["registration"]["binding"]["advice"]["root"]
    for at in (0, len(data) // 2, len(data) - 1):
        bad = bytearray(data)
        bad[at] ^= 1
        assert AW.commit_advice(C, bytes(bad), salts).root.hex() != root


def test_fresh_salts_or_another_compiler_give_another_root(run_a):
    data, salts, root = run_a["data"], _salts(run_a["opening"]), run_a["registration"]["binding"]["advice"]["root"]
    assert AW.commit_advice(C, data).root.hex() != root
    assert AW.commit_advice(ADV.Compiler("test-compiler", "cd" * 64), data, salts).root.hex() != root


def test_advice_over_the_registered_cap_is_refused(run_a):
    a = I1.decode(run_a["data"])
    with pytest.raises(ValueError, match="over the registered cap"):
        ADV.commit(ADV.registered_compiler("test-compiler", "ab" * 64, cap_bits=RUN_A_BITS - 1), a)
    reg, _opening = ADV.commit(ADV.registered_compiler("test-compiler", "ab" * 64, cap_bits=8 * len(run_a["data"])), a)
    assert reg["compiler"]["cap_bits"] == 8 * len(run_a["data"])


def test_the_default_compiler_digest_is_64_bytes():
    c = ADV.registered_compiler()
    assert c.inner.name == ADV.COMPILER and len(bytes.fromhex(c.digest)) == 64 and len(bytes.fromhex(c.inner.digest)) == 64
