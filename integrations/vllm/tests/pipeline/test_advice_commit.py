"""`verity-vllm advice` commits the advice and publishes only its binding, never the advice nor its length, on run A's
advice (`fixtures/advice_run_a`: a recorded step trace and Commit record, Qwen3-8B on one GPU): the root is the one
`advice_windows.commit_advice` and `verity.ml.commitments.advice` give for I1's encoding padded to ℓ, the registration
carries the root and ℓ and no byte or length of the advice, and one changed byte, padding included, changes the root."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from verity_vllm.scheduler import advice as I1
from verity.ml.commitments import advice as AC
from verity_vllm.pipeline import advice as ADV
from verity_vllm.pipeline import advice_windows as AW

RUN_A = Path(__file__).parent / "fixtures" / "advice_run_a"
RUN_A_BITS = 71801
CAP = 1 << 17
C = ADV.registered_compiler("test-compiler", "ab" * 64, cap_bits=CAP)


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
    enc = C.unpad(data)
    assert I1.encode(I1.decode(enc)) == enc and I1.total_bits(I1.decode(enc)) == RUN_A_BITS
    assert 8 * len(data) == CAP and data == enc + b"\x80" + bytes(len(data) - len(enc) - 1)
    assert run_a["opening"]["advice"] == data.hex() and len(salts) == run_a["registration"]["rows"] == len(AW.advice_rows(data)) \
        == CAP // AW.ROW_BITS
    assert root == AW.commit_advice(C, data, salts).root.hex()
    ref = AC.commit(AW.advice_rows(data), AW.ROW_BITS, AW.commitment_binding(C), salts)
    assert root == ref.root.hex()
    for i in (0, len(salts) - 1):
        assert AC.verify(bytes.fromhex(root), len(salts), AW.ROW_BITS, AW.commitment_binding(C), ref.open(i))


def test_the_registration_carries_the_root_and_the_cap_and_not_the_advice_or_its_length(run_a):
    reg, data = run_a["registration"], run_a["data"]
    enc = C.unpad(data)
    assert reg["binding"] == AW.binding(C, reg["binding"]["advice"]["root"], CAP)
    assert set(reg) == {"schema", "scheme", "binding", "rows", "row_bits", "compiler"}
    assert reg["compiler"] == {"name": AW.CAPPED, "digest": C.digest, "cap_bits": CAP,
                               "inner": {"name": "test-compiler", "digest": "ab" * 64}}
    text = run_a["registration_text"]
    assert str(RUN_A_BITS) not in text and str(len(enc)) not in text and str(-(-RUN_A_BITS // AW.ROW_BITS)) not in text
    assert enc.hex() not in text and enc[:32].hex() not in text and enc[-32:].hex() not in text
    assert not any(s in text for s in run_a["opening"]["salts"])
    for digest in (hashlib.sha512(data).hexdigest(), hashlib.sha256(data).hexdigest()):
        assert digest not in text
    assert run_a["report"]["registration"] == reg


def test_a_tampered_advice_byte_changes_the_root(run_a):
    data, salts, root = run_a["data"], _salts(run_a["opening"]), run_a["registration"]["binding"]["advice"]["root"]
    n = len(C.unpad(data))
    for at in (0, n // 2, n - 1, n, len(data) - 1):
        bad = bytearray(data)
        bad[at] ^= 1
        assert AW.commit_advice(C, bytes(bad), salts).root.hex() != root


def test_fresh_salts_or_another_compiler_give_another_root(run_a):
    data, salts, root = run_a["data"], _salts(run_a["opening"]), run_a["registration"]["binding"]["advice"]["root"]
    assert AW.commit_advice(C, data).root.hex() != root
    assert AW.commit_advice(ADV.Compiler("test-compiler", "cd" * 64), data, salts).root.hex() != root


def test_advice_over_the_registered_cap_less_its_end_mark_is_refused(run_a):
    enc = C.unpad(run_a["data"])
    a = I1.decode(enc)
    for cap in (8 * len(enc) - 8, 8 * len(enc)):
        with pytest.raises(ValueError, match="over the registered cap"):
            ADV.commit(ADV.registered_compiler("test-compiler", "ab" * 64, cap_bits=cap), a)
    reg, opening = ADV.commit(ADV.registered_compiler("test-compiler", "ab" * 64, cap_bits=8 * len(enc) + 8), a)
    assert reg["compiler"]["cap_bits"] == reg["binding"]["advice"]["bits"] == 8 * len(enc) + 8
    assert bytes.fromhex(opening["advice"]) == enc + b"\x80"


def test_two_advices_of_different_lengths_register_alike_but_for_their_root(run_a):
    """Run A's advice and a one-request advice, committed under one compiler: every field of the two registrations but the
    root is the same, and the two openings hold as many bytes and salts, so nothing public says either's length."""
    small = AW.advice_of_workload({"requests": [{"prompt_token_ids": [1, 2, 3], "max_tokens": 2}]})
    assert 8 * len(I1.encode(small)) < RUN_A_BITS // 100
    (big, big_open), (little, little_open) = ADV.commit(C, I1.decode(C.unpad(run_a["data"]))), ADV.commit(C, small)
    assert big["binding"]["advice"]["root"] != little["binding"]["advice"]["root"]
    root = {"binding": {**big["binding"], "advice": {**big["binding"]["advice"], "root": "-"}}}
    assert {**big, **root} == {**little, **root} and big["rows"] == CAP // AW.ROW_BITS
    assert [len(o["advice"]) for o in (big_open, little_open)] == [CAP // 4] * 2
    assert [len(o["salts"]) for o in (big_open, little_open)] == [CAP // AW.ROW_BITS] * 2


def test_the_capped_compiler_reads_only_advice_padded_to_its_cap():
    c = ADV.registered_compiler("test-compiler", "ab" * 64, cap_bits=64)
    assert c.pad(b"\x01\x00") == b"\x01\x00\x80" + bytes(5) and c.unpad(c.pad(b"\x01\x00")) == b"\x01\x00"
    assert c.unpad(c.pad(b"")) == b"" and c.unpad(c.pad(bytes(7))) == bytes(7)
    for bad in (b"\x01\x00", b"\x01\x00\x80" + bytes(4), b"\x01\x00\x80" + bytes(6), bytes(8), b"\x01" * 7 + b"\x01"):
        with pytest.raises(ValueError):
            c.unpad(bad)
    with pytest.raises(ValueError, match="over the registered cap"):
        c.pad(bytes(8))
    with pytest.raises(ValueError, match="multiple of 8"):
        ADV.registered_compiler("test-compiler", "ab" * 64, cap_bits=65)


@pytest.mark.parametrize("bits", [0, 1, 7, 8, 9, 1317])
def test_least_cap_is_the_least_width_pad_takes_an_encoding_of_that_many_bits_to(bits):
    data, cap = bytes(-(-bits // 8)), AW.least_cap(bits)
    assert len(AW.unpad(AW.pad(data, cap), cap)) == len(data)
    with pytest.raises(ValueError, match="over the registered cap"):
        AW.pad(data, cap - 8)


def test_the_default_compiler_digest_is_64_bytes():
    c = ADV.registered_compiler()
    assert c.inner.name == ADV.COMPILER and len(bytes.fromhex(c.digest)) == 64 and len(bytes.fromhex(c.inner.digest)) == 64
