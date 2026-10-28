"""The device a served Build vouches for (`acquire.served_device`): cc 9.0 binds FA3 and cc 10.0 and 12.0 FA2, each with its GEMMs' k16
step and its Triton compile target; a device without one of them is refused with every missing binding by name; a device whose compute
capability or SM count is not its registered card's is refused naming the field, both values and the card file, and a name registered
on no card and not excepted is refused; and FA2's launch at vLLM's defaults: a GQA decode launch swapped and split `fa2_num_splits`
ways (at a FULL graph's padded batch and max_model_len), the align kernel's kBlockN at one split, and any other launch the split
kernel's tile unsplit, causal when a sequence has more than one query row.  The split counts here are worked by hand from
`set_params_splitkv` (vllm-flash-attn 506341a1)."""
from __future__ import annotations

import json
from importlib import resources

import pytest

from verity_vllm.acquire import inductor_templates as T
from verity_vllm.acquire import served_device as SD
from verity_vllm.acquire.inductor_templates import Unmodelled
from verity_vllm.engine.profiles.devices import CARD, NO_CARD
from verity_vllm.pipeline import served_match
from verity.ml.definitions.registry import bf16_gemm_check as BC

#: a GQA model's attention (query heads, KV heads, head dim) at head dim 64 and the cc 12.0 device's SM count
HEADS, SMS = (32, 8, 64), 188
FA2 = [{"backend": "FLASH_ATTN", "fa_version": 2}]


def fa2_device(max_model_len: int = 4096) -> SD.Device:
    return SD.device_of((12, 0), SMS, attention=FA2, max_model_len=max_model_len)


def test_each_cc_binds_its_own_attention_and_k16_step() -> None:
    fa3, fa2 = SD.device_of((9, 0), 132, attention=[{"backend": "FLASH_ATTN", "fa_version": 3}]), fa2_device()
    assert (fa3.attention, fa2.attention) == ("FA3", "FA2")
    assert fa3.dot.id == fa2.dot.id == "HopperBF16WgmmaDot16_v1"
    assert fa3.dispatch([1, 2], 8, 32) == {"capture_sizes": [1, 2], "max_num_seqs": 8, "split_cap": 32, "sms": 132}
    assert fa2.dispatch([1, 2], 8, 32) == {"capture_sizes": [1, 2], "max_num_seqs": 8, "sms": SMS, "cc": [12, 0], "attention": "FA2",
                                           "max_model_len": 4096}


def test_cc100_binds_fa2_which_its_row_declares_and_refuses_vllms_default_flashinfer() -> None:
    """The B200: cuBLAS's tcgen05 k16 step and FA2's mma.sync are Hopper's pipeline; vLLM's default there, FLASHINFER, has no
    Definitions, so a served row declares FLASH_ATTN FA2 and a process that ran FLASHINFER is refused by name."""
    b200 = SD.device_of((10, 0), 148, attention=FA2, max_model_len=4096)
    assert (b200.attention, b200.dot.id, b200.sms) == ("FA2", "HopperBF16WgmmaDot16_v1", 148)
    assert SD.device_of((10, 0), 148, max_model_len=4096) == b200
    with pytest.raises(Unmodelled) as e:
        SD.device_of((10, 0), 148, attention=[{"backend": "FLASHINFER", "fa_version": None}], max_model_len=4096)
    msg = str(e.value)
    assert msg.startswith("cc 10.0: no binding for attention FLASHINFER:") and "k16 step" not in msg and "cc=100" not in msg


def test_a_device_without_a_binding_is_refused_naming_every_one() -> None:
    with pytest.raises(Unmodelled) as e:
        SD.device_of((8, 6), 84, attention=[{"backend": "FLASHINFER", "fa_version": None}], max_model_len=4096)
    msg = str(e.value)
    assert msg.startswith("cc 8.6: no binding for ")
    assert "k16 step (no registered GEMM target at cc 8.6)" in msg and "attention FLASHINFER:" in msg and "cc=86" in msg
    with pytest.raises(Unmodelled, match="max_model_len"):
        SD.device_of((12, 0), SMS, attention=FA2)
    with pytest.raises(Unmodelled, match=r"attention FLASH_ATTN \(FA2\)"):
        SD.device_of((9, 0), 132, attention=FA2)
    with pytest.raises(Unmodelled, match="cc=86"):
        T.recognize("k", "@triton.jit\ndef k(x):\n    pass\n", cc=(8, 6))


def test_every_pinned_device_is_registered_on_a_card_or_excepted() -> None:
    """The served targets with a device pin are exactly the names `CARD` and `NO_CARD` list, each in one, and every exception
    says why."""
    pinned = {json.loads(p.read_text())["device"] for p in (resources.files("verity_vllm") / "pins").iterdir() if p.name.endswith(".json")}
    assert not set(CARD) & set(NO_CARD) and set(CARD) | set(NO_CARD) == pinned
    assert all(why.strip() for why in NO_CARD.values())


def test_a_device_with_its_cards_compute_capability_and_sm_count_binds() -> None:
    """Each registered name is a pinned device's, and that pin's compute capability and SM count are its card's."""
    for name, card in CARD.items():
        assert SD.card_of(name) == card
        pin = BC.device_pin(name)
        assert SD.device_of(pin["compute_capability"], pin["sms"], card=card, max_model_len=4096) == \
            SD.device_of(pin["compute_capability"], pin["sms"], max_model_len=4096)
    assert SD.device_of((12, 0), SMS, card="rtx-pro-6000-server", attention=FA2, max_model_len=4096) == fa2_device()


@pytest.mark.parametrize(("cc", "sms", "card", "off"), [
    ((9, 0), 114, "h200-sxm-141gb", "SM count 114, the card's 132"),
    ((10, 0), 132, "h200-sxm-141gb", "compute capability 10.0, the card's 9.0"),
    ((12, 0), 170, "rtx-pro-6000-server", "SM count 170, the card's 188"),
    ((9, 0), 132, "rtx-pro-6000-server", "compute capability 9.0, the card's 12.0; SM count 132, the card's 188"),
])
def test_a_device_that_is_not_its_card_is_refused_naming_each_field(cc: tuple[int, int], sms: int, card: str, off: str) -> None:
    with pytest.raises(Unmodelled) as e:
        SD.device_of(cc, sms, card=card, max_model_len=4096)
    assert str(e.value) == f"the device is not its registered card verity/hardware/cards/{card}.toml: {off}"


def test_acquisition_holds_the_engines_device_to_the_card_its_name_registers() -> None:
    """`served_match.device_of`, which served-match and the Commit's served mode take the served Build's device from: the engine's
    reported name picks the card, and the one name `NO_CARD` excepts (the B200) is held to none."""
    facts = {"device": "NVIDIA H200", "cc": [9, 0], "sms": 132, "attention": [{"backend": "FLASH_ATTN", "fa_version": 3}],
             "max_model_len": 4096}
    assert served_match.device_of(facts).sms == 132
    with pytest.raises(Unmodelled, match=r"cards/h200-sxm-141gb\.toml: SM count 114, the card's 132$"):
        served_match.device_of({**facts, "sms": 114})
    b200 = {**facts, "device": "NVIDIA B200", "cc": [10, 0], "sms": 148, "attention": FA2}
    assert SD.card_of("NVIDIA B200") is None and served_match.device_of(b200).sms == 148


@pytest.mark.parametrize("name", ["NVIDIA H200 NVL", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", ""])
def test_a_device_registered_on_no_card_and_not_excepted_is_refused(name: str) -> None:
    """A name in neither `CARD` nor `NO_CARD` is refused at acquisition, naming the device, both constants and the fix, even when
    its compute capability and SM count are a registered card's."""
    facts = {"device": name, "cc": [9, 0], "sms": 132, "attention": [{"backend": "FLASH_ATTN", "fa_version": 3}], "max_model_len": 4096}
    with pytest.raises(Unmodelled) as e:
        served_match.device_of(facts)
    assert str(e.value) == (f"device {name!r} is in neither engine.profiles.devices.CARD ({sorted(CARD)}) nor NO_CARD "
                            f"({sorted(NO_CARD)}): add its card under verity/hardware/cards/ and map it in CARD")


def test_fa2_full_decode_splits_at_the_capture_batch_and_max_model_len() -> None:
    # bnm = 4 x 8 x ceil(4 / 64) = 32 < 0.8 x 376; 16 key blocks of 256; 8 splits fill 256 of 376 SM slots, the eligible best
    NS, rows, plan = SD.launch(fa2_device(), [1, 1, 1], [10, 300, 700], HEADS, mode="FULL", padded=4, split_cap=32, capture_sizes=[1, 2, 4])
    assert plan == {"fa2": {"swapped": True, "BN": 256, "NS": 8, "batch": 4, "max_seqlen_k": 4096}}
    assert NS == [8, 8, 8]
    assert [[(r["base"], r["statics"]["T"], r["statics"]["NS"], r["statics"]["MASKED_FROM"]) for r in rs] for rs in rows] == [
        [("Fa2AttentionSplit_v1", 10, 8, 0)], [("Fa2AttentionSplit_v1", 300, 8, 1)], [("Fa2AttentionSplit_v1", 700, 8, 2)]]
    assert {r["statics"]["BN"] for rs in rows for r in rs} == {256}


def test_fa2_uncaptured_decode_splits_at_its_own_batch_and_keys() -> None:
    # 3 sequences over at most 300 keys: 2 blocks of 256, 2 splits; over 30 keys, 1 block, the align kernel (kBlockN 128 at D 64)
    _, rows, plan = SD.launch(fa2_device(), [1, 1, 1], [12, 40, 300], HEADS, mode="PIECEWISE", padded=4, split_cap=32, capture_sizes=[4])
    assert plan["fa2"] == {"swapped": True, "BN": 256, "NS": 2, "batch": 3, "max_seqlen_k": 300}
    _, rows, plan = SD.launch(fa2_device(), [1, 1, 1], [12, 20, 30], HEADS, mode="NONE", padded=3, split_cap=32, capture_sizes=[])
    assert plan["fa2"]["NS"] == 1 and plan["fa2"]["BN"] == 128
    assert {(r["base"], r["statics"]["BN"], r["statics"]["MASKED_FROM"]) for rs in rows for r in rs} == {("Attention_v5", 128, 0)}


def test_fa2_one_split_when_the_batch_fills_the_device() -> None:
    # 38 x 8 = 304 >= 0.8 x 376 query-row blocks: one split, the align kernel
    la = SD.fa2_launch([1] * 38, NH=32, KVH=8, D=64, sms=SMS, batch=38, max_seqlen_k=4096)
    assert (la["swapped"], la["NS"], la["BN"]) == (True, 1, 128)
    assert SD.fa2_launch([1], NH=32, KVH=8, D=128, sms=SMS, batch=38, max_seqlen_k=4096)["BN"] == 64


def test_fa2_unswapped_launch_runs_the_split_tile_unsplit() -> None:
    # a mixed step is causal for every sequence: its decode row over 600 keys guards from block 1, where a swapped decode guards block 2
    NS, rows, plan = SD.launch(fa2_device(), [3, 1], [600, 600], HEADS, mode="PIECEWISE", padded=8, split_cap=32, capture_sizes=[8])
    assert plan == {"fa2": {"swapped": False, "BN": 256, "NS": 1, "causal": True}} and NS == [1, 1]
    assert [[(r["base"], r["statics"]["T"], r["statics"]["MASKED_FROM"]) for r in rs] for rs in rows] == [
        [("Attention_v5", 598, 1), ("Attention_v5", 599, 1), ("Attention_v5", 600, 1)], [("Attention_v5", 600, 1)]]
    # without grouped heads a decode launch is not swapped and not causal: its first block only
    la = SD.fa2_launch([1, 1], NH=8, KVH=8, D=64, sms=SMS, batch=2, max_seqlen_k=600)
    assert la == {"swapped": False, "BN": 256, "NS": 1, "causal": False}
    assert [r["statics"]["MASKED_FROM"] for r in SD.fa2_attention_rows(1, 600, la, NH=8, KVH=8, D=64, dot=fa2_device().dot)] == [2]
