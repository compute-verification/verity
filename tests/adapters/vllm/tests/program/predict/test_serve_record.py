"""serve-openai's record options and the preflight its card run passes before the lease, on the processor."""
from __future__ import annotations

import pytest

from verity_vllm.config import UsageError
from verity_vllm.pipeline import cli
from verity_vllm.program.predict import serve as SV


def _opts(tmp_path, *more):
    return cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--out", str(tmp_path / "serve"), *more])


def test_the_record_options_parse_into_the_serves_record(tmp_path):
    opts = _opts(tmp_path, "--window-class", "4,27,2", "--seal", "--sampling-design", "rate:1/40", "--checkpoint-dir",
                 str(tmp_path), "--record-cap", "32", "--twin-ports", "2")
    rec = SV.record_of(tmp_path, opts)
    assert (rec.rule, rec.checkpoint_dir, rec.cap, rec.sample) == ("rate:1/40", str(tmp_path), 32, 2)
    assert SV.record_of(tmp_path, _opts(tmp_path, "--window-class", "4,27,2", "--seal")) is None


@pytest.mark.parametrize(("more", "why"), [
    (("--sampling-design", "rate:1/40", "--checkpoint-dir", "x"), "needs --seal"),
    (("--sampling-design", "rate:1/40", "--seal"), "--checkpoint-dir, which is not given"),
])
def test_a_record_without_its_seal_or_checkpoint_is_refused_by_name(tmp_path, more, why):
    with pytest.raises(UsageError, match=why):
        SV.record_of(tmp_path, _opts(tmp_path, "--window-class", "4,27,2", *more))


def test_the_preflight_passes_off_the_card_and_refuses_a_serve_with_no_class_or_a_fixed_count(tmp_path):
    got = SV.preflight(_opts(tmp_path, "--window-class", "4,27,2", "--seal", "--sampling-design", "rate:1/40",
                             "--checkpoint-dir", str(tmp_path)), card=False)
    assert got["class"] == [4, 27, 2] and got["record"] and got["twin"].startswith("equal")
    with pytest.raises(UsageError, match="--window-class"):
        SV.preflight(_opts(tmp_path), card=False)
    with pytest.raises(ValueError):
        SV.preflight(_opts(tmp_path, "--window-class", "4,27,2", "--seal", "--sampling-design", "subset:1",
                           "--checkpoint-dir", str(tmp_path)), card=False)
