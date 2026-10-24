"""Phase-3 adversarial cases against the prove path (`prove_path`, #3527 @d4d336dd16), from the network-accounting red
team. Each states whether the surface is closed, and the one open surface is the named gap. These complement the
author's `test_prove_path.py`; they hold the CPU-only pure surfaces a malicious prover reaches without the firewall.

Ranked by the red team:
- CLOSED, up to the registered cap ℓ: `Capped.digest` binds ℓ and the inner compiler, so a registration naming another
  cap is a distinct compiler (the `registration-names-another-cap` negative), and `compile_advice` refuses an advice
  over ℓ before any of it is read. The boundary is exact.
- CLOSED: the opening is bound to its committed root. `compile_advice`/`open_advice` refuse non-canonical advice bytes
  and a wrong-scheme document, and `advice_windows.of_opening` recomputes the root from the opened advice and its salts,
  so an opening can't verify under the registration's root yet decode to another advice.
- CLOSED: `advice_before_draw` holds the registration the verifier opened to the one the draw's receipt names, so a
  re-registration after the coins (`advice_after_draw`) is refused as after-the-draw, and a registration binding another
  compiler or advice root is refused by name.
- OPEN, the NAMED GAP (acknowledged, README/`audit`'s `summary["gap"]`): the advice is opened to the verifier IN THE
  CLEAR, and the stand-in compiler `Capped(advice_windows.Traced)` does not compile the advice — it loads the Build's
  Programs OF RECORD, keyed by `requests_of`'s per-request params. So (a) the advice (every request's prompt and output
  lengths and its schedule) is revealed, not zero-knowledge, until the compile unit (`predict.steps.compile_steps`)
  exists; and (b) the correspondence between the committed advice and the audited step units rests on that in-the-clear
  opening, not on a proof. A real compile would derive the Programs from the advice and prove they served the run.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

from verity_vllm.pipeline import advice_windows as AW
from verity.core.protocols.sampled_proofs.one_stage import registration as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import prove_path as PP  # noqa: E402

try:
    import a5 as A5  # noqa: E402
    VERIFIER = A5.VERIFIER
except Exception:  # pragma: no cover - a5 lives beside prove_path in the one_stage dir
    VERIFIER = None


class _Inner:
    """A stand-in for the inner compiler: `Capped` reads only its `name` and `digest`."""
    name = "redteam/inner/v0"
    digest = "0" * 128

    def load(self, rank, world, requests):
        return None


def _advice_bytes():
    c = AW.codec_of()
    wl = {"requests": [{"prompt_token_ids": list(range(4)), "max_tokens": 2, "arrive_step": 0}]}
    return c.encode(AW.advice_of_workload(wl, c))


def test_the_registered_cap_is_exact_and_refused_before_the_advice_is_read():
    """ℓ boundary. The advice at exactly ℓ bits compiles; one byte over ℓ is refused as over the cap, before the bytes
    are decoded (the message quotes the lengths, not a decode error)."""
    data = _advice_bytes()
    assert PP.compile_advice(PP.Capped(_Inner(), 8 * len(data)), data)             # exactly ℓ
    with pytest.raises(ValueError, match="over the registered cap"):
        PP.compile_advice(PP.Capped(_Inner(), 8 * len(data) - 8), data)            # ℓ + 8 bits of advice


def test_capped_digest_binds_the_cap_and_refuses_an_ill_formed_one():
    """`Capped.digest` moves with ℓ, so a registration that names another cap is a distinct compiler (the
    `registration-names-another-cap` negative is sound). The cap is a whole number of bits, at least 8."""
    inner = _Inner()
    assert PP.Capped(inner, 1 << 20).digest != PP.Capped(inner, 1 << 21).digest
    assert PP.Capped(inner, 1 << 20).digest != PP.Capped(_Inner_other(), 1 << 20).digest
    for bad in (True, 7, 1.0, "8"):
        with pytest.raises(ValueError, match="cap"):
            PP.Capped(inner, bad)


def _Inner_other():
    class O(_Inner):
        digest = "1" * 128
    return O()


def test_compile_advice_refuses_non_bytes_and_non_canonical_advice():
    """The compiler is total and its refusals are a function of the input alone: not a byte string, and bytes that
    decode but are not the advice's canonical encoding (a trailing byte, nonzero padding) are refused."""
    data = _advice_bytes()
    with pytest.raises(ValueError, match="byte string"):
        PP.compile_advice(PP.Capped(_Inner()), "not bytes")
    with pytest.raises(ValueError):
        PP.compile_advice(PP.Capped(_Inner()), data + b"\x00")  # bytes past the advice / nonzero padding


def test_the_opening_is_bound_to_its_committed_root():
    """CLOSED. A faithful opening recomputes the committed root; tampering the advice bytes makes a non-canonical (or
    other-version) document, refused; a wrong-scheme document is refused. So an opening can't pass under the
    registration's root yet decode to a different advice."""
    comp = PP.Capped(_Inner())
    data = _advice_bytes()
    _reqs, bound, reg = AW.of_advice(comp, data)
    op = AW.opening(data, reg)
    _rq, b = PP.open_advice(comp, op)
    assert b["advice"] == bound["advice"]                      # faithful: same committed root
    tam = copy.deepcopy(op)
    ba = bytearray(bytes.fromhex(tam["advice"]))
    ba[0] ^= 1
    tam["advice"] = bytes(ba).hex()
    with pytest.raises(ValueError):
        PP.open_advice(comp, tam)                              # decodes differently -> refused
    with pytest.raises(ValueError, match="opening"):
        PP.open_advice(comp, {"scheme": "x", "advice": op["advice"]})


@pytest.mark.skipif(VERIFIER is None, reason="a5 not importable")
def test_advice_before_draw_pins_the_registration_to_the_draw():
    """CLOSED. `advice_before_draw` is ok only when the receipt names the digest of the registration the verifier opened
    (ordered before the coins) under this verifier, and that registration binds the opened compiler and advice root. A
    receipt that drew after another registration, or a registration binding another root, is refused by name — so a
    re-registration after the coins can't be passed off as the drawn one."""
    comp = PP.Capped(_Inner())
    data = _advice_bytes()
    _reqs, bound, _reg = AW.of_advice(comp, data)
    rec = {"window": {"compiler": bound["compiler"], "advice": bound["advice"]}}
    ok = {"registration": R.digest(rec), "verifier": VERIFIER}
    assert PP.advice_before_draw(ok, rec, bound)["ok"]
    assert PP.advice_before_draw({"registration": "00", "verifier": VERIFIER}, rec, bound)["codes"] == ["after-the-draw"]
    bad = {**bound, "advice": {"root": "f" * 128, "bits": bound["advice"]["bits"]}}
    assert "advice" in PP.advice_before_draw(ok, rec, bad)["codes"]


def test_the_named_gap_the_advice_is_opened_in_the_clear():
    """OPEN (the named gap). The opening carries the whole advice in the clear, and the stand-in reads only per-request
    params (`requests_of`), not the committed schedule — so the audited step units are the Build's Programs, keyed by
    those params, and the tie to the committed advice is the opening, not a proof. Until the compile unit exists:
    (a) the advice is revealed (not zero-knowledge); (b) the step units' correspondence to the advice is unproven."""
    comp = PP.Capped(_Inner())
    data = _advice_bytes()
    _reqs, _bound, reg = AW.of_advice(comp, data)
    op = AW.opening(data, reg)
    assert bytes.fromhex(op["advice"]) == data                # the full advice, in the clear
    reqs = AW.requests_of(AW.codec_of().decode(data))
    assert set(reqs[0]) == {"request_id", "arrive_step", "prompt_len", "max_tokens"}  # params only, not the wiring
