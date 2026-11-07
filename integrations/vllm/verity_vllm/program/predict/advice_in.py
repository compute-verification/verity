"""The compiler's advice input (`steps.compile_steps`): I1's bytes or an `Advice` within ℓ, or, under a registration that
selects the advice code, a v1 string within ℓ decoded against its certified record. Anything else is refused by name."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from verity.core.protocols.certifier import advice as I1
from verity_vllm.pipeline import window_advice as WA

from .common import Unexpressible

if TYPE_CHECKING:
    from .steps import Registered


def advice_of(registered: Registered, advice: Any, record: Any = None) -> WA.Advice:
    """The advice as a Commit window's (`window_advice`; I1 v0's when it carries nothing). Under a registration that selects the
    advice code (`Registered.code`), its v1 string's bytes within ℓ, decoded against the certified `record` by the scheduler
    replay (`Registered.code.decode`) into the I1 v0 it carries; otherwise I1's canonical bytes within ℓ, or an I1 or window
    `Advice` whose encoding is. Anything else is refused by name."""
    if registered.code is not None:
        return _coded_advice(registered, advice, record)
    if record is not None:
        raise Unexpressible("a certified record is the advice code's to decode against, and the registration selects none: its "
                            "advice is I1's")
    data = advice_bytes(advice)
    _within_cap(registered, data)
    return as_window_advice(data)


def advice_bytes(advice: Any) -> bytes:
    """I1's canonical bytes of ``advice``: bytes as given, or an I1 or window `Advice` encoded."""
    if isinstance(advice, (I1.Advice, WA.Advice)):
        try:
            return (WA if isinstance(advice, WA.Advice) else I1).encode(advice)
        except Exception as e:  # noqa: BLE001 -- an Advice built around its checks has no encoding: a refusal like any other
            raise Unexpressible(f"ill-formed advice: an Advice with no encoding ({type(e).__name__}: {e})") from None
    if type(advice) is bytes:
        return advice
    raise Unexpressible(f"the advice is I1's bytes or an I1 Advice, not a {type(advice).__name__}: nothing else is advice")


def as_window_advice(data: bytes) -> WA.Advice:
    """``data`` decoded as a Commit window's advice, or refused by name."""
    return _decoded("ill-formed advice", I1.MalformedAdvice, lambda: WA.decode(data))


def _within_cap(registered: Registered, data: bytes) -> None:
    if 8 * len(data) > registered.cap_bits:
        raise Unexpressible(f"the advice is {8 * len(data)} bits, over the registered cap ℓ = {registered.cap_bits} bits")


def _decoded(what: str, refusal: type[Exception], decode: Callable[[], Any]) -> Any:
    """``decode()``, any failure refused by name: ``what: why`` for the decoder's own `refusal`, else the failure named."""
    try:
        return decode()
    except Exception as e:  # noqa: BLE001 -- the decoder's refusal is `refusal`; any other failure on these bytes refuses them too
        raise Unexpressible(f"{what}: {e}" if isinstance(e, refusal) else
                            f"{what}: its decode failed ({type(e).__name__}: {e})") from None


def _coded_advice(registered: Registered, advice: Any, record: Any) -> WA.Advice:
    """A v1 string's I1 v0, held to ℓ before any of it is read and decoded under the registered code at that ℓ."""
    if type(advice) is not bytes:
        raise Unexpressible(f"the registration selects the advice code: its advice is a v1 string's bytes, not a "
                            f"{type(advice).__name__}")
    _within_cap(registered, advice)
    if record is None:
        raise Unexpressible("the advice code decodes against the session's certified record, and none was given")

    def canonical_bytes() -> bytes:
        adv = registered.code.decode(advice, record, registered.cap_bits)
        if type(adv) is not I1.Advice:
            raise TypeError(f"it gave a {type(adv).__name__}, not an I1 Advice")
        return I1.encode(adv)
    return as_window_advice(_decoded("ill-formed v1 advice", ValueError, canonical_bytes))


__all__ = ["advice_bytes", "advice_of", "as_window_advice"]
