"""The pouw settings of H1's static window and of the auditor host's challenger (`pouw.Settings`): their parsing (`of`),
the refusals of a scheme that sets them wrongly (`refusals`), and the ones `verity_vllm.LLM` takes from its built engine
(`fills`, `from_engine`)."""
from __future__ import annotations

from typing import Any, Mapping

from verity.core.protocols.pouw import verifier
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.coins import Challenger
from verity.core.protocols.sampled_proofs.challenger import LiveRecord

from verity_vllm.protocols.pouw import h1 as PH
from verity_vllm.protocols.pouw.h1 import challenger as PHC
from verity_vllm.protocols.pouw.h1 import requests as RQ
from verity_vllm.protocols.pouw.h1 import window as PHW
from verity_vllm.protocols.pouw import pearl_c as PPD

#: The schemes whose coins the auditor host's challenger can sample (the setting ``challenger``): H1's salt and rounds,
#: ncp-v2's and Pearl-C's salt
CHALLENGED = (PH.SCHEME, "ncp-v2", *PPD.SCHEMES)
#: The challenged schemes whose install opens the session the gate reserved itself (``session`` the gate's reserve
#: answer, ``gate`` its open), by the root it opens it on, the one its salt coin follows
OPENS = {PH.SCHEME: "its window-open root (salt.window_open_root, over round 0's request table)",
         **dict.fromkeys(PPD.SCHEMES, "its salt epoch's root (salt.epoch_open_root, over the weights' root)")}


def of(raw: Mapping[str, Any]) -> tuple[Any, ...]:
    """The settings ``rows``, ``steps``, ``runner``, ``rounds``, ``challenger``, ``session``, ``admission``, ``gate``,
    ``served``, ``graphs`` and ``period`` of `raw`, in that order; ValueError names the first bad one."""
    rows, steps = raw.get("rows"), raw.get("steps")
    if rows is not None and (not isinstance(rows, int) or isinstance(rows, bool) or rows < 32 or rows % 32):
        raise ValueError("pouw setting 'rows' is a positive multiple of 32 (H1's T: vLLM's max_num_batched_tokens)")
    if steps is not None and (not isinstance(steps, int) or isinstance(steps, bool) or steps < 1):
        raise ValueError("pouw setting 'steps' is a positive integer (H1's S: the window's engine steps)")
    runner, rounds = raw.get("runner"), raw.get("rounds")
    if runner is not None and not (callable(getattr(runner, "execute_model", None))
                                   and isinstance(getattr(runner, "decode_query_len", None), int)):
        raise ValueError("pouw setting 'runner' is the engine's loaded Model Runner V2 (VLLM_USE_V2_MODEL_RUNNER=1): its "
                         "execute_model, and its decode_query_len, which orders a step's rows (pouw_h1_requests.batch_order)")
    if rounds is not None:
        commitment = getattr(getattr(rounds, "challenger", None), "commitment", None)
        if not callable(getattr(rounds, "record", None)) or commitment is None:
            raise ValueError("pouw setting 'rounds' is the window's round coins: a LiveRecord over a live challenger")
        why = Wd.live_epoch_reason(commitment)
        if why is not None or getattr(rounds, "label", None) != Wd.ROUND:
            raise ValueError(f"pouw setting 'rounds' is a LiveRecord of a live epoch's {Wd.ROUND} coins"
                             + (f": {why}" if why else ""))
    challenger, session = PHC.address(raw.get("challenger")), PHW.session_setting(raw.get("session"))
    admission = raw.get("admission")
    if admission is not None and admission not in RQ.ADMISSIONS:
        raise ValueError(f"pouw setting 'admission' is one of {', '.join(RQ.ADMISSIONS)}")
    gate = raw.get("gate")
    if gate is not None and not callable(gate):
        raise ValueError("pouw setting 'gate' is gate(session, root): the challenger gate's open of the reserved "
                         "session on the root the window's salt follows (pouw_h1_window.Gate)")
    served = raw.get("served")
    if served is not None:
        from verity_vllm.protocols.pouw.h1 import served as SV
        if not isinstance(served, SV.Served) or served.registered or served.refused:
            raise ValueError("pouw setting 'served' is a fresh pouw_h1_served.Served: each call's A and z/y tables and "
                             "the weights' B committed as the window Program's registered values")
    graphs = raw.get("graphs", ())
    if not (isinstance(graphs, (list, tuple))
            and all(isinstance(z, int) and not isinstance(z, bool) and z > 0 for z in graphs)):
        raise ValueError("pouw setting 'graphs' is the engine's CUDA-graph capture sizes, positive integers (none when eager)")
    return (rows, steps, runner, rounds, challenger, session, admission, gate, served, tuple(graphs),
            PHW.period_setting(raw.get("period")))


def fills(scheme: str, settings: Mapping[str, Any]) -> tuple[str, ...]:
    """The pouw settings `from_engine` supplies for ``scheme`` where ``settings`` leave them unset: H1's ``runner`` and,
    without a ``challenger``, its in-process ``coins`` and ``rounds`` (``rounds`` only once ``steps`` is set); nothing
    for any other scheme."""
    if scheme != PH.SCHEME:
        return ()
    keys = [] if settings.get("runner") is not None else ["runner"]
    if settings.get("challenger") is None:
        keys += [k for k in ("coins", "rounds") if settings.get(k) is None]
    steps = settings.get("steps")
    if not (isinstance(steps, int) and not isinstance(steps, bool) and steps >= 1) and "rounds" in keys:
        keys.remove("rounds")
    return tuple(keys)


def from_engine(scheme: str, settings: Mapping[str, Any], llm: Any, epoch: int = 0) -> dict[str, Any]:
    """`fills`' settings from the built ``vllm.LLM`` ``llm``: its in-process model runner, where the scheduler's admissions
    arrive (`engine.handle.model_runner`; RuntimeError on an engine without one), and epoch ``epoch``'s coins held in
    this process, the Lean verifier of record's entropy (`verifier.coins`): its salt coin and ``steps`` round coins.
    Coins held in the serving process credit nothing (`pouw_h1.IN_PROCESS`)."""
    keys = fills(scheme, settings)
    out: dict[str, Any] = {}
    if "runner" in keys:
        from verity_vllm.engine.handle import model_runner
        out["runner"] = model_runner(llm)
    if "coins" in keys:
        out["coins"] = Challenger(epoch, verifier.coins)
    if "rounds" in keys:
        out["rounds"] = LiveRecord(Wd.live_challenger(epoch, settings["steps"], verifier.coins), Wd.ROUND)
    return out


def refusals(scheme: str, settings: Any, stage: str, filled: tuple[str, ...] = ()) -> list[str]:
    """Why `scheme`'s parsed pouw `settings` at `stage` set H1's or the challenger's settings wrongly, one reason each;
    `filled` names the settings the built engine will supply (`pouw.fills`), read as set."""
    out = []
    if scheme == PH.SCHEME and (settings.rows is None or settings.steps is None):
        out.append(f"{scheme}: the static layout's public parameters are the pouw settings 'rows' (T, vLLM's "
                   "max_num_batched_tokens) and 'steps' (S, the window's engine steps)")
    elif scheme != PH.SCHEME and (settings.rows is not None or settings.steps is not None):
        out.append(f"{scheme}: the pouw settings 'rows' and 'steps' are H1's ({PH.SCHEME})")
    runner = settings.runner is not None or "runner" in filled
    rounds = settings.rounds is not None or "rounds" in filled
    if scheme == PH.SCHEME and stage != "capture" and (not runner or (not rounds and settings.challenger is None)):
        out.append(f"{scheme}: each admitted request's entry goes under the root of the round that admits it, before "
                   "that round's coin: the pouw settings 'runner' (the engine's model runner, where its scheduler's "
                   "admissions arrive) and 'challenger' (the auditor host's, HOST:PORT) or, in tests, 'rounds' (the "
                   "window's round coins, a LiveRecord)")
    elif scheme != PH.SCHEME and (settings.runner is not None or settings.rounds is not None
                                  or settings.admission is not None or settings.served is not None):
        out.append(f"{scheme}: the pouw settings 'runner', 'rounds', 'admission' and 'served' are H1's ({PH.SCHEME})")
    if scheme not in CHALLENGED and settings.challenger is not None:
        out.append(f"{scheme}: the pouw setting 'challenger' is for {', '.join(CHALLENGED)} only")
    if (settings.session is not None or settings.gate is not None) and settings.challenger is None:
        out.append(f"{scheme}: the pouw setting 'session' is the session of the challenger at 'challenger'")
    if scheme in CHALLENGED and settings.challenger is not None:
        out += PHW.refusals(scheme, settings, stage, opens=OPENS.get(scheme))
    if scheme in PPD.SCHEMES and settings.challenger is not None and settings.session is not None:
        why = PHW.period_reason(settings.period, settings.session)
        out += [] if why is None else [f"{scheme}: {why}"]
    elif settings.period is not None:
        out.append(f"{scheme}: the pouw setting 'period' is the lane's open period that a Pearl-C install's reserved "
                   f"session joins, so it goes with 'challenger' and 'session', for {', '.join(PPD.SCHEMES)} only")
    if scheme == PH.SCHEME and settings.rounds is not None and settings.steps is not None:
        have = settings.rounds.challenger.commitment.family(settings.rounds.label) - len(settings.rounds.roots)
        if have < settings.steps:
            out.append(f"{scheme}: the round coins have {have} unopened coins, fewer than the window's {settings.steps} "
                       "rounds")
    return out


__all__ = ["CHALLENGED", "OPENS", "fills", "from_engine", "of", "refusals"]
