"""PoUW as a protocol option: H1 + fx1 (``eps1-sm120-h1fx1``, `pouw_h1`) beside the eager int8_exact arm.

Nothing is executed in the linear's place (`executes`): the arm serves its own y, and each call is recorded as its own
Program, compiled to H1 tile units (`verity.protocols.resource_accounting.compute.pouw.prover.circuit.compile`), with its
seed group's roots and its positions.  In the composition's weight phase, with the model untouched and no coin opened,
`commit_weights` commits the arm's int8 weight rows as they are served (`pouw_h1_hooks.committed`); `install` then takes
the execution phase's salt from its challenger (the setting ``coins``, or the auditor device's ``challenger``) and wires
the arm's hooks (`_install_h1`).  The other registered schemes are refused here (`EXECUTED`).  What is a placeholder is
listed in `info` (``placeholder``).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from verity.protocols.resource_accounting.compute.pouw.prover.serving import (
    Challenger,
    CoinCommitment,
    CoinWindow,
    Epoch,
    finish_window,
    H1_CHALLENGER,
    H1_SCHEME,
    H1Run,
    in_process_draw,
    Matmul,
    NCP_CIRCUIT_SCHEMES,
    NCP_SCHEMES,
    program_salt,
    ROUND0,
    Shape,
    use_int8_arm,
    verifier_coins,
    weights_root,
    Workload,
)

import verity_torch.pouw  # noqa: F401 -- importing it hands PoUW its tensors and H1's device
from verity import hooks as H
from verity_vllm.protocols.interface import Context, Option, Target
from verity_vllm.protocols.pouw import h1_hooks as PHK
from verity_vllm.protocols.pouw import h1_settings as PHS
from verity_vllm.protocols.pouw.h1_settings import fills, from_engine  # noqa: F401  (the adapter's hooks)


__all__ = ["Settings", "challenged", "h1_run", "keeps_weight_copy", "linear_types", "refusals", "requests"]

NAME = "pouw"
#: PoUW replaces the computation here; POUS wraps it from outside (`protocol_options.SITES`).
EXECUTES = ("LinearBase.quant_method.apply",)
#: The schemes this adapter runs: H1 + fx1 (`pouw_h1.SCHEME`), beside the int8_exact arm, which serves y itself.
EXECUTED = (H1_SCHEME,)
CHALLENGED = PHS.CHALLENGED
KEYS = ("coins", "retain_steps", "transcript_dir", "rows", "steps", "runner", "rounds", "challenger", "session", "admission",
        "gate", "served", "graphs", "period", "unit", "ks", "draw_session", "commit_record")


def schemes() -> tuple[str, ...]:
    return tuple(sorted({*NCP_SCHEMES, *NCP_CIRCUIT_SCHEMES, H1_SCHEME}))


def keeps_weight_copy(scheme: str) -> bool:
    """True where the executor keeps a weight-derived copy across forwards, so its GEMMs would not read what POUS decodes:
    none here, since H1 serves the arm's own y from the weight it is handed."""
    return False


def executes(scheme: str) -> tuple[str, ...]:
    """The sites where `scheme`'s executor replaces the computation: none under H1, whose arm serves its own y."""
    return () if scheme == H1_SCHEME else EXECUTES


@dataclass(frozen=True)
class Settings:
    #: the execution phase's challenger, in-process: its coin commitment (``commitment``, published before the execution phase) and
    #: ``open(label, after)``, each coin once (`verity.protocols.resource_accounting.compute.pouw.verifier.coins.Challenger`, or a proxy of one)
    coins: Any = None
    retain_steps: int = 0
    transcript_dir: str | None = None
    #: H1's static layout (`pouw_h1.H1Run`): T, the rows of every call (vLLM's ``max_num_batched_tokens``, a multiple of 32),
    #: and S, the steps of the install's execution phase, public parameters that fix every call's rows and positions
    rows: int | None = None
    steps: int | None = None
    #: H1's admissions and rounds (`pouw_h1.H1Run.admit`): the engine's model runner (Model Runner V2, whose
    #: ``decode_query_len`` orders a step's rows, `h1_hooks.batch_order`), whose ``execute_model`` gets each step's
    #: scheduler output, the requests it admits and schedules among it; and the execution phase's round coins, a `LiveRecord` over a
    #: live challenger's round family (`verity.protocols.resource_accounting.compute.pouw.verifier.window.live_challenger`), which opens round s's coin
    #: after round s's root
    runner: Any = None
    rounds: Any = None
    #: H1's salt and round coins from the auditor device's challenger, ``HOST:PORT``
    #: (`pouw_h1_challenger`); ``coins`` and ``rounds`` held in this process in its place are for tests, and credit
    #: nothing; ``session`` is the challenger gate's reserve answer for the execution phase (`pouw_h1_window.reserved_of`),
    #: which the install opens itself, on H1's execution phase-open root (`pouw_h1_settings.OPENS`)
    challenger: str | None = None
    session: Mapping[str, Any] | None = None
    #: H1's admissions (`pouw_h1_requests.ADMISSIONS`): None or ``round-0``, admit in round 0 only (the live tiles
    #: are fixed before the first coin), or ``per-round``
    admission: str | None = None
    #: H1's ``gate(session, root)``: the challenger gate's open of the reserved session on that root
    #: (`pouw_h1_window.Gate`; `pouw_h1_window.Window`, `pouw_h1_window.salt_session`)
    gate: Any = None
    #: H1's served tables (`pouw_h1_served.Served`): each call's and the weights' registered values, under the round roots;
    #: ``graphs`` the engine's CUDA-graph capture sizes, none when eager (`pouw_h1_static.graph_pad`)
    served: Any = None
    graphs: tuple[int, ...] = ()
    #: the lane's open period before ``session``'s reserve, or None (`pouw_h1_window.reserve_salt`, `period_reason`)
    period: Mapping[str, Any] | None = None
    #: H1's execution phase Program's tile unit (`pouw_h1_settings.UNITS`; None is v1's) and the K's it records (`pouw_h1_hooks.layers`)
    unit: str | None = None
    ks: tuple[int, ...] | None = None
    #: H1's sample session: the gate's reserve answer for the execution phase's sampling coin, opened withheld (`pouw_h1_window.DrawSession`)
    draw_session: Mapping[str, Any] | None = None
    #: H1's Commit record named at the execution phase's close in place of its tap's (`pouw_h1_commit.CommitRecord`), or None
    commit_record: Any = None

    @classmethod
    def of(cls, raw: Mapping[str, Any]) -> Settings:
        """PoUW's keys of the protocols config; ValueError names the first bad one."""
        unknown = sorted(set(raw) - set(KEYS))
        if unknown:
            raise ValueError(f"unknown pouw settings {unknown} (known: {', '.join(KEYS)})")
        coins = raw.get("coins")
        if coins is not None and not (isinstance(getattr(coins, "commitment", None), CoinCommitment)
                                      and callable(getattr(coins, "open", None))):
            raise ValueError("pouw setting 'coins' is the epoch's challenger: its coin commitment and open(label, after)")
        retain = raw.get("retain_steps", 0)
        if not isinstance(retain, int) or isinstance(retain, bool) or retain < 0:
            raise ValueError("pouw setting 'retain_steps' is a non-negative integer")
        tdir = raw.get("transcript_dir")
        if tdir is not None and not isinstance(tdir, str):
            raise ValueError("pouw setting 'transcript_dir' is a path string")
        (rows, steps, runner, rounds, challenger, session, admission, gate, served, graphs, period, unit, ks, draw_session,
         commit_record) = PHS.of(raw)
        return cls(coins, retain, tdir, rows, steps, runner, rounds,
                   challenger, session, admission, gate, served, graphs, period, unit, ks, draw_session, commit_record)


def refusals(scheme: str, ctx: Context) -> tuple[str, ...]:
    out = []
    if scheme not in EXECUTED:
        out.append(f"{scheme}: this adapter's executor runs {', '.join(EXECUTED)} only")
    try:
        settings = Settings.of(ctx.settings_of(NAME))
    except ValueError as e:
        out.append(str(e))
    else:
        out += PHS.refusals(scheme, settings, ctx.stage, ctx.filled_of(NAME))
    return tuple(out)


def requests(scheme: str, ctx: Context) -> tuple[H.Request, ...]:
    """Eager execution and one rank."""
    return (H.RequireProperty("eager", True, "compiled execution: the wrapper around each linear layer's apply is not "
                              "graph-capturable"),
            H.RequireProperty("world", 1, "PoUW's units and transcript are per rank, which is not wired"))


def linear_types() -> tuple[type, ...]:
    """The hook-site types: vLLM's ``LinearBase`` (qkv, o, gate_up, down).  lm_head is a ``VocabParallelEmbedding``, not
    covered."""
    from vllm.model_executor.layers.linear import LinearBase
    return (LinearBase,)


def _sites(model: Any, layers: list[Any]) -> dict[str, int]:
    named = getattr(model, "named_modules", None)
    if named is None:
        return {}
    ids = {id(m): w for w, m in enumerate(layers)}
    return {name: ids[id(m)] for name, m in named() if id(m) in ids}


def commit_weights(scheme: str, target: Target) -> dict[str, Any]:
    """The composition's first phase, with the model untouched and no coin opened: the int8_exact arm's int8 weight rows
    as they are served, once a process (`pouw_h1_hooks.committed`), as ``{"weight_roots": {id: hex}, "shapes": {id: [n, k]}}``."""
    ks = Settings.of(target.context.settings_of(NAME)).ks
    return PHK.committed(PHK.layers(target.model, linear_types(), ks, scheme=scheme))


def challenged(settings: Mapping[str, Mapping[str, Any]], enabled: Any, epoch: int = 0) -> dict[str, dict[str, Any]]:
    """`settings` (the loaded protocols config) with PoUW's ``coins`` a challenger of `epoch` held in this process
    (`verity.protocols.resource_accounting.compute.pouw.verifier.coins.Challenger`, its coins the Lean verifier's, `pouw.verifier.coins`) when
    `enabled` (the protocol set) has PoUW, for a pipeline that is its run's own auditor (it replays and judges what it
    served); a config's own ``coins``, or its auditor device's ``challenger`` (the coins' one source), is kept alone."""
    out = {k: dict(v) for k, v in settings.items()}
    takes_salt = any(str(e).partition(":")[0] == NAME for e in (enabled or ()))
    if takes_salt and not {"coins", "challenger"} & set(out.get(NAME, {})):
        out.setdefault(NAME, {})["coins"] = Challenger(epoch, verifier_coins)
    return out


def install(scheme: str, target: Target) -> Option:
    entry = f"{NAME}:{scheme}"
    settings = Settings.of(target.context.settings_of(NAME))
    if settings.coins is None and not (scheme in CHALLENGED and settings.challenger is not None):
        raise ValueError(f"{entry}: no 'coins' setting: the epoch's salt is its challenger's salt coin, opened after the "
                         "weights' root, and there is no beacon")
    return _install_h1(entry, settings, target)


#: The live H1 runs by entry, for the Commit and its checks before the composition closes.
_H1_RUNS: dict[str, H1Run] = {}


def h1_run(entry: str) -> H1Run | None:
    return _H1_RUNS.get(entry)


def _int8_exact() -> Any:
    from verity_vllm.engine import int8_exact

    return int8_exact


def _install_h1(entry: str, settings: Settings, target: Target) -> Option:
    """H1 on the int8_exact arm: the execution phase's salt `audit.salt_of` over the weights' root, as every scheme's, from the
    challenger's salt coin (the auditor device's opened after the execution phase's open root, once round 0 is admitted:
    `pouw_h1.H1Run.open_window`), and each layer's ``observe`` the arm's own with its call recorded and handed to the run's listeners
    (`pouw_h1_hooks.observer`), in the static layout of the settings' ``rows`` and ``steps``; each layer's ``apply`` runs
    under the credit gate's trace (`pouw_h1_hooks.traced`), and the ``runner``'s ``execute_model`` hands each step's admitted
    requests and its batch order to the run (`pouw_h1_hooks.admitting`), each entry and each served row's request under its
    round's root before the ``rounds`` coin, and the execution phase's sampling coin opens at release after its manifest. With the
    setting ``challenger`` the salt and round coins are the auditor device's, from the one session infra's gate
    reserved for the execution phase (``session``, the gate's reserve answer) and ``gate`` opened on its execution phase-open root (the
    open answer, which the commitment carries without its secret; `pouw_h1_window.Window`), closed once the execution phase has
    finished, and the sampling coin is ``draw_session``'s, opened withheld on the sample root (`pouw_h1_window.DrawSession`);
    without it the coins are this process's and the execution phase credits nothing (`pouw_h1.IN_PROCESS`). The
    execution phase's weights' root is its period's (`pouw_h1_served.Served.weights_root`), else `audit.weights_root`, which
    registers no ``b/<K>`` and so credits nothing. ``admission`` is the run's (`pouw_h1_requests.ADMISSIONS`, round 0
    only by default). On an arm that serves z/y records (``zy_rows``, `engine.int8_vllm.Int8PlainLinearMethod`) each
    committed layer whose N is whole 32-column units serves through ``scaled_mm_zy`` over T slots, the records its z. The
    served values are the arm's, untouched. A linear at a K outside ``ks`` serves unhooked (`pouw_h1_hooks.wire`)."""
    use_int8_arm(_int8_exact)
    layers = PHK.layers(target.model, linear_types(), settings.ks)
    committed = target.weight_commitments.get(entry) or commit_weights(H1_SCHEME, target)
    roots = {int(w): bytes.fromhex(r) for w, r in committed["weight_roots"].items()}
    declared = Workload(tuple(Matmul(int(w), Shape(16, k, n), int(w)) for w, (n, k) in committed["shapes"].items()))
    if settings.rows is None or settings.steps is None:
        raise ValueError(f"{entry}: the static layout needs the pouw settings 'rows' (T) and 'steps' (S)")
    session, rounds, draw, salt, opener = None, settings.rounds, None, None, None
    shapes = cast("list[tuple[int, int]]", [tuple(int(d) for d in m.weight.shape) for m in layers])
    modules = {w: f"linear {w}" for w in roots} | _named(target.model, layers)
    wroot = (settings.served and settings.served.weights_root(shapes, modules, settings.rows, settings.steps, layers,
                                                              target.context.stage != "capture")) or weights_root(roots)
    if settings.challenger is not None:
        session = CoinWindow(entry, settings.challenger, settings.steps, wroot, settings.session, settings.gate, settings.draw_session)

        def opener(requests_root: bytes) -> tuple[bytes, Any, Any, Mapping[str, Any]]:
            epoch_salt, *opened = session.open(requests_root)
            return (program_salt(epoch_salt), *opened)
    else:
        commitment, opening = settings.coins.commitment, settings.coins.open("salt", weights_root(roots))
        salt = program_salt(Epoch.start(declared, roots, commitment, opening).salt)
        draw = in_process_draw(rounds, verifier_coins)
    source = H1_CHALLENGER if session is not None else "in-process"
    run = H1Run(entry, salt, modules, shapes, settings.rows, settings.steps, retain=settings.retain_steps,
                   transcript_dir=settings.transcript_dir, commit=target.context.stage != "capture", rounds=rounds, draw=draw,
                   admission=settings.admission or ROUND0, coins_from=source, opener=opener, served=settings.served,
                   session=session.reserved["session"] if session is not None else None, graphs=settings.graphs,
                   commit_record=settings.commit_record)
    run.commits.update(weights_root=wroot.hex(), coin_source=source)
    if session is None:
        run.commits.update(coins=commitment.to_json(), salt_opening=opening.to_json())
        if rounds is not None:
            run.commits["round_coins"] = rounds.challenger.commitment.to_json()
    _H1_RUNS[entry] = run
    h = target.hooks(entry)
    PHK.wire(run, h, layers, settings.runner, settings.rows, settings.ks)
    return Option(name=entry, hooks=h, commits=run.commits, details=run.details, release=lambda: finish_window(run, session))


def _named(model: Any, layers: list[Any]) -> dict[int, str]:
    return {w: name for name, w in _sites(model, layers).items()} | PHK.expert_names(layers)
