"""The network attester as a resource scheme (`verity.protocols.accounting.scheme`): what it bounds is egress,
the bits that leave beyond what the registered circuit explains. Its Lean instance (`NetTiming.Attester.scheme`)
is at one execution phase, its charge there a named premise (`NetTiming.Attester.Phase.ChargeHolds`), and its
incorrect blocks the record's where the committed values carry the attester's rows (`NetTiming.Attester.RowsCommitted`).
Its check is the Lean program ``network-period`` alone (`verifier.period_check.answer`), with no Python twin."""

from __future__ import annotations

import shlex
from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import Any

from verity.circuits.defs import bind
from verity.protocols.accounting.network.prover.egress import EgressBucket
from verity.protocols.accounting.network.prover import ingress_scan as IS
from verity.protocols.accounting.network.prover.ingress_scan import IngressScanBucket, IngressScanStart
from verity.protocols.accounting.network.prover.lengths_row import LengthsRow
from verity.protocols.accounting.network.prover.response_tie import ResponseTieNoStop, ResponseTieStop
from verity.protocols.accounting.network.verifier.blocks import SCAN_STATE
from verity.protocols.accounting.network.verifier.io_request import IoTieRequest
from verity.protocols.accounting.network.verifier.io_tie import IoTieBucket, IoTieContent, IoTieRoute
from verity.protocols.accounting import scheme as S
from verity.protocol import Channel, Role
from verity.protocols.accounting.scheme import Block, Charge, Layout, Order, Port, Scheme, Term, register
from verity.protocols.accounting.network.verifier import parties as NC, period_check
from verity import hooks as H
from verity.protocols.accounting.network.prover import id_relay, workload
from verity.protocols.accounting.network.prover.ids_protocol import IdsProtocol
from verity.protocols.accounting.network.prover.admission import ORDERS, Admitter, FrontFeed, engine, tick_config

DEFINITIONS = (IngressScanStart, IngressScanBucket, EgressBucket, IoTieRoute, IoTieContent, IoTieBucket, IoTieRequest,
               ResponseTieStop, ResponseTieNoStop, LengthsRow)


#: the scheme's settings and their defaults: ``order`` ``arrival`` admits no request in release order (the id relay
#: instead); ``release-order`` and ``tick`` are `prover.admission.Admitter`'s, which reads the rest under ``tick``;
#: ``log``, a directory for the engine's tick log (`prover.admission.engine`), none where empty
SETTINGS = {"order": "arrival", "hold_ms": 5000, "bucket_us": 100_000, "sat_us": 0, "free_us": 0, "token_ns": 0,
            "log": ""}


def _check(values: Mapping[str, Any]) -> str | None:
    if values["order"] not in ("arrival", *ORDERS):
        return f"order one of {['arrival', *ORDERS]}, not {values['order']!r}"
    if values["hold_ms"] < 1:
        return f"hold_ms a positive number of milliseconds, not {values['hold_ms']!r}"
    if values["order"] == "tick":
        try:
            tick_config({k: v for k, v in values.items() if k != "order"}, "tick")
        except ValueError as e:
            return str(e)
    return None


def settings(given: Mapping[str, Any]) -> dict[str, Any]:
    """The network attester's settings with ``given`` over their defaults (`SETTINGS`)."""
    return S.settings(NETWORK_ATTESTER, given)


def attach(exchange: Any, ctx: Mapping[str, Any]) -> Any:
    """The scheme on a served run's ``exchange`` under ``ctx["settings"]`` (`settings`). Under an ``order`` other than
    ``arrival``, its admission (`prover.admission.Admitter`, with ``ctx``'s ``source``, ``submit``, ``abort``, ``take``,
    ``pull``, ``log``, ``clock`` and ``schedule``), installed and returned; else the id relay beside the workload
    (`START_ID_RELAY`). Where the front and the engine run in two processes (``ctx["process"]``), each process's side of
    the certified tick, on the server's feed directory (`prover.admission.FrontFeed`, `prover.admission.engine`)."""
    s = settings(ctx.get("settings", {}))
    if ctx.get("process") in (H.FRONT, H.ENGINE) and "feed" not in ctx:
        raise ValueError(f"{NETWORK_ATTESTER.name}: its {ctx['process']} process was given no feed, the server's directory "
                         "the front and the engine share")
    if ctx.get("process") == H.FRONT:
        return FrontFeed(ctx["feed"], log=s["log"]).attach(exchange, NETWORK_ATTESTER.name)
    if ctx.get("process") == H.ENGINE:
        return engine(s, ctx["feed"]).install(exchange, NETWORK_ATTESTER.name)
    if s["order"] == "arrival":
        for r in requests(ctx):
            exchange.request(NETWORK_ATTESTER.name, r)
        return None
    kw = {k: ctx[k] for k in ("source", "submit", "abort", "take", "pull", "log", "clock", "schedule") if k in ctx}
    return Admitter(s, **kw).install(exchange, NETWORK_ATTESTER.name)


def output_bits(definition: Any) -> Callable[[Mapping[str, int]], int]:
    """A block's cost: its return bits at the public shape (its statics), all of which a wrong block may set."""
    return lambda shape: sum(bind(definition, **shape).signature()[1].leaf_widths())


#: the scheme's request that starts the id relay beside the workload
START_ID_RELAY = H.StartBeside("id-relay", id_relay.READERS)


def requests(ctx: Mapping[str, Any]) -> tuple[Any, ...]:
    """The id relay beside the workload under ``order`` ``arrival`` (`attach`), else nothing."""
    return (START_ID_RELAY,) if settings(ctx.get("settings", {}))["order"] == "arrival" else ()


def _scan_state_layout(shape: Mapping[str, int]) -> Layout:
    """The scan state row's layout at the scan's class (`ingress_scan.Class`, its statics by name): `ingress_scan.layout`'s
    fields in `state_bits`, with the prompt ids a first cell (``K0``) and a continuation (``K``) hold."""
    cls = IS.Class(**shape)
    return Layout(IS.state_bits(cls), IS.layout(cls), {"K0": cls.K0, "K": cls.K})


def _reading(row: Mapping[str, Any]) -> S.Reading:
    q = row["quantity"]
    return S.Reading(int(q["K"]), Fraction(str(q["eps"])), "quantity.eps", {"windows": int(q["windows"])})


NETWORK_ATTESTER = register(Scheme(
    name="network-attester",
    catalog={"attester": "verity.protocols.accounting.network.verifier.network_attester",
             "scan_state_rows": "verity.protocols.accounting.network.prover.scan_state_rows"},
    messages=workload.MESSAGES,
    framings={IdsProtocol.name: IdsProtocol},
    provides=("admission",),
    protocol="verity/certifier/v0",
    resource="egress",
    blocks=tuple(Block(d.id, output_bits(d)) for d in DEFINITIONS),
    assumptions=("trusted-device",),
    log=(Order("window", "sample"), Order("start", "first-tick")),
    # advice is `Phase.freeBits`, and jitter, schedules and cells the factors of the channel `ChanAt` (`Dj ^ k`, `Ss`,
    # `failAtMost cap j` ≤ ∏ (cap + 1)), so each is a term of `Phase.charges`, bounded under `Phase.ChargeHolds`
    charges=(
        Charge("advice", "scans[].free"),
        Charge("jitter", "scans[].jitter"),
        Charge("schedules", "schedules"),
        Charge("cells", "cells"),
        Charge("start", "start: a period's carried start at the state's full width, which `Phase` registers and no "
                        "term of `Phase.charges` holds", bounded=False),
        Charge("tied-entries", "scans[].faulty_entries: every slot entry bit a tied audit's incorrect blocks can set, "
                               "a cost of blocks outside the phase's partition, which no term of `Phase.charges` "
                               "holds", bounded=False),
    ),
    delta_scheme=Term("Claims.ecrBound", ("q",), lambda q: Fraction(q, 2**256)),
    over="bits: the ceiling of log2 of the number of records (the attester's rows and the channel) that committed values "
         "with the phase's incorrect blocks explain",
    under="bits: zero; the attester credits nothing",
    settings=SETTINGS,
    check_settings=_check,
    hooks=("declared", "steps"),
    requires={"one-labelling": "the network attester's rows (`verifier.network_attester_rows`)",
              "one-design": "the proof layer's sample of the scan; the attester's piece writes NOT RUN "
                                    "(`attester_piece.NOT_RUN`)"},
    ports=(Port("scan-state", "input", IngressScanStart.id, "sin", layout=_scan_state_layout, value=SCAN_STATE), Port("released", "output", EgressBucket.id, "o1")),
    requests=requests,
    attach=attach,
    processes={"id-relay": id_relay.process},
    parties=(Role("network-attester", "auditor", per="link", needs=("clock", "coins", "signing-key")),
             Role("workers", "developer", needs=("serve",)),
             Role("verifier", "auditor", needs=("audit", "component", "product"))),
    channels=(Channel("network-attester", "workers", ("out-opening", "records", "cells-salts")),
              Channel("network-attester", "verifier",
                      ("out-record", "certificate", "registration", "deployment", "groups"))),
    channel_schemas=NC.MANIFEST.messages,
    colocated=(("challenger", "network-attester"),),
    needs=frozenset({"windows", "ingress-relay"}),
    depends=("sampled-proofs",),
    pieces={"workers": S.Launch(
        binds=("gate", "peer"), starts=("gate", "peer"),
        entry="verity.protocols.accounting.network.scheme:launch_attester",
        executables=(S.Executable("verity/protocols/accounting/network/verifier/lean", "network-period"),
                     S.Executable("verity/protocols/accounting/network/verifier/lean", "network-rows")))},
    answers=S.Lacks("has no piece that answers outside an exchange"),
    record="communication",
    symbol="K",
    published=("bits", Fraction(1)),
    check=period_check.answer,
    reader=_reading,
    premises=("NetTiming.Attester.Phase.ChargeHolds", "NetTiming.Attester.RowsCommitted"),
    keeps_weight_copy=False,
    **dict.fromkeys(("tables", "output_format", "product_family", "y_committed", "salt_open_root", "kernels"),
                    S.Lacks("not a kernel scheme")),
))

ATTESTER_PIECE = "benchmarks/network_traces/attester_piece.py"
#: what the network attester's front asks of serve-openai: its id-form requests answered in the id form, an id event the
#: 503-byte payload of the front's 512-byte cells; every other request is answered as before
ID_FORM = "--protocol ids --id-payload 503"
#: network attestation's scan class (`attester_piece.V3C_SCAN_CLASS`, w63k4) as served_verdicts.sh's SLOTS,PROMPT,OUTPUTS,
#: LANES,EVERY
SCAN_CLASS = "8,32,4,4,16"
#: the most output tokens a front request asks for (`attester_piece.TOKENS`): its group stays in one execution phase
FRONT_TOKENS = 4


def _attester_hooks(cfg, params, bound, pinned):
    """The network attester's developer side: its front before the requests (CERTIFIER_FRONT, which the served window
    runs), whose requests ask for at most the window's steps, so each is in one window, and its piece
    (``CERTIFIER_CMD``) on the front's own window, which sets no class."""
    py = ".venv/bin/python"
    gate = f'--gate {shlex.quote(bound["gate"])} --peer {shlex.quote(bound["peer"])}'
    tokens = min(FRONT_TOKENS, cfg["workload"]["params"]["window-steps"])
    return {"SERVE_ARGS": ID_FORM,
            "CERTIFIER_FRONT": f'{py} {ATTESTER_PIECE} front "$1" "$2" "$3" {gate} --tokens {tokens}',
            "CERTIFIER_CMD": f'{py} {ATTESTER_PIECE} piece "$@"'}


def launch_attester(declaration: Scheme) -> tuple[Any, tuple[str, ...]]:
    """The network attester's developer piece (its `Launch.entry`): its hooks and the script they run. It launches and
    collects, and decides no verdict."""
    return _attester_hooks, (ATTESTER_PIECE,)


__all__ = ["ATTESTER_PIECE", "DEFINITIONS", "FRONT_TOKENS", "ID_FORM", "NETWORK_ATTESTER", "SCAN_CLASS", "SETTINGS",
           "START_ID_RELAY", "attach", "launch_attester", "output_bits", "requests", "settings"]
