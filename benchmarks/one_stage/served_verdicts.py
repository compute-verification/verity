"""The verdicts on one served window (`served_verdicts.sh`): the window it audits, the subprotocols after
SPOT, and the one record of every verdict (the block record).

    served_verdicts.py window SERVE_DIR [--slots N] [--window K]
    served_verdicts.py gate [--gate CMD] [--lane vllm] [--main FILE]
    served_verdicts.py pieces --serve-run DIR --window K --out DIR [--node NAME] [--only COMPONENT]... [--skip COMPONENT]...
                              [--product P]
    served_verdicts.py pearlplus DIR --served FILE --verify FILE [--unretained FILE] [--gate CMD] [--key-sha512 HEX]
                                 [--main FILE] [--coin COIN_PUT]
    served_verdicts.py pearlplus-period open DIR --gate CMD --name P [--scheme S] [--cubin FILE | --circuit-record FILE]
    served_verdicts.py pearlplus-period close DIR --gate CMD [--lane L --windows PEARL_DIR [--wait S]]
    served_verdicts.py pearlplus-period rule DIR --gate CMD --lane L --coin COIN_RECORD
    served_verdicts.py coin DIR --record FILE [--run RUN_ID] [--run-dir DIR]
    served_verdicts.py record --serve-run DIR --window K --sampled-proofs DIR --pieces DIR --out DIR [--node NAME]
                              [--product P] [--deferred WHY] [--steps LO,HI] [--off-node]
                              [--sampled-lean [--sampled-exe PATH]] [--key-sha512 HEX] [--lane L] [--window-tie DIR]
                              [--tree SHA [--tree-on BRANCH]] [--served-compare FILE] [--composite DIR]
    served_verdicts.py window-record SUMMARY_JSON
    served_verdicts.py witness < PATHS
    served_verdicts.py rebase SERVE_DIR SERVED_DIR < PATHS
    served_verdicts.py composed ANSWER DIR [--gate-exit RC]
    served_verdicts.py public-registration REGISTRATION
    served_verdicts.py coin-trees --challenger C --summary FILE
    served_verdicts.py coins --challenger C --summary FILE
    served_verdicts.py front-line SESSION
    served_verdicts.py failed|said RC [LOG|- ...]
    served_verdicts.py public-lines [--private FILE]
    served_verdicts.py publish OUT --private DIR

What leaves the node names no window and counts none of a window's own: no window's number, no count of the windows closed,
sealed or refused, and no engine step, request count or user of one; nor a load's tokens or span (`OWN_AT`), whose tokens
count its blocks; nor Pearl+'s exclusion tiles, their count and every tile's id (`OWN_AT`, `_tile_verdicts`), nor its
prover's give-up before the coin (`OWN_DOCS`, `GIVE_UP_TEXT`), which the retained rows' values decide.
``public-lines`` prints each line of stdin in its published form (`public_line`), the line in full
appended to ``--private`` (the serve dir's ``private/verdicts.txt``), and ``publish`` holds each file of OUT, once the run
is done, in full under DIR where it names or counts a window or holds a load's own field (`publish`), its public form in
its place. The class's padded counts, the period's declared T and S, the bench Build's requests and the
verifier's coins and samples are public. ``public-registration`` prints what the run publishes of a window's advice
docket (`public_registration`: its compiler's cap in place of the advice's length, which the advice tree's row count
states all the same), ``coin-trees`` why an
audit on the gate's coins took its sessions' coin trees from elsewhere (`coin_trees_refusal`, exit 1), ``coins`` the
V3C-AUDIT line's account of whose coins its audit's were and its coin trees (`coins_said`), and
``front-line`` the CERTIFIER-FRONT line from the front's session (`front_line`: `FRONT_PUBLIC`), ``failed`` a tool's
failure by its exception's class and exit status (`failed`) and ``said`` its last line where it passed, else that (`tool_said`).

``window``: the sealed window to audit, from a serve-openai run's ``seals.jsonl``, ``windows.jsonl`` and ``requests.jsonl``
(`pick`): sealed without error, its Commit passed, each of its requests checked and in no other window (V3b's window
circuit refuses a request carried across windows), at most ``--slots`` of them (its class's); the most requests, then the
earliest. ``--window K`` holds K to the same rules. ``--of-load FILE`` (a fixed load's lines, `verity_vllm.inputs.sender`'s
``--out``) takes the one window whose requests are that load's and no other, and ``--attested`` the one whose requests
are all the network attester's: neither a count nor a setting chooses it, and either refuses unless one window fits. Prints the pick (`pick`): ``{window, requests, users, steps,
refused}``, which the run keeps in the serve dir's ``private/window.json``.

``pieces``: each subprotocol after SPOT (`PIECES`: PoUW, PoUS, the bandwidth audit, the network attester) runs
its piece where one is named (``POUW_CMD``, ``POUS_CMD``, ``BANDWIDTH_CMD``, ``CERTIFIER_CMD``: a command run in the tree as
``bash -c CMD _ SERVE_DIR WINDOW DIR``, which writes its core answer to ``DIR/answer.json``), one at a time, each within
``<COMPONENT>_TIMEOUT`` (else ``PIECE_TIMEOUT``, default `PIECE_TIMEOUT`), else prints a ``MISSING`` line naming it, the
command it lacks and its owner. PoUS's piece runs before the bandwidth audit's: the bandwidth session is over
the store PoUS's piece holds, and that piece writes its answer to the bandwidth piece's ``DIR``, which ``BANDWIDTH_CMD``
collects. Each run leaves ``DIR/piece.json`` (`RAN`: the node it ran on, its exit, its limit). A piece
that ran but wrote no answer is its error, unless it refused to run (``DIR/refused.txt``, `REFUSED`, naming why), which
makes it ``refused`` with that sentence (`block_record.REFUSED`: undecided, never accepted, never dropped); no verdict
is made for a piece that didn't run. A development piece writes ``DIR/development.json`` (`DEVELOPMENT`) in place of a core answer: its verdict is printed as a development run's, and it credits nothing. A piece
that audited another server than the node it ran on says so in ``DIR/served.json`` (`SERVED`: ``{server, source}``,
``source`` ``fixture`` for an answer brought from a recorded run); one that names no server there leaves its answer about
the node it ran on. ``--only`` and ``--skip`` (a component each, repeatable) run those pieces alone, or all but those.
``--product`` (default onsite; served_verdicts.sh passes its PRODUCT) is the product the run composes (`block_record.PRODUCTS`): onsite's
every subprotocol, remote's SPOT and PoUW alone. A piece the product drops isn't run, whatever its command,
and prints a ``DROPPED`` line naming it.

``gate``: before serving with live coins from infra's gate (``--gate``, default ``H1_GATE``): refused while the key's lane
(``--lane``) has a period open, which a reserve would join, and with ``--root`` (a window's run root a session of the run
opens on, V3c's audit's) once an open in the ledger names that root, since the gate opens a root once, ever; prints one
``GATE`` line with the gate's latest pin and, with ``--main``, whether that pin is on the verifier's first-parent line of
main (a session pinned off it is refused).

``program``, ``period`` and ``pouw`` are PoUW H1's, which is deleted: each is refused by name (`H1_DELETED`).

``pearlplus``: PoUW's development piece under POUW=pearlplus (`served_pearlplus.sh`, POUW_CMD's default then), from Pearl+'s
own served run on the card after the window's serve: ``--served`` (`served_passes.py serve`'s document: every pass retained)
and ``--verify`` (its ``verify``'s: Pearl+'s verdict over all the passes, the reference replay of the tiles it samples).
Where either step wrote no document it refuses (`PEARLPLUS_NO_VERDICT`), exit 2, its sentence in ``refused.txt`` and a
POUW-REFUSED line, and writes no answer. Its verdict is accept when that run served within its caps across at least two pass boundaries, Pearl+'s verdict
accepts, and each pass's salt is the live session's that the serve names (`pearlplus_coins`: the session's tree as the gate
``--gate`` fetches it, the ledger and pins log that gate answers now and ``--main``'s line, held by `core_check.live_salt`
under the key ``--key-sha512`` pins), and a challenger sample (the coin record's ``draw``, `verify_run`'s "The coin record")
is its sample session's (`pearlplus_draw`: that session fetched the same way, held to the signed ledger and session); a
run whose coins are an in-process challenger's claims no verdict where nothing
failed (`UNCLAIMED`, `PEARLPLUS_WHY_IN_PROCESS`), and one served under the gate whose coins are not live rejects. Its
document carries the passes, that verdict's line and credit at the scheme's named assumption, where its coins came from
(`pearlplus_coins_from`) and its shares (`served_shares.pearlplus_shares`), and says why it credits nothing (`pearlplus_why`:
its passes are not the window SPOT audited; under PEARLPLUS_TIE, ``--tie W``, the window's own epoch, the
Pearl-C period its live salt session joined, `pearlplus_why_tied`). ``record`` composes an accepted one as PoUW's verdict of record
only where `pearlplus_of_record` holds it to the piece's own check of live coins (their signatures verified on the served
node only: the compose holds just the key file the piece hands to ``--key-sha512``), the lane ``--lane``'s own reserve and
Pearl-C period, a sample session the lane reserved first after that period closed and opened on the run's commitment, the
window's docket chained after Pearl+'s window record and its records audit tied to it accepted, that period's circuit the window-circuit record V3c held the
window's circuit to, and the Lean `PouwVerifier.PearlPlus` verdict on the same run; else its row stays missing, naming the
first condition it fails.

``pearlplus-period open``: the lane's Pearl-C period P, opened at ``--gate`` before Pearl+'s serve with cap 1 (the serve's
one salt session joins it) and circuit the SHA-512 of ``DIR/pearlplus-descriptor.json``: with ``--circuit-record``, the
served window's circuit record (served_verdicts.sh's PEARLPLUS-CIRCUIT, the circuit cut at Pearl+'s linears that step 3's
window-circuit writes), byte for byte; without it, `pearlplus_descriptor` (Pearl+'s protocol, ``--scheme`` and
``--cubin``'s SHA-512), which names no model. Its answer goes in ``DIR/period.json``; ``pearlplus-period close`` closes it
once the serve's session has ended (the untied piece's right before the verifier's sample), and the sample, which joins no
period, must be the lane's first reserve after the close. With ``--lane`` and ``--windows`` (the serve dir's ``pearl/``,
under PEARLPLUS_TIE) the close first reads the ledger, with ``--wait S`` for up to S seconds until it ends each window's
salt session, and records whether the lane's reserves since the open are the windows' salt sessions alone
(`pearlplus_period_close`), its line saying held or REFUSED; it closes either way, and the tied piece samples only after a
period that closed with its reserves held. ``pearlplus-period rule``, once the coin is flipped, prints the
``RULE`` line: whether the coin record ``--coin``'s sample session is lane ``--lane``'s first reserve after the close
(`pearlplus_period_rule`; exit 1 when BROKEN). ``coin``: the verifier's coin record published the moment it is
sampled (`coin`), ``--run-dir`` naming the run's Attempt as its custody where no put preserves it.

``record``: SPOT's answer from the records audit kept under ``--sampled-proofs`` (`benchmarks/prove_path/run.sh`'s with
``PP_RECORDS=1``, or V3c's: ``summary.json``, ``window/audit.json``, and ``window/registration.json``, the docket it
sampled against), read as `served_integrity.answer` reads it (Pearl+'s window tie from the same files under
``--window-tie DIR`` where it is given, the window tied to W); each other subprotocol's from
``--pieces/<component>/answer.json`` (`piece_row`), else a ``missing`` row naming why. Each answer is about the server
`piece_server` names: SPOT's the record's node, a piece's the one its ``served.json`` names, else the node it ran
on, and the wipe's the one its ``served.json`` names. They go into the window report (``verdicts.json``, `row_of`'s rows,
with each one's server) and from it the block record (``block-record.json``, `block_record.from_period`, held by
`block_record.check`), which composes only the answers about the record's node and carries the others as ``elsewhere``;
an answer that names no server isn't carried. An accepted answer whose outputs are not its subprotocol's certificate
(`block_record.subprotocol`) is its subprotocol's error, and so is a bandwidth answer whose profile names no term its error leaves
out (``named_terms``). `block_record` carries only an accepted subprotocol's certificate, so
where one about this node rejects or errs the composed verdict is ``reject`` and no block record is written. Otherwise
the verdict is the block record's, which is the compiled Lean ``onsite check``'s or ``remote check``'s (``remote-v1``'s
under Pearl+) answer, as `composite.py` wrote it to the public composite under ``--composite`` (`served_composite.sh`'s
$OUT/composite, `composite_answer`); a Lean rejection there rejects whatever the rows say. Without one composite of the
product's check, one whose Lean answered, or with a development one (composed on the served node or on a prover node),
no Lean answer is the record's, so it is ``undecided``, which the window report (``-window/v1``) and the ``RECORD`` line
name ``pending``; each says, as ``lean``, where its answer was read or why there is none. A composite not of the
record's window is refused (`composite_tie`): its SPOT case must name the digest of the docket R that
the SPOT audit the record reads sampled against (`window_digest`), and its commitment must bind its cases' digests.
Python accepts nothing here: the exit is 0 only for the Lean's accept, 1 for a reject, a refused composite or a block
record `block_record.check` refuses, 2 for pending.
The subprotocols are those of ``--product`` (default onsite; `block_record.PRODUCTS`): each one the
product drops is ``dropped`` in the report (`dropped_row`, whatever its piece left), not run and in no block record's
``missing``, so remote's is ``pending`` with nothing missing when SPOT and PoUW accepted here, whatever the
onsite pieces. The report's ``composed`` gives that verdict with its reason (`said`), one clause
a subprotocol: its verdict and, unless it accepted here, why; an accept here with what it holds beyond its verdict
(`terms`: when its weights were held and their share of the checkpoint, where its profile says, each δ of the block
record's at which its quantity is not established, and the bandwidth audit's error with the term it leaves out beside
it, `error_said`); PoUW's development piece reads ``development run, credits nothing`` with core's refusal of the
window's credit.
Prints one ``RECORD`` line, naming the product, each subprotocol's verdict
(the development piece's `DEV_VERDICT`), each missing one and why, each dropped one, each rejecting one, each one
elsewhere, and the reason, whose sampled proofs' clause gives its sample's bound at each δ without its count
(`bound_said`), which the serve dir's private copy alone holds (`bound_held`). The report and the line say where they
were composed (``composed_on``: the host, whether it is the record's node, what it read, `COMPOSED_FROM`, or off that
node `COPIED_FROM`, and whether the host is the auditor's device, `auditor_device`), and with ``--deferred`` what
the run left to a follow-on and why (``deferred``: `served_verdicts.sh`'s step 6 on a node short of V3c's audit's
memory, or its compose hand-off), which composes the final record on another host, the record's node still ``--node``,
and with ``--served-compare`` the served compare's verdict line (``served_compare``, `served_compare_line`).
``--steps LO,HI`` gives the window's engine steps, which the record otherwise picks from the serve dir's seals, windows
and requests (`pick`): a compose follow-on holds no requests file. ``--off-node`` composes from the served node's compose
hand-off (``composed_on``'s ``from`` `HANDED_FROM`, which says that a verdict handed in is taken as given, with every
check behind it: `TAKEN_AS_GIVEN`), and is refused on the record's node itself. ``--sampled-lean``
takes SPOT's row from this host's own run of its Lean verifier on the records audit's v2 case, its proofs and
public files (`lean_sampled_row`, ``--sampled-exe`` or built here), not from the summary and audit record the served node
wrote, so an edit to those leaves the row as it is. Each row with an answer says where it came from (its
``answer_from``, and the line's ``answers``): `LEAN_HERE`, `PEARLPLUS_FROM` for PoUW's where Pearl+'s piece is its
verdict of record, or `AS_WRITTEN` and the files it was read from.
``--tree SHA`` names the commit of the tree that composed it, and ``--tree-on`` the branch its host pinned it from, both in
``composed_on`` (the compose gate's: its pin line's).

``witness``: each path of stdin (relative to a serve dir) that is a served run's witness (`witness`: a committed value,
a prover's instance, private or salts file, the advice, a key, a sealed row dir, the requests and their traces), one ``WITNESS
PATH: WHY`` line each; exit 1 when there is one. A compose hand-off lists none, and its follow-on's serve dir holds none.

``rebase``: a hand-off's copy at SERVE_DIR of the served node's serve dir SERVED_DIR (`rebase`): each case among the
paths of stdin (relative to SERVE_DIR) that names ``blobs`` gets them under SERVE_DIR, a blob outside SERVED_DIR refused.
``composed``: the compose gate's answer ANSWER (`host.py`'s ``compose``) written out under DIR (`composed`), one
``GATE-COMPOSE`` line saying what it composed; exit 0 only when it composed and accepted.
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _p in (Path(__file__).resolve().parent, ROOT / "benchmarks" / "one_stage" / "interfaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

#: the window report's format, kept from the all-protocols runner whose report it was
REPORT_FORMAT = "verity/all-protocols-window/v1"
#: the block record's verdict by the window report's name for it: ``-window/v1`` calls an undecided window pending
WINDOW_VERDICT = {"undecided": "pending"}
#: ``record``'s exit by the window's verdict, as composite.py's by its Lean's: 0 only for the compiled Lean check's
#: accept; 1 for a reject (or a refused composite, or a block record its own check refuses), 2 for pending
RECORD_EXIT = {"accept": 0, "reject": 1, "pending": 2}


#: SPOT's owner, its row's
INTEGRITY_OWNER = "compute-accounting"

#: each subprotocol after SPOT, by its runner name: (its name, its component, its verifier's owner, what its piece
#: does), in the chain's order
PIECES: dict[str, tuple[str, str, str, str]] = {
    "work": ("PoUW", "pouw", "compute-accounting",
             "the reference replay of Pearl+'s own served run (served_pearlplus.sh, POUW=pearlplus)"),
    "space": ("PoUS", "pous", "memory-accounting",
              ("PoUS over the served model's int8 weights on the serving card, against the memory attester on "
               "vy-auditor-2; served_verdicts.sh sets it where the tree has benchmarks/pous/served_piece.sh and this node "
               "the challenger's key")),
    "bandwidth": ("the bandwidth audit", "bandwidth", "memory-accounting",
                  ("PoUB-P2's session over the store the PoUS piece holds, which that piece runs (POUS_CMD, "
                   "benchmarks/pous/served_piece.py) and BANDWIDTH_CMD collects (served_piece.py collect-bandwidth)")),
    "communication": ("the network certifier", "certifier", "network-accounting",
                      "the network certifier in front of the serving endpoint"),
}
#: the record's subprotocol outside this chain: serve() runs its piece (WIPE_CMD) on the card before its engine loads
WIPE = ("the wipe", "wipe", "memory-accounting", "the wipe of the serving card before its engine loads")
NAMES = {"integrity": "sampled proofs", **{k: v[0] for k, v in PIECES.items()}, "wipe": WIPE[0]}
INFERENCE_ONLY = "inference-only.json"
INFERENCE_ONLY_ROW = "inference-only"
#: how the inference-only verdict's reason begins where its verifier gave no verdict
UNDECIDED = "undecided: "
INFERENCE_ONLY_UNREAD = ("not read by the combined verdict: the composite combines verity/inference-only/v0's own "
                         "case, not this (c, k, n) verdict")
#: the network attester's request users, as the endpoint records them (`serve_commit.REQUEST_USERS`)
ATTESTER_USER = re.compile(r"user-[0-9]+")
#: under the serve dir's private/: the window the network attester's row is taken from, where it is its front's group's
#: and not the window of record (a tied run's fixed load, which the network attester does not carry)
FRONT_WINDOW = "front-window"
FRONT_GROUP = ("the network attester's verdict is over its front's group, a window of its own, and not over the tied "
               "load's requests: it stands as its own protocol's verdict, its case left out of the composite, and no "
               "certificate of it reaches the block record")
#: a piece's limit unless ``<COMPONENT>_TIMEOUT`` or ``PIECE_TIMEOUT`` sets one, as `timeout` reads it
PIECE_TIMEOUT = "60m"
#: what `pieces` leaves beside a piece it ran: the node it ran on, its exit and its limit
RAN = "piece.json"
#: a piece's refusal to run, naming why: it leaves its subprotocol missing
REFUSED = "refused.txt"
#: the server a piece's answer is about, where it isn't the node `pieces` ran it on (the wipe's always)
SERVED = "served.json"
#: a development piece's document, written in place of its core answer
DEVELOPMENT = "development.json"
#: Pearl+'s piece's refusal where its serve or its verify wrote no document: with no verdict it writes no answer
PEARLPLUS_NO_VERDICT = "pearlplus-no-verdict"


class PearlplusRefused(Exception):
    """Pearl+'s piece refused to write an answer (`pearlplus`); its sentence is in the piece's `REFUSED`."""


#: why a development piece's verdict credits nothing, so its subprotocol's row in the record stays missing
DEV_WHY = ("development run, credits nothing: the record takes no development piece as PoUW's verdict of record, and core's credit "
           "also needs the window in a period whose open commits its draw K, rows and weights")
_PEARLPLUS_UNTIED = ("development run, credits nothing: Pearl+ (pearl-c-h100-v1-h3) served its own seeded load on this card "
                     "after the window's serve, the BF16 checkpoint's linears as its FP8 product, so its passes are not the "
                     "window sampled proofs audited and nothing ties them to that window's Commit")
#: ... under PEARLPLUS_TIE, where serve-openai's Pearl+ arm served the window itself (`pearlplus`'s ``tie``)
_PEARLPLUS_TIED = ("development piece, which the record takes as PoUW's verdict of record, the prover side's, only where "
                   "pearlplus_of_record holds it: Pearl+ (pearl-c-h100-v1-h3) served the window itself, one epoch the window and its passes the "
                   "window's Commit steps, its verdict holding them to the window record W that the window's registration R "
                   "is chained right after (served_passes.py verify --tie W)")
_IN_PROCESS_WHY = ("its epoch's coins are an in-process challenger's, which the serving process holds, so it claims no "
                   "verdict: the reference replay's (verify_run.py) ACCEPT under coins the prover holds shows nothing")
#: the challenger a live session's coins are named from where its open answer carries no ``host`` (`challenger_of`)
UNNAMED_CHALLENGER = "the gate's challenger, which the session's open answer doesn't name"
#: `pearlplus_why` where its epoch's coins are an in-process challenger's: the piece then claims no verdict (`UNCLAIMED`)
PEARLPLUS_WHY_IN_PROCESS = f"{_PEARLPLUS_UNTIED}; {_IN_PROCESS_WHY}"
#: the same under PEARLPLUS_TIE (a live session's: `pearlplus_why_tied`)
PEARLPLUS_WHY_TIED_IN_PROCESS = f"{_PEARLPLUS_TIED}; {_IN_PROCESS_WHY}"
#: the Pearl+ piece's verdict where its epoch's coins are an in-process challenger's and nothing failed
UNCLAIMED = "unclaimed"
#: the replay's samples where verify ran without the gate (`verify_run.node_draw`)
PEARLPLUS_NODE_SAMPLES = ("the Lean verifier's IO.getRandomBytes in verify_run.py on the serving node, one source for the "
                        "run, taken once every pass's roots are read and recorded in the passes' coin.json before any "
                        "replay; the piece puts that record in the evidence store the moment it is drawn and holds the "
                        "verdict to it (`coin`, `pearlplus_coin`)")
#: `pearlplus_coins_from` where its epoch's coins are an in-process challenger's
PEARLPLUS_COINS_IN_PROCESS = {"draws": PEARLPLUS_NODE_SAMPLES,
                              "epoch": "an in-process challenger of epoch 0 (verity_bench's protocols_config, "
                                       "pouw.challenged), held by the serving process"}
#: `pearlplus_epoch_tied` where its epoch's coins are an in-process challenger's
PEARLPLUS_EPOCH_TIED_IN_PROCESS = ("an in-process challenger of the window's epoch over the Lean verifier's "
                                   "entropy (serve_pearl.in_process), held by the serving process")


def challenger_of(served: Mapping[str, Any] | None) -> str:
    """The challenger whose session the served run's coins are from: the host its gate's open answer names
    (``coins.session.host``, ``<host>.<session>``, the string the challenger's own record signs), the deployed auditor's
    or a local gate's scratch host alike; `UNNAMED_CHALLENGER` where the answer carries none."""
    session = ((served or {}).get("coins") or {}).get("session") or {}
    host = session.get("host") if isinstance(session, Mapping) else None
    name = host.partition(".")[0] if isinstance(host, str) else ""
    return name or UNNAMED_CHALLENGER


def pearlplus_why(challenger: str) -> str:
    """Why the Pearl+ piece's verdict (`pearlplus`) credits nothing, its epoch's salt coin a live session's on
    `challenger` (`challenger_of`)."""
    return (f"{_PEARLPLUS_UNTIED}; its epoch's salt coin is a live session's on {challenger}; and an ACCEPT is the "
            "reference replay's (verify_run.py), not PouwVerifier's ZK proof")


def pearlplus_coins_from(challenger: str) -> dict[str, str]:
    """Where Pearl+'s epoch coins and the replay's samples come from (`pearlplus`), its epoch's salt coin a live
    session's on `challenger` (`challenger_of`)."""
    return {"epoch": f"a salt-only session the record names as {challenger}'s, reserved through the lane's gate "
                     "before the engine started, opened on its salt epoch's root; core_check.live_salt held each "
                     "pass's salt to it (`pearlplus_coins`): coins' signatures checked on the served node only; not "
                     "checked by the auditor",
            "draws": f"a draw session of its own the record names as {challenger}'s (verify_run.py's "
                     "\"The run's draw\"), reserved withheld through the lane's gate once its Pearl-C period closed "
                     "and opened on the run's commitment once every pass's roots are read, its draw coin the run's "
                     "one source, recorded in the passes' coin.json before any replay and held to the session "
                     "(`pearlplus_draw`): coins' signatures checked on the served node only; not checked by the "
                     "auditor; the piece stores it as drawn and holds the verdict to it (`coin`, `pearlplus_coin`)"}


def pearlplus_epoch_tied(challenger: str) -> str:
    """Where the epoch's salt coin comes from under PEARLPLUS_TIE (``epoch``; ``draws`` as `pearlplus_coins_from`'s), a
    live session's on `challenger` (`challenger_of`)."""
    return (f"a salt-only session the record names as {challenger}'s for the window, reserved through the "
            "lane's gate as the window opened (serve_pearl.live), opened on its salt epoch's root; "
            "core_check.live_salt held each pass's salt to it (`pearlplus_coins`): coins' signatures checked on "
            "the served node only; not checked by the auditor")


#: `served_step.coins_of`'s ``source`` for a live session's coins
LIVE_SOURCE = "challenger"
#: the evidence store's kind for the verifier's coin record as `coin` publishes it: the record's own schema
#: (`verify_run.COIN_SCHEMA`)
COIN_KIND = "verity-pouw/pearl-plus-coin/v2"
#: `coin`'s copy of the coin record and the store's answer to its put, in the piece's directory
COIN_RECORD, COIN_PUT = "coin.json", "coin-put.json"
#: a `research run` id, whose Attempt keeps its ``out/`` (`coin`'s custody fallback)
RUN_ID = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")
#: the Pearl-C period's circuit descriptor in the piece (`pearlplus_period_open`), whose SHA-512 the period's open commits
PEARLPLUS_DESCRIPTOR = "pearlplus-descriptor.json"
PEARLPLUS_DESCRIPTOR_SCHEMA = "verity-pouw/pearl-plus-period/v1"
#: beside the window docket R under ``--window-tie``: the window-circuit record V3c held R's circuit to
#: (served_verdicts.sh's `window_tie`), whose SHA-512 Pearl+'s period must have committed
CIRCUIT_RECORD = "circuit-record.json"
#: the seconds `coin` gives each ``research data put`` before naming it unanswered
PUT_TIMEOUT_S = 300
#: what `record` composes from, until it runs on an auditor-side host reading each party's own answer
COMPOSED_FROM = ("each piece's answer file as the served node wrote it, not the party that produced it: an edit to that "
                 "file on the node changes its row")
#: what `record` composes from off the served node (a deferral's follow-on): the prover's copy of those files
COPIED_FROM = ("a prover-side copy of each piece's answer file as the served node wrote it, not the party that produced it: "
               "an edit to that file on the node or to its copy changes its row")
#: What a record says of a verdict it was handed: every check behind it, a piece's whole chain of them, ran where the
#: verdict was computed, and where the record is composed none of them is run again or held to anything.
TAKEN_AS_GIVEN = "a verdict computed off the auditor's device and handed in is taken as given, for the whole chain"
#: what `record --off-node` composes from: the served node's compose hand-off (`served_verdicts.sh handoff`), no witness
HANDED_FROM = ("the served node's compose hand-off, held here to the sums it wrote, never a witness: each row's answer from "
               "where its answer_from says, sampled proofs' recomputed here by its Lean verifier from the case's proofs "
               "where the hand-off carries them, and each other's answer file as the served node wrote it, not the party "
               f"that produced it: an edit to such a file on the node before its hand-off changes its row; {TAKEN_AS_GIVEN}")
#: where a row's answer came from (``answer_from``: the report's row, and the ``RECORD`` line's ``answers``)
AS_WRITTEN = "as the served node wrote it"
#: ... SPOT's under ``record --sampled-lean`` (`lean_sampled_row`)
LEAN_HERE = ("recomputed on this host by the Lean verifier (vstar-recursive sampled check) from the proofs and public "
             "files of the window's sampled proofs case, not the served node's answer")
MEMBERS_COVER = ("each sampled proof block proved in its own kind's statement against the one registration's roots on live "
                 "coins; members with no sampled proof block were not proved, and no theorem yet covers a sample that "
                 "misses members; the wires between members are not held; the circuit files' digests are fixed before "
                 "the coins in the parameters but not signed in the docket")
#: ... PoUW's where Pearl+'s piece is its verdict of record (`pearlplus_of_record`)
PEARLPLUS_FROM = (f"{AS_WRITTEN} (its piece files in pouw/): Pearl+'s Lean verdict of record (PouwVerifier.PearlPlus), "
                  "the prover side's, run on the served node over the retained passes; no auditor device re-checks it, since "
                  "its case (the passes' manifests with their salt key, and the opened tiles' rows) is witness")
#: what the composed verdict's reason (`said`) says of that row's accept
PEARLPLUS_SAID = "Pearl+'s Lean verdict of record, the prover side's, which no auditor device re-checks"
WEAKER_TIE = ("a weaker tie, not the tied guarantee: the window's records audit tied to W sampled {design}, not a stratified "
              "rate over a cut whose tables' readers are templates of their own, so its accept shows only that the tables "
              "its sampled proof blocks read hold the served model's own values at W's roots, not that every table W "
              "registers does")
#: ... the files the served node wrote that a row's answer is otherwise read from, by runner name (a piece's: its dir)
WRITTEN = {"integrity": "the records audit's summary.json and window/audit.json"}
#: A served run's witness, by a path's part (`witness`): what a compose hand-off never carries, each with why
WITNESS_DIRS = {"values": "a committed value file", "prover": "a prover's input file (its rows and salts)",
                "served_rows": "a window's committed rows and salts", "certifier-front": "the network attester front's group (its prompts)"}
#: ... by the serve dir's top directory
WITNESS_TOPS = {"windows": "a sealed row dir (the window's rows)", "advice": "the advice and its opening"}
#: ... by a file's name (`fnmatch`)
WITNESS_NAMES = {"inst*.bin": "a prover's instance file (its rows)", "*-private.bin": "a private file (prover only)",
                 "*salt*.bin": "a commitment's salts (prover only)", "*.key": "a key", "requests.jsonl": "the requests' prompt and served ids, rows and seeds",
                 "request_users.jsonl": "the requests' users", "step_trace.jsonl": "the step trace (each request's rows)",
                 "advice_trace*.jsonl": "the advice trace (each request's rows)", "load*.jsonl": "the load's requests",
                 "tokens*.json": "the requests' token ids", "match_stoch.json": "the sampler's rows", "*.npy": "a value array",
                 "*.npz": "a value array", "*.pt": "a tensor", "*.safetensors": "a checkpoint's weights"}
#: a development piece's verdict in the ``RECORD`` line, in place of its row's ``missing``
DEV_VERDICT = "development run, credits nothing"
#: Pearl+'s verdict's protocol (`verify_run.PROTOCOL`)
PEARLPLUS_PROTOCOL = "verity/pouw/pearl-plus/v1"
#: the Python verify's core answer in its piece (`served_pearlplus.sh`'s step 4, a measurement that decides nothing), and
#: the Lean `PouwVerifier.PearlPlus` verdict on the passes and coin record there (`pouw-pearlplus`'s printout), Pearl+'s
#: verdict
PEARLPLUS_ANSWER, PEARLPLUS_LEAN = "pearlplus-answer.json", "pearlplus-lean.json"
#: the δ the Lean verdict's ε is at (`PouwVerifier.PearlPlus.DELTA`, which `pouw-pearlplus` doesn't print)
PEARLPLUS_DELTA = "1/100"
#: why the piece rejects where the Lean wrote no verdict
PEARLPLUS_NO_LEAN = "no Lean PouwVerifier.PearlPlus verdict, so Pearl+ has none"
#: the window's docket R as its records audit under ``--sampled-proofs`` sampled against (a5's copy), and that
#: audit's summary (`spot_row`)
WINDOW_REGISTRATION, AUDIT_SUMMARY = "window/registration.json", "summary.json"
#: why `pearlplus_live_sample` refuses a node's sample source
PEARLPLUS_NODE_SAMPLE = ("the verifier's draw source is drawn on the serving node (verify_run.node_draw, PEARLPLUS_NODE_SAMPLES), "
                       "not a challenger coin opened on the run commitment (served_passes.py verify under "
                       "PEARLC_CHALLENGER_GATE)")
#: the period this serve dir's windows joined, its open's and close's answers (``period``)
PERIOD = "period.json"
def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def _refusal(k: int, seal: Mapping[str, Any], line: Mapping[str, Any] | None, mine: Sequence[Mapping[str, Any]],
             slots: int) -> str | None:
    """Why window ``k`` can't be audited, None when it can."""
    if "error" in seal:
        return f"its seal failed: {seal['error']}"[:300]
    if not line or not line.get("passed"):
        return "its Commit did not pass"
    if not mine:
        return "no request"
    carried = sorted(r.get("user") or r.get("engine_id") for r in mine if r.get("windows") != [k])
    if carried:
        return f"requests carried across windows: {carried}"
    unchecked = sorted(r.get("user") or r.get("engine_id") for r in mine if not (r.get("check") or {}).get("passed"))
    if unchecked:
        return f"requests whose check failed: {unchecked}"
    if len(seal.get("requests") or ()) != len(mine):
        return f"{len(seal.get('requests') or ())} sealed requests, {len(mine)} with a record"
    if len(mine) > slots:
        return f"{len(mine)} requests, more than the class's {slots} slots"
    return None


def pick(serve_dir: Path, slots: int, want: int | None = None, users: Sequence[str] | None = None,
         attested: bool = False) -> dict[str, Any]:
    """The window to audit (module docstring); ValueError when none qualifies, or ``want`` doesn't. ``users``: the one
    window whose requests are exactly those users', and ``attested`` the one whose requests are all the network
    attester's (`ATTESTER_USER`); either refuses unless exactly one window qualifies."""
    seals = {int(s["window"]): s for s in _lines(serve_dir / "seals.jsonl")}
    lines = {int(w["window"]): w for w in _lines(serve_dir / "windows.jsonl")}
    reqs = _lines(serve_dir / "requests.jsonl")
    refused: dict[int, str] = {}
    ok: list[tuple[int, list[dict[str, Any]]]] = []
    for k in sorted(seals):
        mine = [r for r in reqs if k in (r.get("windows") or ())]
        why = _refusal(k, seals[k], lines.get(k), mine, slots)
        if why is None:
            ok.append((k, mine))
        else:
            refused[k] = why
    if want is not None:
        if want not in seals:
            raise ValueError(f"window {want} has no seal (sealed: {sorted(seals)})")
        if want in refused:
            raise ValueError(f"window {want}: {refused[want]}")
        ok = [x for x in ok if x[0] == want]
    if users is not None or attested:
        what = "the fixed load's requests and no other" if users is not None else "the network attester's requests alone"
        fit = [(k, mine) for k, mine in ok
               if (sorted(r.get("user") or "" for r in mine) == sorted(users) if users is not None
                   else mine and all(ATTESTER_USER.fullmatch(r.get("user") or "") for r in mine))]
        if len(fit) != 1:
            raise ValueError(f"{len(fit)} sealed windows hold {what} ({[k for k, _ in fit]}), and the run names one: "
                             f"{json.dumps(refused, sort_keys=True)}")
        ok = fit
    if not ok:
        raise ValueError(f"no sealed window to audit: {json.dumps(refused, sort_keys=True)}")
    k, mine = max(ok, key=lambda x: (len(x[1]), -x[0]))
    lo = sum(int((w.get("commit") or {}).get("steps") or 0) for j, w in lines.items() if j < k)
    hi = lo + int((lines[k].get("commit") or {}).get("steps") or 0)
    return {"window": k, "requests": len(mine), "users": sorted(r.get("user") or "" for r in mine), "steps": [lo, hi],
            "refused": {str(j): why for j, why in refused.items()}}


#: A served window's own fields, never published: the windows (closed, sealed or joined), its users, its engine steps and
#: whether its requests filled the slots, its unpadded population, its advice's length and its windows' advice by number.
#: And the exact credit of a Pearl+ run held to its class's fixed load (``exact_work``: `verify_run.verify_passes`'
#: profile's, the Lean's printout's and `served_passes.py lean`'s record of both): the credit, credited tiles and MACs the
#: rows the passes exclude and give up decide, which the published work replaces by its grid form (`verify_run.grid_work`).
OWN = frozenset({"windows", "users", "engine_steps", "every_slot", "population", "advice_bits", "bits_by_window",
                 "exact_work"})
#: Fields that name a window or count its own where their value holds a number: the window's number, its steps, its
#: requests' count, the windows refused. A window named ``all``, the bench Build's requests by name and a refusal's reason
#: stay.
NUMBERED = frozenset({"window", "steps", "requests", "refused"})
#: A load's own fields, held by path as (parent, key), the parent the nearest key above the field through lists: the
#: tokens of each load of `served_overhead.py metrics`' aggregate, which over a block's outputs are the exact block
#: count, and its span. Held by path, not as bare keys, so the inference-only policy's ``n_inputs.span_s`` stays as
#: written, and with it the policy digest its verdict cites. And a tied Pearl+ piece's ``load.window_steps``, the
#: picked window's steps under another name, held by its path so the caller's own setting, ``serving.window_steps``
#: (`onsite_record.py --window-steps`), stays.
#: And what Pearl+'s verdict's profile says of its exclusion tiles (`verify_run.verify_passes`): their count, and the
#: tiles it replayed by id, the exclusion tiles among the sampled. Which rows a pass gives up, so which tiles are exclusion
#: tiles, is the retained values' to decide: neither is an output, structure, a coin or a commitment. A sampled tile's id
#: is the coin's through the passes' tickets, which the given-up rows weigh, so the list is held whole.
OWN_AT = frozenset({("loads", "tokens"), ("loads", "span_s"), ("load", "window_steps"), ("profile", "exclusion_tiles"),
                    ("profile", "tiles")})
#: Fields held where their value is a document. ``give_up`` is Pearl+'s prover's give-up before the coin (`give_up`, in
#: the piece's ``development.json`` and its line): the A and B rows it gave up, the tiles over a cap before it and left
#: after, the tiles the card marks non-finite and those replayed on the CPU, the passes it finished and whether it left
#: none over a cap. The retained values decide each, and they are the counts the grid form keeps out of the credit
#: (`verify_run.grid_work`), so the document is held whole. That form's own ``give_up``, the give-up in MACs padded up to
#: the window grid, is a number among the work and stays, with the samples over a cap and ε.
OWN_DOCS = frozenset({"give_up"})
#: what a published line says in place of the give-up's own counts
GAVE_UP = "(its counts stay on the node, in the serve dir's private)"
#: The give-up's counts in text: `served_passes.py volunteer`'s line and a pass's (`volunteer_pass`), from its passes'
#: or tiles' count to the line's end (the passes finished, the tiles replayed and over a cap, the rows given up, by call
#: and weight in a pass's, the credit kept and the replay's seconds); the count of tiles a dropped round left over a cap
#: (`served_passes.py retain`'s reason); and an `OWN_DOCS` document quoted in a string.
GIVE_UP_TEXT = (
    (re.compile(r"\b((?:PASS|FAIL): )(?:[\d,]+ passes, )?[\d,]+ tiles (?:replayed|counted on the device)\b[^\n]*"),
     r"\1" + GAVE_UP),
    (re.compile(r"\b[\d,]+ (tiles left over a cap)\b"), r"\1"),
    (re.compile(r'\\?"(?:%s)\\?": ?\{[^{}]*\}(?:, ?)?' % "|".join(sorted(OWN_DOCS))), ""),
)
#: A window's number, step range or count in text, each with what the published text says in its place; and its
#: population where a stream's rule states it, the denominator of a rate stated as K over the N proof blocks the window
#: registers (the form `served_verdicts.sh`'s preflight asks for, which a refusal quotes): the published text keeps
#: only the design, K of each N
WINDOW_TEXT = (
    (re.compile(r'\\?"(?:%s)\\?": ?(?:-?\d+(?:\.\d+)?|\[[^\]]*\]|\{[^{}]*\}|true|false)(?:, ?)?'
                % "|".join(sorted(OWN | NUMBERED))), ""),
    (re.compile(r"(?<![A-Za-z0-9_])w\d+(?![A-Za-z0-9_])"), "wK"),
    (re.compile(r"(?i)\b(windows?)([ :=#]+)\d+(?!\d|[,.]\d)"), r"\1\2K"),
    (re.compile(r"\b\d+ ((?:sealed |closed )?window\(s\))"), r"\1"),
    (re.compile(r"\b\d+ (sealed|closed)\b"), r"\1"),
    (re.compile(r"(?i)\b(steps?) \[\d+, ?\d+[\])]"), r"\1"),
    (re.compile(r"\b\d+/\d+ ok\b"), "ok"),
    (re.compile(r"\b((?:stratified-|work-)?rate:\d+)/\d+"), r"\1/N"),
)
#: A window's number in a file's path under the published dir: the file is held whole
NAMED = re.compile(r"(?<![A-Za-z0-9])w\d+(?![A-Za-z0-9])")
_OPEN = re.compile(r'\{\s*"')
_FIELD = re.compile(r'\\?"(?:%s)\\?":|\\?"(?:%s)\\?": ?\{'
                    % ("|".join(sorted(OWN | NUMBERED)), "|".join(sorted(OWN_DOCS))))
_AT_KEYS = {p: "|".join(sorted(k for q, k in OWN_AT if q == p)) for p in sorted({p for p, _ in OWN_AT})}
#: An `OWN_AT` field in text, by parent: the parent's list opening, and each of its keys holding a number inside that list
AT_TEXT = tuple((re.compile(r'\\?"%s\\?": ?\[' % re.escape(p)),
                 re.compile(r'\\?"(?:%s)\\?": ?-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?:, ?)?' % keys))
                for p, keys in _AT_KEYS.items())
_BRACKET = re.compile(r"[\[\]]")
#: an `OWN_AT` field in a JSON object cut short: its parent's key, then one of its keys
_AT_FIELD = re.compile("|".join(r'\\?"%s\\?":.*?\\?"(?:%s)\\?":' % (re.escape(p), keys) for p, keys in _AT_KEYS.items()))
#: what a published line says in place of a JSON object cut short that names a window's or a load's own field
CUT = "{…} (cut short; in full in the serve dir's private/verdicts.txt)"
#: a file `publish` reads larger than this is held whole
PUBLISH_MAX = 256 << 20


def _numbered(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        return v.isdigit()
    if isinstance(v, Mapping):
        return any(_numbered(x) for x in (*v.keys(), *v.values()))
    return isinstance(v, list) and any(_numbered(x) for x in v)


def _tile_verdicts(key: Any, value: Any) -> bool:
    """Whether the field is a run verdict's replay by tile (Pearl+'s ``verdicts``, the reference's and the Lean's: each
    sampled and each exclusion tile by pass and id), held as the profile's ``tiles`` are (`OWN_AT`). A record's verdicts by
    protocol's name are no list and stay."""
    return key == "verdicts" and isinstance(value, list) and any(isinstance(v, Mapping) and "tile" in v for v in value)


def _held(key: Any, value: Any, parent: Any = None) -> bool:
    if (parent, key) in OWN_AT or _tile_verdicts(key, value) or (key in OWN_DOCS and isinstance(value, Mapping)):
        return True
    if key == "refused":
        return not isinstance(value, str)
    return key in OWN or (key in NUMBERED and _numbered(value))


def _at_text(s: str) -> str:
    """``s`` with no `OWN_AT` field in the list its parent's key opens (to the text's end where the list is cut short)."""
    for opening, field in AT_TEXT:
        out, i = [], 0
        while m := opening.search(s, i):
            end, depth = len(s), 1
            for b in _BRACKET.finditer(s, m.end()):
                depth += 1 if b.group() == "[" else -1
                if not depth:
                    end = b.end()
                    break
            out += [s[i:m.end()], field.sub("", s[m.end():end])]
            i = end
        s = "".join(out) + s[i:]
    return s


def public_text(s: str) -> str:
    """``s`` with each window number, step range and window count `WINDOW_TEXT` finds in its published form, none of the
    counts of Pearl+'s give-up (`GIVE_UP_TEXT`), and no load's own field (`OWN_AT`)."""
    for pat, sub in (*WINDOW_TEXT, *GIVE_UP_TEXT):
        s = pat.sub(sub, s)
    return _at_text(s)


def public_json(o: Any, parent: Any = None) -> Any:
    """``o`` with no field of a window's own (`OWN`, `NUMBERED`), of a load's or of Pearl+'s exclusion tiles (`OWN_AT`,
    by ``parent``, the key ``o`` is under; `_tile_verdicts`), no give-up document (`OWN_DOCS`) and each string in its
    `public_text` form."""
    if isinstance(o, Mapping):
        return {public_text(k) if isinstance(k, str) else k: public_json(v, k) for k, v in o.items() if not _held(k, v, parent)}
    if isinstance(o, list):
        return [public_json(v, parent) for v in o]
    return public_text(o) if isinstance(o, str) else o


def public_line(line: str) -> str:
    """A verdict line as the run publishes it: each JSON object in it in its `public_json` form, one cut short that names a
    window's or a load's own field or a give-up document replaced by `CUT` to the line's end, and the rest in its
    `public_text` form."""
    dec, out, i = json.JSONDecoder(), [], 0
    while (j := line.find("{", i)) >= 0:
        out.append(public_text(line[i:j]))
        try:
            obj, end = dec.raw_decode(line, j)
        except ValueError:
            if _OPEN.match(line, j) and (_FIELD.search(line, j) or _AT_FIELD.search(line, j)):
                out.append(CUT + ("\n" if line.endswith("\n") else ""))
                return "".join(out)
            out.append("{")
            i = j + 1
            continue
        pub = public_json(obj)
        out.append(line[j:end] if pub == obj else json.dumps(pub))
        i = end
    out.append(public_text(line[i:]))
    return "".join(out)


def public_lines(src: Any, dst: Any, private: Path | None) -> None:
    """Each line of ``src`` in full to ``private`` (where the serve dir it's under exists) and in its `public_line` form to
    ``dst``, line by line."""
    keep = None
    if private is not None and private.parent.parent.is_dir():
        with contextlib.suppress(OSError):
            private.parent.mkdir(mode=0o700, exist_ok=True)
            keep = open(private, "a")
    try:
        for line in iter(src.readline, ""):
            if keep is not None:
                keep.write(line if line.endswith("\n") else line + "\n")
                keep.flush()
            dst.write(public_line(line))
            dst.flush()
    finally:
        if keep is not None:
            keep.close()


def public_doc(text: str) -> str:
    """A published file's text in its public form: a JSON document's `public_json`, else each line's `public_line`; ``text``
    itself where nothing in it names or counts a window or holds a load's own field."""
    try:
        doc = json.loads(text)
    except ValueError:
        return "".join(public_line(x) for x in text.splitlines(keepends=True))
    pub = public_json(doc)
    return text if pub == doc else json.dumps(pub, indent=1) + "\n"


def publish(out: Path, private: Path) -> list[str]:
    """Every file under ``out``, the dir that leaves the node, in its public form: one `public_files.PUBLIC` doesn't list
    under ``OUT/``, one whose path names a window, or that `PUBLISH_MAX` keeps from being read, held whole, and one whose
    text names or counts a window or holds a load's own field, Pearl+'s exclusion tiles, its exact credit or its prover's
    give-up (`public_doc`), or holds a field `public_files.WITHOUT` keeps private, held in full, under ``private`` at its
    path under ``out``, its public form left in its place. A file that isn't UTF-8 text is held whole. The paths held,
    in order."""
    import public_files as PF

    held: list[str] = []
    for p in sorted(out.rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        rel = p.relative_to(out).as_posix()
        dst = private / rel
        private.mkdir(mode=0o700, parents=True, exist_ok=True)
        dst.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        data = p.read_bytes() if p.stat().st_size <= PUBLISH_MAX else None
        if data is None or NAMED.search(rel) or PF.listed("OUT/" + rel) is None:
            shutil.move(p, dst)
            held.append(rel)
            continue
        text = data.decode(errors="replace")
        if text.encode() != data:
            shutil.move(p, dst)
            held.append(rel)
            continue
        pub = public_doc(text)
        if drop := PF.without("OUT/" + rel):
            with contextlib.suppress(ValueError):
                if isinstance(doc := json.loads(pub), dict) and set(drop) & doc.keys():
                    pub = json.dumps({k: v for k, v in doc.items() if k not in drop}, sort_keys=True) + "\n"
        if pub != text:
            dst.write_bytes(data)
            p.write_text(pub)
            held.append(rel)
    return held


#: The advice registration's fields that state the advice's own length, which `public_registration` replaces with the
#: cap's. The advice is committed at its own length, so its tree's row count is public all the same.
PADDED = ("binding.advice.bits", "rows")


def _no_digest(compiler: Mapping[str, Any]) -> dict[str, Any]:
    return {k: _no_digest(v) if isinstance(v, Mapping) else v for k, v in compiler.items() if k != "digest"}


def public_registration(reg: Mapping[str, Any]) -> dict[str, Any]:
    """What the run publishes of a window's advice registration: its compiler's ``cap_bits`` in place of the advice's
    length (`PADDED`), the one length the class declares (its rows at ``row_bits`` each), which hides nothing while the
    advice is committed at its own row count; no count or refusal of its requests, no window's number (`public_json`),
    no digest of its compiler, which is built from the description and so tells descriptions apart, and the rest as
    registered. ``padded`` names the fields replaced."""
    cap, row_bits = int(reg["compiler"]["cap_bits"]), int(reg["row_bits"])
    out = {k: v for k, v in reg.items() if k not in ("requests", "compiled", "refused")}
    out["compiler"] = _no_digest(reg["compiler"])
    out["binding"] = {**reg["binding"], "advice": {**reg["binding"]["advice"], "bits": cap}}
    if isinstance(reg["binding"].get("compiler"), dict):
        out["binding"]["compiler"] = _no_digest(reg["binding"]["compiler"])
    out["rows"] = -(-cap // row_bits)
    return public_json({**out, "padded": list(PADDED)})


def challenger_coins(challenger: str) -> str:
    """Whose coins a records audit's ``--challenger`` (run.sh's PP_CHALLENGER) are, as the SAMPLED-PROOFS line says
    them: the auditor device's gate, or a development session (local or loopback), which no record of the auditor's takes.
    No address: the gate is named by its kind alone."""
    if challenger.startswith("gate:"):
        return "the gate's, the auditor's challenger"
    return f"a development challenger's ({challenger.split(':', 1)[0]}), not the auditor's"


def coins_said(challenger: str, summary: Mapping[str, Any] | None) -> str:
    """The V3C-AUDIT line's account of its audit's coins: whose they were (`challenger_coins`) and its ``summary``'s
    ``coin_trees`` (none where the audit wrote no summary or reached no session)."""
    return f"coins {challenger_coins(challenger)}, coin_trees {(summary or {}).get('coin_trees') or 'none'}"


def coin_trees_refusal(challenger: str, summary: Mapping[str, Any] | None) -> str | None:
    """Why a records audit on the gate's coins (``challenger`` ``gate:USER@HOST``) can't be accepted for its sessions'
    coin trees, from its ``summary.json`` (``summary``), or None. The auditor's Lean verifier takes a `--zk` session's
    coins only from the release log of the session that sampled the sampled proof blocks (`OneStageZk.treesHeld`), so
    the summary must say the sessions took their trees from it (``coin_trees`` ``window``, `a5.window_trees`). A
    development session's coins, a plain audit (whose one session runs on a tree of its own by design) and an audit that
    reached no session (no summary, or one naming no ``coin_trees``, which its own verdict refuses) are None, as before."""
    s = summary or {}
    if not challenger.startswith("gate:") or s.get("plain") or "coin_trees" not in s or s["coin_trees"] == "window":
        return None
    return (f"the audit's summary says coin_trees={s['coin_trees']!r}: the sampled proof blocks' sessions took their coin "
            "trees from somewhere other than the gate's session that sampled them, which the auditor's Lean verifier "
            "refuses (OneStageZk.treesHeld); serve from a tree whose a5.window_trees takes them from that session, "
            "with FLOCK_CHALLENGER unset")


#: The network attester front's session fields the CERTIFIER-FRONT line says. The rest (its prompts, request, kept fetch, gate
#: and verdict's cells) count or name the group's requests, and stay in the serve dir.
FRONT_PUBLIC = ("rc", "run", "ended", "session_s", "pinned", "leaked_removed")


def front_line(session: Mapping[str, Any]) -> str:
    """The CERTIFIER-FRONT line's account of the front's ``session.json``: `FRONT_PUBLIC` alone, or that it refused."""
    if "refused" in session:
        return "refused (its reason in the serve dir's private/certifier-front/session.json)"
    return json.dumps({k: session[k] for k in FRONT_PUBLIC if k in session})


#: a traceback's last line: its exception's class, dotted or not, then its message
EXCEPTION = re.compile(r"^(?:[A-Za-z_]\w*\.)*([A-Za-z_]\w*(?:Error|Exception))(?::|$)")


def _logs(logs: Sequence[str]) -> list[str]:
    return [line for p in logs for line in (sys.stdin.read() if p == "-" else
                                            Path(p).read_text(errors="replace") if Path(p).is_file() else "").splitlines()]


def _class(lines: Sequence[str]) -> str:
    found = [m[1] for line in lines if (m := EXCEPTION.match(line))]
    return found[-1] if found else "no exception"


def _refused(lines: Sequence[str]) -> str | None:
    import public_files as PF

    return PF.refusal(next((line for line in reversed(lines) if line.strip()), ""))


def failed(rc: object, logs: Sequence[str]) -> str:
    """A tool's failure by its refusal's name where its log's last line is one of our tools' fixed refusals
    (`public_files.REFUSALS`), else by its last exception's class, and its exit status (``-`` is stdin): a tool's log
    holds paths, hosts and its inputs' values, so no line of it is said."""
    lines = _logs(logs)
    if name := _refused(lines):
        return f"(refused {name}, exit {rc}): its log is in the run's private directory"
    return f"({_class(lines)}, exit {rc}): its log is in the run's private directory"


def tool_said(rc: object, logs: Sequence[str]) -> str:
    """Our tool's own last line where it passed (its verdict), else its refusal's name (`failed`) or its last exception's
    class alone, for a line that says the exit status itself."""
    lines = _logs(logs)
    if str(rc) != "0":
        name = _refused(lines)
        return f"{f'refused {name}' if name else _class(lines)}; its log is in the run's private directory"
    return next((line for line in reversed(lines) if line.strip()), "")


def not_listening(log: Path, rc: int) -> str:
    """A serve that never listened, by `failed`: its log holds paths and the host's address."""
    return f"not listening {failed(rc, [str(log)])}"


def _unset(comp: str, what: str) -> str:
    return f"{comp.upper()}_CMD unset: {what}"


def dropped_row(key: str, product: str) -> dict[str, Any]:
    """The row of a subprotocol ``product`` drops (`block_record.PRODUCTS`, its subprotocols being the components): not run."""
    from verity_e2e_demo import block_record as BR

    by_subprotocol = {"integrity": NAMES["integrity"], **{v[1]: v[0] for v in PIECES.values()}, "wipe": WIPE[0]}
    return {"verdict": "dropped", "owner": PIECES.get(key, WIPE)[2],
            "why": f"not run: the {product} product composes {' and '.join(by_subprotocol[x] for x in BR.PRODUCTS[product])} alone"}


def pieces(serve_dir: Path, window: int, out: Path, env: Mapping[str, str] | None = None,
           node: str | None = None, only: Sequence[str] | None = None, skip: Sequence[str] = (),
           product: str = "onsite") -> dict[str, dict[str, Any]]:
    """Each piece of `PIECES` run where ``<COMPONENT>_CMD`` names it, under `timeout` at its limit, else named missing;
    one line each on stdout, its verdict the one `piece` gives the record. ``only`` and ``skip`` name components: those
    alone, or all but those, so one piece can run beside the others (`served_verdicts.sh` starts the network attester's early).
    A piece ``product`` drops isn't run, whatever its command (`dropped_row`)."""
    from verity_e2e_demo import block_record as BR

    env = dict(os.environ if env is None else env)
    node = node or socket.gethostname()
    got: dict[str, dict[str, Any]] = {}
    for key, (name, comp, owner, what) in PIECES.items():
        if (only is not None and comp not in only) or comp in skip:
            continue
        if comp not in BR.PRODUCTS[product]:
            got[comp] = {k: v for k, v in dropped_row(key, product).items() if k != "owner"}
            print(f"DROPPED {name} ({owner}): {got[comp]['why']}", flush=True)
            continue
        cmd = env.get(f"{comp.upper()}_CMD")
        if not cmd:
            got[comp] = {"verdict": "missing", "why": _unset(comp, what)}
            print(f"MISSING {name} ({comp.upper()}_CMD unset; {owner}): {what}", flush=True)
            continue
        d = out / comp
        d.mkdir(parents=True, exist_ok=True)
        limit = env.get(f"{comp.upper()}_TIMEOUT") or env.get("PIECE_TIMEOUT") or PIECE_TIMEOUT
        with open(d / "piece.log", "w") as log:
            rc = subprocess.run(["timeout", "-k", "1m", limit, "bash", "-c", cmd, "_", str(serve_dir), str(window),
                                 str(d)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, check=False).returncode
        (d / RAN).write_text(json.dumps({"server": node, "rc": rc, "limit": limit}, sort_keys=True) + "\n")
        label, (row, _) = name.upper(), piece(key, out)
        if "development" in row:
            got[comp], label = {**development_line(row["development"]), "rc": rc}, f"{label} (development run, credits nothing)"
        else:
            got[comp] = {**{k: row[k] for k in ("verdict", "reason", "why") if row.get(k) is not None}, "rc": rc}
        print(f"{label} {json.dumps(got[comp], sort_keys=True)}", flush=True)
    return got


def development_line(doc: Mapping[str, Any]) -> dict[str, Any]:
    """A development piece's line (`pouw`'s document): its verdict, why it rejects, the window's session and how each
    check came out, core's credit refusal, and that it credits nothing. `pearlplus`'s: its verdict, why it rejects, where
    its epoch's coins came from, its live session and `pearlplus_coins`' refusal (None when it accepted, ``unchecked``
    when it didn't run), its load, passes and the rounds dropped before the coin, its tie to W (under PEARLPLUS_TIE, else
    None), Pearl+'s verdict's credit (its credited
    and credited cells' MACs and ε), the verifier's coin as published (its store id, whether the put preserved it and
    what it said if not, and the record's SHA-512), the prover's give-up (`give_up`), its shares at the tensor clock, the
    unretained serve's wall and declared share there, and the prover's replay's CPU seconds, cores, the cores it takes to
    keep pace, the card's seconds, its node and the passes it finished (each None without one, or why it has none), the
    execution phase as the challenger saw it (its start, end and seconds on the auditor device's clock, `execution_phase`,
    with the serve's own wall beside it under ``prover``, the served node's clock), what Pearl+'s Python check and Lean
    check each cost where they ran (`pearlplus_checks`) and the class with its requests and rounds (`pearlplus_class`)."""
    if doc.get("piece") == "pearlplus":
        shares = doc.get("shares") or {}
        phase = doc.get("execution_phase")
        at = (shares.get("readings") or {}).get("tensor") or {}
        un, rp = shares.get("unretained"), shares.get("replay")
        credit, published = doc.get("credit"), doc.get("coin")
        check = (doc.get("coins") or {}).get("check") or {}
        return {"verdict": doc.get("verdict"), "development": True, "credited": False, "reasons": doc.get("reasons"),
                "coin_source": doc.get("coin_source"), "session": check.get("session"),
                "coins": check.get("reason", "unchecked"),
                **{k: doc.get(k) for k in ("scheme", "load", "passes", "dropped", "served_tokens", "generate_s", "tie")},
                "credit": None if credit is None else {k: credit.get(k) for k in ("credited", "checked_macs", "epsilon")},
                "coin": None if published is None else {k: published.get(k) for k in ("art", "preserved", "error", "sha512")},
                "give_up": doc.get("give_up"),
                "shares_at_tensor_clock": {k: at.get(k) for k in ("declared_linears", "credited")},
                "unretained": None if un is None else un.get("error") or {
                    "generate_s": un.get("wall_s"),
                    "declared_linears_at_tensor_clock": ((un.get("readings") or {}).get("tensor") or {}).get("declared_linears")},
                "replay": None if rp is None else rp.get("error") or {
                    k: rp.get(k) for k in ("s", "cpus", "cores_to_keep_pace", "device_s", "node", "finished", "of")},
                "execution_phase": None if phase is None else {
                    "clock": "the auditor device's", **{k: phase.get(k) for k in ("start_utc", "end_utc", "seconds", "why")},
                    "prover": phase.get("prover")},
                "checks": doc.get("checks"), "class": doc.get("class")}
    check, credit = doc.get("check") or {}, doc.get("credit") or {}
    return {"verdict": doc.get("verdict"), "development": True, "credited": False, "reasons": doc.get("reasons"),
            "coin_source": doc.get("coin_source"), "session": (doc.get("session") or {}).get("session"),
            "tied": doc.get("tied"), "coins": check.get("reason", "unchecked"), "manifest": check.get("manifest", "unchecked"),
            "gate_passed": doc.get("gate_passed"), "credit": credit.get("window")}


def _gate_out(gate: str, *words: str) -> bytes:
    """The gate's answer to `words` through its command `gate`, the words one shell-quoted line as `commit_coins.ChallengerGate`
    sends them; RuntimeError naming what it said when it refuses."""
    p = subprocess.run([*shlex.split(gate), shlex.join(words)], capture_output=True, check=False)
    if p.returncode:
        said = (p.stdout + p.stderr).decode(errors="replace").strip()
        raise RuntimeError(f"the gate refused {words[0]} (exit {p.returncode}): {said[-400:]}")
    return p.stdout


def gate_ready(gate: str, lane: str, main: Path | None = None, root: str | None = None,
               registration: str | None = None) -> tuple[bool, str]:
    """Whether this run may reserve its windows' sessions at `gate` (the ``gate`` subcommand), and its ``GATE`` line; with
    `root`, the run root a session of this run opens on, also refused once an open in the ledger names it: the gate opens
    a root once, ever. The lane's open period refuses, unless it is the registering period whose ``registration`` is
    `registration` (the deployment's registration record's digest), which the epoch's sessions are to join."""
    from verity.protocols.proof.spot.verifier.challenger import challenger as CH
    from verity.protocols.accounting.compute.pouw.prover.serving.coins import window as PHW

    ledger = _gate_out(gate, "ledger").decode().splitlines()
    pins = [json.loads(x) for x in _gate_out(gate, "pins").decode().splitlines() if x.strip()]
    pin, period = (pins[-1] if pins else {}), PHW.open_period(ledger, lane)
    held = None if root is None else next((e["session"] for e in map(json.loads, filter(str.strip, ledger))
                                           if e["kind"] == "open" and CH.root_of(e) == root.lower()), None)
    on = None if main is None else pin.get("commit") in main.read_text().split()
    where = "" if on is None else f", {'on' if on else 'NOT on'} the verifier's first-parent line of main"
    rooted = "" if root is None else (f"; root {root[:16]}… " + (f"already has its session {held} (one session a root, ever)"
                                                                  if held else "has no session yet"))
    registering = period is not None and registration is not None and period.get("registration") == registration
    joins = " (the registering period, which the epoch's reserves join)" if registering else " (a reserve would join it)"
    stops = (period is not None and not registering) or held is not None
    line = (f"GATE {'refused' if stops else 'ready'}: lane {lane}'s open period "
            f"{period and period.get('period')}{joins if period else ''}{rooted}; the ledger's "
            f"{len(ledger)} entries; pinned {str(pin.get('commit'))[:12]} (main {str(pin.get('main'))[:12]}, "
            f"{pin.get('pinned_utc')}){where}")
    return not stops, line


def _gate_json(gate: str, argv: Sequence[str]) -> dict[str, Any]:
    """The gate's JSON answer to `argv`; RuntimeError naming what it said when it refuses."""
    raw = _gate_out(gate, *argv)
    said = raw.decode(errors="replace").strip()[-600:]
    try:
        answer = json.loads(raw)
    except ValueError as e:
        raise RuntimeError(f"the gate's answer to {argv[0]} is not JSON ({e}): {said}") from None
    if not isinstance(answer, dict) or "refused" in answer:
        raise RuntimeError(f"the gate refused {argv[0]}: {said}")
    return answer


def period_close(serve_dir: Path, gate: str) -> dict[str, Any]:
    """The period whose open `PERIOD` in `serve_dir` records, closed, its answer added to ``PERIOD``."""
    doc = json.loads((serve_dir / PERIOD).read_text())
    doc["close"] = _gate_json(gate, ["period-close", "--name", doc["name"]])
    (serve_dir / PERIOD).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def pearlplus_descriptor(scheme: str, cubin: Path | None) -> bytes:
    """The Pearl-C period's circuit descriptor (`PEARLPLUS_DESCRIPTOR`): Pearl+'s protocol, the served scheme and the
    SHA-512 of the cubin that served it, as canonical JSON. No Pearl+ circuit is registered yet, so this names what the
    period's one salt session serves; a pass's ``descriptor`` for credit's circuit clause is to be these bytes."""
    doc = {"schema": PEARLPLUS_DESCRIPTOR_SCHEMA, "protocol": PEARLPLUS_PROTOCOL, "scheme": scheme,
           "cubin_sha512": None if cubin is None else hashlib.sha512(cubin.read_bytes()).hexdigest()}
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


def pearlplus_period_open(piece_dir: Path, gate: str, name: str, scheme: str, cubin: Path | None = None,
                          record: Path | None = None) -> dict[str, Any]:
    """``pearlplus-period open``: the lane's Pearl-C period `name` opened at `gate` before Pearl+'s serve, of cap 1 (the
    serve's one salt-only session, `bench_arm._LiveSession` or a tied window's `serve_pearl.live`, which joins it) and
    circuit the SHA-512 of its descriptor, written to ``piece_dir/PEARLPLUS_DESCRIPTOR``: the served window's circuit
    record `record` byte for byte where it is given, else `pearlplus_descriptor`. That descriptor is each pass's
    ``descriptor`` for credit's circuit clause (`challenger.circuit_reason`). The open and what it committed go to
    ``piece_dir/PERIOD``, which `period_close` closes. The run's sample session is reserved after the close, so it joins
    no period (`bench_arm.LiveDraw`) and the close lists the salt session alone (`credit.pearl_c_period_credit`)."""
    raw = record.read_bytes() if record is not None else pearlplus_descriptor(scheme, cubin)
    piece_dir.mkdir(parents=True, exist_ok=True)
    (piece_dir / PEARLPLUS_DESCRIPTOR).write_bytes(raw)
    circuit = hashlib.sha512(raw).hexdigest()
    doc = {"name": name, "circuit": circuit, "cap": 1, "descriptor": PEARLPLUS_DESCRIPTOR,
           "circuit_record": None if record is None else str(record),
           "open": _gate_json(gate, ["period-open", "--name", name, "--cap", "1", "--circuit", circuit])}
    (piece_dir / PERIOD).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


#: the hold's refusals where the lane's registration is not this serve's to proceed on
REGISTERED_PERIOD_CLOSED = ("the lane's registering period is closed and no period of the lane is open, so the epoch's "
                            "sessions would join none: the auditor opens the lane's next period, with no registration")
LANE_REGISTERED_BEFORE = ("the lane was registered before, by a period now closed, to another record: a lane takes one "
                          "registration, ever, so this deployment registers on a fresh lane")
REGISTERED_OTHER_DIGEST = ("the lane's open registering period names another record's digest than this serve's: the "
                           "period-open was given another registration record")


def registration_hold(record: Path, lane: str | None, ledger: str | None, key: Path | None, once: bool = False,
                      say: Callable[[str], None] = print, gate_key: str | None = None, key_sha512: str | None = None,
                      every: float = 10.0) -> int:
    """``registration-hold``: with no `lane`, the development line, and 0. Under `lane`, the deployment's registration
    record's digest and the period-open that registers it, then the auditor's `ledger` until it carries the lane's
    registering period-open (`challenger.registration_head`: every entry's signature and the chain tested under the key
    record `key`, whose public key's SHA-512 must be `key_sha512` where given): 0 where that is the record's digest and
    the lane has a period open for the epoch to join; 2 at once where the ledger refuses (a second registration, a
    malformed entry, a signature, a key not `key_sha512`), or registers the lane to another record in a closed period
    (`LANE_REGISTERED_BEFORE`) or an open one (`REGISTERED_OTHER_DIGEST`), or to this one with no period of the lane
    open (`REGISTERED_PERIOD_CLOSED`); 1 where the ledger ends without one, or with `once` holds none as it stands. A
    path is read whole, then as it grows (``tail -F``); ``gate:USER@HOST`` through the gate's ``ledger`` verb with `gate_key`
    (`challenger_session.ledger_lines_at`), again every `every` seconds. The wait's limit is the caller's (`timeout`)."""
    from verity.commitments import docket_digest as DD
    from verity.protocols.proof.spot.verifier.challenger import challenger as CH, challenger_session as AC
    from verity.protocols.proof.spot.verifier.one_stage import docket as DK
    from verity.protocols.accounting.compute.pouw.prover.serving.coins import window as PHW

    if lane is None:
        say(f"REGISTRATION {DK.DEVELOPMENT_WHY}")
        return 0
    digest = DD.digest(json.loads(record.read_text()))
    try:
        k = CH.key_of_record(json.loads(key.read_text()))
    except (OSError, ValueError, CH.CoinError) as e:
        say(f"REGISTRATION-HOLD refused: the challenger's key record {key} is unreadable or malformed "
            f"({type(e).__name__}: {e})")
        return 2
    pinned = hashlib.sha512(k.public).hexdigest()
    if key_sha512 is not None and pinned != key_sha512:
        say(f"REGISTRATION-HOLD refused: the challenger's key record {key} holds key {pinned[:16]}..., not the "
            f"pinned {key_sha512[:16]}...")
        return 2
    say(f"REGISTRATION-HOLD digest {digest} lane {lane} key {pinned}: before the epoch's first reserve, the auditor opens "
        f"the lane's period with it: vy-challenger-gate period-open --name NAME --lane {lane} --cap CAP --circuit CIRCUIT "
        f"--registration {digest}; held until {ledger} carries it")

    def judged(lines: list[str]) -> int | None:
        try:
            head = CH.registration_head(lines, lane, k, AC.lms_verify)
        except CH.CoinError as e:
            if CH.NO_REGISTRATION in str(e):
                return None
            say(f"REGISTRATION-HOLD refused: {e}")
            return 2
        es = [json.loads(x) for x in lines]
        period = next(e["body"]["period"] for e in es if e["kind"] == "period-open"
                      and e["body"].get("lane") == lane and "registration" in e["body"])
        closed = any(e["kind"] == "period-close" and e["body"].get("period") == period for e in es)
        if head != digest:
            why = LANE_REGISTERED_BEFORE if closed else REGISTERED_OTHER_DIGEST
            say(f"REGISTRATION-HOLD refused: {why} (lane {lane}'s period {period} registers {head[:16]}..., this "
                f"serve's record is {digest[:16]}...)")
            return 2
        if PHW.open_period(lines, lane) is None:
            say(f"REGISTRATION-HOLD refused: {REGISTERED_PERIOD_CLOSED} (lane {lane}'s registering period {period})")
            return 2
        say(f"REGISTRATION-HELD digest {digest} lane {lane} key {pinned}: the ledger's line {len(lines) - 1} signs it")
        return 0

    ended = (f"REGISTRATION-HOLD refused: {ledger} holds no registering period-open of lane {lane}: "
             f"{CH.NO_REGISTRATION}")
    if ledger.startswith("gate:"):
        said = None
        while True:
            try:
                rc = judged(AC.ledger_lines_at(ledger, gate_key))
            except CH.CoinError as e:
                rc = None
                if str(e) != said:
                    said = str(e)
                    say(f"REGISTRATION-HOLD reading {ledger}: {e}")
            if rc is not None:
                return rc
            if once:
                say(ended)
                return 1
            time.sleep(every)
    lines = Path(ledger).read_text().splitlines() if Path(ledger).exists() else []
    if lines and (rc := judged(lines)) is not None:
        return rc
    if once:
        say(ended)
        return 1
    follow = subprocess.Popen(["tail", "-F", "-n", f"+{len(lines) + 1}", ledger],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        for line in follow.stdout:
            lines.append(line.rstrip("\n"))
            if (rc := judged(lines)) is not None:
                return rc
        say(ended)
        return 1
    finally:
        follow.kill()
        follow.wait()


def period_reserves(lines: Sequence[str], lane: str, period: str) -> list[str] | None:
    """The sessions lane `lane` reserved (its prover ``for``, else ``by``) in the auditor's ledger `lines` while period
    `period` was open, in ledger order: since its open, and before its close where the ledger closes it; None where the
    ledger opens no such period."""
    got: list[str] | None = None
    for e in map(json.loads, lines):
        b = e.get("body") or {}
        if e.get("kind") == "period-open" and b.get("period") == period:
            got = []
        elif got is not None and e.get("kind") == "period-close" and b.get("period") == period:
            break
        elif got is not None and e.get("kind") == "reserve" and (b.get("for") or b.get("by")) == lane:
            got.append(str(e.get("session")))
    return got


def window_salt_sessions(pearl: Path) -> list[str]:
    """The salt sessions serve-openai's Pearl+ arm reserved, in window order: each ``pearl/w<k>/served.json``'s
    ``coins.reserved.session`` (`serve_pearl.live`'s reserve answer), where one names it. A window with no
    ``served.json`` names none; one whose ``served.json`` can't be read raises ValueError naming it."""
    got = []
    for d in sorted((p for p in pearl.glob("w*") if p.name[1:].isdigit()), key=lambda p: int(p.name[1:])):
        served = d / "served.json"
        if not served.is_file():
            continue
        try:
            sid = ((json.loads(served.read_text()).get("coins") or {}).get("reserved") or {}).get("session")
        except (OSError, ValueError, AttributeError) as e:
            raise ValueError(f"{d.name}/served.json: {type(e).__name__}: {e}") from e
        if sid:
            got.append(str(sid))
    return got


#: seconds between the tied close's reads of the ledger while a window's salt session has no ``end`` in it
PERIOD_POLL_S = 5


def unended(lines: Sequence[str], sessions: Sequence[str]) -> list[str]:
    """Those of `sessions` with no ``end`` entry in the auditor's ledger `lines`, in order."""
    ended = {e.get("session") for e in map(json.loads, lines) if e.get("kind") == "end"}
    return [s for s in sessions if s not in ended]


def pearlplus_period_close(piece_dir: Path, gate: str, lane: str | None = None, windows: Path | None = None,
                           wait_s: float = 0, sleep: Callable[[float], Any] = time.sleep) -> dict[str, Any]:
    """``pearlplus-period close``: the period `pearlplus_period_open` opened in `piece_dir` closed (`period_close`). With
    `lane` and `windows` (the serve dir's ``pearl/``, under PEARLPLUS_TIE) the gate's ledger is read first, and the PERIOD
    document records ``reserves``, ``{lane, since_open, salt_sessions, ok}`` (``error`` in place of ``since_open`` where
    the ledger or a window's ``served.json`` can't be read): ok when the sessions lane `lane` reserved since the open are
    the windows' salt sessions (`window_salt_sessions`), at least one, in order, as the gate's cap of 1 holds them. With
    `wait_s` the ledger is read again every `PERIOD_POLL_S` seconds, for at most `wait_s`, until it ends each of those
    sessions (`unended`): the gate closes a period only once each of its windows has ended, and a session's ``end``
    follows its client's close by the time its challenger takes to exit. It closes either way, since an open period
    refuses the lane's every other reserve; the tied piece samples only after one that closed with its reserves ok."""
    if lane is not None and windows is not None:
        doc = json.loads((piece_dir / PERIOD).read_text())
        salt: list[str] = []
        try:
            salt = window_salt_sessions(windows)
            lines = [x for x in _gate_out(gate, "ledger").decode().splitlines() if x.strip()]
            for _ in range(math.ceil(wait_s / PERIOD_POLL_S)):
                if not unended(lines, salt):
                    break
                sleep(PERIOD_POLL_S)
                lines = [x for x in _gate_out(gate, "ledger").decode().splitlines() if x.strip()]
            got = period_reserves(lines, lane, doc["name"])
            doc["reserves"] = {"lane": lane, "since_open": got, "salt_sessions": salt, "ok": bool(salt) and got == salt}
        except (RuntimeError, ValueError) as e:
            doc["reserves"] = {"lane": lane, "error": f"{type(e).__name__}: {e}"[:600], "salt_sessions": salt, "ok": False}
        (piece_dir / PERIOD).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return period_close(piece_dir, gate)


def period_reserves_said(doc: Mapping[str, Any]) -> str:
    """What `pearlplus_period_close` found of the period's reserves, as its line says it: nothing where it read none, and
    no session or count of them, which would count the windows."""
    r = doc.get("reserves")
    if r is None:
        return ""
    if r.get("ok"):
        return f", its reserves held (lane {r['lane']}'s since its open the windows' salt sessions alone)"
    why = (f"the ledger or a window's served.json can't be read: {r['error']}" if r.get("error") else
           f"the ledger opens no period {doc.get('name')}" if r.get("since_open") is None else
           "no window's served.json names a salt session" if not r.get("salt_sessions") else
           f"lane {r['lane']} reserved in it other sessions than the windows' salt sessions")
    return f", its reserves REFUSED ({why}; {PERIOD}'s reserves), so no draw follows it"


def pearlplus_period_rule(piece_dir: Path, gate: str, lane: str, record: Path) -> tuple[bool, str]:
    """The rule the verifier's sample keeps, read in the gate's ledger once its coin is flipped: the sample session of the coin
    record `record` (``draw.reserved.session``) is lane `lane`'s first reserve after the period in ``piece_dir/PERIOD``
    closed (`bench_arm.LiveDraw` held it to before asking its coin, and
    `pearlplus_live_sample` holds the record to). Whether it held, and its ``RULE`` line; kept as the PERIOD document's
    ``rule``."""
    from verity.protocols.accounting.compute.pouw.prover.serving.coins import window as PHW

    doc = json.loads((piece_dir / PERIOD).read_text())
    name = doc["name"]
    draw = ((json.loads(record.read_text()).get("draw") or {}).get("reserved") or {}).get("session")
    closed, after = PHW.after_close([x for x in _gate_out(gate, "ledger").decode().splitlines() if x.strip()], lane, name)
    first = None if after is None else after.get("session")
    held = closed and draw is not None and first == draw
    doc["rule"] = {"lane": lane, "closed": closed, "first_after": first, "draw": draw, "held": held}
    (piece_dir / PERIOD).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    why = (f"the draw session {draw} is the lane's first reserve after the close" if held else
           f"the ledger doesn't close period {name}" if not closed else
           "the verifier's coin record names no draw session" if draw is None else
           f"the lane's first reserve after the close is {first}, not the draw session {draw}")
    return held, (f"RULE no other reserve on lane {lane} between period {name}'s close and the draw: "
                  f"{'HELD' if held else 'BROKEN'} ({why})")


#: a check only the developer's piece runs, as a premise's ``why`` ends (`RAN_BY`)
PIECE = "run by the developer's piece; no auditor-side entry runs it"
#: what the onsite path adds to a check only the developer's piece runs: the command line's onsite check
#: (`verity_cli.served_table`) is where each accept the served node wrote becomes undecided
ONSITE_UNDECIDED = "the command line's onsite check (verity_cli served_table) holds each accept the served node wrote as undecided"
#: true of what the verifier accepts today (its main form): the sampled proofs' and the inference-only rows say both
MAIN_FORM = True
BLOCKS_ALONE = ("each sampled proof block is proved alone; that a block's inputs are the outputs of the blocks before it "
                "is assumed, not checked")
TIED_BY_CONSTRUCTION = ("the circuit the policy was checked on and the circuit the proofs are about are tied by "
                        "construction only")
#: the owners whose pieces grade their own rows' answers on the served node, no auditor-side entry running the check: a
#: row of theirs ends its ``answer_from`` with its placement's ``graded_by`` (`graded_by`), or `PIECE` where it has none
GRADED_ON_NODE = ("memory-accounting",)
_MAIN_FORM_SAYS = f"{BLOCKS_ALONE}; {TIED_BY_CONSTRUCTION}"
#: who ran the check behind what a row shows, by the record's product, then by row (ending its ``answer_from``)
#: what each product adds to a check only the developer's piece runs
TAIL = {"onsite": f"; {ONSITE_UNDECIDED}", "remote": ""}
RAN_BY: Mapping[str, Mapping[str, str]] = {
    product: {**({"integrity": _MAIN_FORM_SAYS, INFERENCE_ONLY_ROW: _MAIN_FORM_SAYS} if MAIN_FORM else {})}
    for product in TAIL}


def graded_by(row: Mapping[str, Any]) -> str | None:
    """Who graded ``row``'s answer, as its piece's placement says (``answer.profile.placement.graded_by``), or None."""
    profile = (row.get("answer") or {}).get("profile")
    place = profile.get("placement") if isinstance(profile, Mapping) else None
    said = place.get("graded_by") if isinstance(place, Mapping) else None
    return said if isinstance(said, str) and said else None


def declared_unchecked(protocol: Any, said: str = "") -> str:
    """What of its run no auditor-side entry checks, as the scheme declared under ``protocol`` says (its declaration's
    ``unchecked``, each sentence alone, `scheme.sentences`), less what ``said`` says already, or empty."""
    from verity.protocols.run import schemes

    declared = next((schemes.S.sentences(s) for n in schemes.DECLARATIONS
                     if (s := schemes.get(n)).protocol == protocol), ())
    return "; ".join(w for w in declared if w not in said)


def ran_by(key: str, row: Mapping[str, Any], product: str) -> dict[str, Any]:
    """Row ``key`` saying who ran its checks on ``product``'s path (`RAN_BY`): the row's own at the end of its
    ``answer_from``, and then what its
    protocol's declaration says no auditor-side entry checks (`declared_unchecked`)."""
    said, out = RAN_BY[product], dict(row)
    own = graded_by(row) or (PIECE if row.get("owner") in GRADED_ON_NODE else None)
    if "answer_from" in out and (key in said or own):
        out["answer_from"] = f"{out['answer_from']}; {said[key] if key in said else own + TAIL[product]}"
    if "answer_from" in out and (unchecked := declared_unchecked(row.get("protocol"))):
        out["answer_from"] = f"{out['answer_from']}; {unchecked}"
    return out


def _salt_live(served: Mapping[str, Any], out: Path, gate: str | None, key: str | None, main: Path | None,
               fetch: Callable[..., bytes] | None = None) -> dict[str, Any]:
    """The live salt session the served run's ``coins`` names, ``{key, key_sha512, ledger, pins, main, session}``: the
    challenger key whose public key's SHA-512 is `key` (`challenger.pinned_key`) and that pin, the auditor's ledger and the host's pins log as the
    gate `gate` answers now, main's first-parent line `main` (`remote.main_lines`) and the session's files from the
    gate's ``fetch``, each written to `out`. `fetch` replaces the gate's answers (tests)."""
    from verity.protocols.proof.spot.verifier.challenger import remote
    from verity.protocols.accounting.compute.pouw.prover.serving.coins import challenger as PHC

    sid = ((served.get("coins") or {}).get("session") or {}).get("session")
    if not (gate and key and main) and fetch is None:
        raise ValueError("no gate, key pin or main's first-parent line to check the session by (--gate, --key-sha512, "
                         "--main)")
    if not sid:
        raise ValueError("the served run names no live session")
    ask = fetch or (lambda *words: _gate_out(gate, *words))
    tgz, tree = out / "session.tgz", out / "session"
    tgz.write_bytes(ask("fetch", sid))
    with tarfile.open(tgz) as t:
        t.extractall(tree, filter="data")
    for name in ("ledger", "pins"):
        (out / f"{name}.jsonl").write_bytes(ask(name))
    d = tree / sid / "out"
    return {"key": PHC.pinned_key(d / remote.KEY, key).to_json(), "key_sha512": key,
            "ledger": remote.ledger_lines(out / "ledger.jsonl"), "pins": remote.ledger_lines(out / "pins.jsonl"),
            "main": remote.main_lines(main), "session": remote.session_files(d)}


def pearlplus_coins(served: Mapping[str, Any], out: Path, gate: str | None, key: str | None,
                    main: Path | None) -> dict[str, Any]:
    """Whether every pass with calls of Pearl+'s served run (``served``, `served_passes.py serve`'s document) has its
    epoch's salt from the live session its ``coins`` names (`served_step.coins_of`'s): the session's tree as the gate
    `gate`'s ``fetch`` gives it, and the auditor's ledger and the host's pins log as the gate answers now, written to
    `out`, with main's first-parent line `main` (`remote.main_lines`), each pass's manifest held by
    `core_check.live_salt` under the challenger key whose public key's SHA-512 is `key` (`challenger.pinned_key`).
    ``{session, closed, passes, reason}``: the passes checked, and None or the first refusal."""
    from verity.protocols.accounting.compute.pouw.verifier import core_check as CC

    coins = served.get("coins") or {}
    sid = (coins.get("session") or {}).get("session")
    got: dict[str, Any] = {"session": sid, "closed": coins.get("closed"), "passes": 0, "reason": None}
    try:
        live = _salt_live(served, out, gate, key, main)
        for p in served.get("passes") or ():
            if not p.get("units"):
                continue
            _salt, refused = CC.live_salt(live["key"], live["ledger"], live["pins"], live["main"], live["session"],
                                          json.loads(Path(p["manifest"]).read_text()), CC.lms_verify)
            got["passes"] += 1
            if refused is not None:
                got["reason"] = f"pass {p.get('pass')}: {refused[0]}: {refused[1]}"[:600]
                break
        else:
            if not got["passes"]:
                got["reason"] = "the served run has no pass with calls to check"
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, tarfile.TarError) as e:
        got["reason"] = f"{type(e).__name__}: {e}"[:600]
    return got


def _draw_live(record: Path, out: Path, gate: str | None, key: str | None, main: Path | None,
               fetch: Callable[..., bytes] | None = None) -> dict[str, Any] | None:
    """The draw session of the verifier's challenger sample in its coin record `record`, ``{key, key_sha512, ledger,
    pins, main, session}``: the challenger key whose public key's SHA-512 is `key` (`challenger.pinned_key`) and that
    pin, the auditor's ledger and the host's pins log as the gate `gate` answers now, main's first-parent line `main` and
    the session's tree from the gate's ``fetch``, each written to `out` (the tree at ``out/session/<id>``, beside the
    salt session's). None for a node's sample, which has no session. `fetch` replaces the gate's answers (tests)."""
    from verity.protocols.accounting.compute.pouw.verifier import salt as PS
    from verity.protocols.proof.spot.verifier.challenger import remote
    from verity.protocols.accounting.compute.pouw.prover.serving.coins import challenger as PHC

    draw = json.loads(record.read_text()).get("draw") or {}
    if draw.get("from") != "challenger":
        return None
    sid = (draw.get("reserved") or {}).get("session")
    if not (gate and key and main) and fetch is None:
        raise ValueError("no gate, key pin or main's first-parent line to check the draw session by (--gate, "
                         "--key-sha512, --main)")
    if not (isinstance(sid, str) and PS.SESSION.fullmatch(sid)):
        raise ValueError(f"the record's draw names no gate session ({sid!r})")
    ask = fetch or (lambda *words: _gate_out(gate, *words))
    tgz, tree = out / f"draw-{sid}.tgz", out / "session"
    tgz.write_bytes(ask("fetch", sid))
    with tarfile.open(tgz) as t:
        t.extractall(tree, filter="data")
    for name in ("ledger", "pins"):
        (out / f"{name}.jsonl").write_bytes(ask(name))
    d = tree / sid / "out"
    return {"key": PHC.pinned_key(d / remote.KEY, key).to_json(), "key_sha512": key,
            "ledger": remote.ledger_lines(out / "ledger.jsonl"), "pins": remote.ledger_lines(out / "pins.jsonl"),
            "main": remote.main_lines(main), "session": remote.session_files(d)}


def pearlplus_draw(record: Path, out: Path, gate: str | None, key: str | None, main: Path | None,
                   fetch: Callable[..., bytes] | None = None) -> dict[str, Any]:
    """Whether the verifier's sample in its coin record `record` (`verify_run`'s "The coin record") is its sample session's:
    for a challenger sample, its session as `_draw_live` fetches it, the ledger opening that session alone on the run's
    commitment C under the challenger key `key` pins, run from a commit on main's line `main`
    (`challenger.trusted_session_of`), withheld (`challenger.withheld_on`), its signed commitment the record's and of a
    sample session's layout (`window.draw_epoch_reason`), its log showing C as the one root its coin followed, and the
    record's opening its coin ``pouw/draw`` 0 after C (`challenger.live_reason`) with the record's source its coin.
    `fetch` replaces the gate's answers (tests). ``{from, session, reason}``: the sample's origin, its session and None
    or the first refusal; a node's sample has no session and no refusal here (the compose refuses it,
    `pearlplus_live_sample`)."""
    from verity.protocols.accounting.compute.pouw.verifier import window as Wd
    from verity.protocols.proof.spot.verifier.challenger import challenger as Ch
    from verity.protocols.proof.spot.verifier.challenger import challenger_session as CS

    got: dict[str, Any] = {"from": None, "session": None, "reason": None}
    try:
        rec = json.loads(record.read_text())
        draw = rec.get("draw") or {}
        got["from"] = draw.get("from")
        if got["from"] != "challenger":
            return got
        got["session"] = (draw.get("reserved") or {}).get("session")
        live = _draw_live(record, out, gate, key, main, fetch)
        c = rec["commitment"]
        trusted, log = Ch.trusted_session_of(live["ledger"], {"root": c}, Ch.ChallengerKey.from_json(live["key"]),
                                             live["pins"], live["main"], live["session"], CS.lms_verify)
        received = Ch.by_family(trusted, log)
        opening = Ch.LiveOpening.from_json(draw.get("opening"))
        if not Ch.withheld_on(live["ledger"], c):
            raise ValueError(f"the run's draw session: {Ch.WITHHELD}")
        if (why := Wd.draw_epoch_reason(trusted)) is not None:
            raise ValueError(f"the draw session's signed coins: {why}")
        if trusted.to_json() != draw.get("commitment"):
            raise ValueError("the record's draw commitment is not the one the draw session's challenger signed")
        if tuple(received.get(Wd.DRAW, ())) != (bytes.fromhex(c),):
            raise ValueError("the draw session's log does not show the run's commitment as the one root its draw coin "
                             "followed")
        if (why := Ch.live_reason(trusted, opening, Wd.DRAW, 0, bytes.fromhex(c))) is not None:
            raise ValueError(f"the record's draw coin: {why}")
        if opening.coin.hex() != rec.get("source"):
            raise ValueError("the record's source is not its draw session's coin")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError, tarfile.TarError, Ch.CoinError) as e:
        got["reason"] = f"{type(e).__name__}: {e}"[:600]
    return got


#: The draw session `pouw-pearlplus --draw-live` holds a challenger sample to (`pearlplus_draw_live`), in the piece's dir.
PEARLPLUS_DRAW_LIVE = "pearlplus-draw-live.json"


def pearlplus_draw_live(record: Path, out: Path, gate: str | None, key: str | None, main: Path | None,
                        fetch: Callable[..., bytes] | None = None) -> Path | None:
    """The draw session of the verifier's challenger sample in its coin record `record`, as `pouw-pearlplus --draw-live`
    reads it (`Coins.liveDrawReason`): `_draw_live`'s, the files `pearlplus_draw` checks, written to
    ``out/PEARLPLUS_DRAW_LIVE``. None for a node's sample, which has no session; a fetch that fails raises, so the Lean
    step is given no session and its ``--of-record`` refuses."""
    live = _draw_live(record, out, gate, key, main, fetch)
    if live is None:
        return None
    (path := out / PEARLPLUS_DRAW_LIVE).write_text(json.dumps(live) + "\n")
    return path


#: The salt session `pouw-pearlplus --salt-live` holds each live pass's salt to (`pearlplus_salt_live`), in the piece's dir.
PEARLPLUS_SALT_LIVE = "pearlplus-salt-live.json"


def pearlplus_salt_live(served: Mapping[str, Any], out: Path, gate: str | None, key: str | None, main: Path | None,
                        fetch: Callable[..., bytes] | None = None) -> Path | None:
    """The live salt session of Pearl+'s served run (``served``, `served_passes.py serve`'s document), as
    `pouw-pearlplus --salt-live` reads it (`Coins.liveSaltReason`): `_salt_live`'s ``{key, key_sha512, ledger, pins,
    main, session}``, the files `pearlplus_coins` fetches, written to ``out/PEARLPLUS_SALT_LIVE``. None for a run on an
    in-process challenger's coins, which has no session; a fetch that fails raises, so the Lean step is given no session
    and its ``--of-record`` refuses a live run."""
    if not (served.get("coins") or {}).get("session"):
        return None
    live = _salt_live(served, out, gate, key, main, fetch)
    (path := out / PEARLPLUS_SALT_LIVE).write_text(json.dumps(live) + "\n")
    return path


def custody_of(copy: Path, run: str | None, run_dir: Path | None) -> dict[str, str] | None:
    """Where the run's custody keeps `copy`: ``{attempt, path}``, the `research run` Attempt `run` (a valid run id) and
    `copy`'s path in it, when `copy` is under the run's directory `run_dir`'s ``out/`` (what an Attempt carries); else
    None."""
    if not (run and RUN_ID.fullmatch(run) and run_dir is not None):
        return None
    path, base = copy.resolve(), Path(run_dir).resolve()
    if not path.is_relative_to(base / "out"):
        return None
    return {"attempt": run, "path": path.relative_to(base).as_posix()}


def coin(record: Path, out: Path, run: str | None, research: Sequence[str] | None = None,
         put: Callable[..., Any] = subprocess.run, run_dir: Path | None = None) -> dict[str, Any]:
    """The verifier's coin record `record` (`served_passes.py verify`'s ``DIR/coin.json``, written before any replay)
    published the moment it is sampled: copied to ``out/coin.json``, then put in the evidence store as `COIN_KIND` through
    `research` (default the command the shared setting ``RESEARCH`` names, else this Python's ``-m research``), its meta
    the run `run` and the run
    commitment the coin was flipped on. The put asks ``--preserve`` first; where that fails (a job's node holds no store
    write key) the record goes in the node's own store, under the same content-addressed id, which a put of the run's
    custody copy of ``out/coin.json`` under the same kind and meta preserves. A piece publishes one coin: a record other
    than the one ``out/coin.json`` already holds is refused (ValueError), as is one that is not the verifier's. Writes
    and returns ``out/coin-put.json``: ``{sha512, commitment, source, draw, run, art, preserved, error, custody}``,
    ``draw`` the record's ``draw.from``, ``art`` the id (None where no put answered one), ``error`` what the preserving
    put said when it didn't preserve, and ``custody`` the run's Attempt that carries ``out/coin.json`` off the node
    (`custody_of` under the run's directory `run_dir`), the fallback where no put preserved it (vy-dc-h200's workload
    shells hold no store key)."""
    raw = record.read_bytes()
    rec = json.loads(raw)
    if not (isinstance(rec, dict) and rec.get("schema") == COIN_KIND and isinstance(rec.get("commitment"), str)
            and isinstance(rec.get("source"), str)):
        raise ValueError(f"{record} is no {COIN_KIND} record with a commitment and a source")
    copy = out / COIN_RECORD
    if copy.is_file() and copy.read_bytes() != raw:
        raise ValueError(f"{copy} already holds another coin: a piece publishes the one coin its verifier drew")
    out.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(raw)
    if research is None:
        from verity_config.settings import SHARED

        found = shutil.which(SHARED.RESEARCH)
        research = [found] if found else [sys.executable, "-m", "research"]
    meta = json.dumps({"run": run, "commitment": rec["commitment"]}, sort_keys=True)

    def asked(*more: str) -> tuple[int | None, str | None, str]:
        try:
            p = put([*research, "data", "put", "--kind", COIN_KIND, "--meta", meta, "--file", str(copy), "--json", *more],
                    capture_output=True, text=True, check=False, timeout=PUT_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as e:
            return None, None, f"{type(e).__name__}: {e}"
        art = None
        with contextlib.suppress(ValueError, TypeError, KeyError):
            art = json.loads(p.stdout)["id"]
        said = (p.stderr or p.stdout or "").strip().splitlines()
        return p.returncode, art if isinstance(art, str) else None, said[-1] if said else f"exit {p.returncode}"

    rc, art, said = asked("--preserve")
    preserved = rc == 0 and art is not None
    if not preserved:
        _, art, _ = asked()
    got = {"sha512": hashlib.sha512(raw).hexdigest(), "commitment": rec["commitment"], "source": rec["source"],
           "draw": (rec.get("draw") or {}).get("from"), "run": run, "art": art, "preserved": preserved,
           "error": None if preserved else f"research data put --preserve: {said}"[:400],
           "custody": custody_of(copy, run, run_dir)}
    (out / COIN_PUT).write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
    return got


def pearlplus_coin(path: Path, answer: Mapping[str, Any]) -> dict[str, Any]:
    """The verifier's coin as `coin` published it (``path``, its ``coin-put.json``, beside its copy of the record) held to
    Pearl+'s verdict `answer` (its core answer): an accept's run commitment and sample source must be the published
    record's, so the verdict is over the coin published at its sample and not one sampled after it. ``{art, preserved,
    sha512, error, reason}``: the store's id, whether the put preserved it (and what it said if not), the record's
    SHA-512, and None or why the verdict is not held to the published coin; an id the node's store holds but didn't
    preserve is no reason."""
    got: dict[str, Any] = {"art": None, "preserved": None, "sha512": None, "error": None, "reason": None}
    try:
        put = json.loads(path.read_text())
        raw = (path.parent / COIN_RECORD).read_bytes()
        got.update({k: put.get(k) for k in ("art", "preserved", "sha512", "error")})
        rec = json.loads(raw)
        if hashlib.sha512(raw).hexdigest() != put.get("sha512"):
            got["reason"] = "the coin record beside the put is not the one it published"
        elif answer.get("verdict") == "accept" and (answer.get("commitment"), (answer.get("profile") or {}).get("source")) \
                != (rec.get("commitment"), rec.get("source")):
            got["reason"] = ("the verdict's run commitment or draw source is not the published coin record's: the "
                             "verdict's coin is not the one published at its draw")
    except (OSError, ValueError, AttributeError) as e:
        got["reason"] = f"the verifier's coin was not published: {type(e).__name__}: {e}"[:600]
    return got


def give_up(volunteered: Mapping[str, Any]) -> dict[str, Any]:
    """The prover's give-up before the coin (``served_passes.py volunteer``'s document): whose words' counts it took
    (``counts``: ``device``, the card's from the debit cubin ``cubin_sha256``, every ``cross_check``-th tile and each with
    a word the card marks non-finite (``bad_tiles``) replayed on the CPU too, ``replayed`` in all; ``cpu``, every tile
    replayed), the A and B rows it gave up, the tiles over a cap before it and left over after, the passes it finished of
    ``of``, and whether it left none over a cap. The piece's document and its line carry it on the node; `publish` holds
    it whole (`OWN_DOCS`)."""
    dev = volunteered.get("device")
    return {"counts": "cpu" if dev is None else "device",
            **{k: (dev or {}).get(k) for k in ("cubin_sha256", "cross_check", "replayed", "bad_tiles")},
            "a_rows": volunteered.get("voluntary_a"), "b_rows": volunteered.get("voluntary_b"),
            **{k: volunteered.get(k) for k in ("tiles", "over", "left", "of", "ok")},
            "finished": len(volunteered.get("passes") or ())}


def pearlplus_why_tied(reserved: Mapping[str, Any] | None, challenger: str = UNNAMED_CHALLENGER) -> str:
    """Why Pearl+'s piece under PEARLPLUS_TIE credits nothing by itself, its epoch's salt coin a live session's on
    `challenger` (`challenger_of`): the Pearl-C period that session joined, as the gate's reserve answer ``reserved``
    (the window's ``coins.reserved``) names it, or that it joined none."""
    r = reserved or {}
    joined = (f"reserved under lane {r.get('by')}'s key into its Pearl-C period {r['period']}" if r.get("period") else
              "in no period, its reserve having joined none, so the record can't take it")
    return f"{_PEARLPLUS_TIED}; its epoch's salt coin is a live session's on {challenger}, {joined}"


#: whose clock `execution_phase` reads, and where
PHASE_CLOCK = ("the auditor device's, to the second, in its signed ledger's entries as the gate answered them "
               "(ledger.jsonl beside this document); none of these times is the served node's")
#: whose clock the serve's own wall is on, which a piece holds beside `execution_phase`'s times
PROVER_CLOCK = ("the served node's, the prover's: the serve's own wall for its timed call (generate_s), which no auditor "
                "device measured")
#: `served_passes.py lean`'s check document beside the piece's Lean verdict
PEARLPLUS_LEAN_CHECK = "pearlplus-lean-check.json"
#: where Pearl+'s two checks ran, which a piece says beside what each cost
CHECKS_WHERE = ("on the served node, at the serve, over the passes it retained: the node each cost names is the served "
                "node, not the node the audit runs on, and no auditor device runs either check again, since their case is "
                "witness")
#: what a tied piece's ``class`` says its two derived counts are
CLASS_LOAD = ("the class's fixed load, which Pearl+'s check holds the passes to (verify --class): a request a slot "
              "(n_requests) and a round an output (n_rounds), from the class alone; no request's own length is here")


def execution_phase(ledger: Path, session: str | None, period: str | None) -> dict[str, Any]:
    """The execution phase as the challenger saw it, read from the auditor's ledger at `ledger` (`pearlplus_coins`' copy,
    which it held to the challenger's key): the times of the salt session `session`'s ``reserve`` (taken as its window
    opened), ``open`` and ``end`` entries, and of the Pearl-C period `period`'s open and close, each None where the ledger
    has no such entry. ``start_utc`` and ``end_utc`` are the session's reserve and end and ``seconds`` the span between
    them, None without both; ``why`` says what the ledger lacks. Nothing in it is the served node's clock."""
    got: dict[str, Any] = {"clock": PHASE_CLOCK, "start_utc": None, "end_utc": None, "seconds": None,
                           "session": {"reserved_utc": None, "opened_utc": None, "ended_utc": None},
                           "period": {"opened_utc": None, "closed_utc": None}, "why": None}
    at = {"reserve": ("session", "reserved_utc", "started_utc"), "open": ("session", "opened_utc", "started_utc"),
          "end": ("session", "ended_utc", "ended_utc"), "period-open": ("period", "opened_utc", "opened_utc"),
          "period-close": ("period", "closed_utc", "closed_utc")}
    try:
        for line in ledger.read_text().splitlines():
            e = json.loads(line) if line.strip() else {}
            body = e.get("body") if isinstance(e.get("body"), Mapping) else {}
            part, name, field = at.get(e.get("kind"), (None, None, None))
            named = e.get("session") == session if part == "session" else body.get("period") == period
            if part is not None and named and (session if part == "session" else period) and isinstance(body.get(field), str):
                got[part][name] = body[field]
        got.update(start_utc=got["session"]["reserved_utc"], end_utc=got["session"]["ended_utc"])
        if got["start_utc"] and got["end_utc"]:
            a, b = (datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ") for x in (got["start_utc"], got["end_utc"]))
            got["seconds"] = int((b - a).total_seconds())
        lacks = [f"{part}'s {name}" for part in ("session", "period") for name, v in got[part].items() if v is None]
        got["why"] = f"the ledger has no entry for the {', '.join(lacks)}" if lacks else None
    except (OSError, ValueError, AttributeError) as e:
        got["why"] = f"the ledger can't be read: {type(e).__name__}: {e}"[:400]
    return got


def pearlplus_checks(verified: Mapping[str, Any] | None, lean_check: Path) -> dict[str, Any]:
    """What each of Pearl+'s two checks cost where it ran (`served_passes.check_cost`: wall seconds and the node): the
    Python check's from ``verify``'s document `verified`, the Lean check's from ``lean``'s check document at `lean_check`;
    None for a check that recorded none (it didn't run). ``where`` says in words which node that is (`CHECKS_WHERE`)."""
    lean = _json(lean_check)
    return {"where": CHECKS_WHERE, "python": (verified or {}).get("check_cost"),
            "lean": lean.get("check_cost") if isinstance(lean, Mapping) else None}


def pearlplus_class(verified: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The window's class as Pearl+'s check held the passes to it (``verify --class``'s SLOTS,PROMPT,OUTPUTS, its
    document's ``class``), with the two counts its fixed load fixes (`CLASS_LOAD`): the requests, a slot each, and the
    rounds, an output each. Each is the class's, public structure, written from the class alone and from nothing the
    serve measured; None where the check held the passes to no class."""
    cls = (verified or {}).get("class")
    if not (isinstance(cls, (list, tuple)) and len(cls) == 3 and all(type(v) is int and v >= 1 for v in cls)):
        return None
    slots, prompt, outputs = cls
    return {"slots": slots, "prompt_tokens": prompt, "outputs": outputs, "n_requests": slots, "n_rounds": outputs,
            "of": CLASS_LOAD}


def pearlplus(out: Path, served: Mapping[str, Any] | None, verified: Mapping[str, Any] | None,
              unretained: Mapping[str, Any] | None = None, volunteered: Mapping[str, Any] | None = None, *,
              gate: str | None = None, key: str | None = None, main: Path | None = None,
              coin: Path | None = None, tie: Path | None = None) -> dict[str, Any]:
    """PoUW's development piece under POUW=pearlplus (``served_pearlplus.sh``), its document written to
    ``out/development.json``: Pearl+'s own served run (``served``, `served_passes.py serve`'s document), the Python
    verify over every pass it retained (``verified``, its ``verify``'s; None where either wrote none), which drew the
    verifier's coin and whose verdict decides nothing, and Pearl+'s verdict, the Lean's (``out/pearlplus-lean.json``,
    `PEARLPLUS_LEAN`, `served_passes.py lean`'s answer). Where ``served`` or ``verified`` is
    None there is no verdict, so it writes no document: it refuses (`PEARLPLUS_NO_VERDICT`), its sentence in
    ``out/refused.txt`` (`REFUSED`), and raises `PearlplusRefused`. Its verdict is
    accept when the run served within its caps across at least two pass boundaries, the Lean verdict accepts (none written
    rejects, `PEARLPLUS_NO_LEAN`), and its
    epoch's salt coin is its live session's (`pearlplus_coins` through the gate `gate`, the key pin `key` and main's line
    `main`; `pearlplus_coins_from`, naming the challenger as `challenger_of`); with an in-process challenger's coins (`PEARLPLUS_COINS_IN_PROCESS`) a run where nothing
    failed is `UNCLAIMED`, and a run served under the gate whose coins are not live rejects. It carries the Lean check's
    line, the Lean verdict's credit at the scheme's named assumption (``credit``, its ``work``) and the rounds the prover dropped from
    its retention before the coin (``dropped``, which credit nothing and stay declared), its shares are
    `served_shares.pearlplus_shares`'s wherever the run served within its caps, whatever the verdict (a rejected or
    missing one credits 0), and it credits nothing (`pearlplus_why`, or `PEARLPLUS_WHY_IN_PROCESS`). ``unretained``, the
    same load served with no pass retained (``serve --no-retain``), adds the shares' ``unretained``
    (`served_shares.pearlplus_unretained`), or why there are none; it is a measurement and never decides the verdict.
    ``volunteered``, the prover's give-up before the coin (``served_passes.py volunteer``'s document), is ``give_up``
    (`give_up`: whose words' counts it took, the card's or the CPU's, its cross-check and the rows it gave up), and adds
    the shares' ``replay`` (`served_shares.replay_term`: its CPU seconds on its processes, the card's beside them, as the
    cores it takes to keep pace with the timed call's wall, beside the node it ran on and how many of the passes it
    finished), a measurement too.
    ``coin``, the verifier's coin as `coin` published it (its ``coin-put.json``), is ``coin`` (`pearlplus_coin`): its
    store id or why it has none, and an accept whose run commitment or sample source isn't the published record's
    rejects; the record's sample is ``coins.draw`` (`pearlplus_draw`), and a challenger sample its session doesn't hold
    rejects (a node's says so in ``coins.draws``, `PEARLPLUS_NODE_SAMPLES`). The piece names its load (``load``: the requests, their prompt and output tokens, and the tokens a step).
    ``tie``, under PEARLPLUS_TIE, is the window record W the window's passes are held to (``served_passes.py retain
    --window-record``): ``served`` is then serve-openai's Pearl+ arm's document for the window (``pearl/w<k>/served.json``,
    its ``error`` why it isn't ok), its epoch's coins are `pearlplus_epoch_tied`'s (or `PEARLPLUS_EPOCH_TIED_IN_PROCESS`'s),
    its ``load`` is the window's steps, ``tie`` names W's `registration.digest` as ``record`` (None, and ``why``, where W
    can't be read; the record holds the verdict to W, `pearlplus_of_record`), and why it credits nothing by itself names
    the Pearl-C period the epoch's live salt session joined (`pearlplus_why_tied`, or `PEARLPLUS_WHY_TIED_IN_PROCESS`).
    ``execution_phase``, on a live session's coins, is the execution phase as the challenger saw it (`execution_phase`, of
    the ledger `pearlplus_coins` fetched), with the serve's own wall beside it as the prover's (its ``prover``:
    `PROVER_CLOCK` and ``generate_s``), and ``checks`` what Pearl+'s Python check and Lean check each cost on the node
    this piece ran them on (`pearlplus_checks`); both are measurements and neither decides the verdict. ``class`` is the
    class the check held the passes to with the requests and rounds its fixed load fixes (`pearlplus_class`), None
    without one."""
    import served_shares as SS
    from verity.commitments import docket_digest as DD

    out.mkdir(parents=True, exist_ok=True)
    (out / DEVELOPMENT).unlink(missing_ok=True)
    (out / REFUSED).unlink(missing_ok=True)
    wrote_none = [*(["serve-openai's Pearl+ arm wrote no served.json for the window" if tie is not None
                     else "served_passes.py serve wrote no document"] if served is None else []),
                  *(["served_passes.py verify wrote no document"] if verified is None else [])]
    if wrote_none:
        said = (f"{PEARLPLUS_NO_VERDICT}: {' and '.join(wrote_none)}, so Pearl+ has no verdict and the piece writes "
                "no answer")
        (out / REFUSED).write_text(said + "\n")
        raise PearlplusRefused(said)
    reasons = []
    source = (served.get("coins") or {}).get("source", "in-process")
    live = source == LIVE_SOURCE
    if not served.get("ok"):
        reasons.append(f"the window's Pearl+ epoch isn't one round a Commit step: {served.get('error')}"[:800]
                       if tie is not None else
                       f"the served run crossed fewer than two pass boundaries or ran over its caps (passes over their "
                       f"caps: {served.get('over_caps')})")
    try:
        lean = _json(out / PEARLPLUS_LEAN)
    except ValueError as e:
        lean = f"{type(e).__name__}: {e}"
    if not isinstance(lean, dict):
        reasons.append(f"{PEARLPLUS_NO_LEAN} ({PEARLPLUS_LEAN}: {lean or 'none written'})"[:800])
        lean = {}
    elif lean.get("verdict") != "accept":
        reasons.append(f"Pearl+'s verdict rejects the run: {lean.get('reason')}"[:800])
    answer = {"verdict": lean.get("verdict"), "commitment": lean.get("commitment"), "profile": {"source": lean.get("source")}}
    challenger = challenger_of(served)
    coins: dict[str, Any] = dict(pearlplus_coins_from(challenger) if live else PEARLPLUS_COINS_IN_PROCESS)
    if tie is not None:
        coins["epoch"] = pearlplus_epoch_tied(challenger) if live else PEARLPLUS_EPOCH_TIED_IN_PROCESS
    if live:
        coins["check"] = pearlplus_coins(served, out, gate, key, main)
        if coins["check"]["reason"] is not None:
            reasons.append(f"coins: {coins['check']['reason']}")
    elif gate:
        reasons.append(f"coins: {source}, not a live session on the gate, so nothing checks them")
    if coin is not None and (coin.parent / COIN_RECORD).is_file():
        coins["draw"] = pearlplus_draw(coin.parent / COIN_RECORD, out, gate, key, main)
        if coins["draw"]["from"] != "challenger":
            coins["draws"] = PEARLPLUS_NODE_SAMPLES
        elif coins["draw"]["reason"] is not None:
            reasons.append(f"draw: {coins['draw']['reason']}")
    if tie is None:
        why = pearlplus_why(challenger) if live else PEARLPLUS_WHY_IN_PROCESS
    else:
        why = (pearlplus_why_tied((served.get("coins") or {}).get("reserved"), challenger) if live
               else PEARLPLUS_WHY_TIED_IN_PROCESS)
    doc: dict[str, Any] = {"piece": "pearlplus", "credited": False, "why": why, "coin_source": source, "coins": coins,
                           "scheme": None, "arm": None, "passes": None, "load": None, "dropped": None, "line": None,
                           "credit": None, "coin": None, "give_up": None, "shares": None, "tie": None,
                           "execution_phase": None, "checks": pearlplus_checks(verified, out / PEARLPLUS_LEAN_CHECK),
                           "class": pearlplus_class(verified)}
    if live and (out / "ledger.jsonl").is_file():
        said = served.get("coins") or {}
        doc["execution_phase"] = execution_phase(out / "ledger.jsonl", (said.get("session") or {}).get("session"),
                                                 (said.get("reserved") or {}).get("period"))
        doc["execution_phase"]["prover"] = {"clock": PROVER_CLOCK, "generate_s": served.get("generate_s")}
    arm = served.get("arm") or served.get("entry")
    doc.update(arm=arm, scheme=(arm or "").partition(":")[2] or None,
               passes=sum(1 for p in served.get("passes") or () if p.get("units")),
               load={"window_steps": served.get("commit_steps")} if tie is not None else
               {**{k: served.get(k) for k in ("requests", "prompt_tokens", "output_tokens")},
                "max_num_batched_tokens": (served.get("engine") or {}).get("max_num_batched_tokens")},
               **{k: served.get(k) for k in ("spec", "run", "served_tokens", "generate_s", "prompt_tokens", "output_tokens")})
    if tie is not None:
        try:
            doc["tie"] = {"record": DD.digest(json.loads(tie.read_text()))}
        except (OSError, ValueError, AttributeError) as e:
            doc["tie"] = {"record": None, "why": f"W can't be read: {type(e).__name__}: {e}"[:400]}
    checked = _json(out / PEARLPLUS_LEAN_CHECK)
    doc.update(line=checked.get("line") if isinstance(checked, Mapping) else None, credit=lean.get("work"),
               dropped=verified.get("dropped"))
    if coin is not None:
        doc["coin"] = pearlplus_coin(coin, answer)
        if doc["coin"]["reason"] is not None and answer.get("verdict") == "accept":
            reasons.append(f"coin: {doc['coin']['reason']}")
    if volunteered is not None:
        doc["give_up"] = give_up(volunteered)
    if served.get("ok") and "generate_s" not in served:
        doc["shares"] = {"none": "the epoch is a Commit window's (serve_pearl's pearl/w<k>/served.json), which times no "
                                 "generate call of its own"}
    elif served.get("ok"):
        try:
            doc["shares"] = SS.pearlplus_shares(dict(served), {
                "answer": {"verdict": lean.get("verdict"), "public": {"work": lean.get("work")}},
                "timed": checked.get("timed") if isinstance(checked, Mapping) else None,
                "delta": verified.get("delta"), "draws": verified.get("draws")})
        except (KeyError, TypeError, ValueError) as e:
            reasons.append(f"no shares: {type(e).__name__}: {e}"[:600])
        if doc["shares"] is not None and unretained is not None:
            try:
                doc["shares"]["unretained"] = SS.pearlplus_unretained(doc["shares"], dict(served), dict(unretained))
            except (KeyError, TypeError, ValueError) as e:
                doc["shares"]["unretained"] = {"error": f"{type(e).__name__}: {e}"[:600]}
        if doc["shares"] is not None and volunteered is not None:
            try:
                card = (volunteered.get("device") or {}).get("device_s")
                doc["shares"]["replay"] = {**SS.replay_term(volunteered["s"] - (card or 0), volunteered["jobs"],
                                                            doc["shares"]["wall_s"]),
                                           "device_s": card, "node": volunteered.get("node"),
                                           "finished": len(volunteered.get("passes") or ()), "of": volunteered.get("of")}
            except (KeyError, TypeError, ValueError) as e:
                doc["shares"]["replay"] = {"error": f"{type(e).__name__}: {e}"[:600]}
    doc.update(verdict="reject" if reasons else "accept" if live else UNCLAIMED, reasons=reasons)
    (out / DEVELOPMENT).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    return doc


def check_answer(a: Any) -> dict[str, Any]:
    """``a`` when it is an answer in the shapes `Answer.ofJson` reads (`pouw.core_check.answer_of`, the Python mirror of
    the core interface's decoder): an accept has no reason and both its public outputs and its profile, a refusal a
    reason and neither. Anything else is a ValueError, which the row reports as the subprotocol's error."""
    from verity.protocols.accounting.compute.pouw.verifier import core_check as CC

    try:
        CC.answer_of(a)
    except CC.Refused as e:
        raise ValueError(f"not a core answer: {e}") from None
    return dict(a)


def epsilon(a: Mapping[str, Any]) -> str | None:
    """A sampled protocol's bound on incorrect proof blocks (``public.bound``) over its checked level's size (`Core.Integrity`), or
    None for an answer that has no such bound."""
    public, profile = a.get("public"), a.get("profile")
    levels = profile.get("levels") if isinstance(profile, Mapping) else None
    if not isinstance(public, Mapping) or not isinstance(levels, list) or not levels or not isinstance(levels[-1], Mapping):
        return None
    bound, size = public.get("bound"), levels[-1].get("size")
    return str(Fraction(bound, size)) if isinstance(bound, int) and isinstance(size, int) and size > 0 else None


def row_of(owner: str, a: Any, out: Path, run: str | None = None) -> dict[str, Any]:
    """A subprotocol's report row from its answer (checked to be a core answer first); ``run`` is the recording run. A
    one-stage accept whose public ``complete`` is not true is `NOT_COMPLETE`, its reason naming the entries its audit
    record under ``out`` left without a verdict."""
    from verity.commitments.domains import ONE_STAGE_PROTOCOL
    from verity.protocols.proof.spot.verifier.one_stage.verdict_record import NOT_COMPLETE

    a = check_answer(a)
    runs = json.loads((out / "runs.json").read_text()) if (out / "runs.json").exists() else []
    accepted = a["verdict"] == "accept"
    verdict, reason = a["verdict"], a["reason"]
    if accepted and a["protocol"] == ONE_STAGE_PROTOCOL and a["public"].get("complete") is not True:
        unread = None
        try:
            rec = next((_json(f) for f in (out / "window" / "audit.json", out / "audit.json") if f.is_file()), None)
        except ValueError as e:
            rec, unread = None, f"its public `complete` is not true, and its audit record does not parse: {e}"[:600]
        verdicts = rec.get("verdicts") if isinstance(rec, dict) else None
        null = sorted(k for k, v in verdicts.items() if isinstance(v, dict) and v.get("accepted") is None) \
            if isinstance(verdicts, dict) else []
        verdict, reason = NOT_COMPLETE, "its accept is not complete: " + (
            unread or (f"its audit record has no verdict of {', '.join(null)}" if null else "its public `complete` is not true"))
    return {"protocol": a["protocol"], "owner": owner, "verdict": verdict, "reason": reason,
            "commitment": a["commitment"], "quantity": a["public"] if accepted else None,
            "delta": (a["profile"] or {}).get("delta") if isinstance(a["profile"], dict) else None,
            "epsilon": epsilon(a) if accepted else None, "runs": [*([run] if run else []), *runs], "answer": a}


def spot_row(audit_dir: Path, run: str | None = None) -> dict[str, Any]:
    """SPOT's row from the records audit kept under ``audit_dir`` (module docstring)."""
    from verity.protocols.proof.spot.verifier.one_stage import served_integrity as SI

    reg = audit_dir / WINDOW_REGISTRATION
    try:
        a = SI.answer(audit_dir, json.loads(reg.read_text()) if reg.exists() else None)
        return row_of(INTEGRITY_OWNER, a, audit_dir, run)
    except (RuntimeError, ValueError) as e:
        return {"verdict": "error", "owner": INTEGRITY_OWNER, "reason": f"{type(e).__name__}: {e}"[:2000]}


def members_reached(members: Mapping[str, Any], answer: Mapping[str, Any]) -> dict[str, Any]:
    from verity_e2e_demo import block_record as BR

    sample, delta = answer["public"].get("drawn") or [], Fraction(answer["profile"]["delta"])
    by_member = []
    for name, m in sorted(members.items()):
        n = sum(hi - lo for lo, hi in m["units"])
        k = sum(lo <= u < hi for u in sample for lo, hi in m["units"])
        if k:
            by_member.append({"member": name, "of": n, BR.SAMPLED: k})
    return {"members": {"sampled": len(by_member), "registered": len(members)},
            "bounds": BR.member_bounds(by_member, delta), "by_member": by_member}


def window_record(summary: Any) -> tuple[str, str]:
    """The window road's verdict of record and its line, from the records audit's ``summary.json`` alone: ``accept`` only
    where the audit accepted and Lean's members check (`OneStageZk.checkMembersAt`, ``audit.members_lean``) ran and accepted,
    ``undecided`` where Lean did not run, else ``reject``; the line says what the verdict rests on, ``compiled`` as the
    summary states it. ValueError naming the field where the summary lacks one the verdict reads."""
    def field(of: Any, key: str, kind: type, where: str) -> Any:
        if not isinstance(of, Mapping) or not isinstance(of.get(key), kind):
            raise ValueError(f"the audit's summary has no {where}{key} of type {kind.__name__}")
        return of[key]

    ok, audit = field(summary, "ok", bool, ""), field(summary, "audit", dict, "")
    accepted, ml = field(audit, "accepted", bool, "audit."), field(audit, "members_lean", dict, "audit.")
    ran, lean_ok = field(ml, "ran", bool, "audit.members_lean."), field(ml, "ok", bool, "audit.members_lean.")
    compiled = summary.get("compiled", "not stated by the summary")
    rests = "" if compiled is True else "; the verdict does not rest on the circuit's compile check"
    tail = f"compiled: {compiled}{rests}; SPOT's verdict alone: no block record is composed on this road"
    if not ran:
        return "undecided", (f"WINDOW-RECORD undecided: Lean's members check did not run ({ml.get('reason')}), so no verdict of "
                             f"Lean's stands; the audit's own accepted={accepted}; {tail}")
    lean = ((ml.get("detail") or {}).get("lean") or {}) if isinstance(ml.get("detail"), Mapping) else {}
    said = (f"Lean's members check (OneStageZk.checkMembersAt): ok={lean_ok}, codes={ml.get('codes')}, verdict "
            f"{lean.get('verdict')}, output sha512 {lean.get('output_sha512') or 'not kept by this audit'}")
    if not (ok and accepted and lean_ok and lean.get("verdict") == "accept"):
        return "reject", (f"WINDOW-RECORD reject: {said}; the audit's ok={ok}, accepted={accepted}, refused="
                          f"{summary.get('refused')}; {tail}")
    wrong, sampled = field(audit, "wrong_units", dict, "audit."), field(summary, "draw", list, "")
    for k in ("bound", "delta"):
        if k not in wrong:
            raise ValueError(f"the audit's summary has no audit.wrong_units.{k}")
    return "accept", (f"WINDOW-RECORD accept: {said}; {len(sampled)} sampled blocks, at most {wrong['bound']} of its units wrong "
                      f"except with probability {wrong['delta']}; {tail}")


def sampled_exe(exe: str | os.PathLike | None) -> Path:
    """This tree's own SPOT Lean (`served_cases.SAMPLED_LEAN`'s build path): built here if ``exe`` is None, and ``exe``
    only if it resolves to that path; any other is refused (`SAMPLED_EXE_REFUSED`), so `LEAN_HERE` names this tree's."""
    import served_cases as SC

    if exe is None:
        return SC.build(*SC.SAMPLED_LEAN)
    built = SC.SAMPLED_LEAN[0] / ".lake" / "build" / "bin" / SC.SAMPLED_LEAN[1]
    if Path(exe).resolve() != built.resolve():
        raise SC.Refused(f"--sampled-exe {exe} {SAMPLED_EXE_REFUSED} ({built})")
    return built


#: why `sampled_exe` refuses an executable
SAMPLED_EXE_REFUSED = "isn't this tree's own SPOT Lean build path"


def lean_sampled_row(audit_dir: Path, scratch: Path, exe: str | os.PathLike | None = None,
                     run: str | None = None) -> dict[str, Any]:
    """SPOT's row from this host's own run of remote's SPOT Lean (`served_cases.sampled_case`: the
    records audit's v2 case, its blobs the proofs and public files, in the form ``one-stage``; ``exe`` `sampled_exe`'s)
    on files in a 0700 directory under ``scratch``, its answer tied to the window's docket (`LEAN_HERE`),
    whatever the audit's summary and record say; ``executable`` names the file run and its SHA-512. An audit that wrote
    no v2 case (a plain run's, or one a5 refused) leaves no proofs to recompute from: its row is `spot_row`'s, its ``answer_from`` saying so. So does a case whose check
    was not started (`served_cases.NotStarted`: its blob files sum past the limit, or the host makes no memory scope):
    the row is `spot_row`'s, ``answer_from`` and ``not_started`` naming why, the sum and the limit. A case the Lean
    doesn't make or answer, or an answer about another record, is the row's error. An accept of a case in the
    per-member form carries ``covers`` (`MEMBERS_COVER`), ``members`` (sampled of registered) and each sampled
    member's bound over its own proof blocks (``bounds``), and no ε over the workload."""
    import served_cases as SC

    if not (audit_dir / "window" / "case-v2.json").is_file():
        return {**spot_row(audit_dir, run),
                "answer_from": f"{AS_WRITTEN} ({WRITTEN['integrity']}): the records audit wrote no v2 case "
                               "(window/case-v2.json), so there are no proofs to recompute it from"}
    try:
        case = SC.sampled_case(audit_dir)
        built = sampled_exe(exe)
        ran = {"executable": {"name": built.name, "sha512": hashlib.sha512(built.read_bytes()).hexdigest()}}
        answer, _line = SC.lean_answer(built, case, scratch, SC.SAMPLED_CMD)
        SC.tie(answer, SC.SAMPLED, SC.record_of(case)[1])
        row = {**row_of(INTEGRITY_OWNER, answer, audit_dir, run), "answer_from": LEAN_HERE, **ran}
        proofs = case["params"].get("proofs")
        if isinstance(proofs, dict) and "members" in proofs and row["verdict"] == "accept":
            return {**row, "covers": MEMBERS_COVER, "epsilon": None, **members_reached(proofs["members"], answer)}
        return row
    except SC.NotStarted as e:
        return {**spot_row(audit_dir, run), "not_started": {"reason": str(e), **e.line},
                "answer_from": f"{AS_WRITTEN} ({WRITTEN['integrity']}): this host's Lean check of the audit's v2 case was "
                               f"not started: {e}"[:2000]}
    except (SC.Refused, OSError, ValueError) as e:
        return {"verdict": "error", "owner": INTEGRITY_OWNER, "reason": f"{type(e).__name__}: {e}"[:2000],
                "answer_from": LEAN_HERE}


def _json(path: Path) -> Any:
    """``path``'s JSON, None where there's no such file; one that isn't JSON raises ValueError."""
    return json.loads(path.read_text()) if path.exists() else None


def _summary(path: Path) -> dict[str, Any] | None:
    """A records audit's summary.json at ``path`` for the lines that say its coin trees: None where there's none, and one
    that isn't JSON as a summary whose ``coin_trees`` says so."""
    try:
        return _json(path)
    except ValueError:
        return {"coin_trees": f"unread: {path.name} is not JSON"}


def _ledger_period(lines: Sequence[str], session: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The reserve of `session` in the auditor's ledger `lines` and the ``period-open`` body of its prover's period it
    joined (`challenger.verify_periods`' rule: the prover's period open at the reserve), each None where there is none."""
    current: dict[str, dict[str, Any]] = {}
    for x in lines:
        e = json.loads(x)
        b = e.get("body") or {}
        if e.get("kind") == "period-open":
            current[b.get("lane")] = b
        elif e.get("kind") == "period-close":
            current.pop(b.get("lane"), None)
        elif e.get("kind") == "reserve" and e.get("session") == session:
            return e, current.get(b.get("for") or b.get("by"))
    return None, None


def pearlplus_live_sample(record: Mapping[str, Any], piece_dir: Path, key: str, lane: str, check: Mapping[str, Any],
                        lines: Sequence[str], period: str) -> str | None:
    """Why the verifier's sample source in its coin record `record` (the piece's `COIN_RECORD`, `verify_run`'s "The coin
    record") is not, as the piece's check says, a coin the auditor device's challenger under `key` opened on the run
    commitment C, or None. It must be a challenger sample (`PEARLPLUS_NODE_SAMPLE`) whose session the piece's check, run
    on the served node, held to the signed ledger and session (`check`, `pearlplus_draw`'s; no signature is verified
    here), the key file of that session's tree in the piece hashing to `key`; in the piece's ledger `lines`, the
    first reserve of lane `lane` after its Pearl-C period `period` closed, withheld with a sample session's layout alone,
    and the one session opened on C, so a prover who samples again under another session holds a sample no compose takes;
    and the record's opening its coin ``pouw/draw`` 0 after C (`challenger.live_reason`), the record's source that coin."""
    from verity.protocols.accounting.compute.pouw.verifier import window as Wd
    from verity.protocols.proof.spot.verifier.challenger import challenger as Ch
    from verity.protocols.proof.spot.verifier.challenger import remote
    from verity.protocols.accounting.compute.pouw.prover.serving.coins import challenger as PHC
    from verity.protocols.accounting.compute.pouw.prover.serving.coins import window as PHW

    if record.get("schema") != COIN_KIND:
        return f"the verifier's coin record is {record.get('schema')!r}, not {COIN_KIND}, which carries its draw"
    draw = record.get("draw") or {}
    if draw.get("from") != "challenger":
        return PEARLPLUS_NODE_SAMPLE
    sid = (draw.get("reserved") or {}).get("session")
    if check.get("reason") is not None or check.get("session") != sid:
        return f"the draw session's check refused it: {check.get('reason') or 'it checked another session'}"
    try:
        PHC.pinned_key(piece_dir / "session" / str(sid) / "out" / remote.KEY, key)
    except PHC.CoinError as e:
        return f"the draw session's challenger key is not the compose's pin: {e}"
    entries = [json.loads(x) for x in lines]
    closed, after = PHW.after_close(lines, lane, period)
    if not closed:
        return f"the piece's ledger doesn't close the Pearl-C period {period}, so no draw follows it"
    if after is None or after.get("session") != sid:
        return (f"the draw session {sid} is not lane {lane}'s first reserve after its Pearl-C period {period} closed "
                f"(that is {None if after is None else after.get('session')}): a draw is taken once")
    body = after.get("body") or {}
    if body.get("withheld") is not True or [tuple(f) for f in body.get("families") or ()] != list(Wd.DRAW_FAMILIES):
        return f"the draw session {sid} was not reserved withheld with a draw session's layout alone"
    c = record.get("commitment")
    opens = [e.get("session") for e in entries if e.get("kind") == "open"
             and ((e.get("body") or {}).get("window") or {}).get("root") == c]
    if opens != [sid]:
        return f"the ledger opens {opens} on the run's commitment, not the draw session {sid} alone"
    try:
        commitment = Ch.LiveCommitment.from_json(draw.get("commitment"))
        opening = Ch.LiveOpening.from_json(draw.get("opening"))
        why = Wd.draw_epoch_reason(commitment) or Ch.live_reason(commitment, opening, Wd.DRAW, 0, bytes.fromhex(str(c)))
    except (Ch.CoinError, ValueError, TypeError) as e:
        return f"the record's draw is malformed: {type(e).__name__}: {e}"
    if why is not None:
        return f"the record's draw coin: {why}"
    if opening.coin.hex() != record.get("source"):
        return "the record's source is not its draw session's coin"
    return None


def pearlplus_of_record(doc: Mapping[str, Any], piece_dir: Path, spot: Path | None, key: str | None,
                        lane: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Whether Pearl+'s accepted development piece `doc` (`pearlplus`'s, in `piece_dir`) is PoUW's verdict of record: its
    answer and None when it is, else None and the first condition it fails, in order:

    1. **Live coins under the pin.** The piece's check (`pearlplus_coins`, run on the served node, whose reason this
       reads) says the epoch's salt coin is the live session's, every pass held to it; here only the key file in the
       session's tree the piece handed is held to `key` (`challenger.pinned_key`), no signature being verified.
    2. **The lane's own reserve.** The piece's ledger reserves that session by lane `lane`, under its own gate key, not
       for it.
    3. **The lane's open period, the live sample, and the window tie.** The session joined lane `lane`'s period open at
       the reserve, a Pearl-C one (its open commits none of H1's fields: `credit.pearl_c_period_credit` credits a salt
       session in no other). The verifier's sample source is a sample session's coin opened on the run's commitment, that
       session the lane's first reserve after the period closed (`pearlplus_live_sample` over `pearlplus_draw`'s check),
       and the verdict is over the coin published at its sample (`pearlplus_coin`).
       Pearl+'s verdict, the Lean `PouwVerifier.PearlPlus` one (`PEARLPLUS_LEAN`, ``served_passes.py lean``'s answer),
       accepts, credits work over tiles, and held its passes to a window record W (the Lean's ``--tie W``), the
       window's docket R under `spot` is chained right after W, and its records audit is tied to W
       (``records_audit --tie W``) and accepted (its summary's ``audit.accepted``), so the window SPOT audited reads the roots Pearl+ credits. The rule that audit's
       summary states is the sampling design R commits (its kind, and its K unless stratified), and only a stratified
       rate's leaves the answer without a weaker tie. The period's
       open (in the signed ledger) committed the SHA-512 of the window-circuit record V3c held R's circuit to
       (`CIRCUIT_RECORD` beside R), so the period holds the salt session to the circuit of the window served.

    The answer is the Lean's, in core's answer form: its run commitment, its work (in the grid form, with its
    ``give_up``, where it held the passes to their class's fixed load) and W, the coin record's rounds and dropped rounds
    it read, and a profile of its source, over-cap rule, exact work and opened tiles at `PEARLPLUS_DELTA`, with the lane,
    pin, period and sessions. The Python verify's answer (`PEARLPLUS_ANSWER`) is not read. Nothing here decides anew:
    each condition reads what a check or a verifier answered."""
    from verity.protocols.accounting.compute.pouw.verifier import window as Wd
    from verity.protocols.proof.spot.verifier.challenger import remote
    from verity.protocols.proof.spot.verifier.one_stage import draw as D
    from verity.commitments import randomness as RND
    from verity.protocols.proof.spot.verifier.one_stage.verdict_record import DESIGN
    from verity.protocols.accounting.compute.pouw.prover.serving.coins import challenger as PHC

    if doc.get("coin_source") != LIVE_SOURCE:
        return None, f"its epoch's coins come from {doc.get('coin_source')!r}, not a live session on the auditor device's challenger"
    check = (doc.get("coins") or {}).get("check") or {}
    if check.get("reason") is not None or not check.get("passes"):
        return None, f"the live session's check refused its passes: {check.get('reason') or 'it checked none'}"
    if not key:
        return None, "the compose holds the coins to no key pin (--key-sha512, H1_KEY_SHA512)"
    session = str(check.get("session"))
    try:
        PHC.pinned_key(piece_dir / "session" / session / "out" / remote.KEY, key)
    except PHC.CoinError as e:
        return None, f"the live session's challenger key is not the compose's pin: {e}"
    try:
        lines = [x for x in (piece_dir / "ledger.jsonl").read_text().splitlines() if x.strip()]
        reserve, period = _ledger_period(lines, session)
    except (OSError, ValueError, AttributeError) as e:
        return None, f"the piece's ledger can't be read ({type(e).__name__}: {e})"
    if reserve is None:
        return None, f"the piece's ledger holds no reserve of the live session {session}"
    rb = reserve.get("body") or {}
    if rb.get("for") is not None:
        return None, f"the live session was reserved by {rb.get('by')} for lane {rb['for']}, not under the lane's own gate key"
    if not lane or rb.get("by") != lane:
        return None, f"the live session was reserved under lane {rb.get('by')}'s gate key, not lane {lane}'s own"

    if period is None:
        return None, f"the live session joined no period of lane {lane}"
    name = period.get("period")
    if h1 := [f for f in (Wd.K_FIELD, Wd.ROWS_FIELD, Wd.LENGTH_FIELD) if f in period]:
        return None, (f"the live session joined period {name}, a non-Pearl-C one (its open commits {', '.join(h1)}): a Pearl-C salt "
                      "session is credited in a Pearl-C period alone")
    try:
        coin = json.loads((piece_dir / COIN_RECORD).read_text())
    except (OSError, ValueError) as e:
        return None, f"the verifier's coin record can't be read ({type(e).__name__}: {e})"
    drew = (doc.get("coins") or {}).get("draw") or {}
    if (why := pearlplus_live_sample(coin, piece_dir, key, lane, drew, lines, str(name))) is not None:
        return None, why
    published = doc.get("coin") or {}
    if not published or published.get("reason") is not None:
        return None, f"the verdict is not held to the coin published at its draw: {published.get('reason') or 'none published'}"
    try:
        lean = json.loads((piece_dir / PEARLPLUS_LEAN).read_text())
    except (OSError, ValueError) as e:
        return None, f"no Lean PouwVerifier.PearlPlus verdict ({PEARLPLUS_LEAN}: {type(e).__name__}: {e})"
    if not isinstance(lean, dict) or lean.get("verdict") != "accept":
        v = lean.get("verdict") if isinstance(lean, dict) else None
        return None, f"the Lean PouwVerifier.PearlPlus verdict is {v}: {lean.get('reason') if isinstance(lean, dict) else lean}"
    work = lean.get("work") or {}
    credited, tiles = work.get("credited"), work.get("credited_tiles")
    if not (type(credited) is int and credited > 0 and type(tiles) is int and tiles > 0):
        return None, f"the Lean verdict credits {credited!r} MACs over {tiles!r} tiles: nothing"
    if (lean.get("commitment"), lean.get("source")) != (coin.get("commitment"), coin.get("source")):
        return None, (f"the Lean verdict is on run {str(lean.get('commitment'))[:16]}... at source "
                      f"{str(lean.get('source'))[:16]}..., not the verifier's coin record's")
    w = (lean.get("tie") or {}).get("record")
    if not isinstance(w, str):
        return None, ("the Lean verdict holds its passes to no window record W (served_passes.py lean, the Lean's --tie W), "
                      "so nothing ties them to the window")
    if spot is None:
        return None, "the compose names no window registration to tie W to (--window-tie, else --sampled-proofs)"
    try:
        reg = json.loads((spot / WINDOW_REGISTRATION).read_text())
        summary = json.loads((spot / AUDIT_SUMMARY).read_text())
    except (OSError, ValueError) as e:
        return None, f"the window's registration or its audit's summary can't be read ({type(e).__name__}: {e})"
    if reg.get("previous") != w:
        return None, (f"the window's registration R is chained after {str(reg.get('previous'))[:16]}..., not right after "
                      f"Pearl+'s window record W {w[:16]}...")
    if ((summary.get("tie") or {}).get("record")) != w:
        return None, f"the window's records audit is not tied to Pearl+'s window record W {w[:16]}... (records_audit --tie W)"
    if (accepted := (summary.get("audit") or {}).get("accepted")) is not True:
        return None, (f"the window's records audit tied to W didn't accept ({AUDIT_SUMMARY}'s audit.accepted is "
                      f"{accepted!r}), so nothing shows the window SPOT audited reads the roots Pearl+ credits")
    design, committed, n = summary.get("sampling_design"), reg.get(DESIGN), reg.get("population")
    rate, strata = RND.parse_rate(design), RND.parse_strata_rate(design)
    kind = "subset" if rate else None if strata is None else "work" if strata[0] else "stratified"
    k = (None if kind in (None, "stratified") else -1 if type(n) is not int else D.rate_count(*rate, n) if rate
         else D.rate_count(strata[1], strata[2], 1, n))
    if not (isinstance(committed, Mapping) and kind and committed.get(DESIGN) == kind and k in (None, committed.get("k"))):
        return None, (f"the window's records audit states the rule {design!r}, not the one its registration R commits "
                      f"({committed!r} over {n!r} proof blocks): what Pearl+'s tie covers is read from R's rule")
    try:
        held = hashlib.sha512((spot / CIRCUIT_RECORD).read_bytes()).hexdigest()
    except OSError as e:
        return None, (f"no window-circuit record V3c held R's circuit to ({CIRCUIT_RECORD} beside R: {type(e).__name__}), "
                      f"so nothing holds period {name}'s circuit to the window served")
    if period.get("circuit") != held:
        return None, (f"period {name} committed circuit {str(period.get('circuit'))[:16]}..., not {held[:16]}..., the "
                      "SHA-512 of the window-circuit record V3c held R's circuit to: its salt session counts toward "
                      "another circuit than the window served")

    weaker = {} if kind == "stratified" else {"weaker_tie": WEAKER_TIE.format(design=design)}
    return {"protocol": PEARLPLUS_PROTOCOL, "verdict": "accept", "reason": None, "commitment": lean.get("commitment"),
            "public": {"work": {k: work.get(k) for k in ("credited", "credited_tiles", "checked_macs", "rho", "epsilon")}
                       | ({"give_up": work["give_up"]} if "give_up" in work else {}),
                       "unit": "MAC", "tie": lean["tie"],
                       **{k: coin[k] for k in ("rounds", "dropped") if k in coin}},
            "profile": {"delta": PEARLPLUS_DELTA, "source": lean.get("source"), "over_cap": lean.get("over_cap"),
                        "tiles": [[v.get("pass"), v.get("tile")] for v in lean.get("verdicts") or () if isinstance(v, dict)],
                        "lane": lane, "key_sha512": key, "period": name, "session": session,
                        "draw_session": drew.get("session"),
                        "lean": {k: lean.get(k) for k in ("verdict", "commitment", "source", "drawn", "work", "tie")},
                        **weaker}}, None


def piece(key: str, pieces_dir: Path, run: str | None = None, **of_record: Any
          ) -> tuple[dict[str, Any], tuple[str | None, str]]:
    """A subprotocol's `piece_row` (`of_record`: its ``spot``, ``key_sha512`` and ``lane``) and `piece_server`;
    a piece that left a file that isn't JSON is its ``error``, about no server."""
    try:
        return piece_row(key, pieces_dir, run, **of_record), piece_server(key, pieces_dir)
    except ValueError as e:
        return ({"verdict": "error", "owner": PIECES.get(key, WIPE)[2],
                 "reason": f"its piece left a file that isn't JSON: {type(e).__name__}: {e}"[:2000]}, (None, "real"))


def piece_row(key: str, pieces_dir: Path, run: str | None = None, *, spot: Path | None = None,
              key_sha512: str | None = None, lane: str | None = None) -> dict[str, Any]:
    """A subprotocol's runner row from its piece's answer; Pearl+'s accepted development piece is PoUW's verdict of record
    where `pearlplus_of_record` says so (`spot`, `key_sha512` and `lane` what it reads), its row that answer's
    with its document as ``piece``, else ``missing`` with its document, its ``why`` `DEV_VERDICT` and the first condition
    of record it fails; any other development piece's is ``missing`` with its document when it accepts or claims no
    verdict (`UNCLAIMED`; it credits nothing, for the document's ``why``, else `DEV_WHY`), else ``reject``; a piece that
    ran (`RAN`) and wrote neither is its ``error``, unless it refused to run (`REFUSED`): then ``refused`` with its
    sentence as ``reason``; otherwise ``missing``, naming why."""
    from verity_e2e_demo import block_record as BR

    _name, comp, owner, what = PIECES.get(key, WIPE)
    d = pieces_dir / comp
    f, dev, ran = d / "answer.json", d / DEVELOPMENT, _json(d / RAN)
    if not f.exists() and dev.exists():
        doc = _json(dev)
        if not isinstance(doc, dict):
            return {"verdict": "error", "owner": owner, "reason": f"its piece's {DEVELOPMENT} is not a JSON object"}
        if key == "work" and doc.get("piece") == "pearlplus" and doc.get("verdict") == "accept":
            answer, failed = pearlplus_of_record(doc, d, spot, key_sha512, lane)
            if answer is not None:
                return {**row_of(owner, answer, d, run), "piece": doc, "answer_from": PEARLPLUS_FROM}
            return {"verdict": "missing", "owner": owner, "why": f"{DEV_VERDICT}: {failed}"[:2000], "development": doc}
        if doc.get("verdict") in ("accept", UNCLAIMED):
            own = doc.get("why") or DEV_WHY
            refused = (doc.get("credit") or {}).get("window")
            if isinstance(refused, list) and len(refused) == 2:
                refused = f"({refused[0]}) {refused[1]}"
            why = own if refused is None else (f"{own}; core's credit refuses the window "
                                               f"{refused if isinstance(refused, str) else json.dumps(refused)}")
            return {"verdict": "missing", "owner": owner, "why": why[:2000], "development": doc}
        return {"verdict": "reject", "owner": owner, "reason": "; ".join(doc.get("reasons") or ())[:2000],
                "development": doc}
    if not f.exists():
        if (d / REFUSED).exists():
            said = (d / REFUSED).read_text(errors="replace").strip()
            return {"verdict": BR.REFUSED, "owner": owner, "reason": said[:2000]}
        if isinstance(ran, dict):
            late = f", out of time at {ran.get('limit')}" if ran.get("rc") in (124, 137) else ""
            log = "in the run's private directory" if key == "wipe" else f"{comp}/piece.log"
            return {"verdict": "error", "owner": owner,
                    "reason": f"its piece ran (exit {ran.get('rc')}{late}) and wrote no answer; its log is {log}"}
        return {"verdict": "missing", "owner": owner, "why": _unset(comp, what)}
    try:
        return row_of(owner, json.loads(f.read_text()), f.parent, run)
    except ValueError as e:
        return {"verdict": "error", "owner": owner, "reason": f"{type(e).__name__}: {e}"[:2000]}


def piece_server(key: str, pieces_dir: Path) -> tuple[str | None, str]:
    """The server a piece's answer is about and the answer's source (``real`` or ``fixture``): the server its `SERVED`
    record names, else, but for the wipe, the node `pieces` ran it on (`RAN`); None where neither names one."""
    comp = PIECES.get(key, WIPE)[1]
    served, ran = _json(pieces_dir / comp / SERVED), _json(pieces_dir / comp / RAN)
    served = served if isinstance(served, dict) else {}
    ran = ran if isinstance(ran, dict) and key != "wipe" else {}
    return served.get("server") or ran.get("server"), served.get("source") or "real"


def _ratio(e: Fraction, deltas: Sequence[Fraction]) -> str:
    """``e`` exactly, as ``d + 2^-n`` where it is one of ``deltas`` plus a power of two."""
    for d in deltas:
        r = e - d
        if r > 0 and r.numerator == 1 and r.denominator & (r.denominator - 1) == 0:
            return f"{d} + 2^-{r.denominator.bit_length() - 1}"
    return str(e)


def terms(r: Mapping[str, Any], subprotocol: Mapping[str, Any] | None) -> list[str]:
    """What an accepted answer holds beyond its verdict: where its profile says when its weights were held and what share
    of the checkpoint they are, those; where it says which host ran each party (the network attester's ``placement``),
    the link K counts, named, with every other link of the node uncounted and uncertified, and what its `trusted-device`
    premise rests on (the wipe's never holds: under the key its challenger's host is pinned to, it reads that the key
    matched the pins this machine was handed, and otherwise rehearsal, either with no bound); and each δ of the block record's at which its subprotocol's quantity is not established. A placement
    with no links (the front's ``--gate local``) states no link."""
    profile = (r.get("answer") or {}).get("profile")
    w = profile.get("weights") if isinstance(profile, Mapping) else None
    place = profile.get("placement") if isinstance(profile, Mapping) else None
    place = place if isinstance(place, Mapping) else {}
    counted = [f"{x['from']} to {x['to']} through {x['via']}" for x in place.get("links") or () if x.get("counted")]
    others = place.get("uncounted") or "every other link of the node"
    out = ([f"K counts the relayed link alone, {'; '.join(counted)}; {others} is uncounted and uncertified"]
           if counted else [])
    premise = place.get("trusted-device") or place.get("trusted-host")   # a record made before the premise's rename
    out += [f"trusted-device {premise}"] if premise else []
    out += [f"{name} {why}" for name, why in (place.get("unchecked") or {}).items()]
    if isinstance(w, Mapping) and w.get("held"):
        out.append(f"its weights {w['held']}, not concurrently with the block")
    if isinstance(w, Mapping) and w.get("share"):
        outside = (profile.get("outside") or {}).get("bytes") or {}
        out.append(f"they are {float(Fraction(w['share'])):.1%} of the checkpoint ({w.get('covers', 'its weights')})"
                   + (f"; {', '.join(sorted(outside))} are outside its claim" if outside else ""))
    return out + (error_said(subprotocol) + bound_said(subprotocol) if subprotocol else [])


#: why the published lines give the sample's bound without its count: with the number of proof blocks sampled, which they
#: give, the count gives the population's size, which counts the execution phase's tokens
BOUND_HELD = ("b and ε = b/N in the serve dir's private/verdicts.txt: with the number of proof blocks sampled, b gives N, "
              "the population's size, which stays private")


def _sample_at(entry: Mapping[str, Any]) -> tuple[int, dict[str, Mapping[str, Any]]]:
    """A sampled-proofs entry's population size N and its certificate at each δ the record recomputes from its sample;
    none for another subprotocol's entry."""
    at = {d: x for d, x in (entry.get("at") or {}).items() if isinstance(x, Mapping) and x.get("by") == "draw"}
    return int((entry.get("quantity") or {}).get("of") or 0), at


def bound_said(entry: Mapping[str, Any]) -> list[str]:
    """What a sampled-proofs entry of the block record says of its sample in the published lines: at each δ it gives, at
    most b incorrect proof blocks except with probability δ (`BOUND_HELD`: b itself stays in the private copy,
    `bound_held`), and vacuous where b is the population's size N. Another subprotocol's entry says nothing here."""
    n, at = _sample_at(entry)
    vacuous = [d for d, x in at.items() if x["value"] >= n]
    bounded = [d for d in at if d not in vacuous]
    return ([f"at most b incorrect proof blocks except with probability δ, at δ = {' and '.join(bounded)} ({BOUND_HELD})"]
            if bounded else []) + ([f"vacuous at δ = {' and '.join(vacuous)}: every proof block may be incorrect"]
                                   if vacuous else [])


def bound_held(entry: Mapping[str, Any]) -> str | None:
    """The private copy's line of a sampled-proofs entry's bound: at each δ the record gives, at most b of the N proof
    blocks incorrect except with probability δ, with ε = b/N, vacuous where b = N; None for another subprotocol's."""
    n, at = _sample_at(entry)
    said = [f"at most {x['value']} of the {n} proof blocks incorrect except with probability {d}, ε = {x['epsilon']}"
            + (" (vacuous)" if x["value"] >= n else "") for d, x in at.items()]
    return "SAMPLED-PROOFS bound: " + "; ".join(said) if said else None


def keep(serve_dir: Path, line: str) -> None:
    """``line`` in the serve dir's private copy of its verdicts alone (``private/verdicts.txt``), as
    `served_verdicts.sh`'s ``keep`` writes it."""
    private = serve_dir / "private"
    private.mkdir(mode=0o700, exist_ok=True)
    with open(private / "verdicts.txt", "a") as f:
        f.write(line + "\n")


def auditor_device(host: str, manifest: Path | None, served_node: bool = False) -> str:
    """Whether the record was composed on the auditor's device: off it on the served node, which is never the auditor's;
    elsewhere as infra's deploy manifest, where `research` reads the auditor devices (``manifest``:
    ``tools/research/src/research/pods/nebius/deploy.toml``), says of ``host``: off it where it lists the node with
    neither ``auditor = true`` nor ``admin = true``, a node the lanes' keys reach; and not known where it marks the node
    either, lists no such node, can't be read, or none is given. Never on it: a manifest line is no device's word."""
    if served_node:
        return "composed off the auditor's device: on the served node"
    if manifest is None:
        return "not known: no deploy manifest given (record --deploy-manifest)"
    import tomllib  # here, not at the top: the lines `say` publishes run this file under the node's python3, maybe older

    try:
        nodes = tomllib.loads(manifest.read_text()).get("nodes") or {}
    except (OSError, ValueError) as e:
        return f"not known: the deploy manifest {manifest.name} can't be read ({type(e).__name__})"
    node = nodes.get(host)
    if not isinstance(node, Mapping):
        return f"not known: the deploy manifest lists no node {host}"
    if node.get("auditor") is True:
        return f"not known: the deploy manifest marks {host} auditor = true, a line of the manifest, no device's word"
    if node.get("admin") is True:
        return f"not known: the deploy manifest marks {host} admin = true, which it gives the control host too"
    return "composed off the auditor's device"


def error_said(entry: Mapping[str, Any]) -> list[str]:
    """What an accepted subprotocol's block record entry says of its error: each δ of the record's at which its quantity
    is not established, with the error the run established it at, or, where its scheme's bound is refused
    (``unbounded``), with that refusal's sentence. Where the error leaves terms out (the bandwidth
    audit's ``named_terms``) it is said with them beside it, in the terms' own words (`block_record.error_form`, which
    for PoUB's is its ledger's ``DELTA_FORM`` with the error for δ): there, and in a clause of its own where the
    quantity is established at every δ."""
    from verity_e2e_demo import block_record as BR

    named = entry.get("named_terms")
    error = BR.error_form(_ratio(Fraction(entry["error"]), BR.DELTAS), named)
    no = [d for d, x in entry["at"].items() if not x["holds"]]
    if no and "members" in entry["quantity"]:
        return [f"at δ = {d}: {entry['at'][d]['why']}" for d in no]
    if no and "unbounded" in entry:
        return [f"its quantity is not established at δ = {' or '.join(no)}: {entry['unbounded']['why']}",
                *([f"its error reads {error}"] if named else [])]
    if no:
        return [f"its quantity is not established at δ = {' or '.join(no)}: the run established it except with {error}"]
    return [f"its error reads {error}"] if named else []


def said(rows: Mapping[str, Mapping[str, Any]], at: Mapping[str, tuple[str | None, str]], node: str,
         subprotocols: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """The composed verdict's reason: one clause a subprotocol, its verdict and, unless it accepted about ``node``, why; an
    accept about ``node`` with its `terms` (``legs``: each accepted one's block record subprotocol, by runner name)."""
    out = []
    for k, r in rows.items():
        v, server = r.get("verdict"), at[k][0]
        if v == "missing":
            s = r.get("why") if "development" in r else f"missing: {r.get('why')}"
        elif v == "dropped":
            s = f"dropped, {r.get('why')}"
        elif server not in (None, node):
            s = f"{v} about {server}, another server: carried, not composed"
        elif v == "accept":
            t = terms(r, (subprotocols or {}).get(k))
            s = ("accept" if server is not None else "accept, naming no server: not carried") + (f" ({'; '.join(t)})" if t else "")
            if unchecked := declared_unchecked(r.get("protocol"), "; ".join(t)):
                s += f", {unchecked}"
            if str(r.get("answer_from") or "").startswith(PEARLPLUS_FROM):
                s += f", {PEARLPLUS_SAID}"
                if weaker := ((r.get("answer") or {}).get("profile") or {}).get("weaker_tie"):
                    s += f", {weaker}"
        else:
            s = f"{v}: {r.get('reason')}" + (f", {u}" if (u := declared_unchecked(r.get("protocol"))) else "")
        out.append(f"{NAMES[k]}: {s}")
    return "; ".join(out)


def witness(paths: Sequence[str]) -> list[tuple[str, str]]:
    """Each of ``paths`` (relative to a serve dir) that is a served run's witness, with why (`WITNESS_DIRS`,
    `WITNESS_TOPS`, `WITNESS_NAMES`, H1's prover state under ``private/h1``, and a salt file, ``*.salts/``)."""
    out = []
    for p in paths:
        parts = Path(p).parts
        if not parts:
            continue
        why = next((WITNESS_DIRS[x] for x in parts[:-1] if x in WITNESS_DIRS), None)
        why = why or (WITNESS_TOPS.get(parts[0]) if len(parts) > 1 else None)
        why = why or ("H1's prover state (its key)" if parts[:2] == ("private", "h1") else None)
        why = why or next((w for pat, w in WITNESS_NAMES.items() if fnmatch.fnmatchcase(parts[-1], pat)), None)
        why = why or ("a salt file (prover only)" if any(x.endswith(".salts") for x in parts) else None)
        if why:
            out.append((p, why))
    return out


def rebase(serve_dir: Path, served_dir: str, paths: Sequence[str], *, held: bool = False) -> list[str]:
    """Each case among ``paths`` (relative to ``serve_dir``, a hand-off's copy of the served node's serve dir
    ``served_dir``) that names ``blobs``, its blobs rewritten to the same paths under ``serve_dir``; the cases rewritten.
    ValueError on a blob outside ``served_dir``, which no hand-off carries, or on such a file that isn't JSON; with
    ``held``, on a blob that isn't one of ``paths``, the files the copy was held to, so nothing outside them is read."""
    base, done = served_dir.rstrip("/") + "/", []
    listed = {os.path.normpath(p) for p in paths}
    for rel in paths:
        f = serve_dir / rel
        if f.suffix != ".json" or not f.is_file() or b'"blobs"' not in f.read_bytes():
            continue
        case = json.loads(f.read_text())
        if not isinstance(case, dict) or not isinstance(case.get("blobs"), list):
            continue
        outside = [b for b in case["blobs"] if not (isinstance(b, str) and b.startswith(base))]
        if outside:
            raise ValueError(f"{rel} names a blob outside the served node's serve dir {served_dir}: {str(outside[0])[:200]}")
        rels = [os.path.normpath(b[len(base):]) for b in case["blobs"]]
        if held and (unheld := [r for r in rels if r not in listed]):
            raise ValueError(f"{rel} names a blob that isn't a file of the hand-off: {unheld[0][:200]}")
        case["blobs"] = [str(serve_dir / r) for r in rels]
        f.write_text(json.dumps(case))
        done.append(rel)
    return done


def composed(answer: Path, out: Path, gate_exit: int | None = None, err: Path | None = None) -> tuple[int, str]:
    """The compose gate's answer (one JSON line) written out under ``out``: its $OUT's files at their paths, and
    ``gate.json`` (the pin and the exit, without the files). Its exit (0 composed and accepted) and its one line, which
    without an answer names the gate command's stderr ``err`` by its last exception's class alone (`failed`)."""
    try:
        doc = json.loads(answer.read_text().strip().splitlines()[-1])
    except (OSError, IndexError, ValueError) as e:
        return 1, (f"GATE-COMPOSE failed (ssh exit {gate_exit}): no answer from the compose gate ({type(e).__name__}): "
                   f"its stderr {failed(gate_exit, [str(err)] if err else [])}")
    if "refused" in doc:
        return 3, f"GATE-COMPOSE refused by the compose gate: {str(doc['refused'])[:800]}"
    c = doc.get("compose") or {}
    files = c.get("files") or {}
    root = out.resolve()
    for rel, text in files.items():
        f = (out / rel).resolve()
        if root not in f.parents:
            return 1, f"GATE-COMPOSE failed: the answer names {rel[:200]!r}, outside its output directory"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    (out / "gate.json").write_text(json.dumps({**doc, "compose": {k: v for k, v in c.items() if k != "files"}},
                                              indent=1, sort_keys=True) + "\n")
    record = next((ln for ln in files.get("verdicts.txt", "").splitlines() if ln.startswith("RECORD ")), None)
    try:
        line = json.loads(record[len("RECORD "):]) if record else {}
    except ValueError as e:
        return 1, f"GATE-COMPOSE failed: the compose's RECORD line isn't JSON ({e})"
    said = line.get("composed") if isinstance(line, dict) else None
    where = f"at the compose gate's tree {str(doc.get('pinned'))[:12]}"
    rc = 0 if c.get("rc") == 0 and gate_exit in (None, 0) and said == "accept" else 1
    return rc, (f"GATE-COMPOSE {where}: exit {c.get('rc')}, the record {said or 'not written'}, "
                f"{len(files)} files of its $OUT in the run's compose/" + (f", {len(c['not_sent'])} not sent" if c.get("not_sent") else ""))


def served_compare_line(path: Path) -> str:
    """The served compare's verdict line, as `served_compare.py` kept it in ``path`` (its ``line``), or why there is
    none."""
    try:
        line = json.loads(path.read_text())["line"]
    except FileNotFoundError:
        return f"SERVED-COMPARE: none ({path.name} absent: served_compare.py wrote no verdict)"
    except (OSError, ValueError, KeyError, TypeError) as e:
        return f"SERVED-COMPARE: none ({path.name} can't be read: {type(e).__name__}: {e})"[:600]
    return str(line)


class CompositeRefused(ValueError):
    """A composite that is not the record's window's: `record` takes no answer from it and exits 1."""


def window_digest(audit: Path) -> tuple[str | None, str]:
    """The digest of the window's docket R, the record SPOT's audit ``audit`` sampled against
    and every case of the window commits to (`served_cases.window_case`, `served_cases.record_of`), and where it was
    read; or None and why there is none."""
    import served_cases as SC

    try:
        return SC.record_of(SC.window_case(audit))[1], f"{audit}/window/case.json"
    except SC.Refused as e:
        return None, f"SPOT's audit {audit} names no window record: {e}"


def composite_tie(doc: Mapping[str, Any], window: str | None, read: str) -> str | None:
    """Why the public composite ``doc`` is not of the window whose docket has digest ``window`` (read
    from ``read``), or None when it is: its SPOT case (`served_cases.SAMPLED`) names that digest, and its commitment is
    `composite.bind` of its cases' digests, so the commitment its Lean judged binds that window. The Lean ties the other
    cases to the same record (`Core.Composition.same`, `Core.Remote.windows`), and accepts only where every case names
    its digest and the commitment binds them (`Core.Composition.judge`); a composite that accepts with a case naming no
    digest is refused here too."""
    import composite as C
    import served_cases as SC

    if window is None:
        return f"there is no window record to tie it to ({read})"
    cases = [c for c in doc.get("cases") or [] if isinstance(c, dict)]
    named = [c["commitment"] for c in cases if c.get("id") == SC.SAMPLED and c.get("commitment")]
    if named != [window]:
        return (f"its {SC.SAMPLED} case names {named or 'no record'}, not this window's record R "
                f"{window[:16]}… ({read})")
    if not all(c.get("commitment") for c in cases):
        if (doc.get("verdict") or {}).get("verdict") == "accept":
            return "it accepts with a case that names no digest, so its commitment binds no digest of that case"
    else:
        bound = C.bind(str(doc.get("product")), [(str(c.get("id")), bytes.fromhex(c["commitment"]))
                                                  for c in sorted(cases, key=lambda c: str(c.get("id")))]).hex()
        if doc.get("commitment") != bound:
            return "its commitment is not the bind of its cases' digests (composite.bind)"
    return None


def composite_answer(public: Path, product: str, window: str | None = None,
                     read: str = "no window record was read") -> tuple[dict[str, Any] | None, str]:
    """``product``'s Lean check's answer as `composite.py` wrote it to its public output under ``public`` (a directory
    per composite run, `served_composite.sh`'s $OUT/composite), in the block record's form (`block_record.answer_of`:
    the protocol, the verdict, the refusal's public name as the reason, and the composite's commitment), and where it was
    read; or None and why there is no answer to take: a composite.json that doesn't parse, no composite of one of the
    product's checks (`block_record.CHECKS`) or more than one, one whose Lean gave no answer, or one composed as a
    development check, which claims no record. `CompositeRefused` where the composite is not of the window whose
    docket has digest ``window`` (`composite_tie`)."""
    from verity_e2e_demo import block_record as BR

    found, development = [], []
    for d in sorted(p for p in public.iterdir() if (p / "composite.json").is_file()) if public.is_dir() else []:
        try:
            doc = json.loads((d / "composite.json").read_text())
        except ValueError as e:
            return None, f"{d.name}/composite.json doesn't parse, so no composite's answer is taken: {e}"
        if isinstance(doc, dict) and doc.get("product") in BR.CHECKS[product]:
            found.append((d, doc))
        elif isinstance(doc, dict) and str(doc.get("product")).startswith(f"verity/{product}/"):
            development.append(f"{d.name}/composite.json ({doc['product']})")
    if not found and development:
        return None, (f"not of record: development composite: {development[0]}, not the {product} check of record "
                      f"({', '.join(BR.CHECKS[product])})")
    if len(found) != 1:
        return None, (f"{len(found)} composites of the {product} check ({', '.join(BR.CHECKS[product])}) in {public}, "
                      "not one")
    d, doc = found[0]
    where = f"{d.name}/composite.json"
    if doc.get("development"):
        return None, f"{where} is a development check, which claims no record: {str(doc['development'])[:400]}"
    if (tie := composite_tie(doc, window, read)) is not None:
        raise CompositeRefused(f"{where} is refused, not this window's composite: {tie}")
    v = doc.get("verdict") if isinstance(doc.get("verdict"), dict) else {}
    if v.get("id") != doc["product"] or v.get("verdict") not in BR.VERDICTS:
        return None, f"{where}: its Lean gave no answer ({json.dumps(v, sort_keys=True)[:200]})"
    return ({"protocol": v["id"], "verdict": v["verdict"], "reason": v.get("refusal"), "commitment": doc.get("commitment")},
            f"{doc['product']}'s Lean check, compiled, in {where}")


def inference_only_row(v: Mapping[str, Any]) -> dict[str, Any]:
    undecided = str(v.get("reason") or "").startswith(UNDECIDED)
    return {"verdict": {"accept": "accept", "refuse": "error" if undecided else "reject"}.get(v.get("verdict"), "error"),
            "protocol": v.get("format"), "reason": v.get("reason"),
            "answer": {k: v[k] for k in ("classes", "c", "k", "n", "lean", "record_count", "theta", "failure") if k in v},
            "answer_from": f"{AS_WRITTEN} ({INFERENCE_ONLY})", "combined": INFERENCE_ONLY_UNREAD}


def record(rows: Mapping[str, Mapping[str, Any]], *, window: int, steps: tuple[int, int], node: str,
           run: str | None, servers: Mapping[str, tuple[str | None, str]] | None = None, product: str = "onsite",
           deferred: str | None = None, off_node: bool = False, tree: Mapping[str, str] | None = None,
           served_compare: str | None = None, answer: Mapping[str, Any] | None = None,
           lean: str = "no composite was read", inference_only: Mapping[str, Any] | None = None,
           front_group: bool = False, manifest: Path | None = None
           ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """The window report over ``rows`` (by runner name), its block record (None when a subprotocol about this node
    rejected, erred or was undecided), and the ``RECORD`` line's summary, composed for ``product``: each subprotocol it
    drops is its `dropped_row`, about no server. ``servers``: each piece's `piece_server`, by runner name; SPOT's
    answer is about ``node``, and a piece's that names no server isn't carried. ``deferred``: what the run left to a
    follow-on and why, and ``served_compare``: the served compare's verdict line (`served_compare_line`), each of which
    the report and the line carry. ``off_node``: composed from the served node's compose
    hand-off (`HANDED_FROM`); ValueError on ``node`` itself. ``tree``: the composing tree's ``tree`` (its commit) and
    ``tree_on`` (the branch its host pinned it from), which ``composed_on`` carries. ``answer``: the product's compiled
    Lean check's answer (`composite_answer`), the block record's verdict, and ``lean`` where it was read or why there is
    none, which the report and the line carry; a rejection by it rejects whatever the rows say. ``inference_only``: the
    inference-only verifier's verdict, which the report carries as its ``inference-only`` row and the line under
    ``not_combined``, and which neither the combined verdict nor the block record reads: ``reject`` only where the Lean
    verifier gave a verdict, ``error`` where it gave none (`UNDECIDED`). ``front_group``: the network attester's row is
    over its front's group (`FRONT_GROUP`), which its row and the line's ``not_covered`` say, and the block record
    doesn't read. ``manifest``: infra's deploy manifest, by which ``composed_on`` says whether this host is the
    auditor's device (`auditor_device`)."""
    from verity_e2e_demo import block_record as BR

    host = socket.gethostname()
    if off_node and host == node:
        raise ValueError(f"a record composed off the served node is not composed on it: this host is {node}")

    subprotocols = {v: k for k, v in BR.RUNNER_KEYS.items()} | {"wipe": "wipe"}
    drops = [k for k in rows if subprotocols[k] not in BR.PRODUCTS[product]]
    at = ({k: (None, "real") for k in rows} | dict(servers or {}) | {"integrity": (node, "real")}
          | {k: (None, "real") for k in drops})
    rows = {k: dropped_row(k, product) if k in drops else dict(r) for k, r in rows.items()}
    apart = {"communication"} & set(rows) - set(drops) if front_group else set()
    for k in apart:
        rows[k]["requests"] = FRONT_GROUP
    for k, r in rows.items():
        if k not in drops and "answer_from" not in r and (r.get("verdict") != "missing" or "development" in r):
            r["answer_from"] = f"{AS_WRITTEN} ({WRITTEN.get(k) or f'its piece files in {PIECES.get(k, WIPE)[1]}/'})"
    rows = {k: r if k in drops else ran_by(k, r, product) for k, r in rows.items()}
    built: dict[str, dict[str, Any]] = {}
    for k, r in rows.items():
        if r.get("verdict") == "accept" and at[k][0] is not None:
            try:
                built[k] = BR.subprotocol(subprotocols[k], r, server=at[k][0], runs=[], source=at[k][1])
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as e:
                rows[k] = {**r, "verdict": "error", "quantity": None, "epsilon": None,
                           "reason": f"its answer is not the block record's {subprotocols[k]} subprotocol: {type(e).__name__}: {e}"[:2000]}
    where = {"host": host, "served_node": host == node,
             "from": HANDED_FROM if off_node else COMPOSED_FROM if host == node else COPIED_FROM,
             "auditor_device": auditor_device(host, manifest, host == node), **(tree or {})}
    left = {k: v for k, v in (("deferred", deferred), ("served_compare", served_compare)) if v}
    report = {"format": REPORT_FORMAT, "window": window, "node": node, "run": run, "protocols": rows,
              "servers": {k: at[k][0] for k in rows}, "composed_on": where, **left}
    verdicts = {NAMES[k]: DEV_VERDICT if "development" in r and r.get("verdict") == "missing" else r.get("verdict")
                for k, r in rows.items()}
    missing = {NAMES[k]: r.get("why") for k, r in rows.items() if r.get("verdict") == "missing"}
    elsewhere = {NAMES[k]: at[k][0] for k in rows if at[k][0] not in (None, node)}
    unserved = sorted(NAMES[k] for k, r in rows.items()
                      if r.get("verdict") not in ("missing", "dropped", BR.REFUSED) and at[k][0] is None)
    dev = {"development": {NAMES[k]: r["development"]["verdict"] for k, r in rows.items() if "development" in r}}
    dev = dev if dev["development"] else {}
    answers = {NAMES[k]: r["answer_from"] for k, r in rows.items() if "answer_from" in r}
    named = {"product": product, "verdicts": verdicts, "missing": missing, "elsewhere": elsewhere, "composed_on": where,
             "answers": answers, "lean": lean, **left,
             **({"dropped": {NAMES[k]: rows[k]["why"] for k in drops}} if drops else {}),
             **({"unserved": unserved} if unserved else {}), **dev,
             **({"not_covered": {NAMES[k]: FRONT_GROUP for k in apart}} if apart else {})}
    io = {} if inference_only is None else {
        INFERENCE_ONLY_ROW: ran_by(INFERENCE_ONLY_ROW, inference_only_row(inference_only), product)}
    if io:
        named["not_combined"] = {"the inference-only verifier": io[INFERENCE_ONLY_ROW]["verdict"]}
        answers["the inference-only verifier"] = io[INFERENCE_ONLY_ROW]["answer_from"]
    bad = sorted(NAMES[k] for k, r in rows.items()
                 if r.get("verdict") not in ("accept", "missing", "dropped", BR.REFUSED) and NAMES[k] not in elsewhere)
    if bad:
        composed = "reject" if (answer or {}).get("verdict") == "reject" or any(
            rows[k].get("verdict") in ("reject", "error") for k in rows if NAMES[k] in bad) else "pending"
        report["composed"] = {"protocol": REPORT_FORMAT, "verdict": composed, "reason": said(rows, at, node, built),
                              "lean": lean}
        line = {"composed": composed, "by": bad, **named, "block_record": None, "reason": report["composed"]["reason"],
                "why": "block_record carries an accepted subprotocol's certificate only"}
        report["protocols"] = {**rows, **io}
        return report, None, line
    extra = {subprotocols[k]: {"row": r, "server": at[k][0] or node, "runs": r.get("runs") or [run], "source": at[k][1]}
             for k, r in rows.items() if k != "integrity" and k not in apart
             and (r.get("verdict") == "accept" and at[k][0] is not None or r.get("verdict") == BR.REFUSED)}
    composing = {**report, "protocols": {k: r if k == "integrity" else {"verdict": "missing"} for k, r in rows.items()}}
    rec = BR.from_period(composing, steps=steps, deployment=BR.REFERENCE, answer=answer, extra=extra, product=product)
    verdict = WINDOW_VERDICT.get(rec["composed"]["verdict"], rec["composed"]["verdict"])
    report["composed"] = {"protocol": REPORT_FORMAT, "verdict": verdict, "reason": said(rows, at, node, built),
                          "lean": lean}
    line = {"composed": verdict, **named, "reason": report["composed"]["reason"],
            "record_missing": rec["composed"]["missing"],
            "check": BR.check(rec), "steps": list(steps), "node": node,
            "deployment": rec["monitoring_period"]["deployment"]["id"]}
    report["protocols"] = {**rows, **io}
    return report, rec, line


#: the commands of PoUW H1's piece, which is deleted: PoUW's piece is Pearl+'s (`pearlplus`)
H1_DELETED = {c: "PoUW H1 is deleted, so its serve's program, period and development piece are; PoUW's piece is Pearl+'s "
                 "(`pearlplus`, POUW=pearlplus)" for c in ("program", "period", "pouw")}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("window")
    w.add_argument("serve_dir", type=Path)
    w.add_argument("--slots", type=int, default=8)
    w.add_argument("--window", type=int)
    w.add_argument("--of-load", type=Path, help="the fixed load's lines (verity_vllm.inputs.sender --out): the window of its requests")
    w.add_argument("--attested", action="store_true", help="the window of the network attester's requests alone")
    g = sub.add_parser("gate")
    g.add_argument("--gate", default=os.environ.get("H1_GATE"))
    g.add_argument("--lane", default="vllm")
    g.add_argument("--main", type=Path, default=os.environ.get("MAIN_FILE") or None)
    g.add_argument("--root", help="a window's run root (128 hex digits) a session of this run opens on")
    g.add_argument("--registration-record", type=Path,
                   help="the deployment's registration record, whose registering period the lane's open one may be")
    pp = sub.add_parser("pearlplus")
    pp.add_argument("out", type=Path)
    pp.add_argument("--served", type=Path, help="served_passes.py serve's document")
    pp.add_argument("--verify", type=Path, help="served_passes.py verify's document")
    pp.add_argument("--unretained", type=Path, help="served_passes.py serve --no-retain's document of the same load")
    pp.add_argument("--volunteer", type=Path, help="served_passes.py volunteer's document of the run")
    pp.add_argument("--gate", help="the gate's command the serve reserved its live session through "
                                   "(PEARLC_CHALLENGER_GATE): the session is fetched through it")
    pp.add_argument("--key-sha512", help="the SHA-512 of the challenger's public key the deployment pins (H1_KEY_SHA512)")
    pp.add_argument("--main", type=Path, help="main's first-parent line, one commit a line (MAIN_FILE)")
    pp.add_argument("--coin", type=Path, help=f"the verifier's coin as `coin` published it (its {COIN_PUT})")
    pp.add_argument("--tie", type=Path, help="under PEARLPLUS_TIE, the window record W the window's passes are held to "
                                             "(retain --window-record's): --served is then the window's own epoch's")
    dl = sub.add_parser("pearlplus-draw-live")
    dl.add_argument("out", type=Path)
    dl.add_argument("--record", type=Path, required=True, help="the verifier's coin record (served_passes.py verify's)")
    dl.add_argument("--gate", required=True)
    dl.add_argument("--key-sha512", required=True)
    dl.add_argument("--main", type=Path, required=True)
    sl = sub.add_parser("pearlplus-salt-live")
    sl.add_argument("out", type=Path)
    sl.add_argument("--served", type=Path, required=True, help="served_passes.py serve's document")
    sl.add_argument("--gate", required=True)
    sl.add_argument("--key-sha512", required=True)
    sl.add_argument("--main", type=Path, required=True)
    cn = sub.add_parser("coin")
    cn.add_argument("out", type=Path)
    cn.add_argument("--record", type=Path, required=True, help="the verifier's coin record (served_passes.py verify's "
                                                               "DIR/coin.json)")
    cn.add_argument("--run", help="the run the coin was drawn in (RESEARCH_RUN_ID), the put's meta")
    cn.add_argument("--run-dir", type=Path, help="that run's directory (RESEARCH_RUN_DIR): its Attempt carries out/, the "
                                                 "coin's custody where no put preserves it")
    ppe = sub.add_parser("pearlplus-period")
    ppe.add_argument("verb", choices=("open", "close", "rule"))
    ppe.add_argument("piece_dir", type=Path)
    ppe.add_argument("--gate", required=True)
    ppe.add_argument("--name")
    ppe.add_argument("--scheme", help="default: the kernel of Pearl+'s declared option")
    described = ppe.add_mutually_exclusive_group()
    described.add_argument("--cubin", type=Path)
    described.add_argument("--circuit-record", type=Path, help="the served window's circuit record (PEARLPLUS-CIRCUIT's), "
                                                               "whose bytes are the period's descriptor")
    ppe.add_argument("--lane", help="close, rule: the lane whose gate key reserves the run's sessions (PEARLC_CHALLENGER_LANE)")
    ppe.add_argument("--windows", type=Path, help="close, with --lane: the serve dir's pearl/, whose w<k>/served.json name "
                                                  "each window's salt session, the period's reserves held to them first")
    ppe.add_argument("--wait", type=float, default=0, help="close, with --windows: at most this many seconds waited for "
                                                           "the gate's ledger to end each window's salt session")
    ppe.add_argument("--coin", type=Path, help="rule: the verifier's coin record (served_passes.py verify's coin.json)")
    rh = sub.add_parser("registration-hold", help="the serve's hold until the auditor's ledger signs its registration "
                        "for the lane (`registration_hold`); its limit is the caller's timeout")
    rh.add_argument("record", type=Path, help="the deployment's registration record (deployment-registration.json)")
    rh.add_argument("--lane", help="REGISTRATION_LANE; none: the development path, no hold")
    rh.add_argument("--ledger", help="REGISTRATION_LEDGER: the auditor's ledger as its host appends it, or gate:USER@HOST")
    rh.add_argument("--key", type=Path, help="REGISTRATION_KEY: the challenger's key record")
    rh.add_argument("--key-sha512", help="REGISTRATION_KEY_SHA512: the SHA-512 of the challenger's public key the operator "
                    "posted, which the key record must hold")
    rh.add_argument("--gate-key", help="with --ledger gate:USER@HOST: the lane's gate key")
    rh.add_argument("--every", type=float, default=10.0, help="with --ledger gate:USER@HOST: seconds between reads")
    rh.add_argument("--once", action="store_true", help="the ledger as it stands, no wait")
    rg = sub.add_parser("public-registration")
    rg.add_argument("registration", type=Path)
    pl = sub.add_parser("public-lines")
    pl.add_argument("--private", type=Path)
    pb = sub.add_parser("publish")
    pb.add_argument("out", type=Path)
    pb.add_argument("--private", type=Path, required=True)
    wr = sub.add_parser("window-record", help="the window road's verdict of record from the records audit's summary.json: "
                        "exit 0 accept, 1 reject or undecided, 2 a summary that is missing or lacks a field (ERROR)")
    wr.add_argument("summary", type=Path)
    for name in ("coin-trees", "coins"):
        ct = sub.add_parser(name)
        ct.add_argument("--challenger", required=True)
        ct.add_argument("--summary", type=Path, required=True)
    fl = sub.add_parser("front-line")
    fl.add_argument("session", type=Path)
    nl = sub.add_parser("not-listening")
    nl.add_argument("log", type=Path)
    nl.add_argument("rc", type=int)
    for name in ("failed", "said"):
        fa = sub.add_parser(name)
        fa.add_argument("rc")
        fa.add_argument("logs", nargs="*", default=["-"])
    p = sub.add_parser("pieces")
    r = sub.add_parser("record")
    for q in (p, r):
        q.add_argument("--serve-run", type=Path, required=True)
        q.add_argument("--window", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", action="append", choices=[v[1] for v in PIECES.values()])
    p.add_argument("--skip", action="append", default=[], choices=[v[1] for v in PIECES.values()])
    r.add_argument("--sampled-proofs", type=Path, required=True)
    r.add_argument("--pieces", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--deferred", help="what the run left to a follow-on and why, which the record carries")
    r.add_argument("--served-compare", type=Path, help="the served compare's served_compare.json (served_compare.py), "
                                                       "whose verdict line the record names")
    r.add_argument("--steps", help="LO,HI: the window's engine steps, in place of picking them from the serve dir")
    r.add_argument("--off-node", action="store_true", help="composed from the served node's compose hand-off, off it")
    r.add_argument("--sampled-lean", action="store_true",
                   help="sampled proofs' row from this host's run of its Lean on the case's proofs (lean_sampled_row)")
    r.add_argument("--sampled-exe", help="this tree's own vstar-recursive build path, with --sampled-lean, any other "
                                         "refused (default: built here)")
    r.add_argument("--composite", type=Path, help="the composites' public output (served_composite.sh's $OUT/composite): "
                                                  "the product's compiled Lean check's answer there is the record's verdict")
    r.add_argument("--tree", help="the commit of the tree composing the record, which composed_on names")
    r.add_argument("--deploy-manifest", type=Path, help="infra's deploy manifest, which composed_on's auditor_device reads")
    r.add_argument("--tree-on", help="the branch the composing host pinned that tree from, with --tree")
    r.add_argument("--key-sha512", help="the pin the challenger key file Pearl+'s piece hands must hash to for a verdict "
                                        "of record (H1_KEY_SHA512); the coins' signatures are verified on the served "
                                        "node only")
    r.add_argument("--lane", default="vllm", help="the lane whose own gate key Pearl+'s live session must be reserved "
                                                  "under, in its open period, for a verdict of record "
                                                  "(PEARLC_CHALLENGER_LANE)")
    r.add_argument("--window-tie", type=Path, help="where Pearl+'s tie to the window is read for a verdict of record: the "
                   "window tied to W, its window/registration.json, its records audit's summary.json and the "
                   "window-circuit record V3c held R's circuit to, circuit-record.json (default --sampled-proofs)")
    sub.add_parser("witness")
    rb = sub.add_parser("rebase")
    rb.add_argument("serve_dir", type=Path)
    rb.add_argument("served_dir")
    rb.add_argument("--held", action="store_true", help="refuse a blob that isn't one of the files on stdin")
    cd = sub.add_parser("composed")
    cd.add_argument("answer", type=Path)
    cd.add_argument("out", type=Path)
    cd.add_argument("--gate-exit", type=int)
    cd.add_argument("--err", type=Path)
    for q in (p, r):
        q.add_argument("--node", default=socket.gethostname())
        q.add_argument("--product", default="onsite")
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in H1_DELETED:
        print(f"served_verdicts.py {args[0]}: refused: {H1_DELETED[args[0]]}", file=sys.stderr)
        return 2
    a = ap.parse_args(args)
    if a.cmd in ("pieces", "record"):
        from verity_e2e_demo import block_record as BR

        if a.product not in BR.PRODUCTS:
            ap.error(f"--product {a.product!r}: not one of {sorted(BR.PRODUCTS)}")
    if a.cmd == "witness":
        found = witness([x for x in sys.stdin.read().splitlines() if x.strip()])
        for path, why in found:
            print(f"WITNESS {path}: {why}", flush=True)
        return 1 if found else 0
    if a.cmd == "rebase":
        try:
            done = rebase(a.serve_dir, a.served_dir, [x for x in sys.stdin.read().splitlines() if x.strip()], held=a.held)
        except (OSError, ValueError) as e:
            print(f"REBASE refused: {type(e).__name__}: {e}", flush=True)
            return 1
        print(f"REBASED {len(done)} case(s)' blobs from {a.served_dir} to {a.serve_dir}", flush=True)
        return 0
    if a.cmd == "composed":
        a.out.mkdir(parents=True, exist_ok=True)
        rc, line = composed(a.answer, a.out, a.gate_exit, a.err)
        print(line, flush=True)
        return rc
    if a.cmd == "public-registration":
        print(json.dumps(public_registration(json.loads(a.registration.read_text())), indent=1, sort_keys=True))
        return 0
    if a.cmd == "window-record":
        try:
            verdict, line = window_record(json.loads(a.summary.read_text()))
        except (OSError, ValueError) as e:
            print(f"WINDOW-RECORD ERROR: {type(e).__name__}: {e}", flush=True)
            return 2
        print(line, flush=True)
        return 0 if verdict == "accept" else 1
    if a.cmd == "coins":
        print(coins_said(a.challenger, _summary(a.summary)))
        return 0
    if a.cmd == "coin-trees":
        try:
            why = coin_trees_refusal(a.challenger, _json(a.summary) if a.challenger.startswith("gate:") else None)
        except ValueError:
            why = f"the audit's {a.summary} is not JSON, so nothing says where its sessions' coin trees came from"
        if why:
            print(why)
        return 1 if why else 0
    if a.cmd == "front-line":
        try:
            print(front_line(json.loads(a.session.read_text())))
        except (OSError, ValueError) as e:
            print(f"no session ({type(e).__name__})")
            return 1
        return 0
    if a.cmd == "not-listening":
        print(not_listening(a.log, a.rc))
        return 0
    if a.cmd in ("failed", "said"):
        print((failed if a.cmd == "failed" else tool_said)(a.rc, a.logs))
        return 0
    if a.cmd == "registration-hold":
        if a.lane is not None and (a.ledger is None or a.key is None):
            ap.error("registration-hold --lane needs --ledger and --key")
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(124))
        return registration_hold(a.record, a.lane, a.ledger, a.key, once=a.once, say=lambda x: print(x, flush=True),
                                 gate_key=a.gate_key, key_sha512=a.key_sha512, every=a.every)
    if a.cmd == "public-lines":
        public_lines(sys.stdin, sys.stdout, a.private)
        return 0
    if a.cmd == "publish":
        held = publish(a.out, a.private)
        if held:
            print(f"PUBLISH: {a.out}'s files that name a window or count its own are held in full in {a.private}, "
                  "their public form in their place", flush=True)
        return 0
    if a.cmd == "window":
        try:
            users = None if a.of_load is None else [r["user"] for r in _lines(a.of_load)]
            if users == []:
                raise ValueError(f"{a.of_load} names no request of the fixed load")
            print(json.dumps(pick(a.serve_dir, a.slots, a.window, users=users, attested=a.attested), sort_keys=True))
        except (ValueError, KeyError) as e:
            print(f"served_verdicts.py window: {e}", file=sys.stderr)
            return 1
        return 0
    if a.cmd == "gate":
        if not a.gate:
            print("GATE refused: no gate command (--gate, H1_GATE)", flush=True)
            return 1
        try:
            reg = None
            if a.registration_record is not None:
                from verity.commitments import docket_digest as DD
                reg = DD.digest(json.loads(a.registration_record.read_text()))
            ok, line = gate_ready(a.gate, a.lane, a.main, a.root, reg)
        except (KeyError, OSError, RuntimeError, ValueError) as e:
            ok, line = False, f"GATE refused: {type(e).__name__}: {e}"[:800]
        print(line, flush=True)
        return 0 if ok else 1
    if a.cmd == "pearlplus":
        try:
            doc = pearlplus(a.out, _json(a.served) if a.served else None, _json(a.verify) if a.verify else None,
                            _json(a.unretained) if a.unretained else None, _json(a.volunteer) if a.volunteer else None,
                            gate=a.gate, key=a.key_sha512, main=a.main, coin=a.coin, tie=a.tie)
        except PearlplusRefused as e:
            print(f"POUW-REFUSED {e}", flush=True)
            return 2
        print(f"POUW-DEV {json.dumps(development_line(doc), sort_keys=True, default=str)}", flush=True)
        return 0
    if a.cmd == "pearlplus-draw-live":
        try:
            path = pearlplus_draw_live(a.record, a.out, a.gate, a.key_sha512, a.main)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, tarfile.TarError) as e:
            print(f"DRAW-LIVE refused: {type(e).__name__}: {e}"[:800], flush=True)
            return 2
        print(f"DRAW-LIVE {path or "none: the sample is the node's"}", flush=True)
        return 0
    if a.cmd == "pearlplus-salt-live":
        try:
            path = pearlplus_salt_live(json.loads(a.served.read_text()), a.out, a.gate, a.key_sha512, a.main)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, tarfile.TarError) as e:
            print(f"SALT-LIVE refused: {type(e).__name__}: {e}"[:800], flush=True)
            return 2
        print(f"SALT-LIVE {path or "none: the run's coins are an in-process challenger's"}", flush=True)
        return 0
    if a.cmd == "pearlplus-period":
        try:
            if a.verb == "rule":
                if not (a.lane and a.coin):
                    raise ValueError("pearlplus-period rule needs --lane and --coin")
                held, line = pearlplus_period_rule(a.piece_dir, a.gate, a.lane, a.coin)
                print(line, flush=True)
                return 0 if held else 1
            if a.verb == "close":
                if (a.lane is None) != (a.windows is None):
                    raise ValueError("pearlplus-period close takes --lane and --windows together")
                doc = pearlplus_period_close(a.piece_dir, a.gate, a.lane, a.windows, a.wait)
                print(f"PERIOD closed {doc['name']}{period_reserves_said(doc)}: "
                      f"{json.dumps(public_json(doc['close']), sort_keys=True)[:600]}", flush=True)
                return 0
            if not a.name:
                raise ValueError("pearlplus-period open needs --name")
            from verity.protocols.run import schemes

            scheme = a.scheme or schemes.get("pearl-plus").option.partition(":")[2]
            doc = pearlplus_period_open(a.piece_dir, a.gate, a.name, scheme, a.cubin, a.circuit_record)
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            said = "RULE not read" if a.verb == "rule" else "PERIOD refused"
            print(f"{said}: {type(e).__name__}: {e}"[:1200], flush=True)
            return 1
        of = f"the window's circuit record {a.circuit_record}" if a.circuit_record else PEARLPLUS_DESCRIPTOR
        print(f"PERIOD opened {doc['name']}: Pearl-C, cap 1, circuit {doc['circuit'][:16]}... ({of})", flush=True)
        return 0
    if a.cmd == "coin":
        try:
            got = coin(a.record, a.out, a.run or None, run_dir=a.run_dir)
        except (OSError, ValueError) as e:
            print(f"COIN refused: {type(e).__name__}: {e}"[:800], flush=True)
            return 1
        if got["preserved"]:
            where = f"preserved in the evidence store as {got['art']}"
        elif got["art"]:
            where = (f"{got['art']} in this node's store, not preserved ({got['error']}): a put of the run's custody copy "
                     f"of {COIN_RECORD} as {COIN_KIND} with meta {{run, commitment}} preserves that id")
        else:
            where = f"in no store ({got['error']}): {COIN_RECORD} goes with the run's custody"
        if not got["preserved"] and got["custody"]:
            where += f"; the run's Attempt {got['custody']['attempt']} carries it at {got['custody']['path']}"
        print(f"COIN drawn on run commitment {got['commitment'][:16]}..., record sha512 {got['sha512'][:16]}...: {where}",
              flush=True)
        return 0
    if a.cmd == "pieces":
        a.out.mkdir(parents=True, exist_ok=True)
        got = pieces(a.serve_run, a.window, a.out, node=a.node, only=a.only, skip=a.skip, product=a.product)
        from verity_e2e_demo import block_record as BR

        return 0 if all(g["verdict"] in ("accept", "missing", "dropped", BR.REFUSED, UNCLAIMED) for g in got.values()) else 1
    if a.off_node and socket.gethostname() == a.node:
        print(f"RECORD refused: a record composed off the served node is not composed on it: this host is {a.node}",
              flush=True)
        return 1
    if a.steps is not None:
        given = re.fullmatch(r"(\d+),(\d+)", a.steps)
        if given is None:
            ap.error(f"--steps {a.steps!r}: LO,HI, two integers")
        steps = [int(x) for x in given.groups()]
    else:
        steps = pick(a.serve_run, 1 << 30, a.window)["steps"]
    run = os.environ.get("RESEARCH_RUN_ID")
    keys = [*PIECES, "wipe"]
    got = {k: piece(k, a.pieces, run, spot=a.window_tie or a.sampled_proofs, key_sha512=a.key_sha512, lane=a.lane)
           for k in keys}
    if a.sampled_exe is not None and not a.sampled_lean:
        ap.error("--sampled-exe is sampled proofs' Lean, which only --sampled-lean runs")
    if a.tree_on and not a.tree:
        ap.error("--tree-on names the branch of the tree --tree names")
    audit = a.sampled_proofs
    integrity = (lean_sampled_row(audit, a.serve_run / "private" / "scratch", a.sampled_exe, run)
                 if a.sampled_lean else spot_row(audit, run))
    rows = {"integrity": integrity, **{k: row for k, (row, _) in got.items()}}
    refused = False
    try:
        answer, lean = (composite_answer(a.composite, a.product, *window_digest(audit)) if a.composite
                        else (None, "no composite was read"))
    except CompositeRefused as e:
        answer, lean, refused = None, str(e), True
    try:
        io = json.loads((a.out / INFERENCE_ONLY).read_text()) if (a.out / INFERENCE_ONLY).is_file() else None
    except ValueError as e:
        io = {"verdict": "error", "reason": f"{INFERENCE_ONLY} doesn't parse: {e}"}
    if io is not None and not isinstance(io, dict):
        io = {"verdict": "error", "reason": f"{INFERENCE_ONLY} is not an object"}
    report, rec, line = record(rows, window=a.window, steps=(steps[0], steps[1]), node=a.node, run=run,
                               servers={k: at for k, (_, at) in got.items()}, product=a.product, deferred=a.deferred,
                               off_node=a.off_node, tree={"tree": a.tree, **({"tree_on": a.tree_on} if a.tree_on else {})}
                               if a.tree else None,
                               served_compare=None if a.served_compare is None else served_compare_line(a.served_compare),
                               answer=answer, lean=lean, inference_only=io,
                               front_group=(a.serve_run / "private" / FRONT_WINDOW).is_file(), manifest=a.deploy_manifest)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "verdicts.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    if rec is not None:
        (a.out / "block-record.json").write_text(json.dumps(rec, separators=(",", ":"), sort_keys=True) + "\n")
    print("RECORD " + json.dumps(line, sort_keys=True), flush=True)
    if rec is not None and (held := bound_held(rec["legs"].get("integrity") or {})):
        keep(a.serve_run, held)
    if refused or (rec is not None and line["check"]):
        return 1
    return RECORD_EXIT.get(line["composed"], 1)


if __name__ == "__main__":
    sys.exit(main())
