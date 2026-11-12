"""The verdicts on one served window (`served_verdicts.sh`): the window it audits, the subprotocols after
sampled proofs, and the one record of every verdict (the block record).

    served_verdicts.py window SERVE_DIR [--slots N] [--window K]
    served_verdicts.py gate [--gate CMD] [--lane vllm] [--main FILE]
    served_verdicts.py period open SERVE_DIR --name P --circuits DIR --slots N --max-tokens N [--gate CMD] [--draw-k K]
                                 [--model M]
    served_verdicts.py period close SERVE_DIR [--gate CMD]
    served_verdicts.py program OUT [--model M] [--rows T] [--steps S] [--like SERVE_DIR]
    served_verdicts.py pieces --serve-run DIR --window K --out DIR [--node NAME] [--only COMPONENT]... [--skip COMPONENT]...
                              [--product P]
    served_verdicts.py pouw SERVE_DIR WINDOW DIR [--gate CMD] [--key-sha512 HEX] [--main FILE] [--python PY]
                                [--statements FILE] [--live-evidence FILE] [--lean-answers DIR]
    served_verdicts.py pearlplus DIR --served FILE --verify FILE [--unretained FILE] [--gate CMD] [--key-sha512 HEX]
                                 [--main FILE] [--coin COIN_PUT]
    served_verdicts.py pearlplus-period open DIR --gate CMD --name P [--scheme S] [--cubin FILE | --circuit-record FILE]
    served_verdicts.py pearlplus-period close DIR --gate CMD [--lane L --windows PEARL_DIR [--wait S]]
    served_verdicts.py pearlplus-period rule DIR --gate CMD --lane L --coin COIN_RECORD
    served_verdicts.py coin DIR --record FILE [--run RUN_ID] [--run-dir DIR]
    served_verdicts.py record --serve-run DIR --window K --sampled-proofs DIR --pieces DIR --out DIR [--node NAME]
                              [--product P] [--deferred WHY] [--steps LO,HI] [--off-node]
                              [--sampled-lean [--sampled-exe PATH]] [--key-sha512 HEX] [--lane L] [--window-tie DIR]
                              [--tree SHA [--tree-on BRANCH]] [--served-compare FILE]
    served_verdicts.py witness < PATHS
    served_verdicts.py rebase SERVE_DIR SERVED_DIR < PATHS
    served_verdicts.py composed ANSWER DIR [--gate-exit RC]
    served_verdicts.py public-registration REGISTRATION
    served_verdicts.py sampled-line LOG [--partition FILE]
    served_verdicts.py front-line SESSION
    served_verdicts.py public-lines [--private FILE]
    served_verdicts.py publish OUT --private DIR

What leaves the node names no window and counts none of a window's own: no window's number, no count of the windows closed,
sealed or refused, and no engine step, request count or user of one; nor a load's tokens or span (`OWN_AT`), whose tokens
count its blocks. ``public-lines`` prints each line of stdin in its published form (`public_line`), the line in full
appended to ``--private`` (the serve dir's ``private/verdicts.txt``), and ``publish`` holds each file of OUT, once the run
is done, in full under DIR where it names or counts a window or holds a load's own field (`publish`), its public form in
its place. The class's padded counts, the period's declared T and S, the bench Build's requests and the
verifier's coins and samples are public. ``public-registration`` prints what the run publishes of a window's advice
registration (`public_registration`: the advice's length padded to its compiler's cap), ``sampled-line`` the
SAMPLED-PROOFS line from the records audit's log
(`sampled_line`: the partition the sample was over, the members it drew, and no length of the advice or count of the
scope's members), and ``front-line`` the CERTIFIER-FRONT line from the front's session (`front_line`: `FRONT_PUBLIC`).

``window``: the sealed window to audit, from a serve-openai run's ``seals.jsonl``, ``windows.jsonl`` and ``requests.jsonl``
(`pick`): sealed without error, its Commit passed, each of its requests checked and in no other window (V3b's window
circuit refuses a request carried across windows), at most ``--slots`` of them (its class's); the most requests, then the
earliest. ``--window K`` holds K to the same rules. Prints the pick (`pick`): ``{window, requests, users, steps,
refused}``, which the run keeps in the serve dir's ``private/window.json``.

``pieces``: each subprotocol after sampled proofs (`PIECES`: PoUW, PoUS, the bandwidth audit, the network certifier) runs
its piece where one is named (``POUW_CMD``, ``POUS_CMD``, ``BANDWIDTH_CMD``, ``CERTIFIER_CMD``: a command run in the tree as
``bash -c CMD _ SERVE_DIR WINDOW DIR``, which writes its core answer to ``DIR/answer.json``), one at a time, each within
``<COMPONENT>_TIMEOUT`` (else ``PIECE_TIMEOUT``, default `PIECE_TIMEOUT`), else prints a ``MISSING`` line naming it, the
command it lacks and its owner. PoUS's piece runs before the bandwidth audit's: the bandwidth session is over
the store PoUS's piece holds, and that piece writes its answer to the bandwidth piece's ``DIR``, which ``BANDWIDTH_CMD``
collects. Each run leaves ``DIR/piece.json`` (`RAN`: the node it ran on, its exit, its limit). A piece
that ran but wrote no answer is its error, unless it refused to run (``DIR/refused.txt``, `REFUSED`, naming why), which
leaves it missing; no verdict is made for a piece that didn't run. A development piece writes ``DIR/development.json``
(`DEVELOPMENT`) in place of a core answer: its verdict is printed as a development run's, and it credits nothing. A piece
that audited another server than the node it ran on says so in ``DIR/served.json`` (`SERVED`: ``{server, source}``,
``source`` ``fixture`` for an answer brought from a recorded run); one that names no server there leaves its answer about
the node it ran on. ``--only`` and ``--skip`` (a component each, repeatable) run those pieces alone, or all but those.
``--product`` (default onsite; served_verdicts.sh passes its PRODUCT) is the product the run composes (`block_record.PRODUCTS`): onsite's
every subprotocol, remote's sampled proofs and PoUW alone. A piece the product drops isn't run, whatever its command,
and prints a ``DROPPED`` line naming it.

``gate``: before serving with H1's coins from infra's gate (``--gate``, default ``H1_GATE``): refused while the key's lane
(``--lane``) has a period open, which a reserve would join, and with ``--root`` (a window's run root a session of the run
opens on, V3c's audit's) once an open in the ledger names that root, since the gate opens a root once, ever; prints one
``GATE`` line with the gate's latest pin and, with ``--main``, whether that pin is on the verifier's first-parent line of
main (a session pinned off it is refused).

``period open``: the period the served windows join, opened once serve-openai listens and before its first window
reserves (a window opens at its first engine step): its circuit is the Program digest serve-openai declared and wrote
(serve.json's ``protocols.program``, ``h1/window-program.meta.json``), its rows and rounds the server's T and S, and its
circuit files those of ``--circuits`` (``pouw_h1_circuits --program SERVE_DIR/h1 --out DIR --live T,S``'s manifest),
held to that digest, T and S (`period_open_argv`). The live template's file is required. A stratum whose ``k<K>`` file
isn't built yet is left out and named: core's credit refuses a sampled statement at that K (``circuit``), nothing else.
Its fixed length is ``--slots`` requests of ``--max-tokens`` (`credit.LENGTH_FIELD`); its sample K ``--draw-k``, else
`pouw_h1_challenger.draw_k` over the declared layout; its weights the auditor's registration of ``--model`` under the
digest, else none (the gate refuses a root its registry lacks, and serve-openai's own is a local key's, so core's credit
then refuses the window at its weights). The open's answer goes
to ``SERVE_DIR/period.json``. ``period close`` closes that period and adds its answer there.

``program``: serve-openai's declared window Program for ``--model`` at T, S, built on a CPU without serving (`program`,
`DECLARED`) and written as serve-openai writes ``h1/``, so the circuit files can be built before the serve that needs
them (the Program is a function of the linears, T and S, and of the pouw settings ``unit`` and ``ks`` in
``--protocols-config``, which must be serve-openai's own). ``--like`` takes model, T, S and that config from a
serve-openai run and refuses unless the digest is that run's.

``pouw``: PoUW's development piece (``POUW_CMD``): window K's H1 outcome (``windows.jsonl``'s ``pouw_h1``) and record
(``h1/w<K>/pouw-h1-calls.json``); with the auditor host's challenger's coins, the window's session and its sample's
session (the record's ``draw_session``, ``--draw-dir``) as the gate's ``fetch`` gives them, the auditor's ledger and pins
log as the gate answers now and main's first-parent history (``--main``),
checked by `pouw_h1_gpu.check_challenger` under the key the deployment pins (``--key-sha512``). Its verdict is accept when
H1's tables are tied to the window's Commit, its coins are the session the ledger opened and ended for it, and its record
is one manifest under them; core's credit refuses the window and is shown beside it (`DEV_WHY`). Credit reads the
declared Program serve-openai wrote (``h1/window-program.json.gz``), the period ``period open`` opened, and, when given,
the sampled calls' statements (``--statements``) and the window's live evidence (``--live-evidence``,
``h1_prove/live.py evidence``'s), so it reaches each check those reach (`credit_inputs`); with ``--lean-answers`` its
verifier runs take the answers another step kept there by key (`answers.Answers`). The verdict says whether the
window meets the two premises of `security`'s that no check here discharges, ``AcceptedLive`` and ``SaltIsLiveCoin``
(`premises`): each ``{met, why}``, met only where this run's checks show it.

``pearlplus``: PoUW's development piece under POUW=pearlplus (`served_pearlplus.sh`, POUW_CMD's default then), from Pearl+'s
own served run on the card after the window's serve: ``--served`` (`served_passes.py serve`'s document: every pass retained)
and ``--verify`` (its ``verify``'s: Pearl+'s verdict over all the passes, the reference replay of the tiles it draws).
Its verdict is accept when that run served within its caps across at least two pass boundaries, Pearl+'s verdict
accepts, and each pass's salt is the live session's that the serve names (`pearlplus_coins`: the session's tree as the gate
``--gate`` fetches it, the ledger and pins log that gate answers now and ``--main``'s line, held by `core_check.live_salt`
under the key ``--key-sha512`` pins), and a challenger draw (the coin record's ``draw``, `verify_run`'s "The coin record")
is its draw session's (`pearlplus_draw`: that session fetched the same way, held to the signed ledger and session); a
run whose coins are an in-process challenger's claims no verdict where nothing
failed (`UNCLAIMED`, `PEARLPLUS_WHY_IN_PROCESS`), and one served under the gate whose coins are not live rejects. Its
document carries the passes, that verdict's line and credit at Pearl's Assumption 1, where its coins came from
(`PEARLPLUS_COINS`) and its shares (`served_shares.pearlplus_shares`), and says why it credits nothing (`PEARLPLUS_WHY`:
its passes are not the window sampled proofs audited; under PEARLPLUS_TIE, ``--tie W``, the window's own epoch, the
Pearl-C period its live salt session joined, `pearlplus_why_tied`). ``record`` composes an accepted one as PoUW's verdict of record
only where `pearlplus_of_record` holds it to live coins under ``--key-sha512``, the lane ``--lane``'s own reserve and
Pearl-C period, a draw session the lane reserved first after that period closed and opened on the run's commitment, the
window's registration chained after Pearl+'s window record, that period's circuit the window-circuit record V3c held the
window's circuit to, and the Lean `PouwCheck.PearlPlus` verdict on the same run; else its row stays missing, naming the
first condition it fails.

``pearlplus-period open``: the lane's Pearl-C period P, opened at ``--gate`` before Pearl+'s serve with cap 1 (the serve's
one salt session joins it) and circuit the SHA-512 of ``DIR/pearlplus-descriptor.json``: with ``--circuit-record``, the
served window's circuit record (served_verdicts.sh's PEARLPLUS-CIRCUIT, the circuit cut at Pearl+'s linears that step 3's
window-circuit writes), byte for byte; without it, `pearlplus_descriptor` (Pearl+'s protocol, ``--scheme`` and
``--cubin``'s SHA-512), which names no model. Its answer goes in ``DIR/period.json``; ``pearlplus-period close`` closes it
once the serve's session has ended (the untied piece's right before the verifier's draw), and the draw, which joins no
period, must be the lane's first reserve after the close. With ``--lane`` and ``--windows`` (the serve dir's ``pearl/``,
under PEARLPLUS_TIE) the close first reads the ledger, with ``--wait S`` for up to S seconds until it ends each window's
salt session, and records whether the lane's reserves since the open are the windows' salt sessions alone
(`pearlplus_period_close`), its line saying held or REFUSED; it closes either way, and the tied piece draws only after a
period that closed with its reserves held. ``pearlplus-period rule``, once the coin is drawn, prints the
``RULE`` line: whether the coin record ``--coin``'s draw session is lane ``--lane``'s first reserve after the close
(`pearlplus_period_rule`; exit 1 when BROKEN). ``coin``: the verifier's coin record published the moment it is
drawn (`coin`), ``--run-dir`` naming the run's Attempt as its custody where no put preserves it.

``record``: sampled proofs' answer from the records audit kept under ``--sampled-proofs`` (`prove_path/run.sh`'s with
``PP_RECORDS=1``, or V3c's: ``summary.json``, ``window/audit.json``, and ``window/registration.json``, the registration it
sampled against), read as `served_integrity.answer` reads it (Pearl+'s window tie from the same files under
``--window-tie DIR`` where it is given, the window tied to W); each other subprotocol's from
``--pieces/<component>/answer.json`` (`piece_row`), else a ``missing`` row naming why. Each answer is about the server
`piece_server` names: sampled proofs' the record's node, a piece's the one its ``served.json`` names, else the node it ran
on, and the wipe's the one its ``served.json`` names. They go into the window report (``verdicts.json``, `row_of`'s rows,
with each one's server) and from it the block record (``block-record.json``, `block_record.from_period`, held by
`block_record.check`), which composes only the answers about the record's node and carries the others as ``elsewhere``;
an answer that names no server isn't carried. An accepted answer whose outputs are not its leg's certificate
(`block_record.leg`) is its subprotocol's error. `block_record` carries only an accepted subprotocol's certificate, so
where one about this node rejects or errs the composed verdict is ``reject`` and no block record is written. Otherwise
the verdict is the block record's, which is the Lean ``onsite check`` or ``remote check``'s answer; no Lean check runs
here, so it is ``undecided``, which the window report (``-window/v1``) and the ``RECORD`` line name ``pending``.
The subprotocols are those of ``--product`` (default onsite; `block_record.PRODUCTS`): each one the
product drops is ``dropped`` in the report (`dropped_row`, whatever its piece left), not run and in no block record's
``missing``, so remote's is ``pending`` with nothing missing when sampled proofs and PoUW accepted here, whatever the
onsite pieces. The report's ``composed`` gives that verdict with its reason (`said`), one clause
a subprotocol: its verdict and, unless it accepted here, why; an accept here with what it holds beyond its verdict
(`terms`: when its weights were held and their share of the checkpoint, where its profile says, and each δ of the block
record's at which its quantity is not established); PoUW's development piece reads ``development run, credits nothing``
with core's refusal of the window's credit. Prints one ``RECORD`` line, naming the product, each subprotocol's verdict
(the development piece's `DEV_VERDICT`), each missing one and why, each dropped one, each rejecting one, each one
elsewhere, and the reason. The report and the line say where they were composed (``composed_on``: the host, whether it
is the record's node, and what it read, `COMPOSED_FROM`, or off that node `COPIED_FROM`), and with ``--deferred`` what
the run left to a follow-on and why (``deferred``: `served_verdicts.sh`'s step 6 on a node short of V3c's audit's
memory, or its compose hand-off), which composes the final record on another host, the record's node still ``--node``,
and with ``--served-compare`` the served compare's verdict line (``served_compare``, `served_compare_line`).
``--steps LO,HI`` gives the window's engine steps, which the record otherwise picks from the serve dir's seals, windows
and requests (`pick`): a compose follow-on holds no requests file. ``--off-node`` composes from the served node's compose
hand-off (``composed_on``'s ``from`` `HANDED_FROM`), and is refused on the record's node itself. ``--sampled-lean``
takes sampled proofs' row from this host's own run of its Lean verifier on the records audit's v2 case, its proofs and
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
``onsite-written``: each of onsite's own subprotocols with the verdict of the answer its piece wrote beside its case in the
serve dir's private/ or the run's pieces (`onsite_written`), ``none`` where no piece wrote one, on one line, which
`served_verdicts.sh` prints before it defers an onsite run's audit.
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
import socket
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable, Mapping, Sequence
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
#: sampled proofs' owner, its row's
INTEGRITY_OWNER = "compute-accounting"

#: each subprotocol after sampled proofs, by its runner name: (its name, its component, its verifier's owner, what its piece
#: does), in the chain's order
PIECES: dict[str, tuple[str, str, str, str]] = {
    "work": ("PoUW", "pouw", "compute-accounting",
             ("H1's credit on the served window, its coins from vy-challenger-1's gate (H1_GATE), or with POUW=pearlplus "
              "the reference replay of Pearl+'s own served run (served_pearlplus.sh)")),
    "space": ("PoUS", "pous", "memory-accounting",
              ("PoUS over the served model's int8 weights on the serving card, against the memory challenger on "
               "vy-auditor-2; served_verdicts.sh sets it where the tree has benchmarks/pous/served_piece.sh and this node "
               "the challenger's key")),
    "bandwidth": ("the bandwidth audit", "bandwidth", "memory-accounting",
                  ("PoUB-P2's session over the store the PoUS piece holds, which that piece runs (POUS_CMD, "
                   "benchmarks/pous/served_piece.py) and BANDWIDTH_CMD collects (served_bandwidth.py collect)")),
    "communication": ("the network certifier", "certifier", "network-accounting",
                      "the network certifier in front of the serving endpoint"),
}
#: the record's subprotocol outside this chain: serve() runs its piece (WIPE_CMD) on the card before its engine loads
WIPE = ("the wipe", "wipe", "memory-accounting", "the wipe of the serving card before its engine loads")
NAMES = {"integrity": "sampled proofs", **{k: v[0] for k, v in PIECES.items()}, "wipe": WIPE[0]}
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
#: why a development piece's verdict credits nothing, so its subprotocol's row in the record stays missing
DEV_WHY = ("development run, credits nothing: the record takes no H1 piece as PoUW's verdict of record, and core's credit "
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
#: why the Pearl+ piece's verdict (`pearlplus`) credits nothing, its epoch's salt coin a live session's
PEARLPLUS_WHY = (f"{_PEARLPLUS_UNTIED}; its epoch's salt coin is a live session's on vy-challenger-1; and an ACCEPT is the "
                 "reference replay's (verify_run.py), not PouwCheck's ZK proof")
#: the same where its epoch's coins are an in-process challenger's: the piece then claims no verdict (`UNCLAIMED`)
PEARLPLUS_WHY_IN_PROCESS = f"{_PEARLPLUS_UNTIED}; {_IN_PROCESS_WHY}"
#: the same under PEARLPLUS_TIE (a live session's: `pearlplus_why_tied`)
PEARLPLUS_WHY_TIED_IN_PROCESS = f"{_PEARLPLUS_TIED}; {_IN_PROCESS_WHY}"
#: the Pearl+ piece's verdict where its epoch's coins are an in-process challenger's and nothing failed
UNCLAIMED = "unclaimed"
#: where Pearl+'s epoch coins and the replay's draws come from (`pearlplus`), its epoch's salt coin a live session's
PEARLPLUS_COINS = {"epoch": "a salt-only session on vy-challenger-1, reserved through the lane's gate before the engine "
                            "started and opened by the install on its salt epoch's root; each pass's salt is held to it by "
                            "core_check.live_salt through the auditor's signed ledger, the host's pins log and main's "
                            "first-parent line (`pearlplus_coins`)",
                   "draws": "a draw session of its own on vy-challenger-1 (verify_run.py's \"The run's draw\"), reserved "
                            "withheld through the lane's gate once its Pearl-C period closed and opened on the run's "
                            "commitment once every pass's roots are read, its draw coin the run's one source, recorded in "
                            "the passes' coin.json before any replay and held to the session through the auditor's signed "
                            "ledger (`pearlplus_draw`); the piece puts that record in the evidence store the moment it is "
                            "drawn and holds the verdict to it (`coin`, `pearlplus_coin`)"}
#: the replay's draws where verify ran without the gate (`verify_run.node_draw`)
PEARLPLUS_NODE_DRAWS = ("the Lean verifier's IO.getRandomBytes in verify_run.py on the serving node, one source for the "
                        "run, taken once every pass's roots are read and recorded in the passes' coin.json before any "
                        "replay; the piece puts that record in the evidence store the moment it is drawn and holds the "
                        "verdict to it (`coin`, `pearlplus_coin`)")
#: the same where its epoch's coins are an in-process challenger's
PEARLPLUS_COINS_IN_PROCESS = {"draws": PEARLPLUS_NODE_DRAWS,
                              "epoch": "an in-process challenger of epoch 0 (verity_bench's protocols_config, "
                                       "pouw.challenged), held by the serving process"}
#: where the epoch's salt coin comes from under PEARLPLUS_TIE (``epoch``; ``draws`` as above)
PEARLPLUS_EPOCH_TIED = ("a salt-only session on vy-challenger-1 for the window, reserved through the lane's gate as the "
                        "window opened (serve_pearl.live) and opened by the window's install on its salt epoch's root; each "
                        "pass's salt is held to it by core_check.live_salt through the auditor's signed ledger, the host's "
                        "pins log and main's first-parent line (`pearlplus_coins`)")
PEARLPLUS_EPOCH_TIED_IN_PROCESS = ("an in-process challenger of the window's epoch over the Lean verifier of record's "
                                   "entropy (serve_pearl.in_process), held by the serving process")
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
#: beside the window registration R under ``--window-tie``: the window-circuit record V3c held R's circuit to
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
#: what `record --off-node` composes from: the served node's compose hand-off (`served_verdicts.sh handoff`), no witness
HANDED_FROM = ("the served node's compose hand-off, held here to the sums it wrote, never a witness: each row's answer from "
               "where its answer_from says, sampled proofs' recomputed here by its Lean verifier from the case's proofs "
               "where the hand-off carries them, and each other's answer file as the served node wrote it, not the party "
               "that produced it: an edit to such a file on the node before its hand-off changes its row")
#: where a row's answer came from (``answer_from``: the report's row, and the ``RECORD`` line's ``answers``)
AS_WRITTEN = "as the served node wrote it"
#: ... sampled proofs' under ``record --sampled-lean`` (`lean_sampled_row`)
LEAN_HERE = ("recomputed on this host by the Lean verifier (vstar-recursive sampled check) from the proofs and public "
             "files of the window's sampled proofs case, not the served node's answer")
#: ... PoUW's where Pearl+'s piece is its verdict of record (`pearlplus_of_record`)
PEARLPLUS_FROM = (f"{AS_WRITTEN} (its piece files in pouw/): Pearl+'s Lean verdict of record (PouwCheck.PearlPlus), "
                  "the prover side's, run on the served node over the retained passes; no auditor host re-checks it, since "
                  "its case (the passes' manifests with their salt key, and the opened tiles' rows) is witness")
#: what the composed verdict's reason (`said`) says of that row's accept
PEARLPLUS_SAID = "Pearl+'s Lean verdict of record, the prover side's, which no auditor host re-checks"
#: ... the files the served node wrote that a row's answer is otherwise read from, by runner name (a piece's: its dir)
WRITTEN = {"integrity": "the records audit's summary.json and window/audit.json"}
#: A served run's witness, by a path's part (`witness`): what a compose hand-off never carries, each with why
WITNESS_DIRS = {"values": "a committed value file", "prover": "a prover's input file (its rows and salts)",
                "served_rows": "a window's committed rows and salts", "certifier-front": "the certifier front's group (its prompts)"}
#: ... by the serve dir's top directory
WITNESS_TOPS = {"windows": "a sealed row dir (the window's rows)", "advice": "the advice and its opening"}
#: ... by a file's name (`fnmatch`)
WITNESS_NAMES = {"inst*.bin": "a prover's instance file (its rows)", "*-private.bin": "a private file (prover only)",
                 "*salt*.bin": "a commitment's salts (prover only)", "*.key": "a key", "requests.jsonl": "the requests' prompt and served ids, rows and seeds",
                 "request_users.jsonl": "the requests' users", "step_trace.jsonl": "the step trace (each request's rows)",
                 "advice_trace*.jsonl": "the advice trace (each request's rows)", "load*.jsonl": "the load's requests",
                 "tokens*.json": "the requests' token ids", "match_stoch.json": "the sampler's rows",
                 "sampled_replay*.json": "the replay's executed prefixes", "*.npy": "a value array",
                 "*.npz": "a value array", "*.pt": "a tensor", "*.safetensors": "a checkpoint's weights"}
#: a development piece's verdict in the ``RECORD`` line, in place of its row's ``missing``
DEV_VERDICT = "development run, credits nothing"
#: Pearl+'s verdict's protocol (`verify_run.PROTOCOL`)
PEARLPLUS_PROTOCOL = "verity/pouw/pearl-plus/v1"
#: Pearl+'s verdict's core answer in its piece (`served_pearlplus.sh`'s step 4), and the Lean `PouwCheck.PearlPlus`
#: verdict on the same passes and coin record there (`pouw-pearlplus`'s printout)
PEARLPLUS_ANSWER, PEARLPLUS_LEAN = "pearlplus-answer.json", "pearlplus-lean.json"
#: the window's registration R as its records audit under ``--sampled-proofs`` sampled against (a5's copy), and that
#: audit's summary (`spot_row`)
WINDOW_REGISTRATION, AUDIT_SUMMARY = "window/registration.json", "summary.json"
#: why `pearlplus_live_draw` refuses a node's draw source
PEARLPLUS_NODE_DRAW = ("the verifier's draw source is drawn on the serving node (verify_run.node_draw, PEARLPLUS_NODE_DRAWS), "
                       "not a challenger coin opened on the run commitment (served_passes.py verify under "
                       "PEARLC_CHALLENGER_GATE)")
#: H1's record of a window (`pouw_h1.CALLS_FILE`) under the serve dir
H1_CALLS = "h1/w{k}/pouw-h1-calls.json"
#: serve-openai's declared window Program's meta under the serve dir (`pouw_h1.PROGRAM_META`)
H1_META = "h1/window-program.meta.json"
#: `pouw_h1_circuits`' manifest in its ``--out``
CIRCUITS_MANIFEST = "circuit-files.json"
#: the period this serve dir's windows joined, its open's and close's answers (``period``)
PERIOD = "period.json"
#: the int8 linears serve-openai declares its window Program over, by served model: its layers and each layer's
#: ``(module, n, K)`` in vLLM's module order (serve-openai's H1 arm: each ``weight.shape``, named by `pouw._named`)
DECLARED: dict[str, tuple[int, tuple[tuple[str, int, int], ...]]] = {
    "verity-int8/Llama-3.2-1B-INT8": (16, (("self_attn.qkv_proj", 3072, 2048), ("self_attn.o_proj", 2048, 2048),
                                           ("mlp.gate_up_proj", 16384, 2048), ("mlp.down_proj", 2048, 8192))),
}


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


def pick(serve_dir: Path, slots: int, want: int | None = None) -> dict[str, Any]:
    """The window to audit (module docstring); ValueError when none qualifies, or ``want`` doesn't."""
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
    if not ok:
        raise ValueError(f"no sealed window to audit: {json.dumps(refused, sort_keys=True)}")
    k, mine = max(ok, key=lambda x: (len(x[1]), -x[0]))
    lo = sum(int((w.get("commit") or {}).get("steps") or 0) for j, w in lines.items() if j < k)
    hi = lo + int((lines[k].get("commit") or {}).get("steps") or 0)
    return {"window": k, "requests": len(mine), "users": sorted(r.get("user") or "" for r in mine), "steps": [lo, hi],
            "refused": {str(j): why for j, why in refused.items()}}


#: A served window's own fields, never published: the windows (closed, sealed or joined), its users, its engine steps and
#: whether its requests filled the slots, its unpadded population, its advice's length and its windows' advice by number.
OWN = frozenset({"windows", "users", "engine_steps", "every_slot", "population", "advice_bits", "bits_by_window"})
#: Fields that name a window or count its own where their value holds a number: the window's number, its steps, its
#: requests' count, the windows refused. A window named ``all``, the bench Build's requests by name and a refusal's reason
#: stay.
NUMBERED = frozenset({"window", "steps", "requests", "refused"})
#: A load's own fields, held by path as (parent, key), the parent the nearest key above the field through lists: the
#: tokens of each load of `served_overhead.py metrics`' aggregate, which over a block's outputs are the exact block
#: count, and its span. Held by path, not as bare keys, so the inference-only policy's ``n_inputs.span_s`` stays as
#: written, and with it the policy digest its verdict cites.
OWN_AT = frozenset({("loads", "tokens"), ("loads", "span_s")})
#: A window's number, step range or count in text, each with what the published text says in its place
WINDOW_TEXT = (
    (re.compile(r'\\?"(?:%s)\\?": ?(?:-?\d+(?:\.\d+)?|\[[^\]]*\]|\{[^{}]*\}|true|false)(?:, ?)?'
                % "|".join(sorted(OWN | NUMBERED))), ""),
    (re.compile(r"(?<![A-Za-z0-9_])w\d+(?![A-Za-z0-9_])"), "wK"),
    (re.compile(r"(?i)\b(windows?)([ :=#]+)\d+(?!\d|[,.]\d)"), r"\1\2K"),
    (re.compile(r"\b\d+ ((?:sealed |closed )?window\(s\))"), r"\1"),
    (re.compile(r"\b\d+ (sealed|closed)\b"), r"\1"),
    (re.compile(r"(?i)\b(steps?) \[\d+, ?\d+[\])]"), r"\1"),
    (re.compile(r"\b\d+/\d+ ok\b"), "ok"),
)
#: A window's number in a file's path under the published dir: the file is held whole
NAMED = re.compile(r"(?<![A-Za-z0-9])w\d+(?![A-Za-z0-9])")
_OPEN = re.compile(r'\{\s*"')
_FIELD = re.compile(r'\\?"(?:%s)\\?":' % "|".join(sorted(OWN | NUMBERED)))
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
#: The files under the published dir that `publish` leaves as they are, unread: the serve dir's declared Program, the
#: window's class Program, which is public structure. Its JSON runs to hundreds of MB, beyond what `PUBLISH_MAX` lets
#: `publish` read.
PUBLIC_FILES = frozenset({"serve/h1/window-program.json.gz"})


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


def _held(key: Any, value: Any, parent: Any = None) -> bool:
    if (parent, key) in OWN_AT:
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
    """``s`` with each window number, step range and window count `WINDOW_TEXT` finds in its published form, and no load's
    own field (`OWN_AT`)."""
    for pat, sub in WINDOW_TEXT:
        s = pat.sub(sub, s)
    return _at_text(s)


def public_json(o: Any, parent: Any = None) -> Any:
    """``o`` with no field of a window's own (`OWN`, `NUMBERED`) or of a load's (`OWN_AT`, by ``parent``, the key ``o``
    is under) and each string in its `public_text` form."""
    if isinstance(o, Mapping):
        return {public_text(k) if isinstance(k, str) else k: public_json(v, k) for k, v in o.items() if not _held(k, v, parent)}
    if isinstance(o, list):
        return [public_json(v, parent) for v in o]
    return public_text(o) if isinstance(o, str) else o


def public_line(line: str) -> str:
    """A verdict line as the run publishes it: each JSON object in it in its `public_json` form, one cut short that names a
    window's or a load's own field replaced by `CUT` to the line's end, and the rest in its `public_text` form."""
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
    """Every file under ``out``, the dir that leaves the node, in its public form: one whose path names a window, or that
    `PUBLISH_MAX` keeps from being read, held whole, and one whose text names or counts a window or holds a load's own
    field (`public_doc`) held in full, under ``private`` at its path under ``out``, its public form left in its place. A
    file that isn't UTF-8 text is held whole. Only `PUBLIC_FILES` stay as they are, unread. The paths held, in order."""
    held: list[str] = []
    for p in sorted(out.rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        rel = p.relative_to(out).as_posix()
        if rel in PUBLIC_FILES:
            continue
        dst = private / rel
        private.mkdir(mode=0o700, parents=True, exist_ok=True)
        dst.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        data = p.read_bytes() if p.stat().st_size <= PUBLISH_MAX else None
        if data is None or NAMED.search(rel):
            shutil.move(p, dst)
            held.append(rel)
            continue
        text = data.decode(errors="replace")
        if text.encode() != data:
            shutil.move(p, dst)
            held.append(rel)
            continue
        pub = public_doc(text)
        if pub != text:
            dst.write_bytes(data)
            p.write_text(pub)
            held.append(rel)
    return held


#: The advice registration's fields that state the advice's own length, which `public_registration` pads to the cap.
PADDED = ("binding.advice.bits", "rows")


def public_registration(reg: Mapping[str, Any]) -> dict[str, Any]:
    """What the run publishes of a window's advice registration: the advice's length (`PADDED`) at its compiler's
    ``cap_bits``, the one length the class declares (its rows at ``row_bits`` each), no count or refusal of its requests,
    no window's number (`public_json`), and the rest as registered. ``padded`` names the fields padded."""
    cap, row_bits = int(reg["compiler"]["cap_bits"]), int(reg["row_bits"])
    out = {k: v for k, v in reg.items() if k not in ("requests", "compiled", "refused")}
    out["binding"] = {**reg["binding"], "advice": {**reg["binding"]["advice"], "bits": cap}}
    out["rows"] = -(-cap // row_bits)
    return public_json({**out, "padded": list(PADDED)})


def sampled_line(records: Mapping[str, Any], partition: Mapping[str, Any] | None) -> str:
    """The SAMPLED-PROOFS line's account of the records audit's ``RECORDS`` row: what the sample was over (the partition's
    templates, its scope's op path and law), which members it drew, and the row with no length of the advice
    (``leak.advice_bits``) and no ``population``, the scope's unpadded member count, which counts the window's tokens. The
    scope's ``steps``, the window's engine steps, stay on the node too."""
    row = {k: v for k, v in records.items() if k != "population"}
    if isinstance(row.get("leak"), Mapping):
        row["leak"] = {k: v for k, v in row["leak"].items() if k != "advice_bits"}
    part = partition or {}
    scope = part.get("scope") or {}
    over = (f"over {', '.join(part.get('templates') or ['?'])} at {scope.get('op_path_prefix', '?')} "
            f"(law {part.get('law', '?')})")
    drawn = f"drawn members {row.get('draw')}" if row.get("draw") is not None else "no draw"
    return f"{over}, {drawn}: RECORDS {json.dumps(row, sort_keys=True)}"


#: The certifier front's session fields the CERTIFIER-FRONT line says. The rest (its prompts, request, kept fetch, gate
#: and verdict's cells) count or name the group's requests, and stay in the serve dir.
FRONT_PUBLIC = ("rc", "run", "ended", "ended_utc", "session_s", "pinned", "leaked_removed")


def front_line(session: Mapping[str, Any]) -> str:
    """The CERTIFIER-FRONT line's account of the front's ``session.json``: `FRONT_PUBLIC` alone, or that it refused."""
    if "refused" in session:
        return "refused (its reason in the serve dir's private/certifier-front/session.json)"
    return json.dumps({k: session[k] for k in FRONT_PUBLIC if k in session})


def _unset(comp: str, what: str) -> str:
    return f"{comp.upper()}_CMD unset: {what}"


def dropped_row(key: str, product: str) -> dict[str, Any]:
    """The row of a subprotocol ``product`` drops (`block_record.PRODUCTS`, its legs being the components): not run."""
    from verity_e2e_demo import block_record as BR

    by_leg = {"integrity": NAMES["integrity"], **{v[1]: v[0] for v in PIECES.values()}, "wipe": WIPE[0]}
    return {"verdict": "dropped", "owner": PIECES.get(key, WIPE)[2],
            "why": f"not run: the {product} product composes {' and '.join(by_leg[x] for x in BR.PRODUCTS[product])} alone"}


def pieces(serve_dir: Path, window: int, out: Path, env: Mapping[str, str] | None = None,
           node: str | None = None, only: Sequence[str] | None = None, skip: Sequence[str] = (),
           product: str = "onsite") -> dict[str, dict[str, Any]]:
    """Each piece of `PIECES` run where ``<COMPONENT>_CMD`` names it, under `timeout` at its limit, else named missing;
    one line each on stdout, its verdict the one `piece` gives the record. ``only`` and ``skip`` name components: those
    alone, or all but those, so one piece can run beside the others (`served_verdicts.sh` starts the certifier's early).
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
    keep pace, the card's seconds, its node and the passes it finished (each None without one, or why it has none)."""
    if doc.get("piece") == "pearlplus":
        shares = doc.get("shares") or {}
        at = (shares.get("readings") or {}).get("tensor") or {}
        un, rp = shares.get("unretained"), shares.get("replay")
        credit, published = doc.get("credit"), doc.get("coin")
        check = (doc.get("coins") or {}).get("check") or {}
        return {"verdict": doc.get("verdict"), "development": True, "credited": False, "reasons": doc.get("reasons"),
                "coin_source": doc.get("coin_source"), "session": check.get("session"),
                "coins": check.get("reason", "unchecked"),
                **{k: doc.get(k) for k in ("scheme", "load", "passes", "dropped", "served_tokens", "generate_s", "tie")},
                "credit": None if credit is None else {k: credit.get(k) for k in ("credited", "macs", "epsilon")},
                "coin": None if published is None else {k: published.get(k) for k in ("art", "preserved", "error", "sha512")},
                "give_up": doc.get("give_up"),
                "shares_at_tensor_clock": {k: at.get(k) for k in ("declared_linears", "credited")},
                "unretained": None if un is None else un.get("error") or {
                    "generate_s": un.get("wall_s"),
                    "declared_linears_at_tensor_clock": ((un.get("readings") or {}).get("tensor") or {}).get("declared_linears")},
                "replay": None if rp is None else rp.get("error") or {
                    k: rp.get(k) for k in ("s", "cpus", "cores_to_keep_pace", "device_s", "node", "finished", "of")}}
    check, credit = doc.get("check") or {}, doc.get("credit") or {}
    return {"verdict": doc.get("verdict"), "development": True, "credited": False, "reasons": doc.get("reasons"),
            "coin_source": doc.get("coin_source"), "session": (doc.get("session") or {}).get("session"),
            "tied": doc.get("tied"), "coins": check.get("reason", "unchecked"), "manifest": check.get("manifest", "unchecked"),
            "gate_passed": doc.get("gate_passed"), "credit": credit.get("window"),
            **{p: doc.get(p) for p in PREMISES}}


def _gate_out(gate: str, *words: str) -> bytes:
    """The gate's answer to `words` through its command `gate`, the words one shell-quoted line as `commit_coins.ChallengerGate`
    sends them; RuntimeError naming what it said when it refuses."""
    p = subprocess.run([*shlex.split(gate), shlex.join(words)], capture_output=True, check=False)
    if p.returncode:
        said = (p.stdout + p.stderr).decode(errors="replace").strip()
        raise RuntimeError(f"the gate refused {words[0]} (exit {p.returncode}): {said[-400:]}")
    return p.stdout


def gate_ready(gate: str, lane: str, main: Path | None = None, root: str | None = None) -> tuple[bool, str]:
    """Whether this run may reserve its windows' sessions at `gate` (the ``gate`` subcommand), and its ``GATE`` line; with
    `root`, the run root a session of this run opens on, also refused once an open in the ledger names it: the gate opens
    a root once, ever."""
    from verity.protocols.spot.device import challenger as CH
    from verity_vllm.protocols.pouw.h1 import window as PHW

    ledger = _gate_out(gate, "ledger").decode().splitlines()
    pins = [json.loads(x) for x in _gate_out(gate, "pins").decode().splitlines() if x.strip()]
    pin, period = (pins[-1] if pins else {}), PHW.open_period(ledger, lane)
    held = None if root is None else next((e["session"] for e in map(json.loads, filter(str.strip, ledger))
                                           if e["kind"] == "open" and CH.root_of(e) == root.lower()), None)
    on = None if main is None else pin.get("commit") in main.read_text().split()
    where = "" if on is None else f", {'on' if on else 'NOT on'} the verifier's first-parent line of main"
    rooted = "" if root is None else (f"; root {root[:16]}… " + (f"already has its session {held} (one session a root, ever)"
                                                                  if held else "has no session yet"))
    line = (f"GATE {'refused' if period or held else 'ready'}: lane {lane}'s open period "
            f"{period and period.get('period')}{' (a reserve would join it)' if period else ''}{rooted}; the ledger's "
            f"{len(ledger)} entries; pinned {str(pin.get('commit'))[:12]} (main {str(pin.get('main'))[:12]}, "
            f"{pin.get('pinned_utc')}){where}")
    return period is None and held is None, line


def program(out: Path, model: str, rows: int, steps: int, like: Path | None = None,
            config: str | None = None) -> dict[str, Any]:
    """``program``: the window Program serve-openai declares for `model` at T `rows` and S `steps` under `config`, its
    ``--protocols-config`` (`serve_h1.config_of`'s, a JSON object or file), built without serving (`DECLARED`'s linears
    at the pouw setting ``ks``'s K's, over the setting ``unit``'s tile unit, `served.unit_of`) and written as
    serve-openai writes it (`served.write_program`) into `out`; its meta. With `like`, a serve-openai run's dir, its
    model, T, S and config are that run's (serve.json's ``protocols``) and its digest must be the run's
    (``protocols.program``), else ValueError."""
    from verity_vllm.pipeline import serve_h1 as SH
    from verity_vllm.protocols import pouw as PW
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import served as SV

    want, cfg = None, SH.config_of(config)
    if like is not None:
        serve = json.loads((like / "serve.json").read_text())
        proto = serve.get("protocols") or {}
        model, rows, steps, want = serve.get("served_model_name") or model, proto["rows"], proto["steps"], proto["program"]
        cfg = proto.get("config") or {}
    if model not in DECLARED:
        raise ValueError(f"no declared linears for {model!r}: add them to DECLARED ({sorted(DECLARED)})")
    pw = PW.Settings.of(cfg.get("pouw", {}))
    layers, linears = DECLARED[model]
    named = [(f"model.layers.{i}.{m}", n, k) for i in range(layers) for m, n, k in linears if pw.ks is None or k in pw.ks]
    if not named:
        raise ValueError(f"{model} has no linear at K {list(pw.ks or ())} (the pouw setting 'ks'): its K's are "
                         f"{sorted({k for _m, _n, k in linears})}")
    shapes = [(n, k) for _m, n, k in named]
    modules = {w: m for w, (m, _n, _k) in enumerate(named)}
    digest = SV.write_program(SV.declared(shapes, modules, rows, steps, unit=SV.unit_of(pw.unit)),
                              SV.declared_layout(shapes, modules, rows, steps), out, rows=rows, steps=steps)
    if want is not None and digest != want:
        raise ValueError(f"{model}'s Program at T {rows}, S {steps} is {digest[:16]}..., not {like}'s {str(want)[:16]}...: "
                         "DECLARED's linears aren't the ones that run served")
    return json.loads((out / PH.PROGRAM_META).read_text())


def period_open_argv(serve_dir: Path, name: str, circuits: Path, *, slots: int, max_tokens: int,
                     draw_k: int | None = None, model: str | None = None) -> tuple[list[str], dict[str, Any]]:
    """The gate's ``period-open`` of period `name` for serve-openai's run `serve_dir` (``period open``, module docstring)
    and what it commits: ``{circuit, rows, steps, draw_k, circuit_files, missing, length, weights}``, ``missing`` the
    strata with no ``k<K>`` file yet. ValueError naming what to build or rerun when the serve dir or `circuits` doesn't
    hold the period's Program, T and S. Before serve-openai starts (no ``serve.json`` yet) the Program is ``program``'s,
    and `period_served` holds the one it then declares to it."""
    from verity_vllm.protocols.pouw.h1 import challenger as PHC

    served = serve_dir / "serve.json"
    proto = (json.loads(served.read_text()).get("protocols") or {}) if served.is_file() else None
    try:
        meta = json.loads((serve_dir / H1_META).read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"{serve_dir} has no declared window Program ({H1_META}, {type(e).__name__}): serve it with "
                         "--protocols pouw:eps1-sm120-h1fx1 from a tree whose serve_h1 writes it") from None
    circuit, rows, steps = meta["program_digest"], int(meta["rows"]), int(meta["steps"])
    if proto is not None and proto.get("program") != circuit:
        raise ValueError(f"serve.json's Program {str(proto.get('program'))[:16]}... is not {H1_META}'s {circuit[:16]}...")
    redo = (f"build them with python -m tests.protocols.pouw_h1_circuits --program {serve_dir / 'h1'} --out {circuits} "
            f"--live {rows},{steps}")
    try:
        doc = json.loads((circuits / CIRCUITS_MANIFEST).read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"no circuit files at {circuits} ({type(e).__name__}): {redo}") from None
    if doc.get("program") != circuit:
        raise ValueError(f"{circuits}'s circuit files are Program {str(doc.get('program'))[:16]}...'s, not the served "
                         f"{circuit[:16]}...: {redo}")
    lb, files = doc.get("live_bind") or {}, dict(doc.get("files") or {})
    if (lb.get("T"), lb.get("S")) != (rows, steps) or "live" not in files:
        raise ValueError(f"{circuits}'s live circuit file is for T = {lb.get('T')}, S = {lb.get('S')}, not the window's "
                         f"{rows}, {steps}: {redo}")
    strata = sorted({int(k) for _n, k, _w in meta["layout"]["linears"]})
    missing = [k for k in strata if f"k{k}" not in files]
    k = PHC.draw_k(meta["layout"]) if draw_k is None else int(draw_k)
    length = {"max_tokens": [int(max_tokens)] * int(slots), "stop": "fixed"}
    weights = ["--model", model] if model else []
    compact = {"sort_keys": True, "separators": (",", ":")}
    # One served window is two of the gate's sessions, its own and its withheld draw session, and the gate counts
    # each against the period's cap; credit reads the close's list as window, then draw (credit._period_of).
    argv = ["period-open", "--name", name, "--cap", "2", "--circuit", circuit, "--draw-k", str(k), "--rows", str(rows),
            *weights, "--circuit-files", json.dumps(files, **compact), "--length", json.dumps(length, **compact)]
    return argv, {"circuit": circuit, "rows": rows, "steps": steps, "draw_k": k, "circuit_files": files,
                  "missing": missing, "length": length, "weights": weights}


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


def period_open(serve_dir: Path, gate: str, name: str, circuits: Path, **kw: Any) -> dict[str, Any]:
    """``period open`` (module docstring): the open sent, its answer and what it committed, written to
    ``serve_dir/PERIOD``."""
    argv, doc = period_open_argv(serve_dir, name, circuits, **kw)
    doc = {"name": name, **doc, "open": _gate_json(gate, argv)}
    (serve_dir / PERIOD).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def period_served(serve_dir: Path) -> dict[str, Any]:
    """``period served``: the period opened before serve-openai started, whose Program serve-openai must then have
    declared (serve.json's ``protocols.program``); ValueError naming both when it declared another, since credit refuses
    every window of a period whose open committed another Program."""
    doc = json.loads((serve_dir / PERIOD).read_text())
    got = (json.loads((serve_dir / "serve.json").read_text()).get("protocols") or {}).get("program")
    if got != doc["circuit"]:
        raise ValueError(f"serve-openai declared Program {str(got)[:16]}..., not period {doc['name']}'s "
                         f"{doc['circuit'][:16]}... (`program`'s, built from DECLARED): its windows credit nothing")
    return doc


def period_close(serve_dir: Path, gate: str) -> dict[str, Any]:
    """``period close``: the period `period_open` opened for `serve_dir` closed, its answer added to ``PERIOD``."""
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
    ``piece_dir/PERIOD``, which `period_close` closes. The run's draw session is reserved after the close, so it joins
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
    refuses the lane's every other reserve; the tied piece draws only after one that closed with its reserves ok."""
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
    """The rule the verifier's draw keeps, read in the gate's ledger once its coin is drawn: the draw session of the coin
    record `record` (``draw.reserved.session``) is lane `lane`'s first reserve after the period in ``piece_dir/PERIOD``
    closed (`pouw_h1_window.after_close`, which `bench_arm.LiveDraw` held it to before asking its coin, and
    `pearlplus_live_draw` holds the record to). Whether it held, and its ``RULE`` line; kept as the PERIOD document's
    ``rule``."""
    from verity_vllm.protocols.pouw.h1 import window as PHW

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


def credit_inputs(serve_dir: Path, statements: Path | None = None, live: Path | None = None) -> dict[str, str]:
    """What core's credit reads of the window beyond its record and session, as ``--check-challenger``'s flags: the
    declared Program serve-openai wrote (``--descriptor``, `pouw_h1.PROGRAM`), the period ``period open`` opened
    (``--period``, `PERIOD`), the sampled calls' statements (``--statements``) and the live evidence (``--live-evidence``,
    ``h1_prove/live.py evidence``'s), each where it exists."""
    got: dict[str, str] = {}
    if (p := serve_dir / "h1" / "window-program.json.gz").is_file():
        got["--descriptor"] = str(p)
    if (p := serve_dir / PERIOD).is_file():
        got["--period"] = json.loads(p.read_text())["name"]
    for flag, p in (("--statements", statements), ("--live-evidence", live)):
        if p is not None:
            got[flag] = str(p)
    return got


def check_challenger(calls: Path, tree: Path, ledger: Path, pins: Path, main: Path | None, key: str, out: Path,
                     inputs: Mapping[str, str], python: str) -> dict[str, Any]:
    """`pouw_h1_gpu.py --check-challenger` (run by `python` in adapters/vllm, so ``tests`` is that package's) of the
    window's record `calls` against its session's tree `tree`, with credit's `inputs` (`credit_inputs`); its
    ``pouw_h1_challenger.json``."""
    out.mkdir(parents=True, exist_ok=True)
    argv = [python, "-m", "tests.protocols.pouw_h1_gpu", "--check-challenger", str(calls), str(tree), "--key-sha512", key,
            "--ledger", str(ledger), "--pins", str(pins), *(["--main", str(main)] if main else []), "--out", str(out),
            *(x for kv in inputs.items() for x in kv)]
    p = subprocess.run(argv, cwd=ROOT / "adapters" / "vllm", capture_output=True, text=True, check=False)
    (out / "check.log").write_text(p.stdout + p.stderr)
    try:
        return json.loads((out / "pouw_h1_challenger.json").read_text())
    except (OSError, ValueError) as e:
        raise RuntimeError(f"pouw_h1_gpu --check-challenger wrote no verdict (exit {p.returncode}): "
                           f"{(p.stdout + p.stderr).strip()[-400:]}") from e


Check = Callable[[Path, Path, Path, Path, "Path | None", str, Path, Mapping[str, str]], Mapping[str, Any]]

#: the premises the pouw piece's verdict says whether the window meets (`premises`)
PREMISES = ("AcceptedLive", "SaltIsLiveCoin")
#: why a window meets ``AcceptedLive`` (`Accounting.TS.Session`'s, ``accounting-ts-accepted-live``)
LIVE_MET = ("credit accepted the window, and it credits a drawn unit or a live round only where the verifier of record "
            "accepted its session (flock-verify verify --zk with the --registered file credit builds)")
#: why a window meets ``SaltIsLiveCoin`` (`Properties.Pouw.Period`'s)
SALT_MET = ("core_check.salt accepted the window's salt as its live salt coin, opened after the window's open root and read "
            "through the session the auditor's ledger opened for it, under the host's pins log and main, and the record's "
            "Program salt is that coin's (manifest_reason)")


def premises(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Whether the window of `pouw`'s document `doc` meets ``AcceptedLive`` and ``SaltIsLiveCoin``, each ``{met, why}``:
    ``AcceptedLive`` only when core's credit accepted the window (a window credit refuses credits no proof, so it
    doesn't), ``SaltIsLiveCoin`` only when the challenger check accepted the window's coins, its salt among them, and its
    record as one manifest under them."""
    check, credit = doc.get("check"), doc.get("credit")
    reasons = list(doc.get("reasons") or ())
    unchecked = next((r for r in reasons if r.startswith("coins: ")), "; ".join(reasons) or "no challenger check")
    if check is None:
        salt = False, f"the challenger check didn't run on the window: {unchecked}"
    elif check.get("reason") is not None:
        salt = False, f"the challenger check refused the window's coins, its salt among them: {check['reason']}"
    elif check.get("manifest") is not None:
        salt = False, f"the window's record isn't one manifest under its salt coin: {check['manifest']}"
    else:
        salt = True, SALT_MET
    if credit is None:
        live = False, f"credit didn't run on the window, so it credited no proof: {unchecked}"
    elif credit.get("credit") is None:
        why = credit.get("window") or credit.get("manifest") or "no credited calls"
        why = f"{why[0]}: {why[1]}" if isinstance(why, (list, tuple)) else str(why)
        live = False, f"credit refused the window ({why}), so it credited no proof"
    else:
        live = True, LIVE_MET
    return {name: {"met": met, "why": why[:600]} for name, (met, why) in zip(PREMISES, (live, salt), strict=True)}


def pouw(serve_dir: Path, window: int, out: Path, *, gate: str | None, key: str | None, main: Path | None,
         check: Check, statements: Path | None = None, live: Path | None = None,
         lean_answers: Path | None = None) -> dict[str, Any]:
    """PoUW's development piece on window `window` (module docstring), its document written to ``out/development.json``;
    `check` is `check_challenger` with its Python bound, given credit's inputs (`credit_inputs` of `serve_dir`, the sampled
    calls' `statements` and the `live` evidence), its document's ``credit_inputs``, and `lean_answers`, the verifier's
    memo of its runs' answers (``--lean-answers``, `answers.Answers`), whose counts are its ``lean_answers``."""
    out.mkdir(parents=True, exist_ok=True)
    inputs = credit_inputs(serve_dir, statements, live)
    line = next((w for w in _lines(serve_dir / "windows.jsonl") if int(w["window"]) == window), {})
    h1 = line.get("pouw_h1") or {}
    calls = serve_dir / H1_CALLS.format(k=window)
    doc: dict[str, Any] = {"window": window, "credited": False, "why": DEV_WHY, "coin_source": h1.get("coin_source"),
                           "tied": h1.get("tied"), "steps_served": h1.get("steps_served"), "session": h1.get("session"),
                           "gate_passed": (h1.get("gate") or {}).get("gate_passed"), "check": None, "credit": None,
                           "credit_inputs": inputs, "lean_answers": None}
    reasons = []
    if not h1 or "error" in h1:
        reasons.append(f"H1's window: {h1.get('error') or 'its line has no pouw_h1 (served without --protocols)'}")
    elif h1.get("tied") is not True:
        reasons.append(f"ROOTS1: H1's tables are not tied to the window's Commit ({(h1.get('roots1') or {}).get('reason')})")
    if h1 and "error" not in h1 and h1.get("coin_source") != "challenger":
        reasons.append(f"coins: {h1.get('coin_source')}, not the auditor host's challenger's, so nothing checks them")
    elif h1.get("coin_source") == "challenger":
        try:
            if not (gate and key):
                raise ValueError("no gate or no key pin to check the window's session by (H1_GATE, H1_KEY_SHA512)")
            com = json.loads(calls.read_text())["commitment"]
            sid, dsid = com["session"]["session"], (com.get("draw_session") or {}).get("session")
            tgz, tree = out / "session.tgz", out / "session"
            for s in (sid, dsid) if dsid else (sid,):
                tgz.write_bytes(_gate_out(gate, "fetch", s))
                with tarfile.open(tgz) as t:
                    t.extractall(tree, filter="data")
            (out / "ledger.jsonl").write_bytes(_gate_out(gate, "ledger"))
            (out / "pins.jsonl").write_bytes(_gate_out(gate, "pins"))
            got = check(calls, tree / sid, out / "ledger.jsonl", out / "pins.jsonl", main, key, out / "check",
                        {**inputs, **({"--draw-dir": str(tree / dsid)} if dsid else {}),
                         **({"--lean-answers": str(lean_answers)} if lean_answers is not None else {})})
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, tarfile.TarError) as e:
            reasons.append(f"coins: {type(e).__name__}: {e}"[:600])
        else:
            doc["check"] = {k: got.get(k) for k in ("ok", "reason", "manifest", "rounds", "coin_source")}
            doc["credit"] = {k: (got.get("credit") or {}).get(k) for k in ("window", "manifest", "k", "drawn", "eps", "calls", "credit")}
            doc["lean_answers"] = got.get("lean_answers")
            reasons += [f"{k}: {got[r]}" for k, r in (("coins", "reason"), ("manifest", "manifest")) if got.get(r)]
    doc.update(verdict="reject" if reasons else "accept", reasons=reasons)
    doc.update(premises(doc))
    (out / DEVELOPMENT).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    return doc


def pearlplus_coins(served: Mapping[str, Any], out: Path, gate: str | None, key: str | None,
                    main: Path | None) -> dict[str, Any]:
    """Whether every pass with calls of Pearl+'s served run (``served``, `served_passes.py serve`'s document) has its
    epoch's salt from the live session its ``coins`` names (`served_step.coins_of`'s): the session's tree as the gate
    `gate`'s ``fetch`` gives it, and the auditor's ledger and the host's pins log as the gate answers now, written to
    `out`, with main's first-parent line `main` (`remote.main_lines`), each pass's manifest held by
    `core_check.live_salt` under the challenger key whose public key's SHA-512 is `key` (`challenger.pinned_key`).
    ``{session, closed, passes, reason}``: the passes checked, and None or the first refusal."""
    from verity.protocols.pouw.verifier import core_check as CC
    from verity.protocols.spot.device import remote
    from verity_vllm.protocols.pouw.h1 import challenger as PHC

    coins = served.get("coins") or {}
    sid = (coins.get("session") or {}).get("session")
    got: dict[str, Any] = {"session": sid, "closed": coins.get("closed"), "passes": 0, "reason": None}
    try:
        if not (gate and key and main):
            raise ValueError("no gate, key pin or main's first-parent line to check the session by (--gate, --key-sha512, "
                             "--main)")
        if not sid:
            raise ValueError("the served run names no live session")
        tgz, tree = out / "session.tgz", out / "session"
        tgz.write_bytes(_gate_out(gate, "fetch", sid))
        with tarfile.open(tgz) as t:
            t.extractall(tree, filter="data")
        for name in ("ledger", "pins"):
            (out / f"{name}.jsonl").write_bytes(_gate_out(gate, name))
        d = tree / sid / "out"
        live = {"key": PHC.pinned_key(d / remote.KEY, key).to_json(), "ledger": remote.ledger_lines(out / "ledger.jsonl"),
                "pins": remote.ledger_lines(out / "pins.jsonl"), "main": remote.main_lines(main),
                "session": remote.session_files(d)}
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


def pearlplus_draw(record: Path, out: Path, gate: str | None, key: str | None, main: Path | None,
                   fetch: Callable[..., bytes] | None = None) -> dict[str, Any]:
    """Whether the verifier's draw in its coin record `record` (`verify_run`'s "The coin record") is its draw session's:
    for a challenger draw, the session's tree as the gate `gate`'s ``fetch`` gives it, and the auditor's ledger and the
    host's pins log as it answers now, written to `out` (the session's tree at ``out/session/<id>``, beside the salt
    session's), the ledger opening that session alone on the run's commitment C under the challenger key `key` pins, run
    from a commit on main's line `main` (`challenger.trusted_session_of`), withheld (`challenger.withheld_on`), its signed
    commitment the record's and of a draw session's layout (`window.draw_epoch_reason`), its log showing C as the one
    root its coin followed, and the record's opening its coin ``pouw/draw`` 0 after C (`challenger.live_reason`) with
    the record's source its coin. `fetch` replaces the gate's answers (tests). ``{from, session, reason}``: the draw's
    origin, its session and None or the first refusal; a node's draw has no session and no refusal here (the compose
    refuses it, `pearlplus_live_draw`)."""
    from verity.protocols.pouw.verifier import salt as PS
    from verity.protocols.pouw.verifier import window as Wd
    from verity.protocols.spot.device import challenger as Ch
    from verity.protocols.spot.device import remote
    from verity_vllm.protocols.pouw.h1 import challenger as PHC

    got: dict[str, Any] = {"from": None, "session": None, "reason": None}
    try:
        rec = json.loads(record.read_text())
        draw = rec.get("draw") or {}
        got["from"] = draw.get("from")
        if got["from"] != "challenger":
            return got
        sid = got["session"] = (draw.get("reserved") or {}).get("session")
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
        d, c = tree / sid / "out", rec["commitment"]
        ck = PHC.pinned_key(d / remote.KEY, key)
        lines = remote.ledger_lines(out / "ledger.jsonl")
        trusted, received = Ch.trusted_session_of(lines, {"root": c}, ck, remote.ledger_lines(out / "pins.jsonl"),
                                                  remote.main_lines(main), remote.session_files(d), PHC.lms_verify)
        opening = Ch.LiveOpening.from_json(draw.get("opening"))
        if not Ch.withheld_on(lines, c):
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
    published the moment it is drawn: copied to ``out/coin.json``, then put in the evidence store as `COIN_KIND` through
    `research` (default the command the shared setting ``RESEARCH`` names, else this Python's ``-m research``), its meta
    the run `run` and the run
    commitment the coin was drawn on. The put asks ``--preserve`` first; where that fails (a job's node holds no store
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
    Pearl+'s verdict `answer` (its core answer): an accept's run commitment and draw source must be the published
    record's, so the verdict is over the coin published at its draw and not one drawn after it. ``{art, preserved,
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
    ``of``, and whether it left none over a cap."""
    dev = volunteered.get("device")
    return {"counts": "cpu" if dev is None else "device",
            **{k: (dev or {}).get(k) for k in ("cubin_sha256", "cross_check", "replayed", "bad_tiles")},
            "a_rows": volunteered.get("voluntary_a"), "b_rows": volunteered.get("voluntary_b"),
            **{k: volunteered.get(k) for k in ("tiles", "over", "left", "of", "ok")},
            "finished": len(volunteered.get("passes") or ())}


def pearlplus_why_tied(reserved: Mapping[str, Any] | None) -> str:
    """Why Pearl+'s piece under PEARLPLUS_TIE credits nothing by itself, its epoch's salt coin a live session's: the
    Pearl-C period that session joined, as the gate's reserve answer ``reserved`` (the window's ``coins.reserved``)
    names it, or that it joined none."""
    r = reserved or {}
    joined = (f"reserved under lane {r.get('by')}'s key into its Pearl-C period {r['period']}" if r.get("period") else
              "in no period, its reserve having joined none, so the record can't take it")
    return f"{_PEARLPLUS_TIED}; its epoch's salt coin is a live session's on vy-challenger-1, {joined}"


def pearlplus(out: Path, served: Mapping[str, Any] | None, verified: Mapping[str, Any] | None,
              unretained: Mapping[str, Any] | None = None, volunteered: Mapping[str, Any] | None = None, *,
              gate: str | None = None, key: str | None = None, main: Path | None = None,
              coin: Path | None = None, tie: Path | None = None) -> dict[str, Any]:
    """PoUW's development piece under POUW=pearlplus (``served_pearlplus.sh``), its document written to
    ``out/development.json``: Pearl+'s own served run (``served``, `served_passes.py serve`'s document) and Pearl+'s
    verdict over every pass it retained (``verified``, its ``verify``'s; None where either wrote none). Its verdict is
    accept when the run served within its caps across at least two pass boundaries, Pearl+'s verdict accepts, and its
    epoch's salt coin is its live session's (`pearlplus_coins` through the gate `gate`, the key pin `key` and main's line
    `main`; `PEARLPLUS_COINS`); with an in-process challenger's coins (`PEARLPLUS_COINS_IN_PROCESS`) a run where nothing
    failed is `UNCLAIMED`, and a run served under the gate whose coins are not live rejects. It carries that verdict's
    line, credit at Pearl's Assumption 1 (``credit``, the answer's ``public.work``) and the rounds the prover dropped from
    its retention before the coin (``dropped``, which credit nothing and stay declared), its shares are
    `served_shares.pearlplus_shares`'s wherever the run served within its caps, whatever the verdict (a rejected or
    missing one credits 0), and it credits nothing (`PEARLPLUS_WHY`, or `PEARLPLUS_WHY_IN_PROCESS`). ``unretained``, the
    same load served with no pass retained (``serve --no-retain``), adds the shares' ``unretained``
    (`served_shares.pearlplus_unretained`), or why there are none; it is a measurement and never decides the verdict.
    ``volunteered``, the prover's give-up before the coin (``served_passes.py volunteer``'s document), is ``give_up``
    (`give_up`: whose words' counts it took, the card's or the CPU's, its cross-check and the rows it gave up), and adds
    the shares' ``replay`` (`served_shares.replay_term`: its CPU seconds on its processes, the card's beside them, as the
    cores it takes to keep pace with the timed call's wall, beside the node it ran on and how many of the passes it
    finished), a measurement too.
    ``coin``, the verifier's coin as `coin` published it (its ``coin-put.json``), is ``coin`` (`pearlplus_coin`): its
    store id or why it has none, and an accept whose run commitment or draw source isn't the published record's
    rejects; the record's draw is ``coins.draw`` (`pearlplus_draw`), and a challenger draw its session doesn't hold
    rejects (a node's says so in ``coins.draws``, `PEARLPLUS_NODE_DRAWS`). The piece names its load (``load``: the requests, their prompt and output tokens, and the tokens a step).
    ``tie``, under PEARLPLUS_TIE, is the window record W the window's passes are held to (``served_passes.py retain
    --window-record``): ``served`` is then serve-openai's Pearl+ arm's document for the window (``pearl/w<k>/served.json``,
    its ``error`` why it isn't ok), its epoch's coins are `PEARLPLUS_EPOCH_TIED`'s (or `PEARLPLUS_EPOCH_TIED_IN_PROCESS`'s),
    its ``load`` is the window's steps, ``tie`` names W's `registration.digest` as ``record`` (None, and ``why``, where W
    can't be read; the record holds the verdict to W, `pearlplus_of_record`), and why it credits nothing by itself names
    the Pearl-C period the epoch's live salt session joined (`pearlplus_why_tied`, or `PEARLPLUS_WHY_TIED_IN_PROCESS`)."""
    import served_shares as SS
    from verity.protocols.spot.verifier.one_stage import registration as REG

    out.mkdir(parents=True, exist_ok=True)
    reasons = []
    source = None if served is None else (served.get("coins") or {}).get("source", "in-process")
    live = bool(gate) if served is None else source == LIVE_SOURCE
    if served is None:
        reasons.append("no served run: " + ("serve-openai's Pearl+ arm wrote no served.json for the window"
                                            if tie is not None else "served_passes.py serve wrote no document"))
    elif not served.get("ok"):
        reasons.append(f"the window's Pearl+ epoch isn't one round a Commit step: {served.get('error')}"[:800]
                       if tie is not None else
                       f"the served run crossed fewer than two pass boundaries or ran over its caps (passes over their "
                       f"caps: {served.get('over_caps')})")
    answer = (verified or {}).get("answer") or {}
    if verified is None:
        reasons.append("no verdict: served_passes.py verify wrote no document")
    elif answer.get("verdict") != "accept":
        reasons.append(f"Pearl+'s verdict rejects the run: {answer.get('reason')}"[:800])
    coins: dict[str, Any] = dict(PEARLPLUS_COINS if live else PEARLPLUS_COINS_IN_PROCESS)
    if tie is not None:
        coins["epoch"] = PEARLPLUS_EPOCH_TIED if live else PEARLPLUS_EPOCH_TIED_IN_PROCESS
    if served is not None and live:
        coins["check"] = pearlplus_coins(served, out, gate, key, main)
        if coins["check"]["reason"] is not None:
            reasons.append(f"coins: {coins['check']['reason']}")
    elif served is not None and gate:
        reasons.append(f"coins: {source}, not a live session on the gate, so nothing checks them")
    if coin is not None and (coin.parent / COIN_RECORD).is_file():
        coins["draw"] = pearlplus_draw(coin.parent / COIN_RECORD, out, gate, key, main)
        if coins["draw"]["from"] != "challenger":
            coins["draws"] = PEARLPLUS_NODE_DRAWS
        elif coins["draw"]["reason"] is not None:
            reasons.append(f"draw: {coins['draw']['reason']}")
    if tie is None:
        why = PEARLPLUS_WHY if live else PEARLPLUS_WHY_IN_PROCESS
    else:
        why = pearlplus_why_tied(((served or {}).get("coins") or {}).get("reserved")) if live else PEARLPLUS_WHY_TIED_IN_PROCESS
    doc: dict[str, Any] = {"piece": "pearlplus", "credited": False, "why": why, "coin_source": source, "coins": coins,
                           "scheme": None, "arm": None, "passes": None, "load": None, "dropped": None, "line": None,
                           "credit": None, "coin": None, "give_up": None, "shares": None, "tie": None}
    if served is not None:
        arm = served.get("arm") or served.get("entry")
        doc.update(arm=arm, scheme=(arm or "").partition(":")[2] or None,
                   passes=sum(1 for p in served.get("passes") or () if p.get("units")),
                   load={"window_steps": served.get("commit_steps")} if tie is not None else
                   {**{k: served.get(k) for k in ("requests", "prompt_tokens", "output_tokens")},
                    "max_num_batched_tokens": (served.get("engine") or {}).get("max_num_batched_tokens")},
                   **{k: served.get(k) for k in ("spec", "run", "served_tokens", "generate_s", "prompt_tokens", "output_tokens")})
    if tie is not None:
        try:
            doc["tie"] = {"record": REG.digest(json.loads(tie.read_text()))}
        except (OSError, ValueError, AttributeError) as e:
            doc["tie"] = {"record": None, "why": f"W can't be read: {type(e).__name__}: {e}"[:400]}
    if verified is not None:
        doc.update(line=verified.get("line"), credit=(answer.get("public") or {}).get("work"), dropped=verified.get("dropped"))
    if coin is not None:
        doc["coin"] = pearlplus_coin(coin, answer)
        if doc["coin"]["reason"] is not None and answer.get("verdict") == "accept":
            reasons.append(f"coin: {doc['coin']['reason']}")
    if volunteered is not None:
        doc["give_up"] = give_up(volunteered)
    if served is not None and served.get("ok") and "generate_s" not in served:
        doc["shares"] = {"none": "the epoch is a Commit window's (serve_pearl's pearl/w<k>/served.json), which times no "
                                 "generate call of its own"}
    elif served is not None and served.get("ok"):
        try:
            doc["shares"] = SS.pearlplus_shares(dict(served), dict(verified or {}))
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
    from verity.protocols.pouw.verifier import core_check as CC

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
    """A subprotocol's report row from its answer (checked to be a core answer first); ``run`` is the recording run."""
    a = check_answer(a)
    runs = json.loads((out / "runs.json").read_text()) if (out / "runs.json").exists() else []
    accepted = a["verdict"] == "accept"
    return {"protocol": a["protocol"], "owner": owner, "verdict": a["verdict"], "reason": a["reason"],
            "commitment": a["commitment"], "quantity": a["public"] if accepted else None,
            "delta": (a["profile"] or {}).get("delta") if isinstance(a["profile"], dict) else None,
            "epsilon": epsilon(a) if accepted else None, "runs": [*([run] if run else []), *runs], "answer": a}


def spot_row(audit_dir: Path, run: str | None = None) -> dict[str, Any]:
    """Sampled proofs' row from the records audit kept under ``audit_dir`` (module docstring)."""
    import served_integrity as SI

    reg = audit_dir / WINDOW_REGISTRATION
    try:
        a = SI.answer(audit_dir, json.loads(reg.read_text()) if reg.exists() else None)
        return row_of(INTEGRITY_OWNER, a, audit_dir, run)
    except (RuntimeError, ValueError) as e:
        return {"verdict": "error", "owner": INTEGRITY_OWNER, "reason": f"{type(e).__name__}: {e}"[:2000]}


def lean_sampled_row(audit_dir: Path, scratch: Path, exe: str | os.PathLike | None = None,
                     run: str | None = None) -> dict[str, Any]:
    """Sampled proofs' row from this host's own run of remote's sampled proofs Lean (`served_cases.sampled_case`: the
    records audit's v2 case, its blobs the proofs and public files, in the form ``one-stage``; ``exe`` default built here)
    on files in a 0700 directory under ``scratch``, its answer tied to the window's registration record (`LEAN_HERE`),
    whatever the audit's summary and record say. An audit that wrote no v2 case (a plain run's, or one a5 refused) leaves
    no proofs to recompute from: its row is `spot_row`'s, its ``answer_from`` saying so. A case the Lean
    doesn't make or answer, or an answer about another record, is the row's error."""
    import served_cases as SC

    if not (audit_dir / "window" / "case-v2.json").is_file():
        return {**spot_row(audit_dir, run),
                "answer_from": f"{AS_WRITTEN} ({WRITTEN['integrity']}): the records audit wrote no v2 case "
                               "(window/case-v2.json), so there are no proofs to recompute it from"}
    try:
        case = SC.sampled_case(audit_dir)
        answer, _line = SC.lean_answer(exe or SC.build(*SC.SAMPLED_LEAN), case, scratch, SC.SAMPLED_CMD)
        SC.tie(answer, SC.SAMPLED, SC.record_of(case)[1])
        return {**row_of(INTEGRITY_OWNER, answer, audit_dir, run), "answer_from": LEAN_HERE}
    except (SC.Refused, OSError, ValueError) as e:
        return {"verdict": "error", "owner": INTEGRITY_OWNER, "reason": f"{type(e).__name__}: {e}"[:2000],
                "answer_from": LEAN_HERE}


def _json(path: Path) -> Any:
    """``path``'s JSON, None where there's no such file; one that isn't JSON raises ValueError."""
    return json.loads(path.read_text()) if path.exists() else None


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


def pearlplus_live_draw(record: Mapping[str, Any], piece_dir: Path, key: str, lane: str, check: Mapping[str, Any],
                        lines: Sequence[str], period: str) -> str | None:
    """Why the verifier's draw source in its coin record `record` (the piece's `COIN_RECORD`, `verify_run`'s "The coin
    record") is not a coin the auditor host's challenger under `key` opened on the run commitment C, or None. It must be
    a challenger draw (`PEARLPLUS_NODE_DRAW`) whose session the piece's check held to the signed ledger and session
    (`check`, `pearlplus_draw`'s), under `key` (that session's tree in the piece); in the piece's ledger `lines`, the
    first reserve of lane `lane` after its Pearl-C period `period` closed, withheld with a draw session's layout alone,
    and the one session opened on C, so a prover who draws again under another session holds a draw no compose takes;
    and the record's opening its coin ``pouw/draw`` 0 after C (`challenger.live_reason`), the record's source that coin."""
    from verity.protocols.pouw.verifier import window as Wd
    from verity.protocols.spot.device import challenger as Ch
    from verity.protocols.spot.device import remote
    from verity_vllm.protocols.pouw.h1 import challenger as PHC
    from verity_vllm.protocols.pouw.h1 import window as PHW

    if record.get("schema") != COIN_KIND:
        return f"the verifier's coin record is {record.get('schema')!r}, not {COIN_KIND}, which carries its draw"
    draw = record.get("draw") or {}
    if draw.get("from") != "challenger":
        return PEARLPLUS_NODE_DRAW
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


#: a rational as JSON carries ε: an integer, a fraction with a nonzero denominator, or a decimal
_RATIONAL = re.compile(r"[+-]?\d+/0*[1-9]\d*|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def _rational(x: Any) -> Fraction | None:
    """``x`` as a `Fraction` when its text is a rational (`_RATIONAL`), else None."""
    s = str(x).strip()
    return Fraction(s) if not isinstance(x, bool) and _RATIONAL.fullmatch(s) else None


def pearlplus_of_record(doc: Mapping[str, Any], piece_dir: Path, sampled_proofs: Path | None, key: str | None,
                        lane: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Whether Pearl+'s accepted development piece `doc` (`pearlplus`'s, in `piece_dir`) is PoUW's verdict of record: its
    answer and None when it is, else None and the first condition it fails, in order:

    1. **Live coins under the pin.** The epoch's salt coin is the live session's, every pass held to it
       (`pearlplus_coins`' check), and that session's challenger key is the one `key` pins (`challenger.pinned_key` on
       the session's tree the piece fetched).
    2. **The lane's own reserve.** The piece's ledger reserves that session by lane `lane`, under its own gate key, not
       for it.
    3. **The lane's open period, the live draw, and the window tie.** The session joined lane `lane`'s period open at
       the reserve, a Pearl-C one (its open commits none of H1's fields: `credit.pearl_c_period_credit` credits a salt
       session in no other). The verifier's draw source is a draw session's coin opened on the run's commitment, that
       session the lane's first reserve after the period closed (`pearlplus_live_draw` over `pearlplus_draw`'s check),
       and the verdict is over the coin published at its draw (`pearlplus_coin`).
       Pearl+'s verdict (`PEARLPLUS_ANSWER`) held its passes to a window record W (``verify --tie W``), the
       window's registration R under `sampled_proofs` is chained right after W, and its records audit is tied to W
       (``records_audit --tie W``), so the window sampled proofs audited reads the roots Pearl+ credits. The period's
       open (in the signed ledger) committed the SHA-512 of the window-circuit record V3c held R's circuit to
       (`CIRCUIT_RECORD` beside R), so the period holds the salt session to the circuit of the window served.
    4. **Lean.** The Lean `PouwCheck.PearlPlus` verdict (`PEARLPLUS_LEAN`) accepts the same run commitment on the same
       draw source, holds the passes to the same W, and credits work over tiles at the reference's ε.

    The answer is Pearl+'s, its work the Lean's, its profile the reference's with the lane, pin, period, session and the
    Lean verdict. Nothing here decides anew: each condition reads what a check or a verifier answered."""
    from verity.protocols.pouw.verifier import credit as CR
    from verity.protocols.spot.device import remote
    from verity_vllm.protocols.pouw.h1 import challenger as PHC

    if doc.get("coin_source") != LIVE_SOURCE:
        return None, f"its epoch's coins come from {doc.get('coin_source')!r}, not a live session on the auditor host's challenger"
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
    if h1 := [f for f in (CR.K_FIELD, CR.ROWS_FIELD, CR.LENGTH_FIELD) if f in period]:
        return None, (f"the live session joined period {name}, H1's (its open commits {', '.join(h1)}): a Pearl-C salt "
                      "session is credited in a Pearl-C period alone")
    try:
        coin = json.loads((piece_dir / COIN_RECORD).read_text())
    except (OSError, ValueError) as e:
        return None, f"the verifier's coin record can't be read ({type(e).__name__}: {e})"
    drew = (doc.get("coins") or {}).get("draw") or {}
    if (why := pearlplus_live_draw(coin, piece_dir, key, lane, drew, lines, str(name))) is not None:
        return None, why
    published = doc.get("coin") or {}
    if not published or published.get("reason") is not None:
        return None, f"the verdict is not held to the coin published at its draw: {published.get('reason') or 'none published'}"
    try:
        answer = check_answer(json.loads((piece_dir / PEARLPLUS_ANSWER).read_text()))
    except (OSError, ValueError) as e:
        return None, f"Pearl+'s verdict's answer ({PEARLPLUS_ANSWER}) can't be read ({type(e).__name__}: {e})"
    if (answer["protocol"], answer["verdict"]) != (PEARLPLUS_PROTOCOL, "accept"):
        return None, f"Pearl+'s verdict's answer is {answer['protocol']}'s {answer['verdict']}, not {PEARLPLUS_PROTOCOL}'s accept"
    w = ((answer["public"] or {}).get("tie") or {}).get("record")
    if not isinstance(w, str):
        return None, ("Pearl+'s verdict holds its passes to no window record W (served_passes.py verify --tie W), so "
                      "nothing ties them to the window")
    if sampled_proofs is None:
        return None, "the compose names no window registration to tie W to (--window-tie, else --sampled-proofs)"
    try:
        reg = json.loads((sampled_proofs / WINDOW_REGISTRATION).read_text())
        summary = json.loads((sampled_proofs / AUDIT_SUMMARY).read_text())
    except (OSError, ValueError) as e:
        return None, f"the window's registration or its audit's summary can't be read ({type(e).__name__}: {e})"
    if reg.get("previous") != w:
        return None, (f"the window's registration R is chained after {str(reg.get('previous'))[:16]}..., not right after "
                      f"Pearl+'s window record W {w[:16]}...")
    if ((summary.get("tie") or {}).get("record")) != w:
        return None, f"the window's records audit is not tied to Pearl+'s window record W {w[:16]}... (records_audit --tie W)"
    try:
        held = hashlib.sha512((sampled_proofs / CIRCUIT_RECORD).read_bytes()).hexdigest()
    except OSError as e:
        return None, (f"no window-circuit record V3c held R's circuit to ({CIRCUIT_RECORD} beside R: {type(e).__name__}), "
                      f"so nothing holds period {name}'s circuit to the window served")
    if period.get("circuit") != held:
        return None, (f"period {name} committed circuit {str(period.get('circuit'))[:16]}..., not {held[:16]}..., the "
                      "SHA-512 of the window-circuit record V3c held R's circuit to: its salt session counts toward "
                      "another circuit than the window served")

    try:
        lean = json.loads((piece_dir / PEARLPLUS_LEAN).read_text())
    except (OSError, ValueError) as e:
        return None, f"no Lean PouwCheck.PearlPlus verdict ({PEARLPLUS_LEAN}: {type(e).__name__}: {e})"
    if not isinstance(lean, dict) or lean.get("verdict") != "accept":
        v = lean.get("verdict") if isinstance(lean, dict) else None
        return None, f"the Lean PouwCheck.PearlPlus verdict is {v}: {lean.get('reason') if isinstance(lean, dict) else lean}"
    source = (answer["profile"] or {}).get("source")
    if (lean.get("commitment"), lean.get("source")) != (answer["commitment"], source):
        return None, (f"the Lean verdict is on run {str(lean.get('commitment'))[:16]}... at source "
                      f"{str(lean.get('source'))[:16]}..., not Pearl+'s verdict's run and source")
    if ((lean.get("tie") or {}).get("record")) != w:
        return None, f"the Lean verdict doesn't hold the passes to Pearl+'s window record W {w[:16]}..."
    work, ref = lean.get("work") or {}, (answer["public"] or {}).get("work") or {}
    credited, tiles = work.get("credited"), work.get("credited_tiles")
    if not (type(credited) is int and credited > 0 and type(tiles) is int and tiles > 0):
        return None, f"the Lean verdict credits {credited!r} MACs over {tiles!r} tiles: nothing"
    eps = _rational(work.get("epsilon"))
    if eps is None or eps != _rational(ref.get("epsilon")):
        return None, f"the Lean verdict's ε {work.get('epsilon')!r} is not the reference's {ref.get('epsilon')!r}"
    return {**answer, "public": {**answer["public"], "work": {k: work.get(k) for k in
                                                              ("credited", "credited_tiles", "macs", "rho", "epsilon")}},
            "profile": {**answer["profile"], "lane": lane, "key_sha512": key, "period": name, "session": session,
                        "draw_session": drew.get("session"), "reference_work": ref,
                        "lean": {k: lean.get(k) for k in ("verdict", "commitment", "source", "drawn", "work", "tie")}}}, None


def piece(key: str, pieces_dir: Path, run: str | None = None, **of_record: Any
          ) -> tuple[dict[str, Any], tuple[str | None, str]]:
    """A subprotocol's `piece_row` (`of_record`: its ``sampled_proofs``, ``key_sha512`` and ``lane``) and `piece_server`;
    a piece that left a file that isn't JSON is its ``error``, about no server."""
    try:
        return piece_row(key, pieces_dir, run, **of_record), piece_server(key, pieces_dir)
    except ValueError as e:
        return ({"verdict": "error", "owner": PIECES.get(key, WIPE)[2],
                 "reason": f"its piece left a file that isn't JSON: {type(e).__name__}: {e}"[:2000]}, (None, "real"))


def piece_row(key: str, pieces_dir: Path, run: str | None = None, *, sampled_proofs: Path | None = None,
              key_sha512: str | None = None, lane: str | None = None) -> dict[str, Any]:
    """A subprotocol's runner row from its piece's answer; Pearl+'s accepted development piece is PoUW's verdict of record
    where `pearlplus_of_record` says so (`sampled_proofs`, `key_sha512` and `lane` what it reads), its row that answer's
    with its document as ``piece``, else ``missing`` with its document, its ``why`` `DEV_VERDICT` and the first condition
    of record it fails; any other development piece's is ``missing`` with its document when it accepts or claims no
    verdict (`UNCLAIMED`; it credits nothing, for the document's ``why``, else `DEV_WHY`), else ``reject``; a piece that
    ran (`RAN`) and wrote neither is its ``error``, unless it refused to run (`REFUSED`); otherwise ``missing``, naming
    why."""
    _name, comp, owner, what = PIECES.get(key, WIPE)
    d = pieces_dir / comp
    f, dev, ran = d / "answer.json", d / DEVELOPMENT, _json(d / RAN)
    if not f.exists() and dev.exists():
        doc = _json(dev)
        if not isinstance(doc, dict):
            return {"verdict": "error", "owner": owner, "reason": f"its piece's {DEVELOPMENT} is not a JSON object"}
        if key == "work" and doc.get("piece") == "pearlplus" and doc.get("verdict") == "accept":
            answer, failed = pearlplus_of_record(doc, d, sampled_proofs, key_sha512, lane)
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
            return {"verdict": "missing", "owner": owner, "why": f"its piece refused to run: {said}"[:2000]}
        if isinstance(ran, dict):
            late = f", out of time at {ran.get('limit')}" if ran.get("rc") in (124, 137) else ""
            return {"verdict": "error", "owner": owner,
                    "reason": f"its piece ran (exit {ran.get('rc')}{late}) and wrote no answer; its log is {comp}/piece.log"}
        if key == "wipe" and (d / "piece.log").exists():
            return {"verdict": "missing", "owner": owner, "why": f"its piece wrote no answer; its log is {comp}/piece.log"}
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


def terms(r: Mapping[str, Any], leg: Mapping[str, Any] | None) -> list[str]:
    """What an accepted answer holds beyond its verdict: where its profile says when its weights were held and what share
    of the checkpoint they are, those; where it says which host ran each party (the network certifier's ``placement``),
    the link K counts, named, with every other link of the node uncounted and uncertified, and what its `trusted-host`
    premise rests on; and each δ of the block record's at which its leg's quantity is not established. A placement
    with no links (the front's ``--gate local``) states no link."""
    profile = (r.get("answer") or {}).get("profile")
    w = profile.get("weights") if isinstance(profile, Mapping) else None
    place = profile.get("placement") if isinstance(profile, Mapping) else None
    place = place if isinstance(place, Mapping) else {}
    counted = [f"{x['from']} to {x['to']} through {x['via']}" for x in place.get("links") or () if x.get("counted")]
    others = place.get("uncounted") or "every other link of the node"
    out = ([f"K counts the relayed link alone, {'; '.join(counted)}; {others} is uncounted and uncertified"]
           if counted else [])
    out += [f"trusted-host {place['trusted-host']}"] if place.get("trusted-host") else []
    if isinstance(w, Mapping) and w.get("held"):
        out.append(f"its weights {w['held']}, not concurrently with the block")
    if isinstance(w, Mapping) and w.get("share"):
        outside = (profile.get("outside") or {}).get("bytes") or {}
        out.append(f"they are {float(Fraction(w['share'])):.1%} of the checkpoint ({w.get('covers', 'its weights')})"
                   + (f"; {', '.join(sorted(outside))} are outside its claim" if outside else ""))
    if leg:
        no = [d for d, x in leg["at"].items() if not x["holds"]]
        if no:
            from verity_e2e_demo import block_record as BR
            out.append(f"its quantity is not established at δ = {' or '.join(no)}: the run established it except with "
                       f"{_ratio(Fraction(leg['error']), BR.DELTAS)}")
    return out


def said(rows: Mapping[str, Mapping[str, Any]], at: Mapping[str, tuple[str | None, str]], node: str,
         legs: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """The composed verdict's reason: one clause a subprotocol, its verdict and, unless it accepted about ``node``, why; an
    accept about ``node`` with its `terms` (``legs``: each accepted one's block record leg, by runner name)."""
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
            t = terms(r, (legs or {}).get(k))
            s = ("accept" if server is not None else "accept, naming no server: not carried") + (f" ({'; '.join(t)})" if t else "")
            if r.get("answer_from") == PEARLPLUS_FROM:
                s += f", {PEARLPLUS_SAID}"
        else:
            s = f"{v}: {r.get('reason')}"
        out.append(f"{NAMES[k]}: {s}")
    return "; ".join(out)


def witness(paths: Sequence[str]) -> list[tuple[str, str]]:
    """Each of ``paths`` (relative to a serve dir) that is a served run's witness, with why (`WITNESS_DIRS`,
    `WITNESS_TOPS`, `WITNESS_NAMES`, and H1's prover state under ``private/h1``)."""
    out = []
    for p in paths:
        parts = Path(p).parts
        if not parts:
            continue
        why = next((WITNESS_DIRS[x] for x in parts[:-1] if x in WITNESS_DIRS), None)
        why = why or (WITNESS_TOPS.get(parts[0]) if len(parts) > 1 else None)
        why = why or ("H1's prover state (its key)" if parts[:2] == ("private", "h1") else None)
        why = why or next((w for pat, w in WITNESS_NAMES.items() if fnmatch.fnmatchcase(parts[-1], pat)), None)
        if why:
            out.append((p, why))
    return out


def rebase(serve_dir: Path, served_dir: str, paths: Sequence[str]) -> list[str]:
    """Each case among ``paths`` (relative to ``serve_dir``, a hand-off's copy of the served node's serve dir
    ``served_dir``) that names ``blobs``, its blobs rewritten to the same paths under ``serve_dir``; the cases rewritten.
    ValueError on a blob outside ``served_dir``, which no hand-off carries, or on such a file that isn't JSON."""
    base, done = served_dir.rstrip("/") + "/", []
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
        case["blobs"] = [str(serve_dir / b[len(base):]) for b in case["blobs"]]
        f.write_text(json.dumps(case))
        done.append(rel)
    return done


#: under a serve dir's private/, the runner's own directories, whose cases are a composite's, a hand-off's or a
#: deferral's copies and no piece's
NOT_A_PIECES = ("composite", "compose", "deferred")


def onsite_written(serve_dir: Path, pieces: Path) -> dict[str, list[str]]:
    """Each of onsite's own subprotocols, the ones its composite holds and remote's doesn't (`composite.COMPONENTS`), with
    the verdict of every answer of its that lies beside a case under ``serve_dir``'s private/ (outside `NOT_A_PIECES`) or
    under ``pieces``, the run's public pieces: the case as its piece wrote it, with its own answer on it; none where no
    piece wrote one. It reads the answers alone, never a case, and holds no case to its answer: for the composite,
    `onsite_pieces.sh`'s `onsite_cases` holds PoUS's and the wipe's to their pieces' Lean answers. A case with no answer
    beside it counts for no subprotocol, and an answer that isn't JSON raises `ValueError`, naming the directory under
    private/ or ``pieces`` it lies in."""
    import composite as C

    found: dict[str, list[str]] = {p: [] for p in sorted(set(C.COMPONENTS["onsite"]) - set(C.COMPONENTS["remote"]))}
    private = serve_dir / "private"
    skips = [private / x for x in NOT_A_PIECES]
    for root in (private, pieces):
        for f in sorted(root.rglob("case.json")) if root.is_dir() else []:
            d = f.parent
            if any(d == s or s in d.parents for s in skips) or not (d / "answer.json").is_file():
                continue
            try:
                answer = json.loads((d / "answer.json").read_text())
            except ValueError as e:
                raise ValueError(f"the answer beside the case under {(d.relative_to(root).parts or ('.',))[0]} isn't JSON: {e}") from e
            protocol = answer.get("protocol") if isinstance(answer, dict) else None
            if isinstance(protocol, str) and protocol in found:
                found[protocol].append(str(answer.get("verdict")))
    return found


def composed(answer: Path, out: Path, gate_exit: int | None = None, err: Path | None = None) -> tuple[int, str]:
    """The compose gate's answer (one JSON line) written out under ``out``: its $OUT's files at their paths, and
    ``gate.json`` (the pin and the exit, without the files). Its exit (0 composed and accepted) and its one line, which
    without an answer ends with the last line of the gate command's stderr ``err`` (ssh's own refusal, say)."""
    try:
        doc = json.loads(answer.read_text().strip().splitlines()[-1])
    except (OSError, IndexError, ValueError) as e:
        said = [x for x in (err.read_text(errors="replace").splitlines() if err and err.is_file() else []) if x.strip()]
        return 1, (f"GATE-COMPOSE failed (ssh exit {gate_exit}): no answer from the compose gate ({type(e).__name__})"
                   + (f": {said[-1].strip()[:400]}" if said else ""))
    if "refused" in doc:
        return 3, f"GATE-COMPOSE refused by the compose gate: {str(doc['refused'])[:800]}"
    c = doc.get("compose") or {}
    files = c.get("files") or {}
    root = out.resolve()
    for rel, text in files.items():
        f = (out / rel).resolve()
        if root not in f.parents:
            return 1, f"GATE-COMPOSE failed: the answer names {rel[:200]!r}, outside {out}"
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
    where = (f"at the compose gate's tree {str(doc.get('pinned'))[:12]}"
             + (f" (on {doc['pin']['on']})" if (doc.get("pin") or {}).get("on") else ""))
    rc = 0 if c.get("rc") == 0 and gate_exit in (None, 0) else 1
    return rc, (f"GATE-COMPOSE {where}: exit {c.get('rc')}, the record {said or 'not written'}, "
                f"{len(files)} files of its $OUT in {out}" + (f", {len(c['not_sent'])} not sent" if c.get("not_sent") else ""))


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


def record(rows: Mapping[str, Mapping[str, Any]], *, window: int, steps: tuple[int, int], node: str,
           run: str | None, servers: Mapping[str, tuple[str | None, str]] | None = None, product: str = "onsite",
           deferred: str | None = None, off_node: bool = False, tree: Mapping[str, str] | None = None,
           served_compare: str | None = None) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """The window report over ``rows`` (by runner name), its block record (None when a subprotocol about this node
    rejected, erred or was undecided), and the ``RECORD`` line's summary, composed for ``product``: each subprotocol it
    drops is its `dropped_row`, about no server. ``servers``: each piece's `piece_server`, by runner name; sampled proofs'
    answer is about ``node``, and a piece's that names no server isn't carried. ``deferred``: what the run left to a
    follow-on and why, and ``served_compare``: the served compare's verdict line (`served_compare_line`), each of which
    the report and the line carry. ``off_node``: composed from the served node's compose
    hand-off (`HANDED_FROM`); ValueError on ``node`` itself. ``tree``: the composing tree's ``tree`` (its commit) and
    ``tree_on`` (the branch its host pinned it from), which ``composed_on`` carries."""
    from verity_e2e_demo import block_record as BR

    host = socket.gethostname()
    if off_node and host == node:
        raise ValueError(f"a record composed off the served node is not composed on it: this host is {node}")

    legs = {v: k for k, v in BR.RUNNER_KEYS.items()} | {"wipe": "wipe"}
    drops = [k for k in rows if legs[k] not in BR.PRODUCTS[product]]
    at = ({k: (None, "real") for k in rows} | dict(servers or {}) | {"integrity": (node, "real")}
          | {k: (None, "real") for k in drops})
    rows = {k: dropped_row(k, product) if k in drops else dict(r) for k, r in rows.items()}
    for k, r in rows.items():
        if k not in drops and "answer_from" not in r and (r.get("verdict") != "missing" or "development" in r):
            r["answer_from"] = f"{AS_WRITTEN} ({WRITTEN.get(k) or f'its piece files in {PIECES.get(k, WIPE)[1]}/'})"
    built: dict[str, dict[str, Any]] = {}
    for k, r in rows.items():
        if r.get("verdict") == "accept" and at[k][0] is not None:
            try:
                built[k] = BR.leg(legs[k], r, server=at[k][0], runs=[], source=at[k][1])
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as e:
                rows[k] = {**r, "verdict": "error", "quantity": None, "epsilon": None,
                           "reason": f"its answer is not the block record's {legs[k]} leg: {type(e).__name__}: {e}"[:2000]}
    where = {"host": host, "served_node": host == node,
             "from": HANDED_FROM if off_node else COMPOSED_FROM if host == node else COPIED_FROM, **(tree or {})}
    left = {k: v for k, v in (("deferred", deferred), ("served_compare", served_compare)) if v}
    report = {"format": REPORT_FORMAT, "window": window, "node": node, "run": run, "protocols": rows,
              "servers": {k: at[k][0] for k in rows}, "composed_on": where, **left}
    verdicts = {NAMES[k]: DEV_VERDICT if "development" in r and r.get("verdict") == "missing" else r.get("verdict")
                for k, r in rows.items()}
    missing = {NAMES[k]: r.get("why") for k, r in rows.items() if r.get("verdict") == "missing"}
    elsewhere = {NAMES[k]: at[k][0] for k in rows if at[k][0] not in (None, node)}
    unserved = sorted(NAMES[k] for k, r in rows.items()
                      if r.get("verdict") not in ("missing", "dropped") and at[k][0] is None)
    dev = {"development": {NAMES[k]: r["development"]["verdict"] for k, r in rows.items() if "development" in r}}
    dev = dev if dev["development"] else {}
    answers = {NAMES[k]: r["answer_from"] for k, r in rows.items() if "answer_from" in r}
    named = {"product": product, "verdicts": verdicts, "missing": missing, "elsewhere": elsewhere, "composed_on": where,
             "answers": answers, **left, **({"dropped": {NAMES[k]: rows[k]["why"] for k in drops}} if drops else {}),
             **({"unserved": unserved} if unserved else {}), **dev}
    bad = sorted(NAMES[k] for k, r in rows.items()
                 if r.get("verdict") not in ("accept", "missing", "dropped") and NAMES[k] not in elsewhere)
    if bad:
        composed = "reject" if any(rows[k].get("verdict") in ("reject", "error") for k in rows
                                   if NAMES[k] in bad) else "pending"
        report["composed"] = {"protocol": REPORT_FORMAT, "verdict": composed, "reason": said(rows, at, node, built)}
        line = {"composed": composed, "by": bad, **named, "block_record": None, "reason": report["composed"]["reason"],
                "why": "block_record carries an accepted subprotocol's certificate only"}
        return report, None, line
    extra = {legs[k]: {"row": r, "server": at[k][0], "runs": r.get("runs") or [run], "source": at[k][1]}
             for k, r in rows.items() if k != "integrity" and r.get("verdict") == "accept" and at[k][0] is not None}
    composing = {**report, "protocols": {k: r if k == "integrity" else {"verdict": "missing"} for k, r in rows.items()}}
    rec = BR.from_period(composing, steps=steps, deployment=BR.REFERENCE, extra=extra, product=product)
    verdict = WINDOW_VERDICT.get(rec["composed"]["verdict"], rec["composed"]["verdict"])
    report["composed"] = {"protocol": REPORT_FORMAT, "verdict": verdict, "reason": said(rows, at, node, built)}
    line = {"composed": verdict, **named, "reason": report["composed"]["reason"],
            "record_missing": rec["composed"]["missing"],
            "check": BR.check(rec), "steps": list(steps), "node": node,
            "deployment": rec["monitoring_period"]["deployment"]["id"]}
    return report, rec, line


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("window")
    w.add_argument("serve_dir", type=Path)
    w.add_argument("--slots", type=int, default=8)
    w.add_argument("--window", type=int)
    g = sub.add_parser("gate")
    g.add_argument("--gate", default=os.environ.get("H1_GATE"))
    g.add_argument("--lane", default="vllm")
    g.add_argument("--main", type=Path, default=os.environ.get("MAIN_FILE") or None)
    g.add_argument("--root", help="a window's run root (128 hex digits) a session of this run opens on")
    pe = sub.add_parser("period")
    pe.add_argument("verb", choices=("open", "served", "close"))
    pe.add_argument("serve_dir", type=Path)
    pe.add_argument("--gate")
    pe.add_argument("--name")
    pe.add_argument("--circuits", type=Path)
    pe.add_argument("--slots", type=int)
    pe.add_argument("--max-tokens", type=int)
    pe.add_argument("--draw-k", type=int)
    pe.add_argument("--model")
    pr = sub.add_parser("program")
    pr.add_argument("out", type=Path)
    pr.add_argument("--model", default="verity-int8/Llama-3.2-1B-INT8")
    pr.add_argument("--rows", type=int, default=512)
    pr.add_argument("--steps", type=int, default=16)
    pr.add_argument("--like", type=Path)
    pr.add_argument("--protocols-config", help="serve-openai's --protocols-config (H1_PROTOCOLS_CONFIG), the same value")
    o = sub.add_parser("pouw")
    o.add_argument("serve_dir", type=Path)
    o.add_argument("window", type=int)
    o.add_argument("out", type=Path)
    o.add_argument("--gate", default=os.environ.get("H1_GATE"))
    o.add_argument("--key-sha512", default=os.environ.get("H1_KEY_SHA512"))
    o.add_argument("--main", type=Path, default=os.environ.get("MAIN_FILE") or None)
    o.add_argument("--python", default=os.environ.get("PY") or sys.executable)
    o.add_argument("--statements", type=Path)
    o.add_argument("--live-evidence", type=Path)
    o.add_argument("--lean-answers", type=Path, help="the verifier's memo of its runs' answers, in its own scratch")
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
    ppe.add_argument("--scheme", default="pearl-c-h100-v1-h3")
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
    rg = sub.add_parser("public-registration")
    rg.add_argument("registration", type=Path)
    pl = sub.add_parser("public-lines")
    pl.add_argument("--private", type=Path)
    pb = sub.add_parser("publish")
    pb.add_argument("out", type=Path)
    pb.add_argument("--private", type=Path, required=True)
    sl = sub.add_parser("sampled-line")
    sl.add_argument("log", type=Path)
    sl.add_argument("--partition", type=Path)
    fl = sub.add_parser("front-line")
    fl.add_argument("session", type=Path)
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
    r.add_argument("--sampled-exe", help="vstar-recursive, with --sampled-lean (default: built here)")
    r.add_argument("--tree", help="the commit of the tree composing the record, which composed_on names")
    r.add_argument("--tree-on", help="the branch the composing host pinned that tree from, with --tree")
    r.add_argument("--key-sha512", help="the challenger key's pin Pearl+'s live coins are held to for a verdict of record "
                                        "(H1_KEY_SHA512)")
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
    oc = sub.add_parser("onsite-written")
    oc.add_argument("--serve-run", type=Path, required=True)
    oc.add_argument("--pieces", type=Path, required=True)
    cd = sub.add_parser("composed")
    cd.add_argument("answer", type=Path)
    cd.add_argument("out", type=Path)
    cd.add_argument("--gate-exit", type=int)
    cd.add_argument("--err", type=Path)
    for q in (p, r):
        q.add_argument("--node", default=socket.gethostname())
        q.add_argument("--product", default="onsite")
    a = ap.parse_args(argv)
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
            done = rebase(a.serve_dir, a.served_dir, [x for x in sys.stdin.read().splitlines() if x.strip()])
        except (OSError, ValueError) as e:
            print(f"REBASE refused: {type(e).__name__}: {e}", flush=True)
            return 1
        print(f"REBASED {len(done)} case(s)' blobs from {a.served_dir} to {a.serve_dir}", flush=True)
        return 0
    if a.cmd == "onsite-written":
        print("; ".join(f"{p} {', '.join(v) if v else 'none'}" for p, v in onsite_written(a.serve_run, a.pieces).items()),
              flush=True)
        return 0
    if a.cmd == "composed":
        a.out.mkdir(parents=True, exist_ok=True)
        rc, line = composed(a.answer, a.out, a.gate_exit, a.err)
        print(line, flush=True)
        return rc
    if a.cmd == "public-registration":
        print(json.dumps(public_registration(json.loads(a.registration.read_text())), indent=1, sort_keys=True))
        return 0
    if a.cmd == "sampled-line":
        rows = [x[len("RECORDS "):] for x in a.log.read_text().splitlines() if x.startswith("RECORDS ")]
        if not rows:
            print("no RECORDS line")
            return 1
        print(sampled_line(json.loads(rows[0]), json.loads(a.partition.read_text()) if a.partition else None))
        return 0
    if a.cmd == "front-line":
        try:
            print(front_line(json.loads(a.session.read_text())))
        except (OSError, ValueError) as e:
            print(f"no session ({type(e).__name__})")
            return 1
        return 0
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
            print(json.dumps(pick(a.serve_dir, a.slots, a.window), sort_keys=True))
        except ValueError as e:
            print(f"served_verdicts.py window: {e}", file=sys.stderr)
            return 1
        return 0
    if a.cmd == "gate":
        if not a.gate:
            print("GATE refused: no gate command (--gate, H1_GATE)", flush=True)
            return 1
        try:
            ok, line = gate_ready(a.gate, a.lane, a.main, a.root)
        except (KeyError, OSError, RuntimeError, ValueError) as e:
            ok, line = False, f"GATE refused: {type(e).__name__}: {e}"[:800]
        print(line, flush=True)
        return 0 if ok else 1
    if a.cmd == "period":
        try:
            if a.verb == "served":
                doc = period_served(a.serve_dir)
                print(f"PERIOD served {doc['name']}: serve-openai declared its Program {doc['circuit'][:16]}...",
                      flush=True)
                return 0
            if not a.gate:
                raise ValueError("no gate command (--gate)")
            if a.verb == "close":
                doc = period_close(a.serve_dir, a.gate)
                print(f"PERIOD closed {doc['name']}: {json.dumps(public_json(doc['close']), sort_keys=True)[:600]}",
                      flush=True)
                return 0
            if not (a.name and a.circuits and a.slots and a.max_tokens):
                raise ValueError("period open needs --name, --circuits, --slots and --max-tokens")
            doc = period_open(a.serve_dir, a.gate, a.name, a.circuits, slots=a.slots, max_tokens=a.max_tokens,
                              draw_k=a.draw_k, model=a.model)
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            print(f"PERIOD refused: {type(e).__name__}: {e}"[:1200], flush=True)
            return 1
        print(f"PERIOD opened {doc['name']}: circuit {doc['circuit'][:16]}..., T {doc['rows']}, S {doc['steps']}, draw K "
              f"{doc['draw_k']}, weights {' '.join(doc['weights']) or 'none'}, files {sorted(doc['circuit_files'])}"
              + (f" (no k<K> file yet for K = {doc['missing']}: a drawn statement there is refused)" if doc["missing"]
                 else ""), flush=True)
        return 0
    if a.cmd == "program":
        try:
            meta = program(a.out, a.model, a.rows, a.steps, a.like, a.protocols_config)
        except (OSError, ValueError, KeyError) as e:
            print(f"PROGRAM refused: {type(e).__name__}: {e}"[:1200], flush=True)
            return 1
        print(f"PROGRAM {meta['program_digest'][:16]}... T {meta['rows']}, S {meta['steps']}, {meta['bytes']} bytes"
              + (f", serve-openai's at {a.like}" if a.like else ""), flush=True)
        return 0
    if a.cmd == "pouw":
        doc = pouw(a.serve_dir, a.window, a.out, gate=a.gate, key=a.key_sha512, main=a.main,
                   check=lambda *x: check_challenger(*x, python=a.python), statements=a.statements,
                   live=a.live_evidence, lean_answers=a.lean_answers)
        print(f"POUW-DEV {json.dumps(development_line(doc), sort_keys=True, default=str)}", flush=True)
        return 0
    if a.cmd == "pearlplus":
        doc = pearlplus(a.out, _json(a.served) if a.served else None, _json(a.verify) if a.verify else None,
                        _json(a.unretained) if a.unretained else None, _json(a.volunteer) if a.volunteer else None,
                        gate=a.gate, key=a.key_sha512, main=a.main, coin=a.coin, tie=a.tie)
        print(f"POUW-DEV {json.dumps(development_line(doc), sort_keys=True, default=str)}", flush=True)
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
            doc = pearlplus_period_open(a.piece_dir, a.gate, a.name, a.scheme, a.cubin, a.circuit_record)
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
        return 0 if all(g["verdict"] in ("accept", "missing", "dropped", UNCLAIMED) for g in got.values()) else 1
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
    got = {k: piece(k, a.pieces, run, sampled_proofs=a.window_tie or a.sampled_proofs, key_sha512=a.key_sha512, lane=a.lane)
           for k in keys}
    if a.sampled_exe is not None and not a.sampled_lean:
        ap.error("--sampled-exe is sampled proofs' Lean, which only --sampled-lean runs")
    if a.tree_on and not a.tree:
        ap.error("--tree-on names the branch of the tree --tree names")
    integrity = (lean_sampled_row(a.sampled_proofs, a.serve_run / "private" / "scratch", a.sampled_exe, run)
                 if a.sampled_lean else spot_row(a.sampled_proofs, run))
    rows = {"integrity": integrity, **{k: row for k, (row, _) in got.items()}}
    report, rec, line = record(rows, window=a.window, steps=(steps[0], steps[1]), node=a.node, run=run,
                               servers={k: at for k, (_, at) in got.items()}, product=a.product, deferred=a.deferred,
                               off_node=a.off_node, tree={"tree": a.tree, **({"tree_on": a.tree_on} if a.tree_on else {})}
                               if a.tree else None,
                               served_compare=None if a.served_compare is None else served_compare_line(a.served_compare))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "verdicts.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    if rec is not None:
        (a.out / "block-record.json").write_text(json.dumps(rec, separators=(",", ":"), sort_keys=True) + "\n")
    print("RECORD " + json.dumps(line, sort_keys=True), flush=True)
    return 0 if rec is not None and not line["check"] else 1


if __name__ == "__main__":
    sys.exit(main())
