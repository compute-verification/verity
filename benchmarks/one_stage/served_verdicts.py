"""The verdicts on one served window (`served_verdicts.sh`): the window it audits, the subprotocols after
sampled proofs, and the one record of every verdict (the block record).

    served_verdicts.py window SERVE_DIR [--slots N] [--window K]
    served_verdicts.py gate [--gate CMD] [--lane vllm] [--main FILE]
    served_verdicts.py period open SERVE_DIR --name P --circuits DIR --slots N --max-tokens N [--gate CMD] [--draw-k K]
                                 [--model M]
    served_verdicts.py period close SERVE_DIR [--gate CMD]
    served_verdicts.py program OUT [--model M] [--rows T] [--steps S] [--like SERVE_DIR]
    served_verdicts.py pieces --serve-run DIR --window K --out DIR [--node NAME] [--only COMPONENT]... [--skip COMPONENT]...
                              [--product P] [--deferred WHY] [--served NODE]
    served_verdicts.py pouw SERVE_DIR WINDOW DIR [--gate CMD] [--key-sha512 HEX] [--main FILE] [--python PY]
                                [--statements FILE] [--live-evidence FILE] [--lean-answers DIR]
    served_verdicts.py pearlplus DIR --served FILE --verify FILE [--unretained FILE] [--gate CMD] [--key-sha512 HEX]
                                 [--main FILE]
    served_verdicts.py record --serve-run DIR --window K --sampled-proofs DIR --registration FILE --pieces DIR --out DIR
                              [--node NAME] [--product P] [--deferred WHY] [--network-answer FILE]
    served_verdicts.py public-registration REGISTRATION
    served_verdicts.py advice-line SERVE_DIR --window K --compiler FILE --coins-after T
    served_verdicts.py front-line SESSION
    served_verdicts.py public-lines [--private FILE]
    served_verdicts.py publish OUT --private DIR

What leaves the node names no window and counts none of a window's own: no window's number, no count of the windows closed,
sealed or refused, and no engine step, request count or user of one. ``public-lines`` prints each line of stdin in its
published form (`public_line`), the line in full appended to ``--private`` (the serve dir's ``private/verdicts.txt``), and
``publish`` holds each file of OUT, once the run is done, in full under DIR where it names or counts a window (`publish`),
its public form in its place; what a verifier reads of OUT (`RECEIVED`) it leaves with every key and value but its text's
with ``--received-scanned`` (``kind_scan.py --received OUT`` found every field of it one of the public kinds), the
period's circuit files as written (`AS_WRITTEN`), and holds it whole without it. The class's padded counts, the period's
declared T and S, the bench Build's requests and the verifier's coins and draws are public. ``public-registration`` prints
what the run publishes of a window's advice registration (`public_registration`: refused unless the advice was committed
padded to its compiler's cap), ``advice-line`` the ADVICE-LINE verdict on a serve-openai ``--advice`` run (`advice_line`:
B registered before the first window, every window's advice line before sampled proofs' and the certifier's first coin,
the audited window's registration under B), and ``front-line`` the CERTIFIER-FRONT line from the front's session
(`front_line`: `FRONT_PUBLIC`).

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
every subprotocol, remote's sampled proofs and PoUW alone, remote-network's those and the network certifier. A piece
the product drops isn't run, whatever its command,
and prints a ``DROPPED`` line naming it. ``--deferred WHY`` (`served_verdicts.sh`'s step 4 on a node that defers V3c's
audit) runs no split certifier's piece: its period check, which writes the certifier's answer, reads V3c's period.json
and goes with the audit to the deferral's follow-on, so the piece leaves `REFUSED` naming WHY, and the certifier is
missing here, as sampled proofs is. ``--served NODE`` (that follow-on, which runs the piece) is the server the pieces it
runs audit where it isn't this node: each one's `SERVED` names it, unless the piece wrote its own.

``gate``: before serving with H1's coins from infra's gate (``--gate``, default ``H1_GATE``): refused while the key's lane
(``--lane``) has a period open, which a reserve would join; prints one ``GATE`` line with the gate's latest pin and,
with ``--main``, whether that pin is on the verifier's first-parent line of main (a session pinned off it is refused).

``period open``: the period the served windows join, opened once serve-openai listens and before its first window
reserves (a window opens at its first engine step): its circuit is the Program digest serve-openai declared and wrote
(serve.json's ``protocols.program``, ``h1/window-program.meta.json``), its rows and rounds the server's T and S, and its
circuit files those of ``--circuits`` (``pouw_h1_circuits --program SERVE_DIR/h1 --out DIR --live T,S``'s manifest),
held to that digest, T and S (`period_open_argv`). The live template's file is required. A stratum whose ``k<K>`` file
isn't built yet is left out and named: core's credit refuses a drawn statement at that K (``circuit``), nothing else.
Its fixed length is ``--slots`` requests of ``--max-tokens`` (`credit.LENGTH_FIELD`); its draw K ``--draw-k``, else
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
the drawn calls' statements (``--statements``) and the window's live evidence (``--live-evidence``,
``h1_prove/live.py evidence``'s), so it reaches each check those reach (`credit_inputs`); with ``--lean-answers`` its
verifier runs take the answers another step kept there by key (`answers.Answers`). The verdict says whether the
window meets the two premises of `Security`'s that no check here discharges, ``AcceptedLive`` and ``SaltIsLiveCoin``
(`premises`): each ``{met, why}``, met only where this run's checks show it.

``pearlplus``: PoUW's development piece under POUW=pearlplus (`served_pearlplus.sh`, POUW_CMD's default then), from Pearl+'s
own served run on the card after the window's serve: ``--served`` (`served_passes.py serve`'s document: every pass retained)
and ``--verify`` (its ``verify``'s: Pearl+'s verdict over all the passes, the reference replay of the tiles it draws).
Its verdict is accept when that run served within its caps across at least two pass boundaries, Pearl+'s verdict
accepts, and each pass's salt is the live session's that the serve names (`pearlplus_coins`: the session's tree as the gate
``--gate`` fetches it, the ledger and pins log that gate answers now and ``--main``'s line, held by `core_check.live_salt`
under the key ``--key-sha512`` pins); a run whose coins are an in-process challenger's claims no verdict where nothing
failed (`UNCLAIMED`, `PEARLPLUS_WHY_IN_PROCESS`), and one served under the gate whose coins are not live rejects. Its
document carries the passes, that verdict's line and credit at Pearl's Assumption 1, where its coins came from
(`PEARLPLUS_COINS`) and its shares (`served_shares.pearlplus_shares`), and says why it credits nothing (`PEARLPLUS_WHY`:
its passes are not the window sampled proofs audited).

``record``: sampled proofs' answer from the records audit kept under ``--sampled-proofs`` (``summary.json``,
``window/audit.json``) on the registration ``--registration``, read as `served_integrity.answer` reads it:
`served_verdicts.sh` passes V3c's audit of the window committed as its class circuit's units
(``$D/v3c-w<K>-<class>/audit``, on the window commit's ``window/registration.json``), which reads no advice opening. An
audit that opened the advice (`prove_path/run.sh`'s ``PP_RECORDS=1``, a development check), a ``--registration`` that
isn't there and an answer committing to another registration than it are each an ``error`` row; with ``--deferred``
that audit is left to a follow-on, and sampled proofs' row is ``missing``, naming why; each other
subprotocol's from
``--pieces/<component>/answer.json`` (`piece_row`), else a ``missing`` row naming why. With ``--network-answer``, the
network certifier's row where its period step wrote the case alone, its answer undecided (`certifier_piece.py period
--case-only`), is the period's own answer as the composite's network leg gave it (`network_row`: `composite.py`'s
``network-answer.json``, which must bind the commitment the piece's answer binds), or ``missing`` where there is no
such file (`NO_NETWORK_ANSWER`: that leg didn't accept); any other row of it stands. Each answer is about the server
`piece_server` names: sampled proofs' the record's node, a piece's the one its ``served.json`` names, else the node it ran
on, and the wipe's the one its ``served.json`` names. They go into the window report (``verdicts.json``, `row_of`'s rows,
with each one's server) and from it the block record (``block-record.json``, `block_record.from_period`, held by
`block_record.check`), which composes only the answers about the record's node and carries the others as ``elsewhere``;
an answer that names no server isn't carried. An accepted answer whose outputs are not its leg's certificate
(`block_record.leg`) is its subprotocol's error. `block_record` carries only an accepted subprotocol's certificate, so
where one about this node rejects or errs the composed verdict is ``reject`` and no block record is written; it is
``accept`` only when every subprotocol accepted here.
Those are the subprotocols of ``--product`` (default onsite; `block_record.PRODUCTS`): each one the
product drops is ``dropped`` in the report (`dropped_row`, whatever its piece left), not run and composed into no
verdict, so remote's verdict is ``accept`` when sampled proofs and PoUW accepted here, whatever the onsite pieces.
The report's ``composed`` and the block record's ``composed.onsite`` give that verdict with its reason (`said`), one clause
a subprotocol: its verdict and, unless it accepted here, why; an accept here with what it holds beyond its verdict
(`terms`: when its weights were held and their share of the checkpoint, where its profile says, and each δ of the block
record's at which its quantity is not established); PoUW's development piece reads ``development run, credits nothing``
with core's refusal of the window's credit. Prints one ``RECORD`` line, naming the product, each subprotocol's verdict
(the development piece's `DEV_VERDICT`), each missing one and why, each dropped one, each rejecting one, each one
elsewhere, and the reason. The report and the line say where they were composed (``composed_on``: the host, whether it
is the record's node, and what it read, `COMPOSED_FROM`, or off that node `COPIED_FROM`), and with ``--deferred`` what
the run left to a follow-on and why (``deferred``: `served_verdicts.sh`'s step 6 on a node short of V3c's audit's
memory), which composes the final record on another host, the record's node still ``--node``.
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _p in (Path(__file__).resolve().parent, ROOT / "examples" / "h200" / "interfaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

#: the window report's format, kept from the all-protocols runner whose report it was
REPORT_FORMAT = "verity/all-protocols-window/v1"
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
DEV_WHY = ("development run, credits nothing: H1's coins come from vy-challenger-1's gate under the lane's development key, "
           "and a window counts for credit only under the lane's own key; core's credit also needs the window in a period "
           "whose open commits its draw K, rows and weights")
_PEARLPLUS_UNTIED = ("development run, credits nothing: Pearl+ (pearl-c-h100-v1-h3) served its own seeded load on this card "
                     "after the window's serve, the BF16 checkpoint's linears as its FP8 product, so its passes are not the "
                     "window sampled proofs audited and nothing ties them to that window's Commit")
#: why the Pearl+ piece's verdict (`pearlplus`) credits nothing, its epoch's salt coin a live session's
PEARLPLUS_WHY = (f"{_PEARLPLUS_UNTIED}; its epoch's salt coin is a live session's on vy-challenger-1, under the lane's "
                 "development key; and an ACCEPT is the reference replay's (verify_run.py), not PouwCheck's ZK proof")
#: the same where its epoch's coins are an in-process challenger's: the piece then claims no verdict (`UNCLAIMED`)
PEARLPLUS_WHY_IN_PROCESS = (f"{_PEARLPLUS_UNTIED}; its epoch's coins are an in-process challenger's, which the serving "
                            "process holds, so it claims no verdict: the reference replay's (verify_run.py) ACCEPT under "
                            "coins the prover holds shows nothing")
#: the Pearl+ piece's verdict where its epoch's coins are an in-process challenger's and nothing failed
UNCLAIMED = "unclaimed"
#: where Pearl+'s epoch coins and the replay's draws come from (`pearlplus`), its epoch's salt coin a live session's
PEARLPLUS_COINS = {"epoch": "a salt-only session on vy-challenger-1, reserved through the lane's gate before the engine "
                            "started and opened by the install on its salt epoch's root; each pass's salt is held to it by "
                            "core_check.live_salt through the auditor's signed ledger, the host's pins log and main's "
                            "first-parent line (`pearlplus_coins`)",
                   "draws": "the Lean verifier's IO.getRandomBytes in verify_run.py on the serving node, one source for the "
                            "run, taken once every pass's roots are read and recorded in the passes' coin.json before any "
                            "replay; the piece puts that record in the evidence store the moment it is drawn and holds the "
                            "verdict to it (`coin`, `pearlplus_coin`)"}
#: the same where its epoch's coins are an in-process challenger's
PEARLPLUS_COINS_IN_PROCESS = {**PEARLPLUS_COINS,
                              "epoch": "an in-process challenger of epoch 0 (verity_bench's protocols_config, "
                                       "pouw.challenged), held by the serving process"}
#: `served_step.coins_of`'s ``source`` for a live session's coins
LIVE_SOURCE = "challenger"
#: the evidence store's kind for the verifier's coin record as `coin` publishes it: the record's own schema
#: (`verify_run.COIN_SCHEMA`)
COIN_KIND = "verity-pouw/pearl-plus-coin/v1"
#: `coin`'s copy of the coin record and the store's answer to its put, in the piece's directory
COIN_RECORD, COIN_PUT = "coin.json", "coin-put.json"
#: the seconds `coin` gives each ``research data put`` before naming it unanswered
PUT_TIMEOUT_S = 300
#: what `record` composes from, until it runs on an auditor-side host reading each party's own answer
COMPOSED_FROM = ("each piece's answer file as the served node wrote it, not the party that produced it: an edit to that "
                 "file on the node changes its row")
#: what `record` composes from off the served node (a deferral's follow-on): the prover's copy of those files
COPIED_FROM = ("a prover-side copy of each piece's answer file as the served node wrote it, not the party that produced it: "
               "an edit to that file on the node or to its copy changes its row")
#: a development piece's verdict in the ``RECORD`` line, in place of its row's ``missing``
DEV_VERDICT = "development run, credits nothing"
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
#: what a published line says in place of a JSON object cut short that names a window's field
CUT = "{…} (cut short; in full in the serve dir's private/verdicts.txt)"
#: a file `publish` reads larger than this is held whole
PUBLISH_MAX = 256 << 20
#: what a verifier on another machine reads of OUT, by glob (``*`` any path below): the certifier's piece, V3c's audit and
#: the window circuit's inputs and counts, each held field by field by the kind scan (`kind_scan.RECEIVED`), which
#: dropping a field would break (their digests and the record's answer are over them as written)
RECEIVED = ("pieces/certifier/*", "v3c-audit/*", "compiler.json", "registration.json", "weights_registration.json",
            "window-circuit-*.json")
#: what of `RECEIVED` the verifier reads as written, which `publish` never reads: the period's circuit files, V3c's and
#: the scan's, each the circuit the verifier composes itself from a template's public Definition, read by its SHA-512,
#: which the kind scan holds to its name and V3c's period.json or the certifier's case to name (`kind_scan.CIRCUITS`)
AS_WRITTEN = ("v3c-audit/circuits/*", "pieces/certifier/circuits/*")


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


def _held(key: Any, value: Any) -> bool:
    if key == "refused":
        return not isinstance(value, str)
    return key in OWN or (key in NUMBERED and _numbered(value))


def public_text(s: str) -> str:
    """``s`` with each window number, step range and window count `WINDOW_TEXT` finds in its published form."""
    for pat, sub in WINDOW_TEXT:
        s = pat.sub(sub, s)
    return s


def public_json(o: Any) -> Any:
    """``o`` with no field of a window's own (`OWN`, `NUMBERED`) and each string in its `public_text` form."""
    if isinstance(o, Mapping):
        return {public_text(k) if isinstance(k, str) else k: public_json(v) for k, v in o.items() if not _held(k, v)}
    if isinstance(o, list):
        return [public_json(v) for v in o]
    return public_text(o) if isinstance(o, str) else o


def public_line(line: str) -> str:
    """A verdict line as the run publishes it: each JSON object in it in its `public_json` form, one cut short that names a
    window's field replaced by `CUT` to the line's end, and the rest in its `public_text` form."""
    dec, out, i = json.JSONDecoder(), [], 0
    while (j := line.find("{", i)) >= 0:
        out.append(public_text(line[i:j]))
        try:
            obj, end = dec.raw_decode(line, j)
        except ValueError:
            if _OPEN.match(line, j) and _FIELD.search(line, j):
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


def public_strings(o: Any) -> Any:
    """``o`` with each string value in its `public_text` form, its keys and every other value as they are: the public
    form of what a verifier reads (`RECEIVED`), whose fields the kind scan holds to their kinds but whose text it doesn't
    read."""
    if isinstance(o, Mapping):
        return {k: public_strings(v) for k, v in o.items()}
    if isinstance(o, list):
        return [public_strings(v) for v in o]
    return public_text(o) if isinstance(o, str) else o


def public_doc(text: str, form: Callable[[Any], Any] = public_json) -> str:
    """A published file's text in its public form: a JSON document's ``form`` (`public_json`), else each line's
    `public_line`; ``text`` itself where nothing in it names or counts a window."""
    try:
        doc = json.loads(text)
    except ValueError:
        return "".join(public_line(x) for x in text.splitlines(keepends=True))
    pub = form(doc)
    return text if pub == doc else json.dumps(pub, indent=1) + "\n"


def publish(out: Path, private: Path, received_scanned: bool = False) -> list[str]:
    """Every file under ``out``, the dir that leaves the node, in its public form: one whose path names a window, that
    `PUBLISH_MAX` keeps from being read, or that isn't UTF-8 text (named on stderr), held whole, so nothing unscanned
    stays in ``out``, and one whose text names or counts a window (`public_doc`) held in full, under ``private`` at its
    path under ``out``, its public form left in its place. What a verifier reads (`RECEIVED`) is held whole unless
    ``received_scanned`` (the kind scan found every field of it one of the public kinds), and then keeps every key and
    value but its text's (`public_strings`), which a digest or the record reads as written, and `AS_WRITTEN`'s files are
    left as written, unread. The paths held, in order."""
    held: list[str] = []
    for p in sorted(out.rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        rel = p.relative_to(out).as_posix()
        if received_scanned and any(fnmatch.fnmatchcase(rel, g) for g in AS_WRITTEN):
            continue
        received = any(fnmatch.fnmatchcase(rel, g) for g in RECEIVED)
        dst = private / rel
        private.mkdir(mode=0o700, parents=True, exist_ok=True)
        dst.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        data = p.read_bytes() if p.stat().st_size <= PUBLISH_MAX else None
        if data is None or (received and not received_scanned) or NAMED.search(rel):
            shutil.move(p, dst)
            held.append(rel)
            continue
        text = data.decode(errors="replace")
        if text.encode() != data:
            shutil.move(p, dst)
            held.append(rel)
            print(f"PUBLISH: {rel} held whole, unscanned: not UTF-8 text", file=sys.stderr, flush=True)
            continue
        pub = public_doc(text, public_strings if received else public_json)
        if pub != text:
            dst.write_bytes(data)
            p.write_text(pub)
            held.append(rel)
    return held


def padded(reg: Mapping[str, Any]) -> bool:
    """Whether a window's advice registration says no length of the advice: its advice committed padded to its compiler's
    ``cap_bits`` (`advice_windows.pad`), so its length is the cap and its rows the ones the cap fixes at ``row_bits`` each,
    and no field of the bits per egress window."""
    try:
        cap, row_bits = int(reg["compiler"]["cap_bits"]), int(reg["row_bits"])
        return (int(reg["binding"]["advice"]["bits"]), int(reg["rows"])) == (cap, -(-cap // row_bits)) \
            and "bits_by_window" not in reg
    except (KeyError, TypeError, ValueError):
        return False


def public_registration(reg: Mapping[str, Any]) -> dict[str, Any]:
    """What the run publishes of a window's advice registration: the registration as committed, refused (ValueError) unless
    it is `padded`, with no count or refusal of its requests and no window's number (`public_json`)."""
    if not padded(reg):
        raise ValueError("the window's advice registration states the advice's own length: its advice wasn't committed "
                         "padded to its compiler's cap (advice_windows.pad)")
    return public_json({k: v for k, v in reg.items() if k not in ("requests", "compiled", "refused")})


#: serve-openai ``--advice``'s public records in its serve dir (`verity_vllm.pipeline.serve_advice`): B's registration,
#: written before its first window, and each closed window's advice line
ADVICE_REGISTRATION, ADVICE_LINES = "advice_registration.json", "advice.jsonl"


def advice_line(serve_dir: Path, window: int, compiler: Mapping[str, Any], coins_after: float) -> tuple[bool, str]:
    """The ADVICE-LINE verdict on a serve-openai ``--advice`` run: B as the server registered it before its first window
    (`ADVICE_REGISTRATION`) is ``compiler``, the one ``prove_path.py compiler`` recomputes from serve.json, the builder
    of the registration log it committed (B's inner digest, the window advice's format, ℓ_a), and that log's last entry
    the run's day's declaration, B's digest its program (`serve_advice.declared_log`: one program a day); every closed window
    (``windows.jsonl``) has its line (`ADVICE_LINES`), in window order, each under B with ℓ_a and the rows ℓ_a fixes as
    the advice's length and row count, written before ``coins_after`` (unix seconds: before sampled proofs and the network
    certifier draw any coin); and window ``window``'s registration, the one the records audit opens, is its line's root
    under B, its advice committed padded to ℓ_a (`padded`). -> (passed, the line, naming no window)."""
    from verity.core.protocols import registration as REG

    why: list[str] = []
    reg, lines = _json(serve_dir / ADVICE_REGISTRATION), _lines(serve_dir / ADVICE_LINES)
    if reg is None:
        return False, (f"ADVICE-LINE: FAIL: no {ADVICE_REGISTRATION}: the server registered no B (serve-openai without "
                       "--advice)")
    b, cap = reg.get("compiler") or {}, reg.get("cap_bits")
    if (b.get("digest"), b.get("cap_bits"), b.get("inner")) != (compiler.get("digest"), compiler.get("cap_bits"),
                                                                 compiler.get("inner")):
        why.append("the server's B isn't the compiler prove_path.py recomputes from serve.json")
    if cap != b.get("cap_bits"):
        why.append("its ℓ_a isn't its compiler's cap")
    day = reg.get("day")
    try:
        composite, log = REG.log_of(reg["registration"]["log"])
        view = REG.check(REG.Auditor(), composite, log).view()
        want = {"kind": "builder", "digest": (b.get("inner") or {}).get("digest"), "advice": reg.get("advice"), "cap": cap}
        if view.builder != want or view.head != reg["registration"].get("head"):
            why.append("its registration log's builder isn't B's (its inner digest, its advice's format and ℓ_a)")
        last = log[-1] if log else None
        if isinstance(day, bool) or not (isinstance(last, REG.Declaration) and last.day == day
                                         and last.program == b.get("digest")):
            why.append("its registration log's last entry isn't the run's day's declaration of B as the day's program")
    except (KeyError, TypeError, ValueError, REG.Refused) as e:
        why.append(f"its registration log doesn't check ({type(e).__name__})")
    closed = sorted({int(x["window"]) for x in _lines(serve_dir / "windows.jsonl")})
    order = [int(x["window"]) for x in lines]
    if order != sorted(set(order)):
        why.append("the advice lines aren't one a window in window order")
    if sorted(order) != closed:
        why.append("a closed window has no advice line")
    rows = -(-int(cap) // int(reg.get("row_bits") or 1)) if isinstance(cap, int) else None
    for x in lines:
        if x.get("refused") or x.get("empty"):
            continue
        if (x.get("compiler") or {}).get("digest") != b.get("digest") or (x.get("advice") or {}).get("bits") != cap \
                or x.get("rows") != rows:
            why.append("an advice line names another compiler than B, or a length or row count other than ℓ_a's")
            break
    t0 = reg.get("t")
    if any(not (isinstance(x.get("t"), (int, float)) and x["t"] < coins_after) for x in lines):
        why.append("an advice line was written after sampled proofs' or the network certifier's first coin")
    if not isinstance(t0, (int, float)) or any(isinstance(x.get("t"), (int, float)) and x["t"] < t0 for x in lines):
        why.append("an advice line was written before B's registration")
    mine = next((x for x in lines if int(x["window"]) == int(window)), None)
    wreg = _json(serve_dir / "advice" / f"w{int(window)}" / "registration.json")
    if mine is None or mine.get("refused") or mine.get("empty") or wreg is None:
        why.append("the audited window has no committed advice")
    elif (wreg["binding"]["advice"]["root"], wreg["binding"]["compiler"]) != (mine["advice"]["root"], mine["compiler"]) \
            or mine["compiler"] != {"name": b.get("name"), "digest": b.get("digest")}:
        why.append("the audited window's registration isn't its advice line's root under B")
    elif not padded(wreg) or wreg["compiler"].get("cap_bits") != cap:
        why.append("the audited window's advice wasn't committed padded to ℓ_a: its registration states its own length")
    ok = not why
    return ok, (f"ADVICE-LINE: B (compiler digest {str(b.get('digest'))[:16]}…, its advice {reg.get('advice')}, ℓ_a {cap} "
                f"bits) registered by the server once, before its first window, the registration log's builder and the "
                f"compiler prove_path.py recomputes from serve.json, and declared day {day}'s one program; every closed "
                f"window's advice committed by the server "
                f"as it served, padded to ℓ_a, its line (the advice's root, ℓ_a and the rows ℓ_a fixes) out before sampled "
                f"proofs' and the network certifier's first coin; "
                f"the audited window's registration its line's root under B, at ℓ_a: {'PASS' if ok else 'FAIL'}"
                + (f" ({'; '.join(why)})" if why else ""))


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
           product: str = "onsite", deferred: str | None = None,
           served: str | None = None) -> dict[str, dict[str, Any]]:
    """Each piece of `PIECES` run where ``<COMPONENT>_CMD`` names it, under `timeout` at its limit, else named missing;
    one line each on stdout, its verdict the one `piece` gives the record. ``only`` and ``skip`` name components: those
    alone, or all but those, so one piece can run beside the others (`served_verdicts.sh` starts the certifier's early).
    A piece ``product`` drops isn't run, whatever its command (`dropped_row`). ``deferred``: why the split certifier's
    period check is a deferral's follow-on's, so its piece isn't run and leaves `REFUSED` naming that; ``served``: the
    server each piece run here audits, which its `SERVED` names unless the piece wrote its own."""
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
        if deferred and comp == "certifier":
            (out / comp).mkdir(parents=True, exist_ok=True)
            (out / comp / REFUSED).write_text(f"deferred: {deferred}\n")
            got[comp] = {k: v for k, v in piece_row(key, out).items() if k in ("verdict", "why")}
            print(f"{name.upper()} {json.dumps(got[comp], sort_keys=True)}", flush=True)
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
        if served and not (d / SERVED).exists():
            (d / SERVED).write_text(json.dumps({"server": served}) + "\n")
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
    when it didn't run), its load, passes and the rounds dropped before the coin, Pearl+'s verdict's credit (its credited
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
                **{k: doc.get(k) for k in ("scheme", "load", "passes", "dropped", "served_tokens", "generate_s")},
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


def gate_ready(gate: str, lane: str, main: Path | None = None) -> tuple[bool, str]:
    """Whether this run may reserve its windows' sessions at `gate` (the ``gate`` subcommand), and its ``GATE`` line."""
    from verity_vllm.protocols.pouw.h1 import window as PHW

    ledger = _gate_out(gate, "ledger").decode().splitlines()
    pins = [json.loads(x) for x in _gate_out(gate, "pins").decode().splitlines() if x.strip()]
    pin, period = (pins[-1] if pins else {}), PHW.open_period(ledger, lane)
    on = None if main is None else pin.get("commit") in main.read_text().split()
    where = "" if on is None else f", {'on' if on else 'NOT on'} the verifier's first-parent line of main"
    line = (f"GATE {'refused' if period else 'ready'}: lane {lane}'s open period "
            f"{period and period.get('period')}{' (a reserve would join it)' if period else ''}; the ledger's "
            f"{len(ledger)} entries; pinned {str(pin.get('commit'))[:12]} (main {str(pin.get('main'))[:12]}, "
            f"{pin.get('pinned_utc')}){where}")
    return period is None, line


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


def credit_inputs(serve_dir: Path, statements: Path | None = None, live: Path | None = None) -> dict[str, str]:
    """What core's credit reads of the window beyond its record and session, as ``--check-challenger``'s flags: the
    declared Program serve-openai wrote (``--descriptor``, `pouw_h1.PROGRAM`), the period ``period open`` opened
    (``--period``, `PERIOD`), the drawn calls' statements (``--statements``) and the live evidence (``--live-evidence``,
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
    """`pouw_h1_gpu.py --check-challenger` (run by `python` in integrations/vllm, so ``tests`` is that package's) of the
    window's record `calls` against its session's tree `tree`, with credit's `inputs` (`credit_inputs`); its
    ``pouw_h1_challenger.json``."""
    out.mkdir(parents=True, exist_ok=True)
    argv = [python, "-m", "tests.protocols.pouw_h1_gpu", "--check-challenger", str(calls), str(tree), "--key-sha512", key,
            "--ledger", str(ledger), "--pins", str(pins), *(["--main", str(main)] if main else []), "--out", str(out),
            *(x for kv in inputs.items() for x in kv)]
    p = subprocess.run(argv, cwd=ROOT / "integrations" / "vllm", capture_output=True, text=True, check=False)
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
    `check` is `check_challenger` with its Python bound, given credit's inputs (`credit_inputs` of `serve_dir`, the drawn
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
    from verity.core.protocols.pouw import core_check as CC
    from verity.core.protocols.sampled_proofs import remote
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


def coin(record: Path, out: Path, run: str | None, research: Sequence[str] | None = None,
         put: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """The verifier's coin record `record` (`served_passes.py verify`'s ``DIR/coin.json``, written before any replay)
    published the moment it is drawn: copied to ``out/coin.json``, then put in the evidence store as `COIN_KIND` through
    `research` (default the ``research`` command, else this Python's ``-m research``), its meta the run `run` and the run
    commitment the coin was drawn on. The put asks ``--preserve`` first; where that fails (a job's node holds no store
    write key) the record goes in the node's own store, under the same content-addressed id, which a put of the run's
    custody copy of ``out/coin.json`` under the same kind and meta preserves. A piece publishes one coin: a record other
    than the one ``out/coin.json`` already holds is refused (ValueError), as is one that is not the verifier's. Writes
    and returns ``out/coin-put.json``: ``{sha512, commitment, source, run, art, preserved, error}``, ``art`` the id (None
    where no put answered one) and ``error`` what the preserving put said when it didn't preserve."""
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
        found = shutil.which("research")
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
    got = {"sha512": hashlib.sha512(raw).hexdigest(), "commitment": rec["commitment"], "source": rec["source"], "run": run,
           "art": art, "preserved": preserved, "error": None if preserved else f"research data put --preserve: {said}"[:400]}
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


def pearlplus(out: Path, served: Mapping[str, Any] | None, verified: Mapping[str, Any] | None,
              unretained: Mapping[str, Any] | None = None, volunteered: Mapping[str, Any] | None = None, *,
              gate: str | None = None, key: str | None = None, main: Path | None = None,
              coin: Path | None = None) -> dict[str, Any]:
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
    rejects. The piece names its load (``load``: the requests, their prompt and output tokens, and the tokens a step)."""
    import served_shares as SS

    out.mkdir(parents=True, exist_ok=True)
    reasons = []
    source = None if served is None else (served.get("coins") or {}).get("source", "in-process")
    live = bool(gate) if served is None else source == LIVE_SOURCE
    if served is None:
        reasons.append("no served run: served_passes.py serve wrote no document")
    elif not served.get("ok"):
        reasons.append(f"the served run crossed fewer than two pass boundaries or ran over its caps (passes over their "
                       f"caps: {served.get('over_caps')})")
    answer = (verified or {}).get("answer") or {}
    if verified is None:
        reasons.append("no verdict: served_passes.py verify wrote no document")
    elif answer.get("verdict") != "accept":
        reasons.append(f"Pearl+'s verdict rejects the run: {answer.get('reason')}"[:800])
    coins: dict[str, Any] = dict(PEARLPLUS_COINS if live else PEARLPLUS_COINS_IN_PROCESS)
    if served is not None and live:
        coins["check"] = pearlplus_coins(served, out, gate, key, main)
        if coins["check"]["reason"] is not None:
            reasons.append(f"coins: {coins['check']['reason']}")
    elif served is not None and gate:
        reasons.append(f"coins: {source}, not a live session on the gate, so nothing checks them")
    doc: dict[str, Any] = {"piece": "pearlplus", "credited": False, "why": PEARLPLUS_WHY if live else PEARLPLUS_WHY_IN_PROCESS,
                           "coin_source": source, "coins": coins, "scheme": None, "arm": None, "passes": None, "load": None,
                           "dropped": None, "line": None, "credit": None, "coin": None, "give_up": None, "shares": None}
    if served is not None:
        doc.update(arm=served.get("arm"), scheme=(served.get("arm") or "").partition(":")[2] or None,
                   passes=sum(1 for p in served.get("passes") or () if p.get("units")),
                   load={**{k: served.get(k) for k in ("requests", "prompt_tokens", "output_tokens")},
                         "max_num_batched_tokens": (served.get("engine") or {}).get("max_num_batched_tokens")},
                   **{k: served.get(k) for k in ("spec", "run", "served_tokens", "generate_s", "prompt_tokens", "output_tokens")})
    if verified is not None:
        doc.update(line=verified.get("line"), credit=(answer.get("public") or {}).get("work"), dropped=verified.get("dropped"))
    if coin is not None:
        doc["coin"] = pearlplus_coin(coin, answer)
        if doc["coin"]["reason"] is not None and answer.get("verdict") == "accept":
            reasons.append(f"coin: {doc['coin']['reason']}")
    if volunteered is not None:
        doc["give_up"] = give_up(volunteered)
    if served is not None and served.get("ok"):
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
    from verity.core.protocols.pouw import core_check as CC

    try:
        CC.answer_of(a)
    except CC.Refused as e:
        raise ValueError(f"not a core answer: {e}") from None
    return dict(a)


def epsilon(a: Mapping[str, Any]) -> str | None:
    """A sampled protocol's bound on wrong units (``public.bound``) over its checked level's size (`Core.Integrity`), or
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


def sampled_proofs_row(audit_dir: Path, run: str | None, registration: Path) -> dict[str, Any]:
    """Sampled proofs' row from the records audit kept under ``audit_dir``, on ``registration`` (module docstring): an
    ``error`` where that file isn't a registration, or where the answer commits to another (an accept's commitment is
    the audit record's own registration's digest)."""
    import served_integrity as SI
    from verity.core.protocols.sampled_proofs.one_stage import registration as R

    try:
        reg = json.loads(registration.read_text())
        if not isinstance(reg, dict):
            raise ValueError(f"{registration} is not a JSON object")
        a = SI.answer(audit_dir, reg)
        if a.get("commitment") != R.digest(reg):
            raise ValueError(f"the audit's answer commits to another registration than {registration}")
        return row_of(INTEGRITY_OWNER, a, audit_dir, run)
    except (OSError, RuntimeError, ValueError) as e:
        return {"verdict": "error", "owner": INTEGRITY_OWNER, "reason": f"{type(e).__name__}: {e}"[:2000]}


def _json(path: Path) -> Any:
    """``path``'s JSON, None where there's no such file; one that isn't JSON raises ValueError."""
    return json.loads(path.read_text()) if path.exists() else None


def piece(key: str, pieces_dir: Path, run: str | None = None) -> tuple[dict[str, Any], tuple[str | None, str]]:
    """A subprotocol's `piece_row` and `piece_server`; a piece that left a file that isn't JSON is its ``error``, about no
    server."""
    try:
        return piece_row(key, pieces_dir, run), piece_server(key, pieces_dir)
    except ValueError as e:
        return ({"verdict": "error", "owner": PIECES.get(key, WIPE)[2],
                 "reason": f"its piece left a file that isn't JSON: {type(e).__name__}: {e}"[:2000]}, (None, "real"))


def piece_row(key: str, pieces_dir: Path, run: str | None = None) -> dict[str, Any]:
    """A subprotocol's runner row from its piece's answer; a development piece's is ``missing`` with its document when it
    accepts or claims no verdict (`UNCLAIMED`; it credits nothing, for the document's ``why``, else `DEV_WHY`), else
    ``reject``; a piece that ran (`RAN`) and wrote neither is its ``error``, unless it
    refused to run (`REFUSED`); otherwise ``missing``, naming why."""
    _name, comp, owner, what = PIECES.get(key, WIPE)
    d = pieces_dir / comp
    f, dev, ran = d / "answer.json", d / DEVELOPMENT, _json(d / RAN)
    if not f.exists() and dev.exists():
        doc = _json(dev)
        if not isinstance(doc, dict):
            return {"verdict": "error", "owner": owner, "reason": f"its piece's {DEVELOPMENT} is not a JSON object"}
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


#: the network certifier's row where its period step wrote the case alone and the composite's network leg wrote no answer
NO_NETWORK_ANSWER = ("its period step wrote the case alone (--case-only), and the composite's network leg didn't accept: "
                     "not established, the composite's verdict stands")


def network_row(row: Mapping[str, Any], answer: Path, pieces_dir: Path, run: str | None = None) -> dict[str, Any]:
    """The network certifier's row ``row`` (`piece_row`'s) under ``record --network-answer``: where its period step left
    it undecided (``--case-only``), the period's own answer as the composite's network leg gave it, the file
    ``answer`` (`composite.py`'s ``network-answer.json``, written only where that composite accepted), which must bind
    the commitment the piece's answer binds; ``missing`` (`NO_NETWORK_ANSWER`) where there is no such file. Any other
    row stands: a piece that refused, rejected or erred is its own verdict."""
    owner = PIECES["communication"][2]
    if row.get("verdict") != "undecided":
        return dict(row)
    if not answer.is_file():
        return {"verdict": "missing", "owner": owner, "why": NO_NETWORK_ANSWER}
    try:
        got = row_of(owner, json.loads(answer.read_text()), pieces_dir / PIECES["communication"][1], run)
    except ValueError as e:
        return {"verdict": "error", "owner": owner,
                "reason": f"the composite's network answer: {type(e).__name__}: {e}"[:2000]}
    if got["commitment"] != row.get("commitment"):
        return {"verdict": "error", "owner": owner,
                "reason": "the composite's network answer binds another commitment than the period step's case"}
    return got


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
        else:
            s = f"{v}: {r.get('reason')}"
        out.append(f"{NAMES[k]}: {s}")
    return "; ".join(out)


def record(rows: Mapping[str, Mapping[str, Any]], *, window: int, steps: tuple[int, int], node: str,
           run: str | None, servers: Mapping[str, tuple[str | None, str]] | None = None, product: str = "onsite",
           deferred: str | None = None) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """The window report over ``rows`` (by runner name), its block record (None when a subprotocol about this node
    rejected, erred or was undecided), and the ``RECORD`` line's summary, composed for ``product``: each subprotocol it
    drops is its `dropped_row`, about no server. ``servers``: each piece's `piece_server`, by runner name; sampled proofs'
    answer is about ``node``, and a piece's that names no server isn't carried. ``deferred``: what the run left to a
    follow-on and why, which the report and the line carry."""
    from verity_e2e_demo import block_record as BR

    legs = {v: k for k, v in BR.RUNNER_KEYS.items()} | {"wipe": "wipe"}
    drops = [k for k in rows if legs[k] not in BR.PRODUCTS[product]]
    at = ({k: (None, "real") for k in rows} | dict(servers or {}) | {"integrity": (node, "real")}
          | {k: (None, "real") for k in drops})
    rows = {k: dropped_row(k, product) if k in drops else dict(r) for k, r in rows.items()}
    built: dict[str, dict[str, Any]] = {}
    for k, r in rows.items():
        if r.get("verdict") == "accept" and at[k][0] is not None:
            try:
                built[k] = BR.leg(legs[k], r, server=at[k][0], runs=[], source=at[k][1])
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as e:
                rows[k] = {**r, "verdict": "error", "quantity": None, "epsilon": None,
                           "reason": f"its answer is not the block record's {legs[k]} leg: {type(e).__name__}: {e}"[:2000]}
    host = socket.gethostname()
    where = {"host": host, "served_node": host == node, "from": COMPOSED_FROM if host == node else COPIED_FROM}
    left = {"deferred": deferred} if deferred else {}
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
    named = {"product": product, "verdicts": verdicts, "missing": missing, "elsewhere": elsewhere, "composed_on": where,
             **left, **({"dropped": {NAMES[k]: rows[k]["why"] for k in drops}} if drops else {}),
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
    report["composed"] = rec["composed"]["onsite"] = {"protocol": REPORT_FORMAT, "verdict": rec["composed"]["verdict"],
                                                      "reason": said(rows, at, node, built)}
    line = {"composed": rec["composed"]["verdict"], **named, "reason": report["composed"]["reason"],
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
    cn = sub.add_parser("coin")
    cn.add_argument("out", type=Path)
    cn.add_argument("--record", type=Path, required=True, help="the verifier's coin record (served_passes.py verify's "
                                                               "DIR/coin.json)")
    cn.add_argument("--run", help="the run the coin was drawn in (RESEARCH_RUN_ID), the put's meta")
    rg = sub.add_parser("public-registration")
    rg.add_argument("registration", type=Path)
    al = sub.add_parser("advice-line")
    al.add_argument("serve_dir", type=Path)
    al.add_argument("--window", type=int, required=True)
    al.add_argument("--compiler", type=Path, required=True, help="prove_path.py compiler's output for the serve dir")
    al.add_argument("--coins-after", type=float, required=True,
                    help="unix seconds before sampled proofs and the network certifier draw any coin")
    pl = sub.add_parser("public-lines")
    pl.add_argument("--private", type=Path)
    pb = sub.add_parser("publish")
    pb.add_argument("out", type=Path)
    pb.add_argument("--private", type=Path, required=True)
    pb.add_argument("--received-scanned", action="store_true", help="the kind scan found what a verifier reads of OUT "
                    "(RECEIVED) clean: leave its every key and value but its text's, rather than hold it whole")
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
    p.add_argument("--deferred", help="why the split certifier's period check is a deferral's follow-on's: its piece "
                                      "isn't run here")
    p.add_argument("--served", help="the server the pieces run here audit, where it isn't this node")
    r.add_argument("--sampled-proofs", type=Path, required=True)
    r.add_argument("--registration", type=Path, required=True)
    r.add_argument("--pieces", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--deferred", help="what the run left to a follow-on and why, which the record carries")
    r.add_argument("--network-answer", type=Path, help="composite.py's network-answer.json: the network certifier's "
                   "period answer where its period step wrote the case alone (`network_row`)")
    for q in (p, r):
        q.add_argument("--node", default=socket.gethostname())
        q.add_argument("--product", default="onsite")
    a = ap.parse_args(argv)
    if a.cmd in ("pieces", "record"):
        from verity_e2e_demo import block_record as BR

        if a.product not in BR.PRODUCTS:
            ap.error(f"--product {a.product!r}: not one of {sorted(BR.PRODUCTS)}")
    if a.cmd == "public-registration":
        try:
            print(json.dumps(public_registration(json.loads(a.registration.read_text())), indent=1, sort_keys=True))
        except ValueError as e:
            print(f"public-registration: {e}", file=sys.stderr)
            return 1
        return 0
    if a.cmd == "advice-line":
        ok, line = advice_line(a.serve_dir, a.window, json.loads(a.compiler.read_text()), a.coins_after)
        print(line)
        return 0 if ok else 1
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
        held = publish(a.out, a.private, a.received_scanned)
        if held:
            print(f"PUBLISH: {a.out}'s files that name a window or count its own are held in full in {a.private}, "
                  "their public form in their place", flush=True)
        if not a.received_scanned and any(fnmatch.fnmatchcase(h, g) for h in held for g in RECEIVED):
            print(f"PUBLISH: what a verifier reads of {a.out} (RECEIVED) wasn't found clean by the kind scan: held whole "
                  f"in {a.private}", flush=True)
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
            ok, line = gate_ready(a.gate, a.lane, a.main)
        except (OSError, RuntimeError, ValueError) as e:
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
                        gate=a.gate, key=a.key_sha512, main=a.main, coin=a.coin)
        print(f"POUW-DEV {json.dumps(development_line(doc), sort_keys=True, default=str)}", flush=True)
        return 0
    if a.cmd == "coin":
        try:
            got = coin(a.record, a.out, a.run or None)
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
        print(f"COIN drawn on run commitment {got['commitment'][:16]}..., record sha512 {got['sha512'][:16]}...: {where}",
              flush=True)
        return 0
    if a.cmd == "pieces":
        a.out.mkdir(parents=True, exist_ok=True)
        got = pieces(a.serve_run, a.window, a.out, node=a.node, only=a.only, skip=a.skip, product=a.product,
                     deferred=a.deferred, served=a.served)
        return 0 if all(g["verdict"] in ("accept", "missing", "dropped", UNCLAIMED) for g in got.values()) else 1
    steps = pick(a.serve_run, 1 << 30, a.window)["steps"]
    run = os.environ.get("RESEARCH_RUN_ID")
    keys = [*PIECES, "wipe"]
    got = {k: piece(k, a.pieces, run) for k in keys}
    if a.network_answer is not None:
        row, at = got["communication"]
        got["communication"] = (network_row(row, a.network_answer, a.pieces, run), at)
    integrity = ({"verdict": "missing", "owner": INTEGRITY_OWNER, "why": f"deferred: {a.deferred}"[:2000]} if a.deferred
                 else sampled_proofs_row(a.sampled_proofs, run, a.registration))
    rows = {"integrity": integrity, **{k: row for k, (row, _) in got.items()}}
    report, rec, line = record(rows, window=a.window, steps=(steps[0], steps[1]), node=a.node, run=run,
                               servers={k: at for k, (_, at) in got.items()}, product=a.product, deferred=a.deferred)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "verdicts.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    if rec is not None:
        (a.out / "block-record.json").write_text(json.dumps(rec, separators=(",", ":"), sort_keys=True) + "\n")
    print("RECORD " + json.dumps(line, sort_keys=True), flush=True)
    return 0 if rec is not None and not line["check"] else 1


if __name__ == "__main__":
    sys.exit(main())
