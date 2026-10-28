"""The auditor host's challenger service: `serve.py` runs only as the `auditor` user, from a pinned
tree, one session a window, each recorded in a signed append-only ledger before its first coin.

    vy-challenger-gate reserve --peer NAME --scheme S --families JSON [--epoch E] [--minutes M] [--for LANE]
    vy-challenger-gate open --session S --root R
    vy-challenger-gate period-open --name P --cap N --circuit SHA512 [--draw-k K] [--rows T] [--model M | --weights-root W]
                                   [--circuit-files JSON] [--length JSON] [--lane LANE] | period-close --name P
    vy-challenger-gate model-register --model M --circuit SHA512 --descriptor FILE --checkpoint DIR
    vy-challenger-gate model-key --model M --circuit SHA512 [--weights-root W]
    vy-challenger-gate certifier-rehearse --peer NAME --port PORT --minutes M [--tamper late|extra|drop|token]
                                          [--requests N] [--tokens N] [--height H]
                                          ([--in-flight N] [--profile planned|served] [--temperature T] [--top-p P]
                                           | --traffic varied [--rate R] [--prompt-range LO:HI | --prompts < PROMPTS]
                                             [--seed S] [--eos-ids ID,...] [--compact B] [--margin M])
    vy-challenger-gate warden-rehearse --peer NAME --port PORT --minutes M [--requests N] [--tokens N] [--height H]
                                       [--temperature T] [--top-p P]
                                       [--scan [--bucket-ms MS] [--prompts [--window-per-group]] < PROMPTS]
    vy-challenger-gate weights-write [--shape h200|tiny]
    vy-challenger-gate load-relay --peer NAME --port PORT --sink-port PORT --program SHA512 --minutes M
                                  [--shape h200|tiny] [--streams 4] [warden-rehearse's flags]
    vy-challenger-gate status SESSION|RUN | ledger | pins | models | pubkey | fetch SESSION|RUN

- **Who reaches it.** Every key in `research`'s authorized_keys on the host is forced to `vy-challenger-gate` (`command=`),
  which is this file's `gate` as root through one sudoers line. A lane's own key is forced with the lane's name
  (`vy-challenger-gate LANE`, from `/etc/vy/challenger-gate-keys`, `vy-challenger-setup`), and only such a key reserves and
  opens a window: the reserve records that name as `by`, its prover. A key forced with no name (the shared lane key, which
  the control pod's publisher holds) reads only: `status`, `ledger`, `pins`, `pubkey`, `fetch`. A lane names a window and gets a
  session; it never names code. The admin account `vy-admin` runs the same gate (`sudo vy-challenger-gate VERB ...`, `by` vy-admin),
  and it alone may reserve a window for a lane that has no key of its own (`--for LANE`: the reserve records `for` and
  keeps `by` vy-admin, and the window's prover is LANE). Nothing else runs as `research` there, and nothing runs as
  `auditor` but the session unit, `reserve`, `admit` and a rehearsal's unit (the certifier's or the warden's).
- **What runs.** `/opt/vy-challenger/current` -> `/opt/vy-challenger/<commit>`, a root-owned `git archive` of that commit
  (`pin.sh`); the commit is the directory's name, and every entry names it (`pinned`). pin.sh pins only a commit on
  origin/main's first-parent history, which it fetches from GitHub itself, and appends `pin_line(commit, main, utc)` to
  the pins log (`/opt/vy-challenger/pins.jsonl`, root's, `chattr +a`; `record_pin`) before it moves `current`.
  A reserve or an open is refused unless that log records this tree's pin (`pins_refusal`, the verifier's `pins_reason`),
  and each open seals the log as it stands (`pins: {lines, sha512}`), so `pinned_session` can hold the session to a pin
  the log records, and that pin to the auditor's own first-parent list of main. The key (`/var/lib/vy-challenger/key`,
  serve.py's layout) and every session's `out/` are `auditor`'s, in a directory only `auditor` and root can enter.
- **A window** is opened in two steps, each a signed entry appended under the ledger's lock (or refused) and fsynced.
  `reserve` takes the request (peer, scheme, families, epoch, minutes) and appends the session's `reserve` entry with its
  `started_utc` and `expires_utc` (start + minutes); nothing serves yet. From that entry on the window is one of its
  lane's open period, counted toward the period's cap. `open --session S --root R` (R: 128 lowercase hex digits, for
  SALT-WIRE's `window_open_root(weights' root, round-0 request table's root, S)`, which the verifier recomputes) is
  refused unless S is reserved, not yet opened, unexpired, the opener's own (its reserve's `by` or `for`) and its
  period still open, and R is no open's root yet (v1's `commit` and one-step v2 opens included): one session a root,
  ever. `admit` appends the `open` entry, which binds S (`session`), R (`window.root`) and the reserve (`reserve`, its
  `seq`), carries the reserve's request, names its opener (`opened_by`) and takes a port of its peer's certifier coin
  link (the coins' path below), before `serve.py` starts, so
  before its first coin. A reserve not opened by its expiry is its period's refused window. Its `host` is the string the
  challenger's own record signs (`<host>.<session>`), which ties `record.json` to it. A window's time is the auditor's,
  never the requester's: its entries' `seq` and times. `serve.py` gets the root (`--root`: a salt opens only after it)
  and the session's secret (`--secret-file`), which the gate returns to the opener alone and which the client's hello
  must carry; the open entry keeps its SHA-512, `fetch` never ships it, and the session deletes it when it ends. The
  open's answer also carries the session's coin commitment root (`commitment`, from the record serve.py signs before it
  listens; `status` gives it too), so the serving side checks the hello's commitment against it before its first coin.
  Before the first reserve the ledger held one-step v2 opens (`--commit`), which still verify; none follows a reserve.
- **An `end` entry** follows when `serve.py` exits: its exit code, whether its client closed it, the SHA-512 of each file of
  `out/` (which seals the release log), how many coins each family released (`released`, counted from the release log under
  the signed record's commitment; `complete` when every family released all its coins, so a period can refuse a short
  release), the leaf the challenger's record was signed at (`record_leaf`), its commitment root (`commitment`) and how
  many hellos serve.py refused before it accepted one (`refused_hellos`, each in `out/hellos.jsonl`, which `files` seals). A session with an `open` and no `end` never
  ended (its host stopped): a refused window to the verifier, like one whose `closed` or `complete` is false.
- **The coins' path**: the device that runs a server's network certifier produces its live coins. `serve.py` listens
  on this host's 127.0.0.1 at the open's port and opens no tunnel of its own. That port is one of the peer's certifier
  coin link (`coin_ports`: the peer's line `PEER LO-HI` of /etc/vy/coin-ports, within 7811..7899), which
  carries it to the same port of the peer's loopback, where the prover's client connects; a reserve for a peer with no
  such line is refused. The certifier on this host owns that link and counts it (`coin_link.py`,
  `verity.core.protocols.certifier.coin_link`): `run` starts it once serve.py listens and stops it once serve.py exits
  (`coin_link_argv`, `stop_link`), and a link that refuses to start, doesn't listen within `START_SECONDS` (it is
  killed: `first_line`) or ends first (its tunnel did) ends the session (`watch_link`). It listens on another port of this host's loopback, opens `ssh -N -R` from the peer's port to it,
  straight to the peer's own sshd, relays each connection to serve.py re-cut into cells and re-clocked on its grids,
  exempts only the coin openings that open against the session's committed coin root, counts everything else, and
  writes `coin-link.json`, which the `end` entry's `files` seals. The hop from the certifier to the server crosses the
  prover's NIC, so it is the certifier's counted ingress, not a path this side trusts. The challenger side rests on the
  premises `CoinsConfidential` (the stream reaches the certifier only on auditor hosts, here this one) and `OneClock`
  (one clock for both, this host's), stated in `verity/core/protocols/sampled_proofs/README.md`.

  `serve.py --peer`, its own `ssh -R` to the peer, which the prover's sshd decrypts, is the development path alone
  (`run.sh`), and it refuses a window's session (`--root`).
- **A period** is a lane's span of windows between two signed entries, `period-open` and `period-close` (session null),
  which the lane's own gate key writes for its lane (`--name P`), or vy-admin for any lane (`period-open --name P --lane
  LANE`, `period-close --name P`). A period's name is used once, and a lane has one period open at a time. Its open
  commits, before window 0's first coin, the most windows it may hold (`--cap`, a positive integer) and its circuit
  (`--circuit`: the declared circuit's SHA-512, never a Program's id, which the gate records and CHAL1 checks each window
  against), and with `--draw-k K` the draw K the proof system selects each window's units with (`draw_k`, a positive
  integer at most 2^24, which PoUW's credit reads; an open without it commits none), with `--rows T` the served tables'
  rows (`rows`, each round's request and slot tables' leaves, which bound every step's served rows m_s in PoUW's
  credit and whose `h1window.slot` is each window's A slot M; an open without it credits nothing), and with `--model M` (or
  `--weights-root W`, which must be a registered root) the served model's weights' root from the auditor's model registry
  (`weights_root`, `credit.weights_root` of each K's b/<K> the auditor committed under its own salts, which PoUW's
  credit holds every window's weights to, and `model: {name, checkpoint_sha512}`; a root the registry has no entry of
  for the period's circuit is refused; a model it has none of there, a remote memory audit's whose circuit is its
  verifier tree, is signed as `model` alone, its digest core's pin, `pouw/lean/check/models.json`, and refused when the
  pin lacks it; an open without either commits none), and with `--circuit-files JSON` each K's circuit file (`circuit_files`, `{"k<K>": SHA-512}`:
  the circuit file the auditor's lowering built for K's template, which PoUW's credit verifies each window's statements
  on; an open without it commits none), and with `--length JSON` the fixed length its windows serve (`length`,
  `{"max_tokens": [n, ...], "stop": "fixed"}`: each request's public max_tokens, served to that cap whatever it samples,
  which PoUW's credit holds each window's served rows to; an open without it commits none, and credit counts none of
  its windows). Its windows are exactly the reserves (and one-step v2 opens) whose prover (`for`, else `by`) is its
  lane between its open and its close, at most `cap` (the next reserve is refused); every one counts. The close is
  refused while one of them is reserved and unexpired or still runs, and lists each, in order, as {root, session,
  complete} (root null for a reserve never opened; complete when its end is in the ledger and complete), with `complete` true only when it lists at least one window and all are: an
  incomplete window is the period's visible failure. Period entries are taken under the gate's lock, so no window is
  admitted between a close's reading of the ledger and its entry.
- **The ledger** (`/var/lib/vy-challenger/windows.jsonl`, `chattr +a`) is one JSON object a line: `schema`, `seq` (from 0),
  `prev` (SHA-512 of the previous line's bytes, 128 zeros first), `kind`, `session`, `body` and `sig`, an LMS signature under
  the challenger key over `entry_message(I, entry)`. Its entries' LMS leaves strictly increase (`append` refuses an entry
  whose leaf isn't after the last one, as a restored key state would sign), and each record's leaf lies between its
  session's open and end and is no entry's. v2 entries follow the host's v1 entries in one chain, and `verify_ledger`, the
  reference check, reads both, and checks each window's reserve, open and end in that order, its expiry, and the periods
  (`verify_periods` returns them). Infra publishes the ledger and each ended session to the store (`publish.sh`, the control
  pod's watchers loop `challenger-publish`), each put an extension of the last   (`publish.sh` refuses any other head, and
  `extends` is the check), labelled `challenger-ledger head` by infra, and the pins log the same way (`challenger-pins head`).

- **The model registry** (`/var/lib/vy-challenger/models.jsonl`, auditor's, `chattr +a`; `models.py`) is one line a
  registration, vy-admin's `model-register` run as `auditor`. It refuses a checkpoint whose
  files aren't the pinned `checkpoints-int8.json` role's, or that root or auditor alone can't write, or whose digest
  isn't core's pin (`verity/core/protocols/pouw/lean/check/models.json`), and a descriptor whose linears aren't the model's decoder
  layers in order; it commits each K's b/<K> from the pinned shards alone under a fresh salt key it keeps
  (`weights.weights_roots`), and appends the roots and their fold. `model-key` gives a lane the key (the openings of
  public weights), and `models` reads the registry, which `publish.sh` publishes as `challenger-models/v1`.

- **A certifier rehearsal** (`certifier-rehearse`, a lane's own gate key or vy-admin): the network certifier in front of
  a served endpoint, `benchmarks/network_traces/certifier_serve.py rehearse`, run here, on the auditor's machine, as the
  auditor's device would. The gate validates the request (`--peer` one of /etc/vy/peers, `--port` the endpoint's on the
  peer's loopback, `--minutes` 1 to 170, `--tamper` one of `TAMPERS` for a tampering shim instead of an honest one
  (`certifier_serve.py`'s: `late` holds a window-0 session past every sync, `extra` writes one event more and `drop`
  one fewer, which the rehearsal's check refuses, and `token` writes window 0's tokens with other bytes, a run it
  accepts, exit 0, since only the I/O tie refuses it), `--requests` 4 to 400, `--in-flight` 1 to 4 requests sent
  together and `--tokens` 16 to 64 tokens out of each, each recorded and passed on only when given
  (`certifier_serve.py`'s 4 and 64 otherwise), `--profile planned|served` N2's profile, planned by default: the planned
  egress occupancy, or the served profile, whose egress counts derive from the ingress head's, and the sampling
  and the keys' height below). With `--traffic varied` (recorded only then) the rehearsal runs `certifier_serve.py`'s varied plan:
  Poisson arrivals of varied prompts as id requests, each ending at the model's EOS or its `max_tokens` (1 to
  `--tokens`, the cap), and the responses written as id events (each sampled token's id), N2 on the advice profile;
  its own flags are `--rate` arrivals a second (above 0, at most 8), `--prompt-range LO:HI` (1 to 2048 tokens),
  `--seed`, `--eos-ids` (at most 8), `--compact B` (8 to 4096: compact egress framing, each sampled token's
  compact record in a B-byte cell instead of an id event) and `--margin M`
  (1 to 8: the plan's margin over the probed rates and latencies, 2 otherwise; an endpoint that slows under a load its
  probes didn't meet needs more), each recorded and passed on only when given (`parse_varied`). `--prompts` sends the
  caller's own prompts in place of `--prompt-range`'s chat turns: the gate reads them off its stdin as a scan
  rehearsal's, each of up to 2048 ids (`certifier_prompts`), writes them to `witness/prompts.json` (the request records
  their count and ids),
  and request j sends the (j mod n)-th of the n at the same seed's arrivals and `max_tokens`; the plan and the
  certified I/O in `out/` carry them, as they carry every varied request's prompt. It takes no `--in-flight`,
  `--profile`, `--temperature` or `--top-p` (its requests are greedy, each with a seed of its own), nor a `--minutes`
  its arrivals outlast. Each
  request's `user` is `<run>.j` (`--name`, the run), which the shim forwards the endpoint, so a committing endpoint
  records each request under the name the certified I/O reads, and the run's id events (or compact records), records
  and `io.json` are in its `fetch`: `negatives.py tie-run --run TARBALL --ids REQUESTS.jsonl` ties them to the
  endpoint's committed ids, which refuses `token`. The gate refuses while another rehearsal runs, writes
  `certifier/<run>/request.json` and starts its unit
  (`vy-certifier@<run>.service`), which runs `certifier-run` as `auditor`: an `ssh -N -L` with the auditor's tunnel key
  (the peer's line of /etc/vy/peers) from a free port of this host's loopback to 127.0.0.1:PORT on the peer, then the
  rehearsal through it into `out/`, stopped at `--minutes`, then the tunnel closed and `end.json` written (its exit, why
  it ended, the SHA-512 of each file of `out/`, and from `out/summary.json` each run's verdict, network verdict, audit,
  salt and what a second reveal of window 0's salt got). `status RUN` and `fetch RUN` return it. The rehearsal's keys are made in a scratch directory and removed, and `out/` holds no salt. A rehearsal is
  no window of the challenger's: it takes no coin and writes no ledger entry (the ledger's kinds are the verifier's,
  `verify_ledger`'s).
- **A warden rehearsal** (`warden-rehearse`, a lane's own gate key or vy-admin): the byte-cell warden
  (`verity.core.protocols.certifier.warden`) as the auditor's device, `benchmarks/network_traces/warden_serve.py
  rehearse`, with the warden and its client here and the server the peer's endpoint, reached through the same tunnel:
  client ──cells──▶ warden (here) ──tunnel──▶ the peer's 127.0.0.1:PORT. The gate validates `--peer`, `--port` and
  `--minutes` as a certifier rehearsal's (it takes no `--tamper`: the warden is the device, with no shim to tamper),
  `--requests` 4 to 1700 (100 by default) and `--tokens` 16 to 256, the most tokens a request asks for, recorded and
  passed on only when given (`warden_serve.py`'s 256 otherwise), and the sampling and the key's height below. The traffic
  is the gate's (`WARDEN_RATE`, `WARDEN_SEED`, `WARDEN_TAIL_S`), so a request whose `--minutes` is under
  `warden_minutes` of its requests and height is refused, with the least that fits. With `--scan` (recorded only then)
  it is the scan's rehearsal, the warden run `scan_zk.py prove --warden-run` reads from its `fetch`: the schedule is
  scan_zk's (`SCAN_SCHEDULE`, `scan_zk.py schedule`'s flags; the gate's otherwise `warden_serve.py`'s default, which
  the scan's class doesn't serve) and the traffic the same arrivals of warden-form id requests (`rehearse --ids`, at
  the sampling below), so the endpoint must answer each id-form (`io_tie.id_event`s, then `io_tie.TRAILER`): the
  warden frames what the server sends, and the check and the scan refuse a response that isn't by name. `--bucket-ms`
  (with `--scan`, recorded and passed on only when given) is that schedule's tick τ instead of its 100 ms
  (`SCAN_BUCKET_MS`; the scan's class takes any tick), the warden run at it: a request has D ticks from its admission
  to its first egress cell, by which a short answer must be whole, and the gate counts the minutes and sampling blocks
  below at that tick (`scan_schedule`). A scan rehearsal's `--prompts` sends the caller's own requests instead (a period's of a loop, say): the gate reads them off
  its stdin (`warden_prompts`: at most `WARDEN_PROMPTS_BYTES` of JSON, `WARDEN_PROMPTS` prompts of `WARDEN_PROMPT_IDS`
  token ids each, at most the scan's KI of 1024 and below 2^17, its IDB), writes them to `witness/prompts.json` alone (the request records their
  count and ids), and `rehearse --prompts` sends request j the (j mod n)-th of the n with exactly `--tokens`
  (`WARDEN_PROMPT_TOKENS`, the scan's 64 by default), `--requests` (`WARDEN_PROMPT_REQUESTS`) the prompts' count by
  default. `--window-per-group` (at most the scan's 4 lanes of prompts) sends them a group at a time, group g's half a
  bucket into link-window g·G (G = `scan_group_windows` of the tokens), so that group g's cells are exactly windows
  g·G … (g+1)·G − 1, which `out/groups.json` publishes, and `--minutes` must cover `warden_group_minutes` of its
  groups. The demo's stand-in (`benchmarks/network_traces/demo_stub.py`) serves the demo's class, prompts of exactly 8
  byte ids and exactly 24 out, id-form, and prints its prompts: `demo_stub.py prompts | … warden-rehearse --scan
  --prompts --tokens 24`. Its request carries `device: warden`; it
  shares the certifier rehearsal's directory, run names, unit, `status` and `fetch`, and one of either runs at a time.
  The warden writes to `witness/` (0700): its three records, `certified.jsonl` (each record's head and certifier salt
  with its certificate), the cells' salts (each link-window's rows' and wiring's), the traffic, the client's log and
  the check, all the prover's. Its unit publishes to `out/` only what isn't: `deployment.json` (the warden's key,
  session and schedule), `cells.jsonl` (each link-window's registration, `warden_rows`, and its cost),
  `certificates.jsonl` (the certificates alone) and a `--window-per-group` run's `groups.json`.
  `end.json` adds `device`, the check's verdict (`warden_outcome`: accepted, the number of problems and the counts
  `WARDEN_COUNTS` names), each registration's root by value and each witness file by name and size. `fetch RUN` gives
  `witness/` only to the run's own lane key (the request's `by`) and to vy-admin.
- **The load** (a lane's own gate key or vy-admin, in the rehearsal unit, one at a time with the
  rehearsals): the demo's weights, plain INT8 as interface 1 generates them from its public seed, sit on this
  host's store disk (`/var/lib/vy-challenger/store`, `Host.store`, its own mount, `auditor`'s) and are relayed to the
  H200 once a serving session. `weights-write [--shape h200|tiny]` writes the shape's stream there once
  (`load_relay.py write`, `WEIGHTS_MINUTES`): a stream already there is read back and kept only when its tensors'
  SHA-256 fold to the pin of interface 1's manifest (`load_relay.STORE_DIGESTS`), else refused and left unrecorded;
  its end carries the store's record (`store`: written, present or refused). `load-relay` is a warden rehearsal (all
  of `warden-rehearse`'s flags) with the load first, run by the same warden: the gate refuses it unless the store's
  record of `--shape` is interface 1's (`load_relay.read_store`), `--program` is the reading program's digest and
  `--minutes` covers `load_minutes` (the stream and its witness at `LOAD_RATE`, and `LOAD_READY_S`) and then the
  rehearsal's. Its run opens one more tunnel a stream (`--streams`, 4 by default) to `--sink-port` on the peer's
  loopback, where the prover's `load_relay.py sink --streams N --weights W --witness S` listens, and `warden_serve.py
  rehearse --load-store` relays the store over them: each value's bytes held to its digest and committed as they
  cross (`load_relay.relay_load`), every row's salts sent on with them. The witness (4.85 GB at H200) is the sink's
  `--witness` file on the prover's host and nowhere here: this host never writes a salt. The prover then starts its
  endpoint on the weights the sink wrote, and the warden's one signer (its publisher process) signs the load's 733
  heads in the serving session before its first window, so one key and one session carry the load and the windows;
  `out/` publishes `load.json` (the load's registrations, which the block record pins) and `load-relay.json` (the
  relay's counts and the sink's acknowledgements, or why it was refused), and the end carries the relay's and the
  check's `load` (values and how many registered).
- **A rehearsal's sampling** (`sampling`, both verbs): `--temperature` 0 to 2 (0 by default) and `--top-p` above 0
  and at most 1 (1 by default), each a plain decimal of at most 4 places, recorded in every request and passed on
  every one (`sampling_argv`), so every request the rehearsal sends carries both: an endpoint applies its own
  generation config (a committing LLAMA32_1B's: temperature 0.6, top_p 0.9) to a request that omits them. The gate
  takes a top_p below 1, which the compile unit refuses while the seeded top-p sampler's split count reads the live
  row count.
- **A rehearsal's keys** (`--height`, both verbs): LMS (N = 64, W = 1) at a tree height within `REHEARSAL_HEIGHT`, the
  deployed key's (`lms.DEPLOYED`, 20) by default, so that a rehearsal signs as the deployed device does unless its
  request says otherwise; recorded in every request and passed on every one. `certifier_serve.load_key` makes each
  key's 2^H one-time keys in eight processes, minutes a key at 20 (`KEYGEN_LEAVES_S`), within the run's `--minutes`:
  a certifier rehearsal makes two keys (the certifier's and the challenger's) and a warden rehearsal one, which
  `warden_minutes` counts. A warden rehearsal's key signs `WARDEN_WINDOW_LEAVES` one-time signatures a link-window,
  after a load's heads, and its signer refuses every window past its 2^H: so a request whose `warden_windows` its
  height can't sign is refused, with the least height that can (`warden_height`; the gate's most requests at the
  scan's schedule take 13). The rehearsal draws each serve's session itself (32 fresh random bytes, `--session`), as
  the challenger it plays: the challenger here draws none, since a rehearsal is none of its windows.

Every signature of the key (`reserve`'s, `admit`'s, a period's, `end`'s, and `serve.py`'s record and close,
`--sign-lock`) is taken under `sign.lock`, each opening the key's leaf state only to sign, so `lms.LeafState`, which
refuses a second signer, never sees two, and sessions run side by side. The ledger's reference check, `verify_ledger`
and `verify_periods`, is `verity.core.protocols.sampled_proofs.challenger`'s, the verifier's own.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
sys.path.insert(0, str(TREE))

from verity.core.protocols.sampled_proofs.challenger import (LEDGER_ROLE as ROLE, LEDGER_SCHEMA as SCHEMA,  # noqa: E402
                                                             LEDGER_SCHEMA_V1 as SCHEMA_V1,
                                                             LEDGER_TAGS as TAGS, PERIOD_KINDS, RESERVED,
                                                             SESSION_KINDS, cap_of as _cap_of,
                                                             circuit_of as _circuit_of, extends, lane_of, leaf_of,
                                                             ledger_message as entry_message, ledger_text, pin_line,
                                                             pins_reason, pins_seal, root_of, verify_ledger, verify_periods)
from verity.ml.crypto import lms  # noqa: E402

ZERO = "0" * 128
PORTS = range(7811, 7900)
MINUTES = (1, 720)
#: the most a period-open's draw K may be (PoUW's `credit.MAX_K`)
MAX_DRAW_K = 1 << 24
ROOT = re.compile(r"[0-9a-f]{128}")
PINNED = re.compile(r"[0-9a-f]{40}")
#: a circuit file's template: K's tile template (``k<K>``) or the live Program's (``live``), PoUW's `credit._TEMPLATE`
TEMPLATE = re.compile(r"k[1-9][0-9]{0,7}|live")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
SCHEME = re.compile(r"[a-z0-9][a-z0-9/._-]{0,127}")
SESSION = re.compile(r"s\d{8}T\d{6}Z-[0-9a-f]{4}")
CERTIFIER_RUN = re.compile(r"c\d{8}T\d{6}Z-[0-9a-f]{4}")
#: a rehearsal's --minutes: at most 170, so that a run stays inside its unit's RuntimeMaxSec=3h (vy-certifier@.service)
#: with 10 minutes for its tunnels and outcome; an H200 load's 67 then leave about 100 of serving
CERTIFIER_MINUTES = (1, 170)
CERTIFIER_REQUESTS = (4, 400)
CERTIFIER_IN_FLIGHT = (1, 4)
CERTIFIER_TOKENS = (16, 64)
#: a certifier rehearsal's tampering shims, each `certifier_serve.py`'s `--tamper` of that name
TAMPERS = ("late", "extra", "drop", "token", "byte")
CERTIFIER_PROFILES = ("planned", "served")
#: a certifier rehearsal's traffic, `certifier_serve.py`'s `--traffic`
CERTIFIER_TRAFFIC = ("fixed", "varied")
#: varied traffic's bounds: the most arrivals a second, prompts' lengths in tokens, the largest seed, the most EOS ids
CERTIFIER_RATE = 8.0
#: `certifier_serve.py`'s rate when a request gives none
CERTIFIER_VARIED_RATE = 2.0
CERTIFIER_PROMPTS = (1, 2048)
CERTIFIER_SEED = (1 << 32) - 1
CERTIFIER_EOS = 8
#: a compact cell's bytes: at least a compact record's (`active.COMPACT_RECORD`), at most an id event's frame
#: (`certifier_serve.FRAME_BYTES`)
CERTIFIER_COMPACT = (8, 4096)
#: a varied plan's `--margin` over the probed rates and latencies (`certifier_serve.varied_plan`'s 2 otherwise)
CERTIFIER_MARGIN = (1, 8)
#: varied traffic's flags the gate passes on when given, each (flag, the request's key)
CERTIFIER_VARIED = (("--tokens", "tokens"), ("--rate", "rate"), ("--prompt-range", "prompt_range"), ("--seed", "seed"),
                    ("--eos-ids", "eos_ids"), ("--compact", "compact"), ("--margin", "margin"))
#: a rehearsal's `--temperature` bounds; `--top-p` is above 0 and at most 1, and each is a plain decimal (`DECIMAL`)
TEMPERATURE = (0.0, 2.0)
DECIMAL = r"[0-9](\.[0-9]{1,4})?"
#: a rehearsal's `--height` bounds, the LMS keys' tree height: at most the deployed key's, which is the default
REHEARSAL_HEIGHT = (10, lms.DEPLOYED.H)
#: the one-time keys a second `certifier_serve.load_key` makes in its eight processes (16384 in 4.1 s on four cores)
KEYGEN_LEAVES_S = 4096
#: a warden rehearsal's `--requests` (and a scan rehearsal's on its own `--prompts`, `WARDEN_PROMPT_REQUESTS`): 1700 at
#: `WARDEN_RATE` is one session of about 57 minutes, about 1080 of the scan's 3.2 s link-windows, past the 745 that
#: the whole-node wipe needs
WARDEN_REQUESTS = (4, 1700)
#: the one-time signatures a warden rehearsal's key gives a link-window: a certificate for each of its three links'
#: records (ingress, egress and the fails link) and a signed head for the ingress and egress cells' rows
#: (`warden_serve.Publisher.keep` and `attest`)
WARDEN_WINDOW_LEAVES = 5
#: a warden rehearsal's link-window without `--scan`: `warden_serve.py`'s default schedule's (`WardenParams`, 600
#: buckets of 100 ms)
WARDEN_WINDOW_MS = 60_000
WARDEN_TOKENS = (16, 256)
#: the warden rehearsal's traffic, which the gate fixes so that `warden_minutes` bounds it: Poisson arrivals at
#: `WARDEN_RATE` a second drawn from `WARDEN_SEED`, the run lasting `WARDEN_TAIL_S` past the last
WARDEN_RATE, WARDEN_TAIL_S, WARDEN_SEED = 0.5, 60, 0
#: a scan rehearsal's (`--scan`) schedule, scan_zk's (`scan_zk.PARAMS`, `scan_zk.py schedule`), the only one
#: `scan_zk.py prove --warden-run` takes but for its tick (`SCAN_BUCKET_MS`), as `warden_serve.py`'s flags
SCAN_SCHEDULE = ("--cell-bytes", "512", "--slots", "4", "--bucket-ms", "100", "--delay", "27", "--lanes", "4",
                 "--queue", "4", "--window", "32", "--ingress-cells", "47", "--ingress-buckets", "12",
                 "--egress-cells", "65", "--every", "16")
#: the scan's schedule by flag, which `--window-per-group` sizes a group's link-windows on (`scan_group_windows`)
SCAN = {f: int(v) for f, v in zip(SCAN_SCHEDULE[::2], SCAN_SCHEDULE[1::2])}
#: a scan rehearsal's `--bucket-ms`, its tick τ: the endpoint gets a short request in its admission's tick, and its
#: first egress cell leaves D = 27 ticks later, by which an answer of at most r = 4 ids must be whole: D·τ = 2.7 s at
#: the scan's 100 ms, and at the most 108 s, for an endpoint that takes up to about 90 s to answer
SCAN_BUCKET_MS = (SCAN["--bucket-ms"], 4000)
#: a scan rehearsal's `--prompts`, read off the gate's stdin: at most `WARDEN_PROMPTS_BYTES` of JSON, an array of
#: `WARDEN_PROMPTS` prompts, each of `WARDEN_PROMPT_IDS` token ids below 2^`WARDEN_PROMPT_ID_BITS` (the scan's IDB);
#: then `--requests` and `--tokens` (each request's exact count, at most the scan's egress cells but the trailer's)
#: take `WARDEN_PROMPT_REQUESTS` and `WARDEN_PROMPT_TOKENS`. A prompt is at most the scan's KI ids (`scan_zk.CLASS`),
#: since the scan refuses a request of more and the gate refuses it at input instead
WARDEN_PROMPTS_BYTES = 1 << 20
WARDEN_PROMPTS = (1, 400)
WARDEN_PROMPT_IDS = (1, 1024)
WARDEN_PROMPT_ID_BITS = 17
WARDEN_PROMPT_REQUESTS = (1, WARDEN_REQUESTS[1])
WARDEN_PROMPT_TOKENS = (1, SCAN["--egress-cells"] - 1)
#: the prompts' file in `witness/`, the prover's: `warden_serve.py rehearse --prompts` reads it
PROMPTS = "prompts.json"
#: what a warden rehearsal publishes in `out/`, beside its certificates (`certificates.jsonl`); everything else
#: `warden_serve.py rehearse` writes is witness and stays in `witness/`, which `fetch` gives the run's own lane alone.
#: `groups.json` (a `--window-per-group` run's) is each group's request indices and link-windows, the request's
#: public counts and the scan's schedule alone
WARDEN_PUBLIC = ("deployment.json", "cells.jsonl", "groups.json", "load.json", "load-relay.json")
WITNESS = "witness"
#: the shapes of interface 1's weights a store holds and a load relays (`load_relay.SHAPES`)
LOAD_SHAPES = ("h200", "tiny")
#: a load's streams to the H200's sink, each through a tunnel of its own, and how many by default
LOAD_STREAMS, LOAD_DEFAULT_STREAMS = (1, 8), 4
#: the link's rate a load's minutes are sized on (about 95 MB/s, measured over 4 streams from this host to
#: the H200 through the tunnel), and the seconds the endpoint has after the load to answer
LOAD_RATE, LOAD_READY_S = 80e6, 900
#: a store write's minutes: interface 1's H200 stream generated (about 15 s a GiB a core), written and read back
WEIGHTS_MINUTES = 150
TUNNEL_SSH = ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15"]
READ_ONLY = "-"
HOST_ACCOUNTS = ("root", "vy-admin", "research", "auditor")
SECRET = "secret"
OPEN_REQUEST = "open.json"
OUT_FILES = ("record.json", "key.json", "releases.jsonl", "hellos.jsonl", "challenger.json", "address.json", "serve.log",
             "coin-link.json", "coin-link.log")
#: how long serve.py has, once started, to sign its record (`await_commitment`), and the coin link to listen
START_SECONDS = 60.0
#: how long the coin link has, once told to stop, to run to the end of its link-window (D7's T·τ, 60 s) and write its
#: record, before it is killed
LINK_STOP_SECONDS = 90.0


class Refused(Exception):
    """A request the gate or the session refuses, with why."""


#: the auditor host's name, which its records give it
HOST_NAME: str = json.loads((HERE / "host.json").read_text())["name"]


@dataclass(frozen=True)
class Host:
    """Where the service keeps its state, and the name its records give the host."""

    root: Path = Path("/var/lib/vy-challenger")
    peers: Path = Path("/etc/vy/peers")
    name: str = HOST_NAME
    pins: Path = Path("/opt/vy-challenger/pins.jsonl")

    @property
    def coin_ports(self) -> Path:
        """Each peer's certifier coin link (`coin_ports`), beside the peers file: /etc/vy/coin-ports."""
        return self.peers.with_name("coin-ports")

    @property
    def ledger(self) -> Path:
        return self.root / "windows.jsonl"

    @property
    def sessions(self) -> Path:
        return self.root / "sessions"

    @property
    def key(self) -> Path:
        return self.root / "key"

    @property
    def models(self) -> Path:
        return self.root / "models.jsonl"

    @property
    def certifier(self) -> Path:
        return self.root / "certifier"

    @property
    def store(self) -> Path:
        """The store disk (its own mount, `auditor`'s): interface 1's weights stream a shape
        (`load_relay.write_store`)."""
        return self.root / "store"


def utc(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def later(t: str, minutes: int) -> str:
    """`t` (`utc`'s form) plus `minutes`: a reserve's expiry."""
    return utc(datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) + timedelta(minutes=minutes))


def pinned_commit(tree: Path = TREE) -> str:
    """The commit this tree was pinned at: its directory's name under /opt/vy-challenger."""
    if not PINNED.fullmatch(tree.name):
        raise Refused(f"{tree} is not a pinned tree (/opt/vy-challenger/<commit>, pin.sh): no session runs from it")
    return tree.name


def read_pins(path: Path) -> list[str]:
    try:
        return path.read_text().splitlines()
    except FileNotFoundError:
        return []


def append_only(path: Path) -> None:
    subprocess.run(["chattr", "+a", str(path)], check=True)


def record_pin(log: Path, commit: str, main: str, now: datetime, *, seal: Callable[[Path], None] = append_only) -> str:
    """pin.sh's last step before it moves `current`, as root: `pin_line(commit, main, now)` appended to the pins log `log`,
    which is root's, 0644, and append-only (`chattr +a`, which only root can lift, as the ledger) before any line is
    written. A failure here fails the pin, so `current` never names a tree the log doesn't."""
    try:
        line = pin_line(commit, main, utc(now))
    except ValueError as e:
        raise Refused(f"pin {commit!r} at main {main!r}: {e}") from None
    if not log.exists():
        log.touch()
        log.chmod(0o644)
    seal(log)
    with open(log, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    return line


def pins_refusal(host: Host, session: str, pinned: str) -> tuple[dict, str | None]:
    """The host's pins log sealed as it stands (`pins_seal`), and why a session at `pinned` wouldn't count under it: the
    verifier's own check (`pins_reason`), so the gate refuses a window `pinned_session` would."""
    pins = read_pins(host.pins)
    if not pins:
        return pins_seal(pins), (f"{host.pins} records no pin, so no session here would count: pin the tree with "
                                 "vy-challenger-pin, which logs each pin")
    seal = pins_seal(pins)
    return seal, pins_reason(session, {"pinned": pinned, "pins": seal}, pins)


# --- the ledger ---------------------------------------------------------------------------------------------------------------

def read_ledger(path: Path) -> list[bytes]:
    try:
        return [ln for ln in path.read_bytes().split(b"\n") if ln]
    except FileNotFoundError:
        return []


def entries(lines: list[bytes]) -> list[dict]:
    return [json.loads(ln) for ln in lines]


def sessions_of(lines: list[bytes]) -> dict[str, dict[str, dict]]:
    """session -> {"reserve": entry, "open": entry, "end": entry} as far as the ledger has them."""
    out: dict[str, dict[str, dict]] = {}
    for e in entries(lines):
        if e["kind"] in SESSION_KINDS:
            out.setdefault(e["session"], {})[e["kind"]] = e
    return out


def circuit_of(s: str) -> str:
    """A period's circuit as its open records it (`challenger.circuit_of`), or Refused."""
    try:
        return _circuit_of(s)
    except ValueError as e:
        raise Refused(str(e)) from None


def cap_of(v: Any) -> int:
    """A period's cap (`challenger.cap_of`), or Refused."""
    try:
        return _cap_of(v)
    except ValueError as e:
        raise Refused(str(e)) from None


def draw_k_of(s: str) -> int:
    """A period-open's draw K (`--draw-k`, written as `draw_k`): a positive integer at most 2^24, or Refused."""
    if not re.fullmatch(r"[1-9][0-9]{0,7}", s) or int(s) > MAX_DRAW_K:
        raise Refused(f"--draw-k {s!r}: the draw K the proof system selects each window's units with, 1 to 2^24")
    return int(s)


def rows_of(s: str) -> int:
    """A period-open's table rows T (`--rows`, written as `rows`): each round's request and slot tables' leaves, the pouw
    setting ``rows``, a positive integer at most 2^24, or Refused."""
    if not re.fullmatch(r"[1-9][0-9]{0,7}", s) or int(s) > 1 << 24:
        raise Refused(f"--rows {s!r}: the served tables' rows T (each round's request and slot tables' leaves), 1 to 2^24")
    return int(s)


def weights_root_of(s: str) -> str:
    """A period-open's weights' root (`--weights-root`, written as `weights_root`): a SHA-512 digest in lowercase hex, or
    Refused."""
    if not ROOT.fullmatch(s):
        raise Refused(f"--weights-root {s!r}: the served model's registered weights' root (credit.weights_root), a "
                      "SHA-512 digest in 128 lowercase hex")
    return s


def circuit_files_of(s: str) -> dict[str, str]:
    """A period-open's circuit files (`--circuit-files`, written as `circuit_files`): a JSON object taking each K's
    template (``k<K>``, and ``live`` the live Program's) to the SHA-512 of the circuit file the auditor's lowering built
    for it, in lowercase hex, as PoUW's credit reads it (`committed_files`); or Refused."""
    def pairs(items: list) -> dict | None:
        return dict(items) if len({k for k, _ in items}) == len(items) else None

    try:
        files = json.loads(s, object_pairs_hook=pairs)
    except (ValueError, RecursionError):
        files = None
    if not (isinstance(files, dict) and 0 < len(files) <= 64 and all(
            TEMPLATE.fullmatch(k) and type(v) is str and ROOT.fullmatch(v) for k, v in files.items())):
        raise Refused(f"--circuit-files {s[:200]!r}: each K's template to its circuit file's SHA-512, a JSON object "
                      '{"k<K>" or "live": "<128 lowercase hex>", ...} of 1 to 64 distinct K')
    return files


def model_of(s: str) -> str:
    """A period-open's model (`--model`): the int8 checkpoint role the model registry has it under, or Refused."""
    if not NAME.fullmatch(s):
        raise Refused(f"--model {s!r}: a role of the model registry (models), 1 to 64 of [A-Za-z0-9._:-]")
    return s


def length_of(s: str) -> dict:
    """A period-open's fixed length (`--length`, written as `length`): ``{"max_tokens": [...], "stop": "fixed"}``, each
    request's public max_tokens in admission order (1 to 2^16 requests, each 1 to 2^24) served to that cap whatever it
    samples (ignore_eos), as PoUW's credit reads it (`committed_length`); or Refused."""
    try:
        v = json.loads(s)
    except (ValueError, RecursionError):
        v = None
    m = v.get("max_tokens") if isinstance(v, dict) else None
    if not (isinstance(v, dict) and sorted(v) == ["max_tokens", "stop"] and v["stop"] == "fixed" and type(m) is list
            and 0 < len(m) <= 1 << 16 and all(type(n) is int and 1 <= n <= MAX_DRAW_K for n in m)):
        raise Refused(f"--length {s[:200]!r}: each request's public max_tokens and the public-cap done source, a JSON "
                      'object {"max_tokens": [n, ...], "stop": "fixed"} of 1 to 2^16 requests, each 1 to 2^24')
    return v


#: a period-open's optional fields, as the gate hands them to `period` (``FIELD=VALUE``) and its open records them; `model`
#: and `weights_root` name a model registry entry, whose root and model the open signs (`models.resolve`)
PERIOD_FIELDS = {"draw_k": draw_k_of, "rows": rows_of, "weights_root": weights_root_of, "circuit_files": circuit_files_of,
                 "model": model_of, "length": length_of}


def field_text(v: Any) -> str:
    """A period-open field's value as the gate hands it to `period`: a JSON object canonical, anything else as is."""
    return json.dumps(v, sort_keys=True, separators=(",", ":")) if isinstance(v, dict) else str(v)


def counts_at(e: dict) -> bool:
    """Whether a window joins its lane's open period at this entry: at its reserve, or at a v2 open with no reserve (a
    one-step open, before the first reserve)."""
    return e["kind"] == "reserve" or (e["kind"] == "open" and e["schema"] == SCHEMA and "reserve" not in e["body"])


def periods_of(lines: list[bytes]) -> dict[str, dict]:
    """period -> {"lane", "open", "close" (None while open), "windows": [session, ...], "cap", "circuit"}: a period's
    windows are exactly its lane's reserves (and one-step v2 opens) between its period-open and its period-close."""
    periods: dict[str, dict] = {}
    current: dict[str, dict] = {}
    for e in entries(lines):
        if e["kind"] == "period-open":
            p = current[e["body"]["lane"]] = periods[e["body"]["period"]] = {
                "lane": e["body"]["lane"], "open": e, "close": None, "windows": [], "cap": e["body"].get("cap"),
                "circuit": e["body"].get("circuit")}
        elif e["kind"] == "period-close":
            p = periods[e["body"]["period"]]
            p["close"] = e
            current.pop(p["lane"], None)
        elif counts_at(e) and (p := current.get(lane_of(e))) is not None:
            p["windows"].append(e["session"])
    return periods


def window_outcomes(lines: list[bytes], windows: list[str]) -> list[dict]:
    """Each window's outcome as a period-close records it: its root (None when it was never opened, as an expired reserve
    isn't), complete only when its end is in the ledger and complete."""
    sessions = sessions_of(lines)
    return [{"root": root_of(sessions[s]["open"]) if "open" in sessions[s] else None, "session": s,
             "complete": "end" in sessions[s] and sessions[s]["end"]["body"].get("complete") is True} for s in windows]


def roots_of(lines: list[bytes]) -> dict[str, str]:
    """root -> the session its open names, for every open in the ledger."""
    return {root_of(s["open"]): sid for sid, s in sessions_of(lines).items() if "open" in s}


@contextmanager
def locked(path: Path) -> Iterator[None]:
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def append(host: Host, kind: str, session: str, body_of: Callable[[list[bytes]], dict], I: bytes,
           sign: Callable[[bytes], bytes]) -> dict:
    """Append one signed entry, its body `body_of` the ledger as it stands (which may refuse), under the ledger's lock;
    fsynced. Refused, unwritten, when the signature's leaf isn't after the ledger's last entry's (a restored key state)."""
    with locked(host.root / "ledger.lock"):
        lines = read_ledger(host.ledger)
        entry = {"schema": SCHEMA, "seq": len(lines), "prev": hashlib.sha512(lines[-1]).hexdigest() if lines else ZERO,
                 "kind": kind, "session": session, "body": body_of(lines)}
        with locked(host.root / "sign.lock"):
            entry["sig"] = sign(entry_message(I, entry)).hex()
        if lines and leaf_of(entry["sig"]) <= leaf_of(json.loads(lines[-1])["sig"]):
            raise Refused(f"the key signed at leaf {leaf_of(entry['sig'])}, not after the ledger's last entry's: its key state "
                          "was restored or forked, and no entry is written under it")
        fd = os.open(host.ledger, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, ledger_text(entry).encode() + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)
    return entry


# --- the key (serve.py's) -----------------------------------------------------------------------------------------------------

def load_key(host: Host, params=None, workers: int | None = None):
    """The challenger's key and leaf state in `host.key`, made there when absent (serve.py's `load_key`)."""
    sys.path.insert(0, str(HERE))
    import serve
    from verity.ml.crypto import lms
    return serve.load_key(host.key, params or lms.DEPLOYED, workers or len(os.sched_getaffinity(0)))


def key_signer(host: Host, params=None, workers: int | None = None):
    """(I, sign, key_sha512) for `append`; each signature opens the key's leaf state and closes it after."""
    from verity.ml.crypto import lms
    key, state = load_key(host, params, workers)
    state.close()
    public = lms.public_key(key)

    def sign(message: bytes) -> bytes:
        with lms.LeafState(host.key / "state.json", public, key.params.H) as st:
            return lms.sign(key, st, message).encode()
    return public[:16], sign, hashlib.sha512(public).hexdigest()


# --- the model registry (models.py) -------------------------------------------------------------------------------------------

def registry():
    """`models`, imported only by the verbs that read the registry (it brings numpy and PoUW's credit)."""
    sys.path.insert(0, str(HERE))
    import models
    return models


def model_register(host: Host, args: list[str], *, pinned: str, now: datetime, key: bytes | None = None,
                   tree: Path = TREE) -> dict:
    """`model-register ROLE CIRCUIT DESCRIPTOR CHECKPOINT`, as `auditor`: `models.register` under a fresh salt key, the key
    kept in `models.key_path` (0600) and the line appended to the registry under its lock; or Refused."""
    M = registry()
    if len(args) != 4:
        raise Refused(f"model-register: {args}")
    role, circuit, descriptor, checkpoint = model_of(args[0]), circuit_of(args[1]), Path(args[2]), Path(args[3])
    key = key or secrets.token_bytes(32)
    try:
        line = M.register(tree, role, circuit, descriptor.read_bytes(), checkpoint, key, pinned=pinned, now=utc(now),
                          log=lambda s: print(s, file=sys.stderr, flush=True))
    except (M.Refused, OSError) as e:
        raise Refused(str(e)) from None
    kp = M.key_path(host.root, line)
    kp.parent.mkdir(mode=0o700, exist_ok=True)
    fd = os.open(kp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, key)
        os.fsync(fd)
    finally:
        os.close(fd)
    with locked(host.root / "models.lock"):
        fd = os.open(host.models, os.O_WRONLY | os.O_APPEND)
        try:
            os.write(fd, (json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)
    return line


def parse_model(verb: str, argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=f"vy-challenger-gate {verb}", exit_on_error=False, add_help=False, allow_abbrev=False)
    p.add_argument("--model", required=True)
    p.add_argument("--circuit", required=True)
    if verb == "model-key":
        p.add_argument("--weights-root", dest="weights_root", default=None)
    if verb == "model-register":
        p.add_argument("--descriptor", required=True)
        p.add_argument("--checkpoint", required=True)

    def refuse(why: object) -> None:
        raise Refused(f"{verb}: {why}; {' '.join(p.format_usage().split())}")

    p.error = refuse
    try:
        a = p.parse_args(argv)
    except argparse.ArgumentError as e:
        refuse(e)
    a.model, a.circuit = model_of(a.model), circuit_of(a.circuit)
    return a


def gate_model_key(argv: list[str], host: Host) -> dict:
    """The salt key of `--model`'s registration under `--circuit` whose root is `--weights-root` (the period's signed one;
    required when the model is registered more than once there): the prover's openings of its b/<K>."""
    M = registry()
    a = parse_model("model-key", argv)
    try:
        e = M.resolve(M.read(host.models), a.circuit, a.model, a.weights_root and weights_root_of(a.weights_root),
                      unique=True)
    except M.Refused as x:
        raise Refused(str(x)) from None
    key = M.key_path(host.root, e).read_bytes()
    return {"model": e["model"], "circuit": e["circuit"], "weights_root": e["weights_root"], "key": key.hex(),
            "key_sha512": e["key_sha512"]}


def seal_models(host: Host, *, own: Callable[[Path], None], seal: Callable[[Path], None] = append_only) -> None:
    """The registry made auditor's and append-only (`chattr +a`, which only root lifts) before its first line."""
    if not host.models.exists():
        host.models.touch()
        host.models.chmod(0o644)
        own(host.models)
    seal(host.models)


def as_auditor_register(args: list[str]) -> dict:
    p = subprocess.run(_auditor(["model-register", *args]), stdout=subprocess.PIPE, stderr=None, text=True)
    if p.returncode:
        raise Refused(f"model-register exited {p.returncode} (its reason is on stderr above)")
    return json.loads(p.stdout)


# --- the gate (root) ----------------------------------------------------------------------------------------------------------

def peers(host: Host) -> list[str]:
    names = [ln.split()[0] for ln in host.peers.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return [n for n in names if n != host.name]


def coin_ports(host: Host, peer: str) -> range:
    """The ports of `peer`'s certifier coin link: its line `PEER LO-HI` (or `PEER PORT`) of /etc/vy/coin-ports
    (`Host.coin_ports`), ports within `PORTS` of this host's loopback that the link carries to the same ports of the
    peer's loopback; or Refused, since a window's coins reach its peer through that link alone (`CoinsConfidential`)."""
    try:
        lines = host.coin_ports.read_text().splitlines()
    except FileNotFoundError:
        lines = []
    for line in lines:
        words = line.split()
        if words[:1] == [peer] and len(words) == 2 and (m := re.fullmatch(r"(\d{1,5})(?:-(\d{1,5}))?", words[1])):
            lo, hi = int(m[1]), int(m[2] or m[1])
            if PORTS.start <= lo <= hi < PORTS.stop:
                return range(lo, hi + 1)
    raise Refused(f"--peer {peer!r} has no certifier coin link: a window's coins reach its peer only through the link of "
                  f"the certifier on this host, never ssh -R (CoinsConfidential); infra lists its ports as "
                  f"`{peer} LO-HI`, within {PORTS.start}..{PORTS.stop - 1}, in {host.coin_ports}")


def parse_reserve(argv: list[str], host: Host, *, admin: bool = False) -> dict:
    """A reserve's request; `--for LANE` (vy-admin's alone: a development window reserved for a lane without its own key)
    names the window's prover."""
    p = argparse.ArgumentParser(prog="vy-challenger-gate reserve", exit_on_error=False, add_help=False, allow_abbrev=False)
    for flag in ("--peer", "--scheme", "--families"):
        p.add_argument(flag, required=True)
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--minutes", type=int, default=120)
    if admin:
        p.add_argument("--for", dest="lane", default=None)

    def refuse(why: object) -> None:
        raise Refused(f"reserve: {why}; {' '.join(p.format_usage().split())} (no --commit, --period or --lane: the "
                      "window's root comes at its open, its time is the auditor's, its prover your gate key's lane)")

    p.error = refuse
    try:
        a = p.parse_args(argv)
    except argparse.ArgumentError as e:
        refuse(e)
    if a.peer not in peers(host):
        raise Refused(f"--peer {a.peer!r}: one of {peers(host)} (/etc/vy/peers)")
    coin_ports(host, a.peer)
    if not SCHEME.fullmatch(a.scheme):
        raise Refused(f"--scheme {a.scheme!r}: [a-z0-9/._-], up to 128")
    try:
        families = json.loads(a.families)
    except json.JSONDecodeError:
        families = None
    if (not isinstance(families, list) or not families or len(a.families) > 4096
            or not all(isinstance(f, list) and len(f) == 2 and isinstance(f[0], str) and SCHEME.fullmatch(f[0])
                       and type(f[1]) is int and f[1] > 0 for f in families)
            or len({f[0] for f in families}) != len(families)):
        raise Refused("--families: a JSON list of [label, count], labels distinct, count >= 1, up to 4096 bytes")
    if not 0 <= a.epoch < 2 ** 63:
        raise Refused("--epoch: 0 <= E < 2^63")
    if not MINUTES[0] <= a.minutes <= MINUTES[1]:
        raise Refused(f"--minutes: {MINUTES[0]} to {MINUTES[1]}")
    lane = getattr(a, "lane", None)
    if lane is not None and (not NAME.fullmatch(lane) or lane in HOST_ACCOUNTS):
        raise Refused(f"--for {lane!r}: a lane's name")
    return {"peer": a.peer, "scheme": a.scheme, "families": families, "epoch": a.epoch, "minutes": a.minutes,
            **({"for": lane} if lane else {})}


def parse_open(argv: list[str]) -> tuple[str, str]:
    """An open's (session, root): the reserved session and the window's root, 128 lowercase hex digits."""
    p = argparse.ArgumentParser(prog="vy-challenger-gate open", exit_on_error=False, add_help=False, allow_abbrev=False)
    p.add_argument("--session", required=True)
    p.add_argument("--root", required=True)

    def refuse(why: object) -> None:
        raise Refused(f"open: {why}; {' '.join(p.format_usage().split())} (reserve first: its answer names the session)")

    p.error = refuse
    try:
        a = p.parse_args(argv)
    except argparse.ArgumentError as e:
        refuse(e)
    if not SESSION.fullmatch(a.session):
        raise Refused(f"--session {a.session!r}: a reserve's session (sYYYYMMDDTHHMMSSZ-xxxx)")
    if not ROOT.fullmatch(a.root):
        raise Refused("--root: the window's root, 128 lowercase hex digits (a SHA-512)")
    return a.session, a.root


def free_port(lines: list[bytes], active: Callable[[str], bool], ports: range = PORTS) -> int:
    busy = {s["open"]["body"]["port"] for sid, s in sessions_of(lines).items()
            if "open" in s and "end" not in s and active(sid)}
    for port in ports:
        if port not in busy:
            return port
    raise Refused(f"every port of {ports.start}..{ports.stop - 1} has a live session")


def period_holding(lines: list[bytes], session: str) -> tuple[str, dict] | None:
    return next(((n, p) for n, p in periods_of(lines).items() if session in p["windows"]), None)


def cap_refusal(lines: list[bytes], lane: str) -> str | None:
    """Why the lane's next window would be past its open period's cap, or None."""
    for name, p in periods_of(lines).items():
        if p["lane"] == lane and p["close"] is None and p["cap"] is not None and len(p["windows"]) >= p["cap"]:
            return (f"period {name} holds its cap of {p['cap']} windows: close it (period-close --name {name}) and open "
                    "another for more")
    return None


def open_refusal(lines: list[bytes], session: str, opener: str, root: str, now: datetime) -> str | None:
    """Why `opener` may not open `session` at `root` now, or None: the session is reserved, not yet opened, unexpired,
    the opener's own (its reserve's `by` or `for`), its period still open, and the root no open's yet (v1 and v2 too)."""
    s = sessions_of(lines).get(session, {})
    if "reserve" not in s:
        return f"session {session} has no reserve in the ledger: reserve first"
    if "open" in s:
        return f"session {session} is opened already: a reserve opens once"
    rb = s["reserve"]["body"]
    if opener not in {rb["by"], rb.get("for")} - {None}:
        return f"session {session} is {lane_of(s['reserve'])}'s reserve, not {opener}'s"
    if utc(now) > rb["expires_utc"]:
        return f"session {session}'s reserve expired at {rb['expires_utc']}: it is a refused window of its period"
    if (held := period_holding(lines, session)) is not None and held[1]["close"] is not None:
        return f"session {session}'s period {held[0]} has closed, listing it unopened"
    if (sid := roots_of(lines).get(root)) is not None:
        return f"root {root[:16]}… already has its session ({sid}) in the ledger: one session a root, ever"
    return None


def gate_reserve(argv: list[str], host: Host, *, by: str, admin: bool, now: datetime, token: str,
                 reserve: Callable[[str], None], own: Callable[[Path], None]) -> dict:
    """Validate a request and hand it to `reserve` (as `auditor`), which signs the session's reserve entry: the session
    id, its expiry and the period it is a window of. Nothing serves until its open."""
    req = parse_reserve(argv, host, admin=admin)
    lines = read_ledger(host.ledger)
    if (why := cap_refusal(lines, req.get("for") or by)) is not None:
        raise Refused(why)
    session = f"s{now.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{token}"
    d = host.sessions / session
    d.mkdir(mode=0o755)
    req.update(session=session, by=by, requested_utc=utc(now))
    (d / "request.json").write_text(json.dumps(req, indent=1, sort_keys=True) + "\n")
    own(d)
    with locked(host.root / "gate.lock"):
        reserve(session)
    lines = read_ledger(host.ledger)
    entry = sessions_of(lines)[session]["reserve"]
    held = period_holding(lines, session)
    return {"session": session, "seq": entry["seq"], "peer": req["peer"], "by": by,
            **({"for": req["for"]} if "for" in req else {}), "expires_utc": entry["body"]["expires_utc"],
            "period": held and held[0]}


def gate_open(argv: list[str], host: Host, *, by: str, now: datetime, admit: Callable[[str], None],
              start: Callable[[str], None], own: Callable[[Path], None], secret: bytes,
              ready: Callable[[str], str | None]) -> dict:
    """Open a reserved session at its window's root: hand it to `admit` (as `auditor`), which signs the open entry
    binding the session and the root, and `start` its unit (serve.py with --root, so the salt opens only after it): the
    session's port on the peer, its secret, which only this answer carries, and its coin commitment's root once `ready`
    has it (serve.py signs its record before it listens), so the serving side checks the hello's commitment against it
    before its first coin."""
    session, root = parse_open(argv)
    d = host.sessions / session
    with locked(host.root / "gate.lock"):
        if (why := open_refusal(read_ledger(host.ledger), session, by, root, now)) is not None:
            raise Refused(why)
        fd = os.open(d / SECRET, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o400)
        try:
            os.write(fd, secret.hex().encode() + b"\n")
        finally:
            os.close(fd)
        req = {"session": session, "root": root, "by": by, "requested_utc": utc(now),
               "secret_sha512": hashlib.sha512(secret).hexdigest()}
        (d / OPEN_REQUEST).write_text(json.dumps(req, indent=1, sort_keys=True) + "\n")
        own(d / SECRET)
        try:
            admit(session)
        except Refused:
            for f in (SECRET, OPEN_REQUEST):
                (d / f).unlink(missing_ok=True)
            raise
        start(session)
    entry = sessions_of(read_ledger(host.ledger))[session]["open"]
    b = entry["body"]
    return {"session": session, "port": b["port"], "peer": b["peer"], "host": b["host"], "seq": entry["seq"],
            "reserve": b["reserve"], "root": root, "by": b["by"], **({"for": b["for"]} if "for" in b else {}),
            "opened_by": by, "secret": secret.hex(), "commitment": ready(session)}


def gate_status(session: str, host: Host, active: Callable[[str], bool], now: datetime) -> dict:
    if not SESSION.fullmatch(session):
        raise Refused(f"no session {session!r}")
    s = sessions_of(read_ledger(host.ledger)).get(session)
    if s is None:
        raise Refused(f"no session {session} in the ledger")
    expired = "open" not in s and "reserve" in s and utc(now) > s["reserve"]["body"]["expires_utc"]
    return {"session": session, "reserve": s.get("reserve"), "open": s.get("open"), "end": s.get("end"),
            "active": "open" in s and "end" not in s and active(session), "expired": expired,
            "commitment": commitment_of(host.sessions / session / "out")}


def gate_fetch(session: str, host: Host) -> bytes:
    """The session's directory (its request and out/, never its secret) as a gzipped tar."""
    gate_status(session, host, lambda _: False, datetime.now(timezone.utc))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(host.sessions / session, arcname=session,
                filter=lambda ti: None if ti.name == f"{session}/{SECRET}" else ti)
    return buf.getvalue()


def parse_certifier(argv: list[str], host: Host) -> dict:
    """A certifier rehearsal's request: the peer, the endpoint's port on its loopback, the minutes it may run, the
    tampering shim if any, how many requests the plan sends, how many tokens each when given, the keys' height and the
    sampling (`sampling`); on fixed traffic how many in flight when given and N2's profile, on varied traffic (`--traffic
    varied`, recorded only then) its rate, prompts' lengths or prompts on stdin, seed, EOS ids, compact cells and
    margin when given (`parse_varied`)."""
    p = argparse.ArgumentParser(prog="vy-challenger-gate certifier-rehearse", exit_on_error=False, add_help=False,
                                allow_abbrev=False)
    p.add_argument("--peer", required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--minutes", required=True)
    p.add_argument("--tamper", default=None)
    p.add_argument("--requests", default=str(CERTIFIER_REQUESTS[1]))
    p.add_argument("--in-flight", default=None)
    p.add_argument("--tokens", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--traffic", default=CERTIFIER_TRAFFIC[0])
    for flag, _ in CERTIFIER_VARIED[1:]:
        p.add_argument(flag, default=None)
    p.add_argument("--prompts", action="store_true")

    a = _parse_rehearsal(p, argv, host)
    port = bounded("--port", a.port, 1024, 65535)
    minutes = bounded("--minutes", a.minutes, *CERTIFIER_MINUTES)
    requests = bounded("--requests", a.requests, *CERTIFIER_REQUESTS)
    given = {}
    if a.tokens is not None:
        given["tokens"] = bounded("--tokens", a.tokens, *CERTIFIER_TOKENS)
    if a.tamper is not None and a.tamper not in TAMPERS:
        raise Refused(f"--tamper {a.tamper!r}: one of {list(TAMPERS)}, or none for an honest shim")
    if a.traffic not in CERTIFIER_TRAFFIC:
        raise Refused(f"--traffic {a.traffic!r}: one of {list(CERTIFIER_TRAFFIC)}")
    req = {"peer": a.peer, "port": port, "minutes": minutes, "tamper": a.tamper, "requests": requests,
           "height": a.height}
    if a.traffic == "varied":
        return req | {"traffic": "varied"} | a.sampling | given | parse_varied(a, minutes, requests)
    if (flag := next((f for f, k in CERTIFIER_VARIED[1:] if getattr(a, k) is not None),
                     "--prompts" if a.prompts else None)) is not None:
        raise Refused(f"{flag}: varied traffic's (--traffic varied)")
    if a.in_flight is not None:
        given["in_flight"] = bounded("--in-flight", a.in_flight, *CERTIFIER_IN_FLIGHT)
    profile = CERTIFIER_PROFILES[0] if a.profile is None else a.profile
    if profile not in CERTIFIER_PROFILES:
        raise Refused(f"--profile {profile!r}: one of {list(CERTIFIER_PROFILES)}")
    return req | {"profile": profile} | a.sampling | given


def parse_varied(a: argparse.Namespace, minutes: int, requests: int) -> dict:
    """Varied traffic's flags, each recorded only when given (`certifier_serve.py`'s 2 a second, 16:1024, seed 0 and
    the probe's EOS otherwise): `--rate` arrivals a second, above 0 and at most `CERTIFIER_RATE`, a plain decimal;
    `--prompt-range LO:HI` within `CERTIFIER_PROMPTS`; `--seed` 0 to `CERTIFIER_SEED`; `--eos-ids` at most
    `CERTIFIER_EOS` token ids, comma-separated; `--compact B`, B within `CERTIFIER_COMPACT`, compact egress records in
    B-byte cells instead of id events; `--margin M` within `CERTIFIER_MARGIN`, a plain decimal; `--prompts`, the
    prompts on stdin (`certifier_prompts`) in place of `--prompt-range`'s chat turns. Its requests are greedy, each
    with a seed of its own, and N2's profile is the advice, so it takes no `--temperature`, `--top-p`, `--in-flight` or
    `--profile`; and its arrivals must end within `--minutes`."""
    for flag, v in (("--in-flight", a.in_flight), ("--profile", a.profile)):
        if v is not None:
            raise Refused(f"{flag}: fixed traffic's; varied requests arrive at --rate a second and N2's profile is "
                          f"the advice")
    if a.sampling != {"temperature": 0.0, "top_p": 1.0}:
        raise Refused("--temperature, --top-p: fixed traffic's; varied requests are greedy (temperature 0, top_p 1), "
                      "each with a seed of its own")
    given: dict = {}
    if a.rate is not None:
        if not re.fullmatch(DECIMAL, a.rate) or not 0 < float(a.rate) <= CERTIFIER_RATE:
            raise Refused(f"--rate {a.rate!r}: above 0, at most {CERTIFIER_RATE:g}")
        given["rate"] = float(a.rate)
    if a.prompt_range is not None:
        lo, hi = CERTIFIER_PROMPTS
        m = re.fullmatch(r"([1-9][0-9]{0,4}):([1-9][0-9]{0,4})", a.prompt_range)
        if not m or not lo <= int(m[1]) <= int(m[2]) <= hi:
            raise Refused(f"--prompt-range {a.prompt_range!r}: LO:HI, {lo} ≤ LO ≤ HI ≤ {hi}")
        given["prompt_range"] = f"{int(m[1])}:{int(m[2])}"
    if a.seed is not None:
        if not re.fullmatch(r"0|[1-9][0-9]{0,9}", a.seed) or int(a.seed) > CERTIFIER_SEED:
            raise Refused(f"--seed {a.seed!r}: 0 to {CERTIFIER_SEED}")
        given["seed"] = int(a.seed)
    if a.eos_ids is not None:
        one = r"(0|[1-9][0-9]{0,6})"
        if not re.fullmatch(rf"{one}(,{one}){{0,{CERTIFIER_EOS - 1}}}", a.eos_ids):
            raise Refused(f"--eos-ids {a.eos_ids!r}: 1 to {CERTIFIER_EOS} token ids, comma-separated")
        given["eos_ids"] = [int(t) for t in a.eos_ids.split(",")]
    if a.compact is not None:
        given["compact"] = bounded("--compact", a.compact, *CERTIFIER_COMPACT)
    if a.margin is not None:
        lo, hi = CERTIFIER_MARGIN
        if not re.fullmatch(DECIMAL, a.margin) or not lo <= float(a.margin) <= hi:
            raise Refused(f"--margin {a.margin!r}: {lo} to {hi}, a plain decimal")
        given["margin"] = float(a.margin)
    if a.prompts:
        if "prompt_range" in given:
            raise Refused("--prompt-range: the chat turns' lengths, which --prompts replaces with the prompts on stdin")
        given["prompts"] = True
    rate = given.get("rate", CERTIFIER_VARIED_RATE)
    if (span := requests / rate) > minutes * 60:
        raise Refused(f"--minutes {minutes}: {requests} requests at {rate:g} a second arrive over {span:g} seconds: "
                      f"give more --minutes, fewer --requests or a higher --rate")
    return given


def _parse_rehearsal(p: argparse.ArgumentParser, argv: list[str], host: Host) -> argparse.Namespace:
    """A rehearsal verb's arguments, with the three every rehearsal takes, `--temperature` and `--top-p`
    (`a.sampling`) and `--height` (`a.height`, the deployed key's by default), refused (with the verb's usage) when they
    don't parse, `--peer` isn't a peer or the sampling or the height is out of its bounds."""
    def refuse(why: object) -> None:
        raise Refused(f"{p.prog.split()[-1]}: {why}; {' '.join(p.format_usage().split())}")

    p.add_argument("--temperature", default="0")
    p.add_argument("--top-p", default="1")
    p.add_argument("--height", default=str(REHEARSAL_HEIGHT[1]))
    p.error = refuse
    try:
        a = p.parse_args(argv)
    except argparse.ArgumentError as e:
        refuse(e)
    if a.peer not in peers(host):
        raise Refused(f"--peer {a.peer!r}: one of {peers(host)} (/etc/vy/peers)")
    a.sampling = sampling(a.temperature, a.top_p)
    a.height = bounded("--height", a.height, *REHEARSAL_HEIGHT)
    return a


def sampling(temperature: str, top_p: str) -> dict:
    """What every request of a rehearsal samples at: `--temperature` 0 to 2 and `--top-p` above 0 and at most 1, each
    a plain decimal of at most 4 places."""
    if not re.fullmatch(DECIMAL, temperature) or not TEMPERATURE[0] <= float(temperature) <= TEMPERATURE[1]:
        raise Refused(f"--temperature {temperature!r}: {TEMPERATURE[0]:g} to {TEMPERATURE[1]:g}")
    if not re.fullmatch(DECIMAL, top_p) or not 0 < float(top_p) <= 1:
        raise Refused(f"--top-p {top_p!r}: above 0, at most 1")
    return {"temperature": float(temperature), "top_p": float(top_p)}


def sampling_argv(req: dict) -> list[str]:
    """A rehearsal's `--temperature` and `--top-p`, always given, so that every request sends both (0 and 1 for a
    request written before the gate took them)."""
    return ["--temperature", str(req.get("temperature", 0.0)), "--top-p", str(req.get("top_p", 1.0))]


def height_argv(req: dict) -> list[str]:
    """A rehearsal's `--height`, always given, the deployed key's for a request written before the gate took it."""
    return ["--height", str(req.get("height", REHEARSAL_HEIGHT[1]))]


def bounded(flag: str, v: str, lo: int, hi: int) -> int:
    if not re.fullmatch(r"[1-9][0-9]{0,4}", v) or not lo <= int(v) <= hi:
        raise Refused(f"{flag} {v!r}: {lo} to {hi}")
    return int(v)


def warden_minutes(requests: int, height: int = REHEARSAL_HEIGHT[1], window_ms: int = WARDEN_WINDOW_MS) -> int:
    """The fewest minutes a warden rehearsal of `requests` requests takes: its traffic's last arrival (1/`WARDEN_RATE`
    seconds a request), the tail, the rest of the `window_ms` sampling block it ends in (a minute for a shorter one)
    and the check, and its key at `height` (2^height one-time keys at `KEYGEN_LEAVES_S` a second)."""
    return _minutes(round(requests / WARDEN_RATE) + max(0, -(-window_ms // 1000) - WARDEN_TAIL_S), height)


def _minutes(arrivals_s: int, height: int) -> int:
    return -(-(arrivals_s + 3 * WARDEN_TAIL_S + 2 ** height // KEYGEN_LEAVES_S) // 60)


def scan_schedule(bucket_ms: int | None = None) -> tuple[str, ...]:
    """`SCAN_SCHEDULE` at the tick `bucket_ms` (`--bucket-ms`; its own when None)."""
    at = SCAN_SCHEDULE.index("--bucket-ms") + 1
    return SCAN_SCHEDULE if bucket_ms is None else (*SCAN_SCHEDULE[:at], str(bucket_ms), *SCAN_SCHEDULE[at + 1:])


def warden_window_ms(req: dict) -> int:
    """A warden rehearsal's sampling block, in ms: T ticks of the scan's schedule at the request's tick with ``scan``,
    `WARDEN_WINDOW_MS` otherwise."""
    return SCAN["--window"] * req.get("bucket_ms", SCAN["--bucket-ms"]) if req.get("scan") else WARDEN_WINDOW_MS


def warden_windows(requests: int, scan: bool = True, bucket_ms: int = SCAN["--bucket-ms"]) -> int:
    """The most link-windows a warden rehearsal of `requests` requests runs (`warden_serve.rehearse` counts its
    traffic's last arrival and tail): `warden_minutes`'s arrivals and three tails, in link-windows of the scan's
    schedule at the tick `bucket_ms` with `scan` and of `WARDEN_WINDOW_MS` otherwise."""
    window_ms = warden_window_ms({"scan": True, "bucket_ms": bucket_ms} if scan else {})
    return -(-(round(requests / WARDEN_RATE) + 3 * WARDEN_TAIL_S) * 1000 // window_ms)


def warden_height(windows: int, heads: int = 0) -> int:
    """The least key height, at least `REHEARSAL_HEIGHT`'s, whose 2^H one-time keys sign `heads` (a load's) and then
    `windows` link-windows at `WARDEN_WINDOW_LEAVES` each."""
    return max(REHEARSAL_HEIGHT[0], (heads + WARDEN_WINDOW_LEAVES * windows - 1).bit_length())


def load_heads(shape: str) -> int:
    """The heads a load of `shape` signs before the rehearsal's first window: one a value (`load_relay.manifest`)."""
    lr = _load_relay()
    return len(lr.manifest(lr.SHAPES[shape]))


def warden_key(windows: int, height: int, what: str, heads: int = 0) -> None:
    """Refuses a warden rehearsal whose key at `height` can't sign its `windows` link-windows after a load's `heads`,
    with the least height that can (`warden_height`)."""
    if height < (least := warden_height(windows, heads)):
        load = f" after the load's {heads} heads" if heads else ""
        raise Refused(f"--height {height}: {what} take {windows} link-windows, "
                      f"{heads + WARDEN_WINDOW_LEAVES * windows} one-time signatures ({WARDEN_WINDOW_LEAVES} a "
                      f"window{load}), past the key's {1 << height}: give --height {least} or more, or fewer --requests")


def scan_group_windows(tokens: int) -> int:
    """The link-windows a `--window-per-group` group takes at the scan's schedule, each request answered by at most
    `tokens` id events (`warden_serve.group_windows`, which `test_warden_serve` holds this to)."""
    return -(-(SCAN["--delay"] + tokens // SCAN["--slots"] + 3) // SCAN["--window"])


def warden_group_minutes(groups: int, tokens: int, height: int = REHEARSAL_HEIGHT[1],
                         bucket_ms: int = SCAN["--bucket-ms"]) -> int:
    """`warden_minutes` of a `--window-per-group` run: its `groups` groups' windows, at the tick `bucket_ms`, in place
    of the arrivals."""
    return _minutes(-(-groups * scan_group_windows(tokens) * SCAN["--window"] * bucket_ms // 1000), height)


def parse_warden(argv: list[str], host: Host, *, load: bool = False) -> dict:
    """A warden rehearsal's request: the peer, the endpoint's port on its loopback, the minutes it may run (at least
    `warden_minutes` of its requests and height), how many requests its traffic sends, when given the most tokens a
    request asks for (`warden_serve.py`'s 256 otherwise) and `--scan` (the scan's schedule and traffic) with, when
    given, its tick `--bucket-ms` (`SCAN_BUCKET_MS`, which the minutes and the key's sampling blocks are counted at),
    the sampling (`sampling`) and the key's height. `--prompts` (with `--scan`) marks the request's prompts as the gate's stdin,
    and `--window-per-group` (with `--prompts`) sends them a group a link-window's span; `warden_prompts` reads them
    and checks what depends on them. With `load` (`load-relay`) it is the serving session with its load first
    (`parse_load`: the request's ``load``), whose minutes come out of `--minutes` before the rehearsal's."""
    p = argparse.ArgumentParser(prog=f"vy-challenger-gate {'load-relay' if load else 'warden-rehearse'}",
                                exit_on_error=False, add_help=False, allow_abbrev=False)
    p.add_argument("--peer", required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--minutes", required=True)
    p.add_argument("--requests", default=None)
    p.add_argument("--tokens", default=None)
    p.add_argument("--scan", action="store_true")
    p.add_argument("--bucket-ms", default=None)
    p.add_argument("--prompts", action="store_true")
    p.add_argument("--window-per-group", action="store_true")
    if load:
        p.add_argument("--sink-port", required=True)
        p.add_argument("--program", required=True)
        p.add_argument("--shape", default=LOAD_SHAPES[0])
        p.add_argument("--streams", default=str(LOAD_DEFAULT_STREAMS))
    a = _parse_rehearsal(p, argv, host)
    port = bounded("--port", a.port, 1024, 65535)
    minutes = bounded("--minutes", a.minutes, *CERTIFIER_MINUTES)
    if a.prompts and not a.scan:
        raise Refused("--prompts: the scan's id requests' prompts, with --scan")
    if a.bucket_ms is not None and not a.scan:
        raise Refused("--bucket-ms: the scan's tick, with --scan")
    if a.window_per_group and not a.prompts:
        raise Refused("--window-per-group: groups of the --prompts")
    tick = {} if a.bucket_ms is None else {"bucket_ms": bounded("--bucket-ms", a.bucket_ms, *SCAN_BUCKET_MS)}
    loaded = {"load": parse_load(a, host, port)} if load else {}
    spent = loaded["load"]["minutes"] if load else 0
    if minutes <= spent:
        raise Refused(f"--minutes {minutes}: the {loaded['load']['shape']} load takes {spent} minutes (its weights "
                      f"and witness at {LOAD_RATE / 1e6:g} MB/s and {LOAD_READY_S} s for the endpoint), and the "
                      "serving after it more")
    if a.prompts:
        return {"device": "warden", "peer": a.peer, "port": port, "minutes": minutes, "height": a.height,
                "requests": None if a.requests is None else bounded("--requests", a.requests, *WARDEN_PROMPT_REQUESTS),
                **({} if a.tokens is None else {"tokens": bounded("--tokens", a.tokens, *WARDEN_PROMPT_TOKENS)}),
                "scan": True, **tick, "prompts": True, **({"window_per_group": True} if a.window_per_group else {})} | \
            a.sampling | loaded
    requests = bounded("--requests", a.requests or "100", *WARDEN_REQUESTS)
    given = ({} if a.tokens is None else {"tokens": bounded("--tokens", a.tokens, *WARDEN_TOKENS)}) | \
        ({"scan": True} if a.scan else {}) | tick
    if minutes - spent < (least := warden_minutes(requests, a.height, warden_window_ms(given))):
        raise Refused(f"--minutes {minutes}: {requests} requests take {least} minutes (arrivals at {WARDEN_RATE} a "
                      f"second, a {WARDEN_TAIL_S} s tail, the window they end in, the check and the key at height "
                      f"{a.height}){_after_load(spent)}: give --minutes {least + spent} or more, or fewer --requests")
    warden_key(warden_windows(requests, a.scan, tick.get("bucket_ms", SCAN["--bucket-ms"])), a.height,
               f"{requests} requests", load_heads(loaded["load"]["shape"]) if load else 0)
    return {"device": "warden", "peer": a.peer, "port": port, "minutes": minutes, "requests": requests,
            "height": a.height} | a.sampling | given | loaded


def _after_load(spent: int) -> str:
    return f" after the load's {spent}" if spent else ""


def _load_relay():
    """`benchmarks/network_traces/load_relay.py`, the load's store, relay and commit, imported when a verb needs it."""
    if (here := str(TREE / "benchmarks" / "network_traces")) not in sys.path:
        sys.path.insert(0, here)
    import load_relay
    return load_relay


def load_minutes(shape: str) -> int:
    """The fewest minutes a load of `shape` takes: its weights stream and its witness (each row's salts) over the link
    at `LOAD_RATE`, then `LOAD_READY_S` for the prover's endpoint to come up on the weights."""
    lr = _load_relay()
    values = lr.manifest(lr.SHAPES[shape])
    link = values[-1].offset + values[-1].nbytes + lr.SALT_BYTES * sum(v.leaves for v in values)
    return math.ceil((link / LOAD_RATE + LOAD_READY_S) / 60)


def parse_load(a: argparse.Namespace, host: Host, port: int) -> dict:
    """A `load-relay` request's load: the shape (`LOAD_SHAPES`), whose stream the store must hold as `weights-write`
    wrote it (`load_relay.read_store`: its record's digests interface 1's), the reading program's digest (`--program`,
    128 lowercase hex digits), the sink's port on the peer's loopback (`--sink-port`, not the endpoint's), the streams
    (`LOAD_STREAMS`, a tunnel each) and its minutes (`load_minutes`)."""
    if a.shape not in LOAD_SHAPES:
        raise Refused(f"--shape {a.shape!r}: one of {list(LOAD_SHAPES)}")
    if not ROOT.fullmatch(a.program):
        raise Refused(f"--program {a.program!r}: the reading program's digest, 128 lowercase hex digits")
    sink_port = bounded("--sink-port", a.sink_port, 1024, 65535)
    if sink_port == port:
        raise Refused(f"--sink-port {sink_port}: the sink's port, not the endpoint's --port")
    streams = bounded("--streams", a.streams, *LOAD_STREAMS)
    lr = _load_relay()
    try:
        lr.read_store(host.store, lr.SHAPES[a.shape])
    except lr.Refused as e:
        raise Refused(f"the store: {e}") from None
    return {"shape": a.shape, "program": a.program, "sink_port": sink_port, "streams": streams,
            "minutes": load_minutes(a.shape)}


def parse_weights(argv: list[str], host: Host) -> dict:
    """A `weights-write` request: interface 1's weights stream of `--shape` (`LOAD_SHAPES`, h200 by default) into the
    store disk, which infra mounts, within `WEIGHTS_MINUTES`."""
    p = argparse.ArgumentParser(prog="vy-challenger-gate weights-write", exit_on_error=False, add_help=False,
                                allow_abbrev=False)
    p.add_argument("--shape", default=LOAD_SHAPES[0])

    def refuse(why: object) -> None:
        raise Refused(f"weights-write: {why}; {' '.join(p.format_usage().split())}")

    p.error = refuse
    try:
        a = p.parse_args(argv)
    except argparse.ArgumentError as e:
        refuse(e)
    if a.shape not in LOAD_SHAPES:
        raise Refused(f"--shape {a.shape!r}: one of {list(LOAD_SHAPES)}")
    if not host.store.is_dir():
        raise Refused(f"no store disk at {host.store}: infra mounts it, auditor's")
    return {"device": "weights", "shape": a.shape, "minutes": WEIGHTS_MINUTES}


def prompts_of(data: bytes, ids: tuple[int, int] = WARDEN_PROMPT_IDS) -> list[list[int]]:
    """A `--prompts` request's prompts, off the gate's stdin (`data`: at most `WARDEN_PROMPTS_BYTES`, of which the gate
    reads one byte more): JSON, an array of `WARDEN_PROMPTS` prompts, each an array of `ids` token ids (a scan
    rehearsal's `WARDEN_PROMPT_IDS`) below 2^`WARDEN_PROMPT_ID_BITS`."""
    if len(data) > WARDEN_PROMPTS_BYTES:
        raise Refused(f"--prompts: over {WARDEN_PROMPTS_BYTES} bytes on stdin")
    try:
        prompts = json.loads(data)
    except ValueError:
        prompts = None
    (lo, hi), (ilo, ihi) = WARDEN_PROMPTS, ids
    if not (isinstance(prompts, list) and lo <= len(prompts) <= hi
            and all(isinstance(p, list) and ilo <= len(p) <= ihi for p in prompts)
            and all(type(t) is int and 0 <= t < 1 << WARDEN_PROMPT_ID_BITS for p in prompts for t in p)):
        raise Refused(f"--prompts: stdin is a JSON array of {lo} to {hi} prompts, each an array of {ilo} to {ihi} "
                      f"token ids below 2^{WARDEN_PROMPT_ID_BITS}")
    return prompts


def warden_prompts(req: dict, data: bytes) -> bytes:
    """A scan rehearsal's `--prompts` (`prompts_of`), under `--window-per-group` at most the scan's lanes of them (one
    group). Completes the request: `requests` the prompts' count when not given, and `prompts` their count and ids
    alone (the ids are the prover's, `witness/` only); refused when `--minutes` is under `warden_group_minutes` of its
    groups (`warden_minutes` of its requests otherwise), at its tick. Returns the prompts' canonical bytes, the
    witness's `PROMPTS`."""
    prompts = prompts_of(data)
    n = len(prompts)
    if req.get("window_per_group") and n > SCAN["--lanes"]:
        raise Refused(f"--window-per-group: {n} prompts, past a group's {SCAN['--lanes']} (the scan's lanes)")
    requests = req["requests"] or n
    tokens, height, minutes = req.get("tokens", WARDEN_PROMPT_TOKENS[1]), req["height"], req["minutes"]
    spent, tick = req.get("load", {}).get("minutes", 0), req.get("bucket_ms", SCAN["--bucket-ms"])
    if req.get("window_per_group"):
        least, what = warden_group_minutes(-(-requests // n), tokens, height, tick), f"{-(-requests // n)} groups"
        windows = -(-requests // n) * scan_group_windows(tokens)
    else:
        least, what = warden_minutes(requests, height, warden_window_ms(req)), f"{requests} requests"
        windows = warden_windows(requests, bucket_ms=tick)
    if minutes - spent < least:
        raise Refused(f"--minutes {minutes}: {what} take {least} minutes (with the check and the key at height "
                      f"{height}){_after_load(spent)}: give --minutes {least + spent} or more, or fewer --requests")
    warden_key(windows, height, what, load_heads(req["load"]["shape"]) if "load" in req else 0)
    req.update(requests=requests, prompts={"count": n, "ids": sum(map(len, prompts))})
    return json.dumps(prompts, separators=(",", ":")).encode()


def certifier_prompts(req: dict, data: bytes) -> bytes:
    """A varied certifier rehearsal's `--prompts` (`prompts_of`, each of `CERTIFIER_PROMPTS` ids, the varied plan's
    prompt lengths: no scan reads them): request j sends the (j mod n)-th of the n (`certifier_serve.py rehearse
    --prompts-file`), at the same seed's arrivals and `max_tokens`. The request records their count and ids alone.
    Returns the prompts' canonical bytes, the witness's `PROMPTS`."""
    prompts = prompts_of(data, CERTIFIER_PROMPTS)
    req["prompts"] = {"count": len(prompts), "ids": sum(map(len, prompts))}
    return json.dumps(prompts, separators=(",", ":")).encode()


#: each verb the rehearsal unit runs, its request parser: the two rehearsals, the load (a warden rehearsal with its
#: load first) and the store's write
REHEARSALS = {"certifier-rehearse": parse_certifier, "warden-rehearse": parse_warden,
              "load-relay": lambda argv, host: parse_warden(argv, host, load=True), "weights-write": parse_weights}


def certifier_runs(host: Host) -> list[str]:
    return sorted(d.name for d in host.certifier.iterdir() if CERTIFIER_RUN.fullmatch(d.name)) \
        if host.certifier.is_dir() else []


def gate_certifier(argv: list[str], host: Host, *, by: str, now: datetime, token: str, start: Callable[[str], None],
                   own: Callable[[Path], None], active: Callable[[str], bool],
                   parse: Callable[[list[str], Host], dict] = parse_certifier,
                   read: Callable[[int], bytes] = lambda n: b"") -> dict:
    """Validate a rehearsal's request (`parse`: the certifier's or the warden's), refused while another of either runs
    (under the gate's lock), and `start` its unit, which runs `certifier-run` as `auditor`. Returns at once: `status
    RUN` follows it. A `--prompts` request's prompts are `read` off stdin (`warden_prompts`, a certifier's
    `certifier_prompts`) and written to the run's `witness/` (0700, the file 0600, both `auditor`'s), never to its
    request."""
    req = parse(argv, host)
    prompts = None if not req.get("prompts") else (
        warden_prompts if req.get("device") == "warden" else certifier_prompts)(req, read(WARDEN_PROMPTS_BYTES + 1))
    with locked(host.root / "gate.lock"):
        if (running := next((r for r in certifier_runs(host) if active(r)), None)) is not None:
            raise Refused(f"rehearsal {running} still runs: one at a time, a certifier's or a warden's "
                          f"(status {running})")
        if not host.certifier.is_dir():
            host.certifier.mkdir(mode=0o755)
            own(host.certifier)
        run = f"c{now.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{token}"
        d = host.certifier / run
        d.mkdir(mode=0o755)
        req.update(run=run, by=by, requested_utc=utc(now))
        (d / "request.json").write_text(json.dumps(req, indent=1, sort_keys=True) + "\n")
        if prompts is not None:
            (d / WITNESS).mkdir(mode=0o700)
            fd = os.open(d / WITNESS / PROMPTS, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(prompts)
            own(d / WITNESS / PROMPTS)
            own(d / WITNESS)
        own(d)
        start(run)
    return req | {"unit": f"vy-certifier@{run}.service"}


def certifier_status(run: str, host: Host, active: Callable[[str], bool]) -> dict:
    d = host.certifier / run
    if not CERTIFIER_RUN.fullmatch(run) or not (d / "request.json").is_file():
        raise Refused(f"no certifier rehearsal {run!r}")

    def read(path: Path) -> dict | None:
        return json.loads(path.read_text()) if path.is_file() else None
    end = read(d / "end.json")
    return {"run": run, "request": read(d / "request.json"), "end": end, "active": end is None and active(run)}


def certifier_fetch(run: str, host: Host, *, witness: bool = False) -> bytes:
    """The rehearsal's directory (its request, out/ and end) as a gzipped tar, with a warden rehearsal's `witness/`
    only when `witness` (the gate's: the run's own lane, or vy-admin)."""
    certifier_status(run, host, lambda _: False)
    hidden = f"{run}/{WITNESS}"

    def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        return info if witness or (info.name != hidden and not info.name.startswith(hidden + "/")) else None
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(host.certifier / run, arcname=run, filter=keep)
    return buf.getvalue()


USAGE = ("verbs: reserve --peer NAME --scheme S --families JSON [--epoch E] [--minutes M] [--for LANE: vy-admin's] "
         "| open --session S --root R "
         "| period-open --name P --cap N --circuit SHA512 [--draw-k K] [--rows T] [--model M | --weights-root W] [--circuit-files JSON] "
         "[--length JSON] [--lane LANE: vy-admin's] "
         "| period-close --name P "
         "(a lane's own gate key, or vy-admin) "
         "| model-register --model M --circuit SHA512 --descriptor FILE --checkpoint DIR (vy-admin's) "
         "| model-key --model M --circuit SHA512 [--weights-root W] (a lane's own gate key, or vy-admin) "
         "| certifier-rehearse --peer NAME --port PORT --minutes M [--tamper late|extra|drop|token] [--requests N] "
         "[--in-flight N] [--tokens N] [--profile planned|served] [--temperature T] [--top-p P] "
         "[--traffic varied [--rate R] [--prompt-range LO:HI | --prompts: the prompts on stdin] [--seed S] "
         "[--eos-ids ID,...] [--compact B] [--margin M]: no --in-flight, --profile or sampling] "
         "| warden-rehearse --peer NAME --port PORT --minutes M [--requests N] [--tokens N] [--temperature T] "
         "[--top-p P] [--scan [--bucket-ms MS] [--prompts [--window-per-group]]: the prompts on stdin] "
         "(a lane's own gate key, or vy-admin) "
         "| weights-write [--shape h200|tiny] "
         "| load-relay --peer NAME --port PORT --sink-port PORT --program SHA512 --minutes M [--shape h200|tiny] "
         "[--streams N] [warden-rehearse's flags] (a lane's own gate key, or vy-admin) "
         "| status SESSION|RUN | ledger | pins | models | pubkey | fetch SESSION|RUN")


def parse_period(verb: str, argv: list[str], *, lane: str | None) -> tuple[str, str | None, list[str]]:
    """(period, lane, [cap, circuit, then each `PERIOD_FIELDS` field as ``FIELD=VALUE`` when given] for a period-open) of a
    period-open or period-close: a lane key's own lane, or vy-admin's `--lane` (period-open)."""
    p = argparse.ArgumentParser(prog=f"vy-challenger-gate {verb}", exit_on_error=False, add_help=False, allow_abbrev=False)
    p.add_argument("--name", required=True)
    if lane is None and verb == "period-open":
        p.add_argument("--lane", required=True)
    if verb == "period-open":
        p.add_argument("--cap", required=True)
        p.add_argument("--circuit", required=True)
        p.add_argument("--draw-k", dest="draw_k", default=None)
        p.add_argument("--rows", dest="rows", default=None)
        p.add_argument("--weights-root", dest="weights_root", default=None)
        p.add_argument("--circuit-files", dest="circuit_files", default=None)
        p.add_argument("--model", dest="model", default=None)
        p.add_argument("--length", dest="length", default=None)

    def refuse(why: object) -> None:
        raise Refused(f"{verb}: {why}; {' '.join(p.format_usage().split())}")

    p.error = refuse
    try:
        a = p.parse_args(argv)
    except argparse.ArgumentError as e:
        refuse(e)
    named = getattr(a, "lane", None) or lane
    for what, v in (("--name", a.name), ("--lane", named)):
        if v is not None and (not NAME.fullmatch(v) or v in HOST_ACCOUNTS):
            raise Refused(f"{verb}: {what} {v!r}: 1 to 64 of [A-Za-z0-9._:-], starting with a letter or digit")
    if verb != "period-open":
        return a.name, named, []
    cap = cap_of(int(a.cap)) if re.fullmatch(r"[1-9][0-9]{0,8}", a.cap) else cap_of(a.cap)
    return a.name, named, [str(cap), circuit_of(a.circuit)] + [f"{f}={field_text(check(getattr(a, f)))}" for f, check
                                                               in PERIOD_FIELDS.items() if getattr(a, f) is not None]


def gate(argv: list[str], host: Host = Host(), **hooks: Any) -> tuple[int, bytes]:
    """The gate's verbs; (exit code, stdout). A refusal is exit 3 with {"refused": why}. Over ssh (`--ssh WORDS --key NAME`,
    the forced command's) `by` is NAME, the lane the key was registered for, and a key registered with no lane (`-`) reads
    only; run by root (vy-admin's sudo) `by` is the sudoer."""
    def refused(why: str) -> tuple[int, bytes]:
        return 3, json.dumps({"refused": why}).encode() + b"\n"

    by = hooks.get("by") or os.environ.get("SUDO_USER") or "root"
    admin = argv[:1] != ["--ssh"]
    if argv[:1] == ["--ssh"]:
        if len(argv) == 4 and argv[2] == "--key":
            key = argv[3]
        elif len(argv) == 2:
            key = READ_ONLY
        else:
            return refused("the gate's forced command is --ssh WORDS --key NAME")
        if key != READ_ONLY and not NAME.fullmatch(key):
            return refused(f"gate key name {key!r}")
        by = None if key == READ_ONLY else key
        if len(argv[1]) > 8192:
            return refused("the request is over 8192 bytes")
        try:
            argv = shlex.split(argv[1])
        except ValueError as e:
            return refused(f"the request doesn't parse as shell words: {e}")
    verb, rest = (argv[0], argv[1:]) if argv else ("", [])
    active = hooks.get("active") or unit_active
    now = hooks.get("now") or datetime.now(timezone.utc)
    try:
        if verb in ("reserve", "open") and by is None:
            raise Refused("this key reads only: a window is reserved and opened with its prover's own gate key, which "
                          "infra registers for its lane (send infra the public half)")
        if verb == "reserve":
            doc = gate_reserve(rest, host, by=by, admin=admin, now=now, token=hooks.get("token") or secrets.token_hex(2),
                               reserve=hooks.get("reserve") or (lambda s: as_auditor_step("reserve", s)),
                               own=hooks.get("own") or own_by_auditor)
        elif verb == "open":
            doc = gate_open(rest, host, by=by, now=now, admit=hooks.get("admit") or (lambda s: as_auditor_step("admit", s)),
                            start=hooks.get("start") or start_unit, own=hooks.get("own") or own_by_auditor,
                            secret=hooks.get("secret") or secrets.token_bytes(32),
                            ready=hooks.get("ready") or (lambda s: await_commitment(host, s, active)))
        elif verb in PERIOD_KINDS:
            if by is None:
                raise Refused("this key reads only: a period is opened and closed with its lane's own gate key, or by vy-admin")
            name, lane, extra = parse_period(verb, rest, lane=None if admin else by)
            with locked(host.root / "gate.lock"):
                e = (hooks.get("period") or as_auditor_period)([verb, name, lane or READ_ONLY, by, *extra])
            doc = {"seq": e["seq"], "kind": e["kind"], "body": e["body"]}
        elif verb == "model-register":
            if not admin:
                raise Refused("a model is registered by vy-admin alone (sudo vy-challenger-gate model-register ...)")
            a = parse_model(verb, rest)
            seal_models(host, own=hooks.get("own") or own_by_auditor, seal=hooks.get("seal") or append_only)
            line = (hooks.get("register") or as_auditor_register)(
                [a.model, a.circuit, str(Path(a.descriptor).resolve()), str(Path(a.checkpoint).resolve())])
            doc = {k: line[k] for k in ("model", "circuit", "prefix", "checkpoint_sha512", "roots", "weights_root",
                                        "key_sha512", "registered_utc")}
        elif verb == "model-key":
            if by is None:
                raise Refused("this key reads only: a model's salt key goes to a lane's own gate key, or vy-admin")
            doc = gate_model_key(rest, host)
        elif verb in REHEARSALS:
            if by is None:
                what = {"load-relay": "a load", "weights-write": "the store's write"}.get(
                    verb, f"a {verb.removesuffix('-rehearse')} rehearsal")
                raise Refused(f"this key reads only: {what} is started with a lane's own gate key, or by vy-admin")
            doc = gate_certifier(rest, host, by=by, now=now, token=hooks.get("token") or secrets.token_hex(2),
                                 start=hooks.get("start_certifier") or start_certifier_unit,
                                 own=hooks.get("own") or own_by_auditor,
                                 active=hooks.get("certifier_active") or certifier_unit_active, parse=REHEARSALS[verb],
                                 read=hooks.get("read") or sys.stdin.buffer.read)
        elif verb == "models" and not rest:
            return 0, host.models.read_bytes() if host.models.exists() else b""
        elif verb == "status" and len(rest) == 1 and CERTIFIER_RUN.fullmatch(rest[0]):
            doc = certifier_status(rest[0], host, hooks.get("certifier_active") or certifier_unit_active)
        elif verb == "fetch" and len(rest) == 1 and CERTIFIER_RUN.fullmatch(rest[0]):
            request = certifier_status(rest[0], host, lambda _: False)["request"]
            return 0, certifier_fetch(rest[0], host, witness=admin or (by is not None and by == request.get("by")))
        elif verb == "status" and len(rest) == 1:
            doc = gate_status(rest[0], host, active, now)
        elif verb == "ledger" and not rest:
            return 0, host.ledger.read_bytes() if host.ledger.exists() else b""
        elif verb == "pins" and not rest:
            return 0, host.pins.read_bytes() if host.pins.exists() else b""
        elif verb == "pubkey" and not rest:
            doc = json.loads((host.root / "pubkey.json").read_text())
        elif verb == "fetch" and len(rest) == 1:
            return 0, gate_fetch(rest[0], host)
        else:
            raise Refused(USAGE)
    except Refused as e:
        return refused(str(e))
    return 0, json.dumps(doc, sort_keys=True).encode() + b"\n"


def _auditor(args: list[str]) -> list[str]:
    return ["runuser", "-u", "auditor", "--", "/usr/bin/env", "-i", "PYTHONDONTWRITEBYTECODE=1", "HOME=/var/lib/vy-challenger",
            "PATH=/usr/bin:/bin", "/usr/bin/python3", "-I", str(Path(__file__).resolve()), *args]


def as_auditor_step(verb: str, session: str) -> None:
    p = subprocess.run(_auditor([verb, session]), capture_output=True, text=True)
    if p.returncode:
        raise Refused(p.stderr.strip().splitlines()[-1].removeprefix("refused: ") if p.stderr.strip()
                      else f"{verb} exited {p.returncode}")


def as_auditor_period(args: list[str]) -> dict:
    p = subprocess.run(_auditor(args), capture_output=True, text=True)
    if p.returncode:
        raise Refused(p.stderr.strip().splitlines()[-1].removeprefix("refused: ") if p.stderr.strip()
                      else f"{args[0]} exited {p.returncode}")
    return json.loads(p.stdout)


def own_by_auditor(path: Path) -> None:
    import pwd
    pw = pwd.getpwnam("auditor")
    os.chown(path, pw.pw_uid, pw.pw_gid)


def start_unit(session: str) -> None:
    """Blocks until the unit runs (Type=exec), so the next `admit`, under the gate's lock, sees it active."""
    p = subprocess.run(["systemctl", "start", f"vy-challenger@{session}.service"], capture_output=True, text=True)
    if p.returncode:
        raise Refused(f"session {session} was admitted but its unit didn't start ({p.stderr.strip()}): its window is spent, "
                      f"an open entry with no end")


def unit_active(session: str) -> bool:
    return subprocess.run(["systemctl", "is-active", "--quiet", f"vy-challenger@{session}.service"]).returncode == 0


def start_certifier_unit(run: str) -> None:
    p = subprocess.run(["systemctl", "start", f"vy-certifier@{run}.service"], capture_output=True, text=True)
    if p.returncode:
        raise Refused(f"certifier rehearsal {run}'s unit didn't start ({p.stderr.strip()})")


def certifier_unit_active(run: str) -> bool:
    return subprocess.run(["systemctl", "is-active", "--quiet", f"vy-certifier@{run}.service"]).returncode == 0


# --- the session (auditor) ----------------------------------------------------------------------------------------------------

def request_of(host: Host, session: str, name: str = "request.json") -> dict:
    if not SESSION.fullmatch(session):
        raise Refused(f"no session {session!r}")
    return json.loads((host.sessions / session / name).read_text())


def reserve(host: Host, session: str, *, signer, pinned: str, now: datetime) -> dict:
    """The window's `reserve` entry, or Refused when the session is in the ledger or the lane's open period holds its
    cap. From this entry on the window is one of that period's; it expires unopened `minutes` after `started_utc`."""
    req = request_of(host, session)
    I, sign, key_sha512 = signer

    def body_of(lines: list[bytes]) -> dict:
        if session in sessions_of(lines):
            raise Refused(f"session {session} is in the ledger already")
        if (why := pins_refusal(host, session, pinned)[1]) is not None:
            raise Refused(why)
        if (why := cap_refusal(lines, req.get("for") or req["by"])) is not None:
            raise Refused(why)
        return {"by": req["by"], **({"for": req["for"]} if "for" in req else {}), "peer": req["peer"],
                "scheme": req["scheme"], "epoch": req["epoch"], "families": req["families"], "minutes": req["minutes"],
                "pinned": pinned, "key_sha512": key_sha512, "host": f"{host.name}.{session}", "started_utc": utc(now),
                "expires_utc": later(utc(now), req["minutes"])}
    return append(host, "reserve", session, body_of, I, sign)


def admit(host: Host, session: str, *, signer, pinned: str, now: datetime, active: Callable[[str], bool]) -> dict:
    """The window's `open` entry, binding the reserved session and its root, carrying its reserve's request and a port of
    its peer's certifier coin link (`coin_ports`) no live session holds; or Refused (`open_refusal`, read under the
    ledger's lock)."""
    req = request_of(host, session, OPEN_REQUEST)
    I, sign, key_sha512 = signer

    def body_of(lines: list[bytes]) -> dict:
        if (why := open_refusal(lines, session, req["by"], req["root"], now)) is not None:
            raise Refused(why)
        seal, why = pins_refusal(host, session, pinned)
        if why is not None:
            raise Refused(why)
        r = sessions_of(lines)[session]["reserve"]
        return {"window": {"root": req["root"]}, "reserve": r["seq"],
                **{k: r["body"][k] for k in RESERVED if k in r["body"]}, "opened_by": req["by"], "pinned": pinned,
                "pins": seal, "key_sha512": key_sha512, "host": r["body"]["host"],
                "port": free_port(lines, active, coin_ports(host, r["body"]["peer"])), "secret_sha512": req["secret_sha512"], "started_utc": utc(now)}
    return append(host, "open", session, body_of, I, sign)


def period(host: Host, args: list[str], *, signer, pinned: str, now: datetime, active: Callable[[str], bool],
           tree: Path = TREE) -> dict:
    """`period-open NAME LANE BY CAP CIRCUIT [draw_k=K] [rows=T] [weights_root=W] [circuit_files=JSON]` or `period-close NAME LANE|- BY` (LANE - for
    vy-admin's close of any lane's period): the signed entry, or Refused. A period's name is used once and a lane has one
    period open at a time; its open commits the most windows it may hold (`cap`), its circuit and, when given, its draw
    K (`draw_k`) and table rows (`rows`), its model (`models.opened`: the registry's weights' root and model for the circuit, or for a circuit
    the registry has no entry of the model alone, from core's pin in `tree`), its fixed length (`length`) and each K's
    circuit file's SHA-512
    (`circuit_files`); it closes once none of its windows
    still runs, its close listing each window with its outcome."""
    if len(args) not in (range(6, 6 + len(PERIOD_FIELDS) + 1) if args[:1] == ["period-open"] else (4,)):
        raise Refused(f"period: {args}")
    verb, name, lane, by = args[:4]
    if verb == "period-open":
        cap = cap_of(int(args[4])) if re.fullmatch(r"[1-9][0-9]{0,8}", args[4]) else cap_of(args[4])
        circuit = circuit_of(args[5])
        fields = dict(x.partition("=")[::2] for x in args[6:])
        if len(fields) != len(args) - 6 or not set(fields) <= set(PERIOD_FIELDS):
            raise Refused(f"period: {args}")
        given = {f: PERIOD_FIELDS[f](fields[f]) for f in PERIOD_FIELDS if f in fields}
        model, root = given.pop("model", None), given.pop("weights_root", None)
        if model is not None or root is not None:
            M = registry()
            try:
                given.update(M.opened(M.read(host.models), tree, circuit, model, root))
            except M.Refused as e:
                raise Refused(str(e)) from None
    I, sign, key_sha512 = signer

    def body_of(lines: list[bytes]) -> dict:
        periods = periods_of(lines)
        if verb == "period-open":
            if name in periods:
                raise Refused(f"period {name} is in the ledger already: a period's name is used once")
            if (o := next((n for n, p in periods.items() if p["lane"] == lane and p["close"] is None), None)) is not None:
                raise Refused(f"lane {lane} has period {o} open: close it first (periods of a lane never overlap)")
            return {"period": name, "lane": lane, "by": by, "cap": cap, "circuit": circuit, **given, "pinned": pinned,
                    "key_sha512": key_sha512, "opened_utc": utc(now)}
        p = periods.get(name)
        if p is None or p["close"] is not None:
            raise Refused(f"no open period {name}")
        if lane != READ_ONLY and p["lane"] != lane:
            raise Refused(f"period {name} is lane {p['lane']}'s: only its lane's key or vy-admin closes it")
        sessions = sessions_of(lines)
        for w in p["windows"]:
            s = sessions[w]
            if "open" not in s and utc(now) <= s["reserve"]["body"]["expires_utc"]:
                raise Refused(f"period {name}'s window {w} is reserved until {s['reserve']['body']['expires_utc']}: a "
                              "period closes once each of its windows has ended or expired")
            if "open" in s and "end" not in s and active(w):
                raise Refused(f"period {name}'s window {w} still runs: a period closes once each of its windows has "
                              "ended or expired")
        outcomes = window_outcomes(lines, p["windows"])
        return {"period": name, "lane": p["lane"], "by": by, "windows": outcomes,
                "complete": bool(outcomes) and all(w["complete"] for w in outcomes), "closed_utc": utc(now)}

    if verb not in PERIOD_KINDS or not NAME.fullmatch(name) or not (lane == READ_ONLY or NAME.fullmatch(lane)):
        raise Refused(f"period: {args}")
    return append(host, verb, None, body_of, I, sign)


def serve_argv(host: Host, session: str, open_body: dict, tree: Path = TREE) -> list[str]:
    """serve.py for the session: it listens on this host's 127.0.0.1 at the open's port, which its peer's certifier coin
    link carries to the peer (no `--peer`: serve.py opens no tunnel of its own)."""
    return [sys.executable, "-I", str(tree / "benchmarks" / "live_challenger" / "serve.py"),
            "--out", str(host.sessions / session / "out"), "--port", str(open_body["port"]), "--scheme", open_body["scheme"],
            "--families", json.dumps(open_body["families"]), "--epoch", str(open_body["epoch"]),
            "--key", str(host.key), "--host", open_body["host"],
            "--root", open_body["window"]["root"], "--secret-file", str(host.sessions / session / SECRET),
            "--session", session, "--sign-lock", str(host.root / "sign.lock")]


def coin_link_argv(host: Host, session: str, open_body: dict, tree: Path = TREE) -> list[str]:
    """The session's coin link (`coin_link.py`): from the open's port on its peer to serve.py's on this host, for one
    minute more than the window's, at D7's parameters."""
    return [sys.executable, "-I", str(tree / "benchmarks" / "live_challenger" / "coin_link.py"),
            "--out", str(host.sessions / session / "out"), "--port", str(open_body["port"]), "--peer", open_body["peer"],
            "--peers", str(host.peers), "--minutes", str(open_body["minutes"] + 1)]


def stop_link(proc: subprocess.Popen, seconds: float = LINK_STOP_SECONDS) -> int:
    """Stop the coin link: SIGTERM, on which it runs to the end of its link-window, writes its record and closes its
    tunnel; past `seconds`, SIGKILL to its process group, tunnel included."""
    try:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
        return proc.wait(seconds)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        return proc.wait()


def first_line(proc: subprocess.Popen, seconds: float) -> bytes | None:
    """`proc`'s first line of output (empty if it exited first), or None if it printed none within `seconds`, in which
    case its process group is killed."""
    got: list[bytes] = []
    reader = threading.Thread(target=lambda: got.append(proc.stdout.readline()), daemon=True)
    reader.start()
    reader.join(seconds)
    if got:
        return got[0]
    with suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    reader.join()
    return None


def watch_link(link: subprocess.Popen, serve: subprocess.Popen, served: threading.Event, ended: list[str]) -> None:
    """Wait for the coin link. Should it exit before `served` is set (its tunnel ended), stop serve.py, since no coin
    can reach the peer any more, and say so in `ended`."""
    rc = link.wait()
    if not served.is_set():
        ended.append(f"the coin link ended first (exit {rc}; coin-link.log), so no coin could reach the peer")
        with suppress(ProcessLookupError):
            os.killpg(serve.pid, signal.SIGTERM)


def record_of(out: Path):
    """The challenger record the session signed in `out` (`remote.RECORD`), with its JSON; (None, None) until it has one."""
    from verity.core.protocols.sampled_proofs import challenger as C
    from verity.core.protocols.sampled_proofs import remote
    try:
        doc = json.loads((out / remote.RECORD).read_text())
        return C.ChallengerRecord.from_json(doc), doc
    except (OSError, ValueError, KeyError, TypeError, C.CoinError):
        return None, None


def commitment_of(out: Path) -> str | None:
    """The coin commitment's root (hex) of the record the session signed, None until it has one."""
    record, _ = record_of(out)
    return None if record is None else record.commitment.root.hex()


def await_commitment(host: Host, session: str, active: Callable[[str], bool],
                     seconds: float = START_SECONDS) -> str | None:
    """The session's commitment root once serve.py has signed its record (before it listens), or None if its unit stopped
    first or `seconds` passed."""
    deadline = time.monotonic() + seconds
    while (root := commitment_of(host.sessions / session / "out")) is None and active(session) and time.monotonic() < deadline:
        time.sleep(0.2)
    return root


def released_of(out: Path) -> tuple[dict[str, int] | None, int | None]:
    """(coins released by family, counted from the release log under the commitment the record signs; the record's leaf),
    each None when the session left no such file."""
    from verity.core.protocols.sampled_proofs import challenger as C
    from verity.core.protocols.sampled_proofs import remote
    record, doc = record_of(out)
    if record is None:
        return None, None
    try:
        released = {label: len(r) for label, r in remote.released_roots(out / remote.RELEASES, record.commitment).items()}
    except C.CoinError:
        released = None
    return released, leaf_of(doc["signature"])


def run(host: Host, session: str, *, signer, pinned: str, now: Callable[[], datetime], argv: list[str] | None = None,
        wait: Callable[[subprocess.Popen, float], int] | None = None, link: list[str] | None = None) -> dict:
    """Serve the admitted session (serve.py, which takes `sign.lock` for each of its signatures) through its coin link,
    started once serve.py listens and stopped once it exits (a link that refuses to start, doesn't listen within
    `START_SECONDS` or ends first ends the session: `first_line`, `watch_link`), then append its `end` entry and delete
    its secret. Its
    client closed it when serve.py exited 0 and its signed close (`challenger.json`) says so (`ended` null). The link is
    `coin_link_argv`'s; a caller that brings its own serve.py (`argv`: a local gate's, a test's) brings its own `link`,
    None for none, its client then reaching serve.py's port itself."""
    I, sign, _ = signer
    s = sessions_of(read_ledger(host.ledger)).get(session, {})
    if "open" not in s or "end" in s:
        raise Refused(f"session {session} has no open entry, or has ended: nothing to serve")
    body = s["open"]["body"]
    out = host.sessions / session / "out"
    rc, why, link_ended = 2, None, []
    if body["pinned"] != pinned:
        why = f"admitted at {body['pinned']}, but this tree is {pinned}: the pin moved, so the session doesn't run"
    else:
        if argv is None:
            argv, link = serve_argv(host, session, body), coin_link_argv(host, session, body)
        out.mkdir(mode=0o755, exist_ok=True)
        with open(out / "serve.log", "wb") as log:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=log, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
            first = proc.stdout.readline()
        with open(out / "stdout.log", "wb") as f:
            f.write(first)
        linked, watcher, served = None, None, threading.Event()
        if link is not None and first.startswith(b"listening "):
            with open(out / "coin-link.log", "wb") as log:
                linked = subprocess.Popen(link, stdout=subprocess.PIPE, stderr=log, stdin=subprocess.DEVNULL,
                                          start_new_session=True)
                listening = first_line(linked, START_SECONDS)
                log.write(listening or b"")
            if listening is None:
                why = (f"the coin link didn't listen within {START_SECONDS:g} s, so it was killed and no coin could reach "
                       f"the peer (coin-link.log)")
                os.killpg(proc.pid, signal.SIGTERM)
            elif listening.startswith(b"listening "):
                watcher = threading.Thread(target=watch_link, args=(linked, proc, served, link_ended), daemon=True)
                watcher.start()
            else:
                why = f"the coin link didn't start (exit {linked.wait()}; coin-link.log), so no coin could reach the peer"
                os.killpg(proc.pid, signal.SIGTERM)
        rc = (wait or _wait)(proc, body["minutes"] * 60) if why is None else proc.wait()
        served.set()
        proc.stdout.close()
        if linked is not None:
            stop_link(linked)
            linked.stdout.close()
        if watcher is not None:
            watcher.join()
    (host.sessions / session / SECRET).unlink(missing_ok=True)
    summary = {}
    try:
        summary = json.loads((out / "challenger.json").read_text())
    except (OSError, ValueError):
        pass
    if not isinstance(summary, dict):
        summary = {}
    files = {n: hashlib.sha512((out / n).read_bytes()).hexdigest() for n in OUT_FILES if (out / n).is_file()}
    released, record_leaf = released_of(out)
    hellos = (out / "hellos.jsonl").read_text().count("\n") if (out / "hellos.jsonl").is_file() else 0
    complete = released is not None and all(released.get(label) == n for label, n in body["families"])
    closed = rc == 0 and "ended" in summary and summary["ended"] is None
    why = why or (None if closed else next(iter(link_ended), None))
    end = {"rc": rc, "closed": closed,
           "ended": why or summary.get("ended") or ("closed by the client" if closed else f"serve.py exited {rc}"),
           "files": files, "released": released,
           "complete": complete, "record_leaf": record_leaf, "commitment": commitment_of(out), "refused_hellos": hellos,
           "ended_utc": utc(now())}
    return append(host, "end", session, lambda lines: end, I, sign)


def _wait(proc: subprocess.Popen, seconds: float) -> int:
    try:
        return proc.wait(seconds)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        return proc.wait()


# --- the certifier rehearsal (auditor) ----------------------------------------------------------------------------------------

def free_local_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tunnel_up(tunnel: subprocess.Popen, port: int, seconds: float = 60.0) -> bool:
    """Whether the tunnel's local end listens on 127.0.0.1:`port` before it exits or `seconds` pass."""
    import socket
    deadline = time.monotonic() + seconds
    while tunnel.poll() is None and time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return True
        except OSError:
            time.sleep(0.2)
    return False


def certifier_command(req: dict, out: Path, url: str, tree: Path = TREE) -> list[str]:
    """`certifier_serve.py rehearse` of the request, the endpoint at `url` (the tunnel's local end): the honest shim, or
    the request's tamper, with N2 on the request's profile (planned for a request written before it had one), the
    request's in-flight count and tokens only when it has them, and its keys' height (`height_argv`) and sampling
    (`sampling_argv`) always. A varied
    request's (`--traffic varied`) passes its varied flags instead when it has them, its prompts (`--prompts-file`, the
    gate's `PROMPTS` in the run's `witness/`, beside `out`) when it has them, and its run as `--name`, so each
    request's `user` upstream, `RUN.j`, is no other run's."""
    varied = req.get("traffic") == "varied"
    flags = CERTIFIER_VARIED if varied else (("--in-flight", "in_flight"), ("--tokens", "tokens"))
    given = [w for flag, key in flags if key in req for w in (flag, _flag_value(req[key]))]
    prompts = ["--prompts-file", str(out.parent / WITNESS / PROMPTS)] if varied and req.get("prompts") else []
    traffic = ["--traffic", "varied", "--name", req["run"]] if varied else \
        ["--profile", req.get("profile", CERTIFIER_PROFILES[0])]
    return [sys.executable, str(tree / "benchmarks" / "network_traces" / "certifier_serve.py"), "rehearse",
            "--upstream", url, "--out", str(out), *height_argv(req), "--requests", str(req["requests"]), *given,
            *prompts, *sampling_argv(req), *traffic, "--strategies", req["tamper"] or "honest"]


def _flag_value(v: object) -> str:
    """A recorded flag's value as `certifier_serve.py` reads it, EOS ids comma-separated."""
    return ",".join(map(str, v)) if isinstance(v, list) else str(v)


def certifier_env(tree: Path = TREE) -> dict[str, str]:
    """The rehearsal's environment: this unit's PATH (the pinned tree's Lean toolchain first, which builds
    network-records on first use) and the pinned tree for `verity`, since the rehearsal's processes don't run
    isolated (`-I`) as this one does."""
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/var/lib/vy-challenger"),
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(tree)}


def warden_command(req: dict, out: Path, url: str, tree: Path = TREE, *, sinks: list[tuple[str, int]] = (),
                   store: Path | None = None) -> list[str]:
    """`warden_serve.py rehearse` of the request: the warden and its client here, the warden's upstream the endpoint
    at `url` (the tunnel's local end), its key at the request's height (`height_argv`), the request's number of
    requests on the gate's traffic (`WARDEN_RATE`, `WARDEN_SEED`, `WARDEN_TAIL_S`) at its sampling (`sampling_argv`),
    for a scan rehearsal id requests (`--ids`) at the scan's schedule at the request's tick (`scan_schedule`), of its
    prompts (`--prompts`, the gate's `PROMPTS` in `out`, the witness) and in groups (`--window-per-group`) when the
    request says so, and the most tokens a request asks for only when the request has it. A `load-relay` request's
    load goes first: the `store`'s stream of its shape relayed to `sinks` (the sink tunnels' local ends) for its
    program, the endpoint given `LOAD_READY_S` to come up after it."""
    scan = ["--ids", *scan_schedule(req.get("bucket_ms"))] if req.get("scan") else []
    prompts = ["--prompts", str(out / PROMPTS)] if req.get("prompts") else []
    grouped = ["--window-per-group"] if req.get("window_per_group") else []
    given = ["--max-tokens", str(req["tokens"])] if "tokens" in req else []
    load = req.get("load")
    loaded = [] if not load else ["--load-store", str(store), "--load-shape", load["shape"], "--load-sinks",
                                  ",".join(f"{h}:{p}" for h, p in sinks), "--load-program", load["program"],
                                  "--load-ready-s", str(LOAD_READY_S)]
    return [sys.executable, str(tree / "benchmarks" / "network_traces" / "warden_serve.py"), "rehearse",
            "--upstream", url.removeprefix("http://"), "--out", str(out), *height_argv(req),
            "--requests", str(req["requests"]),
            "--rate", str(WARDEN_RATE), "--seed", str(WARDEN_SEED), "--tail", str(WARDEN_TAIL_S), *sampling_argv(req),
            *scan, *prompts, *grouped, *given, *loaded]


def weights_command(req: dict, out: Path, store: Path, tree: Path = TREE) -> list[str]:
    """`load_relay.py write` of a `weights-write` request: its shape's stream into `store` over every core, its record
    (less the digests) to `out/store.json`."""
    return [sys.executable, str(tree / "benchmarks" / "network_traces" / "load_relay.py"), "write", "--store",
            str(store), "--shape", req["shape"], "--workers", str(os.cpu_count() or 1),
            "--out", str(out / "store.json")]


#: the counts of `warden_serve.py check`'s summary a warden rehearsal's end carries: none is a cell's content or a
#: connection's own (a load's are its values and how many registered)
WARDEN_COUNTS = ("requests", "ended", "refused", "failed", "windows", "certified_windows", "certificates",
                 "signer_refused", "cell_rows", "cells", "wire_bits", "cells_in_their_bucket", "tick_lateness_ms",
                 "load")


def warden_outcome(out: Path, witness: Path) -> dict:
    """A warden rehearsal's public outcome, from what `warden_serve.py rehearse` wrote to `witness`:
    `WARDEN_PUBLIC`'s files copied to `out`, `certified.jsonl`'s certificates without the heads and salts they sign
    (`out/certificates.jsonl`), each registration's root by value, the check's verdict (`accepted`, how many problems,
    `WARDEN_COUNTS`), each witness file by name and size alone, and a load's relay (`load-relay.json`: its counts and
    the sink's acknowledgements, or why it was refused)."""
    for name in WARDEN_PUBLIC:
        if (witness / name).is_file():
            shutil.copyfile(witness / name, out / name)
    if (witness / "certified.jsonl").is_file():
        docs = [json.loads(x) for x in (witness / "certified.jsonl").read_text().splitlines()]
        (out / "certificates.jsonl").write_text("".join(
            json.dumps({k: doc[k] for k in ("certificate", "refused") if k in doc}, sort_keys=True) + "\n"
            for doc in docs))
    cells = (out / "cells.jsonl").read_text().splitlines() if (out / "cells.jsonl").is_file() else []
    registrations = {r["value"]: r["root"] for r in (json.loads(x).get("registration") for x in cells) if r}
    try:
        check = json.loads((witness / "check.json").read_text())
    except (OSError, ValueError):
        check = None
    verdict = None
    if isinstance(check, dict):
        summary = check.get("summary") or {}
        verdict = {"accepted": check.get("accepted"), "problems": len(check.get("problems") or []),
                   **{k: summary[k] for k in WARDEN_COUNTS if k in summary}}
    sizes = {str(p.relative_to(witness)): p.stat().st_size for p in sorted(witness.rglob("*")) if p.is_file()}
    try:
        load = json.loads((out / "load-relay.json").read_text())
    except (OSError, ValueError):
        load = None
    return {"verdict": verdict, "registrations": registrations, "witness": sizes,
            **({} if load is None else {"load": load})}


def forward_argv(ssh: list[str], local: int, remote: int) -> list[str]:
    """The tunnel from this host's 127.0.0.1:`local` to 127.0.0.1:`remote` on the peer `ssh` reaches."""
    return [*TUNNEL_SSH, "-L", f"127.0.0.1:{local}:127.0.0.1:{remote}", *ssh]


def _local_ports(n: int) -> list[int]:
    ports: list[int] = []
    while len(ports) < n:
        if (p := free_local_port()) not in ports:
            ports.append(p)
    return ports


def certifier_run(host: Host, run: str, *, pinned: str, now: Callable[[], datetime],
                  tunnel_argv: Callable[[list[str], int], list[str]] | None = None,
                  sink_argv: Callable[[list[str], int, int], list[str]] | None = None,
                  up: Callable[[subprocess.Popen, int], bool] = tunnel_up,
                  command: Callable[..., list[str]] | None = None,
                  wait: Callable[[subprocess.Popen, float], int] | None = None) -> dict:
    """Run the requested rehearsal (the module docstring's "A certifier rehearsal", "A warden rehearsal" and "The
    load"), as `auditor`: open the tunnel to the peer's endpoint (and for a load one to its sink a stream,
    `sink_argv`), run the rehearsal through them for at most the request's minutes, close the tunnels, and write
    `end.json`. A warden rehearsal writes to `witness/`, and its public outcome (`warden_outcome`) to `out/` and the
    end. A store's write opens no tunnel: it runs `weights_command` (`command(req, out, store)`) and its end carries
    the store's record."""
    if not CERTIFIER_RUN.fullmatch(run):
        raise Refused(f"no certifier rehearsal {run!r}")
    d = host.certifier / run
    req = json.loads((d / "request.json").read_text())
    warden, weights, load = req.get("device") == "warden", req.get("device") == "weights", req.get("load")
    command = command or (warden_command if warden else weights_command if weights else certifier_command)
    out = d / "out"
    if out.exists():
        raise Refused(f"certifier rehearsal {run} has run already")
    out.mkdir(mode=0o755)
    if weights:
        return _weights_run(host, d, req, pinned=pinned, now=now, command=command, wait=wait)
    work = out
    if warden:
        work = d / WITNESS
        work.mkdir(mode=0o700, exist_ok=True)
    sys.path.insert(0, str(HERE))
    import serve
    local, *sinks = _local_ports(1 + (load["streams"] if load else 0))
    ssh = serve.peer_ssh(host.peers, req["peer"])
    argv = (tunnel_argv or (lambda ssh, local: forward_argv(ssh, local, req["port"])))(ssh, local)
    rc, why, tunnels = 2, None, []
    with open(out / "tunnel.log", "wb") as log:
        tunnels.append(subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
        for s in sinks:
            tunnels.append(subprocess.Popen((sink_argv or forward_argv)(ssh, s, load["sink_port"]),
                                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
    try:
        if not up(tunnels[0], local):
            why = f"the tunnel to {req['peer']}'s 127.0.0.1:{req['port']} didn't come up (tunnel.log)"
        elif not all(up(t, s) for t, s in zip(tunnels[1:], sinks)):
            why = f"a tunnel to {req['peer']}'s sink at 127.0.0.1:{load['sink_port']} didn't come up (tunnel.log)"
        else:
            to = {"sinks": [("127.0.0.1", s) for s in sinks], "store": host.store} if load else {}
            with open(out / "rehearse.log", "wb") as log:
                proc = subprocess.Popen(command(req, work, f"http://127.0.0.1:{local}", **to), stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, env=certifier_env(),
                                        start_new_session=True)
            rc = (wait or _wait)(proc, req["minutes"] * 60)
            if rc == -signal.SIGTERM:
                why = f"stopped at --minutes {req['minutes']}"
    finally:
        for tunnel in tunnels:
            tunnel.terminate()
            tunnel.wait()
    if warden:
        outcome = {"device": "warden", **warden_outcome(out, work)}
    else:
        try:
            summary = json.loads((out / "summary.json").read_text())
        except (OSError, ValueError):
            summary = {}
        outcome = {"runs": {s: {"verdict": r.get("verdict"),
                                "network": (r.get("network_verdict") or {}).get("accepted"),
                                "audit": r.get("audit_accepted"), "salt": r.get("salt_accepted"),
                                "salt_again": r.get("salt_again")}
                            for s, r in summary.get("runs", {}).items()}}
    return _end(d, rc, why, outcome, pinned=pinned, now=now)


def _end(d: Path, rc: int, why: str | None, outcome: dict, *, pinned: str, now: Callable[[], datetime]) -> dict:
    """A run's `end.json`: its exit, why it ended, `outcome`, and the SHA-512 of each file of `out/`."""
    out = d / "out"
    files = {str(p.relative_to(out)): hashlib.sha512(p.read_bytes()).hexdigest()
             for p in sorted(out.rglob("*")) if p.is_file()}
    end = {"run": d.name, "rc": rc,
           "ended": why or ("the rehearsal ended" if rc == 0 else f"the rehearsal exited {rc}"),
           "pinned": pinned, **outcome, "files": files, "ended_utc": utc(now())}
    (d / "end.json").write_text(json.dumps(end, indent=1, sort_keys=True) + "\n")
    return end


def _weights_run(host: Host, d: Path, req: dict, *, pinned: str, now: Callable[[], datetime],
                 command: Callable[[dict, Path, Path], list[str]],
                 wait: Callable[[subprocess.Popen, float], int] | None) -> dict:
    """A `weights-write` run: `command` (`weights_command`) for at most `WEIGHTS_MINUTES`, its end the store's record
    (`out/store.json`: written, present or refused)."""
    out, why = d / "out", None
    with open(out / "rehearse.log", "wb") as log:
        proc = subprocess.Popen(command(req, out, host.store), stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, env=certifier_env(), start_new_session=True)
    rc = (wait or _wait)(proc, req["minutes"] * 60)
    if rc == -signal.SIGTERM:
        why = f"stopped at {req['minutes']} minutes"
    try:
        store = json.loads((out / "store.json").read_text())
    except (OSError, ValueError):
        store = None
    return _end(d, rc, why, {"device": "weights", "store": store}, pinned=pinned, now=now)


# --- main ---------------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    host = Host()
    if argv[:1] == ["gate"]:
        rc, out = gate(argv[1:], host)
        sys.stdout.buffer.write(out)
        return rc
    try:
        if argv[:1] == ["reserve"] and len(argv) == 2:
            e = reserve(host, argv[1], signer=key_signer(host), pinned=pinned_commit(), now=datetime.now(timezone.utc))
            print(json.dumps({"seq": e["seq"], "session": e["session"]}))
        elif argv[:1] == ["admit"] and len(argv) == 2:
            e = admit(host, argv[1], signer=key_signer(host), pinned=pinned_commit(), now=datetime.now(timezone.utc),
                      active=unit_active)
            print(json.dumps({"seq": e["seq"], "session": e["session"]}))
        elif argv[:1] in (["period-open"], ["period-close"]):
            e = period(host, argv, signer=key_signer(host), pinned=pinned_commit(), now=datetime.now(timezone.utc),
                       active=unit_active)
            print(json.dumps(e))
        elif argv[:1] == ["run"] and len(argv) == 2:
            e = run(host, argv[1], signer=key_signer(host), pinned=pinned_commit(), now=lambda: datetime.now(timezone.utc))
            print(json.dumps({"seq": e["seq"], "session": e["session"], "rc": e["body"]["rc"]}))
        elif argv[:1] == ["certifier-run"] and len(argv) == 2:
            e = certifier_run(host, argv[1], pinned=pinned_commit(), now=lambda: datetime.now(timezone.utc))
            print(json.dumps({k: e[k] for k in ("run", "rc", "ended", "runs", "device") if k in e}))
        elif argv[:1] == ["model-register"] and len(argv) == 5:
            line = model_register(host, argv[1:], pinned=pinned_commit(), now=datetime.now(timezone.utc))
            print(json.dumps(line, sort_keys=True))
        elif argv[:1] == ["pinned"] and len(argv) == 3:
            print(record_pin(host.pins, argv[1], argv[2], datetime.now(timezone.utc)))
        elif argv[:1] == ["keygen"] and len(argv) == 1:
            from verity.ml.crypto import lms
            key, state = load_key(host)
            state.close()
            public = lms.public_key(key)
            doc = {"W": key.params.W, "H": key.params.H, "public": public.hex(),
                   "public_sha512": hashlib.sha512(public).hexdigest()}
            (host.root / "pubkey.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
            print(json.dumps(doc))
        else:
            print("usage: auditor.py gate VERB ... | reserve SESSION | admit SESSION | run SESSION | period-open NAME LANE BY CAP CIRCUIT [DRAW_K] "
                  "| period-close NAME LANE|- BY | certifier-run RUN | pinned COMMIT MAIN | keygen", file=sys.stderr)
            return 2
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
