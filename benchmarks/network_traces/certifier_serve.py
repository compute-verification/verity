"""certifier-serve: the network certifier (`active.Proxy`) on one host, as the route of the measured link in front of a
served vLLM OpenAI endpoint, and the offline check of what it published.

**Topology.** The client, outside the boundary, holds one link connection to the certifier's host. There the certifier
(`active.Proxy` with its salt on, signing every record and every reveal) forwards each request it releases over
loopback to the shim (`active_replay.LiveServer`, the canonical framer, which is the prover's side), with its place in
the release order and the certifier's run id (`x-verity-release`, `x-verity-release-run`: `release`), and the shim
sends it to the endpoint over HTTP or HTTPS with those two headers, so that `serve-openai --admit release-order` takes
requests in the order the certifier released them, and frames the stream it gets back as canonical events:

    client ──link──▶ certifier host: [ Proxy ──loopback──▶ shim ] ──HTTP(S)──▶ endpoint

**What it can't enforce off a namespace lab.** On mockdc's lab the proxy is the block's only route out, since the block's
namespace has no other interface. On real hosts it is only the route of the measured link:
- the endpoint's host can reach the client, or anyone, by any other route, and nothing here sees that traffic; the
  count-only verdict covers the link the client uses, not the endpoint's host;
- the shim's hop to the endpoint (certifier host → endpoint) is inside the boundary and unmeasured: a shim or an
  endpoint that sends a token elsewhere, or takes requests from elsewhere, is not seen;
- the salt reaches the shim on its own connection and is only as private as that hop and the endpoint's host;
- the client and the certifier agree on the origin through the wall clock (`origin_unix`), so a client on another
  host needs its clock within a few tens of ms of the certifier host's (requests are sent mid-bucket);
- the rehearsal (`rehearse`) runs the client on the certifier's host, so its link is loopback, and its check plays the
  ZK server and the challenger in one process, with the network registration of each link-window standing in for
  PoUW's registration of the window, which opens the window's salt;
- the reveal deadline covers the whole run and a margin (`REVEAL_MARGIN_S`), where a deployment's covers one PoUW
  window; and the keys are LMS at H = `HEIGHT` (2^10 one-time keys) unless `--height` says otherwise, where the
  deployed key's is H = 20 (the auditor gate's default, `benchmarks/live_challenger/auditor.py`).

**The served plan** (`plan`, `served_plan`). The endpoint is probed with the one request body every session sends,
straight from the host, `--in-flight` requests at a time after one uncounted round: `/v1/completions`, streamed, a
prompt of exactly `--prompt-tokens` tokens (`PROMPT` padded with `FILLER` in the endpoint's own tokenizer, `/tokenize`,
and cut there), exactly `--tokens` tokens out (`max_tokens` and `ignore_eos`), at `--temperature` (0)
and `--top-p` (1), both always sent, since an endpoint applies its own generation config to a request that omits
them. F, the
frames of a response, is that token count, not the endpoint's events: vLLM v1 merges a request's tokens that its API
server hasn't read yet into one chunk, so with several in flight a response's last two tokens sometimes share one
event and sometimes don't, though the tokens are the same. vLLM's detokenizer holds back a token whose bytes end
inside a UTF-8 character and streams it as a chunk of no text, its bytes in the next chunk, so it is still one event.
An event is a chunk of at least one token, so a response whose tokens are each at most `active.TOKEN_BYTES` bytes is
at most F canonical events, and the shim pads one that ends short with empty canonical events (`active.event(b"")`,
what a chunk of no text gives) up to F, after its last (`pad_to`, `active_replay.LiveServer`). A longer token is one
event per `TOKEN_BYTES` of it (GLM-5.3's tokenizer has 124, runs of spaces up to 512 bytes among them), so a response
that holds one is longer than its tokens: `--response-frames` declares F, at least the body's tokens. Each probed
response's raw count, which must be 1 … F, and its seconds to the last are kept; a response of more than F in the run
isn't cut, and N2's counts refuse its window. D, the latency budget in buckets, is
four times the slowest response plus half a second, as `lab_run.py` sets it, or, when a group's responses that late
don't fit in the window they're sent in (below), the most that does, down to twice the slowest plus half a second
(`DoesntFit` below that); `--delta-buckets` sets it. Window 0's first group is released in its bucket 2, a bucket
later than any other window's, and a D whose group fits only from bucket 1 doesn't fit: it would leave window 0
without a request, and `late` without a window-0 session to hold. The plan sends `--requests` requests `--in-flight`
at a time, the groups D + 1 buckets apart in the middle of a bucket, each request padded to the same number of link
frames. The endpoint serves a group side by side, and the shim writes the group's responses one after another in the
order their requests reached it (`request_major`), so the canonical order is request-major and the egress counts don't
depend on how the endpoint interleaves them. The plan is calibrated (`calibration.calibrate`) on the model of that
run: every frame of a response ready D buckets after its request's release. Every window declares the calibration's
envelope, refused when its δ is below D. Every window ends idle: a group whose responses wouldn't all be released by
the window's last bucket goes to the next window, so each window's egress counts follow from its own requests. On the
planned profile no request is released in a window's first bucket (the salt frame's), and such a group goes to the
next window's second. The run lasts one idle window past the last request's.

**N2's profile** (`plan --profile`). N2 (`service.LinkWindows`) gives no verdict on a link whose egress end has neither
an occupancy profile fixed before the run nor a served profile. The ingress end has no occupancy profile (the Program
reads requests from outside).
- `planned` (the default): the plan's `occupancy` is the Program's egress counts (`schedule.expected_grids`) under the
  declared syncs, on the model's releases and F frames per response, which the rehearsal's own client and the shim's
  padding fix before the run, and N2 holds each egress head's counts to them.
- `served`: no egress profile is planned. The plan's `occupancy` is `{"kind": "served", "response_frames": F,
  "request_frames": F_in, "own": 1}`, F_in the request's frames rounded up to a power of two, and N2 is
  `LinkWindows(served=ServedProfile(F, F_in, windows, 1))` (`served_profile`): each window's egress counts are the ones
  its ingress head's derive under the window's declared sync (`served.NetworkServedCounts`), with the salt frame at the
  head of bucket 0 taken off (own 1). The proxy (`active.Proxy(served=…)`) admits each request by the served admission
  rule (`served.Admission`), only while its window, with it, still ends idle, and holds the rest for the next window,
  so a window whose counts follow from its own requests is the proxy's doing, not the client's. A group that doesn't
  fit goes to the next window's first bucket, which admission takes behind the salt frame. The plan's `model_release`
  is the admission's (`active_replay.model_releases(admission=…)`), and a D at which admission would hold one of the
  plan's requests past the window it was sent in doesn't fit. The check reads each window's served counts in the clear
  too (`served_counts`, the rehearsal's read of the witness, beside the audit's); N2's verdict shows only accept or
  reject.

**The varied plan** (`plan --traffic varied`, `varied_plan`) drops every premise of the served plan's on the traffic:
no fixed response length, no fixed request size, no request-major order, no window that must end idle, no min_tokens
or ignore_eos. `--requests` requests arrive as a Poisson process at `--rate` a second (`varied_requests`), each sent in
the middle of the bucket before the one its arrival falls in. Each prompt is a list of the endpoint's token ids, a chat
turn (a question that the model answers in a word or at length, and filler) cut to a length log-uniform in
`--prompt-range`, with `max_tokens` uniform in 1 … `--tokens`, a seed of its own, temperature 0 and the model's own
EOS; with `--prompts-file` (a JSON array of prompts of token ids, the gate's `certifier-rehearse --prompts`) request
j's prompt is the (j mod n)-th of the file's n instead, at the same seed's arrivals, `max_tokens` and seeds. Its
session is the body's `user`, the request's id, `--name`.j (j its index, five digits), which the shim forwards the
endpoint, so a run named apart from every other one the endpoint served names each of its requests there as the
certified I/O does. Each is an id request (`active.id_request`): its sampling
fields at fixed places, each prompt id in a field of its own, in `active.id_request_frames(prompt length)` link frames,
a function of its length alone. The proxy opens each request's connection to the shim with the request's ingress
ordinal (`Proxy(ordinals=True)`), and the shim writes the responses interleaved, in the order the endpoint streams
them, as id events (`active.id_event`: the ordinal and one sampled token's id, the EOS token included, and no text):
it adds `UPSTREAM_FIELDS` to each body so that every chunk carries its token ids, as the committing server's do. The
first `--probes` requests are probed straight to the endpoint at their arrival times
(`probe_varied`). From them the plan models every frame of every request as ready `delta0_buckets` + ⌊k/ρ0⌋ buckets
after its release, with ρ0 = (cap − 1)/m at most the slowest probed response's rate over `--margin` (2) and every probed
token's latency, times `margin` and plus `extra_s` (0.5 s), covered. Its calibration is that model's at cap frames a
request, so the envelope it declares in every window meets every modelled frame. The run lasts until every modelled
frame is handed before its last window. N2's profile is `advice`: no egress profile, N2's advised counts derive the
counts from the ingress head and the advice, and this check gives no N2 verdict.

**The certified I/O** (`certified_io`, the check on the advice profile), read from the records' frames alone: each
request's ingress ordinal (its stream on the link), release bucket and id request's fields (`user`, prompt ids,
`max_tokens`, the sampling fields) from the ingress frames, and its response's token ids from the egress id events
that name its ordinal, with each frame's canonical position and egress bucket. Accepted when every request is an id
request whose `user` (the request's id) no other request has, every egress frame is an id event of a request released
by its bucket, and every response keeps the finish rule: at most `max_tokens` tokens, ending at its first EOS token
(the plan's `eos_token_ids`) or at `max_tokens`. `extra`'s token after a response's last breaks it. The check writes
the I/O to `io.json` for the I/O tie and N2's advised counts, and compares it with the ids the shim wrote and the
requests the plan sent.
On the advice profile each link-window is registered by its one-stage registration record (its salted root, chained)
with the check's own receipt, which no proof stream issues since N2 gives no outcome, and its certified records are
held to the certifier's key and session (`_certified`). The verdict is `accepted` when every window's records are
certified, the I/O, the audit and the salt accept.

**Subcommands**, all from the repository root or with `benchmarks/network_traces` on the path:

    python certifier_serve.py keys --dir KEYDIR [--height 10]
    python certifier_serve.py plan --upstream URL --out DIR [--model NAME] [--requests 400] [--in-flight 4]
        [--temperature 0] [--top-p 1]
        [--tokens 64] [--response-frames F] [--prompt-tokens 128] [--probes 8] [--delta-buckets D]
        [--profile planned|served] [--traffic fixed|varied] [--rate 2] [--prompt-range 16:1024 | --prompts-file FILE]
        [--seed 0] [--eos-ids ID,...] [--compact B] [--margin 2] [--name varied]
    python certifier_serve.py serve --plan PLAN --calibration REPORT --upstream URL --out DIR --link HOST:PORT
        --reveal HOST:PORT --key KEYDIR --challenger-key PUBLIC.json --session HEX [--start-in 10]
        [--tamper late|extra|drop|token|byte] [--out-shape SHAPE.json] [--paced-ms 50]
    python certifier_serve.py client --plan PLAN --calibration REPORT --deployment DEPLOYMENT.json --out DIR
    python certifier_serve.py check --run DIR --challenger KEYDIR [--expect accepted|refused]
    python certifier_serve.py rehearse (--upstream URL | --vllm MODEL) --out DIR [plan's options] [--strategies S,...]
        [--out-shape SHAPE.json] [--paced-ms 50]

- `keys` makes an LMS key (`seed`, 0600, `public`, its tree, and its leaf state `state.json`), or loads one, and writes
  `public.json`, the public key a deployment record carries. The certifier's key and the challenger's are each one.
- `serve` starts the shim, then the certifier's key in a process of its own (`CertifierProcess`, which keeps and
  certifies every record and signs every reveal off the proxy's GIL), the proxy, the link's listener and the reveal
  endpoint (`active.serve_reveals`, which also serves each window's certificates' public part once both its records
  are certified: `CertifyingPublisher.public`), and writes `deployment.json`, the auditor's record of the run: the
  certifier's public key, the proxy's run id, the certifier session (`--session`, which the challenger samples), the
  links, the addresses, the origin and the reveal deadline. It takes one link connection, which must come before the
  origin. The proxy writes each egress bucket through its pacer, `--paced-ms` (`PACED_MS`) after the bucket's boundary
  (`active.Proxy(paced=…)`), so that nothing on the proxy's loop moves when a bucket leaves; `--paced-ms 0` writes it
  on the loop at its tick. Once the link has ended it stops the shim, writes the logs and prints `{"served": ...}`, then serves reveals until every
  window's salt is revealed or the last window's reveal deadline has passed. `--tamper` makes the shim an adversary of
  `active_replay.adversary_plan`: `late` holds one session of window 0 past every sync (on a varied plan, past every
  frame written before it is due), which the certifier's own audit faults; `extra` writes one event more at the end of
  the first session's response, after its padding (F + 1 in all), on time and logged, so the records are complete and
  only N2's counts against the profile refuse it (on a varied plan, an id event of token 0, which the certified I/O's
  finish rule refuses). The key's directory may not be inside `--out`. `--out-shape SHAPE.json`: the output value's
  shape (`out_commit.OutShape`'s fields: the reading program's digest, the value, its words a row and rows, which the
  deployment takes from the circuit's registered read, `interface.read_shape`, never a flag of its own). At each egress
  link-window's close the network certifier commits the output it relayed, its canonical events' text read as the hex
  of u16 words (`out_words`), cut into the shape's rows (`out_rows`, exactly rows × words, no padding), and signs its
  head with its own key in its session (`certify_out`, `out_commit.commit_out` under the `CertifierProcess`'s running
  `Issuer`): the record in `out-records.jsonl`, what the auditor's `out_commit.check_out` reads, and the rows with
  their salts in `out-witness.jsonl`, the prover's. A window whose output isn't the shape's is a refusal line
  (``out-shape``), committed under no root. `deployment.json` carries the shape as ``out``.
- `client` sends the plan's requests on the link as `active_replay.Peer` does, at the deployment's origin, and reads
  the wire to its end (`peer.json`). Then it asks the reveal endpoint for each window's certificates and keeps those
  that verify under the deployment's certifier key and session (`keep_certificates`: `certificates.jsonl`, and the
  clock certificates in `clocks.jsonl`, public parts only); it refuses each other one by name, a clock certificate
  whose signed digest an earlier window's signs among them (`check_certificates`), and exits 1.
- `check` is the auditor's offline check of a served run (`check`): per link-window, the certifier's audit of the raw
  and committed records, N2's in-the-clear verdict through `LinkWindows` (commit, register, select, deliver,
  `unproved_outcome`: the check proves nothing, so the verdict reads the opening; `n2_zk.py` proves it) on the certified
  records, each window's salt asked of the certifier through a `RemoteCertifier` signed with the
  challenger's key (window 0's asked twice: the second is refused), then `network_verdict` and the salt audit
  (`active_replay.audit_salt`: the salt frames, and that the shim received each revealed salt). It must run while
  `serve` serves reveals.
- `rehearse` makes both keys in a scratch directory outside `--out`, plans, and for each of `--strategies` (honest, late,
  extra by default, or any of `--tamper`'s) runs `serve` (with `--out-shape` when given) and `client` as processes
  and `check`; `summary.json` holds
  each run's verdicts, and it exits 0 when `check` accepts honest, token and byte and refuses the rest. With
  `--vllm MODEL` it starts vLLM on this host's GPU first (`active_replay.start_vllm`). Besides late and extra,
  `--tamper drop` leaves out the first event of the first session's response (F − 1), on time and logged, so only N2's
  counts refuse it, and
  `--tamper token` writes every token of window 0's responses carrying other bytes (`active_replay.other_token`), so
  its frame counts are honest's and `check` accepts it: only the I/O tie, which `check` doesn't run, refuses certified
  tokens that aren't the committed sampled ones. `--tamper byte` changes one hex digit of the first session's response
  (`active_replay.other_hex`): one output byte of a response whose text is its outputs' hex, which `check` accepts and
  only the output tie refuses.

**Outputs** in a run's directory: `plan.json`, `calibration.json`, `deployment.json`, the proxy's records
(`requests.records`, `responses.records`, `committed.jsonl`) and `certified.jsonl` (each record's head, the
certifier's salt for it and certificate, or the link-window the signer refused), `clocks.jsonl` (the certifier's
salted clock certificate of each egress window's sync with its opening, the sync and its salt, which an advised
link's verifier of record reads: `read_clocks`), `proxy.json.gz`, the shim's `server.json` and `server.events.gz`, the
client's `peer.json`, `certificates.jsonl` and `clocks.jsonl` (in `client/` in a rehearsal), `serve.json` and
`check.json`, and with `--out-shape` `out-records.jsonl` (public) and `out-witness.jsonl` (the out value's rows and
salts, witness). Records, their heads, the syncs the certifier signed, the certifier's salts and the shim's events are
witness, kept behind the ZK server in a deployment; a rehearsal publishes them so that its check can be replayed. A
certificate's public part is public, which the client keeps: its key id, link, direction and window, the digest it
signs and its LMS signature, in the deployment's session. The digest is the head's, or the sync's clock digest,
salted by its own `salt_c` (`certificate_digest`, `clock_commitment`), a hiding commitment, so the signature over it
shows nothing of the head or the sync. No PoUW salt is written, only its SHA-512
(`active_replay.fingerprint`). With `CERTIFIER_SERVE_TIMING` set, `serve` also writes `publisher_timing.json`: when
each record was handed over, kept and certified, on the proxy's clock, and each step's CPU seconds (`tick_load.py`
reads it).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import multiprocessing
import os
import random
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Collection, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import active_replay as ar
from verity.ml.crypto import lms
from verity.core.protocols.certifier import audit as host_audit, calibration, out_commit
from verity.core.protocols.certifier.active import COMPACT_RECORD, DATA, HEADER, ID_EVENT_BYTES, REVEAL_LINE_BYTES, \
    Events, Proxy, RemoteCertifier, Requests, compact_record_data, decode_frame, event_body, event_data, fresh_salt, \
    id_event_data, id_request, id_request_fields, request_frame_count, serve_reveals
from verity.core.protocols.certifier.certificate import Certificate, CertifiedClock, CertifiedRecord, Direction, \
    Issuer, LmsKey, LmsSigner, RecordHead, SignerRefused, parameters_digest
from verity.core.protocols.certifier.grid import COMPLETE, Reason, Status, occupancy, row_count
from verity.core.protocols.certifier.lean_audit import Audit
from verity.core.protocols.certifier.schedule import ClockSync, expected_grids, handing_buckets
from verity.core.protocols.certifier.served import Admission, ServedProfile
from verity.core.protocols.certifier.service import LinkWindows
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.core.protocols.sampled_proofs.service import Registration

HERE = Path(__file__).resolve().parent
BUCKET_MS, WINDOW_BUCKETS, FRAME_BYTES, QUEUE = 100, 60, 4096, 1 << 16
EGRESS, INGRESS, N2_LINK = "responses", "requests", "served"
#: the rehearsal's client writes here in its run, as on a host of its own (`check` reads its `peer.json` there)
CLIENT = "client"
HEIGHT = 10
#: the processes `load_key` makes a key above `HEIGHT` in (at H = 20, about two minutes on eight cores); at or below
#: it a key takes about a second in one, and a caller on a thread forks nothing
KEYGEN_WORKERS = 8
PROMPT = "Count from one to one hundred in words, separated by commas:"
FILLER = " one, two, three, four, five, six, seven, eight, nine, ten,"
REVEAL_MARGIN_S = 300.0
#: when the proxy's pacer writes each egress bucket, in ms after its boundary: twice the proxy's slack, so that a tick
#: in time has handed it before
PACED_MS = 50.0
PROOF_DEADLINE_S = 600.0
DELTA = 2.0 ** -40
STRATEGIES = ("honest", "late", "extra")
TAMPERS = ("late", *ar.LIVE_TAMPERS)
# `check`'s verdict on each run it expects to accept: honest, and token and byte, whose frames only the I/O tie reads
ACCEPTED = ("honest", "token", "byte")
PROFILES = ("planned", "served")
TRAFFIC = ("fixed", "varied")
#: a varied plan's N2 profile: the egress counts derive from the ingress head and the advice (N2's advised counts)
ADVICE = "advice"
#: a varied plan's first arrival is after this, in window 0's bucket 2 or later
START_MS = 150.0
#: a varied prompt is one chat turn in Llama 3's template (the endpoint's tokenizer adds the BOS): a question, which
#: the model answers in a word or at length, then filler it is told to ignore
CHAT_HEAD = "<|start_header_id|>user<|end_header_id|>\n\n"
CHAT_TAIL = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
QUESTIONS = (
    "What is the capital of France? Answer with one word, then stop. Ignore the notes after this line.\n\n",
    "How many legs does a spider have? Answer with a number only. Ignore the notes after this line.\n\n",
    "Say yes or no: is water wet? Ignore the notes after this line.\n\n",
    "Write a long story about a lighthouse keeper and a storm. Ignore the notes after this line.\n\n",
    "List the planets of the solar system with one sentence on each. Ignore the notes after this line.\n\n",
    "Explain in detail how a bicycle's gears work. Ignore the notes after this line.\n\n",
)
VARIED_FILLER = ("Notes: the meeting moved to Thursday, the printer on the third floor is out of toner, the parking lot "
                 "will be repainted next week, and the cafeteria serves soup on Mondays and Fridays. ")
#: what the shim and the probe add to each body they send the endpoint, so that each chunk carries its token ids
UPSTREAM_FIELDS = {"return_token_ids": True}
CHALLENGER = "certifier-serve-check"
#: `--out-shape`'s files in a run: each egress link-window's out record (public, what the auditor receives) and its
#: witness, the rows and their salts (the prover's alone)
OUT_RECORDS, OUT_WITNESS = "out-records.jsonl", "out-witness.jsonl"
NO_N2 = ("no verdict on the advice profile: N2 derives the egress counts from the ingress head and the advice "
         "(its advised counts), which this check doesn't read")


# Keys.

def load_key(directory: Path, height: int = HEIGHT) -> lms.PrivateKey:
    """The LMS key (N = 64, W = 1) in `directory`, made there first at `height` when absent (above `HEIGHT` in
    `KEYGEN_WORKERS` processes), in the challengers' layout
    (`benchmarks/live_challenger/serve.py`'s `load_key`): `seed` (secret, 0600), `public` (`I ‖ T[1]`), `tree.bin`
    (its tree, checked against `public` and `tree.sha512`), its leaf state `state.json`, and `public.json`, the public
    key as a deployment record carries it."""
    import hashlib

    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    seed_p, public_p, tree_p, check_p, state_p = (directory / n for n in ("seed", "public", "tree.bin", "tree.sha512",
                                                                          "state.json"))
    if (directory / "public.json").exists():
        height = json.loads((directory / "public.json").read_text())["params"][2]
    params = lms.Params(64, 1, height)
    if not seed_p.exists():
        fd = os.open(seed_p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, secrets.token_bytes(params.N))
            os.fsync(fd)
        finally:
            os.close(fd)
    seed = seed_p.read_bytes()
    if tree_p.exists() and public_p.exists() and check_p.exists():
        public, raw = public_p.read_bytes(), tree_p.read_bytes()
        tree = (b"",) + tuple(raw[params.N * r:params.N * (r + 1)] for r in range(len(raw) // params.N))
        key = lms.PrivateKey(params, public[:16], seed, tree)
        if (len(tree) != 2 ** (params.H + 1) or lms.public_key(key) != public
                or hashlib.sha512(public + raw).digest() != check_p.read_bytes()):
            raise SystemExit(f"{tree_p} is not the tree of {public_p} at {params}")
    else:
        key = lms.keygen(params, seed, workers=KEYGEN_WORKERS if params.H > HEIGHT else 1)
        raw, public = b"".join(key.tree[1:]), lms.public_key(key)
        for path, data in ((tree_p, raw), (public_p, public), (check_p, hashlib.sha512(public + raw).digest())):
            path.write_bytes(data)
    if not state_p.exists():
        lms.LeafState.create(state_p, public, params.H)
    (directory / "public.json").write_text(json.dumps(public_json(LmsKey(params, public))))
    return key


def signer(directory: Path, height: int = HEIGHT) -> LmsSigner:
    return LmsSigner(load_key(directory, height), directory / "state.json")


def public_json(key: LmsKey) -> dict:
    return {"params": [key.params.N, key.params.W, key.params.H], "public": key.public.hex(), "key_id": key.key_id.hex()}


def read_public(doc: dict) -> LmsKey:
    return LmsKey(lms.Params(*doc["params"]), bytes.fromhex(doc["public"]))


# Certified records on disk: one JSON line each, the head's and the certificate's fields and the certifier's salt for
# the head (`salt_c`), bytes in hex.

_BYTES = {"parameters", "root", "key_id", "digest", "signature"}


def _enc(value):
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Direction):
        return value.value
    if isinstance(value, Status):
        return None if value.reason is None else value.reason.value
    return list(value) if isinstance(value, tuple) else value


def _dec(field: str, value):
    if field in _BYTES:
        return bytes.fromhex(value)
    if field == "direction":
        return Direction(value)
    if field == "status":
        return COMPLETE if value is None else Status(Reason(value))
    return tuple(value) if field == "counts" else value


def certificate_json(c: Certificate) -> dict:
    return {f: _enc(getattr(c, f)) for f in Certificate.__slots__}


def certificate_from_json(doc: dict) -> Certificate:
    return Certificate(**{f: _dec(f, doc[f]) for f in Certificate.__slots__})


def certified_json(record: CertifiedRecord) -> dict:
    return {"salt_c": record.salt.hex(), "head": {f: _enc(getattr(record.head, f)) for f in RecordHead.__slots__},
            "certificate": certificate_json(record.certificate)}


def certified_from_json(doc: dict) -> CertifiedRecord:
    return CertifiedRecord(RecordHead(**{f: _dec(f, v) for f, v in doc["head"].items()}), bytes.fromhex(doc["salt_c"]),
                           certificate_from_json(doc["certificate"]))


def clock_json(clock: CertifiedClock) -> dict:
    """One line of the certifier's `clocks.jsonl`: the sync and its salt, witness, and the certificate."""
    return {"sync": [clock.sync.delta, str(clock.sync.rho)], "salt_c": clock.salt.hex(),
            "certificate": certificate_json(clock.certificate)}


def clock_from_json(doc: dict) -> CertifiedClock:
    if "salt_c" not in doc:
        raise ValueError("a clock certificate without its opening's salt, which opens no salted digest: an unsalted "
                         "one from before the salted form, or the client's public copy")
    delta, rho = doc["sync"]
    return CertifiedClock(ClockSync(delta, Fraction(rho)), bytes.fromhex(doc["salt_c"]),
                          certificate_from_json(doc["certificate"]))


def read_clocks(path: Path) -> list[CertifiedClock]:
    """The certifier's `clocks.jsonl`: its salted clock certificates with their openings, one for each egress window it
    signed the sync of, witness (a link's clock source of record, `LinkWindows(certified_clocks=…)`); none for a run
    without the file. Raises `ValueError` on a line without its salt: an unsalted clock certificate, or a line of the
    client's `clocks.jsonl`, which holds the certificates alone (`keep_certificates`)."""
    return [clock_from_json(json.loads(line)) for line in path.read_text().splitlines()] if path.exists() else []


def read_certified(path: Path) -> tuple[dict[tuple[Direction, int], CertifiedRecord], list]:
    """`certified.jsonl`: the certified records by (direction, window), and the link-windows the signer refused."""
    got, refused = {}, []
    for line in path.read_text().splitlines() if path.exists() else []:
        doc = json.loads(line)
        if "refused" in doc:
            refused.append(doc["refused"])
        else:
            record = certified_from_json(doc)
            got[(record.direction, record.window)] = record
    return got, refused


def out_words(record) -> list[int] | None:
    """The output an egress record carries (`--out-shape`): the bytes of its canonical events in record order
    (`active.event_data`), read as the hex of u16 little-endian words, as the torch loop's stub sends its outputs; None
    when it carries no event or their text isn't that."""
    text = b"".join(d for row in record.grid for slot in row if slot is not None
                    for kind, _, payload in (decode_frame(slot),) if kind == DATA
                    for d in (event_data(payload),) if d is not None)
    try:
        raw = bytes.fromhex(text.decode("ascii"))
    except ValueError:
        return None
    if not raw or len(raw) % 2:
        return None
    return [int.from_bytes(raw[i:i + 2], "little") for i in range(0, len(raw), 2)]


def out_rows(words: Sequence[int], shape: out_commit.OutShape) -> list[list[int]] | None:
    """`words` cut into `shape`'s rows, ``shape.words`` consecutive words a row in the record's order; None unless they
    are exactly ``shape.rows`` rows (no padding, nothing dropped)."""
    w = shape.words
    if len(words) != shape.rows * w:
        return None
    return [list(words[i:i + w]) for i in range(0, len(words), w)]


def certify_out(out: Path, record, issuer: Issuer, shape: out_commit.OutShape) -> out_commit.CertifiedOut | None:
    """The network certifier's output value at an egress link-window's close (`--out-shape`): what the record carries
    (`out_words`) cut into `shape`'s rows (`out_rows`) and committed as the reading program's registered value at that
    shape, its head signed by the running `issuer` in its session for the link-window (`out_commit.commit_out`). Its
    public record is one line of `OUT_RECORDS`; the link-window under ``refused`` when the signer refuses, or under
    ``out-shape`` (with its ``detail``) when the output isn't the shape's rows, neither committed; its witness, the
    rows and their salts, one line of `OUT_WITNESS`, which goes to the prover alone. A record that carries no output
    value commits none."""
    words = out_words(record)
    if words is None:
        return None
    rows, at = out_rows(words, shape), [out_commit.LINK, out_commit.DIRECTION.value, record.window]
    if rows is None:
        got, line = None, {"out-shape": at, "detail": f"{len(words)} words, not {shape.rows} rows of {shape.words}"}
    else:
        try:
            got = out_commit.commit_out(rows, shape=shape, key=issuer, session=issuer.session, window=record.window)
        except SignerRefused:
            got, line = None, {"refused": at}
        else:
            line = got.public
            with open(out / OUT_WITNESS, "a") as fh:
                fh.write(json.dumps({"window": record.window, "program": shape.program, "rows": rows,
                                     "salts": [s.hex() for s in got.salts]}) + "\n")
    with open(out / OUT_RECORDS, "a") as fh:
        fh.write(json.dumps(line, sort_keys=True) + "\n")
    return got


class CertifyingPublisher(ar.Publisher):
    """The served proxy's `publish`: `Publisher`'s files, and each record certified by `issuer` as the network certifier
    hands it over, under the N2 link's name `link`, the one name both ends of a link-window share
    (`service.LinkWindows.link`): one line of `certified.jsonl` each, or the link-window under `refused` when the
    signer refuses. An egress record handed with the sync the shaper ran its window under (`ServedProxy`) gets that
    sync's salted clock certificate too (`Issuer.salted_clock`), one line of `clocks.jsonl` with its opening, none when
    the signer refuses. With `out_shape` (`out_commit.OutShape`'s doc), each egress record's output value is committed
    at that shape and signed by the same issuer (`certify_out`). `public` gives a window's certificates without the
    heads, syncs and salts, for the reveal endpoint to serve."""

    def __init__(self, out: Path, setup: ar.Setup, issuer: Issuer, link: str, threaded: bool = True,
                 out_shape: dict | None = None) -> None:
        super().__init__(out, setup, threaded)
        self.issuer, self.link = issuer, link
        self.out_shape = None if out_shape is None else out_commit.OutShape.of(out_shape)
        self.ends = {setup.egress_link: (Direction.EGRESS, parameters_digest(setup.egress_params, setup.sync_set)),
                     setup.ingress_link: (Direction.INGRESS, parameters_digest(setup.ingress_params))}
        self.timing: list | None = [] if os.environ.get("CERTIFIER_SERVE_TIMING") else None
        self.issued: dict[int, list[tuple[Direction, CertifiedRecord | None, CertifiedClock | None]]] = {}

    def public(self, window: int) -> dict | None:
        """Window `window`'s certificates' public part once both its records are kept, else None: the session they are
        signed in, each certificate and each clock certificate (`certificate_json`), and the link-windows the signer
        refused (`unsigned`). Each signs a digest salted by its own `salt_c`, a hiding commitment, so it shows nothing
        of the head or the sync it signs; heads, syncs and salts stay in `certified.jsonl` and `clocks.jsonl`."""
        kept = list(self.issued.get(window, ()))
        if len(kept) < len(self.ends):
            return None
        return {"session": self.issuer.session.hex(),
                "certificates": [certificate_json(c.certificate) for _, c, _ in kept if c is not None],
                "clocks": [certificate_json(k.certificate) for _, _, k in kept if k is not None],
                "unsigned": [[self.link, d.value, window] for d, c, _ in kept if c is None]}

    def __call__(self, record, params, sync: ClockSync | None = None) -> None:
        if self.executor is None:
            self.keep(record, params, sync)
        else:
            self.futures.append(self.executor.submit(self.keep, record, params, sync))

    def keep(self, record, params, sync: ClockSync | None = None) -> None:
        start = time.monotonic(), time.thread_time()
        super().keep(record, params)
        kept = time.monotonic(), time.thread_time()
        direction, parameters = self.ends[record.link]
        certified = self.issuer.certify(replace(record, link=self.link), direction, params, parameters)
        line = ({"refused": [self.link, direction.value, record.window]} if certified is None
                else certified_json(certified))
        with open(self.out / "certified.jsonl", "a") as fh:
            fh.write(json.dumps(line) + "\n")
        clock = self.issuer.salted_clock(self.link, record.window, sync) \
            if sync is not None and direction is Direction.EGRESS else None
        if clock is not None:
            with open(self.out / "clocks.jsonl", "a") as fh:
                fh.write(json.dumps(clock_json(clock)) + "\n")
        self.issued.setdefault(record.window, []).append((direction, certified, clock))
        if self.out_shape is not None and direction is Direction.EGRESS:
            certify_out(self.out, record, self.issuer, self.out_shape)
        if self.timing is not None:
            end = time.monotonic(), time.thread_time()
            self.timing.append({"link": record.link, "window": record.window, "start": start[0], "kept": kept[0],
                                "end": end[0], "keep_cpu": kept[1] - start[1], "certify_cpu": end[1] - kept[1]})


_CERTIFIER: dict = {}


def _certifier_start(key: Path, out: Path, report: dict, session: bytes, link: str,
                     out_shape: dict | None = None) -> None:
    own = signer(key)
    _CERTIFIER.update(signer=own, publish=CertifyingPublisher(out, ar.Setup(report), Issuer(own, session), link,
                                                              threaded=False, out_shape=out_shape))


def _certifier_call(name: str, *args):
    own, publish = _CERTIFIER["signer"], _CERTIFIER["publish"]
    if name == "keep":
        return publish.keep(*args)
    if name == "refused":
        return publish.issuer.refused, publish.timing
    if name == "certificates":
        return publish.public(*args)
    if name == "close":
        return own.close()
    return getattr(own, name)(*args)


class CertifierProcess:
    """The served proxy's `publish` and its reveals' `signer` in one spawned process, which alone holds the certifier's
    key and its leaf state: each record kept and certified as `CertifyingPublisher(threaded=False)` keeps it, and each
    reveal signed, one call at a time in the order they are made. Certifying a window's records hashes for
    milliseconds, so a thread would hold the proxy's GIL at its window's end. `close` waits for every record (the
    signer stays up for the reveals), `refused` gives the issuer's refusals, `certificates` a window's public part
    (`CertifyingPublisher.public`, after every record handed before it) and `shutdown` closes the signer. With
    `out_shape`, each egress link-window's output value is committed at it and signed in that process too
    (`certify_out`)."""

    def __init__(self, key: Path, out: Path, report: dict, session: bytes, link: str,
                 out_shape: dict | None = None) -> None:
        self.pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=_certifier_start,
                                        initargs=(key, out, report, session, link, out_shape))
        self.key: LmsKey = self.pool.submit(_certifier_call, "public_key").result()
        self.futures: list = []
        self.submits: list | None = [] if os.environ.get("CERTIFIER_SERVE_TIMING") else None

    @property
    def key_id(self) -> bytes:
        return self.key.key_id

    def public_key(self) -> LmsKey:
        return self.key

    def start(self, session: bytes) -> None:
        self.pool.submit(_certifier_call, "start", session).result()

    def sign(self, message: bytes) -> bytes:
        return self.pool.submit(_certifier_call, "sign", message).result()

    def certificates(self, window: int) -> dict | None:
        return self.pool.submit(_certifier_call, "certificates", window).result()

    def __call__(self, record, params, sync: ClockSync | None = None) -> None:
        if self.submits is not None:
            self.submits.append((record.link, record.window, time.monotonic()))
        self.futures.append(self.pool.submit(_certifier_call, "keep", record, params, sync))

    def close(self) -> None:
        for f in self.futures:
            f.result()

    def refused(self) -> tuple[list, list | None]:
        """The issuer's refused link-windows, and the publisher's timing (`CERTIFIER_SERVE_TIMING`) with each record's
        submit time."""
        refused, timing = self.pool.submit(_certifier_call, "refused").result()
        if timing is not None and self.submits is not None:
            submits = {(link, window): t for link, window, t in self.submits}
            timing = [row | {"submit": submits[(row["link"], row["window"])]} for row in timing]
        return refused, timing

    def shutdown(self) -> None:
        try:
            self.pool.submit(_certifier_call, "close").result()
        finally:
            self.pool.shutdown()


# The served plan.

def request_body(model: str, tokens: int, prompt: str | list[int] = PROMPT, temperature: float = 0.0,
                 top_p: float = 1.0) -> dict:
    """The request every session of a served plan sends: exactly `tokens` tokens (`max_tokens`, never stopped early:
    `ignore_eos`), streamed, at `temperature` and `top_p` (greedy by default). Both are always sent: an endpoint
    applies its own generation config to a request that omits them. No `min_tokens`: it masks the EOS logit, which the
    served Program's sampling doesn't model, so the vLLM integration's advice refuses it."""
    return {"model": model, "prompt": prompt, "max_tokens": tokens, "stream": True,
            "ignore_eos": True, "temperature": temperature, "top_p": top_p}


def _call(url: str, path: str, body: dict | None = None, api_key_env: str | None = None) -> dict:
    headers = {"content-type": "application/json"}
    if api_key_env:
        headers["authorization"] = f"Bearer {os.environ[api_key_env]}"
    data = None if body is None else json.dumps(body).encode()
    with urllib.request.urlopen(urllib.request.Request(url.rstrip("/") + path, data, headers), timeout=60) as r:
        return json.loads(r.read())


def served_model(url: str, api_key_env: str | None = None) -> str:
    """The one model the endpoint serves (`/v1/models`)."""
    ids = [m["id"] for m in _call(url, "/v1/models", api_key_env=api_key_env)["data"]]
    if len(ids) != 1:
        raise SystemExit(f"{url} serves {ids}: name the model (--model)")
    return ids[0]


def prompt_tokens(url: str, model: str, n: int, api_key_env: str | None = None) -> list[int]:
    """A prompt of exactly `n` tokens of the endpoint's tokenizer (vLLM's `/tokenize`): `PROMPT` padded with
    `FILLER`, cut at `n`."""
    tokens = _call(url, "/tokenize", {"model": model, "prompt": PROMPT + FILLER * n}, api_key_env)["tokens"]
    if len(tokens) < n:
        raise SystemExit(f"{url}'s tokenizer makes {len(tokens)} tokens of the padded prompt, fewer than {n}")
    return tokens[:n]


def upstream_params(url: str, api_key_env: str | None = None) -> dict:
    """The shim's `LiveServer` parameters for the endpoint at `url` (`http(s)://HOST[:PORT][/PREFIX]`)."""
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise SystemExit(f"an endpoint is http(s)://HOST[:PORT][/PREFIX], not {url!r}")
    tls = u.scheme == "https"
    return {"upstream": [u.hostname, u.port or (443 if tls else 80)], "tls": tls, "prefix": u.path.rstrip("/"),
            "api_key_env": api_key_env}


def probe(url: str, body: dict, n: int = 3, api_key_env: str | None = None, clock=time.monotonic,
          in_flight: int = 1, warmup: int = 0) -> dict:
    """`n` requests of `body` straight to the endpoint at `url`, `in_flight` at a time, after `warmup` that aren't
    counted: each response's raw count of canonical events, as the shim cuts them (`active_replay.canonical_events`)
    before it pads (`events`), and its seconds to the first and to the last."""
    headers = {"content-type": "application/json"}
    if api_key_env:
        headers["authorization"] = f"Bearer {os.environ[api_key_env]}"

    def one(_) -> dict:
        request = urllib.request.Request(url.rstrip("/") + ar.LIVE_PATH.decode(), json.dumps(body).encode(), headers)
        t0, first, count, cutter = clock(), None, 0, Events()
        with urllib.request.urlopen(request, timeout=300) as response:
            while data := response.read1(1 << 16):
                k = sum(len(ar.canonical_events(e)) for e in cutter.feed(data))
                if k and first is None:
                    first = clock() - t0
                count += k
        return {"events": count, "first_s": first, "last_s": clock() - t0}

    with ThreadPoolExecutor(in_flight) as pool:
        warm = list(pool.map(one, range(warmup)))
        responses = list(pool.map(one, range(n)))
    return {"url": url, "body": body, "in_flight": in_flight, "warmup": warm, "responses": responses}


class DoesntFit(ValueError):
    """No served plan at that latency budget: a group's responses aren't all released in the window they're sent in."""


def served_plan(probed: dict, requests: int = 8, delta_buckets: int | None = None,
                name: str = "served", in_flight: int = 1, profile: str = "planned",
                frames: int | None = None, prompts: Sequence[list[int]] | None = None,
                window_per_group: bool = False) -> tuple[dict, dict]:
    """The calibration report and the plan of a served run of `requests` requests of the probed body, sent `in_flight`
    at a time (the module docstring's "The served plan"), with N2's `profile` (`occupancy`, "N2's profile"): the
    planned egress occupancy, or the served profile, at the latency budget `delta_buckets`, or else at the largest of 4
    down to 2 times the slowest response plus half a second that fits. F, a response's frames, is `frames`, by default
    the body's tokens. `prompts`: request j's prompt is ``prompts[j % len(prompts)]`` (each its own body, which its peer
    entry carries), the probed body's the first. `window_per_group`: each group starts a window of its own, so group g
    is window g's traffic alone."""
    if profile not in PROFILES:
        raise ValueError(f"N2's profile is one of {PROFILES}, not {profile!r}")
    body = probed["body"]
    tokens = body["max_tokens"]
    F = frames or tokens
    if body.get("ignore_eos") is not True or not isinstance(tokens, int) or tokens < 1 or "min_tokens" in body:
        raise ValueError("a served plan's body asks for exactly its tokens (max_tokens, ignore_eos) and no min_tokens, "
                         f"which the vLLM integration's advice refuses: not {body}")
    if F < tokens:
        raise ValueError(f"F, a response's frames, is at least the body's {tokens} tokens, not {F}")
    lengths = sorted({r["events"] for r in probed["responses"]})
    if not lengths or lengths[0] < 1 or lengths[-1] > F:
        raise ValueError(f"the endpoint's responses to one body of {tokens} tokens are {lengths} events long: each is "
                         f"1 … F = {F}, an event a chunk of at least one token, and a token of more than "
                         f"{ar.TOKEN_BYTES} bytes one event per {ar.TOKEN_BYTES}: declare F at least the longest "
                         "(--response-frames)")
    slowest = max(r["last_s"] for r in probed["responses"])
    if delta_buckets:
        return _served_plan(probed, requests, delta_buckets, name, in_flight, profile, F, prompts, window_per_group)
    most, least = (math.ceil((k * slowest * 1000 + 500) / BUCKET_MS) for k in (4, 2))
    for D in range(most, least - 1, -1):
        try:
            return _served_plan(probed, requests, D, name, in_flight, profile, F, prompts, window_per_group)
        except DoesntFit:
            pass
    raise DoesntFit(f"{in_flight} responses of {F} events don't fit in a window of {WINDOW_BUCKETS} buckets at a "
                    f"latency budget D of {least} … {most} buckets (2 to 4 × the slowest, {slowest:.3f} s, + 0.5 s): "
                    "give one (--delta-buckets), or send fewer in flight")


def _served_plan(probed: dict, requests: int, D: int, name: str, in_flight: int,
                 profile: str = "planned", F: int | None = None, prompts: Sequence[list[int]] | None = None,
                 window_per_group: bool = False) -> tuple[dict, dict]:
    body = probed["body"]
    F = F or body["max_tokens"]
    lengths = sorted({r["events"] for r in probed["responses"]})
    slowest = max(r["last_s"] for r in probed["responses"])
    names = [f"{name}.{j}" for j in range(requests)]
    bodies = [body | {"prompt": list(prompts[j % len(prompts)])} if prompts else body for j in range(requests)]
    groups = [names[j:j + in_flight] for j in range(0, requests, in_flight)]
    frames_in = max(request_frame_count(len(ar.live_request_bytes(s, b, None, FRAME_BYTES)), FRAME_BYTES)
                    for s, b in zip(names, bodies))
    served = profile == "served"
    if served:  # the served counts take requests of a power of two of frames
        frames_in = 1 << (frames_in - 1).bit_length()
    T = WINDOW_BUCKETS
    for span in range(D + 1, T - 1):
        starts, a = [], 2
        for g, _ in enumerate(groups):
            if window_per_group and g:
                a = max(a, (starts[-1] // T + 1) * T + (0 if served else 1))
            if a % T == 0 and not served:
                a += 1
            elif a % T + span > T - 1:
                a = (a // T + 1) * T + (0 if served else 1)
            starts.append(a)
            a += D + 1
        if starts[0] >= T:
            raise DoesntFit(f"{in_flight} responses of {F} events, D = {D} buckets after their requests, don't fit in "
                            f"window 0 from its bucket 2: give a smaller latency budget (--delta-buckets)")
        anchors = [a for g, a in zip(groups, starts) for _ in g]
        sends = [(a - 0.5) * BUCKET_MS for a in anchors]
        frames = [{"session": s, "index": k, "anchor": a, "ready": a + D} for s, a in zip(names, anchors)
                  for k in range(F)]
        report = calibration.calibrate({
            "bucket_ms": BUCKET_MS, "window_buckets": T, "frame_bytes": FRAME_BYTES, "queue_capacity": QUEUE,
            "configurations": [{"name": name, "egress": [{"link": EGRESS, "frames": frames}], "ingress": [
                {"link": INGRESS, "arrival_ms": [t for t in sends for _ in range(frames_in)]}]}]})
        setup = ar.Setup(report)
        e, i, envelope = setup.egress_params, setup.ingress_params, setup.envelope
        if envelope.delta < D:
            raise DoesntFit(f"the calibration's envelope δ = {envelope.delta} is below the latency budget D = {D}")
        sessions = [[s, j + 1, [D] * F] for j, s in enumerate(names)]
        peer = [{"stream": j + 1, "session": s, "frames": frames_in, "trace_ms": t,
                 "enqueue": math.ceil(t / i.bucket_ms), "send_ms": t, **({"body": bodies[j]} if prompts else {})}
                for j, (s, t) in enumerate(zip(names, sends))]
        windows = anchors[-1] // T + 2
        if served:
            occ = {"kind": "served", "egress": None, "ingress": None, "response_frames": F, "request_frames": frames_in,
                   "own": 1}
            admission = Admission(served_profile({"windows": windows, "occupancy": occ}), e.allocation,
                                  lambda w: envelope)
            release = ar.model_releases(peer, i, windows, e.bucket_ms, admission)
            model = ar.model_frames(sessions, release, windows * T)
            if all(q["stream"] in release and release[q["stream"]] // T == q["enqueue"] // T for q in peer):
                break
        else:
            release = ar.model_releases(peer, i, windows, e.bucket_ms)
            model = ar.model_frames(sessions, release, windows * T)
            grids = [list(occupancy(g)) for g in expected_grids(model[0], [envelope] * windows, e)]
            if all(sum(map(sum, grids[:w + 1])) == F * sum(a // T <= w for a in anchors) for w in range(windows)):
                occ = {"kind": "planned", "egress": grids, "ingress": None}
                break
    else:
        raise DoesntFit(f"{in_flight} responses of {F} events, D = {D} buckets after their requests, don't fit in a "
                        f"window of {T} buckets: give a smaller latency budget (--delta-buckets)")
    chosen, _, misses = calibration.choose_syncs([model], setup.sync_set, T)
    eager = ClockSync(min(setup.sync_set.deltas), max(setup.sync_set.rhos))
    chosen = (list(chosen) + [eager] * windows)[:windows]
    plan = {"name": name, "windows": windows, "sessions": sessions, "requests": peer,
            "model_release": {str(k): v for k, v in release.items()}, "model_misses": misses,
            "syncs": [[envelope.delta, str(envelope.rho)]] * windows, "sync_policy": "envelope",
            "model_syncs": [[s.delta, str(s.rho)] for s in chosen], "live_model": body["model"], "body": body,
            "seed": 0, "occupancy": occ, "request_major": True,
            "pad_to": F, **({"prompts": [list(p) for p in prompts]} if prompts else {}),
            **({"window_per_group": True} if window_per_group else {}),
            "served": {"tokens": body["max_tokens"], "response_frames": F, "probe_events": lengths,
                       "delta_buckets": D, "gap_buckets": D + 1, "span_buckets": span, "slowest_s": slowest,
                       "request_frames": frames_in, "in_flight": in_flight, "requests": requests}}
    return report, plan


def served_profile(plan: dict) -> ServedProfile | None:
    """N2's served profile of a plan on the served profile (`occupancy` of kind `served`): F frames a response, F_in a
    request, the plan's windows and the salt frame heading each (own 1); None on the planned profile."""
    o = plan.get("occupancy") or {}
    if o.get("kind") != "served":
        return None
    return ServedProfile(o["response_frames"], o["request_frames"], plan["windows"], o["own"])


# The varied plan.

def tokenize(url: str, model: str, text: str, api_key_env: str | None = None) -> list[int]:
    return _call(url, "/tokenize", {"model": model, "prompt": text}, api_key_env)["tokens"]


def varied_fields(model: str, q: dict) -> dict:
    """A varied request's fields (`active.id_request`'s, less its `user`): its prompt's ids, its `max_tokens` and
    `seed`, greedy (temperature 0, top_p 1), stopped by the model's EOS."""
    return {"model": model, "prompt": q["prompt"], "max_tokens": q["max_tokens"], "temperature": 0.0, "top_p": 1.0,
            "seed": q["seed"]}


def varied_body(model: str, q: dict) -> dict:
    """The body the shim forwards the endpoint for a varied request (its id request's, less its `user`), streamed,
    with no stop strings."""
    return varied_fields(model, q) | {"stop": [], "stream": True}


def varied_requests(url: str, model: str, n: int, *, rate: float = 2.0, prompt_tokens: tuple[int, int] = (16, 1024),
                    max_tokens: int = 64, seed: int = 0, name: str = "varied",
                    api_key_env: str | None = None, prompts: Sequence[list[int]] | None = None) -> list[dict]:
    """`n` requests of varied traffic: Poisson arrivals at `rate` a second from `START_MS`; each prompt a chat turn
    (`CHAT_HEAD`, one of `QUESTIONS`, `VARIED_FILLER` and `CHAT_TAIL`, in the endpoint's tokens, `/tokenize`) cut to a
    length log-uniform in `prompt_tokens`, or with `prompts` request j's the (j mod n)-th of them, at the same seed's
    arrivals, `max_tokens` and seeds; `max_tokens` uniform in 1 … `max_tokens`; a u32 `seed` of its own. Its
    session, the body's `user`, is `name.j` with j its index, five digits."""
    lo, hi = prompt_tokens
    if not 1 <= lo <= hi or max_tokens < 1 or rate <= 0:
        raise ValueError(f"prompts of 1 ≤ {lo} … {hi} tokens, max_tokens ≥ 1 and a positive rate, not {rate}")
    if prompts is not None and not (prompts and all(prompts)):
        raise ValueError("prompts: at least one, each of at least one token id")
    if prompts is None:
        bos = tokenize(url, model, "", api_key_env)
        tail = tokenize(url, model, CHAT_TAIL, api_key_env)[len(bos):]
        filler, k = [], 1
        while len(filler) < hi:
            filler, k = tokenize(url, model, VARIED_FILLER * k, api_key_env)[len(bos):], 2 * k
        heads = [tokenize(url, model, CHAT_HEAD + q, api_key_env) for q in QUESTIONS]
    rng, seeds, at, out = random.Random(seed), random.Random(f"{seed}/seeds"), START_MS, []
    for j in range(n):
        at += rng.expovariate(rate) * 1000
        length = round(lo * (hi / lo) ** rng.random())
        head = rng.randrange(len(QUESTIONS))
        prompt = list(prompts[j % len(prompts)]) if prompts is not None else \
            ((heads[head] + filler)[:max(0, length - len(tail))] + tail)[:length]
        out.append({"session": f"{name}.{j:05d}", "at_ms": at, "prompt": prompt,
                    "max_tokens": rng.randint(1, max_tokens), "seed": seeds.randrange(1 << 32)})
    return out


def probe_varied(url: str, model: str, requests: list[dict], api_key_env: str | None = None, clock=time.monotonic,
                 sleep=time.sleep) -> dict:
    """`requests` straight to the endpoint at `url`, each at its arrival time from the first's, with
    `UPSTREAM_FIELDS`: each response's tokens, the seconds from its send to each token (`token_s`), its finish reason
    and its last token's id."""
    headers = {"content-type": "application/json"}
    if api_key_env:
        headers["authorization"] = f"Bearer {os.environ[api_key_env]}"
    t0, base = clock(), min((q["at_ms"] for q in requests), default=0.0)

    def one(q: dict) -> dict:
        if (wait := t0 + (q["at_ms"] - base) / 1000 - clock()) > 0:
            sleep(wait)
        body = json.dumps(varied_body(model, q) | UPSTREAM_FIELDS).encode()
        request = urllib.request.Request(url.rstrip("/") + ar.LIVE_PATH.decode(), body, headers)
        sent, ids, token_s, finish, cutter = clock(), [], [], None, Events()
        with urllib.request.urlopen(request, timeout=300) as response:
            while data := response.read1(1 << 16):
                now = clock() - sent
                for e in cutter.feed(data):
                    if (got := ar.chunk_tokens(e)) is not None:
                        ids += got
                        token_s += [now] * len(got)
                        chunk = json.loads(e.partition(b"data:")[2])
                        finish = next((c["finish_reason"] for c in chunk.get("choices") or () if c.get("finish_reason")),
                                      finish)
        return {"session": q["session"], "max_tokens": q["max_tokens"], "prompt_tokens": len(q["prompt"]),
                "tokens": len(ids), "token_s": token_s, "finish": finish, "last": ids[-1] if ids else None}

    with ThreadPoolExecutor(max(1, len(requests))) as pool:
        responses = list(pool.map(one, requests))
    return {"url": url, "model": model, "upstream_fields": UPSTREAM_FIELDS, "responses": responses}


def varied_plan(probed: dict, requests: list[dict], *, name: str = "varied", seed: int = 0, margin: float = 2.0,
                extra_s: float = 0.5, eos: Sequence[int] | None = None,
                compact: int | None = None) -> tuple[dict, dict]:
    """The calibration report and the plan of a varied run of `requests` (the module docstring's "The varied plan");
    with `compact` = B, of compact egress records in B-byte cells (the report's `egress_frame_bytes`), ingress
    unchanged."""
    if compact is not None and compact < COMPACT_RECORD.size:
        raise ValueError(f"a compact cell holds a {COMPACT_RECORD.size}-byte record, not {compact} bytes")
    T, tau = WINDOW_BUCKETS, BUCKET_MS
    responses = probed["responses"]
    if not requests or any(r["tokens"] < 1 for r in responses):
        raise ValueError("a varied plan has requests, and every probed response at least one token")
    cap = max(q["max_tokens"] for q in requests)
    rates = [(r["tokens"] - 1) * tau / ((r["token_s"][-1] - r["token_s"][0]) * 1000) for r in responses
             if r["tokens"] > 1 and r["token_s"][-1] > r["token_s"][0]]
    m = max(1, math.ceil((cap - 1) * margin / min(rates, default=cap))) if cap > 1 else 1
    rho = Fraction(max(1, cap - 1), m)

    def lag(k: int) -> int:
        return (k * rho.denominator) // rho.numerator

    D0 = max([1] + [math.ceil((margin * s + extra_s) * 1000 / tau) - lag(k) for r in responses
                    for k, s in enumerate(r["token_s"])])
    names = [q["session"] for q in requests]
    anchors = [max(2, math.ceil(q["at_ms"] / tau)) for q in requests]
    sends = [(a - 0.5) * tau for a in anchors]
    model = probed["model"]
    frames_in = [request_frame_count(len(id_request(**varied_fields(model, q), user=q["session"],
                                                    frame_bytes=FRAME_BYTES)), FRAME_BYTES) for q in requests]
    full = [D0 + lag(k) for k in range(cap)]
    keys = sorted((a + d, a, j, k) for j, a in enumerate(anchors) for k, d in enumerate(full))
    report = calibration.calibrate({
        "bucket_ms": tau, "window_buckets": T, "frame_bytes": FRAME_BYTES, "queue_capacity": QUEUE,
        "configurations": [{"name": name, "egress": [{"link": EGRESS, "frames": [
            {"session": names[j], "index": k, "anchor": a, "ready": rd} for rd, a, j, k in keys]}],
            "ingress": [{"link": INGRESS, "arrival_ms": [t for t, n in zip(sends, frames_in) for _ in range(n)]}]}]})
    if compact is not None:
        report["grid"]["egress_frame_bytes"] = compact
        for link in report["egress"]:
            link |= {"frame_bytes": compact, "bandwidth_bps": link["allocation"] * compact * 8 / (tau / 1000)}
    setup = ar.Setup(report)
    e, i, envelope = setup.egress_params, setup.ingress_params, setup.envelope
    sessions = [[s, j + 1, full[:q["max_tokens"]]] for j, (s, q) in enumerate(zip(names, requests))]
    peer = [{"stream": j + 1, "session": s, "frames": n, "trace_ms": q["at_ms"], "enqueue": a, "send_ms": t,
             "id_request": varied_fields(model, q)}
            for j, (s, q, n, a, t) in enumerate(zip(names, requests, frames_in, anchors, sends))]
    frames_total, windows = sum(len(ds) for _, _, ds in sessions), anchors[-1] // T + 2
    while True:
        release = ar.model_releases(peer, i, windows, e.bucket_ms, salt=True)
        frames, ready = ar.model_frames(sessions, release, windows * T)
        hands = handing_buckets(frames, [envelope] * windows, T)
        if len(release) == len(peer) and len(frames) == frames_total and max(hands) < (windows - 1) * T:
            break
        windows += 1
    if any(r > h for r, h in zip(ready, hands)):
        raise DoesntFit(f"the calibration's envelope {envelope} doesn't cover the model's frames")
    chosen, _, misses = calibration.choose_syncs([(frames, ready)], setup.sync_set, T)
    eager = ClockSync(min(setup.sync_set.deltas), max(setup.sync_set.rhos))
    chosen = (list(chosen) + [eager] * windows)[:windows]
    stops = sorted({r["last"] for r in responses if r["finish"] == "stop"} if eos is None else set(eos))
    plan = {"name": name, "windows": windows, "sessions": sessions, "requests": peer,
            "model_release": {str(k): v for k, v in release.items()}, "model_misses": misses,
            "syncs": [[envelope.delta, str(envelope.rho)]] * windows, "sync_policy": "envelope",
            "model_syncs": [[s.delta, str(s.rho)] for s in chosen], "live_model": model, "seed": seed,
            "occupancy": {"kind": ADVICE, "egress": None, "ingress": None}, "request_major": False,
            "token_ids": True, "upstream_fields": probed["upstream_fields"], "compact": compact,
            "varied": {"requests": len(requests), "max_tokens": cap, "eos_token_ids": stops,
                       "prompt_tokens": [min(len(q["prompt"]) for q in requests),
                                         max(len(q["prompt"]) for q in requests)],
                       "delta0_buckets": D0, "rho0": str(rho), "margin": margin, "extra_s": extra_s,
                       "probes": len(responses), "probe_finish": dict(Counter(r["finish"] for r in responses)),
                       "frames_in": dict(Counter(frames_in))}}
    return report, plan


# The certifier.

class ServedProxy(ar.ReleaseProxy):
    """`active.Proxy` whose server gets each request with its place in the release order (`active_replay.ReleaseProxy`),
    keeping the windows whose salt it has revealed, so that `serve` stops once every one is, and handing its `publish`,
    with each egress record, the sync its own shaper ran the window under (`Shaper.syncs`), which the certifier signs
    (`CertifyingPublisher`, `CertifierProcess`)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.revealed: set[int] = set()
        if kwargs.get("publish") is not None:
            publish, shaper = self.publish, self.shaper

            def clocked(record, params) -> None:
                ran = shaper.syncs
                egress = record.link == shaper.certifier.link and record.window < len(ran)
                publish(record, params, ran[record.window] if egress else None)

            self.publish = clocked

    def reveal(self, window, registration, request=None):
        out = super().reveal(window, registration, request)
        self.revealed.add(window)
        return out


def deployment(proxy: Proxy, setup: ar.Setup, windows: int, *, session: bytes, n2_link: str, link: tuple[str, int],
               reveal: tuple[str, int], origin_unix: float, url: str) -> dict:
    """The auditor's record of one served run (`deployment.json`): what the challenger and the check read, the
    certifier's public key and the proxy's run id among it, and the challenger key's id the proxy registered."""
    e = setup.egress_params
    return {"certifier_key": public_json(proxy.signer.public_key()), "run_id": proxy.run_id.hex(),
            "session": session.hex(), "links": {"egress": setup.egress_link, "ingress": setup.ingress_link,
                                                "n2": n2_link},
            "link": list(link), "reveal": list(reveal), "origin_unix": origin_unix, "windows": windows,
            "window_s": e.buckets * e.bucket_ms / 1000, "reveal_deadline_s": proxy.reveal_deadline,
            "challenger_key_id": proxy.challenger_key.key_id.hex(), "upstream": url}


def _announce(line: dict) -> None:
    print(json.dumps(line), flush=True)


def _address(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    return host, int(port)


def _out_shape(path: str) -> dict:
    """``--out-shape``'s file read back as `out_commit.OutShape`'s doc, refused unless it is one."""
    try:
        return out_commit.OutShape.of(json.loads(Path(path).read_text())).doc()
    except (OSError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"{path}: not an out shape ({e})") from None


async def serve(plan: dict, report: dict, out: Path, *, url: str, upstream: dict, link: tuple[str, int],
                reveal: tuple[str, int], key: Path, challenger_key: LmsKey, session: bytes,
                start_in: float = 10.0, reveal_deadline: float | None = None, n2_link: str = N2_LINK,
                tamper: str | None = None, out_shape: dict | None = None, paced_ms: float = PACED_MS,
                announce=_announce) -> dict:
    """One served run on the host's clock (the module docstring's `serve`), the certifier's key (the directory `key`)
    held by its `CertifierProcess`, which with `out_shape` also commits each egress link-window's output value at that
    shape, the deployment's ``out``, and each egress bucket written `paced_ms` after its boundary by the proxy's pacer
    (on its loop, with 0). Returns what `serve.json` holds."""
    out.mkdir(parents=True, exist_ok=True)
    setup = ar.Setup(report)
    e, i, windows = setup.egress_params, setup.ingress_params, plan["windows"]
    window_s = e.buckets * e.bucket_ms / 1000
    deadline = reveal_deadline or (windows + 1) * window_s + REVEAL_MARGIN_S
    (out / "plan.json").write_text(json.dumps(plan))
    (out / "calibration.json").write_text(json.dumps(report))
    shim_plan = plan if tamper is None else ar.adversary_plan(plan, setup, tamper)
    ctx = multiprocessing.get_context("spawn")
    conn, child = ctx.Pipe()
    shim = ctx.Process(target=ar.server_process, args=(shim_plan, ar.wire_params(setup) | upstream, child, str(out)))
    shim.start()
    loop = asyncio.get_running_loop()
    shim_port = await loop.run_in_executor(None, conn.recv)
    certifier = CertifierProcess(key, out, report, session, n2_link, out_shape)
    origin = time.monotonic() + start_in
    proxy = ServedProxy(("127.0.0.1", shim_port), e, i, setup.sync_set,
                        [ClockSync(d, Fraction(r)) for d, r in plan["syncs"]], origin=origin, publish=certifier,
                        egress_link=setup.egress_link, ingress_link=setup.ingress_link, salt=fresh_salt,
                        reveal_deadline=deadline, challenger_key=challenger_key, signer=certifier,
                        served=served_profile(plan), ordinals=bool(plan.get("token_ids")),
                        compact=bool(plan.get("compact")), paced=paced_ms / 1000 if paced_ms else None)
    taken, served = loop.create_future(), loop.create_future()

    async def on_link(reader, writer) -> None:
        if taken.done():  # one link: a second connection is never the client's
            writer.close()
            return
        taken.set_result(None)
        try:
            await proxy.run(reader, writer)
        finally:
            served.set_result(None)

    listening = await asyncio.start_server(on_link, *link)
    reveals = await serve_reveals(proxy, *reveal, certificates=certifier.certificates)
    conn.send(origin)
    record = deployment(proxy, setup, windows, session=session, n2_link=n2_link,
                        link=(link[0], listening.sockets[0].getsockname()[1]),
                        reveal=(reveal[0], reveals.sockets[0].getsockname()[1]),
                        origin_unix=time.time() + origin - time.monotonic(), url=url)
    if out_shape is not None:
        record["out"] = out_shape
    (out / "deployment.json").write_text(json.dumps(record, indent=1))
    announce({"listening": record["link"], "reveal": record["reveal"], "deployment": str(out / "deployment.json")})
    log = {"tamper": tamper, "windows": windows}
    try:
        await asyncio.wait_for(asyncio.shield(taken), timeout=max(0.0, origin - time.monotonic()))
    except TimeoutError:
        log["error"] = "no link connection before the origin"
    else:
        await served
    listening.close()
    certifier.close()
    conn.send("stop")
    await loop.run_in_executor(None, shim.join)
    ar.write_logs(out, [proxy])
    announce({"served": "error" not in log, "out": str(out)})
    end = origin + windows * window_s + deadline
    while "error" not in log and len(proxy.revealed) < windows and time.monotonic() < end:
        await asyncio.sleep(0.25)
    reveals.close()
    refused, timing = certifier.refused()
    certifier.shutdown()
    log |= {"revealed": sorted(proxy.revealed), "issuer_refused": [[l, d.value, w] for l, d, w in refused],
            "shim_exit": shim.exitcode}
    (out / "serve.json").write_text(json.dumps(log, indent=1))
    if timing is not None:
        (out / "publisher_timing.json").write_text(json.dumps(
            {"origin": origin, "switchinterval": sys.getswitchinterval(),
             "bucket_ms": {"egress": e.bucket_ms, "ingress": i.bucket_ms}, "start": proxy.start,
             "links": {"egress": setup.egress_link, "ingress": setup.ingress_link}, "calls": timing}))
    return log


async def client(plan: dict, report: dict, deployment: dict, out: Path,
                 link: tuple[str, int] | None = None) -> list[str]:
    """The client on the link (`active_replay.Peer`), at the deployment's origin on this host's clock, connecting to
    `link`, by default the deployment's; once the wire has ended, it keeps every window's certificates
    (`keep_certificates`) and returns what it refused."""
    out.mkdir(parents=True, exist_ok=True)
    peer = ar.Peer(plan, ar.wire_params(ar.Setup(report)))
    reader, writer = await asyncio.open_connection(*(link or deployment["link"]), limit=1 << 22)
    origin = time.monotonic() + deployment["origin_unix"] - time.time()
    sending = asyncio.create_task(peer.send(writer, origin))
    await peer.read(reader, origin)
    sending.cancel()
    writer.close()
    ar.write_logs(out, peer=peer)
    return await asyncio.to_thread(keep_certificates, deployment, out)


def ask_certificates(address: Sequence, window: int, timeout: float | None = 60.0) -> dict:
    """The reveal endpoint's answer for window `window`'s certificates (`active.serve_reveals`): one connection, one
    request line, one answer line."""
    with socket.create_connection(tuple(address), timeout=timeout) as sock, sock.makefile("rwb") as stream:
        stream.write(json.dumps({"window": window, "kind": "certificates"}).encode() + b"\n")
        stream.flush()
        sock.shutdown(socket.SHUT_WR)
        return json.loads(stream.readline(REVEAL_LINE_BYTES))


def check_certificates(deployment: dict, window: int, answer: dict,
                       seen: dict[bytes, str] | None = None) -> tuple[list, list, list, list[str]]:
    """Window `window`'s certificates and clock certificates as the reveal endpoint served them
    (`CertifyingPublisher.public`, each the `Certificate` alone), each held to the deployment's certifier key and
    session: (certificates, clock certificates, the link-windows the signer refused, refusals). A refusal names the
    certificate and what fails: another key's, another link-window's, a clock certificate that isn't egress, a
    signature that doesn't verify over its fields in the session (forged, or a field altered, the digest it signs
    among them), a second of its direction, or a clock certificate signing the digest an earlier window's in `seen`
    (digest to name, which this adds to) signs: its salt was reused, under which two windows of one sync sign one
    digest. The client holds no sync or salt, so it checks no opening. An answer that is a refusal, another window's
    or another session's, or not certificates, is refused whole."""
    key, session, link = read_public(deployment["certifier_key"]), deployment["session"], deployment["links"]["n2"]
    seen = {} if seen is None else seen
    try:
        if set(answer) == {"refused"}:
            raise ValueError(f"the certifier refused them: {answer['refused']}")
        if (answer["window"], answer["session"]) != (window, session):
            raise ValueError(f"the answer is window {answer['window']}'s in session {answer['session'][:16]}…, not "
                             f"the deployment's session {session[:16]}…")
        certificates = [certificate_from_json(d) for d in answer["certificates"]]
        clocks = [certificate_from_json(d) for d in answer["clocks"]]
        unsigned = [[u[0], Direction(u[1]).value, u[2]] for u in answer["unsigned"]]
    except (KeyError, TypeError, ValueError, IndexError) as e:
        return [], [], [], [f"window {window}'s certificates: {e}"]
    s, kept, refused = bytes.fromhex(session), {"certificate": [], "clock certificate": []}, []
    for what, c in [("certificate", c) for c in certificates] + [("clock certificate", k) for k in clocks]:
        name = f"window {window}'s {c.direction.value} {what} of {c.link} (key {c.key_id.hex()[:16]}…)"
        clock = what == "clock certificate"
        if c.key_id != key.key_id:
            refused.append(f"{name}: another key's, not the deployment's {key.key_id.hex()[:16]}…")
        elif (c.link, c.window) != (link, window):
            refused.append(f"{name}: it names {c.link} window {c.window}, not {link} window {window}")
        elif clock and c.direction is not Direction.EGRESS:
            refused.append(f"{name}: a clock is an egress window's")
        elif not c.verify(key, s):
            refused.append(f"{name}: its signature doesn't verify under the deployment's key in its session")
        elif any(x.direction is c.direction for x in kept[what]):
            refused.append(f"{name}: a second one of its direction")
        elif clock and c.digest in seen:
            refused.append(f"{name}: its signed digest is also {seen[c.digest]}'s, so its salt was reused, under "
                           "which two windows of one sync sign one digest")
        else:
            if clock:
                seen[c.digest] = f"window {c.window}'s clock certificate"
            kept[what].append(c)
    return kept["certificate"], kept["clock certificate"], unsigned, refused


def keep_certificates(deployment: dict, out: Path, ask=ask_certificates) -> list[str]:
    """The client's copy of each window's certificates' public part, asked of the reveal endpoint (`ask`) and held
    to the deployment's key and session (`check_certificates`): those that verify go in `certificates.jsonl`, each
    with its session (a link-window the signer refused as `unsigned`), and the clock certificates in `clocks.jsonl`,
    each with its session too, beside `peer.json`. Returns the refusals, each naming what it refused."""
    refused: list[str] = []
    seen: dict[bytes, str] = {}
    with open(out / "certificates.jsonl", "w") as certs, open(out / "clocks.jsonl", "w") as clocks:
        for w in range(deployment["windows"]):
            try:
                answer = ask(deployment["reveal"], w)
            except (OSError, ValueError) as e:
                refused.append(f"window {w}'s certificates: the reveal endpoint gave no answer: {e}")
                continue
            got, synced, unsigned, no = check_certificates(deployment, w, answer, seen)
            refused += no
            certs.writelines(json.dumps({"session": deployment["session"]} | certificate_json(c)) + "\n" for c in got)
            certs.writelines(json.dumps({"unsigned": u}) + "\n" for u in unsigned)
            clocks.writelines(json.dumps({"session": deployment["session"]} | certificate_json(k)) + "\n"
                              for k in synced)
    return refused


# The offline check.

def certified_io(plan: dict, setup: ar.Setup, in_records, out_records, aborted: Collection[str] = ()) -> dict:
    """The certified I/O of a varied run, read from the records' frames alone (the module docstring's "The certified
    I/O"): each request's ingress ordinal (its stream), release and fields (`active.id_request_fields`) from the
    ingress frames, and its response's token ids and frames (canonical position, egress bucket) from the egress id
    events that name its ordinal. `accepted` when every request is an id request whose `user` no other request has,
    every egress frame is an id event of a request released no later than the frame's bucket, and every response keeps
    the finish rule: at most `max_tokens` tokens, ending at its first EOS (`eos_token_ids`) or at `max_tokens`. On a
    compact plan each egress frame is a compact cell (`active.compact_record_data`), and the finish is the records'
    too: only the last is marked, a stop on an EOS id, `COMPACT_LAST` alone at `max_tokens` exactly, and no other
    record carries an EOS id. Where a response stops is the server's word here, as an EOS id event's is: only the I/O
    tie holds it to the ids the model sampled (`active.compact_finish_violations`). `aborted` are the users whose
    finish in the Commit is a client's abort, which nothing on the link says: one of theirs ends instead short of
    `max_tokens`, unmarked and on no EOS id (finish `abort`), and its marked finish is refused. `check` reads no Commit,
    so it gives none."""
    e, i = setup.egress_params, setup.ingress_params
    compact = bool(plan.get("compact"))
    marks: dict[int, list] = defaultdict(list)
    release, _ = ar.ingress_releases(in_records, i, e)
    data, split = defaultdict(bytes), Requests()
    for rec in sorted(in_records, key=lambda r: r.window):
        for row in rec.grid:
            for slot in row[:row_count(row)]:
                kind, _, payload = decode_frame(slot)
                if kind == DATA:
                    stream, chunk, _ = split.feed(payload)
                    data[stream] += chunk
    requests, by_ordinal, violations, users = [], {}, [], {}
    for stream in sorted(data):
        fields = id_request_fields(data[stream], i.frame_bytes)
        if fields is None:
            violations.append([stream, "a request that isn't an id request"])
            continue
        if (first := users.setdefault(fields["user"], stream)) != stream:
            violations.append([stream, f"a request whose user names request {first} too"])
        r = by_ordinal[stream] = fields | {"stream": stream, "release": release.get(stream), "output": [],
                                           "frames": []}
        requests.append(r)
    position = 0
    for rec in sorted(out_records, key=lambda r: r.window):
        for t, row in enumerate(rec.grid):
            for slot in row[:row_count(row)]:
                if compact:
                    got = compact_record_data(slot)
                else:
                    kind, _, payload = decode_frame(slot)
                    d = id_event_data(payload) if kind == DATA else None
                    got = None if d is None else (*d, None)
                r = by_ordinal.get(got[0]) if got is not None else None
                bucket = rec.window * e.buckets + t
                if r is None:
                    violations.append([position, "an egress frame that isn't an id event of a released request"])
                elif r["release"] is None or bucket < r["release"]:
                    violations.append([position, "an id event before its request's release"])
                else:
                    r["output"].append(got[1])
                    r["frames"].append([position, bucket])
                    marks[r["stream"]].append(got[2])
                position += 1
    eos, finish = set(plan["varied"]["eos_token_ids"]), Counter()
    for r in requests:
        out, m, gone = r["output"], r["max_tokens"], r["user"] in aborted
        stops = [k for k, t in enumerate(out) if t in eos]
        ends = marks[r["stream"]]
        if not out:
            r["finish"] = "no response"
        elif len(out) > m:
            r["finish"] = "longer than max_tokens"
        elif compact and any(ends[:-1]):
            r["finish"] = "a token after its finish"
        elif compact and any(ends[k] != "stop" for k in stops):
            r["finish"] = "an EOS id outside a stop record"
        elif compact and ends[-1] == "stop" and not stops:
            r["finish"] = "a stop on an id that isn't EOS"
        elif compact:
            short, unmarked = "a length finish short of max_tokens", "no finish"
            if gone:
                unmarked = "abort" if len(out) < m else "an abort at max_tokens"
            r["finish"] = {"stop": "eos", "length": "length" if len(out) == m else short, None: unmarked}[ends[-1]]
        elif stops and stops[0] != len(out) - 1:
            r["finish"] = "a token after EOS"
        else:
            r["finish"] = ("eos" if stops else "length" if len(out) == m else "abort" if gone
                           else "short of max_tokens without EOS")
        if gone and r["finish"] in ("eos", "length"):
            r["finish"] = f"a finish {r['finish']} where the Commit's is abort"
        finish[r["finish"]] += 1
        if r["finish"] not in ("eos", "length", "abort"):
            violations.append([r["stream"], r["finish"]])
    return {"accepted": not violations and bool(requests), "requests": len(requests),
            "tokens": sum(len(r["output"]) for r in requests), "finish": dict(finish), "violations": violations[:20],
            "records": requests}


def _shim_ids(run: Path, compact: bool = False) -> dict[tuple[str, int], list[int]]:
    """The token ids the shim wrote (`server.events.gz`), in its order, by each event's session and the ordinal it
    names; its compact records' under `compact`."""
    out: dict[tuple[str, int], list[int]] = defaultdict(list)
    for _, name, _, _, e in sorted(ar.read_events(run / "server.events.gz")):
        if compact:
            if (c := compact_record_data(e[-COMPACT_RECORD.size :])) is not None:
                out[name, c[0]].append(c[1])
        elif (got := id_event_data(event_body(e))) is not None:
            out[name, got[0]].append(got[1])
    return dict(out)


def egress_bytes(setup: ar.Setup, out_records) -> dict:
    """Per egress window, the link's bytes (T·r·B, at the constant rate) and its frames' (Σ counts · B), at the run's
    B and, for the same frames (one a token either way), at an id event's cell (`ID_EVENT_BYTES` and its header in a
    256-byte frame) and at the rehearsal's id-event frame (`FRAME_BYTES`)."""
    e = setup.egress_params
    frames = [sum(occupancy(r.grid)) for r in sorted(out_records, key=lambda r: r.window)]
    mean = sum(frames) / len(frames) if frames else 0.0
    sizes = {"run": e.frame_bytes, "id_event_cell": 1 << (HEADER.size + ID_EVENT_BYTES - 1).bit_length(),
             "id_event_frame": FRAME_BYTES}
    return {"frame_bytes": e.frame_bytes, "windows": len(frames), "frames_per_window": round(mean, 3),
            "frames_max": max(frames, default=0),
            "wire_per_window": {k: e.buckets * e.allocation * b for k, b in sizes.items()},
            "frames_bytes_per_window": {k: round(mean * b, 1) for k, b in sizes.items()}}


def _certified(lw: LinkWindows, opening, syncs: Sequence[ClockSync]) -> bool:
    """The host audit of one link-window's two certified records under the certifier's key and session, comparing no
    grid: N2's audit (`LinkWindows.verify`) without its head checks, which need a profile."""
    w, (egress, ingress) = opening.window, opening.records
    a = host_audit.Audit()
    out = a.check_link(lw.link, [egress], (), list(syncs)[:w + 1], lw.sync_set, lw.params, key=lw.key,
                       session=lw.certifier_session, grid=False)
    inn = a.check_ingress_link(lw.link, [ingress], w + 1, key=lw.key, params=lw.ingress_params,
                               session=lw.certifier_session)
    return len(out) > w and out[w].accepted and inn[w].accepted


def _no_coins(n: int) -> bytes:
    raise AssertionError("Selection All draws every link-window and reads no coins")


def _refusal(err: Exception) -> str:
    return f"refused: {err}"


def check(run: Path, challenger: LmsSigner, *, clock=time.monotonic, timeout: float | None = 60.0,
          proof_deadline: float = PROOF_DEADLINE_S, audit=Audit) -> dict:
    """The auditor's offline check of the served run in `run` (the module docstring's `check`), with `challenger` the
    challenger's signer, whose public key the certifier registered. Its `verdict` is `accepted` when `network_verdict`
    accepts, the certifier's audit accepts every link-window raw and committed, every window's salt was revealed once and
    only once, and the salt audit accepts; else `refused`."""
    plan = json.loads((run / "plan.json").read_text())
    report = json.loads((run / "calibration.json").read_text())
    deployment = json.loads((run / "deployment.json").read_text())
    setup = ar.Setup(report)
    windows, key = plan["windows"], read_public(deployment["certifier_key"])
    syncs = [ClockSync(d, Fraction(r)) for d, r in plan["syncs"]]
    profile, served = plan.get("occupancy") or {}, served_profile(plan)
    advice = profile.get("kind") == ADVICE
    lw = LinkWindows(deployment["links"]["n2"], setup.egress_params, setup.sync_set, key, deployment["run_id"],
                     bytes.fromhex(deployment["session"]), ingress=setup.ingress_params, proof_deadline=proof_deadline,
                     occupancy=(profile.get("egress"), profile.get("ingress")), served=served)
    remote = RemoteCertifier(*deployment["reveal"], key, challenger, deployment["links"]["ingress"],
                             bytes.fromhex(deployment["run_id"]), timeout=timeout)
    stream = lw.stream(budget=windows, delta=DELTA, challenger=CHALLENGER, coins=_no_coins, clock=clock)
    certified, refused = read_certified(run / "certified.jsonl")
    rows, salts, previous, again = [], {}, None, None
    for w in range(windows):
        row: dict = {"window": w}
        rows.append(row)
        pair = certified.get((Direction.EGRESS, w)), certified.get((Direction.INGRESS, w))
        if None in pair:
            row["n2"] = row["salt"] = "no certified record"
            break
        opening = lw.commit(w, *pair)
        if advice:
            record = lw.registration(opening, previous)
            registration = Registration(record, R.receipt(record, CHALLENGER, received=str(w)), R.digest(record), w)
        else:
            registration = lw.register(stream, lw.registration(opening, previous))
        previous = registration.digest
        try:
            salts[w] = remote.reveal(w, registration)
            row["salt"] = "revealed"
        except RuntimeError as err:
            row["salt"] = _refusal(err)
        if w == 0:
            try:
                remote.reveal(0, registration)
                again = "revealed again"
            except RuntimeError as err:
                again = _refusal(err)
        row["salt_fingerprint"] = ar.fingerprint(salts[w]) if w in salts else None
        if advice:
            row["certified"] = _certified(lw, opening, syncs)
            row["n2"] = NO_N2
            continue
        if served is not None:
            row["served_counts"] = lw.served_check(opening, syncs)
        draw = stream.select(registration)
        try:
            outcome = (stream.deliver(draw, [lw.session(opening, draw)])
                       or lw.unproved_outcome(stream, draw, opening, syncs))
        except ValueError as err:
            row["n2"] = f"no verdict: {err}"
            break
        row["n2"] = {"accepted": outcome.accepted, "codes": list(outcome.verdict.codes), "public": dict(outcome.public)}
    if advice:
        network = {"accepted": None, "windows": windows, "error": NO_N2}
    else:
        try:
            network = lw.network_verdict(stream, windows)
        except ValueError as err:
            network = {"accepted": False, "error": str(err)}
    client = run / CLIENT if (run / CLIENT / "peer.json").exists() else run
    audited = ar.audit_configuration(plan, report, run, audit=audit, client=client)
    e, i = setup.egress_params, setup.ingress_params
    in_records = [ar.unpack_record(b, i.frame_bytes) for b in ar.read_blobs(run / "requests.records")]
    out_records = [ar.unpack_record(b, e.frame_bytes) for b in ar.read_blobs(run / "responses.records")]
    server = json.loads((run / "server.json").read_text())
    salt = ar.audit_salt(setup, salts, in_records, out_records, server, windows, audit())
    io = None
    if advice:
        io = certified_io(plan, setup, in_records, out_records)
        records = io.pop("records")
        (run / "io.json").write_text(json.dumps({"windows": windows, "requests": records}))
        shim = _shim_ids(run, bool(plan.get("compact")))
        io["ids_are_the_shims"] = {(r["user"], r["stream"]): r["output"] for r in records if r["output"]} == shim
        sent = {q["session"]: q["id_request"] | {"user": q["session"]} for q in plan["requests"]}
        fields = ("model", "user", "prompt", "max_tokens", "temperature", "top_p", "seed")
        io["requests_are_the_plans"] = all({k: r[k] for k in fields} == sent.get(r["user"]) for r in records)
    by_window: dict[int, dict] = {}
    for r in audited["rows"]:
        by_window.setdefault(r["window"], {})[r["direction"]] = {"raw": r["raw"], "committed": r["committed"]}
    for row in rows:
        row["audit"] = by_window.get(row["window"], {})
    audit_ok = bool(audited["rows"]) and all(r["raw"] is None and r["committed"] is None for r in audited["rows"])
    salt_ok = (len(salts) == windows and again is not None and again.startswith("refused")
               and salt["accepted"] and salt["block_received"] and not salt["egress_holds_salt"])
    if advice:
        certified_ok = len(rows) == windows and all(r.get("certified") for r in rows)
        accepted = certified_ok and io["accepted"] and audit_ok and salt_ok
    else:
        certified_ok, accepted = None, bool(network["accepted"]) and audit_ok and salt_ok
    out = {"verdict": "accepted" if accepted else "refused", "windows": windows,
           "network_verdict": network, "audit_accepted": audit_ok, "salt_accepted": salt_ok,
           "salt_again": again, "salt": salt, "profile": profile.get("kind"), "issuer_refused": refused,
           "rows": rows, "comparisons": audited["comparisons"], "timing": audited["timing"],
           "egress_bytes": egress_bytes(setup, out_records)}
    return out if not advice else out | {"certified_accepted": certified_ok, "io": io}


def table(result: dict) -> str:
    """The check's verdicts, one line per link-window."""
    lines = [f"verdict {result['verdict']}: network_verdict {result['network_verdict'].get('accepted')}, audit "
             f"{result['audit_accepted']}, salt {result['salt_accepted']} (window 0 again: {result['salt_again']}), "
             f"N2 on the {result['profile']} profile"]
    if (io := result.get("io")) is not None:
        lines.append(f"  certified {result['certified_accepted']}; I/O {io['accepted']}: {io['requests']} requests, "
                     f"{io['tokens']} tokens, finish {io['finish']}, ids are the shim's {io['ids_are_the_shims']}, "
                     f"requests are the plan's {io['requests_are_the_plans']}, violations {io['violations'][:3]}")
    for row in result["rows"]:
        a = row.get("audit", {})
        n2 = row.get("n2")
        n2 = ("accept" if n2["accepted"] else f"reject {n2['codes']}") if isinstance(n2, dict) else n2
        n2 = "none (advice profile)" if n2 == NO_N2 else n2
        counts = f" (served counts {row['served_counts']})" if "served_counts" in row else ""
        certified = f"; certified {row['certified']}" if "certified" in row else ""
        lines.append(f"  window {row['window']}: audit egress {a.get('egress')} ingress {a.get('ingress')}; "
                     f"N2 {n2}{counts}{certified}; salt {row.get('salt')}")
    return "\n".join(lines)


# The rehearsal.

def rehearse_one(out: Path, strategy: str, *, url: str, certifier_dir: Path, challenger_dir: Path,
                 api_key_env: str | None, start_in: float, out_shape: dict | None = None,
                 paced_ms: float = PACED_MS) -> dict:
    """One served run of the plan in `out` as processes: `serve` (with `--tamper` unless honest, `--out-shape`, the
    run's ``out-shape.json``, when given, and `--paced-ms`), then `client` (into the run's `client/`), then `check`
    while `serve` serves reveals."""
    run = out / strategy
    run.mkdir(parents=True, exist_ok=True)
    if out_shape is not None:
        (run / "out-shape.json").write_text(json.dumps(out_shape))
    me = [sys.executable, str(HERE / "certifier_serve.py")]
    served = me + ["serve", "--plan", str(out / "plan.json"), "--calibration", str(out / "calibration.json"),
                   "--upstream", url, "--out", str(run), "--link", "127.0.0.1:0", "--reveal", "127.0.0.1:0",
                   "--key", str(certifier_dir), "--challenger-key", str(challenger_dir / "public.json"),
                   "--session", secrets.token_hex(32), "--start-in", str(start_in), "--paced-ms", str(paced_ms)]
    served += (["--tamper", strategy] if strategy != "honest" else []) + \
        (["--api-key-env", api_key_env] if api_key_env else []) + \
        (["--out-shape", str(run / "out-shape.json")] if out_shape is not None else [])
    with open(run / "serve.log", "w") as log:
        proc = subprocess.Popen(served, stdout=subprocess.PIPE, stderr=log, text=True)
        try:
            listening = json.loads(proc.stdout.readline() or "{}")
            if "listening" not in listening:
                raise SystemExit(f"{strategy}: serve exited before listening (serve.log)")
            with open(run / "client.log", "w") as clog:
                client_exit = subprocess.run(me + ["client", "--plan", str(out / "plan.json"), "--calibration",
                                                   str(out / "calibration.json"), "--deployment",
                                                   str(run / "deployment.json"), "--out", str(run / CLIENT)],
                                             stdout=clog, stderr=subprocess.STDOUT).returncode
            done = json.loads(proc.stdout.readline() or "{}")
            if not done.get("served"):
                raise SystemExit(f"{strategy}: serve served no run (serve.log)")
            challenger = signer(challenger_dir)
            try:
                result = check(run, challenger)
            finally:
                challenger.close()
            serve_exit = proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
    result |= {"strategy": strategy, "exit": {"serve": serve_exit, "client": client_exit}}
    (run / "check.json").write_text(json.dumps(result, indent=1))
    print(f"{strategy}: {table(result)}", flush=True)
    return result


def plan_of(args, url: str, clock=time.monotonic, sleep=time.sleep) -> tuple[dict, dict, dict]:
    """(probe, calibration, plan) of the endpoint at `url` (`plan`'s): each request `--tokens` tokens out of a prompt of
    exactly `--prompt-tokens` tokens (`PROMPT` as text with 0), probed and planned `--in-flight` at a time; or, with
    `--traffic varied`, `--requests` requests of varied traffic (`varied_requests`: `--rate`, `--prompt-range` or
    `--prompts-file`'s prompts, `--tokens` the cap, `--seed`, `--name` the users'), the first `--probes` of them probed
    at their arrival times."""
    model = args.model or served_model(url, args.api_key_env)
    prompts_file = getattr(args, "prompts_file", None)
    if getattr(args, "traffic", TRAFFIC[0]) == "varied":
        lo, _, hi = args.prompt_range.partition(":")
        name = getattr(args, "name", None) or "varied"
        requests = varied_requests(url, model, args.requests, rate=args.rate, prompt_tokens=(int(lo), int(hi)),
                                   max_tokens=args.tokens, seed=args.seed, name=name, api_key_env=args.api_key_env,
                                   prompts=None if prompts_file is None else json.loads(prompts_file.read_text()))
        probed = probe_varied(url, model, requests[:args.probes], args.api_key_env, clock, sleep)
        eos = [int(t) for t in args.eos_ids.split(",")] if args.eos_ids else None
        report, plan = varied_plan(probed, requests, seed=args.seed, eos=eos, compact=getattr(args, "compact", None),
                                   margin=getattr(args, "margin", 2.0))
        return probed, report, plan
    if prompts_file is not None:
        raise SystemExit("--prompts-file: varied traffic's (--traffic varied); fixed traffic's prompts are --prompts")
    prompts = parse_prompts(args.prompts) if getattr(args, "prompts", None) else None
    if prompts:
        prompt = prompts[0]
    else:
        prompt = prompt_tokens(url, model, args.prompt_tokens, args.api_key_env) if args.prompt_tokens else PROMPT
    probed = probe(url, request_body(model, args.tokens, prompt, args.temperature, args.top_p), args.probes,
                   args.api_key_env, clock,
                   in_flight=args.in_flight, warmup=args.in_flight)
    report, plan = served_plan(probed, args.requests, args.delta_buckets, in_flight=args.in_flight,
                               profile=args.profile, frames=args.response_frames, prompts=prompts,
                               window_per_group=getattr(args, "window_per_group", False))
    return probed, report, plan


def parse_prompts(text: str) -> list[list[int]]:
    """``--prompts``: prompts of token ids, ``;`` between prompts and ``,`` between ids (``3;7``: [[3], [7]])."""
    prompts = [[int(t) for t in p.split(",")] for p in text.split(";") if p.strip()]
    if not prompts or not all(prompts):
        raise SystemExit(f"--prompts: token ids, ',' between them and ';' between prompts, not {text!r}")
    return prompts


def rehearse(args) -> int:
    from verity.core.protocols.certifier import lean_audit
    from verity.core.protocols.sampled_proofs.one_stage import verifier

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    # both built before the first run, so that the check's first registration and verdict don't build them while the
    # reveal deadline runs (on a loaded host a build outlasts window 0's margin)
    if "VERITY_NETWORK_RECORDS_EXE" not in os.environ:
        os.environ["VERITY_NETWORK_RECORDS_EXE"] = str(lean_audit.build())
    verifier.executable()
    scratch = Path(tempfile.mkdtemp(prefix="certifier-serve-"))
    certifier_dir, challenger_dir = scratch / "certifier", scratch / "challenger"
    for d in (certifier_dir, challenger_dir):
        load_key(d, args.height)
    started: list = []
    try:
        url = args.upstream
        if url is None:
            host, port = ar.start_vllm(argparse.Namespace(vllm=args.vllm, served_model=args.model,
                                                          vllm_cpus=args.vllm_cpus), out, started)
            url = f"http://{host}:{port}"
        probed, report, plan = plan_of(args, url)
        (out / "probe.json").write_text(json.dumps(probed, indent=1))
        (out / "plan.json").write_text(json.dumps(plan))
        (out / "calibration.json").write_text(json.dumps(report))
        results = {s: rehearse_one(out, s, url=url, certifier_dir=certifier_dir, challenger_dir=challenger_dir,
                                   api_key_env=args.api_key_env, start_in=args.start_in, out_shape=args.out_shape,
                                   paced_ms=args.paced_ms)
                   for s in args.strategies.split(",")}
    finally:
        shutil.rmtree(scratch)
        if started:
            import record

            record.stop_server(started[0])
    summary = {"upstream": url, "vllm": args.vllm, "model": plan["live_model"], "lms_height": args.height,
               "served": plan.get("served"), "varied": plan.get("varied"), "windows": plan["windows"],
               "profile": plan["occupancy"]["kind"],
               "compact": plan.get("compact"),
               "runs": {s: {k: r.get(k) for k in ("verdict", "network_verdict", "audit_accepted", "salt_accepted",
                                                  "salt_again", "certified_accepted", "io", "exit", "egress_bytes")}
                        | {"windows": [{k: row.get(k) for k in ("window", "n2", "served_counts", "certified", "salt",
                                                                 "audit")}
                                       for row in r["rows"]]}
                        for s, r in results.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    ok = all(r["verdict"] == ("accepted" if s in ACCEPTED else "refused") for s, r in results.items())
    print(f"rehearsal {'as expected' if ok else 'NOT as expected'}: "
          f"{ {s: r['verdict'] for s, r in results.items()} }", flush=True)
    return 0 if ok else 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="certifier_serve.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    k = sub.add_parser("keys", help="make or load an LMS key and write its public.json")
    k.add_argument("--dir", required=True, type=Path)
    k.add_argument("--height", type=int, default=HEIGHT)
    p = sub.add_parser("plan", help="probe the endpoint and write the served plan and its calibration")
    s = sub.add_parser("serve", help="the certifier in front of the endpoint, for one run")
    c = sub.add_parser("client", help="the client on the link")
    x = sub.add_parser("check", help="the auditor's offline check of a run")
    r = sub.add_parser("rehearse", help="keys, plan, then serve, client and check per strategy")
    for q in (p, s, r):
        q.add_argument("--upstream", required=q is not r, help="the endpoint, http(s)://HOST[:PORT][/PREFIX]")
        q.add_argument("--api-key-env", help="the environment variable that holds the endpoint's bearer key")
    for q in (p, r):
        q.add_argument("--model", help="the endpoint's model name (default: the one /v1/models lists)")
        q.add_argument("--temperature", type=float, default=0.0, help="every request's temperature")
        q.add_argument("--top-p", type=float, default=1.0, help="every request's top_p")
        q.add_argument("--requests", type=int, default=400)
        q.add_argument("--in-flight", type=int, default=4, help="requests sent together, served side by side")
        q.add_argument("--tokens", type=int, default=64,
                       help="fixed: max_tokens of every request (ignore_eos, no min_tokens); varied: max_tokens' cap")
        q.add_argument("--response-frames", type=int,
                       help="fixed: F, a response's canonical events, padded to (default: --tokens); a token of more "
                            "than 64 bytes is one event per 64")
        q.add_argument("--prompt-tokens", type=int, default=128,
                       help="the prompt's length in the endpoint's tokens (/tokenize); 0: the text prompt as is")
        q.add_argument("--prompts", help="fixed: request j's prompt is the (j mod n)-th of these n prompts of token "
                                         "ids, ';' between prompts and ',' between ids (in place of --prompt-tokens)")
        q.add_argument("--window-per-group", action="store_true",
                       help="fixed: each group of --in-flight requests starts a window of its own")
        q.add_argument("--probes", type=int, default=8)
        q.add_argument("--delta-buckets", type=int,
                       help="the latency budget D (default: 4 × the slowest + 0.5 s, or the most that fits down to 2 ×)")
        q.add_argument("--profile", choices=PROFILES, default=PROFILES[0],
                       help="N2's profile: the planned egress occupancy, or the served profile, whose egress "
                            "counts derive from the ingress head's")
        q.add_argument("--traffic", choices=TRAFFIC, default=TRAFFIC[0],
                       help="fixed: one body every session (--tokens, --prompt-tokens, --in-flight, --profile); varied: "
                            "Poisson arrivals of varied prompts, real EOS, events carrying token ids (N2 on the advice "
                            "profile)")
        q.add_argument("--rate", type=float, default=2.0, help="varied: arrivals a second")
        q.add_argument("--prompt-range", default="16:1024", help="varied: LO:HI, the prompts' lengths in tokens")
        q.add_argument("--prompts-file", type=Path,
                       help="varied: a JSON array of prompts of token ids, request j's the (j mod n)-th of the n "
                            "(in place of --prompt-range's chat turns); the gate's certifier-rehearse --prompts")
        q.add_argument("--name", default="varied",
                       help="varied: the users' prefix: request j's user, its session, is NAME.j (j five digits)")
        q.add_argument("--seed", type=int, default=0, help="varied: the traffic's seed")
        q.add_argument("--eos-ids", help="varied: the model's EOS token ids, comma-separated (default: the last ids "
                                         "of the probe's responses that stopped)")
        q.add_argument("--compact", type=int, metavar="B",
                       help="varied: compact egress records (`active.compact_record`) in cells of B >= 8 bytes, "
                            "the shim writing each token's record (default: id events in 4096-byte frames); the "
                            "gate's certifier-rehearse --traffic varied --compact B runs it on the auditor's host")
        q.add_argument("--margin", type=float, default=2.0,
                       help="varied: the plan's margin over the probed rates and latencies (an endpoint that slows "
                            "under load the probes didn't meet needs more)")
    for q in (p, s, c, r):
        q.add_argument("--out", required=True, type=Path)
    for q in (s, c):
        q.add_argument("--plan", required=True, type=Path)
        q.add_argument("--calibration", required=True, type=Path)
    s.add_argument("--link", required=True, type=_address, help="HOST:PORT the client's link connects to")
    s.add_argument("--reveal", required=True, type=_address, help="HOST:PORT of the reveal endpoint")
    s.add_argument("--key", required=True, type=Path, help="the certifier's key directory (keys)")
    s.add_argument("--challenger-key", required=True, type=Path, help="the challenger's public.json")
    s.add_argument("--session", required=True, help="the certifier session, 32 bytes in hex, the challenger's coins")
    s.add_argument("--n2-link", default=N2_LINK)
    s.add_argument("--reveal-deadline", type=float, help="seconds past a window's end (default: the run's + 300)")
    for q in (s, r):
        q.add_argument("--start-in", type=float, default=10.0, help="the origin, seconds after the proxy is up")
        q.add_argument("--out-shape", type=_out_shape, metavar="SHAPE.json",
                       help="the output value's shape (out_commit.OutShape's fields, from the reading circuit's "
                            "registered read): the network certifier commits each egress link-window's output (the "
                            "hex of u16 words) in its rows and signs its head (out-records.jsonl; its rows and salts "
                            "in out-witness.jsonl, the prover's)")
        q.add_argument("--paced-ms", type=float, default=PACED_MS,
                       help="when the proxy's pacer writes each egress bucket, ms after its boundary (past its 25 ms "
                            "slack; 0: on the proxy's loop at its tick)")
    s.add_argument("--tamper", choices=TAMPERS,
                   help="late: the shim holds a window-0 session past every sync; extra: it writes one event more; "
                        "drop: one fewer; token: window 0's tokens' bytes changed")
    c.add_argument("--deployment", required=True, type=Path)
    c.add_argument("--link", type=_address, help="HOST:PORT of the link, if not the deployment's")
    x.add_argument("--run", required=True, type=Path)
    x.add_argument("--challenger", required=True, type=Path, help="the challenger's key directory")
    x.add_argument("--expect", choices=("accepted", "refused"))
    r.add_argument("--vllm", metavar="MODEL", help="start vLLM for MODEL on this host's GPU instead of --upstream")
    r.add_argument("--vllm-cpus", help="with --vllm: its CPUs (taskset -c)")
    r.add_argument("--height", type=int, default=HEIGHT, help="the LMS keys' tree height")
    r.add_argument("--strategies", default=",".join(STRATEGIES))
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)

    if args.command == "keys":
        load_key(args.dir, args.height)
        print((args.dir / "public.json").read_text())
        return 0
    if args.command == "plan":
        probed, report, plan = plan_of(args, args.upstream)
        args.out.mkdir(parents=True, exist_ok=True)
        for name, doc in (("probe.json", probed), ("plan.json", plan), ("calibration.json", report)):
            (args.out / name).write_text(json.dumps(doc, indent=1))
        print(json.dumps((plan.get("served") or plan["varied"]) | {"windows": plan["windows"]}))
        return 0
    if args.command == "serve":
        out = args.out.resolve()
        if out == args.key.resolve() or out in args.key.resolve().parents:
            raise SystemExit(f"the key {args.key} is inside --out {args.out}, which is published")
        session = bytes.fromhex(args.session)
        if len(session) != 32:
            raise SystemExit("--session is 32 bytes in hex")
        ar._quiet_runtime()
        log = asyncio.run(serve(json.loads(args.plan.read_text()), json.loads(args.calibration.read_text()), args.out,
                                url=args.upstream, upstream=upstream_params(args.upstream, args.api_key_env),
                                link=args.link, reveal=args.reveal, key=args.key,
                                challenger_key=read_public(json.loads(args.challenger_key.read_text())),
                                session=session, start_in=args.start_in, reveal_deadline=args.reveal_deadline,
                                n2_link=args.n2_link, tamper=args.tamper, out_shape=args.out_shape,
                                paced_ms=args.paced_ms))
        return 1 if "error" in log else 0
    if args.command == "client":
        ar._quiet_runtime()
        refused = asyncio.run(client(json.loads(args.plan.read_text()), json.loads(args.calibration.read_text()),
                                     json.loads(args.deployment.read_text()), args.out, args.link))
        print(json.dumps({"certificates_refused": refused}), flush=True)
        return 1 if refused else 0
    if args.command == "check":
        challenger = signer(args.challenger)
        try:
            result = check(args.run, challenger)
        finally:
            challenger.close()
        (args.run / "check.json").write_text(json.dumps(result, indent=1))
        print(table(result))
        return 0 if args.expect in (None, result["verdict"]) else 1
    return rehearse(args)


if __name__ == "__main__":
    raise SystemExit(main())
