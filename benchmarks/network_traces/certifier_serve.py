"""certifier-serve: the network certifier (`active.Proxy`) on one host, as the route of the measured link in front of a
served vLLM OpenAI endpoint, and the offline check of what it published.

**Topology.** The client, outside the boundary, holds one link connection to the certifier's host. There the certifier
(`active.Proxy` with its salt on, signing every record and every reveal) forwards each request it releases over
loopback to the shim (`active_replay.LiveServer`, the canonical framer, which is the prover's side), and the shim sends
it to the endpoint over HTTP or HTTPS and frames the stream it gets back as canonical events:

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
  firewall and the challenger in one process, with the network registration of each link-window standing in for
  PoUW's registration of the window, which opens the window's salt;
- the reveal deadline covers the whole run and a margin (`REVEAL_MARGIN_S`), where a deployment's covers one PoUW
  window; and the keys are LMS at H = `HEIGHT` (2^10 one-time keys), not the deployed H = 20.

**The served plan** (`plan`, `served_plan`). The endpoint is probed with the one request body every session sends,
straight from the host, `--in-flight` requests at a time after one uncounted round: `/v1/completions`, streamed, a
prompt of exactly `--prompt-tokens` tokens (`PROMPT` padded with `FILLER` in the endpoint's own tokenizer, `/tokenize`,
and cut there), exactly `--tokens` tokens out (`min_tokens` and `max_tokens`, `ignore_eos`), temperature 0. F, the
frames of a response, is that token count, not the endpoint's events: vLLM v1 merges a request's tokens that its API
server hasn't read yet into one chunk, so with several in flight a response's last two tokens sometimes share one
event and sometimes don't, though the tokens are the same. An event is a chunk of at least one token, so a response
is at most F canonical events, and the shim pads one that ends short with empty canonical events (`active.event(b"")`,
what a chunk of no text gives) up to F, after its last (`pad_to`, `active_replay.LiveServer`). Each probed response's
raw count, which must be 1 … F, and its seconds to the last are kept; a response of more than F (a chunk of more than
`active.TOKEN_BYTES` bytes of text) isn't cut, and N2's counts refuse its window. D, the latency budget in buckets, is
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
  too (`served_counts`, the rehearsal's read of the witness, beside the audit's); N2's verdict of record shows only
  accept or reject.

**Subcommands**, all from the repository root or with `benchmarks/network_traces` on the path:

    python certifier_serve.py keys --dir KEYDIR [--height 10]
    python certifier_serve.py plan --upstream URL --out DIR [--model NAME] [--requests 400] [--in-flight 4]
        [--tokens 64] [--prompt-tokens 128] [--probes 8] [--delta-buckets D] [--profile planned|served]
    python certifier_serve.py serve --plan PLAN --calibration REPORT --upstream URL --out DIR --link HOST:PORT
        --reveal HOST:PORT --key KEYDIR --challenger-key PUBLIC.json --session HEX [--start-in 10] [--tamper late|extra]
    python certifier_serve.py client --plan PLAN --calibration REPORT --deployment DEPLOYMENT.json --out DIR
    python certifier_serve.py check --run DIR --challenger KEYDIR [--expect accepted|refused]
    python certifier_serve.py rehearse (--upstream URL | --vllm MODEL) --out DIR [plan's options] [--strategies S,...]

- `keys` makes an LMS key (`seed`, 0600, `public`, its tree, and its leaf state `state.json`), or loads one, and writes
  `public.json`, the public key a deployment record carries. The certifier's key and the challenger's are each one.
- `serve` starts the shim, then the certifier's key in a process of its own (`CertifierProcess`, which keeps and
  certifies every record and signs every reveal off the proxy's GIL), the proxy, the link's listener and the reveal
  endpoint (`active.serve_reveals`), and writes `deployment.json`, the auditor's record of the run: the certifier's
  public key, the proxy's run id, the certifier session (`--session`, which the challenger draws), the links, the
  addresses, the origin and the reveal deadline. It takes one link connection, which must come before the origin. Once
  the link has ended it stops the shim, writes the logs and prints `{"served": ...}`, then serves reveals until every
  window's salt is revealed or the last window's reveal deadline has passed. `--tamper` makes the shim an adversary of
  `active_replay.adversary_plan`: `late` holds one session of window 0 past every sync, which the certifier's own audit
  faults; `extra` writes one event more at the end of the first session's response, after its padding (F + 1 in all),
  on time and logged, so the records are complete and only N2's counts against the profile refuse it. The key's
  directory may not be inside `--out`.
- `client` sends the plan's requests on the link as `active_replay.Peer` does, at the deployment's origin, and reads
  the wire to its end (`peer.json`).
- `check` is the auditor's offline check of a served run (`check`): per link-window, the certifier's audit of the raw
  and committed records, N2's verdict of record through `LinkWindows` (commit, register, select, deliver, outcome) on
  the certified records, each window's salt asked of the certifier through a `RemoteCertifier` signed with the
  challenger's key (window 0's asked twice: the second is refused), then `network_verdict` and the salt audit
  (`active_replay.audit_salt`: the salt frames, and that the shim received each revealed salt). It must run while
  `serve` serves reveals.
- `rehearse` makes both keys in a scratch directory outside `--out`, plans, and for each of `--strategies` (honest, late, extra)
  runs `serve` and `client` as processes and `check`; `summary.json` holds each run's verdicts. With `--vllm MODEL` it
  starts vLLM on this host's GPU first (`active_replay.start_vllm`).

**Outputs** in a run's directory: `plan.json`, `calibration.json`, `deployment.json`, the proxy's records
(`requests.records`, `responses.records`, `committed.jsonl`) and `certified.jsonl` (each record's head, the
certifier's salt for it and certificate, or the link-window the signer refused), `proxy.json.gz`, the shim's
`server.json` and `server.events.gz`, the client's `peer.json`, `serve.json` and `check.json`. Records, certificates,
the certifier's salts and the shim's events are witness, kept behind the firewall in a deployment; a rehearsal
publishes them so that its check can be replayed. No PoUW salt is written, only its SHA-512
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
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import active_replay as ar
from verity.ml.crypto import lms
from verity.core.protocols.certifier import calibration
from verity.core.protocols.certifier.active import Events, Proxy, RemoteCertifier, fresh_salt, request_frame_count, \
    serve_reveals
from verity.core.protocols.certifier.certificate import Certificate, CertifiedRecord, Direction, Issuer, LmsKey, \
    LmsSigner, RecordHead, parameters_digest
from verity.core.protocols.certifier.grid import COMPLETE, Reason, Status, occupancy
from verity.core.protocols.certifier.lean_audit import Audit
from verity.core.protocols.certifier.schedule import ClockSync, expected_grids
from verity.core.protocols.certifier.served import Admission, ServedProfile
from verity.core.protocols.certifier.service import LinkWindows

HERE = Path(__file__).resolve().parent
BUCKET_MS, WINDOW_BUCKETS, FRAME_BYTES, QUEUE = 100, 60, 4096, 1 << 16
EGRESS, INGRESS, N2_LINK = "responses", "requests", "served"
HEIGHT = 10
PROMPT = "Count from one to one hundred in words, separated by commas:"
FILLER = " one, two, three, four, five, six, seven, eight, nine, ten,"
REVEAL_MARGIN_S = 300.0
PROOF_DEADLINE_S = 600.0
DELTA = 2.0 ** -40
STRATEGIES = ("honest", "late", "extra")
PROFILES = ("planned", "served")


# Keys.

def load_key(directory: Path, height: int = HEIGHT) -> lms.PrivateKey:
    """The LMS key (N = 64, W = 1) in `directory`, made there first at `height` when absent, in the challengers' layout
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
        key = lms.keygen(params, seed)
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


def certified_json(record: CertifiedRecord) -> dict:
    return {"salt_c": record.salt.hex(),
            **{part: {f: _enc(getattr(obj, f)) for f in type(obj).__slots__}
               for part, obj in (("head", record.head), ("certificate", record.certificate))}}


def certified_from_json(doc: dict) -> CertifiedRecord:
    return CertifiedRecord(RecordHead(**{f: _dec(f, v) for f, v in doc["head"].items()}), bytes.fromhex(doc["salt_c"]),
                           Certificate(**{f: _dec(f, v) for f, v in doc["certificate"].items()}))


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


class CertifyingPublisher(ar.Publisher):
    """The served proxy's `publish`: `Publisher`'s files, and each record certified by `issuer` as the network certifier
    hands it over, under the N2 link's name `link`, the one name both ends of a link-window share
    (`service.LinkWindows.link`): one line of `certified.jsonl` each, or the link-window under `refused` when the
    signer refuses."""

    def __init__(self, out: Path, setup: ar.Setup, issuer: Issuer, link: str, threaded: bool = True) -> None:
        super().__init__(out, setup, threaded)
        self.issuer, self.link = issuer, link
        self.ends = {setup.egress_link: (Direction.EGRESS, parameters_digest(setup.egress_params, setup.sync_set)),
                     setup.ingress_link: (Direction.INGRESS, parameters_digest(setup.ingress_params))}
        self.timing: list | None = [] if os.environ.get("CERTIFIER_SERVE_TIMING") else None

    def keep(self, record, params) -> None:
        start = time.monotonic(), time.thread_time()
        super().keep(record, params)
        kept = time.monotonic(), time.thread_time()
        direction, parameters = self.ends[record.link]
        certified = self.issuer.certify(replace(record, link=self.link), direction, params, parameters)
        line = ({"refused": [self.link, direction.value, record.window]} if certified is None
                else certified_json(certified))
        with open(self.out / "certified.jsonl", "a") as fh:
            fh.write(json.dumps(line) + "\n")
        if self.timing is not None:
            end = time.monotonic(), time.thread_time()
            self.timing.append({"link": record.link, "window": record.window, "start": start[0], "kept": kept[0],
                                "end": end[0], "keep_cpu": kept[1] - start[1], "certify_cpu": end[1] - kept[1]})


_CERTIFIER: dict = {}


def _certifier_start(key: Path, out: Path, report: dict, session: bytes, link: str) -> None:
    own = signer(key)
    _CERTIFIER.update(signer=own, publish=CertifyingPublisher(out, ar.Setup(report), Issuer(own, session), link,
                                                              threaded=False))


def _certifier_call(name: str, *args):
    own, publish = _CERTIFIER["signer"], _CERTIFIER["publish"]
    if name == "keep":
        return publish.keep(*args)
    if name == "refused":
        return publish.issuer.refused, publish.timing
    if name == "close":
        return own.close()
    return getattr(own, name)(*args)


class CertifierProcess:
    """The served proxy's `publish` and its reveals' `signer` in one spawned process, which alone holds the certifier's
    key and its leaf state: each record kept and certified as `CertifyingPublisher(threaded=False)` keeps it, and each
    reveal signed, one call at a time in the order they are made. Certifying a window's records hashes for
    milliseconds, so a thread would hold the proxy's GIL at its window's end. `close` waits for every record (the
    signer stays up for the reveals), `refused` gives the issuer's refusals and `shutdown` closes the signer."""

    def __init__(self, key: Path, out: Path, report: dict, session: bytes, link: str) -> None:
        self.pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=_certifier_start, initargs=(key, out, report, session, link))
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

    def __call__(self, record, params) -> None:
        if self.submits is not None:
            self.submits.append((record.link, record.window, time.monotonic()))
        self.futures.append(self.pool.submit(_certifier_call, "keep", record, params))

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

def request_body(model: str, tokens: int, prompt: str | list[int] = PROMPT) -> dict:
    """The request every session of a served plan sends: exactly `tokens` tokens (`min_tokens` and `max_tokens`, never
    stopped early), greedy, streamed."""
    return {"model": model, "prompt": prompt, "max_tokens": tokens, "min_tokens": tokens, "stream": True,
            "ignore_eos": True, "temperature": 0}


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
                name: str = "served", in_flight: int = 1, profile: str = "planned") -> tuple[dict, dict]:
    """The calibration report and the plan of a served run of `requests` requests of the probed body, sent `in_flight`
    at a time (the module docstring's "The served plan"), with N2's `profile` (`occupancy`, "N2's profile"): the
    planned egress occupancy, or the served profile, at the latency budget `delta_buckets`, or else at the largest of 4
    down to 2 times the slowest response plus half a second that fits."""
    if profile not in PROFILES:
        raise ValueError(f"N2's profile is one of {PROFILES}, not {profile!r}")
    body = probed["body"]
    F = body["max_tokens"]
    if body.get("min_tokens") != F or not body.get("ignore_eos"):
        raise ValueError("a served plan's body asks for exactly its tokens (min_tokens = max_tokens, ignore_eos), "
                         f"not {body}")
    lengths = sorted({r["events"] for r in probed["responses"]})
    if not lengths or lengths[0] < 1 or lengths[-1] > F:
        raise ValueError(f"the endpoint's responses to one body of {F} tokens are {lengths} events long: each is 1 … "
                         f"{F}, an event a chunk of at least one token and at most {ar.TOKEN_BYTES} bytes of text")
    slowest = max(r["last_s"] for r in probed["responses"])
    if delta_buckets:
        return _served_plan(probed, requests, delta_buckets, name, in_flight, profile)
    most, least = (math.ceil((k * slowest * 1000 + 500) / BUCKET_MS) for k in (4, 2))
    for D in range(most, least - 1, -1):
        try:
            return _served_plan(probed, requests, D, name, in_flight, profile)
        except DoesntFit:
            pass
    raise DoesntFit(f"{in_flight} responses of {F} events don't fit in a window of {WINDOW_BUCKETS} buckets at a "
                    f"latency budget D of {least} … {most} buckets (2 to 4 × the slowest, {slowest:.3f} s, + 0.5 s): "
                    "give one (--delta-buckets), or send fewer in flight")


def _served_plan(probed: dict, requests: int, D: int, name: str, in_flight: int,
                 profile: str = "planned") -> tuple[dict, dict]:
    body, F = probed["body"], probed["body"]["max_tokens"]
    lengths = sorted({r["events"] for r in probed["responses"]})
    slowest = max(r["last_s"] for r in probed["responses"])
    names = [f"{name}.{j}" for j in range(requests)]
    groups = [names[j:j + in_flight] for j in range(0, requests, in_flight)]
    frames_in = max(request_frame_count(len(ar.live_request_bytes(s, body, None, FRAME_BYTES)), FRAME_BYTES)
                    for s in names)
    served = profile == "served"
    if served:  # the served counts take requests of a power of two of frames
        frames_in = 1 << (frames_in - 1).bit_length()
    T = WINDOW_BUCKETS
    for span in range(D + 1, T - 1):
        starts, a = [], 2
        for _ in groups:
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
                 "enqueue": math.ceil(t / i.bucket_ms), "send_ms": t} for j, (s, t) in enumerate(zip(names, sends))]
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
            "pad_to": F,
            "served": {"tokens": F, "probe_events": lengths, "delta_buckets": D, "gap_buckets": D + 1,
                       "span_buckets": span, "slowest_s": slowest, "request_frames": frames_in, "in_flight": in_flight,
                       "requests": requests}}
    return report, plan


def served_profile(plan: dict) -> ServedProfile | None:
    """N2's served profile of a plan on the served profile (`occupancy` of kind `served`): F frames a response, F_in a
    request, the plan's windows and the salt frame heading each (own 1); None on the planned profile."""
    o = plan.get("occupancy") or {}
    if o.get("kind") != "served":
        return None
    return ServedProfile(o["response_frames"], o["request_frames"], plan["windows"], o["own"])


# The certifier.

class ServedProxy(Proxy):
    """`active.Proxy`, keeping the windows whose salt it has revealed, so that `serve` stops once every one is."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.revealed: set[int] = set()

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


async def serve(plan: dict, report: dict, out: Path, *, url: str, upstream: dict, link: tuple[str, int],
                reveal: tuple[str, int], key: Path, challenger_key: LmsKey, session: bytes,
                start_in: float = 10.0, reveal_deadline: float | None = None, n2_link: str = N2_LINK,
                tamper: str | None = None, announce=_announce) -> dict:
    """One served run on the host's clock (the module docstring's `serve`), the certifier's key (the directory `key`)
    held by its `CertifierProcess`. Returns what `serve.json` holds."""
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
    certifier = CertifierProcess(key, out, report, session, n2_link)
    origin = time.monotonic() + start_in
    proxy = ServedProxy(("127.0.0.1", shim_port), e, i, setup.sync_set,
                        [ClockSync(d, Fraction(r)) for d, r in plan["syncs"]], origin=origin, publish=certifier,
                        egress_link=setup.egress_link, ingress_link=setup.ingress_link, salt=fresh_salt,
                        reveal_deadline=deadline, challenger_key=challenger_key, signer=certifier,
                        served=served_profile(plan))
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
    reveals = await serve_reveals(proxy, *reveal)
    conn.send(origin)
    record = deployment(proxy, setup, windows, session=session, n2_link=n2_link,
                        link=(link[0], listening.sockets[0].getsockname()[1]),
                        reveal=(reveal[0], reveals.sockets[0].getsockname()[1]),
                        origin_unix=time.time() + origin - time.monotonic(), url=url)
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


async def client(plan: dict, report: dict, deployment: dict, out: Path, link: tuple[str, int] | None = None) -> None:
    """The client on the link (`active_replay.Peer`), at the deployment's origin on this host's clock, connecting to
    `link`, by default the deployment's."""
    peer = ar.Peer(plan, ar.wire_params(ar.Setup(report)))
    reader, writer = await asyncio.open_connection(*(link or deployment["link"]), limit=1 << 22)
    origin = time.monotonic() + deployment["origin_unix"] - time.time()
    sending = asyncio.create_task(peer.send(writer, origin))
    await peer.read(reader, origin)
    sending.cancel()
    writer.close()
    ar.write_logs(out, peer=peer)


# The offline check.

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
    lw = LinkWindows(deployment["links"]["n2"], setup.egress_params, setup.sync_set, key, deployment["run_id"],
                     bytes.fromhex(deployment["session"]), ingress=setup.ingress_params, proof_deadline=proof_deadline,
                     occupancy=(profile.get("egress"), profile.get("ingress")), served=served)
    remote = RemoteCertifier(*deployment["reveal"], key, challenger, deployment["links"]["ingress"],
                             bytes.fromhex(deployment["run_id"]), timeout=timeout)
    stream = lw.stream(budget=windows, delta=DELTA, challenger="certifier-serve-check", coins=_no_coins, clock=clock)
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
        if served is not None:
            row["served_counts"] = lw.served_check(opening, syncs)
        draw = stream.select(registration)
        try:
            outcome = (stream.deliver(draw, [lw.session(opening, draw)])
                       or lw.outcome(stream, draw, opening, syncs))
        except ValueError as err:
            row["n2"] = f"no verdict: {err}"
            break
        row["n2"] = {"accepted": outcome.accepted, "codes": list(outcome.verdict.codes), "public": dict(outcome.public)}
        row["salt_fingerprint"] = ar.fingerprint(salts[w]) if w in salts else None
    try:
        network = lw.network_verdict(stream, windows)
    except ValueError as err:
        network = {"accepted": False, "error": str(err)}
    audited = ar.audit_configuration(plan, report, run, audit=audit)
    e, i = setup.egress_params, setup.ingress_params
    in_records = [ar.unpack_record(b, i.frame_bytes) for b in ar.read_blobs(run / "requests.records")]
    out_records = [ar.unpack_record(b, e.frame_bytes) for b in ar.read_blobs(run / "responses.records")]
    server = json.loads((run / "server.json").read_text())
    salt = ar.audit_salt(setup, salts, in_records, out_records, server, windows, audit())
    by_window: dict[int, dict] = {}
    for r in audited["rows"]:
        by_window.setdefault(r["window"], {})[r["direction"]] = {"raw": r["raw"], "committed": r["committed"]}
    for row in rows:
        row["audit"] = by_window.get(row["window"], {})
    audit_ok = bool(audited["rows"]) and all(r["raw"] is None and r["committed"] is None for r in audited["rows"])
    salt_ok = (len(salts) == windows and again is not None and again.startswith("refused")
               and salt["accepted"] and salt["block_received"] and not salt["egress_holds_salt"])
    accepted = bool(network["accepted"]) and audit_ok and salt_ok
    return {"verdict": "accepted" if accepted else "refused", "windows": windows,
            "network_verdict": network, "audit_accepted": audit_ok, "salt_accepted": salt_ok,
            "salt_again": again, "salt": salt, "profile": profile.get("kind"), "issuer_refused": refused,
            "rows": rows, "comparisons": audited["comparisons"], "timing": audited["timing"]}


def table(result: dict) -> str:
    """The check's verdicts, one line per link-window."""
    lines = [f"verdict {result['verdict']}: network_verdict {result['network_verdict'].get('accepted')}, audit "
             f"{result['audit_accepted']}, salt {result['salt_accepted']} (window 0 again: {result['salt_again']}), "
             f"N2 on the {result['profile']} profile"]
    for row in result["rows"]:
        a = row.get("audit", {})
        n2 = row.get("n2")
        n2 = ("accept" if n2["accepted"] else f"reject {n2['codes']}") if isinstance(n2, dict) else n2
        counts = f" (served counts {row['served_counts']})" if "served_counts" in row else ""
        lines.append(f"  window {row['window']}: audit egress {a.get('egress')} ingress {a.get('ingress')}; "
                     f"N2 {n2}{counts}; salt {row.get('salt')}")
    return "\n".join(lines)


# The rehearsal.

def rehearse_one(out: Path, strategy: str, *, url: str, certifier_dir: Path, challenger_dir: Path,
                 api_key_env: str | None, start_in: float) -> dict:
    """One served run of the plan in `out` as processes: `serve` (with `--tamper` unless honest), then `client`, then
    `check` while `serve` serves reveals."""
    run = out / strategy
    run.mkdir(parents=True, exist_ok=True)
    me = [sys.executable, str(HERE / "certifier_serve.py")]
    served = me + ["serve", "--plan", str(out / "plan.json"), "--calibration", str(out / "calibration.json"),
                   "--upstream", url, "--out", str(run), "--link", "127.0.0.1:0", "--reveal", "127.0.0.1:0",
                   "--key", str(certifier_dir), "--challenger-key", str(challenger_dir / "public.json"),
                   "--session", secrets.token_hex(32), "--start-in", str(start_in)]
    served += (["--tamper", strategy] if strategy != "honest" else []) + \
        (["--api-key-env", api_key_env] if api_key_env else [])
    with open(run / "serve.log", "w") as log:
        proc = subprocess.Popen(served, stdout=subprocess.PIPE, stderr=log, text=True)
        try:
            listening = json.loads(proc.stdout.readline() or "{}")
            if "listening" not in listening:
                raise SystemExit(f"{strategy}: serve exited before listening (serve.log)")
            with open(run / "client.log", "w") as clog:
                client_exit = subprocess.run(me + ["client", "--plan", str(out / "plan.json"), "--calibration",
                                                   str(out / "calibration.json"), "--deployment",
                                                   str(run / "deployment.json"), "--out", str(run)],
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


def plan_of(args, url: str, clock=time.monotonic) -> tuple[dict, dict, dict]:
    """(probe, calibration, plan) of the endpoint at `url` (`plan`'s): each request `--tokens` tokens out of a prompt of
    exactly `--prompt-tokens` tokens (`PROMPT` as text with 0), probed and planned `--in-flight` at a time."""
    model = args.model or served_model(url, args.api_key_env)
    prompt = prompt_tokens(url, model, args.prompt_tokens, args.api_key_env) if args.prompt_tokens else PROMPT
    probed = probe(url, request_body(model, args.tokens, prompt), args.probes, args.api_key_env, clock,
                   in_flight=args.in_flight, warmup=args.in_flight)
    report, plan = served_plan(probed, args.requests, args.delta_buckets, in_flight=args.in_flight,
                               profile=args.profile)
    return probed, report, plan


def rehearse(args) -> int:
    from verity.core.protocols.certifier import lean_audit

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    # built before the first run, so that the check's first verdict doesn't build it while reveals wait
    if "VERITY_NETWORK_RECORDS_EXE" not in os.environ:
        os.environ["VERITY_NETWORK_RECORDS_EXE"] = str(lean_audit.build())
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
                                   api_key_env=args.api_key_env, start_in=args.start_in)
                   for s in args.strategies.split(",")}
    finally:
        shutil.rmtree(scratch)
        if started:
            import record

            record.stop_server(started[0])
    summary = {"upstream": url, "vllm": args.vllm, "model": plan["live_model"], "lms_height": args.height,
               "served": plan["served"], "windows": plan["windows"], "profile": plan["occupancy"]["kind"],
               "runs": {s: {k: r[k] for k in ("verdict", "network_verdict", "audit_accepted", "salt_accepted",
                                              "salt_again", "exit")}
                        | {"windows": [{k: row.get(k) for k in ("window", "n2", "served_counts", "salt", "audit")}
                                       for row in r["rows"]]}
                        for s, r in results.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    expected = {"honest": "accepted"}
    ok = all(r["verdict"] == expected.get(s, "refused") for s, r in results.items())
    print(f"rehearsal {'as expected' if ok else 'NOT as expected'}: "
          f"{ {s: r['verdict'] for s, r in results.items()} }", flush=True)
    return 0 if ok else 1


def main(argv=None) -> int:
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
        q.add_argument("--requests", type=int, default=400)
        q.add_argument("--in-flight", type=int, default=4, help="requests sent together, served side by side")
        q.add_argument("--tokens", type=int, default=64, help="min_tokens and max_tokens of every request (ignore_eos)")
        q.add_argument("--prompt-tokens", type=int, default=128,
                       help="the prompt's length in the endpoint's tokens (/tokenize); 0: the text prompt as is")
        q.add_argument("--probes", type=int, default=8)
        q.add_argument("--delta-buckets", type=int,
                       help="the latency budget D (default: 4 × the slowest + 0.5 s, or the most that fits down to 2 ×)")
        q.add_argument("--profile", choices=PROFILES, default=PROFILES[0],
                       help="N2's profile: the planned egress occupancy, or the served profile (NET-54), whose egress "
                            "counts derive from the ingress head's")
    for q in (p, s, c, r):
        q.add_argument("--out", required=True, type=Path)
    for q in (s, c):
        q.add_argument("--plan", required=True, type=Path)
        q.add_argument("--calibration", required=True, type=Path)
    s.add_argument("--link", required=True, type=_address, help="HOST:PORT the client's link connects to")
    s.add_argument("--reveal", required=True, type=_address, help="HOST:PORT of the reveal endpoint")
    s.add_argument("--key", required=True, type=Path, help="the certifier's key directory (keys)")
    s.add_argument("--challenger-key", required=True, type=Path, help="the challenger's public.json")
    s.add_argument("--session", required=True, help="the certifier session, 32 bytes in hex, the challenger's draw")
    s.add_argument("--n2-link", default=N2_LINK)
    s.add_argument("--reveal-deadline", type=float, help="seconds past a window's end (default: the run's + 300)")
    for q in (s, r):
        q.add_argument("--start-in", type=float, default=10.0, help="the origin, seconds after the proxy is up")
    s.add_argument("--tamper", choices=STRATEGIES[1:],
                   help="late: the shim holds a window-0 session past every sync; extra: it writes one event more")
    c.add_argument("--deployment", required=True, type=Path)
    c.add_argument("--link", type=_address, help="HOST:PORT of the link, if not the deployment's")
    x.add_argument("--run", required=True, type=Path)
    x.add_argument("--challenger", required=True, type=Path, help="the challenger's key directory")
    x.add_argument("--expect", choices=("accepted", "refused"))
    r.add_argument("--vllm", metavar="MODEL", help="start vLLM for MODEL on this host's GPU instead of --upstream")
    r.add_argument("--vllm-cpus", help="with --vllm: its CPUs (taskset -c)")
    r.add_argument("--height", type=int, default=HEIGHT, help="the LMS keys' tree height")
    r.add_argument("--strategies", default=",".join(STRATEGIES))
    args = ap.parse_args(argv)

    if args.command == "keys":
        load_key(args.dir, args.height)
        print((args.dir / "public.json").read_text())
        return 0
    if args.command == "plan":
        probed, report, plan = plan_of(args, args.upstream)
        args.out.mkdir(parents=True, exist_ok=True)
        for name, doc in (("probe.json", probed), ("plan.json", plan), ("calibration.json", report)):
            (args.out / name).write_text(json.dumps(doc, indent=1))
        print(json.dumps(plan["served"] | {"windows": plan["windows"]}))
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
                                n2_link=args.n2_link, tamper=args.tamper))
        return 1 if "error" in log else 0
    if args.command == "client":
        ar._quiet_runtime()
        asyncio.run(client(json.loads(args.plan.read_text()), json.loads(args.calibration.read_text()),
                           json.loads(args.deployment.read_text()), args.out, args.link))
        return 0
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
