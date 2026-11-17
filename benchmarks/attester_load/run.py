"""`verity bench attester_load`: what the network attester costs the requests it carries.

    verity bench attester_load [--class w63k4] [--reps 5] [--requests 12] [--load 0.8] [--json] [--out DIR]
    verity bench attester_load --vllm-python VENV/bin/python [--model unsloth/Llama-3.2-1B]   the model arm too

Each arm sends the same requests at the same evenly spaced load to one endpoint, through the network attester and
straight to it, alternating, and measures each request's time to the first and to the last byte of its response. The
network attester runs as deployed (`network_attester_serve.py serve`, under a key made for the run) at a scan class's
schedule (`scan_zk.py schedule --class`), on this machine. The schedule delays every response on purpose: a request's
first egress cell leaves D buckets after its admission, and its last D + ⌊k/r⌋ buckets after it, so most of each added
time is a parameter, which the result states beside it, and so is the capacity, lanes / ((D + 1 + ⌊k/r⌋)·τ) requests a
second for responses of k ids. The stub arm runs on a CPU against `network_attester_serve.py fake`, in the id form; the
model arm runs only where `--vllm-python` names a Python with vLLM, on a card, against vLLM's OpenAI server, in the bytes
form, since that server answers in its own framing: the schedule and its cells' buckets are the id form's.

It writes its three numbers only when its guard holds: a delay planted in the network attester's release
(`--plant-ms` on its pacer) shows in the added time to the first byte, and the direct arm against itself reads near zero.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import secrets
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SERVE = ROOT / "verity/protocols/resource_accounting/network/attester/network_attester_serve.py"
SCAN_ZK = ROOT / "verity/protocols/resource_accounting/network/prover/scan_zk.py"
FIRST, LAST, TICK = ("network-accounting/attester-added-first-byte", "network-accounting/attester-added-last-byte",
                     "network-accounting/attester-tick-lateness")
#: a cell's header bytes (`link_frame.HEADER`): an id event fills the rest of its cell, and the id form fails any other size
CELL_HEAD = 9
#: the most the direct arm may read against itself, and the most a planted delay's reading may miss it by, in ms
SELF_TOL_MS, PLANT_TOL_MS = 5.0, 10.0
STUB = "stub-cpu"
#: the session's log as `network_attester_serve.py serve` names it: each connection's outcome, the refusals, the ticks'
#: lateness
SESSION_LOG = "warden.json"


def schedule(scan_class: str, overrides: dict[str, int]) -> dict[str, int]:
    """The scan class's schedule, as `scan_zk.py schedule --class` prints its flags, with `overrides` in place."""
    words = subprocess.run([sys.executable, str(SCAN_ZK), "schedule", "--class", scan_class], check=True,
                           capture_output=True, text=True).stdout.split()
    got = {words[i][2:].replace("-", "_"): int(words[i + 1]) for i in range(0, len(words), 2)}
    return got | {k: v for k, v in overrides.items() if v is not None}


def stated(s: dict[str, int], tokens: int, paced_ms: float) -> dict[str, float]:
    """What the schedule fixes for responses of `tokens` ids: a connection holds its lane from its admission's bucket
    through its last egress cell's, D + ⌊k/r⌋, so a lane carries one every D + 1 + ⌊k/r⌋ buckets."""
    last_bucket = s["delay"] + tokens // s["slots"]
    hold = last_bucket + 1
    return {"capacity_per_s": s["lanes"] / (hold * s["bucket_ms"] / 1000), "lane_hold_buckets": hold,
            "scheduled_first_ms": s["delay"] * s["bucket_ms"] + paced_ms,
            "scheduled_last_bucket": last_bucket, "scheduled_last_ms": last_bucket * s["bucket_ms"] + paced_ms}


def traffic(n: int, rate: float, tokens: int, prompt_ids: int, tag: str, *, egress: str, model: str,
            start_s: float = 0.2) -> list[dict]:
    """`n` requests `1/rate` seconds apart, each `tokens` ids of a prompt of `prompt_ids` ids, in the client's form:
    an id request (`id_form`'s fields) in the id form, else a streamed `/v1/completions` body. Each names its `user`,
    which is how the endpoint's side tells them apart."""
    rng = random.Random(tag)
    out = []
    for i in range(n):
        prompt = [rng.randrange(128_000) for _ in range(prompt_ids)]
        fields = {"model": model, "user": f"{tag}-{i}", "prompt": prompt, "max_tokens": tokens, "temperature": 0.0,
                  "top_p": 1.0}
        item = {"ids": fields | {"seed": i}} if egress == "id" else {"body": fields | {"stream": True, "ignore_eos": True}}
        out.append({"at": round(start_s + i / rate, 4)} | item)
    return out


def user_of(item: dict) -> str:
    return (item.get("ids") or item["body"])["user"]


class Tap:
    """A loopback relay in front of the endpoint that both arms reach: it keeps each request's bytes by its `user`, so
    the direct arm replays exactly what the network attester released to the endpoint."""

    def __init__(self, upstream: tuple[str, int]) -> None:
        self.upstream, self.requests, self.cut = upstream, {}, []
        self.loop = asyncio.new_event_loop()
        ready = threading.Event()
        threading.Thread(target=self._run, args=(ready,), daemon=True).start()
        ready.wait()

    def _run(self, ready: threading.Event) -> None:
        asyncio.set_event_loop(self.loop)
        self.server = self.loop.run_until_complete(asyncio.start_server(self._one, "127.0.0.1", 0))
        self.address = self.server.sockets[0].getsockname()[:2]
        ready.set()
        self.loop.run_forever()

    async def _one(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            m = re.search(rb"(?im)^content-length:\s*(\d+)", head)
            request = head + (await reader.readexactly(int(m.group(1))) if m else b"")
            user = re.search(rb'"user":\s*"([^"]+)"', request)
            if user:
                self.requests[user.group(1).decode()] = request
            up_r, up_w = await asyncio.open_connection(*self.upstream)
            up_w.write(request)
            await up_w.drain()
            while chunk := await up_r.read(1 << 16):
                writer.write(chunk)
                await writer.drain()
            up_w.close()
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError) as e:
            self.cut.append(type(e).__name__)
        finally:
            writer.close()


async def _direct(address: tuple[str, int], items: list[tuple[float, bytes]]) -> list[dict]:
    t0 = time.monotonic()

    async def one(at: float, request: bytes) -> dict:
        await asyncio.sleep(max(0.0, t0 + at - time.monotonic()))
        sent = time.monotonic()
        reader, writer = await asyncio.open_connection(*address)
        writer.write(request)
        await writer.drain()
        first, n = None, 0
        while chunk := await reader.read(1 << 16):
            first = first or time.monotonic()
            n += len(chunk)
        last = time.monotonic()
        writer.close()
        return {"first_ms": None if first is None else (first - sent) * 1000, "last_ms": (last - sent) * 1000,
                "bytes": n}

    return await asyncio.gather(*(one(at, r) for at, r in items))


def direct(tap: Tap, load: list[dict]) -> dict[str, dict]:
    """The direct arm: each request's bytes as the endpoint got them through the network attester, sent at the same
    times straight to the endpoint (through the tap, as the other arm), each first and last byte timed from its send."""
    items = [(item["at"], tap.requests[user_of(item)]) for item in load if user_of(item) in tap.requests]
    got = asyncio.run(_direct(tap.address, items))
    return dict(zip([user_of(i) for i in load if user_of(i) in tap.requests], got))


def _cmd(*args: str, out: Path | None = None) -> subprocess.CompletedProcess:
    log = open(out, "w") if out else subprocess.DEVNULL
    try:
        return subprocess.run([sys.executable, str(SERVE), *args], stdout=log, stderr=subprocess.STDOUT, check=False)
    finally:
        if out:
            log.close()


def attester(work: Path, s: dict[str, int], load: list[dict], upstream: tuple[str, int], *, egress: str,
             paced_ms: float, slack_ms: float, start_in: float) -> dict:
    """One session of the network attester in front of `upstream`, as deployed, over `load`: each request's first and
    last byte from its send, the check's verdict, the refusals and the ticks' lateness."""
    work.mkdir(parents=True)
    (work / "traffic.json").write_text(json.dumps(load))
    key = work.with_name(work.name + ".key")
    if _cmd("keys", "--dir", str(key)).returncode:
        raise SystemExit(f"the network attester's key wasn't made in {key}")
    window_s = s["window"] * s["bucket_ms"] / 1000
    tail = 2 * (s["delay"] + 2 + s["ingress_buckets"]) * s["bucket_ms"] / 1000 + 1
    windows = max(1, math.ceil((load[-1]["at"] + tail) / window_s))
    flags = [x for k, v in s.items() for x in (f"--{k.replace('_', '-')}", str(v))]
    with open(work / "serve.log", "w") as log:
        serving = subprocess.Popen([sys.executable, str(SERVE), "serve", "--upstream", "%s:%d" % upstream,
                                    "--listen", "127.0.0.1:0", "--out", str(work), "--key", str(key),
                                    "--session", secrets.token_hex(32), "--windows", str(windows),
                                    "--start-in", str(start_in), "--slack", str(slack_ms / 1000),
                                    "--paced-ms", str(paced_ms), "--egress", egress, *flags],
                                   stdout=subprocess.PIPE, stderr=log, text=True)
        try:
            if "listening" not in json.loads(serving.stdout.readline() or "{}"):
                serving.wait()
                said = (work / "serve.log").read_text().strip().splitlines()[-1:]
                raise SystemExit(f"the network attester exited before listening: {said}")
            _cmd("client", "--deployment", str(work / "deployment.json"), "--traffic", str(work / "traffic.json"),
                 "--out", str(work), out=work / "client.log")
            serving.wait()
        finally:
            if serving.poll() is None:
                serving.terminate()
                serving.wait()
    _cmd("check", "--run", str(work), out=work / "check.log")
    check = json.loads((work / "check.json").read_text()) if (work / "check.json").exists() else {"accepted": False}
    log = json.loads((work / SESSION_LOG).read_text())
    per: dict[str, dict] = {}
    for c in json.loads((work / "client.json").read_text())["connections"]:
        kinds, at = c.get("kinds", ""), c.get("arrivals_ms", [])
        sent = c.get("sent_ms", 0.0)
        first = next((t for t, k in zip(at, kinds) if k in "de"), None)
        last = next((t for t, k in zip(at, kinds) if k == "e"), None)
        per[user_of(load[c["index"]])] = {"outcome": c.get("outcome"),
                                          "first_ms": None if first is None else first - sent,
                                          "last_ms": None if last is None else last - sent}
    return {"requests": per, "accepted": bool(check.get("accepted")), "refused": len(log.get("refused") or []),
            "turned_away": len(log.get("turned_away") or []), "lateness_ms": log.get("lateness_ms") or {}}


def added(att: dict[str, dict], dire: dict[str, dict]) -> dict[str, Any]:
    """The mean added time to the first and last byte over the requests both arms carried to their end."""
    both = [u for u, a in att.items() if a["outcome"] == "end" and a["first_ms"] is not None
            and dire.get(u, {}).get("first_ms") is not None]
    if not both:
        return {"n": 0, "first_ms": None, "last_ms": None}
    mean = lambda arm, k: statistics.fmean(arm[u][k] for u in both)  # noqa: E731
    return {"n": len(both), "first_ms": mean(att, "first_ms") - mean(dire, "first_ms"),
            "last_ms": mean(att, "last_ms") - mean(dire, "last_ms"),
            "through_first_ms": mean(att, "first_ms"), "direct_first_ms": mean(dire, "first_ms"),
            "through_last_ms": mean(att, "last_ms"), "direct_last_ms": mean(dire, "last_ms")}


def self_read(a: dict[str, dict], b: dict[str, dict]) -> float | None:
    both = [u for u in a if a[u]["first_ms"] is not None and b.get(u, {}).get("first_ms") is not None]
    return statistics.fmean(b[u]["first_ms"] - a[u]["first_ms"] for u in both) if both else None


#: Student's t at 97.5% by degrees of freedom; past the table, the normal's
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         15: 2.131, 20: 2.086, 30: 2.042}


def mean_ci(xs: Sequence[float]) -> tuple[float, list[float] | None]:
    """The mean over repetitions and its 95% t-interval (none from one)."""
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, None
    df = len(xs) - 1
    t = _T975.get(df) or next((v for k, v in sorted(_T975.items()) if k >= df), 1.96)
    h = t * statistics.stdev(xs) / math.sqrt(len(xs))
    return m, [m - h, m + h]


def guard(plant_ms: float, planted_shift_ms: float | None, self_ms: float | None) -> list[str]:
    """Why the arm's numbers don't stand, if they don't: the planted delay must show in the added time to the first
    byte, and the direct arm read against itself must be near zero."""
    out = []
    if planted_shift_ms is None or abs(planted_shift_ms - plant_ms) > PLANT_TOL_MS:
        out.append(f"a delay of {plant_ms:g} ms planted in the network attester's release moved the added time to the "
                   f"first byte by {planted_shift_ms} ms, not within {PLANT_TOL_MS:g} ms of it")
    if self_ms is None or abs(self_ms) > SELF_TOL_MS:
        out.append(f"the direct arm against itself reads {self_ms} ms, not within {SELF_TOL_MS:g} ms of zero")
    return out


def arm(name: str, upstream: tuple[str, int], s: dict[str, int], a: argparse.Namespace, work: Path, *, egress: str,
        model: str, fields: dict[str, Any]) -> dict:
    """One arm end to end: the repetitions at the stated load, the planted delay, the direct arm against itself, and the
    session at the full stated capacity whose ticks are read."""
    tokens = s["egress_cells"] - 1 if a.tokens is None else a.tokens
    st = stated(s, tokens, a.paced_ms)
    rate = a.load * st["capacity_per_s"]
    tap = Tap(upstream)
    kw = dict(egress=egress, slack_ms=a.slack_ms, start_in=a.start_in)

    def load(tag: str, at_rate: float = rate) -> list[dict]:
        return traffic(a.requests, at_rate, tokens, a.prompt_ids, tag, egress=egress, model=model)

    reps, last_load, last_direct = [], None, None
    for r in range(a.reps):
        last_load = load(f"{name}-rep{r}")
        att = attester(work / f"{name}-rep{r}", s, last_load, tap.address, paced_ms=a.paced_ms, **kw)
        last_direct = direct(tap, last_load)
        reps.append(added(att["requests"], last_direct) | {k: att[k] for k in ("accepted", "refused", "turned_away")})
    again = direct(tap, last_load)
    planted_load = load(f"{name}-planted")
    planted = attester(work / f"{name}-planted", s, planted_load, tap.address, paced_ms=a.paced_ms + a.plant_ms, **kw)
    planted_added = added(planted["requests"], direct(tap, planted_load))
    full = attester(work / f"{name}-full", s, load(f"{name}-full", st["capacity_per_s"]), tap.address,
                    paced_ms=a.paced_ms, **kw)
    firsts = [r["first_ms"] for r in reps if r["first_ms"] is not None]
    lasts = [r["last_ms"] for r in reps if r["last_ms"] is not None]
    base = statistics.fmean(firsts) if firsts else None
    shift = None if base is None or planted_added["first_ms"] is None else planted_added["first_ms"] - base
    self_ms = self_read(last_direct, again)
    failed = guard(a.plant_ms, shift, self_ms) + (["no request ended in both arms"] if not firsts else [])
    first, first_ci = mean_ci(firsts) if firsts else (None, None)
    last, last_ci = mean_ci(lasts) if lasts else (None, None)
    arms = {k: statistics.fmean(r[k] for r in reps if r.get(k) is not None) if firsts else None
            for k in ("through_first_ms", "direct_first_ms", "through_last_ms", "direct_last_ms")}
    return {"member": name, "egress": egress, "model": model, "class": a.scan_class, "schedule": s, "tokens": tokens,
            "prompt_ids": a.prompt_ids, "paced_ms": a.paced_ms, "load_per_s": rate, "load_share": a.load,
            "requests": a.requests, "reps": a.reps, **st, **fields, **arms,
            "added_first_ms": first, "added_first_ci95": first_ci, "added_last_ms": last, "added_last_ci95": last_ci,
            "tick_lateness_ms": full["lateness_ms"], "full": {k: full[k] for k in ("accepted", "refused", "turned_away")},
            "relay_cut": len(tap.cut), "repetitions": reps,
            "guard": {"plant_ms": a.plant_ms, "planted_shift_ms": shift, "direct_against_itself_ms": self_ms,
                      "failed": failed}}


def stub(a: argparse.Namespace, s: dict[str, int], work: Path) -> dict:
    fake = subprocess.Popen([sys.executable, str(SERVE), "fake", "--listen", "127.0.0.1:0", "--first-ms",
                             str(a.first_ms), "--token-ms", str(a.token_ms), "--id-payload",
                             str(s["cell_bytes"] - CELL_HEAD)], stdout=subprocess.PIPE, text=True)
    try:
        upstream = tuple(json.loads(fake.stdout.readline())["listening"])
        return arm(STUB, upstream, s, a, work, egress="id", model="fake",
                   fields={"card": "none", "endpoint": f"network_attester_serve.py fake, its first id {a.first_ms:g} ms "
                                                       "after a request, every id at once"})
    finally:
        fake.terminate()
        fake.wait()


def card() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    return (r.stdout.split("\n")[0].strip() or None) if r.returncode == 0 else None


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", text.lower().replace("nvidia", "")).strip("-")


def model_arm(a: argparse.Namespace, s: dict[str, int], work: Path) -> dict:
    """The model arm: vLLM's OpenAI server of `--model` on this machine's card, the network attester in the bytes form."""
    gpu = card()
    port = 18000 + secrets.randbelow(2000)
    with open(work / "vllm.log", "w") as log:
        server = subprocess.Popen([a.vllm_python, "-m", "vllm.entrypoints.openai.api_server", "--model", a.model,
                                   "--host", "127.0.0.1", "--port", str(port), *a.server_arg],
                                  stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + a.model_ready_s
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5).read()
                break
            except OSError:
                if server.poll() is not None or time.monotonic() > deadline:
                    raise SystemExit(f"vLLM didn't serve {a.model} ({work / 'vllm.log'})") from None
                time.sleep(2)
        name = f"{slug(a.model.rsplit('/', 1)[-1])}-{slug(gpu or 'none')}"
        return arm(name, ("127.0.0.1", port), s, a, work, egress="bytes", model=a.model,
                   fields={"card": gpu, "endpoint": f"vLLM's OpenAI server of {a.model}"})
    finally:
        server.terminate()
        server.wait()


def render(doc: dict) -> str:
    lines = [f"# What the network attester costs the requests it carries, at scan class {doc['class']}", ""]
    for r in doc["arms"]:
        ci = lambda c: "" if c is None else f" [{c[0]:.1f}, {c[1]:.1f}]"  # noqa: E731
        g = r["guard"]
        lines += [f"## {r['member']} ({r['egress']} form, {r['endpoint']})",
                  f"schedule {r['schedule']}",
                  f"stated capacity {r['capacity_per_s']:.3f} requests a second (a lane held {r['lane_hold_buckets']} "
                  f"buckets for {r['tokens']} ids); load {r['load_per_s']:.3f} a second ({r['load_share']:g} of it), "
                  f"{r['requests']} requests × {r['reps']} repetitions",
                  f"added time to the first byte {r['added_first_ms']:.1f} ms{ci(r['added_first_ci95'])}: "
                  f"{r['through_first_ms']:.1f} ms through the network attester, whose schedule holds every first "
                  f"byte {r['scheduled_first_ms']:.0f} ms (D·τ and the pacer) past its admission, against "
                  f"{r['direct_first_ms']:.1f} ms straight to the endpoint",
                  f"added time to the last byte {r['added_last_ms']:.1f} ms{ci(r['added_last_ci95'])}: "
                  f"{r['through_last_ms']:.1f} ms through it, a full response's last cell scheduled at bucket "
                  f"{r['scheduled_last_bucket']} ({r['scheduled_last_ms']:.0f} ms), against {r['direct_last_ms']:.1f} ms"
                  if r["added_last_ms"] is not None else "added time to the last byte: none ended",
                  f"tick lateness at the full stated capacity {r['tick_lateness_ms']} ms; check {r['full']}",
                  f"guard: planted {g['plant_ms']:g} ms read {g['planted_shift_ms']} ms; direct against itself "
                  f"{g['direct_against_itself_ms']} ms; " + ("holds" if not g["failed"] else "FAILED: " + "; ".join(g["failed"])),
                  ""]
    lines += [doc["model_arm"], ""]
    return "\n".join(lines)


def write(doc: dict, out: Path) -> int:
    """The arms whose guard holds, as the three metrics through the harness kit's writer: under a research run its
    result, else `out/result.json`."""
    from verity_bench import kit

    w = kit.writer(str(out))
    n = 0
    for r in doc["arms"]:
        if r["guard"]["failed"]:
            continue
        extra = {k: v for k, v in r.items() if k not in ("repetitions", "member")}
        member = r["member"]
        w.measurement(FIRST, r["added_first_ms"], "ms", r["added_first_ci95"], mark="measured", member=member, extra=extra)
        if r["added_last_ms"] is not None:
            w.measurement(LAST, r["added_last_ms"], "ms", r["added_last_ci95"], mark="measured", member=member,
                          extra=extra)
        if r["tick_lateness_ms"].get("p99") is not None:
            w.measurement(TICK, r["tick_lateness_ms"]["p99"], "ms", mark="measured", member=member, extra=extra)
        n += 1
    return n


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="verity bench attester_load", description=__doc__.splitlines()[0])
    ap.add_argument("--class", dest="scan_class", default="w63k4", help="the scan class whose schedule the network "
                    "attester runs (scan_zk.py schedule --class; default w63k4, the onsite application's)")
    for name in ("cell-bytes", "slots", "bucket-ms", "delay", "lanes", "queue", "window", "ingress-cells",
                 "ingress-buckets", "egress-cells", "every"):
        ap.add_argument(f"--{name}", type=int, help="this parameter of the schedule in place of the class's")
    ap.add_argument("--reps", type=int, default=5, help="repetitions of each arm at the stated load")
    ap.add_argument("--requests", type=int, default=12, help="requests a session")
    ap.add_argument("--load", type=float, default=0.8, help="the load as a share of the stated capacity")
    ap.add_argument("--tokens", type=int, help="ids a response asks for (default: the egress cells less the trailer's)")
    ap.add_argument("--prompt-ids", type=int, default=32, help="ids a prompt holds")
    ap.add_argument("--paced-ms", type=float, default=50.0, help="the network attester's pacer: each bucket released "
                    "this many ms after its boundary")
    ap.add_argument("--plant-ms", type=float, default=30.0, help="the guard's delay, added to the pacer in one session")
    ap.add_argument("--slack-ms", type=float, default=25.0, help="ms a tick may run late before it faults")
    ap.add_argument("--start-in", type=float, default=1.0, help="seconds from a session's start to its origin")
    ap.add_argument("--first-ms", type=float, default=50.0, help="the stub endpoint's ms to its answer")
    ap.add_argument("--token-ms", type=float, default=20.0, help="the stub endpoint's ms an id")
    ap.add_argument("--vllm-python", help="a Python with vLLM, to run the model arm on this machine's card")
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B", help="the model arm's model")
    ap.add_argument("--server-arg", action="append", default=[], help="an argument for vLLM's server (repeatable)")
    ap.add_argument("--model-ready-s", type=float, default=900.0)
    ap.add_argument("--json", action="store_true", help="the document as JSON in place of the print")
    ap.add_argument("--out", type=Path, help="also write the result through the harness kit's writer for DIR (under a "
                    "research run it writes the run's result whatever DIR is)")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = parser().parse_args(argv)
    if not a.slack_ms < a.paced_ms < a.paced_ms + a.plant_ms:
        raise SystemExit("--paced-ms is above --slack-ms, and --plant-ms above 0")
    s = schedule(a.scan_class, {k: getattr(a, k) for k in ("cell_bytes", "slots", "bucket_ms", "delay", "lanes",
                                                           "queue", "window", "ingress_cells", "ingress_buckets",
                                                           "egress_cells", "every")})
    if a.paced_ms + a.plant_ms >= s["bucket_ms"]:
        raise SystemExit(f"--paced-ms and --plant-ms ({a.paced_ms + a.plant_ms:g} ms) are within a bucket "
                         f"({s['bucket_ms']} ms)")
    work = Path(tempfile.mkdtemp(prefix="attester-load-"))
    try:
        arms = [stub(a, s, work)]
        gpu = card()
        if a.vllm_python and gpu:
            arms.append(model_arm(a, s, work))
            said = f"The model arm ran: {a.model} on {gpu}."
        else:
            said = ("The model arm was not run: " + ("no card" if not gpu else "no --vllm-python") + ".")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    doc = {"class": a.scan_class, "arms": arms, "model_arm": said}
    print(json.dumps(doc, indent=1) if a.json else render(doc), flush=True)
    failed = [r["member"] for r in arms if r["guard"]["failed"]]
    from verity_bench import settings

    if a.out is not None or settings.is_set("RESEARCH_RUN_DIR"):
        n = write(doc, a.out or Path("out"))
        print(f"wrote {n} arm(s)' measurements" + (f"; none for {failed}, whose guard failed" if failed else ""),
              file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
