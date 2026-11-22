"""The one sender: an input dataset of the adapter's kind, read through the door (`verity_vllm.inputs.serves.door`) before anything
is sent, then played at its timing against a `verity-vllm serve-openai` endpoint.

    adapters/vllm/tools/served_run/inputs.py send --port 18190 --dataset NAME --timing at-once [--release RUN] --out FILE

Each record is a `/v1/completions` body, sent at its time from the play's start. With ``--release RUN`` (a run id in hex,
as `verity.release.release_fields` reads it) request j carries ``x-verity-release: j`` and ``x-verity-release-run: RUN``,
its place in the dataset's order, as the network attester numbers its releases; at ``at-once`` request 0 then goes out
only once the endpoint has answered every other one's head: under ``--admit release-order`` serve-openai answers a
streamed request's head once it holds it and holds each until every earlier-released one is there, so request 0 brings
them all into the engine together.

One JSON line a request goes to ``--out``: its user, release, time, prompt length, max_tokens and response. Beside it
(`timings_path`: load.jsonl's are load.timings.json) one JSON document: the dataset's name, timing and identity,
``start_unix``, the wall clock at the play's start (the server's step ends, step_trace.jsonl's ``t_end``, are on that
clock), and for each request in order its ``user``, ``release``, ``at_s``, ``send_s``, ``first_token_s`` (when the first
streamed event carrying output ids was read), ``done_s`` (when its response ended), ``tokens`` (the output ids it
streamed) and ``ok``, each time in seconds on the monotonic clock from the play's start. They are the client's
measurements; no verifier reads them.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def post(port: int, b: dict, headers: dict | None = None, times: dict | None = None,
         clock: Callable[[], float] = time.monotonic, opened: Callable[[], None] | None = None) -> dict:
    """The response to ``b``, a streamed one as its output ids and finish. With ``times``, its ``clock`` readings: ``send``
    as the request goes out, ``first_token`` as the first event carrying output ids is read (streamed only) and ``done``
    once the response is read. ``opened`` is called once the response's head is read."""
    def stamp(k: str) -> None:
        if times is not None:
            times[k] = clock()

    stamp("send")
    with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(b).encode(),
                                headers={"content-type": "application/json", **(headers or {})}), timeout=600) as r:
        if opened is not None:
            opened()
        if not b.get("stream"):
            got = json.loads(r.read())
            stamp("done")
            return got
        ids, finish = [], None
        for line in r:
            e = line.decode().strip().removeprefix("data: ")
            if e.startswith("{"):
                for c in json.loads(e).get("choices") or ():
                    if c.get("token_ids") and not ids:
                        stamp("first_token")
                    ids += c["token_ids"] if c.get("token_ids") else []
                    finish = c.get("finish_reason") or finish
    stamp("done")
    return {"token_ids": ids, "finish_reason": finish}


def timings_path(out: str | Path) -> Path:
    """The timings file beside the record ``out``."""
    out = Path(out)
    return out.with_name(out.name.removesuffix(".jsonl") + ".timings.json")


def timings(reqs: list[dict], got: list, times: list[dict], t0: float, *, release: str | None, start_unix: float,
            **said: Any) -> dict:
    """The timings document (the module docstring): each request's ``times`` less ``t0``, the play's start, and ``said``
    (the dataset's name, timing and identity)."""
    return {**said, "clock": "monotonic seconds from the release run's start", "release_run": release,
            "start_unix": start_unix,
            "requests": [{"user": q["user"], "release": j if release else None, "at_s": q["at_s"],
                          **{f"{k}_s": t[k] - t0 if k in t else None for k in ("send", "first_token", "done")},
                          "tokens": len((r or {}).get("token_ids") or ()), "ok": "error" not in (r or {})}
                         for j, (q, r, t) in enumerate(zip(reqs, got, times))]}


def play(port: int, reqs: list[dict], *, release: str | None, delays: Sequence[float] | None = None,
         clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
         sleep: Callable[[float], None] = time.sleep) -> tuple[list, list[dict], float, float]:
    """Each of ``reqs`` ({user, at_s, body}) sent at ``at_s`` (plus its ``delays``) from the start, one thread a request:
    the responses, each one's times, the start on ``clock`` and on ``wall``. With ``release`` and every ``at_s`` 0,
    request 0 waits for every other one's head (the module docstring)."""
    delays = list(delays or [0.0] * len(reqs))
    together = bool(release) and all(q["at_s"] == 0.0 for q in reqs)
    got: list = [None] * len(reqs)
    times: list[dict] = [{} for _ in reqs]
    heads = [threading.Event() for _ in reqs]
    t0, start_unix = clock(), wall()

    def go(j: int) -> None:
        if together and j == 0:
            for h in heads[1:]:
                h.wait()
        if (wait := t0 + reqs[j]["at_s"] + delays[j] - clock()) > 0:
            sleep(wait)
        headers = {"x-verity-release": str(j), "x-verity-release-run": release} if release else {}
        try:
            got[j] = post(port, reqs[j]["body"], headers, times[j], clock, heads[j].set)
        except Exception as e:  # noqa: BLE001  (recorded with the request)
            got[j] = {"error": f"{type(e).__name__}: {e}"}
        finally:
            heads[j].set()

    ts = [threading.Thread(target=go, args=(j,)) for j in range(len(reqs))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return got, times, t0, start_unix


def write(out: str | Path, reqs: list[dict], got: list, *, release: str | None, delays: Sequence[float] | None = None) -> None:
    """One JSON line a request appended to ``out`` (the module docstring)."""
    delays = list(delays or [0.0] * len(reqs))
    with open(out, "a") as fh:
        fh.writelines(json.dumps({"user": q["user"], "release": j if release else None, "at_s": q["at_s"], "delay_s": delays[j],
                                  "prompt": len(q["body"]["prompt"]), "max_tokens": q["body"]["max_tokens"], "response": r}) + "\n"
                      for j, (q, r) in enumerate(zip(reqs, got)))


def send(port: int, dataset: str, timing: str, *, release: str | None, out: str | Path,
         clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
         sleep: Callable[[float], None] = time.sleep) -> int:
    """The input dataset ``dataset`` at ``timing`` through the door, played and written (the module docstring): 0 when every
    request is answered, 1 when one is not, 3 when the door or the dataset refuses, before anything is sent."""
    import input_datasets

    from verity_vllm.inputs import serves
    try:
        ds = input_datasets.open(dataset)
        rule, parameters = serves.timing(timing)
        timed = ds.timed(rule, **parameters)
        pairs = list(timed)
        bodies = serves.door(ds.kind, (b for _, b in pairs))
        reqs = [{"user": b["user"], "at_s": t, "body": b} for (t, _), b in zip(pairs, bodies)]
    except (serves.Refused, input_datasets.Refused) as e:
        print("[send] REFUSED {}: {}".format(*serves.refusal(e)), flush=True)
        return 3
    got, times, t0, start_unix = play(port, reqs, release=release, clock=clock, wall=wall, sleep=sleep)
    write(out, reqs, got, release=release)
    timings_path(out).write_text(json.dumps(timings(reqs, got, times, t0, release=release, start_unix=start_unix,
                                                    dataset=dataset, timing=timing, identity=timed.identity), indent=1))
    ok = sum("error" not in (r or {}) for r in got)
    print(f"[send] {dataset} at {timing}: {ok}/{len(got)} ok, prompts {min(len(q['body']['prompt']) for q in reqs)}…"
          f"{max(len(q['body']['prompt']) for q in reqs)} ids; identity {timed.identity}", flush=True)
    return 0 if ok == len(got) else 1
