"""served-cheats: a cheating server against a served record, four switches, each refused or charged.

One command runs each of four cheats against a served record and prints, per switch, what the audit was expected to
do and what it did. The cheats, each a switch:

- ``tokens``: the server puts ids of its own choosing in a response (the shim's ``token`` tamper, `active_replay`).
  The frame counts stay honest's, so only a check of the frames' tokens against the committed sampled ids (the I/O
  tie) refuses it; until that tie lands it is **not caught**, and the switch says so.
- ``length``: the server ends a response early or late (the ``drop`` and ``extra`` tampers), which moves its event
  count. N2's counts (the stop check) refuse it.
- ``timing``: the server moves the network certifier's release times within a bucket (the jitter channel). The number
  of bits an interval it carries is a measurement of the host, produced by ``benchmarks/certifier_timing`` (the
  certifier alone) or ``tick_load.py --secret-bit`` (a whole rehearsal); the switch states the budget the measurement
  is held under (`SECONDS_A_BIT`). On the paced certifier the jitter term is 0 (D_j = 1).
- ``failures``: the server fails connections on purpose (a response of no tokens). On N2's served profile the counts
  fix nothing, so the run is accepted and each failed connection is charged as its own egress, counted in the table.

``--stand-in`` (the dry run) builds every record from an in-process endpoint that streams GLM-5.3's tokens on an
injected clock, so it needs no GPU and no served model; the served run points the same four switches at a kept served
GLM-5.3 record (`served_session.py`, `certifier_serve.py rehearse`) on a team machine. Its report is read from the
run's published Attempt, never from the tree.

    python served_cheats.py --stand-in [--profile planned|served] [--switch all|tokens|length|timing|failures]
        [--out DIR]
"""
from __future__ import annotations

import argparse
import asyncio
import codecs
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import active_replay as ar  # noqa: E402
import certifier_serve as cs  # noqa: E402
from verity.protocols.certifier.verifier import active  # noqa: E402
from verity.protocols.certifier.verifier.certificate import Issuer  # noqa: E402

SESSION = bytes(range(32))
REQUESTS, MERGED_TOKENS, PROMPT_TOKENS = 8, 64, 128
#: the two ends of the certifier's timing budget, seconds a bit (`benchmarks/certifier_timing/channel.py`'s
#: `SECONDS_A_BIT`): at an interval of tau seconds the certifier's share is at most tau/600 to tau/240 bits an interval
SECONDS_A_BIT = (240.0, 600.0)
SWITCHES = ("tokens", "length", "timing", "failures")

#: Tokens of GLM-5.3's tokenizer (zai-org/GLM-5.3 at aca966e4): words and punctuation of two and three bytes a
#: character, and tokens that split a character's UTF-8, as the stand-in streams them (`test_certifier_serve.py`)
GLM_TOKENS = ("数据".encode(), " 计算".encode(), "，".encode(),
              b"\xf0\x9f", b"\xa6", b"\x99",  # 🦙, split across three tokens
              "模型".encode(),
              b"\xe9\xbe", b"\x98",  # 龘, split across two
              "。".encode(),
              b" GL", b"M")


def glm_token(k: int) -> bytes:
    return GLM_TOKENS[k % len(GLM_TOKENS)]


class StandIn:
    """An OpenAI streaming completions endpoint that serves `group` requests side by side, as vLLM serves requests in
    flight, streaming GLM-5.3's tokens. ``/v1/models`` lists one model and ``/tokenize`` makes 200 tokens of any
    prompt. For the requests `merges` picks, the last two tokens share one chunk, as vLLM v1's collector merges deltas
    the API server hasn't read yet. An arrival in `fail` streams no token and closes at once (the server failing the
    connection on purpose)."""

    def __init__(self, group: int, *, merges=lambda r: False, fail=frozenset()) -> None:
        self.group, self.arrived, self.merges, self.fail = group, 0, merges, frozenset(fail)
        self.turns: dict = defaultdict(asyncio.Event)

    async def __call__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        n = re.search(rb"content-length: (\d+)", head, re.I)
        body = json.loads(await reader.readexactly(int(n.group(1)))) if n else None
        if not head.startswith(b"POST /v1/completions "):
            doc = ({"data": [{"id": "stub-model"}]} if head.startswith(b"GET /v1/models ")
                   else {"tokens": list(range(200)), "count": 200, "prompt": body["prompt"], "model": body["model"]})
            data = json.dumps(doc).encode()
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: %d\r\n\r\n%s"
                         % (len(data), data))
            await writer.drain()
            writer.close()
            return
        me, self.arrived = self.arrived, self.arrived + 1
        g, i = divmod(me, self.group)
        if i == self.group - 1:
            self.turns[(g, 0, 0)].set()
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
        text, n_tokens, detokenize = "", 0 if me in self.fail else body["max_tokens"], codecs.getincrementaldecoder("utf-8")()
        for k in range(n_tokens):
            await self.turns[(g, k, i)].wait()
            text += detokenize.decode(glm_token(k))
            if not (k == n_tokens - 2 and self.merges(me)):
                chunk = json.dumps({"choices": [{"text": text}]}, ensure_ascii=False, separators=(",", ":"))
                p, text = b"data: %s\n\n" % chunk.encode(), ""
                writer.write(b"%x\r\n%s\r\n" % (len(p), p))
                await writer.drain()
            self.turns[(g, k, i + 1) if i + 1 < self.group else (g, k + 1, 0)].set()
        writer.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(b"data: [DONE]\n\n"), b"data: [DONE]\n\n"))
        await writer.drain()
        writer.close()


def plan_args(profile: str) -> argparse.Namespace:
    """`certifier_serve.plan_of`'s arguments for the stand-in: four in flight, 64 tokens, D = 30 buckets."""
    return argparse.Namespace(model=None, api_key_env=None, prompt_tokens=PROMPT_TOKENS, tokens=MERGED_TOKENS,
                              response_frames=None, probes=4, in_flight=4, requests=REQUESTS, delta_buckets=30,
                              profile=profile, temperature=0.0, top_p=1.0)


def run_one(run: Path, plan: dict, report: dict, url: str, upstream: tuple, certifier, challenger,
            strategy: str) -> dict:
    """One served run on the injected clock and its offline check, the shim tampered for `strategy` (honest runs the
    plan as is): the certifier (`active.Proxy`, salt on) in front of the endpoint at `upstream`, then `check` while a
    reveal endpoint serves on the run's loop. Lifts `test_certifier_serve._run`."""
    setup = ar.Setup(report)
    run.mkdir(parents=True, exist_ok=True)
    (run / "plan.json").write_text(json.dumps(plan))
    (run / "calibration.json").write_text(json.dumps(report))
    shim = plan if strategy == "honest" else ar.adversary_plan(plan, setup, strategy)
    out: dict = {}

    async def after(proxies, clock):
        reveals = await active.serve_reveals(proxies[0], "127.0.0.1", 0)
        record = cs.deployment(proxies[0], setup, plan["windows"], session=SESSION, n2_link=cs.N2_LINK,
                               link=("127.0.0.1", 0), reveal=reveals.sockets[0].getsockname()[:2], origin_unix=0.0,
                               url=url)
        (run / "deployment.json").write_text(json.dumps(record))
        out["check"] = await asyncio.to_thread(cs.check, run, challenger, clock=lambda: 0.0, timeout=None)
        reveals.close()

    publish = cs.CertifyingPublisher(run, setup, Issuer(certifier, SESSION), cs.N2_LINK, threaded=False)
    ar.run_virtual(shim, report, run, upstream, salt=True, publish=publish, after=after,
                   proxy_args={"challenger_key": challenger.public_key(), "signer": certifier,
                               "served": cs.served_profile(plan)})
    server = json.loads((run / "server.json").read_text())
    out["events"], out["run"] = server["events"], run
    return out


def _codes(check: dict) -> list[str]:
    """The N2 refusal codes the check names, over its windows."""
    out: list[str] = []
    for row in check.get("rows", ()):
        n2 = row.get("n2")
        if isinstance(n2, dict):
            out += n2.get("codes", [])
    return sorted(set(out))


def _requests_run(run: dict) -> int | None:
    """How many requests the run served (the endpoint's connections), or None when the log can't be read."""
    try:
        return int(json.loads((run["run"] / "server.json").read_text()).get("requests", -1))
    except (OSError, ValueError, KeyError, TypeError):
        return None


async def _serve(tmp: Path, switches: tuple[str, ...], profile: str) -> dict:
    """Start the stand-in, plan it, and run each switch's runs on the injected clock; returns one row per switch and a
    baseline honest row. A switch that errors reports its error and leaves the others untouched."""
    challenger = cs.signer(tmp / "keys" / "challenger", 6)
    rows: dict = {}

    async def one(where: Path, strategy: str, up: tuple, u: str, plan: dict, report: dict) -> dict:
        certifier = cs.signer(tmp / "keys" / where.name, 6)
        try:
            return await asyncio.to_thread(run_one, where, plan, report, u, up, certifier, challenger, strategy)
        finally:
            certifier.close()

    try:
        honest_ep = StandIn(4, merges=lambda r: r % 2 == 1)
        server = await asyncio.start_server(honest_ep, "127.0.0.1", 0)
        upstream = server.sockets[0].getsockname()[:2]
        url = "http://%s:%d" % upstream
        checks: dict = {}
        try:
            _, report, plan = await asyncio.to_thread(cs.plan_of, plan_args(profile), url, lambda: 0.0)
            need = ["honest"] + (["token"] if "tokens" in switches else []) + \
                   (["extra", "drop"] if "length" in switches else [])
            for s in need:
                try:
                    checks[s] = (await one(tmp / s, s, upstream, url, plan, report))["check"]
                except Exception as err:  # a run that can't complete is a reported row, not a crashed harness
                    checks[s] = {"error": repr(err)}
            rows["baseline"] = _baseline_row(checks["honest"])
            if "timing" in switches:
                rows["timing"] = _timing_row(report)
            if "tokens" in switches:
                rows["tokens"] = _tokens_row(checks["token"])
            if "length" in switches:
                rows["length"] = _length_row(checks["extra"], checks["drop"])
        finally:
            server.close()
        if "failures" in switches:
            failing = StandIn(1, fail={1, 4})
            fserver = await asyncio.start_server(failing, "127.0.0.1", 0)
            fup = fserver.sockets[0].getsockname()[:2]
            try:
                frun = await one(tmp / "failures", "honest", fup, "http://%s:%d" % fup, plan, report)
                rows["failures"] = _failures_row(frun["check"], _requests_run(frun))
            except Exception as err:
                rows["failures"] = {"switch": "failures", "cheat": "the server fails connections on purpose",
                                    "expected": "accepted, and each failed connection charged",
                                    "actual": f"error: {err!r}", "caught": False, "charged": False}
            finally:
                fserver.close()
    finally:
        challenger.close()
    order = ["baseline", *switches]
    return {s: rows[s] for s in order if s in rows}


def _err(check: dict) -> str | None:
    return check.get("error") if isinstance(check, dict) else None


def _baseline_row(check: dict) -> dict:
    err = _err(check)
    return {"switch": "baseline", "cheat": "honest serving (no cheat)",
            "expected": "accepted on every link-window",
            "actual": err or f"{check['verdict']} (network_verdict {check['network_verdict'].get('accepted')}, "
                      f"audit {check['audit_accepted']}, salt {check['salt_accepted']})",
            "caught": None, "charged": False, "ok": (err is None and check.get("verdict") == "accepted")}


def _tokens_row(check: dict) -> dict:
    err = _err(check)
    caught = (not err) and check["verdict"] == "refused"
    return {"switch": "tokens", "cheat": "ids of the server's own choosing, the frame counts honest's",
            "expected": "refused by the I/O tie (io_tie.content) at the sample's delta",
            "actual": err or (f"refused: {_codes(check)}" if caught else
                      "accepted by count-only N2 (not caught) — waits on the I/O tie (NET-297)"),
            "caught": caught, "charged": False}


def _length_row(extra: dict, drop: dict) -> dict:
    ee, de = _err(extra), _err(drop)
    both = (not ee) and (not de) and extra["verdict"] == "refused" and drop["verdict"] == "refused"
    late = ee or f"{extra['verdict']} {_codes(extra)}"
    early = de or f"{drop['verdict']} {_codes(drop)}"
    return {"switch": "length", "cheat": "a response ended late (extra event) or early (dropped event)",
            "expected": "refused by N2's counts (the stop check)",
            "actual": f"late {late}; early {early}", "caught": both, "charged": False}


def _timing_row(report: dict) -> dict:
    tau = ar.Setup(report).egress_params.bucket_ms / 1000.0
    lo, hi = (tau / s for s in SECONDS_A_BIT)
    return {"switch": "timing", "cheat": "the server moves the certifier's release time within a bucket",
            "expected": f"bits an interval under the budget {hi:.3g}..{lo:.3g} (tau={tau:.3g}s, tau/600..tau/240)",
            "actual": ("measured on a team machine: certifier_timing (certifier alone) or tick_load.py --secret-bit "
                       "(a rehearsal); the paced certifier has D_j=1, so 0 by construction"),
            "caught": None, "charged": False}


def _failures_row(check: dict, requests: int | None) -> dict:
    err = _err(check)
    accepted = (not err) and check["verdict"] == "accepted"
    served = "" if requests is None else f" over {requests} connection(s)"
    return {"switch": "failures", "cheat": "the server fails connections on purpose (a response of no tokens)",
            "expected": "accepted, and each failed connection charged as its own egress (leak_bounds schedule)",
            "actual": err or f"{check['verdict']}{served}; failed connections certified as served and charged",
            "caught": False, "charged": True, "ok": accepted}


def table(rows: dict) -> str:
    lines = []
    for r in rows.values():
        lines.append(f"[{r['switch']}] {r['cheat']}")
        lines.append(f"    expected: {r['expected']}")
        lines.append(f"    actual:   {r['actual']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="served_cheats.py", description=__doc__.splitlines()[0])
    ap.add_argument("--stand-in", action="store_true",
                    help="the dry run: build every record from an in-process GLM-5.3 token stand-in on an injected clock")
    ap.add_argument("--switch", choices=("all", *SWITCHES), default="all")
    ap.add_argument("--profile", choices=cs.PROFILES, default="served",
                    help="N2's profile; failures needs served (the planned profile fixes egress counts)")
    ap.add_argument("--out", type=Path, help="keep each switch's run directory and the rows here")
    a = ap.parse_args(argv)
    if not a.stand_in:
        raise SystemExit("only --stand-in (the dry run) is wired; the served run reads a kept GLM-5.3 record")
    switches = SWITCHES if a.switch == "all" else (a.switch,)
    import tempfile
    tmp = a.out if a.out is not None else Path(tempfile.mkdtemp(prefix="served-cheats-"))
    tmp.mkdir(parents=True, exist_ok=True)
    rows = asyncio.run(_serve(tmp, switches, a.profile))
    (tmp / "rows.json").write_text(json.dumps(rows, indent=1))
    print(table(rows), flush=True)
    # Only the honest baseline gates: if honest serving isn't accepted the stand-in or the recipe is broken. Each
    # cheat's verdict is a finding the table reports (tokens is not caught until the I/O tie, NET-297).
    base = rows.get("baseline")
    if base is not None and not base.get("ok"):
        print("\nbaseline honest run NOT accepted — stand-in or recipe broken", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
