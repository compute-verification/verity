"""What the relay's runner starts a run with: a connection's hand and flags, the keeper of its rows, the served
model's name, and the process's own usage."""
from __future__ import annotations

import argparse
import gc
import http.client
import json
import os
import re
import resource
import sys
import time
import urllib.request

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from verity.protocols.resource_accounting.network.verifier import blob_files as bf, connection_rows as CR
from verity.protocols.resource_accounting.network.verifier.certificate import Direction, Issuer
from verity.protocols.resource_accounting.network.verifier.connection_rows_check import FILED, MISSED, NO_HEAD, ROWS, SALTS
from verity.protocols.resource_accounting.network.verifier.network_certifier import LENGTHS_FORMS, Released, NetworkCertifierParams

#: the publisher queue's name for a hand's items, where a record's rows have their link's
HAND = "connection-rows"


def _class(text: str) -> tuple[int, ...]:
    refusal = argparse.ArgumentTypeError(f"a class is P,O,ROWS under --lengths 1 and P,O under --lengths 2, integers, "
                                         f"not {text!r}")
    try:
        numbers = tuple(int(x) for x in text.split(","))
    except ValueError:
        raise refusal from None
    if len(numbers) not in (2, 3):
        raise refusal
    return numbers


def flags(q: argparse.ArgumentParser) -> None:
    """The connection rows as `q`'s flags, beside its schedule's."""
    q.add_argument("--lengths", type=int, choices=LENGTHS_FORMS, default=None,
                   help="the lengths rows' form (network certifier's \"The lengths rows\"): the network certifier hands each "
                        "execution phase's releases, and the publisher commits and signs their tables; under 2 "
                        "every execution phase of the run ends, the last included, in its head or in the network "
                        "certifier's signed word that it has none (default: none)")
    q.add_argument("--content", type=_class, nargs="+", default=None, metavar="P,O[,ROWS]",
                   help="with --lengths: the content rows' classes, in order (content_rows' \"The classes\"): a "
                        "class's size of prompt ids, its size of ids released and, under --lengths 1 and only "
                        "there, its tables' rows; under --lengths 2 its tables hold a row a release filed in it "
                        "(default: none)")
    q.add_argument("--request-words", action="store_true",
                   help="with --lengths 2, --content and --max-tokens-cap: the request words (content_rows' \"A "
                        "request's words\"), a fourth table a class of each released request's max_tokens "
                        "and count of prompt ids as the network certifier reads them (default: none)")
    head_flags(q)


def head_flags(q: argparse.ArgumentParser) -> None:
    """The head's tick as `q`'s two flags, which a schedule sets together."""
    q.add_argument("--head-offset", type=int, default=None, metavar="N",
                   help="with --lengths and --head-wait: each execution phase's line of rows.jsonl leaves at a tick "
                        "of the schedule, N buckets after the close that hands it (network certifier's \"The head's tick\"), "
                        "written by the network certifier's tick, which notes the tick it left at in its own log "
                        "alone (default: none, the publisher writes it when it has signed)")
    q.add_argument("--head-wait", type=int, default=None, metavar="W",
                   help="with --head-offset: a line not back at its tick leaves at the first tick it is back by, "
                        "at most W ticks later, and never after; 0: a late line never leaves (no default)")


def schedule(params: NetworkCertifierParams, args: argparse.Namespace) -> NetworkCertifierParams:
    """`params` with the connection rows `args` names (`flags`); the same schedule where it names none."""
    content = getattr(args, "content", None)
    return replace(params, lengths=getattr(args, "lengths", None), content=content and tuple(content),
                   request_words=True if getattr(args, "request_words", False) else None,
                   head_offset=getattr(args, "head_offset", None), head_wait=getattr(args, "head_wait", None))


def flags_of(params: NetworkCertifierParams) -> list[str]:
    """The flags that give a schedule `params`' connection rows: none for a schedule without them."""
    out = [] if params.lengths is None else ["--lengths", str(params.lengths)]
    out += [] if params.content is None else ["--content", *(",".join(map(str, c)) for c in params.content)]
    out += ["--request-words"] if params.request_words else []
    return out + ([] if params.head_offset is None else ["--head-offset", str(params.head_offset), "--head-wait",
                                                         str(params.head_wait)])


def hand(window: int, strides: Sequence[Sequence[Released]] | None) -> Iterator[tuple]:
    """An execution phase's hand as the publisher queue's items: one a slot stride that holds a release, then its
    close with the strides' count; for an execution phase that ends without releases (`strides` None), its close
    alone, with no count."""
    for j, stride in enumerate(strides or ()):
        if stride:
            yield HAND, (window, j), stride
    yield HAND, (window, None), None if strides is None else len(strides)


class Keeper:
    """The publisher's keep of the hands (the module docstring's "The hand"): called with each item of `hand`, less
    its name, it lays, commits and signs an execution phase's tables at its close, or signs the word that it has no
    head. `keep` takes a whole hand, or None, as `NetworkCertifier(lengths=…)` gives it to a publisher in the network
    certifier's own process. In a schedule with a head offset it writes no line: it hands each one to `back`
    (execution phase, line), which that schedule and no other has."""

    def __init__(self, out: Path, issuer: Issuer, params: NetworkCertifierParams, link: str, back=None) -> None:
        if (back is None) != (params.head_offset is None):
            raise ValueError("in a schedule with a head offset, and in no other, the network certifier's tick writes "
                             "each execution phase's line, so the keeper hands the line back (`back`)")
        self.out, self.issuer, self.params, self.link, self.back = out, issuer, params, link, back
        self.open: dict[int, dict[int, Sequence[Released]]] = {}

    def _kept(self, window: int, line: dict, salts: bytes) -> None:
        """An execution phase's salts to `rows.salts`, and its line to `rows.jsonl` or back to the network
        certifier."""
        bf.write_blobs(self.out / SALTS, salts)
        if self.back is not None:
            self.back(window, line)
            return
        with open(self.out / ROWS, "a") as fh:
            fh.write(json.dumps(line) + "\n")

    def __call__(self, at: tuple[int, int | None], data) -> None:
        window, j = at
        if j is not None:
            self.open.setdefault(window, {})[j] = data
            return
        got = self.open.pop(window, {})
        self.keep(window, None if data is None else tuple(tuple(got.get(j, ())) for j in range(data)))

    def keep(self, window: int, strides: Sequence[Sequence[Released]] | None) -> None:
        if strides is None:
            t0 = time.perf_counter()
            cert = CR.attest_no_head(self.issuer, self.link, window, self.params)
            line = ({"refused": [self.link, Direction.EGRESS.value, window]} if cert is None
                    else {NO_HEAD: CR.no_head(cert, self.params)})
            self._kept(window, line | {"cost": {"sign_ms": round((time.perf_counter() - t0) * 1000, 3)}}, b"")
            self._filed(CR.not_filed(self.link, window))
            return
        t0 = time.perf_counter()
        tables = CR.tables(self.params, strides)
        t1 = time.perf_counter()
        rows = CR.Rows.commit(self.link, window, tables, self.params, self.issuer.session)
        head = rows.head
        t2 = time.perf_counter()
        cert = self.issuer.sign(self.link, Direction.EGRESS, window, head.digest())
        t3 = time.perf_counter()
        cost = {"tables": len(tables), "rows": sum(len(t) for t in tables.values()),
                "row_bytes": sum(len(r) for t in tables.values() for r in t),
                "build_ms": round((t1 - t0) * 1000, 3), "commit_ms": round((t2 - t1) * 1000, 3),
                "sign_ms": round((t3 - t2) * 1000, 3)}
        line = ({"refused": [self.link, Direction.EGRESS.value, window]} if cert is None
                else {"registration": CR.registration(head, cert, self.params)})
        self._kept(window, line | {"cost": cost}, rows.witness)
        self._filed(CR.not_filed(self.link, window) if cert is None else CR.filed_record(head, self.params, strides))

    def _filed(self, record: Mapping) -> None:
        """An execution phase's line of `filed.jsonl`, in a schedule with content rows: written once its head is
        signed, or its word that it has none, so the list is final where it is written."""
        if self.params.content is not None:
            with open(self.out / FILED, "a") as fh:
                fh.write(json.dumps(record) + "\n")


class Lines:
    """The network certifier's `leave` in a schedule with a head offset (`connection_rows_run`'s "The hand"): called
    by its tick with an execution phase, the tick and the line the publisher handed back, it appends the line to
    `rows.jsonl` as it was handed back; with None for the line, the note that the execution phase's line missed its
    last tick. It writes nothing of the tick, which the device's own log alone holds (`NetworkCertifier.heads`). The file is
    opened once, before the run."""

    def __init__(self, out: Path, link: str) -> None:
        self.link, self.file = link, open(out / ROWS, "a")

    def __call__(self, window: int, tick: int, line: dict | None) -> None:
        doc = {MISSED: [self.link, Direction.EGRESS.value, window]} if line is None else line
        self.file.write(json.dumps(doc) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def peak_rss_mb() -> float:
    """This process's peak resident set (`VmHWM`; `ru_maxrss` survives exec, so a spawned child reports its parent's)."""
    status = Path("/proc/self/status").read_text()
    return int(re.search(r"VmHWM:\s+(\d+) kB", status).group(1)) / 1024


def _quiet_runtime() -> None:
    gc.disable()
    sys.setswitchinterval(0.001)
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))  # a live run holds two sockets per stream in flight


def cpu_ranges(cpus) -> str:
    """CPU indices in the kernel's list form, their runs joined by commas ("0-3,8,10-11"). One string: a record that
    holds it writes no run of consecutive integers, which is the same bytes in any file that lists consecutive
    integers, a verifier outcome's proved units among them."""
    runs: list[list[int]] = []
    for c in sorted(cpus):
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _usage() -> dict:
    return {"max_rss_mb": peak_rss_mb(), "cpu_s": sum(os.times()[:2]), "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "cpus": cpu_ranges(os.sched_getaffinity(0))}


def _call(url: str, path: str, body: dict | None = None, key: str | None = None) -> dict:
    """The JSON the endpoint at `url` (`http(s)://HOST[:PORT][/PREFIX]`) answers at `path`: to a GET, or to a POST of
    `body` as JSON, sent `key` as the bearer key when there is one. A `url` of any other form is refused by name
    before anything is opened, so only HTTP is spoken. An answer that isn't a 2xx raises `OSError`, as a connection
    that fails does; a redirect is such an answer."""
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise SystemExit(f"an endpoint is http(s)://HOST[:PORT][/PREFIX], not {url!r}")
    headers = {"content-type": "application/json", "connection": "close"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    connect = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    connection = connect(u.hostname, u.port, timeout=60)
    try:
        connection.request("GET" if body is None else "POST", u.path.rstrip("/") + path,
                           None if body is None else json.dumps(body).encode(), headers)
        answer = connection.getresponse()
        if not 200 <= answer.status < 300:
            raise OSError(f"{url.rstrip('/')}{path} answered {answer.status} {answer.reason}")
        return json.loads(answer.read())
    finally:
        connection.close()


def served_model(url: str, key: str | None = None) -> str:
    """The one model the endpoint serves (`/v1/models`), asked with the bearer key `key` when there is one."""
    ids = [m["id"] for m in _call(url, "/v1/models", key=key)["data"]]
    if len(ids) != 1:
        raise SystemExit(f"{url} serves {ids}: name the model (--model)")
    return ids[0]
