"""The strings a served window's close hands `Window.close` (``strings``): each of its `leaf_rows` ports' salts, unit u's the
stream of the window's key at index (template, port), leaf u (`salt_stream.fill`, one call a port), and its rows' ``b || c``
under them: a row the serve acquired hashed where it is committed (`window_plan.stage`; on a card native_rows.cu's
`verity_rows_h_launch`, one parameter build and one launch a port), every other row on the host from the window's recorded
rows. `HostTwin` computes the card's strings on the processor."""
from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

import numpy as np
from verity.commitments import hm96
from verity.commitments import randomness as RND
from verity.commitments.cuda.salt_stream.engine import LEAF_BYTES, fill
from verity.commitments.rowleaf import (
    SCHEMA_HM96_SHA512_ROW_V2,
    sha512_row_v2_digest,
    sha512_row_v2_prefix,
)

from .window_plan import stage

__all__ = ["HANDOFF", "Device", "HostTwin", "acquired", "block_of", "card_ports", "card_strings", "handed_strings",
           "leaf_strings", "load_card", "save_card", "strings_of"]

INST = hm96.SHA512
KEY = INST.default_key
BC_BYTES = 128
#: the twin's sample of ports, under a key drawn from the OS for the one comparison
TWIN_SAMPLE_DOMAIN = "verity/vllm/twin-sample/v1"


def _sample[T](items: Sequence[T], k: int) -> list[T]:
    return [items[i] for i in RND.derive(os.urandom(64), TWIN_SAMPLE_DOMAIN, {}).subset(min(k, len(items)), len(items))]


def _stride(lb: int) -> int:
    return -(-lb // 16) * 16


class HostTwin:
    """The card's strings on the processor: numpy rows and salts, the salts from the host salt library
    (`salt_stream.host_lib`), each row's ``b || c`` its v2 row digest's hm96 string."""

    def __init__(self, lib: Any) -> None:
        self.lib = lib

    def salts(self, key: Any, index: tuple[int, int], n: int) -> np.ndarray:
        out = np.empty((n, LEAF_BYTES), dtype=np.uint8)
        fill(self.lib, out, key, index, 0)
        return out

    def rows(self, rows: Sequence[Any], stride: int) -> np.ndarray:
        out = np.zeros((len(rows), stride), dtype=np.uint8)
        for i, r in enumerate(rows):
            r = np.asarray(r, dtype=np.uint8).reshape(-1)
            out[i, :len(r)] = r
        return out

    def strings(self, rows: np.ndarray, lb: int, ys: np.ndarray) -> np.ndarray:
        return host_strings(rows, lb, ys)

    def host(self, x: Any) -> np.ndarray:
        return np.asarray(x)


class Device:
    """The same on a CUDA device: the salts from the device salt library ``lib`` (`salt_stream.device_lib`) into device
    memory, the strings native_rows.cu's with the port's parameters built once (`RowStep`), on the device's current stream."""

    def __init__(self, lib: Any, device: Any) -> None:
        import torch

        from verity_vllm.commit import step_rows
        from verity_vllm.commit.committer.native_ext import ext

        self.lib, self.device, self.ext = lib, torch.device(device), ext()
        self.hasher = self.ext.StepHasher()
        self.tn = torch.frombuffer(bytearray(step_rows.t_nibbles(KEY)), dtype=torch.uint8).to(self.device)

    def _stream(self) -> int:
        import torch

        return int(torch.cuda.current_stream(self.device).cuda_stream)

    def salts(self, key: Any, index: tuple[int, int], n: int) -> Any:
        import torch

        out = torch.empty((n, LEAF_BYTES), dtype=torch.uint8, device=self.device)
        fill(self.lib, out, key, index, 0, stream=self._stream())
        return out

    def rows(self, rows: Sequence[Any], stride: int) -> Any:
        import torch

        out = torch.zeros((len(rows), stride), dtype=torch.uint8, device=self.device)
        for i, r in enumerate(rows):
            r = r.reshape(-1)
            out[i, :r.numel()] = r
        return out

    def strings(self, rows: Any, lb: int, ys: Any) -> Any:
        # the domain id and the salt key are unread: no frame leaf is made, and every row's salt is given
        rs = self.ext.RowStep(bytes(64), bytes(64), sha512_row_v2_prefix(8 * lb), INST.salt_prefix, INST.leaf_prefix(KEY),
                              SCHEMA_HM96_SHA512_ROW_V2, lb)
        n, stride = int(rows.shape[0]), int(rows.shape[1])
        return self.hasher.rows_h(rs, rows, n * stride, 0, self.tn, self._stream(), stride=stride, salts=ys.contiguous(),
                                  want_bc=True)[1]

    def host(self, x: Any) -> np.ndarray:
        return x.cpu().numpy()


def host_strings(rows: np.ndarray, lb: int, ys: np.ndarray) -> np.ndarray:
    """``(m, 128)`` uint8: each row's first ``lb`` bytes' ``b || c`` under its salt, as the kernel computes it."""
    out = b"".join(INST.commit_string(KEY, sha512_row_v2_digest(bytes(r[:lb]), 8 * lb), bytes(y)) for r, y in zip(rows, ys))
    return np.frombuffer(out, dtype=np.uint8).reshape(len(rows), BC_BYTES)


def block_of(com: Any) -> Callable[[int, str], tuple[Any, int]]:
    """``block(step, name)`` over committer ``com``'s retained blocks (`window_plan.stage`'s): the step's block, a uint8
    tensor, and the member's offset in it (on the card where the member was staged there, else in its step's stream)."""
    from verity_vllm.commit.opened import meta_by_name

    def block(step: int, name: str) -> tuple[Any, int]:
        m = meta_by_name(com._layouts[step], name)
        if m is None:
            raise ValueError(f"step {step} commits no member {name}")
        dev = int(getattr(m, "dev_off", -1))
        return com._gpu_blocks[step][0], dev if dev >= 0 else int(m.stream_off)

    return block


def acquired(plan: Any, block: Callable[[int, str], tuple[Any, int]], ports: Collection) -> dict[tuple[str, str], dict[int, Any]]:
    """By each of ``ports`` (template id, output) a served node feeds (`Plan.feeds`), its acquired units' rows where they are
    committed (`stage`), by unit."""
    out: dict[tuple[str, str], dict[int, Any]] = {}
    for f in plan.feeds:
        if (f.tid, f.out) in ports:
            rows = stage(f, block)
            out.setdefault((f.tid, f.out), {}).update({f.unit + i: rows[i] for i in range(f.rows)})
    return out


def leaf_strings(leaf_rows: Mapping[tuple[str, str], Any], key: Any, hasher: Any,
                 staged: Mapping[tuple[str, str], Mapping[int, Any]]) -> tuple[dict[tuple[str, str], tuple[bytes, bytes]], dict]:
    """Each `leaf_rows` port's salts and strings (`Window.close`'s ``strings``) under ``key``: a unit in ``staged`` hashed
    by ``hasher`` from its acquired row, every other unit on the host from its recorded row (`LeafRows.rows`); a port of a
    schema other than ``hm96-sha512/row/v2``, which the kernel's row hash is, refused by name. -> the strings and their cost:
    units hashed where acquired and on the host, and each one's seconds."""
    out: dict[tuple[str, str], tuple[bytes, bytes]] = {}
    cost = {"acquired": 0, "acquired_s": 0.0, "host": 0, "host_s": 0.0}
    for k, lr in leaf_rows.items():
        if lr.schema != SCHEMA_HM96_SHA512_ROW_V2:
            raise ValueError(f"strings: {k} is {lr.schema}, and the row hash computes {SCHEMA_HM96_SHA512_ROW_V2}'s strings alone")
        units, rows = list(lr.units), lr.rows
        lb = 2 * int(rows.shape[1])
        ys = hasher.salts(key, (int(lr.template), int(lr.port)), units[-1] + 1)[units]
        hy, bc = hasher.host(ys), np.empty((len(units), BC_BYTES), dtype=np.uint8)
        fed = staged.get(k, {})
        at = [i for i, u in enumerate(units) if u in fed]
        if at:
            t0 = time.perf_counter()
            bc[at] = hasher.host(hasher.strings(hasher.rows([fed[units[i]] for i in at], _stride(lb)), lb, ys[at]))
            cost["acquired_s"] += time.perf_counter() - t0
        if rest := [i for i, u in enumerate(units) if u not in fed]:
            t0 = time.perf_counter()
            r = np.zeros((len(rest), lb), dtype=np.uint8)
            r[:] = np.ascontiguousarray(rows[rest]).astype("<u2").view(np.uint8)
            bc[rest] = host_strings(r, lb, hy[rest])
            cost["host_s"] += time.perf_counter() - t0
        cost["acquired"], cost["host"] = cost["acquired"] + len(at), cost["host"] + len(rest)
        out[k] = (hy.tobytes(), bc.tobytes())
    cost["host_s"], cost["acquired_s"] = round(cost["host_s"], 3), round(cost["acquired_s"], 3)
    return out, cost


def strings_of(win: Any, key: Any, hasher: Any, plan: Any = None, block: Callable[[int, str], tuple[Any, int]] | None = None,
               *, twin: HostTwin | None = None, sample: int = 0,
               cost: dict[str, Any] | None = None) -> dict[tuple[str, str], tuple[bytes, bytes]]:
    """`leaf_strings` of ``win``'s `leaf_rows`, the units ``plan`` feeds staged from ``block``; given ``twin``, up to
    ``sample`` ports with acquired units, drawn from the OS, recomputed whole by it from host copies of the same rows and
    compared byte for byte. Prints one ``WINDOW-STRINGS`` line, its cost and the comparison, also into ``cost``."""
    lrs = win.leaf_rows()
    staged = acquired(plan, block, lrs) if plan is not None and block is not None else {}
    strings, got = leaf_strings(lrs, key, hasher, staged)
    got = {"ports": len(strings), "where": type(hasher).__name__, **got}
    if twin is not None and sample:
        fed = sorted(k for k in staged if k in lrs)
        pick = _sample(fed, sample)
        t0 = time.perf_counter()
        host = {k: {u: hasher.host(r) for u, r in staged[k].items()} for k in pick}
        again, _ = leaf_strings({k: lrs[k] for k in pick}, key, twin, host)
        unequal = [list(k) for k in pick if again[k] != strings[k]]
        got["twin"] = {"ports": len(pick), "of": len(fed), "units": sum(len(host[k]) for k in pick), "equal": not unequal,
                       "unequal": unequal[:4], "s": round(time.perf_counter() - t0, 3)}
    if cost is not None:
        cost.update(got)
    sys.stderr.write("WINDOW-STRINGS " + json.dumps(got) + "\n")
    sys.stderr.flush()
    return strings


def card_ports(circuit: Any, ports: Collection) -> dict[tuple[str, str], tuple[int, int, int]]:
    """By each of ``ports`` (template id, port), the numbering `Window.leaf_rows` gives it as `interface.members` lays a
    member's ports from the circuit: its template's ordinal, its ordinal among the member's cut rows then outputs, and its
    row's bytes (a member names each port's schema, so an output's row is its own words). `handed_strings` holds it to
    the window's."""
    tids = {t: i for i, t in enumerate(circuit.ids)}
    out = {}
    for tid, port in ports:
        lay = circuit.layouts[tid][0]
        laid = [(p, w) for p, w, _ in lay.cut(circuit.cuts.get(tid, {}))] + list(lay.outs)
        out[(tid, port)] = (tids[tid], [p for p, _ in laid].index(port), 2 * dict(laid)[port])
    return out


def card_strings(plan: Any, block: Callable[[int, str], tuple[Any, int]], circuit: Any, key: Any,
                 hasher: Any) -> tuple[dict[tuple[str, str], tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]], dict]:
    """The strings of every unit ``plan`` feeds, hashed by ``hasher`` (on a card, `Device`) under ``key`` from its row where
    it is committed (`acquired`), at `card_ports`' numbering. -> by port, its numbering, units, the rows' host copies and
    their strings; and the cost."""
    t0 = time.perf_counter()
    staged = acquired(plan, block, {(f.tid, f.out) for f in plan.feeds})
    idx, out = card_ports(circuit, staged), {}
    for k, fed in staged.items():
        t, p, lb = idx[k]
        units = sorted(fed)
        ys = hasher.salts(key, (t, p), units[-1] + 1)[units]
        rows = hasher.rows([fed[u] for u in units], _stride(lb))
        bc = hasher.host(hasher.strings(rows, lb, ys))
        out[k] = ((t, p, lb), np.asarray(units, dtype=np.int64), np.ascontiguousarray(hasher.host(rows)[:, :lb]), bc)
    return out, {"where": type(hasher).__name__, "ports": len(out), "units": sum(len(v[1]) for v in out.values()),
                 "s": round(time.perf_counter() - t0, 3)}


#: what the card leaves for the host to finish the record from, in a window's ``handoff/``: its salt key, so private
HANDOFF = ("handoff.json", "card.npz", "key.bin")


def save_card(to: Any, card: Mapping, key: Any, meta: Mapping[str, Any]) -> None:
    """``card`` (`card_strings`) and ``key`` written as `HANDOFF` under ``to``, ``meta`` beside the ports."""
    from pathlib import Path

    to = Path(to)
    to.mkdir(parents=True, exist_ok=True)
    (to / HANDOFF[2]).write_bytes(key.value)
    ks = sorted(card)
    np.savez(to / HANDOFF[1], **{f"{x}{i}": card[k][j] for i, k in enumerate(ks) for j, x in ((1, "u"), (2, "r"), (3, "s"))})
    (to / HANDOFF[0]).write_text(json.dumps({**meta, "ports": [[*k, *card[k][0]] for k in ks]}) + "\n")


def load_card(to: Any) -> tuple[Any, dict, dict[str, Any]]:
    """`save_card`'s ``(key, card, meta)`` from ``to``; refused by name where one of `HANDOFF` is missing."""
    from pathlib import Path

    to = Path(to)
    if missing := [f for f in HANDOFF if not (to / f).is_file()]:
        raise ValueError(f"handoff {to}: no {', '.join(missing)}; the card's part of the close did not leave it")
    meta = json.loads((to / HANDOFF[0]).read_text())
    z = np.load(to / HANDOFF[1])
    card = {(tid, port): ((t, p, lb), z[f"u{i}"], z[f"r{i}"], z[f"s{i}"]) for i, (tid, port, t, p, lb) in enumerate(meta["ports"])}
    from verity.commitments.randomness import Key

    return Key((to / HANDOFF[2]).read_bytes()), card, meta


def handed_strings(win: Any, key: Any, twin: HostTwin, card: Mapping, *, sample: int = 0,
                   cost: dict[str, Any] | None = None) -> dict[tuple[str, str], tuple[bytes, bytes]]:
    """`Window.close`'s strings with each unit ``card`` holds (`card_strings`, under ``key``) the card's, every other unit on
    the host from its recorded row. Refused by the port's name: a handed port whose numbering differs from the window's
    `leaf_rows`, or a handed row not the recorded one; and, on up to ``sample`` handed ports drawn from the OS, a string
    of the card's that the host reference ``twin`` doesn't recompute byte for byte. Prints one ``WINDOW-STRINGS`` line."""
    lrs = win.leaf_rows()
    out, got = {}, {"ports": len(lrs), "where": "card, handed", "acquired": 0, "host": 0, "host_s": 0.0, "handed_unleafed": 0}
    for k, lr in lrs.items():
        if lr.schema != SCHEMA_HM96_SHA512_ROW_V2:
            raise ValueError(f"strings: {k} is {lr.schema}, and the row hash computes {SCHEMA_HM96_SHA512_ROW_V2}'s strings alone")
        units, rows = list(lr.units), lr.rows
        lb = 2 * int(rows.shape[1])
        ys = twin.salts(key, (int(lr.template), int(lr.port)), units[-1] + 1)[units]
        bc, pos, at = np.empty((len(units), BC_BYTES), dtype=np.uint8), {u: i for i, u in enumerate(units)}, []
        if (h := card.get(k)) is not None:
            if tuple(h[0]) != (int(lr.template), int(lr.port), lb):
                raise ValueError(f"handoff: port {k}: the card hashed it as (template, port, bytes) {tuple(h[0])}, the window's "
                                 f"leaf rows are {(int(lr.template), int(lr.port), lb)}")
            pairs = [(pos[int(u)], j) for j, u in enumerate(h[1]) if int(u) in pos]
            at, js = [i for i, _ in pairs], [j for _, j in pairs]
            mine = np.ascontiguousarray(rows[at]).astype("<u2").view(np.uint8).reshape(len(at), lb)
            if not np.array_equal(mine, h[2][js]):
                raise ValueError(f"handoff: port {k}: a row the card hashed is not the window's recorded row")
            bc[at] = h[3][js]
            got["handed_unleafed"] += len(h[1]) - len(at)
        taken = set(at)
        if rest := [i for i in range(len(units)) if i not in taken]:
            t0 = time.perf_counter()
            r = np.ascontiguousarray(rows[rest]).astype("<u2").view(np.uint8).reshape(len(rest), lb)
            bc[rest] = host_strings(r, lb, ys[rest])
            got["host_s"] += time.perf_counter() - t0
        got["acquired"], got["host"] = got["acquired"] + len(at), got["host"] + len(rest)
        out[k] = (ys.tobytes(), bc.tobytes())
    fed = sorted(k for k in card if k in lrs)
    pick = _sample(fed, sample)
    t0 = time.perf_counter()
    for k in pick:
        (t, p, lb), units, rows, bc = card[k]
        ys = twin.salts(key, (t, p), int(units[-1]) + 1)[units]
        if not np.array_equal(host_strings(rows, lb, ys), bc):
            raise ValueError(f"WINDOW-STRINGS: port {k}: the card's strings are not the host reference's, byte for byte")
    got["host_s"] = round(got["host_s"], 3)
    got["twin"] = {"ports": len(pick), "of": len(fed), "units": sum(len(card[k][1]) for k in pick), "equal": True,
                   "unequal": [], "s": round(time.perf_counter() - t0, 3)}
    if cost is not None:
        cost.update(got)
    sys.stderr.write("WINDOW-STRINGS " + json.dumps(got) + "\n")
    sys.stderr.flush()
    return out
