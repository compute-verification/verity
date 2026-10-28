"""The proof system's challenger on its auditor host, in this process: a `LiveChallenger` its test hands in, its session
opened under an id with its record signed with an LMS key (`remote.open_session`), and `remote.serve` on one end of a socket
pair, the serving side's `RemoteChallenger` on the other; the session's close is signed at the key's next leaf. The
auditor's signed ledger (`Ledger`, infra's ``windows.jsonl``, v2) opens the session for its window before the first coin and
ends it once it has ended, sealing its files (the refused hellos' log, empty here, among them) and naming its record's
leaf, its commitment and how many hellos it refused. The verifier reads `directory` under `key`, as the ledger opens and ends it.
`Gate` is infra's v2 gate instead, whose ledger `pouw_h1_challenger.Ledger` reads with core's check. Each seals the host's
pins log (`PINS`, one pin of `PINNED`, every open's ``pinned``) in its opens as the gate does (``pins: {lines, sha512}``),
and writes it as infra publishes it (``pins.jsonl``, the gate's ``pins``), so `challenger.pinned_session` holds each
session to it, and to the verifier's copy of main's first-parent history `MAIN`, which holds `PINNED`. Each holds a
`Registry`, infra's model registry in this process, whose registration a period opened with ``model`` signs (the gate's
``period-open --model``) and whose key `Gate.model_key` hands the lane (``model-key``), over any challenger."""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import itertools
import json
import os
import random
import socket
import threading
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timedelta
from pathlib import Path

from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw import weights as Wt
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import challenger as LC
from verity.core.protocols.sampled_proofs import remote as R
from verity.ml.crypto import lms
from verity_vllm.protocols.pouw.h1 import challenger as PHC

HOST = "vy-challenger-1"
#: the commit every open here was run from (``pinned``), the host's pins log recording it as main's, and the
#: verifier's own ``git rev-list --first-parent origin/main``, which holds it (`challenger.pinned_session`)
PINNED = "0" * 40
PINS = (LC.pin_line(PINNED, PINNED, "2026-10-20T10:00:00Z"),)
MAIN = ("e1" * 20, PINNED)
#: the int8 checkpoint role a period's open names with its weights' root (``--model``), one core's credit pins
MODEL = "LLAMA31_8B_INT8"
#: the tree infra's model registry reads core's pins in (`models.PINS`)
TREE = Path(__file__).resolve().parents[4]


@functools.cache
def models():
    """Infra's model registry, ``benchmarks/live_challenger/models.py``, as a module of its own."""
    spec = importlib.util.spec_from_file_location("live_challenger_models", TREE / "benchmarks/live_challenger/models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def linear_rows(codes, scales) -> Iterator[bytes]:
    """A linear's B rows (`weights.row_bytes`) from its int8 codes ``(n, K)`` and f32 s_w ``(n,)`` or ``(n, 1)``, as
    `models.rows_of` reads them from a checkpoint."""
    import numpy as np
    s = np.asarray(scales, "<f4").reshape(-1)
    for j, row in enumerate(np.asarray(codes, np.int8)):
        yield Wt.row_bytes(row.tobytes(), s[j].tobytes())


class Registry:
    """Infra's model registry (`models`: the gate's ``model-register``, ``model-key`` and ``period-open --model``) in this
    process. `register` commits a model's ``b/<K>`` as `models.register` does: the window Program's linears from its
    descriptor (`models.layout`), each K's root by core's `weights.weights_roots` under a fresh salt key, and the line
    appended in `models.register`'s form; `key` is the gate's ``model-key`` answer (`models.resolve`, one registration
    of the model under the circuit) and `opened` what a period-open signs for its model (`models.opened`), each Refused
    (`models.Refused`) as the gate refuses. What differs from infra's: the checkpoint is ``rows``, the test's own linears
    by name, not a pinned checkpoint's files, so its ``checkpoint_sha512`` is core's pin of the role (`models.
    pinned_checkpoint`), unchecked, and its linears need not be a Llama decoder's."""

    def __init__(self, entropy: Callable[[int], bytes] | None = None) -> None:
        self.lines: list[dict] = []
        self._keys: dict[str, bytes] = {}
        self._entropy = entropy or random.Random("registry").randbytes

    def register(self, model: str, circuit: str, descriptor: bytes, rows: Callable[[str, int], Iterable[bytes]]) -> dict:
        """The registry's line of `model`'s weights under `circuit` (`descriptor`'s SHA-512), ``rows(name, K)`` each
        linear's B rows in the window's order."""
        M = models()
        pin = M.pinned_checkpoint(TREE, model)
        prefix, linears = M.layout(descriptor, circuit)
        key = self._entropy(32)
        shapes = [(n, k) for _name, k, n in linears]
        roots = Wt.weights_roots(circuit, prefix, shapes, lambda w: rows(linears[w][0], linears[w][1]), key)
        leaves = {k: sum(n for n, kw in shapes if kw == k) for k in roots}
        line = {"model": model, "circuit": circuit, "prefix": prefix, "checkpoint_sha512": pin,
                "roots": {str(k): roots[k].hex() for k in sorted(roots)},
                "entries": [Wt.entry(circuit, prefix, k, leaves[k], roots[k]) for k in sorted(roots)],
                "weights_root": Wt.weights_root(roots).hex(), "key_sha512": hashlib.sha512(key).hexdigest(),
                "pinned": PINNED, "registered_utc": "2026-10-20T10:30:00Z"}
        self._keys[line["key_sha512"]] = key
        self.lines.append(line)
        return line

    def key(self, model: str, circuit: str, weights_root: str | None = None) -> dict:
        """The gate's ``model-key --model M --circuit C [--weights-root W]`` answer (`auditor.gate_model_key`'s)."""
        e = models().resolve(self.lines, circuit, model, weights_root, unique=True)
        return {"model": e["model"], "circuit": e["circuit"], "weights_root": e["weights_root"],
                "key": self._keys[e["key_sha512"]].hex(), "key_sha512": e["key_sha512"]}

    def opened(self, circuit: str | None, model: str | None, weights_root: str | None) -> dict:
        """What a period-open of `circuit` naming `model` (``--model``) or `weights_root` (``--weights-root``) signs."""
        return models().opened(self.lines, TREE, circuit, model, weights_root)


def write_pins(path: Path, pins=PINS) -> Path:
    """`pins` (the host's pins log, its lines) at `path`, as infra publishes it (the gate's ``pins``)."""
    path.write_text("".join(line + "\n" for line in pins))
    return path


def entropy(seed: int):
    """A seeded entropy source: a challenger's coins and salts, a distinct seed per challenger."""
    return random.Random(seed).randbytes


def challenger_key(seed: int, H: int = 4) -> tuple[lms.PrivateKey, LC.ChallengerKey]:
    """Key `seed`: 2^H leaves (sixteen by default, five a reserved session: the ledger's reserve and open, the record, the
    close and the ledger's end)."""
    k = lms.keygen(lms.Params(64, 4, H), bytes([seed]) * 64)
    return k, LC.ChallengerKey(k.params.W, k.params.H, lms.public_key(k))


def window(seed: int) -> dict:
    """A window's identity, ``{root}``, as the auditor's ledger opens it."""
    return {"root": random.Random(f"window/{seed}").randbytes(64).hex()}


WINDOW = window(0)


def session_id(n: int) -> str:
    """Gate session id `n`, of the form infra's gate gives a reserved session (``s<UTC>-<4 hex>``)."""
    return f"s20261020T110800Z-{n:04x}"


class Gate:
    """Infra's challenger gate (`benchmarks/live_challenger/auditor.py`, v2) in this process, its ledger's entries
    written as the gate's `append` writes them and signed under `private` at rising one-time keys, from `leaf` on:
    `reserve` appends a window's reserve entry, `open` its open entry (two-step after its reserve, else one-step) and
    returns the gate's open answer (``commitment`` null, as before serve.py has signed its record), `take` the next leaf
    for a session's record, `end` a session's end entry over its ``out/`` as the gate's ``run`` computes it,
    `period_open` and `period_close` a lane's period entries (the close listing its windows' outcomes as the gate
    counts them), `model_register` and `model_key` the model registry's verbs (`registry`), and `lines` is the ledger as
    infra publishes it. Unlike the gate it appends whatever it is asked, so a test writes the ledgers the verifier must
    refuse, except that a period-open naming a model or a weights' root signs what the registry resolves it to, and is
    refused as the gate refuses it."""

    def __init__(self, private: lms.PrivateKey, key: LC.ChallengerKey, leaf: int = 0,
                 registry: Registry | None = None) -> None:
        self.private, self.key, self._leaf, self.lines, self._opens = private, key, leaf, [], {}
        self.registry = registry or Registry()
        self.pins = list(PINS)
        self._reserves: dict[str, dict] = {}
        self._periods: dict[str, dict] = {}
        self._current: dict[str, str] = {}
        self._complete: dict[str, bool] = {}

    @property
    def pin(self) -> str:
        return hashlib.sha512(self.key.public).hexdigest()

    def _join(self, lane: str, session: str) -> None:
        if (name := self._current.get(lane)) is not None:
            self._periods[name]["windows"].append(session)

    def reserve(self, session: str, families, *, scheme: str = Wd.LIVE_SCHEME, epoch: int = 0, by: str = "vllm",
                for_: str | None = None, minutes: int = 10, started: str = "2026-10-20T11:08:00Z", **override) -> dict:
        """A window's reserve entry, its expiry `started` plus `minutes` (the gate's ``reserve``), its fields replaced by
        `override`; its session's `open` is then two-step."""
        expires = (datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ") + timedelta(minutes=minutes)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        body = {"by": by, **({"for": for_} if for_ else {}), "peer": "vy-nebius-1", "scheme": scheme, "epoch": epoch,
                "families": [list(f) for f in families], "minutes": minutes, "pinned": PINNED, "key_sha512": self.pin,
                "host": f"{HOST}.{session}", "started_utc": started, "expires_utc": expires, **override}
        e = self.append("reserve", session, body)
        self._reserves[session] = e
        self._join(for_ or by, session)
        return e

    def model_register(self, model: str, circuit: str, descriptor: bytes,
                       rows: Callable[[str, int], Iterable[bytes]]) -> dict:
        """vy-admin's ``model-register --model M --circuit C --descriptor FILE``, the checkpoint ``rows`` (`Registry`):
        the gate's answer."""
        line = self.registry.register(model, circuit, descriptor, rows)
        return {k: line[k] for k in ("model", "circuit", "prefix", "checkpoint_sha512", "roots", "weights_root",
                                     "key_sha512", "registered_utc")}

    def model_key(self, model: str, circuit: str, weights_root: str | None = None) -> dict:
        """A lane's ``model-key --model M --circuit C [--weights-root W]``: the gate's answer, or ``{refused}``."""
        try:
            return self.registry.key(model, circuit, weights_root)
        except models().Refused as e:
            return {"refused": str(e)}

    def period_open(self, name: str, lane: str = "vllm", *, by: str = "vllm", cap: int | None = None,
                    circuit: str | None = None, draw_k: int | None = None, rows: int | None = None,
                    model: str | None = None, weights_root: str | None = None, length: dict | None = None,
                    **override) -> dict:
        """The period's open as the gate's ``period-open`` writes it, with its ``cap``, ``circuit``, ``draw_k`` (the
        proof system's committed draw K, `credit.K_FIELD`), ``rows`` (the table rows T, `credit.ROWS_FIELD`) and
        ``length`` (``--length``, `credit.LENGTH_FIELD`) when
        given, and with ``model`` (``--model``) or
        ``weights_root`` (``--weights-root``) what the registry resolves them to (`Registry.opened`), Refused
        (`models.Refused`) for a root it has no entry of."""
        signed = self.registry.opened(circuit, model, weights_root)
        body = {"period": name, "lane": lane, "by": by, **({"cap": cap} if cap is not None else {}),
                **({"circuit": circuit} if circuit is not None else {}),
                **({CR.K_FIELD: draw_k} if draw_k is not None else {}),
                **({CR.ROWS_FIELD: rows} if rows is not None else {}),
                **({CR.LENGTH_FIELD: length} if length is not None else {}), **signed, "pinned": PINNED,
                "key_sha512": self.pin, "opened_utc": "2026-10-20T11:00:00Z", **override}
        self._periods[name], self._current[lane] = {"lane": lane, "windows": []}, name
        return self.append("period-open", None, body)

    def period_close(self, name: str, **override) -> dict:
        """The period's close as the gate's ``period`` writes it: each of its windows, in order, with its root (None
        when it was never opened) and whether its end is complete, and complete when it has one and all are."""
        p = self._periods[name]
        windows = [{"root": self._opens[s]["window"]["root"] if s in self._opens else None, "session": s,
                    "complete": self._complete.get(s, False)} for s in p["windows"]]
        body = {"period": name, "lane": p["lane"], "by": "vllm", "windows": windows,
                "complete": bool(windows) and all(w["complete"] for w in windows), "closed_utc": "2026-10-20T12:00:00Z",
                **override}
        if self._current.get(p["lane"]) == name:
            del self._current[p["lane"]]
        return self.append("period-close", None, body)

    def take(self) -> int:
        self._leaf += 1
        return self._leaf - 1

    def append(self, kind: str, session: str | None, body: dict, *, schema: str = LC.LEDGER_SCHEMA,
               leaf: int | None = None) -> dict:
        prev = hashlib.sha512(self.lines[-1].encode()).hexdigest() if self.lines else "0" * 128
        entry = {"schema": schema, "seq": len(self.lines), "prev": prev, "kind": kind, "session": session, "body": body}
        message = LC.ledger_message(self.key.public[:16], entry)
        if leaf is not None:
            self._leaf = max(self._leaf, leaf)
        entry["sig"] = lms.sign_at(self.private, self.take() if leaf is None else leaf, message).encode().hex()
        if leaf is not None:
            self._leaf = max(self._leaf, leaf + 1)
        self.lines.append(LC.ledger_text(entry))
        return entry

    def open(self, session: str, root: str, families=(), *, scheme: str = Wd.LIVE_SCHEME, epoch: int = 0,
             key_sha512: str | None = None, by: str = "vllm", v1: bool = False, leaf: int | None = None,
             **override) -> dict:
        """A window's open entry and the gate's answer: two-step (binding its reserve, carrying its request, opened by
        `by`) when `reserve` reserved the session, else one-step (``families`` and the rest its own); its body's fields
        replaced by `override`."""
        secret = hashlib.sha512(session.encode()).hexdigest()[:64]
        reserved = self._reserves.get(session)
        if reserved is not None:
            rb = reserved["body"]
            body = {"window": {"root": root}, "reserve": reserved["seq"], **{k: rb[k] for k in LC.RESERVED if k in rb},
                    "opened_by": by, "pinned": PINNED, "pins": LC.pins_seal(self.pins),
                    "key_sha512": key_sha512 or self.pin, "host": rb["host"], "port": 7811, "secret_sha512": hashlib.sha512(bytes.fromhex(secret)).hexdigest(),
                    "started_utc": rb["started_utc"]}
        else:
            body = {"window": {"commit": root, "period": "p"} if v1 else {"root": root}, "pinned": PINNED,
                    "pins": LC.pins_seal(self.pins),
                    "host": f"{HOST}.{session}", "key_sha512": key_sha512 or self.pin,
                    "peer": "vy-nebius-1", "port": 7811, "scheme": scheme, "epoch": epoch,
                    "families": [list(f) for f in families], "minutes": 10, "started_utc": "2026-10-20T11:08:46Z",
                    **({"lane": by, "by": "research"} if v1 else
                       {"by": by, "secret_sha512": hashlib.sha512(bytes.fromhex(secret)).hexdigest()})}
        body.update(override)
        e = self.append("open", session, body, schema=LC.LEDGER_SCHEMA_V1 if v1 else LC.LEDGER_SCHEMA, leaf=leaf)
        self._opens[session] = body
        if reserved is None and not v1:
            self._join(body.get("for") or body["by"], session)
        return {"session": session, "port": body["port"], "peer": body["peer"], "host": body["host"], "seq": e["seq"],
                "root": root, "by": body["by"], **({"for": body["for"]} if "for" in body else {}),
                **({"reserve": body["reserve"], "opened_by": by} if reserved is not None else {}), "secret": secret,
                "commitment": None}

    def end(self, session: str, out: Path, closed: bool = True, **override) -> dict:
        """The end entry the gate's ``run`` appends over `out`, its fields replaced by `override`, signed after the leaf
        its record was signed at."""
        files = {n: hashlib.sha512((out / n).read_bytes()).hexdigest() for n in sorted(os.listdir(out))}
        doc = json.loads((out / R.RECORD).read_text())
        record = LC.ChallengerRecord.from_json(doc)
        try:
            released = {k: len(r) for k, r in R.released_roots(out / R.RELEASES, record.commitment).items()}
        except LC.CoinError:
            released = None
        families = self._opens[session]["families"]
        hellos = out / R.HELLOS
        body = {"rc": 0 if closed else 1, "closed": closed, "files": files,
                "ended": "closed by the client" if closed else "serve.py exited 1", "released": released,
                "complete": released is not None and all(released.get(k) == n for k, n in families),
                "record_leaf": LC.leaf_of(doc["signature"]), "commitment": record.commitment.root.hex(),
                "refused_hellos": len(hellos.read_text().splitlines()) if hellos.is_file() else 0,
                "ended_utc": "2026-10-20T11:08:53Z", **override}
        if type(body["record_leaf"]) is int:
            self._leaf = max(self._leaf, body["record_leaf"] + 1)
        self._complete[session] = body["complete"] is True
        return self.append("end", session, body)

    def ledger(self, pins=None, main=MAIN) -> PHC.Ledger:
        """The ledger as `pouw_h1_challenger.Ledger` reads it, under the host's pins log `pins` (its lines; by default
        the gate's own) and the verifier's copy of main's first-parent history `main` (None: no copy)."""
        return PHC.Ledger.of([self.lines], self.key, pins=[self.pins if pins is None else list(pins)], main=main)

    def pins_head(self, path: Path) -> Path:
        """The gate's pins log at `path`, as infra publishes it."""
        return write_pins(path, self.pins)


class Ledger:
    """The auditor's ledger at `path`, its entries signed under key `seed` at the next of `leaves` (shared with the
    challenger's own signatures under that key, of height `H`), naming key `named` in its opens (by default its own);
    `reserve` reserves a session, whose `open` then binds that reserve (the gate's two-step open). Its opens seal the
    host's pins log `PINS`, which it writes beside `path` (`pins_path`, infra's published ``pins.jsonl``). Its period
    opens sign the model registry `registry`'s registrations (`Registry.opened`)."""

    def __init__(self, path: Path, seed: int, leaves: Iterator[int] | None = None,
                 named: LC.ChallengerKey | None = None, H: int = 4, registry: Registry | None = None) -> None:
        self.path, self.lines = path, []
        self.registry = registry or Registry()
        self.pins_path = write_pins(path.parent / f"{path.stem}.pins.jsonl")
        self.private, self.key = challenger_key(seed, H)
        self.leaves = itertools.count() if leaves is None else leaves
        self.key_sha512 = hashlib.sha512((named or self.key).public).hexdigest()
        self._reserves: dict[str, int] = {}
        self._period: str | None = None
        self._windows: list[str] = []
        self._roots: dict[str, str] = {}
        self._complete: dict[str, bool] = {}

    def append(self, kind: str, session: str, body: dict, leaf: int | None = None) -> None:
        q = next(self.leaves) if leaf is None else leaf
        sign = lambda m: lms.sign_at(self.private, q, m).encode()  # noqa: E731
        self.lines.append(LC.ledger_line(self.lines, kind, session, body, self.key.public[:16], sign))
        self.path.write_text("".join(line + "\n" for line in self.lines))

    def period_open(self, name: str, *, circuit: str, cap: int = 1, draw_k: int | None = None, rows: int | None = None,
                    model: str | None = None, weights_root: str | None = None, circuit_files: dict | None = None,
                    length: dict | None = None, leaf: int | None = None) -> None:
        """Lane vllm's period `name`, holding the sessions reserved until its `period_close`, under `cap` and `circuit`,
        with ``draw_k`` (`credit.K_FIELD`), ``rows`` (the table rows T, `credit.ROWS_FIELD`), ``model`` (the gate's
        ``--model``: `registry`'s registration's weights' root,
        signed with its ``model``) or ``weights_root`` (``--weights-root``, Refused unless a registration's),
        ``circuit_files`` (each K's circuit file's SHA-512, ``{k<K>: SHA-512}``) and ``length`` (the gate's ``--length``,
        `credit.LENGTH_FIELD`) when given."""
        signed = self.registry.opened(circuit, model, weights_root)
        self._period, self._windows = name, []
        self.append("period-open", None, {"period": name, "lane": "vllm", "by": "vllm", "cap": cap, "circuit": circuit,
                                          **({CR.K_FIELD: draw_k} if draw_k is not None else {}),
                                          **({CR.ROWS_FIELD: rows} if rows is not None else {}), **signed,
                                          **({"circuit_files": circuit_files} if circuit_files is not None else {}),
                                          **({CR.LENGTH_FIELD: length} if length is not None else {}),
                                          "key_sha512": self.key_sha512}, leaf)

    def period_close(self, name: str, leaf: int | None = None) -> None:
        """The close of period `name`, listing its windows as core's check counts them."""
        ws = [{"root": self._roots.get(s), "session": s, "complete": self._complete.get(s, False)} for s in self._windows]
        self.append("period-close", None, {"period": name, "lane": "vllm", "by": "vllm", "windows": ws,
                                           "complete": bool(ws) and all(w["complete"] for w in ws)}, leaf)
        self._period = None

    def reserve(self, session: str, leaf: int | None = None) -> None:
        self._reserves[session] = len(self.lines)
        if self._period is not None:
            self._windows.append(session)
        self.append("reserve", session, {"by": "vllm", "minutes": 10, "started_utc": "2026-10-20T11:08:00Z",
                                         "expires_utc": "2026-10-20T11:18:00Z", "key_sha512": self.key_sha512}, leaf)

    def open(self, session: str, window: dict, leaf: int | None = None) -> None:
        reserved = ({"reserve": self._reserves[session], "minutes": 10, "opened_by": "vllm",
                     "started_utc": "2026-10-20T11:08:46Z"} if session in self._reserves else {})
        self.append("open", session, {"window": window, "host": f"{HOST}.{session}", "pinned": PINNED,
                                      "pins": LC.pins_seal(PINS),
                                      "key_sha512": self.key_sha512, "port": 7811, "by": "vllm", **reserved}, leaf)
        self._roots[session] = window.get("root")

    def end(self, session: str, directory: Path, closed: bool) -> None:
        files = {p.name: hashlib.sha512(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}
        record = json.loads((directory / R.RECORD).read_text())
        self._complete[session] = closed
        self.append("end", session, {"closed": closed, "files": files, "rc": 0 if closed else 1, "complete": closed,
                                     "record_leaf": int(record["signature"][:8], 16),
                                     "commitment": record["commitment"]["root"],
                                     "refused_hellos": len((directory / R.HELLOS).read_text().splitlines())})


class Host:
    """`challenger` in session `session` of `window` (`WINDOW` by default), its record and close signed under key
    `seed` at the next of `leaves` (hosts sharing a key share them) and written to `directory`, serving one client; the
    ledger `ledger` (a new one beside `directory` unless given) opens the session before its record and ends it once
    `ended`. When `gated` the ledger reserves the session at once (unless not `reserved`) and opens it only at `open`,
    the gate's open of the reserved session, for the window ``{root}`` the serving side names (an H1 window's open root,
    `pouw_h1_window.Window`), at the leaf held before the record's, since the ledger's check wants a record's leaf between
    its session's open's and end's."""

    def __init__(self, directory: Path, challenger: LC.LiveChallenger, seed: int, session: str = session_id(1),
                 leaves: Iterator[int] | None = None, ledger: Ledger | None = None, window: dict | None = None,
                 gated: bool = False, reserved: bool = True) -> None:
        self.directory, self.challenger, self.session_id = directory, challenger, session
        directory.mkdir()
        private, self.key = challenger_key(seed)
        leaves = itertools.count() if leaves is None else leaves
        self.ledger = ledger or Ledger(directory.parent / f"{directory.name}.jsonl", seed, leaves)
        self.window = None if gated else WINDOW if window is None else window
        if self.window is not None:
            self.ledger.open(session, self.window)
        elif reserved:
            self.ledger.reserve(session)
        self._open_leaf = next(leaves) if gated else None
        session_ = R.open_session(challenger, self.key, f"{HOST}.{session}",
                                  lambda m: lms.sign_at(private, next(leaves), m).encode(), directory, session)
        (directory / R.HELLOS).write_text("")
        ours, theirs = socket.socketpair()
        self.reason: str | None = None
        self._ended = False
        self._thread = threading.Thread(target=self._serve, args=(ours, session_))
        self._thread.start()
        self.client = R.RemoteChallenger(theirs)

    def _serve(self, sock: socket.socket, session: R.ChallengePhase) -> None:
        with sock:
            self.reason = R.serve(sock, session, itertools.count().__next__)

    def open(self, session: str, root: str) -> dict:
        """The gate's open of `session` on `root` (`pouw_h1_window.Gate`): the ledger opens it for ``{root}``, and
        the answer names its ledger entry, its root and the challenger's coin commitment."""
        self.window = {"root": root}
        self.ledger.open(session, self.window, self._open_leaf)
        return {"session": session, "seq": len(self.ledger.lines) - 1, "root": root,
                "commitment": self.challenger.commitment.root.hex()}

    def ended(self) -> str | None:
        """Join the session once its client has ended it, and the ledger ends it: None when the client closed it, else
        why it ended."""
        self._thread.join()
        if not self._ended:
            self._ended = True
            self.ledger.end(self.session_id, self.directory, self.reason is None)
        return self.reason
