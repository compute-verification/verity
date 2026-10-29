"""The window circuit's weights (`window_circuit`): each weight a port reads is its own registered parameter at the
port's row width (`window_builder.weight_param`: ``weights/<label>@<words>``), and a weight is a value like any other, its
rows committed under the committer's secret salts (`weight_value`), a hiding commitment.

The committer registers its weights once per model and key, before the first window's coins (`weights_registration`, a
`records.weights_registration`): each weight parameter's root, refused for a weight whose served words aren't the
checkpoint's pinned shards' (the checkpoint tie is the committer's own check). The verifier pins those roots and never a
root it computes: it holds the registration to its own weights record's shapes (`weights_record`, each weight parameter's
value, leaves, words and domain and no root; `registration_pins`), so what it learns of the weights is each weight's size
and its hiding root. `verity-vllm advice --weights-of-record` records the registration's roots as each window's, and a
window's committer commits the words it serves under the same key (`weight_values`), so a weight served otherwise
has another root and its reads are free (`interface.input_count`).

A field the weights record holds to an analytic table (the rotary table: its words are the engine's libm's, within a
bound of the analytic function of config.json, with no byte digest) is registered at the words the engine serves
(``tables``, by the record's table name), refused unless they are within that bound (`model.check_cos_sin`): the
committer's own check, as the checkpoint tie is for a digested weight. The verifier's record states its shape from its
declared one, so the verifier pins it like any weight; with no served words it stays unrooted and its reads free. That
the registered weights are a named checkpoint's, and the rotary table the analytic one's, is the committer's word until a
zero-knowledge registration proof ties them to the checkpoint's public shards and config.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.windows import interface as IF
from verity_vllm.check import weights_of_record as WR
from verity_vllm.protocols.pouw.h1 import registered as H1R
from verity_vllm.records.weights_registration import WeightsRegistration

from .window_builder import WEIGHTS, Grain, table_param, weight_param
from .window_circuit import WindowClass, class_program, lower

#: the class `weights_record` reads the weights record and the weight parameters of: every class's are the model's, at a grain
WEIGHTS_CLASS = WindowClass(1, 2, 2)
#: the salt nonce of the first weight parameter a committer commits under its key, each next one the next (`weight_nonces`):
#: above every nonce PoUW and the request tables take (`pouw.h1.served`, `pouw.h1.requests`, below 2^62)
WEIGHTS_NONCE = 1 << 62
_GRAIN = Grain()


def _bytes_le(words: Any) -> np.ndarray:
    """``words``' elements' little-endian bytes, in order."""
    a = np.ascontiguousarray(words)
    return np.ascontiguousarray(a, a.dtype.newbyteorder("<")).reshape(-1).view(np.uint8)


def _param(param: str) -> tuple[str, int, bool]:
    """Weight parameter ``param``'s weight, row width and whether it is an embedding's table: ``weights/<label>@<w>``
    (`window_builder.weight_param`) or ``weights/<label>@<w>/table`` (`window_builder.table_param`)."""
    body = param.removeprefix(f"{WEIGHTS}/")
    table = body.endswith("/table")
    label, _, w = body.removesuffix("/table").rpartition("@")
    if not w.isdigit() or int(w) < 1 or (table_param if table else weight_param)(label, int(w)) != param:
        raise ValueError(f"{param!r} is no weight parameter, {WEIGHTS}/<label>@<words> or its /table")
    return label, int(w), table


def weight_params(params: Any) -> list[str]:
    """The weight parameters among a circuit's registered ``params``, sorted: every one but a per-window value's (the scan
    state rows)."""
    return sorted(p for p in params if p.startswith(f"{WEIGHTS}/"))


def _rows(param: str, words: Any) -> np.ndarray:
    """Weight parameter ``param``'s rows (`_param`): ``words``, the weight's elements in the weights record's order, as their
    little-endian bytes' u16 words (`interface.words`) in rows of its width."""
    w = _param(param)[1]
    data = _bytes_le(words)
    if not data.size or data.size % (2 * w):
        raise ValueError(f"{param}: {data.size} bytes are no whole rows of {w} u16 words")
    return data.view("<u2").reshape(-1, w)


def _entry(program: str, param: str, words: int, registers: int) -> dict[str, Any]:
    """Weight parameter ``param``'s entry with no root, its weight ``words`` u16 words in rows of its width, as the circuit
    registers ``registers`` words of it."""
    w = _param(param)[1]
    if not words or words % w:
        raise ValueError(f"{param}: {words} words are no whole rows of {w} u16 words")
    if words != registers:
        raise ValueError(f"{param}: {words} words of record, and the circuit registers {registers}")
    return {"value": param, "leaves": words // w, "words": w, "domain": G.domain(program, param, w, words // w).domain_id.hex()}


def _analytic(rec: dict[str, Any], of: dict[str, list[str]]) -> list[dict[str, Any]]:
    """The weights record's fields held to an analytic table (``pinned: analytic``, the rotary table) that the circuit
    registers."""
    return [f for f in rec["fields"] if f.get("pinned") == "analytic" and of.get(f["field"])]


def _words(f: dict[str, Any]) -> int:
    """Field ``f``'s u16 words, from its declared shape and value bits."""
    return int(np.prod(f["shape"], dtype=np.int64)) * int(f["bits"]) // 16


def _table_words(f: dict[str, Any], served: Any) -> np.ndarray:
    """``served``, the words the engine serves of analytic field ``f``, as its table's u16 words; refused (ValueError) by
    name unless they are its declared shape's 16-bit words within the analytic table's bound (`check.weights_of_record`'s
    check of a served table, `model.check_cos_sin`)."""
    a = np.ascontiguousarray(served)
    if a.dtype.itemsize != 2:
        raise ValueError(f"served {a.dtype} words, not the table's 16-bit ones")
    rep = WR._check_analytic(f, a.view(np.uint16))
    if rep["status"] != "pass":
        raise ValueError(f"served words outside the analytic table's bound ({rep.get('why')})")
    return a.view(np.uint16)


def weight_value(program: str, param: str, words: Any, key: bytes, nonce: int, keep_rows: bool = False) -> G.Registered:
    """Weight parameter ``param``'s registered value: its rows (``words`` in rows of its width) committed as core's
    `registered.Registered` commits ``hm96-sha512/row/v2`` rows, or a table's ``hm96-sha512/row-seg/v1`` rows (the leaves
    its opening reads in-circuit, `gates.embedding`), with `pouw.h1.registered.commit` (on the GPU where this process has
    one) under ``program``, the weights record's program digest, row j salted `audit.hm96_salt(key, nonce, j)` of the
    committer's secret 32-byte ``key``: a hiding commitment, so its root says nothing of the weight to anyone without the
    key, a verifier holding a candidate checkpoint included. ``keep_rows``: hold the rows (a window's committer, which
    opens them)."""
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("a weight's salts are drawn under the committer's 32-byte key")
    return H1R.commit(program, param, _rows(param, words), key, int(nonce), keep_rows=keep_rows,
                      schema=G.SCHEMA_SEG if _param(param)[2] else G.SCHEMA_V2)


def weight_nonces(params: Any) -> dict[str, int]:
    """Each weight parameter's salt nonce under the committer's key: `WEIGHTS_NONCE` plus its place among ``params``'
    weight parameters (`weight_params`), so no two take one salt."""
    return {p: WEIGHTS_NONCE + i for i, p in enumerate(weight_params(params))}


def _fields(registered: Any) -> tuple[Any, list[tuple[str, list[int], int]]]:
    """The class Program the weights are read of and its weights record's fields (name, shape, value bits)."""
    from .common import _arr_shape

    r = class_program(registered, WEIGHTS_CLASS)
    return r, [(n, list(_arr_shape(t)), next(iter(t.width_set()))) for n, t in dict(r.program.fn.signature()[0])[WEIGHTS].fields]


def _weights(registered: Any, grain: Grain) -> tuple[Any, list[tuple[str, list[int], int]], dict[str, int], dict[str, list[str]]]:
    """`_fields`, the weight parameters the class circuit registers at ``grain`` (their words), wired or not, and those by
    field."""
    r, fields = _fields(registered)
    every = lower(registered, WEIGHTS_CLASS, None, grain, wired=False).circuit.registered.params
    params = {p: every[p] for p in weight_params(every)}
    of: dict[str, list[str]] = {}
    for p in params:
        of.setdefault(_param(p)[0], []).append(p)
    return r, fields, params, of


def _composed(registered: Any, r: Any, fields: list, checkpoint_dir: str, checkpoints_manifest: str | None,
              kv_cache_dtype: str, on_field: Any) -> dict[str, Any]:
    """`check.weights_of_record` of ``fields``, composed from ``checkpoint_dir``'s shards as ``checkpoints_manifest``
    (default the repository's ``manifests/checkpoints.json``) pins them at ``registered``'s model and revision, each field's
    words handed to ``on_field`` as digested."""
    from verity_vllm.config import CHECKPOINTS

    model = registered.phi.model
    ck = WR.CheckpointIndex(checkpoint_dir, checkpoints_manifest=str(checkpoints_manifest or CHECKPOINTS), model=model["repo"],
                            revision=model["revision"])
    return WR.weights_of_record({"instances": {}, "result": {"program_digest": r.digest}, "dir": None}, ck, prefix="",
                                kv_cache_dtype=kv_cache_dtype, fields=fields, on_field=on_field)


def _unrooted(rec: dict[str, Any], of: dict[str, list[str]], rooted: Any) -> dict[str, str]:
    """The weight parameters with no root, each with why: a field with no byte digest, or an analytic table whose served
    words weren't given."""
    why = {f["field"]: f.get("why") or ("held to the analytic table within a bound, no byte digest, and its served words not "
                                        "given" if f.get("pinned") == "analytic" else "no byte digest")
           for f in rec["fields"] if not f.get("digest")}
    return {p: why.get(label, "no field of the weights record") for label, ps in sorted(of.items()) for p in ps if p not in rooted}


def weights_record(registered: Any, checkpoint_dir: str, checkpoints_manifest: str | None = None,
                   kv_cache_dtype: str = "auto", grain: Grain = _GRAIN) -> dict[str, Any]:
    """The weights of record of the circuit's registered weights (`check.weights_of_record`), the verifier's: each field of
    the class Program's weights record, in its order, composed from the checkpoint's pinned shards (``checkpoints_manifest``,
    default the repository's ``manifests/checkpoints.json``) at ``registered``'s model and revision, and each weight
    parameter the class circuit registers at ``grain`` (``weights/<label>@<words>``, one a weight and row width) its
    entry's shape, ``registered_entries``: core's `registered.Registered.entry()` with no root (its value, leaves, words
    and domain under the record's program digest), what the verifier holds the committer's registration to
    (`registration_pins`). No root: a weight's root is the committer's, under its secret salts (`weights_registration`).
    A field held to an analytic table (the rotary table, within a bound of config.json's function) has its entry from
    its declared shape, the verifier never reading its words; a parameter whose field has neither is ``unrooted`` with
    why, and its reads stay free. The weights record is the model's, one at every class; its parameters are the grain's,
    which the record names."""
    r, fields, params, of = _weights(registered, grain)
    shapes: dict[str, dict[str, Any]] = {}

    def shape(name: str, words: np.ndarray) -> None:
        for p in of.get(name, ()):
            shapes[p] = _entry(r.digest, p, _rows(p, words).size, params[p])

    rec = _composed(registered, r, fields, checkpoint_dir, checkpoints_manifest, kv_cache_dtype, shape)
    for f in _analytic(rec, of):
        for p in of[f["field"]]:
            shapes[p] = _entry(r.digest, p, _words(f), params[p])
    return {**rec, "grain": asdict(grain), "registered_entries": dict(sorted(shapes.items())),
            "unrooted": _unrooted(rec, of, shapes)}


def weight_values(registered: Any, served: Any, key: bytes, grain: Grain = _GRAIN, keep_rows: bool = False) -> dict[str, G.Registered]:
    """The weights a committer serves, committed (a window's committer or the registration's): each weight
    parameter the class circuit registers at ``grain`` whose field ``served`` names (its words, as `check.weights_of_record`
    composes the field), its `weight_value` under ``key`` at its nonce (`weight_nonces`), its rows held with ``keep_rows``.
    So a window's root for a weight is the registered one (`weights_registration`) exactly when it serves the registered
    words, and the verifier's pin holds the reads of no other (`interface.input_count`)."""
    r, _fields, params, of = _weights(registered, grain)
    nonces = weight_nonces(params)
    return {p: weight_value(r.digest, p, served[name], key, nonces[p], keep_rows) for name in sorted(of) if name in served
            for p in of[name]}


def record_words(registered: Any, checkpoint_dir: str, checkpoints_manifest: str | None = None,
                 kv_cache_dtype: str = "auto") -> dict[str, np.ndarray]:
    """Each field of the class Program's weights record composed from ``checkpoint_dir``'s shards (`check.weights_of_record`),
    by field, its words as digested: the words a committer serving that checkpoint serves (`weight_values`), where it reads
    them from the checkpoint rather than the engine's tensors."""
    r, fields = _fields(registered)
    out: dict[str, np.ndarray] = {}
    _composed(registered, r, fields, checkpoint_dir, checkpoints_manifest, kv_cache_dtype,
              lambda name, words: out.__setitem__(name, np.array(words)))
    return out


def weights_registration(registered: Any, key: bytes, checkpoint_dir: str, served: Any = None,
                         checkpoints_manifest: str | None = None, kv_cache_dtype: str = "auto",
                         grain: Grain = _GRAIN, tables: Any = None) -> dict[str, Any]:
    """The committer's registration of the weights it serves (`records.weights_registration`), written once per model and
    key before the first window's coins and pinned by the verifier (core's ``registration.check(pinned=...)``,
    ``records_audit.py --registered`` and ``--pins``, `registration_pins`): each weight parameter the class circuit
    registers at ``grain`` its `weight_value` under ``key`` at its nonce (`weight_nonces`), its entry core's
    `registered.Registered.entry()`. The key never leaves the committer: the registration names no salt.

    The checkpoint tie is the committer's own: each field is composed from the checkpoint's pinned shards, and a field
    ``served`` (the words the committer serves, by field, as `check.weights_of_record` composes them) names other words
    for, or a field with a root it doesn't name, is refused (ValueError) by name, as is a checkpoint not of record or one
    whose manifest entry is another model or revision than ``registered``'s. With no ``served``, what it registers is the
    checkpoint's own words.

    A field the weights record holds to an analytic table is registered at ``tables``' words for that table (the words the
    engine serves, by the record's table name: ``cos_sin``, the rotary table), refused by name unless they are within its
    bound (`_table_words`); a table ``tables`` doesn't name leaves its parameters unrooted."""
    r, fields, params, of = _weights(registered, grain)
    nonces = weight_nonces(params)
    entries: dict[str, dict[str, Any]] = {}
    refused: dict[str, str] = {}

    def register(name: str, words: np.ndarray) -> None:
        for p in of[name]:
            v = weight_value(r.digest, p, words, key, nonces[p])
            if v.words * v.leaves != params[p]:
                raise ValueError(f"{p}: {v.words * v.leaves} words of record, and the circuit registers {params[p]}")
            entries[p] = v.entry()

    def commit(name: str, words: np.ndarray) -> None:
        if not of.get(name):
            return
        if served is not None and name not in served:
            refused[name] = "not served"
            return
        if served is not None and not np.array_equal(_bytes_le(served[name]), _bytes_le(words)):
            refused[name] = "served words that aren't the checkpoint's"
            return
        register(name, words)

    rec = _composed(registered, r, fields, checkpoint_dir, checkpoints_manifest, kv_cache_dtype, commit)
    for f in _analytic(rec, of):
        if (tables or {}).get(f["analytic"]["table"]) is None:
            continue
        try:
            words = _table_words(f, tables[f["analytic"]["table"]])
        except ValueError as e:
            refused[f["field"]] = str(e)
            continue
        register(f["field"], words)
    model = registered.phi.model
    if rec.get("of_record") is not True:
        why = rec.get("why") or f"shards pinned: {rec['checkpoint'].get('all_shards_pinned')}"
        raise ValueError(f"the checkpoint is not of record ({why}): a committer registers the weights of its pinned shards")
    entry = rec["checkpoint"]["entry"]
    if (entry["repo"], entry["revision"]) != (model["repo"], model["revision"]):
        raise ValueError(f"the checkpoint's manifest entry is {entry['repo']}@{entry['revision']}, not the registered model "
                         f"{model['repo']}@{model['revision']}: a committer registers only its own model's weights")
    if refused:
        raise ValueError(f"no weight registered: {refused} (a committer registers a weight only as the checkpoint's pinned "
                         "shards hold it, an analytic table only within its bound)")
    return WeightsRegistration(program=r.digest, repo=model["repo"], revision=model["revision"], grain=asdict(grain),
                               values=entries, unrooted=_unrooted(rec, of, entries)).dump()


def registration_pins(registration: Any, record: dict[str, Any]) -> dict[str, str]:
    """The verifier's pins of the weights, ``{param: root}``: the committer's registration's roots (`weights_registration`,
    read as `records.weights_registration` reads it), each held to the shape the verifier's own weights record states for it
    (`weights_record`'s ``registered_entries``: its value, leaves, words and domain, which binds the record's program).
    Refused (ValueError) unless the registration registers exactly the record's weight parameters, each at its shape, but
    an analytic table's it names unrooted (its served words not given), whose reads stay free."""
    reg, want = WeightsRegistration.of(registration, "the weights registration"), record.get("registered_entries")
    if not isinstance(want, dict):
        raise ValueError("a weights registration is held to a weights record's registered_entries, and the record states none")
    analytic = {f["field"] for f in record.get("fields", ()) if f.get("pinned") == "analytic"}
    unrooted = {p for p in reg.unrooted if _param(p)[0] in analytic}
    if odd := sorted((set(reg.values) - set(want)) | (set(want) - set(reg.values) - unrooted)):
        raise ValueError(f"the registration registers {odd} unlike the record")
    if bad := sorted(p for p, e in reg.values.items() if {k: v for k, v in e.items() if k != "root"} != want[p]):
        raise ValueError(f"the registration's entries for {bad} aren't the record's shapes")
    pins = reg.roots
    IF.held_params(pins, {})
    return pins


__all__ = ["WEIGHTS_CLASS", "WEIGHTS_NONCE", "Grain", "WeightsRegistration", "record_words", "registration_pins", "weight_nonces",
           "weight_value", "weight_values", "weights_record", "weights_registration"]
