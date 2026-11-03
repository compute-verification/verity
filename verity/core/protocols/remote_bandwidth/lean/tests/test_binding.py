"""`store.py`'s binding held to its Lean (`RemoteAudit.Store`, served by `remote-audit-difftest`), refusals as outputs
so that both sides' reasons are compared word for word: the pin from a pin file's bytes (`store.pin_of`, and
`model_pin` for `--model`) in each shape it reads (a pin, a registry line, a registry's JSON lines, core's
`models.json`, and that file itself) and each refusal (not UTF-8, neither JSON nor JSON lines, a registry without one
checkpoint of the role, a role core's pin lacks, a malformed pin, a `checkpoint_sha512` that isn't its files', no
config.json or no .safetensors file, another model's pin); canonical JSON and `checkpoint_sha512`; the payload salts;
the expected root from the weights chunks' digests, held also to `audit.payload_tree` over the store's payloads; and
`Binding.hold` on hellos of the pinned model's store and on each refusal (a store short of the arenas, another
layout, a malformed node, the root under another key, of another checkpoint, of another size, or not a root), with
the expected root its record keeps; and `binding`, the binding as the verifier's Lean process starts it (`bind`: the
pin from its file's bytes, then the store's shape) and holds each hello to it (`hold`), on the honest binding and each
refusal. The verifier decides `bound` with the Lean's `bind` and `hold` (`benchmarks/remote_session/verifier.py`), so
the Lean is of record for the binding, the hello's decision and the expected root; the pin's other readers
(`read_pin`) are Python's."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import ra_cases as M
import verity_difftest as D
from verity.core.protocols.remote_bandwidth import audit, codec, store

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

MODELS_JSON = Path(__file__).resolve().parents[3] / "pouw" / "lean" / "check" / "models.json"
ROLE = "SYNTHETIC_BF16"
NODE = b"node-under-test"
KEY = bytes(range(32))


OF_RECORD = {"binding": "lean", "hold": "lean", "expected_root": "lean"}


def port(fn, reference, inputs=st.nothing(), need_rejects: bool = True) -> D.Port:
    return D.Port(M.EXE, fn, reference, inputs, package=M.PACKAGE, of_record=OF_RECORD.get(fn, "python"),
                  rejects=M.REFUSALS, need_rejects=need_rejects)


server = D.server_fixture(port("pin", lambda v: v))


def expand(seed: bytes, n: int) -> bytes:
    return b"".join(hashlib.sha512(seed + c.to_bytes(4, "big")).digest() for c in range(-(-n // 64)))[:n]


def refused(why: str) -> tuple[str, str]:
    return "output", D.canonical({"refused": why})


# --- the pin ----------------------------------------------------------------------------------------------------------

def pin(v) -> dict:
    data, path, role, model = M.hex_(v, "data"), M._str(v, "path"), M._get(v, "role"), M._bool(v, "model")
    if role is not None and not isinstance(role, str):
        raise D.Reject("role: not a string")
    if model and role is None:
        raise D.Reject("a binding's pin is a role's")
    try:
        return {"pin": (store.model_pin if model else store.pin_of)(data, path, role)}
    except store.Refused as e:
        return {"refused": str(e)}


def files_of(seed: bytes, shards: int = 2, extra: tuple[str, ...] = ()) -> dict:
    names = ["config.json", *(f"model-{s + 1:05d}-of-{shards:05d}.safetensors" for s in range(shards)), *extra]
    return {n: hashlib.sha512(seed + n.encode()).hexdigest() for n in names}


def a_pin(files: dict, name: str = ROLE, **over) -> dict:
    return {"name": name, "checkpoint": files, "checkpoint_sha512": store.checkpoint_sha512(files)} | over


def core(files: dict, roles=(ROLE, "OTHER_BF16")) -> dict:
    return {r: {"checkpoint_sha512": store.checkpoint_sha512(fs), "files": fs, "recipe": "bf16/v1",
                "repo": f"example/{r}", "revision": "0" * 40}
            for r, fs in zip(roles, (files, files_of(b"other")))}


def lines(*docs) -> str:
    return "".join(json.dumps(d) + "\n" for d in docs)


def case(text: str | bytes, role: str | None = ROLE, model: bool = False, path: str = "pins/pin.json") -> dict:
    data = text.encode() if isinstance(text, str) else text
    return {"data": data.hex(), "path": path, "role": role, "model": model}


NAMES = st.sampled_from(["config.json", "tokenizer.json", "model.safetensors", "é.safetensors", "a b.json",
                         "model-00001-of-00002.safetensors", "\U0001f600.bin", 'q"\\.json'])


@st.composite
def pin_inputs(draw) -> dict:
    """A pin in one of the shapes `pin_of` reads, written as JSON in one of several ways, as it is or with one fault."""
    files = files_of(draw(st.binary(max_size=4)), draw(st.integers(1, 3)))
    files |= {n: hashlib.sha512(n.encode()).hexdigest() for n in draw(st.lists(NAMES, max_size=2))}
    role = draw(st.sampled_from([ROLE, "llama31-8b", "é-model"]))
    shape = draw(st.sampled_from(["pin", "line", "registry", "core"]))
    f = draw(st.integers(0, 24))
    if f == 1:
        files = {**files, draw(st.sampled_from(["", "a/b", ".hidden", "../x"])): "ab" * 64}
    elif f == 2:
        files = {**files, "config.json": draw(st.sampled_from(["AB" * 64, "ab" * 63, "", 7, None]))}
    elif f == 3:
        files = {k: h for k, h in files.items() if k != "config.json"}
    elif f == 4:
        files = {k: h for k, h in files.items() if not k.endswith(".safetensors")}
    elif f == 20:
        files = {}
    doc: dict = {"pin": a_pin(files, role), "line": {"model": role} | a_pin(files, role), "registry": {},
                 "core": core(files, (role, "OTHER_BF16"))}[shape]
    if f == 5 and shape in ("pin", "line"):
        doc[draw(st.sampled_from(["checkpoint_sha512", "name", "checkpoint"]))] = draw(
            st.sampled_from(["", "cd" * 64, 1, None, [], {}]))
    elif f == 6 and shape in ("pin", "line"):
        del doc[draw(st.sampled_from(sorted(doc)))]
    elif f == 7 and shape == "core":
        doc[role]["checkpoint_sha512"] = draw(st.sampled_from(["cd" * 64, 2, None]))
    elif f == 8 and shape == "core":
        doc[role] = draw(st.sampled_from([{}, {"files": files}, "x"]))
    if shape == "registry":
        mine = {"model": role, "checkpoint": files, "checkpoint_sha512": store.checkpoint_sha512(files)}
        other = {"model": "other", "checkpoint": files_of(b"o"), "checkpoint_sha512": "00" * 64}
        rows = draw(st.permutations([mine, other] + [mine] * draw(st.integers(0, 2))))
        if f == 9:
            rows = [r for r in rows if r is not mine]
        elif f == 10:
            rows = rows + [{"model": role, "checkpoint": draw(st.sampled_from([files_of(b"x"), None, 3, [files]]))}]
        elif f == 11:
            rows = rows + [{"name": role, "model": None, "checkpoint": files}]
        text = "".join(json.dumps(r) + draw(st.sampled_from(["\n", "\r\n", "\n\n", "\n \t\n"])) for r in rows)
        if f == 12:
            text += draw(st.sampled_from(["{", "nope\n", "[1,\n", "NaN\n", '"\\ud800"\n']))
    else:
        text = json.dumps(doc, indent=draw(st.sampled_from([None, 1])), ensure_ascii=draw(st.booleans()),
                          sort_keys=draw(st.booleans()))
    if f == 13:
        text = text[:draw(st.integers(0, max(len(text) - 1, 0)))]
    elif f == 14:
        text = text.replace('"' + role + '"', '"' + role + '\\ud800"', 1)
    elif f == 15:
        text = text.replace('"' + role + '"', '"' + role + '\\ud83d\\ude00"', 1)
    elif f == 16:
        text = draw(st.sampled_from(["\ufeff", " \t\r\n", "[", "1e5 ", "NaN", "null", "1e99999999999999999999 "])) + text
    elif f == 17:
        text = text.replace("{", '{"x": 1e400, "y": -0.5e-99999999999999999999999, ', 1)
    elif f == 18 and text.endswith("}"):
        text = text[:-1] + ', "checkpoint_sha512": "' + "ef" * 64 + '"}'
    data = text.encode()
    if f == 19:
        data = draw(st.sampled_from([b"\xff", b"\xc0\x80", b"\xed\xa0\x80"])) + data
    model = draw(st.booleans())
    want = draw(st.sampled_from([role, role, "llama31-8b", None]))
    if model and want is None:
        want = role
    return {"data": data.hex(), "path": draw(st.sampled_from(["pin.json", "/m/models.jsonl", "pins/é.json"])),
            "role": want, "model": model}


def test_pin(server) -> None:
    stats = D.run(port("pin", pin, pin_inputs(), need_rejects=False), examples=400, server=server)
    assert stats.outputs > 300, stats


def test_cores_models_json_is_read_as_each_of_its_roles_pins(server) -> None:
    """Core's pin itself, read for each of its roles, for --model and not, and for a role it lacks."""
    data = MODELS_JSON.read_bytes()
    doc = json.loads(data)
    assert doc
    for role in sorted(doc):
        for model in (False, True):
            got = D.replay(port("pin", pin), case(data, role, model, str(MODELS_JSON)), server)
            assert got == ("output", D.canonical({"pin": {"name": role, "checkpoint": doc[role]["files"],
                                                         "checkpoint_sha512": doc[role]["checkpoint_sha512"]}}))
    roles = json.dumps(sorted(doc), separators=(",", ":"))
    for role, named in (("NO_SUCH_ROLE", '"NO_SUCH_ROLE"'), (None, "null")):
        assert D.replay(port("pin", pin), case(data, role, path="models.json"), server) == refused(
            f"the pin models.json pins no role {named}: its roles are {roles}")


def test_each_refusal_of_a_pin_is_the_leans_word_for_word(server) -> None:
    """One pin a refusal, each replayed through both sides: the reason is the same text, and it is the refusal meant."""
    files = files_of(b"f")
    good = a_pin(files)
    digest = store.checkpoint_sha512(files)
    no_config = {k: h for k, h in files.items() if k != "config.json"}
    form = "{name, checkpoint: {file: SHA-512 hex}, checkpoint_sha512}"
    p = "pins/pin.json"
    cases = [
        (case(b"\xff" + json.dumps(good).encode()), f"the pin {p} is not UTF-8"),
        (case("{\n\"name\": 1,\n"), f"the pin {p} is neither JSON nor JSON lines: its line 1 is not JSON"),
        (case(lines(good) + "\n  \nNaN\n"), f"the pin {p} is neither JSON nor JSON lines: its line 4 is not JSON"),
        (case(json.dumps(good).replace('"name"', '"na\\udc00me"')),
         f"the pin {p} is neither JSON nor JSON lines: its line 1 is not JSON"),
        (case(lines({"model": "x"} | good, {"model": "y"} | good)),
         f"the registry {p} has 0 lines of {ROLE}, pinning 0 checkpoints: one is the pin"),
        (case(lines({"model": ROLE} | good, {"model": ROLE} | a_pin(files_of(b"g")))),
         f"the registry {p} has 2 lines of {ROLE}, pinning 2 checkpoints: one is the pin"),
        (case(lines(good, good), role=None), f"the registry {p} has 0 lines of None, pinning 0 checkpoints: one is the pin"),
        (case(json.dumps(core(files)), role="X"),
         f'the pin {p} pins no role "X": its roles are ["OTHER_BF16","{ROLE}"]'),
        (case(json.dumps({"é\U0001f600": {}})), f'the pin {p} pins no role "{ROLE}": its roles are ["\\u00e9\\ud83d\\ude00"]'),
        (case(json.dumps(good | {"name": ""})), f"the pin {p} is not {form} with a file name (no directory) a file"),
        (case(json.dumps(good | {"checkpoint": {**files, "a/b": "ab" * 64}})),
         f"the pin {p} is not {form} with a file name (no directory) a file"),
        (case(json.dumps(good | {"checkpoint": {**files, "config.json": "AB" * 64}})),
         f"the pin {p} is not {form} with a file name (no directory) a file"),
        (case(json.dumps({ROLE: {"files": {}}, "B": {}})),
         f"the pin {p} is not {{ROLE: {{checkpoint_sha512, files: {{file: SHA-512 hex}}, ...}}}} with a file name (no "
         "directory) a file"),
        (case(json.dumps(good | {"checkpoint_sha512": "cd" * 64})),
         f"the pin {p}'s checkpoint_sha512 is not the SHA-512 of its checkpoint's canonical JSON ({digest[:16]}...)"),
        (case(json.dumps(good)[:-1] + ', "checkpoint_sha512": "' + "ef" * 64 + '"}'),
         f"the pin {p}'s checkpoint_sha512 is not the SHA-512 of its checkpoint's canonical JSON ({digest[:16]}...)"),
        (case(json.dumps(a_pin(no_config))),
         f"the pin {p} pins no config.json or no .safetensors file: the layout reads both"),
        (case(json.dumps(a_pin(files, "other")), model=True), f"the pin {p} is other's, not --model {ROLE}'s"),
    ]
    for v, why in cases:
        assert D.replay(port("pin", pin), v, server) == refused(why), why
    for v in (case(json.dumps(good)), case(lines({"model": ROLE} | good, {"model": "o"} | a_pin(no_config))),
              case(json.dumps(core(files))), case("\r\n" + lines(good)[:-1] + "\r\n\r\n", model=True)):
        assert D.replay(port("pin", pin), v, server) == ("output", D.canonical({"pin": good}))


# --- canonical JSON, the checkpoint's digest and the salts --------------------------------------------------------------

def _has_number(v) -> bool:
    if isinstance(v, dict):
        return any(_has_number(x) for x in v.values())
    if isinstance(v, list):
        return any(_has_number(x) for x in v)
    return isinstance(v, int) and not isinstance(v, bool)


def canonical_json(v) -> str:
    value = M._get(v, "value")
    if _has_number(value):
        raise D.Reject("a value with a number")
    return store.canonical_json(value).decode()


TEXT = st.text(max_size=6) | st.sampled_from(['"\\', "\x00\x1f\x7f", "\u2028é", "\U0001f600\uffff"])
VALUES = st.recursive(st.none() | st.booleans() | TEXT,
                      lambda c: st.lists(c, max_size=3) | st.dictionaries(TEXT, c, max_size=3), max_leaves=8)


def test_canonical_json(server) -> None:
    inputs = st.one_of(VALUES, st.sampled_from([[1], {"a": [None, 0]}])).map(lambda x: {"value": x})
    stats = D.run(port("canonical_json", canonical_json, inputs), examples=300, server=server)
    assert stats.outputs > 250, stats


def checkpoint(v) -> dict:
    files = M._get(v, "files")
    if not isinstance(files, dict) or any(not isinstance(h, str) for h in files.values()):
        raise D.Reject("files is an object of strings")
    return {"canonical": store.canonical_json(files).decode(), "sha512": store.checkpoint_sha512(files)}


def test_checkpoint_sha512(server) -> None:
    files = st.dictionaries(TEXT | NAMES, st.sampled_from(["ab" * 64, "", "é"]) | TEXT, max_size=5)
    inputs = st.one_of(files, files.map(lambda f: {**f, "x": 1}), st.just([])).map(lambda f: {"files": f})
    stats = D.run(port("checkpoint_sha512", checkpoint, inputs), examples=200, server=server)
    assert stats.outputs > 80, stats


def store_salt(v) -> str:
    key, label, c = M.hex_(v, "key"), M.hex_(v, "label"), M.nat(v, "c")
    if c >= 2 ** 64:
        raise D.Reject("a block index is a u64")
    return store.store_salt(key, label, c).hex()


def test_store_salt(server) -> None:
    inputs = st.fixed_dictionaries({
        "key": st.binary(max_size=300).map(bytes.hex) | st.just(KEY.hex()),
        "label": st.just(store.PAYLOAD_LABEL.hex()) | st.binary(max_size=8).map(bytes.hex),
        "c": st.integers(0, 2 ** 64 - 1) | st.sampled_from([0, 2 ** 64, -1])})
    stats = D.run(port("store_salt", store_salt, inputs), examples=150, server=server)
    assert stats.outputs > 60, stats


def test_store_salt_at_every_message_length_modulo_a_sha512_block(server) -> None:
    """Keys of 0 to 128 bytes make the salt's message every length modulo SHA-512's 128-byte block, so the Lean's
    counter mode (`Unboxed.xof`) meets each boundary its shared blocks, its counter and its padding can cross, which
    random keys rarely reach."""
    for n in range(129):
        value = {"key": bytes(range(n)).hex(), "label": store.PAYLOAD_LABEL.hex(), "c": 2 ** 64 - 1}
        assert D.replay(port("store_salt", store_salt), value, server)[0] == "output"


# --- the expected root and the hello's decision -----------------------------------------------------------------------

def _digests(v) -> list[bytes]:
    out = M.hexes(v, "digests")
    if not out or any(len(d) != 64 for d in out):
        raise D.Reject("an arena's weights are one or more blocks' digests of 64 bytes")
    return out


def _arena(digests: list[bytes]) -> store.Arena:
    return store.Arena("", (), len(digests))


def _write(tmp: Path, digests: list[bytes]) -> Path:
    path = tmp / "weights.sha512rows"
    path.write_bytes(b"".join(digests))
    return path


def expected_root(tmp: Path):
    """`store.expected_root` in this process, its subtrees of `2^chunk_log2` leaves (the Lean's on `workers` threads)."""
    def reference(v) -> str:
        digests = _digests(v)
        engines, arena_blocks, host = M.nat(v, "engines"), M.nat(v, "arena_blocks"), M.nat(v, "host_blocks")
        node, key, chunk_log2 = M.hex_(v, "node"), M.hex_(v, "key"), M.nat(v, "chunk_log2")
        M.nat(v, "workers")
        s = store.Store(_arena(digests), engines, arena_blocks, host)
        return store.expected_root(s, node, key, _write(tmp, digests), 1, chunk_log2).hex()
    return reference


@st.composite
def shapes(draw) -> dict:
    w = draw(st.integers(1, 4))
    return {"digests": [expand(draw(st.binary(max_size=4)), 64).hex() for _ in range(w)],
            "engines": draw(st.integers(1, 3)), "arena_blocks": w + draw(st.integers(0, 3)),
            "host_blocks": draw(st.integers(0, 5)), "node": NODE.hex(), "key": KEY.hex()}


@st.composite
def root_inputs(draw) -> dict:
    v = draw(shapes()) | {"workers": draw(st.integers(0, 4)), "chunk_log2": draw(st.integers(0, 6))}
    f = draw(st.integers(0, 12))
    if f == 1:
        v["engines"] = 0
    elif f == 2:
        v["arena_blocks"] = len(v["digests"]) - 1
    elif f == 3:
        v["digests"] = draw(st.sampled_from([[], ["00" * 63], v["digests"] + ["00" * 65]]))
    elif f == 4:
        v["node"] = draw(st.sampled_from(["", "00" * 65, "07", "ab" * 64]))
    elif f == 5:
        v["key"] = draw(st.binary(max_size=40)).hex()
    return v


def test_expected_root(server, tmp_path) -> None:
    stats = D.run(port("expected_root", expected_root(tmp_path), root_inputs()), examples=150, server=server)
    assert stats.outputs > 60, stats


@pytest.mark.parametrize("engines, extra, host, workers, chunk_log2",
                         [(1, 0, 0, 1, 16), (2, 3, 5, 3, 1), (3, 1, 2, 2, 2), (3, 1, 2, 8, 0)])
def test_the_expected_root_is_the_payload_trees_over_the_stores_payloads(server, engines, extra, host, workers,
                                                                         chunk_log2) -> None:
    """Independently of `store.expected_root`: `audit.payload_tree` over the store's payloads, chunk j salted with
    `store_salt(key, PAYLOAD_LABEL, j)`, is the Lean's root from the weights chunks' digests alone, whole or as
    subtrees of `2^chunk_log2` leaves on `workers` threads."""
    weights = [expand(b"weights %d" % u, codec.BLOCK_BYTES) for u in range(3)]
    arena = weights + [bytes(codec.BLOCK_BYTES)] * extra
    payloads = arena * engines + [bytes(codec.BLOCK_BYTES)] * host
    salts = [store.store_salt(KEY, store.PAYLOAD_LABEL, j) for j in range(len(payloads))]
    root = audit.payload_tree(NODE, payloads, salts).commitment.root
    v = {"digests": [codec.digest(w).hex() for w in weights], "engines": engines, "arena_blocks": len(arena),
         "host_blocks": host, "node": NODE.hex(), "key": KEY.hex(), "workers": workers, "chunk_log2": chunk_log2}
    lean = D.replay(port("expected_root", lambda _v: root.hex()), v, server)
    assert lean == ("output", D.canonical(root.hex()))


def _region(r) -> tuple:
    if not isinstance(r, list) or len(r) != 4:
        raise D.Reject("a region is [kind, tier, first, blocks]")
    kind, tier, first, n = r
    if kind not in ("weights", "kv", "filler", "activations") or tier not in ("hbm", "host"):
        raise D.Reject("a region's kind and tier are regions_of's")
    return kind, tier, M._int(first, "first"), M._int(n, "blocks")


def _hello(v) -> tuple:
    """A hello's store as the Lean's `helloOf` reads it, in its order."""
    layout = None if M._get(v, "layout") is None else tuple(_region(r) for r in M._list(v, "layout"))
    return M.hex_(v, "node"), M.int_(v, "blocks"), M.hex_(v, "payload_root"), M.hex_(v, "key"), layout


def _binding(tmp: Path, role: str, digests: list[bytes], engines: int, arena_blocks: int) -> store.Binding:
    """A `Binding` of `role` over an arena of the weights chunks' digests `digests`, past what it reads of the files,
    once `Store`'s check of its arenas passed."""
    b = object.__new__(store.Binding)
    b.role, b.arena, b.engines, b.arena_blocks, b.workers = role, _arena(digests), engines, arena_blocks, 1
    b.digests, b.state = _write(tmp, digests), {}
    return b


def _held(b: store.Binding, hello: tuple) -> dict:
    """`b.hold(hello)` as the Lean's `hold` answers it: the regions, or why not, and the expected root kept."""
    node, blocks, root, key, layout = hello
    try:
        return {"regions": [list(r) for r in b.hold(node, blocks, root, key, layout)],
                "expected_root": b.state["expected_root"]}
    except ValueError as e:
        return {"refused": str(e), "expected_root": b.state.get("expected_root")}


def hold(tmp: Path):
    """`Binding.hold` of a binding of the input's role over an arena of its digests, as `Binding` starts it (`Store`'s
    check of its arenas) without reading a checkpoint."""
    def reference(v) -> dict:
        role, digests = M._str(v, "role"), _digests(v)
        engines, arena_blocks = M.int_(v, "engines"), M.int_(v, "arena_blocks")
        hello = _hello(v)
        try:
            store.Store(_arena(digests), engines, arena_blocks)
        except store.Refused as e:
            return {"refused": str(e), "expected_root": None}
        return _held(_binding(tmp, role, digests, engines, arena_blocks), hello)
    return reference


def _regions(engines: int, arena_blocks: int, w: int, host: int) -> list[list]:
    return [[r["kind"], r["tier"], r["first"], r["blocks"]]
            for r in store.Store(store.Arena("", (), w), engines, arena_blocks, host).regions()]


def _root(tmp: Path, v: dict, key: bytes, host: int) -> str:
    digests = [bytes.fromhex(d) for d in v["digests"]]
    s = store.Store(_arena(digests), v["engines"], v["arena_blocks"], host)
    return store.expected_root(s, NODE, key, _write(tmp, digests)).hex()


def hold_inputs(tmp: Path):
    @st.composite
    def inputs(draw) -> dict:
        """The pinned model's store under the verifier's key, with its layout or none, as it is or with one fault."""
        v = draw(shapes())
        w, engines, arena_blocks, host = len(v["digests"]), v["engines"], v["arena_blocks"], v.pop("host_blocks")
        v |= {"role": draw(st.sampled_from([ROLE, "é"])), "blocks": engines * arena_blocks + host,
              "layout": draw(st.sampled_from([None, _regions(engines, arena_blocks, w, host)]))}
        f = draw(st.integers(0, 20))
        root_key, root_host = KEY, host
        if f == 1:
            v["blocks"] = engines * arena_blocks - draw(st.integers(1, 3))
        elif f == 2:
            v["layout"] = draw(st.sampled_from([[], _regions(engines, arena_blocks + 1, w, host),
                                                _regions(engines, arena_blocks, w, host)[:-1],
                                                [["kv", *r[1:]] for r in _regions(engines, arena_blocks, w, host)]]))
        elif f == 3:
            v["node"] = draw(st.sampled_from(["", "00" * 65]))
        elif f == 4:
            root_key = bytes(32)
        elif f == 5:
            v["blocks"] += draw(st.integers(1, 3))
        elif f == 6:
            v["engines"] = draw(st.sampled_from([0, -1]))
        elif f == 7:
            v["arena_blocks"] = w - draw(st.integers(1, 2))
        v["payload_root"] = _root(tmp, v | {"engines": max(engines, 1), "arena_blocks": max(arena_blocks, w)},
                                  root_key, root_host)
        if f == 8:
            v["payload_root"] = draw(st.sampled_from(["00" * 64, v["payload_root"][:-2] + "00", "ab" * 32, ""]))
        elif f == 9:
            v["digests"] = v["digests"][::-1] if w > 1 else [expand(b"other", 64).hex()]
        return v
    return inputs()


def test_hold(server, tmp_path) -> None:
    stats = D.run(port("hold", hold(tmp_path), hold_inputs(tmp_path), need_rejects=False), examples=250,
                  server=server)
    assert stats.outputs > 200, stats


def test_each_decision_of_a_hello_is_the_leans_word_for_word(server, tmp_path) -> None:
    """A hello of the pinned model's store bound, with and without its layout, and one a refusal, each replayed through
    both sides: the regions or the reason are the same text, and the record's expected root the same."""
    digests = [expand(b"w%d" % u, 64).hex() for u in range(6)]
    base = {"role": ROLE, "digests": digests, "engines": 2, "arena_blocks": 7, "blocks": 16, "node": NODE.hex(),
            "key": KEY.hex(), "layout": None}
    want = _root(tmp_path, base, KEY, 2)
    other = _root(tmp_path, base, bytes(32), 2)
    regions = _regions(2, 7, 6, 2)
    ok = {"regions": regions, "expected_root": want}
    for layout in (None, regions):
        assert D.replay(port("hold", hold(tmp_path)), base | {"payload_root": want, "layout": layout}, server) == (
            "output", D.canonical(ok))
    wider = _regions(2, 8, 6, 0)
    shown = "[('weights', 'hbm', 0, 6), ('filler', 'hbm', 6, 1), ('weights', 'hbm', 7, 6), ('filler', 'hbm', 13, 1)]"
    cases = [
        (base | {"payload_root": other},
         f"the hello's payload root {other[:16]}... is not the pinned model {ROLE}'s store (2 arenas of 7 blocks, 6 of "
         "weights each, then 2 host filler blocks) under this verifier's payload key, whose root is "
         f"{want[:16]}...: register the store `remote_bandwidth.store` lays out, salted with the key the verifier sent",
         want),
        (base | {"payload_root": want, "blocks": 13},
         f"the hello's store is 13 blocks, fewer than the pinned model's 2 arenas of 7", None),
        (base | {"payload_root": want, "layout": wider},
         "the hello's layout [('weights', 'hbm', 0, 6), ('filler', 'hbm', 6, 2), ('weights', 'hbm', 8, 6), "
         f"('filler', 'hbm', 14, 2)]... is not the pinned model {ROLE}'s {shown}...", None),
        (base | {"payload_root": want, "node": ""}, "a node's id is 1 to 64 bytes", None),
        (base | {"payload_root": want, "engines": 0},
         "a store is one or more arenas and no fewer than zero host blocks, not 0 and 0", None),
        (base | {"payload_root": want, "arena_blocks": 5}, "the weights take 6 blocks, more than an arena's 5", None),
    ]
    for v, why, root in cases:
        assert D.replay(port("hold", hold(tmp_path)), v, server) == (
            "output", D.canonical({"refused": why, "expected_root": root})), why


# --- the verifier's binding: `bind`, then `hold` of each hello ---------------------------------------------------------

def binding(tmp: Path):
    """The binding as `Binding` starts it, past what it reads of the files: the pin from its file's bytes (`model_pin`)
    and `Store`'s check of the arenas, `{"pin", "weights_blocks"}` or why not (the Lean's `bind`); then, once bound,
    `Binding.hold` of each hello (the Lean's `hold`)."""
    def reference(v) -> dict:
        data, path, role, digests = M.hex_(v, "data"), M._str(v, "path"), M._str(v, "role"), _digests(v)
        engines, arena_blocks = M.int_(v, "engines"), M.int_(v, "arena_blocks")
        M.nat(v, "workers")
        hellos = M._list(v, "hellos")
        try:
            pin = store.model_pin(data, path, role)
            store.Store(_arena(digests), engines, arena_blocks)
        except store.Refused as e:
            return {"bind": {"refused": str(e)}, "holds": []}
        b = _binding(tmp, role, digests, engines, arena_blocks)
        return {"bind": {"pin": pin, "weights_blocks": len(digests)}, "holds": [_held(b, _hello(h)) for h in hellos]}
    return reference


def binding_inputs(tmp: Path):
    @st.composite
    def inputs(draw) -> dict:
        """`hold_inputs`' store and hello, then another hello of that store or not, and the role's pin as a pin file
        holds it (a pin, a registry's line, core's `models.json`), as it is or with one fault."""
        v = draw(hold_inputs(tmp))
        hellos = [{k: v.pop(k) for k in ("node", "blocks", "payload_root", "key", "layout")}]
        if draw(st.booleans()):
            engines, arena_blocks, host = max(v["engines"], 1), max(v["arena_blocks"], len(v["digests"])), draw(
                st.integers(0, 3))
            root = _root(tmp, v | {"engines": engines, "arena_blocks": arena_blocks}, KEY, host)
            hellos.append({"node": NODE.hex(), "blocks": engines * arena_blocks + host, "payload_root": root,
                           "key": KEY.hex(), "layout": None})
        role, files = v["role"], files_of(draw(st.binary(max_size=4)))
        g = draw(st.integers(0, 9))
        pin = a_pin(files, "other" if g == 1 else role)
        if g == 2:
            pin["checkpoint_sha512"] = "cd" * 64
        text = {3: lines({"model": role} | pin), 4: json.dumps(core(files, (role, "OTHER_BF16"))),
                5: json.dumps(core(files, ("OTHER_BF16", "X")))}.get(g, json.dumps(pin))
        data = text.encode()
        if g == 6:
            data = b"\xff" + data
        elif g == 7:
            data = data[:len(data) // 2]
        return v | {"data": data.hex(), "path": draw(st.sampled_from(["pin.json", "pins/é.json"])),
                    "workers": draw(st.integers(0, 4)), "hellos": hellos}
    return inputs()


def test_binding(server, tmp_path) -> None:
    stats = D.run(port("binding", binding(tmp_path), binding_inputs(tmp_path), need_rejects=False), examples=150,
                  server=server)
    assert stats.outputs > 120, stats


def test_each_decision_of_the_verifiers_binding_is_the_leans_word_for_word(server, tmp_path) -> None:
    """The binding the verifier starts, bound, holding hellos of its store with and without its layout and two it
    refuses; then each refusal of `bind` (another model's pin, a pin that isn't its checkpoint's, a pin that isn't
    UTF-8, too few arenas, an arena short of the weights, and a pin and a shape both refused: the pin's first), replayed
    through both sides: the pin, the regions and every reason are the same text."""
    digests = [expand(b"w%d" % u, 64).hex() for u in range(6)]
    shape = {"digests": digests, "engines": 2, "arena_blocks": 7}
    want, other = _root(tmp_path, shape, KEY, 2), _root(tmp_path, shape, bytes(32), 2)
    regions = _regions(2, 7, 6, 2)
    hello = {"node": NODE.hex(), "blocks": 16, "payload_root": want, "key": KEY.hex(), "layout": None}
    good, p = a_pin(files_of(b"f")), "pins/pin.json"
    v = shape | {"data": json.dumps(good).encode().hex(), "path": p, "role": ROLE, "workers": 3,
                 "hellos": [hello, hello | {"layout": regions}, hello | {"payload_root": other}, hello | {"blocks": 13}]}
    ok = {"regions": regions, "expected_root": want}
    another = (f"the hello's payload root {other[:16]}... is not the pinned model {ROLE}'s store (2 arenas of 7 blocks, "
               "6 of weights each, then 2 host filler blocks) under this verifier's payload key, whose root is "
               f"{want[:16]}...: register the store `remote_bandwidth.store` lays out, salted with the key the verifier "
               "sent")
    short = "the hello's store is 13 blocks, fewer than the pinned model's 2 arenas of 7"
    assert D.replay(port("binding", binding(tmp_path)), v, server) == ("output", D.canonical({
        "bind": {"pin": good, "weights_blocks": 6},
        "holds": [ok, ok, {"refused": another, "expected_root": want}, {"refused": short, "expected_root": None}]}))
    others = json.dumps(a_pin(files_of(b"f"), "other")).encode().hex()
    cases = [
        (v | {"data": others}, f"the pin {p} is other's, not --model {ROLE}'s"),
        (v | {"data": json.dumps(good | {"checkpoint_sha512": "cd" * 64}).encode().hex()},
         f"the pin {p}'s checkpoint_sha512 is not the SHA-512 of its checkpoint's canonical JSON "
         f"({good['checkpoint_sha512'][:16]}...)"),
        (v | {"data": (b"\xff" + json.dumps(good).encode()).hex()}, f"the pin {p} is not UTF-8"),
        (v | {"engines": 0}, "a store is one or more arenas and no fewer than zero host blocks, not 0 and 0"),
        (v | {"arena_blocks": 5}, "the weights take 6 blocks, more than an arena's 5"),
        (v | {"data": others, "engines": 0}, f"the pin {p} is other's, not --model {ROLE}'s"),
    ]
    for c, why in cases:
        assert D.replay(port("binding", binding(tmp_path)), c, server) == (
            "output", D.canonical({"bind": {"refused": why}, "holds": []})), why
