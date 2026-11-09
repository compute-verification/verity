"""A window's edge (`pipeline.edge_rows`): each edge row is its writer's output row, committed through the served Commit and carried
into the next window's registration, and a root over anything else is refused by name, by the full check and when drawn; on the
toy (`edge.kv`, one writer) and on GLM-5.3's latent row, two ports each at its own writer's grain (`edge.kv_c`, `edge.kv_pe`)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from verity.core.protocols.spot.one_stage import registered as REG
from verity.core.protocols.spot.one_stage import registration as R
from verity_vllm.pipeline import edge_rows as ER, serving_rows as SRW

from tests.pipeline import served_edge as SE

NOT_TREE = "its root is not the tree over its rows' sources' strings"


def _serve(root: Path, shape: SE.Shape) -> dict[str, Any]:
    pytest.importorskip("torch")
    w0, w1 = SE.run(root, shape)
    v0 = SE.verify(w0)
    return {"w0": w0, "w1": w1, "v0": v0, "v1": SE.verify(w1, w0, v0["strings"])}


@pytest.fixture(scope="module")
def served(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _serve(tmp_path_factory.mktemp("edge"), SE.TOY)


@pytest.fixture(scope="module")
def glm(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _serve(tmp_path_factory.mktemp("edge-glm"), SE.GLM)


def _private(c: dict[str, Any], port: str) -> tuple[np.ndarray, list[bytes]]:
    doc = json.loads((Path(c["hook"]) / ER.FILE).read_text())
    k = list(doc["ports"]).index(port)
    with np.load(Path(c["hook"]) / ER.PRIVATE) as z:
        return z[f"rows{k}"].copy(), [bytes(s) for s in z[f"salts{k}"]]


def _forged(c: dict[str, Any], v: dict[str, Any], port: str, row: int, bit: int) -> tuple[dict, dict]:
    """A prover's record and edge file for window ``c`` whose ``port`` has row ``row``'s first leaf one bit off (``bit``), committed
    again under the window's own salts, so its domain and leaves are the honest ones."""
    rows, salts = _private(c, port)
    rows[row, 0] ^= 1 << bit
    p = next(q for q in v["ports"] if q.name == port)
    reg = REG.Registered.commit(v["window"].partition["program"], port, rows.tolist(), salts, word_bits=p.word_bits,
                                window=ER.of_request(c["edge"]).window)
    doc = json.loads(json.dumps(v["doc"]))
    doc["ports"][port] |= {"root": reg.root.hex(), "paths": [reg.path(r).hex() for r in range(reg.leaves)]}
    rec = {**v["record"], "roots": [{**r, "root": reg.root.hex()} if r["port"] == port else r for r in v["record"]["roots"]]}
    return rec, doc


def test_a_running_requests_edges_go_from_one_served_window_into_the_next_ones_registration(served: dict[str, Any]) -> None:
    w0, w1, v0, v1 = served["w0"], served["w1"], served["v0"], served["v1"]
    for v in (v0, v1):
        assert v["registration"].ok, v["registration"].to_json()
        assert v["edge"] == {} and v["edge_drawn"] == {}
    roots0 = {r["port"]: r["root"] for r in v0["record"]["roots"]}
    assert v0["record"].get("registered") in (None, [])
    assert v1["record"]["previous"] == R.digest(v0["record"])
    assert {e["value"]: e["root"] for e in v1["record"]["registered"]} == {p: roots0[p] for p in ("edge.kv", "edge.last")}

    keys = [r["i"] for r in w0["inst"]["rows"] if r["spec"] == SE.KEY]
    kv0, _ = _private(w0, "edge.kv")
    last0, _ = _private(w0, "edge.last")
    for k, rid in enumerate(SE.PROMPTS):
        for i, row in enumerate(keys):
            key = np.asarray(w0["world"].values[(rid, row)], dtype=np.uint64).ravel()[:SE.KW]
            assert kv0[k * len(keys) + i].tolist() == key.tolist()
        assert w1["world"].prompts[rid] == [int(last0[k][0])]

    kv1, _ = _private(w1, "edge.kv")
    held = [k for k, (unit, _s) in enumerate(next(p for p in v1["ports"] if p.name == "edge.kv").sources) if unit is None]
    assert len(held) == SE.HISTORY * len(SE.PROMPTS)
    assert kv1[held].tolist() == kv0.tolist()
    assert [v1["strings"]["edge.kv"][k] for k in held] == v0["strings"]["edge.kv"]


@pytest.mark.parametrize(("k", "port", "row", "bit"), [(0, "edge.kv", 5, 0), (0, "edge.last", 1, 17), (1, "edge.kv", 4, 3),
                                                        (1, "edge.kv", 2, 9)])
def test_an_edge_row_one_bit_off_its_writers_output_is_refused_by_name(served: dict[str, Any], k: int, port: str, row: int,
                                                                       bit: int) -> None:
    _refused(served, k, port, row, bit)


def _refused(served: dict[str, Any], k: int, port: str, row: int, bit: int) -> None:
    c, v = served[f"w{k}"], served[f"v{k}"]
    last, held = (None, None) if k == 0 else (served["w0"], served["v0"]["strings"])
    rec, doc = _forged(c, v, port, row, bit)
    unit, src = next(p for p in v["ports"] if p.name == port).sources[row]
    others = [u for u in range(v["window"].population) if u != unit]

    got = SE.verify(c, last, held, units=[] if unit is None else [unit], record=rec, doc=doc)
    assert got["registration"].ok, got["registration"].to_json()
    assert got["edge"] == {port: NOT_TREE}
    if unit is None:
        assert got["edge_drawn"] == {}
    else:
        assert got["edge_drawn"] == {port: f"row {row}, unit {unit}'s {src}, does not open its root with its writer's string"}
        assert SE.verify(c, last, held, units=others, record=rec, doc=doc)["edge_drawn"] == {}


def test_glm_s_latent_row_is_two_edge_values_each_its_writers_output_rows(glm: dict[str, Any]) -> None:
    """GLM-5.3's latent row: `edge.kv_c` the kv norm's output rows and `edge.kv_pe` the kv rope's, each wired to its own writer
    in window 0, held into window 1 under window 0's roots, and checked there with window 1's own rows."""
    w0, w1, v0, v1 = glm["w0"], glm["w1"], glm["v0"], glm["v1"]
    for v in (v0, v1):
        assert v["registration"].ok, v["registration"].to_json()
        assert v["edge"] == {} and v["edge_drawn"] == {}
    assert [(p.name, len(p.sources)) for p in v0["ports"]] == [("edge.kv_c", 8), ("edge.kv_pe", 8), ("edge.last", 2)]
    assert [(p.name, len(p.sources), sum(u is None for u, _s in p.sources)) for p in v1["ports"]] == [
        ("edge.kv_c", 12, 8), ("edge.kv_pe", 12, 8), ("edge.last", 2, 0)]
    roots0 = {r["port"]: r["root"] for r in v0["record"]["roots"]}
    assert {e["value"]: e["root"] for e in v1["record"]["registered"]} == {p: roots0[p] for p in ("edge.kv_c", "edge.kv_pe", "edge.last")}
    for port, words in (("edge.kv_c", SE.KV_C), ("edge.kv_pe", SE.KV_PE)):
        writers = [r["i"] for r in w0["inst"]["rows"] if r["spec"] == SE.GLM.writers[port]]
        rows0, _ = _private(w0, port)
        assert rows0.tolist() == [np.asarray(w0["world"].values[(rid, i)], dtype=np.uint64).ravel()[:words].tolist()
                                  for rid in SE.PROMPTS for i in writers]
        rows1, _ = _private(w1, port)
        held = [k for k, (unit, _s) in enumerate(next(p for p in v1["ports"] if p.name == port).sources) if unit is None]
        assert rows1[held].tolist() == rows0.tolist() and [v1["strings"][port][k] for k in held] == v0["strings"][port]


@pytest.mark.parametrize(("k", "port", "row", "bit"), [(0, "edge.kv_c", 5, 0), (0, "edge.kv_pe", 3, 7), (1, "edge.kv_c", 4, 3),
                                                        (1, "edge.kv_pe", 4, 11), (1, "edge.kv_c", 2, 9), (1, "edge.kv_pe", 1, 2)])
def test_a_glm_latent_edge_row_one_bit_off_its_writers_output_is_refused_by_name(glm: dict[str, Any], k: int, port: str, row: int,
                                                                                 bit: int) -> None:
    _refused(glm, k, port, row, bit)


def test_glm_s_latent_row_as_one_edge_value_is_refused_by_name() -> None:
    """The unsplit latent row, the kv norm's 512 words then the kv rope's 64, has no one writer: refused by name."""
    pytest.importorskip("torch")
    inst, _ = SE.program(0, SE.GLM)
    norms, ropes = ([r["i"] for r in inst["rows"] if r["spec"] == t] for t in (SE.NORM, SE.PE))
    joined = {"edge.kv": {"bits": 16, "runs": [run for a, b in zip(norms, ropes) for run in (["n", a, 0, SE.KV_C], ["n", b, 0, SE.KV_PE])]},
              "edge.last": inst[ER.HEADER]["edge.last"]}
    with pytest.raises(ValueError, match=r"edge\.kv of request r0: row 0 is not one run of 576 leaves"):
        _wired(joined, words={"edge.kv": SE.KV_C + SE.KV_PE, "edge.last": 1}, shape=SE.GLM)


def _wired(header: dict[str, Any] | None = None, words: dict[str, int] | None = None, templates: list[str] | None = None,
           order: tuple[str, ...] = tuple(SE.PROMPTS), C: int = 0, before: dict | None = None, shape: SE.Shape = SE.TOY) -> Any:
    inst, _woff = SE.program(C, shape)
    if header is not None:
        inst[ER.HEADER] = header
    programs = {rid: inst for rid in SE.PROMPTS}
    window, rows = SRW.population(programs, {"templates": templates or shape.templates, "law": SE.LAW}, SE.OPEN[0],
                                  values=ER.values(before))
    spec = ER.of_request({"window": SE.OPEN[0], "words": words or shape.words, "order": list(order), "last": None})
    return ER.wiring(window, rows, programs, spec, before)


def test_an_edge_row_not_one_proof_blocks_whole_output_is_refused_by_name() -> None:
    pytest.importorskip("torch")
    assert [(p.name, len(p.sources)) for p in _wired()] == [("edge.kv", 8), ("edge.last", 2)]
    inst, _ = SE.program(0)
    keys = [r["i"] for r in inst["rows"] if r["spec"] == SE.KEY]
    kv = inst[ER.HEADER]["edge.kv"]
    with pytest.raises(ValueError, match=r"edge\.kv of request r0: row 0 is not one run of 128 leaves"):
        _wired(words={**SE.WORDS, "edge.kv": 2 * SE.KW})
    halves = {**inst[ER.HEADER], "edge.kv": {**kv, "runs": [["n", i, lo, lo + 32] for i in keys for lo in (0, 32)]}}
    with pytest.raises(ValueError, match=r"edge\.kv of request r0: row 0 .*whole output row"):
        _wired(halves, words={**SE.WORDS, "edge.kv": 32})
    with pytest.raises(ValueError, match=r"edge\.kv of request r0: row 0 .*whole output row"):
        _wired(templates=[SE.SELECT])
    with pytest.raises(ValueError, match=r"request r2"):
        _wired(order=("r0", "r1", "r2"))
    with pytest.raises(ValueError, match=r"edge\.kv of request r0: row 0 reads history the last window did not leave it"):
        _wired(C=SE.HISTORY)


def test_a_held_row_is_the_requests_own_history_row_the_last_window_left(served: dict[str, Any]) -> None:
    before = json.loads((Path(served["w0"]["hook"]) / ER.FILE).read_text())
    kv = _wired(C=SE.HISTORY, before=before)[0]
    assert kv.sources[:SE.HISTORY] == tuple((None, k) for k in range(SE.HISTORY))
    assert kv.sources[SE.HISTORY + 2:2 * SE.HISTORY + 2] == tuple((None, k) for k in range(SE.HISTORY, 2 * SE.HISTORY))
    with pytest.raises(ValueError, match=rf"edge\.kv of request r0: row {SE.HISTORY} reads history the last window did not "
                                         rf"leave it \({SE.HISTORY} rows of edge\.kv\)"):
        _wired(C=SE.HISTORY + 1, before=before)
    with pytest.raises(ValueError, match=rf"edge\.kv of request r0: the last window's rows are {SE.KW} leaves of 16 bits, not 32 of 16"):
        _wired(C=SE.HISTORY, before=before, words={**SE.WORDS, "edge.kv": 32})
    inst, _ = SE.program(SE.HISTORY)
    weights = {**inst[ER.HEADER], "edge.kv": {"bits": 16, "runs": [["p", 2, 0, SE.KW]]}}
    with pytest.raises(ValueError, match=r"edge\.kv of request r0: row 0 is a row of parameter weights, not of its history prefix"):
        _wired(weights, C=SE.HISTORY, before=before)
