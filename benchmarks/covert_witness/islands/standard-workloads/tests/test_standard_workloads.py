"""standard_workloads.py builds `training/standard-workloads`: a row per workload and measured dial setting, whose
ρ is formalism.published's under the builder's `READING`, A (with its binding kind, what it rests on and its kinds as
published gives them, a measured row's rests_on opening with algorithm_policy's TRACED where ρ is the products term's
or its floor's, reading B's forms among their sensitivities; null and "unmeasured" where it has no bound, never
infinite) at the headline, the row's own session on the model it serves (rho_own, beside the session's c and packed
ids), and two sensitivities, the console's default (rho_default and rho_console) and the registered serving block
(rho_served), a distillation row's student's own session a third, in the note; whose rho_inputs give the dial's ρ at
each (page_rho) and whose input_status says which inputs te_core measured (an unmeasured Δ_grad's steps and which ρ
columns' Fit verdict rests on it, and the products kind's input); a generation run's row is its traced rounds alone,
each GEMM on the served codes priced as NVFP4, with no backward, and null with its reason where a traced round's trace
counts fewer tokens than its forward positions, and an RL run's learner row those rollouts plus its traced learner,
each part's share of the gates in its run cell; a workload no run measures is a modelled row,
te_core.workloads' model priced the same way; it reads traces from te_core 4.4 on; the panel keeps to the panels
route's limits, a cell up to 4,096
characters, never replaces a served panel with one that drops its rows, and cites its tables in the research run's
out/; a PUT takes a cited body only over the panel the build read, then reads it back; and a run a plan restates is
read with its record's tokens a step restated and its stored traces against the restated count."""
import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from te_core import VERSION, algorithm_policy, annotate, console, formalism, panels, policy, workloads
from te_core.gates import of_log

#: the builder, loaded from its file: on sys.path, the island's directory would hide the standard library's trace
#: module behind the island's trace.py
_SPEC = importlib.util.spec_from_file_location("standard_workloads",
                                               Path(__file__).resolve().parents[1] / "standard_workloads.py")
SW = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(SW)
#: a distillation run's row as the builder measured it from the store: the rows file's row, the run's quantities at its
#: last checkpoint (formalism.Measured), its start's shape, what it read, and the panel's cells it gave
PINNED = Path(__file__).parent / "fixtures" / "distill.json"
#: the columns the pinned cells predate, the headline's: each test checks them against te_core directly
OWN = ("rho_own", "own_c_gates", "own_k_bits")
#: the pinned run's student, as its notes give it (distill.py's `student` and `student_parameters`)
STUDENT = {"d_model": 256, "n_layer": 6, "n_head": 4, "d_ff": 704, "vocab": 32_000, "parameters": 13_012_224}
#: the commits a test build's source names (`commits`'s: this checkout's, the integrator's branch and its commit)
REVS = ("c0ffee0000", "integrator", "facade0000")
#: stored tables as a note cites them (`cite`'s: their path in a research run's out/)
TABLES = "run:r20261103-000000-0000/out/tables"
U, R = 16, 4
#: a served matrix (out U, in 2U) and the GEMMs on it, the activations and a registered matrix in a round of 2 x 3
#: tokens (2 sequences of a 3-token prompt and 2 generated tokens, the last never fed back)
SERVED = {(2 * U, U)}
NOTES = {"args": {"steps": 2, "update": "full", "head": "fp4"}, "served_code_bits": 4.5 * U * 2 * U,
         "served_lora_bits": 16 * R * 3 * U, "rounds": [{"after_step": 1, "prompt": 3, "generated": 2, "sequences": 2},
                                                        {"after_step": 2, "prompt": 3, "generated": 2, "sequences": 2}]}
CODED = ["fwd", "updated", "mm", json.dumps([[8, 2 * U], [2 * U, U]]), 1, 8 * 2 * U * U]
ROUND_LOG = [CODED, ["fwd", "act", "mm", json.dumps([[8, U], [U, 8]]), 1, 8 * U * 8],
             ["fwd", "theta", "mm", json.dumps([[8, U], [U, 2 * U]]), 1, 8 * U * 2 * U],
             ["fwd", "updated", "mul", json.dumps([[8, U], [U]]), 1, 8 * U]]
#: a traced step's product of two fresh operands at the console's k, its operands made whole by its own chain class
#: (`fed`, `P`: te_core.capture.products' whole-making system), whose fixed point the console's k reaches, and neither
#: reading an embedding's rows (`gathered`)
PRODUCT = {"count": 2, "phase": "forward", "op": "mm", "dims": {"batch": 1, "m": 4, "k": 1024, "n": 8},
           "token_dims": ["m"], "macs": 32768, "outputs": 32, "affected": [True, True], "widths": [16, 16],
           "rates": [16.0, 16.0], "paid": [True, True], "self": False, "b_c": 32, "k_whole": 1024,
           "gates_per_mac": 1480.0, "cell_gates": 23682.0, "fed": [[1024, 1.0]], "P": 1.0, "gathered": [False, False]}


def products_rows(*rows, steps: float | None = None, p_match: tuple = (), dial: tuple = (), elementwise: tuple = (),
                  capture: str = formalism.ELEMENTWISE_FROM, code: str | None = None) -> dict:
    """`rows` as a run's products (formalism.products_of_trace): a traced step's, captured by `capture` (and written
    by the code of VERSION `code`, where the trace records it), times `steps` (None: the step captured on its own,
    which stands for one after the run's first ones, so M keeps it whole), with the ops that break P-match
    `p_match` and the products the step writes elementwise `elementwise`, and the same rows at each k of `dial` (a
    trace's `at`)."""
    block = {"capture": capture, "capture_code": code, "k": float(policy.CONSOLE[1]), "order": "unknown", "tokens": 8,
             "gates": None, "rows": list(rows), "p_match": list(p_match), "elementwise_products": list(elementwise)}
    at = {f"{k:.0f}": {"k": float(k), "rows": list(rows), "p_match": list(p_match)} for k in dial}
    return formalism.products_of_trace(block | ({"at": at} if at else {}), steps)


def record(round_tokens: int = 6, round_log=ROUND_LOG) -> dict:
    """generate.py's layout: training steps 0 and 2, each followed by a round (steps 1 and 3) whose counted tokens are
    its prompts', and two checkpoints."""
    train = [["fwd", "updated", "mm", json.dumps([[64, 2 * U], [2 * U, U]]), 1, 64 * 2 * U * U],
             ["bwd", "grad", "mm", json.dumps([[2 * U, 64], [64, U]]), 2, 2 * 64 * 2 * U * U]]
    steps = [{"step": i, "tokens": round_tokens if i % 2 else 64, "struct": "round" if i % 2 else "train"}
             for i in range(4)]
    return {"steps": steps, "structures": {"train": {"updated": [[1]], "log": train},
                                           "round": {"updated": [], "log": round_log}},
            "setup": {"step": -1, "by": {"other/param": 32}, "macs": 32},
            "tail": {"step": 4, "by": {}, "macs": 0}, "checkpoints": [{"step": 2}, {"step": 4}]}


def build(m: formalism.Measured, fx: dict, monkeypatch) -> tuple[dict, list, dict]:
    monkeypatch.setattr(SW, "traces", lambda arts: {})
    monkeypatch.setattr(SW, "measured", lambda row, traced: (m, fx["shape"], fx["read"]))
    p, details = SW.build({"rows": [fx["row"]]}, REVS, stamp="2026-11-03T00:00:00Z")
    (row,) = p["rows"]
    return p, details, dict(zip(SW.COLUMNS, row))


def at(m: formalism.Measured) -> dict[str, dict]:
    return {name: formalism.at(m, c, k) for name, (c, k) in SW.POINTS.items()}


def test_the_pinned_row_is_its_runs_quantities_priced_as_algorithm_policy_prices_a_run_at_each_point(monkeypatch):
    fx = json.loads(PINNED.read_text())
    m = formalism.from_json(fx["measured"])
    p, details, cells = build(m, fx, monkeypatch)
    pinned = dict(zip([c for c in SW.COLUMNS if c not in OWN], fx["cells"], strict=True))
    assert {k: v for k, v in cells.items() if k not in {"te_core", *OWN}} == \
        {k: v for k, v in pinned.items() if k != "te_core"}
    assert cells["te_core"] == VERSION
    session, pts = SW.own(fx["shape"])
    assert session == algorithm_policy.own_session(fx["shape"]) and details[0]["own_session"] == SW.finite(session)
    assert pts == {"own": (session["gates"], algorithm_policy.OWN_K)} | SW.POINTS and algorithm_policy.OWN_K == 65_536
    assert (cells["own_c_gates"], cells["own_k_bits"]) == (pytest.approx(session["gates"], rel=1e-6), 61_440)
    assert p["source"] == (f"python islands/standard-workloads/standard_workloads.py at c0ffee0000 (te_core {VERSION}, "
                           f"on integrator at facade0000)")
    assert SW.source(("c0ffee0000", "a branch", "c0ffee0000")) == (
        f"python islands/standard-workloads/standard_workloads.py at c0ffee0000 (te_core {VERSION}, on a branch)")
    inputs, st = json.loads(cells["rho_inputs"]), json.loads(cells["input_status"])
    assert [k for k in inputs if k != panels.FRESH] == list(SW.KEYS) and list(st) == [*SW.KEYS, "products"]
    assert inputs[panels.FRESH] == formalism.fresh_per_c(m) is None
    assert {st[k] for k in SW.KEYS} == {"measured"}
    bounds = {name: formalism.bound(m, c, k) for name, (c, k) in SW.POINTS.items()}
    assert st["products"] == SW.products(bounds) == f"assumed: {bounds['console']['unmeasured']['fresh_products']}"
    got = {name: formalism.at(m, c, k) for name, (c, k) in pts.items()}
    assert SW.POINTS == {"console": algorithm_policy.CONSOLE, "served": panels.SERVED}
    one = {name: formalism.published(m, c, k, SW.READING) for name, (c, k) in pts.items()}
    for name, (c, k) in pts.items():
        assert math.isclose(algorithm_policy.page_rho(inputs, c, k)[0], got[name]["rho"], rel_tol=1e-4)
        assert cells[f"rho_{name}"] == SW.rho_cell(one[name]["rho"])
    assert cells["rho_default"] == cells["rho_console"]
    b = formalism.bound(m, *pts["console"])
    assert (cells["reading"], cells["binding_kind"], cells["rests_on"]) == \
        (SW.READING, one["console"]["binding_kind"], one["console"]["rests_on"])
    kinds = json.loads(cells["kinds"])
    assert cells["kinds"] == SW.kinds(one["console"], b) and list(kinds["sensitivity"]) == list(SW.SENSITIVITIES)
    for name in SW.SENSITIVITIES:
        assert kinds["sensitivity"][name] == pytest.approx(b["sensitivities"][name], rel=1e-3)
    assert details[0]["at"] == SW.finite(got) and details[0]["published"] == SW.finite(one)
    assert details[0]["bound"]["console"]["terms"] == b["terms"]
    assert not SW.refused(p) and not SW.refused(p | {"note": SW.note(p, TABLES)})


def test_under_reading_a_a_row_is_publisheds_rho_and_basis_as_is_with_its_terms_null_where_unmeasured(monkeypatch):
    """The pinned row's token kind under reading A (its trace's entries and its macs_log's tiles) is at most 1 at the
    console's and fresh products are unmeasured (an op log can't tell a fresh operand): its ρ, binding kind and basis
    are published's as it gives them, the kinds cell holding the token term, fresh products' null and published's
    partial and why; a measured term above 1 is published with its binding kind at each point; without an op log no
    kind is measured: null and unmeasured; an infinite ρ is refused. The note names the headline first, then the
    sensitivities."""
    fx = json.loads(PINNED.read_text())
    m = formalism.from_json(fx["measured"])
    assert SW.READING == "A"
    c, k = algorithm_policy.CONSOLE
    b = formalism.bound(m, c, k)
    assert b["terms"]["token"] is not None and b["terms"]["token"] <= 1 and b["terms"]["fresh_products"] is None
    one = formalism.published(m, c, k, "A")
    p, _, cells = build(m, fx, monkeypatch)
    assert (cells["reading"], cells["rho_console"], cells["binding_kind"], cells["rests_on"]) == \
        ("A", SW.rho_cell(one["rho"]), one["binding_kind"], one["rests_on"])
    kinds = json.loads(cells["kinds"])
    assert list(kinds["bound"]) == list(formalism.KINDS) and kinds["bound"]["fresh_products"] is None
    assert kinds["bound"]["token"] == pytest.approx(b["terms"]["token"], rel=1e-3)
    assert (kinds.get("partial", False), kinds.get("why")) == (one["partial"], one["why"])
    assert p["note"] == (
        "Reading A. rho_own: own session (4,096 in + 1,024 out; distill's teacher), k = 65,536; at the packed 61,440 "
        "rho is at least this. Sensitivities: rho_default = rho_console (45M), rho_served (serving block). fresh_old "
        "rests on P-width extended, false on honest circuits. Attention's products are priced at 16 bits, as the card "
        f"computes them; at 32 bits a measured rho is {SW.ATTENTION_32} of this.")
    assert algorithm_policy.OWN_K_LABEL == "k = 65,536 (16-bit ids); at the packed 61,440 rho is at least this"
    assert formalism.RESTS_ON_CELL["fresh_products"].startswith("theorem of record, given (H): ")
    assert formalism.CELL_SENSITIVITIES[-1] == "fresh_old" and formalism.SENSITIVITIES["fresh_old"].startswith(
        "rests on P-width extended, false on honest circuits: ")
    bare = dataclasses.replace(m, ops=None)
    _, _, cells = build(bare, fx, monkeypatch)
    assert (cells["rho_own"], cells["rho_default"], cells["rho_console"], cells["rho_served"], cells["rests_on"]) == \
        (None, None, None, None, "unmeasured")
    assert cells["binding_kind"] == formalism.published(bare, c, k, "A")["binding_kind"]
    assert json.loads(cells["kinds"])["bound"] == {"fresh_products": None, "token": None, "decoding": None}
    real = formalism.published
    above = {"rho": 2.5, "binding_kind": "token", "rests_on": formalism.RESTS_ON_CELL["token"]}
    monkeypatch.setattr(formalism, "published", lambda *a: real(*a) | above)
    _, _, cells = build(m, fx, monkeypatch)
    assert (cells["rho_own"], cells["rho_default"], cells["rho_console"], cells["rho_served"], cells["binding_kind"],
            cells["rests_on"]) == (2.5, 2.5, 2.5, 2.5, "token", formalism.RESTS_ON_CELL["token"])
    monkeypatch.setattr(formalism, "published", lambda *a: one | {"rho": math.inf})
    with pytest.raises(SystemExit, match="reading A's rho is infinite at own, console and served"):
        build(m, fx, monkeypatch)


def test_traces_from_te_core_4_4_to_this_one_are_read_and_others_refused(tmp_path, monkeypatch):
    (tmp_path / "traces").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"te_commit": "c0ffee"}))
    (tmp_path / "traces" / "a.json").write_text(json.dumps({"run": "r1", "step": 1, "trace": {"structure": {}}}))
    (tmp_path / "traces" / "b.json").write_text(json.dumps({"run": "r1", "step": 2, "trace": None}))
    monkeypatch.setattr(SW, "fetch", lambda art, *paths: tmp_path)
    for v in (SW.TRACED_SINCE, VERSION):
        monkeypatch.setattr(SW, "version_at", lambda commit, v=v: v)
        assert SW.traces(["art:t"]) == {"r1": {1: {"run": "r1", "step": 1, "trace": {"structure": {}}, "from": "art:t",
                                                   "te_core": v}}}
    for v in ("4.3", "9.0"):
        monkeypatch.setattr(SW, "version_at", lambda commit, v=v: v)
        with pytest.raises(SystemExit, match=f"traced by te_core {v}, not {SW.TRACED_SINCE} to {VERSION}"):
            SW.traces(["art:t"])


def test_a_step_traced_in_two_arts_is_read_from_the_newer_te_cores_trace_whatever_their_order(tmp_path, monkeypatch):
    """A run retraced on a newer te_core keeps its older art in the rows file beside the new one: each step is read from
    the newer te_core's trace in either order, and of two traces by one te_core, from the later art's, a tree of the
    traces or the research run's record holding them in out/ ("again")."""
    for art, v in (("old", "4.12.11"), ("new", "4.12.13"), ("again", "4.12.13")):
        d = tmp_path / art / ("out" if art == "again" else "")
        (d / "traces").mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"te_commit": v}))
        (d / "traces" / "r1-1.json").write_text(json.dumps({"run": "r1", "step": 1, "trace": {"by": art}}))
    monkeypatch.setattr(SW, "fetch", lambda art, *paths: tmp_path / art)
    monkeypatch.setattr(SW, "version_at", lambda commit: commit)
    for arts in (["old", "new"], ["new", "old"]):
        assert SW.traces(arts)["r1"][1] == {"run": "r1", "step": 1, "trace": {"by": "new"}, "from": "new",
                                            "te_core": "4.12.13"}
    assert SW.traces(["new", "again"])["r1"][1]["from"] == "again"


def test_a_publish_never_drops_a_served_runs_row_nor_replaces_a_newer_te_cores():
    def panel(runs, te=VERSION):
        rows = [x if isinstance(x, tuple) else ("standard", x) for x in runs]
        cells = [{"workload": w, "model": "45M", "run": f"{r} art:x (about)", "te_core": te} for w, r in rows]
        return {"columns": SW.COLUMNS, "rows": [[c.get(name) for name in SW.COLUMNS] for c in cells]}
    mine = panel(["r1", "r2"])
    assert SW.emptied(mine, None) == [] and SW.emptied(mine, panel(["r2", "r1"], "4.4")) == []
    assert SW.emptied(mine, panel(["r1", "r3"])) == ["it drops the served row of r3"]
    assert SW.emptied(mine, panel(["r1"], "9.0")) == [f"the route serves te_core 9.0, newer than {VERSION}"]
    modelled = panel(["r1", ("rl", SW.MODELLED), ("inference", SW.MODELLED)])
    assert SW.emptied(modelled, modelled) == [] and SW.emptied(panel(["r1", ("rl", SW.MODELLED)]), modelled) == \
        [f"it drops the served row of inference 45M ({SW.MODELLED})"]


def test_a_modelled_row_is_the_workload_models_products_priced_at_each_point():
    """A workload no run measures (the rows file's RL with the updated weights between updates, and inference on an
    unregistered model) is te_core.workloads' model at the row's dials, its products at each point's k priced as a
    run's are: its ρ at each point workloads.rho's, the headline's at the model's own session, which on the 45M
    reference is the console's default itself, its run cell `MODELLED` with what the model is, its dials every
    argument of the model, each input the model's, no capture code, and what it rests on published's, which no trace's
    label opens; the rows file's words for it name no code."""
    rows = [r for r in json.loads(SW.ROWS.read_text())["rows"] if "modelled" in r]
    assert [r["workload"] for r in rows] == ["rl", "inference"]
    p, details = SW.build({"rows": rows}, REVS, stamp="2026-11-03T00:00:00Z")
    for r, row, d in zip(rows, p["rows"], details, strict=True):
        cells, model = dict(zip(SW.COLUMNS, row)), workloads.model(r["workload"], **r["dials"])
        assert cells["run"] == f"{SW.MODELLED} ({r['modelled']})" and cells["dials"] == SW.compact(model.dials)
        session, pts = SW.own({"parameters": model.dials["n_params"]})
        assert pts["own"] == SW.POINTS["console"] and d["own_session"] == SW.finite(session)
        assert (cells["own_c_gates"], cells["own_k_bits"]) == (pytest.approx(algorithm_policy.CONSOLE[0], rel=1e-6),
                                                               65_536)
        for name, (c, k) in pts.items():
            assert cells[f"rho_{name}"] == SW.rho_cell(workloads.rho(model, c, k)["rho"])
        assert cells["rho_own"] == cells["rho_console"] and "student_session" not in d
        con = workloads.rho(model, *SW.POINTS["console"])
        assert cells["rests_on"] == (con["products_rests_on"] if con["basis"] == "fresh_products" else
                                     formalism.RESTS_ON_CELL[con["basis"]])
        assert not cells["rests_on"].startswith(algorithm_policy.TRACED)
        assert cells["rho_default"] == cells["rho_console"] and cells["capture_code"] is None
        st = json.loads(cells["input_status"])
        assert list(st) == [*SW.KEYS, "products"] and all(v.startswith("modelled: ") for v in st.values())
        assert (d["run"], d["art"], d["modelled"], d["read"]["derived"]) == (None, None, r["modelled"], model.derived)
    assert not SW.refused(p) and not SW.refused(p | {"note": SW.note(p, TABLES)})
    assert [r["modelled"] for r in rows if SW.CODE.search(r["modelled"])] == []


def test_the_note_cites_tables_in_a_research_runs_out_and_a_build_elsewhere_cites_none(tmp_path, monkeypatch):
    run = tmp_path / "r20261104-081500-abcd"
    for name in ("RESEARCH_RUN_DIR", "RESEARCH_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    assert SW.cite(run / "out") is None
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run))
    assert SW.cite(run / "out") == "run:r20261104-081500-abcd/out/tables"
    assert SW.cite(run / "out" / "sw") == "run:r20261104-081500-abcd/out/sw/tables"
    assert SW.cite(run) is None and SW.cite(tmp_path / "elsewhere") is None
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261104-081500-beef")
    assert SW.cite(run / "out") == "run:r20261104-081500-beef/out/tables"
    assert SW.TABLES.search(SW.note({"rows": []}, SW.cite(run / "out")))
    for tables in (None, "art:" + "0" * 64):
        assert not SW.TABLES.search(SW.note({"rows": []}, tables))


def test_a_branch_named_for_the_source_must_hold_the_checkouts_commit(monkeypatch):
    """`commits` with `branch` names that branch in place of the integrator's, at this checkout's own commit unless a
    base is given, and refuses a branch that doesn't hold the commit here or on origin."""
    import types

    import pin

    def git(*args, **kw):
        if args[0] == "rev-parse":
            return types.SimpleNamespace(returncode=0, stdout=("facade0000ffff" if "--verify" in args else "c0ffee0000ffff"))
        if args[:2] == ("merge-base", "--is-ancestor"):
            return types.SimpleNamespace(returncode=0 if args[3] in ("refs/remotes/origin/held", "HEAD") else 1, stdout="")
        assert args[0] == "status", args
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(pin, "git", git)
    assert SW.commits(branch="held") == ("c0ffee0000", "held", "c0ffee0000")
    assert SW.commits("a base", "held") == ("c0ffee0000", "held", "facade0000")
    assert SW.commits("a base") == ("c0ffee0000", pin.BRANCH, "facade0000")
    with pytest.raises(SystemExit, match="--branch gone doesn't hold this checkout's commit"):
        SW.commits(branch="gone")


def test_a_put_takes_a_cited_body_only_over_the_panel_the_build_read_and_reads_it_back(tmp_path, monkeypatch, capsys):
    """`put` PUTs a build's console body where the route still serves the panel the build read (`--served`), the body
    keeps its rows, the route takes it and its note cites stored tables; then the panel it GETs must be the body.
    Where someone PUT since, nothing is PUT. A dry put is refused where a put is, and PUTs nothing."""
    fx = json.loads(PINNED.read_text())
    p, _, _ = build(formalism.from_json(fx["measured"]), fx, monkeypatch)
    old = json.loads(console.body(p | {"note": SW.note(p, TABLES)}))
    route, puts = {SW.PANEL: old}, []

    def put(q, status="200"):
        puts.append(q)
        route[SW.PANEL] = json.loads(console.body(q))
        return status

    def body(q, name="body.json"):
        (tmp_path / name).write_bytes(console.body(q))
        return tmp_path / name

    monkeypatch.setattr(console, "served", lambda: dict(route))
    monkeypatch.setattr(console, "put", put)
    sent = {SW.PANEL: old}
    with pytest.raises(SystemExit, match="cites no stored tables"):
        SW.put(body(p), sent)
    with pytest.raises(SystemExit, match="drops the served row of r20261103"):
        SW.put(body(p | {"rows": [], "note": SW.note(p, "run:r20261104-081500-abcd/out/tables")}), sent)
    with pytest.raises(SystemExit, match="someone PUT over it"):
        SW.put(body(p), {SW.PANEL: old | {"updated_at": "2026-11-02T00:00:00Z"}})
    with pytest.raises(SystemExit, match="cites no stored tables"):
        SW.put(body(p), sent, dry=True)
    assert puts == []
    new = json.loads(console.body(p | {"note": SW.note(p, "run:r20261104-081500-abcd/out/tables")}))
    monkeypatch.setattr(console, "refusal", lambda q: "another checkout's te_core")
    capsys.readouterr()
    assert SW.put(body(new), sent, dry=True) == 0 and puts == [] and route[SW.PANEL] == old
    sent_body = console.body(new)
    assert capsys.readouterr().out.strip() == (
        f"dry: {SW.PANEL}: sha256 {hashlib.sha256(sent_body).hexdigest()}, {len(sent_body)} bytes, {len(new['rows'])} "
        f"rows, note {len(new['note'])} characters; console.put would refuse: another checkout's te_core; nothing "
        f"stored, nothing PUT")
    assert SW.put(body(new), sent) == 0 and puts == [new] and route[SW.PANEL] == new
    monkeypatch.setattr(console, "put", lambda q: "400 refused")
    assert SW.put(body(old), {SW.PANEL: new}) == 1
    monkeypatch.setattr(console, "put", lambda q: "200")
    assert SW.put(body(old), {SW.PANEL: new}) == 1 and route[SW.PANEL] == new


def test_a_put_takes_the_routes_integer_for_a_whole_float_as_that_number(tmp_path, monkeypatch, capsys):
    """The route writes a float with no fractional part as an integer (6.52837e+20 comes back 652837000000000000000),
    which Python's != holds unequal to the float: the guard on the panel the build read and the read-back compare as
    numbers, and still catch a changed cell."""
    fx = json.loads(PINNED.read_text())
    p, _, _ = build(formalism.from_json(fx["measured"]), fx, monkeypatch)
    c = SW.COLUMNS.index("own_c_gates")

    def big(q: dict, value) -> dict:
        return q | {"rows": [[value if i == c else v for i, v in enumerate(q["rows"][0])], *q["rows"][1:]]}

    old = big(json.loads(console.body(p | {"note": SW.note(p, TABLES)})), 6.52837e+20)
    new = big(json.loads(console.body(p | {"note": SW.note(p, "run:r20261104-081500-abcd/out/tables")})), 6.52837e+20)
    route = {SW.PANEL: big(old, 652837000000000000000)}

    def put(q):
        route[SW.PANEL] = big(json.loads(console.body(q)), 652837000000000000000)
        return "200"

    monkeypatch.setattr(console, "served", lambda: dict(route))
    monkeypatch.setattr(console, "put", put)
    monkeypatch.setattr(console, "refusal", lambda q: None)
    (tmp_path / "body.json").write_bytes(console.body(new))
    assert route[SW.PANEL] != old
    capsys.readouterr()
    assert SW.put(tmp_path / "body.json", {SW.PANEL: old}) == 0
    assert f"GET {SW.PANEL}: the body, {len(new['rows'])} rows" in capsys.readouterr().out
    monkeypatch.setattr(console, "put", lambda q: route.update({SW.PANEL: big(q, 652838000000000000000)}) or "200")
    assert SW.put(tmp_path / "body.json", {SW.PANEL: route[SW.PANEL]}) == 1
    assert "the route serves another panel than the body (rows differ)" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="someone PUT over it"):
        SW.put(tmp_path / "body.json", {SW.PANEL: big(route[SW.PANEL], 652839000000000000000)})


def test_a_row_whose_inputs_dont_give_te_cores_rho_is_refused(monkeypatch):
    fx = json.loads(PINNED.read_text())
    m = formalism.from_json(fx["measured"])
    w_fwd = policy.w_fwd(fx["shape"])
    monkeypatch.setattr(algorithm_policy, "page_rho", lambda d, c, k: (1.01 * formalism.at(m, c, k)["rho"], "target"))
    with pytest.raises(ValueError, match="at console"):
        SW.priced(m, w_fwd, at(m))
    with pytest.raises(ValueError, match="w_gates, w_fit_gates not finite"):
        SW.priced(formalism.Measured((formalism.Part(1e9, 0, 0, 0),), 16, 8192, formalism.even(1e6),
                                     formalism.even(1e6), 1e6), w_fwd, {})


def test_a_row_whose_token_dials_miss_its_measured_tokens_is_refused_with_the_value_to_write(monkeypatch):
    """A step of two micro-batches the counter counts one of: a dial read off its count is half the run's tokens."""
    fx = json.loads(PINNED.read_text())
    m = formalism.from_json(fx["measured"])
    assert SW.token_dials(fx["row"]["dials"], m.tokens) == []
    half = fx["row"] | {"dials": fx["row"]["dials"] | {"tokens_per_param": fx["row"]["dials"]["tokens_per_param"] / 2}}
    with pytest.raises(SystemExit, match=r"its dial tokens_per_param is 0\.75547, te_core measured 1\.51095: the dial"):
        build(m, fx | {"row": half}, monkeypatch)
    assert SW.token_dials({"n_params": 10, "tokens": 50.0}, 100.0) == ["its dial tokens is 50, te_core measured 100: "
                                                                        "the dial is the run's"]
    assert SW.token_dials({"n_params": 10, "tokens_per_param": 10.04}, 100.0) == []


def test_an_input_is_measured_where_te_core_counted_traced_or_read_it_and_assumed_where_it_stubbed_it():
    traced = formalism.TRACED | {"gates": "counted MACs in gates (te_core.gates)", "u1_bits": formalism.TRACED["u_in"],
                                 "delta_bits": formalism.STUBS["delta_bits"], "decoder_gates": formalism.DECODER["none"]}
    parts = (formalism.Part(3e12, 100, 2.4e12, 1e12, 3, 4e6), formalism.Part(1e12, 50, 0.8e12, 0, 2, None))

    def status(labels, override=None):
        return SW.status(formalism.Measured(parts, 15, 8192, formalism.even(1e6), formalism.even(1e6), 1e6, 0.0,
                                            labels), override)

    st = status(traced)
    assert {k for k, v in st.items() if v != "measured"} == {"delta_grad_bits"}
    assert st["delta_grad_bits"] == ("assumed: unmeasured on 2 of 5 steps in structures with no traced Δ_grad, so the "
                                     "slowdown code fails Fit at every k; with it at 0, Fit still fails at every "
                                     "point, so no rho column rests on it (the traced steps' largest 4e6 bits)")
    st = status(traced | {k: formalism.STUBS[k] for k in ("tau_bits", "u", "token_gates")})
    assert st["tau_bits"] == f"assumed: {formalism.STUBS['tau_bits']}"
    assert st["sigma"] == st["w_gates"] == st["w_fit_gates"] == f"assumed: {formalism.STUBS['token_gates']}"
    assert st["u_points"] == f"assumed: {formalism.STUBS['u']}" and st["u_in_points"] == "measured"
    assert status(traced | {"decoder_gates": formalism.DECODER["witness"]})["d_gates"] == "measured"
    assert status({k: v for k, v in traced.items() if k != "a_bits"})["a_bits"] == \
        "assumed: no label from the slowdown code"
    assert status(traced, {"tau_bits": "assumed: why"})["tau_bits"] == "assumed: why"


def test_an_unmeasured_delta_grad_names_its_steps_and_the_rho_columns_whose_fit_verdict_rests_on_it():
    """A step with tokens has no Δ_grad where te_core.annotate gives it none of the traced structures: one that updates
    nothing, whose trace measures nothing or has no traced updating step beside it (from te_core 4.11 such a step
    takes its measured Δ_grad, 0, where both hold), the program's last untraced, or one no traced step shares. The
    label names them, refused unless they number te_core's, and says a ρ column's Fit verdict rests on them only
    where Fit holds at its point with their Δ_grad at 0."""
    struct = {"updated": [[1.0, 2, 3, 4, "x"]]}
    rec = {"steps": [{"step": 0, "struct": "first", "tokens": 64}, {"step": 1, "struct": "train", "tokens": 64},
                     {"step": 2, "struct": "odd", "tokens": 64}, {"step": 3, "struct": "sync", "tokens": 0},
                     {"step": 4, "struct": "train", "tokens": 64}, {"step": 5, "struct": "end", "tokens": 64}],
           "structures": {"first": {"updated": []}, "train": struct, "odd": struct, "sync": {"updated": []},
                          "end": struct}}
    gaps = SW.untraced(rec, {"first": struct, "train": struct})
    assert gaps == ["step 0: its structure updates nothing, and the slowdown code takes such a step's trace only "
                    "where the trace measures the step and a traced step of the run updates something",
                    "step 2: no traced step shares its structure",
                    "step 5: the program's last, its structure having no other step, not traced"]
    f = {"tail_share": 0.0, "u_in": [[0.0, 0.0], [1.0, 8.0]], "tau_ids": 1.0, "tau_other_bits": 0.0, "u1_bits": 8.0}
    measured = {"first": {"updated": [], "formalism": f | {"delta_grad_bits": 0}},
                "train": struct | {"formalism": f | {"delta_grad_bits": 10}}}
    assert SW.untraced(rec, measured) == gaps[1:]
    assert SW.untraced(rec, measured, "why")[-1] == f"{gaps[-1]}, since why"
    idle = SW.untraced(rec, {"first": measured["first"]})
    assert [g.split(":")[0] for g in idle] == ["step 0", "step 1", "step 2", "step 4", "step 5"] and idle[0] == gaps[0]
    parts = (formalism.Part(3e12, 100, 2.4e12, 1e12, 3, 1e3), formalism.Part(1e12, 50, 0.8e12, 0, 3, None))
    m = formalism.Measured(parts, 0.0, 8192, formalism.even(1e6), formalism.even(1e6), 1e3, 0.0, {},
                           delta_t_bits=1e3)
    assert SW.held(m) == ["console", "served"] and not formalism.at(m, *SW.POINTS["console"])["fit"]
    assert SW.status(m, read={"untraced": gaps, "steps": 6})["delta_grad_bits"] == (
        f"assumed: unmeasured on 3 of 6 steps ({'; '.join(gaps)}), so the slowdown code fails Fit at every k; "
        f"rho_console and rho_served rest on it, Fit holding at their points with it at 0 (the traced steps' largest "
        f"{algorithm_policy.num(1e3)} bits)")
    alone = SW.status(m, read={"untraced": gaps, "steps": 6}, points={"own": SW.POINTS["console"]})
    assert "every k; rho_own rests on it, Fit holding at its point with it at 0 (" in alone["delta_grad_bits"]
    four = dataclasses.replace(m, parts=(parts[0], dataclasses.replace(parts[1], steps=4)))
    got = SW.status(four, read={"untraced": ["a", "b", "c", "d"], "steps": 7})["delta_grad_bits"]
    assert got.startswith("assumed: unmeasured on 4 of 7 steps (a; b; c; and 1 more), so")
    with pytest.raises(ValueError, match="te_core has no Delta_grad for 3 steps, `untraced` names 2"):
        SW.status(m, read={"untraced": gaps[:2], "steps": 6})


def test_the_products_input_is_measured_at_the_consoles_default_or_assumed_with_te_cores_reason():
    """`products`, the products kind's input (the theorem of record's term on M): measured where te_core prices a
    trace's product rows at the console's default and at the serving block, whose k the trace keeps them at too, with
    its reason where the serving block's k leaves them unmeasured (they were classified at the console's alone);
    measured where the term is the theorem's floor, 0 (a contraction too short to bound), and on a step that writes
    matrix products elementwise or whose rows predate the check for them, which M leaves out, though the old figure
    (fresh_old) is void or unmeasured there, its reason in the bound; assumed where the dominant traced step's stand in
    for steps of another shape, or with te_core's reason where unmeasured (an op log's, a round untraced, rows captured
    before te_core 4.12.1 that don't say which operands read an embedding's rows, a step that breaks P-match); and
    assumed where untraced steps that update count none."""
    fx = json.loads(PINNED.read_text())
    m = formalism.from_json(fx["measured"])

    def bounds(p):
        return {name: formalism.bound(dataclasses.replace(m, products=p), c, k) for name, (c, k) in SW.POINTS.items()}

    def unmeasured(p, kind="fresh_products"):
        return bounds(p)["console"]["unmeasured"].get(kind)

    rows = products_rows(PRODUCT)
    b = bounds(rows)
    off = b["served"]["unmeasured"]["fresh_products"]
    assert unmeasured(rows) is None and b["console"]["terms"]["fresh_products"] > 0
    assert off.startswith(f"unmeasured: the products were classified at k = {policy.CONSOLE[1]:g}")
    measured = f"measured at the console's default; at the serving block, {off}"
    assert SW.products(b) == measured
    both = bounds(products_rows(PRODUCT, dial=(panels.SERVED[1],)))
    assert not {*both["console"]["unmeasured"], *both["served"]["unmeasured"]} & {"fresh_products"}
    assert SW.products(both) == "measured"
    for since in (("gathered",), ("fed", "P", "gathered")):
        old = products_rows({x: v for x, v in PRODUCT.items() if x not in since})
        assert unmeasured(old) == formalism.HE_RETRACE
        assert SW.products(bounds(old)) == f"assumed: {formalism.HE_RETRACE}"
    short = PRODUCT | {"dims": PRODUCT["dims"] | {"k": 8}, "macs": 256, "k_whole": 8, "fed": [[8, 1.0]]}
    floor = bounds(products_rows(short))
    assert floor["console"]["terms"]["fresh_products"] == 0 and floor["console"]["void"] == {}
    assert floor["console"]["unmeasured"]["fresh_old"] == formalism.FRESH_VOID
    assert SW.products(floor) == measured
    unchecked = products_rows(PRODUCT, capture=formalism.SYSTEM_FROM)
    outer = {"op": "aten.mul.Tensor", "count": 2, "macs": 8192.0}
    written = products_rows(PRODUCT, elementwise=[outer])
    assert unchecked["elementwise_products"] is None and unmeasured(unchecked, "fresh_old") == formalism.FRESH_UNCHECKED
    assert unmeasured(written, "fresh_old").startswith(
        "unmeasured: the traced step writes matrix products as elementwise steps")
    for p in (unchecked, written):
        assert bounds(p)["console"]["terms"]["fresh_products"] == b["console"]["terms"]["fresh_products"]
        assert SW.products(bounds(p)) == measured
    gather = {"op": "aten.embedding.default", "fan_out": 4.0, "count": 1}
    broken = unmeasured(products_rows(PRODUCT, p_match=[gather]))
    assert broken.startswith(f"unmeasured: the traced step breaks the premise that {formalism.P_MATCH}")
    assert "aten.embedding.default" in broken
    assert SW.products(bounds(products_rows(PRODUCT, p_match=[gather]))) == f"assumed: {broken}"
    stand = rows | {"stand_in": {"structures": ["a", "c"], "step_tokens": [4, 8], "traced_tokens": 8,
                                 "why": formalism.STAND_IN}}
    shapes = ("the run's traced steps differ in shape, so the dominant traced step's products stand in for all of them, "
              "scaled by tokens (2 traced structures)")
    assert SW.products(bounds(stand)) == f"assumed: {shapes}; at the serving block, {off}"
    last = "step 9: the program's last, its structure having no other step, not traced"
    alone = "the slowdown code counting the traced steps' products alone"
    assert SW.products(b, left=[last]) == f"assumed: none on 1 untraced step ({last}), {alone}; at the serving block, {off}"
    assert SW.products(bounds(stand), left=[last] * 4) == (
        f"assumed: {shapes}; and none on 4 untraced steps ({last}; {last}; {last}; and 1 more), {alone}; at the serving "
        f"block, {off}")
    why = bounds(None)["console"]["unmeasured"]["fresh_products"]
    assert SW.products(bounds(None)) == f"assumed: {why}"


def test_the_builders_own_words_in_a_published_string_name_no_code():
    """What the builder adds to a row's input_status (its rounds' labels, its untraced steps, an unmeasured Δ_grad's
    and the products input's wording) and its note names the code only as training's slowdown code: `CODE` finds no
    code name in it, as it finds one in te_core's own names."""
    assert all(SW.CODE.search(s) for s in ("te_core 4.12", "io_bound", "rows_of", "formalism.published", "ONE_CLASS",
                                            "FRESH_VOID", "capture_step", "generate.py", "gemm/fp4", "`untraced`"))
    struct = {"updated": [[1.0, 2, 3, 4, "x"]]}
    rec = {"steps": [{"step": 0, "struct": "first", "tokens": 64}, {"step": 1, "struct": "train", "tokens": 64},
                     {"step": 2, "struct": "odd", "tokens": 64}, {"step": 3, "struct": "end", "tokens": 64}],
           "structures": {"first": {"updated": []}, "train": struct, "odd": struct, "end": struct}}
    gaps = SW.untraced(rec, {"first": struct, "train": struct})
    parts = (formalism.Part(3e12, 100, 2.4e12, 1e12, 3, 1e3), formalism.Part(1e12, 50, 0.8e12, 0, 3, None))
    m = formalism.Measured(parts, 0.0, 8192, formalism.even(1e6), formalism.even(1e6), 1e3, 0.0, {},
                           delta_t_bits=1e3)
    st = SW.status(m, read={"untraced": gaps, "steps": 4})
    con = {"unmeasured": {}, "inputs": {"fresh_products": {"stand_in": {"structures": ["a", "c"]}}}}
    b = {"console": con, "served": {"unmeasured": {"fresh_products": "why"}}}
    said = [SW.products(b, left=gaps * 2),
            SW.products(b | {"console": con | {"unmeasured": {"fresh_products": "why"}}}),
            SW.products(b | {"console": {"unmeasured": {}, "inputs": {"fresh_products": {}}}})]
    student = [{"student_session": {"published": {"rho": 110_662.4}}}]
    notes = [SW.note({"rows": [[{"reading": x, "model": "20M"}.get(c) for c in SW.COLUMNS]]}, TABLES, student)
             for x in ("A", "B")]
    assert "; distill at its student's: 20M 110,662. " in notes[0]
    words = [SW.IDLE, SW.NO_AXIS, *SW.ROUNDS.values(), *SW.LEARNER.values(), SW.LEARNER_W_FIT,
             SW.split({"rollouts": 0.01, "learner": 0.99}), *gaps, *st.values(), *said, *notes]
    assert len(gaps) == 3 and [s for s in words if SW.CODE.search(s)] == []


def test_a_binding_fresh_products_term_is_publisheds_and_the_products_input_names_the_untraced_steps_that_update(
        monkeypatch):
    """Where the products term binds, the row's binding kind and what it rests on (the theorem of record on the subset
    of M it takes, given (H) at its β_w, a weight's, and β_x, an activation's at most β(k), and (E) at its b_y) are
    published's as it gives them, opening with algorithm_policy's `TRACED` as algorithm_policy's rows do, and the
    kinds cell's premises algorithm_policy's; the row's products input names its untraced steps but a step whose
    structure updates nothing (`IDLE`), which has no products of two operands that depend on the update."""
    fx = json.loads(PINNED.read_text())
    m = dataclasses.replace(formalism.from_json(fx["measured"]), products=products_rows(PRODUCT, steps=1e9))
    last = "step 9: the program's last, its structure having no other step, not traced"
    fx |= {"read": fx["read"] | {"untraced": [f"step 0: {SW.IDLE}", last]}}
    one = formalism.published(m, *SW.POINTS["console"], SW.READING)
    he = formalism.bound(m, *SW.POINTS["console"])["inputs"]["fresh_products"]
    assert one["binding_kind"] == "fresh_products" and one["rests_on"] == formalism.he_rests_on(he)
    assert (he["subset"], he["beta_w"], he["beta_x"], he["b_y"]) == ("M", 16, 15.9, 16)
    assert one["rests_on"].startswith(
        f"theorem of record on M, given (H): operands are hard at β_w = 16 and β_x = 15.9 bits a word "
        f"({formalism.BETA_K_LABEL}; the casts of one value count as one word), and (E): exports are wide at b_y = 16 "
        f"bits an output")
    _, _, cells = build(m, fx, monkeypatch)
    assert (cells["binding_kind"], cells["rests_on"]) == ("fresh_products",
                                                          f"{algorithm_policy.TRACED}; {one['rests_on']}")
    assert cells["rests_on"] == algorithm_policy.rests_on(one)
    assert json.loads(cells["kinds"])["rests_on"] == {"fresh_products": formalism.RESTS_ON_CELL["fresh_products"]}
    assert json.loads(cells["input_status"])["products"].startswith(f"assumed: none on 1 untraced step ({last}),")


def test_the_headline_is_the_served_models_own_session_and_a_distillation_rows_student_a_sensitivity(monkeypatch):
    """rho_own, the first ρ column, is published's ρ at the row's own session (algorithm_policy.own_session: 4,096
    tokens in and 1,024 out on the model the row serves, for distillation the start, the teacher) at
    algorithm_policy's OWN_K, beside the session's c and its packed ids; the row's binding_kind and rests_on are the
    console default's, and the build refuses a row whose own session binds otherwise. A distillation row's ρ at its
    student's own session is no column: the details keep it and the note gives it among the sensitivities."""
    fx = json.loads(PINNED.read_text())
    m = dataclasses.replace(formalism.from_json(fx["measured"]), products=products_rows(PRODUCT, steps=1e9))
    assert policy.parameters(STUDENT) == STUDENT["parameters"]
    session, s = algorithm_policy.own_session(fx["shape"]), algorithm_policy.own_session(STUDENT)
    mine = formalism.published(m, session["gates"], algorithm_policy.OWN_K, SW.READING)
    student = formalism.published(m, s["gates"], algorithm_policy.OWN_K, SW.READING)
    fx |= {"read": fx["read"] | {"student": STUDENT}}
    p, details, cells = build(m, fx, monkeypatch)
    assert [c for c in SW.COLUMNS if c.startswith("rho_")][0] == "rho_own"
    assert (cells["rho_own"], cells["own_k_bits"]) == (SW.rho_cell(mine["rho"]), session["k_bits"])
    assert mine["binding_kind"] == cells["binding_kind"] == "fresh_products" and session["k_bits"] == 61_440
    assert cells["rho_own"] != cells["rho_console"] and s["gates"] < session["gates"] < algorithm_policy.CONSOLE[0]
    got = details[0]["student_session"]
    assert (got["gates"], got["k_bits"], got["published"]) == (s["gates"], 61_440, SW.finite(student))
    assert f"; distill at its student's: 20M {SW.figure(student['rho'])}. fresh_old " in p["note"]
    assert not SW.refused(p | {"note": SW.note(p, TABLES, details)})
    real = formalism.published
    monkeypatch.setattr(formalism, "published",
                        lambda *a: real(*a) | ({"binding_kind": "token"} if a[1] == session["gates"] else {}))
    with pytest.raises(SystemExit, match="rho_own binds on token .* but the row's binding_kind and rests_on are the "
                                         "console default's alone"):
        build(m, fx, monkeypatch)


def test_a_row_names_the_capture_code_its_trace_records_and_null_where_it_records_none(monkeypatch):
    """`capture_code` is published's: the VERSION of the code that wrote the row's trace, which a trace records from
    te_core 4.12.14 on, null for an older trace (the pinned row's, which has no product rows, is null too)."""
    fx = json.loads(PINNED.read_text())
    m = formalism.from_json(fx["measured"])
    for code in (None, formalism.CAPTURE_CODE_SINCE):
        traced = dataclasses.replace(m, products=products_rows(PRODUCT, dial=(panels.SERVED[1],), code=code))
        one = formalism.published(traced, *SW.POINTS["console"], SW.READING)
        _, _, cells = build(traced, fx, monkeypatch)
        assert cells["capture_code"] == one["capture_code"] == code


def test_a_generation_runs_record_is_its_rounds_with_gemms_on_served_codes_priced_as_nvfp4():
    """A round keeps the counter's tokens, its prompts' positions; its forward positions (each prompt token and every
    generated token but the last) are what its token dials count."""
    rec = record()
    got = SW.rounds(rec, NOTES, SERVED)
    assert [s["step"] for s in got["steps"]] == [1, 3] == list(SW.round_at(NOTES))
    assert [s["tokens"] for s in got["steps"]] == [2 * 3] * 2
    assert [SW.positions(r) for r in NOTES["rounds"]] == [2 * (3 + 2 - 1)] * 2
    assert SW.forward_positions(NOTES) == 2 * 2 * (3 + 2 - 1)
    assert list(got["structures"]) == ["round"]
    st = got["structures"]["round"]
    assert st == rec["structures"]["round"] | {"log": st["log"]}
    assert st["log"][0] == [*CODED[:3], f"{CODED[3]} {SW.FP4}", *CODED[4:]]
    assert st["log"][1:] == ROUND_LOG[1:]
    ops = {" ".join(str(x) for x in r[:4]): r[4:6] for r in st["log"]}
    assert of_log({"ops": ops})["kinds"]["gemm/fp4"]["macs"] == CODED[5]
    assert got["setup"]["by"] == got["tail"]["by"] == {} and got["setup"]["macs"] == got["tail"]["macs"] == 0
    assert got["checkpoints"] == [{"step": 4}]
    assert set(SW.ROUNDS) <= set(SW.KEYS) and all(v.startswith("assumed: ") for v in SW.ROUNDS.values())
    assert set(SW.LEARNER) <= set(SW.KEYS) and all(v.startswith("assumed: ") for v in SW.LEARNER.values())
    whole = SW.rounds(rec, NOTES, SERVED, learner=True)
    assert whole["steps"] == rec["steps"] and whole["checkpoints"] == [{"step": 4}]
    assert (whole["setup"], whole["tail"]) == (rec["setup"], rec["tail"])
    assert whole["structures"] == rec["structures"] | {"round": st}
    assert SW.coded(rec["structures"]["train"]["log"][0], SERVED)


def test_a_rounds_row_has_no_backward_and_its_token_dials_count_its_forward_positions(monkeypatch):
    """A rounds row's Δ_grad, partial gradient and gradient cut are 0, whatever cut its trace measured (its decode
    loop's values), its input_status carries the rounds' substitutions, and its token dials are checked against its
    rounds' forward positions, which a trace with the decode loop's positions on its token axis counts."""
    fx = json.loads(PINNED.read_text())
    m = formalism.from_json(fx["measured"])
    cut = dataclasses.replace(m, parts=tuple(dataclasses.replace(p, delta_grad_bits=6e8, partial_bits=6e8,
                                                                 grad_bits=6e8) for p in m.parts))
    got = SW.no_backward(cut)
    assert {(p.delta_grad_bits, p.partial_bits, p.grad_bits) for p in got.parts} == {(0.0, 0.0, 0.0)}
    assert got.labels == cut.labels | {"delta_grad_bits": SW.ROUNDS["delta_grad_bits"]}
    assert (got.tokens, got.gates, got.sigma) == (cut.tokens, cut.gates, cut.sigma)
    fwd = 3 * m.tokens
    row = fx["row"] | {"rounds": True, "dials": {"n_params": 10, "tokens": fwd}}
    read = fx["read"] | {"forward_positions": fwd, "axis": [{"step": 1, "traced": fwd, "forward": fwd}]}
    _, _, cells = build(got, fx | {"row": row, "read": read}, monkeypatch)
    st = json.loads(cells["input_status"])
    assert {k: st[k] for k in SW.ROUNDS} == SW.ROUNDS
    miss = re.escape(f"its dial tokens is {m.tokens:g}, its rounds' forward positions are")
    with pytest.raises(SystemExit, match=miss):
        build(got, fx | {"row": row | {"dials": {"tokens": m.tokens}}, "read": read}, monkeypatch)


def test_a_rounds_row_whose_trace_lacks_the_decode_positions_is_null_with_the_reason(monkeypatch):
    """Where a traced round's trace counts fewer tokens than its forward positions (the capture's token axis holding
    its prompts' positions alone), the row is null, nothing priced: every ρ column, its inputs and their sources null,
    its binding kind unmeasured, and the reason its rests_on and its kinds' why; its run, dials, session and capture
    code stay, and the route takes it. A trace counting more, or none of whose traced steps is a round, is refused."""
    traced = {1: {"measured": {"tokens": 2 * 3}}, 2: {"measured": {"tokens": 64}},
              3: {"measured": {"tokens": 2 * (3 + 2 - 1)}}}
    rounds = SW.axis(NOTES, traced)
    assert rounds == [{"step": 1, "traced": 6, "forward": 8}, {"step": 3, "traced": 8, "forward": 8}]
    assert SW.NO_AXIS == "unmeasured: the capture puts decode positions on no token axis"
    fx = json.loads(PINNED.read_text())
    m = dataclasses.replace(formalism.from_json(fx["measured"]), products=products_rows(
        PRODUCT, dial=(panels.SERVED[1],), code=formalism.CAPTURE_CODE_SINCE))
    row = fx["row"] | {"rounds": True, "dials": {"n_params": 10, "tokens": 16}}
    read = fx["read"] | {"forward_positions": 16, "axis": rounds}

    def priced(*args, **kw):
        raise AssertionError("a null row is priced")
    for name in ("at", "published", "bound"):
        monkeypatch.setattr(formalism, name, priced)
    p, details, cells = build(m, fx | {"row": row, "read": read}, monkeypatch)
    null = ("rho_own", "rho_inputs", "input_status", "rho_default", "rho_console", "rho_served")
    assert {c: cells[c] for c in null} == dict.fromkeys(null)
    assert (cells["binding_kind"], cells["rests_on"]) == (algorithm_policy.UNMEASURED, SW.NO_AXIS)
    assert json.loads(cells["kinds"]) == {"bound": dict.fromkeys(formalism.KINDS),
                                          "sensitivity": dict.fromkeys(SW.SENSITIVITIES), "why": SW.NO_AXIS}
    session = algorithm_policy.own_session(fx["shape"])
    assert (cells["workload"], cells["model"], cells["dials"], cells["reading"], cells["te_core"],
            cells["capture_code"]) == (row["workload"], row["model"], SW.compact(row["dials"]), SW.READING, VERSION,
                                       formalism.CAPTURE_CODE_SINCE)
    assert cells["run"].startswith(f"{row['run']} {row['art']}")
    assert (cells["own_c_gates"], cells["own_k_bits"]) == (pytest.approx(session["gates"], rel=1e-6),
                                                           session["k_bits"])
    assert details[0]["unmeasured"] == SW.NO_AXIS and details[0]["read"]["axis"] == rounds
    assert not {"at", "published", "bound", "parts"} & set(details[0])
    assert not SW.refused(p) and not SW.refused(p | {"note": SW.note(p, TABLES, details)})
    with pytest.raises(SystemExit, match=re.escape("(step 1: 9 traced, 8 forward), positions no forward reads")):
        build(m, fx | {"row": row, "read": read | {"axis": [{"step": 1, "traced": 9, "forward": 8}]}}, monkeypatch)
    with pytest.raises(SystemExit, match="no traced step is a round"):
        build(m, fx | {"row": row, "read": read | {"axis": []}}, monkeypatch)


def test_a_read_back_past_the_runs_last_step_is_dropped_and_listed_and_any_other_kept():
    """A trace of a run's last step records the program's code after it, which no step's graph carries (generate.py's
    notes reading the last round's ids), so te_core's refusal doesn't see a read-back listed there; one inside the run
    still refuses the row."""
    late = {"kind": "item() of a int64 tensor", "where": "generate.py:230", "steps": [603, 603], "count": 48}
    inside = {"kind": "item() of a float tensor", "where": "train.py:9", "steps": [602, 603], "count": 1}
    st = {"log": [], "formalism": {"data_dependent": [late, inside], "carried_scalars": None, "affected_share": 0.5,
                                   "u1_bits": 8.0}}
    got, after = SW.past_last({"r": st, "clean": {"formalism": {"data_dependent": [], "carried_scalars": []}}}, 602)
    assert got["r"] == st | {"formalism": st["formalism"] | {"data_dependent": [inside]}}
    assert after == {"r": [late]} and got["clean"] == {"formalism": {"data_dependent": [], "carried_scalars": []}}
    assert formalism.refusal({"formalism": {"trace": got["r"]["formalism"]}}).startswith("data-dependent: not measured")
    alone, _ = SW.past_last({"r": st | {"formalism": st["formalism"] | {"data_dependent": [late]}}}, 602)
    assert formalism.refusal({"formalism": {"trace": alone["r"]["formalism"]}}) is None


def test_a_rounds_row_with_no_traced_round_is_refused(tmp_path, monkeypatch):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "run.json").write_text(json.dumps(record() | {"start": {"name": "s-t20"}}))
    (tmp_path / "meta.json").write_text(json.dumps({"notes": NOTES}))
    monkeypatch.setattr(SW, "fetch", lambda art, *paths: tmp_path)
    monkeypatch.setattr(SW, "start", lambda name: ({"vocab": 1000}, {}))
    with pytest.raises(SystemExit, match="rounds r1: no round is traced: trace its rounds"):
        SW.measured({"workload": "rl", "run": "r1", "art": "art:x", "rounds": True}, {})


@pytest.mark.parametrize("rec, match", [
    (record(round_tokens=8), "8 tokens counted, not its prompts'"),
    (record() | {"steps": record()["steps"][:3]}, "not generate.py's layout"),
    (record(round_log=[*ROUND_LOG, ["bwd", "grad", "mm", json.dumps([[U, 8], [8, U]]), 1, 8 * U * U]]), "a backward"),
    (record() | {"steps": [s | {"struct": "train"} if s["step"] == 3 else s for s in record()["steps"]]},
     "round structure train is a training step's too"),
])
def test_a_record_not_in_generate_pys_layout_or_with_a_round_that_trains_is_refused(rec, match):
    with pytest.raises(ValueError, match=match):
        SW.rounds(rec, NOTES, SERVED)


def test_a_rounds_delta_is_the_served_update():
    tensors = {"embed": ["bfloat16", [1000, U]], "blocks.0.w": ["bfloat16", [U, 2 * U]], "blocks.0.g": ["bfloat16", [U]],
               "norm": ["bfloat16", [U]]}
    lora = SW.round_formalism(NOTES | {"args": NOTES["args"] | {"update": "lora"}}, tensors)
    assert lora == {"delta_bits": NOTES["served_lora_bits"], "labels": SW.ROUND_LABELS}
    assert SW.round_formalism(NOTES, tensors)["delta_bits"] == NOTES["served_code_bits"] + 16 * 2 * U
    bf16_head = NOTES | {"args": NOTES["args"] | {"head": "bf16"}}
    assert SW.round_formalism(bf16_head, tensors)["delta_bits"] == NOTES["served_code_bits"] + 16 * (2 * U + 1000 * U)


#: a traced step's quantities (te_core.capture.circuit.measure's) beside each traced structure's own
TRACE = {"tau_ids": 1.0, "tau_other_bits": 0.0, "a_bits": 256.0, "u_in": [[0.0, 0.0], [1.0, 1024.0]], "u1_bits": 1024.0,
         "tail_share": 0.0, "data_dependent": [], "carried_scalars": []}


def gates_of(log: list, n: int) -> float:
    """A structure's op log over `n` steps in gates, as te_core.formalism.of_record counts a part's."""
    return of_log({"ops": {" ".join(str(x) for x in r[:4]): [r[4] * n, r[5] * n] for r in log}})["gates"]


def test_a_learner_row_is_its_rollouts_plus_its_learner_each_from_its_own_traced_step(tmp_path, monkeypatch):
    """An RL run's learner row is the whole run: its rounds the rollouts, as a rounds row reads them (each GEMM on the
    served codes priced as NVFP4, Δ_grad, partial gradient and gradient cut 0 whatever their trace's cut), and its
    training steps the learner, as its traced step after the first update measured them (its GEMMs as counted, though
    a forward's operand has a served matrix's shape), so Fit binds on the learner's Δ_grad, and Δ is the run's E; its
    parts are named in te_core's order, each with its share of the gates. A row with no training step traced after the
    first update is refused."""
    rec = record()
    rec["structures"]["round"] = rec["structures"]["round"] | {"updated": [[1]]}
    for s in rec["steps"]:
        s["macs"] = 100 if s["struct"] == "round" else 1000
    rec |= {"start": {"name": "s-t20"}, "checkpoints": [{"step": 2}, {"step": 4, "e_bytes": 100}]}
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "run.json").write_text(json.dumps(rec))
    (tmp_path / "meta.json").write_text(json.dumps({"notes": NOTES}))
    monkeypatch.setattr(SW, "fetch", lambda art, *paths: tmp_path)
    tensors = {"embed": ["bfloat16", [1000, U]], "blocks.0.w": ["bfloat16", [U, 2 * U]]}
    monkeypatch.setattr(SW, "start", lambda name: ({"vocab": 1000, "d_model": U}, tensors))

    def ingest(d, ledger, *, structures=None, **kw):
        """te_core.ledger.ingest's row at the run's last checkpoint, past its last step: its formalism block of the
        record with the traced structures, and E's size."""
        r = json.loads((d / "run.json").read_text())
        cp = r["checkpoints"][-1]
        return [{"formalism": formalism.of_record(annotate.with_structures(r, structures)),
                 "output": {"e_bytes": cp["e_bytes"]}, "checkpoint_step": cp["step"], "invalid": []}]
    monkeypatch.setattr(SW, "ingest", ingest)
    learn = {"updated": [[50, 0, 800, 16, 16]], "cut_bits": 640, "formalism": TRACE | {
        "sigma": 0.9, "affected_share": 0.75, "token_factor": 1.0, "delta_grad_bits": 640, "partial_cut_bits": 320,
        "grad_cut_bits": 640}}
    roll = {"updated": [[20, 0, 160, 8, 16]], "cut_bits": 96, "formalism": TRACE | {
        "sigma": 0.8, "affected_share": 0.5, "token_factor": 8 / 6, "delta_grad_bits": 96, "partial_cut_bits": 96,
        "grad_cut_bits": 96}}

    def trace(structure: dict, tokens: int) -> dict:
        return {"trace": {"structure": structure}, "measured": {"tokens": tokens}, "from": "art:t", "te_core": VERSION}
    traced = {1: trace(roll, 8), 2: trace(learn, 64), 3: trace(roll, 8)}
    row = {"workload": "rl", "run": "r1", "art": "art:x", "rounds": True, "learner": True}
    m, _, read = SW.measured(row, traced)
    assert read["parts"] == ["setup", "learner", "rollouts"] and read["round_steps"] == [1, 3]
    assert read["axis"] == [{"step": 1, "traced": 8, "forward": 8}, {"step": 3, "traced": 8, "forward": 8}]
    setup, learner, rollouts = m.parts
    assert (learner.delta_grad_bits, learner.partial_bits, learner.grad_bits) == (640, 320, 640)
    assert (rollouts.delta_grad_bits, rollouts.partial_bits, rollouts.grad_bits) == (0.0, 0.0, 0.0)
    assert m.fit_delta_grad_bits == 640 and m.delta_bits == 8 * 100
    assert (learner.tokens, rollouts.tokens) == (2 * 64, 2 * SW.positions(NOTES["rounds"][0]))
    marked = SW.rounds(record(), NOTES, SERVED)["structures"]["round"]["log"]
    assert learner.gates == gates_of(rec["structures"]["train"]["log"], 2)
    assert rollouts.gates == gates_of(marked, 2) and rollouts.token_gates == pytest.approx(0.8 * rollouts.gates)
    assert read["shares"] == {"setup": setup.gates / m.gates, "learner": learner.gates / m.gates,
                              "rollouts": rollouts.gates / m.gates}
    assert sum(read["shares"].values()) == pytest.approx(1.0) and read["learner_tokens"] == learner.tokens
    for only in ({1: traced[1], 3: traced[3]}, {0: trace(learn, 64), 1: traced[1]}):
        with pytest.raises(SystemExit, match="rl r1: no learner step is traced: trace a training step after its "
                                             "first update"):
            SW.measured(row, only)


def test_a_learner_rows_run_cell_gives_its_parts_shares_and_its_status_the_rollouts_substitutions(monkeypatch):
    """A learner row's run cell ends with its two parts' shares of its gates, and its input_status carries the
    rollouts' substitutions beside the learner's measured quantities (`LEARNER`), w_fit the rollouts' where theirs is
    the least w, else measured; its Δ_grad is the learner's, measured. Its token dials are its learner's tokens and
    its rollouts' forward positions per learner token, each refused where it misses the run's."""
    fx = json.loads(PINNED.read_text())
    m = formalism.from_json(fx["measured"])
    setup, learn = m.parts
    roll = formalism.Part(learn.gates / 99, 1e6, 0.9 * learn.gates / 99, 0.5 * learn.gates / 99, 3, 0.0, 0.0, 0.0)
    m = dataclasses.replace(m, parts=(setup, learn, roll))
    names = ["setup", "learner", "rollouts"]
    share = SW.shares(m, names)
    assert share["rollouts"] == pytest.approx(0.01, rel=1e-6) and sum(share.values()) == pytest.approx(1.0)
    dials = {"n_params": 10, "tokens": learn.tokens, "samples_per_trained": roll.tokens / learn.tokens}
    row = fx["row"] | {"workload": "rl", "rounds": True, "learner": True, "dials": dials}
    read = fx["read"] | {"forward_positions": roll.tokens, "axis": [{"step": 1, "traced": 8, "forward": 8}],
                         "parts": names, "shares": share, "learner_tokens": learn.tokens}
    for k, v in (("tokens", roll.tokens + learn.tokens), ("samples_per_trained", 1.0)):
        with pytest.raises(SystemExit, match=re.escape(f"its dial {k} is {v:g}, its ")):
            build(m, fx | {"row": row | {"dials": dials | {k: v}}, "read": read}, monkeypatch)
    _, details, cells = build(m, fx | {"row": row, "read": read}, monkeypatch)
    assert cells["run"] == f"{row['run']} {row['art']} ({row['about']}; {SW.split(share)})"
    assert SW.split(share) == "the rollouts 1% of its gates and the learner 99%"
    st = json.loads(cells["input_status"])
    assert {k: st[k] for k in SW.LEARNER} == SW.LEARNER and st["w_fit_gates"] == SW.LEARNER_W_FIT
    assert st["delta_grad_bits"] == "measured" and m.fit_delta_grad_bits == learn.delta_grad_bits
    assert details[0]["read"]["shares"] == share and len(details[0]["parts"]) == 3
    slow = dataclasses.replace(m, parts=(setup, learn, dataclasses.replace(roll, tokens=1.0)))
    assert "w_fit_gates" not in SW.learner_status(slow, names)


def test_the_route_takes_a_cell_of_4096_characters_and_refuses_a_longer_one_a_long_note_or_a_long_body():
    p = {"title": "t", "note": "n", "source": "s", "columns": ["a", "b"], "rows": [["x" * console.CELL_MAX, 1.5]]}
    assert not SW.refused(p)
    long = "x" * (console.CELL_MAX + 1)
    assert SW.refused(p | {"rows": [[long, 1.5]]}) == [f"row 0's a is {long!r:.40}"]
    assert SW.refused(p | {"rows": [["x", True]]}) == ["row 0's b is True"]
    assert SW.refused(p | {"note": "n" * (console.NOTE_MAX + 1)}) == \
        [f"its note is {console.NOTE_MAX + 1} characters, over {console.NOTE_MAX}"]
    big = p | {"rows": p["rows"] * (console.BODY_MAX // console.CELL_MAX)}
    assert SW.refused(big) == [f"its body is {len(console.body(big))} bytes, over {console.BODY_MAX}"]



def test_a_run_a_plan_restates_is_read_restated_and_its_stored_traces_against_the_restated_count(monkeypatch):
    """A plan's entry restates the record's tokens a step as the scorer's restate.py does (a structure the entry lists
    states its own count, every other the factor times its record's), and each stored trace of the run is read
    against the restated count of its step: its step_tokens and its products block's tokens are that count, its
    token_factor its own tokens over it, and the stored trace is left as it was. A record the plan didn't read, a
    restating that isn't the plan's and a products block that doesn't count the record's tokens are refused. A build
    hands a row its run's entry, and no other row one."""
    before = {"steps": [{"step": i, "struct": sid, "tokens": 4} for i, sid in enumerate("abc")]}
    after = {"steps": [{"step": i, "struct": sid, "tokens": n} for i, (sid, n) in enumerate(zip("abc", (8, 4, 8)))]}
    raw = json.dumps(before).encode()
    entry = {"factor": 2.0, "tokens": {"b": {"4": 4}}, "before_sha256": hashlib.sha256(raw).hexdigest(),
             "after_sha256": hashlib.sha256(json.dumps(after).encode()).hexdigest()}

    def trace(tokens: int) -> dict:
        f = {"tokens": tokens, "step_tokens": 4, "token_factor": tokens / 4, "products": {"tokens": 4, "rows": []}}
        return {"trace": {"formalism": copy.deepcopy(f), "structure": {"formalism": copy.deepcopy(f)}},
                "measured": {"tokens": tokens, "step_tokens": 4, "token_factor": tokens / 4}}

    stored = {0: trace(8), 1: trace(4)}
    record, read = SW.restated(raw, entry, stored)
    assert record == after and stored == {0: trace(8), 1: trace(4)}
    for f in (read[0]["trace"]["formalism"], read[0]["trace"]["structure"]["formalism"]):
        assert (f["step_tokens"], f["token_factor"], f["products"]["tokens"]) == (8, 1.0, 8)
    assert read[0]["measured"] == {"tokens": 8, "step_tokens": 8, "token_factor": 1.0} and read[1] == stored[1]
    with pytest.raises(SystemExit, match="isn't the one the plan read"):
        SW.restated(raw + b" ", entry, stored)
    with pytest.raises(SystemExit, match="isn't the plan's"):
        SW.restated(raw, entry | {"factor": 4.0}, stored)
    off = {0: trace(8)}
    off[0]["trace"]["formalism"]["products"]["tokens"] = 2
    with pytest.raises(SystemExit, match="products block counts 2 tokens, not the record's 4"):
        SW.restated(raw, entry, off)

    fx = json.loads(PINNED.read_text())
    m, got = formalism.from_json(fx["measured"]), []
    monkeypatch.setattr(SW, "traces", lambda arts: {})
    monkeypatch.setattr(SW, "measured", lambda *args: got.append(args[2:]) or (m, fx["shape"], fx["read"]))
    for plan in (None, {}, {"another run": entry}, {fx["row"]["run"]: entry}):
        SW.build({"rows": [fx["row"]]}, REVS, stamp="2026-11-03T00:00:00Z", plan=plan)
    assert got == [(), (), (), (entry,)]
