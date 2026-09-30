"""The attempt ledger (`harness/ledger.py`, written by `harness/verify.py`): every arm row of a verified run becomes one
``kernel-attempt/v1`` record named by its kernel variant, with its parent, tier, gate version, metrics, step times, γ at
both FP32 prices and the verifier's lines; a verifier's negative control is judged; records publish into the evidence
store once (content-addressed, labels not repeated); and the store's records come back as `panel.py` rows that follow the
panel's rules (a screen never plots, a measurement stands only with every verifier field)."""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "harness"))

import arm as A  # noqa: E402
import ledger as L  # noqa: E402
import verify as V  # noqa: E402

RUN, COMMIT = "r20260930-170000-abcd", "0123456789abcdef0123456789abcdef01234567"
PREFILL, DECODE = "m8192-n8192-k8192", "m32-n8192-k8192"
SHAPES = {PREFILL: ("prefill", 8192), DECODE: ("decode", 32)}

#: prints one line per transcript and control under the run: ACCEPT the transcripts, REJECT the controls (exit 0, as a
#: verifier does when it rejected what it was meant to), and ACCEPT a control that says "accept me"
VERIFIER = """
import hashlib, sys
from pathlib import Path
run = Path(sys.argv[1])
for m in sorted(run.glob("*/*/*/*.json")):
    raw = m.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    ok = m.parts[-4] == "transcripts" or b"accept me" in raw
    print(("ACCEPT" if ok else "REJECT") + f" {run.name}/{m.relative_to(run)} sha256:{sha}")
"""


@pytest.fixture(autouse=True)
def store_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "absent-store.toml"))
    monkeypatch.setenv("RESEARCH_REPO", str(tmp_path))
    for k in ("RESEARCH_RUN_ID", "RESEARCH_SOURCE_SHA", "GPU_LEASE_WHO", "POUW_TIER", "POUW_LEDGER_PUBLISH"):
        monkeypatch.delenv(k, raising=False)


def kernel_dir(tmp_path: Path, text: str = "kernel v1") -> Path:
    d = tmp_path / "kernel"
    d.mkdir(exist_ok=True)
    (d / "run.py").write_text("# the pipeline\n")
    (d / "k.cu").write_text(text)
    (d / "k.cubin").write_bytes(hashlib.sha256(text.encode()).digest())
    return d


def variant(tmp_path: Path, text: str = "kernel v1") -> dict:
    d = kernel_dir(tmp_path, text)
    return L.variant_record(family="pearl-c", device="sm120", schemes={"pearl-c-sm120-v2-hot-h3"}, sources=["run.py", "k.cu"],
                            root=d, binaries=[d / "k.cubin"], prices="cast-8.72")


GAMMA = {"8": {"cap": "3/1000", "floor": "1/1000"}, "1047/125": {"cap": "29/10000", "floor": "1/1000"}, "basis": "Derived"}


def result(run_dir: Path, var: dict | None, *, gamma=GAMMA, control: bool = True, lineage: dict | None = None) -> Path:
    """A bench.py result for one arm `pc` at both headlines, with gates, controls, the SASS gate and a decode split."""
    shapes = []
    for label, (phase, m) in SHAPES.items():
        p = {"phase": phase, "shape": label, "slowdown": 1.21 if phase == "prefill" else 1.25, "hash_free": 1.1,
             "baseline": "fp8-e4m3", "baseline_kernel": "lt13_algo_5", "baseline_ms": 1.446,
             "sm_clock_arm": "2090,2092", "sm_clock_base": "2091,2093"}
        row = {"graph": {"total_x": p["slowdown"], "hash_free_x": 1.1, "total_ms": 1.75, "hash_free_ms": 1.59,
                         "hashing_ms": 0.16, "baseline": "fp8-e4m3", "baseline_ms": 1.446},
               "gamma": gamma, "w_ref": "100", "credited": "99", "unit": "fp8-mac", "panel": p}
        shape = {"label": label, "m": m, "k": 8192, "n": 8192, "phase": phase, "arms": {"pc": row}}
        if phase == "decode":
            p |= {"decode_method": "dependent-chain", "chain_length": 32}
            shape["chain"] = {"arms": {"pc": {"split": {"graph": {"before_ms": 0.004, "during_ms": 0.05, "after_ms": 0.012,
                                                                  "derive_ms": 0.004, "side_ms": 0.058,
                                                                  "hidden_by_side_ms": 0.008}}}}}
        shapes.append(shape)
    decl = {"per": "run", "transcript": "transcripts/{arm}/{shape}/manifest.json",
            "commands": [["{python}", str(run_dir.parent / "v.py"), "{run_dir}"]]}
    if control:
        decl["control"] = "negative/{arm}/{shape}/manifest.json"
    build = {"binaries": [], "panel": {"line": "pearl-c-sm120", "version": "v2-hot"}, "scheme": "pearl-c-sm120-v2-hot-h3"}
    if var is not None:
        build["variant"] = var
    build["lineage"] = lineage if lineage is not None else {
        "parent": "art:" + "1" * 64, "approach": "pouw/pearl-c-sm120", "mechanism": ["hash:side-stream", "cast:packed-sts64"],
        "hypothesis": "tile hashing on the side lane hides it under the next call",
        "phase_groups": {"before": ["a_tree", "forming"], "gemm": ["gemm", "cleanup"], "after": ["tile_hash", "screen"]}}
    arms = {"pc": {"stream": "fp8", "precision": "fp8-e4m3", "build": build, "verifier": A.check_verifier("pc", {"verifier": decl})}}
    gates = [{"key": ["arm", label, "pc", role], "ok": True, "detail": {}} for label in SHAPES for role in ("hash", "nohash")]
    negatives = [{"key": ["known-bad", PREFILL, "pc", "hash", "flip-output-byte"], "rejected": True}]
    sass = {"version": "sass-gate/v4", "ok": True,
            "binaries": [{"source": "arm:pc", "ok": True, "cubins": [{"toolkit": "12.9.86", "functions": 3}]}]}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir.parent / "v.py").write_text(VERIFIER)
    for label in SHAPES:
        for top, text in (("transcripts", f'{{"shape": "{label}"}}'), ("negative", f'{{"wrote": "nothing", "at": "{label}"}}')):
            f = run_dir / top / "pc" / label / "manifest.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
    path = run_dir / "bench.json"
    path.write_text(json.dumps({"harness": {"interface": A.VERSION}, "shapes": shapes, "arms": arms, "gates": gates,
                                "negative_controls": negatives, "sass_gate": sass, "clocks": {"label": "locked-2100"},
                                "versions": {"driver": "580.173.02", "nvcc": "12.9.86"}}))
    return path


def verify(path: Path, *extra: str) -> tuple[int, dict]:
    code = V.main([str(path), "--run-id", RUN, "--commit", COMMIT, "--by", "bc-test", *extra])
    return code, json.loads(path.read_text())


def records(run_dir: Path) -> dict[str, dict]:
    return {r["meta"]["shape"]: r for r in L.read(run_dir)}


# ---- the variant ---------------------------------------------------------------------------------------------------------------


def test_a_variant_is_named_by_its_sources_digest_in_any_order_and_a_changed_source_is_a_new_variant(tmp_path):
    v = variant(tmp_path)
    assert v["id"] == f"pearl-c-sm120/{v['source_sha256'][:12]}"
    d = tmp_path / "kernel"
    assert L.source_sha256(["k.cu", "run.py"], d) == L.source_sha256([d / "run.py", "k.cu"], d) == v["source_sha256"]
    assert v["binary_sha256"] == hashlib.sha256((d / "k.cubin").read_bytes()).hexdigest()
    assert variant(tmp_path, "kernel v1, unroll 2")["id"] != v["id"]
    with pytest.raises(L.LedgerError, match="twice"):
        L.source_sha256(["k.cu", "k.cu"], d)
    with pytest.raises(L.LedgerError, match="at least one source"):
        L.source_sha256([], d)


def test_the_variant_and_the_groups_are_verity_pouw_serving_s_when_it_is_importable(tmp_path):
    S = L.serving()
    if S is None:
        pytest.skip("verity_pouw.serving (#567) isn't on this tree yet")
    v = variant(tmp_path)
    kv = S.KernelVariant(family="pearl-c", device="sm120", schemes=frozenset(v["schemes"]), schedules=frozenset(v["schedules"]),
                         shapes=S.ShapeRule(), source_sha256=v["source_sha256"], binary_sha256=v["binary_sha256"],
                         loader=v["loader"], prices=v["prices"])
    assert kv.id == v["id"]
    assert tuple(S.GROUPS) == L.GROUPS
    gate = records_of(tmp_path)[PREFILL]["meta"]["gate"]
    assert set(gate) == set(S.GateRecord.__dataclass_fields__)


def records_of(tmp_path: Path) -> dict[str, dict]:
    run_dir = tmp_path / RUN
    verify(result(run_dir, variant(tmp_path)), "--tier", "2b")
    return records(run_dir)


# ---- γ at both FP32 prices -----------------------------------------------------------------------------------------------------


def test_gamma_publishes_the_larger_of_the_two_prices_and_only_with_both():
    g = L.gamma_block(GAMMA, None)
    assert g["published"] == "3/1000" and g["missing"] == []
    assert set(g["by_price"]) == {"8", "1047/125"}
    spelled = L.gamma_block({"8.00": "0.0036162", "8.376": "0.0036218", "reading": "forming-credited"}, None)
    assert spelled["published"] == str(Fraction("0.0036218")) and spelled["reading"] == "forming-credited"
    one = L.gamma_block({"8": {"cap": "3/1000"}}, None)
    assert one["published"] is None and one["missing"] == ["gamma at FADD 1047/125 (cap)"]
    flat = L.gamma_block({"cap": "3/1000", "floor": "1/1000", "basis": "one price"}, None)
    assert flat["published"] is None and "unstated" in flat["by_price"] and len(flat["missing"]) == 2
    assert L.gamma_block(flat, {"8": "1/100", "1047/125": "1/50"})["published"] == "1/50"
    assert len(L.gamma_block(None, None)["missing"]) == 2


# ---- verify.py writes the records ----------------------------------------------------------------------------------------------


def test_verify_writes_one_record_per_arm_and_shape_with_everything_the_ledger_keeps(tmp_path, capsys):
    run_dir = tmp_path / RUN
    var = variant(tmp_path)
    code, res = verify(result(run_dir, var), "--tier", "2b")
    assert code == V.EXIT_OK and res["verification"]["ok"], res["verification"]
    assert sorted(res["ledger"]["records"]) == [f"ledger/pc/{s}.json" for s in sorted(SHAPES)]
    recs = records(run_dir)
    m = recs[DECODE]["meta"]
    assert m["variant"] == var["id"] and m["variant_record"] == var and m["parent"] == "art:" + "1" * 64
    assert recs[DECODE]["refs"] == {"parent": "art:" + "1" * 64}
    assert m["tier"] == "2b" and m["run"] == RUN and m["harness"] == COMMIT and m["by"] == "bc-test"
    assert m["gates"] == {"version": L.GATES, "arm_interface": A.VERSION, "verify": V.VERSION, "sass_gate": "sass-gate/v4"}
    assert m["gate"] == {"variant": var["id"], "run": RUN, "gates": L.GATES, "shapes": [[8192, 8192]], "exact": True,
                         "poisoned_accepted": True, "controls_rejected": True, "schedules_equal": True, "sass_ok": True,
                         "toolkit": "12.9.86", "prices": "cast-8.72"}
    assert m["metrics"]["slowdown"] == 1.25 and m["metrics"]["decode_method"] == "dependent-chain"
    assert m["metrics"]["sm_clock"] == {"arm": [2090.0, 2092.0], "base": [2091.0, 2093.0]}
    assert m["steps"]["phases"]["gemm"] == 0.05 and m["steps"]["hashing_ms"] == 0.16
    assert m["steps"]["phase_groups"]["after"] == ["tile_hash", "screen"]
    assert m["gamma"]["published"] == "3/1000"
    assert m["hypothesis"] == "tile hashing on the side lane hides it under the next call"
    assert m["mechanism"] == ["cast:packed-sts64", "hash:side-stream"] and m["approach"] == "pouw/pearl-c-sm120"
    assert m["verify"]["accepted"] and RUN in m["verify"]["transcript"]
    assert m["verify"]["negative_control"].startswith(f"REJECT {RUN}/negative/pc/{DECODE}/")
    assert m["missing"] == []
    assert recs[DECODE]["labels"] == [[L.KERNEL_OUTCOME, "screened"]]
    assert "phase" not in recs[PREFILL]["meta"]["steps"] or recs[PREFILL]["meta"]["steps"].get("phases") is None
    args = res["shapes"][1]["arms"]["pc"]["panel"]["panel_args"]
    assert args[args.index("--negative-control") + 1].startswith("REJECT")
    assert capsys.readouterr().out.count("--negative-control") == 2


def test_the_run_s_flags_win_over_the_arm_s_lineage_and_a_bare_arm_still_gets_a_record(tmp_path):
    run_dir = tmp_path / RUN
    code, _ = verify(result(run_dir, None, gamma=None, lineage={}), "--hypothesis", "  a   run's own\nline ",
                     "--parent", "pearl-c-sm120/0123456789ab", "--outcome", "dropped", "--reason", "slower than its parent")
    assert code == V.EXIT_OK
    r = records(run_dir)[PREFILL]
    m = r["meta"]
    assert m["hypothesis"] == "a run's own line" and m["parent"] == "pearl-c-sm120/0123456789ab" and r["refs"] == {}
    assert m["tier"] == "2a" and r["labels"] == [[L.KERNEL_OUTCOME, "dropped"], [L.KERNEL_REASON, "slower than its parent"]]
    assert m["missing"] == ["variant", "gamma at FADD 8 (cap)", "gamma at FADD 1047/125 (cap)"]
    assert m["gate"]["variant"] is None and m["gate"]["controls_rejected"]


def test_bad_run_flags_are_usage_errors_and_nothing_is_written(tmp_path):
    run_dir = tmp_path / RUN
    path = result(run_dir, None)
    assert V.main([str(path), "--run-id", RUN, "--commit", COMMIT, "--outcome", "dropped"]) == V.EXIT_USAGE
    assert V.main([str(path), "--run-id", RUN, "--commit", COMMIT, "--hypothesis", "x" * 241]) == V.EXIT_USAGE
    assert not (run_dir / "ledger").exists()
    assert V.main([str(path), "--run-id", RUN, "--commit", COMMIT, "--no-ledger"]) == V.EXIT_OK
    assert not (run_dir / "ledger").exists()


def test_a_control_the_verifier_accepts_rejects_its_row(tmp_path):
    run_dir = tmp_path / RUN
    path = result(run_dir, None)
    (run_dir / "negative" / "pc" / PREFILL / "manifest.json").write_text('{"accept me": 1}')
    code, res = verify(path)
    assert code == V.EXIT_REJECTED
    p = res["shapes"][0]["arms"]["pc"]["panel"]
    assert p["verification"] == {"accepted": False, "why": "the verifier accepted the negative control"}
    rows = {r["shape"]: r for r in res["verification"]["rows"]}
    assert rows[DECODE]["accepted"] and rows[DECODE]["control"]["rejected"]
    assert not records(run_dir)[PREFILL]["meta"]["verify"]["accepted"]


def test_a_missing_control_leaves_the_row_accepted_without_a_negative_control(tmp_path):
    run_dir = tmp_path / RUN
    path = result(run_dir, variant(tmp_path))
    (run_dir / "negative" / "pc" / DECODE / "manifest.json").unlink()
    code, res = verify(path)
    assert code == V.EXIT_OK
    p = res["shapes"][1]["arms"]["pc"]["panel"]
    assert p["verification"] == {"accepted": True} and "negative_control" not in p
    assert "--negative-control" not in p["panel_args"]
    assert records(run_dir)[DECODE]["meta"]["missing"] == ["negative control"]


def test_per_shape_control_runs_don_t_fail_the_step(tmp_path):
    run_dir = tmp_path / RUN
    path = result(run_dir, None)
    script = tmp_path / "s.py"
    script.write_text("import hashlib, sys\nraw = open(sys.argv[1], 'rb').read()\nsha = hashlib.sha256(raw).hexdigest()\n"
                      "neg = b'nothing' in raw\nprint(('REJECT' if neg else 'ACCEPT') + ' ' + sys.argv[2] + ' sha256:' + sha)\n"
                      "sys.exit(1 if neg else 0)\n")
    res = json.loads(path.read_text())
    res["arms"]["pc"]["verifier"] = A.check_verifier("pc", {"verifier": {
        "transcript": "transcripts/{arm}/{shape}/manifest.json", "control": "negative/{arm}/{shape}/manifest.json",
        "commands": [["{python}", str(script), "{transcript}", "{run_id}"]]}})
    path.write_text(json.dumps(res))
    code, res = verify(path)
    assert code == V.EXIT_OK and res["verification"]["ok"]
    cmds = res["verification"]["commands"]
    assert [c["control"] for c in cmds] == [False, True, False, True] and [c["exit"] for c in cmds] == [0, 1, 0, 1]
    control = (run_dir / "negative" / "pc" / PREFILL / "manifest.json").read_bytes()
    assert res["shapes"][0]["arms"]["pc"]["panel"]["negative_control"] == f"REJECT {RUN} sha256:" + hashlib.sha256(
        control).hexdigest()


def test_check_verifier_refuses_a_bad_control():
    good = {"transcript": "t/{arm}/{shape}.json", "commands": [["v"]]}
    assert A.check_verifier("x", {"verifier": good | {"control": "n/{arm}/{shape}.json"}})["control"] == "n/{arm}/{shape}.json"
    with pytest.raises(A.ArmError, match="control must be another file"):
        A.check_verifier("x", {"verifier": good | {"control": good["transcript"]}})
    with pytest.raises(A.ArmError, match="control .* must name"):
        A.check_verifier("x", {"verifier": good | {"control": "n/{arm}.json"}})


# ---- the store -----------------------------------------------------------------------------------------------------------------


def test_publishing_twice_is_one_record_each_and_the_labels_are_not_repeated(tmp_path):
    run_dir = tmp_path / RUN
    verify(result(run_dir, variant(tmp_path)), "--tier", "2b", "--publish")
    res = json.loads((run_dir / "bench.json").read_text())
    first = res["ledger"]["published"]
    assert len(first) == 2 and all(x["labels"] == [L.KERNEL_OUTCOME] for x in first)
    assert L.main(["publish", str(run_dir), "--by", "bc-test"]) == 0
    st = L.open_store()
    again = L.publish(L.read(run_dir), store=st, by="bc-test")
    assert sorted(x["art"] for x in again) == sorted(x["art"] for x in first) and all(x["labels"] == [] for x in again)
    art = first[0]["art"]
    assert st.get_manifest(art).kind == L.KIND
    assert [lab.value for lab in st.labels(art, key=L.KERNEL_OUTCOME)] == ["screened"]
    assert len(L.attempts(st)) == 2


def test_an_outcome_label_is_refused_off_vocabulary_and_changes_the_panel_row(tmp_path):
    run_dir = tmp_path / RUN
    verify(result(run_dir, variant(tmp_path)), "--tier", "2b", "--publish")
    st = L.open_store()
    art = next(a for a, m, _ in L.attempts(st) if m["shape"] == PREFILL)
    with pytest.raises(L.LedgerError, match="needs a one-line reason"):
        L.outcome(art, "voided", reason=None, by="bc-test", store=st)
    assert L.outcome(art, "voided", reason="timed another commitment", by="bc-test", store=st) == [L.KERNEL_OUTCOME,
                                                                                                    L.KERNEL_REASON]
    assert L.outcome(art, "voided", reason="timed another commitment", by="bc-test", store=st) == []
    row = next(r for r in L.panel_rows(L.attempts(st)) if r["record"] == art)
    assert row["voided"] == "timed another commitment" and "outcome voided: timed another commitment" in row["description"]


def test_store_records_come_back_as_panel_rows_under_the_panel_s_rules(tmp_path):
    run_dir = tmp_path / RUN
    var = variant(tmp_path)
    verify(result(run_dir, var), "--tier", "2b", "--publish")
    rows = {r["shape"]: r for r in L.panel_rows(L.attempts())}
    r = rows[DECODE]
    assert r["kind"] == "measured" and r["line"] == "pearl-c-sm120" and r["version"] == "v2-hot" and r["attempt"] is None
    assert r["verifier"]["negative_control"].startswith("REJECT") and r["sm_clock_mhz"]["baseline"] == [2091.0, 2093.0]
    assert r["gamma"] == 0.003 and r["decode_method"] == "dependent-chain" and r["precision"] == "fp8"
    assert r["source"].endswith("(kernel-attempt/v1)") and var["id"] in r["description"] and "store_refused" not in r

    screen = tmp_path / "screen" / RUN
    verify(result(screen, var), "--publish")
    shapes = {x["shape"] for x in L.panel_rows(L.attempts())}
    assert {f"{PREFILL}:screen", f"{DECODE}:screen"} <= shapes

    bare = tmp_path / "bare" / RUN
    path = result(bare, var)
    (bare / "negative" / "pc" / PREFILL / "manifest.json").unlink()
    verify(path, "--tier", "2b", "--publish")
    refused = [x for x in L.panel_rows(L.attempts()) if x["run"] == RUN and x.get("store_refused")]
    assert refused and refused[0]["store_refused"] == "no verifier reject line, naming the run, on the negative control"
    assert "verifier" not in refused[0]


def test_measured_refusal_follows_panel_append():
    meta = {"run": RUN, "gates": {"version": L.GATES},
            "verify": {"accepted": True, "accept": f"ACCEPT {RUN}", "transcript": f"{RUN}:t sha256:x",
                       "negative_control": f"REJECT {RUN}/negative"},
            "metrics": {"sm_clock": {"arm": [2090.0], "base": [2091.0]}}}
    assert L.measured_refusal(meta) is None
    assert "predates" in L.measured_refusal(meta | {"gates": {"version": "pouw-gates/1"}})
    assert "naming the run" in L.measured_refusal(meta | {"verify": meta["verify"] | {"negative_control": "REJECT other"}})
    assert "doesn't name the run" in L.measured_refusal(meta | {"verify": meta["verify"] | {"transcript": "elsewhere"}})
    assert "SM clocks" in L.measured_refusal(meta | {"metrics": {"sm_clock": {"arm": [2090.0], "base": []}}})


def test_the_ledger_s_words_are_the_store_s():
    _, vocab = L.store_module()
    from research.store import kinds

    assert L.OUTCOMES == vocab.KERNEL_OUTCOMES
    assert (L.KERNEL_OUTCOME, L.KERNEL_REASON) == (vocab.KERNEL_OUTCOME, vocab.KERNEL_REASON)
    assert vocab.check(L.KERNEL_OUTCOME, "frontier") is None and vocab.check(L.KERNEL_OUTCOME, "maybe")
    assert L.KIND in kinds.KINDS
