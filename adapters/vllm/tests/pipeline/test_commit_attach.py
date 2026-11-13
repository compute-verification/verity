"""`verity_vllm.pipeline.commit_attach`: the Commit's acquisition on a built committer, for its required-value manifest.
The committer, the engine's model and each step that needs them (the hidden source's build, the taps, the MoE block source,
the acquisition plan and its installation) are stand-ins that keep their calls; the manifest's reading is the real one
(`check.commit_rules.hidden_classes_required`, `query.manifest.format.promoted_observation`,
`acquire.fa2_plane_classes`).  What is held: which sources are attached and which refused, the required classes that come
back, what is armed on the committer, the record written, and every line.  CPU only."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import verity_vllm.pipeline.commit_attach as CA
from verity_vllm.acquire import install, stage
from verity_vllm.acquire.sources import hidden_source as HS
from verity_vllm.acquire.sources import moe_source
from verity_vllm.check.commit_rules import required_classes_default

Record = dict[str, Any]
Lines = Callable[[], list[str]]

FA2, FA3 = "fa2_hidden_m1_stream", "fa3_hidden_m1_stream"
CLASSES = ("module_outputs", FA2)
TAP = "/taps/verity_fa2_matReq.so"
SERVED = object()          # a served source: the committer wraps nothing, so no acquisition plan is made
ENGINE = SimpleNamespace(runner="runner")
PLAN = SimpleNamespace(digest="d" * 16)


def manifest(identities: list[Record] | None = None, /, **by_family: int) -> Record:
    return {"populations": {"by_family": by_family}, "identities": identities or []}


def promoted(op: str = "model.layers.0.norm", consumers: tuple[str, ...] = ("model.layers.0.mlp",)) -> Record:
    return {"op_path": op, "output_member": "out", "promoted": True, "consumers": list(consumers), "width": 8, "dtype": "bf16",
            "spec": "RmsNorm_v1"}


def options(**given: Any) -> SimpleNamespace:
    defaults: Record = {"hidden_so": None, "hidden_ns": None, "tap_cap_mb": None, "out": None, "tap_sources": "the taps' options",
                        "program_dir": None, "programs_root": None, "moe_source": "off", "required_manifest": "required_manifest.json",
                        "required_classes": None}
    return SimpleNamespace(**defaults | given)


def committer(*without: str, modules: tuple[str, ...] = ("model.layers.0.mlp",)) -> SimpleNamespace:
    """A committer's face as `attach_of_record` reads it, less the attributes named."""
    com = SimpleNamespace(name="stub", sources=[], _modules=[], acquire_inputs={}, identity_check=lambda: None,
                          model=SimpleNamespace(named_modules=lambda: [(m, object()) for m in modules]))
    com.attach_source = com.sources.append
    for name in without:
        delattr(com, name)
    return com


class Source:
    """A hidden source, as built: only what it was built from."""

    def __init__(self, **built: Any) -> None:
        self.built = built


class BoundedSource(Source):
    """A hidden source whose tap is one bounded device buffer."""
    tap_cap_bytes = 64 << 20
    telemetry_path = "unset"


class UnboundedSource(BoundedSource):
    tap_cap_bytes = 0


class Docs:
    """The stage's documents: its lines and its plan."""

    def __init__(self, plan: Any) -> None:
        self.plan = plan

    def log_lines(self, tag: str) -> list[str]:
        return [f"[{tag}] runtime tree 3 modules", f"[{tag}] acquisition plan abcd"]


class Acquisition:
    """Each step that needs the engine's objects, a stand-in that keeps its call."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, source: type[Source] = Source, layers: tuple[str, ...] = (),
                 tapped: tuple[str, ...] = (), plan: Any = PLAN, installed: Any = None) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.source, self.layers, self.tapped, self.plan = source, layers, tapped, plan
        self.installed: Any = {"modules_after": ["m0", "m1"], "plan_inputs": ["m2"], "deferred_sites": ["s0"]} if installed is None else installed
        monkeypatch.setattr(HS, "make_hidden_source", self.make_hidden_source)
        monkeypatch.setattr(CA.TAPS, "attach", self.taps)
        monkeypatch.setattr(moe_source, "MoeBlockSource", self.moe)
        monkeypatch.setattr(stage, "plan_at_stage", self.plan_at_stage)
        monkeypatch.setattr(install, "apply_plan_to_committer", self.apply)
        monkeypatch.setattr(CA.settings, "put", lambda name, value: self.calls.append(("setting", name, value)))

    def named(self, name: str) -> list[tuple[Any, ...]]:
        return [c for c in self.calls if c[0] == name]

    def make_hidden_source(self, com: Any, so: str, **kw: Any) -> Source:
        self.calls.append(("hidden", com, so, kw))
        return self.source(so=so, **kw)

    def taps(self, com: Any, required_manifest: Any, classes: tuple[str, ...], taps: Any, **kw: Any) -> tuple[str, ...]:
        self.calls.append(("taps", com, required_manifest, classes, taps, kw))
        return classes + self.tapped

    def moe(self, com: Any, model: Any) -> SimpleNamespace:
        self.calls.append(("moe", com, model))
        return SimpleNamespace(layers=list(self.layers))

    def plan_at_stage(self, model: Any, runner: Any, required_manifest: Any, out: Path, *, source: str) -> Docs:
        self.calls.append(("plan", model, runner, required_manifest, out, source))
        return Docs(self.plan)

    def apply(self, com: Any, plan: Any) -> Record:
        self.calls.append(("install", com, plan))
        if isinstance(self.installed, Exception):
            raise self.installed
        return dict(self.installed)


@pytest.fixture
def said(caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]) -> Lines:
    """The lines `commit_attach` has logged since the last call, each after its level; it prints none."""
    caplog.set_level(logging.INFO, logger=CA.__name__)

    def lines() -> list[str]:
        assert capsys.readouterr().out == ""
        got = [f"{r.levelname} {r.getMessage()}" for r in caplog.records if r.name == CA.__name__]
        caplog.clear()
        return got
    return lines


def attach(com: Any, args: Any, required_manifest: Record | None, out: Path, *, classes: tuple[str, ...] = CLASSES, cli: bool = False,
           family: str | None = None, fa: int | None = 2, served: Any = SERVED, cb_plan: Any = None) -> SimpleNamespace:
    return CA.attach_of_record(com, ENGINE, args, required_manifest, out, kind="rows-sha512", required_classes=classes,
                               required_classes_cli=cli, hidden_family=family, fa_version=fa, cb_plan=cb_plan, served=served)


def refused(run: Callable[[], Any]) -> None:
    """``run`` ends the process with exit status 3, as the Commit refuses."""
    with pytest.raises(SystemExit) as stop:
        run()
    assert stop.value.code == 3


NO_PLAN = "INFO [delta] committer stub: no acquisition plan (the served source acquires every value of the Build)"
NO_RECORD = ("WARNING [delta] NOTE: X-03 attention plane classes: no Build record with attention_impls_observed under --program-dir/--programs-root "
             "(None, None); the uniformity rule runs (a Program with two attention classes fails X-03 closed)")


@pytest.mark.parametrize("fa, family, classes", [(2, FA2, required_classes_default(2)), (3, FA3, required_classes_default(3)),
                                                 (2, FA3, required_classes_default(3)), (None, FA2, required_classes_default(None)),
                                                 (3, None, tuple(c for c in required_classes_default(3) if c != FA3))])
def test_the_required_classes_are_the_engine_s_attention_s_with_the_manifest_s_hidden_class(
        monkeypatch: pytest.MonkeyPatch, fa: int | None, family: str | None, classes: tuple[str, ...]) -> None:
    asked: list[tuple[Any, Any]] = []

    def attention_hidden(required_manifest: Any, model: Any) -> Record:
        asked.append((required_manifest, model))
        return {"family": family}
    monkeypatch.setattr(HS, "attention_hidden", attention_hidden)
    assert CA.classes_of_record({"identities": []}, "the model", fa) == (classes, family)
    assert asked == [({"identities": []}, "the model")] and (family is None or family in classes)


@pytest.mark.parametrize("cli, classes", [(False, ("module_outputs",)), (True, CLASSES)])
def test_a_manifest_that_names_no_hidden_identity_attaches_no_hidden_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: Lines,
                                                                            cli: bool, classes: tuple[str, ...]) -> None:
    steps, com = Acquisition(monkeypatch), committer()
    got = attach(com, options(hidden_so=TAP), manifest(instance_outputs=8), tmp_path, cli=cli, family=FA2)
    assert steps.named("hidden") == [] and com.sources == []
    assert (got.required_classes, got.hidden_required, got.moe_attached) == (classes, False, False)
    assert steps.named("taps")[0][3] == classes, "the taps see the classes without the hidden one, unless they were given"
    assert said() == [
        ("INFO [delta] hidden source NOT attached: the required-value manifest names no fa2_hidden_m1_stream / fa3_hidden_m1_stream identity "
         f"(families ['instance_outputs']); required classes {list(classes)}"), NO_RECORD, NO_PLAN]


@pytest.mark.parametrize("required", [{"identities": []}, {"populations": None, "identities": []}, {"populations": {}, "identities": []},
                                      {"populations": {"by_family": None}, "identities": []}, manifest()])
def test_a_manifest_that_counts_no_family_names_no_hidden_identity_and_no_moe_plane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                                    said: Lines, required: Record) -> None:
    """A manifest may hold no `populations`, or none by family (absent, null or empty): it names no family, so neither source is
    attached."""
    steps, com = Acquisition(monkeypatch, layers=("L0",)), committer()
    got = attach(com, options(hidden_so=TAP, moe_source="auto"), required, tmp_path, family=FA2)
    assert steps.named("hidden") == [] and steps.named("moe") == [] and com.sources == []
    assert (got.required_classes, got.hidden_required, got.moe_attached) == (("module_outputs",), False, False)
    assert said() == [
        ("INFO [delta] hidden source NOT attached: the required-value manifest names no fa2_hidden_m1_stream / fa3_hidden_m1_stream identity "
         "(families []); required classes ['module_outputs']"),
        ("INFO [delta] MoE block source NOT attached: the required-value manifest names no moe_block_stream identity (families []; the "
         "Program's MoE construction has no MOE-01 rows) -- nothing is staged that the manifest does not name"), NO_RECORD, NO_PLAN]


def test_the_hidden_source_is_built_for_the_engine_s_attention_and_its_bounded_tap_is_told_where_to_write(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: Lines) -> None:
    steps, com, required = Acquisition(monkeypatch, source=BoundedSource), committer(), manifest(**{FA2: 4})
    got = attach(com, options(hidden_so=TAP, tap_cap_mb=64, out=str(tmp_path)), required, tmp_path, family=FA2)
    (src,) = com.sources
    assert steps.calls[:2] == [("setting", "VERITY_FA2_TAP_CAP_MB", "64"),
                               ("hidden", com, TAP, {"ns": "verity_fa2_matReq", "flash_attn_version": 2, "fa2_cls": None, "fa3_cls": None,
                                                     "manifest": required})]
    assert src.built["so"] == TAP and src.telemetry_path == str(tmp_path / "fa2_tap_windows.jsonl")
    assert (got.required_classes, got.hidden_required) == (CLASSES, True)
    assert said() == [
        f"INFO [delta] hidden source attached: {TAP} -> committer stub (BoundedSource, engine FA2)",
        ("INFO [delta] fa2 tap bound: cap 64 MiB (ONE bounded device buffer; launches above it are split into sequence groups); telemetry "
         f"{tmp_path / 'fa2_tap_windows.jsonl'}"), NO_RECORD, NO_PLAN]


def test_the_hidden_source_takes_the_committer_s_own_classes_and_the_namespace_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                                    said: Lines) -> None:
    steps, com = Acquisition(monkeypatch, source=UnboundedSource), committer()
    com.hidden_source_cls, com.hidden_source_cls_fa3 = "the native FA2 source", "the native FA3 source"
    attach(com, options(hidden_so=TAP, hidden_ns="ns"), None, tmp_path, cli=True, fa=3)
    assert steps.named("setting") == [], "no bound was given, so none is set"
    assert steps.named("hidden") == [("hidden", com, TAP, {"ns": "ns", "flash_attn_version": 3, "fa2_cls": "the native FA2 source",
                                                           "fa3_cls": "the native FA3 source", "manifest": None})]
    assert com.sources[0].telemetry_path == "unset", "with no out dir the telemetry path is left as built"
    assert said() == [f"INFO [delta] hidden source attached: {TAP} -> committer stub (UnboundedSource, engine FA3)",
                      "INFO [delta] fa2 tap bound: cap 0 MiB (OFF: unbounded per-geometry buffers); telemetry unset", NO_PLAN]


def test_without_a_manifest_and_without_given_classes_or_a_hidden_family_no_hidden_source_is_built(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: Lines) -> None:
    steps, com = Acquisition(monkeypatch), committer()
    got = attach(com, options(hidden_so=TAP), None, tmp_path)
    assert steps.named("hidden") == [] and com.sources == [] and (got.required_classes, got.hidden_required) == (CLASSES, None)
    assert attach(committer(), options(), manifest(**{FA2: 4}), tmp_path, family=FA2).hidden_required is True and steps.named("hidden") == []
    assert said() == [NO_PLAN, NO_RECORD, NO_PLAN]


def test_the_taps_are_attached_for_the_manifest_and_the_classes_they_return_are_the_commit_s(tmp_path: Path,
                                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    steps, com, required = Acquisition(monkeypatch, tapped=("norm_scales",)), committer(), manifest(instance_outputs=8)
    got = attach(com, options(program_dir="/programs/b1", programs_root="/programs"), required, tmp_path, cli=True, cb_plan="the plan")
    assert steps.named("taps") == [("taps", com, required, CLASSES, "the taps' options",
                                    {"explicit_classes": True, "programs": ("/programs/b1", "/programs"), "call_boundary_plan": "the plan"})]
    assert got.required_classes == CLASSES + ("norm_scales",)


@pytest.mark.parametrize("mode, line", [
    ("off", None),
    ("auto", ("INFO [delta] MoE block source NOT attached: the required-value manifest names no moe_block_stream identity (families "
              "['instance_outputs']; the Program's MoE construction has no MOE-01 rows) -- nothing is staged that the manifest does not name"))])
def test_a_manifest_that_names_no_moe_plane_attaches_no_moe_block_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: Lines,
                                                                         mode: str, line: str | None) -> None:
    steps, com = Acquisition(monkeypatch, layers=("L0",)), committer()
    got = attach(com, options(moe_source=mode), manifest(instance_outputs=8), tmp_path)
    assert steps.named("moe") == [] and com.sources == [] and got.moe_attached is False
    assert said() == [*([line] if line else []), NO_RECORD, NO_PLAN]


def test_an_moe_block_source_asked_for_is_refused_where_the_manifest_names_no_plane_or_the_model_has_no_moe_layer(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: Lines) -> None:
    steps = Acquisition(monkeypatch)
    refused(lambda: attach(committer(), options(moe_source="on"), manifest(instance_outputs=8), tmp_path))
    no_plane = ("ERROR [delta] REFUSE: --moe-source on but the required-value manifest required_manifest.json names no moe_block_stream identity "
                "(the Program has no MOE-01 rows -- a padded construction's MoE identity is the module output): the source would stage planes "
                "the arena has no slot for")
    assert steps.named("moe") == [] and said() == [no_plane]
    com = committer()
    refused(lambda: attach(com, options(moe_source="on"), manifest(moe_block_stream=6), tmp_path))
    assert steps.named("moe") == [("moe", com, com.model)] and com.sources == []
    assert said() == ["ERROR [delta] REFUSE: --moe-source on but the model has no MoE-runner modules"]
    assert attach(com, options(moe_source="auto"), manifest(moe_block_stream=6), tmp_path).moe_attached is False
    assert said() == [NO_RECORD, NO_PLAN]


@pytest.mark.parametrize("given, classes", [(None, CLASSES + ("moe_block_stream",)), (["module_outputs"], CLASSES)])
def test_a_model_s_moe_layers_get_the_block_source_and_its_class_unless_the_classes_were_given(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: Lines, given: list[str] | None, classes: tuple[str, ...]) -> None:
    steps, com = Acquisition(monkeypatch, layers=("L0", "L1")), committer()
    got = attach(com, options(moe_source="auto", required_classes=given), manifest(moe_block_stream=6), tmp_path)
    assert steps.named("moe") == [("moe", com, com.model)] and [s.layers for s in com.sources] == [["L0", "L1"]]
    assert (got.moe_attached, got.required_classes) == (True, classes)
    assert said() == [f"INFO [delta] MoE block source attached: 2 MoE layers -> committer stub; required classes {list(classes)}", NO_RECORD, NO_PLAN]
    bare = committer("model")
    assert attach(bare, options(moe_source="on"), None, tmp_path, served=SERVED).moe_attached is False
    assert len(steps.named("moe")) == 1, "a committer with no model is given no MoE block source"


def test_a_promoted_member_is_acquired_at_its_first_consumer_s_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: Lines) -> None:
    Acquisition(monkeypatch)
    com, required = committer("identity_check"), manifest([promoted(), promoted(), {"op_path": "model.layers.0.mlp", "output_member": "out"}])
    attach(com, options(), required, tmp_path)
    assert com.acquire_inputs == {"model.layers.0.mlp": 16}
    assert said() == [
        ("INFO [delta] acquisition extended by the manifest (promoted members): 1 consumer input(s) ['model.layers.0.mlp/in'] bind "
         "['model.layers.0.norm/out'] (2 promoted manifest rows; model.layers.0.norm: RmsNorm_v1 width 8 bf16 at model.layers.0.mlp)"), NO_PLAN]
    away = committer("identity_check", modules=("model.layers.1.mlp",))
    attach(away, options(), required, tmp_path)
    assert away.acquire_inputs == {"model.layers.0.mlp": 16}
    assert said()[1] == ("WARNING [delta] NOTE: 1 promoted-member consumer(s) are not modules of the built model (['model.layers.0.mlp']): their "
                         "identities stay unbound (coverage names them)")
    for untouched in (committer("identity_check", "acquire_inputs"), committer("identity_check", "model")):
        attach(untouched, options(), required, tmp_path)
        assert getattr(untouched, "acquire_inputs", {}) == {} and said() == [NO_PLAN]
    plain = committer("identity_check")
    attach(plain, options(), manifest(instance_outputs=8), tmp_path)
    assert plain.acquire_inputs == {} and said() == [NO_PLAN], "a manifest with no promoted row arms nothing and says nothing"


def test_a_promoted_member_with_no_observation_point_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: Lines) -> None:
    Acquisition(monkeypatch)
    com = committer()
    refused(lambda: attach(com, options(), manifest([promoted(consumers=())]), tmp_path))
    no_point = ("ERROR [delta] REFUSE: the required-value manifest's promoted members have no unambiguous observation point: ['promoted row "
                "model.layers.0.norm/out (spec RmsNorm_v1) names no consumer: no observation point']")
    assert com.acquire_inputs == {} and said() == [no_point]


def test_the_attention_plane_classes_come_from_the_build_s_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: Lines) -> None:
    Acquisition(monkeypatch)
    (tmp_path / "result.json").write_text(json.dumps({"attention_impls_observed": {
        "model.layers.0.self_attn.attn": {"num_heads": 4, "num_kv_heads": 2, "head_size": 8, "sliding_window": None},
        "model.layers.1.self_attn.attn": {"num_heads": 4, "num_kv_heads": 4, "head_size": 8, "sliding_window": 16}}}))
    com = committer()
    attach(com, options(program_dir=str(tmp_path)), manifest(instance_outputs=8), tmp_path)
    assert [(d.layer, d.NH, d.KVH, d.D, d.window) for d in com.attention_layer_decls] == [(0, 4, 2, 8, None), (1, 4, 4, 8, 16)]
    assert said() == [
        (f"INFO [delta] X-03 attention plane classes: 2 layer declaration(s) from {tmp_path / 'result.json'} -> classes [('global-gqa', 1), "
         "('windowed-mha', 1)]; per-layer expectation derived per step (fa2_plane_classes), uniformity not assumed"), NO_PLAN]
    for com in (committer("identity_check"), committer()):
        attach(com, options(program_dir=str(tmp_path)), None if hasattr(com, "identity_check") else manifest(instance_outputs=8), tmp_path)
        assert not hasattr(com, "attention_layer_decls") and said() == [NO_PLAN], "no manifest, or no identity check: no declaration"
    bare = committer()
    attach(bare, options(program_dir=str(tmp_path / "absent"), programs_root=str(tmp_path / "none")), manifest(instance_outputs=8), tmp_path)
    assert not hasattr(bare, "attention_layer_decls")
    assert said() == [
        ("WARNING [delta] NOTE: X-03 attention plane classes: no Build record with attention_impls_observed under --program-dir/--programs-root "
         f"({str(tmp_path / 'absent')!r}, {str(tmp_path / 'none')!r}); the uniformity rule runs (a Program with two attention classes fails "
         "X-03 closed)"), NO_PLAN]


def test_the_committer_s_hooks_come_from_the_acquisition_plan_and_its_record_is_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                                       said: Lines) -> None:
    steps, com, required = Acquisition(monkeypatch), committer("identity_check"), manifest(instance_outputs=8)
    got = attach(com, options(), required, tmp_path, served=None)
    assert steps.named("plan") == [("plan", com.model, "runner", required, tmp_path, "commit_delta rows-sha512")]
    assert steps.named("install") == [("install", com, steps.plan)]
    assert json.loads((tmp_path / "acquisition_engine.json").read_text()) == {
        "engine": "v2", "plan_digest": "d" * 16, "modules_after": ["m0", "m1"], "plan_inputs": ["m2"], "deferred_sites": ["s0"]}
    assert (got.required_classes, got.moe_attached, got.hidden_required) == (CLASSES, False, False)
    assert said() == ["INFO [delta] runtime tree 3 modules", "INFO [delta] acquisition plan abcd",
                      "INFO [delta] committer stub: 2 module output hooks from the plan, consumer inputs ['m2'], deferred sites 1"]


@pytest.mark.parametrize("without, required", [((), None), (("_modules",), manifest(instance_outputs=8)), (("model",), manifest(instance_outputs=8))])
def test_a_plan_needs_a_manifest_and_a_committer_that_hooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: Lines,
                                                            without: tuple[str, ...], required: Record | None) -> None:
    steps = Acquisition(monkeypatch)
    refused(lambda: attach(committer("identity_check", "acquire_inputs", *without), options(), required, tmp_path, served=None))
    assert steps.named("plan") == [] and not (tmp_path / "acquisition_engine.json").exists()
    assert said() == ["ERROR [delta] REFUSE: the acquisition plan needs --required-manifest and a hooking committer (got rows-sha512)"]


def test_a_plan_that_names_unsupported_values_is_refused_and_nothing_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                                said: Lines) -> None:
    steps = Acquisition(monkeypatch, installed=install.IncompletePlan("2 Values are UNSUPPORTED"))
    refused(lambda: attach(committer("identity_check"), options(), manifest(instance_outputs=8), tmp_path, served=None))
    assert len(steps.named("install")) == 1 and not (tmp_path / "acquisition_engine.json").exists()
    assert said() == ["INFO [delta] runtime tree 3 modules", "INFO [delta] acquisition plan abcd",
                      "ERROR [delta] REFUSE: acquisition plan: 2 Values are UNSUPPORTED"]


def test_a_stage_that_drew_no_plan_is_refused_before_anything_is_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: Lines) -> None:
    steps = Acquisition(monkeypatch, plan=None)
    refused(lambda: attach(committer("identity_check"), options(), manifest(instance_outputs=8), tmp_path, served=None))
    assert steps.named("install") == [] and not (tmp_path / "acquisition_engine.json").exists()
    assert said()[-1] == "ERROR [delta] REFUSE: acquisition plan: the stage drew no acquisition plan"

