"""`verity_vllm.pipeline.tp.rank_worker`: the header a rank's `tp2_install` hands its observer.  The engine's adapter makes the profile
manifest for one GPU, so a rank writes the task's width into its copy (`parallelism` and `engine_kwargs`), takes the profile id of
that copy and says so in the header's notes; a profile that is not the engine's manifest (another schema, no schema, no dict) goes on
as the adapter made it.  CPU-only: the rank is `tests.pipeline.test_tp_target`'s stand-in worker, rank 1 of 2; `make_header` returns
a header made here; the observer is a stand-in that keeps the header it is given and stops the install there."""
from __future__ import annotations

import pytest

from tests.pipeline import test_tp_target as TT
from verity_vllm.engine import engine_profile as prof
from verity_vllm.observe import capture
from verity_vllm.observe.events import RunHeader

METHOD = "sha256 of the manifest"
ADDED = ("; tp2_worker: parallelism/engine_kwargs.tensor_parallel_size=2 added to the manifest before hashing "
         "(the adapter's own id was adapter-id)")


class _Installed(Exception):
    """The stand-in observer has the header."""


def _handed_on(monkeypatch, tmp_path, profile) -> tuple[RunHeader, RunHeader]:
    """(the header the adapter made over ``profile``, the header `tp2_install` handed the observer)."""
    made = RunHeader("log/v1", "2026-01-01T00:00:00Z", "adapter-id", profile, {"repo": "org/model"}, {"sha256": "w"},
                     {"profile_id_method": METHOD})
    seen: list[RunHeader] = []

    def observer(engine, *, header, **kw):
        seen.append(header)
        raise _Installed

    monkeypatch.setattr(capture, "make_header", lambda engine, workload, **kw: made)
    monkeypatch.setattr(capture, "Capture", observer)
    _, w = TT._extension(monkeypatch, object())
    with pytest.raises(_Installed):
        w.tp2_install(out_dir=str(tmp_path), workload=TT.WORKLOAD, case="C", cp={"repo": "org/model"}, engine_kwargs={}, env_flags={},
                      snapshot_steps=[], snapshot_max_bytes=0, adapter_revision="r", wheel_sha256=None, vllm_commit=None)
    (header,) = seen
    return made, header


@pytest.mark.parametrize("blocks, widened", [
    ({"parallelism": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1}, "engine_kwargs": {"tensor_parallel_size": 1, "seed": 0}},
     {"parallelism": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1}, "engine_kwargs": {"tensor_parallel_size": 2, "seed": 0}}),
    ({}, {"parallelism": {"tensor_parallel_size": 2}, "engine_kwargs": {"tensor_parallel_size": 2}}),
], ids=["both blocks", "neither block"])
def test_a_rank_writes_the_tasks_width_into_the_engines_manifest_and_takes_the_profile_id_of_that(monkeypatch, tmp_path, blocks, widened):
    made, header = _handed_on(monkeypatch, tmp_path, {"schema": "verity-vllm/profile/v1", "gpu": {"name": "g"}, **blocks})
    manifest = {"schema": "verity-vllm/profile/v1", "gpu": {"name": "g"}, **widened}
    assert header.profile == manifest and header.profile_id == prof.profile_id_of(manifest) != "adapter-id"
    assert header.notes == {"profile_id_method": METHOD + ADDED} and made.notes == {"profile_id_method": METHOD}
    assert (header.schema, header.created_utc, header.model, header.workload) == (made.schema, made.created_utc, made.model, made.workload)


@pytest.mark.parametrize("profile", [{"schema": "verity-vllm/profile/v0", "parallelism": {"tensor_parallel_size": 1}}, {}, None],
                         ids=["another schema", "no schema", "no dict"])
def test_a_profile_that_is_not_the_engines_manifest_goes_on_as_the_adapter_made_it(monkeypatch, tmp_path, profile):
    before = None if profile is None else {k: dict(v) if isinstance(v, dict) else v for k, v in profile.items()}
    made, header = _handed_on(monkeypatch, tmp_path, profile)
    assert header is made and header.profile == before and header.profile_id == "adapter-id"
    assert header.notes == {"profile_id_method": METHOD}
