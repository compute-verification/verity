"""The collector's staging defaults (`native_collect.window_mb_default`, the levels and bounded-staging defaults) and the Commit's
(`pipeline.commit.resolve_staging`): unset, the windowed worker stages over a ring of 256 MiB windows, bounded, with no levels when
it keeps no values; every default has its off-switch, and a collector off the windowed worker is refused.  The committer runs on
`test_native_collect_plan_cache`'s CPU fakes; the roots under each setting are the byte-identity suites' (`test_windowed_rows_sha512`,
`test_native_collect_plan_cache`)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.commit import test_native_collect_plan_cache as P  # noqa: E402
from verity_vllm.commit.committer import native_collect as NC  # noqa: E402
from verity_vllm.pipeline import cli  # noqa: E402

ENV = ("VERITY_RETAIN", "VERITY_WINDOW_MB", "VERITY_WINDOW_SLOTS", "VERITY_STAGING_BOUNDED")


def committer(mp, retain: str, gather: bool = True, retain_levels: bool | None = None, **env: str) -> NC.NativeCollectCommitter:
    for k in ENV:
        mp.delenv(k, raising=False)
    mp.setenv("VERITY_RETAIN", retain)
    for k, v in env.items():
        mp.setenv(k, v)
    fake = P._Torch()
    mp.setattr(P.nh, "torch", fake)
    mp.setattr(NC, "torch", fake)
    ext = P.fake_ext()
    mp.setattr(NC, "ext", lambda *a, **k: ext)
    eng = SimpleNamespace(model=torch.nn.Module(), runner=P._Runner())
    return NC.NativeCollectCommitter(eng, run_id="defaults", program_digest=P.PD, geo_digest=P.GEO, arena_bytes=None, wrap=True,
                                     gather=gather, native_hooks=gather, auto_blocks=gather, native_worker=True, fast_copy=False,
                                     retain_levels=retain_levels)


def staging(com: NC.NativeCollectCommitter) -> tuple[int, int, bool, bool]:
    return com.win_bytes >> 20, com.win_slots_req, com.bounded, com.retain_levels


@pytest.mark.parametrize("retain,env,want", [
    ("none", {}, (256, 8, True, False)),
    ("host", {}, (256, 8, True, True)),
    ("none", {"VERITY_STAGING_BOUNDED": "0"}, (256, 8, False, False)),
    ("host", {"VERITY_WINDOW_MB": "16", "VERITY_WINDOW_SLOTS": "3"}, (16, 3, True, True)),
], ids=["none", "host", "bounded-off", "explicit"])
def test_the_windowed_worker_s_defaults(monkeypatch, retain, env, want):
    com = committer(monkeypatch, retain, **env)
    assert staging(com) == want
    assert com.windowed is True and com.commit_scheme == "rows-sha512"


def test_a_collector_off_the_windowed_worker_is_refused(monkeypatch):
    """The collector commits rows-sha512 on the windowed worker only: one that would stage per-step blocks (``--retain device``
    with no window, no window, or no gather) is refused by name, and so is a window without gather; under every retain mode a
    windowed collector's scheme is rows-sha512."""
    for retain, gather, env in (("device", True, {}), ("none", True, {"VERITY_WINDOW_MB": "0"}), ("none", False, {})):
        with pytest.raises(ValueError, match="rows-sha512 is implemented on the windowed worker"):
            committer(monkeypatch, retain, gather=gather, **env)
    with pytest.raises(ValueError, match="windowed staging needs native_collect_v2b"):
        committer(monkeypatch, "none", gather=False, VERITY_WINDOW_MB="256")
    assert committer(monkeypatch, "device", VERITY_WINDOW_MB="16").commit_scheme == "rows-sha512"


def test_an_explicit_retain_levels_wins(monkeypatch):
    assert committer(monkeypatch, "none", retain_levels=True).retain_levels is True
    assert committer(monkeypatch, "host", retain_levels=False).retain_levels is False
    with pytest.raises(ValueError, match="retain_levels=False is implemented on the windowed path only"):
        committer(monkeypatch, "device", retain_levels=False)


@pytest.mark.parametrize("argv,want", [
    ([], ("none", 256, 8, True, True)),
    (["--committer", "hooks_only"], ("none", 0, None, False, True)),
    (["--committer", "native_collect_v2b", "--retain", "host"], ("host", 256, 8, True, True)),
    (["--committer", "native_collect_v2b", "--retain", "device", "--no-open-after-release"], ("device", 0, None, False, False)),
    (["--committer", "native_collect_v2b", "--window-mb", "0"], ("none", 0, None, False, True)),
    (["--committer", "native_collect_v2b", "--no-bounded-staging"], ("none", 256, 8, False, True)),
    (["--committer", "native_collect_v2b", "--window-mb", "16", "--window-slots", "3", "--bounded-staging"], ("none", 16, 3, True, True)),
    (["--arms", "native_collect_v2b,hooks_only"], ("none", 0, None, False, True)),
], ids=["v2b", "hooks-only", "v2b-host", "v2b-device", "window-off", "bounded-off", "explicit", "mixed-arms"])
def test_the_commit_resolves_the_collector_s_defaults(argv, want):
    from verity_vllm.pipeline import commit as cd

    a = cd.resolve_staging(cli.parse("commit", ["--out", "o", *argv], {}))
    assert (a.retain, a.window_mb, a.window_slots, a.bounded_staging, a.open_after_release) == want


def test_tp_commit_reads_the_ranks_ring_as_their_committers_do():
    from verity_vllm.pipeline.tp import commit as tc

    assert tc.ring_slots("native_collect_v2b", None, None, "none") == 8
    assert tc.ring_slots("native_collect_v2b", None, None, None) is None          # unset VERITY_RETAIN: device, per-step blocks
    assert tc.ring_slots("native_collect_v2b", "256", "3", "host") == 3
    assert tc.ring_slots("native_collect_v2b", "0", "8", "host") is None
    assert tc.ring_slots("native_host", None, None, "none") is None
