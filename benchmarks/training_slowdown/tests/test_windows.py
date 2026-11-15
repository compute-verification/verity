"""Fresh windows: a config asks for the windows its start's own training did not read; the harness serves the job
only those, refuses the config on a start that doesn't say what it read, and records which windows it served. A
config that doesn't ask is the job it was."""
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
from synthetic import PARTS, SHAPE, run_job, tiny_registry, write_config

from training_slowdown import config, run, starts
from training_slowdown.data import Data, Stream, first_windows, read_windows

ATTACKS = Path(__file__).resolve().parents[1] / "attacks"
CTX = SHAPE["ctx"]
BUDGET = {"steps": 4, "batch": 2}
#: the part of the tiny stream the start names as read, the middle one, and the one window it holds
READ, READ_IDS = (1,), [2]
#: the tiny stream's windows: two of its first part, one of its second, three of its third
FIRST = [0, 2, 3, 6]


def registry_of(installed: Path) -> dict:
    """The registry `tiny_registry` wrote beside the directory its start is installed under."""
    return starts.registry(installed.parent / "starts.toml")


def asking(root: Path, read) -> tuple[config.Config, Data, Path]:
    """A config of the tiny start that asks for fresh windows, where the start names `read` as read
    (`tiny_registry`): the config, the data and the directory the start is installed under."""
    installed = tiny_registry(root / "reg", read).parent / "starts"
    cfg = config.load(write_config(root / "fresh.toml", budget=BUDGET, windows="fresh"), registry_of(installed))
    return cfg, Data.of(cfg.start.data, registry_of(installed)), installed


def tokens_of(data: Data, ids: list[int]) -> list[list[int]]:
    """The tokens of the windows `ids`, read from the parts on disk: window i of a part is its ctx + 1 tokens from
    i * ctx."""
    rows = []
    for i in ids:
        p = max(q for q in range(len(PARTS)) if FIRST[q] <= i)
        part = np.fromfile(data.train / "parts" / f"{p:03d}.bin", dtype=np.uint16)
        rows.append(part[(i - FIRST[p]) * CTX: (i - FIRST[p]) * CTX + CTX + 1].tolist())
    return rows


def test_a_window_has_one_id_in_the_stream_and_a_part_the_start_names_gives_every_id_it_holds(tmp_path):
    cfg, data, _ = asking(tmp_path, READ)
    parts = data.manifest()["parts"]
    assert first_windows(parts, CTX).tolist() == FIRST == Stream(data, 0, ctx=CTX).starts.tolist()
    assert first_windows(parts, 16).tolist() == [0, *np.cumsum([(n - 1) // 16 for n in PARTS]).tolist()]
    assert read_windows(parts, CTX, [parts[2], parts[0]]) == [
        {"part": 0, "sha256": parts[0]["sha256"], "windows": [0, 2]},
        {"part": 2, "sha256": parts[2]["sha256"], "windows": [3, 6]}]
    # a start names a part by its sha256 alone, and the manifest's every part of that sha256 counts as read
    twice = [*parts, parts[1] | {"index": 3}]
    assert [e["windows"] for e in read_windows(twice, CTX, [{"sha256": parts[1]["sha256"]}])] == [[2, 3], [6, 7]]
    assert all(len(p["sha256"]) == 64 for p in parts) and len({p["sha256"] for p in parts}) == len(PARTS)


def test_a_config_that_asks_is_served_its_seeds_order_without_the_windows_its_start_read(tmp_path):
    """Every window of a part the start names is left out, in every seed's order, round after round; what is served
    is the stream's own windows at the ids the stream records."""
    cfg, data, root = asking(tmp_path, READ)
    assert cfg.windows == "fresh" and cfg.run()["windows"] == "fresh"
    for seed in (0, 1, 7):
        every = np.random.default_rng(seed).permutation(FIRST[-1]).tolist()
        stream = run.stream_of(cfg, data, seed, root)
        want = [i for i in every if i not in READ_IDS]
        assert stream.order.tolist() == want and len(want) == 5 and stream.rounds == 0
        rows = [stream.next(2) for _ in range(5)]
        # five fresh windows, two a step: the fifth is never served, and the order starts again at every third step
        ids = [sorted(want[at: at + 2]) for at in (0, 2, 0, 2, 0)]
        assert stream.served == ids and stream.rounds == 2 and stream.at == 2
        assert not {i for step in stream.served for i in step} & set(READ_IDS)
        assert [r.tolist() for r in rows] == [tokens_of(data, step) for step in ids]
        assert stream.record() == {"ctx": CTX, "of": 6, "fresh": 5, "served": ids, "rounds": 2, "excluded": [
            {"part": 1, "sha256": data.manifest()["parts"][1]["sha256"], "windows": [2, 3]}]}


NO_PARTS = "names no part of the stream its own training read .provenance.data.parts."


@pytest.mark.parametrize("read, why", [
    (None, NO_PARTS),
    ((), NO_PARTS),
    ([{"index": 0, "tokens": PARTS[0]}], "a part the start's start.json names .* has no sha256"),
    ((0, {"sha256": "0" * 64}),
     f"names 1 of its 2 parts .* by a sha256 the stream's manifest lists no part under, the first {'0' * 64}"),
], ids=["no provenance.data", "no part", "a part without sha256", "a part the stream lacks"])
def test_a_config_that_asks_is_refused_on_a_start_that_does_not_say_what_it_read(tmp_path, read, why):
    """The refusal names the config, the start and what to do: say in start.json which parts the start read, by the
    sha256 the stream's manifest lists each under, and register the file; or drop `windows`. The same config
    without `windows` is served every window on the same start, which nothing then reads the parts of."""
    cfg, data, root = asking(tmp_path, read)
    fix = (f"A job is served fresh windows only when its start says what it read: write the parts its training read "
           f"into {root / 'tiny' / 'start.json'}'s provenance.data.parts, each with the `sha256` that "
           f"{data.train / 'manifest.json'} lists the part under, and register the new file .start_json_sha256 of "
           f".starts.tiny. in starts.toml.; or drop `windows` from the config, which then serves every window")
    with pytest.raises(SystemExit, match=f"config {cfg.path}: windows = 'fresh' at start tiny: .*{why}.*[.] {fix}"):
        run.stream_of(cfg, data, 0, root)
    every = config.load(write_config(tmp_path / "every.toml", budget=BUDGET), registry_of(root))
    assert run.stream_of(every, data, 0, root).order.tolist() == np.random.default_rng(0).permutation(6).tolist()


@pytest.mark.parametrize("read, left, fix", [((0, 2), 1, "lower budget.batch to 1 or less, or drop `windows`"),
                                             ((0, 1, 2), 0, "drop `windows`")])
def test_a_config_that_asks_is_refused_where_a_step_takes_more_windows_than_are_fresh(tmp_path, read, left, fix):
    cfg, data, root = asking(tmp_path, read)
    with pytest.raises(SystemExit, match=f"windows = 'fresh' at start tiny: the start read all but {left} of the "
                                         f"stream's 6 windows, and a step takes 2: {fix} from the config"):
        run.stream_of(cfg, data, 0, root)


def test_a_config_asks_for_fresh_windows_or_for_nothing(tmp_path):
    reg = starts.registry(tiny_registry(tmp_path / "reg"))
    for windows in ("seen", "", "random"):
        with pytest.raises(SystemExit, match=f"windows is {windows!r}, not 'fresh' .* drop the key for every window"):
            config.load(write_config(tmp_path / "a.toml", windows=windows), reg)
    with pytest.raises(SystemExit, match="windows is bool, not str"):
        config.load(write_config(tmp_path / "a.toml", windows=True), reg)
    every, fresh = (config.load(write_config(tmp_path / "a.toml", windows=w), reg) for w in (None, "fresh"))
    assert every.windows == "" and "windows" not in every.run() and fresh.run() == every.run() | {"windows": "fresh"}
    assert every.digest != fresh.digest and every.jobs()[0].label != fresh.jobs()[0].label


@pytest.mark.parametrize("path", sorted(ATTACKS.glob("*.toml")), ids=lambda p: p.stem)
def test_a_config_that_does_not_ask_is_the_job_it_was(path):
    """Each config under attacks/: what decides its run is the seven entries it was before a config could ask for
    windows, its dials those its digest holds (`Config.undigested` names the rest), and its digest their sha256, so
    the label its records are found under doesn't move. One that asks states so besides them."""
    cfg = config.load(path, starts.registry())
    s = cfg.start
    was = {"start": s.name, "theta": s.theta_file_sha256, "data": s.data, "route": cfg.route,
           "route_version": cfg.route_version,
           "dials": {k: v for k, v in cfg.dials.items() if k not in cfg.undigested}, "budget": cfg.budget}
    asked = {"windows": cfg.windows} if "windows" in tomllib.loads(path.read_text()) else {}
    canonical = json.dumps(was | asked, sort_keys=True, separators=(",", ":")).encode()
    assert cfg.run() == was | asked and cfg.record()["run"] == cfg.run()
    assert cfg.digest == hashlib.sha256(canonical).hexdigest() == cfg.record()["digest"]


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """One job of the tiny config asking for fresh windows, as a node runs it."""
    return run_job(tmp_path_factory.mktemp("fresh"), read=READ, budget=BUDGET, windows="fresh")


def test_a_jobs_record_says_which_windows_it_was_served(ran):
    """The record of a job served fresh windows lists the parts left out, by sha256, the ids of the windows each
    step was served and how many times their order started again. Whether the job read a window its start read is
    then a comparison of ids: the start.json on disk names its parts, the stream's manifest gives each part's
    ids, and none of them is among the ids served."""
    rec, cfg, reg, _ = ran
    registry = starts.registry(reg)
    data = Data.of("tiny", registry)
    served = rec["windows"]["served"]
    assert rec["invalid"] == [] and rec["config"]["run"]["windows"] == "fresh" == cfg.windows
    assert len(served) == len(rec["steps"]) == 4 and all(len(step) == 2 and step == sorted(step) for step in served)

    named = json.loads((reg.parent / "starts" / "tiny" / "start.json").read_text())["provenance"]["data"]["parts"]
    parts = data.manifest()["parts"]
    first = np.cumsum([0] + [(p["tokens"] - 1) // rec["windows"]["ctx"] for p in parts]).tolist()
    read = [(i, p["sha256"], list(range(first[i], first[i + 1]))) for i, p in enumerate(parts)
            if p["sha256"] in {q["sha256"] for q in named}]
    assert [(i, ids) for i, _, ids in read] == [(1, READ_IDS)]
    assert not {i for step in served for i in step} & {i for _, _, ids in read for i in ids}
    assert rec["windows"]["excluded"] == [{"part": i, "sha256": sha, "windows": [ids[0], ids[-1] + 1]}
                                          for i, sha, ids in read]

    want = [i for i in np.random.default_rng(rec["seed"]).permutation(first[-1]).tolist() if i not in READ_IDS]
    assert served == [sorted(want[at: at + 2]) for at in (0, 2, 0, 2)]
    assert {k: v for k, v in rec["windows"].items() if k not in ("served", "excluded")} == {
        "ctx": CTX, "of": 6, "fresh": 5, "rounds": 1}


def test_a_job_refuses_a_config_that_asks_on_a_start_that_does_not_say_what_it_read(tmp_path, capfd):
    """The job as a node runs it: the refusal ends it before the route trains a step, with no record and the message
    on its error stream."""
    reg = tiny_registry(tmp_path / "reg")
    path = write_config(tmp_path / "fresh.toml", budget=BUDGET, windows="fresh")
    out = tmp_path / "out"
    assert run.main(["--config", path, "--seed", 0, "--split", "search", "--out", out, "--starts", reg,
                     "--device", "cpu"]) == 1
    assert not (out / "run.json").exists()
    said = capfd.readouterr().err
    assert (f"config {path}: windows = 'fresh' at start tiny: the start's start.json names no part of the stream its "
            f"own training read (provenance.data.parts). A job is served fresh windows only when") in said
