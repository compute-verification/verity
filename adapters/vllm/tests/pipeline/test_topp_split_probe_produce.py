"""`topp_split_probe.produce`, whole, on a CPU.

On a pod `produce` launches vLLM's three split top-p kernels and its own instrumented copy of the mask kernel on a GPU, and
files what they return.  Here each kernel is a stand-in on `tests.fake_vllm`'s modules that writes the model's answer
(`verity.definitions.registry.topp_split`) where the kernel writes its own, the temperature kernel is the model's division,
and the device is the CPU.  So no case below says a kernel is right, or that the model equals one: the fixture of record
shows that (`tests/program/test_topp_split_geometry_of_record.py`).  These show what `produce` does around the kernels: which
it launches on what, under which names and encodings it files their outputs, what its manifest says of each case and of the
device, and that `check` reads the fixture it wrote.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import itertools
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

import numpy as np
import torch
from verity.definitions.registry import topp_geometry
from verity.definitions.registry import topp_split as TS

from tests import fake_vllm
from verity_vllm.pipeline import topp_split_probe as tp

f32 = np.float32
NUM_SM, SEED = 8, 5
GPU, VLLM, TRITON = "a GPU, device 0", "0.0-stand-in", "0.0-stand-in-triton"
SITE = "vllm.v1.sample.ops.topk_topp_triton"
TOP_P, TEMPERATURE = 0.9, 0.7
#: The case the kernels leave untouched (p = 1) and the record's.
UNTOUCHED, RECORD = 3, 4
#: The drain's vocabulary: one whose slices are a lane longer under 32 splits than under 33, so a row made for the wrong count differs.
DRAIN_V = 2048
#: What the stand-in clock adds a reading; the first is `produce`'s start.
TICK = 1.26


def _cases(with_record: bool) -> list[dict]:
    cases: list[dict] = [
        {"id": 0, "kind": "random", "S": 2, "V": 1024, "seed": 2, "p": 0.5},        # no ties, and its two Z values differ
        {"id": 1, "kind": "ties", "S": 4, "V": 1024, "seed": 13, "k": 12, "keep_frac": 0.75, "group_bias": 1, "plateau": 6.5},
        {"id": 2, "kind": "ties", "S": 1, "V": 1024, "seed": 9, "k": 3, "keep_frac": 0.5, "group_bias": 0, "plateau": 3.5},
        {"id": UNTOUCHED, "kind": "random", "S": 2, "V": 1024, "seed": 3, "p": 1.0},
    ]
    if with_record:
        cases.append({"id": RECORD, "kind": "record", "S": 2, "V": 1024, "row": 0})
    cases.append({"id": len(cases), "kind": "drain", "V": DRAIN_V, "seed": 100, "k": 24, "keep_frac": 0.5, "p": None})
    return cases


def _record(d: Path) -> str:
    """A record of one row: its bf16 logits as the upper halves of their fp32 words, and the device's facts beside it."""
    path = str(d / "rows.npz")
    np.savez(path, r0_bf16_logits_bits=(tp.gen_random(11, 1024).view(np.uint32) >> 16).astype(np.uint16))
    Path(path + ".json").write_text(json.dumps({"r0": {"device": {"top_p": TOP_P, "temperature": TEMPERATURE}}}))
    return path


class _Kernel:
    """A Triton kernel's stand-in: indexed by its grid, then called with the kernel's arguments."""

    def __init__(self, name: str, body: Any, log: list[tuple[Any, ...]]) -> None:
        self.name, self.body, self.log = name, body, log

    def __getitem__(self, grid: tuple[int, ...]) -> Any:
        def launch(*args: Any, **constants: Any) -> None:
            self.log.append((self.name, grid, args, constants))
            self.body(*args, **constants)
        return launch


class _Stdout(io.StringIO):
    """A stdout that counts its flushes."""

    flushes = 0

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


class _Pod:
    """A pod for one `produce`: vLLM's kernels, the probe's own and the temperature kernel, each the model's answer written
    where the kernel writes; a device of ``NUM_SM`` SMs that is the CPU; a clock that adds ``TICK`` a reading; and the
    manifest's cases, small ones."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.log: list[tuple[Any, ...]] = []
        self.asked: list[tuple[int, bool]] = []
        self.answer: tuple[np.ndarray, np.ndarray, np.ndarray, dict] | None = None
        fake = fake_vllm.install(monkeypatch)
        fake["vllm"].__version__ = VLLM
        site = fake[SITE]
        site._SPLIT_FANOUT, site._SPLIT_ROUNDS, site._SPLIT_MAX_SPLITS = TS.SPLIT_FANOUT, TS.SPLIT_ROUNDS, topp_geometry.SPLIT_MAX_SPLITS
        site._topp_sb_stats_kernel = _Kernel("stats", self._stats, self.log)
        site._topp_sb_step_kernel = _Kernel("step", lambda *args, **constants: None, self.log)
        site._topp_sb_mask_kernel = _Kernel("mask", self._mask, self.log)
        site._apply_topp_split = self._split
        fake.set("vllm.utils.platform_utils:num_compute_units", self._num_sm)
        fake.set("vllm.v1.worker.gpu.sample.gumbel:apply_temperature", self._temper)
        triton = fake_vllm.Module("triton")
        triton.__version__ = TRITON
        monkeypatch.setitem(sys.modules, "triton", triton)
        monkeypatch.setattr(tp, "_mask_probe_kernel", lambda: _Kernel("probe", self._probe, self.log))
        monkeypatch.setattr(tp, "build_manifest", self._manifest)
        clock = itertools.count(100.0, TICK)
        monkeypatch.setattr(tp, "time", types.SimpleNamespace(time=lambda: next(clock)))
        cpu = torch.device("cpu")

        def device(name: str) -> torch.device:
            self.log.append(("device", name))
            return cpu

        monkeypatch.setattr(torch, "device", device)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: f"a GPU, device {device}")
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: self.log.append(("sync",)))

    def _manifest(self, seed: int, with_record: bool = False) -> list[dict]:
        self.asked.append((seed, with_record))
        return _cases(with_record)

    def _num_sm(self, index: int | None) -> int:
        self.log.append(("num_sm", index))
        return NUM_SM

    def _temper(self, logits: torch.Tensor, index: torch.Tensor, temperature: torch.Tensor) -> None:
        self.log.append(("temper", index.tolist(), temperature.tolist()))
        logits.copy_(torch.from_numpy(TS.tl_div(logits.numpy()[0].copy(), f32(temperature[0].item()), TS.Cfg())).reshape(1, -1))

    def _stats(self, logits: torch.Tensor, stride: int, stats: torch.Tensor, out: torch.Tensor, p: torch.Tensor, *, VOCAB_SIZE: int,
               S: int, **constants: Any) -> None:
        self.answer = TS.topp_split_row(logits.numpy()[0].copy(), float(p[0]), S, cfg=TS.Cfg.for_vocab(VOCAB_SIZE, S))
        stats.copy_(torch.from_numpy(self.answer[1]))

    def _probe(self, logits: torch.Tensor, stride: int, stats: torch.Tensor, parts: torch.Tensor, p: torch.Tensor, z_out: torch.Tensor,
               keep_out: torch.Tensor, sc_out: torch.Tensor, *, VOCAB_SIZE: int, S: int, **constants: Any) -> None:
        assert self.answer is not None
        _out, model_stats, _parts, tr = self.answer
        if tr["early"] is not None:
            return                                                   # the kernel returns before its stores
        cfg = TS.Cfg.for_vocab(VOCAB_SIZE, S)
        z = np.asarray(TS.row_stats(model_stats, S, cfg)[1], f32)
        lane = np.array([tp.thread_lane(v, VOCAB_SIZE, S, cfg.spt_x) for v in range(VOCAB_SIZE)])
        z_out.copy_(torch.from_numpy(z[topp_geometry.z_group(lane, S)]))
        keep_out.copy_(torch.from_numpy(tr["keep"].astype(np.int32)))
        sc: np.ndarray = np.zeros((S, 8), f32)
        for column, name in enumerate(("M", "Z", "pivot_logit", "dup", "numkeep", "numdup")):
            sc[:, column] = tr[name]
        sc[:, 7] = tr["fin"]
        sc_out.copy_(torch.from_numpy(sc.reshape(-1)))

    def _mask(self, logits: torch.Tensor, *args: Any, **constants: Any) -> None:
        assert self.answer is not None
        logits.copy_(torch.from_numpy(self.answer[0]).reshape(1, -1))

    def _split(self, logits: torch.Tensor, k: Any, p: torch.Tensor, mask_value: float, num_sm: int) -> None:
        batch = int(logits.shape[0])
        self.log.append(("split", batch, k, mask_value, num_sm))
        s = topp_geometry.splits_for(batch, num_sm)
        for b in range(batch):
            keep = TS.topp_keep_row(logits[b].numpy().copy(), float(p[b]), s)
            logits[b][torch.from_numpy(~keep)] = mask_value

    def launches(self, name: str) -> list[tuple[Any, ...]]:
        return [entry for entry in self.log if entry[0] == name]


@pytest.fixture(scope="module")
def produced(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """One `produce` with a record, into a directory that is there, on the stand-in pod: the pod, the directory it wrote, the
    record's path and its stdout."""
    d = tmp_path_factory.mktemp("produced")
    record = _record(d)
    (d / "out").mkdir()
    stdout = _Stdout()
    with pytest.MonkeyPatch.context() as monkeypatch, contextlib.redirect_stdout(stdout):
        pod = _Pod(monkeypatch)
        tp.produce(str(d / "out"), record, SEED)
    return types.SimpleNamespace(pod=pod, out=d / "out", record=record, lines=stdout.getvalue().splitlines(), flushes=stdout.flushes,
                                 manifest=json.loads((d / "out" / "manifest.json").read_text()))


def _model(case: dict, record: str | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """The model's masked row, stats and trace for a case of the manifest, from the case alone."""
    loaded = (np.load(record), json.loads(Path(record + ".json").read_text())) if record else None
    x, p = tp.case_row(case, loaded)
    if case["kind"] == "record":
        x = TS.tl_div(x, f32(TEMPERATURE), TS.Cfg())
    out, stats, _parts, tr = TS.topp_split_row(x, p, case["S"])
    return out, stats, tr


def test_check_passes_the_fixture_produce_wrote(produced):
    rep = tp.check(str(produced.out), produced.record, verbose=False)
    assert rep["ok"] and rep["fail"] == []
    assert (rep["n"], rep["mask_ok"], rep["stats_ok"]) == (5, 5, 5)
    assert rep["zgroup_checked"] == rep["zgroup_ok"] >= 1
    assert [(b["B"], b["rows_ok"]) for b in rep["drain"]] == [(b, b) for b in tp.drain_batches(NUM_SM)]


def test_the_manifest_says_the_device_the_versions_the_seed_and_the_record(produced):
    man = produced.manifest
    assert {k: man[k] for k in ("seed", "num_sm", "gpu", "torch", "triton", "vllm", "record")} == {
        "seed": SEED, "num_sm": NUM_SM, "gpu": GPU, "torch": torch.__version__, "triton": TRITON, "vllm": VLLM, "record": "rows.npz"}
    assert sorted(man) == ["cases", "gpu", "num_sm", "record", "secs", "seed", "torch", "triton", "vllm"]
    assert man["secs"] == round(TICK, 1) != TICK
    assert (produced.out / "manifest.json").read_text().startswith('{\n "cases": [\n  {\n   "id": 0,')
    assert produced.pod.asked == [(SEED, True)]
    assert produced.pod.launches("device") == [("device", "cuda:0")]


def test_the_manifest_lists_the_cases_in_order_with_the_drain_last(produced):
    cases = produced.manifest["cases"]
    assert [(c["id"], c["kind"]) for c in cases] == [(0, "random"), (1, "ties"), (2, "ties"), (3, "random"), (4, "record"), (5, "drain")]


@pytest.mark.parametrize("cid", [0, 1, 2, RECORD])
def test_a_case_s_facts_are_the_kernels_answers(produced, cid):
    case = produced.manifest["cases"][cid]
    out, _stats, tr = _model(_cases(True)[cid], produced.record)
    kept = int((out.view(np.uint32) != tp.NEG_INF_BITS).sum())
    assert 0 < kept < case["V"]
    assert (case["kept"], case["numkeep"], case["numdup"]) == (kept, tr["numkeep"], tr["numdup"])
    assert case["ties"] is bool(tr["numkeep"] < tr["numdup"]) and case["probe_keep_eq_real"] is True
    assert case["p"] == pytest.approx(TOP_P if cid == RECORD else tp.case_row(_cases(True)[cid])[1], abs=1e-7)


def test_two_cases_have_ties_and_one_has_none(produced):
    assert [(c["ties"], c["z_nunique"]) for c in produced.manifest["cases"][:3]] == [(False, 2), (True, 1), (True, 1)]


def test_a_row_the_kernels_leave_untouched_is_kept_whole_and_has_no_probe_facts(produced):
    case = produced.manifest["cases"][UNTOUCHED]
    assert (case["p"], case["kept"], case["z_nunique"]) == (1.0, case["V"], 1)
    assert (case["probe_keep_eq_real"], case["ties"], case["numkeep"], case["numdup"]) == (None, None, None, None)
    with np.load(produced.out / "fixture.npz") as z:
        assert np.isnan(z[f"c{UNTOUCHED}_sc"]).all() and z[f"c{UNTOUCHED}_zvals"].tolist() == [0]


def test_the_record_s_row_is_tempered_by_the_temperature_kernel_and_its_words_digested(produced):
    case = produced.manifest["cases"][RECORD]
    assert produced.pod.launches("temper") == [("temper", [0], [pytest.approx(TEMPERATURE)])]
    bf, _p = tp.case_row({"kind": "record", "row": 0}, (np.load(produced.record), json.loads(Path(produced.record + ".json").read_text())))
    tempered = TS.tl_div(bf, f32(TEMPERATURE), TS.Cfg())
    assert case["temperature"] == TEMPERATURE
    assert case["x_in_sha"] == hashlib.sha256(tempered.view(np.uint32).tobytes()).hexdigest()[:16]


def test_the_fixture_files_each_case_s_arrays_under_its_id(produced):
    with np.load(produced.out / "fixture.npz") as z:
        names = set(z.files)
        for cid in (0, 1, 2, RECORD):
            out, stats, tr = _model(_cases(True)[cid], produced.record)
            assert np.array_equal(np.unpackbits(z[f"c{cid}_keep"], bitorder="little")[:1024].astype(bool), tr["keep"])
            assert np.array_equal(z[f"c{cid}_stats"].view(np.uint32), stats.view(np.uint32))
            assert z[f"c{cid}_sc"].shape == (_cases(True)[cid]["S"], 8)
            assert (f"c{cid}_zbit" in names) == (len(z[f"c{cid}_zvals"]) == 2)
        assert len(z["c0_zvals"]) == 2 and z["c0_zvals"].dtype == np.uint32
    cases = {f"c{cid}_{part}" for cid in range(5) for part in ("keep", "stats", "sc", "zvals")}
    assert names - {"c0_zbit", "c1_zbit", "c2_zbit", "c4_zbit"} == cases | {f"drain_B{b}_keep" for b in tp.drain_batches(NUM_SM)}


def test_a_case_launches_the_stats_five_steps_the_probe_on_a_copy_and_the_mask_on_the_row(produced):
    log = [entry for entry in produced.pod.log if entry[0] in ("stats", "step", "probe", "mask", "sync", "split")]
    rounds = TS.SPLIT_ROUNDS
    order = ["stats", *["step"] * rounds, "probe", "mask", "sync"]
    assert [entry[0] for entry in log[:5 * len(order)]] == order * 5 and log[5 * len(order)][0] == "split"
    stats, steps, probe, mask = log[0], log[1:1 + rounds], log[1 + rounds], log[2 + rounds]
    assert [entry[1] for entry in (stats, *steps, probe, mask)] == [(2,)] * (rounds + 3)
    assert [step[2][6] for step in steps] == list(range(rounds))
    row, probe_in = stats[2][0], probe[2][0]
    assert mask[2][0] is row and all(step[2][0] is row for step in steps)
    assert probe_in is not row and probe_in.shape == row.shape == (1, 1024)
    assert [entry[2][1] for entry in (stats, *steps, probe, mask)] == [1024] * (rounds + 3)       # a row's stride, not a lane's
    parts = steps[0][2][3]
    assert parts.shape == (1, rounds, 2, TS.SPLIT_FANOUT, 3) and all(entry[2][3] is parts for entry in (*steps, probe, mask))
    shared = {"S": 2, "F": TS.SPLIT_FANOUT, "NUM_ROUNDS": rounds, "VOCAB_SIZE": 1024, "num_warps": 8}
    assert stats[3] == {"HAS_K": False, "VOCAB_SIZE": 1024, "S": 2, "BLOCK": 8192, "num_warps": 8}
    assert [step[3] for step in steps] == [{**shared, "HAS_K": False, "BLOCK": 2048}] * rounds
    assert probe[3] == {**shared, "BLOCK": 8192}
    assert mask[3] == {**shared, "HAS_K": False, "MASK_VALUE": float("-inf"), "BLOCK": 8192}


def test_the_drain_launches_vllm_s_entry_once_a_batch_size_on_the_first_rows(produced):
    drain = produced.manifest["cases"][-1]
    sizes = list(tp.drain_batches(NUM_SM))
    assert produced.pod.launches("split") == [("split", b, None, float("-inf"), NUM_SM) for b in sizes]
    assert produced.pod.launches("num_sm") == [("num_sm", None)]
    assert drain["num_sm"] == NUM_SM and len(drain["rows"]) == max(tp.DRAIN_BATCHES)
    assert drain["rows"][:2] == [{"seed": 100, "S_gen": 16, "p": drain["rows"][0]["p"]}, {"seed": 101, "S_gen": 32, "p": drain["rows"][1]["p"]}]
    assert [(b["B"], b["S"]) for b in drain["batches"]] == [(b, topp_geometry.splits_for(b, NUM_SM)) for b in sizes]
    with np.load(produced.out / "fixture.npz") as z:
        for b in drain["batches"]:
            keep = np.unpackbits(z[f"drain_B{b['B']}_keep"], axis=1, bitorder="little")[:, :DRAIN_V]
            assert keep.shape == (b["B"], DRAIN_V) and b["kept"] == keep.sum(axis=1).tolist()


def test_produce_prints_a_line_a_case_a_line_a_drain_batch_and_the_totals(produced):
    lines = produced.lines
    sizes = list(tp.drain_batches(NUM_SM))
    assert len(lines) == 5 + len(sizes) + 1
    assert [json.loads(line)["id"] for line in lines[:5]] == [0, 1, 2, 3, 4]
    assert sorted(json.loads(lines[0])) == sorted(
        ["id", "kind", "S", "V", "p", "kept", "z_nunique", "ties", "numkeep", "numdup", "probe_keep_eq_real"])
    assert [(said["drain"], said["S"]) for said in map(json.loads, lines[5:-1])] == [(b, topp_geometry.splits_for(b, NUM_SM)) for b in sizes]
    assert all(sorted(json.loads(line)) == ["S", "drain", "kept"] for line in lines[5:-1])
    assert all(len(json.loads(line)["kept"]) == min(8, size) for line, size in zip(lines[5:-1], sizes, strict=True))
    npz = produced.out / "fixture.npz"
    assert lines[-1] == f"cases 6 -> {npz} {npz.stat().st_size / 1024:.0f} KB {2 * TICK:.1f}s"
    assert produced.flushes == len(lines) - 1                        # each line but the totals is flushed as it is printed


def test_without_a_record_no_record_case_is_made_and_nothing_is_tempered(monkeypatch, tmp_path, capsys):
    pod = _Pod(monkeypatch)
    tp.produce(str(tmp_path / "deep" / "out"), None, SEED)
    man = json.loads((tmp_path / "deep" / "out" / "manifest.json").read_text())
    assert pod.asked == [(SEED, False)] and man["record"] is None and pod.launches("temper") == []
    assert [c["kind"] for c in man["cases"]] == ["random", "ties", "ties", "random", "drain"]
    assert tp.check(str(tmp_path / "deep" / "out"), verbose=False)["ok"]
    assert len(capsys.readouterr().out.splitlines()) == 4 + len(tp.drain_batches(NUM_SM)) + 1
