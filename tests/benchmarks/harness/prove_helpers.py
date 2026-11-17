"""What `verity bench prove`'s tests share: the small tables' census and rows, the store with a directory remote and
the records it holds, the Lean verifier's record, and the fixture that keeps the command out of the suite's own
research run. `test_prove.py`'s docstring describes the fixtures."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from verity_bench import exfiltration as E
from verity_bench import lean_verify as L
from verity_bench import prove as P
from verity_bench import prove_flock as PF
from verity_bench import prove_records as R

REPO = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).parent / "fixtures" / "prove"
W = R.WORKLOADS["glm53"]


@pytest.fixture(autouse=True)
def _outside_a_research_run(monkeypatch):
    """No test of the command writes into the result of a research run the suite itself runs under."""

    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)


# ---- the rules, on small tables ------------------------------------------------------------------------------------------------

#: one request of 4 tokens: two block types cut from a GEMM row, one from a norm row, and one no step ran
CENSUS: dict[str, Any] = {
    "tokens": {"prefill": 3, "decode": 1}, "per_request": {"circuit_macs": 4000, "macs": 3600},
    "definitions": {
        'Gemm_v1{K=64,N=1,DOT={"fn":"Dot_v1"}}': {"k_log": 20, "ands": 1000, "of": {"Linear_v1": "row"}, "prefill": 6,
                                                  "decode": 2},
        "Gemm_v1{K=32,N=1}": {"k_log": 19, "ands": 400, "of": {"Linear_v1": "row"}, "prefill": 3, "decode": 1},
        'Norm_v1{N=8,EPS={"f64":"0x1.0p-3"},FMA=true,TREE=[0,[1,2]]}': {"k_log": 22, "ands": 3000,
                                                                    "of": {"Norm_v1": "row", "Add_v1": "part"},
                                                                    "prefill": 1, "decode": 1},
        "Idle_v1{N=1}": {"k_log": 18, "of": {"Idle_v1": "row"}, "prefill": 0, "decode": 0},
    },
}
GEMM64, GEMM32, NORM = (name for name in CENSUS["definitions"] if not name.startswith("Idle"))
CUT = {"name": "split", "label": "a toy cut", "replaces": ["Add_v1, Norm_v1"],
       "strata": [{"block": "Half_v1{N=4}", "kind": "Half_v1", "per_token": 1.0, "k_log": 21, "ands": 2000},
                  {"block": "Bare_v1{N=4}", "kind": "Bare_v1", "per_token": 0.25, "ands": 500},
                  {"block": "Unused_v1{N=4}", "kind": "Unused_v1", "per_token": 0, "k_log": 21}]}
#: single sessions: the K=64 GEMM's own, and another Definition's, both at 2 s a statement's fixed cost
SESSIONS = (PF.Session(GEMM64, 20, 0.5, 2.0, 1 << 15), PF.Session("Other_v1{N=2}", 22, 2.0, 2.0, 1 << 13))


def steady(type_: str, definition: str, run: str, blocks: int, gpu: float | None, *, full: int = 8, k_log: int = 20,
           verify: float | None = None, refused: str | None = None, provers: int = 1, art: str | None = None,
           cores: int = P.HOST_CORES, form: str | None = None) -> PF.Steady:
    """A steady-state row, timed at the host's cores unless `cores` says otherwise, its statements of `definition`
    unless `form` names another Definition."""

    return PF.Steady(type_, definition, k_log, blocks, full, run, R.ART_RATES if art else R.RATES, gpu, verify, refused,
                    provers, art, cores, form)


def lean(definition: str, run: str, seconds: tuple[float, ...], size: int, *, k_log: int = 20, blocks: int = 8,
         full: int = 8) -> PF.LeanVerify:
    return PF.LeanVerify(definition, k_log, blocks, full, seconds, size, run, "node-a (64 cores, a CPU)", L.COVER_LABEL)


def lean_shares(measured: tuple[int, float], family: tuple[int, float], fitted: tuple[int, float]) -> dict:
    return {source: {"block_types": n, "share": pytest.approx(share)}
            for source, (n, share) in zip(P.LEAN_SOURCES, (measured, family, fitted), strict=True)}


def plan(cells: list[dict], **over) -> P.SamplingPlan:
    return P.SamplingPlan.of({"case": "toy", "scope": "world year", "period_s": 100.0, "q": 0.01, "gpu_seconds": 800.0,
                              "replay_sample": 10.0, "cells": cells} | over, "art:plan")


#: each row of the refusals fixture that names a staged Definition, and why it isn't a steady-state measurement: the
#: check of the steady-state rules it didn't pass, the check's answer and its reason, then what it lacks to be priced
ROW_WHYS = {
    "good": None,
    "unaccepted": "`accepted` (fail): 8 of its 9 sessions accepted",
    "refused": "`accepted` (fail): 1 sessions refused",
    "void": "`not_void` (fail): void: out of device memory: prover-gpu0",
    "exited": "`exits` (fail): exited nonzero: prover-gpu0 rc -9",
    "unflat": "`flat` (fail): its statement cache's hit rate wasn't flat",
    "ungpued": "its row lacks `gpus`",
    "bare": "`flat` (cannot tell): its row lacks `flat`; its row lacks `gpus`, `stage.blocks`, `stage.full`",
}


# ---- the store side ------------------------------------------------------------------------------------------------------------


class Store:
    """A machine holding a workload's records: a local store with a directory remote, each art and run on it."""

    def __init__(self, tmp: Path):
        from research.store.local import LocalStore

        self.tmp, self.remote = tmp, tmp / "remote"
        (self.remote / "attempts").mkdir(parents=True)
        self.config = tmp / "store.toml"
        self.config.write_text(f'[remote]\ntype = "fs"\npath = "{self.remote}"\n')
        self.store = LocalStore(tmp / "producer", config_path=self.config)
        self._ts = 0

    def fresh(self, name: str = "fresh"):
        from research.store.local import LocalStore

        return LocalStore(self.tmp / name, config_path=self.config)

    def art(self, kind: str, meta: dict, *, file: Path | None = None, tree: Path | None = None) -> str:
        art = self.store.put_artifact(kind, meta, file=file, tree=tree)
        self.store.push(art)
        return art

    def _scratch(self) -> Path:
        self._ts += 1
        path = self.tmp / "docs" / str(self._ts)
        path.mkdir(parents=True)
        return path

    def doc(self, kind: str, name: str, doc: dict, meta: dict | None = None) -> str:
        """A file art holding `doc`."""

        path = self._scratch() / name
        path.write_text(json.dumps(doc))
        return self.art(kind, meta or {}, file=path)

    def tree(self, kind: str, name: str, doc: Any, meta: dict | None = None) -> str:
        """A tree art holding `doc` as its file `name`."""

        root = self._scratch()
        (root / name).write_text(json.dumps(doc))
        return self.art(kind, meta or {}, tree=root)

    def ran(self, run: str, files: Path | None, *, rc: int = 0, tool: str = "bash", validation: str | None = None) -> None:
        """A run's attempt on the remote, its files a run record (`files`: None for a run that kept none), recorded under
        `tool`, its result's validation `validation`."""

        from research.store import canon

        record = self.art("run-record/v1", {}, tree=files) if files else None
        attempt: dict[str, Any] = {
            "schema": "research/attempt/v0.1", "id": run, "tool": {"name": tool, "version": None},
            "derivation": "drv:" + "0" * 64, "reuse_key": None, "argv": [], "params": {}, "key_params": [], "inputs": {},
            "conditions": {}, "key_conditions": [], "source": {"commit": "c" * 40, "dirty": False}, "closure": None,
            "execution": {"start_utc": f"{run[1:5]}-{run[5:7]}-{run[7:9]}T00:00:00Z"}, "cost": None,
            "status": {"state": "done" if rc == 0 else "failed", "rc": rc, "validation": validation},
            "outputs": {"result": None, "run_record": record}, "telemetry": None,
        }
        (self.remote / "attempts" / f"{run}.json").write_bytes(canon.dumps(attempt))

    def label(self, target: str, key: str, value: str) -> None:
        self._ts += 1
        self.store.add_label(target, key, value, by="benchmarks", ts=f"2026-01-01T00:{self._ts // 60:02d}:{self._ts % 60:02d}Z")
        self.store.push_labels([target])

    def record(self, target: str, slot: str) -> str:
        self.label(target, R.RUN_OF_RECORD, W.record(slot))
        return target


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


RUNS = fixture("records.json")[R.STEADY]


def run_files(tmp: Path, run: str) -> Path:
    """A fixture run's files as its run record holds them: its proof server rate under `out/rate/`."""

    files = tmp / "run-files" / run
    if not files.exists():
        shutil.copytree(FIXTURES / "steady" / run, files / "out")
    return files


def census_art(s: Store, **meta) -> str:
    return s.tree(R.CENSUS_KIND, R.CENSUS_FILE, fixture("census.json"), {"model": W.model, "run": "r20260101-000000-0000"} | meta)


def plan_art(s: Store, **change) -> str:
    return s.tree("evidence/v1", R.PLAN_FILE, fixture("mix.json") | change)


#: the exfiltration run of record whose world year rows `fixtures/prove/exfiltration/` holds, cut to the fields its
#: reader reads (`exfiltration.at_share`)
DESIGN = fixture("records.json")[E.RECORD]


def designed(s: Store, tmp: Path, run: str = DESIGN) -> str:
    """The exfiltration run of record's rows in the store, its run files holding them as the planner's driver writes
    them, labelled its record."""

    files = tmp / "design-files" / run
    (files / E.ROWS_FILES[0]).parent.mkdir(parents=True)
    shutil.copy(FIXTURES / "exfiltration" / DESIGN / "rows.jsonl", files / E.ROWS_FILES[0])
    s.ran(run, files)
    s.label(run, R.RUN_OF_RECORD, E.RECORD)
    return run


def recorded(tmp: Path, **plan_change) -> tuple[Store, dict[str, str]]:
    """The fixtures' records in a store, each labelled its slot's record: the census, its cut, the single-session card,
    the three proof server runs and the sampling plan (with `plan_change` over its fields)."""

    s = Store(tmp)
    arts = {R.CENSUS: s.record(census_art(s), R.CENSUS),
            R.CUT: s.record(s.doc(R.CARD_SCHEMA, "partition.json", fixture("partition.json")), R.CUT),
            R.SINGLE: s.record(s.doc(R.CARD_SCHEMA, "proof_overhead.json", fixture("proof_overhead.json")), R.SINGLE),
            R.SAMPLING: s.record(plan_art(s, **plan_change), R.SAMPLING)}
    for run in RUNS:
        s.ran(run, run_files(tmp, run), rc=0 if run == RUNS[-1] else 1)
        s.record(run, R.STEADY)
    return s, arts


def registered(block: str) -> str:
    """The id of the Definition the tree registers as a census type's Boolean version, asked of the registry itself."""

    from verity.circuits.ops.boolean import versions as boolean
    from verity.circuits.call_scope import spec_base, spec_statics, specialization
    from verity_vllm.definitions.registry.catalog import load

    return boolean.boolean_version(specialization(spec_base(block), spec_statics(block), load()[0])).id


RMSNORM = next(name for name in fixture("census.json")["definitions"] if name.startswith("RMSNormFusedCudaVector8_v1{N=6144,"))
LEAN_HOST = {"node": "cpu-node-0", "cores": 64, "host_cores": 128, "cpu_model": "A CPU 9000", "mem_gib": 512.0, "commit": "c" * 40}


def lean_record(run: str, *, word: str | None = RMSNORM, seconds: tuple[float, ...] = (600.0, 500.0, 520.0),
                size: int = 2_000_000, why: str | None = None, **block) -> dict:
    """A made-up record of `verity bench prove lean-verify`: a full statement of 256 blocks of 2^27 witness bits, each
    run accepted, or its last refused with `why`."""

    runs: list[dict[str, Any]] = [{"wall_s": x, "rc": 0, "accepted": True, "why": None} for x in seconds]
    if why:
        runs[-1] |= {"rc": 1, "accepted": False, "why": why}
    return {"schema": L.SCHEMA, "run": run,
            "block": {"word": word, "definition": "Staged_v2{N=1}", "k_log": 27, "blocks": 256, "full": 256} | block,
            "proof": {"bytes": size}, "verifier": {"repeat": len(seconds), "runs": runs}, "accepted": not why,
            "label": None if why else L.COVER_LABEL, "host": dict(LEAN_HOST)}


def lean_ran(s: Store, tmp: Path, run: str, doc: dict | None, *, name: str = L.RECORD) -> str:
    """`run` in the store with `doc` as its `out/lean-verify.json` (None: a run that wrote none), a Lean verifier run
    of record."""

    files = tmp / "lean-files" / run / "out"
    files.mkdir(parents=True)
    (files / "job.log").write_text("ran")
    if doc is not None:
        (files / name).write_text(json.dumps(doc))
    s.ran(run, files.parent)
    return s.record(run, R.LEAN)


# ---- the Lean verifier's runner ------------------------------------------------------------------------------------------------


class Verifier:
    """The Lean verifier as the runner starts it, with no process: `archive-put` names each file by a key, and each
    `verify` takes the next verdict (None: it prints none), a made-up clock advancing by the seconds it took."""

    def __init__(self, monkeypatch, verdicts: list[tuple[dict | None, float]], *, put_rc: int = 0):
        self.verdicts, self.put_rc, self.now = list(verdicts), put_rc, 0.0
        self.calls: list[list[str]] = []
        monkeypatch.setattr(L, "spawn", self.spawn)
        monkeypatch.setattr(L, "clock", lambda: self.now)
        monkeypatch.setattr(L, "host", lambda: LEAN_HOST)

    def spawn(self, cmd, log: Path) -> tuple[int, str, dict]:
        self.calls.append(list(cmd))
        if cmd[1] == "archive-put":
            assert Path(cmd[2]).is_dir()
            self.now += 0.25
            return self.put_rc, "".join(f"key-{Path(p).name} {p}\n" for p in cmd[3:]), {}
        verdict, took = self.verdicts.pop(0)
        self.now += took
        session = cmd[cmd.index("--session") + 1]
        log.write_text(f"inputs: 40 ms\nsetup: 900 ms\n{session}: 2000 ms\n")
        accepted = bool(verdict and verdict["accepted"])
        return (0 if accepted else 1, "" if verdict is None else "VERDICT " + json.dumps(verdict) + "\n",
                {"cpu_s": took * 2, "user_s": took, "maxrss_gib": 1.5})


#: a made-up program descriptor of the census type `RMSNORM`, whose `annotations` are in no digest
PROGRAM = {"schema": "made-up", "program_fn": RMSNORM, "definitions": {"Staged_v1{N=1}": [1, 2]}, "annotations": {"theta": [3]}}
#: the program's file, spelled as no canonical form is
PROGRAM_FILE = json.dumps(PROGRAM, indent=1).encode()


def _sha512(prefix: bytes, obj: object) -> str:
    return hashlib.sha512(prefix + json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


#: the program's digest as the verifier computes it, over the descriptor without `annotations`, and what a unit
#: statement's circuit states as `program_sha512`: the digest of its `program_digests`, which no program file has
PROGRAM_SHA512 = _sha512(b"", {k: v for k, v in PROGRAM.items() if k != "annotations"})
UNIT_SHA512 = _sha512(b"", {"unit/made-up": hashlib.sha256(b"the staged Definition's id and the row layout").hexdigest()})
#: a made-up partition object, naming `PROGRAM` as a bound statement's names its Program
PARTITION = {"format": "made-up", "program": PROGRAM_SHA512, "query": {"name": "Q_made_up", "version": 1, "params": {"X": 16}}}


def circuit_text(rule: str = L.PARTITION_V1, size: int = 4000) -> bytes:
    """A circuit file of `size` bytes whose second line states a unit's `program_sha512` and binds, under `rule`,
    `PARTITION`."""

    meta = {"k_log": 27, "program_sha512": UNIT_SHA512,
            "partition": {"rule": rule, "digest": _sha512(rule.encode() + b"\0", PARTITION)}}
    head = b"flock-circuit unit/made-up\nMETA " + json.dumps(meta).encode() + b"\n"
    return head + b"x" * (size - len(head))


def kept_proof(tmp: Path, *, stage: dict | None = None, rule: str = L.PARTITION_COVER) -> Path:
    """A kept proof's tree: a staged statement's files, its circuit under `rule`, and one session's two proof files."""

    tree = tmp / "proof"
    (tree / "stage").mkdir(parents=True)
    (tree / "session").mkdir()
    for name, data in (("circuit.txt", circuit_text(rule)), ("pub-256.bin", b"x" * 64), ("program-word.json", PROGRAM_FILE)):
        (tree / "stage" / name).write_bytes(data)
    if stage is not None:
        (tree / "stage" / "stage.json").write_text(json.dumps(stage))
    (tree / "session" / "session.json").write_text("{}")
    (tree / "session" / "circuit.rep0.bin").write_bytes(b"p" * 300)
    (tree / "session" / "circuit.rep1.bin").write_bytes(b"p" * 200)
    return tree


STAGE = {"word": RMSNORM, "definition": "Staged_v2{N=1}", "k_log": 27, "n": 256}
#: the path the runner is given for the verifier: the fake starts no process, so no file is there
VERIFIER = "bin/lean-verifier"
ACCEPTED = {"accepted": True}
