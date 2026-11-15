"""What the tests of `verity bench exfiltration` share: the fixture's runs (`fixtures/exfiltration/`) and the
store that holds them, the network attester's calculator, and a run whose stop verdicts are of record."""

from __future__ import annotations

import json
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path

from verity_bench import exfiltration as E

REPO = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).parent / "fixtures" / "exfiltration"
RECORDS = json.loads((FIXTURES / "records.json").read_text())["compute-audit"]
ROWS, PLAN, PLANNED = RECORDS["rows"], RECORDS["plan"], RECORDS["planned"]
#: the world year's figure a hand computation gave on the fixture's rows, in GB: the command's row's here
YEAR_GB = 14239.837095861332
#: the row the fixture's runs were cut from, and the bound of the row (i) the tests make up from its rows over theirs
CUT_ROW = "(i) honest"
ASSUMED = 1.25


def as_row(x: dict) -> dict:
    """A row or a plan of the fixture's case as the command's row."""

    return x | {"row": E.ROW} if x.get("row") == CUT_ROW else x


def as_assumed(x: dict) -> dict:
    """A row of the fixture's case as a made-up row (i), its bound `ASSUMED` times theirs."""

    return x | {"row": E.ASSUMED_ROW, "bound_bits": ASSUMED * x["bound_bits"]}


def term():
    """The network attester's calculator, as the command loads it."""

    return E._load(REPO, "network_traces", "leak_bounds")


def delta(lb, response_ids: int = E.RESPONSE_IDS) -> Fraction:
    """The scans' δ the command gives the calculator where it is given none: the one at which the world's year, the
    row it prints that sums the most execution phases, carries `PERIOD_ERROR`."""

    year = lb.document(lb.inputs(response_ids=response_ids), only=("world",), periods=("year",))["rows"][0]
    return E.PERIOD_ERROR / Fraction(year["phases"])


def printed(text: str) -> list[tuple[str, str]]:
    """Everything a print shows, each with what it is printed under: a line of prose under itself, and a table's cell
    under its column's header and its row's name."""

    out, head = [], None
    for line in text.splitlines():
        if not line.startswith("|"):
            out, head = [*out, (line, line)], None
            continue
        row = [c.strip() for c in line.strip("|").split("|")]
        if head is None:
            head = row
        elif set(row) != {"---"}:
            out += [(f"{h} | {row[0]}", c) for h, c in zip(head, row, strict=True)]
    return out



def row(beta: float, total: float, bits: float, **more) -> dict:
    return {"scope": "world year", "step": "s", "mix": "m", "b10": beta, "total": total, "bound_bits": bits,
            "target_bits": 50.0, "weights_bits": 100.0, "delta_seen": 0.01, "S": 8.0, "gpu_seconds": 4.0, "serve": 0.01,
            "prove": total - 0.01, "q": 0.5, "replay_sample": 3.0, "requests": 7.0, **more}


class Store:
    """A machine holding the planner's runs: a local store with a directory remote, each run's attempt and files on it."""

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

    def ran(self, run: str, rows: list[dict] | Mapping[str, list[dict] | str] | None, *, plan: dict | None = None,
            name: str = E.ROWS_FILES[0]) -> str:
        """A run's attempt on the remote, its run record holding `rows` as the planner's rows, in the file `name` or
        by file (a file's rows, or its text), and `plan` as its plan (neither: it kept no files)."""

        from research.store import canon

        record = None
        if rows is not None or plan is not None:
            tree = self.tmp / "run-files" / f"{run}-{self._ts}"
            files = rows if isinstance(rows, Mapping) else {name: rows}
            for path, text in (*((f, its if isinstance(its, str) else its and "".join(json.dumps(r) + "\n" for r in its))
                                 for f, its in files.items()),
                               (E.PLAN_FILE, plan and json.dumps(plan))):
                if text:
                    (tree / path).parent.mkdir(parents=True, exist_ok=True)
                    (tree / path).write_text(text)
            record = self.store.put_artifact("run-record/v1", {}, tree=tree)
            self.store.push(record)
        attempt = {
            "schema": "research/attempt/v0.1", "id": run, "tool": {"name": "bash", "version": None},
            "derivation": "drv:" + "0" * 64, "reuse_key": None, "argv": [], "params": {}, "key_params": [], "inputs": {},
            "conditions": {}, "key_conditions": [], "source": {"commit": "c" * 40, "dirty": False}, "closure": None,
            "execution": {"start_utc": f"{run[1:5]}-{run[5:7]}-{run[7:9]}T00:00:00Z"}, "cost": None,
            "status": {"state": "done", "rc": 0, "validation": None},
            "outputs": {"result": None, "run_record": record}, "telemetry": None,
        }
        (self.remote / "attempts" / f"{run}.json").write_bytes(canon.dumps(attempt))
        return run

    def label(self, target: str, key: str = E.RUN_OF_RECORD, value: str = E.RECORD) -> str:
        self._ts += 1
        self.store.add_label(target, key, value, by="benchmarks", ts=f"2026-01-01T00:{self._ts // 60:02d}:{self._ts % 60:02d}Z")
        self.store.push_labels([target])
        return target


def rows(run: str = ROWS) -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / run / "rows.jsonl").read_text().splitlines()]


def plan() -> dict:
    """The fixture's plan, the command's row's."""

    return as_row(json.loads((FIXTURES / PLAN / "mix.json").read_text()))


def mine(scope: str | None = None) -> list[dict]:
    """The fixture's rows of the case the command takes, as its row: the row's variant on one request count, at eight
    heads."""

    return [as_row(r) for r in rows() if (r["world"], r["row"], r["h8"]) == (E.WORLD, CUT_ROW, "h8lin")
            and r["step"].startswith(E.STEP) and scope in (None, r["scope"])]


def record() -> dict[str, list[dict]]:
    """What a run of record holds of the fixture's rows, one variant of each row, by file as the planner's driver
    writes them: row (i) (made up) and the rows it must pass over in the first, the command's row in the second."""

    other = [r for r in rows() if r["row"] != CUT_ROW]
    return {E.ROWS_FILES[0]: [as_assumed(r) for r in mine()] + other, E.ROWS_FILES[1]: mine()}


def recorded(tmp: Path, *, planned: bool = True) -> Store:
    """The fixtures' runs in a store, each labelled the record: the rows over the two year scopes, and with `planned`
    the run that holds the world year's plan and the plan's own row."""

    s = Store(tmp)
    s.label(s.ran(ROWS, record()))
    if planned:
        s.label(s.ran(PLAN, [as_row(r) for r in rows(PLAN)], plan=plan()))
    return s


#: a made-up served run's record, as its reader gives it: the window served 1,000 s on 8 GPUs, 8,000 GPU-seconds, and the
#: audit's 960 of them, 12%: two proving sessions on a GPU each (480), staging (20), the replay (240), the interior
#: commitment (20) and the serving charge over stock on the same load (8,000 - 7,800)
SERVED = {"run": "r20261113-000000-5e7d", "serving": {"seconds": 1000.0, "gpus": 8},
          "prove": [{"seconds": 300.0, "gpus": 1}, {"seconds": 180.0, "gpus": 1}], "staging": [{"seconds": 20.0, "gpus": 1}],
          "replay": [{"seconds": 30.0, "gpus": 8}], "interior": [{"seconds": 5.0, "gpus": 4}],
          "stock": {"seconds": 975.0, "gpus": 8}, "prove_seconds_source": "busy"}
HOLDER = "r20261113-000000-a000"


def with_verdicts(tmp: Path, verdicts: list[dict] | str, run: str = HOLDER,
                  held: Mapping[str, list[dict]] | None = None) -> Store:
    """The fixture's rows of record, and ``run`` labelled the stop class's record holding ``verdicts`` (or that text)
    and, with ``held``, those files beside them (`held_phase`)."""

    s = recorded(tmp, planned=False)
    s.label(s.ran(run, {E.VERDICTS_FILE: verdicts, **(held or {})}), value=E.VERDICTS)
    return s


def reader():
    """The stop verdicts' reader, as the command loads it."""

    return E._reader()


def held_phase(formats: Mapping[str, str]) -> dict[str, list[dict]]:
    """A run's files, in the stop verdicts' reader's own format and by the names it reads, of one execution phase
    of a link with everything the reader holds a class to: the registrations' file, whose first line says what its
    offline check was handed, with the registration of a head whose one class holds 4 rows; the four verdicts on
    that class and head, each accepted and complete, the stop record's and the response tie's at the registered
    rate; each verifier's negative control, refused for its word; and the serve's registration with those two
    verdicts' streams, each registering every block of 4. ``formats`` is each verdict's format (`listed`). The
    reader holds no execution phase on them: it asks the network attester's rule without a schedule, and the rule
    charges no class below a schedule it was not handed. Nothing here is read by the command: the reader reads it
    and answers."""

    sv = reader()
    CC, C = sv.CC, sv.C
    tables = [("lengths", 4, 8), ("fails", 4, 8), *((name, 4, 16) for name in sv.CT.names(0))]
    head = C.Head("responses", 3, bytes(64), tuple(C.Entry(name, rows, size, bytes(64), bytes(64))
                                                   for name, rows, size in tables))
    registration = {"format": C.REGISTRATION, "link": head.link, "window": head.window,
                    "parameters": head.parameters.hex(), "word_bits": C.WORD_BITS, "row_schema": C.SCHEMA,
                    "tables": [{"name": e.name, "rows": e.rows, "words": e.row_bytes * 8 // C.WORD_BITS,
                                "domain": e.domain.hex(), "root": e.root.hex()} for e in head.tables],
                    "certificate": {"key": "", "digest": head.digest().hex(), "signature": ""}}
    serve = {"format": "a serve's registration"}
    streams = {kind: {"format": "a stream's registration", "window": f"the {kind}'s stream", sv.POPULATION: 4,
                      sv.DESIGN: {sv.DESIGN: sv.SUBSET, "k": 4}, sv.PREVIOUS: sv.digest_of(serve)} for kind in sv.RATED}
    by = {kind: {"module": f"the {kind}'s verifier", "commit": "a commit"} for kind in CC.VERDICTS}
    verdicts = [{CC.RUN: f"run of the {kind}", CC.SERVE: "the serve", CC.REGISTRATION: sv.digest_of(serve),
                 CC.VERIFIER: by[kind],
                 CC.VERDICT: {sv.FORMAT: formats[kind], CC.LINK: head.link, CC.WINDOW: head.window, CC.CLASS: 0,
                              CC.HEAD: head.digest().hex(), "registration": "a digest", CC.ACCEPTED: True,
                              CC.COMPLETE: True, "withheld": None, "codes": []},
                 CC.PROVED_BY: None} | ({sv.RATE: "1/1", sv.STREAM: sv.digest_of(streams[kind])} if kind in sv.RATED else {})
                for kind in CC.VERDICTS]
    controls = [{CC.RUN: f"control run of the {kind}", CC.VERIFIER: by[kind], sv.CONTROL: sv.CONTROLS[kind][0],
                 CC.VERDICT: {sv.FORMAT: formats[kind], CC.ACCEPTED: False, "withheld": None, "codes": ["a refusal"]},
                 sv.REASON: sv.REASONS[kind]} for kind in CC.VERDICTS]
    under = {"format": sv.CHECKED, sv.CHECKED_UNDER: {sv.KEY_SHA512: "5a" * 64, sv.SESSION: "c3" * 32}}
    return {sv.REGISTRATIONS_FILE: [under, registration], "out/the-verdicts.jsonl": verdicts,
            "out/the-controls.jsonl": controls, sv.SERVES_FILE: [serve, *streams.values()]}
