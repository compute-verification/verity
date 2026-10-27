"""The timed audit's verdict of record, two ways: `memory_challenger.verdict` (the Python reference, `protocol.TimedAudit`
replayed on the record) and `pous_check`'s `pous-timed-audit` (`PousCheck.TimedAudit`), on generated records and on the
records of runs.

    uv run python verity/core/protocols/pous/check/timed_audit.py --generated [--out FILE]
    uv run python verity/core/protocols/pous/check/timed_audit.py --replay RUN_DIR... [--out FILE]

`--generated` compares the two on honest P2 v4 records, at reduced widths and at the registered point, and on negative
controls of each part of the verdict (a wrong, flipped, short or long label, a leaf file of another store under its own
root, a wrong index, source or order, a late answer, too few or too many answers, no audit, a wrong root or limit, the
certificate's refusals of `k`, `Δ` and the RTT allowance, and records that are not P2 v4 or malformed), and through the
files on records whose JSON holds floats, `NaN` or `Infinity` (`file_instances`). It prints each
difference with both outputs, then `N instances, M differences, K negative controls`, and exits 0 iff M is 0. A negative
control is an instance built to be refused, faulted or rejected; the generator checks the reference does not accept it.

`--replay` judges each audit document of a run's record (`bench/audit-p2-<arm>.json` with `bench/ready-p2-<arm>.json`
and `bench/leaves-p2.bin`, as `band_gpu/harness.py` keeps them) both ways and prints
`N runs, N same verdict, K negative controls` over the documents it can replay. A document whose answers do not carry
their labels cannot be replayed (the audit checks each label against its leaf), and neither can a record of another
version than P2 v4; it says which, and counts neither. Where a document kept the verdicts its run computed (`verdicts`,
`judge`'s), each one that ran is held to the replay's of the same name, and `M with recorded verdicts, M recorded as
replayed` follows; the exit is 0 iff every document replays the same both ways and as recorded.

Lean implements the verdict only: the records are the memory challenger's, which Python's audit wrote.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from verity_difftest import Server, build, canonical

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
EXE = "pous-timed-audit"


def challenger():
    """`benchmarks/pous/band_gpu/memory_challenger.py`, the reference."""
    path = ROOT / "benchmarks" / "pous" / "band_gpu" / "memory_challenger.py"
    spec = importlib.util.spec_from_file_location("pous_memory_challenger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lean_verdict(ready: Any, leaves: bytes, doc: Any) -> dict[str, Any]:
    """`memory_challenger.verdict` by `pous-timed-audit READY LEAVES AUDIT`."""
    exe = build(HERE, EXE)
    with tempfile.TemporaryDirectory() as tmp:
        files = [Path(tmp) / name for name in ("ready.json", "leaves.bin", "audit.json")]
        files[0].write_text(json.dumps(ready))
        files[1].write_bytes(leaves)
        files[2].write_text(json.dumps(doc))
        p = subprocess.run([str(exe), *map(str, files)], capture_output=True, text=True)
    if p.returncode not in (0, 1):
        raise RuntimeError(f"{EXE} failed ({p.returncode}): {p.stderr}")
    return json.loads(p.stdout)


class Store:
    """A P2 v4 store for the verdict: its labels need only match their leaves (the audit checks leaves, not codewords)."""

    def __init__(self, params, seed: bytes, salt: bytes = b"s" * 24) -> None:
        from verity.core.protocols.pous.verifier import VerifierKey, leaf_hash

        self.params, self.seed, self.salt = params, seed, salt
        leaves = [leaf_hash(params, salt, i, self.block(i)) for i in range(params.blocks)]
        self.leaves, self.vk = b"".join(leaves), VerifierKey.from_leaves(params, salt, leaves)

    def block(self, i: int) -> bytes:
        return hashlib.shake_256(self.seed + i.to_bytes(8, "big")).digest(self.params.block_bytes)

    @property
    def ready(self) -> dict[str, Any]:
        return {"scheme": self.params.codec.name, "params": self.params.manifest(), "salt": self.salt.hex(), "port": 0}


def record(s: Store, rng: random.Random, *, k: int, deadline: int, rtt: int, strict: bool, audits: int,
           elapsed: Callable[[int], int] | None = None) -> dict[str, Any]:
    """An honest audit document, as `memory_challenger.audit` writes it: each answer the label of the index revealed."""
    from verity.core.protocols.pous import protocol

    limit = deadline + rtt
    elapsed = elapsed or (lambda j: rng.randint(0, limit))
    records = []
    for _ in range(audits):
        source = rng.randbytes(32)
        key = protocol.challenge_key(source, s.vk)
        answers = []
        for j in range(k):
            i = key.uniform(s.params.blocks, j)
            answers.append({"index": i, "elapsed_ns": elapsed(j), "device_ns": 0, "server_ns": 0, "block": s.block(i).hex()})
        records.append({"source": source.hex(), "answers": answers})
    return {"root": s.vk.root.hex(), "clock": "logical", "deadline_ns": deadline, "rtt_allowance_ns": rtt, "limit_ns": limit,
            "k": k, "strict": strict, "records": records}


def _edit(doc: dict, f: Callable[[dict], Any]) -> dict:
    out = json.loads(json.dumps(doc))
    f(out)
    return out


def _answer(doc: dict, a: int = 0, j: int = 0) -> dict:
    return doc["records"][a]["answers"][j]


def _flip(h: str) -> str:
    return h[:-1] + ("0" if h[-1] != "0" else "1")


def controls(s: Store, doc: dict, other: Store) -> Iterator[tuple[str, Any, bytes, Any]]:
    """Negative controls of an honest document `doc` of `s` (with at least two answers and an index of another label):
    `(name, ready, leaves, doc)`."""
    ready, leaves, a0 = s.ready, s.leaves, _answer(doc)
    elsewhere = next(i for i in range(s.params.blocks) if i != a0["index"])
    blk = a0["block"]
    for name, f in [
        ("wrong block", lambda d: _answer(d).update(block=s.block(elsewhere).hex())),
        ("flipped block", lambda d: _answer(d).update(block=_flip(blk))),
        ("short block", lambda d: _answer(d).update(block=blk[:-2])),
        ("long block", lambda d: _answer(d).update(block=blk + "00")),
        ("empty block", lambda d: _answer(d).update(block="")),
        ("zero block", lambda d: _answer(d).update(block="00" * s.params.block_bytes)),
        ("wrong index", lambda d: _answer(d).update(index=elsewhere)),
        ("index past the blocks", lambda d: _answer(d).update(index=s.params.blocks)),
        ("wrong source", lambda d: d["records"][0].update(source=_flip(d["records"][0]["source"]))),
        ("answers swapped", lambda d: d["records"][0]["answers"].reverse()),
        ("late", lambda d: _answer(d).update(elapsed_ns=d["limit_ns"] + 1)),
        ("every answer late", lambda d: [a.update(elapsed_ns=10 ** 12) for a in d["records"][0]["answers"]]),
        ("too few answers", lambda d: d["records"][0]["answers"].pop()),
        ("no answers", lambda d: d["records"][0].update(answers=[])),
        ("too many answers", lambda d: d["records"][0]["answers"].append(dict(_answer(d)))),
        ("no audit", lambda d: d.update(records=[])),
        ("wrong root", lambda d: d.update(root=_flip(d["root"]))),
        ("short root", lambda d: d.update(root=d["root"][:-2])),
        ("limit not Δ plus the allowance", lambda d: d.update(limit_ns=d["limit_ns"] + 1)),
        ("record not an object", lambda d: d.update(records=[[]])),
        ("no source", lambda d: d["records"][0].pop("source")),
        ("empty source", lambda d: d["records"][0].update(source="")),
        ("uppercase source", lambda d: d["records"][0].update(source=d["records"][0]["source"].upper())),
        ("answers not a list", lambda d: d["records"][0].update(answers={})),
        ("answer not an object", lambda d: d["records"][0]["answers"].__setitem__(0, 0)),
        ("index a string", lambda d: _answer(d).update(index=str(a0["index"]))),
        ("no elapsed time", lambda d: _answer(d).pop("elapsed_ns")),
        ("negative elapsed time", lambda d: _answer(d).update(elapsed_ns=-1)),
        ("elapsed time a boolean", lambda d: _answer(d).update(elapsed_ns=True)),
        ("no block", lambda d: _answer(d).pop("block")),
        ("odd-length block", lambda d: _answer(d).update(block=blk[:-1])),
        ("uppercase block", lambda d: _answer(d).update(block=blk.upper()) if blk != blk.upper() else
         _answer(d).update(block="0A" + blk[2:])),
        ("block not hex", lambda d: _answer(d).update(block="zz" + blk[2:])),
        ("strict a number", lambda d: d.update(strict=1)),
        ("no strict", lambda d: d.pop("strict")),
        ("k a string", lambda d: d.update(k=str(d["k"]))),
        ("k null", lambda d: d.update(k=None)),
        ("negative deadline", lambda d: d.update(deadline_ns=-1)),
        ("no allowance", lambda d: d.pop("rtt_allowance_ns")),
        ("records not a list", lambda d: d.update(records={})),
    ]:
        yield name, ready, leaves, _edit(doc, f)
    yield "document not an object", ready, leaves, [doc]
    yield "another store's leaves under its own root", ready, other.leaves, {**doc, "root": other.vk.root.hex()}
    yield "another store's leaves", ready, other.leaves, doc
    yield "leaves short", ready, leaves[:-64], doc
    yield "leaves cut", ready, leaves[:-1], doc
    yield "another salt", {**ready, "salt": "00" * 24}, leaves, doc
    yield "salt short", {**ready, "salt": ready["salt"][:-2]}, leaves, doc
    yield "salt uppercase", {**ready, "salt": "AB" * 24}, leaves, doc
    yield "no salt", {k: v for k, v in ready.items() if k != "salt"}, leaves, doc
    yield "scheme v3", {**ready, "scheme": ready["scheme"].replace("/v4", "/v3")}, leaves, doc
    yield "scheme v2", {**ready, "scheme": ready["scheme"].replace("/v4", "/v2")}, leaves, doc
    yield "scheme at another width", {**ready, "scheme": "p2-8/v4"}, leaves, doc
    m = ready["params"]
    for name, edit in [("chacha8 expander", {"expander": "chacha8"}), ("v3 expander", {"expander": "shake256-v1"}),
                       ("no expander", {"expander": None}), ("extra key", {"extra": 1}),
                       ("profile", {"profile": "p2-1B"}), ("scheme band", {"scheme": "band"}),
                       ("segments 0", {"segments": 0}), ("w a string", {"w": str(m["w"])}),
                       ("c0 ≢ 1 mod 4", {"c0": m["c0"] + 2}), ("w not whole bytes", {"w": m["w"] + 1}),
                       ("blocks per segment past 2^16", {"blocks_per_segment": (1 << 16) + 1})]:
        params = {k: v for k, v in {**m, **edit}.items() if v is not None}
        yield f"manifest: {name}", {**ready, "params": params}, leaves, doc
    yield "ready not an object", [], leaves, doc
    yield "no params", {k: v for k, v in ready.items() if k != "params"}, leaves, doc


def instances(seed: int = 0) -> Iterator[dict[str, Any]]:
    """Every instance: `{"name", "negative", "ready", "leaves", "audit"}`."""
    from verity.core.protocols.pous.schemes import p2 as P2

    rng = random.Random(seed)
    delta, cap = P2.DELTA_NS, P2.RTT_ALLOWANCE_CAP_NS

    def inst(name, s_or_ready, leaves, doc, negative):
        ready = s_or_ready.ready if isinstance(s_or_ready, Store) else s_or_ready
        return {"name": name, "negative": negative, "ready": ready, "leaves": leaves, "audit": doc}

    small = [(24, 5, 1, 1), (24, 65533, 3, 2), (32, 5, 8, 3), (64, 189, 2, 1), (64, 189, 16, 2), (128, 173, 5, 4),
             (256, 189, 4, 1), (512, 569, 1, 7), (1024, 105, 3, 1)]
    for w, c0, bps, segments in small:
        s = Store(P2.P2Params(w, c0, bps, segments), rng.randbytes(16))
        other = Store(s.params, rng.randbytes(16), salt=rng.randbytes(24))
        k = rng.randint(2, 6)
        tag = f"p2-{w}/v4 c0={c0} {bps}x{segments}"
        doc = record(s, rng, k=k, deadline=rng.randint(0, 10 ** 6), rtt=rng.randint(0, 10 ** 6), strict=False, audits=3)
        yield inst(f"{tag} honest research", s, s.leaves, doc, False)
        yield inst(f"{tag} honest, every answer at the limit", s, s.leaves,
                   record(s, rng, k=k, deadline=delta, rtt=cap, strict=False, audits=1, elapsed=lambda j: delta + cap), False)
        yield inst(f"{tag} k = 0", s, s.leaves, record(s, rng, k=0, deadline=delta, rtt=cap, strict=False, audits=1), False)
        yield inst(f"{tag} no k", s, s.leaves, {kk: v for kk, v in record(s, rng, k=132, deadline=delta, rtt=0, strict=False,
                                                                      audits=1).items() if kk != "k"}, False)
        yield inst(f"{tag} strict: no certificate", s, s.leaves,
                   record(s, rng, k=k, deadline=delta, rtt=cap, strict=True, audits=1), True)
        yield inst(f"{tag} second audit wrong", s, s.leaves,
                   _edit(doc, lambda d: d["records"][1]["answers"][0].update(block=_flip(d["records"][1]["answers"][0]["block"]))),
                   True)
        if s.params.blocks >= 8:
            for name, ready, leaves, bad in controls(s, doc, other):
                yield inst(f"{tag} {name}", ready, leaves, bad, True)

    s = Store(P2.P2Params(*P2.POINT), rng.randbytes(16))
    other = Store(s.params, rng.randbytes(16))
    k = P2.CERT_K
    tag = "p2-16448/v4 at the registered point"
    doc = record(s, rng, k=k, deadline=delta, rtt=cap, strict=True, audits=2)
    yield inst(f"{tag} honest strict", s, s.leaves, doc, False)
    yield inst(f"{tag} honest strict, no k", s, s.leaves, {kk: v for kk, v in doc.items() if kk != "k"}, False)
    yield inst(f"{tag} honest research", s, s.leaves, {**doc, "strict": False}, False)
    yield inst(f"{tag} honest strict, an answer at the limit", s, s.leaves,
               _edit(doc, lambda d: _answer(d).update(elapsed_ns=delta + cap)), False)
    yield inst(f"{tag} honest strict, Δ + RTT under the cap", s, s.leaves,
               record(s, rng, k=k, deadline=delta, rtt=1000, strict=True, audits=1), False)
    yield inst(f"{tag} honest strict, k past the certificate's", s, s.leaves,
               record(s, rng, k=k + 1, deadline=delta, rtt=cap, strict=True, audits=1), False)
    for name, kw in [("k below the certificate's", {"k": k - 1}), ("Δ not the spec's", {"deadline": delta + 1}),
                     ("Δ below the spec's", {"deadline": delta - 1}), ("RTT past the cap", {"rtt": cap + 1}),
                     ("the limit fits a root call", {"rtt": cap + 1000}), ("k below and RTT past the cap",
                                                                          {"k": k - 1, "rtt": cap + 1})]:
        args = {"k": k, "deadline": delta, "rtt": cap, **kw}
        bad = record(s, rng, strict=True, audits=1, **args)
        yield inst(f"{tag} strict, {name}", s, s.leaves, bad, True)
        yield inst(f"{tag} research, {name}", s, s.leaves, {**bad, "strict": False}, False)
        yield inst(f"{tag} strict refused as recorded, {name}", s, s.leaves, {**bad, "records": []}, True)
    for name, ready, leaves, bad in controls(s, doc, other):
        if "manifest" in name or "salt" in name or "scheme" in name or "leaves" in name or "block" in name or "late" in name:
            yield inst(f"{tag} {name}", ready, leaves, bad, True)


def file_instances(seed: int = 0) -> Iterator[dict[str, Any]]:
    """Records whose files hold what the server's canonical JSON can't carry, compared through the files (`lean_verdict`):
    floats Python's `json` writes in exponent form (`1e+16`, which `Json.parse` alone reads as a natural), and `NaN`,
    `Infinity` and `-Infinity` in fields the verdict reads and in fields it doesn't."""
    from verity.core.protocols.pous.schemes import p2 as P2

    rng = random.Random(seed)
    s = Store(P2.P2Params(64, 189, 4, 2), rng.randbytes(16))
    big = 10 ** 16
    doc = record(s, rng, k=3, deadline=big, rtt=0, strict=False, audits=1, elapsed=lambda j: j)

    def inst(name, doc, ready=None, negative=True):
        return {"name": f"file: {name}", "negative": negative, "ready": ready or s.ready, "leaves": s.leaves, "audit": doc}

    yield inst("honest, Δ 10^16", doc, negative=False)
    yield inst("Δ and the limit the float 1e16", {**doc, "deadline_ns": float(big), "limit_ns": float(big)})
    yield inst("the limit the float 1e16", {**doc, "limit_ns": float(big)})
    yield inst("Δ the float 1e16", {**doc, "deadline_ns": float(big)})
    yield inst("an elapsed time the float 1e16", _edit(doc, lambda d: _answer(d).update(elapsed_ns=float(big))))
    yield inst("an elapsed time the float 1e-05", _edit(doc, lambda d: _answer(d).update(elapsed_ns=1e-05)))
    yield inst("k the float 3.0", {**doc, "k": 3.0})
    yield inst("Δ NaN", {**doc, "deadline_ns": float("nan")})
    yield inst("the clock NaN, unread", {**doc, "clock": float("nan")}, negative=False)
    yield inst("the clock -Infinity, unread", {**doc, "clock": float("-inf")}, negative=False)
    yield inst("an answer's device time Infinity, unread",
               _edit(doc, lambda d: _answer(d).update(device_ns=float("inf"))), negative=False)
    yield inst("the ready file's port NaN, unread", doc, ready={**s.ready, "port": float("nan")}, negative=False)
    yield inst("the manifest's w the float 64.0", doc, ready={**s.ready, "params": {**s.ready["params"], "w": 64.0}})


def python_verdict(mc, inst: dict) -> str:
    try:
        return canonical(mc.verdict(inst["ready"], inst["leaves"], inst["audit"]))
    except Exception as e:
        return canonical({"exception": f"{type(e).__name__}: {e}"})


def _lean_side(server: Server, inst: dict) -> str:
    kind, text = server.call("verdict", {"ready": inst["ready"], "leaves": inst["leaves"].hex(), "audit": inst["audit"]})
    return text if kind == "output" else canonical({"reject": text})


def compare(seed: int = 0) -> list[dict]:
    """Every generated instance with both outputs."""
    mc, out = challenger(), []
    with Server([str(build(HERE, EXE)), "serve"]) as lean:
        for inst in instances(seed):
            out.append({"id": inst["name"], "negative": inst["negative"], "python": python_verdict(mc, inst),
                        "lean": _lean_side(lean, inst)})
    for inst in file_instances(seed):
        out.append({"id": inst["name"], "negative": inst["negative"], "python": python_verdict(mc, inst),
                    "lean": canonical(lean_verdict(inst["ready"], inst["leaves"], inst["audit"]))})
    return out


def accepted(text: str) -> bool:
    return json.loads(text).get("accepted") is True


def report(results: list[dict], out: Path | None = None) -> int:
    """Print each difference, each negative control the reference accepts, and the count; write `out`; the exit code."""
    differences = [r for r in results if r["python"] != r["lean"]]
    for r in differences:
        print(f"DIFFERENCE {r['id']}\n  python {r['python']}\n  lean   {r['lean']}")
    vacuous = [r for r in results if r["negative"] and accepted(r["python"])]
    for r in vacuous:
        print(f"ACCEPTED NEGATIVE CONTROL {r['id']}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(canonical(r | {"same": r["python"] == r["lean"]}) + "\n" for r in results))
    negatives = sum(r["negative"] for r in results)
    print(f"{len(results)} instances, {len(differences)} differences, {negatives} negative controls")
    return 1 if differences or vacuous or not any(accepted(r["python"]) for r in results) else 0


def replay(runs: list[Path], out: Path | None = None) -> int:
    """Each audit document of each run's record judged both ways, and each verdict the run recorded (`verdicts`, but a Lean
    `unavailable` there) held to the replay's of the same name; prints what it cannot replay and why; writes `out`, one
    JSON line per document replayed."""
    mc = challenger()
    replayed = same = negatives = recorded = agree = 0
    rows = []
    for run in runs:
        bench = run / "bench" if (run / "bench").is_dir() else run
        docs = sorted(bench.glob("audit-p2-*.json"))
        if not docs:
            print(f"{run}: no audit-p2-*.json")
            continue
        for path in docs:
            arm = path.stem.removeprefix("audit-p2-")
            ready_path, leaves_path = bench / f"ready-p2-{arm}.json", bench / "leaves-p2.bin"
            doc = json.loads(path.read_text())
            lacks = [f"no {p.name}" for p in (ready_path, leaves_path) if not p.exists()]
            ready = json.loads(ready_path.read_text()) if ready_path.exists() else {}
            scheme, expander = ready.get("scheme"), (ready.get("params") or {}).get("expander")
            if ready and (not str(scheme).endswith("/v4") or expander != "sha512-ctr-v4"):
                lacks.append(f"{scheme} with expander {expander} is not P2 v4")
            answers = [a for rec in doc.get("records", []) for a in rec.get("answers", [])]
            if unblocked := sum("block" not in a for a in answers):
                lacks.append(f"{unblocked} of {len(answers)} answers do not record their block")
            if lacks:
                print(f"{run.name} {arm}: cannot replay, {'; '.join(lacks)}")
                continue
            leaves = leaves_path.read_bytes()
            py, lean = canonical(mc.verdict(ready, leaves, doc)), canonical(lean_verdict(ready, leaves, doc))
            replayed += 1
            same += py == lean
            negatives += not accepted(py)
            row = {"run": run.name, "arm": arm, "python": py, "lean": lean, "same": py == lean, "negative": not accepted(py)}
            kept = {name: canonical(v) for name, v in (doc.get("verdicts") or {}).items()
                    if not (isinstance(v, dict) and "unavailable" in v)}
            note = ""
            if kept:
                row["recorded"] = kept
                row["recorded_same"] = all(v == {"python": py, "lean": lean}.get(name) for name, v in kept.items())
                recorded += 1
                agree += row["recorded_same"]
                note = f"; recorded {' and '.join(sorted(kept))} {'the same' if row['recorded_same'] else 'DIFFERENT'}"
            rows.append(row)
            print(f"{run.name} {arm}: {'same' if py == lean else 'DIFFERENT'} {py if py == lean else py + ' / ' + lean}{note}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(canonical(r) + "\n" for r in rows))
    print(f"{replayed} runs, {same} same verdict, {negatives} negative controls")
    if recorded:
        print(f"{recorded} with recorded verdicts, {agree} recorded as replayed")
    return 0 if replayed and same == replayed and agree == recorded else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generated", action="store_true", help="the generated instances")
    mode.add_argument("--replay", type=Path, nargs="+", metavar="RUN_DIR", help="the audit documents of these runs")
    ap.add_argument("--out", type=Path, help="write one JSON line per instance (or replayed document) with both outputs")
    a = ap.parse_args(argv)
    if a.replay:
        return replay(a.replay, a.out)
    return report(compare(), a.out)


if __name__ == "__main__":
    sys.exit(main())
