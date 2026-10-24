"""Prove path: a committed TP run and its advice through the proof service, for network accounting under arbitrary vLLM usage.

Each rank's step units are Programs a registered compiler made from the advice (`pipeline.advice_windows`), not a declared
workload's. The advice is private: the prover commits I1's canonical encoding under hiding rows (`advice_windows.commit_advice`),
and each rank's registration binds the compiler's digest and that commitment's root, before any coin. The prover commits each
rank's window over the compiler's Programs from the rank's sealed bundle. The verifier, handed the commitment's opening (the
named gap: the advice is opened to the verifier until the compile unit exists, README.md), compiles the advice with its own copy
of the compiler, derives its own window and audits the rank through the one-stage audit's per-rank `service.Stream`
(`benchmarks/one_stage/a5.py`'s `audit`): register, select on the verifier's own live coins (read only after the receipt),
C-Flock `--zk` behind the Lean firewall on the drawn units, Lean's verdicts, and the outcome, which the Lean verifier of record's
`Run.run` must agree with. It then checks that the advice was fixed before the draw (`advice_before_draw`).

    prove_path.py advice --workload F --out FILE
    prove_path.py commit --tp-commit DIR --build-root DIR --advice FILE --opening FILE --partition F --out WINDOWS \\
        [--cap-bits L] [--ranks 0,1] [--jobs N] [--forge MEMBER:UNIT]
    prove_path.py audit --tp-commit DIR --build-root DIR --opening FILE --windows WINDOWS --partition F --template-sets DIR \\
        --m0-pythonpath PATH --bin flock-circuit --lean flock-verify --firewall flock-firewall --out DIR \\
        [--cap-bits L] [--ranks 0,1] [--jobs N] [--forged]

* **advice**: a declared run's I1 advice from its workload file (`advice_windows.advice_of_workload`), its canonical encoding.
  A served run's comes from its step trace instead (ADVICE-SRC); `compile_advice` reads either.
* **commit** (the prover): the advice compiled (`compile_advice`) and committed, its opening (the advice and its salts) written to
  ``--opening``, the prover's private file, never under the run's out/; then each rank's stand-in window from its bundle over
  the compiler's Programs for it (`advice_windows.commit_window`), ``WINDOWS/rank<r>/``. ``--forge MEMBER:UNIT`` commits that
  unit with bit 0 of its first output word flipped (`forging`): its rows, commit strings and roots all agree with the flipped
  word, so only the proof can refuse it.
* **audit** (the verifier): each rank's window from the record, the opened advice and its own compiler (`open_advice`,
  `advice_windows.window_of_advice`), audited by a5 on the verifier's coins with a5's advice negatives and this driver's beside
  them, then `advice_before_draw` on the draw's receipt and the advice-root-after-the-draw negative (`advice_after_draw`);
  ``OUT/summary.json`` over the ranks. ``--forged``: the prover runs with its self-check skipped (``FC_ZK_SELF_CHECK=skip``) so it
  sends proofs of the false witness, and a rank passes when its registration passed every check before the draw and Lean
  refused the outcome (`forged_row`).

The compiler is `Capped` over `advice_windows.Traced` (the Build's Programs of record, refused unless of record), until the
ahead-of-time compiler (`verity_vllm.predict.steps.compile_steps`) makes a TP rank's Programs. `Capped` adds the public cap ℓ on
the advice's length to the compiler's registered digest, and `compile_advice` is total on every byte string: the requests, or
a `ValueError` naming why, the same one on every run.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "one_stage"))

from verity.core.protocols.sampled_proofs.one_stage import registration as R  # noqa: E402
from verity.ml.circuits.codec import canonical_json  # noqa: E402
from verity_vllm.pipeline import advice_windows as AW  # noqa: E402
from verity_vllm.pipeline.tp import serving_rows as SRT  # noqa: E402

CAPPED = "verity/prove-path/capped-compiler/v0"
#: ℓ, the public cap on the advice's encoding in bits: 1 MiB, far above a TP Commit's few hundred requests (I1 spends tens of
#: bits a row)
CAP_BITS = 1 << 23


@dataclass(frozen=True)
class Capped:
    """A registered compiler with ℓ, the public cap on the advice's length in bits. Its digest names the inner compiler's and ℓ,
    so the registration's ``compiler`` fixes ℓ before the draw as it fixes the code; `load` is the inner compiler's."""

    inner: Any
    cap_bits: int = CAP_BITS
    name: str = CAPPED

    def __post_init__(self) -> None:
        if isinstance(self.cap_bits, bool) or not isinstance(self.cap_bits, int) or self.cap_bits < 8:
            raise ValueError(f"the cap ℓ is a whole number of bits, at least 8, not {self.cap_bits!r}")

    @property
    def digest(self) -> str:
        return hashlib.sha512(canonical_json({"name": self.name, "cap_bits": self.cap_bits,
                                              "compiler": {"name": str(self.inner.name), "digest": self.inner.digest}})).hexdigest()

    def load(self, rank: int, world: int, requests: Any) -> Any:
        return self.inner.load(rank, world, requests)


def compiler_of(tp_commit: str, build_root: str, cap_bits: int = CAP_BITS) -> Capped:
    return Capped(AW.Traced.for_commit(tp_commit, build_root), cap_bits)


def compile_advice(compiler: Capped, data: Any, codec: Any = None) -> list[dict]:
    """The compiler on any input: the requests it compiles (`advice_windows.requests_of`), or a `ValueError` naming why it refuses
    the input, a function of the input alone. Refused in order: not bytes; over the cap ℓ (before any of it is read); not I1
    advice (`MalformedAdvice`, the codec's first broken rule); not its canonical encoding; a request the compiler can't run (a
    chunked prefill, a prefix-cache hit, a preemption, a gap)."""
    if not isinstance(data, bytes):
        raise ValueError(f"the advice is a byte string, not a {type(data).__name__}")
    if 8 * len(data) > compiler.cap_bits:
        raise ValueError(f"the advice is {8 * len(data)} bits, over the registered cap ℓ = {compiler.cap_bits} bits")
    c = AW.codec_of(codec)
    advice = c.decode(data)
    if c.encode(advice) != data:
        raise ValueError("the advice's bytes are not its canonical encoding")
    return AW.requests_of(advice)


def commit_advice(compiler: Capped, data: bytes) -> tuple[list[dict], dict, dict]:
    """The prover's side: the requests compiled from the advice (`compile_advice`), the window's binding (the compiler's digest
    and the root of the advice's hiding commitment, `advice_windows.of_advice`) and the opening (`advice_windows.opening`)."""
    requests = compile_advice(compiler, data)
    _reqs, bound, reg = AW.of_advice(compiler, data)
    return requests, bound, AW.opening(data, reg)


def open_advice(compiler: Capped, doc: Any) -> tuple[list[dict], dict]:
    """The verifier's side of the opening (the named gap): the requests it compiles from the opened advice and the binding the
    registration must carry (`advice_windows.of_opening`: the root recomputed from the advice and its salts). A malformed opening is
    a ValueError, as `compile_advice`'s refusals are."""
    if not isinstance(doc, Mapping) or doc.get("scheme") != AW.SCHEME or not isinstance(doc.get("advice"), str):
        raise ValueError(f"an advice opening is a {AW.SCHEME} document with the advice in hex")
    try:
        data = bytes.fromhex(doc["advice"])
    except ValueError as e:
        raise ValueError(f"an advice opening holds the advice in hex: {e}") from e
    return compile_advice(compiler, data), AW.of_opening(compiler, doc)[1]


def compiled(a: argparse.Namespace) -> tuple[Capped, list[dict], dict, dict]:
    """The verifier's compiler, requests and binding from the opening ``a.opening``, and the opening."""
    compiler = compiler_of(a.tp_commit, a.build_root, a.cap_bits)
    opening = json.loads(Path(a.opening).read_text())
    return (compiler, *open_advice(compiler, opening), opening)


@contextlib.contextmanager
def forging(member: int, unit: int) -> Iterator[None]:
    """Inside, serving rows commit member ``member``'s unit ``unit`` with bit 0 of its first output word flipped, before its commit
    strings and roots are made over it."""
    from verity_vllm.pipeline import serving_rows as SRW

    real = SRW.read_all

    def read_all(ms, rows, store, weights, arrivals=None):
        got = real(ms, rows, store, weights, arrivals)
        port = ms[member].outs[0][0]
        got[member][0]["outs"][port][unit, 0] ^= 1
        return got

    SRW.read_all = read_all
    try:
        yield
    finally:
        SRW.read_all = real


def ranks_of(tp_commit: str, ranks: str | None) -> tuple[int, list[int]]:
    world = int(SRT.record(tp_commit)["world"])
    return world, [int(x) for x in ranks.split(",")] if ranks else list(range(world))


def in_processes(fn, jobs: int, items: list) -> Iterator[tuple[Any, Any]]:
    """``fn(item)`` of each item, in up to ``jobs`` spawned processes (a rank's Programs don't pickle, so a process does a rank);
    -> (item, result) as each finishes."""
    if jobs <= 1 or len(items) <= 1:
        for x in items:
            yield x, fn(x)
        return
    with ProcessPoolExecutor(min(jobs, len(items)), mp_context=multiprocessing.get_context("spawn")) as ex:
        futures = {ex.submit(fn, x): x for x in items}
        for f in as_completed(futures):
            yield futures[f], f.result()


def commit_rank(job: tuple[argparse.Namespace, int, list[dict], dict]) -> dict:
    a, r, requests, bound = job
    compiler = compiler_of(a.tp_commit, a.build_root, a.cap_bits)
    forge = [int(x) for x in a.forge.split(":")] if a.forge else None
    with forging(*forge) if forge else contextlib.nullcontext():
        got = AW.commit_window(a.tp_commit, r, a.partition, Path(a.out) / f"rank{r}", compiler, requests, bound, workers=a.workers)
    return {**got, "forged": {"member": forge[0], "unit": forge[1]} if forge else None}


def commit(a: argparse.Namespace) -> int:
    """The advice committed once, so every rank's registration binds one root, then each rank's window."""
    _world, ranks = ranks_of(a.tp_commit, a.ranks)
    requests, bound, opening = commit_advice(compiler_of(a.tp_commit, a.build_root, a.cap_bits), Path(a.advice).read_bytes())
    Path(a.opening).write_text(json.dumps(opening) + "\n")
    rows = {}
    for (_a, r, _q, _b), got in in_processes(commit_rank, a.jobs, [(a, r, requests, bound) for r in ranks]):
        rows[str(r)] = got
        print(f"[prove-path commit] rank {r}: {json.dumps(got, sort_keys=True, default=str)}", flush=True)
    Path(a.out, "commit.json").write_text(json.dumps({"binding": bound, "ranks": rows}, indent=1, sort_keys=True, default=str))
    return 0


def more_negatives(a: argparse.Namespace, r: int, world: int, compiler: Capped, mine: Any, held: dict, served: Path,
                   opening: dict) -> dict[str, dict]:
    """What the verifier refuses before any draw about the advice: a5's (`a5.advice_negatives`: another compiler, another advice
    root, the opening under another salt), a registration binding no compiler or advice (a window over Programs of record passed
    off as the advice's), one whose compiler names another cap ℓ, and another rank's registration."""
    import a5 as A5

    neg = A5.advice_negatives(argparse.Namespace(**{**vars(a), "other_advice_opening": None, "other_windows": None}), r, compiler,
                              mine, held, served, opening)
    w = json.loads((served / "registration.json").read_text())["window"]
    cases = {"registration-binds-no-compiler-or-advice (source)": {k: v for k, v in w.items() if k not in SRT.BINDING},
             "registration-names-another-cap (source)": {**w, "compiler": {
                 "name": CAPPED, "digest": Capped(compiler.inner, 2 * compiler.cap_bits).digest}}}
    theirs = served.parent / f"rank{(r + 1) % world}" / "registration.json"
    if world > 1 and theirs.exists():
        cases["registration-of-another-rank (source)"] = json.loads(theirs.read_text())["window"]
    for name, x in cases.items():
        v = SRT.source(x, mine, held)
        neg[name] = {"refused": not v["ok"], "codes": v["codes"]}
    return neg


def advice_before_draw(receipt: Mapping[str, Any], rec: Mapping[str, Any], bound: Mapping[str, Any]) -> dict:
    """That the advice was fixed before the coins, checked rather than assumed, as `pouw.credit.draw_root` checks the credit bits:
    the digest of the registration the verifier holds the opening to (`registration.digest`, which covers the window's
    ``compiler`` and ``advice`` root) must be the one the draw's receipt names, the receipt the stream wrote when it received the
    registration and before its select read any coin (a5's ``verifier-coins-before-the-registration``: a select on a
    registration the stream never received reads none); and that registration must bind the compiler and the root the verifier
    opened. -> {ok, codes, detail}: ``after-the-draw`` where the draw followed another registration, ``compiler`` or ``advice``
    where the registration binds another."""
    import a5 as A5

    w = rec.get("window") or {}
    ok = {"after-the-draw": receipt.get("registration") == R.digest(rec) and receipt.get("verifier") == A5.VERIFIER,
          "compiler": w.get("compiler") == bound["compiler"], "advice": w.get("advice") == bound["advice"]}
    return {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v],
            "detail": {"drawn_after": receipt.get("registration"), "registration": R.digest(rec), "received": receipt.get("received")}}


def advice_after_draw(compiler: Capped, receipt: Mapping[str, Any], rec: Mapping[str, Any], opening: Mapping[str, Any], mine: Any,
                      held: dict) -> dict:
    """The negative: once the coins are read, the prover commits the same advice again (fresh salts, so a fresh root) and hands
    in a registration binding that root, with its opening. Every check before the draw passes on it (`source` against the new
    opening: the root is a true commitment of the advice), and `advice_before_draw` refuses it (``after-the-draw``): the draw
    followed the registration received first."""
    data = bytes.fromhex(opening["advice"])
    late = AW.opening(data, AW.commit_advice(compiler, data))
    _reqs, bound = open_advice(compiler, late)
    late_rec = {**rec, "window": {**rec["window"], "advice": bound["advice"]}}
    src = SRT.source(late_rec["window"], mine, {**held, **bound})
    v = advice_before_draw(receipt, late_rec, bound)
    return {"refused": not v["ok"], "codes": v["codes"], "before_draw_checks_pass": src["ok"]}


def forged_row(s: dict, forged: dict | None) -> dict:
    """A forged rank's verdict: its registration passed every check before the draw, the forged unit was drawn, and the outcome is
    a reject with Lean's verifier refusing a drawn member."""
    verdicts = (s.get("audit") or {}).get("verdicts") or {}
    lean = {k: v for k, v in verdicts.items() if k.startswith("lean-verify/")}
    unit = None
    if forged is not None:
        unit = next(m["base"] for m in s.get("members") or [] if m["k"] == forged["member"]) + forged["unit"]
    checks = {"checks_before_draw": bool(s.get("checks_before_draw")) and all(s["checks_before_draw"].values()),
              "forged_unit_drawn": unit is not None and unit in (s.get("draw") or []),
              "rejected": (s.get("audit") or {}).get("accepted") is False,
              "lean_refused": any(v is False for v in lean.values())}
    return {"ok": all(checks.values()), "checks": checks, "why": (s.get("audit") or {}).get("why")}


def audit_rank(job: tuple[argparse.Namespace, int, int]) -> dict:
    import a5 as A5

    a, world, r = job
    t0 = time.time()
    compiler, requests, bound, opening = compiled(a)
    mine, held = AW.window_of_advice(a.tp_commit, r, a.partition, compiler, requests, bound)
    window_s = round(time.time() - t0, 2)
    served, out = Path(a.windows) / f"rank{r}", Path(a.out) / f"rank{r}"
    more = more_negatives(a, r, world, compiler, mine, held, served, opening)
    s, ok = A5.audit(a, A5.template_sets(Path(a.template_sets)), out, mine, served,
                     lambda w: SRT.source(w, mine, held), a.port + A5.RANK_PORTS * r,
                     {"rank": r, "tp": held["tp"], "binding": bound, "requests": requests, "verifier_window_s": window_s}, more, rank=r)
    row = {"ok": ok, "seconds": round(time.time() - t0, 1),
           **{x: s.get(x) for x in ("run_root", "population", "members", "registration", "source", "refused", "checks_before_draw",
                                    "audit", "draw", "draw_counts", "negatives", "coins")}}
    if s.get("draw") is not None:
        rec, receipt = (json.loads((out / f).read_text()) for f in ("registration.json", "receipt.json"))
        row["advice_before_draw"] = advice_before_draw(receipt, rec, bound)
        row["negatives"] = {**(row["negatives"] or {}),
                            "advice-root-after-the-draw": advice_after_draw(compiler, receipt, rec, opening, mine, held)}
        ordered = row["advice_before_draw"]["ok"] and row["negatives"]["advice-root-after-the-draw"]["refused"]
    else:
        ordered = False
    row["ok"] = ok and ordered
    if a.forged:
        forged = ((json.loads((Path(a.windows) / "commit.json").read_text())["ranks"]).get(str(r)) or {}).get("forged")
        row["forged"] = {"unit": forged, **forged_row(s, forged)}
        row["ok"] = row["forged"]["ok"] and ordered
    return row


def audit(a: argparse.Namespace) -> int:
    rec = SRT.record(a.tp_commit)
    world, ranks = ranks_of(a.tp_commit, a.ranks)
    a.coins, a.verifier_coins, a.window_root = None, True, rec["tp_run_root"]
    a.rule = json.loads(Path(a.partition).read_text()).get("law")
    if a.forged:
        os.environ["FC_ZK_SELF_CHECK"] = "skip"
    compiler, requests, bound, _opening = compiled(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"tp_run_root": rec["tp_run_root"], "world": world, "compiler": bound["compiler"], "advice": bound["advice"],
                     "cap_bits": compiler.cap_bits, "inner_compiler": {"name": compiler.inner.name, "digest": compiler.inner.digest},
                     "gap": "the advice is opened to the verifier until the compile unit exists (README.md)",
                     "requests": requests, "rule": a.rule, "delta": a.delta, "forged": a.forged,
                     "coins": {"source": "verifier", "what": "each rank's service.Stream on the verifier's own live coins (the "
                                                             "Lean's IO.getRandomBytes), read once the stream received the rank's "
                                                             "registration"},
                     "ranks": {}}
    for (_a, _w, r), row in in_processes(audit_rank, a.jobs, [(a, world, r) for r in ranks]):
        summary["ranks"][str(r)] = row
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    summary["ok"] = sorted(summary["ranks"]) == sorted(map(str, ranks)) and all(x["ok"] for x in summary["ranks"].values())
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    shown = {r: {x: s.get(x) for x in ("ok", "population", "draw", "draw_counts", "seconds")}
             | {"advice_before_draw": (s.get("advice_before_draw") or {}).get("ok")}
             | ({"forged": s["forged"]} if a.forged else {"accepted": (s.get("audit") or {}).get("accepted"),
                                                         "negatives_refused": all((s.get("negatives") or {"none": False}).values())})
             for r, s in summary["ranks"].items()}
    print("PROVE-PATH " + json.dumps({"tp_run_root": rec["tp_run_root"], "forged": a.forged, "ok": summary["ok"], "ranks": shown},
                                     sort_keys=True, default=str))
    return 0 if summary["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("advice", help="a declared run's I1 advice from its workload file")
    p.add_argument("--workload", required=True)
    p.add_argument("--out", required=True, help="the advice's file: I1's canonical encoding")
    for name in ("commit", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--tp-commit", required=True, help="the TP Commit's record (its --out: runs.jsonl, versions.json)")
        p.add_argument("--build-root", required=True, help="each rank's Programs of record in rank<r>/ (the stand-in compiler's)")
        p.add_argument("--opening", required=True, help="the advice's opening (advice_windows.opening): commit writes it, the "
                       "prover's private file; audit reads it, the named gap")
        p.add_argument("--cap-bits", type=int, default=CAP_BITS, help="ℓ, the public cap on the advice's length in bits, "
                       "registered in the compiler's digest")
        p.add_argument("--partition", required=True, help="the verifier's partition file: {templates, law, scope?}")
        p.add_argument("--ranks", default=None, help="comma-separated (default every rank)")
        p.add_argument("--jobs", type=int, default=1, help="ranks at once, each in its own process")
        p.add_argument("--out", required=True)
    p = sub.choices["commit"]
    p.add_argument("--advice", required=True, help="the advice's file: I1's canonical encoding (the prover's, private)")
    p.add_argument("--forge", help="MEMBER:UNIT: commit that unit with bit 0 of its first output word flipped")
    p.add_argument("--workers", type=int, default=None, help="commit workers per rank (default min(16, CPUs))")
    p = sub.choices["audit"]
    p.add_argument("--windows", required=True, help="the prover's windows, rank<r>/ each (prove_path.py commit --out)")
    p.add_argument("--template-sets", required=True, help="one input set per member template, for the lowering")
    p.add_argument("--m0-pythonpath", required=True)
    p.add_argument("--bin", required=True)
    p.add_argument("--lean", required=True)
    p.add_argument("--firewall", required=True)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--port", type=int, default=7651)
    p.add_argument("--session-timeout", type=float, default=14400)
    p.add_argument("--forged", action="store_true", help="the windows hold a forged unit (commit --forge): every rank must be refused")
    a = ap.parse_args(argv)
    if a.cmd == "advice":
        c = AW.codec_of()
        advice = AW.advice_of_workload(json.loads(Path(a.workload).read_text()), c)
        data = c.encode(advice)
        AW.requests_of(advice)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_bytes(data)
        print(json.dumps({"advice": a.out, "requests": len(advice.requests), "steps": len(advice.steps), "bits": c.total_bits(advice)}))
        return 0
    return commit(a) if a.cmd == "commit" else audit(a)


if __name__ == "__main__":
    raise SystemExit(main())
