"""The lead's device gate: its `GateRecord` from one run of run.sh (CHECK=1), and whether serving admits it.

  uv run python benchmarks/pouw/np_kernel/gate.py OUT --run RUN [--json] [--write]
  uv run python benchmarks/pouw/np_kernel/gate.py --recorded [--json]

``--write`` records an admitting gate as the lead's (``gate.json``: the variant, the files it was built from with their
sha256, and the GateRecord). ``--recorded`` reads that record and names the variant by this tree's copies of those files,
so a source changed since the run names another variant, which the recorded gate doesn't admit.

OUT is the run's out directory. The variant is named by the tree's files the build read (``sources.sha256``, each still
this tree's, or the gate refuses: the record would name other sources) under `serving.LEAD_CONFIG`, and the binary by
the SASS gate's record (``sass_gate.json``), which gives ``sass_ok`` and the pinned toolkit. ``ssn check``'s lines
(``check.jsonl``) on the lead's layout give the rest:

- ``exact``: every honest line accepted (every exported word, final output and formed register equal to the host
  reference, and every served output with |p| within the line's ``fold_bound`` equal to stock's epilogue), and the lines
  agree on that bound, which becomes the record's ``fold_bound``;
- ``poisoned_accepted``: every honest line read back buffers filled with 0xA5 before the call;
- ``controls_rejected``: there is a no-write control (nothing formed, no GEMM) and each was rejected;
- ``schedules_equal``: the lead runs the serial schedule alone, so there is no other schedule to hold to its bytes;
- ``shapes``: the (n, K) of each accepted honest line with a served output checked (extreme data puts every output past
  the bound, unchecked).

Prints the variant, the record and `serving.variant_refusals` at Llama-3.2-1B's fused linears under the serial schedule;
exits 0 only when they are empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
#: ssn check's name of the lead's layout (layout.h's layout_name under BLC=2, ROT=1, FOLD=1)
LEAD_LAYOUT = "blc4a_t64_e7_s1_rot_fold"
#: Llama-3.2-1B's fused linears, (n, K): qkv, o, gate_up, down (ssn.cu's time_all and check)
SERVED = ((3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192))
#: the gate's own code: this file, ssn.cu (its check, which the source digest also covers), the SASS gate and its pins
GATE_CODE = (HERE / "gate.py", HERE / "ssn.cu", ROOT / "benchmarks/pouw/harness/sass_gate.py",
             ROOT / "benchmarks/pouw/harness/sass_pins.json")
#: the lead's GateRecord of record: the last run that admitted it (``--write``)
RECORD = HERE / "gate.json"


class GateError(Exception):
    pass


def _listed(out: Path) -> list[tuple[str, str]]:
    return [tuple(line.split(None, 1)) for line in (out / "sources.sha256").read_text().splitlines() if line.strip()]


def sources(out: Path, root: Path = ROOT) -> list[Path]:
    """The files sources.sha256 names, each checked against the tree at `root`."""
    listed = _listed(out)
    if not listed:
        raise GateError(f"{out / 'sources.sha256'} names no file")
    stale = [p for sha, p in listed if not (root / p).is_file() or hashlib.sha256((root / p).read_bytes()).hexdigest() != sha]
    if stale:
        raise GateError(f"the run built from other sources than this tree's: {', '.join(stale)} differ (gate the run's "
                        "own commit)")
    return [Path(p) for _, p in listed]


def binary(out: Path) -> tuple[str, bool, str]:
    """The SASS gate's record of the one binary: its sha256, its verdict and the pinned toolkit of its cubins."""
    rec = json.loads((out / "sass_gate.json").read_text())
    bins = rec.get("binaries") or []
    if len(bins) != 1:
        raise GateError(f"sass_gate.json holds {len(bins)} binaries, not run.sh's one")
    b = bins[0]
    tks = sorted({c.get("toolkit") or "unpinned" for c in b.get("cubins", []) if c.get("functions")})
    return b["sha256"], bool(rec.get("ok") and b.get("ok")), ",".join(tks) or "unpinned"


def lead_lines(out: Path) -> list[dict]:
    lines = [json.loads(s) for s in (out / "check.jsonl").read_text().splitlines() if s.startswith("{")]
    return [x for x in lines if x.get("check") == LEAD_LAYOUT]


def gate_version() -> str:
    h = hashlib.sha256()
    for p in GATE_CODE:
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return f"np-kernel-gate/{h.hexdigest()[:12]}"


def _ledger():
    if str(HERE.parent / "harness") not in sys.path:
        sys.path.insert(0, str(HERE.parent / "harness"))
    import ledger
    return ledger


def record(out: Path, run: str, root: Path = ROOT):
    """The lead's variant and its GateRecord from the run's out directory, its sources checked against the tree at
    `root`."""
    from verity.core.protocols.pouw.serving import LEAD_CONFIG, GateRecord, lead_variant

    ledger = _ledger()
    files = sources(out, root)
    sha, sass_ok, toolkit = binary(out)
    v = lead_variant(ledger.source_sha256(files, root, dict(LEAD_CONFIG)), sha, "np-kernel")
    lines = lead_lines(out)
    honest = [x for x in lines if x.get("control") is None]
    controls = [x for x in lines if x.get("control") == "no-write"]
    bounds = {x.get("fold_bound") for x in honest}
    exact = bool(honest) and len(bounds) == 1 and not any(x["rejected"] for x in honest)
    g = GateRecord(variant=v.id, run=run, gates=gate_version(),
                   shapes=tuple(sorted({(x["call_n"], x["K"]) for x in honest
                                        if not x["rejected"] and x["served_outputs"] > 0})),
                   exact=exact, poisoned_accepted=bool(honest) and all(x.get("poison") == "0xA5" for x in honest),
                   controls_rejected=bool(controls) and all(x["rejected"] for x in controls),
                   schedules_equal=v.schedules == {"serial"}, sass_ok=sass_ok, toolkit=toolkit,
                   fold_bound=bounds.pop() if len(bounds) == 1 else None)
    return v, g


def recorded(root: Path = ROOT, path: Path = RECORD):
    """The lead's variant as the tree at `root` names it, from the files `path`'s record lists, and the recorded
    GateRecord."""
    from verity.core.protocols.pouw.serving import LEAD_CONFIG, GateRecord, lead_variant

    rec = json.loads(path.read_text())
    g = GateRecord(**{**rec["gate"], "shapes": tuple(tuple(s) for s in rec["gate"]["shapes"])})
    v = lead_variant(_ledger().source_sha256([Path(p) for p in rec["sources"]], root, dict(LEAD_CONFIG)),
                     rec["binary_sha256"], "np-kernel")
    return v, g


def main(argv: list[str] | None = None, root: Path = ROOT, path: Path = RECORD) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, nargs="?", help="run.sh's out directory")
    ap.add_argument("--run", help="the run's id")
    ap.add_argument("--json", action="store_true", help="one JSON object in place of the lines")
    ap.add_argument("--write", action="store_true", help=f"record an admitting gate as the lead's ({RECORD.name})")
    ap.add_argument("--recorded", action="store_true", help=f"the recorded gate ({RECORD.name}) against this tree")
    a = ap.parse_args(argv)
    if a.recorded == bool(a.out and a.run) or (a.recorded and a.write):
        ap.error("give OUT --run RUN [--write], or --recorded")
    from dataclasses import asdict

    from verity.core.protocols.pouw.serving import POUW_V2, SERIAL, variant_refusals
    try:
        v, g = recorded(root, path) if a.recorded else record(a.out, a.run, root)
    except (GateError, OSError, KeyError, ValueError, TypeError) as ex:
        print(f"gate: {ex}", file=sys.stderr)
        return 2
    refusals = variant_refusals(v, POUW_V2, SERIAL.name, SERVED, g)
    if a.write and not refusals:
        path.write_text(json.dumps({"variant": v.id, "source_sha256": v.source_sha256, "binary_sha256": v.binary_sha256,
                                    "sources": {p: sha for sha, p in _listed(a.out)}, "gate": asdict(g)}, indent=1) + "\n")
    if a.json:
        print(json.dumps({"variant": v.id, "source_sha256": v.source_sha256, "binary_sha256": v.binary_sha256,
                          "gate": asdict(g), "refusals": refusals}, indent=1))
    else:
        print(f"variant {v.id} (binary {v.binary_sha256[:12]})")
        print(f"gate {json.dumps(asdict(g))}")
        print("\n".join(f"refused: {r}" for r in refusals) or "admitted")
    return 0 if not refusals else 1


if __name__ == "__main__":
    sys.exit(main())
