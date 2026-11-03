"""`flock-circuit selftest --record-dir DIR` (M0 at bb9a2a90) as replay sessions for `agree.py`.

    python selftest_records.py RECORD_DIR OUT_DIR

Each `DIR/<case>.json` (`case`, `expect`, `accepted`, `why`, `record`) and its proofs `DIR/<case>.rep<r>.proof` become
`OUT_DIR/<case>/session.json` and `circuit.rep<r>.bin`; a case whose session stopped before a proof (refused at `Hello`) gets
empty proof files, so both verifiers judge its record. A case whose statement is false (`DIR/<case>.pub.bin`, written by the
967b8d06 vectors patch) gets it as its own `pub.bin`, which both verifiers read. Each case's `case.json` carries its expected verdict and the live
verdict upstream's own verifier gave (as the vectors patch's forgeries do), which `agree.py` reads.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_text import session_text  # noqa: E402


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    n = 0
    for f in sorted(src.glob("*.json")):
        c = json.loads(f.read_text())
        d = out / c["case"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "session.json").write_text(session_text(c["record"]))
        for r in (0, 1):
            p = src / f"{c['case']}.rep{r}.proof"
            (d / f"circuit.rep{r}.bin").write_bytes(p.read_bytes() if p.exists() else b"")
        false_statement = src / f"{c['case']}.pub.bin"
        if false_statement.exists():
            (d / "pub.bin").write_bytes(false_statement.read_bytes())
        (d / "case.json").write_text(json.dumps({"case": c["case"], "expect": c["expect"],
                                                 "live": {"accepted": c["accepted"], "why": c["why"]}}, indent=1))
        n += 1
    print(json.dumps({"cases": n}))


if __name__ == "__main__":
    main()
