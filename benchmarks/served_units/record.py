"""record.py OUT [MODEL]: each proof block of OUT/summary.json whose Lean checks all gave the verdict they must writes
`circuits/proof-block-prove-seconds` and `circuits/proof-block-peak-memory` into the run's result."""
import json
import re
import sys
from pathlib import Path

from verity_bench import kit

PROVE, PEAK = "circuits/proof-block-prove-seconds", "circuits/proof-block-peak-memory"


def machine(card: str) -> str:
    return "cpu" if card == "none" else kit.card_short(card)


def member(model: str, tag: str, machine: str) -> str:
    if model == "synthetic":
        return f"synthetic-{tag}-{machine}"
    return "-".join([model, *([] if machine == "cpu" else [machine]), re.sub(r"[^a-z0-9]+", "-", tag.lower()).strip("-")])


def main(out: Path, model: str = "synthetic") -> None:
    rows = json.loads((out / "summary.json").read_text())
    w = kit.writer(out)
    for r in rows:
        zk = r.get("zk") or {}
        if not (r["as_expected"] and zk.get("prove_total_s_median") and zk.get("prover_maxrss_gb")):
            continue
        fields = {"model": model, "block": r["definition"], "machine": machine(w.stamp["card"])}
        for name, value, unit in ((PROVE, zk["prove_total_s_median"], "seconds"), (PEAK, zk["prover_maxrss_gb"], "GiB")):
            m = w.measurement(name, value, unit, mark="measured", member=member(model, r["tag"], fields["machine"]), extra=fields)
            print(json.dumps({k: m[k] for k in ("name", "value", "unit", "member", "model", "block", "machine")}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), *sys.argv[2:3])
