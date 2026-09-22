"""The output-freedom certificate ``U`` of a CORE run, through the SHIPPED calculator ``veritor.analysis.certified.certified_bound``.

The handoff: "publish the actually computed U even if it equals the tiny circuit's full output width".  The summary Theorem 6.3 reads
(``veritor.analysis.certified.Summary``) is built HERE from the partition (labelled ASSUMED where it is a modelling choice, not a derivation
by ``summary_from_table`` from a compiled kind table):

* families: one per VU kind, ``n_k`` = its VU count, ``w_k`` = 16 bits (every VU of this circuit outputs exactly one bf16 word);
* banks: one bank of ``N_R`` replay units, ``v_a = 2N`` fallible VUs each, exit cut ``c_a = 16·N`` bits (the RU's output tuple ``h_r``);
* ``W_out = 16·N`` bits (the designated output ``h_R``), the cap;
* θ = (q, s, δ = ``security.delta`` of the instance, B_ex).

``python -m e2e.demo_core.certificate --run <archive dir>`` reads ``run_config.json`` + ``protocol_instance.json`` of an archived run and
writes ``certificate_U.json`` next to them; ``--table`` prints every archived run under ``out/eod/e8/runs``.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO), str(_REPO / "src"), str(_REPO / "vllm-poc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from veritor.analysis.certified import Bank, Family, Summary, certified_bound  # noqa: E402
from veritor.core.policy import VerificationPolicy  # noqa: E402

from . import circuit as C  # noqa: E402

__all__ = ["certificate", "summary_of"]

WORD_BITS = 16


def summary_of(part: C.Partition) -> Summary:
    fams = tuple(Family(kind, int(n), WORD_BITS) for kind, n in sorted(part.kind_counts().items()))
    banks = (Bank("DemoCoreStage", part.n_r, 2 * part.N, WORD_BITS * part.N),)
    return Summary(fams, banks, WORD_BITS * part.N)


def certificate(part: C.Partition, *, q: str | Fraction, s: str | Fraction, B_ex: int, delta: str | Fraction) -> dict[str, Any]:
    summ = summary_of(part)
    pol = VerificationPolicy(q=Fraction(q), s=Fraction(s))
    cb = certified_bound(summ, pol, Fraction(delta), allowance=int(B_ex))
    W = cb.out_width
    return {
        "schema": "veritor/demo-core/certificate-U/v1",
        "calculator": "veritor.analysis.certified.certified_bound (SHIPPED; Appendix A Bound(C, S, theta))",
        "label": "MODELED/ANALYTICAL: certified arithmetic of the shipped calculator over a summary built from the partition (see summary.label)",
        "theta": {"q": str(pol.q), "s": str(pol.s), "delta": str(Fraction(delta)), "B_ex": int(B_ex), "B_ex_unit": "replay_units"},
        "summary": {"families": [{"kind": f.kind, "n_k": f.count, "w_k_bits": f.width} for f in summ.families],
                    "banks": [{"kind": b.kind, "B_a": b.copies, "v_a": b.units, "c_a_bits": b.exit_width} for b in summ.banks],
                    "W_out_bits": W, "N_V": summ.N_V, "N_R": summ.N_R,
                    "label": "ASSUMED: family width = one bf16 output word (16 bits) per VU; bank = one RU of 2N fallible VUs behind its N-word output "
                             "tuple (16N bits); W = the designated output h_R (16N bits). Built from demo_core.Partition, NOT by summary_from_table."},
        "U_bits": int(cb.bits), "U_equals_W": bool(cb.bits >= W), "precap_bits": cb.precap_bits, "W_bits": W, "capped": bool(cb.capped),
        "branch": cb.status.branch, "winner": {"origin": cb.status.winner.origin, "value_bits": float(cb.status.winner.value)},
        "reading": ("U = 0: q = s = 1 with B_ex = 0 -- every VU's relation is proved; nothing about the output is the prover's choice (the endpoint branch)"
                    if cb.bits == 0 else
                    f"U = W = {W} bits: the pre-cap certified minimum ({cb.precap_bits} bits) exceeds the declared output domain, so the certificate "
                    "is the trivial one -- a 2N·R-VU circuit at q, s in (0, 1) demonstrates the protocol's mechanics, not a nontrivial output-freedom bound"
                    if cb.bits >= W else f"U = {cb.bits} bits of W = {W}"),
    }


def for_run(run: Path) -> dict[str, Any]:
    rc = json.loads((run / "run_config.json").read_text())
    inst = json.loads((run / "protocol_instance.json").read_text())
    part = C.partition(int(rc["N"]), int(rc["R"]))
    delta = inst["security"]["delta"]
    doc = certificate(part, q=rc["q"], s=rc["s"], B_ex=int(rc["B_ex"]), delta=delta)
    doc["run"] = {"name": rc.get("name", run.name), "program_digest": part.program_digest, "partition_digest": part.digest,
                  "instance_digest": (run / "instance" / "instance_digest.txt").read_text().strip() if (run / "instance" / "instance_digest.txt").exists() else None}
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", default=[], help="archived run directory (writes certificate_U.json into it)")
    ap.add_argument("--table", action="store_true", help="every run under out/eod/e8/runs")
    ap.add_argument("--write", action="store_true", help="write certificate_U.json into each run directory")
    a = ap.parse_args(argv)
    runs = [Path(r) for r in a.run]
    if a.table:
        runs += sorted((_REPO / "out" / "eod" / "e8" / "runs").glob("CORE-*"))
    for run in runs:
        doc = for_run(run)
        if a.write:
            (run / "certificate_U.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        t = doc["theta"]
        print(f"{doc['run']['name']:<22} q={t['q']:<4} s={t['s']:<4} B_ex={t['B_ex']}  N_V={doc['summary']['N_V']:<4} N_R={doc['summary']['N_R']:<3} "
              f"U = {doc['U_bits']} bits (W = {doc['W_bits']}; precap {doc['precap_bits']}; {doc['branch']}/{doc['winner']['origin']})"
              f"{'  [U = W]' if doc['U_equals_W'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
