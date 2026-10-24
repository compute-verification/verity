"""REDTEAM-ADVICE phase-18 target 2 (#3886): the ChargeLink wiring into the compile slice, and
`forced admissions = code_v1 on all 339 windows` for vLLM's served run 8eb1.

Independent of `benchmarks/network_e2e/encodings/tests/test_served_window.py`: re-derives the 339 windows'
decisions, the charge chain (`charge_chain.session`, ChargeLink_v1 at D=32 under the async-light book), code_v1's
charge of the same decisions, and the forced-admission counts, then probes:

  A. the chain == code_v1 of the decisions, summing to 5,500 bits, forced == [1]*339 (the drained-engine open);
  B. the structural fields: no prefix caching, no cache_hit / resumed / preempted record in 6,169 steps, so the
     residual's only non-decision field that could exist is `order`;
  C. the forced admission is load-bearing-free: charged as a decision (a 1) it changes 330 windows (5,968 bits),
     so a prover can't silently relabel a decision as a forced admission to save bits -- the replay pins `forced`
     and the scan's en = count pins the decisions (compile-slice.md "What binds the charge's inputs");
  D. the `order` channel the compile slice does NOT charge: a window with an arrival-vs-release deviation costs more
     under code_v1's full residual than under the decisions alone, so the decisions-only charge equals the full
     residual only when `order` is empty (8eb1: assumed via release-order serving; not in the shipped fixture).
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENC = HERE / "benchmarks" / "network_e2e" / "encodings"
RUN = ENC / "tests" / "fixtures" / "vllm-8eb1"
ARITH = HERE / "verity" / "core" / "protocols" / "certifier" / "tests" / "fixtures" / "arith_encode_vectors.json"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RP = _load("enc_replay", ENC / "replay.py")
RV = _load("enc_replay_vectors", ENC / "replay_vectors.py")
CV = _load("enc_code_v1", ENC / "code_v1.py")
from verity.core.protocols.certifier import charge_chain as CC  # noqa: E402

D = 32


def async_light_book() -> dict:
    return next(s for s in json.loads(ARITH.read_text())["streams"]
                if s["source"].startswith("async-light"))["decisions"]["book"]


def derive() -> tuple[list[list[int]], list[int], list[list[int]]]:
    """Each of the 339 windows' decisions (`per`), its forced-admission count (`forced`), and the decisions with each
    forced admission spliced in as a 1 at its admission step (`as_ones`)."""
    recs = RP.records(RUN / "advice_trace.jsonl.gz")
    per, forced, as_ones = [], [], []
    for w in range(339):
        ds, rp = RP.decisions(RP.load_step_trace(RV.served_window(recs, w)[0]))
        per.append([int(d.arrived) for d in ds])
        forced.append(sum(rp.forced))
        ones = [(rp.admitted[i], i, 1) for i, f in enumerate(rp.forced) if f]
        as_ones.append([b for *_at, b in sorted([(d.step, d.request, int(d.arrived)) for d in ds] + ones)])
    return per, forced, as_ones


def probe_A_and_C(per, forced, as_ones) -> None:
    book = async_light_book()
    ses = CC.session(D, book, [([1] * len(d) + [0] * (D - len(d)), d + [0] * (D - len(d))) for d in per])
    chain = [s[0] for s in ses]
    lens = list(CV.encode([CV.Window(arrived=d) for d in per], CV.Book.from_json(book)).lens)
    other = list(CV.encode([CV.Window(arrived=d) for d in as_ones], CV.Book.from_json(book)).lens)
    print("A. chain(ChargeLink_v1,D=32) == code_v1-of-decisions per window:", chain == lens)
    print("   sum of code_v1-of-decisions charges:", sum(lens), "(expected 5500)")
    print("   forced admissions per window all exactly one:", forced == [1] * 339)
    print("C. forced admission charged as a decision (a 1):")
    print("   windows whose charge changes:", sum(a != b for a, b in zip(other, lens)), "(expected 330)")
    print("   total then:", sum(other), "(expected 5968, i.e. +468 bits the design does not charge)")
    assert chain == lens and sum(lens) == 5500 and forced == [1] * 339
    assert (sum(a != b for a, b in zip(other, lens)), sum(other)) == (330, 5968)


def probe_B() -> None:
    recs = [json.loads(l) for l in gzip.decompress((RUN / "advice_trace.jsonl.gz").read_bytes()).decode().splitlines()]
    hdr = recs[0]["config"]
    steps = [r for r in recs if r.get("kind") == "step"]
    hits = sum(1 for r in steps if r.get("cache_hit"))
    resumed = sum(1 for r in steps if r.get("resumed"))
    preempted = sum(1 for r in steps if r.get("preempted"))
    print("B. enable_prefix_caching:", hdr.get("enable_prefix_caching"),
          "| cache_hit records:", hits, "| resumed:", resumed, "| preempted:", preempted,
          "| steps:", len(steps))
    assert hdr.get("enable_prefix_caching") is False and (hits, resumed, preempted) == (0, 0, 0)


def probe_D() -> None:
    """A window of three requests released 0<1<2 but taken 0,2,1 (one deviation). code_v1's full residual charges the
    order entry; the decisions alone do not. The compile slice commits only the decisions (ChargeLink_v1{D}), so its
    charge equals the full residual only where `order` is empty."""
    book = CV.Book.uniform(CV.Caps(order_len=1))
    dec = [1, 1, 1]
    plain = CV.Window(arrived=dec)
    with_order = CV.Window(arrived=dec, order=((2, (1,)),))  # arrival at position 2 ranked 1 among those left
    bits_plain = CV.encode([plain], book).lens[0]
    bits_order = CV.encode([with_order], book).lens[0]
    print("D. one window, decisions [1,1,1]:")
    print("   code_v1 charge, decisions only :", bits_plain)
    print("   code_v1 charge, one order entry:", bits_order, "(+%d bits)" % (bits_order - bits_plain))
    print("   -> the compile slice commits only the decisions, so it charges", bits_plain,
          "for this window; the order bits are the I/O tie's (PromptTie.order) or zero under release-order serving")
    assert bits_order > bits_plain


def main() -> int:
    per, forced, as_ones = derive()
    probe_A_and_C(per, forced, as_ones)
    probe_B()
    probe_D()
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
