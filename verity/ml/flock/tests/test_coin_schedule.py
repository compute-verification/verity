"""M2's committed coin schedule is one schedule on both sides: the prover's `coin_spec_of` (flock-circuit) and the Lean
verifier of record's `Flock.Zk.coinSpec` pin the same rounds per stream and coins per round, and the verifier's own `Hello`
refuses any other, so a change on one side alone refuses every honest --zk session. 2^9 rounds: the live PoUW unit
(`PouwLiveSlots_v2`) needs more than 256 (a run at 2^8 refused every honest window)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PROVER = ROOT / "verity/ml/flock/live/src/bin/flock-circuit.rs"
VERIFIER = ROOT / "verity/core/service/flock/Flock/Zk.lean"

ROUNDS_LOG, BLOCK = 9, 512


def test_prover_and_verifier_pin_one_coin_schedule():
    rs = re.search(r"fn coin_spec_of\(.*?Spec \{ streams, rounds_log: (\d+), block: (\d+) \}", PROVER.read_text(), re.S)
    lean = re.search(r"def coinSpec \(tables : Array String\) : CoinTree\.Spec :=\n.*?\), (\d+), (\d+)⟩", VERIFIER.read_text(),
                     re.S)
    assert rs and lean, "coin_spec_of or Flock.Zk.coinSpec changed shape: update this test's patterns"
    assert (int(rs[1]), int(rs[2])) == (int(lean[1]), int(lean[2])) == (ROUNDS_LOG, BLOCK)
