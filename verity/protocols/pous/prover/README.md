# PoUS's circuits (`verity.protocols.pous.prover`)

PoUS's Boolean circuits: what a prover proves about a store of P2 labels over committed values. Each module's docstring
is its circuit's specification, with its Definitions' signatures, bit orders and what it computes. PoUS's verifier reads
none of them.

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `circuit.PousModSquare` (`circuit.py`) | `PousModSquare_v1{W, C0}`: `x² mod (2^W − C0)` for every `x < 2^W`, the step P2's decode takes twice a label. Its parts are Definitions too (`PousSquare`, `PousModFold`, `PousToom3Eval`, `PousToom3Interp`, `PousSquareJoin`, `PousAdd`), with the builders they are traced from. `P2_W` and `P2_C0` are P2's point. |
| `decode_circuit.PousP2Decode` (`decode_circuit.py`) | `PousP2Decode_v2{W, C0, L}`: P2 v4's decode of one label, from the label and its expansion base to its payload and one `ok` bit. Its parts are the field steps (`PousP2Sq`, `PousP2Negate`, `PousP2Sigma`, `PousP2Finish`) and counter-mode SHA-512 (`PousSha512Mid`, `PousSha512Head`, `PousSha512Ctr`, `PousSha512CtrChunk`, `PousByteSwap`). |
| `answer_circuit.PousAnswerOpens` (`answer_circuit.py`) | `PousAnswerOpens_v1{N}`: one bit, whether the label the memory challenger received in a round is the label under the store's hiding leaf. Its parts are `PousRowDigest`, `PousRowDigestEnd`, `PousSha512Compress`, `PousHm96Commit` and `PousEqual`. |
| `verdict_circuit.PousVerdict` (`verdict_circuit.py`) | `PousVerdict_v1{K, M, LIMIT}`: the verdict over a session's `K` timed rounds as one proof block, 1 iff at most `M` rounds miss. `gates(K: int, M: int, LIMIT: int) -> tuple[list[Gate], list[int]]` is its gate list, which the Lean restates gate for gate (`PousCheck.Verdict.prog`). `round_verdict`, `round_hit` and `verdict_value(K: int, M: int, LIMIT: int, records: Sequence[bytes], expected: Sequence[bytes]) -> int` are the same verdict on bytes, the reference of the tests. |
| `uv run circuit-check verity/protocols/pous/prover/circuit.circuit-check.py`, or another module's sidecar | circuit-check's report on the Definitions that sidecar's bindings reach. Each of the four modules has its sidecar, and `circuit.circuit-check.json` holds the pin of `PousModSquare`'s binding. |

Read from outside today: circuit-check, through the sidecars; the Lean check's difftest of the verdict
(`verity/protocols/pous/verifier/check/tests/test_verdict_difftest.py`); and the circuits' own tests, which are in
`verity/protocols/pous/verifier/tests/` (this directory has no suite of its own). No served run builds a statement
from these circuits.
