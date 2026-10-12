# b-verifier: where PROTOCOL.md, the Python B-Ligero code and the independent Rust verifier disagree

Lane `b-verifier` (track B), 2026-09-22.  The verifier is `archive/ligero-verify/` (`verity-ligero-verify`, no
dependencies: its own BabyBear, NTT / coset code, BLAKE3, BLAKE2b-256, SHAKE-256, `ligero-*/v1` parsers, and the
proximity / linear / quadratic / chain tests), written from `archive/direct/ligero/PROTOCOL.md` + `HANDOFF.md` +
`note:r20-proof/red-team-vu/20260922T1042Z-report-redteam-vu` F1-F3, with the Python sources read as the reference of record wherever the document is silent.
Every such place is an entry below.  Statuses follow `verity/ml/circuits/silicon/discrepancy_log.json` (`open` / `resolved` /
`excluded`); each entry also carries a structured block (`id`, `issue`, `document`, `code`, `verifier`, `status`,
`owner`, `evidence`) so the coordinator can lift it into the campaign log.

**Post b-zk-fix (lane `b-verifier-2`, `main` a52441c).**  The protocol changed under the verifier (PROTOCOL.md 8b/8c:
masked `q`, coin-only challenges in `--mode interactive`, `t = 288 / t_pad = 512 / D = 7` in `--mode fiat-shamir`);
the verifier was re-derived from the new text and re-gated on fresh proof sets in all four modes (ZK / non-ZK x
interactive / Fiat-Shamir).  D1 re-checked (still open, both modes), D3 resolved by the fix, D4 superseded by **D7**
(an interactive-mode transcript is not transferable evidence: 49/49 forged files for false claims accepted by both
file verifiers), D5 extended with the new byte facts, D8 (named-reason differences) added.  On the decision workload
(4096 VUs, `l = 16384`, 25 sub-batches) the two verifiers agree on every file in every mode
(`note:r20-proof/b-verifier/20260922T1132Z-report-b-verifier`, "post b-zk-fix").

## Entries

Each entry is a file under `discrepancies/`, with its structured block; a new entry is a new
`discrepancies/d<n>.md` and a line here.

- [D1](discrepancies/d1.md), resolved: The Python verifier never checks that its parameters meet the target; at small `l` non-ZK they do not
- [D2](discrepancies/d2.md), open: The statement digest binds a Python `repr(float)`
- [D3](discrepancies/d3.md), resolved (b-zk-fix): HVZK-mode proofs carry coin commitments and openings that nothing binds
- [D4](discrepancies/d4.md), excluded (superseded by D7): The Python verifier is interactive; a file verifier checks a Fiat-Shamir transcript
- [D5](discrepancies/d5.md), open: Things the document does not say that a verifier needs (taken from the code, now written down)
- [D6](discrepancies/d6.md), resolved: A transcript mutation never reaches the tests
- [D7](discrepancies/d7.md), open: An interactive-mode transcript is not transferable evidence (post b-zk-fix; the file verifier's verdict is conditional)
- [D8](discrepancies/d8.md), excluded: Named-reason differences that are not verdict differences (post b-zk-fix)
- [D9](discrepancies/d9.md), open: `authentication = included` (lane `auth-included`): a third hash, two new file versions, and the check order
- [D10](discrepancies/d10.md), resolved (b-batch-bound): No verifier enforced the union bound across sub-batches
- [D11](discrepancies/d11.md), open: Relations beyond the Ampere BF16 unit (lane `ligero-verify-relations`): what a statement does not say, and the per-relation pin
- [D12](discrepancies/d12.md), open (finding): The v2 public-selection relations (lane `relmin-lookup`): the verifier computes part of the step, and the pins are no longer the operand words
- [D13](discrepancies/d13.md), open: `authentication = included-hash` (lane `hash-relation`): the operands leave the statement, Poseidon2 enters the relation
- [D14](discrepancies/d14.md), open: The v3 private-operand relations (lane `relmin-private`): the pins are the operand words again, packed for FP8, and the verifier computes nothing
- [D15](discrepancies/d15.md), open: Folded (x4) private-operand-safe relations (lane `fold-private`): same VU claim, four steps per column, K = 4k
