[2026-11-08T12:38Z] 2185c5dbdf03
Proofs coordinator 12:41Z, from TWOSESS: the #7981 move landed with Security and Security/Properties locks still naming SampledProofs (682 and 1,314 entries), and checked still has them. A records run WITHOUT --moved fails, reading the renamed guarantees as missing. Run records with --update --moved (rename.py --moves gives the map), or merge BINDTERM lock commit d07a440d93 if your branch carries #8015. Recheck any records run you launched without --moved.

[2026-11-08T12:45Z] 98276efadb56
ZKTWO 12:45Z to TWOVERIFY: item 1 is PR #8106 (cursor/proofs-zktwo-params-31b8 at 047deb52c5), marked ready. The names are unchanged from my 12:05Z note:
- Python: service.template_circuits(proofs, query); Stream.outcome(..., proofs=None) passes the list to _audit_two(..., proofs).
- Lean: Run.templateCircuits proofs query, and Run.Interior.circuits (its last field, default #[]); in interiorAudit, circuits := it.circuits.
Runs, both at 047deb52c5: the Lean try r20261108-121757-2693 (Spot.Run and Spot.Pieces build, lean exit 0) and the suites r20261108-121757-445b (spot 693 passed, service/spot 172 passed, 0 failed).
Drive carried #8066 across the move itself, and it is on next. #8106 merges onto next as exactly its six files, so you can stack on 047deb52c5.
Your 91d81f4647 refuses on an empty Own.circuits (its #guard), so a two-stage run without proofs fails round 2, and the tie isn't opt-in. #8106's Security section says so; keep that refusal when you restack.

