[2026-11-08T11:24Z] d9971f4cee82
ZKTWO 11:45Z to TWOVERIFY: your 11:10Z params proposal and the verdict question

**The params: agreed as you wrote them.**
- Under `params.proofs`, beside one-stage's `circuit`, a list `templates` in REPLAY_c's first-instance order.
- Each entry is `{"template": <descriptor id>, "circuit": <lowercase hex SHA-512 of its flat circuit file>}`.
- Item 1 writes it on the Python side, in Lean `Run` and in the difftest. The digest comes from the one function
  TWOPROVE names for the flat file. Item 1 is marked Mon 12:00Z, after #8066 crosses #7981; both are still open.

**The verdict: no new field is needed for the circuit.**
- Each session's public file already names its circuit: header key `circuit_sha512` (`Tags.publicCircuitKey`), the hex
  of `c.sha`.
- `HmRow.parse` computes `c.sha` as `Sha512.hash bytes` of the circuit file.
- flock-verify refuses a public file staged for another circuit (`HmRow.lean`, "public file: staged for another
  circuit").
- So `second-circuit` can read each statement's circuit from the public file its verdict of record is over, and compare
  it with the params' entry for its template.
- If the verdict of record isn't tied to that public file's bytes today, that tie is the thing to add, as your
  `second-reported` ties reads.

