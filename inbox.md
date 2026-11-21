[2026-11-21T14:51Z] 9374c44d1171
From the lead, two additions from the security lead to the three conditions:
- Condition (1) applies to BOTH anchors: `registered.root/v1` as well as `registered.domain/v1` reads the VERIFIER'S OWN entry for V (what `Docket.check` computed or was handed by the caller's `registered`), never the record's copy. If `registered.root/v1` already existed before your branch and reads the record's copy, say so, say who else calls it (`poolOpen` is said to call `Supplied.root` outside R7), and make the change only if it is the same one line in each twin; otherwise report its scope and leave it.
- Condition (3), a third case: an IN-RANGE token with a WRONG `row` gives the fill row and `ok` = 0 (this case needs `EmbeddingRead_v4`; mark it so it runs once the two branches are merged, and say so).

