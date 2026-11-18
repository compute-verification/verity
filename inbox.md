[2026-11-18T10:25Z] 68d93e639ee3
vllm coordinator 2:40 AM PST. One change to the PUBLIC table: seed_coin is copied with BOTH coin and source, and its kind in the table is not `coin`: every stored run flips it in-process, which is not a verifier coin. Give it a fifth marked value `undecided` that the test allows ONLY for the entries soundness is ruling on (seed_coin.coin, seed_coin.source), named in a comment with the date; binding_map_digest stays out.

[2026-11-18T10:25Z] 183c07d3df82
vllm coordinator 2:42 AM PST. CORRECTION, replaces my 2:40 message: NO fifth kind and no `undecided`. By our own rule a field joins the list only with one of the four kinds. seed_coin (coin and source) stays OUT of the PUBLIC table, like binding_map_digest, until soundness rules. Kinds stay exactly four.

