module

public import Proofs.Flock.Claims.Complete.Json.Spec
public import Proofs.Flock.Claims.Complete.Json.Canon
public import Proofs.Flock.Claims.Complete.Json.Hex
public import Proofs.Flock.Claims.Complete.Json.Record
public import Proofs.Flock.Claims.Complete.Json.Decode
public import Proofs.Flock.Claims.Complete.Json.Session
public import Proofs.Flock.Claims.Complete.Json.Live
public import Proofs.Flock.Claims.Complete.Json.Stmt
public import Proofs.Flock.Claims.Complete.Json.Goals

/-! The session record's JSON round trip: the lemmas on alike values it reads (`Spec`'s lemmas on `Same`), the hex round trips (`Hex`: `unhex_hexOf`, `f128OfHex_toHex`), and the honest record as a JSON value with
the decoder's reads through `Same` (`Record`: `recordObj`, `good_mkObj`, `str_same`, …), and the decoder's checks
S1–S14 on it (`Decode`: `decode_recordObj`), what the one-table session returns and sends (`Session`: `recordJson_eq`,
`sessionP_facts`, with a rep's rounds its sends, `repP_rec`), the record's live session and sample (`Live`:
`live_recordObj`, `recordDraw_recordObj`), what the record needs of the statement, from its setup (`Stmt`: `stmtOk`),
and B's record checks at the honest run under named hypotheses (`Goals`: `Run.json`, `decodes_goal`, …,
`custodyRounds_goal`). -/

@[expose] public section
