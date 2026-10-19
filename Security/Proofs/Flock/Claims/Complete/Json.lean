import Proofs.Flock.Claims.Complete.Json.Spec
import Proofs.Flock.Claims.Complete.Json.Canon
import Proofs.Flock.Claims.Complete.Json.Hex
import Proofs.Flock.Claims.Complete.Json.Record
import Proofs.Flock.Claims.Complete.Json.Decode
import Proofs.Flock.Claims.Complete.Json.Session
import Proofs.Flock.Claims.Complete.Json.Live
import Proofs.Flock.Claims.Complete.Json.Stmt
import Proofs.Flock.Claims.Complete.Json.Goals

/-! The session record's JSON round trip: what it needs from the JSON library (`JsonSpec`, with `Spec`'s lemmas
on `Same`), the hex round trips (`Hex`: `unhex_hexOf`, `f128OfHex_toHex`), and the honest record as a JSON value with
the decoder's reads through `Same` (`Record`: `recordObj`, `good_mkObj`, `str_same`, …), and the decoder's checks
S1–S14 on it (`Decode`: `decode_recordObj`), what the one-table session returns and sends (`Session`: `recordJson_eq`,
`sessionP_facts`, with a rep's rounds its sends, `repP_rec`), the record's live session and draw (`Live`:
`live_recordObj`, `recordDraw_recordObj`), what the record needs of the statement, from its setup (`Stmt`: `stmtOk`),
and B's record checks at the honest run under named hypotheses (`Goals`: `Run.json`, `decodes_goal`, …,
`custodyRounds_goal`). -/
