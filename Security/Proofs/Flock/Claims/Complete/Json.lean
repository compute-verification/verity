import Proofs.Flock.Claims.Complete.Json.Spec
import Proofs.Flock.Claims.Complete.Json.Hex
import Proofs.Flock.Claims.Complete.Json.Record
import Proofs.Flock.Claims.Complete.Json.Decode
import Proofs.Flock.Claims.Complete.Json.Session
import Proofs.Flock.Claims.Complete.Json.Live

/-! The session record's JSON round trip: what it needs from the JSON library (`Spec`: `JsonSpec`, `Same`, `Good`), the
hex round trips (`Hex`: `unhex_hexOf`, `f128OfHex_toHex`), and the honest record as a JSON value with the decoder's reads
through `Same` (`Record`: `recordObj`, `good_mkObj`, `str_same`, …), and the decoder's checks S1–S14 on it (`Decode`:
`decode_recordObj`), what the one-table session returns and sends first (`Session`: `recordJson_eq`,
`sessionP_facts`), and the record's live session and draw (`Live`: `live_recordObj`, `recordDraw_recordObj`). -/
