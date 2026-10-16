import Proofs.Flock.Claims.Complete.Json.Spec
import Proofs.Flock.Claims.Complete.Json.Hex
import Proofs.Flock.Claims.Complete.Json.Record

/-! The session record's JSON round trip: what it needs from the JSON library (`Spec`: `JsonSpec`, `Same`, `Good`), the
hex round trips (`Hex`: `unhex_hexOf`, `f128OfHex_toHex`), and the honest record as a JSON value with the decoder's reads
through `Same` (`Record`: `recordObj`, `good_mkObj`, `str_same`, …). -/
