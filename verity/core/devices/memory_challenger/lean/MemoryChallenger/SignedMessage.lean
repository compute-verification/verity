import Lms.Verify
import MemoryChallenger.Records

/-!
# The message a memory challenger's session record signs

`lms.memory_challenger_message(I, session, record)`, PoUS's role of the LMS key: `SHA-512([tag] ‖ I ‖ [[session]] ‖
[record])`, each `[·]` a u64 big-endian length then the bytes, the tag `verity/memory-challenger-record/v1`. The context
is the session id framed once, and framed again as the context. `Lms.Verify` checks a signature over it, and the
tests sign it with `Lms.signAt`.
-/

namespace MemoryChallenger

open Flock

/-- `lms._field`: a u64 big-endian length, then the bytes. -/
def field (b : ByteArray) : ByteArray := be64 b.size.toUInt64 ++ b

def MEMORY_CHALLENGER_TAG : String := "verity/memory-challenger-record/v1"

/-- `memory_challenger_message(I, session, record)`. -/
def memoryChallengerMessage (I session record : ByteArray) : E ByteArray := do
  if I.size ≠ 16 then throw s!"an LMS identifier is 16 bytes, not {I.size}"
  pure (Sha512.hash (field MEMORY_CHALLENGER_TAG.toUTF8 ++ I ++ field (field session) ++ field record))

end MemoryChallenger
