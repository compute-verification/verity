import PousCheck.Randomness

/-!
# The audit's challenges

`p2/PROTOCOL.md` §7 (`reference.audit_id`, `audit_key`, `round_key`): an audit's id and, round by round, the block index and
nonce a challenge must carry. In mode 0 the verifier's secret `source` keys the audit; in mode 1 round `j`'s 32-byte
beacon value keys round `j` and is also its nonce. Every key is bound to the epoch and the verification root, so no
challenge exists before setup has committed to both.
-/

namespace PousCheck.Challenge

open Randomness

def AUDIT_ID_DOMAIN := "verity/pous/p2-audit-id/v4"
def CHALLENGE_DOMAIN := "verity/pous/p2-audit-challenge/v4"
def ROUND_DOMAIN := "verity/pous/p2-audit-round/v4"
/-- A nonce's length, bytes. -/
def NONCE := 32

/-- `derive(source, AUDIT_ID_DOMAIN, {counter, epoch}).stream("id", 32)`: the `counter`-th audit's id in `epoch`. -/
def auditId (source epoch : ByteArray) (counter : Nat) : ByteArray :=
  stream (derive source AUDIT_ID_DOMAIN [("counter", .int counter), ("epoch", .bytes epoch)]) [.str "id"] 32

/-- Mode 0's key: `derive(source, CHALLENGE_DOMAIN, {audit, epoch, root, salt})`. -/
def auditKey (source aid epoch root salt : ByteArray) : ByteArray :=
  derive source CHALLENGE_DOMAIN [("audit", .bytes aid), ("epoch", .bytes epoch), ("root", .bytes root), ("salt", .bytes salt)]

/-- Mode 1's key for round `j`: `derive(beacon, ROUND_DOMAIN, {audit, epoch, root, round})`. -/
def roundKey (beacon aid epoch root : ByteArray) (j : Nat) : ByteArray :=
  derive beacon ROUND_DOMAIN [("audit", .bytes aid), ("epoch", .bytes epoch), ("root", .bytes root), ("round", .int j)]

/-- Round `j`'s block index under `key`: `key.uniform(blocks, j)`; `none` only for `blocks = 0`. -/
def index (key : ByteArray) (blocks j : Nat) : Option Nat := uniform key blocks [.int j]

/-- Round `j`'s mode-0 nonce: `key.stream(("nonce", j), 32)`. -/
def nonce (key : ByteArray) (j : Nat) : ByteArray := stream key [.str "nonce", .int j] NONCE

end PousCheck.Challenge
