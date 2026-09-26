/-
PROTOCOL.md §6: the replay transcript. The only source of coins is the session record; every coin is returned only after
the framed bytes that precede it hash to the recorded digest (R2).
-/
import Flock.Bytes
import Flock.Hash

namespace Flock

/-- One recorded round: the coin count, the digest of the framed bytes before the coin, the coins. -/
structure Round where
  n : Nat
  sha : ByteArray
  coins : Array F128
  deriving Inhabited

structure Transcript where
  rounds : Array Round
  /-- rounds consumed -/
  k : Nat
  /-- framed bytes absorbed since the last coin -/
  pending : ByteArray
  deriving Inhabited

namespace Transcript

def header (op kind : UInt8) (len : Nat) : ByteArray :=
  (ByteArray.mk #[op, kind, 0, 0, 0, 0, 0, 0]) ++ le64 len.toUInt64

def padded (b : ByteArray) : ByteArray := Id.run do
  let mut out := b
  for _ in [0:(16 - b.size % 16) % 16] do out := out.push 0
  return out

def init (rounds : Array Round) (domain : ByteArray) : Transcript :=
  ⟨rounds, 0, header 1 0 domain.size ++ padded domain⟩

abbrev TM := StateT Transcript V

def label (l : String) : TM Unit :=
  let b := l.toUTF8
  modify fun t => { t with pending := t.pending ++ header 2 0 b.size ++ padded b }

def absorbF (v : F128) : TM Unit :=
  modify fun t => { t with pending := F128.pushBytes (t.pending ++ header 3 1 1) v }

def absorbSlice (vs : Array F128) : TM Unit :=
  modify fun t => { t with pending := vs.foldl F128.pushBytes (t.pending ++ header 3 2 vs.size) }

def absorbK (v : F256) : TM Unit := absorbSlice #[v.c0, v.c1]

def absorbBytes (b : ByteArray) : TM Unit :=
  modify fun t => { t with pending := t.pending ++ header 5 0 b.size ++ padded b }

/-- A proof-of-work nonce: a message, never checked (R4). -/
def pow (nonce : UInt64) : TM Unit := absorbBytes (le64 nonce)

def squeeze (kind : UInt8) (n : Nat) : TM (Array F128) := do
  let t ← get
  let c := t.pending ++ header 4 kind n
  if n == 0 then
    set { t with pending := c }
    return #[]
  match t.rounds[t.k]? with
  | none => throw s!"R2: round {t.k}: the verifier issued no such coin"
  | some r =>
    if r.n != n then throw s!"R2: round {t.k}: {n} coins requested, {r.n} issued"
    if Sha256.hash c != r.sha then throw s!"R2: round {t.k}: the proof's messages differ from those committed before the coin"
    set { t with pending := ByteArray.empty, k := t.k + 1 }
    return r.coins

def sample : TM F128 := do
  let c ← squeeze 1 1
  return c.getD 0 0

def sampleVec (n : Nat) : TM (Array F128) := squeeze 2 n

def sampleK : TM F256 := do
  let c ← squeeze 2 2
  return ⟨c.getD 0 0, c.getD 1 0⟩

end Transcript

end Flock
