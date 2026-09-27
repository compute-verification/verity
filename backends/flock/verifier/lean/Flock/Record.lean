/-
PROTOCOL.md §5: the session record (`flock-live-session/v1`) and the session checks S1–S14, S17.
-/
import Lean.Data.Json
import Flock.Transcript
import Flock.Draw

namespace Flock

open Lean (Json)

structure StreamRec where
  name : String
  parent : String
  label : String
  at_ : Nat
  rootSha : Option String
  rounds : Array Round
  deriving Inhabited

/-- What the verifier configures about a session (§4); nothing here comes from the record. -/
structure SessionSpec where
  table : String
  domain : Nat → String
  sigma : ByteArray
  hello : String
  mPts : Nat
  points : Nat := 2
  /-- the statement's coin server retains every round's bytes: each round must carry them (§6.2) -/
  retainRounds : Bool := false
  /-- the statement's unit draw (the public header's `unit_draw`, §7.3) and its instance count -/
  unitDraw : Option Json := none
  instances : Nat := 0
  deriving Inhabited

structure Session where
  rootF : ByteArray
  rootB : ByteArray
  publics : ByteArray
  points : Array (Array F128)
  y : Array F128
  streams : Array StreamRec
  proofSha : Array (String × String)
  deriving Inhabited

namespace Record

def fld (j : Json) (k : String) : V Json :=
  match j.getObjVal? k with
  | .ok v => .ok v
  | .error _ => .error s!"record: {k} missing"

def str (j : Json) (k : String) : V String := do
  match (← fld j k).getStr? with
  | .ok s => .ok s
  | .error _ => .error s!"record: {k} is not a string"

def arr (j : Json) (k : String) : V (Array Json) := do
  match (← fld j k).getArr? with
  | .ok a => .ok a
  | .error _ => .error s!"record: {k} is not a list"

def nat (j : Json) (k : String) : V Nat := do
  match (← fld j k).getNat? with
  | .ok a => .ok a
  | .error _ => .error s!"record: {k} is not a natural number"

def isNull (j : Json) (k : String) : Bool :=
  match j.getObjVal? k with
  | .ok Json.null => true
  | .ok _ => false
  | .error _ => true

def f128s (a : Array Json) : V (Array F128) := a.mapM fun x => match x.getStr? with
  | .ok s => f128OfHex s
  | .error _ => .error "record: an F128 is not a hex string"

def check (b : Bool) (why : String) : V Unit := if b then pure () else throw why

/-- Decode the record and run S1–S14 (§5.3). -/
def decode (spec : SessionSpec) (text : String) : V Session := do
  let j ← match Json.parse text with
    | .ok j => pure j
    | .error e => throw s!"record: not JSON: {e}"
  -- S1
  check ((← str j "kind") == "flock-live-session/v1") "S1: not a flock-live-session/v1 record"
  check (isNull j "aborted") "S1: the live session aborted"
  -- S2
  check ((← str j "hello") == spec.hello) "S2/R7: the session parameters are not the verifier's"
  -- S3
  let cfg ← fld j "config"
  check ((← fld cfg "require_link") == Json.bool true) "S3/F1: require_link is not true"
  check ((← str j "link_mode") == "exchange") "S3/E1: link_mode is not exchange"
  let l ← fld j "link"
  check ((← str l "mode") == "exchange") "S3/E1: link.mode is not exchange"
  -- S4
  check ((← str l "sigma") == hexOf spec.sigma) "S4: the record's Σ is not the statement's"
  -- U1–U3 (§7.3)
  Draw.check (j.getObjVal? "unit_draw").toOption spec.unitDraw spec.instances
  let t := spec.table
  -- S5
  let rootB ← unhex (← str (← fld l "root_b") t)
  check (rootB.size > 0) "S5: empty root_B"
  check ((← str (← fld l "root_b_sha256") t) == hexOf (Sha256.hash rootB)) "S5: root_b differs from root_b_sha256"
  -- S6 (D1: root_F = Σ)
  let rootF ← unhex (← str l "root_f")
  check (rootF.size > 0) "S6: empty root_F"
  check ((← str l "root_f_sha256") == hexOf (Sha256.hash rootF)) "S6: root_f differs from root_f_sha256"
  check (rootF == spec.sigma) "S6/D1: root_F is not Σ"
  -- S7 (an absent key is empty bytes)
  let pubs ← fld l "publics"
  let pubShas ← fld l "publics_sha256"
  let publics ← if isNull pubs t then pure ByteArray.empty else unhex (← str pubs t)
  if !(publics.size == 0 && isNull pubShas t) then
    check ((← str pubShas t) == hexOf (Sha256.hash publics)) "S7: publics differ from publics_sha256"
  -- S9
  let pts ← (← arr l "points").mapM fun p => match p.getArr? with
    | .ok a => f128s a
    | .error _ => .error "record: a link point is not a list"
  check (pts.size == spec.points && pts.all (·.size == spec.mPts)) "S9: the link points are not the configured shape"
  -- S10
  let y ← f128s (← arr l "y")
  check (y.size == spec.points && y.all (·.isZero)) "S10: y is not the configured shape, or not zero"
  -- S11, S12, S13
  let streams ← (← arr j "streams").mapM fun s => do
    let name ← str s "stream"
    let rounds ← (← arr s "rounds").mapIdxM fun k r => do
      check ((← nat r "k") == k) s!"S12: {name}: rounds out of order"
      let n ← nat r "n"
      let sha ← unhex (← str r "msg_sha256")
      let coins ← f128s (← arr r "coins")
      check (1 ≤ n && n ≤ 65536 && coins.size == n && sha.size == 32) s!"S12: {name} round {k} is malformed"
      -- the framed bytes, when the server retained them: bound to the digest, then compared exactly at replay (§6.2)
      check (!spec.retainRounds || !isNull r "msg") s!"S12: {name} round {k}: the statement retains round bytes and the record has none"
      let bytes ← if isNull r "msg" then pure none else do
        let b ← unhex (← str r "msg")
        check (Sha256.hash b == sha && b.size == (← nat r "msg_len")) s!"S12: {name} round {k}: the retained bytes are not the recorded digest's"
        pure (some b)
      pure (⟨n, sha, coins, bytes⟩ : Round)
    let rootSha := match (← fld s "root_sha256").getStr? with
      | .ok x => some x
      | .error _ => none
    pure (⟨name, ← str s "parent", ← str s "label", ← nat s "at", rootSha, rounds⟩ : StreamRec)
  let want := [s!"{t}/rep0", s!"{t}/rep1"]
  check (streams.size == 2 && want.all (fun w => streams.any (·.name == w))) "S11/D2: the streams are not exactly the two rep streams"
  for s in streams do
    let r := if s.name == s!"{t}/rep0" then 0 else 1
    check (s.parent == "" && s.at_ == 0 && s.label == spec.domain r && s.rounds.size > 0) s!"S11: {s.name} is not a root stream of this statement"
    check (s.rootSha == some (hexOf (Sha256.hash rootB))) s!"S13/R1: {s.name} bound a root other than the committed root_B"
  -- S14
  let mut lb := rootF ++ rootB
  for p in pts do
    for c in p do lb := F128.pushBytes lb c
  for c in y do lb := F128.pushBytes lb c
  check ((← str j "link_sha256") == hexOf (Sha256.hash lb)) "S14: link_sha256 is not the exchange it covers"
  let proofSha ← (← arr j "proof_sha256").mapM fun p => do pure ((← str p "stream"), (← str p "sha256"))
  return { rootF, rootB, publics, points := pts, y, streams, proofSha }

end Record

end Flock
