/-
PROTOCOL.md §5: the session record (`flock-live-session/v1`) and the session checks S1–S14, S17.
-/
import Lean.Data.Json
import Flock.Transcript
import Flock.Draw
import Flock.CoinTree

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
  /-- the session's tables, in table order (one, or #306's several: `Stmt.setupTables`) -/
  tables : Array String
  /-- a table's rep stream domain -/
  domain : String → Nat → String
  sigma : ByteArray
  hello : String
  mPts : Nat
  points : Nat := 2
  /-- the statement's challenger retains every round's bytes: each round must carry them (§6.2) -/
  retainRounds : Bool := false
  /-- the session's unit draw (one table's: the public header's `unit_draw`, §7.3; several tables': the draw they split, never
  a table's part) and its instance count -/
  unitDraw : Option Json := none
  instances : Nat := 0
  /-- the record's digests are SHA-512 under `_sha512` keys (PR #83's `SessionConfig::sha512_digests`), else SHA-256 -/
  sha512Digests : Bool := false
  /-- the verifier's keyed coin tree (`coin-tree/hm96-sha512/v2`), when its `Hello` names one: R7 takes the prover's nonce -/
  coinTree : Option CoinTree.Spec := none
  deriving Inhabited

/-- The record's digest key suffix and hash (`dk`, `dg`). -/
def SessionSpec.dk (spec : SessionSpec) : String := if spec.sha512Digests then "sha512" else "sha256"
def SessionSpec.dg (spec : SessionSpec) (b : ByteArray) : ByteArray :=
  if spec.sha512Digests then Sha512.hash b else Sha256.hash b

/-- The session's rep streams, table by table: the table's index, the rep, and the stream `<t>/rep<r>`. -/
def SessionSpec.repStreams (spec : SessionSpec) : Array (Nat × Nat × String) :=
  (Array.range spec.tables.size).flatMap fun i => #[0, 1].map fun r => (i, r, s!"{spec.tables[i]!}/rep{r}")

structure Session where
  rootF : ByteArray
  /-- each table's root_B and public words, in the spec's table order -/
  rootB : Array ByteArray
  publics : Array ByteArray
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
  match spec.coinTree with
  | none => check ((← str j "hello") == spec.hello) "S2/R7: the session parameters are not the verifier's"
  | some ct =>
    let some own := (Json.parse spec.hello).toOption | throw "S2/R7: the verifier's Hello is not JSON"
    let nonce ← CoinTree.checkHello own (← str j "hello")
    let _ ← CoinTree.checkRecord ct nonce (← fld j "coin_tree")
  -- S3
  let cfg ← fld j "config"
  check ((← fld cfg "require_link") == Json.bool true) "S3/F1: require_link is not true"
  check ((← str j "link_mode") == "exchange") "S3/E1: link_mode is not exchange"
  let l ← fld j "link"
  check ((← str l "mode") == "exchange") "S3/E1: link.mode is not exchange"
  -- S4
  check ((← str l "sigma") == hexOf spec.sigma) "S4: the record's Σ is not the statement's"
  -- U1–U3 (§7.3): a null draw is no draw
  let recDraw := match j.getObjVal? "unit_draw" with
    | .ok Json.null => none
    | .ok d => some d
    | .error _ => none
  Draw.check recDraw spec.unitDraw spec.instances
  let dk := spec.dk
  -- S5 (keys of other tables are ignored)
  let rootB ← spec.tables.mapM fun t => do
    let rb ← unhex (← str (← fld l "root_b") t)
    check (rb.size > 0) s!"S5: {t}: empty root_B"
    check ((← str (← fld l s!"root_b_{dk}") t) == hexOf (spec.dg rb)) s!"S5: {t}: root_b differs from root_b_{dk}"
    pure rb
  -- S6 (D1: root_F = Σ)
  let rootF ← unhex (← str l "root_f")
  check (rootF.size > 0) "S6: empty root_F"
  check ((← str l s!"root_f_{dk}") == hexOf (spec.dg rootF)) s!"S6: root_f differs from root_f_{dk}"
  check (rootF == spec.sigma) "S6/D1: root_F is not Σ"
  -- S7 (an absent key is empty bytes)
  let pubs ← fld l "publics"
  let pubShas ← fld l s!"publics_{dk}"
  let publics ← spec.tables.mapM fun t => do
    let p ← if isNull pubs t then pure ByteArray.empty else unhex (← str pubs t)
    if !(p.size == 0 && isNull pubShas t) then
      check ((← str pubShas t) == hexOf (spec.dg p)) s!"S7: {t}: publics differ from publics_{dk}"
    pure p
  -- S9
  let pts ← (← arr l "points").mapM fun p => match p.getArr? with
    | .ok a => f128s a
    | .error _ => .error "record: a link point is not a list"
  check (pts.size == spec.points && pts.all (·.size == spec.mPts)) "S9: the link points are not the configured shape"
  -- S10
  let y ← f128s (← arr l "y")
  check (y.size == spec.points * spec.tables.size && y.all (·.isZero)) "S10: y is not the configured shape, or not zero"
  -- S11, S12, S13
  let streams ← (← arr j "streams").mapM fun s => do
    let name ← str s "stream"
    let rounds ← (← arr s "rounds").mapIdxM fun k r => do
      check ((← nat r "k") == k) s!"S12: {name}: rounds out of order"
      let n ← nat r "n"
      let sha ← unhex (← str r s!"msg_{dk}")
      let coins ← f128s (← arr r "coins")
      check (1 ≤ n && n ≤ 65536 && coins.size == n && sha.size == (spec.dg ByteArray.empty).size) s!"S12: {name} round {k} is malformed"
      -- the framed bytes, when the server retained them: bound to the digest, then compared exactly at replay (§6.2)
      check (!spec.retainRounds || !isNull r "msg") s!"S12: {name} round {k}: the statement retains round bytes and the record has none"
      let bytes ← if isNull r "msg" then pure none else do
        let b ← unhex (← str r "msg")
        check (spec.dg b == sha && b.size == (← nat r "msg_len")) s!"S12: {name} round {k}: the retained bytes are not the recorded digest's"
        pure (some b)
      pure (⟨n, sha, coins, bytes⟩ : Round)
    let rootSha := match (← fld s s!"root_{dk}").getStr? with
      | .ok x => some x
      | .error _ => none
    pure (⟨name, ← str s "parent", ← str s "label", ← nat s "at", rootSha, rounds⟩ : StreamRec)
  let want := spec.repStreams
  check (streams.size == want.size && want.all (fun (_, _, w) => streams.any (·.name == w)))
    "S11/D2: the streams are not exactly the tables' rep streams"
  for (i, r, name) in want do
    let some s := streams.find? (·.name == name) | throw s!"S11: no stream {name}"
    check (s.parent == "" && s.at_ == 0 && s.label == spec.domain spec.tables[i]! r && s.rounds.size > 0)
      s!"S11: {s.name} is not a root stream of this statement"
    check (s.rootSha == some (hexOf (spec.dg rootB[i]!))) s!"S13/R1: {s.name} bound a root other than the committed root_B"
  -- S14 (the roots in lexicographic table order)
  let mut lb := rootF
  for (_, rb) in (spec.tables.zip rootB).qsort (·.1 < ·.1) do lb := lb ++ rb
  for p in pts do
    for c in p do lb := F128.pushBytes lb c
  for c in y do lb := F128.pushBytes lb c
  check ((← str j s!"link_{dk}") == hexOf (spec.dg lb)) s!"S14: link_{dk} is not the exchange it covers"
  let proofSha ← (← arr j s!"proof_{dk}").mapM fun p => do pure ((← str p "stream"), (← str p dk))
  return { rootF, rootB, publics, points := pts, y, streams, proofSha }

end Record

end Flock
