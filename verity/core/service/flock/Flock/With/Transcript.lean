/-
§5–§6 over the hash oracle (`Flock.With.Hashes`): the replay transcript's squeeze checks a round's digest through the oracle,
and the session record's digests are the oracle's. A transcript action that hashes nothing is the compiled verifier's own,
lifted (`liftTM`).
-/
import Flock.With.Hashes
import Flock.Record

namespace Flock

open Transcript
open Lean (Json)

/-- The replay transcript over the verdict monad `M`. -/
abbrev TMW (M : Type → Type) := StateT Transcript M

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- A transcript action that hashes nothing, over `M`. -/
def liftTM {α : Type} (x : TM α) : TMW M α := fun t => monadLift (x t)

/-- `Round.answers` through the oracle. -/
def Round.answersWith (hs : Hashes M) (r : Round) (c : ByteArray) : M Bool :=
  match r.bytes with
  | some b => pure (decide (c = b))
  | none => do
    let d ← if r.sha.size = 64 then hs.sha512 c else hs.sha256 c
    pure (decide (d = r.sha))

namespace Transcript

/-- `squeeze` through the oracle. -/
def squeezeWith (hs : Hashes M) (kind : UInt8) (n : Nat) : TMW M (Array F128) := fun t =>
  let c := framed t kind n
  if n = 0 then pure (#[], { t with pending := c })
  else match t.rounds[t.k]? with
    | none => throw s!"R2: round {t.k}: the verifier issued no such coin"
    | some r =>
      if r.n ≠ n then throw s!"R2: round {t.k}: {n} coins requested, {r.n} issued"
      else do
        let ok ← r.answersWith hs c
        if !ok then
          throw (match r.bytes with
            | some _ => s!"R2: round {t.k}: the proof's messages are not the bytes the server retained before the coin"
            | none => s!"R2: round {t.k}: the proof's messages differ from those committed before the coin")
        else pure (r.coins, { t with pending := ByteArray.empty, k := t.k + 1 })

def sampleWith (hs : Hashes M) : TMW M F128 := do
  let c ← squeezeWith hs 1 1
  return c.getD 0 0

def sampleVecWith (hs : Hashes M) (n : Nat) : TMW M (Array F128) := squeezeWith hs 2 n

def sampleKWith (hs : Hashes M) : TMW M F256 := do
  let c ← squeezeWith hs 2 2
  return ⟨c.getD 0 0, c.getD 1 0⟩

end Transcript

/-- The record's digest (`SessionSpec.dg`) through the oracle. -/
def SessionSpec.dgWith (hs : Hashes M) (spec : SessionSpec) (b : ByteArray) : M ByteArray :=
  if spec.sha512Digests then hs.sha512 b else hs.sha256 b

namespace Record

/-- `Record.decode` with the record's digests through the oracle. -/
def decodeWith (hs : Hashes M) (spec : SessionSpec) (text : String) : M Session := do
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
    check ((← str (← fld l s!"root_b_{dk}") t) == hexOf (← spec.dgWith hs rb)) s!"S5: {t}: root_b differs from root_b_{dk}"
    pure rb
  -- S6 (D1: root_F = Σ)
  let rootF ← unhex (← str l "root_f")
  check (rootF.size > 0) "S6: empty root_F"
  check ((← str l s!"root_f_{dk}") == hexOf (← spec.dgWith hs rootF)) s!"S6: root_f differs from root_f_{dk}"
  check (rootF == spec.sigma) "S6/D1: root_F is not Σ"
  -- S7 (an absent key is empty bytes)
  let pubs ← fld l "publics"
  let pubShas ← fld l s!"publics_{dk}"
  let publics ← spec.tables.mapM fun t => do
    let p ← if isNull pubs t then pure ByteArray.empty else unhex (← str pubs t)
    if !(p.size == 0 && isNull pubShas t) then
      check ((← str pubShas t) == hexOf (← spec.dgWith hs p)) s!"S7: {t}: publics differ from publics_{dk}"
    pure p
  -- S9
  let pts ← ((← arr l "points").mapM fun p => match p.getArr? with
    | .ok a => f128s a
    | .error _ => .error "record: a link point is not a list" : V (Array (Array F128)))
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
      check (1 ≤ n && n ≤ 65536 && coins.size == n && sha.size == (← spec.dgWith hs ByteArray.empty).size) s!"S12: {name} round {k} is malformed"
      -- the framed bytes, when the server retained them: bound to the digest, then compared exactly at replay (§6.2)
      check (!spec.retainRounds || !isNull r "msg") s!"S12: {name} round {k}: the statement retains round bytes and the record has none"
      let bytes ← if isNull r "msg" then pure none else do
        let b ← unhex (← str r "msg")
        check ((← spec.dgWith hs b) == sha && b.size == (← nat r "msg_len")) s!"S12: {name} round {k}: the retained bytes are not the recorded digest's"
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
    check (s.rootSha == some (hexOf (← spec.dgWith hs rootB[i]!))) s!"S13/R1: {s.name} bound a root other than the committed root_B"
  -- S14 (the roots in lexicographic table order)
  let mut lb := rootF
  for (_, rb) in (spec.tables.zip rootB).qsort (·.1 < ·.1) do lb := lb ++ rb
  for p in pts do
    for c in p do lb := F128.pushBytes lb c
  for c in y do lb := F128.pushBytes lb c
  check ((← str j s!"link_{dk}") == hexOf (← spec.dgWith hs lb)) s!"S14: link_{dk} is not the exchange it covers"
  let proofSha ← (← arr j s!"proof_{dk}").mapM fun p => do pure ((← str p "stream"), (← str p dk))
  return { rootF, rootB, publics, points := pts, y, streams, proofSha }

end Record

end Flock
