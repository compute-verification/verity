/-
The element-level verdict (`verifyWithME`): `verifyWithM` with its digests of encodings dropped, for the recursion's
analysis (the element-level inner verifier, `Claims.Recursion.Bound.DE`). It is never deployed.

`verifyWithMK keep` (with `Record.decodeWithK keep`) is `verifyWithM`'s text with each comparison of a record digest
with the digest of an encoding read through `keep`: S5 `root_b_{dk}`, S6 `root_f_{dk}`, S7 `publics_{dk}`, S12
`msg_{dk}`'s width and the retained bytes' digest, S13 the rep streams' `root_{dk}`, S14 `link_{dk}`, S17
`proof_{dk}`. At `keep := id` it is `verifyWithM` (`verifyWithM_eq_verifyWithMK`, by `rfl`); at `keep := fun _ => true`
it is the element-level verdict. Every hash call and every other check is kept, the element-level ones (S6/D1
`root_F = Σ`, S12's round shape and `msg_len`, S15, S16, S18) and the record's shape among them.

So the element-level verdict accepts wherever the shipped one does, with the same oracle log
(`verifyWithE_of_verifyWith`, `verifyLoggedE_of_verifyLogged`). `Sim.impSt` relates two runs of one `StV σ` where the
second ends `.ok`, at the same state, wherever the first does; it walks the two `keep`s in step.
-/
import Flock.With.Log

namespace Flock

open Transcript
open Lean (Json)

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

namespace Record

/-- **`Record.decodeJWith` with its digests of encodings read through `keep`** (S5–S7, S12's digest width and
retained bytes' digest, S13, S14), at the record's parsed value `j`: `keep := id` is `decodeJWith`
(`decodeJWith_eq_decodeJWithK`). -/
def decodeJWithK (keep : Bool → Bool) (hs : Hashes M) (spec : SessionSpec) (j : Json) : M Session := do
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
  -- a match, not `Json`'s `==` (a `partial def`, so no proof can evaluate it)
  check ((← fld cfg "require_link") matches Json.bool true) "S3/F1: require_link is not true"
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
    check (keep ((← str (← fld l s!"root_b_{dk}") t) == hexOf (← spec.dgWith hs rb)))
      s!"S5: {t}: root_b differs from root_b_{dk}"
    pure rb
  -- S6 (D1: root_F = Σ)
  let rootF ← unhex (← str l "root_f")
  check (rootF.size > 0) "S6: empty root_F"
  check (keep ((← str l s!"root_f_{dk}") == hexOf (← spec.dgWith hs rootF))) s!"S6: root_f differs from root_f_{dk}"
  check (rootF == spec.sigma) "S6/D1: root_F is not Σ"
  -- S7 (an absent key is empty bytes)
  let pubs ← fld l "publics"
  let pubShas ← fld l s!"publics_{dk}"
  let publics ← spec.tables.mapM fun t => do
    let p ← if isNull pubs t then pure ByteArray.empty else unhex (← str pubs t)
    if !(p.size == 0 && isNull pubShas t) then
      check (keep ((← str pubShas t) == hexOf (← spec.dgWith hs p))) s!"S7: {t}: publics differ from publics_{dk}"
    pure p
  -- S9
  let pts ← ((← arr l "points").mapM fun p => match p.getArr? with
    | .ok a => f128s a
    | .error _ => .error "record: a link point is not a list" : V (Array (Array F128)))
  check (pts.size == spec.points && pts.all (·.size == spec.mPts)) "S9: the link points are not the configured shape"
  -- S10
  let y ← f128s (← arr l "y")
  check (y.size == spec.points * spec.tables.size && y.all (·.isZero))
    "S10: y is not the configured shape, or not zero"
  -- S11, S12, S13
  let streams ← (← arr j "streams").mapM fun s => do
    let name ← str s "stream"
    let rounds ← (← arr s "rounds").mapIdxM fun k r => do
      check ((← nat r "k") == k) s!"S12: {name}: rounds out of order"
      let n ← nat r "n"
      let sha ← unhex (← str r s!"msg_{dk}")
      let coins ← f128s (← arr r "coins")
      check (1 ≤ n && n ≤ 65536 && coins.size == n && keep (sha.size == (← spec.dgWith hs ByteArray.empty).size))
        s!"S12: {name} round {k} is malformed"
      -- the framed bytes, when the server retained them: bound to the digest, then compared exactly at replay (§6.2)
      check (!spec.retainRounds || !isNull r "msg")
        s!"S12: {name} round {k}: the statement retains round bytes and the record has none"
      let bytes ← if isNull r "msg" then pure none else do
        let b ← unhex (← str r "msg")
        check (keep ((← spec.dgWith hs b) == sha) && b.size == (← nat r "msg_len"))
          s!"S12: {name} round {k}: the retained bytes are not the recorded digest's"
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
    check (keep (s.rootSha == some (hexOf (← spec.dgWith hs rootB[i]!))))
      s!"S13/R1: {s.name} bound a root other than the committed root_B"
  -- S14 (the roots in lexicographic table order)
  let mut lb := rootF
  for (_, rb) in (spec.tables.zip rootB).qsort (·.1 < ·.1) do lb := lb ++ rb
  for p in pts do
    for c in p do lb := F128.pushBytes lb c
  for c in y do lb := F128.pushBytes lb c
  check (keep ((← str j s!"link_{dk}") == hexOf (← spec.dgWith hs lb)))
    s!"S14: link_{dk} is not the exchange it covers"
  let proofSha ← (← arr j s!"proof_{dk}").mapM fun p => do pure ((← str p "stream"), (← str p dk))
  return { rootF, rootB, publics, points := pts, y, streams, proofSha }

/-- **`Record.decodeWith` with its digests of encodings read through `keep`**: the parse, then `decodeJWithK`.
`keep := id` is `decodeWith` (`decodeWith_eq_decodeWithK`), `keep := fun _ => true` drops them (`decodeWithE`). -/
def decodeWithK (keep : Bool → Bool) (hs : Hashes M) (spec : SessionSpec) (text : String) : M Session := do
  let j ← match Json.parse text with
    | .ok j => pure j
    | .error e => throw s!"record: not JSON: {e}"
  decodeJWithK keep hs spec j

theorem decodeWithK_of_parse [LawfulMonad M] (keep : Bool → Bool) (hs : Hashes M) (spec : SessionSpec)
    {text : String} {j : Json} (h : Json.parse text = .ok j) :
    decodeWithK keep hs spec text = decodeJWithK keep hs spec j := by
  simp only [decodeWithK, h, pure_bind]

theorem decodeJWith_eq_decodeJWithK (hs : Hashes M) (spec : SessionSpec) (j : Json) :
    decodeJWith hs spec j = decodeJWithK id hs spec j :=
  rfl

/-- `Record.decodeWith` with its digests of encodings dropped. -/
def decodeWithE (hs : Hashes M) (spec : SessionSpec) (text : String) : M Session :=
  decodeWithK (fun _ => true) hs spec text

theorem decodeWith_eq_decodeWithK (hs : Hashes M) (spec : SessionSpec) (text : String) :
    decodeWith hs spec text = decodeWithK id hs spec text :=
  rfl

end Record

/-- **`verifyWithM` with its digests of encodings read through `keep`** (`Record.decodeWithK`'s, and S17's proof-file
digests): `keep := id` is `verifyWithM` (`verifyWithM_eq_verifyWithMK`). -/
def verifyWithMK (keep : Bool → Bool) (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) : M Unit := do
  if tabs.size != spec.tables.size || tabs.size == 0 then throw "setup: one statement per table expected"
  let sess ← Record.decodeWithK keep hs spec record
  for (st, p) in tabs.zip sess.publics do st.publics p
  let schs ← tabs.mapM fun st => match fast100 st.m with
    | some sch => pure sch
    | none => throw s!"m = {st.m}: no fast100 schedule"
  let want := spec.repStreams
  if proofs.size != want.size then throw s!"{want.size} proofs expected"
  if sess.proofSha.size != want.size || !want.all (fun (_, _, w) => sess.proofSha.any (·.1 == w)) then
    throw s!"S17: proof_{spec.dk} does not list exactly the tables' rep streams"
  for ((_, _, w), b) in want.zip proofs do
    if !keep ((sess.proofSha.find? (·.1 == w)).map (·.2) == some (hexOf (← spec.dgWith hs b))) then
      throw s!"S17: {w}: the proof is not the one the server received"
  for i in [0:tabs.size] do
    let (st, sch) := (tabs[i]!, schs[i]!)
    let c0 ← verifyRepWith hs spec st sch sess i 0 proofs[2 * i]!
    let c1 ← verifyRepWith hs spec st sch sess i 1 proofs[2 * i + 1]!
    if c0 != c1 then throw s!"S18/R1: {spec.tables[i]!}: the reps' caps differ"

/-- **The element-level verdict**: `verifyWithM` with its digests of encodings dropped. `.ok ()` is Accept. -/
def verifyWithME (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) : M Unit :=
  verifyWithMK (fun _ => true) hs spec tabs record proofs

theorem verifyWithM_eq_verifyWithMK (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) : verifyWithM hs spec tabs record proofs = verifyWithMK id hs spec tabs record proofs :=
  rfl

/-- `verifyWithMK keep` at the record's parsed value `j` (`Record.decodeJWithK`); at a record that parses to `j` it is
`verifyWithMK keep` (`verifyWithMK_of_parse`), and at `keep := id` it is `verifyJWithM`
(`verifyJWithM_eq_verifyJWithMK`). -/
def verifyJWithMK (keep : Bool → Bool) (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (j : Json)
    (proofs : Array ByteArray) : M Unit := do
  if tabs.size != spec.tables.size || tabs.size == 0 then throw "setup: one statement per table expected"
  let sess ← Record.decodeJWithK keep hs spec j
  for (st, p) in tabs.zip sess.publics do st.publics p
  let schs ← tabs.mapM fun st => match fast100 st.m with
    | some sch => pure sch
    | none => throw s!"m = {st.m}: no fast100 schedule"
  let want := spec.repStreams
  if proofs.size != want.size then throw s!"{want.size} proofs expected"
  if sess.proofSha.size != want.size || !want.all (fun (_, _, w) => sess.proofSha.any (·.1 == w)) then
    throw s!"S17: proof_{spec.dk} does not list exactly the tables' rep streams"
  for ((_, _, w), b) in want.zip proofs do
    if !keep ((sess.proofSha.find? (·.1 == w)).map (·.2) == some (hexOf (← spec.dgWith hs b))) then
      throw s!"S17: {w}: the proof is not the one the server received"
  for i in [0:tabs.size] do
    let (st, sch) := (tabs[i]!, schs[i]!)
    let c0 ← verifyRepWith hs spec st sch sess i 0 proofs[2 * i]!
    let c1 ← verifyRepWith hs spec st sch sess i 1 proofs[2 * i + 1]!
    if c0 != c1 then throw s!"S18/R1: {spec.tables[i]!}: the reps' caps differ"

theorem verifyWithMK_of_parse [LawfulMonad M] (keep : Bool → Bool) (hs : Hashes M) (spec : SessionSpec)
    (tabs : Array Setup) {record : String} {j : Json} (proofs : Array ByteArray) (h : Json.parse record = .ok j) :
    verifyWithMK keep hs spec tabs record proofs = verifyJWithMK keep hs spec tabs j proofs := by
  simp only [verifyWithMK, verifyJWithMK, Record.decodeWithK_of_parse keep hs spec h]

theorem verifyJWithM_eq_verifyJWithMK (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (j : Json)
    (proofs : Array ByteArray) : verifyJWithM hs spec tabs j proofs = verifyJWithMK id hs spec tabs j proofs :=
  rfl

/-- The element-level verdict at the record's parsed value `j`. -/
def verifyJWithME (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (j : Json) (proofs : Array ByteArray) :
    M Unit :=
  verifyJWithMK (fun _ => true) hs spec tabs j proofs

/-- The element-level verdict with SHA-512 replaced by `h`. -/
def verifyWithE (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) : V Unit :=
  verifyWithME (Hashes.ofSha512 h) spec tabs record proofs

/-- `verifyWithME` with the logging oracle from the log `log`. -/
def verifyLoggedE (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) (log : Array ByteArray := #[]) : Except String Unit × Array ByteArray :=
  (verifyWithME (Hashes.logged h) spec tabs record proofs).go log

namespace With

theorem decodeWithK_sim {N : Type → Type} [Monad N] [MonadExceptOf String N] [MonadLiftT V N] (S : Sim M N)
    [LawfulMonad M] [LawfulMonad N] {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (keep : Bool → Bool) (spec : SessionSpec) (text : String) :
    S.R (Record.decodeWithK keep hs spec text) (Record.decodeWithK keep hs' spec text) := by
  unfold Record.decodeWithK Record.decodeJWithK
  sim_walk [dgWith_sim S hh]

theorem decodeJWithK_sim {N : Type → Type} [Monad N] [MonadExceptOf String N] [MonadLiftT V N] (S : Sim M N)
    [LawfulMonad M] [LawfulMonad N] {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (keep : Bool → Bool) (spec : SessionSpec) (j : Json) :
    S.R (Record.decodeJWithK keep hs spec j) (Record.decodeJWithK keep hs' spec j) := by
  unfold Record.decodeJWithK
  sim_walk [dgWith_sim S hh]

theorem verifyWithMK_sim {N : Type → Type} [Monad N] [MonadExceptOf String N] [MonadLiftT V N] (S : Sim M N)
    [LawfulMonad M] [LawfulMonad N] {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (keep : Bool → Bool) (spec : SessionSpec) (tabs : Array Setup) (record : String) (proofs : Array ByteArray) :
    S.R (verifyWithMK keep hs spec tabs record proofs) (verifyWithMK keep hs' spec tabs record proofs) := by
  unfold verifyWithMK
  sim_walk [decodeWithK_sim S hh, dgWith_sim S hh, verifyRepWith_sim S hh]

theorem verifyJWithMK_sim {N : Type → Type} [Monad N] [MonadExceptOf String N] [MonadLiftT V N] (S : Sim M N)
    [LawfulMonad M] [LawfulMonad N] {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (keep : Bool → Bool) (spec : SessionSpec) (tabs : Array Setup) (j : Json) (proofs : Array ByteArray) :
    S.R (verifyJWithMK keep hs spec tabs j proofs) (verifyJWithMK keep hs' spec tabs j proofs) := by
  unfold verifyJWithMK
  sim_walk [decodeJWithK_sim S hh, dgWith_sim S hh, verifyRepWith_sim S hh]

theorem verifyWithME_sim {N : Type → Type} [Monad N] [MonadExceptOf String N] [MonadLiftT V N] (S : Sim M N)
    [LawfulMonad M] [LawfulMonad N] {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (spec : SessionSpec) (tabs : Array Setup) (record : String) (proofs : Array ByteArray) :
    S.R (verifyWithME hs spec tabs record proofs) (verifyWithME hs' spec tabs record proofs) :=
  verifyWithMK_sim S hh _ spec tabs record proofs

namespace Sim

/-- **Two runs of one `StV σ`, the first's acceptance the second's**: from every state where the first ends `.ok` the
second ends there too, at the same state. -/
def impSt (σ : Type) : Sim (StV σ) (StV σ) where
  R x y := ∀ s a s', x.go s = (.ok a, s') → y.go s = (.ok a, s')
  pure _ _ _ _ h := h
  bind {α β x y f g} hxy hfg s b s'' h := by
    simp only [StV.go, Id.run, ExceptT.run_bind, StateT.run_bind] at h ⊢
    rcases hr : x.run.run s with ⟨r, s'⟩
    rw [hr] at h
    cases r with
    | error e => cases h
    | ok a =>
      have hy := hxy s a s' (by simp only [StV.go, Id.run]; exact hr)
      simp only [StV.go, Id.run] at hy
      rw [hy]
      exact hfg a s' b s'' h
  throw _ s _ _ h := by cases h
  lift _ _ _ _ h := h

theorem impSt_hashes (σ : Type) (hs : Flock.Hashes (StV σ)) : (impSt σ).Hashes hs hs where
  sha512 _ _ _ _ h := h
  sha256 _ _ _ _ h := h
  blake3 _ _ _ _ h := h
  blake3Keyed _ _ _ _ _ h := h

variable {σ : Type} {k₁ k₂ : Bool → Bool}

theorem check_go_ok {b : Bool} {why : String} {s s' : σ} {a : Unit}
    (h : (monadLift (Record.check b why) : StV σ Unit).go s = (.ok a, s')) : b = true := by
  cases b
  · cases h
  · rfl

/-- A digest a run reads through `k₁` and the other through `k₂`, where `k₁`'s acceptance is `k₂`'s. -/
theorem impSt_keep (hk : ∀ b, k₁ b = true → k₂ b = true) (c : Bool) (why : String) :
    (impSt σ).R (monadLift (Record.check (k₁ c) why) : StV σ Unit) (monadLift (Record.check (k₂ c) why)) := by
  intro s a s' h
  have h₂ := hk c (check_go_ok h)
  rw [check_go_ok h] at h
  rw [h₂]
  exact h

theorem impSt_keep_and (hk : ∀ b, k₁ b = true → k₂ b = true) (a c : Bool) (why : String) :
    (impSt σ).R (monadLift (Record.check (a && k₁ c) why) : StV σ Unit)
      (monadLift (Record.check (a && k₂ c) why)) := by
  intro s u s' h
  have h₁ := Bool.and_eq_true_iff.1 (check_go_ok h)
  rw [h₁.1, h₁.2] at h
  rw [h₁.1, hk c h₁.2]
  exact h

theorem impSt_and_keep (hk : ∀ b, k₁ b = true → k₂ b = true) (a c : Bool) (why : String) :
    (impSt σ).R (monadLift (Record.check (k₁ c && a) why) : StV σ Unit)
      (monadLift (Record.check (k₂ c && a) why)) := by
  intro s u s' h
  have h₁ := Bool.and_eq_true_iff.1 (check_go_ok h)
  rw [h₁.1, h₁.2] at h
  rw [h₁.2, hk c h₁.1]
  exact h

/-- A refusal on a digest a run reads through `k₁` and the other through `k₂`: where the first goes on, so does the
second. -/
theorem impSt_keep_ite {β : Type} (hk : ∀ b, k₁ b = true → k₂ b = true) (c : Bool) (e : String)
    (f f' : PUnit → StV σ β) {z z' : StV σ β} (hz : (impSt σ).R z z') :
    (impSt σ).R (if (!k₁ c) = true then MonadExcept.throw e >>= f else z)
      (if (!k₂ c) = true then MonadExcept.throw e >>= f' else z') := by
  intro s a s' h
  cases h₁ : k₁ c
  · rw [h₁] at h
    simp only [Bool.not_false, ↓reduceIte] at h
    cases h
  · rw [h₁] at h
    rw [hk c h₁]
    simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h ⊢
    exact hz s a s' h

end Sim

theorem decodeWithK_impSt {σ : Type} (hs : Flock.Hashes (StV σ)) {k₁ k₂ : Bool → Bool}
    (hk : ∀ b, k₁ b = true → k₂ b = true) (spec : SessionSpec) (text : String) :
    (Sim.impSt σ).R (Record.decodeWithK k₁ hs spec text) (Record.decodeWithK k₂ hs spec text) := by
  unfold Record.decodeWithK Record.decodeJWithK
  sim_walk [dgWith_sim (Sim.impSt σ) (Sim.impSt_hashes σ hs), Sim.impSt_keep hk, Sim.impSt_keep_and hk,
    Sim.impSt_and_keep hk]

theorem decodeJWithK_impSt {σ : Type} (hs : Flock.Hashes (StV σ)) {k₁ k₂ : Bool → Bool}
    (hk : ∀ b, k₁ b = true → k₂ b = true) (spec : SessionSpec) (j : Json) :
    (Sim.impSt σ).R (Record.decodeJWithK k₁ hs spec j) (Record.decodeJWithK k₂ hs spec j) := by
  unfold Record.decodeJWithK
  sim_walk [dgWith_sim (Sim.impSt σ) (Sim.impSt_hashes σ hs), Sim.impSt_keep hk, Sim.impSt_keep_and hk,
    Sim.impSt_and_keep hk]

theorem verifyWithMK_impSt {σ : Type} (hs : Flock.Hashes (StV σ)) {k₁ k₂ : Bool → Bool}
    (hk : ∀ b, k₁ b = true → k₂ b = true) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) :
    (Sim.impSt σ).R (verifyWithMK k₁ hs spec tabs record proofs) (verifyWithMK k₂ hs spec tabs record proofs) := by
  unfold verifyWithMK
  sim_walk [decodeWithK_impSt hs hk, dgWith_sim (Sim.impSt σ) (Sim.impSt_hashes σ hs),
    verifyRepWith_sim (Sim.impSt σ) (Sim.impSt_hashes σ hs), Sim.impSt_keep_ite hk]

theorem verifyJWithMK_impSt {σ : Type} (hs : Flock.Hashes (StV σ)) {k₁ k₂ : Bool → Bool}
    (hk : ∀ b, k₁ b = true → k₂ b = true) (spec : SessionSpec) (tabs : Array Setup) (j : Json)
    (proofs : Array ByteArray) :
    (Sim.impSt σ).R (verifyJWithMK k₁ hs spec tabs j proofs) (verifyJWithMK k₂ hs spec tabs j proofs) := by
  unfold verifyJWithMK
  sim_walk [decodeJWithK_impSt hs hk, dgWith_sim (Sim.impSt σ) (Sim.impSt_hashes σ hs),
    verifyRepWith_sim (Sim.impSt σ) (Sim.impSt_hashes σ hs), Sim.impSt_keep_ite hk]

end With

/-- **The element-level verdict accepts wherever the shipped one does, with the same log**: from every log, a run of
`verifyWithM` with the logging oracle that ends `.ok` is a run of `verifyWithME` that ends there, at the same log. -/
theorem verifyLoggedE_of_verifyLogged (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup)
    (record : String) (proofs : Array ByteArray) (log log' : Array ByteArray)
    (hv : verifyLogged h spec tabs record proofs log = (.ok (), log')) :
    verifyLoggedE h spec tabs record proofs log = (.ok (), log') :=
  With.verifyWithMK_impSt (Hashes.logged h) (k₁ := id) (fun _ _ => rfl) spec tabs record proofs log () log' hv

theorem verifyLoggedE_fst (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) (log : Array ByteArray) :
    (verifyLoggedE h spec tabs record proofs log).1 = verifyWithE h spec tabs record proofs :=
  With.verifyWithME_sim (With.Sim.forget _) (With.Sim.forget_logged h) spec tabs record proofs log

/-- **The element-level verdict accepts wherever the shipped one does.** -/
theorem verifyWithE_of_verifyWith (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup)
    (record : String) (proofs : Array ByteArray) (hv : verifyWith h spec tabs record proofs = .ok ()) :
    verifyWithE h spec tabs record proofs = .ok () := by
  rw [← verifyLogged_fst h spec tabs record proofs #[]] at hv
  rw [← verifyLoggedE_fst h spec tabs record proofs #[]]
  rcases hl : verifyLogged h spec tabs record proofs #[] with ⟨r, log'⟩
  rw [hl] at hv
  cases hv
  rw [verifyLoggedE_of_verifyLogged h spec tabs record proofs #[] log' hl]

end Flock
