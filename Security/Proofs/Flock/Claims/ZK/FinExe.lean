import Proofs.Flock.Claims.ZK.Split

/-!
# The deployed plan's last message is the honest prover's (`FinExe`)

`FinExe` (`Sent`'s `hfin` at the deployed plan) asks that after `refK x` openings `ops` whose coins every table admits,
`finExe` of the tables' views is `honestPR`'s last message. Both are `Final.encode` of a session's verdict, `[]` where it
throws (`finMsg`):

* **`honestPR`'s side.** Where every table admits, no outer coin is `1` (`outerHit_false`), so `honestPR` runs `honestP`
  on the mask witness, and fed `refK x = 1 + rounds` answers it sends `finMsg` of the session answered by them
  (`honestPR_finMsg`, `ZkTables.honestP_err`'s argument at a session that returns): `liveRunAt`'s verdict.
* **`finExe`'s side** is `finMsg` of `runCut` at the views' source (`finExe_eq`).

`LiveEnd` (the session answered by the openings returns what the session answered by the opened coins and stopped
after the last round returns) and `SimFin` (the session read from the views returns the session's verdict) close the
gap: `finExe_of`.
-/

namespace FlockComplete.ZkTables

open Flock FlockDeploy

/-- **The last message of a session's verdict at `x`**: `Final.encode` of its record and proof files, `[]` where it threw
(`prog`'s). -/
def finMsg (x : Claims.Msg) : Except String (String × Array ByteArray) → Claims.Msg
  | .ok (record, proofs) => Final.encode ⟨x, record, proofs⟩
  | .error _ => []

/-- **Fed `1 + rounds` answers, the honest masked prover sends the session's verdict**: `finMsg` of the session answered
by the answers `bs` (its sends are at most `1 + rounds`, `sessionPZk_sends`). -/
theorem honestP_finMsg (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk)
    {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts))
    (hspec : spec.tables.size = sts.size) (hm : ∀ j (hj : j < sts.size), 6 ≤ sts[j].m)
    (bs : List Claims.Msg) (hl : 1 + rounds true spec sts ≤ bs.length) :
    ((feed (prog x w ω) bs).eval H).1 =
      finMsg x ((sessionPZk (asksS fun k => bs.getD k []) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts)
        (MaskRank.masksOf sts) (traces sts w) ω).go #[]).1 := by
  obtain ⟨bs', h1, h2⟩ := sessionPZk_sim H (fun k => bs.getD k []) spec (tabsOf sts) (MaskRank.masksOf sts)
    (traces sts w) ω #[]
  generalize hY : (sessionPZk (asksS fun k => bs.getD k []) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts)
    (MaskRank.masksOf sts) (traces sts w) ω).go #[] = res at h1 h2 ⊢
  obtain ⟨R, L⟩ := res
  have hsend := sendsC_ran (sessionPZk_sends (fun k => bs.getD k []) (Claims.hOf H) spec sts (MaskRank.masksOf sts)
    (traces sts w) ω hspec hm) hY
  dsimp only at h1 h2 ⊢
  rw [h2, Array.empty_append, List.size_toArray] at hsend
  simp only [Array.size_empty] at h1
  have hsim : (prog x w ω).sim H (fun k => bs.getD k []) 0 = (bs'.map (List.cons 0), finMsg x R) := by
    unfold prog
    rw [hs]
    dsimp only
    change (PI.bind (sessionPZk asks Hashes.pi spec (tabsOf sts) (MaskRank.masksOf sts) (traces sts w) ω).run _).sim
      _ _ 0 = _
    rw [PI.sim_bind, h1]
    rcases R with e | ⟨record, proofs⟩
    · show (_ ++ [], _) = _
      rw [List.append_nil]
      rfl
    · show (_ ++ [], _) = _
      rw [List.append_nil]
      rfl
  have key := feed_sim_ge H (fun k => bs.getD k []) (prog x w ω) 0 bs.length (by
    rw [hsim, List.length_map]
    omega)
  simp only [Nat.zero_add, map_getD_range] at key
  rw [key, hsim]

/-- **`honestPR`'s last message is the session's verdict** at a history whose answers `bs`, at least `1 + rounds` of
them, hold no outer coin `1`: `finMsg` of the session on the mask witness answered by them. -/
theorem honestPR_finMsg (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk)
    {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts))
    (hspec : spec.tables.size = sts.size) (hm : ∀ j (hj : j < sts.size), 6 ≤ sts[j].m)
    (hist : List (Claims.Msg × Claims.Msg)) (bs : List Claims.Msg) (hh : hist.map Prod.snd = bs)
    (hl : 1 + rounds true spec sts ≤ bs.length) (ho : outerHit x bs = false) :
    (((honestPR w).next x hist ω).eval H).1 =
      finMsg x ((sessionPZk (asksS fun k => bs.getD k []) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts)
        (MaskRank.masksOf sts) (traces sts (maskWOf x w ω.mask)) ω).go #[]).1 := by
  rw [honestPR_next (show outerHit x (hist.map Prod.snd) = false by rw [hh]; exact ho)]
  show ((feed (prog x (maskWOf x w ω.mask) ω) (hist.map Prod.snd)).eval H).1 = _
  rw [hh]
  exact honestP_finMsg H x _ ω hs hspec hm bs hl

end FlockComplete.ZkTables

namespace FlockComplete

open Flock

/-! ## The session from a source against the session, part by part -/

/-- Two runs of `PS` alike from every history are one program. -/
theorem ext_go {α : Type} {x y : PS α} (h : ∀ l, x.go l = y.go l) : x = y :=
  ExceptT.ext (StateT.ext fun l => h l)

theorem go_liftV {α : Type} (v : V α) (l : Array Claims.Msg) : (monadLift v : PS α).go l = (v, l) := by
  cases v <;> rfl

theorem go_map' {α β : Type} (f : α → β) (y : PS α) (l : Array Claims.Msg) :
    (f <$> y).go l = ((y.go l).1.map f, (y.go l).2) := by
  rw [← bind_pure_comp, go_bind]
  rcases y.go l with ⟨r, l'⟩
  cases r <;> rfl

theorem bind_go_congr {α β : Type} {x y : PS α} {l : Array Claims.Msg} (h : x.go l = y.go l) (g : α → PS β) :
    (x >>= g).go l = (y >>= g).go l := by
  rw [go_bind, go_bind, h]

theorem liftV_bind_go {α β : Type} (v : V α) (f g : α → PS β) (l : Array Claims.Msg)
    (h : ∀ a, v = .ok a → (f a).go l = (g a).go l) :
    ((monadLift v : PS α) >>= f).go l = ((monadLift v : PS α) >>= g).go l := by
  rw [go_bind, go_bind, go_liftV]
  cases v with
  | error _ => rfl
  | ok a => exact h a rfl

theorem liftV_bind_go_ok {α β : Type} {v : V α} {g : α → PS β} {l l' : Array Claims.Msg} {b : β}
    (h : ((monadLift v : PS α) >>= g).go l = (.ok b, l')) : ∃ a, v = .ok a ∧ (g a).go l = (.ok b, l') := by
  rw [go_bind, go_liftV] at h
  cases v with
  | error _ => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem quiet_map {α β : Type} (f : α → β) {x : V α} {y : PS α} (h : quietS.R x y) :
    quietS.R (f <$> x) (f <$> y) := fun l => by
  rw [go_map', h l]
  cases x <;> rfl

/-- A quiet run mapped, then bound, against the run bound: alike where each value the run returns binds alike. -/
theorem map_bind_go_quiet {α γ β : Type} (F : α → γ) {x : V α} {y : PS α} (hq : quietS.R x y) (f : γ → PS β)
    (g : α → PS β) (l : Array Claims.Msg) (h : ∀ a, x = .ok a → (f (F a)).go l = (g a).go l) :
    ((F <$> y) >>= f).go l = (y >>= g).go l := by
  rw [go_bind, go_bind, go_map', hq l]
  cases x with
  | error _ => rfl
  | ok a => exact h a rfl

theorem quiet_bind_ok {α β : Type} {x : V α} {y : PS α} (hq : quietS.R x y) {g : α → PS β}
    {l l' : Array Claims.Msg} {b : β} (h : (y >>= g).go l = (.ok b, l')) : ∃ a, x = .ok a ∧ (g a).go l = (.ok b, l') := by
  rw [go_bind, hq l] at h
  cases x with
  | error _ => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- **The link message through `asksCut`**: recorded, then answered by round `0`'s coin unless the session stops
there. -/
theorem asksCut_link_go (coin : ℕ → ℕ → F128) (n : ℕ) (a b : Array ByteArray) (l : Array Claims.Msg) :
    ((asksCut coin n).link a b).go l =
      (if n ≤ l.size then .error "past the openings" else .ok (coin l.size),
        l.push (FlockDeploy.linkBody a.toList b.toList)) := by
  have e : ((asksCut coin n).link a b).go l =
      (if n ≤ l.size then (throw "past the openings" : PS (ℕ → F128)) else pure (coin l.size)).go
        (l.push (FlockDeploy.linkBody a.toList b.toList)) := rfl
  rw [e]
  split <;> rfl

/-- **Two loops in step**: where each body runs as the other's from every state the invariant `I` holds at, and the
right body's next state keeps it, the loops run alike. -/
theorem forIn_go_eq {α β : Type} (xs : List α) (f g : α → β → PS (ForInStep β))
    (I : ℕ → β → Array Claims.Msg → Prop)
    (hstep : ∀ k (hk : k < xs.length) b l, I k b l → (f xs[k] b).go l = (g xs[k] b).go l)
    (hinv : ∀ k (hk : k < xs.length) b l b' l', I k b l → (g xs[k] b).go l = (.ok (.yield b'), l') →
      I (k + 1) b' l')
    (b : β) (l : Array Claims.Msg) (h0 : I 0 b l) : (forIn xs b f).go l = (forIn xs b g).go l := by
  induction xs generalizing I b l with
  | nil => rfl
  | cons a as ih =>
    have e0 : (f a b).go l = (g a b).go l := hstep 0 (by simp) b l h0
    rw [List.forIn_cons, List.forIn_cons, go_bind, go_bind, e0]
    rcases e : (g a b).go l with ⟨r, l'⟩
    cases r with
    | error _ => rfl
    | ok s =>
      cases s with
      | done b' => rfl
      | yield b' =>
        exact ih (fun k => I (k + 1)) (fun k hk b l hI => hstep (k + 1) (by simpa using hk) b l hI)
          (fun k hk b l b' l' hI h => hinv (k + 1) (by simpa using hk) b l b' l' hI h) b' l'
          (hinv 0 (by simp) b l b' l' h0 e)

theorem List.mapM_congr_mem {m : Type → Type} [Monad m] [LawfulMonad m] {α β : Type} {f g : α → m β} :
    ∀ {xs : List α}, (∀ a ∈ xs, f a = g a) → xs.mapM f = xs.mapM g
  | [], _ => rfl
  | a :: as, h => by
    rw [List.mapM_cons, List.mapM_cons, h a List.mem_cons_self,
      List.mapM_congr_mem fun b hb => h b (List.mem_cons_of_mem a hb)]

theorem Array.mapM_congr_mem {m : Type → Type} [Monad m] [LawfulMonad m] {α β : Type} {f g : α → m β}
    {xs : Array α} (h : ∀ a ∈ xs, f a = g a) : xs.mapM f = xs.mapM g := by
  rw [Array.mapM_eq_mapM_toList, Array.mapM_eq_mapM_toList,
    List.mapM_congr_mem fun a ha => h a (Array.mem_toList_iff.1 ha)]

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **`sessionCut` from the link message's coin `c1` on** (`afterLinkPZk`'s), level 0's roots `rootB` sent. -/
def afterLinkCut (io : Asks M) (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (words : Array (List Nat))
    (src : Nat → TabSrc) (schs : Array (Schedule × Nat)) (cms : Array Commit) (rootB : Array ByteArray)
    (c1 : Nat → F128) : M (String × Array ByteArray) := do
  let points := (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun j => c1 (p * spec.mPts + j)
  let mut streams : Array (Array Round) := #[]
  let mut proofs : Array ByteArray := #[]
  for (i, r, _) in spec.repStreams do
    let st := tabs[i]!
    let sch := schs[i]!.1
    let extras ← (monadLift (st.extra points) : M _)
    let lay := Zk.PadLayout.new st.m st.kLog sch.k0
    let ic ← padsTreeCut hs ((lay.len + Zk.K_PAD).nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE) ((src i).rep r)
    let dom := (spec.domain spec.tables[i]! r).toUTF8
    let (p, t) ← (repCut io hs st sch r dom cms[i]! extras ic ((src i).rep r) (priorOf st streams r) words[i]!).run
      (Transcript.init #[] dom)
    streams := streams.push t.rounds
    proofs := proofs.push (FlockSoundness.Discharge.ZkView.encZk (shapeZk st sch extras.size) p)
  let record ← recordJson hs spec rootB points streams proofs
  return (record, proofs)

/-- **The session from the source in three parts** (`sessionPZk_eq`'s): the schedules, level 0's commitments from the
leaves, the link message, and the rest. -/
theorem sessionCut_eq (io : Asks M) (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup)
    (words : Array (List Nat)) (src : Nat → TabSrc) :
    sessionCut io hs spec tabs words src = (do
      let schs ← schedsPZk tabs
      let cms ← (Array.range tabs.size).mapM fun i =>
        commit0Cut hs tabs[i]!.merkle (schs[i]!.1.levels.getD 0 default) (src i)
      let c1 ← io.link (cms.map fun c => flat c.cap) (tabs.map fun _ => ByteArray.empty)
      afterLinkCut io hs spec tabs words src schs cms (cms.map fun c => flat c.cap) c1) := rfl

omit [MonadExceptOf String M] [MonadLiftT V M] in
/-- **Level 0's commitment from a source with the leaves of another** is the other's with the source's rows and
salts. -/
theorem commit0Cut_leaf0 (hs : Hashes M) [LawfulMonad M] (ms : MerkleScheme) (lv : Level) {src src' : TabSrc}
    (h : src'.leaf0 = src.leaf0) :
    commit0Cut hs ms lv src' = (fun cm => { cm with
      rows := (Array.range (2 ^ lv.d)).map src'.row0
      salts := (Array.range (2 ^ lv.d)).map src'.salt0 }) <$> commit0Cut hs ms lv src := by
  unfold commit0Cut
  rw [h, map_bind]
  refine bind_congr fun _ => ?_
  rw [map_pure]

omit [MonadExceptOf String M] [MonadLiftT V M] in
/-- **A rep's pads commitment from a source with the leaves of another** is the other's with the source's columns and
salts. -/
theorem padsTreeCut_leaf (hs : Hashes M) [LawfulMonad M] (n : Nat) {src src' : RepSrc}
    (h : src'.padLeaf = src.padLeaf) :
    padsTreeCut hs n src' = (fun ic => { ic with
      ch := (Array.range n).map src'.ch
      cm := (Array.range n).map src'.cm
      salts := (Array.range n).map src'.padSalt }) <$> padsTreeCut hs n src := by
  unfold padsTreeCut
  rw [h, map_bind]
  refine bind_congr fun _ => ?_
  rw [map_pure]

end FlockComplete

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **`finExe` is the verdict's last message** of the session read from the views (`runCut` at `tabSrcView`). -/
theorem finExe_eq {x : Msg} {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (c : Fin (refK x) → FlockDeploy.Coin) (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.View) :
    finExe PE x c v =
      FlockComplete.ZkTables.finMsg x (runCut x c (refK x) spec (tabSrcView PE x (czExe PE x c) v)).1 := by
  unfold finExe
  rw [hsx]
  dsimp only
  generalize (runCut x c (refK x) spec (tabSrcView PE x (czExe PE x c) v)).1 = R
  rcases R with e | ⟨record, proofs⟩ <;> rfl

/-- **Where every table admits the opened coins, no outer coin is `1`**: after `refK x` openings `ops`, every
zerocheck outer coin among their answers (`outerSlotsZk`) is table `j`'s rep `r`'s outer coin `i` at the opened coins
(`outerAt`), which `czExe`'s `rOuter` reads (`PointReads`) and `Admits` keeps off `1`. -/
theorem outerHit_false {x : Msg} (hx : Live x) {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (ops : List (Msg × Msg)) (hl : ops.length = refK x)
    (hA : ∀ j, (tabC PE x j).Admits (czExe PE x (padOps ops) j)) :
    FlockComplete.ZkTables.outerHit x (ops.map fun p => coinOfD p.2) = false := by
  rw [← Bool.not_eq_true]
  intro h
  unfold FlockComplete.ZkTables.outerHit at h
  rw [List.any_eq_true] at h
  obtain ⟨⟨k, i⟩, hmem, hk⟩ := h
  unfold FlockComplete.outerSlotsZk at hmem
  rw [hsx] at hmem
  simp only [List.mem_flatMap, List.mem_range, List.mem_map, Prod.mk.injEq] at hmem
  obtain ⟨q, hq, i', hi', rfl, rfl⟩ := hmem
  rw [Bool.and_eq_true, decide_eq_true_iff, decide_eq_true_iff, List.length_map] at hk
  obtain ⟨hk, h1⟩ := hk
  have hJ : jOf x = (stsOf x).size := by
    unfold jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hlt : q / 2 < (stsOf x).size := by omega
  have hj : q / 2 < jOf x := hJ ▸ hlt
  have hr : q % 2 < 2 := Nat.mod_lt _ (by norm_num)
  have hq' : 2 * (q / 2) + q % 2 = q := Nat.div_add_mod q 2
  rw [getElem!_pos (stsOf x) (q / 2) hlt] at hi'
  have hm35 := ((FlockComplete.ZkTables.tables_of_setup hx.1 hsx).2.2.2 (q / 2) hlt).2.1
  have hslot : outerSlot x (q / 2) (q % 2) = FlockComplete.repStartZk spec (stsOf x) q + 1 := by
    unfold outerSlot
    rw [hsx, hq']
  have hval : outerAt (padOps (x := x) ops) (q / 2) (q % 2) i' = 1 := by
    have e : outerAt (padOps (x := x) ops) (q / 2) (q % 2) i' =
        slotAt (padOps (x := x) ops) (outerSlot x (q / 2) (q % 2)) i' := rfl
    rw [e, hslot, slotAt_padOps ops hl (by omega) (by omega)]
    exact h1
  exact (hA ⟨q / 2, hj⟩).2.2.2 ⟨q % 2, hr⟩ _ (List.mem_ofFn.2 ⟨⟨i', hi'⟩, rfl⟩) hval

/-- **The tables' real views at the opened coins `c`**, on `w`'s table witnesses and the tape `τ` (`viewsAt`; it reads
neither the rep rounds nor the round plan's encodings). -/
noncomputable def viewsE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) :
    ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.View :=
  (session (fun _ _ => 0) (tableData PE QE UE)).viewsAt x c (wtOf PE x w) τ

/-- **What the deployed session's last message reads from the views** (`tabSrcView`). -/
noncomputable def srcV (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) :
    ℕ → FlockComplete.TabSrc :=
  tabSrcView PE x (czExe PE x c) (viewsE x c w τ)

/-- **The session answered by the openings returns what the session answered by the opened coins does**, stopped after
the last round (`PrefixLive` at `n = refK x`, executable code only): after `refK x` openings `ops` whose coins every
table admits. -/
def LiveEnd : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (ω : FlockComplete.TapeZk) (ops : List (Msg × Msg)), ops.length = refK x →
    (∀ j, (tabC PE x j).Admits (czExe PE x (padOps ops) j)) →
    ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
      (liveRunAt x w ω ops spec).1 = (runExe x w ω (padOps (x := x) ops) (refK x) spec).1

/-- **The session read from the views returns the session's verdict** (`SimSession` at the last round, with the final
message's data): at a live statement and a witness in the deployed relation, at opened coins every table admits,
`sessionCut` on the views' source (`srcV`) returns what the masked session returns, both answered by the coins. -/
def SimFin : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin),
    (∀ j, (tabC PE x j).Admits (czExe PE x c j)) → ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
      (runCut x c (refK x) spec (srcV x c w τ)).1 = (runExe x w (UE.tapeZk τ) c (refK x) spec).1

/-- **`FinExe` from the last round's two sessions**: `finExe` of the views is `finMsg` of `runCut` on them
(`finExe_eq`), which is the session's verdict (`SimFin`), which is the live session's (`LiveEnd`), which `honestPR`
sends (`honestPR_finMsg`, no outer coin `1` by `outerHit_false`). Where `x` sets up nothing both are `[]`. -/
theorem finExe_of (hL : LiveEnd) (hS : SimFin) : FinExe := by
  intro H hH nr x hx w hR ν t ops hl hA
  show finExe PE x (padOps ops) (viewsE x (padOps ops) w t) =
    (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2)) (UE.tapeZk t)).eval H).1
  cases hsx0 : FlockDeploy.setupOf true x with
  | none =>
    have e1 : finExe PE x (padOps ops) (viewsE x (padOps ops) w t) = [] := by
      unfold finExe
      rw [hsx0]
    rw [e1]
    by_cases ho : FlockComplete.ZkTables.outerHit x ((ops.map fun p => (p.1, coinOfD p.2)).map Prod.snd) = true
    · rw [FlockComplete.ZkTables.honestPR_stop ho]
      rfl
    · rw [FlockComplete.ZkTables.honestPR_next (by simpa using ho)]
      show [] = ((FlockComplete.feed (FlockComplete.ZkTables.prog x _ _) _).eval H).1
      unfold FlockComplete.ZkTables.prog
      rw [hsx0]
      rfl
  | some v =>
    obtain ⟨s', spec', sts'⟩ := v
    have hst : stsOf x = sts' := by
      unfold stsOf
      rw [hsx0]
    have hpos : 0 < (stsOf x).size := by
      rw [hst]
      exact (FlockComplete.ZkTables.tables_of_setup hx.1 hsx0).1
    obtain ⟨s, spec, hsx⟩ := setupOf_of_pos hpos
    obtain ⟨-, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
    have hK := refK_eq hsx
    have hh : (ops.map fun p => (p.1, coinOfD p.2)).map Prod.snd = ops.map fun p => coinOfD p.2 := by
      rw [List.map_map]
      rfl
    rw [finExe_eq hsx, FlockComplete.ZkTables.honestPR_finMsg H x w _ hsx hspec
      (fun i hi => by have := (hall i hi).1; omega) _ _ hh (by rw [List.length_map]; omega)
      (outerHit_false hx hsx ops hl hA), hH]
    exact congrArg (FlockComplete.ZkTables.finMsg x)
      ((hS x hx w hR t (padOps ops) hA s spec hsx).trans (hL x hx w hR _ ops hl hA s spec hsx).symm)

/-! ## The commitments read from the views -/

/-- **The real views cut are the cut** (`cutV_viewsAt`). -/
theorem cutV_viewsE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) :
    (session (fun _ _ => 0) (tableData PE QE UE)).cutV x c (viewsE x c w τ) j = cutE x c w τ j :=
  (congrFun ((session (fun _ _ => 0) (tableData PE QE UE)).cutV_viewsAt x c (wtOf PE x w) τ) j).trans
    (congrFun (cutOf_eq (fun _ _ => 0) x c w τ) j)

/-- The views' leaves are the cut's: level 0's and each rep's pads tree's. -/
theorem leaves_viewsE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) :
    (viewCut (viewsE x c w τ j)).leaves0 = (cutE x c w τ j).leaves0 ∧
      (viewCut (viewsE x c w τ j)).leavesP = (cutE x c w τ j).leavesP := by
  have h1 := congrArg FlockSoundness.ZK.Table.Cut.leaves0 (cutV_viewsE x c w τ j)
  have h2 := congrArg FlockSoundness.ZK.Table.Cut.leavesP (cutV_viewsE x c w τ j)
  exact ⟨h1, h2⟩

theorem srcV_leaf0 (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) {i : ℕ}
    (hi : i < jOf x) : (srcV x c w τ i).leaf0 = (srcE x c w τ i).leaf0 := by
  have e1 : (srcV x c w τ i).leaf0 =
      leafAt PE x (czExe PE x c) (fun j => viewCut (viewsE x c w τ j)) ⟨i, hi⟩ := by
    simp only [srcV, tabSrcView, tabSrcCut, hi, ↓reduceDIte]
  have e2 : (srcE x c w τ i).leaf0 = leafAt PE x (czExe PE x c) (cutE x c w τ) ⟨i, hi⟩ := by
    simp only [srcE, tabSrcCut, hi, ↓reduceDIte]
  rw [e1, e2]
  unfold leafAt
  rw [(leaves_viewsE x c w τ ⟨i, hi⟩).1]

theorem srcV_padLeaf (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) {i r : ℕ}
    (hi : i < jOf x) (hr : r < 2) : ((srcV x c w τ i).rep r).padLeaf = ((srcE x c w τ i).rep r).padLeaf := by
  have e1 : ((srcV x c w τ i).rep r).padLeaf =
      padLeafCut PE x (czExe PE x c) (fun j => viewCut (viewsE x c w τ j)) ⟨i, hi⟩ ⟨r, hr⟩ := by
    simp only [srcV, tabSrcView, tabSrcCut, repSrcCut, hi, hr, ↓reduceDIte]
  have e2 : ((srcE x c w τ i).rep r).padLeaf = padLeafCut PE x (czExe PE x c) (cutE x c w τ) ⟨i, hi⟩ ⟨r, hr⟩ := by
    simp only [srcE, tabSrcCut, repSrcCut, hi, hr, ↓reduceDIte]
  rw [e1, e2]
  unfold padLeafCut
  rw [(leaves_viewsE x c w τ ⟨i, hi⟩).2]

/-- **Table `i`'s level-0 commitment as the views' source builds it**: the session's `cm` with the views' rows and
salts by tree position, `lv` its level. -/
noncomputable def fill0 (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) (i : ℕ)
    (lv : _root_.Flock.Level) (cm : FlockComplete.Commit) : FlockComplete.Commit :=
  { cm with rows := (Array.range (2 ^ lv.d)).map (srcV x c w τ i).row0,
            salts := (Array.range (2 ^ lv.d)).map (srcV x c w τ i).salt0 }

/-- **Table `i`'s rep `r` pads commitment as the views' source builds it**: the session's `ic` with the views' columns
and salts at its `n` leaves. -/
noncomputable def fillP (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (i r n : ℕ) (ic : FlockComplete.InnerCommit) : FlockComplete.InnerCommit :=
  { ic with ch := (Array.range n).map ((srcV x c w τ i).rep r).ch,
            cm := (Array.range n).map ((srcV x c w τ i).rep r).cm,
            salts := (Array.range n).map ((srcV x c w τ i).rep r).padSalt }

/-- **Level 0's commitment from the views** is the session's with the views' rows and salts (`SimCommit`, the views'
leaves the cut's). -/
theorem commit0Cut_srcV (hSC : SimCommit) {x : Msg} (hx : Live x) {w : Claims.Flock.Wit} (hR : RZk x w)
    (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) {i : ℕ} (hi : i < jOf x) :
    FlockComplete.commit0Cut hsE (FlockDeploy.tabsOf (stsOf x))[i]!.merkle ((zkTab x ⟨i, hi⟩).sch.levels.getD 0 default)
        (srcV x c w τ i) =
      fill0 x c w τ i ((zkTab x ⟨i, hi⟩).sch.levels.getD 0 default) <$> FlockComplete.commitPZk hsE
        (FlockDeploy.tabsOf (stsOf x))[i]!.merkle ((zkTab x ⟨i, hi⟩).sch.levels.getD 0 default) (zkTab x ⟨i, hi⟩).tPad
        (FlockComplete.packed (trsE x w (UE.tapeZk τ))[i]! (FlockDeploy.tabsOf (stsOf x))[i]!.m)
        ((UE.tapeZk τ).extra i) ((UE.tapeZk τ).lanePad i) ((UE.tapeZk τ).salt i 0 0) := by
  rw [FlockComplete.commit0Cut_leaf0 hsE _ _ (srcV_leaf0 x c w τ hi),
    FlockComplete.ext_go fun l => (hSC x hx w hR τ c ⟨i, hi⟩ l).1, Functor.map_map]
  rfl

/-- **Rep `r`'s pads commitment from the views** is the session's with the views' columns and salts. -/
theorem padsTreeCut_srcV (hSC : SimCommit) {x : Msg} (hx : Live x) {w : Claims.Flock.Wit} (hR : RZk x w)
    (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) {i r : ℕ} (hi : i < jOf x) (hr : r < 2) :
    FlockComplete.padsTreeCut hsE
        (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[i]!.m
          (FlockDeploy.tabsOf (stsOf x))[i]!.kLog (zkTab x ⟨i, hi⟩).sch.k0).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo *
          2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE) ((srcV x c w τ i).rep r) =
      fillP x c w τ i r
        (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[i]!.m
          (FlockDeploy.tabsOf (stsOf x))[i]!.kLog (zkTab x ⟨i, hi⟩).sch.k0).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo *
          2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE) <$>
        FlockComplete.innerCommitP hsE
          (FlockComplete.padsOf (UE.tapeZk τ) i r
            (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[i]!.m
              (FlockDeploy.tabsOf (stsOf x))[i]!.kLog (zkTab x ⟨i, hi⟩).sch.k0))
          ((UE.tapeZk τ).innerSalt i r) := by
  rw [FlockComplete.padsTreeCut_leaf hsE _ (srcV_padLeaf x c w τ hi hr),
    FlockComplete.ext_go fun l => (hSC x hx w hR τ c ⟨i, hi⟩ l).2 ⟨r, hr⟩, Functor.map_map]
  rfl

/-! ## The reps from the views -/

/-- The session's hashes in the verdict monad. -/
abbrev hsV : _root_.Flock.Hashes _root_.Flock.V := _root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash

/-- Table `i`'s rep `r`'s transcript domain. -/
def domE (spec : _root_.Flock.SessionSpec) (i r : ℕ) : ByteArray := (spec.domain spec.tables[i]! r).toUTF8

/-- Table `i`'s pads layout. -/
noncomputable def layE (x : Msg) (i : ℕ) (hi : i < jOf x) : _root_.Flock.Zk.PadLayout :=
  _root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[i]!.m (FlockDeploy.tabsOf (stsOf x))[i]!.kLog
    (zkTab x ⟨i, hi⟩).sch.k0

/-- **Table `i`'s rep `r` in the session** answered by the opened coins `c`, on level 0's commitment `cm0`, the extra
claims and the pads commitment `ic`, from a fresh transcript at rep `0`'s claim points (`priorE`). -/
noncomputable def repHE (x : Msg) (w : Claims.Flock.Wit) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin)
    (spec : _root_.Flock.SessionSpec) (i : ℕ) (hi : i < jOf x) (r : ℕ) (cm0 : FlockComplete.Commit)
    (extras : Array _root_.Flock.Claim) (ic : FlockComplete.InnerCommit) :
    FlockComplete.PS (_root_.Flock.Zk.ZkProof × _root_.Flock.Transcript) :=
  (FlockComplete.repPZk (ioE c (refK x)) hsE (FlockDeploy.tabsOf (stsOf x))[i]! (zkTab x ⟨i, hi⟩).sch
    (zkTab x ⟨i, hi⟩).tPad r (domE spec i r) (trsE x w (UE.tapeZk τ))[i]!
    (FlockComplete.packed (trsE x w (UE.tapeZk τ))[i]! (FlockDeploy.tabsOf (stsOf x))[i]!.m) cm0 extras
    (FlockComplete.padsOf (UE.tapeZk τ) i r (layE x i hi)) ic ((UE.tapeZk τ).extra i) ((UE.tapeZk τ).lanePad i)
    ((UE.tapeZk τ).salt i r) ((UE.tapeZk τ).nonce i r) (priorE c i r)
    (FlockComplete.MaskRank.masksOf (stsOf x))[i]!).run (_root_.Flock.Transcript.init #[] (domE spec i r))

/-- **The context a rep of the session runs in**: a live statement, a witness in the deployed relation, opened coins
every table admits, the session's level-0 commitment `cm0` of table `i`, its rep `r`'s pads commitment `ic` and the
statement's extra claims at the link points, and a history at the rep's first round (`repBase`). -/
def RepCtx (x : Msg) (w : Claims.Flock.Wit) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin)
    (spec : _root_.Flock.SessionSpec) (i : ℕ) (hi : i < jOf x) (r : ℕ) (cm0 : FlockComplete.Commit)
    (extras : Array _root_.Flock.Claim) (ic : FlockComplete.InnerCommit) (l : Array Msg) : Prop :=
  Live x ∧ RZk x w ∧ (∀ j, (tabC PE x j).Admits (czExe PE x c j)) ∧
    (∃ s, FlockDeploy.setupOf true x = some (s, spec, stsOf x)) ∧ r < 2 ∧
    FlockComplete.commitPZk hsV (FlockDeploy.tabsOf (stsOf x))[i]!.merkle ((zkTab x ⟨i, hi⟩).sch.levels.getD 0 default)
      (zkTab x ⟨i, hi⟩).tPad (FlockComplete.packed (trsE x w (UE.tapeZk τ))[i]! (FlockDeploy.tabsOf (stsOf x))[i]!.m)
      ((UE.tapeZk τ).extra i) ((UE.tapeZk τ).lanePad i) ((UE.tapeZk τ).salt i 0 0) = .ok cm0 ∧
    FlockComplete.innerCommitP hsV (FlockComplete.padsOf (UE.tapeZk τ) i r (layE x i hi))
      ((UE.tapeZk τ).innerSalt i r) = .ok ic ∧
    (FlockDeploy.tabsOf (stsOf x))[i]!.extra (ptsE c spec) = .ok extras ∧ l.size = repBase x (2 * i + r)

/-- **A rep read from the views is the session's rep** (`SimSession`'s rep at the last round, with what only the final
message reads): in a rep's context, `repCut` on the views' source, with level 0's and the pads' commitments the session's
but their rows, columns and salts the views' (`fill0`, `fillP`), runs as the session's rep. -/
def RepFin : Prop :=
  ∀ x w τ c spec i hi r cm0 extras ic l, RepCtx x w τ c spec i hi r cm0 extras ic l →
    ((FlockComplete.repCut (ioE c (refK x)) hsE (FlockDeploy.tabsOf (stsOf x))[i]! (zkTab x ⟨i, hi⟩).sch r
        (domE spec i r) (fill0 x c w τ i ((zkTab x ⟨i, hi⟩).sch.levels.getD 0 default) cm0) extras
        (fillP x c w τ i r (((layE x i hi).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo *
          2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE) ic) ((srcV x c w τ i).rep r) (priorE c i r)
        (FlockComplete.MaskRank.masksOf (stsOf x))[i]!).run (_root_.Flock.Transcript.init #[] (domE spec i r))).go l =
      (repHE x w τ c spec i hi r cm0 extras ic).go l

/-- **A rep that returns sends its rounds**: in a rep's context, where the session's rep returns, the history is at the
next rep's first round (`SimSession`'s rep count). -/
def RepExact : Prop :=
  ∀ x w τ c spec i hi r cm0 extras ic l, RepCtx x w τ c spec i hi r cm0 extras ic l →
    ∀ p t l', (repHE x w τ c spec i hi r cm0 extras ic).go l = (.ok (p, t), l') → l'.size = repBase x (2 * i + r + 1)

/-- **Rep `0`'s rounds hold its claim points at the opened coins** (`ptsOpen`; `SimSession`'s): in rep `0`'s context,
where the session's rep returns, its rounds' claim points are `ptsOpen`'s. -/
def RepPoints : Prop :=
  ∀ x w τ c spec i hi cm0 extras ic l, RepCtx x w τ c spec i hi 0 cm0 extras ic l →
    ∀ p t l', (repHE x w τ c spec i hi 0 cm0 extras ic).go l = (.ok (p, t), l') →
      FlockComplete.MaskRank.roundPoints (FlockDeploy.tabsOf (stsOf x))[i]!.m (FlockDeploy.tabsOf (stsOf x))[i]!.kLog
        t.rounds = ptsOpen c (stsOf x)[i]! (2 * i)

/-! ## The session from the views -/

theorem commitPZk_quiet (ms : _root_.Flock.MerkleScheme) (lv : _root_.Flock.Level) (tPad : ℕ)
    (g : Array _root_.Flock.F128) (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ByteArray) :
    FlockComplete.quietS.R (FlockComplete.commitPZk hsV ms lv tPad g extra lanePad salt)
      (FlockComplete.commitPZk hsE ms lv tPad g extra lanePad salt) :=
  FlockComplete.commitRowsP_quiet _ ms lv _ salt

theorem ioE_link_go {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (a b : Array ByteArray)
    (l : Array Msg) :
    ((ioE c n).link a b).go l =
      (if n ≤ l.size then .error "past the openings" else .ok (fun i => slotAt c l.size i),
        l.push (FlockDeploy.linkBody a.toList b.toList)) :=
  FlockComplete.asksCut_link_go _ n a b l

theorem tabs_size {x : Msg} (hx : Live x) : (FlockDeploy.tabsOf (stsOf x)).size = jOf x := by
  have hJ : jOf x = (stsOf x).size := by
    unfold jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  rw [hJ]
  simp [FlockDeploy.tabsOf]

/-- **Level 0's commitments from the views**, from any history: they send nothing and return the session's
commitments `cms` with the views' rows and salts (`commit0Cut_srcV`; quiet as `commitPZk` is). -/
theorem commits_srcV (hSC : SimCommit) {x : Msg} (hx : Live x) {w : Claims.Flock.Wit} (hR : RZk x w)
    (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hschs : ∀ j : Fin (jOf x), schs[j.val]! = ((zkTab x j).sch, (zkTab x j).tPad))
    {cms : Array FlockComplete.Commit}
    (hcms : FlockComplete.commitsPZk hsV (FlockDeploy.tabsOf (stsOf x)) schs
      (FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x)) (trsE x w (UE.tapeZk τ))) (UE.tapeZk τ) = .ok cms)
    (l : Array Msg) :
    ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM (fun i => FlockComplete.commit0Cut hsE
        (FlockDeploy.tabsOf (stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default) (srcV x c w τ i))).go l =
      (.ok ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).map fun i =>
        fill0 x c w τ i (schs[i]!.1.levels.getD 0 default) cms[i]!), l) := by
  have hn := tabs_size hx
  obtain ⟨hcs, hcm⟩ := FlockSoundness.Discharge.Exec.range_mapM_ok hcms
  have hq : FlockComplete.quietS.R
      ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM fun i =>
        fill0 x c w τ i (schs[i]!.1.levels.getD 0 default) <$>
          FlockComplete.commitPZk hsV (FlockDeploy.tabsOf (stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
            schs[i]!.2 (FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x)) (trsE x w (UE.tapeZk τ)))[i]!
            ((UE.tapeZk τ).extra i) ((UE.tapeZk τ).lanePad i) ((UE.tapeZk τ).salt i 0 0))
      ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM fun i =>
        fill0 x c w τ i (schs[i]!.1.levels.getD 0 default) <$>
          FlockComplete.commitPZk hsE (FlockDeploy.tabsOf (stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
            schs[i]!.2 (FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x)) (trsE x w (UE.tapeZk τ)))[i]!
            ((UE.tapeZk τ).extra i) ((UE.tapeZk τ).lanePad i) ((UE.tapeZk τ).salt i 0 0)) :=
    _root_.Flock.With.Sim.mapM_array FlockComplete.quietS _ fun i =>
      FlockComplete.quiet_map _ (commitPZk_quiet _ _ _ _ _ _ _)
  rw [FlockComplete.Array.mapM_congr_mem fun i hi => ?_, hq l, FlockComplete.mapM_ok_mem _ _ fun i hi => ?_]
  · have hi' : i < cms.size := by rw [hcs]; simpa using hi
    show fill0 x c w τ i _ <$> _ = _
    rw [hcm i hi', getElem!_pos cms i hi']
    rfl
  · have hi' : i < jOf x := by rw [← hn]; simpa using hi
    have e : schs[i]! = ((zkTab x ⟨i, hi'⟩).sch, (zkTab x ⟨i, hi'⟩).tPad) := hschs ⟨i, hi'⟩
    rw [e, FlockComplete.tablesP, FlockSoundness.Refine.getElem!_range_map _ (by rw [hn]; exact hi')]
    exact commit0Cut_srcV hSC hx hR τ c hi'

/-- **`SimFin` from the commitments and the reps**: the session read from the views runs as the masked session,
part by part: the schedules (quiet), level 0's commitments (the session's with the views' rows and salts,
`commits_srcV`, so the same caps), the link message, and rep by rep (`RepFin`), each rep at its first round
(`RepExact`) with rep `0`'s claim points ahead of rep `1` (`RepPoints`); then the same record. -/
theorem simFin_of (hSC : SimCommit) (hF : RepFin) (hE : RepExact) (hP : RepPoints) : SimFin := by
  intro x hx w hR τ c hA s spec hsx
  obtain ⟨-, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hn := tabs_size hx
  have hJ : jOf x = (stsOf x).size := by
    unfold jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hK := refK_eq hsx
  obtain ⟨schs, hs, hschs⟩ := schedsPZk_live hx
  have hms : ∀ i < (FlockDeploy.tabsOf (stsOf x)).size,
      (FlockDeploy.tabsOf (stsOf x))[i]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    intro i hi
    have hi' : i < (stsOf x).size := by simpa [FlockDeploy.tabsOf] using hi
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) i hi]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    exact (hall i hi').2.2
  obtain ⟨cms, hcm, hsize, -⟩ := FlockComplete.commitsPZk_caps (FlockDeploy.tabsOf (stsOf x)) schs
    (FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x)) (trsE x w (UE.tapeZk τ))) (UE.tapeZk τ) hms
  obtain ⟨-, hcm1⟩ := FlockSoundness.Discharge.Exec.range_mapM_ok hcm
  have hcm0 : ∀ i (hi : i < jOf x), FlockComplete.commitPZk hsV (FlockDeploy.tabsOf (stsOf x))[i]!.merkle
      ((zkTab x ⟨i, hi⟩).sch.levels.getD 0 default) (zkTab x ⟨i, hi⟩).tPad
      (FlockComplete.packed (trsE x w (UE.tapeZk τ))[i]! (FlockDeploy.tabsOf (stsOf x))[i]!.m)
      ((UE.tapeZk τ).extra i) ((UE.tapeZk τ).lanePad i) ((UE.tapeZk τ).salt i 0 0) = .ok cms[i]! := by
    intro i hi
    have hi' : i < cms.size := by rw [hsize, hn]; exact hi
    have h := hcm1 i hi'
    have e : schs[i]! = ((zkTab x ⟨i, hi⟩).sch, (zkTab x ⟨i, hi⟩).tPad) := hschs ⟨i, hi⟩
    rw [e, FlockComplete.tablesP, FlockSoundness.Refine.getElem!_range_map _ (by rw [hn]; exact hi)] at h
    rw [getElem!_pos cms i hi']
    exact h
  have hroot : ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).map fun i =>
      fill0 x c w τ i (schs[i]!.1.levels.getD 0 default) cms[i]!).map (fun cm => FlockComplete.flat cm.cap) =
      cms.map fun cm => FlockComplete.flat cm.cap := by
    refine Array.ext (by simp [hsize]) fun i h1 _ => ?_
    simp only [Array.getElem_map, Array.getElem_range]
    rw [getElem!_pos cms i (by simpa [hsize] using h1)]
    rfl
  refine congrArg Prod.fst ?_
  unfold runCut runExe
  rw [show FlockComplete.asksCut (fun k i => slotAt c k i) (refK x) = ioE c (refK x) from rfl,
    FlockComplete.sessionCut_eq, FlockComplete.sessionPZk_eq, FlockComplete.go_bind, FlockComplete.go_bind,
    FlockComplete.schedsPZk_quiet, hs]
  dsimp only
  rw [FlockComplete.go_bind, FlockComplete.go_bind, commits_srcV hSC hx hR τ c hschs hcm,
    FlockComplete.commitsPZk_quiet, hcm]
  dsimp only
  rw [hroot, FlockComplete.go_bind, FlockComplete.go_bind, ioE_link_go, ite_eq_right_of_eq_false _ _ (eq_false (by rw [hK]; simp))]
  dsimp only
  unfold FlockComplete.afterLinkCut FlockComplete.afterLinkPZk
  dsimp only
  refine FlockComplete.bind_go_congr ?_ _
  rw [← Array.forIn_toList, ← Array.forIn_toList, FlockDeploy.repStreams_toList]
  refine FlockComplete.forIn_go_eq _ _ _ (fun k b l => l.size = repBase x k ∧ (k % 2 = 1 →
      FlockComplete.MaskRank.roundPoints (FlockDeploy.tabsOf (stsOf x))[k / 2]!.m
        (FlockDeploy.tabsOf (stsOf x))[k / 2]!.kLog b.1.back! = ptsOpen c (stsOf x)[k / 2]! (2 * (k / 2))))
    ?_ ?_ _ _ ⟨?_, fun h => absurd h (by norm_num)⟩
  · intro k hk b l hI
    have hk' : k < 2 * (stsOf x).size := by simpa [hspec] using hk
    have hi : k / 2 < jOf x := by rw [hJ]; omega
    have hr : k % 2 < 2 := Nat.mod_lt _ (by norm_num)
    have esch : schs[k / 2]! = ((zkTab x ⟨k / 2, hi⟩).sch, (zkTab x ⟨k / 2, hi⟩).tPad) := hschs ⟨k / 2, hi⟩
    have hpri : FlockComplete.priorOf (FlockDeploy.tabsOf (stsOf x))[k / 2]! b.1 (k % 2) = priorE c (k / 2) (k % 2) := by
      unfold FlockComplete.priorOf priorE
      by_cases h1 : k % 2 = 1
      · rw [ite_eq_left_of_eq_true _ _ (eq_true h1), ite_eq_left_of_eq_true _ _ (eq_true h1)]
        exact hI.2 h1
      · rw [ite_eq_right_of_eq_false _ _ (eq_false h1), ite_eq_right_of_eq_false _ _ (eq_false h1)]
    simp only [List.getElem_map, List.getElem_range]
    rw [esch]
    dsimp only
    refine FlockComplete.liftV_bind_go _ _ _ _ fun extras hxe => ?_
    rw [padsTreeCut_srcV hSC hx hR τ c hi hr]
    refine FlockComplete.map_bind_go_quiet _ (FlockComplete.innerCommitP_quiet _ _ _) _ _ _ fun ic hic => ?_
    refine FlockComplete.bind_go_congr ?_ _
    rw [FlockSoundness.Refine.getElem!_range_map _ (by rw [hn]; exact hi), esch, FlockComplete.tablesP,
      FlockSoundness.Refine.getElem!_range_map _ (by rw [hn]; exact hi), hpri]
    dsimp only
    exact hF x w τ c spec (k / 2) hi (k % 2) cms[k / 2]! extras ic l
      ⟨hx, hR, hA, ⟨s, hsx⟩, hr, hcm0 _ hi, hic, hxe, by rw [hI.1, Nat.div_add_mod]⟩
  · intro k hk b l b' l' hI h
    have hk' : k < 2 * (stsOf x).size := by simpa [hspec] using hk
    have hi : k / 2 < jOf x := by rw [hJ]; omega
    have hr : k % 2 < 2 := Nat.mod_lt _ (by norm_num)
    have esch : schs[k / 2]! = ((zkTab x ⟨k / 2, hi⟩).sch, (zkTab x ⟨k / 2, hi⟩).tPad) := hschs ⟨k / 2, hi⟩
    have hpri : FlockComplete.priorOf (FlockDeploy.tabsOf (stsOf x))[k / 2]! b.1 (k % 2) = priorE c (k / 2) (k % 2) := by
      unfold FlockComplete.priorOf priorE
      by_cases h1 : k % 2 = 1
      · rw [ite_eq_left_of_eq_true _ _ (eq_true h1), ite_eq_left_of_eq_true _ _ (eq_true h1)]
        exact hI.2 h1
      · rw [ite_eq_right_of_eq_false _ _ (eq_false h1), ite_eq_right_of_eq_false _ _ (eq_false h1)]
    simp only [List.getElem_map, List.getElem_range] at h
    rw [esch] at h
    dsimp only at h
    obtain ⟨extras, hxe, h⟩ := FlockComplete.liftV_bind_go_ok h
    obtain ⟨ic, hic, h⟩ := FlockComplete.quiet_bind_ok (FlockComplete.innerCommitP_quiet _ _ _) h
    obtain ⟨⟨p, t⟩, l'', hrep, h⟩ := FlockComplete.JsonRT.go_bind_ok h
    simp only [FlockComplete.JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq, ForInStep.yield.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    rw [FlockComplete.tablesP, FlockSoundness.Refine.getElem!_range_map _ (by rw [hn]; exact hi), hpri] at hrep
    have hctx : RepCtx x w τ c spec (k / 2) hi (k % 2) cms[k / 2]! extras ic l :=
      ⟨hx, hR, hA, ⟨s, hsx⟩, hr, hcm0 _ hi, hic, hxe, by rw [hI.1, Nat.div_add_mod]⟩
    have hsz := hE x w τ c spec (k / 2) hi (k % 2) _ _ _ l hctx p t l'' hrep
    refine ⟨by rw [hsz, Nat.div_add_mod], fun h1 => ?_⟩
    have h0 : k % 2 = 0 := by omega
    have hk2 : (k + 1) / 2 = k / 2 := by omega
    rw [hk2]
    show FlockComplete.MaskRank.roundPoints _ _ (b.1.push t.rounds).back! = _
    rw [Array.back!_push]
    rw [h0] at hctx hrep
    exact hP x w τ c spec (k / 2) hi _ _ _ l hctx p t l'' hrep
  · rw [repBase_eq hsx]
    simp [FlockComplete.repStartZk]

/-- **`FinExe` from the reps read from the views** (`finExe_of`, `simFin_of`): what is left is the live session's
last round (`LiveEnd`), the commitments from the leaves (`SimCommit`) and, per rep, the rep from the views
(`RepFin`), its round count (`RepExact`) and rep `0`'s claim points (`RepPoints`). -/
theorem finExe_of_reps (hL : LiveEnd) (hSC : SimCommit) (hF : RepFin) (hE : RepExact) (hP : RepPoints) :
    FinExe :=
  finExe_of hL (simFin_of hSC hF hE hP)

/-! ## Where every table admits, the views' source is the cut's with the final message's data -/

/-- **Every check of the session passes** where every table admits the opened coins (`coin_facts`): the rank check over
any claims, each `W^r` bijective and each `β ≠ 0`. -/
theorem famGuard_admits {x : Msg} {c : Fin (refK x) → FlockDeploy.Coin}
    (hA : ∀ j, (tabC PE x j).Admits (czExe PE x c j)) (j : Fin (jOf x)) (r : Fin 2)
    (p : FlockSoundness.ZK.Session.Phase) :
    famGuard (fun j => ((tabC PE x j).T (czExe PE x c j)).tab) (fun j => ((tabC PE x j).T (czExe PE x c j)).rep)
      j r p := by
  obtain ⟨hrank, hW, hb⟩ := Composed.coin_facts (tabC PE x j) (czExe PE x c j) (hA j) (fun _ => rfl)
    (fun _ => ⟨rfl, rfl⟩) (fun _ _ => rfl) (fun _ => rfl)
  cases p
  · exact FlockSoundness.ZK.Table.Rank.rankOn _ hrank _
  · exact hW r
  · exact hb r

/-- **Where every table admits, the cut is the views' with every guarded message sent** (`famGuard_admits`). -/
theorem cutE_eq_viewCut {x : Msg} {c : Fin (refK x) → FlockDeploy.Coin}
    (hA : ∀ j, (tabC PE x j).Admits (czExe PE x c j)) (w : Claims.Flock.Wit) (τ : UE.Tape) (j : Fin (jOf x)) :
    cutE x c w τ j = viewCut (viewsE x c w τ j) := by
  rw [← cutV_viewsE]
  have hP : ∀ r p, famPasses (fun j => ((tabC PE x j).T (czExe PE x c j)).tab)
      (fun j => ((tabC PE x j).T (czExe PE x c j)).rep) j r p :=
    fun _ _ j' r' p' _ => famGuard_admits hA j' r' p'
  show FlockSoundness.ZK.Table.cut _ (famSent (fun j => ((tabC PE x j).T (czExe PE x c j)).tab)
    (fun j => ((tabC PE x j).T (czExe PE x c j)).rep) j) (viewPre _ (viewsE x c w τ j)) = _
  simp only [FlockSoundness.ZK.Table.cut, famSent, hP, ite_true, viewCut, viewPre]

/-- A rep's source without what only the final message reads: `τ`, its salt, and the pads' columns and salts. -/
def eraseS (s : FlockComplete.RepSrc) : FlockComplete.RepSrc :=
  { s with tau := 0, tauSalt := ByteArray.empty, ch := fun _ => 0, cm := fun _ => 0,
           padSalt := fun _ => ByteArray.empty }

theorem eraseS_upd (s : FlockComplete.RepSrc) (a : _root_.Flock.F128) (b : ByteArray) (f g : ℕ → _root_.Flock.F128)
    (k : ℕ → ByteArray) (h0 : s.tau = 0) (h1 : s.tauSalt = ByteArray.empty) (h2 : s.ch = fun _ => 0)
    (h3 : s.cm = fun _ => 0) (h4 : s.padSalt = fun _ => ByteArray.empty) :
    eraseS { s with tau := a, tauSalt := b, ch := f, cm := g, padSalt := k } = s := by
  cases s
  simp only at h0 h1 h2 h3 h4
  subst h0 h1 h2 h3 h4
  rfl

/-- **Where every table admits, a rep's source from the views is the cut's** but for what only the final message
reads (`cutE_eq_viewCut`). -/
theorem eraseS_srcV {x : Msg} {c : Fin (refK x) → FlockDeploy.Coin}
    (hA : ∀ j, (tabC PE x j).Admits (czExe PE x c j)) (w : Claims.Flock.Wit) (τ : UE.Tape) {i r : ℕ}
    (hi : i < jOf x) (hr : r < 2) : eraseS ((srcV x c w τ i).rep r) = (srcE x c w τ i).rep r := by
  have e : (fun j => viewCut (viewsE x c w τ j)) = cutE x c w τ :=
    funext fun j => (cutE_eq_viewCut hA w τ j).symm
  show eraseS ((tabSrcView PE x (czExe PE x c) (viewsE x c w τ) i).rep r) =
    (tabSrcCut PE x (czExe PE x c) (cutE x c w τ) i).rep r
  generalize viewsE x c w τ = v at e ⊢
  generalize cutE x c w τ = cu at e ⊢
  subst e
  unfold tabSrcView tabSrcCut
  rw [dite_eq_left hi, dite_eq_left hi]
  dsimp only
  rw [dite_eq_left hr, dite_eq_left hr]
  exact eraseS_upd _ _ _ _ _ _ rfl rfl rfl rfl rfl

end Masked

end Claims.ZKMal
