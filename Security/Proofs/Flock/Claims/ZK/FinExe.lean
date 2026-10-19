import Proofs.Flock.Claims.ZK.Split
import Proofs.Flock.Claims.ZK.SimPiop
import Proofs.Flock.Claims.Complete.Zk.Level0Opens

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

* **`SimFin` rep by rep** (`simFin_of`, `finExe_of_reps`): the commitments from the views' leaves (`SimCommit`), and
  each rep read from the views runs as the session's (`RepFin`), sends its rounds (`RepExact`) and, at rep `0`, holds its
  claim points at the opened coins (`RepPoints`).
* **A rep read from the views** (`repFin_of`): where every table admits, no coin plan refuses (`not_refused_of_admits`),
  so the hybrids `SimPiop`, `SimLig`, `SimInner` make the rep from the cut the session's up to `eraseZ`
  (`repCut_erased`), and the views' commitments and source change only what `eraseZ` forgets (`FinRep.repCut_eraseC`,
  `FinRep.repCut_eraseIS`); the rest is the final-only data (`RepFinal`). `τ`'s salt is the tape's in both
  (`srcV_tauSalt`, `repFinal_of`), leaving level 0's opened rows and salts, `τ`, and the pads' opened columns and salts
  (`RepOpenFinal`).
* **The rep count and claim points** (`repExact`, `repPoints`): the run answered by `asksCut` is the run answered by
  `asksS` at the coins' bytes (`FinRep.repPZk_cut`), whose rounds `repPZk_exact` and `repPZk_points` count.

`finExe_of_final` is `FinExe` from `LiveEnd`, `SimCommit`, the hybrids and `RepOpenFinal`. The rep from the views sends
its source's `τ`, the view's (`srcV_tau`), so `RepOpenFinal` is the session's `τ` the view's (`RepTau`, `InnerTauVal`'s
reading at the last message) and the four fields at opened positions (`RepOpenRest`): `finExe_of_rest`.
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
theorem finAsksCut_link_go (coin : ℕ → ℕ → F128) (n : ℕ) (a b : Array ByteArray) (l : Array Claims.Msg) :
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
theorem finSessionCut_eq (io : Asks M) (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup)
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
noncomputable def finLayE (x : Msg) (i : ℕ) (hi : i < jOf x) : _root_.Flock.Zk.PadLayout :=
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
    (FlockComplete.padsOf (UE.tapeZk τ) i r (finLayE x i hi)) ic ((UE.tapeZk τ).extra i) ((UE.tapeZk τ).lanePad i)
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
    FlockComplete.innerCommitP hsV (FlockComplete.padsOf (UE.tapeZk τ) i r (finLayE x i hi))
      ((UE.tapeZk τ).innerSalt i r) = .ok ic ∧
    (FlockDeploy.tabsOf (stsOf x))[i]!.extra (ptsE c spec) = .ok extras ∧ l.size = repBase x (2 * i + r)

/-- **A rep read from the views is the session's rep** (`SimSession`'s rep at the last round, with what only the final
message reads): in a rep's context, `repCut` on the views' source, with level 0's and the pads' commitments the session's
but their rows, columns and salts the views' (`fill0`, `fillP`), runs as the session's rep. -/
def RepFin : Prop :=
  ∀ x w τ c spec i hi r cm0 extras ic l, RepCtx x w τ c spec i hi r cm0 extras ic l →
    ((FlockComplete.repCut (ioE c (refK x)) hsE (FlockDeploy.tabsOf (stsOf x))[i]! (zkTab x ⟨i, hi⟩).sch r
        (domE spec i r) (fill0 x c w τ i ((zkTab x ⟨i, hi⟩).sch.levels.getD 0 default) cm0) extras
        (fillP x c w τ i r (((finLayE x i hi).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo *
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
  FlockComplete.finAsksCut_link_go _ n a b l

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
    FlockComplete.finSessionCut_eq, FlockComplete.sessionPZk_eq, FlockComplete.go_bind, FlockComplete.go_bind,
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

/-- **`maskRankOk_compose`'s converse**: where the plan's points `pts`, `A`'s, have full rank over the words `ws` read
with `getD`, the executable's rank check passes on `A`. -/
theorem maskRankOk_of_composed {A : Array (List _root_.Flock.F128)} {ws : List ℕ} {k : ℕ}
    {pts : Fin k → List _root_.Flock.F128} (hk : A.size = k) (hp : ∀ i : Fin k, pts i = A[i.val]!)
    (h : Composed.MaskRankOk pts fun x : Fin ws.length => ws.getD x.val 0) :
    FlockComplete.MaskRank.maskRankOk A ws = true := by
  subst hk
  unfold FlockComplete.MaskRank.maskRankOk
  simp only [decide_eq_true_eq]
  intro a
  obtain ⟨u, hu⟩ := h fun i => FlockLevel3.GF128.of (a i)
  refine ⟨fun x => decide (u x = 1), fun i => ?_⟩
  apply (show Function.Injective FlockLevel3.GF128.of from fun _ _ h => h)
  rw [FlockComplete.MaskRank.of_sumSel, ← congrFun hu i]
  refine Finset.sum_congr rfl fun x _ => ?_
  dsimp only
  rw [hp i, getElem!_pos A i.val i.isLt, List.getD_eq_getElem _ _ x.isLt]
  rcases (by decide : ∀ z : ZMod 2, z = 0 ∨ z = 1) (u x) with h0 | h1
  · rw [h0]
    simp
  · rw [h1]
    simp

/-- **Full rank survives dropping points**: the rank check passing on `A ++ B` passes on `A`. -/
theorem maskRankOk_of_append {A B : Array (List _root_.Flock.F128)} {ws : List ℕ}
    (h : FlockComplete.MaskRank.maskRankOk (A ++ B) ws = true) : FlockComplete.MaskRank.maskRankOk A ws = true := by
  unfold FlockComplete.MaskRank.maskRankOk at h ⊢
  simp only [decide_eq_true_eq] at h ⊢
  intro a
  obtain ⟨f, hf⟩ := h fun i => if hi : i.val < A.size then a ⟨i.val, hi⟩ else 0
  refine ⟨f, fun i => ?_⟩
  have e := hf ⟨i.val, by rw [Array.size_append]; omega⟩
  rw [dite_eq_left i.isLt] at e
  refine Eq.trans ?_ e
  congr 1
  funext j
  simp only [Fin.getElem_fin]
  rw [Array.getElem_append_left i.isLt]

/-- **Where every table admits, no refusal shows** (`Refused`): every outer coin, read by the plan at its slot
(`pointReads_czExe`), is not `1`; every `β` and pair, read at their rounds (`coinReads_czExe`), passes; and every rank
check passes, the plan's points being the executable's (`pointsOf_reads`) and full rank of all four giving full rank of
rep `0`'s two (`maskRankOk_of_composed`, `maskRankOk_of_append`). -/
theorem not_refused_of_admits {x : Msg} (hx : Live x) {c : Fin (refK x) → FlockDeploy.Coin}
    (hA : ∀ j, (tabC PE x j).Admits (czExe PE x c j)) (n : ℕ) : ¬ Refused x c n := by
  have hJ : jOf x = (stsOf x).size := by
    unfold jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  rintro (⟨j, hj, r, hr, -, e, he, h1⟩ | ⟨j, hj, r, hr, ⟨-, -, -, -, hbad⟩ | ⟨-, hρ⟩ | ⟨-, hβ⟩⟩)
  · have hjJ : j < jOf x := hJ ▸ hj
    rw [getElem!_pos (stsOf x) j hj] at he
    refine (hA ⟨j, hjJ⟩).2.2.2 ⟨r, hr⟩ _ (List.mem_ofFn.2 ⟨⟨e, he⟩, ?_⟩) h1
    rw [(pointReads_czExe PE x c ⟨j, hjJ⟩ ⟨r, hr⟩).2.2]
  · have hjJ : j < jOf x := hJ ▸ hj
    obtain ⟨s, spec, hsx⟩ := setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
    have hfull : FlockComplete.MaskRank.maskRankOk (ptsOpen c (stsOf x)[j] (2 * j) ++
        ptsOpen c (stsOf x)[j] (2 * j + 1)) (Composed.maskWordsOf (stsOf x)[j]) = true := by
      refine maskRankOk_of_composed (k := _root_.Flock.Zk.RANK_POINTS) (by rfl) (fun k => ?_) ((hA ⟨j, hjJ⟩).2.1)
      refine pointsOf_reads ((czExe PE x c ⟨j, hjJ⟩).rest :
          Composed.RestOf (zkTab x ⟨j, hjJ⟩).table (stAt x ⟨j, hjJ⟩))
        (fun r n e => slotAt c (repBase x (2 * j + r) + n) e)
        (fun r => (pointReads_czExe PE x c ⟨j, hjJ⟩ r).1) (fun r => (pointReads_czExe PE x c ⟨j, hjJ⟩ r).2.1)
        (fun r => ?_) k
      rw [(pointReads_czExe PE x c ⟨j, hjJ⟩ r).2.2]
      funext i
      unfold outerAt slotAt
      rw [outerSlot_eq hsx]
    unfold RankBad at hbad
    rw [getElem!_pos (stsOf x) j hj] at hbad
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · rw [ite_eq_right_of_eq_false _ _ (eq_false (by norm_num)), Array.empty_append, Nat.add_zero,
        maskRankOk_of_append hfull] at hbad
      cases hbad
    · rw [ite_eq_left_of_eq_true _ _ (eq_true rfl), hfull] at hbad
      cases hbad
  · have hjJ : j < jOf x := hJ ▸ hj
    rw [← (coinReads_czExe PE x c ⟨j, hjJ⟩ ⟨r, hr⟩).2, (hA ⟨j, hjJ⟩).2.2.1 ⟨r, hr⟩] at hρ
    cases hρ
  · have hjJ : j < jOf x := hJ ▸ hj
    rw [← (coinReads_czExe PE x c ⟨j, hjJ⟩ ⟨r, hr⟩).1, (hA ⟨j, hjJ⟩).1 ⟨r, hr⟩] at hβ
    cases hβ

/-! ## A rep from the cut against the session's, its final-only data forgotten

`SimSession`'s hybrid chain (`repMix 0` is the cut's rep at the erased commitments, `repMix 3` the session's rep, both
with what only the final message reads forgotten), restated here so that this file imports `Split` alone. -/

namespace FinRep

open FlockComplete

theorem eraseC_cap (cm : Commit) : (eraseC cm).cap = cm.cap := rfl

theorem eraseIC_root (ic : InnerCommit) : (eraseIC ic).root = ic.root := rfl

theorem innerCut_eraseIC (io : Asks FlockComplete.PS) (lay : _root_.Flock.Zk.PadLayout) (ic : InnerCommit)
    (finalC : _root_.Flock.F128) (src : RepSrc) (cons : Array (String × _root_.Flock.Zk.Lin)) :
    eraseIP <$> innerCut io lay (eraseIC ic) finalC src cons = eraseIP <$> innerCut io lay ic finalC src cons := by
  unfold innerCut
  simp only [map_bind, map_pure, apply_ite (Functor.map (f := _root_.Flock.TMW FlockComplete.PS) eraseIP),
    eraseIP, eraseIC, Array.size_map, Array.map_map, Function.comp_def]

theorem bind_eraseZ {xa xb : _root_.Flock.TMW FlockComplete.PS _root_.Flock.Zk.InnerProof}
    (h : eraseIP <$> xa = eraseIP <$> xb) (p : _root_.Flock.Proof) (zl : _root_.Flock.Zk.Level0Msgs) :
    (xa >>= fun inner => pure (eraseZ { proof := p, level0 := zl, inner })) =
      (xb >>= fun inner => pure (eraseZ { proof := p, level0 := zl, inner })) := by
  have e : ∀ xc : _root_.Flock.TMW FlockComplete.PS _root_.Flock.Zk.InnerProof,
      (xc >>= fun inner => pure (eraseZ { proof := p, level0 := zl, inner })) =
        (fun ip => ({ proof := { p with ligerito := eraseL p.ligerito }, level0 := zl, inner := ip } :
          _root_.Flock.Zk.ZkProof)) <$> (eraseIP <$> xc) := fun xc => by
    rw [← comp_map, ← bind_pure_comp]
    rfl
  rw [e, e, h]

theorem repCut_mix0 (io : Asks FlockComplete.PS) (hs : _root_.Flock.Hashes FlockComplete.PS) (st : _root_.Flock.Setup)
    (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (tr : FlockComplete.Trace) (t : Array _root_.Flock.F128)
    (cm0 : Commit) (extras : Array _root_.Flock.Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List _root_.Flock.F128)) (words : List ℕ) (src : RepSrc) :
    repMix 0 io hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words src =
      eraseZ <$> repCut io hs st sch rep dom (eraseC cm0) extras (eraseIC ic) src prior words := by
  unfold repMix repCut
  simp only [↓reduceIte, Nat.zero_le, eraseC_cap, eraseIC_root]
  rw [map_bind]; refine bind_congr fun ⟨zp, lp, ss, T, sws⟩ => ?_; dsimp only
  rw [map_bind]; refine bind_congr fun ⟨lg, zl, rec⟩ => ?_; dsimp only
  rw [map_bind]; refine bind_congr fun tr' => ?_
  rw [map_bind]; refine bind_congr fun ⟨rp, _⟩ => ?_; dsimp only
  rw [map_bind]; refine bind_congr fun lgc => ?_
  rw [map_bind]
  simp only [map_pure]
  exact bind_eraseZ (innerCut_eraseIC _ _ _ _ _ _).symm _ _

theorem replay_eraseL (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout) (cap : Array ByteArray)
    (params : _root_.Flock.PcsParams) (zp : _root_.Flock.ZerocheckProof) (lp : _root_.Flock.LincheckProof)
    (ss : Array (Array _root_.Flock.F128)) (lg : _root_.Flock.LigeritoProof) (root : ByteArray) :
    _root_.Flock.Zk.replay st lay
        { cap, params, zerocheck := zp, lincheck := lp, ringSwitches := ss, ligerito := eraseL lg } root =
      _root_.Flock.Zk.replay st lay
        { cap, params, zerocheck := zp, lincheck := lp, ringSwitches := ss, ligerito := lg } root := by
  unfold _root_.Flock.Zk.replay
  dsimp only

theorem ligeritoConstraints_eraseL (lay : _root_.Flock.Zk.PadLayout) (rec : _root_.Flock.Zk.L0Rec)
    (lg : _root_.Flock.LigeritoProof) (zl : _root_.Flock.Zk.Level0Msgs) :
    _root_.Flock.Zk.ligeritoConstraints lay rec (eraseL lg) zl = _root_.Flock.Zk.ligeritoConstraints lay rec lg zl := by
  unfold _root_.Flock.Zk.ligeritoConstraints eraseL
  dsimp only

theorem eraseL_eraseL (lg : _root_.Flock.LigeritoProof) : eraseL (eraseL lg) = eraseL lg := by
  simp only [eraseL, eraseLP, Array.map_map, Function.comp_def]

theorem repMix3_eq (io : Asks FlockComplete.PS) (hs : _root_.Flock.Hashes FlockComplete.PS) (st : _root_.Flock.Setup)
    (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (tr : FlockComplete.Trace) (t : Array _root_.Flock.F128)
    (cm0 : Commit) (extras : Array _root_.Flock.Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List _root_.Flock.F128)) (words : List ℕ) (src : RepSrc) :
    repMix 3 io hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words src =
      eraseZ <$> repPZk io hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words := by
  unfold repMix repPZk piopPZk
  simp only [show ¬((3 : ℕ) = 0) by decide, show ¬((3 : ℕ) ≤ 1) by decide, show ¬((3 : ℕ) ≤ 2) by decide,
    ↓reduceIte]
  rw [bind_assoc, map_bind]; refine bind_congr fun ⟨zp, zc⟩ => ?_; dsimp only
  rw [bind_assoc, map_bind]; refine bind_congr fun ⟨lp, lc⟩ => ?_; dsimp only
  rw [bind_assoc, map_bind]; refine bind_congr fun _ => ?_
  rw [bind_assoc, map_bind]; refine bind_congr fun ⟨ss, T, sws⟩ => ?_; dsimp only
  rw [pure_bind]; dsimp only
  rw [map_bind]; refine bind_congr fun ⟨lg, zl, rec⟩ => ?_; dsimp only
  rw [pure_bind]; dsimp only
  simp only [replay_eraseL, ligeritoConstraints_eraseL]
  rw [map_bind]; refine bind_congr fun tr' => ?_
  rw [map_bind]; refine bind_congr fun ⟨rp, _⟩ => ?_; dsimp only
  rw [map_bind]; refine bind_congr fun lgc => ?_
  rw [map_bind]; refine bind_congr fun inner => ?_
  rw [map_pure]
  simp only [eraseZ, eraseL_eraseL]

/-! ### The cut's rep reads its final-only inputs only into the final-only data -/

/-- A level's queries without the opening: the positions and the consistency coins (`queryP`'s). -/
def queryPos (io : Asks FlockComplete.PS) (nonce : ℕ → ℕ → UInt64) (lv : _root_.Flock.Level) :
    LP FlockComplete.PS (Array ℕ × Array _root_.Flock.F128) := do
  powP nonce 0
  let coins ← liftT (sampleVecP io lv.q)
  powP nonce 1
  let alpha ← liftT (sampleVecP io (Nat.log2 (lv.q - 1) + 1))
  return (_root_.Flock.positions lv coins, alpha)

theorem queryP_eq (io : Asks FlockComplete.PS) (nonce : ℕ → ℕ → UInt64) (ms : _root_.Flock.MerkleScheme)
    (lv : _root_.Flock.Level) (cm : Commit) :
    queryP io nonce ms lv cm =
      (fun pa => (openP ms cm lv pa.1, consW lv.cols pa.1 pa.2)) <$> queryPos io nonce lv := by
  unfold queryP queryPos
  simp only [map_bind, map_pure]

theorem eraseLP_openP {cm cm' : Commit} (h : cm.layers = cm'.layers) (ms : _root_.Flock.MerkleScheme)
    (lv : _root_.Flock.Level) (pos : Array ℕ) : eraseLP (openP ms cm lv pos) = eraseLP (openP ms cm' lv pos) := by
  unfold eraseLP openP
  rw [h]
  split_ifs <;> simp [Array.map_map, Function.comp_def]

theorem eraseLP_eraseLP (op : _root_.Flock.LevelProof) : eraseLP (eraseLP op) = eraseLP op := by
  simp only [eraseLP, Array.map_map, Function.comp_def]

/-- A tuple with its fourth entry mapped. -/
def at4 {A B C D E : Type} (f : D → D) (t : A × B × C × D × E) : A × B × C × D × E :=
  (t.1, t.2.1, t.2.2.1, f t.2.2.2.1, t.2.2.2.2)

theorem run_bind_erase {σ α β : Type} {x y : StateT σ (_root_.Flock.TMW FlockComplete.PS) α} (e : α → α) (s : σ)
    {F : α × σ → _root_.Flock.TMW FlockComplete.PS β} (hF : ∀ a s, F (a, s) = F (e a, s)) (hxy : e <$> x = e <$> y) :
    x.run s >>= F = y.run s >>= F := by
  have k : ∀ z : StateT σ (_root_.Flock.TMW FlockComplete.PS) α, z.run s >>= F = (e <$> z).run s >>= F := by
    intro z
    rw [StateT.run_map, bind_map_left]
    exact bind_congr fun ⟨a, s'⟩ => hF a s'
  rw [k x, k y, hxy]

/-- **The cut's Ligerito opening reads level 0's commitment only through its cap and tree**, but for the opened rows
and salts, which only the final message reads. -/
theorem ligeritoCut_erase (io : Asks FlockComplete.PS) (hs : _root_.Flock.Hashes FlockComplete.PS)
    (ms : _root_.Flock.MerkleScheme) (sch : _root_.Flock.Schedule) (T : _root_.Flock.F128) {cm cm' : Commit}
    (hcap : cm.cap = cm'.cap) (hlay : cm.layers = cm'.layers) (b : Array _root_.Flock.F128)
    (lay : _root_.Flock.Zk.PadLayout) (src : RepSrc) :
    (fun z => (eraseL z.1, z.2)) <$> ligeritoCut io hs ms sch T cm b lay src =
      (fun z => (eraseL z.1, z.2)) <$> ligeritoCut io hs ms sch T cm' b lay src := by
  unfold ligeritoCut
  rw [hcap]
  simp only [map_bind]
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  refine run_bind_erase (at4 eraseLP) _ ?_ ?_
  · rintro ⟨caps, rproofs, fin, op0, zl, initCh, rho, b0, y⟩ s
    simp only [at4, map_pure, eraseL, eraseLP_eraseLP]
  · simp only [map_bind, map_pure, queryP_eq, bind_map_left, at4, eraseLP_openP hlay]

/-- **The cut's inner proof reads the pads commitment's columns and salts, `τ` and its salt only into the final-only
data**: it reads the commitment's size, root and tree, and the source's `(ρ_in, σ_in)`, `τ`'s commitment and `Y`. -/
theorem innerCut_erase (io : Asks FlockComplete.PS) (lay : _root_.Flock.Zk.PadLayout) {ic ic' : InnerCommit}
    (hn : ic.ch.size = ic'.ch.size) (hroot : ic.root = ic'.root) (hlay : ic.layers = ic'.layers)
    (finalC : _root_.Flock.F128) {src src' : RepSrc} (hs : eraseS src = eraseS src')
    (cons : Array (String × _root_.Flock.Zk.Lin)) :
    eraseIP <$> innerCut io lay ic finalC src cons = eraseIP <$> innerCut io lay ic' finalC src' cons := by
  have h1 : src.rhoIn = src'.rhoIn := by
    have e := congrArg RepSrc.rhoIn hs
    exact e
  have h2 : src.sigmaIn = src'.sigmaIn := by
    have e := congrArg RepSrc.sigmaIn hs
    exact e
  have h3 : src.tauCom = src'.tauCom := by
    have e := congrArg RepSrc.tauCom hs
    exact e
  have h4 : src.y = src'.y := by
    have e := congrArg RepSrc.y hs
    exact e
  unfold innerCut
  simp only [map_bind, map_pure, apply_ite (Functor.map (f := _root_.Flock.TMW FlockComplete.PS) eraseIP),
    eraseIP, Array.map_map, Function.comp_def, hn, hroot, hlay, h1, h2, h3, h4]

theorem bind_erase {α β : Type} {x y : _root_.Flock.TMW FlockComplete.PS α} (e : α → α)
    {F : α → _root_.Flock.TMW FlockComplete.PS β} (he : e <$> x = e <$> y) (hF : ∀ a, F a = F (e a)) :
    x >>= F = y >>= F := by
  have k : ∀ z : _root_.Flock.TMW FlockComplete.PS α, z >>= F = (e <$> z) >>= F := fun z => by
    rw [bind_map_left]
    exact bind_congr hF
  rw [k x, k y, he]

/-- **The cut's rep reads its pads commitment and source only into the final-only data** (`innerCut_erase`). -/
theorem repCut_eraseIS (io : Asks FlockComplete.PS) (hs : _root_.Flock.Hashes FlockComplete.PS)
    (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule) (rep : ℕ) (dom : ByteArray) (cm : Commit)
    (extras : Array _root_.Flock.Claim) {ic ic' : InnerCommit} (hn : ic.ch.size = ic'.ch.size)
    (hroot : ic.root = ic'.root) (hlay : ic.layers = ic'.layers) {src src' : RepSrc} (hsrc : eraseS src = eraseS src')
    (prior : Array (List _root_.Flock.F128)) (words : List ℕ) :
    eraseZ <$> repCut io hs st sch rep dom cm extras ic src prior words =
      eraseZ <$> repCut io hs st sch rep dom cm extras ic' src' prior words := by
  have hm : src.masked = src'.masked := by
    have e := congrArg RepSrc.masked hsrc
    exact e
  have hsv : src.sv = src'.sv := by
    have e := congrArg RepSrc.sv hsrc
    exact e
  have hhi : src.hiSalt = src'.hiSalt := by
    have e := congrArg RepSrc.hiSalt hsrc
    exact e
  have hy1 : src.y1 = src'.y1 := by
    have e := congrArg RepSrc.y1 hsrc
    exact e
  have hyb : src.ybar = src'.ybar := by
    have e := congrArg RepSrc.ybar hsrc
    exact e
  have hp : ∀ lay cap root, piopCut io st lay src cap root extras prior words =
      piopCut io st lay src' cap root extras prior words := fun _ _ _ => by
    unfold piopCut
    rw [hm, hsv]
  have hl : ∀ T b lay, ligeritoCut io hs st.merkle sch T cm b lay src =
      ligeritoCut io hs st.merkle sch T cm b lay src' := fun _ _ _ => by
    unfold ligeritoCut
    rw [hm, hhi, hy1, hyb]
  unfold repCut
  simp only [hroot, hp, hl, map_bind, map_pure]
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  exact bind_eraseZ (innerCut_erase _ _ hn hroot hlay _ hsrc _) _ _

/-- **The cut's rep reads level 0's commitment only through its cap and tree** but for the final-only data
(`ligeritoCut_erase`; the replay and the constraints read no opened row, `replay_eraseL`, `ligeritoConstraints_eraseL`). -/
theorem repCut_eraseC (io : Asks FlockComplete.PS) (hs : _root_.Flock.Hashes FlockComplete.PS)
    (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule) (rep : ℕ) (dom : ByteArray) {cm cm' : Commit}
    (hcap : cm.cap = cm'.cap) (hlay : cm.layers = cm'.layers) (extras : Array _root_.Flock.Claim) (ic : InnerCommit)
    (src : RepSrc) (prior : Array (List _root_.Flock.F128)) (words : List ℕ) :
    eraseZ <$> repCut io hs st sch rep dom cm extras ic src prior words =
      eraseZ <$> repCut io hs st sch rep dom cm' extras ic src prior words := by
  unfold repCut
  rw [hcap]
  simp only [map_bind, map_pure]
  refine bind_congr fun _ => ?_
  refine bind_erase (fun z => (eraseL z.1, z.2)) (ligeritoCut_erase _ _ _ _ _ hcap hlay _ _ _) ?_
  rintro ⟨lg, zl, rec⟩
  simp only [replay_eraseL, ligeritoConstraints_eraseL, eraseZ, eraseL_eraseL]

/-- The pads commitment's columns are the code of the pads at its leaves, whatever the hashes return. -/
theorem innerCommitP_map_ch {M : Type → Type} [Monad M] [LawfulMonad M] [MonadExceptOf String M]
    [MonadLiftT _root_.Flock.V M] (hs : _root_.Flock.Hashes M) (p : PadsZk) (salt : ℕ → ByteArray) :
    InnerCommit.ch <$> innerCommitP hs p salt =
      (fun _ => (Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)).map
        (_root_.Flock.Zk.codeAt (p.h ++ p.padH))) <$> innerCommitP hs p salt := by
  unfold innerCommitP
  simp only [map_bind, map_pure]

/-- The pads commitment's columns: the code of the pads at every one of its `n` leaves. Proved over an abstract
`Hashes` (`innerCommitP_map_ch`): case analysis on the SHA-512 instance makes the kernel evaluate the hashes. -/
theorem innerCommitP_ch {p : PadsZk} {salt : ℕ → ByteArray} {ic : InnerCommit}
    (h : (innerCommitP hsV p salt : _root_.Flock.V _) = .ok ic) :
    ic.ch = (Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)).map
      (_root_.Flock.Zk.codeAt (p.h ++ p.padH)) := by
  have e := innerCommitP_map_ch hsV p salt
  rw [h] at e
  exact Except.ok.inj e

/-! ### `τ`'s salt is a constant of each rep -/

theorem map_ite' {m : Type → Type} [Functor m] {α β : Type} (f : α → β) (c : Prop) [Decidable c] (a b : m α) :
    f <$> (if c then a else b) = if c then f <$> a else f <$> b := by
  split <;> rfl

section TauSalt

variable {M : Type → Type} [Monad M] [LawfulMonad M] [MonadExceptOf String M] [MonadLiftT _root_.Flock.V M]

theorem innerP_map_tauSalt (io : Asks M) (hs : _root_.Flock.Hashes M) (lay : _root_.Flock.Zk.PadLayout) (p : PadsZk)
    (ic : InnerCommit) (finalC : _root_.Flock.F128) (cons : Array (String × _root_.Flock.Zk.Lin)) :
    (fun z => z.tauSalt) <$> innerP io hs lay p ic finalC cons =
      (fun _ => p.tauSalt) <$> innerP io hs lay p ic finalC cons := by
  unfold innerP
  simp only [map_bind, map_pure, map_ite']

/-- **The session's rep proves under the pads' `τ` salt**, whatever it returns. -/
theorem repPZk_map_tauSalt (io : Asks M) (hs : _root_.Flock.Hashes M) (st : _root_.Flock.Setup)
    (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (tr : _) (t : Array _root_.Flock.F128) (cm0 : Commit)
    (extras : Array _root_.Flock.Claim) (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → _root_.Flock.F128)
    (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (prior : Array (List _root_.Flock.F128)) (words : List ℕ) :
    (fun z => z.inner.tauSalt) <$> repPZk io hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce
        prior words =
      (fun _ => pads.tauSalt) <$> repPZk io hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce
        prior words := by
  unfold repPZk
  simp only [map_bind, bind_pure_comp, Functor.map_map, innerP_map_tauSalt]

theorem innerCut_map_tauSalt (io : Asks M) (lay : _root_.Flock.Zk.PadLayout) (ic : InnerCommit)
    (finalC : _root_.Flock.F128) (src : RepSrc) (cons : Array (String × _root_.Flock.Zk.Lin)) :
    (fun z => z.tauSalt) <$> innerCut io lay ic finalC src cons =
      (fun _ => src.tauSalt) <$> innerCut io lay ic finalC src cons := by
  unfold innerCut
  simp only [map_bind, map_pure, map_ite']

/-- **The cut's rep proves under its source's `τ` salt**, whatever it returns. -/
theorem repCut_map_tauSalt (io : Asks M) (hs : _root_.Flock.Hashes M) (st : _root_.Flock.Setup)
    (sch : _root_.Flock.Schedule) (rep : ℕ) (dom : ByteArray) (cm : Commit) (extras : Array _root_.Flock.Claim)
    (ic : InnerCommit) (src : RepSrc) (prior : Array (List _root_.Flock.F128)) (words : List ℕ) :
    (fun z => z.inner.tauSalt) <$> repCut io hs st sch rep dom cm extras ic src prior words =
      (fun _ => src.tauSalt) <$> repCut io hs st sch rep dom cm extras ic src prior words := by
  unfold repCut
  simp only [map_bind, bind_pure_comp, Functor.map_map, innerCut_map_tauSalt]

theorem innerCut_map_tau (io : Asks M) (lay : _root_.Flock.Zk.PadLayout) (ic : InnerCommit)
    (finalC : _root_.Flock.F128) (src : RepSrc) (cons : Array (String × _root_.Flock.Zk.Lin)) :
    (fun z => z.tau) <$> innerCut io lay ic finalC src cons = (fun _ => src.tau) <$> innerCut io lay ic finalC src cons := by
  unfold innerCut
  simp only [map_bind, map_pure, map_ite']

/-- **The cut's rep sends its source's `τ`**, whatever it returns. -/
theorem repCut_map_tau (io : Asks M) (hs : _root_.Flock.Hashes M) (st : _root_.Flock.Setup)
    (sch : _root_.Flock.Schedule) (rep : ℕ) (dom : ByteArray) (cm : Commit) (extras : Array _root_.Flock.Claim)
    (ic : InnerCommit) (src : RepSrc) (prior : Array (List _root_.Flock.F128)) (words : List ℕ) :
    (fun z => z.inner.tau) <$> repCut io hs st sch rep dom cm extras ic src prior words =
      (fun _ => src.tau) <$> repCut io hs st sch rep dom cm extras ic src prior words := by
  unfold repCut
  simp only [map_bind, bind_pure_comp, Functor.map_map, innerCut_map_tau]

end TauSalt

/-- A rep run that returns has the value of a field its every run has. -/
theorem field_of_map {β : Type} {m : _root_.Flock.TMW FlockComplete.PS _root_.Flock.Zk.ZkProof}
    (F : _root_.Flock.Zk.ZkProof → β) {b : β} (hm : F <$> m = (fun _ => b) <$> m) {s : _root_.Flock.Transcript}
    {l l' : Array Msg} {p : _root_.Flock.Zk.ZkProof} {t : _root_.Flock.Transcript}
    (h : (m.run s).go l = (.ok (p, t), l')) : F p = b := by
  have e := congrArg (fun z => (z.run s).go l) hm
  simp only [StateT.run_map, FlockComplete.go_map', h, Except.map, Prod.mk.injEq, Except.ok.injEq] at e
  exact e.1.1

/-- A rep run that returns proves under the salt its every run proves under. -/
theorem tauSalt_of_map {m : _root_.Flock.TMW FlockComplete.PS _root_.Flock.Zk.ZkProof} {b : ByteArray}
    (hm : (fun z => z.inner.tauSalt) <$> m = (fun _ => b) <$> m) {s : _root_.Flock.Transcript} {l l' : Array Msg}
    {p : _root_.Flock.Zk.ZkProof} {t : _root_.Flock.Transcript} (h : (m.run s).go l = (.ok (p, t), l')) :
    p.inner.tauSalt = b := by
  have e := congrArg (fun z => (z.run s).go l) hm
  simp only [StateT.run_map, FlockComplete.go_map', h, Except.map, Prod.mk.injEq, Except.ok.injEq] at e
  exact e.1.1

end FinRep

/-- **Table `j`'s rep `r` from the cut is the session's rep, its final-only data forgotten** (`eraseZ`): at opened coins
every table admits (so no refusal shows, `not_refused_of_admits`), from its first round, at the erased commitments and
the cut's source, through the hybrids `SimPiop`, `SimLig`, `SimInner`. -/
theorem repCut_erased (h0 : SimPiop) (h1 : SimLig) (h2 : SimInner) {x : Msg} (hx : Live x) {w : Claims.Flock.Wit}
    (hw : RZk x w) (τ : UE.Tape) {c : Fin (refK x) → FlockDeploy.Coin} (hA : ∀ j, (tabC PE x j).Admits (czExe PE x c j))
    {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (j : Fin (jOf x)) (r : Fin 2) (dom : ByteArray) (cm0 : FlockComplete.Commit) (ic : FlockComplete.InnerCommit)
    (extras : Array _root_.Flock.Claim) (he : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    (l : Array Msg) (hl : l.size = repBase x (2 * j.val + r.val)) :
    (((FlockComplete.eraseZ <$> FlockComplete.repCut (ioE c (refK x)) hsE (FlockDeploy.tabsOf (stsOf x))[j.val]!
        (zkTab x j).sch r.val dom (FlockComplete.eraseC cm0) extras (FlockComplete.eraseIC ic)
        ((srcE x c w τ j.val).rep r.val) (priorE c j.val r.val)
        (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!).run (_root_.Flock.Transcript.init #[] dom)).go l) =
      (((FlockComplete.eraseZ <$> FlockComplete.repPZk (ioE c (refK x)) hsE (FlockDeploy.tabsOf (stsOf x))[j.val]!
        (zkTab x j).sch (zkTab x j).tPad r.val dom (trsE x w (UE.tapeZk τ))[j.val]!
        (FlockComplete.packed (trsE x w (UE.tapeZk τ))[j.val]! (FlockDeploy.tabsOf (stsOf x))[j.val]!.m) cm0 extras
        (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val
          (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
            (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0)) ic
        ((UE.tapeZk τ).extra j.val) ((UE.tapeZk τ).lanePad j.val) ((UE.tapeZk τ).salt j.val r.val)
        ((UE.tapeZk τ).nonce j.val r.val) (priorE c j.val r.val)
        (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!).run (_root_.Flock.Transcript.init #[] dom)).go l) := by
  have hn := not_refused_of_admits hx hA (refK x)
  rw [← FinRep.repCut_mix0 (tPad := (zkTab x j).tPad) (tr := (trsE x w (UE.tapeZk τ))[j.val]!)
    (t := FlockComplete.packed (trsE x w (UE.tapeZk τ))[j.val]! (FlockDeploy.tabsOf (stsOf x))[j.val]!.m)
    (pads := FlockComplete.padsOf (UE.tapeZk τ) j.val r.val
      (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0))
    (extra := (UE.tapeZk τ).extra j.val) (lanePad := (UE.tapeZk τ).lanePad j.val)
    (salt := (UE.tapeZk τ).salt j.val r.val) (nonce := (UE.tapeZk τ).nonce j.val r.val)]
  change ((mixE 0 x w τ c (refK x) j r dom cm0 extras ic).run _).go l = _
  rw [h0 x hx w hw τ c _ hn s spec hsx j r dom cm0 ic extras he l hl,
    h1 x hx w hw τ c _ hn s spec hsx j r dom cm0 ic extras he l hl,
    h2 x hx w hw τ c _ hn s spec hsx j r dom cm0 ic extras he l hl]
  unfold mixE
  rw [FinRep.repMix3_eq]

/-! ## The final-only data -/

/-- **Two rep proofs agree on their final-only data**: level 0's opened rows and salts, `τ` and its salt, and the pads'
opened columns and salts, the fields `eraseZ` forgets. -/
def FinalEq (p q : _root_.Flock.Zk.ZkProof) : Prop :=
  p.proof.ligerito.initialProof.rows = q.proof.ligerito.initialProof.rows ∧
    p.proof.ligerito.initialProof.salts = q.proof.ligerito.initialProof.salts ∧
    p.inner.tau = q.inner.tau ∧ p.inner.tauSalt = q.inner.tauSalt ∧ p.inner.opened = q.inner.opened ∧
    p.inner.salts = q.inner.salts

/-- A rep proof is its erasure and its final-only data. -/
theorem eq_of_eraseZ {p q : _root_.Flock.Zk.ZkProof} (h : FlockComplete.eraseZ p = FlockComplete.eraseZ q)
    (hf : FinalEq p q) : p = q := by
  obtain ⟨⟨cap, pa, zc, lc, rs, ⟨ic, ⟨rw0, sb, sl⟩, rc, rp, fp, ms, qn, ov, cn, csn⟩⟩, l0,
    ⟨root, rho, sig, tc, ts, tau, y, op, isl, pth⟩⟩ := p
  obtain ⟨⟨cap', pa', zc', lc', rs', ⟨ic', ⟨rw0', sb', sl'⟩, rc', rp', fp', ms', qn', ov', cn', csn'⟩⟩, l0',
    ⟨root', rho', sig', tc', ts', tau', y', op', isl', pth'⟩⟩ := q
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := hf
  simp only at h1 h2 h3 h4 h5 h6
  subst h1 h2 h3 h4 h5 h6
  simp only [FlockComplete.eraseZ, FlockComplete.eraseL, FlockComplete.eraseLP, FlockComplete.eraseIP,
    _root_.Flock.Zk.ZkProof.mk.injEq, _root_.Flock.Proof.mk.injEq, _root_.Flock.LigeritoProof.mk.injEq,
    _root_.Flock.LevelProof.mk.injEq, _root_.Flock.Zk.InnerProof.mk.injEq] at h ⊢
  simp_all

/-- **A run is fixed by its erasure and its final-only data**: two runs of the rep monad equal once `eraseZ` is applied
to their proofs, whose final-only data agree wherever both return, are equal. -/
theorem eq_of_go_eraseZ {A B : Except String (_root_.Flock.Zk.ZkProof × _root_.Flock.Transcript) × Array Msg}
    (h : (A.1.map (fun p => (FlockComplete.eraseZ p.1, p.2)), A.2) =
      (B.1.map (fun p => (FlockComplete.eraseZ p.1, p.2)), B.2))
    (hf : ∀ p q t l', A = (.ok (p, t), l') → B = (.ok (q, t), l') →
      FlockComplete.eraseZ p = FlockComplete.eraseZ q → FinalEq p q) : A = B := by
  obtain ⟨a, la⟩ := A
  obtain ⟨b, lb⟩ := B
  simp only [Prod.mk.injEq] at h
  obtain ⟨h, rfl⟩ := h
  rcases a with e | ⟨p, t⟩ <;> rcases b with e' | ⟨q, t'⟩ <;>
    simp only [Except.map, Except.error.injEq, Except.ok.injEq, Prod.mk.injEq, reduceCtorEq] at h
  · rw [h]
  · obtain ⟨hz, rfl⟩ := h
    rw [eq_of_eraseZ hz (hf p q t la rfl rfl hz)]

/-- **Table `i`'s rep `r` read from the views**: `repCut` on the views' source, with level 0's and the pads'
commitments the session's but their rows, columns and salts the views' (`fill0`, `fillP`): `RepFin`'s left side. -/
noncomputable def repVE (x : Msg) (w : Claims.Flock.Wit) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin)
    (spec : _root_.Flock.SessionSpec) (i : ℕ) (hi : i < jOf x) (r : ℕ) (cm0 : FlockComplete.Commit)
    (extras : Array _root_.Flock.Claim) (ic : FlockComplete.InnerCommit) :
    FlockComplete.PS (_root_.Flock.Zk.ZkProof × _root_.Flock.Transcript) :=
  (FlockComplete.repCut (ioE c (refK x)) hsE (FlockDeploy.tabsOf (stsOf x))[i]! (zkTab x ⟨i, hi⟩).sch r
    (domE spec i r) (fill0 x c w τ i ((zkTab x ⟨i, hi⟩).sch.levels.getD 0 default) cm0) extras
    (fillP x c w τ i r (((finLayE x i hi).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo *
      2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE) ic) ((srcV x c w τ i).rep r) (priorE c i r)
    (FlockComplete.MaskRank.masksOf (stsOf x))[i]!).run (_root_.Flock.Transcript.init #[] (domE spec i r))

/-- **The final-only data read from the views is the session's**: in a rep's context, where the rep read from the views
and the session's rep both return, at one transcript and history and with proofs equal up to `eraseZ`, their final-only
data agree. -/
def RepFinal : Prop :=
  ∀ x w τ c spec i hi r cm0 extras ic l, RepCtx x w τ c spec i hi r cm0 extras ic l →
    ∀ p q t l', (repVE x w τ c spec i hi r cm0 extras ic).go l = (.ok (p, t), l') →
      (repHE x w τ c spec i hi r cm0 extras ic).go l = (.ok (q, t), l') →
      FlockComplete.eraseZ p = FlockComplete.eraseZ q → FinalEq p q

/-- **`RepFin` from the hybrids and the final-only data**: the rep read from the views is the cut's rep up to the
final-only data (`FinRep.repCut_eraseC`, `FinRep.repCut_eraseIS`, `eraseS_srcV`), hence the session's up to it
(`repCut_erased`), hence the session's (`RepFinal`). -/
theorem repFin_of (h0 : SimPiop) (h1 : SimLig) (h2 : SimInner) (hFin : RepFinal) : RepFin := by
  intro x w τ c spec i hi r cm0 extras ic l hC
  obtain ⟨hx, hw, hA, ⟨s, hsx⟩, hr, hcm, hic, he, hl⟩ := hC
  have hE := repCut_erased h0 h1 h2 hx hw τ hA hsx ⟨i, hi⟩ ⟨r, hr⟩ (domE spec i r) cm0 ic extras he l hl
  have hsrc : eraseS ((srcV x c w τ i).rep r) = eraseS ((srcE x c w τ i).rep r) :=
    (eraseS_srcV hA w τ hi hr).trans (by rw [← eraseS_srcV hA w τ hi hr]; rfl)
  have hn : (fillP x c w τ i r (((finLayE x i hi).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo *
      2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE) ic).ch.size = (FlockComplete.eraseIC ic).ch.size := by
    simp [fillP, FlockComplete.eraseIC, FinRep.innerCommitP_ch hic, FlockComplete.padsOf]
  have E := (FinRep.repCut_eraseC (ioE c (refK x)) hsE (FlockDeploy.tabsOf (stsOf x))[i]! (zkTab x ⟨i, hi⟩).sch r
      (domE spec i r) (cm := fill0 x c w τ i ((zkTab x ⟨i, hi⟩).sch.levels.getD 0 default) cm0)
      (cm' := FlockComplete.eraseC cm0) rfl rfl extras
      (fillP x c w τ i r (((finLayE x i hi).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo *
        2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE) ic) ((srcV x c w τ i).rep r) (priorE c i r)
      (FlockComplete.MaskRank.masksOf (stsOf x))[i]!).trans
    (FinRep.repCut_eraseIS (ioE c (refK x)) hsE (FlockDeploy.tabsOf (stsOf x))[i]! (zkTab x ⟨i, hi⟩).sch r
      (domE spec i r) (FlockComplete.eraseC cm0) extras hn rfl rfl hsrc (priorE c i r)
      (FlockComplete.MaskRank.masksOf (stsOf x))[i]!)
  have hE' := (congrArg (fun z => (z.run (_root_.Flock.Transcript.init #[] (domE spec i r))).go l) E).trans hE
  simp only [StateT.run_map, FlockComplete.go_map'] at hE'
  exact eq_of_go_eraseZ (A := (repVE x w τ c spec i hi r cm0 extras ic).go l)
    (B := (repHE x w τ c spec i hi r cm0 extras ic).go l) hE'
    fun p q t l' hp hq hz => hFin x w τ c spec i hi r cm0 extras ic l
      ⟨hx, hw, hA, ⟨s, hsx⟩, hr, hcm, hic, he, hl⟩ p q t l' hp hq hz

/-- **The views' `τ` salt is the honest pads'**: table `i`'s rep `r` source from the views reads `y_τ`'s bytes from the
tape's cell `i`, which `padsOf` fits to `TAU_SALT_BYTES`. -/
theorem srcV_tauSalt (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) {i r : ℕ}
    (hi : i < jOf x) (hr : r < 2) (lay : _root_.Flock.Zk.PadLayout) :
    ((srcV x c w τ i).rep r).tauSalt = (FlockComplete.padsOf (UE.tapeZk τ) i r lay).tauSalt := by
  have hc : i < Claims.Flock.zkTableCap := lt_of_lt_of_le hi (jOf_le_cap x)
  show ((tabSrcView PE x (czExe PE x c) (viewsE x c w τ) i).rep r).tauSalt = _
  unfold tabSrcView
  rw [dite_eq_left hi]
  dsimp only
  rw [dite_eq_left hr]
  show ZkView.saltBytes ((τ ⟨i, hc⟩).2.1.2.2 ⟨r, hr⟩) =
    FlockComplete.fitBytes 192 (fAt ByteArray.empty (fun r => ZkView.saltBytes ((cellAt τ i).2.1.2.2 r)) r)
  rw [cellAt_lt τ hc, fAt_lt _ _ hr, fitBytes_of_size (saltBytes_size _)]

/-- **Two rep proofs agree on their final-only data that the run decides**: level 0's opened rows and salts, `τ`, and
the pads' opened columns and salts (`FinalEq` but `τ`'s salt, which each rep fixes in advance). -/
def FinalOpen (p q : _root_.Flock.Zk.ZkProof) : Prop :=
  p.proof.ligerito.initialProof.rows = q.proof.ligerito.initialProof.rows ∧
    p.proof.ligerito.initialProof.salts = q.proof.ligerito.initialProof.salts ∧
    p.inner.tau = q.inner.tau ∧ p.inner.opened = q.inner.opened ∧ p.inner.salts = q.inner.salts

/-- **`RepFinal` but `τ`'s salt**: in a rep's context, where the rep read from the views and the session's rep both
return, at one transcript and history and with proofs equal up to `eraseZ`, level 0's opened rows and salts, `τ`, and
the pads' opened columns and salts agree. -/
def RepOpenFinal : Prop :=
  ∀ x w τ c spec i hi r cm0 extras ic l, RepCtx x w τ c spec i hi r cm0 extras ic l →
    ∀ p q t l', (repVE x w τ c spec i hi r cm0 extras ic).go l = (.ok (p, t), l') →
      (repHE x w τ c spec i hi r cm0 extras ic).go l = (.ok (q, t), l') →
      FlockComplete.eraseZ p = FlockComplete.eraseZ q → FinalOpen p q

/-- **`RepFinal` from the run-decided fields**: `τ`'s salt is the views' in the rep from the views
(`FinRep.repCut_map_tauSalt`) and the pads' in the session's (`FinRep.repPZk_map_tauSalt`), and those are equal
(`srcV_tauSalt`). -/
theorem repFinal_of (h : RepOpenFinal) : RepFinal := by
  intro x w τ c spec i hi r cm0 extras ic l hC p q t l' hp hq hz
  obtain ⟨h1, h2, h3, h4, h5⟩ := h x w τ c spec i hi r cm0 extras ic l hC p q t l' hp hq hz
  refine ⟨h1, h2, h3, ?_, h4, h5⟩
  rw [FinRep.tauSalt_of_map (FinRep.repCut_map_tauSalt _ _ _ _ _ _ _ _ _ _ _ _) hp,
    FinRep.tauSalt_of_map (FinRep.repPZk_map_tauSalt _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) hq]
  exact srcV_tauSalt x c w τ hi hC.2.2.2.2.1 _

/-- **The views' `τ` is table `i`'s view's** (`View.tau`, `InnerTauVal`'s right side). -/
theorem srcV_tau (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) {i r : ℕ}
    (hi : i < jOf x) (hr : r < 2) :
    ((srcV x c w τ i).rep r).tau = FlockLevel3.GF128.val ((viewsE x c w τ ⟨i, hi⟩).tau ⟨r, hr⟩) := by
  show ((tabSrcView PE x (czExe PE x c) (viewsE x c w τ) i).rep r).tau = _
  unfold tabSrcView
  rw [dite_eq_left hi]
  dsimp only
  rw [dite_eq_left hr]

/-- **`τ` in the session's rep is its view's**: in a rep's context, where the session's rep returns, its `τ` is table
`i`'s view's (`InnerTauVal`'s reading at the last message). -/
def RepTau : Prop :=
  ∀ x w τ c spec i hi r cm0 extras ic l, RepCtx x w τ c spec i hi r cm0 extras ic l →
    ∀ q t l', (repHE x w τ c spec i hi r cm0 extras ic).go l = (.ok (q, t), l') → ∀ hr : r < 2,
      q.inner.tau = FlockLevel3.GF128.val ((viewsE x c w τ ⟨i, hi⟩).tau ⟨r, hr⟩)

/-- **Two rep proofs agree on the final-only data at opened positions**: level 0's opened rows and salts, and the
pads' opened columns and salts. -/
def FinalRest (p q : _root_.Flock.Zk.ZkProof) : Prop :=
  p.proof.ligerito.initialProof.rows = q.proof.ligerito.initialProof.rows ∧
    p.proof.ligerito.initialProof.salts = q.proof.ligerito.initialProof.salts ∧
    p.inner.opened = q.inner.opened ∧ p.inner.salts = q.inner.salts

/-- **`RepOpenFinal` at the opened positions**: in a rep's context, where the rep read from the views and the session's
rep both return, at one transcript and history and with proofs equal up to `eraseZ`, level 0's opened rows and salts and
the pads' opened columns and salts agree. -/
def RepOpenRest : Prop :=
  ∀ x w τ c spec i hi r cm0 extras ic l, RepCtx x w τ c spec i hi r cm0 extras ic l →
    ∀ p q t l', (repVE x w τ c spec i hi r cm0 extras ic).go l = (.ok (p, t), l') →
      (repHE x w τ c spec i hi r cm0 extras ic).go l = (.ok (q, t), l') →
      FlockComplete.eraseZ p = FlockComplete.eraseZ q → FinalRest p q

/-- **`RepOpenFinal` from `τ` and the opened positions**: the rep read from the views sends its source's `τ`
(`FinRep.repCut_map_tau`), the view's (`srcV_tau`), which is the session's (`RepTau`). -/
theorem repOpenFinal_of (hT : RepTau) (hR : RepOpenRest) : RepOpenFinal := by
  intro x w τ c spec i hi r cm0 extras ic l hC p q t l' hp hq hz
  obtain ⟨h1, h2, h3, h4⟩ := hR x w τ c spec i hi r cm0 extras ic l hC p q t l' hp hq hz
  refine ⟨h1, h2, ?_, h3, h4⟩
  have hr := hC.2.2.2.2.1
  rw [FinRep.field_of_map (fun z => z.inner.tau) (FinRep.repCut_map_tau _ _ _ _ _ _ _ _ _ _ _ _) hp,
    hT x w τ c spec i hi r cm0 extras ic l hC q t l' hq hr]
  exact srcV_tau x c w τ hi hr

/-! ## The opened coins as replies

`SimSession`'s bridge from `asksCut` to `asksS` (`cutS`, `repPZk_cut`, `respOf`), restated here so that this file
imports `Split` alone. -/

namespace FinRep

open FlockComplete Flock.With

/-- **A run that returns** from the bodies `l` returns the same on the right. -/
def cutS : Sim FlockComplete.PS FlockComplete.PS where
  R x y := ∀ l a l', x.go l = (.ok a, l') → y.go l = (.ok a, l')
  pure _ _ _ _ h := h
  bind {α β x y f g} hxy hfg l b l'' h := by
    obtain ⟨a, l', h1, h2⟩ := JsonRT.go_bind_ok h
    rw [go_bind, hxy l a l' h1]
    exact hfg a l' b l'' h2
  throw _ _ _ _ h := by cases h
  lift _ _ _ _ h := h

theorem cutS_refl {α : Type} (x : FlockComplete.PS α) : cutS.R x x := fun _ _ _ h => h

theorem cutS_reflT {α : Type} (x : _root_.Flock.TMW FlockComplete.PS α) : (cutS.st _root_.Flock.Transcript).R x x :=
  fun _ => cutS_refl _

theorem cutS_reflL {α : Type} (x : LP FlockComplete.PS α) :
    ((cutS.st _root_.Flock.Transcript).st LigState).R x x :=
  fun _ => cutS_reflT _

theorem asksCut_round_go (coin : ℕ → ℕ → _root_.Flock.F128) (n : ℕ) (c : ByteArray) (l : Array Msg) :
    ((asksCut coin n).round c).go l =
      (if n ≤ l.size then .error "past the openings" else .ok (coin l.size), l.push (Claims.toMsg c)) := by
  have e : ((asksCut coin n).round c).go l =
      ((if n ≤ l.size then throw "past the openings" else pure (coin l.size) : FlockComplete.PS _)).go
        (l.push (Claims.toMsg c)) := rfl
  rw [e]
  split_ifs <;> rfl

theorem cutS_round {coin : ℕ → ℕ → _root_.Flock.F128} {n : ℕ} {resp : ℕ → Msg} (hr : ∀ k, coinOf (resp k) = coin k)
    (c : ByteArray) : cutS.R ((asksCut coin n).round c) ((asksS resp).round c) := by
  intro l v l' e
  rw [asksCut_round_go] at e
  by_cases hn : n ≤ l.size
  · simp only [hn, ↓reduceIte, Prod.mk.injEq, reduceCtorEq, false_and] at e
  · simp only [hn, ↓reduceIte, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    show (Except.ok (coinOf (resp l.size)), l.push (Claims.toMsg c)) = _
    rw [hr]

section Bridge

variable {coin : ℕ → ℕ → _root_.Flock.F128} {n : ℕ} {resp : ℕ → Msg}

theorem squeezeP_cut (hr : ∀ k, coinOf (resp k) = coin k) (kind : UInt8) (m : ℕ) :
    (cutS.st _root_.Flock.Transcript).R (squeezeP (asksCut coin n) kind m) (squeezeP (asksS resp) kind m) := by
  intro t
  unfold squeezeP
  sim_walk [cutS_round hr]

theorem sampleP_cut (hr : ∀ k, coinOf (resp k) = coin k) :
    (cutS.st _root_.Flock.Transcript).R (sampleP (asksCut coin n)) (sampleP (asksS resp)) := by
  unfold sampleP
  sim_walk [squeezeP_cut hr]

theorem sampleKP_cut (hr : ∀ k, coinOf (resp k) = coin k) :
    (cutS.st _root_.Flock.Transcript).R (sampleKP (asksCut coin n)) (sampleKP (asksS resp)) := by
  unfold sampleKP
  sim_walk [squeezeP_cut hr]

theorem liftT_cut {α : Type} {x y : _root_.Flock.TMW FlockComplete.PS α} (h : (cutS.st _root_.Flock.Transcript).R x y) :
    ((cutS.st _root_.Flock.Transcript).st LigState).R (liftT x) (liftT y) :=
  Sim.lift_st _ h

theorem zerocheckPZk_cut (hr : ∀ k, coinOf (resp k) = coin k) (m : ℕ) (digest cap root : ByteArray)
    (lay : _root_.Flock.Zk.PadLayout) (h : Array _root_.Flock.F128) (tr : FlockComplete.Trace) :
    (cutS.st _root_.Flock.Transcript).R (zerocheckPZk (asksCut coin n) m digest cap root lay h tr)
      (zerocheckPZk (asksS resp) m digest cap root lay h tr) := by
  unfold zerocheckPZk sampleVecP
  sim_walk [squeezeP_cut hr, sampleP_cut hr]

theorem lincheckPZk_cut (hr : ∀ k, coinOf (resp k) = coin k) (c : _root_.Flock.CircuitFold) (m : ℕ)
    (lay : _root_.Flock.Zk.PadLayout) (h : Array _root_.Flock.F128) (tr : FlockComplete.Trace) (zc : _root_.Flock.ZcOut) :
    (cutS.st _root_.Flock.Transcript).R (lincheckPZk (asksCut coin n) c m lay h tr zc)
      (lincheckPZk (asksS resp) c m lay h tr zc) := by
  unfold lincheckPZk
  sim_walk [sampleP_cut hr]

theorem ringSwitchP_cut (hr : ∀ k, coinOf (resp k) = coin k) (tr : FlockComplete.Trace) (claims : Array _root_.Flock.Claim) :
    (cutS.st _root_.Flock.Transcript).R (ringSwitchP (asksCut coin n) tr claims) (ringSwitchP (asksS resp) tr claims) := by
  unfold ringSwitchP sampleVecP
  sim_walk [squeezeP_cut hr]

theorem challengeP_cut (hr : ∀ k, coinOf (resp k) = coin k) (split : Bool) :
    ((cutS.st _root_.Flock.Transcript).st LigState).R (challengeP (asksCut coin n) split)
      (challengeP (asksS resp) split) := by
  unfold challengeP
  sim_walk [liftT_cut, sampleKP_cut hr, cutS_reflL]

theorem oodIntroP_cut (hr : ∀ k, coinOf (resp k) = coin k) (nonce : ℕ → ℕ → UInt64) (g : Array _root_.Flock.F128)
    (dim : ℕ) :
    ((cutS.st _root_.Flock.Transcript).st LigState).R (oodIntroP (asksCut coin n) nonce g dim)
      (oodIntroP (asksS resp) nonce g dim) := by
  unfold oodIntroP sampleVecP
  sim_walk [liftT_cut, squeezeP_cut hr, sampleP_cut hr, cutS_reflL]

theorem queryP_cut (hr : ∀ k, coinOf (resp k) = coin k) (nonce : ℕ → ℕ → UInt64) (ms : _root_.Flock.MerkleScheme)
    (lv : _root_.Flock.Level) (cm : Commit) :
    ((cutS.st _root_.Flock.Transcript).st LigState).R (queryP (asksCut coin n) nonce ms lv cm)
      (queryP (asksS resp) nonce ms lv cm) := by
  unfold queryP sampleVecP
  sim_walk [liftT_cut, squeezeP_cut hr, cutS_reflL]

theorem ligeritoPZk_cut (hr : ∀ k, coinOf (resp k) = coin k) (hs : _root_.Flock.Hashes FlockComplete.PS)
    (ms : _root_.Flock.MerkleScheme) (sch : _root_.Flock.Schedule) (T : _root_.Flock.F128) (cm0 : Commit)
    (t b : Array _root_.Flock.F128) (lay : _root_.Flock.Zk.PadLayout) (h : Array _root_.Flock.F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    (cutS.st _root_.Flock.Transcript).R
      (ligeritoPZk (asksCut coin n) hs ms sch T cm0 t b lay h rep tPad extra lanePad salt nonce)
      (ligeritoPZk (asksS resp) hs ms sch T cm0 t b lay h rep tPad extra lanePad salt nonce) := by
  unfold ligeritoPZk sampleVecP
  sim_walk [liftT_cut, squeezeP_cut hr, sampleP_cut hr, sampleKP_cut hr, challengeP_cut hr, oodIntroP_cut hr,
    queryP_cut hr, cutS_reflL, cutS_reflT, cutS_refl]

theorem innerP_cut (hr : ∀ k, coinOf (resp k) = coin k) (hs : _root_.Flock.Hashes FlockComplete.PS)
    (lay : _root_.Flock.Zk.PadLayout) (p : PadsZk) (ic : InnerCommit) (finalC : _root_.Flock.F128)
    (cons : Array (String × _root_.Flock.Zk.Lin)) :
    (cutS.st _root_.Flock.Transcript).R (innerP (asksCut coin n) hs lay p ic finalC cons)
      (innerP (asksS resp) hs lay p ic finalC cons) := by
  unfold innerP sampleVecP
  sim_walk [sampleP_cut hr, squeezeP_cut hr, cutS_reflT, cutS_refl]

/-- **A masked rep answered by `asksCut` that returns** is the rep answered by `asksS` at the same coins. -/
theorem repPZk_cut (hr : ∀ k, coinOf (resp k) = coin k) (hs : _root_.Flock.Hashes FlockComplete.PS) (st : _root_.Flock.Setup)
    (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (tr : FlockComplete.Trace) (t : Array _root_.Flock.F128)
    (cm0 : Commit) (extras : Array _root_.Flock.Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List _root_.Flock.F128)) (words : List ℕ) :
    (cutS.st _root_.Flock.Transcript).R
      (repPZk (asksCut coin n) hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words)
      (repPZk (asksS resp) hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words) := by
  unfold repPZk rankCheckP
  sim_walk [zerocheckPZk_cut hr, lincheckPZk_cut hr, ringSwitchP_cut hr, ligeritoPZk_cut hr, innerP_cut hr,
    cutS_reflT, cutS_refl]

end Bridge

/-- **The opened coins' replies**: round `k`'s coin's bytes, the zero coin's past the pre-final rounds. -/
def respOf {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (k : ℕ) : Msg :=
  (if h : k < refK x then c ⟨k, h⟩ else default : FlockDeploy.Coin).bytes

theorem coinOf_respOf {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (k : ℕ) :
    coinOf (respOf c k) = fun i => slotAt c k i := by
  funext i
  unfold respOf slotAt
  rw [JsonRT.coinOf_bytes]
  split_ifs with hk
  · rfl
  · unfold FlockDeploy.Coin.at
    split_ifs <;> rfl

/-- **The session's rep `r` of table `i` that returns, answered by the replies** (`repPZk_cut`), with the facts about
its table that its round count and claim points read. -/
theorem repHE_asks {x : Msg} {w : Claims.Flock.Wit} {τ : UE.Tape} {c : Fin (refK x) → FlockDeploy.Coin}
    {spec : _root_.Flock.SessionSpec} {i : ℕ} {hi : i < jOf x} {r : ℕ} {cm0 : Commit} {extras : Array _root_.Flock.Claim}
    {ic : InnerCommit} {l : Array Msg} (hC : RepCtx x w τ c spec i hi r cm0 extras ic l) {p t l'}
    (e : (repHE x w τ c spec i hi r cm0 extras ic).go l = (.ok (p, t), l')) :
    ((repPZk (asksS (respOf c)) hsE (_root_.Flock.Setup.ofCircuit (stsOf x)[i]!) (zkTab x ⟨i, hi⟩).sch
      (zkTab x ⟨i, hi⟩).tPad r (domE spec i r) (trsE x w (UE.tapeZk τ))[i]!
      (packed (trsE x w (UE.tapeZk τ))[i]! (_root_.Flock.Setup.ofCircuit (stsOf x)[i]!).m) cm0 extras
      (padsOf (UE.tapeZk τ) i r (finLayE x i hi)) ic ((UE.tapeZk τ).extra i) ((UE.tapeZk τ).lanePad i)
      ((UE.tapeZk τ).salt i r) ((UE.tapeZk τ).nonce i r) (priorE c i r)
      (MaskRank.masksOf (stsOf x))[i]!).run (_root_.Flock.Transcript.init #[] (domE spec i r))).go l =
        (.ok (p, t), l') ∧
      _root_.Flock.Zk.schedule (stsOf x)[i]!.m = .ok ((zkTab x ⟨i, hi⟩).sch, (zkTab x ⟨i, hi⟩).tPad) ∧
      14 ≤ (stsOf x)[i]!.m ∧ 6 ≤ (stsOf x)[i]!.c.kLog ∧
      extras.size = spec.points * (stsOf x)[i]!.regions.size := by
  obtain ⟨hx, -, -, ⟨s, hsx⟩, -, -, -, hex, -⟩ := hC
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hi' : i < (stsOf x).size := lt_of_jOf ⟨i, hi⟩
  have hst : (stsOf x)[i]! = stAt x ⟨i, hi⟩ := getElem!_pos _ _ _
  have htab : (FlockDeploy.tabsOf (stsOf x))[i]! = _root_.Flock.Setup.ofCircuit (stsOf x)[i]! := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) i (by simpa [FlockDeploy.tabsOf] using hi'),
      getElem!_pos (stsOf x) i hi']
    simp [FlockDeploy.tabsOf]
  have e' := repPZk_cut (coin := fun k i => slotAt c k i) (n := refK x) (resp := respOf c)
    (fun k => coinOf_respOf c k) hsE _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ (_root_.Flock.Transcript.init #[] _) l _ l' e
  rw [htab] at e' hex
  refine ⟨e', ?_, ?_, ?_, ?_⟩
  · rw [hst]
    exact (zkTab x ⟨i, hi⟩).hz
  · have := (hall i hi').1
    rw [getElem!_pos (stsOf x) i hi']
    omega
  · rw [hst]
    exact (zkTab x ⟨i, hi⟩).wf.1.hk6
  · rw [JsonRT.extraClaims_size hex]
    simp [Nat.mul_comm, ptsE]

end FinRep

/-- **A rep that returns sends its rounds** (`SimSession`'s rep count): answered by the replies (`FinRep.repHE_asks`),
the rep sends `repRoundsT`'s rounds (`repPZk_exact`), which are `repRounds`'s (`repRoundsT_eq`), from `repBase`. -/
theorem repExact : RepExact := by
  intro x w τ c spec i hi r cm0 extras ic l hC p t l' e
  obtain ⟨e', hv, hm, -, hx'⟩ := FinRep.repHE_asks hC e
  obtain ⟨-, -, -, ⟨s, hsx⟩, hr, -, -, -, hl⟩ := hC
  obtain ⟨-, -, hsm, hk0, -, hdim, hlv, -, -⟩ := FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ hv
  obtain ⟨hn7, hL2, -, -, -, -, -⟩ := hdim
  obtain ⟨hl3, -, -, -⟩ := FlockComplete.repPZk_exact (hm := hm) (hn := by rw [hn7, hsm]; omega) (hk0 := hk0)
    (hL := hL2) (hq := fun i hi => (hlv i hi).q_ne_zero) (h := e')
  have e6 : (_root_.Flock.Setup.ofCircuit (stsOf x)[i]!).fold.kLog = (stsOf x)[i]!.c.kLog := rfl
  have e7 : (_root_.Flock.Setup.ofCircuit (stsOf x)[i]!).m = (stsOf x)[i]!.m := rfl
  rw [repBase_eq hsx, FlockComplete.ZkTables.repStart_succ, show (2 * i + r) / 2 = i by omega,
    ← FlockComplete.ZkTables.repRoundsT_eq hv hx', hl3, hl, repBase_eq hsx]
  unfold FlockComplete.ZkTables.repRoundsT FlockComplete.piopZk
  rw [e6, e7, hx']
  dsimp only
  omega

/-- **Rep `0`'s rounds hold its claim points at the opened coins** (`SimSession`'s): answered by the replies
(`FinRep.repHE_asks`), its rounds' points are the replies' coins from `repBase` (`repPZk_points`), which are
`ptsOpen`'s. -/
theorem repPoints : RepPoints := by
  intro x w τ c spec i hi cm0 extras ic l hC p t l' e
  obtain ⟨e', hv, hm, hk6, -⟩ := FinRep.repHE_asks hC e
  obtain ⟨-, -, -, ⟨s, hsx⟩, -, -, -, -, hl⟩ := hC
  obtain ⟨-, -, hsm, hk0, -, hdim, hlv, -, -⟩ := FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ hv
  obtain ⟨hn7, hL2, -, -, -, -, -⟩ := hdim
  obtain ⟨hl3, -, -, -⟩ := FlockComplete.repPZk_exact (hm := hm) (hn := by rw [hn7, hsm]; omega) (hk0 := hk0)
    (hL := hL2) (hq := fun i hi => (hlv i hi).q_ne_zero) (h := e')
  have e6 : (_root_.Flock.Setup.ofCircuit (stsOf x)[i]!).fold.kLog = (stsOf x)[i]!.c.kLog := rfl
  have e7 : (_root_.Flock.Setup.ofCircuit (stsOf x)[i]!).m = (stsOf x)[i]!.m := rfl
  have e8 : (_root_.Flock.Setup.ofCircuit (stsOf x)[i]!).kLog = (stsOf x)[i]!.c.kLog := rfl
  have hpts := FlockComplete.ZkTables.repPZk_points (hm := hm) (hk := hk6) (h := e') (hsz := by
    rw [hl3]
    unfold FlockComplete.piopZk
    rw [e6, e7, e8]
    omega)
  simp only [FinRep.coinOf_respOf] at hpts
  have htab : (FlockDeploy.tabsOf (stsOf x))[i]! = _root_.Flock.Setup.ofCircuit (stsOf x)[i]! := by
    have hi' : i < (stsOf x).size := lt_of_jOf ⟨i, hi⟩
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) i (by simpa [FlockDeploy.tabsOf] using hi'),
      getElem!_pos (stsOf x) i hi']
    simp [FlockDeploy.tabsOf]
  rw [Nat.add_zero] at hl
  rw [htab]
  unfold ptsOpen
  rw [← hl]
  exact hpts

/-- **`FinExe` from the hybrids, the live end and the per-rep facts**: `finExe_of_reps` with `RepFin` from `SimPiop`,
`SimLig`, `SimInner` and the final-only data (`repFin_of`). -/
theorem finExe_of_parts (hL : LiveEnd) (hSC : SimCommit) (h0 : SimPiop) (h1 : SimLig) (h2 : SimInner)
    (hFin : RepFinal) (hE : RepExact) (hP : RepPoints) : FinExe :=
  finExe_of_reps hL hSC (repFin_of h0 h1 h2 hFin) hE hP

/-- **`FinExe` from the hybrids, the live end, the per-rep facts and the run-decided final-only data**
(`finExe_of_parts` with `RepFinal` from `RepOpenFinal`, `repFinal_of`). -/
theorem finExe_of_open (hL : LiveEnd) (hSC : SimCommit) (h0 : SimPiop) (h1 : SimLig) (h2 : SimInner)
    (hO : RepOpenFinal) (hE : RepExact) (hP : RepPoints) : FinExe :=
  finExe_of_parts hL hSC h0 h1 h2 (repFinal_of hO) hE hP

/-- **`FinExe` from the hybrids, the live end and the run-decided final-only data**: `finExe_of_open` with the rep
count and rep `0`'s claim points proved (`repExact`, `repPoints`). -/
theorem finExe_of_final (hL : LiveEnd) (hSC : SimCommit) (h0 : SimPiop) (h1 : SimLig) (h2 : SimInner)
    (hO : RepOpenFinal) : FinExe :=
  finExe_of_open hL hSC h0 h1 h2 hO repExact repPoints

/-- **`FinExe` from the hybrids, the live end, the session's `τ` and the opened positions** (`finExe_of_final` with
`RepOpenFinal` from `RepTau` and `RepOpenRest`, `repOpenFinal_of`). -/
theorem finExe_of_rest (hL : LiveEnd) (hSC : SimCommit) (h0 : SimPiop) (h1 : SimLig) (h2 : SimInner) (hT : RepTau)
    (hR : RepOpenRest) : FinExe :=
  finExe_of_final hL hSC h0 h1 h2 (repOpenFinal_of hT hR)

/-! ## Level 0's opened rows and salts

Both reps send the same PIOP (`piopCut_alike` at the cut's source, `piopSrcE`), `piopZk` rounds from `repBase`, and
open level 0 at the query of Ligerito's round `k₀ + 8` (`FinRep.ligeritoCut_open0`, `L0.ligeritoPZk_opens`): the
same positions, `q0At`'s. Those are opened positions of the table (`padPos` puts the queried positions first, and at
most `t_pad` are queried), where the views' rows and salts are the session's (`Round0.exeOf_inl`'s rows,
`Agrees0.salt`). -/

namespace FinRep

open FlockComplete FlockComplete.Replay FlockComplete.Replay.Lg FlockComplete.L0 Flock.With

section Level0Bridge

variable {coin : ℕ → ℕ → _root_.Flock.F128} {n : ℕ} {resp : ℕ → Msg}

theorem piopPZk_cut (hr : ∀ k, coinOf (resp k) = coin k) (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout)
    (h : Array _root_.Flock.F128) (tr : FlockComplete.Trace) (cap root : ByteArray) (extras : Array _root_.Flock.Claim)
    (prior : Array (List _root_.Flock.F128)) (words : List ℕ) :
    (cutS.st _root_.Flock.Transcript).R (piopPZk (asksCut coin n) st lay h tr cap root extras prior words)
      (piopPZk (asksS resp) st lay h tr cap root extras prior words) := by
  unfold piopPZk rankCheckP
  sim_walk [zerocheckPZk_cut hr, lincheckPZk_cut hr, ringSwitchP_cut hr, cutS_reflT, cutS_refl]

theorem ligeritoCut_cut (hr : ∀ k, coinOf (resp k) = coin k) (hs : _root_.Flock.Hashes FlockComplete.PS)
    (ms : _root_.Flock.MerkleScheme) (sch : _root_.Flock.Schedule) (T : _root_.Flock.F128) (cm0 : Commit)
    (b : Array _root_.Flock.F128) (lay : _root_.Flock.Zk.PadLayout) (src : RepSrc) :
    (cutS.st _root_.Flock.Transcript).R (ligeritoCut (asksCut coin n) hs ms sch T cm0 b lay src)
      (ligeritoCut (asksS resp) hs ms sch T cm0 b lay src) := by
  unfold ligeritoCut sampleVecP
  sim_walk [liftT_cut, squeezeP_cut hr, sampleP_cut hr, sampleKP_cut hr, challengeP_cut hr, oodIntroP_cut hr,
    queryP_cut hr, cutS_reflL, cutS_reflT, cutS_refl]

end Level0Bridge

/-- **The PIOP's exact rounds at `asksS`**: `piopZk` bodies sent and `piopZk` rounds taken. -/
theorem piopPZk_counts (resp : ℕ → Msg) (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout)
    (h : Array _root_.Flock.F128) (tr : FlockComplete.Trace) (cap root : ByteArray) (extras : Array _root_.Flock.Claim)
    (prior : Array (List _root_.Flock.F128)) (words : List ℕ) (hm : 14 ≤ st.m) {t t' : _root_.Flock.Transcript}
    {l l' : Array Msg} {o : _root_.Flock.ZerocheckProof × _root_.Flock.LincheckProof × Array (Array _root_.Flock.F128) ×
      _root_.Flock.F128 × Array _root_.Flock.Switched}
    (hrun : ((piopPZk (asksS resp) st lay h tr cap root extras prior words).run t).go l = (.ok (o, t'), l')) :
    l'.size = l.size + piopZk st extras ∧ t'.k = t.k + piopZk st extras := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckPZk_go resp hm st.digest cap root lay h tr t l
  unfold piopPZk at hrun
  rw [bind_go hz] at hrun
  dsimp only at hrun
  have h' := hrun
  rw [StateT.run_bind] at h'
  obtain ⟨_, _, h₁, -⟩ := JsonRT.go_bind_ok h'
  obtain ⟨full, hfold⟩ := lincheckPZk_fold resp _ _ _ _ _ _ _ _ h₁
  obtain ⟨l₂, hl₂, hl⟩ := lincheckPZk_go resp st.fold st.m lay h tr _ full _ l₁ hfold
  rw [bind_go hl] at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨u, t₃⟩, l₃', hk, hrun⟩ := JsonRT.go_bind_ok hrun
  obtain ⟨ht₃, hl₃⟩ := rankCheckP_go hk
  subst t₃ l₃'
  obtain ⟨l₃, hl₃, hrs⟩ := ringSwitchP_go resp tr (_root_.Flock.abcClaims st.m st.kLog _ _ ++ extras)
    (by simp [_root_.Flock.abcClaims]) _ l₂
  rw [bind_go hrs] at hrun
  obtain ⟨he, rfl⟩ := Prod.mk.inj hrun
  obtain ⟨-, rfl⟩ := Prod.mk.inj (Except.ok.inj he)
  have hsz : (_root_.Flock.abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) ++
      extras).size = 2 + extras.size := by simp [_root_.Flock.abcClaims, Nat.add_comm]
  rw [Rs.tEnd_k, LcZ.tEnd_k, ZcZ.tEnd_k]
  unfold piopZk
  omega

/-- **What every returning run of the prover's step `y` returns** has `Q`. -/
def LRuns {β : Type} (y : LP FlockComplete.PS β) (Q : β → Prop) : Prop :=
  ∀ (st : LigState) (t : _root_.Flock.Transcript) (l : Array Msg) (b : β) (st' : LigState) (t' : _root_.Flock.Transcript)
    (l' : Array Msg), ((y.run st).run t).go l = (.ok ((b, st'), t'), l') → Q b

theorem LRuns.bind {α β : Type} {x : LP FlockComplete.PS α} {f : α → LP FlockComplete.PS β} {Q : β → Prop}
    (hf : ∀ a, LRuns (f a) Q) : LRuns (x >>= f) Q := by
  intro st t l b st' t' l' h
  obtain ⟨a, st₁, t₁, l₁, -, h2⟩ := lp_bind_ok h
  exact hf a st₁ t₁ l₁ b st' t' l' h2

theorem LRuns.pure {β : Type} {b : β} {Q : β → Prop} (hb : Q b) : LRuns (Pure.pure b : LP FlockComplete.PS β) Q := by
  intro st t l b' st' t' l' h
  obtain ⟨rfl, -, -, -⟩ := ok_inj (show (Except.ok ((b, st), t), l) = _ from h)
  exact hb

/-- **What every returning run of the prover's step `y` from round `k` with `L + k` bodies sent returns** has `Q`
(`KRuns` without the rounds at the end). -/
def KPost (L : ℕ) {β : Type} (y : LP FlockComplete.PS β) (k : ℕ) (Q : β → Prop) : Prop :=
  ∀ (st : LigState) (t : _root_.Flock.Transcript) (l : Array Msg) (b : β) (st' : LigState) (t' : _root_.Flock.Transcript)
    (l' : Array Msg), t.k = k → l.size = L + k → ((y.run st).run t).go l = (.ok ((b, st'), t'), l') → Q b

theorem KPost.bind {L : ℕ} {α β : Type} {x : LP FlockComplete.PS α} {f : α → LP FlockComplete.PS β} {k₁ k₂ : ℕ}
    {Qa : α → Prop} {Q : β → Prop} (hx : KRuns L x k₁ k₂ Qa) (hf : ∀ a, Qa a → KPost L (f a) k₂ Q) :
    KPost L (x >>= f) k₁ Q := by
  intro st t l b st' t' l' hk hl h
  obtain ⟨a, st₁, t₁, l₁, h1, h2⟩ := lp_bind_ok h
  obtain ⟨hk₁, hl₁, hQa⟩ := hx st t l a st₁ t₁ l₁ hk hl h1
  exact hf a hQa st₁ t₁ l₁ b st' t' l' hk₁ hl₁ h2

theorem KPost.ofL {L : ℕ} {β : Type} {y : LP FlockComplete.PS β} {k : ℕ} {Q : β → Prop} (h : LRuns y Q) :
    KPost L y k Q :=
  fun st t l b st' t' l' _ _ e => h st t l b st' t' l' e

theorem KPost.oodPush {L : ℕ} {β : Type} (y : _root_.Flock.F128) {c : PUnit → LP FlockComplete.PS β} {k : ℕ}
    {Q : β → Prop} (h : KPost L (c ()) k Q) :
    KPost L (_root_.modify (fun st : LigState => { st with oods := st.oods.push y }) >>= fun _ =>
      liftT (_root_.Flock.liftTM (_root_.Flock.Transcript.absorbF y)) >>= c) k Q := by
  intro st t l b st' t' l' hk hl e
  obtain ⟨a, st₁, t₁, l₁, h1, h2⟩ := lp_bind_ok e
  obtain ⟨-, rfl, rfl, rfl⟩ := ok_inj (show (Except.ok (((), { st with oods := st.oods.push y }), t), l) = _ from h1)
  obtain ⟨u, st₂, t₂, l₂, h3, h4⟩ := lp_bind_ok h2
  obtain ⟨rfl, rfl, rfl, rfl⟩ :=
    ok_inj (show (Except.ok (((), { st with oods := st.oods.push y }), ap (pF y) t), l) = _ from h3)
  exact h _ _ _ b st' t' l' (by rw [ap_k]; exact hk) hl h4

/-- **The cut's Ligerito opens level 0 honestly**: run from round `k` with `L + k` bodies sent, it opens `cm0` at the
positions of round `k + k₀ + 8`'s coins, as the session's (`L0.ligeritoPZk_opens`); what it sends after that query
does not change the opening. -/
theorem ligeritoCut_open0 (resp : ℕ → Msg) (h : ByteArray → ByteArray) (ms : _root_.Flock.MerkleScheme)
    (sch : _root_.Flock.Schedule) (T : _root_.Flock.F128) (cm0 : Commit) (b : Array _root_.Flock.F128)
    (lay : _root_.Flock.Zk.PadLayout) (src : RepSrc) (L : ℕ) (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0)
    (hq : (sch.levels.getD 0 default).q ≠ 0) {t : _root_.Flock.Transcript} {l : Array Msg} (hl : l.size = L + t.k)
    {P : _root_.Flock.LigeritoProof} {zl : _root_.Flock.Zk.Level0Msgs} {rc : _root_.Flock.Zk.L0Rec}
    {t' : _root_.Flock.Transcript} {l' : Array Msg}
    (hrun : ((ligeritoCut (asksS resp) (_root_.Flock.Hashes.ofSha512 h) ms sch T cm0 b lay src).run t).go l =
      (.ok ((P, zl, rc), t'), l')) :
    P.initialProof = openP ms cm0 (sch.levels.getD 0 default) (_root_.Flock.positions (sch.levels.getD 0 default)
      ((Array.range (sch.levels.getD 0 default).q).map (coinOf (resp (L + (t.k + sch.k0 + 8)))))) := by
  unfold ligeritoCut at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l)] at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨⟨⟨caps, rproofs, fin, op0, zl', initCh, rho, b0', y0⟩, stF⟩, t₁⟩, l₁, h1, h2⟩ := JsonRT.go_bind_ok hrun
  obtain ⟨h21, rfl⟩ := Prod.mk.inj h2
  obtain ⟨hP3, rfl⟩ := Prod.mk.inj (Except.ok.inj h21)
  obtain ⟨hP, -⟩ := Prod.mk.inj hP3
  subst hP
  set k := t.k
  have key := (?prog : KPost (β := Array (Array ByteArray) × Array _root_.Flock.LevelProof × _root_.Flock.FinalProof ×
      _root_.Flock.LevelProof × _root_.Flock.Zk.Level0Msgs × Array _root_.Flock.F256 ×
      (_root_.Flock.F256 × _root_.Flock.F256) × _root_.Flock.F128 × _root_.Flock.F128) L _ k fun r =>
      r.2.2.2.1 = openP ms cm0 (sch.levels.getD 0 default) (_root_.Flock.positions (sch.levels.getD 0 default)
        ((Array.range (sch.levels.getD 0 default).q).map (coinOf (resp (L + (k + sch.k0 + 8))))))) _ _ l _ _ _ _
    (by rfl) hl h1
  case prog =>
    refine KPost.bind (KRuns.sampleVec resp hn _) fun z _ => KPost.oodPush _ ?_
    refine KPost.bind (KRuns.pow _ 2 _) fun _ _ => KPost.bind (KRuns.sample resp _) fun _ _ =>
      KPost.bind (KRuns.modify _ _) fun _ _ => KPost.bind (KRuns.sendMsg _ _) fun _ _ => ?_
    refine KPost.bind ((KRuns.loop (σ := Array _root_.Flock.F256 × (_root_.Flock.F256 × _root_.Flock.F256) ×
      _root_.Flock.Zk.Level0Msgs) _
      (fun j => k + 1 + 1 + j + if j = sch.k0 then 2 else 0) (fun _ _ => True) sch.k0 ?lane _ trivial).cast
      (by rw [ite_eq_right (by omega)]) (by rw [ite_eq_left rfl])) fun _ _ => ?_
    case lane =>
      intro j hj s _
      refine (KRuns.sampleK resp _).bind fun _ _ => (KRuns.modify _ _).bind fun _ _ => ?_
      by_cases hj1 : j + 1 = sch.k0
      · refine KRuns.ite (fun _ => ?_) (fun hne => absurd (by simp [hj1]) hne)
        refine (KRuns.get _).bind fun _ _ => ?_
        refine (KRuns.absorb _ (pKv _) (fun t => Flock.RecAlgebra.absorbSlice_eq _ t) _).bind fun _ _ =>
          (KRuns.absorb _ (pKv _) (fun t => Flock.RecAlgebra.absorbSlice_eq _ t) _).bind fun _ _ => ?_
        refine (KRuns.sampleK resp _).bind fun _ _ => (KRuns.sampleK resp _).bind fun _ _ => ?_
        refine KRuns.ite (fun _ => KRuns.throwBind _ _ _ _ _) (fun _ => ?_)
        refine (KRuns.absorb _ (pLabel _) (fun t => Flock.RecAlgebra.label_eq _ t) _).bind fun _ _ =>
          (KRuns.absorb _ (pKv _) (fun t => Flock.RecAlgebra.absorbSlice_eq _ t) _).bind fun _ _ =>
          (KRuns.absorb _ (pSlice _) (fun t => Flock.RecAlgebra.absorbSlice_eq _ t) _).bind fun _ _ => ?_
        exact (KRuns.modify _ _).bind fun _ _ => (KRuns.get _).bind fun _ _ => (KRuns.sendMsg _ _).bind fun _ _ =>
          KRuns.pure ⟨_, rfl, trivial⟩ (by split_ifs <;> omega)
      · refine KRuns.ite (fun hc => absurd (by simpa using hc) hj1) fun _ => ?_
        exact (KRuns.sendMsg _ _).bind fun _ _ => KRuns.pure ⟨_, rfl, trivial⟩ (by split_ifs <;> omega)
    refine KPost.bind (KRuns.get _) fun d _ => KPost.bind (KRuns.commit h ms _ _ _ _) fun cm1 _ => ?_
    refine KPost.bind (KRuns.absorb _ (pBytes _) (fun t => Flock.RecAlgebra.absorbBytes_eq _ t) _) fun _ _ => ?_
    refine KPost.bind (KRuns.oodIntro resp _ _ (Nat.succ_ne_zero _) _) fun _ _ =>
      KPost.bind (KRuns.oodIntro resp _ _ (Nat.succ_ne_zero _) _) fun _ _ => ?_
    refine KPost.bind (KRuns.query resp _ ms _ cm0 hq _) fun x hx => KPost.ofL ?_
    refine LRuns.bind fun _ => LRuns.bind fun _ => LRuns.bind fun _ => LRuns.bind fun _ => LRuns.bind fun _ =>
      LRuns.pure ?_
    rw [hx]
    congr 5
    omega
  exact key

/-- **The session's rep opens level 0 at round `k₀ + 8` of its Ligerito**: answered by the replies from a fresh
transcript, after the PIOP's `piopZk` rounds (`ligeritoPZk_opens`); the replay, constraints and inner proof leave the
Ligerito proof as it is. -/
theorem repPZk_open0 (resp : ℕ → Msg) (hsh : ByteArray → ByteArray) (st : _root_.Flock.Setup)
    (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (tr : FlockComplete.Trace) (tb : Array _root_.Flock.F128)
    (cm0 : Commit) (extras : Array _root_.Flock.Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List _root_.Flock.F128)) (words : List ℕ) (hm : 14 ≤ st.m) (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0)
    (hL : 2 ≤ sch.levels.size) (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0)
    {t t' : _root_.Flock.Transcript} (ht : t.k = 0) {l l' : Array Msg} {P : _root_.Flock.Zk.ZkProof}
    (h : ((repPZk (asksS resp) (_root_.Flock.Hashes.ofSha512 hsh) st sch tPad rep dom tr tb cm0 extras pads ic extra
      lanePad salt nonce prior words).run t).go l = (.ok (P, t'), l')) :
    P.proof.ligerito.initialProof = openP st.merkle cm0 (sch.levels.getD 0 default)
      (_root_.Flock.positions (sch.levels.getD 0 default) ((Array.range (sch.levels.getD 0 default).q).map
        (coinOf (resp (l.size + (piopZk st extras + sch.k0 + 8)))))) := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckPZk_go resp hm st.digest (flat cm0.cap) ic.root
    (_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0) pads.h tr t l
  unfold repPZk at h
  rw [bind_go hz] at h
  dsimp only at h
  have h' := h
  rw [StateT.run_bind] at h'
  obtain ⟨_, _, h₁, -⟩ := JsonRT.go_bind_ok h'
  obtain ⟨full, hfold⟩ := lincheckPZk_fold resp _ _ _ _ _ _ _ _ h₁
  obtain ⟨l₂, hl₂, hl⟩ := lincheckPZk_go resp st.fold st.m (_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0) pads.h
    tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full _ l₁ hfold
  rw [bind_go hl] at h
  dsimp only at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨u, t₃⟩, l₃', hk, h⟩ := JsonRT.go_bind_ok h
  obtain ⟨ht₃, hl₃⟩ := rankCheckP_go hk
  subst t₃ l₃'
  obtain ⟨l₃, hl₃, hrs⟩ := ringSwitchP_go resp tr (_root_.Flock.abcClaims st.m st.kLog
    (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) ++
      extras) (by simp [_root_.Flock.abcClaims]) _ l₂
  rw [bind_go hrs] at h
  dsimp only at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨⟨lg, zl, rec⟩, t₄⟩, l₄, hlg, h⟩ := JsonRT.go_bind_ok h
  have hsz : (_root_.Flock.abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) ++
      extras).size = 2 + extras.size := by simp [_root_.Flock.abcClaims, Nat.add_comm]
  have e3 : l₃.size = l.size + piopZk st extras := by unfold piopZk; omega
  have h0 := (ligeritoPZk_opens resp hsh st.merkle sch _ cm0 _ _ _ _ rep tPad extra lanePad salt nonce l.size hn hk0 hL
    hq (by rw [Rs.tEnd_k, LcZ.tEnd_k, ZcZ.tEnd_k, ht, hsz, e3]; unfold piopZk; omega) hlg).1
  rw [bind_go (show ((get : _root_.Flock.TMW FlockComplete.PS _root_.Flock.Transcript).run t₄).go l₄ =
    (.ok (t₄, t₄), l₄) from rfl)] at h
  dsimp only at h
  obtain ⟨⟨rp, x⟩, -, h⟩ := liftV_bind_ok h
  dsimp only at h
  obtain ⟨lgc, -, h⟩ := liftV_bind_ok h
  rw [StateT.run_bind] at h
  obtain ⟨⟨inner, t₅⟩, l₅, -, h⟩ := JsonRT.go_bind_ok h
  obtain ⟨h5, -⟩ := Prod.mk.inj h
  obtain ⟨hP, -⟩ := Prod.mk.inj (Except.ok.inj h5)
  subst hP
  refine h0.trans ?_
  unfold posAt cfOf qOff
  rw [if_pos (by omega)]
  congr 5
  rw [Rs.tEnd_k, LcZ.tEnd_k, ZcZ.tEnd_k, ht, hsz]
  unfold piopZk
  omega

/-- A source the cut's PIOP reads as `PiopSrc` reads only its masked values and switched vectors, which `eraseS`
keeps. -/
theorem piopSrc_of_eraseS {coin : ℕ → ℕ → _root_.Flock.F128} {n : ℕ} {st : _root_.Flock.Setup}
    {lay : _root_.Flock.Zk.PadLayout} {h : Array _root_.Flock.F128} {tr : FlockComplete.Trace} {s : RepSrc}
    {extras : Array _root_.Flock.Claim} {L : ℕ} (hv : PiopSrc coin n st lay h tr (eraseS s) extras L) :
    PiopSrc coin n st lay h tr s extras L :=
  ⟨hv.r1ab, hv.r1c, hv.ml0, hv.ml1, hv.fa, hv.fb, hv.fc, hv.lc, hv.zp, hv.sv⟩

/-- **The cut's rep opens level 0 at round `k₀ + 8` of its Ligerito**, as the session's (`repPZk_open0`): answered by
`asksCut` from a fresh transcript at a source at the cut's coins (`PiopSrc`), its PIOP is the session's
(`piopCut_alike`), so takes `piopZk` rounds (`piopPZk_counts` at `asksS`), and its Ligerito opens its level-0
commitment `cm0` there (`ligeritoCut_open0`). -/
theorem repCut_open0 {coin : ℕ → ℕ → _root_.Flock.F128} {n : ℕ} {resp : ℕ → Msg} (hr : ∀ k, coinOf (resp k) = coin k)
    (hsh : ByteArray → ByteArray) (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule) (rep : ℕ) (dom : ByteArray)
    (cm0 : Commit) (extras : Array _root_.Flock.Claim) (ic : InnerCommit) (src : RepSrc)
    (prior : Array (List _root_.Flock.F128)) (words : List ℕ) (h : Array _root_.Flock.F128) (tr : FlockComplete.Trace)
    (hm : 14 ≤ st.m) (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0) (hq : (sch.levels.getD 0 default).q ≠ 0)
    {t t' : _root_.Flock.Transcript} (ht : t.k = 0) {l l' : Array Msg}
    (hv : PiopSrc coin n st (_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0) h tr src extras l.size)
    {P : _root_.Flock.Zk.ZkProof}
    (hrun : ((repCut (asksCut coin n) (_root_.Flock.Hashes.ofSha512 hsh) st sch rep dom cm0 extras ic src prior
      words).run t).go l = (.ok (P, t'), l')) :
    P.proof.ligerito.initialProof = openP st.merkle cm0 (sch.levels.getD 0 default)
      (_root_.Flock.positions (sch.levels.getD 0 default) ((Array.range (sch.levels.getD 0 default).q).map
        (coinOf (resp (l.size + (piopZk st extras + sch.k0 + 8)))))) := by
  unfold repCut at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨⟨zp, lp, ss, T, sws⟩, t₃⟩, l₃, h1, h2⟩ := JsonRT.go_bind_ok hrun
  rw [(piopCut_alike coin n st _ h tr src _ _ extras prior words hm t l hv).eq] at h1
  obtain ⟨hl₃, ht₃⟩ := piopPZk_counts resp st _ h tr _ _ extras prior words hm
    (piopPZk_cut (n := n) hr _ _ _ _ _ _ _ _ _ t l _ l₃ h1)
  dsimp only at h2
  rw [StateT.run_bind] at h2
  obtain ⟨⟨⟨lg, zl, rec⟩, t₄⟩, l₄, h3, h4⟩ := JsonRT.go_bind_ok h2
  have h0 := ligeritoCut_open0 resp hsh st.merkle sch _ cm0 _ _ src l.size hn hk0 hq
    (by rw [hl₃, ht₃, ht]; omega) (ligeritoCut_cut (n := n) hr _ _ _ _ _ _ _ _ t₃ l₃ _ l₄ h3)
  dsimp only at h4
  rw [bind_go (show ((get : _root_.Flock.TMW FlockComplete.PS _root_.Flock.Transcript).run t₄).go l₄ =
    (.ok (t₄, t₄), l₄) from rfl)] at h4
  obtain ⟨⟨rp, x⟩, -, h4⟩ := liftV_bind_ok h4
  dsimp only at h4
  obtain ⟨lgc, -, h4⟩ := liftV_bind_ok h4
  rw [StateT.run_bind] at h4
  obtain ⟨⟨inner, t₅⟩, l₅, -, h4⟩ := JsonRT.go_bind_ok h4
  obtain ⟨h5, -⟩ := Prod.mk.inj h4
  obtain ⟨hP, -⟩ := Prod.mk.inj (Except.ok.inj h5)
  subst hP
  refine h0.trans ?_
  rw [ht₃, ht, Nat.zero_add]

/-- **A level-0 commitment's salts are its fitted salts**, at every tree position. -/
theorem commitRowsP_salts {hs : _root_.Flock.Hashes _root_.Flock.V} {ms : _root_.Flock.MerkleScheme}
    {lv : _root_.Flock.Level} {rows : Array (Array _root_.Flock.F128)} {salt : ℕ → ByteArray} {cm : Commit}
    (h : commitRowsP hs ms lv rows salt = .ok cm) :
    cm.salts = (Array.range (2 ^ lv.d)).map fun q => fitBytes ms.saltBytes (salt q) := by
  unfold commitRowsP at h
  obtain ⟨leaves, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨r, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases h
  rfl

/-- **Table `j`'s level-0 row at tree position `p` is the masked prover's** (`Round0.exeOf_inl`'s row): at a layout
agreeing at round `0`, `encodeRowsZk`'s row `p`, at every coin vector. -/
theorem row0_exe (P : Composed.LigProver) (L : TapeLayout P)
    (z : (Fin Claims.Flock.zkTableCap → L.Cell) → FlockComplete.TapeZk) (hA : Agrees0 P L z) {x : Msg}
    (w : Claims.Flock.Wit) (t : Fin Claims.Flock.zkTableCap → L.Cell) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (j : Fin (jOf x)) (p : Fin ((tabC P x j).z.nS + (tabC P x j).z.nU)) :
    ZkView.row0 (((tabC P x j).T (cz j)).col0At (wtOf P x w j) (((TapePlan.ofLayout L z).tape x t).1 j) p) =
      (FlockComplete.encodeRowsZk (lv0 x j) (zkTab x j).tPad
        ((FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x)) (FlockComplete.ZkTables.traces (stsOf x)
          (FlockComplete.ZkTables.maskWOf x w ((TapePlan.ofLayout L z).tapeZk t).mask)))[j.val]!)
        ((z t).extra j.val) ((z t).lanePad j.val))[p.val]! := by
  obtain ⟨hnL, hL, htp, ht, hd, -⟩ := tabC_z P x j
  rw [col0At_tabC]
  refine FlockComplete.row0_eq_encodeRowsZk (lv0 x j) _ _ _ _ (by rw [← hd]; exact p.isLt) hnL hL htp ht _
    (fun l => ?_) (fun r i => ?_)
  · show ∑ c, ((tabC P x j).T (cz j)).lanes (wtOf P x w j) (((TapePlan.ofLayout L z).tape x t).1 j).1 l c *
        Composed.eAt (lv0 x j).cols p.val c +
      ∑ k, (((TapePlan.ofLayout L z).tape x t).1 j).2.1.1 l k * Composed.gAt (lv0 x j).cols p.val k = _
    simp only [lanes_tabC]
    exact congrArg (_ + ·) (Finset.sum_congr rfl fun k _ => congrArg (· * _) (hA.pad x t j l k))
  · show ∑ c, (((TapePlan.ofLayout L z).tape x t).1 j).2.1.2.1 r i c * Composed.eAt (lv0 x j).cols p.val c +
      ∑ k, (((TapePlan.ofLayout L z).tape x t).1 j).2.1.2.2 r i k * Composed.gAt (lv0 x j).cols p.val k = _
    exact congrArg₂ (· + ·) (Finset.sum_congr rfl fun c _ => congrArg (· * _) (hA.ext x t j r i c))
      (Finset.sum_congr rfl fun k _ => congrArg (· * _) (hA.extPad x t j r i k))

/-- `at0` at an opened position's tree position is its value. -/
theorem at0_s0 (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest) (j : Fin (jOf x))
    {α : Type} (d : α) (f : Fin (tabC P x j).z.nS → α) (a : Fin (tabC P x j).z.nS) :
    at0 P x cz j d f (((tabC P x j).T (cz j)).s0 (.inl a)).val = f a := by
  unfold at0
  simp only [Fin.isLt, ↓reduceDIte, Fin.eta, Equiv.symm_apply_apply]

/-- **The views' level-0 row and salt at an opened position are the session's**: at the tree position of opened
position `a`, table `i`'s source from the views reads `encodeRowsZk`'s row there (`row0_exe`) and the tape's level-0
salt fitted to 192 bytes (`Agrees0.salt`). -/
theorem srcV_open0 (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) {i : ℕ}
    (hi : i < jOf x) (a : Fin (tabC PE x ⟨i, hi⟩).z.nS) :
    (srcV x c w τ i).row0 (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).s0 (.inl a)).val =
        (encodeRowsZk (lv0 x ⟨i, hi⟩) (zkTab x ⟨i, hi⟩).tPad
          (packed (trsE x w (UE.tapeZk τ))[i]! (FlockDeploy.tabsOf (stsOf x))[i]!.m)
          ((UE.tapeZk τ).extra i) ((UE.tapeZk τ).lanePad i))[(((tabC PE x ⟨i, hi⟩).T
            (czExe PE x c ⟨i, hi⟩)).s0 (.inl a)).val]! ∧
      (srcV x c w τ i).salt0 (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).s0 (.inl a)).val =
        fitBytes 192 ((UE.tapeZk τ).salt i 0 0 (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).s0 (.inl a)).val) := by
  have hj : i < (FlockDeploy.tabsOf (stsOf x)).size := by simpa [FlockDeploy.tabsOf] using lt_of_jOf ⟨i, hi⟩
  have e : ((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).col0At (wtOf PE x w ⟨i, hi⟩) ((UE.tape x τ).1 ⟨i, hi⟩)
      (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).s0 (.inl a)) =
      ((viewsE x c w τ ⟨i, hi⟩).rows a, (viewsE x c w τ ⟨i, hi⟩).xrows a) := by
    unfold ZkView.Coined.col0At
    simp only [Equiv.symm_apply_apply]
    rfl
  have es : (viewsE x c w τ ⟨i, hi⟩).salts0 a =
      (UE.tape x τ).2.1 ⟨⟨i, hi⟩, .inl (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).s0 (.inl a))⟩ := by
    set_option maxRecDepth 8000 in rfl
  have hrow := row0_exe PE (TapeLayout.exe PE) exeTape (agrees0_exe PE) w τ (czExe PE x c) ⟨i, hi⟩
    (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).s0 (.inl a))
  have hsl := (agrees0_exe PE).salt x τ ⟨i, hi⟩ (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).s0 (.inl a))
  show (tabSrcView PE x (czExe PE x c) (viewsE x c w τ) i).row0 _ = _ ∧
    (tabSrcView PE x (czExe PE x c) (viewsE x c w τ) i).salt0 _ = _
  unfold tabSrcView
  rw [dite_eq_left hi]
  dsimp only
  rw [at0_s0, at0_s0]
  refine ⟨((congrArg ZkView.row0 e).symm.trans hrow).trans ?_, (congrArg ZkView.saltBytes es).trans hsl⟩
  rw [FlockComplete.tablesP, FlockSoundness.Refine.getElem!_range_map _ hj]
  rfl

/-- **A queried position below `L` is opened** (`FinPads.padPos_of_mem`): where at most `t` positions are queried,
`padPos` places each queried position below `L` at some index below `t`, since it lists them first. -/
theorem padPos_mem {L t : ℕ} {qs : List ℕ} (hl : qs.length ≤ t) {q : ℕ} (hq : q ∈ qs) (hqL : q < L) :
    ∃ a : Fin t, Composed.padPos L t qs a = q := by
  have hFn : ((List.range L).filter fun y => decide (y ∈ qs)).Nodup := List.nodup_range.filter _
  have hFs : (List.range L).filter (fun y => decide (y ∈ qs)) ⊆ qs := fun y hy => by
    simpa using (List.mem_filter.1 hy).2
  have hFl := (hFn.subperm hFs).length_le
  have hqF : q ∈ (List.range L).filter fun y => decide (y ∈ qs) :=
    List.mem_filter.2 ⟨List.mem_range.2 hqL, decide_eq_true hq⟩
  obtain ⟨k, hk, hkq⟩ := List.getElem_of_mem hqF
  refine ⟨⟨k, by omega⟩, ?_⟩
  simp only [Composed.padPos]
  rw [List.getD_eq_getElem _ _ (by rw [List.length_append]; omega), List.getElem_append_left hk]
  exact hkq

/-- **Table `j`'s rep `r`'s level-0 query positions are opened positions**: each is below `2^d` (`positions_lt_d`),
queried by the coins (`q0s`, below `2^logLen`), of which there are `2·q = t_pad`, so `padPos` opens it, at the tree
position of some opened position (`coinedOf_s0_val`). -/
theorem q0At_opened {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) (r : Fin 2) {p : ℕ}
    (hp : p ∈ q0At c j r) :
    p < 2 ^ (lv0 x j).d ∧ ∃ a : Fin (tabC PE x j).z.nS, (((tabC PE x j).T (czExe PE x c j)).s0 (.inl a)).val = p := by
  obtain ⟨-, -, -, -, ht2, hdim, hlv, -, -⟩ := FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ (zkTab x j).hz
  obtain ⟨-, hL2, -, -, -, -, -⟩ := hdim
  obtain ⟨hd, -, -, -, -, -⟩ := zkLevel0 (zkTab x j).hz
  obtain ⟨hrate, -⟩ := FlockSoundness.Discharge.ZkExec.zk_schedule_pad _ _ _ (zkTab x j).hz
  obtain ⟨hts, hLs, hcs⟩ := tabSizes x j
  have hseam : FlockSoundness.Refine.LevelSeam (lv0 x j) := hlv 0 (by omega)
  have hlt : p < 2 ^ (lv0 x j).d := positions_lt_d hseam _ p hp
  refine ⟨hlt, ?_⟩
  have hd1 : (lv0 x j).d = (zkTab x j).table.sch.level₀.logCols + 1 := by
    rw [hcs]
    unfold lv0
    omega
  obtain ⟨e, he, hep⟩ := Array.mem_iff_getElem.1 hp
  have hsz : (q0At c j r).size = (lv0 x j).q := FlockSoundness.Refine.positions_size _ hseam _
  have he2 : e < (zkTab x j).table.t / 2 := by
    show e < (zkTab x j).tPad / 2
    rw [ht2]
    unfold lv0 at hsz
    omega
  have hmem : p ∈ (Composed.bodyOf (zkTab x j).table (stAt x j)).queries (czExe PE x c j).rest := by
    show p ∈ (List.ofFn ((czExe PE x c j).rest.q0s 0) ++ List.ofFn ((czExe PE x c j).rest.q0s 1)).map Fin.val
    have hv : ((czExe PE x c j).rest.q0s r ⟨e, he2⟩).val = p := by
      show (q0At c j r)[e]! % 2 ^ (zkTab x j).table.sch.level₀.logLen = p
      rw [getElem!_pos _ e he, hep,
        Nat.mod_eq_of_lt (lt_of_lt_of_le hlt (Nat.pow_le_pow_right (by norm_num) (by omega)))]
    refine List.mem_map.2 ⟨_, ?_, hv⟩
    fin_cases r
    · exact List.mem_append_left _ (List.mem_ofFn.2 ⟨_, rfl⟩)
    · exact List.mem_append_right _ (List.mem_ofFn.2 ⟨_, rfl⟩)
  have hlen : ((Composed.bodyOf (zkTab x j).table (stAt x j)).queries (czExe PE x c j).rest).length ≤
      (zkTab x j).table.t := by
    show ((List.ofFn ((czExe PE x c j).rest.q0s 0) ++ List.ofFn ((czExe PE x c j).rest.q0s 1)).map Fin.val).length ≤ _
    simp only [List.length_map, List.length_append, List.length_ofFn]
    omega
  obtain ⟨a, ha⟩ := padPos_mem hlen hmem
    (show p < 2 ^ ((zkTab x j).table.sch.level₀.logCols + 1) by rw [← hd1]; exact hlt)
  exact ⟨a, (Composed.coinedOf_s0_val PE _ _ _ hts hLs (.inl a)).trans ha⟩

/-- **Level 0's commitment from the views agrees with the session's at the queried positions**: there `fill0`'s row
and salt are the views' (`srcV_open0`), which are the session's commitment's (`commitRowsP_rows`,
`commitRowsP_salts`, at `hm96-sha512/v1`'s 192-byte salts). -/
theorem fill0_open0 {x : Msg} {s spec} (hx : Live x) (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) {i : ℕ} (hi : i < jOf x) {r : ℕ}
    (hr : r < 2) {cm0 : Commit}
    (hcm : commitPZk hsV (FlockDeploy.tabsOf (stsOf x))[i]!.merkle (lv0 x ⟨i, hi⟩) (zkTab x ⟨i, hi⟩).tPad
      (packed (trsE x w (UE.tapeZk τ))[i]! (FlockDeploy.tabsOf (stsOf x))[i]!.m)
      ((UE.tapeZk τ).extra i) ((UE.tapeZk τ).lanePad i) ((UE.tapeZk τ).salt i 0 0) = .ok cm0)
    {p : ℕ} (hp : p ∈ q0At c ⟨i, hi⟩ ⟨r, hr⟩) :
    (fill0 x c w τ i (lv0 x ⟨i, hi⟩) cm0).rows[p]! = cm0.rows[p]! ∧
      (fill0 x c w τ i (lv0 x ⟨i, hi⟩) cm0).salts[p]! = cm0.salts[p]! := by
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hi' : i < (stsOf x).size := lt_of_jOf ⟨i, hi⟩
  have hget : (FlockDeploy.tabsOf (stsOf x))[i]! = _root_.Flock.Setup.ofCircuit (stAt x ⟨i, hi⟩) := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) i (by simpa [FlockDeploy.tabsOf] using hi')]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    rfl
  have hm96 : (FlockDeploy.tabsOf (stsOf x))[i]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    rw [hget]
    exact (hall i hi').2.2
  have hms : (FlockDeploy.tabsOf (stsOf x))[i]!.merkle.saltBytes = 192 := by
    rw [hm96]
    rfl
  obtain ⟨hlt, a, ha⟩ := q0At_opened c ⟨i, hi⟩ ⟨r, hr⟩ hp
  obtain ⟨hrow, hsalt⟩ := srcV_open0 x c w τ hi a
  rw [ha] at hrow hsalt
  unfold commitPZk at hcm
  have hrows := commitRowsP_rows hcm
  have hsalts := commitRowsP_salts hcm
  refine ⟨?_, ?_⟩
  · show ((Array.range (2 ^ (lv0 x ⟨i, hi⟩).d)).map (srcV x c w τ i).row0)[p]! = _
    rw [FlockSoundness.Refine.getElem!_range_map _ hlt, hrow, hrows]
  · show ((Array.range (2 ^ (lv0 x ⟨i, hi⟩).d)).map (srcV x c w τ i).salt0)[p]! = _
    rw [FlockSoundness.Refine.getElem!_range_map _ hlt, hsalt, hsalts, FlockSoundness.Refine.getElem!_range_map _ hlt,
      hms]

end FinRep

/-- **Level 0's opened rows and salts read from the views are the session's** (`FinalRest`'s level-0 half): in a rep's
context, where the rep read from the views and the session's rep both return, both open level 0 at the coins of
Ligerito's round `k₀ + 8` after the PIOP (`FinRep.repCut_open0`, `FinRep.repPZk_open0`), the slot `repSlot` names, so
at `q0At`'s positions, where `fill0` is the session's commitment (`FinRep.fill0_open0`). -/
theorem repOpenRest_l0 : ∀ x w τ c spec i hi r cm0 extras ic l, RepCtx x w τ c spec i hi r cm0 extras ic l →
    ∀ p q t l', (repVE x w τ c spec i hi r cm0 extras ic).go l = (.ok (p, t), l') →
      (repHE x w τ c spec i hi r cm0 extras ic).go l = (.ok (q, t), l') →
      FlockComplete.eraseZ p = FlockComplete.eraseZ q →
        p.proof.ligerito.initialProof.rows = q.proof.ligerito.initialProof.rows ∧
          p.proof.ligerito.initialProof.salts = q.proof.ligerito.initialProof.salts := by
  intro x w τ c spec i hi r cm0 extras ic l hC p q t l' hp hq _
  obtain ⟨e', hv, hm, hk6, hx'⟩ := FinRep.repHE_asks hC hq
  obtain ⟨hx, hw, hA, ⟨s, hsx⟩, hr, hcm, -, he, hl⟩ := hC
  obtain ⟨-, -, hsm, hk0, -, hdim, hlv, -, -⟩ := FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ hv
  obtain ⟨hn7, hL2, -, -, -, -, -⟩ := hdim
  have hi' : i < (stsOf x).size := lt_of_jOf ⟨i, hi⟩
  have htab : (FlockDeploy.tabsOf (stsOf x))[i]! = _root_.Flock.Setup.ofCircuit (stsOf x)[i]! := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) i (by simpa [FlockDeploy.tabsOf] using hi'),
      getElem!_pos (stsOf x) i hi']
    simp [FlockDeploy.tabsOf]
  have hQ := FinRep.repPZk_open0 (hm := hm) (hn := by rw [hn7, hsm]; omega) (hk0 := hk0) (hL := hL2)
    (hq := fun i hi => (hlv i hi).q_ne_zero) (ht := rfl) (h := e')
  obtain ⟨-, hsv⟩ := piopSrcE x hx w hw τ c (refK x) (not_refused_of_admits hx hA (refK x)) s spec hsx ⟨i, hi⟩
    ⟨r, hr⟩ extras he
  dsimp only at hsv
  rw [← eraseS_srcV hA w τ hi hr, ← hl] at hsv
  have hP := FinRep.repCut_open0 (coin := fun k i => slotAt c k i) (n := refK x) (resp := FinRep.respOf c)
    (fun k => FinRep.coinOf_respOf c k) _ _ _ _ _ _ _ _ _ _ _ _ _ (by rw [htab]; exact hm)
    (by rw [hn7, hsm]; omega) hk0 ((hlv 0 (by omega)).q_ne_zero) rfl (FinRep.piopSrc_of_eraseS hsv) hp
  rw [htab] at hP
  have hslot : l.size + (FlockComplete.piopZk (_root_.Flock.Setup.ofCircuit (stsOf x)[i]!) extras +
      (zkTab x ⟨i, hi⟩).sch.k0 + 8) = repSlot x (2 * i + r) (fun sch => sch.k0 + 8) := by
    have hz : _root_.Flock.Zk.schedule (stsOf x)[(2 * i + r) / 2]!.m =
        .ok ((zkTab x ⟨i, hi⟩).sch, (zkTab x ⟨i, hi⟩).tPad) := by
      rw [show (2 * i + r) / 2 = i by omega]
      exact hv
    have e6 : (_root_.Flock.Setup.ofCircuit (stsOf x)[i]!).fold.kLog = (stsOf x)[i]!.c.kLog := rfl
    have e7 : (_root_.Flock.Setup.ofCircuit (stsOf x)[i]!).m = (stsOf x)[i]!.m := rfl
    rw [repSlot_eq hsx hz, show (2 * i + r) / 2 = i by omega, hl, repBase_eq hsx]
    unfold FlockComplete.piopZk FlockComplete.piopRoundsZk
    rw [e6, e7, hx']
    omega
  rw [FinRep.coinOf_respOf, hslot] at hP hQ
  rw [hP, hQ]
  have key : ∀ p ∈ q0At c ⟨i, hi⟩ ⟨r, hr⟩,
      (fill0 x c w τ i (lv0 x ⟨i, hi⟩) cm0).rows[p]! = cm0.rows[p]! ∧
        (fill0 x c w τ i (lv0 x ⟨i, hi⟩) cm0).salts[p]! = cm0.salts[p]! :=
    fun p hp => FinRep.fill0_open0 hx hsx c w τ hi hr hcm hp
  unfold FlockComplete.openP
  refine ⟨Array.map_congr_left fun p hp => (key p hp).1, ?_⟩
  dsimp only
  split_ifs
  · rfl
  · exact Array.map_congr_left fun p hp => (key p hp).2

end Masked

end Claims.ZKMal
