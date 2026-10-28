import Proofs.Flock.Claims.Recursion.DecodeLog
import Proofs.Flock.Claims.DeployOC

/-!
# The plain verdict's SHA-512 calls

`verdict_log`: a run of `verifyWithM` with the logging oracle appends the decoder's calls, S17's digests of the proof
files, and each table's two reps' own calls (`repLog`, what the rep appends to the empty log), and nothing else.
`verifyRepWith_congr` is `verifyRep_congr` for any oracle.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness.Refine

theorem hOf_toMsg (h : ByteArray → ByteArray) : Claims.hOf (fun m => Claims.toMsg (h (Claims.ofMsg m))) = h := by
  funext b
  simp [Claims.hOf]

/-- A `LogV` run that simulates an oracle program appends to any log what it appends to the empty one. -/
theorem go_shift {α : Type} {H : Claims.Msg → Claims.Msg} {x : Claims.OCV α} {y : LogV α} (hR : (Claims.ocLog H).R x y)
    (l : Array ByteArray) : y.go l = ((y.go #[]).1, l ++ (y.go #[]).2) := by
  obtain ⟨a1, a2⟩ := hR l
  obtain ⟨b1, b2⟩ := hR #[]
  refine Prod.ext (a1.trans b1.symm) (Array.ext' ?_)
  rw [a2, Array.toList_append, b2]
  simp

/-- A rep's own SHA-512 calls: what it appends to the empty log. -/
def repLog (h : ByteArray → ByteArray) (spec : SessionSpec) (st : Setup) (sch : Schedule) (sess : Session) (i r : ℕ)
    (bytes : ByteArray) : List ByteArray :=
  ((verifyRepWith (Hashes.logged h) spec st sch sess i r bytes).go #[]).2.toList

theorem rep_shift (h : ByteArray → ByteArray) (spec : SessionSpec) (st : Setup) (sch : Schedule) (sess : Session)
    (i r : ℕ) (bytes : ByteArray) (l : Array ByteArray) :
    (verifyRepWith (Hashes.logged h) spec st sch sess i r bytes).go l =
      (((verifyRepWith (Hashes.logged h) spec st sch sess i r bytes).go #[]).1,
        l ++ ((verifyRepWith (Hashes.logged h) spec st sch sess i r bytes).go #[]).2) := by
  have hR := With.verifyRepWith_sim (Claims.ocLog (fun m => Claims.toMsg (h (Claims.ofMsg m))))
    (Claims.ocLog_hashes _) spec st sch sess i r bytes
  rw [hOf_toMsg] at hR
  exact go_shift hR l

/-- **A rep reads its session only through its stream's rounds, the points and its table's `root_B`**, whatever the
oracle. -/
theorem verifyRepWith_congr {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT Flock.V M]
    (hs : Hashes M)
    {spec : SessionSpec} {st : Setup} {sch : Flock.Schedule} {sess sess' : Session} {i r : ℕ}
    (bytes : ByteArray) {y y' : StreamRec}
    (hy : sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some y)
    (hy' : sess'.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some y')
    (hr : y.rounds = y'.rounds) (hP : sess.points = sess'.points) (hB : sess.rootB[i]! = sess'.rootB[i]!) :
    verifyRepWith hs spec st sch sess i r bytes = verifyRepWith hs spec st sch sess' i r bytes := by
  unfold verifyRepWith
  dsimp only
  rw [hy, hy']
  dsimp only
  rw [hP, hB, hr]

/-- `verdict_log` at `verifyWithMK keep`. -/
theorem verdictK_log (keep : Bool → Bool) {h : ByteArray → ByteArray} {spec : SessionSpec} {tabs : Array Setup}
    {record : String} {proofs : Array ByteArray} {l₀ : Array ByteArray} :
    RetL (verifyWithMK keep (Hashes.logged h) spec tabs record proofs) l₀ fun _ l' =>
      ∃ (sess : Session) (schs : Array Schedule) (l₁ : Array ByteArray),
        (Record.decodeWithK keep (Hashes.logged h) spec record).go l₀ = (.ok sess, l₁) ∧
        List.Forall₂ (fun st sch => fast100 st.m = some sch) tabs.toList schs.toList ∧
        tabs.size = spec.tables.size ∧ proofs.size = spec.repStreams.size ∧
        l'.toList = l₁.toList ++ (spec.repStreams.zip proofs).toList.flatMap (fun x => dgl spec x.2) ++
          (List.range tabs.size).flatMap fun i =>
            repLog h spec tabs[i]! schs[i]! sess i 0 proofs[2 * i]! ++
              repLog h spec tabs[i]! schs[i]! sess i 1 proofs[2 * i + 1]! := by
  unfold verifyWithMK
  retl_walk []
  all_goals
    subst_vars
    have hp := ‹StV.go (forIn (tabs.zip sess.publics) _ _) _ = _›
    have h17 := ‹StV.go (forIn (spec.repStreams.zip proofs) _ _) _ = _›
    have hr := ‹StV.go (forIn [:tabs.size] _ _) _ = _›
    obtain ⟨dp, fp, gp⟩ := forIn_log (R := fun _ d => d = []) (fun x _ l => by
      obtain ⟨st, p⟩ := x
      dsimp only
      retl_walk []
      exact ⟨rfl, [], by simp, rfl⟩) hp
    obtain ⟨ys, es, fs, gs⟩ := mapM_log (R := fun st sch d => fast100 st.m = some sch ∧ d = []) (fun st l => by
      retl_walk []
      all_goals exact ⟨[], by simp, by assumption, rfl⟩) h_schs
    obtain ⟨d17, f17, g17⟩ := forIn_log (R := fun x d => d = dgl spec x.2) (fun x _ l => by
      obtain ⟨⟨_, _, w⟩, b⟩ := x
      dsimp only
      retl_walk []
      all_goals exact ⟨rfl, _, by simp [dgL_toList], rfl⟩) h17
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hr
    simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, ← List.range_eq_range'] at hr
    obtain ⟨dr, fr, gr⟩ := list_forIn_log (R := fun i d =>
        d = repLog h spec tabs[i]! schs[i]! sess i 0 proofs[2 * i]! ++
          repLog h spec tabs[i]! schs[i]! sess i 1 proofs[2 * i + 1]!) _ (fun i _ l => by
      retl_walk []
      all_goals
        have e0 := (Prod.mk.inj ((rep_shift ..).symm.trans h_c0)).2
        have e1 := (Prod.mk.inj ((rep_shift ..).symm.trans h_c1)).2
        refine ⟨rfl, _, ?_, rfl⟩
        rw [← e1, ← e0]
        simp [repLog]) _ _ _ hr
    have hc1 := ‹¬(tabs.size != spec.tables.size || tabs.size == 0) = true›
    have hc2 := ‹¬(proofs.size != spec.repStreams.size) = true›
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, not_or, Decidable.not_not] at hc1 hc2
    refine ⟨sess, schs, l_sess, h_sess, ?_, hc1.1, hc2, ?_⟩
    · rw [es]
      exact List.forall₂_map_right_iff.2 (fs.imp fun _ _ h => h.1)
    · rw [gr, g17, gs, gp, flatten_eq_flatMap (fun _ _ h => h) fp, flatMap_nil',
        flatten_snd (R := fun (x : Setup) y => fast100 x.m = some y) (F := fun _ => []) fs, flatMap_nil',
        flatten_eq_flatMap (fun _ _ h => h) f17, flatten_eq_flatMap (fun _ _ h => h) fr]
      simp only [List.append_nil, List.append_assoc]

theorem verdict_log {h : ByteArray → ByteArray} {spec : SessionSpec} {tabs : Array Setup} {record : String}
    {proofs : Array ByteArray} {l₀ : Array ByteArray} :
    RetL (verifyWithM (Hashes.logged h) spec tabs record proofs) l₀ fun _ l' =>
      ∃ (sess : Session) (schs : Array Schedule) (l₁ : Array ByteArray),
        (Record.decodeWith (Hashes.logged h) spec record).go l₀ = (.ok sess, l₁) ∧
        List.Forall₂ (fun st sch => fast100 st.m = some sch) tabs.toList schs.toList ∧
        tabs.size = spec.tables.size ∧ proofs.size = spec.repStreams.size ∧
        l'.toList = l₁.toList ++ (spec.repStreams.zip proofs).toList.flatMap (fun x => dgl spec x.2) ++
          (List.range tabs.size).flatMap fun i =>
            repLog h spec tabs[i]! schs[i]! sess i 0 proofs[2 * i]! ++
              repLog h spec tabs[i]! schs[i]! sess i 1 proofs[2 * i + 1]! :=
  verdictK_log id

end FlockDeploy.KS
