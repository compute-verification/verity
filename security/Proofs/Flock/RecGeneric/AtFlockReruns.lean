module

public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.RecGeneric.AtFlockPlans
public import Proofs.Flock.RecGeneric.BadDev

/-!
# L1 at Flock: the composite's reruns are `Claims.Recursion`'s samples

At today's compiled record (`Comp.ofRec D IV I DO decO`, whose reruns `Comp.innerModel IV` and `Comp.verModel IV`
read) each definition of a rerun's view agrees with its counterpart over `Claims.Recursion`'s samples:

* the inner view `E_O`'s witnesses give is the record's (`dec_ofRec`, from `RecAt.runs`), so a rerun is accepted in
  `innerModel` exactly where `accAtI` holds (`innerModel_acc_iff`);
* an accepted rerun has witnesses (`wits_of_acc`), and its searched evaluations are `Deploy.callsAt` at `drawWits`
  (`innerModel_log_of_acc`); at every rerun `innerModel`'s and `verModel`'s are among the sample's `drawCalls`
  (`rerunCalls_sub`, `verModel_log_sub`), which are true where the calls at given witnesses and those of no witnesses
  are (`drawCalls_true_at`);
* a run finder's coins map uniformly onto `BinCoins` (`binEquiv`) and `ConflictCoinsI` (`conflictEquiv`), and onto
  `CapCoinsI` by `Fin.consEquiv`, the fresh runs read through `slotRuns` (`slotRuns_le`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-! ## Forks, loops and slots -/

section Pick

variable {U : Type}

theorem firstPick_some {g : U → Prop} : ∀ {M : ℕ} {us : Fin M → U} {v : U}, firstPick g M us = some v → g v
  | 0, _, _, h => by simp [firstPick] at h
  | M + 1, us, v, h => by
    unfold firstPick at h
    split_ifs at h with hg
    · exact Option.some.inj h ▸ hg
    · exact firstPick_some h

/-- **A loop read through `f`** picks the image of its pick, where `g` is `g'` read through `f`. -/
theorem firstPick_map {U' : Type} (f : U → U') (g : U → Prop) (g' : U' → Prop) (h : ∀ u, g u ↔ g' (f u)) :
    ∀ (M : ℕ) (us : Fin M → U), (firstPick g M us).map f = firstPick g' M (fun i => f (us i))
  | 0, _ => rfl
  | M + 1, us => by
    by_cases hg : g (us 0)
    · simp [firstPick, hg, (h _).1 hg]
    · have hg' : ¬ g' (f (us 0)) := fun h' => hg ((h _).2 h')
      simp only [firstPick, hg, hg', ↓reduceIte]
      exact firstPick_map f g g' h M (Fin.tail us)

/-- Two maps that agree where `g` holds agree on the loop's pick. -/
theorem firstPick_map_congr {β : Type} {g : U → Prop} {f f' : U → β} (h : ∀ u, g u → f u = f' u) (M : ℕ)
    (us : Fin M → U) : (firstPick g M us).map f = (firstPick g M us).map f' := by
  rcases hp : firstPick g M us with _ | v
  · rfl
  · exact congrArg some (h v (firstPick_some hp))

end Pick

theorem flatMap_toList {α β γ : Type} (l : List α) (o : α → Option β) (L : β → List γ) :
    (l.flatMap fun a => (o a).toList).flatMap L = l.flatMap fun a => ((o a).map L).getD [] := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [List.flatMap_cons, List.flatMap_append, ih]
    cases o a <;> simp

section Slots

variable {Coin C : Type} {n : ℕ}

theorem slotRuns_le {s : ℕ} (us : Fin (s + 1) → (Fin n → Coin) × C) {i : ℕ} (h : i ≤ s) :
    slotRuns us i = us ⟨i, Nat.lt_succ_of_le h⟩ :=
  congrArg us (Fin.ext (Nat.min_eq_left h))

theorem slotRuns_cons_one_add {s : ℕ} (u : (Fin n → Coin) × C) (vs : Fin s → (Fin n → Coin) × C) (t : Fin s) :
    slotRuns (Fin.cons u vs : Fin (s + 1) → (Fin n → Coin) × C) (1 + t.val) = vs t := by
  rw [slotRuns_le _ (by omega : 1 + t.val ≤ s)]
  have ht : (⟨1 + t.val, by omega⟩ : Fin (s + 1)) = t.succ := Fin.ext (by simp only [Fin.val_succ]; omega)
  rw [ht]
  exact Fin.cons_succ (α := fun _ => (Fin n → Coin) × C) u vs t

end Slots

/-! ## A rerun's view at the compiled record -/

section Bridge

variable {Coin W Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col']
  (D : RecAt Coin) (IV : Verifier Coin) (I : Claims.Recursion.Inner Coin W) (DO : Claims.Flock.Deploy Coin Col' T')
  (decO : List (ℕ × Col') → List T') {H : Msg → Msg} {Ω : Type} {P : Prover Ω} {x : Msg}

/-- **The inner view `E_O`'s output gives is the record's** (`RecAt.runs`), where the record's inner statement is the
compiled one's. -/
theorem dec_ofRec (hxI : D.xI x = Claims.Recursion.xI (D.record IV) x) (r : Fin ((D.record IV).V.k x) → Coin) :
    ∀ o, (Comp.ofRec D IV I DO decO).dec IV H x r o = Claims.Recursion.dec H (D.record IV) x r o
  | none => rfl
  | some ws => by
    show (run H (replay (D.msgs x ws)) IV (Claims.Recursion.xI (D.record IV) x) () fun i => coinSeq r i.val).seen =
      (D.record IV).innerView H x r ws
    rw [D.runs IV H x r ws, hxI]

/-- **A rerun is accepted in `innerModel` exactly where its sample's inner view accepts** (`accAtI`). -/
theorem innerModel_acc_iff (hxI : D.xI x = Claims.Recursion.xI (D.record IV) x) (ω : Ω)
    (u : (Fin ((D.record IV).V.k x) → Coin) × (EO (D.record IV) I DO decO).C x) :
    ((Comp.ofRec D IV I DO decO).innerModel IV H P x).acc ω u ↔ accAtI H (D.record IV) I DO decO P x ω u := by
  change ((Comp.ofRec D IV I DO decO).dec IV H x u.1 ((Comp.ofRec D IV I DO decO).wits H P x ω u)).out = .acc ↔ _
  rw [dec_ofRec D IV I DO decO hxI]
  exact Iff.rfl

/-- An accepted rerun has witnesses: the inner view of none rejects. -/
theorem wits_of_acc {ω : Ω} {u : (Fin ((D.record IV).V.k x) → Coin) × (EO (D.record IV) I DO decO).C x}
    (h : ((Comp.ofRec D IV I DO decO).innerModel IV H P x).acc ω u) :
    ∃ ws, (Comp.ofRec D IV I DO decO).wits H P x ω u = some ws := by
  change ((Comp.ofRec D IV I DO decO).dec IV H x u.1 ((Comp.ofRec D IV I DO decO).wits H P x ω u)).out = .acc at h
  rcases hw : (Comp.ofRec D IV I DO decO).wits H P x ω u with _ | ws
  · rw [hw] at h
    simp [Comp.dec] at h
  · exact ⟨ws, rfl⟩

theorem drawWits_of_wits {ω : Ω} {u : (Fin ((D.record IV).V.k x) → Coin) × (EO (D.record IV) I DO decO).C x}
    {ws : List Claims.Flock.Wit} (hw : (Comp.ofRec D IV I DO decO).wits H P x ω u = some ws) :
    drawWits H (D.record IV) I DO decO P x ω u = ws := by
  have h' : ((EO (D.record IV) I DO decO).eval H P x ω u.1 u.2).1 = some ws := hw
  unfold drawWits
  rw [h', Option.getD_some]

/-- **An accepted rerun searches its sample's recorded and inner calls** (`Deploy.callsAt` at `drawWits`). -/
theorem innerModel_log_of_acc {ω : Ω}
    {u : (Fin ((D.record IV).V.k x) → Coin) × (EO (D.record IV) I DO decO).C x}
    (h : ((Comp.ofRec D IV I DO decO).innerModel IV H P x).acc ω u) :
    ((Comp.ofRec D IV I DO decO).innerModel IV H P x).log ω u =
      (D.record IV).callsAt H x u.1 (drawWits H (D.record IV) I DO decO P x ω u) := by
  obtain ⟨ws, hw⟩ := wits_of_acc D IV I DO decO h
  rw [drawWits_of_wits D IV I DO decO hw]
  exact Comp.rerunCalls_some hw

/-- **A rerun's searched evaluations are among its sample's** (`drawCalls`). -/
theorem rerunCalls_sub {ω : Ω} {u : (Fin ((D.record IV).V.k x) → Coin) × (EO (D.record IV) I DO decO).C x}
    {e : Msg × Msg} (he : e ∈ (Comp.ofRec D IV I DO decO).rerunCalls H P x ω u) :
    e ∈ drawCalls H (D.record IV) I DO decO P x ω u := by
  rcases hw : (Comp.ofRec D IV I DO decO).wits H P x ω u with _ | ws
  · unfold Comp.rerunCalls at he
    rw [hw] at he
    simp at he
  · rw [Comp.rerunCalls_some hw] at he
    unfold drawCalls
    rw [drawWits_of_wits D IV I DO decO hw]
    exact List.mem_append_left _ he

/-- **`verModel`'s evaluations at a rerun are among its sample's**: the searched ones, then the compiled verifier's own
at the run's staged statements, which are `drawCalls`' second half. -/
theorem verModel_log_sub {ω : Ω} {u : (Fin ((D.record IV).V.k x) → Coin) × (EO (D.record IV) I DO decO).C x}
    {e : Msg × Msg} (he : e ∈ ((Comp.ofRec D IV I DO decO).verModel IV H P x).log ω u) :
    e ∈ drawCalls H (D.record IV) I DO decO P x ω u := by
  rcases List.mem_append.1 he with he | he
  · exact rerunCalls_sub D IV I DO decO he
  · unfold drawCalls
    exact List.mem_append_right _ he

/-- **A sample's evaluations are true** at a statement where those the compiled record's witnesses record are
(`CallsTrue` at `x`) and those of no witnesses are: `E_O` gives witnesses only on accepted runs (`EO_some_acc`), and
every other evaluation is `(u, H u)`. -/
theorem drawCalls_true_at (hT : (Comp.ofRec D IV I DO decO).CallsTrue H x)
    (hN : ∀ e ∈ (D.record IV).calls x [], H e.1 = e.2) (ω : Ω)
    (d : (Fin ((D.record IV).V.k x) → Coin) × (EO (D.record IV) I DO decO).C x) :
    ∀ e ∈ drawCalls H (D.record IV) I DO decO P x ω d, H e.1 = e.2 := by
  intro e he
  unfold drawCalls Deploy.callsAt at he
  rcases List.mem_append.1 he with he | he
  · rcases List.mem_append.1 he with he | he
    · rcases hw : ((EO (D.record IV) I DO decO).eval H P x ω d.1 d.2).1 with _ | ws
      · have h0 : drawWits H (D.record IV) I DO decO P x ω d = [] := by
          unfold drawWits
          rw [hw, Option.getD_none]
        rw [h0] at he
        exact hN e he
      · have h0 : drawWits H (D.record IV) I DO decO P x ω d = ws := by
          unfold drawWits
          rw [hw, Option.getD_some]
        rw [h0] at he
        exact hT Ω P ω d.1 d.2 ws ⟨EO_some_acc H (D.record IV) I DO decO P x ω d.1 d.2 hw, hw⟩ e
          (List.mem_append_left _ he)
    · obtain ⟨u, -, rfl⟩ := List.mem_map.1 he
      rfl
  · unfold setupCalls at he
    obtain ⟨xO, -, he⟩ := List.mem_flatMap.1 he
    obtain ⟨u, -, rfl⟩ := List.mem_map.1 he
    rfl

end Bridge

/-! ## Run finders' coins as the samples' -/

/-- **`B_I`'s coins as `forkPlan`'s**: the round, the tape, and the two runs, each its compiled coins and `E_O`'s. -/
def binEquiv (J Ω A B : Type) : J × Ω × A × A × B × B ≃ J × Ω × (Fin (1 + 1) → A × B) where
  toFun s := (s.1, s.2.1, ![(s.2.2.1, s.2.2.2.2.1), (s.2.2.2.1, s.2.2.2.2.2)])
  invFun t := (t.1, t.2.1, (t.2.2 0).1, (t.2.2 1).1, (t.2.2 0).2, (t.2.2 1).2)
  left_inv _ := rfl
  right_inv t := by
    refine Prod.ext rfl (Prod.ext rfl (funext fun i => ?_))
    fin_cases i <;> rfl

/-- **`F0I`'s coins as `conflictPlan`'s**: the selector's one value, the tape, the first run, then the waits' runs, run
`r` of wait `t` in slot `1 + R·t + r`. -/
def conflictEquiv (Ω U : Type) (k R : ℕ) : Ω × U × (Fin k → Fin R → U) ≃ Fin 1 × Ω × (Fin (k * R + 1) → U) :=
  (Equiv.prodCongr (Equiv.refl Ω) ((Equiv.prodCongr (Equiv.refl U)
    ((Equiv.curry (Fin k) (Fin R) U).symm.trans (Equiv.arrowCongr finProdFinEquiv (Equiv.refl U)))).trans
      (Fin.consEquiv fun _ : Fin (k * R + 1) => U))).trans (Equiv.uniqueProd _ (Fin 1)).symm

theorem conflictEquiv_apply (Ω U : Type) (k R : ℕ) (s : Ω × U × (Fin k → Fin R → U)) :
    conflictEquiv Ω U k R s =
      (default, s.1, Fin.cons s.2.1 fun p => s.2.2 (finProdFinEquiv.symm p).1 (finProdFinEquiv.symm p).2) :=
  rfl

theorem wait_slot_le {k R : ℕ} (t : Fin k) (r : Fin R) : 1 + R * t.val + r.val ≤ k * R := by
  have h := Nat.mul_le_mul_left R (Nat.succ_le_of_lt t.isLt)
  have hr := r.isLt
  rw [Nat.mul_succ] at h
  rw [Nat.mul_comm k R]
  omega

/-- The slot of run `r` of wait `t` holds that run. -/
theorem slotRuns_wait {Coin C : Type} {n k R : ℕ} (u : (Fin n → Coin) × C)
    (ws : Fin k → Fin R → (Fin n → Coin) × C) (t : Fin k) (r : Fin R) :
    slotRuns (Fin.cons u fun p => ws (finProdFinEquiv.symm p).1 (finProdFinEquiv.symm p).2 :
      Fin (k * R + 1) → (Fin n → Coin) × C) (1 + R * t.val + r.val) = ws t r := by
  rw [slotRuns_le _ (wait_slot_le t r)]
  have hp : (⟨1 + R * t.val + r.val, Nat.lt_succ_of_le (wait_slot_le t r)⟩ : Fin (k * R + 1)) =
      (finProdFinEquiv (t, r)).succ :=
    Fin.ext (by
      show 1 + R * t.val + r.val = r.val + R * t.val + 1
      omega)
  rw [hp, Fin.cons_succ, Equiv.symm_apply_apply]

end Claims.RecGeneric
