import Definitions.Accounting.Guarantees.TS.FoldFanIn
import Proofs.Accounting.TS.FanIn
import Proofs.Pouw.Receipt

/-!
# The fan-in floor of a fold

* `card_le_of_out`: outputs `out` that hold a value injective in each input see every input, so `FanInCount` and
  at most `f` sources a node give `|ι| ≤ c + (f − 1)·n`.
* `FoldFanIn`: a fold of binding steps is injective in each word (`foldl_ne`: the words before it reach one state,
  its step is injective in the word, and the steps after it are injective in the state).
* `DuoFoldFanIn`, `Fx1FoldFanIn`: duo's and fx1's runs are injective in each word
  (`Pouw.SecurityProofs.Receipt.DuoRunWordInjective`, `Fx1RunWordInjective`).
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Finset

namespace Fold

/-- Outputs holding a value that moves with each input, over nodes of at most `f` sources: `|ι| ≤ c + (f − 1)·n`. -/
theorem card_le_of_out {ι W : Type} [Fintype ι] [Nontrivial W] [Inhabited W] {c f : ℕ} (D : Dataflow ι W)
    (out : Fin c → ι ⊕ Fin D.n) (F : (ι → W) → Fin c → W) (hF : ∀ x v, D.Runs x v → (fun o => v (out o)) = F x)
    (hinj : ∀ j x x', (∀ j', j' ≠ j → x j' = x' j') → x j ≠ x' j → F x ≠ F x')
    (hf : ∀ i, (D.src i).card ≤ f) : Fintype.card ι ≤ c + (f - 1) * D.n := by
  classical
  have hO : D.Sees (univ.image out) := by
    intro j x x' v v' hv hv' hx hne
    have h := hinj j x x' hx hne
    rw [← hF x v hv, ← hF x' v' hv'] at h
    obtain ⟨o, ho⟩ := Function.ne_iff.1 h
    exact ⟨out o, mem_image_of_mem out (mem_univ o), ho⟩
  have h1 := FanInCount D (univ.image out) hO
  have h2 : (univ.image out).card ≤ c := card_image_le.trans (by simp)
  have h3 : ∑ i, D.merges i ≤ (f - 1) * D.n := by
    calc ∑ i, D.merges i ≤ ∑ _i : Fin D.n, (f - 1) := sum_le_sum fun i _ => Nat.sub_le_sub_right (hf i) 1
      _ = (f - 1) * D.n := by simp [mul_comm]
  omega

/-- `card_le_of_out` for two outputs holding a pair. -/
theorem card_le_of_pair {ι W : Type} [Fintype ι] [Nontrivial W] [Inhabited W] {f : ℕ} (D : Dataflow ι W)
    (out : Fin 2 → ι ⊕ Fin D.n) (P : (ι → W) → W × W) (hP : ∀ x v, D.Runs x v → (v (out 0), v (out 1)) = P x)
    (hinj : ∀ j x x', (∀ j', j' ≠ j → x j' = x' j') → x j ≠ x' j → P x ≠ P x')
    (hf : ∀ i, (D.src i).card ≤ f) : Fintype.card ι ≤ 2 + (f - 1) * D.n := by
  refine card_le_of_out D out (fun x => ![(P x).1, (P x).2]) (fun x v hv => ?_) (fun j x x' hx hne h => ?_) hf
  · funext o
    fin_cases o <;> simp [← hP x v hv]
  · exact hinj j x x' hx hne (Prod.ext (by simpa using congrFun h 0) (by simpa using congrFun h 1))

theorem foldl_ne {σ α : Type} :
    ∀ {N : ℕ} (g : Fin N → σ → α → σ), (∀ i s, Function.Injective (g i s)) →
      (∀ i d, Function.Injective fun s => g i s d) → ∀ (s : σ) (i : Fin N) (d e : Fin N → α),
      (∀ j, j ≠ i → d j = e j) → d i ≠ e i →
      Fin.foldl N (fun t j => g j t (d j)) s ≠ Fin.foldl N (fun t j => g j t (e j)) s := by
  intro N
  induction N with
  | zero => exact fun _ _ _ _ i => i.elim0
  | succ N ih =>
    intro g hw hs s i d e hde hi
    simp only [Fin.foldl_succ_last]
    revert hde hi
    refine Fin.lastCases ?_ (fun i' => ?_) i
    · intro hde hi
      have hpre : (Fin.foldl N (fun t j => g j.castSucc t (d j.castSucc)) s) =
          Fin.foldl N (fun t j => g j.castSucc t (e j.castSucc)) s := by
        congr 1
        funext t j
        rw [hde j.castSucc (Fin.castSucc_ne_last j)]
      rw [hpre]
      exact fun h => hi (hw _ _ h)
    · intro hde hi
      have hlast : d (Fin.last N) = e (Fin.last N) := hde _ (Fin.castSucc_ne_last i').symm
      rw [hlast]
      intro h
      refine ih (fun j => g j.castSucc) (fun j => hw j.castSucc) (fun j => hs j.castSucc) s i'
        (fun j => d j.castSucc) (fun j => e j.castSucc) (fun j hj => hde _ ?_) hi (hs _ _ h)
      exact fun h' => hj (Fin.castSucc_injective _ h')

/-- A run that differs from another in exactly the word `j` is that run with word `j` replaced. -/
theorem eq_update {ι W : Type} [DecidableEq ι] {j : ι} {x x' : ι → W} (hx : ∀ j', j' ≠ j → x j' = x' j') :
    x' = Function.update x j (x' j) := by
  funext j'
  by_cases h : j' = j
  · subst h
    simp
  · simp [h, hx j' h]

instance nontrivial (w : ℕ) : Nontrivial (BitVec (w + 1)) := ⟨⟨0, 1, by simp⟩⟩

end Fold

open Fold

theorem FoldFanIn : Accounting.Guarantees.TS.FoldFanIn := by
  intro w N c f g s D out
  by_cases hrun : ∃ x v, D.Runs x v ∧ (fun o => v (out o)) ≠ foldRun (fun i => (g i).1) s x
  · exact Or.inl hrun
  by_cases hsrc : ∃ i, f < (D.src i).card
  · exact Or.inr (Or.inl hsrc)
  refine Or.inr (Or.inr ?_)
  simp only [not_exists, not_and, not_not, not_lt, ne_eq] at hrun hsrc
  have h := card_le_of_out D out (foldRun (fun i => (g i).1) s) hrun
    (fun j x x' hx hne => foldl_ne (fun i => (g i).1) (fun i => (g i).2.1) (fun i => (g i).2.2) s j x x' hx hne)
    hsrc
  simpa using h

theorem DuoFoldFanIn : Accounting.Guarantees.TS.DuoFoldFanIn := by
  intro n L f k a r s D out
  by_cases hrun : ∃ x v, D.Runs x v ∧ (v (out 0), v (out 1)) ≠ Pouw.Receipt.duoRun k (fun j => (a j).1) r x s
  · exact Or.inl hrun
  by_cases hsrc : ∃ i, f < (D.src i).card
  · exact Or.inr (Or.inl hsrc)
  refine Or.inr (Or.inr ?_)
  simp only [not_exists, not_and, not_not, not_lt, ne_eq] at hrun hsrc
  have h := card_le_of_pair D out (fun x => Pouw.Receipt.duoRun k (fun j => (a j).1) r x s) hrun
    (fun j x x' hx hne he => hne (by
      have hu := Pouw.SecurityProofs.Receipt.DuoRunWordInjective (n + 1) L k a r s x j
        (show Pouw.Receipt.duoRun k (fun j => (a j).1) r (Function.update x j (x j)) s =
          Pouw.Receipt.duoRun k (fun j => (a j).1) r (Function.update x j (x' j)) s by
          rw [Function.update_eq_self, ← eq_update hx]
          exact he)
      exact hu))
    hsrc
  simpa using h

theorem Fx1FoldFanIn : Accounting.Guarantees.TS.Fx1FoldFanIn := by
  intro n L f k a r m s D out
  by_cases hrun : ∃ x v, D.Runs x v ∧
      (v (out 0), v (out 1)) ≠ Pouw.Receipt.fx1Run k (fun j => (a j).1) r (fun j => (m j).1) x s
  · exact Or.inl hrun
  by_cases hsrc : ∃ i, f < (D.src i).card
  · exact Or.inr (Or.inl hsrc)
  refine Or.inr (Or.inr ?_)
  simp only [not_exists, not_and, not_not, not_lt, ne_eq] at hrun hsrc
  have h := card_le_of_pair D out (fun x => Pouw.Receipt.fx1Run k (fun j => (a j).1) r (fun j => (m j).1) x s) hrun
    (fun j x x' hx hne he => hne (by
      have hu := Pouw.SecurityProofs.Receipt.Fx1RunWordInjective (n + 1) L k a r m s x j
        (show Pouw.Receipt.fx1Run k (fun j => (a j).1) r (fun j => (m j).1) (Function.update x j (x j)) s =
          Pouw.Receipt.fx1Run k (fun j => (a j).1) r (fun j => (m j).1) (Function.update x j (x' j)) s by
          rw [Function.update_eq_self, ← eq_update hx]
          exact he)
      exact hu))
    hsrc
  simpa using h

end Accounting.SecurityProofs.TS
