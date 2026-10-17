import Proofs.Flock.VBridge.Algebra.Build

/-!
# What V*'s residuals compute

`residuals_eq_run`: on the S `structureOf` builds and the v `verifierRows` gives at the rep's coins, `GfResiduals`'
residuals at a w are `run` over F, at those coins, on the messages w holds (`decoded`), when the running sums w holds
differ by the consistency sums weighted by `kappa`.

* `build_eval`: the residuals of a built S are the symbolic residuals evaluated (`Lin.eval`), each consistency sum
  read through the running sums.
-/

namespace Flock.RecAlgebra

open Flock

section Eval

variable {sh : Shape} (w comb : Array F128) (cons : Nat → Nat → F128)

theorem kapSum_init (lvs : Array Level) (r : Nat) :
    kapSum (lvs.map fun lv => Array.replicate 2 (Array.replicate (2 ^ lv.lanes) (0 : F128))) r cons = 0 := by
  apply kapSum_zero
  intro a b c ha hc
  rw [Array.size_map] at ha
  simp only [getElem!_def, Array.getElem?_map, Array.getElem?_eq_getElem ha, Option.map_some] at hc ⊢
  rcases Nat.lt_or_ge b 2 with hb | hb
  · simp only [Array.getElem?_replicate, hb, ite_true, Array.size_replicate] at hc ⊢
    simp only [hc, ite_true]
  · simp only [Array.getElem?_replicate, show ¬b < 2 by omega, ite_false] at hc
    exact absurd hc (Nat.not_lt_zero _)

/-- The residuals of a built S at w, the coefficient slots and the comb are the symbolic residuals evaluated at the
message elements they hold, each consistency sum read through the running sums (`hacc`). -/
theorem build_eval {lins : Array Lin} {b : Built} (hb : build sh lins = .ok b)
    (hw : w.size = b.S.nw) (hc : comb.size = 64)
    (hacc : ∀ r < 2, valOf sh w comb cons (.acc1 r) + valOf sh w comb cons (.acc0 r) = kapSum b.kappa r cons) :
    residuals b.S w (b.vVals ++ comb) = .ok (lins.map (Lin.eval (valOf sh w comb cons))) := by
  unfold build at hb
  generalize hK : sh.sched.levels.map (fun lv => Array.replicate 2 (Array.replicate (2 ^ lv.lanes) (0 : F128))) = K
    at hb
  generalize hc0 : (lins.toList.map slotsOf).sum = comb0 at hb
  simp only at hb
  cases hB : (buildAll sh (2 + sh.claims) comb0 lins.toList 0).run { kappa := K } with
  | error e => rw [hB] at hb; cases hb
  | ok p =>
    obtain ⟨res, st⟩ := p
    rw [hB] at hb
    simp only [bind, Except.bind] at hb
    split at hb
    · cases hb
    · rename_i hn
      cases hb
      have hB' : buildAll sh (2 + sh.claims) comb0 lins.toList 0 { kappa := K } = .ok (res, st) := hB
      have hinit : OpsOk sh comb0 { kappa := K } := ⟨rfl, fun _ _ hk => by cases hk⟩
      obtain ⟨-, -, hk, hfin, hval⟩ := buildAll_spec w comb cons lins.toList 0 _ _ _ hinit hB'
      have hn1 : st.vNames.size = comb0 := by omega
      have hn2 : st.vVals.size = comb0 := by omega
      simp only at hw hacc ⊢
      unfold residuals
      split
      · rename_i hbad
        simp only [Array.size_append, Array.size_map, Array.size_range, hw, hn1, hn2, hc] at hbad
        omega
      congr 1
      apply Array.toList_inj.mp
      rw [Array.toList_map, Array.toList_map, hval _ st.vVals ⟨⟨#[], by simp⟩, ⟨#[], by simp⟩⟩ hn2]
      conv_rhs => rw [← List.zipIdx_map_fst 0 lins.toList, List.map_map]
      apply List.map_congr_left
      intro fi hfi
      obtain ⟨f, i⟩ := fi
      obtain ⟨-, hi, hf⟩ := List.mem_zipIdx hfi
      simp only [Function.comp, resForm]
      rw [lin_eval_split]
      by_cases hcf : consOf f = []
      · simp only [hcf, ite_true, tEval, f128_add_zero]
      · simp only [hcf, ite_false]
        have hr := hfin (f, i) hfi hcf
        simp only at hr
        rw [hacc _ (by omega), hk, ← hK, kapSum_init]
        simp only [Nat.zero_le, ite_true]
        rw [show 2 + sh.claims + (i - (2 + sh.claims)) - 0 = i by omega, List.getD_eq_getElem?_getD,
          List.getElem?_eq_getElem (by simpa using hi), Option.getD_some, f128_zero_add]
        simp only [Nat.sub_zero] at hf
        rw [← hf]

/-- The messages w holds (`symbolic`'s elements at `valOf`): its elements, the comb, and the levels' consistency
sums `cons`. -/
def decoded (sh : Shape) (w comb : Array F128) (cons : Nat → Nat → F128) : Messages F128 :=
  (symbolic sh).map (Lin.eval (valOf sh w comb cons))

/-- `GfResiduals` on the S `structureOf` builds and the v `verifierRows` gives at the rep's coins computes `run` over
F at those coins, on the messages w holds, when the running sums w holds differ by the consistency sums weighted by
`kappa` (`hacc`). At a holo shape each comb element also carries w's `h` (`valOf`, `hOf`). -/
theorem residuals_eq_run {rounds : Array (Array F128)} {extras : Array Claim} {v : Array F128} {info : Info Lin}
    {kappa : Array (Array (Array F128))} {S : GfS}
    (hrows : verifierRows sh rounds extras comb = .ok (v, info, kappa)) (hS : structureOf sh = .ok S)
    (hw : w.size = S.nw) (hc : comb.size = 64)
    (hacc : ∀ r < 2, valOf sh w comb cons (.acc1 r) + valOf sh w comb cons (.acc0 r) = kapSum kappa r cons) :
    ∃ rs k, run sh (decoded sh w comb cons) extras 0 (Coins.ofRounds rounds) =
        .ok ((rs, info.map (Lin.eval (valOf sh w comb cons))), k) ∧ residuals S w v = .ok rs := by
  unfold verifierRows at hrows
  cases hr : run sh (symbolic sh) extras 0 (Coins.ofRounds rounds) with
  | error e =>
    have hl : residualLins sh (Coins.ofRounds rounds) extras = .error e := by
      show Prod.fst <$> run sh (symbolic sh) extras 0 (Coins.ofRounds rounds) = _
      rw [hr]; rfl
    rw [hl] at hrows; cases hrows
  | ok q =>
    obtain ⟨⟨lins, info'⟩, k⟩ := q
    have hl : residualLins sh (Coins.ofRounds rounds) extras = .ok (lins, info') := by
      show Prod.fst <$> run sh (symbolic sh) extras 0 (Coins.ofRounds rounds) = _
      rw [hr]; rfl
    rw [hl] at hrows
    simp only [bind, Except.bind] at hrows
    cases hb : build sh lins with
    | error e => rw [hb] at hrows; cases hrows
    | ok b =>
      rw [hb] at hrows
      simp only at hrows
      split at hrows
      · cases hrows
      · rename_i hSb
        cases hrows
        rw [hS] at hSb
        have hbS : b.S = S := by simpa [Except.toOption] using hSb
        subst hbS
        exact ⟨_, k, run_sim (Lin.eval_hom (valOf sh w comb cons)) sh (symbolic sh) extras 0 _ _ _ hr,
          build_eval w comb cons hb hw hc hacc⟩

end Eval

end Flock.RecAlgebra
