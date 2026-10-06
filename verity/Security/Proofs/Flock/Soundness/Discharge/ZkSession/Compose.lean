import Proofs.Flock.Soundness.Discharge.ZkSession.InnerSound

/-!
# A masked rep followed by the inner proof (live/PROTOCOL.md §5, VEIL Lemma 4.3)

Under `--zk` a rep starts by committing its pads (`Ch, Cμ`, §3), runs its masked phases, which end with the rows the
verifier could not check itself (affine forms in the pads, §4), and closes with the inner proof on those rows (§5).
`repZK phase` is that composition for any masked phase `phase : Game (rows × verdict)`; the verdict is what the
verifier still checks in the clear.

`zk_compose`: if for every pad vector `h` with `h_ab = h_fa·h_fb` the masked phase ends with a true verdict and rows
vanishing at `h` with probability at most `ε` (the coupling: VEIL's "`v' − h` is a prover strategy for the unmasked
protocol", which bounds it by the clear protocol's soundness), then the rep accepts with probability at most
`ε + εInner`. The pads are fixed by the commitment before the phase starts, and in the unique-decoding regime the
inner proof's `InnerGood` names one pad vector, the decoding of `Ch`.
-/

namespace FlockSoundness.Discharge.ZkSession

open Game Code
open scoped ENNReal NNReal

variable {F : Type} [Field F] [Fintype F] [DecidableEq F]

/-- **A `--zk` rep**: the pads commitment, the masked phase, and the inner proof on the phase's rows. -/
def repZK {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N) {nA : ℕ} (tr : Triple s)
    (phase : Game ((Fin nA → Lin F s) × Bool)) : Game Bool := do
  let C ← recv ((Fin N → F) × (Fin N → F))
  let o ← phase
  let b ← innerGame dom s nP q pos o.1 tr C.1 C.2
  pure (o.2 && b)

omit [Fintype F] in
/-- In the unique-decoding regime a word is `δ`-close to at most one codeword. -/
theorem agree_unique {N L : ℕ} (dom : Fin N ↪ F) (hL : L ≤ N) (δ : ℝ≥0) (hδ1 : δ ≤ 1)
    (hud : 2 * (δ : ℝ) * N ≤ N - L) {u : Fin N → F} {m m' : Fin L → F}
    (h1 : Agree δ u (enc dom m)) (h2 : Agree δ u (enc dom m')) : m = m' := by
  classical
  obtain ⟨S₁, hS₁, hu₁⟩ := h1
  obtain ⟨S₂, hS₂, hu₂⟩ := h2
  apply enc_injective dom hL
  have hr₁ := real_le_of_card hδ1 hS₁
  have hr₂ := real_le_of_card hδ1 hS₂
  have hU : (S₁ ∪ S₂).card ≤ N := by simpa using Finset.card_le_univ (S₁ ∪ S₂)
  have hsum := Finset.card_union_add_card_inter S₁ S₂
  have hI : L ≤ (S₁ ∩ S₂).card := by
    have : (L : ℝ) ≤ (S₁ ∩ S₂).card := by
      have h' : ((S₁ ∪ S₂).card : ℝ) + (S₁ ∩ S₂).card = S₁.card + S₂.card := by exact_mod_cast hsum
      have hU' : ((S₁ ∪ S₂).card : ℝ) ≤ N := by exact_mod_cast hU
      nlinarith
    exact_mod_cast this
  exact ReedSolomon.eq_of_agree_of_card_le (enc_mem dom m) (enc_mem dom m') hI fun j hj => by
    rw [← hu₁ j (Finset.mem_inter.1 hj).1, hu₂ j (Finset.mem_inter.1 hj).2]

/-- **Composition.** If the masked phase, at every pad vector whose product pad is the product, ends with a true verdict
and vanishing rows with probability at most `ε`, the `--zk` rep accepts with probability at most `ε + εInner`. -/
theorem zk_compose [CharP F 2] {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) [NeZero (s + nP)] (pos : F → Fin N)
    (hpos : ∀ D : Finset (Fin N), prCoin (fun x => pos x ∈ D) ≤ (D.card : ℝ≥0∞) / N)
    {nA : ℕ} (tr : Triple s) (δ : ℝ≥0) (hL : s + nP ≤ N)
    (hδ : δ ≤ relativeUniqueDecodingRadius (ReedSolomon.code dom (s + nP) : Set (Fin N → F)))
    (phase : Game ((Fin nA → Lin F s) × Bool)) (ε : ℝ≥0∞)
    (hcouple : ∀ h : Fin s → F, h tr.ab = h tr.fa * h tr.fb →
      value (fun o => o.2 = true ∧ ∀ k, (o.1 k).eval h = 0) phase ≤ ε) :
    value (· = true) (repZK dom s nP q pos tr phase) ≤ ε + εInner N F q δ := by
  classical
  have : NeZero N := ⟨by have := NeZero.ne (s + nP); omega⟩
  have hud := udr_bound dom hL δ hδ
  have hNpos : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  have hδ1 : δ ≤ 1 := by
    rw [← NNReal.coe_le_coe, NNReal.coe_one]
    nlinarith
  unfold repZK
  simp only [bind_eq, recv, bind_send, bind_ret, value_send]
  refine iSup_le fun C => ?_
  refine (value_bind_le phase _ _ (fun o => ¬ (o.2 = true ∧ InnerGood dom s nP o.1 tr δ C.1))
    (εInner N F q δ) ?_).trans ?_
  · intro o ho
    rw [show ((innerGame dom s nP q pos o.1 tr C.1 C.2).bind fun b => (pure (o.2 && b) : Game Bool)) =
        (innerGame dom s nP q pos o.1 tr C.1 C.2).map (fun b => o.2 && b) from rfl, value_map]
    by_cases hv : o.2 = true
    · have hng : ¬ InnerGood dom s nP o.1 tr δ C.1 := fun hg => ho ⟨hv, hg⟩
      refine le_trans (value_mono (fun b hb => ?_) _)
        (inner_sound dom s nP q pos hpos o.1 tr C.1 C.2 δ hL hδ hng)
      simpa [hv] using hb
    · refine le_trans (value_mono (q := fun _ => False) (fun b hb => ?_) _) ?_
      · simp only [Bool.and_eq_true] at hb
        exact hv hb.1
      · rw [value_false]
        exact zero_le
  · refine add_le_add_left ?_ _
    by_cases hdec : ∃ m : Fin (s + nP) → F, Agree δ C.1 (enc dom m) ∧
        head m tr.ab = head m tr.fa * head m tr.fb
    · obtain ⟨m, hm, hab⟩ := hdec
      refine le_trans (value_mono (fun o ho => ⟨(not_not.1 ho).1, ?_⟩) phase) (hcouple (head m) hab)
      obtain ⟨m', hm', hrows, _⟩ := (not_not.1 ho).2
      rw [agree_unique dom hL δ hδ1 hud hm hm']
      exact hrows
    · refine le_trans (value_mono (q := fun _ => False) (fun o ho => ?_) phase) ?_
      · obtain ⟨m', hm', _, hab'⟩ := (not_not.1 ho).2
        exact hdec ⟨m', hm', hab'⟩
      · rw [value_false]
        exact zero_le

end FlockSoundness.Discharge.ZkSession
