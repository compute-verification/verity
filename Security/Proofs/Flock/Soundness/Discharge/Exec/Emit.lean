import Proofs.Flock.Soundness.Refine.Frames

/-!
# The frames a run emits (R11b)

A run of the replay transcript appends frames to its pending bytes, and each squeeze of at least one coin closes a
round: the round retained the pending bytes and the squeeze's header, when the server retains round bytes (every
`hm96Rows` tag pinned today does: `accepts_allRetained`, for public-output statements, `hiddenOutputs = false`).
`Emitted t t' fs` says a run from `t` to `t'` absorbed and squeezed the frames `fs`, in order.

* `chunks fs`: the rounds `fs` closes, each the frames up to and including its squeeze, and the frames after the last.
* `Emitted`: the run consumed one round per chunk, each retaining its chunk's bytes (the first also `t`'s pending bytes),
  and left the rest pending.
* `Emitted.trans`: runs compose by appending their frames (`chunks_append`).
* The transcript's steps (`label_emitted`, `absorbF_emitted`, …, `squeeze_emitted`, `sample_emitted`) and loops
  (`forIn_emitted`).
-/

namespace FlockSoundness.Discharge.Exec

open Flock Refine Transcript

/-- A frame that closes a round: a squeeze of at least one coin. -/
def closes : Frame → Bool
  | .sq _ n => n != 0
  | _ => false

/-- A frame list's rounds: the frames up to and including each squeeze that closes a round, and the frames after the
last one. -/
def chunks : List Frame → List (List Frame) × List Frame
  | [] => ([], [])
  | f :: fs =>
    if closes f then ([f] :: (chunks fs).1, (chunks fs).2)
    else match (chunks fs).1 with
      | [] => ([], f :: (chunks fs).2)
      | c :: cs => ((f :: c) :: cs, (chunks fs).2)

/-- Two runs' rounds, one after the other: the second run's first round also carries what the first left pending. -/
def glue (a b : List (List Frame) × List Frame) : List (List Frame) × List Frame :=
  match b.1 with
  | [] => (a.1, a.2 ++ b.2)
  | c :: cs => (a.1 ++ (a.2 ++ c) :: cs, b.2)

theorem chunks_append : ∀ (fs₁ fs₂ : List Frame), chunks (fs₁ ++ fs₂) = glue (chunks fs₁) (chunks fs₂)
  | [], fs₂ => by
      unfold glue
      rcases h : (chunks fs₂).1 with _ | ⟨c, cs⟩
      · simp [chunks, ← h]
      · simp [chunks, ← h]
  | f :: fs₁, fs₂ => by
      rw [List.cons_append]
      simp only [chunks]
      rw [chunks_append fs₁ fs₂]
      unfold glue
      rcases h₂ : (chunks fs₂).1 with _ | ⟨c, cs⟩ <;>
      by_cases hf : closes f <;> simp only [hf, ite_true, ite_false, Bool.false_eq_true] <;>
      rcases h₁ : (chunks fs₁).1 with _ | ⟨c₁, cs₁⟩ <;> simp

theorem encs_append (a b : List Frame) : encs (a ++ b) = encs a ++ encs b := by simp [encs]

/-- **A run emitted `fs`**: from `t` to `t'`, it consumed one round per chunk of `fs`, each retaining its chunk's bytes
(the first also `t`'s pending bytes), and left the frames after the last chunk pending. -/
structure Emitted (t t' : Transcript) (fs : List Frame) : Prop where
  rounds : t'.rounds = t.rounds
  k : t'.k = t.k + (chunks fs).1.length
  ret : ∀ i < (chunks fs).1.length, Retains (t.rounds[t.k + i]!)
    ((if i = 0 then t.pending.data.toList else []) ++ encs ((chunks fs).1.getD i []))
  pending : t'.pending.data.toList = (if (chunks fs).1 = [] then t.pending.data.toList else []) ++ encs (chunks fs).2

theorem Emitted.refl (t : Transcript) : Emitted t t [] :=
  ⟨rfl, by simp [chunks], fun i hi => by simp [chunks] at hi, by simp [chunks, encs]⟩

theorem Emitted.trans {t t₁ t₂ : Transcript} {fs₁ fs₂ : List Frame} (h₁ : Emitted t t₁ fs₁) (h₂ : Emitted t₁ t₂ fs₂) :
    Emitted t t₂ (fs₁ ++ fs₂) := by
  obtain ⟨r₁, k₁, R₁, p₁⟩ := h₁
  obtain ⟨r₂, k₂, R₂, p₂⟩ := h₂
  have hg := chunks_append fs₁ fs₂
  unfold glue at hg
  rcases hc : (chunks fs₂).1 with _ | ⟨c, cs⟩
  · rw [hc] at hg k₂ R₂ p₂
    simp only at hg
    refine ⟨r₂.trans r₁, by rw [hg, k₂, k₁]; simp, fun i hi => ?_, ?_⟩
    · rw [hg] at hi ⊢
      exact R₁ i hi
    · rw [hg]
      simp only [ite_true] at p₂
      rw [p₂, p₁, encs_append, List.append_assoc]
  · rw [hc] at hg k₂ R₂ p₂
    simp only at hg
    refine ⟨r₂.trans r₁, by rw [hg, k₂, k₁]; simp; omega, fun i hi => ?_, ?_⟩
    · rw [hg] at hi ⊢
      simp only [List.length_append, List.length_cons] at hi
      rcases Nat.lt_or_ge i (chunks fs₁).1.length with hlt | hge
      · rw [List.getD_append _ _ _ _ hlt]
        exact R₁ i hlt
      · rw [List.getD_append_right _ _ _ _ hge]
        rcases Nat.eq_or_lt_of_le hge with heq | hgt
        · subst heq
          have := R₂ 0 (by simp)
          rw [r₁, k₁] at this
          simp only [ite_true, Nat.add_zero, List.getD_cons_zero, p₁] at this
          simp only [Nat.sub_self, List.getD_cons_zero, encs_append]
          by_cases h0 : (chunks fs₁).1 = []
          · simp only [h0, List.length_nil, ite_true] at this ⊢
            simpa [List.append_assoc] using this
          · have hl : (chunks fs₁).1.length ≠ 0 := fun e => h0 (List.eq_nil_of_length_eq_zero e)
            simp only [h0, hl, ite_false] at this ⊢
            simpa [List.append_assoc] using this
        · obtain ⟨j, rfl⟩ : ∃ j, i = (chunks fs₁).1.length + 1 + j := ⟨i - (chunks fs₁).1.length - 1, by omega⟩
          have := R₂ (1 + j) (by simp; omega)
          rw [r₁, k₁] at this
          have hne : (chunks fs₁).1.length + 1 + j ≠ 0 := by omega
          simp only [hne, ite_false, List.nil_append]
          rw [show (chunks fs₁).1.length + 1 + j - (chunks fs₁).1.length = 1 + j by omega,
            show t.k + ((chunks fs₁).1.length + 1 + j) = t.k + (chunks fs₁).1.length + (1 + j) by omega]
          simpa [show 1 + j ≠ 0 by omega, Nat.add_comm 1 j] using this
    · rw [hg]
      simp only [reduceCtorEq, ite_false, List.nil_append] at p₂
      simp [p₂]

/-! ## The transcript's steps -/

theorem emitted_absorb {f : Frame} (hf : closes f = false) {t t₁ : Transcript} (hr : t₁.rounds = t.rounds)
    (hk : t₁.k = t.k) (hp : t₁.pending.data.toList = t.pending.data.toList ++ f.enc) : Emitted t t₁ [f] := by
  have hc : chunks [f] = ([], [f]) := by simp [chunks, hf]
  exact ⟨hr, by rw [hc, hk]; rfl, fun i hi => by rw [hc] at hi; simp at hi, by rw [hc, hp]; simp [encs]⟩

theorem label_emitted {l : String} {t t₁ : Transcript} {u : Unit} (h : label l t = .ok (u, t₁)) :
    Emitted t t₁ [.label l.toUTF8] := by
  unfold label at h
  cases h
  exact emitted_absorb rfl rfl rfl (label_enc _ _)

theorem absorbF_emitted {v : F128} {t t₁ : Transcript} {u : Unit} (h : absorbF v t = .ok (u, t₁)) :
    Emitted t t₁ [.one v] := by
  unfold absorbF at h
  cases h
  exact emitted_absorb rfl rfl rfl (absorbF_enc _ _)

theorem absorbSlice_emitted {vs : Array F128} {t t₁ : Transcript} {u : Unit} (h : absorbSlice vs t = .ok (u, t₁)) :
    Emitted t t₁ [.vec vs.toList] := by
  unfold absorbSlice at h
  cases h
  exact emitted_absorb rfl rfl rfl (absorbSlice_enc _ _)

theorem absorbBytes_emitted {b : ByteArray} {t t₁ : Transcript} {u : Unit} (h : absorbBytes b t = .ok (u, t₁)) :
    Emitted t t₁ [.bytes b] := by
  unfold absorbBytes at h
  cases h
  exact emitted_absorb rfl rfl rfl (absorbBytes_enc _ _)

theorem squeeze_emitted {kind : UInt8} {n : ℕ} (hn : n ≠ 0) {t t₁ : Transcript} {cs : Array F128}
    (h : squeeze kind n t = .ok (cs, t₁)) (hret : AllRetained t.rounds) :
    Emitted t t₁ [.sq kind n] ∧ cs = (t.rounds[t.k]!).coins := by
  have h' : (squeeze kind n >>= pure) t = .ok (cs, t₁) := by rw [bind_pure]; exact h
  obtain ⟨-, -, hR, h₂⟩ := squeeze_bindA hn h' hret
  obtain ⟨rfl, rfl⟩ := pure_ok h₂
  have hc : chunks [.sq kind n] = ([[.sq kind n]], []) := by simp [chunks, closes, hn]
  refine ⟨⟨rfl, by rw [hc]; rfl, fun i hi => ?_, by rw [hc]; simp [encs]⟩, rfl⟩
  rw [hc] at hi ⊢
  have hi0 : i = 0 := by simpa using hi
  subst hi0
  simpa [encs] using hR

theorem sample_emitted {t t₁ : Transcript} {v : F128} (h : sample t = .ok (v, t₁)) (hret : AllRetained t.rounds) :
    Emitted t t₁ [.sq 1 1] ∧ v = (t.rounds[t.k]!).coins.getD 0 0 := by
  have h' : (sample >>= pure) t = .ok (v, t₁) := by rw [bind_pure]; exact h
  obtain ⟨-, -, hR, h₂⟩ := sample_bindA h' hret
  obtain ⟨rfl, rfl⟩ := pure_ok h₂
  have hc : chunks [.sq 1 1] = ([[.sq 1 1]], []) := by simp [chunks, closes]
  refine ⟨⟨rfl, by rw [hc]; rfl, fun i hi => ?_, by rw [hc]; simp [encs]⟩, rfl⟩
  rw [hc] at hi ⊢
  have hi0 : i = 0 := by simpa using hi
  subst hi0
  simpa [encs] using hR

theorem sampleVec_emitted {n : ℕ} (hn : n ≠ 0) {t t₁ : Transcript} {cs : Array F128} (h : sampleVec n t = .ok (cs, t₁))
    (hret : AllRetained t.rounds) : Emitted t t₁ [.sq 2 n] ∧ cs = (t.rounds[t.k]!).coins :=
  squeeze_emitted hn h hret

theorem sampleK_emitted {t t₁ : Transcript} {v : F256} (h : sampleK t = .ok (v, t₁)) (hret : AllRetained t.rounds) :
    Emitted t t₁ [.sq 2 2] ∧ v = ⟨(t.rounds[t.k]!).coins.getD 0 0, (t.rounds[t.k]!).coins.getD 1 0⟩ := by
  have h' : (sampleK >>= pure) t = .ok (v, t₁) := by rw [bind_pure]; exact h
  obtain ⟨-, -, hR, h₂⟩ := sampleK_bindA h' hret
  obtain ⟨rfl, rfl⟩ := pure_ok h₂
  have hc : chunks [.sq 2 2] = ([[.sq 2 2]], []) := by simp [chunks, closes]
  refine ⟨⟨rfl, by rw [hc]; rfl, fun i hi => ?_, by rw [hc]; simp [encs]⟩, rfl⟩
  rw [hc] at hi ⊢
  have hi0 : i = 0 := by simpa using hi
  subst hi0
  simpa [encs] using hR

/-- **A loop emits its iterations' frames**, in order, when each iteration emits its own. -/
theorem forIn_emitted {σ : Type} (body : ℕ → σ → TM (ForInStep σ)) (F : ℕ → List Frame)
    (hbody : ∀ i s t b t', AllRetained t.rounds → body i s t = .ok (b, t') →
      ∃ s', b = ForInStep.yield s' ∧ Emitted t t' (F i)) :
    ∀ (n a : ℕ) (s : σ) (t t' : Transcript) (s' : σ), AllRetained t.rounds →
      (forIn (List.range' a n) s body : TM σ) t = .ok (s', t') → Emitted t t' ((List.range' a n).flatMap F)
  | 0, a, s, t, t', s', _, h => by
      simp only [List.range'_zero, List.forIn_nil] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h
      exact Emitted.refl _
  | n + 1, a, s, t, t', s', hret, h => by
      rw [forIn_range'_cons] at h
      obtain ⟨r, t₁, h₁, h₂⟩ := bind_ok h
      obtain ⟨s₁, rfl, he₁⟩ := hbody a s t r t₁ hret h₁
      have he₂ := forIn_emitted body F hbody n (a + 1) s₁ t₁ t' s' (by rw [he₁.rounds]; exact hret) h₂
      rw [List.range'_succ, List.flatMap_cons]
      exact he₁.trans he₂

end FlockSoundness.Discharge.Exec
