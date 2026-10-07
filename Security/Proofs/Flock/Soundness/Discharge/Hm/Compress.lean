/-
What the compression gadget's rows force (`Flock.HmNets.compress`): Σ and σ, the schedule, one round, the rounds by induction,
and the feed-forward, each against `Spec`. No `decide` on any of it.
-/
import Proofs.Flock.Soundness.Discharge.Hm.Word

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

/-! ## Σ and σ -/

theorem length_rot64 {w : Word} (h : w.length = 64) {n : ℕ} (hn : n ≤ 64) : (rot w n).length = 64 := by
  rw [length_rot w n (by omega), h]

theorem length_shr64 {w : Word} (h : w.length = 64) {n : ℕ} (hn : n ≤ 64) : (shr w n).length = 64 := by
  rw [length_shr w n (by omega), h]

theorem length_xor3_64 {a b c : Word} (ha : a.length = 64) (hb : b.length = 64) (hc : c.length = 64) :
    (xor3 a b c).length = 64 := by
  simp [ha, hb, hc]

theorem length_bsig0W {a : Word} (h : a.length = 64) : (bsig0W a).length = 64 :=
  length_xor3_64 (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num))

theorem length_bsig1W {a : Word} (h : a.length = 64) : (bsig1W a).length = 64 :=
  length_xor3_64 (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num))

theorem length_ssig0W {a : Word} (h : a.length = 64) : (ssig0W a).length = 64 :=
  length_xor3_64 (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num)) (length_shr64 h (by norm_num))

theorem length_ssig1W {a : Word} (h : a.length = 64) : (ssig1W a).length = 64 :=
  length_xor3_64 (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num)) (length_shr64 h (by norm_num))

theorem bv_bsig0W (z : Var → Bool) {a : Word} (h : a.length = 64) : bv z (bsig0W a) = bsig0 (bv z a) := by
  rw [bsig0W, bv_xor3 z (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num)),
    bv_rot z h (by norm_num), bv_rot z h (by norm_num), bv_rot z h (by norm_num), bsig0]

theorem bv_bsig1W (z : Var → Bool) {a : Word} (h : a.length = 64) : bv z (bsig1W a) = bsig1 (bv z a) := by
  rw [bsig1W, bv_xor3 z (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num)),
    bv_rot z h (by norm_num), bv_rot z h (by norm_num), bv_rot z h (by norm_num), bsig1]

theorem bv_ssig0W (z : Var → Bool) {a : Word} (h : a.length = 64) : bv z (ssig0W a) = ssig0 (bv z a) := by
  rw [ssig0W, bv_xor3 z (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num)) (length_shr64 h (by norm_num)),
    bv_rot z h (by norm_num), bv_rot z h (by norm_num), bv_shr z h (by norm_num), ssig0]

theorem bv_ssig1W (z : Var → Bool) {a : Word} (h : a.length = 64) : bv z (ssig1W a) = ssig1 (bv z a) := by
  rw [ssig1W, bv_xor3 z (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num)) (length_shr64 h (by norm_num)),
    bv_rot z h (by norm_num), bv_rot z h (by norm_num), bv_shr z h (by norm_num), ssig1]

/-! ## The schedule -/

/-- Every word of the list has 64 bits. -/
def Words64 (w : List Word) : Prop := ∀ x ∈ w, x.length = 64

theorem bv_nil (z : Var → Bool) : bv z [] = 0 := rfl

theorem getD_map_bv (z : Var → Bool) (w : List Word) (i : ℕ) : (w.map (bv z)).getD i 0 = bv z (w.getD i []) := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map, ← bv_nil z, Option.getD_map]

theorem Words64.getD {w : List Word} (hw : Words64 w) {i : ℕ} (hi : i < w.length) : (w.getD i []).length = 64 := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, Option.getD_some]
  exact hw _ (List.getElem_mem hi)

theorem sound_schedStep (w : List Word) : Sound (schedStep w) fun z w' =>
    Words64 w → 16 ≤ w.length → Words64 w' ∧ w'.length = w.length + 1 ∧ w'.map (bv z) = schedStepV (w.map (bv z)) := by
  unfold schedStep
  refine Sound.bind' (sound_addN _ _) fun x => Sound.bind' (sound_commitW x) fun y =>
    Sound.pure' fun z hy hx hw hlen => ?_
  obtain ⟨hx1, hx2⟩ := hx (hw.getD (by omega)) (by
    intro v hv
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl | rfl
    · exact length_ssig0W (hw.getD (by omega))
    · exact hw.getD (by omega)
    · exact length_ssig1W (hw.getD (by omega)))
  obtain ⟨hy1, hy2⟩ := hy hx1
  refine ⟨fun v hv => ?_, by simp, ?_⟩
  · rcases List.mem_append.1 hv with hv | hv
    · exact hw v hv
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
      subst hv
      exact hy1
  · simp only [List.map_append, List.map_cons, List.map_nil, schedStepV, List.length_map, hy2, hx2, List.foldl,
      getD_map_bv, bv_ssig0W z (hw.getD (show w.length - 15 < w.length by omega)),
      bv_ssig1W z (hw.getD (show w.length - 2 < w.length by omega))]

theorem sound_schedule : ∀ (n : ℕ) (w : List Word), Sound (schedule n w) fun z w' =>
    Words64 w → 16 ≤ w.length → Words64 w' ∧ w'.length = w.length + n ∧ w'.map (bv z) = scheduleV n (w.map (bv z))
  | 0, w => Sound.pure' fun z hw _ => ⟨hw, rfl, rfl⟩
  | n + 1, w => Sound.bind' (sound_schedStep w) fun w₁ => Sound.mono (sound_schedule n w₁) fun z w' ih h hw hlen => by
      obtain ⟨h1, h2, h3⟩ := h hw hlen
      obtain ⟨i1, i2, i3⟩ := ih h1 (by omega)
      exact ⟨i1, by omega, by rw [i3, h3]; rfl⟩

/-! ## One round -/

/-- All eight registers have 64 bits. -/
def Len64 (r : Regs Word) : Prop :=
  r.a.length = 64 ∧ r.b.length = 64 ∧ r.c.length = 64 ∧ r.d.length = 64 ∧ r.e.length = 64 ∧ r.f.length = 64 ∧
    r.g.length = 64 ∧ r.h.length = 64

theorem sound_roundW (k : BitVec 64) (wt : Word) (r : Regs Word) : Sound (roundW k wt r) fun z r' =>
    Len64 r → wt.length = 64 → Len64 r' ∧ r'.map (bv z) = roundV k (bv z wt) (r.map (bv z)) := by
  unfold roundW
  refine Sound.bind' (sound_chW _ _ _) fun ch => Sound.bind' (sound_addN _ _) fun t1 =>
    Sound.bind' (sound_majW _ _ _) fun mj => Sound.bind' (sound_add2 _ _) fun t2 =>
    Sound.bind' (sound_add2 _ _) fun e₁ => Sound.bind' (sound_commitW e₁) fun e₂ =>
    Sound.bind' (sound_add2 _ _) fun a₁ => Sound.bind' (sound_commitW a₁) fun a₂ =>
    Sound.pure' fun z ha₂ ha₁ he₂ he₁ ht2 hmj ht1 hch hr hwt => ?_
  obtain ⟨la, lb, lc, ld, le, lf, lg, lh⟩ := hr
  obtain ⟨ch1, ch2⟩ := hch le lf lg
  obtain ⟨t11, t12⟩ := ht1 lh (by
    intro v hv
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl | rfl | rfl
    · exact length_bsig1W le
    · exact ch1
    · exact length_constW k
    · exact hwt)
  obtain ⟨mj1, mj2⟩ := hmj la lb lc
  obtain ⟨t21, t22⟩ := ht2 (length_bsig0W la) mj1
  obtain ⟨e11, e12⟩ := he₁ ld t11
  obtain ⟨e21, e22⟩ := he₂ e11
  obtain ⟨a11, a12⟩ := ha₁ t11 t21
  obtain ⟨a21, a22⟩ := ha₂ a11
  refine ⟨⟨a21, la, lb, lc, e21, le, lf, lg⟩, ?_⟩
  simp only [Regs.map, roundV, a22, a12, t22, t12, e22, e12, mj2, ch2, List.map_cons, List.map_nil, List.foldl_cons,
    List.foldl_nil, bv_bsig0W z la, bv_bsig1W z le, bv_constW]

/-! ## The rounds -/

theorem sound_rounds (w : List Word) : ∀ (n t : ℕ) (r : Regs Word), Sound (rounds w n t r) fun z r' =>
    Words64 w → t + n ≤ w.length → Len64 r → Len64 r' ∧ r'.map (bv z) = roundsV (w.map (bv z)) n t (r.map (bv z))
  | 0, t, r => Sound.pure' fun z _ _ hr => ⟨hr, rfl⟩
  | n + 1, t, r => Sound.bind' (sound_roundW _ _ r) fun r₁ => Sound.mono (sound_rounds w n (t + 1) r₁)
      fun z r' ih h hw htn hr => by
        obtain ⟨h1, h2⟩ := h hr (hw.getD (by omega))
        obtain ⟨i1, i2⟩ := ih hw (by omega) h1
        refine ⟨i1, ?_⟩
        rw [i2, h2, roundsV, getD_map_bv]

/-! ## The compression -/

theorem sound_mapM_word {β : Type} (f : β → M Word) (Q : β → Prop) (g : (Var → Bool) → β → BitVec 64)
    (hf : ∀ x, Sound (f x) fun z r => Q x → r.length = 64 ∧ bv z r = g z x) :
    ∀ l : List β, Sound (l.mapM f) fun z rs => (∀ x ∈ l, Q x) → Words64 rs ∧ rs.map (bv z) = l.map (g z)
  | [] => Sound.pure' fun z _ => ⟨fun _ h => absurd h (by simp), rfl⟩
  | x :: l => by
      rw [List.mapM_cons]
      refine Sound.bind' (hf x) fun r => Sound.bind' (sound_mapM_word f Q g hf l) fun rs =>
        Sound.pure' fun z hl hr hq => ?_
      obtain ⟨hr1, hr2⟩ := hr (hq x List.mem_cons_self)
      obtain ⟨hl1, hl2⟩ := hl fun y hy => hq y (List.mem_cons_of_mem _ hy)
      refine ⟨fun v hv => ?_, by simp [hr2, hl2]⟩
      rcases List.mem_cons.1 hv with rfl | hv
      · exact hr1
      · exact hl1 v hv

theorem Regs.map_ofList {α β : Type} [Inhabited α] [Inhabited β] (f : α → β) {l : List α} (hl : l.length = 8) :
    (Regs.ofList l).map f = Regs.ofList (l.map f) := by
  match l, hl with
  | [a, b, c, d, e, f', g, h], _ => rfl

theorem len64_ofList {l : List Word} (hl : l.length = 8) (hw : Words64 l) : Len64 (Regs.ofList l) := by
  match l, hl with
  | [a, b, c, d, e, f, g, h], _ =>
    simp only [Words64, List.mem_cons, forall_eq_or_imp, List.not_mem_nil, false_imp_iff, imp_true_iff, and_true] at hw
    exact hw

theorem Regs.length_toList {α : Type} (r : Regs α) : r.toList.length = 8 := rfl

theorem Regs.map_toList {α β : Type} (f : α → β) (r : Regs α) : (r.map f).toList = r.toList.map f := rfl

theorem len64_toList {r : Regs Word} (h : Len64 r) : Words64 r.toList := by
  obtain ⟨la, lb, lc, ld, le, lf, lg, lh⟩ := h
  intro v hv
  simp only [Regs.toList, List.mem_cons, List.not_mem_nil, or_false] at hv
  rcases hv with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> assumption

/-- **The compression gadget computes SHA-512's compression**: whenever its rows hold, its 8 output words are
`compressV` of its chaining value and message words. -/
theorem sound_compress (cv msg : List Word) : Sound (compress cv msg) fun z out =>
    cv.length = 8 → Words64 cv → msg.length = 16 → Words64 msg →
      Words64 out ∧ out.map (bv z) = compressV (cv.map (bv z)) (msg.map (bv z)) := by
  unfold compress
  refine Sound.bind' (sound_schedule 64 msg) fun w => Sound.bind' (sound_rounds w 80 0 _) fun r =>
    Sound.mono (sound_mapM_word _ (fun x : Word × Word => x.1.length = 64 ∧ x.2.length = 64)
      (fun z x => bv z x.1 + bv z x.2) (fun x => Sound.mono (sound_add2 x.1 x.2) fun z r h hq => h hq.1 hq.2) _)
    fun z out hout hr hw hcv hcvl hmsg hmsgl => ?_
  obtain ⟨w1, w2, w3⟩ := hw hmsgl (by omega)
  obtain ⟨r1, r2⟩ := hr w1 (by omega) (len64_ofList hcv hcvl)
  have hrl := len64_toList r1
  obtain ⟨o1, o2⟩ := hout fun x hx => ⟨hcvl _ (List.of_mem_zip hx).1, hrl _ (List.of_mem_zip hx).2⟩
  refine ⟨o1, ?_⟩
  rw [o2, compressV, ← w3, ← Regs.map_ofList _ hcv, ← r2, Regs.map_toList, List.zipWith_map, List.zip_eq_zipWith,
    List.map_zipWith]

end FlockSoundness.Discharge.Hm
