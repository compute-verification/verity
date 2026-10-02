/-
What the compression gadget's rows force (`Flock.HmNets.compress`): Σ and σ, the schedule, one round, the rounds by
induction, and the feed-forward, each against `Spec`. No `decide` on any of it.
-/
import FlockSoundness.Discharge.Hm.Word

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and round Regs

/-! ## Σ and σ -/

theorem length_rot64 {w : Word} (h : w.length = 64) {n : ℕ} (hn : n ≤ 64) : (rot w n).length = 64 := by
  rw [length_rot w n (by omega), h]

theorem length_shr64 {w : Word} (h : w.length = 64) {n : ℕ} (hn : n ≤ 64) : (shr w n).length = 64 := by
  rw [length_shr w n (by omega), h]

theorem length_xor3_64 {a b c : Word} (ha : a.length = 64) (hb : b.length = 64) (hc : c.length = 64) :
    (xor3 a b c).length = 64 := by
  simp [ha, hb, hc]

theorem length_xor3_rot {a : Word} (h : a.length = 64) {p q r : ℕ} (hp : p ≤ 64) (hq : q ≤ 64) (hr : r ≤ 64) :
    (xor3 (rot a p) (rot a q) (rot a r)).length = 64 :=
  length_xor3_64 (length_rot64 h hp) (length_rot64 h hq) (length_rot64 h hr)

theorem length_xor3_shr {a : Word} (h : a.length = 64) {p q r : ℕ} (hp : p ≤ 64) (hq : q ≤ 64) (hr : r ≤ 64) :
    (xor3 (rot a p) (rot a q) (shr a r)).length = 64 :=
  length_xor3_64 (length_rot64 h hp) (length_rot64 h hq) (length_shr64 h hr)

theorem bv_bsig0W (z : ℕ → Bool) {a : Word} (h : a.length = 64) :
    bv z (xor3 (rot a 28) (rot a 34) (rot a 39)) = bsig0 (bv z a) := by
  rw [bv_xor3 z (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num)),
    bv_rot z h (by norm_num), bv_rot z h (by norm_num), bv_rot z h (by norm_num), bsig0]

theorem bv_bsig1W (z : ℕ → Bool) {a : Word} (h : a.length = 64) :
    bv z (xor3 (rot a 14) (rot a 18) (rot a 41)) = bsig1 (bv z a) := by
  rw [bv_xor3 z (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num)),
    bv_rot z h (by norm_num), bv_rot z h (by norm_num), bv_rot z h (by norm_num), bsig1]

theorem bv_ssig0W (z : ℕ → Bool) {a : Word} (h : a.length = 64) :
    bv z (xor3 (rot a 1) (rot a 8) (shr a 7)) = ssig0 (bv z a) := by
  rw [bv_xor3 z (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num)) (length_shr64 h (by norm_num)),
    bv_rot z h (by norm_num), bv_rot z h (by norm_num), bv_shr z h (by norm_num), ssig0]

theorem bv_ssig1W (z : ℕ → Bool) {a : Word} (h : a.length = 64) :
    bv z (xor3 (rot a 19) (rot a 61) (shr a 6)) = ssig1 (bv z a) := by
  rw [bv_xor3 z (length_rot64 h (by norm_num)) (length_rot64 h (by norm_num)) (length_shr64 h (by norm_num)),
    bv_rot z h (by norm_num), bv_rot z h (by norm_num), bv_shr z h (by norm_num), ssig1]

/-! ## The schedule -/

/-- Every word of the list has 64 bits. -/
def Words64 (w : List Word) : Prop := ∀ x ∈ w, x.length = 64

theorem bv_nil (z : ℕ → Bool) : bv z [] = 0 := rfl

theorem getD_map_bv (z : ℕ → Bool) (w : List Word) (i : ℕ) : (w.map (bv z)).getD i 0 = bv z (w.getD i []) := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map, ← bv_nil z, Option.getD_map]

theorem Words64.getD {w : List Word} (hw : Words64 w) {i : ℕ} (hi : i < w.length) : (w.getD i []).length = 64 := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, Option.getD_some]
  exact hw _ (List.getElem_mem hi)

theorem arr_getElem! (w : Array Word) (i : ℕ) : w[i]! = w.toList.getD i [] := by
  rw [getElem!_def, List.getD_eq_getElem?_getD, Array.getElem?_toList]
  cases w[i]? <;> rfl

theorem sound_schedule : ∀ (n : ℕ) (w : Array Word), Sound (schedule n w) fun z w' =>
    Words64 w.toList → 16 ≤ w.size →
      Words64 w'.toList ∧ w'.size = w.size + n ∧ w'.toList.map (bv z) = scheduleV n (w.toList.map (bv z))
  | 0, w => by
      unfold schedule
      exact Sound.pure' fun z hw _ => ⟨hw, rfl, rfl⟩
  | n + 1, w => by
      unfold schedule
      refine Sound.bind' (sound_addN _ _) fun x => Sound.bind' (sound_commitW x) fun y =>
        Sound.mono (sound_schedule n (w.push y)) fun z w' ih hy hx hw hlen => ?_
      have hl : w.toList.length = w.size := Array.length_toList
      simp only [arr_getElem!] at hx
      obtain ⟨hx1, hx2⟩ := hx (hw.getD (by omega)) (by
        intro v hv
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
        rcases hv with rfl | rfl | rfl
        · exact length_xor3_shr (hw.getD (by omega)) (by norm_num) (by norm_num) (by norm_num)
        · exact hw.getD (by omega)
        · exact length_xor3_shr (hw.getD (by omega)) (by norm_num) (by norm_num) (by norm_num))
      obtain ⟨hy1, hy2⟩ := hy hx1
      have hw' : Words64 (w.push y).toList := by
        intro v hv
        rw [Array.toList_push] at hv
        rcases List.mem_append.1 hv with hv | hv
        · exact hw v hv
        · simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
          subst hv
          exact hy1
      obtain ⟨i1, i2, i3⟩ := ih hw' (by simp; omega)
      refine ⟨i1, by simp at i2; omega, ?_⟩
      rw [i3, scheduleV]
      congr 1
      simp only [Array.toList_push, List.map_append, List.map_cons, List.map_nil, schedStepV, List.length_map, hl,
        hy2, hx2, List.foldl, getD_map_bv, bv_ssig0W z (hw.getD (show w.size - 15 < w.toList.length by omega)),
        bv_ssig1W z (hw.getD (show w.size - 2 < w.toList.length by omega))]

/-! ## One round -/

/-- All eight registers have 64 bits. -/
def Regs.Len64 (r : Regs Word) : Prop :=
  r.a.length = 64 ∧ r.b.length = 64 ∧ r.c.length = 64 ∧ r.d.length = 64 ∧ r.e.length = 64 ∧ r.f.length = 64 ∧
    r.g.length = 64 ∧ r.h.length = 64

/-- The builder's registers (`Flock.HmNets.Regs`, a tuple) as `Regs`. -/
def toR (r : Flock.HmNets.Regs) : Regs Word :=
  ⟨r.1, r.2.1, r.2.2.1, r.2.2.2.1, r.2.2.2.2.1, r.2.2.2.2.2.1, r.2.2.2.2.2.2.1, r.2.2.2.2.2.2.2⟩

theorem sound_round (w : Word) (k : UInt64) (r : Flock.HmNets.Regs) : Sound (Flock.HmNets.round w k r) fun z r' =>
    (toR r).Len64 → w.length = 64 →
      (toR r').Len64 ∧ (toR r').map (bv z) = roundV k.toBitVec (bv z w) ((toR r).map (bv z)) := by
  obtain ⟨a, b, c, d, e, f, g, h⟩ := r
  unfold Flock.HmNets.round
  dsimp only
  refine Sound.bind' (sound_chW _ _ _) fun ch => Sound.bind' (sound_addN _ _) fun t1 =>
    Sound.bind' (sound_majW _ _ _) fun mj => Sound.bind' (sound_addN _ _) fun t2 =>
    Sound.bind' (sound_add2 _ _) fun e₁ => Sound.bind' (sound_commitW e₁) fun e₂ =>
    Sound.bind' (sound_add2 _ _) fun a₁ => Sound.bind' (sound_commitW a₁) fun a₂ =>
    Sound.pure' fun z ha₂ ha₁ he₂ he₁ ht2 hmj ht1 hch hr hwt => ?_
  obtain ⟨la, lb, lc, ld, le, lf, lg, lh⟩ := hr
  dsimp only [toR] at la lb lc ld le lf lg lh
  obtain ⟨ch1, ch2⟩ := hch le lf lg
  obtain ⟨t11, t12⟩ := ht1 lh (by
    intro v hv
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl | rfl | rfl
    · exact length_xor3_rot le (by norm_num) (by norm_num) (by norm_num)
    · exact ch1
    · exact length_constW k
    · exact hwt)
  obtain ⟨mj1, mj2⟩ := hmj la lb lc
  obtain ⟨t21, t22⟩ := ht2 (length_xor3_rot la (by norm_num) (by norm_num) (by norm_num)) (by
    intro v hv
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
    subst hv
    exact mj1)
  obtain ⟨e11, e12⟩ := he₁ ld t11
  obtain ⟨e21, e22⟩ := he₂ e11
  obtain ⟨a11, a12⟩ := ha₁ t11 t21
  obtain ⟨a21, a22⟩ := ha₂ a11
  refine ⟨⟨a21, la, lb, lc, e21, le, lf, lg⟩, ?_⟩
  simp only [toR, Regs.map, roundV, a22, a12, t22, t12, e22, e12, mj2, ch2, List.map_cons, List.map_nil,
    List.foldl_cons, List.foldl_nil, bv_bsig0W z la, bv_bsig1W z le, bv_constW]

/-! ## The rounds -/

theorem sound_rounds (w : Array Word) : ∀ (n t : ℕ) (r : Flock.HmNets.Regs),
    Sound ((List.range' t n).foldlM (fun r t => Flock.HmNets.round w[t]! Flock.Sha512.k[t]! r) r) fun z r' =>
      Words64 w.toList → t + n ≤ w.size → (toR r).Len64 →
        (toR r').Len64 ∧ (toR r').map (bv z) = roundsV (w.toList.map (bv z)) n t ((toR r).map (bv z))
  | 0, t, r => Sound.pure' fun z _ _ hr => ⟨hr, rfl⟩
  | n + 1, t, r => by
      rw [List.range'_succ, List.foldlM_cons]
      exact Sound.bind' (sound_round _ _ r) fun r₁ => Sound.mono (sound_rounds w n (t + 1) r₁)
        fun z r' ih h hw htn hr => by
          have hl : w.toList.length = w.size := Array.length_toList
          rw [arr_getElem!] at h
          obtain ⟨h1, h2⟩ := h hr (hw.getD (by omega))
          obtain ⟨i1, i2⟩ := ih hw (by omega) h1
          refine ⟨i1, ?_⟩
          rw [i2, h2, roundsV, getD_map_bv, kAt]

/-! ## The compression -/

theorem sound_mapM_word {β : Type} (f : β → M Word) (Q : β → Prop) (g : (ℕ → Bool) → β → BitVec 64)
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

theorem Regs.len64_ofList {l : List Word} (hl : l.length = 8) (hw : Words64 l) : (Regs.ofList l).Len64 := by
  match l, hl with
  | [a, b, c, d, e, f, g, h], _ =>
    simp only [Words64, List.mem_cons, forall_eq_or_imp, List.not_mem_nil, false_imp_iff, imp_true_iff, and_true] at hw
    exact hw

theorem Regs.length_toList {α : Type} (r : Regs α) : r.toList.length = 8 := rfl

theorem Regs.map_toList {α β : Type} (f : α → β) (r : Regs α) : (r.map f).toList = r.toList.map f := rfl

theorem Regs.len64_toList {r : Regs Word} (h : r.Len64) : Words64 r.toList := by
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
  rw [List.range_eq_range']
  refine Sound.bind' (sound_schedule 64 _) fun w => Sound.bind' (sound_rounds w 80 0 _) fun r => ?_
  obtain ⟨a, b, c, d, e, f, g, h⟩ := r
  dsimp only
  refine Sound.mono (sound_mapM_word _ (fun x : Word × Word => x.1.length = 64 ∧ x.2.length = 64)
      (fun z x => bv z x.1 + bv z x.2) (fun ⟨x, y⟩ => Sound.mono (sound_add2 x y) fun z r h hq => h hq.1 hq.2) _)
    fun z out hout hr hw hcv hcvl hmsg hmsgl => ?_
  obtain ⟨w1, w2, w3⟩ := hw hmsgl (by simp [hmsg])
  have hr0 : toR (cv[0]!, cv[1]!, cv[2]!, cv[3]!, cv[4]!, cv[5]!, cv[6]!, cv[7]!) = Regs.ofList cv := rfl
  rw [hr0] at hr
  obtain ⟨r1, r2⟩ := hr w1 (by simp at w2; omega) (Regs.len64_ofList hcv hcvl)
  have hrl := Regs.len64_toList r1
  have hrt : (toR (a, b, c, d, e, f, g, h)).toList = [a, b, c, d, e, f, g, h] := rfl
  rw [hrt] at hrl
  obtain ⟨o1, o2⟩ := hout fun x hx => ⟨hcvl _ (List.of_mem_zip hx).1, hrl _ (List.of_mem_zip hx).2⟩
  refine ⟨o1, ?_⟩
  rw [o2, compressV, ← w3, ← Regs.map_ofList _ hcv, ← r2, Regs.map_toList, hrt, List.zipWith_map,
    List.zip_eq_zipWith, List.map_zipWith]

end FlockSoundness.Discharge.Hm
