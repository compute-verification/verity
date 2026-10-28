import Proofs.Flock.Soundness.TmplSplitVal

/-!
# What `walk` reads, against `refVal`

- `refX_eq`, `refS_eq`: `Tab.ref`'s weight for a reference, times nothing more, is the slot weight `refVal` reads at
  `readAt`; `rowLow_e`, `rowLow_e_sum`: a row's low factor times its chunk's high factor is its `E` weight.
- `refVal_stable`: a copy's `refVal` doesn't depend on the depth bound past its Definition, so a table walked once serves
  every placement.
- `forIn_list_sumS`, `forIn_range_sumS`, `forIn_array_sumS`, `sv_bind_ok`: `walk`'s loops and binds in `StateT`, each step
  adding a known term while an invariant of the value and the state holds.
-/

open Flock Flock.Tmpl FlockLevel3 GF128 Std

namespace FlockSoundness.TmplFold

/-! ## A reference's weight and a row's factor -/

theorem refX_eq (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (hR : pt.rT = Tmpl.halves pt.r) {d s β al off H : ℕ}
    {as : Array ℕ} {k : ℕ} (h6 : 6 ≤ β + al) (hL : β + al - 6 ≤ pt.r.size) (hk : as.getD k 0 < 2 ^ β)
    (ip : ℕ → List ℕ) (v : ℕ) :
    refX (t.defs.getD d default) pt s (β + al - 6) H (2 ^ al) (2 ^ β) as (eqHalf pt.r 0 (β + al - 6))
        (fun y => off + Flock.Tmpl.posIn t lays (t.defs.size + 1) d y) (fun t' k => Gs pt s ((ip k).getD t' pt.qc)) k v =
      Gs pt s (readAt t lays pt d (as.getD k 0 + 2 ^ β * (H * 2 ^ al + off)) (2 ^ β) (ip k) v) := by
  unfold refX readAt
  by_cases h1 : v < (t.defs.getD d default).cols
  · simp only [h1, ↓reduceIte]
    rw [show as.getD k 0 + 2 ^ β * (H * 2 ^ al + off) + 2 ^ β * Flock.Tmpl.posIn t lays (t.defs.size + 1) d v =
        as.getD k 0 + 2 ^ β * (H * 2 ^ al + (off + Flock.Tmpl.posIn t lays (t.defs.size + 1) d v)) by ring,
      g_chunk pt hR h6 hL hk]
    split_ifs <;> simp
  · by_cases h2 : v < (t.defs.getD d default).cols + (t.defs.getD d default).ins
    · simp only [h1, h2, ↓reduceIte]
    · simp only [h1, h2, ↓reduceIte]
      rfl

theorem refS_eq (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (hR : pt.rT = Tmpl.halves pt.r) {d s β al off H a : ℕ}
    (h6 : 6 ≤ β) (hL : β + al - 6 ≤ pt.r.size) (ha : a < 2 ^ β) (ip : List ℕ) (v : ℕ) :
    refS (t.defs.getD d default) pt (β + al - 6) H (2 ^ al) (eqHalf pt.r (β - 6) al)
        (fun y => off + Flock.Tmpl.posIn t lays (t.defs.size + 1) d y) (gLow pt s β a) (Gs pt s pt.qc)
        (fun t' => Gs pt s (ip.getD t' pt.qc)) v =
      Gs pt s (readAt t lays pt d (a + 2 ^ β * (H * 2 ^ al + off)) (2 ^ β) ip v) := by
  unfold refS readAt
  by_cases h1 : v < (t.defs.getD d default).cols
  · simp only [h1, ↓reduceIte]
    rw [show a + 2 ^ β * (H * 2 ^ al + off) + 2 ^ β * Flock.Tmpl.posIn t lays (t.defs.size + 1) d v =
        a + 2 ^ β * (H * 2 ^ al + (off + Flock.Tmpl.posIn t lays (t.defs.size + 1) d v)) by ring,
      g_chunk_sum pt hR h6 hL ha]
  · by_cases h2 : v < (t.defs.getD d default).cols + (t.defs.getD d default).ins
    · simp only [h1, h2, ↓reduceIte]
    · simp only [h1, h2, ↓reduceIte]

theorem rowLow_e (pt : FoldPt) (hT : pt.rhoT = Tmpl.halves pt.rho) {β al : ℕ} {as : Array ℕ} {k : ℕ}
    (h6 : 6 ≤ β + al) (hL : β + al - 6 ≤ pt.rho.size) (hk : as.getD k 0 < 2 ^ β) (H P : ℕ) :
    of (Tab.rowLow pt false (2 ^ al) (2 ^ β) as (eqHalf pt.rho 0 (β + al - 6)) k P) *
        hiE pt (β + al - 6) H (P / 2 ^ al) =
      of (pt.e (as.getD k 0 + 2 ^ β * (H * 2 ^ al + P))) := by
  rw [e_chunk pt hT h6 hL hk]
  rfl

theorem rowLow_e_sum (pt : FoldPt) (hT : pt.rhoT = Tmpl.halves pt.rho) {β al a : ℕ} {as : Array ℕ} {k : ℕ}
    (h6 : 6 ≤ β) (hL : β + al - 6 ≤ pt.rho.size) (ha : a < 2 ^ β) (H P : ℕ) :
    of (eLow pt β a) * (of (Tab.rowLow pt true (2 ^ al) (2 ^ β) as (eqHalf pt.rho (β - 6) al) k P) *
        hiE pt (β + al - 6) H (P / 2 ^ al)) =
      of (pt.e (a + 2 ^ β * (H * 2 ^ al + P))) := by
  rw [e_chunk_sum pt hT h6 hL ha, mul_assoc]
  rfl

/-! ## The depth bound -/

/-- **`refVal` past its Definition's depth**: with every callee an earlier Definition, `refVal` at any bound above `d`
is the same. -/
theorem refVal_stable {t : Tmpl} {lays : Array Lay} {pt : FoldPt} {s : ℕ}
    (hcall : ∀ d i, i < (t.defs.getD d default).calls.size → ((t.defs.getD d default).calls.getD i default).def_ < d) :
    ∀ f f' d A B ip, d < f → d < f' → refVal t lays pt s f d A B ip = refVal t lays pt s f' d A B ip
  | 0, _, _, _, _, _, h, _ => absurd h (Nat.not_lt_zero _)
  | _ + 1, 0, _, _, _, _, _, h => absurd h (Nat.not_lt_zero _)
  | f + 1, f' + 1, d, A, B, ip, h, h' => by
    simp only [refVal]
    congr 1
    refine Finset.sum_congr rfl fun i hi => Finset.sum_congr rfl fun j _ => ?_
    have hc := hcall d i (Finset.mem_range.mp hi)
    exact refVal_stable hcall f f' _ _ _ _ (by omega) (by omega)

/-! ## Loops and binds in `StateT` -/

theorem sv_bind_ok {σ α β : Type} {x : StateT σ V α} {f : α → StateT σ V β} {st st' : σ} {b : β}
    (h : (x >>= f).run st = .ok (b, st')) : ∃ a st1, x.run st = .ok (a, st1) ∧ (f a).run st1 = .ok (b, st') := by
  rw [StateT.run_bind] at h
  obtain ⟨⟨a, st1⟩, h1, h2⟩ := except_bind_ok h
  exact ⟨a, st1, h1, h2⟩

theorem forIn_list_sumS {α β σ : Type} (S : β → GF128) (Q : β → σ → Prop) (g : α → GF128) :
    ∀ (l : List α) {f : α → β → StateT σ V (ForInStep β)} {b b' : β} {st st' : σ},
      (forIn l b f).run st = .ok (b', st') → Q b st →
      (∀ a ∈ l, ∀ b st r st1, (f a b).run st = .ok (r, st1) → Q b st →
        ∃ b', r = .yield b' ∧ Q b' st1 ∧ S b' = S b + g a) →
      Q b' st' ∧ S b' = S b + (l.map g).sum
  | [], f, b, b', st, st', h, hq, _ => by
    rw [List.forIn_nil, StateT.run_pure] at h
    cases h
    exact ⟨hq, by rw [List.map_nil, List.sum_nil, add_zero]⟩
  | a :: l, f, b, b', st, st', h, hq, hf => by
    rw [List.forIn_cons] at h
    obtain ⟨r, st1, hr, h⟩ := sv_bind_ok h
    obtain ⟨b1, rfl, hq1, hs1⟩ := hf a (List.mem_cons_self ..) b st r st1 hr hq
    obtain ⟨hq2, hs2⟩ := forIn_list_sumS S Q g l h hq1 (fun a ha => hf a (List.mem_cons_of_mem _ ha))
    exact ⟨hq2, by rw [hs2, hs1, List.map_cons, List.sum_cons, add_assoc]⟩

theorem forIn_array_sumS {α β σ : Type} (S : β → GF128) (Q : β → σ → Prop) (g : α → GF128) {l : Array α}
    {f : α → β → StateT σ V (ForInStep β)} {b b' : β} {st st' : σ} (h : (forIn l b f).run st = .ok (b', st'))
    (hq : Q b st)
    (hf : ∀ a ∈ l, ∀ b st r st1, (f a b).run st = .ok (r, st1) → Q b st →
      ∃ b', r = .yield b' ∧ Q b' st1 ∧ S b' = S b + g a) :
    Q b' st' ∧ S b' = S b + (l.toList.map g).sum := by
  rw [← Array.forIn_toList] at h
  exact forIn_list_sumS S Q g l.toList h hq fun a ha => hf a (Array.mem_toList_iff.mp ha)

theorem forIn_range_sumS {β σ : Type} (S : β → GF128) (Q : β → σ → Prop) (g : ℕ → GF128) {n : ℕ}
    {f : ℕ → β → StateT σ V (ForInStep β)} {b b' : β} {st st' : σ} (h : (forIn [:n] b f).run st = .ok (b', st'))
    (hq : Q b st)
    (hf : ∀ k < n, ∀ b st r st1, (f k b).run st = .ok (r, st1) → Q b st →
      ∃ b', r = .yield b' ∧ Q b' st1 ∧ S b' = S b + g k) :
    Q b' st' ∧ S b' = S b + ∑ k ∈ Finset.range n, g k := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have e : List.range' 0 [:n].size 1 = List.range n := by
    rw [List.range_eq_range']; simp [Std.Legacy.Range.size]
  rw [e] at h
  rw [← list_range_sum]
  exact forIn_list_sumS S Q g _ h hq fun k hk => hf k (List.mem_range.mp hk)

/-! ## The `eq` halves cache -/

/-- Every cached half is the `eq` table of its key. -/
def HalvesOk (pt : FoldPt) (st : SplitSt) : Prop :=
  ∀ key T, st.halves.get? key = some T → T = eqHalf (if key.1 then pt.rho else pt.r) key.2.1 key.2.2

theorem half_ok {pt : FoldPt} {rho : Bool} {lo n : ℕ} {st st' : SplitSt} {T : Half}
    (h : (half pt rho lo n).run st = .ok (T, st')) (hq : HalvesOk pt st) :
    T = eqHalf (if rho then pt.rho else pt.r) lo n ∧ HalvesOk pt st' ∧ st'.tabs = st.tabs := by
  unfold half at h
  rw [StateT.run_bind, StateT.run_get] at h
  simp only [pure_bind] at h
  cases hk : st.halves.get? (rho, lo, n) with
  | some T' =>
    rw [hk] at h
    simp only [StateT.run_pure] at h
    cases h
    exact ⟨hq _ _ hk, hq, rfl⟩
  | none =>
    rw [hk] at h
    simp only [StateT.run_bind, StateT.run_modify, StateT.run_pure, pure_bind] at h
    cases h
    refine ⟨rfl, fun key T hT => ?_, rfl⟩
    dsimp only at hT
    rw [HashMap.get?_eq_getElem?, HashMap.getElem?_insert] at hT
    by_cases he : (rho, lo, n) = key
    · subst he
      simp at hT
      exact hT.symm
    · rw [if_neg (by simpa using he)] at hT
      exact hq key T (by rwa [HashMap.get?_eq_getElem?])

end FlockSoundness.TmplFold
