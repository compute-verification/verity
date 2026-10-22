import Proofs.Flock.Soundness.TmplFoldClass

/-!
# A class's fold is its copies' reference value (PRF-133, part 3, stage 2)

`refVal`: the slot-`s` value of one copy of a Definition, its own column `y` at `A + B · posIn y` and its inputs at
`ip`, summed over its rows and, copy by copy, its calls' (each read at the position it reads, `readAt`). `foldDef_exp`:
an explicit class's `expVal` is its copies' `refVal`s.
-/

open Flock Flock.Tmpl FlockLevel3 GF128

namespace FlockSoundness.TmplFold

/-- One entry group's value: references `cc.1` of a row of `E` weight `cc.2 · ex`, each read at `R v`. -/
noncomputable def grpVal (pt : FoldPt) (s : ℕ) (ex : F128) (R : ℕ → ℕ) (cc : Array ℕ × F128) : GF128 :=
  (cc.1.toList.map fun v => of (cc.2 * ex) * Gs pt s (R v)).sum

/-- One row's value, its `A` entries weighted by `α` and its `B` entries by 1. -/
noncomputable def rowVal (pt : FoldPt) (s : ℕ) (ex : F128) (R : ℕ → ℕ) (r : ℕ × Array ℕ × Array ℕ) : GF128 :=
  ([(r.2.1, pt.alpha), (r.2.2, (1 : F128))].map (grpVal pt s ex R)).sum

/-- Where Definition `d`'s copy (own column `y` at `A + B · posIn y`, inputs at `ip`) reads reference `v`. -/
def readAt (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (d A B : ℕ) (ip : List ℕ) (v : ℕ) : ℕ :=
  if v < (t.defs.getD d default).cols then A + B * Flock.Tmpl.posIn t lays (t.defs.size + 1) d v
  else if v < (t.defs.getD d default).cols + (t.defs.getD d default).ins then
    ip.getD (v - (t.defs.getD d default).cols) pt.qc
  else pt.qc

/-- **The reference value** of Definition `d`'s copy at `(A, B)` with inputs at `ip`. -/
noncomputable def refVal (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (s : ℕ) : ℕ → ℕ → ℕ → ℕ → List ℕ → GF128
  | 0, _, _, _, _ => 0
  | fuel + 1, d, A, B, ip =>
    ((t.defs.getD d default).rows.toList.map fun r =>
        rowVal pt s (pt.e (A + B * Flock.Tmpl.posIn t lays (t.defs.size + 1) d r.1)) (readAt t lays pt d A B ip) r).sum +
      ∑ i ∈ Finset.range (t.defs.getD d default).calls.size,
        ∑ j ∈ Finset.range ((t.defs.getD d default).calls.getD i default).count,
          refVal t lays pt s fuel ((t.defs.getD d default).calls.getD i default).def_
            (A + B * (((lays.getD d default).places.getD i default).nb +
              j * ((lays.getD d default).places.getD i default).cs))
            (B * ((lays.getD d default).places.getD i default).ms)
            (((t.defs.getD d default).calls.getD i default).bound (readAt t lays pt d A B ip) j)

theorem pure_ok {α ε : Type} {a b : α} (h : (pure a : Except ε α) = .ok b) : a = b := by
  cases h; rfl

theorem getD_replicate {α : Type} (n : ℕ) (a : α) (i : ℕ) : (Array.replicate n a).getD i a = a := by
  by_cases h : i < n <;> simp [h]

theorem getD_replicate₂ (ins nk t k : ℕ) :
    ((Array.replicate ins (Array.replicate nk (0 : F128))).getD t #[]).getD k 0 = 0 := by
  by_cases h : t < ins <;> by_cases hk : k < nk <;> simp [h, hk]

theorem shape_init (ins nk : ℕ) :
    Shape ins nk ⟨Array.replicate 64 0, 0, 0, Array.replicate ins (Array.replicate nk 0), 0⟩ :=
  ⟨by simp, by simp, fun t ht => by simp [ht]⟩

theorem expVal_init (s ins nk : ℕ) (iv : ℕ → ℕ → GF128) :
    expVal s ins nk iv ⟨Array.replicate 64 0, 0, 0, Array.replicate ins (Array.replicate nk 0), 0⟩ = 0 := by
  unfold expVal
  simp only [getD_replicate, getD_replicate₂, of_zero', zero_mul, Finset.sum_const_zero, add_zero]

theorem list_sum_comm {α : Type} (l : List α) (n : ℕ) (f : ℕ → α → GF128) :
    (l.map fun r => ∑ k ∈ Finset.range n, f k r).sum = ∑ k ∈ Finset.range n, (l.map (f k)).sum := by
  induction l with
  | nil => simp
  | cons a l ih => rw [List.map_cons, List.sum_cons, ih, ← Finset.sum_add_distrib]; simp

/-- The reads of copy `k` of an explicit class `(β, h, as)`, its inputs at `ip k`. -/
def readExp (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (d β h : ℕ) (as : Array ℕ) (ip : ℕ → List ℕ) (k v : ℕ) : ℕ :=
  if v < (t.defs.getD d default).cols then as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d v)
  else if v < (t.defs.getD d default).cols + (t.defs.getD d default).ins then
    (ip k).getD (v - (t.defs.getD d default).cols) pt.qc
  else pt.qc

theorem readAt_exp (t : Tmpl) (lays : Array Lay) (pt : FoldPt) (d β h : ℕ) (as : Array ℕ) (ip : ℕ → List ℕ) (k : ℕ) :
    readAt t lays pt d (as.getD k 0 + 2 ^ β * h) (2 ^ β) (ip k) = readExp t lays pt d β h as ip k := by
  funext v
  unfold readAt readExp
  rw [Nat.mul_add, Nat.add_assoc]

/-- An explicit class's own rows: the rows loop of `foldDef` adds each row's value over the class copies. -/
theorem foldDef_exp_rows {t : Tmpl} {lays : Array Lay} {pt : FoldPt} {s : ℕ} (hs : s < 64) {fuel d β h : ℕ}
    {as : Array ℕ} {o : Out} (hd : foldDef t lays pt (fuel + 1) d β h as = .ok o) (h6 : ¬ 6 ≤ β)
    (ip : ℕ → List ℕ) :
    Shape (t.defs.getD d default).ins as.size o ∧
      expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)) o =
        ((t.defs.getD d default).rows.toList.map fun r => ∑ k ∈ Finset.range as.size,
          rowVal pt s (pt.e (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d r.1)))
            (readExp t lays pt d β h as ip k) r).sum + 0 := by
  rw [foldDef] at hd
  simp only [h6, decide_false, ↓reduceIte] at hd
  obtain ⟨oR, hR, hd⟩ := except_bind_ok hd
  have key := forIn_array_sum (expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)))
    (Shape (t.defs.getD d default).ins as.size) (fun r => ∑ k ∈ Finset.range as.size,
      rowVal pt s (pt.e (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d r.1)))
        (readExp t lays pt d β h as ip k) r) hR (shape_init _ _) ?_
  · obtain ⟨hq, hsum⟩ := key
    rw [expVal_init, zero_add] at hsum
    sorry
  · intro ⟨x, ra, rb⟩ _ b r hr hq
    dsimp only at hr
    obtain ⟨b1, hk, hr⟩ := except_bind_ok hr
    cases hr
    have key := forIn_range_sum (expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)))
      (Shape (t.defs.getD d default).ins as.size) (fun k =>
        rowVal pt s (pt.e (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d x)))
          (readExp t lays pt d β h as ip k) (x, ra, rb)) hk hq ?_
    · obtain ⟨hq1, hs1⟩ := key
      exact ⟨_, rfl, hq1, hs1⟩
    intro k hk' b r hr hq
    obtain ⟨b2, hc, hr⟩ := except_bind_ok hr
    cases hr
    have key := forIn_list_sum (expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)))
      (Shape (t.defs.getD d default).ins as.size)
      (grpVal pt s (pt.e (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d x)))
        (readExp t lays pt d β h as ip k)) _ hc hq ?_
    · obtain ⟨hq2, hs2⟩ := key
      exact ⟨_, rfl, hq2, hs2⟩
    intro cc _ b r hr hq
    obtain ⟨b3, hv, hr⟩ := except_bind_ok hr
    cases hr
    have key := forIn_array_sum (expVal s (t.defs.getD d default).ins as.size (fun t' k => Gs pt s ((ip k).getD t' pt.qc)))
      (Shape (t.defs.getD d default).ins as.size)
      (fun v => of (cc.2 * pt.e (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d x))) *
        Gs pt s (readExp t lays pt d β h as ip k v)) hv hq ?_
    · obtain ⟨hq3, hs3⟩ := key
      exact ⟨_, rfl, hq3, hs3⟩
    intro v _ b r hr hq
    cases hr
    refine ⟨_, rfl, ?_⟩
    refine (ref_exp pt hs hk' _ _ _
      (of (cc.2 * pt.e (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d x))) *
        Gs pt s (as.getD k 0 + 2 ^ β * (h + Flock.Tmpl.posIn t lays (t.defs.size + 1) d v))) hq
      (fun _ => own_exp pt hs _ _ _ hq)).imp id (fun e => e.trans ?_)
    unfold readExp
    split_ifs <;> rfl

end FlockSoundness.TmplFold
