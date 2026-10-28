import NciCheck.Desc
import Mathlib.Tactic

/-!
# The description-sized NCI check against the unrolled one

`NciCheck.Desc.evaluate` (`verity/core/protocols/network/nci/lean/NciCheck/Desc.lean`) computes per IU its size and
its novel reads from one summary per Definition, where `NciCheck.evaluate` walks the circuit gate by gate. What is
proved here is the summaries' arithmetic, each step stated against the function that computes it:

* `sizes_eq_count`: an IU's size as `Desc.sizes` sums it, root node by root node, is the number of gates the expanded
  division puts in it (each gate in its root node's IU), which is what `NciCheck.tally` counts gate by gate
  (`sizes_eq_tally` in `Proofs.Properties.Nci`).
* `affine_resolves`: the batch fast path's gate for leaf `e` of a batch output (`Piece.affine`) is the gate §7's
  resolution reaches member by member (`Flock.Program.resolveStep`: member `e / L`, its return leaf `e % L`).
* `sum_copies`: copies of a frame of gates, one period apart and each within its period, have copies times one copy's
  width: `novelWidth`'s count of a batch's whole members that no other read meets.

Not proved here: that `analyze`'s reads per root node are the unrolled operands outside it (what `Prog.gate` returns,
through §7's resolution), `Novel.ReadsExact`, proved in `Proofs.Properties.Nci.Agrees` (`Novel.readsExact`). That
`novelWidth` is the width of the union of the gates it is given is proved in `Width` (`novelWidthExact`). The difftest
holds the description-sized check against the unrolled one too (`runs` `NciCheck.DescDifftestMain` in the package's
`lean-audit.json`: 3,993 declarations).
-/

namespace NciCheck.Properties.SecurityProofs.Desc

open NciCheck.Desc

/-! ## Sizes -/

private theorem getElem!_modify' {α : Type} [Inhabited α] (xs : Array α) (j : ℕ) (f : α → α) {i : ℕ}
    (hi : i < xs.size) : (xs.modify j f)[i]! = if j = i then f xs[i]! else xs[i]! := by
  rw [getElem!_pos _ i (by simpa using hi), Array.getElem_modify, getElem!_pos xs i hi]

/-- `sizes`' first `m` root nodes: per IU, the gates of its nodes among them. -/
theorem sizes_inv (offs : Array ℕ) (own : Array (Option ℕ)) (n : ℕ) : ∀ m,
    let s := (List.range m).foldl (fun s k => match own[k]! with
      | some u => s.modify u (· + (offs[k + 1]! - offs[k]!))
      | none => s) (Array.replicate n 0)
    s.size = n ∧ ∀ u < n, s[u]! = ∑ k ∈ (Finset.range m).filter (fun k => own[k]! = some u), (offs[k + 1]! - offs[k]!)
  | 0 => by
    refine ⟨by simp, fun u hu => ?_⟩
    rw [List.range_zero, List.foldl_nil, getElem!_pos _ u (by simpa using hu), Array.getElem_replicate]
    simp
  | m + 1 => by
    obtain ⟨hs, hT⟩ := sizes_inv offs own n m
    intro s
    simp only [s, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    generalize (List.range m).foldl _ _ = A at hs hT
    rcases ho : own[m]! with _ | v
    · refine ⟨hs, fun u hu => ?_⟩
      rw [hT u hu, Finset.range_add_one, Finset.filter_insert]
      simp [ho]
    · refine ⟨by simp [hs], fun u hu => ?_⟩
      rw [getElem!_modify' _ _ _ (by omega), hT u hu, Finset.range_add_one, Finset.filter_insert]
      by_cases hvu : v = u
      · subst hvu
        rw [ite_eq_left rfl, ite_eq_left ho, Finset.sum_insert (by simp)]
        omega
      · rw [ite_eq_right hvu, ite_eq_right (by rw [ho]; exact fun h => hvu (Option.some.inj h))]

theorem sizes_spec (offs : Array ℕ) (own : Array (Option ℕ)) (n : ℕ) :
    (sizes offs own n).size = n ∧ ∀ u < n, (sizes offs own n)[u]! =
      ∑ k ∈ (Finset.range own.size).filter (fun k => own[k]! = some u), (offs[k + 1]! - offs[k]!) :=
  sizes_inv offs own n own.size

/-- The gates of `[0, offs[m])` a predicate holds of, block by block: node `k`'s `[offs[k], offs[k + 1])`. -/
theorem card_filter_blocks (offs : Array ℕ) (K : ℕ) (h0 : offs[0]! = 0) (hmono : ∀ k < K, offs[k]! ≤ offs[k + 1]!)
    (p : ℕ → Prop) [DecidablePred p] : ∀ m ≤ K,
    ((Finset.range offs[m]!).filter p).card = ∑ k ∈ Finset.range m, ((Finset.Ico offs[k]! offs[k + 1]!).filter p).card
  | 0, _ => by simp [h0]
  | m + 1, hm => by
    rw [Finset.sum_range_succ, ← card_filter_blocks offs K h0 hmono p m (by omega), Finset.range_eq_Ico,
      Finset.range_eq_Ico, ← Finset.Ico_union_Ico_eq_Ico (Nat.zero_le _) (hmono m (by omega)), Finset.filter_union,
      Finset.card_union_of_disjoint (Finset.disjoint_filter_filter (Finset.Ico_disjoint_Ico_consecutive _ _ _))]

/-- **An IU's size is its number of gates.** On a division that is `own`'s expansion, root node `k` holding gates
`[offs[k], offs[k + 1])` (`offs` from `0` to the circuit's `N`, as the canonical layout lays the root's nodes out) and
each of them in IU `own[k]`, `Desc.sizes` gives per IU `u` the number of gates the division puts in `u`. -/
theorem sizes_eq_count (offs : Array ℕ) (own : Array (Option ℕ)) (n N : ℕ) (owner : ℕ → Option ℕ)
    (h0 : offs[0]! = 0) (hN : offs[own.size]! = N) (hmono : ∀ k < own.size, offs[k]! ≤ offs[k + 1]!)
    (hexp : ∀ k < own.size, ∀ g, offs[k]! ≤ g → g < offs[k + 1]! → owner g = own[k]!) (u : ℕ) (hu : u < n) :
    (sizes offs own n)[u]! = ((Finset.range N).filter fun g => owner g = some u).card := by
  rw [(sizes_spec offs own n).2 u hu, ← hN, card_filter_blocks offs own.size h0 hmono _ own.size le_rfl,
    Finset.sum_filter]
  refine Finset.sum_congr rfl fun k hk => ?_
  have hk : k < own.size := Finset.mem_range.1 hk
  have hblock : ∀ g ∈ Finset.Ico offs[k]! offs[k + 1]!, owner g = own[k]! := fun g hg =>
    hexp k hk g (Finset.mem_Ico.1 hg).1 (Finset.mem_Ico.1 hg).2
  split_ifs with h
  · rw [Finset.filter_true_of_mem fun g hg => (hblock g hg).trans h, Nat.card_Ico]
  · rw [Finset.filter_false_of_mem fun g hg => by rw [hblock g hg]; exact h, Finset.card_empty]

/-! ## The batch fast path -/

theorem affine_at (q : Piece) (w : ℕ) (A F : ℤ) (t : ℕ) : (q.affine w A F).at t = A + q.at t * F := by
  simp only [Piece.affine, Piece.at]
  ring

/-- **The batch fast path is §7's resolution, member by member.** Leaf `e` of the output of a batch node at gate `o`,
whose members have `G` gates each and return their `L` leaves as the gates `b + t * s` of their activation, is
(`Flock.Program.resolveStep`) leaf `e % L` of member `e / L`, whose activation starts at `o + (e / L) * G`: gate
`o + (e / L) * G + (b + (e % L) * s)`. Where `down` takes the fast path (`L = 1`, or `G = L * s`: the members' runs
make one), that is the gate its piece (`Piece.affine`, `A = o + b`) gives the element `t` that names leaf `e`. -/
theorem affine_resolves (q : Piece) (w : ℕ) (o b s : ℤ) (G L : ℕ) (hL : 0 < L) (h : L = 1 ∨ (G : ℤ) = L * s)
    (t e : ℕ) (he : q.at t = e) :
    (q.affine w (o + b) (if L == 1 then (G : ℤ) else s)).at t = o + ((e / L : ℕ) : ℤ) * G + (b + ((e % L : ℕ) : ℤ) * s) := by
  rw [affine_at, he]
  have hd : ((e / L : ℕ) : ℤ) * L + ((e % L : ℕ) : ℤ) = e := by exact_mod_cast Nat.div_add_mod' e L
  rcases h with rfl | hG
  · simp
    ring
  · have hL1 : (L == 1) = false ∨ L = 1 := by
      by_cases h1 : L = 1
      · exact Or.inr h1
      · exact Or.inl (by simpa using h1)
    rcases hL1 with h1 | rfl
    · simp only [h1, Bool.false_eq_true, ite_false]
      rw [hG, ← hd]
      ring
    · simp only [Nat.div_one, Nat.mod_one, beq_self_eq_true, ite_true, Nat.cast_zero, zero_mul, add_zero]
      ring

/-! ## Copies of one frame -/

/-- **Copies count as copies times one copy.** `cnt` copies of a frame of gates `S` (each below `span ≤ period`), copy
`m` at `lo + m * period`, whose widths repeat from copy to copy (a batch's members, at the same offsets), have total
width `cnt` times one copy's: `novelWidth`'s count of a frame that no other read meets, `x.cnt * distinctWidth x.res`. -/
theorem sum_copies (S : Finset ℕ) (lo period span cnt : ℕ) (hS : ∀ x ∈ S, x < span) (hp : span ≤ period)
    (w : ℕ → ℕ) (hw : ∀ m < cnt, ∀ x ∈ S, w (lo + m * period + x) = w (lo + x)) :
    ∑ y ∈ (Finset.range cnt).biUnion (fun m => S.image (lo + m * period + ·)), w y = cnt * ∑ x ∈ S, w (lo + x) := by
  have hdisj : (↑(Finset.range cnt) : Set ℕ).PairwiseDisjoint fun m => S.image (lo + m * period + ·) := by
    intro m _ m' _ hne
    simp only [Function.onFun]
    rw [Finset.disjoint_left]
    intro y hy hy'
    obtain ⟨x, hx, rfl⟩ := Finset.mem_image.1 hy
    obtain ⟨x', hx', hxx⟩ := Finset.mem_image.1 hy'
    have h1 := hS x hx
    have h2 := hS x' hx'
    rcases Nat.lt_or_gt_of_ne hne with hlt | hlt
    · have : (m + 1) * period ≤ m' * period := Nat.mul_le_mul_right _ hlt
      nlinarith
    · have : (m' + 1) * period ≤ m * period := Nat.mul_le_mul_right _ hlt
      nlinarith
  rw [Finset.sum_biUnion hdisj, show cnt * ∑ x ∈ S, w (lo + x) = ∑ _m ∈ Finset.range cnt, ∑ x ∈ S, w (lo + x) by simp]
  refine Finset.sum_congr rfl fun m hm => ?_
  rw [Finset.sum_image fun x _ y _ h => by simpa using h]
  exact Finset.sum_congr rfl fun x hx => hw m (Finset.mem_range.1 hm) x hx

end NciCheck.Properties.SecurityProofs.Desc
