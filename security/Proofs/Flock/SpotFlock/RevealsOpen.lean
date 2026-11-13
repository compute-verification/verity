module

public import Proofs.Flock.SpotFlock.Reveals
public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.SpotFlock.Cap
public import Proofs.Flock.SamplingGeneric.TwoDeployed

/-!
# What SPOT's reveal opens

Each opening a unit's reveal makes is at a gate its proof block needs opened (`pos_mem_needs`), so every position a
run's extractions open lies in `opensSet`, the gates some proof block of the partition needs opened: a set the
statement fixes before the run. It is the `Spos` of `Claims.SamplingGeneric.Compiled.pr_regHit_le_canon_pos`, and
`code_reveal_pos` states it at the deployed code's reveal.

At the deployed code, unit by unit:

* `code_reveal_at`: each opening is at a gate of the unit's `needs`, under that gate's roots (`Reveal.rootsAt`);
* `hpos_spot`: the per-unit form the sample's position bounds take as `hpos`, at SPOT[Flock]'s compiled two-stage
  verifier;
* `code_reveal_roots`, `code_reveal_iroots`: under the statement's roots, an interior gate of the unit's replay block
  only where its interior roots are the statement's; under replay block `ρ'`'s interior roots, an interior gate of the
  unit's own replay block showing those roots, or the statement's roots showing them. The interior roots are the
  round-1 message's, so the prover can make two of them, or one and the statement's, equal.
-/

@[expose] public section

namespace SpotFlock.Reveal

open Claims Claims.Sampling SamplingDeploy

variable (part : AuditStatement → Option Partition) (two : AuditStatement → Option Design.Two)
  (ir : Seen → ℕ → List Msg) (op : AuditStatement → ℕ → Seen → Claims.Flock.Wit → ℕ → List Msg → Opening Bool)

/-- **The positions a run can open** at `s`: every gate some proof block of the partition needs opened. -/
def opensSet (s : AuditStatement) : Finset ℕ :=
  (Finset.range (((part s).map fun P => P.blocks.length).getD 0)).biUnion fun u => (needs part s u).toFinset

theorem pos_mem_needs (s : AuditStatement) (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit) :
    ∀ p ∈ reveal part two ir op s u v wF, p.2.pos ∈ needs part s u := by
  intro p hp
  obtain ⟨g, hg, rfl⟩ := List.mem_map.1 hp
  exact hg

theorem pos_mem_opensSet (s : AuditStatement) (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit) :
    ∀ p ∈ reveal part two ir op s u v wF, p.2.pos ∈ opensSet part s := by
  intro p hp
  have hn := pos_mem_needs part two ir op s u v wF p hp
  refine Finset.mem_biUnion.2 ⟨u, ?_, List.mem_toFinset.2 hn⟩
  rw [Finset.mem_range]
  unfold needs at hn
  cases hP : part s with
  | none => simp [hP] at hn
  | some P =>
    simp only [hP, Option.map_some, Option.getD_some] at hn ⊢
    cases hb : P.blocks[u]? with
    | none => simp [hb] at hn
    | some b => exact (List.getElem?_eq_some_iff.1 hb).1

/-- `opensSet` has at most the blocks' needs, summed. -/
theorem card_opensSet_le (s : AuditStatement) :
    (opensSet part s).card ≤
      ∑ u ∈ Finset.range (((part s).map fun P => P.blocks.length).getD 0), (needs part s u).length :=
  Finset.card_biUnion_le.trans (Finset.sum_le_sum fun _ _ => List.toFinset_card_le _)

/-- A gate's roots are the statement's or some replay block's interior roots. -/
theorem rootsAt_cases (s : AuditStatement) (u : ℕ) (v : Seen) (g : ℕ) :
    rootsAt part two ir s u v g = s.roots ∨ ∃ ρ, rootsAt part two ir s u v g = ir v ρ := by
  unfold rootsAt
  split <;> (try split) <;> (try split) <;> (try split) <;> first | exact Or.inl rfl | exact Or.inr ⟨_, rfl⟩

end SpotFlock.Reveal

namespace SamplingDeploy

theorem card_biUnion_toFinset_le (l : List ℕ) (f : ℕ → Finset ℕ) :
    (l.toFinset.biUnion f).card ≤ (l.map fun a => (f a).card).sum := by
  induction l with
  | nil => simp
  | cons a l ih =>
    rw [List.toFinset_cons, Finset.biUnion_insert, List.map_cons, List.sum_cons]
    exact (Finset.card_union_le _ _).trans (Nat.add_le_add_left ih _)

/-- Where the second stage draws, the checked proof blocks are the always-checked ones and the drawn ones. -/
theorem RegTwo.checked_subset (g : RegTwo) (N : ℕ) (src₁ src₂ : ByteArray) {draw : Lean.Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂ = .ok (draw, n)) :
    (g.checked N src₁ src₂).toFinset ⊆ (Finset.range N).filter (fun v => g.always v = true) ∪
      ((Spot.Draw.proved draw).toList.map fun j => g.blockAt (g.sample src₁) j.toNat).toFinset := by
  intro v hv
  rw [List.mem_toFinset] at hv
  simp only [RegTwo.checked, hsel] at hv
  obtain ⟨hvN, hor⟩ := List.mem_filter.1 hv
  rcases Bool.or_eq_true_iff.1 hor with hal | hany
  · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨Finset.mem_range.2 (List.mem_range.1 hvN), hal⟩)
  · obtain ⟨j, hj, hjv⟩ := Array.any_eq_true'.1 hany
    exact Finset.mem_union_right _
      (List.mem_toFinset.2 (List.mem_map.2 ⟨j, Array.mem_toList_iff.2 hj, beq_iff_eq.1 hjv⟩))

/-- **The gates the checked proof blocks need, counted**: where the second stage draws, at most the always-checked
blocks' needs plus each drawn block's. -/
theorem RegTwo.card_biUnion_checked_le (g : RegTwo) (N : ℕ) (src₁ src₂ : ByteArray) {draw : Lean.Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂ = .ok (draw, n)) (needs : ℕ → Finset ℕ) :
    ((g.checked N src₁ src₂).toFinset.biUnion needs).card ≤
      ∑ v ∈ (Finset.range N).filter (fun v => g.always v = true), (needs v).card +
        ((Spot.Draw.proved draw).toList.map fun j => (needs (g.blockAt (g.sample src₁) j.toNat)).card).sum := by
  refine (Finset.card_le_card
    (Finset.biUnion_subset_biUnion_of_subset_left needs (g.checked_subset N src₁ src₂ hsel))).trans ?_
  rw [Finset.union_biUnion]
  refine (Finset.card_union_le _ _).trans (Nat.add_le_add Finset.card_biUnion_le ?_)
  have := card_biUnion_toFinset_le
    ((Spot.Draw.proved draw).toList.map fun j => g.blockAt (g.sample src₁) j.toNat) needs
  simpa only [List.map_map, Function.comp_def] using this

/-- With no always-checked block, at most the drawn blocks' count times the most gates a block needs. -/
theorem RegTwo.card_biUnion_checked_le_mul (g : RegTwo) (N : ℕ) (src₁ src₂ : ByteArray) {draw : Lean.Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂ = .ok (draw, n)) (needs : ℕ → Finset ℕ)
    (hal : ∀ v, g.always v = false) (M : ℕ) (hM : ∀ u, (needs u).card ≤ M) :
    ((g.checked N src₁ src₂).toFinset.biUnion needs).card ≤ (Spot.Draw.proved draw).size * M := by
  have h0 : (Finset.range N).filter (fun v => g.always v = true) = ∅ :=
    Finset.filter_false_of_mem fun v _ => by simp [hal v]
  have hb := g.card_biUnion_checked_le N src₁ src₂ hsel needs
  rw [h0, Finset.sum_empty, Nat.zero_add] at hb
  refine hb.trans ?_
  have hs := List.sum_le_length_nsmul
    ((Spot.Draw.proved draw).toList.map fun j => (needs (g.blockAt (g.sample src₁) j.toNat)).card) M
    (by
      intro y hy
      obtain ⟨j, _, rfl⟩ := List.mem_map.1 hy
      exact hM _)
  simpa only [List.length_map, Array.length_toList, smul_eq_mul] using hs

end SamplingDeploy

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **The deployed code's reveal opens only `opensSet`**: every opening SPOT[Flock]'s two-stage code reveals for a unit
is at a gate some proof block of the partition needs opened. -/
theorem code_reveal_pos (s : AuditStatement) (u : ℕ) (v : Seen) (w : Claims.Flock.Wit) :
    ∀ p ∈ (code h).reveal s u v w, p.2.pos ∈ Reveal.opensSet (partition h) s :=
  Reveal.pos_mem_opensSet (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) s u v w

/-- **Each opening the deployed code's reveal makes for unit `u` is at a gate `u`'s proof block needs, under that
gate's roots** (`Reveal.rootsAt`): the interior roots `iroots x v ρ` of `u`'s replay block `ρ` at one of its interior
gates, the statement's roots otherwise. The interior roots don't read the run's statement, so any `x` names them. -/
theorem code_reveal_at (x : Msg) (s : AuditStatement) (u : ℕ) (v : Seen) (w : Claims.Flock.Wit) :
    ∀ p ∈ (code h).reveal s u v w, p.2.pos ∈ Reveal.needs (partition h) s u ∧
      p.1 = Reveal.rootsAt (partition h) (two h) (iroots x) s u v p.2.pos := by
  intro p hp
  obtain ⟨g, hg, rfl⟩ := (Reveal.mem_reveal (partition h) (two h) (iroots []) (Reveal.frameAt h.frame)).1 hp
  exact ⟨hg, rfl⟩

/-- **The deployed two-stage verifier's reveal opens only gates the unit's proof block needs**, under every root, at
SPOT[Flock]'s compiled verifier, with `needs u` the gates `Reveal.needs` lists. -/
theorem reveal_needs_spot {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)
    (s : AuditStatement) :
    ∀ u v w, ∀ p ∈ (compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.reveal s u v w,
      p.2.pos ∈ (Reveal.needs (partition h) s u).toFinset :=
  fun u v w p hp => List.mem_toFinset.2 (code_reveal_at h [] s u v w p hp).1

/-- **`reveal_needs_spot` under the statement's roots**: the per-unit position hypothesis `hpos` of the sample's
first-hit bounds, at SPOT[Flock]'s compiled verifier. -/
theorem hpos_spot {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)
    (s : AuditStatement) :
    ∀ u v w, ∀ p ∈ (compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.reveal s u v w,
      p.1 = s.roots → p.2.pos ∈ (Reveal.needs (partition h) s u).toFinset :=
  fun u v w p hp _ => reveal_needs_spot h F s u v w p hp

/-- **The deployed code's checked proof blocks are its registration's**, on the bytes of the two stages' coins. -/
theorem code_checked_sub (x : Msg) (r : Fin ((code h).V.k x) → FlockDeploy.Coin) :
    ∀ u ∈ (code h).checked x r, u ∈ h.reg.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
      (src (r (order h x).first)) (stageSrc (stageCoins h x r)) := by
  intro u hu
  change u ∈ ((reg h x).map fun g => g.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
    (src (r (order h x).first)) (stageSrc (stageCoins h x r))).getD [] at hu
  cases hd : decode x with
  | none => simp [reg, hd] at hu
  | some _ => simpa [reg, hd] using hu

/-- **K at the deployed code**: where the second stage draws, the gates the checked proof blocks need number at most
the always-checked blocks' needs plus each drawn block's. -/
theorem code_card_biUnion_le (x : Msg) (r : Fin ((code h).V.k x) → FlockDeploy.Coin) (needs : ℕ → Finset ℕ)
    {draw : Lean.Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks
      (stageSrc (stageCoins h x r)) = .ok (draw, n)) :
    (((code h).checked x r).toFinset.biUnion needs).card ≤
      ∑ v ∈ (Finset.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter
          (fun v => h.reg.always v = true), (needs v).card +
        ((Spot.Draw.proved draw).toList.map fun j =>
          (needs (h.reg.blockAt (h.reg.sample (src (r (order h x).first))) j.toNat)).card).sum :=
  (Finset.card_le_card (Finset.biUnion_subset_biUnion_of_subset_left needs fun u hu =>
    List.mem_toFinset.2 (code_checked_sub h x r u (List.mem_toFinset.1 hu)))).trans
    (h.reg.card_biUnion_checked_le _ _ _ hsel needs)

/-- **K at the deployed code with no always-checked block**: at most the drawn blocks' count times the most gates a
block needs. -/
theorem code_card_biUnion_le_mul (x : Msg) (r : Fin ((code h).V.k x) → FlockDeploy.Coin) (needs : ℕ → Finset ℕ)
    {draw : Lean.Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks
      (stageSrc (stageCoins h x r)) = .ok (draw, n))
    (hal : ∀ v, h.reg.always v = false) (M : ℕ) (hM : ∀ u, (needs u).card ≤ M) :
    (((code h).checked x r).toFinset.biUnion needs).card ≤ (Spot.Draw.proved draw).size * M :=
  (Finset.card_le_card (Finset.biUnion_subset_biUnion_of_subset_left needs fun u hu =>
    List.mem_toFinset.2 (code_checked_sub h x r u (List.mem_toFinset.1 hu)))).trans
    (h.reg.card_biUnion_checked_le_mul _ _ _ hsel needs hal M hM)

/-- **An opening under the statement's roots is at a gate `u`'s proof block needs, and at an interior gate of `u`'s
replay block only where that block's interior roots are the statement's.** -/
theorem code_reveal_roots (x : Msg) {s : AuditStatement} {P : Partition} {d : Design.Two}
    (hP : partition h s = some P) (hd : two h s = some d) {u ρ : ℕ} {rb : List ℕ}
    (hρ : replayOf P d.part u = some ρ) (hrb : d.part.blocks[ρ]? = some rb) (v : Seen) (w : Claims.Flock.Wit) :
    ∀ p ∈ (code h).reveal s u v w, p.1 = s.roots → p.2.pos ∈ Reveal.needs (partition h) s u ∧
      (p.2.pos ∈ rb ∧ s.prog.Boundary rb p.2.pos = false → iroots x v ρ = s.roots) := by
  intro p hp hroot
  obtain ⟨hn, hr⟩ := code_reveal_at h x s u v w p hp
  rw [Reveal.rootsAt_eq (partition h) (two h) (iroots x) hP hd hρ hrb] at hr
  refine ⟨hn, fun hint => ?_⟩
  rw [ite_eq_left hint] at hr
  exact hr.symm.trans hroot

/-- **An opening under replay block `ρ'`'s interior roots is at a gate `u`'s proof block needs, and is either at an
interior gate of `u`'s own replay block, whose interior roots are then `ρ'`'s, or under the statement's roots, which
are then `ρ'`'s interior roots.** -/
theorem code_reveal_iroots (x : Msg) {s : AuditStatement} {P : Partition} {d : Design.Two}
    (hP : partition h s = some P) (hd : two h s = some d) (u ρ' : ℕ) (v : Seen) (w : Claims.Flock.Wit) :
    ∀ p ∈ (code h).reveal s u v w, p.1 = iroots x v ρ' → p.2.pos ∈ Reveal.needs (partition h) s u ∧
      ((∃ ρ rb, replayOf P d.part u = some ρ ∧ d.part.blocks[ρ]? = some rb ∧ p.2.pos ∈ rb ∧
          s.prog.Boundary rb p.2.pos = false ∧ iroots x v ρ = iroots x v ρ') ∨ s.roots = iroots x v ρ') := by
  intro p hp hroot
  obtain ⟨hn, hr⟩ := code_reveal_at h x s u v w p hp
  refine ⟨hn, ?_⟩
  cases hρ : replayOf P d.part u with
  | none =>
    simp only [Reveal.rootsAt, hP, hd, hρ] at hr
    exact Or.inr (hr.symm.trans hroot)
  | some ρ =>
    cases hrb : d.part.blocks[ρ]? with
    | none =>
      simp only [Reveal.rootsAt, hP, hd, hρ, hrb] at hr
      exact Or.inr (hr.symm.trans hroot)
    | some rb =>
      rw [Reveal.rootsAt_eq (partition h) (two h) (iroots x) hP hd hρ hrb] at hr
      by_cases hint : p.2.pos ∈ rb ∧ s.prog.Boundary rb p.2.pos = false
      · rw [ite_eq_left hint] at hr
        exact Or.inl ⟨ρ, rb, rfl, hrb, hint.1, hint.2, hr.symm.trans hroot⟩
      · rw [ite_eq_right hint] at hr
        exact Or.inr (hr.symm.trans hroot)

end SpotFlock
