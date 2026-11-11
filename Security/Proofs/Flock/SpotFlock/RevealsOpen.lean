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
  Finset.card_biUnion_le.trans (Finset.sum_le_sum fun u _ => List.toFinset_card_le _)

end SpotFlock.Reveal

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

/-- **The deployed two-stage verifier's reveal opens, under the statement's roots, only gates the unit's proof block
needs**: the per-unit position hypothesis `hpos` of the sample's first-hit bounds, at SPOT[Flock]'s compiled verifier,
with `needs u` the gates `Reveal.needs` lists. It holds under every root. -/
theorem hpos_spot {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)
    (s : AuditStatement) :
    ∀ u v w, ∀ p ∈ (compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.reveal s u v w,
      p.1 = s.roots → p.2.pos ∈ (Reveal.needs (partition h) s u).toFinset :=
  fun u v w p hp _ => List.mem_toFinset.2 (code_reveal_at h [] s u v w p hp).1

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
  rw [if_pos hint] at hr
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
      · rw [if_pos hint] at hr
        exact Or.inl ⟨ρ, rb, rfl, rfl, hint.1, hint.2, hr.symm.trans hroot⟩
      · rw [if_neg hint] at hr
        exact Or.inr (hr.symm.trans hroot)

end SpotFlock
