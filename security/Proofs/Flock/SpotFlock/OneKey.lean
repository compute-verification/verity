module

public import Proofs.Flock.SpotFlock.OpensWhole
public import Proofs.Flock.SpotFlock.ScopeFast
public import Proofs.Core.SamplingGeneric.FirstHitClass

/-!
# Where no proof block reads another's gates, the gates a block needs have one key

`gateKey` (`OpensWhole`) gives a gate the proof blocks that need it opened, the replay block that holds it and
whether it is on that block's boundary. How many keys the gates one proof block needs have depends on the program's
wiring. This file is the case of a program whose proof blocks stand side by side:

* `Closed C b`: each gate of `b` reads only gates of `b` and constants.
* `needs_of_closed`: a closed block needs only gates of its own, those its boundary commits.
* `needs_io_of_closed`: in a partition of closed blocks that hold no constant, a gate a block needs is returned or
  is an input, since no gate outside the block reads it.
* `needs_single`, `needs_single_code`, `needers_eq_singleton`: there a gate a block needs is needed by that block
  alone.
* `gateKey_eq_of_needs`, `clsOf_eq_of_needs`: there, at a statement in the two-stage scope, any two gates one proof
  block needs have one key, so the block's needed gates are one class of the key.

The two hypotheses on the statement's program, that every proof block is closed and holds no constant, are
hypotheses here: nothing in this file derives them for a particular program or partition.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-- **A closed set of gates**: each gate of `b` reads only gates of `b` and constants. -/
def Closed (C : Prog) (b : List ℕ) : Prop :=
  ∀ g ∈ b, ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, i ∈ b ∨ ∃ c, C.gates[i]? = some (.const c)

/-- **A closed block needs only gates of its own**, those its boundary commits. -/
theorem needs_of_closed {C : Prog} {b : List ℕ} (hc : Closed C b) {g : ℕ} (hg : C.Needs b g) :
    g ∈ b ∧ C.Boundary b g = true := by
  unfold Prog.Needs at hg
  rcases hg with hg | ⟨⟨g', hg', x, hx, hi⟩, hgb, hnc⟩
  · exact hg
  · rcases hc g' hg' x hx g hi with hin | ⟨c, hcc⟩
    · exact absurd hin hgb
    · exact absurd hcc (hnc c)

/-- **In a partition of closed proof blocks, a needed gate is needed by one block**: two blocks that need one gate
are one block. -/
theorem needs_single {C : Prog} {Pt : Partition} (hnd : Pt.gates.Nodup) (hc : ∀ b ∈ Pt.blocks, Closed C b)
    {u v : ℕ} {bu bv : List ℕ} (hu : Pt.blocks[u]? = some bu) (hv : Pt.blocks[v]? = some bv) {g : ℕ}
    (hgu : C.Needs bu g) (hgv : C.Needs bv g) : v = u :=
  eq_of_mem_of_nodup hnd hv hu (needs_of_closed (hc bv (List.mem_of_getElem? hv)) hgv).1
    (needs_of_closed (hc bu (List.mem_of_getElem? hu)) hgu).1

/-- **In a partition of closed proof blocks that hold no constant, a gate a block needs is returned or is an
input**: a gate outside the block that read it would be a gate of another block, which reads only its own gates and
constants. -/
theorem needs_io_of_closed {C : Prog} {Pt : Partition} (hpart : Pt.Partitions C)
    (hc : ∀ b ∈ Pt.blocks, Closed C b) (hnc : ∀ b ∈ Pt.blocks, ∀ g ∈ b, ∀ c, C.gates[g]? ≠ some (.const c))
    {u : ℕ} {bu : List ℕ} (hu : Pt.blocks[u]? = some bu) {g : ℕ} (hg : C.Needs bu g) :
    g ∈ C.outs ∨ C.gates[g]? = some .input := by
  have hbu : bu ∈ Pt.blocks := List.mem_of_getElem? hu
  obtain ⟨hgb, hB⟩ := needs_of_closed (hc bu hbu) hg
  unfold Prog.Boundary at hB
  simp only [Bool.or_eq_true, List.contains_iff_mem, decide_eq_true_eq, List.any_eq_true, List.mem_range,
    Bool.and_eq_true] at hB
  rcases hB with (hout | hin) | ⟨g', hg', hg'u, hr⟩
  · exact Or.inl hout
  · exact Or.inr hin
  · exfalso
    obtain ⟨x, hx, hi⟩ := C.mem_reads_getD.1 hr
    have hg'P : g' ∈ Pt.gates := by
      refine hpart.2.2.1 g' hg' (fun c hc' => ?_) (fun hun => ?_)
      · rw [hx] at hc'
        obtain rfl := Option.some.inj hc'
        simp [BGate.reads] at hi
      · rw [hun.1] at hx
        obtain rfl := Option.some.inj hx
        simp [BGate.reads] at hi
    obtain ⟨v, hv, hgv⟩ := exists_block hg'P
    have hbv : Pt.blocks[v]? = some (Pt.blocks.getD v []) := getElem?_getD hv
    have hbvm : Pt.blocks.getD v [] ∈ Pt.blocks := List.mem_of_getElem? hbv
    rcases hc _ hbvm g' hgv x hx g hi with hin' | ⟨c, hcc⟩
    · have hvu : v = u := eq_of_mem_of_nodup hpart.1 hbv hu hin' hgb
      have h1 : Pt.blocks.getD v [] = bu := by
        rw [hvu]
        exact getD_of_getElem? hu
      rw [h1] at hgv
      exact hg'u hgv
    · exact hnc bu hbu g hgb c hcc

variable (h : Held)

/-- **Where the statement's proof blocks are closed, a gate one block needs opened no other block needs opened.** -/
theorem needs_single_code {s : AuditStatement} {Pt : Partition} (hP : partition h s = some Pt)
    (hnd : Pt.gates.Nodup) (hc : ∀ b ∈ Pt.blocks, Closed s.prog b) {u v g : ℕ}
    (hgu : g ∈ Reveal.needs (partition h) s u) (hgv : g ∈ Reveal.needs (partition h) s v) : v = u := by
  have hu : u < Pt.blocks.length := Reveal.lt_of_mem_needs (partition h) hP hgu
  have hv : v < Pt.blocks.length := Reveal.lt_of_mem_needs (partition h) hP hgv
  have hbu : Pt.blocks[u]? = some (Pt.blocks.getD u []) := getElem?_getD hu
  have hbv : Pt.blocks[v]? = some (Pt.blocks.getD v []) := getElem?_getD hv
  exact needs_single hnd hc hbu hbv ((Reveal.mem_needs (partition h) hP hbu).1 hgu)
    ((Reveal.mem_needs (partition h) hP hbv).1 hgv)

/-- **Where the statement's proof blocks are closed, the proof blocks that need a gate of block `u`'s needs opened
are `u` alone.** -/
theorem needers_eq_singleton {s : AuditStatement} {Pt : Partition} (hP : partition h s = some Pt)
    (hnd : Pt.gates.Nodup) (hc : ∀ b ∈ Pt.blocks, Closed s.prog b) {u g : ℕ}
    (hg : g ∈ Reveal.needs (partition h) s u) :
    ((Finset.range Pt.blocks.length).filter fun v => g ∈ Reveal.needs (partition h) s v) = {u} := by
  refine Finset.ext fun v => ?_
  rw [Finset.mem_filter, Finset.mem_range, Finset.mem_singleton]
  refine ⟨fun hv => needs_single_code h hP hnd hc hg hv.2, fun hv => ?_⟩
  rw [hv]
  exact ⟨Reveal.lt_of_mem_needs (partition h) hP hg, hg⟩

/-- **One key a proof block**: at a statement in the two-stage scope whose proof blocks are closed and hold no
constant, any two gates one proof block needs opened have one key: that block alone needs them, they lie in its
replay block, and each, returned or an input, is on that block's boundary. -/
theorem gateKey_eq_of_needs {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hP : partition h s = some Pt)
    (hin : InScopeTwo Pt d s) (hc : ∀ b ∈ Pt.blocks, Closed s.prog b)
    (hnc : ∀ b ∈ Pt.blocks, ∀ g ∈ b, ∀ c, s.prog.gates[g]? ≠ some (.const c)) {u g g' : ℕ}
    (hg : g ∈ Reveal.needs (partition h) s u) (hg' : g' ∈ Reveal.needs (partition h) s u) :
    gateKey h s Pt d g = gateKey h s Pt d g' := by
  have hu : u < Pt.blocks.length := Reveal.lt_of_mem_needs (partition h) hP hg
  have hbu : Pt.blocks[u]? = some Pt.blocks[u] := List.getElem?_eq_getElem hu
  have hbm : Pt.blocks[u] ∈ Pt.blocks := List.getElem_mem hu
  have hN : s.prog.Needs Pt.blocks[u] g := (Reveal.mem_needs (partition h) hP hbu).1 hg
  have hN' : s.prog.Needs Pt.blocks[u] g' := (Reveal.mem_needs (partition h) hP hbu).1 hg'
  obtain ⟨r, hr, hsub⟩ := (replayOf_isSome_iff hu).1 (hin.2.2.2.1 u hu)
  obtain ⟨ρ, hρ⟩ := List.getElem?_of_mem hr
  have hgb : gateBlock d.part g = some ρ :=
    gateBlock_of_mem hin.2.2.1 hρ (hsub g (needs_of_closed (hc _ hbm) hN).1)
  have hgb' : gateBlock d.part g' = some ρ :=
    gateBlock_of_mem hin.2.2.1 hρ (hsub g' (needs_of_closed (hc _ hbm) hN').1)
  have hB : ∀ a : ℕ, s.prog.Needs Pt.blocks[u] a → ∀ rb : List ℕ, s.prog.Boundary rb a = true := by
    intro a ha rb
    simp only [Prog.Boundary, Bool.or_eq_true, List.contains_iff_mem, decide_eq_true_eq]
    exact Or.inl (needs_io_of_closed hin.2.1 hc hnc hbu ha)
  have h1 : (gateKey h s Pt d g).1 = (gateKey h s Pt d g').1 :=
    (needers_eq_singleton h hP hin.2.1.1 hc hg).trans (needers_eq_singleton h hP hin.2.1.1 hc hg').symm
  have h2 : (gateKey h s Pt d g).2.1 = (gateKey h s Pt d g').2.1 := hgb.trans hgb'.symm
  have h3 : (gateKey h s Pt d g).2.2 = (gateKey h s Pt d g').2.2 := by
    show ((gateBlock d.part g).any fun ρ => s.prog.Boundary (d.part.blocks.getD ρ []) g) =
      ((gateBlock d.part g').any fun ρ => s.prog.Boundary (d.part.blocks.getD ρ []) g')
    simp only [hgb, hgb', Option.any_some, hB g hN, hB g' hN']
  exact Prod.ext h1 (Prod.ext h2 h3)

/-- **One class a proof block**: there, under the classes of the key (`Verity.FirstHit.clsOf` at `gateKey`), any two
gates one proof block needs opened are in one class. -/
theorem clsOf_eq_of_needs {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hP : partition h s = some Pt)
    (hin : InScopeTwo Pt d s) (hc : ∀ b ∈ Pt.blocks, Closed s.prog b)
    (hnc : ∀ b ∈ Pt.blocks, ∀ g ∈ b, ∀ c, s.prog.gates[g]? ≠ some (.const c)) {n u : ℕ} (g g' : Fin n)
    (hg : g.val ∈ Reveal.needs (partition h) s u) (hg' : g'.val ∈ Reveal.needs (partition h) s u) :
    Verity.FirstHit.clsOf (fun a : Fin n => gateKey h s Pt d a.val) g =
      Verity.FirstHit.clsOf (fun a : Fin n => gateKey h s Pt d a.val) g' :=
  (Verity.FirstHit.clsOf_eq_iff (fun a : Fin n => gateKey h s Pt d a.val) g g').2
    (gateKey_eq_of_needs h hP hin hc hnc hg hg')

end SpotFlock
