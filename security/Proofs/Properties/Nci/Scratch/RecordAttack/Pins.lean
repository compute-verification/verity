module

public import Proofs.Properties.Nci.Record

/-!
# What two circuits of one record share

A scratch file, never merged.

* **Two circuits one record lays out are one circuit, renumbered** (`served_pins`): a bijection between their gates
  keeps every width, every kind, every read and θ. So a set of gates has one size and one count of fresh bits in
  both (`served_pins_counts`).
* **The blocks are pinned where no gate lies in units of two slots** (`OneSlot`, `served_pins_blocks`): then block
  `u` of one circuit and block `u` of the other have one size and one count of fresh bits. `Served` leaves this one
  thing open: `Divides` asks that a computed gate's block is the slot of a unit that holds it, and no clause says
  that only one unit does.
-/

@[expose] public section

namespace NciCheck.Properties.Record.Attack

open Lean (Json)
open Flock.Program (Prog Scope)
open Flock.TemplateQuery (TRange)

/-- Under a renumbering that keeps widths, reads and θ, a set of gates keeps its fresh bits. -/
theorem novelReads_image {d d' : Verity.RegisteredCircuit} {σ : Fin d.D.N → Fin d'.D.N}
    (hσ : Function.Bijective σ) (hw : ∀ x, d'.D.w (σ x) = d.D.w x)
    (hr : ∀ x a, σ a ∈ d'.D.reads (σ x) ↔ a ∈ d.D.reads x) (hθ : ∀ x, σ x ∈ d'.θ ↔ x ∈ d.θ)
    (R : Finset (Fin d.D.N)) : Nci.novelReads d' (R.image σ) = Nci.novelReads d R := by
  have hset : d'.D.into (R.image σ) \ d'.θ = (d.D.into R \ d.θ).image σ := by
    ext y
    obtain ⟨x, rfl⟩ := hσ.2 y
    simp only [Verity.Circuit.into, Finset.mem_sdiff, Finset.mem_filter, Finset.mem_univ, true_and,
      Finset.mem_image, hθ]
    constructor
    · rintro ⟨⟨hnot, g, ⟨g₀, hg₀, rfl⟩, hread⟩, hnθ⟩
      exact ⟨x, ⟨⟨fun hx => hnot ⟨x, hx, rfl⟩, g₀, hg₀, (hr g₀ x).mp hread⟩, hnθ⟩, rfl⟩
    · rintro ⟨x', ⟨⟨hnot, g₀, hg₀, hread⟩, hnθ⟩, hx'⟩
      have hxx : x' = x := hσ.1 hx'
      rw [hxx] at hnot hread hnθ
      refine ⟨⟨?_, σ g₀, ⟨g₀, hg₀, rfl⟩, (hr g₀ x).mpr hread⟩, hnθ⟩
      rintro ⟨z, hz, hzx⟩
      have hzz : z = x := hσ.1 hzx
      rw [hzz] at hz
      exact hnot hz
  show d'.D.width (d'.D.into (R.image σ) \ d'.θ) = d.D.width (d.D.into R \ d.θ)
  rw [hset]
  unfold Verity.Circuit.width
  rw [Finset.sum_image hσ.1.injOn]
  exact Finset.sum_congr rfl fun x _ => hw x

/-- No gate lies in units of two slots: two units of the Program that hold one gate have one slot. -/
def OneSlot (F : Form) (c : F.Case) : Prop :=
  ∀ (P : Prog) (rs : Array TRange), Flock.Program.decode (F.inst c).program.toUTF8 = .ok P →
    Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P)
      ((F.wiring c).templates.map (·.1)).toArray = .ok rs →
    ∀ (g : ℕ) (tid : String) (u : ℕ) (sc : Scope) (tid' : String) (u' : ℕ) (sc' : Scope),
      UnitAt P rs tid u sc → sc.off ≤ g → g < sc.off + (P.defs[sc.d]!).gates →
      UnitAt P rs tid' u' sc' → sc'.off ≤ g → g < sc'.off + (P.defs[sc'.d]!).gates →
      F.slot c tid u = F.slot c tid' u'

/-- **Two circuits one record lays out are one circuit, renumbered.** A bijection between their gates keeps widths,
kinds, reads and θ; and, where no gate lies in units of two slots, each gate's block. -/
theorem served_pins {F : Form} {c : F.Case} {d d' : Verity.RegisteredCircuit} (h : Served F c d)
    (h' : Served F c d') :
    ∃ σ : Fin d.D.N → Fin d'.D.N, Function.Bijective σ ∧ (∀ x, d'.D.w (σ x) = d.D.w x) ∧
      (∀ x, d'.D.isInput (σ x) = d.D.isInput x) ∧ (∀ x a, σ a ∈ d'.D.reads (σ x) ↔ a ∈ d.D.reads x) ∧
      (∀ x, σ x ∈ d'.θ ↔ x ∈ d.θ) ∧
      (OneSlot F c → ∀ x, (d'.𝓘.unit (σ x)).map Fin.val = (d.𝓘.unit x).map Fin.val) := by
  obtain ⟨P, doc, W, S, rs, at_, h1, -, -, -, -, -, -, h8, h9, -, -, -, -, -, hlay, hnames, hdiv⟩ := h
  obtain ⟨P', doc', W', S', rs', at', h1', -, -, -, -, -, -, h8', h9', -, -, -, -, -, hlay', hnames', hdiv'⟩ := h'
  have hP : P' = P := Except.ok.inj (h1'.symm.trans h1)
  subst hP
  have hS : S' = S := Except.ok.inj (h8'.symm.trans h8)
  subst hS
  have hrs : rs' = rs := Except.ok.inj (h9'.symm.trans h9)
  subst hrs
  obtain ⟨hinj, hrange, hgate⟩ := hlay
  obtain ⟨hinj', hrange', hgate'⟩ := hlay'
  have hex : ∀ x, ∃ x', at' x' = at_ x := fun x => (hrange' (at_ x)).mpr ((hrange (at_ x)).mp ⟨x, rfl⟩)
  choose σ hσ using hex
  have hσinj : Function.Injective σ := fun x y hxy =>
    hinj ((hσ x).symm.trans ((congrArg at' hxy).trans (hσ y)))
  have hσsurj : Function.Surjective σ := fun y => by
    obtain ⟨x, hx⟩ := (hrange (at' y)).mpr ((hrange' (at' y)).mp ⟨y, rfl⟩)
    exact ⟨x, hinj' ((hσ x).trans hx)⟩
  have hkind : ∀ x, d'.D.isInput (σ x) = d.D.isInput x := by
    intro x
    obtain ⟨p, ops, sc, hg, -, hi, -⟩ := hgate x
    obtain ⟨p', ops', sc', hg', -, hi', -⟩ := hgate' (σ x)
    rw [hσ x, hg] at hg'
    have hp : p = p' := congrArg Prod.fst (Except.ok.inj hg')
    subst hp
    exact hi'.trans hi.symm
  refine ⟨σ, ⟨hσinj, hσsurj⟩, ?_, hkind, ?_, ?_, ?_⟩
  · intro x
    obtain ⟨p, ops, sc, hg, hw, -, -⟩ := hgate x
    obtain ⟨p', ops', sc', hg', hw', -, -⟩ := hgate' (σ x)
    rw [hσ x, hg] at hg'
    have hp : p = p' := congrArg Prod.fst (Except.ok.inj hg')
    subst hp
    have hval := hw'.symm.trans hw
    injection hval
  · intro x a
    obtain ⟨p, ops, sc, hg, -, -, hr⟩ := hgate x
    obtain ⟨p', ops', sc', hg', -, -, hr'⟩ := hgate' (σ x)
    rw [hσ x, hg] at hg'
    have hops : ops = ops' := congrArg (fun t => t.2.1) (Except.ok.inj hg')
    subst hops
    have hread := hr' (σ a)
    rw [hσ a] at hread
    exact hread.trans (hr a).symm
  · intro x
    have hmem := hnames' (σ x)
    rw [hσ x] at hmem
    exact hmem.trans (hnames x).symm
  · intro hone x
    cases hin : d.D.isInput x with
    | true =>
      have hin' : d'.D.isInput (σ x) = true := (hkind x).trans hin
      simp [(d.𝓘.input_iff x).mpr hin, (d'.𝓘.input_iff (σ x)).mpr hin']
    | false =>
      obtain ⟨tid, u, sc, s, hu, hlo, hhi, hs, hmap⟩ := hdiv x hin
      obtain ⟨tid', u', sc', s', hu', hlo', hhi', hs', hmap'⟩ := hdiv' (σ x) ((hkind x).trans hin)
      rw [hσ x] at hlo' hhi'
      have hss := hone _ _ h1 h9 (at_ x) tid u sc tid' u' sc' hu hlo hhi hu' hlo' hhi'
      rw [hs, hs'] at hss
      exact hmap'.trans (hss.symm.trans hmap.symm)

/-- **A set of gates has one size and one count of fresh bits in both.** -/
theorem served_pins_counts {F : Form} {c : F.Case} {d d' : Verity.RegisteredCircuit} (h : Served F c d)
    (h' : Served F c d') :
    ∃ σ : Fin d.D.N → Fin d'.D.N, Function.Bijective σ ∧ ∀ R : Finset (Fin d.D.N),
      d'.D.size (R.image σ) = d.D.size R ∧ Nci.novelReads d' (R.image σ) = Nci.novelReads d R := by
  obtain ⟨σ, hσ, hw, -, hr, hθ, -⟩ := served_pins h h'
  exact ⟨σ, hσ, fun R => ⟨Finset.card_image_of_injective R hσ.1, novelReads_image hσ hw hr hθ R⟩⟩

/-- Two options of block numbers that agree as numbers name the same block. -/
theorem unit_iff {n n' : ℕ} {o : Option (Fin n)} {o' : Option (Fin n')} (h : o'.map Fin.val = o.map Fin.val)
    {u : Fin n} {u' : Fin n'} (huu : u.val = u'.val) : o = some u ↔ o' = some u' := by
  constructor
  · rintro rfl
    cases o' with
    | none => simp at h
    | some v =>
      have h2 : (some v.val : Option ℕ) = some u.val := h
      exact congrArg some (Fin.ext ((Option.some.inj h2).trans huu))
  · rintro rfl
    cases o with
    | none => simp at h
    | some v =>
      have h2 : (some u'.val : Option ℕ) = some v.val := h
      exact congrArg some (Fin.ext ((Option.some.inj h2).symm.trans huu.symm))

/-- **The blocks are pinned where no gate lies in units of two slots**: block `u` of one circuit and block `u` of
the other have one size and one count of fresh bits. -/
theorem served_pins_blocks {F : Form} {c : F.Case} {d d' : Verity.RegisteredCircuit} (h : Served F c d)
    (h' : Served F c d') (hone : OneSlot F c) (u : Fin d.n) (u' : Fin d'.n) (huu : u.val = u'.val) :
    d'.D.size (d'.𝓘.gates u') = d.D.size (d.𝓘.gates u) ∧
      Nci.novelReads d' (d'.𝓘.gates u') = Nci.novelReads d (d.𝓘.gates u) := by
  obtain ⟨σ, hσ, hw, -, hr, hθ, hunit⟩ := served_pins h h'
  have himg : (d.𝓘.gates u).image σ = d'.𝓘.gates u' := by
    ext y
    obtain ⟨x, rfl⟩ := hσ.2 y
    simp only [Finset.mem_image, Verity.Partition.gates, Finset.mem_filter, Finset.mem_univ, true_and]
    constructor
    · rintro ⟨a, ha, hax⟩
      have hxa : a = x := hσ.1 hax
      rw [hxa] at ha
      exact (unit_iff (hunit hone x) huu).mp ha
    · intro hx
      exact ⟨x, (unit_iff (hunit hone x) huu).mpr hx, rfl⟩
  rw [← himg]
  exact ⟨Finset.card_image_of_injective _ hσ.1, novelReads_image hσ hw hr hθ _⟩

end NciCheck.Properties.Record.Attack
