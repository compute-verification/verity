import Mathlib
import Definitions.Accounting.Confidence
import Definitions.Accounting.Physical
import Proofs.Accounting.Actual.Split

/-!
# Widths aren't information: a link of capacity 0 carries a bit

`Links` bounds the widths a link carries. `LocallyCausal` lets a location's behaviour depend on everything it hears,
and `heard` records each wire with its arrival tick and the sender's tick, leaf and width. A wire may have width 0.
So a link of capacity 0 carries width-0 wires, and when they are sent carries information.

`zero_link` is such a strategy over one coin `b`, at leaves `true` (the verifier) and `false` (`X`), joined by one link
of capacity 0 and latency 1:
* `g0`, the verifier's input, holds `b`;
* `g1`, at the verifier, has width 0 and is sent at tick `b`;
* `g2`, at `X` at tick 3, reads `g1` and has value `b`.

It is locally causal, `X` has no inputs, and every run is feasible, with `M_X = 0`. Yet what `X` runs depends on the
coin. The same works with the arrival tick instead of the sender's. So the side budget of the bandwidth audit on the
actual circuit (B3 of `note:console/20261012T0645Z-draft-actual-circuit-model` v2.1), `β_side (T + RTT)`, needs
widths to be information. Wires between leaves need width at least 1, a location hears values with their arrival
ticks only, and the budget counts arrival timing.
-/

namespace Accounting.SecurityProofs.Actual.Timing

open Verity Finset

/-- The widths: `g1` is 0, the others 1. -/
abbrev wid (g : Fin 3) : ℕ := if g.val = 1 then 0 else 1

/-- The circuit at coin `b`: `g1` reads `g0` and `g2` reads `g1`. Every gate is the constant `b` at its width. -/
abbrev circ (b : Bool) : Circuit where
  N := 3
  w := wid
  isInput g := decide (g = 0)
  reads g := if g = 1 then {0} else if g = 2 then {1} else ∅
  reads_lt g a ha := by fin_cases g <;> simp_all
  input_reads g hg := by fin_cases g <;> simp_all
  op g _ := BitVec.ofNat (wid g) b.toNat
  op_congr _ _ _ _ := rfl
  outputs := ∅

/-- The actual circuit at coin `b`: `g0` at the verifier at tick 0, `g1` at the verifier at tick `b`, `g2` at `X` at
tick 3. -/
abbrev placed (b : Bool) : Placed Bool :=
  { circ b with
    x := fun g => decide (g.val < 2)
    t := fun g => if g.val = 1 then b.toNat else if g.val = 2 then 3 else 0 }

/-- The run at coin `b`: every gate is `b` at its width. -/
abbrev run (b : Bool) : Actual.Run Bool where
  A := placed b
  val g := BitVec.ofNat (wid g) b.toNat
  faithful _ _ := rfl

abbrev S : Bool → Actual.Run Bool := run

/-- The hardware: `X = {false}` the one unit, with no memory; one link of capacity 0 between the leaves, latency 1
across it; the verifier trusted. -/
def H : Hardware Bool Unit where
  units := {{false}}
  ℓ x y := if x = y then 0 else 1
  compute _ := 0
  M _ := 0
  ends _ := (true, false)
  β _ := 0
  tRoot := 0
  cert := ∅
  trusted := {true}
  bucket := id
  jitter := 0

/-- The annotation: no price, every wire on the one link, no P2 root, the verifier's gates as they are, no record. -/
def ann (b : Bool) : Annotation (S b).A Unit where
  π _ := 0
  link _ _ := ()
  root _ _ := False
  spec := ⟨(S b).A.reads, (S b).A.op⟩
  record := ∅

theorem reads_x (b : Bool) (g a : Fin (S b).A.N) (ha : a ∈ (S b).A.reads g) : (S b).A.x a = true := by
  fin_cases g <;> fin_cases a <;> simp_all

/-- Everything read from another leaf is `g1`, of width 0. -/
theorem width_cross (b : Bool) (T : Finset (Fin (S b).A.N)) (hT : ∀ a ∈ T, a = (1 : Fin 3)) (n : ℕ) :
    (S b).A.width T ≤ 0 * n := by
  simp only [zero_mul, nonpos_iff_eq_zero]
  refine Finset.sum_eq_zero fun a ha => ?_
  rw [hT a ha]
  rfl

theorem feasible (b : Bool) : Feasible H (S b).A (ann b) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro g a ha
    fin_cases g <;> fin_cases a <;> cases b <;> simp_all [H]
  · intro X _ a c
    simp [Placed.work, ann]
  · intro k p _ hr
    exact hr.elim
  · intro X hX t
    simp only [H, Set.mem_singleton_iff] at hX
    subst hX
    unfold Placed.state Circuit.width
    refine (Finset.sum_eq_zero fun a ha => ?_).le
    exfalso
    simp only [mem_filter] at ha
    obtain ⟨ha, hx⟩ := ha
    obtain ⟨-, g, -, hag⟩ := (mem_into _).1 ha
    have h1 := reads_x b g a hag
    simp at h1 hx
    omega
  · refine ⟨fun g a ha hne => ?_, fun e y a c => ⟨?_, ?_⟩⟩
    · fin_cases g <;> fin_cases a <;> simp_all [H]
    · refine width_cross b _ (fun z hz => ?_) _
      simp only [Placed.readOn, mem_filter, mem_univ, true_and] at hz
      obtain ⟨g, hzg, hne, -⟩ := hz
      fin_cases g <;> fin_cases z <;> simp_all
    · refine width_cross b _ (fun z hz => ?_) _
      simp only [Placed.sentOn, mem_filter, mem_univ, true_and] at hz
      obtain ⟨-, g, hzg, hne, -⟩ := hz
      fin_cases g <;> fin_cases z <;> simp_all
  · refine ⟨by simp [H], fun g _ _ => ⟨rfl, rfl⟩, fun k => ?_⟩
    simp [H, Circuit.width]

theorem locallyCausal : LocallyCausal S := by
  intro X t r r' hi hh
  by_cases hr : r = r'
  · subst hr; rfl
  have hv : r.toNat ≠ r'.toNat := by cases r <;> cases r' <;> simp_all
  have hX : true ∉ X := by
    intro hX
    have : ((0 : ℕ), true, 1, r.toNat) ∈ (S r).inputs X := ⟨0, by simp, hX, by cases r <;> rfl⟩
    rw [hi] at this
    obtain ⟨g, hg, -, he⟩ := this
    fin_cases g <;> cases r <;> cases r' <;> simp_all [Actual.Run.seen]
  have hT : ¬ (false ∈ X ∧ 4 ≤ t) := by
    rintro ⟨hf, ht⟩
    have : ((3 : ℕ), (r.toNat, true, 0, 0)) ∈ (S r).heard X t :=
      ⟨2, 1, by simp, by simpa using hf, by simpa using hX, by simp; omega, by cases r <;> rfl⟩
    rw [hh] at this
    obtain ⟨g, a, hga, -, -, -, he⟩ := this
    fin_cases g <;> fin_cases a <;> cases r <;> cases r' <;> simp_all [Actual.Run.seen]
  have hempty : ∀ b, (S b).before X t = ∅ := by
    intro b
    ext p
    simp only [Actual.Run.before, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
    intro g hg hgt _
    fin_cases g <;> simp at hg hgt <;> first | exact hX hg | exact hT ⟨hg, by omega⟩
  rw [hempty, hempty]

theorem inputs_X (r r' : Bool) : (S r).inputs (↑({true} : Finset Bool))ᶜ = (S r').inputs (↑({true} : Finset Bool))ᶜ := by
  have : ∀ b, (S b).inputs (↑({true} : Finset Bool))ᶜ = ∅ := by
    intro b
    ext p
    simp only [Actual.Run.inputs, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
    intro g hg hx _
    fin_cases g <;> simp_all
  rw [this, this]

theorem before_ne : (S false).before {false} 4 ≠ (S true).before {false} 4 := by
  intro h
  have : ((3 : ℕ), false, 1, (true).toNat) ∈ (S true).before {false} 4 := ⟨2, by simp, by simp, rfl⟩
  rw [← h] at this
  obtain ⟨g, hg, hgt, he⟩ := this
  fin_cases g <;> simp_all [Actual.Run.seen]

/-- **A link of capacity 0 carries a bit.** A locally causal strategy over one coin in which `X` has no inputs, every
run is feasible with `M_X = 0` and the one link between `X` and the verifier has capacity 0, yet what `X` runs depends
on the coin. -/
theorem zero_link :
    LocallyCausal S ∧ (∀ r r', (S r).inputs (↑({true} : Finset Bool))ᶜ = (S r').inputs (↑({true} : Finset Bool))ᶜ) ∧
      (∀ b, Feasible H (S b).A (ann b)) ∧ ({false} ∈ H.units ∧ H.M {false} = 0) ∧ (∀ e, H.β e = 0) ∧
      (S false).before {false} 4 ≠ (S true).before {false} 4 :=
  ⟨locallyCausal, inputs_X, feasible, ⟨rfl, rfl⟩, fun _ => rfl, before_ne⟩

end Accounting.SecurityProofs.Actual.Timing
