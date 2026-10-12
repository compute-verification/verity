import Mathlib
import Definitions.Accounting.Confidence
import Definitions.Accounting.Physical
import Proofs.Accounting.Actual.Split

/-!
# Causality at every location doesn't confine a location's past to its state

`LocallyCausal` makes what a location `X` does before `t` a function of its inputs and of everything it read from
outside since tick 0. `mem_X(t)` (`Placed.state`) counts only the gates `X` carries across `t`. A strategy's circuit
varies with the coins, so `X` can read a value before a window, carry nothing across the window's start, and in the
window run a gate that reads nothing and whose value is what it read: such a gate is faithful at every constant.

`not_markov` is such a strategy over one coin `b`, at leaves `true` (the verifier) and `false` (`X`):
* `g0`, the verifier's input, holds `b`; `g1`, at the verifier at tick 0, copies it; `g2`, at `X` at tick 1, reads `g1`
  over the one link;
* `g3`, at `X` at tick 3, reads nothing and has value `b`.

It is locally causal, its prover's inputs don't depend on the coin, and every run is feasible, with `M_X = 0`. In the
window `[2, 4)`, `X` carries nothing across tick 2 and reads nothing, yet what it runs there differs with `b`. So a
protocol that reads `mem_X` as what `X` can know at a challenge (PoUS's responder, the bandwidth audit's cores, B2 and
B3 of `note:console/20261012T0645Z-draft-actual-circuit-model` v2.1) needs causality in its Markov form: what `X` does
in `[t, t')` is a function of the values of the gates it carries across `t` and of what it reads from outside in
`[t, t')`.
-/

namespace Accounting.SecurityProofs.Actual.Markov

open Verity Finset

/-- The circuit at coin `b`: `g1` copies `g0`, `g2` copies `g1`, and `g3` reads nothing and is `b`. -/
abbrev circ (b : Bool) : Circuit where
  N := 4
  w _ := 1
  isInput g := decide (g = 0)
  reads g := if g = 1 then {0} else if g = 2 then {1} else ∅
  reads_lt g a ha := by fin_cases g <;> simp_all
  input_reads g hg := by fin_cases g <;> simp_all
  op g v := if g = 1 then v 0 else if g = 2 then v 1 else BitVec.ofBool b
  op_congr g v v' h := by fin_cases g <;> simp_all
  outputs := ∅

/-- The actual circuit at coin `b`: `g0` and `g1` at the verifier at tick 0, `g2` at `X` at tick 1, `g3` at `X` at
tick 3. -/
abbrev placed (b : Bool) : Placed Bool :=
  { circ b with
    x := fun g => decide (g.val < 2)
    t := fun g => if g.val = 2 then 1 else if g.val = 3 then 3 else 0 }

/-- The run at coin `b`: every gate's value is `b`. -/
abbrev run (b : Bool) : Actual.Run Bool where
  A := placed b
  val _ := BitVec.ofBool b
  faithful g hg := by
    fin_cases g <;> simp_all [placed, circ]

/-- The strategy: coin `b` to its run. -/
abbrev S : Bool → Actual.Run Bool := run

/-- The hardware: `X = {false}` the one unit, with no memory; one link of a bit a tick between the leaves, latency 1
across it; the verifier trusted. -/
def H : Hardware Bool Unit where
  units := {{false}}
  ℓ x y := if x = y then 0 else 1
  compute _ := 0
  M _ := 0
  ends _ := (true, false)
  β _ := 1
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

theorem seen_eq (b : Bool) (g : Fin (S b).A.N) :
    (S b).seen g = ((S b).A.t g, (S b).A.x g, 1, b.toNat) := by
  cases b <;> rfl

theorem reads_x (b : Bool) (g a : Fin (S b).A.N) (ha : a ∈ (S b).A.reads g) : (S b).A.x a = true := by
  fin_cases g <;> fin_cases a <;> simp_all [S, run, placed, circ]

theorem state_zero_all (b : Bool) (t : ℕ) : (S b).A.state {false} t = 0 := by
  unfold Placed.state Circuit.width
  refine Finset.sum_eq_zero fun a ha => ?_
  exfalso
  simp only [mem_filter] at ha
  obtain ⟨ha, hx⟩ := ha
  obtain ⟨-, g, -, hag⟩ := (mem_into _).1 ha
  have h1 := reads_x b g a hag
  simp at h1 hx
  omega

/-- A set of gates read by a gate of `X` from outside it, a single gate of width 1 read at tick 1 and sent at tick 0:
at most a bit a tick of any window that contains that tick. -/
theorem width_le_window (b : Bool) (T : Finset (Fin (S b).A.N)) (hT : T ⊆ {(1 : Fin 4)}) (a c : ℕ)
    (hw : (1 : Fin 4) ∈ T → a < c) : (S b).A.width T ≤ 1 * (c - a) := by
  rcases Finset.subset_singleton_iff.1 hT with h | h
  · simp [h, Circuit.width]
  · have := hw (by simp [h])
    subst h
    simp only [Circuit.width, sum_singleton, one_mul]
    change 1 ≤ c - a
    omega

theorem feasible (b : Bool) : Feasible H (S b).A (ann b) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro g a ha
    fin_cases g <;> fin_cases a <;> simp_all [S, run, placed, circ, H]
  · intro X _ a c
    simp [Placed.work, ann]
  · intro k p _ hr
    exact hr.elim
  · intro X hX t
    simp only [H, Set.mem_singleton_iff] at hX
    subst hX
    rw [state_zero_all]
    exact Nat.zero_le _
  · refine ⟨fun g a ha hne => ?_, fun e y a c => ⟨?_, ?_⟩⟩
    · fin_cases g <;> fin_cases a <;> simp_all [S, run, placed, circ, H]
    · refine width_le_window b _ ?_ a c ?_
      · intro z hz
        simp only [Placed.readOn, mem_filter, mem_univ, true_and] at hz
        obtain ⟨g, hzg, hne, -⟩ := hz
        fin_cases g <;> fin_cases z <;> simp_all [S, run, placed, circ]
      · intro hz
        simp only [Placed.readOn, mem_filter, mem_univ, true_and] at hz
        obtain ⟨g, hzg, hgt⟩ := hz
        fin_cases g <;> simp at hzg hgt ⊢
        omega
    · refine width_le_window b _ ?_ a c ?_
      · intro z hz
        simp only [Placed.sentOn, mem_filter, mem_univ, true_and] at hz
        obtain ⟨-, g, hzg, hne, -⟩ := hz
        fin_cases g <;> fin_cases z <;> simp_all [S, run, placed, circ]
      · intro hz
        simp only [Placed.sentOn, mem_filter, mem_univ, true_and] at hz
        obtain ⟨hgt, -⟩ := hz
        simp at hgt
        omega
  · refine ⟨by simp [H], fun g _ _ => ⟨rfl, rfl⟩, fun k => ?_⟩
    simp [H, Circuit.width]

theorem locallyCausal : LocallyCausal S := by
  intro X t r r' hi hh
  by_cases hr : r = r'
  · subst hr; rfl
  have hv : r.toNat ≠ r'.toNat := by cases r <;> cases r' <;> simp_all
  -- Inputs equal at different coins: the verifier's leaf is not in `X`.
  have hX : true ∉ X := by
    intro hX
    have : ((0 : ℕ), true, 1, r.toNat) ∈ (S r).inputs X := ⟨0, by simp, hX, by
      rw [seen_eq]; rfl⟩
    rw [hi] at this
    obtain ⟨g, hg, -, he⟩ := this
    rw [seen_eq] at he
    exact hv (congrArg (fun p => p.2.2.2) he)
  -- Heard equal: `X` doesn't hear `g1` before `t`.
  have hT : ¬ (false ∈ X ∧ 2 ≤ t) := by
    rintro ⟨hf, ht⟩
    have : ((1 : ℕ), ((0 : ℕ), true, 1, r.toNat)) ∈ (S r).heard X t :=
      ⟨2, 1, by simp [S, run, placed, circ], by simpa [S, run, placed, circ] using hf,
        by simpa [circ] using hX, by simp [circ]; omega, by rw [seen_eq]; rfl⟩
    rw [hh] at this
    obtain ⟨g, a, -, -, -, -, he⟩ := this
    rw [seen_eq] at he
    exact hv (congrArg (fun p => p.2.2.2.2) he)
  -- So `X` has run nothing before `t`, at either coin.
  have hempty : ∀ b, (S b).before X t = ∅ := by
    intro b
    ext p
    simp only [Actual.Run.before, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
    intro g hg hgt _
    fin_cases g <;> simp at hg hgt <;> first | exact hX hg | exact hT ⟨hg, by omega⟩
  rw [hempty, hempty]

theorem inputs_prover (r r' : Bool) : (S r).inputs (↑({true} : Finset Bool))ᶜ = (S r').inputs (↑({true} : Finset Bool))ᶜ := by
  have : ∀ b, (S b).inputs (↑({true} : Finset Bool))ᶜ = ∅ := by
    intro b
    ext p
    simp only [Actual.Run.inputs, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
    intro g hg hx _
    fin_cases g <;> simp_all
  rw [this, this]

theorem state_zero (b : Bool) : (S b).A.state {false} 2 = 0 := state_zero_all b 2

theorem ingress_zero (b : Bool) : (S b).A.ingress {false} (Set.Ico 2 4) = 0 := by
  unfold Placed.ingress Circuit.width
  refine Finset.sum_eq_zero fun a ha => ?_
  exfalso
  simp only [mem_filter] at ha
  obtain ⟨ha, -⟩ := ha
  obtain ⟨-, g, hg, hag⟩ := (mem_into _).1 ha
  obtain ⟨-, hgx, hgt⟩ := (mem_region _).1 hg
  fin_cases g <;> fin_cases a <;> simp_all [S, run, placed, circ]

/-- What `X = {false}` runs during `[2, 4)`: its gates placed then, when and with what values. -/
def window (b : Bool) : Set (ℕ × Bool × ℕ × ℕ) :=
  {p | ∃ g, (S b).A.x g ∈ ({false} : Finset Bool) ∧ (S b).A.t g ∈ Set.Ico 2 4 ∧ p = (S b).seen g}

theorem window_ne : window false ≠ window true := by
  intro h
  have : ((3 : ℕ), false, 1, (true).toNat) ∈ window true :=
    ⟨3, by simp [circ], by simp [circ], by rw [seen_eq]; rfl⟩
  rw [← h] at this
  obtain ⟨g, -, -, he⟩ := this
  rw [seen_eq] at he
  simp at he

/-- **Causality at every location and memory don't confine a location's past to its state.** A locally causal strategy
over one coin whose prover's inputs don't depend on it and whose every run is feasible, with `M_X = 0` at
`X = {false}`, in which `X` carries nothing across tick 2 and reads nothing during `[2, 4)`, yet what `X` runs during
`[2, 4)` depends on the coin. -/
theorem not_markov :
    LocallyCausal S ∧ (∀ r r', (S r).inputs (↑({true} : Finset Bool))ᶜ = (S r').inputs (↑({true} : Finset Bool))ᶜ) ∧
      (∀ b, Feasible H (S b).A (ann b)) ∧ ({false} ∈ H.units ∧ H.M {false} = 0) ∧
      (∀ b, (S b).A.state {false} 2 = 0 ∧ (S b).A.ingress {false} (Set.Ico 2 4) = 0) ∧
      window false ≠ window true :=
  ⟨locallyCausal, inputs_prover, feasible, ⟨rfl, rfl⟩, fun b => ⟨state_zero b, ingress_zero b⟩, window_ne⟩

end Accounting.SecurityProofs.Actual.Markov
