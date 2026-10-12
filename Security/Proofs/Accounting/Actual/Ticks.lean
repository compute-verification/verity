import Mathlib
import Definitions.Accounting.Confidence
import Definitions.Accounting.Physical
import Proofs.Accounting.Actual.Split

/-!
# A held gate's tick carries a bit that memory doesn't count

`CausalFrom` makes what a location does a function of its state *as seen*, each gate with its tick, leaf, width and
value; `MemoryOver` (and the per-run `Memory`) bound the state's *width*. A gate of width 0 placed at a tick that
depends on the coins is in the state at width 0, and its tick tells the location the coin. So a location with no
memory carries information across a window in which it hears nothing.

`held_tick` is such a strategy over one coin `b`, at leaves `true` (the verifier) and `false` (`X`), joined by one link
of a bit a tick (as an alphabet: two signals a tick) and latency 1:
* `g0`, the verifier's input, holds `b`; `g2`, at the verifier at tick 10, reads it, so the verifier's state depends on
  the coin until it is done;
* `g1`, at the verifier, has width 0 and is sent at tick `1 + b`;
* `g3`, at `X` at tick `2 + b`, reads `g1` and has width 0;
* `g4`, at `X` at tick 6, reads `g3`, has width 1 and value `b`.

It is causal from every tick, its links are alphabets of `2^1` signals, every run is feasible with `M_X = 0`, `X` holds
no bits over continuations at any tick and has no inputs, and nothing crosses into `X` during `[5, 7)`; yet what `X`
does during `[5, 7)` depends on the coin. With `n` such gates the ticks carry about `n log₂ T` bits at width 0. So the
reductions of B2 and B3 (`note:console/20261012T0645Z-draft-actual-circuit-model`, section 8), which make `X` a model
prover whose state is `mem_X(t₀)` bits, need the state's information bounded, not its width. One form: memory as an
alphabet, at each tick at most `2^{M_X}` states as seen across the strategy's runs (`MemoryAlphabet`), which this
strategy fails (`not_memoryAlphabet`); or causality keyed by the state's values without their ticks.
-/

namespace Accounting.SecurityProofs.Actual.Ticks

open Verity Finset

/-- The widths: `g1` and `g3` are 0, the others 1. -/
abbrev wid (g : Fin 5) : ℕ := if g.val = 1 ∨ g.val = 3 then 0 else 1

/-- The circuit at coin `b`: `g1` and `g2` read `g0`, `g3` reads `g1`, `g4` reads `g3`. Every gate is the constant `b`
at its width. -/
abbrev circ (b : Bool) : Circuit where
  N := 5
  w := wid
  isInput g := decide (g = 0)
  reads g := if g = 1 ∨ g = 2 then {0} else if g = 3 then {1} else if g = 4 then {3} else ∅
  reads_lt g a ha := by fin_cases g <;> simp_all
  input_reads g hg := by fin_cases g <;> simp_all
  op g _ := BitVec.ofNat (wid g) b.toNat
  op_congr _ _ _ _ := rfl
  outputs := ∅

/-- The actual circuit at coin `b`: `g0`, `g1`, `g2` at the verifier at ticks 0, `1 + b`, 10; `g3`, `g4` at `X` at
ticks `2 + b`, 6. -/
abbrev placed (b : Bool) : Placed Bool :=
  { circ b with
    x := fun g => decide (g.val < 3)
    t := fun g => if g.val = 1 then 1 + b.toNat else if g.val = 2 then 10 else if g.val = 3 then 2 + b.toNat else
      if g.val = 4 then 6 else 0 }

/-- The run at coin `b`: every gate is `b` at its width. -/
abbrev run (b : Bool) : Actual.Run Bool where
  A := placed b
  val g := BitVec.ofNat (wid g) b.toNat
  faithful _ _ := rfl

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
    (S b).seen g = ((S b).A.t g, (S b).A.x g, wid g, (BitVec.ofNat (wid g) b.toNat).toNat) := rfl

/-- A gate seen at width 1 shows the coin. -/
theorem seen_coin (b : Bool) (g : Fin (S b).A.N) (h : ((S b).seen g).2.2.1 = 1) : ((S b).seen g).2.2.2 = b.toNat := by
  fin_cases g <;> cases b <;> simp_all [seen_eq]

/-- The one gate of `X` at width 0 is `g3`, at tick `2 + b`. -/
theorem seen_tick (b : Bool) (g : Fin (S b).A.N) (hx : ((S b).seen g).2.1 = false) (hw : ((S b).seen g).2.2.1 = 0) :
    ((S b).seen g).1 = 2 + b.toNat := by
  fin_cases g <;> simp_all [seen_eq]

/-- The only gate of `X` that any gate reads is `g3`. -/
theorem read_x (b : Bool) (g a : Fin (S b).A.N) (ha : a ∈ (S b).A.reads g) (hx : (S b).A.x a = false) :
    a = (3 : Fin 5) := by
  fin_cases g <;> fin_cases a <;> simp_all

/-- No gate placed in `X` during `I`: `X` does nothing then. -/
theorem acts_empty (b : Bool) (X : Set Bool) (I : Set ℕ) (h : ∀ g, (S b).A.x g ∈ X → (S b).A.t g ∉ I) :
    (S b).acts X I = (∅, ∅) := by
  simp only [Actual.Run.acts, Actual.Run.sent, Prod.mk.injEq]
  constructor
  · ext p
    simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
    exact fun g hg ht => absurd ht (h g hg)
  · ext p
    simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
    exact fun _ a _ hx _ ht => absurd ht (h a hx)

theorem causalFrom : CausalFrom S := by
  intro X t u r r' hs hc
  by_cases hr : r = r'
  · subst hr; rfl
  have hv : r.toNat ≠ r'.toNat := by cases r <;> cases r' <;> simp_all
  have hb := Bool.toNat_le r
  have hb' := Bool.toNat_le r'
  by_cases hT : true ∈ X
  · by_cases ht : t ≤ 10
    · -- The verifier still holds its coin `g0` (read by `g2` at tick 10), at width 1.
      exfalso
      have hmem : (0 : Fin 5) ∈ stateOver S X t r := by
        simp only [stateOver, Finset.mem_filter, Finset.mem_univ, true_and]
        refine ⟨by simpa using hT, fun h => by have := ((mem_region _).1 h).1; simp at this, r, rfl, 0, ?_, rfl⟩
        refine (mem_into _).2 ⟨fun h => by have := ((mem_region _).1 h).1; simp at this, 2, ?_, by simp⟩
        exact (mem_region _).2 ⟨rfl, by simpa using hT, by simpa using ht⟩
      have : (S r).seen 0 ∈ (S r').seen '' ↑(stateOver S X t r') := hs ▸ ⟨0, hmem, rfl⟩
      obtain ⟨a, -, ha⟩ := this
      have h1 := seen_coin r' a (by rw [ha]; cases r <;> rfl)
      rw [ha, seen_eq] at h1
      cases r <;> cases r' <;> simp_all
    · rw [acts_empty r _ _ fun g _ hg => ?_, acts_empty r' _ _ fun g _ hg => ?_] <;>
        · simp only [Set.mem_Ico] at hg
          fin_cases g <;> simp at hg <;> omega
  by_cases hF : false ∈ X
  · have hX : ∀ g : Fin 5, ((decide (g.val < 3) : Bool) ∈ (↑X : Set Bool)) ↔ 3 ≤ g.val := by
      intro g
      by_cases h : g.val < 3
      · simp [h, hT]
      · simp [h, hF]
        omega
    by_cases h7 : 7 ≤ t
    · rw [acts_empty r _ _ fun g hg ht => ?_, acts_empty r' _ _ fun g hg ht => ?_] <;>
        · simp only [Set.mem_Ico] at ht
          have := (hX g).1 hg
          fin_cases g <;> simp at this ht <;> omega
    by_cases h3 : 3 ≤ t
    · -- `g3` of the run at `false`, placed at tick 2 and read by `g4` at tick 6, is in its state; the other run's `g3`
      -- is at tick 3.
      exfalso
      have hst : ∀ b : Bool, b = false → (3 : Fin 5) ∈ stateOver S X t b := by
        rintro b rfl
        simp only [stateOver, Finset.mem_filter, Finset.mem_univ, true_and]
        refine ⟨by simpa using hF, fun h => ?_, false, rfl, 3, ?_, rfl⟩
        · have := ((mem_region _).1 h).2.2
          simp at this
          omega
        refine (mem_into _).2 ⟨fun h => ?_, 4, ?_, by simp⟩
        · have := ((mem_region _).1 h).2.2
          simp at this
          omega
        · exact (mem_region _).2 ⟨rfl, by simpa using hF, by simp; omega⟩
      have key : ∀ b b' : Bool, b = false → b' = true →
          (S b).seen '' ↑(stateOver S X t b) ≠ (S b').seen '' ↑(stateOver S X t b') := by
        rintro b b' rfl rfl he
        have : (S false).seen 3 ∈ (S true).seen '' ↑(stateOver S X t true) := he ▸ ⟨3, hst false rfl, rfl⟩
        obtain ⟨a, -, ha⟩ := this
        have := seen_tick true a (by rw [ha]; rfl) (by rw [ha]; rfl)
        rw [ha] at this
        simp [seen_eq] at this
      cases r <;> cases r' <;> simp_all
    · by_cases hu : u ≤ 2
      · rw [acts_empty r _ _ fun g hg ht => ?_, acts_empty r' _ _ fun g hg ht => ?_] <;>
          · simp only [Set.mem_Ico] at ht
            have := (hX g).1 hg
            fin_cases g <;> simp at this ht <;> cases r <;> cases r' <;> simp_all <;> omega
      · -- What crosses into `X` during `[t, u)`: `g1`, read at tick `2 + b`.
        exfalso
        have hcr : ∀ b : Bool, b = false →
            ((2 : ℕ), false, (S b).seen 1) ∈ (S b).crossing (↑X)ᶜ ↑X (Set.Ico t u) := by
          rintro b rfl
          exact ⟨3, 1, by simp, by simpa using hT, by simpa using hF, by simp; omega, rfl⟩
        have key : ∀ b b' : Bool, b = false → b' = true →
            (S b).crossing (↑X)ᶜ ↑X (Set.Ico t u) ≠ (S b').crossing (↑X)ᶜ ↑X (Set.Ico t u) := by
          rintro b b' rfl rfl he
          have : ((2 : ℕ), false, (S false).seen 1) ∈ (S true).crossing (↑X)ᶜ ↑X (Set.Ico t u) := he ▸ hcr false rfl
          obtain ⟨g, a, hga, hxa, hxg, -, he⟩ := this
          simp only [Set.mem_compl_iff, Finset.mem_coe] at hxa hxg
          fin_cases g <;> fin_cases a <;> simp_all
        cases r <;> cases r' <;> simp_all
  · rw [acts_empty r _ _ fun g hg _ => ?_, acts_empty r' _ _ fun g hg _ => ?_] <;>
      · simp only [Finset.mem_coe] at hg
        cases h : (decide (g.val < 3) : Bool) <;> simp_all

theorem admissible : Admissible S := ⟨Bool, S, id, causalFrom, fun _ => rfl⟩

/-- `X` holds nothing of positive width over continuations: the gates of `X` that any gate reads are `g3`. -/
theorem memoryOver : MemoryOver S {{false}} (fun _ => 0) := by
  intro r X hX t
  simp only [Set.mem_singleton_iff] at hX
  subst hX
  unfold Circuit.width
  refine (Finset.sum_eq_zero fun a ha => ?_).le
  simp only [stateOver, Finset.mem_filter, Finset.mem_univ, true_and] at ha
  obtain ⟨hx, -, r'', -, a', ha', he⟩ := ha
  obtain ⟨-, g, -, hga⟩ := (mem_into _).1 ha'
  have hxa : (S r).A.x a = false := by simpa using hx
  have hxa' : (S r'').A.x a' = false := by
    have := congrArg (fun p => p.2.1) he
    simp only [Actual.Run.seen] at this
    exact this.trans hxa
  have h3 := read_x r'' g a' hga hxa'
  have := congrArg (fun p => p.2.2.1) he
  simp only [Actual.Run.seen] at this
  show wid a = 0
  rw [← this, h3]
  rfl

/-- Two coins, so each link delivers at most two signals a tick: an alphabet of `2^1`. -/
theorem alphabet : Alphabet S (fun _ : Unit => 1) (fun b => (ann b).link) := by
  intro e y u
  refine (Set.encard_le_encard (s := Set.range fun r => (S r).signal ((ann r).link) e y u)
    (t := {(S false).signal ((ann false).link) e y u, (S true).signal ((ann true).link) e y u}) ?_).trans ?_
  · rintro _ ⟨r, rfl⟩
    cases r <;> simp
  · refine (Set.encard_insert_le _ _).trans ?_
    simp
    norm_num

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
    simp only [mem_filter] at ha
    obtain ⟨ha, hx⟩ := ha
    obtain ⟨-, g, -, hag⟩ := (mem_into _).1 ha
    rw [read_x b g a hag (by simpa using hx)]
    rfl
  · refine ⟨fun g a ha hne => ?_, fun e y a c => ⟨?_, ?_⟩⟩
    · fin_cases g <;> fin_cases a <;> simp_all [H]
    · refine le_trans (Finset.sum_eq_zero fun z hz => ?_).le (Nat.zero_le _)
      simp only [Placed.readOn, mem_filter, mem_univ, true_and] at hz
      obtain ⟨g, hzg, hne, -⟩ := hz
      fin_cases g <;> fin_cases z <;> simp_all
    · refine le_trans (Finset.sum_eq_zero fun z hz => ?_).le (Nat.zero_le _)
      simp only [Placed.sentOn, mem_filter, mem_univ, true_and] at hz
      obtain ⟨-, g, hzg, hne, -⟩ := hz
      fin_cases g <;> fin_cases z <;> simp_all
  · refine ⟨by simp [H], fun g _ _ => ⟨rfl, rfl⟩, fun k => ?_⟩
    simp [H, Circuit.width]

theorem inputs_prover (r r' : Bool) :
    (S r).inputs (↑({true} : Finset Bool))ᶜ = (S r').inputs (↑({true} : Finset Bool))ᶜ := by
  have : ∀ b, (S b).inputs (↑({true} : Finset Bool))ᶜ = ∅ := by
    intro b
    ext p
    simp only [Actual.Run.inputs, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
    intro g hg hx _
    fin_cases g <;> simp_all
  rw [this, this]

theorem crossing_empty (b : Bool) :
    (S b).crossing (↑({false} : Finset Bool))ᶜ ↑({false} : Finset Bool) (Set.Ico 5 7) = ∅ := by
  ext p
  simp only [Actual.Run.crossing, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
  intro g a hga hxa hxg hgt _
  simp only [Set.mem_compl_iff, Finset.coe_singleton, Set.mem_singleton_iff, Set.mem_Ico] at hxa hxg hgt
  fin_cases g <;> fin_cases a <;> cases b <;> simp_all

theorem acts_ne :
    (S false).acts ↑({false} : Finset Bool) (Set.Ico 5 7) ≠ (S true).acts ↑({false} : Finset Bool) (Set.Ico 5 7) := by
  intro h
  have : (S false).seen 4 ∈ ((S true).acts ↑({false} : Finset Bool) (Set.Ico 5 7)).1 :=
    h ▸ ⟨4, by simp, by simp, rfl⟩
  obtain ⟨g, hg, hgt, he⟩ := this
  simp only [Finset.coe_singleton, Set.mem_singleton_iff, Set.mem_Ico] at hg hgt
  fin_cases g <;> simp_all [seen_eq]

/-- **A held gate's tick carries a bit.** The strategy `S` is causal from every tick (so admissible), its links are
alphabets of two signals a tick, every run is feasible with `M_X = 0`, `X` holds no bits over continuations at any tick
and has no inputs, and nothing crosses into `X` during `[5, 7)`; yet what `X` does during `[5, 7)` depends on the
coin. -/
theorem held_tick :
    CausalFrom S ∧ Admissible S ∧ Alphabet S (fun _ : Unit => 1) (fun b => (ann b).link) ∧
      (∀ b, Feasible H (S b).A (ann b)) ∧ ({false} ∈ H.units ∧ H.M {false} = 0) ∧
      MemoryOver S {{false}} (fun _ => 0) ∧
      (∀ r r', (S r).inputs (↑({true} : Finset Bool))ᶜ = (S r').inputs (↑({true} : Finset Bool))ᶜ) ∧
      (∀ b, (S b).crossing (↑({false} : Finset Bool))ᶜ ↑({false} : Finset Bool) (Set.Ico 5 7) = ∅) ∧
      (S false).acts ↑({false} : Finset Bool) (Set.Ico 5 7) ≠ (S true).acts ↑({false} : Finset Bool) (Set.Ico 5 7) :=
  ⟨causalFrom, admissible, alphabet, feasible, ⟨by simp [H], rfl⟩, memoryOver, inputs_prover, crossing_empty, acts_ne⟩

/-- A candidate repair: **memory as an alphabet**, at each tick each `X ∈ 𝒳` is in at most `2^{M X}` states as seen
across the runs of `S`, so ticks and leaves count as information. -/
def MemoryAlphabet {Ω L : Type*} (S : Ω → Actual.Run L) (𝒳 : Set (Finset L)) (M : Finset L → ℕ) : Prop :=
  ∀ X ∈ 𝒳, ∀ t, (Set.range fun r => (S r).seen '' ↑(stateOver S X t r)).encard ≤ ((2 ^ M X : ℕ) : ℕ∞)

/-- `S` holds its bit in two states at tick 5, so memory as an alphabet refuses it at `M_X = 0`. -/
theorem not_memoryAlphabet : ¬ MemoryAlphabet S {{false}} (fun _ => 0) := by
  intro h
  have h5 := h {false} rfl 5
  have hst : ∀ b, (3 : Fin 5) ∈ stateOver S {false} 5 b := by
    intro b
    simp only [stateOver, Finset.mem_filter, Finset.mem_univ, true_and]
    refine ⟨by simp, fun h => ?_, b, rfl, 3, ?_, rfl⟩
    · have := ((mem_region _).1 h).2.2
      cases b <;> simp at this
    refine (mem_into _).2 ⟨fun h => ?_, 4, ?_, by simp⟩
    · have := ((mem_region _).1 h).2.2
      cases b <;> simp at this
    · exact (mem_region _).2 ⟨rfl, by simp, by simp⟩
  have hne : (S false).seen '' ↑(stateOver S {false} 5 false) ≠ (S true).seen '' ↑(stateOver S {false} 5 true) := by
    intro he
    have : (S false).seen 3 ∈ (S true).seen '' ↑(stateOver S {false} 5 true) := he ▸ ⟨3, hst false, rfl⟩
    obtain ⟨a, -, ha⟩ := this
    have := seen_tick true a (by rw [ha]; rfl) (by rw [ha]; rfl)
    rw [ha] at this
    simp [seen_eq] at this
  have h2 : (2 : ℕ∞) ≤ (Set.range fun r => (S r).seen '' ↑(stateOver S {false} 5 r)).encard := by
    rw [show (2 : ℕ∞) = ({(S false).seen '' ↑(stateOver S {false} 5 false),
      (S true).seen '' ↑(stateOver S {false} 5 true)} : Set _).encard by rw [Set.encard_pair hne]]
    refine Set.encard_le_encard ?_
    rintro _ (rfl | rfl)
    exacts [⟨false, rfl⟩, ⟨true, rfl⟩]
  have := h2.trans h5
  norm_num at this

end Accounting.SecurityProofs.Actual.Ticks
