import Proofs.Accounting.Onsite.EndToEnd

/-!
# The onsite guarantees' honest and nontrivial lemmas

A hand-built instance at small numbers, not the onsite H200 reference's. The reference circuit `refCircuit` is a weight
(gate 0) and an activation (gate 1), one bit each, and their product (gate 2, an AND), its output; its threading
`refThreading` has one thread, gate 1's, and gate 2 its one counted multiplication, the weight used once. `refRun`
places it on one leaf, the inputs at tick 0 and the product at tick 1. At `onsiteAt b_w` (one server, `ω = 1`,
`Γ = 1`, `R = 2`, no link budget, `b_a = 1`, `(d, e) = (1, 0)`), `K = 2`, `Λ(K) = 1/(2 b_w)` and `t_min = b_w N` for `N`
counted multiplications (`tMin_onsiteAt`). The honest parameters are `b_w = 1`, the weight's width, where the weights'
incompressibility is tight: the product's subcircuit reads 2 novel bits and is charged `b_w + b_a = 2`.

The period `r₀` over `refCircuit` is one run (`G₀`, no coin) whose every subprotocol and its one wipe accept, at
certificates every run meets (`c₀`), so each claim holds and each error is `0`.

* `OnsiteTimeLowerBound_honest`: its hypotheses hold at `refRun` at horizon 2, with `M = N = 1`, and `t_min = 1 > 0`.
* `OnsiteTimeLowerBound_nontrivial`: without `WeightsIncompressible` it is false, at `b_w = 100` (above the weight's
  width), where `t_min = 100 > 2`.
* `OnsiteEndToEnd_honest`: its hypotheses hold at the honest period, where every subprotocol and the wipe accept and the
  run succeeds with probability 1, at horizon 2 above `t_min = 1`.
* `OnsiteEndToEnd_nontrivial`: at a verifier that accepts everything, without `IntegrityContains` it is false at
  horizon 0, and without coverage (claiming `N = 3` counted multiplications of a run that does 1) or without
  `WeightsIncompressible` (at `b_w = 100`) at horizon 2.
-/

namespace Accounting.SecurityProofs.OnsiteWitness

open Verity Verity.Game Finset

/-- The witnesses' reference circuit: a weight (gate 0) and an activation (gate 1), one bit each, and their product
(gate 2), its output. -/
abbrev refCircuit : Circuit where
  N := 3
  w := fun _ => 1
  isInput g := decide (g.val < 2)
  reads g := if g = 2 then {0, 1} else ∅
  reads_lt g a ha := by
    fin_cases g <;> simp at ha
    rcases ha with rfl | rfl <;> decide
  input_reads g hg := by
    fin_cases g <;> simp_all
  op g v := if g = 2 then (v 0 &&& v 1 : BitVec 1) else 0
  op_congr g v v' hv := by
    by_cases hg : g = 2
    · subst hg
      simp [hv 0 (by simp), hv 1 (by simp)]
    · simp [hg]
  outputs := {2}

theorem mem_reads {g a : Fin refCircuit.N} (h : a ∈ refCircuit.reads g) : g = 2 ∧ (a = 0 ∨ a = 1) := by
  fin_cases g <;> simp at h ⊢
  exact h

theorem refCircuit_computed : refCircuit.computed = {2} := by decide

/-- `refCircuit` with an empty baseline: both inputs are novel. -/
abbrev refDecl : Declaration where
  D := refCircuit
  θ := ∅
  θ_input _ h := absurd h (Finset.notMem_empty _)

/-- The witnesses' threading: one thread, gate 1's; the weight gate 0; gate 2 the one counted multiplication. -/
def refThreading : Nci.Threading refDecl where
  m := 1
  W := {0}
  thread g := if g = 1 then some 0 else none
  thread_W g hg := by simp at hg; subst hg; decide
  M := {2}
  M_computed := by decide
  weight _ := 0
  weight_W _ _ := by simp
  weight_reads g hg := by simp at hg; subst hg; decide
  act _ := 1
  act_reads g hg := by simp at hg; subst hg; decide
  act_thread g hg := by simp at hg; subst hg; decide
  reuse := 1
  reuse_le _ _ := (card_filter_le _ _).trans (by simp)

/-- `refCircuit` placed on one leaf: the inputs at tick 0, the product at tick 1, each gate in its own cell. -/
abbrev refPlaced : Placed Unit where
  toCircuit := refCircuit
  x _ := ()
  t g := if g = 2 then 1 else 0
  cell g := g.val
  cell_inj _ _ h := Fin.ext (congrArg (fun p : Unit × ℕ × ℕ => p.2.1) h)

/-- `refPlaced`'s run: every value 0, the leaf holding the inputs throughout. -/
def refRun : Actual.Run Unit where
  A := refPlaced
  val _ := 0
  faithful g hg := by
    fin_cases g <;> simp_all
  held _ _ := {0, 1}
  holding := by
    refine ⟨fun y t a ha => ?_, fun a ha => ?_, fun g a ha _ s _ _ => ?_⟩
    · simp only [mem_insert, mem_singleton] at ha
      rcases ha with rfl | rfl <;> exact ⟨rfl, Or.inl rfl⟩
    · fin_cases a <;> simp_all
    · rcases (mem_reads ha).2 with rfl | rfl <;> simp

/-- The witnesses' hardware: latency 0, one gate a tick per location, every link of capacity 0. -/
def honestHardware : Hardware Unit Unit where
  units := ∅
  ℓ _ _ := 0
  compute _ := 1
  M _ := 0
  ends _ := ((), ())
  β _ := 0
  ring := ∅
  D := 0
  span := 0
  cert := ∅
  trusted := ∅
  bucket := id
  jitter := 0
  record _ := ∅

/-- The witnesses' onsite parameters at weight price `b_w`: one server, the one leaf; `ω = 1`, `Γ = 1`, `R = 2`, no
link budget, `b_a = 1`, `refCircuit` threaded by `refThreading` at `(d, e) = (1, 0)`. -/
noncomputable def onsiteAt (b_w : ℝ) : Onsite Unit Unit where
  hw := honestHardware
  n := 1
  X _ := {()}
  ω := 1
  Γ := 1
  R := 2
  F := 0
  α := 0
  b_w := b_w
  b_a := 1
  c := refDecl
  T := refThreading
  d := 1
  e := 0

/-- The honest onsite parameters: `onsiteAt` at `b_w = 1`, the weight's width. -/
noncomputable abbrev honestOnsite : Onsite Unit Unit := onsiteAt 1

theorem K_onsiteAt (b_w : ℝ) : (onsiteAt b_w).K = 2 := by
  simp [Onsite.K, onsiteAt]

/-- **The floor at `onsiteAt b_w`**: `Λ(K) = 1/(2 b_w)`, so `t_min = b_w N`. -/
theorem tMin_onsiteAt {b_w : ℝ} (hb : 0 < b_w) (N : ℕ) : (onsiteAt b_w).tMin N = b_w * N := by
  have hΛ : (onsiteAt b_w).Λ = 1 / (2 * b_w) := by
    simp only [Onsite.Λ, Nci.Dampener.Lambda, K_onsiteAt]
    simp only [onsiteAt, refThreading, Nat.cast_one, Nat.cast_zero]
    rw [max_eq_right (by positivity)]
    field_simp
    ring
  simp only [Onsite.tMin, hΛ, K_onsiteAt]
  simp only [onsiteAt, Nat.cast_one]
  field_simp
  ring

/-! ### The onsite class at `refRun` -/

/-- The run's one region: the product, at tick 1. -/
theorem region_eq {X : Finset Unit} (hX : () ∈ X) {I : Set ℕ} (h : 1 ∈ I) : refPlaced.region X I = {2} := by
  ext g
  fin_cases g <;> simp [Placed.region, Circuit.computed, hX, h]

theorem region_eq_empty (X : Finset Unit) {I : Set ℕ} (h : 1 ∉ I) : refPlaced.region X I = ∅ := by
  ext g
  fin_cases g <;> simp [Placed.region, Circuit.computed, h]

theorem width_le_two (T : Finset (Fin refPlaced.N)) (hT : ∀ a ∈ T, a = 0 ∨ a = 1) : refPlaced.width T ≤ 2 := by
  have hsub : T ⊆ {0, 1} := fun a ha => by simpa using hT a ha
  calc refPlaced.width T = T.card := by simp [Circuit.width]
    _ ≤ ({0, 1} : Finset (Fin 3)).card := card_le_card hsub
    _ = 2 := by decide

theorem causal (b_w : ℝ) : refPlaced.Causal (onsiteAt b_w).hw.ℓ := by
  intro g a ha
  obtain ⟨rfl, h⟩ := mem_reads ha
  rcases h with rfl | rfl <;> simp [onsiteAt, honestHardware]

theorem computeRate (b_w : ℝ) :
    refPlaced.ComputeRate (Set.range (onsiteAt b_w).X) (fun _ => 1) (onsiteAt b_w).hw.compute := by
  rintro Y ⟨i, rfl⟩ a b
  by_cases h : 1 ∈ Set.Ico a b
  · rw [region_eq (by simp [onsiteAt]) h]
    simp only [Set.mem_Ico] at h
    simp only [Placed.work, sum_singleton, onsiteAt, honestHardware]
    omega
  · rw [region_eq_empty _ h]
    simp [Placed.work]

theorem links (b_w : ℝ) : refPlaced.Links (onsiteAt b_w).hw.ends (onsiteAt b_w).hw.β fun _ _ => () := by
  refine ⟨fun g a _ hne => absurd rfl hne, fun e y a b => ?_⟩
  have : refPlaced.readOn (fun _ _ => ()) e y (Set.Ico a b) = ∅ := by
    ext g
    simp [Placed.readOn]
  simp [this, Circuit.width]

theorem linkCapacity_eq (b_w : ℝ) (X : Finset Unit) :
    linkCapacity (onsiteAt b_w).hw.ends (onsiteAt b_w).hw.β X = 0 := by
  simp [linkCapacity, onsiteAt, honestHardware]

theorem state_le (b_w : ℝ) (X : Finset Unit) (τ : ℕ) : (refPlaced.state X τ : ℝ) ≤ (onsiteAt b_w).R := by
  have h : refPlaced.state X τ ≤ 2 := by
    unfold Placed.state
    refine width_le_two _ fun a ha => ?_
    simp only [mem_filter, Circuit.into, mem_univ, true_and] at ha
    obtain ⟨⟨_, g, _, hg⟩, _⟩ := ha
    exact (mem_reads hg).2
  have : (refPlaced.state X τ : ℝ) ≤ 2 := by exact_mod_cast h
  simpa [onsiteAt] using this

theorem contains {X : Finset Unit} (hX : () ∈ X) {t : ℕ} (ht : 1 < t)
    (hθ : ∀ a ∈ (∅ : Finset (Fin refPlaced.N)), refPlaced.isInput a = true) :
    refDecl.Contains (refPlaced.declaration X t ∅ hθ) := by
  have hr : refPlaced.region X (Set.Iio t) = {2} := region_eq hX (by simpa using ht)
  refine ⟨⟨id, Function.injective_id, fun _ => rfl, ?_, ?_, ?_, ?_⟩⟩
  · intro g
    fin_cases g <;> simp [Placed.declaration, Placed.cut, hr]
  · intro g hg
    fin_cases g <;> simp_all [Placed.declaration, Placed.cut]
  · intro g hg v
    fin_cases g <;> simp_all [Placed.declaration, Placed.cut]
    rfl
  · intro o ho
    simp only [refDecl, refCircuit, mem_singleton] at ho
    subst ho
    show _ ∈ refPlaced.region X (Set.Iio t)
    rw [hr]
    exact mem_singleton_self _

/-- `refRun` is in the onsite class at `onsiteAt b_w`, at every horizon `t ≥ 2`. -/
theorem onsiteClass (b_w : ℝ) {t : ℕ} (ht : 1 < t)
    (hθ : ∀ a ∈ (∅ : Finset (Fin refPlaced.N)), refPlaced.isInput a = true) :
    OnsiteClass refPlaced (onsiteAt b_w).hw.ℓ (onsiteAt b_w).hw.compute (onsiteAt b_w).hw.ends (onsiteAt b_w).hw.β
      (fun _ _ => ()) (onsiteAt b_w).n (onsiteAt b_w).X ∅ hθ t (onsiteAt b_w).ω (onsiteAt b_w).R (onsiteAt b_w).F
      (onsiteAt b_w).α (onsiteAt b_w).c :=
  ⟨causal b_w, computeRate b_w, links b_w, fun i => by rw [linkCapacity_eq]; simp [onsiteAt],
    fun i k _ => state_le b_w _ _,
    contains (mem_biUnion.2 ⟨⟨0, by simp [onsiteAt]⟩, mem_univ _, by simp [onsiteAt]⟩) ht hθ⟩

/-! ### The parameters -/

theorem threaded (S : ℝ) : refThreading.Threaded 1 S 1 0 := by
  intro R hR t ht
  rcases Finset.subset_singleton_iff.1 (refCircuit_computed ▸ hR.1) with rfl | rfl
  · simp [Nci.Threading.threadMults, Nci.Threading.mults]
  · exfalso
    have : (refThreading.threadReads {2} t).card = 1 := by
      fin_cases t
      decide
    omega

theorem weightsIncompressible (S : ℝ) : Nci.Assumptions.WeightsIncompressible refDecl refThreading 1 S 1 1 := by
  intro R hR
  rcases Finset.subset_singleton_iff.1 (refCircuit_computed ▸ hR.1) with rfl | rfl
  · simp [Nci.Threading.weightsUsed, Nci.Threading.mults, Nci.Threading.threadReads, Circuit.into]
  · have h1 : (refThreading.weightsUsed {2}).card = 1 := by decide
    have h2 : ∑ t, (refThreading.threadReads {2} t).card = 1 := by decide
    have h3 : Nci.novelReads refDecl {2} = 2 := by decide
    rw [h1, h2, h3]
    norm_num

theorem valid {b_w : ℝ} (hb : 0 < b_w) : (onsiteAt b_w).Valid :=
  ⟨by simp [onsiteAt], fun i j hij => absurd (Fin.ext (by
      have hi := i.2
      have hj := j.2
      simp only [onsiteAt] at hi hj
      omega)) hij, hb, by simp [onsiteAt],
    by simp [onsiteAt], fun i => by simp [onsiteAt, honestHardware], threaded _⟩

theorem weightsIncompressible_honest :
    Nci.Assumptions.WeightsIncompressible honestOnsite.c honestOnsite.T honestOnsite.Γ honestOnsite.K
      honestOnsite.b_w honestOnsite.b_a :=
  weightsIncompressible _

theorem card_M (b_w : ℝ) : (onsiteAt b_w).T.M.card = 1 := by
  simp [onsiteAt, refThreading]

theorem mem_X (b_w : ℝ) : () ∈ univ.biUnion (onsiteAt b_w).X :=
  mem_biUnion.2 ⟨⟨0, by simp [onsiteAt]⟩, mem_univ _, by simp [onsiteAt]⟩

/-! ### The period -/

/-- The witnesses' device: no resource. -/
def d₀ : Device Unit := ⟨0, 0, fun _ => 0, 0, fun _ => 0⟩

/-- A step that charges nothing. -/
def step₀ : Step Unit Unit := ⟨fun _ _ => 0, fun _ => 0, fun _ _ => 0, fun _ => 0, fun _ => 0⟩

/-- `refCircuit`'s computed gate in one unit. -/
def parts₀ : Verity.Partition refCircuit 1 where
  unit g := if refCircuit.isInput g then none else some 0
  input_iff g := by cases h : refCircuit.isInput g <;> simp

/-- The period: one unit, no opening shown, the schedule idle, and every subprotocol accepting. -/
def r₀ : PeriodLive d₀ Unit refCircuit Unit Unit where
  units := 1
  parts := parts₀
  acct _ := ()
  win := ⟨0, 2⟩
  commit _ := ()
  shows _ _ _ := False
  schedule _ _ := step₀
  fits _ _ := by simp [Step.Fits, step₀, d₀]
  commWithin _ _ := True
  accepts _ _ := True

/-- Certificates every run meets: an integrity share of 1, and no work, space, reads or communication bound. -/
def c₀ : Certificates := ⟨1, 0, 0, 0, 0⟩

/-- The price of nothing. -/
def p₀ : Price := fun _ => 0

/-- The one wipe: accepted, over no fill, nothing kept. -/
def w₀ : WipeLive d₀ Unit := ⟨0, le_rfl, fun _ => True, fun _ => 0⟩

/-- The game: one run, no coin. -/
def G₀ : Game Unit := .ret ()

theorem prob_G₀_eq_one {p : Unit → Prop} (s : Strategy G₀) (h : p ()) : prob p G₀ s = 1 := by
  simp [G₀, prob, h]

theorem holds₀ (s : Strategy G₀) (a : Audit) (o : Unit) : r₀.Holds c₀ p₀ G₀ s a o := by
  cases a
  · simp only [PeriodLive.Holds, IntegrityHolds]
    intro X _
    have h : (unbound r₀.parts (r₀.opens G₀ s o) X).card ≤ r₀.units := (card_le_univ _).trans_eq (Fintype.card_fin _)
    simp only [c₀, one_mul]
    exact_mod_cast h
  · simp [PeriodLive.Holds, c₀]
  · simp [PeriodLive.Holds, c₀]
  · simp [PeriodLive.Holds, c₀]
  · exact trivial

theorem sound₀ (a : Audit) : r₀.Sound G₀ c₀ p₀ a 0 := fun s =>
  (prob_mono (q := fun _ => False) (fun o h => h.2.1 (holds₀ s a o)) G₀ s).trans
    (by rw [prob_false]; exact zero_le)

theorem noBreak₀ (s : Strategy G₀) (o : Unit) : ¬ r₀.Break G₀ s o :=
  fun ⟨_, _, _, ⟨_, _, _, h⟩, _⟩ => h

theorem survives₀ (o : Unit) : w₀.Survives 0 o := by
  simp only [WipeLive.Survives, w₀, d₀]
  norm_num

theorem wipes₀ (s : Strategy G₀) :
    prob (fun o => w₀.accepts o ∧ ¬ w₀.Survives 0 o) G₀ s ≤ ENNReal.ofReal 0 :=
  (prob_mono (q := fun _ => False) (fun o h => h.2 (survives₀ o)) G₀ s).trans (by rw [prob_false]; exact zero_le)

/-- The world: every run's actual circuit is `refRun`'s, its wires on the one link, with an empty baseline. -/
abbrev S₀ : Unit → Actual.Run Unit := fun _ => refRun

abbrev link₀ : ∀ o, Fin (S₀ o).A.N → Fin (S₀ o).A.N → Unit := fun _ _ _ => ()

abbrev θ₀ : ∀ o, Finset (Fin (S₀ o).A.N) := fun _ => ∅

theorem hθ₀ : ∀ o, ∀ a ∈ θ₀ o, (S₀ o).A.isInput a = true := fun _ a ha => absurd ha (notMem_empty a)

theorem certifierLinks₀ (b_w : ℝ) :
    CertifierLinks S₀ (onsiteAt b_w).hw.ends (onsiteAt b_w).hw.β link₀ (onsiteAt b_w).n (onsiteAt b_w).X
      (onsiteAt b_w).α (onsiteAt b_w).F :=
  fun _ => ⟨links b_w, fun i => by rw [linkCapacity_eq]; simp [onsiteAt]⟩

theorem wipeState₀ (b_w : ℝ) (t : ℕ) :
    Assumptions.WipeState S₀ (fun _ : Fin 1 => w₀) 0 (onsiteAt b_w).X t (onsiteAt b_w).ω (onsiteAt b_w).R :=
  fun _ _ _ _ _ => state_le b_w _ _

theorem integrityContains₀ (b_w : ℝ) {t : ℕ} (ht : 1 < t) (s : Strategy G₀) :
    Assumptions.IntegrityContains r₀ G₀ s c₀ p₀ S₀ (onsiteAt b_w).X t θ₀ hθ₀ (onsiteAt b_w).c :=
  fun _ _ _ => contains (mem_X b_w) ht _

theorem coverage₀ (InClass success : Unit → Prop) : Assumptions.Coverage G₀ InClass success (fun _ => 1) 1 0 :=
  fun s => (prob_mono (q := fun _ => False) (fun _ h => absurd h.2.2 (lt_irrefl 1)) G₀ s).trans
    (by rw [prob_false]; exact zero_le)

/-- At `r₀` and `onsiteAt b_w`, a run that succeeds below the floor is accepted with no break with probability 1. -/
theorem cheap_prob (b_w : ℝ) (t N : ℕ) (ht : (t : ℝ) < (onsiteAt b_w).tMin N) (s : Strategy G₀) :
    prob (fun o => ((∀ a, r₀.accepts a o) ∧ ∀ k : Fin 1, ((fun _ : Fin 1 => w₀) k).accepts o) ∧
      (fun _ : Unit => True) o ∧ (t : ℝ) < (onsiteAt b_w).tMin N ∧ ¬ r₀.Break G₀ s o) G₀ s = 1 :=
  prob_G₀_eq_one s ⟨⟨fun _ => trivial, fun _ => trivial⟩, trivial, ht, noBreak₀ s ()⟩

end Accounting.SecurityProofs.OnsiteWitness

namespace Accounting.SecurityProofs

open Verity Verity.Game Finset OnsiteWitness

/-- **`OnsiteTimeLowerBound`'s honest witness**: `refRun`'s circuit at horizon 2 is in the onsite class at the honest
parameters, whose weights are incompressible, with `M = N = 1` and the floor `t_min = 1 > 0`. -/
theorem OnsiteTimeLowerBound_honest : Accounting.Guarantees.OnsiteTimeLowerBound_honest :=
  ⟨Unit, Unit, inferInstance, inferInstance, honestOnsite, refPlaced, fun _ _ => (), ∅,
    fun a ha => absurd ha (notMem_empty a), 2, 1, 1, valid one_pos, onsiteClass 1 (by norm_num) _,
    weightsIncompressible_honest, (card_M 1).ge, le_rfl, by rw [tMin_onsiteAt one_pos]; norm_num⟩

/-- **`OnsiteTimeLowerBound`'s nontrivial witness**: without `WeightsIncompressible` it is false at `b_w = 100`, a weight
price above the weight's width, where `refRun`'s circuit is in the class at horizon 2 and the floor is 100 ticks. -/
theorem OnsiteTimeLowerBound_nontrivial : Accounting.Guarantees.OnsiteTimeLowerBound_nontrivial := fun h => by
  have := (h (onsiteAt 100) refPlaced (fun _ _ => ()) ∅ (fun a ha => absurd ha (notMem_empty a)) 2 1 1
    (valid (by norm_num)) (onsiteClass 100 (by norm_num) _) (card_M 100).ge le_rfl).1
  rw [tMin_onsiteAt (by norm_num)] at this
  norm_num at this

/-- **`OnsiteEndToEnd`'s honest witness**: at the honest parameters and the period `r₀`, whose every subprotocol and
wipe accepts and whose run succeeds with probability 1 at every strategy, every hypothesis holds at errors `0`, at
horizon 2 above the floor `t_min = 1` of `N = 1` counted multiplications. -/
theorem OnsiteEndToEnd_honest : Accounting.Guarantees.OnsiteEndToEnd_honest :=
  ⟨Unit, Unit, inferInstance, d₀, refCircuit, Unit, Unit, r₀, G₀, c₀, p₀, 1, fun _ => w₀, 0, Unit, Unit,
    inferInstance, inferInstance, honestOnsite, S₀, link₀, θ₀, hθ₀, 2, fun _ => 1, fun _ => True, 1, valid one_pos,
    weightsIncompressible_honest, fun _ => (card_M 1).ge, sound₀,
    fun _ s => le_antisymm ((wipes₀ s).trans ENNReal.ofReal_zero.le) bot_le, fun _ => causal 1,
    fun _ => computeRate 1, certifierLinks₀ 1, wipeState₀ 1 2, coverage₀ _ _, integrityContains₀ 1 (by norm_num),
    fun s => prob_G₀_eq_one s ⟨⟨fun _ => trivial, fun _ => trivial⟩, trivial, noBreak₀ s ()⟩,
    by rw [tMin_onsiteAt one_pos]; norm_num⟩

/-- **`OnsiteEndToEnd`'s nontrivial witness**, at `r₀`, which accepts every run, and its one wipe, which does too:
without `IntegrityContains`, a run at horizon 0 at the honest parameters succeeds below the floor of one counted
multiplication; without coverage, a run that does 1 counted multiplication, claimed to need `N = 3`, succeeds at
horizon 2 below the floor of 3 ticks; without `WeightsIncompressible`, at `b_w = 100` a run at horizon 2 succeeds below
the floor of 100 ticks. Each with probability 1, while every other hypothesis holds at errors `0`. -/
theorem OnsiteEndToEnd_nontrivial : Accounting.Guarantees.OnsiteEndToEnd_nontrivial := by
  refine ⟨fun h => ?_, fun h => ?_, fun h => ?_⟩
  · have := h d₀ refCircuit Unit Unit r₀ G₀ c₀ p₀ (fun _ => 0) 1 (fun _ => w₀) 0 (fun _ => 0) honestOnsite S₀ link₀
      θ₀ hθ₀ 0 (fun _ => 1) (fun _ => True) 1 0 (fun _ _ => trivial) (fun _ _ => trivial) (valid one_pos)
      weightsIncompressible_honest (fun _ => (card_M 1).ge) sound₀ (fun _ => wipes₀) (fun _ => causal 1)
      (fun _ => computeRate 1) (certifierLinks₀ 1) (wipeState₀ 1 0) (coverage₀ _ _) PUnit.unit
    simp only [ENNReal.ofReal_zero, Finset.sum_const_zero, add_zero, nonpos_iff_eq_zero] at this
    exact one_ne_zero
      ((cheap_prob 1 0 1 (by rw [tMin_onsiteAt one_pos]; norm_num) PUnit.unit).symm.trans this)
  · have := h d₀ refCircuit Unit Unit r₀ G₀ c₀ p₀ (fun _ => 0) 1 (fun _ => w₀) 0 (fun _ => 0) honestOnsite S₀ link₀
      θ₀ hθ₀ 2 (fun _ => 1) (fun _ => True) 3 0 (fun _ _ => trivial) (fun _ _ => trivial) (valid one_pos)
      weightsIncompressible_honest (fun _ => (card_M 1).ge) sound₀ (fun _ => wipes₀) (fun _ => causal 1)
      (fun _ => computeRate 1) (certifierLinks₀ 1) (wipeState₀ 1 2) PUnit.unit (integrityContains₀ 1 (by norm_num) _)
    simp only [ENNReal.ofReal_zero, Finset.sum_const_zero, add_zero, nonpos_iff_eq_zero] at this
    exact one_ne_zero
      ((cheap_prob 1 2 3 (by rw [tMin_onsiteAt one_pos]; norm_num) PUnit.unit).symm.trans this)
  · have := h d₀ refCircuit Unit Unit r₀ G₀ c₀ p₀ (fun _ => 0) 1 (fun _ => w₀) 0 (fun _ => 0) (onsiteAt 100) S₀
      link₀ θ₀ hθ₀ 2 (fun _ => 1) (fun _ => True) 1 0 (fun _ _ => trivial) (fun _ _ => trivial) (valid (by norm_num))
      (fun _ => (card_M 100).ge) sound₀ (fun _ => wipes₀) (fun _ => causal 100) (fun _ => computeRate 100)
      (certifierLinks₀ 100) (wipeState₀ 100 2) (coverage₀ _ _) PUnit.unit (integrityContains₀ 100 (by norm_num) _)
    simp only [ENNReal.ofReal_zero, Finset.sum_const_zero, add_zero, nonpos_iff_eq_zero] at this
    exact one_ne_zero
      ((cheap_prob 100 2 1 (by rw [tMin_onsiteAt (by norm_num)]; norm_num) PUnit.unit).symm.trans this)

end Accounting.SecurityProofs
