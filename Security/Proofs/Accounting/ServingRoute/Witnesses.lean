import Proofs.Accounting.ServingRoute
import Proofs.Accounting.ServingBlocks.Witnesses
import Definitions.Accounting.Guarantees.ServingRoute.Witnesses

/-!
# The serving route: honest and nontrivial lemmas

The proofs of `Security/Definitions/Accounting/Guarantees/ServingRoute/Witnesses.lean`:

* `Gate`: `Accounting.ServingBlocks.Counter`'s declaration (one gate reading a registered input, computing one operation
  of weight 1), its gate a product of one activation, with three decoders: `echo` (outputs its program), `chatty`
  (prints `[true]` on every program) and `mute` (outputs nothing).
* `Serve`: one constant gate placed on one server at tick 0, its one operation of weight 1, and the serving parameters
  of one server with compute `1`, windows of one tick, no state at a wipe, no links and no dark leaves, at
  `(c_S, k_S) = (1, 0)` and `q = 1`, so `q_S = 1` and `windowsMin N = N`. The play `served` (the gate's one unit a
  serving block) is in the serving class, and `idle` (no serving block, the gate's unit left in the dark remainder at no
  dark leaf) is not.

The `example`s at the end apply each guarantee to its lemmas, so the lemmas are the premises as the guarantee states
them.
-/

namespace Accounting.SecurityProofs.ServingRoute

open Verity Finset Accounting.ServingBlocks Accounting.ServingRoute
open scoped ENNReal

namespace Gate

/-- `Counter`'s gate is a product of one activation. -/
abbrev M : Nci.Ldl.Products Counter.d.D Unit Unit where
  J := {()}
  rep _ := 1
  rep_inj _ _ _ _ _ := Subsingleton.elim _ _
  rep_op _ _ := by decide
  act _ := ()

/-- The decoder that outputs its program. -/
abbrev echo : Nci.Ldl.Decoder := fun p _ => some p

/-- The decoder that prints `[true]` on every program, the empty one included. -/
abbrev chatty : Nci.Ldl.Decoder := fun _ _ => some [true]

/-- The decoder that outputs nothing. -/
abbrev mute : Nci.Ldl.Decoder := fun _ _ => none

abbrev y : Finset (Fin Counter.d.D.N) → Counter.d.D.Transcript → List Bool := fun _ _ => [true]

abbrev inp : Counter.d.D.Transcript := fun _ => 0

theorem boolean : Counter.d.D.Boolean := ⟨by decide, by decide⟩

theorem reads_zero : ∀ g ∈ Counter.d.D.computed, (Counter.d.unregistered {g}).card ≤ 0 := by decide

theorem reps : Counter.J.V ⊆ M.J.image M.rep := by decide

theorem fanOut : M.FanOut 1 := fun _ => (card_le_univ _).trans (by simp)

theorem N_lt : Counter.d.D.N < 2 ^ 2 := by decide

theorem acts_le (n : ℕ) (P : Partition Counter.d.D n) (u : Fin n) : ((M.acts P u).card : ℝ) ≤ 1 := by
  exact_mod_cast (card_le_univ _).trans (by simp)

/-- At a cap below the operation's weight, the gate alone is a block over it. -/
theorem not_cap (q : ℕ → ℕ → ℝ) (c k : ℕ) (hc : 1 ≤ c) (hq : q c k < 1) : ¬ PerBlockCap Counter.d Counter.J q c k :=
  fun h => by
    have h1 := h {1} (by decide) ((show ({1} : Finset (Fin 2)).card ≤ 1 by decide).trans hc)
      ((show (Counter.d.unregistered {1}).card ≤ 0 by decide).trans (Nat.zero_le k))
    rw [SecurityProofs.ServingBlocks.Counted.inside_eq Counter.J SecurityProofs.ServingBlocks.Counter.ω_one] at h1
    have h2 : (Counter.J.V.filter fun v => Counter.J.body v ⊆ {1}).card = 1 := by decide
    rw [h2, Nat.cast_one] at h1
    linarith

theorem evaluates_echo (n : ℕ) (R : Nci.Ldl.Registration Counter.d n) (t : ℕ) :
    Nci.Ldl.Assumptions.EvaluatesBlocks echo t R (fun u => y (R.P.gates u)) 1 :=
  ⟨fun _ _ => [true], fun _ _ _ => ⟨fun _ _ _ => rfl, by simp, rfl⟩⟩

theorem evaluates_chatty (n : ℕ) (R : Nci.Ldl.Registration Counter.d n) (t : ℕ) :
    Nci.Ldl.Assumptions.EvaluatesBlocks chatty t R (fun u => y (R.P.gates u)) 0 :=
  ⟨fun _ _ => [], fun _ _ _ => ⟨fun _ _ _ => rfl, by simp, rfl⟩⟩

theorem rate_echo (n : ℕ) (R : Nci.Ldl.Registration Counter.d n) (t : ℕ) (X : Counter.d.D.Transcript) :
    Nci.Ldl.Assumptions.ActivationRate 1 0 t echo R M (fun u => y (R.P.gates u)) X := fun u => by
  have h1 : (1 : ℕ∞) ≤ Nci.Ldl.LDL echo t [true] :=
    le_iInf₂ fun p hp => by cases Option.some_injective _ hp; simp
  calc ENNReal.ofReal (1 * (M.acts R.P u).card - 0) ≤ 1 := by
        rw [one_mul, sub_zero]; exact ENNReal.ofReal_le_one.2 (acts_le n R.P u)
    _ ≤ _ := by simpa using ENat.toENNReal_le.2 h1

theorem rate_mute (n : ℕ) (R : Nci.Ldl.Registration Counter.d n) (t : ℕ) (X : Counter.d.D.Transcript) (β Λ : ℝ) :
    Nci.Ldl.Assumptions.ActivationRate β Λ t mute R M (fun u => y (R.P.gates u)) X := fun u => by
  have : Nci.Ldl.LDL mute t (y (R.P.gates u) X) = ⊤ := by simp [Nci.Ldl.LDL]
  rw [this, ENat.toENNReal_top]
  exact le_top

end Gate

theorem ServingPerBlockCap_honest : Guarantees.ServingRoute.ServingPerBlockCap_honest :=
  ⟨Counter.d, Counter.J, Gate.M, 1, fun _ _ => Gate.echo, 0, 1, 2, Gate.y, 1, 0, Gate.inp, 1, 0, fun _ _ => 3,
    ServingBlocks.Counter.total_eq, Gate.boolean, fun _ _ => le_rfl, Gate.reps, Gate.fanOut, le_rfl, Gate.reads_zero,
    Gate.N_lt, one_pos, fun n R _ _ _ => Gate.evaluates_echo n R 0, fun n R _ _ _ => Gate.rate_echo n R 0 _,
    by norm_num⟩

theorem ServingPerBlockCap_nontrivial : Guarantees.ServingRoute.ServingPerBlockCap_nontrivial := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · -- `chatty` evaluates every block from no bits, so its activations cost nothing: only `ActivationRate` excludes it
    exact Gate.not_cap (fun _ _ => 1 / 2) 1 0 le_rfl (by norm_num) (h Counter.d Counter.J Gate.M 1
      (fun _ _ => Gate.chatty) 0 0 2 Gate.y 4 0 Gate.inp 1 0 (fun _ _ => 1 / 2) Gate.boolean (fun _ _ => le_rfl)
      Gate.reps Gate.fanOut le_rfl Gate.reads_zero Gate.N_lt (by norm_num) (fun n R _ _ _ => Gate.evaluates_chatty n R 0)
      (by norm_num))
  · -- under `mute` every activation rate holds, as nothing is described: only `EvaluatesBlocks` excludes it
    exact Gate.not_cap (fun _ _ => 1 / 2) 1 0 le_rfl (by norm_num) (h Counter.d Counter.J Gate.M 1
      (fun _ _ => Gate.mute) 0 0 2 Gate.y 4 0 Gate.inp 1 0 (fun _ _ => 1 / 2) Gate.boolean (fun _ _ => le_rfl)
      Gate.reps Gate.fanOut le_rfl Gate.reads_zero Gate.N_lt (by norm_num) (fun n R _ _ _ => Gate.rate_mute n R 0 _ 4 0)
      (by norm_num))

namespace Serve

/-- One computed gate, a constant. -/
abbrev C : Circuit where
  N := 1
  w _ := 1
  isInput _ := false
  reads _ := ∅
  reads_lt _ _ h := absurd h (Finset.notMem_empty _)
  input_reads _ h := absurd h (by decide)
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {0}

/-- The gate, placed on the one leaf `()` at tick 0. -/
abbrev A : Placed Unit where
  toCircuit := C
  x _ := ()
  t _ := 0
  cell _ := 0
  cell_inj a b _ := Subsingleton.elim a b

/-- The gate's operation, of weight 1. -/
abbrev J : Counted A.toCircuit where
  V := {0}
  body v := {v}
  ω _ := 1
  V_computed := by decide
  rep_mem _ _ := Finset.mem_singleton_self _
  ω_nonneg _ _ := zero_le_one

/-- The gate in the one unit. -/
abbrev part : Partition A.toCircuit 1 := ⟨fun _ => some 0, by decide⟩

theorem into_eq (S : Finset (Fin A.N)) : A.into S = ∅ := by
  ext a
  simp [Circuit.into]

/-- One server, leaf `()`, with compute `1` a tick, and one link of capacity `0`. -/
abbrev hw : Hardware Unit Unit where
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
  bucket _ := 0
  jitter := 0
  record _ := ∅

/-- Windows of one tick, no state at a wipe, no link capacity, no dark leaves, at `(c_S, k_S) = (1, 0)` and `q = 1`. -/
abbrev P : Params Unit Unit Unit where
  hw := hw
  X _ := {()}
  dark := ∅
  ω := 1
  R := 0
  F := 0
  nDark := 0
  cS := 1
  q _ _ := 1
  μ := 1

/-- The gate's unit is a serving block. -/
abbrev served : Play Unit := ⟨A, ∅, fun _ h => absurd h (Finset.notMem_empty _), J, 1, part, {0}⟩

/-- No serving block: the gate's unit is left in the dark remainder, at no dark leaf. -/
abbrev idle : Play Unit := ⟨A, ∅, fun _ h => absurd h (Finset.notMem_empty _), J, 1, part, ∅⟩

theorem qS_eq : P.qS = 1 := by simp [Params.qS, Params.kS]

theorem windowsMin_eq (N : ℝ) : P.windowsMin N = N := by simp [Params.windowsMin, qS_eq]

theorem hcS : ∀ s, P.hw.compute (P.X s) * P.ω ≤ P.cS := fun _ => le_rfl

theorem served_windows : served.windows = 1 := rfl

theorem idle_windows : idle.windows = 0 := rfl

theorem total_eq : J.total = 1 := by simp [Counted.total]

theorem served_class : P.Class served := by
  refine ⟨fun _ _ h => absurd h (Finset.notMem_empty _), ?_, ⟨fun _ _ => (), fun _ _ h => absurd h (Finset.notMem_empty _),
    fun e y a b => ?_⟩, fun _ => by simp [linkCapacity], fun _ _ => ⟨(), 0, ?_, ?_⟩,
    fun u hu => absurd (Finset.mem_singleton.2 (Subsingleton.elim u 0)) hu, by simp [Placed.region],
    fun I _ hc _ => ?_, by unfold OperationBodies; decide, by unfold Overlap; decide, fun _ => le_rfl,
    fun _ _ => le_rfl⟩
  · rintro _ ⟨s, rfl⟩ a b
    have hw1 : served.A.work (fun _ => 1) (served.A.region (P.X s) (Set.Ico a b)) =
        (served.A.region (P.X s) (Set.Ico a b)).card := by simp [Placed.work]
    rw [hw1, one_mul]
    rcases Nat.lt_or_ge a b with hab | hab
    · exact (card_le_univ _).trans (by simp; omega)
    · rw [Finset.card_eq_zero.2 (Finset.eq_empty_of_forall_notMem fun g hg => ?_)]
      · exact Nat.zero_le _
      rw [SecurityProofs.Actual.mem_region] at hg
      have := hg.2.2
      simp only [Set.mem_Ico] at this
      omega
  · have : served.A.readOn (fun _ _ => ()) e y (Set.Ico a b) = ∅ := by
      ext g
      simp [Placed.readOn]
    simp [this, Circuit.width]
  · simp only [Placed.state, Circuit.width]
    rw [into_eq]
    simp
  · ext g
    rw [Partition.mem_gates, SecurityProofs.Actual.mem_region]
    exact ⟨fun _ => ⟨rfl, Finset.mem_singleton.2 rfl, Set.mem_Ico.2 ⟨le_rfl, Nat.zero_lt_one⟩⟩,
      fun _ => congrArg some (Subsingleton.elim _ _)⟩
  · exact (SecurityProofs.ServingBlocks.Counted.inside_le_card J (fun _ _ => le_rfl) I).trans
      (show (I.card : ℝ) ≤ 1 by exact_mod_cast hc)

theorem idle_not_class : ¬ P.Class idle := fun h => by
  have := h.2.2.2.2.2.1 0 (Finset.notMem_empty 0) ((Partition.mem_gates idle.P (u := 0) (g := 0)).2 rfl)
  rw [SecurityProofs.Actual.mem_region] at this
  simp at this

end Serve

theorem ServingEndToEnd_honest : Guarantees.ServingRoute.ServingEndToEnd_honest :=
  ⟨Game.ret (), Serve.P, fun _ _ => Serve.served, fun _ _ => True, fun _ _ => True, fun _ _ => True, 1, Serve.hcS,
    by rw [Serve.qS_eq]; norm_num, by rw [Serve.windowsMin_eq]; norm_num, fun _ => by simp [Game.prob],
    fun _ => by simp [Game.prob], fun _ => by simp [Game.prob, Serve.served_class],
    fun _ => by simp [Game.prob, Serve.total_eq]⟩

theorem ServingEndToEnd_nontrivial : Guarantees.ServingRoute.ServingEndToEnd_nontrivial := by
  have hq : 0 < Serve.P.qS := by rw [Serve.qS_eq]; norm_num
  refine ⟨fun h => ?_, fun h => ?_, fun h => ?_⟩
  · -- the claims fail on every run, which serves nothing: only the joint guarantee excludes it
    have := h Unit Unit Unit Unit (Game.ret ()) Serve.P (fun _ _ => Serve.idle) (fun _ _ => True) (fun _ _ => False)
      (fun _ _ => True) 1 (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ _ => trivial) Serve.hcS hq
      (fun _ => by simp [Game.prob]) (fun _ => by simp [Game.prob, Serve.idle_not_class]) PUnit.unit
    simp [Game.prob, Serve.idle_windows, Serve.windowsMin_eq] at this
  · -- the claims hold and nothing is served, so the run is outside the class: only the bridge excludes it
    have := h Unit Unit Unit Unit (Game.ret ()) Serve.P (fun _ _ => Serve.idle) (fun _ _ => True) (fun _ _ => True)
      (fun _ _ => True) 1 (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ _ => trivial) Serve.hcS hq
      (fun _ => by simp [Game.prob]) (fun _ => by simp [Game.prob, Serve.idle_not_class]) PUnit.unit
    simp [Game.prob, Serve.idle_windows, Serve.windowsMin_eq] at this
  · -- the run is in the class and counts `1 < N = 2`: only coverage excludes it
    have := h Unit Unit Unit Unit (Game.ret ()) Serve.P (fun _ _ => Serve.served) (fun _ _ => True) (fun _ _ => True)
      (fun _ _ => True) 2 (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ _ => trivial) Serve.hcS hq
      (fun _ => by simp [Game.prob]) (fun _ => by simp [Game.prob, Serve.served_class]) PUnit.unit
    simp [Game.prob, Serve.served_windows, Serve.windowsMin_eq] at this

/-! Each lemma is its guarantee's premises as the guarantee states them: the guarantee applies at the honest instance. -/

example : PerBlockCap Counter.d Counter.J (fun _ _ => 3) 1 0 :=
  ServingPerBlockCap Counter.d Counter.J Gate.M 1 (fun _ _ => Gate.echo) 0 1 2 Gate.y 1 0 Gate.inp 1 0 (fun _ _ => 3)
    Gate.boolean (fun _ _ => le_rfl) Gate.reps Gate.fanOut le_rfl Gate.reads_zero Gate.N_lt one_pos
    (fun n R _ _ _ => Gate.evaluates_echo n R 0) (fun n R _ _ _ => Gate.rate_echo n R 0 _) (by norm_num)

example : Game.prob (fun _ => True ∧ True ∧ ((Serve.served.windows : ℕ) : ℝ) < Serve.P.windowsMin 1) (Game.ret ())
    PUnit.unit ≤ 0 + 0 + 0 :=
  ServingEndToEnd (Game.ret ()) Serve.P (fun _ _ => Serve.served) (fun _ _ => True) (fun _ _ => True)
    (fun _ _ => True) 1 (fun _ => 0) (fun _ => 0) (fun _ => 0) Serve.hcS (by rw [Serve.qS_eq]; norm_num)
    (fun _ => by simp [Game.prob]) (fun _ => by simp [Game.prob, Serve.served_class])
    (fun _ => by simp [Game.prob, Serve.total_eq]) PUnit.unit

end Accounting.SecurityProofs.ServingRoute
