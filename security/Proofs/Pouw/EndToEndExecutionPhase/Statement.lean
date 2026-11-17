module

public import Definitions.Pouw.Game.Params
public import Definitions.Pouw.Assumptions.Game
public import Definitions.Pouw.Assumptions.NCP.Assumptions
public import Proofs.Flock.Soundness.Audit.Extraction
public import Proofs.Flock.Soundness.Audit.RegDraw
public import Proofs.Flock.Soundness.Audit.Window

/-! PoUW's work bound at the window audit, a statement with no proof; `TileWork` and `StraightLineExtraction` are
open named premises. -/

@[expose] public section

namespace Pouw.WindowWork

open Finset FlockSoundness.Audit
open scoped ENNReal

variable {Q R S : Type}

noncomputable def incorrectWork (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Pouw.Trace) : ℝ :=
  ∑ u ∈ range U.layout.N \ correctSet P U H s τ, (P.Wref U.layout u : ℝ)

/-- Every unit is one output coordinate, shape `(1, k, 1)`: a `K`-chunked s8 dot with s32 running sums. -/
def coordinates (L : Layout) : Prop := ∀ u < L.N, (L.shape u).m = 1 ∧ (L.shape u).n = 1

/-- Coordinate units, each a dot of positive length `k` (a whole dot, or a `K`-chunk of 2048 committing its s32
running sum). -/
def coordDomain (L : Layout) : Prop := coordinates L ∧ ∀ u < L.N, 0 < (L.shape u).k

/-- The work step at coordinate units (a named premise, open), summed over the window because a cost model prices
a whole run. No tile-unit premise (TT, TT_NCP) instantiates it; its instance in Lean is the metered model
(`tileWork_metered`). -/
def TileWork [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (γu : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain coordDomain → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM,
    A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T < (1 - γu) * ∑ u ∈ correctSet P U x.1 x.2 (A.trace U x.1 x.2), (P.Wref U.layout u : ℝ))
      ≤ δ q U.layout.N

/-- A device that pays each committed coordinate's registered size at the discount `g` and can't commit one for
less. -/
noncomputable def meteredModel (P : Protocol Q R S) (g : ℝ) : CostModel Q R S where
  Prog := Workload → (ℕ → Pouw.Mat) → Pouw.State → S → QTree Q R Pouw.Trace
  run p H U act st s := (p U act st s).eval H
  cost p H U act st s := (1 - g) * ∑ u ∈ (range U.layout.N).filter (fun u => ((p U act st s).eval H u).isSome),
    (P.Wref U.layout u : ℝ)
  calls p H U act st s := (p U act st s).calls H

theorem meteredModel_queries (P : Protocol Q R S) (g : ℝ) : (meteredModel P g).Queries :=
  fun p U act st s => ⟨p U act st s, fun _ => ⟨rfl, rfl⟩⟩

theorem tileWork_metered [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) {g : ℝ}
    (hg : g ≤ 1) : TileWork (meteredModel P g) P g fun _ _ => 0 := by
  intro U _ T _ q _ A hA
  have hempty : ∀ x : (Q → R) × S,
      ¬ T < (1 - g) * ∑ u ∈ correctSet P U x.1 x.2 (A.trace U x.1 x.2), (P.Wref U.layout u : ℝ) := by
    intro x hlt
    have hsub : correctSet P U x.1 x.2 (A.trace U x.1 x.2) ⊆
        (range U.layout.N).filter (fun u => ((A.online U A.act (A.pre.eval x.1) x.2).eval x.1 u).isSome) := by
      intro u hu
      simp only [correctSet, Finset.mem_filter] at hu
      obtain ⟨hu, c, hc, -⟩ := hu
      refine Finset.mem_filter.2 ⟨hu, ?_⟩
      change (A.trace U x.1 x.2 u).isSome
      rw [hc]
      rfl
    have hle := Finset.sum_le_sum_of_subset_of_nonneg hsub fun u _ _ => (Nat.cast_nonneg (P.Wref U.layout u) : (0 : ℝ) ≤ _)
    have hcost := (hA x.1 x.2).1
    have : (1 - g) * ∑ u ∈ correctSet P U x.1 x.2 (A.trace U x.1 x.2), (P.Wref U.layout u : ℝ) ≤ T :=
      (mul_le_mul_of_nonneg_left hle (sub_nonneg.2 hg)).trans hcost
    exact absurd hlt (not_lt.2 this)
  simp only [pr, Claims.Pr, Claims.prUniform, hempty, Finset.filter_false, Finset.card_empty, Nat.cast_zero, zero_div, le_refl]

theorem coordDomain_inhabited : ∃ U : Workload, U.InDomain coordDomain ∧ U.layout.N = 1 := by
  refine ⟨⟨⟨1, fun _ => ⟨1, 4096, 1⟩, fun _ => 0⟩, fun _ _ _ => 0⟩, ⟨by norm_num, ⟨fun _ _ => ⟨rfl, rfl⟩,
    fun _ _ => by norm_num⟩, fun _ _ _ _ _ _ => ⟨by norm_num, by norm_num⟩⟩, rfl⟩

/-- A window prover's model: its programs, each one's play in the window audit, and what the model charges for it.
`sessCalls` is free: nothing ties it to the play's queries, so no bound on `E[β]` at counted queries holds of it. -/
structure ProverModel (Q R S : Type) {n : ℕ} (Reg : Type) (Lr : Reg → SamplingDesign n)
    (session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool) where
  Prog : Type
  play : Prog → (H : Q → R) → Workload → (ℕ → Pouw.Mat) → Pouw.State → (s : S) →
    FlockSoundness.Game.Strategy (auditReg Lr (session H s))
  cost : Prog → (Q → R) → Workload → (ℕ → Pouw.Mat) → Pouw.State → S → ℝ
  received : Prog → (Q → R) → Workload → (ℕ → Pouw.Mat) → Pouw.State → S → ℕ
  cutoff : Prog → (Q → R) → Workload → (ℕ → Pouw.Mat) → Pouw.State → S → ℕ
  calls : Prog → (Q → R) → Workload → (ℕ → Pouw.Mat) → Pouw.State → S → ℕ
  sessCalls : Prog → (Q → R) → Workload → (ℕ → Pouw.Mat) → Pouw.State → S → ℕ

section Prover

variable {n : ℕ} {Reg : Type} {Lr : Reg → SamplingDesign n}
  {session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool}

/-- The registration is an oracle program's output and `calls` counts its queries; the sessions' calls
(`sessCalls`) are not tied to an oracle program. -/
def ProverModel.RegQueries (PM : ProverModel Q R S Reg Lr session) : Prop :=
  ∀ (prog : PM.Prog) (U : Workload) (act : ℕ → Pouw.Mat) (st : Pouw.State) (s : S), ∃ t : QTree Q R Reg,
    ∀ H : Q → R, (PM.play prog H U act st s).1 = t.eval H ∧ t.calls H = PM.calls prog H U act st s

/-- A window prover, chosen after the workload: a preprocessing oracle program that never reads the salt, a
program of the prover model, and the activation input it reads. -/
structure Prover (PM : ProverModel Q R S Reg Lr session) where
  pre : QTree Q R Pouw.State
  online : PM.Prog
  act : ℕ → Pouw.Mat

def Prover.play {PM : ProverModel Q R S Reg Lr session} (B : Prover PM) (U : Workload) (H : Q → R) (s : S) :
    FlockSoundness.Game.Strategy (auditReg Lr (session H s)) :=
  PM.play B.online H U B.act (B.pre.eval H) s

def Prover.received {PM : ProverModel Q R S Reg Lr session} (B : Prover PM) (U : Workload) (H : Q → R) (s : S) : ℕ :=
  PM.received B.online H U B.act (B.pre.eval H) s

def Prover.cutoff {PM : ProverModel Q R S Reg Lr session} (B : Prover PM) (U : Workload) (H : Q → R) (s : S) : ℕ :=
  PM.cutoff B.online H U B.act (B.pre.eval H) s

def Prover.Bounded {PM : ProverModel Q R S Reg Lr session} (B : Prover PM) (U : Workload) (T : ℝ) (q : ℕ) : Prop :=
  ∀ (H : Q → R) (s : S), PM.cost B.online H U B.act (B.pre.eval H) s ≤ T ∧
    B.pre.calls H + PM.calls B.online H U B.act (B.pre.eval H) s +
      PM.sessCalls B.online H U B.act (B.pre.eval H) s ≤ q

/-- The sessions accept and the registration is received before the epoch's sampling coin, an order in the
challenger's record and not a time. A later window is refused because its registration may depend on the sample,
which the window audit's game doesn't model. -/
def AcceptsBy {PM : ProverModel Q R S Reg Lr session} (B : Prover PM) (U : Workload) (H : Q → R) (s : S)
    (o : Bool × Finset (Fin n)) : Prop :=
  o.1 = true ∧ B.received U H s < B.cutoff U H s

end Prover

/-- Straight-line extraction of the registration's trace (a named premise, open; the random-oracle claim), claimed
only on runs received before the sampling coin's cutoff. It reads no session query: a session that opens a root to
anything but `Ccom` is charged in `ExecutionPhaseBound`'s `β`. -/
def StraightLineExtraction [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {n : ℕ} {Reg : Type}
    {Lr : Reg → SamplingDesign n} {session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool}
    (PM : ProverModel Q R S Reg Lr session) (CM : CostModel Q R S) (D : Layout → Prop)
    (Ccom : (Q → R) → S → Reg → Pouw.Trace) (Text : ℝ) (qext : ℕ) (εid : ℕ → ℕ → ℝ) : Prop :=
  PM.RegQueries → ∀ U : Workload, U.InDomain D → ∀ (T : ℝ) (q : ℕ) (B : Prover PM), B.Bounded U T q →
    ∃ A : Adversary CM, A.Bounded U (T + Text) (q + qext) ∧
      pr (fun x : (Q → R) × S =>
        B.received U x.1 x.2 < B.cutoff U x.1 x.2 ∧ Ccom x.1 x.2 (B.play U x.1 x.2).1 ≠ A.trace U x.1 x.2) ≤
        εid q U.layout.N

/-- A hypothesis of `EndToEndExecutionPhase`. No constant bounds `β`, because a play that holds a collision of `H`
passes with extraction failing; `W` holds PoUW's units only, so a filler adds nothing to it. -/
def ExecutionPhaseBound (P : Protocol Q R S) (U : Workload) (ε : ℝ) {n : ℕ} {Reg : Type} {Lr : Reg → SamplingDesign n}
    {session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool}
    (Ccom : (Q → R) → S → Reg → Pouw.Trace)
    (σ' : (H : Q → R) → (s : S) → FlockSoundness.Game.Strategy (auditReg Lr (session H s)))
    (β : (Q → R) → S → ℝ≥0∞) : Prop :=
  ∀ (H : Q → R) (s : S), FlockSoundness.Game.prob
      (fun o => o.1 = true ∧ ε * totalWref P U.layout ≤ incorrectWork P U H s (Ccom H s (σ' H s).1))
      (auditReg Lr (session H s)) (σ' H s) ≤ β H s

def Realizes (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) {n m : ℕ} (σ : Fin n → Fin m) (w : Fin m → ℕ)
    (cl : Fin n → Finset (Fin n)) {Tr : Type} (incorrect : Tr → Finset (Fin n)) (tx : Tr → Pouw.Trace) : Prop :=
  ∀ X : Tr, incorrectWork P U H s (tx X) ≤ (SamplingDesign.unsoundWork σ w cl (incorrect X) : ℝ)

/-- Each PoUW unit is one template instance; audit units that are no PoUW unit's instance, the fillers among them,
are unconstrained. -/
def CoordinateInstances (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) {n m : ℕ} (σ : Fin n → Fin m)
    (w : Fin m → ℕ) {Tr : Type} (incorrect : Tr → Finset (Fin n)) (tx : Tr → Pouw.Trace) (a : ℕ → Fin n) : Prop :=
  (∀ u < U.layout.N, ∀ u' < U.layout.N, a u = a u' → u = u') ∧
    (∀ u < U.layout.N, P.Wref U.layout u ≤ w (σ (a u))) ∧
    ∀ (X : Tr) (u : ℕ), u < U.layout.N → u ∉ correctSet P U H s (tx X) → a u ∈ incorrect X

def RealizesOfCoordinates : Prop :=
  ∀ (Q R S : Type) (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (n m : ℕ) (σ : Fin n → Fin m)
    (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) (Tr : Type) (incorrect : Tr → Finset (Fin n))
    (tx : Tr → Pouw.Trace) (a : ℕ → Fin n),
    CoordinateInstances P U H s σ w incorrect tx a → Realizes P U H s σ w cl incorrect tx

theorem realizes_of_coordinates : RealizesOfCoordinates := by
  intro Q R S P U H s n m σ w cl Tr incorrect tx a ⟨hinj, hw, hwrong⟩ X
  set D := range U.layout.N \ correctSet P U H s (tx X)
  have hD : ∀ u ∈ D, u < U.layout.N := fun u hu => Finset.mem_range.1 (Finset.mem_sdiff.1 hu).1
  have h1 : ∑ u ∈ D, P.Wref U.layout u ≤ ∑ u ∈ D, w (σ (a u)) := Finset.sum_le_sum fun u hu => hw u (hD u hu)
  have h2 : ∑ v ∈ D.image a, w (σ v) = ∑ u ∈ D, w (σ (a u)) :=
    Finset.sum_image fun u hu u' hu' h => hinj u (hD u hu) u' (hD u' hu') h
  have h3 : D.image a ⊆ incorrect X := by
    intro v hv
    obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 hv
    exact hwrong X u (hD u hu) (Finset.mem_sdiff.1 hu).2
  have h4 : ∑ v ∈ D.image a, w (σ v) ≤ SamplingDesign.workOf σ w (incorrect X) := Finset.sum_le_sum_of_subset h3
  have h5 := SamplingDesign.workOf_le_unsoundWork σ w cl (incorrect X)
  have hN : ∑ u ∈ D, P.Wref U.layout u ≤ SamplingDesign.unsoundWork σ w cl (incorrect X) := h1.trans (h2 ▸ h4.trans h5)
  unfold incorrectWork
  exact_mod_cast hN

/-- `ExecutionPhaseBound` at the service's sampling design: a statement with no proof. -/
def ExecutionPhaseBoundOfService : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (U : Workload)
    (ε : ℝ), 0 < ε → ε ≤ 1 →
    ∀ (N K : ℕ) (hK : K ≤ N) (m : ℕ) (σ : Fin N → Fin m) (w : Fin m → ℕ) (wmax : ℕ), (∀ t, w t ≤ wmax) →
    ∀ (Reg : Type) (session : (Q → R) → S → Finset (Fin N) → Reg → FlockSoundness.Game Bool)
      (An : (H : Q → R) → (s : S) → ExtractionAnalysis (SamplingDesign.subset N K hK) Reg (session H s))
      (tx : (H : Q → R) → (s : S) → (An H s).Tr → Pouw.Trace) (Ccom : (Q → R) → S → Reg → Pouw.Trace),
    (∀ H s R' τ, tx H s ((An H s).committed R' τ) = Ccom H s R') →
    (∀ H s, Realizes P U H s σ w (fun _ => ∅) (An H s).incorrect (tx H s)) →
    ∀ εks εbind : (H : Q → R) → (s : S) → (R' : Reg) → Cont (SamplingDesign.subset N K hK) Reg (session H s) R' → ℝ≥0∞,
    (∀ H s, (An H s).KnowledgeSound (εks H s)) → (∀ H s, (An H s).BindSound (εbind H s)) →
    ∀ σ' : (H : Q → R) → (s : S) →
      FlockSoundness.Game.Strategy (auditReg (fun _ : Reg => SamplingDesign.subset N K hK) (session H s)),
      ExecutionPhaseBound P U ε Ccom σ' fun H s =>
        (SamplingDesign.subset N K hK).miss ⌈ε * totalWref P U.layout / (wmax : ℝ)⌉₊ +
          εks H s (reg (atReg (σ' H s))) (cont (atReg (σ' H s))) +
          εbind H s (reg (atReg (σ' H s))) (cont (atReg (σ' H s)))

end Pouw.WindowWork

namespace Pouw.Guarantees

open Finset FlockSoundness.Audit
open scoped ENNReal

/-- PoUW's work bound at the window audit, a statement with no proof. The challenger's wall-clock deadline only
refuses more, so the bound holds of what the service accepts at any deadline. -/
def EndToEndExecutionPhase : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (γu : ℝ) (η : ℕ → ℕ → ℝ),
    WindowWork.TileWork CM P γu η →
    ∀ ε : ℝ, 0 < ε → ε ≤ 1 →
    ∀ (n K : ℕ) (hK : K ≤ n) (Reg : Type)
      (session : (Q → R) → S → Finset (Fin n) → Reg → FlockSoundness.Game Bool)
      (Ccom : (Q → R) → S → Reg → Trace)
      (PM : WindowWork.ProverModel Q R S Reg (fun _ : Reg => SamplingDesign.subset n K hK) session) (Text : ℝ) (qext : ℕ)
      (εid : ℕ → ℕ → ℝ),
    PM.RegQueries → 0 ≤ Text → WindowWork.StraightLineExtraction PM CM WindowWork.coordDomain Ccom Text qext εid →
    ∀ U : Workload, U.InDomain WindowWork.coordDomain → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q + qext ≤ 2 ^ 64 →
    ∀ B : WindowWork.Prover PM, B.Bounded U T q →
    ∀ β : (Q → R) → S → ℝ≥0∞, WindowWork.ExecutionPhaseBound P U ε Ccom (B.play U) β →
      FlockSoundness.Game.avg (fun x : (Q → R) × S =>
        FlockSoundness.Game.prob
          (fun o => WindowWork.AcceptsBy B U x.1 x.2 o ∧
            T + Text < (1 - γu) * (1 - ε) * totalWref P U.layout)
          (auditReg (fun _ : Reg => SamplingDesign.subset n K hK) (session x.1 x.2)) (B.play U x.1 x.2))
        ≤ FlockSoundness.Game.avg (fun x : (Q → R) × S => β x.1 x.2) + ENNReal.ofReal (εid q U.layout.N) +
          ENNReal.ofReal (η (q + qext) U.layout.N)

end Pouw.Guarantees
