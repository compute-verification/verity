import Proofs.Core.ProofSystem
import Proofs.Flock.Claims.ERep
import Proofs.Flock.Claims.Recursion.Wald

/-!
# Flock as a proof system

Flock's record as the interface's proof system (`system`, at a decoder) and its masked record as a zero-knowledge one
(`zkSystem`), whose public shape is the recursion's numbers (`Recursion.Nums`). At the plain sample its KS is
`Flock.KS`'s (`system_ks_iff`, `ks_iff_system`); against aborting provers it is `Inner.KSAbort` at `Inner.ofFlock`
(`system_ksAbort_iff`). Its extractor reruns as Wald's bound says (`system_wald`) and returns only witnesses in the
relation (`system_selfChecking`). The shape bounds its errors (`zkSystem_byShape`), and its zero knowledge is `Flock.ZK`
at each attempt budget's simulator (`zkSystem_zk_iff`).
-/

namespace Claims.Flock

open Verity
open scoped NNReal

private theorem R_of_check {parse : Msg → Option Stmt} {x : Msg} {w : Wit} (h : check parse x w = true) :
    R parse x w := by
  unfold check at h
  cases hp : parse x with
  | none => rw [hp] at h; cases h
  | some s => rw [hp] at h; exact ⟨s, hp, of_decide_eq_true h⟩

section Systems

variable {Coin Col T : Type}

/-- **Flock as a proof system**, at a decoder: its verifier, relation and scope; `E^rep` with its reruns, `1 + m′·k` in
expectation and `1 + m′·K` at most; `κ` and `B`'s term at `γ`; its reference prover and completeness error; and the
prover's stop and message lengths, which the record doesn't state. -/
noncomputable def system (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (stops : (Msg → Msg) → Msg → List Coin → Bool) (len : Msg → ℕ → ℕ) : ProofSystem Coin Wit where
  V := D.V
  R := R D.parse
  Scope := Scope D
  scopeCheck := D.scopeCheck
  E _ := PExtractor.rep D.V D.waits D.k D.cap (fun v => [D.cols v]) (assemble D.parse D.table decode D.read)
    (check D.parse)
  N x := 1 + D.waits x * D.k x
  M x := 1 + D.waits x * D.cap x
  κ := κ D
  terms := [KTerm.B (γ D)]
  Tape := D.Tape
  P := D.P
  δc := D.δc
  stops := stops
  len := len

/-- **A statement's public shape**: the numbers Flock's errors read, at its leaves and coin leaves. -/
noncomputable def shape (D : Deploy Coin Col T) (leaves coinLeaves : Msg → ℕ) (x : Msg) : Recursion.Nums :=
  ⟨D.εtable x, D.N₀ x, D.weight x, D.V.k x, D.waits x, D.cap x, leaves x, coinLeaves x⟩

/-- **Masked Flock as a zero-knowledge proof system**: `system` at the masked record, with its masked prover over the
`Hello` round and `V`'s; simulators `Sim t` at each attempt budget wherever `Hides` holds, at the errors Flock's `--zk`
zero knowledge has, `Nums.εzk` and `4t·3^t`; and each statement's shape. -/
noncomputable def zkSystem (D : ZkDeploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (stops : (Msg → Msg) → Msg → List Coin → Bool) (len : Msg → ℕ → ℕ) (leaves coinLeaves : Msg → ℕ)
    (Sim : ℕ → Simulator) (Hides : (Msg → Msg) → ℝ≥0 → Prop) : ZkSystem Coin Wit where
  toProofSystem := system D.toDeploy decode stops len
  kZK x := D.V.k x + 1
  Nonce _ := D.Nonce
  Rest _ := D.Rest
  Pz w _ := D.prover w
  Sim := Sim
  εzk t δ₁ x := (shape D.toDeploy leaves coinLeaves x).εzk t δ₁
  γzk t _ := 4 * t * 3 ^ t
  Hides := Hides
  Shape := Recursion.Nums
  shape := shape D.toDeploy leaves coinLeaves
  CS n S := Fin n.waits → Fin n.cap → S
  κS := Recursion.Nums.κ
  wS n := [n.γ]
  NS n := 1 + n.waits * n.k
  MS n := 1 + n.waits * n.cap
  εzkS t δ₁ n := n.εzk t δ₁
  γzkS t _ := 4 * t * 3 ^ t

variable (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (stops : (Msg → Msg) → Msg → List Coin → Bool)
  (len : Msg → ℕ → ℕ)

/-- The system's extractor at the plain sample is Flock's, `E D decode`. -/
theorem system_E_toExtractor [Fintype Coin] [Nonempty Coin] (H : Msg → Msg) :
    ((system D decode stops len).E H).toExtractor = E D decode :=
  rfl

/-- **The system's KS is Flock's**, at the decoder. -/
theorem system_ks_iff [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    (system D decode stops len).KS H ↔ Claims.KS H D.V (R D.parse) (E D decode) (κ D) (γ D) :=
  ProofSystem.ks_iff_claims H _ (γ D) rfl

/-- **`Flock.KS` is the system's KS** at every decoder that meets the list-decoding spec. -/
theorem ks_iff_system [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Msg → Msg) :
    KS H D ↔ ∀ decode, ListWithin D.len D.enc D.θ decode → (system D decode stops len).KS H :=
  forall₂_congr fun decode _ => (system_ks_iff D decode stops len H).symm

/-- **The system's KS against aborting provers is `Inner.KSAbort` at `Inner.ofFlock`.** -/
theorem system_ksAbort_iff [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    (system D decode stops len).KSAbort H ↔ Recursion.Inner.KSAbort H (Recursion.Inner.ofFlock D decode) :=
  ksAbort_iff_inner H (Recursion.Inner.ofFlock D decode)

/-- **The system's extractor reruns as Wald's bound says**: `1 + m′·k` times in expectation, weighted by any nonnegative
cost of a sample, and `1 + m′·K` at most. -/
theorem system_wald (H : Msg → Msg) : (system D decode stops len).Wald H := by
  refine ⟨fun x F ω f hf => ?_, fun x F ω d c => ?_⟩
  · have h := erepProg_cost_exp H (F.trace ω) hf (D.waits x) (D.k x) (D.cap x) (fun v => [D.cols v])
      (assemble D.parse D.table decode D.read x) (check D.parse x)
    show _ ≤ ((1 + D.waits x * D.k x : ℕ) : ℝ) * Ex f
    push_cast
    exact h
  · exact erepProg_runs_le H (F.trace ω) _ _ _ _ _ _ d c

/-- **The system's extractor returns only witnesses in the relation**, over every sample space: `E^rep` returns a
candidate only where `check` accepts it. -/
theorem system_selfChecking (H : Msg → Msg) : (system D decode stops len).SelfChecking H := by
  intro x Dsp runs d c w h
  change (RP.eval H runs (.rerun d fun v₀ =>
    if v₀.out = .acc then
      firstWait (D.k x) (fun v => [D.cols v]) (assemble D.parse D.table decode D.read x) (check D.parse x)
        ((List.finRange (D.waits x)).map fun i => (List.finRange (D.cap x)).map (c i))
    else .ret none)).1 = some w at h
  simp only [RP.eval] at h
  split at h
  · exact R_of_check (firstWait_some H runs _ _ _ _ _ w h)
  · cases h

/-- The system's completeness is Flock's. -/
theorem system_complete_iff [Fintype Coin] [Inhabited Coin] (H : Msg → Msg) :
    (system D decode stops len).Complete H ↔ Complete H D :=
  Iff.rfl

/-- The system's readable scope is Flock's. -/
theorem system_scopeMatches_iff : (system D decode stops len).ScopeMatches ↔ ScopeMatches D :=
  Iff.rfl

/-- **Masked Flock's errors are its shape's**: its extractor's coins, `κ`, `γ`, reruns and simulation errors are
functions of the statement's `Nums`. -/
theorem zkSystem_byShape (D : ZkDeploy Coin Col T) (leaves coinLeaves : Msg → ℕ) (Sim : ℕ → Simulator)
    (Hides : (Msg → Msg) → ℝ≥0 → Prop) : (zkSystem D decode stops len leaves coinLeaves Sim Hides).ByShape :=
  fun _ => ⟨fun _ => rfl, le_rfl, .cons le_rfl .nil, le_rfl, le_rfl, fun _ _ => le_rfl, fun _ => le_rfl⟩

/-- **Masked Flock's zero knowledge is `Flock.ZK`** at each attempt budget's simulator and errors, wherever the
commitments hide. -/
theorem zkSystem_zk_iff (H : Msg → Msg) (D : ZkDeploy Coin Col T) (leaves coinLeaves : Msg → ℕ)
    (Sim : ℕ → Simulator) (Hides : (Msg → Msg) → ℝ≥0 → Prop) (hzk : D.zk = true) :
    (zkSystem D decode stops len leaves coinLeaves Sim Hides).ZK H ↔
      ∀ t, 0 < t → ∀ δ₁ : ℝ≥0, Hides H δ₁ →
        ZK H { D with S := Sim t } (fun x => (shape D.toDeploy leaves coinLeaves x).εzk t δ₁)
          (fun _ => 4 * t * 3 ^ t) := by
  constructor
  · intro h t ht δ₁ hδ
    refine ⟨hzk, fun x w hR Θ Vs θ hk Ev => ?_⟩
    exact h t ht δ₁ hδ x x w ⟨rfl, hR⟩ Θ Vs θ hk Ev
  · rintro h t ht δ₁ hδ x₀ x w ⟨rfl, hR⟩ Θ Vs θ hk Ev
    exact (h t ht δ₁ hδ).2 x w hR Θ Vs θ hk Ev

end Systems

end Claims.Flock
