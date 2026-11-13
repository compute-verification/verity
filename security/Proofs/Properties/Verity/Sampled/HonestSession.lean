module

public import Proofs.Properties.Verity.Sampled.OpensAt

/-!
# An honest instance that runs a session

`Instance.Honest.inst` runs no session, so `remote_soundness_deployedReady`'s session premise holds of it vacuously.
Here the idle deployment at v2's reading is generalized to coin families `fs` and a registration `R` (`HonestAt.base`,
`HonestAt.inst`), and specialized to one session `A₀` at the parameters `θ₀` (`HonestAt.instOne`): one coin family of
`MAX_COINS` coins there, the registration's files `files` there and the parameters' digests' (`registeredOf`)
elsewhere.

`HonestAt.premises_one`: at `instOne`, `remote_soundness_deployedReady`'s premises hold wherever v2 sets `A₀` up on
the registered files at `θ₀` (`SetsUp`) on parameters it reads, given a play on files naming the same circuit and public
files that `remote`'s sampled verdict accepts; PoUW's and the inference-only policy's targets hold at `⊤`, as at
`Instance.Honest`.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestAt

open scoped ENNReal

variable (fs : ByteArray → List (String × ℕ)) (R : ByteArray → Array ByteArray)

/-- The idle deployment's signer at the coin families `fs`. -/
def signer (θ : ByteArray) : Signer := Honest.deployment.signer fs θ

/-- The idle deployment's history at each play, at the coin families `fs`. -/
noncomputable abbrev dev : Exchange → Properties.Challenger.History := Device.dev (Honest.deployment.setupOf fs)

/-- **The declared calls** at the coin families `fs`, as `Honest.cost`. -/
noncomputable def cost (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (Verity.play (fs θ) (signer fs θ) θ F c)) : ℕ :=
  Finset.univ.sup fun t : Verity.Game.Tape (Verity.play (fs θ) (signer fs θ) θ F c) σ =>
    Binding.breakCalls (dev fs (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
      (Verity.Game.Tape.out _ σ t).sent.blobs.size

/-- **`Honest.base` at the coin families `fs` and the registration `R`** (`registeredFrom`). -/
noncomputable def base (δ : ℝ≥0∞) : Vocab Verity.Idle.device Unit :=
  { Verity.Idle.vocab δ 1 with
    families := fs
    signer := signer fs
    read := readOf
    programDigest := programDigestOf
    publicDigest := publicDigestOf
    registered := registeredFrom R
    place := fun θ F _ g => placeOfRead (readOf θ F) g
    extract := fun _ _ _ _ _ => 0
    finder := fun _ _ _ _ _ => Verity.Idle.finder
    cost := cost fs
    finder_calls := fun _ _ _ _ _ _ => Nat.zero_le _
    finders := fun _ _ _ _ => []
    premises := fun _ _ _ _ _ => True
    forger := fun _ _ _ _ _ => Verity.Idle.forger
    forger_calls := fun _ _ _ _ _ _ => Nat.zero_le _
    signatures := Honest.deployment.signatures
    signs_le := Honest.deployment.signs_le fs
    forger_signed := fun _ _ _ _ _ _ => Nat.zero_le _
    computed := fun _ _ _ _ _ => ∅
    macs := fun _ _ _ _ _ => 0
    world := fun _ _ _ _ _ => Verity.Idle.world
    world_window := fun _ _ _ _ _ => rfl
    world_capacity := fun _ _ _ _ _ => rfl
    theta := fun _ _ => ∅
    theta_input := fun _ _ _ h => absurd h (Finset.notMem_empty _)
    schedule := fun θ F => Honest.blocks (readOf θ F).circuit
    wipe := fun θ F => Honest.wipe (readOf θ F).circuit
    wipeRun := fun θ F c _ => Honest.ofRun (readOf θ F).circuit (Verity.play (fs θ) (signer fs θ) θ F c)
    honest := fun θ F c =>
      Verity.requestThen (fs θ) (signer fs θ) θ F c .empty ⟨.empty, fun _ => ⟨(.empty, #[]), ⟨⟩⟩⟩ }

theorem costs (δ : ℝ≥0∞) : CostsRead (base fs R δ) (dev fs) := fun θ F c σ t =>
  Finset.le_sup (f := fun t : Verity.Game.Tape (Verity.play (fs θ) (signer fs θ) θ F c) σ =>
    Binding.breakCalls (dev fs (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
      (Verity.Game.Tape.out _ σ t).sent.blobs.size) (Finset.mem_univ t)

/-- **The instance at `base fs R δ`, the registration `R` and the session `session`**, where each family `fs` names
samples `MAX_COINS` coins. -/
noncomputable def inst (δ : ℝ≥0∞) (hfs : ∀ θ l m, (l, m) ∈ fs θ → m = _root_.Spot.Challenger.MAX_COINS)
    (session : ByteArray → Option Live) : Instance Verity.Idle.device Unit where
  base := base fs R δ
  deployment := Honest.deployment
  signer _ := rfl
  read _ _ := rfl
  programDigest _ := rfl
  publicDigest _ := rfl
  place _ _ _ _ := rfl
  registration := R
  registered _ := rfl
  costs := costs fs R δ
  coins := hfs
  session := session

/-- At `base fs R ⊤` every subprotocol's error is `⊤`. -/
theorem error_top (a : Subprotocol) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy ((base fs R ⊤).game θ F c)) : (base fs R ⊤).error a θ F c σ = ⊤ := by
  unfold Vocab.error Accounting.condError
  split_ifs <;> simp [base, Verity.Idle.vocab]

/-- At `base fs R ⊤` every subprotocol meets its target at every composite. -/
theorem target_top (k : Verity.Registration.Composite) (a : Subprotocol) : (base fs R ⊤).Target k a := by
  intro θ F c σ
  rw [error_top]
  exact le_top

/-! ## One session -/

section One

variable (θ₀ : ByteArray) (label : String) (files : Array ByteArray)

open Classical in
/-- One coin family of `MAX_COINS` coins, `label`, at `θ₀`; none elsewhere. -/
noncomputable def fsOne (θ : ByteArray) : List (String × ℕ) :=
  if θ = θ₀ then [(label, _root_.Spot.Challenger.MAX_COINS)] else []

open Classical in
/-- The registration's files `files` at `θ₀`; the parameters' digests' (`registeredOf`) elsewhere. -/
noncomputable def ROne (θ : ByteArray) : Array ByteArray := if θ = θ₀ then files else registeredOf θ

theorem fsOne_coins : ∀ θ l m, (l, m) ∈ fsOne θ₀ label θ → m = _root_.Spot.Challenger.MAX_COINS := by
  intro θ l m h
  unfold fsOne at h
  split_ifs at h
  · simp only [List.mem_singleton, Prod.mk.injEq] at h
    exact h.2
  · cases h

open Classical in
/-- **The instance that runs the session `A₀` at `θ₀`**, at `base (fsOne θ₀ label) (ROne θ₀ files) ⊤`. -/
noncomputable def instOne (A₀ : Live) : Instance Verity.Idle.device Unit :=
  inst (fsOne θ₀ label) (ROne θ₀ files) ⊤ (fsOne_coins θ₀ label) fun θ => if θ = θ₀ then some A₀ else none

theorem instOne_session (A₀ : Live) : (instOne θ₀ label files A₀).session θ₀ = some A₀ := by
  simp [instOne, inst]

/-- **The registered files at `θ₀` are `files`** where `files` hold a file with each digest the parameters name. -/
theorem registered_one (h : (base (fsOne θ₀ label) (ROne θ₀ files) ⊤).Registered θ₀ files) :
    registeredFrom (ROne θ₀ files) θ₀ = files := by
  have hR : ROne θ₀ files θ₀ = files := by simp [ROne]
  unfold registeredFrom
  rw [hR]
  exact ite_eq_left h

variable {θ₀ label files}

/-- **`remote_soundness_deployedReady`'s premises at the instance that runs `A₀` at `θ₀`**, wherever v2 sets `A₀` up
on the registered files at `θ₀` on parameters it reads, given a play on files naming the same circuit and public files,
which `remote`'s sampled verdict accepts and whose prover sent the tables (`SessionRegistered`'s hypotheses there). -/
theorem premises_one {A₀ : Live}
    (hS : (∃ p, ParamsAt θ₀ p) →
      (∃ F c o, (instOne θ₀ label files A₀).base.circuitFile θ₀ F =
          (instOne θ₀ label files A₀).base.circuitFile θ₀ ((instOne θ₀ label files A₀).base.registered θ₀) ∧
        (instOne θ₀ label files A₀).base.publicFile θ₀ F =
          (instOne θ₀ label files A₀).base.publicFile θ₀ ((instOne θ₀ label files A₀).base.registered θ₀) ∧
        Verity.Game.Reaches ((instOne θ₀ label files A₀).base.game θ₀ F c) o ∧
        (instOne θ₀ label files A₀).base.Verdict .remote .sampled o ∧ TablesSent o.sent) →
      SetsUp (instOne θ₀ label files A₀).base θ₀ ((instOne θ₀ label files A₀).base.registered θ₀) A₀) :
    (instOne θ₀ label files A₀).SessionRegistered .remote ∧
      (instOne θ₀ label files A₀).base.PouwSound .remote ∧
      (instOne θ₀ label files A₀).base.InferenceOnlySound .remote := by
  classical
  refine ⟨fun θ A h => ?_, target_top _ _ .remote .pouw, target_top _ _ .remote .inferenceOnly⟩
  have h' : (if θ = θ₀ then some A₀ else none) = some A := h
  by_cases hθ : θ = θ₀
  · rw [ite_eq_left hθ] at h'
    cases Option.some.inj h'
    subst hθ
    exact hS
  · rw [ite_eq_right hθ] at h'
    cases h'

end One

end Verity.Sampled.Instance.HonestAt
