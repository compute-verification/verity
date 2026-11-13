module

public import Proofs.Properties.Verity.Sampled.OpensAt

/-!
# An honest instance that runs a session

`Instance.Honest.inst` runs no session, so `remote_soundness_deployedReady`'s premises hold of it vacuously. Here the
idle deployment at v2's reading is generalized to coin families `fs` and registered files `R` (`HonestAt.base`,
`HonestAt.inst`), and specialized to one session `A₀` at the parameters `θ₀` (`HonestAt.instOne`): one coin family of
`MAX_COINS` coins there, the registered files `files` there and the parameters' digests' (`registeredOf`) elsewhere.

`HonestAt.premises_one`: at `instOne`, `remote_soundness_deployedReady`'s six premises hold wherever the session's
files give its openings (`Live.OpensReady`), v2 sets it up on `files` (`SetsUp`), `files` pass the file check
(`Vocab.Registered`) and the device's log opens only the record's rounds (`LogOpensRecorded`). PoUW's and the
inference-only policy's targets hold at `⊤`, as at `Instance.Honest`.
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

/-- **`Honest.base` at the coin families `fs` and the registered files `R`.** -/
noncomputable def base (δ : ℝ≥0∞) : Vocab Verity.Idle.device Unit :=
  { Verity.Idle.vocab δ 1 with
    families := fs
    signer := signer fs
    read := readOf
    programDigest := programDigestOf
    publicDigest := publicDigestOf
    registered := R
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

/-- **The instance at `base fs R δ` and the session `session`**, where each family `fs` names samples `MAX_COINS`
coins. -/
noncomputable def inst (δ : ℝ≥0∞) (hfs : ∀ θ l m, (l, m) ∈ fs θ → m = _root_.Spot.Challenger.MAX_COINS)
    (session : ByteArray → Option Live) : Instance Verity.Idle.device Unit where
  base := base fs R δ
  deployment := Honest.deployment
  signer _ := rfl
  read _ _ := rfl
  programDigest _ := rfl
  publicDigest _ := rfl
  place _ _ _ _ := rfl
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

variable {fs R}

/-- **The registered files pass the file check wherever some files do**, where at each parameters they are the
parameters' digests' (`registeredOf`) or pass it. -/
theorem registeredFiles {δ : ℝ≥0∞} {hfs : ∀ θ l m, (l, m) ∈ fs θ → m = _root_.Spot.Challenger.MAX_COINS}
    {session : ByteArray → Option Live}
    (h : ∀ θ, R θ = registeredOf θ ∨ (base fs R δ).Registered θ (R θ)) : (inst fs R δ hfs session).RegisteredFiles := by
  intro θ F _ hF
  show (base fs R δ).Registered θ (R θ)
  rcases h θ with he | hR
  · obtain ⟨⟨f, -, hc⟩, ⟨g, -, hp⟩⟩ := hF
    have hc' : programDigestOf θ = some (Verity.sha512 f.data.toList) := hc
    have hp' : publicDigestOf θ = some (Verity.sha512 g.data.toList) := hp
    obtain ⟨x, hx, hxe⟩ := fileWith_mem ⟨f, hc'⟩
    obtain ⟨y, hy, hye⟩ := fileWith_mem ⟨g, hp'⟩
    unfold Vocab.Registered
    rw [he, registeredOf]
    exact ⟨⟨x, Array.mem_append_left _ (Array.mem_append_left _ hx), hxe⟩,
      ⟨y, Array.mem_append_left _ (Array.mem_append_right _ hy), hye⟩⟩
  · exact hR

/-- **Every session is `OpensReady`** where every session the instance runs is. -/
theorem sessionsReady {δ : ℝ≥0∞} {hfs : ∀ θ l m, (l, m) ∈ fs θ → m = _root_.Spot.Challenger.MAX_COINS}
    {session : ByteArray → Option Live} (h : ∀ θ A, session θ = some A → A.OpensReady) :
    (inst fs R δ hfs session).SessionsReady :=
  fun θ _ A hA _ => h θ A hA

/-- **v2 sets each session up on the registered files** where it does at every session the instance runs. -/
theorem sessionRegistered {δ : ℝ≥0∞} {hfs : ∀ θ l m, (l, m) ∈ fs θ → m = _root_.Spot.Challenger.MAX_COINS}
    {session : ByteArray → Option Live} (k : Verity.Registration.Composite)
    (h : ∀ θ A, session θ = some A → SetsUp (base fs R δ) θ (R θ) A) :
    (inst fs R δ hfs session).SessionRegistered k :=
  fun θ A hA _ _ => h θ A hA

/-! ## One session -/

section One

variable (θ₀ : ByteArray) (label : String) (files : Array ByteArray)

open Classical in
/-- One coin family of `MAX_COINS` coins, `label`, at `θ₀`; none elsewhere. -/
noncomputable def fsOne (θ : ByteArray) : List (String × ℕ) :=
  if θ = θ₀ then [(label, _root_.Spot.Challenger.MAX_COINS)] else []

open Classical in
/-- The registered files `files` at `θ₀`; the parameters' digests' (`registeredOf`) elsewhere. -/
noncomputable def ROne (θ : ByteArray) : Array ByteArray := if θ = θ₀ then files else registeredOf θ

theorem ROne_self : ROne θ₀ files θ₀ = files := by simp [ROne]

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

variable {θ₀ label files}

/-- **`remote_soundness_deployedReady`'s premises at the instance that runs `A₀` at `θ₀`**, wherever `A₀`'s files
give its openings, v2 sets it up on `files`, `files` pass the file check, and the device's log opens only the
record's rounds. -/
theorem premises_one {A₀ : Live} (hR : A₀.OpensReady)
    (hS : SetsUp (base (fsOne θ₀ label) (ROne θ₀ files) ⊤) θ₀ files A₀)
    (hF : (base (fsOne θ₀ label) (ROne θ₀ files) ⊤).Registered θ₀ files)
    (hO : (instOne θ₀ label files A₀).LogOpensRecorded .remote) :
    (instOne θ₀ label files A₀).SessionsReady ∧ (instOne θ₀ label files A₀).RegisteredFiles ∧
      (instOne θ₀ label files A₀).SessionRegistered .remote ∧
      (instOne θ₀ label files A₀).LogOpensRecorded .remote ∧
      (instOne θ₀ label files A₀).base.PouwSound .remote ∧
      (instOne θ₀ label files A₀).base.InferenceOnlySound .remote := by
  classical
  have hsess : ∀ θ A, (if θ = θ₀ then some A₀ else none) = some A → θ = θ₀ ∧ A = A₀ := by
    intro θ A h
    by_cases hθ : θ = θ₀
    · rw [ite_eq_left hθ] at h
      exact ⟨hθ, (Option.some.inj h).symm⟩
    · rw [ite_eq_right hθ] at h
      cases h
  refine ⟨sessionsReady fun θ A h => ?_, registeredFiles fun θ => ?_, sessionRegistered .remote fun θ A h => ?_, hO,
    target_top _ _ .remote .pouw, target_top _ _ .remote .inferenceOnly⟩
  · obtain ⟨-, rfl⟩ := hsess θ A h
    exact hR
  · by_cases hθ : θ = θ₀
    · subst hθ
      right
      rw [ROne_self]
      exact hF
    · left
      unfold ROne
      rw [ite_eq_right hθ]
  · obtain ⟨rfl, rfl⟩ := hsess θ A h
    show SetsUp _ _ (ROne _ files _) _
    rw [ROne_self]
    exact hS

end One

end Verity.Sampled.Instance.HonestAt
