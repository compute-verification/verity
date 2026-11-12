module

public import Proofs.Properties.Verity.Assembled
public import Proofs.Properties.Verity.Sampled.Binding
public import Proofs.Properties.Verity.Sampled.CountsLayout
public import Proofs.Properties.Verity.Sampled.MovesReachAt
public import Proofs.Properties.Verity.Sampled.Place
public import Proofs.Flock.Recursive.TopFinder

/-!
# SPOT's vocabulary as deployed

The vocabulary SPOT's target reads (`Deployed`, `Covers`) built from the other subprotocols' names over the deployed
device (`Instance.base`), so `spotSound_of_targets` and `remote_soundness_of_targets` apply to it with every field of
`Deployed` and `Covers` either proved or one of the named `Prop`s below.

* **The base** (`Instance`): a vocabulary whose signer is the deployed device's (`Device.Deployment.signer`) and whose
  shared names are v2's (its reading of the committed files is `readOf`, its digests `programDigestOf` and
  `publicDigestOf`), whose declared calls cover the deployed device's reading of what the prover sent at its history
  (`CostsRead` at `Device.dev`, what the read finder and the forger spend beside the strategy), and the deployment's
  session at each parameters (`Instance.session`), the bound's parameters fixed by `θ`.
* **The sessions** (`Instance.sessions`): the deployment's session wherever v2 sets it up on the committed files
  (`SetsUp`).
* **The sampled names** (`Instance.stage`, at constructions `K` over the base, whose game and declared calls are the
  stage's): v2's log (`coinsRead`), the binding's read finder and forger (`Binding.readFinder`, `Binding.forger`) at
  `2t` calls, the further finders (the file finders under `ecr/sha-512`, `fileSpecs`, and at a
  session `K`'s two-continuation finders under `cr/sha-512`, `strictSpecs`), the session's premises at the projected
  strategy (`premisesOf`), no loss, and the session's δ (`Live.delta`) at the deployment's session, `⊤` where it
  runs none (`deltaOf`). Each premise is
  asked of the finders built from the one strategy `error` is at. The other subprotocols' names are the base's, so
  their targets are the base's (`pouwSound_of_base`, `inferenceOnlySound_of_base`).
* **The extractor** (`Instance.vocab`): the session's at the projected strategy.
* **The coupling** (`couplingTarget_at`, at the constructions `constructionsAt` assembles at the deployed device's
  history): `couplingTargetAt_counts` at the stage, whose signer premise is the base's (`Instance.signer`), whose
  session is one v2 sets up (`sessions_some`), whose digests are v2's (`digestsAt_of`) and whose session reads
  `MAX_COINS` frames' bytes, each coin family the base names sampling `MAX_COINS` coins (`Instance.coins`,
  `lb_of_sessions`).
* **The registered files** (`RegisteredFiles`): wherever some committed files pass the file check, the registered
  files do, which holds of registered files built from the parameters' digests (`registeredOf`,
  `registeredFiles_of`), the honest instance's.
* **What stays open**: the commitment format's openings (`OpensAt`, from the sessions' files in
  `Sampled/OpensAt.lean`), v2's session at the registered files (`SessionAt`, from `SessionRegistered` in
  `Sampled/SessionAt.lean`), and of the coupling the device's log opening only the record's rounds
  (`LogOpensRecorded`, `remote_soundness_deployedAt`), each of the sessions the deployment runs.
-/

@[expose] public section

namespace Verity.Sampled

open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The file finders -/

/-- A one-coin finder's success is `0` or `1`, so below `1` it is `0`. -/
theorem pr_unit {p : Unit → Prop} (h : Claims.Pr p < 1) : Claims.Pr p = 0 := by
  classical
  unfold Claims.Pr Claims.prUniform at *
  by_cases hp : p ()
  · rw [Finset.filter_true_of_mem fun x _ => by cases x; exact hp] at h
    simp at h
  · rw [Finset.filter_false_of_mem fun x _ => by cases x; exact hp]
    simp

theorem budget_lt {n : ℕ} (hn : n < 2 ^ 256) : (n : ℝ) / 2 ^ (256 : ℝ) < 1 := by
  have h2 : (2 : ℝ) ^ (256 : ℝ) = ((2 ^ 256 : ℕ) : ℝ) := by
    rw [show (256 : ℝ) = ((256 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    norm_num
  rw [h2, div_lt_one (by positivity)]
  exact_mod_cast hn

/-- A finder that never finds a collision meets the strict birthday bound. -/
theorem strict_of_zero {f : Claims.Finder} (h : f.Pr Verity.sha512Msg = 0) :
    Claims.StrictBirthday Verity.sha512Msg f := fun q _ => by
  rw [h]
  exact div_nonneg (sq_nonneg q) (Real.rpow_nonneg (by norm_num) _)

theorem sha512Msg_of_sha512 {a b : Claims.Msg} (h : Verity.sha512 a = Verity.sha512 b) :
    Verity.sha512Msg a = Verity.sha512Msg b := by
  have ha := FlockSoundness.Refine.sha512_size ⟨a.toArray⟩
  have hb := FlockSoundness.Refine.sha512_size ⟨b.toArray⟩
  unfold Verity.sha512Msg
  apply List.ext_getElem
  · rw [Array.length_toList, Array.length_toList, ByteArray.size_data, ByteArray.size_data, ha, hb]
  · intro i h1 h2
    have h1' : i < (Flock.Sha512.hash ⟨a.toArray⟩).data.size := by
      rw [Array.length_toList] at h1; exact h1
    have h2' : i < (Flock.Sha512.hash ⟨b.toArray⟩).data.size := by
      rw [Array.length_toList] at h2; exact h2
    have hi64 : i < 64 := by rw [ByteArray.size_data, ha] at h1'; exact h1'
    have hi := congrFun h ⟨i, hi64⟩
    have h1s : i < (Flock.Sha512.hash ⟨a.toArray⟩).size := h1'
    have h2s : i < (Flock.Sha512.hash ⟨b.toArray⟩).size := h2'
    rw [Array.getElem_toList, Array.getElem_toList]
    simpa [Verity.sha512, Array.getD, h1', h2', h1s, h2s] using hi

/-- **The file a digest names has the digest**, where the committed files hold one with it. -/
theorem namedFile_digest {h : Option Hash.Digest} {F : Array ByteArray}
    (hF : ∃ f ∈ F, h = some (Verity.sha512 f.data.toList)) : h = some (Verity.sha512 (namedFile h F).data.toList) := by
  obtain ⟨f, hf, hh⟩ := hF
  unfold namedFile
  cases hfind : F.find? (fun f => decide (h = some (Verity.sha512 f.data.toList))) with
  | none => exact absurd (decide_eq_true hh) (Array.find?_eq_none.1 hfind f hf)
  | some g =>
    have hg := Array.find?_some hfind
    exact of_decide_eq_true hg

/-- **A file finder below one success names one file**: where both the committed files and the registered files hold
a file with the digest, a file finder that outputs no collision with probability `1` names the registered file. -/
theorem namedFile_eq {h : Option Hash.Digest} {F R : Array ByteArray}
    (hF : ∃ f ∈ F, h = some (Verity.sha512 f.data.toList)) (hR : ∃ f ∈ R, h = some (Verity.sha512 f.data.toList))
    (hp : (namedFinder h F R).Pr Verity.sha512Msg < 1) : namedFile h F = namedFile h R := by
  by_contra hne
  have hd := (namedFile_digest hF).symm.trans (namedFile_digest hR)
  have h1 := namedFinder_collides hne (sha512Msg_of_sha512 (Option.some.inj hd))
  rw [h1] at hp
  exact lt_irrefl _ hp

/-! ## The registered files -/

open Classical in
/-- **A file with the digest `h`**, where some file has it. -/
noncomputable def fileWith (h : Option Hash.Digest) : Option ByteArray :=
  if hx : ∃ f : ByteArray, h = some (Verity.sha512 f.data.toList) then some hx.choose else none

theorem fileWith_mem {h : Option Hash.Digest} (hx : ∃ f : ByteArray, h = some (Verity.sha512 f.data.toList)) :
    ∃ f ∈ (fileWith h).toArray, h = some (Verity.sha512 f.data.toList) := by
  have he : fileWith h = some hx.choose := by
    unfold fileWith
    rw [dif_pos hx]
  refine ⟨hx.choose, ?_, hx.choose_spec⟩
  rw [he]
  simp [Option.toArray]

/-- **The files the parameters' digests name**: a file with the circuit file's digest and one with the public file's
that v2's parameters hold (`programDigestOf`, `publicDigestOf`), where some file has it, and the lookup tables that
circuit file registers (`pinnedTables`). -/
noncomputable def registeredOf (θ : ByteArray) : Array ByteArray :=
  (fileWith (programDigestOf θ)).toArray ++ (fileWith (publicDigestOf θ)).toArray ++
    (((fileWith (programDigestOf θ)).map fun f => (pinnedTables f).map (·.2)).getD []).toArray

/-! ## The base -/

/-- **The base's declared calls cover the device's reading**: on every tape of the period's game, the strings `findBreak`
hashes in the play's records (`Binding.breakCalls`) and one call per blob the prover sent are at most the strategy's
declared calls, so the read finder and the forger spend at most twice them. -/
def CostsRead (V : Vocab d α) (dev : Exchange → Properties.Challenger.History) : Prop :=
  ∀ θ F c (σ : Verity.Game.Strategy (V.game θ F c)) (t : Verity.Game.Tape (V.game θ F c) σ),
    Binding.breakCalls (dev (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
      (Verity.Game.Tape.out _ σ t).sent.blobs.size ≤ V.cost θ F c σ

/-- **The deployment, before SPOT's names**: the other subprotocols' vocabulary over the deployed device
(`base`), whose signer is the deployed device's (`deployment`, `signer`), whose reading of the committed files and
digests are v2's (`read`, `programDigest`, `publicDigest`), whose place for each gate is the committed files' (`place`,
`placeOfRead`) and whose declared calls cover the deployed device's reading at its history (`costs`, `Device.dev`); and
the session the deployment runs at each parameters (`session`), whose bound's parameters `θ` fixes; each coin family
the base names samples the deployed device's `MAX_COINS` coins (`coins`, its one-stage rounds). A deployed instance fills
`signer`, `place` and `coins` at its construction, by `rfl` or a lemma, as the honest instance does; no theorem takes
any of them as a hypothesis of its own: `couplingTarget_at` reads the signer's and the coins' from these fields, and
`opensAt_of_sessions` the place's. -/
structure Instance (d : Accounting.Device ι) (α : Type) [Fintype α] where
  base : Vocab d α
  deployment : Device.Deployment
  signer : ∀ θ, base.signer θ = deployment.signer base.families θ
  read : ∀ θ F, base.read θ F = readOf θ F
  programDigest : ∀ θ, base.programDigest θ = programDigestOf θ
  publicDigest : ∀ θ, base.publicDigest θ = publicDigestOf θ
  place : ∀ θ F c g, base.place θ F c g = placeOfRead (base.read θ F) g
  costs : CostsRead base (Device.dev (deployment.setupOf base.families))
  coins : ∀ θ l m, (l, m) ∈ base.families θ → m = _root_.Spot.Challenger.MAX_COINS
  session : ByteArray → Option Live

namespace Instance

variable (I : Instance d α)

/-- **The deployed device's history at each play** (`Device.dev`), at the base's layouts. -/
noncomputable abbrev dev : Exchange → Properties.Challenger.History :=
  Device.dev (I.deployment.setupOf I.base.families)

/-- **SPOT's further finders as deployed**: the circuit file's and the public file's (`Vocab.fileFinder`,
`Vocab.publicFinder`) at two calls each. -/
noncomputable def fileSpecs (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    List (FinderSpec (I.base.game θ F c) (I.base.cost θ F c)) :=
  [{ finder := fun _ => I.base.fileFinder θ F, budget := fun _ => 2, calls := fun _ _ => Nat.le_refl 2 },
   { finder := fun _ => I.base.publicFinder θ F, budget := fun _ => 2, calls := fun _ _ => Nat.le_refl 2 }]

noncomputable def coinsReadOf : Subprotocol → Sent → Option (List Release)
  | .sampled => Verity.Sampled.coinsRead
  | a => I.base.coinsRead a

noncomputable def finderOf : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (I.base.game θ F c) → Claims.Finder
  | .sampled => Binding.readFinder I.base I.dev
  | a => I.base.finder a

noncomputable def forgerOf : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (I.base.game θ F c) → Hash.Assumptions.LmsForger
  | .sampled => Binding.forger I.base I.dev
  | a => I.base.forger a

def callsMulOf (θ : ByteArray) : Subprotocol → ℕ
  | .sampled => 2
  | a => I.base.callsMul θ a

def callsAddOf (θ : ByteArray) : Subprotocol → ℕ
  | .sampled => 0
  | a => I.base.callsAdd θ a

open Classical in
/-- **The deployment's sessions**: at the parameters `θ` and the committed files `F`, the deployment's session where
v2 sets it up on them (`SetsUp`). -/
noncomputable def sessions (θ : ByteArray) (F : Array ByteArray) : Option Live :=
  (I.session θ).bind fun A => if SetsUp I.base θ F A then some A else none

theorem sessions_some {θ : ByteArray} {F : Array ByteArray} {A : Live} (h : I.sessions θ F = some A) :
    I.session θ = some A ∧ SetsUp I.base θ F A := by
  unfold sessions at h
  cases hA : I.session θ with
  | none => rw [hA] at h; cases h
  | some B =>
    rw [hA] at h
    dsimp only [Option.bind] at h
    split_ifs at h with hs
    cases h
    exact ⟨rfl, hs⟩

/-- **A session of the deployment reads `MAX_COINS` frames' bytes**: v2 sets it up at a coin family the base names
(`SetsUp.params`), which samples `MAX_COINS` coins of `COIN_BYTES` bytes (`coins`). -/
theorem lb_of_sessions {θ : ByteArray} {F : Array ByteArray} {A : Live} (h : I.sessions θ F = some A) :
    A.Lb 0 = _root_.Spot.Challenger.MAX_COINS * 64 := by
  obtain ⟨_, -, -, -, -, -, -, _, l, m, -, hm, hL⟩ := (I.sessions_some h).2.params
  rw [hL, I.coins θ l m hm]
  rfl

theorem sessions_of {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : I.session θ = some A)
    (hs : SetsUp I.base θ F A) : I.sessions θ F = some A := by
  unfold sessions
  rw [hA]
  dsimp only [Option.bind]
  rw [if_pos hs]

variable (K : Constructions I.base)

/-- **A two-continuation finder as a further finder**: `K`'s at round `j` of the session `A`, asked under `cr/sha-512`
at its `2t + 2h` calls (`Constructions.msg_calls`, `Live.qM`). -/
noncomputable def msgSpec (A : Live) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (j : Fin A.rounds) :
    FinderSpec (I.base.game θ F c) (I.base.cost θ F c) where
  finder σ := K.msgFinder A θ F c σ j
  budget t := 2 * t + 2 * A.h
  calls σ x := by
    have h := K.msg_calls A θ F c σ j x
    unfold Live.qM at h
    exact_mod_cast h
  claim := .cr

/-- **The record's two-continuation finder as a further finder**: `K`'s at the session `A`, asked under `cr/sha-512`
at its `2t + 2h` calls (`Constructions.record_calls`). -/
noncomputable def recordSpec (A : Live) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    FinderSpec (I.base.game θ F c) (I.base.cost θ F c) where
  finder σ := K.recordFinder A θ F c σ
  budget t := 2 * t + 2 * A.h
  calls σ x := by
    have h := K.record_calls A θ F c σ x
    unfold Live.qM at h
    exact_mod_cast h
  claim := .cr

/-- **The bound digests' two-continuation finder as a further finder**: `K`'s at the session `A`, asked under
`cr/sha-512` at its `2t + 2h` calls (`Constructions.bind_calls`). -/
noncomputable def bindSpec (A : Live) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    FinderSpec (I.base.game θ F c) (I.base.cost θ F c) where
  finder σ := K.bindFinder A θ F c σ
  budget t := 2 * t + 2 * A.h
  calls σ x := by
    have h := K.bind_calls A θ F c σ x
    unfold Live.qM at h
    exact_mod_cast h
  claim := .cr

/-- **The session's two-continuation finders as further finders**: at the deployment's session, one per round
(`msgSpec`), the record's (`recordSpec`) and the bound digests' (`bindSpec`); none elsewhere. -/
noncomputable def strictSpecs (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    List (FinderSpec (I.base.game θ F c) (I.base.cost θ F c)) :=
  match I.sessions θ F with
  | some A => (List.finRange A.rounds).map (I.msgSpec K A θ F c) ++ [I.recordSpec K A θ F c, I.bindSpec K A θ F c]
  | none => []

noncomputable def findersOf : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    List (FinderSpec (I.base.game θ F c) (I.base.cost θ F c))
  | .sampled => fun θ F c => I.fileSpecs θ F c ++ I.strictSpecs K θ F c
  | a => I.base.finders a

/-- **SPOT's session premises as deployed**: at the deployment's session, the session's at the strategy
projected there (`Live.FinderCR`), asked of this strategy alone; `True` where there is none. -/
def premisesOf : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (I.base.game θ F c) → Prop
  | .sampled => fun θ F c σ =>
    ∀ A, I.sessions θ F = some A → (K.live A θ F c σ).FinderCR (K.project A θ F c σ) (I.base.cost θ F c σ)
  | a => I.base.premises a

/-- The read finder's published collision bound at `t` calls (`Vocab.crBound` at `callsMulOf`, `callsAddOf`). -/
noncomputable def sampledBound (θ : ByteArray) (t : ℕ) : ℝ :=
  ((I.callsMulOf θ .sampled * t + I.callsAddOf θ .sampled : ℕ) : ℝ) / 2 ^ (256 : ℝ)

/-- **SPOT's δ at the parameters `θ`**: the deployment's session's (`Live.delta`) at the read finder's
published bound; `⊤` where the deployment runs no session, which bounds nothing there. -/
noncomputable def deltaOf (θ : ByteArray) : Subprotocol → ℕ → ℝ≥0∞
  | .sampled, t => ((I.session θ).map fun A => A.delta t (I.sampledBound θ t)).getD ⊤
  | a, t => I.base.delta θ a t

theorem deltaOf_some {θ : ByteArray} {A : Live} (h : I.session θ = some A) (t : ℕ) :
    I.deltaOf θ .sampled t = A.delta t (I.sampledBound θ t) := by
  simp [deltaOf, h]

theorem deltaOf_none {θ : ByteArray} (h : I.session θ = none) (t : ℕ) : I.deltaOf θ .sampled t = ⊤ := by
  simp [deltaOf, h]

noncomputable def lossOf : Subprotocol → ByteArray → ℝ≥0∞ → ℝ≥0∞
  | .sampled => fun _ _ => 0
  | a => I.base.loss a

theorem finderOf_calls : ∀ a θ F c σ (x : (I.finderOf a θ F c σ).S),
    (I.finderOf a θ F c σ).cost x ≤ I.callsMulOf θ a * I.base.cost θ F c σ + I.callsAddOf θ a
  | .sampled, θ, F, c, σ, x =>
    calc (I.finderOf .sampled θ F c σ).cost x
        = I.base.cost θ F c σ + (Binding.breakCalls (I.dev (Verity.Game.Tape.out _ σ x))
            (Verity.Game.Tape.out _ σ x).sent + (Verity.Game.Tape.out _ σ x).sent.blobs.size) := Nat.add_assoc _ _ _
      _ ≤ I.base.cost θ F c σ + I.base.cost θ F c σ := Nat.add_le_add_left (I.costs θ F c σ x) _
      _ = 2 * I.base.cost θ F c σ + 0 := by omega
  | .pouw, θ, F, c, σ, x => I.base.finder_calls .pouw θ F c σ x
  | .inferenceOnly, θ, F, c, σ, x => I.base.finder_calls .inferenceOnly θ F c σ x
  | .pous, θ, F, c, σ, x => I.base.finder_calls .pous θ F c σ x
  | .poub, θ, F, c, σ, x => I.base.finder_calls .poub θ F c σ x
  | .network, θ, F, c, σ, x => I.base.finder_calls .network θ F c σ x
  | .wipe, θ, F, c, σ, x => I.base.finder_calls .wipe θ F c σ x

theorem forgerOf_calls : ∀ a θ F c σ (x : (I.forgerOf a θ F c σ).S),
    (I.forgerOf a θ F c σ).cost x ≤ I.callsMulOf θ a * I.base.cost θ F c σ + I.callsAddOf θ a
  | .sampled, θ, F, c, σ, x =>
    calc (I.forgerOf .sampled θ F c σ).cost x
        = I.base.cost θ F c σ + Binding.breakCalls (I.dev (Verity.Game.Tape.out _ σ x))
            (Verity.Game.Tape.out _ σ x).sent := rfl
      _ ≤ I.base.cost θ F c σ + I.base.cost θ F c σ :=
          Nat.add_le_add_left ((Nat.le_add_right _ _).trans (I.costs θ F c σ x)) _
      _ = 2 * I.base.cost θ F c σ + 0 := by omega
  | .pouw, θ, F, c, σ, x => I.base.forger_calls .pouw θ F c σ x
  | .inferenceOnly, θ, F, c, σ, x => I.base.forger_calls .inferenceOnly θ F c σ x
  | .pous, θ, F, c, σ, x => I.base.forger_calls .pous θ F c σ x
  | .poub, θ, F, c, σ, x => I.base.forger_calls .poub θ F c σ x
  | .network, θ, F, c, σ, x => I.base.forger_calls .network θ F c σ x
  | .wipe, θ, F, c, σ, x => I.base.forger_calls .wipe θ F c σ x

theorem forgerOf_signed : ∀ a θ F c σ (x : (I.forgerOf a θ F c σ).S),
    ((I.forgerOf a θ F c σ).signed x).length ≤ I.base.signatures θ
  | .sampled, θ, F, c, σ, x => Binding.forger_signed_le I.base I.dev θ F c σ x
  | .pouw, θ, F, c, σ, x => I.base.forger_signed .pouw θ F c σ x
  | .inferenceOnly, θ, F, c, σ, x => I.base.forger_signed .inferenceOnly θ F c σ x
  | .pous, θ, F, c, σ, x => I.base.forger_signed .pous θ F c σ x
  | .poub, θ, F, c, σ, x => I.base.forger_signed .poub θ F c σ x
  | .network, θ, F, c, σ, x => I.base.forger_signed .network θ F c σ x
  | .wipe, θ, F, c, σ, x => I.base.forger_signed .wipe θ F c σ x

/-- **The deployed vocabulary but for the extractor**: the base with SPOT's names (v2's log, the binding's
read finder and forger, the further finders and the session's premises at `K`'s finders and projection, the session's
δ and no loss). -/
noncomputable def stage : Vocab d α :=
  { I.base with
    coinsRead := I.coinsReadOf
    finder := I.finderOf
    callsMul := I.callsMulOf
    callsAdd := I.callsAddOf
    finder_calls := I.finderOf_calls
    finders := I.findersOf K
    premises := I.premisesOf K
    forger := I.forgerOf
    forger_calls := I.forgerOf_calls
    forger_signed := I.forgerOf_signed
    delta := I.deltaOf
    loss := I.lossOf }

/-- A session set up at one vocabulary is set up at another with the same families, files, reading and share. -/
theorem setsUp_congr {V V' : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {A : Live}
    (hf : V'.families θ = V.families θ) (hc : V'.circuitFile θ F = V.circuitFile θ F)
    (hp : V'.publicFile θ F = V.publicFile θ F) (hr : V'.read θ F = V.read θ F)
    (he : (V'.certOf θ).ε = (V.certOf θ).ε) (h : SetsUp V θ F A) : SetsUp V' θ F A where
  params := by rw [hf]; exact h.params
  tags := h.tags
  count := h.count
  work := h.work
  stratified := h.stratified
  closure := h.closure
  own := h.own
  circuit := h.circuit.trans hc.symm
  publicFile := h.publicFile.trans hp.symm
  read := hr.trans h.read
  share := he.trans h.share

open Classical in
/-- **The extractor at the deployment's sessions**: at a session, the session's at the projected strategy
(`Live.extract`), read on the committed files' circuit (`SetsUp.read`); elsewhere the base's. -/
noncomputable def extractOf (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy ((I.stage K).game θ F c)) : (I.base.read θ F).circuit.Trace :=
  if h : ∃ A, I.sessions θ F = some A then
    cast (congrArg (fun r : Verity.FileRead => r.circuit.Trace) (I.sessions_some h.choose_spec).2.read.symm)
      ((K.live h.choose θ F c σ).extract (K.project h.choose θ F c σ))
  else I.base.extract θ F c σ

/-- **The deployed vocabulary**: `stage` with the extractor at the deployment's sessions. -/
noncomputable def vocab : Vocab d α := { I.stage K with extract := I.extractOf K }

/-- **The constructions at the stage**: `K`'s, whose strategies are the same. -/
noncomputable def liftStage : Constructions (I.stage K) where
  coins := K.coins
  project := K.project
  readFinder := K.readFinder
  forger := K.forger
  msgFinder := K.msgFinder
  recordFinder := K.recordFinder
  bindFinder := K.bindFinder
  msg_calls := K.msg_calls
  record_calls := K.record_calls
  bind_calls := K.bind_calls

/-- **Constructions over the base at the stage**: `K'`'s, whose strategies are the same. -/
noncomputable def stageOf (K' : Constructions I.base) : Constructions (I.stage K) where
  coins := K'.coins
  project := K'.project
  readFinder := K'.readFinder
  forger := K'.forger
  msgFinder := K'.msgFinder
  recordFinder := K'.recordFinder
  bindFinder := K'.bindFinder
  msg_calls := K'.msg_calls
  record_calls := K'.record_calls
  bind_calls := K'.bind_calls

/-- **The constructions at the deployed vocabulary**: `K`'s, whose strategies are the same. -/
noncomputable def lift : Constructions (I.vocab K) where
  coins := K.coins
  project := K.project
  readFinder := K.readFinder
  forger := K.forger
  msgFinder := K.msgFinder
  recordFinder := K.recordFinder
  bindFinder := K.bindFinder
  msg_calls := K.msg_calls
  record_calls := K.record_calls
  bind_calls := K.bind_calls

/-! ## The fields of `Deployed` -/

/-- **The commitment format's openings at the deployment's sessions**: wherever a session's deployed commitment opens
a gate (Flock's `commitDepL`), the base's openings (`Vocab.Opens`, a row leaf at the base's positions under
`Hash.RowLeaf`) hold of it. -/
def OpensAt : Prop :=
  ∀ θ F A, I.session θ = some A → SetsUp I.base θ F A → ∀ c, A.OpensUnder (I.base.read θ F) (I.base.Opens θ F c)

theorem fileCR {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy ((I.stage K).game θ F c)} (h : (I.stage K).finderCR .sampled θ F c σ) :
    (I.base.fileFinder θ F).Pr Verity.sha512Msg < 1 ∧ (I.base.publicFinder θ F).Pr Verity.sha512Msg < 1 := by
  have hall := List.forall_iff_forall_mem.1 (show finderCROf (I.fileSpecs θ F c ++ I.strictSpecs K θ F c) σ from h)
  have hl : finderCROf (I.fileSpecs θ F c) σ :=
    List.forall_iff_forall_mem.2 fun f hf => hall f (List.mem_append_left _ hf)
  obtain ⟨h1, h2⟩ := hl
  have lt2 : ((2 : ℕ) : ℝ) / 2 ^ (256 : ℝ) < 1 := budget_lt (by norm_num)
  exact ⟨lt_of_le_of_lt h1 lt2, lt_of_le_of_lt h2 lt2⟩

/-- **The two-continuation finders' premises**: at the deployment's session, where the further finders meet theirs,
each two-continuation finder `K` builds from the strategy meets the strict birthday bound (`strictSpecs`). -/
theorem strictCR {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {A : Live}
    {σ : Verity.Game.Strategy ((I.stage K).game θ F c)} (hA : I.sessions θ F = some A)
    (h : (I.stage K).finderCR .sampled θ F c σ) :
    (∀ j, Claims.StrictBirthday Verity.sha512Msg (K.msgFinder A θ F c σ j)) ∧
      Claims.StrictBirthday Verity.sha512Msg (K.recordFinder A θ F c σ) ∧
      Claims.StrictBirthday Verity.sha512Msg (K.bindFinder A θ F c σ) := by
  have hall := List.forall_iff_forall_mem.1 (show finderCROf (I.fileSpecs θ F c ++ I.strictSpecs K θ F c) σ from h)
  have hs : I.strictSpecs K θ F c =
      (List.finRange A.rounds).map (I.msgSpec K A θ F c) ++ [I.recordSpec K A θ F c, I.bindSpec K A θ F c] := by
    unfold strictSpecs
    rw [hA]
  refine ⟨fun j => hall (I.msgSpec K A θ F c j) (List.mem_append_right _ ?_),
    hall (I.recordSpec K A θ F c) (List.mem_append_right _ ?_),
    hall (I.bindSpec K A θ F c) (List.mem_append_right _ ?_)⟩
  · rw [hs]
    exact List.mem_append_left _ (List.mem_map.2 ⟨j, List.mem_finRange j, rfl⟩)
  · rw [hs]
    exact List.mem_append_right _ (by simp)
  · rw [hs]
    exact List.mem_append_right _ (by simp)

/-- **`Deployed` at the deployed vocabulary**, from the commitment format's openings, at constructions whose read finder
and forger are the binding's (`Binding.withBinding`, `Constructions.assembled`): SPOT's per-finder premises are
`error`'s condition at this strategy (`strictSpecs`, `premisesOf`, `fileSpecs`). -/
theorem deployed (hKr : ∀ θ F c σ, K.readFinder θ F c σ = Binding.readFinder I.base I.dev θ F c σ)
    (hKf : ∀ θ F c σ, K.forger θ F c σ = Binding.forger I.base I.dev θ F c σ) (hO : I.OpensAt) :
    Deployed (I.vocab K) (I.lift K) I.sessions where
  setsUp θ F A h := setsUp_congr (V := I.base) rfl rfl rfl rfl rfl (I.sessions_some h).2
  digests θ := digestsAt_of (I.vocab K) I.programDigest I.publicDigest I.read θ
  coinsRead _ := rfl
  finder θ F c σ := (hKr θ F c σ).symm
  forger θ F c σ := (hKf θ F c σ).symm
  loss _ _ := rfl
  extract θ F A h c σ := by
    have he : ∃ A, I.sessions θ F = some A := ⟨A, h⟩
    have hc : he.choose = A := Option.some.inj (he.choose_spec.symm.trans h)
    show HEq (I.extractOf K θ F c σ) ((K.live A θ F c σ).extract (K.project A θ F c σ))
    unfold extractOf
    rw [dif_pos he]
    refine (cast_heq _ _).trans ?_
    rw [hc]
    exact HEq.rfl
  opens θ F A h c := hO θ F A (I.sessions_some h).1 (I.sessions_some h).2 c
  finderCR θ F A h c σ hcr hpr := by
    obtain ⟨hm, hr, hb⟩ := I.strictCR K h hcr
    obtain ⟨h1, h2⟩ := I.fileCR K hcr
    exact ⟨hpr A h, hm, hr, hb, strict_of_zero (pr_unit h1), strict_of_zero (pr_unit h2)⟩
  premises _ _ _ _ := Iff.rfl
  delta θ F A h t := (I.deltaOf_some (I.sessions_some h).1 t).symm.le

/-! ## `Covers` -/

/-- **The registered files pass the file check wherever some files do**, at parameters naming the one-stage form:
where some committed files hold a file with each digest the parameters hold, as the verifier's check of an accepting
play's files finds them (`Vocab.Verdict`), the registered files hold one too. No check of the verifier reads the
registered files: this is the registration's, whose parameters hold the SHA-512s of the files it registers. -/
def RegisteredFiles : Prop :=
  ∀ θ F, (∃ p, ParamsAt θ p) → I.base.Registered θ F → I.base.Registered θ (I.base.registered θ)

/-- **Registered files built from the parameters' digests pass the file check wherever some files do**
(`registeredOf`). -/
theorem registeredFiles_of (h : ∀ θ, I.base.registered θ = registeredOf θ) : I.RegisteredFiles := by
  intro θ F _ hF
  unfold Vocab.Registered at hF ⊢
  obtain ⟨⟨f, -, hc⟩, ⟨g, -, hp⟩⟩ := hF
  rw [I.programDigest] at hc ⊢
  rw [I.publicDigest] at hp ⊢
  rw [h, registeredOf]
  obtain ⟨x, hx, hxe⟩ := fileWith_mem ⟨f, hc⟩
  obtain ⟨y, hy, hye⟩ := fileWith_mem ⟨g, hp⟩
  exact ⟨⟨x, Array.mem_append_left _ (Array.mem_append_left _ hx), hxe⟩,
    ⟨y, Array.mem_append_left _ (Array.mem_append_right _ hy), hye⟩⟩

/-- **v2's session at the registered files' files**: at parameters naming the one-stage form where the deployment runs
a session, committed files that name the registered circuit and public files, and that some play's sampled verdict
accepts, are files v2 sets that session up on. -/
def SessionAt (k : Verity.Registration.Composite) : Prop :=
  ∀ θ F A, I.session θ = some A → (∃ p, ParamsAt θ p) →
    I.base.circuitFile θ F = I.base.circuitFile θ (I.base.registered θ) →
    I.base.publicFile θ F = I.base.publicFile θ (I.base.registered θ) →
    (∃ c o, Verity.Game.Reaches (I.base.game θ F c) o ∧ I.base.Verdict k .sampled o) → SetsUp I.base θ F A

/-- A play that some strategy wins with nonzero probability is won on some tape. -/
theorem exists_tape_of_prob_ne {β : Type} {p : β → Prop} {g : Verity.Game β} {σ : Verity.Game.Strategy g}
    (h : Verity.Game.prob p g σ ≠ 0) : ∃ t : Verity.Game.Tape g σ, p (Verity.Game.Tape.out g σ t) := by
  by_contra hn
  simp only [not_exists] at hn
  refine h (le_antisymm ?_ zero_le)
  calc Verity.Game.prob p g σ = Verity.Game.prCoin fun t => p (Verity.Game.Tape.out g σ t) :=
        Verity.Game.prob_eq_prCoin _ _ σ
    _ ≤ Verity.Game.prCoin fun _ : Verity.Game.Tape g σ => False := Verity.Game.prCoin_mono fun t ht => hn t ht
    _ = 0 := Verity.Game.prCoin_false

/-- **`Covers` at the deployed vocabulary**: where the deployment runs no session, the published δ is `⊤`
(`deltaOf`); where it runs one that v2 doesn't set up on the committed files, an accepting play's files pass the file
check, so under the further finders' premises they name the registered files (`namedFile_eq`), where v2 sets the
session up (`SessionAt`). -/
theorem covers (k : Verity.Registration.Composite) (hRg : I.RegisteredFiles) (hSa : I.SessionAt k) :
    Covers (I.vocab K) k I.sessions := by
  intro θ F hS hP c σ hcr
  rcases hA : I.session θ with _ | A
  · exact le_top.trans_eq (I.deltaOf_none hA _).symm
  refine le_of_eq_of_le ?_ zero_le
  by_contra hne
  obtain ⟨t, hv⟩ := exists_tape_of_prob_ne hne
  have hreach : Verity.Game.Reaches (I.base.game θ F c) (Verity.Game.Tape.out _ σ t) :=
    Verity.Game.Tape.reaches_out _ σ t
  have hv' : I.base.Verdict k .sampled (Verity.Game.Tape.out _ σ t) := hv
  obtain ⟨hθ, hF, -⟩ := Binding.play_reaches hreach
  have hreg : I.base.Registered θ F := by
    have h1 := hv'.1
    rw [hθ, hF] at h1
    exact h1
  have hR := hRg θ F hP hreg
  obtain ⟨h1, h2⟩ := I.fileCR K hcr
  have hs := hSa θ F A hA hP (namedFile_eq hreg.1 hR.1 h1) (namedFile_eq hreg.2 hR.2 h2) ⟨c, _, hreach, hv'⟩
  rw [I.sessions_of hA hs] at hS
  cases hS

/-! ## The other subprotocols' targets are the base's -/

theorem pouwSound_of_base (k : Verity.Registration.Composite) (h : I.base.PouwSound k) : (I.vocab K).PouwSound k :=
  h

theorem inferenceOnlySound_of_base (k : Verity.Registration.Composite) (h : I.base.InferenceOnlySound k) :
    (I.vocab K).InferenceOnlySound k :=
  h

/-- **The coupling at the deployed vocabulary is the stage's**: the period's game, the verdict and the constructions are
the same. -/
theorem couplingTarget_lift (k : Verity.Registration.Composite) {A : Live} {θ : ByteArray} {F : Array ByteArray}
    (h : CouplingTarget (I.stage K) k (I.liftStage K) A θ F) : CouplingTarget (I.vocab K) k (I.lift K) A θ F :=
  h

/-! ## The coupling at the deployment -/

/-- **The constructions at the deployment**: `K₀`'s assembled (`Constructions.assembled`) at the deployed device's
history (`dev`), composite `k`, the readings and decoders at the session's layout, the family v2's parameters name
(`labelAt`), the digest `afterAt` and the session's bytes as `MAX_COINS` fresh frames (`srcCoinsAt`), as
`couplingTargetAt_counts` reads them. -/
noncomputable def constructionsAt (K₀ : Constructions I.base) (k : Verity.Registration.Composite)
    (reg : (A : Live) → Option Claims.Msg → A.Reg)
    (record : Live → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Live → ByteArray → Array ByteArray → ByteArray → Claims.Msg) : Constructions I.base :=
  K₀.assembled fieldEnc I.dev k
    (fun A θ F c => readingAt A.tables A.layout (record A θ F c) (recordDigest A θ F c))
    (fun A _ _ _ _ => decodersAt (reg A) A.layout) (fun θ _ _ _ => labelAt θ) (afterAt I.base k)
    (srcCoinsAt _root_.Spot.Challenger.MAX_COINS)

/-- **The device's log opens only the record's rounds**, at composite `k`: at every play on files v2 sets one of the
deployment's sessions up on, which `k`'s sampled verdict accepts and whose log v2 reads, a record v2 parses from what
the prover sent holds each round the log opens (`OpenedRecorded`). -/
def LogOpensRecorded (k : Verity.Registration.Composite) : Prop :=
  ∀ θ F A, I.sessions θ F = some A →
  ∀ c (σ : Verity.Game.Strategy (I.base.game θ F c)) (x : Verity.Game.CoinTape (I.base.game θ F c) σ),
    I.base.Verdict k .sampled (Verity.Game.CoinTape.run (I.base.game θ F c) σ x) →
    Binding.Logged Binding.readsV2 (Verity.Game.CoinTape.run (I.base.game θ F c) σ x) →
    ∀ text Jr, sentRecord (Verity.Game.CoinTape.run (I.base.game θ F c) σ x).sent = some text →
      Flock.JsonParse.parse text = .ok Jr →
      OpenedRecorded Jr (Verity.Game.CoinTape.run (I.base.game θ F c) σ x).log

/-- **The coupling at each of the deployment's sessions** (`couplingTargetAt_counts`, at the stage): at the
constructions `constructionsAt` assembles, the deployed device's signer is the vocabulary's (`Instance.signer`), the
session is one v2 sets up (`sessions_some`), v2's parameters name the vocabulary's digests (`digestsAt_of`) and the
session reads `MAX_COINS` frames' bytes (`lb_of_sessions`), so what is left is the device's log opening only the
record's rounds (`LogOpensRecorded`). -/
theorem couplingTarget_at (K₀ : Constructions I.base) (k : Verity.Registration.Composite)
    (reg : (A : Live) → Option Claims.Msg → A.Reg)
    (record : Live → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Live → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hO : I.LogOpensRecorded k) :
    ∀ θ F A, I.sessions θ F = some A →
      CouplingTarget (I.stage (I.constructionsAt K₀ k reg record recordDigest)) k
        (I.liftStage (I.constructionsAt K₀ k reg record recordDigest)) A θ F := fun θ F A h =>
  couplingTargetAt_counts (V := I.stage (I.constructionsAt K₀ k reg record recordDigest))
    (I.stageOf (I.constructionsAt K₀ k reg record recordDigest) K₀) I.deployment reg record recordDigest
    (I.signer θ) (setsUp_congr (V := I.base) rfl rfl rfl rfl rfl (I.sessions_some h).2)
    (digestsAt_of (I.stage (I.constructionsAt K₀ k reg record recordDigest)) I.programDigest I.publicDigest I.read θ)
    (I.lb_of_sessions h) (hO θ F A h)

end Instance

/-- **Soundness at remote's verifier at the deployed vocabulary**: remote's compiled verifier accepts files
that passed the commit phase's check while one of its three claims fails with probability at most their errors summed,
at the vocabulary `Instance.vocab` builds from the deployment and `K`, wherever the commitment format's openings
(`OpensAt`), the registered files' file check (`RegisteredFiles`), v2's session at the registered files (`SessionAt`),
the coupling at each session (`CouplingTarget`, at the stage), and the base's PoUW and inference-only targets hold. The premises at the finders `K` builds from a strategy are `error`'s condition
at that strategy. -/
theorem remote_soundness_deployed (I : Instance d α) (K : Constructions I.base)
    (hKr : ∀ θ F c σ, K.readFinder θ F c σ = Binding.readFinder I.base I.dev θ F c σ)
    (hKf : ∀ θ F c σ, K.forger θ F c σ = Binding.forger I.base I.dev θ F c σ) (hO : I.OpensAt)
    (hRg : I.RegisteredFiles) (hSa : I.SessionAt .remote)
    (hC : ∀ θ F A, I.sessions θ F = some A → CouplingTarget (I.stage K) .remote (I.liftStage K) A θ F)
    (hW : I.base.PouwSound .remote) (hI : I.base.InferenceOnlySound .remote) :
    (I.vocab K).SoundnessAt .remote :=
  Verity.remote_soundness_of_targets (I.vocab K) (I.deployed K hKr hKf hO) (I.covers K .remote hRg hSa)
    (fun θ F A h => I.couplingTarget_lift K .remote (hC θ F A h)) (I.pouwSound_of_base K .remote hW)
    (I.inferenceOnlySound_of_base K .remote hI)

/-- **`remote_soundness_deployed` at the binding's constructions** (`Binding.withBinding`): any constructions with
the binding's read finder and forger, so those two premises hold by definition. -/
theorem remote_soundness_deployed_withBinding (I : Instance d α) (K₀ : Constructions I.base) (hO : I.OpensAt)
    (hRg : I.RegisteredFiles) (hSa : I.SessionAt .remote)
    (hC : ∀ θ F A, I.sessions θ F = some A →
      CouplingTarget (I.stage (Binding.withBinding K₀ I.dev)) .remote (I.liftStage (Binding.withBinding K₀ I.dev)) A θ F)
    (hW : I.base.PouwSound .remote) (hI : I.base.InferenceOnlySound .remote) :
    (I.vocab (Binding.withBinding K₀ I.dev)).SoundnessAt .remote :=
  remote_soundness_deployed I (Binding.withBinding K₀ I.dev) (fun _ _ _ _ => rfl) (fun _ _ _ _ => rfl) hO hRg hSa
    hC hW hI

/-- **`remote_soundness_deployed` at the deployment's constructions** (`Instance.constructionsAt`): the coupling at
each session is `couplingTarget_at`'s, so of it what is left is the device's log opening only the record's rounds
(`LogOpensRecorded`). -/
theorem remote_soundness_deployedAt (I : Instance d α) (K₀ : Constructions I.base)
    (reg : (A : Live) → Option Claims.Msg → A.Reg)
    (record : Live → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Live → ByteArray → Array ByteArray → ByteArray → Claims.Msg) (hO : I.OpensAt)
    (hRg : I.RegisteredFiles) (hSa : I.SessionAt .remote)
    (hOR : I.LogOpensRecorded .remote)
    (hW : I.base.PouwSound .remote) (hI : I.base.InferenceOnlySound .remote) :
    (I.vocab (I.constructionsAt K₀ .remote reg record recordDigest)).SoundnessAt .remote :=
  remote_soundness_deployed I _ (fun _ _ _ _ => rfl) (fun _ _ _ _ => rfl) hO hRg hSa
    (I.couplingTarget_at K₀ .remote reg record recordDigest hOR) hW hI

/-! ## An instance: v2's reading on the idle device -/

namespace Instance.Honest

/-- Any circuit's schedule as one empty block, as `Idle.blocks` is the empty file's. -/
def blocks (C : Verity.Circuit) : Verity.Blocks C 1 where
  block _ := ⟨⟨0, 0⟩, ∅⟩
  ordered _ _ _ := Nat.le_refl 0
  disjoint _ _ _ := Finset.disjoint_empty_left _

/-- `Idle.wipe` at `blocks C`: it fills nothing, nothing leaves the device, and the challenger accepts nothing. -/
def wipe (C : Verity.Circuit) : Verity.Wipe Verity.Idle.device (blocks C) 0 Unit Unit where
  fill := 0
  fill_le := Nat.zero_le _
  egress := 0
  carry _ _ := 0
  filled σ _ := (σ, fun i => i.elim0)
  accepts _ _ _ := False
  after σ _ _ := σ

/-- `Idle.ofRun` at `wipe C`. -/
def ofRun (C : Verity.Circuit) {O : Type} (G : Verity.Game O) : (wipe C).OfRun G where
  S_φ := 0
  hold _ := ⟨fun _ => false, fun _ => false⟩
  keep _ _ _ i := i.elim0
  view _ := (fun i => i.elim0, (), ())

/-- **The idle deployment**: a run with no key, coins or ledger, which opens and closes no ledger line, samples nothing
and signs every message with the empty signature. -/
def deployment : Device.Deployment where
  run _ _ _ _ _ := ⟨ByteArray.empty, "", "", "", 0, [], fun _ => ByteArray.empty, fun _ => ByteArray.empty, none,
    fun _ => 0, []⟩
  opens _ _ := []
  closes _ _ _ _ _ := []
  ledgerLines _ := 0
  ledger_le _ _ _ _ _ := Nat.le_refl 0
  draws _ := 0
  sign _ _ _ := []

/-- The idle deployment's signer, at no coin family. -/
def signer (θ : ByteArray) : Signer := deployment.signer (fun _ => []) θ

/-- The idle deployment's history at each play (`Device.dev`). -/
noncomputable abbrev dev : Exchange → Properties.Challenger.History := Device.dev (deployment.setupOf fun _ => [])

/-- **The declared calls**: the most, over the play's tapes, of what the device's reading of what the prover sent
hashes (`Binding.breakCalls`) and one per blob it sent. -/
noncomputable def cost (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (Verity.play [] (signer θ) θ F c)) : ℕ :=
  Finset.univ.sup fun t : Verity.Game.Tape (Verity.play [] (signer θ) θ F c) σ =>
    Binding.breakCalls (dev (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
      (Verity.Game.Tape.out _ σ t).sent.blobs.size

/-- **The idle deployment at v2's reading**: `Idle.vocab δ 1` with the idle deployment's signer (`signer`) and its
signature count, v2's reading of the committed files and its digests, the files the digests name as the registered
files (`registeredOf`), the declared calls `cost`, and every name that
reads the committed circuit or a strategy of the play at them. -/
noncomputable def base (δ : ℝ≥0∞) : Vocab Verity.Idle.device Unit :=
  { Verity.Idle.vocab δ 1 with
    signer := signer
    read := readOf
    programDigest := programDigestOf
    publicDigest := publicDigestOf
    registered := registeredOf
    place := fun θ F _ g => placeOfRead (readOf θ F) g
    extract := fun _ _ _ _ _ => 0
    finder := fun _ _ _ _ _ => Verity.Idle.finder
    cost := cost
    finder_calls := fun _ _ _ _ _ _ => Nat.zero_le _
    finders := fun _ _ _ _ => []
    premises := fun _ _ _ _ _ => True
    forger := fun _ _ _ _ _ => Verity.Idle.forger
    forger_calls := fun _ _ _ _ _ _ => Nat.zero_le _
    signatures := deployment.signatures
    signs_le := deployment.signs_le fun _ => []
    forger_signed := fun _ _ _ _ _ _ => Nat.zero_le _
    computed := fun _ _ _ _ _ => ∅
    world := fun _ _ _ _ _ => Verity.Idle.world
    world_window := fun _ _ _ _ _ => rfl
    world_capacity := fun _ _ _ _ _ => rfl
    theta := fun _ _ => ∅
    theta_input := fun _ _ _ h => absurd h (Finset.notMem_empty _)
    schedule := fun θ F => blocks (readOf θ F).circuit
    wipe := fun θ F => wipe (readOf θ F).circuit
    wipeRun := fun θ F c _ => ofRun (readOf θ F).circuit (Verity.play [] (signer θ) θ F c)
    honest := fun θ F c =>
      Verity.requestThen [] (signer θ) θ F c .empty ⟨.empty, fun _ => ⟨(.empty, #[]), ⟨⟩⟩⟩ }

theorem costs (δ : ℝ≥0∞) : CostsRead (base δ) dev := fun θ F c σ t =>
  Finset.le_sup (f := fun t : Verity.Game.Tape (Verity.play [] (signer θ) θ F c) σ =>
    Binding.breakCalls (dev (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
      (Verity.Game.Tape.out _ σ t).sent.blobs.size) (Finset.mem_univ t)

/-- **The instance at the deployment's session `session`**: `base δ` at the idle deployment, whose signer is the
deployment's and whose reading and digests are v2's by definition. -/
noncomputable def inst (δ : ℝ≥0∞) (session : ByteArray → Option Live) : Instance Verity.Idle.device Unit where
  base := base δ
  deployment := deployment
  signer _ := rfl
  read _ _ := rfl
  programDigest _ := rfl
  publicDigest _ := rfl
  place _ _ _ _ := rfl
  costs := costs δ
  coins _ _ _ h := by cases h
  session := session

theorem sessions_none (δ : ℝ≥0∞) (θ : ByteArray) (F : Array ByteArray) :
    (inst δ fun _ => none).sessions θ F = none := rfl

/-- At no session the openings hold of every session, there being none. -/
theorem opensAt_none (δ : ℝ≥0∞) : (inst δ fun _ => none).OpensAt := fun _ _ _ h => by simp [inst] at h

/-- The honest instance's registered files, the parameters' digests' (`registeredOf`), pass the file check wherever
some files do. -/
theorem registeredFiles (δ : ℝ≥0∞) (session : ByteArray → Option Live) : (inst δ session).RegisteredFiles :=
  (inst δ session).registeredFiles_of fun _ => rfl

/-- At no session the device's log opens only the record's rounds at every session, there being none. -/
theorem logOpensRecorded_none (δ : ℝ≥0∞) (k : Verity.Registration.Composite) :
    (inst δ fun _ => none).LogOpensRecorded k := fun θ F A h => by
  rw [sessions_none] at h
  cases h

/-- At `base ⊤` every subprotocol's error is `⊤`. -/
theorem error_top (a : Subprotocol) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy ((base ⊤).game θ F c)) : (base ⊤).error a θ F c σ = ⊤ := by
  unfold Vocab.error Accounting.condError
  split_ifs <;> simp [base, Verity.Idle.vocab]

/-- At `base ⊤` every subprotocol meets its target at every composite. -/
theorem target_top (k : Verity.Registration.Composite) (a : Subprotocol) : (base ⊤).Target k a := by
  intro θ F c σ
  rw [error_top]
  exact le_top

end Instance.Honest

end Verity.Sampled
