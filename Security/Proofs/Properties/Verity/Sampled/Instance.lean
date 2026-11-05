module

public import Proofs.Properties.Verity.Assembled
public import Proofs.Properties.Verity.Sampled.Binding
public import Proofs.Flock.Recursive.TopFinder

/-!
# Sampled proofs' vocabulary as deployed

The vocabulary the leg reads (`Deployed`, `Covers`) built from the other subprotocols' names over the deployed device
(`Instance.base`), so `sampledProofsSound_of_targets` and `remote_soundness_of_targets` apply to it with every field
of `Deployed` and `Covers` either proved or one of the named `Prop`s below.

* **The base** (`Instance`): a vocabulary whose shared names are v2's (its reading of the committed files is `readOf`,
  its digests `programDigestOf` and `publicDigestOf`), whose declared calls cover the device's reading of what the
  prover sent (`CostsRead`, what the read finder and the forger spend beside the strategy), the device's history at each
  play, and the deployment's session at each parameters (`Instance.session`), the bound's parameters fixed by `θ`.
* **The sampled names** (`Instance.stage`): v2's log (`coinsRead`), the binding's read finder and forger
  (`Binding.readFinder`, `Binding.forger`) at `2t` calls, the file finders and the tables' finder (`fileSpecs`) as the
  further finders, no loss, and the session's δ (`Live.delta`) at the deployment's session. The other subprotocols'
  names are the base's, so their targets are the base's (`pouwSound_of_base`, `inferenceOnlySound_of_base`).
* **The sessions** (`Instance.sessions`): the deployment's session wherever v2 sets it up on the committed files
  (`SetsUp`), and the extractor the session's at the projected strategy (`Instance.vocab`).
* **What stays open**: the commitment format's openings (`OpensAt`), the strict premises the session's finders and the
  two-continuation finders read (`StrictAt`), which `Verity.Game.FinderSpec`'s linear form can't carry, the registered
  files' file check (`RegisteredFiles`) and v2's session at the registered files (`SessionAt`).
-/

@[expose] public section

namespace Verity.Sampled

open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The file finders and the tables' finder -/

/-- **The committed file holding a pinned table**: the first of `F` whose SHA-512's hex is `sha`, as `tablesOf` looks
it up on the committed files (`tablesOf_ofArray`). -/
def tableFile (sha : String) (F : Array ByteArray) : Option ByteArray :=
  F.find? fun b => Flock.hexOf (Flock.Sha512.hash b) == sha

theorem tablesOf_ofArray (F : Array ByteArray) :
    _root_.SampledProofs.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray F) =
      Flock.MUFU_TABLES.filterMap fun t => (tableFile t.sha512 F).map fun b => (t.name, b) := by
  unfold _root_.SampledProofs.OneStageZk.tablesOf _root_.Core.Blobs.ofArray tableFile
  congr 1
  funext t
  rw [Array.find?_map, Option.map_map]
  rfl

/-- **A pinned table's two files**: the table the committed files `F` and the registered files `R` hold under the
digest `sha`, where both hold one and they differ. -/
noncomputable def tablePair (F R : Array ByteArray) (sha : String) : Option (Claims.Msg × Claims.Msg) :=
  match tableFile sha F, tableFile sha R with
  | some a, some b => if a = b then none else some (a.data.toList, b.data.toList)
  | _, _ => none

/-- **The tables' finder**: on its one coin it hashes each pinned table the committed files and the registered files
hold, and outputs the first two that differ, a SHA-512 collision (`tablePair_collision`). -/
noncomputable def tableFinder (F R : Array ByteArray) : Claims.Finder where
  S := Unit
  out _ := Flock.MUFU_TABLES.findSome? fun t => tablePair F R t.sha512
  cost _ := 2 * Flock.MUFU_TABLES.length

/-- **The committed files' tables agree with the registered files'** wherever both hold the pinned table. -/
def TablesAgree (F R : Array ByteArray) : Prop :=
  ∀ t ∈ Flock.MUFU_TABLES, ∀ a b, tableFile t.sha512 F = some a → tableFile t.sha512 R = some b → a = b

theorem tableFile_hash {sha : String} {F : Array ByteArray} {a : ByteArray} (h : tableFile sha F = some a) :
    Flock.hexOf (Flock.Sha512.hash a) = sha := by
  unfold tableFile at h
  exact beq_iff_eq.1 (Array.find?_some h)

theorem tablePair_collision {F R : Array ByteArray} {sha : String} {u v : Claims.Msg}
    (h : tablePair F R sha = some (u, v)) : Claims.Collision Verity.sha512Msg u v := by
  unfold tablePair at h
  cases ha : tableFile sha F with
  | none => rw [ha] at h; cases h
  | some a =>
    cases hb : tableFile sha R with
    | none => rw [ha, hb] at h; cases h
    | some b =>
      rw [ha, hb] at h
      dsimp only at h
      by_cases hab : a = b
      · rw [if_pos hab] at h; cases h
      rw [if_neg hab] at h
      cases h
      have hh : Flock.Sha512.hash a = Flock.Sha512.hash b :=
        Flock.SecurityProofs.TopFinder.hexOf_inj ((tableFile_hash ha).trans (tableFile_hash hb).symm)
      refine ⟨fun e => hab (ByteArray.ext (Array.ext' e)), ?_⟩
      simp only [Verity.sha512Msg, Binding.bytes_ofList, hh]

/-- A one-coin finder's success is `0` or `1`, so below `1` it is `0`. -/
theorem pr_unit {p : Unit → Prop} (h : Claims.Pr p < 1) : Claims.Pr p = 0 := by
  classical
  unfold Claims.Pr at *
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
    have hg := of_decide_eq_true (Array.find?_some hfind)
    exact hg

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

theorem tablesAgree_of {F R : Array ByteArray} (hp : (tableFinder F R).Pr Verity.sha512Msg < 1) : TablesAgree F R := by
  have h0 : (tableFinder F R).Pr Verity.sha512Msg = 0 := pr_unit hp
  have hnone : (tableFinder F R).out () = none := by
    cases ho : (tableFinder F R).out () with
    | none => rfl
    | some uv =>
      have h1 : (tableFinder F R).Pr Verity.sha512Msg = 1 := by
        unfold Claims.Finder.Pr Claims.Pr
        obtain ⟨t, -, ht⟩ := List.exists_of_findSome?_eq_some ho
        have hs : ∀ s : (tableFinder F R).S,
            ∃ u v, (tableFinder F R).out s = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v :=
          fun s => by cases s; exact ⟨uv.1, uv.2, ho, tablePair_collision ht⟩
        classical
        rw [Finset.filter_true_of_mem fun s _ => hs s, Finset.card_univ]
        exact div_self (Nat.cast_ne_zero.2 Fintype.card_ne_zero)
      rw [h1] at h0
      exact absurd h0 one_ne_zero
  intro t ht a b ha hb
  have hn := List.findSome?_eq_none_iff.1 hnone t ht
  by_contra hab
  simp only [tablePair, ha, hb, if_neg hab, reduceCtorEq] at hn

/-! ## The base -/

/-- **The base's declared calls cover the device's reading**: on every tape of the period's game, the strings `findBreak`
hashes in the play's records (`Binding.breakCalls`) and one call per blob the prover sent are at most the strategy's
declared calls, so the read finder and the forger spend at most twice them. -/
def CostsRead (V : Vocab d α) (dev : Exchange → Properties.Challenger.History) : Prop :=
  ∀ θ F c (σ : Verity.Game.Strategy (V.game θ F c)) (t : Verity.Game.Tape (V.game θ F c) σ),
    Binding.breakCalls (dev (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
      (Verity.Game.Tape.out _ σ t).sent.blobs.size ≤ V.cost θ F c σ

/-- **The deployment, before sampled proofs' names**: the other subprotocols' vocabulary over the deployed device
(`base`), whose reading of the committed files and digests are v2's (`read`, `programDigest`, `publicDigest`) and whose
declared calls cover the device's reading (`costs`); the device's history at each play (`dev`); and the session the
deployment runs at each parameters (`session`), whose bound's parameters `θ` fixes. -/
structure Instance (d : Accounting.Device ι) (α : Type) [Fintype α] where
  base : Vocab d α
  dev : Exchange → Properties.Challenger.History
  read : ∀ θ F, base.read θ F = readOf θ F
  programDigest : ∀ θ, base.programDigest θ = programDigestOf θ
  publicDigest : ∀ θ, base.publicDigest θ = publicDigestOf θ
  costs : CostsRead base dev
  session : ByteArray → Option Live

namespace Instance

variable (I : Instance d α)

/-- **Sampled proofs' further finders as deployed**: the circuit file's and the public file's (`Vocab.fileFinder`,
`Vocab.publicFinder`) at two calls each, and the tables' (`tableFinder`) at two per pinned table. -/
noncomputable def fileSpecs (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    List (FinderSpec (I.base.game θ F c) (I.base.cost θ F c)) :=
  [{ finder := fun _ => I.base.fileFinder θ F, budget := fun _ => 2, calls := fun _ _ => Nat.le_refl 2 },
   { finder := fun _ => I.base.publicFinder θ F, budget := fun _ => 2, calls := fun _ _ => Nat.le_refl 2 },
   { finder := fun _ => tableFinder F (I.base.registered θ), budget := fun _ => 2 * Flock.MUFU_TABLES.length,
     calls := fun _ _ => Nat.le_refl _ }]

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

noncomputable def findersOf : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    List (FinderSpec (I.base.game θ F c) (I.base.cost θ F c))
  | .sampled => I.fileSpecs
  | a => I.base.finders a

/-- **Sampled proofs' δ at the parameters `θ`**: the deployment's session's (`Live.delta`) at the read finder's
published bound; `0` where the deployment runs no session, where no play is covered by one (`Covers`). -/
noncomputable def deltaOf (θ : ByteArray) : Subprotocol → ℕ → ℝ≥0∞
  | .sampled, t => ((I.session θ).map fun A => A.delta t (I.base.crBound θ .sampled t)).getD 0
  | a, t => I.base.delta θ a t

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

/-- **The deployed vocabulary but for the extractor**: the base with sampled proofs' names (v2's log, the binding's
read finder and forger, the further finders, the session's δ and no loss). -/
noncomputable def stage : Vocab d α :=
  { I.base with
    coinsRead := I.coinsReadOf
    finder := I.finderOf
    callsMul := I.callsMulOf
    callsAdd := I.callsAddOf
    finder_calls := I.finderOf_calls
    finders := I.findersOf
    forger := I.forgerOf
    forger_calls := I.forgerOf_calls
    forger_signed := I.forgerOf_signed
    delta := I.deltaOf
    loss := I.lossOf }

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

theorem sessions_of {θ : ByteArray} {F : Array ByteArray} {A : Live} (hA : I.session θ = some A)
    (hs : SetsUp I.base θ F A) : I.sessions θ F = some A := by
  unfold sessions
  rw [hA]
  dsimp only [Option.bind]
  rw [if_pos hs]

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

variable (K : Constructions I.stage)

open Classical in
/-- **The extractor at the deployment's sessions**: at a session, the session's at the projected strategy
(`Live.extract`), read on the committed files' circuit (`SetsUp.read`); elsewhere the base's. -/
noncomputable def extractOf (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (I.stage.game θ F c)) : (I.base.read θ F).circuit.Transcript :=
  if h : ∃ A, I.sessions θ F = some A then
    cast (congrArg (fun r : Verity.FileRead => r.circuit.Transcript) (I.sessions_some h.choose_spec).2.read.symm)
      ((K.live h.choose θ F c σ).extract (K.project h.choose θ F c σ))
  else I.base.extract θ F c σ

/-- **The deployed vocabulary**: `stage` with the extractor at the deployment's sessions. -/
noncomputable def vocab : Vocab d α := { I.stage with extract := I.extractOf K }

/-- **The constructions at the deployed vocabulary**: `K`'s, whose strategies are the same. -/
noncomputable def lift : Constructions (I.vocab K) where
  coins := K.coins
  project := K.project
  readFinder := K.readFinder
  forger := K.forger
  msgFinder := K.msgFinder
  recordFinder := K.recordFinder
  msg_calls := K.msg_calls
  record_calls := K.record_calls

/-! ## The fields of `Deployed` -/

/-- **The commitment format's openings at the deployment's sessions**: wherever a session's deployed commitment opens
a gate (C-Flock's `commitDepL`), the base's openings (`Vocab.Opens`, a row leaf at the base's positions under
`Hash.RowLeaf`) hold of it. -/
def OpensAt : Prop :=
  ∀ θ F A, I.session θ = some A → SetsUp I.base θ F A → ∀ c, A.OpensUnder (I.base.read θ F) (I.base.Opens θ F c)

/-- **The strict premises at the deployment's sessions**: where the further finders meet their premises, the session's
finders meet theirs (`Live.FinderCR`) and each two-continuation finder the strict birthday bound. `FinderSpec.CR` is
the linear `ecr` form only, so these aren't among `Vocab.finderCR`'s. -/
def StrictAt : Prop :=
  ∀ θ F A, I.sessions θ F = some A → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (I.stage.game θ F c)),
    I.stage.finderCR .sampled θ F c σ →
      (K.live A θ F c σ).FinderCR (K.project A θ F c σ) (I.base.cost θ F c σ) ∧
        (∀ j, Claims.StrictBirthday Verity.sha512Msg (K.msgFinder A θ F c σ j)) ∧
        Claims.StrictBirthday Verity.sha512Msg (K.recordFinder A θ F c σ)

theorem fileCR {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {σ : Verity.Game.Strategy (I.stage.game θ F c)}
    (h : I.stage.finderCR .sampled θ F c σ) :
    (I.base.fileFinder θ F).Pr Verity.sha512Msg < 1 ∧ (I.base.publicFinder θ F).Pr Verity.sha512Msg < 1 ∧
      (tableFinder F (I.base.registered θ)).Pr Verity.sha512Msg < 1 := by
  have hl : finderCROf (I.fileSpecs θ F c) σ := h
  obtain ⟨h1, h2, h3⟩ := hl
  have lt2 : ((2 : ℕ) : ℝ) / 2 ^ (256 : ℝ) < 1 := budget_lt (by norm_num)
  have hT : Flock.MUFU_TABLES.length = 4 := rfl
  have ltT : ((2 * Flock.MUFU_TABLES.length : ℕ) : ℝ) / 2 ^ (256 : ℝ) < 1 := budget_lt (by rw [hT]; norm_num)
  exact ⟨lt_of_le_of_lt h1 lt2, lt_of_le_of_lt h2 lt2, lt_of_le_of_lt h3 ltT⟩

/-- **`Deployed` at the deployed vocabulary**, from the commitment format's openings and the strict premises, at
constructions whose read finder and forger are the binding's (`Binding.withBinding`, `Constructions.assembled`). -/
theorem deployed (hKr : ∀ θ F c σ, K.readFinder θ F c σ = Binding.readFinder I.base I.dev θ F c σ)
    (hKf : ∀ θ F c σ, K.forger θ F c σ = Binding.forger I.base I.dev θ F c σ) (hO : I.OpensAt) (hF : I.StrictAt K) :
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
  finderCR θ F A h c σ hcr := by
    obtain ⟨hs, hm, hr⟩ := hF θ F A h c σ hcr
    obtain ⟨h1, h2, -⟩ := I.fileCR hcr
    exact ⟨hs, hm, hr, strict_of_zero (pr_unit h1), strict_of_zero (pr_unit h2)⟩
  delta θ F A h t := by
    show A.delta t _ ≤ ((I.session θ).map fun A => A.delta t (I.base.crBound θ .sampled t)).getD 0
    rw [(I.sessions_some h).1]
    exact le_rfl

/-! ## `Covers` -/

/-- **The registered files pass the file check** wherever the parameters name the one-stage form. -/
def RegisteredFiles : Prop := ∀ θ, (∃ p, ParamsAt θ p) → I.base.Registered θ (I.base.registered θ)

/-- **v2's session at the registered files' files**: at parameters naming the one-stage form, committed files that
name the registered circuit and public files and hold the registered lookup tables wherever both hold one, and that
some play's sampled verdict accepts, are files the deployment's session is the one v2 sets up on. -/
def SessionAt (k : Verity.Registration.Composite) : Prop :=
  ∀ θ F, (∃ p, ParamsAt θ p) → I.base.circuitFile θ F = I.base.circuitFile θ (I.base.registered θ) →
    I.base.publicFile θ F = I.base.publicFile θ (I.base.registered θ) → TablesAgree F (I.base.registered θ) →
    (∃ c o, Verity.Game.Reaches (I.base.game θ F c) o ∧ I.base.Verdict k .sampled o) →
    ∃ A, I.session θ = some A ∧ SetsUp I.base θ F A

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

/-- **`Covers` at the deployed vocabulary**: where the deployment's sessions hold none, an accepting play's files pass
the file check, so under the further finders' premises they name the registered files and agree with the registered
tables (`namedFile_eq`, `tablesAgree_of`), where v2's session is the deployment's (`SessionAt`). -/
theorem covers (k : Verity.Registration.Composite) (hRg : I.RegisteredFiles) (hSa : I.SessionAt k) :
    Covers (I.vocab K) k I.sessions := by
  intro θ F hS hP c σ hcr
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
  have hR := hRg θ hP
  obtain ⟨h1, h2, h3⟩ := I.fileCR hcr
  obtain ⟨A, hA, hs⟩ := hSa θ F hP (namedFile_eq hreg.1 hR.1 h1) (namedFile_eq hreg.2 hR.2 h2) (tablesAgree_of h3)
    ⟨c, _, hreach, hv'⟩
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
    (h : CouplingTarget I.stage k K A θ F) : CouplingTarget (I.vocab K) k (I.lift K) A θ F :=
  h

end Instance

/-- **Soundness at remote's verifier of record at the deployed vocabulary**: remote's compiled verifier accepts files
that passed the commit phase's check while one of its three claims fails with probability at most their errors summed,
at the vocabulary `Instance.vocab` builds from the deployment, wherever the commitment format's openings (`OpensAt`),
the strict premises (`StrictAt`), the registered files' file check (`RegisteredFiles`), v2's session at the registered
files (`SessionAt`), the coupling at each session (`CouplingTarget`, at the stage), the bridge, the recursive form's
target, and the base's Pearl+ and inference-only targets hold. -/
theorem remote_soundness_deployed (I : Instance d α) (K : Constructions I.stage)
    (hKr : ∀ θ F c σ, K.readFinder θ F c σ = Binding.readFinder I.base I.dev θ F c σ)
    (hKf : ∀ θ F c σ, K.forger θ F c σ = Binding.forger I.base I.dev θ F c σ) (hO : I.OpensAt) (hF : I.StrictAt K)
    (hRg : I.RegisteredFiles) (hSa : I.SessionAt .remote)
    (hC : ∀ θ F A, I.sessions θ F = some A → CouplingTarget I.stage .remote K A θ F) (hB : UnboundFaulty)
    (hR : RecursiveForm (I.vocab K) .remote) (hW : I.base.PouwSound .remote)
    (hI : I.base.InferenceOnlySound .remote) :
    (I.vocab K).SoundnessAt .remote :=
  Verity.remote_soundness_of_targets (I.vocab K) (I.deployed K hKr hKf hO hF) (I.covers K .remote hRg hSa)
    (fun θ F A h => I.couplingTarget_lift K .remote (hC θ F A h)) hB hR (I.pouwSound_of_base K .remote hW)
    (I.inferenceOnlySound_of_base K .remote hI)

/-- **`remote_soundness_deployed` at the binding's constructions** (`Binding.withBinding`): any constructions with
the binding's read finder and forger, so those two premises hold by definition. -/
theorem remote_soundness_deployed_withBinding (I : Instance d α) (K₀ : Constructions I.stage) (hO : I.OpensAt)
    (hF : I.StrictAt (Binding.withBinding K₀ I.dev)) (hRg : I.RegisteredFiles) (hSa : I.SessionAt .remote)
    (hC : ∀ θ F A, I.sessions θ F = some A → CouplingTarget I.stage .remote (Binding.withBinding K₀ I.dev) A θ F)
    (hB : UnboundFaulty) (hR : RecursiveForm (I.vocab (Binding.withBinding K₀ I.dev)) .remote)
    (hW : I.base.PouwSound .remote) (hI : I.base.InferenceOnlySound .remote) :
    (I.vocab (Binding.withBinding K₀ I.dev)).SoundnessAt .remote :=
  remote_soundness_deployed I (Binding.withBinding K₀ I.dev) (fun _ _ _ _ => rfl) (fun _ _ _ _ => rfl) hO hF hRg hSa
    hC hB hR hW hI

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

/-- The device's history at every play: no run and no ledger line. -/
def dev : Exchange → Properties.Challenger.History := fun _ => ⟨fun _ => [], fun _ => []⟩

/-- **The declared calls**: the most, over the play's tapes, of what the device's reading of what the prover sent
hashes (`Binding.breakCalls`) and one per blob it sent. -/
noncomputable def cost (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (Verity.play [] Verity.Idle.signer θ F c)) : ℕ :=
  Finset.univ.sup fun t : Verity.Game.Tape (Verity.play [] Verity.Idle.signer θ F c) σ =>
    Binding.breakCalls (dev (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
      (Verity.Game.Tape.out _ σ t).sent.blobs.size

/-- **The idle deployment at v2's reading**: `Idle.vocab 0 1` with v2's reading of the committed files and its
digests, the declared calls `cost`, and every name that reads the committed circuit at it. -/
noncomputable def base : Vocab Verity.Idle.device Unit :=
  { Verity.Idle.vocab 0 1 with
    read := readOf
    programDigest := programDigestOf
    publicDigest := publicDigestOf
    place := fun _ _ _ _ => none
    extract := fun _ _ _ _ _ => 0
    cost := cost
    finder_calls := fun _ _ _ _ _ _ => Nat.zero_le _
    finders := fun _ _ _ _ => []
    forger_calls := fun _ _ _ _ _ _ => Nat.zero_le _
    computed := fun _ _ _ _ _ => ∅
    theta := fun _ _ => ∅
    theta_input := fun _ _ _ h => absurd h (Finset.notMem_empty _)
    schedule := fun θ F => blocks (readOf θ F).circuit
    wipe := fun θ F => wipe (readOf θ F).circuit
    wipeRun := fun θ F c _ => ofRun (readOf θ F).circuit (Verity.play [] Verity.Idle.signer θ F c) }

theorem costs : CostsRead base dev := fun θ F c σ t =>
  Finset.le_sup (f := fun t : Verity.Game.Tape (Verity.play [] Verity.Idle.signer θ F c) σ =>
    Binding.breakCalls (dev (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
      (Verity.Game.Tape.out _ σ t).sent.blobs.size) (Finset.mem_univ t)

/-- **The instance at the deployment's session `session`**: `base`, whose reading and digests are v2's by definition. -/
noncomputable def inst (session : ByteArray → Option Live) : Instance Verity.Idle.device Unit where
  base := base
  dev := dev
  read _ _ := rfl
  programDigest _ := rfl
  publicDigest _ := rfl
  costs := costs
  session := session

theorem sessions_none (θ : ByteArray) (F : Array ByteArray) : (inst fun _ => none).sessions θ F = none := rfl

/-- At no session the openings hold of every session, there being none. -/
theorem opensAt_none : (inst fun _ => none).OpensAt := fun _ _ _ h => by simp [inst] at h

/-- At no session the strict premises hold at every session, there being none. -/
theorem strictAt_none (K : Constructions (inst fun _ => none).stage) : (inst fun _ => none).StrictAt K :=
  fun θ F A h => by rw [sessions_none] at h; cases h

end Instance.Honest

end Verity.Sampled
