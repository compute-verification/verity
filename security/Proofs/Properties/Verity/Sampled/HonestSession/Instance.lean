module

public import Proofs.Properties.Verity.Sampled.HonestSession.Live
public import Proofs.Properties.Verity.Sampled.HonestSession.Params
public import Proofs.Properties.Verity.Sampled.OpensAt
public import Proofs.Properties.Verity.Sampled.SessionAt

/-!
# The honest deployed instance with a session

The idle deployment at v2's reading (`Instance.Honest.base ⊤`) that registers the circuit and public files at `θ₀`,
advertises the challenger family `label₀` there, and runs `live` at `θ₀`. Away from `θ₀` it is the idle deployment.

v2 reads the public file of the registered files `#[circuitFile, publicFile]` by its digest (`Vocab.publicFile`,
`Verity.namedFile`), so the session v2 sets up there is `live` only where the two files' SHA-512s differ. The instance
runs `live` at `θ₀` exactly there (`session`); where they don't, the two files are a SHA-512 collision (`collision`).
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open Lean (Json)
open scoped ENNReal

/-- The two registered files have distinct SHA-512s. -/
def Distinct : Prop := Verity.sha512 circuitFile.data.toList ≠ Verity.sha512 publicFile.data.toList

/-- Where they don't, the two files are a SHA-512 collision: they differ, at their first bytes. -/
theorem collision (h : ¬ Distinct) :
    Hash.Collision Verity.sha512 circuitFile.data.toList publicFile.data.toList := by
  sorry -- piece P9 (instance): the files differ at their first byte (`f` against `{`)

/-- The challenger family the parameters name. -/
def families (θ' : ByteArray) : List (String × ℕ) :=
  if θ' = θ₀ then [(label₀, _root_.Spot.Challenger.MAX_COINS)] else []

/-- The circuit file registered at `θ'`. -/
def circuitFileAt (θ' : ByteArray) : ByteArray := if θ' = θ₀ then circuitFile else θ'

/-- The public file registered at `θ'`. -/
def publicFileAt (θ' : ByteArray) : ByteArray := if θ' = θ₀ then publicFile else θ'

/-- The idle deployment's signer over the family. -/
def deployedSigner (θ' : ByteArray) : Signer := Instance.Honest.deployment.signer families θ'

/-- The idle deployment's history at the family. -/
noncomputable abbrev dev : Exchange → Properties.Challenger.History :=
  Device.dev (Instance.Honest.deployment.setupOf families)

/-- The declared calls on the family's game. -/
noncomputable def cost (θ' : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (Verity.play (families θ') (deployedSigner θ') θ' F c)) : ℕ :=
  Finset.univ.sup fun t : Verity.Game.Tape (Verity.play (families θ') (deployedSigner θ') θ' F c) σ =>
    Binding.breakCalls (dev (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
      (Verity.Game.Tape.out _ σ t).sent.blobs.size

/-- The idle vocabulary with the registration and the family. -/
noncomputable def base : Vocab Verity.Idle.device Unit :=
  { Instance.Honest.base ⊤ with
    registered := fun θ' => #[circuitFileAt θ', publicFileAt θ']
    families := families
    signer := deployedSigner
    cost := cost
    honest := fun θ' F c =>
      Verity.requestThen (families θ') (deployedSigner θ') θ' F c .empty
        ⟨.empty, fun _ => ⟨(.empty, #[]), ⟨⟩⟩⟩ }

theorem costs : CostsRead base dev := fun θ' F c σ t =>
  Finset.le_sup
    (f := fun t : Verity.Game.Tape (Verity.play (families θ') (deployedSigner θ') θ' F c) σ =>
      Binding.breakCalls (dev (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
        (Verity.Game.Tape.out _ σ t).sent.blobs.size)
    (Finset.mem_univ t)

open Classical in
/-- The deployment runs `live` at `θ₀`, where the registered files' SHA-512s differ. -/
noncomputable def session (θ' : ByteArray) : Option Live :=
  if θ' = θ₀ ∧ Distinct then some live else none

theorem session_some {θ' : ByteArray} {A : Live} (h : session θ' = some A) : θ' = θ₀ ∧ Distinct ∧ A = live := by
  unfold session at h
  split at h
  · exact ⟨‹_ ∧ _›.1, ‹_ ∧ _›.2, (Option.some.inj h).symm⟩
  · cases h

/-- The honest deployed instance with a session. -/
noncomputable def inst : Instance Verity.Idle.device Unit where
  base := base
  deployment := Instance.Honest.deployment
  signer _ := rfl
  tables := #[]
  read _ _ := rfl
  programDigest _ := rfl
  publicDigest _ := rfl
  place _ _ _ _ := rfl
  costs := costs
  coins θ' l m hm := by
    simp only [base, families] at hm
    split at hm
    · exact (Prod.mk.inj (List.mem_singleton.1 hm)).2
    · simp at hm
  session := session
  circuitFile := circuitFileAt
  publicFile := publicFileAt
  circuitHash θ' A h := by
    obtain ⟨rfl, -, -⟩ := session_some h
    show programDigestOf θ₀ = some (Verity.sha512 (if θ₀ = θ₀ then circuitFile else θ₀).data.toList)
    rw [if_pos rfl]
    exact (Params.digests JsonCanon.jsonFacts).1
  publicHash θ' A h := by
    obtain ⟨rfl, -, -⟩ := session_some h
    show publicDigestOf θ₀ = some (Verity.sha512 (if θ₀ = θ₀ then publicFile else θ₀).data.toList)
    rw [if_pos rfl]
    exact (Params.digests JsonCanon.jsonFacts).2
  registered θ' := (Array.append_empty _).symm

theorem namedFile_first (a b : ByteArray) :
    Verity.namedFile (some (Verity.sha512 a.data.toList)) #[a, b] = a := by
  sorry -- piece P9 (instance)

theorem namedFile_second {a b : ByteArray} (h : Verity.sha512 a.data.toList ≠ Verity.sha512 b.data.toList) :
    Verity.namedFile (some (Verity.sha512 b.data.toList)) #[a, b] = b := by
  sorry -- piece P9 (instance)

theorem circuit_at : base.circuitFile θ₀ (base.registered θ₀) = circuitFile := by
  sorry -- piece P9 (instance): `Params.digests` and `namedFile_first`

theorem public_at (h : Distinct) : base.publicFile θ₀ (base.registered θ₀) = publicFile := by
  sorry -- piece P9 (instance): `Params.digests` and `namedFile_second`

/-- `live` is the session v2 sets up on `θ₀` and the registered files, where their SHA-512s differ. -/
theorem sets_up (hd : Distinct) : SetsUp base θ₀ (base.registered θ₀) live := by
  obtain ⟨p, hp, hpart, hprog, hvp, hvq, hvl, hvt, hch⟩ := Params.params_spec JsonCanon.jsonFacts
  have hI := live_inputs
  have hcirc : live.I.circuitFile = base.circuitFile θ₀ (base.registered θ₀) := by
    rw [hI, circuit_at] <;> rfl
  have hpub : live.I.publicFile = base.publicFile θ₀ (base.registered θ₀) := by
    rw [hI, public_at hd] <;> rfl
  refine
    { params := ⟨p, hp, by rw [hI, hpart] <;> rfl, by rw [hI, hprog] <;> rfl, ?_, ?_, ?_, key₀, label₀,
          _root_.Spot.Challenger.MAX_COINS, hch, by show (label₀, _) ∈ families θ₀; simp [families], by rw [live_lb] <;> rfl⟩
      tags := by rw [hI]; exact tags_find
      count := by rw [hI] <;> rfl
      work := by rw [hI] <;> rfl
      stratified := by rw [hI] <;> rfl
      closure := by rw [hI] <;> rfl
      own := live_own
      circuit := hcirc
      publicFile := hpub
      read := ?_
      share := ?_ }
  · rw [Live.design, live_n, live_desc, live_query, live_law, live_table]
    exact Program.lawAt_of _ hvp hvq hvl hvt
  · rw [live_desc, hvp]
    exact Program.program_parse
  · rw [live_query, hvq]
  · show readOf PinnedTables.empty θ₀ (base.registered θ₀) = live.fileRead
    refine readOf_eq base PinnedTables.empty (fun _ => rfl) (fun _ => rfl) ?_ ?_ hcirc hpub ?_
    · rw [hI]; exact tags_find
    · rw [hI] <;> rfl
    · rw [hI]
      sorry -- piece P9 (instance): `pinnedTables` at no registered table is `[]`
  · rw [live_epsilon] <;> rfl

theorem sessions_ready : inst.SessionsReady := by
  intro θ' F A hs _
  obtain ⟨-, -, rfl⟩ := session_some hs
  exact live_ready

theorem session_registered : inst.SessionRegistered .remote := by
  intro θ' A hs _ _
  obtain ⟨rfl, hd, rfl⟩ := session_some hs
  exact sets_up hd

theorem error_top (a : Subprotocol) (θ' : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (base.game θ' F c)) : base.error a θ' F c σ = ⊤ := by
  unfold Vocab.error Accounting.condError
  split_ifs <;> simp [base, Instance.Honest.base, Verity.Idle.vocab]

theorem target_top (k : Verity.Registration.Composite) (a : Subprotocol) : base.Target k a := by
  intro θ' F c σ
  rw [error_top]
  exact le_top

end Verity.Sampled.Instance.HonestSession
