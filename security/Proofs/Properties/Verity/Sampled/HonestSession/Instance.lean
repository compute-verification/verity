module

public import Proofs.Properties.Verity.Sampled.HonestSession.Live
public import Proofs.Properties.Verity.Sampled.HonestSession.Params
public import Proofs.Properties.Verity.Sampled.OpensAt
public import Proofs.Properties.Verity.Sampled.SessionAt
import all Lean.Data.Json.Basic

/-!
# The honest deployed instance with a session

The idle deployment at v2's reading (`Instance.Honest.base ⊤`) that registers the circuit and public files at `θ₀`, its
session at each parameters the one `liveOf` chooses on the registered files at `live`'s bound (`bound`). Away from `θ₀` it
is the idle deployment, its registered files the parameters twice.

v2 reads the public file of the registered files `#[circuitFile, publicFile]` by its digest (`Vocab.publicFile`,
`Verity.namedFile`), so v2 sets `live` up there only where the two files' SHA-512s differ (`sets_up`, `live_spec`). The
instance runs a session at `θ₀` exactly there (`session_at`); where they don't, the two files are a SHA-512 collision
(`collision`).
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open Lean (Json)
open scoped ENNReal

private theorem toUTF8_head_of_toList {s : String} {c : Char} {cs : List Char} {b : UInt8}
    (hb : String.utf8EncodeChar c = [b]) (h : s.toList = c :: cs) : s.toUTF8.data.toList.head? = some b := by
  rw [← String.ofList_toList (s := s), h, String.toUTF8_eq_toByteArray, String.toByteArray_ofList]
  simp [List.utf8Encode, hb]

private theorem head_append {x : ByteArray} {b : UInt8} (h : x.data.toList.head? = some b) (y : ByteArray) :
    (x ++ y).data.toList.head? = some b := by
  rw [ByteArray.data_append, Array.toList_append]
  cases hx : x.data.toList with
  | nil => rw [hx] at h; simp at h
  | cons a l => rw [hx] at h; simpa using h

private theorem circuitFileOf_head (ls lh : Flock.HmNets.Laid) : (circuitFileOf ls lh).data.toList.head? = some 102 := by
  obtain ⟨rest, h⟩ : ∃ rest, circuitLines ls lh = ("flock-circuit " ++ unitName) :: rest := ⟨_, rfl⟩
  unfold circuitFileOf
  rw [h, RecDeploy.VStarNets.joinNl_cons]
  have e : ("flock-circuit " ++ unitName).toList = 'f' :: ("lock-circuit ".toList ++ unitName.toList) := by
    rw [String.toList_append]; rfl
  exact toUTF8_head_of_toList (c := 'f')
    (cs := "lock-circuit ".toList ++ unitName.toList ++ "\n".toList ++ (RecDeploy.VStarNets.joinNl rest).toList)
    (by decide) (by rw [String.toList_append, String.toList_append, e]; rfl)

private theorem circuitFile_head : circuitFile.data.toList.head? = some 102 := circuitFileOf_head _ _

private theorem jsonChars_mkObj (l : List (String × Json)) : ∃ cs, Flock.ZkServer.jsonChars (Json.mkObj l) = '{' :: cs := by
  unfold Json.mkObj
  exact ⟨_, rfl⟩

private theorem print_mkObj_head (l : List (String × Json)) (rest : ByteArray) :
    ((Flock.ZkServer.printJson (Json.mkObj l)).toUTF8 ++ "\n".toUTF8 ++ rest).data.toList.head? = some 123 := by
  obtain ⟨cs, h⟩ := jsonChars_mkObj l
  exact head_append (head_append (toUTF8_head_of_toList (c := '{') (cs := cs) (by decide)
    (by rw [Flock.ZkServer.printJson, String.toList_ofList, h])) _) _

private theorem publicFile_head : publicFile.data.toList.head? = some 123 := by
  unfold publicFile hdrJ
  exact print_mkObj_head _ _

/-- The two registered files have distinct SHA-512s. -/
def Distinct : Prop := Verity.sha512 circuitFile.data.toList ≠ Verity.sha512 publicFile.data.toList

/-- Where they don't, the two files are a SHA-512 collision: they differ, at their first bytes. -/
theorem collision (h : ¬ Distinct) :
    Hash.Collision Verity.sha512 circuitFile.data.toList publicFile.data.toList := by
  refine ⟨fun e => ?_, Classical.not_not.1 h⟩
  have := congrArg List.head? e
  rw [circuitFile_head, publicFile_head] at this
  simp at this

/-- The circuit file registered at `θ'`. -/
def circuitFileAt (θ' : ByteArray) : ByteArray := if θ' = θ₀ then circuitFile else θ'

/-- The public file registered at `θ'`. -/
def publicFileAt (θ' : ByteArray) : ByteArray := if θ' = θ₀ then publicFile else θ'

/-- The idle deployment at v2's reading with the registration. -/
noncomputable def base : Vocab Verity.Idle.device Unit :=
  { Instance.Honest.base ⊤ with registered := fun θ' => #[circuitFileAt θ', publicFileAt θ'] }

theorem namedFile_first (a b : ByteArray) :
    Verity.namedFile (some (Verity.sha512 a.data.toList)) #[a, b] = a := by
  simp [Verity.namedFile, Array.find?]

theorem namedFile_second {a b : ByteArray} (h : Verity.sha512 a.data.toList ≠ Verity.sha512 b.data.toList) :
    Verity.namedFile (some (Verity.sha512 b.data.toList)) #[a, b] = b := by
  have h' : ¬ Verity.sha512 b.data.toList = Verity.sha512 a.data.toList := fun e => h e.symm
  simp [Verity.namedFile, Array.find?, h']

private theorem pinnedTables_empty (cf : ByteArray) : pinnedTables PinnedTables.empty cf = [] := by
  unfold pinnedTables
  rw [List.filterMap_eq_nil_iff]
  intro t _
  simp [PinnedTables.empty, PinnedTables.ofFiles, tableFile]

private theorem registered_at : base.registered θ₀ = #[circuitFile, publicFile] := by
  show #[circuitFileAt θ₀, publicFileAt θ₀] = _
  simp [circuitFileAt, publicFileAt]

/-- The registered files pass the file check at `θ₀`: they hold the circuit and public files the parameters name. -/
theorem registered_θ₀ : base.Registered θ₀ (base.registered θ₀) := by
  unfold Vocab.Registered
  rw [registered_at]
  exact ⟨⟨circuitFile, by simp, Params.digests.1⟩, ⟨publicFile, by simp, Params.digests.2⟩⟩

/-- At `θ'`, the registered file with the digest `h` names: the one at `θ₀`, else `θ'` itself, both files there. -/
private theorem digest_at {θ' f : ByteArray} {h : Option Hash.Digest} (hf : f ∈ base.registered θ')
    (hd : h = some (Verity.sha512 f.data.toList)) (hθ : θ' ≠ θ₀) : h = some (Verity.sha512 θ'.data.toList) := by
  change f ∈ #[circuitFileAt θ', publicFileAt θ'] at hf
  have hf' : f = θ' := by simpa [circuitFileAt, publicFileAt, hθ] using hf
  rw [hd, hf']

theorem circuit_at : base.circuitFile θ₀ (base.registered θ₀) = circuitFile := by
  have hd : base.programDigest θ₀ = some (Verity.sha512 circuitFile.data.toList) :=
    (show base.programDigest θ₀ = programDigestOf θ₀ from rfl).trans (Params.digests).1
  unfold Vocab.circuitFile
  rw [hd, registered_at]
  exact namedFile_first _ _

theorem public_at (h : Distinct) : base.publicFile θ₀ (base.registered θ₀) = publicFile := by
  have hd : base.publicDigest θ₀ = some (Verity.sha512 publicFile.data.toList) :=
    (show base.publicDigest θ₀ = publicDigestOf θ₀ from rfl).trans (Params.digests).2
  unfold Vocab.publicFile
  rw [hd, registered_at]
  exact namedFile_second h

/-- The parameters `θ₀` name no members: v2's parameters there are `Params.params_spec`'s, which name none. -/
theorem one_statement : OneStatement θ₀ := by
  intro p hp
  obtain ⟨p₀, hp₀, -, -, -, -, -, -, -, hm⟩ := Params.params_spec
  rw [ParamsAt.unique hp hp₀]
  exact hm

section

variable (hC : CircuitFileFacts) (hP : PublicFileFacts) (hS : SetupFacts) (hPr : ProgramFacts)

/-- The bound's parameters: `live`'s. -/
noncomputable def bound : SessionBound :=
  ⟨(live hC hP hS hPr).k, (live hC hP hS hPr).Rw, (live hC hP hS hPr).M, (live hC hP hS hPr).ρ,
    (live hC hP hS hPr).v, (live hC hP hS hPr).w, (live hC hP hS hPr).h⟩

/-- **The honest deployed instance with a session**: `base` at the idle deployment, which registers no pinned table and
registers the circuit and public files at `θ₀`, whose session is `liveOf`'s on them at `live`'s bound. -/
noncomputable def inst : Instance Verity.Idle.device Unit where
  base := base
  deployment := Instance.Honest.deployment
  signer _ := rfl
  tables := #[]
  read _ _ := rfl
  programDigest _ := rfl
  publicDigest _ := rfl
  place _ _ _ _ := rfl
  costs := Instance.Honest.costs ⊤
  coins _ _ _ h := (Prod.mk.inj (List.mem_singleton.mp h)).2
  bound _ := bound hC hP hS hPr
  session θ' := liveOf base (bound hC hP hS hPr) θ' (base.registered θ')
  sessionOf _ := rfl
  circuitFile := circuitFileAt
  publicFile := publicFileAt
  circuitHash θ' _ h := by
    obtain ⟨⟨f, hf, hd⟩, -⟩ := (liveOf_spec h).registered
    by_cases hθ : θ' = θ₀
    · subst hθ
      show programDigestOf θ₀ = some (Verity.sha512 (if θ₀ = θ₀ then circuitFile else θ₀).data.toList)
      rw [ite_eq_left rfl]
      exact (Params.digests).1
    · rw [show circuitFileAt θ' = θ' from ite_eq_right hθ]
      exact digest_at hf hd hθ
  publicHash θ' _ h := by
    obtain ⟨-, ⟨f, hf, hd⟩⟩ := (liveOf_spec h).registered
    by_cases hθ : θ' = θ₀
    · subst hθ
      show publicDigestOf θ₀ = some (Verity.sha512 (if θ₀ = θ₀ then publicFile else θ₀).data.toList)
      rw [ite_eq_left rfl]
      exact (Params.digests).2
    · rw [show publicFileAt θ' = θ' from ite_eq_right hθ]
      exact digest_at hf hd hθ
  registered _ := Array.append_empty.symm

end

/-- `live` is the session v2 sets up on `θ₀` and the registered files, where their SHA-512s differ. -/
theorem sets_up (hC : CircuitFileFacts) (hP : PublicFileFacts) (hS : SetupFacts) (hPr : ProgramFacts)
    (hd : Distinct) : SetsUp base θ₀ (base.registered θ₀) (live hC hP hS hPr) := by
  obtain ⟨p, hp, hpart, hprog, hvp, hvq, hvl, hvt, hch, -⟩ := Params.params_spec
  have hI := live_inputs hC hP hS hPr
  have hcirc : (live hC hP hS hPr).I.circuitFile = base.circuitFile θ₀ (base.registered θ₀) := by
    rw [hI, circuit_at]; rfl
  have hpub : (live hC hP hS hPr).I.publicFile = base.publicFile θ₀ (base.registered θ₀) := by
    rw [hI, public_at hd]; rfl
  refine
    { params := ⟨p, hp, by rw [hI, hpart]; rfl, by rw [hI, hprog]; rfl, ?_, ?_, ?_, key₀, label₀,
          _root_.Spot.Challenger.MAX_COINS, hch, ?_, by rw [live_lb hC hP hS hPr]⟩
      tags := by rw [hI]; exact tags_find
      count := by rw [hI]; rfl
      work := by rw [hI]; rfl
      stratified := by rw [hI]; rfl
      closure := by rw [hI]; rfl
      own := live_own hC hP hS hPr
      circuit := hcirc
      publicFile := hpub
      read := ?_
      share := ?_
      single := one_statement }
  · rw [Live.design, live_n hC hP hS hPr, live_desc hC hP hS hPr, live_query hC hP hS hPr, live_law hC hP hS hPr,
      live_table hC hP hS hPr]
    exact hPr.lawAt_of _ hvp hvq hvl hvt
  · rw [live_desc hC hP hS hPr, hvp]
    exact hPr.program_parse
  · rw [live_query hC hP hS hPr, hvq]
  · show (label₀, _) ∈ [(labelAt θ₀, _)]
    rw [labelAt_eq hp hch]
    exact List.mem_singleton_self _
  · show Verity.ExecutionPhaseFiles.one (readOf PinnedTables.empty θ₀ (base.registered θ₀)) =
      .one (live hC hP hS hPr).fileRead
    congr 1
    refine readOf_eq base PinnedTables.empty (fun _ => rfl) (fun _ => rfl) ?_ ?_ hcirc hpub ?_
    · rw [hI]; exact tags_find
    · rw [hI]; rfl
    · rw [hI]
      exact (pinnedTables_empty _).symm
  · rw [live_epsilon hC hP hS hPr]; rfl

/-- `live` meets `LiveSpec` at `θ₀`, the registered files and its own bound, where their SHA-512s differ. -/
theorem live_spec (hC : CircuitFileFacts) (hP : PublicFileFacts) (hS : SetupFacts) (hPr : ProgramFacts)
    (hd : Distinct) : LiveSpec base (bound hC hP hS hPr) θ₀ (base.registered θ₀) (live hC hP hS hPr) := by
  have h := sets_up hC hP hS hPr hd
  obtain ⟨p, hp, h1, h2, h3, h4, h5, -⟩ := h.params
  exact
    { params := ⟨p, hp, h1, h2, h3, h4, h5⟩
      lb := by rw [live_lb hC hP hS hPr]
      tags := h.tags
      count := h.count
      work := h.work
      stratified := h.stratified
      closure := h.closure
      own := h.own
      circuit := h.circuit
      publicFile := h.publicFile
      read := h.read
      share := h.share
      k := rfl
      Rw := rfl
      M := rfl
      ρ := rfl
      v := rfl
      w := rfl
      h := rfl
      registered := registered_θ₀
      single := h.single }

/-- **The instance runs a session at `θ₀`**, where the registered files' SHA-512s differ: the one `liveOf` chooses among
those meeting `LiveSpec` there, which `live` does. -/
theorem session_at (hC : CircuitFileFacts) (hP : PublicFileFacts) (hS : SetupFacts) (hPr : ProgramFacts)
    (hd : Distinct) : ∃ A, (inst hC hP hS hPr).session θ₀ = some A ∧
      LiveSpec base (bound hC hP hS hPr) θ₀ (base.registered θ₀) A := by
  have hx : ∃ A, LiveSpec base (bound hC hP hS hPr) θ₀ (base.registered θ₀) A := ⟨_, live_spec hC hP hS hPr hd⟩
  refine ⟨hx.choose, ?_, hx.choose_spec⟩
  show liveOf base (bound hC hP hS hPr) θ₀ (base.registered θ₀) = some hx.choose
  unfold liveOf
  rw [dif_pos hx]

/-- **The coin commitment's families hold the family v2's parameters name**: the base's are the honest instance's
(`Instance.Honest.families`), the family `labelAt θ` names at every `θ`, which is the parameters' challenger's
(`Binding.dispatch_deployed`, `labelAt_eq`). -/
theorem families_pinned (hC : CircuitFileFacts) (hP : PublicFileFacts) (hS : SetupFacts) (hPr : ProgramFacts)
    (k : Verity.Registration.Composite) : (inst hC hP hS hPr).FamiliesPinned k := by
  intro θ p hp _
  obtain ⟨_, _, _, _, hj⟩ := id hp
  obtain ⟨key, label, hch, -⟩ := Binding.dispatch_deployed hj
  refine ⟨key, label, _root_.Spot.Challenger.MAX_COINS, hch, ?_⟩
  show (label, _) ∈ [(labelAt θ, _)]
  rw [labelAt_eq hp hch]
  exact List.mem_singleton_self _

theorem error_top (a : Subprotocol) (θ' : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (base.game θ' F c)) : base.error a θ' F c σ = ⊤ := by
  unfold Vocab.error ResourceAccounting.condError
  split_ifs <;> simp [base, Instance.Honest.base, Verity.Idle.vocab]

theorem target_top (k : Verity.Registration.Composite) (a : Subprotocol) : base.Target k a := by
  intro θ' F c σ
  rw [error_top]
  exact le_top

end Verity.Sampled.Instance.HonestSession
