module

public import Proofs.Properties.Verity.Sampled.HonestSession.Live
public import Proofs.Properties.Verity.Sampled.OpensAt
public import Proofs.Properties.Verity.Sampled.SessionAt

/-!
# A session-bearing honest deployed instance

This instance registers the concrete circuit and public file, advertises the concrete challenger family, and runs the
accepted `live` session exactly at its parameters. Away from those parameters it is the existing idle deployment.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open scoped ENNReal

/-- The decoded component-parameter list. -/
def decodedResult :=
  _root_.Core.canonicalJson θ >>= _root_.Core.Composition.decodeParams

set_option maxHeartbeats 0 in
theorem decoded_some : decodedResult.toOption.isSome := by decide +kernel

def decoded : List _root_.Core.Composition.Named := decodedResult.toOption.get decoded_some

theorem decoded_ok : decodedResult = .ok decoded := by
  have hds := decoded_some
  generalize h : decodedResult = r at hds ⊢
  cases r <;> simp_all [decoded]

set_option maxHeartbeats 0 in
theorem member_some :
    (decoded.find? (·.id == VStar.Recursive.Sampled.PROTOCOL)).isSome := by
  decide +kernel

def member : _root_.Core.Composition.Named :=
  (decoded.find? (·.id == VStar.Recursive.Sampled.PROTOCOL)).get member_some

theorem member_ok :
    decoded.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) = some member := by
  have hms := member_some
  generalize h : decoded.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) = x at hms ⊢
  cases x <;> simp_all [member]

def innerResult :=
  _root_.Core.canonicalJson member.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson

set_option maxHeartbeats 0 in
theorem inner_some : innerResult.toOption.isSome := by decide +kernel

def inner := innerResult.toOption.get inner_some

set_option maxHeartbeats 0 in
theorem left_some :
    (match inner with
      | .inl p => some p
      | .inr _ => none).isSome := by
  decide +kernel

/-- The concrete one-stage-v2 parameters. -/
def params : _root_.Spot.OneStageZk.Params :=
  (match inner with
    | .inl p => some p
    | .inr _ => none).get left_some

theorem inner_ok : innerResult = .ok (.inl params) := by
  have his := inner_some
  have hls := left_some
  generalize h : innerResult = r at his ⊢
  cases r with
  | error e => simp at his
  | ok x =>
    cases x with
    | inl p => simp [params, inner, h]
    | inr p => simp [inner, h] at hls

/-- The concrete parameters contain one-stage-v2's parameters. -/
theorem params_at : ParamsAt θ params :=
  ⟨decoded, decoded_ok, member, member_ok, inner_ok⟩

set_option maxHeartbeats 0 in
set_option maxRecDepth 1000000 in
/-- The parameter digests are the registered circuit and public file's SHA-512 digests. -/
theorem registered_digests :
    programDigestOf θ = some (Verity.sha512 circuitFile.data.toList) ∧
      publicDigestOf θ = some (Verity.sha512 publicFile.data.toList) := by
  decide +kernel

/-- The challenger family named by the parameters. -/
def families (θ' : ByteArray) : List (String × ℕ) :=
  if θ' = θ then [("honest-session", _root_.Spot.Challenger.MAX_COINS)] else []

/-- The files registered by the honest deployment. -/
def registeredFiles (θ' : ByteArray) : Array ByteArray :=
  if θ' = θ then files else #[θ']

/-- The idle deployment's signer over the one concrete challenger family. -/
def deployedSigner (θ' : ByteArray) : Signer :=
  Instance.Honest.deployment.signer families θ'

/-- The idle deployment's history at the concrete family. -/
noncomputable abbrev dev : Exchange → Properties.Challenger.History :=
  Device.dev (Instance.Honest.deployment.setupOf families)

/-- Calls made by the binding read on the concrete-family game. -/
noncomputable def cost (θ' : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (Verity.play (families θ') (deployedSigner θ') θ' F c)) : ℕ :=
  Finset.univ.sup fun t : Verity.Game.Tape (Verity.play (families θ') (deployedSigner θ') θ' F c) σ =>
    Binding.breakCalls (dev (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
      (Verity.Game.Tape.out _ σ t).sent.blobs.size

/-- The idle vocabulary with the concrete registration and challenger family. -/
noncomputable def base : Vocab Verity.Idle.device Unit :=
  { Instance.Honest.base ⊤ with
    registered := registeredFiles
    families := families
    signer := deployedSigner
    cost := cost
    honest := fun θ' F c =>
      Verity.requestThen (families θ') (deployedSigner θ') θ' F c .empty
        ⟨.empty, fun _ => ⟨(.empty, #[]), ⟨⟩⟩⟩ }

/-- The declared cost covers the concrete-family deployment's binding read. -/
theorem costs : CostsRead base dev := fun θ' F c σ t =>
  Finset.le_sup
    (f := fun t : Verity.Game.Tape (Verity.play (families θ') (deployedSigner θ') θ' F c) σ =>
      Binding.breakCalls (dev (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
        (Verity.Game.Tape.out _ σ t).sent.blobs.size)
    (Finset.mem_univ t)

/-- The deployment runs the concrete accepted session exactly at its parameters. -/
noncomputable def session (θ' : ByteArray) : Option Live :=
  if θ' = θ then some live else none

/-- The honest deployed instance with a proving session. -/
noncomputable def inst : Instance Verity.Idle.device Unit where
  base := base
  deployment := Instance.Honest.deployment
  signer _ := rfl
  pinned := Pinned.empty
  read _ _ := rfl
  programDigest _ := rfl
  publicDigest _ := rfl
  place _ _ _ _ := rfl
  costs := costs
  coins θ' l m hm := by
    simp only [families] at hm
    split at hm
    · simpa using hm
    · simp at hm
  session := session

set_option maxHeartbeats 0 in
set_option maxRecDepth 1000000 in
/-- The registered files name the circuit and public file at the parameters' digests. -/
theorem registered_at : base.Registered θ files := by
  constructor
  · refine ⟨circuitFile, by simp [files], ?_⟩
    exact registered_digests.1
  · refine ⟨publicFile, by simp [files], ?_⟩
    exact registered_digests.2

set_option maxHeartbeats 0 in
/-- The concrete parameters agree with every input the accepted session uses. -/
theorem parameter_inputs :
    inputs.partition = some params.own.partition ∧
    inputs.program = some params.own.program ∧
    lawAt params.base.verifier 32 = .ok (Flock.Bridge.designAt desc query law none 32) ∧
    _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok params.base.verifier.program ∧
    query = params.base.verifier.query ∧
    params.base.challenger = some
      ({ params := _root_.Lms.deployed, pub := ByteArray.mk (Array.replicate 80 0) }, "honest-session") := by
  repeat' apply And.intro
  all_goals rfl

/-- The concrete session is set up on the registered files. -/
theorem sets_up : SetsUp base θ files live := by
  have hI := live_inputs
  have hn := live_n
  have hd := live_desc
  have hq := live_query
  have hl := live_law
  have htb := live_table
  refine
    { params := ⟨params, params_at, ?_, ?_, ?_, ?_, ?_, _, "honest-session",
          _root_.Spot.Challenger.MAX_COINS, parameter_inputs.2.2.2.2.2, ?_, ?_⟩
      tags := ?_
      count := ?_
      work := ?_
      stratified := ?_
      closure := ?_
      own := ?_
      circuit := ?_
      publicFile := ?_
      read := ?_
      share := ?_ }
  · simpa [hI] using parameter_inputs.1
  · simpa [hI] using parameter_inputs.2.1
  · simpa [Live.design, hn, hd, hq, hl, htb] using parameter_inputs.2.2.1
  · simpa [hd] using parameter_inputs.2.2.2.1
  · exact hq
  · simp [base, families]
  · rw [live_lb]
    rfl
  · simpa [hI]
  · simpa [hI]
  · simpa [hI]
  · simpa [hI]
  · simpa [hI]
  · rfl
  · simp only [base, Vocab.circuitFile]
    rw [registered_digests.1]
    simp [files, Verity.namedFile]
  · simp only [base, Vocab.publicFile]
    rw [registered_digests.2]
    simp [files, Verity.namedFile]
  · rw [show base.read θ files = readOf Pinned.empty θ files by rfl]
    have hcirc : inputs.circuitFile = base.circuitFile θ files := by
      simp only [inputs, base, Vocab.circuitFile]
      rw [registered_digests.1]
      simp [files, Verity.namedFile]
    have hpublic : inputs.publicFile = base.publicFile θ files := by
      simp only [inputs, base, Vocab.publicFile]
      rw [registered_digests.2]
      simp [files, Verity.namedFile]
    apply readOf_eq base Pinned.empty (fun _ => rfl) (fun _ => rfl)
    · simpa [hI]
    · simpa [hI]
    · simpa [hI] using hcirc
    · simpa [hI] using hpublic
    · simp [hI, inputs, pinnedTables, Pinned.empty, base, Vocab.circuitFile,
        registered_digests.1, files, Verity.namedFile]
  · simpa [base] using live_epsilon.symm

/-- Every session the deployment runs is ready for commitment openings. -/
theorem sessions_ready : inst.SessionsReady := by
  intro θ' F A hs _
  simp only [inst, session] at hs
  split at hs with hθ
  · obtain rfl := Option.some.inj hs
    exact live_ready
  · cases hs

/-- The deployment's registered files pass the file check at its session. -/
theorem registered_files : inst.RegisteredFiles := by
  intro θ' F A hs _ _
  simp only [inst, session] at hs
  split at hs with hθ
  · subst θ'
    simp only [base, registeredFiles, if_pos rfl]
    exact registered_at
  · cases hs

/-- The remote verifier's session at the registered files is the accepted concrete session. -/
theorem session_registered : inst.SessionRegistered .remote := by
  intro θ' A hs _ _
  simp only [inst, session] at hs
  split at hs with hθ
  · subst θ'
    obtain rfl := Option.some.inj hs
    simpa [base, registeredFiles] using sets_up
  · cases hs

/-- At the session-bearing base every subprotocol's error remains `⊤`. -/
theorem error_top (a : Subprotocol) (θ' : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (base.game θ' F c)) : base.error a θ' F c σ = ⊤ := by
  unfold Vocab.error Accounting.condError
  split_ifs <;> simp [base, Instance.Honest.base, Verity.Idle.vocab]

/-- At the session-bearing base every subprotocol meets its target. -/
theorem target_top (k : Verity.Registration.Composite) (a : Subprotocol) : base.Target k a := by
  intro θ' F c σ
  rw [error_top]
  exact le_top

end Verity.Sampled.Instance.HonestSession
