module

public import Properties.Pous.MemoryChallenger
public import Definitions.Core.InProofs
public import Properties.Verity.Sha512
public import Mathlib.Probability.ProbabilityMassFunction.Monad
public import Lms.Sign
public import Proofs.Properties.ResourceAccounting.InProofs
public import Proofs.Properties.Pous.MemoryChallenger
import Proofs.Properties.Pous.WipeSigned
import Proofs.Properties.Pous.MemoryChallenger.Session
import Proofs.Properties.Hash.Lms

/-! An accepted run meeting `RunGives`' three hypotheses gives a session `verdict` accepts; it says nothing of the
deployed run, where nothing proves `Gives` and `Carried`. -/

@[expose] public section

namespace MemoryChallenger.Properties.InProofs

open Verity.InProofs Verity.Game
open scoped ENNReal

structure Vals where
  m : ByteArray
  sig : ByteArray
  ended : String
  rounds : Array Round
  round : Round

def challengePhaseOf (c : ChallengePhaseCommitment) (scheme : String) (pk : ByteArray) (sd sg : Vals) : SignedChallengePhase :=
  { commitment := c, ended := sd.ended, rounds := sd.rounds, scheme := scheme, publicKey := pk, signature := sg.sig }

def ClearOk (schemes : Schemes) (trusted : List ByteArray) (scheme : String) (pk : ByteArray) (c : ChallengePhaseCommitment)
    (openings : Array CoinOpening) (idx : Nat → Nat) : Prop :=
  (∃ p, schemes.lookup scheme = some p) ∧ pk ∈ trusted ∧
    ∀ j < c.k, ∃ o, openings.findRev? (·.j == j) = some o ∧ o.index = idx j ∧ verifyCoin c o = true

def PieceOk (schemes : Schemes) (scheme : String) (pk : ByteArray) (c : ChallengePhaseCommitment)
    (expected : Nat → Option ByteArray) (idx : Nat → Nat) (u : Fin (c.k + 2)) (v : Vals) : Prop :=
  if u.val = 0 then ∀ p, schemes.lookup scheme = some p → lmsVerify p pk c.session v.m v.sig = .ok true
  else if u.val = 1 then
    message c v.ended v.rounds = .ok v.m ∧ v.ended = COMPLETE ∧ v.rounds.size = c.k ∧
      ∀ j (hj : j < v.rounds.size), 0 ≤ v.rounds[j].elapsedNs ∧ v.rounds[j].elapsedNs ≤ c.limitNs
  else
    v.round.j = u.val - 2 ∧ v.round.index = idx (u.val - 2) ∧ v.round.index < c.blocks ∧
      expected v.round.index = some v.round.digest

def Carried {Ω : Type} (c : ChallengePhaseCommitment) (r : Verity.InProofs.Run Ω (c.k + 2) fun _ => Vals) : Prop :=
  ∀ ω, r.accepts ω →
    ((r.ports ⟨0, by omega⟩ ω).m = (r.ports ⟨1, by omega⟩ ω).m ∧
      ∀ j (hj : j < c.k), (r.ports ⟨1, by omega⟩ ω).rounds[j]? = some (r.ports ⟨j + 2, by omega⟩ ω).round) ∨
    Hash.CollisionIn Verity.sha512 (r.hashInputs ω)

/-- **An accepted session gives a session `verdict` accepts, except `ε`, for any run meeting its hypotheses**; it says
nothing of the deployed run, where nothing proves `Gives` and `Carried`. -/
def RunGives : Prop :=
  ∀ (schemes : Schemes) (trusted : List ByteArray) (expected : Nat → Option ByteArray) (c : ChallengePhaseCommitment)
    (openings : Array CoinOpening) (scheme : String) (pk : ByteArray) (idx : Nat → Nat)
    (Ω : Type) (μ : PMF Ω) (r : Verity.InProofs.Run Ω (c.k + 2) fun _ => Vals) (right : Fin (c.k + 2) → Vals → Prop)
    (ε : ℝ≥0∞),
    Gives μ r right 1 ε →
    right ≤ PieceOk schemes scheme pk c expected idx →
    Carried c r →
    prPMF μ (fun ω => r.accepts ω ∧ ClearOk schemes trusted scheme pk c openings idx ∧
        (verdict schemes trusted expected
          (challengePhaseOf c scheme pk (r.ports ⟨1, by omega⟩ ω) (r.ports ⟨0, by omega⟩ ω)) c openings).accepted = false ∧
        ¬ Hash.CollisionIn Verity.sha512 (r.hashInputs ω)) ≤ ε

def constRun {n : ℕ} {V : Type} (v : Fin n → V) : Verity.InProofs.Run Unit n fun _ => V :=
  ⟨fun _ => True, fun u _ => v u, fun _ => []⟩

/-- **`RunGives`' hypotheses hold, with `verdict`'s acceptance, at an honest session**: a key `Lms.keygen` derives at
S1's deployed parameters, trusted alone, signs a session of one round that ended complete on time, and the run that
always accepts holds that session's values at each of its three proof blocks and hashes nothing. -/
def RunGivesHonest : Prop :=
  ∃ (seed : ByteArray) (k : Lms.PrivateKey) (expected : Nat → Option ByteArray) (c : ChallengePhaseCommitment)
    (openings : Array CoinOpening) (idx : Nat → Nat) (v : Fin (c.k + 2) → Vals),
    Lms.keygen Lms.deployed seed = .ok k ∧ c.k = 1 ∧
      Gives (PMF.pure ()) (constRun v) (PieceOk SCHEMES LMS k.publicKey c expected idx) 1 0 ∧
      Carried c (constRun v) ∧ ClearOk SCHEMES [k.publicKey] LMS k.publicKey c openings idx ∧
      (verdict SCHEMES [k.publicKey] expected
        (challengePhaseOf c LMS k.publicKey ((constRun v).ports ⟨1, by omega⟩ ()) ((constRun v).ports ⟨0, by omega⟩ ())) c
        openings).accepted = true

/-- **`RunGives`' conclusion fails without its hypotheses**: under S1's scheme (`SCHEMES`, `LMS`) and a trusted key,
a session of no rounds that did not end complete passes the clear checks and `verdict` rejects it, so a run that always
accepts it and hashes nothing gives the event probability `1`, and the bound fails at `ε = 0`. -/
def RunGivesNontrivial : Prop :=
  ∃ (c : ChallengePhaseCommitment) (pk : ByteArray) (v : Fin (c.k + 2) → Vals),
    ¬ prPMF (PMF.pure ()) (fun ω => (constRun v).accepts ω ∧ ClearOk SCHEMES [pk] LMS pk c #[] (fun _ => 0) ∧
        (verdict SCHEMES [pk] (fun _ => none)
          (challengePhaseOf c LMS pk ((constRun v).ports ⟨1, by omega⟩ ω) ((constRun v).ports ⟨0, by omega⟩ ω)) c
          #[]).accepted = false ∧
        ¬ Hash.CollisionIn Verity.sha512 ((constRun v).hashInputs ω)) ≤ 0

end MemoryChallenger.Properties.InProofs

namespace MemoryChallenger.Properties.InProofs.SecurityProofs

open MemoryChallenger MemoryChallenger.Properties.InProofs MemoryChallenger.Properties.SecurityProofs
open Verity.InProofs Properties.ResourceAccounting.InProofs Verity.Game

theorem runGives : RunGives := by
  intro schemes trusted expected c openings scheme pk idx Ω μ r right ε g h hc
  refine le_trans (prPMF_mono μ fun ω hω => ⟨hω.1, ?_⟩) (Gives.mono h g)
  classical
  obtain ⟨ha, ⟨⟨p, hp⟩, ht, hco⟩, hrej, hn⟩ := hω
  rw [Nat.one_le_iff_ne_zero, Ne, Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
  intro hall
  have hok : ∀ u, PieceOk schemes scheme pk c expected idx u (r.ports u ω) := fun u => by
    by_contra hu
    exact hall u (by simp [Run.failing, hu])
  obtain ⟨hm, hrd⟩ := (hc ω ha).resolve_right hn
  have h0 := hok ⟨0, by omega⟩
  have h1 := hok ⟨1, by omega⟩
  simp only [PieceOk, ↓reduceIte] at h0
  simp only [PieceOk, one_ne_zero, ↓reduceIte] at h1
  obtain ⟨hmsg, he, hsz, htime⟩ := h1
  set sd := r.ports ⟨1, by omega⟩ ω
  set sg := r.ports ⟨0, by omega⟩ ω
  have hround : ∀ j (hj : j < sd.rounds.size), sd.rounds[j] = (r.ports ⟨j + 2, by omega⟩ ω).round := fun j hj => by
    have := hrd j (by omega)
    rw [Array.getElem?_eq_getElem hj] at this
    exact Option.some.inj this
  rw [Bool.eq_false_iff] at hrej
  refine hrej (verdict_accepted_of (p := p) (m := sd.m) hp
    (List.contains_iff_exists_mem_beq.2 ⟨pk, ht, by show (pk.data == pk.data) = true; exact beq_self_eq_true _⟩) hmsg ?_
    (commitment_beq_self c) he hsz ?_)
  · show lmsVerify p pk c.session sd.m sg.sig = .ok true
    rw [← hm]
    exact h0 p hp
  · intro j hj
    have hj' : j < c.k := by simp only [challengePhaseOf] at hj; omega
    obtain ⟨o, ho, hoi, hvc⟩ := hco j hj'
    have hpj := hok ⟨j + 2, by omega⟩
    simp only [PieceOk, show j + 2 ≠ 0 by omega, show j + 2 ≠ 1 by omega, ↓reduceIte, Nat.add_sub_cancel] at hpj
    obtain ⟨hrj, hri, hb, hd⟩ := hpj
    simp only [challengePhaseOf] at hj ⊢
    simp only [hround j hj]
    have ht := htime j hj
    simp only [hround j hj] at ht
    exact ⟨⟨o, ho, hrj, by rw [hoi, hri], hvc⟩, hb, hd, ht⟩

def honestWipe : ByteArray := ⟨Array.replicate 32 0⟩
def honestCoinKey : ByteArray := ⟨Array.replicate 256 0⟩
def honestSalt : ByteArray := ⟨Array.replicate 192 0⟩
def honestDigest : ByteArray := ⟨Array.replicate 64 0⟩
def honestRound : Round := ⟨0, 0, 0, 0, honestDigest⟩
def honestOpening : CoinOpening := ⟨0, 0, honestSalt, #[]⟩

theorem honest_leaf : ∃ leaf, coinLeaf honestCoinKey honestWipe 0 0 honestSalt = .ok leaf := by
  have hk : honestCoinKey.size = HM96.keyBytes ∧ honestCoinKey.get! (HM96.keyBytes - 1) &&& 128 = 0 := by
    refine ⟨by simp [honestCoinKey, ByteArray.size, HM96, Flock.Hm96.sha512, Flock.Hm96.keyBytes, Flock.Hm96.keyBits], ?_⟩
    have h255 : HM96.keyBytes - 1 = 255 := by simp [HM96, Flock.Hm96.sha512, Flock.Hm96.keyBytes, Flock.Hm96.keyBits]
    rw [h255]
    set_option maxRecDepth 100000 in decide
  have hs : honestSalt.size = HM96.saltBytes := by simp [honestSalt, ByteArray.size, HM96, Flock.Hm96.sha512]
  simp [coinLeaf, coinDigest, beN, checkKey, checkSalt, Bind.bind, Except.bind, Pure.pure, Except.pure, hk, hs]

def honestCommitment (leaf : ByteArray) : ChallengePhaseCommitment :=
  { session := honestWipe, coinRoot := (domain COIN_BINDING honestWipe 1).leaf 0 0 COIN_SCHEMA leaf,
    coinKey := honestCoinKey, blocks := 1, k := 1, limitNs := 20000000, blockBytes := 4096, binding := .empty }

theorem honest_coin {leaf : ByteArray} (h : coinLeaf honestCoinKey honestWipe 0 0 honestSalt = .ok leaf) :
    verifyCoin (honestCommitment leaf) honestOpening = true := by
  simp [verifyCoin, honestCommitment, honestOpening, h]
  simp [Flock.FV3.Domain.verifyOpening, Flock.FV3.Domain.climb, Flock.FV3.Scheme.climbL, domain,
    Flock.FV3.Sha512.Domain.make, Flock.FV3.Domain.make, Flock.FV3.Domain.leaf, Flock.FV3.Scheme.hash, Flock.FV3.hash,
    Flock.FV3.Sha512.scheme, Properties.Certifier.Bytes.hash_size]
  exact ⟨by decide, Properties.CertifierDevice.Proofs.Lms.beq_self _⟩

theorem honestDigest_size : honestDigest.size = 64 := by decide

theorem honest_message (leaf : ByteArray) : ∃ m, message (honestCommitment leaf) COMPLETE #[honestRound] = .ok m :=
  MemoryChallenger.Session.Properties.SecurityProofs.message_one _ _ honestDigest_size

theorem runGives_honest : RunGivesHonest := by
  obtain ⟨k, hkg, I, hk⟩ := Wipe.honest_keygen
  obtain ⟨-, hIk, -, hIs, -, -, -⟩ := Properties.CertifierDevice.Proofs.Lms.key_spec hk
  subst hIk
  have hroot : (k.tree[1]!).size = Lms.deployed.N :=
    Properties.CertifierDevice.Proofs.Lms.node_size hk (by decide) 1 le_rfl (by decide)
  have hpub : k.publicKey.size = 16 + Lms.deployed.N := by
    simp only [Lms.PrivateKey.publicKey, ByteArray.size_append, hIs, hroot]
  have hI : k.publicKey.extract 0 16 = k.I := ByteArray.extract_append_eq_left hIs.symm
  obtain ⟨leaf, hleaf⟩ := honest_leaf
  obtain ⟨m, hm⟩ := honest_message leaf
  obtain ⟨M, hM⟩ : ∃ M, memoryChallengerMessage k.I honestWipe m = .ok M := by
    unfold memoryChallengerMessage
    simp [hIs]
    exact ⟨_, rfl⟩
  obtain ⟨sig, hsig⟩ :=
    Properties.CertifierDevice.Proofs.Lms.lmsSigns 64 1 20 false Lms.deployed k.I _ k 0 M rfl hk (by decide)
  obtain ⟨hver, hdec⟩ :=
    Properties.CertifierDevice.Proofs.Lms.lmsCorrect 64 1 20 false Lms.deployed k.I _ k 0 M sig rfl hk hsig
  have hlms : lmsVerify Lms.deployed k.publicKey honestWipe m sig.encode = .ok true := by
    unfold lmsVerify
    simp only [hpub, ne_eq, not_true_eq_false, ↓reduceIte, hdec, hI, hM, hver, Bind.bind, Except.bind]
  have hcoin := honest_coin hleaf
  let V : Vals := ⟨m, sig.encode, COMPLETE, #[honestRound], honestRound⟩
  have hlook : SCHEMES.lookup LMS = some Lms.deployed := rfl
  have hpiece : ∀ u, PieceOk SCHEMES LMS k.publicKey (honestCommitment leaf) (fun _ => some honestDigest) (fun _ => 0) u
      V := by
    rintro ⟨u, hu⟩
    have hu' : u < 3 := hu
    rcases u with _ | _ | _ | u
    · simp only [PieceOk, ↓reduceIte]
      intro p hp
      rw [hlook] at hp
      cases hp
      exact hlms
    · simp only [PieceOk, show (0 + 1 : ℕ) ≠ 0 from by omega, ↓reduceIte]
      refine ⟨hm, rfl, rfl, fun j hj => ?_⟩
      obtain rfl : j = 0 := by simp [V] at hj; omega
      simp [V, honestRound, Round.elapsedNs, honestCommitment]
    · simp only [PieceOk, show (0 + 1 + 1 : ℕ) ≠ 0 from by omega, show (0 + 1 + 1 : ℕ) ≠ 1 from by omega, ↓reduceIte]
      exact ⟨rfl, rfl, by simp [V, honestRound, honestCommitment], rfl⟩
    · omega
  have hclear : ClearOk SCHEMES [k.publicKey] LMS k.publicKey (honestCommitment leaf) #[honestOpening] (fun _ => 0) := by
    refine ⟨⟨_, hlook⟩, List.mem_singleton.2 rfl, fun j hj => ?_⟩
    obtain rfl : j = 0 := by simp [honestCommitment] at hj; omega
    exact ⟨honestOpening, by simp [honestOpening], rfl, hcoin⟩
  refine ⟨_, k, fun _ => some honestDigest, honestCommitment leaf, #[honestOpening], fun _ => 0, fun _ => V, hkg, rfl,
    ?_, ?_, hclear, ?_⟩
  · simp [Gives, prPMF, Run.failing, constRun, hpiece]
  · intro _ _
    left
    refine ⟨rfl, fun j hj => ?_⟩
    obtain rfl : j = 0 := by simp [honestCommitment] at hj; omega
    rfl
  · refine verdict_accepted_of (p := Lms.deployed) (m := m) hlook
      (List.contains_iff_exists_mem_beq.2 ⟨k.publicKey, List.mem_singleton_self _,
        by show (k.publicKey.data == k.publicKey.data) = true; exact beq_self_eq_true _⟩)
      hm hlms (commitment_beq_self _) rfl rfl ?_
    intro j hj
    obtain rfl : j = 0 := by simp [challengePhaseOf, constRun, V] at hj; omega
    exact ⟨⟨honestOpening, by simp [honestOpening], rfl, rfl, hcoin⟩,
      by simp [challengePhaseOf, constRun, V, honestRound, honestCommitment], rfl,
      by simp [challengePhaseOf, constRun, V, honestRound, Round.elapsedNs],
      by simp [challengePhaseOf, constRun, V, honestRound, Round.elapsedNs, honestCommitment]⟩

def noRun : Vals := ⟨.empty, .empty, "", #[], default⟩

theorem runGives_nontrivial : RunGivesNontrivial := by
  refine ⟨default, .empty, fun _ => noRun, fun h => ?_⟩
  have hclear : ClearOk SCHEMES [ByteArray.empty] LMS .empty default #[] (fun _ => 0) :=
    ⟨⟨_, rfl⟩, List.mem_singleton.2 rfl, fun j hj => absurd hj (Nat.not_lt_zero _)⟩
  have hrej : (verdict SCHEMES [ByteArray.empty] (fun _ => none)
      (challengePhaseOf default LMS .empty noRun noRun) default #[]).accepted = false := by
    rw [Bool.eq_false_iff]
    intro ha
    obtain ⟨-, -, he, -⟩ := verdictAccepts _ _ _ _ _ _ ha
    exact absurd he (by decide)
  have hn : ¬ Hash.CollisionIn Verity.sha512 [] := by simp [Hash.CollisionIn]
  simp only [prPMF, PMF.toOuterMeasure_pure_apply, Set.mem_ofPred_eq] at h
  simp only [challengePhaseOf] at hrej
  simp only [constRun] at h
  simp only [hclear, hrej, hn, challengePhaseOf, and_true, not_false_eq_true, ↓reduceIte] at h
  exact absurd h (by simp)

end MemoryChallenger.Properties.InProofs.SecurityProofs
