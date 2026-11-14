module

public import Properties.Pous.InProofs
public import Proofs.Properties.Accounting.InProofs
public import Proofs.Properties.Pous.MemoryChallenger
import Proofs.Properties.Pous.WipeSigned
import Proofs.Properties.NetworkCertifier.Device.Lms

/-!
# Proof of PoUS's session inside proofs (`Properties.Pous.InProofs`)

When every proof block's value meets its piece (`PieceOk`), the carried rows open to one value and the verifier's clear
checks pass, the session the run opens meets every hypothesis of the converse of `VerdictAccepts`
(`verdict_accepted_of`), so `verdict` accepts it. So an accepted run in the event, with no collision, has a proof block
whose value fails `PieceOk`, and `Gives` at `right` gives it at `PieceOk` (`Gives.mono`).
-/

@[expose] public section

namespace MemoryChallenger.Properties.InProofs.SecurityProofs

open MemoryChallenger MemoryChallenger.Properties.InProofs MemoryChallenger.Properties.SecurityProofs
open Verity.InProofs Properties.Accounting.InProofs Verity.Game

theorem sessionGives : SessionGives := by
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
  refine hrej (verdict_accepted_of schemes trusted expected _ c openings p hp ht sd.m hmsg ?_ rfl he hsz ?_)
  · show lmsVerify p pk c.session sd.m sg.sig = .ok true
    rw [← hm]
    exact h0 p hp
  · intro j hj
    have hj' : j < c.k := by simp only [sessionOf] at hj; omega
    obtain ⟨o, ho, hoi, hvc⟩ := hco j hj'
    have hpj := hok ⟨j + 2, by omega⟩
    simp only [PieceOk, show j + 2 ≠ 0 by omega, show j + 2 ≠ 1 by omega, ↓reduceIte, Nat.add_sub_cancel] at hpj
    obtain ⟨hrj, hri, hb, hd⟩ := hpj
    simp only [sessionOf] at hj ⊢
    simp only [hround j hj]
    have ht := htime j hj
    simp only [hround j hj] at ht
    exact ⟨⟨o, ho, hrj, by rw [hoi, hri], hvc⟩, hb, hd, ht⟩

/-- The honest session's coin: round 0 at block 0, its tree's one leaf. -/
def honestSession : ByteArray := ⟨Array.replicate 32 0⟩
def honestCoinKey : ByteArray := ⟨Array.replicate 256 0⟩
def honestSalt : ByteArray := ⟨Array.replicate 192 0⟩
def honestRound : Round := ⟨0, 0, 0, 0, .empty⟩
def honestOpening : CoinOpening := ⟨0, 0, honestSalt, #[]⟩

theorem honest_leaf : ∃ leaf, coinLeaf honestCoinKey honestSession 0 0 honestSalt = .ok leaf := by
  have hk : honestCoinKey.size = HM96.keyBytes ∧ honestCoinKey.get! (HM96.keyBytes - 1) &&& 128 = 0 := by
    refine ⟨by simp [honestCoinKey, ByteArray.size, HM96, Flock.Hm96.sha512, Flock.Hm96.keyBytes, Flock.Hm96.keyBits], ?_⟩
    have h255 : HM96.keyBytes - 1 = 255 := by simp [HM96, Flock.Hm96.sha512, Flock.Hm96.keyBytes, Flock.Hm96.keyBits]
    rw [h255]
    set_option maxRecDepth 100000 in decide
  have hs : honestSalt.size = HM96.saltBytes := by simp [honestSalt, ByteArray.size, HM96, Flock.Hm96.sha512]
  simp [coinLeaf, coinDigest, beN, checkKey, checkSalt, Bind.bind, Except.bind, Pure.pure, Except.pure, hk, hs]

/-- The commitment of a session of one round over one block, its coin root the one leaf `leaf`. -/
def honestCommitment (leaf : ByteArray) : SessionCommitment :=
  { session := honestSession, coinRoot := (domain COIN_BINDING honestSession 1).leaf 0 0 COIN_SCHEMA leaf,
    coinKey := honestCoinKey, blocks := 1, k := 1, limitNs := 20000000, blockBytes := 4096, binding := .empty }

theorem honest_coin {leaf : ByteArray} (h : coinLeaf honestCoinKey honestSession 0 0 honestSalt = .ok leaf) :
    verifyCoin (honestCommitment leaf) honestOpening = true := by
  simp [verifyCoin, honestCommitment, honestOpening, h]
  simp [Flock.FV3.Domain.verifyOpening, Flock.FV3.Domain.climb, Flock.FV3.Scheme.climbL, domain,
    Flock.FV3.Sha512.Domain.make, Flock.FV3.Domain.make, Flock.FV3.Domain.leaf, Flock.FV3.Scheme.hash, Flock.FV3.hash,
    Flock.FV3.Sha512.scheme, Properties.NetworkCertifier.Bytes.hash_size]
  exact ⟨by decide, Properties.CertifierDevice.Proofs.Lms.beq_self _⟩

theorem honest_message (leaf : ByteArray) : ∃ m, message (honestCommitment leaf) COMPLETE #[honestRound] = .ok m := by
  simp [message, recordsRoot, Round.value, honestRound, beN, Bind.bind, Except.bind, Pure.pure, Except.pure]

theorem sessionGives_honest : SessionGivesHonest := by
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
  obtain ⟨M, hM⟩ : ∃ M, memoryChallengerMessage k.I honestSession m = .ok M := by
    unfold memoryChallengerMessage
    simp [hIs]
    exact ⟨_, rfl⟩
  obtain ⟨sig, hsig⟩ :=
    Properties.CertifierDevice.Proofs.Lms.lmsSigns 64 1 20 false Lms.deployed k.I _ k 0 M rfl hk (by decide)
  obtain ⟨hver, hdec⟩ :=
    Properties.CertifierDevice.Proofs.Lms.lmsCorrect 64 1 20 false Lms.deployed k.I _ k 0 M sig rfl hk hsig
  have hlms : lmsVerify Lms.deployed k.publicKey honestSession m sig.encode = .ok true := by
    unfold lmsVerify
    simp only [hpub, ne_eq, not_true_eq_false, ↓reduceIte, hdec, hI, hM, hver, Bind.bind, Except.bind]
  have hcoin := honest_coin hleaf
  let V : Vals := ⟨m, sig.encode, COMPLETE, #[honestRound], honestRound⟩
  have hlook : SCHEMES.lookup LMS = some Lms.deployed := rfl
  have hpiece : ∀ u, PieceOk SCHEMES LMS k.publicKey (honestCommitment leaf) (fun _ => some .empty) (fun _ => 0) u V := by
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
  refine ⟨_, k, fun _ => some .empty, honestCommitment leaf, #[honestOpening], fun _ => 0, fun _ => V, hkg, rfl,
    ?_, ?_, hclear, ?_⟩
  · simp [Gives, prPMF, Run.failing, constRun, hpiece]
  · intro _ _
    left
    refine ⟨rfl, fun j hj => ?_⟩
    obtain rfl : j = 0 := by simp [honestCommitment] at hj; omega
    rfl
  · refine verdict_accepted_of _ _ _ _ _ _ Lms.deployed hlook (List.mem_singleton.2 rfl) m hm hlms rfl rfl rfl ?_
    intro j hj
    obtain rfl : j = 0 := by simp [sessionOf, constRun, V] at hj; omega
    exact ⟨⟨honestOpening, by simp [honestOpening], rfl, rfl, hcoin⟩,
      by simp [sessionOf, constRun, V, honestRound, honestCommitment], rfl,
      by simp [sessionOf, constRun, V, honestRound, Round.elapsedNs],
      by simp [sessionOf, constRun, V, honestRound, Round.elapsedNs, honestCommitment]⟩

/-- The values of a session that ended nowhere: no digest, no signature, no rounds. -/
def noSession : Vals := ⟨.empty, .empty, "", #[], default⟩

theorem sessionGives_nontrivial : SessionGivesNontrivial := by
  refine ⟨default, .empty, fun _ => noSession, fun h => ?_⟩
  have hclear : ClearOk SCHEMES [ByteArray.empty] LMS .empty default #[] (fun _ => 0) :=
    ⟨⟨_, rfl⟩, List.mem_singleton.2 rfl, fun j hj => absurd hj (Nat.not_lt_zero _)⟩
  have hrej : (verdict SCHEMES [ByteArray.empty] (fun _ => none)
      (sessionOf default LMS .empty noSession noSession) default #[]).accepted = false := by
    rw [Bool.eq_false_iff]
    intro ha
    obtain ⟨-, -, he, -⟩ := verdictAccepts _ _ _ _ _ _ ha
    exact absurd he (by decide)
  have hn : ¬ Hash.CollisionIn Verity.sha512 [] := by simp [Hash.CollisionIn]
  simp only [prPMF, PMF.toOuterMeasure_pure_apply, Set.mem_ofPred_eq] at h
  simp only [sessionOf] at hrej
  simp only [constRun] at h
  simp only [hclear, hrej, hn, sessionOf, and_true, not_false_eq_true, ↓reduceIte] at h
  exact absurd h (by simp)

end MemoryChallenger.Properties.InProofs.SecurityProofs
