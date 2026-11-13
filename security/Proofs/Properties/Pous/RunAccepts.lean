module

public import Proofs.Properties.Pous.Run
public import Proofs.Properties.Pous.MemoryChallenger.Attest

/-!
# An accepted run of the composed check, without evaluating SHA-512

`runCheck_accepts`: `PousCheck.Run.check` accepts a run at P2's deployed point under S1's LMS. Every block's digest in
the table is zero, and the setup's root is `Merkle.mth` of the table's leaves. The session is complete: its 105
rounds each open block 0 with the table's row, on time. Each round's coin opens in a coin tree whose nodes are the
frame-v3 nodes over their children (`nodeAt`), so a coin's siblings climb to its root whatever the hashes are
(`climbL_nodeAt`). The session's message is signed at leaf 0 of a key `Lms.key` derives, the verifier's one trusted
key. Each digest a check compares is the expression it is compared with, so no SHA-512 is evaluated.
-/

@[expose] public section

namespace Pous.EndToEnd.SecurityProofs.Honest

open MemoryChallenger MemoryChallenger.Properties.SecurityProofs
open MemoryChallenger.Attest.Properties.SecurityProofs (pub_spec signs)

/-- `ByteArray`'s `==` is its arrays', which is lawful. -/
local instance instLawfulBEqByteArray_honest : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

theorem zeros_size (n : Nat) : (PousCheck.Prim.zeros n).size = n :=
  Array.size_replicate

theorem node_size (d : Flock.FV3.Sha512.Domain) (lv i : Nat) (a b : ByteArray) : (d.node lv i a b).size = 64 :=
  Properties.CertifierDevice.Proofs.Lms.sha512_size _

theorem leaf_size (d : Flock.FV3.Sha512.Domain) (r q : Nat) (s : String) (v : ByteArray) :
    (d.leaf r q s v).size = 64 :=
  Properties.CertifierDevice.Proofs.Lms.sha512_size _

/-! ## A frame-v3 tree over any leaves -/

/-- Node `i` of level `lv` of the frame-v3 tree over the leaves `lf`: leaf `i` at level 0, and above it the domain's
node over its two children. -/
def nodeAt (d : Flock.FV3.Sha512.Domain) (lf : Nat → ByteArray) : Nat → Nat → ByteArray
  | 0, i => lf i
  | lv + 1, i => d.node lv i (nodeAt d lf lv (2 * i)) (nodeAt d lf lv (2 * i + 1))

/-- The other child of `i`'s parent. -/
def sibling (i : Nat) : Nat := if i % 2 = 0 then i + 1 else i - 1

/-- The siblings of node `i` of level `lv` and of its next `K − 1` ancestors, lowest first. -/
def sibs (d : Flock.FV3.Sha512.Domain) (lf : Nat → ByteArray) : Nat → Nat → Nat → List ByteArray
  | 0, _, _ => []
  | K + 1, lv, i => nodeAt d lf lv (sibling i) :: sibs d lf K (lv + 1) (i / 2)

theorem climb_step (d : Flock.FV3.Sha512.Domain) (lf : Nat → ByteArray) (lv i : Nat) (rest : List ByteArray) :
    Flock.FV3.Sha512.scheme.climbL d.id (nodeAt d lf lv i) i lv (nodeAt d lf lv (sibling i) :: rest) =
      Flock.FV3.Sha512.scheme.climbL d.id (nodeAt d lf (lv + 1) (i / 2)) (i / 2) (lv + 1) rest := by
  simp only [Flock.FV3.Scheme.climbL]
  congr 1
  by_cases h : i % 2 = 0
  · simp only [sibling, h, ↓reduceIte]
    show _ = d.node lv (i / 2) (nodeAt d lf lv (2 * (i / 2))) (nodeAt d lf lv (2 * (i / 2) + 1))
    rw [show 2 * (i / 2) + 1 = i + 1 by omega, show 2 * (i / 2) = i by omega]
    rfl
  · simp only [sibling, h, ↓reduceIte]
    show _ = d.node lv (i / 2) (nodeAt d lf lv (2 * (i / 2))) (nodeAt d lf lv (2 * (i / 2) + 1))
    rw [show 2 * (i / 2) + 1 = i by omega, show 2 * (i / 2) = i - 1 by omega]
    rfl

/-- Node `i` of level `lv` climbs through its `K` siblings to its ancestor `K` levels up. -/
theorem climbL_nodeAt (d : Flock.FV3.Sha512.Domain) (lf : Nat → ByteArray) : ∀ K lv i,
    Flock.FV3.Sha512.scheme.climbL d.id (nodeAt d lf lv i) i lv (sibs d lf K lv i) = nodeAt d lf (lv + K) (i / 2 ^ K)
  | 0, lv, i => by simp [sibs, Flock.FV3.Scheme.climbL]
  | K + 1, lv, i => by
    rw [sibs, climb_step, climbL_nodeAt d lf K (lv + 1) (i / 2), Nat.div_div_eq_div_mul, ← Nat.pow_succ',
      show lv + 1 + K = lv + (K + 1) by omega]

theorem nodeAt_size (d : Flock.FV3.Sha512.Domain) (lf : Nat → ByteArray) (hlf : ∀ i, (lf i).size = 64) :
    ∀ lv i, (nodeAt d lf lv i).size = 64
  | 0, i => hlf i
  | lv + 1, i => node_size d lv i _ _

theorem sibs_length (d : Flock.FV3.Sha512.Domain) (lf : Nat → ByteArray) : ∀ K lv i, (sibs d lf K lv i).length = K
  | 0, _, _ => rfl
  | K + 1, lv, i => by simp [sibs, sibs_length d lf K]

theorem sibs_size (d : Flock.FV3.Sha512.Domain) (lf : Nat → ByteArray) (hlf : ∀ i, (lf i).size = 64) :
    ∀ K lv i, ∀ x ∈ sibs d lf K lv i, x.size = 64
  | 0, _, _ => by simp [sibs]
  | K + 1, lv, i => by
    intro x hx
    simp only [sibs, List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact nodeAt_size d lf hlf _ _
    · exact sibs_size d lf hlf K _ _ x hx

theorem log2ceil_105 : Flock.log2ceil 105 = 7 := by
  have h : Nat.log2 104 = 6 := (Nat.log2_eq_iff (by decide)).2 ⟨by decide, by decide⟩
  simp [Flock.log2ceil, h]

/-! ## The coins -/

def honestSid : ByteArray := PousCheck.Prim.zeros 32
def honestCoinKey : ByteArray := PousCheck.Prim.zeros 256
def honestSalt : ByteArray := PousCheck.Prim.zeros 192

theorem zeros_get! {n i : Nat} (h : i < n) : (PousCheck.Prim.zeros n).get! i = 0 := by
  show (Array.replicate n (0 : UInt8))[i]! = 0
  rw [getElem!_pos _ i (by simpa using h)]
  simp

theorem checkKey_honest : checkKey honestCoinKey = .ok () := by
  have hk : HM96.keyBytes = 256 := rfl
  unfold checkKey
  simp [hk, honestCoinKey, zeros_size, zeros_get! (show 255 < 256 by decide), pure, Except.pure]

theorem checkSalt_honest : checkSalt honestSalt = .ok () := by
  have hs : HM96.saltBytes = 192 := rfl
  unfold checkSalt
  simp [hs, honestSalt, zeros_size, pure, Except.pure]

/-- Coin `j`'s hm96 leaf, of block 0 under the honest key and salt. -/
def coinValue (j : Nat) : ByteArray :=
  match coinLeaf honestCoinKey honestSid j 0 honestSalt with
  | .ok v => v
  | .error _ => .empty

theorem coinLeaf_honest {j : Nat} (hj : j < 2 ^ 32) :
    coinLeaf honestCoinKey honestSid j 0 honestSalt = .ok (coinValue j) := by
  have h32 : j < 4294967296 := hj
  obtain ⟨v, hv⟩ : ∃ v, coinLeaf honestCoinKey honestSid j 0 honestSalt = .ok v := by
    unfold coinLeaf coinDigest
    simp [beN, h32, checkKey_honest, checkSalt_honest, bind, Except.bind, pure, Except.pure]
  unfold coinValue
  rw [hv]

/-- The coin tree's leaf at `j`: coin `j`'s leaf in the coin domain of 105 coins. -/
def coinLeafAt (j : Nat) : ByteArray := (domain COIN_BINDING honestSid 105).leaf j j COIN_SCHEMA (coinValue j)

def honestCoinRoot : ByteArray := nodeAt (domain COIN_BINDING honestSid 105) coinLeafAt 7 0

def honestOpening (j : Nat) : CoinOpening :=
  ⟨j, 0, honestSalt, (sibs (domain COIN_BINDING honestSid 105) coinLeafAt 7 0 j).toArray⟩

def honestOpenings : Array CoinOpening := (Array.range 105).map honestOpening

theorem verifyOpening_coin {j : Nat} (hj : j < 105) :
    (domain COIN_BINDING honestSid 105).verifyOpening honestCoinRoot 105 j (coinValue j) COIN_SCHEMA
      (honestOpening j).path = true := by
  have hlf : ∀ i, (coinLeafAt i).size = 64 := fun i => leaf_size _ _ _ _ _
  have hbytes : Flock.FV3.Sha512.scheme.bytes = 64 := rfl
  have hall : (honestOpening j).path.all (·.size == Flock.FV3.Sha512.scheme.bytes) = true := by
    rw [Array.all_eq_true]
    intro i hi
    have hm : (sibs (domain COIN_BINDING honestSid 105) coinLeafAt 7 0 j)[i]'(by simpa [honestOpening] using hi) ∈
        sibs (domain COIN_BINDING honestSid 105) coinLeafAt 7 0 j := List.getElem_mem _
    simp [honestOpening, hbytes, sibs_size _ _ hlf 7 0 j _ hm]
  have hlen : (honestOpening j).path.size = Flock.log2ceil (domain COIN_BINDING honestSid 105).count := by
    simp [honestOpening, sibs_length, log2ceil_105, domain, Flock.FV3.Sha512.Domain.make, Flock.FV3.Domain.make]
  have hclimb : (domain COIN_BINDING honestSid 105).climb j COIN_SCHEMA (coinValue j) (honestOpening j).path =
      honestCoinRoot := by
    have h := climbL_nodeAt (domain COIN_BINDING honestSid 105) coinLeafAt 7 0 j
    rw [Nat.zero_add, Nat.div_eq_of_lt (show j < 2 ^ 7 by omega)] at h
    exact h
  have hroot : honestCoinRoot.size = 64 := nodeAt_size _ _ hlf 7 0
  unfold Flock.FV3.Domain.verifyOpening
  rw [hclimb, hall]
  simp only [hroot, hbytes, hlen]
  simp [domain, Flock.FV3.Sha512.Domain.make, Flock.FV3.Domain.make, Flock.FV3.Sha512.scheme, hj]

theorem findRev_honest {j : Nat} (hj : j < 105) :
    honestOpenings.findRev? (·.j == j) = some (honestOpening j) := by
  rw [Array.findRev?_eq_find?_reverse]
  have hsome : (honestOpenings.reverse.find? (·.j == j)).isSome :=
    Array.find?_isSome.2 ⟨honestOpening j, by simpa [honestOpenings] using ⟨j, hj, rfl⟩, by simp [honestOpening]⟩
  obtain ⟨o, ho⟩ := Option.isSome_iff_exists.1 hsome
  have hp := Array.find?_some ho
  have hm := Array.mem_of_find?_eq_some ho
  simp only [honestOpenings, Array.mem_reverse, Array.mem_map, Array.mem_range] at hm
  obtain ⟨a, -, rfl⟩ := hm
  simp only [honestOpening, beq_iff_eq] at hp
  subst hp
  exact ho

/-! ## The table, the setup and the session -/

/-- Every block's digest zero. -/
def honestDigests : ByteArray :=
  PousCheck.Prim.zeros (PousCheck.P2.Params.deployed.blocks * PousCheck.Merkle.DIGEST)

def honestLeaves : Array ByteArray :=
  PousCheck.Run.rowLeaves PousCheck.P2.Params.deployed (PousCheck.Prim.zeros 24) honestDigests

def honestPub : PousCheck.SetupProtocol.Public :=
  ⟨PousCheck.P2.Params.deployed.blocks, PousCheck.Prim.zeros 64,
    PousCheck.Merkle.mth honestLeaves 0 honestLeaves.size, PousCheck.Prim.zeros 24⟩

theorem blocks_pos : 0 < PousCheck.P2.Params.deployed.blocks := by decide

theorem root_honest : (PousCheck.Merkle.root honestLeaves == some honestPub.root) = true := by
  have hne : honestLeaves.isEmpty = false := by
    rw [Bool.eq_false_iff]
    intro h
    rw [Array.isEmpty_iff_size_eq_zero] at h
    simp [honestLeaves, PousCheck.Run.rowLeaves] at h
    exact absurd h (by decide)
  unfold PousCheck.Merkle.root
  simp only [hne, Bool.false_eq_true, ↓reduceIte]
  exact beq_self_eq_true _

/-- Round `j`: coin `j` opened block 0, sent and received at 0 ns, with the table's row 0. -/
def honestRound (j : Nat) : Round := ⟨j, 0, 0, 0, PousCheck.Run.row honestDigests 0⟩

def honestRounds : Array Round := (Array.range 105).map honestRound

def honestCommitment : SessionCommitment :=
  ⟨honestSid, honestCoinRoot, honestCoinKey, PousCheck.P2.Params.deployed.blocks, 105,
    PousCheck.Run.DEPLOYED.limitNs, PousCheck.P2.Params.deployed.blockBytes, honestPub.root⟩

theorem commitment_beq_self (c : SessionCommitment) : (c == c) = true := by
  cases c
  simp only [BEq.beq]
  simp [instBEqSessionCommitment.beq]

theorem verifyCoin_honest {j : Nat} (hj : j < 105) : verifyCoin honestCommitment (honestOpening j) = true := by
  unfold verifyCoin
  have hc := coinLeaf_honest (j := j) (by omega)
  simp only [honestCommitment] at hc ⊢
  simp only [show (honestOpening j).j = j from rfl, show (honestOpening j).index = 0 from rfl,
    show (honestOpening j).salt = honestSalt from rfl, hc, hj, blocks_pos, decide_true, Bool.true_and]
  exact verifyOpening_coin hj

theorem rounds_honest : ∀ j (hj : j < honestRounds.size),
    (∃ o, honestOpenings.findRev? (·.j == j) = some o ∧ honestRounds[j].j = j ∧
      o.index = honestRounds[j].index ∧ verifyCoin honestCommitment o = true) ∧
    honestRounds[j].index < honestCommitment.blocks ∧
    PousCheck.Run.expected PousCheck.P2.Params.deployed honestDigests honestRounds[j].index =
      some honestRounds[j].digest ∧
    0 ≤ honestRounds[j].elapsedNs ∧ honestRounds[j].elapsedNs ≤ honestCommitment.limitNs := by
  intro j hj
  have hj' : j < 105 := by simpa [honestRounds] using hj
  have hr : honestRounds[j] = honestRound j := by simp [honestRounds]
  simp only [hr]
  exact ⟨⟨honestOpening j, findRev_honest hj', rfl, rfl, verifyCoin_honest hj'⟩,
    by simp [honestRound, honestCommitment, blocks_pos],
    by simp [PousCheck.Run.expected, honestRound, blocks_pos],
    by simp [Round.elapsedNs, honestRound],
    by simp [Round.elapsedNs, honestRound, honestCommitment]⟩

theorem mapM_ok {α β ε : Type} (f : α → Except ε β) :
    ∀ l : List α, (∀ a ∈ l, ∃ b, f a = .ok b) → ∃ bs, l.mapM f = .ok bs
  | [], _ => ⟨[], by simp; rfl⟩
  | a :: l, h => by
    obtain ⟨b, hb⟩ := h a List.mem_cons_self
    obtain ⟨bs, hbs⟩ := mapM_ok f l fun x hx => h x (List.mem_cons_of_mem _ hx)
    exact ⟨b :: bs, by simp [List.mapM_cons, hb, hbs, bind, Except.bind, pure, Except.pure]⟩

theorem message_honest : ∃ m, message honestCommitment COMPLETE honestRounds = .ok m := by
  obtain ⟨vs, hvs⟩ : ∃ vs, honestRounds.mapM Round.value = .ok vs := by
    rw [Array.mapM_eq_mapM_toList]
    obtain ⟨bs, hbs⟩ := mapM_ok Round.value honestRounds.toList fun a ha => by
      simp only [honestRounds, Array.toList_map, List.mem_map, Array.toList_range, List.mem_range] at ha
      obtain ⟨j, hj, rfl⟩ := ha
      have h32 : j < 4294967296 := by omega
      unfold Round.value
      simp [beN, honestRound, h32, bind, Except.bind, pure, Except.pure]
    exact ⟨bs.toArray, by rw [hbs]; rfl⟩
  unfold message recordsRoot
  simp only [hvs, bind, Except.bind, pure, Except.pure]
  exact ⟨_, rfl⟩

/-! ## The composed check -/

/-- `Run.checkAt` accepts a run that passes each of its checks. -/
theorem checkAt_of {schemes : Schemes} {pt : PousCheck.Run.Point} {trusted : List ByteArray}
    {pub : PousCheck.SetupProtocol.Public} {digests : ByteArray} {r : PousCheck.Run.ChallengePhaseRecord}
    {sid : ByteArray} (h1 : (pub.blocks == pt.params.blocks) = true)
    (h2 : (pub.salt.size == PousCheck.Run.SALT) = true)
    (h3 : (digests.size == pt.params.blocks * PousCheck.Merkle.DIGEST) = true)
    (h4 : (PousCheck.Merkle.root (PousCheck.Run.rowLeaves pt.params pub.salt digests) == some pub.root) = true)
    (h5 : (r.commitment.blocks == pt.params.blocks) = true) (h6 : (r.commitment.k == pt.k) = true)
    (h7 : decide (r.commitment.limitNs ≤ pt.limitNs) = true)
    (h8 : (r.commitment.blockBytes == pt.params.blockBytes) = true) (h9 : (r.commitment.binding == pub.root) = true)
    (h10 : (r.commitment.session == sid) = true)
    (h11 : (verdict schemes trusted (PousCheck.Run.expected pt.params digests) r.signed r.commitment
      r.openings).accepted = true) :
    PousCheck.Run.checkAt schemes pt trusted pub digests r sid = .ok () := by
  unfold PousCheck.Run.checkAt
  simp only [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, PousCheck.Run.need, ↓reduceIte, bind, Except.bind, pure,
    Except.pure]

/-- **The composed check accepts a run**, at P2's deployed point under S1's LMS: the table of zero digests, the setup
whose root is its leaves', and a complete session of 105 rounds at block 0, each coin opened in its tree and the
session signed at leaf 0 of a key `Lms.key` derives at `Lms.deployed`, which is the verifier's one trusted key. -/
theorem runCheck_accepts : ∃ (trusted : List ByteArray) (pub : PousCheck.SetupProtocol.Public) (digests : ByteArray)
    (r : PousCheck.Run.ChallengePhaseRecord) (sid : ByteArray), PousCheck.Run.check trusted pub digests r sid = .ok () := by
  obtain ⟨k, hk⟩ := Properties.CertifierDevice.Proofs.Lms.key_ok Lms.deployed (I := PousCheck.Prim.zeros 16) rfl
    (PousCheck.Prim.zeros 64)
  obtain ⟨hIs, hps, hpI⟩ := pub_spec hk
  obtain ⟨m, hm⟩ := message_honest
  obtain ⟨σ, hσs, hσv⟩ := signs hk
    (Flock.Sha512.hash (field MEMORY_CHALLENGER_TAG.toUTF8 ++ PousCheck.Prim.zeros 16 ++ field (field honestSid) ++
      field m))
  have hdec := (Properties.CertifierDevice.Proofs.Lms.lmsCorrect 64 1 20 false Lms.deployed _ _ k 0 _ σ rfl hk hσs).2
  have hv : lmsVerify Lms.deployed k.publicKey honestSid m σ.encode = .ok true := by
    unfold lmsVerify memoryChallengerMessage
    simp only [hps, hpI, hIs, hdec, ne_eq, not_true_eq_false, ↓reduceIte, bind, Except.bind, pure, Except.pure]
    exact hσv
  refine ⟨[k.publicKey], honestPub, honestDigests,
    ⟨⟨honestCommitment, COMPLETE, honestRounds, LMS, k.publicKey, σ.encode⟩, honestCommitment, honestOpenings⟩,
    honestSid, ?_⟩
  unfold PousCheck.Run.check
  refine checkAt_of (beq_self_eq_true _) (by simp [honestPub, zeros_size, PousCheck.Run.SALT])
    (by simp [honestDigests, zeros_size, PousCheck.Run.DEPLOYED]) root_honest (beq_self_eq_true _) (beq_self_eq_true _)
    (by simp [honestCommitment]) (beq_self_eq_true _) (beq_self_eq_true _) (beq_self_eq_true _) ?_
  exact verdict_accepted_of (p := Lms.deployed) (m := m) (by simp [SCHEMES]) (by simp) hm hv
    (commitment_beq_self _) rfl (by simp [honestRounds, honestCommitment]) rounds_honest

end Pous.EndToEnd.SecurityProofs.Honest
