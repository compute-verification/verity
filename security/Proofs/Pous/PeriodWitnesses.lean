module

public import Proofs.Pous.Obligation
public import Proofs.Pous.PeriodAt
public import Proofs.Pous.PeriodDeployed
public import Proofs.Pous.Pooled
public import Properties.Pous.Period.Witnesses
public import Proofs.Properties.Pous.SetupAccepts

/-!
# PoUS's per-period guarantees: their honest lemmas at one instance

`periodHeld_honest`, `obligationHeld_honest`, `periodHeldAt_honest` and `periodHeldDemo_honest` hold at every run and
audit meeting their hypotheses; here each is taken at one built instance (`Properties.Pous.Period.Witnesses`), so its
premises are shown to hold together, the certificate `HONEST_S` fitting the state bound beside the device's on-chip
state and side channel (`Fits`):

* `periodHeld`, `periodHeldDeployed`: the honest period (`honestRun` on `honestDevice`, one tick a round) and the honest
  strategy against `sidAudit`, a 32-byte session id with nothing else in it, at the deployed budget
  `(RING_D, RING_Q, RING_L)`; `periodHeld_certified`, `periodHeldDeployed_certified`: the same against the closed
  `certifiedAudit`, whose setup and 105-round block-0 session accept and whose trusted device holds that signed session,
  or two strings that this audit hashes collide.
* `periodCapacity`: the same, on the honest period whose capacity verdict is the verifier's whole decision outside a
  break (`verdictRun`), as `PeriodCapacity` takes the joint run's wiring.
* `obligationHeld`: the same, with the store's tags the snapshot's indices (`tag = id`) and the other tags' keys among
  the other audits' coins (`obligationRun`, `obligationAudit`), some strategy meeting both premises.
* `periodHeldAt`: the honest period at the deployed point `Run.DEPLOYED` and `N = 2^19`, whose `FULL` bits exceed
  `SAt (2^19)`; `periodHeldDemo`: a period at Demo B's `N = 1038·2^16` whose account holds its whole store
  (`demoRun`), above `SAt N`.

`periodCapacity` is nontrivial at the empty period (`emptyRun`), whose capacity verdict accepts every window: its
conclusion fails. The others' nontrivial lemmas are the accept-all events of `periodHeld_nontrivial` (at `2^19`
blocks) and of `periodHeldDemo_nontrivialDemo` (at Demo B's).
-/

@[expose] public section

namespace Pous.Period.SecurityProofs

open Pous.EndToEnd PousCheck PousCheck.Properties
open PousCheck.Properties.SecurityProofs.Accepts (storeStrings keyPrefix)
open Pous.EndToEnd.SecurityProofs.Honest
open scoped ENNReal

set_option exponentiation.threshold 20000

/-- `ByteArray`'s `==` is its arrays', which is lawful. -/
local instance instLawfulBEqByteArray_certified : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

theorem zero_mem_ticks : (0 : ℕ) ∈ (honestRun.win .capacity).ticks := by
  simp [honestRun, periodRun, ResourceAccounting.Window.ticks, K]

theorem honest_fits (r : ResourceAccounting.Run honestDevice Unit Unit Unit
    ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)) (hr : r.win .capacity = ⟨0, K⟩) :
    Fits honestDevice (r.win .capacity).len 1 HONEST_S Pous.EndToEnd.S := by
  rw [hr, S_eq]
  show (HONEST_S : ℝ) * ((K : ℕ) : ℝ) + ((2 ^ 30 : ℕ) : ℝ) + ((∑ _i : Unit, 437248000 * 1 : ℕ) : ℝ) ≤
    ((8169621180 : ℕ) : ℝ)
  norm_num [HONEST_S, K]

/-- **`periodHeld`'s honest lemma at one instance**: the honest period and strategy against `sidAudit` meet the device
model at the deployed budget and the instantiation, and the certificate fits. -/
theorem periodHeld_honestInstance :
    DeviceModel honestRun sidAudit honestStrategy RING_D RING_Q RING_L 1 ∧ Instantiates sidAudit honestStrategy ∧
      Fits honestDevice (honestRun.win .capacity).len 1 HONEST_S Pous.EndToEnd.S := by
  have h := periodHeld_honest honestRun sidAudit RING_D RING_Q RING_L 1 sidAudit_sid
    fun _ => ⟨0, zero_mem_ticks, le_rfl⟩
  exact ⟨h.1, h.2, honest_fits honestRun rfl⟩

/-- **`periodHeldDeployed`'s honest lemma**: `periodHeld_honestInstance`, its budget being the deployed one. -/
theorem periodHeldDeployed_honest :
    DeviceModel honestRun sidAudit honestStrategy RING_D RING_Q RING_L 1 ∧ Instantiates sidAudit honestStrategy ∧
      Fits honestDevice (honestRun.win .capacity).len 1 HONEST_S Pous.EndToEnd.S :=
  periodHeld_honestInstance

noncomputable def certifiedSetupState : Identity.Sha512.State :=
  Classical.choose PousCheck.Properties.SecurityProofs.Accepts.commitState_deployed

theorem certifiedSetupState_spec :
    Commit.commitState P2.Params.deployed = .ok certifiedSetupState :=
  Classical.choose_spec PousCheck.Properties.SecurityProofs.Accepts.commitState_deployed

def certifiedSalt : ByteArray := PousCheck.Prim.zeros 24

noncomputable def certifiedStore : Array (Array ByteArray) :=
  PousCheck.Properties.SecurityProofs.Accepts.honestEncoding (keyPrefix certifiedSalt)

noncomputable def certifiedCommitment : SetupProtocol.Commitment :=
  ⟨PousCheck.Properties.SecurityProofs.Accepts.honestW certifiedSalt certifiedSetupState, certifiedSalt⟩

noncomputable def certifiedPublic : SetupProtocol.Public :=
  ⟨P2.Params.deployed.blocks,
    PousCheck.Properties.SecurityProofs.Accepts.honestW certifiedSalt certifiedSetupState,
    (Merkle.root (PousCheck.Properties.SecurityProofs.Accepts.vkRun certifiedSalt certifiedSetupState
      (PousCheck.Properties.SecurityProofs.Accepts.honestSegment (keyPrefix certifiedSalt))
      (fun _ => PousCheck.Properties.SecurityProofs.Accepts.segPayload) P2.Params.deployed.segments).2).getD .empty,
    certifiedSalt⟩

noncomputable def certifiedPrivateKey : Lms.PrivateKey :=
  Classical.choose (Properties.CertifierDevice.Proofs.Lms.key_ok Lms.deployed
    (I := PousCheck.Prim.zeros 16) (zeros_size 16) (PousCheck.Prim.zeros 64))

theorem certifiedPrivateKey_spec :
    Lms.key Lms.deployed (PousCheck.Prim.zeros 16) (PousCheck.Prim.zeros 64) = .ok certifiedPrivateKey :=
  Classical.choose_spec (Properties.CertifierDevice.Proofs.Lms.key_ok Lms.deployed
    (I := PousCheck.Prim.zeros 16) (zeros_size 16) (PousCheck.Prim.zeros 64))

theorem certifiedDigests_le :
    64 ≤ (PousCheck.Properties.SecurityProofs.Accepts.setupDigests certifiedSalt).size :=
  size_le_of_table (by
    rw [PousCheck.Properties.SecurityProofs.Accepts.setupDigests]
    exact PousCheck.Properties.SecurityProofs.Accepts.digestTable_size
      (fun _ => PousCheck.Properties.SecurityProofs.Accepts.rowDigest_size _) _)

noncomputable def certifiedRecordMessage : ByteArray :=
  Classical.choose (message_honest certifiedPublic.root
    (PousCheck.Properties.SecurityProofs.Accepts.setupDigests certifiedSalt) certifiedDigests_le)

theorem certifiedRecordMessage_spec :
    MemoryAttester.message (honestCommitment certifiedPublic.root) MemoryAttester.COMPLETE
      (honestRounds (PousCheck.Properties.SecurityProofs.Accepts.setupDigests certifiedSalt)) =
        .ok certifiedRecordMessage :=
  Classical.choose_spec (message_honest certifiedPublic.root
    (PousCheck.Properties.SecurityProofs.Accepts.setupDigests certifiedSalt) certifiedDigests_le)

noncomputable def certifiedLmsMessage : ByteArray :=
  Flock.Sha512.hash (MemoryAttester.field MemoryAttester.MEMORY_CHALLENGER_TAG.toUTF8 ++
    PousCheck.Prim.zeros 16 ++ MemoryAttester.field (MemoryAttester.field honestSid) ++
    MemoryAttester.field certifiedRecordMessage)

noncomputable def certifiedSignature : Lms.Signature :=
  Classical.choose (MemoryAttester.Attest.Properties.SecurityProofs.signs
    certifiedPrivateKey_spec certifiedLmsMessage)

theorem certifiedSignature_spec :
    Lms.signAt certifiedPrivateKey 0 certifiedLmsMessage = .ok certifiedSignature ∧
      Lms.verify Lms.deployed certifiedPrivateKey.publicKey certifiedLmsMessage certifiedSignature = .ok true :=
  Classical.choose_spec (MemoryAttester.Attest.Properties.SecurityProofs.signs
    certifiedPrivateKey_spec certifiedLmsMessage)

noncomputable def certifiedRecord : PousCheck.Run.ChallengePhaseRecord :=
  ⟨⟨honestCommitment certifiedPublic.root, MemoryAttester.COMPLETE,
      honestRounds (PousCheck.Properties.SecurityProofs.Accepts.setupDigests certifiedSalt),
      MemoryAttester.LMS, certifiedPrivateKey.publicKey, certifiedSignature.encode⟩,
    honestCommitment certifiedPublic.root, honestOpenings⟩

/-- The honest session's coin tree: the honest session id, coin root and key, the deployed blocks, and 105 indices,
all block 0, as `honestRounds` records them. -/
def honestCoins : MemoryAttester.CoinTree :=
  ⟨honestSid, PousCheck.P2.Params.deployed.blocks, Array.replicate 105 0, honestCoinKey, #[], #[], honestCoinRoot⟩

/-- The memory challenger's honest run: `honestCoins`, the deployed limit and block size, bound to `root`, observing
`block` sent and received at time 0 in each of 105 rounds. -/
def honestChallengerRun (root block : ByteArray) : MemoryAttester.Properties.Run :=
  ⟨honestCoins, PousCheck.Run.DEPLOYED.limitNs, PousCheck.P2.Params.deployed.blockBytes, root,
    List.replicate 105 ⟨0, 0, block⟩⟩

noncomputable def certifiedStoreBlock : ByteArray := (flat certifiedStore)[0]!

/-- The closed honest audit: setup's honest store and output under the zero salt, one trusted LMS key and its accepted
session, and that key's challenger record of 105 timely answers to block 0. -/
noncomputable def certifiedAudit : Audits Unit Unit where
  trusted := [certifiedPrivateKey.publicKey]
  sid := honestSid
  commitment _ := certifiedCommitment
  store _ := certifiedStore
  pub _ := certifiedPublic
  digests _ := PousCheck.Properties.SecurityProofs.Accepts.setupDigests certifiedSalt
  W _ _ := false
  session _ := certifiedRecord
  device _ := ⟨fun pk =>
    if pk = certifiedPrivateKey.publicKey then [honestChallengerRun certifiedPublic.root certifiedStoreBlock] else []⟩

theorem certified_setup_accepts (hP : Pous.P2.Assumptions.PrimeP16448)
    (hk : PousCheck.Properties.SecurityProofs.Accepts.KeysOk (keyPrefix certifiedSalt)) :
    SetupProtocol.check P2.Params.deployed certifiedCommitment certifiedStore = .accept certifiedPublic () := by
  simpa [certifiedCommitment, certifiedStore, certifiedPublic, certifiedSalt] using
    PousCheck.Properties.SecurityProofs.Accepts.check_honest hP (zeros_size 24) hk certifiedSetupState_spec

/-! The certified store, its digest table and its root are closed terms, and their bodies are at hand to whoever
checks this module with every import's bodies: a step that leaves the kernel two syntactically different but equal
closed terms under a projection or a `BEq` can send it into building the store or hashing it. So each such step below
is a lemma over free variables (a salt, a store, a block, a table, a key), and its instance at the certified values
states its terms as the certified declarations state them. -/

theorem setupDigests_size (salt : ByteArray) :
    (PousCheck.Properties.SecurityProofs.Accepts.setupDigests salt).size =
      P2.Params.deployed.blocks * Merkle.DIGEST := by
  rw [PousCheck.Properties.SecurityProofs.Accepts.setupDigests]
  exact PousCheck.Properties.SecurityProofs.Accepts.digestTable_size
    (fun _ => PousCheck.Properties.SecurityProofs.Accepts.rowDigest_size _) _

theorem beq_some_getD {o : Option ByteArray} {x : ByteArray} (h : o = some x) :
    (o == some (o.getD .empty)) = true := by
  subst h
  rw [Option.getD_some]
  exact beq_self_eq_true _

/-- The root of setup's leaves under any salt is the root its published output records. -/
theorem root_setup_beq (salt : ByteArray) (st0 : Identity.Sha512.State) :
    (Merkle.root (PousCheck.Run.rowLeaves P2.Params.deployed salt
        (PousCheck.Properties.SecurityProofs.Accepts.setupDigests salt)) ==
      some ((Merkle.root (PousCheck.Properties.SecurityProofs.Accepts.vkRun salt st0
        (PousCheck.Properties.SecurityProofs.Accepts.honestSegment (keyPrefix salt))
        (fun _ => PousCheck.Properties.SecurityProofs.Accepts.segPayload) P2.Params.deployed.segments).2).getD
          .empty)) = true := by
  rw [PousCheck.Properties.SecurityProofs.Accepts.honest_leaves]
  exact beq_some_getD (PousCheck.Properties.SecurityProofs.Accepts.root_setup salt)

theorem certifiedPublic_blocks : certifiedPublic.blocks = P2.Params.deployed.blocks := by
  rw [certifiedPublic]

theorem certifiedPublic_saltSize : certifiedPublic.salt.size = PousCheck.Run.SALT := by
  simp [certifiedPublic, certifiedSalt, PousCheck.Run.SALT, zeros_size]

theorem certifiedPublic_root :
    (Merkle.root (PousCheck.Run.rowLeaves P2.Params.deployed certifiedPublic.salt
      (PousCheck.Properties.SecurityProofs.Accepts.setupDigests certifiedSalt)) == some certifiedPublic.root) = true := by
  simp only [certifiedPublic]
  exact root_setup_beq certifiedSalt certifiedSetupState

/-- The memory challenger's LMS check accepts the session's message `m` signed at leaf 0 of the key `k` derives. -/
theorem lmsVerify_of {k : Lms.PrivateKey} {m : ByteArray} {σ : Lms.Signature}
    (hk : Lms.key Lms.deployed (PousCheck.Prim.zeros 16) (PousCheck.Prim.zeros 64) = .ok k)
    (hσs : Lms.signAt k 0 (Flock.Sha512.hash (MemoryAttester.field MemoryAttester.MEMORY_CHALLENGER_TAG.toUTF8 ++
      PousCheck.Prim.zeros 16 ++ MemoryAttester.field (MemoryAttester.field honestSid) ++
      MemoryAttester.field m)) = .ok σ)
    (hσv : Lms.verify Lms.deployed k.publicKey (Flock.Sha512.hash
      (MemoryAttester.field MemoryAttester.MEMORY_CHALLENGER_TAG.toUTF8 ++ PousCheck.Prim.zeros 16 ++
        MemoryAttester.field (MemoryAttester.field honestSid) ++ MemoryAttester.field m)) σ = .ok true) :
    MemoryAttester.lmsVerify Lms.deployed k.publicKey honestSid m σ.encode = .ok true := by
  obtain ⟨hIs, hps, hpI⟩ := MemoryAttester.Attest.Properties.SecurityProofs.pub_spec hk
  have hdec := (Properties.CertifierDevice.Proofs.Lms.lmsCorrect 64 1 20 false Lms.deployed _ _ k 0 _ σ rfl hk
    hσs).2
  unfold MemoryAttester.lmsVerify MemoryAttester.memoryAttesterMessage
  simp only [hps, hpI, hIs, hdec, ne_eq, not_true_eq_false, ↓reduceIte, bind, Except.bind, pure, Except.pure]
  exact hσv

/-- `Run.check` accepts the honest session's record on a published setup `pub` and a table `digests` that agree: 105
rounds at block 0 with `digests`' row 0, bound to `pub.root`, its message `m` signed with `σ` at leaf 0 of the key
`k`, which is the one trusted. -/
theorem runCheck_honestRecord (pub : SetupProtocol.Public) (digests : ByteArray) (k : Lms.PrivateKey)
    (m : ByteArray) (σ : Lms.Signature) (hb : pub.blocks = P2.Params.deployed.blocks)
    (hs : pub.salt.size = PousCheck.Run.SALT) (hd : digests.size = P2.Params.deployed.blocks * Merkle.DIGEST)
    (hr : (Merkle.root (PousCheck.Run.rowLeaves P2.Params.deployed pub.salt digests) == some pub.root) = true)
    (hm : MemoryAttester.message (honestCommitment pub.root) MemoryAttester.COMPLETE (honestRounds digests) =
      .ok m)
    (hv : MemoryAttester.lmsVerify Lms.deployed k.publicKey honestSid m σ.encode = .ok true) :
    PousCheck.Run.check [k.publicKey] pub digests
      ⟨⟨honestCommitment pub.root, MemoryAttester.COMPLETE, honestRounds digests, MemoryAttester.LMS,
        k.publicKey, σ.encode⟩, honestCommitment pub.root, honestOpenings⟩ honestSid = .ok () := by
  unfold PousCheck.Run.check
  refine checkAt_of (by rw [hb]; exact beq_self_eq_true _) (by simp [hs])
    (by simp [hd, PousCheck.Run.DEPLOYED]) hr (beq_self_eq_true _) (beq_self_eq_true _)
    (by simp [honestCommitment]) (beq_self_eq_true _) (beq_self_eq_true _) (beq_self_eq_true _) ?_
  exact MemoryAttester.Properties.SecurityProofs.verdict_accepted_of (p := Lms.deployed) (m := m)
    (by simp [MemoryAttester.SCHEMES]) (by simp) hm hv
    (MemoryAttester.Properties.SecurityProofs.commitment_beq_self _) rfl
    (by simp [honestRounds, honestCommitment]) (rounds_honest pub.root digests)

theorem certified_run_accepts :
    PousCheck.Run.check [certifiedPrivateKey.publicKey] certifiedPublic
      (PousCheck.Properties.SecurityProofs.Accepts.setupDigests certifiedSalt) certifiedRecord honestSid = .ok () := by
  rw [certifiedRecord]
  exact runCheck_honestRecord certifiedPublic (PousCheck.Properties.SecurityProofs.Accepts.setupDigests certifiedSalt)
    certifiedPrivateKey certifiedRecordMessage certifiedSignature certifiedPublic_blocks certifiedPublic_saltSize
    (setupDigests_size certifiedSalt) certifiedPublic_root certifiedRecordMessage_spec
    (lmsVerify_of certifiedPrivateKey_spec certifiedSignature_spec.1 certifiedSignature_spec.2)

/-- The strings an audit's play hashes at the coin `ω`: the store's key-stream inputs under its salt (`storeStrings`),
and the strings `RunBreak` and `TableBreak` read. -/
def playStrings (P : Audits Unit Unit)
    (ω : Unit × Unit × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)) : List ByteArray :=
  storeStrings (keyPrefix (P.commitment (ω.1, ω.2.1)).salt) ++
    (MemoryAttester.Properties.runStrings (P.session ω).signed (P.device ω) ++
      blockStrings (P.session ω).signed.publicKey (P.device ω) (P.store (ω.1, ω.2.1))) ++
    tableStrings (P.pub (ω.1, ω.2.1)) (P.digests (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1))

/-- **A collision in an audit's play**: at some coin, two different strings of `playStrings` have one SHA-512
digest. -/
def PlayCollision (P : Audits Unit Unit) : Prop :=
  ∃ ω, ∃ x ∈ playStrings P ω, ∃ y ∈ playStrings P ω,
    x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y

theorem certifiedStore_shaped
    (hk : PousCheck.Properties.SecurityProofs.Accepts.KeysOk (keyPrefix certifiedSalt)) :
    Shaped certifiedStore := by
  refine ⟨?_, ?_⟩
  · change (PousCheck.Properties.SecurityProofs.Accepts.honestEncoding (keyPrefix certifiedSalt)).size =
      PousCheck.P2.Params.deployed.segments
    rw [PousCheck.Properties.SecurityProofs.Accepts.honestEncoding_size]
    rfl
  intro s hs
  have hs8 : s < 8 := by
    simpa [certifiedStore, PousCheck.Properties.SecurityProofs.Accepts.honestEncoding_size] using hs
  have hseg : certifiedStore[s] =
      PousCheck.Properties.SecurityProofs.Accepts.honestSegment (keyPrefix certifiedSalt) s := by
    exact PousCheck.Properties.SecurityProofs.Accepts.honestEncoding_get _ s (by simpa [certifiedStore] using hs)
  rw [hseg]
  refine ⟨PousCheck.Properties.SecurityProofs.Accepts.honestSegment_size _ _, fun v hv => ?_⟩
  rw [PousCheck.Properties.SecurityProofs.Accepts.honestSegment_get]
  obtain ⟨k, t, -, hkey⟩ := hk s v hs8 (by
    have := PousCheck.Properties.SecurityProofs.Accepts.honestSegment_size (keyPrefix certifiedSalt) s
    omega)
  rw [PousCheck.Properties.SecurityProofs.Accepts.honestBlock_eq hkey,
    PousCheck.Properties.SecurityProofs.beOfNat_size]
  rfl

/-- Block 0 of a shaped store, the first of its flattened blocks, is block 0 of its segment 0. -/
theorem flat_zero_of_some {C : Array (Array ByteArray)} (hC : Shaped C) {seg : Array ByteArray} {b : ByteArray}
    (hseg : C[0]? = some seg) (hb : seg[0]? = some b) : (flat C)[0]! = b := by
  have hflat := Pous.EndToEnd.SecurityProofs.flat_getElem? hC 0 0 (by decide)
  simp only [Nat.zero_mul, Nat.zero_add, hseg, Option.bind_some, hb] at hflat
  exact Pous.EndToEnd.SecurityProofs.getElem!_of_some hflat

theorem honestEncoding_getElem?_zero (pre : ByteArray) :
    (PousCheck.Properties.SecurityProofs.Accepts.honestEncoding pre)[0]? =
      some (PousCheck.Properties.SecurityProofs.Accepts.honestSegment pre 0) := by
  have h : 0 < (PousCheck.Properties.SecurityProofs.Accepts.honestEncoding pre).size := by
    rw [PousCheck.Properties.SecurityProofs.Accepts.honestEncoding_size]
    decide
  rw [Array.getElem?_eq_getElem h, PousCheck.Properties.SecurityProofs.Accepts.honestEncoding_get pre 0 h]

theorem honestSegment_getElem?_zero (pre : ByteArray) (s : Nat) :
    (PousCheck.Properties.SecurityProofs.Accepts.honestSegment pre s)[0]? =
      some (PousCheck.Properties.SecurityProofs.Accepts.honestBlock pre s 0) := by
  have h : 0 < (PousCheck.Properties.SecurityProofs.Accepts.honestSegment pre s).size := by
    rw [PousCheck.Properties.SecurityProofs.Accepts.honestSegment_size]
    decide
  rw [Array.getElem?_eq_getElem h, PousCheck.Properties.SecurityProofs.Accepts.honestSegment_get pre s 0 h]

theorem certifiedStoreBlock_eq
    (hk : PousCheck.Properties.SecurityProofs.Accepts.KeysOk (keyPrefix certifiedSalt)) :
    certifiedStoreBlock =
      PousCheck.Properties.SecurityProofs.Accepts.honestBlock (keyPrefix certifiedSalt) 0 0 := by
  rw [certifiedStoreBlock, certifiedStore]
  exact flat_zero_of_some (certifiedStore_shaped hk) (honestEncoding_getElem?_zero _) (honestSegment_getElem?_zero _ 0)

theorem certifiedStoreBlock_size
    (hk : PousCheck.Properties.SecurityProofs.Accepts.KeysOk (keyPrefix certifiedSalt)) :
    certifiedStoreBlock.size = P2.Params.deployed.blockBytes := by
  rw [certifiedStoreBlock_eq hk]
  obtain ⟨k, t, -, hkey⟩ := hk 0 0 (by decide) (by decide)
  rw [PousCheck.Properties.SecurityProofs.Accepts.honestBlock_eq hkey,
    PousCheck.Properties.SecurityProofs.beOfNat_size]
  rfl

/-- Under any salt, the memory challenger's digest of honest block 0 is row 0 of setup's digest table. -/
theorem honestBlock_digest_row (salt : ByteArray) :
    MemoryAttester.blockDigest (PousCheck.Properties.SecurityProofs.Accepts.honestBlock (keyPrefix salt) 0 0) =
      PousCheck.Run.row (PousCheck.Properties.SecurityProofs.Accepts.setupDigests salt) 0 := by
  rw [PousCheck.Properties.SecurityProofs.Accepts.setupDigests,
    PousCheck.Properties.SecurityProofs.Accepts.digestTable_row
      (fun _ => PousCheck.Properties.SecurityProofs.Accepts.rowDigest_size _)
      P2.Params.deployed.blocks 0 (by decide)]
  simp only [Nat.zero_div, Nat.zero_mod]
  rfl

theorem certifiedStoreBlock_digest
    (hk : PousCheck.Properties.SecurityProofs.Accepts.KeysOk (keyPrefix certifiedSalt)) :
    MemoryAttester.blockDigest certifiedStoreBlock =
      PousCheck.Run.row (PousCheck.Properties.SecurityProofs.Accepts.setupDigests certifiedSalt) 0 := by
  rw [certifiedStoreBlock_eq hk]
  exact honestBlock_digest_row certifiedSalt

noncomputable def certifiedObserved : MemoryAttester.Properties.Observed := ⟨0, 0, certifiedStoreBlock⟩

/-- The challenger's record of `n ≤ 105` timely answers with a block `b` of the deployed size whose digest is row 0
of a table `d`: `n` honest rounds of `d`, complete at 105. -/
theorem recorded_prefix_of {b d : ByteArray} (hb : b.size = P2.Params.deployed.blockBytes)
    (hd : MemoryAttester.blockDigest b = PousCheck.Run.row d 0) :
    ∀ n, n ≤ 105 →
      (List.replicate n (⟨0, 0, b⟩ : MemoryAttester.Properties.Observed)).foldl
          (MemoryAttester.Properties.Recorded.round honestCoins PousCheck.Run.DEPLOYED.limitNs
            P2.Params.deployed.blockBytes) (⟨#[], none⟩ : MemoryAttester.Properties.Recorded) =
        ⟨(Array.range n).map (honestRound d), if n = 105 then some MemoryAttester.COMPLETE else none⟩
  | 0, _ => by simp
  | n + 1, hn => by
      rw [List.replicate_succ', List.foldl_append, recorded_prefix_of hb hd n (by omega)]
      simp only [List.foldl_cons, List.foldl_nil]
      have hn105 : n ≠ 105 := by omega
      simp only [hn105, ↓reduceIte]
      have hidx : (Array.replicate 105 0)[n]! = 0 := by
        rw [getElem!_pos _ n (by simp; omega)]
        simp
      simp [MemoryAttester.Properties.Recorded.round, hb, honestCoins, MemoryAttester.CoinTree.k, hidx, hd,
        honestRound, MemoryAttester.Round.elapsedNs, Array.range_succ]

theorem honest_recorded_prefix
    (hk : PousCheck.Properties.SecurityProofs.Accepts.KeysOk (keyPrefix certifiedSalt)) :
    ∀ n, n ≤ 105 →
      (List.replicate n certifiedObserved).foldl
          (MemoryAttester.Properties.Recorded.round honestCoins PousCheck.Run.DEPLOYED.limitNs
            P2.Params.deployed.blockBytes) (⟨#[], none⟩ : MemoryAttester.Properties.Recorded) =
        ⟨(Array.range n).map
            (honestRound (PousCheck.Properties.SecurityProofs.Accepts.setupDigests certifiedSalt)),
          if n = 105 then some MemoryAttester.COMPLETE else none⟩ :=
  recorded_prefix_of (certifiedStoreBlock_size hk) (certifiedStoreBlock_digest hk)

/-- The challenger's honest run bound to any root with a block `b` of the deployed size whose digest is row 0 of a
table `d` records the 105 honest rounds of `d`, complete. -/
theorem challengerRun_recorded_of (root : ByteArray) {b d : ByteArray} (hb : b.size = P2.Params.deployed.blockBytes)
    (hd : MemoryAttester.blockDigest b = PousCheck.Run.row d 0) :
    (honestChallengerRun root b).recorded = ⟨honestRounds d, some MemoryAttester.COMPLETE⟩ := by
  unfold MemoryAttester.Properties.Run.recorded honestChallengerRun
  simpa [honestRounds] using recorded_prefix_of hb hd 105 le_rfl

theorem honestChallengerRun_recorded
    (hk : PousCheck.Properties.SecurityProofs.Accepts.KeysOk (keyPrefix certifiedSalt)) :
    (honestChallengerRun certifiedPublic.root certifiedStoreBlock).recorded =
      ⟨honestRounds (PousCheck.Properties.SecurityProofs.Accepts.setupDigests certifiedSalt),
        some MemoryAttester.COMPLETE⟩ :=
  challengerRun_recorded_of certifiedPublic.root (certifiedStoreBlock_size hk) (certifiedStoreBlock_digest hk)

theorem certified_signed
    (hk : PousCheck.Properties.SecurityProofs.Accepts.KeysOk (keyPrefix certifiedSalt))
    (ω : Unit × Unit × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    {m : ByteArray} (hm : MemoryAttester.Properties.challengePhaseMessage (certifiedAudit.session ω).signed = some m) :
    MemoryAttester.Properties.Signed (certifiedAudit.device ω) certifiedPrivateKey.publicKey m := by
  let run := honestChallengerRun certifiedPublic.root certifiedStoreBlock
  refine ⟨run, ?_, MemoryAttester.COMPLETE, certifiedRecordMessage, ?_, ?_, ?_⟩
  · simp [certifiedAudit, run]
  · simpa [run] using congrArg MemoryAttester.Properties.Recorded.ended (honestChallengerRun_recorded hk)
  · have hr := honestChallengerRun_recorded hk
    change MemoryAttester.message (honestCommitment certifiedPublic.root) MemoryAttester.COMPLETE
      run.recorded.rounds = .ok certifiedRecordMessage
    rw [show run.recorded = ⟨honestRounds
      (PousCheck.Properties.SecurityProofs.Accepts.setupDigests certifiedSalt),
        some MemoryAttester.COMPLETE⟩ from hr]
    exact certifiedRecordMessage_spec
  · change ((MemoryAttester.message (honestCommitment certifiedPublic.root) MemoryAttester.COMPLETE
      (honestRounds (PousCheck.Properties.SecurityProofs.Accepts.setupDigests certifiedSalt)) >>=
        MemoryAttester.memoryAttesterMessage (certifiedPrivateKey.publicKey.extract 0 16) honestSid).toOption) =
      some m at hm
    rw [certifiedRecordMessage_spec] at hm
    simp only [bind, Except.bind] at hm
    change MemoryAttester.memoryAttesterMessage (certifiedPrivateKey.publicKey.extract 0 16)
      honestSid certifiedRecordMessage = .ok m
    have toOption_eq_ok {ε α : Type} {e : Except ε α} {a : α} (h : e.toOption = some a) : e = .ok a := by
      cases e <;> simp_all [Except.toOption]
    exact toOption_eq_ok hm

theorem playCollision_of_run {P : Audits Unit Unit}
    {ω : Unit × Unit × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)}
    (h : ∃ x ∈ MemoryAttester.Properties.runStrings (P.session ω).signed (P.device ω) ++
        blockStrings (P.session ω).signed.publicKey (P.device ω) (P.store (ω.1, ω.2.1)),
      ∃ y ∈ MemoryAttester.Properties.runStrings (P.session ω).signed (P.device ω) ++
        blockStrings (P.session ω).signed.publicKey (P.device ω) (P.store (ω.1, ω.2.1)),
        x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y) :
    PlayCollision P := by
  rcases h with ⟨x, hx, y, hy, hne, heq⟩
  have mem_play {z : ByteArray} (hz : z ∈
      MemoryAttester.Properties.runStrings (P.session ω).signed (P.device ω) ++
        blockStrings (P.session ω).signed.publicKey (P.device ω) (P.store (ω.1, ω.2.1))) :
      z ∈ playStrings P ω := by
    rcases List.mem_append.1 hz with hz | hz
    · exact List.mem_append_left _ (List.mem_append_right _ (List.mem_append_left _ hz))
    · exact List.mem_append_left _ (List.mem_append_right _ (List.mem_append_right _ hz))
  exact ⟨ω, x, mem_play hx, y, mem_play hy, hne, heq⟩

theorem playCollision_of_table {P : Audits Unit Unit}
    {ω : Unit × Unit × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)}
    (h : TableBreak (P.pub (ω.1, ω.2.1)) (P.digests (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1))) :
    PlayCollision P := by
  rcases h with ⟨x, hx, y, hy, hne, heq⟩
  exact ⟨ω, x, List.mem_append_right _ hx, y, List.mem_append_right _ hy, hne, heq⟩

theorem certified_not_break
    (hk : PousCheck.Properties.SecurityProofs.Accepts.KeysOk (keyPrefix certifiedSalt))
    (ω : Unit × Unit × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (hcol : ¬ PlayCollision certifiedAudit) :
    ¬ certifiedAudit.Break ω := by
  intro h
  rcases h with hrun | htable
  · rcases hrun with hhash | hforge
    · exact hcol (playCollision_of_run hhash)
    · rcases hforge with ⟨p, m, hp, hkey, hm, hv, hnot⟩
      exact hnot (certified_signed hk ω hm)
  · exact hcol (playCollision_of_table htable)

/-- **`periodHeld`'s closed certified honest instance**: the honest period and strategy meet the device model at the
literal deployed budget, the instantiation and fit; and `certifiedAudit` has an accepted play outside its own bounded
collision event. The session challenges block 0 in each of its 105 rounds. Nothing ties that session to the ring
game's sample at the coin, because no premise asks it. -/
theorem periodHeld_certified (hP : Pous.P2.Assumptions.PrimeP16448) :
    DeviceModel honestRun certifiedAudit honestStrategy RING_D RING_Q RING_L 1 ∧
      Instantiates certifiedAudit honestStrategy ∧
      Fits honestDevice (honestRun.win .capacity).len 1 HONEST_S Pous.EndToEnd.S ∧
      ((∃ ω, certifiedAudit.Accepts ω ∧ ¬ certifiedAudit.Break ω) ∨ PlayCollision certifiedAudit) := by
  have hmodel := periodHeld_honest honestRun certifiedAudit RING_D RING_Q RING_L 1
    (by simpa [certifiedAudit, honestSid] using zeros_size 32)
    fun _ => ⟨0, zero_mem_ticks, le_rfl⟩
  refine ⟨hmodel.1, hmodel.2, honest_fits honestRun rfl, ?_⟩
  obtain ⟨x⟩ := (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).nonemptyΩ
  let ω : Unit × Unit × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams) :=
    ((), (), x, default, default)
  rcases PousCheck.Properties.SecurityProofs.Accepts.keysOk_or_collision (keyPrefix certifiedSalt)
      (PousCheck.Properties.SecurityProofs.Accepts.pad128_size _) with hk | hstore
  · by_cases hcollision : PlayCollision certifiedAudit
    · exact Or.inr hcollision
    · refine Or.inl ⟨ω, ?_, certified_not_break hk ω hcollision⟩
      change SetupProtocol.check P2.Params.deployed certifiedCommitment certifiedStore =
          .accept certifiedPublic () ∧
        PousCheck.Run.check [certifiedPrivateKey.publicKey] certifiedPublic
          (PousCheck.Properties.SecurityProofs.Accepts.setupDigests certifiedSalt) certifiedRecord honestSid = .ok ()
      exact ⟨certified_setup_accepts hP hk, certified_run_accepts⟩
  · rcases hstore with ⟨a, ha, b, hb, hab, heq⟩
    refine Or.inr ⟨ω, a, ?_, b, ?_, hab, heq⟩
    · simp [playStrings, certifiedAudit, certifiedCommitment, ha]
    · simp [playStrings, certifiedAudit, certifiedCommitment, hb]

/-- **`periodHeldDeployed`'s certified honest instance**: `periodHeld_certified`, at the deployed budget. -/
theorem periodHeldDeployed_certified (hP : Pous.P2.Assumptions.PrimeP16448) :
    DeviceModel honestRun certifiedAudit honestStrategy RING_D RING_Q RING_L 1 ∧
      Instantiates certifiedAudit honestStrategy ∧
      Fits honestDevice (honestRun.win .capacity).len 1 HONEST_S Pous.EndToEnd.S ∧
      ((∃ ω, certifiedAudit.Accepts ω ∧ ¬ certifiedAudit.Break ω) ∨ PlayCollision certifiedAudit) :=
  periodHeld_certified hP

/-- **`periodCapacity`'s honest lemma at one instance**: on the honest period whose capacity verdict is the verifier's
whole decision outside a break, the device model, the instantiation, the certificate's fit and the verdict's wiring
hold. -/
theorem periodCapacity_honestInstance :
    DeviceModel verdictRun sidAudit honestStrategy RING_D RING_Q RING_L 1 ∧ Instantiates sidAudit honestStrategy ∧
      Fits honestDevice (verdictRun.win .capacity).len 1 HONEST_S Pous.EndToEnd.S ∧
        ∀ ω, verdictRun.accepts .capacity ω → sidAudit.Accepts ω ∧ ¬ sidAudit.Break ω := by
  have h := periodHeld_honest verdictRun sidAudit RING_D RING_Q RING_L 1 sidAudit_sid
    fun _ => ⟨0, zero_mem_ticks, le_rfl⟩
  exact ⟨h.1, h.2, honest_fits verdictRun rfl, fun _ hω => hω⟩

/-- **`periodCapacity` is nontrivial**: at the empty period (`emptyRun`), whose capacity verdict accepts every window
and whose account holds nothing, the certificate `HONEST_S` is not sound per run at `δ + ε_max`: every window is
accepted short. -/
theorem periodCapacity_nontrivialEmpty :
    ¬ (emptyRun.CapacitySoundPerRun HONEST_S ((Params.δ : ℝ) + Params.εMax.toReal) ∧
      emptyRun.CapacitySound HONEST_S ((Params.δ : ℝ) + Params.εMax.toReal)) := by
  rintro ⟨h, -⟩
  have h1 : Pouw.pr (fun x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams =>
      emptyRun.accepts .capacity ((), (), x) ∧
        (emptyRun.held ((), (), x) : ℝ) < HONEST_S * (emptyRun.win .capacity).len) = 1 := by
    refine Pouw.pr_eq_one fun x => ⟨trivial, ?_⟩
    have he : emptyRun.held ((), (), x) = K * 0 := held_periodRun 0 _ _
    rw [he]
    show ((K * 0 : ℕ) : ℝ) < (HONEST_S : ℝ) * ((K : ℕ) : ℝ)
    norm_num [HONEST_S, K]
  have h2 := h1.symm.le.trans (h () ())
  rw [εMax_toReal] at h2
  norm_num [Params.δ] at h2

theorem obligationRun_mem_ticks : (0 : ℕ) ∈ (obligationRun.win .capacity).ticks := by
  simp [obligationRun, ResourceAccounting.Window.ticks, K]

/-- **`obligationHeld`'s honest lemma at one instance**: at the snapshot `tag = id`, the honest period and a strategy
(the lookup strategy) against `obligationAudit` meet the device model at the deployed budget and the instantiation, and
the certificate fits. -/
theorem obligationHeld_honestInstance :
    ∃ τ : Strategy Unit (Unit × IdKeys),
      DeviceModel obligationRun obligationAudit τ RING_D RING_Q RING_L 1 ∧ Instantiates obligationAudit τ ∧
        Fits honestDevice (obligationRun.win .capacity).len 1 HONEST_S Pous.EndToEnd.S := by
  have h := periodHeld_honest obligationRun obligationAudit RING_D RING_Q RING_L 1 rfl
    fun _ => ⟨0, obligationRun_mem_ticks, le_rfl⟩
  refine ⟨lookup Unit (Unit × IdKeys), h.1, h.2, ?_⟩
  rw [S_eq]
  show (HONEST_S : ℝ) * ((K : ℕ) : ℝ) + ((2 ^ 30 : ℕ) : ℝ) + ((∑ _i : Unit, 437248000 * 1 : ℕ) : ℝ) ≤
    ((8169621180 : ℕ) : ℝ)
  norm_num [HONEST_S, K]

/-- **`periodHeldAt`'s honest lemma at one instance**: at the deployed point and `2^19` blocks, the honest period, whose
account holds `FULL` bits above `SAt (2^19)`, meets the ring-model device premise at the deployed budget, and the
certificate fits `SAt (2^19)`. -/
theorem periodHeldAt_honestInstance :
    RingDeviceAt (2 ^ 19) PousCheck.Run.DEPLOYED honestRun (sidAuditAt (2 ^ 19)) RING_D RING_Q RING_L 1 ∧
      Fits honestDevice (honestRun.win .capacity).len 1 HONEST_S (SAt (2 ^ 19)) := by
  have hS : SAt (2 ^ 19) = 8169621180 := PousP2Ring.stateBound_p2ring_b19 _ _
  refine ⟨ringDeviceAt_honest (2 ^ 19) PousCheck.Run.DEPLOYED honestRun (sidAuditAt (2 ^ 19)) RING_D RING_Q RING_L 1
    fun _ => ⟨0, zero_mem_ticks, ?_⟩, ?_⟩
  · rw [hS]
    show 8169621180 < FULL
    norm_num [FULL]
  · rw [hS]
    show (HONEST_S : ℝ) * ((K : ℕ) : ℝ) + ((2 ^ 30 : ℕ) : ℝ) + ((∑ _i : Unit, 437248000 * 1 : ℕ) : ℝ) ≤
      ((8169621180 : ℕ) : ℝ)
    norm_num [HONEST_S, K]

theorem demoRun_mem_ticks : (0 : ℕ) ∈ (demoRun.win .capacity).ticks := by
  simp [demoRun, ResourceAccounting.Window.ticks, K]

/-- **`periodHeldDemo`'s honest lemma at one instance**: at Demo B's point and `N = 1038·2^16`, a period whose account
holds the whole store, above `SAt N`, meets the ring-model device premise at the deployed budget, and the certificate
fits Demo B's state bound. -/
theorem periodHeldDemo_honestInstance :
    RingDeviceAt (1038 * 2 ^ 16) demoPoint demoRun (sidAuditAt (1038 * 2 ^ 16)) RING_D RING_Q RING_L 1 ∧
      Fits demoDevice (demoRun.win .capacity).len 1 HONEST_S 1060008348186 := by
  have hS : SAt (1038 * 2 ^ 16) = 1060008348186 := Pous.SecurityProofs.stateBound_p2ring_demo _ _ _ rfl
  refine ⟨ringDeviceAt_honest (1038 * 2 ^ 16) demoPoint demoRun (sidAuditAt (1038 * 2 ^ 16)) RING_D RING_Q RING_L 1
    fun _ => ⟨0, demoRun_mem_ticks, ?_⟩, ?_⟩
  · rw [hS]
    show 1060008348186 < DEMO_FULL
    norm_num [DEMO_FULL]
  · show (HONEST_S : ℝ) * ((K : ℕ) : ℝ) + ((2 ^ 30 : ℕ) : ℝ) + ((∑ _i : Unit, 437248000 * 1 : ℕ) : ℝ) ≤
      ((1060008348186 : ℕ) : ℝ)
    norm_num [HONEST_S, K]

/-- **`periodHeldDemo` is nontrivial at Demo B's store**: `periodHeldAt_nontrivial` at `N = 1038·2^16`. -/
theorem periodHeldDemo_nontrivialDemo :
    ¬ Pouw.pr (fun _ : (ringRO (1038 * 2 ^ 16) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams => True) ≤
      (Params.δ : ℝ) + Params.εMax.toReal :=
  periodHeldAt_nontrivial (1038 * 2 ^ 16)

end Pous.Period.SecurityProofs
