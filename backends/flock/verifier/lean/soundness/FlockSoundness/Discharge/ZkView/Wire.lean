import Flock.ZkProof

/-!
# A `--zk` proof file's bytes

`flock-verify verify --zk` reads each rep's proof file with `Flock.Zk.decode sh`: `bincode 1.3` of
`(Commitment, R1csProofLigerito, InnerProof)` with the `--zk` differences (`Flock/ZkProof.lean`). `encZk sh p` is the
file the prover writes for the decoded proof `p`: each `Vec` as its `u64` length then its elements, each `F128` as
`le64(lo) ‖ le64(hi)`, the vectors the verifier requires empty (`matrix_evals`, the grinding nonces but the query, claim and
consistency nonces, `sumcheck_transcript`) empty, each ring switch's grinding nonce `0`, and level 0's hiding messages
`Some`. `Shaped sh p` is what the decoder checks of the lengths: every fixed-length vector at its length for the shape, every
digest `hashLen` bytes, every opened salt `saltLen` bytes, the inner proof's salts 192 bytes and its path entries 64.
-/

namespace FlockSoundness.Discharge.ZkView

open Flock Zk

/-- A length or a `u64` field, little-endian. -/
def wLen (n : Nat) : ByteArray := le64 n.toUInt64

/-- A `Vec<T>`: its length, then its elements. -/
def wVec {α : Type} (f : α → ByteArray) (xs : Array α) : ByteArray := xs.foldl (fun b x => b ++ f x) (wLen xs.size)

/-- A `Vec<u8>`. -/
def wBytes (b : ByteArray) : ByteArray := wLen b.size ++ b

/-- An `(F128, F128)`. -/
def wPair (p : F128 × F128) : ByteArray := p.1.toBytes ++ p.2.toBytes

/-- An `F256`: `c0`, then `c1`. -/
def wF256 (x : F256) : ByteArray := x.c0.toBytes ++ x.c1.toBytes

/-- Each level's `opened_salts`, present when the scheme salts its leaves. -/
def wSalts (sh : Shape) (xs : Array ByteArray) : ByteArray :=
  match sh.saltLen with
  | none => ByteArray.empty
  | some _ => wVec id xs

/-- A level's openings: its rows, its Merkle siblings, its salts. -/
def wLevel (sh : Shape) (p : LevelProof) : ByteArray :=
  wVec (wVec F128.toBytes) p.rows ++ wVec id p.siblings ++ wSalts sh p.salts

/-- The commitment's parameters. -/
def wParams (p : PcsParams) : ByteArray :=
  le64 p.m ++ le64 p.logInvRate ++ le64 p.logBatchSize ++ le32 p.profile ++
    (match p.numLanes with
     | none => ByteArray.mk #[0]
     | some x => ByteArray.mk #[1] ++ le64 x) ++
    le32 p.merkleHash

/-- `R1csProofLigerito`'s Ligerito proof, level 0's hiding messages excluded. -/
def wLigerito (sh : Shape) (l : LigeritoProof) : ByteArray :=
  wVec id l.initialCap ++ wLevel sh l.initialProof ++ wVec (wVec id) l.recursiveCaps ++
    wVec (wLevel sh) l.recursiveProofs ++
    (wVec F128.toBytes l.finalProof.yr ++ wVec (wVec F128.toBytes) l.finalProof.rows ++ wVec id l.finalProof.siblings ++
      wSalts sh l.finalProof.salts) ++
    wLen 0 ++ wVec (fun m : F256 × F256 => wF256 m.1 ++ wF256 m.2) l.messages ++ wVec le64 l.queryNonces ++
    wVec F128.toBytes l.oodValues ++ wLen 0 ++ wVec le64 l.claimNonces ++ wVec le64 l.consistencyNonces

/-- The commitment and `R1csProofLigerito` (§8's pair, with the `--zk` differences). -/
def wProof (sh : Shape) (p : Proof) : ByteArray :=
  wVec id p.cap ++ wParams p.params ++
    (wVec F128.toBytes p.zerocheck.round1ab ++ wVec F128.toBytes p.zerocheck.round1c ++ wVec wPair p.zerocheck.rounds ++
      p.zerocheck.finalA.toBytes ++ p.zerocheck.finalB.toBytes ++ p.zerocheck.finalC.toBytes ++ wLen 0) ++
    (wVec wPair p.lincheck.rounds ++ wVec F128.toBytes p.lincheck.zPartial ++ wLen 0 ++ wLen 0) ++
    wVec (fun v => wVec F128.toBytes v ++ wLen 0) p.ringSwitches ++ wLen 0 ++
    wLigerito sh p.ligerito

/-- Level 0's hiding messages, as `Some`. -/
def wLevel0 (m : Level0Msgs) : ByteArray :=
  ByteArray.mk #[1] ++ wF256 m.e0 ++ wF256 m.e1 ++ wF256 m.tPrime ++ wVec wF256 m.ybar

/-- The inner proof (`zk_veil::InnerProof`). -/
def wInner (p : InnerProof) : ByteArray :=
  wBytes p.root ++ p.rho.toBytes ++ p.sigma.toBytes ++ wBytes p.tauCommit ++ wBytes p.tauSalt ++ p.tau.toBytes ++
    wVec F128.toBytes p.y ++ wVec wPair p.opened ++ wVec id p.salts ++ wVec (wVec id) p.paths

/-- **A `--zk` proof file**: the bytes `flock-circuit --zk` writes for the decoded proof `p` at shape `sh`. -/
def encZk (sh : Shape) (p : ZkProof) : ByteArray :=
  wProof sh p.proof ++ wLevel0 p.level0 ++ wInner p.inner

/-- A level's `opened_salts` as the decoder reads them for `q` opened rows. -/
def SaltsOk (sh : Shape) (q : Nat) (xs : Array ByteArray) : Prop :=
  match sh.saltLen with
  | none => xs = #[]
  | some 0 => xs = #[]
  | some s => xs.size = q ∧ ∀ x ∈ xs, x.size = s

/-- A level's openings at its shape: `q` rows of `w` lanes, `q·(d − cap)` siblings of `hashLen` bytes, the salts. -/
def LevelOk (sh : Shape) (l : Level) (w : Nat) (p : LevelProof) : Prop :=
  p.rows.size = l.q ∧ (∀ r ∈ p.rows, r.size = w) ∧ p.siblings.size = l.q * (l.d - l.cap) ∧
    (∀ s ∈ p.siblings, s.size = sh.hashLen) ∧ SaltsOk sh l.q p.salts

/-- A cap of level `l`: `2^cap` digests of `hashLen` bytes. -/
def CapOk (sh : Shape) (l : Level) (c : Array ByteArray) : Prop := c.size = 2 ^ l.cap ∧ ∀ s ∈ c, s.size = sh.hashLen

/-- **The lengths `Zk.decode sh` checks**, and a file short enough for its lengths to fit a `u64`. -/
def Shaped (sh : Shape) (p : ZkProof) : Prop :=
  let s := sh.sched
  let R := s.levels.size - 1
  let lvl := fun (i : Nat) => s.levels.getD i default
  let lg := p.proof.ligerito
  0 < sh.hashLen ∧ 1 ≤ R ∧ (encZk sh p).size < 2 ^ 64 ∧
    CapOk sh (lvl 0) p.proof.cap ∧
    p.proof.zerocheck.round1ab.size = 64 ∧ p.proof.zerocheck.round1c.size = 64 ∧
    p.proof.zerocheck.rounds.size = sh.m - 6 ∧
    p.proof.lincheck.rounds.size = sh.kLog - 6 ∧ p.proof.lincheck.zPartial.size = 64 ∧
    p.proof.ringSwitches.size = sh.claims ∧ (∀ v ∈ p.proof.ringSwitches, v.size = 128) ∧
    CapOk sh (lvl 0) lg.initialCap ∧ LevelOk sh (lvl 0) (2 ^ (lvl 0).lanes + EXTRA_LANES) lg.initialProof ∧
    lg.recursiveCaps.size = R ∧ (∀ i (h : i < lg.recursiveCaps.size), CapOk sh (lvl (i + 1)) lg.recursiveCaps[i]) ∧
    lg.recursiveProofs.size = R - 1 ∧ (∀ i (h : i < lg.recursiveProofs.size), LevelOk sh (lvl (i + 1)) 16 lg.recursiveProofs[i]) ∧
    lg.finalProof.yr.size = 2 ^ (s.ext + 1) ∧
    LevelOk sh (lvl R) 16 ⟨lg.finalProof.rows, lg.finalProof.siblings, lg.finalProof.salts⟩ ∧
    lg.messages.size = s.k0 + 8 + 7 * (R - 1) ∧ lg.queryNonces.size = R + 1 ∧ lg.oodValues.size = 1 + 2 * R ∧
    lg.claimNonces.size = 3 * R + 2 ∧ lg.consistencyNonces.size = R + 1 ∧
    (∀ x ∈ p.inner.salts, x.size = 192) ∧ (∀ q ∈ p.inner.paths, ∀ x ∈ q, x.size = 64)

end FlockSoundness.Discharge.ZkView
