/-
PROTOCOL.md §8: the proof container, `bincode 1.3` of `(Commitment, R1csProofLigerito)`.
-/
import Flock.Bytes
import Flock.Schedule

namespace Flock

structure PcsParams where
  m : UInt64
  logInvRate : UInt64
  logBatchSize : UInt64
  profile : UInt32
  numLanes : Option UInt64
  merkleHash : UInt32
  deriving BEq, Repr, Inhabited

structure ZerocheckProof where
  round1ab : Array F128
  round1c : Array F128
  rounds : Array (F128 × F128)
  finalA : F128
  finalB : F128
  finalC : F128
  deriving Inhabited

structure LincheckProof where
  rounds : Array (F128 × F128)
  zPartial : Array F128
  deriving Inhabited

structure LevelProof where
  rows : Array (Array F128)
  siblings : Array ByteArray
  /-- `opened_salts`: one per opened row for a salted leaf scheme -/
  salts : Array ByteArray := #[]
  deriving Inhabited

structure FinalProof where
  yr : Array F128
  rows : Array (Array F128)
  siblings : Array ByteArray
  salts : Array ByteArray := #[]
  deriving Inhabited

structure LigeritoProof where
  initialCap : Array ByteArray
  initialProof : LevelProof
  recursiveCaps : Array (Array ByteArray)
  recursiveProofs : Array LevelProof
  finalProof : FinalProof
  messages : Array (F256 × F256)
  queryNonces : Array UInt64
  oodValues : Array F128
  claimNonces : Array UInt64
  consistencyNonces : Array UInt64
  deriving Inhabited

structure Proof where
  cap : Array ByteArray
  params : PcsParams
  zerocheck : ZerocheckProof
  lincheck : LincheckProof
  ringSwitches : Array (Array F128)
  ligerito : LigeritoProof
  deriving Inhabited

/-- The shape a proof must have (§8.3). -/
structure Shape where
  m : Nat
  kLog : Nat
  claims : Nat
  sched : Schedule
  /-- the Merkle digest width (§14): every cap entry and sibling -/
  hashLen : Nat := 32
  /-- each level's openings carry `opened_salts` (PR #83's Flock patch from `2908d078`): this many bytes per opened row -/
  saltLen : Option Nat := none
  /-- hidden unit circuits (`rec_algebra.Shape.holo`): the prover absorbs `h`, the hidden part's fold, in a round of its
  own after `z_partial`, and V*'s lincheck reads the comb's operand as `comb[l] + h[l]` -/
  holo : Bool := false
  deriving Inhabited

/-! ## A plain proof's one file (§8.1)

`Wire.proofFile sh p` is the file a prover writes for `p` at `sh`: `bincode 1.3` of the pair, each `Vec` as its `u64`
length then its elements, each `F128` as `le64(lo) ‖ le64(hi)`, the vectors the verifier requires empty empty, every
grinding nonce `0` (R4), then `Wire.plainEnd`. `decodeProof` accepts a file only if it is this encoding of what it
decodes to. -/

namespace Wire

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

/-- The commitment and `R1csProofLigerito` (§8's pair, which a `--zk` file's proof also starts with). -/
def wProof (sh : Shape) (p : Proof) : ByteArray :=
  wVec id p.cap ++ wParams p.params ++
    (wVec F128.toBytes p.zerocheck.round1ab ++ wVec F128.toBytes p.zerocheck.round1c ++ wVec wPair p.zerocheck.rounds ++
      p.zerocheck.finalA.toBytes ++ p.zerocheck.finalB.toBytes ++ p.zerocheck.finalC.toBytes ++ wLen 0) ++
    (wVec wPair p.lincheck.rounds ++ wVec F128.toBytes p.lincheck.zPartial ++ wLen 0 ++ wLen 0) ++
    wVec (fun v => wVec F128.toBytes v ++ wLen 0) p.ringSwitches ++ wLen 0 ++
    wLigerito sh p.ligerito

/-- What follows the pair in every deployed prover's file: the zk patch's `zk_level0 = None` tag, one byte `0`
(`pod/60-circuit.sh` builds with `flock-zk-b684b12.patch`). -/
def plainEnd : ByteArray := ByteArray.mk #[0]

/-- **A plain proof's file**: the pair, then `plainEnd`. -/
def proofFile (sh : Shape) (p : Proof) : ByteArray := wProof sh p ++ plainEnd

end Wire

open Reader in
/-- The pair's reader (§8.1): every length at its value for the shape, then nothing or the zk patch's `None` tag
(`Reader.runPlain`). `decodeProof` pins the second. -/
def decodeRead (sh : Shape) (bytes : ByteArray) : V Proof := do
  let s := sh.sched
  let R := s.levels.size - 1
  let lvl := fun (i : Nat) => s.levels.getD i default
  let hash : M ByteArray := Reader.bytes sh.hashLen
  let salts := fun (q : Nat) => match sh.saltLen with
    | none => (pure #[] : M (Array ByteArray))
    | some 0 => vecExact 0 (Reader.bytes 0) "opened_salts"
    | some n => vecExact q (Reader.bytes n) "opened_salts"
  let prog : M Proof := do
    -- Commitment
    let cap ← vecExact (2 ^ (lvl 0).cap) hash "cap"
    let m ← u64
    let lir ← u64
    let lbs ← u64
    let profile ← u32
    let tag ← u8
    let numLanes ← match tag with
      | 0 => pure none
      | 1 => some <$> u64
      | _ => throw "num_lanes: bad Option tag"
    let mh ← u32
    let params : PcsParams := ⟨m, lir, lbs, profile, numLanes, mh⟩
    -- zerocheck
    let r1ab ← vecExact 64 f128 "round1_ab"
    let r1c ← vecExact 64 f128 "round1_c"
    let rounds ← vecExact (sh.m - 6) (do let a ← f128; let b ← f128; pure (a, b)) "multilinear_rounds"
    let fa ← f128
    let fb ← f128
    let fc ← f128
    let _ ← vecExact 0 u64 "zerocheck grinding_nonces"
    -- lincheck
    let lr ← vecExact (sh.kLog - 6) (do let a ← f128; let b ← f128; pure (a, b)) "lincheck rounds"
    let zp ← vecExact 64 f128 "z_partial"
    let _ ← vecExact 0 (do let _ ← f128; let _ ← f128; pure ()) "matrix_evals"
    let _ ← vecExact 0 u64 "lincheck grinding_nonces"
    -- opening
    let rs ← vecExact sh.claims (do
        let v ← vecExact 128 f128 "s_hat_v"
        let nonce ← u64
        if nonce != 0 then throw "ring switch grinding_nonce ≠ 0"
        pure v) "ring_switches"
    let _ ← vecExact 0 u64 "batching_nonces"
    -- Ligerito
    let icap ← vecExact (2 ^ (lvl 0).cap) hash "initial_cap"
    let l0 := lvl 0
    let iproof : LevelProof := ⟨← vecExact l0.q (vecExact (2 ^ l0.lanes) f128 "row") "opened_rows",
      ← vecExact (l0.q * (l0.d - l0.cap)) hash "merkle_proof", ← salts l0.q⟩
    let mut rcaps := #[]
    let n ← u64
    if n.toNat != R then throw s!"recursive_caps: length {n}, want {R}"
    for i in [0:R] do
      rcaps := rcaps.push (← vecExact (2 ^ (lvl (i + 1)).cap) hash "recursive cap")
    let np ← u64
    if np.toNat + 1 != R then throw s!"recursive_proofs: length {np}, want {R - 1}"
    let mut rproofs := #[]
    for i in [0:R - 1] do
      let li := lvl (i + 1)
      rproofs := rproofs.push ⟨← vecExact li.q (vecExact 16 f128 "row") "opened_rows",
        ← vecExact (li.q * (li.d - li.cap)) hash "merkle_proof", ← salts li.q⟩
    let lR := lvl R
    let fproof : FinalProof := ⟨← vecExact (2 ^ (s.ext + 1)) f128 "yr",
      ← vecExact lR.q (vecExact 16 f128 "row") "opened_rows", ← vecExact (lR.q * (lR.d - lR.cap)) hash "merkle_proof", ← salts lR.q⟩
    let _ ← vecExact 0 (do let _ ← f128; let _ ← f128; pure ()) "sumcheck_transcript"
    let msgs ← vecExact (s.k0 + 8 + 7 * (R - 1)) (do let a ← f256; let b ← f256; pure (a, b)) "sumcheck_transcript_f256"
    let qn ← vecExact (R + 1) (nonce0 "grinding_nonce") "grinding_nonces"
    let ood ← vecExact (1 + 2 * R) f128 "ood_values"
    let _ ← vecExact 0 u64 "fold_grinding_nonces"
    let cbn ← vecExact (3 * R + 2) (nonce0 "claim_batch_grinding_nonce") "claim_batch_grinding_nonces"
    let con ← vecExact (R + 1) (nonce0 "consistency_batch_grinding_nonce") "consistency_batch_grinding_nonces"
    return { cap, params, zerocheck := ⟨r1ab, r1c, rounds, fa, fb, fc⟩, lincheck := ⟨lr, zp⟩, ringSwitches := rs,
             ligerito := ⟨icap, iproof, rcaps, rproofs, fproof, msgs, qn, ood, cbn, con⟩ }
  runPlain prog bytes "proof"

/-- Why a file that reads as `p` is not `p`'s one file `file`. -/
def fileMismatch (file bytes : ByteArray) : String :=
  if bytes.size + 1 == file.size then
    "zk_level0: the file ends without the `None` tag every deployed prover writes after a plain proof (upstream's form: --upstream-plain)"
  else "proof: the file is not the one encoding of the proof it decodes to"

/-- **The plain proof file** (§8.1): what `decodeRead` reads, if the file is its one encoding at the shape
(`Wire.proofFile`), so a file's digest is a function of what it decodes to (`decodeProof_canonical`). -/
def decodeProof (sh : Shape) (bytes : ByteArray) : V Proof :=
  match decodeRead sh bytes with
  | .error e => .error e
  | .ok p => if Wire.proofFile sh p = bytes then .ok p else .error (fileMismatch (Wire.proofFile sh p) bytes)

theorem decodeProof_read {sh : Shape} {bytes : ByteArray} {p : Proof} (h : decodeProof sh bytes = .ok p) :
    decodeRead sh bytes = .ok p ∧ Wire.proofFile sh p = bytes := by
  unfold decodeProof at h
  split at h
  · cases h
  · rename_i q hq
    split at h
    · cases h
      exact ⟨hq, by assumption⟩
    · cases h

theorem decodeProof_of_read {sh : Shape} {p : Proof} (h : decodeRead sh (Wire.proofFile sh p) = .ok p) :
    decodeProof sh (Wire.proofFile sh p) = .ok p := by
  unfold decodeProof
  rw [h]
  simp

/-- **Canonical proof files**: two files `decodeProof` accepts at one shape that decode alike are one file. -/
theorem decodeProof_canonical {sh : Shape} {b b' : ByteArray} {p : Proof} (h : decodeProof sh b = .ok p)
    (h' : decodeProof sh b' = .ok p) : b = b' :=
  (decodeProof_read h).2.symm.trans (decodeProof_read h').2

/-- **What the verifier decodes from a plain proof file** under its statement's `Tags.upstreamPlain` (`flock-verify verify
--upstream-plain`, §8.1; `verifyFiles`, whose S17 hashes the file itself): upstream's form ends after the pair, so the
file then the zk patch's `None` tag; by default the file itself, the deployed form (`verify`). Each form is refused where the other is pinned: a file without the tag by default
(`fileMismatch`), and one with it under the flag, which then ends in a second `0` (`Reader.runPlain`). -/
def plainBytes (upstream : Bool) (file : ByteArray) : ByteArray := if upstream then file ++ Wire.plainEnd else file

theorem plainBytes_inj {u : Bool} {b b' : ByteArray} (h : plainBytes u b = plainBytes u b') : b = b' := by
  cases u
  · exact h
  · apply ByteArray.ext
    have := congrArg ByteArray.data h
    simp only [plainBytes, ite_true, ByteArray.data_append] at this
    exact (Array.append_left_inj _).1 this

/-- **Canonical proof files at the pinned form**: under either flag, two files `verify` decodes alike at one shape are
one file. -/
theorem plainBytes_canonical {u : Bool} {sh : Shape} {b b' : ByteArray} {p : Proof}
    (h : decodeProof sh (plainBytes u b) = .ok p) (h' : decodeProof sh (plainBytes u b') = .ok p) : b = b' :=
  plainBytes_inj (decodeProof_canonical h h')

end Flock
