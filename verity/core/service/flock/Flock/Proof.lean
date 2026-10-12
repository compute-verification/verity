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

open Reader in
def decodeProof (sh : Shape) (bytes : ByteArray) : V Proof := do
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
    let _ ← vecAny 32 (do let _ ← f128; let _ ← f128; pure ()) "matrix_evals"
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
    let qn ← vecExact (R + 1) u64 "grinding_nonces"
    let ood ← vecExact (1 + 2 * R) f128 "ood_values"
    let _ ← vecExact 0 u64 "fold_grinding_nonces"
    let cbn ← vecExact (3 * R + 2) u64 "claim_batch_grinding_nonces"
    let con ← vecExact (R + 1) u64 "consistency_batch_grinding_nonces"
    -- bytes after the pair are ignored (§8.1)
    return { cap, params, zerocheck := ⟨r1ab, r1c, rounds, fa, fb, fc⟩, lincheck := ⟨lr, zp⟩, ringSwitches := rs,
             ligerito := ⟨icap, iproof, rcaps, rproofs, fproof, msgs, qn, ood, cbn, con⟩ }
  let (p, _) ← prog.run ⟨bytes, 0⟩
  return p

end Flock
