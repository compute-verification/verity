/-
A `--zk` proof file (live PROTOCOL.md, `flock-circuit --zk`): `bincode 1.3` of `(Commitment, R1csProofLigerito, InnerProof)`.
The pair is §8's with three differences: `matrix_evals` is empty, every level-0 row carries four more lanes (the extra lanes,
a pair per rep), and the Ligerito proof ends with level 0's hiding messages (`zk_level0`, an `Option`). The inner proof
follows the pair. Bytes after it are ignored, as after §8's pair.
-/
import Flock.Proof

namespace Flock.Zk

/-- Level 0's hiding messages (upstream's `Level0ZkProof`): the pair's sums `e`, the blinded claim `T'` and the padding's
combination `ybar`. -/
structure Level0Msgs where
  e0 : F256
  e1 : F256
  tPrime : F256
  ybar : Array F256
  deriving Inhabited

/-- The inner proof (live PROTOCOL.md §5, `zk_veil::InnerProof`): the pads' root, the triple's openings `ρ, σ`, `τ`'s
commitment and its opening, `Y = h + β μ`, and the opened columns with their salts and paths. -/
structure InnerProof where
  root : ByteArray
  rho : F128
  sigma : F128
  tauCommit : ByteArray
  tauSalt : ByteArray
  tau : F128
  y : Array F128
  opened : Array (F128 × F128)
  salts : Array ByteArray
  paths : Array (Array ByteArray)
  deriving Inhabited

/-- Extra lanes in a level-0 row: two per rep, for two reps. -/
def EXTRA_LANES : Nat := 4

/-- A `--zk` proof, decoded. -/
structure ZkProof where
  proof : Proof
  level0 : Level0Msgs
  inner : InnerProof
  deriving Inhabited

open Reader

/-- A `Vec<u8>`. -/
def byteVec : M ByteArray := do
  let n ← u64
  Reader.bytes n.toNat

def innerReader : M InnerProof := do
  let root ← byteVec
  let rho ← f128
  let sigma ← f128
  let tauCommit ← byteVec
  let tauSalt ← byteVec
  let tau ← f128
  let y ← vecAny 16 f128 "y"
  let opened ← vecAny 32 (do let a ← f128; let b ← f128; pure (a, b)) "opened"
  let salts ← vecAny 192 (Reader.bytes 192) "salts"
  let paths ← vecAny 8 (vecAny 64 (Reader.bytes 64) "path") "paths"
  return { root, rho, sigma, tauCommit, tauSalt, tau, y, opened, salts, paths }

/-- §8.1's reader for a `--zk` file of shape `sh` (whose level 0 is the `--zk` schedule's), then the inner proof. -/
def reader (sh : Shape) : M ZkProof := do
  let s := sh.sched
  let R := s.levels.size - 1
  let lvl := fun (i : Nat) => s.levels.getD i default
  let hash : M ByteArray := Reader.bytes sh.hashLen
  let salts := fun (q : Nat) => match sh.saltLen with
    | none => (pure #[] : M (Array ByteArray))
    | some 0 => vecExact 0 (Reader.bytes 0) "opened_salts"
    | some n => vecExact q (Reader.bytes n) "opened_salts"
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
  let iproof : LevelProof := ⟨← vecExact l0.q (vecExact (2 ^ l0.lanes + EXTRA_LANES) f128 "row") "opened_rows",
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
  let level0 ← match ← u8 with
    | 1 => pure (⟨← f256, ← f256, ← f256, ← vecAny 32 f256 "ybar"⟩ : Level0Msgs)
    | 0 => throw "zk: level 0's hiding messages are missing"
    | _ => throw "zk_level0: bad Option tag"
  let inner ← innerReader
  return { proof := { cap, params, zerocheck := ⟨r1ab, r1c, rounds, fa, fb, fc⟩, lincheck := ⟨lr, zp⟩, ringSwitches := rs,
                      ligerito := ⟨icap, iproof, rcaps, rproofs, fproof, msgs, qn, ood, cbn, con⟩ },
           level0, inner }

def decode (sh : Shape) (bytes : ByteArray) : V ZkProof := do
  let (p, _) ← (reader sh).run ⟨bytes, 0⟩
  return p

end Flock.Zk
