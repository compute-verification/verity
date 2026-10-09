import Proofs.Flock.VBridge.Family.Files
import Proofs.Flock.VBridge.Algebra.Schedule
import Proofs.Flock.Soundness.Discharge.ZkView.Decode

/-!
# The inner proof file's bytes: `encodeProof`, and `TableSrc.Encodes`

`encodeProof sh p` is the file the prover writes for the proof `p` at shape `sh`: `bincode 1.3` of the commitment and
`R1csProofLigerito` (PROTOCOL.md §8), each `Vec` as its `u64` length then its elements, each `F128` as
`le64(lo) ‖ le64(hi)`, the vectors the verifier requires empty (`matrix_evals`, the grinding nonces but the query, claim
and consistency nonces, `sumcheck_transcript`) empty, each ring switch's grinding nonce `0`. It is `--zk`'s `wProof`
(ZkView/Wire.lean), the pair that file's proof starts with.

* `decodeProof_encodeProof`: `decodeProof sh (encodeProof sh p) = .ok p` for every `p` of the shape (`ProofShaped`):
  every length `decodeProof` checks (Proof.lean:81-158) at its value, a schedule of at least two levels (at one level
  `recursive_proofs`' length check, Proof.lean:130, refuses every file), digests of positive width, and a file that fits
  a `u64`.
* `TableSrc.encodes_of`: `TableSrc.Encodes` at `bytes r := encodeProof (shapeOf …) (proof r)`, from the shapes of the
  level-0 cap and the Ligerito openings the table's proof files carry (`CapShaped`, `LigShaped`), V*'s shape
  (`ShapeIs`, and its claims `2 + n`), and the file's fit.
-/

namespace FlockVBridge.Family

open Flock FlockSoundness.Refine FlockSoundness.Discharge FlockVBridge.Compiled
open Flock.RecAlgebra (decoded)

/-- **The inner proof file** of `p` at shape `sh`, as `decodeProof sh` reads it (Proof.lean:81-158). -/
def encodeProof (sh : Shape) (p : Proof) : ByteArray := ZkView.wProof sh p

/-- **The lengths `decodeProof sh` checks** (Proof.lean:94-151), a schedule of at least two levels, digests of positive
width, and a file whose length fits a `u64`. -/
def ProofShaped (sh : Shape) (p : Proof) : Prop :=
  let s := sh.sched
  let R := s.levels.size - 1
  let lvl := fun (i : Nat) => s.levels.getD i default
  let lg := p.ligerito
  0 < sh.hashLen ∧ 1 ≤ R ∧ (encodeProof sh p).size < 2 ^ 64 ∧
    ZkView.CapOk sh (lvl 0) p.cap ∧
    p.zerocheck.round1ab.size = 64 ∧ p.zerocheck.round1c.size = 64 ∧
    p.zerocheck.rounds.size = sh.m - 6 ∧
    p.lincheck.rounds.size = sh.kLog - 6 ∧ p.lincheck.zPartial.size = 64 ∧
    p.ringSwitches.size = sh.claims ∧ (∀ v ∈ p.ringSwitches, v.size = 128) ∧
    ZkView.CapOk sh (lvl 0) lg.initialCap ∧ ZkView.LevelOk sh (lvl 0) (2 ^ (lvl 0).lanes) lg.initialProof ∧
    lg.recursiveCaps.size = R ∧
    (∀ i (h : i < lg.recursiveCaps.size), ZkView.CapOk sh (lvl (i + 1)) lg.recursiveCaps[i]) ∧
    lg.recursiveProofs.size = R - 1 ∧
    (∀ i (h : i < lg.recursiveProofs.size), ZkView.LevelOk sh (lvl (i + 1)) 16 lg.recursiveProofs[i]) ∧
    lg.finalProof.yr.size = 2 ^ (s.ext + 1) ∧
    ZkView.LevelOk sh (lvl R) 16 ⟨lg.finalProof.rows, lg.finalProof.siblings, lg.finalProof.salts⟩ ∧
    lg.messages.size = s.k0 + 8 + 7 * (R - 1) ∧ lg.queryNonces.size = R + 1 ∧ lg.oodValues.size = 1 + 2 * R ∧
    lg.claimNonces.size = 3 * R + 2 ∧ lg.consistencyNonces.size = R + 1

theorem ok_of_reads {m : Reader.M Proof} {b : ByteArray} {x : Proof} (h : ZkView.Reads m b x)
    (hb : b.size < 2 ^ 64) : (do let (p, _) ← m.run ⟨b, 0⟩; return p : V Proof) = .ok x := by
  have e := h ByteArray.empty ByteArray.empty (by rw [ByteArray.empty_append, ByteArray.append_empty]; exact hb)
  rw [ByteArray.empty_append, ByteArray.append_empty] at e
  rw [show (⟨b, 0⟩ : Reader) = ⟨b, ByteArray.empty.size⟩ from rfl, e]
  rfl

/-- `matrix_evals`, written empty, read by `vecAny` (Proof.lean:111). -/
theorem reads_matrixEvals :
    ZkView.Reads (Reader.vecAny 32 (do let _ ← Reader.f128; let _ ← Reader.f128; pure ()) "matrix_evals")
      (ZkView.wLen 0) #[] :=
  ZkView.Reads.congr (ZkView.Reads.vecAny (f := fun _ => ByteArray.empty) (#[] : Array Unit) _ (by omega) (by simp)
    (by simp)) (by simp [ZkView.wVec]) rfl

/-- **`decodeProof` inverts `encodeProof`** at every proof of the shape. -/
theorem decodeProof_encodeProof (sh : Shape) (p : Proof) (hp : ProofShaped sh p) :
    decodeProof sh (encodeProof sh p) = .ok p := by
  obtain ⟨cap, ⟨m, lir, lbs, prof, nl, mh⟩, ⟨r1ab, r1c, rounds, fa, fb, fc⟩, ⟨lr, zp⟩, rs,
    ⟨icap, ⟨rows0, sib0, salt0⟩, rcaps, rps, ⟨yr, rowsR, sibR, saltR⟩, msgs, qn, ood, cbn, con⟩⟩ := p
  obtain ⟨hL, hR, hfit, hcap, h1ab, h1c, hrounds, hlr, hzp, hrs, hrs128, hicap, hl0, hrcs, hrc, hrps, hrp, hyr, hfin,
    hmsg, hqn, hood, hcbn, hcon⟩ := hp
  dsimp only at *
  unfold decodeProof
  refine ok_of_reads ?_ hfit
  simp only [encodeProof, ZkView.wProof, ZkView.wParams, ZkView.wLigerito, ZkView.wLevel, ByteArray.append_assoc]
  rw [ZkView.wVec_eq (ZkView.wVec id) rcaps, ZkView.wVec_eq (ZkView.wLevel sh) rps]
  simp only [ByteArray.append_assoc]
  refine ZkView.Reads.bind (ZkView.Reads.hashes sh hL cap _ hcap.1 hcap.2) ?_
  refine ZkView.Reads.bind (ZkView.Reads.u64 m) ?_
  refine ZkView.Reads.bind (ZkView.Reads.u64 lir) ?_
  refine ZkView.Reads.bind (ZkView.Reads.u64 lbs) ?_
  refine ZkView.Reads.bind (ZkView.Reads.u32 prof) ?_
  rcases nl with _ | nl
  refine ZkView.Reads.bind (ZkView.Reads.u8 0) ?_
  simp only [pure_bind]
  swap
  simp only [ByteArray.append_assoc]
  refine ZkView.Reads.bind (ZkView.Reads.u8 1) ?_
  simp only []
  refine ZkView.Reads.bind (ZkView.Reads.map some (ZkView.Reads.u64 nl)) ?_
  all_goals
    refine ZkView.Reads.bind (ZkView.Reads.u32 mh) ?_
    refine ZkView.Reads.bind (ZkView.Reads.f128s r1ab _ h1ab) ?_
    refine ZkView.Reads.bind (ZkView.Reads.f128s r1c _ h1c) ?_
    refine ZkView.Reads.bind (ZkView.Reads.vecExact rounds _ hrounds
      (fun x _ => by rw [ZkView.wPair, ByteArray.size_append, ZkView.toBytes_size, ZkView.toBytes_size]; omega)
      fun x _ => ZkView.Reads.pair x) ?_
    refine ZkView.Reads.bind (ZkView.Reads.f128 fa) ?_
    refine ZkView.Reads.bind (ZkView.Reads.f128 fb) ?_
    refine ZkView.Reads.bind (ZkView.Reads.f128 fc) ?_
    refine ZkView.Reads.bind (ZkView.Reads.vec0 _ _) ?_
    refine ZkView.Reads.bind (ZkView.Reads.vecExact lr _ hlr
      (fun x _ => by rw [ZkView.wPair, ByteArray.size_append, ZkView.toBytes_size, ZkView.toBytes_size]; omega)
      fun x _ => ZkView.Reads.pair x) ?_
    refine ZkView.Reads.bind (ZkView.Reads.f128s zp _ hzp) ?_
    refine ZkView.Reads.bind reads_matrixEvals ?_
    refine ZkView.Reads.bind (ZkView.Reads.vec0 _ _) ?_
    refine ZkView.Reads.bind (ZkView.Reads.vecExact rs _ hrs
      (fun v _ => by have := ZkView.wVec_size_ge F128.toBytes v; rw [ByteArray.size_append]; omega)
      fun v hv => ZkView.Reads.ringSwitch v (hrs128 v hv)) ?_
    refine ZkView.Reads.bind (ZkView.Reads.vec0 _ _) ?_
    refine ZkView.Reads.bind (ZkView.Reads.hashes sh hL icap _ hicap.1 hicap.2) ?_
    refine ZkView.Reads.bind (ZkView.Reads.rows rows0 hl0.1 hl0.2.1) ?_
    refine ZkView.Reads.bind (ZkView.Reads.hashes sh hL sib0 _ hl0.2.2.1 hl0.2.2.2.1) ?_
    refine ZkView.Reads.bind (ZkView.Reads.salts sh _ salt0 hl0.2.2.2.2) ?_
    refine ZkView.Reads.bind (ZkView.Reads.u64 _) (ZkView.Reads.of_fit fun hb => ?_)
    have hrc64 : rcaps.size < 2 ^ 64 := by
      have h1 := ZkView.fit_left hb
      have := ZkView.catL_size_ge (ZkView.wVec id) 8 rcaps.toList fun x _ => ZkView.wVec_size_ge id x
      rw [Array.length_toList] at this
      omega
    have hc : (rcaps.size != sh.sched.levels.size - 1) = false := by simp [hrcs]
    simp only [ZkView.toNat_toUInt64_of_lt hrc64, hc, Bool.false_eq_true, ite_false]
    refine ZkView.Reads.bind (x := rcaps) ?_ ?_
    · exact ZkView.Reads.congr (ZkView.Reads.forIn_range _ (ZkView.wVec id) rcaps _ hrcs #[] fun i hi acc =>
        ZkView.Reads.congr (ZkView.Reads.bind (ZkView.Reads.hashes sh hL _ _ (hrc i hi).1 (hrc i hi).2)
          (ZkView.Reads.pure _)) ByteArray.append_empty rfl) rfl Array.empty_append
    refine ZkView.Reads.bind (ZkView.Reads.u64 _) (ZkView.Reads.of_fit fun hb => ?_)
    have hrp64 : rps.size < 2 ^ 64 := by
      have := ZkView.catL_size_ge (ZkView.wLevel sh) 8 rps.toList fun x _ => by
        have := ZkView.wVec_size_ge (ZkView.wVec F128.toBytes) x.rows
        simp only [ZkView.wLevel, ByteArray.size_append]
        omega
      have h1 := ZkView.fit_left hb
      rw [Array.length_toList] at this
      omega
    have hnp : (rps.size + 1 != sh.sched.levels.size - 1) = false := by simp; omega
    simp only [ZkView.toNat_toUInt64_of_lt hrp64, hnp, Bool.false_eq_true, ite_false]
    refine ZkView.Reads.bind (x := rps) ?_ ?_
    · exact ZkView.Reads.congr (ZkView.Reads.forIn_range _ (ZkView.wLevel sh) rps _ hrps #[] fun i hi acc =>
        ZkView.Reads.congr (ZkView.Reads.bind (ZkView.Reads.rows _ (hrp i hi).1 (hrp i hi).2.1)
          (ZkView.Reads.bind (ZkView.Reads.hashes sh hL _ _ (hrp i hi).2.2.1 (hrp i hi).2.2.2.1)
            (ZkView.Reads.bind (ZkView.Reads.salts sh _ _ (hrp i hi).2.2.2.2) (ZkView.Reads.pure _))))
          (by simp only [ZkView.wLevel, ByteArray.append_assoc, ByteArray.append_empty]) rfl)
        rfl Array.empty_append
    refine ZkView.Reads.bind (ZkView.Reads.f128s yr _ hyr) ?_
    refine ZkView.Reads.bind (ZkView.Reads.rows rowsR hfin.1 hfin.2.1) ?_
    refine ZkView.Reads.bind (ZkView.Reads.hashes sh hL sibR _ hfin.2.2.1 hfin.2.2.2.1) ?_
    refine ZkView.Reads.bind (ZkView.Reads.salts sh _ saltR hfin.2.2.2.2) ?_
    refine ZkView.Reads.bind (ZkView.Reads.vec0 _ _) ?_
    refine ZkView.Reads.bind (ZkView.Reads.vecExact msgs _ hmsg
      (fun x _ => by simp only [ZkView.wF256, ByteArray.size_append, ZkView.toBytes_size]; omega)
      fun x _ => ZkView.Reads.pair256 x) ?_
    refine ZkView.Reads.bind (ZkView.Reads.u64s qn _ hqn) ?_
    refine ZkView.Reads.bind (ZkView.Reads.f128s ood _ hood) ?_
    refine ZkView.Reads.bind (ZkView.Reads.vec0 _ _) ?_
    refine ZkView.Reads.bind (ZkView.Reads.u64s cbn _ hcbn) ?_
    exact ZkView.Reads.last (ZkView.Reads.u64s con _ hcon) rfl

/-! ## The table's proof files -/

/-- **The level-0 cap has the schedule's shape**: `2^cap` digests of the scheme's width (`decodeProof`'s `cap` and
`initial_cap`, Proof.lean:94, 119). The cap V*'s layers hold (`cap`, defined by the join). -/
def CapShaped (st : Setup) (sch : Flock.Schedule) (cap : Array ByteArray) : Prop :=
  ZkView.CapOk (shapeOf st sch 0) (sch.levels.getD 0 default) cap

/-- **The Ligerito openings have the schedule's shape**: each level's openings, the recursive caps and the three nonce
vectors at the lengths `decodeProof` checks (Proof.lean:120-151). The openings V*'s layers hold (`lig`, defined by the
join). -/
def LigShaped (st : Setup) (sch : Flock.Schedule) (lig : LigeritoProof) : Prop :=
  let sh := shapeOf st sch 0
  let R := sch.levels.size - 1
  let lvl := fun (i : Nat) => sch.levels.getD i default
  ZkView.LevelOk sh (lvl 0) (2 ^ (lvl 0).lanes) lig.initialProof ∧
    lig.recursiveCaps.size = R ∧
    (∀ i (h : i < lig.recursiveCaps.size), ZkView.CapOk sh (lvl (i + 1)) lig.recursiveCaps[i]) ∧
    lig.recursiveProofs.size = R - 1 ∧
    (∀ i (h : i < lig.recursiveProofs.size), ZkView.LevelOk sh (lvl (i + 1)) 16 lig.recursiveProofs[i]) ∧
    ZkView.LevelOk sh (lvl R) 16 ⟨lig.finalProof.rows, lig.finalProof.siblings, lig.finalProof.salts⟩ ∧
    lig.queryNonces.size = R + 1 ∧ lig.claimNonces.size = 3 * R + 2 ∧ lig.consistencyNonces.size = R + 1

theorem shl_two (e : ℕ) : 2 <<< e = 2 ^ (e + 1) := by
  rw [Nat.shiftLeft_eq, Nat.pow_succ, Nat.mul_comm]

/-- **A proof file at V*'s shape has the decoder's**: `proofOf`'s messages are `decoded`'s, whose lengths are the
shape's, and its cap and Ligerito openings have the schedule's shape. -/
theorem proofOf_shaped {st : Setup} {sch : Flock.Schedule} {sh : Shape} {n : ℕ} {params : PcsParams}
    {cap : Array ByteArray} {lig : LigeritoProof} {d : RepData} (hsh : ShapeIs sh st sch) (hcl : sh.claims = 2 + n)
    (hcap : CapShaped st sch cap) (hlig : LigShaped st sch lig) (hL : 0 < st.merkle.digestLen)
    (hR : 1 ≤ sch.levels.size - 1) (hfit : (encodeProof (shapeOf st sch n) (proofOf sh params cap lig d)).size < 2 ^ 64) :
    ProofShaped (shapeOf st sch n) (proofOf sh params cap lig d) := by
  obtain ⟨hm, hk, rfl⟩ := hsh
  obtain ⟨hl0, hrcs, hrc, hrps, hrp, hfin, hqn, hcbn, hcon⟩ := hlig
  refine ⟨hL, hR, hfit, hcap, ?_, ?_, ?_, ?_, ?_, ?_, ?_, hcap, hl0, hrcs, hrc, hrps, hrp, ?_, hfin, ?_, hqn, ?_, hcbn,
    hcon⟩
  all_goals simp [proofOf, decoded, RecAlgebra.Messages.map, RecAlgebra.symbolic, shapeOf, hm, hk, hcl, shl_two]

namespace TableSrc

variable (s : TableSrc)

/-- **V*'s shape has the statement's claims**: `2` and the setup's `2 · |regions|` extra claims. `verifierRows`
accepting a rep's coins at V*'s shape gives `claims = 2 + |extras|` (`claims_of`), and the rep's extras are the setup's
(`ExtrasAre`, `PointsOK`). -/
def ClaimsAre : Prop := s.sh.claims = 2 + 2 * s.S.regions.length

theorem claims_of {r : Fin 2} {x} (hrows : RecAlgebra.verifierRows s.sh (s.d r).rounds (s.d r).extras (s.d r).comb = .ok x)
    (hxs : (s.d r).extras.size = 2 * s.S.regions.length) : s.ClaimsAre := by
  rw [ClaimsAre, RecAlgebra.verifierRows_claims hrows, hxs]

/-- **Each rep's proof file fits a `u64`** at the shape `verifyRep` decodes with. -/
def Fits : Prop :=
  ∀ r, (encodeProof (shapeOf s.st s.sch (2 * s.S.regions.length)) (s.proof r)).size < 2 ^ 64

/-- **`Encodes`** at `bytes r := encodeProof (shapeOf …) (proof r)`: V*'s shape (`ShapeIs`, `ClaimsAre`), the cap's and
the Ligerito openings' shapes, digests of positive width, a schedule of at least two levels, and files that fit. -/
theorem encodes_of (hsh : ShapeIs s.sh s.st s.sch) (hcl : s.ClaimsAre) (hcap : CapShaped s.st s.sch s.cap)
    (hlig : ∀ r, LigShaped s.st s.sch (s.lig r)) (hL : 0 < s.st.merkle.digestLen) (hR : 1 ≤ s.sch.levels.size - 1)
    (hfit : s.Fits) : s.Encodes := fun r =>
  ⟨encodeProof _ (s.proof r), decodeProof_encodeProof _ _
    (proofOf_shaped hsh hcl hcap (hlig _) hL hR (hfit r))⟩

end TableSrc

end FlockVBridge.Family
