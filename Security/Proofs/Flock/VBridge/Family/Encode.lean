import Proofs.Flock.VBridge.Family.Files
import Proofs.Flock.VBridge.Family.Rep
import Proofs.Flock.VBridge.Algebra.Schedule
import Proofs.Flock.Soundness.Discharge.ZkView.Decode

/-!
# The inner proof file's bytes: `encodeProof`, and `TableSrc.Encodes`

`encodeProof sh p` is the file the prover writes for the proof `p` at shape `sh`: `bincode 1.3` of the commitment and
`R1csProofLigerito` (PROTOCOL.md §8), each `Vec` as its `u64` length then its elements, each `F128` as
`le64(lo) ‖ le64(hi)`, the vectors the verifier requires empty (`matrix_evals`, the grinding nonces but the query, claim
and consistency nonces, `sumcheck_transcript`) empty, each ring switch's grinding nonce `0`, then the zk patch's
`zk_level0 = None` tag, as every deployed prover writes it: the verifier's `Flock.Wire.proofFile`, the one file
`decodeProof` accepts for `p`. Its pair is `--zk`'s `wProof` (ZkView/Wire.lean), the pair that file's proof starts
with.

* `decodeProof_encodeProof`: `decodeProof sh (encodeProof sh p) = .ok p` for every `p` of the shape (`ProofShaped`):
  every length `decodeProof` checks (`decodeRead`) at its value, a schedule of at least two levels (at one level
  `recursive_proofs`' length check refuses every file), digests of positive width, and a file that fits a `u64`.
* `encodeProof_size_le`: the file is at most `fileBound sh` bytes (the pair's bound and the tag's byte); at hm96-sha512 and a `fast100` schedule that is at
  most `2^32` bytes and `2064` per ring switch (`fast100_file`), and at the inner statement's unsalted SHA-512, whose
  salts are no larger, too (`shapeOf_plain`, `schedPart_plain`).
* `TableSrc.encodes_of`: `TableSrc.Encodes` at `bytes r := encodeProof (shapeOf …) (proof r)`, from the setup
  (`SetupRealizes`), V*'s shape (`ShapeIs`, and its claims `2 + n`: `ClaimsAre`, from `HFit` and `ExtrasAre` by
  `claimsAre_of`), the shapes of the level-0 cap and the Ligerito openings the proof files carry (`CapShaped`,
  `LigShaped`), and fewer than `2^48` regions (`RegionsFit`). `hEnc_of` is it at the family; `hdec_filesAt` is
  `hdec` at the table files.
-/

namespace FlockVBridge.Family

open Flock FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Recursion FlockVBridge.Compiled
open Flock.RecAlgebra (decoded)

/-- **The inner proof file** of `p` at shape `sh`, the one `decodeProof sh` accepts for it (Proof.lean). -/
def encodeProof (sh : Shape) (p : Proof) : ByteArray := Flock.Wire.proofFile sh p

/-- **The lengths `decodeProof sh` checks** (Proof.lean:94-151). -/
def Lengths (sh : Shape) (p : Proof) : Prop :=
  let s := sh.sched
  let R := s.levels.size - 1
  let lvl := fun (i : Nat) => s.levels.getD i default
  let lg := p.ligerito
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
    lg.claimNonces.size = 3 * R + 2 ∧ lg.consistencyNonces.size = R + 1 ∧ ZkView.NoncesZero lg

/-- **A proof of the shape**: the lengths `decodeProof sh` checks, a schedule of at least two levels, digests of positive
width, and a file whose length fits a `u64`. -/
def ProofShaped (sh : Shape) (p : Proof) : Prop :=
  0 < sh.hashLen ∧ 1 ≤ sh.sched.levels.size - 1 ∧ (encodeProof sh p).size < 2 ^ 64 ∧ Lengths sh p

theorem ok_of_reads {m : Reader.M Proof} {b : ByteArray} {x : Proof} {what : String}
    (h : ZkView.Reads m b x) (hb : (b ++ Flock.Wire.plainEnd).size < 2 ^ 64) :
    Reader.runPlain m (b ++ Flock.Wire.plainEnd) what = .ok x := by
  have e := h ByteArray.empty Flock.Wire.plainEnd (by rw [ByteArray.empty_append]; exact hb)
  rw [ByteArray.empty_append] at e
  exact Reader.runPlain_of_run_tag e (Nat.zero_add b.size)

/-- **`decodeProof` inverts `encodeProof`** at every proof of the shape. -/
theorem decodeProof_encodeProof (sh : Shape) (p : Proof) (hp : ProofShaped sh p) :
    decodeProof sh (encodeProof sh p) = .ok p := by
  obtain ⟨cap, ⟨m, lir, lbs, prof, nl, mh⟩, ⟨r1ab, r1c, rounds, fa, fb, fc⟩, ⟨lr, zp⟩, rs,
    ⟨icap, ⟨rows0, sib0, salt0⟩, rcaps, rps, ⟨yr, rowsR, sibR, saltR⟩, msgs, qn, ood, cbn, con⟩⟩ := p
  obtain ⟨hL, hR, hfit, hcap, h1ab, h1c, hrounds, hlr, hzp, hrs, hrs128, hicap, hl0, hrcs, hrc, hrps, hrp, hyr, hfin,
    hmsg, hqn, hood, hcbn, hcon, hzq, hzc, hzk⟩ := hp
  dsimp only at *
  rw [encodeProof] at hfit ⊢
  refine Flock.decodeProof_of_read ?_
  unfold decodeRead
  rw [Flock.Wire.proofFile] at hfit ⊢
  refine ok_of_reads ?_ hfit
  simp only [ZkView.wProof, ZkView.wParams, ZkView.wLigerito, ZkView.wLevel, ByteArray.append_assoc]
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
    refine ZkView.Reads.bind (ZkView.Reads.vec0 _ _) ?_
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
    refine ZkView.Reads.bind (ZkView.Reads.zeros qn _ _ hqn hzq) ?_
    refine ZkView.Reads.bind (ZkView.Reads.f128s ood _ hood) ?_
    refine ZkView.Reads.bind (ZkView.Reads.vec0 _ _) ?_
    refine ZkView.Reads.bind (ZkView.Reads.zeros cbn _ _ hcbn hzc) ?_
    exact ZkView.Reads.last (ZkView.Reads.zeros con _ _ hcon hzk) rfl

/-! ## The table's proof files -/

/-- **The level-0 cap has the schedule's shape**: `2^cap` digests of the scheme's width (`decodeProof`'s `cap` and
`initial_cap`, Proof.lean:94, 119). The cap V*'s layers hold (`cap`, defined by the join). -/
def CapShaped (st : Setup) (sch : Flock.Schedule) (cap : Array ByteArray) : Prop :=
  ZkView.CapOk (shapeOf st sch 0) (sch.levels.getD 0 default) cap

/-- **The Ligerito openings have the schedule's shape**: each level's openings, the recursive caps and the three nonce
vectors at the lengths `decodeProof` checks (Proof.lean:120-151), every nonce `0`. The openings V*'s layers hold (`lig`,
defined by the join). -/
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
    lig.queryNonces.size = R + 1 ∧ lig.claimNonces.size = 3 * R + 2 ∧ lig.consistencyNonces.size = R + 1 ∧
    ZkView.NoncesZero lig

theorem shl_two (e : ℕ) : 2 <<< e = 2 ^ (e + 1) := by
  rw [Nat.shiftLeft_eq, Nat.pow_succ, Nat.mul_comm]

/-- **A proof file at V*'s shape has the decoder's lengths**: `proofOf`'s messages are `decoded`'s, whose lengths are
the shape's, and its cap and Ligerito openings have the schedule's shape. -/
theorem proofOf_lengths {st : Setup} {sch : Flock.Schedule} {sh : Shape} {n : ℕ} {params : PcsParams}
    {cap : Array ByteArray} {lig : LigeritoProof} {d : RepData} (hsh : ShapeIs sh st sch) (hcl : sh.claims = 2 + n)
    (hcap : CapShaped st sch cap) (hlig : LigShaped st sch lig) :
    Lengths (shapeOf st sch n) (proofOf sh params cap lig d) := by
  obtain ⟨hm, hk, rfl⟩ := hsh
  obtain ⟨hl0, hrcs, hrc, hrps, hrp, hfin, hqn, hcbn, hcon, hz⟩ := hlig
  refine ⟨hcap, ?_, ?_, ?_, ?_, ?_, ?_, ?_, hcap, hl0, hrcs, hrc, hrps, hrp, ?_, hfin, ?_, hqn, ?_, hcbn, hcon, hz⟩
  all_goals simp [proofOf, decoded, RecAlgebra.Messages.map, RecAlgebra.symbolic, shapeOf, hm, hk, hcl, shl_two]

/-! ## The file's length -/

theorem catL_size {α : Type} (f : α → ByteArray) (l : List α) :
    (ZkView.catL f l).size = (l.map fun x => (f x).size).sum := by
  induction l with
  | nil => rfl
  | cons x l ih => rw [ZkView.catL, ByteArray.size_append, ih, List.map_cons, List.sum_cons]

theorem wVec_size {α : Type} (f : α → ByteArray) (xs : Array α) :
    (ZkView.wVec f xs).size = 8 + (xs.toList.map fun x => (f x).size).sum := by
  rw [ZkView.wVec_eq, ByteArray.size_append, ZkView.wLen, ZkView.le64_size, catL_size]

theorem wVec_size_idx {α : Type} (f : α → ByteArray) (xs : Array α) (G : ℕ → ℕ)
    (h : ∀ i (hi : i < xs.size), (f xs[i]).size = G i) :
    (ZkView.wVec f xs).size = 8 + ((List.range xs.size).map G).sum := by
  rw [wVec_size]
  congr 2
  apply List.ext_getElem (by simp)
  intro i h1 h2
  simp only [List.getElem_map, Array.getElem_toList, List.getElem_range]
  exact h i (by simpa using h1)

theorem wVec_size_const {α : Type} (f : α → ByteArray) (xs : Array α) (k : ℕ) (h : ∀ x ∈ xs, (f x).size = k) :
    (ZkView.wVec f xs).size = 8 + xs.size * k := by
  rw [wVec_size_idx f xs (fun _ => k) fun i hi => h _ (Array.getElem_mem hi)]
  simp

/-- A level's `opened_salts` length for `q` opened rows. -/
def saltsSize (sl : Option ℕ) (q : ℕ) : ℕ :=
  match sl with
  | none => 0
  | some s => 8 + q * s

theorem wSalts_size {sh : Shape} {q : ℕ} {xs : Array ByteArray} (h : ZkView.SaltsOk sh q xs) :
    (ZkView.wSalts sh xs).size = saltsSize sh.saltLen q := by
  unfold ZkView.SaltsOk at h
  unfold ZkView.wSalts saltsSize
  rcases hs : sh.saltLen with _ | _ | n
  · rfl
  · simp only [hs] at h ⊢
    subst h
    rw [wVec_size_const id #[] 0 (by simp)]
    simp
  · simp only [hs] at h ⊢
    rw [wVec_size_const id xs (n + 1) h.2, h.1]

/-- A level's openings' length: `q` rows of `w` elements, `q·(d − cap)` digests of `H` bytes, the salts. -/
def levelSize (H : ℕ) (sl : Option ℕ) (l : Level) (w : ℕ) : ℕ :=
  8 + l.q * (8 + w * 16) + (8 + l.q * (l.d - l.cap) * H) + saltsSize sl l.q

theorem rowsSize {rows : Array (Array F128)} {w : ℕ} (h : ∀ r ∈ rows, r.size = w) :
    (ZkView.wVec (ZkView.wVec F128.toBytes) rows).size = 8 + rows.size * (8 + w * 16) :=
  wVec_size_const _ _ _ fun r hr => by
    rw [wVec_size_const F128.toBytes r 16 fun x _ => ZkView.toBytes_size x, h r hr]

theorem wLevel_size {sh : Shape} {l : Level} {w : ℕ} {p : LevelProof} (h : ZkView.LevelOk sh l w p) :
    (ZkView.wLevel sh p).size = levelSize sh.hashLen sh.saltLen l w := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  rw [ZkView.wLevel, ByteArray.size_append, ByteArray.size_append, rowsSize h2, wVec_size_const id _ sh.hashLen h4,
    wSalts_size h5, h1, h3, levelSize]

theorem wParams_size_le (p : PcsParams) : (ZkView.wParams p).size ≤ 41 := by
  obtain ⟨a, b, c, d, e, f⟩ := p
  have h0 : (ByteArray.mk #[0]).size = 1 := rfl
  have h1 : (ByteArray.mk #[1]).size = 1 := rfl
  rcases e with _ | e <;>
    simp [ZkView.wParams, ByteArray.size_append, ZkView.le64_size, ZkView.le32_size, h0, h1]

/-- The part of a file's length its schedule fixes, at digest width `H` and salts `sl`: the commitment (its parameters
at most 41 bytes), the fixed vectors and empty ones, each vector's length, and Ligerito's openings and messages. -/
def schedPart (s : Flock.Schedule) (H : ℕ) (sl : Option ℕ) : ℕ :=
  let R := s.levels.size - 1
  let lvl := fun (i : Nat) => s.levels.getD i default
  (8 + 2 ^ (lvl 0).cap * H) + 41 + (8 + 64 * 16) + (8 + 64 * 16) + 8 + 48 + 8 + 8 + (8 + 64 * 16) + 8 + 8 + 8 + 8 +
    (8 + 2 ^ (lvl 0).cap * H) + levelSize H sl (lvl 0) (2 ^ (lvl 0).lanes) +
    (8 + ((List.range R).map fun i => 8 + 2 ^ (lvl (i + 1)).cap * H).sum) +
    (8 + ((List.range (R - 1)).map fun i => levelSize H sl (lvl (i + 1)) 16).sum) +
    (8 + 2 ^ (s.ext + 1) * 16) + levelSize H sl (lvl R) 16 + 8 + (8 + (s.k0 + 8 + 7 * (R - 1)) * 64) +
    (8 + (R + 1) * 8) + (8 + (1 + 2 * R) * 16) + 8 + (8 + (3 * R + 2) * 8) + (8 + (R + 1) * 8)

/-- **A bound on a file's length at shape `sh`**: the schedule's part, the zerocheck's and lincheck's rounds,
`2064` bytes per ring switch, and the `zk_level0` tag's byte. -/
def fileBound (sh : Shape) : ℕ :=
  schedPart sh.sched sh.hashLen sh.saltLen + 32 * (sh.m - 6) + 32 * (sh.kLog - 6) + 2064 * sh.claims + 1

theorem encodeProof_size_le (sh : Shape) (p : Proof) (hp : Lengths sh p) : (encodeProof sh p).size ≤ fileBound sh := by
  obtain ⟨cap, params, ⟨r1ab, r1c, rounds, fa, fb, fc⟩, ⟨lr, zp⟩, rs,
    ⟨icap, l0, rcaps, rps, ⟨yr, rowsR, sibR, saltR⟩, msgs, qn, ood, cbn, con⟩⟩ := p
  obtain ⟨hcap, h1ab, h1c, hrounds, hlr, hzp, hrs, hrs128, hicap, hl0, hrcs, hrc, hrps, hrp, hyr, hfin,
    hmsg, hqn, hood, hcbn, hcon, -⟩ := hp
  dsimp only at *
  have ht : ∀ a : F128, a.toBytes.size = 16 := ZkView.toBytes_size
  have hw0 : (ZkView.wLen 0).size = 8 := ZkView.le64_size _
  have hpair : ∀ q : F128 × F128, (ZkView.wPair q).size = 32 := fun q => by
    rw [ZkView.wPair, ByteArray.size_append, ht, ht]
  have e1 := wVec_size_const id cap _ hcap.2
  have e2 := wVec_size_const F128.toBytes r1ab 16 fun x _ => ht x
  have e3 := wVec_size_const F128.toBytes r1c 16 fun x _ => ht x
  have e4 := wVec_size_const ZkView.wPair rounds 32 fun x _ => hpair x
  have e5 := wVec_size_const ZkView.wPair lr 32 fun x _ => hpair x
  have e6 := wVec_size_const F128.toBytes zp 16 fun x _ => ht x
  have e7 := wVec_size_const (fun v => ZkView.wVec F128.toBytes v ++ ZkView.wLen 0) rs 2064 fun v hv => by
    rw [ByteArray.size_append, wVec_size_const F128.toBytes v 16 (fun x _ => ht x), hrs128 v hv, hw0]
  have e8 := wVec_size_const id icap _ hicap.2
  have e9 := wLevel_size hl0
  have e10 := wVec_size_idx (ZkView.wVec id) rcaps
    (fun i => 8 + 2 ^ (sh.sched.levels.getD (i + 1) default).cap * sh.hashLen) fun i hi => by
      rw [wVec_size_const id _ _ (hrc i hi).2, (hrc i hi).1]
  have e11 := wVec_size_idx (ZkView.wLevel sh) rps
    (fun i => levelSize sh.hashLen sh.saltLen (sh.sched.levels.getD (i + 1) default) 16) fun i hi =>
      wLevel_size (hrp i hi)
  have e12 := wVec_size_const F128.toBytes yr 16 fun x _ => ht x
  have e13 := wLevel_size hfin
  rw [ZkView.wLevel, ByteArray.size_append, ByteArray.size_append] at e13
  have e14 := wVec_size_const (fun m : F256 × F256 => ZkView.wF256 m.1 ++ ZkView.wF256 m.2) msgs 64 fun x _ => by
    simp only [ZkView.wF256, ByteArray.size_append, ht]
  have e15 := wVec_size_const le64 qn 8 fun x _ => ZkView.le64_size x
  have e16 := wVec_size_const F128.toBytes ood 16 fun x _ => ht x
  have e17 := wVec_size_const le64 cbn 8 fun x _ => ZkView.le64_size x
  have e18 := wVec_size_const le64 con 8 fun x _ => ZkView.le64_size x
  have hpar := wParams_size_le params
  rw [hcap.1] at e1
  rw [h1ab] at e2
  rw [h1c] at e3
  rw [hrounds] at e4
  rw [hlr] at e5
  rw [hzp] at e6
  rw [hrs] at e7
  rw [hicap.1] at e8
  rw [hrcs] at e10
  rw [hrps] at e11
  rw [hyr] at e12
  rw [hmsg] at e14
  rw [hqn] at e15
  rw [hood] at e16
  rw [hcbn] at e17
  rw [hcon] at e18
  have htag : Flock.Wire.plainEnd.size = 1 := rfl
  simp only [encodeProof, Flock.Wire.proofFile, ZkView.wProof, ZkView.wLigerito, ByteArray.size_append, ht, hw0] at *
  simp only [fileBound, schedPart]
  omega

/-! ## At the verifier's schedules -/

/-- Every executable `fast100` schedule has at least two levels and fixes at most `2^32` bytes of a file at
hm96-sha512's digests and salts, and is for an `m` of at most 35. -/
theorem fast100_file (m : ℕ) (s : Flock.Schedule) (hs : Flock.fast100 m = some s) :
    2 ≤ s.levels.size ∧ schedPart s 64 (some 192) ≤ 2 ^ 32 ∧ m ≤ 35 := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ Flock.fast100 m.val,
        decide (2 ≤ s.levels.size ∧ schedPart s 64 (some 192) ≤ 2 ^ 32 ∧ m.val ≤ 35) = true := by
      decide +kernel
    simpa using key ⟨m, hm⟩ s hs
  · rw [exec_fast100_none m (by omega)] at hs; cases hs

theorem sum_map_le {l : List ℕ} {f g : ℕ → ℕ} (h : ∀ i, f i ≤ g i) : (l.map f).sum ≤ (l.map g).sum := by
  induction l with
  | nil => simp
  | cons a l ih => simp only [List.map_cons, List.sum_cons]; exact Nat.add_le_add (h a) ih

/-- A file's schedule part grows with its salts. -/
theorem schedPart_le {s : Flock.Schedule} {H : ℕ} {sl sl' : Option ℕ} (h : ∀ q, saltsSize sl q ≤ saltsSize sl' q) :
    schedPart s H sl ≤ schedPart s H sl' := by
  have hl : ∀ l w, levelSize H sl l w ≤ levelSize H sl' l w := fun l w => by
    unfold levelSize; have := h l.q; omega
  have hs := sum_map_le (l := List.range (s.levels.size - 1 - 1))
    (f := fun i => levelSize H sl (s.levels.getD (i + 1) default) 16)
    (g := fun i => levelSize H sl' (s.levels.getD (i + 1) default) 16) fun i => hl _ _
  have h0 := hl (s.levels.getD 0 default) (2 ^ (s.levels.getD 0 default).lanes)
  have hR := hl (s.levels.getD (s.levels.size - 1) default) 16
  simp only [schedPart]
  omega

/-- **The inner statement's shape** at `flock-leaf/sha512-unsalted`: 64-byte digests and no salts (`none`, or `some 0`
when the setup opens salts). -/
theorem shapeOf_plain {st : Setup} {sch : Flock.Schedule} {n : ℕ}
    (h : st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted") :
    (shapeOf st sch n).hashLen = 64 ∧ ((shapeOf st sch n).saltLen = none ∨ (shapeOf st sch n).saltLen = some 0) := by
  refine ⟨by simp [shapeOf, h, MerkleScheme.sha512], ?_⟩
  cases ho : st.openedSalts <;> simp [shapeOf, h, MerkleScheme.sha512, ho]

theorem schedPart_plain {s : Flock.Schedule} {sl : Option ℕ} (hsl : sl = none ∨ sl = some 0) :
    schedPart s 64 sl ≤ schedPart s 64 (some 192) :=
  schedPart_le fun q => by rcases hsl with rfl | rfl <;> simp [saltsSize]

namespace TableSrc

variable (s : TableSrc)

/-- **V*'s shape has the statement's claims**: `2` and the setup's `2 · |regions|` extra claims. `verifierRows`
accepting a rep's coins at V*'s shape gives `claims = 2 + |extras|` (`claims_of`), and the rep's extras are the setup's
(`ExtrasAre`, `PointsOK`). -/
def ClaimsAre : Prop := s.sh.claims = 2 + 2 * s.S.regions.length

theorem claims_of {r : Fin 2} {x} (hrows : RecAlgebra.verifierRows s.sh (s.d r).rounds (s.d r).extras (s.d r).comb = .ok x)
    (hxs : (s.d r).extras.size = 2 * s.S.regions.length) : s.ClaimsAre := by
  rw [ClaimsAre, RecAlgebra.verifierRows_claims hrows, hxs]

/-- **The statement has fewer than `2^48` regions**, so that each rep's proof file, with its `2 + 2·|regions|` ring
switches of 2064 bytes each, fits a `u64` (`decodeProof` reads every length as a `u64`, Proof.lean:94-151). The
verifier's setup refuses more than 1,024 (`checkInRange`, Statement.lean:114), so it holds at a statement with a setup's
regions (`regionsFit_of_setupH`, Lig.lean). -/
def RegionsFit : Prop := s.S.regions.length < 2 ^ 48

/-- **`Encodes`** at `bytes r := encodeProof (shapeOf …) (proof r)`: the setup's (`SetupRealizes`: plain SHA-512 and a
`fast100` schedule), V*'s shape (`ShapeIs`, `ClaimsAre`), the cap's and the Ligerito openings' shapes, and the
statement's region count. -/
theorem encodes_of {ptLocal mPts : ℕ} (hset : SetupRealizes s.st s.S s.sch ptLocal mPts) (hsh : ShapeIs s.sh s.st s.sch)
    (hcl : s.ClaimsAre) (hcap : CapShaped s.st s.sch s.cap) (hlig : ∀ r, LigShaped s.st s.sch (s.lig r))
    (hreg : s.RegionsFit) : s.Encodes := by
  obtain ⟨-, -, -, hms, hf, -, hkm, -, -⟩ := hset
  obtain ⟨hlv, hpart, hm35⟩ := fast100_file _ _ hf
  obtain ⟨hH, hsl⟩ := shapeOf_plain (sch := s.sch) (n := 2 * s.S.regions.length) hms
  have hpart' := (schedPart_plain (s := s.sch) hsl).trans hpart
  intro r
  have hlen := proofOf_lengths (params := pcsParams s.st.merkle s.sch) (d := s.d (fin2 r)) hsh hcl hcap (hlig (fin2 r))
  have hle := encodeProof_size_le _ _ hlen
  have hfit : (encodeProof (shapeOf s.st s.sch (2 * s.S.regions.length)) (s.proof r)).size < 2 ^ 64 := by
    refine lt_of_le_of_lt hle ?_
    unfold fileBound
    rw [hH]
    have e : (shapeOf s.st s.sch (2 * s.S.regions.length)).sched = s.sch := rfl
    have em : (shapeOf s.st s.sch (2 * s.S.regions.length)).m = s.st.m := rfl
    have ek : (shapeOf s.st s.sch (2 * s.S.regions.length)).kLog = s.st.kLog := rfl
    have ec : (shapeOf s.st s.sch (2 * s.S.regions.length)).claims = 2 + 2 * s.S.regions.length := rfl
    rw [e, em, ek, ec]
    unfold RegionsFit at hreg
    omega
  exact ⟨encodeProof _ (s.proof r), decodeProof_encodeProof _ _ ⟨by rw [hH]; omega, show 1 ≤ s.sch.levels.size - 1 by omega, hfit, hlen⟩⟩

/-- **`hdec`** at the table files: each rep's bytes decode to its proof file at the shape `verifyRep` decodes
with (Verify.lean:40-41), whenever the files have encodings (`Encodes`). -/
theorem hdec_filesAt {cls : HClass} (pts : Fin 2 → Fin cls.mPts → FlockLevel3.GF128)
    (hset : SetupRealizes s.st s.S s.sch cls.ptLocal cls.mPts) (henc : s.Encodes) (r : ℕ) :
    decodeProof (shapeOf (s.filesAt cls pts).st (s.filesAt cls pts).sch (s.filesAt cls pts).extras.size)
      ((s.filesAt cls pts).bytes r) = .ok ((s.filesAt cls pts).proof r) := by
  obtain ⟨e, he, hext, -⟩ := s.points_ok hset pts
  have hX : s.extrasAt (s.points cls.ptLocal cls.mPts pts) = e := s.extrasAt_of he
  have hsz : e.size = 2 * s.S.regions.length := by rw [(hext e he).1, FlockSoundness.extraClaims_length]
  obtain ⟨b, hb⟩ := henc r
  have hb' : decodeProof (shapeOf s.st s.sch (s.extrasAt (s.points cls.ptLocal cls.mPts pts)).size) b =
      .ok (s.proof r) := by
    rw [hX, hsz]; exact hb
  exact Classical.epsilon_spec (p := fun b => decodeProof (shapeOf s.st s.sch
    (s.extrasAt (s.points cls.ptLocal cls.mPts pts)).size) b = .ok (s.proof r)) ⟨b, hb'⟩

end TableSrc

section claims

variable {nT nR : ℕ} (T : Fin nT → Type) (CT : Fin nR → Type) {cls : HClass}

/-- **`ClaimsAre` from `verifierRows` accepting rep 0's coins** (`HFit`) and the reps' extra claims being the
setup's at the inner transcript's link points (`ExtrasAre`). -/
theorem TableSrc.claimsAre_of (s : TableSrc) (tr : List (List (SMsg T) × ((k : Fin nR) → CT k)))
    (hset : SetupRealizes s.st s.S s.sch cls.ptLocal cls.mPts) (hxs : s.ExtrasAre T CT cls tr)
    (hrows : ∃ x, RecAlgebra.verifierRows s.sh (s.d 0).rounds (s.d 0).extras (s.d 0).comb = .ok x) :
    s.ClaimsAre := by
  obtain ⟨x, hx⟩ := hrows
  obtain ⟨e, he, hext, hes⟩ := s.points_ok hset (ptsOf T CT tr)
  have h0 := hxs 0 _ ⟨e, he, hext, hes⟩
  rw [he] at h0
  have hd : (s.d 0).extras = e := (Except.ok.inj h0).symm
  refine s.claims_of hx ?_
  rw [hd, (hext e he).1, FlockSoundness.extraClaims_length]

end claims

section family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec FlockSoundness.Model FlockSoundness.Audit
open FlockLevel3 FlockVBridge.InnerAccepts FlockVBridge.Stages

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
variable (spec : SessionSpec) (st : Reg → L.Ω → Setup)
    (cap : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray)
    (lig : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof)

/-- **`henc`** of `hFiles_of` and `hRuns_of`: at every accepted outer proof, each rep's proof file has an encoding
(`encodeProof`), from the setup (`SetupRealizes`), V*'s shape (`ShapeIs`), `verifierRows` accepting the reps' coins
(`HFit`) and the reps' extra claims (`ExtrasAre`), the shapes of the cap and openings V*'s layers hold (`CapShaped`,
`LigShaped`), and the statement's region count (`RegionsFit`). -/
theorem hEnc_of (hset : ∀ R ω, SetupRealizes (st R ω) ((F.x R ω).toStatement regs) sch cls.ptLocal cls.mPts)
    (hsh : ∀ R ω, ShapeIs F.sh (st R ω) sch) (hfit : F.HFit dg N k Rw)
    (hxs : F.AtAccepted dg N k Rw fun R ω t l =>
      (F.tableSrc spec st cap lig R ω t l).ExtrasAre T CT cls ((F.wmsg l).zip (t.map Prod.snd)))
    (hcap : F.AtAccepted dg N k Rw fun R ω _ l => CapShaped (st R ω) sch (cap l))
    (hlig : F.AtAccepted dg N k Rw fun R ω _ l => ∀ r, LigShaped (st R ω) sch (lig l r))
    (hreg : ∀ R ω, ((F.x R ω).toStatement regs).regions.length < 2 ^ 48) :
    F.AtAccepted dg N k Rw fun R ω t l => (F.tableSrc spec st cap lig R ω t l).Encodes := by
  intro R ω t R₂ τ o hacc
  exact (F.tableSrc spec st cap lig R ω t _).encodes_of (hset R ω) (hsh R ω)
    (TableSrc.claimsAre_of T CT _ _ (hset R ω) (hxs R ω t R₂ τ o hacc) (hfit R ω t R₂ τ o hacc 0))
    (hcap R ω t R₂ τ o hacc) (hlig R ω t R₂ τ o hacc) (hreg R ω)

end VStarFamily

end family

end FlockVBridge.Family
