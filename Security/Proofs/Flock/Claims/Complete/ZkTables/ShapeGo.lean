import Proofs.Flock.Claims.Complete.ZkTables.RunsGo
import Proofs.Flock.Claims.Complete.Zk.InnerMerkle
import Proofs.Flock.Claims.Complete.Zk.ProverOk
import Proofs.Flock.Claims.Complete.Zk.Json.Files

/-!
# The honest masked rep proof fits its shape

`RepShapedT` beyond level 0's shape (`ShapeRest`): a returning masked rep (`repPZk_split`) is `ZkView.Shaped` at the
shape the masked verifier decodes it at.

* **The masked PIOP's lengths** are its closed forms' (`ZcZ.proof`, `LcZ.proof`, `Rs.proof`): `64`, `64` and `m − 6`
  zerocheck vectors, `k_log − 6` lincheck rounds and `64` partial evaluations, one ring switch of `128` per claim.
* **The cap** is level 0's masked commitment's (`commitRowsP_capSize`); **Ligerito's lengths** and `t_pad` padding
  coefficients are level 0's (`Level0.ProverShape`).
* **The inner proof's** salts are the pads' commitment's, fitted to 192 bytes (`innerCommitP_salts`), and its path
  entries are its tree's nodes, each a 64-byte digest (`innerTree_entry`), at columns below its width.
* **The file fits its length word** (`encZk_lt`): the commitment and PIOP are plain's bound with level 0's rows
  `EXTRA_LANES` wider (`wProofZk_size_le`, `schedPartZk`), level 0's messages `105 + 32·t_pad` bytes, and the inner
  proof at most `424 + 16·|y| + 168·(232 + 64·depth)`; every `--zk` schedule's part is under `2^32`
  (`zkSchedule_file`), and a good case's table has at most 2,050 claims (`Case.extras_le`).
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT FlockComplete.Replay FlockSoundness.Refine
open FlockSoundness.Discharge

/-! ## The file's length -/

/-- `Vec` of elements of at most `k` bytes. -/
theorem wVec_size_le {α : Type} (f : α → ByteArray) (xs : Array α) (k : ℕ) (h : ∀ x ∈ xs, (f x).size ≤ k) :
    (ZkView.wVec f xs).size ≤ 8 + xs.size * k := by
  rw [FlockVBridge.Family.wVec_size]
  have h' := List.sum_le_length_nsmul (xs.toList.map fun x => (f x).size) k (by
    intro a ha
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 ha
    exact h x (Array.mem_toList_iff.1 hx))
  rw [List.length_map, Array.length_toList, smul_eq_mul] at h'
  omega

/-- A level's openings, its rows `EXTRA_LANES` wider: `64` bytes more per opened row. -/
theorem levelSize_extra (H : ℕ) (sl : Option ℕ) (l : Level) (w : ℕ) :
    FlockVBridge.Family.levelSize H sl l (w + Zk.EXTRA_LANES) = FlockVBridge.Family.levelSize H sl l w + 64 * l.q := by
  unfold FlockVBridge.Family.levelSize Zk.EXTRA_LANES
  ring

/-- **A masked file's schedule part**: plain's (`schedPart`), level 0's opened rows `EXTRA_LANES` wider. -/
def schedPartZk (s : Schedule) (H : ℕ) (sl : Option ℕ) : ℕ :=
  FlockVBridge.Family.schedPart s H sl + 64 * (s.levels.getD 0 default).q

/-- **The commitment and the masked PIOP's bytes**: plain's bound (`encodeProof_size_le`), level 0's rows
`EXTRA_LANES` wider. -/
theorem wProofZk_size_le (sh : Shape) (p : Proof) (hcap : ZkView.CapOk sh (sh.sched.levels.getD 0 default) p.cap)
    (h1ab : p.zerocheck.round1ab.size = 64) (h1c : p.zerocheck.round1c.size = 64)
    (hrounds : p.zerocheck.rounds.size = sh.m - 6) (hlr : p.lincheck.rounds.size = sh.kLog - 6)
    (hzp : p.lincheck.zPartial.size = 64) (hrs : p.ringSwitches.size = sh.claims)
    (hrs128 : ∀ v ∈ p.ringSwitches, v.size = 128) (hlg : Level0.LgShaped sh p.ligerito) :
    (ZkView.wProof sh p).size ≤
      schedPartZk sh.sched sh.hashLen sh.saltLen + 32 * (sh.m - 6) + 32 * (sh.kLog - 6) + 2064 * sh.claims := by
  obtain ⟨cap, params, ⟨r1ab, r1c, rounds, fa, fb, fc⟩, ⟨lr, zp⟩, rs,
    ⟨icap, l0, rcaps, rps, ⟨yr, rowsR, sibR, saltR⟩, msgs, qn, ood, cbn, con⟩⟩ := p
  obtain ⟨hicap, hl0, hrcs, hrc, hrps, hrp, hyr, hfin, hmsg, hqn, hood, hcbn, hcon⟩ := hlg
  dsimp only at *
  have ht : ∀ a : F128, a.toBytes.size = 16 := ZkView.toBytes_size
  have hw0 : (ZkView.wLen 0).size = 8 := ZkView.le64_size _
  have hpair : ∀ q : F128 × F128, (ZkView.wPair q).size = 32 := fun q => by
    rw [ZkView.wPair, ByteArray.size_append, ht, ht]
  have e1 := FlockVBridge.Family.wVec_size_const id cap _ hcap.2
  have e2 := FlockVBridge.Family.wVec_size_const F128.toBytes r1ab 16 fun x _ => ht x
  have e3 := FlockVBridge.Family.wVec_size_const F128.toBytes r1c 16 fun x _ => ht x
  have e4 := FlockVBridge.Family.wVec_size_const ZkView.wPair rounds 32 fun x _ => hpair x
  have e5 := FlockVBridge.Family.wVec_size_const ZkView.wPair lr 32 fun x _ => hpair x
  have e6 := FlockVBridge.Family.wVec_size_const F128.toBytes zp 16 fun x _ => ht x
  have e7 := FlockVBridge.Family.wVec_size_const (fun v => ZkView.wVec F128.toBytes v ++ ZkView.wLen 0) rs 2064
    fun v hv => by
      rw [ByteArray.size_append, FlockVBridge.Family.wVec_size_const F128.toBytes v 16 (fun x _ => ht x), hrs128 v hv,
        hw0]
  have e8 := FlockVBridge.Family.wVec_size_const id icap _ hicap.2
  have e9 := FlockVBridge.Family.wLevel_size hl0
  rw [levelSize_extra] at e9
  have e10 := FlockVBridge.Family.wVec_size_idx (ZkView.wVec id) rcaps
    (fun i => 8 + 2 ^ (sh.sched.levels.getD (i + 1) default).cap * sh.hashLen) fun i hi => by
      rw [FlockVBridge.Family.wVec_size_const id _ _ (hrc i hi).2, (hrc i hi).1]
  have e11 := FlockVBridge.Family.wVec_size_idx (ZkView.wLevel sh) rps
    (fun i => FlockVBridge.Family.levelSize sh.hashLen sh.saltLen (sh.sched.levels.getD (i + 1) default) 16)
    fun i hi => FlockVBridge.Family.wLevel_size (hrp i hi)
  have e12 := FlockVBridge.Family.wVec_size_const F128.toBytes yr 16 fun x _ => ht x
  have e13 := FlockVBridge.Family.wLevel_size hfin
  rw [ZkView.wLevel, ByteArray.size_append, ByteArray.size_append] at e13
  have e14 := FlockVBridge.Family.wVec_size_const (fun m : F256 × F256 => ZkView.wF256 m.1 ++ ZkView.wF256 m.2) msgs
    64 fun x _ => by simp only [ZkView.wF256, ByteArray.size_append, ht]
  have e15 := FlockVBridge.Family.wVec_size_const le64 qn 8 fun x _ => ZkView.le64_size x
  have e16 := FlockVBridge.Family.wVec_size_const F128.toBytes ood 16 fun x _ => ht x
  have e17 := FlockVBridge.Family.wVec_size_const le64 cbn 8 fun x _ => ZkView.le64_size x
  have e18 := FlockVBridge.Family.wVec_size_const le64 con 8 fun x _ => ZkView.le64_size x
  have hpar := FlockVBridge.Family.wParams_size_le params
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
  simp only [ZkView.wProof, ZkView.wLigerito, ByteArray.size_append, ht, hw0] at *
  simp only [schedPartZk, FlockVBridge.Family.schedPart]
  omega

/-- **Level 0's hiding messages' bytes**: the tag, three K elements and the padding coefficients. -/
theorem wLevel0_size (z : Zk.Level0Msgs) : (ZkView.wLevel0 z).size = 105 + 32 * z.ybar.size := by
  have hk : ∀ x : F256, (ZkView.wF256 x).size = 32 := fun x => by
    simp only [ZkView.wF256, ByteArray.size_append, ZkView.toBytes_size]
  rw [ZkView.wLevel0, ByteArray.size_append, ByteArray.size_append, ByteArray.size_append, ByteArray.size_append,
    FlockVBridge.Family.wVec_size_const _ _ 32 fun x _ => hk x, hk, hk, hk]
  show 1 + 32 + 32 + 32 + (8 + z.ybar.size * 32) = _
  omega

/-- **The inner proof's bytes**, at a 64-byte root and `τ` commitment, a 192-byte `τ` salt, `168` openings, salts
of 192 bytes, and paths of at most `nD` 64-byte entries. -/
theorem wInner_size_le (p : Zk.InnerProof) {nY nD : ℕ} (hr : p.root.size = 64) (htc : p.tauCommit.size = 64)
    (hts : p.tauSalt.size = 192) (hy : p.y.size ≤ nY) (ho : p.opened.size = 168) (hsn : p.salts.size = 168)
    (hs : ∀ x ∈ p.salts, x.size = 192) (hp : p.paths.size = 168) (hpd : ∀ q ∈ p.paths, q.size ≤ nD)
    (hpe : ∀ q ∈ p.paths, ∀ x ∈ q, x.size = 64) :
    (ZkView.wInner p).size ≤ 424 + 16 * nY + 168 * (232 + 64 * nD) := by
  have ht : ∀ a : F128, a.toBytes.size = 16 := ZkView.toBytes_size
  have hpair : ∀ q : F128 × F128, (ZkView.wPair q).size = 32 := fun q => by
    rw [ZkView.wPair, ByteArray.size_append, ht, ht]
  have e1 := FlockVBridge.Family.wVec_size_const F128.toBytes p.y 16 fun x _ => ht x
  have e2 := FlockVBridge.Family.wVec_size_const ZkView.wPair p.opened 32 fun x _ => hpair x
  have e3 := FlockVBridge.Family.wVec_size_const id p.salts 192 hs
  have e4 := wVec_size_le (ZkView.wVec id) p.paths (8 + nD * 64) fun q hq => by
    rw [FlockVBridge.Family.wVec_size_const id q 64 (hpe q hq)]
    have := Nat.mul_le_mul_right 64 (hpd q hq)
    omega
  simp only [ZkView.wInner, ZkView.wBytes, ZkView.wLen, ByteArray.size_append, ZkView.le64_size, ht, hr, htc, hts,
    e1, e2, e3, ho, hsn] at *
  rw [hp] at e4
  have := Nat.mul_le_mul_right 16 hy
  omega

/-- **Every `--zk` schedule's file part is under `2^32`** at hm96-sha512's digests and salts, and its `t_pad` is at
most `1024`. -/
theorem zkSchedule_file {m : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) {v : Schedule × ℕ} (h : Zk.schedule m = .ok v) :
    schedPartZk v.1 64 (some 192) ≤ 2 ^ 32 ∧ v.2 ≤ 1024 := by
  have key : ∀ k : Fin 11, (match Zk.schedule (k.val + 25) with
      | .ok v => decide (schedPartZk v.1 64 (some 192) ≤ 2 ^ 32 ∧ v.2 ≤ 1024)
      | .error _ => false) = true := by
    decide +kernel
  have := key ⟨m - 25, by omega⟩
  simp only [show m - 25 + 25 = m by omega, h, decide_eq_true_eq] at this
  exact this

/-! ## The inner proof's salts and paths -/

/-- The pads' commitment's salts: leaf `j`'s salt fitted to 192 bytes. -/
theorem innerCommitP_salts {hs : Hashes V} {p : PadsZk} {salt : ℕ → ByteArray} {ic : InnerCommit}
    (h : innerCommitP hs p salt = .ok ic) :
    ic.salts = (Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)).map
      fun j => fitBytes 192 (salt j) := by
  unfold innerCommitP at h
  obtain ⟨leaves, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨r, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases h
  rfl

/-- **Each entry of the pads' tree is a 64-byte digest**, for a SHA-512 of 64-byte outputs: layer `k`'s entries
below `2^(log₂ n − k)` are leaves (`k = 0`) or nodes. -/
theorem innerTree_entry {h : ByteArray → ByteArray} (h64 : ∀ b, (h b).size = 64) {n : ℕ} {ic : InnerCommit}
    (ht : InnerTree (Hashes.ofSha512 h) n ic) (hn : n = 2 ^ Nat.log2 n) {k j : ℕ} (hk : k ≤ Nat.log2 n)
    (hj : j < 2 ^ (Nat.log2 n - k)) : ((ic.layers[k]!)[j]!).size = 64 := by
  rcases k with _ | k
  · have hl := ht.leaf j (by rw [hn]; simpa using hj)
    obtain ⟨x, -, hl⟩ := FlockSoundness.Refine.ebind_ok hl
    obtain ⟨d, hd⟩ : ∃ d, (Hm96.Default512.leafWith (Hashes.ofSha512 h : Hashes V) x ic.salts[j]!) = .ok (h d) :=
      ⟨_, rfl⟩
    rw [hd] at hl
    rw [← Except.ok.inj hl]
    exact h64 d
  · exact hm96_node h64 _ _ (ht.node k (by omega) j (by rwa [show Nat.log2 n - k - 1 = Nat.log2 n - (k + 1) by omega]))

/-- A column's sibling index at layer `k < log₂ n` is within the layer. -/
theorem sibling_lt {a j k : ℕ} (hj : j < 2 ^ a) (hk : k < a) : (j >>> k) ^^^ 1 < 2 ^ (a - k) := by
  have h1 : j >>> k < 2 ^ (a - k) := by
    rw [Nat.shiftRight_eq_div_pow, Nat.div_lt_iff_lt_mul (Nat.two_pow_pos k), ← Nat.pow_add,
      show a - k + k = a by omega]
    exact hj
  exact Nat.xor_lt_two_pow h1 (Nat.one_lt_two_pow (by omega))

/-- `x` in `pushes f n` is some `f j`, `j < n`. -/
theorem mem_pushes {α : Type} {f : ℕ → α} {n : ℕ} {x : α} (h : x ∈ pushes f n) : ∃ j < n, x = f j := by
  induction n with
  | zero => simp [pushes] at h
  | succ n ih =>
    rw [pushes, Array.mem_push] at h
    rcases h with h | rfl
    · obtain ⟨j, hj, rfl⟩ := ih h
      exact ⟨j, by omega, rfl⟩
    · exact ⟨n, by omega, rfl⟩

end FlockComplete.ZkTables
