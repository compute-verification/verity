module

public import Proofs.Flock.Claims.Complete.ZkTables.RepRuns
public import Proofs.Flock.Claims.Complete.Zk.RoundBytes
public import Proofs.Flock.Claims.Complete.Zk.Level0HonestRun
public import Proofs.Flock.Claims.Complete.Zk.Level0Premises
public import Proofs.Flock.Claims.Complete.Zk.Level0Verifier
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
import Proofs.Flock.Claims.Complete.Json.Goals
import Proofs.Flock.Claims.Complete.Replay.LigeritoBasis
import Proofs.Flock.Claims.Complete.Replay.LigeritoSum
import Proofs.Flock.Claims.Complete.Replay.ProverOk
import Proofs.Flock.Claims.Complete.Zk.Count
import Proofs.Flock.Claims.Complete.Zk.Json.Files
import Proofs.Flock.Claims.Complete.Zk.RingSwitch
import Proofs.Flock.Claims.Complete.Zk.Level0Honest
import Proofs.Flock.Claims.Complete.Zk.PadAlgebra

/-!
# The masked prover's rep: its round sizes

* `Lg.capBytes_size`, `ybarFlat_size`, `y_padsOf_size`, `padLayout_len_le`: the honest sizes a masked rep absorbs.
* `zkSchedule_sizes`: at `25 ≤ m ≤ 35` the `--zk` schedule's `t_pad`, `ext`, `k_0` and every level's cap are small,
  by the kernel's evaluation at either statement version.
* `rep_bytes`: every round of a masked rep that returns holds at most `|d| + 2^16` bytes, `d` its stream's domain. The
  rep splits into its PIOP (`ZcZ.tEnd`, `LcZ.tEnd`, `Rs.tEnd`), its Ligerito, which ends at the masked model's
  transcript (`L0.honestZ`), and its inner proof (`innerP_ret`); each stage's rounds are bounded by the bytes it absorbs.
-/

@[expose] public section

namespace FlockComplete.Replay

open Flock Transcript FlockComplete

/-- A cap of `w`-byte digests absorbs `w` bytes per digest (`flat_size_of`). -/
theorem Lg.capBytes_size {c : Array ByteArray} {w : ℕ} (h : ∀ x ∈ c, x.size = w) :
    (Lg.capBytes c).size = w * c.size :=
  flat_size_of h

theorem y_padsOf_size (ω : TapeZk) (i rep : ℕ) (lay : Zk.PadLayout) (cs : InZ.Challenges) :
    (InZ.y (padsOf ω i rep lay) cs).size = lay.len + Zk.K_PAD := by
  simp only [InZ.y, Array.size_map, Array.size_range, Array.size_append, padsOf_size]
  simp [padsOf]

theorem padLayout_len_le (m kLog k0 : ℕ) : (Zk.PadLayout.new m kLog k0).len ≤ 2 * m + 2 * kLog + 4 * k0 + 210 := by
  simp only [Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.hab, Zk.PadLayout.zp, Zk.PadLayout.lc,
    Zk.PadLayout.fc, Zk.PadLayout.fa, Zk.PadLayout.new]
  omega

end FlockComplete.Replay

namespace FlockComplete.L0

open Flock Transcript FlockComplete Replay

variable {wideLeaf : Bool}

theorem ybarFlat_size (ybar : Array F256) : (ybarFlat ybar).size = 2 * ybar.size := by
  have key : ∀ (l : List F256) (acc : Array F128),
      (l.foldl (fun acc v => (acc.push v.c0).push v.c1) acc).size = acc.size + 2 * l.length := by
    intro l
    induction l with
    | nil => intro acc; simp
    | cons x l ih =>
      intro acc
      rw [List.foldl_cons, ih]
      simp only [Array.size_push, List.length_cons]
      omega
  unfold ybarFlat
  rw [← Array.foldl_toList, key]
  simp

/-- **The `--zk` schedule is small**: `t_pad ≤ 554`, `ext ≤ 8`, `k_0 ≤ 10` and every level's cap at most `8`. -/
theorem zkSchedule_sizes {m : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) {v : Schedule × ℕ}
    (h : Zk.schedule wideLeaf m = .ok v) :
    v.2 ≤ 554 ∧ v.1.ext ≤ 8 ∧ v.1.k0 ≤ 10 ∧ ∀ i < v.1.levels.size, (v.1.levels.getD i default).cap ≤ 8 := by
  have key : ∀ (w : Bool) (k : Fin 11), (match Zk.schedule w (k.val + 25) with
      | .ok v => decide (v.2 ≤ 554 ∧ v.1.ext ≤ 8 ∧ v.1.k0 ≤ 10 ∧
          ∀ i < v.1.levels.size, (v.1.levels.getD i default).cap ≤ 8)
      | .error _ => false) = true := by
    decide +kernel
  have hkey := key wideLeaf ⟨m - 25, by omega⟩
  simp only [show m - 25 + 25 = m by omega, h, decide_eq_true_eq] at hkey
  exact hkey

end FlockComplete.L0

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy FlockComplete.Replay

variable {H : Claims.Msg → Claims.Msg}

/-- **Level 0's masked cap is at most `2^14` bytes**: `2^cap` digests of 64 bytes, `cap ≤ 8`. -/
theorem capBytes_le (h64 : Claims.Out64 H) {wideLeaf : Bool} {m : ℕ} (h25 : 25 ≤ m) (h35 : m ≤ 35)
    {v : Schedule × ℕ} (hv : Zk.schedule wideLeaf m = .ok v) {g extra lanePad salt} {cm : Commit}
    (hcm : commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) MerkleScheme.hm96Sha512 (v.1.levels.getD 0 default)
      v.2 g extra lanePad salt = .ok cm) :
    (Lg.capBytes cm.cap).size ≤ 16384 := by
  have h64' := JsonRT.hOf_size h64
  obtain ⟨-, -, -, hcap8⟩ := L0.zkSchedule_sizes h25 h35 hv
  obtain ⟨-, -, hL, -, -, -, -, -, -, -, hcapd, -⟩ := L0.zkSch_facts h25 h35 hv
  have hnode : ∀ l r v, (MerkleScheme.hm96Sha512.opsWith (Hashes.ofSha512 (Claims.hOf H) : Hashes V)).node l r =
      .ok v → v.size = MerkleScheme.hm96Sha512.digestLen := fun _ _ _ e => hm96_node h64' _ _ e
  have hleaf : ∀ row s v, (MerkleScheme.hm96Sha512.opsWith (Hashes.ofSha512 (Claims.hOf H) : Hashes V)).leaf row s =
      .ok v → v.size = MerkleScheme.hm96Sha512.digestLen := fun _ _ _ e => hm96_leaf h64' _ _ e
  obtain ⟨-, -, -, -, -, hcs0, hcw0⟩ := L0.commitPZk_shape hnode hleaf (hcapd 0 (by omega)) hcm
  rw [Lg.capBytes_size hcw0, hcs0]
  exact (Nat.mul_le_mul (by decide : MerkleScheme.hm96Sha512.digestLen ≤ 64)
    (Nat.pow_le_pow_right (by decide) (hcap8 0 (by omega)))).trans (by decide)

/-- **Every round of a masked rep that returns holds at most `|d| + 2^16` bytes**, `d` its stream's domain, at a
statement of `25 ≤ m ≤ 35` under `hm96-sha512/v1` with `k_log ≤ m`. -/
theorem rep_bytes (h64 : Claims.Out64 H) (st : Flock.Stmt) (tr : Trace) (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35)
    (hms : st.merkle = MerkleScheme.hm96Sha512) (hkm : st.c.kLog ≤ st.m) (ω : TapeZk) (i : ℕ) (d : ByteArray) {k : ℕ} (r : Fin k → Coin) (v : Schedule × ℕ) (cm : Commit)
    (pts : Array (Array F128)) (e : Array Claim) (rep : ℕ) (ic : InnerCommit)
    (hin : RepIn H st tr ω i v cm pts e rep ic) (hrep : rep < 2) (prior : Array (List F128)) (ws : List ℕ)
    (l : Array Claims.Msg) (p : Zk.ZkProof) (t' : Transcript) (l' : Array Claims.Msg)
    (hgo : (repT H r st tr ω i d v cm e rep ic prior ws).go l = (.ok (p, t'), l')) :
    RoundsLe (d.size + 65536) t' := by
  have hm14 : 14 ≤ (Setup.ofCircuit st).m := by show 14 ≤ st.m; omega
  have hmerk : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 := hms
  set resp : ℕ → Claims.Msg := fun j => (Claims.coinSeq r j).bytes with hresp
  set lay := Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0 with hlay
  set pads := padsOf ω i rep lay with hpads
  unfold repT at hgo
  obtain ⟨full, lL, lg, zl, rec, tI, lI, rp, x, lgc, hfold, hlL, hlig, hre, hlgc, hinn, hP, hzl⟩ :=
    repPZk_split resp (Claims.hOf H) (Setup.ofCircuit st) v.1 v.2 rep d tr (packed tr (Setup.ofCircuit st).m) cm e
      pads ic (ω.extra i) (ω.lanePad i) (ω.salt i rep) (ω.nonce i rep) prior ws hm14 _ l hgo
  have hn : v.1.n = st.m - 7 := zkSchedule_n hin.schedule
  have hx : ∀ c ∈ claimsR resp (Setup.ofCircuit st) tr l.size e, c.x.size = st.m - 6 :=
    claims_xsize' st hkm (by omega) tr (Zc.coinsAt resp l.size) (lcCoinsR resp (Setup.ofCircuit st) l.size) pts e
      hin.extra
  have hcs : (claimsR resp (Setup.ofCircuit st) tr l.size e).size = 2 + e.size := by simp [claimsR, abcClaims]
  have hpk : (tPiop resp (Setup.ofCircuit st) tr l.size lay pads.h (Setup.ofCircuit st).digest (flat cm.cap) ic.root
      full e (Transcript.init #[] d)).k = 3 + (st.m - 6) + 3 + (st.c.kLog - 6) + (2 + e.size) + 1 := by
    simp only [tPiop, Rs.tEnd_k, LcZ.tEnd_k, ZcZ.tEnd_k, hcs]
    show 0 + 3 + (st.m - 6) + 3 + (st.c.kLog - 6) + (2 + e.size) + 1 = _
    omega
  have hLk : lL.size = l.size + (tPiop resp (Setup.ofCircuit st) tr l.size lay pads.h (Setup.ofCircuit st).digest
      (flat cm.cap) ic.root full e (Transcript.init #[] d)).k := by
    rw [hlL, hpk]
    have : piopSends (Setup.ofCircuit st) e = 3 + (st.m - 6) + 3 + (st.c.kLog - 6) + (2 + e.size + 1) := rfl
    omega
  have h64' := JsonRT.hOf_size h64
  obtain ⟨hv2, hext, hk0, hcap8⟩ := L0.zkSchedule_sizes h25 h35 hin.schedule
  have hcm : commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) MerkleScheme.hm96Sha512
      (v.1.levels.getD 0 default) v.2 (packed tr (Setup.ofCircuit st).m) (ω.extra i) (ω.lanePad i) (ω.salt i 0 0) =
        .ok cm := by
    rw [← hmerk]; exact hin.commit
  have hcap0 := capBytes_le h64 h25 h35 hin.schedule hcm
  rw [hmerk] at hlig
  obtain ⟨ht', -⟩ := L0.honestZ (sch := v.1) (tPad := v.2) resp (Claims.hOf H) MerkleScheme.hm96Sha512 _ cm _ _ lay
    pads.h rep (ω.extra i) (ω.lanePad i) (ω.salt i rep) (ω.nonce i rep)
    (basis (Rs.sws (claimsR resp (Setup.ofCircuit st) tr l.size e) (rsCoinsR resp (Setup.ofCircuit st) l.size e)))
    l.size _ lL h25 h35 hin.schedule hrep (by simp [packed, hn]; rfl) (by simp [basisVals, hn]; rfl)
    (L0.commitPZk_rows hcm) (fun q e' y hqe hy => by rw [hn] at hqe ⊢; exact Lg.basis_sws _ _ _ hx q e' y hqe hy)
    hLk hlig
  obtain ⟨hlg, hyb⟩ := L0.proverShape (Setup.ofCircuit st).wideLeaf resp (Claims.hOf H) (Setup.ofCircuit st).m v
    { m := 0, kLog := 0, claims := 0, sched := v.1, hashLen := MerkleScheme.hm96Sha512.digestLen,
      saltLen := if MerkleScheme.hm96Sha512.saltBytes = 0 then none else some MerkleScheme.hm96Sha512.saltBytes }
    _ cm _ _ lay pads.h rep (ω.extra i) (ω.lanePad i) (ω.salt i 0 0) (ω.salt i rep) (ω.nonce i rep) _ tI lL lI lg zl
    rec h64' h25 h35 hin.schedule rfl rfl rfl (by simp [packed, hn]; rfl) (by simp [basisVals, hn]; rfl) hcm hlig
  obtain ⟨-, -, hrs, hrcap, -, -, hyr, -⟩ := hlg
  dsimp only at hrs hrcap hyr
  have hrc : ∀ j : ℕ, (Lg.capBytes lg.recursiveCaps[j]!).size ≤ 16384 := by
    intro j
    by_cases hj : j < lg.recursiveCaps.size
    · obtain ⟨hs1, hs2⟩ := hrcap j hj
      rw [getElem!_pos lg.recursiveCaps j hj, Lg.capBytes_size hs2, hs1]
      exact (Nat.mul_le_mul (by decide : MerkleScheme.hm96Sha512.digestLen ≤ 64)
        (Nat.pow_le_pow_right (by decide) (hcap8 (j + 1) (by omega)))).trans (by decide)
    · rw [getElem!_neg lg.recursiveCaps j hj, show (default : Array ByteArray) = #[] from rfl]
      simp [Lg.capBytes]
  have hyr' : lg.finalProof.yr.size ≤ 512 := by
    rw [hyr]
    exact (Nat.pow_le_pow_right (by decide) (by omega : v.1.ext + 1 ≤ 9)).trans (by decide)
  have hyb' : (L0.ybarFlat zl.ybar).size ≤ 1108 := by
    rw [L0.ybarFlat_size, hyb]
    omega
  have hD : (Setup.ofCircuit st).digest.size ≤ 16384 := by
    show (st.digest.extract 0 32).size ≤ _
    rw [ByteArray.size_extract]
    omega
  have hroot : ic.root.size ≤ 16384 := by
    rw [JsonRT.innerCommitP_root h64' _ _ hin.innerCommit]
    decide
  have hlen : lay.len ≤ 2 * st.m + 2 * st.c.kLog + 4 * v.1.k0 + 210 := padLayout_len_le _ _ _
  obtain ⟨-, -, -, ht''⟩ := innerP_ret resp (Claims.hOf H) _ _ _ _ _ _ _ hinn
  rw [ht'', ht']
  refine (InZ.wb_tEnd (P := 0) (C := 16384) (NI := 600) ?_ ((tcOf_size h64' _ _).le.trans (by decide)) ?_
    (by omega) (by omega) (by omega) (by omega)).1
  · refine L0.wb_outZ (C := 16384) (NY := 512) (NB := 1108) ?_ hcap0 hrc hyr' hyb' (by omega)
    unfold tPiop
    refine Rs.wb_tEnd (P := 0) (V := 128) ?_ (fun _ => (Rs.switchVec_size _ _).le) (by omega)
    refine LcZ.wb_tEnd (P := 64) ?_ (by omega) (by omega)
    exact ZcZ.wb_tEnd (Wb.init d) hD (hcap0 : (flat cm.cap).size ≤ 16384) hroot (by omega) (by omega)
  · rw [hpads, y_padsOf_size]
    show lay.len + 192 ≤ 600
    omega

end FlockComplete.ZkTables
