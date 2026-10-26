import Proofs.Flock.SessionLoop.FinRepErasedCut
import Proofs.Flock.SessionLoop.FinRepXs
import Proofs.Flock.SessionLoop.FinRepView
import Proofs.Flock.Claims.ZK.PiopFree
import Proofs.Flock.Claims.ZK.Round1Root

/-!
# The views' cut PIOP returns where the firewall's rep did

The cut PIOP returns on its coins alone (`Masked.piopCut_of_repFwC`), so where the session's rep on the loop's table
and request returned (`RepAt`, as `finRepCut` runs it at the opened coins), the cut PIOP from any source, layout, cap,
root and domain returns from the rep's base, at table `j`'s setup, the views' extra claims, the earlier reps' points
and the mask slot's words. No premise on the views' masked values, and none on the rows.

**`Loop.piopCut_views_ret`**: that return. At the views' source, `flat cms[j]!.cap`, `ic.root` and the session's
domain it is `FinPiop`'s first conjunct and the `hpiop` that `swSrcE_hmk_cut` and `swSrcE_l0_live` take.

**`Loop.finPiop`**: `FinPiop` itself, its cap and pads-root bounds from the `--zk` schedule (`commit0Cut_cap_lt`) and
the pads tree's size (`padsTreeCut_root_go`). **`Loop.finRepErased_of_fc`**: `finRepErased_of_cut` without it.
**`Loop.swSrcE_l0_views`**: `swSrcE_l0_live` with no PIOP premise, which reads no rows.
-/

namespace FlockComplete

open _root_.Flock

/-- The `--zk` schedule's level 0 caps strictly below its depth: its tree hashes at least once. -/
theorem JsonRT.zkSchedule_cap0_lt {m : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) {v : _root_.Flock.Schedule × ℕ}
    (h : _root_.Flock.Zk.schedule m = .ok v) :
    (v.1.levels.getD 0 default).cap < (v.1.levels.getD 0 default).d := by
  have key : ∀ k : Fin 11, (match _root_.Flock.Zk.schedule (k.val + 25) with
      | .ok v => decide ((v.1.levels.getD 0 default).cap < (v.1.levels.getD 0 default).d)
      | .error _ => false) = true := by
    decide +kernel
  have := key ⟨m - 25, by omega⟩
  simp only [show m - 25 + 25 = m by omega, h, decide_eq_true_eq] at this
  exact this

/-- **Level 0's cap from a source is under `2^64` bytes**: through SHA-512 under `hm96-sha512/v1`, at a level whose
tree hashes at least once and that caps at most `32`, its cap is `2^cap` digests. -/
theorem commit0Cut_cap_lt (lv : Level) (src : TabSrc) (hc : lv.cap < lv.d) (h32 : lv.cap ≤ 32) :
    ∃ cm, (commit0Cut (Hashes.ofSha512 Sha512.hash : Hashes V) MerkleScheme.hm96Sha512 lv src : V Commit) = .ok cm ∧
      (flat cm.cap).size < 2 ^ 64 := by
  obtain ⟨cm, hcm, hcap⟩ := commit0Cut_cap lv src
  refine ⟨cm, hcm, ?_⟩
  obtain ⟨h, hh⟩ : ∃ h, lv.d - lv.cap = h + 1 := ⟨lv.d - lv.cap - 1, by omega⟩
  have hmem : ∀ b ∈ cm.cap, b.size = 64 := by
    rw [hcap, hh]
    intro b hb
    simp only [layerOf, Array.mem_map] at hb
    obtain ⟨k, -, rfl⟩ := hb
    exact Sha512.hash_size _
  have hsz : cm.cap.size = 2 ^ lv.cap := by
    rw [hcap, layerOf_size _ _ _ _ (Nat.sub_le _ _)]
    congr 1
    omega
  rw [flat_size hmem, hsz]
  calc 64 * 2 ^ lv.cap ≤ 64 * 2 ^ 32 := Nat.mul_le_mul_left _ (Nat.pow_le_pow_right (by norm_num) h32)
    _ < 2 ^ 64 := by norm_num

/-- **A pads tree's root is under `2^64` bytes** through SHA-512, where the tree hashes at least once: it is a digest
of its last layer, or empty. -/
theorem padsTreeCut_root (n : ℕ) (src : RepSrc) (hn : 1 ≤ Nat.log2 n) :
    ∃ ic, (padsTreeCut (Hashes.ofSha512 Sha512.hash : Hashes V) n src : V InnerCommit) = .ok ic ∧
      ic.root.size < 2 ^ 64 := by
  unfold padsTreeCut
  simp only [bind, Except.bind]
  generalize hF : forIn (m := V) (β := Array (Array ByteArray) × Array ByteArray) [0:Nat.log2 n]
    (#[(Array.range n).map src.padLeaf], (Array.range n).map src.padLeaf) _ = e
  obtain ⟨⟨L, C⟩, hr, hI⟩ := forIn_ok_inv' (Nat.log2 n)
    (#[(Array.range n).map src.padLeaf], (Array.range n).map src.padLeaf)
    (fun i (p : Array (Array ByteArray) × Array ByteArray) => 1 ≤ i → ∀ b ∈ p.2, b.size = 64)
    (fun h0 => absurd h0 (by omega)) (by
      intro i _ p _
      refine ⟨(p.1.push ((Array.range (p.2.size / 2)).map fun i => Sha512.hash (p.2[2 * i]! ++ p.2[2 * i + 1]!)),
        (Array.range (p.2.size / 2)).map fun i => Sha512.hash (p.2[2 * i]! ++ p.2[2 * i + 1]!)), ?_, ?_⟩
      · rw [mapM_ok _ fun i => (rfl : (MerkleOps.hm96With (Hashes.ofSha512 Sha512.hash : Hashes V)).node
          p.2[2 * i]! p.2[2 * i + 1]! = .ok (Sha512.hash (p.2[2 * i]! ++ p.2[2 * i + 1]!)))]
        rfl
      · intro _ b hb
        obtain ⟨k, -, rfl⟩ := Array.mem_map.1 hb
        exact Sha512.hash_size _) hF
  subst hr
  refine ⟨_, rfl, ?_⟩
  show (C[0]!).size < 2 ^ 64
  by_cases hc : 0 < C.size
  · rw [getElem!_pos C 0 hc, hI hn _ (Array.getElem_mem hc)]
    norm_num
  · rw [getElem!_neg C 0 hc]
    decide

/-- **A rep's pads root through the cut's hashes is under `2^64` bytes**, at a tree of at least two leaves. -/
theorem padsTreeCut_root_go {N : ℕ} {src : RepSrc} {ic : InnerCommit} (hN : 2 ≤ N)
    (h : ∀ l, (padsTreeCut (Hashes.ofSha512 Sha512.hash : Hashes PS) N src).go l = (.ok ic, l)) :
    ic.root.size < 2 ^ 64 := by
  have hv := h #[]
  rw [CutEq.padsTreeCut_quiet] at hv
  obtain ⟨ic₀, h₀, hr₀⟩ := padsTreeCut_root N src ((Nat.le_log2 (by omega)).2 (by simpa using hN))
  rw [Except.ok.inj ((Prod.mk.inj hv).1.symm.trans h₀)]
  exact hr₀

/-- A pads tree's leaf count, `2^3` times a power of two, is at least two. -/
theorem two_le_padsN (P : ℕ) : 2 ≤ P.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE := by
  have hP : 0 < P.nextPowerOfTwo := Nat.pos_of_isPowerOfTwo (Nat.isPowerOfTwo_nextPowerOfTwo _)
  have h8 : 2 ^ Zk.INNER_LOG_INV_RATE = 8 := by
    unfold Zk.INNER_LOG_INV_RATE
    norm_num
  rw [h8]
  omega

end FlockComplete

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Loop

variable (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)

set_option maxHeartbeats 1000000 in
/-- **The views' cut PIOP returns**: in the masked scope, at a witness in the relation, where the session's rep on the
loop's table and request at table `j`'s rep `r` returned (`RepAt`), the cut PIOP at table `j`'s setup, answered by the
opened coins, from any source, layout, cap, root and domain, at the views' extra claims, the earlier reps' points and
the mask slot's words, returns from the rep's base. -/
theorem piopCut_views_ret (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Chk} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)} (hF : FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops)
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs)
    (j : Fin (Masked.jOf x)) (r : Fin 2) {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
    {q : _root_.Flock.Firewall.Finish.RepIn} {f : ByteArray}
    (hRep : RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec (2 * j.val + r.val) tb hi q f)
    {extras : Array _root_.Flock.Claim}
    (hex : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (viewPoints x ops spec) = .ok extras)
    (lay : _root_.Flock.Zk.PadLayout) (rs : FlockComplete.RepSrc) (cap root dom : ByteArray) {l : Array Msg}
    (hl : l.size = Masked.repBase x (2 * j.val + r.val)) :
    ∃ res tr₁ l₁, ((FlockComplete.piopCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x))
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]! lay rs cap root extras (Masked.priorE (finCoins x ops) j.val r.val)
      (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[j.val]!).run (_root_.Flock.Transcript.init #[] dom)).go l =
      (.ok (res, tr₁), l₁) := by
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hj : j.val < (Masked.stsOf x).size := Masked.lt_of_jOf j
  obtain ⟨-, -, hts, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hk : 2 * j.val + r.val < 2 * spec.tables.size := by
    have := r.isLt
    omega
  have hk2 : (2 * j.val + r.val) / 2 = j.val := by
    have := r.isLt
    omega
  have hA := admits_of_finCtx L₀ src (realOf src) wk hServe hF
  obtain ⟨ic', p, tr, hic', hfw, -⟩ := _root_.Flock.Firewall.Finish.proofFile_ok hRep.2.2
  have hcut := finRepCut L₀ src (realOf src) wk x w hsc ν t κ m₀ ρ₀ ops hF s0 spec hsx _ hk tb hi q _ hRep ic' p tr
    hic' hfw l hl
  obtain ⟨htb, -⟩ := finRepTab (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hs hk hRep
  rw [hk2] at htb
  rw [← htb] at hcut
  have hxs := finRepXs L₀ src (realOf src) wk x w hsc hR ν t κ m₀ ρ₀ ops hF s0 spec hsx _ hk extras
    (by rw [hk2]; exact hex) tb hi q _ hRep
  have hcx := congrArg Array.size (_root_.Flock.Firewall.Finish.claimsOf_map_x q.xs)
  have hex' := congrArg Array.size hxs
  rw [Array.size_map] at hcx hex'
  have hes : extras.size = (_root_.Flock.Firewall.Finish.claimsOf q.xs).size := by omega
  have hst := Masked.tabsOf_stAt x j
  have hstj : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hj
  have hsm : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m = (Masked.stsOf x)[j.val]!.m := by
    rw [hst, hstj]
    rfl
  have hsk : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog = (Masked.stsOf x)[j.val]!.c.kLog := by
    rw [hst, hstj]
    rfl
  have hfk : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.fold.kLog =
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog := by
    rw [hst]
    rfl
  obtain ⟨res, t₃, l₃, h⟩ := Masked.piopCut_of_repFwC hx (c := finCoins x ops) hA hj r.isLt (refK x) hsm hsk hfk
    (go_eq_of_run_fst hcut) hl lay rs cap root hes (t₀' := _root_.Flock.Transcript.init #[] dom) rfl
  rw [FlockComplete.MaskRank.masksOf_getElem! hj]
  exact ⟨res, t₃, l₃, h⟩

set_option maxHeartbeats 1000000 in
/-- **Level 0's values at the views' source, with no PIOP premise**: `swSrcE_l0_live` at `piopCut_views_ret`'s return
from the views' source, `flat cms[j]!.cap`, `ic.root` and the session's domain. -/
theorem swSrcE_l0_views (vs : finS.SwVals)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Chk} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)} (hF : FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops)
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit}
    (hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut Masked.hsE (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle
        (schs[i]!.1.levels.getD 0 default)
        (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t i)).go l = (.ok cms, l))
    (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hexts : ∀ k < 2 * j.val + r.val, ∃ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras)
    (hprev : ∀ k < 2 * j.val + r.val, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras → ∀ ic,
      (∀ l, (FlockComplete.padsTreeCut Masked.hsE
        (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.m
          (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.kLog schs[k / 2]!.1.k0).len +
          _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
        ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k / 2)).rep (k % 2))).go l =
          (.ok ic, l)) →
      ∀ l : Array Msg, l.size = Masked.repBase x k →
      ∃ p tr l', ((FlockComplete.repCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x)) Masked.hsE
          (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! schs[k / 2]!.1 (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 cms[k / 2]! extras ic
          ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k / 2)).rep (k % 2))
          (Masked.priorE (finCoins x ops) (k / 2) (k % 2))
          (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[k / 2]!).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l =
          (.ok (p, tr), l') ∧ l'.size = Masked.repBase x (k + 1))
    {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray} {q : _root_.Flock.Firewall.Finish.RepIn}
    {f : ByteArray} (hRep : RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec (2 * j.val + r.val) tb hi q f)
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (viewPoints x ops spec) = .ok extras)
    {ic : FlockComplete.InnerCommit}
    (hic : ∀ l, (FlockComplete.padsTreeCut Masked.hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)).go l = (.ok ic, l))
    {l : Array Msg} (hl : l.size = Masked.repBase x (2 * j.val + r.val))
    {k : ℕ} (hlo : (Masked.layE x j).zp 64 ≤ k) (hhi : k < (Masked.layE x j).hab) :
    ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val).masked k =
      FlockComplete.LigLock.l0Clear (Masked.ligCoin (finCoins x ops)) (Masked.ligP0 x j r) (Masked.zkTab x j).sch tb.g
          (Masked.ligBasis x (finCoins x ops) w t j r extras) (Masked.layE x j) tb.extra r.val k +
        (_root_.Flock.Firewall.Finish.padsOf q.draws).h[k]! := by
  obtain ⟨res, tr₁, l₁, hpiop⟩ := piopCut_views_ret L₀ src wk x w hsc hR hServe hF hsx hs j r hRep hext
    (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0)
    ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)
    (FlockComplete.flat cms[j.val]!.cap) ic.root (spec.domain spec.tables[j.val]! r.val).toUTF8 hl
  exact swSrcE_l0_live L₀ src wk x w vs hsc hR hServe hL0 hE hc hF hsx hs hcms j r hexts hprev hRep hext hic hl
    hpiop hlo hhi

set_option maxHeartbeats 1000000 in
/-- **`FinPiop` holds**: its return is `piopCut_views_ret` at the views' source, `flat cms[j]!.cap`, `ic.root` and the
session's domain; level 0's cap is `2^cap` digests at a `--zk` schedule that caps below its depth and at most `32`
(`commit0Cut_cap_lt`), and the pads root is a digest of a tree of at least two leaves (`padsTreeCut_root_go`). -/
theorem finPiop (vs : finS.SwVals)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x) :
    FinPiop vs L₀ src wk x w := by
  unfold FinPiop
  intro ν t κ m₀ ρ₀ ops hF s0 spec hsx schs hs cms hcms j r _ _ tb hi q f hRep extras hex ic hic l hl
  refine ⟨piopCut_views_ret L₀ src wk x w hsc hR hServe hF hsx hs j r hRep hex _ _ _ _ _ hl, ?_,
    FlockComplete.padsTreeCut_root_go (FlockComplete.two_le_padsN _) hic⟩
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hj : j.val < (Masked.stsOf x).size := Masked.lt_of_jOf j
  have hjt : j.val < (FlockDeploy.tabsOf (Masked.stsOf x)).size := by simpa [FlockDeploy.tabsOf] using hj
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  obtain ⟨schs', hs', hsj⟩ := Masked.schedsPZk_live hx
  obtain rfl : schs = schs' := Except.ok.inj (hs.symm.trans hs')
  have hq : FlockComplete.quietS.R
      ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
        FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes V)
          (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
          (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t i))
      ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
        FlockComplete.commit0Cut Masked.hsE
          (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
          (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t i)) := by
    sim_walk [FlockComplete.CutEq.commit0Cut_quiet]
  have hv := hcms #[]
  rw [hq #[]] at hv
  obtain ⟨hsz, hall'⟩ := FlockSoundness.Discharge.Exec.range_mapM_ok (Prod.mk.inj hv).1
  have hjc : j.val < cms.size := by
    rw [hsz]
    exact hjt
  have hcj : (FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes V)
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.merkle (schs[j.val]!.1.levels.getD 0 default)
      (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val) : V _) = .ok cms[j.val]! := by
    rw [getElem!_pos cms _ hjc]
    exact hall' j.val hjc
  have hstj : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hj
  have hst' : Masked.stAt x j = (Masked.stsOf x)[j.val] :=
    hstj.symm.trans (getElem!_pos (Masked.stsOf x) j.val hj)
  have hmk : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    rw [Masked.tabsOf_stAt, hst']
    exact (hall j.val hj).2.2
  have hm1 : 25 ≤ (Masked.stAt x j).m := by
    rw [hst']
    exact (hall j.val hj).1
  have hm2 : (Masked.stAt x j).m ≤ 35 := by
    rw [hst']
    exact (hall j.val hj).2.1
  have hc32 := (FlockComplete.JsonRT.zkSchedule_cap0 hm1 hm2 (Masked.zkTab x j).hz).2
  have hclt := FlockComplete.JsonRT.zkSchedule_cap0_lt hm1 hm2 (Masked.zkTab x j).hz
  obtain ⟨cm, hcm, hlt⟩ := FlockComplete.commit0Cut_cap_lt _
    (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val) hclt hc32
  rw [hsj j, hmk, hcm] at hcj
  rw [← Except.ok.inj hcj]
  exact hlt

set_option maxHeartbeats 1000000 in
/-- **`FinRepErased` with no PIOP premise**: `finRepErased_of_cut` at `finPiop`, leaving the rows and `FinFc`. -/
theorem finRepErased_of_fc (vs : finS.SwVals)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hS : Loop.RepSrc L₀ src x w) (hS0 : Level0Src src x w)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs)
    (hRows : ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops →
      ∀ j : Fin (Masked.jOf x), ((finS.C x j).T (finS.cz x (finCoins x ops) j)).tab.onVals.RowsAt
        (finS.wt x w j, vs x w ν (finCoins x ops) j)
        (finS.randAt x (finCoins x ops) t j).1.1 (finS.randAt x (finCoins x ops) t j).1.2.1.2.1)
    (hFc : FinFc vs L₀ src wk x w) :
    FinRepErased vs L₀ src (realOf src) wk x w :=
  finRepErased_of_cut vs L₀ src wk x w hsc hR hServe hS hS0 hL0 hE hc hRows
    (finPiop L₀ src wk x w vs hsc hR hServe) hFc

end Loop

end Claims.ZKServer
