import Proofs.Flock.Claims.ZK.FinishFwRep

/-!
# Level 0's values from the firewall's run

The firewall computes level 0's opening itself: where `repFwC` returns, its Ligerito proof's out-of-domain value,
lane messages and pair's sums are `ligeritoPZk_lanes`' closed forms at the coins from the round the opening begins, on
its packed table and the basis of its PIOP's switched vectors, plus the pads. Those closed forms are `l0Clear`'s at the
same coins, table and basis (`LigLock.tabS_foldKs`, `LigLock.wS_foldKs`, and the lane coins, point and weights at the
replies, `Level0.laneCh_coins`, `Level0.lanePoint_coins`, `Level0.laneW0_coins`).

* **`Finish.repFwC_l0`**: where `repFwC` returns and its PIOP ends at round `p₀` with basis `b`, level 0's values in its
  proof are `l0Clear` at the coins from `p₀`, the packed table and `b`, plus the pads.
-/

namespace FlockComplete.LigLock

open Flock Replay

/-- **`tabS` is the iterated fold** of the table at the lane coins, lowest variable first. -/
theorem tabS_foldKs (coin : ℕ → ℕ → F128) (p0 : ℕ) (t : Array F128) :
    ∀ k, tabS coin p0 t k = Lg.foldKs ((List.range k).map (chS coin p0)) (embedF t)
  | 0 => rfl
  | k + 1 => by
    rw [tabS, tabS_foldKs coin p0 t k, List.range_succ, List.map_append]
    unfold Lg.foldKs
    rw [List.foldl_append]
    rfl

/-- **`wS` is the iterated fold** of the first weights at the lane coins. -/
theorem wS_foldKs (coin : ℕ → ℕ → F128) (p0 : ℕ) (b : Array F128) (n : ℕ) :
    ∀ k, wS coin p0 b n k = Lg.foldKs ((List.range k).map (chS coin p0)) (wS coin p0 b n 0)
  | 0 => rfl
  | k + 1 => by
    rw [show wS coin p0 b n (k + 1) = foldK (chS coin p0 k) (wS coin p0 b n k) from rfl,
      wS_foldKs coin p0 b n k, List.range_succ, List.map_append]
    unfold Lg.foldKs
    rw [List.foldl_append]
    rfl

end FlockComplete.LigLock

namespace FlockComplete.ZkTables.Level0

open Flock

variable {coin : ℕ → ℕ → F128} {resp : ℕ → Claims.Msg} (hc : ∀ k, coinOf (resp k) = coin k)
include hc

/-- The lane coins at replies whose coins are `coin`'s are `chS`'s. -/
theorem laneCh_coins (k L : ℕ) : laneCh resp k L = (List.range k).map (LigLock.chS coin L) := by
  unfold laneCh
  refine List.map_congr_left fun i _ => ?_
  unfold coinKOf
  rw [hc]
  rfl

/-- The out-of-domain point at replies whose coins are `coin`'s is `zS`'s. -/
theorem lanePoint_coins (n L : ℕ) : lanePoint resp n L = LigLock.zS coin L n := by
  unfold lanePoint
  rw [hc]
  rfl

/-- The first weights at replies whose coins are `coin`'s are `wS`'s. -/
theorem laneW0_coins (n L : ℕ) (b : Array F128) : laneW0 resp n L b = LigLock.wS coin L b n 0 := by
  unfold laneW0
  rw [lanePoint_coins hc, hc]
  rfl

end FlockComplete.ZkTables.Level0

namespace Flock.Firewall.Finish

open Flock Transcript
open FlockComplete.LigLock (l0Clear l0Clear_ood l0Clear_msg l0Clear_e tabS wS)

theorem take_range_map {α : Type} (f : ℕ → α) {j k : ℕ} (hj : j ≤ k) :
    ((List.range k).map f).take j = (List.range j).map f := by
  rw [← List.map_take, List.take_range, Nat.min_eq_left hj]

set_option maxHeartbeats 1000000 in
/-- **Level 0's values from the firewall's run**: where `repFwC`, answered by `asksCut coin n` at replies `resp` whose
coins are `coin`'s, returns, and its PIOP ends at round `p₀` having switched to the basis `b`, its proof's
out-of-domain value, lane messages and pair's sums are level 0's clear values at the coins from `p₀` on the packed
table `g` and `b` (`l0Clear`), plus the pads, at each value's layout position. -/
theorem repFwC_l0 (coin : ℕ → ℕ → F128) (n : ℕ) (resp : ℕ → Claims.Msg)
    (hc : ∀ k, FlockComplete.coinOf (resp k) = coin k) (st : Setup) (sch : Flock.Schedule) (tPad rep : ℕ)
    (dom : ByteArray) (g : Array F128) (cm0 : FlockComplete.Commit) (extras : Array Claim) (pads : FlockComplete.PadsZk)
    (ic : FlockComplete.InnerCommit) (vals : Array F128) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0) (p0 : ℕ) (b : Array F128) {t₀ : Transcript} {l : Array Claims.Msg}
    (hb : ∀ zp lp ss T sws t₃ l₃,
      ((piopFw (FlockComplete.asksCut coin n) st (Zk.PadLayout.new st.m st.kLog sch.k0) g
          (fun i => vals[i]! + pads.h[i]!) (FlockComplete.flat cm0.cap) ic.root extras).run t₀).go l =
        (.ok ((zp, lp, ss, T, sws), t₃), l₃) →
      FlockComplete.basisVals sws (st.m - 7) = b ∧ l₃.size = p0)
    {p : Zk.ZkProof} {t : Transcript} {l' : Array Claims.Msg}
    (h : ((repFwC (FlockComplete.asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch tPad rep dom g cm0 extras pads ic
      vals extra lanePad salt).run t₀).go l = (.ok (p, t), l')) :
    p.proof.ligerito.oodValues[0]! =
        l0Clear coin p0 sch g b (Zk.PadLayout.new st.m st.kLog sch.k0) extra rep
          ((Zk.PadLayout.new st.m st.kLog sch.k0).lgOod 0) +
          pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).lgOod 0]! ∧
      (∀ j < sch.k0, p.proof.ligerito.messages[j]! =
        (⟨l0Clear coin p0 sch g b (Zk.PadLayout.new st.m st.kLog sch.k0) extra rep
              ((Zk.PadLayout.new st.m st.kLog sch.k0).lgMsg j 0) +
            pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).lgMsg j 0]!,
          l0Clear coin p0 sch g b (Zk.PadLayout.new st.m st.kLog sch.k0) extra rep
              ((Zk.PadLayout.new st.m st.kLog sch.k0).lgMsg j 1) +
            pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).lgMsg j 1]!⟩,
         ⟨l0Clear coin p0 sch g b (Zk.PadLayout.new st.m st.kLog sch.k0) extra rep
              ((Zk.PadLayout.new st.m st.kLog sch.k0).lgMsg j 2) +
            pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).lgMsg j 2]!,
          l0Clear coin p0 sch g b (Zk.PadLayout.new st.m st.kLog sch.k0) extra rep
              ((Zk.PadLayout.new st.m st.kLog sch.k0).lgMsg j 3) +
            pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).lgMsg j 3]!⟩)) ∧
      (∀ i < 2, (if i = 0 then p.level0.e0 else p.level0.e1) =
        ⟨l0Clear coin p0 sch g b (Zk.PadLayout.new st.m st.kLog sch.k0) extra rep
              ((Zk.PadLayout.new st.m st.kLog sch.k0).lgE i 0) +
            pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).lgE i 0]!,
          l0Clear coin p0 sch g b (Zk.PadLayout.new st.m st.kLog sch.k0) extra rep
              ((Zk.PadLayout.new st.m st.kLog sch.k0).lgE i 1) +
            pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).lgE i 1]!⟩) := by
  dsimp only [repFwC] at h
  obtain ⟨⟨zp, zc⟩, t₁, l₁, h₁, h⟩ := run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lp, lc⟩, t₂, l₂, h₂, h⟩ := run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨ss, T, sws⟩, t₃, l₃, h₃, h⟩ := run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lg, zl, rec⟩, t₄, l₄, h₄, h⟩ := run_bind_ok h
  try dsimp only at h
  -- the PIOP ends at `p₀` with basis `b`
  have hpf : ((piopFw (FlockComplete.asksCut coin n) st (Zk.PadLayout.new st.m st.kLog sch.k0) g
      (fun i => vals[i]! + pads.h[i]!) (FlockComplete.flat cm0.cap) ic.root extras).run t₀).go l =
        (.ok ((zp, lp, ss, T, sws), t₃), l₃) := by
    unfold piopFw
    rw [FlockComplete.Replay.bind_go h₁]
    try dsimp only
    rw [FlockComplete.Replay.bind_go h₂]
    try dsimp only
    rw [FlockComplete.Replay.bind_go h₃]
    rfl
  obtain ⟨hbv, hl₃⟩ := hb _ _ _ _ _ _ _ hpf
  -- the proof's Ligerito and level 0 are the opening's
  obtain ⟨tr', t₅, l₅, h₅, hg⟩ := run_bind_ok h
  rw [get_run_go] at h₅
  simp only [Prod.mk.injEq, Except.ok.injEq] at h₅
  obtain ⟨⟨htr, ht₅⟩, hl₅⟩ := h₅
  subst tr' t₅ l₅
  try dsimp only at hg
  obtain ⟨⟨rp, tt⟩, t₆, l₆, h₆, hr₆⟩ := run_bind_ok hg
  obtain ⟨-, ht₆, hl₆⟩ := liftV_run_ok h₆
  subst t₆ l₆
  try dsimp only at hr₆
  obtain ⟨lgc, t₇, l₇, h₇, hc'⟩ := run_bind_ok hr₆
  obtain ⟨-, ht₇, hl₇⟩ := liftV_run_ok h₇
  subst t₇ l₇
  try dsimp only at hc'
  split at hc'
  · obtain ⟨_, _, _, h', _⟩ := run_bind_ok hc'
    rw [throw_run_go] at h'
    cases h'
  obtain ⟨ip, t₈, l₈, -, hp⟩ := run_bind_ok hc'
  simp only [StateT.run_pure, FlockComplete.JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at hp
  obtain ⟨⟨rfl, -⟩, -⟩ := hp
  -- the opening's lane phase in closed form, at the replies from `p₀`
  have hS4 := FlockComplete.SimInner.ligeritoPZk_ok coin n resp hc _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ t₃ l₃ _ _ h₄
  have hLF := FlockComplete.L0.ligeritoPZk_lanes resp _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hn hk0 hS4
  rw [hbv, hl₃] at hLF
  have hfT : FlockComplete.Replay.Lg.foldKs (FlockComplete.ZkTables.Level0.laneCh resp sch.k0 p0)
      (FlockComplete.embedF g) = tabS coin p0 g sch.k0 := by
    rw [FlockComplete.ZkTables.Level0.laneCh_coins hc, FlockComplete.LigLock.tabS_foldKs]
  have hfW : FlockComplete.Replay.Lg.foldKs (FlockComplete.ZkTables.Level0.laneCh resp sch.k0 p0)
      (FlockComplete.ZkTables.Level0.laneW0 resp sch.n p0 b) = wS coin p0 b sch.n sch.k0 := by
    rw [FlockComplete.ZkTables.Level0.laneCh_coins hc, FlockComplete.ZkTables.Level0.laneW0_coins hc]
    exact (FlockComplete.LigLock.wS_foldKs coin p0 b sch.n sch.k0).symm
  refine ⟨?_, fun j hj => ?_, fun i hi => ?_⟩
  · refine hLF.ood.trans ?_
    rw [l0Clear_ood, FlockComplete.ZkTables.Level0.lanePoint_coins hc]
  · refine (hLF.msgs j hj).trans ?_
    have hj' : j ≤ sch.k0 := Nat.le_of_lt hj
    rw [FlockComplete.ZkTables.Level0.laneCh_coins hc, take_range_map _ hj',
      FlockComplete.ZkTables.Level0.laneW0_coins hc, ← FlockComplete.LigLock.tabS_foldKs,
      ← FlockComplete.LigLock.wS_foldKs,
      l0Clear_msg coin p0 sch g b (Zk.PadLayout.new st.m st.kLog sch.k0) extra rep rfl (c := 0) hj (by norm_num),
      l0Clear_msg coin p0 sch g b (Zk.PadLayout.new st.m st.kLog sch.k0) extra rep rfl (c := 1) hj (by norm_num),
      l0Clear_msg coin p0 sch g b (Zk.PadLayout.new st.m st.kLog sch.k0) extra rep rfl (c := 2) hj (by norm_num),
      l0Clear_msg coin p0 sch g b (Zk.PadLayout.new st.m st.kLog sch.k0) extra rep rfl (c := 3) hj (by norm_num)]
    rfl
  · rw [l0Clear_e coin p0 sch g b (Zk.PadLayout.new st.m st.kLog sch.k0) extra rep rfl (i := i) (c := 0) (by norm_num),
      l0Clear_e coin p0 sch g b (Zk.PadLayout.new st.m st.kLog sch.k0) extra rep rfl (i := i) (c := 1) (by norm_num)]
    interval_cases i
    · show zl.e0 = _
      rw [hLF.e0, hfT, hfW]
      rfl
    · show zl.e1 = _
      rw [hLF.e1, hfT, hfW]
      rfl

end Flock.Firewall.Finish
