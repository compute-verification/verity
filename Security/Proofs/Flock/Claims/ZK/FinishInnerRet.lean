import Proofs.Flock.Claims.ZK.FinishInnerAt

/-!
# A returning cut rep, stopped at its inner proof's first round

Where the rep from a source, answered by `asksCut` from the log `l`, returns, the same rep with the history cut after
`CutEq.innerAt l.size` bodies stops past its Ligerito opening, with no pending bytes, and sends the inner proof's first
body at the source's `masked lay.fc`: its PIOP's zerocheck proof carries that `finalC` (`piopCut_fc_go`), the PIOP and
the opening run as before below the cut (`CutPrefix`), the replay and the constraints are pure, and the inner proof's
first coin is the one the cut refuses (`innerCut_first`). It reads no commitment but the rep's own.

* `repCut_inner_of_ret`: that stop.
-/

namespace Flock.Firewall.Finish

open Flock Transcript
open FlockComplete (PS)

set_option maxHeartbeats 2000000 in
/-- **A returning cut rep, at its inner proof's first round**: where the rep from `src` at level 0's commitment `cm0`
and the pads commitment `ic`, answered by `asksCut` from `l`, returns, at a schedule the rep's walk takes
(`CutEq.SchOk`), the same rep with the history cut after `CutEq.innerAt l.size` bodies stops past its Ligerito opening
with no pending bytes and sends the inner proof's first body there, at the source's `masked lay.fc`. -/
theorem repCut_inner_of_ret (coin : ℕ → ℕ → F128) (n : ℕ) (st : Setup) (sch : Flock.Schedule) (rep : ℕ)
    (dom : ByteArray) (cm0 : FlockComplete.Commit) (extras : Array Claim) (ic : FlockComplete.InnerCommit)
    (src : FlockComplete.RepSrc) (prior : Array (List F128)) (words : List ℕ) (t₀ : Transcript)
    (l : Array Claims.Msg) (hs : FlockComplete.CutEq.SchOk st sch)
    {p : Zk.ZkProof} {t : Transcript} {l' : Array Claims.Msg}
    (h : ((FlockComplete.repCut (FlockComplete.asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch rep dom cm0 extras
      ic src prior words).run t₀).go l = (.ok (p, t), l')) :
    ∃ (t₄ : Transcript) (L : Array Claims.Msg), t₄.pending = ByteArray.empty ∧
      L.size = FlockComplete.CutEq.innerAt l.size st sch extras ∧
      ((FlockComplete.repCut (FlockComplete.asksCut coin L.size) (Hashes.ofSha512 Sha512.hash) st sch rep dom cm0 extras
          ic src prior words).run t₀).go l =
        (.error "past the openings", L.push (Claims.toMsg
          (framed (FlockComplete.Replay.InZ.t₁ (src.masked (Zk.PadLayout.new st.m st.kLog sch.k0).fc) t₄) 1 1))) := by
  dsimp only [FlockComplete.repCut] at h
  obtain ⟨⟨zp, lp, ss, T, sws⟩, t₃, l₃, hPC, h⟩ := run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lg, zl, rec⟩, t₄, l₄, hL, h⟩ := run_bind_ok h
  try dsimp only at h
  obtain ⟨tr', t₅, l₅, h₅, hg⟩ := run_bind_ok h
  rw [get_run_go] at h₅
  simp only [Prod.mk.injEq, Except.ok.injEq] at h₅
  obtain ⟨⟨htr, ht₅⟩, hl₅⟩ := h₅
  subst tr' t₅ l₅
  try dsimp only at hg
  obtain ⟨⟨rp, tt⟩, t₆, l₆, h₆, hr⟩ := run_bind_ok hg
  obtain ⟨hrp, ht₆, hl₆⟩ := liftV_run_ok h₆
  subst t₆ l₆
  try dsimp only at hr
  obtain ⟨lgc, t₇, l₇, h₇, hc⟩ := run_bind_ok hr
  obtain ⟨hlgc, ht₇, hl₇⟩ := liftV_run_ok h₇
  subst t₇ l₇
  try dsimp only at hc
  obtain ⟨inner, t₈, l₈, h₈, -⟩ := run_bind_ok hc
  -- the inner proof drew its first coin, so that round is below `n`
  have hl₄n : l₄.size < n := by
    unfold FlockComplete.innerCut at h₈
    try dsimp only at h₈
    obtain ⟨_, t₉, l₉, h₉, h₈⟩ := run_bind_ok h₈
    obtain rfl := FlockComplete.SimInner.liftTM_ok_log h₉
    obtain ⟨_, t₁₀, l₁₀, h₁₀, h₈⟩ := run_bind_ok h₈
    obtain rfl := FlockComplete.SimInner.liftTM_ok_log h₁₀
    obtain ⟨_, t₁₁, l₁₁, h₁₁, -⟩ := run_bind_ok h₈
    obtain ⟨hn', -, -, -⟩ :=
      FlockComplete.SimInner.ite_ok ((FlockComplete.SimInner.sampleCut_go coin n _ _).symm.trans h₁₁)
    omega
  -- the PIOP's zerocheck proof carries the source's `finalC`
  have hPF := piopCut_fc_go _ st _ src src.masked _ _ extras prior words (by exact Nat.le_refl _)
    (fun _ _ => rfl) hPC
  have hrpF := replay_ok_fc (src.masked (Zk.PadLayout.new st.m st.kLog sch.k0).fc) hrp
  obtain ⟨rp', hrpS⟩ := hrpF
  -- the logs grow, and below the inner proof's first round the cut at it runs as the cut at `n`
  obtain ⟨bs₃, hbs₃, -⟩ := FlockComplete.CutPrefix.go_of_shift
    (FlockComplete.CutPrefix.piopCut_shift coin n st _ src _ _ extras prior words t₀) rfl hPF
  obtain ⟨bs₄, hbs₄, -⟩ := FlockComplete.CutPrefix.go_of_shift
    (FlockComplete.CutPrefix.ligeritoCut_shift coin n st.merkle sch T _ _ _ src t₃) rfl hL
  have hs₃ : l.size ≤ l₃.size := by rw [hbs₃, Array.size_append]; omega
  have hs₄ : l₃.size ≤ l₄.size := by rw [hbs₄, Array.size_append]; omega
  have hPCc : ((FlockComplete.piopCut (FlockComplete.asksCut coin l₄.size) st (Zk.PadLayout.new st.m st.kLog sch.k0)
      src (FlockComplete.flat cm0.cap) ic.root extras prior words).run t₀).go l =
      (.ok (({ zp with finalC := src.masked (Zk.PadLayout.new st.m st.kLog sch.k0).fc }, lp, ss, T, sws), t₃), l₃) :=
    FlockComplete.CutPrefix.eq_of_agree
      (FlockComplete.CutPrefix.piopCut_cut_agree coin (Nat.le_of_lt hl₄n) st _ src _ _ extras prior words t₀)
      (le_trans hs₃ hs₄) hPF hs₄
  have hLc : ((FlockComplete.ligeritoCut (FlockComplete.asksCut coin l₄.size) (Hashes.ofSha512 Sha512.hash) st.merkle
      sch T cm0 (FlockComplete.basisVals sws (st.m - 7)) (Zk.PadLayout.new st.m st.kLog sch.k0)
      src).run t₃).go l₃ = (.ok ((lg, zl, rec), t₄), l₄) :=
    FlockComplete.CutPrefix.eq_of_agree
      (FlockComplete.CutPrefix.ligeritoCut_cut_agree coin (Nat.le_of_lt hl₄n) st.merkle sch T _ _ _ src t₃)
      hs₄ hL (Nat.le_refl _)
  -- the stop's place
  obtain ⟨-, hl₃, hn₃⟩ := FlockComplete.piopCut_at hs.m hPC default
  have hl₄ := ((FlockComplete.CutEq.ligeritoCut_eqAt coin n st.merkle sch T cm0
      (FlockComplete.basisVals sws (st.m - 7)) (Zk.PadLayout.new st.m st.kLog sch.k0) src src hs.n hs.k0 hs.levels
      hs.q t₃ (l := l₃) rfl hn₃ (fun _ => rfl) (fun _ _ _ _ _ => rfl) (fun _ _ _ _ _ => rfl)
      (fun _ => ⟨rfl, rfl, rfl⟩)).2 _ _ hL).1
  refine ⟨t₄, l₄, ligeritoCut_end coin n st.merkle sch T _ _ _ src t₃ l₃ _ t₄ l₄ hL, ?_, ?_⟩
  · rw [hl₄, hl₃]
    simp only [FlockComplete.CutEq.innerAt, FlockComplete.CutEq.ligAt]
    omega
  dsimp only [FlockComplete.repCut]
  rw [FlockComplete.Replay.bind_go hPCc]
  try dsimp only
  rw [FlockComplete.Replay.bind_go hLc]
  try dsimp only
  rw [FlockComplete.Replay.bind_go (get_run_go _ _)]
  try dsimp only
  rw [FlockComplete.Replay.bind_go (liftV_run_go hrpS _ _)]
  try dsimp only
  rw [FlockComplete.Replay.bind_go (liftV_run_go hlgc _ _)]
  try dsimp only
  exact FlockComplete.LigLock.bind_go_err (innerCut_first coin _ _ _ src _ t₄ (Nat.le_refl _))

/-- **A returning cut rep's PIOP returns**: the PIOP is the rep's first step. -/
theorem piopCut_of_repCut (coin : ℕ → ℕ → F128) (n : ℕ) (st : Setup) (sch : Flock.Schedule) (rep : ℕ)
    (dom : ByteArray) (cm0 : FlockComplete.Commit) (extras : Array Claim) (ic : FlockComplete.InnerCommit)
    (src : FlockComplete.RepSrc) (prior : Array (List F128)) (words : List ℕ) (t₀ : Transcript)
    (l : Array Claims.Msg) {p : Zk.ZkProof} {t : Transcript} {l' : Array Claims.Msg}
    (h : ((FlockComplete.repCut (FlockComplete.asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch rep dom cm0 extras
      ic src prior words).run t₀).go l = (.ok (p, t), l')) :
    ∃ res t₃ l₃, ((FlockComplete.piopCut (FlockComplete.asksCut coin n) st (Zk.PadLayout.new st.m st.kLog sch.k0) src
      (FlockComplete.flat cm0.cap) ic.root extras prior words).run t₀).go l = (.ok (res, t₃), l₃) := by
  dsimp only [FlockComplete.repCut] at h
  obtain ⟨res, t₃, l₃, hPC, -⟩ := run_bind_ok h
  exact ⟨res, t₃, l₃, hPC⟩

end Flock.Firewall.Finish
