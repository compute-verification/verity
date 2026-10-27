import Proofs.Flock.Claims.ZK.RepReplay
import Proofs.Flock.Claims.ZK.RepInner
import Proofs.Flock.Claims.ZK.L0PinCut
import Proofs.Flock.Claims.ZK.FinishTau
import Proofs.Flock.Claims.Complete.Zk.Lanes

/-!
# The cut rep returns where the firewall's rep does

Answered by `asksCut`, the cut's rep from any source runs the firewall's phases at other values: its PIOP returns
where the firewall's phases did and its rank check passes (`piopCut_of_phases`), its Ligerito opening where the
firewall's did, with as many messages, lane challenges and out-of-domain values (`ligeritoPZk_cut_fw`), its replay
where the fold at the firewall's coins returned (`replay_fold_coins`, `replay_cut_ok`), its lane constraints at the
record's sizes, and its inner proof where the firewall's did (`innerP_cut_fw`). No step reads the source's values.

* **`ligeritoConstraints_sizes`**: a returned lane phase read a record of the layout's shape.
* **`repCut_of_repFwC`**: where the firewall's rep returned from the empty transcript, the cut's rep from any source,
  level-0 commitment, pads commitment and extra claims of the same count returns from a log of the same size, with a
  log of the same size, wherever its rank check passes.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockComplete.CutFree

open Flock Transcript Replay

/-- **A returned lane phase read a record of the layout's shape.** -/
theorem ligeritoConstraints_sizes {lay : Zk.PadLayout} {rec : Zk.L0Rec} {lg : LigeritoProof} {msgs : Zk.Level0Msgs}
    {v : Array (String × Zk.Lin)} (h : Zk.ligeritoConstraints lay rec lg msgs = .ok v) :
    rec.ood.size = lay.nOod0 ∧ rec.challenges.size = lay.k0 ∧ lay.k0 ≤ lg.messages.size := by
  unfold Zk.ligeritoConstraints at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hc
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not, decide_eq_true_eq, not_lt] at hc
    exact ⟨hc.1.1, hc.1.2, hc.2⟩

set_option maxHeartbeats 4000000 in
/-- **The cut rep returns where the firewall's did**: answered by `asksCut` at replies with its coins, where the
firewall's rep at SHA-512 returned from the empty transcript and `l`, the cut's rep at SHA-512 from any source,
level-0 commitment, pads commitment, domain and extra claims of the same count returns from the empty transcript and a
log of `l`'s size, wherever its rank check passes, with a log of the firewall's size. -/
theorem repCut_of_repFwC (coin : ℕ → ℕ → F128) (n : ℕ) {resp : ℕ → Claims.Msg} (hr : ∀ k, coinOf (resp k) = coin k)
    {st : Setup} (hm : 14 ≤ st.m) (hkk : st.kLog = st.fold.kLog) {sch : Schedule} {tPad rep : ℕ} {dom : ByteArray}
    {g : Array F128} {cm0 : Commit} {extras : Array Claim} {pads : PadsZk} {ic : InnerCommit} {vals : Array F128}
    {extra lanePad : ℕ → ℕ → F128} {salt : ℕ → ℕ → ByteArray} {l : Array Claims.Msg} {p : Zk.ZkProof}
    {t : Transcript} {l' : Array Claims.Msg}
    (h : ((_root_.Flock.Firewall.Finish.repFwC (asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch tPad rep dom g
      cm0 extras pads ic vals extra lanePad salt).run (Transcript.init #[] dom)).go l = (.ok (p, t), l'))
    (h' : ByteArray → ByteArray) (rep' : ℕ) (dom' : ByteArray) (cm0' : Commit) {extras' : Array Claim}
    (hes : extras'.size = extras.size) (ic' : InnerCommit) (src : RepSrc) (prior : Array (List F128))
    (words : List ℕ) {L : Array Claims.Msg} (hLs : L.size = l.size)
    (hrk : ∀ zp zc t₁ l₁ lp lc t₂ l₂,
      ((zerocheckCut (asksCut coin n) st.m st.digest (flat cm0'.cap) ic'.root (Zk.PadLayout.new st.m st.kLog sch.k0)
          src.masked).run (Transcript.init #[] dom')).go L = (.ok ((zp, zc), t₁), l₁) →
        ((lincheckCut (asksCut coin n) st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0) src.masked zc).run
          t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) →
        MaskRank.maskRankOk (prior ++ MaskRank.roundPoints st.m st.kLog t₂.rounds) words = true) :
    ∃ q t₅ L₅, ((repCut (asksCut coin n) (Hashes.ofSha512 h') st sch rep' dom' cm0' extras' ic' src prior words).run
      (Transcript.init #[] dom')).go L = (.ok (q, t₅), L₅) ∧ L₅.size = l'.size := by
  -- the firewall's rep, in phases
  dsimp only [_root_.Flock.Firewall.Finish.repFwC] at h
  obtain ⟨⟨zp, zc⟩, t₁, l₁, h₁, h⟩ := run_ok h
  try dsimp only at h
  obtain ⟨⟨lp, lc⟩, t₂, l₂, h₂, h⟩ := run_ok h
  try dsimp only at h
  obtain ⟨⟨ss, T, sws⟩, t₃, l₃, h₃, h⟩ := run_ok h
  try dsimp only at h
  obtain ⟨⟨lg, zl, rec⟩, t₄, l₄, h₄, h⟩ := run_ok h
  try dsimp only at h
  obtain ⟨tr', t₅, l₅, h₅, hg⟩ := run_ok h
  rw [_root_.Flock.Firewall.Finish.get_run_go] at h₅
  simp only [Prod.mk.injEq, Except.ok.injEq] at h₅
  obtain ⟨⟨htr, ht₅⟩, hl₅⟩ := h₅
  subst tr' t₅ l₅
  try dsimp only at hg
  obtain ⟨⟨rp, tt⟩, t₆, l₆, h₆, hg⟩ := run_ok hg
  obtain ⟨hrp, ht₆, hl₆⟩ := _root_.Flock.Firewall.Finish.liftV_run_ok h₆
  subst t₆ l₆
  try dsimp only at hg
  obtain ⟨lgc, t₇, l₇, h₇, hg⟩ := run_ok hg
  obtain ⟨hlgc, ht₇, hl₇⟩ := _root_.Flock.Firewall.Finish.liftV_run_ok h₇
  subst t₇ l₇
  try dsimp only at hg
  split at hg
  · obtain ⟨_, _, _, h', _⟩ := run_ok hg
    rw [_root_.Flock.Firewall.Finish.throw_run_go] at h'
    cases h'
  obtain ⟨inner, t₈, l₈, h₈, h₉⟩ := run_ok hg
  rw [pure_go] at h₉
  simp only [Prod.mk.injEq, Except.ok.injEq] at h₉
  obtain ⟨-, rfl⟩ := h₉
  -- the firewall's rounds are answered by the coins from `l.size`
  have hI0 : RankAt.Inv coin l.size (Transcript.init #[] dom) l :=
    ⟨fun j hj => by simp [Transcript.init] at hj, by simp [Transcript.init]⟩
  have hI1 := ((RankAt.zerocheckCut_grows coin n l.size _ _ _ _ _ _) _ _ _ _ _ h₁).2 hI0
  have hI2 := ((RankAt.lincheckCut_grows coin n l.size _ _ _ _ _) _ _ _ _ _ h₂).2 hI1
  have hI3 := ((RankAt.ringSwitchCut_grows coin n l.size _ _) _ _ _ _ _ h₃).2 hI2
  have hI4 := ((RankAt.ligeritoPZk_grows coin n l.size _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) _ _ _ _ _ h₄).2 hI3
  have hL := replay_fold_coins coin hI4.1 hm hkk _ _ _ _ hrp
  rw [← hLs] at hL
  -- the cut's PIOP
  obtain ⟨⟨zp', lp', ss', T', sws'⟩, t₃', l₃', hP', hsz₃⟩ := piopCut_of_phases hm h₁ h₂ h₃
    (Zk.PadLayout.new st.m st.kLog sch.k0) src (flat cm0'.cap) ic'.root hes prior words (Transcript.init #[] dom')
    hLs hrk
  obtain ⟨zc', t₁', l₁', lc', t₂', l₂', h₁', h₂', -, h₃'⟩ := piopCut_split_rank hP'
  -- the cut's Ligerito opening
  obtain ⟨⟨lg', zl', rec'⟩, t₄', l₄', h₄', hsz₄, hmsg, hch, hood⟩ :=
    ligeritoPZk_cut_fw coin n Sha512.hash h' _ st.merkle sch _ T' _ cm0' _ _ (basisVals sws' (st.m - 7)) _
      (Zk.PadLayout.new st.m st.kLog sch.k0) _ _ _ _ _ _ src _ t₃' _ l₃' hsz₃ _ _ _ h₄
  -- the cut's replay
  have hR : Replay.Prefix t₂'.rounds t₄'.rounds :=
    (RankAt.prefix_of_grows coin 0 (RankAt.ringSwitchCut_grows coin n 0 _ _) h₃').trans
      (RankAt.prefix_of_grows coin 0 (RankAt.ligeritoCut_grows coin n 0 _ _ _ _ _ _ _ _) h₄')
  obtain ⟨⟨rp', tt'⟩, hq⟩ := replay_cut_ok coin n hr hm hkk (lay := Zk.PadLayout.new st.m st.kLog sch.k0)
    (Nat.le_refl _) (by show st.fold.kLog - 6 ≤ st.kLog - 6; rw [hkk]) src.masked
    { cap := cm0'.cap, params := pcsParams st.merkle sch, zerocheck := zp', lincheck := lp', ringSwitches := ss',
      ligerito := lg' } ic'.root dom' h₁' h₂' hR hL
  -- the cut's lane constraints
  obtain ⟨ho, hc, hk0⟩ := ligeritoConstraints_sizes hlgc
  obtain ⟨lgc', hlgc'⟩ := ligeritoConstraints_ok (lay := Zk.PadLayout.new st.m st.kLog sch.k0) (msgs := zl')
    (by rw [← hood]; exact ho) (by rw [← hch]; exact hc) (by rw [← hmsg]; exact hk0)
  -- the cut's inner proof
  obtain ⟨inner', t₈', l₈', h₈', hsz₈, -⟩ := innerP_cut_fw coin n Sha512.hash _ pads ic ic' _ zp'.finalC src _
    (rp'.cons ++ #[("ring switch ab", rp'.abValue.addConst (Zk.claimCheck rp'.claims[0]! ss'[0]!)),
      ("ring switch c", rp'.cValue.addConst (Zk.claimCheck rp'.claims[1]! ss'[1]!))] ++ lgc')
    (t' := t₄') hsz₄ _ _ _ h₈
  unfold repCut
  try dsimp only
  rw [bind_go hP']
  try dsimp only
  rw [bind_go h₄']
  try dsimp only
  rw [bind_go (_root_.Flock.Firewall.Finish.get_run_go _ _)]
  try dsimp only
  rw [bind_go (_root_.Flock.Firewall.Finish.liftV_run_go hq _ _)]
  try dsimp only
  rw [bind_go (_root_.Flock.Firewall.Finish.liftV_run_go hlgc' _ _)]
  try dsimp only
  rw [bind_go h₈']
  exact ⟨_, _, _, pure_go _ _ _, hsz₈⟩

end FlockComplete.CutFree
