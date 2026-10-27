import Proofs.Flock.Claims.ZK.FinishSrc
import Proofs.Flock.Claims.ZK.CutPrefix

/-!
# The rep from a source that misses the firewall at `final_c`

`repCut_fw` takes a source whose masked values below `zp 64` are the firewall's. Here the source may differ at
`final_c`'s pad (`lay.fc`): the PIOP reads `mk lay.fc` only into the zerocheck proof it returns (`zerocheckCut_fc`), the
replay reads that proof's `finalC` only into its constraints (`replay_ok_fc`), and Ligerito and the constraints do not
read it. So where the firewall's rep returns, the rep from the source runs through its PIOP, Ligerito opening, replay and
constraints with the firewall's coins and logs, and its first inner body, `label LABEL_INNER ‖ absorbF finalC`, carries
the source's `masked lay.fc` (`repCut_fw_inner`).

* `zerocheckCut_fc`, `lincheckCut_fc`, `piopCut_fc_go`: the PIOP at masked values that differ at `fc` alone.
* `replay_fc`, `replay_ok_fc`: the replay at zerocheck proofs that differ in `finalC` alone.
* `innerCut_first`: the inner proof stopped at its first round sends `InZ.t₁`'s frame.
* `repCut_fw_inner`: the rep from the source stopped at the inner proof's first round.
-/

namespace Flock.Firewall.Finish

open Flock Transcript
open FlockComplete (PS)

theorem bind_congr₂ {m : Type → Type} [Monad m] {α β : Type} {x y : m α} {f g : α → m β} (hxy : x = y)
    (h : ∀ a, f a = g a) : x >>= f = y >>= g := by
  subst hxy
  exact bind_congr h

/-! ## The PIOP -/

set_option maxHeartbeats 1000000 in
/-- **The zerocheck reads `mk lay.fc` only into its proof's `finalC`.** -/
theorem zerocheckCut_fc (io : FlockComplete.Asks PS) (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout)
    {mk mk' : ℕ → F128} (hm : m - 6 ≤ lay.nMlv) (hne : ∀ i, i ≠ lay.fc → mk' i = mk i) :
    FlockComplete.zerocheckCut io m digest cap root lay mk' =
      (fun r => ({ r.1 with finalC := mk' lay.fc }, r.2)) <$>
        FlockComplete.zerocheckCut io m digest cap root lay mk := by
  have e1 : ((Array.range 64).map fun i => mk' (lay.r1ab i)) = (Array.range 64).map fun i => mk (lay.r1ab i) :=
    Array.map_congr_left fun i hi => hne _ (by
      have := Array.mem_range.1 hi
      simp only [Zk.PadLayout.r1ab, Zk.PadLayout.fc, Zk.PadLayout.fa]
      omega)
  have e2 : ((Array.range 64).map fun i => mk' (lay.r1c i)) = (Array.range 64).map fun i => mk (lay.r1c i) :=
    Array.map_congr_left fun i hi => hne _ (by
      have := Array.mem_range.1 hi
      simp only [Zk.PadLayout.r1c, Zk.PadLayout.fc, Zk.PadLayout.fa]
      omega)
  have hfa : mk' lay.fa = mk lay.fa := hne _ (by simp only [Zk.PadLayout.fc]; omega)
  have hfb : mk' lay.fb = mk lay.fb := hne _ (by simp only [Zk.PadLayout.fc, Zk.PadLayout.fb]; omega)
  unfold FlockComplete.zerocheckCut
  simp only [map_bind, map_pure, e1, e2, hfa, hfb]
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  refine bind_congr fun _ => ?_
  refine bind_congr₂ (forIn_range_congr _ _ fun i hi b => ?_) fun _ => ?_
  · try dsimp only
    rw [hne (lay.ml i 0) (by simp only [Zk.PadLayout.ml, Zk.PadLayout.fc, Zk.PadLayout.fa]; omega),
      hne (lay.ml i 1) (by simp only [Zk.PadLayout.ml, Zk.PadLayout.fc, Zk.PadLayout.fa]; omega)]
  · rfl

/-- **The lincheck does not read `mk lay.fc`.** -/
theorem lincheckCut_fc (io : FlockComplete.Asks PS) (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout)
    {mk mk' : ℕ → F128} (zc : ZcOut) (hne : ∀ i, i ≠ lay.fc → mk' i = mk i) :
    FlockComplete.lincheckCut io c m lay mk' zc = FlockComplete.lincheckCut io c m lay mk zc := by
  have hlc : ∀ j t, mk' (lay.lc j t) = mk (lay.lc j t) := fun j t =>
    hne _ (by simp only [Zk.PadLayout.lc]; omega)
  have hzp : ∀ i, mk' (lay.zp i) = mk (lay.zp i) := fun i =>
    hne _ (by simp only [Zk.PadLayout.zp, Zk.PadLayout.lc]; omega)
  unfold FlockComplete.lincheckCut
  simp only [hlc, hzp]

theorem map_run_go {α β : Type} (f : α → β) {x : TMW PS α} {t t' : Transcript} {l l' : Array Claims.Msg} {a : α}
    (h : (x.run t).go l = (.ok (a, t'), l')) : ((f <$> x).run t).go l = (.ok (f a, t'), l') := by
  rw [← bind_pure_comp, FlockComplete.Replay.bind_go h]
  rfl

/-- **The PIOP from a source that differs at `fc` alone**: where the PIOP from `{src with masked := mk}` returns, the
PIOP from `src` returns the same, its zerocheck proof's `finalC` the source's `masked lay.fc`. -/
theorem piopCut_fc_go (io : FlockComplete.Asks PS) (st : Setup) (lay : Zk.PadLayout) (src : FlockComplete.RepSrc)
    (mk : ℕ → F128) (cap root : ByteArray) (extras : Array Claim) (prior : Array (List F128)) (words : List ℕ)
    (hm : st.m - 6 ≤ lay.nMlv) (hne : ∀ i, i ≠ lay.fc → src.masked i = mk i)
    {zp : ZerocheckProof} {lp : LincheckProof} {ss : Array (Array F128)} {T : F128} {sws : Array Switched}
    {t₀ t : Transcript} {l l' : Array Claims.Msg}
    (h : ((FlockComplete.piopCut io st lay { src with masked := mk } cap root extras prior words).run t₀).go l =
      (.ok ((zp, lp, ss, T, sws), t), l')) :
    ((FlockComplete.piopCut io st lay src cap root extras prior words).run t₀).go l =
      (.ok (({ zp with finalC := src.masked lay.fc }, lp, ss, T, sws), t), l') := by
  unfold FlockComplete.piopCut at h ⊢
  obtain ⟨⟨zp₁, zc⟩, t₁, l₁, h₁, h⟩ := run_bind_ok h
  try dsimp only at h h₁
  obtain ⟨⟨lp₁, lc⟩, t₂, l₂, h₂, h⟩ := run_bind_ok h
  try dsimp only at h h₂
  obtain ⟨_, t₃, l₃, h₃, h⟩ := run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨ss₁, T₁, sws₁⟩, t₄, l₄, h₄, h⟩ := run_bind_ok h
  try dsimp only at h
  rw [FlockComplete.Replay.pure_go] at h
  simp only [Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨⟨⟨rfl, rfl, rfl, rfl, rfl⟩, rfl⟩, rfl⟩ := h
  rw [zerocheckCut_fc io st.m st.digest cap root lay hm hne,
    FlockComplete.Replay.bind_go (map_run_go _ h₁)]
  try dsimp only
  rw [lincheckCut_fc io st.fold st.m lay zc hne, FlockComplete.Replay.bind_go h₂]
  try dsimp only
  rw [FlockComplete.Replay.bind_go h₃]
  try dsimp only
  rw [FlockComplete.Replay.bind_go h₄]
  rfl

/-! ## The replay -/

theorem map_unit_pure {α : Type} (a b : α) :
    ((fun _ => ()) <$> (pure a : TM α)) = (fun _ => ()) <$> (pure b : TM α) := by
  simp only [map_pure]

set_option maxHeartbeats 2000000 in
/-- **The replay reads the zerocheck proof's `finalC` only into what it returns.** -/
theorem replay_fc (st : Setup) (lay : Zk.PadLayout) (cap : Array ByteArray) (params : PcsParams)
    (zp : ZerocheckProof) (lp : LincheckProof) (ss : Array (Array F128)) (lg : LigeritoProof) (root : ByteArray)
    (v : F128) :
    (fun _ => ()) <$> Zk.replay st lay
        { cap, params, zerocheck := { zp with finalC := v }, lincheck := lp, ringSwitches := ss, ligerito := lg }
        root =
      (fun _ => ()) <$> Zk.replay st lay
        { cap, params, zerocheck := zp, lincheck := lp, ringSwitches := ss, ligerito := lg } root := by
  unfold Zk.replay
  simp only [map_bind]
  all_goals repeat (refine bind_congr fun _ => ?_)
  all_goals
    split
    all_goals
      try simp only [map_bind]
      repeat (first
        | exact map_unit_pure _ _
        | refine bind_congr fun _ => ?_)

/-- **The replay returns at zerocheck proofs that differ in `finalC` alone**, at the same transcript. -/
theorem replay_ok_fc {st : Setup} {lay : Zk.PadLayout} {cap : Array ByteArray} {params : PcsParams}
    {zp : ZerocheckProof} {lp : LincheckProof} {ss : Array (Array F128)} {lg : LigeritoProof} {root : ByteArray}
    (v : F128) {t tt : Transcript} {rp : Zk.Replayed}
    (h : (Zk.replay st lay
        { cap, params, zerocheck := zp, lincheck := lp, ringSwitches := ss, ligerito := lg } root).run t =
      .ok (rp, tt)) :
    ∃ rp', (Zk.replay st lay
        { cap, params, zerocheck := { zp with finalC := v }, lincheck := lp, ringSwitches := ss, ligerito := lg }
        root).run t = .ok (rp', tt) := by
  have e2 := congrArg (fun z => StateT.run z t) (replay_fc st lay cap params zp lp ss lg root v)
  simp only [StateT.run_map, h] at e2
  revert e2
  cases (Zk.replay st lay
      { cap, params, zerocheck := { zp with finalC := v }, lincheck := lp, ringSwitches := ss, ligerito := lg }
      root).run t with
  | error _ => intro e2; cases e2
  | ok q =>
    intro e2
    obtain ⟨a', t'⟩ := q
    cases e2
    exact ⟨a', rfl⟩

/-! ## The inner proof's first round -/

/-- **The inner proof stopped at its first round** sends that round's frame, at `InZ.t₁`: the transcript after
`label LABEL_INNER ‖ absorbF finalC`. -/
theorem innerCut_first (coin : ℕ → ℕ → F128) {n : ℕ} (lay : Zk.PadLayout) (ic : FlockComplete.InnerCommit)
    (finalC : F128) (src : FlockComplete.RepSrc) (cons : Array (String × Zk.Lin)) (t : Transcript)
    {l : Array Claims.Msg} (hn : n ≤ l.size) :
    ((FlockComplete.innerCut (FlockComplete.asksCut coin n) lay ic finalC src cons).run t).go l =
      (.error "past the openings",
        l.push (Claims.toMsg (framed (FlockComplete.Replay.InZ.t₁ finalC t) 1 1))) := by
  unfold FlockComplete.innerCut
  try dsimp only
  rw [FlockComplete.Replay.bind_go (FlockComplete.Replay.liftTM_ok (x := label Zk.LABEL_INNER) (t := t)
      (t' := ⟨t.rounds, t.k, FlockComplete.Replay.pLabel Zk.LABEL_INNER t.pending⟩) (a := ()) rfl l)]
  rw [FlockComplete.Replay.bind_go (FlockComplete.Replay.liftTM_ok (x := absorbF finalC)
      (t := ⟨t.rounds, t.k, FlockComplete.Replay.pLabel Zk.LABEL_INNER t.pending⟩)
      (t' := FlockComplete.Replay.InZ.t₁ finalC t) (a := ()) rfl l)]
  have hs := FlockComplete.SimInner.sampleCut_go coin n (FlockComplete.Replay.InZ.t₁ finalC t) l
  rw [if_pos hn] at hs
  exact FlockComplete.LigLock.bind_go_err hs

/-! ## The rep -/

set_option maxHeartbeats 2000000 in
/-- **The rep from a source that misses the firewall at `fc` reaches its inner proof**: answered by `asksCut` from the
log `l`, where the firewall's rep returns, a source that holds what the firewall computes below `zp 64` but at `fc`
(`hmk`), with `repCut_fw`'s `hpiop` and `hlig` and no `hinner`, runs its PIOP (the firewall's, its zerocheck proof's
`finalC` the source's `masked lay.fc`) and its Ligerito opening at the firewall's logs, with the inner proof's first
round below `n`; stopped there, the rep from the source sends the inner proof's first frame, at the source's
`masked lay.fc`. -/
theorem repCut_fw_inner (coin : ℕ → ℕ → F128) (n : ℕ) (st : Setup) (hm : st.merkle = MerkleScheme.hm96Sha512)
    (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (g : Array F128) (cm0 : FlockComplete.Commit)
    (extras : Array Claim) (pads : FlockComplete.PadsZk) (ic : FlockComplete.InnerCommit) (vals : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (src : FlockComplete.RepSrc)
    (prior : Array (List F128)) (words : List ℕ) (t₀ : Transcript) (l : Array Claims.Msg)
    (hm14 : 14 ≤ st.m) (hn : l.size ≤ n) (hkL : st.fold.kLog = st.kLog)
    (hmk : ∀ i < (Zk.PadLayout.new st.m st.kLog sch.k0).zp 64, i ≠ (Zk.PadLayout.new st.m st.kLog sch.k0).fc →
      src.masked i = vals[i]! + pads.h[i]!)
    (hpiop : ∀ zp zc t₁ l₁ lp lc t₂ l₂,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest (FlockComplete.flat cm0.cap) ic.root
          (Zk.PadLayout.new st.m st.kLog sch.k0) fun i => vals[i]! + pads.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0)
          (fun i => vals[i]! + pads.h[i]!) zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) →
      FlockComplete.MaskRank.maskRankOk (prior ++ FlockComplete.MaskRank.roundPoints st.m st.kLog t₂.rounds) words =
          true ∧
        ∀ k < (abcClaims st.m st.kLog zc lc ++ extras).size,
          src.sv k = switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
    (hlig : ∀ zp lp ss T sws t₃ l₃,
      ((piopFw (FlockComplete.asksCut coin n) st (Zk.PadLayout.new st.m st.kLog sch.k0) g
          (fun i => vals[i]! + pads.h[i]!) (FlockComplete.flat cm0.cap) ic.root extras).run t₀).go l =
        (.ok ((zp, lp, ss, T, sws), t₃), l₃) →
      FlockComplete.LigLock.LigHyp coin n l₃.size st.merkle sch g (FlockComplete.basisVals sws (st.m - 7))
        (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h rep tPad extra lanePad salt src)
    {p : Zk.ZkProof} {t : Transcript} {l' : Array Claims.Msg}
    (h : ((repFwC (FlockComplete.asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch tPad rep dom g cm0 extras pads ic
      vals extra lanePad salt).run t₀).go l = (.ok (p, t), l')) :
    ∃ (zp : ZerocheckProof) (lp : LincheckProof) (ss : Array (Array F128)) (T : F128) (sws : Array Switched)
      (t₃ : Transcript) (l₃ : Array Claims.Msg) (lg : LigeritoProof) (zl : Zk.Level0Msgs) (rec : Zk.L0Rec)
      (t₄ : Transcript) (l₄ : Array Claims.Msg),
      ((FlockComplete.piopCut (FlockComplete.asksCut coin n) st (Zk.PadLayout.new st.m st.kLog sch.k0) src
          (FlockComplete.flat cm0.cap) ic.root extras prior words).run t₀).go l =
        (.ok (({ zp with finalC := src.masked (Zk.PadLayout.new st.m st.kLog sch.k0).fc }, lp, ss, T, sws), t₃),
          l₃) ∧
      ((FlockComplete.ligeritoCut (FlockComplete.asksCut coin n) (Hashes.ofSha512 Sha512.hash) st.merkle sch T
          (FlockComplete.eraseC cm0) (FlockComplete.basisVals sws (st.m - 7)) (Zk.PadLayout.new st.m st.kLog sch.k0)
          src).run t₃).go l₃ = (.ok ((lg, zl, rec), t₄), l₄) ∧
      l₄.size < n ∧
      ((FlockComplete.repCut (FlockComplete.asksCut coin l₄.size) (Hashes.ofSha512 Sha512.hash) st sch rep dom
          (FlockComplete.eraseC cm0) extras ic src prior words).run t₀).go l =
        (.error "past the openings", l₄.push (Claims.toMsg
          (framed (FlockComplete.Replay.InZ.t₁ (src.masked (Zk.PadLayout.new st.m st.kLog sch.k0).fc) t₄) 1 1))) := by
  dsimp only [repFwC] at h
  obtain ⟨⟨zp, zc⟩, t₁, l₁, h₁, h⟩ := run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lp, lc⟩, t₂, l₂, h₂, h⟩ := run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨ss, T, sws⟩, t₃, l₃, h₃, h⟩ := run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lg, zl, rec⟩, t₄, l₄, h₄, h⟩ := run_bind_ok h
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
  split at hc
  · obtain ⟨_, _, _, h', _⟩ := run_bind_ok hc
    rw [throw_run_go] at h'
    cases h'
  obtain ⟨inner, t₈, l₈, h₈, -⟩ := run_bind_ok hc
  -- the firewall's inner proof drew its first coin, so that round is below `n`
  have hl₄n : l₄.size < n := by
    unfold FlockComplete.innerP at h₈
    try dsimp only at h₈
    obtain ⟨_, t₉, l₉, h₉, h₈⟩ := run_bind_ok h₈
    obtain rfl := FlockComplete.SimInner.liftTM_ok_log h₉
    obtain ⟨_, t₁₀, l₁₀, h₁₀, h₈⟩ := run_bind_ok h₈
    obtain rfl := FlockComplete.SimInner.liftTM_ok_log h₁₀
    obtain ⟨_, t₁₁, l₁₁, h₁₁, -⟩ := run_bind_ok h₈
    obtain ⟨hn', -, -, -⟩ :=
      FlockComplete.SimInner.ite_ok ((FlockComplete.SimInner.sampleCut_go coin n _ _).symm.trans h₁₁)
    omega
  have hP : ((piopFw (FlockComplete.asksCut coin n) st (Zk.PadLayout.new st.m st.kLog sch.k0) g
      (fun i => vals[i]! + pads.h[i]!) (FlockComplete.flat cm0.cap) ic.root extras).run t₀).go l =
      (.ok ((zp, lp, ss, T, sws), t₃), l₃) := by
    unfold piopFw
    rw [FlockComplete.Replay.bind_go h₁]
    try dsimp only
    rw [FlockComplete.Replay.bind_go h₂]
    try dsimp only
    rw [FlockComplete.Replay.bind_go h₃]
    rfl
  -- the PIOP: the firewall's at the source made the firewall's at `fc`, then the source's
  have hPC' := piopCut_fw coin n st _ g _
    { src with masked := fun i => if i = (Zk.PadLayout.new st.m st.kLog sch.k0).fc then
        vals[(Zk.PadLayout.new st.m st.kLog sch.k0).fc]! + pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).fc]!
      else src.masked i }
    _ _ extras prior words hm14 (by exact Nat.le_refl _) (by rw [hkL]; exact Nat.le_refl _)
    (fun i hi => by
      try dsimp only
      split_ifs with he
      · rw [he]
      · exact hmk i hi he)
    hn hpiop hP
  have hPC := piopCut_fc_go _ st _ src _ _ _ extras prior words (by exact Nat.le_refl _)
    (fun i hi => by simp only [if_neg hi]) hPC'
  -- Ligerito: the source's opening is the firewall's
  have h₄' := h₄
  rw [← hm] at h₄'
  have hL : ((FlockComplete.ligeritoCut (FlockComplete.asksCut coin n) (Hashes.ofSha512 Sha512.hash) st.merkle sch T
      (FlockComplete.eraseC cm0) (FlockComplete.basisVals sws (st.m - 7)) (Zk.PadLayout.new st.m st.kLog sch.k0)
      src).run t₃).go l₃ = (.ok ((FlockComplete.eraseL lg, zl, rec), t₄), l₄) := by
    rw [FlockComplete.LigLock.ligeritoCut_alike coin n _ st.merkle sch T cm0 g _ _ pads.h rep tPad extra lanePad salt
      src t₃ l₃ (hlig zp lp ss T sws t₃ l₃ hP), FlockComplete.Replay.bind_go h₄']
    rfl
  -- the replay and the constraints
  have hrp' : (Zk.replay st (Zk.PadLayout.new st.m st.kLog sch.k0)
      { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp, ringSwitches := ss,
        ligerito := FlockComplete.eraseL lg } ic.root).run (Transcript.init t₄.rounds dom) = .ok (rp, tt) := by
    rw [replay_eraseL']
    exact hrp
  obtain ⟨rp', hrpS⟩ := replay_ok_fc (src.masked (Zk.PadLayout.new st.m st.kLog sch.k0).fc) hrp'
  have hlgc' : Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rec (FlockComplete.eraseL lg) zl =
      .ok lgc := by
    rw [ligeritoConstraints_eraseL']
    exact hlgc
  -- the logs grow, and below the inner proof's first round the cut at it runs as the cut at `n`
  obtain ⟨bs₃, hbs₃, -⟩ := FlockComplete.CutPrefix.go_of_shift
    (FlockComplete.CutPrefix.piopCut_shift coin n st _ src _ _ extras prior words t₀) rfl hPC
  obtain ⟨bs₄, hbs₄, -⟩ := FlockComplete.CutPrefix.go_of_shift
    (FlockComplete.CutPrefix.ligeritoCut_shift coin n st.merkle sch T _ _ _ src t₃) rfl hL
  have hs₃ : l.size ≤ l₃.size := by rw [hbs₃, Array.size_append]; omega
  have hs₄ : l₃.size ≤ l₄.size := by rw [hbs₄, Array.size_append]; omega
  have hPCc : ((FlockComplete.piopCut (FlockComplete.asksCut coin l₄.size) st (Zk.PadLayout.new st.m st.kLog sch.k0)
      src (FlockComplete.flat cm0.cap) ic.root extras prior words).run t₀).go l =
      (.ok (({ zp with finalC := src.masked (Zk.PadLayout.new st.m st.kLog sch.k0).fc }, lp, ss, T, sws), t₃), l₃) :=
    FlockComplete.CutPrefix.eq_of_agree
      (FlockComplete.CutPrefix.piopCut_cut_agree coin (Nat.le_of_lt hl₄n) st _ src _ _ extras prior words t₀)
      (le_trans hs₃ hs₄) hPC hs₄
  have hLc : ((FlockComplete.ligeritoCut (FlockComplete.asksCut coin l₄.size) (Hashes.ofSha512 Sha512.hash) st.merkle
      sch T (FlockComplete.eraseC cm0) (FlockComplete.basisVals sws (st.m - 7)) (Zk.PadLayout.new st.m st.kLog sch.k0)
      src).run t₃).go l₃ = (.ok ((FlockComplete.eraseL lg, zl, rec), t₄), l₄) :=
    FlockComplete.CutPrefix.eq_of_agree
      (FlockComplete.CutPrefix.ligeritoCut_cut_agree coin (Nat.le_of_lt hl₄n) st.merkle sch T _ _ _ src t₃)
      hs₄ hL (Nat.le_refl _)
  refine ⟨zp, lp, ss, T, sws, t₃, l₃, FlockComplete.eraseL lg, zl, rec, t₄, l₄, hPC, hL, hl₄n, ?_⟩
  dsimp only [FlockComplete.repCut]
  rw [eraseC_cap', FlockComplete.Replay.bind_go hPCc]
  try dsimp only
  rw [FlockComplete.Replay.bind_go hLc]
  try dsimp only
  rw [FlockComplete.Replay.bind_go (get_run_go _ _)]
  try dsimp only
  rw [FlockComplete.Replay.bind_go (liftV_run_go hrpS _ _)]
  try dsimp only
  rw [FlockComplete.Replay.bind_go (liftV_run_go hlgc' _ _)]
  try dsimp only
  exact FlockComplete.LigLock.bind_go_err (innerCut_first coin _ _ _ src _ t₄ (Nat.le_refl _))

end Flock.Firewall.Finish
