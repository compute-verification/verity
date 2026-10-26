import Proofs.Flock.Claims.ZK.FinishRep
import Proofs.Flock.Claims.ZK.SimLig
import Proofs.Flock.Claims.ZK.SimInner
import Proofs.Flock.Claims.ZK.CausalExe

/-!
# The firewall's rep against the rep from a source

`flock-firewall finish`'s rep (`repFwC`) computes what it sends from the passed values and the rep's pads; the session's
rep from a source (`FlockComplete.repCut`) reads the same messages from the source. Here both run on the session's sends
(`asksCut`), and where the source holds what the firewall computes, they run alike up to what only the final message
reads (`eraseZ`):

* **The PIOP** (`piopFw`, `piopCut_fw`): the firewall's zerocheck, lincheck and ring switch are the source's where the
  source's masked values are `vals + h` below `zp 64`, the only ones they read
  (`FlockComplete.CutEq.zerocheckCut_eqAt`, `lincheckCut_eqAt`),
  and its switched vectors `switchVec g` at the claims, and where the rank check, which the firewall leaves to the
  frame, passes (`ringSwitchCut_congr`, `rankCheck_pass_go`).
* **Ligerito** and **the inner proof**: the session's hybrids, `FlockComplete.LigLock.ligeritoCut_alike` at `LigHyp`
  and `FlockComplete.SimInner.innerCut_alike` at `InnerSrc`, at the firewall's run. Its inner proof stops on a failing
  batched constraint, so where it returns, that constraint held (`innerP_batched`).
* **`repCut_fw`**: where the firewall's rep returns, the rep from the source, at level 0's commitment erased, returns the
  same transcript and log, and a proof equal to the firewall's up to `eraseZ`.
-/

namespace Flock.Firewall.Finish

open Flock Transcript

theorem forIn_range_congr {m : Type → Type} [Monad m] {β : Type} (n : ℕ) (b : β)
    {f g : ℕ → β → m (ForInStep β)} (h : ∀ k < n, ∀ b, f k b = g k b) :
    forIn [:n] b f = forIn [:n] b g := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (b : β), a + k = n → forIn (List.range' a k) b f = forIn (List.range' a k) b g from
    key n 0 b (by omega)
  intro k
  induction k with
  | zero => intro a b _; rfl
  | succ k ih =>
    intro a b ha
    rw [List.range'_succ, List.forIn_cons, List.forIn_cons, h a (by omega) b]
    congr 1
    funext r
    cases r with
    | done _ => rfl
    | yield b' => exact ih (a + 1) b' (by omega)

section

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **The firewall's PIOP**: `repFwC`'s zerocheck, lincheck and ring switch, sending the masked values `mk` and the
switched vectors of `g` at the claims. -/
def piopFw (io : FlockComplete.Asks M) (st : Setup) (lay : Zk.PadLayout) (g : Array F128) (mk : ℕ → F128)
    (cap root : ByteArray) (extras : Array Claim) :
    TMW M (ZerocheckProof × LincheckProof × Array (Array F128) × F128 × Array Switched) := do
  let (zp, zc) ← FlockComplete.zerocheckCut io st.m st.digest cap root lay mk
  let (lp, lc) ← FlockComplete.lincheckCut io st.fold st.m lay mk zc
  let claims := abcClaims st.m st.kLog zc lc ++ extras
  let (ss, T, sws) ← FlockComplete.ringSwitchCut io (fun k => switchVec g claims[k]!) claims
  return (zp, lp, ss, T, sws)

/-- **The ring switch reads claim `k`'s switched vector only below the claim count.** -/
theorem ringSwitchCut_congr (io : FlockComplete.Asks M) {sv sv' : ℕ → Array F128} {claims : Array Claim}
    (h : ∀ k < claims.size, sv k = sv' k) :
    FlockComplete.ringSwitchCut io sv claims = FlockComplete.ringSwitchCut io sv' claims := by
  unfold FlockComplete.ringSwitchCut
  dsimp only
  refine bind_congr fun _ => ?_
  refine congrArg (· >>= _) (forIn_range_congr _ _ fun k hk b => ?_)
  rw [h k hk]

end

open FlockComplete (PS)

/-! ## Runs of `PS` step by step -/

theorem run_bind_ok {α β : Type} {x : TMW PS α} {f : α → TMW PS β} {t t'' : Transcript}
    {l l'' : Array Claims.Msg} {b : β} (h : ((x >>= f).run t).go l = (.ok (b, t''), l'')) :
    ∃ a t' l', (x.run t).go l = (.ok (a, t'), l') ∧ ((f a).run t').go l' = (.ok (b, t''), l'') := by
  rw [StateT.run_bind, FlockComplete.go_bind] at h
  rcases e : (x.run t).go l with ⟨r, l'⟩
  rw [e] at h
  cases r with
  | error _ => cases h
  | ok a =>
    obtain ⟨a, t'⟩ := a
    exact ⟨a, t', l', rfl, h⟩

theorem get_run_go (t : Transcript) (l : Array Claims.Msg) :
    ((get : TMW PS Transcript).run t).go l = (.ok (t, t), l) := rfl

theorem throw_run_go {α : Type} (e : String) (t : Transcript) (l : Array Claims.Msg) :
    ((throw e : TMW PS α).run t).go l = (.error e, l) := rfl

/-- **Where the inner proof returns at `asksCut`, its batched constraint held** on the pads, at the coins of the rounds
it drew from: `innerP` stops on a batched constraint that fails. -/
theorem innerP_batched {coin : ℕ → ℕ → F128} {n : ℕ} {hs : Hashes PS} {lay : Zk.PadLayout}
    {p : FlockComplete.PadsZk} {ic : FlockComplete.InnerCommit} {finalC : F128} {cons : Array (String × Zk.Lin)}
    {t t' : Transcript} {l l' : Array Claims.Msg} {i : Zk.InnerProof}
    (h : ((FlockComplete.innerP (FlockComplete.asksCut coin n) hs lay p ic finalC cons).run t).go l =
      (.ok (i, t'), l')) :
    ((FlockComplete.Replay.InZ.lin lay p cons (FlockComplete.SimInner.cutCoins coin l.size)).c +
      _root_.Flock.dot (FlockComplete.Replay.InZ.lin lay p cons (FlockComplete.SimInner.cutCoins coin l.size)).a
        p.h).isZero = true := by
  unfold FlockComplete.innerP at h
  try dsimp only at h
  obtain ⟨_, t₁, l₁, h₁, h⟩ := run_bind_ok h
  obtain rfl := FlockComplete.SimInner.liftTM_ok_log h₁
  obtain ⟨_, t₂, l₂, h₂, h⟩ := run_bind_ok h
  obtain rfl := FlockComplete.SimInner.liftTM_ok_log h₂
  obtain ⟨eps, t₃, l₃, h₃, h⟩ := run_bind_ok h
  obtain ⟨-, rfl, rfl, rfl⟩ :=
    FlockComplete.SimInner.ite_ok ((FlockComplete.SimInner.sampleCut_go coin n t₂ l).symm.trans h₃)
  try dsimp only at h
  obtain ⟨_, t₄, l₄, h₄, h⟩ := run_bind_ok h
  obtain rfl := FlockComplete.SimInner.liftTM_ok_log h₄
  obtain ⟨gs, t₅, l₅, h₅, h⟩ := run_bind_ok h
  obtain ⟨-, rfl, rfl, rfl⟩ := FlockComplete.SimInner.ite_ok
    ((FlockComplete.SimInner.sampleVecCut_go coin n (by simp [Zk.triple]) t₄ _).symm.trans h₅)
  try dsimp only at h
  simp only [Array.size_push] at h
  simp only [FlockComplete.Replay.InZ.lin, FlockComplete.Replay.InZ.all, FlockComplete.Replay.InZ.rho,
    FlockComplete.Replay.InZ.sigma, FlockComplete.SimInner.cutCoins]
  by_contra hz
  rw [Bool.not_eq_true] at hz
  simp only [hz, Bool.not_false, eq_self_iff_true, ↓reduceIte] at h
  all_goals first
    | (rw [throw_run_go] at h; cases h)
    | (obtain ⟨_, _, _, h', _⟩ := run_bind_ok h; rw [throw_run_go] at h'; cases h')

theorem go_liftV' {α : Type} (v : V α) (l : Array Claims.Msg) : (monadLift v : PS α).go l = (v, l) := by
  cases v <;> rfl

theorem liftV_run_ok {α : Type} {v : V α} {t t' : Transcript} {l l' : Array Claims.Msg} {a : α}
    (h : ((StateT.lift (monadLift v : PS α) : TMW PS α).run t).go l = (.ok (a, t'), l')) :
    v = .ok a ∧ t' = t ∧ l' = l := by
  rw [StateT.run_lift, FlockComplete.go_bind, go_liftV'] at h
  cases v with
  | error _ => cases h
  | ok b =>
    cases h
    exact ⟨rfl, rfl, rfl⟩

theorem liftV_run_go {α : Type} {v : V α} {a : α} (hv : v = .ok a) (t : Transcript) (l : Array Claims.Msg) :
    ((StateT.lift (monadLift v : PS α) : TMW PS α).run t).go l = (.ok (a, t), l) := by
  subst hv
  rfl

theorem rankCheck_pass_go {m kLog : ℕ} {prior : Array (List F128)} {words : List ℕ} {t : Transcript}
    (l : Array Claims.Msg)
    (h : FlockComplete.MaskRank.maskRankOk (prior ++ FlockComplete.MaskRank.roundPoints m kLog t.rounds) words = true) :
    ((FlockComplete.rankCheckP m kLog prior words : TMW PS Unit).run t).go l = (.ok ((), t), l) := by
  have e : ((FlockComplete.rankCheckP m kLog prior words : TMW PS Unit).run t).go l =
      ((if FlockComplete.MaskRank.maskRankOk (prior ++ FlockComplete.MaskRank.roundPoints m kLog t.rounds) words then
          pure () else throw "degenerate coins: the mask slot's weights are rank-deficient" : TMW PS Unit).run t).go l :=
    rfl
  rw [e, if_pos h]
  rfl

/-! ## What only the final message reads -/

theorem eraseC_cap' (cm : FlockComplete.Commit) : (FlockComplete.eraseC cm).cap = cm.cap := rfl

theorem replay_eraseL' (st : Setup) (lay : Zk.PadLayout) (cap : Array ByteArray) (params : PcsParams)
    (zp : ZerocheckProof) (lp : LincheckProof) (ss : Array (Array F128)) (lg : LigeritoProof) (root : ByteArray) :
    Zk.replay st lay
        { cap, params, zerocheck := zp, lincheck := lp, ringSwitches := ss, ligerito := FlockComplete.eraseL lg } root =
      Zk.replay st lay { cap, params, zerocheck := zp, lincheck := lp, ringSwitches := ss, ligerito := lg } root := by
  unfold Zk.replay
  dsimp only

theorem ligeritoConstraints_eraseL' (lay : Zk.PadLayout) (rec : Zk.L0Rec) (lg : LigeritoProof)
    (zl : Zk.Level0Msgs) :
    Zk.ligeritoConstraints lay rec (FlockComplete.eraseL lg) zl = Zk.ligeritoConstraints lay rec lg zl := by
  unfold Zk.ligeritoConstraints FlockComplete.eraseL
  dsimp only

theorem eraseL_eraseL' (lg : LigeritoProof) : FlockComplete.eraseL (FlockComplete.eraseL lg) = FlockComplete.eraseL lg := by
  simp only [FlockComplete.eraseL, FlockComplete.eraseLP, Array.map_map, Function.comp_def]

/-! ## The PIOP -/

/-- **The source's PIOP is the firewall's**: answered by `asksCut`, from a source whose masked values below `zp 64` are
the firewall's `mk`, where at the firewall's zerocheck and lincheck the rank check passes and the source's switched
vectors are the firewall's at the claims, `piopCut` returns what `piopFw` returns. -/
theorem piopCut_fw (coin : ℕ → ℕ → F128) (n : ℕ) (st : Setup) (lay : Zk.PadLayout) (g : Array F128)
    (mk : ℕ → F128) (src : FlockComplete.RepSrc) (cap root : ByteArray) (extras : Array Claim)
    (prior : Array (List F128)) (words : List ℕ) (hm14 : 14 ≤ st.m) (hm : st.m - 6 ≤ lay.nMlv)
    (hk : st.fold.kLog - 6 ≤ lay.nLc) (hmk : ∀ i < lay.zp 64, src.masked i = mk i) {t₀ : Transcript}
    {l : Array Claims.Msg} (hn : l.size ≤ n)
    (hpiop : ∀ zp zc t₁ l₁ lp lc t₂ l₂,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut coin n) st.m st.digest cap root lay mk).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut coin n) st.fold st.m lay mk zc).run t₁).go l₁ =
        (.ok ((lp, lc), t₂), l₂) →
      FlockComplete.MaskRank.maskRankOk (prior ++ FlockComplete.MaskRank.roundPoints st.m st.kLog t₂.rounds) words =
          true ∧
        ∀ k < (abcClaims st.m st.kLog zc lc ++ extras).size,
          src.sv k = switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
    {r : ZerocheckProof × LincheckProof × Array (Array F128) × F128 × Array Switched} {t₃ : Transcript}
    {l₃ : Array Claims.Msg}
    (h : ((piopFw (FlockComplete.asksCut coin n) st lay g mk cap root extras).run t₀).go l = (.ok (r, t₃), l₃)) :
    ((FlockComplete.piopCut (FlockComplete.asksCut coin n) st lay src cap root extras prior words).run t₀).go l =
      (.ok (r, t₃), l₃) := by
  unfold piopFw at h
  obtain ⟨⟨zp, zc⟩, t₁, l₁, h₁, h⟩ := run_bind_ok h
  dsimp only at h
  obtain ⟨⟨lp, lc⟩, t₂, l₂, h₂, h⟩ := run_bind_ok h
  dsimp only at h
  obtain ⟨hrk, hsv⟩ := hpiop zp zc t₁ l₁ lp lc t₂ l₂ h₁ h₂
  have hfa : lay.fa + 2 < lay.zp 64 := by
    simp only [Zk.PadLayout.zp, Zk.PadLayout.lc, Zk.PadLayout.fc]; omega
  have ez := FlockComplete.CutEq.zerocheckCut_eqAt coin n hm14 st.digest cap root lay src.masked mk t₀ rfl hn
    (fun _ i hi => ⟨hmk _ (by simp only [Zk.PadLayout.r1ab, Zk.PadLayout.fa] at hfa ⊢; omega),
      hmk _ (by simp only [Zk.PadLayout.r1c, Zk.PadLayout.fa] at hfa ⊢; omega)⟩)
    (fun i hi _ => ⟨hmk _ (by simp only [Zk.PadLayout.ml, Zk.PadLayout.fa] at hfa ⊢; omega),
      hmk _ (by simp only [Zk.PadLayout.ml, Zk.PadLayout.fa] at hfa ⊢; omega)⟩)
    (fun _ => ⟨hmk _ (by omega), hmk _ (by simp only [Zk.PadLayout.fb]; omega),
      hmk _ (by simp only [Zk.PadLayout.fc]; omega)⟩)
  have h₁' := ez.1.trans h₁
  have hl₁ : l₁.size ≤ n := (ez.2 _ _ h₁').1.trans_le (ez.2 _ _ h₁').2.1
  have el := FlockComplete.CutEq.lincheckCut_eqAt coin n st.fold st.m lay src.masked mk zc t₁ rfl hl₁
    (fun j hj _ => ⟨hmk _ (by simp only [Zk.PadLayout.zp, Zk.PadLayout.lc]; omega),
      hmk _ (by simp only [Zk.PadLayout.zp, Zk.PadLayout.lc]; omega)⟩)
    (fun _ i hi => hmk _ (by simp only [Zk.PadLayout.zp]; omega))
  have h₂' := el.1.trans h₂
  unfold FlockComplete.piopCut
  rw [FlockComplete.Replay.bind_go h₁']
  dsimp only
  rw [FlockComplete.Replay.bind_go h₂']
  dsimp only
  rw [FlockComplete.Replay.bind_go (rankCheck_pass_go l₂ hrk), ringSwitchCut_congr _ hsv]
  exact h

/-! ## The rep -/

/-- **The rep from the source is the firewall's up to `eraseZ`**: answered by `asksCut` from the log `l`, where the
firewall's rep returns a proof `p`, the rep from a source that holds what the firewall computes returns, at level 0's
commitment erased, the same transcript and log and a proof `p'` with `eraseZ p' = eraseZ p`. The source holds: the
masked values `vals + h` below `zp 64`, the PIOP's (`hmk`, where the fold's `kLog` is the setup's: the firewall's
passed values are those `zp 64` words); at the firewall's zerocheck and lincheck, a passing rank check and its switched
vectors (`hpiop`); at its PIOP, level 0's opening (`hlig`, `LigHyp`), whose masked values from `zp 64` to `h_ab` are
the firewall's own; and at its Ligerito and constraints, the inner proof's (`hinner`, `InnerSrc`), where the batched
constraint held, as the firewall's own inner proof checks (`innerP_batched`). -/
theorem repCut_fw (coin : ℕ → ℕ → F128) (n : ℕ) (st : Setup) (hm : st.merkle = MerkleScheme.hm96Sha512)
    (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (g : Array F128) (cm0 : FlockComplete.Commit)
    (extras : Array Claim) (pads : FlockComplete.PadsZk) (ic : FlockComplete.InnerCommit) (vals : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (src : FlockComplete.RepSrc)
    (prior : Array (List F128)) (words : List ℕ) (t₀ : Transcript) (l : Array Claims.Msg)
    (hm14 : 14 ≤ st.m) (hn : l.size ≤ n) (hkL : st.fold.kLog = st.kLog)
    (hmk : ∀ i < (Zk.PadLayout.new st.m st.kLog sch.k0).zp 64, src.masked i = vals[i]! + pads.h[i]!)
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
    (hinner : ∀ zp lp ss T sws t₃ l₃ lg zl rec t₄ l₄ rp tt lgc,
      ((piopFw (FlockComplete.asksCut coin n) st (Zk.PadLayout.new st.m st.kLog sch.k0) g
          (fun i => vals[i]! + pads.h[i]!) (FlockComplete.flat cm0.cap) ic.root extras).run t₀).go l =
        (.ok ((zp, lp, ss, T, sws), t₃), l₃) →
      ((FlockComplete.ligeritoPZk (FlockComplete.asksCut coin n) (Hashes.ofSha512 Sha512.hash)
          MerkleScheme.hm96Sha512 sch T cm0 g (FlockComplete.basisVals sws (st.m - 7))
          (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h rep tPad extra lanePad salt fun _ _ => 0).run t₃).go l₃ =
        (.ok ((lg, zl, rec), t₄), l₄) →
      (Zk.replay st (Zk.PadLayout.new st.m st.kLog sch.k0)
          { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp, ringSwitches := ss,
            ligerito := lg } ic.root).run (Transcript.init t₄.rounds dom) = .ok (rp, tt) →
      Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rec lg zl = .ok lgc →
      ((FlockComplete.Replay.InZ.lin (Zk.PadLayout.new st.m st.kLog sch.k0) pads
          (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! ss[0]!)),
            ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)
          (FlockComplete.SimInner.cutCoins coin l₄.size)).c +
        _root_.Flock.dot (FlockComplete.Replay.InZ.lin (Zk.PadLayout.new st.m st.kLog sch.k0) pads
          (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! ss[0]!)),
            ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)
          (FlockComplete.SimInner.cutCoins coin l₄.size)).a pads.h).isZero = true →
      FlockComplete.SimInner.InnerSrc coin n Sha512.hash (Zk.PadLayout.new st.m st.kLog sch.k0) pads
        (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! ss[0]!)),
          ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc) src l₄.size)
    {p : Zk.ZkProof} {t : Transcript} {l' : Array Claims.Msg}
    (h : ((repFwC (FlockComplete.asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch tPad rep dom g cm0 extras pads ic
      vals extra lanePad salt).run t₀).go l = (.ok (p, t), l')) :
    ∃ p', ((FlockComplete.repCut (FlockComplete.asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch rep dom
        (FlockComplete.eraseC cm0) extras ic src prior words).run t₀).go l = (.ok (p', t), l') ∧
      FlockComplete.eraseZ p' = FlockComplete.eraseZ p := by
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
  obtain ⟨inner, t₈, l₈, h₈, hf⟩ := run_bind_ok hc
  cases hf
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
  have hPC := piopCut_fw coin n st _ g _ src _ _ extras prior words hm14 (by exact Nat.le_refl _)
    (by rw [hkL]; exact Nat.le_refl _) hmk hn hpiop hP
  have h₄' := h₄
  rw [← hm] at h₄'
  have hL : ((FlockComplete.ligeritoCut (FlockComplete.asksCut coin n) (Hashes.ofSha512 Sha512.hash) st.merkle sch T
      (FlockComplete.eraseC cm0) (FlockComplete.basisVals sws (st.m - 7)) (Zk.PadLayout.new st.m st.kLog sch.k0)
      src).run t₃).go l₃ = (.ok ((FlockComplete.eraseL lg, zl, rec), t₄), l₄) := by
    rw [FlockComplete.LigLock.ligeritoCut_alike coin n _ st.merkle sch T cm0 g _ _ pads.h rep tPad extra lanePad salt
      src t₃ l₃ (hlig zp lp ss T sws t₃ l₃ hP), FlockComplete.Replay.bind_go h₄']
    rfl
  have hrp' : (Zk.replay st (Zk.PadLayout.new st.m st.kLog sch.k0)
      { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp, ringSwitches := ss,
        ligerito := FlockComplete.eraseL lg } ic.root).run (Transcript.init t₄.rounds dom) = .ok (rp, tt) := by
    rw [replay_eraseL']
    exact hrp
  have hlgc' : Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rec (FlockComplete.eraseL lg) zl =
      .ok lgc := by
    rw [ligeritoConstraints_eraseL']
    exact hlgc
  have hsrc := hinner zp lp ss T sws t₃ l₃ lg zl rec t₄ l₄ rp tt lgc hP h₄ hrp hlgc (innerP_batched h₈)
  have hI := FlockComplete.SimInner.innerCut_alike (f := fun i => (pure (FlockComplete.eraseIP i) : TMW PS _))
    (ic := ic) (finalC := zp.finalC) (t := t₄)
    (fun i i' e => by simp only [e]) hsrc
  rw [FlockComplete.Replay.bind_go h₈] at hI
  obtain ⟨inner', t₉, l₉, h₉, hI⟩ := run_bind_ok hI
  try dsimp only at hI
  rw [FlockComplete.Replay.pure_go] at hI
  simp only [Prod.mk.injEq, Except.ok.injEq] at hI
  obtain ⟨⟨hie, rfl⟩, rfl⟩ := hI
  refine ⟨{ proof := { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp,
                       ringSwitches := ss, ligerito := FlockComplete.eraseL lg }, level0 := zl, inner := inner' },
    ?_, ?_⟩
  · dsimp only [FlockComplete.repCut]
    rw [eraseC_cap', FlockComplete.Replay.bind_go hPC]
    try dsimp only
    rw [FlockComplete.Replay.bind_go hL]
    try dsimp only
    rw [FlockComplete.Replay.bind_go (get_run_go _ _)]
    try dsimp only
    rw [FlockComplete.Replay.bind_go (liftV_run_go hrp' _ _)]
    try dsimp only
    rw [FlockComplete.Replay.bind_go (liftV_run_go hlgc' _ _)]
    try dsimp only
    rw [FlockComplete.Replay.bind_go h₉]
    rfl
  · simp only [FlockComplete.eraseZ, eraseL_eraseL', hie]

end Flock.Firewall.Finish
