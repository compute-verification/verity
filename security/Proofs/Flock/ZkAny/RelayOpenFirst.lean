module

public import Proofs.Flock.Claims.ZK.L0PinBind

/-!
# The opening's first round

Past the algebraic phase's rounds, a rep's schedule's next round, where it has one, is the opening's first
(`Flags.PcsRound`): the ring switch's first claim round, from the PIOP's empty pending bytes, squeezing `7`.

* **The cut ring switch at its first ask** (`Switch.ringSwitchCut_zero`): cut at round `0`, it stops past the openings
  having sent its first body (`Switch.rsBody` at `0`), the loop's first step throwing (`Switch.forIn_first_err`).
* **The opening fed** (`Bind.openStep_first`): where `openStep`, fed, sends anything, its first round is that body.
* **The schedule** (`Bind.sched_pcs`): a schedule with a round at the algebraic phase's round count has the opening's
  first round there, the opening's count `7`.
-/

@[expose] public section

namespace Claims.ZKServer.Switch

open _root_.Flock (F128 Transcript TMW)
open _root_.Flock.Transcript (framed header padded)
open _root_.Flock.Firewall.Frame (LABEL_PCS)
open FlockSoundness.Refine (Frame encs label_enc absorbSlice_enc framed_enc)
open FlockComplete (PS asksCut ringSwitchCut)
open FlockComplete.Replay (afterSq bind_go liftTM_ok)

/-- **A loop whose first step throws**, then a continuation, throws with the first step's log. -/
theorem forIn_first_err {σ β : Type} (body : ℕ → σ → TMW PS (ForInStep σ)) {n : ℕ} (hn : n ≠ 0) (s₀ : σ)
    (f : σ → TMW PS β) {t : Flock.Transcript} {l l' : Array Claims.Msg} {e : String}
    (h : ((body 0 s₀).run t).go l = (.error e, l')) :
    ((forIn [:n] s₀ body >>= f).run t).go l = (.error e, l') := by
  refine FlockComplete.bind_go_err ?_
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  rw [List.range'_succ, List.forIn_cons]
  exact FlockComplete.bind_go_err h

theorem rsLoopB_zero (sv : ℕ → Array F128) (coin : ℕ → ℕ → F128) (L : ℕ) (t : Flock.Transcript) {N : ℕ}
    (hN : N ≠ 0) : rsLoopB sv coin L (pcsT t) 0 = rsBody sv N t.pending.data.toList 0 := by
  rw [rsLoopB_eq, rsBody, ite_eq_left (Nat.pos_of_ne_zero hN), ite_eq_left rfl]
  simp only [rsLoopT, pcsT, label_enc, List.append_assoc]

/-- **The cut ring switch, cut at its first ask**: from the empty log it stops past the openings having sent its first
body after `t`'s pending bytes. -/
theorem ringSwitchCut_zero (coin : ℕ → ℕ → F128) (sv : ℕ → Array F128) (claims : Array _root_.Flock.Claim)
    (hN : claims.size ≠ 0) (t : Flock.Transcript) :
    ((ringSwitchCut (asksCut coin 0) sv claims).run t).go #[] =
      (.error "past the openings", #[rsBody sv claims.size t.pending.data.toList 0]) := by
  unfold ringSwitchCut
  dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  try dsimp only
  refine forIn_first_err _ hN _ _ ?_
  try dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [FlockComplete.bind_go_err (x := FlockComplete.sampleVecP (asksCut coin 0) 7)
    (FlockComplete.squeezeP_cut_err coin 2 (by decide) _ (Nat.zero_le _))]
  rw [← rsLoopB_zero sv coin 0 t hN]
  rfl

end Claims.ZKServer.Switch

namespace Claims.ZKServer.Bind

open _root_.Flock (F128 Transcript TMW)
open _root_.Flock.Firewall.Frame (Cfg Masker)
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks openStep)
open _root_.Flock.Firewall.Finish (repOpening repPiop commitFw claimsOf ringSwitchP_eq switchVec)
open FlockSoundness.Refine (Frame encs)
open FlockComplete (PS asksCut ringSwitchCut)
open FlockComplete.Replay (bind_go liftTM_ok)
open FlockComplete.LigLock.Fed (feedR sentQ pendC feed_eq CoinAt repOpening_feed cutAsks cutAsks_toC openStep_sent)
open Switch (rsBody)
open Loop

/-- **The opening's first round is the ring switch's first**: where `openStep`, fed, sends anything, its first round is
the switch's first body from empty pending bytes, a claim's switched vector of `128` words. -/
theorem openStep_first (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (coins : Array (Array F128)) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (cs : List (Array F128))
    (h0 : sentQ (feedR (openStep t rep dom xs hi coins) cs) ≠ []) :
    ∃ sv : ℕ → Array F128, (∀ k, (sv k).size = 128) ∧
      ((sentQ (feedR (openStep t rep dom xs hi coins) cs)).take 1).map Claims.toMsg =
        [rsBody sv (xs.size + 2) [] 0] := by
  rcases openStep_sent t rep dom xs hi coins hdom hdg hcn hm cs with h | ⟨r, hp, hr, hQ⟩
  · exact absurd h h0
  have hna : r.1.1.2.2.size = xs.size + 2 := by
    have := Count.piop_na coins t.st (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
      (_root_.Flock.Firewall.Finish.flat t.cm0.cap) ByteArray.empty (claimsOf xs) (fun _ => 0) hdg hcn (by decide) hm _
      (Sched.trAlike_init dom hdom) #[]
    rw [hp] at this
    have h' : r.1.1.2.2.size = (claimsOf xs).size + 2 := this
    simpa only [claimsOf, Array.size_map] using h'
  have hpe : r.1.2.pending = ByteArray.empty := Switch.endE_repPiop _ _ _ _ _ _ _ _ _ _ hp
  refine ⟨fun k => switchVec t.g r.1.1.2.2[k]!, fun k => by simp [switchVec], ?_⟩
  have hc : CoinAt (fun j i => (cs.getD j #[])[i]!) (#[] : Array Claims.Msg).size cs := by
    intro j hj
    funext i
    simp [List.getD_eq_getElem?_getD, hj]
  obtain ⟨-, hF⟩ := repOpening_feed (fun j i => (cs.getD j #[])[i]!) 0 coins t.st hmk t.sch
    (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) t.tPad rep t.g t.cm0 r.1.1.2.2
    (Array.replicate (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).len 0) t.extra t.lanePad
    (fun l => _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty)) r.1.2 r.2 #[] cs (by omega) hc
  rw [hQ] at h0 ⊢
  have e := hF (by simp) (by
    rw [Array.size_empty, Nat.zero_add]
    exact Nat.pos_of_ne_zero fun hz => h0 (List.eq_nil_of_length_eq_zero hz))
  have hL := Switch.ringSwitchCut_zero (fun j i => (cs.getD j #[])[i]!) (fun k => switchVec t.g r.1.1.2.2[k]!)
    r.1.1.2.2 (by omega) r.1.2
  have hro : ((repOpening (cutAsks (fun j i => (cs.getD j #[])[i]!) 0)
      (fun lv g s => pure (commitFw lv g s)) t.st t.sch (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
      t.tPad rep t.g t.cm0 r.1.1.2.2
      (Array.replicate (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).len 0) t.extra t.lanePad
      (fun l => _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty))).run r.1.2).go #[] =
      (.error "past the openings",
        #[rsBody (fun k => switchVec t.g r.1.1.2.2[k]!) r.1.1.2.2.size r.1.2.pending.data.toList 0]) := by
    unfold repOpening
    rw [ringSwitchP_eq, cutAsks_toC]
    exact FlockComplete.bind_go_err hL
  have e2 := congrArg Array.toList (Prod.mk.inj (hro.symm.trans e)).2
  rw [hpe, empty_toList, hna] at e2
  simp only [Array.toList_append, List.nil_append, Array.size_empty, Nat.sub_zero, Nat.zero_add] at e2
  exact e2.symm

/-- **A rep's round at the algebraic phase's round count is the opening's first**: fed at least the algebraic phase's
coins, a rep's schedule that has a round there has the ring switch's first round, squeezing the opening's count `7`. -/
theorem sched_pcs {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hcr : alg.size ≤ cr.size) (hlt : alg.size < sched.size)
    (hdom : dom.size < 2 ^ 64) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hm : t.st.m < 2 ^ 64)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0)
    (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63) :
    Flags.PcsRound (sched.getD alg.size (.empty, 0)) := by
  obtain ⟨hdrop, -⟩ := schedSoFar_sentQ hal hs hcr
  have hne : sentQ (feedR (openStep t rep dom xs hi (cr.extract 0 alg.size))
      (cr.extract alg.size cr.size).toList) ≠ [] := by
    intro he
    rw [he] at hdrop
    have := congrArg List.length hdrop
    simp only [List.length_drop, List.length_map, Array.length_toList, List.length_nil] at this
    omega
  obtain ⟨sv, hsv, hbody⟩ := openStep_first t rep dom xs hi (cr.extract 0 alg.size) hdom hdg hcap hm hmk
    (cr.extract alg.size cr.size).toList hne
  have hb := sched_bytes (N := 0) (B := rsBody sv (xs.size + 2) []) hdrop
    (by exact hbody)
    (Nat.pos_of_ne_zero fun hz => hne (List.eq_nil_of_length_eq_zero hz)) (le_refl 0)
  have hN0 : xs.size + 2 ≠ 0 := by omega
  have hc := sched_count hal hs hdom hdg hcap hm hx hk hn hq (i := 0) (by omega)
  rw [openList_lt (i := 0) t.sch hN0 (by omega)] at hc
  rw [Nat.add_zero] at hb hc
  rw [show sched.getD alg.size (.empty, 0) = ((sched.getD alg.size (.empty, 0)).1, 7) from Prod.ext rfl hc]
  exact Switch.pcsRound_rs sv hN0 (by rw [hsv]; decide) hb

end Claims.ZKServer.Bind
