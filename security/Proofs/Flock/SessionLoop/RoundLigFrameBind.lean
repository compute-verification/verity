module

public import Proofs.Flock.Claims.ZK.L0PinBind

/-!
# A rep's scheduled opening past level 0, fed

`L0PinFrame`'s `openStep_l0` and `L0PinBind`'s `sched_l0` read level 0's rounds off a rep's schedule; past level 0 the
schedule is still the opening fed the history's coins, so its rounds are the cut run's log, Ligerito's past the ring
switch included.

* **The opening checks its coins' counts** (`Sized`, `sized_openStep`): each round it asks for coins it asks for the
  count its bytes' last eight hold, and fails fed another count; so where a feeding has not failed, each coin it was fed
  holds its round's count (`feed_sized`).
* **The cut log** (`openStep_lig`): where the ring switch's cut run returns at `l₂`, Ligerito's cut run from there logs
  what the opening, fed, sent up to the cut's round, after the log `l`.
* **`Bind.sched_lig`**: so a rep's schedule past its algebraic phase, fed at least the ring switch's and `γ`'s coins and
  fewer than the opening asks for, is that log after `l`: its entry at the cut's round is the schedule's next round, and
  each coin the history holds before it holds the count its round's bytes hold.
-/

@[expose] public section

namespace FlockComplete.LigLock.Fed

open _root_.Flock (F128)
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks openStep)
open _root_.Flock.Firewall.Finish (AsksRel repOpening repPiop repOpening_rel repPiop_rel commitFw ligeritoPZk_fw
  ringSwitchP_eq claimsOf)

/-! ## The opening checks its coins' counts -/

/-- **A step program checks its coins' counts**: each round it asks for, it asks for the count its bytes' last eight
hold, and fed another count it fails. -/
def Sized {α : Type} : Step α → Prop
  | .done _ => True
  | .fail _ => True
  | .ask c n k => n = (_root_.Flock.u64OfLE c (c.size - 8)).toNat ∧
      (∀ z : Array F128, z.size ≠ n → ∃ e, k z = .fail e) ∧ ∀ z, Sized (k z)

theorem sized_bind {α β : Type} {g : α → Step β} (hg : ∀ a, Sized (g a)) :
    ∀ s : Step α, Sized s → Sized (s.bind g)
  | .done a, _ => hg a
  | .fail _, _ => trivial
  | .ask _ _ k, ⟨hn, hf, hk⟩ => ⟨hn, fun z hz => (hf z hz).elim fun e he =>
      ⟨e, (congrArg (fun s => Step.bind s g) he).trans rfl⟩, fun z => sized_bind hg (k z) (hk z)⟩

/-- **The opening's programs check their coins' counts**, from every state. -/
def sizedSim : _root_.Flock.With.Sim OpenM OpenM where
  R x _ := ∀ got, Sized (x.run got)
  pure _ _ := trivial
  bind hxy hfg got := by
    rw [StateT.run_bind]
    exact sized_bind (fun p => hfg p.1 p.2) _ (hxy got)
  throw _ _ := trivial
  lift v _ := by cases v <;> exact trivial

theorem sized_pure_bind {α β : Type} {a : α} {g : α → Step β} (h : Sized (g a)) : Sized (Step.bind (pure a) g) := h

theorem sized_round (coins : Got) (live : Bool) (c : ByteArray) (got : Got) :
    Sized (((stepAsks coins live).round c).run got) := by
  rcases hn : coins[got.size]? with _ | cs
  · cases live
    · simp only [stepAsks, StateT.run, bind, StateT.bind, get, getThe, MonadStateOf.get, StateT.get, pure, Step.bind,
        hn, throw, throwThe, MonadExceptOf.throw]
      exact trivial
    · simp only [stepAsks, StateT.run, bind, StateT.bind, get, getThe, MonadStateOf.get, StateT.get, pure, Step.bind,
        hn, StateT.lift]
      refine ⟨rfl, fun z hz => ?_, fun z => ?_⟩
      · simp only [hz, bne_iff_ne, ne_eq, not_false_eq_true, ite_true, throw, throwThe, MonadExceptOf.throw,
          Function.comp_apply, StateT.lift, StateT.bind, bind, Step.bind]
        exact ⟨_, rfl⟩
      · by_cases hz : z.size = (_root_.Flock.u64OfLE c (c.size - 8)).toNat
        · simp only [hz, bne_self_eq_false, Bool.false_eq_true, ite_false, set, StateT.set, MonadStateOf.set,
            StateT.pure, StateT.bind, pure, bind, Step.bind]
          exact trivial
        · simp only [hz, bne_iff_ne, ne_eq, not_false_eq_true, ite_true, throw, throwThe, MonadExceptOf.throw,
            Function.comp_apply, StateT.lift, StateT.bind, bind, Step.bind]
          exact trivial
  · cases live
    all_goals
      simp only [stepAsks, StateT.run, bind, StateT.bind, get, getThe, MonadStateOf.get, StateT.get, pure, Step.bind,
        hn]
      refine sized_pure_bind ?_; dsimp only; split <;> exact trivial

theorem asksRel_sized (coins : Got) (live : Bool) :
    AsksRel sizedSim (stepAsks coins live).toC (stepAsks coins live).toC :=
  ⟨fun _ _ _ => trivial, fun c got => sized_round coins live c got⟩

theorem sized_run' {α : Type} (p : _root_.Flock.TMW OpenM α) (t : _root_.Flock.Transcript) (s : Got)
    (h : Sized ((p.run t).run s)) : Sized ((p.run' t).run' s) := by
  refine sized_bind (fun _ => ?_) _ (sized_bind (fun _ => ?_) _ h) <;> exact trivial

theorem sized_getThe_bind {β : Type} {f : Got → _root_.Flock.TMW OpenM β} {t : _root_.Flock.Transcript} {got : Got}
    (h : Sized (((f got).run t).run got)) : Sized (((getThe Got >>= f).run t).run got) := h

/-- **A rep's scheduled opening checks its coins' counts.** -/
theorem sized_openStep (t : _root_.Flock.Firewall.Finish.Tab) (rep : Nat) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (coins : Got)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) :
    Sized (openStep t rep dom xs hi coins) := by
  unfold openStep
  dsimp only
  refine sized_run' _ _ _ ?_
  rw [StateT.run_bind, StateT.run_bind]
  refine sized_bind (fun r => ?_) _ (repPiop_rel sizedSim (asksRel_sized coins false) t.st _ _ _ _ _ _ _)
  refine sized_getThe_bind ?_
  split
  · exact trivial
  · rw [StateT.run_bind, StateT.run_bind]
    refine sized_bind (fun _ => ?_) _
      (repOpening_rel sizedSim (asksRel_sized coins true) t.st hmk t.sch _ _ _ _ _ _ _ _ _ _ _ _)
    exact trivial

/-- **Fed coins the program checks**: where the feeding has not failed, each coin fed to a round holds the count the
round's bytes hold. -/
theorem feed_sized {α : Type} : ∀ (s : Step α) (cs : List (Array F128)), Sized s →
    (∀ e, (feedR s cs).1 ≠ .fail e) → ∀ i, i < cs.length → ∀ b, (sentQ (feedR s cs))[i]? = some b →
      (cs.getD i #[]).size = (_root_.Flock.u64OfLE b (b.size - 8)).toNat
  | .done a, cs, _, _, i, _, b, hb => by
    have e : sentQ (feedR (Step.done a) cs) = [] := by cases cs <;> rfl
    rw [e] at hb
    simp at hb
  | .fail e, cs, _, hf, _, _, _, _ => (hf e (by cases cs <;> rfl)).elim
  | .ask _ _ _, [], _, _, _, hi, _, _ => by simp at hi
  | .ask c n k, z :: rest, h, hf, i, hi, b, hb => by
    obtain ⟨hn, hk, hsc⟩ := h
    have hf' : ∀ e, (feedR (k z) rest).1 ≠ .fail e := fun e he => hf e he
    have hz : z.size = n := by
      by_contra hne
      obtain ⟨e, he⟩ := hk z hne
      exact hf' e ((congrArg (fun s => (feedR s rest).1) he).trans (by cases rest <;> rfl))
    have e2 : sentQ (feedR (.ask c n k) (z :: rest)) = c :: sentQ (feedR (k z) rest) := rfl
    rw [e2] at hb
    rcases i with _ | i
    · simp only [List.getElem?_cons_zero, Option.some.injEq] at hb
      subst hb
      simp only [List.getD_cons_zero]
      exact hz.trans hn
    · simp only [List.getElem?_cons_succ] at hb
      simp only [List.getD_cons_succ]
      exact feed_sized (k z) rest (hsc z) hf' i (by simp only [List.length_cons] at hi; omega) b hb

/-! ## The cut log -/

/-- A step that logs nothing after `x` leaves `x`'s log. -/
theorem log_bind_silent {α β : Type} {x : _root_.Flock.TMW PS α} {f : α → _root_.Flock.TMW PS β}
    (hf : ∀ a t l, (((f a).run t).go l).2 = l) (t : _root_.Flock.Transcript) (l : Array Claims.Msg) :
    (((x >>= f).run t).go l).2 = ((x.run t).go l).2 := by
  rw [StateT.run_bind, FlockComplete.go_bind]
  generalize (x.run t).go l = y
  obtain ⟨r, l'⟩ := y
  cases r with
  | error e => rfl
  | ok a => exact hf a.1 a.2 l'

/-- **Ligerito's cut log, past the ring switch**: fed the coins from the log `l`'s round on, where `openStep` sends
anything its PIOP returned having taken the algebraic phase's coins; and where the ring switch's cut run at a cut `n`
within the rounds the opening sent and asks for returns at `l₂`, Ligerito's cut run from there logs, after `l`, the
rounds the opening sent up to the cut's. -/
theorem openStep_lig (t : _root_.Flock.Firewall.Finish.Tab) (rep : Nat) (dom : ByteArray) (xs : Array (Array F128))
    (hi : Array ByteArray) (coins : Got) (hdom : dom.size < 2 ^ 64) (hdg : t.st.digest.size < 2 ^ 64)
    (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hm : t.st.m < 2 ^ 64)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (cs : List (Array F128))
    (coin : Nat → Nat → F128) (l : Array Claims.Msg) (hc : CoinAt coin l.size cs)
    (h0 : sentQ (feedR (openStep t rep dom xs hi coins) cs) ≠ []) :
    ∃ r, ((repPiop (stepAsks coins false) t.st (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
          (_root_.Flock.Firewall.Finish.flat t.cm0.cap) ByteArray.empty (claimsOf xs) fun _ => 0).run
          (_root_.Flock.Transcript.init #[] dom)).run #[] = .done r ∧ r.2.size = coins.size ∧
      ∀ {n : Nat} {ss : Array (Array F128)} {T : F128} {sws : Array _root_.Flock.Switched}
        {tr₂ : _root_.Flock.Transcript} {l₂ : Array Claims.Msg}, l.size ≤ n →
        n < l.size + (sentQ (feedR (openStep t rep dom xs hi coins) cs)).length →
        ((ringSwitchCut (asksCut coin n)
          (fun k => _root_.Flock.Firewall.Finish.switchVec t.g r.1.1.2.2[k]!) r.1.1.2.2).run
          r.1.2).go l = (.ok ((ss, T, sws), tr₂), l₂) →
        (((ligeritoPZk (asksCut coin n) (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
            _root_.Flock.MerkleScheme.hm96Sha512 t.sch T t.cm0.toC t.g (basisVals sws (t.st.m - 7))
            (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
            (Array.replicate (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).len 0) rep t.tPad t.extra
            t.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty))
            (fun _ _ => 0)).run tr₂).go l₂).2 =
          l ++ (((sentQ (feedR (openStep t rep dom xs hi coins) cs)).take (n + 1 - l.size)).map
            Claims.toMsg).toArray := by
  rcases openStep_sent t rep dom xs hi coins hdom hdg hcn hm cs with h | ⟨r, hp, hr, hQ⟩
  · exact absurd h h0
  refine ⟨r, hp, hr, fun {n ss T sws tr₂ l₂} hlo hhi hrs => ?_⟩
  rw [hQ] at hhi ⊢
  obtain ⟨-, hF⟩ := repOpening_feed coin n coins t.st hmk t.sch
    (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) t.tPad rep t.g t.cm0 r.1.1.2.2
    (Array.replicate (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).len 0) t.extra t.lanePad
    (fun l => _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty)) r.1.2 r.2 l cs (by omega) hc
  refine Eq.trans ?_ (congrArg Prod.snd (hF hlo hhi))
  show _ = (((repOpening (cutAsks coin n) (fun lv g s => pure (commitFw lv g s)) t.st t.sch
    (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) t.tPad rep t.g t.cm0 r.1.1.2.2
    (Array.replicate (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).len 0) t.extra t.lanePad
    (fun l => _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty))).run r.1.2).go l).2
  unfold repOpening
  rw [ringSwitchP_eq, cutAsks_toC, FlockComplete.Replay.bind_go hrs, hmk]
  dsimp only
  rw [ligeritoPZk_fw, cutAsks_toC]
  refine Eq.symm (log_bind_silent (fun a _ _ => ?_) _ _)
  obtain ⟨_, _, _⟩ := a
  rfl

theorem msg_inj {a b : ByteArray} (h : Claims.toMsg a = Claims.toMsg b) : a = b := by
  simpa only [Claims.ofMsg_toMsg] using congrArg Claims.ofMsg h

theorem log_at {l : Array Claims.Msg} {Q : List ByteArray} {p : Nat} (hp : l.size ≤ p) :
    (l ++ (Q.map Claims.toMsg).toArray)[p]? = (Q[p - l.size]?).map Claims.toMsg := by
  rw [Array.getElem?_append_right hp]
  simp only [List.getElem?_toArray, List.getElem?_map]

end FlockComplete.LigLock.Fed

namespace Claims.ZKServer.Bind

open _root_.Flock (F128 Transcript TMW)
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks openStep)
open _root_.Flock.Firewall.Finish (repOpening repPiop commitFw claimsOf switchVec)
open FlockComplete (asksCut ringSwitchCut)
open FlockComplete.LigLock.Fed (feedR sentQ CoinAt)
open Loop

/-- **A rep's schedule past level 0 is Ligerito's cut log**: fed the coins of the algebraic phase's rounds, the ring
switch's, `γ`'s and fewer than the opening asks for, and given coins that are the fed ones from the log `l`'s round on,
the ring switch's cut run at the cut `n` (`l`'s rounds and the coins past the algebraic phase) returns at `l₂`, and
Ligerito's cut run from there logs `L`: `l`, then the schedule's rounds past the algebraic phase, up to its next round,
which is `L`'s at `n`. The switch's basis is the switched claims at the rep's claims and the fed coins, and each fed coin
holds the count its round's bytes hold. -/
theorem sched_lig {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64)
    (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63)
    (hm14 : 14 ≤ t.st.m) (hc1 : alg.size + (xs.size + 2) + 1 ≤ cr.size)
    (hc2 : cr.size < alg.size + (Count.openList (xs.size + 2) t.sch).length)
    (coin : ℕ → ℕ → F128) (l : Array Claims.Msg) (hc : CoinAt coin l.size (cr.extract alg.size cr.size).toList)
    {n : ℕ} (hnl : l.size + (cr.size - alg.size) = n) :
    cr.size < sched.size ∧
      ∃ (T : F128) (sws : Array _root_.Flock.Switched) (tr₂ : Flock.Transcript) (l₂ L : Array Claims.Msg),
        l₂.size = l.size + (xs.size + 2) + 1 ∧
        (∃ zc lc, FlockComplete.LigLock.Fed.PtsAt t.st.m t.st.fold.kLog
            (FlockComplete.LigLock.Fed.recCoin (cr.extract 0 alg.size)) 0 zc lc ∧
          sws = FlockComplete.Replay.Rs.sws (_root_.Flock.abcClaims t.st.m t.st.kLog zc lc ++ claimsOf xs)
            ⟨fun k i => ((cr.extract alg.size cr.size).toList.getD k #[])[i]!,
              fun i => ((cr.extract alg.size cr.size).toList.getD (xs.size + 2) #[])[i]!⟩) ∧
        (((FlockComplete.ligeritoPZk (asksCut coin n) (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
            _root_.Flock.MerkleScheme.hm96Sha512 t.sch T t.cm0.toC t.g (FlockComplete.basisVals sws (t.st.m - 7))
            (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
            (Array.replicate (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).len 0) rep t.tPad t.extra
            t.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty))
            (fun _ _ => 0)).run tr₂).go l₂).2 = L ∧
        L = l ++ (((sched.toList.map Prod.fst).drop alg.size).map Claims.toMsg).toArray ∧
        L[n]? = some (Claims.toMsg (sched.getD cr.size (.empty, 0)).1) ∧
        ∀ p, l.size ≤ p → p < n → ∀ b, L[p]? = some (Claims.toMsg b) →
          ((cr.extract alg.size cr.size).toList.getD (p - l.size) #[]).size =
            (_root_.Flock.u64OfLE b (b.size - 8)).toNat := by
  obtain ⟨hdrop, hnf⟩ := schedSoFar_sentQ hal hs (by omega)
  have hcsl : (cr.extract alg.size cr.size).toList.length = cr.size - alg.size := by
    simp only [Array.length_toList, Array.size_extract]
    omega
  have hlenF := feed_part _ _ _
    (Count.openStep_end t rep dom xs hi (cr.extract 0 alg.size) hdom hdg hcn hm hx hk hn hq)
    (by rw [hcsl]; omega) hnf
  rw [hcsl] at hlenF
  have hsz : sched.size = cr.size + 1 := by
    have := congrArg List.length hdrop
    simp only [List.length_drop, List.length_map, Array.length_toList] at this
    omega
  refine ⟨by omega, ?_⟩
  obtain ⟨r, hp, -, hL⟩ := FlockComplete.LigLock.Fed.openStep_lig t rep dom xs hi (cr.extract 0 alg.size) hdom hdg
    hcn hm hmk (cr.extract alg.size cr.size).toList coin l hc
    (by intro h0; rw [h0] at hlenF; simp at hlenF)
  have hna : r.1.1.2.2.size = xs.size + 2 := by
    have := Count.piop_na (cr.extract 0 alg.size) t.st (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
      (_root_.Flock.Firewall.Finish.flat t.cm0.cap) ByteArray.empty (claimsOf xs) (fun _ => 0) hdg hcn (by decide) hm _
      (Sched.trAlike_init dom hdom) #[]
    rw [hp] at this
    have h' : r.1.1.2.2.size = (claimsOf xs).size + 2 := this
    simpa only [claimsOf, Array.size_map] using h'
  obtain ⟨⟨ss, T, sws⟩, t', L', hL', -, -⟩ := Switch.ringSwitchCut_bodies coin n
    (fun k => switchVec t.g r.1.1.2.2[k]!) r.1.1.2.2 (by omega) r.1.2 l (by rw [hna]; omega)
  have key := hL (n := n) (by omega) (by rw [hlenF]; omega) hL'
  obtain ⟨zc, lc, hcl, hpts⟩ := FlockComplete.LigLock.Fed.repPiop_rec _ _ _ _ _ _ _ hm14 _ hp
  obtain ⟨hsw, hl2, -⟩ := FlockComplete.ringSwitchCut_sws _ _ _ _ (by omega) hL'
  rw [← hdrop, List.take_of_length_le (by rw [hdrop, hlenF]; omega)] at key
  refine ⟨T, sws, t', L', _, by rw [hl2, hna], ⟨zc, lc, hpts, ?_⟩, rfl, key, ?_, ?_⟩
  · rw [hsw, ← hcl]
    refine FlockComplete.LigLock.Fed.sws_coins_congr _ (fun k hk i _ => ?_) (fun k _ => ?_)
    · show coin (l.size + k) i = ((cr.extract alg.size cr.size).toList.getD k #[])[i]!
      rw [hc k (by rw [hcsl]; omega), List.getD_eq_getElem _ _ (by rw [hcsl]; omega)]
    · show coin (l.size + r.1.1.2.2.size) k = ((cr.extract alg.size cr.size).toList.getD (xs.size + 2) #[])[k]!
      rw [hna, hc (xs.size + 2) (by rw [hcsl]; omega), List.getD_eq_getElem _ _ (by rw [hcsl]; omega)]
  · rw [key, FlockComplete.LigLock.Fed.log_at (by omega), show n - l.size = cr.size - alg.size by omega,
      List.getElem?_drop, List.getElem?_map, Array.getElem?_toList,
      show alg.size + (cr.size - alg.size) = cr.size by omega, Array.getElem?_eq_getElem (by omega)]
    simp only [Option.map_some, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (show cr.size < sched.size by
      omega), Option.getD_some]
  · intro p h1 h2 b hb
    rw [key, FlockComplete.LigLock.Fed.log_at h1] at hb
    obtain ⟨b', hb', hbb⟩ := Option.map_eq_some_iff.mp hb
    rw [FlockComplete.LigLock.Fed.msg_inj hbb] at hb'
    rw [hdrop] at hb'
    exact FlockComplete.LigLock.Fed.feed_sized _ _
      (FlockComplete.LigLock.Fed.sized_openStep t rep dom xs hi (cr.extract 0 alg.size) hmk) hnf (p - l.size)
      (by rw [hcsl]; omega) b hb'

end Claims.ZKServer.Bind
