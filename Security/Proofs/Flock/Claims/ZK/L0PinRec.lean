import Proofs.Flock.Claims.ZK.L0PinFrame
import Proofs.Flock.Claims.ZK.FinishSwPts

/-!
# A rep's PIOP on its recorded coins

`Drive.openStep` runs a rep's PIOP on the algebraic phase's coins the frame recorded (`stepAsks coins false`), and level
0's basis reads the claims it returns. Answered by recorded coins, the PIOP is the cut run at those coins, so its claims'
points are the coins':

* **The recorded simulation** (`recSim`, `RecR`): where a program over `OpenM` returns from a record `got`, the run over
  `PS` from any log of `got`'s length returns the same, at a log of the new record's length.
* **The channels** (`recR_round`, `asksRel_rec`): a recorded round is `asksCut`'s at the recorded coins (`recCoin`).
* **`repPiop_rec`**: a rep's PIOP that returns on recorded coins returns the zerocheck's and lincheck's claims at the
  recorded coins' points (`PtsAt`), then the extra claims.
* **The points' reads** (`ptsAt_congr`, `ptsAt_claims_x`, `sws_coins_congr`): the claims' points and the switched claims
  read only their rounds' first words.
-/

namespace FlockComplete.LigLock.Fed

open _root_.Flock (F128 ZcOut LcOut abcClaims)
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks)
open _root_.Flock.Firewall.Finish (AsksRel Asks.toC repPiop)
open FlockComplete.Replay (pushes pushes_congr)

/-- **The recorded coins as a coin function**: round `k`'s word `i`, zero past either. -/
def recCoin (coins : Got) (k i : ℕ) : F128 := (coins.toList.getD k #[])[i]!

/-- **The recorded relation**: where the program over `OpenM` returns from the record `got`, the run over `PS` from any
log of `got`'s length returns the same, at a log of the new record's length. -/
def RecR {α : Type} (x : OpenM α) (y : PS α) : Prop :=
  ∀ (got : Got) a got', x.run got = .done (a, got') → ∀ l : Array Claims.Msg, l.size = got.size →
    ∃ l', y.go l = (.ok a, l') ∧ l'.size = got'.size

theorem recR_bind {α β : Type} {x : OpenM α} {y : PS α} {f : α → OpenM β} {g : α → PS β} (hxy : RecR x y)
    (hfg : ∀ a, RecR (f a) (g a)) : RecR (x >>= f) (y >>= g) := by
  intro got b got'' h l hl
  rw [StateT.run_bind, step_bind_eq] at h
  rcases hx : x.run got with ⟨a, got1⟩ | e | ⟨c, n, k⟩
  · rw [hx] at h
    simp only [Step.bind] at h
    obtain ⟨l1, hy, hl1⟩ := hxy got a got1 hx l hl
    obtain ⟨l2, hg, hl2⟩ := hfg a got1 b got'' h l1 hl1
    refine ⟨l2, ?_, hl2⟩
    rw [go_bind, hy]
    exact hg
  · rw [hx] at h
    simp [Step.bind] at h
  · rw [hx] at h
    simp [Step.bind] at h

theorem recR_done {α : Type} {x : OpenM α} {y : PS α} (a : α) (hx : ∀ got, x.run got = .done (a, got))
    (hy : ∀ l, y.go l = (.ok a, l)) : RecR x y := by
  intro got a' got' h l hl
  rw [hx] at h
  cases h
  exact ⟨l, hy l, hl⟩

theorem recR_fail {α : Type} {x : OpenM α} {y : PS α} (e : String) (hx : ∀ got, x.run got = .fail e) :
    RecR x y := by
  intro got a' got' h
  rw [hx] at h
  cases h

/-- **The recorded simulation.** -/
def recSim : _root_.Flock.With.Sim OpenM PS where
  R := RecR
  pure a := recR_done a (fun _ => rfl) (fun _ => rfl)
  bind := recR_bind
  throw e := recR_fail e fun _ => rfl
  lift v := by
    cases v with
    | error e => exact recR_fail e fun _ => rfl
    | ok a => exact recR_done a (fun _ => rfl) (fun _ => rfl)

theorem recCoin_of {coins : Got} {k : ℕ} {cs : Array F128} (hk : coins[k]? = some cs) :
    recCoin coins k = fun i => cs[i]! := by
  funext i
  unfold recCoin
  rw [List.getD_eq_getElem?_getD, Array.getElem?_toList, hk, Option.getD_some]

/-- **A recorded round is `asksCut`'s** at the recorded coins: it returns round `k`'s coins, as the cut run returns its
log's. -/
theorem recR_round (coins : Got) (c : ByteArray) :
    RecR ((stepAsks coins false).round c) ((asksCut (recCoin coins) coins.size).round c) := by
  intro got a got' h l hl
  have hcut : ((asksCut (recCoin coins) coins.size).round c).go l =
      ((if coins.size ≤ l.size then throw "past the openings" else pure (recCoin coins l.size) : PS _)).go
        (l.push (Claims.toMsg c)) := rfl
  rw [hcut]
  rcases hk : coins[got.size]? with _ | cs
  · simp only [stepAsks, StateT.run, bind, StateT.bind, get, getThe, MonadStateOf.get, StateT.get, pure, Step.bind,
      hk, throw, throwThe, MonadExceptOf.throw] at h
    first
      | (change Step.fail _ = _ at h; cases h)
      | exact nomatch h
      | cases h
      | simp [Functor.map, Step.bind] at h
  · have hlt : got.size < coins.size := (Array.getElem?_eq_some_iff.1 hk).1
    simp only [stepAsks, StateT.run, bind, StateT.bind, get, getThe, MonadStateOf.get, StateT.get, pure, Step.bind,
      hk, StateT.pure] at h
    by_cases hz : cs.size = (_root_.Flock.u64OfLE c (c.size - 8)).toNat
    · simp only [hz, bne_self_eq_false, Bool.false_eq_true, ite_false, set, StateT.set, MonadStateOf.set,
        StateT.pure, StateT.bind, pure, bind, Step.bind] at h
      try simp only [Step.done.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      simp only [show ¬ coins.size ≤ l.size by omega, ↓reduceIte]
      rw [recCoin_of (hl ▸ hk)]
      exact ⟨_, rfl, by simp [hl]⟩
    · simp only [hz, bne_iff_ne, ne_eq, not_false_eq_true, ite_true, throw, throwThe, MonadExceptOf.throw,
        Function.comp_apply, StateT.lift, StateT.bind, bind, Step.bind] at h
      all_goals simp at h

theorem recR_link (coins : Got) (a b : Array ByteArray) :
    RecR ((stepAsks coins false).link a b) ((asksCut (recCoin coins) coins.size).link a b) :=
  recR_fail _ fun _ => rfl

/-- **The recorded channel is the cut's** at the recorded coins. -/
theorem asksRel_rec (coins : Got) :
    AsksRel recSim (stepAsks coins false).toC (cutAsks (recCoin coins) coins.size).toC :=
  ⟨recR_link coins, recR_round coins⟩

/-! ## The claims' points -/

/-- **The points a rep's claims read**: the zerocheck's `ρ` and `r` and the lincheck's `r_in`, at the coins from round
`L`, as `ZcOutAt` and `LcOutAt` give them. -/
def PtsAt (m kLog : ℕ) (coin : ℕ → ℕ → F128) (L : ℕ) (zc : ZcOut) (lc : LcOut) : Prop :=
  zc.rho = pushes (fun i => coin (L + 3 + i) 0) (m - 6) ∧
    zc.rRest = _root_.Flock.pinned ++ (Array.range (m - 13)).map (coin (L + 1)) ∧
    lc.rIn = (pushes (fun j => coin (L + 3 + (m - 6) + 2 + j) 0) (kLog - 6)).reverse

theorem ptsAt_of {m kLog : ℕ} {coin : ℕ → ℕ → F128} {L : ℕ} {zc : ZcOut} {lc : LcOut}
    (hz : ZcOutAt m (zcCoins coin L) zc) (hl : LcOutAt (kLog - 6) (lcCoins coin (L + 3 + (m - 6)) (kLog - 6)) lc) :
    PtsAt m kLog coin L zc lc :=
  ⟨hz.2.1, hz.2.2, hl.2⟩

/-- The closed-form zerocheck and lincheck at the coins from `L` have their points. -/
theorem ptsAt_out (m : ℕ) (c : _root_.Flock.CircuitFold) (coin : ℕ → ℕ → F128) (L : ℕ) (tr : FlockComplete.Trace) :
    PtsAt m c.kLog coin L (Replay.Zc.out m (zcCoins coin L) tr)
      (Replay.Lc.out c m tr (Replay.Zc.out m (zcCoins coin L) tr) (lcCoins coin (L + 3 + (m - 6)) (c.kLog - 6))) :=
  ⟨rfl, rfl, rfl⟩

/-- **The points read their rounds' first words only.** -/
theorem ptsAt_congr {m kLog : ℕ} {coin coin' : ℕ → ℕ → F128} {L L' : ℕ} {zc : ZcOut} {lc : LcOut}
    (h1 : ∀ i < m - 13, coin (L + 1) i = coin' (L' + 1) i)
    (h3 : ∀ i < m - 6, coin (L + 3 + i) 0 = coin' (L' + 3 + i) 0)
    (h5 : ∀ j < kLog - 6, coin (L + 3 + (m - 6) + 2 + j) 0 = coin' (L' + 3 + (m - 6) + 2 + j) 0)
    (h : PtsAt m kLog coin L zc lc) : PtsAt m kLog coin' L' zc lc := by
  obtain ⟨e1, e2, e3⟩ := h
  refine ⟨e1.trans (pushes_congr h3), e2.trans ?_, e3.trans (by rw [pushes_congr h5])⟩
  congr 1
  apply Array.ext (by simp)
  intro i hi _
  simp only [Array.size_map, Array.size_range] at hi
  simp only [Array.getElem_map, Array.getElem_range]
  exact h1 i hi

theorem claims_x2 {A B e e' : Array _root_.Flock.Claim} (hA : A.size = 2) (hB : B.size = 2) (h₀ : A[0]!.x = B[0]!.x)
    (h₁ : A[1]!.x = B[1]!.x) (he : ∀ k : ℕ, e[k]!.x = e'[k]!.x) : ∀ k : ℕ, (A ++ e)[k]!.x = (B ++ e')[k]!.x
  | 0 => by rw [Replay.get_left (by omega), Replay.get_left (by omega)]; exact h₀
  | 1 => by rw [Replay.get_left (by omega), Replay.get_left (by omega)]; exact h₁
  | k + 2 => by rw [Replay.get_right (by omega), Replay.get_right (by omega), hA, hB, Nat.add_sub_cancel]; exact he k

/-- **Claims at the same points read the same points**: the zerocheck's and lincheck's claims at two outputs with the
same points, then extra claims at the same points. -/
theorem ptsAt_claims_x {m kLog kL : ℕ} {coin : ℕ → ℕ → F128} {L : ℕ} {zc zc' : ZcOut} {lc lc' : LcOut}
    (h : PtsAt m kLog coin L zc lc) (h' : PtsAt m kLog coin L zc' lc') {e e' : Array _root_.Flock.Claim}
    (he : ∀ k : ℕ, e[k]!.x = e'[k]!.x) :
    ∀ k : ℕ, (abcClaims m kL zc lc ++ e)[k]!.x = (abcClaims m kL zc' lc' ++ e')[k]!.x :=
  claims_x2 (abcClaims_size _ _ _ _) (abcClaims_size _ _ _ _)
    (Replay.abc_x (h.1.trans h'.1.symm) (h.2.1.trans h'.2.1.symm) (h.2.2.trans h'.2.2.symm)).1
    (Replay.abc_x (h.1.trans h'.1.symm) (h.2.1.trans h'.2.1.symm) (h.2.2.trans h'.2.2.symm)).2 he

/-- **The switched claims read each claim's seven `r''` and the claims' `γ`** only. -/
theorem sws_coins_congr (claims : Array _root_.Flock.Claim) {cs cs' : Replay.Rs.Coins}
    (hr : ∀ k < claims.size, ∀ i < 7, cs.cr k i = cs'.cr k i) (hg : ∀ k < claims.size, cs.cg k = cs'.cg k) :
    Replay.Rs.sws claims cs = Replay.Rs.sws claims cs' := by
  have hpc : Replay.Rs.perClaim claims cs = Replay.Rs.perClaim claims cs' := by
    unfold Replay.Rs.perClaim
    refine pushes_congr fun k hk => ?_
    have e : cs.rpp k = cs'.rpp k := by
      unfold Replay.Rs.Coins.rpp
      apply Array.ext (by simp)
      intro i hi _
      simp only [Array.size_map, Array.size_range] at hi
      simp only [Array.getElem_map, Array.getElem_range]
      exact hr k hk i hi
    rw [e]
  have hga : Replay.Rs.gamma cs claims.size = Replay.Rs.gamma cs' claims.size := by
    unfold Replay.Rs.gamma
    apply Array.ext (by simp)
    intro i hi _
    simp only [Array.size_map, Array.size_range] at hi
    simp only [Array.getElem_map, Array.getElem_range]
    exact hg i hi
  unfold Replay.Rs.sws
  rw [hpc, hga]

/-! ## The PIOP on recorded coins -/

/-- **A rep's PIOP on recorded coins**: where it returns from no record, its claims are the zerocheck's and lincheck's
at the recorded coins' points (`PtsAt` from round `0`), then the extra claims. -/
theorem repPiop_rec (coins : Got) (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout) (cap root : ByteArray)
    (extras : Array _root_.Flock.Claim) (mk : ℕ → F128) (hm : 14 ≤ st.m) (t : _root_.Flock.Transcript)
    {r : ((_root_.Flock.ZerocheckProof × _root_.Flock.LincheckProof × Array _root_.Flock.Claim) × _root_.Flock.Transcript) × Got}
    (h : ((repPiop (stepAsks coins false) st lay cap root extras mk).run t).run #[] = .done r) :
    ∃ zc lc, r.1.1.2.2 = abcClaims st.m st.kLog zc lc ++ extras ∧
      PtsAt st.m st.fold.kLog (recCoin coins) 0 zc lc := by
  have hR := _root_.Flock.Firewall.Finish.repPiop_rel recSim (asksRel_rec coins) st lay cap root extras mk
  obtain ⟨l', hy, -⟩ := hR t #[] r.1 r.2 h #[] rfl
  obtain ⟨⟨⟨zp, lp, cl⟩, t''⟩, g⟩ := r
  dsimp only at hy ⊢
  unfold repPiop at hy
  obtain ⟨⟨zp', zc⟩, t₁, l₁, h₁, hy⟩ := _root_.Flock.Firewall.Finish.run_bind_ok hy
  dsimp only at hy
  obtain ⟨⟨lp', lc⟩, t₂, l₂, h₂, hy⟩ := _root_.Flock.Firewall.Finish.run_bind_ok hy
  have hy' : (Except.ok ((zp', lp', abcClaims st.m st.kLog zc lc ++ extras), t₂), l₂) =
      ((Except.ok ((zp, lp, cl), t'') : Except String _), l') := hy
  simp only [Prod.mk.injEq, Except.ok.injEq] at hy'
  obtain ⟨⟨⟨-, -, hcl⟩, -⟩, -⟩ := hy'
  have h₁' : ((zerocheckCut (asksCut (recCoin coins) coins.size) st.m st.digest cap root lay mk).run t).go #[] =
      (.ok ((zp', zc), t₁), l₁) := h₁
  have h₂' : ((lincheckCut (asksCut (recCoin coins) coins.size) st.fold 0 lay mk zc).run t₁).go l₁ =
      (.ok ((lp', lc), t₂), l₂) := h₂
  obtain ⟨hz, hl₁⟩ := zerocheckCut_at (recCoin coins) coins.size hm _ _ _ _ _ h₁'
  obtain ⟨hlc, -⟩ := lincheckCut_at (recCoin coins) coins.size _ _ _ _ _ h₂'
  rw [hl₁] at hlc
  exact ⟨zc, lc, hcl.symm, ptsAt_of hz hlc⟩

end FlockComplete.LigLock.Fed
