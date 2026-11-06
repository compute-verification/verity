import Proofs.Flock.C3.RegLevels.Kappa

/-!
# n7's levels: the coef `checkStackedLevel` registers (`hreg`)

`level_of_check`: an accepted `rec-open-stacked/l<LANES>-h<H>` unit (`RecCoins.checkStackedLevel`) ran
`RecCoins.opening`, its `[l] := ls` filter picks out every level with `LANES` lanes and `H` siblings below its cap,
and its `coef` port passed `checkValue` at the verifier's rows: opening `j`'s `coefAt (o.kappa[l]!) LANES eq(α_l)[j]`,
α the verifier's (`o.lg.conss[l]!`). `hreg_of_check` is `Kappa`'s `hreg` at that level, at any `ops` whose `coef` is
the values the check accepted (`hval`), and `coef_of_check` is `SessionAt`'s coef conjunct there (`alpha_of_opening`).
-/

namespace FlockC3.RegLevels

/-- **What an accepted `rec-open-stacked` unit checked**: `RecCoins.opening` returned `o`, every level of `LANES` lanes
and `H` siblings below its cap is the filter's `l`, and the `coef` port holds the verifier's rows at `l`. -/
theorem level_of_check {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {pub : Flock.Public}
    {own : Array Flock.Registered.Port} {LANES H : ℕ}
    (h : Flock.RecCoins.checkStackedLevel c3 c pub own LANES H = .ok ()) :
    ∃ (o : Flock.RecCoins.RegisteredOpening) (l : ℕ) (what : ℕ → String), Flock.RecCoins.opening c3 c.unit.name = .ok o ∧
      (∀ ℓ < o.sh.sched.levels.size, 2 ^ (o.sh.sched.levels[ℓ]!).lanes = LANES →
        (o.sh.sched.levels[ℓ]!).d - (o.sh.sched.levels[ℓ]!).cap = H → ℓ = l) ∧
      Flock.RecCoins.checkValue c pub own "coef" Flock.RecCoins.COEF_VALUE
        ((Array.range (o.sh.sched.levels[l]!).q).map fun j =>
          Flock.RecCoins.words (Flock.RecCoins.coefAt (o.kappa[l]!) LANES (Flock.eqTable (o.lg.conss[l]!).alpha)[j]!))
        what = .ok () := by
  unfold Flock.RecCoins.checkStackedLevel at h
  dsimp only at h
  obtain ⟨o, hop, h⟩ := FlockVBridge.PartS.bind_ok h
  split at h
  · rename_i l hls
    split at h
    · cases h
    · obtain ⟨_u, hcv, -⟩ := FlockVBridge.PartS.bind_ok h
      refine ⟨o, l, _, hop, fun ℓ hℓ hL hH => ?_, hcv⟩
      have hm : ℓ ∈ [l] := by
        rw [← hls]
        exact List.mem_filter.mpr ⟨List.mem_range.mpr hℓ, by simp only [Bool.and_eq_true, beq_iff_eq]; exact ⟨hL, hH⟩⟩
      exact List.mem_singleton.mp hm
  · cases h

/-- **`hreg` from the check** at the level `ℓ` the unit names: at any `ops` whose level-`ℓ` `coef` is the values the
check accepted (`hval`), opening `j`'s is the verifier's, at its own α. -/
theorem hreg_of_check {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {pub : Flock.Public}
    {own : Array Flock.Registered.Port} {LANES H : ℕ}
    (h : Flock.RecCoins.checkStackedLevel c3 c pub own LANES H = .ok ())
    {o : Flock.RecCoins.RegisteredOpening} (hop : Flock.RecCoins.opening c3 c.unit.name = .ok o) {ℓ : ℕ}
    (hℓ : ℓ < o.sh.sched.levels.size) (hL : FlockVBridge.AccChain.levelLanes o.sh ℓ = LANES)
    (hH : (o.sh.sched.levels[ℓ]!).d - (o.sh.sched.levels[ℓ]!).cap = H) {ops : FlockVBridge.AccChain.Openings}
    (hq : ops.q ℓ = (o.sh.sched.levels[ℓ]!).q)
    (hval : ∀ (e : ℕ → Array Flock.F128) (what : ℕ → String),
      Flock.RecCoins.checkValue c pub own "coef" Flock.RecCoins.COEF_VALUE
        ((Array.range (ops.q ℓ)).map fun j => Flock.RecCoins.words (e j)) what = .ok () →
      ∀ j < ops.q ℓ, ops.coef ℓ j = e j) :
    ∀ j < ops.q ℓ, ops.coef ℓ j = Flock.RecCoins.coefAt (o.kappa[ℓ]!) (FlockVBridge.AccChain.levelLanes o.sh ℓ)
      (Flock.eqTable (o.lg.conss[ℓ]!).alpha)[j]! := by
  obtain ⟨o', l, what, hop', hu, hcv⟩ := level_of_check h
  obtain rfl := Except.ok.inj (hop'.symm.trans hop)
  obtain rfl := hu ℓ hℓ hL hH
  rw [← hq] at hcv
  rw [hL]
  exact hval _ what hcv

/-- **`SessionAt`'s coef tie from the check**: at the level the unit names, an `ops` whose `coef` is the values the
check accepted (`hval`) and whose α is the run's (`halpha`, `alpha_of_opening`) has `coef` at its own α. -/
theorem coef_of_check {c3 : Flock.RecCoins.C3} {c : Flock.Circuit} {pub : Flock.Public}
    {own : Array Flock.Registered.Port} {LANES H : ℕ}
    (h : Flock.RecCoins.checkStackedLevel c3 c pub own LANES H = .ok ())
    {o : Flock.RecCoins.RegisteredOpening} (hop : Flock.RecCoins.opening c3 c.unit.name = .ok o) {sh : Flock.Shape}
    (hosh : o.sh = sh) {w : Array Flock.F128} {ops : FlockVBridge.AccChain.Openings}
    {a : Array (Flock.RecAlgebra.AClaim Flock.F128)} {cs : Array (Array Flock.F128)} {rsr lgr : Array Flock.F128}
    {rs : Flock.RecAlgebra.RsPts Flock.F128} {k₂ k₁ : ℕ} {lg : Flock.RecAlgebra.LgPts}
    (hcust : c3.coins c.unit.name s!"{Flock.RecVStar.STACKED}/rep0" = .ok cs)
    (hx : ∀ as bs, FlockC3.Opening.coinClaim c3 c.unit.name sh 0 = .ok as →
      FlockC3.Opening.coinClaim c3 c.unit.name sh 1 = .ok bs →
      ∃ v : Fin 2 → Flock.F128, a = #[⟨v 0, as.skip, as.x⟩, ⟨v 1, bs.skip, bs.x⟩])
    (h₁ : Flock.RecAlgebra.ringSwitch (msgsO sh w ops) a 0 (Flock.RecAlgebra.CoinSource.ofRounds cs) = .ok ((rsr, rs), k₂))
    (h₂ : Flock.RecAlgebra.ligerito sh.sched (msgsO sh w ops) ((msgsO sh w ops).tAtom.getD rs.T) rs.sws k₂
      (Flock.RecAlgebra.CoinSource.ofRounds cs) = .ok ((lgr, lg), k₁))
    (halpha : ∀ ℓ < sh.sched.levels.size, ops.alpha ℓ = (lg.conss[ℓ]!).alpha) {ℓ : ℕ}
    (hℓ : ℓ < sh.sched.levels.size) (hL : FlockVBridge.AccChain.levelLanes sh ℓ = LANES)
    (hH : (sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap = H) (hq : ops.q ℓ = (sh.sched.levels[ℓ]!).q)
    (hval : ∀ (e : ℕ → Array Flock.F128) (what : ℕ → String),
      Flock.RecCoins.checkValue c pub own "coef" Flock.RecCoins.COEF_VALUE
        ((Array.range (ops.q ℓ)).map fun j => Flock.RecCoins.words (e j)) what = .ok () →
      ∀ j < ops.q ℓ, ops.coef ℓ j = e j) :
    ∀ j < ops.q ℓ, ops.coef ℓ j = Flock.RecCoins.coefAt (o.kappa[ℓ]!) (FlockVBridge.AccChain.levelLanes sh ℓ)
      (Flock.eqTable (ops.alpha ℓ))[j]! := by
  subst hosh
  intro j hj
  rw [hreg_of_check h hop hℓ hL hH hq hval j hj, alpha_of_opening hop rfl hcust hx h₁ h₂ halpha ℓ hℓ]

end FlockC3.RegLevels
