import Proofs.Flock.C3.RegLevels.RsT
import Proofs.Flock.C3.Opening.SessionCoins
import Proofs.Flock.VBridge.Algebra.Sim

/-!
# n7's levels: the verifier's α is the run's (`SessionAt`'s coef tie)

`SessionAt` asks that each opening's `coef` be `RecCoins.coefAt` at `o.kappa` and eq(`ops.alpha`); the verifier
registers it at its own α (`checkStackedLevel`), the points of `RecCoins.opening`'s Ligerito on `stacked/rep0`'s coins.
This is the algebra that makes the two one. `opening_lg`: `RecCoins.opening` ran ring switching on the symbolic
messages and the claims `coinClaim` builds, from round 0 of `stacked/rep0`'s coins, then Ligerito from the absorbed
`T`, and `o.lg` is Ligerito's points. `lg_eq_of_opening`: ring switching's switched claims and next round depend only
on the coins and the claims' points (`ringSwitch_spec`), and Ligerito's points not on the messages (`ligerito_sim` at
`Lin.eval`), so a run on the messages w holds, from round 0 of the same coins, at claims with the verifier's points,
has the verifier's points. That the run's rounds are `stacked/rep0`'s (`hcust`) is the custody, a hypothesis here.
`alpha_of_opening` and `coef_of_opening` are the tie at any `ops` whose α is the run's.
-/

namespace FlockC3.RegLevels

/-- **What `RecCoins.opening` ran**: ring switching on the symbolic messages and the claims `coinClaim` builds, from
round 0 of `stacked/rep0`'s coins, then Ligerito from the absorbed `T` on its output, whose points are `o.lg`. -/
theorem opening_lg {c3 : Flock.RecCoins.C3} {unit : String} {o : Flock.RecCoins.Opening}
    (h : Flock.RecCoins.opening c3 unit = .ok o) :
    ∃ (c : Array (Array Flock.F128)) (as bs : Flock.RecAlgebra.AClaim Flock.RecAlgebra.Lin)
      (rsr : Array Flock.RecAlgebra.Lin) (rs : Flock.RecAlgebra.RsPts Flock.RecAlgebra.Lin) (k₂ k₁ : ℕ)
      (lgr : Array Flock.RecAlgebra.Lin),
      c3.coins unit s!"{Flock.RecVStar.STACKED}/rep0" = .ok c ∧
      FlockC3.Opening.coinClaim c3 unit o.sh 0 = .ok as ∧ FlockC3.Opening.coinClaim c3 unit o.sh 1 = .ok bs ∧
      Flock.RecAlgebra.ringSwitch (Flock.RecAlgebra.symbolic o.sh) #[as, bs] 0 (Flock.RecAlgebra.Coins.ofRounds c) =
        .ok ((rsr, rs), k₂) ∧
      Flock.RecAlgebra.ligerito o.sh.sched (Flock.RecAlgebra.symbolic o.sh)
        ((Flock.RecAlgebra.symbolic o.sh).tAtom.getD rs.T) rs.sws k₂ (Flock.RecAlgebra.Coins.ofRounds c) =
        .ok ((lgr, o.lg), k₁) := by
  unfold Flock.RecCoins.opening at h
  cases hr : c3.stacked with
  | none => rw [hr] at h; cases h
  | some r =>
    rw [hr] at h
    dsimp only at h
    cases hsh : Flock.RecOpening.shape? r.m r.claims with
    | none => rw [hsh] at h; cases h
    | some sh =>
      rw [hsh] at h
      dsimp only at h
      split at h
      · cases h
      · obtain ⟨cs, hcs, h⟩ := FlockVBridge.PartS.bind_ok h
        obtain ⟨a, b, ha, hb, rfl⟩ := FlockC3.Opening.mapM_two hcs
        obtain ⟨c, hc, h⟩ := FlockVBridge.PartS.bind_ok h
        have h := FlockC3.Opening.mapError_ok h
        obtain ⟨⟨lins, lg⟩, hrun, h⟩ := FlockVBridge.PartS.bind_ok h
        obtain ⟨⟨_res, _bs⟩, _hbuild, h⟩ := FlockVBridge.PartS.bind_ok h
        split at h
        · cases h
        · split at h
          · cases h
          cases h
          obtain ⟨k₁, hx⟩ := FlockC3.Opening.run'_ok hrun
          obtain ⟨⟨rsr, rs⟩, k₂, h₁, hx⟩ := Flock.RecAlgebra.cm_bind hx
          obtain ⟨⟨lgr, lg'⟩, k₃, h₂, hx⟩ := Flock.RecAlgebra.cm_bind hx
          cases Flock.RecAlgebra.cm_pure hx
          exact ⟨c, a, b, rsr, rs, k₂, _, lgr, hc, ha, hb, h₁, h₂⟩

/-- **A run from round 0 of the verifier's coins has the verifier's Ligerito points**: ring switching on the messages
w holds, at claims with the points `coinClaim` builds (any values), from round 0 of `stacked/rep0`'s coins (`hcust`),
switches the same claims and ends at the same round (`ringSwitch_spec`), and Ligerito from the absorbed `T` on them is
`RecCoins.opening`'s under `Lin.eval` (`ligerito_sim`), whose points are the symbolic run's. -/
theorem lg_eq_of_opening {c3 : Flock.RecCoins.C3} {unit : String} {o : Flock.RecCoins.Opening}
    (hop : Flock.RecCoins.opening c3 unit = .ok o) {w : Array Flock.F128} {cons : ℕ → ℕ → Flock.F128}
    {a : Array (Flock.RecAlgebra.AClaim Flock.F128)} {c : Array (Array Flock.F128)} {rsr lgr : Array Flock.F128}
    {rs : Flock.RecAlgebra.RsPts Flock.F128} {k₂ k₁ : ℕ} {lg : Flock.RecAlgebra.LgPts}
    (hcust : c3.coins unit s!"{Flock.RecVStar.STACKED}/rep0" = .ok c)
    (hx : ∀ as bs, FlockC3.Opening.coinClaim c3 unit o.sh 0 = .ok as →
      FlockC3.Opening.coinClaim c3 unit o.sh 1 = .ok bs →
      ∃ v : Fin 2 → Flock.F128, a = #[⟨v 0, as.skip, as.x⟩, ⟨v 1, bs.skip, bs.x⟩])
    (h₁ : Flock.RecAlgebra.ringSwitch (FlockC3.Opening.decodedO o.sh w cons) a 0 (Flock.RecAlgebra.Coins.ofRounds c) =
      .ok ((rsr, rs), k₂))
    (h₂ : Flock.RecAlgebra.ligerito o.sh.sched (FlockC3.Opening.decodedO o.sh w cons)
      ((FlockC3.Opening.decodedO o.sh w cons).tAtom.getD rs.T) rs.sws k₂ (Flock.RecAlgebra.Coins.ofRounds c) =
      .ok ((lgr, lg), k₁)) :
    lg = o.lg := by
  obtain ⟨c', as, bs, rsr', rs', k₂', k₁', lgr', hc', has, hbs, h₁', h₂'⟩ := opening_lg hop
  rw [hcust] at hc'
  obtain rfl := Except.ok.inj hc'
  obtain ⟨v, rfl⟩ := hx as bs has hbs
  have hs2 : ∀ {β : Type} (x y : β), (#[x, y]).size = 2 := fun _ _ => rfl
  obtain ⟨_betas, -, -, hg, hk, -, hsw, hj⟩ := Flock.RecAlgebra.ringSwitch_spec (by rw [hs2]; omega) h₁
  obtain ⟨_betas', -, -, hg', hk', -, hsw', hj'⟩ := Flock.RecAlgebra.ringSwitch_spec (by rw [hs2]; omega) h₁'
  rw [hs2] at hg hg' hk hk' hsw hsw'
  have hsws : rs.sws = rs'.sws := by
    refine Array.ext (hsw.trans hsw'.symm) fun j h1 h2 => ?_
    have hj2 : j < 2 := by omega
    obtain ⟨c₁, hc₁, -, -, hs₁⟩ := hj j (by rw [hs2]; omega)
    obtain ⟨c₂, hc₂, -, -, hs₂⟩ := hj' j (by rw [hs2]; omega)
    obtain rfl : c₁ = c₂ := Except.ok.inj (hc₁.symm.trans hc₂)
    have e1 := (Array.getElem?_eq_getElem h1).symm.trans hs₁
    have e2 := (Array.getElem?_eq_getElem h2).symm.trans hs₂
    rw [Except.ok.inj (hg.symm.trans hg')] at e1
    rcases (by omega : j = 0 ∨ j = 1) with rfl | rfl
    · exact Option.some.inj (e1.trans e2.symm)
    · exact Option.some.inj (e1.trans e2.symm)
  have ht : (FlockC3.Opening.decodedO o.sh w cons).tAtom.getD rs.T =
      Flock.RecAlgebra.Lin.eval (FlockC3.Opening.valO o.sh w cons)
        ((Flock.RecAlgebra.symbolic o.sh).tAtom.getD rs'.T) := rfl
  rw [ht, hsws, hk.trans hk'.symm] at h₂
  have hsim := Flock.RecAlgebra.ligerito_sim (Flock.RecAlgebra.Lin.eval_hom (FlockC3.Opening.valO o.sh w cons))
    o.sh.sched (Flock.RecAlgebra.symbolic o.sh) _ rs'.sws k₂' _ _ _ h₂'
  exact congrArg (fun q : (Array Flock.F128 × Flock.RecAlgebra.LgPts) × ℕ => q.1.2)
    (Except.ok.inj (h₂.symm.trans hsim))

/-- **The verifier's α is the run's** (`SessionAt`'s coef tie, its algebra), at any `ops` whose α is the points' of a
run from round 0 of `stacked/rep0`'s coins (`hcust`) on the messages w holds at `ops`' consistency sums, at claims with
the verifier's points (`lg_eq_of_opening`). -/
theorem alpha_of_opening {c3 : Flock.RecCoins.C3} {unit : String} {o : Flock.RecCoins.Opening}
    (hop : Flock.RecCoins.opening c3 unit = .ok o) {sh : Flock.Shape} (hosh : o.sh = sh) {w : Array Flock.F128}
    {ops : FlockVBridge.AccChain.Openings} {a : Array (Flock.RecAlgebra.AClaim Flock.F128)}
    {c : Array (Array Flock.F128)} {rsr lgr : Array Flock.F128} {rs : Flock.RecAlgebra.RsPts Flock.F128} {k₂ k₁ : ℕ}
    {lg : Flock.RecAlgebra.LgPts} (hcust : c3.coins unit s!"{Flock.RecVStar.STACKED}/rep0" = .ok c)
    (hx : ∀ as bs, FlockC3.Opening.coinClaim c3 unit sh 0 = .ok as → FlockC3.Opening.coinClaim c3 unit sh 1 = .ok bs →
      ∃ v : Fin 2 → Flock.F128, a = #[⟨v 0, as.skip, as.x⟩, ⟨v 1, bs.skip, bs.x⟩])
    (h₁ : Flock.RecAlgebra.ringSwitch (msgsO sh w ops) a 0 (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((rsr, rs), k₂))
    (h₂ : Flock.RecAlgebra.ligerito sh.sched (msgsO sh w ops) ((msgsO sh w ops).tAtom.getD rs.T) rs.sws k₂
      (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((lgr, lg), k₁))
    (halpha : ∀ ℓ < sh.sched.levels.size, ops.alpha ℓ = (lg.conss[ℓ]!).alpha) :
    ∀ ℓ < sh.sched.levels.size, (o.lg.conss[ℓ]!).alpha = ops.alpha ℓ := by
  subst hosh
  rw [← lg_eq_of_opening hop hcust hx h₁ h₂]
  exact fun ℓ hℓ => (halpha ℓ hℓ).symm

/-- **`SessionAt`'s coef tie** at such `ops`: an opening's `coef` as the verifier registers it, at its own α
(`checkStackedLevel`), is the one at `ops`' α. -/
theorem coef_of_opening {c3 : Flock.RecCoins.C3} {unit : String} {o : Flock.RecCoins.Opening}
    (hop : Flock.RecCoins.opening c3 unit = .ok o) {sh : Flock.Shape} (hosh : o.sh = sh) {w : Array Flock.F128}
    {ops : FlockVBridge.AccChain.Openings} {a : Array (Flock.RecAlgebra.AClaim Flock.F128)}
    {c : Array (Array Flock.F128)} {rsr lgr : Array Flock.F128} {rs : Flock.RecAlgebra.RsPts Flock.F128} {k₂ k₁ : ℕ}
    {lg : Flock.RecAlgebra.LgPts} (hcust : c3.coins unit s!"{Flock.RecVStar.STACKED}/rep0" = .ok c)
    (hx : ∀ as bs, FlockC3.Opening.coinClaim c3 unit sh 0 = .ok as → FlockC3.Opening.coinClaim c3 unit sh 1 = .ok bs →
      ∃ v : Fin 2 → Flock.F128, a = #[⟨v 0, as.skip, as.x⟩, ⟨v 1, bs.skip, bs.x⟩])
    (h₁ : Flock.RecAlgebra.ringSwitch (msgsO sh w ops) a 0 (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((rsr, rs), k₂))
    (h₂ : Flock.RecAlgebra.ligerito sh.sched (msgsO sh w ops) ((msgsO sh w ops).tAtom.getD rs.T) rs.sws k₂
      (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((lgr, lg), k₁))
    (halpha : ∀ ℓ < sh.sched.levels.size, ops.alpha ℓ = (lg.conss[ℓ]!).alpha)
    (hreg : ∀ ℓ < sh.sched.levels.size, ∀ j < ops.q ℓ, ops.coef ℓ j =
      Flock.RecCoins.coefAt (o.kappa[ℓ]!) (FlockVBridge.AccChain.levelLanes sh ℓ)
        (Flock.eqTable (o.lg.conss[ℓ]!).alpha)[j]!) :
    ∀ ℓ < sh.sched.levels.size, ∀ j < ops.q ℓ, ops.coef ℓ j =
      Flock.RecCoins.coefAt (o.kappa[ℓ]!) (FlockVBridge.AccChain.levelLanes sh ℓ) (Flock.eqTable (ops.alpha ℓ))[j]! :=
  fun ℓ hℓ j hj => by rw [hreg ℓ hℓ j hj, alpha_of_opening hop hosh hcust hx h₁ h₂ halpha ℓ hℓ]

end FlockC3.RegLevels
