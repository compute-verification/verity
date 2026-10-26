import Definitions.Accounting.Guarantees.Compose
import Definitions.Nci.Remote

/-!
# PoUW's credit, window by window over a monitoring period: the bridge to the remote dampener

`Nci.Guarantees.DampenerRemotePeriod` takes PoUW's leg per window: over a period of `T` windows on live coins, window
`t`'s coins `Ω t` uniform and independent of the others', window `t`'s certificate is wrong (`wrong t`: accepted with
the window's time below `(1 − ε)·T_cred`) with probability at most `p` over its own coins, at every value `ξ` of the
others'. `CircuitCreditPerWindow` says that leg follows when the deployment's window `t`, at each value of the other
windows' coins, is a play `CircuitCredit` covers, certified by `cert t ξ`, whose failure `wrong t` implies
(`CreditPlayOn`).

The deployment has no Lean object of its own a period of windows could be read off. `Accounting.PeriodRun`
(`JointPeriod`) is one period on product coins, each audit on its own coordinate, but its PoUW claim is device work at a
price (`c.W ≤ work p ξ`) on the device schedule, with an abstract verdict, and it holds no `Play`, `Declared` circuit or
`Anatomy`; `Accounting.Compose.Period` (`ComposePeriod`) is PoUW's own sequential period game, which credits live words
over an accepted prefix, not time. So the period here is the smallest object the dampener reads, `(T, Ω, wrong)`, and a
window is tied to `CircuitCredit` by `CreditPlayOn`:

* the window's coins carry the play's coins and others (`Ω t ≃ Paths … × R`, `R` the coins of the window that the play
  doesn't read: those of the other protocols drawn with it), and the play, its proof run and its folded words may
  depend on the value of `R`, as they may on `ξ`; every premise of `CircuitCredit` holds of the play at each value;
* the certificate is exposed: it is `cert t ξ`, the `certOf … .live` that `CircuitCredit` concludes, so a composer can
  tie its `eps` and `credit` to the dampener's `ε_W` and `c t`;
* `wrong t` implies the certificate's failure, not the converse: a block's verdict is PoUW's and the other strata's,
  so the window accepts less often than the play.

A window that carries no PoUW play (a finer cut's coordinate holding only another protocol's coins) is never wrong.
`DampenerRemotePeriod` doesn't read the certificate: its `c t`, `ε_W` and `W` are its own, and tying them to `cert`
restates its residual premise (`W ξ` the accepted windows' credits, `ε_W` at least each window's `eps`, `wrong t` the
certificate's failure), which is more than a parameter, so the tie stays with whoever composes the two.

Reviewer: compute-accounting. The proof, `Accounting.SecurityProofs.Compose.CircuitCreditPerWindow`, is
`Accounting.SecurityProofs.Compose.CircuitCredit` at each value of `R`'s play and a change of measure:
`Nci.Dampener.prob` under the uniform law on `Ω t` is the card ratio `Pouw.RelativeSecurity.prob` takes, an event of the
window's coins inside the play's failure pulled back along `Ω t ≃ Paths … × R` has at most its probability, and that
is the mean over `R` of each value's (`Pouw.RelativeSecurity.prob_le_of_slices`). What no proof here gives is that the
deployment's window is such a play:

* `CircuitCredit` speaks of a declared circuit; nothing builds a `Declared` from a compiled Program, and
  `Properties.Compose.UnitsOfLayout`, which would tie its units, commit coins and draw to the service's, is unproved.
* The salt is the play's coin 0, a coordinate of its path (`Paths C Cd.rounds`), not of `R`: it comes from its own
  salt-only live session (#4405), released after `X` (#4573). Saying so is #4363's `SaltIsLiveCoin`
  (`Properties.Pouw.Period`), which the deployment's tie must establish.
* `Ω t ≃ Paths … × R` is exact only in the model of independent uniform streams that
  `FlockSoundness.Audit.Law.drawOn_escape_le` uses: the tiles are one stratum of the single stratified draw (one
  64-byte draw coin, one SHA-512 stream per stratum), so the deployment's tie must name that assumption.
* `Pl.game = serviceGame … Kd` draws a uniform `K_d`-subset of all `N_p` tiles, but the service draws the work law
  inside the joint draw (one stratum per width `K`, `k_s = ⌈K·w_s·n_s/W⌉`: `strata` and `law` in
  `verity/core/protocols/pouw/credit.py`), so a window with two widths, or any window in the joint draw, can't meet it.
  `CircuitCredit`'s `p = 3 · 2⁻⁴⁰` is therefore not yet a served figure. The lane decided provisionally (26 Oct) to
  restate the remote dampener's per-window legs over `Flock.Guarantees.JointStratified` (#4560 stated, #4590 proved),
  the bound for the draw the service runs.
* `Scope`'s gaps: `LayerCommitted`, and `Declared.Live` (H1's salt round is seeded until #4405 lands).
-/

namespace Accounting.Guarantees.Compose

open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **A `CircuitCredit` play on coins `Ω'` certified by `c`, whose failure `E` implies, at `δ`.** There are a declared
circuit `Cd`, its scheme `Sc`, sheet `Sh`, model `Z`, honest prover `H`, tiles `N_p`, draw size `K_d`, stages `W`, the
coins `R` of `Ω'` the play doesn't read (`e : Ω' ≃ Paths … × R`), at each value `r` of them a play `Pl r`, proof run
`pr r` and folded words `folded r`, certificate `lam`, rates and terms, and device window `win`, such that every
premise of `CircuitCredit` holds of them at each `r`, `c` is the certificate `CircuitCredit` concludes
(`certOf … .live`), and `E` at the coins `a` implies that certificate's failure at `a`'s play: `Pl (e a).2` accepts
`(e a).1` with time below `(1 − ε)·T_cred`. Its probability bound, `δ_s + δ_bind + δ_ks`, is at most `δ`. -/
def CreditPlayOn (Ω' : Type) [Fintype Ω'] (E : Ω' → Prop) (δ : ℝ) (c : Accounting.Compose.Certificate) : Prop :=
  ∃ (ι K P T L U F X C Ξ V : Type) (_ : Fintype K) (_ : Fintype P) (_ : DecidableEq P) (_ : Fintype T)
    (_ : Fintype L) (_ : Fintype U) (_ : Fintype F) (_ : Fintype X) (_ : Nonempty F) (_ : Nonempty X)
    (_ : Fintype C) (_ : Fintype Ξ) (_ : Nonempty V)
    (Sc : Scheme K) (Cd : Declared ι) (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X) (H : Honest K)
    (Np Kd : ℕ) (_ : Cd.units ≤ Np) (hr : 0 < Cd.rounds) (hK : Kd ≤ Np) (W : SplitStages C (Fin Np) Cd.rounds)
    (R : Type) (_ : Fintype R)
    (Pl : R → Play (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pr : R → ProofRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V)
    (folded : R → Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 → ℝ) (lam : SheetRow P → ℝ)
    (ρ p p0 wmax Λ δs δbind δks : ℝ) (win : Accounting.Window)
    (e : Ω' ≃ Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 × R),
    (∀ r : R, Scope Sc Cd δs ∧
      0 < ρ ∧ (∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) ∧
      (∀ j, stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ wmax) ∧
      (∀ t : Fin Np, 0 < shareCone Sc Cd Np t → tileLuck Sc Cd Np t ≤ p ∧ tileLuck0 Sc Cd Np t ≤ p0) ∧
      0 < Λ ∧ SplitRate Kd ρ p p0 (aOf Sc Cd Z lam) (bOf Sc Cd Z lam) wmax Λ ∧
      (Pl r).game = serviceGame (Ξ := Ξ) (tileStage Cd Np) W (tileCone Sc Cd Np) 1 1 Kd ∧
      (Cd.layer hr hK (pr r)).Of (Pl r) ∧ (∀ w, prob (fun x => (Cd.layer hr hK (pr r)).MovedIn x w) ≤ δbind) ∧
      (Cd.layer hr hK (pr r)).JointSound δks ∧ (Cd.layer hr hK (pr r)).Pins ∧
      PerTile Sc Cd W (fun k G _ => OperandHMu (Sc.mu k) G) ∧
      PerTile Sc Cd W (fun k G _ => ReceiptLuckAt (Sc.rB k) G) ∧
      PerTile Sc Cd W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0) ∧
      UnitFixed (Pl r) (Cd.layer hr hK (pr r)) (tileMode Cd W) (fun _ t => (tileWords Sc Cd Np t : ℝ)) (folded r) ∧
      Fits Sc Cd Sh Z ∧ ModelScope Sc Cd Z lam ∧ (∀ x, ((Pl r).run x).Split) ∧
      (∀ x, SheetClock Sh ((Pl r).run x).run) ∧ (∀ x, SheetDispatch ((Pl r).run x).run) ∧
      (∀ x, SheetTensor Sh ((Pl r).run x).run) ∧ (∀ x, SheetPipe Sh ((Pl r).run x).run) ∧
      (∀ x, SheetSmem Sh ((Pl r).run x).run) ∧ (∀ x, SheetWords Z ((Pl r).run x)) ∧
      (∀ x, SheetSemantics Z ((Pl r).run x)) ∧ (∀ x, FoldOptFoldedAt Z ((Pl r).run x) (folded r x) lam) ∧
      (∀ x, FillOpt Z ((Pl r).run x))) ∧
    c = certOf Sc Cd Z H lam (rOf Sc Cd Np (aOf Sc Cd Z lam) (bOf Sc Cd Z lam) Λ δs) (δs + (δbind + δks)) win
      CoinMode.live ∧
    (∀ a, E a → (Pl (e a).2).accepts (e a).1 ∧ ((Pl (e a).2).run (e a).1).run.time < (1 - c.eps) * c.credit) ∧
    c.delta ≤ δ

/-- **PoUW's credit, window by window** (the bridge to `Nci.Guarantees.DampenerRemotePeriod`'s PoUW leg). A
monitoring period of `T` windows, window `t` with its own uniform coins `Ω t` (a window of the deployment, or a finer
cut's coordinate where its PoUW and sampled-proofs coins are drawn at different times), `cert t ξ` window `t`'s
certificate at the coins `ξ`, and `wrong t` a failure of it. If, at every value `ξ` of the other windows' coins,
window `t` is a `CircuitCredit` play certified by `cert t ξ` whose failure `wrong t` implies, at `p ≥ 0`
(`CreditPlayOn`, `p ≥ δ_s + δ_bind + δ_ks`, `3 · 2⁻⁴⁰` at the deployed point for a uniform `K_d`-subset draw, which is
not the served window's), or carries no PoUW play and is never wrong, then `DampenerRemotePeriod`'s PoUW hypothesis
holds: over window `t`'s own coins (`ξ[t ↦ a]`, `a` uniform on `Ω t`), its certificate is wrong with probability at
most `p`. The play and its certificate may depend on the other windows' coins, earlier ones (the prover's adaptivity)
and later ones alike: the bound holds at each of their values. A composer's `cert t` reads no coin of window `t`'s;
the bound doesn't need it to. -/
def CircuitCreditPerWindow : Prop :=
  ∀ (T : ℕ) (Ω : Fin T → Type) [∀ t, Fintype (Ω t)] [∀ t, Nonempty (Ω t)]
    (wrong : Fin T → ((t : Fin T) → Ω t) → Prop)
    (cert : Fin T → ((t : Fin T) → Ω t) → Accounting.Compose.Certificate) (p : ℝ),
    0 ≤ p →
    (∀ t (ξ : (t : Fin T) → Ω t),
      CreditPlayOn (Ω t) (fun a => wrong t (Function.update ξ t a)) p (cert t ξ) ∨
        ∀ a, ¬ wrong t (Function.update ξ t a)) →
    ∀ t (ξ : (t : Fin T) → Ω t),
      Nci.Dampener.prob (PMF.uniformOfFintype (Ω t)) (fun a => wrong t (Function.update ξ t a)) ≤ p

end Accounting.Guarantees.Compose
