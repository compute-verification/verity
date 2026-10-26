import Definitions.Accounting.Guarantees.Compose
import Definitions.Nci.Remote

/-!
# PoUW's credit, window by window over a monitoring period: the bridge to the remote dampener

`Nci.Guarantees.DampenerRemotePeriod` takes PoUW's leg per window: over a period of `T` windows on live coins, window
`t`'s coins `Ω t` uniform and independent of the others', window `t`'s certificate is wrong (`wrong t`: accepted with
the window's time below `(1 − ε)·T_cred`) with probability at most `p` over its own coins, at every value `ξ` of the
others'. `CircuitCreditPerWindow` says that leg follows when the deployment's window `t`, at each value of the other
windows' coins, is a play `CircuitCredit` covers whose failure is `wrong t` (`CreditPlayOn`).

The deployment has no Lean object of its own a period of windows could be read off. `Accounting.PeriodRun`
(`JointPeriod`) is one period on product coins, each audit on its own coordinate, but its PoUW claim is device work at a
price (`c.W ≤ work p ξ`) on the device schedule, with an abstract verdict, and it holds no `Play`, `Declared` circuit or
`Anatomy`; `Accounting.Compose.Period` (`ComposePeriod`) is PoUW's own sequential period game, which credits live words
over an accepted prefix, not time. So the period here is the smallest object the dampener reads, `(T, Ω, wrong)`, and a
window is tied to `CircuitCredit` by `CreditPlayOn`: the window's coins carry the play's coins (`Ω t ≃ Paths … × R`,
`R` the coins of the window that the play doesn't read: those of the other protocols drawn with it), every premise of
`CircuitCredit` holds of the play, and `wrong t` is its certificate's failure. A window that carries no PoUW play (a
finer cut's coordinate holding only another protocol's coins) is never wrong.

Reviewer: compute-accounting. The proof, `Accounting.SecurityProofs.Compose.CircuitCreditPerWindow`, is
`Accounting.SecurityProofs.Compose.CircuitCredit` at each window's play and a change of measure: `Nci.Dampener.prob`
under the uniform law on `Ω t` is the card ratio `Pouw.RelativeSecurity.prob` takes, and pulling an event of the play's
coins back along `Ω t ≃ Paths … × R` multiplies its count and the space's by `|R|`. What no proof here gives is that the
deployment's window is such a play: `CircuitCredit` speaks of a declared circuit, nothing builds a `Declared` from a
compiled Program, and `Properties.Compose.UnitsOfLayout`, which would tie its units, commit coins and draw to the
service's, is unproved.
-/

namespace Accounting.Guarantees.Compose

open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **A `CircuitCredit` play on coins `Ω'` whose failure is `E`, at `δ`.** There are a declared circuit `Cd`, its scheme
`Sc`, sheet `Sh`, model `Z`, honest prover `H`, tiles `N_p`, draw size `K_d`, stages `W`, play `Pl`, proof run `pr`,
folded words, certificate `lam`, rates and terms, and device window `win`, every premise of `CircuitCredit` holds of
them, and the coins `Ω'` are the play's coins and some others `R` (`e : Ω' ≃ Paths … × R`), such that `E` at the coins
`a` is the failure of the certificate `CircuitCredit` concludes (`certOf … .live`): the play accepts `(e a).1` with
time below `(1 − ε)·T_cred`. Its probability bound, `δ_s + δ_bind + δ_ks`, is at most `δ`. -/
def CreditPlayOn (Ω' : Type) [Fintype Ω'] (E : Ω' → Prop) (δ : ℝ) : Prop :=
  ∃ (ι K P T L U F X C Ξ V : Type) (_ : Fintype K) (_ : Fintype P) (_ : DecidableEq P) (_ : Fintype T)
    (_ : Fintype L) (_ : Fintype U) (_ : Fintype F) (_ : Fintype X) (_ : Nonempty F) (_ : Nonempty X)
    (_ : Fintype C) (_ : Fintype Ξ) (_ : Nonempty V)
    (Sc : Scheme K) (Cd : Declared ι) (Sh : Sheet K P) (Z : Model K (SheetRow P) T L U F X) (H : Honest K)
    (Np Kd : ℕ) (_ : Cd.units ≤ Np) (hr : 0 < Cd.rounds) (hK : Kd ≤ Np) (W : SplitStages C (Fin Np) Cd.rounds)
    (Pl : Play (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 K T L U)
    (pr : ProofRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V)
    (folded : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 → ℝ) (lam : SheetRow P → ℝ)
    (ρ p p0 wmax Λ δs δbind δks : ℝ) (win : Accounting.Window)
    (R : Type) (_ : Fintype R) (e : Ω' ≃ Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 × R),
    (Scope Sc Cd δs ∧
      0 < ρ ∧ (∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) ∧
      (∀ j, stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ wmax) ∧
      (∀ t : Fin Np, 0 < shareCone Sc Cd Np t → tileLuck Sc Cd Np t ≤ p ∧ tileLuck0 Sc Cd Np t ≤ p0) ∧
      0 < Λ ∧ SplitRate Kd ρ p p0 (aOf Sc Cd Z lam) (bOf Sc Cd Z lam) wmax Λ ∧
      Pl.game = serviceGame (Ξ := Ξ) (tileStage Cd Np) W (tileCone Sc Cd Np) 1 1 Kd ∧
      (Cd.layer hr hK pr).Of Pl ∧ (∀ w, prob (fun x => (Cd.layer hr hK pr).MovedIn x w) ≤ δbind) ∧
      (Cd.layer hr hK pr).JointSound δks ∧ (Cd.layer hr hK pr).Pins ∧
      PerTile Sc Cd W (fun k G _ => OperandHMu (Sc.mu k) G) ∧
      PerTile Sc Cd W (fun k G _ => ReceiptLuckAt (Sc.rB k) G) ∧
      PerTile Sc Cd W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0) ∧
      UnitFixed Pl (Cd.layer hr hK pr) (tileMode Cd W) (fun _ t => (tileWords Sc Cd Np t : ℝ)) folded ∧
      Fits Sc Cd Sh Z ∧ ModelScope Sc Cd Z lam ∧ (∀ x, (Pl.run x).Split) ∧
      (∀ x, SheetClock Sh (Pl.run x).run) ∧ (∀ x, SheetDispatch (Pl.run x).run) ∧
      (∀ x, SheetTensor Sh (Pl.run x).run) ∧ (∀ x, SheetPipe Sh (Pl.run x).run) ∧
      (∀ x, SheetSmem Sh (Pl.run x).run) ∧ (∀ x, SheetWords Z (Pl.run x)) ∧ (∀ x, SheetSemantics Z (Pl.run x)) ∧
      (∀ x, FoldOptFoldedAt Z (Pl.run x) (folded x) lam) ∧ (∀ x, FillOpt Z (Pl.run x))) ∧
    let c := certOf Sc Cd Z H lam (rOf Sc Cd Np (aOf Sc Cd Z lam) (bOf Sc Cd Z lam) Λ δs) (δs + (δbind + δks)) win
      CoinMode.live
    (∀ a, E a ↔ Pl.accepts (e a).1 ∧ (Pl.run (e a).1).run.time < (1 - c.eps) * c.credit) ∧ c.delta ≤ δ

/-- **PoUW's credit, window by window** (the bridge to `Nci.Guarantees.DampenerRemotePeriod`'s PoUW leg). A
monitoring period of `T` windows, window `t` with its own uniform coins `Ω t` (a window of the deployment, or a finer
cut's coordinate where its PoUW and sampled-proofs coins are drawn at different times), and `wrong t` the failure of
window `t`'s certificate. If, at every value `ξ` of the other windows' coins, window `t` is a `CircuitCredit` play whose
failure is `wrong t` at `p ≥ 0` (`CreditPlayOn`, `p ≥ δ_s + δ_bind + δ_ks`, `3 · 2⁻⁴⁰` at the deployed point), or
carries no PoUW play and is never wrong, then `DampenerRemotePeriod`'s PoUW hypothesis holds: over window `t`'s own
coins (`ξ[t ↦ a]`, `a` uniform on `Ω t`), its certificate is wrong with probability at most `p`. The play may depend
on the other windows' coins, earlier ones (the prover's adaptivity) and later ones alike: the bound holds at each of
their values. -/
def CircuitCreditPerWindow : Prop :=
  ∀ (T : ℕ) (Ω : Fin T → Type) [∀ t, Fintype (Ω t)] [∀ t, Nonempty (Ω t)]
    (wrong : Fin T → ((t : Fin T) → Ω t) → Prop) (p : ℝ),
    0 ≤ p →
    (∀ t (ξ : (t : Fin T) → Ω t),
      CreditPlayOn (Ω t) (fun a => wrong t (Function.update ξ t a)) p ∨ ∀ a, ¬ wrong t (Function.update ξ t a)) →
    ∀ t (ξ : (t : Fin T) → Ω t),
      Nci.Dampener.prob (PMF.uniformOfFintype (Ω t)) (fun a => wrong t (Function.update ξ t a)) ≤ p

end Accounting.Guarantees.Compose
