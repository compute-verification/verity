import Proofs.Flock.EndToEnd.Joint.MT.Draw
import Proofs.Flock.EndToEnd.Period.Run
import Proofs.Flock.Soundness.Discharge.ZkHidden.PublicJ
import Proofs.Flock.Soundness.Assumptions.Zk
import Flock.Desc

/-!
# The joint stratified bound over sampling blocks of several tables (trusted)

`JointStratified` and `JointStratifiedHorizon` (`Joint/Statement.lean`) at sampling blocks proved in `J` tables in
parallel. A sampling block is one hidden-output `--zk` session of `J` tables over `n` proof units, whose units the
verifier draws by one live stratified draw (`Flock.Draw.drawOS` at `.stratified`, `Law.execOS`): stratum `s` holds
`n_s` of the sampling block's units, `k_s` of them are drawn uniformly without replacement, independently across strata,
and the strata run over every table's units, so a table's subcircuits are strata of the one draw. `verify --zk
--session-tables J` proves the drawn units in `J` equal consecutive parts, table `j` part `j` (`drawJ`, `partS`); when
the units are laid out table by table and every table's strata draw the same number of units, part `j` is table `j`'s
own stratified draw. The record's custody is `RecordCustodyZKJS` (`live-verifier` at the stratified law); that the
parts are proved and set up with the session's link points is flock-verify's own arithmetic
(`ZkHidden.drawOkZKHJ_of_accepts_stratified`).

`StratifiedBlockMT` is such a sampling block, with every binder of `EndToEndHidden` but the prover and its law, as
`Period`'s `SamplingBlock` is. Its wrong units are `wrongRegZ` in the committed transcript fixed at registration
(`XplurZC`). The bound is `JointStratified`'s, with the same budget and the same confidence set:

* `JointStratifiedMT`, per sampling block: except with probability `δ + ε_ks + ε_bind`, if flock-verify accepts, the
  sampling block's wrong units `B` lie in the confidence set (`Law.stratified`'s escape of `B` is at least `δ`), so they
  spend at most the one budget, `Σ_s k_s·x_s/n_s ≤ ln(1/δ)` over every stratum of every table, and their ANDs and their
  output bits are each at most the most any set in the confidence set holds (`StratifiedBlockMT.maxHarm`).
* `JointStratifiedHorizonMT`, pooled: the same over a period of `T` such sampling blocks played by one adaptive prover,
  with one budget for the whole period: except with probability `δ + Σ_t (ε_ks,t + ε_bind,t)`.

`ε_ks` is the sampling block's knowledge term at the worst registration (`StratifiedBlockMT.epsKs`) and `ε_bind` its
binding term `δ_link` (`StratifiedBlockMT.link`). The named assumptions are A3 (`uniform/io-getrandombytes`) for the
draw's bytes, `live-verifier` (`StratifiedBlockMT.hCust`, `RecordCustodyZKJS`), and `cr/sha-512` (`TableCRZ`) and
`ecr/sha-512` (`LinkCRZC`) at the strategy each sampling block is played with (in a period, at every strategy it is
played with on some run, `StratifiedMT.Plays`), stated flat. Their proofs are `Flock.SecurityProofs.JointStratifiedMT`
and `Flock.SecurityProofs.JointStratifiedHorizonMT` (`Joint/MT/Bound.lean`).

C-Flock's definitions are still in its proofs package, so, like `EndToEndHidden`, this module imports them.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **A sampling block of `J` tables whose draw is stratified**: one hidden-output `--zk` session of `nTab I` tables over
`n` proof units, with every binder of `EndToEndHidden` but the prover and its law. Its statement (circuit descriptor,
public file); its strata over every table's units, unit `u` in stratum `sub u`, `ks s` of stratum `s`'s units drawn, the
executable strata `St` describing them (`Law.ExecStrata`); its record (draw files, session writer, reference setup at
table 0, `∑ s, ks s / nTab I` drawn units a table); the verifier's byte sources, one per stratum; the verify command's
tags; `live-verifier` at the stratified law (`hCust`); the count curve's parameters; and the knowledge finders' budgets
`qF`, `qS`. -/
structure StratifiedBlockMT where
  n : ℕ
  Reg : Type
  I : Inputs
  ht : I.tags.typed = false
  D : Flock.Desc
  hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D
  hU : Built D.unroll.unit
  pub : Public
  hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub
  hpt : pub.tables = none
  hscope : Discharge.Layout.scopeOkH D.unroll = true
  m : ℕ
  sub : Fin n → Fin m
  ks : Fin m → ℕ
  hks : ∀ s, ks s ≤ (Law.stratum sub s).card
  St : Fin m → Flock.Draw.Stratum
  hS : Law.ExecStrata sub ks St
  Kd : ℕ
  pop : ℕ
  dj : Finset (Fin n) → Lean.Json
  S₀ : Finset (Fin n)
  d₀ : Lean.Json
  y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g ((∑ s, ks s) / nTab I)) S₀ d₀
  wr : (S : Finset (Fin n)) → (R : Reg) →
    LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g ((∑ s, ks s) / nTab I)) →
      Discharge.Exec.Transcript
  dg : ByteArray → ByteArray
  N : ℕ
  Ω : Type
  [fintypeΩ : Fintype Ω]
  [nonemptyΩ : Nonempty Ω]
  Lb : Fin m → ℕ
  src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256
  hTags : TagsOkZHJ I
  hCust : Assumptions.Zk.RecordCustodyZKJS I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr Kd
    (Array.ofFn St) pop
  k : ℕ
  Rw : ℕ
  M : ℕ
  hk : 1 ≤ k
  hRw : 1 ≤ Rw
  hM : 1 ≤ M
  ρ : ℝ
  hρ : ρ < 1
  hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ
  t' : ℝ
  ht' : 0 ≤ t'
  qF : ℝ
  qS : ℝ

attribute [instance] StratifiedBlockMT.fintypeΩ StratifiedBlockMT.nonemptyΩ

namespace StratifiedBlockMT

open Classical

variable (w : StratifiedBlockMT)

/-- The sampling block's draw: `Flock.Draw.drawOS`'s stratified draw on the verifier's byte sources, over every
table's units. -/
def L : Law w.n := Law.execOS (n := w.n) w.src (.stratified w.Kd (Array.ofFn w.St)) w.pop

/-- The stratified law the draw is held to under A3 (`Law.execOS_escape_le`): `ks s` of stratum `s`'s units, uniform
without replacement, independently across strata. -/
def ideal : Law w.n := Law.stratified w.sub w.ks w.hks

/-- The sampling block's audit as it runs: `EndToEndHidden`'s game at the stratified draw. -/
noncomputable def game : Game (w.Reg × Finset (Fin w.n) × Discharge.Exec.Transcript) :=
  auditLive w.L w.Reg (liveOfZJ w.dg (planZHJ w.dj w.y₀) w.N w.wr)

/-- The key program's partition into proof units. -/
noncomputable def P := (keyProg w.D.unroll w.hU w.n).partition (Discharge.Layout.outsOf w.D.unroll w.hU w.n)

/-- The public file's commitments. -/
noncomputable def vb :=
  (rsZHJ w.dj w.y₀ w.ht (Flock.Desc.parse_flat w.hD) w.hscope w.hU w.hpub w.hpt (Reg := w.Reg)).bindingZ
  (rsZHJ_computes w.dj w.y₀ w.ht (Flock.Desc.parse_flat w.hD) w.hscope w.hU w.hpub w.hpt (Reg := w.Reg))

/-- The model audit's strategy that a live strategy `τ` refines to: `EndToEndHidden`'s `σ`, with the draw at each
accepted outcome from custody at the stratified law (`drawOkZKHJ_of_accepts_stratified`). -/
noncomputable def sim (τ : Strategy w.game) :=
  simZJ execArith hs512 (planZHJ w.dj w.y₀) w.I w.wr (opnAtZHJ (planZHJ w.dj w.y₀) w.I w.dj) w.dg
    (decAtZJ (planZHJ w.dj w.y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) w.N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ w.I w.wr _ _ _ _ _
      (runsZCJH_liveA w.I w.ht (Flock.Desc.parse_flat w.hD) (Discharge.Layout.scopeOkH_spec w.hU w.hscope).2.1 w.dj
        w.y₀ w.wr w.L w.hTags w.hCust.1
        fun ω R o hacc => drawOkZKHJ_of_accepts_stratified w.I w.ht (Flock.Desc.parse_flat w.hD) w.sub w.ks w.hks
          w.St w.hS w.Kd w.pop w.Lb w.src w.dj w.hCust.2 w.hCust.1 ω R o hacc)) τ

/-- **The sampling block's wrong units** at the strategy `τ`: `wrongRegZ` in the committed transcript fixed at
registration (`XplurZC`), as in `EndToEndHidden`. -/
noncomputable def wrongSet (τ : Strategy w.game) : Finset (Fin w.n) :=
  wrongRegZ execArith w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb (reg (w.sim τ))
    (XplurZC execArith hs512 w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb (reg (w.sim τ)) (cont (w.sim τ)) w.k w.Rw)

/-- The knowledge term `ε_ks`, `ksAvgStrictZ` at the worst registration. -/
noncomputable def epsKs : ℝ≥0∞ := ⨆ R : w.Reg, ksAvgStrictZ (planZHJ w.dj w.y₀) w.L w.k w.qF w.qS R

/-- The binding term `ε_bind`, `δ_link = Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))`, as in `EndToEndHidden`. -/
noncomputable def link : ℝ≥0∞ :=
  ENNReal.ofReal ((∑ p : w.vb.Pos,
      (Fintype.card (ΩcZ execArith w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb w.L p) : ℝ) / Fintype.card w.L.Ω) *
    (2 * (w.t' * (1 + w.k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * w.M) + w.k / (Real.exp 1 * w.Rw)) / (1 - w.ρ))

/-- **The detection budget a wrong-unit set `B` spends**, `Σ_s k_s·x_s/n_s` over every stratum of every table, with
`x_s` its units in stratum `s`. -/
noncomputable def spend (B : Finset (Fin w.n)) : ℝ :=
  ∑ s, (w.ks s : ℝ) * (B.filter (w.sub · = s)).card / (Law.stratum w.sub s).card

/-- **The most harm in the confidence set**: the largest total weight `Σ_{u ∈ B} h u` over the wrong-unit sets `B` the
stratified draw misses with probability at least `δ`. -/
noncomputable def maxHarm (h : Fin w.n → ℕ) (δ : ℝ) : ℝ :=
  ⨆ B : {B : Finset (Fin w.n) // ENNReal.ofReal δ ≤ w.ideal.escape B}, ((∑ u ∈ B.1, h u : ℕ) : ℝ)

/-- **`B` is within the sampling block's one budget at `δ`**: the stratified draw misses it with probability at least
`δ`, so it spends at most `ln(1/δ)` of the detection budget across every stratum of every table, and its ANDs and its
output bits are each at most the most any set in the confidence set holds. -/
def Within (δ : ℝ) (ands bits : Fin w.n → ℕ) (B : Finset (Fin w.n)) : Prop :=
  ENNReal.ofReal δ ≤ w.ideal.escape B ∧ w.spend B ≤ Real.log (1 / δ) ∧
    ((∑ u ∈ B, ands u : ℕ) : ℝ) ≤ w.maxHarm ands δ ∧ ((∑ u ∈ B, bits u : ℕ) : ℝ) ≤ w.maxHarm bits δ

end StratifiedBlockMT

/-- **The joint stratified bound, per sampling block of `J` tables.** For a sampling block as in `StratifiedBlockMT`,
proved in `nTab I` tables, whose one live stratified draw takes `k_s` of the `n_s` units of each stratum `s` across the
tables, every `δ > 0` and every per-unit weights `ands` (each unit's ANDs) and `bits` (its output bits), and every live
prover `τ`, the probability that flock-verify accepts and the sampling block's wrong units `B` are not within its one
budget is at most
$$\delta + \varepsilon_{ks} + \varepsilon_{bind}:$$
outside it, an accepted sampling block's wrong units escape the stratified draw with probability at least `δ`, spend
`Σ_s k_s·x_s/n_s ≤ ln(1/δ)` across all its strata, and hold at most `maxHarm ands δ` ANDs and at most
`maxHarm bits δ` output bits. Its named assumptions: A3 (`Assumptions.UniformRandomBytes`) for the draw's bytes,
`live-verifier` (`StratifiedBlockMT.hCust`, `RecordCustodyZKJS`), and `cr/sha-512` (`TableCRZ`) and `ecr/sha-512`
(`LinkCRZC`) at `τ`. -/
def JointStratifiedMT : Prop :=
  ∀ (w : StratifiedBlockMT) (δ : ℝ) (ands bits : Fin w.n → ℕ), 0 < δ →
    Assumptions.UniformRandomBytes w.Lb w.src →
    ∀ τ : Strategy w.game,
    (∀ ω j, TableCRZ execArith hs512 (planZHJ w.dj w.y₀ (w.L.draw ω) (reg (w.sim τ))) j
      (zcStrat execArith hs512 (planZHJ w.dj w.y₀) (cont (w.sim τ) ω)) w.qF w.qS) →
    LinkCRZC execArith hs512 w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb (reg (w.sim τ)) (cont (w.sim τ)) w.k
      w.Rw w.M w.t' →
    prob (fun o => LiveAcceptsZK w.I o ∧ ¬ w.Within δ ands bits (w.wrongSet τ)) w.game τ ≤
      ENNReal.ofReal δ + w.epsKs + w.link

/-- What a period's run records of sampling block `idx`: whether flock-verify accepted it, its wrong units
(`StratifiedBlockMT.wrongSet`), and whether a drawn unit is one of them. -/
structure BlockRecMT (W : ℕ → StratifiedBlockMT) where
  idx : ℕ
  accepted : Prop
  wrong : Finset (Fin (W idx).n)
  drawnWrong : Prop

namespace BlockRecMT

variable {W : ℕ → StratifiedBlockMT}

/-- The probability that the record's sampling block's stratified draw misses its wrong units. -/
noncomputable def escape (r : BlockRecMT W) : ℝ≥0∞ := (W r.idx).ideal.escape r.wrong

/-- The detection budget the record's wrong units spend, `Σ_s k_s·x_s/n_s` over its sampling block's strata. -/
noncomputable def spend (r : BlockRecMT W) : ℝ := (W r.idx).spend r.wrong

/-- The record's wrong units' total weight `Σ_{u ∈ B} h u`, with `h t` sampling block `t`'s per-unit weights. -/
def harm (h : (t : ℕ) → Fin (W t).n → ℕ) (r : BlockRecMT W) : ℕ := ∑ u ∈ r.wrong, h r.idx u

end BlockRecMT

namespace StratifiedBlockMT

open Classical

/-- Sampling block `t`'s record, at the strategy `τ` that played it and its output `o`. -/
noncomputable def record (W : ℕ → StratifiedBlockMT) (t : ℕ) (τ : Strategy (W t).game)
    (o : (W t).Reg × Finset (Fin (W t).n) × Discharge.Exec.Transcript) : BlockRecMT W :=
  ⟨t, LiveAcceptsZK (W t).I o, (W t).wrongSet τ, ∃ u ∈ o.2.1, u ∈ (W t).wrongSet τ⟩

end StratifiedBlockMT

namespace StratifiedMT

/-- **A period of `T` sampling blocks of `J` tables**, sampling block `t`'s audit after sampling block `t − 1`'s,
against one prover. -/
noncomputable def game (W : ℕ → StratifiedBlockMT) (T : ℕ) : Game Unit := periodG (fun t => (W t).game) 0 T

/-- The period's run at the prover strategy `s`: each sampling block's record, in order. -/
noncomputable def run (W : ℕ → StratifiedBlockMT) (T : ℕ) (s : Strategy (game W T)) : Game (List (BlockRecMT W)) :=
  periodRun (fun t => (W t).game) (StratifiedBlockMT.record W) 0 T s

/-- **The strategies the period plays sampling block `t` with**: `τ` plays sampling block `t` on some run of the period
at `s`. -/
def Plays (W : ℕ → StratifiedBlockMT) (T : ℕ) (s : Strategy (game W T)) (t : ℕ) (τ : Strategy (W t).game) : Prop :=
  PeriodPlays (fun t => (W t).game) 0 T s t τ

/-- **The most harm in the period's confidence set**: the largest total weight over every choice of a wrong-unit set
in each of the run's sampling blocks that their draws miss together with probability at least `δ`. -/
noncomputable def maxHarm (W : ℕ → StratifiedBlockMT) (h : (t : ℕ) → Fin (W t).n → ℕ) (δ : ℝ)
    (l : List (BlockRecMT W)) : ℝ :=
  ⨆ B : {B : (i : Fin l.length) → Finset (Fin (W l[i].idx).n) //
      ENNReal.ofReal δ ≤ ∏ i, (W l[i].idx).ideal.escape (B i)},
    ((∑ i, ∑ u ∈ B.1 i, h l[i].idx u : ℕ) : ℝ)

/-- **The run's wrong units are within the period's one budget at `δ`**: the sampling blocks' draws miss them together
with probability at least `δ`, so they spend at most `ln(1/δ)` of the detection budget across every stratum of every
sampling block, and their ANDs and their output bits in all are each at most the most any choice in the period's
confidence set holds. -/
def Within (W : ℕ → StratifiedBlockMT) (δ : ℝ) (ands bits : (t : ℕ) → Fin (W t).n → ℕ) (l : List (BlockRecMT W)) :
    Prop :=
  ENNReal.ofReal δ ≤ (l.map BlockRecMT.escape).prod ∧ (l.map BlockRecMT.spend).sum ≤ Real.log (1 / δ) ∧
    (((l.map (BlockRecMT.harm ands)).sum : ℕ) : ℝ) ≤ maxHarm W ands δ l ∧
    (((l.map (BlockRecMT.harm bits)).sum : ℕ) : ℝ) ≤ maxHarm W bits δ l

end StratifiedMT

/-- **The joint stratified bound over a period of sampling blocks of `J` tables, pooled.** In a period of `T` sampling
blocks `W t` as in `StratifiedBlockMT`, each proved in its session's tables with a fresh live stratified draw, played by
one adaptive prover `s`, for every `δ > 0` and every per-unit weights `ands t` (each unit's ANDs) and `bits t` (its
output bits), the probability that flock-verify accepts every sampling block and their wrong units are not within the
period's one budget is at most
$$\delta + \sum_t (\varepsilon_{ks,t} + \varepsilon_{bind,t}):$$
outside it, when every sampling block is accepted, their draws miss their wrong units together with probability at
least `δ`, the wrong units spend `Σ_t Σ_s k_{t,s}·x_{t,s}/n_{t,s} ≤ ln(1/δ)` across every stratum of every sampling
block, and their ANDs and output bits in all are at most the period's `maxHarm`. The sampling term is `δ` however many
sampling blocks and tables the period has. Its named assumptions, at every sampling block: A3
(`Assumptions.UniformRandomBytes`) for its draw's bytes, `live-verifier` (`StratifiedBlockMT.hCust`,
`RecordCustodyZKJS`), and `cr/sha-512` (`TableCRZ`) and `ecr/sha-512` (`LinkCRZC`) at every strategy it is played with
on some run (`StratifiedMT.Plays`). -/
def JointStratifiedHorizonMT : Prop :=
  ∀ (W : ℕ → StratifiedBlockMT) (T : ℕ) (δ : ℝ) (ands bits : (t : ℕ) → Fin (W t).n → ℕ), 0 < δ →
    (∀ t < T, Assumptions.UniformRandomBytes (W t).Lb (W t).src) →
    ∀ s : Strategy (StratifiedMT.game W T),
    (∀ t (τ : Strategy (W t).game), StratifiedMT.Plays W T s t τ → ∀ ω j,
      TableCRZ execArith hs512 (planZHJ (W t).dj (W t).y₀ ((W t).L.draw ω) (reg ((W t).sim τ))) j
        (zcStrat execArith hs512 (planZHJ (W t).dj (W t).y₀) (cont ((W t).sim τ) ω)) (W t).qF (W t).qS) →
    (∀ t (τ : Strategy (W t).game), StratifiedMT.Plays W T s t τ →
      LinkCRZC execArith hs512 (W t).P (planZHJ (W t).dj (W t).y₀) (tabZHJ (W t).dj (W t).y₀) (W t).vb
        (reg ((W t).sim τ)) (cont ((W t).sim τ)) (W t).k (W t).Rw (W t).M (W t).t') →
    value (fun l => (∀ r ∈ l, r.accepted) ∧ ¬ StratifiedMT.Within W δ ands bits l) (StratifiedMT.run W T s) ≤
      ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ((W t).epsKs + (W t).link)

end Flock.Guarantees
