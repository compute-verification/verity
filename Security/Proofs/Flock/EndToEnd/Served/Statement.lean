import Proofs.Flock.EndToEnd.Joint.Statement
import Proofs.Flock.EndToEnd.Served.Draw
import SampledProofs.Draw

/-!
# The served sampling block's draw is a `StratifiedBlock` play (trusted)

The sampled-proofs service draws a sampling block's units once, after the block is committed and registered: the
challenger, the auditor's device on a machine the prover doesn't control, reads its coins from the Lean verifier of
record's `IO.getRandomBytes` (`verifier.coins`, A3, `uniform/io-getrandombytes`), and `Select.select` draws the units
from them by `SampledProofs.Draw.drawUnits`, one stream read in order, stratum after stratum (`challenger.py`,
`coins.draw_first`, which the Python holds to the Lean's draw). `JointStratified` is stated for the model's live draw
(`Law.execOS`), which reads one tape per stratum from a uniform source.

`StratifiedBlock.served` is a sampling block with its draw's source the served one: the challenger's coins, a finite
family of `Lc` bytes over a finite space `Ωc`, cut into one tape per stratum where each stratum stops reading
(`Served.tapes`), with spare bytes after each segment that no draw ending on the coins reads.

* `ServedDraw`: on coins uniform by A3, the served tapes are uniform (A3 at the tapes, which `JointStratified` takes),
  and wherever the served draw on the coins ends, the model's draw on the tapes is the same set of units.
* `JointServed`: so `JointStratified`'s bound holds at the served sampling block, with A3 at the challenger's coins in
  place of A3 at the tapes; its other named assumptions are `JointStratified`'s.
* `ServedDraw_honest`, `ServedDraw_nontrivial`, `JointServed_honest`, `JointServed_nontrivial`: the honest and
  nontrivial lemmas.

The coins are the challenger's, drawn before the epoch and committed under hm96 leaves; the game draws them after the
prover registers (`auditLive`: the registration, then the coin), so the theorems take them independent of the prover's
commitments, which holds with the challenger on a host the prover doesn't control, releasing a coin only after its
root (`trusted-host`), and the hm96 leaves hiding the coins perfectly. A draw that runs out of coins is a visible
failure the service never retries, so no accepted draw reads a spare byte. `Select.select` replays the bytes the draw
read, the coins' first bytes (`coins.read_reason`), on which `drawUnits` draws what it draws on the whole family.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.HiddenDec
open scoped ENNReal

namespace StratifiedBlock

/-- **The sampling block as served**: `w`'s session, strata and record, its draw read from the challenger's coins
`coins` over `Ωc`, `Lc` bytes, cut into one tape per stratum (`Served.tapes`), the spare bytes uniform. -/
noncomputable def served (w : StratifiedBlock) (Ωc : Type) [Fintype Ωc] [Nonempty Ωc] (Lc : ℕ)
    (coins : Ωc → Fin Lc → Fin 256) : StratifiedBlock :=
  { w with
    Ω := Ωc × Served.Pads w.m Lc
    fintypeΩ := inferInstance
    nonemptyΩ := inferInstance
    Lb := fun _ => Lc
    src := fun ω => Served.tapes w.St (coins ω.1) ω.2 }

/-- **`JointStratified`'s bound at the sampling block `w`**: for every `δ > 0`, every per-unit weights and every live
prover `τ` at whose strategy `cr/sha-512` (`TableCRZ`) and `ecr/sha-512` (`LinkCRZC`) hold, flock-verify accepts and
`w`'s wrong units are outside its one budget with probability at most `δ + ε_ks + ε_bind`. -/
def Bound (w : StratifiedBlock) : Prop :=
  ∀ (δ : ℝ) (ands bits : Fin w.n → ℕ), 0 < δ →
    ∀ τ : Strategy w.game,
    (∀ ω j, TableCRZ execArith hs512 (planAtZKH w.dj w.y₀ (w.L.draw ω) (reg (w.sim τ))) j
      (zcStrat execArith hs512 (planAtZKH w.dj w.y₀) (cont (w.sim τ) ω)) w.qF w.qS) →
    LinkCRZC execArith hs512 w.P (planAtZKH w.dj w.y₀) (tabAtZKH w.dj w.y₀) w.vb (reg (w.sim τ)) (cont (w.sim τ)) w.k
      w.Rw w.M w.t' →
    prob (fun o => LiveAcceptsZK w.I o ∧ ¬ w.Within δ ands bits (w.wrongSet τ)) w.game τ ≤
      ENNReal.ofReal δ + w.epsKs + w.link

end StratifiedBlock

/-- **The served stratified draw is a play of the live draw on uniform tapes.** For every stratified law (`K`, the
strata `St`) over a population of `pop` units, read as `n` units, and every finite family of the challenger's coins
`coins` over `Ωc`, `Lc` bytes, uniform by A3 (`Assumptions.UniformRandomBytes`, `uniform/io-getrandombytes`, one
stream): the served tapes (`Served.tapes`: each stratum's segment of the coins, then uniform spare bytes) are uniform by
A3 too, and at every outcome whose coins the served draw (`SampledProofs.Draw.drawUnits`) draws `units` from, the live
draw on the tapes (`Law.execOS`, `StratifiedBlock.L`'s) draws exactly `units`. -/
def ServedDraw : Prop :=
  ∀ (n m : ℕ) (St : Fin m → Flock.Draw.Stratum) (K pop : ℕ) (Ωc : Type) [Fintype Ωc] [Nonempty Ωc] (Lc : ℕ)
    (coins : Ωc → Fin Lc → Fin 256),
    Assumptions.UniformRandomBytes (fun _ : Fin 1 => Lc) (fun ω _ => coins ω) →
    Assumptions.UniformRandomBytes (fun _ : Fin m => Lc)
        (fun ω : Ωc × Served.Pads m Lc => Served.tapes St (coins ω.1) ω.2) ∧
      ∀ (ω : Ωc × Served.Pads m Lc) (units : Array ℕ) (read : ℕ),
        SampledProofs.Draw.drawUnits (.stratified K (Array.ofFn St)) pop (Law.streamOf (coins ω.1)) =
          .ok (units, read) →
        (Law.execOS (n := n) (fun ω : Ωc × Served.Pads m Lc => Served.tapes St (coins ω.1) ω.2)
          (.stratified K (Array.ofFn St)) pop).draw ω = Finset.univ.filter fun u => u.val ∈ units

/-- **The joint stratified bound at a served sampling block.** For every sampling block `w` as in `StratifiedBlock`
and every finite family of the challenger's coins `coins` over `Ωc`, `Lc` bytes, uniform by A3
(`uniform/io-getrandombytes`, one stream), the sampling block as served (`StratifiedBlock.served`) draws, wherever the
served draw on the coins (`SampledProofs.Draw.drawUnits`) draws `units`, exactly `units`, and meets `JointStratified`'s
bound (`StratifiedBlock.Bound`): except with probability `δ + ε_ks + ε_bind`, if flock-verify accepts, its wrong units
are within its one budget. Its named assumptions are A3 at the coins, `live-verifier` (`StratifiedBlock.hCust`), and
`cr/sha-512` (`TableCRZ`) and `ecr/sha-512` (`LinkCRZC`) at the prover's strategy, `Bound`'s hypotheses. -/
def JointServed : Prop :=
  ∀ (w : StratifiedBlock) (Ωc : Type) [Fintype Ωc] [Nonempty Ωc] (Lc : ℕ) (coins : Ωc → Fin Lc → Fin 256),
    Assumptions.UniformRandomBytes (fun _ : Fin 1 => Lc) (fun ω _ => coins ω) →
    (∀ (ω : Ωc × Served.Pads w.m Lc) (units : Array ℕ) (read : ℕ),
      SampledProofs.Draw.drawUnits (.stratified w.Kd (Array.ofFn w.St)) w.pop (Law.streamOf (coins ω.1)) =
        .ok (units, read) →
      (w.served Ωc Lc coins).L.draw ω = Finset.univ.filter fun u => u.val ∈ units) ∧
    (w.served Ωc Lc coins).Bound

namespace Served

/-- The honest instance's one stratum: units `[0, 2)`, one of them drawn. -/
def honestStratum : Flock.Draw.Stratum := { template := "unit", units := #[(0, 2)], k := 1 }

/-- The honest instance's strata: the one stratum. -/
def honestSt : Fin 1 → Flock.Draw.Stratum := fun _ => honestStratum

/-- The honest instance's coins: one read of the stratum, `2 · 1 · width 2 + 64` bytes. -/
def honestLc : ℕ := honestStratum.need

/-- The honest coins' value the honest draw is computed at: every byte `0`. -/
def honestZeros : Fin honestLc → Fin 256 := fun _ => 0

end Served

/-- **`ServedDraw`'s honest lemma**: `ServedDraw`'s hypotheses hold at an explicit instance, and its draw ends there.
The law is one stratum of the units `[0, 2)`, one of them drawn (`Served.honestSt`, `K = 1`, `pop = 2`, read as
`n = 2` units). The challenger's coins are every `honestLc`-byte string (`Ωc`, nonempty, `coins = id`), with
`honestLc = 2 · 1 · width 2 + 64 ≥ 1`. A3 holds at its coins, and on the coins all `0` the served stratified draw
(`SampledProofs.Draw.drawUnits`) ends, having drawn its units. It is a one-stratum miniature of the deployed draw:
`stratified:1` over the units of three templates, drawn by `service.Stream` from the Lean verifier's
`IO.getRandomBytes` after the registration. -/
def ServedDraw_honest : Prop :=
  0 < Served.honestLc ∧
  Assumptions.UniformRandomBytes (fun _ : Fin 1 => Served.honestLc)
    (fun (c : Fin Served.honestLc → Fin 256) _ => c) ∧
  ∃ units read, SampledProofs.Draw.drawUnits (.stratified 1 (Array.ofFn Served.honestSt)) 2
    (Law.streamOf Served.honestZeros) = .ok (units, read)

/-- **`ServedDraw`'s nontrivial lemma**: with a draw check `acc` that accepts every units array in place of the served
draw (`drawUnits … = .ok (units, read)`), `ServedDraw`'s conclusion fails at the honest instance: the live draw on the
tapes is one set at each outcome, not every set the check accepts. -/
def ServedDraw_nontrivial : Prop :=
  ∀ acc : Array ℕ → Prop, (∀ units, acc units) →
    ¬ ∀ (ω : (Fin Served.honestLc → Fin 256) × Served.Pads 1 Served.honestLc) (units : Array ℕ), acc units →
      (Law.execOS (n := 2) (fun ω : (Fin Served.honestLc → Fin 256) × Served.Pads 1 Served.honestLc =>
          Served.tapes Served.honestSt ω.1 ω.2) (.stratified 1 (Array.ofFn Served.honestSt)) 2).draw ω =
        Finset.univ.filter fun u => u.val ∈ units

/-- **`JointServed`'s honest lemma**: at the honest challenger, whose coins are every `Lc`-byte string (`Ωc`, nonempty,
`coins = id`) of any nonzero length, A3 at the coins holds, and so A3 at every served sampling block's tapes, which
`JointStratified` takes. The challenger is the deployed one: `service.Stream` reading the Lean verifier's
`IO.getRandomBytes` after the registration. -/
def JointServed_honest : Prop :=
  ∀ (w : StratifiedBlock) (Lc : ℕ), 0 < Lc →
    Assumptions.UniformRandomBytes (fun _ : Fin 1 => Lc) (fun (c : Fin Lc → Fin 256) _ => c) ∧
    Assumptions.UniformRandomBytes (w.served (Fin Lc → Fin 256) Lc id).Lb (w.served (Fin Lc → Fin 256) Lc id).src

/-- **`JointServed`'s nontrivial lemma**: with a verifier that accepts every output in place of flock-verify, the bound
fails at every served sampling block whose wrong units at `τ` are outside its budget and whose errors sum to less than
`1`. -/
def JointServed_nontrivial : Prop :=
  ∀ (w : StratifiedBlock) (Ωc : Type) [Fintype Ωc] [Nonempty Ωc] (Lc : ℕ) (coins : Ωc → Fin Lc → Fin 256)
    (δ : ℝ) (ands bits : Fin w.n → ℕ) (τ : Strategy (w.served Ωc Lc coins).game)
    (acc : w.Reg × Finset (Fin w.n) × Discharge.Exec.Transcript → Prop), (∀ o, acc o) →
    ¬ (w.served Ωc Lc coins).Within δ ands bits ((w.served Ωc Lc coins).wrongSet τ) →
    ENNReal.ofReal δ + (w.served Ωc Lc coins).epsKs + (w.served Ωc Lc coins).link < 1 →
    ¬ prob (fun o => acc o ∧ ¬ (w.served Ωc Lc coins).Within δ ands bits ((w.served Ωc Lc coins).wrongSet τ))
        (w.served Ωc Lc coins).game τ ≤
      ENNReal.ofReal δ + (w.served Ωc Lc coins).epsKs + (w.served Ωc Lc coins).link

end Flock.Guarantees
