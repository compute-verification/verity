import Proofs.Flock.Bridge.Statement

/-!
# The bridge's lemmas, as statements (targets, not proved)

Each is a `def … : Prop` over `Bridge.Defs` and `Bridge.Deployed`, so a proof plugs into `SampledKSLive` as stated:

* `FaultyRightTarget` (M1's core): a unit whose committed wires each carry, in some opening of their registered commit
  string, their value in the evaluation on `Xr`'s inputs (`RightRegZ`'s second conjunct), and each of whose registered
  openings at a wire the constants `κ` don't fix opens its wire's claims position under `Cm` (`TransportsLive`'s
  conclusion), and whose evaluation is `κ`'s constant at each constant wire, is not faulty in `witDep Xr`. So
  `faulty ⊆ wrongRegZ`.
* `PartBlocksTarget`: `partOf P κ` has `n` slots, one per unit, each the unit's proof block with the input rows it owns.
* `PaTarget` (M4): `designOf kd`'s `P_a` on `partOf P κ` is the subset miss `C(n − |B|, kd)/C(n, kd)` at a set of
  population slots.
* `CountTarget` (M4): outside `RelS` at `δ = C(n − K₀, kd)/C(n, kd)`, at least `min K₀ (n − kd + 1)` population slots
  are faulty.
* `AssemblyTarget` (M4): `TransportsLive`, those and `EndToEndHidden` give `SampledKSLive`.
* `RelDeployTarget` (rel_deploy): at the deployed sampling record (`SamplingDeploy.deployOf`) of a verifier's code
  whose layout is `commitDep`'s (`LayoutIs`) and which parses the statement's bytes to `stmtOf` and partitions it to
  `partOf`, a `witDep` outside `RelS` at SHA-512 and `commitDep` is outside the record's `R`.
-/

namespace Flock.Bridge

open FlockSoundness.Audit

/-- **M1's core**: a unit whose committed wires each carry, in some opening of their registered commit string, their
value in `C`'s evaluation on `Xr`'s inputs, each of whose registered openings at a wire the constants `κ` don't fix
opens its wire's position under `Cm` and `roots`, and whose evaluation is `κ`'s constant at each constant wire it
reads, is not faulty in `witDep C Cm H roots Xr` at the statement at constants `κ`. -/
def FaultyRightTarget : Prop :=
  ∀ {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) {Pos Val Opn Com : Type}
    (pos : Fin C.N → Pos) (bit : Val → Fin C.N → Bool) (val : Opn → Val) (com : Opn → Com) (cs : Pos → Com)
    (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.Commitment Bool) (roots : List Claims.Msg)
    (Xr : Fin C.N → Bool) (κ : Fin C.N → Option Bool) (u : Fin n),
    (∃ op : Pos → Opn, ∀ g ∈ P.io u, com (op (pos g)) = cs (pos g) ∧ bit (val (op (pos g))) g = C.eval Xr g) →
    (∀ g ∈ P.io u, κ g = none → ∀ o : Opn, com o = cs (pos g) →
      ∃ d : Claims.Msg, Cm.Opens H roots ⟨posOf C g, bit (val o) g, d⟩ = true) →
    (∀ g ∈ P.io u, ∀ b, κ g = some b → C.eval Xr g = b) →
    ∀ (kd : ℕ) (desc : Claims.Msg) (q : Claims.Sampling.Query),
      u.val ∉ Claims.Sampling.faulty H Cm (stmtOf P κ kd desc q roots) (partOf P κ) (witDep C Cm H roots Xr)

/-- `partOf P κ` has one slot per unit. -/
def PartBlocksTarget : Prop :=
  ∀ {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool),
    (partOf P κ).blocks.length = n

/-- **The Pa lemma**: on `partOf P κ`, `designOf kd` misses a set `B` of slots with probability
`C(n − |B|, kd)/C(n, kd)`. -/
def PaTarget : Prop :=
  ∀ {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    (kd : ℕ) (B : Finset ℕ), B ⊆ Finset.range n →
    (designOf kd).Pa (partOf P κ) B = ((n - B.card).choose kd : ℝ) / (n.choose kd : ℝ)

/-- **Outside `RelS` is a count**: at `δ = C(n − K₀, kd)/C(n, kd)`, `1 ≤ kd ≤ n`, a witness outside `RelS` has at
least `min K₀ (n − kd + 1)` faulty population slots, slots below `n`. -/
def CountTarget : Prop :=
  ∀ {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (kd K₀ : ℕ), 1 ≤ kd → kd ≤ n →
    ∀ (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.Commitment Bool) (κ : Fin C.N → Option Bool)
      (desc : Claims.Msg) (q : Claims.Sampling.Query) (roots : List Claims.Msg) (w : Claims.Sampling.Wit),
      ¬ Claims.Sampling.RelS H Cm (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)) (stmtOf P κ kd desc q roots)
          (partOf P κ) w →
      min K₀ (n - kd + 1) ≤
        ((Claims.Sampling.faulty H Cm (stmtOf P κ kd desc q roots) (partOf P κ) w).filter (· < n)).card

/-- **The assembly**: the registered openings opening the deployed commitment, M1's core, the slot count and the count
lemma turn `EndToEndHidden` into `SampledKSLive`. -/
def AssemblyTarget : Prop :=
  TransportsLive → FaultyRightTarget → PartBlocksTarget → CountTarget → Flock.Guarantees.EndToEndHidden →
    Flock.Guarantees.SampledKSLive

/-- **rel_deploy**: at the deployed sampling record of a verifier's code whose layout is `commitDep`'s, and which
parses `x` to `stmtOf` and partitions it to `partOf`, a `witDep` outside `RelS` at SHA-512 and `commitDep` is outside
the record's `R`. -/
def RelDeployTarget : Prop :=
  ∀ {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (cd : SamplingDeploy.Code)
    {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (rows : Fin C.N → Option RowAt) (c : Flock.Circuit) (nl kd : ℕ) (κ : Fin C.N → Option Bool) (desc : Claims.Msg)
    (q : Claims.Sampling.Query) (roots : List Claims.Msg) (Xr : Fin C.N → Bool) (δ : ℝ) (x : Claims.Msg),
    LayoutIs C rows c nl roots cd →
    cd.parse x = some (stmtOf P κ kd desc q roots) → cd.partition (stmtOf P κ kd desc q roots) = some (partOf P κ) →
    ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg (commitDep C rows c nl roots) δ (stmtOf P κ kd desc q roots)
      (partOf P κ) (witDep C (commitDep C rows c nl roots) Claims.Sampling.sha512Msg roots Xr) →
    ¬ Claims.Sampling.R Claims.Sampling.sha512Msg (SamplingDeploy.deployOf F cd).commit cd.parse cd.partition δ x
      (witDep C (commitDep C rows c nl roots) Claims.Sampling.sha512Msg roots Xr)

end Flock.Bridge
