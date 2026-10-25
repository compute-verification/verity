import Proofs.Flock.Bridge.Statement

/-!
# The bridge's lemmas, as statements (targets, not proved)

Each is a `def … : Prop` over `Bridge.Defs`, so a proof plugs into `SampledKSLive` as stated:

* `FaultyRightTarget` (M1's core): a unit whose committed wires each carry, in some opening of their registered commit
  string, their value in the evaluation on `X`'s inputs (`RightRegZ`'s second conjunct) is not faulty in `witOf X`
  under `commitOf`. So `faulty ⊆ wrongRegZ`.
* `PartRusTarget`: `partOf P` has `n` slots.
* `PaTarget` (M4): `designOf kd`'s `P_a` on `partOf P` is the subset miss `C(n − |B|, kd)/C(n, kd)`.
* `CountTarget` (M4): outside `RelS` at `δ = C(n − K₀, kd)/C(n, kd)`, at least `min K₀ (n − kd + 1)` units are
  faulty.
* `AssemblyTarget` (M4): those and `EndToEndHidden` give `SampledKSLive`.
* `RelDeployTarget` (rel_deploy): at a sampling record that parses the statement's bytes to `stmtOf` and partitions it
  to `partOf`, and whose commitment passes only openings `commitOf` passes, outside `RelS` is outside the record's `R`.
-/

namespace Flock.Bridge

open FlockSoundness.Audit

/-- **M1's core**: a unit whose committed wires each carry, in some opening of their registered commit string, their
value in `C`'s evaluation on `X`'s inputs is not faulty in `witOf C X` under `commitOf`. -/
def FaultyRightTarget : Prop :=
  ∀ {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) {Pos Val Opn Com : Type} (pos : Fin C.N → Pos)
    (bit : Val → Fin C.N → Bool) (val : Opn → Val) (com : Opn → Com) (cs : Pos → Com) (Xr : Fin C.N → Bool)
    (u : Fin n),
    (∃ op : Pos → Opn, ∀ g ∈ P.io u, com (op (pos g)) = cs (pos g) ∧ bit (val (op (pos g))) g = C.eval Xr g) →
    ∀ (H : Claims.Msg → Claims.Msg) (kd : ℕ) (desc : Claims.Msg) (q : Claims.Sampling.Query) (roots : List Claims.Msg),
      u.val ∉ Claims.Sampling.faulty H (commitOf pos bit val com cs) (stmtOf P kd desc q roots) (partOf P) (witOf C Xr)

/-- `partOf P` has one slot per unit. -/
def PartRusTarget : Prop :=
  ∀ {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n), (partOf P).rus.length = n

/-- **The Pa lemma**: on `partOf P`, `designOf kd` misses a set `B` of slots with probability `C(n − |B|, kd)/C(n, kd)`. -/
def PaTarget : Prop :=
  ∀ {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (kd : ℕ) (B : Finset ℕ), B ⊆ Finset.range n →
    (designOf kd).Pa (partOf P) B = ((n - B.card).choose kd : ℝ) / (n.choose kd : ℝ)

/-- **Outside `RelS` is a count**: at `δ = C(n − K₀, kd)/C(n, kd)`, `1 ≤ kd ≤ n`, a witness outside `RelS` has at
least `min K₀ (n − kd + 1)` faulty units. -/
def CountTarget : Prop :=
  ∀ {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (kd K₀ : ℕ), 1 ≤ kd → kd ≤ n →
    ∀ (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.Commitment Bool) (desc : Claims.Msg)
      (q : Claims.Sampling.Query) (roots : List Claims.Msg) (w : Claims.Sampling.Wit),
      ¬ Claims.Sampling.RelS H Cm (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)) (stmtOf P kd desc q roots) (partOf P) w →
      min K₀ (n - kd + 1) ≤ (Claims.Sampling.faulty H Cm (stmtOf P kd desc q roots) (partOf P) w).card

/-- **The assembly**: M1's core, the slot count and the count lemma turn `EndToEndHidden` into `SampledKSLive`. -/
def AssemblyTarget : Prop :=
  FaultyRightTarget → PartRusTarget → CountTarget → Flock.Guarantees.EndToEndHidden → Flock.Guarantees.SampledKSLive

/-- A record's commitment passes only openings `Cm` passes, under `roots`. -/
def CommitRefines (Cd Cm : Claims.Sampling.Commitment Bool) (H : Claims.Msg → Claims.Msg) (roots : List Claims.Msg) :
    Prop :=
  ∀ o, Cd.Opens H roots o = true → Cm.Opens H roots o = true

/-- **rel_deploy**: at a sampling record that parses `x` to `stmtOf`, partitions it to `partOf`, and whose commitment
refines `commitOf`, a `witOf` outside `RelS` is outside the record's `R`. -/
def RelDeployTarget : Prop :=
  ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (D : Claims.Sampling.Deploy Coin Col T) {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (kd : ℕ) (desc : Claims.Msg)
    (q : Claims.Sampling.Query) (roots : List Claims.Msg) {Pos Val Opn Com : Type} (pos : Fin C.N → Pos)
    (bit : Val → Fin C.N → Bool) (val : Opn → Val) (com : Opn → Com) (cs : Pos → Com) (Xr : Fin C.N → Bool)
    (H : Claims.Msg → Claims.Msg) (δ : ℝ) (x : Claims.Msg),
    D.parse x = some (stmtOf P kd desc q roots) → D.partition (stmtOf P kd desc q roots) = some (partOf P) →
    CommitRefines D.commit (commitOf pos bit val com cs) H roots →
    ¬ Claims.Sampling.RelS H (commitOf pos bit val com cs) δ (stmtOf P kd desc q roots) (partOf P) (witOf C Xr) →
    ¬ Claims.Sampling.R H D.commit D.parse D.partition δ x (witOf C Xr)

end Flock.Bridge
