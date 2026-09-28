import Proofs.Flock.Soundness.Audit.OneStage

/-!
# Audits whose draw depends on the registration

The pinned `audit L Reg session` draws `L.draw ω`, which can't see the registration `R`. A keyed draw whose context
binds the registration, such as the PoUW window key's receipt digest (#423), is a law `L R` for each `R`. `auditReg L`
is that game: registration `R`, then the coin of `L R`, then the session on `L R`'s draw.

* `atReg σ`: the strategy of `audit (L (reg σ))` that registers `σ`'s `R` and plays its continuation.
* `prob_auditReg`: **it has the same outcome probabilities** (`rfl`), since the registration is the strategy's first move,
  fixed before the coin. So every guarantee about `audit L_c` applies to `auditReg L`, strategy by strategy, at
  `c := R`, with its hypotheses taken for every `R`.
-/

namespace FlockSoundness.Audit

open Game

variable {n : ℕ}

/-- **The audit whose draw depends on the registration**: `R`, then `L R`'s coin, then the session on its draw. -/
def auditReg {Reg : Type} (L : Reg → Law n) (session : Finset (Fin n) → Reg → Game Bool) :
    Game (Bool × Finset (Fin n)) :=
  .send Reg fun R => .coin (L R).Ω fun ω => (session ((L R).draw ω) R).map fun ok => (ok, (L R).draw ω)

/-- The strategy of the fixed-law game at `σ`'s registration that plays `σ`. -/
def atReg {Reg : Type} {L : Reg → Law n} {session : Finset (Fin n) → Reg → Game Bool}
    (σ : Strategy (auditReg L session)) : Strategy (audit (L σ.1) Reg session) :=
  ⟨σ.1, σ.2⟩

/-- **A strategy of the registration-dependent audit is a strategy of the fixed-law audit at its registration's law**,
with the same outcome probabilities. -/
theorem prob_auditReg {Reg : Type} (L : Reg → Law n) (session : Finset (Fin n) → Reg → Game Bool)
    (p : Bool × Finset (Fin n) → Prop) (σ : Strategy (auditReg L session)) :
    prob p (auditReg L session) σ = prob p (audit (L σ.1) Reg session) (atReg σ) := rfl

theorem reg_atReg {Reg : Type} {L : Reg → Law n} {session : Finset (Fin n) → Reg → Game Bool}
    (σ : Strategy (auditReg L session)) : reg (atReg σ) = σ.1 := rfl

end FlockSoundness.Audit
