module

public import Properties.Pous.Period.Audits

/-!
# PoUS's instantiation in the ring model

`Instantiates P τ`, the named assumption `pous-ring-instantiation` PoUS's per-run guarantees take: on a certified
capacity coin, the memory challenger's sample is the ring game's and the prover's strategy answers, in the ring model,
the challenged blocks of `C`. It is stated over the deployed check's record (`Audits`), which the spec can't import, so
it is a named assumption under `Properties.Assumptions`, in a module of its own that reads only that audit and the
ring game.
-/

@[expose] public section

namespace Pous.Period

open Pous.EndToEnd PousCheck PousCheck.Properties

set_option exponentiation.threshold 20000

/-- `pous-ring-instantiation`, per run, in reduction form: **in the ring model, the period's strategy answers what its
certified session recorded.** On every capacity coin where the period is certified (`Certified`, which reads the memory
challenger's signed `AnsweredOnTime`: every recorded answer is the challenged block of `C*`), the challenger's sample is
the ring game's (`Drawn`, the code's sampler and coin tree on the coin's own streams, which says nothing of how they are
distributed: `Hash.Assumptions.Sha512Prf` carries that, in `PeriodHeldSha512`) and `τ`'s answers in the ring model are
the challenged blocks of `C`. It carries the challenger's record into the ring model and bounds no strategy's state
or time; it is the step from the deployed device to the ring model, so no Lean theorem discharges it. -/
def Instantiates {ΩW ΩQ : Type} (P : Audits ΩW ΩQ) (τ : Strategy ΩW ΩQ) : Prop :=
  ∀ (o : ΩW × ΩQ) (x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams),
    P.Certified (o.1, o.2, x) →
      ∃ I v, Drawn P.sid x I v ∧ P.answers τ o x I = List.ofFn fun j => (sch.enc (P.W o) x.1 ()).2 (I j)

end Pous.Period
