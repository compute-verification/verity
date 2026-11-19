module

public import Definitions.Poub.Guarantees.RemoteDraw

/-!
# The openings' pooled bound on the raw coin: the statement, not yet a guarantee

`ResourceAccounting.Guarantees.Remote.PooledSample` bounds the wrong receipts of an accepted session when window `t`'s
sample is `Key.subset` on the first blocks `prf t` derives from its coin, at error
`δ + Σ_t s_t/2^256 + Σ_t adv_t + binding`. The verifier draws no such sample: window `t`'s sample is the proof layer's
draw on its sample's coins as the challenger released them (`Poub.receiptSample`, `Spot.Draw.sampleUnits (.subset k)`),
modelled here by `Run.openedRaw` (`drawRaw`), none when the coins run out, which ends the session (`Session.close` throws). On that draw there
is no key, so no distinguisher's advantage, and no rejected first block, so no `s_t/2^256`: a draw that runs out is a
refusal, not an error.

`PooledSampleRaw` is the bound on that draw, at error `δ` plus the receipts' binding term at the draw's openings. Its
premises are `Live` and `Counted` as they are, and `RefusedRaw`, `Refused` read at `openedRaw`: on an accepting outcome
every window's draw ended and each receipt it opened is correct or moved. Its proof would take, window by window, the
proof layer's miss bound `FlockSoundness.Discharge.ZkLink.sampleUnits_miss_le` (a subset of `k` of `U` misses a set
with probability at most `C(U − w, k)/C(U, k) ≤ (g/U)^k`, `Run.bound`), through `drawRaw U k c` being
`Spot.Draw.sampleUnits (.subset k) U` on the coin's bytes, and then the pooling step as it is (`Sample.pr_live_le`,
`Sample.prod_le_of_pooled_lt`), with no hybrid. No lock record reads this file.
-/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.Remote.RawSample

open ResourceAccounting.Remote.Held ResourceAccounting.Remote.Sample ResourceAccounting.Remote.Sample.Assumptions

/-- The receipts window `t` opens by the draw on its raw coin, empty where the draw ran out (a refused close). -/
def openings {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} (D : Run n m s κ X) (t : Fin n) (ω : SampleCoins n κ X) :
    Finset ℕ :=
  (D.openedRaw t ω).getD ∅

/-- **`Refused` at the draw on the raw coin**: on every accepting outcome every window's draw ended, and each receipt it
opened is among the window's correct ones or moved. -/
def RefusedRaw {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} [Fintype X] (D : Run n m s κ X)
    (T : Trace (SampleCoins n κ X)) (R : Receipts n κ X) : Prop :=
  ∀ ω, T.accepts ω → ∀ t, ∃ S, D.openedRaw t ω = some S ∧
    ∀ p ∈ S, p ∈ D.correct t ω ∨ R.Moved (openings D) t ω p

/-- **The openings' pooled bound on the raw coin.** On a session meeting `Live`, `RefusedRaw` and `Counted`, the
probability that it is accepted with more wrong receipts than the record's pooled bound at `δ` is at most `δ` plus the
receipts' binding term at the draw's openings. -/
def PooledSampleRaw : Prop :=
  ∀ (n m : ℕ) (k : Fin n → ℕ) (κ : ℕ) (X : Type) [Fintype X] (D : Run n m k κ X)
    (T : Trace (SampleCoins n κ X)) (R : Receipts n κ X)
    (δ : ℝ), 0 ≤ δ → Live D → RefusedRaw D T R → Counted D T →
    T.Holds (δ + (R.binding (openings D)).toReal) fun ω => (T.incorrect ω : ℝ) ≤ D.pooled δ ω

end ResourceAccounting.SecurityProofs.Remote.RawSample
