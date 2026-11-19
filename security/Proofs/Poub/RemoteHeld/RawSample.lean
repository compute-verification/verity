module

public import Definitions.Poub.Guarantees.RemoteDraw
public import Properties.Spot.Draw
public import Poub.Shared

/-!
# The openings' pooled bound on the raw coin: the statement, not yet a guarantee

`ResourceAccounting.Guarantees.Remote.PooledSample` bounds the wrong receipts of an accepted session when window `t`'s
sample is `Key.subset` on the first blocks `prf t` derives from its coin, at error
`δ + Σ_t s_t/2^256 + Σ_t adv_t + binding`. The verifier draws no such sample: window `t`'s sample is the proof layer's
draw, `Spot.Draw.sampleUnits (.subset k)`, on its sample's coins as the challenger released them (`Poub.receiptSample`,
`receiptSampleRead_draw`). `openedRaw` is that draw on the bytes of window `t`'s coin (`coinBytes`), none when the
coins run out, which ends the session (`Session.close` throws). There is no second model of the draw: on it there is no
key, so no distinguisher's advantage, and no rejected first block, so no `s_t/2^256`.

`PooledSampleRaw` is the bound on that draw, at error `δ` plus the receipts' binding term at the draw's openings. Its
premises are `Live` and `Counted` as they are, and `RefusedRaw`, `Refused` read at `openedRaw`: on an accepting outcome
every window's draw ended and each receipt it opened is correct or moved. It is an instance of the proof layer's
monitoring-period product bound, `Spot.Sampling.Periods.prCoin_forall_prod_le` at `c = δ`: one period per window, its
event that the window's draw lands in its correct receipts, its predictable bound `Run.bound` (fixed before the
window's coin by `Live`), each from the proof layer's miss bound `Flock.Bridge.sampleUnits_miss_le`
(`C(g, k)/C(U, k) ≤ (g/U)^k`); the audit's own part is the arithmetic `Sample.prod_le_of_pooled_lt` (more wrong
receipts than `pooled δ` puts the product of the bounds at most `δ`). Until that is proved this is a statement: no lock
record reads this file.
-/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.Remote.RawSample

open ResourceAccounting.Remote.Held ResourceAccounting.Remote.Sample ResourceAccounting.Remote.Sample.Assumptions

/-- A `κ`-bit coin's bytes: byte `i` its bits `8i, …, 8i + 7`, the first least significant (`Pous.bitsEquiv`). -/
def coinBytes {κ : ℕ} (c : Pous.Bits κ) : Fin (κ / 8) → Fin 256 := fun i =>
  Fin.cast (by norm_num) (Pous.bitsEquiv 8 fun j => c ⟨8 * i + j, by have := i.isLt; have := j.isLt; omega⟩)

/-- **The receipts window `t` opens at coins `ω`**: the proof layer's draw of the window's `min(s_t, U_t)` among its
`U_t` on its coin's bytes, none when it runs out. -/
def openedRaw {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} (D : Run n m s κ X) (t : Fin n)
    (ω : SampleCoins n κ X) : Option (Finset ℕ) :=
  match Spot.Draw.sampleUnits (.subset (D.samples t ω)) (D.size t ω) (Properties.Spot.bytesOf (coinBytes (ω.2 t))) with
  | .ok (us, _) => some us.toList.toFinset
  | .error _ => none

/-- **The verifier's sample is the proof layer's draw**: `Poub.receiptSampleRead` draws its window's `min(samples, U)`
among its `U` receipts with `Spot.Draw.sampleUnits (.subset _)` on the sample's coins as released, and places each unit
at its pass and position; it has no draw of its own. -/
theorem receiptSampleRead_draw (samples : ℕ) (src : ByteArray) (lengths : Array ℕ) (us : Array ℕ) (read : ℕ)
    (hk : min samples ((lengths.foldl (fun (acc : Array ℕ × ℕ) n => (acc.1.push (acc.2 + n), acc.2 + n))
      (#[], 0)).1.back?.getD 0) ≠ 0)
    (h : Spot.Draw.sampleUnits (.subset (min samples ((lengths.foldl
      (fun (acc : Array ℕ × ℕ) n => (acc.1.push (acc.2 + n), acc.2 + n)) (#[], 0)).1.back?.getD 0)))
      ((lengths.foldl (fun (acc : Array ℕ × ℕ) n => (acc.1.push (acc.2 + n), acc.2 + n)) (#[], 0)).1.back?.getD 0)
      src = .ok (us, read)) :
    _root_.Poub.receiptSampleRead samples src lengths =
      .ok (us.map (fun u =>
        let ends := (lengths.foldl (fun (acc : Array ℕ × ℕ) n => (acc.1.push (acc.2 + n), acc.2 + n)) (#[], 0)).1
        let i := _root_.Poub.bisectRight ends u
        (i, u - (if i = 0 then 0 else ends[i - 1]!))), read) := by
  unfold _root_.Poub.receiptSampleRead
  simp only [hk, ite_false, h]
  rfl

/-- The receipts window `t` opens by the draw on its raw coin, empty where the draw ran out (a refused close). -/
def openings {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} (D : Run n m s κ X) (t : Fin n) (ω : SampleCoins n κ X) :
    Finset ℕ :=
  (openedRaw D t ω).getD ∅

/-- **`Refused` at the draw on the raw coin**: on every accepting outcome every window's draw ended, and each receipt it
opened is among the window's correct ones or moved. -/
def RefusedRaw {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} [Fintype X] (D : Run n m s κ X)
    (T : Trace (SampleCoins n κ X)) (R : Receipts n κ X) : Prop :=
  ∀ ω, T.accepts ω → ∀ t, ∃ S, openedRaw D t ω = some S ∧
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
