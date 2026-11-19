module

public import Definitions.Poub.Guarantees.RemoteDraw
public import Properties.Spot.Draw
public import Poub.Shared
import Proofs.Flock.Claims.Sampling.Select.Bounds
import Proofs.Spot.Sampling.Work
import Proofs.Core.Game.Tape

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
open scoped ENNReal

/-- Bit `j` of byte `i` of a `κ`-bit coin: bit `8i + j`. -/
def byteBit {κ : ℕ} (p : Fin (κ / 8) × Fin 8) : Fin κ :=
  ⟨8 * p.1 + p.2, by have := p.1.isLt; have := p.2.isLt; omega⟩

/-- Bytes from their bits: byte `i` is bits `(i, 0), …, (i, 7)`, the first least significant (`Pous.bitsEquiv`). -/
def bytesEquiv (L : ℕ) : (Fin L × Fin 8 → Bool) ≃ (Fin L → Fin 256) :=
  (Equiv.curry (Fin L) (Fin 8) Bool).trans
    (Equiv.piCongrRight fun _ => (Pous.bitsEquiv 8).trans (finCongr (by norm_num)))

/-- A `κ`-bit coin's bytes: byte `i` its bits `8i, …, 8i + 7`, the first least significant. -/
def coinBytes {κ : ℕ} (c : Pous.Bits κ) : Fin (κ / 8) → Fin 256 :=
  bytesEquiv (κ / 8) fun p => c (byteBit p)

theorem byteBit_injective {κ : ℕ} : Function.Injective (byteBit (κ := κ)) := by
  rintro ⟨⟨i, hi⟩, ⟨j, hj⟩⟩ ⟨⟨i', hi'⟩, ⟨j', hj'⟩⟩ h
  simp only [byteBit, Fin.mk.injEq] at h
  exact Prod.ext (Fin.ext (by dsimp only; omega)) (Fin.ext (by dsimp only; omega))

/-- A uniform coin read through an injective map of its coordinates is uniform. -/
theorem avg_comp_injective {I J : Type} [Fintype I] [Fintype J] [DecidableEq I] [DecidableEq J] {ι : J → I}
    (hι : Function.Injective ι) (f : (J → Bool) → ℝ≥0∞) :
    Verity.Game.avg (fun c : I → Bool => f fun j => c (ι j)) = Verity.Game.avg f := by
  classical
  let e : (I → Bool) ≃ (J → Bool) × ({i // i ∉ Set.range ι} → Bool) :=
    (Equiv.piEquivPiSubtypeProd (· ∈ Set.range ι) fun _ => Bool).trans
      (Equiv.prodCongr (Equiv.arrowCongr (Equiv.ofInjective ι hι).symm (Equiv.refl Bool)) (Equiv.refl _))
  calc Verity.Game.avg (fun c : I → Bool => f fun j => c (ι j))
      = Verity.Game.avg (fun y : (J → Bool) × ({i // i ∉ Set.range ι} → Bool) => f y.1) :=
        Verity.Game.Tape.avg_equiv e fun y => f y.1
    _ = Verity.Game.avg f := by
        rw [Verity.Game.Tape.avg_prod]
        exact congrArg Verity.Game.avg (funext fun a =>
          Verity.Game.avg_const (C := {i // i ∉ Set.range ι} → Bool) (f a))

/-- The bytes of a uniform coin are uniform. -/
theorem prCoin_coinBytes {κ : ℕ} (P : (Fin (κ / 8) → Fin 256) → Prop) :
    Verity.Game.prCoin (fun c : Pous.Bits κ => P (coinBytes c)) = Verity.Game.prCoin P := by
  classical
  unfold Verity.Game.prCoin
  exact (avg_comp_injective (byteBit_injective (κ := κ))
    fun d => if P (bytesEquiv (κ / 8) d) then (1 : ℝ≥0∞) else 0).trans
    (Verity.Game.Tape.avg_equiv (bytesEquiv (κ / 8)) fun d => if P d then (1 : ℝ≥0∞) else 0)

/-- **A window's bound** (piece of `Periods`' instance): the proof layer's draw of `k` of `U` on a uniform coin's bytes
lands in `C` with probability at most `(g/U)^k`, `0` when `g < k`, `g` the receipts of `C` among the `U`: the miss bound
`Properties.Spot.SecurityProofs.one_bound` at the receipts outside `C`, `C(g, k)/C(U, k)`, and
`Spot.Sampling.SamplingDesign.choose_div_le_pow`. -/
theorem window_le {κ : ℕ} (U k : ℕ) (C : Finset ℕ) (hk : k ≤ U) :
    Verity.Game.prCoin (fun c : Pous.Bits κ => ∃ us read,
        Spot.Draw.sampleUnits (.subset k) U (Properties.Spot.bytesOf (coinBytes c)) = .ok (us, read) ∧
          us.toList.toFinset ⊆ C) ≤
      ENNReal.ofReal (if k ≤ ((Finset.range U).filter (· ∈ C)).card then
        ((((Finset.range U).filter (· ∈ C)).card : ℝ) / U) ^ k else 0) := by
  classical
  set W := (Finset.range U).filter (· ∉ C) with hWdef
  have hW : W ⊆ Finset.range U := Finset.filter_subset _ _
  have hsum : ((Finset.range U).filter (· ∈ C)).card + W.card = U := by
    rw [hWdef]
    simpa using Finset.card_filter_add_card_filter_not (s := Finset.range U) (p := (· ∈ C))
  have hg : U - W.card = ((Finset.range U).filter (· ∈ C)).card := by omega
  refine (prCoin_coinBytes (κ := κ) fun src => ∃ us read,
      Spot.Draw.sampleUnits (.subset k) U (Properties.Spot.bytesOf src) = .ok (us, read) ∧
        us.toList.toFinset ⊆ C).trans_le ?_
  rw [Verity.Game.prCoin_eq_ofReal_pr]
  refine ENNReal.ofReal_le_ofReal ?_
  refine (Claims.Pr_mono ?_).trans (((Properties.Spot.SecurityProofs.one_bound U W hW _).1 k hk).trans ?_)
  · rintro src ⟨us, read, h, hs⟩
    exact ⟨us, read, h, fun u hu hmem => (Finset.mem_filter.1 hu).2 (hs (by simpa using hmem))⟩
  rw [hg]
  split_ifs with hkg
  · rcases Nat.eq_zero_or_pos U with rfl | hU
    · obtain rfl : k = 0 := by omega
      simp
    · have h : ((U - W.card).choose k : ℝ) / (U.choose k : ℝ) ≤ (((U - W.card : ℕ) : ℝ) / U) ^ k := by
        exact_mod_cast NNReal.coe_le_coe.2 (Spot.Sampling.SamplingDesign.choose_div_le_pow (m := W.card) hk hU)
      rwa [hg] at h
  · rw [Nat.choose_eq_zero_of_lt (by omega)]
    simp

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
