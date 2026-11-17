module

public import Flock.NetAt
public import Properties.Flock.Circuit.Rows
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Combinatorics.Matroid.Init
import Mathlib.Data.Nat.Totient
import Proofs.Flock.Soundness.Audit.Circuit
import Proofs.Flock.Soundness.ExecParse
import Proofs.Flock.Soundness.SparseRows

/-! A net committed at a position map holds exactly when its logical net does (`ofRowsAt_sat`); no verifier path reads
`ofRowsAt`. -/

@[expose] public section

namespace FlockSoundness

open Flock Audit

def NetSat (m : Net) (w : ℕ) (z : ℕ → Bool) : Prop :=
  ∀ p < w, z p = (Verity.Boolean.xorSum (m.a.row p).toList z && Verity.Boolean.xorSum (m.b.row p).toList z)

theorem ofRowsAt_sat {name : String} {useful constPos : ℕ} {ig og : Array Group} {ra rb : Array (Array ℕ)}
    {pos : Array ℕ} {n n' : Net} (hn : Net.ofRows name useful constPos ig og ra rb = .ok n)
    (h : Net.ofRowsAt name useful constPos ig og ra rb pos = .ok n') (hu : n.unitLog ≤ 32) (hu' : n'.unitLog ≤ 32)
    (z : ℕ → Bool) :
    NetSat n' (2 ^ n'.unitLog) z ↔
      NetSat n useful (z ∘ (pos.getD · 0)) ∧
        ∀ p < 2 ^ n'.unitLog, (∀ x < useful, pos.getD x 0 ≠ p) → z p = false := by
  obtain ⟨n₀, hn₀, -, -, hl, ha', hb', -, -, -, -, -, hp, -, -⟩ := Net.ofRowsAt_ok h
  have e : n₀ = n := by rw [hn] at hn₀; cases hn₀; rfl
  subst e
  obtain ⟨-, ha, hb, -⟩ := Net.ofRows_ok hn
  have w32 : ∀ k, k ≤ 32 → 2 ^ k ≤ 2 ^ 32 := fun k hk => Nat.pow_le_pow_right (by decide) hk
  have rowA := ofRows_row ha (w32 _ hu)
  have rowB := ofRows_row hb (w32 _ hu)
  have rowA' := ofRows_row ha' (w32 _ hu')
  have rowB' := ofRows_row hb' (w32 _ hu')
  have hsize : pos.getD constPos 0 + 1 ≤ 2 ^ n'.unitLog := by
    rw [hl]
    exact Nat.le_trans (le_two_pow_log2ceil _) (Nat.pow_le_pow_right (by decide) (Nat.le_max_right _ _))
  have hat : ∀ x, x < useful →
      (Verity.Boolean.xorSum (n'.a.row (pos.getD x 0)).toList z && Verity.Boolean.xorSum (n'.b.row (pos.getD x 0)).toList z) =
        (Verity.Boolean.xorSum (n₀.a.row x).toList (z ∘ (pos.getD · 0)) && Verity.Boolean.xorSum (n₀.b.row x).toList (z ∘ (pos.getD · 0))) := by
    intro x hx
    rw [rowA', rowB', Net.placed_at hp ra hx, Net.placed_at hp rb hx, rowA, rowB, Array.toList_map,
      Array.toList_map, xorSum_map, xorSum_map]
  have hgap : ∀ p, (∀ x, x < useful → pos.getD x 0 ≠ p) →
      (Verity.Boolean.xorSum (n'.a.row p).toList z && Verity.Boolean.xorSum (n'.b.row p).toList z) = false := by
    intro p hg
    rw [rowA', rowB', Net.placed_gap hp ra hg, Net.placed_gap hp rb hg]
    rfl
  constructor
  · intro hs
    refine ⟨fun x hx => ?_, fun p hp' hg => ?_⟩
    · have := hs _ (Nat.lt_of_lt_of_le (Net.posOk_lt hp hx) hsize)
      rw [hat x hx] at this
      exact this
    · rw [hs p hp', hgap p hg]
  · intro ⟨hls, hgs⟩ p hp'
    by_cases hx : ∃ x, x < useful ∧ pos.getD x 0 = p
    · obtain ⟨x, hx, rfl⟩ := hx
      rw [hat x hx]
      exact hls x hx
    · have hg : ∀ x, x < useful → pos.getD x 0 ≠ p := fun x h1 h2 => hx ⟨x, h1, h2⟩
      rw [hgs p hp' hg, hgap p hg]

theorem rowsSat_of_netSat {net : Net} (hB : Built net) (hu : net.unitLog ≤ 32) {w : ℕ} (hw : net.constPos ≤ w)
    {z : ℕ → Bool} (hs : NetSat net w z) : (Rows.ofNet hB).Sat fun c => z c.val := by
  intro i
  have hc : ((Rows.ofNet hB).comp i).val < net.constPos := by
    have hi : i.val < net.constPos - min (net.inWords * WORD) net.constPos := i.isLt
    show min (net.inWords * WORD) net.constPos + i.val < net.constPos
    omega
  have e := hs _ (Nat.lt_of_lt_of_le hc hw)
  show z ((Rows.ofNet hB).comp i).val =
    (Verity.Boolean.xorSum ((Rows.ofNet hB).a i) (z ∘ Fin.val) && Verity.Boolean.xorSum ((Rows.ofNet hB).b i) (z ∘ Fin.val))
  rw [← xorSum_map, ← xorSum_map, Rows.ofNet_a hB hu i, Rows.ofNet_b hB hu i]
  exact e

theorem ofRowsAt_rowsSat {name : String} {useful constPos : ℕ} {ig og : Array Group} {ra rb : Array (Array ℕ)}
    {pos : Array ℕ} {n n' : Net} (hn : Net.ofRows name useful constPos ig og ra rb = .ok n)
    (h : Net.ofRowsAt name useful constPos ig og ra rb pos = .ok n') (hu : n.unitLog ≤ 32) (hu' : n'.unitLog ≤ 32)
    {z : ℕ → Bool} (hs : NetSat n' (2 ^ n'.unitLog) z) :
    (Rows.ofNet ⟨_, _, _, _, _, _, _, hn⟩).Sat (fun c => z (pos.getD c.val 0)) ∧
      ∀ p < 2 ^ n'.unitLog, (∀ x < useful, pos.getD x 0 ≠ p) → z p = false := by
  obtain ⟨hls, hgs⟩ := (ofRowsAt_sat hn h hu hu' z).1 hs
  refine ⟨rowsSat_of_netSat _ hu ?_ hls, hgs⟩
  obtain ⟨-, -, -, -, hc, -⟩ := Net.ofRows_ok hn
  have := (ofRows_groups hn).1
  omega

end FlockSoundness
