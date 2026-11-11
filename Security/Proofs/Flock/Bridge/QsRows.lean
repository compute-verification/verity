module

public import Proofs.Flock.Bridge.QsRows.Split
public import Proofs.Flock.Bridge.QsRows.RunsOut
public import Proofs.Flock.Bridge.QsRows.Pos

/-!
# `Q_s` of the session's value binding, by the rows a sample reads

`Q_s = Σ_c |Ω_c|/|Ω|` is the average number of commit strings a sample reads. At the session's value binding (`vbOf`) a
sampled proof block's wires sit at its `c.ports.size` rows and at the zero's and the constant's gates
(`vbOf_pos_io`), the same two gates for every proof block. So a sample of at most `k_d` proof blocks reads at most
`k_d · c.ports.size + 2` commit strings, whatever the number of wires in a row.

* `qs_vbOf_le_split`: `Q_s ≤ k_d · n_p + 2 + (|bad|/|Ω|) · n · n_p` for any sampling design, where every sample outside
  `bad` holds at most `k_d` proof blocks and `n_p = c.ports.size`.
* `card_div_le_of_prCoin_le`: a share of the coins as a real number.
* `qs_vbOf_execOS_le`: at the live subset sample on a uniform byte source (A3) holding the first read's
  `2 · k_d · width n + 64` bytes, `Q_s ≤ k_d · n_p + 2 + k_d · 2⁻⁶⁴ · n · n_p`: the sample holds `k_d` proof blocks
  unless it runs out of bytes, which it does with probability at most `k_d · 2⁻⁶⁴` (`Law.runOS_subset_none_le`), and
  then it holds every proof block.
-/

@[expose] public section

namespace Flock.Bridge

open Claims Claims.Sampling
open FlockSoundness.Refine FlockSoundness.Binding
open scoped ENNReal

/-- **A share of the coins as a real number**: an event of probability at most `k · 2⁻⁶⁴` holds at no more than that
share of the coins. -/
theorem card_div_le_of_prCoin_le {Ω : Type} [Fintype Ω] (p : Ω → Prop) [DecidablePred p] (k : ℕ)
    (h : FlockSoundness.Game.prCoin p ≤ (k : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹) :
    ((Finset.univ.filter p).card : ℝ) / Fintype.card Ω ≤ (k : ℝ) * (2 ^ 64 : ℝ)⁻¹ := by
  have hne : (k : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ ≠ ∞ :=
    ENNReal.mul_ne_top (ENNReal.natCast_ne_top k) (ENNReal.inv_ne_top.2 (pow_ne_zero 64 two_ne_zero))
  have h2 := ENNReal.toReal_mono hne h
  rw [FlockSoundness.Game.prCoin_eq_card, ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast,
    ENNReal.toReal_mul, ENNReal.toReal_natCast, ENNReal.toReal_inv, ENNReal.toReal_pow, ENNReal.toReal_ofNat] at h2
  exact h2

section QsRows

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkLink
  FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed FlockSoundness.Discharge.ZkHidden
  FlockSoundness.Discharge.ZkBind FlockSoundness.Discharge FlockSoundness.Discharge.Composed

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} {dj : Finset (Fin n) → Lean.Json} {S₀ : Finset (Fin n)}
  {d₀ : Lean.Json} {y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀} {ht : I.tags.typed = false} {c : Flock.Circuit}
  {hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c} {hscope : Discharge.Layout.scopeOkH c = true}
  {hU : Built c.unit} {pub : Flock.Public} {hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub}
  {hpt : pub.tables = none}

/-- **`Q_s ≤ k_d · n_p + 2 + (|bad|/|Ω|) · n · n_p` at the session's value binding**, `n_p = c.ports.size` the rows
of a proof block: a sample outside `bad` holds at most `k_d` proof blocks, each of whose wires sits at one of its
`n_p` rows or at the zero's or the constant's gate (`vbOf_pos_io`); a sample in `bad` holds at most all `n`. -/
theorem qs_vbOf_le_split (L : FlockSoundness.Audit.Law n) (kd : ℕ) (bad : Finset L.Ω)
    (hgood : ∀ ω, ω ∉ bad → (L.draw ω).card ≤ kd) :
    ∑ p : (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).Pos,
        (Fintype.card (ΩcZ execArith ((keyProg c hU n).partition (Discharge.Layout.outsOf c hU n))
          (planZHJ dj y₀) (tabZHJ dj y₀) (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)) L p) : ℝ) /
          Fintype.card L.Ω ≤
      ((kd : ℝ) * c.ports.size + 2) + (bad.card : ℝ) / Fintype.card L.Ω * ((n : ℝ) * c.ports.size) := by
  classical
  have h := qs_le_split execArith ((keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)) (planZHJ dj y₀)
    (tabZHJ dj y₀) (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)) L
    (fun u => Finset.univ.image fun q : Fin c.ports.size => (Sum.inl (u, q) : PosH c hU n))
    {(Sum.inr (Discharge.Layout.zeroG (c := c) (hU := hU) (n := n)) : PosH c hU n),
      Sum.inr (Discharge.Layout.oneG (c := c) (hU := hU) (n := n))} kd c.ports.size
    (fun u => Finset.card_image_le.trans (le_of_eq (by rw [Finset.card_univ, Fintype.card_fin])))
    (fun u g hg => by
      rcases vbOf_pos_io (dj := dj) (y₀ := y₀) (ht := ht) (hc := hc) (hscope := hscope) (hpub := hpub) (hpt := hpt)
        (Reg := Reg) u hg with ⟨q, hq⟩ | h0 | h1
      · exact Or.inl (Finset.mem_image.2 ⟨q, Finset.mem_univ _, hq.symm⟩)
      · exact Or.inr (by rw [h0]; exact Finset.mem_insert_self _ _)
      · exact Or.inr (by rw [h1]; exact Finset.mem_insert_of_mem (Finset.mem_singleton_self _)))
    bad hgood
  refine h.trans (add_le_add (add_le_add le_rfl ?_) le_rfl)
  exact_mod_cast Finset.card_le_two

/-- **`Q_s ≤ k_d · n_p + 2 + k_d · 2⁻⁶⁴ · n · n_p` at the live subset sample**, on a uniform byte source (A3) that
holds the first read's `2 · k_d · width n + 64` bytes: the sample holds `k_d` proof blocks unless it runs out of bytes
(`Law.execOS_subset_draw_card_le`), with probability at most `k_d · 2⁻⁶⁴` (`Law.runOS_subset_none_le`). -/
theorem qs_vbOf_execOS_le {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : FlockSoundness.Assumptions.UniformRandomBytes Lb src)
    (kd : ℕ) (hkd : kd ≤ n) (hLb : 2 * kd * Flock.Draw.width n + 64 ≤ Lb 0) :
    ∑ p : (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).Pos,
        (Fintype.card (ΩcZ execArith ((keyProg c hU n).partition (Discharge.Layout.outsOf c hU n))
          (planZHJ dj y₀) (tabZHJ dj y₀) (vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg))
          (Law.execOS (n := n) src (.subset kd) n) p) : ℝ) /
          Fintype.card (Law.execOS (n := n) src (.subset kd) n).Ω ≤
      ((kd : ℝ) * c.ports.size + 2) + (kd : ℝ) * (2 ^ 64 : ℝ)⁻¹ * ((n : ℝ) * c.ports.size) := by
  classical
  have hbad : ((Finset.univ.filter fun ω : Ω => Law.runOS (src ω) (.subset kd) n = none).card : ℝ) /
      Fintype.card Ω ≤ (kd : ℝ) * (2 ^ 64 : ℝ)⁻¹ :=
    card_div_le_of_prCoin_le _ kd (Law.runOS_subset_none_le n kd hkd Lb src hA3 hLb)
  have h := qs_vbOf_le_split (dj := dj) (y₀ := y₀) (ht := ht) (hc := hc) (hscope := hscope) (hU := hU) (hpub := hpub)
    (hpt := hpt) (Reg := Reg) (Law.execOS (n := n) src (.subset kd) n) kd
    (Finset.univ.filter fun ω : Ω => Law.runOS (src ω) (.subset kd) n = none)
    (fun ω hω => Law.execOS_subset_draw_card_le src kd ω fun h0 =>
      hω (Finset.mem_filter.2 ⟨Finset.mem_univ _, h0⟩))
  exact h.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_right hbad (by positivity)))

end QsRows

end Flock.Bridge
