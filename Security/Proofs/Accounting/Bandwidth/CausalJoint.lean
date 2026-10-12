import Mathlib
import Proofs.Accounting.Bandwidth.Causal
import Proofs.Accounting.Bandwidth.TickedJoint

/-!
# Bandwidth adds, by counting (Mbw), on a strategy causal from every tick

Mbw (section 5 of `note:console/20261012T0645Z-draft-actual-circuit-model`) for the chip of a strategy
`S : (Store × Bits n) × Oracle → Actual.Run L`, causal from every tick, whose view determines `z`: a decoder `ζ`
recovers `z` from the chip's content at the window's start and what crosses into it in the window (the console's "state
at the seed, its links' deliveries in the window"). HBM's and the rest's replies may depend on `z` as well as the store
(`SplitsJoint`, `Causal.Splits` with them at `z`); the chip's widths, queries and crossings are one interface's.

`run_eq_joint` is `Causal.run_eq` for such a strategy: one transducer from `CausalFrom S` gives one `Ticked.Chip` for
every `z`, and `Interface.observe_eq` replays each `z`'s run. `audit_joint` is then `Ticked.audit_joint`'s bound: at
`e = 0`, the probability that the chip's tags pass while `reads + (M + 1) + βs T + λ < k b + n` is at most
`AuditRO`'s error. So an accept gives `n ≤ reads − cert_bw`, `cert_bw = k b − (M + 1) − βs T − λ` what
`Causal.audit_reads_ro` certifies.
-/

namespace Bandwidth

open Pous (Bits pr)
open Accounting Accounting.SecurityProofs.Actual
open scoped ENNReal

namespace Causal

namespace Interface

variable {L : Type*} {N b h : ℕ}

/-- `I` with HBM's and the rest's replies replaced. -/
def withReplies (I : Interface L N b h) (hbm : Store N b → ℕ → Acts L → ℕ → Bool)
    (side : Store N b → Oracle N b h → ℕ → Acts L → ℕ → Bool) : Interface L N b h :=
  { I with hbm := hbm, side := side }

theorem crosses_withReplies (I : Interface L N b h) (hbm : Store N b → ℕ → Acts L → ℕ → Bool)
    (side : Store N b → Oracle N b h → ℕ → Acts L → ℕ → Bool) {σ : Type*} (t₀ : ℕ)
    (F : ℕ → σ → List (Cross L) → Acts L) (st : σ) (o : List (Ticked.Obs h)) :
    (I.withReplies hbm side).crosses t₀ F st o = I.crosses t₀ F st o := by
  induction o with
  | nil => rfl
  | cons x o ih => simp only [crosses, ih]; rfl

theorem chip_withReplies (I : Interface L N b h) (hbm : Store N b → ℕ → Acts L → ℕ → Bool)
    (side : Store N b → Oracle N b h → ℕ → Acts L → ℕ → Bool) {S : ℕ} (t₀ : ℕ)
    (F : ℕ → Bits S → List (Cross L) → Acts L) (τ : Acts L → Fin N → Bits h) :
    (I.withReplies hbm side).chip t₀ F τ = I.chip t₀ F τ := by
  have hd : ∀ st o, (I.withReplies hbm side).done t₀ F st o = I.done t₀ F st o := fun st o => by
    simp only [done, crosses_withReplies]
  simp only [chip, hd]
  rfl

end Interface

/-- **The chip's links split what crosses into it**, for a strategy over the store, `z` and the oracle: at each tick of
`[t₀, t₀ + T)`, what crosses into `X` is `I.cross` of what `X` did since `t₀` and of what it observes through `I` with
HBM's replies `hbm z` and the rest's `side z`. -/
def SplitsJoint {L : Type*} {N b h n : ℕ} (S : (Store N b × Bits n) × Oracle N b h → Actual.Run L) (X : Finset L)
    (t₀ T βs : ℕ) (I : Interface L N b h) (hbm : Bits n → Store N b → ℕ → Acts L → ℕ → Bool)
    (side : Bits n → Store N b → Oracle N b h → ℕ → Acts L → ℕ → Bool) : Prop :=
  ∀ E z O, ∀ i < T, (S ((E, z), O)).crossing (↑X)ᶜ ↑X {t₀ + i} =
    I.cross (t₀ + i) ((S ((E, z), O)).acts ↑X (Set.Ico t₀ (t₀ + i)))
      ((I.withReplies (hbm z) (side z)).obs βs E O (t₀ + i) ((S ((E, z), O)).acts ↑X (Set.Ico t₀ (t₀ + i))))

/-- The content a padded state holds, when it holds one of at most `M` bits. -/
noncomputable def unpad (M : ℕ) (st : Bits (M + 1)) : List Bool :=
  open Classical in
  if hst : ∃ c : List Bool, c.length ≤ M ∧ pad M c = st then hst.choose else []

theorem unpad_pad {M : ℕ} {c : List Bool} (hc : c.length ≤ M) : unpad M (pad M c) = c := by
  have hst : ∃ c' : List Bool, c'.length ≤ M ∧ pad M c' = pad M c := ⟨c, hc, rfl⟩
  simp only [unpad, hst, dite_true]
  exact pad_inj hst.choose_spec.1 hc hst.choose_spec.2

end Causal

namespace SecurityProofs.Causal

open Bandwidth.Causal

variable {L : Type*} [LinearOrder L] {N b h n : ℕ}

/-- **The chip of a joint strategy causal from every tick is one `Ticked.Chip`**: with its state at `t₀` at most `M` bits
and fixed by the store and `z`, and its links splitting what crosses into it with HBM's and the rest's replies at `z`,
there are a chip of state `M + 1` bits, its state from the store and `z`, and HBM and the rest at each `z` such that,
for every store, `z` and oracle, the chip's tags are `τ` of what `X` did in the window, its HBM reads are the widths
summed, and what it observed fixes the run's crossings in the window. Its queries are the interface's, and the rest
ignores the oracle when the side does. -/
theorem run_eq_joint (S : (Store N b × Bits n) × Oracle N b h → Actual.Run L) (hc : CausalFrom S) (X : Finset L)
    (t₀ T M βs : ℕ) (hM : ∀ r, (S r).mem X t₀ ≤ M)
    (hpre : ∀ e O O', (S (e, O)).content X t₀ = (S (e, O')).content X t₀) (I : Interface L N b h)
    (H : Bits n → Store N b → ℕ → Acts L → ℕ → Bool) (Sd : Bits n → Store N b → Oracle N b h → ℕ → Acts L → ℕ → Bool)
    (hsplit : SplitsJoint S X t₀ T βs I H Sd) (τ : Acts L → Fin N → Bits h) :
    ∃ (F : ℕ → Bits (M + 1) → List (Cross L) → Acts L) (state : Store N b → Bits n → Bits (M + 1))
      (hbm : Bits n → Ticked.Hbm N b) (rest : Bits n → Ticked.Rest N b h),
      (∀ E z, state E z = pad M ((S ((E, z), fun _ _ => false)).content X t₀)) ∧
      (∀ E z O,
        (I.chip t₀ F τ).tags (state E z)
            (Ticked.observe (I.chip t₀ F τ) βs (state E z) (hbm z) (rest z) E O T [] [] []) =
          τ ((S ((E, z), O)).acts ↑X (Set.Ico t₀ (t₀ + T))) ∧
        Ticked.hbmBits (Ticked.observe (I.chip t₀ F τ) βs (state E z) (hbm z) (rest z) E O T [] [] []) =
          ∑ i ∈ Finset.range T, I.width (t₀ + i) ((S ((E, z), O)).acts ↑X (Set.Ico t₀ (t₀ + i))) ∧
        I.crosses t₀ F (state E z) (Ticked.observe (I.chip t₀ F τ) βs (state E z) (hbm z) (rest z) E O T [] [] []) =
          (List.range T).map fun i => (S ((E, z), O)).crossing (↑X)ᶜ ↑X {t₀ + i}) ∧
      ((∀ z E O O' u a, Sd z E O u a = Sd z E O' u a) → ∀ z E O O', rest z E O = rest z E O') := by
  obtain ⟨F, hF⟩ := transducer_bits hc X t₀ (fun r u => (S r).crossing (↑X)ᶜ ↑X {u}) (fun _ _ _ h => h) M hM
  let O₀ : Oracle N b h := fun _ _ => false
  let state : Store N b → Bits n → Bits (M + 1) := fun E z => pad M ((S ((E, z), O₀)).content X t₀)
  refine ⟨F, state, fun z => (I.withReplies (H z) (Sd z)).hbmOf t₀ F (fun E => state E z),
    fun z => (I.withReplies (H z) (Sd z)).restOf t₀ F (fun E => state E z), fun E z => rfl, fun E z O => ?_,
    fun hside z E O O' => ?_⟩
  · have hac : ∀ m ≤ T, F m (state E z) ((List.range m).map fun i => (S ((E, z), O)).crossing (↑X)ᶜ ↑X {t₀ + i}) =
        (S ((E, z), O)).acts ↑X (Set.Ico t₀ (t₀ + m)) := by
      intro m _
      rw [hF ((E, z), O) m]
      simp only [state, hpre (E, z) O₀ O]
    obtain ⟨h1, h2, h3⟩ := (I.withReplies (H z) (Sd z)).observe_eq t₀ T βs F τ (fun E => state E z) E O _ _ hac
      (hsplit E z O) T [] [] [] (by simp) rfl
    rw [Interface.chip_withReplies, Interface.crosses_withReplies] at h1
    rw [Interface.chip_withReplies] at h2 h3
    simp only [List.length_nil, zero_add] at h1 h2 h3
    refine ⟨?_, ?_, ?_⟩
    · change τ (F _ (state E z) (I.crosses t₀ F (state E z) _)) = _
      rw [h1, h2, hac T le_rfl]
    · rw [h3]
      simp only [Ticked.hbmBits, List.map_nil, List.sum_nil, zero_add]
      rfl
    · rw [h1, h2]
  · funext v u j
    simp only [Interface.restOf, Interface.withReplies, hside z E O O']

/-- **Mbw on a strategy causal from every tick, in the reads form** (`e = 0`): with the chip's state at `t₀` at most `M`
bits and fixed by the store and `z`, its links splitting what crosses into it (`SplitsJoint`, the rest's signal ignoring
the oracle, at most `qt` oracle queries a tick), the chip committing to `τ` of what it did in the window, and its view
determining `z` (`hζ`: `ζ` of its content at `t₀` and of what crossed into it in the window is `z`), the probability
over the store, `z`, the oracle and the sample that the tags pass while `reads + (M + 1) + βs T + λ < k b + n` is at
most `AuditRO`'s error. -/
theorem audit_joint (S : (Store N b × Bits n) × Oracle N b h → Actual.Run L) (hc : CausalFrom S) (X : Finset L)
    (t₀ T M βs qt : ℕ) (hM : ∀ r, (S r).mem X t₀ ≤ M)
    (hpre : ∀ e O O', (S (e, O)).content X t₀ = (S (e, O')).content X t₀) (I : Interface L N b h)
    (hq : ∀ u a, (I.ask u a).length ≤ qt) (H : Bits n → Store N b → ℕ → Acts L → ℕ → Bool)
    (Sd : Bits n → Store N b → Oracle N b h → ℕ → Acts L → ℕ → Bool)
    (hside : ∀ z E O O' u a, Sd z E O u a = Sd z E O' u a) (hsplit : SplitsJoint S X t₀ T βs I H Sd)
    (ζ : List Bool → List (Cross L) → Bits n)
    (hζ : ∀ E z O, ζ ((S ((E, z), O)).content X t₀)
      ((List.range T).map fun i => (S ((E, z), O)).crossing (↑X)ᶜ ↑X {t₀ + i}) = z)
    (τ : Acts L → Fin N → Bits h) (s k lam : ℕ) :
    pr (fun x : (Store N b × Bits n) × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1.1 (x.2.2 i))) ∧
          (∑ i ∈ Finset.range T, I.width (t₀ + i) ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + i)))) + (M + 1) +
            βs * T + lam < k * b + n)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  obtain ⟨F, state, hbm, rest, hst, hrun, hro⟩ := run_eq_joint S hc X t₀ T M βs hM hpre I H Sd hsplit τ
  let ζt : Bits (M + 1) → List (Ticked.Obs h) → Bits n := fun st o => ζ (unpad M st) (I.crosses t₀ F st o)
  have hζt : ∀ E z O,
      ζt (state E z) (Ticked.observe (I.chip t₀ F τ) βs (state E z) (hbm z) (rest z) E O T [] [] []) = z := by
    intro E z O
    show ζ (unpad M (state E z)) (I.crosses t₀ F (state E z) _) = z
    rw [(hrun E z O).2.2, hst E z,
      unpad_pad ((content_length_le (S ((E, z), fun _ _ => false)) X t₀).trans (hM _)),
      hpre (E, z) (fun _ _ => false) O]
    exact hζ E z O
  refine (pr_mono fun x hx => ?_).trans
    (Ticked.audit_joint (I.chip t₀ F τ) βs qt T (fun st o => hq _ _) ζt state hbm rest (hro hside) hζt s k lam)
  obtain ⟨⟨E, z⟩, O, Is⟩ := x
  obtain ⟨ht, hr, -⟩ := hrun E z O
  refine ⟨fun i => ?_, ?_⟩
  · rw [ht]
    exact hx.1 i
  · rw [hr]
    exact hx.2

end SecurityProofs.Causal

end Bandwidth
