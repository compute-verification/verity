import Proofs.Flock.Soundness.Discharge.ZkView.OsSalts

/-!
# hm96 leaves salted from one committed key hide their roots

The view is the key's own hm96 leaf, over `ks`'s digest `xk ks` under an OS salt `y₀`, and `n` hm96 leaves over the roots
`x`, whose salts come from the key: `k_w = F₁ ks` (SHA-512(ks ‖ salt_w)) and `y_i = F₂ k_w i` (SHA-512(k_w ‖ pos_i ‖ r)).
Each `F` is taken to pass the tests the hybrid needs (`PassesTest`, A4's one-sided form, one test at a time).

The hybrid, one direction (`to_ideal`): the key's leaf to an ideal one (`ideal_leaf_swap`, `δ₁`), so `ks` is read only
through `F₁`; then `k_w` uniform (`adv₁`); then the salts uniform (`adv₂`); then the `n` leaves ideal
(`ideal_leaves_swap`, `n·δ₁`). The ideal view reads no root. `from_ideal` is the other direction, by complements, and
`hides_derived` composes them at `x` and `x'`. `hides_derived_os` reads `(ks, y₀)` from the OS's bytes (A3), and
`hides_derived_os_sym` is its bound at one PRF advantage and one OS slack.
-/

namespace Pouw.SecurityProofs.Lead.Commit

open FlockSoundness FlockSoundness.Game FlockSoundness.ZK FlockSoundness.Discharge.ZkView
open scoped ENNReal

/-- A keyed map passes the test `T` up to `adv`: `T` of its output at a uniform key is at most `T` of a uniform value,
plus `adv`. -/
def PassesTest {A V : Type} [Fintype A] [Fintype V] (F : A → V) (T : V → Prop) (adv : ℝ≥0∞) : Prop :=
  prCoin (fun a => T (F a)) ≤ prCoin T + adv

/-- **A keyed step beside independent coins**: where `F` passes each test `T w`, its output at a uniform key may be
taken uniform, within `adv`. -/
theorem prf_step {W A V : Type} [Fintype W] [Nonempty W] [Fintype A] [Fintype V] (F : A → V) (T : W → V → Prop)
    {adv : ℝ≥0∞} (h : ∀ w, PassesTest F (T w) adv) :
    prCoin (fun p : W × A => T p.1 (F p.2)) ≤ prCoin (fun p : W × V => T p.1 p.2) + adv := by
  rw [prCoin_prod_split (fun w a => T w (F a)), prCoin_prod_split (fun w v => T w v)]
  calc avg (fun w => prCoin fun a => T w (F a))
      ≤ avg (fun w => prCoin (fun v => T w v) + adv) := avg_mono fun w => h w
    _ = avg (fun w => prCoin fun v => T w v) + adv := by rw [avg_add, avg_const]

/-- The `n` instances' leaves at the roots `x` and the salts `ys`. -/
abbrev leaves {n : ℕ} {Y B C D : Type} [Add B] (My : Y → B) (c : Y → C) (H : B × C → D) (x : Fin n → B)
    (ys : Fin n → Y) : Fin n → D :=
  fun i => H (x i + My (ys i), c (ys i))

/-- The published view at the roots `x`, over the key `ks` and the key leaf's salt `y₀`. -/
abbrev realView {n : ℕ} {Ks K Y B C D : Type} [Add B] (My : Y → B) (c : Y → C) (H : B × C → D) (xk : Ks → B)
    (F₁ : Ks → K) (F₂ : K → Fin n → Y) (x : Fin n → B) (p : Ks × Y) : D × (Fin n → D) :=
  (H (xk p.1 + My p.2, c p.2), leaves My c H x (F₂ (F₁ p.1)))

/-- The ideal view: every leaf over a fresh uniform value. It reads no root. -/
abbrev idealView {n : ℕ} {Y B C D : Type} (c : Y → C) (H : B × C → D) (q : (B × Y) × (Fin n → B × Y)) :
    D × (Fin n → D) :=
  (H (q.1.1, c q.1.2), fun i => H ((q.2 i).1, c (q.2 i).2))

/-- **To the ideal view**: `(n + 1)·δ₁ + adv₁ + adv₂`. -/
theorem to_ideal {n : ℕ} {Ks K Y B C D : Type} [Fintype Ks] [Nonempty Ks] [Fintype K] [Fintype Y] [Nonempty Y]
    [AddCommGroup B] [Fintype B] (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁)
    (H : B × C → D) (xk : Ks → B) (F₁ : Ks → K) (F₂ : K → Fin n → Y) (x : Fin n → B) (E : D × (Fin n → D) → Prop)
    {adv₁ adv₂ : ℝ≥0∞}
    (h₁ : ∀ w : B × Y, PassesTest F₁ (fun k => E (H (w.1, c w.2), leaves My c H x (F₂ k))) adv₁)
    (h₂ : ∀ w : B × Y, PassesTest F₂ (fun ys => E (H (w.1, c w.2), leaves My c H x ys)) adv₂) :
    prCoin (fun p : Ks × Y => E (realView My c H xk F₁ F₂ x p))
      ≤ prCoin (fun q : (B × Y) × (Fin n → B × Y) => E (idealView c H q)) + ((n + 1) * δ₁ + adv₁ + adv₂) := by
  have : Nonempty B := ⟨0⟩
  have s1 := ideal_leaf_swap My c hT1 H xk (fun ks d => E (d, leaves My c H x (F₂ (F₁ ks))))
  have sw : prCoin (fun p : Ks × (B × Y) => E (H (p.2.1, c p.2.2), leaves My c H x (F₂ (F₁ p.1))))
      = prCoin (fun p : (B × Y) × Ks => E (H (p.1.1, c p.1.2), leaves My c H x (F₂ (F₁ p.2)))) :=
    prCoin_congr_equiv (Equiv.prodComm Ks (B × Y))
      (fun p : (B × Y) × Ks => E (H (p.1.1, c p.1.2), leaves My c H x (F₂ (F₁ p.2))))
  have s2 := prf_step F₁ (fun (w : B × Y) k => E (H (w.1, c w.2), leaves My c H x (F₂ k))) h₁
  have s3 := prf_step F₂ (fun (w : B × Y) ys => E (H (w.1, c w.2), leaves My c H x ys)) h₂
  have s4 := ideal_leaves_swap My c hT1 H n (Ω := B × Y) (fun _ => x) (fun w ds => E (H (w.1, c w.2), ds))
  calc prCoin (fun p : Ks × Y => E (realView My c H xk F₁ F₂ x p))
      ≤ prCoin (fun p : Ks × (B × Y) => E (H (p.2.1, c p.2.2), leaves My c H x (F₂ (F₁ p.1)))) + δ₁ := s1
    _ = prCoin (fun p : (B × Y) × Ks => E (H (p.1.1, c p.1.2), leaves My c H x (F₂ (F₁ p.2)))) + δ₁ := by
        rw [sw]
    _ ≤ (prCoin (fun p : (B × Y) × K => E (H (p.1.1, c p.1.2), leaves My c H x (F₂ p.2))) + adv₁) + δ₁ :=
        add_le_add s2 le_rfl
    _ ≤ ((prCoin (fun p : (B × Y) × (Fin n → Y) => E (H (p.1.1, c p.1.2), leaves My c H x p.2)) + adv₂) + adv₁)
          + δ₁ :=
        add_le_add (add_le_add s3 le_rfl) le_rfl
    _ ≤ (((prCoin (fun q : (B × Y) × (Fin n → B × Y) => E (idealView c H q)) + n * δ₁) + adv₂) + adv₁) + δ₁ :=
        add_le_add (add_le_add (add_le_add s4 le_rfl) le_rfl) le_rfl
    _ = prCoin (fun q : (B × Y) × (Fin n → B × Y) => E (idealView c H q)) + ((n + 1) * δ₁ + adv₁ + adv₂) := by
        ring

/-- **From the ideal view**, by `to_ideal` on the complement. -/
theorem from_ideal {n : ℕ} {Ks K Y B C D : Type} [Fintype Ks] [Nonempty Ks] [Fintype K] [Fintype Y] [Nonempty Y]
    [AddCommGroup B] [Fintype B] (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁)
    (H : B × C → D) (xk : Ks → B) (F₁ : Ks → K) (F₂ : K → Fin n → Y) (x : Fin n → B) (E : D × (Fin n → D) → Prop)
    {adv₁ adv₂ : ℝ≥0∞}
    (h₁ : ∀ w : B × Y, PassesTest F₁ (fun k => ¬ E (H (w.1, c w.2), leaves My c H x (F₂ k))) adv₁)
    (h₂ : ∀ w : B × Y, PassesTest F₂ (fun ys => ¬ E (H (w.1, c w.2), leaves My c H x ys)) adv₂) :
    prCoin (fun q : (B × Y) × (Fin n → B × Y) => E (idealView c H q))
      ≤ prCoin (fun p : Ks × Y => E (realView My c H xk F₁ F₂ x p)) + ((n + 1) * δ₁ + adv₁ + adv₂) := by
  have : Nonempty B := ⟨0⟩
  have h := to_ideal My c hT1 H xk F₁ F₂ x (fun v => ¬ E v) h₁ h₂
  have a1 := prCoin_add_not (fun p : Ks × Y => E (realView My c H xk F₁ F₂ x p))
  have a2 := prCoin_add_not (fun q : (B × Y) × (Fin n → B × Y) => E (idealView c H q))
  have hfin := prCoin_ne_top (fun q : (B × Y) × (Fin n → B × Y) => ¬ E (idealView c H q))
  refine (ENNReal.add_le_add_iff_right hfin).1 ?_
  calc prCoin (fun q : (B × Y) × (Fin n → B × Y) => E (idealView c H q))
        + prCoin (fun q : (B × Y) × (Fin n → B × Y) => ¬ E (idealView c H q))
      = prCoin (fun p : Ks × Y => E (realView My c H xk F₁ F₂ x p))
        + prCoin (fun p : Ks × Y => ¬ E (realView My c H xk F₁ F₂ x p)) := by rw [a1, a2]
    _ ≤ prCoin (fun p : Ks × Y => E (realView My c H xk F₁ F₂ x p))
        + (prCoin (fun q : (B × Y) × (Fin n → B × Y) => ¬ E (idealView c H q)) + ((n + 1) * δ₁ + adv₁ + adv₂)) :=
        add_le_add le_rfl h
    _ = _ := by ring

/-- **Leaves salted from one committed key hide the roots**: any event on the view at `x` is at most its probability at
`x'` plus `2·((n + 1)·δ₁ + adv₁ + adv₂)`, each `F` passing the tests at `x` and the complements' at `x'`. -/
theorem hides_derived {n : ℕ} {Ks K Y B C D : Type} [Fintype Ks] [Nonempty Ks] [Fintype K] [Fintype Y]
    [Nonempty Y] [AddCommGroup B] [Fintype B] (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × C → D) (xk : Ks → B) (F₁ : Ks → K) (F₂ : K → Fin n → Y)
    (x x' : Fin n → B) (E : D × (Fin n → D) → Prop) {adv₁ adv₂ : ℝ≥0∞}
    (h₁ : ∀ w : B × Y, PassesTest F₁ (fun k => E (H (w.1, c w.2), leaves My c H x (F₂ k))) adv₁)
    (h₂ : ∀ w : B × Y, PassesTest F₂ (fun ys => E (H (w.1, c w.2), leaves My c H x ys)) adv₂)
    (h₁' : ∀ w : B × Y, PassesTest F₁ (fun k => ¬ E (H (w.1, c w.2), leaves My c H x' (F₂ k))) adv₁)
    (h₂' : ∀ w : B × Y, PassesTest F₂ (fun ys => ¬ E (H (w.1, c w.2), leaves My c H x' ys)) adv₂) :
    prCoin (fun p : Ks × Y => E (realView My c H xk F₁ F₂ x p))
      ≤ prCoin (fun p : Ks × Y => E (realView My c H xk F₁ F₂ x' p)) + 2 * ((n + 1) * δ₁ + adv₁ + adv₂) := by
  calc prCoin (fun p : Ks × Y => E (realView My c H xk F₁ F₂ x p))
      ≤ prCoin (fun q : (B × Y) × (Fin n → B × Y) => E (idealView c H q)) + ((n + 1) * δ₁ + adv₁ + adv₂) :=
        to_ideal My c hT1 H xk F₁ F₂ x E h₁ h₂
    _ ≤ (prCoin (fun p : Ks × Y => E (realView My c H xk F₁ F₂ x' p)) + ((n + 1) * δ₁ + adv₁ + adv₂))
          + ((n + 1) * δ₁ + adv₁ + adv₂) :=
        add_le_add (from_ideal My c hT1 H xk F₁ F₂ x' E h₁' h₂') le_rfl
    _ = _ := by ring

/-- **The same with `(ks, y₀)` from the OS** (A3 per test): `2·((n + 1)·δ₁ + adv₁ + adv₂) + η + η'`, which is
`2·((n + 1)·δ₁ + η + 2·Adv)` at `adv₁ = adv₂ = Adv` and `η = η'`. -/
theorem hides_derived_os {n : ℕ} {Ks K Y B C D Ωos ι : Type} [Fintype Ks] [Nonempty Ks] [Fintype K] [Fintype Y]
    [Nonempty Y] [AddCommGroup B] [Fintype B] [Fintype Ωos] [Fintype ι] [DecidableEq ι] {L : ι → ℕ}
    {t : Ωos → (i : ι) → Fin (L i) → Fin 256} (e : ((i : ι) → Fin (L i) → Fin 256) ≃ Ks × Y)
    (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × C → D) (xk : Ks → B)
    (F₁ : Ks → K) (F₂ : K → Fin n → Y) (x x' : Fin n → B) (E : D × (Fin n → D) → Prop) {adv₁ adv₂ η η' : ℝ≥0∞}
    (h₁ : ∀ w : B × Y, PassesTest F₁ (fun k => E (H (w.1, c w.2), leaves My c H x (F₂ k))) adv₁)
    (h₂ : ∀ w : B × Y, PassesTest F₂ (fun ys => E (H (w.1, c w.2), leaves My c H x ys)) adv₂)
    (h₁' : ∀ w : B × Y, PassesTest F₁ (fun k => ¬ E (H (w.1, c w.2), leaves My c H x' (F₂ k))) adv₁)
    (h₂' : ∀ w : B × Y, PassesTest F₂ (fun ys => ¬ E (H (w.1, c w.2), leaves My c H x' ys)) adv₂)
    (hA3 : FlockSoundness.Assumptions.UniformRandomBytesTest L t (fun b => E (realView My c H xk F₁ F₂ x (e b))) η)
    (hA3' : FlockSoundness.Assumptions.UniformRandomBytesTest L t (fun b => E (realView My c H xk F₁ F₂ x' (e b))) η') :
    prCoin (fun ω => E (realView My c H xk F₁ F₂ x (e (t ω))))
      ≤ prCoin (fun ω => E (realView My c H xk F₁ F₂ x' (e (t ω)))) + (2 * ((n + 1) * δ₁ + adv₁ + adv₂) + η + η') := by
  have o1 := (prCoin_os e (fun p => E (realView My c H xk F₁ F₂ x p)) hA3).1
  have o2 := (prCoin_os e (fun p => E (realView My c H xk F₁ F₂ x' p)) hA3').2
  have m := hides_derived My c hT1 H xk F₁ F₂ x x' E h₁ h₂ h₁' h₂'
  calc prCoin (fun ω => E (realView My c H xk F₁ F₂ x (e (t ω))))
      ≤ prCoin (fun p : Ks × Y => E (realView My c H xk F₁ F₂ x p)) + η := o1
    _ ≤ (prCoin (fun p : Ks × Y => E (realView My c H xk F₁ F₂ x' p)) + 2 * ((n + 1) * δ₁ + adv₁ + adv₂)) + η :=
        add_le_add m le_rfl
    _ ≤ ((prCoin (fun ω => E (realView My c H xk F₁ F₂ x' (e (t ω)))) + η') + 2 * ((n + 1) * δ₁ + adv₁ + adv₂))
          + η :=
        add_le_add (add_le_add o2 le_rfl) le_rfl
    _ = _ := by ring

/-- **The bound at one PRF advantage and one OS slack**: `2·((n + 1)·δ₁ + η + 2·Adv)`. -/
theorem hides_derived_os_sym {n : ℕ} {Ks K Y B C D Ωos ι : Type} [Fintype Ks] [Nonempty Ks] [Fintype K] [Fintype Y]
    [Nonempty Y] [AddCommGroup B] [Fintype B] [Fintype Ωos] [Fintype ι] [DecidableEq ι] {L : ι → ℕ}
    {t : Ωos → (i : ι) → Fin (L i) → Fin 256} (e : ((i : ι) → Fin (L i) → Fin 256) ≃ Ks × Y)
    (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × C → D) (xk : Ks → B)
    (F₁ : Ks → K) (F₂ : K → Fin n → Y) (x x' : Fin n → B) (E : D × (Fin n → D) → Prop) {Adv η : ℝ≥0∞}
    (h₁ : ∀ w : B × Y, PassesTest F₁ (fun k => E (H (w.1, c w.2), leaves My c H x (F₂ k))) Adv)
    (h₂ : ∀ w : B × Y, PassesTest F₂ (fun ys => E (H (w.1, c w.2), leaves My c H x ys)) Adv)
    (h₁' : ∀ w : B × Y, PassesTest F₁ (fun k => ¬ E (H (w.1, c w.2), leaves My c H x' (F₂ k))) Adv)
    (h₂' : ∀ w : B × Y, PassesTest F₂ (fun ys => ¬ E (H (w.1, c w.2), leaves My c H x' ys)) Adv)
    (hA3 : FlockSoundness.Assumptions.UniformRandomBytesTest L t (fun b => E (realView My c H xk F₁ F₂ x (e b))) η)
    (hA3' : FlockSoundness.Assumptions.UniformRandomBytesTest L t (fun b => E (realView My c H xk F₁ F₂ x' (e b))) η) :
    prCoin (fun ω => E (realView My c H xk F₁ F₂ x (e (t ω))))
      ≤ prCoin (fun ω => E (realView My c H xk F₁ F₂ x' (e (t ω)))) + 2 * ((n + 1) * δ₁ + η + 2 * Adv) := by
  calc _ ≤ _ := hides_derived_os e My c hT1 H xk F₁ F₂ x x' E h₁ h₂ h₁' h₂' hA3 hA3'
    _ = _ := by ring

end Pouw.SecurityProofs.Lead.Commit
