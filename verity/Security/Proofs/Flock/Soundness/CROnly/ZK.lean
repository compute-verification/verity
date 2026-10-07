import Proofs.Flock.Soundness.CROnly.Hm96Sha512
import Proofs.Flock.Soundness.ZK.Session
import Proofs.Flock.Soundness.ZK.AdaptivePrefinal

/-!
# The ZK theorems with hm96 hiding discharged

`Session.session_shvzk_hm96` (Lemma B) and `Session.adaptive_prefinal_hm96_tape` (Lemma C) take
`Assumptions.Hm96Hiding My cs δ₁` as a hypothesis. `hm96Hiding_gap` proves it, with no hypothesis, at `δ₁` the hiding gap
`hidingGap My cs`; here both theorems are restated at that `δ₁` (`session_shvzk_gap`, `adaptive_prefinal_gap`), and at any
bound `δ` on the gap (`session_shvzk_le`, `adaptive_prefinal_le`). For `hm96-sha512` the gap is at most `2^-193` at all
but a `2^-64` fraction of the keys (`hm96_sha512_bad_keys`), so what remains is that the pinned key is uniform
(`hash-derived-key`, a model fact of the common reference string).

These are zero-knowledge statements, about what the prover's messages reveal (at fixed coins for Lemma B, against any
`V*` for Lemma C), over any tables `T`. None is a soundness statement, so the `--zk` verifier's place for `final_c'`
(#793) is outside them.
-/

namespace FlockSoundness.CROnly

open Finset ZK
open Game (prCoin)
open scoped ENNReal

theorem hm96Hiding_le {Y B C : Type} [Fintype Y] [Fintype B] [DecidableEq B] [DecidableEq C] [Nonempty Y]
    [Nonempty B] {My : Y → B} {c : Y → C} {δ : ℝ} (h : hidingGap My c ≤ δ) :
    Assumptions.Hm96Hiding My c (ENNReal.ofReal δ) :=
  hm96Hiding_mono (hm96Hiding_gap My c) (ENNReal.ofReal_le_ofReal h)

open Classical in
/-- **hm96 hides at all but a `2^-e` fraction of its keys**: for `n` output bits, an `l`-bit salt and a salt hash `c`
with at most `2^m` values, the keys `κ` at which `Hm96Hiding (M(κ)·) c 2^-b` fails are at most a `2^-e` fraction,
whenever `n + m + 2a ≤ l` and `b + e = a + 1`. -/
theorem hm96_bad_keys {F : Type} [Field F] [Fintype F] [DecidableEq F] (hF : Fintype.card F = 2) {C : Type}
    [DecidableEq C] {n l m a b e : ℕ} (c : (Fin l → F) → C) (hc : #(univ.image c) ≤ 2 ^ m) (h : n + m + 2 * a ≤ l)
    (hbe : b + e = a + 1) :
    (#{κ : Fin (n + l - 1) → F | ¬ Assumptions.Hm96Hiding (hankel κ) c (ENNReal.ofReal (1 / 2 ^ b))} : ℝ) ≤
      Fintype.card (Fin (n + l - 1) → F) / 2 ^ e := by
  refine le_trans ?_ (hm96_tail hF c hc h hbe)
  have hsub : (univ.filter fun κ : Fin (n + l - 1) → F =>
        ¬ Assumptions.Hm96Hiding (hankel κ) c (ENNReal.ofReal (1 / 2 ^ b))) ⊆
      univ.filter fun κ : Fin (n + l - 1) → F => 1 / 2 ^ b < hidingGap (hankel κ) c := fun κ hκ => by
    rw [mem_filter] at hκ ⊢
    exact ⟨hκ.1, lt_of_not_ge fun hg => hκ.2 (hm96Hiding_le hg)⟩
  exact_mod_cast card_le_card hsub

open Classical in
/-- **`hm96-sha512` hides at all but `2^-64` of its keys**: the keys `κ` at which `Hm96Hiding (M(κ)·) c 2^-193` fails
are at most a `2^-64` fraction, for any salt hash `c` with at most `2^512` values. -/
theorem hm96_sha512_bad_keys {F : Type} [Field F] [Fintype F] [DecidableEq F] (hF : Fintype.card F = 2) {C : Type}
    [DecidableEq C] (c : (Fin 1536 → F) → C) (hc : #(univ.image c) ≤ 2 ^ 512) :
    (#{κ : Fin (512 + 1536 - 1) → F |
        ¬ Assumptions.Hm96Hiding (hankel κ) c (ENNReal.ofReal (1 / 2 ^ 193))} : ℝ) ≤
      Fintype.card (Fin (512 + 1536 - 1) → F) / 2 ^ 64 :=
  hm96_bad_keys hF (n := 512) (a := 256) c hc (by norm_num) (by norm_num)

section Session

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [AddCommGroup B] {J : ℕ}
  (T : Fin J → Table F K W Salt (B × Salt) D Hi VL) [CharP F 2]
  [AddCommGroup Salt] [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K]
  [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi]

/-- **Lemma B with real hm96 leaves, at a bound on the hiding gap**: `Session.session_shvzk_hm96` with every table's leaves
under one hash `My` of the salt and one salt hash `cs`, and `δ₁` any bound `δ` on their hiding gap. -/
theorem session_shvzk_le {C : Type} [DecidableEq C] (M : ∀ j, (T j).Hm96 C) (My : Salt → B) (cs : Salt → C)
    (hMy : ∀ j, (M j).My = My) (hcs : ∀ j, (M j).cs = cs) {δ : ℝ} (hδ : hidingGap My cs ≤ δ)
    (hrank : ∀ j, (T j).Rank) (hpad : ∀ j, (T j).PadOnto)
    (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2) (hpads : ∀ j, (T j).PadsOnto)
    (hβ : ∀ j r, (T j).β r ≠ 0) (w : Fin J → W) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hinner : ∀ j, (T j).InnerHolds (w j)) (w0 : Fin J → W) (E : (∀ j, (T j).View) → Prop) :
    prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => E fun j => (M j).viewR (w j) (p j).1 (p j).2)
        ≤ prCoin (fun p : ∀ j, (T j).SimRand × ((T j).Hid → Salt) => E fun j => (M j).simR (w0 j) (p j).1 (p j).2)
          + ∑ j, 2 * Fintype.card (T j).Hid * ENNReal.ofReal δ
      ∧ prCoin (fun p : ∀ j, (T j).SimRand × ((T j).Hid → Salt) => E fun j => (M j).simR (w0 j) (p j).1 (p j).2)
        ≤ prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => E fun j => (M j).viewR (w j) (p j).1 (p j).2)
          + ∑ j, 2 * Fintype.card (T j).Hid * ENNReal.ofReal δ := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty B := ⟨0⟩
  exact ZK.Session.session_shvzk_hm96 T M (fun j => by rw [hMy j, hcs j]; exact hm96Hiding_le hδ) hrank hpad hW hpads
    hβ w hreg hinner w0 E

/-- **Lemma B with real hm96 leaves, with no hiding hypothesis**: `session_shvzk_le` at `δ₁` the hiding gap itself. -/
theorem session_shvzk_gap {C : Type} [DecidableEq C] (M : ∀ j, (T j).Hm96 C) (My : Salt → B) (cs : Salt → C)
    (hMy : ∀ j, (M j).My = My) (hcs : ∀ j, (M j).cs = cs)
    (hrank : ∀ j, (T j).Rank) (hpad : ∀ j, (T j).PadOnto)
    (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2) (hpads : ∀ j, (T j).PadsOnto)
    (hβ : ∀ j r, (T j).β r ≠ 0) (w : Fin J → W) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hinner : ∀ j, (T j).InnerHolds (w j)) (w0 : Fin J → W) (E : (∀ j, (T j).View) → Prop) :
    prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => E fun j => (M j).viewR (w j) (p j).1 (p j).2)
        ≤ prCoin (fun p : ∀ j, (T j).SimRand × ((T j).Hid → Salt) => E fun j => (M j).simR (w0 j) (p j).1 (p j).2)
          + ∑ j, 2 * Fintype.card (T j).Hid * ENNReal.ofReal (hidingGap My cs)
      ∧ prCoin (fun p : ∀ j, (T j).SimRand × ((T j).Hid → Salt) => E fun j => (M j).simR (w0 j) (p j).1 (p j).2)
        ≤ prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => E fun j => (M j).viewR (w j) (p j).1 (p j).2)
          + ∑ j, 2 * Fintype.card (T j).Hid * ENNReal.ofReal (hidingGap My cs) :=
  session_shvzk_le T M My cs hMy hcs le_rfl hrank hpad hW hpads hβ w hreg hinner w0 E

end Session

section Real

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt] [AddCommGroup B]
  [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B]
  [Fintype Hi] [DecidableEq Hi] {J : ℕ}

/-- **Lemma C at a bound on the hiding gap**: `Session.adaptive_prefinal_hm96_tape` with `δ₁` any bound `δ` on the hiding
gap of the leaves' salt hashes `(My, cs)`. -/
theorem adaptive_prefinal_le {n : ℕ} {Cn M Cc Ω L Tp : Type} [Inhabited M] [DecidableEq M] [Fintype Ω]
    [Nonempty Ω] [Fintype L] [DecidableEq L] [Fintype Tp] [Nonempty Tp] [DecidableEq Cc]
    (T : (Fin n → Cn) → Fin J → Table F K W Salt (B × Salt) D Hi VL)
    (rep : ∀ c j, Fin (T c j).nC → Fin 2) (My : Salt → B) (cs : Salt → Cc) {δ : ℝ} (hδ : hidingGap My cs ≤ δ)
    (H : B × Cc → D)
    (msgs : (Fin J → W) → Ω → (Fin n → Cn) → (L → D) → Fin n → M) (dig : (Fin J → W) → Ω → (Fin n → Cn) → L → B)
    (P : (Fin J → W) → Strategy (Ω × (L → B × Cc)) M Cn n)
    (hP : ∀ w ω p c, (P w).msg (ω, p) c = msgs w ω c fun l => H (dig w ω c l + (p l).1, (p l).2))
    (e : ∀ c, Ω × (L → B × Salt) → ∀ j, (T c j).Rand) (k : (Fin n → Cn) → ℕ)
    (he : ∀ c r, Nat.card {x // e c x = r} = k c) (out : ∀ c, (∀ j, (T c j).Cut) → Fin n → M)
    (hI : ∀ w ω u c, msgs w ω c (fun l => H ((u l).1, cs (u l).2))
      = out c fun j => (T c j).cut (ZK.Session.sent (T c) (rep c) j) ((T c j).preView (w j) (e c (ω, u) j)))
    (w w' : Fin J → W) (hreg : ∀ c j, (T c j).sReg (w j) = (T c j).pubReg)
    (hreg' : ∀ c j, (T c j).sReg (w' j) = (T c j).pubReg) (V : Tp → Responder M Cn n)
    (G : Tp → (Fin n → M) → Prop) :
    prCoin (fun q : Tp × (Ω × (L → Salt)) =>
        G q.1 (run (P w) (V q.1) (q.2.1, fun l => (My (q.2.2 l), cs (q.2.2 l)))))
      ≤ prCoin (fun q : Tp × (Ω × (L → Salt)) =>
          G q.1 (run (P w') (V q.1) (q.2.1, fun l => (My (q.2.2 l), cs (q.2.2 l)))))
        + 2 * Fintype.card L * ENNReal.ofReal δ := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty B := ⟨0⟩
  exact ZK.Session.adaptive_prefinal_hm96_tape T rep My cs (hm96Hiding_le hδ) H msgs dig P hP e k he out hI w w' hreg
    hreg' V G

/-- **Lemma C with no hiding hypothesis**: `adaptive_prefinal_le` at `δ₁` the hiding gap itself. -/
theorem adaptive_prefinal_gap {n : ℕ} {Cn M Cc Ω L Tp : Type} [Inhabited M] [DecidableEq M] [Fintype Ω]
    [Nonempty Ω] [Fintype L] [DecidableEq L] [Fintype Tp] [Nonempty Tp] [DecidableEq Cc]
    (T : (Fin n → Cn) → Fin J → Table F K W Salt (B × Salt) D Hi VL)
    (rep : ∀ c j, Fin (T c j).nC → Fin 2) (My : Salt → B) (cs : Salt → Cc) (H : B × Cc → D)
    (msgs : (Fin J → W) → Ω → (Fin n → Cn) → (L → D) → Fin n → M) (dig : (Fin J → W) → Ω → (Fin n → Cn) → L → B)
    (P : (Fin J → W) → Strategy (Ω × (L → B × Cc)) M Cn n)
    (hP : ∀ w ω p c, (P w).msg (ω, p) c = msgs w ω c fun l => H (dig w ω c l + (p l).1, (p l).2))
    (e : ∀ c, Ω × (L → B × Salt) → ∀ j, (T c j).Rand) (k : (Fin n → Cn) → ℕ)
    (he : ∀ c r, Nat.card {x // e c x = r} = k c) (out : ∀ c, (∀ j, (T c j).Cut) → Fin n → M)
    (hI : ∀ w ω u c, msgs w ω c (fun l => H ((u l).1, cs (u l).2))
      = out c fun j => (T c j).cut (ZK.Session.sent (T c) (rep c) j) ((T c j).preView (w j) (e c (ω, u) j)))
    (w w' : Fin J → W) (hreg : ∀ c j, (T c j).sReg (w j) = (T c j).pubReg)
    (hreg' : ∀ c j, (T c j).sReg (w' j) = (T c j).pubReg) (V : Tp → Responder M Cn n)
    (G : Tp → (Fin n → M) → Prop) :
    prCoin (fun q : Tp × (Ω × (L → Salt)) =>
        G q.1 (run (P w) (V q.1) (q.2.1, fun l => (My (q.2.2 l), cs (q.2.2 l)))))
      ≤ prCoin (fun q : Tp × (Ω × (L → Salt)) =>
          G q.1 (run (P w') (V q.1) (q.2.1, fun l => (My (q.2.2 l), cs (q.2.2 l)))))
        + 2 * Fintype.card L * ENNReal.ofReal (hidingGap My cs) :=
  adaptive_prefinal_le T rep My cs le_rfl H msgs dig P hP e k he out hI w w' hreg hreg' V G

end Real

end FlockSoundness.CROnly
