import Proofs.Flock.Soundness.CROnly.ZK

/-!
# Lemma C at a uniform hm96 key

`adaptive_prefinal_gap` bounds Lemma C's gap at a fixed key by `2·|L|·δ₁(κ)`, with `δ₁(κ)` the hiding gap of the leaves'
salt hashes under key `κ`. Here the key is drawn uniformly, first and independently of the salts, and everything else
(the witnesses, the table, the prover, the verifier `V*`, the event and the leaf hash) may depend on it: the gap is then
`2·|L|` times the gap's mean over the key (`adaptive_prefinal_key`), and for `hm96-sha512` that mean is at most `2^-257`
(`adaptive_prefinal_hm96_sha512`). The leaf hash depends on the key because hm96-sha512's leaves are
`H(leaf_prefix(key) ‖ ·)` with `leaf_prefix(key) = leaf tag ‖ H(key)` (`hm96/PROTOCOL.md` §2). So Lemma C needs of the
key only that it is uniform.
-/

namespace FlockSoundness.CROnly

open Finset ZK
open Game (prCoin avg avg_mono avg_add)
open scoped ENNReal

/-- **A bound at every fixed `t`, on average over a uniform `t`**: `ZK.prCoin_avg_close` with a bound that depends on
`t`. -/
theorem prCoin_avg_close_var {Tp X : Type} [Fintype Tp] [Fintype X] (A A' : Tp → X → Prop) (ε : Tp → ℝ≥0∞)
    (h : ∀ t, prCoin (A t) ≤ prCoin (A' t) + ε t) :
    prCoin (fun p : Tp × X => A p.1 p.2) ≤ prCoin (fun p : Tp × X => A' p.1 p.2) + avg ε := by
  rw [prCoin_prod_split A, prCoin_prod_split A']
  calc avg (fun t => prCoin (A t)) ≤ avg (fun t => prCoin (A' t) + ε t) := avg_mono h
    _ = _ := avg_add _ _

theorem avg_const_mul' {C : Type} [Fintype C] (x : ℝ≥0∞) (f : C → ℝ≥0∞) :
    avg (fun c => x * f c) = x * avg f := by
  unfold Game.avg
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun c _ => mul_left_comm _ _ _

theorem avg_ofReal_le {K : Type} [Fintype K] [Nonempty K] (f : K → ℝ) (hf : ∀ k, 0 ≤ f k) {b : ℝ}
    (h : ∑ k, f k ≤ Fintype.card K * b) : avg (fun k => ENNReal.ofReal (f k)) ≤ ENNReal.ofReal b := by
  have hc : (0 : ℝ) < Fintype.card K := by exact_mod_cast Fintype.card_pos
  have e : avg (fun k => ENNReal.ofReal (f k)) = ENNReal.ofReal ((∑ k, f k) / Fintype.card K) := by
    unfold Game.avg
    rw [div_eq_mul_inv, Finset.sum_mul, ENNReal.ofReal_sum_of_nonneg fun k _ =>
      mul_nonneg (hf k) (inv_nonneg.2 hc.le)]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [ENNReal.ofReal_mul (hf k), ENNReal.ofReal_inv_of_pos hc, ENNReal.ofReal_natCast, mul_comm]
  rw [e]
  exact ENNReal.ofReal_le_ofReal ((div_le_iff₀ hc).2 (h.trans_eq (mul_comm _ _)))

section Real

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt] [AddCommGroup B]
  [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B]
  [Fintype Hi] [DecidableEq Hi] {J : ℕ}

/-- **Lemma C at a uniform key**: `adaptive_prefinal_gap` with the key `κ` that hashes the leaves' salts (`My κ`) drawn
uniformly and independently of the prover's coins, the salts and the verifier's tape. The witnesses `w κ`, `w' κ`, the
table, the prover, the verifier `V*`, the event and the leaf hash `H κ` may all depend on the key, which is public. The
gap is `2·|L|` times the hiding gap's mean over the key. -/
theorem adaptive_prefinal_key {n : ℕ} {Cn M Cc Ω L Tp Key : Type} [Inhabited M] [DecidableEq M] [Fintype Ω]
    [Nonempty Ω] [Fintype L] [DecidableEq L] [Fintype Tp] [Nonempty Tp] [DecidableEq Cc] [Fintype Key]
    (T : Key → (Fin n → Cn) → Fin J → Table F K W Salt (B × Salt) D Hi VL)
    (rep : ∀ κ c j, Fin (T κ c j).nC → Fin 2) (My : Key → Salt → B) (cs : Salt → Cc) (H : Key → B × Cc → D)
    (msgs : Key → (Fin J → W) → Ω → (Fin n → Cn) → (L → D) → Fin n → M)
    (dig : Key → (Fin J → W) → Ω → (Fin n → Cn) → L → B)
    (P : Key → (Fin J → W) → Strategy (Ω × (L → B × Cc)) M Cn n)
    (hP : ∀ κ w ω p c, (P κ w).msg (ω, p) c = msgs κ w ω c fun l => H κ (dig κ w ω c l + (p l).1, (p l).2))
    (e : ∀ κ c, Ω × (L → B × Salt) → ∀ j, (T κ c j).Rand) (k : Key → (Fin n → Cn) → ℕ)
    (he : ∀ κ c r, Nat.card {x // e κ c x = r} = k κ c) (out : ∀ κ c, (∀ j, (T κ c j).Cut) → Fin n → M)
    (hI : ∀ κ w ω u c, msgs κ w ω c (fun l => H κ ((u l).1, cs (u l).2))
      = out κ c fun j =>
          (T κ c j).cut (ZK.Session.sent (T κ c) (rep κ c) j) ((T κ c j).preView (w j) (e κ c (ω, u) j)))
    (w w' : Key → Fin J → W) (hreg : ∀ κ c j, (T κ c j).sReg (w κ j) = (T κ c j).pubReg)
    (hreg' : ∀ κ c j, (T κ c j).sReg (w' κ j) = (T κ c j).pubReg) (V : Key → Tp → Responder M Cn n)
    (G : Key → Tp → (Fin n → M) → Prop) :
    prCoin (fun q : Key × (Tp × (Ω × (L → Salt))) =>
        G q.1 q.2.1 (run (P q.1 (w q.1)) (V q.1 q.2.1) (q.2.2.1, fun l => (My q.1 (q.2.2.2 l), cs (q.2.2.2 l)))))
      ≤ prCoin (fun q : Key × (Tp × (Ω × (L → Salt))) =>
          G q.1 q.2.1 (run (P q.1 (w' q.1)) (V q.1 q.2.1) (q.2.2.1, fun l => (My q.1 (q.2.2.2 l), cs (q.2.2.2 l)))))
        + 2 * Fintype.card L * avg fun κ => ENNReal.ofReal (hidingGap (My κ) cs) := by
  have h := prCoin_avg_close_var
    (fun κ (q : Tp × (Ω × (L → Salt))) =>
      G κ q.1 (run (P κ (w κ)) (V κ q.1) (q.2.1, fun l => (My κ (q.2.2 l), cs (q.2.2 l)))))
    (fun κ (q : Tp × (Ω × (L → Salt))) =>
      G κ q.1 (run (P κ (w' κ)) (V κ q.1) (q.2.1, fun l => (My κ (q.2.2 l), cs (q.2.2 l)))))
    (fun κ => 2 * (Fintype.card L : ℝ≥0∞) * ENNReal.ofReal (hidingGap (My κ) cs))
    fun κ => adaptive_prefinal_gap (T κ) (rep κ) (My κ) cs (H κ) (msgs κ) (dig κ) (P κ) (hP κ) (e κ) (k κ) (he κ)
      (out κ) (hI κ) (w κ) (w' κ) (hreg κ) (hreg' κ) (V κ) (G κ)
  rw [avg_const_mul'] at h
  exact h

end Real

/-- **Lemma C for `hm96-sha512` at a uniform key**: `adaptive_prefinal_key` with salts of 1536 bits, hashed by the
Hankel matrix of a uniform 2047-bit key into 512 bits, a salt hash `cs` with at most `2^512` values (SHA-512's), and a
leaf hash `H κ` that may depend on the key (hm96-sha512's leaf prefix carries `H(key)`): the gap is at most
`2·|L|·2^-257`. Bits are a field `Fb` of two elements. -/
theorem adaptive_prefinal_hm96_sha512 {Fb : Type} [Field Fb] [Fintype Fb] [DecidableEq Fb]
    (hF : Fintype.card Fb = 2) {F K W D Hi VL : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Hi] [Fintype F]
    [DecidableEq F] [Fintype Hi] [DecidableEq Hi] {J n : ℕ} {Cn M Cc Ω L Tp : Type} [Inhabited M] [DecidableEq M]
    [Fintype Ω] [Nonempty Ω] [Fintype L] [DecidableEq L] [Fintype Tp] [Nonempty Tp] [DecidableEq Cc]
    (T : (Fin (512 + 1536 - 1) → Fb) → (Fin n → Cn) → Fin J →
      Table F K W (Fin 1536 → Fb) ((Fin 512 → Fb) × (Fin 1536 → Fb)) D Hi VL)
    (rep : ∀ κ c j, Fin (T κ c j).nC → Fin 2) (cs : (Fin 1536 → Fb) → Cc) (hcs : #(univ.image cs) ≤ 2 ^ 512)
    (H : (Fin (512 + 1536 - 1) → Fb) → (Fin 512 → Fb) × Cc → D)
    (msgs : (Fin (512 + 1536 - 1) → Fb) → (Fin J → W) → Ω → (Fin n → Cn) → (L → D) → Fin n → M)
    (dig : (Fin (512 + 1536 - 1) → Fb) → (Fin J → W) → Ω → (Fin n → Cn) → L → Fin 512 → Fb)
    (P : (Fin (512 + 1536 - 1) → Fb) → (Fin J → W) → Strategy (Ω × (L → (Fin 512 → Fb) × Cc)) M Cn n)
    (hP : ∀ κ w ω p c, (P κ w).msg (ω, p) c = msgs κ w ω c fun l => H κ (dig κ w ω c l + (p l).1, (p l).2))
    (e : ∀ κ c, Ω × (L → (Fin 512 → Fb) × (Fin 1536 → Fb)) → ∀ j, (T κ c j).Rand)
    (k : (Fin (512 + 1536 - 1) → Fb) → (Fin n → Cn) → ℕ)
    (he : ∀ κ c r, Nat.card {x // e κ c x = r} = k κ c) (out : ∀ κ c, (∀ j, (T κ c j).Cut) → Fin n → M)
    (hI : ∀ κ w ω u c, msgs κ w ω c (fun l => H κ ((u l).1, cs (u l).2))
      = out κ c fun j =>
          (T κ c j).cut (ZK.Session.sent (T κ c) (rep κ c) j) ((T κ c j).preView (w j) (e κ c (ω, u) j)))
    (w w' : (Fin (512 + 1536 - 1) → Fb) → Fin J → W) (hreg : ∀ κ c j, (T κ c j).sReg (w κ j) = (T κ c j).pubReg)
    (hreg' : ∀ κ c j, (T κ c j).sReg (w' κ j) = (T κ c j).pubReg)
    (V : (Fin (512 + 1536 - 1) → Fb) → Tp → Responder M Cn n)
    (G : (Fin (512 + 1536 - 1) → Fb) → Tp → (Fin n → M) → Prop) :
    prCoin (fun q : (Fin (512 + 1536 - 1) → Fb) × (Tp × (Ω × (L → Fin 1536 → Fb))) =>
        G q.1 q.2.1 (run (P q.1 (w q.1)) (V q.1 q.2.1) (q.2.2.1, fun l => (hankel q.1 (q.2.2.2 l), cs (q.2.2.2 l)))))
      ≤ prCoin (fun q : (Fin (512 + 1536 - 1) → Fb) × (Tp × (Ω × (L → Fin 1536 → Fb))) =>
          G q.1 q.2.1 (run (P q.1 (w' q.1)) (V q.1 q.2.1) (q.2.2.1, fun l => (hankel q.1 (q.2.2.2 l), cs (q.2.2.2 l)))))
        + 2 * Fintype.card L * ENNReal.ofReal (1 / 2 ^ 257) :=
  (adaptive_prefinal_key T rep (fun κ => hankel κ) cs H msgs dig P hP e k he out hI w w' hreg hreg' V G).trans
    (add_le_add le_rfl (by
      gcongr
      exact avg_ofReal_le _ (fun κ => hidingGap_nonneg _ _) (hm96_sha512_sum hF cs hcs)))

end FlockSoundness.CROnly
