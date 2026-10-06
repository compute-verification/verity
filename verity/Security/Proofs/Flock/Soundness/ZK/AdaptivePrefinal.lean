import Proofs.Flock.Soundness.ZK.Refusal
import Proofs.Flock.Soundness.ZK.Adaptivity
import Proofs.Flock.Soundness.ZK.RealLeaves

/-!
# Zero knowledge: Lemma C, the pre-final views against an adaptive verifier

Lemma C (`docs/zk-proof-public.md` §4.4) bounds, against any verifier `V*`, the distance between the real prover's
pre-final transcript (everything before the proofs, which follow the session's last coin) on the witness and on the
dummy witness: `δ_pre ≤ 2·δ₁·(N_hid + 2J)`. `V*` is adaptive: its coin openings, its aborts and its invalid openings are
answers that read the prover's messages, so the coins and where the session stops are functions of the transcript. The
proof's route, here in Lean:

1. **Exact, in world `W₀`** (every pre-final leaf ideal), at every fixed answer vector `c`: the prover's messages are a
   function `out c` of the session's pre-final view cut where the session stops at `c`'s coins (`Session.sent`), and that
   cut is witness-independent (`Session.session_prefinal_indep_cut`, Lemma A at every coin vector with the schedule). T5
   (`card_run_eq`) lifts the equality to the transcript against `V*`: `Session.adaptive_prefinal_indep`.
2. **Real leaves** (T1, `ideal_leaves_swap_fintype`), adaptively. A leaf is `H(dig + M·y, c(y))`, its digest `dig` read
   at the answers (τ's commitment commits `⟨λ, μ⟩`, and `λ` reads the coins). T1 is applied with digest 0 to the pair
   `(M·y, c(y))` itself, which the transcript reads only through the leaves, whatever `V*` does: the prover then masks each
   digest with a fresh uniform `U` in place of `M·y`, within `|L|·δ₁`. At each fixed `c`, adding the digest to `U` is a
   bijection of `U`, so that prover's messages are distributed as with ideal leaves `H(U, c(y))`, and T5 lifts that too.
   Each witness costs `|L|·δ₁`: `Session.adaptive_prefinal_hm96`, within `2·|L|·δ₁`, with `L` every leaf of the
   pre-final view: level 0's, both pads trees', and the `2J` commitments to `τ`.
3. **`V*`'s own tape**, uniform and independent of the prover's: `prCoin_avg_close` averages the bound over it,
   `Session.adaptive_prefinal_hm96_tape`.

The prover is given by its messages at fixed answers, which is how the code computes them: `msgs w ω c leaves`, the
messages on witness `w`, randomness `ω` and the leaves it commits; and in world `W₀`, `out c` of the cut view, with the
tables' randomness read from `(ω, U)` through a map `e c` with fibers of one size (the tables at `c`'s coins open
coin-dependent positions, so their randomness types vary with `c`, and `e c` keeps the part they read).
-/

namespace FlockSoundness.ZK

open Finset Game
open scoped ENNReal

-- As in `SHVZK`: the randomness and the view are products of up to fifteen factors.
set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

section Generic

/-- A strategy run on randomness read through `f`. -/
def Strategy.comap {Ω Ω' M C : Type} {n : ℕ} (P : Strategy Ω M C n) (f : Ω' → Ω) : Strategy Ω' M C n where
  msg x := P.msg (f x)
  causal x := P.causal (f x)

theorem run_comap {Ω Ω' M C : Type} {n : ℕ} [Inhabited M] (P : Strategy Ω M C n) (f : Ω' → Ω) (V : Responder M C n)
    (x : Ω') : run (P.comap f) V x = run P V (f x) := rfl

/-- On one space, equal distributions are equal fiber counts. -/
theorem SameDist.card_eq {Ω V : Type} [Fintype Ω] [Nonempty Ω] [DecidableEq V] {f g : Ω → V} (h : SameDist f g)
    (v : V) : #{ω | f ω = v} = #{ω | g ω = v} := by
  have e : #{ω | f ω = v} * Fintype.card Ω = #{ω | g ω = v} * Fintype.card Ω := by convert h v
  exact Nat.eq_of_mul_eq_mul_right Fintype.card_pos e

/-- Through a map whose fibers have one size `k`, each fiber of a function is `k` times its fiber on the target. -/
theorem card_comp_of_fiber {Ω R V : Type} [Fintype Ω] [Fintype R] [DecidableEq R] [DecidableEq V] (e : Ω → R)
    {k : ℕ} (he : ∀ r, Nat.card {x // e x = r} = k) (f : R → V) (v : V) :
    #{x | f (e x) = v} = k * #{r | f r = v} := by
  have hk : ∀ r, #{x | e x = r} = k := fun r => by
    rw [← he r, Nat.card_eq_fintype_card, Fintype.card_subtype]
  rw [Finset.card_eq_sum_card_fiberwise (f := e) (t := ({r | f r = v} : Finset R)) (fun x hx => by simpa using hx)]
  have hs : ∀ r ∈ ({r | f r = v} : Finset R), #(({x | f (e x) = v} : Finset Ω).filter fun x => e x = r) = k := by
    intro r hr
    rw [Finset.filter_filter, ← hk r]
    congr 1
    ext x
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hr ⊢
    exact ⟨fun h => h.2, fun h => ⟨h ▸ hr, h⟩⟩
  rw [Finset.sum_congr rfl hs, Finset.sum_const, smul_eq_mul, mul_comm]

/-- **A digest under a fresh uniform mask is a fresh uniform value**: on randomness `(ω, (U_l, y_l)_l)`, messages that read
leaves `H(dig_l(ω) + U_l, c(y_l))` take every value as often as with `H(U_l, c(y_l))`. -/
theorem card_msg_pad {Ω L B Y Cc D V : Type} [Fintype Ω] [Fintype L] [DecidableEq L] [AddCommGroup B] [Fintype B]
    [DecidableEq B] [Fintype Y] [DecidableEq Y] [DecidableEq V] (cs : Y → Cc) (H : B × Cc → D) (m : Ω → (L → D) → V)
    (dig : Ω → L → B) (v : V) :
    #{x : Ω × (L → B × Y) | m x.1 (fun l => H (dig x.1 l + (x.2 l).1, cs (x.2 l).2)) = v}
      = #{x : Ω × (L → B × Y) | m x.1 (fun l => H ((x.2 l).1, cs (x.2 l).2)) = v} := by
  let φ : Ω × (L → B × Y) ≃ Ω × (L → B × Y) :=
    { toFun := fun x => (x.1, fun l => (dig x.1 l + (x.2 l).1, (x.2 l).2))
      invFun := fun x => (x.1, fun l => (-dig x.1 l + (x.2 l).1, (x.2 l).2))
      left_inv := fun x => Prod.ext rfl (funext fun l => Prod.ext (neg_add_cancel_left _ _) rfl)
      right_inv := fun x => Prod.ext rfl (funext fun l => Prod.ext (add_neg_cancel_left _ _) rfl) }
  refine Finset.card_equiv φ (fun x => ?_)
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, φ, Equiv.coe_fn_mk]

/-- **A verifier's own tape**: a bound that holds at every fixed tape `t` holds on a uniform tape drawn independently of
the experiment. -/
theorem prCoin_avg_close {Tp X : Type} [Fintype Tp] [Nonempty Tp] [Fintype X] (A A' : Tp → X → Prop) {ε : ℝ≥0∞}
    (h : ∀ t, prCoin (A t) ≤ prCoin (A' t) + ε) :
    prCoin (fun p : Tp × X => A p.1 p.2) ≤ prCoin (fun p : Tp × X => A' p.1 p.2) + ε := by
  rw [prCoin_prod_split A, prCoin_prod_split A']
  calc avg (fun t => prCoin (A t)) ≤ avg (fun t => prCoin (A' t) + ε) := avg_mono h
    _ = _ := by rw [avg_add, avg_const]

end Generic

namespace Session

section Ideal

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt] [AddCommGroup Seed]
  [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt] [Fintype Seed] [DecidableEq Seed]
  [Fintype Hi] [DecidableEq Hi] {J : ℕ}

/-- **At fixed answers**: messages that are a function of the session's cut pre-view, on the tables' randomness read
through a map with fibers of one size, take every value as often on two witnesses carrying the public region data. -/
theorem card_cut_eq {Ω V : Type} [Fintype Ω] [DecidableEq V] (T : Fin J → Table F K W Salt Seed D Hi VL)
    (rep : ∀ j, Fin (T j).nC → Fin 2) (e : Ω → ∀ j, (T j).Rand) {k : ℕ} (he : ∀ r, Nat.card {x // e x = r} = k)
    (out : (∀ j, (T j).Cut) → V) (w w' : Fin J → W) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hreg' : ∀ j, (T j).sReg (w' j) = (T j).pubReg) (v : V) :
    #{x | out (fun j => (T j).cut (sent T rep j) ((T j).preView (w j) (e x j))) = v}
      = #{x | out (fun j => (T j).cut (sent T rep j) ((T j).preView (w' j) (e x j))) = v} := by
  classical
  have : Nonempty (∀ j, (T j).Rand) := ⟨fun _ => 0⟩
  rw [card_comp_of_fiber e he (fun r => out fun j => (T j).cut (sent T rep j) ((T j).preView (w j) (r j))),
    card_comp_of_fiber e he (fun r => out fun j => (T j).cut (sent T rep j) ((T j).preView (w' j) (r j)))]
  congr 1
  exact ((session_prefinal_indep_cut T rep w w' hreg hreg').comp out).card_eq v

/-- **Lemma C in world `W₀`** (§4.4, exact): against any deterministic verifier `V*` (its tape fixed), two provers whose
messages at every fixed answer vector `c` are the same function `out c` of the session's pre-final view, cut where the
session stops at `c`'s coins, on witnesses `w` and `w'` carrying the public region data, give transcripts that take every
value equally often. `T c` is the session's tables at the coins `c` opens, `rep c` their claims' reps, and `e c` reads
the tables' randomness from the prover's through fibers of one size. -/
theorem adaptive_prefinal_indep {n : ℕ} {Cn M Ω : Type} [Inhabited M] [DecidableEq M] [Fintype Ω]
    (T : (Fin n → Cn) → Fin J → Table F K W Salt Seed D Hi VL) (rep : ∀ c j, Fin (T c j).nC → Fin 2)
    (e : ∀ c, Ω → ∀ j, (T c j).Rand) (k : (Fin n → Cn) → ℕ) (he : ∀ c r, Nat.card {x // e c x = r} = k c)
    (out : ∀ c, (∀ j, (T c j).Cut) → Fin n → M) (w w' : Fin J → W)
    (hreg : ∀ c j, (T c j).sReg (w j) = (T c j).pubReg) (hreg' : ∀ c j, (T c j).sReg (w' j) = (T c j).pubReg)
    (P P' : Strategy Ω M Cn n)
    (hP : ∀ x c, P.msg x c = out c fun j => (T c j).cut (sent (T c) (rep c) j) ((T c j).preView (w j) (e c x j)))
    (hP' : ∀ x c, P'.msg x c = out c fun j => (T c j).cut (sent (T c) (rep c) j) ((T c j).preView (w' j) (e c x j)))
    (V : Responder M Cn n) (τ : Fin n → M) :
    #{x | run P V x = τ} = #{x | run P' V x = τ} := by
  refine card_run_eq P P' V (fun c σ => ?_) τ
  have e1 : #{x | P.msg x c = σ}
      = #{x | out c (fun j => (T c j).cut (sent (T c) (rep c) j) ((T c j).preView (w j) (e c x j))) = σ} := by
    congr 1
    ext x
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, hP]
  have e2 : #{x | P'.msg x c = σ}
      = #{x | out c (fun j => (T c j).cut (sent (T c) (rep c) j) ((T c j).preView (w' j) (e c x j))) = σ} := by
    congr 1
    ext x
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, hP']
  rw [e1, e2]
  exact card_cut_eq (T c) (rep c) (e c) (he c) (out c) w w' (hreg c) (hreg' c) σ

end Ideal

section Real

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt] [AddCommGroup B]
  [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B]
  [Fintype Hi] [DecidableEq Hi] {J : ℕ}

/-- **Lemma C** (§4.4, `δ_pre ≤ 2·δ₁·(N_hid + 2J)`): against any deterministic verifier `V*` (its tape fixed), every event
on the real prover's pre-final transcript has probability on witness `w` at most its probability on witness `w'` plus
`2·|L|·δ₁`, for two witnesses carrying the public region data (the dummy witness is one). `L` is every leaf of the
pre-final view, each a real hm96 leaf `H(dig + M·y, c(y))` under a fresh salt `y` the transcript doesn't otherwise read.

The prover (`P w`) is given on its randomness `ω` and each leaf's pair `(M·y, c(y))`, as `msgs` of the leaves it commits,
whose digests `dig` may read the answers (`hP`). In world `W₀` (each leaf `H(U, c(y))` for fresh `U`), its messages at every
fixed answer vector `c` are `out c` of the session's pre-final view cut where the session stops at `c`'s coins (`hI`), with
the tables' randomness read from `(ω, (U, y))` through `e c`, whose fibers have one size. -/
theorem adaptive_prefinal_hm96 {n : ℕ} {Cn M Cc Ω L : Type} [Inhabited M] [DecidableEq M] [Fintype Ω] [Nonempty Ω]
    [Fintype L] [DecidableEq L] (T : (Fin n → Cn) → Fin J → Table F K W Salt (B × Salt) D Hi VL)
    (rep : ∀ c j, Fin (T c j).nC → Fin 2) (My : Salt → B) (cs : Salt → Cc) {δ₁ : ℝ≥0∞}
    (hT1 : Assumptions.Hm96Hiding My cs δ₁) (H : B × Cc → D)
    (msgs : (Fin J → W) → Ω → (Fin n → Cn) → (L → D) → Fin n → M) (dig : (Fin J → W) → Ω → (Fin n → Cn) → L → B)
    (P : (Fin J → W) → Strategy (Ω × (L → B × Cc)) M Cn n)
    (hP : ∀ w ω p c, (P w).msg (ω, p) c = msgs w ω c fun l => H (dig w ω c l + (p l).1, (p l).2))
    (e : ∀ c, Ω × (L → B × Salt) → ∀ j, (T c j).Rand) (k : (Fin n → Cn) → ℕ)
    (he : ∀ c r, Nat.card {x // e c x = r} = k c) (out : ∀ c, (∀ j, (T c j).Cut) → Fin n → M)
    (hI : ∀ w ω u c, msgs w ω c (fun l => H ((u l).1, cs (u l).2))
      = out c fun j => (T c j).cut (sent (T c) (rep c) j) ((T c j).preView (w j) (e c (ω, u) j)))
    (w w' : Fin J → W) (hreg : ∀ c j, (T c j).sReg (w j) = (T c j).pubReg)
    (hreg' : ∀ c j, (T c j).sReg (w' j) = (T c j).pubReg) (V : Responder M Cn n) (G : (Fin n → M) → Prop) :
    prCoin (fun p : Ω × (L → Salt) => G (run (P w) V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
      ≤ prCoin (fun p : Ω × (L → Salt) => G (run (P w') V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
        + 2 * Fintype.card L * δ₁ := by
  classical
  have : Nonempty Salt := ⟨0⟩
  let f : Ω × (L → B × Salt) → Ω × (L → B × Cc) := fun x => (x.1, fun l => ((x.2 l).1, cs (x.2 l).2))
  -- each witness's prover with each leaf's `M·y` replaced by a fresh uniform `U`: the same transcript distribution
  have hcount : ∀ w0 : Fin J → W, ∀ c σ, #{x | ((P w0).comap f).msg x c = σ}
      = #{x | out c (fun j => (T c j).cut (sent (T c) (rep c) j) ((T c j).preView (w0 j) (e c x j))) = σ} := by
    intro w0 c σ
    calc #{x | ((P w0).comap f).msg x c = σ}
        = #{x : Ω × (L → B × Salt) | msgs w0 x.1 c (fun l => H (dig w0 x.1 c l + (x.2 l).1, cs (x.2 l).2)) = σ} := by
          congr 1
          ext x
          simp only [Finset.mem_filter, Finset.mem_univ, true_and, Strategy.comap, f, hP]
      _ = #{x : Ω × (L → B × Salt) | msgs w0 x.1 c (fun l => H ((x.2 l).1, cs (x.2 l).2)) = σ} :=
          card_msg_pad cs H (fun ω leaves => msgs w0 ω c leaves) (fun ω => dig w0 ω c) σ
      _ = _ := by
          congr 1
          ext x
          simp only [Finset.mem_filter, Finset.mem_univ, true_and, hI]
  have hruns : ∀ τ, #{x | run ((P w).comap f) V x = τ} = #{x | run ((P w').comap f) V x = τ} :=
    card_run_eq _ _ V fun c σ => by
      rw [hcount w c σ, hcount w' c σ]
      exact card_cut_eq (T c) (rep c) (e c) (he c) (out c) w w' (hreg c) (hreg' c) σ
  have hpad : prCoin (fun x : Ω × (L → B × Salt) => G (run (P w) V (f x)))
      = prCoin (fun x : Ω × (L → B × Salt) => G (run (P w') V (f x))) :=
    (SameDist.of_card_fiber_eq hruns).prCoin_eq G
  -- T1 for each witness, with digest 0: the pair `(M·y, c(y))` against `(U, c(y))`
  have h1 := ideal_leaves_swap_fintype My cs hT1 (id : B × Cc → B × Cc) (fun (_ : Ω) (_ : L) => (0 : B))
    (fun ω leaves => G (run (P w) V (ω, leaves)))
  have h2 := real_leaves_swap_fintype My cs hT1 (id : B × Cc → B × Cc) (fun (_ : Ω) (_ : L) => (0 : B))
    (fun ω leaves => G (run (P w') V (ω, leaves)))
  simp only [id, zero_add] at h1 h2
  calc prCoin (fun p : Ω × (L → Salt) => G (run (P w) V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
      ≤ prCoin (fun x : Ω × (L → B × Salt) => G (run (P w) V (f x))) + Fintype.card L * δ₁ := h1
    _ = prCoin (fun x : Ω × (L → B × Salt) => G (run (P w') V (f x))) + Fintype.card L * δ₁ := by rw [hpad]
    _ ≤ (prCoin (fun p : Ω × (L → Salt) => G (run (P w') V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
          + Fintype.card L * δ₁) + Fintype.card L * δ₁ := add_le_add h2 le_rfl
    _ = _ := by ring

/-- **Lemma C against a randomized verifier**: `adaptive_prefinal_hm96` with `V*`'s tape `t` uniform on `Tp`, drawn
independently of the prover's randomness and salts. -/
theorem adaptive_prefinal_hm96_tape {n : ℕ} {Cn M Cc Ω L Tp : Type} [Inhabited M] [DecidableEq M] [Fintype Ω]
    [Nonempty Ω] [Fintype L] [DecidableEq L] [Fintype Tp] [Nonempty Tp]
    (T : (Fin n → Cn) → Fin J → Table F K W Salt (B × Salt) D Hi VL)
    (rep : ∀ c j, Fin (T c j).nC → Fin 2) (My : Salt → B) (cs : Salt → Cc) {δ₁ : ℝ≥0∞}
    (hT1 : Assumptions.Hm96Hiding My cs δ₁) (H : B × Cc → D)
    (msgs : (Fin J → W) → Ω → (Fin n → Cn) → (L → D) → Fin n → M) (dig : (Fin J → W) → Ω → (Fin n → Cn) → L → B)
    (P : (Fin J → W) → Strategy (Ω × (L → B × Cc)) M Cn n)
    (hP : ∀ w ω p c, (P w).msg (ω, p) c = msgs w ω c fun l => H (dig w ω c l + (p l).1, (p l).2))
    (e : ∀ c, Ω × (L → B × Salt) → ∀ j, (T c j).Rand) (k : (Fin n → Cn) → ℕ)
    (he : ∀ c r, Nat.card {x // e c x = r} = k c) (out : ∀ c, (∀ j, (T c j).Cut) → Fin n → M)
    (hI : ∀ w ω u c, msgs w ω c (fun l => H ((u l).1, cs (u l).2))
      = out c fun j => (T c j).cut (sent (T c) (rep c) j) ((T c j).preView (w j) (e c (ω, u) j)))
    (w w' : Fin J → W) (hreg : ∀ c j, (T c j).sReg (w j) = (T c j).pubReg)
    (hreg' : ∀ c j, (T c j).sReg (w' j) = (T c j).pubReg) (V : Tp → Responder M Cn n)
    (G : Tp → (Fin n → M) → Prop) :
    prCoin (fun q : Tp × (Ω × (L → Salt)) =>
        G q.1 (run (P w) (V q.1) (q.2.1, fun l => (My (q.2.2 l), cs (q.2.2 l)))))
      ≤ prCoin (fun q : Tp × (Ω × (L → Salt)) =>
          G q.1 (run (P w') (V q.1) (q.2.1, fun l => (My (q.2.2 l), cs (q.2.2 l)))))
        + 2 * Fintype.card L * δ₁ :=
  prCoin_avg_close (fun t (p : Ω × (L → Salt)) => G t (run (P w) (V t) (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
    (fun t (p : Ω × (L → Salt)) => G t (run (P w') (V t) (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
    fun t => adaptive_prefinal_hm96 T rep My cs hT1 H msgs dig P hP e k he out hI w w' hreg hreg' (V t) (G t)

end Real

end Session

end FlockSoundness.ZK
