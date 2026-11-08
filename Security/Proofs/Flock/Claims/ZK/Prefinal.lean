module

public import Proofs.Flock.Claims.ZK.Shvzk
public import Definitions.Flock.ZK.SHVZK
public import Proofs.Flock.Claims.ZK.Prefix
public import Proofs.Flock.Soundness.ZK.Dist
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.RingTheory.WittVector.IsPoly
import Mathlib.Tactic.ENatToNat
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Soundness.ZK.AdaptivePrefinal
import Proofs.Flock.Soundness.ZK.GK.Link
import Proofs.Flock.Soundness.ZK.Hiding
import Proofs.Flock.Soundness.ZK.RealLeaves
import Proofs.Flock.Soundness.ZK.Refusal
import Proofs.Flock.Soundness.ZK.Session
import Std.Tactic.BVDecide.Normalize.Prop
import VCVio.Prelude.Core

/-!
# Lemma C at tables of their own types

`Session.adaptive_prefinal_hm96` (Lemma C) takes `J` tables of one type. The tables `verify --zk` sets up each have their
own witness, hidden-value and Ligerito types (`--session-tables J`). Lemma C reads the tables only through world `W₀`'s
witness independence at each fixed answer vector (`Session.card_cut_eq`), so:

* `prefinal_of_ideal`: Lemma C for a prover whose messages at fixed answers, at ideal leaves `H(U, c(y))`, take every
  value equally often on the two witnesses; the leaves' hybrid (T1) and the adaptive verifier (T5) are
  `Session.adaptive_prefinal_hm96`'s.
* `prefinal_sw_of_ideal`: the same for a prover `S` whose messages read the salts of leaves `L₀` beyond their leaves:
  `L₁` goes ideal at `S`, `S` meets the honest prover with real `L₀` leaves, `L₀` goes ideal there, and the witnesses
  swap at ideal leaves; the bound is `2·|L₀ ⊕ L₁|·δ₁` both ways.
* `famSent`, `famSent_admits`, `famCard_cut_eq`: the session's schedule (`Session.sent`) and Lemma A at every coin vector
  (`Session.session_prefinal_indep_cut`) for tables of their own types, from each table's own Lemma A
  (`Table.table_prefinal_indep_cut`).
* `Proto.runs_close_of_ideal`, `Proto.refused_close_of_ideal`: `Proto.runs_close_hm96`'s and `Proto.shvzk_refused_hm96`'s
  bounds from `prefinal_of_ideal`; `Proto.runs_close_sw_of_ideal`, `Proto.refused_close_sw_of_ideal`: the same from
  `prefinal_sw_of_ideal`.
-/

@[expose] public section

namespace Claims.ZKMal

open Claims Finset

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- instance search over the ZK view's and coins' product types

/-! ## Lemma C from world `W₀` -/

/-- **Lemma C from world `W₀`'s witness independence** (`Session.adaptive_prefinal_hm96` with `hW₀` for its `hI`, `he`,
`hreg`): against any deterministic verifier, every event on the prover's transcript has probability on `w` at most its
probability on `w'` plus `2·|L|·δ₁`, when at every fixed answer vector its messages at ideal leaves `H(U, c(y))` take
every value equally often on `w` and on `w'`. -/
theorem prefinal_of_ideal {n : ℕ} {Cn M Cc Ω L Salt B D Wt : Type} [Inhabited M] [DecidableEq M] [Fintype Ω]
    [Nonempty Ω] [Fintype L] [DecidableEq L] [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt]
    [Fintype B] [DecidableEq B] (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁)
    (Hl : B × Cc → D) (msgs : Wt → Ω → (Fin n → Cn) → (L → D) → Fin n → M)
    (dig : Wt → Ω → (Fin n → Cn) → L → B) (P : Wt → FlockSoundness.ZK.Strategy (Ω × (L → B × Cc)) M Cn n)
    (hP : ∀ w ω p c, (P w).msg (ω, p) c = msgs w ω c fun l => Hl (dig w ω c l + (p l).1, (p l).2)) (w w' : Wt)
    (hW₀ : ∀ c σ, #{u : Ω × (L → B × Salt) | msgs w u.1 c (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ}
      = #{u : Ω × (L → B × Salt) | msgs w' u.1 c (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ})
    (V : FlockSoundness.ZK.Responder M Cn n) (E : (Fin n → M) → Prop) :
    FlockSoundness.Game.prCoin (fun p : Ω × (L → Salt) =>
        E (FlockSoundness.ZK.run (P w) V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
      ≤ FlockSoundness.Game.prCoin (fun p : Ω × (L → Salt) =>
          E (FlockSoundness.ZK.run (P w') V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
        + 2 * Fintype.card L * δ₁ := by
  classical
  have : Nonempty Salt := ⟨0⟩
  let f : Ω × (L → B × Salt) → Ω × (L → B × Cc) := fun x => (x.1, fun l => ((x.2 l).1, cs (x.2 l).2))
  have hcount : ∀ w0 : Wt, ∀ c σ, #{x | ((P w0).comap f).msg x c = σ}
      = #{u : Ω × (L → B × Salt) | msgs w0 u.1 c (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ} := by
    intro w0 c σ
    calc #{x | ((P w0).comap f).msg x c = σ}
        = #{x : Ω × (L → B × Salt) | msgs w0 x.1 c (fun l => Hl (dig w0 x.1 c l + (x.2 l).1, cs (x.2 l).2)) = σ} := by
          congr 1
          ext x
          simp only [Finset.mem_filter, Finset.mem_univ, true_and, FlockSoundness.ZK.Strategy.comap, f, hP]
      _ = _ := FlockSoundness.ZK.card_msg_pad cs Hl (fun ω leaves => msgs w0 ω c leaves) (fun ω => dig w0 ω c) σ
  have hruns : ∀ τ, #{x | FlockSoundness.ZK.run ((P w).comap f) V x = τ}
      = #{x | FlockSoundness.ZK.run ((P w').comap f) V x = τ} :=
    FlockSoundness.ZK.card_run_eq _ _ V fun c σ => by rw [hcount w c σ, hcount w' c σ]; exact hW₀ c σ
  have hpad : FlockSoundness.Game.prCoin (fun x : Ω × (L → B × Salt) => E (FlockSoundness.ZK.run (P w) V (f x)))
      = FlockSoundness.Game.prCoin (fun x : Ω × (L → B × Salt) => E (FlockSoundness.ZK.run (P w') V (f x))) :=
    (FlockSoundness.ZK.SameDist.of_card_fiber_eq hruns).prCoin_eq E
  have h1 := FlockSoundness.ZK.ideal_leaves_swap_fintype My cs hT1 (id : B × Cc → B × Cc)
    (fun (_ : Ω) (_ : L) => (0 : B)) (fun ω leaves => E (FlockSoundness.ZK.run (P w) V (ω, leaves)))
  have h2 := FlockSoundness.ZK.real_leaves_swap_fintype My cs hT1 (id : B × Cc → B × Cc)
    (fun (_ : Ω) (_ : L) => (0 : B)) (fun ω leaves => E (FlockSoundness.ZK.run (P w') V (ω, leaves)))
  simp only [id, zero_add] at h1 h2
  calc FlockSoundness.Game.prCoin (fun p : Ω × (L → Salt) =>
        E (FlockSoundness.ZK.run (P w) V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
      ≤ FlockSoundness.Game.prCoin (fun x : Ω × (L → B × Salt) => E (FlockSoundness.ZK.run (P w) V (f x)))
        + Fintype.card L * δ₁ := h1
    _ = FlockSoundness.Game.prCoin (fun x : Ω × (L → B × Salt) => E (FlockSoundness.ZK.run (P w') V (f x)))
        + Fintype.card L * δ₁ := by rw [hpad]
    _ ≤ (FlockSoundness.Game.prCoin (fun p : Ω × (L → Salt) =>
          E (FlockSoundness.ZK.run (P w') V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
          + Fintype.card L * δ₁) + Fintype.card L * δ₁ := add_le_add h2 le_rfl
    _ = _ := by ring

open FlockSoundness FlockSoundness.ZK FlockSoundness.Game in
/-- **Lemma C at a prover whose leaves `L₀` are read before its messages** (`prefinal_of_ideal`, reordered): `S` runs on
randomness `ω`, the salts of the leaves `L₀` (which its messages may read beyond their leaves) and the pairs of the
other leaves `L₁`. When at every fixed answer vector, at ideal `L₁` leaves, `S`'s messages take every value as often as
`P wt`'s with real `L₀` leaves under the same salts, and `P`'s messages at ideal leaves take every value as often on `wt`
and on `w₀`, every event on `S`'s transcript at real leaves is within `2·|L₀ ⊕ L₁|·δ₁` of its probability on `P w₀`'s,
both ways: `L₁` goes ideal at `S`, `L₀` goes ideal at `P wt`, and every leaf goes real at `P w₀`. -/
theorem prefinal_sw_of_ideal {n : ℕ} {Cn M Cc Ω L0 L1 Salt B D Wt : Type} [Inhabited M] [DecidableEq M] [Fintype Ω]
    [Nonempty Ω] [Fintype L0] [DecidableEq L0] [Fintype L1] [DecidableEq L1] [AddCommGroup Salt] [AddCommGroup B]
    [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding My cs δ₁) (Hl : B × Cc → D)
    (msgs : Wt → Ω → (Fin n → Cn) → (L0 ⊕ L1 → D) → Fin n → M) (dig : Wt → Ω → (Fin n → Cn) → L0 ⊕ L1 → B)
    (P : Wt → Strategy (Ω × (L0 ⊕ L1 → B × Cc)) M Cn n)
    (hP : ∀ w ω p c, (P w).msg (ω, p) c = msgs w ω c fun l => Hl (dig w ω c l + (p l).1, (p l).2))
    (msgsS : Ω × (L0 → Salt) → (Fin n → Cn) → (L1 → D) → Fin n → M) (digS : Ω × (L0 → Salt) → (Fin n → Cn) → L1 → B)
    (S : Strategy ((Ω × (L0 → Salt)) × (L1 → B × Cc)) M Cn n)
    (hS : ∀ q p c, S.msg (q, p) c = msgsS q c fun l => Hl (digS q c l + (p l).1, (p l).2)) (wt w₀ : Wt)
    (hsw : ∀ c σ, #{u : (Ω × (L0 → Salt)) × (L1 → B × Salt) | msgsS u.1 c (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ}
      = #{u : (Ω × (L0 → Salt)) × (L1 → B × Salt) | msgs wt u.1.1 c (Sum.elim
          (fun l => Hl (dig wt u.1.1 c (.inl l) + My (u.1.2 l), cs (u.1.2 l))) fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ})
    (hW₀ : ∀ c σ, #{u : Ω × (L0 ⊕ L1 → B × Salt) | msgs wt u.1 c (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ}
      = #{u : Ω × (L0 ⊕ L1 → B × Salt) | msgs w₀ u.1 c (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ})
    (V : Responder M Cn n) (E : (Fin n → M) → Prop) :
    prCoin (fun p : Ω × (L0 ⊕ L1 → Salt) =>
        E (ZK.run S V ((p.1, fun l => p.2 (.inl l)), fun l => (My (p.2 (.inr l)), cs (p.2 (.inr l))))))
      ≤ prCoin (fun p : Ω × (L0 ⊕ L1 → Salt) => E (ZK.run (P w₀) V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
        + 2 * Fintype.card (L0 ⊕ L1) * δ₁
    ∧ prCoin (fun p : Ω × (L0 ⊕ L1 → Salt) => E (ZK.run (P w₀) V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
      ≤ prCoin (fun p : Ω × (L0 ⊕ L1 → Salt) =>
          E (ZK.run S V ((p.1, fun l => p.2 (.inl l)), fun l => (My (p.2 (.inr l)), cs (p.2 (.inr l))))))
        + 2 * Fintype.card (L0 ⊕ L1) * δ₁ := by
  classical
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty B := ⟨0⟩
  let eA : (Ω × (L0 → Salt)) × (L1 → Salt) ≃ Ω × (L0 ⊕ L1 → Salt) :=
    { toFun := fun p => (p.1.1, Sum.elim p.1.2 p.2)
      invFun := fun p => ((p.1, fun l => p.2 (.inl l)), fun l => p.2 (.inr l))
      left_inv := fun _ => rfl
      right_inv := fun p => Prod.ext rfl (funext fun l => by rcases l with l | l <;> rfl) }
  have hA : prCoin (fun p : Ω × (L0 ⊕ L1 → Salt) =>
        E (ZK.run S V ((p.1, fun l => p.2 (.inl l)), fun l => (My (p.2 (.inr l)), cs (p.2 (.inr l))))))
      = prCoin (fun p : (Ω × (L0 → Salt)) × (L1 → Salt) =>
          E (ZK.run S V (p.1, fun l => (My (p.2 l), cs (p.2 l))))) :=
    (prCoin_congr_equiv eA (fun p : Ω × (L0 ⊕ L1 → Salt) =>
      E (ZK.run S V ((p.1, fun l => p.2 (.inl l)), fun l => (My (p.2 (.inr l)), cs (p.2 (.inr l))))))).symm
  have s1 := ideal_leaves_swap_fintype My cs hT1 (id : B × Cc → B × Cc)
    (fun (_ : Ω × (L0 → Salt)) (_ : L1) => (0 : B)) (fun q leaves => E (ZK.run S V (q, leaves)))
  have s2 := real_leaves_swap_fintype My cs hT1 (id : B × Cc → B × Cc)
    (fun (_ : Ω × (L0 → Salt)) (_ : L1) => (0 : B)) (fun q leaves => E (ZK.run S V (q, leaves)))
  simp only [id, zero_add] at s1 s2
  let f1 : (Ω × (L0 → Salt)) × (L1 → B × Salt) → (Ω × (L0 → Salt)) × (L1 → B × Cc) :=
    fun x => (x.1, fun l => ((x.2 l).1, cs (x.2 l).2))
  let g : (Ω × (L0 → Salt)) × (L1 → B × Salt) → Ω × (L0 ⊕ L1 → B × Cc) :=
    fun x => (x.1.1, Sum.elim (fun l => (My (x.1.2 l), cs (x.1.2 l))) fun l => ((x.2 l).1, cs (x.2 l).2))
  have hcS : ∀ c σ, #{x | (S.comap f1).msg x c = σ}
      = #{u : (Ω × (L0 → Salt)) × (L1 → B × Salt) | msgsS u.1 c (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ} := by
    intro c σ
    calc #{x | (S.comap f1).msg x c = σ}
        = #{x : (Ω × (L0 → Salt)) × (L1 → B × Salt) |
            msgsS x.1 c (fun l => Hl (digS x.1 c l + (x.2 l).1, cs (x.2 l).2)) = σ} := by
          congr 1
          ext x
          simp only [Finset.mem_filter, Finset.mem_univ, true_and, Strategy.comap, f1, hS]
      _ = _ := card_msg_pad cs Hl (fun q leaves => msgsS q c leaves) (fun q => digS q c) σ
  have hcH : ∀ c σ, #{x | ((P wt).comap g).msg x c = σ}
      = #{u : (Ω × (L0 → Salt)) × (L1 → B × Salt) | msgs wt u.1.1 c (Sum.elim
          (fun l => Hl (dig wt u.1.1 c (.inl l) + My (u.1.2 l), cs (u.1.2 l))) fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ} := by
    intro c σ
    calc #{x | ((P wt).comap g).msg x c = σ}
        = #{x : (Ω × (L0 → Salt)) × (L1 → B × Salt) | msgs wt x.1.1 c (Sum.elim
            (fun l => Hl (dig wt x.1.1 c (.inl l) + My (x.1.2 l), cs (x.1.2 l)))
            fun l => Hl (dig wt x.1.1 c (.inr l) + (x.2 l).1, cs (x.2 l).2)) = σ} := by
          congr 1
          ext x
          simp only [Finset.mem_filter, Finset.mem_univ, true_and, Strategy.comap, g, hP]
          have e : (fun l => Hl (dig wt x.1.1 c l + ((Sum.elim (fun l => (My (x.1.2 l), cs (x.1.2 l)))
              fun l => ((x.2 l).1, cs (x.2 l).2)) l).1, ((Sum.elim (fun l => (My (x.1.2 l), cs (x.1.2 l)))
              fun l => ((x.2 l).1, cs (x.2 l).2)) l).2)) = Sum.elim
              (fun l => Hl (dig wt x.1.1 c (.inl l) + My (x.1.2 l), cs (x.1.2 l)))
              fun l => Hl (dig wt x.1.1 c (.inr l) + (x.2 l).1, cs (x.2 l).2) :=
            funext fun l => by rcases l with l | l <;> rfl
          rw [e]
      _ = _ := card_msg_pad cs Hl (fun (q : Ω × (L0 → Salt)) (leaves : L1 → D) => msgs wt q.1 c (Sum.elim
            (fun l => Hl (dig wt q.1 c (.inl l) + My (q.2 l), cs (q.2 l))) leaves))
            (fun (q : Ω × (L0 → Salt)) (l : L1) => dig wt q.1 c (.inr l)) σ
  have hruns : ∀ τ, #{x | ZK.run (S.comap f1) V x = τ} = #{x | ZK.run ((P wt).comap g) V x = τ} :=
    card_run_eq _ _ V fun c σ => by rw [hcS c σ, hcH c σ]; exact hsw c σ
  have hsg : prCoin (fun x : (Ω × (L0 → Salt)) × (L1 → B × Salt) => E (ZK.run S V (f1 x)))
      = prCoin (fun x : (Ω × (L0 → Salt)) × (L1 → B × Salt) => E (ZK.run (P wt) V (g x))) :=
    (SameDist.of_card_fiber_eq hruns).prCoin_eq E
  let eC : (Ω × (L1 → B × Salt)) × (L0 → Salt) ≃ (Ω × (L0 → Salt)) × (L1 → B × Salt) :=
    { toFun := fun p => ((p.1.1, p.2), p.1.2)
      invFun := fun p => ((p.1.1, p.2), p.1.2)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  let eD : (Ω × (L1 → B × Salt)) × (L0 → B × Salt) ≃ Ω × (L0 ⊕ L1 → B × Salt) :=
    { toFun := fun p => (p.1.1, Sum.elim p.2 p.1.2)
      invFun := fun p => ((p.1, fun l => p.2 (.inr l)), fun l => p.2 (.inl l))
      left_inv := fun _ => rfl
      right_inv := fun p => Prod.ext rfl (funext fun l => by rcases l with l | l <;> rfl) }
  let G0 : Ω × (L1 → B × Salt) → (L0 → B × Cc) → Prop := fun q pairs =>
    E (ZK.run (P wt) V (q.1, Sum.elim pairs fun l => ((q.2 l).1, cs (q.2 l).2)))
  have t1 := ideal_leaves_swap_fintype My cs hT1 (id : B × Cc → B × Cc)
    (fun (_ : Ω × (L1 → B × Salt)) (_ : L0) => (0 : B)) G0
  have t2 := real_leaves_swap_fintype My cs hT1 (id : B × Cc → B × Cc)
    (fun (_ : Ω × (L1 → B × Salt)) (_ : L0) => (0 : B)) G0
  simp only [id, zero_add] at t1 t2
  let f : Ω × (L0 ⊕ L1 → B × Salt) → Ω × (L0 ⊕ L1 → B × Cc) := fun x => (x.1, fun l => ((x.2 l).1, cs (x.2 l).2))
  have hI : prCoin (fun p : (Ω × (L1 → B × Salt)) × (L0 → Salt) => G0 p.1 fun i => (My (p.2 i), cs (p.2 i)))
      = prCoin (fun x : (Ω × (L0 → Salt)) × (L1 → B × Salt) => E (ZK.run (P wt) V (g x))) :=
    prCoin_congr_equiv eC (fun x : (Ω × (L0 → Salt)) × (L1 → B × Salt) => E (ZK.run (P wt) V (g x)))
  have hJ : prCoin (fun p : (Ω × (L1 → B × Salt)) × (L0 → B × Salt) => G0 p.1 fun i => ((p.2 i).1, cs (p.2 i).2))
      = prCoin (fun x : Ω × (L0 ⊕ L1 → B × Salt) => E (ZK.run (P wt) V (f x))) := by
    refine Eq.trans ?_ (prCoin_congr_equiv eD (fun x : Ω × (L0 ⊕ L1 → B × Salt) => E (ZK.run (P wt) V (f x))))
    congr 1
    funext p
    have e : (fun l => (((Sum.elim p.2 p.1.2 : L0 ⊕ L1 → B × Salt) l).1,
        cs ((Sum.elim p.2 p.1.2 : L0 ⊕ L1 → B × Salt) l).2)) =
        Sum.elim (fun i => ((p.2 i).1, cs (p.2 i).2)) fun l => ((p.1.2 l).1, cs (p.1.2 l).2) :=
      funext fun l => by rcases l with l | l <;> rfl
    show E (ZK.run (P wt) V (p.1.1, Sum.elim (fun i => ((p.2 i).1, cs (p.2 i).2)) fun l => ((p.1.2 l).1, cs (p.1.2 l).2)))
      = E (ZK.run (P wt) V (p.1.1, fun l => (((Sum.elim p.2 p.1.2 : L0 ⊕ L1 → B × Salt) l).1,
        cs ((Sum.elim p.2 p.1.2 : L0 ⊕ L1 → B × Salt) l).2)))
    rw [e]
  rw [hI, hJ] at t1 t2
  have hcount : ∀ w0 : Wt, ∀ c σ, #{x | ((P w0).comap f).msg x c = σ}
      = #{u : Ω × (L0 ⊕ L1 → B × Salt) | msgs w0 u.1 c (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ} := by
    intro w0 c σ
    calc #{x | ((P w0).comap f).msg x c = σ}
        = #{x : Ω × (L0 ⊕ L1 → B × Salt) |
            msgs w0 x.1 c (fun l => Hl (dig w0 x.1 c l + (x.2 l).1, cs (x.2 l).2)) = σ} := by
          congr 1
          ext x
          simp only [Finset.mem_filter, Finset.mem_univ, true_and, Strategy.comap, f, hP]
      _ = _ := card_msg_pad cs Hl (fun ω leaves => msgs w0 ω c leaves) (fun ω => dig w0 ω c) σ
  have hruns₀ : ∀ τ, #{x | ZK.run ((P wt).comap f) V x = τ} = #{x | ZK.run ((P w₀).comap f) V x = τ} :=
    card_run_eq _ _ V fun c σ => by rw [hcount wt c σ, hcount w₀ c σ]; exact hW₀ c σ
  have hpad : prCoin (fun x : Ω × (L0 ⊕ L1 → B × Salt) => E (ZK.run (P wt) V (f x)))
      = prCoin (fun x : Ω × (L0 ⊕ L1 → B × Salt) => E (ZK.run (P w₀) V (f x))) :=
    (SameDist.of_card_fiber_eq hruns₀).prCoin_eq E
  have r1 := ideal_leaves_swap_fintype My cs hT1 (id : B × Cc → B × Cc)
    (fun (_ : Ω) (_ : L0 ⊕ L1) => (0 : B)) (fun ω leaves => E (ZK.run (P w₀) V (ω, leaves)))
  have r2 := real_leaves_swap_fintype My cs hT1 (id : B × Cc → B × Cc)
    (fun (_ : Ω) (_ : L0 ⊕ L1) => (0 : B)) (fun ω leaves => E (ZK.run (P w₀) V (ω, leaves)))
  simp only [id, zero_add] at r1 r2
  have hc : (Fintype.card (L0 ⊕ L1) : ENNReal) = Fintype.card L0 + Fintype.card L1 := by
    rw [Fintype.card_sum]
    push_cast
    rfl
  rw [hA, hc]
  constructor
  · calc prCoin (fun p : (Ω × (L0 → Salt)) × (L1 → Salt) => E (ZK.run S V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
        ≤ prCoin (fun x : (Ω × (L0 → Salt)) × (L1 → B × Salt) => E (ZK.run S V (f1 x))) + Fintype.card L1 * δ₁ := s1
      _ = prCoin (fun x : (Ω × (L0 → Salt)) × (L1 → B × Salt) => E (ZK.run (P wt) V (g x)))
          + Fintype.card L1 * δ₁ := by rw [hsg]
      _ ≤ (prCoin (fun x : Ω × (L0 ⊕ L1 → B × Salt) => E (ZK.run (P wt) V (f x))) + Fintype.card L0 * δ₁)
          + Fintype.card L1 * δ₁ := add_le_add t1 le_rfl
      _ = (prCoin (fun x : Ω × (L0 ⊕ L1 → B × Salt) => E (ZK.run (P w₀) V (f x))) + Fintype.card L0 * δ₁)
          + Fintype.card L1 * δ₁ := by rw [hpad]
      _ ≤ ((prCoin (fun p : Ω × (L0 ⊕ L1 → Salt) => E (ZK.run (P w₀) V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
            + Fintype.card (L0 ⊕ L1) * δ₁) + Fintype.card L0 * δ₁) + Fintype.card L1 * δ₁ :=
          add_le_add (add_le_add r2 le_rfl) le_rfl
      _ = _ := by rw [hc]; ring
  · calc prCoin (fun p : Ω × (L0 ⊕ L1 → Salt) => E (ZK.run (P w₀) V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
        ≤ prCoin (fun x : Ω × (L0 ⊕ L1 → B × Salt) => E (ZK.run (P w₀) V (f x))) + Fintype.card (L0 ⊕ L1) * δ₁ := r1
      _ = prCoin (fun x : Ω × (L0 ⊕ L1 → B × Salt) => E (ZK.run (P wt) V (f x)))
          + Fintype.card (L0 ⊕ L1) * δ₁ := by rw [hpad]
      _ ≤ (prCoin (fun x : (Ω × (L0 → Salt)) × (L1 → B × Salt) => E (ZK.run (P wt) V (g x))) + Fintype.card L0 * δ₁)
          + Fintype.card (L0 ⊕ L1) * δ₁ := add_le_add t2 le_rfl
      _ = (prCoin (fun x : (Ω × (L0 → Salt)) × (L1 → B × Salt) => E (ZK.run S V (f1 x))) + Fintype.card L0 * δ₁)
          + Fintype.card (L0 ⊕ L1) * δ₁ := by rw [hsg]
      _ ≤ ((prCoin (fun p : (Ω × (L0 → Salt)) × (L1 → Salt) => E (ZK.run S V (p.1, fun l => (My (p.2 l), cs (p.2 l)))))
            + Fintype.card L1 * δ₁) + Fintype.card L0 * δ₁) + Fintype.card (L0 ⊕ L1) * δ₁ :=
          add_le_add (add_le_add s2 le_rfl) le_rfl
      _ = _ := by rw [hc]; ring

/-! ## The schedule and Lemma A, for tables of their own types -/

/-- **The check at table `j`, rep `r`, phase `p`** (`Session.guard`), for tables of their own types. -/
def famGuard {F K Salt Seed D : Type} [Field F] [Field K] [CharP F 2] {J : ℕ} {Wf Hi VL : Fin J → Type}
    (T : ∀ j, FlockSoundness.ZK.Table F K (Wf j) Salt Seed D (Hi j) (VL j)) (rep : ∀ j, Fin (T j).nC → Fin 2)
    (j : Fin J) (r : Fin 2) : FlockSoundness.ZK.Session.Phase → Prop
  | .opening => (T j).RankOn fun k => rep j k ≤ r
  | .restart => Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2
  | .inner => (T j).β r ≠ 0

/-- **The session reaches past table `j`, rep `r`, phase `p`** (`Session.passes`): every check up to it in the session's
order passes. -/
def famPasses {F K Salt Seed D : Type} [Field F] [Field K] [CharP F 2] {J : ℕ} {Wf Hi VL : Fin J → Type}
    (T : ∀ j, FlockSoundness.ZK.Table F K (Wf j) Salt Seed D (Hi j) (VL j)) (rep : ∀ j, Fin (T j).nC → Fin 2)
    (j : Fin J) (r : Fin 2) (p : FlockSoundness.ZK.Session.Phase) : Prop :=
  ∀ j' r' p', FlockSoundness.ZK.Session.pos j' r' p' ≤ FlockSoundness.ZK.Session.pos j r p → famGuard T rep j' r' p'

/-- **What table `j` sent** when the session stops at its first failing check (`Session.sent`), for tables of their own
types. -/
def famSent {F K Salt Seed D : Type} [Field F] [Field K] [CharP F 2] {J : ℕ} {Wf Hi VL : Fin J → Type}
    (T : ∀ j, FlockSoundness.ZK.Table F K (Wf j) Salt Seed D (Hi j) (VL j)) (rep : ∀ j, Fin (T j).nC → Fin 2)
    (j : Fin J) : (T j).Sent where
  cl k := famPasses T rep j (rep j k) .opening
  y1 r := famPasses T rep j r .restart
  fin r := famPasses T rep j r .inner

/-- **The schedule** (`Session.sent_admits`): at every coin vector, what each table sent is admitted. -/
theorem famSent_admits {F K Salt Seed D : Type} [Field F] [Field K] [CharP F 2] {J : ℕ} {Wf Hi VL : Fin J → Type}
    (T : ∀ j, FlockSoundness.ZK.Table F K (Wf j) Salt Seed D (Hi j) (VL j)) (rep : ∀ j, Fin (T j).nC → Fin 2)
    (j : Fin J) : (T j).Admits (famSent T rep j) := by
  refine ⟨?_, fun r hr => hr j r .restart le_rfl, fun r hr => hr j r .inner le_rfl⟩
  by_cases h1 : famPasses T rep j 1 .opening
  · refine FlockSoundness.ZK.Table.RankOn.mono (T j) (fun k _ => ?_) (h1 j 1 .opening le_rfl)
    have := (rep j k).isLt
    rw [Fin.le_iff_val_le_val]
    simp only [Fin.isValue, Fin.val_one]
    omega
  · by_cases h0 : famPasses T rep j 0 .opening
    · refine FlockSoundness.ZK.Table.RankOn.mono (T j) (fun k hk => ?_) (h0 j 0 .opening le_rfl)
      have := (rep j k).isLt
      rw [Fin.le_iff_val_le_val]
      by_contra hne
      have e : rep j k = 1 := Fin.ext (by simp only [Fin.isValue, Fin.val_zero, Fin.val_one] at hne ⊢; omega)
      exact h1 (e ▸ hk)
    · refine FlockSoundness.ZK.Table.rankOn_of_forall_not (T j) fun k hk =>
        h0 fun a b c hc => hk a b c (hc.trans ?_)
      simp only [FlockSoundness.ZK.Session.pos, FlockSoundness.ZK.Session.Phase.idx, Fin.isValue, Fin.val_zero]
      omega

/-- **Lemma A at every coin vector, for tables of their own types** (`Session.session_prefinal_indep_cut`): the session's
pre-final views in world `W₀` on two tuples of witnesses carrying the public region data, each table's cut to what it
sent, are equally distributed. -/
theorem famSession_prefinal_indep_cut {F K Salt Seed D : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt]
    [AddCommGroup Seed] [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt] [Fintype Seed]
    [DecidableEq Seed] {J : ℕ} {Wf Hi VL : Fin J → Type} [∀ j, AddCommGroup (Hi j)] [∀ j, Fintype (Hi j)]
    [∀ j, DecidableEq (Hi j)] (T : ∀ j, FlockSoundness.ZK.Table F K (Wf j) Salt Seed D (Hi j) (VL j))
    (rep : ∀ j, Fin (T j).nC → Fin 2) (w w' : ∀ j, Wf j) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hreg' : ∀ j, (T j).sReg (w' j) = (T j).pubReg) :
    FlockSoundness.ZK.SameDist
      (fun ω : ∀ j, (T j).Rand => fun j => (T j).cut (famSent T rep j) ((T j).preView (w j) (ω j)))
      (fun ω : ∀ j, (T j).Rand => fun j => (T j).cut (famSent T rep j) ((T j).preView (w' j) (ω j))) :=
  FlockSoundness.ZK.SameDist.pi (fun j => fun ω => (T j).cut (famSent T rep j) ((T j).preView (w j) ω))
    (fun j => fun ω => (T j).cut (famSent T rep j) ((T j).preView (w' j) ω))
    fun j => (T j).table_prefinal_indep_cut (famSent T rep j) (famSent_admits T rep j) (w j) (w' j) (hreg j) (hreg' j)

/-- **At fixed answers, for tables of their own types** (`Session.card_cut_eq`): messages that are a function of the
session's cut pre-view, on the tables' randomness read through a map with fibers of one size, take every value as often
on two witnesses carrying the public region data. -/
theorem famCard_cut_eq {F K Salt Seed D : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt]
    [AddCommGroup Seed] [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt] [Fintype Seed]
    [DecidableEq Seed] {J : ℕ} {Wf Hi VL : Fin J → Type} [∀ j, AddCommGroup (Hi j)] [∀ j, Fintype (Hi j)]
    [∀ j, DecidableEq (Hi j)] {Ω V : Type} [Fintype Ω] [DecidableEq V]
    (T : ∀ j, FlockSoundness.ZK.Table F K (Wf j) Salt Seed D (Hi j) (VL j)) (rep : ∀ j, Fin (T j).nC → Fin 2)
    (e : Ω → ∀ j, (T j).Rand) {k : ℕ} (he : ∀ r, Nat.card {x // e x = r} = k) (out : (∀ j, (T j).Cut) → V)
    (w w' : ∀ j, Wf j) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hreg' : ∀ j, (T j).sReg (w' j) = (T j).pubReg) (v : V) :
    #{x | out (fun j => (T j).cut (famSent T rep j) ((T j).preView (w j) (e x j))) = v}
      = #{x | out (fun j => (T j).cut (famSent T rep j) ((T j).preView (w' j) (e x j))) = v} := by
  classical
  have : Nonempty (∀ j, (T j).Rand) := ⟨fun _ => 0⟩
  rw [FlockSoundness.ZK.card_comp_of_fiber e he
      (fun r => out fun j => (T j).cut (famSent T rep j) ((T j).preView (w j) (r j))),
    FlockSoundness.ZK.card_comp_of_fiber e he
      (fun r => out fun j => (T j).cut (famSent T rep j) ((T j).preView (w' j) (r j)))]
  congr 1
  exact ((famSession_prefinal_indep_cut T rep w w' hreg hreg').comp out).card_eq v

/-! ## The bounds `Flock.zkR_gkE_hm96` reads -/

namespace Proto

/-- **Lemma C's bound on the runs before the last coin, from world `W₀`** (`Proto.runs_close_hm96` with `hW₀` for its
tables): the prover's messages before the last coin on `w` and on the dummy are `P`'s on real hm96 leaves, and `P`'s
messages at fixed answers and ideal leaves take every value as often on both table witnesses. -/
theorem runs_close_of_ideal {W : Type} (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (w : W)
    {Salt B Cc D Wt Ω L : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B]
    [DecidableEq B] [Fintype Ω] [Nonempty Ω] [Fintype L] [DecidableEq L] (My : Salt → B) (cs : Salt → Cc)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤) (Hl : B × Cc → D)
    (msgs : Wt → Ω → (Fin (G.K x + 1) → Out) → (L → D) → Fin (G.K x + 1) → Msg)
    (dig : Wt → Ω → (Fin (G.K x + 1) → Out) → L → B)
    (P : Wt → FlockSoundness.ZK.Strategy (Ω × (L → B × Cc)) Msg Out (G.K x + 1))
    (hP : ∀ wt ω p a, (P wt).msg (ω, p) a = msgs wt ω a fun l => Hl (dig wt ω a l + (p l).1, (p l).2))
    (wt wt₀ : Wt)
    (hW₀ : ∀ a σ, #{u : Ω × (L → B × Salt) | msgs wt u.1 a (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ}
      = #{u : Ω × (L → B × Salt) | msgs wt₀ u.1 a (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ})
    (eR : G.Rand → Ω × (L → Salt)) (kR : ℕ) (heR : ∀ p, Nat.card {r // eR r = p} = kR)
    (hlink : ∀ r, (G.preStrat H x ν (G.msg x w ν)).msg r =
      (P wt).msg ((eR r).1, fun l => (My ((eR r).2 l), cs ((eR r).2 l))))
    (hlink₀ : ∀ r, (G.preStrat H x ν (G.dmsg x ν)).msg r =
      (P wt₀).msg ((eR r).1, fun l => (My ((eR r).2 l), cs ((eR r).2 l))))
    (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1)) (E : (Fin (G.K x + 1) → Msg) → Prop) :
    |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
      Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))|
      ≤ 2 * Fintype.card L * δ₁.toReal := by
  have : Nonempty Salt := ⟨0⟩
  have eW : Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L → Salt) =>
        E (FlockSoundness.ZK.run (P wt) V (p.1, fun l => (My (p.2 l), cs (p.2 l))))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    congr 1
    funext r
    rw [run_congr _ _ V r _ (hlink r)]
  have eD : Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r)) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L → Salt) =>
        E (FlockSoundness.ZK.run (P wt₀) V (p.1, fun l => (My (p.2 l), cs (p.2 l))))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    congr 1
    funext r
    rw [run_congr _ _ V r _ (hlink₀ r)]
  have h1 := prefinal_of_ideal My cs hT1 Hl msgs dig P hP wt wt₀ hW₀ V E
  have h2 := prefinal_of_ideal My cs hT1 Hl msgs dig P hP wt₀ wt (fun a σ => (hW₀ a σ).symm) V E
  have hε : 2 * (Fintype.card L : ENNReal) * δ₁ ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top (by simp) (ENNReal.natCast_ne_top _)) hδ₁
  have hεr : (2 * (Fintype.card L : ENNReal) * δ₁).toReal = 2 * Fintype.card L * δ₁.toReal := by
    simp [ENNReal.toReal_mul]
  have a := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h1
  have b := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h2
  rw [hεr] at a b
  rw [eW, eD, abs_le]
  constructor <;> linarith

/-- **Lemma C at a coin vector where the prover refuses, from world `W₀`** (`Proto.shvzk_refused_hm96`'s bound with `hW₀`
for its tables): the honest prover's messages at `c` and the dummy-witness prover's are `msgs` of real hm96 leaves, and
`msgs` at ideal leaves takes every value as often on both table witnesses. -/
theorem refused_close_of_ideal {W : Type} (G : Proto W) (H : Msg → Msg) (x : Msg) (w : W) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) {Salt B Cc D Wt Ω L : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt]
    [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Ω] [Nonempty Ω] [Fintype L] [DecidableEq L]
    (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Hl : B × Cc → D) (msgs : Wt → Ω → (L → D) → List Msg) (dig : Wt → Ω → L → B) (wt wt₀ : Wt)
    (hW₀ : ∀ σ, #{u : Ω × (L → B × Salt) | msgs wt u.1 (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ}
      = #{u : Ω × (L → B × Salt) | msgs wt₀ u.1 (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ})
    (eR : G.Rand → Ω × (L → Salt)) (kR : ℕ) (heR : ∀ p, Nat.card {r // eR r = p} = kR)
    (hR : ∀ r, G.realVec H x w ν r c =
      msgs wt (eR r).1 fun l => Hl (dig wt (eR r).1 l + My ((eR r).2 l), cs ((eR r).2 l)))
    (hD : ∀ r, G.dVec H x ν r c =
      msgs wt₀ (eR r).1 fun l => Hl (dig wt₀ (eR r).1 l + My ((eR r).2 l), cs ((eR r).2 l)))
    (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun r : G.Rand => E (G.dVec H x ν r c))|
      ≤ 2 * Fintype.card L * δ₁.toReal := by
  classical
  have : Nonempty Salt := ⟨0⟩
  have one : ∀ (m : Ω × (L → B × Salt) → List Msg) (σ : Fin 1 → List Msg),
      #{u | (fun _ : Fin 1 => m u) = σ} = #{u | m u = σ 0} := by
    intro m σ
    congr 1
    ext u
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨fun h => congrFun h 0, fun h => funext fun i => by rw [Subsingleton.elim i 0]; exact h⟩
  have hW₁ : ∀ wa wb : Wt, (∀ σ, #{u : Ω × (L → B × Salt) | msgs wa u.1 (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ}
      = #{u : Ω × (L → B × Salt) | msgs wb u.1 (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ}) →
      ∀ (_ : Fin 1 → Unit) (σ : Fin 1 → List Msg),
      #{u : Ω × (L → B × Salt) | (fun _ : Fin 1 => msgs wa u.1 (fun l => Hl ((u.2 l).1, cs (u.2 l).2))) = σ}
        = #{u : Ω × (L → B × Salt) | (fun _ : Fin 1 => msgs wb u.1 (fun l => Hl ((u.2 l).1, cs (u.2 l).2))) = σ} :=
    fun wa wb h _ σ => by rw [one, one]; exact h (σ 0)
  have h1 := prefinal_of_ideal (n := 1) (Cn := Unit) My cs hT1 Hl (fun wt' ω _ ls _ => msgs wt' ω ls)
    (fun wt' ω _ l => dig wt' ω l)
    (fun wt' => oneRound fun p => msgs wt' p.1 fun l => Hl (dig wt' p.1 l + (p.2 l).1, (p.2 l).2))
    (fun _ _ _ _ => rfl) wt wt₀ (hW₁ wt wt₀ hW₀) (unitResp _) (fun τ => E (τ 0))
  have h2 := prefinal_of_ideal (n := 1) (Cn := Unit) My cs hT1 Hl (fun wt' ω _ ls _ => msgs wt' ω ls)
    (fun wt' ω _ l => dig wt' ω l)
    (fun wt' => oneRound fun p => msgs wt' p.1 fun l => Hl (dig wt' p.1 l + (p.2 l).1, (p.2 l).2))
    (fun _ _ _ _ => rfl) wt₀ wt (hW₁ wt₀ wt fun σ => (hW₀ σ).symm) (unitResp _) (fun τ => E (τ 0))
  simp only [run_oneRound] at h1 h2
  have eW : Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L → Salt) =>
        E (msgs wt p.1 fun l => Hl (dig wt p.1 l + My (p.2 l), cs (p.2 l)))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    simp only [hR]
  have eD : Pr (fun r : G.Rand => E (G.dVec H x ν r c)) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L → Salt) =>
        E (msgs wt₀ p.1 fun l => Hl (dig wt₀ p.1 l + My (p.2 l), cs (p.2 l)))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    simp only [hD]
  have hε : 2 * (Fintype.card L : ENNReal) * δ₁ ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top (by simp) (ENNReal.natCast_ne_top _)) hδ₁
  have hεr : (2 * (Fintype.card L : ENNReal) * δ₁).toReal = 2 * Fintype.card L * δ₁.toReal := by
    simp [ENNReal.toReal_mul]
  have a := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h1
  have b := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h2
  rw [hεr] at a b
  rw [eW, eD, abs_le]
  constructor <;> linarith

/-- **Lemma C's bound on the runs before the last coin, at a prover whose leaves `L₀` are read before its messages**
(`prefinal_sw_of_ideal`): the prover's messages before the last coin on `w` are `S`'s, which reads `L₀`'s salts, and on
the dummy `P`'s on real hm96 leaves; at fixed answers and ideal `L₁` leaves, `S`'s messages take every value as often as
`P`'s on `wt` with real `L₀` leaves, and `P`'s at ideal leaves as often on both table witnesses. -/
theorem runs_close_sw_of_ideal {W : Type} (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (w : W)
    {Salt B Cc D Wt Ω L0 L1 : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt] [DecidableEq Salt]
    [Fintype B] [DecidableEq B] [Fintype Ω] [Nonempty Ω] [Fintype L0] [DecidableEq L0] [Fintype L1] [DecidableEq L1]
    (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Hl : B × Cc → D) (msgs : Wt → Ω → (Fin (G.K x + 1) → Out) → (L0 ⊕ L1 → D) → Fin (G.K x + 1) → Msg)
    (dig : Wt → Ω → (Fin (G.K x + 1) → Out) → L0 ⊕ L1 → B)
    (P : Wt → FlockSoundness.ZK.Strategy (Ω × (L0 ⊕ L1 → B × Cc)) Msg Out (G.K x + 1))
    (hP : ∀ wt ω p a, (P wt).msg (ω, p) a = msgs wt ω a fun l => Hl (dig wt ω a l + (p l).1, (p l).2))
    (msgsS : Ω × (L0 → Salt) → (Fin (G.K x + 1) → Out) → (L1 → D) → Fin (G.K x + 1) → Msg)
    (digS : Ω × (L0 → Salt) → (Fin (G.K x + 1) → Out) → L1 → B)
    (S : FlockSoundness.ZK.Strategy ((Ω × (L0 → Salt)) × (L1 → B × Cc)) Msg Out (G.K x + 1))
    (hS : ∀ q p a, S.msg (q, p) a = msgsS q a fun l => Hl (digS q a l + (p l).1, (p l).2)) (wt wt₀ : Wt)
    (hsw : ∀ a σ, #{u : (Ω × (L0 → Salt)) × (L1 → B × Salt) | msgsS u.1 a (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ}
      = #{u : (Ω × (L0 → Salt)) × (L1 → B × Salt) | msgs wt u.1.1 a (Sum.elim
          (fun l => Hl (dig wt u.1.1 a (.inl l) + My (u.1.2 l), cs (u.1.2 l))) fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ})
    (hW₀ : ∀ a σ, #{u : Ω × (L0 ⊕ L1 → B × Salt) | msgs wt u.1 a (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ}
      = #{u : Ω × (L0 ⊕ L1 → B × Salt) | msgs wt₀ u.1 a (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ})
    (eR : G.Rand → Ω × (L0 ⊕ L1 → Salt)) (kR : ℕ) (heR : ∀ p, Nat.card {r // eR r = p} = kR)
    (hlink : ∀ r, (G.preStrat H x ν (G.msg x w ν)).msg r =
      S.msg (((eR r).1, fun l => (eR r).2 (.inl l)), fun l => (My ((eR r).2 (.inr l)), cs ((eR r).2 (.inr l)))))
    (hlink₀ : ∀ r, (G.preStrat H x ν (G.dmsg x ν)).msg r =
      (P wt₀).msg ((eR r).1, fun l => (My ((eR r).2 l), cs ((eR r).2 l))))
    (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1)) (E : (Fin (G.K x + 1) → Msg) → Prop) :
    |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
      Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))|
      ≤ 2 * Fintype.card (L0 ⊕ L1) * δ₁.toReal := by
  have : Nonempty Salt := ⟨0⟩
  have eW : Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L0 ⊕ L1 → Salt) => E (FlockSoundness.ZK.run S V
        ((p.1, fun l => p.2 (.inl l)), fun l => (My (p.2 (.inr l)), cs (p.2 (.inr l)))))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    congr 1
    funext r
    rw [run_congr _ _ V r _ (hlink r)]
  have eD : Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r)) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L0 ⊕ L1 → Salt) =>
        E (FlockSoundness.ZK.run (P wt₀) V (p.1, fun l => (My (p.2 l), cs (p.2 l))))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    congr 1
    funext r
    rw [run_congr _ _ V r _ (hlink₀ r)]
  obtain ⟨h1, h2⟩ := prefinal_sw_of_ideal My cs hT1 Hl msgs dig P hP msgsS digS S hS wt wt₀ hsw hW₀ V E
  have hε : 2 * (Fintype.card (L0 ⊕ L1) : ENNReal) * δ₁ ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top (by simp) (ENNReal.natCast_ne_top _)) hδ₁
  have hεr : (2 * (Fintype.card (L0 ⊕ L1) : ENNReal) * δ₁).toReal = 2 * Fintype.card (L0 ⊕ L1) * δ₁.toReal := by
    simp [ENNReal.toReal_mul, -Fintype.card_sum]
  have a := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h1
  have b := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h2
  rw [hεr] at a b
  rw [eW, eD, abs_le]
  constructor <;> linarith

/-- **Lemma C at a coin vector where the prover refuses, at a prover whose leaves `L₀` are read before its messages**
(`prefinal_sw_of_ideal` in one round): the prover's messages at `c` are `msgsS` of `L₀`'s salts and real `L₁` leaves, the
dummy-witness prover's `msgs` of real hm96 leaves; at ideal `L₁` leaves `msgsS` takes every value as often as `msgs` on
`wt` with real `L₀` leaves, and `msgs` at ideal leaves as often on both table witnesses. -/
theorem refused_close_sw_of_ideal {W : Type} (G : Proto W) (H : Msg → Msg) (x : Msg) (w : W) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) {Salt B Cc D Wt Ω L0 L1 : Type} [AddCommGroup Salt] [AddCommGroup B] [Fintype Salt]
    [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Ω] [Nonempty Ω] [Fintype L0] [DecidableEq L0]
    [Fintype L1] [DecidableEq L1] (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (Hl : B × Cc → D) (msgs : Wt → Ω → (L0 ⊕ L1 → D) → List Msg) (dig : Wt → Ω → L0 ⊕ L1 → B)
    (msgsS : Ω × (L0 → Salt) → (L1 → D) → List Msg) (digS : Ω × (L0 → Salt) → L1 → B) (wt wt₀ : Wt)
    (hsw : ∀ σ, #{u : (Ω × (L0 → Salt)) × (L1 → B × Salt) | msgsS u.1 (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ}
      = #{u : (Ω × (L0 → Salt)) × (L1 → B × Salt) | msgs wt u.1.1 (Sum.elim
          (fun l => Hl (dig wt u.1.1 (.inl l) + My (u.1.2 l), cs (u.1.2 l))) fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ})
    (hW₀ : ∀ σ, #{u : Ω × (L0 ⊕ L1 → B × Salt) | msgs wt u.1 (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ}
      = #{u : Ω × (L0 ⊕ L1 → B × Salt) | msgs wt₀ u.1 (fun l => Hl ((u.2 l).1, cs (u.2 l).2)) = σ})
    (eR : G.Rand → Ω × (L0 ⊕ L1 → Salt)) (kR : ℕ) (heR : ∀ p, Nat.card {r // eR r = p} = kR)
    (hR : ∀ r, G.realVec H x w ν r c = msgsS ((eR r).1, fun l => (eR r).2 (.inl l)) fun l =>
      Hl (digS ((eR r).1, fun l => (eR r).2 (.inl l)) l + My ((eR r).2 (.inr l)), cs ((eR r).2 (.inr l))))
    (hD : ∀ r, G.dVec H x ν r c =
      msgs wt₀ (eR r).1 fun l => Hl (dig wt₀ (eR r).1 l + My ((eR r).2 l), cs ((eR r).2 l)))
    (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun r : G.Rand => E (G.dVec H x ν r c))|
      ≤ 2 * Fintype.card (L0 ⊕ L1) * δ₁.toReal := by
  classical
  have : Nonempty Salt := ⟨0⟩
  have one : ∀ {X : Type} [Fintype X] (m : X → List Msg) (σ : Fin 1 → List Msg),
      #{u | (fun _ : Fin 1 => m u) = σ} = #{u | m u = σ 0} := by
    intro X _ m σ
    congr 1
    ext u
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨fun h => congrFun h 0, fun h => funext fun i => by rw [Subsingleton.elim i 0]; exact h⟩
  obtain ⟨h1, h2⟩ := prefinal_sw_of_ideal (n := 1) (Cn := Unit) My cs hT1 Hl (fun wt' ω _ ls _ => msgs wt' ω ls)
    (fun wt' ω _ l => dig wt' ω l)
    (fun wt' => oneRound fun p => msgs wt' p.1 fun l => Hl (dig wt' p.1 l + (p.2 l).1, (p.2 l).2))
    (fun _ _ _ _ => rfl) (fun q _ ls _ => msgsS q ls) (fun q _ l => digS q l)
    (oneRound fun p => msgsS p.1 fun l => Hl (digS p.1 l + (p.2 l).1, (p.2 l).2)) (fun _ _ _ => rfl) wt wt₀
    (fun _ σ => by rw [one, one]; exact hsw (σ 0)) (fun _ σ => by rw [one, one]; exact hW₀ (σ 0)) (unitResp _)
    (fun τ => E (τ 0))
  simp only [run_oneRound] at h1 h2
  have eW : Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L0 ⊕ L1 → Salt) => E (msgsS (p.1, fun l => p.2 (.inl l)) fun l =>
        Hl (digS (p.1, fun l => p.2 (.inl l)) l + My (p.2 (.inr l)), cs (p.2 (.inr l))))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    simp only [hR]
  have eD : Pr (fun r : G.Rand => E (G.dVec H x ν r c)) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L0 ⊕ L1 → Salt) =>
        E (msgs wt₀ p.1 fun l => Hl (dig wt₀ p.1 l + My (p.2 l), cs (p.2 l)))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    simp only [hD]
  have hε : 2 * (Fintype.card (L0 ⊕ L1) : ENNReal) * δ₁ ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top (by simp) (ENNReal.natCast_ne_top _)) hδ₁
  have hεr : (2 * (Fintype.card (L0 ⊕ L1) : ENNReal) * δ₁).toReal = 2 * Fintype.card (L0 ⊕ L1) * δ₁.toReal := by
    simp [ENNReal.toReal_mul, -Fintype.card_sum]
  have a := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h1
  have b := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h2
  rw [hεr] at a b
  rw [eW, eD, abs_le]
  constructor <;> linarith

end Proto

end Claims.ZKMal
