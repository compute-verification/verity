import Proofs.Flock.Claims.KS.ConflictTied

/-!
# `TableBoundθ` statement by statement

`TableBoundθ` (`KS.Joint`) is a bound at each statement (`TableBoundθAt`). Where the verifier accepts no run and the
weight covers the fork event's `k`, it holds with no escapes (`tableBoundθAt_of_noAcc`, `Escapes.nil`). A statement the
deployed verifier does not set up is one: it has no tables, so `ε_table`, `N₀`, `k` and the weight are `0` there
(`εtable_none`, `k_none_plain`, `k_none_zk`, `weight_none`), and an accepted run sets up (`acc_verdict`, `noAcc_of_none`). So at both
records `TableBoundθ` is its bound at the statements that set up (`plain_tableBoundθ_of_setup`,
`zk_tableBoundθ_of_setup`), and KS at SHA-512 rests on that alone (`plain_KSθ_of_setup`, `zk_KSθ_of_setup`).
-/

namespace Claims.Flock

/-- **No escapes.** -/
def Escapes.nil {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) :
    Escapes H V P x where
  I := Empty
  c := Empty.elim
  j := Empty.elim
  ev := Empty.elim
  L := Empty
  β := Empty.elim
  ev₁ := Empty.elim
  c_nonneg i := i.elim
  β_nonneg l := l.elim
  tied i := i.elim
  tied₁ l := l.elim

theorem Escapes.term_nil {Ω Coin : Type} [Fintype Coin] [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin)
    (P : Prover Ω) (x : Msg) (ω : Ω) : (Escapes.nil H V P x).term ω = 0 := by
  unfold Escapes.term
  exact (congrArg₂ (· + ·) (Finset.sum_eq_zero fun i _ => Empty.elim i)
    (Finset.sum_eq_zero fun l _ => Empty.elim l)).trans (add_zero 0)

/-- **No escapes fit every weight that covers the fork event's.** -/
theorem Escapes.fitsFork_nil {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω)
    (x : Msg) (C c : ℝ) (h : c ≤ 2 * C) : (Escapes.nil H V P x).FitsFork C c := by
  refine ⟨1, fun r => ?_, ?_, ?_⟩
  · have h0 : (Finset.univ.filter fun i : (Escapes.nil H V P x).I => (Escapes.nil H V P x).j i = r).card = 0 :=
      Finset.card_eq_zero.2 (Finset.eq_empty_of_forall_notMem fun i => i.elim)
    rw [h0]
    split <;> omega
  · have h0 : (∑ i : (Escapes.nil H V P x).I, (Escapes.nil H V P x).c i) = 0 :=
      Finset.sum_eq_zero fun i => i.elim
    rw [h0, Nat.cast_one, one_mul, add_zero]
    exact h
  · have h0 : (∑ l : (Escapes.nil H V P x).L, (Escapes.nil H V P x).β l) = 0 :=
      Finset.sum_eq_zero fun l => l.elim
    rw [h0]
    exact Nat.cast_nonneg _

/-- **`TableBoundθ` at one statement `x`.** -/
def TableBoundθAt {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) (Ok : Msg → Col → Prop) (x : Msg) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x,
    e.FitsFork (D.weight x) (D.k x) ∧ ∀ (ω : Ω) (C : Fin (Rd.N x) → Col), (∀ p, Ok x (C p)) →
      ¬ GoodTable H D Rd x C →
      accPr H D.V P x ω ≤ D.εtable x + e.term ω +
        Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
          ∃ (p : Fin (Rd.N x)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen) p.val = some u ∧ u ≠ C p

theorem tableBoundθ_iff {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) (Ok : Msg → Col → Prop) :
    TableBoundθ H D Rd Ok ↔ ∀ x, TableBoundθAt H D Rd Ok x :=
  Iff.rfl

/-- **Where the verifier accepts no run**, `TableBoundθ` holds with no escapes, when the weight covers `k` and
`ε_table ≥ 0`. -/
theorem tableBoundθAt_of_noAcc {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) (Ok : Msg → Col → Prop) (x : Msg)
    (hno : NoAcc H D.V x) (hk : (D.k x : ℝ) ≤ 2 * D.weight x) (hε : 0 ≤ D.εtable x) :
    TableBoundθAt H D Rd Ok x := by
  intro Ω _ _ P
  refine ⟨Escapes.nil H D.V P x, Escapes.fitsFork_nil H D.V P x _ _ hk, fun ω C _ _ => ?_⟩
  have h0 : accPr H D.V P x ω = 0 := by
    unfold accPr
    rw [Pr_congr (q := fun _ => False) fun r => ⟨fun h => hno Ω P ω r h, False.elim⟩, Pr_false]
  rw [h0, Escapes.term_nil]
  have := Pr_nonneg fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
    ∃ (p : Fin (Rd.N x)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen) p.val = some u ∧ u ≠ C p
  linarith

end Claims.Flock

namespace FlockDeploy.KS

open Flock

/-- **A statement the deployed verifier does not set up accepts no run**, at SHA-512. -/
theorem noAcc_of_none {zk : Bool} {x : Claims.Msg} (h : setupOf zk x = none) :
    Claims.Flock.NoAcc sha512 (V zk) x := by
  by_contra hacc
  obtain ⟨s, spec, sts, -, -, hs, -⟩ := acc_verdict hacc
  rw [h] at hs
  cases hs

theorem tables_none {zk : Bool} {x : Claims.Msg} (h : setupOf zk x = none) : tables zk x = [] := by
  unfold tables
  rw [h]

theorem εtable_none {zk : Bool} {x : Claims.Msg} (h : setupOf zk x = none) : εtable zk x = 0 := by
  unfold εtable
  rw [tables_none h]
  rfl

theorem weight_none {zk : Bool} {x : Claims.Msg} (h : setupOf zk x = none) : weight zk x = 0 := by
  unfold weight
  rw [tables_none h]
  rfl

theorem N₀_none {zk : Bool} {x : Claims.Msg} (h : setupOf zk x = none) : N₀ zk x = 0 := by
  unfold N₀
  rw [tables_none h]
  rfl

theorem k_none_plain (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} (h : setupOf false x = none) :
    (deployPlain rel).k x = 0 := by
  simp only [Claims.Flock.Deploy.k, deployPlain, N₀_none h, mul_zero, zero_div, Nat.ceil_zero]

theorem k_none_zk (rel : Readable.Parsed → Readable.Relation) {x : Claims.Msg} (h : setupOf true x = none) :
    (deployZk rel).k x = 0 := by
  simp only [Claims.Flock.Deploy.k, deployZk, N₀_none h, mul_zero, zero_div, Nat.ceil_zero]

/-- **The plain record: `TableBoundθ` from its bound at the statements that set up.** -/
theorem plain_tableBoundθ_of_setup (rel : Readable.Parsed → Readable.Relation)
    (h : ∀ x s spec sts, setupOf false x = some (s, spec, sts) →
      Claims.Flock.TableBoundθAt sha512 (deployPlain rel) plainRadius (wide false) x) :
    Claims.Flock.TableBoundθ sha512 (deployPlain rel) plainRadius (wide false) := by
  refine (Claims.Flock.tableBoundθ_iff _ _ _ _).2 fun x => ?_
  cases hs : setupOf false x with
  | some v => exact h x v.1 v.2.1 v.2.2 hs
  | none =>
    refine Claims.Flock.tableBoundθAt_of_noAcc _ _ _ _ x (noAcc_of_none hs) ?_ (εtable_nonneg false x)
    show ((deployPlain rel).k x : ℝ) ≤ 2 * weight false x
    rw [k_none_plain rel hs, weight_none hs]
    norm_num

/-- **The masked (`--zk`) record: `TableBoundθ` from its bound at the statements that set up.** -/
theorem zk_tableBoundθ_of_setup (rel : Readable.Parsed → Readable.Relation)
    (h : ∀ x s spec sts, setupOf true x = some (s, spec, sts) →
      Claims.Flock.TableBoundθAt sha512 (deployZk rel) zkRadius (wide true) x) :
    Claims.Flock.TableBoundθ sha512 (deployZk rel) zkRadius (wide true) := by
  refine (Claims.Flock.tableBoundθ_iff _ _ _ _).2 fun x => ?_
  cases hs : setupOf true x with
  | some v => exact h x v.1 v.2.1 v.2.2 hs
  | none =>
    refine Claims.Flock.tableBoundθAt_of_noAcc _ _ _ _ x (noAcc_of_none hs) ?_ (εtable_nonneg true x)
    show ((deployZk rel).k x : ℝ) ≤ 2 * weight true x
    rw [k_none_zk rel hs, weight_none hs]
    norm_num

/-- **The deployed plain verifier: KS at the filled word**, at SHA-512, from a fixed table's bound at the statements
that set up. -/
theorem plain_KSθ_of_setup (rel : Readable.Parsed → Readable.Relation)
    (h : ∀ x s spec sts, setupOf false x = some (s, spec, sts) →
      Claims.Flock.TableBoundθAt sha512 (deployPlain rel) plainRadius (wide false) x) :
    Claims.Flock.KSθ sha512 (deployPlain rel) plainRadius :=
  plain_KSθ_of_table rel (plain_tableBoundθ_of_setup rel h)

/-- **The deployed masked (`--zk`) verifier: KS at the filled word**, at SHA-512, from a fixed table's bound at the
statements that set up. -/
theorem zk_KSθ_of_setup (rel : Readable.Parsed → Readable.Relation)
    (h : ∀ x s spec sts, setupOf true x = some (s, spec, sts) →
      Claims.Flock.TableBoundθAt sha512 (deployZk rel) zkRadius (wide true) x) :
    Claims.Flock.KSθ sha512 (deployZk rel) zkRadius :=
  zk_KSθ_of_table rel (zk_tableBoundθ_of_setup rel h)

end FlockDeploy.KS
