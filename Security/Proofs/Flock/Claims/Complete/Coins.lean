import Proofs.Flock.Claims.Complete.Run
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Level3.ArithFacts

/-!
# δc: where the honest prover can fail

Plain Flock is perfectly complete but for one step. Zerocheck's round `i` recovers `g(0) = (claim + r·g(1))·(1 + r)⁻¹`
at the outer coin `r = r_i` (`bindAndZerocheckWith`), and `F128.inv 0 = 0`: at `r = 1` the verifier's `g(0)` is `0`
whatever the honest prover's is, and the next claim need not hold. So the honest prover can fail only where some
zerocheck outer coin is `1` (`Bad`), at most `2^-128` per coin element (`pr_slot_le`).

* **The outer coins** (`outerSlots`): rep `q`'s second round (`rOuter`, after `r_skip`), elements `0` to `m − 14`, at
  its index among the verifier's coins (`repStart`: the link's coin, then the earlier reps' recorded rounds,
  `spec.repStreams` order).
* **`ε`** is their count over `2^128`; `pr_bad_le`: `Pr[Bad] ≤ ε`, a union bound.
* **`δc`** is `ε` on the statements the deployed verifier can accept against `H` (`ready H`), and `1` elsewhere: a
  statement that sets up no session, or whose session draws no unit (`checks` ends in `drawLaw zk none`, which refuses),
  is refused at every run, and the last message's length words (`Final.encode`) fit only a statement under `2^56` bytes:
  the record carries the statement's `hello`, which JSON escaping can grow sixfold, past `2^64` bytes for a statement
  near it. Where a table's regions do not fit (`RegionsFit`: some region's bytes at some block are not its free bits),
  the extra claims refuse (`Stmt.regionValue`), the honest prover's as the verifier's, and the relation, on the
  circuit's values, does not pin the public bytes the regions read. The verifier sets the statement up against `H`
  (`setupWith (hOf H)`), and the statement pins digests of SHA-512 that this setup recomputes with `H`; the honest
  prover sets up at SHA-512, so `ready H` asks that the two setups agree. At `sha512` they do, and `ready` asks no more
  (`ready_sha512`).
* **The composition** (`complete_of`): if the honest prover is accepted at every coin outside `Bad` on every statement in
  scope, ready, with a witness in the relation, then `Claims.Flock.Complete H (deployPlain rel) honestP (δc H)`.
-/

namespace FlockComplete

open Flock FlockDeploy

/-! ## The probability of the bad coins -/

theorem Pr_or_le {α : Type} [Fintype α] (p q : α → Prop) :
    Claims.Pr (fun a => p a ∨ q a) ≤ Claims.Pr p + Claims.Pr q := by
  classical
  unfold Claims.Pr
  rw [← add_div]
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  norm_cast
  refine le_trans (Finset.card_le_card ?_) (Finset.card_union_le _ _)
  intro a
  simp only [Finset.mem_filter, Finset.mem_union, Finset.mem_univ, true_and]
  exact id

/-- **The union bound** over a list of events. -/
theorem Pr_exists_le {α β : Type} [Fintype α] (E : β → α → Prop) :
    ∀ L : List β, Claims.Pr (fun a => ∃ b ∈ L, E b a) ≤ (L.map fun b => Claims.Pr (E b)).sum
  | [] => by simp [Claims.Pr_false]
  | b :: L => by
    refine le_trans (le_of_eq (Claims.Pr_congr (q := fun a => E b a ∨ ∃ b' ∈ L, E b' a) fun a => ?_))
      (le_trans (Pr_or_le (E b) _) ?_)
    · simp
    · simp only [List.map_cons, List.sum_cons]
      exact (add_le_add_iff_left _).2 (Pr_exists_le E L)

theorem Pr_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (p : β → Prop) :
    Claims.Pr (fun a => p (e a)) = Claims.Pr p := by
  classical
  rw [Claims.Pr_eq_sum (fun a => p (e a)), Claims.Pr_eq_sum p, Fintype.card_congr e]
  congr 1
  exact e.sum_comp (fun b => if p b then (1 : ℝ) else 0)

theorem Pr_eq_const {α : Type} [Fintype α] (c : α) : Claims.Pr (fun a : α => a = c) = 1 / Fintype.card α := by
  classical
  unfold Claims.Pr
  congr 1
  rw [Finset.filter_eq' Finset.univ c]
  simp

theorem Pr_not {α : Type} [Fintype α] [Nonempty α] (p : α → Prop) :
    Claims.Pr (fun a => ¬ p a) = 1 - Claims.Pr p := by
  classical
  rw [Claims.Pr_eq_sum (fun a => ¬ p a), Claims.Pr_eq_sum p]
  have hc : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  have e : ∀ a, (if ¬ p a then (1 : ℝ) else 0) = 1 - (if p a then 1 else 0) := by
    intro a; split_ifs <;> simp_all
  rw [Finset.sum_congr rfl fun a _ => e a, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
    mul_one]
  field_simp

/-- **A coin element is `1` with probability at most `2^-128`**, at any index (`0` past the coins or the elements). -/
theorem pr_slot_le {k : ℕ} (j i : ℕ) :
    Claims.Pr (fun r : Fin k → Coin => (Claims.coinSeq r j).at i = 1) ≤ 1 / 2 ^ 128 := by
  by_cases hj : j < k
  · by_cases hi : i < 65536
    · let e1 : (Fin k → Coin) ≃ Coin × ({j' // j' ≠ (⟨j, hj⟩ : Fin k)} → Coin) :=
        Equiv.piSplitAt (⟨j, hj⟩ : Fin k) fun _ => Coin
      let e2 : Coin ≃ FlockLevel3.GF128 × ({i' // i' ≠ (⟨i, hi⟩ : Fin 65536)} → FlockLevel3.GF128) :=
        Equiv.piSplitAt (⟨i, hi⟩ : Fin 65536) fun _ => FlockLevel3.GF128
      have : Nonempty FlockLevel3.GF128 := ⟨(0 : F128)⟩
      have hev : ∀ r : Fin k → Coin, (Claims.coinSeq r j).at i = 1 ↔
          (e1.trans (Equiv.prodCongr e2 (Equiv.refl _)) r).1.1 = (1 : F128) := by
        intro r
        simp only [Claims.coinSeq, hj, dite_true, Coin.at, hi]
        rfl
      rw [Claims.Pr_congr hev, Pr_equiv (e1.trans (Equiv.prodCongr e2 (Equiv.refl _)))
        (fun s => s.1.1 = (1 : F128)), Claims.Pr_fst (fun s : FlockLevel3.GF128 × _ => s.1 = (1 : F128)),
        Claims.Pr_fst (fun g : FlockLevel3.GF128 => g = (1 : F128))]
      exact le_of_eq ((Pr_eq_const (α := FlockLevel3.GF128) (1 : F128)).trans
        (by rw [FlockLevel3.card_F]; norm_num))
    · refine le_trans (le_of_eq (Claims.Pr_congr (q := fun _ => False) fun r => ?_)) ?_
      · simp only [Coin.at, hi, dite_false, iff_false]
        decide
      · rw [Claims.Pr_false]; positivity
  · refine le_trans (le_of_eq (Claims.Pr_congr (q := fun _ => False) fun r => ?_)) ?_
    · simp only [Claims.coinSeq, hj, dite_false, iff_false]
      show ¬ (default : Coin).at i = 1
      unfold Coin.at
      split
      · show ¬ (0 : F128) = 1
        decide
      · decide
    · rw [Claims.Pr_false]; positivity

/-! ## δc -/

/-- **Rep `q`'s first coin's index** (table `q / 2`'s rep `q % 2`, `spec.repStreams` order): the link's coin, then every
earlier rep's recorded rounds. -/
def repStart (spec : SessionSpec) (sts : Array Flock.Stmt) (q : ℕ) : ℕ :=
  1 + ((List.range q).map fun q' => repRounds false spec sts[q' / 2]!).sum

/-- **The zerocheck outer coins** of the session the statement sets up: rep `q`'s second round (`rOuter`, after
`r_skip`), elements `0` to `m − 14`, as (coin index, element) pairs. -/
def outerSlots (x : Claims.Msg) : List (ℕ × ℕ) :=
  match setupOf false x with
  | some (_, spec, sts) => (List.range (2 * sts.size)).flatMap fun q =>
      (List.range (sts[q / 2]!.m - 13)).map fun i => (repStart spec sts q + 1, i)
  | none => []

/-- **The bad coins**: a zerocheck outer coin is `1`, where the verifier's `(1 + r)⁻¹` is `0⁻¹ = 0`. -/
def Bad (x : Claims.Msg) {k : ℕ} (r : Fin k → Coin) : Prop := ∃ p ∈ outerSlots x, (Claims.coinSeq r p.1).at p.2 = 1

/-- `ε`: the outer coins' count over `2^128`. -/
noncomputable def ε (x : Claims.Msg) : ℝ := (outerSlots x).length / 2 ^ 128

/-- **A statement's regions fit**: each region's bytes at each block are its free bits. -/
def RegionsFit (st : Flock.Stmt) : Prop :=
  ∀ reg ∈ st.regions, ∀ b < 2 ^ st.nbl, (reg.bytes b).size * 8 = 2 ^ reg.free.size

open Classical in
/-- **The statements the deployed verifier can accept** against `H`: it sets up the session, its setup against `H` is
the one at SHA-512, the session draws a unit, the statement is under `2^56` bytes, so that the last message's lengths
fit their words, and each table's regions fit. -/
noncomputable def ready (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) : Bool :=
  match setupOf false x with
  | some (s, spec, sts) => s.read.draw.isSome && decide (x.length < 2 ^ 56) && decide (∀ st ∈ sts, RegionsFit st) &&
      decide (setupWith (Claims.hOf H) s.read = .ok (spec, sts))
  | none => false

open Classical in
/-- **At SHA-512, `ready` asks only for a drawn session, the statement's length and its regions**: the setup against
`sha512` is the statement's. -/
theorem ready_sha512 (x : Claims.Msg) :
    ready sha512 x = match setupOf false x with
      | some (s, _, sts) => s.read.draw.isSome && decide (x.length < 2 ^ 56) && decide (∀ st ∈ sts, RegionsFit st)
      | none => false := by
  unfold ready
  split
  · rename_i s spec sts hs
    rw [hOf_sha512, setupWith_sha512, Ins.setup_read (setupOf_some hs).2]
    simp
  · rfl

/-- **Flock's `δc` at the plain verifier** against `H`: `ε` where the verifier can accept, else `1`. -/
noncomputable def δc (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) : ℝ := if ready H x then ε x else 1

/-- **`Pr[Bad] ≤ ε`**: the union bound over the outer coins. -/
theorem pr_bad_le (x : Claims.Msg) {k : ℕ} : Claims.Pr (fun r : Fin k → Coin => Bad x r) ≤ ε x := by
  have h1 := Pr_exists_le (fun p (r : Fin k → Coin) => (Claims.coinSeq r p.1).at p.2 = 1) (outerSlots x)
  have h2 : ((outerSlots x).map fun p =>
      Claims.Pr (fun (r : Fin k → Coin) => (Claims.coinSeq r p.1).at p.2 = 1)).sum ≤ ε x := by
    refine le_trans (List.sum_le_length_nsmul _ (1 / 2 ^ 128) ?_) (le_of_eq ?_)
    · intro y hy
      obtain ⟨p, -, rfl⟩ := List.mem_map.1 hy
      exact pr_slot_le p.1 p.2
    · rw [List.length_map, nsmul_eq_mul, ε]
      ring
  exact le_trans h1 h2

/-! ## The composition -/

/-- **Completeness from acceptance at the good coins**: if, on every statement in scope that the verifier can accept
and every witness in the relation, the honest prover is accepted at every tape and every coin outside `Bad`, then Flock's
plain verifier is complete for `honestP` with error `δc`. -/
theorem complete_of (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hacc : ∀ x w, Claims.Flock.Scope (deployPlain rel) x → Claims.Flock.R (deployPlain rel).parse x w →
      ready H x = true → ∀ (ω : Tape) (r : Fin ((V false).k x) → Coin), ¬ Bad x r →
        (Claims.run H (honestP w) (V false) x ω r).out = .acc) :
    Claims.Flock.Complete H (deployPlain rel) honestP (δc H) := by
  intro _ x w hs hr ω
  unfold δc
  split
  · rename_i hready
    calc 1 - ε x ≤ 1 - Claims.Pr (fun r : Fin ((V false).k x) → Coin => Bad x r) := by
          linarith [pr_bad_le x (k := (V false).k x)]
      _ = Claims.Pr (fun r : Fin ((V false).k x) → Coin => ¬ Bad x r) := (Pr_not _).symm
      _ ≤ Claims.Pr (fun r : Fin ((V false).k x) → Coin =>
            (Claims.run H (honestP w) (V false) x ω r).out = .acc) :=
          Claims.Pr_mono fun r hr' => hacc x w hs hr hready ω r hr'
  · rw [sub_self]
    exact Claims.Pr_nonneg _

end FlockComplete
