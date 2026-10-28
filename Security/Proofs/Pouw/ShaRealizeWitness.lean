import Proofs.Pouw.ShaRealize

/-!
# An honest prover meets `ShaRealize`'s realization (block I)

The realization premise isn't empty on the provers it is for. On a toy accelerator model whose device premises hold
(`toyG_premises`), the honest prover of a no-fold round on `k` rounds registered at one instance `a` (`honest`), over
any digest set `D` (SHA-512's `2^512` values among them), is accepted on every oracle and every coin
(`honest_accepted`) and has a `Realization` at every cap `Q`, query budget 1, schedule 1, advice of 0 bits and
walk-and-log cost 0 (`honest_realization`). So `extract_of_realization` applies to it, at `ξ = 0` and
`κ_x = (1 + 3k)/|D|` (`honest_extractAdvAt`).

* `toyG`: programs are functions with a cost and size 0, so (G1) holds at memory 0. There is one resource, a run
  uses its program's cost of it and makes as many oracle calls, each using one unit (G3), and nothing is imported.
* `toyNF D`: one word (`n = n' = nb = L = s = 1`), no noise, the oracle a table `Word ⊕ (D × D) → D` (leaves and
  nodes), the root the leaf's hash and the opening its check.
* `honest a k`: every round computes the registered instance's word (cost 1) and hashes it (cost 1), and its answer
  opens it at no cost. Its words program is the walk over its one logged query, so the walk costs nothing beyond its
  work.

It doesn't show `ShaRealize`, a realization for every prover: `toyG`'s programs read the oracle table as data, so
`Accel.OracleCalls` fails on it, and `ShaRealize` assumes it.
-/

namespace Pouw.SecurityProofs.ShaRealize.Witness

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.MerkleExtract

/-- Programs are functions with a cost, of size 0. -/
abbrev toyMc : AlgModel where
  Alg I O := (I → O) × ℕ
  eval f := f.1
  time f := f.2
  cost f _ := f.2
  size _ := 0
  seq f g := (g.1 ∘ f.1, f.2 + g.2)
  pair f g := (fun x => (f.1 x, g.1 x), f.2 + g.2)
  fst := (Prod.fst, 0)
  snd := (Prod.snd, 0)
  eval_seq _ _ _ := rfl
  eval_pair _ _ _ := rfl
  eval_fst _ := rfl
  eval_snd _ := rfl
  time_nonneg _ := Nat.cast_nonneg _
  cost_le_time _ _ := le_rfl
  size_le_time _ := Nat.cast_nonneg _
  size_nonneg _ := le_rfl
  time_seq_le _ _ := by push_cast; rfl
  le_time_seq _ g := by push_cast; linarith [Nat.cast_nonneg (α := ℝ) g.2]
  time_pair_le _ _ := by push_cast; rfl
  le_time_pair_left _ g := by push_cast; linarith [Nat.cast_nonneg (α := ℝ) g.2]
  le_time_pair_right f _ := by push_cast; linarith [Nat.cast_nonneg (α := ℝ) f.2]
  time_fst := by simp
  time_snd := by simp

/-- One resource, used at a program's cost, as many oracle calls as its cost (a call uses one unit), nothing held
or imported, and memory 0. -/
abbrev toyG : Accel where
  Mc := toyMc
  Res := Unit
  rate _ := 1
  mem := 0
  use f _ _ := (Prod.snd f : ℕ)
  held _ _ := 0
  wall f _ := (Prod.snd f : ℕ)
  ext _ _ := 0
  calls f _ := (Prod.snd f : ℕ)
  callUse _ := 1
  P1 := True
  P2 := True

/-- `toyG` meets the device premises. -/
theorem toyG_premises : toyG.G1 ∧ toyG.G2 ∧ toyG.G3 ∧ toyG.P1 ∧ toyG.P2 := by
  refine ⟨fun _ _ => ?_, ⟨fun _ => one_pos, fun f _ => ⟨le_rfl, fun _ => ⟨Nat.cast_nonneg _, ?_⟩⟩⟩,
    fun f _ => ⟨Nat.cast_nonneg _, fun _ => ?_⟩, trivial, trivial⟩
  · show (0 : ℝ) + 0 ≤ 0
    norm_num
  · show ((Prod.snd f : ℕ) : ℝ) ≤ 1 * ((Prod.snd f : ℕ) : ℝ)
    rw [one_mul]
  · show ((Prod.snd f : ℕ) : ℝ) * 1 ≤ ((Prod.snd f : ℕ) : ℝ)
    rw [mul_one]

variable (D : Type) [Fintype D] [DecidableEq D] [Inhabited D]

/-- The points the oracle answers: leaves and nodes. -/
abbrev toyX : Type := Word ⊕ (D × D)

/-- The tree encodings: a leaf is `inl`, a node `inr`. -/
def toyE : Enc (toyX D) D Word where
  leaf := Sum.inl
  node l r := Sum.inr (l, r)
  leaf_inj := Sum.inl_injective
  node_inj h := Prod.mk.inj (Sum.inr.inj h)
  disj _ _ _ := Sum.inl_ne_inr

/-- One word, no noise, its leaf's hash the root, and the opening the leaf's check. -/
abbrev toyNF : NoFold where
  n := 1
  n' := 1
  nb := 1
  L := 1
  Draw := Unit
  noise ab _ := ab
  s := 1
  Chk := Unit
  cells _ _ := (0, 0, 0)
  Orc := toyX D → D
  Root := D
  merkle o T := o ((toyE D).leaf (T 0 0 0))
  Path := Unit
  opens o r _ v _ := o ((toyE D).leaf v) = r

variable {D} (a : (noFoldRound (toyNF D)).Inst) (k : ℕ)

/-- The honest word at the instance `a`. -/
def word : Word := noFoldWord (toyNF D) a 0 0 0

/-- **The honest prover** on rounds registered at `a`: computes the word and hashes it (cost 2), and opens it. -/
def honest : Prover toyG (noFoldRound (toyNF D)) k where
  St := Unit
  start := ()
  work _ := (fun p => ((), (toyNF D).merkle p.1 fun _ _ _ => word a), 2)
  absorb _ _ _ := ()
  answer _ := (fun _ _ => (word a, ()), 0)

omit [Inhabited D] in
/-- The honest prover is accepted on every oracle and every coin. -/
theorem honest_accepted (o : (noFoldRound (toyNF D)).Orc) (x : Fin k → (toyNF D).Draw × (toyNF D).Chk) :
    (honest a k).Accepted (fun _ => a) o x :=
  fun _ _ => ⟨rfl, rfl⟩

omit [Inhabited D] in
/-- The verifier's check is the Merkle path check, along the empty position. -/
theorem toy_hopens : ∀ o r c v p, (toyNF D).opens o r c v p →
    Opens ((Equiv.refl _ : (toyNF D).Orc ≃ (toyX D → D)) o) (toyE D) (id r) ((fun _ => []) p) v ∧
      ((fun _ : (toyNF D).Path => ([] : List (Bool × D × D))) p).map Prod.fst =
        (fun _ : Fin (toyNF D).n × Fin (toyNF D).n' × Fin (toyNF D).nb => ([] : List Bool)) c :=
  fun _ _ _ _ _ h => ⟨h, rfl⟩

/-- The honest prover's log: its one query, the leaf, before every round's root. -/
def log : QueryLog (noFoldRound (toyNF D)) k (toyX D) D 1 fun _ => 1 where
  qry _ _ _ := (toyE D).leaf (word a)
  qry_dep _ _ _ _ _ := rfl
  fresh _ _ i j h := absurd h (by have := i.isLt; have := j.isLt; rw [Fin.lt_def]; omega)
  root _ _ u := u 0
  root_dep _ _ _ _ h := h 0 (by decide)

/-- The walk over the honest log finds the honest word. -/
theorem walk_log (x : Fin k → (toyNF D).Draw × (toyNF D).Chk) (w : Fin k) (u : Fin 1 → D) :
    Pouw.MerkleExtract.ext (((log a k).prover x w).oracle u) (toyE D) (((log a k).prover x w).pre u) (u 0) [] =
      some (word a) := by
  have hor : ((log a k).prover x w).oracle u ((toyE D).leaf (word a)) = u 0 := by
    unfold MerkleExtract.Prover.oracle
    rw [dite_eq_left ⟨0, rfl⟩]
    exact congrArg u (Subsingleton.elim _ _)
  have hw : ∃ v, (toyE D).leaf v ∈ ((log a k).prover x w).pre u ∧
      ((log a k).prover x w).oracle u ((toyE D).leaf v) = u 0 :=
    ⟨word a, ⟨0, by decide, rfl⟩, hor⟩
  rw [Pouw.MerkleExtract.ext, dite_eq_left hw]
  obtain ⟨⟨_, _, hi⟩, _⟩ := Classical.choose_spec hw
  exact congrArg some ((toyE D).leaf_inj hi).symm

/-- **The honest prover has a realization** at every cap `Q`, query budget 1, schedule 1, advice of 0 bits, and
walk-and-log cost 0. -/
theorem honest_realization (Q : ℝ) :
    Realization toyG (toyNF D) k (fun _ => a) Q (toyE D) (Equiv.refl _) id (fun _ => []) (fun _ => []) 0 0
      (honest a k) 1 (fun _ => 1) 0 := by
  refine ⟨log a k, Unit, fun _ _ _ => (), fun _ => (fun _ _ _ _ => word a, 1),
    fun _ => (fun p => (toyNF D).merkle p.1 p.2.2, 1), ⟨fun _ => 0, fun _ _ _ => rfl⟩, fun _ _ _ _ _ => rfl,
    fun _ _ _ => rfl, ?_, ?_, fun _ _ _ _ hu _ => (hu 0).symm,
    fun _ _ _ _ _ _ _ _ _ hy => ⟨0, (List.mem_singleton.mp hy).symm⟩, ?_⟩
  · intro _ _ _ _
    show ∑ _ : Fin k, (((1 : ℕ) : ℝ) + ((1 : ℕ) : ℝ)) ≤
      ∑ _ : Fin k, ((2 : ℕ) : ℝ) + ((k : ℝ) * 0 + ((1 : ℕ) : ℝ) * 0)
    norm_num
  · intro _ _ _
    show (0 : ℝ) + 0 ≤ 0
    norm_num
  · intro _ x _ u _ _ w _
    exact (congrArg (fun t => Option.getD t 0) (walk_log a k x w u)).symm

/-- **`extract_of_realization` applies to the honest prover** at every cap `Q`, advice of 0 bits, `ξ = 0` and
`κ_x = (1 + 3k)/|D|`. -/
theorem honest_extractAdvAt (Q : ℝ) :
    ExtractAdvAt toyG (noFoldRound (toyNF D)) k (fun _ => a) Q 0 (fun _ => 0)
      (((1 + 3 * k : ℕ) : ℝ) / Fintype.card D) (honest a k) := by
  have h := extract_of_realization (toyE D) (Equiv.refl _) id (fun _ => []) (fun _ => []) toy_hopens (honest a k)
    (honest_realization a k Q)
  have hκ : 1 * 1 + ∑ _ : Fin k, 1 * (2 * 1 + 1) = 1 + 3 * k := by simp [mul_comm]
  convert h using 2
  · simp
  · rw [hκ]

/-- At SHA-512's digests, `κ_x = (1 + 3k)/2^512`. -/
theorem honest_extractAdvAt_sha512 (a : (noFoldRound (toyNF (Fin (2 ^ 512)))).Inst) (Q : ℝ) :
    ExtractAdvAt toyG (noFoldRound (toyNF (Fin (2 ^ 512)))) k (fun _ => a) Q 0 (fun _ => 0)
      (((1 + 3 * k : ℕ) : ℝ) / 2 ^ 512) (honest a k) := by
  have h := honest_extractAdvAt a k Q
  rwa [Fintype.card_fin, Nat.cast_pow, Nat.cast_ofNat] at h

end Pouw.SecurityProofs.ShaRealize.Witness
