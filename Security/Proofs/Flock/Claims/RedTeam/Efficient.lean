import Proofs.Flock.Claims.ERepCost

/-!
# Red team: `Efficient` holds at a step count of `0`

`DecoderComplexity decode counted size p` reads the step count and the size from its own arguments, so `counted` that
reports `0` steps meets it at `p = 0` for every decoder, and Flock's `Efficient` is then `ListWithin` alone
(`ERepCost` holds at every record): it says nothing about the decoder's cost. A decoder that lists every trace meets
`ListWithin` wherever there are finitely many (`listWithin_univ`), so at such a record `Efficient` holds at `p = 0` for
the decoder that tries everything (`efficient_univ`).

* `decoderComplexity_zero`: every decoder, at `counted = (decode, 0)` and `p = 0`.
* `efficient_zero`: Flock's `Efficient` at every decoder that meets `ListWithin`, at `p = 0`; at the deployed decoders,
  `RedTeam.Deployed`.
* `listWithin_univ`, `not_listWithin_nil`: `ListWithin` is met (by the decoder that lists every trace, at finitely many;
  at the deployed codes, `GS.listWithin_deployPlain`, `GS.listWithin_deployZk`) and constrains the decoder (the one that
  lists nothing fails it).
* `decoderComplexity_listLength`: a list-length bound `(decode os).length ≤ p.eval (size os)` at the same free `size`
  and `p` adds nothing: both hold at `size = (decode ·).length`, `p = X`, for every decoder.
-/

namespace Claims.RedTeam

/-- **The decoder's complexity at a step count of `0`**, at `p = 0` and every size. -/
theorem decoderComplexity_zero {I T : Type} (decode : I → T) (size : I → ℕ) :
    DecoderComplexity decode (fun i => (decode i, 0)) size 0 :=
  fun _ => ⟨rfl, Nat.zero_le _⟩

/-- **A list-length bound beside `DecoderComplexity`, at the same `size` and `p`, holds for every decoder** at
`size = (decode ·).length` and `p = X`: the form of the per-table batch's `Efficient`
(`∀ os, (decode os).length ≤ p.eval (size os)`), which reads `size` from its own arguments too. -/
theorem decoderComplexity_listLength {I T : Type} (decode : I → List T) :
    DecoderComplexity decode (fun i => (decode i, 0)) (fun i => (decode i).length) Polynomial.X ∧
      ∀ i, (decode i).length ≤ Polynomial.X.eval ((decode i).length) :=
  ⟨fun _ => ⟨rfl, Nat.zero_le _⟩, fun _ => by simp⟩

/-- **Flock's `Efficient` at `p = 0`**, at every decoder that meets the list-decoding spec. -/
theorem efficient_zero {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (hl : Flock.ListWithin D.len D.enc D.θ decode) :
    Flock.Efficient H D decode (fun os => (decode os, 0)) (fun _ => 0) 0 :=
  Flock.Efficient_of_decoder H D decode _ _ 0 hl (decoderComplexity_zero decode _)

/-- The decoder that lists every trace. -/
noncomputable def decodeUniv {Col T : Type} [Fintype T] : List (ℕ × Col) → List T := fun _ => Finset.univ.toList

theorem listWithin_univ {Col T : Type} [DecidableEq Col] [Zero Col] [Fintype T] (len : T → ℕ) (enc : T → ℕ → Col)
    (θ : T → ℝ) : Flock.ListWithin len enc θ (decodeUniv (Col := Col)) :=
  fun _ M _ => Finset.mem_toList.2 (Finset.mem_univ M)

theorem fill_enc {Col T : Type} [Zero Col] (enc : T → ℕ → Col) (M : T) (n : ℕ) (i : Fin n) :
    Flock.fill n ((List.range n).map fun j => (j, enc M j)) i = enc M i.val := by
  have hsome : ((List.range n).find? fun j => j == i.val).isSome :=
    List.find?_isSome.2 ⟨i.val, List.mem_range.2 i.isLt, beq_self_eq_true _⟩
  obtain ⟨j, hj⟩ := Option.isSome_iff_exists.1 hsome
  have hji : j = i.val := by simpa using List.find?_some hj
  subst hji
  simp [Flock.fill, List.find?_map, hj]

/-- **`ListWithin` constrains the decoder**: the decoder that lists nothing fails it at every code with a trace of
nonnegative radius, whose own encoding, opened everywhere, agrees with it everywhere. -/
theorem not_listWithin_nil {Col T : Type} [DecidableEq Col] [Zero Col] (len : T → ℕ) (enc : T → ℕ → Col)
    (θ : T → ℝ) (M : T) (hθ : 0 ≤ θ M) : ¬ Flock.ListWithin len enc θ fun _ => [] := by
  intro h
  refine absurd (h ((List.range (len M)).map fun j => (j, enc M j)) M ?_) List.not_mem_nil
  rw [Finset.filter_true_of_mem fun i _ => (fill_enc enc M (len M) i).symm, Finset.card_univ, Fintype.card_fin]
  have : (0 : ℝ) ≤ len M := Nat.cast_nonneg _
  nlinarith

/-- **Flock's `Efficient` at `p = 0` for the decoder that tries every trace**, at a record with finitely many. -/
theorem efficient_univ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] [Fintype T] (H : Msg → Msg) (D : Flock.Deploy Coin Col T) :
    Flock.Efficient H D decodeUniv (fun os => (decodeUniv os, 0)) (fun _ => 0) 0 :=
  efficient_zero H D _ (listWithin_univ D.len D.enc D.θ)

end Claims.RedTeam
