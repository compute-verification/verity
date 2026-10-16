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
-/

namespace Claims.RedTeam

/-- **The decoder's complexity at a step count of `0`**, at `p = 0` and every size. -/
theorem decoderComplexity_zero {I T : Type} (decode : I → T) (size : I → ℕ) :
    DecoderComplexity decode (fun i => (decode i, 0)) size 0 :=
  fun _ => ⟨rfl, Nat.zero_le _⟩

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

/-- **Flock's `Efficient` at `p = 0` for the decoder that tries every trace**, at a record with finitely many. -/
theorem efficient_univ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Zero Col] [Fintype T] (H : Msg → Msg) (D : Flock.Deploy Coin Col T) :
    Flock.Efficient H D decodeUniv (fun os => (decodeUniv os, 0)) (fun _ => 0) 0 :=
  efficient_zero H D _ (listWithin_univ D.len D.enc D.θ)

end Claims.RedTeam
