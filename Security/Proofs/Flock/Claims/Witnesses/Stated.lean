import Proofs.Flock.Claims.Witnesses.Sampling

/-!
# Each claim fails somewhere, stated over the spec

The lock's `claims` section (`Security/lean-audit.json`) names a theorem `C_fails` for each claim `C`: past its `∀`
and `∃` binders it is `¬ C …`, so `C` is not true of everything. A guarantee's statement reads only the spec, so each
quantifies (`∃`) over the toy instance it fails at, and its proof is the toy witness of `Witnesses.Shared`, `Flock` or
`Sampling`. A record whose decoder's columns are words of bits takes the zero word as an instance (`Zero (List Bool)`),
which the statement quantifies over too. `Complete`'s, Flock's and the sampling one's are `CompleteFails`'; the
ZK-recursion compiler's are `Witnesses.Recursion`'s and `RecursionMeets`'.
-/

namespace Claims

/-! ## The shared forms -/

/-- **`KS` fails**: on a statement with no witness, at every `κ < 1` and `γ`, some verifier fails it at every
extractor. -/
theorem KS_fails {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (R : Rel W)
    (κ γ : Msg → ℝ) (x : Msg) (hR : ∀ w, ¬ R x w) (hκ : κ x < 1) :
    ∃ V : Verifier Coin, ∀ E : Extractor Coin W V, ¬ Claims.KS H V R E κ γ :=
  ⟨_, fun E => Witness.KS_fails H R E κ γ x hR hκ⟩

/-- **`KSLin` fails** there too, at some finders and every `β`. -/
theorem KSLin_fails {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (R : Rel W)
    (κ γ β : Msg → ℝ) (x : Msg) (hR : ∀ w, ¬ R x w) (hκ : κ x < 1) :
    ∃ (V : Verifier Coin) (Fe : FinderFamily), ∀ E : Extractor Coin W V, ¬ Claims.KSLin H V R E κ γ β Fe :=
  ⟨_, _, fun E => Witness.KSLin_fails H R E κ γ β x hR hκ⟩

/-- **`Bits` fails**: on a statement with no witness, at every `λ > 0`, some verifier fails it at every extractor. -/
theorem Bits_fails {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (R : Rel W)
    (lam : Msg → ℝ) (x : Msg) (hR : ∀ w, ¬ R x w) (hlam : 0 < lam x) :
    ∃ V : Verifier Coin, ∀ E : Extractor Coin W V, ¬ Claims.Bits H V R E lam :=
  ⟨_, fun E => Witness.Bits_fails H R E lam x hR hlam⟩

/-- **`ScopeMatches` fails** at an empty scope and a check that takes everything. -/
theorem ScopeMatches_fails : ¬ Claims.ScopeMatches (fun _ => False) (fun _ => true) :=
  Witness.ScopeMatches_fails

/-- **`ZKmalR` fails**: at every `ε_zk < 1` and `γ`, some prover and simulator fail it. -/
theorem ZKmalR_fails {N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (εzk γ : Msg → ℝ) (x : Msg) (hε : εzk x < 1) :
    ∃ (P : Unit → Prover (N × Ω)) (S : Simulator), ¬ Claims.ZKmalR H (fun _ => 1) P (fun _ _ => True) S εzk γ :=
  ⟨_, _, Witness.ZKmalR_fails H [] [] εzk γ x hε⟩

/-- **`ZKmalRx` fails**: at every `ε_zk < 1` and `γ`, some prover and simulator fail it, at one tape at every
statement, where it is `ZKmalR` (`ZKMal.zkmalRx_const`). -/
theorem ZKmalRx_fails {N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (εzk γ : Msg → ℝ) (x : Msg) (hε : εzk x < 1) :
    ∃ (P : Unit → (x : Msg) → Prover (N × Ω)) (S : Simulator),
      ¬ Claims.ZKmalRx (fun _ => N) (fun _ => Ω) H (fun _ => 1) P (fun _ _ => True) S εzk γ :=
  ⟨fun _ _ => ZKMal.quietP N Ω [], ZKMal.collSim [] [],
    (ZKMal.zkmalRx_const H _ (fun _ => ZKMal.quietP N Ω []) _ _ εzk γ).not.2 (Witness.ZKmalR_fails H [] [] εzk γ x hε)⟩

/-- **`Nontrivial` fails**: some verifier rejects a recorded run. -/
theorem Nontrivial_fails {Coin : Type} [Inhabited Coin] (H : Msg → Msg) :
    ∃ (V : Verifier Coin) (recs : List (Claims.Recorded V)), ¬ Claims.Nontrivial H V recs :=
  ⟨_, _, Witness.Nontrivial_fails H⟩

/-! ## Flock -/

/-- **Flock's `KS` fails** at some record. -/
theorem Flock.KS_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ D : Claims.Flock.Deploy Coin Unit Unit, ¬ Claims.Flock.KS H D :=
  ⟨_, Witness.Flock.KS_fails H⟩

/-- **Flock's `Bits` fails** at some record. -/
theorem Flock.Bits_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ D : Claims.Flock.Deploy Coin Unit Unit, ¬ Claims.Flock.Bits H D :=
  ⟨_, Witness.Flock.Bits_fails H⟩

/-- **Flock's `ScopeMatches` fails** at some record. -/
theorem Flock.ScopeMatches_fails {Coin : Type} :
    ∃ D : Claims.Flock.Deploy Coin Unit Unit, ¬ Claims.Flock.ScopeMatches D :=
  ⟨_, Witness.Flock.ScopeMatches_fails⟩

/-- **Flock's `Nontrivial` fails** at some record and recorded run. -/
theorem Flock.Nontrivial_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (D : Claims.Flock.Deploy Coin Unit Unit) (recs : List (Claims.Recorded D.V)),
      ¬ Claims.Flock.Nontrivial H D recs :=
  ⟨_, _, Witness.Flock.Nontrivial_fails H⟩

/-- **Flock's `ZK` fails**: at every `ε_zk < 1` and `γ`, at some masked record, which carries its prover and
simulator. -/
theorem Flock.ZK_fails (H : Msg → Msg) (εzk γ : Msg → ℝ) (x : Msg) (hε : εzk x < 1) :
    ∃ D : Claims.Flock.ZkDeploy Unit Unit Unit, ¬ Claims.Flock.ZK H D εzk γ :=
  ⟨_, Witness.Flock.ZK_fails H [] [] εzk γ x hε⟩

/-- **Flock's `Efficient` fails** at the zero bound, at some record. -/
theorem Flock.Efficient_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (z : Zero (List Bool)) (D : Claims.Flock.Deploy Coin (List Bool) Unit),
      ¬ (letI := z; Claims.Flock.Efficient H D 0) :=
  ⟨Witness.Flock.zeroCol, _, Witness.Flock.Efficient_fails H⟩

/-! ## The sampling compiler -/

/-- **The sampling `KS` fails** at some record. -/
theorem Sampling.KS_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ D : Claims.Sampling.Deploy Coin Unit Unit, ¬ Claims.Sampling.KS H D :=
  ⟨_, Witness.Sampling.KS_fails H⟩

/-- **The sampling `KSCount` fails** at some record. -/
theorem Sampling.KSCount_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ D : Claims.Sampling.Deploy Coin Unit Unit, ¬ Claims.Sampling.KSCount H D :=
  ⟨_, Witness.Sampling.KSCount_fails H⟩

/-- **The sampling `Bits` fails** at some record. -/
theorem Sampling.Bits_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ D : Claims.Sampling.Deploy Coin Unit Unit, ¬ Claims.Sampling.Bits H D :=
  ⟨_, Witness.Sampling.Bits_fails H⟩

/-- **`KSLinOn` fails** somewhere: else the sampling `KS`, which is `KSLinOn` at each of its decoders, would hold at
every record. -/
theorem Sampling.KSLinOn_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (V : Verifier Coin) (Scope : Msg → Prop) (R : Rel Claims.Sampling.Wit)
      (E : Extractor Coin Claims.Sampling.Wit V) (κ γ β : Msg → ℝ) (Fe : FinderFamily),
      ¬ Claims.Sampling.KSLinOn H V Scope R E κ γ β Fe := by
  by_contra h
  push Not at h
  exact Witness.Sampling.KS_fails H fun _ _ _ _ => h _ _ _ _ _ _ _ _

/-- **`BitsLin` fails** somewhere: else the sampling `Bits`, which is `BitsLin` at each of its decoders, would hold at
every record. -/
theorem Sampling.BitsLin_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (V : Verifier Coin) (Scope : Msg → Prop) (R : Rel Claims.Sampling.Wit)
      (E : Extractor Coin Claims.Sampling.Wit V) (Fe : FinderFamily) (lam : Msg → ℝ),
      ¬ Claims.Sampling.BitsLin H V Scope R E Fe lam := by
  by_contra h
  push Not at h
  exact Witness.Sampling.Bits_fails H fun _ _ _ _ => h _ _ _ _ _ _

/-- **The sampling `Binding` fails** at some record. -/
theorem Sampling.Binding_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] :
    ∃ D : Claims.Sampling.Deploy Coin Unit Unit, ¬ Claims.Sampling.Binding D :=
  ⟨_, Witness.Sampling.Binding_fails⟩

/-- **The sampling `Universal` fails** at some record. -/
theorem Sampling.Universal_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] :
    ∃ D : Claims.Sampling.Deploy Coin Unit Unit, ¬ Claims.Sampling.Universal D :=
  ⟨_, Witness.Sampling.Universal_fails⟩

/-- **The sampling `Nontrivial` fails** at some record and recorded run. -/
theorem Sampling.Nontrivial_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (D : Claims.Sampling.Deploy Coin Unit Unit) (recs : List (Claims.Recorded D.V)),
      ¬ Claims.Sampling.Nontrivial H D recs :=
  ⟨_, _, Witness.Sampling.Nontrivial_fails H⟩

/-- **The sampling `ZK` fails**: at every `ε_zk < 1` and `γ`, at some masked record, which carries its prover and
simulator. -/
theorem Sampling.ZK_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (εzk γ : Msg → ℝ) (x : Msg) (hε : εzk x < 1) :
    ∃ D : Claims.Sampling.ZkDeploy Coin Unit Unit, ¬ Claims.Sampling.ZK H D εzk γ :=
  ⟨_, Witness.Sampling.ZK_fails (Coin := Coin) H [] [] εzk γ x hε⟩

/-- **The sampling `Efficient` fails** at the zero bound, at some record. -/
theorem Sampling.Efficient_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (z : Zero (List Bool)) (D : Claims.Sampling.Deploy Coin (List Bool) Unit),
      ¬ (letI := z; Claims.Sampling.Efficient H D 0) :=
  ⟨Witness.Sampling.zeroCol, _, Witness.Sampling.Efficient_fails H⟩

/-- **The sampling `ScopeMatches` fails** at some record. -/
theorem Sampling.ScopeMatches_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] :
    ∃ (z : Zero (List Bool)) (D : Claims.Sampling.Deploy Coin (List Bool) Unit),
      ¬ (letI := z; Claims.Sampling.ScopeMatches D) :=
  ⟨Witness.Sampling.zeroCol, _, Witness.Sampling.ScopeMatches_fails⟩

end Claims
