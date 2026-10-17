import Proofs.Flock.Claims.Witnesses.Recursion
import Proofs.Flock.Claims.Witnesses.Sampling

/-!
# Each claim fails somewhere, stated over the spec

The lock's `claims` section (`Security/lean-audit.json`) names a theorem `C_fails` for each claim `C`: past its `∀`
and `∃` binders it is `¬ C …`, so `C` is not true of everything. A guarantee's statement reads only the spec, so each
quantifies (`∃`) over the toy instance it fails at, and its proof is the toy witness of `Witnesses.Shared`, `Flock`,
`Sampling` or `Recursion`. A record whose decoder's columns are words of bits takes the zero word as an instance
(`Zero (List Bool)`), which the statement quantifies over too. `Complete`'s, Flock's and the sampling one's are
`CompleteFails`'.
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

/-- **`SimCost` fails**: some simulator makes more than one call. -/
theorem SimCost_fails (H : Msg → Msg) : ∃ S : Simulator, ¬ Claims.SimCost H S fun _ => 1 :=
  ⟨_, Witness.SimCost_fails H [] []⟩

/-- **`ExtractRunsLe` fails**: at every verifier, some extractor reruns the prover. -/
theorem ExtractRunsLe_fails {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (V : Verifier Coin) : ∃ E : Extractor Coin W V, ¬ Claims.ExtractRunsLe H E fun _ => 0 :=
  ⟨_, Witness.ExtractRunsLe_fails H V⟩

/-- **`ExtractRunsExpLe` fails**: at every verifier, some extractor reruns the prover in expectation. -/
theorem ExtractRunsExpLe_fails {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (V : Verifier Coin) : ∃ E : Extractor Coin W V, ¬ Claims.ExtractRunsExpLe H E fun _ => 0 :=
  ⟨_, Witness.ExtractRunsExpLe_fails H V⟩

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

/-- **Flock's `ZK` fails**: at every `ε_zk < 1` and `γ`, at some masked record, prover and simulator. -/
theorem Flock.ZK_fails {N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (εzk γ : Msg → ℝ) (x : Msg) (hε : εzk x < 1) :
    ∃ (D : Claims.Flock.Deploy Unit Unit Unit) (P : Claims.Flock.Wit → Prover (N × Ω)) (S : Simulator),
      ¬ Claims.Flock.ZK H D P S εzk γ :=
  ⟨_, _, _, Witness.Flock.ZK_fails H [] [] εzk γ x hε⟩

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

/-- **`BitsLin` fails** somewhere: else the sampling `Bits`, which is `BitsLin` at each of its decoders, would hold at
every record. -/
theorem Sampling.BitsLin_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (V : Verifier Coin) (R : Rel Claims.Sampling.Wit) (E : Extractor Coin Claims.Sampling.Wit V)
      (Fe : FinderFamily) (lam : Msg → ℝ), ¬ Claims.Sampling.BitsLin H V R E Fe lam := by
  by_contra h
  push Not at h
  exact Witness.Sampling.Bits_fails H fun _ _ _ _ => h _ _ _ _ _

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

/-- **The sampling `ZK` fails**: at every `ε_zk < 1` and `γ`, at some record, prover and simulator. -/
theorem Sampling.ZK_fails {Coin N Ω : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [Fintype N] [Nonempty N]
    [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (εzk γ : Msg → ℝ) (x : Msg) (hε : εzk x < 1) :
    ∃ (D : Claims.Sampling.Deploy Coin Unit Unit) (P : Claims.Sampling.Wit → Prover (N × Ω))
      (S : Claims.Sampling.SimLeak), ¬ Claims.Sampling.ZK H D P S εzk γ :=
  ⟨_, _, _, Witness.Sampling.ZK_fails H [] [] εzk γ x hε⟩

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

/-- **`ExtractOwnLe` fails**: at every verifier, some extractor calls `H`. -/
theorem Sampling.ExtractOwnLe_fails {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (V : Verifier Coin) : ∃ E : Extractor Coin W V, ¬ Claims.Sampling.ExtractOwnLe H E fun _ => 0 :=
  ⟨_, Witness.Sampling.ExtractOwnLe_fails H V⟩

/-! ## The ZK-recursion compiler -/

/-- **The recursion's `KS` fails** at some record, inner protocol and outer record. -/
theorem Recursion.KS_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit)
      (DO : Claims.Flock.Deploy Coin Unit Unit), ¬ Claims.Recursion.KS H D I DO :=
  ⟨_, _, _, Witness.Recursion.KS_fails H⟩

/-- **The private variant's `KS` fails** at some record, inner protocol and outer record. -/
theorem Recursion.KSPriv_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit)
      (DO : Claims.Flock.Deploy Coin Unit Unit), ¬ Claims.Recursion.KSPriv H D I DO :=
  ⟨_, _, _, Witness.Recursion.KSPriv_fails H⟩

/-- **The recursion's `KS` over Flock fails** at some record and inner and outer Flock records. -/
theorem Recursion.KSFlock_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (D : Claims.Recursion.Deploy Coin) (DI DO : Claims.Flock.Deploy Coin Unit Unit),
      ¬ Claims.Recursion.KSFlock H D DI DO :=
  ⟨_, _, _, Witness.Recursion.KSFlock_fails H⟩

/-- **The recursion's `Bits` fails** at some record, inner protocol and outer record. -/
theorem Recursion.Bits_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit)
      (DO : Claims.Flock.Deploy Coin Unit Unit), ¬ Claims.Recursion.Bits H D I DO :=
  ⟨_, _, _, Witness.Recursion.Bits_fails H⟩

/-- **The recursion's `Bits` over Flock fails** at some record and inner and outer Flock records. -/
theorem Recursion.BitsFlock_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (D : Claims.Recursion.Deploy Coin) (DI DO : Claims.Flock.Deploy Coin Unit Unit),
      ¬ Claims.Recursion.BitsFlock H D DI DO :=
  ⟨_, _, _, Witness.Recursion.BitsFlock_fails H⟩

/-- **The recursion's `Efficient` fails** at some record, inner protocol, outer record and decoder. -/
theorem Recursion.Efficient_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit)
      (DO : Claims.Flock.Deploy Coin Unit Unit) (decO : List (ℕ × Unit) → List Unit),
      ¬ Claims.Recursion.Efficient H D I DO decO :=
  ⟨_, _, _, _, Witness.Recursion.Efficient_fails H⟩

/-- **The recursion's `ScopeMatches` fails** at some record and inner protocol. -/
theorem Recursion.ScopeMatches_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] :
    ∃ (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit),
      ¬ Claims.Recursion.ScopeMatches D I :=
  ⟨_, _, Witness.Recursion.ScopeMatches_fails⟩

/-- **The recursion's `Complete` fails**: at every `δ_c`, wherever `δ_c < 1`, at some hash, record and inner
protocol. -/
theorem Recursion.Complete_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (δc : Msg → ℝ)
    (x : Msg) (hδ : δc x < 1) :
    ∃ (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit),
      ¬ Claims.Recursion.Complete H D I δc :=
  ⟨_, _, _, Witness.Recursion.Complete_fails δc x hδ⟩

/-- **The recursion's `Nontrivial` fails** at some record and recorded run. -/
theorem Recursion.Nontrivial_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (D : Claims.Recursion.Deploy Coin) (recs : List (Claims.Recorded D.V)), ¬ Claims.Recursion.Nontrivial H D recs :=
  ⟨_, _, Witness.Recursion.Nontrivial_fails H⟩

/-- **The recursion's `ZK` fails**: at every outer simulator family, at some record, inner protocol, outer record and
prover. -/
theorem Recursion.ZK_fails {Coin N Ω : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [Fintype N]
    [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (SO : ℕ → Msg → ℕ → Claims.Recursion.OSim) :
    ∃ (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit)
      (DO : Claims.Flock.Deploy Coin Unit Unit) (P : Claims.Flock.Wit → Prover (N × Ω)),
      ¬ Claims.Recursion.ZK H D I DO P SO :=
  ⟨_, _, _, _, Witness.Recursion.ZK_fails H SO⟩

/-- **The private variant's `ZK` fails** there too. -/
theorem Recursion.ZKPriv_fails {Coin N Ω : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [Fintype N]
    [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (SO : ℕ → Msg → ℕ → Claims.Recursion.OSim) :
    ∃ (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit)
      (DO : Claims.Flock.Deploy Coin Unit Unit) (P : Claims.Flock.Wit → Prover (N × Ω)),
      ¬ Claims.Recursion.ZKPriv H D I DO P SO :=
  ⟨_, _, _, _, Witness.Recursion.ZKPriv_fails H SO⟩

/-- **The inner protocol's `KS` fails** at some inner protocol. -/
theorem Recursion.Inner.KS_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ I : Claims.Recursion.Inner Coin Claims.Flock.Wit, ¬ I.KS H :=
  ⟨_, Witness.Recursion.Inner.KS_fails H⟩

/-- **The inner protocol's `KS` against aborting provers fails** at some inner protocol. -/
theorem Recursion.Inner.KSAbort_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    (H : Msg → Msg) : ∃ I : Claims.Recursion.Inner Coin Claims.Flock.Wit, ¬ I.KSAbort H :=
  ⟨_, Witness.Recursion.Inner.KSAbort_fails H⟩

end Claims
