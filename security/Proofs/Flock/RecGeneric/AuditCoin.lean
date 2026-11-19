module

public import Definitions.Core.Scheme
public import Proofs.Flock.RecGeneric.Sampled

/-!
# The recursive audit with the coin in the outer statement

`Flock.Guarantees.RecursiveAuditBoundReg` hands the outer verifier the sample: its statement is `stmt S R`. Here the
outer statement is `stmt R ω`, a function of the registration and the coin, so two coins that draw one sample are two
statements, and the verifier is handed the coin, never the sample. The game: the prover registers `R`, the verifier
flips `L R`'s coin `ω`, and the compiled verifier runs at `stmt R ω` (`sessionOf`).

B1's objects are typed by the sample, so they are read at the coin-tagged design (`tagged`): `L`'s coins, the sample at
`ω` being `L`'s units each tagged with `ω`, and `ω`'s own unit, which reads nothing. A sample of `tagged L` names its
coin (`coinOf`), and the outer statement at a sample is the statement at the coin it names (`stmtAt`); the guarantee
reads witnesses only at the samples the design draws (`JointExtractor.SoundOn`), so a witness proves only the units its
coin sampled. `coinJoint` is `jointOf` there,
at the one registration the prover sent: its trial at a position draws a coin whose sample reads the position and runs
`E*` at that coin's statement.

Statements and their definitions only.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.SampledGeneric

variable {n : ℕ}

/-- A unit of `L`'s coin-tagged design: a coin and one of `L`'s units, or the coin alone. -/
noncomputable def untag (L : SamplingDesign n) (u : Fin (Fintype.card (L.Ω × Option (Fin n)))) :
    L.Ω × Option (Fin n) :=
  (Fintype.equivFin _).symm u

open Classical in
/-- **The coin-tagged design**: `L`'s coins; the sample at `ω` is `L`'s, each unit tagged with `ω`, and `ω`'s own
unit. -/
noncomputable def tagged (L : SamplingDesign n) : SamplingDesign (Fintype.card (L.Ω × Option (Fin n))) where
  Ω := L.Ω
  draw ω := (insert (ω, none) ((L.draw ω).image fun u => (ω, some u))).map (Fintype.equivFin _).toEmbedding

open Classical in
/-- **The coin a sample of `tagged L` names**: the coin of its least unit. -/
noncomputable def coinOf (L : SamplingDesign n) (S : Finset (Fin (Fintype.card (L.Ω × Option (Fin n))))) : L.Ω :=
  if h : S.Nonempty then (untag L (S.min' h)).1 else Classical.arbitrary L.Ω

/-- **The outer statement at a sample of `tagged L`**: `stmt` at the coin it names. -/
noncomputable def stmtAt (L : SamplingDesign n) (stmt : L.Ω → Msg)
    (S : Finset (Fin (Fintype.card (L.Ω × Option (Fin n))))) : Msg :=
  stmt (coinOf L S)

/-- **`jointOf` at the coin**: at `L`'s coin-tagged design and the one registration `R` (`Unit`), the outer statement at
a sample is `stmtAt`, a tagged unit `(ω, u)` reads `reads u` and opens as `opens ω u`, and a coin's own
unit reads nothing. -/
noncomputable abbrev coinJoint {Coin W Reg Dh : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    {Hc : List UInt8 → Dh} (H : Msg → Msg) (V : Verifier Coin) (Rl : Rel W) (E : Claims.Extractor Coin W V)
    (L : SamplingDesign n) (B : Binding Reg Hc) (R : Reg) (stmt : L.Ω → Msg) (reads : Fin n → Finset B.Pos)
    (opens : L.Ω → Fin n → W → B.Pos → B.Opening) :=
  jointOf H V Rl E (fun S (_ : Unit) => stmtAt L stmt S) ({ B with cs := fun _ => B.cs R } : Binding Unit Hc)
    (fun u => ((untag L u).2.map reads).getD ∅)
    (fun S _ u w c => ((untag L u).2.map fun u' => opens (coinOf L S) u' w c).getD default)

end Claims.RecGeneric

namespace Flock.Guarantees

open Claims Claims.RecGeneric FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.SampledGeneric
open scoped ENNReal

/-- **The recursive audit with the coin in the outer statement, at the registered design.** For every compiled verifier
`K` over the inner protocol `I` and the server `Sv` with L1's hypotheses at every registration and coin (`Comp.OuterKSAt`,
nonnegative `κ_O` and `γ_O`, `Comp.Contract` at `stmt R ω`), design `L R`
read from the registration, binding commitment `B` to the trace, and witnesses of `stmt R ω` that open `R`'s commitment
at every position a unit of `ω`'s sample reads and make that unit correct in every trace that agrees with those
openings: at every strategy `σ` of the game (register `R`, flip `L R`'s
coin `ω`, run `K.V` at `stmt R ω`) whose link finders meet ecr/sha-512 at the trials' counted calls (`Trials.CR`),
`Verity.AuditBound` holds at the design `L σ.1` it registered, with the incorrect set `F` of the plurality of the trials
at the coin (`coinJoint`), and the error
$$\mathbb E_\omega\, \mathrm{Comp.err}(\mathrm{stmt}(R, \omega), P_\omega) + \varepsilon_{\rm bind},$$
with `P_ω` the claims prover `σ` plays at `ω`. -/
def RecursiveAuditBoundCoin : Prop :=
  ∀ (Coin W : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Hash.Msg → Hash.Msg) (I : Proto Coin W)
    (Sv : Server) (K : Claims.RecGeneric.Comp Coin) (mI : Fin I.nT → ℕ) (Mo : ℕ) (n : ℕ) (Reg Dh : Type)
    (Hc : List UInt8 → Dh) (L : Reg → SamplingDesign n) (B : Binding Reg Hc)
    (Correct : Reg → B.Trace → Fin n → Prop) (stmt : (R : Reg) → (L R).Ω → Hash.Msg)
    (reads : Fin n → Finset B.Pos) (opens : (R : Reg) → (L R).Ω → Fin n → W → B.Pos → B.Opening) (M : ℕ),
    I.WF → I.KSAbortLin H → I.LastCoinFree H → Sv.Binds →
    (∀ R ω, K.OuterKSAt H (stmt R ω) ∧ 0 ≤ K.κO (stmt R ω) ∧ 0 ≤ K.γO (stmt R ω) ∧
      K.Contract H (Sv.IVStar I.V I.ans) (fun v => I.opens (Sv.openView v)) (stmt R ω)) →
    (∀ R ω u w, u ∈ (L R).draw ω → K.R I (stmt R ω) w →
      (∀ c ∈ reads u, B.com (opens R ω u w c) = B.cs R c) ∧
        ∀ tr : B.Trace, (∀ c ∈ reads u, B.val (opens R ω u w c) = tr c) → Correct R tr u) →
    1 ≤ M →
    ∀ σ : Strategy (.send Reg fun R => .coin (L R).Ω fun ω => sessionOf H K.V (stmt R ω)),
      ((coinJoint H K.V (K.R I) (K.E I Sv) (L σ.1) B σ.1 (stmt σ.1) reads (opens σ.1)).toTrials
          (tagged (L σ.1))).CR () (fun ω => ⟨(σ.2 ω).1, fun _ => PUnit.unit⟩) M →
      Verity.AuditBound (fun _ => L σ.1)
        (fun p => prob p (.send Reg fun R => .coin (L R).Ω fun ω => sessionOf H K.V (stmt R ω)) σ)
        (fun o => o = true)
        (fun _ => incorrect B Correct σ.1
          (((coinJoint H K.V (K.R I) (K.E I Sv) (L σ.1) B σ.1 (stmt σ.1) reads (opens σ.1)).toTrials
            (tagged (L σ.1))).plur () (fun ω => ⟨(σ.2 ω).1, fun _ => PUnit.unit⟩)))
        (avg (fun ω => ENNReal.ofReal (K.err I Sv H (σ.2 ω).1 (stmt σ.1 ω) mI Mo)) +
          ((coinJoint H K.V (K.R I) (K.E I Sv) (L σ.1) B σ.1 (stmt σ.1) reads (opens σ.1)).toTrials
            (tagged (L σ.1))).bindErr () (fun ω => ⟨(σ.2 ω).1, fun _ => PUnit.unit⟩) M)

end Flock.Guarantees
