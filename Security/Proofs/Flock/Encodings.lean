module

public import Definitions.Claims.Encodings
public import Proofs.Flock.Encodings.Binding
public import Proofs.Flock.Encodings.Relink

/-!
# The registered encodings: their binding statements, proved, and the composed statements

The four binding statements of `Definitions.Claims.Encodings` are proved with no assumption, each a collision reduction:
the sum by position binds when its parts do (`byPosBinds`), the encoding binds within the unique decoding radius at a
binding column check (`codeBinds`), the link binds the record to the encoding (`linkBinds`), all in
`Encodings/Binding.lean`; and the sampling relation under the sum carries over the link to the record, or the checks
collide (`relinkR`, `Encodings/Relink.lean`, through `M_δ`'s downward closure at every design, `design_M_down`). Their
witnesses are `Claims.Witnesses.EncodingsGeneric`'s.

The composed statements (`CompilerKSLinked`, `CompilerKSRegLinked`) are stated here, over the definitions of
`Definitions.Claims.Encodings`: closed statements over every proof system, compiled verifier and link, not yet proved,
and so no claims.
-/

@[expose] public section

namespace Claims.Encodings

open Claims.Sampling (CommitmentScheme Bext)
open Claims.SamplingGeneric (Arg Compiled)

/-- **The sampled compiler through a link is knowledge sound at the record**: every compiled sampled verifier over a
knowledge-sound proof system for proof blocks that meets the contract, whose commitment is the sum of a binding encoding
and a binding record, with a knowledge-sound link that meets its contract and names its statements, at every `δ ≥ 0`
and every pair of provers (the audit's reading the link's run): the linked bad event has probability at most the audit's
error (`Compiled.KS`'s, at the sum), the link's (`RegLink.ε`) and `Pr[B_link]`. -/
def CompilerKSLinked : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (Lk : RegLink Coin) (H : Msg → Msg) (enc : List Msg → ℕ → Bool) (Ce Cm : CommitmentScheme Bool),
    A.KS H → Lk.A.KS H → L.commit = byPos enc Ce Cm → Ce.Binds H → Cm.Binds H → L.Sessions H →
    Lk.Opens H enc Cm Ce → ∀ δ : ℝ, 0 ≤ δ → ∀ x, L.Scope x → (∀ s, L.parse x = some s → Lk.names s) →
    ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (Pℓ : Prover Ω) (Pa : Prover (Ω × (Fin (Lk.A.V.k Lk.x) → Coin))),
      PrBadLinked H L Lk enc Cm δ x Pℓ Pa ≤
        L.κ δ x + Real.sqrt (L.γ x * PrB H L.V Pa x) + (Bext H L.V L.E _ Pa x).Pr H + Lk.ε H Pℓ +
          (Blink H L Lk Cm Ce x Pℓ Pa).Pr H

/-- **The committed form through a link**: as `CompilerKSLinked`, the link's and the audit's runs accept with a sampled
proof block in the committed trace's incorrect set through the link (`FregL`, fixed by the registration and the tape
before the window's coins) with probability at most `εregLinked`, and so with that set outside `M_δ` with probability at
most `δ + εregLinked`, at every `δ ≥ 0`. -/
def CompilerKSRegLinked : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (Lk : RegLink Coin) (H : Msg → Msg) (enc : List Msg → ℕ → Bool) (Ce Cm : CommitmentScheme Bool),
    A.KS H → Lk.A.KS H → L.commit = byPos enc Ce Cm → Ce.Binds H → Cm.Binds H → L.Sessions H →
    Lk.Opens H enc Cm Ce → ∀ x, L.Scope x → (∀ s, L.parse x = some s → Lk.names s) →
    ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (Pℓ : Prover Ω) (Pa : Prover (Ω × (Fin (Lk.A.V.k Lk.x) → Coin))),
      Pr (fun q : Ω × (Fin (Lk.A.V.k Lk.x) → Coin) × Lk.A.E.C Lk.x × (Fin (L.V.k x) → Coin) =>
          (run H Pℓ Lk.A.V Lk.x q.1 q.2.1).out = .acc ∧ (run H Pa L.V x (q.1, q.2.1) q.2.2.2).out = .acc ∧
            ∃ u ∈ L.checked x q.2.2.2, u ∈ FregL H L Lk enc Cm Ce x Pℓ Pa q.1 q.2.1 q.2.2.1) ≤
          εregLinked H L Lk Cm Ce x Pℓ Pa ∧
        ∀ s, L.parse x = some s → ∀ δ : ℝ, 0 ≤ δ →
          Pr (fun q : Ω × (Fin (Lk.A.V.k Lk.x) → Coin) × Lk.A.E.C Lk.x × (Fin (L.V.k x) → Coin) =>
              (run H Pℓ Lk.A.V Lk.x q.1 q.2.1).out = .acc ∧ (run H Pa L.V x (q.1, q.2.1) q.2.2.2).out = .acc ∧
                FregL H L Lk enc Cm Ce x Pℓ Pa q.1 q.2.1 q.2.2.1 ∉ s.design.M (L.part s) δ) ≤
            δ + εregLinked H L Lk Cm Ce x Pℓ Pa

end Claims.Encodings
