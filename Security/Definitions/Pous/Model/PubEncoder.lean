import Definitions.Pous.Model.Audit

/-!
# The public-encoder setup check and audit game

Verbatim copies of the definitions the public-encoder pins (`BandPubMeets64`, `BandPubMeets14`,
`BandPubMeets14Global`) read, from `lean/submissions/public-encoder/PubEncoderModel.lean`. That file already uses the
trusted layer's names (namespace `Pous`): everything but `PubMeets` is in this file, and `PubMeets` in
`Security/Definitions/Pous/Accounting/PubMeets.lean`. So there is no renaming, and the source's
`PubEncoderModel.lean` must stop declaring these names (import this file instead) when it enters, or the two clash.
`Scheme.EncDec`, which no pinned statement reads, stays in the source.

An untrusted server encodes: its encoder `E` outputs `(pp*, C*)` from `(W, ω, r)`, a setup check `V` with its own coins
`ξ` accepts or rejects it as the encoding of the committed `W`, and preprocessing and the audit run on `(pp*, C*)`.
The commitment to `W` and the binding of `C*` are idealized, and the check's cost is not in any statement.
-/

namespace Pous

open scoped ENNReal

variable {M : IdealModel}

/-- **A setup check** for a scheme: its coins `Ξ`, drawn uniformly after the server's output, and the predicate
`accepts W pp C ξ f`: accept `(pp, C)` as the encoding of the committed `W` under coins `ξ`, reaching the primitive
through `f`. -/
structure SetupCheck (sch : Scheme M) : Type 1 where
  /-- The check's coins. -/
  Ξ : Type
  [fintypeΞ : Fintype Ξ]
  [nonemptyΞ : Nonempty Ξ]
  /-- Accept `(pp, C)` as the encoding of `W`. -/
  accepts : Bits sch.n → Bits sch.t → sch.Code → Ξ → (M.Qry → M.Ans) → Prop

attribute [instance] SetupCheck.fintypeΞ SetupCheck.nonemptyΞ

/-- **Completeness**: the honest encoding is accepted, for every `W`, primitive, coins and check coins. -/
def SetupCheck.Complete {sch : Scheme M} (V : SetupCheck sch) : Prop :=
  ∀ (W : Bits sch.n) (ω : M.Ω) (r : sch.Coins) (ξ : V.Ξ),
    V.accepts W (sch.enc W ω r).1 (sch.enc W ω r).2 ξ (M.answer ω)

/-- **The exact check** (milestone 1): decode the server's `(pp, C)` and compare with the committed `W`. No coins. -/
def exactCheck (sch : Scheme M) : SetupCheck sch where
  Ξ := Unit
  accepts W pp C _ f := sch.dec pp C f = W

/-- The server's encoder: any function of `W`, the whole primitive and the coins. -/
abbrev PubEncoder (sch : Scheme M) : Type := Bits sch.n → M.Ω → sch.Coins → Bits sch.t × sch.Code

/-- Preprocessing after a public setup: any function of `W`, the server's `(pp, C)`, the coins, the check's coins and
the whole primitive, with an `S`-bit output. -/
abbrev PubPreprocessor (sch : Scheme M) (Ξ : Type) (S : ℕ) : Type :=
  Bits sch.n → Bits sch.t → sch.Code → sch.Coins → Ξ → M.Ω → Bits S

/-- The randomness of one public-encoder audit: the primitive, the coins, the check's coins and the challenge
indices. -/
abbrev PubOutcome (sch : Scheme M) (Ξ : Type) (k : ℕ) : Type := M.Ω × sch.Coins × Ξ × (Fin k → Fin sch.B)

/-- The check accepts the server's output and the simultaneous audit on it accepts. -/
def pubSimAccepts (sch : Scheme M) (V : SetupCheck sch) {S D Q k : ℕ} (W : Bits sch.n) (E : PubEncoder sch)
    (A₁ : PubPreprocessor sch V.Ξ S) (A₂ : SimResponder sch S D Q k) (x : PubOutcome sch V.Ξ k) : Prop :=
  let ω := x.1
  let r := x.2.1
  let ξ := x.2.2.1
  let I := x.2.2.2
  let pp := (E W ω r).1
  let C := (E W ω r).2
  V.accepts W pp C ξ (M.answer ω) ∧
    (A₂.prog W pp (A₁ W pp C r ξ ω) I).run (M.answer ω) = fun j => C (I j)

/-- The check accepts the server's output and the sequential audit with `k` rounds on it accepts. -/
def pubSeqAccepts (sch : Scheme M) (V : SetupCheck sch) {S D Q : ℕ} (k : ℕ) (W : Bits sch.n) (E : PubEncoder sch)
    (A₁ : PubPreprocessor sch V.Ξ S) (A₂ : SeqResponder sch S D Q) (x : PubOutcome sch V.Ξ k) : Prop :=
  let ω := x.1
  let r := x.2.1
  let ξ := x.2.2.1
  let I := x.2.2.2
  let pp := (E W ω r).1
  let C := (E W ω r).2
  V.accepts W pp C ξ (M.answer ω) ∧
    seqAnswers (M.answer ω) (A₂.step W pp) 0 (List.ofFn I) (A₁ W pp C r ξ ω) = List.ofFn fun j => C (I j)

/-- **The public-encoder security target.** For every `W`, every server encoder, every `S`-bit preprocessing and
every `(D, Q)`-bounded responder, the check and the `k`-challenge audit both accept with probability at most
`δ + ε`. -/
def PubAuditSecure (sch : Scheme M) (V : SetupCheck sch) : Reveal → (S D Q k : ℕ) → (δ ε : ℝ≥0∞) → Prop
  | .simultaneous, S, D, Q, k, δ, ε =>
      ∀ (W : Bits sch.n) (E : PubEncoder sch) (A₁ : PubPreprocessor sch V.Ξ S) (A₂ : SimResponder sch S D Q k),
        pr (pubSimAccepts sch V W E A₁ A₂) ≤ δ + ε
  | .sequential, S, D, Q, k, δ, ε =>
      ∀ (W : Bits sch.n) (E : PubEncoder sch) (A₁ : PubPreprocessor sch V.Ξ S) (A₂ : SeqResponder sch S D Q),
        pr (pubSeqAccepts sch V k W E A₁ A₂) ≤ δ + ε

end Pous
