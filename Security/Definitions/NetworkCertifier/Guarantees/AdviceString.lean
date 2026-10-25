import Definitions.NetworkCertifier.AdviceString
import Definitions.NetworkCertifier.Assumptions
import Definitions.NetworkCertifier.Assumptions.AdviceString
import Mathlib.Data.Set.Card
import Mathlib.Algebra.BigOperators.Fin

/-!
# The leak bound: the advice string's charge

The advice string is the constant-rate Program's content advice, so the accepted-runs bound
`A_c + W·log2 #Σ_sync + n·log2 D` holds with `A_c = Σ_w b w`: the public per-window caps on the advice string's
charge, fixed before the run. These guarantees state it, for any charge `bits` that meets `BitsCount` and any compiler
that is precommitted and deterministic. With advice v1's charge they hold with no assumption on the charge
(`Guarantees/AdviceV1` here and in `Definitions`). Acceptance is existential (`AcceptedWithin`): a run counts
if some advice within the caps explains it, so no premise holds only of an honest run. Each run's egress must equal the
compiled Program's (`Accepted`). The all-frames egress tie backs that only when no step unit can be faulty: every step
unit drawn `all`, or a sampled audit at `δ ≥ miss(1)`. At a smaller `δ` an accepted sampled audit gives `Accepted` only
with the faulty units' committed gates held as advice (`SampledAuditAccepted`), so the charge adds their committed bits
and `log2` of the number of possible faulty sets, except with probability `δ + ε_ks + ε_bind` over the selection
(`SampledLeakAccepted`, with `SampledAuditAccepted` as its hypothesis and `EncardAccSeqsSampledLe` its count,
`Definitions/NetworkCertifier/Guarantees/SampledCharge`). It charges every value of the free inputs other than the
registered ones, not the caps here, since `RAcc` pins the inputs only to the trace.

The caps are summed over the `W` windows with advice, whatever the horizon. Charging a window only for the windows up
to it would need a causality assumption about the compiler, which these guarantees don't take.
-/

namespace NetTiming.Guarantees

open Finset

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak bound, on accepted sequences.** Take the constant-rate Program whose content advice is an advice string,
charged `bits a w` in window `w`, where `bits` counts. Its observation sequences of length `n` (within the `W`
windows with advice, and the first `m` windows), with jitter below `D`, accepted with advice charged at most `b w` in
every window `w`, number at most `2^(Σ_w b w) · #Σ_sync^min(W, m) · D^n`. That is
`Σ_w b w + min(W, m)·log2 #Σ_sync + n·log2 D` bits. -/
def EncardAccSeqsWithinLeConstantRate : Prop :=
  ∀ {L K S W : ℕ} {In Adv : Type} (bits : Adv → Fin W → ℕ) (hbits : BitsCount bits) (b : Fin W → ℕ) (T : ℕ)
    (ss : SyncSet) (frames : List In → Adv → Fin L → List LogicalFrame) (op : ℕ → List (Obs L K S) → In)
    (n D m : ℕ) (hn : n ≤ W * T) (hm : n ≤ m * T),
    (AccSeqsWithin bits b (constantRate (S := S) T W ss frames) op n D).encard ≤
      ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞)

/-- **The leak bound, through the wire (W2 and J).** The compiler is precommitted (the same for every secret) and
deterministic (its frames read the advice and the certified ingress only). A decoder reads the wire of the first `n`
steps and recovers every secret in `Zs`, each of whose runs the compiled Program accepts with advice charged at most
`b w` in every window `w`. Then `Zs` has at most `2^(Σ_w b w) · #Σ_sync^min(W, m) · D^n` elements: a covert sender
inside X moves at most `Σ_w b w + min(W, m)·log2 #Σ_sync + n·log2 D` bits through egress, content and timing
together. -/
def EncardDecodableWithinLeCompiled : Prop :=
  ∀ {L K S W : ℕ} {Z In Act Wire Adv Ω : Type}
    (I : Interaction L K S Z In Act Wire) (T : ℕ) (ss : SyncSet)
    (bits : Adv → Fin W → ℕ) (hbits : BitsCount bits) (b : Fin W → ℕ)
    (compilerOf : Z → Compiler L In Adv Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z))
    (n D m : ℕ) (hn : n ≤ W * T) (hm : n ≤ m * T)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs,
      AcceptedWithin bits b (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op n (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z),
    Zs.encard ≤ ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞)

/-- **The leak bound per window (W2 and J).** `EncardDecodableWithinLeCompiled` over all `W` windows, `n = W·T`: a
decoder of the wire recovers at most `∏_w (2^(b w) · #Σ_sync · D^T)` secrets, so a covert sender inside X moves at
most `b w + log2 #Σ_sync + T·log2 D` bits per window `w`. The factor is the bound's, not a count per window: advice
charged to one window can first show in a later one, so the bound is cumulative over the run. -/
def EncardDecodableWithinLeCompiledPerWindow : Prop :=
  ∀ {L K S W : ℕ} {Z In Act Wire Adv Ω : Type}
    (I : Interaction L K S Z In Act Wire) (T : ℕ) (ss : SyncSet)
    (bits : Adv → Fin W → ℕ) (hbits : BitsCount bits) (b : Fin W → ℕ)
    (compilerOf : Z → Compiler L In Adv Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z)) (D : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs, AcceptedWithin bits b (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op (W * T)
      (I.obsRun z (W * T)))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z (W * T)) = z),
    Zs.encard ≤ ((∏ w : Fin W, (2 ^ b w * ss.size * D ^ T) : ℕ) : ℕ∞)

end NetTiming.Guarantees
