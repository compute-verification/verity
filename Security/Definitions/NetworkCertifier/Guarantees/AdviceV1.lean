import Definitions.NetworkCertifier.AdviceV1
import Definitions.NetworkCertifier.Guarantees.AdviceString
-- These two keep `^` on `ℕ` elaborating through `Monoid.npow`, as the statements' records were taken.
import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# The leak bound through the wire with advice v1's charge

`Guarantees.AdviceString`'s two wire bounds with the advice string advice v1's decision streams (`AdviceV1.Streams`)
and its charge (`AdviceV1.bits`, `ChargeCount_v1{N}`'s count). `BitsCount` is not a premise: advice v1's charge meets it
under any registered book (`EncardChargedLe`, `Definitions/NetworkCertifier/Guarantees/AdviceV1Kraft`). The premises left
are the wire's (W2, J) and the compiler's (precommitted, deterministic).

The scope and the one hypothesis left are `EncardChargedLe`'s: the windows are decision-only (their four counts sent as
0), with the asks (`en`) pinned by the always-proved compile slice's scan; and that `ChargeCount_v1`'s circuit computes
`AdviceV1.window` is not proved in Lean (circuit-check holds the circuit to the Python reference, and the Lean
definitions are held to that reference by the charge difftest and `CodeChargesV1`).
-/

namespace NetTiming.Guarantees

open AdviceV1 Finset

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak bound through the wire, with advice v1's charge (W2 and J).** `EncardDecodableWithinLeCompiled` with
advice v1's decision streams under the registered book `bk` and windows of `N` slots, so with no assumption on the
charge: a decoder of the wire's first `n` steps recovers at most `2^(Σ_w b w) · #Σ_sync^min(W, m) · D^n` secrets whose
runs the compiled Program accepts with advice v1 charged at most `b w` in every window `w`. Decision-only windows, and
the circuit's agreement with the reference is the one hypothesis left (module doc). -/
def EncardDecodableWithinLeCompiledV1 : Prop :=
  ∀ {L K S W : ℕ} {Z In Act Wire Ω : Type}
    (I : Interaction L K S Z In Act Wire) (T : ℕ) (ss : SyncSet) (bk : Book) (N : ℕ) (b : Fin W → ℕ)
    (compilerOf : Z → Compiler L In (Streams bk N W) Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z))
    (n D m : ℕ) (hn : n ≤ W * T) (hm : n ≤ m * T)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs,
      AcceptedWithin (AdviceV1.bits bk N) b (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op n
        (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z),
    Zs.encard ≤ ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞)

/-- **The leak bound per window, with advice v1's charge (W2 and J).** `EncardDecodableWithinLeCompiledPerWindow` with
advice v1's decision streams and charge: over all `W` windows, a decoder of the wire recovers at most
`∏_w (2^(b w) · #Σ_sync · D^T)` secrets. Decision-only windows, and the circuit's agreement with the reference is the one
hypothesis left (module doc). -/
def EncardDecodableWithinLeCompiledPerWindowV1 : Prop :=
  ∀ {L K S W : ℕ} {Z In Act Wire Ω : Type}
    (I : Interaction L K S Z In Act Wire) (T : ℕ) (ss : SyncSet) (bk : Book) (N : ℕ) (b : Fin W → ℕ)
    (compilerOf : Z → Compiler L In (Streams bk N W) Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z)) (D : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs, AcceptedWithin (AdviceV1.bits bk N) b
      (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op (W * T) (I.obsRun z (W * T)))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z (W * T)) = z),
    Zs.encard ≤ ((∏ w : Fin W, (2 ^ b w * ss.size * D ^ T) : ℕ) : ℕ∞)

end NetTiming.Guarantees
