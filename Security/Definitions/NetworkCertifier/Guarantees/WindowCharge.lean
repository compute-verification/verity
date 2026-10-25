import Definitions.NetworkCertifier.WindowCharge
import Definitions.NetworkCertifier.Guarantees.AdviceString

/-!
# The leak bound through the wire with advice v1's whole-window charge

`Guarantees.AdviceString`'s two wire bounds with the advice string each window's decisions, hits and order entries
(`WindowCharge.Streams`) and its charge (`WindowCharge.bits`, `ChargeCount_v2{N, H, O}`'s count: the whole window as
`code_v1` codes it, every count inside the coder). `BitsCount` is not a premise: the charge meets it under any
registered book (`EncardChargedLeWhole`, `Definitions/NetworkCertifier/Guarantees/WindowChargeKraft`). The premises left
are the wire's (W2, J) and the compiler's (precommitted, deterministic).

The scope and the one hypothesis left are `EncardChargedLeWhole`'s: every advice v1 window, with n_resumes and n_lag
sent as 0 and the asks pinned by the always-proved compile slice's scan; and that `ChargeCount_v2`'s circuit computes
`WindowCharge.window` is not proved in Lean (circuit-check holds the circuit to the Python reference, and the Lean
definitions are held to that reference by the window charge's difftest and `CodeChargesWhole`).
-/

namespace NetTiming.Guarantees

open WindowCharge Finset

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak bound through the wire, with advice v1's whole-window charge (W2 and J).**
`EncardDecodableWithinLeCompiled` with each window's decisions, hits and order entries under the registered book `bk`
and windows of `N` decision, `H` hit and `O` order slots, so with no assumption on the charge: a decoder of the wire's
first `n` steps recovers at most `2^(Σ_w b w) · #Σ_sync^min(W, m) · D^n` secrets whose runs the compiled Program
accepts with advice charged at most `b w` in every window `w`. The scope is `EncardChargedLeWhole`'s, and the
circuit's agreement with the reference is the one hypothesis left (module doc). -/
def EncardDecodableWithinLeCompiledV1Whole : Prop :=
  ∀ {L K S W : ℕ} {Z In Act Wire Ω : Type}
    (I : Interaction L K S Z In Act Wire) (T : ℕ) (ss : SyncSet) (bk : Book) (N H O : ℕ) (b : Fin W → ℕ)
    (compilerOf : Z → Compiler L In (Streams bk N H O W) Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z))
    (n D m : ℕ) (hn : n ≤ W * T) (hm : n ≤ m * T)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs,
      AcceptedWithin (WindowCharge.bits bk N H O) b (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op n
        (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z),
    Zs.encard ≤ ((2 ^ (∑ w, b w) * ss.size ^ min W m * D ^ n : ℕ) : ℕ∞)

/-- **The leak bound per window, with advice v1's whole-window charge (W2 and J).**
`EncardDecodableWithinLeCompiledPerWindow` with each window's decisions, hits and order entries and their charge:
over all `W` windows, a decoder of the wire recovers at most `∏_w (2^(b w) · #Σ_sync · D^T)` secrets. The scope is
`EncardChargedLeWhole`'s, and the circuit's agreement with the reference is the one hypothesis left (module doc). -/
def EncardDecodableWithinLeCompiledPerWindowV1Whole : Prop :=
  ∀ {L K S W : ℕ} {Z In Act Wire Ω : Type}
    (I : Interaction L K S Z In Act Wire) (T : ℕ) (ss : SyncSet) (bk : Book) (N H O : ℕ) (b : Fin W → ℕ)
    (compilerOf : Z → Compiler L In (Streams bk N H O W) Ω) (env : Z → Ω)
    (hpre : CompilerPrecommitted compilerOf) (hdet : ∀ z, CompilerDeterministic (compilerOf z)) (D : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D)
    (Zs : Set Z)
    (hacc : ∀ z ∈ Zs, AcceptedWithin (WindowCharge.bits bk N H O) b
      (compiledProgram (S := S) T W ss (compilerOf z) (env z)) I.op (W * T) (I.obsRun z (W * T)))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z (W * T)) = z),
    Zs.encard ≤ ((∏ w : Fin W, (2 ^ b w * ss.size * D ^ T) : ℕ) : ℕ∞)

end NetTiming.Guarantees
