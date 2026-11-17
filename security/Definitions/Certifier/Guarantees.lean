module

public import Definitions.Certifier
public import Definitions.Certifier.Assumptions
public import Definitions.Certifier.Ingress
public import Mathlib.Data.Set.Card
import Mathlib.Data.Sym.Sym2.Init

/-! The network attester's guarantees, each proved as `NetTiming.SecurityProofs.G`; a guarantee without that theorem is
open, and nothing may cite it. -/

@[expose] public section

namespace NetTiming.Guarantees

open Finset

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The ingress timing bound, from the records.** As `EncardIngressDecodableLe` below, but the decoder inside X reads
the records of the first `n` steps, everything X's devices (`op`) read, rather than the wire. This needs no W2, and the
wire version follows from it. -/
def EncardIngressObsDecodableLe : Prop :=
  ∀ {L K S : ℕ} {Msg In Act Wire : Type}
    (I : Interaction L K S Msg In Act Wire) (n D : ℕ) (r : Fin L → ℕ)
    (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (c : Fin L → List (List Bool)) (Ms : Set Msg)
    (hacc : ∀ m ∈ Ms, StatusComplete n (I.obsRun m n))
    (hc : ∀ m ∈ Ms, ∀ ℓ, linkFrames (I.obsRun m n) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ m ∈ Ms, dec (I.obsRun m n) = m),
    Ms.encard ≤ (((∏ ℓ, (r ℓ + 1)) ^ n * D ^ n : ℕ) : ℕ∞)

/-- **The ingress timing bound.** Read the interaction inward (`Definitions.Certifier.Ingress`): the secret `m` is the
outside's message, `snd m` chooses when ingress frames arrive and reacts to every record (and so to X's egress), `op`
is X's devices, the record's `rows` are what the `L` ingress links release into X, and `wire` is everything that enters
X. Take messages `Ms` whose runs keep every status complete up to `n`, and whose frames on each link `ℓ`, in the order
released, are a prefix of the prescribed content `c ℓ` (the Program's input, charged as reads). If a decoder inside X
reads the wire and recovers every message in `Ms`, then under W2, W4 with allocation `r ℓ`, and J, there are at most
`(∏_ℓ (r ℓ + 1))^n · D^n` of them: `n · Σ_ℓ log2(r ℓ + 1)` bits of timing, which is `T·log2(r_in + 1)` per ingress
execution phase, and `n·log2 D` of jitter, none under J's `D = 1`. -/
def EncardIngressDecodableLe : Prop :=
  ∀ {L K S : ℕ} {Msg In Act Wire : Type}
    (I : Interaction L K S Msg In Act Wire) (n D : ℕ) (r : Fin L → ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hW4 : Packed I.certifier)
    (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (c : Fin L → List (List Bool)) (Ms : Set Msg)
    (hacc : ∀ m ∈ Ms, StatusComplete n (I.obsRun m n))
    (hc : ∀ m ∈ Ms, ∀ ℓ, linkFrames (I.obsRun m n) ℓ <+: c ℓ)
    (dec : List Wire → Msg) (hdec : ∀ m ∈ Ms, dec (I.wireRun m n) = m),
    Ms.encard ≤ (((∏ ℓ, (r ℓ + 1)) ^ n * D ^ n : ℕ) : ℕ∞)

/-- **Content plus timing, from the records.** As `EncardIngressObsDecodableLe`, but the content is no longer
prescribed: each message's frames on every link `ℓ` are a prefix of `c ℓ` for some content assignment `c` in `Cs`.
Then there are at most `#Cs · (∏_ℓ (r ℓ + 1))^n · D^n` messages: `log2 #Cs` bits of content plus the timing bound. -/
def EncardIngressObsDecodableLeContents : Prop :=
  ∀ {L K S : ℕ} {Msg In Act Wire : Type}
    (I : Interaction L K S Msg In Act Wire) (n D : ℕ) (r : Fin L → ℕ)
    (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (Cs : Set (Fin L → List (List Bool))) (Ms : Set Msg)
    (hacc : ∀ m ∈ Ms, StatusComplete n (I.obsRun m n))
    (hc : ∀ m ∈ Ms, ∃ c ∈ Cs, ∀ ℓ, linkFrames (I.obsRun m n) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ m ∈ Ms, dec (I.obsRun m n) = m),
    Ms.encard ≤ Cs.encard * (((∏ ℓ, (r ℓ + 1)) ^ n * D ^ n : ℕ) : ℕ∞)

/-- **Content plus timing.** As `EncardIngressDecodableLe` (the decoder reads the wire, under W2), but each message's
frames follow some content assignment in `Cs`: at most `#Cs · (∏_ℓ (r ℓ + 1))^n · D^n` messages. -/
def EncardIngressDecodableLeContents : Prop :=
  ∀ {L K S : ℕ} {Msg In Act Wire : Type}
    (I : Interaction L K S Msg In Act Wire) (n D : ℕ) (r : Fin L → ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hW4 : Packed I.certifier)
    (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (Cs : Set (Fin L → List (List Bool))) (Ms : Set Msg)
    (hacc : ∀ m ∈ Ms, StatusComplete n (I.obsRun m n))
    (hc : ∀ m ∈ Ms, ∃ c ∈ Cs, ∀ ℓ, linkFrames (I.obsRun m n) ℓ <+: c ℓ)
    (dec : List Wire → Msg) (hdec : ∀ m ∈ Ms, dec (I.wireRun m n) = m),
    Ms.encard ≤ Cs.encard * (((∏ ℓ, (r ℓ + 1)) ^ n * D ^ n : ℕ) : ℕ∞)

/-- **The K charge's unit: `T·log2(r ℓ + 1)` bits per ingress execution phase.** `EncardIngressObsDecodableLe` over the
monitoring period's first `m` whole windows of `T` buckets, its first `n = m·T` steps: for one content assignment `c`,
at most `(∏_ℓ (r ℓ + 1)^T · D^T)^m` messages. Each execution phase contributes the factor `(r ℓ + 1)^T`, its
`T·log2(r ℓ + 1)` bits, and each window `D^T` of jitter, 1 under J. -/
def EncardIngressObsDecodableLePerExecutionPhase : Prop :=
  ∀ {L K S : ℕ} {Msg In Act Wire : Type}
    (I : Interaction L K S Msg In Act Wire) (T m D : ℕ) (r : Fin L → ℕ)
    (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (c : Fin L → List (List Bool)) (Ms : Set Msg)
    (hacc : ∀ msg ∈ Ms, StatusComplete (m * T) (I.obsRun msg (m * T)))
    (hc : ∀ msg ∈ Ms, ∀ ℓ, linkFrames (I.obsRun msg (m * T)) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ msg ∈ Ms, dec (I.obsRun msg (m * T)) = msg),
    Ms.encard ≤ ((((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m : ℕ) : ℕ∞)

/-- **Content plus timing, per execution phase.** `EncardIngressObsDecodableLeContents` over the first `m` whole windows
of `T` buckets: at most `#Cs · (∏_ℓ (r ℓ + 1)^T · D^T)^m` messages, `log2 #Cs` bits of content plus
`T·Σ_ℓ log2(r ℓ + 1)` bits of timing and `T·log2 D` of jitter per window. -/
def EncardIngressObsDecodableLeContentsPerExecutionPhase : Prop :=
  ∀ {L K S : ℕ} {Msg In Act Wire : Type}
    (I : Interaction L K S Msg In Act Wire) (T m D : ℕ) (r : Fin L → ℕ)
    (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (Cs : Set (Fin L → List (List Bool))) (Ms : Set Msg)
    (hacc : ∀ msg ∈ Ms, StatusComplete (m * T) (I.obsRun msg (m * T)))
    (hc : ∀ msg ∈ Ms, ∃ c ∈ Cs, ∀ ℓ, linkFrames (I.obsRun msg (m * T)) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ msg ∈ Ms, dec (I.obsRun msg (m * T)) = msg),
    Ms.encard ≤ Cs.encard * ((((∏ ℓ, (r ℓ + 1) ^ T) * D ^ T) ^ m : ℕ) : ℕ∞)

/-! ## The schedule's declared bound and the coins: nothing implements either. -/

end NetTiming.Guarantees
