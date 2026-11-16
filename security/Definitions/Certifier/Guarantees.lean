module

public import Definitions.Certifier
public import Definitions.Certifier.Assumptions
public import Definitions.Certifier.Ingress
public import Mathlib.Data.Set.Card
import Mathlib.Data.Sym.Sym2.Init

/-!
# The network attester's guarantees

Each guarantee is a `def G : Prop`, stated over `Protocol`'s definitions with the assumptions it rests on as hypotheses
(`Assumptions`). Its proof is the theorem `NetTiming.SecurityProofs.G : NetTiming.Guarantees.G`, and a guarantee without
one is open: nothing may cite it until it is proved. The statements that are only steps of these proofs, Theorem 1(b)'s
counting bound among them, are in `Proofs.Certifier.Lemmas`, under the same names.

`docs/network-transparency/timing-channel.md`: the ingress bound (§2.5, "Ingress content"; §3.2).
-/

@[expose] public section

namespace NetTiming.Guarantees

open Finset

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-! ## The ingress bound: what the outside can deliver into X through timing -/

/-- **The ingress timing bound.** Read the interaction inward (`Definitions.Certifier.Ingress`): the secret `m` is the
outside's message, `snd m` chooses when ingress frames arrive and reacts to every record (and so to X's egress), `op`
is X's devices, and the record's `rows` are what the `L` ingress links release into X. Take messages `Ms` whose runs
keep every status complete up to `n`, and whose frames on each link `ℓ`, in the order released, are a prefix of `c ℓ`
for some content assignment `c` in `Cs`. If a decoder inside X reads the records of the first `n` steps, everything
X's devices read, and recovers every message in `Ms`, then under W4 with allocation `r ℓ` and J there are at most
`#Cs · (∏_ℓ (r ℓ + 1))^n · D^n` of them: `log2 #Cs` bits of content, `n · Σ_ℓ log2(r ℓ + 1)` bits of timing, which is
`T·log2(r_in + 1)` per ingress execution phase of `T` buckets, and `n·log2 D` of jitter, none under J's `D = 1`. A
decoder that reads the wire reads a function of the records under W2 (`Regenerates`). -/
def EncardIngressObsDecodableLeContents : Prop :=
  ∀ {L K S : ℕ} {Msg In Act Wire : Type}
    (I : Interaction L K S Msg In Act Wire) (n D : ℕ) (r : Fin L → ℕ)
    (hW4 : Packed I.certifier) (hr : Allocated I.certifier r) (hJ : JitterBounded I.certifier D)
    (Cs : Set (Fin L → List (List Bool))) (Ms : Set Msg)
    (hacc : ∀ m ∈ Ms, StatusComplete n (I.obsRun m n))
    (hc : ∀ m ∈ Ms, ∃ c ∈ Cs, ∀ ℓ, linkFrames (I.obsRun m n) ℓ <+: c ℓ)
    (dec : List (Obs L K S) → Msg) (hdec : ∀ m ∈ Ms, dec (I.obsRun m n) = m),
    Ms.encard ≤ Cs.encard * (((∏ ℓ, (r ℓ + 1)) ^ n * D ^ n : ℕ) : ℕ∞)

end NetTiming.Guarantees
