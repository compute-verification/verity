import Definitions.Accounting.Attribution
import Models.Accounting.Assumptions.Attribution
import Models.Accounting.Assumptions.PouwFloor
import Models.Accounting.Assumptions.PouwWork

/-!
# Compute attribution: PoUW's credit at one block (trusted, not proved)

The physical layer's compute statement (decisions 111 and 113, `note:20261018T0730Z-draft-attrib`, red-teamed in
`note:20261018T0850Z-draft-rt-attrib`): PoUW's logical certificate says the work was done; this says the block did it.
A block is a location of the hardware (`X ∈ H.units`), its work is the actual circuit's gates placed at its leaves in
the window (`Accounting.PouwWork.credits`, `Placed.work` over `Placed.region`), and the outsourcing term is the chance
that the outside finds the window's salt, not the bytes that cross:

* the salt reaches the window only through the block (`Assumptions.SaltCustody`);
* the block's leak about its coins through its uncertified links, every physical channel among them, is under `kc`
  bits a window (`Assumptions.SaltContainment`, the premise about the prover's host);
* the certifiers' jitter doesn't depend on the salt (`Assumptions.SaltFreeJitter`), so only `∂X`'s records carry the
  salt out, and they are a function of the boundary's accepted observations (`Assumptions.BoundaryObs`), at most
  `#Ac · (#Σ_sync · D^T)^m` of them (`NetTiming.Guarantees.EncardAccSeqsLeConstantRatePerWindow`, proved);
* so what the outside does in the window takes at most `A = #Ac · (#Σ_sync · D^T)^m · 2^kc` values at each draw, and
  with at most `q` noise-hash calls (`Assumptions.OutsideQueries`) it finds the salt with probability at most
  `q · A / #Ωs`, `q · 2^(k − 512)` at `A = 2^k`;
* the floor at that advice (`Assumptions.PouwFloorAdvice`, conjectured, scoped to full-rank per-entry noise) puts the
  work at the block.

What it attributes is the floor `Wb · Nmin`, the work the noise forces and custody pins, not the strategy's `credits`:
the credited gates that no noise forces (the clean products, the non-matmul ops, the peel's clean side) could have run
anywhere, and the account reports them as outsourceable. #2475's demo pins 6,144 of 12,320 credited at one node.

The window is PoUW's, `Verdict.window` (`[ts, td)`, the verifier's ticks). It replaces `PouwLocalOff`'s isolation
(help 0) with custody: a serving block takes in prompts, activations and anything else that doesn't depend on the salt,
and the statement charges none of it. So a block need not hold the whole model. Pipeline stages on other blocks are
fine: what crosses between them is the clean activation, which the served path carries without noise (decisions 105
and 117). Splitting one PoUW unit's reduction across blocks is not: the noised partial sums depend on the salt and
cross before registration, and custody fails. No PoUW unit's `k`-reduction crosses a block's boundary; tensor
parallelism stays inside a block.

Shared custody with memory: one release a block and window can carry compute's salt and memory's `σ_t` and `c_a`
(NET-18). The outside's advice is about the joint release, so `A` is shared, not summed: bundling doesn't raise it. The
threshold binds on the smallest coin, memory's `c_a`, since all of the advice may concern one coin. This statement is
compute's, about the salt alone.

Placement is the actual circuit's, `Accounting.Placed` (each gate's leaf, tick and cell): the block's credit is the
gates `Placed.region` puts at its leaves in the window, and the statement quantifies over every strategy, since a prover
that passes PoUW need not run the declared circuit as placed. Each block has its own salt and its own verdict, whose
units are the PoUW units placed at the block's leaves. A declared execution's engine step
(`Accounting.Requirement.Execution.step`) is not read: the window is in ticks, and which ticks an engine step spans is
the audit's timing.

Not proved. The intended proof, on each draw's strategy `fun s => S (s, d)`: B4's argument
(`Accounting.Guarantees.B4`) over the window's ticks with the jitter factor dropped (`SaltFreeJitter`) and the
uncertified links' term replaced by `SaltContainment`'s range; the reduction from every certifier to `∂X`'s by induction
on ticks; `BoundaryObs` and `NetTiming.SecurityProofs.EncardAccSeqsLeConstantRatePerWindow` for `OutsideAdvice` at `A`;
then `PouwFloorAdvice` and the union bound of `Accounting.SecurityProofs.PouwLocalOff`. Listed under `shape.frozen` as
`unstated` until it is.
-/

namespace Accounting.Guarantees

open Accounting.PouwWork

/-- **PoUW's work floor at one block.** For any hardware `H`, any strategy `S` on coins split into the window's salt
and the later draw, any verdict `P`, any location `X` of `H` (the block), credit pricing `π`, marking `m` and
noise-hash marking `hash`: if the strategy is causal, the links are as declared and the certifiers are trusted (B4's
premises) with jitter that doesn't depend on the salt, the salt reaches the window only through the block, the block's
uncertified links carry fewer than `kc` bits about it, `∂X`'s records in the window are a function of the boundary's
accepted observations over `mw ≤ Wn` windows of `T` buckets, the outside makes at most `q` noise-hash calls in the
window, and PoUW's A2, A2b and the floor at `A = #Ac · (#Σ_sync · D^T)^mw · 2^kc` advice values hold, then the
verifier accepts while the block's own gates outside `m` made fewer than `Wb · Nmin` credits in the window with
probability at most `δ_det + δ_luck + δ_floor + q · A / #Ωs`. The attributed work is `Wb · Nmin`. It says which block
computed; within the block, not which GPU. -/
def PouwBlockCredit : Prop :=
  ∀ {L E Ωs Ωd : Type} {Lk K Sz : ℕ} {In Ac : Type} [LinearOrder L] [Fintype L] [Fintype E] [Fintype Ωs]
    [Fintype Ωd] [Fintype Ac]
    (H : Hardware L E) (S : Ωs × Ωd → Actual.Run L) (P : Verdict (Ωs × Ωd) L) (X : Finset L)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) (spec : ∀ r, (S r).A.Spec)
    (π : ∀ r, Fin (S r).A.N → ℕ) (m hash : ∀ r, Finset (Fin (S r).A.N))
    (T Wn : ℕ) (ss : NetTiming.SyncSet) (frames : List In → Ac → Fin Lk → List NetTiming.LogicalFrame)
    (mw D kc q Wb Nmin : ℕ) (δdet δluck δfloor : ℝ),
    X ∈ H.units → mw ≤ Wn → CausalStrategy S H.ℓ → DeclaredLinks S H.ends H.β link →
    TrustedUnits S H.trusted H.cert spec H.record H.bucket H.jitter →
    Assumptions.SaltFreeJitter S H.cert H.record H.bucket H.jitter →
    Assumptions.SaltCustody S H.cert X P.window → Assumptions.SaltContainment S H.cert X P.window kc →
    Assumptions.BoundaryObs K Sz S (Attribution.boundaryCert H X) H.record P.window T Wn ss frames mw D →
    Assumptions.OutsideQueries S hash H.cert X P.window q →
    Assumptions.PouwDetection S P Nmin δdet → Assumptions.PouwLuck S P Nmin δluck →
    Assumptions.PouwFloorAdvice S P X H.cert hash π m Wb Nmin q
      (Fintype.card Ac * (ss.size * D ^ T) ^ mw * 2 ^ kc) δfloor →
    Pouw.pr (fun r => Accepts S P r ∧ credits S P X π m r < Wb * Nmin) ≤
      δdet + δluck + δfloor +
        ((q * (Fintype.card Ac * (ss.size * D ^ T) ^ mw * 2 ^ kc) : ℕ) : ℝ) / Fintype.card Ωs

end Accounting.Guarantees
