import Definitions.Accounting.Attribution
import Models.Accounting.Assumptions.Attribution
import Models.Accounting.Assumptions.PouwFloor
import Models.Accounting.Assumptions.PouwWork

/-!
# Compute attribution: PoUW's credit at one block (trusted, not proved)

The physical layer's compute statement (decisions 111, 113 and 118, `note:20261018T0730Z-draft-attrib`, red-teamed in
`note:20261018T0850Z-draft-rt-attrib` and `note:compute-accounting/20261018T1948Z-report-redteam-salt-custody`):
PoUW's logical certificate says the work was done; this says the block did it. A block is a location of the hardware
(`X ∈ H.units`), its work is the actual circuit's gates placed at its leaves in the window
(`Accounting.PouwWork.credits`, `Placed.work` over `Placed.region`), and the outsourcing term is the chance that the
outside finds the window's salt, not the bytes that cross.

The salt is decision 118's: `s` bits per block and window, drawn by the block's ingress certifier from its own
randomness at `ts` (NET-18's `fresh_salt`, `s = 512`), written into the block only, kept in a sealed slot of the
certifier's record and revealed to the verifier on the window's registration receipt. Custody is over the outside only:
the certifiers are the auditor's devices and may depend on the salt, so long as what they send outside the block
doesn't.

* what the outside and the certifiers off the block's boundary can tell at `ts` depends on the window's coin only
  through its public commitment (`Assumptions.SaltCustody`), which hides the salt up to `δhide`
  (`Assumptions.SaltHiding`; `δhide = 0` at the sealed certifier salt);
* where the boundary audit accepts, the block's leak about its coins through its uncertified links, every physical
  channel among them, is under `kc` bits a window (`Assumptions.SaltContainment`, the premise about the prover's host,
  a claim);
* what each certifier sends outside the block until `td` is a function of its open record (its record but the sealed
  salt frame's slot, `Attribution.openCells`) and of jitter that doesn't depend on the salt
  (`Assumptions.SaltFreeJitter`), and where the boundary audit accepts `∂X`'s open records are a function of the
  commitment, the leak and the boundary's accepted observations (`Assumptions.BoundaryObs`), at most
  `#Ac · (#Σ_sync · D^T)^m` of them at each leak value (`NetTiming.Guarantees.EncardAccSeqsLeConstantRatePerWindow`,
  proved);
* so where the boundary audit accepts, what the outside does in the window takes at most
  `A = #Ac · (#Σ_sync · D^T)^m · 2^kc` values at each draw and commitment, and with at most `q` noise-hash calls
  (`Assumptions.OutsideQueries`) it finds the salt with probability at most `q · A / 2^s + δhide`, `q · 2^(k − s) + δhide`
  at `A = 2^k`;
* the floor at that advice (`Assumptions.PouwFloorAdvice`, conjectured, scoped to the noise whose floor is the
  committed words' coin × coin products: H1 is in it while it commits a word at least every 32 keys, the peel's clean
  side is not) puts the work at the block.

The headline is at decision 112's (GRIND) `q = 2^128`, the outside's query budget with no economic argument. The
outsourcing term `q · 2^(k − s)` is at most `2^−64` when `k ≤ s − 192`, and the measured `k` (the boundary's bits plus
`kc`) is 31–59:

* at the certifier's 512-bit salt (`s = 512`, `δhide = 0`), `k ≤ 320`, and the measured `k` gives `2^−353` to `2^−325`,
  261 to 289 bits under `2^−64`;
* at PoUW's epoch salt on `main` (a 32-byte coin, `s = 256` however long its stream, `δhide = 2^−192`), `k ≤ 64`, and the
  measured `k` gives `2^−97` to `2^−69`, 5 to 33 bits under. That salt is opened to the prover at `ts` and shared by
  every block of the epoch, so `SaltCustody` fails at it anyway: custody needs PoUW's noise seeded from the certifier's
  per-block salt.

`s` and `q` stay parameters of the statement.

The event is that PoUW accepts and the boundary audit accepts (`bacc`, its verdict at each coin), the two verdicts the
account conditions the block's credit on. A strategy that fails the boundary audit at some salts is inside the
statement: at those coins it gets no credit, so it is charged as a rejection, and the boundary's premises
(`SaltContainment`, `BoundaryObs`) need to hold only where the audit accepts.

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

Not proved. The intended proof, at each draw and commitment value, on the strategy restricted to the window coins
where `bacc` holds: by induction on the window's ticks, what the outside does and what the certifiers send it are a
function of the leak and the boundary's observations, from `SaltCustody` at `ts` (the outside's and the off-boundary
certifiers' state), `CausalStrategy` at each location, `SaltFreeJitter` for each certifier's sends outside the block
(its open record and the draw's jitter), `BoundaryObs` for `∂X`'s open records and `CausalStrategy` with
`DeclaredLinks` for an off-boundary certifier's (it reads nothing from `X`, since a link to `X` would put it on `∂X`);
then `SaltContainment` and `NetTiming.SecurityProofs.EncardAccSeqsLeConstantRatePerWindow` at each of the `2^kc` leak
values for `OutsideAdvice` at `A`; then `PouwFloorAdvice` at `B = bacc` with `SaltHiding`, and the union bound of
`Accounting.SecurityProofs.PouwLocalOff` (a coin of the event is in `bacc`, so either in `Z` or forced to the floor
unless a block is wrong). Listed under `shape.frozen` as `unstated` until it is.
-/

namespace Accounting.Guarantees

open Accounting.PouwWork

/-- **PoUW's work floor at one block.** For any salt width `s`, hardware `H`, strategy `S` on coins split into the
window's coin (an `s`-bit salt and the hiding randomness of its public commitment `com`) and the later draw, verdict `P`,
boundary-audit verdict `bacc`, location `X` of `H` (the block), sealed record cells `seal` (the salt frame's slot),
credit pricing `π`, marking `m` and noise-hash marking `hash`: if the strategy is causal, the links are as declared and
the certifiers are trusted (B4's premises), what the outside and the certifiers off `∂X` can tell at `ts` depends on
the window's coin only through `com`, which hides the salt up to `δhide`, what each certifier sends outside the block
until `td` is a function of its open record and jitter that doesn't depend on the salt, where the boundary audit
accepts the block's uncertified links carry fewer than `kc` bits about its coins and `∂X`'s open records before `td`
are a function of the commitment, the leak and the boundary's accepted observations over `mw ≤ Wn` windows of `T`
buckets, the outside makes at most `q` noise-hash calls in the window, and PoUW's A2, A2b and the floor at
`A = #Ac · (#Σ_sync · D^T)^mw · 2^kc` advice values hold, then PoUW and the boundary audit both accept while the block's
own gates outside `m` made fewer than `Wb · Nmin` credits in the window with probability at most
`δ_det + δ_luck + δ_floor + q · A / 2^s + δhide`: past the first three, at most `2^−64 + δhide` at `q = 2^128` and
`A ≤ 2^(s − 192)`. The attributed work is `Wb · Nmin`. It says which block computed; within the block, not which GPU. -/
def PouwBlockCredit : Prop :=
  ∀ {L E Ωh Ωd C : Type} {Lk K Sz : ℕ} {In Ac : Type} [LinearOrder L] [Fintype L] [Fintype E] [Fintype Ωh]
    [Fintype Ωd] [Fintype Ac]
    (s : ℕ) (H : Hardware L E) (S : (Fin (2 ^ s) × Ωh) × Ωd → Actual.Run L)
    (P : Verdict ((Fin (2 ^ s) × Ωh) × Ωd) L) (bacc : (Fin (2 ^ s) × Ωh) × Ωd → Prop) (com : Fin (2 ^ s) × Ωh → C)
    (X : Finset L) (seal : L → Set ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) (spec : ∀ r, (S r).A.Spec)
    (π : ∀ r, Fin (S r).A.N → ℕ) (m hash : ∀ r, Finset (Fin (S r).A.N))
    (T Wn : ℕ) (ss : NetTiming.SyncSet) (frames : List In → Ac → Fin Lk → List NetTiming.LogicalFrame)
    (mw D kc q Wb Nmin : ℕ) (δdet δluck δfloor δhide : ℝ),
    X ∈ H.units → mw ≤ Wn → CausalStrategy S H.ℓ → DeclaredLinks S H.ends H.β link →
    TrustedUnits S H.trusted H.cert spec H.record H.bucket H.jitter →
    Assumptions.SaltFreeJitter S H.cert X (Attribution.openCells H.record seal) H.bucket H.jitter P.window →
    Assumptions.SaltCustody S com H.cert X (Attribution.boundaryCert H X) P.window →
    Assumptions.SaltHiding com δhide →
    Assumptions.SaltContainment S bacc H.cert X P.window kc →
    Assumptions.BoundaryObs K Sz S bacc com H.cert X (Attribution.boundaryCert H X)
      (Attribution.openCells H.record seal) P.window T Wn ss frames mw D →
    Assumptions.OutsideQueries S hash H.cert X P.window q →
    Assumptions.PouwDetection S P Nmin δdet → Assumptions.PouwLuck S P Nmin δluck →
    Assumptions.PouwFloorAdvice s S P bacc com X H.cert hash π m Wb Nmin q
      (Fintype.card Ac * (ss.size * D ^ T) ^ mw * 2 ^ kc) δfloor δhide →
    Pouw.pr (fun r => Accepts S P r ∧ bacc r ∧ credits S P X π m r < Wb * Nmin) ≤
      δdet + δluck + δfloor +
        ((q * (Fintype.card Ac * (ss.size * D ^ T) ^ mw * 2 ^ kc) : ℕ) : ℝ) / 2 ^ s + δhide

end Accounting.Guarantees
