import Definitions.Pouw.Assumptions.Lead

/-!
# The lead's work guarantee (trusted statements)

The lead is `tag-blc4a-t64-rot/served/s8/k32/w/l1w/l1a/min` (`Pouw.Lead.lead`). Its guarantee's proof is
`Pouw.SecurityProofs.Lead.Compose.G : Pouw.Guarantees.Lead.G`, composed from the pieces under `Proofs/Pouw/Lead/`.

* `G`: for every prover strategy the node can run on a call, its priced W1 work summed over the coin values on which
  the call is accepted is at least the honest work less the gap, at each of those values (`|O| · honest − |Ω| · gap`),
  or one of those values' executions hashed a SHA-512 collision with the honest commitment. With `O` every coin value,
  it is the average: `E_ω[work] ≥ honest − gap`. The gap is `gapRegs`, a defined function of the layout. An
  accepted call exchanges at most its share of its forward's egress cap (`Layout.callBits`), so a forward's network
  terms sum to at most its cap's. Acceptance is what the verifier sees: a call it admits (`Layout.Admits`: an inner
  dimension of at least 1 and at most `2^40` MMA tiles; it refuses every other call), the honest root registered
  within the call's bits. So `G` is conditional on the registered root being the honest one.
* `G_tail`: `G`'s tail form, under `G`'s premises: the share of the coin values on which a strategy is accepted, hashed
  no collision and worked below `32 · (honestRegs − gapFixed)` is at most `etaLead`. `G` sums work over the accepted
  coins, so it can't tell a small loss on every coin from a large one on a few; `G_tail` can. The fixed gap holds the
  head, the network, the coin-branch mean and a slack of 0.08% of stock, and `p*`; the tail holds the tables' union
  bound and the coin-branch count's tail. `GapFixedAt` evaluates the fixed gap: at most 0.162% of stock at the three
  points at cap 32 (0.486% at cap 320), and at decode 2.06% of served compute. `TailAt`: `etaLead ≤ 2^−40` there, for
  calls of at most `2^30` tiles. `Honest` is `G_tail`'s honest lemma too, as their premises are the same.
* `GapAt`: the gap's share of stock at T 512 × 64, T 512 × 512 and T 8,192, for every call of `n, K ≥ 4,096`, at cap
  32: at most 0.190%, 0.168% and 0.165%. `GapAt320`: at cap 320, at most 0.513%, 0.491% and 0.488%. At these points
  every row is served, and stock and served compute differ only by the padding of `n` to a multiple of 8.
* `GapAtDecode`: at decode (`m = 1`, committed as 16 rows), the gap is at most 3.71% of served compute (the unpadded
  MACs, `servedRegs`) at cap 32 and 4.03% at cap 320. Served compute is what credit counts, so this is decode's
  figure; the share of the padded stock, 0.231% and 0.252%, counts 15 padding rows as compute and is secondary.
* `Honest`: the hypotheses are jointly satisfiable at every call the verifier admits: the nominal W1 table, and some
  coin stream, commitment format and set of strategies meeting every premise, with an honest strategy accepted at
  every coin value. The commitment format is existential, so `Honest` does not say the served format meets
  `CommitL1w`. Every call of at most `2^15` rows, `2^20` columns and an inner dimension from 1 to `2^16` is admitted
  (`Statement.admits_of_caps`), Llama-3.2-1B's included.
* `Nontrivial`: the conclusion fails for a verifier that accepts everything: at the three points the honest work
  exceeds the gap, and a strategy that does nothing does no work and hashes nothing.
* `NontrivialTail`: the same for `G_tail`: at the three points the honest work exceeds the fixed gap, and `etaLead < 1`
  at calls of at most `2^30` tiles.
-/

namespace Pouw.Guarantees.Lead

open Pouw.Lead Pouw.Assumptions.Lead Pouw.Dimension.H100

/-- **The lead's work guarantee.** For every price table `pr` meeting `H32`, hash `H`, call
`m × K × n` at `T` tokens (every call, decode included: `lead` commits it over whole MMA tiles), data `κ`, certifier's
coin stream `stream`, commitment format `C` with its form classifier `isA` and set `Real` of the strategies the node
can run, under the named premises (an exchanged bit replacing at most `betaLead`), every strategy `σ` the node's
processors run (`can`) and every set `O` of coin values on which its call is accepted (a call the verifier admits,
within its bits) has
`Σ_{ω ∈ O} work(ω) ≥ 32 · (|O| · honestRegs − |Ω| · gapRegs)`, or at some `ω ∈ O` its execution hashed a SHA-512
collision with the honest commitment's messages. -/
def G : Prop :=
  ∀ (pr : Prices100) (H : Hash.Msg → Hash.Digest) (T m n K : ℕ)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (stream : CoinStream (lead T m n K)) (C : Pouw.Lead.Commit (lead T m n K))
    (isA : Hash.Msg → Bool) (can Real : Strategy (lead T m n K) → Prop),
    CommitL1w H C isA κ →
    H32 pr → W1 pr Real → TileGrain H C κ stream Real → OneInstructionMinimal Real → CardWithinAccelW1 can Real →
    ShaWrites isA Real → HonestHashRuled H C κ Real →
    OutsourceWithinEgressLead pr (betaLead (lead T m n K) C) H C κ Real →
    ∀ σ, can σ → ∀ O : Finset (Coins (lead T m n K)), (∀ ω ∈ O, Accepted (lead T m n K) H C κ σ ω) →
      32 * ((O.card : ℚ) * honestRegs (lead T m n K) C -
          (Fintype.card (Coins (lead T m n K)) : ℚ) * gapRegs (lead T m n K) C) ≤
        ∑ ω ∈ O, (σ.work pr ω : ℚ) ∨
      ∃ ω ∈ O, Break (lead T m n K) H C κ σ ω

/-- **The lead's work guarantee, tail form.** Under `G`'s premises, for every strategy `σ` the node's processors run,
the share of the coin values on which its call is accepted, its execution hashed no SHA-512 collision with the honest
commitment, and its priced W1 work is below `32 · (honestRegs − gapFixed)` is at most `etaLead`. The fixed gap
(`gapFixed`) is the head, the network, the coin-branch mean and slack and `p*`; `etaLead` is the tables' union bound
over tiles and steps and the coin-branch count's tail beyond its slack. -/
def G_tail : Prop :=
  ∀ (pr : Prices100) (H : Hash.Msg → Hash.Digest) (T m n K : ℕ)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (stream : CoinStream (lead T m n K)) (C : Pouw.Lead.Commit (lead T m n K))
    (isA : Hash.Msg → Bool) (can Real : Strategy (lead T m n K) → Prop),
    CommitL1w H C isA κ →
    H32 pr → W1 pr Real → TileGrain H C κ stream Real → OneInstructionMinimal Real → CardWithinAccelW1 can Real →
    ShaWrites isA Real → HonestHashRuled H C κ Real →
    OutsourceWithinEgressLead pr (betaLead (lead T m n K) C) H C κ Real →
    ∀ σ, can σ →
      Pouw.pr (fun ω : Coins (lead T m n K) =>
          Accepted (lead T m n K) H C κ σ ω ∧ ¬ Break (lead T m n K) H C κ σ ω ∧
            (σ.work pr ω : ℚ) < 32 * (honestRegs (lead T m n K) C - gapFixed (lead T m n K) C)) ≤
        etaLead (lead T m n K)

/-- A call's commitment at the lead's layout: its words chunks are `wordChunks`. -/
def ChunksOf (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) : Prop := C.chunks = wordChunks L

/-- **The gap at the three points, cap 32**: for every call of `n, K ≥ 4,096`, the gap's share of stock is at most
0.190% at T 512 × 64, 0.168% at T 512 × 512 and 0.165% at T 8,192. -/
def GapAt : Prop :=
  ∀ n K : ℕ, 4096 ≤ n → 4096 ≤ K →
    (∀ C : Pouw.Lead.Commit (pt512x64 n K), ChunksOf _ C → gapShare _ C ≤ 190 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (pt512x512 n K), ChunksOf _ C → gapShare _ C ≤ 168 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (pt8192 n K), ChunksOf _ C → gapShare _ C ≤ 165 / 100000)

/-- **The gap at the three points, cap 320**: at most 0.513%, 0.491% and 0.488%. -/
def GapAt320 : Prop :=
  ∀ n K : ℕ, 4096 ≤ n → 4096 ≤ K →
    (∀ C : Pouw.Lead.Commit (lead320 512 64 n K), ChunksOf _ C → gapShare _ C ≤ 513 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (lead320 512 512 n K), ChunksOf _ C → gapShare _ C ≤ 491 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (lead320 8192 8192 n K), ChunksOf _ C → gapShare _ C ≤ 488 / 100000)

/-- **Decode** (`m = 1`), which `G` covers padded to 16 rows: for every call of `n, K ≥ 4,096`, the gap is at most
3.71% of served compute (the call's unpadded MACs, `servedRegs`) at cap 32 and 4.03% at cap 320. Secondary: at most
0.231% and 0.252% of the padded call's stock, which counts the 15 padding rows as compute. -/
def GapAtDecode : Prop :=
  ∀ T n K : ℕ, 4096 ≤ n → 4096 ≤ K →
    (∀ C : Pouw.Lead.Commit (lead T 1 n K), ChunksOf _ C →
      gapServed _ C ≤ 371 / 10000 ∧ gapShare _ C ≤ 231 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (lead320 T 1 n K), ChunksOf _ C →
      gapServed _ C ≤ 403 / 10000 ∧ gapShare _ C ≤ 252 / 100000)

/-- **The tail's fixed gap at the three points and decode**: for every call of `n, K ≥ 4,096`, `gapFixed` is at most
0.162% of stock at T 512 × 64, T 512 × 512 and T 8,192 at cap 32 and 0.486% at cap 320; at decode, at most 2.06%
of served compute at cap 32 and 2.38% at cap 320, and 0.129% and 0.149% of the padded stock. -/
def GapFixedAt : Prop :=
  ∀ T n K : ℕ, 4096 ≤ n → 4096 ≤ K →
    (∀ C : Pouw.Lead.Commit (pt512x64 n K), ChunksOf _ C → gapFixedShare _ C ≤ 162 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (pt512x512 n K), ChunksOf _ C → gapFixedShare _ C ≤ 162 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (pt8192 n K), ChunksOf _ C → gapFixedShare _ C ≤ 162 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (lead320 512 64 n K), ChunksOf _ C → gapFixedShare _ C ≤ 486 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (lead320 512 512 n K), ChunksOf _ C → gapFixedShare _ C ≤ 486 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (lead320 8192 8192 n K), ChunksOf _ C → gapFixedShare _ C ≤ 486 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (lead T 1 n K), ChunksOf _ C →
      gapFixedServed _ C ≤ 206 / 10000 ∧ gapFixedShare _ C ≤ 129 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (lead320 T 1 n K), ChunksOf _ C →
      gapFixedServed _ C ≤ 238 / 10000 ∧ gapFixedShare _ C ≤ 149 / 100000)

/-- **The tail at the three points and decode**: for every call of `n, K ≥ 4,096` with at most `2^30` committed tiles
(Llama-3.2-1B's largest call has `2^29.2`), `etaLead ≤ 2^−40`. The tail doesn't read the cap, so it holds at cap 320
too. The tables' tail is about `2^−71` a tile and step; the coin-branch tail is the concentration piece's. -/
def TailAt : Prop :=
  ∀ T n K : ℕ, 4096 ≤ n → 4096 ≤ K →
    ∀ L ∈ [pt512x64 n K, pt512x512 n K, pt8192 n K, lead T 1 n K],
      L.rowBlocks * L.colBlocks * L.S ≤ 2 ^ 30 → etaLead L ≤ 1 / 2 ^ 40

/-- **The honest lemma**: at every call of the lead the verifier admits (`Layout.Admits`), for every coin stream and
every commitment format and form classifier meeting `CommitL1w`, the hypotheses are jointly satisfiable: the nominal
W1 table meets `H32`, and some set of strategies meets every premise of `G`, with an honest strategy accepted at every
coin value. The protocol's stream and format are among those it covers. And at every call some format and classifier
meet `CommitL1w`, so the first part is not vacuous. -/
def Honest : Prop :=
  (∀ (T m n K : ℕ) (H : Hash.Msg → Hash.Digest)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (stream : CoinStream (lead T m n K)) (C : Pouw.Lead.Commit (lead T m n K)) (isA : Hash.Msg → Bool),
    CommitL1w H C isA κ → (lead T m n K).Admits →
    ∃ (Real : Strategy (lead T m n K) → Prop) (σ : Strategy (lead T m n K)),
      CommitL1w H C isA κ ∧ H32 h100Nominal ∧ W1 h100Nominal Real ∧ TileGrain H C κ stream Real ∧
      OneInstructionMinimal Real ∧ CardWithinAccelW1 Real Real ∧ ShaWrites isA Real ∧ HonestHashRuled H C κ Real ∧
      OutsourceWithinEgressLead h100Nominal (betaLead (lead T m n K) C) H C κ Real ∧ Real σ ∧
      ∀ ω, Accepted (lead T m n K) H C κ σ ω) ∧
  ∀ (T m n K : ℕ) (H : Hash.Msg → Hash.Digest)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ),
    ∃ (C : Pouw.Lead.Commit (lead T m n K)) (isA : Hash.Msg → Bool), CommitL1w H C isA κ

/-- **The nontrivial lemma**: `G`'s conclusion fails for a verifier that accepts everything. At the three points the
honest work exceeds the gap, and a strategy that does nothing does no priced work and hashes nothing, so with every
coin value accepted only a collision among the honest commitment's own messages would meet the conclusion. -/
def Nontrivial : Prop :=
  ∀ n K : ℕ, 4096 ≤ n → 4096 ≤ K →
    (∀ C : Pouw.Lead.Commit (pt512x64 n K), ChunksOf _ C → gapRegs _ C < honestRegs _ C) ∧
    (∀ C : Pouw.Lead.Commit (pt512x512 n K), ChunksOf _ C → gapRegs _ C < honestRegs _ C) ∧
    (∀ C : Pouw.Lead.Commit (pt8192 n K), ChunksOf _ C → gapRegs _ C < honestRegs _ C) ∧
    ∀ (L : Pouw.Lead.Layout) (pr : Prices100), ∃ σ : Strategy L, ∀ ω, σ.work pr ω = 0 ∧ (σ.exec ω).hashed = []

/-- **The tail form's nontrivial lemma**: `G_tail`'s conclusion fails for a verifier that accepts everything. At the
three points the honest work exceeds the fixed gap, and at calls of at most `2^30` tiles `etaLead < 1`. So
`Nontrivial`'s strategy that does nothing, accepted at every coin value and below `32 · (honestRegs − gapFixed)` at
each, puts every coin value in `G_tail`'s event, more than `etaLead`, unless the honest commitment's own messages
collide. -/
def NontrivialTail : Prop :=
  ∀ n K : ℕ, 4096 ≤ n → 4096 ≤ K → ∀ L ∈ [pt512x64 n K, pt512x512 n K, pt8192 n K],
    (∀ C : Pouw.Lead.Commit L, ChunksOf L C → gapFixed L C < honestRegs L C) ∧
      (L.rowBlocks * L.colBlocks * L.S ≤ 2 ^ 30 → etaLead L < 1)

end Pouw.Guarantees.Lead
