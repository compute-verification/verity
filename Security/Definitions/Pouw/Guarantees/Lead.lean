import Definitions.Pouw.Assumptions.Lead

/-!
# The lead's work guarantee (trusted statements)

The lead is `tag-blc4a-t64-rot/served/s8/k32/w/l1w/l1a/min` (`Pouw.Lead.lead`). Its guarantee's proof is
`Pouw.SecurityProofs.Lead.Compose.G : Pouw.Guarantees.Lead.G`, composed from the pieces under `Proofs/Pouw/Lead/`.

* `G`: for every prover strategy the node can run on a call, its priced W1 work summed over the certifier's entropy
  values at whose coins the call is accepted is at least the honest work less the gap, at each of those values
  (`|O| · honest − |Ω| · gap`), or one of those values' executions hashed a SHA-512 collision with the honest
  commitment. The coins at an entropy value are the ones the certifier releases (`released`), which `CertifierCoins`
  (`trusted-host`) makes a one-to-one decoding of it. With `O` every value, it is the average over uniform coins:
  `E_ω[work] ≥ honest − gap`. The gap is `gapRegs`, a defined function of the layout. An
  accepted call exchanges at most its share of its forward's egress cap (`Layout.callBits`), so a forward's network
  terms sum to at most its cap's. Acceptance is what the verifier sees: a call it admits (`Layout.Admits`: an inner
  dimension of at least 1 and at most `2^40` MMA tiles; it refuses every other call), the honest root registered
  within the call's bits. So `G` is conditional on the registered root being the honest one.
* `GapAt`: the gap's share of stock at T 512 × 64, T 512 × 512 and T 8,192, for every call of `n, K ≥ 4,096`, at cap
  32: at most 0.190%, 0.168% and 0.165%. `GapAt320`: at cap 320, at most 0.514%, 0.492% and 0.489%. At these points
  every row is served, and stock and served compute differ only by the padding of `n` to a multiple of 8.
* `GapAtDecode`: at decode (`m = 1`, committed as 16 rows), the gap is at most 3.71% of served compute (the unpadded
  MACs, `servedRegs`) at cap 32 and 4.03% at cap 320. Served compute is what credit counts, so this is decode's
  figure; the share of the padded stock, 0.231% and 0.252%, counts 15 padding rows as compute and is secondary.
* `Honest`: the hypotheses are jointly satisfiable at every call the verifier admits: some certifier running
  `certCode`, the nominal W1 table, and some coin stream, commitment format and set of strategies meeting every other
  premise, with an honest strategy accepted at every coin value. The commitment format is existential, so `Honest`
  does not say the served format meets `CommitL1w`. Every call of at most `2^15` rows, `2^20` columns and an inner
  dimension from 1 to `2^16` is admitted (`Statement.admits_of_caps`), Llama-3.2-1B's included.
* `Nontrivial`: the conclusion fails for a verifier that accepts everything: at the three points the honest work
  exceeds the gap, and a strategy that does nothing does no work and hashes nothing.
-/

namespace Pouw.Guarantees.Lead

open Pouw.Lead Pouw.Assumptions.Lead Pouw.Dimension.H100

/-- **The lead's work guarantee.** For every price table `pr` meeting `H32`, hash `H`, call
`m × K × n` at `T` tokens (every call, decode included: `lead` commits it over whole MMA tiles), data `κ`, certifier's
coin stream `stream`, commitment format `C` with its form classifier `isA`, set `Real` of the strategies the node can
run, and network certifier `cert` run as `runC` with its entropy's decoding `coinsOf`, under the named premises
(`CertifierCoins` among them, and an exchanged bit replacing at most `betaLead`), every strategy `σ` the node's
processors run (`can`) and every set `O` of entropy values at whose released coins its call is accepted (a call the
verifier admits, within its bits) has
`Σ_{e ∈ O} work(released e) ≥ 32 · (|O| · honestRegs − |Ω| · gapRegs)`, `|Ω|` the number of coin values, or at
some `e ∈ O` its execution hashed a SHA-512 collision with the honest commitment's messages. -/
def G : Prop :=
  ∀ (pr : Prices100) (H : Hash.Msg → Hash.Digest) (T m n K : ℕ)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (stream : CoinStream (lead T m n K)) (C : Pouw.Lead.Commit (lead T m n K))
    (isA : Hash.Msg → Bool) (can Real : Strategy (lead T m n K) → Prop)
    (Device Ent : Type) (cert : Device) (coinsOf : Ent ≃ Coins (lead T m n K))
    (runC : Device → Hash.Digest × Ent → Hash.Digest × Coins (lead T m n K)),
    CertifierCoins cert coinsOf runC → CommitL1w H C isA κ →
    H32 pr → W1 pr Real → TileGrain H C κ stream Real → OneInstructionMinimal Real → CardWithinAccelW1 can Real →
    ShaWrites isA Real → HonestHashRuled H C κ Real →
    OutsourceWithinEgressLead pr (betaLead (lead T m n K) C) H C κ Real →
    ∀ σ, can σ → ∀ O : Finset Ent, (∀ e ∈ O, Accepted (lead T m n K) H C κ σ (released cert runC σ e)) →
      32 * ((O.card : ℚ) * honestRegs (lead T m n K) C -
          (Fintype.card (Coins (lead T m n K)) : ℚ) * gapRegs (lead T m n K) C) ≤
        ∑ e ∈ O, (σ.work pr (released cert runC σ e) : ℚ) ∨
      ∃ e ∈ O, Break (lead T m n K) H C κ σ (released cert runC σ e)

/-- A call's commitment at the lead's layout: its words chunks are `wordChunks`. -/
def ChunksOf (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) : Prop := C.chunks = wordChunks L

/-- **The gap at the three points, cap 32**: for every call of `n, K ≥ 4,096`, the gap's share of stock is at most
0.190% at T 512 × 64, 0.168% at T 512 × 512 and 0.165% at T 8,192. -/
def GapAt : Prop :=
  ∀ n K : ℕ, 4096 ≤ n → 4096 ≤ K →
    (∀ C : Pouw.Lead.Commit (pt512x64 n K), ChunksOf _ C → gapShare _ C ≤ 190 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (pt512x512 n K), ChunksOf _ C → gapShare _ C ≤ 168 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (pt8192 n K), ChunksOf _ C → gapShare _ C ≤ 165 / 100000)

/-- **The gap at the three points, cap 320**: at most 0.514%, 0.492% and 0.489%. -/
def GapAt320 : Prop :=
  ∀ n K : ℕ, 4096 ≤ n → 4096 ≤ K →
    (∀ C : Pouw.Lead.Commit (lead320 512 64 n K), ChunksOf _ C → gapShare _ C ≤ 514 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (lead320 512 512 n K), ChunksOf _ C → gapShare _ C ≤ 492 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (lead320 8192 8192 n K), ChunksOf _ C → gapShare _ C ≤ 489 / 100000)

/-- **Decode** (`m = 1`), which `G` covers padded to 16 rows: for every call of `n, K ≥ 4,096`, the gap is at most
3.71% of served compute (the call's unpadded MACs, `servedRegs`) at cap 32 and 4.03% at cap 320. Secondary: at most
0.231% and 0.252% of the padded call's stock, which counts the 15 padding rows as compute. -/
def GapAtDecode : Prop :=
  ∀ T n K : ℕ, 4096 ≤ n → 4096 ≤ K →
    (∀ C : Pouw.Lead.Commit (lead T 1 n K), ChunksOf _ C →
      gapServed _ C ≤ 371 / 10000 ∧ gapShare _ C ≤ 231 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (lead320 T 1 n K), ChunksOf _ C →
      gapServed _ C ≤ 403 / 10000 ∧ gapShare _ C ≤ 252 / 100000)

/-- **The honest lemma**: at every call of the lead the verifier admits (`Layout.Admits`), the hypotheses are jointly
satisfiable: some certifier meets `CertifierCoins`, the nominal W1 table meets `H32`, and for some coin stream,
commitment format and set of strategies every other premise of `G` holds, with an honest strategy accepted at every
coin value. The stream and the format are existential, not the protocol's. -/
def Honest : Prop :=
  ∀ (T m n K : ℕ) (H : Hash.Msg → Hash.Digest)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ), (lead T m n K).Admits →
    (∃ (Device Ent : Type) (cert : Device) (coinsOf : Ent ≃ Coins (lead T m n K))
      (runC : Device → Hash.Digest × Ent → Hash.Digest × Coins (lead T m n K)), CertifierCoins cert coinsOf runC) ∧
    ∃ (stream : CoinStream (lead T m n K)) (C : Pouw.Lead.Commit (lead T m n K)) (isA : Hash.Msg → Bool)
      (Real : Strategy (lead T m n K) → Prop) (σ : Strategy (lead T m n K)),
      CommitL1w H C isA κ ∧ H32 h100Nominal ∧ W1 h100Nominal Real ∧ TileGrain H C κ stream Real ∧
      OneInstructionMinimal Real ∧ CardWithinAccelW1 Real Real ∧ ShaWrites isA Real ∧ HonestHashRuled H C κ Real ∧
      OutsourceWithinEgressLead h100Nominal (betaLead (lead T m n K) C) H C κ Real ∧ Real σ ∧
      ∀ ω, Accepted (lead T m n K) H C κ σ ω

/-- **The nontrivial lemma**: `G`'s conclusion fails for a verifier that accepts everything. At the three points the
honest work exceeds the gap, and a strategy that does nothing does no priced work and hashes nothing, so with every
coin value accepted only a collision among the honest commitment's own messages would meet the conclusion. -/
def Nontrivial : Prop :=
  ∀ n K : ℕ, 4096 ≤ n → 4096 ≤ K →
    (∀ C : Pouw.Lead.Commit (pt512x64 n K), ChunksOf _ C → gapRegs _ C < honestRegs _ C) ∧
    (∀ C : Pouw.Lead.Commit (pt512x512 n K), ChunksOf _ C → gapRegs _ C < honestRegs _ C) ∧
    (∀ C : Pouw.Lead.Commit (pt8192 n K), ChunksOf _ C → gapRegs _ C < honestRegs _ C) ∧
    ∀ (L : Pouw.Lead.Layout) (pr : Prices100), ∃ σ : Strategy L, ∀ ω, σ.work pr ω = 0 ∧ (σ.exec ω).hashed = []

end Pouw.Guarantees.Lead
