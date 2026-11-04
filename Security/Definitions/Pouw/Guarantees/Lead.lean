import Definitions.Pouw.Assumptions.Lead
import Definitions.Pouw.Lead.Deployed

/-!
# The lead's work guarantee (trusted statements)

The lead is `tag-blc4a-t64-rot/served/s8/k32/w/l1w/l1a/min` (`Pouw.Lead.lead`). Its guarantee's proof is
`Pouw.SecurityProofs.Lead.Compose.G : Pouw.Guarantees.Lead.G`, composed from the pieces under `Proofs/Pouw/Lead/`.

* `G`: for every prover strategy the node can run on a call, its priced W1 work, plus `betaLead` a bit it exchanged,
  summed over the certifier's entropy values at whose coins the call is accepted, is at least the honest work less
  the gap at each of those values (`|O| · honest − |Ω| · gap`), or one of those values' executions hashed a SHA-512
  collision with the honest commitment. The coins at an entropy value are the ones the certifier releases
  (`released`), which `CertifierCoins` (`trusted-host`) makes a one-to-one decoding of it. With `O` every value, it
  is the average over uniform coins: `E_ω[work + β · egress] ≥ honest − gap`. The gap is `gapRegs`, a defined function
  of the layout. Acceptance is what the verifier sees of a call: a call it admits (`Layout.Admits`: an inner dimension
  of at least 1 and at most `2^40` MMA tiles; it refuses every other call), and `A`'s and the words' honest roots
  registered, `A`'s before the coins. So `G` is conditional on both registered roots being the honest ones.
* `G_tail`: `G`'s tail form, under `G`'s premises: the share of the certifier's entropy values at whose coins a
  strategy's call is accepted, hashed no collision and worked, `betaLead` a bit exchanged included, below
  `32 · (honestRegs − gapFixed)` is at most `etaLead`. `G` sums work over the accepted values, so it can't tell a small
  loss on every value from a large one on a few; `G_tail` can. The fixed gap holds the head, the coin-branch mean and
  a slack of 0.08% of padded stock, and `p*`; the tail holds the tables' union bound and the coin-branch count's tail.
* `NetworkFwd`: the bits are bounded per forward, where the network certifier counts them. Under `SerialForwards`, an
  accepted forward's network term (`Forward.networkRegs`, the `β · egress` that `G` charges, over its calls) is at
  most the largest `β` times the forward's bits over 32. `NetworkLlama`: over Llama-3.2-1B's own calls, at most
  0.0363% of their served stock at cap 32 and 0.363% at cap 320. A forward's gap is its calls' gaps and its network
  term.
* `GapAll`: at every call the verifier admits, the gap is at most `gapBound`, a closed form, of served stock.
* `GapAt`: the gap's share of served stock at T 512 × 64, T 512 × 512 and T 8,192, for every call of `n ≥ 512` and
  `K ≥ 2,048`, the served shapes': at most 0.157%, 0.134% and 0.131%. The cap doesn't enter the gap.
* `GapAtDecode`: at decode (`m = 1`, committed as 16 rows), the gap is at most 3.79% of served stock.
* `GapFixedAt`: for every call of `n, K ≥ 4,096`, the tail's fixed gap is at most 0.127% of served stock at the three
  points and 2.04% at decode. `TailAt`: `etaLead ≤ 2^−34` there, for calls of at most `2^30` tiles.
* `Honest`: the hypotheses are jointly satisfiable at every call the verifier admits, at the deployed instance: some
  certifier running `certCode`, and at the deployed commitment (`l1wCommit`: A in 1 KiB segments, the forward's words
  in its calls' chunks) and the certifier's coin stream (`certStream`), the nominal W1 table and some set of
  strategies meeting every other premise, with an honest strategy accepted at every coin value. The deployed
  definitions are to be tied to the served code (difftest). Every call of at most `2^15` rows, `2^20` columns and an
  inner dimension from 1 to `2^16` is admitted (`Statement.admits_of_caps`), Llama-3.2-1B's included. `G_tail`'s
  premises are `G`'s, so `Honest` is its honest lemma too.
* `Nontrivial`: the conclusion fails for a verifier that accepts everything: at the three points the honest work
  exceeds the gap, and a strategy that does nothing does no work, exchanges nothing and hashes nothing.
* `NontrivialTail`: the same for `G_tail`: at the three points the honest work exceeds the fixed gap, and
  `etaLead < 1` at calls of at most `2^30` tiles.
-/

namespace Pouw.Guarantees.Lead

open Pouw.Lead Pouw.Assumptions.Lead Pouw.Dimension.H100

/-- **The lead's work guarantee.** For every price table `pr` meeting `H32`, hash `H`, call
`m × K × n` at `T` tokens (every call, decode included: `lead` commits it over whole MMA tiles), data `κ`, certifier's
coin stream `stream`, commitment format `C` with its form classifier `isA`, set `Real` of the strategies the node can
run, and network certifier `cert` run as `runC` with its entropy's decoding `coinsOf`, under the named premises
(`CertifierCoins` among them, and an exchanged bit replacing at most `betaLead`), every strategy `σ` the node's
processors run (`can`) and every set `O` of entropy values at whose released coins its call is accepted (a call the
verifier admits, at the honest roots) has
`Σ_{e ∈ O} (work(released e) + betaLead · egress(released e)) ≥ 32 · (|O| · honestRegs − |Ω| · gapRegs)`, `|Ω|` the
number of coin values, or at some `e ∈ O` its execution hashed a SHA-512 collision with the honest commitment's
messages. The exchanged bits are bounded per forward (`NetworkFwd`). -/
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
        ∑ e ∈ O, ((σ.work pr (released cert runC σ e) : ℚ) +
          betaLead (lead T m n K) C * (σ.exec (released cert runC σ e)).egress) ∨
      ∃ e ∈ O, Break (lead T m n K) H C κ σ (released cert runC σ e)

/-- **The lead's work guarantee, tail form.** Under `G`'s premises, for every strategy `σ` the node's processors run,
the share of the certifier's entropy values at whose released coins its call is accepted, its execution hashed no
SHA-512 collision with the honest commitment, and its priced W1 work plus `betaLead` a bit it exchanged is below
`32 · (honestRegs − gapFixed)` is at most `etaLead`. The fixed gap (`gapFixed`) is the head, the coin-branch mean and
slack, and `p*`; `etaLead` is the tables' union bound over tiles and steps and the coin-branch count's tail beyond its
slack. -/
def G_tail : Prop :=
  ∀ (pr : Prices100) (H : Hash.Msg → Hash.Digest) (T m n K : ℕ)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (stream : CoinStream (lead T m n K)) (C : Pouw.Lead.Commit (lead T m n K))
    (isA : Hash.Msg → Bool) (can Real : Strategy (lead T m n K) → Prop)
    (Device Ent : Type) [Fintype Ent] (cert : Device) (coinsOf : Ent ≃ Coins (lead T m n K))
    (runC : Device → Hash.Digest × Ent → Hash.Digest × Coins (lead T m n K)),
    CertifierCoins cert coinsOf runC → CommitL1w H C isA κ →
    H32 pr → W1 pr Real → TileGrain H C κ stream Real → OneInstructionMinimal Real → CardWithinAccelW1 can Real →
    ShaWrites isA Real → HonestHashRuled H C κ Real →
    OutsourceWithinEgressLead pr (betaLead (lead T m n K) C) H C κ Real →
    ∀ σ, can σ →
      Pouw.pr (fun e : Ent =>
          Accepted (lead T m n K) H C κ σ (released cert runC σ e) ∧
            ¬ Break (lead T m n K) H C κ σ (released cert runC σ e) ∧
            (σ.work pr (released cert runC σ e) : ℚ) +
                betaLead (lead T m n K) C * (σ.exec (released cert runC σ e)).egress <
              32 * (honestRegs (lead T m n K) C - gapFixed (lead T m n K) C)) ≤
        etaLead (lead T m n K)

/-- **A forward's network term**: for every forward the network certifier accepts (`Forward.Accepted`), under
`SerialForwards`, and every `B ≥ 0` at least each of its calls' `betaLead`, the network term `G` charges its calls
(`Forward.networkRegs`) is at most `B` times the forward's bits (`Forward.bits`) over 32. -/
def NetworkFwd : Prop :=
  ∀ (F : Forward) (B : ℚ), SerialForwards F → F.Accepted → 0 ≤ B → (∀ c ∈ F.calls, betaLead c.L c.C ≤ B) →
    F.networkRegs ≤ B * F.bits / 32

/-- A call's commitment at the lead's layout: its words chunks are `wordChunks`. -/
def ChunksOf (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) : Prop := C.chunks = wordChunks L

/-- **A forward of Llama-3.2-1B's calls**: for every forward the network certifier accepts, under `SerialForwards`,
whose every call is one of Llama-3.2-1B's linears (`llamaCalls`) at the lead or at cap 320 (`lead320`), at any number
of rows, with the lead's words chunks, the network term is at most `0.0363% · cap / 32` of each call's served stock,
summed: 0.0363% at cap 32 and 0.363% at cap 320. -/
def NetworkLlama : Prop :=
  ∀ F : Forward, SerialForwards F → F.Accepted →
    (∀ c ∈ F.calls, (∃ T m : ℕ, ∃ x ∈ llamaCalls, c.L = lead T m x.1 x.2 ∨ c.L = lead320 T m x.1 x.2) ∧
      ChunksOf c.L c.C) →
    F.networkRegs ≤ 363 / 1000000 * (F.calls.map fun c => (c.L.cap / 32 : ℚ) * servedRegs c.L).sum

/-- **The gap at every admitted call**: at every call of `m ≥ 1` rows and `n ≥ 1` columns the verifier admits
(`Layout.Admits`), the gap is at most `gapBound m n K` of served stock. -/
def GapAll : Prop :=
  ∀ T m n K : ℕ, 1 ≤ m → 1 ≤ n → (lead T m n K).Admits →
    ∀ C : Pouw.Lead.Commit (lead T m n K), ChunksOf _ C → gapServed _ C ≤ gapBound m n K

/-- **The gap at the three points**: for every call of `n ≥ 512` and `K ≥ 2,048`, the served shapes', the gap's
share of served stock is at most 0.157% at T 512 × 64, 0.134% at T 512 × 512 and 0.131% at T 8,192. -/
def GapAt : Prop :=
  ∀ n K : ℕ, 512 ≤ n → 2048 ≤ K →
    (∀ C : Pouw.Lead.Commit (pt512x64 n K), ChunksOf _ C → gapServed _ C ≤ 157 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (pt512x512 n K), ChunksOf _ C → gapServed _ C ≤ 134 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (pt8192 n K), ChunksOf _ C → gapServed _ C ≤ 131 / 100000)

/-- **Decode** (`m = 1`), which `G` covers padded to 16 rows: for every call of `n ≥ 512` and `K ≥ 2,048`, the gap is
at most 3.79% of served stock, the one served row's plain matmul. -/
def GapAtDecode : Prop :=
  ∀ T n K : ℕ, 512 ≤ n → 2048 ≤ K →
    ∀ C : Pouw.Lead.Commit (lead T 1 n K), ChunksOf _ C → gapServed _ C ≤ 379 / 10000

/-- **The tail's fixed gap at the three points and decode**: for every call of `n, K ≥ 4,096`, `gapFixed` is at most
0.127% of served stock at T 512 × 64, T 512 × 512 and T 8,192, and at most 2.04% at decode, where 16 rows are
committed for the one served. The cap doesn't enter the fixed gap. -/
def GapFixedAt : Prop :=
  ∀ T n K : ℕ, 4096 ≤ n → 4096 ≤ K →
    (∀ C : Pouw.Lead.Commit (pt512x64 n K), ChunksOf _ C → gapFixedServed _ C ≤ 127 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (pt512x512 n K), ChunksOf _ C → gapFixedServed _ C ≤ 127 / 100000) ∧
    (∀ C : Pouw.Lead.Commit (pt8192 n K), ChunksOf _ C → gapFixedServed _ C ≤ 127 / 100000) ∧
    ∀ C : Pouw.Lead.Commit (lead T 1 n K), ChunksOf _ C → gapFixedServed _ C ≤ 204 / 10000

/-- **The tail at the three points and decode**: for every call of `n, K ≥ 4,096` with at most `2^30` committed tiles
(Llama-3.2-1B's largest call has `2^29.2`), `etaLead ≤ 2^−34`. The tables' term is about `2^−35`, from the coin
stream's `2^40` tiles at `2^−105` each, and the coin-branch tail at most `2^−42`. The tail doesn't read the cap. -/
def TailAt : Prop :=
  ∀ T n K : ℕ, 4096 ≤ n → 4096 ≤ K →
    ∀ L ∈ [pt512x64 n K, pt512x512 n K, pt8192 n K, lead T 1 n K],
      L.rowBlocks * L.colBlocks * L.S ≤ 2 ^ 30 → etaLead L ≤ 1 / 2 ^ 34

/-- **The honest lemma**: at every call of the lead the verifier admits (`Layout.Admits`), the hypotheses are jointly
satisfiable at the deployed instance: some certifier meets `CertifierCoins`, and at the deployed commitment
(`l1wCommit`), its form classifier (`l1wIsA`) and the certifier's coin stream (`certStream`) the nominal W1 table and
some set of strategies meet every other premise of `G`, with an honest strategy accepted at every coin value. That is
for every A of at most `2^64` bytes, all commitment parameters whose column tile is a positive number of blocks, and
every forward of the call whose words leaves number at most `2^64` (`range-u64` domains refuse more). -/
def Honest : Prop :=
  ∀ (T m n K : ℕ) (H : Hash.Msg → Hash.Digest)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ), (lead T m n K).Admits →
    (∃ (Device Ent : Type) (cert : Device) (coinsOf : Ent ≃ Coins (lead T m n K))
      (runC : Device → Hash.Digest × Ent → Hash.Digest × Coins (lead T m n K)), CertifierCoins cert coinsOf runC) ∧
    ∀ (P : L1w) (F : CallForward) (A : Fin (lead T m n K).rows → Fin (lead T m n K).K → UInt8),
      0 < P.nb → m * K ≤ 2 ^ 64 → F.leaves P.nb (lead T m n K) ≤ 2 ^ 64 →
      ∃ (Real : Strategy (lead T m n K) → Prop) (σ : Strategy (lead T m n K)),
        CommitL1w H (l1wCommit _ P F H A) (l1wIsA _ P F H A) κ ∧ H32 h100Nominal ∧ W1 h100Nominal Real ∧
        TileGrain H (l1wCommit _ P F H A) κ (certStream _) Real ∧ OneInstructionMinimal Real ∧
        CardWithinAccelW1 Real Real ∧ ShaWrites (l1wIsA _ P F H A) Real ∧
        HonestHashRuled H (l1wCommit _ P F H A) κ Real ∧
        OutsourceWithinEgressLead h100Nominal (betaLead (lead T m n K) (l1wCommit _ P F H A)) H
          (l1wCommit _ P F H A) κ Real ∧
        Real σ ∧ ∀ ω, Accepted (lead T m n K) H (l1wCommit _ P F H A) κ σ ω

/-- **The nontrivial lemma**: `G`'s conclusion fails for a verifier that accepts everything. At the three points the
honest work exceeds the gap, and a strategy that does nothing does no priced work, exchanges nothing and hashes
nothing, so with every coin value accepted only a collision among the honest commitment's own messages would meet the
conclusion. -/
def Nontrivial : Prop :=
  ∀ n K : ℕ, 512 ≤ n → 2048 ≤ K →
    (∀ C : Pouw.Lead.Commit (pt512x64 n K), ChunksOf _ C → gapRegs _ C < honestRegs _ C) ∧
    (∀ C : Pouw.Lead.Commit (pt512x512 n K), ChunksOf _ C → gapRegs _ C < honestRegs _ C) ∧
    (∀ C : Pouw.Lead.Commit (pt8192 n K), ChunksOf _ C → gapRegs _ C < honestRegs _ C) ∧
    ∀ (L : Pouw.Lead.Layout) (pr : Prices100), ∃ σ : Strategy L,
      ∀ ω, σ.work pr ω = 0 ∧ (σ.exec ω).egress = 0 ∧ (σ.exec ω).hashed = []

/-- **The tail form's nontrivial lemma**: `G_tail`'s conclusion fails for a verifier that accepts everything. At the
three points the honest work exceeds the fixed gap, and at calls of at most `2^30` tiles `etaLead < 1`. So
`Nontrivial`'s strategy that does nothing, accepted at every coin value and below `32 · (honestRegs − gapFixed)` at
each, puts every entropy value in `G_tail`'s event, more than `etaLead`, unless the honest commitment's own messages
collide. -/
def NontrivialTail : Prop :=
  ∀ n K : ℕ, 4096 ≤ n → 4096 ≤ K → ∀ L ∈ [pt512x64 n K, pt512x512 n K, pt8192 n K],
    (∀ C : Pouw.Lead.Commit L, ChunksOf L C → gapFixed L C < honestRegs L C) ∧
      (L.rowBlocks * L.colBlocks * L.S ≤ 2 ^ 30 → etaLead L < 1)

end Pouw.Guarantees.Lead
