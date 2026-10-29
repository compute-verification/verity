import Definitions.Pouw.Guarantees.Lead
import Mathlib.Data.Fin.VecNotation

/-!
# The lead's layout with coin lanes on every step, and its network term (definitions and stubs to move)

The definitions the network piece's statements read, and the statements themselves. Each definition moves unchanged
into `Definitions/Pouw/Lead/` (`namespace Pouw.Lead`), where it replaces the definition of its name, and each
statement into `Definitions/Pouw/Guarantees/Lead.lean` (`namespace Pouw.Guarantees.Lead`):

* into `Layout.lean`, step 0 carries the three coin lanes too, the layout the kernel serves (lane coins on step 1):
  - `dataLanes`: 20 data lanes at step 0 and 28 at each later step, so `stepsFor K = 1 + ⌈(K − 20)/28⌉`;
  - `coinLanes`: the three coin lanes `Σ_ℓ A′[i, ℓ] · s_t[⌊i/tile⌋, ℓ, j]`, which `coinTerm` adds at every step, and
    `word`, which reads that `coinTerm`;
  - the served model's forward: `llamaShapes`, `leadMacs` and `llamaCall`, Llama-3.2-1B at `T` tokens in calls of `m`
    rows;
* into `Gap.lean`, the network term reads the bits `E` the certifier counts for the call:
  - `betaLead = n_max · W_mac`: `n_max = 256 / 3` MACs a coin bit and `W_mac = 32 · honestRegs / (m n K)`, the honest
    W1 work a MAC;
  - `networkGap L C E = betaLead L C · E / 32` registers, and `egressShare`, a call's stock share of the cap;
* into `Guarantees/Lead.lean`, the statements `Network.lean` proves:
  - `CellsRot`: every cell of the lead, step 0 included, does at most `256 / 3` MACs a bit of the message it is
    computed from;
  - `NetworkAt` and `NetworkAt320`: at the three points, a call whose egress is at most its stock share of the cap
    has a network term of at most 0.036% (cap 32) and 0.36% (cap 320) of its stock;
  - `NetworkForward`: over a forward of Llama-3.2-1B, egress summed within `cap · T` bits, the calls' network terms
    sum to at most a share of the forward's stock, 0.0362% at cap 32 and 0.362% at cap 320.
-/

namespace Pouw.SecurityProofs.Lead.Network

open Pouw.Lead Pouw.Guarantees.Lead

/-- Data lanes of step `t`: 20 at step 0, whose other lanes are the three coin lanes, a zero lane, the row tag against
`G` and `a₁` against the column code, and 28 at each later step, whose other lanes are the three coin lanes and the tag
lane. -/
def dataLanes (t : ℕ) : ℕ := if t = 0 then 20 else 28

/-- k32 steps for inner dimension `K`: step 0 holds 20 data lanes and each later step 28, `1 + ⌈(K − 20)/28⌉`. -/
def stepsFor (K : ℕ) : ℕ := 1 + (K - 20 + 27) / 28

/-- **The coin lanes** of row `i` at step `t`, column `j`: `Σ_ℓ A′[i, ℓ] · s_t[⌊i/tile⌋, ℓ, j]`, `-rot`'s weights
against its tile's three lane coins. -/
def coinLanes (L : Pouw.Lead.Layout) (ω : Coins L) (i : Fin L.m) (t : Fin L.S) (j : Fin L.n) : ℤ :=
  ∑ ℓ : Fin 3, rotWeight i ℓ * (if h : (i : ℕ) / L.tile < L.tiles then ω.s t ⟨(i : ℕ) / L.tile, h⟩ ℓ j else 0)

/-- **Step `t`'s coin term** in row `i`, column `j`: step 0's `T(i)·G[:, j] + a₁[i]·C[:, j]` or a later step's tag lane
`1 + b_t[i]`, and at every step the coin lanes. -/
def coinTerm (L : Pouw.Lead.Layout) (ω : Coins L) (i : Fin L.m) (t : Fin L.S) (j : Fin L.n) : ℤ :=
  (if (t : ℕ) = 0 then
    (∑ q : Fin 4, rowTag i q * ω.G q j) + ∑ q : Fin 4, ((ω.2.1 i q : ℕ) : ℤ) * colCode j q
  else 1 + ((ω.1 i t : ℕ) : ℤ)) + coinLanes L ω i t j

/-- **The committed word** of row `i`, step `t`, column `j`: `−β_j + Σ_{τ ≤ t} (κ i τ j + coinTerm i τ j)`, with `κ`
the data's step dot products, fixed before the coins. -/
def word (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) (i : Fin L.m) (t : Fin L.S)
    (j : Fin L.n) : ℤ :=
  -bias L.S j + ∑ τ : Fin L.S, if τ ≤ t then κ i τ j + coinTerm L ω i τ j else 0

/-- Llama-3.2-1B's four matmuls a layer, `(n, K)`: QKV, O, gate-up and down. -/
def llamaShapes : Fin 4 → ℕ × ℕ := ![(3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192)]

/-- Llama-3.2-1B's matmuls' MACs a token: 16 layers of its four, `Σ n K`. -/
def leadMacs : ℕ := 16 * ∑ q : Fin 4, (llamaShapes q).1 * (llamaShapes q).2

/-- **A forward of Llama-3.2-1B**: at `T` tokens in calls of `m` rows and an egress cap of `cap` bits a token, the call
of row block `c.1`, layer `c.2.1` and matmul `c.2.2`. -/
def llamaCall (cap T m : ℕ) (c : Fin (T / m) × Fin 16 × Fin 4) : Pouw.Lead.Layout :=
  { lead T m (llamaShapes c.2.2).1 (llamaShapes c.2.2).2 with cap := cap }

/-- `β = n_max · W_mac`: the priced W1 work an exchanged bit replaces at most, `n_max = 256 / 3` MACs a coin bit
(`CellsRot`) times the honest W1 work a MAC, `32 · honestRegs / (m n K)`. -/
def betaLead (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) : ℚ := 256 / 3 * (32 * honestRegs L C / (L.m * L.n * L.K))

/-- **The network term** of a call that exchanges `E` bits, in registers: `β E / 32`, at 32 W1 a register. -/
def networkGap (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) (E : ℚ) : ℚ := betaLead L C * E / 32

/-- **A call's stock share of the egress cap**: `cap · m n K / leadMacs` bits, a forward's `cap` bits a token spread
over its `leadMacs` MACs a token in proportion to the call's `m n K`. -/
def egressShare (L : Pouw.Lead.Layout) : ℚ := L.cap * (L.m * L.n * L.K) / leadMacs

/-- **`n_max ≤ 256 / 3` at every cell of the lead**: rows `R` of one 64-row coin tile at step `t` (step 0 included),
by the columns `J`, whose step-`t` words an outsourcer gets from a message of `b` bits at every coin value, do
`|R| · |J| · dataLanes t` MACs, at most `256 / 3` a bit of the message. -/
def CellsRot : Prop :=
  ∀ (L : Pouw.Lead.Layout), L.tile = 64 → L.e = 7 → ∀ (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (t : Fin L.S)
    (k : Fin L.tiles) (R : Finset (Fin L.m)), (∀ i ∈ R, (i : ℕ) / L.tile = k) → ∀ (J : Finset (Fin L.n)) (b : ℕ)
    (h : Coins L → Fin (2 ^ b)) (g : Fin (2 ^ b) → Fin L.m → Fin L.n → ℤ),
    (∀ ω, ∀ i ∈ R, ∀ j ∈ J, g (h ω) i j = word L κ ω i t j) →
    3 * (R.card * J.card * dataLanes t) ≤ 256 * b

/-- **The network term at the three points, cap 32**: for every call of `n, K ≥ 4,096` whose egress `E` is at most
its stock share of 32 bits a token, at most 0.036% of stock at T 512 × 64, T 512 × 512 and T 8,192. -/
def NetworkAt : Prop :=
  ∀ n K : ℕ, 4096 ≤ n → 4096 ≤ K →
    (∀ C : Pouw.Lead.Commit (pt512x64 n K), ChunksOf _ C → ∀ E : ℚ, E ≤ egressShare (pt512x64 n K) →
      networkGap _ C E ≤ 36 / 100000 * stockRegs (pt512x64 n K)) ∧
    (∀ C : Pouw.Lead.Commit (pt512x512 n K), ChunksOf _ C → ∀ E : ℚ, E ≤ egressShare (pt512x512 n K) →
      networkGap _ C E ≤ 36 / 100000 * stockRegs (pt512x512 n K)) ∧
    (∀ C : Pouw.Lead.Commit (pt8192 n K), ChunksOf _ C → ∀ E : ℚ, E ≤ egressShare (pt8192 n K) →
      networkGap _ C E ≤ 36 / 100000 * stockRegs (pt8192 n K))

/-- **The network term at the three points, cap 320**: at most 0.36% of stock. -/
def NetworkAt320 : Prop :=
  ∀ n K : ℕ, 4096 ≤ n → 4096 ≤ K →
    (∀ C : Pouw.Lead.Commit (lead320 512 64 n K), ChunksOf _ C → ∀ E : ℚ, E ≤ egressShare (lead320 512 64 n K) →
      networkGap _ C E ≤ 36 / 10000 * stockRegs (lead320 512 64 n K)) ∧
    (∀ C : Pouw.Lead.Commit (lead320 512 512 n K), ChunksOf _ C → ∀ E : ℚ, E ≤ egressShare (lead320 512 512 n K) →
      networkGap _ C E ≤ 36 / 10000 * stockRegs (lead320 512 512 n K)) ∧
    (∀ C : Pouw.Lead.Commit (lead320 8192 8192 n K), ChunksOf _ C → ∀ E : ℚ, E ≤ egressShare (lead320 8192 8192 n K) →
      networkGap _ C E ≤ 36 / 10000 * stockRegs (lead320 8192 8192 n K))

/-- **The network term over a forward of Llama-3.2-1B** at cap `cap`: at `T` tokens in calls of `m ∣ T` rows, the
calls' egress summed within `cap · T` bits, their network terms sum to at most `share` of the forward's stock. -/
def NetworkForward (cap : ℕ) (share : ℚ) : Prop :=
  ∀ T m : ℕ, 1 ≤ m → m ∣ T → ∀ C : ∀ c, Pouw.Lead.Commit (llamaCall cap T m c), (∀ c, ChunksOf _ (C c)) →
    ∀ E : Fin (T / m) × Fin 16 × Fin 4 → ℚ, (∀ c, 0 ≤ E c) → ∑ c, E c ≤ cap * T →
      ∑ c, networkGap _ (C c) (E c) ≤ share * ∑ c, stockRegs (llamaCall cap T m c)

end Pouw.SecurityProofs.Lead.Network
