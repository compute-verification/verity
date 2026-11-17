module

public import Definitions.Poub.RateAudit
public import Definitions.Poub.RateAuditInteractive

/-! The remote rate audit's additions: a store over the node's memory tiers, and chained passes whose roots the prover
may grind. -/

@[expose] public section

namespace ResourceAccounting.Rate

open Pous (Bits)

namespace IChip

variable {N b h ρ : ℕ} {α : Type}

/-- The bits read from tier `t` of device memory in a world, having sent `v` so far: one per `ingress` step at an
address of tier `t`. -/
def tierBits {τ : ℕ} (tier : ℕ → Fin τ) (t : Fin τ) (w : IWorld N b h ρ) : List Bool → IChip N b h ρ α → ℕ
  | _, done _ => 0
  | v, tag x k => (k (w.oracle x)).tierBits tier t w v
  | v, ingress a k => (k (w.mem a)).tierBits tier t w v + if tier a = t then 1 else 0
  | v, side a k => (k (w.side v a)).tierBits tier t w v
  | v, stream i k => (k (w.stream i)).tierBits tier t w v
  | v, rebuild j k => (k (w.store j)).tierBits tier t w v
  | v, send β p => p.tierBits tier t w (β :: v)

end IChip

namespace InteractiveProver

variable {N b h ρ S L q : ℕ}

/-- The device's reads of tier `t` in the window, in bits. -/
def tierReads {τ : ℕ} (A : InteractiveProver N b h ρ S L q) (tier : ℕ → Fin τ) (t : Fin τ) (E : Store N b)
    (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  (A.chip (A.state E)).tierBits tier t (A.world E R O) []

/-- The words rebuilt in the window, each at `c_enc` of compute. -/
def rebuilt (A : InteractiveProver N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  (A.chip (A.state E)).rebuilds (A.world E R O) []

/-- The words whose committed receipt is the oracle's under the stream, however the device came by it. -/
def correct (A : InteractiveProver N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    Finset (Fin N) :=
  Finset.univ.filter (fun j => A.tags E R O j = O (R, j, E j))

/-- Each tier's read capacity over the window bounds the chip's reads of it, from every state and in every world. -/
def TierCapped {τ : ℕ} (A : InteractiveProver N b h ρ S L q) (tier : ℕ → Fin τ) (cap : Fin τ → ℕ) : Prop :=
  ∀ (st : Bits S) (w : IWorld N b h ρ) (t : Fin τ), (A.chip st).tierBits tier t w [] ≤ cap t

end InteractiveProver

/-- A pass's coins in a chained window: its certified stream, its oracle family `H(σ_t, j, ·, ·)` at every root of `r`
bits, and its sample. -/
abbrev PassCoins (N b h ρ r s : ℕ) : Type := Bits ρ × (Bits r → Oracle N b h ρ) × (Fin s → Fin N)

/-- The pass coins are finite, as a product: an instance of its own, as for `Coins`. -/
instance instFintypePassCoins (N b h ρ r s : ℕ) : Fintype (PassCoins N b h ρ r s) :=
  inferInstanceAs (Fintype (_ × _ × _))

/-- A chained prover over `P` passes with `r`-bit roots: pass `j`'s device, an interactive device of `RateAudit`'s
class, and the root its seed is sampled at, each any function of the earlier passes' coins (the root also of the
store). -/
structure Chain (P N b h ρ S L q r s : ℕ) where
  device : (j : Fin P) → ({i : Fin P // i < j} → PassCoins N b h ρ r s) → InteractiveProver N b h ρ S L q
  root : (j : Fin P) → ({i : Fin P // i < j} → PassCoins N b h ρ r s) → Store N b → Bits r

namespace Chain

variable {P N b h ρ S L q r s : ℕ}

/-- Pass `j`'s device at the window's coins `ω`: it reads the earlier passes' coins. -/
def dev (A : Chain P N b h ρ S L q r s) (j : Fin P) (ω : Fin P → PassCoins N b h ρ r s) :
    InteractiveProver N b h ρ S L q :=
  A.device j (fun i => ω i.1)

/-- Pass `j`'s oracle: its family at the root its seed is sampled at, `H(σ_t, j, root_{j−1}, ·)`. -/
def oracle (A : Chain P N b h ρ S L q r s) (j : Fin P) (E : Store N b) (ω : Fin P → PassCoins N b h ρ r s) :
    Oracle N b h ρ :=
  (ω j).2.1 (A.root j (fun i => ω i.1) E)

end Chain

/-- A chained prover that grinds its roots: pass `j`'s root is one of `G` candidates, which the earlier passes' coins
and the store fix, picked as any function of pass `j`'s whole oracle family; pass `j`'s device may depend on the pick
too. -/
structure GrindChain (P N b h ρ S L q r s G : ℕ) where
  device : (j : Fin P) → ({i : Fin P // i < j} → PassCoins N b h ρ r s) → Fin G → InteractiveProver N b h ρ S L q
  cand : (j : Fin P) → ({i : Fin P // i < j} → PassCoins N b h ρ r s) → Store N b → Fin G → Bits r
  pick : (j : Fin P) → ({i : Fin P // i < j} → PassCoins N b h ρ r s) → Store N b → (Bits r → Oracle N b h ρ) →
    Fin G

namespace GrindChain

variable {P N b h ρ S L q r s G : ℕ}

/-- The candidate pass `j` keeps at the window's coins `ω`. -/
def kept (A : GrindChain P N b h ρ S L q r s G) (j : Fin P) (E : Store N b) (ω : Fin P → PassCoins N b h ρ r s) :
    Fin G :=
  A.pick j (fun i => ω i.1) E (ω j).2.1

/-- Pass `j`'s device at the window's coins `ω`, at the candidate it kept. -/
def dev (A : GrindChain P N b h ρ S L q r s G) (j : Fin P) (E : Store N b) (ω : Fin P → PassCoins N b h ρ r s) :
    InteractiveProver N b h ρ S L q :=
  A.device j (fun i => ω i.1) (A.kept j E ω)

/-- Pass `j`'s oracle: its family at the kept candidate root. -/
def oracle (A : GrindChain P N b h ρ S L q r s G) (j : Fin P) (E : Store N b) (ω : Fin P → PassCoins N b h ρ r s) :
    Oracle N b h ρ :=
  (ω j).2.1 (A.cand j (fun i => ω i.1) E (A.kept j E ω))

def atCand (A : GrindChain P N b h ρ S L q r s G) (c : Fin G) : Chain P N b h ρ S L q r s where
  device j η := A.device j η c
  root j η E := A.cand j η E c

end GrindChain

end ResourceAccounting.Rate
