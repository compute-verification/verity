import Definitions.Accounting.RateAuditInteractive
import Definitions.Accounting.RateAuditRemote
import Definitions.Accounting.RateAuditPooled

/-!
# The remote rate audit as deployed: one game (trusted)

The game the composed guarantee (`Definitions/Accounting/Guarantees/RateAuditDeployed.lean`) is about: a session of `n`
windows on one store, each window `P` chained passes seeded by Fiat–Shamir, which the prover may grind, closed by one
pooled draw of `d` openings. It puts together what the pieces state separately:

* **the chip** (`PIChip`) is `RateAuditInteractive.lean`'s, with sends and side sources that answer what it sent, and
  with `RateAuditPartial.lean`'s partial completion in place of the all-or-nothing rebuild: `complete j x g` obtains
  any `x ≤ b` bits `g (E_j)` of word `j`. A whole rebuild is `complete j b _ id`;
* **the device** (`PIDevice`) reads a store laid out over memory tiers (`RateAuditRemote.lean`): a read of any tier is a
  read of device memory, counted per tier by `PIChip.tierBits`. Its on-chip state is at most `S` bits, its side link
  carries at most `L` bits a pass (the certified stream's `ρ` among them), and it makes at most `q` oracle queries a
  pass, in every world;
* **the window** (`DeployedWindow`) is `RateAuditRemote.lean`'s `GrindChain` of such devices: pass `j`'s root is one
  of `G` candidates the earlier passes' coins and the store fix, picked as any function of pass `j`'s whole oracle
  family, and pass `j`'s device may depend on the pick;
* **the session** (`DeployedSession`) is `RateAuditPooled.lean`'s `Session` of such windows: window `w`'s prover is
  any function of the earlier windows' coins, their draws among them, and each window draws `d` of its `P·N`
  receipts after its last root (`WindowCoins`).

A receipt is the oracle's tag `H(σ, R, j, E_j)` of the whole word; the deployed receipt hashes a 16-row sketch of it,
which this game does not model (the PR's gap list). What the guarantee charges for each route to a word without
reading it is a named term of `γ` (`muChip`, `muLink`, `muRebuild`, `muLambda`), each a fraction of a pass's `N·b`
bits; the draw's shortfall is `d̄` over the monitoring period and `d_w` in a window.
-/

namespace Accounting.Rate

open Pous (Bits)

/-- A chip program in the deployed window: `IChip`'s steps, with `complete j x hx g k` in place of `rebuild`, which
obtains the `x ≤ b` bits `g (E_j)` of word `j` and continues with `k`. -/
inductive PIChip (N b h ρ : ℕ) (α : Type) : Type where
  | done : α → PIChip N b h ρ α
  | tag : Bits ρ × Fin N × Bits b → (Bits h → PIChip N b h ρ α) → PIChip N b h ρ α
  | ingress : ℕ → (Bool → PIChip N b h ρ α) → PIChip N b h ρ α
  | side : ℕ → (Bool → PIChip N b h ρ α) → PIChip N b h ρ α
  | stream : Fin ρ → (Bool → PIChip N b h ρ α) → PIChip N b h ρ α
  | complete : Fin N → (x : ℕ) → x ≤ b → (Bits b → Bits x) → (Bits x → PIChip N b h ρ α) → PIChip N b h ρ α
  | send : Bool → PIChip N b h ρ α → PIChip N b h ρ α

namespace PIChip

variable {N b h ρ : ℕ} {α : Type}

/-- The program's value in a world, having sent `v` so far. -/
def run (w : IWorld N b h ρ) : List Bool → PIChip N b h ρ α → α
  | _, done a => a
  | v, tag x k => (k (w.oracle x)).run w v
  | v, ingress a k => (k (w.mem a)).run w v
  | v, side a k => (k (w.side v a)).run w v
  | v, stream i k => (k (w.stream i)).run w v
  | v, complete j _ _ g k => (k (g (w.store j))).run w v
  | v, send β p => p.run w (β :: v)

/-- The bits read from tier `t` of device memory in a world, having sent `v` so far: one per `ingress` step at an
address of tier `t`. -/
def tierBits {τ : ℕ} (tier : ℕ → Fin τ) (t : Fin τ) (w : IWorld N b h ρ) : List Bool → PIChip N b h ρ α → ℕ
  | _, done _ => 0
  | v, tag x k => (k (w.oracle x)).tierBits tier t w v
  | v, ingress a k => (k (w.mem a)).tierBits tier t w v + if tier a = t then 1 else 0
  | v, side a k => (k (w.side v a)).tierBits tier t w v
  | v, stream i k => (k (w.stream i)).tierBits tier t w v
  | v, complete j _ _ g k => (k (g (w.store j))).tierBits tier t w v
  | v, send β p => p.tierBits tier t w (β :: v)

/-- The bits that came in over the side link in a world, having sent `v` so far: one per `side` step and one per
`stream` step. Sends are not counted. -/
def linkBits (w : IWorld N b h ρ) : List Bool → PIChip N b h ρ α → ℕ
  | _, done _ => 0
  | v, tag x k => (k (w.oracle x)).linkBits w v
  | v, ingress a k => (k (w.mem a)).linkBits w v
  | v, side a k => (k (w.side v a)).linkBits w v + 1
  | v, stream i k => (k (w.stream i)).linkBits w v + 1
  | v, complete j _ _ g k => (k (g (w.store j))).linkBits w v
  | v, send β p => p.linkBits w (β :: v)

/-- The bits completed in a world, having sent `v` so far: `x` per `complete` step of `x` bits. -/
def completed (w : IWorld N b h ρ) : List Bool → PIChip N b h ρ α → ℕ
  | _, done _ => 0
  | v, tag x k => (k (w.oracle x)).completed w v
  | v, ingress a k => (k (w.mem a)).completed w v
  | v, side a k => (k (w.side v a)).completed w v
  | v, stream i k => (k (w.stream i)).completed w v
  | v, complete j x _ g k => (k (g (w.store j))).completed w v + x
  | v, send β p => p.completed w (β :: v)

/-- The compute the completions cost in a world, having sent `v` so far, at `cost x` per `complete` step of `x`
bits. -/
def work (cost : ℕ → ℕ) (w : IWorld N b h ρ) : List Bool → PIChip N b h ρ α → ℕ
  | _, done _ => 0
  | v, tag x k => (k (w.oracle x)).work cost w v
  | v, ingress a k => (k (w.mem a)).work cost w v
  | v, side a k => (k (w.side v a)).work cost w v
  | v, stream i k => (k (w.stream i)).work cost w v
  | v, complete j x _ g k => (k (g (w.store j))).work cost w v + cost x
  | v, send β p => p.work cost w (β :: v)

/-- The oracle queries in a world, in order, having sent `v` so far. -/
def queries (w : IWorld N b h ρ) : List Bool → PIChip N b h ρ α → List (Bits ρ × Fin N × Bits b)
  | _, done _ => []
  | v, tag x k => x :: (k (w.oracle x)).queries w v
  | v, ingress a k => (k (w.mem a)).queries w v
  | v, side a k => (k (w.side v a)).queries w v
  | v, stream i k => (k (w.stream i)).queries w v
  | v, complete j _ _ g k => (k (g (w.store j))).queries w v
  | v, send β p => p.queries w (β :: v)

end PIChip

/-- A pass's device in the deployed window: `InteractiveDevice`'s on-chip state (`S` bits) and device memory laid out
from the store before the seed and the stream, side sources that answer as any function of the store, the certified
stream, the oracle and the bits sent so far, and a chip program with completions from the state. In every world,
starting from nothing sent, the program takes in at most `L` bits over the side link (side sources and the certified
stream together) and makes at most `q` oracle queries. -/
structure PIDevice (N b h ρ S L q : ℕ) where
  state : Payloads N b → Bits S
  mem : Payloads N b → ℕ → Bool
  side : Payloads N b → Bits ρ → Oracle N b h ρ → List Bool → ℕ → Bool
  chip : Bits S → PIChip N b h ρ (Fin N → Bits h)
  bounded : ∀ (st : Bits S) (w : IWorld N b h ρ), (chip st).linkBits w [] ≤ L ∧ ((chip st).queries w []).length ≤ q

namespace PIDevice

variable {N b h ρ S L q : ℕ}

/-- The world a device's chip runs in, for store `E`, certified stream `R` and oracle `O`. -/
def world (A : PIDevice N b h ρ S L q) (E : Payloads N b) (R : Bits ρ) (O : Oracle N b h ρ) : IWorld N b h ρ :=
  ⟨O, A.mem E, A.side E R O, R, E⟩

/-- The receipts the device commits to. -/
def tags (A : PIDevice N b h ρ S L q) (E : Payloads N b) (R : Bits ρ) (O : Oracle N b h ρ) : Fin N → Bits h :=
  (A.chip (A.state E)).run (A.world E R O) []

/-- The device's reads of tier `t` in the pass, in bits. -/
def tierReads {τ : ℕ} (A : PIDevice N b h ρ S L q) (tier : ℕ → Fin τ) (t : Fin τ) (E : Payloads N b) (R : Bits ρ)
    (O : Oracle N b h ρ) : ℕ :=
  (A.chip (A.state E)).tierBits tier t (A.world E R O) []

/-- The bits the device completed in the pass. -/
def completed (A : PIDevice N b h ρ S L q) (E : Payloads N b) (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  (A.chip (A.state E)).completed (A.world E R O) []

/-- The compute the device's completions cost in the pass, at `cost x` per completion of `x` bits. -/
def work (A : PIDevice N b h ρ S L q) (cost : ℕ → ℕ) (E : Payloads N b) (R : Bits ρ) (O : Oracle N b h ρ) : ℕ :=
  (A.chip (A.state E)).work cost (A.world E R O) []

/-- **A pass's completion work split across the node's concurrent compute devices.** `rate i` is device `i`'s
measured completion-work rate in the unit of `cost`, and `duration` is the pass's duration in the same time unit.
`workAt` assigns every unit of the pass device's completion work to exactly one device; each device gets at most its
rate times the pass duration.

The index covers every processor that can complete blocks, GPUs and host CPUs included. Thus the node's ceiling is
`(∑ i, rate i) * duration`, rather than a rate with time left implicit. -/
structure TimeBudget (A : PIDevice N b h ρ S L q) (cost : ℕ → ℕ) (k duration : ℕ) (rate : Fin k → ℕ) where
  workAt : Bits S → IWorld N b h ρ → Fin k → ℕ
  work_eq : ∀ st w, (A.chip st).work cost w [] = ∑ i, workAt st w i
  workAt_le : ∀ st w i, workAt st w i ≤ rate i * duration

/-- The words whose committed receipt is the oracle's under the stream, however the device came by it. -/
def correct (A : PIDevice N b h ρ S L q) (E : Payloads N b) (R : Bits ρ) (O : Oracle N b h ρ) : Finset (Fin N) :=
  Finset.univ.filter (fun j => A.tags E R O j = O (R, j, E j))

end PIDevice

/-- A deployed window: `P` chained passes whose roots the prover grinds against their Fiat–Shamir seeds
(`fiat-shamir`), as `GrindChain`, each pass's device a `PIDevice`. Pass `j`'s device and its `G` candidate roots are
any function of the earlier passes' coins (the candidates also of the store), and the kept candidate any function of
pass `j`'s whole oracle family. -/
structure DeployedWindow (P N b h ρ S L q r s G : ℕ) where
  device : (j : Fin P) → ({i : Fin P // i < j} → PassCoins N b h ρ r s) → Fin G → PIDevice N b h ρ S L q
  cand : (j : Fin P) → ({i : Fin P // i < j} → PassCoins N b h ρ r s) → Payloads N b → Fin G → Bits r
  pick : (j : Fin P) → ({i : Fin P // i < j} → PassCoins N b h ρ r s) → Payloads N b → (Bits r → Oracle N b h ρ) →
    Fin G

namespace DeployedWindow

variable {P N b h ρ S L q r s G : ℕ}

/-- The candidate pass `j` keeps at the window's coins `ω`. -/
def kept (A : DeployedWindow P N b h ρ S L q r s G) (j : Fin P) (E : Payloads N b) (ω : Fin P → PassCoins N b h ρ r s) :
    Fin G :=
  A.pick j (fun i => ω i.1) E (ω j).2.1

/-- Pass `j`'s device at the window's coins `ω`, at the candidate it kept. -/
def dev (A : DeployedWindow P N b h ρ S L q r s G) (j : Fin P) (E : Payloads N b) (ω : Fin P → PassCoins N b h ρ r s) :
    PIDevice N b h ρ S L q :=
  A.device j (fun i => ω i.1) (A.kept j E ω)

/-- Pass `j`'s oracle: its family at the kept candidate root, `H(σ_t, j, root_{j−1}, ·)`. -/
def oracle (A : DeployedWindow P N b h ρ S L q r s G) (j : Fin P) (E : Payloads N b)
    (ω : Fin P → PassCoins N b h ρ r s) : Oracle N b h ρ :=
  (ω j).2.1 (A.cand j (fun i => ω i.1) E (A.kept j E ω))

/-- The window's correct receipts, as the (pass, word) pairs whose opening its pooled draw passes. -/
def correctAll (A : DeployedWindow P N b h ρ S L q r s G) (E : Payloads N b) (ω : Fin P → PassCoins N b h ρ r s) :
    Finset (Fin P × Fin N) :=
  Finset.univ.filter (fun p => p.2 ∈ (A.dev p.1 E ω).correct E (ω p.1).1 (A.oracle p.1 E ω))

/-- The window's reads of every tier, in bits, summed over its passes. Completions are not reads: what they cost is
the compute `PIChip.work`, which the guarantee's assumptions turn into the term `muRebuild`. -/
def reads {τ : ℕ} (A : DeployedWindow P N b h ρ S L q r s G) (tier : ℕ → Fin τ) (E : Payloads N b)
    (ω : Fin P → PassCoins N b h ρ r s) : ℕ :=
  ∑ j, ∑ t, (A.dev j E ω).tierReads tier t E (ω j).1 (A.oracle j E ω)

end DeployedWindow

/-- A deployed session of `n` windows on one store, each closed by its pooled draw of `d` openings: window `w`'s prover
is any function of the earlier windows' coins, their draws among them. -/
structure DeployedSession (n P N b h ρ S L q r s d G : ℕ) where
  window : (w : Fin n) → ({v : Fin n // v < w} → WindowCoins P N b h ρ r s d) → DeployedWindow P N b h ρ S L q r s G

namespace DeployedSession

variable {n P N b h ρ S L q r s d G : ℕ}

/-- Window `w`'s prover at the session's coins `ω`: it reads the earlier windows' coins. -/
def win (A : DeployedSession n P N b h ρ S L q r s d G) (w : Fin n) (ω : Fin n → WindowCoins P N b h ρ r s d) :
    DeployedWindow P N b h ρ S L q r s G :=
  A.window w (fun v => ω v.1)

end DeployedSession

namespace Deployed

/-- The bits a pass can complete with compute `F`, when `x` bits cost at least `x/b` of a word's price
`rounds · c_root` (`Accounting.Assumptions.WorkHardness`): `b·F/(rounds·c_root)`. -/
def rebuildBits (b rounds cRoot F : ℕ) : ℕ := b * F / (rounds * cRoot)

/-- `μ_chip`: the on-chip state, `S` bits a pass, as a fraction of the pass's `N·b`. -/
noncomputable def muChip (N b S : ℕ) : ℝ := S / (N * b)

/-- `μ_link`: the side link's bits a pass beyond the certified stream, `L − ρ`, as a fraction of the pass's `N·b`. -/
noncomputable def muLink (N b ρ L : ℕ) : ℝ := ((L - ρ : ℕ) : ℝ) / (N * b)

/-- `μ_rebuild`: the bits a pass can complete within its squaring ceiling `F`, as a fraction of the pass's `N·b`. -/
noncomputable def muRebuild (N b rounds cRoot F : ℕ) : ℝ := (rebuildBits b rounds cRoot F : ℝ) / (N * b)

/-- `μ_λ`: the compression slack `λ` a pass, as a fraction of the pass's `N·b`. -/
noncomputable def muLambda (N b lam : ℕ) : ℝ := lam / (N * b)

/-- A pass's `γ` beside the draw: `μ_chip + μ_link + μ_rebuild + μ_λ`. -/
noncomputable def gammaPass (N b ρ S L lam rounds cRoot F : ℕ) : ℝ :=
  muChip N b S + muLink N b ρ L + muRebuild N b rounds cRoot F + muLambda N b lam

end Deployed

end Accounting.Rate
