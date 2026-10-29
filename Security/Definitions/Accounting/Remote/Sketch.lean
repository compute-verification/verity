import Definitions.Accounting.RateAuditSketch

/-!
# The receipts' sketch over a remote audit session: the seeds' keys and the passes' devices (trusted)

The model `Accounting.Remote.Held.Assumptions.SketchBound` is derived on (`Accounting.Guarantees.Remote`'s
`SketchBoundSession`). A remote audit session over a registered store of `N` blocks runs `P` credited passes. Pass `j`'s
seed (`seed_i` in `remote_bandwidth.receipts`) fixes its key: the `k_s × 4096` signs `S_i` it derives and the receipt
hash at that seed, `y = SHA512(Z(SKETCH_RECEIPT_DOMAIN ‖ frame(seed_i)) ‖ u64be(j) ‖ σ)` with `σ = S_i·e_j`, an `h`-bit
tag of the block's index and its sketch words (`RateSketch.sketch`). There is no certified stream (`ρ = 0`). The prover
may grind a pass's seed: pass `j` has `G j` candidate seeds, each with its own key, and the receipts it sends are one
candidate's (`kept`).

**The outcome** (`Outcome`) is the prover's own coins `β`, the registered store (`N` blocks of `m` 32-byte sectors)
and `Hs`, what SHA-512's outputs are read over, all uniform and independent. The session's keys are SHA-512's at its
candidate seeds (`Session.keys`, any function of the outcome). That they are uniform, whatever the prover's coins and
the store, is the named premise `Accounting.Assumptions.SketchHashRandom` (`random-oracle`), not the coin type.

**The model's coins** (`ModelCoins`) are the prover's own, the store and every candidate's key: its signs, and the receipt
hash at its seed as a function (`RateSketch.Oracle`). `Session.coins` reads them off an outcome.

**The devices** are `RateSketch.Device`'s verbatim-sector devices (`rate-sketch-verbatim`), one per candidate: pass
`j`'s candidate `c` runs `device b j η c`, chosen by the prover's coins `b` and the keys `η` of the earlier passes. A
device obtains the store's sectors only whole and as they are: those it holds on chip when the pass opens and those it
reads from node memory, takes in over a link, or rebuilds (`Device.obtained`).
-/

namespace Accounting.Remote.Sketch

open Accounting.RateSketch

/-- A seed's key: the signs it derives and the receipt hash at the seed. -/
abbrev Key (N m h ks : ℕ) : Type := SketchSigns ks m × Oracle N 0 h Sector (Fin ks → ZMod (2 ^ 32))

/-- The session's keys: pass `j`'s `G j` candidate seeds'. -/
abbrev Keys (P N m h ks : ℕ) (G : Fin P → ℕ) : Type := (j : Fin P) → Fin (G j) → Key N m h ks

/-- The model's coins: the prover's own, the registered store, and the keys. Not the live coins (`Verity.Coins`): they
hold no index draw, and the prover's own coins are among them. -/
abbrev ModelCoins (β : Type) (P N m h ks : ℕ) (G : Fin P → ℕ) : Type := β × Data N m 0 Sector × Keys P N m h ks G

/-- A session's outcome: the prover's own coins, the registered store, and `Hs`, what SHA-512's outputs are read
over. -/
abbrev Outcome (β Hs : Type) (N m : ℕ) : Type := β × Data N m 0 Sector × Hs

instance instFintypeKey (N m h ks : ℕ) : Fintype (Key N m h ks) := by
  letI : Fintype (Oracle N 0 h Sector (Fin ks → ZMod (2 ^ 32))) := inferInstance
  exact inferInstance

instance instFintypeKeys (P N m h ks : ℕ) (G : Fin P → ℕ) : Fintype (Keys P N m h ks G) := by
  letI : Fintype (Key N m h ks) := instFintypeKey N m h ks
  exact inferInstance

instance instFintypeModelCoins (β : Type) [Fintype β] (P N m h ks : ℕ) (G : Fin P → ℕ) :
    Fintype (ModelCoins β P N m h ks G) := by
  letI : Fintype (Keys P N m h ks G) := instFintypeKeys P N m h ks G
  exact inferInstance

/-- A remote audit session: pass `j`'s candidate `c` runs `device b j η c` on the prover's coins `b` and the earlier
passes' keys `η`; `keys b D x` is every candidate's key, SHA-512's at its seed, for the prover's coins `b`, the store
`D` and SHA-512's outputs read at `x`; and `kept ω j` is the candidate whose receipts pass `j` sends, any function of
the outcome. -/
structure Session (β Hs : Type) (P N m h S L q ks : ℕ) (G : Fin P → ℕ) where
  device : β → (j : Fin P) → ((i : {i : Fin P // i < j}) → Fin (G i) → Key N m h ks) →
    Fin (G j) → RateSketch.Device N m 0 h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32))
  keys : β → Data N m 0 Sector → Hs → Keys P N m h ks G
  kept : Outcome β Hs N m → (j : Fin P) → Fin (G j)

namespace Session

variable {β Hs : Type} {P N m h S L q ks : ℕ} {G : Fin P → ℕ}

/-- The model's coins at an outcome: the prover's own, the store, and the keys SHA-512 gives at the candidate
seeds. -/
def coins (A : Session β Hs P N m h S L q ks G) (ω : Outcome β Hs N m) : ModelCoins β P N m h ks G :=
  (ω.1, ω.2.1, A.keys ω.1 ω.2.1 ω.2.2)

/-- Pass `j`'s candidate `c` at the model's coins. -/
def devAt (A : Session β Hs P N m h S L q ks G) (x : ModelCoins β P N m h ks G) (j : Fin P) (c : Fin (G j)) :
    RateSketch.Device N m 0 h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)) :=
  A.device x.1 j (fun i => x.2.2 i.1) c

/-- The blocks whose receipts pass `j`'s candidate `c` gets right at an outcome: its tag is the receipt hash of the
block's index and sketch at the candidate's key. -/
def correctAt (A : Session β Hs P N m h S L q ks G) (ω : Outcome β Hs N m) (j : Fin P) (c : Fin (G j)) :
    Finset (Fin N) :=
  (A.devAt (A.coins ω) j c).correct sketch ((A.coins ω).2.2 j c).1 ((A.coins ω).2.2 j c).2 ω.2.1

/-- Pass `j`'s candidate `c` obtained block `t` whole at an outcome: every one of its sectors. -/
def Whole (A : Session β Hs P N m h S L q ks G) (ω : Outcome β Hs N m) (j : Fin P) (c : Fin (G j))
    (t : Fin N) : Prop :=
  ∀ u, Sum.inl (t, u) ∈
    (A.devAt (A.coins ω) j c).obtained ((A.coins ω).2.2 j c).1 ((A.coins ω).2.2 j c).2 ω.2.1

end Session

end Accounting.Remote.Sketch
