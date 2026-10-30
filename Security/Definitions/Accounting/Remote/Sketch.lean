import Definitions.Accounting.RateAuditSketch
import Definitions.Core.Period

/-!
# The receipts' sketch over a remote audit session: the seeds' keys and the passes' devices (trusted)

The model `Accounting.Remote.Held.Assumptions.SketchBound` is derived on (`Accounting.Guarantees.Remote`'s
`SketchBoundSession`). A remote audit session over a registered store of `N` blocks runs `P` credited passes. Pass `j`'s
seed (`seed_i` in `remote_bandwidth.receipts`) fixes its key: the `k_s × 4096` signs `S_i` it derives and the receipt
hash at that seed, `y = SHA512(Z(SKETCH_RECEIPT_DOMAIN ‖ frame(seed_i)) ‖ u64be(j) ‖ σ)` with `σ = S_i·e_j`, an `h`-bit
tag of the block's index and its sketch words (`RateSketch.sketch`). There is no certified stream (`ρ = 0`). The prover
may grind a pass's seed: pass `j` has `G j` candidate seeds, each with its own key, and the receipts it sends are one
candidate's (`kept`, any function of the outcome).

**The outcome** (`Outcome`) is the prover's own coins `β`, the registered store (`N` blocks of `m` 32-byte sectors)
and `Hs`, what SHA-512's outputs are read over, all uniform and independent. The session's keys are SHA-512's at its
candidate seeds (`keys`, any function of the outcome). That they are uniform, whatever the prover's coins and the
store, is the named premise `Accounting.Assumptions.SketchHashRandom` (`random-oracle`), not the coin type.

**The model's coins** (`ModelCoins`) are the prover's own, the store and every candidate's key: its signs, and the receipt
hash at its seed as a function (`RateSketch.Oracle`). `coins keys` reads them off an outcome.

**The session** (`Session`) is a play over the passes (`Verity.Play`), pass `j`'s coin its candidates' keys
(`PassKeys`) and its move a device per prover coin and candidate (`PassDevices`): pass `j`'s candidate `c` runs
`device b j η c`, chosen by the prover's coins `b` and the keys `η` of the earlier passes. The devices are
`RateSketch.Device`'s verbatim-sector devices (`rate-sketch-verbatim`). A device obtains the store's sectors only whole
and as they are: those it holds on chip when the pass opens and those it reads from node memory, takes in over a link,
or rebuilds (`Device.obtained`).
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

/-- Pass `j`'s coin: its `G j` candidate seeds' keys. -/
abbrev PassKeys (N m h ks : ℕ) {P : ℕ} (G : Fin P → ℕ) (j : Fin P) : Type := Fin (G j) → Key N m h ks

/-- Pass `j`'s move: each candidate's device, for each of the prover's coins. -/
abbrev PassDevices (β : Type) (N m h S L q ks : ℕ) {P : ℕ} (G : Fin P → ℕ) (j : Fin P) : Type :=
  β → Fin (G j) → RateSketch.Device N m 0 h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32))

/-- **A remote audit session**: a play over its `P` passes (`Verity.Play`), pass `j` drawing its candidates' keys and
moving its candidates' devices, chosen by the prover's coins and the earlier passes' keys. -/
abbrev Session (β : Type) (P N m h S L q ks : ℕ) (G : Fin P → ℕ) : Type :=
  Verity.Play P (PassKeys N m h ks G) (PassDevices β N m h S L q ks G)

variable {β Hs : Type} {P N m h S L q ks : ℕ} {G : Fin P → ℕ}

/-- The model's coins at an outcome: the prover's own, the store, and the keys `keys b D x` SHA-512 gives at the
candidate seeds, for the prover's coins `b`, the store `D` and SHA-512's outputs read at `x`. -/
def coins (keys : β → Data N m 0 Sector → Hs → Keys P N m h ks G) (ω : Outcome β Hs N m) : ModelCoins β P N m h ks G :=
  (ω.1, ω.2.1, keys ω.1 ω.2.1 ω.2.2)

namespace Session

/-- Pass `j`'s candidate `c` on the prover's coins `b` and the earlier passes' keys `η`: the play's move at `j`. -/
def device (A : Session β P N m h S L q ks G) (b : β) (j : Fin P)
    (η : (i : {i : Fin P // i < j}) → Fin (G i) → Key N m h ks) (c : Fin (G j)) :
    RateSketch.Device N m 0 h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)) :=
  A j η b c

/-- Pass `j`'s candidate `c` at the model's coins. -/
def devAt (A : Session β P N m h S L q ks G) (x : ModelCoins β P N m h ks G) (j : Fin P) (c : Fin (G j)) :
    RateSketch.Device N m 0 h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)) :=
  A.device x.1 j (fun i => x.2.2 i.1) c

/-- The blocks whose receipts pass `j`'s candidate `c` gets right at an outcome, at the keys `keys` gives: its tag is
the receipt hash of the block's index and sketch at the candidate's key. -/
def correctAt (A : Session β P N m h S L q ks G) (keys : β → Data N m 0 Sector → Hs → Keys P N m h ks G)
    (ω : Outcome β Hs N m) (j : Fin P) (c : Fin (G j)) : Finset (Fin N) :=
  (A.devAt (coins keys ω) j c).correct sketch ((coins keys ω).2.2 j c).1 ((coins keys ω).2.2 j c).2 ω.2.1

/-- Pass `j`'s candidate `c` obtained block `t` whole at an outcome, at the keys `keys` gives: every one of its
sectors. -/
def Whole (A : Session β P N m h S L q ks G) (keys : β → Data N m 0 Sector → Hs → Keys P N m h ks G)
    (ω : Outcome β Hs N m) (j : Fin P) (c : Fin (G j)) (t : Fin N) : Prop :=
  ∀ u, Sum.inl (t, u) ∈
    (A.devAt (coins keys ω) j c).obtained ((coins keys ω).2.2 j c).1 ((coins keys ω).2.2 j c).2 ω.2.1

end Session

end Accounting.Remote.Sketch
