import Proofs.Accounting.BoundReceipt.Audit

/-!
# A node with several memory tiers in the bound-receipt game

Under the actual-circuit model v2.9 a memory tier is a location whose links all go to the node's compute, so the
node's compute and all its tiers (each GPU's HBM, host DRAM) are one location of the bound-receipt game, its GPU, and
everything outside the node is the rest. As in the remote rate audit (`RateAuditRemote.lean`), the GPU's device memory
is one address space split into tiers by `tier : ℕ → Fin τ`, and its chip is everything on the node that computes.

The game's footprint is then the sum of the tiers' footprints (`Node.footprint_eq_sum`), and each tier's footprint is
at most the bits its port carried to the node's compute (`Chip.tierAddrs_card_le`), the remote statement's `Run.read`
at that tier. So `BoundReceiptLink` bounds the node's reads summed over its tiers (`TieredBoundReceiptLink`), the
remote statement's `Run.reads`.
-/

namespace Accounting.Rate.Chip

open Pous (Bits)

variable {N b h ρ : ℕ} {α : Type} {τ : ℕ}

/-- The bits read from tier `t` of device memory in a world: one per `ingress` step at an address of tier `t`. -/
def tierBits (tier : ℕ → Fin τ) (t : Fin τ) (w : World N b h ρ) : Chip N b h ρ α → ℕ
  | done _ => 0
  | tag x k => (k (w.oracle x)).tierBits tier t w
  | ingress a k => (k (w.mem a)).tierBits tier t w + if tier a = t then 1 else 0
  | side a k => (k (w.side a)).tierBits tier t w
  | stream i k => (k (w.stream i)).tierBits tier t w
  | rebuild j k => (k (w.store j)).tierBits tier t w

/-- The tiers' reads together are the device-memory ingress. -/
theorem sum_tierBits (tier : ℕ → Fin τ) (w : World N b h ρ) (c : Chip N b h ρ α) :
    ∑ t, c.tierBits tier t w = c.ingressBits w := by
  induction c with
  | done _ => simp [tierBits, ingressBits]
  | tag x k ih => simpa [tierBits, ingressBits] using ih (w.oracle x)
  | ingress a k ih => simp [tierBits, ingressBits, Finset.sum_add_distrib, ih]
  | side a k ih => simpa [tierBits, ingressBits] using ih (w.side a)
  | stream i k ih => simpa [tierBits, ingressBits] using ih (w.stream i)
  | rebuild j k ih => simpa [tierBits, ingressBits] using ih (w.store j)

/-- A tier's distinct addresses read are at most the bits read from it. -/
theorem tierAddrs_card_le (tier : ℕ → Fin τ) (t : Fin τ) (w : World N b h ρ) (c : Chip N b h ρ α) :
    ((c.addrs w).filter fun a => tier a = t).card ≤ c.tierBits tier t w := by
  induction c with
  | done _ => simp [addrs]
  | tag x k ih => simpa [addrs, tierBits] using ih (w.oracle x)
  | ingress a k ih =>
    simp only [addrs, tierBits, Finset.filter_insert]
    split_ifs with ht
    · exact (Finset.card_insert_le _ _).trans (Nat.add_le_add_right (ih (w.mem a)) 1)
    · exact (ih (w.mem a)).trans (Nat.le_add_right _ _)
  | side a k ih => simpa [addrs, tierBits] using ih (w.side a)
  | stream i k ih => simpa [addrs, tierBits] using ih (w.stream i)
  | rebuild j k ih => simpa [addrs, tierBits] using ih (w.store j)

end Accounting.Rate.Chip

namespace Accounting.BoundReceipt.Node

open Accounting.Rate
open Pous (Bits)

variable {N a b h S L q Lz qz τ : ℕ}

/-- The GPU's footprint in tier `t`: the distinct addresses of tier `t` it read in the window, in bits. -/
def tierFootprint (tier : ℕ → Fin τ) (t : Fin τ) (A : Node N a b h S L q Lz qz) (E : Store N b) (z : Store N a)
    (G : Mix N a b h) : ℕ :=
  ((((A.gpu z).chip ((A.gpu z).state E)).addrs ((A.gpu z).world E noStream (atZ G z))).filter
    fun x => tier x = t).card

/-- The bits tier `t`'s port carried to the node's compute in the window. -/
def tierReads (tier : ℕ → Fin τ) (t : Fin τ) (A : Node N a b h S L q Lz qz) (E : Store N b) (z : Store N a)
    (G : Mix N a b h) : ℕ :=
  ((A.gpu z).chip ((A.gpu z).state E)).tierBits tier t ((A.gpu z).world E noStream (atZ G z))

theorem footprint_eq_sum (tier : ℕ → Fin τ) (A : Node N a b h S L q Lz qz) (E : Store N b) (z : Store N a)
    (G : Mix N a b h) : A.footprint E z G = ∑ t, A.tierFootprint tier t E z G :=
  Finset.card_eq_sum_card_fiberwise fun x _ => Finset.mem_univ (tier x)

theorem footprint_le_sum_tierReads (tier : ℕ → Fin τ) (A : Node N a b h S L q Lz qz) (E : Store N b)
    (z : Store N a) (G : Mix N a b h) : A.footprint E z G ≤ ∑ t, A.tierReads tier t E z G := by
  rw [footprint_eq_sum tier]
  exact Finset.sum_le_sum fun t _ => Chip.tierAddrs_card_le tier t _ _

end Accounting.BoundReceipt.Node

namespace Accounting.SecurityProofs

open Accounting.Rate Accounting.BoundReceipt
open Pous (Bits pr)
open scoped ENNReal

/-- **The link relay, on a node with several memory tiers.** `BoundReceiptLink` with the GPU location's memory split
into tiers by `tier`: the probability that the verifier accepts, the GPU's compute is within `F`, the rest's tile work
within `F_z`, and the bits the tiers' ports carried to the node's compute, summed over the tiers, fall short of
`RateAudit`'s floor at `k − m` blocks, is at most `BoundReceiptLink`'s bound. -/
theorem TieredBoundReceiptLink {τ : ℕ} (tier : ℕ → Fin τ) (N a b h S L q Lz qz s k m F cTag cEnc w Fz lam : ℕ)
    (A : Node N a b h S L q Lz qz) (hm : (Lz + lam) * w + Fz * a < (m + 1) * a * w) :
    pr (fun x : Store N b × BoundReceipt.Coins N a b h s =>
        A.Accepts x.1 x.2.1 x.2.2.1 x.2.2.2 ∧ A.compute cTag cEnc x.1 x.2.1 x.2.2.1 ≤ F ∧
          A.tileWork w x.1 x.2.1 x.2.2.1 ≤ Fz ∧
          (∑ t, A.tierReads tier t x.1 x.2.1 x.2.2.1) * cEnc < floor b 0 S L F cTag cEnc (k - m) lam)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + 2 * (s : ℝ≥0∞) / 2 ^ h + (q.choose (k - m) : ℝ≥0∞) / 2 ^ lam +
        (qz.choose (m + 1) : ℝ≥0∞) / 2 ^ lam :=
  (Bandwidth.pr_mono fun _ ⟨h1, h2, h3, h4⟩ =>
    ⟨h1, h2, h3, lt_of_le_of_lt (Nat.mul_le_mul_right _ (Node.footprint_le_sum_tierReads tier A _ _ _)) h4⟩).trans
    (BoundReceiptLink N a b h S L q Lz qz s k m F cTag cEnc w Fz lam A hm)

end Accounting.SecurityProofs
