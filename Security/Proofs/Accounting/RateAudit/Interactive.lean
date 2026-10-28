import Mathlib
import Proofs.Accounting.Device
import Proofs.Accounting.RateAudit.Relay

/-!
# The rate audit with interactive side sources: the proof

An interactive device is a relay device (`InteractiveProver.toRelay`), as in the bandwidth audit
(`Proofs/Accounting/Bandwidth/Interactive.lean`). Its chip, translated (`IChip.toChip`), drops its sends, and where it
would read side address `a` having sent `v`, it reads the one address `addr v a`; the relay device's side source answers
there what the interactive source answers to `v` at `a` (`InteractiveProver.relaySide`), for the store, the stream and
the oracle. In every world the translation keeps the run, the ingress, the footprint, the link bits, the rebuilds and
the queries (`IChip.toChip_run` and the five after it), so the device's tags, footprint, compute and bounds are the
relay device's. The bounds are then `RelayDevice.audit`'s, `RateAudit`'s, when the answers ignore the oracle
(`RateAuditInteractiveRO`), and `RelayDevice.auditRelay`'s otherwise (`RateAuditInteractive`).

What the sends carry never enters the bounds, because the compression behind both encodes the bits the chip read, not
what its sources hold: a source addressed by everything sent may be as large as it likes.

The class is not vacuous: a `Prover` is an interactive prover that never sends (`Prover.toInteractive`), with the same
tags, ingress, footprint and compute, so every run that realizes `RateAudit`'s game realizes this one
(`RealizesRate.toInteractive`), with side sources that ignore the oracle; `Proofs/Accounting/Sanity.lean` applies it to
its honest rate run.
-/

namespace Accounting.Rate

open Pous (Bits pr)
open scoped ENNReal

namespace IChip

variable {N b h ρ : ℕ} {α : Type}

/-- A side read at address `a` after sending `v`, as one address. -/
def addr (v : List Bool) (a : ℕ) : ℕ := Encodable.encode (v, a)

/-- The interactive chip as a chip: a side read at `a` after sending `v` reads `addr v a`, and sends are dropped. -/
def toChip : List Bool → IChip N b h ρ α → Chip N b h ρ α
  | _, done a => Chip.done a
  | v, tag x k => Chip.tag x fun y => (k y).toChip v
  | v, ingress a k => Chip.ingress a fun β => (k β).toChip v
  | v, side a k => Chip.side (addr v a) fun β => (k β).toChip v
  | v, stream i k => Chip.stream i fun β => (k β).toChip v
  | v, rebuild j k => Chip.rebuild j fun x => (k x).toChip v
  | v, send β p => p.toChip (β :: v)

/-- The interactive world a chip's world is: a side read at `a` after sending `v` returns the side source at
`addr v a`. -/
def lift (w : World N b h ρ) : IWorld N b h ρ := ⟨w.oracle, w.mem, fun v a => w.side (addr v a), w.stream, w.store⟩

theorem toChip_run (w : World N b h ρ) (p : IChip N b h ρ α) (v : List Bool) :
    (p.toChip v).run w = p.run (lift w) v := by
  induction p generalizing v with
  | done a => rfl
  | tag x k ih => exact ih _ v
  | ingress a k ih => exact ih _ v
  | side a k ih => exact ih _ v
  | stream i k ih => exact ih _ v
  | rebuild j k ih => exact ih _ v
  | send β p ih => exact ih (β :: v)

theorem toChip_ingressBits (w : World N b h ρ) (p : IChip N b h ρ α) (v : List Bool) :
    (p.toChip v).ingressBits w = p.ingressBits (lift w) v := by
  induction p generalizing v with
  | done a => rfl
  | tag x k ih => exact ih _ v
  | ingress a k ih => exact congrArg (· + 1) (ih _ v)
  | side a k ih => exact ih _ v
  | stream i k ih => exact ih _ v
  | rebuild j k ih => exact ih _ v
  | send β p ih => exact ih (β :: v)

theorem toChip_addrs (w : World N b h ρ) (p : IChip N b h ρ α) (v : List Bool) :
    (p.toChip v).addrs w = p.addrs (lift w) v := by
  induction p generalizing v with
  | done a => rfl
  | tag x k ih => exact ih _ v
  | ingress a k ih => exact congrArg (insert a) (ih _ v)
  | side a k ih => exact ih _ v
  | stream i k ih => exact ih _ v
  | rebuild j k ih => exact ih _ v
  | send β p ih => exact ih (β :: v)

theorem toChip_linkBits (w : World N b h ρ) (p : IChip N b h ρ α) (v : List Bool) :
    (p.toChip v).linkBits w = p.linkBits (lift w) v := by
  induction p generalizing v with
  | done a => rfl
  | tag x k ih => exact ih _ v
  | ingress a k ih => exact ih _ v
  | side a k ih => exact congrArg (· + 1) (ih _ v)
  | stream i k ih => exact congrArg (· + 1) (ih _ v)
  | rebuild j k ih => exact ih _ v
  | send β p ih => exact ih (β :: v)

theorem toChip_rebuilds (w : World N b h ρ) (p : IChip N b h ρ α) (v : List Bool) :
    (p.toChip v).rebuilds w = p.rebuilds (lift w) v := by
  induction p generalizing v with
  | done a => rfl
  | tag x k ih => exact ih _ v
  | ingress a k ih => exact ih _ v
  | side a k ih => exact ih _ v
  | stream i k ih => exact ih _ v
  | rebuild j k ih => exact congrArg (· + 1) (ih _ v)
  | send β p ih => exact ih (β :: v)

theorem toChip_queries (w : World N b h ρ) (p : IChip N b h ρ α) (v : List Bool) :
    (p.toChip v).queries w = p.queries (lift w) v := by
  induction p generalizing v with
  | done a => rfl
  | tag x k ih => exact congrArg (x :: ·) (ih _ v)
  | ingress a k ih => exact ih _ v
  | side a k ih => exact ih _ v
  | stream i k ih => exact ih _ v
  | rebuild j k ih => exact ih _ v
  | send β p ih => exact ih (β :: v)

end IChip

namespace InteractiveProver

variable {N b h ρ S L q : ℕ}

/-- The relay device's side source: at `addr v a`, the answer to `v` at `a`; nothing at an address that is no such
pair. -/
def relaySide (A : InteractiveProver N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) (n : ℕ) : Bool :=
  match (Encodable.decode n : Option (List Bool × ℕ)) with
  | some p => A.side E R O p.1 p.2
  | none => false

/-- **An interactive device as a relay device**: the same state and device memory, the side source `relaySide`, and
the chip translated. Its bounds are the interactive device's, in the interactive world each chip world is. -/
def toRelay (A : InteractiveProver N b h ρ S L q) : RelayDevice N b h ρ S L q where
  state := A.state
  mem := A.mem
  side := A.relaySide
  chip st := (A.chip st).toChip []
  bounded st w := by
    rw [IChip.toChip_linkBits, IChip.toChip_queries]
    exact A.bounded st (IChip.lift w)

theorem lift_world (A : InteractiveProver N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    IChip.lift (A.toRelay.world E R O) = A.world E R O := by
  simp only [IChip.lift, RelayDevice.world, toRelay, world, IWorld.mk.injEq, true_and, and_true]
  funext v a
  simp [relaySide, IChip.addr, Encodable.encodek]

theorem toRelay_tags (A : InteractiveProver N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    A.toRelay.tags E R O = A.tags E R O := by
  change ((A.chip (A.state E)).toChip []).run (A.toRelay.world E R O) = _
  rw [IChip.toChip_run, lift_world]
  rfl

theorem toRelay_footprint (A : InteractiveProver N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    A.toRelay.footprint E R O = A.footprint E R O := by
  change (((A.chip (A.state E)).toChip []).addrs (A.toRelay.world E R O)).card = _
  rw [IChip.toChip_addrs, lift_world]
  rfl

theorem toRelay_compute (A : InteractiveProver N b h ρ S L q) (cTag cEnc : ℕ) (E : Store N b) (R : Bits ρ)
    (O : Oracle N b h ρ) : A.toRelay.compute cTag cEnc E R O = A.compute cTag cEnc E R O := by
  change (((A.chip (A.state E)).toChip []).queries (A.toRelay.world E R O)).length * cTag +
    ((A.chip (A.state E)).toChip []).rebuilds (A.toRelay.world E R O) * cEnc = _
  rw [IChip.toChip_queries, IChip.toChip_rebuilds, lift_world]
  rfl

theorem toRelay_accepts {s : ℕ} (A : InteractiveProver N b h ρ S L q) (E : Store N b) (R : Bits ρ)
    (O : Oracle N b h ρ) (I : Fin s → Fin N) : A.toRelay.Accepts E R O I ↔ A.Accepts E R O I := by
  simp only [RelayDevice.Accepts, Accepts, toRelay_tags]

/-- The footprint is at most the ingress: each address is read at least once. -/
theorem footprint_le_ingress (A : InteractiveProver N b h ρ S L q) (E : Store N b) (R : Bits ρ)
    (O : Oracle N b h ρ) : A.footprint E R O ≤ A.ingress E R O := by
  rw [← toRelay_footprint]
  have h := Chip.card_addrs_le (A.toRelay.world E R O) ((A.chip (A.state E)).toChip [])
  rw [IChip.toChip_ingressBits, lift_world] at h
  exact h

/-- When the answers ignore the oracle, so does the relay device's side source. -/
theorem toRelay_side (A : InteractiveProver N b h ρ S L q) (hA : ∀ E R O O', A.side E R O = A.side E R O') :
    ∀ E R O O', A.toRelay.side E R O = A.toRelay.side E R O' := by
  intro E R O O'
  funext n
  simp only [toRelay, relaySide, hA E R O O']

end InteractiveProver

/-! ## A device as an interactive device -/

/-- The chip world an interactive world is after the chip has sent `v`. -/
def IWorld.atSent {N b h ρ : ℕ} (w : IWorld N b h ρ) (v : List Bool) : World N b h ρ :=
  ⟨w.oracle, w.mem, w.side v, w.stream, w.store⟩

namespace Chip

variable {N b h ρ : ℕ} {α : Type}

/-- A chip as an interactive chip that never sends. -/
def toIChip : Chip N b h ρ α → IChip N b h ρ α
  | done a => IChip.done a
  | tag x k => IChip.tag x fun y => (k y).toIChip
  | ingress a k => IChip.ingress a fun β => (k β).toIChip
  | side a k => IChip.side a fun β => (k β).toIChip
  | stream i k => IChip.stream i fun β => (k β).toIChip
  | rebuild j k => IChip.rebuild j fun x => (k x).toIChip

theorem toIChip_run (c : Chip N b h ρ α) (w : IWorld N b h ρ) (v : List Bool) :
    c.toIChip.run w v = c.run (w.atSent v) := by
  induction c with
  | done a => rfl
  | tag x k ih => exact ih _
  | ingress a k ih => exact ih _
  | side a k ih => exact ih _
  | stream i k ih => exact ih _
  | rebuild j k ih => exact ih _

theorem toIChip_ingressBits (c : Chip N b h ρ α) (w : IWorld N b h ρ) (v : List Bool) :
    c.toIChip.ingressBits w v = c.ingressBits (w.atSent v) := by
  induction c with
  | done a => rfl
  | tag x k ih => exact ih _
  | ingress a k ih => exact congrArg (· + 1) (ih _)
  | side a k ih => exact ih _
  | stream i k ih => exact ih _
  | rebuild j k ih => exact ih _

theorem toIChip_addrs (c : Chip N b h ρ α) (w : IWorld N b h ρ) (v : List Bool) :
    c.toIChip.addrs w v = c.addrs (w.atSent v) := by
  induction c with
  | done a => rfl
  | tag x k ih => exact ih _
  | ingress a k ih => exact congrArg (insert a) (ih _)
  | side a k ih => exact ih _
  | stream i k ih => exact ih _
  | rebuild j k ih => exact ih _

theorem toIChip_linkBits (c : Chip N b h ρ α) (w : IWorld N b h ρ) (v : List Bool) :
    c.toIChip.linkBits w v = c.linkBits (w.atSent v) := by
  induction c with
  | done a => rfl
  | tag x k ih => exact ih _
  | ingress a k ih => exact ih _
  | side a k ih => exact congrArg (· + 1) (ih _)
  | stream i k ih => exact congrArg (· + 1) (ih _)
  | rebuild j k ih => exact ih _

theorem toIChip_rebuilds (c : Chip N b h ρ α) (w : IWorld N b h ρ) (v : List Bool) :
    c.toIChip.rebuilds w v = c.rebuilds (w.atSent v) := by
  induction c with
  | done a => rfl
  | tag x k ih => exact ih _
  | ingress a k ih => exact ih _
  | side a k ih => exact ih _
  | stream i k ih => exact ih _
  | rebuild j k ih => exact congrArg (· + 1) (ih _)

theorem toIChip_queries (c : Chip N b h ρ α) (w : IWorld N b h ρ) (v : List Bool) :
    c.toIChip.queries w v = c.queries (w.atSent v) := by
  induction c with
  | done a => rfl
  | tag x k ih => exact congrArg (x :: ·) (ih _)
  | ingress a k ih => exact ih _
  | side a k ih => exact ih _
  | stream i k ih => exact ih _
  | rebuild j k ih => exact ih _

end Chip

namespace Prover

variable {N b h ρ S L q : ℕ}

/-- A device as an interactive device that never sends: its side sources answer every read as the device's do. -/
def toInteractive (A : Prover N b h ρ S L q) : InteractiveProver N b h ρ S L q where
  state := A.state
  mem := A.mem
  side E _ _ _ := A.side E
  chip st := (A.chip st).toIChip
  bounded st w := by
    rw [Chip.toIChip_linkBits, Chip.toIChip_queries]
    exact A.bounded st (w.atSent [])

theorem toInteractive_tags (A : Prover N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    A.toInteractive.tags E R O = A.tags E R O :=
  Chip.toIChip_run _ _ _

theorem toInteractive_ingress (A : Prover N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    A.toInteractive.ingress E R O = A.ingress E R O :=
  Chip.toIChip_ingressBits _ _ _

theorem toInteractive_compute (A : Prover N b h ρ S L q) (cTag cEnc : ℕ) (E : Store N b) (R : Bits ρ)
    (O : Oracle N b h ρ) : A.toInteractive.compute cTag cEnc E R O = A.compute cTag cEnc E R O := by
  simp only [InteractiveProver.compute, Prover.compute, toInteractive, Chip.toIChip_queries, Chip.toIChip_rebuilds]
  rfl

theorem toInteractive_accepts {s : ℕ} (A : Prover N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ)
    (I : Fin s → Fin N) : A.toInteractive.Accepts E R O I ↔ A.Accepts E R O I := by
  simp only [InteractiveProver.Accepts, Prover.Accepts, toInteractive_tags]

end Prover

end Accounting.Rate

namespace Accounting

open Accounting.Rate

/-- A run that realizes `RateAudit`'s game realizes the game with interactive side sources, by devices that never send
and whose side sources ignore the oracle. -/
def RealizesRate.toInteractive {p : Price} {cTag cEnc N b h ρ s : ℕ} {ι : Type} [Fintype ι] {d : Device ι}
    {α ΩW ΩS : Type} [Fintype α] [DecidableEq α] {r : Run d α ΩW (RateCoins N b h ρ s) ΩS}
    (real : RealizesRate p cTag cEnc r) : RealizesRateInteractive p cTag cEnc r where
  dev z := (real.dev z).toInteractive
  accepts_iff x y s' := (real.accepts_iff x y s').trans (Rate.Prover.toInteractive_accepts _ _ _ _ _).symm
  ingress_le x y s' := (Rate.Prover.toInteractive_ingress _ _ _ _).trans_le (real.ingress_le x y s')
  compute_le x y s' := (Rate.Prover.toInteractive_compute _ _ _ _ _ _).trans_le (real.compute_le x y s')

theorem RealizesRate.toInteractive_side {p : Price} {cTag cEnc N b h ρ s : ℕ} {ι : Type} [Fintype ι] {d : Device ι}
    {α ΩW ΩS : Type} [Fintype α] [DecidableEq α] {r : Run d α ΩW (RateCoins N b h ρ s) ΩS}
    (real : RealizesRate p cTag cEnc r) :
    ∀ (x : ΩW) (s' : ΩS) E R O O', (real.toInteractive.dev (x, s')).side E R O =
      (real.toInteractive.dev (x, s')).side E R O' :=
  fun _ _ _ _ _ _ => rfl

end Accounting

namespace Accounting.SecurityProofs

open Accounting.Rate Accounting.Rate.InteractiveProver
open Pous (Bits pr)
open scoped ENNReal

/-- **The rate audit with interactive side sources that ignore the oracle**
(`Accounting.Guarantees.RateAuditInteractiveRO`). -/
theorem RateAuditInteractiveRO : Accounting.Guarantees.RateAuditInteractiveRO := by
  intro N b h ρ S L q s k F cTag cEnc lam A hA
  rw [Bandwidth.pr_coins_flat]
  dsimp only
  have hev : (fun x : Store N b × (Bits ρ × Oracle N b h ρ × (Fin s → Fin N)) =>
        A.Accepts x.1 x.2.1 x.2.2.1 x.2.2.2 ∧ A.compute cTag cEnc x.1 x.2.1 x.2.2.1 ≤ F ∧
          A.footprint x.1 x.2.1 x.2.2.1 * cEnc < floor b ρ S L F cTag cEnc k lam) =
      fun x => A.toRelay.Accepts x.1 x.2.1 x.2.2.1 x.2.2.2 ∧ A.toRelay.compute cTag cEnc x.1 x.2.1 x.2.2.1 ≤ F ∧
          A.toRelay.footprint x.1 x.2.1 x.2.2.1 * cEnc < floor b ρ S L F cTag cEnc k lam := by
    funext x
    simp only [toRelay_accepts, toRelay_compute, toRelay_footprint]
  rw [hev]
  exact A.toRelay.audit (toRelay_side A hA) s k F cTag cEnc lam

/-- **The rate audit with interactive side sources, in the random-oracle model**
(`Accounting.Guarantees.RateAuditInteractive`). -/
theorem RateAuditInteractive : Accounting.Guarantees.RateAuditInteractive := by
  intro N b h ρ S L q s k m F cTag cEnc lam A
  rw [Bandwidth.pr_coins_flat]
  dsimp only
  have hev : (fun x : Store N b × (Bits ρ × Oracle N b h ρ × (Fin s → Fin N)) =>
        A.Accepts x.1 x.2.1 x.2.2.1 x.2.2.2 ∧ A.compute cTag cEnc x.1 x.2.1 x.2.2.1 ≤ F ∧
          A.footprint x.1 x.2.1 x.2.2.1 * cEnc < floor b ρ S L F cTag cEnc (k - m) lam) =
      fun x => A.toRelay.Accepts x.1 x.2.1 x.2.2.1 x.2.2.2 ∧ A.toRelay.compute cTag cEnc x.1 x.2.1 x.2.2.1 ≤ F ∧
          A.toRelay.footprint x.1 x.2.1 x.2.2.1 * cEnc < floor b ρ S L F cTag cEnc (k - m) lam := by
    funext x
    simp only [toRelay_accepts, toRelay_compute, toRelay_footprint]
  rw [hev]
  exact A.toRelay.auditRelay s k m F cTag cEnc lam

/-- **The rate audit with interactive side sources as a joint run's bandwidth guarantee**
(`Accounting.Guarantees.RateOfAuditInteractive`). -/
theorem RateOfAuditInteractive : Accounting.Guarantees.RateOfAuditInteractive := by
  intro N b h ρ s k m cTag cEnc lam W p ι α ΩW ΩS _ _ _ d r real x s'
  refine (Bandwidth.pr_mono (fun y hy => ?_)).trans
    (RateAuditInteractive N b h ρ d.chip (∑ i, d.cap i * (r.win .bandwidth).len)
      (d.rate .oracle * (r.win .bandwidth).len) s k m (d.priced p * (r.win .compute).len - W) cTag cEnc lam
      (real.dev (x, s')))
  obtain ⟨hacc, hW, hread⟩ := hy
  refine ⟨(real.accepts_iff x y s').1 hacc, ?_, ?_⟩
  · have h1 := real.compute_le x y s'
    have h2 := r.work_add_othersWork p (x, y, s')
    omega
  · have h3 : (real.dev (x, s')).footprint y.1 y.2.1.1 y.2.1.2 * cEnc ≤ r.read (x, y, s') * cEnc :=
      Nat.mul_le_mul_right _ (((real.dev (x, s')).footprint_le_ingress _ _ _).trans (real.ingress_le x y s'))
    exact lt_of_le_of_lt h3 hread

/-- **The rate audit with interactive side sources that ignore the oracle, as a joint run's bandwidth guarantee**
(`Accounting.Guarantees.RateOfAuditInteractiveRO`). -/
theorem RateOfAuditInteractiveRO : Accounting.Guarantees.RateOfAuditInteractiveRO := by
  intro N b h ρ s k cTag cEnc lam W p ι α ΩW ΩS _ _ _ d r real hA x s'
  refine (Bandwidth.pr_mono (fun y hy => ?_)).trans
    (RateAuditInteractiveRO N b h ρ d.chip (∑ i, d.cap i * (r.win .bandwidth).len)
      (d.rate .oracle * (r.win .bandwidth).len) s k (d.priced p * (r.win .compute).len - W) cTag cEnc lam
      (real.dev (x, s')) (hA x s'))
  obtain ⟨hacc, hW, hread⟩ := hy
  refine ⟨(real.accepts_iff x y s').1 hacc, ?_, ?_⟩
  · have h1 := real.compute_le x y s'
    have h2 := r.work_add_othersWork p (x, y, s')
    omega
  · have h3 : (real.dev (x, s')).footprint y.1 y.2.1.1 y.2.1.2 * cEnc ≤ r.read (x, y, s') * cEnc :=
      Nat.mul_le_mul_right _ (((real.dev (x, s')).footprint_le_ingress _ _ _).trans (real.ingress_le x y s'))
    exact lt_of_le_of_lt h3 hread

end Accounting.SecurityProofs
