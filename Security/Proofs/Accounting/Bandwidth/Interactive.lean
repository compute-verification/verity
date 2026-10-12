import Mathlib
import Proofs.Accounting.Bandwidth.Relay

/-!
# The bandwidth audit v0 with interactive side sources: the proof

An interactive device is a relay device. Its chip, translated (`IChip.toChip`), drops its sends, and where it would
read side address `a` having sent `v`, it reads the one address `addr v a`. The relay device's side source answers there
what the interactive source answers to `v` at `a` (`InteractiveDevice.relaySide`). In every world the translation
keeps the run, the ingress, the side bits, the regenerations and the queries (`IChip.toChip_run` and the four after
it), so the device's tags, ingress and bounds are the relay device's. The bound is then `AuditRelay`'s
(`AuditInteractive`), and `AuditRO`'s when the answers ignore the oracle (`AuditInteractiveRO`), through the relay
device at any one oracle (`RelayDevice.atOracle`).

What the sends carry never enters the bound, because the compression behind both bounds encodes the bits the chip
read, not what its sources hold: a source addressed by everything sent may be as large as it likes.
-/

namespace Bandwidth

open Pous (Bits pr)
open scoped ENNReal

namespace IChip

variable {N b h : ℕ} {α : Type}

/-- A side read at address `a` after sending `v`, as one address. -/
def addr (v : List Bool) (a : ℕ) : ℕ := Encodable.encode (v, a)

/-- The interactive chip as a chip: a side read at `a` after sending `v` reads `addr v a`, and sends are dropped. -/
def toChip : List Bool → IChip N b h α → Chip N b h α
  | _, done a => Chip.done a
  | v, tag x k => Chip.tag x fun y => (k y).toChip v
  | v, ingress a k => Chip.ingress a fun β => (k β).toChip v
  | v, side a k => Chip.side (addr v a) fun β => (k β).toChip v
  | v, regen j k => Chip.regen j fun x => (k x).toChip v
  | v, send β p => p.toChip (β :: v)

/-- The interactive world a chip's world is: a side read at `a` after sending `v` returns the side source at
`addr v a`. -/
def lift (w : World N b h) : IWorld N b h := ⟨w.oracle, w.mem, fun v a => w.side (addr v a), w.block⟩

theorem toChip_run (w : World N b h) (p : IChip N b h α) (v : List Bool) :
    (p.toChip v).run w = p.run (lift w) v := by
  induction p generalizing v with
  | done a => rfl
  | tag x k ih => exact ih _ v
  | ingress a k ih => exact ih _ v
  | side a k ih => exact ih _ v
  | regen j k ih => exact ih _ v
  | send β p ih => exact ih (β :: v)

theorem toChip_ingressBits (w : World N b h) (p : IChip N b h α) (v : List Bool) :
    (p.toChip v).ingressBits w = p.ingressBits (lift w) v := by
  induction p generalizing v with
  | done a => rfl
  | tag x k ih => exact ih _ v
  | ingress a k ih => exact congrArg (· + 1) (ih _ v)
  | side a k ih => exact ih _ v
  | regen j k ih => exact ih _ v
  | send β p ih => exact ih (β :: v)

theorem toChip_sideBits (w : World N b h) (p : IChip N b h α) (v : List Bool) :
    (p.toChip v).sideBits w = p.sideBits (lift w) v := by
  induction p generalizing v with
  | done a => rfl
  | tag x k ih => exact ih _ v
  | ingress a k ih => exact ih _ v
  | side a k ih => exact congrArg (· + 1) (ih _ v)
  | regen j k ih => exact ih _ v
  | send β p ih => exact ih (β :: v)

theorem toChip_regens (w : World N b h) (p : IChip N b h α) (v : List Bool) :
    (p.toChip v).regens w = p.regens (lift w) v := by
  induction p generalizing v with
  | done a => rfl
  | tag x k ih => exact ih _ v
  | ingress a k ih => exact ih _ v
  | side a k ih => exact ih _ v
  | regen j k ih => exact congrArg (· + 1) (ih _ v)
  | send β p ih => exact ih (β :: v)

theorem toChip_queries (w : World N b h) (p : IChip N b h α) (v : List Bool) :
    (p.toChip v).queries w = p.queries (lift w) v := by
  induction p generalizing v with
  | done a => rfl
  | tag x k ih => exact congrArg (x :: ·) (ih _ v)
  | ingress a k ih => exact ih _ v
  | side a k ih => exact ih _ v
  | regen j k ih => exact ih _ v
  | send β p ih => exact ih (β :: v)

end IChip

namespace InteractiveDevice

variable {N b h S c G q : ℕ}

/-- The relay device's side source: at `addr v a`, the answer to `v` at `a`; nothing at an address that is no such
pair. -/
def relaySide (A : InteractiveDevice N b h S c G q) (E : Store N b) (O : Oracle N b h) (n : ℕ) : Bool :=
  match (Encodable.decode n : Option (List Bool × ℕ)) with
  | some p => A.side E O p.1 p.2
  | none => false

/-- **An interactive device as a relay device**: the same state and device memory, the side source `relaySide`, and
the chip translated. Its bounds are the interactive device's, in the interactive world each chip world is. -/
def toRelay (A : InteractiveDevice N b h S c G q) : RelayDevice N b h S c G q where
  state := A.state
  mem := A.mem
  side := A.relaySide
  chip st := (A.chip st).toChip []
  bounded st w := by
    rw [IChip.toChip_sideBits, IChip.toChip_regens, IChip.toChip_queries]
    exact A.bounded st (IChip.lift w)

theorem lift_world (A : InteractiveDevice N b h S c G q) (E : Store N b) (O : Oracle N b h) :
    IChip.lift (A.toRelay.world E O) = A.world E O := by
  simp only [IChip.lift, RelayDevice.world, toRelay, world, IWorld.mk.injEq, true_and, and_true]
  funext v a
  simp [relaySide, IChip.addr, Encodable.encodek]

theorem toRelay_tags (A : InteractiveDevice N b h S c G q) (E : Store N b) (O : Oracle N b h) :
    A.toRelay.tags E O = A.tags E O := by
  change ((A.chip (A.state E)).toChip []).run (A.toRelay.world E O) = _
  rw [IChip.toChip_run, lift_world]
  rfl

theorem toRelay_ingress (A : InteractiveDevice N b h S c G q) (E : Store N b) (O : Oracle N b h) :
    A.toRelay.ingress E O = A.ingress E O := by
  change ((A.chip (A.state E)).toChip []).ingressBits (A.toRelay.world E O) = _
  rw [IChip.toChip_ingressBits, lift_world]
  rfl

theorem toRelay_accepts {s : ℕ} (A : InteractiveDevice N b h S c G q) (E : Store N b) (O : Oracle N b h)
    (I : Fin s → Fin N) : A.toRelay.Accepts E O I ↔ A.Accepts E O I := by
  simp only [RelayDevice.Accepts, Accepts, toRelay_tags]

/-- When the answers ignore the oracle, the relay device at any one oracle runs, at every oracle, as the relay device
does. -/
theorem atOracle_world (A : InteractiveDevice N b h S c G q) (hA : ∀ E O O', A.side E O = A.side E O')
    (O₀ : Oracle N b h) (E : Store N b) (O : Oracle N b h) :
    (A.toRelay.atOracle O₀).world E O = A.toRelay.world E O := by
  simp only [Device.world, RelayDevice.world, RelayDevice.atOracle, World.mk.injEq, true_and, and_true]
  funext n
  simp only [toRelay, relaySide, hA E O₀ O]

theorem atOracle_tags (A : InteractiveDevice N b h S c G q) (hA : ∀ E O O', A.side E O = A.side E O')
    (O₀ : Oracle N b h) (E : Store N b) (O : Oracle N b h) :
    (A.toRelay.atOracle O₀).tags E O = A.tags E O := by
  rw [← toRelay_tags]
  change (A.toRelay.chip (A.toRelay.state E)).run ((A.toRelay.atOracle O₀).world E O) = _
  rw [atOracle_world A hA]
  rfl

theorem atOracle_ingress (A : InteractiveDevice N b h S c G q) (hA : ∀ E O O', A.side E O = A.side E O')
    (O₀ : Oracle N b h) (E : Store N b) (O : Oracle N b h) :
    (A.toRelay.atOracle O₀).ingress E O = A.ingress E O := by
  rw [← toRelay_ingress]
  change (A.toRelay.chip (A.toRelay.state E)).ingressBits ((A.toRelay.atOracle O₀).world E O) = _
  rw [atOracle_world A hA]
  rfl

end InteractiveDevice

namespace SecurityProofs

open InteractiveDevice

/-- **The bandwidth audit v0 with interactive side sources** (`Bandwidth.Guarantees.AuditInteractive`). -/
theorem AuditInteractive : Bandwidth.Guarantees.AuditInteractive := by
  intro N b h S c G q s k m lam A
  have hev : (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        A.Accepts x.1 x.2.1 x.2.2 ∧ A.ingress x.1 x.2.1 + S + c + m * b + G * b + lam < k * b) =
      fun x => A.toRelay.Accepts x.1 x.2.1 x.2.2 ∧ A.toRelay.ingress x.1 x.2.1 + S + c + m * b + G * b + lam < k * b := by
    funext x
    rw [toRelay_accepts, toRelay_ingress]
  rw [hev]
  exact AuditRelay N b h S c G q s k m lam A.toRelay

/-- **The bandwidth audit v0 with interactive side sources that ignore the oracle**
(`Bandwidth.Guarantees.AuditInteractiveRO`). -/
theorem AuditInteractiveRO : Bandwidth.Guarantees.AuditInteractiveRO := by
  intro N b h S c G q s k lam A hA
  let O₀ : Oracle N b h := fun _ _ => false
  have hev : (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        A.Accepts x.1 x.2.1 x.2.2 ∧ A.ingress x.1 x.2.1 + S + c + G * b + lam < k * b) =
      fun x => Accepts (A.toRelay.atOracle O₀) x.1 x.2.1 x.2.2 ∧
        (A.toRelay.atOracle O₀).ingress x.1 x.2.1 + S + c + G * b + lam < k * b := by
    funext x
    simp only [Bandwidth.Accepts, InteractiveDevice.Accepts, atOracle_tags A hA, atOracle_ingress A hA]
  rw [hev]
  exact AuditRO N b h S c G q s k lam (A.toRelay.atOracle O₀)

end SecurityProofs

end Bandwidth
