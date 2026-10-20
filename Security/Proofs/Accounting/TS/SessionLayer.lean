import Proofs.Flock.SoundAtBridge

/-!
# A window's proofs read through their C-Flock sessions

`Flock.Guarantees.SoundAtBridge` (#2909) takes four premises on the proof layer's side. Two of them say how the layer
reads its sessions, and a layer built from its sessions meets them by definition. `onSessions pf sess out rows` is `pf`
with:

* acceptance: proof `i` of round `w` is accepted only where flock-verify `--zk --registered` accepts its session's
  outcome (`LiveAcceptsZKR`), so `AcceptedLive` holds (`onSessions_acceptedLive`);
* the opening: it is the session's value layer (`XplurZC`, the plurality the rows extraction finds at the registration)
  read as a layer value, the only value a hiding leaf has, so `OpenedPlurAt` holds at `δo = 0`
  (`onSessions_openedPlur`).

Its relation and rightness are `pf`'s, so `Pins` and `RelOfRows` are `pf`'s (`onSessions_pins`,
`onSessions_relOfRows`). `onSessions_soundAt` is the bridge on it: `SoundAt δ` from `EndToEndRegisteredRows` (with
its named assumptions at every session, `RowsSession.CR`), the window's law (`WindowLaw`), the relation's reading of
the drawn units (`RelOfRows`) and the sessions' bound `n · bound ≤ δ`.

Binding (`ProofLayer.BindingAt`, `ProofLayer.Forked`) reads `opened`, so on this layer it is about the extracted values:
round `w`'s extracted leaves are the same on the continuations that redraw σ_w.
-/

namespace Accounting.SecurityProofs.TS.SessionLayer

open Accounting.TS Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

variable {Ω : Type} {S nπ : ℕ} {V : Type}

/-- **`pf` read through its sessions**: proof `i` of round `w` on the path `x` is the session `sess x w i` with
outcome `out x w i`. It is accepted when `pf` accepts it and flock-verify accepts the session's outcome; its opening is
the session's value layer read as a layer value (`rows`); its relation and rightness are `pf`'s. -/
noncomputable def onSessions (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) :
    ProofLayer Ω S nπ V where
  accepted x w i := pf.accepted x w i ∧ LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i)
  opened x w i := rows (sess x w i) (sess x w i).Xv
  rel := pf.rel
  right := pf.right

variable (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
  (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V)

/-- **`AcceptedLive` by definition.** -/
theorem onSessions_acceptedLive [Fintype Ω] : AcceptedLive (onSessions pf sess out rows) sess out :=
  fun _ _ _ h => h.2

/-- **`OpenedPlurAt` at `0` by definition.** -/
theorem onSessions_openedPlur [Fintype Ω] : OpenedPlurAt (onSessions pf sess out rows) sess rows 0 :=
  openedPlur_zero (onSessions pf sess out rows) sess rows fun _ _ _ _ => rfl

/-- The relations pin on the session layer exactly when they pin on `pf`. -/
theorem onSessions_pins : (onSessions pf sess out rows).Pins ↔ pf.Pins := Iff.rfl

/-- `RelOfRows` on the session layer is `RelOfRows` on `pf`. -/
theorem onSessions_relOfRows [Fintype Ω] :
    RelOfRows (onSessions pf sess out rows) sess out rows ↔ RelOfRows pf sess out rows :=
  Iff.rfl

/-- **The bridge on the session layer**: with `AcceptedLive` and `OpenedPlurAt` met by definition,
`EndToEndRegisteredRows` gives `SoundAt δ` from its named assumptions at every session, the window's law, `RelOfRows`
on `pf`, and `n · bound ≤ δ` at every session a path runs. -/
theorem onSessions_soundAt (hE : Flock.Guarantees.EndToEndRegisteredRows) [Fintype Ω] (δ : ℝ)
    (hCR : ∀ x w i, (sess x w i).CR) (hLaw : WindowLaw sess out) (hRel : RelOfRows pf sess out rows)
    (hδ : ∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) (hδ0 : 0 ≤ δ) :
    (onSessions pf sess out rows).SoundAt δ := by
  have h := soundAtBridge_of_rows hE (onSessions pf sess out rows) sess out rows 0 δ hCR hLaw
    (onSessions_acceptedLive pf sess out rows) (onSessions_openedPlur pf sess out rows) hRel hδ hδ0
  rwa [zero_add] at h

end Accounting.SecurityProofs.TS.SessionLayer
