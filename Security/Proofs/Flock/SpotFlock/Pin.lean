module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.Claims.Sampling.Path

/-!
# The circuit pin as round 2's driver checks it

Round 2's driver refuses a session unless its circuit file's SHA-512 is the committed `circuit_sha512`, the reference
statement's circuit file's (`proofs.verify`), while `pinOk` holds the circuit file to the reference's byte for byte.
`pinHash` is the driver's check: the session statement is the reference one in every field `pinOk` reads but the
circuit file, whose SHA-512 is the reference's. The two differ only at a SHA-512 collision, which `pinColl` outputs.

* `pinHash_of_pinOk`: a statement `pinOk` passes, `pinHash` passes.
* `pinOk_of_pinHash`: a statement `pinHash` passes whose circuit file exposes no collision (`pinColl = none`), `pinOk`
  passes.
* `pinColl_collides`: what `pinColl` outputs is a SHA-512 collision (`Claims.Collision sha512Msg`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling
open Properties.NetworkCertifier.Bytes (bs bs_inj)

variable (h : Held)

/-- **Template `t`'s session statement is its reference one, its circuit file held by SHA-512**, as round 2's driver
holds it: decoded as `w`, it is `xS₀ t`'s in every field but its sample, its registered reads, its public file and its
circuit file, and its circuit file's SHA-512 is the reference's. -/
def pinHash (t : ℕ) (w : FlockDeploy.Raw) : Bool :=
  match FlockDeploy.decode (h.xS₀ t) with
  | some w₀ =>
    decide ({ w with draw := none, registered := none, publicFile := .empty, circuit := .empty } =
        { w₀ with draw := none, registered := none, publicFile := .empty, circuit := .empty }) &&
      decide (_root_.Flock.Sha512.hash w.circuit = _root_.Flock.Sha512.hash w₀.circuit)
  | none => false

/-- **The collision template `t`'s session statement exposes**: the reference's circuit file and the session's, where
the two differ and have one SHA-512 digest. -/
def pinColl (t : ℕ) (w : FlockDeploy.Raw) : Option (Msg × Msg) :=
  match FlockDeploy.decode (h.xS₀ t) with
  | some w₀ =>
    if w.circuit ≠ w₀.circuit ∧ _root_.Flock.Sha512.hash w.circuit = _root_.Flock.Sha512.hash w₀.circuit then
      some (bs w₀.circuit, bs w.circuit)
    else none
  | none => none

variable {h}

theorem pinHash_of_pinOk {t : ℕ} {w : FlockDeploy.Raw} (hp : pinOk h t w = true) : pinHash h t w = true := by
  unfold pinOk at hp
  unfold pinHash
  cases hd : FlockDeploy.decode (h.xS₀ t) with
  | none => simp [hd] at hp
  | some w₀ =>
    simp only [hd, decide_eq_true_eq] at hp
    have hc : w.circuit = w₀.circuit := by simpa using congrArg FlockDeploy.Raw.circuit hp
    simp only [Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨congrArg (fun v : FlockDeploy.Raw => { v with circuit := .empty }) hp, by rw [hc]⟩

theorem pinOk_of_pinHash {t : ℕ} {w : FlockDeploy.Raw} (hp : pinHash h t w = true) (hn : pinColl h t w = none) :
    pinOk h t w = true := by
  unfold pinHash at hp
  unfold pinColl at hn
  unfold pinOk
  cases hd : FlockDeploy.decode (h.xS₀ t) with
  | none => simp [hd] at hp
  | some w₀ =>
    simp only [hd, Bool.and_eq_true, decide_eq_true_eq] at hp hn
    obtain ⟨e, hh⟩ := hp
    have hc : w.circuit = w₀.circuit := by
      by_contra hne
      simp [hne, hh] at hn
    simp only [decide_eq_true_eq]
    obtain ⟨⟩ := w
    obtain ⟨⟩ := w₀
    simp only [FlockDeploy.Raw.mk.injEq] at e ⊢
    simp_all

theorem pinColl_collides {t : ℕ} {w : FlockDeploy.Raw} {u v : Msg} (hp : pinColl h t w = some (u, v)) :
    Claims.Collision sha512Msg u v := by
  unfold pinColl at hp
  cases hd : FlockDeploy.decode (h.xS₀ t) with
  | none => simp [hd] at hp
  | some w₀ =>
    simp only [hd] at hp
    split at hp
    · rename_i hc
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hp)
      refine ⟨fun e => hc.1 (bs_inj e).symm, ?_⟩
      simp only [sha512Msg, bytes_bs, hc.2]
    · simp at hp

end SpotFlock
