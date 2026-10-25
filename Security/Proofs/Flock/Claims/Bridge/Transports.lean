import Proofs.Flock.Claims.Bridge.BindsColl
import Proofs.Flock.Claims.Sampling.Deployed

/-!
# `Transports` at the deployed commitment: where it fails

`Transports` asks, of every opening `q` of a registered commit string at a wire `φ g` of the unit, a claims opening at
gate `g` of `q`'s bit that passes `Cm` under the statement's roots. Choosing `tr` by choice, it is exactly the existence
of such a passing opening. At the deployed sampling commitment (`SamplingDeploy.commit c`, `deployOf_commit`: hm96
leaves of `sha512/row/v2` rows under the registered frame-v3 path check), nothing the game's registration fixes makes one
pass:

* `not_transports`: if no opening at `g` passes, `Transports` fails at every `tr`, as soon as some registered opening
  at `φ g` exists (an honest registrant's).
* `deploy_opens_of_at_none`: no opening passes at a position the roots don't place (`c.at_ roots g = none`), and
  `deploy_opens_of_root`: none passes at a placed root that is none of the verifier's own ports' roots. `Transports`
  reads `s.roots`, which neither the registration nor `SampledKSLive` ties to the registration's leaves.
* `deploy_opens_of_short`: none passes at a hash whose answers are never 64 bytes: the hm96 leaf refuses. `Transports`
  is stated at every `H`, while the registration's leaves are SHA-512's (`H512`).

So `Transports` at the deployed commitment needs `H = SHA-512` and a coupling of the statement's roots to the
registration: each registered leaf `regLeaf R p` sits in the frame-v3 tree under `s.roots` at the leaf index
`c.at_ s.roots g` gives, as a leaf of the same row bytes. The last holds only at `sha512/row/v2` ports: at a
`sha512/row/v1` port (C-Flock's default) or a `row-seg/v1` one the registered leaf hashes `rowPrefix w ‖ bytes`, the
deployed commitment's `rowPrefixV2 ℓ ‖ bits`, and one leaf of both is a SHA-512 collision.
-/

namespace FlockSoundness.Discharge.ZkLink.WrongSet

open Game Model Accounting Audit ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] {mPts : ℕ}
variable {A : Arith F K} {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type}
  {planZ : Finset (Fin n) → Reg → List (TabZK F mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length}
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  {vb : ValueBindingZ A P planZ tab decode Hc}

/-- **`Transports` fails where no opening passes**: a registered opening at a wire of the unit, and no claims opening at
its gate that passes `Cm` under the roots, refute `Transports` at every `tr`. -/
theorem not_transports {H : Claims.Msg → Claims.Msg} {Cm : Claims.Sampling.Commitment Bool}
    {s : Claims.Sampling.Stmt} {R : Reg} {u : Fin n} {φ : ℕ → Fin C.N}
    (tr : ℕ → vb.Opening → Claims.Sampling.Opening Bool) {g : ℕ} (hg : φ g ∈ P.io u) {q : vb.Opening}
    (hq : vb.com q = vb.cs R (vb.pos (φ g)))
    (hno : ∀ o : Claims.Sampling.Opening Bool, o.pos = g → Cm.Opens H s.roots o = false) :
    ¬ Transports A P planZ tab vb H Cm s R u φ tr := by
  intro h
  obtain ⟨hp, -, hO⟩ := h g q hg hq
  rw [hno _ hp] at hO
  exact Bool.false_ne_true hO

end FlockSoundness.Discharge.ZkLink.WrongSet

namespace SamplingDeploy

open Claims Claims.Sampling
open Properties.NetworkCertifier.Bytes (bs)

/-- **No opening passes at a position the roots don't place.** -/
theorem deploy_opens_of_at_none (c : Code) (H : Msg → Msg) (roots : List Msg) (o : Opening Bool)
    (h : c.at_ roots o.pos = none) : (commit c).Opens H roots o = false := by
  unfold Commitment.Opens commit pathCommit
  simp only [h]
  rfl

/-- **No opening passes at a root that is none of the verifier's own ports' roots.** -/
theorem deploy_opens_of_root (c : Code) (H : Msg → Msg) (roots : List Msg) (o : Opening Bool) {ri : Msg × ℕ}
    (h : c.at_ roots o.pos = some ri) (hr : c.own.find? (fun p => decide (bs p.root = ri.1)) = none) :
    (commit c).Opens H roots o = false := by
  unfold Commitment.Opens commit pathCommit
  simp only [h, ocBind_eval]
  split
  · simp only [regPath, hr]
    rfl
  · rfl

/-- **No opening passes at a hash whose answers are never 64 bytes**: the hm96 leaf refuses. -/
theorem deploy_opens_of_short (c : Code) (H : Msg → Msg) (hH : ∀ m, (H m).length ≠ 64) (roots : List Msg)
    (o : Opening Bool) : (commit c).Opens H roots o = false := by
  unfold Commitment.Opens commit pathCommit
  dsimp only
  split
  · rw [ocBind_eval, hm96Leaf_fst]
    split_ifs with h
    · exact absurd h.1 (hH _)
    · rfl
  · rfl

end SamplingDeploy

namespace FlockSoundness.Discharge.ZkLink.WrongSet

open Game Model Accounting Audit ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] {mPts : ℕ}
variable {A : Arith F K} {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type}
  {planZ : Finset (Fin n) → Reg → List (TabZK F mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length}
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  {vb : ValueBindingZ A P planZ tab decode Hc}

/-- **`Transports` is false at the deployed commitment under roots that don't place the gate**, at every hash and
`tr`, once the unit's wire `φ g` has a registered opening. -/
theorem not_transports_deploy_of_at_none (c : SamplingDeploy.Code) (H : Claims.Msg → Claims.Msg)
    {s : Claims.Sampling.Stmt} {R : Reg} {u : Fin n} {φ : ℕ → Fin C.N}
    (tr : ℕ → vb.Opening → Claims.Sampling.Opening Bool) {g : ℕ} (hg : φ g ∈ P.io u) {q : vb.Opening}
    (hq : vb.com q = vb.cs R (vb.pos (φ g))) (h : c.at_ s.roots g = none) :
    ¬ Transports A P planZ tab vb H (SamplingDeploy.commit c) s R u φ tr :=
  not_transports tr hg hq fun o ho => SamplingDeploy.deploy_opens_of_at_none c H s.roots o (ho ▸ h)

/-- **`Transports` is false at the deployed commitment at a hash whose answers are never 64 bytes**, at every
statement and `tr`, once the unit's wire `φ g` has a registered opening. -/
theorem not_transports_deploy_of_short (c : SamplingDeploy.Code) {H : Claims.Msg → Claims.Msg}
    (hH : ∀ m, (H m).length ≠ 64) {s : Claims.Sampling.Stmt} {R : Reg} {u : Fin n} {φ : ℕ → Fin C.N}
    (tr : ℕ → vb.Opening → Claims.Sampling.Opening Bool) {g : ℕ} (hg : φ g ∈ P.io u) {q : vb.Opening}
    (hq : vb.com q = vb.cs R (vb.pos (φ g))) :
    ¬ Transports A P planZ tab vb H (SamplingDeploy.commit c) s R u φ tr :=
  not_transports tr hg hq fun o _ => SamplingDeploy.deploy_opens_of_short c H hH s.roots o

end FlockSoundness.Discharge.ZkLink.WrongSet
