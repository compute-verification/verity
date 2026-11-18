module

public import Proofs.Flock.Soundness.Assumptions.CustodyZk
public import Proofs.Flock.Soundness.Discharge.Composed.Sound
public import Proofs.Flock.Soundness.Discharge.Recursion.Defs
public import Proofs.Flock.Soundness.Discharge.Recursion.Flock
public import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedHJ
public import Proofs.Flock.Soundness.Registered.ReadsX
public import Proofs.Flock.Soundness.Registered.Defs
public import Proofs.Flock.Soundness.Binding.Leaf
import Proofs.Flock.Soundness.Game
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar

/-! # What V*'s bridge obligation reads of `Recursive.Registrants`, without its recursion imports. -/

@[expose] public section

namespace FlockSoundness.Registered

open Game Refine Binding Audit

section Clash

variable {C : Verity.Boolean.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Binding.Layout P plan tab decode} {rs : HmRowsX lay}
  {C' : Verity.Boolean.Circuit} {n' : ℕ} {P' : Audit.Partition C' n'} {Reg₁ : Type} {mPts' : ℕ}
  {plan' : Finset (Fin n') → Reg₁ → List (TabSpec mPts')}
  {tab' : (S : Finset (Fin n')) → (R : Reg₁) → Fin n' → Fin (plan' S R).length}
  {decode' : Audit.Partition.DecoderB (C := C') (Reg := Reg₁) plan' tab'}
  {lay' : Binding.Layout P' plan' tab' decode'} {rs' : HmRowsX lay'}

/-- **Two registrant records clash**: at registered reads `p` of `rd` and `p'` of `rd'`, the openings
(`TwoOpenings.pair`, under `p`'s root and position) or the hm96 leaves over the two commit strings' encodings
(`hm96Pair` of `rs.enc p` and `rs'.enc p'`) are a SHA-512 collision. It asks nothing of the two reads' roots or
positions, so it covers a collision however an obligation keys them (one `Registered.Port`, or one registration). -/
def ReadsX.Clash (rd : ReadsX lay rs) (rd' : ReadsX lay' rs') : Prop :=
  ∃ p p', rd.isRead p ∧ rd'.isRead p' ∧
    (Collides (⟨rd.port p, rd.readAt p, rd.leaf p, rd.wpath p, rd'.leaf p', rd'.wpath p'⟩ : TwoOpenings).pair ∨
      Collides (hm96Pair (rs.enc p (rd.wv p)) (rd.ws p) (rs'.enc p' (rd'.wv p')) (rd'.ws p')))

end Clash

end FlockSoundness.Registered

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg

open scoped ENNReal

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}

variable {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type}

/-- **Two of V*'s sessions clash at a history**: at the history `(R, ω, t)` and registration `R₂`, the registrant
records of some sessions `j`, `j'` clash (`ReadsX.Clash`). -/
def RegColl (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (R : Reg) (ω : L.Ω)
    (t : List (Cm × I.Coin)) (R₂ : Reg₂) : Prop :=
  ∃ j j', (zs R ω t R₂ j).rd.Clash (zs R ω t R₂ j').rd

/-- **`VBridge` off `Bad`**: at every history `(R, ω, t)` and registration `R₂` where `Bad` fails, and strategies `τ` of
V*'s sessions there, at every record that is a genuine session-game outcome (each session's outcome is one its game
reaches against `τ j`, `Game.Outcome`), when every session accepts and every layer is `Good`, the inner verifier
accepts the messages `wmsg` reads from the layers. -/
def VBridgeOff (Bad : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → Prop)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg) : Prop :=
  ∀ R ω t R₂, ¬ Bad R ω t R₂ → ∀ (τ : (j : Fin S) → Strategy ((zs R ω t R₂ j).game dg N))
    (o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript),
    (∀ j, Game.Outcome ((zs R ω t R₂ j).game dg N) (τ j) (o j)) →
    (∀ j, LiveAcceptsZKR (zs R ω t R₂ j).I (zs R ω t R₂ j).own (o j) ∧
      (zs R ω t R₂ j).Good ((zs R ω t R₂ j).layer dg N k Rw (τ j))) →
    I.accepts (x R ω) ((wmsg fun j => (zs R ω t R₂ j).layer dg N k Rw (τ j)).zip (t.map Prod.snd))

/-- **`VBridgeOff` at a `Bad` of the sessions' outcomes**: as `VBridgeOff`, with `Bad` reading the record `o` of
genuine session-game outcomes too, so asked only at the records where `Bad R ω t R₂ o` fails. -/
def VBridgeOffOut
    (Bad : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ →
      ((j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript) → Prop)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg) : Prop :=
  ∀ R ω t R₂ (τ : (j : Fin S) → Strategy ((zs R ω t R₂ j).game dg N))
    (o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript),
    (∀ j, Game.Outcome ((zs R ω t R₂ j).game dg N) (τ j) (o j)) → ¬ Bad R ω t R₂ o →
    (∀ j, LiveAcceptsZKR (zs R ω t R₂ j).I (zs R ω t R₂ j).own (o j) ∧
      (zs R ω t R₂ j).Good ((zs R ω t R₂ j).layer dg N k Rw (τ j))) →
    I.accepts (x R ω) ((wmsg fun j => (zs R ω t R₂ j).layer dg N k Rw (τ j)).zip (t.map Prod.snd))

end FlockSoundness.Discharge.Recursion
