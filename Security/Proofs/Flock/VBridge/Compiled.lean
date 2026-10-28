import Proofs.Flock.Recursive.Target
import Proofs.Flock.Soundness.Assumptions.Recursive
import Proofs.Flock.VBridge.Algebra.Ligerito
import Proofs.Flock.VBridge.Algebra.Zerocheck
import Proofs.Flock.VBridge.Algebra.Lincheck

/-!
# VBridge G: `Flock.Assumptions.VBridge` at the compiled inner protocol

`vbridge_compiled` concludes `Flock.Assumptions.VBridge` with the inner protocol C-Flock's compiled table
(`flockInnerCL`: round messages are the level-0 cap, each round's proof messages and Merkle tops with the prover's
choices of rep, then, in the round after the last coin, the openings with their salts, which is what the firewall
commits and V* registers). Its hypotheses are the pieces still open, each a
statement about V* or the verifier, so that each can be proved on its own against this interface:

* `AlgebraDecodes` (G items 1–3): at an accepted outer proof whose layers are all `Good`, each rep's values in V*'s
  layers (`rep`) meet the preconditions E's stage theorems take, apart from the running sums: the verifier rows at the
  rep's coins (`verifierRows`), the structure (`structureOf`), the ports' sizes, and every residual zero. Its proof reads
  the algebra units the verifier pins, `PartS.res_partS` for each part, every part present (coverage), and the
  decode of the port columns from the registered reads.
* `AccChains` (G item 4): at the same outer proofs, each rep's running sums `rec-acc` differ by the levels' consistency
  sums weighted by `kappa`, E's `hacc`, from the chain of level sessions sharing `R₂`.
* `StagesAccept` (G item 5): at the same outer proofs, reps meeting E's preconditions make the compiled inner verifier
  accept the transcript V*'s layers hold at the rounds' coins. Its proof applies E's stage theorems
  (`zerocheck_of_residuals`, `lincheck_of_residuals`, `ringSwitch_of_residuals`, `ligerito_of_residuals`), each stage's
  converse (the checks hold, so the verifier's stage returns), the openings from V*'s `rec-open` units
  (`sound_recOpen`), and `table_tableAfterL` with `flockInnerCL_accepts`.
* `VCommits`, `VDecodes`, `VFull`: facts about V*'s public files and laws, the other three conjuncts of
  `Flock.Assumptions.VBridge` as they stand. They are about a concrete V* statement family, which Lean has not built
  yet (G item 6).

The composition itself is proved here; nothing in it is assumed beyond the hypotheses.
-/

namespace FlockVBridge.Compiled

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open Flock.RecAlgebra (verifierRows structureOf residuals valOf kapSum)

/-! ## What V*'s layers hold for one rep -/

/-- One rep's values in V*'s layers, as E's stage theorems read them: the message elements `w`, the comb, the levels'
consistency sums, the rounds' coins and the extra claims. -/
structure RepData where
  w : Array F128
  comb : Array F128
  cons : ℕ → ℕ → F128
  rounds : Array (Array F128)
  extras : Array Claim

/-- **E's preconditions at a rep, but the running sums**: the verifier rows at the rep's coins, the structure, the
ports' sizes, and every residual zero. -/
def AlgebraHolds (sh : Shape) (d : RepData) : Prop :=
  ∃ v info kappa S rs, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) ∧ structureOf sh = .ok S ∧
    d.w.size = S.nw ∧ d.comb.size = 64 ∧ residuals S d.w v = .ok rs ∧ ∀ x ∈ rs, x = 0

/-- **E's running-sums precondition at a rep** (`hacc`): at the verifier rows' `kappa`, each final-check component's
running sums add to the levels' consistency sums weighted by `kappa`. -/
def AccHolds (sh : Shape) (d : RepData) : Prop :=
  ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
    ∀ r < 2, valOf sh d.w d.comb d.cons (.acc1 r) + valOf sh d.w d.comb d.cons (.acc0 r) = kapSum kappa r d.cons

/-! ## The outer proofs `VBridge` reads -/

section outer

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}

/-- **An accepted outer proof**: at the history `(R, ω, t)` and registration `R₂`, every session's outcome
`flock-verify --zk --registered` accepts and every session's value layer is `Good`, no two sessions' registrant
records clash there (`RegColl`), and each session's record is a genuine session-game outcome: one its game reaches
against `τ j` (`Game.Outcome`; `VBridgeOff RegColl`'s premise). -/
def Accepted (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (R : Reg) (ω : L.Ω) (t : List (Cm × I.Coin)) (R₂ : Reg₂)
    (τ : (j : Fin S) → Strategy ((zs R ω t R₂ j).game dg N))
    (o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript) : Prop :=
  (∀ j, LiveAcceptsZKR (zs R ω t R₂ j).I (zs R ω t R₂ j).own (o j) ∧
    (zs R ω t R₂ j).Good ((zs R ω t R₂ j).layer dg N k Rw (τ j))) ∧ ¬ RegColl zs R ω t R₂ ∧
    ∀ j, Game.Outcome ((zs R ω t R₂ j).game dg N) (τ j) (o j)

/-- The sessions' value layers at those strategies. -/
noncomputable def layers (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (R : Reg) (ω : L.Ω) (t : List (Cm × I.Coin)) (R₂ : Reg₂)
    (τ : (j : Fin S) → Strategy ((zs R ω t R₂ j).game dg N)) : (j : Fin S) → (Vs j).Bits :=
  fun j => (zs R ω t R₂ j).layer dg N k Rw (τ j)

/-- **G items 1–3**: at every accepted outer proof, each rep's values at the history and V*'s layers meet E's
preconditions but the running sums. -/
def AlgebraDecodes (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Shape)
    (rep : Reg → L.Ω → List (Cm × I.Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData) : Prop :=
  ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
    AlgebraHolds sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r)

/-- **G item 4**: at every accepted outer proof, each rep's running sums meet E's `hacc`. -/
def AccChains (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Shape)
    (rep : Reg → L.Ω → List (Cm × I.Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData) : Prop :=
  ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r, AccHolds sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r)

/-- **G item 5**: at every accepted outer proof whose reps meet E's preconditions, the inner verifier `I` accepts the
transcript V*'s layers hold (`wmsg`) at the rounds' coins, on the hidden circuit's statement `x R ω`. -/
def StagesAccept (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Shape) (rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg) : Prop :=
  ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o →
    (∀ r, AlgebraHolds sh (rep (layers zs dg N k Rw R ω t R₂ τ) r) ∧
      AccHolds sh (rep (layers zs dg N k Rw R ω t R₂ τ) r)) →
    I.accepts (x R ω) ((wmsg (layers zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd))

end outer

/-! ## The bridge at the compiled inner protocol -/

section target

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [DecidableEq K] {D Sl : Type}
variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

/-- **VBridge G: V*'s circuits compute C-Flock's compiled inner verifier.** `Flock.Assumptions.VBridge` with the inner
protocol `flockInnerCL` (the compiled table on the class's hidden statements, the schedule `(T, CT)` with live coins),
from the pieces still open: the algebra decoded from V*'s layers (`AlgebraDecodes`), the running sums' chain
(`AccChains`), the inner verifier's stages run from E's conclusions (`StagesAccept`), and V*'s files and laws
(`VCommits`, `VDecodesR`, `VFullR`). -/
theorem vbridge_compiled (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (sch : Accounting.Schedule) {n : ℕ} {L : Law n} {Reg : Type} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (zs : Reg → L.Ω → List (Cm × (flockInnerCL T CT A H E Lf cls regs sch).Coin) → Reg₂ → (j : Fin S) →
      (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (x : Reg → L.Ω → Hidden cls)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (flockInnerCL T CT A H E Lf cls regs sch).Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) →
      (flockInnerCL T CT A H E Lf cls regs sch).Msg)
    -- V*'s algebra shape, and each rep's values in its layers
    (sh : Shape) (rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    -- G items 1–3, 4 and 5
    (halg : AlgebraDecodes zs dg N k Rw sh fun _ _ _ => rep) (hacc : AccChains zs dg N k Rw sh fun _ _ _ => rep)
    (hst : StagesAccept zs dg N k Rw sh rep x wmsg)
    -- V*'s public files and laws (G item 6)
    (hcm : VCommits zs Sk cmt) (hdec : VDecodesR wmsg (fun _ _ a b => a = b) Sk msg)
    (hfull : VFullR (flockInnerCL T CT A H E Lf cls regs sch).r Sk) :
    Flock.Assumptions.VBridge zs dg N k Rw x wmsg Sk cmt msg :=
  ⟨fun R ω t R₂ hc τ o hout h => hst R ω t R₂ τ o ⟨h, hc, hout⟩ fun r =>
    ⟨halg R ω t R₂ τ o ⟨h, hc, hout⟩ r, hacc R ω t R₂ τ o ⟨h, hc, hout⟩ r⟩, hcm, hdec, hfull⟩

end target

end FlockVBridge.Compiled
