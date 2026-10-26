import Proofs.Flock.Claims.KS.SimLead
import Proofs.Flock.Claims.KS.PlainSched
import Proofs.Flock.Claims.Recursion.PlainK

/-!
# The simulated strategy after `Commit` leads with rep 0, from the live prover's schedule

`simAt`, the simulated strategy after `Commit` at one table, is at each link point the simulation's strategy
(`simRJ`) with its dead branches answered by empty openings, read at the table's one batch position (`simAt_heq`). So
`lead_simRJ` makes it lead with rep 0 for `n` coins wherever the live strategy submits its next `n` rounds on rep 0
and the rest on rep 1:

* `leadsAfter_simAt`: `LeadsAfter` at `simAt`, from `SchedL` at every link point;
* `leadsAfter_plainSimAt`: so at `plainSimAt`, from `SchedL` at the plain simulator's live strategy (`plainLive`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

section Casts

variable {β : Type}

theorem coin_k_eq {C : Type} {iF : Fintype C} {iN : Nonempty C} {k k' : C → Game β}
    (e : @Game.coin β C iF iN k = @Game.coin β C iF iN k') : k = k' := by
  cases e
  rfl

/-- A strategy cast along an equation of coin games, at a coin. -/
theorem cast_coin_heq {C : Type} {iF iF' : Fintype C} {iN : Nonempty C} {k k' : C → Game β}
    (eg : @Game.coin β C iF iN k = @Game.coin β C iF' iN k') (σ : Strategy (@Game.coin β C iF iN k)) (c : C) :
    HEq (cast (congrArg Strategy eg) σ c) (σ c) := by
  obtain rfl : iF = iF' := Subsingleton.elim _ _
  obtain rfl := coin_k_eq eg
  rfl

/-- Equal strategies of equal games sending a message then drawing a coin agree after the message, at a coin. -/
theorem heq_snd_coin {M C : Type} {iF iF' : Fintype C} {iN : Nonempty C} {k k' : M → C → Game β}
    (e : Game.send M (fun m => @Game.coin β C iF iN (k m)) = Game.send M (fun m => @Game.coin β C iF' iN (k' m)))
    {s : Strategy (Game.send M fun m => @Game.coin β C iF iN (k m))}
    {s' : Strategy (Game.send M fun m => @Game.coin β C iF' iN (k' m))} (h : HEq s s') (c : C) :
    HEq (s.2 c) (s'.2 c) := by
  obtain rfl : iF = iF' := Subsingleton.elim _ _
  obtain rfl : k = k' := funext fun m => coin_k_eq (congrFun (eq_of_heq (Game.send.inj e).2) m)
  obtain rfl := eq_of_heq h
  rfl

end Casts

/-- The salted session's simulated strategy is the session simulation's strategy, carried along
`sessionBL_eq_modelSession`. -/
theorem simSessG_heq {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (ts : List (TabSpec mPts)) (ok : Model.OutCL 𝔽 D Sl → Prop)
    (pL : LiveOutJ ts.length mPts → Prop) (opn : Array ByteArray → Fin ts.length → Model.Opens 𝔽 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A (ts.get j)) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) :
    HEq (simSessG A Hc E Lf ts ok pL opn dg dec one fin capOf N hdec Pr)
      ((Sim.mapDead (answerWith (nulOp (D := D) (Sl := Sl)))
        (simSession (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) dfltOutL pL opn dg
          dec one fin capOf N (fun j => repsOf (D := D) A (ts.get j))
          (fun j _ _ => Model.allInh_repC A _ (ts.get j).S (ts.get j).sch (ts.get j).hm _) hdec)).strategy Pr) := by
  unfold simSessG
  exact eqRec_heq_self _ _

section Gen

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D] [Inhabited Sl] {mPts : ℕ}
  (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc)
  (t : TabSpec mPts) (ok : Model.OutCL 𝔽 D Sl → Prop) (pL : LiveOutJ [t].length mPts → Prop)
  (opn : Array ByteArray → Fin [t].length → Model.Opens 𝔽 D × Model.Salts Sl) (dg : ByteArray → ByteArray)
  (dec : Fin [t].length → ℕ → ByteArray → List Msg) (one : Fin [t].length → ℕ → Bool)
  (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin [t].length → Model.Cap D)
  (N : ℕ)
  (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
    opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)

/-- **`simAt` at the link points `pts`** is the session simulation's strategy there, its dead branches answered with
empty openings. -/
theorem simAt_heq (Pr : Strategy (liveSession dg [t].length mPts N)) (pts : Fin 2 → Fin mPts → 𝔽) :
    HEq (simAt A Hc E Lf t ok pL opn dg dec one fin capOf N hdec Pr.1 Pr rfl pts)
      (((Sim.mapDead (answerWith (nulOp (D := D) (Sl := Sl)))
        (simSession (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) dfltOutL pL opn dg
          dec one fin capOf N (fun j => repsOf (D := D) A ([t].get j))
          (fun j _ _ => Model.allInh_repC A _ ([t].get j).S ([t].get j).sch ([t].get j).hm _) hdec)).strategy
        Pr).2 pts) := by
  have h₁ := cast_coin_heq
    (congrArg (fun c => sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] c)
      (congrFun (simSessG_fst A Hc E Lf [t] ok pL opn dg dec one fin capOf N hdec Pr) ⟨0, Nat.one_pos⟩))
    (afterOne A Hc E Lf t (simSessG A Hc E Lf [t] ok pL opn dg dec one fin capOf N hdec Pr)) pts
  have h₂ := cast_coin_heq (sessAfterBL_one A Hc E Lf t (simSessG A Hc E Lf [t] ok pL opn dg dec one fin capOf N hdec
    Pr).1) (simSessG A Hc E Lf [t] ok pL opn dg dec one fin capOf N hdec Pr).2 pts
  have h₃ := heq_snd_coin (sessionBL_eq_modelSession A Hc E Lf [t])
    (simSessG_heq A Hc E Lf [t] ok pL opn dg dec one fin capOf N hdec Pr).symm pts
  exact h₁.trans (h₂.trans h₃.symm)

/-- **The simulated strategy after `Commit` leads with rep 0 for `n` coins** at every link point where the live
strategy submits its next `n` rounds on rep 0 and the rest on rep 1 (`SchedL`), when rep 0 sends no final message. -/
theorem leadsAfter_simAt (hfin : ∀ proofs, fin ⟨0, Nat.one_pos⟩ 0 proofs = []) {n : ℕ} (rootB₀ : Array ByteArray)
    (Pr : Strategy (liveSession dg [t].length mPts N)) (hP : Pr.1 = rootB₀)
    (hS : ∀ pts, SchedL dg Pr.1 pts N (fun _ _ => []) n (Pr.2 pts)) :
    LeadsAfter A Hc E Lf t.S t.sch t.hm t.ptLocal mPts (capOf rootB₀ ⟨0, Nat.one_pos⟩) n
      (simAt A Hc E Lf t ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP) := by
  subst hP
  intro pts
  have hA := Sim.All.mono (R' := fun b _ => pL b)
    (D' := DeadAt (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun j => repsOf (D := D) A ([t].get j)))
    (fun _ _ h => h.1) (fun _ _ h => h) _
    (all_simSession (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) dfltOutL pL opn dg dec one
      fin capOf N (fun j => repsOf (D := D) A ([t].get j))
      (fun j _ _ => Model.allInh_repC A _ ([t].get j).S ([t].get j).sch ([t].get j).hm _) hdec Pr.1 pts)
  exact lead_of_eq (congrArg (unB _) (eq_of_heq (simAt_heq A Hc E Lf t ok pL opn dg dec one fin capOf N hdec Pr
      pts)).symm)
    (lead_simRJ (Dead := NulOut) (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) dfltOutL pL
      opn dg dec one fin (capOf Pr.1) Pr.1 pts nulOp (fun _ _ _ _ _ => rfl) hfin N _ _ _ _ (Pr.2 pts) n (hS pts) hA)

end Gen

/-! ## At the plain simulator -/

/-- The plain simulator's live strategy. -/
noncomputable abbrev plainLive (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (hw : 2 * spec.mPts ≤ 65536)
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (u : plainTape x) :
    Strategy (liveSession (plainDg spec) 1 spec.mPts (plainN x)) :=
  liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512 (plainSim P x spec st) x
    (ω, u)

/-- The plain simulator's live strategy. -/
noncomputable abbrev plainLiveK (keep : Bool → Bool) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (hw : 2 * spec.mPts ≤ 65536)
    {Ω : Type} (P : Claims.Prover Ω) (ω : Ω) (u : plainTape x) :
    Strategy (liveSession (plainDg spec) 1 spec.mPts (plainN x)) :=
  liveStrat (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512 (plainSimK keep P x spec st) x
    (ω, u)

/-- **`plainSimAt` leads with rep 0 for `n` coins** where the plain simulator's live strategy submits its next `n`
rounds on rep 0 and the rest on rep 1. -/
theorem leadsAfter_plainSimAt {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDec L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) {n : ℕ}
    (hS : ∀ pts, SchedL (plainDg spec) (plainLive x spec st hw P ω u).1 pts (plainN x) (fun _ _ => []) n
      ((plainLive x spec st hw P ω u).2 pts)) :
    LeadsAfter execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
      ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) n
      (plainSimAt L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u) :=
  leadsAfter_simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
    (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ _ opn (plainDg spec)
    (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
    (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (fun _ => rfl) (plainRoot P x ω)
    (plainLive x spec st hw P ω u) (plainSim_root P x spec st ω hw u) hS

end FlockDeploy.KS
