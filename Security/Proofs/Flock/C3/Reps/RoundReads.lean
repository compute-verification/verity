import Proofs.Flock.C3.Reps.RoundLinks

/-!
# `RoundReads` and `WRoundReads` at the verifier's message-round ports

At `own := fun j => ownAt ss j`, the verifier's message ports of each statement, the reads file of each statement holds
the statement's message-round ports (`OwnHas`, `ownHas_ownAt`), so U's links (`C3/Reps/RoundLinks.lean`) give the
entries `RoundReads` and `WRoundReads` ask for: R's opening part's `m<i>` under `rec-opening-c<i>` at row 0
(`roundRead_opening`), a residuals' part's `m<m>` under `rec-c<m>` at rep `ρ`'s row (`wRoundRead_inner`), each where
the circuit has the port and `checkMsgNames` passed. The accepted sessions' reads hold `ownAt`'s ports (`OwnIn`,
`ownIn_checked`, from `checkOwn`). Left as hypotheses: the stage selector at R's rounds (`hsel`, U's `holoSel_opening`
at the map) and at the reps' (`holoSel_stageOf`), a part's session's unit name (`hpar`, `wSel_of_rounds`' `hck`) and
the unit a rep reads at (`hun`).

* `roundReads_ownAt`, `roundReads_checked`: `RoundReads` at `ownAt ss`, and at the checked family.
* `wRoundReads_ownAt`, `wRoundReads_checked`: `WRoundReads` at `ownAt ss` where a run is accepted, and at the checked
  family.
-/

namespace FlockC3.Reps.RoundReads

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.PubRows
open FlockC3.Compose
open FlockC3.Reps (OwnHas MsgNamesAt roundRead_opening wRoundRead_inner msgNamesAt_of_accepted)
open FlockVBridge.Family (partsOf)

theorem mem_ownAt {ss : Array VStar.Stmt} {j : ℕ} {s : VStar.Stmt} (hj : ss[j]? = some s)
    {q : String × String × Array (Option ℕ)} (hq : q ∈ RecVStar.ownPorts s) :
    { (default : Flock.Registered.Port) with name := q.1, value := q.2.1, reads := q.2.2 } ∈ ownAt ss j := by
  simp only [ownAt, hj, Option.map_some, Option.getD_some, List.map_toArray, List.mem_toArray]
  exact List.mem_map_of_mem (List.mem_append_left _ hq)

/-- **`ownAt` holds each statement's message-round ports** (`OwnHas`, U's `hown`). -/
theorem ownHas_ownAt {ish : RecVStar.Inner} {ss : Array VStar.Stmt} (hss : RecVStar.statements ish = .ok ss) (j : ℕ)
    (ss' : Array VStar.Stmt) (sv : VStar.Stmt) (hss' : RecVStar.statements ish = .ok ss') (hj : ss'[j]? = some sv) :
    OwnHas (ownAt ss j) sv := by
  obtain rfl : ss' = ss := Except.ok.inj (hss'.symm.trans hss)
  intro e he
  exact ⟨_, mem_ownAt hj he, rfl⟩

section Ats

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **`RoundReads` at the verifier's message-round ports**: at a copy whose circuit has a port `m<i>`,
`roundRead_opening` at `ownAt ss` (`ownHas_ownAt`), where stage `c₀ + i` reads `rec-opening-c<i>` at row 0
(`hsel`). -/
theorem roundReads_ownAt {ish : RecVStar.Inner} {bind : Fin S → VStar.Bindings}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    {ss : Array VStar.Stmt} {sel : ℕ → String → ℕ → Prop} {c₀ : ℕ}
    (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs) (hst : SessStmt ish bind zs dg N k Rw)
    (hmn : MsgNamesAt ish zs dg N k Rw) (hss : RecVStar.statements ish = .ok ss)
    (hsel : ∀ i < 2, sel (c₀ + i) s!"rec-opening-c{i}" 0) :
    RoundReads zs dg N k Rw sh w (fun j => ownAt ss j) sel c₀ := by
  intro i hi _ _ _ _ _ hO q hq
  exact roundRead_opening hob hst hmn (fun s => ownHas_ownAt hss s) (hsel i hi) hO q hq

/-- **`WRoundReads` at the verifier's message-round ports**, where a run is accepted (`hA`): at part `p`'s session,
whose unit is named as a part's (`hpar`) and whose circuit has a port `m<m>`, `wRoundRead_inner` at `ownAt ss`
(`ownHas_ownAt`), at rep `ρ`'s unit (`hun`) where rep `ρ`'s stage of round `m` reads `rec-c<m>` at row `ρ`
(`hsel`). -/
theorem wRoundReads_ownAt {ish : RecVStar.Inner} {bind : Fin S → VStar.Bindings}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape} {ss : Array VStar.Stmt}
    {sel : ℕ → String → ℕ → Prop} {stage : Fin 2 → ℕ → ℕ} {sess : ℕ → Fin S}
    {un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv}
    (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs) (hst : SessStmt ish bind zs dg N k Rw)
    (hmn : MsgNamesAt ish zs dg N k Rw) (hss : RecVStar.statements ish = .ok ss) {R : Reg} {ω : L.Ω}
    {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o} (hA : FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o)
    (hpar : ∀ p < (partsOf sh).size, ∃ m kk cl j,
      RecResiduals.params (Vs (sess p)).c.unit.name = some (m, kk, cl, j))
    (hun : ∀ p (ρ : Fin 2), ρ.val < (Vs (sess p)).nv → (un p ρ).val = ρ.val)
    (hsel : ∀ p < (partsOf sh).size, ∀ (m : ℕ) (ρ : Fin 2) (q : Fin (Vs (sess p)).c.ports.size),
      (Vs (sess p)).c.ports[q].name = s!"m{m}" → sel (stage ρ m) s!"rec-c{m}" ρ.val) :
    WRoundReads sh (fun j => ownAt ss j) sel stage sess un := by
  intro p hp m ρ q hq
  obtain ⟨_, _, _, _, hpm⟩ := hpar p hp
  exact wRoundRead_inner hob hst hmn hA (fun s => ownHas_ownAt hss s) hun hpm ρ (hsel p hp m ρ q hq) q hq

end Ats

section Checked

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

variable {xR : HoloReg GF128 Dig} {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
  {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin (vstarCount ish)) → (Vs j).OuterR Reg'}
  {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr} {coins : String → Option Flock.RecCoins.Source}
  {rows : Option (Array Flock.Registered.Port)} {own : Fin (vstarCount ish) → Array Flock.Registered.Port}
  {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (ℐ).Msg}
  {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}

/-- **`RoundReads` at the checked family** (`C3Family.checked`), at `ownAt ss`: `SessStmt` from the prover's sessions'
tags (`sessStmt_checked`, the guarantee's `_hF`), `MsgNamesAt` from the verifier's checks (`msgNamesAt_of_accepted`,
`checksAccepted_checked`). -/
theorem roundReads_checked (hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {sh : Flock.Shape} {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Array F128}
    {c₀ : ℕ} (hsel : ∀ i < 2, sel (c₀ + i) s!"rec-opening-c{i}" 0) :
    RoundReads (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs
      dg N k Rw sh w (fun j => ownAt ss j) sel c₀ :=
  roundReads_ownAt hob (sessStmt_checked hF)
    (msgNamesAt_of_accepted
      (F := C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg)
      checksAccepted_checked) hss hsel

/-- **`WRoundReads` at the checked family** (`C3Family.checked`), at `ownAt ss`, where one of its runs is accepted:
`wRoundReads_ownAt` with `SessStmt` and `MsgNamesAt` as `roundReads_checked` has them. -/
theorem wRoundReads_checked (hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I) {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {sh : Flock.Shape} {stage : Fin 2 → ℕ → ℕ} {sess : ℕ → Fin (vstarCount ish)}
    {un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv} {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (hA : FlockVBridge.Compiled.Accepted (I := ℐ)
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs dg N k Rw
        R ω t R₂ τ o)
    (hpar : ∀ p < (partsOf sh).size, ∃ m kk cl j,
      RecResiduals.params (Vs (sess p)).c.unit.name = some (m, kk, cl, j))
    (hun : ∀ p (ρ : Fin 2), ρ.val < (Vs (sess p)).nv → (un p ρ).val = ρ.val)
    (hsel : ∀ p < (partsOf sh).size, ∀ (m : ℕ) (ρ : Fin 2) (q : Fin (Vs (sess p)).c.ports.size),
      (Vs (sess p)).c.ports[q].name = s!"m{m}" → sel (stage ρ m) s!"rec-c{m}" ρ.val) :
    WRoundReads sh (fun j => ownAt ss j) sel stage sess un :=
  wRoundReads_ownAt hob (sessStmt_checked (dg := dg) (N := N) (k := k) (Rw := Rw) hF)
    (msgNamesAt_of_accepted
      (F := C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg)
      checksAccepted_checked) hss hA hpar hun hsel

end Checked

end FlockC3.Reps.RoundReads
