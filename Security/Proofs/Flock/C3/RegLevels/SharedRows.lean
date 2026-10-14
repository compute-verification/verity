import Proofs.Flock.C3.RegLevels.AccReads

/-!
# `SharedOwnRoot`: two sessions register a shared value under one root, from `checkShared`

`row_agree` reads two of V*'s accepted sessions' entries for one value at one root, leaf count and domain. The
verifier's statement check runs `RecVStar.checkShared` against its record of V*'s shared rows (`--rec-rows`, one file
for every session of the family): a port of a session's reads file reading a value some entry of `rows` reads has the
first such entry's registration (`portDiff?` but reads), and a port reading R's tops `rec-top-stacked-L<l>` needs such
an entry. So:

* `site_of_checkShared`: each port of the reads file reading a value `SharedValue` names (one `rows` registers, or R's
  tops) is the first entry of `rows` reading it, but in reads;
* `shared_of_checkShared`: two reads files the check passed register such a value under one root, leaf count and
  domain (both are that entry's, `site_of_portDiff`);
* `sharedOwnRoot_of_checks`, `sharedOwnRoot_checked`: so at a family whose accepted sessions pass the statement check
  (`ChecksAccepted`; the checked family's, `checksAccepted_checked`) `SharedOwnRoot` holds at `rows`. A value `rows`
  does not register, other than R's tops, is not covered.
-/

namespace FlockC3.RegLevels

open Flock
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion

/-- **A value the verifier's shared rows register, or R's tops** (`checkShared`'s two cases). -/
def SharedValue (rows : Option (Array Flock.Registered.Port)) (v : String) : Prop :=
  (rows.bind (·.find? (·.value == v))).isSome ∨ (RecVStar.stackedTopLevel v).isSome

/-- **`checkShared`'s loop**: a port of `own` reading a `SharedValue` agrees, but in reads, with the first entry of `rows`
reading its value. -/
theorem site_of_checkShared {s : VStar.Stmt} {own : Array Flock.Registered.Port}
    {rows : Option (Array Flock.Registered.Port)} (h : RecVStar.checkShared s own rows = .ok ())
    {p : Flock.Registered.Port} (hp : p ∈ own) (hv : SharedValue rows p.value) :
    ∃ w, rows.bind (·.find? (·.value == p.value)) = some w ∧ RecVStar.portDiff? p w false = none := by
  unfold RecVStar.checkShared at h
  rw [← Array.forIn_toList] at h
  obtain ⟨_, h, -⟩ := FlockSoundness.Refine.ebind_ok h
  refine FlockC3.PubRows.rows_forIn_each (fun p => SharedValue rows p.value →
      ∃ w, rows.bind (·.find? (·.value == p.value)) = some w ∧ RecVStar.portDiff? p w false = none)
    ?_ _ h p (Array.mem_toList_iff.2 hp) hv
  intro p r hr
  split at hr
  · rename_i w hw
    split at hr
    · obtain ⟨_, h2, -⟩ := FlockSoundness.Refine.ebind_ok hr
      cases h2
    · rename_i hn
      refine ⟨fun _ => ⟨w, hw, ?_⟩, ?_⟩
      · cases hd : RecVStar.portDiff? p w false with
        | none => rfl
        | some d => exact absurd hd (hn d)
      · cases hr; rfl
  · rename_i hw
    split at hr
    · obtain ⟨_, h2, -⟩ := FlockSoundness.Refine.ebind_ok hr
      cases h2
    · rename_i hn
      refine ⟨fun hv => ?_, by cases hr; rfl⟩
      rcases hv with hv | hv
      · rw [hw] at hv; cases hv
      · exact absurd hv hn

/-- **Two reads files `checkShared` passed at one `rows` register a `SharedValue` under one root, leaf count and
domain**: both entries are the first entry of `rows` reading it, but in reads. -/
theorem shared_of_checkShared {s s' : VStar.Stmt} {own own' : Array Flock.Registered.Port}
    {rows : Option (Array Flock.Registered.Port)} (h : RecVStar.checkShared s own rows = .ok ())
    (h' : RecVStar.checkShared s' own' rows = .ok ()) {p p' : Flock.Registered.Port} (hp : p ∈ own) (hp' : p' ∈ own')
    (hv : p.value = p'.value) (hs : SharedValue rows p.value) :
    p.root = p'.root ∧ p.leaves = p'.leaves ∧ p.domain = p'.domain := by
  obtain ⟨w, hw, hd⟩ := site_of_checkShared h hp hs
  obtain ⟨w', hw', hd'⟩ := site_of_checkShared h' hp' (hv ▸ hs)
  rw [← hv, hw] at hw'
  cases hw'
  have e := (site_of_portDiff hd).trans (site_of_portDiff hd').symm
  simp only [FlockVBridge.Family.site, Prod.mk.injEq] at e
  exact ⟨e.2.1, e.2.2.1, e.2.2.2.2.2⟩

section Family

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **Two of V*'s accepted sessions at a history register a shared value under one root, leaf count and domain**: a
value the verifier's shared rows `rows` register, or R's tops (`SharedValue`). -/
def SharedOwnRoot (rows : Option (Array Flock.Registered.Port)) (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) :
    Prop :=
  ∀ R ω t R₂ j j', (∃ tr, AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr) →
    (∃ tr, AcceptsZKR (F.zs R ω t R₂ j').I (F.zs R ω t R₂ j').own tr) →
    ∀ p ∈ (F.zs R ω t R₂ j).own, ∀ p' ∈ (F.zs R ω t R₂ j').own, p.value = p'.value → SharedValue rows p.value →
      p.root = p'.root ∧ p.leaves = p'.leaves ∧ p.domain = p'.domain

/-- **`SharedOwnRoot` where the accepted sessions pass the statement check** (`ChecksAccepted`): each session's first
statement's check (`setup_zero_c`) ran `checkShared` on its reads file against `rows` (`regs_of_checkStmt`). -/
theorem sharedOwnRoot_of_checks {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂}
    {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hca : FlockC3.PubRows.ChecksAccepted F rows wr reg) : SharedOwnRoot rows F := by
  intro R ω t R₂ j j' hex hex' p hp p' hp' hv hs
  have one : ∀ j, (∃ tr, AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr) →
      ∃ s, RecVStar.checkShared s (F.zs R ω t R₂ j).own rows = .ok () := by
    intro j hex
    obtain ⟨tr, -, -, spec, sts, hs, hfm⟩ := hca R ω t R₂ j hex
    obtain ⟨h0, -⟩ := setup_zero_c (F.zs R ω t R₂ j).ht (F.zs R ω t R₂ j).hTags.1 (F.zs R ω t R₂ j).hcH hs
    obtain ⟨-, s, -, -, -, -, h⟩ := regs_of_checkStmt (array_forM_ok hfm 0 h0)
    exact ⟨s, h⟩
  obtain ⟨s, h⟩ := one j hex
  obtain ⟨s', h'⟩ := one j' hex'
  exact shared_of_checkShared h h' hp hp' hv hs

variable {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg}
  {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}

/-- **The checked family's `SharedOwnRoot`** at its shared rows (`checksAccepted_checked`, `sharedOwnRoot_of_checks`). -/
theorem sharedOwnRoot_checked {coins : String → Option Flock.RecCoins.Source}
    {rows : Option (Array Flock.Registered.Port)}
    {zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
      (Vs j).OuterR Reg'}
    {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr} :
    SharedOwnRoot rows (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) :=
  sharedOwnRoot_of_checks (FlockC3.PubRows.checksAccepted_checked (coins := coins) (rows := rows) (zr := zr)
    (hCust := hCust) (h := h))

end Family

end FlockC3.RegLevels
