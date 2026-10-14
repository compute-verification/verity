import Proofs.Flock.C3.Reps.RoundLinks

/-!
# `RoundReads` and `WRoundReads` at the verifier's message-round ports

At `own := fun j => ownAt ss j`, the verifier's message ports of each statement, the reads file of each statement holds
the statement's message-round ports (`OwnHas`, `ownHas_ownAt`), so U's links (`C3/Reps/RoundLinks.lean`) give the
entries `RoundReads` and `WRoundReads` ask for: R's opening part's `m<i>` under `rec-opening-c<i>` at row 0
(`roundRead_opening`), a residuals' part's `m<m>` under `rec-c<m>` at rep `ρ`'s row (`wRoundRead_inner`), each where
the circuit has the port and `checkMsgNames` passed. The accepted sessions' reads hold `ownAt`'s ports (`OwnIn`,
`ownIn_checked`, from `checkOwn`). At the map's `sel` (`holoMap`), R's first two message rounds are R's rounds 0 and 1,
one history round apart (`stackedRounds_open`, by evaluating `stackedRounds`; `roundAt_opening`), so the map reads
`rec-opening-c<i>` at `c₀ + i`, `c₀` R's round 0's history round (`holoSel_opening_at`). At the reps' rounds the map
reads a part's port `m<m>` at `stageOf σ c₀` (`holoSel_parts`, from `holoSel_stageOf`) where `m` is one of the reps'
message rounds and `roundOf σ m` the reps' history rounds before its stream round (`hm`, left as a hypothesis, with
`c₀` rep 0's history rounds). Left as hypotheses too: a part's session's unit name (`hpar`, `wSel_of_rounds`' `hck`)
and the unit a rep reads at (`hun`).

* `roundReads_ownAt`, `roundReads_checked`: `RoundReads` at `ownAt ss`, and at the checked family; at the map's `sel`
  and `c₀ := roundAt L .stacked 0`, `roundReads_checked` takes `holoSel_opening_at` as its `hsel`.
* `holoSel_parts`: `wRoundReads_checked`'s `hsel` at the map's `sel` and `stageOf σ c₀`.
* `wRoundReads_ownAt`, `wRoundReads_checked`: `WRoundReads` at `ownAt ss` where some outer proof is accepted
  (`AcceptedAt`), and at the checked family.
-/

namespace FlockC3.Reps.RoundReads

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.PubRows
open FlockC3.Compose
open FlockC3.Reps (OwnHas MsgNamesAt AcceptedAt roundRead_opening wRoundRead_inner msgNamesAt_of_accepted)
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

/-- **`WRoundReads` at the verifier's message-round ports**, where some outer proof is accepted (`hA`): at part `p`'s
session, whose unit is named as a part's (`hpar`) and whose circuit has a port `m<m>`, `wRoundRead_inner` at `ownAt ss`
(`ownHas_ownAt`), at rep `ρ`'s unit (`hun`) where rep `ρ`'s stage of round `m` reads `rec-c<m>` at row `ρ`
(`hsel`). -/
theorem wRoundReads_ownAt {ish : RecVStar.Inner} {bind : Fin S → VStar.Bindings}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape} {ss : Array VStar.Stmt}
    {sel : ℕ → String → ℕ → Prop} {stage : Fin 2 → ℕ → ℕ} {sess : ℕ → Fin S}
    {un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv}
    (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs) (hst : SessStmt ish bind zs dg N k Rw)
    (hmn : MsgNamesAt ish zs dg N k Rw) (hss : RecVStar.statements ish = .ok ss) (hA : AcceptedAt zs dg N k Rw)
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

/-! ## R's opening's rounds at the map -/

section Opening

open Flock.RecVStar Flock.Guarantees

private theorem bind_some {α β : Type} {x : Option α} {f : α → Option β} {b : β} (h : (x >>= f) = some b) :
    ∃ a, x = some a ∧ f a = some b := by
  cases x with
  | none => cases h
  | some a => exact ⟨a, rfl, h⟩

/-- **R's stream opens with its two message rounds**: its rounds 0 and 1 absorb its first two messages, and round 0 takes
one history round, by evaluating `stackedRounds` at each `m` below 36; past 35 `fast100` has none, nor has
`stackedRounds`. -/
theorem stackedRounds_open {m : ℕ} {rs : Array SchedRound} (h : stackedRounds ⟨m, STACKED_CLAIMS⟩ = some rs) :
    (msgRounds rs).take 2 = [0, 1] ∧ heldBefore .stacked rs 1 = 1 := by
  by_cases hm : m < 36
  · have key : ∀ m' < 36, (stackedRounds ⟨m', STACKED_CLAIMS⟩).all
        (fun rs => decide ((msgRounds rs).take 2 = [0, 1] ∧ heldBefore .stacked rs 1 = 1)) := by
      decide +kernel
    have := key m hm
    rw [h] at this
    simpa using this
  · have hb : (m < 22 || m > 35) = true := by
      simp only [Bool.or_eq_true, decide_eq_true_eq]
      omega
    have h0 : Flock.fast100 m = none := by
      unfold Flock.fast100
      simp [hb]
    have hn : stackedRounds ⟨m, STACKED_CLAIMS⟩ = none := by
      unfold stackedRounds
      rw [h0]
      rfl
    rw [hn] at h
    cases h

/-- **R's rounds take their history rounds after every other stream's**: in streams ending in R's, R's round `k` is its
round 0's history round plus R's rounds before `k`. -/
theorem roundIn_stacked (P : List (HoloStream × Array SchedRound)) (hP : ∀ e ∈ P, e.1.idx ≠ HoloStream.stacked.idx)
    (rs : Array SchedRound) (k : ℕ) :
    roundIn (P ++ [(.stacked, rs)]) .stacked k = roundIn (P ++ [(.stacked, rs)]) .stacked 0 + heldBefore .stacked rs k := by
  induction P with
  | nil =>
    show (if HoloStream.stacked.idx = HoloStream.stacked.idx then heldBefore .stacked rs k else _) =
      (if HoloStream.stacked.idx = HoloStream.stacked.idx then heldBefore .stacked rs 0 else _) + _
    rw [if_pos rfl, if_pos rfl]
    simp [heldBefore]
  | cons e P ih =>
    obtain ⟨s', a⟩ := e
    have hne : s'.idx ≠ HoloStream.stacked.idx := hP _ List.mem_cons_self
    have ih' := ih fun e he => hP e (List.mem_cons_of_mem _ he)
    show (if s'.idx = HoloStream.stacked.idx then _ else heldBefore s' a a.size + roundIn (P ++ [(.stacked, rs)]) .stacked k) =
      (if s'.idx = HoloStream.stacked.idx then _ else
        heldBefore s' a a.size + roundIn (P ++ [(.stacked, rs)]) .stacked 0) + _
    rw [if_neg hne, if_neg hne, ih']
    omega

/-- **`holoStreams` ends in R's stream** where the inner shape names R: the streams before it are the reps' and the
sparse phases', and R's rounds are `stackedRounds`'. -/
theorem holoStreams_stacked {m kLog claims : ℕ} {sp : Option RecSparse.Shape} {r : StackedShape}
    {L : List (HoloStream × Array SchedRound)} (h : holoStreams m kLog claims sp (some r) = some L) :
    ∃ rep, holoRounds m kLog claims = some rep ∧ ∃ P rs, L = P ++ [(.stacked, rs)] ∧
      (∀ e ∈ P, e.1.idx ≠ HoloStream.stacked.idx) ∧ stackedRounds r = some rs := by
  unfold holoStreams at h
  obtain ⟨rep, hrep, h⟩ := bind_some h
  obtain ⟨R, hR, h⟩ := bind_some h
  obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 hR
  refine ⟨rep, hrep, _, rs, (Option.some.inj h).symm, ?_, hrs⟩
  intro e he
  cases sp <;> simp at he <;> rcases he with rfl | rfl | rfl | rfl <;> simp [HoloStream.idx]

/-- **R's first two message rounds take consecutive history rounds**, the first at R's round 0's, where the inner shape
names R at `STACKED_CLAIMS` claims. -/
theorem roundAt_opening {m kLog claims mR : ℕ} {sp : Option RecSparse.Shape} {L : List (HoloStream × Array SchedRound)}
    (h : holoStreams m kLog claims sp (some ⟨mR, STACKED_CLAIMS⟩) = some L) :
    ∃ rep, holoRounds m kLog claims = some rep ∧ ∃ rs, stackedRounds ⟨mR, STACKED_CLAIMS⟩ = some rs ∧
      ∀ i < 2, ∃ hl : i < (msgRounds rs).length, roundAt L .stacked ((msgRounds rs)[i]'hl) = roundAt L .stacked 0 + i := by
  obtain ⟨rep, hrep, P, rs, rfl, hP, hrs⟩ := holoStreams_stacked h
  obtain ⟨ht, h1⟩ := stackedRounds_open hrs
  refine ⟨rep, hrep, rs, hrs, fun i hi => ?_⟩
  have hlen : 2 ≤ (msgRounds rs).length := by
    have := congrArg List.length ht
    rw [List.length_take] at this
    simp at this
    omega
  refine ⟨by omega, ?_⟩
  have hget : (msgRounds rs)[i]'(by omega) = i := by
    have e : ((msgRounds rs).take 2)[i]'(by rw [List.length_take]; omega) = (msgRounds rs)[i]'(by omega) :=
      List.getElem_take
    rw [← e]
    simp only [ht]
    rcases (show i = 0 ∨ i = 1 by omega) with rfl | rfl <;> rfl
  rw [hget]
  unfold roundAt
  rw [roundIn_stacked P hP rs i]
  rcases (show i = 0 ∨ i = 1 by omega) with rfl | rfl
  · simp [heldBefore]
  · rw [h1]
    omega

/-- **The map reads R's opening's round `i` at `rec-opening-c<i>`, row 0, at R's round 0's history round plus `i`**
(`holoSel_opening` at `roundAt_opening`), where the inner shape names R at `STACKED_CLAIMS` claims
(`FlockC3.VStarList.stacked_of_candidate3`). -/
theorem holoSel_opening_at {st : Flock.Stmt} {hp : HoloParams} {Cm : Type} {ish : RecVStar.Inner} {tbl : String}
    {bm : Cm → String → ℕ → List UInt8} {L : List (HoloStream × Array SchedRound)}
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some L) {mR : ℕ}
    (hst : ish.stacked = some ⟨mR, STACKED_CLAIMS⟩) :
    ∀ i < 2, (holoMap st hp Cm ish tbl bm).sel (roundAt L .stacked 0 + i) s!"rec-opening-c{i}" 0 := by
  intro i hi
  have hS' := hS
  rw [hst] at hS'
  obtain ⟨rep, hr, rs, hrs, hat⟩ := roundAt_opening hS'
  obtain ⟨hl, he⟩ := hat i hi
  rw [← he]
  exact holoSel_opening hS hr (by rw [hst]; exact hrs) hl

theorem heldBefore_rep (ρ : Fin 2) (rs : Array SchedRound) (k : ℕ) :
    heldBefore (.rep ρ) rs k = heldBefore (.rep 0) rs k := rfl

/-- **The map reads each part's round `m` at rep `ρ`'s stage of it** (`holoSel_stageOf`), where `c₀` is rep 0's history
rounds and, at each port `m<m>` of a part's circuit, `m` is one of the reps' message rounds whose stream round has
`roundOf σ m` of the reps' history rounds before it (`hm`): `WRoundReads`' stage selector at the map. -/
theorem holoSel_parts {st : Flock.Stmt} {hp : HoloParams} {Cm : Type} {ish : RecVStar.Inner} {tbl : String}
    {bm : Cm → String → ℕ → List UInt8} {rs : Array SchedRound} {L : List (HoloStream × Array SchedRound)}
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some ((.rep 0, rs) :: (.rep 1, rs) :: L))
    (hr : holoRounds ish.m ish.kLog ish.claims = some rs) {σ : List FlockSoundness.Discharge.Exec.Kind} {c₀ : ℕ}
    (hc₀ : c₀ = heldBefore (.rep 0) rs rs.size) {S : ℕ} {Vs : Fin S → VStmt} {sh : Flock.Shape} {sess : ℕ → Fin S}
    (hm : ∀ p < (partsOf sh).size, ∀ (m : ℕ) (q : Fin (Vs (sess p)).c.ports.size),
      (Vs (sess p)).c.ports[q].name = s!"m{m}" → ∃ hm : m < (msgRounds rs).length,
        FlockC3.Sparse.roundOf σ m = heldBefore (.rep 0) rs ((msgRounds rs)[m]'hm)) :
    ∀ p < (partsOf sh).size, ∀ (m : ℕ) (ρ : Fin 2) (q : Fin (Vs (sess p)).c.ports.size),
      (Vs (sess p)).c.ports[q].name = s!"m{m}" →
        (holoMap st hp Cm ish tbl bm).sel (stageOf σ c₀ ρ m) s!"rec-c{m}" ρ.val := by
  intro p hp' m ρ q hq
  obtain ⟨hm', hσ⟩ := hm p hp' m q hq
  exact holoSel_stageOf hS hr ρ hm' hc₀ (hσ.trans (heldBefore_rep ρ rs _).symm)

end Opening

section Checked

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

-- The session binders carry `ℐ` in their types, so they go on each theorem, never in a `variable` block: a
-- `variable` whose binders use a local notation over section variables makes Lean silently drop every declaration
-- after it (the audit's `dropped` check, #1923, refuses the package).

/-- **`RoundReads` at the checked family** (`C3Family.checked`), at `ownAt ss`: `SessStmt` from the prover's sessions'
tags (`sessStmt_checked`, the guarantee's `_hF`), `MsgNamesAt` from the verifier's checks (`msgNamesAt_of_accepted`,
`checksAccepted_checked`). -/
theorem roundReads_checked {xR : HoloReg GF128 Dig} {ish : RecVStar.Inner} {ss : Array VStar.Stmt}
    {hss : RecVStar.statements ish = .ok ss} {bind : Fin (vstarCount ish) → VStar.Bindings}
    {Vs : Fin (vstarCount ish) → VStmt} {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
    {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin (vstarCount ish)) → (Vs j).OuterR Reg'}
    {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr} {coins : String → Option Flock.RecCoins.Source}
    {rows : Option (Array Flock.Registered.Port)} {own : Fin (vstarCount ish) → Array Flock.Registered.Port}
    {sel : ℕ → String → ℕ → Prop}
    {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
    {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (ℐ).Msg}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {sh : Flock.Shape} {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Array F128}
    {c₀ : ℕ} (hsel : ∀ i < 2, sel (c₀ + i) s!"rec-opening-c{i}" 0) :
    RoundReads (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs
      dg N k Rw sh w (fun j => ownAt ss j) sel c₀ :=
  roundReads_ownAt hob (sessStmt_checked hF)
    (msgNamesAt_of_accepted
      (F := C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg)
      checksAccepted_checked) hss hsel

/-- **`WRoundReads` at the checked family** (`C3Family.checked`), at `ownAt ss`, where one of its outer proofs is
accepted: `wRoundReads_ownAt` with `SessStmt` and `MsgNamesAt` as `roundReads_checked` has them. -/
theorem wRoundReads_checked {xR : HoloReg GF128 Dig} {ish : RecVStar.Inner} {ss : Array VStar.Stmt}
    {hss : RecVStar.statements ish = .ok ss} {bind : Fin (vstarCount ish) → VStar.Bindings}
    {Vs : Fin (vstarCount ish) → VStmt} {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
    {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin (vstarCount ish)) → (Vs j).OuterR Reg'}
    {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr} {coins : String → Option Flock.RecCoins.Source}
    {rows : Option (Array Flock.Registered.Port)} {own : Fin (vstarCount ish) → Array Flock.Registered.Port}
    {sel : ℕ → String → ℕ → Prop}
    {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
    {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (ℐ).Msg}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I) {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {sh : Flock.Shape} {stage : Fin 2 → ℕ → ℕ} {sess : ℕ → Fin (vstarCount ish)}
    {un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv}
    (hA : AcceptedAt (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs
      dg N k Rw)
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
