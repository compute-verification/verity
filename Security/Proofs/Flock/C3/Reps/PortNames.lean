import Proofs.Flock.C3.Reps.ReadWhole
import Proofs.Flock.C3.Reps.WRead
import Proofs.Flock.C3.Compose.RsFacts

/-!
# The checks' port names: `RsKeys` and `WSel`

`RecOpening.check` and `RecResiduals.check` refuse a part whose circuit keys a bit of a row-port element by a port not
named as its round (`RecResiduals.misnamed`): `m<i>` for claim `i`'s ring switch (`RecOpening.rsName`), `m<r>` for each
round `r` holding an element of w (`RecResiduals.roundNames`).

* `keyPort_of_srcKey`: at `unitLaid`'s input group (`inGroups_laid`), the port `Aliased.srcKey` keys a column by is
  `keyPort`'s; `name_of_misnamed`, `name_of_srcKey`: a part `misnamed` passed has each name its element needs there.
* `rsKeys`: `RsKeys` holds outright, `OpenCopy` carrying the opening's check.
* `wSel_of_rounds`: `WSel` at `wLoc`'s columns and the sent keys, where each part's session passed `RecResiduals.check`
  at the shape, each message's keys are its round's, and the reads file reads round `m`'s commitment at port `m<m>`
  where its stage reads it (`WRoundReads`).
-/

namespace FlockC3.Reps.PortNames

open Flock FlockSoundness FlockSoundness.Discharge

/-- `unitLaid`'s input group: one 16-bit port per word from column 0. -/
theorem portCols_laid (w N : ℕ) :
    (portCols #[(⟨0, w, Array.replicate N 16⟩ : Group)]).toList = (List.range N).map fun j => (16 * j, 16) := by
  rw [Aliased.portCols_one, HiddenExec.portCols_16 _ _ fun b hb => by simp at hb; exact hb.2]
  simp

theorem getElem!_laid {w N p : ℕ} (hp : p < N) :
    (portCols #[(⟨0, w, Array.replicate N 16⟩ : Group)])[p]! = (16 * p, 16) := by
  have hl := portCols_laid w N
  have hs : p < (portCols #[(⟨0, w, Array.replicate N 16⟩ : Group)]).size := by
    rw [← Array.length_toList, hl]
    simpa using hp
  rw [getElem!_pos _ p hs, ← Array.getElem_toList]
  simp [hl]

/-- **A stacked input keyed by a message bit is keyed by `keyPort`'s port**, at a unit whose input group is
`unitLaid`'s: copy `k`'s column `col` below the inputs' end. -/
theorem keyPort_of_srcKey {c : Circuit} (hnd : (portBits c.unit.inGroups).Nodup) {w N : ℕ}
    (hg : c.unit.inGroups = #[⟨0, w, Array.replicate N 16⟩]) {k col : ℕ} (hcol : col < c.unit.inWords * WORD)
    {q byte bit : ℕ} (h : Aliased.srcKey c (k * (c.unit.inWords * WORD) + col) = .msg q byte bit) :
    RecResiduals.keyPort c k col = some q := by
  have hW : 0 < c.unit.inWords * WORD := by omega
  have hmod : (k * (c.unit.inWords * WORD) + col) % (c.unit.inWords * WORD) = col := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hcol]
  have hdiv : (k * (c.unit.inWords * WORD) + col) / (c.unit.inWords * WORD) = k := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hW, Nat.div_eq_of_lt hcol, Nat.zero_add]
  unfold Aliased.srcKey at h
  rw [hmod, hdiv] at h
  by_cases hm : col ∈ portBits c.unit.inGroups
  · rw [ite_eq_left hm] at h
    obtain ⟨p, t, hp, ht, hpt⟩ := Aliased.mem_portBits hm
    rw [hpt, portOf_eq hnd hp ht, Nat.add_sub_cancel_left] at h
    have hsz : (portCols c.unit.inGroups).size = N := by
      rw [← Array.length_toList, hg, portCols_laid]
      simp
    rw [hsz] at hp
    rw [hg, getElem!_laid hp] at ht hpt
    have h1 : col / 16 = p := by omega
    have h2 : col % 16 = t := by omega
    have hpN : p < (c.unit.inGroups.getD 0 default).bits.size := by
      rw [hg]
      simpa using hp
    unfold RecResiduals.keyPort
    rw [h1, h2]
    unfold Aliased.srcAt at h
    rw [ite_eq_left hpN] at h ⊢
    by_cases hl : (c.leavesIn.getD k #[])[p]! < 0
    · rw [ite_eq_left hl] at h
      cases h
    · rw [ite_eq_right hl] at h ⊢
      have hts : Typed.starts c = RecResiduals.leafStarts c := rfl
      dsimp only at h ⊢
      rw [hts] at h
      split at h
      · rename_i hb
        rw [ite_eq_left hb]
        exact congrArg some (Aliased.Src.msg.inj h).1
      · cases h
  · rw [ite_eq_right hm] at h
    cases h

/-- **`misnamed` passes only where each keyed bit's port has each name its element needs.** -/
theorem name_of_misnamed {c : Circuit} {ns : List ℕ} {n : ℕ} {need : ℕ → List String}
    (h : RecResiduals.misnamed c ns n need = none) {k a b q : ℕ} (hk : k < max 1 c.upv) (ha : a < n) (hb : b < 128)
    (hq : RecResiduals.keyPort c k (RecResiduals.rowCol ns 0 (128 * a + b)) = some q) :
    ∀ nm ∈ need a, c.ports[q]!.name = nm := by
  unfold RecResiduals.misnamed at h
  have h1 := List.findSome?_eq_none_iff.1 h k (List.mem_range.2 hk)
  have h2 := List.findSome?_eq_none_iff.1 h1 a (List.mem_range.2 ha)
  have h3 := List.findSome?_eq_none_iff.1 h2 b (List.mem_range.2 hb)
  rw [hq] at h3
  dsimp only at h3
  split at h3
  · rename_i hall
    intro nm hnm
    exact (beq_iff_eq.1 (List.all_eq_true.1 hall nm hnm)).symm
  · cases h3

/-- `roundNames` names round `m` for a key round `m` holds. -/
theorem mem_roundNames {rounds : Array (Array RecAlgebra.Key)} {m : ℕ} {x : RecAlgebra.Key}
    (hx : x ∈ rounds[m]!) : s!"m{m}" ∈ RecResiduals.roundNames rounds (some x) := by
  have hm : m < rounds.size := by
    by_contra hm
    rw [getElem!_neg rounds m hm, show (default : Array RecAlgebra.Key) = #[] from rfl] at hx
    simp at hx
  unfold RecResiduals.roundNames
  exact List.mem_map.2 ⟨m, List.mem_filter.2 ⟨List.mem_range.2 hm, Array.contains_iff_mem.2 hx⟩, rfl⟩

/-- `rsName` names round `i` for claim `i`'s ring switch's element. -/
theorem rsName_eq {sh : Shape} {W : Array ℕ} {a i u : ℕ} (h : (RecOpening.wNames sh)[W[a]!]? = some (.rs i u)) :
    RecOpening.rsName sh W a = [s!"m{i}"] := by
  unfold RecOpening.rsName
  rw [h]

/-- `unitLaid`'s net's input group is one 16-bit port per word from column 0. -/
theorem inGroups_laid {S : RecOpen.GfS} {ns : List ℕ} {nv : ℕ} {name : String} {net : Net}
    (hn : HmNets.netOf name (RecResiduals.unitLaid S ns nv) = .ok net) :
    ∃ w N, net.inGroups = #[⟨0, w, Array.replicate N 16⟩] :=
  ⟨_, _, (Net.ofRows_ok hn).2.2.2.2.2.1.trans (Hm.layout_inGroups _ _ _ _)⟩

/-- **A keyed bit of a row-port element that `misnamed` passed is keyed by a port with each name the element needs**
(`keyPort_of_srcKey`, `name_of_misnamed`). -/
theorem name_of_srcKey {c : Circuit} (hnd : (portBits c.unit.inGroups).Nodup) {w N : ℕ}
    (hg : c.unit.inGroups = #[⟨0, w, Array.replicate N 16⟩]) {ns : List ℕ} {n : ℕ} {need : ℕ → List String}
    (h : RecResiduals.misnamed c ns n need = none) {k a b : ℕ} (hk : k < max 1 c.upv) (ha : a < n) (hb : b < 128)
    (hcol : RecResiduals.rowCol ns 0 (128 * a + b) < c.unit.inWords * WORD) {q byte bit : ℕ}
    (hkey : Aliased.srcKey c (k * (c.unit.inWords * WORD) + RecResiduals.rowCol ns 0 (128 * a + b)) =
      .msg q byte bit) :
    ∀ nm ∈ need a, c.ports[q]!.name = nm :=
  name_of_misnamed h hk ha hb (keyPort_of_srcKey hnd hg hcol hkey)

end FlockC3.Reps.PortNames

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.OfVStar (rowCol partRep)
open FlockVBridge.PartS (wIdx)

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **Ring switch `i`'s columns are keyed by port `m<i>`** (`RsKeys`): `OpenCopy` carries the opening's check, which
refuses a unit keying a bit of claim `i`'s ring-switch element by a port not named `m<i>` (`name_of_srcKey` at
`rsName`). -/
theorem rsKeys {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128} :
    RsKeys zs dg N k Rw sh w := by
  intro i _ gs s un j p hO S0 hS hg hpg S₁ ns nv hpart q e t he hwi b hb q' kk _ hkey
  obtain ⟨-, hcol, -⟩ := openCopy_key hO hS hg hpg hpart he hb
  have hrc : rowCol (partRep ns nv).ports 0 (128 * q + b) = RecResiduals.rowCol ns 0 (128 * q + b) :=
    (FlockC3.Reps.ReadWhole.rowCol_eq ns 0 _).symm
  rw [FlockSoundness.Discharge.Layout.ofNet_nIn_eq, hrc] at hkey hcol
  obtain ⟨_, _, _, _, _, _, _, -, hz⟩ := hO
  obtain ⟨_, hj, _, hc, -, m, cl, hpm, hsh, -⟩ := hz
  obtain ⟨S₂, ns₂, nv₂, W, hpart₂, hn, hW, -, -, hmis⟩ := FlockC3.Reps.ReadWhole.check_rsWhole hc hpm hsh
  obtain ⟨rfl, rfl, rfl⟩ : S₁ = S₂ ∧ ns = ns₂ ∧ nv = nv₂ := by
    simpa only [Prod.mk.injEq] using Except.ok.inj (hpart.symm.trans hpart₂)
  obtain rfl := FlockC3.Reps.ReadWhole.partW_eq hS hg hpg hW
  obtain ⟨hq, hqe⟩ := Array.getElem?_eq_some_iff.1 he
  have hk : (RecOpening.wNames sh)[(wIdx (RecOpening.portSizes sh) gs[p].2.1)[q]!]? = some (.rs i t) := by
    rw [getElem!_pos (wIdx (RecOpening.portSizes sh) gs[p].2.1) q hq, hqe]
    exact FlockC3.Reps.ReadWhole.getElem?_of_wi hwi
  obtain ⟨_, _, hig⟩ := FlockC3.Reps.PortNames.inGroups_laid hn
  have hname := FlockC3.Reps.PortNames.name_of_srcKey (FlockSoundness.Discharge.Aliased.portBits_nodup (Vs s).hU)
    hig hmis (Nat.lt_of_lt_of_le hj (Nat.le_max_right _ _)) hq hb hcol hkey s!"m{i}"
    (by rw [FlockC3.Reps.PortNames.rsName_eq hk]; exact List.mem_singleton_self _)
  rw [getElem!_pos (Vs s).c.ports q'.val q'.isLt] at hname
  exact hname

end

end FlockC3.Compose

namespace FlockC3.Reps

open Flock
open FlockVBridge.Family (partsOf wIOf nsOf nvOf rowColOf partsOf_eq)
open FlockVBridge.OfVStar (rowCol partRep)
open FlockSoundness.Discharge.Recursion (VStmt)

/-- **Rep `ρ`'s stage of round `m` reads each port keying a column of an element of `w` part `p` holds**, where round
`m` holds the element's key: the part's session passed `RecResiduals.check`, which refuses a unit keying a bit of it by
a port not named `m<m>` (`name_of_srcKey` at `roundNames`), and the reads file reads port `m<m>` where the stage reads
it (`WRoundReads`). -/
theorem selPort_of_part {S : ℕ} {Vs : Fin S → VStmt} {sh : Flock.Shape} {sess : ℕ → Fin S}
    {un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv} {own : Fin S → Array Flock.Registered.Port}
    {sel : ℕ → String → ℕ → Prop} {stage : Fin 2 → ℕ → ℕ}
    (hck : ∀ p < (partsOf sh).size, RecResiduals.check (Vs (sess p)).c = .ok () ∧
      (Vs (sess p)).c.unit.unitLog ≤ 32 ∧ ∃ m k cl sh₀,
        RecResiduals.params (Vs (sess p)).c.unit.name = some (m, k, cl, p) ∧ RecResiduals.shape? m k cl = some sh₀ ∧
        RecResiduals.holoOf (Vs (sess p)).c = .ok sh.holo ∧ sh = { sh₀ with holo := sh.holo })
    (hr : FlockC3.Compose.WRoundReads sh own sel stage sess un) {p a i : ℕ} (hp : p < (partsOf sh).size)
    (ha : (wIOf sh p)[a]? = some i) {x : RecAlgebra.Key} (hwi : RecAlgebra.wi sh x = some i) {m : ℕ}
    (hxm : x ∈ (RecAlgebra.rounds sh)[m]!) (ρ : Fin 2) {b : ℕ} (hb : b < 128) (q : Fin (Vs (sess p)).c.ports.size)
    {kq : ℕ} (hkey : FlockSoundness.Discharge.Aliased.srcKey (Vs (sess p)).c
      (0 * (FlockSoundness.Rows.ofNet (Vs (sess p)).hU).nIn + rowColOf sh p (128 * a + b)) =
        .msg q.val (kq / 8) (kq % 8)) :
    SelPort Vs own sel (stage ρ m) (sess p) (un p ρ) q := by
  obtain ⟨hc, -, m', k', cl, sh₀, hpar, hsh, hh, hsheq⟩ := hck p hp
  obtain ⟨S₁, ns, nv, hpart, hn, -, W, hW, hmis⟩ := ReadWhole.check_whole hc hpar hsh hh
  rw [← hsheq] at hpart hW hmis
  obtain ⟨S0, gs, hS, hg⟩ := ReadWhole.part_ok hpart
  have hpo := partsOf_eq hS hg
  have hpg : p < gs.size := by rw [← hpo]; exact hp
  obtain ⟨-, hns, hnv⟩ := FlockVBridge.OfVStar.part_eq hS hg hpg hpart
  have hgp : (partsOf sh)[p]! = gs[p] := by rw [hpo]; exact getElem!_pos gs p hpg
  obtain rfl := ReadWhole.partW_eqR hS hg hpg hW
  have hWI : wIOf sh p = FlockVBridge.PartS.wIdx (RecResiduals.portSizes sh) gs[p].2.1 := by
    unfold wIOf
    rw [hgp]
  rw [hWI] at ha
  obtain ⟨haW, hWa⟩ := Array.getElem?_eq_some_iff.1 ha
  have hx : (RecAlgebra.wNames sh)[(FlockVBridge.PartS.wIdx (RecResiduals.portSizes sh) gs[p].2.1)[a]!]? =
      some x := by
    rw [getElem!_pos (FlockVBridge.PartS.wIdx (RecResiduals.portSizes sh) gs[p].2.1) a haW, hWa]
    exact ReadWhole.getElem?_of_wiA hwi
  have hsz : (FlockVBridge.PartS.wIdx (RecResiduals.portSizes sh) gs[p].2.1).size = ns.sum := by
    rw [FlockVBridge.OfVStar.wIdx_size, hns]
  have hcol := FlockVBridge.OfVStar.rowCol_lt_nIn (Vs (sess p)).hU hn (q := 128 * a + b) (by omega)
  have hrc₀ : rowColOf sh p (128 * a + b) = rowCol (partRep ns nv).ports 0 (128 * a + b) := by
    unfold rowColOf nsOf nvOf
    rw [hgp, hns, hnv]
  have hrc : rowCol (partRep ns nv).ports 0 (128 * a + b) = RecResiduals.rowCol ns 0 (128 * a + b) :=
    (ReadWhole.rowCol_eq ns 0 _).symm
  rw [FlockSoundness.Discharge.Layout.ofNet_nIn_eq, hrc] at hcol
  rw [FlockSoundness.Discharge.Layout.ofNet_nIn_eq, hrc₀, hrc] at hkey
  obtain ⟨_, _, hig⟩ := PortNames.inGroups_laid hn
  have hname := PortNames.name_of_srcKey (FlockSoundness.Discharge.Aliased.portBits_nodup (Vs (sess p)).hU) hig
    hmis (Nat.lt_of_lt_of_le Nat.zero_lt_one (Nat.le_max_left _ _)) haW hb hcol hkey s!"m{m}"
    (by show s!"m{m}" ∈ RecResiduals.roundNames _ _; rw [hx]; exact PortNames.mem_roundNames hxm)
  rw [getElem!_pos (Vs (sess p)).c.ports q.val q.isLt] at hname
  obtain ⟨pt, hpt, hn₁, -, y, hy, hsy⟩ := hr p hp m ρ q hname
  exact ⟨pt, hpt, hn₁.trans hname.symm, y, hy, hsy⟩

/-- **The reads file reads each sent key's ports at `wLoc`'s columns** (`WSel` at the sent keys): where each part's
session passed `RecResiduals.check` at the shape, each message's keys are its round's, and the reads file reads round
`m`'s commitment at port `m<m>` where rep `ρ`'s stage of it reads it (`selPort_of_part` at `wLoc`'s part). -/
theorem wSel_of_rounds {S : ℕ} {Vs : Fin S → VStmt} {sh : Flock.Shape} {sess : ℕ → Fin S}
    {un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv} {own : Fin S → Array Flock.Registered.Port}
    {sel : ℕ → String → ℕ → Prop} {σ : List FlockSoundness.Discharge.Exec.Kind} {c₀ : ℕ}
    {keys : ℕ → List RecAlgebra.Key}
    (hck : ∀ p < (partsOf sh).size, RecResiduals.check (Vs (sess p)).c = .ok () ∧
      (Vs (sess p)).c.unit.unitLog ≤ 32 ∧ ∃ m k cl sh₀,
        RecResiduals.params (Vs (sess p)).c.unit.name = some (m, k, cl, p) ∧ RecResiduals.shape? m k cl = some sh₀ ∧
        RecResiduals.holoOf (Vs (sess p)).c = .ok sh.holo ∧ sh = { sh₀ with holo := sh.holo })
    (hkr : ∀ m, ∀ x ∈ keys m, x ∈ (RecAlgebra.rounds sh)[m]!)
    (hr : FlockC3.Compose.WRoundReads sh own sel (stageOf σ c₀) sess un) :
    WSel sh own sel (fun _ => wSes Vs sh sess) (wUn Vs sh sess un) (fun _ => wCols Vs sh sess) (sentK σ c₀ keys) := by
  rintro s ρ x i ⟨m, rfl, hxm, -⟩ - hwi c ⟨hex, b, hb, rfl⟩ q kq - hkey
  have hloc : WLoc Vs sh sess i (wLoc Vs sh sess i) := by
    unfold wLoc
    rw [dite_eq_left hex]
    exact Classical.choose_spec hex
  exact selPort_of_part hck hr hloc.1 hloc.2.1 hwi (hkr m x hxm) ρ hb q hkey

end FlockC3.Reps
