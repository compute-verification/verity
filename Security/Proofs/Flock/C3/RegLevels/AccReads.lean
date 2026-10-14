import Proofs.Flock.C3.RegLevels.OpensRegs
import Proofs.Flock.C3.PubRows.Shape
import Proofs.Flock.VBridge.Family.AccRows
import Proofs.Flock.Recursive.Raw
import Proofs.Flock.Recursive.Stage

/-!
# n11: `OpenKeysS`' reads-file conjuncts at a level's `acc` and `acc_out`, from `checkAcc` and `checkAccReads`

`OpenKeysS` (`Family/AccRows.lean`) keys each opening's `acc` (`acc_out`) column at one registration, `rec-acc`'s, read
at the opening's row of the chain (`ColKeyedS`: the reads file names the column's port, the read's port has the
registration, and the read is at the position). A level's session's reads file gives all three where the verifier
checked its statement (`checkAcc_of_checkStmt`):

* `names_of_checkAcc`: `checkAcc` refuses a reads file naming a port twice (`checkNames`), so two entries of `own`
  with one name are one, which `find?`'s first entry of a name, all `checkAcc` and `checkAccReads` read, needs;
* `value_of_checkAcc`, `site_of_checkAcc`: `checkAcc` finds each of the statement's `accPorts` reading `rec-acc`, and
  every entry reading `rec-acc` has the registration (`site`) of `acc0` in the verifier's shared rows
  (`site_of_portDiff`);
* `reads_of_checkAccReads`: `checkAccReads` finds `acc` reading `accReads`' rows, and `acc_out` the next ones
  (`accReads_recOpen`: the table's level `l`'s unit `u` reads row `u / q · n + opensBelow l + u % q`;
  `accReads_recOpenStacked`: R's level `l`'s unit `u` reads row `2 · n + opensBelow' l + u`; `getElem?_accRow`);
* `accPort_of_checks`, `accPort_of_checkStmt`: so `own`'s entry of each name has `acc0`'s registration and those reads;
* `accRead_drawn`: at a drawn unit, `OwnReadsHJ` puts a registered read of an `acc` (`acc_out`) column at `own`'s entry
  of its name, so the reads file names the column's port and the read's port has `acc0`'s registration; its position
  is `accReads`' row at the drawn statement's label of the unit (`idx`, which `OwnReadsHJ` reads), not at the unit;
* `readsOwn_of_hown`: V*'s sessions draw every unit (`VLaw.live`, `execOS_subset_full_mem`), so at an accepted outcome
  the session's game reaches (`Reached`) `OwnReadsHJ` holds at every registered read: its port is `own`'s entry of its
  name, read at the drawn statement's label of the unit (`AtLabel`; `HmRow.drawn` labels unit `u` with the registered
  population's `units.indices` at `u`, which no check here holds to `u`);
* `accRead_of_checked`: so at such an outcome of a session whose statements the verifier checked (`ChecksAccepted`), the
  read is also at `accReads`' row of the unit's label. `OpenKeysS` reads it at the unit's own row, the step `RowsHeld`'s
  label conjunct makes.
-/

namespace FlockC3.RegLevels

open Flock
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion

local instance lawfulBEqByteArrayAcc : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-! ## The verifier's checks -/

/-- **Two ports `portDiff?` finds no field of, but reads, apart have one registration.** -/
theorem site_of_portDiff {a b : Flock.Registered.Port} (h : Flock.RecVStar.portDiff? a b false = none) :
    FlockVBridge.Family.site a = FlockVBridge.Family.site b := by
  have e1 : a.value = b.value := Classical.byContradiction fun hne => by
    simp [Flock.RecVStar.portDiff?, hne] at h
  have e2 : a.root = b.root := Classical.byContradiction fun hne => by
    simp [Flock.RecVStar.portDiff?, e1, hne] at h
  have e3 : a.leaves = b.leaves := Classical.byContradiction fun hne => by
    simp [Flock.RecVStar.portDiff?, e1, e2, hne] at h
  have e4 : a.words = b.words := Classical.byContradiction fun hne => by
    simp [Flock.RecVStar.portDiff?, e1, e2, e3, hne] at h
  have e5 : a.wordBits = b.wordBits := Classical.byContradiction fun hne => by
    simp [Flock.RecVStar.portDiff?, e1, e2, e3, e4, hne] at h
  have e6 : a.domain = b.domain := Classical.byContradiction fun hne => by
    simp [Flock.RecVStar.portDiff?, e1, e2, e3, e4, e5, hne] at h
  simp only [FlockVBridge.Family.site, e1, e2, e3, e4, e5, e6]

/-- **`checkAcc` names each port once** (`checkNames`): two entries of `own` with one name are one. -/
theorem names_of_checkAcc {s : VStar.Stmt} {own : Array Flock.Registered.Port}
    {rows : Option (Array Flock.Registered.Port)} (h : Flock.RecVStar.checkAcc s own rows = .ok ()) :
    ∀ p ∈ own, ∀ p' ∈ own, p.name = p'.name → p = p' := by
  unfold Flock.RecVStar.checkAcc at h
  obtain ⟨⟨⟩, h, -⟩ := FlockSoundness.Refine.ebind_ok h
  unfold Flock.RecVStar.checkNames at h
  split at h
  · cases h
  · rename_i hd
    simp only [Flock.RecVStar.dupName?, List.find?_eq_none] at hd
    have hnd : (own.toList.map (·.name)).Nodup :=
      List.nodup_iff_count_eq_one.2 fun nm hm => by simpa using hd nm hm
    intro p hp p' hp' he
    exact List.inj_on_of_nodup_map hnd (Array.mem_toList_iff.2 hp) (Array.mem_toList_iff.2 hp') he

/-- **`checkAcc`'s first loop**: each of the statement's `accPorts` is `own`'s first entry of its name, reading
`rec-acc`. -/
theorem value_of_checkAcc {s : VStar.Stmt} {own : Array Flock.Registered.Port}
    {rows : Option (Array Flock.Registered.Port)} (h : Flock.RecVStar.checkAcc s own rows = .ok ()) :
    ∀ nm ∈ Flock.RecVStar.accPorts s, ∃ p, own.find? (·.name == nm) = some p ∧ p.value = Flock.RecVStar.ACC_VALUE := by
  unfold Flock.RecVStar.checkAcc at h
  obtain ⟨⟨⟩, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, h, -⟩ := FlockSoundness.Refine.ebind_ok h
  refine FlockC3.PubRows.rows_forIn_each _ ?_ _ h
  intro nm r hr
  split at hr
  · rename_i p hf
    split at hr
    · obtain ⟨_, h2, -⟩ := FlockSoundness.Refine.ebind_ok hr
      cases h2
    · rename_i hv
      refine ⟨⟨p, hf, by simpa using hv⟩, ?_⟩
      cases hr
      rfl
  · obtain ⟨_, h2, -⟩ := FlockSoundness.Refine.ebind_ok hr
    cases h2

/-- **`checkAcc`'s registration**: every entry of `own` reading `rec-acc` has the registration of `acc0` in the
verifier's shared rows (the entries reading `rec-acc` have the first one's, and it is `acc0`'s). -/
theorem site_of_checkAcc {s : VStar.Stmt} {own : Array Flock.Registered.Port}
    {rows : Option (Array Flock.Registered.Port)} (h : Flock.RecVStar.checkAcc s own rows = .ok ())
    {b : Flock.Registered.Port} (hb : b ∈ own) (hv : b.value = Flock.RecVStar.ACC_VALUE) :
    ∃ rws w, rows = some rws ∧ rws.find? (·.name == "acc0") = some w ∧
      FlockVBridge.Family.site b = FlockVBridge.Family.site w := by
  unfold Flock.RecVStar.checkAcc at h
  obtain ⟨⟨⟩, -, h'⟩ := FlockSoundness.Refine.ebind_ok h
  clear h
  have h := h'
  clear h'
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  dsimp only at h
  have hbm : b ∈ own.filter (·.value == Flock.RecVStar.ACC_VALUE) := Array.mem_filter.2 ⟨hb, by simpa using hv⟩
  split at h
  · rename_i a ha
    obtain ⟨_, h1, h⟩ := FlockSoundness.Refine.ebind_ok h
    rw [← Array.forIn_toList] at h1
    have hab : Flock.RecVStar.portDiff? a b false = none := by
      refine FlockC3.PubRows.rows_forIn_each (fun b => Flock.RecVStar.portDiff? a b false = none) ?_ _ h1 b
        (Array.mem_toList_iff.2 hbm)
      intro b r hr
      split at hr
      · obtain ⟨_, h2, -⟩ := FlockSoundness.Refine.ebind_ok hr
        cases h2
      · rename_i hn
        refine ⟨?_, ?_⟩
        · cases hd : Flock.RecVStar.portDiff? a b false with
          | none => rfl
          | some d => exact absurd hd (hn d)
        · cases hr; rfl
    split at h
    · rename_i rws _ _
      split at h
      · rename_i w hw
        split at h
        · obtain ⟨_, h2, -⟩ := FlockSoundness.Refine.ebind_ok h
          cases h2
        · obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
          split at h
          · cases h
          · rename_i haw
            have haw' : Flock.RecVStar.portDiff? a w false = none := by
              cases hd : Flock.RecVStar.portDiff? a w false with
              | none => rfl
              | some d => exact absurd hd (haw d)
            exact ⟨rws, w, rfl, hw, (site_of_portDiff hab).symm.trans (site_of_portDiff haw')⟩
      · cases h
    · cases h
  · rename_i ha
    obtain ⟨i, hi, -⟩ := Array.mem_iff_getElem.1 hbm
    rw [Array.getElem?_eq_getElem (Nat.lt_of_le_of_lt (Nat.zero_le i) hi)] at ha
    exact (ha _ rfl).elim

/-- **`checkAccReads`' loop**: `own`'s first entries named `acc` and `acc_out` read `accReads`' rows, the second one
on. -/
theorem reads_of_checkAccReads {i : Flock.RecVStar.Inner} {s : VStar.Stmt} {own : Array Flock.Registered.Port}
    (h : Flock.RecVStar.checkAccReads i s own = .ok ()) {want : Array (Option ℕ)}
    (hw : Flock.RecVStar.accReads i s = .ok (some want)) :
    ∀ nd ∈ [("acc", 0), ("acc_out", 1)], ∃ p, own.find? (·.name == nd.1) = some p ∧
      p.reads = want.map (·.map (· + nd.2)) := by
  unfold Flock.RecVStar.checkAccReads at h
  rw [hw] at h
  obtain ⟨_, h0, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases h0
  dsimp only at h
  obtain ⟨_, h, -⟩ := FlockSoundness.Refine.ebind_ok h
  refine FlockC3.PubRows.rows_forIn_each _ ?_ _ h
  intro ⟨nm, d⟩ r hr
  dsimp only at hr
  split at hr
  · rename_i p hf
    split at hr
    · obtain ⟨_, h2, -⟩ := FlockSoundness.Refine.ebind_ok hr
      cases h2
    · rename_i hne
      refine ⟨⟨p, hf, by simpa using hne⟩, ?_⟩
      cases hr
      rfl
  · obtain ⟨_, h2, -⟩ := FlockSoundness.Refine.ebind_ok hr
    cases h2

/-- **The table's level `l`'s rows of the chain** (`accReads` at `.recOpen`): unit `u` reads row
`u / q · n + opensBelow l + u % q`. -/
theorem accReads_recOpen {i : Flock.RecVStar.Inner} {sch : Flock.Schedule} (hsch : Flock.fast100 i.m = some sch)
    (l LANES H q : ℕ) : Flock.RecVStar.accReads i (.recOpen l LANES H q) =
      .ok (some ((Array.range (2 * (sch.levels[l]!).q)).map fun u => some (u / (sch.levels[l]!).q *
        Flock.RecVStar.opensBelow sch sch.levels.size + Flock.RecVStar.opensBelow sch l +
          u % (sch.levels[l]!).q))) := by
  simp only [Flock.RecVStar.accReads, hsch]
  rfl

/-- **R's level `l`'s rows of the chain** (`accReads` at `.recOpenStacked`): unit `u` reads row
`2 · n + opensBelow' l + u`, past both reps' openings. -/
theorem accReads_recOpenStacked {i : Flock.RecVStar.Inner} {sch sr : Flock.Schedule}
    {r : Flock.RecVStar.StackedShape} (hsch : Flock.fast100 i.m = some sch) (hr : i.stacked = some r)
    (hsr : Flock.fast100 r.m = some sr) (l LANES H q : ℕ) :
    Flock.RecVStar.accReads i (.recOpenStacked l LANES H q) =
      .ok (some ((Array.range (sr.levels[l]!).q).map fun u =>
        some (2 * Flock.RecVStar.opensBelow sch sch.levels.size + Flock.RecVStar.opensBelow sr l + u))) := by
  simp only [Flock.RecVStar.accReads, hsch, hr, hsr]
  rfl

/-- A unit's row in `accReads`' list, `d` on. -/
theorem getElem?_accRow (f : ℕ → ℕ) {m d u : ℕ} (hu : u < m) :
    (((Array.range m).map fun u => some (f u)).map (·.map (· + d)))[u]? = some (some (f u + d)) := by
  simp [hu]

/-- **A level's `acc` or `acc_out` entry, from `checkAcc` and `checkAccReads`**, `own` naming each port once
(`names_of_checkAcc`): the entry `pa` of the name has `acc0`'s registration in the verifier's shared rows and reads
`accReads`' rows, the second one on. -/
theorem accPort_of_checks {i : Flock.RecVStar.Inner} {s : VStar.Stmt} {own : Array Flock.Registered.Port}
    {rows : Option (Array Flock.Registered.Port)} (hacc : Flock.RecVStar.checkAcc s own rows = .ok ())
    (hrd : Flock.RecVStar.checkAccReads i s own = .ok ()) (hl : Flock.RecVStar.accPorts s = ["acc", "acc_out"])
    {want : Array (Option ℕ)} (hw : Flock.RecVStar.accReads i s = .ok (some want)) {nm : String} {d : ℕ}
    (hnm : (nm, d) ∈ [("acc", 0), ("acc_out", 1)]) {pa : Flock.Registered.Port} (hpa : pa ∈ own)
    (hpn : pa.name = nm) :
    (∃ rws w, rows = some rws ∧ rws.find? (·.name == "acc0") = some w ∧
      FlockVBridge.Family.site pa = FlockVBridge.Family.site w) ∧
      pa.reads = want.map (·.map (· + d)) := by
  have hnm' : nm ∈ Flock.RecVStar.accPorts s := by
    rw [hl]
    simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hnm
    rcases hnm with ⟨rfl, -⟩ | ⟨rfl, -⟩ <;> simp
  obtain ⟨p, hf, hv⟩ := value_of_checkAcc hacc nm hnm'
  obtain rfl : p = pa := names_of_checkAcc hacc p (Array.mem_of_find?_eq_some hf) pa hpa
    ((by simpa using Array.find?_some hf : p.name = nm).trans hpn.symm)
  obtain ⟨p', hf', hr⟩ := reads_of_checkAccReads hrd hw (nm, d) hnm
  obtain rfl : p' = p := Option.some.inj (hf'.symm.trans hf)
  exact ⟨site_of_checkAcc hacc hpa hv, hr⟩

/-- **`accPort_of_checks` at a statement the verifier checked as V*'s statement `j`** (`checkAcc_of_checkStmt`), the
list's `s` (`hj`). -/
theorem accPort_of_checkStmt {i : Flock.RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {own : Array Flock.Registered.Port} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (h : Flock.RecVStar.checkStmt (some i) tags tables st circuitB partition program (some own) (some j) (some f)
      rows = .ok ()) {s : VStar.Stmt}
    (hj : ∀ ss : Array VStar.Stmt, Flock.RecVStar.statements i = .ok ss → ss[j]? = some s)
    (hl : Flock.RecVStar.accPorts s = ["acc", "acc_out"]) {want : Array (Option ℕ)}
    (hw : Flock.RecVStar.accReads i s = .ok (some want)) {nm : String} {d : ℕ}
    (hnm : (nm, d) ∈ [("acc", 0), ("acc_out", 1)]) {pa : Flock.Registered.Port} (hpa : pa ∈ own) (hpn : pa.name = nm) :
    (∃ rws w, rows = some rws ∧ rws.find? (·.name == "acc0") = some w ∧
      FlockVBridge.Family.site pa = FlockVBridge.Family.site w) ∧
      pa.reads = want.map (·.map (· + d)) := by
  obtain ⟨ss, s', hss, hs', hacc, hrd⟩ := checkAcc_of_checkStmt h
  obtain rfl : s' = s := Option.some.inj (hs'.symm.trans (hj ss hss))
  exact accPort_of_checks hacc hrd hl hw hnm hpa hpn

/-! ## At a drawn unit -/

section Drawn

open FlockSoundness.Discharge.ZkReg (LiveAcceptsZKR)

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

/-- An outcome the session's game reaches: the registration sent, the law's draw at some coins, and the transcript the
session writes there (`ZkOuter.game`'s outcomes, `prob_auditLive_liveOfZJ_mono`'s). -/
def Reached (o : Reg' × Finset (Fin nv) × Exec.Transcript) : Prop :=
  ∃ ω R' x, o = (R', Lv.draw ω, zo.wr (Lv.draw ω) R' x)

/-- **A drawn unit's registered read of a level's `acc` (`acc_out`) column, from `OwnReadsHJ`** (`zo.hown`) and the
verifier's checks of the session's statement `s` (`accPort_of_checks`): the reads file names the column's port, and the
read's port has `acc0`'s registration; its position is `accReads`' row at some index (the drawn statement's label of
the unit, `OwnReadsHJ`'s). -/
theorem accRead_drawn {ω : Lv.Ω} {R' : Reg'} {o}
    (hacc : LiveAcceptsZKR zo.I zo.own (R', Lv.draw ω, zo.wr (Lv.draw ω) R' o)) {u : Fin nv} (hu : u ∈ Lv.draw ω)
    {q : Fin c.ports.size} (hr : zo.rd.isRead (.inl (u, q))) {i : Flock.RecVStar.Inner} {s : VStar.Stmt}
    {rows : Option (Array Flock.Registered.Port)} (hck : Flock.RecVStar.checkAcc s zo.own rows = .ok ())
    (hrd : Flock.RecVStar.checkAccReads i s zo.own = .ok ()) (hl : Flock.RecVStar.accPorts s = ["acc", "acc_out"])
    {want : Array (Option ℕ)} (hw : Flock.RecVStar.accReads i s = .ok (some want)) {nm : String} {d : ℕ}
    (hnm : (nm, d) ∈ [("acc", 0), ("acc_out", 1)]) (hq : c.ports[q].name = nm) :
    (∃ pt ∈ zo.own, pt.name = c.ports[q].name) ∧
      (∃ rws w, rows = some rws ∧ rws.find? (·.name == "acc0") = some w ∧
        FlockVBridge.Family.site (zo.rd.port (.inl (u, q))) = FlockVBridge.Family.site w) ∧
      ∃ x : ℕ, (want.map (·.map (· + d)))[x]? = some (some (zo.rd.readAt (.inl (u, q)))) := by
  obtain ⟨u₁, q₁, -, D, -, idx, h₁, hmem, hf, -, -, -, -, -, hpos⟩ :=
    zo.hown ω R' o hacc (.inl (u, q)) ⟨hr, hu⟩
  simp only [Sum.inl.injEq, Prod.mk.injEq] at h₁
  obtain ⟨rfl, rfl⟩ := h₁
  have hn : (zo.rd.port (.inl (u, q))).name = nm := by
    have h1 := List.find?_some hf
    rw [beq_iff_eq, getElem!_pos c.ports q.val q.isLt] at h1
    exact h1.symm.trans hq
  obtain ⟨hsite, hreads⟩ := accPort_of_checks hck hrd hl hw hnm hmem hn
  refine ⟨⟨_, hmem, hn.trans hq.symm⟩, hsite, idx.getD (D.proved.idxOf u.val) 0, ?_⟩
  rw [← hreads]
  exact hpos

end Drawn

/-! ## At the family's accepted sessions -/

section Family

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **`reads` reads `pos` at the drawn statement's label of unit `u`** (`OwnReadsHJ`'s position): at the draw the
transcript `tr` records, `u`'s table's drawn statement labels `u` with `idx.getD (D.proved.idxOf u.val) 0`
(`Registered.unitIndices`), which `loadPublic` holds only to ascending, so not to `u` (`RowsHeld` ends with this
label's read). -/
def AtLabel (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂)
    (j : Fin (vstarCount F.ish)) (tr : Exec.Transcript) (u : Fin (F.Vs j).nv) (reads : Array (Option ℕ)) (pos : ℕ) :
    Prop :=
  ∃ d D dp idx, recordDraw tr.record = some d ∧
    Draw.UnitDraw.ofJson (drawJ (F.zs R ω t R₂ j).I d (partOf (F.zs R ω t R₂ j).I d u.val)) = .ok D ∧
    HmOut.drawn (F.Vs j).c (F.zs R ω t R₂ j).pub (drawJ (F.zs R ω t R₂ j).I d (partOf (F.zs R ω t R₂ j).I d u.val)) =
      .ok dp ∧ Registered.unitIndices dp = .ok idx ∧ u.val ∈ D.proved ∧
    reads[idx.getD (D.proved.idxOf u.val) 0]? = some (some pos)

/-- **`OwnReadsHJ`'s conclusion at every unit, from `hown`**: V*'s sessions draw every unit (`VLaw.live`, the full draw,
`execOS_subset_full_mem`), so at an accepted outcome a session's game reaches (`Reached`) every registered read is a
drawn unit's: its port is `own`'s entry of its name, read at the unit's label (`AtLabel`). -/
theorem readsOwn_of_hown {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (j : Fin (vstarCount F.ish))
    (hre : Reached (F.zs R ω t R₂ j) (o j)) (u : Fin (F.Vs j).nv) (q : Fin (F.Vs j).c.ports.size)
    (hr : (F.zs R ω t R₂ j).rd.isRead (.inl (u, q))) :
    (F.zs R ω t R₂ j).rd.port (.inl (u, q)) ∈ (F.zs R ω t R₂ j).own ∧
      ((F.zs R ω t R₂ j).rd.port (.inl (u, q))).name = (F.Vs j).c.ports[q].name ∧
      AtLabel F R ω t R₂ j (o j).2.2 u ((F.zs R ω t R₂ j).rd.port (.inl (u, q))).reads
        ((F.zs R ω t R₂ j).rd.readAt (.inl (u, q))) := by
  obtain ⟨ω', R', x, ho⟩ := hre
  have hacc := (h.1 j).1
  rw [ho] at hacc
  obtain ⟨u₁, q₁, d, D, dp, idx, h₁, hmem, hf, hd, hD, hdp, hidx, hu, hpos⟩ :=
    (F.zs R ω t R₂ j).hown ω' R' x hacc (.inl (u, q)) ⟨hr, execOS_subset_full_mem (F.Vs j).law.src ω' u⟩
  simp only [Sum.inl.injEq, Prod.mk.injEq] at h₁
  obtain ⟨rfl, rfl⟩ := h₁
  have h1 := List.find?_some hf
  rw [beq_iff_eq, getElem!_pos (F.Vs j).c.ports q.val q.isLt] at h1
  rw [ho]
  exact ⟨hmem, h1.symm, d, D, dp, idx, hd, hD, hdp, hidx, hu, hpos⟩

/-- **`OpenKeysS`' reads-file conjuncts at an accepted session's level** (`accPort_of_checkStmt`): where the verifier
checked the session's statements (`ChecksAccepted`; the first table's, `setup_zero_c`), at an accepted outcome the
session reaches (`Reached`), a registered read of the level's `acc` (`acc_out`) column at unit `u` is a read of a port
the reads file names, of `acc0`'s registration in the shared rows, at `accReads`' row (the next one) of `u`'s label
(`AtLabel`). `OpenKeysS` reads it at `u`'s own row, so that step is `RowsHeld`'s, the read at the label being the read
at the unit. -/
theorem accRead_of_checked {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hchk : FlockC3.PubRows.ChecksAccepted F rows wr reg) {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (j : Fin (vstarCount F.ish))
    (hre : Reached (F.zs R ω t R₂ j) (o j)) {s : VStar.Stmt}
    (hs : F.stmts[(j : ℕ)]? = some s) (hl : Flock.RecVStar.accPorts s = ["acc", "acc_out"])
    {want : Array (Option ℕ)} (hw : Flock.RecVStar.accReads F.ish s = .ok (some want)) {nm : String} {d : ℕ}
    (hnm : (nm, d) ∈ [("acc", 0), ("acc_out", 1)]) {u : Fin (F.Vs j).nv} {q : Fin (F.Vs j).c.ports.size}
    (hq : (F.Vs j).c.ports[q].name = nm) (hr : (F.zs R ω t R₂ j).rd.isRead (.inl (u, q))) :
    (∃ pt ∈ (F.zs R ω t R₂ j).own, pt.name = (F.Vs j).c.ports[q].name) ∧
      (∃ rws w, rows = some rws ∧ rws.find? (·.name == "acc0") = some w ∧
        FlockVBridge.Family.site ((F.zs R ω t R₂ j).rd.port (.inl (u, q))) = FlockVBridge.Family.site w) ∧
      AtLabel F R ω t R₂ j (o j).2.2 u (want.map (·.map (· + d))) ((F.zs R ω t R₂ j).rd.readAt (.inl (u, q))) := by
  obtain ⟨_, -, -, _, _, hset, hfm⟩ := hchk R ω t R₂ j ⟨_, (h.1 j).1⟩
  obtain ⟨h0, -⟩ := setup_zero_c (F.zs R ω t R₂ j).ht (F.zs R ω t R₂ j).hTags.1 (F.zs R ω t R₂ j).hcH hset
  obtain ⟨hmem, hname, hpos⟩ := readsOwn_of_hown h j hre u q hr
  obtain ⟨hsite, hreads⟩ := accPort_of_checkStmt (array_forM_ok hfm 0 h0)
    (fun ss hss => by rw [← Except.ok.inj (F.hstmts.symm.trans hss)]; exact hs) hl hw hnm hmem (hname.trans hq)
  refine ⟨⟨_, hmem, hname⟩, hsite, ?_⟩
  rw [← hreads]
  exact hpos

end Family

end FlockC3.RegLevels
