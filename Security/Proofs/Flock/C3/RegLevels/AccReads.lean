import Proofs.Flock.C3.RegLevels.OpensRegs
import Proofs.Flock.C3.PubRows.Shape
import Proofs.Flock.VBridge.Family.AccRows
import Proofs.Flock.Recursive.Raw

/-!
# n11: `OpenKeysS`' reads-file conjuncts at a level's `acc` and `acc_out`, from `checkAcc` and `checkAccReads`

`OpenKeysS` (`Family/AccRows.lean`) keys each opening's `acc` (`acc_out`) column at one registration, `rec-acc`'s, read
at the opening's row of the chain (`ColKeyedS`: the reads file names the column's port, the read's port has the
registration, and the read is at the position). A level's session's reads file gives all three where the verifier
checked its statement (`checkAcc_of_checkStmt`):

* `value_of_checkAcc`, `site_of_checkAcc`: `checkAcc` finds each of the statement's `accPorts` reading `rec-acc`, and
  every entry reading `rec-acc` has the registration (`site`) of `acc0` in the verifier's shared rows
  (`site_of_portDiff`);
* `reads_of_checkAccReads`: `checkAccReads` finds `acc` reading `accReads`' rows, and `acc_out` the next ones
  (`accReads_recOpen`: the table's level `l`'s unit `u` reads row `u / q · n + opensBelow l + u % q`;
  `accReads_recOpenStacked`: R's level `l`'s unit `u` reads row `2 · n + opensBelow' l + u`; `getElem?_accRow`);
* `accPort_of_checks`, `accPort_of_checkStmt`: so `own`'s entry of each name has `acc0`'s registration and those reads,
  where `own` names each port once (`hnd`);
* `accRead_drawn`: at a drawn unit, `OwnReadsHJ` puts a registered read of an `acc` (`acc_out`) column at `own`'s entry
  of its name, so the reads file names the column's port and the read's port has `acc0`'s registration; its position
  is `accReads`' row at the drawn statement's label of the unit (`idx`, which `OwnReadsHJ` reads), not at the unit;
* `ReadsOwnC3`, `accRead_of_checked`: where a registered read's port is `own`'s entry of its name read at the unit's
  position at every unit (`ReadsOwnC3`, `OwnReadsHJ`'s conclusion with the unit for its label), at an accepted session
  whose statements the verifier checked (`ChecksAccepted`), the read is also at `accReads`' row of its unit.

`checkAcc` and `checkAccReads` read `find?`'s first entry of a name, so both need `own`'s names unique (`hnd`), which
neither the verifier nor the family states.
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

/-- **`checkAcc`'s first loop**: each of the statement's `accPorts` is `own`'s first entry of its name, reading
`rec-acc`. -/
theorem value_of_checkAcc {s : VStar.Stmt} {own : Array Flock.Registered.Port}
    {rows : Option (Array Flock.Registered.Port)} (h : Flock.RecVStar.checkAcc s own rows = .ok ()) :
    ∀ nm ∈ Flock.RecVStar.accPorts s, ∃ p, own.find? (·.name == nm) = some p ∧ p.value = Flock.RecVStar.ACC_VALUE := by
  unfold Flock.RecVStar.checkAcc at h
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

/-- **A level's `acc` or `acc_out` entry, from `checkAcc` and `checkAccReads`**, where `own` names each port once
(`hnd`): the entry `pa` of the name has `acc0`'s registration in the verifier's shared rows and reads `accReads`' rows,
the second one on. -/
theorem accPort_of_checks {i : Flock.RecVStar.Inner} {s : VStar.Stmt} {own : Array Flock.Registered.Port}
    {rows : Option (Array Flock.Registered.Port)} (hacc : Flock.RecVStar.checkAcc s own rows = .ok ())
    (hrd : Flock.RecVStar.checkAccReads i s own = .ok ()) (hl : Flock.RecVStar.accPorts s = ["acc", "acc_out"])
    {want : Array (Option ℕ)} (hw : Flock.RecVStar.accReads i s = .ok (some want))
    (hnd : ∀ p ∈ own, ∀ p' ∈ own, p.name = p'.name → p = p') {nm : String} {d : ℕ}
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
  obtain rfl : p = pa := hnd p (Array.mem_of_find?_eq_some hf) pa hpa
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
    (hw : Flock.RecVStar.accReads i s = .ok (some want)) (hnd : ∀ p ∈ own, ∀ p' ∈ own, p.name = p'.name → p = p')
    {nm : String} {d : ℕ} (hnm : (nm, d) ∈ [("acc", 0), ("acc_out", 1)]) {pa : Flock.Registered.Port}
    (hpa : pa ∈ own) (hpn : pa.name = nm) :
    (∃ rws w, rows = some rws ∧ rws.find? (·.name == "acc0") = some w ∧
      FlockVBridge.Family.site pa = FlockVBridge.Family.site w) ∧
      pa.reads = want.map (·.map (· + d)) := by
  obtain ⟨ss, s', hss, hs', hacc, hrd⟩ := checkAcc_of_checkStmt h
  obtain rfl : s' = s := Option.some.inj (hs'.symm.trans (hj ss hss))
  exact accPort_of_checks hacc hrd hl hw hnd hnm hpa hpn

/-! ## At a drawn unit -/

section Drawn

open FlockSoundness.Discharge.ZkReg (LiveAcceptsZKR)

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

/-- **A drawn unit's registered read of a level's `acc` (`acc_out`) column, from `OwnReadsHJ`** (`zo.hown`) and the
verifier's checks of the session's statement `s` (`accPort_of_checks`): the reads file names the column's port, and the
read's port has `acc0`'s registration; its position is `accReads`' row at some index (the drawn statement's label of
the unit, `OwnReadsHJ`'s). -/
theorem accRead_drawn {ω : Lv.Ω} {R' : Reg'} {o}
    (hacc : LiveAcceptsZKR zo.I zo.own (R', Lv.draw ω, zo.wr (Lv.draw ω) R' o)) {u : Fin nv} (hu : u ∈ Lv.draw ω)
    {q : Fin c.ports.size} (hr : zo.rd.isRead (.inl (u, q))) {i : Flock.RecVStar.Inner} {s : VStar.Stmt}
    {rows : Option (Array Flock.Registered.Port)} (hck : Flock.RecVStar.checkAcc s zo.own rows = .ok ())
    (hrd : Flock.RecVStar.checkAccReads i s zo.own = .ok ()) (hl : Flock.RecVStar.accPorts s = ["acc", "acc_out"])
    {want : Array (Option ℕ)} (hw : Flock.RecVStar.accReads i s = .ok (some want))
    (hnd : ∀ p ∈ zo.own, ∀ p' ∈ zo.own, p.name = p'.name → p = p') {nm : String} {d : ℕ}
    (hnm : (nm, d) ∈ [("acc", 0), ("acc_out", 1)]) (hq : c.ports[q].name = nm) :
    (∃ pt ∈ zo.own, pt.name = c.ports[q].name) ∧
      (∃ rws w, rows = some rws ∧ rws.find? (·.name == "acc0") = some w ∧
        FlockVBridge.Family.site (zo.rd.port (.inl (u, q))) = FlockVBridge.Family.site w) ∧
      ∃ x, (want.map (·.map (· + d)))[x]? = some (some (zo.rd.readAt (.inl (u, q)))) := by
  obtain ⟨u₁, q₁, -, D, -, idx, h₁, hmem, hf, -, -, -, -, -, hpos⟩ :=
    zo.hown ω R' o hacc (.inl (u, q)) ⟨hr, hu⟩
  simp only [Sum.inl.injEq, Prod.mk.injEq] at h₁
  obtain ⟨rfl, rfl⟩ := h₁
  have hn : (zo.rd.port (.inl (u, q))).name = nm := by
    have h1 : (c.ports[q.val]!.name == (zo.rd.port (.inl (u, q))).name) = true := List.find?_some hf
    rw [getElem!_pos c.ports q.val q.isLt] at h1
    exact (beq_iff_eq.1 h1).symm.trans hq
  obtain ⟨hsite, hreads⟩ := accPort_of_checks hck hrd hl hw hnd hnm hmem hn
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

/-- (3) `OwnReadsHJ`'s conclusion at every unit, not only drawn ones: a registered read's port is `own`'s entry of its
name, read at the unit's position. -/
def ReadsOwnC3 (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ j (u : Fin (F.Vs j).nv)
    (q : Fin (F.Vs j).c.ports.size), (F.zs R ω t R₂ j).rd.isRead (.inl (u, q)) →
      (F.zs R ω t R₂ j).rd.port (.inl (u, q)) ∈ (F.zs R ω t R₂ j).own ∧
      ((F.zs R ω t R₂ j).rd.port (.inl (u, q))).name = (F.Vs j).c.ports[q].name ∧
      ((F.zs R ω t R₂ j).rd.port (.inl (u, q))).reads[u.val]? =
        some (some ((F.zs R ω t R₂ j).rd.readAt (.inl (u, q))))

/-- **`OpenKeysS`' reads-file conjuncts at an accepted session's level** (`accPort_of_checkStmt`): where the verifier
checked the session's statements (`ChecksAccepted`; the first table's, `setup_zero_c`), its reads file names each port
once (`hnd`) and a registered read's port is `own`'s entry of its name read at the unit's position (`ReadsOwnC3`), a
registered read of the level's `acc` (`acc_out`) column at unit `u` is a read of a port the reads file names, of
`acc0`'s registration in the shared rows, at `accReads`' row of `u` (the next one). -/
theorem accRead_of_checked {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hchk : FlockC3.PubRows.ChecksAccepted F rows wr reg) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    (hro : ReadsOwnC3 F dg N k Rw)
    (hnd : ∀ R ω t R₂ j, ∀ p ∈ (F.zs R ω t R₂ j).own, ∀ p' ∈ (F.zs R ω t R₂ j).own, p.name = p'.name → p = p')
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂}
    {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (j : Fin (vstarCount F.ish)) {s : VStar.Stmt}
    (hs : F.stmts[(j : ℕ)]? = some s) (hl : Flock.RecVStar.accPorts s = ["acc", "acc_out"])
    {want : Array (Option ℕ)} (hw : Flock.RecVStar.accReads F.ish s = .ok (some want)) {nm : String} {d : ℕ}
    (hnm : (nm, d) ∈ [("acc", 0), ("acc_out", 1)]) {u : Fin (F.Vs j).nv} {q : Fin (F.Vs j).c.ports.size}
    (hq : (F.Vs j).c.ports[q].name = nm) (hr : (F.zs R ω t R₂ j).rd.isRead (.inl (u, q))) :
    (∃ pt ∈ (F.zs R ω t R₂ j).own, pt.name = (F.Vs j).c.ports[q].name) ∧
      (∃ rws w, rows = some rws ∧ rws.find? (·.name == "acc0") = some w ∧
        FlockVBridge.Family.site ((F.zs R ω t R₂ j).rd.port (.inl (u, q))) = FlockVBridge.Family.site w) ∧
      (want.map (·.map (· + d)))[u.val]? = some (some ((F.zs R ω t R₂ j).rd.readAt (.inl (u, q)))) := by
  obtain ⟨_, -, -, _, _, hset, hfm⟩ := hchk R ω t R₂ j ⟨_, (h.1 j).1⟩
  obtain ⟨h0, -⟩ := setup_zero_c (F.zs R ω t R₂ j).ht (F.zs R ω t R₂ j).hTags.1 (F.zs R ω t R₂ j).hcH hset
  obtain ⟨hmem, hname, hpos⟩ := hro R ω t R₂ τ o h j u q hr
  obtain ⟨hsite, hreads⟩ := accPort_of_checkStmt (array_forM_ok hfm 0 h0)
    (fun ss hss => by rw [← Except.ok.inj (F.hstmts.symm.trans hss)]; exact hs) hl hw (hnd R ω t R₂ j) hnm hmem
    (hname.trans hq)
  refine ⟨⟨_, hmem, hname⟩, hsite, ?_⟩
  rw [← hreads]
  exact hpos

end Family

end FlockC3.RegLevels

#print axioms FlockC3.RegLevels.checkAcc_of_checkStmt
#print axioms FlockC3.RegLevels.site_of_checkAcc
#print axioms FlockC3.RegLevels.reads_of_checkAccReads
#print axioms FlockC3.RegLevels.accPort_of_checks
#print axioms FlockC3.RegLevels.accPort_of_checkStmt
#print axioms FlockC3.RegLevels.accRead_drawn
#print axioms FlockC3.RegLevels.accRead_of_checked
