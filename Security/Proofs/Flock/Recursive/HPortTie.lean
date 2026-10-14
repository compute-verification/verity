import Proofs.Flock.C3.Compose.Tied
import Proofs.Flock.C3.Sparse.SessionAt
import Proofs.Flock.VBridge.Family.Registrants
import Proofs.Flock.C3.RegLevels.SharedRows
import Std.Data.String.ToNat

/-!
# The sparse phase's `h` from the rep's, across two units' reads of one registered row

The RecSparse unit reads its `h` port under `rec-c{hr}` at row `i` (`RecVStar.ownPorts (.sparse _ hr)`, which
`checkOwn` enforces), and rep `i`'s algebra unit reads the same registration's row `i` as its `m<hr>` port. Two
sessions' registered reads at one root and position are one row's bytes or a SHA-512 collision between the two
registrant records (`registrants_eq_or_collides`), and that collision is `RegColl`'s (`ReadsX.Clash`), which `Accepted`
excludes and `regFinder` prices once per audit: no new event, no guarantee record moved.

* `regRows_eq_or_clash`, `regVal_bit_eq`: at any sessions, two registered reads at one registration and position hold
  one row's bytes or two records clash (both preimages kept, `registrants_eq_or_collides_site`: the sparse unit's `h`
  and the algebra unit's `m<hr>` differ by name); with no clash and both rows unsegmented, one row's bits.
* `rep_of_unit_h`, `rep_h_of_unit`: a rep unit plays `holoRepC` to an output whose `h` is the unit's `hOfW` (what
  `holoRep_refines` gives and `rep_of_unit` drops), so the tied run's rep `i` has that `h`.
* `HPortReads` (named): the cross-unit read, at one copy of the RecSparse unit and the values `x` of rep `i`'s `h`.
* `h_of_reads`: at no clash, a sparse copy at w (`SparseCopyAt`), and `HPortReads` at `hOfW` of some rep unit of rep
  `i`'s steps, w's first 64 elements are the tied rep's `h`.
* `SrcHReads` (named, family form), `SrcHTie`, `srcHTie_of`: the same at every accepted outer proof, at the copy of
  rep `i`'s unit (`un = i`, the row it reads); `SrcHTie` is `SparseRepTie`'s `h` half at `tiedSrc stp`.
* `site_reads_of_portDiff`, `regChecks_of_checkStmt`, `rowPort_of_checkRows`, `own_of_checkOwn`: `checkRows` pins a
  part's row port's entry to `--rec-rows`' entry of its name, registration and reads, and `checkOwn` a message-round
  port's entry to its value and reads (`mIndex_m`: `m<hr>` is round `hr`'s).
* `ReadsOwnAt`, `readsOwnAt_of`: a registered read is the reads file's entry of its name at the unit's own row, from
  `readsOwn_of_hown` and the label check at reached, checked outcomes (`ReachedChecked`, named: `accRead_at_unit`'s
  premises at every session).
* `hSite_of_checked`: the two reads' registration and position: both entries read `rec-c<hr>` at both reps' rows
  (`checkOwn`), `--rec-rows` registers it (`checkRows` at the part's `m<hr>`), and `checkShared` holds both to its
  first entry of that value (`site_of_checkShared`).
* `readsUnsegC3`: no read is of a segmented row (`VStmt.hscope`'s `v1Ports`).
* `hKey_of_check`: at a copy of a unit `RecSparse.check` accepted, the key of `h`'s element `l`, bit `e` is bit
  `128 l + e` of the port named `h` (`rowsOk`'s row 0, `leavesOk`, `srcKey_inOrder`).
* `srcHReads_of`: `SrcHReads` from those, the columns of the RecSparse circuit's `h` (`HCols`, named) and the algebra
  unit's row of rep `i`'s `h` (`HAlgRows`, named).
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg FlockSoundness.Registered
open FlockSoundness.Discharge.HiddenExec (encPortH)

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type}

/-- **Two registered reads at one registration and position hold one row's bytes, or two records clash**: the two
rows' bytes at their ports' encodings are one, or the two records' openings or hm96 leaves collide there, `RegColl`'s
event at these sessions. -/
theorem regRows_eq_or_clash (zs : (j : Fin S) → (Vs j).Outer Reg') {s s' : Fin S} {u : Fin (Vs s).nv}
    {u' : Fin (Vs s').nv} {q : Fin (Vs s).c.ports.size} {q' : Fin (Vs s').c.ports.size}
    (hr : (zs s).rd.isRead (.inl (u, q))) (hr' : (zs s').rd.isRead (.inl (u', q')))
    (hsite : FlockVBridge.Family.site ((zs s).rd.port (.inl (u, q))) =
      FlockVBridge.Family.site ((zs s').rd.port (.inl (u', q'))))
    (hat : (zs s).rd.readAt (.inl (u, q)) = (zs s').rd.readAt (.inl (u', q'))) :
    encPortH (Vs s).c.ports[q.val]! (FlockVBridge.Decode.regVal (zs s) (.inl (u, q))) =
        encPortH (Vs s').c.ports[q'.val]! (FlockVBridge.Decode.regVal (zs s') (.inl (u', q'))) ∨
      ∃ j j', (zs j).rd.Clash (zs j').rd := by
  rcases FlockVBridge.Family.registrants_eq_or_collides_site (zs s).rd (zs s').rd hr hr' hsite hat with he | hc | hc
  · exact Or.inl he
  · exact Or.inr ⟨s, s', .inl (u, q), .inl (u', q'), hr, hr', Or.inl hc⟩
  · exact Or.inr ⟨s, s', .inl (u, q), .inl (u', q'), hr, hr', Or.inr hc⟩

/-- **Two registered reads at one registration and position carry one row's bits** where no two records clash (`Accepted`'s
`¬ RegColl`) and neither row is segmented (`bits_of_encPortH`). -/
theorem regVal_bit_eq (zs : (j : Fin S) → (Vs j).Outer Reg') (hnc : ¬ ∃ j j', (zs j).rd.Clash (zs j').rd)
    {s s' : Fin S} {u : Fin (Vs s).nv} {u' : Fin (Vs s').nv} {q : Fin (Vs s).c.ports.size}
    {q' : Fin (Vs s').c.ports.size} (hr : (zs s).rd.isRead (.inl (u, q))) (hr' : (zs s').rd.isRead (.inl (u', q')))
    (hsite : FlockVBridge.Family.site ((zs s).rd.port (.inl (u, q))) =
      FlockVBridge.Family.site ((zs s').rd.port (.inl (u', q'))))
    (hat : (zs s).rd.readAt (.inl (u, q)) = (zs s').rd.readAt (.inl (u', q')))
    (hs : (Vs s).c.ports[q.val]!.segs = none) (hs' : (Vs s').c.ports[q'.val]!.segs = none) (b : ℕ)
    (hb : b < Layout.portRowW (Vs s).c q) (hb' : b < Layout.portRowW (Vs s').c q') :
    (FlockVBridge.Decode.regVal (zs s) (.inl (u, q))).2
        ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
      (FlockVBridge.Decode.regVal (zs s') (.inl (u', q'))).2
        ⟨b, lt_of_lt_of_le hb' ((Finset.le_sup (Finset.mem_univ q')).trans (Nat.le_succ _))⟩ := by
  rcases regRows_eq_or_clash zs hr hr' hsite hat with he | hc
  · exact FlockVBridge.Family.bits_of_encPortH hs hs' he b _ _
  · exact absurd hc hnc

end FlockSoundness.Discharge.Recursion

namespace FlockC3.HoloValues

noncomputable section

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Refine
open Flock.RecAlgebra (hOfW)
open FlockSoundness.Model (holoRepC extraClaims)
open FlockSoundness.Discharge.Recursion (ExecHolo)
open FlockC3.Compose (C3Run)

/-- **A rep unit plays `holoRepC` to an output whose `h` is the unit's `hOfW`** (`rep_of_unit`, keeping
`holoRep_refines`' `lc.h`). -/
theorem rep_of_unit_h {S : Model.Statement} {sch₀ : Accounting.Schedule} {extra : List (Model.Claim 𝔽 S.m)}
    {cap₀ : Model.Cap Dig} {os : Model.Opens 𝔽 Dig} {r : ℕ} {ts : List Game.Step} (hm : 13 ≤ S.m)
    (U : RepUnit S sch₀ extra cap₀ os r ts) :
    ∃ zc lc pend, Game.PlaysT (holoRepC (D := Dig) execArith r S sch₀ hm extra) ts (some (zc, lc), pend) ∧
      lc.h = vecOf 64 (hOfW U.sh U.d.w) := by
  have h22 := exec_fast100_ge U.fast
  obtain ⟨hprog, zc₁, t₁, h₁, hk₁⟩ := holoProg_of_zclc U.sh U.d U.alg U.acc U.st U.sch U.p U.extras U.t U.k0 U.coins
    U.wf U.m (by omega) U.kL U.zp U.lp U.foldPts U.n₀ U.n₁ U.n₂ U.nr U.a₀ U.a₁ U.a₂ U.ar U.lc U.cl U.rs U.rsShape
    U.rsVal U.rsRounds U.lg
  have hdims := fast100_dims U.st.m U.sch U.fast
  have hlv : ∀ i < U.sch.levels.size, LevelSeam (U.sch.levels.getD i default) := fun i hi =>
    fast100_level U.st.m U.sch U.fast _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  obtain ⟨-, kz, -, -, -, hkz, zc, lc, pend, hpl, hh, -⟩ := holoRep_refines U.st S hm
    (by have := U.mS; omega) U.sch U.mS U.kS U.fold extra U.extras U.extrasLen U.extrasRel hdims.1 hlv
    (fun i hi => fast100_alpha U.st.m U.sch U.fast _
      (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi))
    (hdims.2.trans U.mS) U.p (hOfW U.sh U.d.w) r U.t U.t₄ U.wf hprog capDig os
    (fun ℓ _ j _ => show (os r ℓ j).1 = (openRow U.p.ligerito U.sch ℓ j).1 from congrArg Prod.fst (U.opens ℓ j))
  have hkz' := hkz zc₁ t₁ h₁
  rw [hk₁] at hkz'
  subst hkz'
  rw [U.sched] at hpl
  exact ⟨zc, lc, pend, playsT_of_spine (holoRepC_spine (D := Dig) execArith r S sch₀ hm extra) hpl U.kinds U.msgs
    U.coinSteps, hh⟩

/-- **The tied run's rep `i` has the `h` of any rep unit of its steps**: `tieReps` sets rep `i` to its play's output
(`playOut_eq`), whose `h` is the unit's `hOfW` (`rep_of_unit_h`). -/
theorem rep_h_of_unit {e : ExecHolo} {u : C3Run e} {i : Fin 2}
    (U : RepUnit (e.cls.pubStatement e.regs) e.sch
      (extraClaims (e.cls.pubStatement e.regs) e.cls.ptLocal e.cls.mPts u.pts) u.cap₀ u.os.1 i (u.ts i))
    (l : Fin 64) : ((FlockC3.Compose.tie u).rep i).2.h l = fOf (hOfW U.sh U.d.w)[l.val]! := by
  obtain ⟨zc, lc, pend, hpl, hh⟩ := rep_of_unit_h e.cls.hm U
  have e1 : (FlockC3.Compose.tie u).rep i =
      ((FlockC3.Compose.playOut (repAt e u i) (u.ts i) (some (u.rep i), u.pend i)).1).getD (u.rep i) := rfl
  rw [e1, FlockC3.Compose.playOut_eq _ hpl]
  show lc.h l = _
  rw [hh]
  rfl

end

end FlockC3.HoloValues

namespace FlockC3.HPort

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge (wbit wbit_elem bitOf)
open FlockVBridge.Decode (RegBit regVal)
open FlockVBridge.OfVStar (ReadCol rowCol)
open FlockVBridge.AccChain (eq_of_bitOf)
open Flock.RecAlgebra (hOfW)

local instance lawfulBEqByteArrayH : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

section checks

/-- **Two ports `portDiff?` finds no field of have one registration and one set of reads.** -/
theorem site_reads_of_portDiff {a b : Flock.Registered.Port} (h : Flock.RecVStar.portDiff? a b = none) :
    FlockVBridge.Family.site a = FlockVBridge.Family.site b ∧ a.reads = b.reads := by
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
  have e7 : a.reads = b.reads := Classical.byContradiction fun hne => by
    simp [Flock.RecVStar.portDiff?, e1, e2, e3, e4, e5, e6, hne] at h
  exact ⟨by simp only [FlockVBridge.Family.site, e1, e2, e3, e4, e5, e6], e7⟩

/-- **`checkStmt` ran `checkOwn`, `checkRows` and `checkShared` on V*'s statement `j`** at the verifier's own reads
(`regs_of_checkStmt`'s walk). -/
theorem regChecks_of_checkStmt {i : Flock.RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {own : Array Flock.Registered.Port} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (h : Flock.RecVStar.checkStmt (some i) tags tables st circuitB partition program (some own) (some j) (some f)
      rows = .ok ()) :
    ∃ ss s, Flock.RecVStar.statements i = .ok ss ∧ ss[j]? = some s ∧ Flock.RecVStar.checkOwn s own = .ok () ∧
      Flock.RecVStar.checkRows s own rows = .ok () ∧ Flock.RecVStar.checkShared s own rows = .ok () := by
  unfold Flock.RecVStar.checkStmt at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
    cases h1
  · split at h
    · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
      cases h1
    · split at h
      · split at h
        · cases h
        obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
        obtain ⟨s, hs, h⟩ := FlockSoundness.Refine.ebind_ok h
        obtain ⟨hck, -⟩ := FlockSoundness.Discharge.Recursion.checkCircuit_ok hs
        obtain ⟨ss, hss, hj, -⟩ := FlockSoundness.Discharge.Recursion.check_ok hck
        obtain ⟨⟨⟩, -, h⟩ := FlockSoundness.Refine.ebind_ok h
        obtain ⟨⟨⟩, hown, h⟩ := FlockSoundness.Refine.ebind_ok h
        obtain ⟨⟨⟩, hrw, h⟩ := FlockSoundness.Refine.ebind_ok h
        obtain ⟨⟨⟩, -, h⟩ := FlockSoundness.Refine.ebind_ok h
        obtain ⟨⟨⟩, -, h⟩ := FlockSoundness.Refine.ebind_ok h
        exact ⟨ss, s, hss, hj, hown, hrw, h⟩
      · cases h

/-- `m<n>`'s `mIndex` is `n`. -/
theorem mIndex_m (n : ℕ) : Flock.RecVStar.mIndex s!"m{n}" = some n := by
  have hm : (s!"m{n}").toList = 'm' :: (Nat.repr n).toList := by
    rw [String.toList_append]; rfl
  have hd : ((s!"m{n}").drop 1).toString = Nat.repr n := by
    apply String.toList_inj.mp
    rw [String.Slice.toString_eq, String.toList_copy_drop, hm]; rfl
  have hs : (s!"m{n}").startsWith "m" = true := by
    rw [String.startsWith_string_iff, hm]; exact ⟨_, rfl⟩
  unfold Flock.RecVStar.mIndex
  rw [ite_eq_left hs, hd, Nat.toNat?_repr]

/-- A row port of a statement is a part's (`rowPorts` of any other statement is empty), one of its `S.ports`. -/
theorem inner_of_rowPorts {s : VStar.Stmt} {nm : String} (h : nm ∈ Flock.RecVStar.rowPorts s) :
    ∃ j name p x, s = .inner j name p ∧ (nm, x) ∈ p.S.ports := by
  cases s <;> simp only [Flock.RecVStar.rowPorts, List.not_mem_nil] at h
  rename_i j name p
  obtain ⟨⟨a, x⟩, hx, rfl⟩ := List.mem_map.1 h
  exact ⟨j, name, p, x, rfl, hx⟩

/-- **A part's row port `m<hr>` is a message-round port under `rec-c<hr>` at both reps' rows** (`ownPorts`). -/
theorem mem_ownPorts_inner {j : ℕ} {name : String} {p : VStar.Part} {hr x : ℕ} (hx : (s!"m{hr}", x) ∈ p.S.ports) :
    (s!"m{hr}", s!"rec-c{hr}", #[some 0, some 1]) ∈ Flock.RecVStar.ownPorts (.inner j name p) := by
  unfold Flock.RecVStar.ownPorts
  exact List.mem_filterMap.2 ⟨_, hx, by dsimp only; rw [mIndex_m]; rfl⟩

/-- **RecSparse's `h` is a message-round port under `rec-c<hr>` at both reps' rows** (`ownPorts`). -/
theorem mem_ownPorts_sparse {sh : Flock.RecSparse.Shape} {hr : ℕ} :
    ("h", s!"rec-c{hr}", #[some 0, some 1]) ∈ Flock.RecVStar.ownPorts (.sparse sh hr) := by
  unfold Flock.RecVStar.ownPorts
  exact List.mem_cons_self ..

/-- **`checkOwn` pins the reads file's entry of a message-round port's name**, the file naming each port once. -/
theorem own_of_checkOwn {s : VStar.Stmt} {own : Array Flock.Registered.Port} (h : Flock.RecVStar.checkOwn s own = .ok ())
    {nm v : String} {rd : Array (Option ℕ)} (hq : (nm, v, rd) ∈ Flock.RecVStar.ownPorts s) {p : Flock.Registered.Port}
    (hp : p ∈ own) (hn : p.name = nm) (hnd : ∀ p ∈ own, ∀ p' ∈ own, p.name = p'.name → p = p') :
    p.value = v ∧ p.reads = rd := by
  obtain ⟨p', hp', he⟩ := List.mem_map.1 ((FlockC3.PubRows.checkOwn_ok h).1 _ hq)
  simp only [Prod.mk.injEq] at he
  obtain ⟨hn', hv, hrd⟩ := he
  obtain rfl := hnd p' (Array.mem_toList_iff.1 (List.mem_filter.1 hp').1) p hp (hn'.trans hn.symm)
  exact ⟨hv, hrd⟩

/-- **`checkRows`' loop**: each of the statement's row ports has a first entry in the reads file and one in
`--rec-rows`, and `portDiff?` finds no field (reads included) in which they differ. -/
theorem rowPort_of_checkRows {s : VStar.Stmt} {own : Array Flock.Registered.Port}
    {rows : Option (Array Flock.Registered.Port)} (h : Flock.RecVStar.checkRows s own rows = .ok ()) {nm : String}
    (hnm : nm ∈ Flock.RecVStar.rowPorts s) :
    ∃ rws reg w, rows = some rws ∧ own.find? (·.name == nm) = some reg ∧ rws.find? (·.name == nm) = some w ∧
      Flock.RecVStar.portDiff? reg w = none := by
  have hne : (Flock.RecVStar.rowPorts s).isEmpty = false := by
    cases he : (Flock.RecVStar.rowPorts s).isEmpty with
    | false => rfl
    | true => exact absurd (List.isEmpty_iff.1 he) (List.ne_nil_of_mem hnm)
  cases rows with
  | none =>
    unfold Flock.RecVStar.checkRows at h
    dsimp only at h
    split at h
    · cases h
    · split at h
      · rename_i ht; rw [hne] at ht; cases ht
      · cases h
  | some rws =>
    unfold Flock.RecVStar.checkRows at h
    dsimp only at h
    split at h
    · cases h
    · split at h
      · rename_i ht; rw [hne] at ht; cases ht
      · obtain ⟨_, h, -⟩ := FlockSoundness.Refine.ebind_ok h
        suffices hP : ∀ x ∈ Flock.RecVStar.rowPorts s, ∃ reg w, own.find? (·.name == x) = some reg ∧
            rws.find? (·.name == x) = some w ∧ Flock.RecVStar.portDiff? reg w = none by
          obtain ⟨reg, w, h1, h2, h3⟩ := hP nm hnm
          exact ⟨rws, reg, w, rfl, h1, h2, h3⟩
        refine FlockC3.PubRows.rows_forIn_each _ ?_ _ h
        intro nm' r hr
        try dsimp only at hr
        split at hr
        · rename_i reg hreg
          split at hr
          · rename_i w hw
            split at hr
            · cases hr
            · rename_i hn
              refine ⟨⟨reg, w, hreg, hw, ?_⟩, ?_⟩
              · cases hd : Flock.RecVStar.portDiff? reg w with
                | none => rfl
                | some d => exact absurd hd (hn d)
              · cases hr; rfl
          · cases hr
        · cases hr

/-- **A V* circuit's rows are unsegmented** (`scopeOk`'s `v1Ports`). -/
theorem segs_of_scopeOk {c : Flock.Circuit} (h : Layout.scopeOk c = true) (q : Fin c.ports.size) :
    c.ports[q.val]!.segs = none := by
  unfold Layout.scopeOk Layout.v1Ports at h
  simp only [Bool.and_eq_true, Array.all_eq_true', Bool.not_eq_true', beq_iff_eq, decide_eq_true_eq] at h
  rw [getElem!_pos c.ports q.val q.isLt]
  obtain ⟨⟨⟨-, hs⟩, -⟩, -⟩ := h.2 _ (Array.getElem_mem q.isLt)
  simpa using hs

end checks

section point

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type}

/-- Obligation: the cross-unit read, at copy `j` of unit `un` of session `s` (a RecSparse copy) and the values `x` of
rep `i`'s `h`. Each column of `h`'s element `l`, bit `e` (message column `128 l + e`, `mp`'s first port) is an input
column a row reads (W's `SparseCols`), and where its key is bit `b` of a registered row `q` of the copy's unit, that row
is unsegmented and some session `s'`'s unit `u'` reads a registered row `q'` holding bit `b`, unsegmented, at the same
registration (`site`: the two ports differ by name, `h` and `m<hr>`) and position, whose registrant bit `b` is bit `e`
of `x l`. `srcHReads_of` gives it from the verifier's checks and three named pieces. -/
def HPortReads (zs : (j : Fin S) → (Vs j).Outer Reg') (mp : List (String × ℕ)) (s : Fin S) (un : Fin (Vs s).nv)
    (j : ℕ) (x : Fin 64 → Flock.F128) : Prop :=
  (∀ (l : Fin 64) (e : ℕ), e < 128 → ReadCol (FlockSoundness.Rows.ofNet (Vs s).hU) (rowCol mp 0 (128 * l.val + e))) ∧
    ∀ (l : Fin 64) (e : ℕ), e < 128 → ∀ (q : Fin (Vs s).c.ports.size) (b : ℕ), b < Layout.portRowW (Vs s).c q →
      Aliased.srcKey (Vs s).c (j * (FlockSoundness.Rows.ofNet (Vs s).hU).nIn + rowCol mp 0 (128 * l.val + e)) =
          .msg q.val (b / 8) (b % 8) →
      (zs s).rd.isRead (.inl (un, q)) →
      (Vs s).c.ports[q.val]!.segs = none ∧
        ∃ (s' : Fin S) (u' : Fin (Vs s').nv) (q' : Fin (Vs s').c.ports.size) (hb' : b < Layout.portRowW (Vs s').c q'),
          (zs s').rd.isRead (.inl (u', q')) ∧ (Vs s').c.ports[q'.val]!.segs = none ∧
          FlockVBridge.Family.site ((zs s).rd.port (.inl (un, q))) =
            FlockVBridge.Family.site ((zs s').rd.port (.inl (u', q'))) ∧
          (zs s).rd.readAt (.inl (un, q)) = (zs s').rd.readAt (.inl (u', q')) ∧
          (regVal (zs s') (.inl (u', q'))).2
              ⟨b, lt_of_lt_of_le hb' ((Finset.le_sup (Finset.mem_univ q')).trans (Nat.le_succ _))⟩ =
            bitOf (x l) e

/-- Obligation (W's `SparseCols`): each column of `h`'s element `l`, bit `e` in the RecSparse circuit is an input
column a row reads. -/
def HCols (V : VStmt) (mp : List (String × ℕ)) : Prop :=
  ∀ (l : Fin 64) (e : ℕ), e < 128 → ReadCol (FlockSoundness.Rows.ofNet V.hU) (rowCol mp 0 (128 * l.val + e))

/-- **The RecSparse circuit's `h` keys, from the verifier's check** (`srcKey_inOrder`): at a unit `RecSparse.check`
accepted, row 0 is `h` at 512 words (`rowsOk`) and copy `j` reads the rows' words in order (`leavesOk`), so where the
key of `h`'s element `l`, bit `e` at copy `j` is bit `b` of row `q`, `q` is `h` and `b` is `128 l + e`. -/
theorem hKey_of_check {V : VStmt} (hck : Flock.RecSparse.check V.c = .ok ()) {sh : Flock.RecSparse.Shape}
    (hp : Flock.RecSparse.params V.c.unit.name = some sh) {j : ℕ} (hj : j < V.c.upv) (l : Fin 64) {e : ℕ}
    (he : e < 128) (q : Fin V.c.ports.size) (b : ℕ)
    (hkey : Aliased.srcKey V.c (j * (FlockSoundness.Rows.ofNet V.hU).nIn +
        rowCol (Flock.RecSparse.messagePorts sh) 0 (128 * l.val + e)) = .msg q.val (b / 8) (b % 8)) :
    V.c.ports[q].name = "h" ∧ b = 128 * l.val + e := by
  obtain ⟨-, -, -, hrows, hlv, hn⟩ := FlockVBridge.Sparse.check_sparse_all hck hp
  obtain ⟨h0, hname, hw, hbits⟩ := FlockVBridge.Sparse.rowsOk_h hrows
  have hn' : Flock.HmNets.netOf (Flock.RecSparse.unitName sh) (Flock.RecSparse.unitLaid sh) = .ok V.c.unit := hn
  have hl64 := l.isLt
  obtain ⟨rest, hmp⟩ := FlockVBridge.Sparse.messagePorts_head sh
  have hcol : rowCol (Flock.RecSparse.messagePorts sh) 0 (128 * l.val + e) = 128 * l.val + e := by
    rw [hmp]
    simp only [rowCol]
    split <;> omega
  have hsz : (Array.replicate (Flock.RecSparse.inWords sh) 16).size = Flock.RecSparse.inWords sh :=
    Array.size_replicate
  have h16 : ∀ x ∈ (Array.replicate (Flock.RecSparse.inWords sh) 16).toList, x = 16 := by
    intro x hx
    rw [Array.toList_replicate] at hx
    exact List.eq_of_mem_replicate hx
  have hN : 16 * (Array.replicate (Flock.RecSparse.inWords sh) 16).size ≤ V.c.unit.inWords * Flock.WORD := by
    rw [FlockVBridge.Sparse.inWords_sparse hn', hsz]
    show 16 * _ ≤
      Flock.HmNets.groupWords ((Array.replicate (Flock.RecSparse.inWords sh) 16).foldl (· + ·) 0) * Flock.WORD
    rw [← Array.foldl_toList, Array.toList_replicate, FlockVBridge.foldl_replicate16, Nat.zero_add]
    exact Flock.HmNets.le_groupWords _
  have hst : (FlockSoundness.Typed.starts V.c)[0]! = 0 := by
    rw [FlockSoundness.Discharge.Layout.starts_get V.c h0]
    simp
  have hin := FlockVBridge.Sparse.h_le_inWords sh
  have key := FlockC3.RegLevels.srcKey_inOrder (FlockVBridge.Sparse.inGroups_sparse hn') rfl h16
    (Aliased.portBits_nodup V.hU) (j := j) (k := 128 * l.val + e)
    (by rw [hsz]; exact FlockVBridge.Sparse.leavesOk_get hlv hj) h0 (by simp only [hw]; omega)
    (by simp only [hbits]; omega) (by rw [hst, hsz]; simp only [hw]; omega) hN
  rw [hst, Nat.mul_zero, Nat.zero_add] at key
  rw [hcol, FlockSoundness.Discharge.Layout.ofNet_nIn_eq, key] at hkey
  injection hkey with hq0 h8 hm8
  obtain ⟨q, hq⟩ := q
  dsimp only at hq0
  subst hq0
  exact ⟨hname, by omega⟩

/-- `h`'s 64 elements are the first port's: their bits are below the message ports' total. -/
theorem h_lt_ports (sh : Flock.RecSparse.Shape) {l e : ℕ} (hl : l < 64) (he : e < 128) :
    128 * l + e < 128 * ((Flock.RecSparse.messagePorts sh).map (·.2)).sum := by
  have h64 : 64 ≤ ((Flock.RecSparse.messagePorts sh).map (·.2)).sum := by
    simp only [Flock.RecSparse.messagePorts, List.append_assoc, List.singleton_append]
    exact Nat.le_add_right_of_le (le_of_eq rfl)
  omega

/-- **w's element `l` is `x l`** at a RecSparse copy at w (`SparseCopyAt`'s registered message bits), under the
cross-unit read (`HPortReads`) and no clash: the copy's `h` row and the algebra unit's row are one row's bits
(`regVal_bit_eq`). -/
theorem w_eq_of_reads {zs : (j : Fin S) → (Vs j).Outer Reg'} {Lb : (j : Fin S) → (Vs j).Bits} {e : ExecHolo}
    {w : Array Flock.F128} {c : FlockVBridge.Sparse.SCoins} {s : Fin S} {un : Fin (Vs s).nv} {j : ℕ}
    {hj : j < (Vs s).c.upv} {x : Fin 64 → Flock.F128} (hnc : ¬ ∃ j j', (zs j).rd.Clash (zs j').rd)
    (hcp : FlockC3.Sparse.SparseCopyAt zs Lb e w c s un j hj)
    (hx : HPortReads zs (Flock.RecSparse.messagePorts (FlockC3.Sparse.unitShape e)) s un j x) (l : Fin 64) :
    w[l.val]! = x l := by
  refine eq_of_bitOf fun t ht => ?_
  have hq := h_lt_ports (FlockC3.Sparse.unitShape e) l.isLt ht
  obtain ⟨-, q, b, hb, hkey, hread, hval⟩ := hcp.2.2.2.2.1 _ hq (hx.1 l t ht)
  obtain ⟨hseg, s', u', q', hb', hread', hseg', hsite, hat, hv⟩ := hx.2 l t ht q b hb hkey hread
  rw [← hv, ← regVal_bit_eq zs hnc hread hread' hsite hat hseg hseg' b hb hb']
  refine ((wbit_elem (fun k => w[k]!) l.val ht).symm.trans ?_)
  exact hval.symm

/-- **w's first 64 elements are the tied rep's `h`** (`w_eq_of_reads` at the values of some rep unit of rep `i`'s
steps, `rep_h_of_unit`). -/
theorem h_of_reads {zs : (j : Fin S) → (Vs j).Outer Reg'} {Lb : (j : Fin S) → (Vs j).Bits} {e : ExecHolo}
    {u : C3Run e} {i : Fin 2} {w : Array Flock.F128} {c : FlockVBridge.Sparse.SCoins} {s : Fin S}
    {un : Fin (Vs s).nv} {j : ℕ} {hj : j < (Vs s).c.upv} (hnc : ¬ ∃ j j', (zs j).rd.Clash (zs j').rd)
    (hcp : FlockC3.Sparse.SparseCopyAt zs Lb e w c s un j hj)
    (U : FlockC3.HoloValues.RepUnit (e.cls.pubStatement e.regs) e.sch
      (extraClaims (e.cls.pubStatement e.regs) e.cls.ptLocal e.cls.mPts u.pts) u.cap₀ u.os.1 i (u.ts i))
    (hx : HPortReads zs (Flock.RecSparse.messagePorts (FlockC3.Sparse.unitShape e)) s un j
      fun l => (hOfW U.sh U.d.w)[l.val]!) :
    ∀ l : Fin 64, fOf w[l.val]! = ((tie u).rep i).2.h l := fun l => by
  rw [FlockC3.HoloValues.rep_h_of_unit U l, w_eq_of_reads hnc hcp hx l]

end point

section src

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- Obligation: `HPortReads` at every accepted outer proof, every rep `i`, every RecSparse copy at rep `i`'s words
`w` (`SparseCopyAt`, which `SparseSessionAt` gives) at rep `i`'s unit (`un = i`: unit `un` reads row `un`, rep `un`'s
`h`), and some rep unit of rep `i`'s steps in `stp` (`RepTie`, with `U.d` the rep's decode, as `RepUnitOfAlg` gives
it). `srcHReads_of` discharges it. -/
def SrcHReads (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
    ∃ U : FlockC3.HoloValues.RepUnit ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch
        (extraClaims ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.ptLocal
          (holoAt st hp).cls.mPts (stp R ω t (layers zs dg N k Rw R ω t R₂ τ)).pts)
        (stp R ω t (layers zs dg N k Rw R ω t R₂ τ)).cap₀ (stp R ω t (layers zs dg N k Rw R ω t R₂ τ)).os.1 i
        ((stp R ω t (layers zs dg N k Rw R ω t R₂ τ)).ts i),
      ∀ (s : Fin S) (un : Fin (Vs s).nv) (j : ℕ) (hj : j < (Vs s).c.upv) (c : FlockVBridge.Sparse.SCoins),
        FlockC3.Sparse.SparseCopyAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
            (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i) c s un j hj → un.val = i.val →
          HPortReads (zs R ω t R₂) (Flock.RecSparse.messagePorts (FlockC3.Sparse.unitShape (holoAt st hp))) s un j
            fun l => (hOfW U.sh U.d.w)[l.val]!

/-- **`SparseRepTie`'s `h` half at the tied source**: at every accepted outer proof, where some RecSparse copy at rep
`i`'s unit holds rep `i`'s phase at `w` (`SparseCopyAt`), w's first 64 elements are rep `i`'s `h` in `tiedSrc stp`. -/
def SrcHTie (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ (i : Fin 2) (s : Fin S) (un : Fin (Vs s).nv)
    (j : ℕ) (hj : j < (Vs s).c.upv) (c : FlockVBridge.Sparse.SCoins),
    FlockC3.Sparse.SparseCopyAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
        (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i) c s un j hj → un.val = i.val →
      ∀ l : Fin 64, fOf (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i)[l.val]! =
        ((tiedSrc stp R ω t (layers zs dg N k Rw R ω t R₂ τ)).rep i).2.h l

/-- **`SrcHTie` from the cross-unit reads** (`h_of_reads` at `Accepted`'s `¬ RegColl`). -/
theorem srcHTie_of {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128}
    (h : SrcHReads zs dg N k Rw stp w) : SrcHTie zs dg N k Rw stp w := by
  intro R ω t R₂ τ o hA i s un j hj c hcp hun
  obtain ⟨U, hU⟩ := h R ω t R₂ τ o hA i
  exact h_of_reads (u := stp R ω t (layers zs dg N k Rw R ω t R₂ τ)) hA.2 hcp U (hU s un j hj c hcp hun)

/-- **`SrcHTie` at a copy of `SparseSessionAt`'s form at rep `i`'s unit**: the form `SparseRepTie`'s `h` conjunct
takes, where `SparseSessionAt`'s copy is at unit `i` (which it does not yet state). -/
theorem h_of_sessionAt {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128}
    (h : SrcHTie zs dg N k Rw stp w) {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂}
    {τ : (j : Fin S) → Strategy ((zs R ω t R₂ j).game dg N)}
    {o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript} (hA : Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o)
    {i : Fin 2} {c : FlockVBridge.Sparse.SCoins}
    (hs : ∃ (s : Fin S) (un : Fin (Vs s).nv) (j : ℕ) (hj : j < (Vs s).c.upv), un.val = i.val ∧
      FlockC3.Sparse.SparseCopyAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
        (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i) c s un j hj) :
    ∀ l : Fin 64, fOf (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i)[l.val]! =
      ((tiedSrc stp R ω t (layers zs dg N k Rw R ω t R₂ τ)).rep i).2.h l := by
  obtain ⟨s, un, j, hj, hun, hcp⟩ := hs
  exact h R ω t R₂ τ o hA i s un j hj c hcp hun

end src

section family

open FlockC3.PubRows (ChecksAccepted)
open FlockC3.RegLevels (Reached readsOwn_of_hown SharedValue site_of_checkShared site_of_portDiff)

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

/-- Obligation (`accRead_at_unit`'s premises, at every session): each accepted outcome is one its session's game
reaches (`Reached`) and passes candidate 3's checks at its own transcript (`ChecksC3`, which the event's `ChecksVR`
states there). -/
def ReachedChecked (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ j : Fin (vstarCount F.ish),
    Reached (F.zs R ω t R₂ j) (o j) ∧ ∃ (ish : Flock.RecVStar.Inner) (cs : Option Flock.RecVStar.InnerCommits)
      (coins : Option Flock.RecCoins.Source) (f : VStar.Bindings) (rows' : Option (Array Flock.Registered.Port)),
      Flock.Guarantees.ChecksC3 ish cs coins j f rows' (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own (o j).2.2

/-- **A registered read's port is the reads file's entry of its name, read at the unit's own row.** -/
def ReadsOwnAt (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ j (u : Fin (F.Vs j).nv)
    (q : Fin (F.Vs j).c.ports.size), (F.zs R ω t R₂ j).rd.isRead (.inl (u, q)) →
      (F.zs R ω t R₂ j).rd.port (.inl (u, q)) ∈ (F.zs R ω t R₂ j).own ∧
      ((F.zs R ω t R₂ j).rd.port (.inl (u, q))).name = (F.Vs j).c.ports[q].name ∧
      ((F.zs R ω t R₂ j).rd.port (.inl (u, q))).reads[u.val]? =
        some (some ((F.zs R ω t R₂ j).rd.readAt (.inl (u, q))))

/-- **`ReadsOwnAt` from `ReachedChecked`**: `readsOwn_of_hown` puts the read at the unit's label, which the label check
makes the unit (`label_of_checked`), as `accRead_at_unit` does at a level's `acc`. -/
theorem readsOwnAt_of {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    (hrc : ReachedChecked F dg N k Rw) : ReadsOwnAt F dg N k Rw := by
  intro R ω t R₂ τ o h j u q hr
  obtain ⟨hre, _, _, _, _, _, hc3⟩ := hrc R ω t R₂ τ o h j
  obtain ⟨hmem, hname, d, D, dp, idx, hd, hD, hdp, hidx, hu, hpos⟩ := readsOwn_of_hown h j hre u q hr
  obtain ⟨spec, sts, hst, -, -, -, hlb⟩ := hc3
  rw [FlockC3.PubRows.label_of_checked (F.zs R ω t R₂ j) (h.1 j).1 ⟨spec, sts, hst, hlb⟩ hd hD hdp hidx hu] at hpos
  exact ⟨hmem, hname, hpos⟩

/-- `ReadsUnsegmented`'s, at candidate 3's family: no registered read is of a segmented row. -/
def ReadsUnsegC3 (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ j (u : Fin (F.Vs j).nv)
    (q : Fin (F.Vs j).c.ports.size), (F.zs R ω t R₂ j).rd.isRead (.inl (u, q)) → (F.Vs j).c.ports[q.val]!.segs = none

/-- **`ReadsUnsegC3` holds**: every V* circuit's rows are unsegmented (`VStmt.hscope`, `segs_of_scopeOk`), read or not. -/
theorem readsUnsegC3 (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) :
    ReadsUnsegC3 F dg N k Rw :=
  fun _ _ _ _ _ _ _ j _ q _ => segs_of_scopeOk (F.Vs j).hscope q

/-- Obligation (the rep's decode of `h` at its algebra unit): at every accepted outer proof and rep `i`, some rep unit
`U` of rep `i`'s steps is such that where a RecSparse copy at unit `i` holds rep `i`'s phase, its session is
`.sparse _ hr` and some part's session reads, at unit `i`, the row port `m<hr>` (a row port of its statement, which
`checkRows` pins), whose bit `128 l + e` is bit `e` of `hOfW U.sh U.d.w`'s element `l` (`hOfW_valOf` at `.h l`,
`RepMaps`' columns of `U.d.w`, with `U.d` from `RepUnitOfAlg`). -/
def HAlgRows (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → C3Run (holoAt st hp))
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → Fin 2 →
      Array Flock.F128) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
    ∃ U : FlockC3.HoloValues.RepUnit ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch
        (extraClaims ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.ptLocal
          (holoAt st hp).cls.mPts (stp R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).pts)
        (stp R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).cap₀ (stp R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).os.1 i
        ((stp R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).ts i),
      ∀ (s : Fin (vstarCount F.ish)) (un : Fin (F.Vs s).nv) (j : ℕ) (hj : j < (F.Vs s).c.upv)
        (c : FlockVBridge.Sparse.SCoins),
        FlockC3.Sparse.SparseCopyAt (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
            (w R ω t (layers F.zs dg N k Rw R ω t R₂ τ) i) c s un j hj → un.val = i.val →
          ∃ (sh : Flock.RecSparse.Shape) (hr : ℕ), F.stmts[(s : ℕ)]? = some (.sparse sh hr) ∧
            ∃ (s' : Fin (vstarCount F.ish)) (u' : Fin (F.Vs s').nv) (q' : Fin (F.Vs s').c.ports.size)
              (st' : VStar.Stmt), u'.val = i.val ∧ F.stmts[(s' : ℕ)]? = some st' ∧
              (F.Vs s').c.ports[q'].name = s!"m{hr}" ∧ s!"m{hr}" ∈ Flock.RecVStar.rowPorts st' ∧
              (F.zs R ω t R₂ s').rd.isRead (.inl (u', q')) ∧
              ∀ (l : Fin 64) (e : ℕ), e < 128 → ∃ hb : 128 * l.val + e < Layout.portRowW (F.Vs s').c q',
                (regVal (F.zs R ω t R₂ s') (.inl (u', q'))).2
                    ⟨128 * l.val + e, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q')).trans (Nat.le_succ _))⟩ =
                  bitOf ((hOfW U.sh U.d.w)[l.val]!) e

/-- **The sparse unit's `h` and the algebra unit's `m<hr>` reads have one registration and position**: `checkOwn` puts
both reads' entries (`ReadsOwnAt`, the reads file's names unique) under `rec-c<hr>` at both reps' rows, `checkRows`
registers `rec-c<hr>` in `--rec-rows` (the part's `m<hr>` entry), and `checkShared` holds each to `--rec-rows`' first
entry of that value in every field but reads (`site_of_checkShared`), so units of one index read one row. -/
theorem hSite_of_checked {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json} (hchk : ChecksAccepted F rows wr reg)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hro : ReadsOwnAt F dg N k Rw)
    (hnd : ∀ R ω t R₂ j, ∀ p ∈ (F.zs R ω t R₂ j).own, ∀ p' ∈ (F.zs R ω t R₂ j).own, p.name = p'.name → p = p')
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) {s s' : Fin (vstarCount F.ish)} {sh : Flock.RecSparse.Shape}
    {hr : ℕ} (hss : F.stmts[(s : ℕ)]? = some (.sparse sh hr)) {st' : VStar.Stmt} (hss' : F.stmts[(s' : ℕ)]? = some st')
    {un : Fin (F.Vs s).nv} {u' : Fin (F.Vs s').nv} {q : Fin (F.Vs s).c.ports.size} {q' : Fin (F.Vs s').c.ports.size}
    (hq : (F.Vs s).c.ports[q].name = "h") (hq' : (F.Vs s').c.ports[q'].name = s!"m{hr}")
    (hrp : s!"m{hr}" ∈ Flock.RecVStar.rowPorts st') (hu : un.val = u'.val)
    (hrd : (F.zs R ω t R₂ s).rd.isRead (.inl (un, q))) (hrd' : (F.zs R ω t R₂ s').rd.isRead (.inl (u', q'))) :
    FlockVBridge.Family.site ((F.zs R ω t R₂ s).rd.port (.inl (un, q))) =
        FlockVBridge.Family.site ((F.zs R ω t R₂ s').rd.port (.inl (u', q'))) ∧
      (F.zs R ω t R₂ s).rd.readAt (.inl (un, q)) = (F.zs R ω t R₂ s').rd.readAt (.inl (u', q')) := by
  have chk : ∀ j : Fin (vstarCount F.ish), ∃ sj, F.stmts[(j : ℕ)]? = some sj ∧
      Flock.RecVStar.checkOwn sj (F.zs R ω t R₂ j).own = .ok () ∧
      Flock.RecVStar.checkRows sj (F.zs R ω t R₂ j).own rows = .ok () ∧
      Flock.RecVStar.checkShared sj (F.zs R ω t R₂ j).own rows = .ok () := fun j => by
    obtain ⟨_, -, -, _, _, hset, hfm⟩ := hchk R ω t R₂ j ⟨_, (h.1 j).1⟩
    obtain ⟨h0, -⟩ := setup_zero_c (F.zs R ω t R₂ j).ht (F.zs R ω t R₂ j).hTags.1 (F.zs R ω t R₂ j).hcH hset
    obtain ⟨ss, sj, hss₁, hj₁, ho, hrw, hsh⟩ := regChecks_of_checkStmt (array_forM_ok hfm 0 h0)
    obtain rfl : ss = F.stmts := Except.ok.inj (hss₁.symm.trans F.hstmts)
    exact ⟨sj, hj₁, ho, hrw, hsh⟩
  obtain ⟨hmem, hname, hpos⟩ := hro R ω t R₂ τ o h s un q hrd
  obtain ⟨hmem', hname', hpos'⟩ := hro R ω t R₂ τ o h s' u' q' hrd'
  obtain ⟨sa, hsa, hoa, -, hsha⟩ := chk s
  obtain rfl : sa = .sparse sh hr := Option.some.inj (hsa.symm.trans hss)
  obtain ⟨sb, hsb, hob, hrwb, hshb⟩ := chk s'
  obtain rfl : sb = st' := Option.some.inj (hsb.symm.trans hss')
  obtain ⟨jn, nmn, pt, x, rfl, hx⟩ := inner_of_rowPorts hrp
  obtain ⟨hva, hra⟩ := own_of_checkOwn hoa mem_ownPorts_sparse hmem (hname.trans hq) (hnd R ω t R₂ s)
  obtain ⟨hvb, hrb⟩ := own_of_checkOwn hob (mem_ownPorts_inner hx) hmem' (hname'.trans hq') (hnd R ω t R₂ s')
  obtain ⟨rws, pb, w, hrows, hpb, hw, hd⟩ := rowPort_of_checkRows hrwb hrp
  obtain rfl : pb = (F.zs R ω t R₂ s').rd.port (.inl (u', q')) :=
    hnd R ω t R₂ s' _ (Array.mem_of_find?_eq_some hpb) _ hmem'
      ((by simpa using Array.find?_some hpb : pb.name = s!"m{hr}").trans (hname'.trans hq').symm)
  have hwv : w.value = s!"rec-c{hr}" := (congrArg Prod.fst (site_reads_of_portDiff hd).1).symm.trans hvb
  have hsv : SharedValue rows ((F.zs R ω t R₂ s).rd.port (.inl (un, q))).value := Or.inl (by
    rw [hrows, hva, Option.bind_some]
    exact Array.find?_isSome.2 ⟨w, Array.mem_of_find?_eq_some hw, by simp [hwv]⟩)
  obtain ⟨w₀, hw₀, hd₀⟩ := site_of_checkShared hsha hmem hsv
  obtain ⟨w₁, hw₁, hd₁⟩ := site_of_checkShared hshb hmem' (by rw [hvb, ← hva]; exact hsv)
  rw [hvb, ← hva, hw₀] at hw₁
  obtain rfl := Option.some.inj hw₁
  refine ⟨(site_of_portDiff hd₀).trans (site_of_portDiff hd₁).symm, ?_⟩
  rw [hra] at hpos
  rw [hrb, ← hu] at hpos'
  exact Option.some.inj (Option.some.inj (hpos.symm.trans hpos'))

/-- **`SrcHReads` from the verifier's checks** (`hSite_of_checked`, `readsUnsegC3`) and the named pieces: the reads at
the units' own rows (`ReadsOwnAt`, from `ReachedChecked` by `readsOwnAt_of`), the RecSparse circuit's `h` keys
(`HCols`; its keys `hKey_of_check`) and the algebra unit's row of rep `i`'s `h` (`HAlgRows`). -/
theorem srcHReads_of {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json} (hchk : ChecksAccepted F rows wr reg)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hro : ReadsOwnAt F dg N k Rw)
    (hnd : ∀ R ω t R₂ j, ∀ p ∈ (F.zs R ω t R₂ j).own, ∀ p' ∈ (F.zs R ω t R₂ j).own, p.name = p'.name → p = p')
    (hcols : ∀ s : Fin (vstarCount F.ish), Flock.RecSparse.check (F.Vs s).c = .ok () →
      Flock.RecSparse.params (F.Vs s).c.unit.name = some (FlockC3.Sparse.unitShape (holoAt st hp)) →
      HCols (F.Vs s) (Flock.RecSparse.messagePorts (FlockC3.Sparse.unitShape (holoAt st hp))))
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → C3Run (holoAt st hp)}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → Fin 2 →
      Array Flock.F128}
    (halg : HAlgRows F dg N k Rw stp w) : SrcHReads F.zs dg N k Rw stp w := by
  intro R ω t R₂ τ o hA i
  obtain ⟨U, hU⟩ := halg R ω t R₂ τ o hA i
  refine ⟨U, fun s un j hj c hcp hun => ?_⟩
  obtain ⟨sh, hr, hss, s', u', q', st', hu', hss', hq', hrp, hr', hbits⟩ := hU s un j hj c hcp hun
  refine ⟨hcols s hcp.1 hcp.2.1, fun l e he q b _ hkey hread => ?_⟩
  obtain ⟨hq, rfl⟩ := hKey_of_check hcp.1 hcp.2.1 hj l he q b hkey
  obtain ⟨hb', hv⟩ := hbits l e he
  obtain ⟨hst, hat⟩ := hSite_of_checked hchk hro hnd hA hss hss' hq hq' hrp (hun.trans hu'.symm) hread hr'
  have hseg := readsUnsegC3 F dg N k Rw R ω t R₂ τ o hA
  exact ⟨hseg s un q hread, s', u', q', hb', hr', hseg s' u' q' hr', hst, hat, hv⟩

end family

end

end FlockC3.HPort
