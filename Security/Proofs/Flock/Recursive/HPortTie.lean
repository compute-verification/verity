import Proofs.Flock.C3.Compose.Tied
import Proofs.Flock.C3.Sparse.SessionAt
import Proofs.Flock.VBridge.Family.Registrants
import Proofs.Flock.C3.RegLevels.AccReads

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
* `site_reads_of_portDiff`, `checkRows_of_checkStmt`, `rowPort_of_checkRows`: `checkRows` pins a part's row port's
  entry to `--rec-rows`' entry of its name, registration and reads.
* `hSite_of_checked`: the two reads' registration and position, from `checkRows` at the algebra session, the reads
  file's entries (`ReadsOwnC3`, names unique) and `SparseHSite` (named: `h`'s entry against `--rec-rows`' `m<hr>`,
  which no check of the verifier makes yet; `checkRows` reads no row port of RecSparse).
* `srcHReads_of`: `SrcHReads` from those, no read segmented (`ReadsUnsegC3`), the RecSparse circuit's `h` keys
  (`HKeys`, named) and the algebra unit's row of rep `i`'s `h` (`HAlgRows`, named).
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

/-- **`checkStmt` ran `checkRows` on V*'s statement `j`** at the verifier's own reads (`checkAcc_of_checkStmt`'s
walk). -/
theorem checkRows_of_checkStmt {i : Flock.RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {own : Array Flock.Registered.Port} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (h : Flock.RecVStar.checkStmt (some i) tags tables st circuitB partition program (some own) (some j) (some f)
      rows = .ok ()) :
    ∃ ss s, Flock.RecVStar.statements i = .ok ss ∧ ss[j]? = some s ∧ Flock.RecVStar.checkRows s own rows = .ok () := by
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
        obtain ⟨⟨⟩, -, h⟩ := FlockSoundness.Refine.ebind_ok h
        obtain ⟨⟨⟩, hrw, -⟩ := FlockSoundness.Refine.ebind_ok h
        exact ⟨ss, s, hss, hj, hrw⟩
      · cases h

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

/-- Obligation (the RecSparse circuit's parse): each column of `h`'s element `l`, bit `e` at copy `j` is an input
column a row reads (W's `SparseCols`), and where its key is bit `b` of row `q`, `q` is the port named `h` and `b` is
`128 l + e` (the whole registered row from bit 0, element `l` at `128 l`). -/
def HKeys (V : VStmt) (mp : List (String × ℕ)) (j : ℕ) : Prop :=
  (∀ (l : Fin 64) (e : ℕ), e < 128 → ReadCol (FlockSoundness.Rows.ofNet V.hU) (rowCol mp 0 (128 * l.val + e))) ∧
    ∀ (l : Fin 64) (e : ℕ), e < 128 → ∀ (q : Fin V.c.ports.size) (b : ℕ), b < Layout.portRowW V.c q →
      Aliased.srcKey V.c (j * (FlockSoundness.Rows.ofNet V.hU).nIn + rowCol mp 0 (128 * l.val + e)) =
          .msg q.val (b / 8) (b % 8) →
      V.c.ports[q].name = "h" ∧ b = 128 * l.val + e

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
open FlockC3.RegLevels (ReadsOwnC3)

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

/-- Obligation, which no check of the verifier makes: at a RecSparse session (`.sparse _ hr`), the reads file's `h`
entry is `--rec-rows`' `m<hr>` entry in every field `portDiff?` compares, as `checkRows` pins a part's `m<hr>`
(`rowPorts` of RecSparse is empty). A fail-closed check in `checkStmt` at `.sparse _ hr` would give it. -/
def SparseHSite (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (rows : Option (Array Flock.Registered.Port))
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (s : Fin (vstarCount F.ish))
    (sh : Flock.RecSparse.Shape) (hr : ℕ), F.stmts[(s : ℕ)]? = some (.sparse sh hr) →
    ∀ pa ∈ (F.zs R ω t R₂ s).own, pa.name = "h" →
      ∃ rws w, rows = some rws ∧ rws.find? (·.name == s!"m{hr}") = some w ∧ Flock.RecVStar.portDiff? pa w = none

/-- Obligation (`ReadsUnsegmented`'s, at candidate 3's family): no registered read is of a segmented row. -/
def ReadsUnsegC3 (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ j (u : Fin (F.Vs j).nv)
    (q : Fin (F.Vs j).c.ports.size), (F.zs R ω t R₂ j).rd.isRead (.inl (u, q)) → (F.Vs j).c.ports[q.val]!.segs = none

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

/-- **The sparse unit's `h` and the algebra unit's `m<hr>` reads have one registration and position**: the reads
file's entries of the two reads' ports (`ReadsOwnC3`) are `--rec-rows`' `m<hr>` in registration and reads, the
sparse one by `SparseHSite` and the part's by `checkRows` (`rowPort_of_checkRows`, the reads file's names unique), so
units of one index read one row. -/
theorem hSite_of_checked {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json} (hchk : ChecksAccepted F rows wr reg)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hro : ReadsOwnC3 F dg N k Rw)
    (hnd : ∀ R ω t R₂ j, ∀ p ∈ (F.zs R ω t R₂ j).own, ∀ p' ∈ (F.zs R ω t R₂ j).own, p.name = p'.name → p = p')
    (hsite : SparseHSite F rows dg N k Rw) {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) {s s' : Fin (vstarCount F.ish)} {sh : Flock.RecSparse.Shape}
    {hr : ℕ} (hss : F.stmts[(s : ℕ)]? = some (.sparse sh hr)) {st' : VStar.Stmt} (hss' : F.stmts[(s' : ℕ)]? = some st')
    {un : Fin (F.Vs s).nv} {u' : Fin (F.Vs s').nv} {q : Fin (F.Vs s).c.ports.size} {q' : Fin (F.Vs s').c.ports.size}
    (hq : (F.Vs s).c.ports[q].name = "h") (hq' : (F.Vs s').c.ports[q'].name = s!"m{hr}")
    (hrp : s!"m{hr}" ∈ Flock.RecVStar.rowPorts st') (hu : un.val = u'.val)
    (hrd : (F.zs R ω t R₂ s).rd.isRead (.inl (un, q))) (hrd' : (F.zs R ω t R₂ s').rd.isRead (.inl (u', q'))) :
    FlockVBridge.Family.site ((F.zs R ω t R₂ s).rd.port (.inl (un, q))) =
        FlockVBridge.Family.site ((F.zs R ω t R₂ s').rd.port (.inl (u', q'))) ∧
      (F.zs R ω t R₂ s).rd.readAt (.inl (un, q)) = (F.zs R ω t R₂ s').rd.readAt (.inl (u', q')) := by
  obtain ⟨hmem, hname, hpos⟩ := hro R ω t R₂ τ o h s un q hrd
  obtain ⟨hmem', hname', hpos'⟩ := hro R ω t R₂ τ o h s' u' q' hrd'
  obtain ⟨rws, w, hrows, hw, hd⟩ := hsite R ω t R₂ τ o h s sh hr hss _ hmem (hname.trans hq)
  obtain ⟨_, -, -, _, _, hset, hfm⟩ := hchk R ω t R₂ s' ⟨_, (h.1 s').1⟩
  obtain ⟨h0, -⟩ := setup_zero_c (F.zs R ω t R₂ s').ht (F.zs R ω t R₂ s').hTags.1 (F.zs R ω t R₂ s').hcH hset
  obtain ⟨ss, s₁, hss₁, hj₁, hrw⟩ := checkRows_of_checkStmt (array_forM_ok hfm 0 h0)
  obtain rfl : ss = F.stmts := Except.ok.inj (hss₁.symm.trans F.hstmts)
  obtain rfl : s₁ = st' := Option.some.inj (hj₁.symm.trans hss')
  obtain ⟨rws', pb, w', hrows', hpb, hw', hd'⟩ := rowPort_of_checkRows hrw hrp
  obtain rfl : rws' = rws := Option.some.inj (hrows'.symm.trans hrows)
  obtain rfl : w' = w := Option.some.inj (hw'.symm.trans hw)
  obtain rfl : pb = (F.zs R ω t R₂ s').rd.port (.inl (u', q')) :=
    hnd R ω t R₂ s' _ (Array.mem_of_find?_eq_some hpb) _ hmem'
      ((by simpa using Array.find?_some hpb : pb.name = s!"m{hr}").trans (hname'.trans hq').symm)
  obtain ⟨hs1, hr1⟩ := site_reads_of_portDiff hd
  obtain ⟨hs2, hr2⟩ := site_reads_of_portDiff hd'
  refine ⟨hs1.trans hs2.symm, ?_⟩
  rw [hr1, ← hr2, hu] at hpos
  exact Option.some.inj (Option.some.inj (hpos.symm.trans hpos'))

/-- **`SrcHReads` from the verifier's checks** (`hSite_of_checked`) and the named pieces: no registered read
segmented (`ReadsUnsegC3`), the RecSparse circuit's `h` keys (`HKeys`), the algebra unit's row of rep `i`'s `h`
(`HAlgRows`) and the sparse `h` entry's registration (`SparseHSite`). -/
theorem srcHReads_of {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json} (hchk : ChecksAccepted F rows wr reg)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hro : ReadsOwnC3 F dg N k Rw)
    (hnd : ∀ R ω t R₂ j, ∀ p ∈ (F.zs R ω t R₂ j).own, ∀ p' ∈ (F.zs R ω t R₂ j).own, p.name = p'.name → p = p')
    (hsite : SparseHSite F rows dg N k Rw) (hseg : ReadsUnsegC3 F dg N k Rw)
    (hkeys : ∀ (s : Fin (vstarCount F.ish)) (j : ℕ), Flock.RecSparse.check (F.Vs s).c = .ok () →
      Flock.RecSparse.params (F.Vs s).c.unit.name = some (FlockC3.Sparse.unitShape (holoAt st hp)) →
      HKeys (F.Vs s) (Flock.RecSparse.messagePorts (FlockC3.Sparse.unitShape (holoAt st hp))) j)
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → C3Run (holoAt st hp)}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → Fin 2 →
      Array Flock.F128}
    (halg : HAlgRows F dg N k Rw stp w) : SrcHReads F.zs dg N k Rw stp w := by
  intro R ω t R₂ τ o hA i
  obtain ⟨U, hU⟩ := halg R ω t R₂ τ o hA i
  refine ⟨U, fun s un j hj c hcp hun => ?_⟩
  obtain ⟨sh, hr, hss, s', u', q', st', hu', hss', hq', hrp, hr', hbits⟩ := hU s un j hj c hcp hun
  have hk := hkeys s j hcp.1 hcp.2.1
  refine ⟨hk.1, fun l e he q b hb hkey hread => ?_⟩
  obtain ⟨hq, rfl⟩ := hk.2 l e he q b hb hkey
  obtain ⟨hb', hv⟩ := hbits l e he
  obtain ⟨hst, hat⟩ := hSite_of_checked hchk hro hnd hsite hA hss hss' hq hq' hrp (hun.trans hu'.symm) hread hr'
  exact ⟨hseg R ω t R₂ τ o hA s un q hread, s', u', q', hb', hr', hseg R ω t R₂ τ o hA s' u' q' hr', hst, hat, hv⟩

end family

end

end FlockC3.HPort
