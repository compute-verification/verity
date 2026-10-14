import Proofs.Flock.C3.Compose.Tied
import Proofs.Flock.C3.Sparse.SessionAt
import Proofs.Flock.VBridge.Family.Registrants

/-!
# The sparse phase's `h` from the rep's, across two units' reads of one registered row

The RecSparse unit reads its `h` port under `rec-c{hr}` at row `i` (`RecVStar.ownPorts (.sparse _ hr)`, which
`checkOwn` enforces), and rep `i`'s algebra unit reads the same registration's row `i` as its `m<hr>` port. Two
sessions' registered reads at one root and position are one row's bytes or a SHA-512 collision between the two
registrant records (`registrants_eq_or_collides`), and that collision is `RegColl`'s (`ReadsX.Clash`), which `Accepted`
excludes and `regFinder` prices once per audit: no new event, no guarantee record moved.

* `regRows_eq_or_clash`, `regVal_bit_eq`: at any sessions, two registered reads at one root and position hold one
  row's bytes or two records clash (both preimages kept); with no clash and both rows unsegmented, one row's bits.
* `rep_of_unit_h`, `rep_h_of_unit`: a rep unit plays `holoRepC` to an output whose `h` is the unit's `hOfW` (what
  `holoRep_refines` gives and `rep_of_unit` drops), so the tied run's rep `i` has that `h`.
* `HPortReads` (named): the cross-unit read, at one copy of the RecSparse unit and the values `x` of rep `i`'s `h`.
* `h_of_reads`: at no clash, a sparse copy at w (`SparseCopyAt`), and `HPortReads` at `hOfW` of some rep unit of rep
  `i`'s steps, w's first 64 elements are the tied rep's `h`.
* `SrcHReads` (named, family form), `SrcHTie`, `srcHTie_of`: the same at every accepted outer proof; `SrcHTie` is
  `SparseRepTie`'s `h` half at `tiedSrc stp`.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg FlockSoundness.Registered
open FlockSoundness.Discharge.HiddenExec (encPortH)

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type}

/-- **Two registered reads at one root and position hold one row's bytes, or two records clash**: the two rows' bytes
at their ports' encodings are one, or the two records' openings or hm96 leaves collide there, `RegColl`'s event at
these sessions. -/
theorem regRows_eq_or_clash (zs : (j : Fin S) → (Vs j).Outer Reg') {s s' : Fin S} {u : Fin (Vs s).nv}
    {u' : Fin (Vs s').nv} {q : Fin (Vs s).c.ports.size} {q' : Fin (Vs s').c.ports.size}
    (hr : (zs s).rd.isRead (.inl (u, q))) (hr' : (zs s').rd.isRead (.inl (u', q')))
    (hport : (zs s).rd.port (.inl (u, q)) = (zs s').rd.port (.inl (u', q')))
    (hat : (zs s).rd.readAt (.inl (u, q)) = (zs s').rd.readAt (.inl (u', q'))) :
    encPortH (Vs s).c.ports[q.val]! (FlockVBridge.Decode.regVal (zs s) (.inl (u, q))) =
        encPortH (Vs s').c.ports[q'.val]! (FlockVBridge.Decode.regVal (zs s') (.inl (u', q'))) ∨
      ∃ j j', (zs j).rd.Clash (zs j').rd := by
  rcases FlockVBridge.Family.registrants_eq_or_collides (zs s).rd (zs s').rd hr hr' hport hat with he | hc | hc
  · exact Or.inl he
  · exact Or.inr ⟨s, s', .inl (u, q), .inl (u', q'), hr, hr', Or.inl hc⟩
  · exact Or.inr ⟨s, s', .inl (u, q), .inl (u', q'), hr, hr', Or.inr hc⟩

/-- **Two registered reads at one root and position carry one row's bits** where no two records clash (`Accepted`'s
`¬ RegColl`) and neither row is segmented (`bits_of_encPortH`). -/
theorem regVal_bit_eq (zs : (j : Fin S) → (Vs j).Outer Reg') (hnc : ¬ ∃ j j', (zs j).rd.Clash (zs j').rd)
    {s s' : Fin S} {u : Fin (Vs s).nv} {u' : Fin (Vs s').nv} {q : Fin (Vs s).c.ports.size}
    {q' : Fin (Vs s').c.ports.size} (hr : (zs s).rd.isRead (.inl (u, q))) (hr' : (zs s').rd.isRead (.inl (u', q')))
    (hport : (zs s).rd.port (.inl (u, q)) = (zs s').rd.port (.inl (u', q')))
    (hat : (zs s).rd.readAt (.inl (u, q)) = (zs s').rd.readAt (.inl (u', q')))
    (hs : (Vs s).c.ports[q.val]!.segs = none) (hs' : (Vs s').c.ports[q'.val]!.segs = none) (b : ℕ)
    (hb : b < Layout.portRowW (Vs s).c q) (hb' : b < Layout.portRowW (Vs s').c q') :
    (FlockVBridge.Decode.regVal (zs s) (.inl (u, q))).2
        ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
      (FlockVBridge.Decode.regVal (zs s') (.inl (u', q'))).2
        ⟨b, lt_of_lt_of_le hb' ((Finset.le_sup (Finset.mem_univ q')).trans (Nat.le_succ _))⟩ := by
  rcases regRows_eq_or_clash zs hr hr' hport hat with he | hc
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

section point

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type}

/-- Obligation: the cross-unit read, at copy `j` of unit `un` of session `s` (a RecSparse copy) and the values `x` of
rep `i`'s `h`. Each column of `h`'s element `l`, bit `e` (message column `128 l + e`, `mp`'s first port) is an input
column a row reads (W's `SparseCols`), and where its key is bit `b` of a registered row `q` of the copy's unit, that row
is unsegmented and some session `s'`'s unit `u'` reads a registered row `q'` holding bit `b`, unsegmented, at the same
root and position, whose registrant bit `b` is bit `e` of `x l`. The root and position are `checkOwn`'s: the sparse
unit's `("h", rec-c{hr}, #[some 0, some 1])` entry and rep `i`'s algebra unit's `(m<hr>, rec-c{hr}, …)` entry, both at
row `i` (S's `OwnIn`, `mem_ownPorts_of_checkOwn`); bit `e` of `x l` at bit `b` of `m<hr>`'s row is the rep's decode of
its `h` (`hOfW_valOf` at `.h l`, `RepMaps`' columns of `w`). -/
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
          (zs s).rd.port (.inl (un, q)) = (zs s').rd.port (.inl (u', q')) ∧
          (zs s).rd.readAt (.inl (un, q)) = (zs s').rd.readAt (.inl (u', q')) ∧
          (regVal (zs s') (.inl (u', q'))).2
              ⟨b, lt_of_lt_of_le hb' ((Finset.le_sup (Finset.mem_univ q')).trans (Nat.le_succ _))⟩ =
            bitOf (x l) e

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
  obtain ⟨hseg, s', u', q', hb', hread', hseg', hport, hat, hv⟩ := hx.2 l t ht q b hb hkey hread
  rw [← hv, ← regVal_bit_eq zs hnc hread hread' hport hat hseg hseg' b hb hb']
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
`w` (`SparseCopyAt`, which `SparseSessionAt` gives), and some rep unit of rep `i`'s steps in `stp` (`RepTie`, with
`U.d` the rep's decode, as `RepUnitOfAlg` gives it). -/
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
            (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i) c s un j hj →
          HPortReads (zs R ω t R₂) (Flock.RecSparse.messagePorts (FlockC3.Sparse.unitShape (holoAt st hp))) s un j
            fun l => (hOfW U.sh U.d.w)[l.val]!

/-- **`SparseRepTie`'s `h` half at the tied source**: at every accepted outer proof, where some RecSparse copy holds
rep `i`'s phase at `w` (`SparseCopyAt`), w's first 64 elements are rep `i`'s `h` in `tiedSrc stp`. -/
def SrcHTie (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ (i : Fin 2) (s : Fin S) (un : Fin (Vs s).nv)
    (j : ℕ) (hj : j < (Vs s).c.upv) (c : FlockVBridge.Sparse.SCoins),
    FlockC3.Sparse.SparseCopyAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
        (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i) c s un j hj →
      ∀ l : Fin 64, fOf (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i)[l.val]! =
        ((tiedSrc stp R ω t (layers zs dg N k Rw R ω t R₂ τ)).rep i).2.h l

/-- **`SrcHTie` from the cross-unit reads** (`h_of_reads` at `Accepted`'s `¬ RegColl`). -/
theorem srcHTie_of {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128}
    (h : SrcHReads zs dg N k Rw stp w) : SrcHTie zs dg N k Rw stp w := by
  intro R ω t R₂ τ o hA i s un j hj c hcp
  obtain ⟨U, hU⟩ := h R ω t R₂ τ o hA i
  exact h_of_reads (u := stp R ω t (layers zs dg N k Rw R ω t R₂ τ)) hA.2 hcp U (hU s un j hj c hcp)

/-- **`SrcHTie` at the copy `SparseSessionAt` names**: the form `SparseRepTie`'s `h` conjunct takes. -/
theorem h_of_sessionAt {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128}
    (h : SrcHTie zs dg N k Rw stp w) {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂}
    {τ : (j : Fin S) → Strategy ((zs R ω t R₂ j).game dg N)}
    {o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript} (hA : Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o)
    {i : Fin 2} {c3 : Flock.RecCoins.C3} {unit : String} {c : FlockVBridge.Sparse.SCoins}
    (hs : FlockC3.Sparse.SparseSessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp) c3 unit i
      (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i) c) :
    ∀ l : Fin 64, fOf (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i)[l.val]! =
      ((tiedSrc stp R ω t (layers zs dg N k Rw R ω t R₂ τ)).rep i).2.h l := by
  obtain ⟨-, -, s, un, j, hj, hcp⟩ := hs
  exact h R ω t R₂ τ o hA i s un j hj c hcp

end src

end

end FlockC3.HPort
