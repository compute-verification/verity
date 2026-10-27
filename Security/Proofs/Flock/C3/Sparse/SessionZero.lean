import Proofs.Flock.C3.Sparse.SessionReg
import Proofs.Flock.C3.Sparse.SrcLayer
import Proofs.Flock.C3.RegLevels.OutRegs
import Proofs.Flock.Recursive.PartTie

/-!
# `sparseSessionAt_of_regs`' zero outputs, from check 5's `checkZero`

Check 5 holds the sparse unit's `out` to the zero rows `rec-sparse-zero` under `zeroSalt` (`checkSparse`'s `checkZero`),
so off `ValueColl` the registrant's bits at an `out` read are 0, and copy 0's output columns, which the unit's output row
holds, are registered as 0 (`sparseSessionAt_of_regs`' `hz`):

* `sparseZero_tie`: `zero_tie` at a `rec-sparse` unit (`sparseCheck_of_check`, `sparse_of_check`,
  `innerRows_sparseOut`, `value_tie_collR` at `valueSrcAll_zero`; META's `program_sha512` from `checkRows_root`);
* `SparseOutsLaid`, `outBitRow_sparse`: at the sparse unit's output layout (one port `out` of `outWords` words, the
  first past the input rows, copy `j` returning words in order; a fact of V*'s circuit file, which `RecSparse.check`
  does not read, as `OutsLaid` at R's levels), bit `b` of the output is bit `b` of row `nIn`;
* `outCols_of_sparse`, `regOuts_sparse`: at a unit built from `unitLaid`, the output starts at `sparseBase`, and column
  `sparseBase + b` of copy `j` is registered as bit `b` of row `nIn` where that row is a registered read
  (`regOut_of_outBitRow`);
* `sparseSessionAt_of_zero`: `sparseSessionAt_of_regs` with `hz` from these, at the session's `out` read (`hout`) and
  off `ValueColl` at the family's sources (`hnc`).
-/

namespace FlockC3.Sparse

noncomputable section

open Flock
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge FlockVBridge.AccChain

/-! ## The registrant's `out` bits -/

/-- **A sparse unit's `out` read is zero, or `ValueColl`** (`zero_tie` at `rec-sparse`): where check 5 accepted the
`rec-sparse` unit at a source `f` and a coins record `C`, at a registered read `(u, q)` of its `out` port (without
segments), the read is of one of the two zero rows (`rec-sparse-zero`, `checkZero`), and every bit the registrant holds
there is 0. -/
theorem sparseZero_tie {n : ℕ} {L : Law n} {Reg Cm Reg₂ Reg' : Type} {I : Inner} {S : ℕ} {Vs : Fin S → VStmt}
    {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {src : List (Cm × I.Coin) → Option Flock.RecCoins.Source} {c3 : List (Cm × I.Coin) → Option Flock.RecCoins.C3}
    {f : Flock.RecCoins.Source} {C : Flock.RecCoins.C3} {R : Reg} {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂}
    {j : Fin S} {sj : Flock.Stmt} {u : Fin (Vs j).nv} {q : Fin (Vs j).c.ports.size} (hc : sj.c = (Vs j).c)
    (hck : Flock.RecCoins.check sj (some (zs R ω t R₂ j).own) (some f) (some C) = .ok ()) (hf : src t = some f)
    (h3 : c3 t = some C) (hso : Flock.RecCoins.stackedParams (Vs j).c.unit.name = none)
    (hop : Flock.RecOpening.params (Vs j).c.unit.name = none) {sh : Flock.RecSparse.Shape}
    (hsp : Flock.RecSparse.params (Vs j).c.unit.name = some sh) (hname : ((Vs j).c.ports[q.val]!).name = "out")
    (hp : (zs R ω t R₂ j).rd.isRead (.inl (u, q)))
    (hone : ∀ reg ∈ (zs R ω t R₂ j).own, reg.name = ((Vs j).c.ports[q.val]!).name →
      reg = (zs R ω t R₂ j).rd.port (.inl (u, q)))
    (hpos : ((zs R ω t R₂ j).rd.port (.inl (u, q))).reads[u.val]? =
      some (some ((zs R ω t R₂ j).rd.readAt (.inl (u, q)))))
    (hseg : (Vs j).c.ports[q.val]!.segs = none) :
    u.val < 2 ∧
      ((∀ (b : ℕ) (hb : b < FlockSoundness.Discharge.Layout.portRowW (Vs j).c q),
        (FlockVBridge.Decode.regVal (zs R ω t R₂ j) (.inl (u, q))).2
            ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ = false) ∨
      ValueColl zs (valueSrcAll src c3) R ω t R₂) := by
  obtain ⟨inn, hl, hck'⟩ := sparseCheck_of_check hck (by rw [hc]; exact hso) (by rw [hc]; exact hop)
    (by rw [hc]; exact hsp)
  obtain ⟨-, -, -, -, hz⟩ := sparse_of_check hck'
  rw [hc] at hl hz
  obtain ⟨-, prog, -, -, -, hprog, -⟩ := FlockC3.RegLevels.checkRows_root hz
  have hv : valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name =
      some (Flock.RecCoins.zeroRows 2 (Flock.RecSparse.outWords sh), Flock.RecCoins.SPARSE_ZERO_VALUE) := by
    rw [hf, h3, valueRowsAll_load hso hop hl, hname]
    exact innerRows_sparseOut hsp
  obtain ⟨hu, hor⟩ := value_tie_collR (zs := zs) (vsrc := valueSrcAll src c3)
    (salt := fun _ => Flock.RecCoins.zeroSalt Flock.RecCoins.SPARSE_ZERO_VALUE) hz hprog
    (valueSrcAll_zero hv hprog (hz := Or.inr rfl)) hp (fun reg hr hr' => hone reg hr (hr'.trans hname.symm)) hpos hseg
  refine ⟨by simpa only [Flock.RecCoins.zeroRows, Array.size_replicate] using hu, hor.imp (fun he b hb => ?_) id⟩
  rw [he b hb, zeroRows_get, Nat.zero_testBit]

/-! ## The sparse unit's output row -/

/-- **V*'s output layout at copy `j` of the sparse unit** (`rec_sparse.unit_circuit`, `ir_lower.layout`): one output
port, `out`, of `outWords sh` words, the first past the input rows, copy `j` returning words `0 .. outWords sh - 1` in
order. A fact of V*'s circuit file, which `RecSparse.check` does not read (it reads the unit), as `OutsLaid` is at
R's levels. -/
def SparseOutsLaid (c : Flock.Circuit) (sh : Flock.RecSparse.Shape) (j : ℕ) : Prop :=
  c.outPorts = #[("out", Flock.RecSparse.outWords sh)] ∧
    c.leavesOut.getD j #[] = Array.range (Flock.RecSparse.outWords sh) ∧
    ∃ hq : HmOut.nIn c < c.ports.size, (c.ports[HmOut.nIn c]'hq).name = "out" ∧
      (c.ports[HmOut.nIn c]'hq).words = Flock.RecSparse.outWords sh

/-- **Bit `b` of the sparse unit's output is bit `b` of output row `nIn`** (`SparseOutsLaid`). -/
theorem outBitRow_sparse {c : Flock.Circuit} {sh : Flock.RecSparse.Shape} {j : ℕ} (h : SparseOutsLaid c sh j) {n : ℕ}
    (u : Fin n) {b : ℕ} (hb : b < 16 * Flock.RecSparse.outWords sh) :
    ∃ hq : HmOut.nIn c < c.ports.size, ∃ hk : b < Layout.portRowW c ⟨_, hq⟩,
      HiddenExec.outBitRow c u (j, b / 16, b % 16) = some (u, ⟨⟨_, hq⟩, ⟨b, hk⟩⟩) := by
  obtain ⟨hop, hlo, hq, -, hw⟩ := h
  have hW : HiddenExec.outWordOf c (j, b / 16, b % 16) = b / 16 := by
    show (c.leavesOut.getD j #[])[b / 16]! = b / 16
    rw [hlo, getElem!_pos (Array.range _) (b / 16) (by rw [Array.size_range]; omega)]
    simp
  have hs : c.outPorts.size = 1 := by rw [hop] <;> rfl
  have hA0 : Hidden.outAt c 0 = 0 := by simp [Hidden.outAt]
  have hN : c.outWords = Flock.RecSparse.outWords sh := by
    rw [← HiddenExec.outAt_size c, hs]
    unfold Hidden.outAt
    rw [hop] <;> rfl
  have hP : HiddenExec.outPortOf c (b / 16) = 0 := by
    have := (HiddenExec.outPortOf_spec c (o := b / 16) (by rw [hN]; omega)).1
    rw [hs] at this
    omega
  have hk : b < Layout.portRowW c ⟨_, hq⟩ := by
    show b < 16 * (c.ports[HmOut.nIn c]'hq).words
    rw [hw]
    exact hb
  refine ⟨hq, hk, FlockC3.RegLevels.outBitRow_eq u hq (by rw [hW, hP, Nat.add_zero]) ?_ hk⟩
  rw [hW, hP, hA0]
  show 16 * (b / 16 - 0) + b % 16 = b
  omega

section Copy

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

/-- **A unit built from `unitLaid` returns its outputs from `sparseBase`**: the output group's first column, within
the computed columns, past at least the outputs' bits (`outCols_of_laid`'s twin). -/
theorem outCols_of_sparse {sh : Flock.RecSparse.Shape} {name : String}
    (hl : Flock.HmNets.netOf name (Flock.RecSparse.unitLaid sh) = .ok c.unit) :
    (c.unit.outGroups.getD 0 default).col * Flock.WORD = FlockVBridge.Sparse.sparseBase sh ∧
      (Rows.ofNet hU).nIn ≤ FlockVBridge.Sparse.sparseBase sh ∧
      FlockVBridge.Sparse.sparseBase sh + 16 * Flock.RecSparse.outWords sh ≤
        (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp := by
  have e : Flock.RecSparse.unitLaid sh = Flock.HmNets.layout ((Flock.RecSparse.unitNet' sh).run Flock.HmNets.St.empty).2
      (Array.replicate (Flock.RecSparse.inWords sh) 16) (Array.replicate (Flock.RecSparse.outWords sh) 16)
      ((Flock.RecSparse.unitNet' sh).run Flock.HmNets.St.empty).1.toArray := rfl
  obtain ⟨-, -, -, -, hcp, -, hog, -⟩ := Flock.Net.ofRows_ok hl
  obtain ⟨h1, -⟩ := Flock.HmNets.layout_rows ((Flock.RecSparse.unitNet' sh).run Flock.HmNets.St.empty).2
    (Array.replicate (Flock.RecSparse.inWords sh) 16) (Array.replicate (Flock.RecSparse.outWords sh) 16)
    ((Flock.RecSparse.unitNet' sh).run Flock.HmNets.St.empty).1.toArray
  have hsum : (Array.replicate (Flock.RecSparse.outWords sh) 16).foldl (· + ·) 0 = 16 * Flock.RecSparse.outWords sh := by
    rw [FlockVBridge.OfVStar.foldl_replicate]
    exact Nat.mul_comm _ _
  have hob := Flock.HmNets.le_groupWords ((Array.replicate (Flock.RecSparse.outWords sh) 16).foldl (· + ·) 0)
  refine ⟨?_, ?_, ?_⟩
  · rw [hog, e, Hm.layout_outGroups]
    exact Hm.outStart_word _ _ _
  · rw [FlockSoundness.Discharge.Layout.ofNet_nIn_eq (hU := hU), FlockVBridge.Sparse.inWords_sparse hl]
    exact Decode.inRows_le_outStart _ _ _
  · rw [show (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp = c.unit.constPos from Rows.ofNet_one hU, hcp, e, h1]
    unfold FlockVBridge.Sparse.sparseBase FlockVBridge.Sparse.sparseInBits Hm.outStart Hm.inRowsOf Hm.spanOf
    omega

/-- **The sparse unit's output columns are registered where its output row is a registered read**
(`regOut_of_outBitRow` at `outBitRow_sparse`): at a unit built from `unitLaid`, at its output layout
(`SparseOutsLaid`), column `sparseBase + b` of copy `j` is registered as bit `b` of the registrant's row `nIn`. -/
theorem regOuts_sparse {u : Fin nv} {j : ℕ} (hj : j < c.upv) {sh : Flock.RecSparse.Shape} {name : String}
    (hl : Flock.HmNets.netOf name (Flock.RecSparse.unitLaid sh) = .ok c.unit) (hlaid : SparseOutsLaid c sh j)
    (hread : ∀ hq : HmOut.nIn c < c.ports.size, Decode.regRead zo (.inl (u, ⟨_, hq⟩))) {b : ℕ}
    (hb : b < 16 * Flock.RecSparse.outWords sh) :
    ∃ hq : HmOut.nIn c < c.ports.size, ∃ hk : b < Layout.portRowW c ⟨_, hq⟩,
      RegOut zo u hj (FlockVBridge.Sparse.sparseBase sh + b)
        ((Decode.regVal zo (.inl (u, ⟨_, hq⟩))).2 (Fin.castLE ((Finset.le_sup (f := Layout.portRowW c)
          (Finset.mem_univ (⟨HmOut.nIn c, hq⟩ : Fin c.ports.size))).trans (Nat.le_succ _)) ⟨b, hk⟩)) := by
  obtain ⟨hcW, hIn, hEnd⟩ := outCols_of_sparse (hU := hU) hl
  obtain ⟨hq, hk, hrow⟩ := outBitRow_sparse hlaid u hb
  have hj' : b / 16 < (c.leavesOut.getD j #[]).size := by
    rw [hlaid.2.1, Array.size_range]
    omega
  have hcol : Layout.outCol c (b / 16) (b % 16) = FlockVBridge.Sparse.sparseBase sh + b := by
    unfold Layout.outCol
    rw [hcW]
    omega
  refine ⟨hq, hk, ?_⟩
  rw [← hcol]
  exact FlockC3.RegLevels.regOut_of_outBitRow zo hj hj' (Nat.mod_lt _ (by decide)) (by rw [hcol]; omega)
    (by rw [hcol]; omega) hrow (hread hq)

end Copy

/-! ## The discharge -/

section Family

open Game Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose
open FlockVBridge.Decode (RegBit)
open FlockVBridge.OfVStar (ReadCol rowCol)
open FlockVBridge.Sparse (SCoins nw verifierRow tauBase vBaseS sparseBase)

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(n7) `sparseSessionAt_of_regs` with its zero outputs from check 5** (`hz`): at the session's `out` read of copy 0
of unit `u` (`hout`: a registered read, the reads file's one `out` entry, at the row its reads name, without
segments), at the sparse unit's output layout (`hlaid`), and off `ValueColl` at the family's sources (`hnc`), each of
copy 0's residual output columns is registered as 0: `sparseZero_tie` at check 5's accepted statement
(`sparseCheck_at`), `regOuts_sparse` at the unit's layout (`check_sparse`). -/
theorem sparseSessionAt_of_zero (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}
    {coins : String → Option Flock.RecCoins.Source} (hchk : CoinsAccepted F rows wr reg coins)
    (hsp : FlockC3.VStarList.SparseIs F.ish hp) (hell : hp.EllOk (execClassAt st).cls.kLog)
    {s : Fin (vstarCount F.ish)} {sh : Flock.RecSparse.Shape} {hR : ℕ} (hs : F.stmts[(s : ℕ)]? = some (.sparse sh hR))
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {R : Reg} {ω : L.Ω} {t : List (Cm × (holoInnerAt T CT st hp).Coin)}
    {R₂ : Reg₂} {τ : (j : Fin (vstarCount F.ish)) → Strategy ((F.zs R ω t R₂ j).game dg N)}
    {o : (j : Fin (vstarCount F.ish)) → Reg' × Finset (Fin (F.Vs j).nv) × Exec.Transcript}
    (hA : Accepted (I := holoInnerAt T CT st hp) F.zs dg N k Rw R ω t R₂ τ o) (i : Fin 2) (u : Fin (F.Vs s).nv)
    (hu : u.val = i.val) {C : Flock.RecCoins.C3}
    (hC : ∀ cs, innerCommitsAt (wr t).1 (wr t).2 st reg = .ok cs → Flock.Guarantees.c3Of F.ish (some cs) = some C)
    {c : SCoins} (hcs : c.cs = csOf C (Flock.RecSparse.unitName (unitShape (holoAt st hp))) i.val)
    (hone : layers F.zs dg N k Rw R ω t R₂ τ s
      (Const.Prog.oneAt (Placed.keyProg (F.Vs s).c (F.Vs s).hU (F.Vs s).nv) u) = true)
    (hkeys : ∀ q < 128 * ((Flock.RecSparse.messagePorts (unitShape (holoAt st hp))).map (·.2)).sum,
      ReadCol (Rows.ofNet (F.Vs s).hU) (rowCol (Flock.RecSparse.messagePorts (unitShape (holoAt st hp))) 0 q) →
        ∃ b, RegBit (F.zs R ω t R₂ s) u 0 (rowCol (Flock.RecSparse.messagePorts (unitShape (holoAt st hp))) 0 q) b)
    (hv : ∀ q < 128 * Flock.RecSparse.nv (unitShape (holoAt st hp)),
      RegBit (F.zs R ω t R₂ s) u 0 (vBaseS (unitShape (holoAt st hp)) + q)
        (wbit ((verifierRow (holoAt st hp).cls.shape c)[·]!) q))
    (htau : (unitShape (holoAt st hp)).stacked = true → ∀ x < (unitShape (holoAt st hp)).S, ∀ b < 16,
      RegBit (F.zs R ω t R₂ s) u 0 (tauBase (unitShape (holoAt st hp)) + (16 * x + b))
        (((tauArr (holoAt st hp))[x]!).testBit b))
    (hlaid : SparseOutsLaid (F.Vs s).c (unitShape (holoAt st hp)) 0)
    (hout : ∀ hq : HmOut.nIn (F.Vs s).c < (F.Vs s).c.ports.size,
      (F.zs R ω t R₂ s).rd.isRead (.inl (u, ⟨_, hq⟩)) ∧
      (∀ pt ∈ (F.zs R ω t R₂ s).own, pt.name = "out" → pt = (F.zs R ω t R₂ s).rd.port (.inl (u, ⟨_, hq⟩))) ∧
      ((F.zs R ω t R₂ s).rd.port (.inl (u, ⟨_, hq⟩))).reads[u.val]? =
        some (some ((F.zs R ω t R₂ s).rd.readAt (.inl (u, ⟨_, hq⟩)))) ∧
      ((F.Vs s).c.ports[HmOut.nIn (F.Vs s).c]!).segs = none)
    (hnc : ¬ ValueColl F.zs (valueSrcAll (fun t => coins (wr t).2)
      (fun t => Flock.Guarantees.c3Of F.ish (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption)) R ω t R₂) :
    SparseSessionAt (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ) (holoAt st hp) C
        (Flock.RecSparse.unitName (unitShape (holoAt st hp))) i
        (wSparse F.Vs (holoAt st hp).cls.shape (Flock.RecSparse.messagePorts (unitShape (holoAt st hp)))
          (layers F.zs dg N k Rw R ω t R₂ τ) s u) c ∧
      ∃ (s' : Fin (vstarCount F.ish)) (un : Fin (F.Vs s').nv) (j : ℕ) (hj : j < (F.Vs s').c.upv), un.val = i.val ∧
        SparseCopyAt (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
          (wSparse F.Vs (holoAt st hp).cls.shape (Flock.RecSparse.messagePorts (unitShape (holoAt st hp)))
            (layers F.zs dg N k Rw R ω t R₂ τ) s u) c s' un j hj := by
  refine sparseSessionAt_of_regs F hchk hsp hell hs hA i u hu hC hcs hone hkeys hv htau fun x hx => ?_
  obtain ⟨sj, f, c3, hc, hf, hc3, hck, hpar⟩ := sparseCheck_at F hchk hsp hell hs ⟨(o s).2.2, (hA.1 s).1⟩
  have hname : (F.Vs s).c.unit.name = Flock.RecSparse.unitName (unitShape (holoAt st hp)) :=
    (FlockVBridge.Sparse.unitName_params hpar).symm
  obtain ⟨-, -, -, hn⟩ := FlockVBridge.Sparse.check_sparse (parse_facts (F.zs R ω t R₂ s).hc).recSparse hpar
  have hn' : Flock.HmNets.netOf (Flock.RecSparse.unitName (unitShape (holoAt st hp)))
      (Flock.RecSparse.unitLaid (unitShape (holoAt st hp))) = .ok (F.Vs s).c.unit := hn
  have hx' : x < 16 * Flock.RecSparse.outWords (unitShape (holoAt st hp)) := by
    have := FlockVBridge.Sparse.outWords_ge (unitShape (holoAt st hp))
    omega
  obtain ⟨hq, hk, hreg⟩ := regOuts_sparse (F.zs R ω t R₂ s) (u := u) (FlockC3.Reps.upv_of_outer (F.zs R ω t R₂ s))
    hn' hlaid (fun hq => (hout hq).1) hx'
  obtain ⟨hrd, hown, hpos, hseg⟩ := hout hq
  obtain ⟨hq', hnm, -⟩ := hlaid.2.2
  have hname' : ((F.Vs s).c.ports[(⟨HmOut.nIn (F.Vs s).c, hq⟩ : Fin (F.Vs s).c.ports.size).val]!).name = "out" := by
    show ((F.Vs s).c.ports[HmOut.nIn (F.Vs s).c]!).name = "out"
    rw [getElem!_pos (F.Vs s).c.ports _ hq]
    exact hnm
  obtain ⟨-, hor⟩ := sparseZero_tie (zs := F.zs) (src := fun t => coins (wr t).2)
    (c3 := fun t => Flock.Guarantees.c3Of F.ish (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption) hc hck hf hc3
    (by rw [hname]; exact stackedParams_sparse _) (by rw [hname]; exact openingParams_sparse _) hpar hname' hrd
    (fun pt hpt hpn => hown pt hpt (hpn.trans hname')) hpos hseg
  rcases hor with hbits | hcoll
  · convert hreg using 1
    exact (hbits x hk).symm
  · exact absurd hcoll hnc

/-- **The inner-session record decodes at an accepted candidate-3 proof**: the sparse statement's check 5 loads its
source; `hfiles` identifies it with the files of this record (the verifier's setup gives that form at an accepted
candidate-3 session, `FlockC3.VStarList.hcoins_of_acceptsC3`), and `Inner.load` decodes it. -/
theorem record_decodes_of_accepted (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}
    {coins : String → Option Flock.RecCoins.Source} (hchk : CoinsAccepted F rows wr reg coins)
    (hsp : FlockC3.VStarList.SparseIs F.ish hp) (hell : hp.EllOk (execClassAt st).cls.kLog)
    {s : Fin (vstarCount F.ish)} {sh : Flock.RecSparse.Shape} {hR : ℕ}
    (hs : F.stmts[(s : ℕ)]? = some (.sparse sh hR))
    (hfiles : ∀ r src, coins r = some src → ∃ f, src = .files f ∧ f.record = r)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {R : Reg} {ω : L.Ω}
    {t : List (Cm × (holoInnerAt T CT st hp).Coin)} {R₂ : Reg₂}
    {τ : (j : Fin (vstarCount F.ish)) → Strategy ((F.zs R ω t R₂ j).game dg N)}
    {o : (j : Fin (vstarCount F.ish)) → Reg' × Finset (Fin (F.Vs j).nv) × Exec.Transcript}
    (hA : Accepted (I := holoInnerAt T CT st hp) F.zs dg N k Rw R ω t R₂ τ o) :
    ∃ spec sess, Flock.Record.decode spec (wr t).2 = .ok sess := by
  obtain ⟨sj, src, c3, hc, hsrc, -, hck, hpar⟩ :=
    sparseCheck_at F hchk hsp hell hs ⟨(o s).2.2, (hA.1 s).1⟩
  obtain ⟨f, rfl, hrecord⟩ := hfiles (wr t).2 src hsrc
  have hname : (F.Vs s).c.unit.name = Flock.RecSparse.unitName (unitShape (holoAt st hp)) :=
    (FlockVBridge.Sparse.unitName_params hpar).symm
  obtain ⟨inn, hload, -⟩ := FlockSoundness.Discharge.Recursion.sparseCheck_of_check hck
    (by rw [hc, hname]; exact stackedParams_sparse _)
    (by rw [hc, hname]; exact openingParams_sparse _) (by rw [hc]; exact hpar)
  obtain ⟨spec, sess, hdec⟩ := FlockSoundness.Discharge.Recursion.record_decode_of_load hload
  exact ⟨spec, sess, by simpa [hrecord] using hdec⟩

end Family

end

end FlockC3.Sparse
