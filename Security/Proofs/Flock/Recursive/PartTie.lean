import Proofs.Flock.Recursive.ValueChecked
import Proofs.Flock.C3.RegLevels.VRows
import Proofs.Flock.C3.RegLevels.OutRegs

/-!
# The value tie at a part's `v` and `out`: `VRowsAre`, `ZeroReads` and `OwnNamesV` at one read

At a `rec-residuals` unit check 5 accepted (`residualsCheck_of_check`, `residualsBoth_of_check`), off `ValueColl` at the
verifier's rows (`valueSrcAll`), the obligations of `Family/VCoins.lean` at one part's session:

* `vRows_tie`: the registrant's bits at a `v` read of rep `u` are `wbit` of part `i`'s elements of `vOf` at the rep's
  rounds, extra claims and comb (`value_tie_residuals` at `residualRows`, then `FlockC3.RegLevels.rows_vOf` at
  `residualRowsOf`'s rows), `VRowsAre`'s bit;
* `zero_tie`: the registrant's bits at an `out` read are 0 (`checkZero` at `rec-residuals-zero`, `value_tie_collR` at
  the zero rows), `ZeroReads`' bit at the read `(u, q)`;
* `own_v_out`: the part's reads file names `v` and `out` (`own_of_checkValue`, `own_of_checkRows`), `OwnNamesV`;
* `PartOutsLaid`, `outBitRow_part`, `zeroOut_of_laid`, `zero_regOut`: at V*'s part output layout (one output row `out` of
  128 words), copy `j`'s output bit `e` is bit `e` of the `out` row (`HiddenExec.outBitRow`), so the gate at its column
  is at the `out` read (`rowOfH`, `FlockC3.RegLevels.regOut_of_outBitRow`) and `zero_tie`'s 0 is its bit there:
  `RegOut … false`, `ZeroReads` at one column.

The family's obligations hold these at every accepted outer proof; there they need `¬ ValueColl` threaded (the
guarantee's disjunct, `VBridgeCOff`), the rep's rounds and comb as the family's (`Family.InnerIsRep`), and `hone`,
`hpos` from the family's reads (`Family.readOne_of`).
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit

variable {n : ℕ} {L : Law n} {Reg Cm Reg₂ Reg' : Type} {I : Inner} {S : ℕ} {Vs : Fin S → VStmt}
  {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
  {src : List (Cm × I.Coin) → Option Flock.RecCoins.Source} {c3 : List (Cm × I.Coin) → Option Flock.RecCoins.C3}
  {f : Flock.RecCoins.Source} {R : Reg} {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂}
  {j : Fin S} {sj : Flock.Stmt} {prog : String} {u : Fin (Vs j).nv} {q : Fin (Vs j).c.ports.size}

theorem residualRowsOf_of {inn : Flock.RecCoins.Inner} {i : ℕ} {rows : Array (Array ℕ)} {sh : Flock.Shape}
    {extras : Array Flock.Claim} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} {a b vs : Array ℕ}
    (halg : inn.algebra = .ok (sh, extras)) (hgs : residualGroups sh = .ok gs) (hgi : gs[i]? = some (a, b, vs))
    (h : residualRows inn i = some rows) : residualRowsOf inn sh extras vs = .ok rows := by
  rcases hR : residualRowsOf inn sh extras vs with e | r
  · simp only [residualRows, halg, hgs, hgi, hR] at h <;> cases h
  · simp only [residualRows, halg, hgs, hgi, hR, Option.some.injEq] at h
    rw [h]

/-- **A part's `v` read is `vOf`'s bits, or `ValueColl`** (`VRowsAre` at one read): where check 5 accepted the
`rec-residuals` unit of part `i` at the inner session `inn` that `f` loads, whose algebra is `(sh, extras)` and whose
part `i` has the elements `vs`, and the rep `u` the check reads has the rounds `rounds` and, where `vs` reaches the comb,
the comb `comb`, the registrant's bit `b` at a `v` read `(u, q)` is `wbit` of `vs`'s elements of `vOf` there. -/
theorem vRows_tie (hc : sj.c = (Vs j).c)
    (hck : Flock.RecCoins.check sj (some (zs R ω t R₂ j).own) (some f) (c3 t) = .ok ()) (hf : src t = some f)
    (hso : Flock.RecCoins.stackedParams (Vs j).c.unit.name = none)
    (hop : Flock.RecOpening.params (Vs j).c.unit.name = none) (hsp : Flock.RecSparse.params (Vs j).c.unit.name = none)
    (ho : Flock.RecOpen.params (Vs j).c.unit.name = none) {m k cl i : ℕ}
    (hres : Flock.RecResiduals.params (Vs j).c.unit.name = some (m, k, cl, i))
    (hname : ((Vs j).c.ports[q.val]!).name = "v")
    (hprog : (Vs j).c.metaJ.getObjValAs? String "program_sha512" = .ok prog)
    (hp : (zs R ω t R₂ j).rd.isRead (.inl (u, q)))
    (hone : ∀ reg ∈ (zs R ω t R₂ j).own, reg.name = ((Vs j).c.ports[q.val]!).name →
      reg = (zs R ω t R₂ j).rd.port (.inl (u, q)))
    (hpos : ((zs R ω t R₂ j).rd.port (.inl (u, q))).reads[u.val]? =
      some (some ((zs R ω t R₂ j).rd.readAt (.inl (u, q)))))
    (hseg : (Vs j).c.ports[q.val]!.segs = none)
    {inn : Flock.RecCoins.Inner} (hl : f.load (Vs j).c.unit.name = .ok inn) {sh : Flock.Shape}
    {extras : Array Flock.Claim} (halg : inn.algebra = .ok (sh, extras))
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hgs : residualGroups sh = .ok gs) {a b' vs : Array ℕ}
    (hgi : gs[i]? = some (a, b', vs)) {rounds : Array (Array Flock.F128)} {comb : Array Flock.F128}
    (hrep : ∀ rep, inn.rep sh extras u.val = .ok rep →
      rep.rounds = rounds ∧ (vs.any (· ≥ rep.v.size - 64) = true → inn.comb u.val rep = .ok comb))
    (hnc : ¬ ValueColl zs (valueSrcAll src c3) R ω t R₂) :
    ∀ b < 128 * vs.size, ∀ hb : b < FlockSoundness.Discharge.Layout.portRowW (Vs j).c q,
      (FlockVBridge.Decode.regVal (zs R ω t R₂ j) (.inl (u, q))).2
          ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
        FlockVBridge.wbit (FlockVBridge.Decode.sub vs (FlockVBridge.Decode.vOf sh rounds extras comb)) b := by
  obtain ⟨rows, value, hv, hu, hor⟩ :=
    value_tie_residuals hc hck hf hso hop hsp ho hres hname hprog hp hone hpos hseg
  obtain ⟨inn', hl', hck'⟩ := residualsCheck_of_check hck (by rw [hc]; exact hso) (by rw [hc]; exact hop)
    (by rw [hc]; exact hsp) (by rw [hc]; exact ho) (by rw [hc]; exact hres)
  rw [hc, hl] at hl'
  obtain rfl : inn = inn' := Except.ok.inj hl'
  obtain ⟨rows', -, hrr, -⟩ := residuals_of_check hck'
  have hv' : valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name =
      some (rows', Flock.RecCoins.V_VALUE) := by
    rw [hf, valueRowsAll_load hso hop hl, hname]
    exact innerRows_v hsp ho hres hrr
  have hrw : rows = rows' := by
    have h2 := hv.symm.trans hv'
    simp only [Option.some.injEq, Prod.mk.injEq] at h2
    exact h2.1
  subst hrw
  obtain ⟨hsz, hall⟩ := FlockC3.RegLevels.rows_vOf (inn := inn) (sh := sh) (extras := extras) (vs := vs)
    (rows := rows) (residualRowsOf_of halg hgs hgi hrr)
  obtain ⟨rep, hrp, hbits⟩ := hall u.val (by omega)
  obtain ⟨hrd, hcomb⟩ := hrep rep hrp
  intro b hbv hb
  rcases hor with he | hcoll
  · rw [he b hb, hbits comb hcomb b hbv, hrd]
  · exact absurd hcoll hnc

/-- **A part's `out` read is zero, or `ValueColl`** (`ZeroReads`' bit at the read `(u, q)`): where check 5 accepted
the `rec-residuals` unit of part `i`, at a registered read `(u, q)` of its `out` port (without segments), the read is of
one of the two zero rows (`rec-residuals-zero`, `checkZero`), and every bit the registrant holds there is 0. -/
theorem zero_tie (hc : sj.c = (Vs j).c)
    (hck : Flock.RecCoins.check sj (some (zs R ω t R₂ j).own) (some f) (c3 t) = .ok ()) (hf : src t = some f)
    (hso : Flock.RecCoins.stackedParams (Vs j).c.unit.name = none)
    (hop : Flock.RecOpening.params (Vs j).c.unit.name = none) (hsp : Flock.RecSparse.params (Vs j).c.unit.name = none)
    (ho : Flock.RecOpen.params (Vs j).c.unit.name = none) {m k cl i : ℕ}
    (hres : Flock.RecResiduals.params (Vs j).c.unit.name = some (m, k, cl, i))
    (hname : ((Vs j).c.ports[q.val]!).name = "out")
    (hprog : (Vs j).c.metaJ.getObjValAs? String "program_sha512" = .ok prog)
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
  obtain ⟨inn, hl, hck'⟩ := residualsCheck_of_check hck (by rw [hc]; exact hso) (by rw [hc]; exact hop)
    (by rw [hc]; exact hsp) (by rw [hc]; exact ho) (by rw [hc]; exact hres)
  obtain ⟨-, -, -, -, hz⟩ := residualsBoth_of_check hck'
  rw [hc] at hl hz
  have hv : valueRowsAll (src t) (c3 t) (Vs j).c ((Vs j).c.ports[q.val]!).name =
      some (Flock.RecCoins.zeroRows 2 Flock.RecCoins.OUT_WORDS, Flock.RecCoins.ZERO_VALUE) := by
    rw [hf, valueRowsAll_load hso hop hl, hname]
    exact innerRows_out hsp ho hres
  obtain ⟨hu, hor⟩ := value_tie_collR (zs := zs) (vsrc := valueSrcAll src c3)
    (salt := fun _ => Flock.RecCoins.zeroSalt Flock.RecCoins.ZERO_VALUE) hz hprog (valueSrcAll_zero hv hprog) hp
    (fun reg hr hr' => hone reg hr (hr'.trans hname.symm)) hpos hseg
  refine ⟨by simpa only [Flock.RecCoins.zeroRows, Array.size_replicate] using hu, hor.imp (fun he b hb => ?_) id⟩
  rw [he b hb, zeroRows_get, Nat.zero_testBit]

/-- **A part's reads file names `v` and `out`** (`OwnNamesV` at one session): check 5 accepted the part's
`rec-residuals` unit, whose `checkValue` and `checkZero` find each in the reads file. -/
theorem own_v_out {own : Array Flock.Registered.Port}
    (hck : Flock.RecCoins.check sj (some own) (some f) (c3 t) = .ok ())
    (hso : Flock.RecCoins.stackedParams sj.c.unit.name = none) (hop : Flock.RecOpening.params sj.c.unit.name = none)
    (hsp : Flock.RecSparse.params sj.c.unit.name = none) (ho : Flock.RecOpen.params sj.c.unit.name = none)
    {m k cl i : ℕ} (hres : Flock.RecResiduals.params sj.c.unit.name = some (m, k, cl, i)) :
    (∃ pt ∈ own, pt.name = "v") ∧ ∃ pt ∈ own, pt.name = "out" := by
  obtain ⟨inn, -, hck'⟩ := residualsCheck_of_check hck hso hop hsp ho hres
  obtain ⟨_, _, _, hcv, hz⟩ := residualsBoth_of_check hck'
  exact ⟨FlockC3.RegLevels.own_of_checkValue hcv, FlockC3.RegLevels.own_of_checkRows hz⟩

/-! ## A part's `out` gates on its `out` row -/

/-- **V*'s output layout at copy `j` of a part's unit** (`rec_residuals.lowering`, `out0 … out127`): one output port `out`
of 128 words, copy `j` returning words `0..127` in order, its one output row the first past the input rows, without
segments. A fact of V*'s circuit file (the parse, as `PartOutCols`), which check 5 does not read. -/
def PartOutsLaid (c : Flock.Circuit) (j : ℕ) : Prop :=
  c.outPorts = #[("out", 128)] ∧ c.leavesOut.getD j #[] = Array.range 128 ∧
    ∃ hq : Flock.HmOut.nIn c < c.ports.size, (c.ports[Flock.HmOut.nIn c]'hq).name = "out" ∧
      (c.ports[Flock.HmOut.nIn c]'hq).words = 128 ∧ (c.ports[Flock.HmOut.nIn c]'hq).segs = none

/-- **Bit `e` of a part's output is bit `e` of its `out` row** (`PartOutsLaid`): copy `j`'s output word `e / 16` is word
`e / 16` of output port 0. -/
theorem outBitRow_part {c : Flock.Circuit} {j : ℕ} (h : PartOutsLaid c j) {n : ℕ} (u : Fin n) {e : ℕ} (he : e < 2048) :
    ∃ hq : Flock.HmOut.nIn c < c.ports.size, ∃ hk : e < Layout.portRowW c ⟨_, hq⟩,
      HiddenExec.outBitRow c u (j, e / 16, e % 16) = some (u, ⟨⟨_, hq⟩, ⟨e, hk⟩⟩) := by
  obtain ⟨hop, hlo, hq, -, hw, -⟩ := h
  have hW : HiddenExec.outWordOf c (j, e / 16, e % 16) = e / 16 := by
    show (c.leavesOut.getD j #[])[e / 16]! = e / 16
    rw [hlo, getElem!_pos (Array.range 128) (e / 16) (by rw [Array.size_range]; omega)]
    simp
  have hs : c.outPorts.size = 1 := by rw [hop] <;> rfl
  have hN : c.outWords = 128 := by
    rw [← HiddenExec.outAt_size c, hs]
    unfold Hidden.outAt
    rw [hop] <;> rfl
  have hP : HiddenExec.outPortOf c (e / 16) = 0 := by
    have := (HiddenExec.outPortOf_spec c (o := e / 16) (by rw [hN]; omega)).1
    rw [hs] at this
    omega
  have hk : e < Layout.portRowW c ⟨_, hq⟩ := by
    show e < 16 * (c.ports[Flock.HmOut.nIn c]'hq).words
    rw [hw]
    omega
  refine ⟨hq, hk, FlockC3.RegLevels.outBitRow_eq u hq (by rw [hW, hP, Nat.add_zero]) ?_ hk⟩
  rw [hW, hP]
  show 16 * (e / 16 - Hidden.outAt c 0) + e % 16 = e
  rw [show Hidden.outAt c 0 = 0 by simp [Hidden.outAt]]
  omega

/-- **A part's `out` column on a registered zero bit is a registered 0** (`RegOut … false`): copy `j`'s column
`base + e`, its output `e / 16`'s bit `e % 16` (`hoc`, the parse's `PartOutCols`), is bit `e` of the `out` row
(`outBitRow_part`), so its gate's commit string is the `out` read (`rowOfH`, `FlockC3.RegLevels.regOut_of_outBitRow`),
whose registrant holds 0 at bit `e` (`hzero`). -/
theorem zeroOut_of_laid {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit}
    {hscope : Layout.scopeOk c = true} {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv) {u : Fin nv}
    {cpy : ℕ} (hcp : cpy < c.upv) (hlaid : PartOutsLaid c cpy) {base e : ℕ} (he : e < 2048)
    (hoc : base + e = Layout.outCol c (e / 16) (e % 16)) (h1 : (Rows.ofNet hU).nIn ≤ base + e)
    (h2 : base + e < (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp)
    (hread : ∀ hq : Flock.HmOut.nIn c < c.ports.size, FlockVBridge.Decode.regRead zo (.inl (u, ⟨_, hq⟩)))
    (hzero : ∀ (hq : Flock.HmOut.nIn c < c.ports.size) (hk : e < Layout.portRowW c ⟨_, hq⟩),
      (FlockVBridge.Decode.regVal zo (.inl (u, ⟨_, hq⟩))).2
        ⟨e, lt_of_lt_of_le hk ((Finset.le_sup (Finset.mem_univ (⟨_, hq⟩ : Fin c.ports.size))).trans (Nat.le_succ _))⟩ =
        false) :
    FlockVBridge.AccChain.RegOut zo u hcp (base + e) false := by
  obtain ⟨hq, hk, hrow⟩ := outBitRow_part hlaid u he
  have hj' : e / 16 < (c.leavesOut.getD cpy #[]).size := by
    rw [hlaid.2.1, Array.size_range]
    omega
  obtain ⟨hcol, hm, hr, hb⟩ := FlockC3.RegLevels.regOut_of_outBitRow zo hcp (j' := e / 16) (t' := e % 16) hj'
    (Nat.mod_lt _ (by decide)) (by rw [← hoc]; exact h1) (by rw [← hoc]; exact h2) hrow (hread hq)
  rw [hoc]
  exact ⟨hcol, hm, hr, hb.trans (hzero hq hk)⟩

/-- **A part's `out` column is a registered 0, or `ValueColl`** (`ZeroReads` at one column, the read position with its
bit): where check 5 accepted the `rec-residuals` unit of part `i` and V*'s part output layout holds at copy `cpy`,
`zero_tie` at unit `u`'s `out` read gives the registrant's bits there as 0, and `zeroOut_of_laid` the gate at column
`base + e` as that read's bit `e`. -/
theorem zero_regOut (hc : sj.c = (Vs j).c)
    (hck : Flock.RecCoins.check sj (some (zs R ω t R₂ j).own) (some f) (c3 t) = .ok ()) (hf : src t = some f)
    (hso : Flock.RecCoins.stackedParams (Vs j).c.unit.name = none)
    (hop : Flock.RecOpening.params (Vs j).c.unit.name = none) (hsp : Flock.RecSparse.params (Vs j).c.unit.name = none)
    (ho : Flock.RecOpen.params (Vs j).c.unit.name = none) {m k cl i : ℕ}
    (hres : Flock.RecResiduals.params (Vs j).c.unit.name = some (m, k, cl, i))
    (hprog : (Vs j).c.metaJ.getObjValAs? String "program_sha512" = .ok prog)
    {cpy : ℕ} (hcp : cpy < (Vs j).c.upv) (hlaid : PartOutsLaid (Vs j).c cpy)
    (hp : ∀ hq : Flock.HmOut.nIn (Vs j).c < (Vs j).c.ports.size, (zs R ω t R₂ j).rd.isRead (.inl (u, ⟨_, hq⟩)))
    (hone : ∀ hq : Flock.HmOut.nIn (Vs j).c < (Vs j).c.ports.size, ∀ reg ∈ (zs R ω t R₂ j).own, reg.name = "out" →
      reg = (zs R ω t R₂ j).rd.port (.inl (u, ⟨_, hq⟩)))
    (hpos : ∀ hq : Flock.HmOut.nIn (Vs j).c < (Vs j).c.ports.size,
      ((zs R ω t R₂ j).rd.port (.inl (u, ⟨_, hq⟩))).reads[u.val]? =
        some (some ((zs R ω t R₂ j).rd.readAt (.inl (u, ⟨_, hq⟩)))))
    {base e : ℕ} (he : e < 2048) (hoc : base + e = Layout.outCol (Vs j).c (e / 16) (e % 16))
    (h1 : (Rows.ofNet (Vs j).hU).nIn ≤ base + e)
    (h2 : base + e < (Rows.ofNet (Vs j).hU).nIn + (Rows.ofNet (Vs j).hU).nComp) :
    FlockVBridge.AccChain.RegOut (zs R ω t R₂ j) u hcp (base + e) false ∨
      ValueColl zs (valueSrcAll src c3) R ω t R₂ := by
  obtain ⟨hq, hnm, -, hseg⟩ := hlaid.2.2
  obtain ⟨-, hor⟩ := zero_tie (q := ⟨_, hq⟩) hc hck hf hso hop hsp ho hres
    (by simp only [getElem!_pos (Vs j).c.ports (Flock.HmOut.nIn (Vs j).c) hq, hnm]) hprog (hp hq)
    (fun reg hr hr' => hone hq reg hr (hr'.trans (by simp only [getElem!_pos (Vs j).c.ports (Flock.HmOut.nIn (Vs j).c) hq, hnm]))) (hpos hq)
    (by simp only [getElem!_pos (Vs j).c.ports (Flock.HmOut.nIn (Vs j).c) hq, hseg])
  exact hor.imp (fun hz => zeroOut_of_laid (zs R ω t R₂ j) hcp hlaid he hoc h1 h2 hp fun _ hk => hz e hk) id

end FlockSoundness.Discharge.Recursion
