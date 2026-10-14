import Proofs.Flock.VBridge.Family.Transcript

/-!
# `MsgsRead` at the committed rounds

`MsgsRead` asks that the table's round-`i` messages be a function of stage `i`'s masked commit-string wires, at every
round `i`. Two facts bound it:

* the messages read only the values w holds (`repMsgs_proofOf_w`): the proof file's messages (`repMsgs`) read the
  `decoded` keys of `symbolic`, none of which is a `comb`, `cons` or `rsT` key, so neither the comb nor the
  consistency sums enter a message; `final_c` (`fcOf`) reads the comb but is not one of `repMsgs`;
* the last round's messages are the openings (`tableTape`'s last step, `opensOf`), which V*'s units read from their
  private `row` and `aux` ports or from `rec-rows-L<level>` (`rec_outer.py:1-40`), never a round commitment
  (`SelRounds`: `rec-c<i>` or `rec-top-L<level>` at V*'s, `Flock/Registered.lean:182-193`). Two accepted outer proofs that
  differ only in the last inner coin agree on every committed row yet open different positions
  (`rec_vstar.positions`), so `MsgsRead` is false at `i = r − 1` for every `sel` with `SelRounds sel`.

`MsgsRead` also needs V*'s own staging `sel`: at a `sel` that selects nothing, which `SelRounds` admits, no wire is
masked and `MsgsRead` asks that all accepted outer proofs have the same messages (`msgsRead_bot`).

`MsgsReadC` is `MsgsRead` at the committed rounds `i + 1 < r`, and `decOf_specC` is `decOf_spec` there.
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Discharge.PlaysCoin FlockSoundness.Refine
open FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open Flock.RecAlgebra (decoded valOf symbolic)

/-! ## The messages read only w -/

section w

variable (sh : Shape) (w comb comb' : Array F128) (cons cons' : ℕ → ℕ → F128)

theorem decoded_ab : (decoded sh w comb cons).ab = (decoded sh w comb' cons').ab := by
  simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
    RecAlgebra.Lin.eval_var, RecAlgebra.KV.map]; try rfl
theorem decoded_c : (decoded sh w comb cons).c = (decoded sh w comb' cons').c := by
  simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
    RecAlgebra.Lin.eval_var, RecAlgebra.KV.map]; try rfl
theorem decoded_zc : (decoded sh w comb cons).zc = (decoded sh w comb' cons').zc := by
  simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
    RecAlgebra.Lin.eval_var, RecAlgebra.KV.map]; try rfl
theorem decoded_fa : (decoded sh w comb cons).fa = (decoded sh w comb' cons').fa := by
  simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
    RecAlgebra.Lin.eval_var, RecAlgebra.KV.map]; try rfl
theorem decoded_fb : (decoded sh w comb cons).fb = (decoded sh w comb' cons').fb := by
  simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
    RecAlgebra.Lin.eval_var, RecAlgebra.KV.map]; try rfl
theorem decoded_lc : (decoded sh w comb cons).lc = (decoded sh w comb' cons').lc := by
  simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
    RecAlgebra.Lin.eval_var, RecAlgebra.KV.map]; try rfl
theorem decoded_zp : (decoded sh w comb cons).zp = (decoded sh w comb' cons').zp := by
  simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
    RecAlgebra.Lin.eval_var, RecAlgebra.KV.map]; try rfl
theorem decoded_rs : (decoded sh w comb cons).rs = (decoded sh w comb' cons').rs := by
  simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
    RecAlgebra.Lin.eval_var, RecAlgebra.KV.map]; try rfl
theorem decoded_ood : (decoded sh w comb cons).ood = (decoded sh w comb' cons').ood := by
  simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
    RecAlgebra.Lin.eval_var, RecAlgebra.KV.map]; try rfl
theorem decoded_msgs : (decoded sh w comb cons).msgs = (decoded sh w comb' cons').msgs := by
  simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
    RecAlgebra.Lin.eval_var, RecAlgebra.KV.map]; try rfl
theorem decoded_yr : (decoded sh w comb cons).yr = (decoded sh w comb' cons').yr := by
  simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
    RecAlgebra.Lin.eval_var, RecAlgebra.KV.map]; try rfl

/-- **A rep's messages read only w**: the proof files (`proofOf`) of two reps that hold the same w have the same
messages (`repMsgs`), whatever their comb, consistency sums, coins and claims. -/
theorem repMsgs_proofOf_w (st : Setup) (sch : Flock.Schedule) (nc : ℕ) (params : PcsParams) (cap : Array ByteArray)
    (lig : LigeritoProof) {D : Type} (capD : Array ByteArray → Cap D) {d d' : RepData} (hw : d.w = d'.w) :
    repMsgs st sch nc (proofOf sh params cap lig d) capD = repMsgs st sch nc (proofOf sh params cap lig d') capD := by
  obtain ⟨w, comb, cons, rounds, extras⟩ := d
  obtain ⟨w', comb', cons', rounds', extras'⟩ := d'
  obtain rfl : w = w' := hw
  simp only [repMsgs, zcMsgs, proofOf, decoded_ab sh w comb comb' cons cons', decoded_c sh w comb comb' cons cons',
    decoded_zc sh w comb comb' cons cons', decoded_fa sh w comb comb' cons cons', decoded_fb sh w comb comb' cons cons',
    decoded_lc sh w comb comb' cons cons', decoded_zp sh w comb comb' cons cons', decoded_rs sh w comb comb' cons cons',
    decoded_ood sh w comb comb' cons cons', decoded_msgs sh w comb comb' cons cons',
    decoded_yr sh w comb comb' cons cons']

end w

/-! ## A selection that selects nothing -/

theorem selRounds_bot : VStarFamily.SelRounds fun _ _ _ => False := fun _ _ _ h => h.elim

/-- **No wire is masked at a selection that selects nothing**: stage `i`'s mask is the same at every layer. -/
theorem maskOf_skOf_bot {S : ℕ} (Vs : Fin S → VStmt) (own : Fin S → Array Flock.Registered.Port) (i : ℕ)
    (Xs Ys : (j : Fin S) → (Vs j).Bits) :
    maskOf Vs (skOf Vs own fun _ _ _ => False) i Xs = maskOf Vs (skOf Vs own fun _ _ _ => False) i Ys := by
  funext j g
  have hn : posK (Vs j).c (Vs j).hU (Vs j).nv g ∉ skOf Vs own (fun _ _ _ => False) i j := by
    generalize posK (Vs j).c (Vs j).hU (Vs j).nv g = x
    rcases x with ⟨u, q⟩ | x
    · rw [mem_skOf_inl]
      rintro ⟨_, _, _, _, _, h⟩
      exact h
    · exact inr_not_mem_skOf x
  simp [maskOf, hn]

/-! ## At the family -/

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
variable (spec : SessionSpec) (st : Reg → L.Ω → Setup)
    (cap : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray)
    (lig : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof)
variable (own : Fin F.stmts.size → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)

/-- Obligation: discharged from the stage files at V*'s staging `sel` (`rec_vstage.messages`: algebra round `k` of
rep `r` registered under `rec-c<k>` at row `r`, `rec_vstage.py:140-168`; the level-0 cap under `rec-top-L0`, every
cap node opened once by the first stratum, `rec_vstar.positions`), with each round's w slots at the rows its
`sel i` names (`locOf`) and the inner rounds aligned with V*'s (`tableTape`'s weave).
**`MsgsRead` at the committed rounds**: two accepted outer proofs whose layers agree on the read wires of stage `i`'s
commit strings have the same round-`i` table messages, at every round `i` but the last. The last round is the
openings, which no round commitment holds. -/
def MsgsReadC : Prop :=
  ∀ i, i + 1 < (ℐ).r → ∀ R ω t R₂ τ o R' ω' t' R₂' τ' o', Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    Accepted (I := ℐ) F.zs dg N k Rw R' ω' t' R₂' τ' o' →
    maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R ω t R₂ τ) =
      maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R' ω' t' R₂' τ') →
    ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t (layers F.zs dg N k Rw R ω t R₂ τ))[i]? =
      ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R' ω' t' (layers F.zs dg N k Rw R' ω' t' R₂' τ'))[i]?

/-- **`MsgsRead` at a selection that selects nothing** (`selRounds_bot`) asks that every two accepted outer proofs
have the same table messages at every round: V*'s inner sessions, whose statements differ, are not such a family, so
`MsgsRead` is an obligation at V*'s staging `sel`, not at every `sel` with `SelRounds sel`. -/
theorem msgsRead_bot (hread : F.MsgsRead dg N k Rw spec st cap lig own fun _ _ _ => False) (i : ℕ)
    {R ω t R₂ τ o R' ω' t' R₂' τ' o'} (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o)
    (h' : Accepted (I := ℐ) F.zs dg N k Rw R' ω' t' R₂' τ' o') :
    ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t (layers F.zs dg N k Rw R ω t R₂ τ))[i]? =
      ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R' ω' t' (layers F.zs dg N k Rw R' ω' t' R₂' τ'))[i]? :=
  hread i R ω t R₂ τ o R' ω' t' R₂' τ' o' h h' (maskOf_skOf_bot F.Vs own i _ _)

theorem msgsReadC_of (hread : F.MsgsRead dg N k Rw spec st cap lig own sel) :
    F.MsgsReadC dg N k Rw spec st cap lig own sel :=
  fun i _ => hread i

/-- **`decOf_spec` at the committed rounds** (`MsgsReadC`, `MsgTys`). -/
theorem decOf_specC (hread : F.MsgsReadC dg N k Rw spec st cap lig own sel)
    (htys : F.MsgTys dg N k Rw spec st cap lig)
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) {i : ℕ} (hi : i + 1 < (ℐ).r) :
    roundMsgs T (F.decOf dg N k Rw spec st cap lig own sel i
      (maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R ω t R₂ τ))) =
    ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t
      (layers F.zs dg N k Rw R ω t R₂ τ))[i]?.getD [] := by
  have hex : ∃ g : List Msg, ∃ R' ω' t' R₂' τ' o', Accepted (I := ℐ) F.zs dg N k Rw R' ω' t' R₂' τ' o' ∧
      maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R' ω' t' R₂' τ') =
        maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R ω t R₂ τ) ∧
      ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R' ω' t'
        (layers F.zs dg N k Rw R' ω' t' R₂' τ'))[i]?.getD [] = g := ⟨_, R, ω, t, R₂, τ, o, h, rfl, rfl⟩
  have hd : F.decOf dg N k Rw spec st cap lig own sel i
      (maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R ω t R₂ τ)) = toSs T (Classical.choose hex) := by
    unfold decOf
    exact dite_eq_left_of_eq_true (eq_true hex)
  rw [hd]
  obtain ⟨R', ω', t', R₂', τ', o', h', hX', hg'⟩ := Classical.choose_spec hex
  have e := hread i hi R ω t R₂ τ o R' ω' t' R₂' τ' o' h h' hX'.symm
  have hc : Classical.choose hex = ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t
      (layers F.zs dg N k Rw R ω t R₂ τ))[i]?.getD [] := by
    rw [← hg', ← e]
  rw [hc]
  apply roundMsgs_toSs
  intro m hm
  cases hti : ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t
      (layers F.zs dg N k Rw R ω t R₂ τ))[i]? with
  | none => rw [hti] at hm; simp at hm
  | some g =>
      rw [hti, Option.getD_some] at hm
      exact htys R ω t R₂ τ o h g (List.mem_of_getElem? hti) m hm

end VStarFamily

end family

end FlockVBridge.Family
