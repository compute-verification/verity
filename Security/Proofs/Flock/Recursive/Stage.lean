import Proofs.Flock.Recursive.Transfer
import Proofs.Flock.Recursive.Flock
import Proofs.Flock.Soundness.Discharge.Exec.LawDraw

/-!
# Recursion: V*'s stages, with one set of openings at every history

`zk_sessions_recursive_upto` takes one `Openings` for every history and stage, while the value binding of a session of
V* at a history (`ZkOuter.vb`) is built from that history's inputs. What the openings read is fixed by the session's
circuit `c`: the commit strings (`PosK`: unit `u`'s row of port `q`, input or output, or a single gate), their values
(`ValK`: bit strings up to the widest row) and each one's leaf scheme (`schemeK`: `hm96X` of the row's port's bytes,
`HiddenExec.encKeyH`, at a row, `SHA-512(bitCode v)` at a single gate). `obK` is those openings, and the value
binding's openings at each history are `obK` by definition (`ZkOuter.openingsOf_vb`); what a history's inputs change is
only the registered digests (`ValueBindingZ.cs`, from the public file). V*'s sessions together open `obS`: session
`j`'s `obK`, tagged by `j` (`Openings.sigma`).

* `candWeightK`: `candWeightZC` read off a session's law alone (`ZkOuter.candWeight_vb`);
* `ZkOuter.candTransfer`: `zk_session_candValsR` with `candTermZC_le` at one session, the trial for each commit string a
  finder of `obK`'s openings (`ZkOuter.trial`), and the slack `ZkOuter.stageSlack` (`slackIf` where the session's
  knowledge finders satisfy `cr/sha-512`, else `1`);
* `execOS_subset_full_mem`, `vCovers_full`: V*'s live law, the full draw, draws every unit, so every unit's wires are
  covered;
* `sessions_stage_le`: at V*'s sessions in a row, a layer that disagrees on a wire of a session's commit strings with
  the values `wv` is at most the sessions' expected slack plus each commit string's weight times its trial's chance of
  opening the session's registered digest there off `wv`; `outerV_stage_le` is the same at the registration `R₂` the
  prover sends (`outerV`), and `trialV` the trial there;
* `VCommits`, `VDecodes`, `VFull`: what V*'s inputs, circuits and laws must do at each stage: session `j`'s public file
  at history `t` registers `t`'s stage-`i` commitment at the stage's commit strings `Sk i j`, at every registration
  `R₂`, the layers' round-`i` message is read from the wires of those commit strings, and every such wire is a wire of
  some unit;
* `ReadW`, `VDecodesR`, `VFullR`: a wire some unit reads, and `VDecodes` and `VFull` at those wires alone
  (`Proofs.Flock.Recursive.StageRead`);
* `zk_sessions_recursive_full`: `zk_sessions_recursive_upto` at `obS`, its transfer from `sessions_stage_le`;
* `zk_sessions_recursive_fork`: the same in fork form, a round's fork finder finding a collision (`RoundFork`), a
  session's (`SessionFork`), or the bound with no finder budgets: the sessions' `ZkOuter.boundCR`, `r` times their
  `ZkOuter.slackCR` (`ZkOuter.slackIf_le`) and the inner error.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg
open ZkHidden (planZHJ tabZHJ)
open scoped ENNReal

section Fixed

variable (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ)

/-- **V*'s commit strings**: unit `u`'s row of port `q`, or a single gate (`Readers.ofRows`' positions at
`HiddenExec.keyReadersH`). -/
abbrev PosK : Type := (Fin n × Fin c.ports.size) ⊕ Fin ((keyProg c hU n).circuit (Layout.outsOf c hU n)).N

/-- **Their values**: bit strings of at most the widest row's `16 · words` bits, with their width. -/
abbrev ValK : Type := Layout.BitVal (Finset.univ.sup (Layout.portRowW c) + 1)

/-- **Each commit string's leaf scheme**: `hm96X` of the bytes its port's hash reads (`HiddenExec.encKeyH`:
`sha512/row/v1`'s, `/row/v2`'s or `/row-seg/v1`'s, `HiddenExec.encPortH`) at a row, `SHA-512(bitCode v)` at a single
gate. -/
noncomputable def schemeK (p : PosK c hU n) : Binding.ExplicitLeaf (ValK c) :=
  if p.isLeft = true then
    Binding.ExplicitLeaf.hm96X (HiddenExec.encKeyH c p) (HiddenExec.extKeyH c p) (HiddenExec.extKeyH_spec c p)
  else Binding.ExplicitLeaf.plain Layout.bitCode Layout.bitCode_inj

/-- **V*'s openings**: a commit string, a value and a salt, opening the commit string's leaf of the value under the
salt; two openings of one commit string's leaf to different values are a SHA-512 collision (`ExplicitLeaf.pair`). -/
noncomputable def obK : Openings H512 (ValK c) where
  Op := PosK c hU n × ValK c × List UInt8
  Dg := PosK c hU n × Dig
  com o := (o.1, (schemeK c hU n o.1).leaf o.2.1 o.2.2)
  val o := o.2.1
  collide o o' := (schemeK c hU n o.1).pair o.2.1 o.2.2 o'.2.1 o'.2.2
  collide_spec o o' hc hv := by
    obtain ⟨p, v, y⟩ := o
    obtain ⟨p', v', y'⟩ := o'
    simp only [Prod.mk.injEq] at hc
    obtain ⟨rfl, hl⟩ := hc
    exact (schemeK c hU n p).pair_spec v y v' y' hv hl

/-- The commit string holding wire `g` (`Readers.ofRows`' `pos`): its row when `HiddenExec.rowOfH` has one (an input
row's bit, or a returned output's bit of its output row), else its own. -/
noncomputable def posK (g : Fin ((keyProg c hU n).circuit (Layout.outsOf c hU n)).N) : PosK c hU n :=
  match HiddenExec.rowOfH c hU n (Layout.outsOf c hU n) g with
  | some (u, ⟨q, _⟩) => .inl (u, q)
  | none => .inr g

/-- Wire `g`'s place in its commit string's value (`Readers.ofRows`' `idx`): bit `k` of its row, or a single gate's
only bit. -/
noncomputable def idxK (g : Fin ((keyProg c hU n).circuit (Layout.outsOf c hU n)).N) :
    Fin (Finset.univ.sup (Layout.portRowW c) + 1) :=
  match HiddenExec.rowOfH c hU n (Layout.outsOf c hU n) g with
  | some (_, ⟨q, k⟩) => Fin.castLE ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)) k
  | none => ⟨0, Nat.succ_pos _⟩

/-- Wire `g`'s bit in the value `v` of its commit string. -/
noncomputable def bitK (v : ValK c) (g : Fin ((keyProg c hU n).circuit (Layout.outsOf c hU n)).N) : Bool :=
  v.2 (idxK c hU n g)

/-- A draw of `Lv` reads commit string `pos`: a drawn unit reads a wire it holds. -/
def ReadsDrawK (Lv : Law n) (pos : PosK c hU n) (ω : Lv.Ω) : Prop :=
  ∃ u ∈ Lv.draw ω, ∃ g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u, posK c hU n g = pos

/-- **Every draw of `Lv` reads every wire of the commit strings `Sk`**: each is a wire of a drawn unit. -/
def VCovers (Lv : Law n) (Sk : Finset (PosK c hU n)) : Prop :=
  ∀ (ω : Lv.Ω) g, posK c hU n g ∈ Sk → ∃ u ∈ Lv.draw ω, g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u

open Classical in
/-- **Commit string `pos`'s weight** at the stage's commit strings `Sk`: on `Sk`, the fraction of `Lv`'s draws reading
it over `1 − ρ`. -/
noncomputable def candWeightK (Lv : Law n) (Sk : Finset (PosK c hU n)) (ρ : ℝ) (pos : PosK c hU n) : ℝ≥0∞ :=
  if pos ∈ Sk then
    ENNReal.ofReal ((Fintype.card {ω // ReadsDrawK c hU n Lv pos ω} : ℝ) / Fintype.card Lv.Ω / (1 - ρ))
  else 0

end Fixed

/-- **An event of the live audit's outcomes at the law's draws**: implications that hold at every outcome
`(R, L.draw ω, t)` bound probabilities. -/
theorem prob_auditLive_mono {n : ℕ} {L : Law n} {Reg : Type} (live : Finset (Fin n) → Reg → Game Exec.Transcript)
    (τ : Strategy (auditLive L Reg live)) {p q : Reg × Finset (Fin n) × Exec.Transcript → Prop}
    (h : ∀ R ω t, p (R, L.draw ω, t) → q (R, L.draw ω, t)) :
    prob p (auditLive L Reg live) τ ≤ prob q (auditLive L Reg live) τ := by
  show avg (fun ω => prob p ((live (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω)) ≤
    avg fun ω => prob q ((live (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω)
  refine avg_mono fun ω => ?_
  rw [prob_map, prob_map]
  exact prob_mono (fun t => h τ.1 ω t) _ _

/-- **A live subset draw that succeeds is `Flock.Draw.subset` on the source's one stream.** -/
theorem runOS_subset {L : Fin 1 → ℕ} (f : (i : Fin 1) → Fin (L i) → Fin 256) (k n : ℕ) {d : Flock.Draw.UnitDraw}
    (hd : Law.runOS f (.subset k) n = some d) : Flock.Draw.subset k n (Law.streamOf (f 0)) = some d.units := by
  unfold Law.runOS at hd
  rw [Law.drawWith_subset_run] at hd
  cases ho : (Flock.Draw.onWith (Law.tapeGet (Law.tapesOf f) 0) (2 * k * Flock.Draw.width n + 64)
      (Flock.Draw.subset k n)).run (fun _ => 0) with
  | none => rw [ho] at hd; exact absurd hd (by simp)
  | some p =>
    obtain ⟨u, pos'⟩ := p
    rw [ho] at hd
    cases u with
    | none => exact absurd hd (by simp)
    | some D =>
      have hD : D = d.units := by rw [← Option.some.inj hd]
      have h1 := (Law.onWith_tape (Law.tapesOf f) 0 _ (by omega) _ (fun a b u h => Law.subset_append _ _ a b u h) _
        pos' _ rfl ho).2 D rfl
      have h2 : Law.tapesOf f 0 = Law.streamOf (f 0) := by simp [Law.tapesOf, Array.getD_eq_getD_getElem?]
      rw [h2] at h1
      rw [← hD]; exact h1

/-- **The full draw draws every unit**: a run of `subset n` over `[0, n)` that succeeds is a permutation's values on
all of `[0, n)`, and a run that reads past the source's bytes draws every unit. -/
theorem execOS_subset_full_mem {n : ℕ} {Ω : Type} [Fintype Ω] [Nonempty Ω] {L : Fin 1 → ℕ}
    (t : Ω → (i : Fin 1) → Fin (L i) → Fin 256) (ω : Ω) (u : Fin n) :
    u ∈ (Law.execOS (n := n) t (.subset n) n).draw ω := by
  show u ∈ (match Law.runOS (t ω) (.subset n) n with
    | some d => Finset.univ.filter fun u : Fin n => u.val ∈ d.units
    | none => Finset.univ)
  cases h : Law.runOS (t ω) (.subset n) n with
  | none => exact Finset.mem_univ _
  | some d =>
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    obtain ⟨π, -, hπ, hmem⟩ := FlockSoundness.Discharge.Exec.subset_spec (runOS_subset (t ω) n n h)
    have himg : (Finset.range n).image π = Finset.range n :=
      Finset.eq_of_subset_of_card_le
        (fun x hx => by
          obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hx
          exact Finset.mem_range.2 (hπ i (Finset.mem_range.1 hi)))
        (Finset.card_image_of_injective _ π.injective).ge
    obtain ⟨i, hi, hiu⟩ :=
      Finset.mem_image.1 (by rw [himg]; exact Finset.mem_range.2 u.isLt : u.val ∈ (Finset.range n).image π)
    exact (hmem u.val).2 ⟨i, Finset.mem_range.1 hi, hiu⟩

namespace ZkOuter

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv) (dg : ByteArray → ByteArray) (N : ℕ)

/-- **The value binding's openings are `obK` at every history.** -/
theorem openingsOf_vb :
    openingsOf execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ zo.dj zo.y₀)
      (tabZHJ zo.dj zo.y₀) zo.vb = obK c hU nv := rfl

/-- **The trial for commit string `pos`** of V*'s session at the strategy `τ`, as a finder of `obK`'s openings. -/
noncomputable def trial (k Rw : ℕ) (τ : Strategy (zo.game dg N)) (pos : PosK c hU nv) :
    Finder (Finset (obK c hU nv).Op) :=
  trialFinderZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ zo.dj zo.y₀)
    (tabZHJ zo.dj zo.y₀) zo.vb (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) k Rw pos

/-- **A stage's slack at the strategy `τ`**: `ksAvgStrictZ + linkBoundZC + candSlackZC` of V*'s session. -/
noncomputable def stageSlack (k Rw M : ℕ) (ρ t' qF qS : ℝ) (τ : Strategy (zo.game dg N)) : ℝ≥0∞ :=
  ksAvgStrictZ (planZHJ zo.dj zo.y₀) Lv k qF qS (reg (zo.sim dg N τ)) +
    linkBoundZC execArith hs512 ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ zo.dj zo.y₀)
      (tabZHJ zo.dj zo.y₀) zo.vb k Rw M ρ t' (reg (zo.sim dg N τ)) (cont (zo.sim dg N τ)) +
    candSlackZC (L := Lv) execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ zo.dj zo.y₀)
      (tabZHJ zo.dj zo.y₀) zo.vb k Rw ρ

open Classical in
/-- **A stage's slack where the strategy's knowledge finders satisfy `cr/sha-512`** (`TabCR`), else `1`. -/
noncomputable def slackIf (k Rw M : ℕ) (ρ t' qF qS : ℝ) (τ : Strategy (zo.game dg N)) : ℝ≥0∞ :=
  if zo.TabCR dg N qF qS τ then zo.stageSlack dg N k Rw M ρ t' qF qS τ else 1

/-- **A stage's slack in fork form**: the drawn units' statistical errors, the tables' (`tableErrZ`), the link's
(`linkErrZC`) and the waits' (`waitErrZC`); `stageSlack` with no finder budgets and no SHA-512 evaluations to count. -/
noncomputable def slackCR (k Rw M : ℕ) (ρ : ℝ) (τ : Strategy (zo.game dg N)) : ℝ≥0∞ :=
  tableErrZ (planZHJ zo.dj zo.y₀) Lv k (reg (zo.sim dg N τ)) +
    linkErrZC execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ (Reg := Reg') zo.dj zo.y₀)
      (tabZHJ zo.dj zo.y₀) zo.vb Lv k Rw M ρ +
    waitErrZC execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ (Reg := Reg') zo.dj zo.y₀)
      (tabZHJ zo.dj zo.y₀) zo.vb Lv k Rw ρ

/-- **With no fork, `slackIf` at no budgets is the fork-form slack.** -/
theorem slackIf_le {k Rw M : ℕ} (ρ : ℝ) {τ : Strategy (zo.game dg N)} (h : ¬ zo.Fork dg N k Rw M τ) :
    zo.slackIf dg N k Rw M ρ 0 0 0 τ ≤ zo.slackCR dg N k Rw M ρ τ := by
  unfold slackIf
  split_ifs with hc
  · refine le_of_eq ?_
    unfold stageSlack slackCR
    rw [ksAvgStrictZ_zero, linkBoundZC_zero _ _ _ _ _ _ _ _ (zo.cr_of_not_fork dg N h).2, candSlackZC_eq]
  · exact absurd (zo.cr_of_not_fork dg N h).1.1 hc

/-- The value binding's commit string of each wire is `posK`'s. -/
theorem vb_pos (g : Fin ((keyProg c hU nv).circuit (Layout.outsOf c hU nv)).N) : zo.vb.pos g = posK c hU nv g := by
  dsimp only [vb, vbHJ, rsZHJS, HmRowsZX.bindingZ, LayoutZ.bindingZ, ZkHidden.layZHJ, LayoutZ.ofShadow,
    ZkHidden.layShHJ, HiddenExec.keyLayoutH, Layout.Readers.layout, Layout.ofBits, HiddenExec.keyReadersH]
  unfold posK
  rcases h : HiddenExec.rowOfH c hU nv (Layout.outsOf c hU nv) g with _ | ⟨u, q, k⟩
  · exact Layout.ofRows_pos_none _ h
  · exact Layout.ofRows_pos_some _ h

/-- The value binding's bit of each wire is `bitK`'s. -/
theorem vb_bit (v : ValK c) (g : Fin ((keyProg c hU nv).circuit (Layout.outsOf c hU nv)).N) :
    zo.vb.bit v g = bitK c hU nv v g := by
  dsimp only [vb, vbHJ, rsZHJS, HmRowsZX.bindingZ, LayoutZ.bindingZ, ZkHidden.layZHJ, LayoutZ.ofShadow,
    ZkHidden.layShHJ, HiddenExec.keyLayoutH, Layout.Readers.layout, Layout.ofBits, HiddenExec.keyReadersH,
    Layout.Readers.ofRows]
  unfold bitK idxK
  rcases HiddenExec.rowOfH c hU nv (Layout.outsOf c hU nv) g with _ | ⟨u, q, k⟩ <;> rfl

open Classical in
/-- **`candWeightZC` of V*'s session reads only V*'s law**: it is `candWeightK`. -/
theorem candWeight_vb (Sk : Finset (PosK c hU nv)) (ρ : ℝ) (pos : PosK c hU nv) :
    candWeightZC (L := Lv) execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ zo.dj zo.y₀)
      (tabZHJ zo.dj zo.y₀) zo.vb Sk ρ pos = candWeightK c hU nv Lv Sk ρ pos := by
  unfold candWeightZC candWeightK
  split_ifs with h1 h2 h2
  · exact congrArg (fun x : ℕ => ENNReal.ofReal ((x : ℝ) / Fintype.card Lv.Ω / (1 - ρ)))
      (Fintype.card_congr (Equiv.subtypeEquivRight fun ω => by
        unfold ReadsDrawZ ReadsZ ReadsDrawK
        simp only [vb_pos]
        exact Iff.rfl))
  · exact absurd h1 h2
  · exact absurd h2 h1
  · rfl

/-- **A stage's transfer at one session**, in `htr`'s shape with `obK`: `Pr[accept ∧ a drawn unit reads a wire of Sk
whose layer bit is not wv's] ≤ stageSlack + Σ_pos candWeightK pos · Pr[the trial for pos opens pos's registered digest
to a value other than wv pos]`; `zk_session_candValsR` with `candTermZC_le`. -/
theorem candTransfer {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg') zo.dj zo.y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (τ : Strategy (zo.game dg N)) {qF qS : ℝ} (hCR : zo.TabCR dg N qF qS τ)
    (Sk : Finset (PosK c hU nv)) (wv : PosK c hU nv → ValK c) :
    prob (fun o => LiveAcceptsZKR zo.I zo.own o ∧
        ∃ u ∈ o.2.1, ∃ g ∈ ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u,
          CandBadZ execArith ((keyProg c hU nv).partition (Layout.outsOf c hU nv)) (planZHJ zo.dj zo.y₀)
            (tabZHJ zo.dj zo.y₀) zo.vb Sk wv (zo.layer dg N k Rw τ) g) (zo.game dg N) τ ≤
      zo.stageSlack dg N k Rw M ρ t' qF qS τ +
        ∑ pos, candWeightK c hU nv Lv Sk ρ pos *
          prob (OffAt (obK c hU nv) (zo.vb.cs (reg (zo.sim dg N τ)) pos) (wv pos)) (zo.trial dg N k Rw τ pos).g
            (zo.trial dg N k Rw τ pos).s := by
  refine ((zk_session_candValsR zo.I zo.ht zo.own zo.dj zo.y₀ zo.wr dg N zo.hcH hU zo.hpub
    (Layout.scopeOkH_of_scopeOk hscope) Lv
    zo.hDraw zo.hTags zo.hRec hk hRw hM hρ hr ht' Sk wv τ hCR).trans (add_le_add le_rfl
      (candTermZC_le execArith hs512 _ _ _ zo.vb Sk wv _ _ k Rw hρ))).trans_eq ?_
  rw [← add_assoc]
  refine congrArg _ (Finset.sum_congr rfl fun pos _ => ?_)
  rw [candWeight_vb zo Sk ρ pos]
  rfl

end ZkOuter

section Covers

variable {nv : ℕ} {c : Flock.Circuit} {hU : Built c.unit}

/-- **Coverage by the full draw**: when every wire of `Sk` is a wire of a unit, every draw of the live full draw reads
every wire of `Sk` (`execOS_subset_full_mem`). -/
theorem vCovers_full {Ω : Type} [Fintype Ω] [Nonempty Ω] {Lb : Fin 1 → ℕ}
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (Sk : Finset (PosK c hU nv))
    (hfull : ∀ g, posK c hU nv g ∈ Sk → ∃ u, g ∈ ((keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u) :
    VCovers c hU nv (Law.execOS (n := nv) src (.subset nv) nv) Sk := by
  intro ω g hg
  obtain ⟨u, hgu⟩ := hfull g hg
  exact ⟨u, execOS_subset_full_mem src ω u, hgu⟩

end Covers

section Sigma

variable {S : ℕ}

/-- The collision two tagged openings output: their session's, when they are of one session. -/
noncomputable def sigmaCollide {Vt : Fin S → Type} (ob : (j : Fin S) → Openings H512 (Vt j)) :
    (Σ j, (ob j).Op) → (Σ j, (ob j).Op) → List UInt8 × List UInt8
  | ⟨j, a⟩, ⟨j', a'⟩ =>
    if h : j = j' then (ob j').collide (cast (congrArg (fun i => (ob i).Op) h) a) a' else ([], [])

theorem sigmaCollide_self {Vt : Fin S → Type} (ob : (j : Fin S) → Openings H512 (Vt j)) (j : Fin S)
    (a a' : (ob j).Op) : sigmaCollide ob ⟨j, a⟩ ⟨j, a'⟩ = (ob j).collide a a' :=
  dite_eq_left rfl

/-- **The openings of a family of sessions**: an opening of session `j`'s openings, tagged by `j`, opening the tagged
digest to the tagged value; two openings of one tagged digest are of one session, so to different values they are that
session's collision. -/
noncomputable def Openings.sigma {Vt : Fin S → Type} (ob : (j : Fin S) → Openings H512 (Vt j)) :
    Openings H512 (Option (Σ j, Vt j)) where
  Op := Σ j, (ob j).Op
  Dg := Σ j, (ob j).Dg
  com o := ⟨o.1, (ob o.1).com o.2⟩
  val o := some ⟨o.1, (ob o.1).val o.2⟩
  collide := sigmaCollide ob
  collide_spec := by
    rintro ⟨j, a⟩ ⟨j', a'⟩ hc hv
    dsimp only at hc hv
    obtain ⟨rfl, hc⟩ := Sigma.mk.inj hc
    rw [sigmaCollide_self]
    exact (ob _).collide_spec a a' (eq_of_heq hc) fun h => hv (by rw [h])

/-- Session `j`'s value in a tagged value: the value, when tagged `j`. -/
noncomputable def projV {Vt : Fin S → Type} [∀ j, Nonempty (Vt j)] (j : Fin S) : Option (Σ j, Vt j) → Vt j
  | some ⟨j', w⟩ => if h : j' = j then cast (congrArg Vt h) w else Classical.arbitrary _
  | none => Classical.arbitrary _

theorem projV_mk {Vt : Fin S → Type} [∀ j, Nonempty (Vt j)] (j : Fin S) (w : Vt j) : projV j (some ⟨j, w⟩) = w :=
  dite_eq_left rfl

end Sigma

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type}

/-- **V*'s openings**: session `j`'s `obK`, tagged by `j`. -/
noncomputable abbrev obS (Vs : Fin S → VStmt) :=
  Openings.sigma fun j => obK (Vs j).c (Vs j).hU (Vs j).nv

/-- **V*'s commit strings' digests** of the commitment `e`: session `j`'s `cmt j e`, tagged by `j`. -/
def cmtS {Cm : Type} (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (e : Cm) (p : Σ j, PosK (Vs j).c (Vs j).hU (Vs j).nv) : (obS Vs).Dg :=
  ⟨p.1, cmt p.1 e p.2⟩

/-- **The message tagged values open to**: `msg` at each session's values (`projV`). -/
noncomputable def msgS {Msg : Type}
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → Msg) (i : ℕ)
    (v : (Σ j, PosK (Vs j).c (Vs j).hU (Vs j).nv) → Option (Σ j, ValK (Vs j).c)) : Msg :=
  msg i fun j p => projV j (v ⟨j, p⟩)

/-- **The trial for a tagged commit string** `⟨j, p⟩` at V*'s sessions: play the sessions, then session `j`'s trial
for `p` at the strategy the prover named there, its openings tagged by `j`. -/
noncomputable def trialS (zs : (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (s : Strategy (sessions zs dg N)) (pos : Σ j, PosK (Vs j).c (Vs j).hU (Vs j).nv) :
    Finder (Finset (obS Vs).Op) :=
  ⟨(sessions zs dg N).bind fun rs =>
      ((zs pos.1).trial dg N k Rw ((zs pos.1).ofBase dg N (rs pos.1).1) pos.2).g.map
        (Finset.map (Function.Embedding.sigmaMk (β := fun j => (obK (Vs j).c (Vs j).hU (Vs j).nv).Op) pos.1)),
    Strategy.bind _ _ s fun rs =>
      Strategy.map _ _ ((zs pos.1).trial dg N k Rw ((zs pos.1).ofBase dg N (rs pos.1).1) pos.2).s⟩

/-- **A tagged commit string's weight** at stage `i`: its session's `candWeightK` on `Sk i j`. -/
noncomputable def weightS (Vs : Fin S → VStmt)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)) (ρ : ℝ) (i : ℕ)
    (pos : Σ j, PosK (Vs j).c (Vs j).hU (Vs j).nv) : ℝ≥0∞ :=
  candWeightK (Vs pos.1).c (Vs pos.1).hU (Vs pos.1).nv (Vs pos.1).law.live (Sk i pos.1) ρ pos.2

/-- **A session's trial, played inside the sessions, is a tagged trial**: an opening of session `j`'s digest `d` off
session `j`'s value of `v` is an opening of the tagged digest off `v`. -/
theorem trialS_le (zs : (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (s : Strategy (sessions zs dg N)) (j : Fin S) (p : PosK (Vs j).c (Vs j).hU (Vs j).nv)
    (d : (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg) (v : Option (Σ j, ValK (Vs j).c)) :
    exG (fun rs : VRec Vs Reg' dg N => prob (OffAt (obK (Vs j).c (Vs j).hU (Vs j).nv) d (projV j v))
        ((zs j).trial dg N k Rw ((zs j).ofBase dg N (rs j).1) p).g
        ((zs j).trial dg N k Rw ((zs j).ofBase dg N (rs j).1) p).s) (sessions zs dg N) s ≤
      prob (OffAt (obS Vs) ⟨j, d⟩ v) (trialS zs dg N k Rw s ⟨j, p⟩).g (trialS zs dg N k Rw s ⟨j, p⟩).s := by
  dsimp only [trialS]
  rw [prob_bind_strat]
  refine exG_mono (fun rs => ?_) _ _
  rw [prob_map_strat]
  refine prob_mono (fun os ho => ?_) _ _
  obtain ⟨o, hmem, hc, hv⟩ := ho
  exact ⟨⟨j, o⟩, Finset.mem_map_of_mem _ hmem,
    congrArg (Sigma.mk (β := fun j => (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg) j) hc,
    fun h => hv (by
      have h' := congrArg (projV j) h
      rwa [show (obS Vs).val ⟨j, o⟩ = some ⟨j, (obK (Vs j).c (Vs j).hU (Vs j).nv).val o⟩ from rfl, projV_mk] at h')⟩

/-- **A stage's transfer at V*'s sessions**: when every layer agreeing with the values `wv` on the wires of the commit
strings `Sk j` satisfies `Q`, the sessions accept with `Q` failing at most the sessions' expected `slackIf` plus, at
each session's commit string, its weight times the chance its trial opens the session's registered digest `cs j pos`
off `wv j pos` (`ZkOuter.candTransfer` at each session, `prob_seqC_exists_le`). Every wire of `Sk j` is a unit's, so it
is read at every draw (`hfull`, `vCovers_full`). -/
theorem sessions_stage_le
    (zs : (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ j S' R' i, rateZ (planZHJ (Reg := Reg') (zs j).dj (zs j).y₀ S' R') i k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (qF qS : ℝ) (Sk : (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (hfull : ∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk j →
      ∃ u, g ∈ ((keyProg (Vs j).c (Vs j).hU (Vs j).nv).partition (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).io u)
    (cs : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (hcs : ∀ j, (∃ tr, AcceptsZKR (zs j).I (zs j).own tr) → ∀ pos ∈ Sk j, ∀ R' : Reg', (zs j).vb.cs R' pos = cs j pos)
    (Q : ((j : Fin S) → (Vs j).Bits) → Prop)
    (wv : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c)
    (hQ : ∀ Xt : (j : Fin S) → (Vs j).Bits, (∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk j →
      Xt j g = bitK (Vs j).c (Vs j).hU (Vs j).nv (wv j (posK (Vs j).c (Vs j).hU (Vs j).nv g)) g) → Q Xt)
    (s : Strategy (sessions zs dg N)) :
    prob (fun b => Accepts zs b ∧ ¬ Q (layers zs dg N k Rw b)) (sessions zs dg N) s ≤
      sumS zs dg N (fun j τ => (zs j).slackIf dg N k Rw M ρ t' qF qS τ) s +
        ∑ j, ∑ pos, candWeightK (Vs j).c (Vs j).hU (Vs j).nv (Vs j).law.live (Sk j) ρ pos *
          exG (fun rs : VRec Vs Reg' dg N =>
            prob (OffAt (obK (Vs j).c (Vs j).hU (Vs j).nv) (cs j pos) (wv j pos))
            ((zs j).trial dg N k Rw ((zs j).ofBase dg N (rs j).1) pos).g
            ((zs j).trial dg N k Rw ((zs j).ofBase dg N (rs j).1) pos).s) (sessions zs dg N) s := by
  classical
  have hmono : ∀ b : VRec Vs Reg' dg N, Accepts zs b ∧ ¬ Q (layers zs dg N k Rw b) →
      ∃ j, LiveAcceptsZKR (zs j).I (zs j).own (b j).2 ∧ ∃ g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk j ∧
        (zs j).layer dg N k Rw ((zs j).ofBase dg N (b j).1) g ≠
          bitK (Vs j).c (Vs j).hU (Vs j).nv (wv j (posK (Vs j).c (Vs j).hU (Vs j).nv g)) g := by
    rintro b ⟨hacc, hnq⟩
    by_contra hno
    refine hnq (hQ _ fun j g hg => ?_)
    by_contra hne
    exact hno ⟨j, hacc j, g, hg, hne⟩
  have hB : ∀ j τ₀, prob (fun a => LiveAcceptsZKR (zs j).I (zs j).own a ∧
        ∃ g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk j ∧
          (zs j).layer dg N k Rw ((zs j).ofBase dg N τ₀) g ≠
            bitK (Vs j).c (Vs j).hU (Vs j).nv (wv j (posK (Vs j).c (Vs j).hU (Vs j).nv g)) g)
        ((zs j).game dg N) ((zs j).ofBase dg N τ₀) ≤
      (zs j).slackIf dg N k Rw M ρ t' qF qS ((zs j).ofBase dg N τ₀) +
        ∑ pos, candWeightK (Vs j).c (Vs j).hU (Vs j).nv (Vs j).law.live (Sk j) ρ pos *
          prob (OffAt (obK (Vs j).c (Vs j).hU (Vs j).nv) (cs j pos) (wv j pos))
            ((zs j).trial dg N k Rw ((zs j).ofBase dg N τ₀) pos).g
            ((zs j).trial dg N k Rw ((zs j).ofBase dg N τ₀) pos).s := by
    intro j τ₀
    by_cases hj : ∃ tr, AcceptsZKR (zs j).I (zs j).own tr
    swap
    · exact ((prob_mono (fun a ha => (hj ⟨a.2.2, ha.1⟩).elim) _ _).trans (prob_false _ _).le).trans (zero_le ..)
    unfold ZkOuter.slackIf
    split_ifs with hc
    · refine le_trans (prob_auditLive_mono _ _ ?_)
        (((zs j).candTransfer dg N hk hRw hM hρ (hr j) ht' _ hc (Sk j) (wv j)).trans
          (add_le_add le_rfl (Finset.sum_le_sum fun pos _ => ?_)))
      · rintro R' ω' tr ⟨hacc, g, hg, hne⟩
        obtain ⟨u, hu, hgu⟩ := vCovers_full (Vs j).law.src (Sk j) (hfull j) ω' g hg
        refine ⟨hacc, u, hu, g, hgu, ?_⟩
        unfold CandBadZ
        rw [ZkOuter.vb_pos, ZkOuter.vb_bit]
        exact ⟨hg, hne⟩
      · by_cases hp : pos ∈ Sk j
        · rw [hcs j hj pos hp]
        · simp [candWeightK, hp]
    · exact (prob_le_one _ _ _).trans le_self_add
  refine ((prob_mono hmono _ _).trans (prob_seqC_exists_le S _ _
    (fun j r => LiveAcceptsZKR (zs j).I (zs j).own r.2 ∧ ∃ g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk j ∧
      (zs j).layer dg N k Rw ((zs j).ofBase dg N r.1) g ≠
        bitK (Vs j).c (Vs j).hU (Vs j).nv (wv j (posK (Vs j).c (Vs j).hU (Vs j).nv g)) g)
    (fun j τ₀ => (zs j).slackIf dg N k Rw M ρ t' qF qS ((zs j).ofBase dg N τ₀) +
      ∑ pos, candWeightK (Vs j).c (Vs j).hU (Vs j).nv (Vs j).law.live (Sk j) ρ pos *
        prob (OffAt (obK (Vs j).c (Vs j).hU (Vs j).nv) (cs j pos) (wv j pos))
          ((zs j).trial dg N k Rw ((zs j).ofBase dg N τ₀) pos).g
          ((zs j).trial dg N k Rw ((zs j).ofBase dg N τ₀) pos).s) hB s)).trans_eq ?_
  simp only [Finset.sum_add_distrib]
  rw [exG_add]
  refine congrArg₂ (· + ·) rfl ?_
  rw [exG_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [exG_sum]
  exact Finset.sum_congr rfl fun pos _ => exG_const_mul _ _ _ _

variable {Reg₂ : Type}

/-- **The trial for a tagged commit string** at V*'s outer proof: `trialS` at the registration the prover sends. -/
noncomputable def trialV (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (s : Strategy (outerV zs dg N)) (pos : Σ j, PosK (Vs j).c (Vs j).hU (Vs j).nv) : Finder (Finset (obS Vs).Op) :=
  trialS (zs s.1) dg N k Rw (sessOf zs dg N s) pos

/-- **A stage's transfer at V*'s outer proof**: `sessions_stage_le` at the registration the prover sends, the session's
registered digests at `Sk j` being `cs j` at every registration where it accepts some transcript. -/
theorem outerV_stage_le
    (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R₂ j S' R' i, rateZ (planZHJ (Reg := Reg') (zs R₂ j).dj (zs R₂ j).y₀ S' R') i k ≤ ρ) {t' : ℝ}
    (ht' : 0 ≤ t') (qF qS : ℝ) (Sk : (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (hfull : ∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk j →
      ∃ u, g ∈ ((keyProg (Vs j).c (Vs j).hU (Vs j).nv).partition (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).io u)
    (cs : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (hcs : ∀ R₂ j, (∃ tr, AcceptsZKR (zs R₂ j).I (zs R₂ j).own tr) → ∀ pos ∈ Sk j, ∀ R' : Reg',
      (zs R₂ j).vb.cs R' pos = cs j pos)
    (Q : ((j : Fin S) → (Vs j).Bits) → Prop)
    (wv : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c)
    (hQ : ∀ Xt : (j : Fin S) → (Vs j).Bits, (∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk j →
      Xt j g = bitK (Vs j).c (Vs j).hU (Vs j).nv (wv j (posK (Vs j).c (Vs j).hU (Vs j).nv g)) g) → Q Xt)
    (s : Strategy (outerV zs dg N)) :
    prob (fun o => AcceptsV zs o ∧ ¬ Q (layers (zs o.1) dg N k Rw o.2)) (outerV zs dg N) s ≤
      sumV zs dg N (fun R₂ j τ => (zs R₂ j).slackIf dg N k Rw M ρ t' qF qS τ) s +
        ∑ j, ∑ pos, candWeightK (Vs j).c (Vs j).hU (Vs j).nv (Vs j).law.live (Sk j) ρ pos *
          exG (fun rs : VRec Vs Reg' dg N =>
            prob (OffAt (obK (Vs j).c (Vs j).hU (Vs j).nv) (cs j pos) (wv j pos))
            ((zs s.1 j).trial dg N k Rw ((zs s.1 j).ofBase dg N (rs j).1) pos).g
            ((zs s.1 j).trial dg N k Rw ((zs s.1 j).ofBase dg N (rs j).1) pos).s) (sessions (zs s.1) dg N)
            (sessOf zs dg N s) := by
  rw [prob_outerV]
  exact sessions_stage_le (zs s.1) dg N hk hRw hM hρ (hr s.1) ht' qF qS Sk hfull cs (hcs s.1) Q wv hQ
    (sessOf zs dg N s)

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type}

/-- **V*'s public files register the stages' commitments**: at every history `t` and registration `R₂`, where `t`'s
stage-`i` commitment is `e.1` and session `j` accepts some transcript, the digest session `j`'s value binding registers
at each commit string of `Sk i j` (`cs`, the public file's leaf there) is `cmt j e.1`'s. A session that accepts nothing
may be another history's, registering that history's commitments; the transfer reads only accepting sessions. -/
def VCommits (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg) : Prop :=
  ∀ i R ω t (e : Cm × I.Coin), t[i]? = some e → ∀ R₂ j, (∃ tr, AcceptsZKR (zs R ω t R₂ j).I (zs R ω t R₂ j).own tr) →
    ∀ pos ∈ Sk i j, ∀ R' : Reg', (zs R ω t R₂ j).vb.cs R' pos = cmt j e.1 pos

/-- **V*'s circuits decode each stage's message from its commit strings**: layers whose bits on the wires of each
`Sk i j` are those of the values `v j` hold a round-`i` message the inner verifier reads as `msg i v`, at any coins. -/
def VDecodes (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg) (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg) : Prop :=
  ∀ i < I.r, ∀ (ρs : List I.Coin) (Xt : (j : Fin S) → (Vs j).Bits)
    (v : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c),
    (∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk i j →
      Xt j g = bitK (Vs j).c (Vs j).hU (Vs j).nv (v j (posK (Vs j).c (Vs j).hU (Vs j).nv g)) g) →
    Option.Rel (same i ρs) (wmsg Xt)[i]? (some (msg i v))

/-- **The stages are read from units' wires**: each wire of `Sk i j` (`i < r`) is a wire of some unit, so session `j`'s
law, the full draw, draws it. -/
def VFull (r : ℕ) (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)) : Prop :=
  ∀ i < r, ∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk i j →
    ∃ u, g ∈ ((keyProg (Vs j).c (Vs j).hU (Vs j).nv).partition (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).io u

variable (Vs) in
/-- **Wire `g` of session `j` is read**: a committed wire of some unit. -/
def ReadW (j : Fin S) (g : Fin ((keyProg (Vs j).c (Vs j).hU (Vs j).nv).circuit
    (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).N) : Prop :=
  ∃ u, g ∈ ((keyProg (Vs j).c (Vs j).hU (Vs j).nv).partition (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).io u

/-- **`VDecodes` at the read wires**: layers whose bits on the read wires of each `Sk i j` are those of the values `v j`
hold a round-`i` message the inner verifier reads as `msg i v`, at any coins. It asks more than `VDecodes` (the message
must be fixed by the read wires alone), and it is true at V* when every wire `wmsg` depends on is some unit's `io`,
which is what VBridge G's `vbridgeRead_assembled` builds. -/
def VDecodesR (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg) (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg) : Prop :=
  ∀ i < I.r, ∀ (ρs : List I.Coin) (Xt : (j : Fin S) → (Vs j).Bits)
    (v : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c),
    (∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk i j → ReadW Vs j g →
      Xt j g = bitK (Vs j).c (Vs j).hU (Vs j).nv (v j (posK (Vs j).c (Vs j).hU (Vs j).nv g)) g) →
    Option.Rel (same i ρs) (wmsg Xt)[i]? (some (msg i v))

/-- **`VFull` at the read wires**: each read wire of `Sk i j` (`i < r`) is a wire of some unit, which holds of every read
wire (`vFullR_of_read`). -/
def VFullR (r : ℕ) (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)) : Prop :=
  ∀ i < r, ∀ j g, posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk i j → ReadW Vs j g →
    ∃ u, g ∈ ((keyProg (Vs j).c (Vs j).hU (Vs j).nv).partition (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).io u

theorem vFullR_of_read (r : ℕ) (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)) : VFullR r Sk :=
  fun _ _ _ _ _ h => h

/-- **The recursive audit with V*'s `--zk --registered` sessions as the outer proof is sound, with its stages' transfer
from V*'s sessions**: `zk_sessions_recursive_upto` at V*'s tagged openings `obS`, stage `i`'s trial for each tagged
commit string `trialV`, its slack the sessions' expected `slackIf` and its weights `weightS` on `Sk i`, the transfer
from `outerV_stage_le` on `VCommits`, `VDecodes` and `VFull`. -/
theorem zk_sessions_recursive_full
    -- V*'s laws: A3
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    -- V*'s sessions at each history and registration
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the count curve's parameters, and the finders' budgets
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i, rateZ (planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    {t' : ℝ} (ht' : 0 ≤ t') (qF qS qW qR : ℝ)
    -- the recursive prover
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    -- the hidden circuit's statement, the messages V*'s layers hold, and V*'s bridge
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (hbr : VBridge zs dg N k Rw x wmsg)
    -- what the inner verifier reads of each round's message
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsame : ∀ y (ms ms' : List I.Msg) (ρs : List I.Coin), (∀ i < ρs.length, Option.Rel (same i ρs) ms[i]? ms'[i]?) →
      I.accepts y (ms.zip ρs) → I.accepts y (ms'.zip ρs))
    -- the stages: their commit strings, commitments and messages, V*'s registration and decoding of them, the units
    -- they are read from, and cr/sha-512 for the fork finders
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (hcmt : VCommits zs Sk cmt) (hdec : VDecodes wmsg same Sk msg) (hfull : VFull I.r Sk)
    (q : ℕ → ℝ)
    (hCR : ∀ i < I.r, ∀ ω pos,
      (forkAt (obS Vs) σ (cmtS cmt) (fun R ω t s => trialV (zs R ω t) dg N k Rw s) i ω pos).CR H512 (q i))
    -- the inner protocol
    {εin : ℝ≥0∞} (hin : InnerSound I εin) :
    prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
        (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
          (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) +
        ∑ i ∈ Finset.range I.r,
          (expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
              (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackIf dg N k Rw M ρ t' qF qS τ) s) +
            (∑ pos, weightS Vs Sk ρ i pos) * ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin := by
  refine zk_sessions_recursive_upto hA3 zs dg N hk hRw hM hρ hr ht' qF qS qW qR σ x wmsg hbr same hsame
    (obS Vs) (cmtS cmt) (msgS msg) (fun _ R ω t s => trialV (zs R ω t) dg N k Rw s)
    (fun _ R ω t s => sumV (zs R ω t) dg N (fun R₂ j τ => (zs R ω t R₂ j).slackIf dg N k Rw M ρ t' qF qS τ) s)
    (weightS Vs Sk ρ) q (fun i hi ω => AllLeaves.of_forall (fun t s' v e he => ?_) _) hCR hin
  refine (outerV_stage_le (zs σ.1 ω t) dg N hk hRw hM hρ (hr σ.1 ω t) ht' qF qS (Sk i) (hfull i hi)
    (fun j pos => cmt j e.1 pos) (fun R₂ j hj pos hp R' => hcmt i σ.1 ω t e he R₂ j hj pos hp R')
    (fun Xs => Option.Rel (same i (t.map Prod.snd)) (wmsg Xs)[i]? (some (msgS msg i v)))
    (fun j p => projV j (v ⟨j, p⟩)) (fun Xt hXt => hdec i hi _ Xt (fun j p => projV j (v ⟨j, p⟩)) hXt) _).trans
    (add_le_add le_rfl ?_)
  rw [Fintype.sum_sigma]
  exact Finset.sum_le_sum fun j _ => Finset.sum_le_sum fun p _ =>
    mul_le_mul' le_rfl (trialS_le (zs σ.1 ω t _) dg N k Rw _ j p _ _)

/-- **A round fork**: at some round `i < r`, draw `ω` and tagged commit string `pos`, the fork finder of round `i`'s
commitment (`forkAt`: two continuations of the prover's own strategy after its round-`i` commitment, each running the
trial for `pos` at the outer proof) outputs a SHA-512 collision at an outcome it reaches. -/
def RoundFork (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg) : Prop :=
  ∃ i < I.r, ∃ ω pos,
    (forkAt (obS Vs) σ (cmtS cmt) (fun R ω t s => trialV (zs R ω t) dg N k Rw s) i ω pos).Finds H512

/-- **A session fork**: at some leaf the prover reaches (a draw `ω`, a transcript `t` and its strategy `s` of V*'s outer
proof there), at some outcome of V*'s sessions that `s` reaches, the strategy `s` names for some session `j` has a fork
(`ZkOuter.Fork`). -/
def SessionFork (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw M : ℕ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N)) : Prop :=
  ExAt σ fun ω t s => ∃ rs, Game.Outcome (sessions (zs σ.1 ω t s.1) dg N) (sessOf (zs σ.1 ω t) dg N s) rs ∧
    ∃ j, (zs σ.1 ω t s.1 j).Fork dg N k Rw M ((zs σ.1 ω t s.1 j).ofBase dg N (rs j).1)

/-- **The recursive audit with V*'s sessions as the outer proof is sound, in fork form**: a run of the prover's own
strategy finds a SHA-512 collision at a round's commitment (`RoundFork`) or in a session (`SessionFork`), or the audit
accepts a false statement with probability at most the sessions' expected statistical errors (`ZkOuter.boundCR`), `r`
times their expected stage slack (`ZkOuter.slackCR`), and the inner error. `zk_sessions_recursive_full` at no finder
budgets and no SHA-512 evaluations to count: with no fork, every finder satisfies `cr/sha-512` at budget `0` and every
link finder `ecr/sha-512` at cost `0`. -/
theorem zk_sessions_recursive_fork
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i, rateZ (planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (hbr : VBridge zs dg N k Rw x wmsg)
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsame : ∀ y (ms ms' : List I.Msg) (ρs : List I.Coin), (∀ i < ρs.length, Option.Rel (same i ρs) ms[i]? ms'[i]?) →
      I.accepts y (ms.zip ρs) → I.accepts y (ms'.zip ρs))
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (hcmt : VCommits zs Sk cmt) (hdec : VDecodes wmsg same Sk msg) (hfull : VFull I.r Sk)
    {εin : ℝ≥0∞} (hin : InnerSound I εin) :
    RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
          (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin := by
  by_cases h1 : RoundFork zs dg N k Rw σ cmt
  · exact Or.inl h1
  by_cases h2 : SessionFork zs dg N k Rw M σ
  · exact Or.inr (Or.inl h2)
  refine Or.inr (Or.inr ?_)
  have hno := allAt_of_not_exAt h2
  have hB : expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ 0 0 0 0 0 τ) s) ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) := by
    refine expAt_mono_all (hno.mono fun ω t s hs => ?_)
    unfold sumV sumS
    exact exG_mono_outcome _ _ fun rs hrs => Finset.sum_le_sum fun j _ =>
      ZkOuter.boundIf_le _ _ _ _ _ fun hf => hs ⟨rs, hrs, j, hf⟩
  have hsl : expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackIf dg N k Rw M ρ 0 0 0 τ) s) ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) := by
    refine expAt_mono_all (hno.mono fun ω t s hs => ?_)
    unfold sumV sumS
    exact exG_mono_outcome _ _ fun rs hrs => Finset.sum_le_sum fun j _ =>
      ZkOuter.slackIf_le _ _ _ _ fun hf => hs ⟨rs, hrs, j, hf⟩
  refine (zk_sessions_recursive_full hA3 zs dg N hk hRw hM hρ hr le_rfl 0 0 0 0 σ x wmsg hbr same hsame Sk cmt msg
    hcmt hdec hfull (fun _ => 0) (fun i hi ω pos => Finder.cr_of_not_finds (fun hf => h1 ⟨i, hi, ω, pos, hf⟩) 0)
    hin).trans (add_le_add (add_le_add hB ?_) le_rfl)
  simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_div, Real.sqrt_zero, ENNReal.ofReal_zero,
    mul_zero, add_zero]
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  exact mul_le_mul' le_rfl hsl

end FlockSoundness.Discharge.Recursion
