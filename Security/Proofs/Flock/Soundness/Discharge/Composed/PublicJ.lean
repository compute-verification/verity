import Proofs.Flock.Soundness.Discharge.Composed.CustodyJ
import Proofs.Flock.Soundness.Discharge.ZkLink.Public

/-!
# `zk_session_soundJ_custody`, "wrong" measured against the public file

`zk_session_soundJ_custody_reg`: `zk_session_soundJ_custody` with its event's "wrong" `wrongRegZ` at the prover's
registration (`ZkLink.Public`) in place of `Partition.wrong`, at the same bound. A unit counts as wrong unless it is
correct in the committed transcript and one opening per commit string makes the key program's evaluation on the committed
inputs, on every committed wire the unit reads or computes, the bit of an opening of that wire's registered commit
string: on its outputs, the program's output on the committed inputs opens the registered output commitments. Its
wrong units contain `zk_session_soundJ_custody`'s (`wrong_subset_wrongRegZ`), so its event implies that one's.

* `Layout.keyProg_readsOwn`: the key program's units read only their own gates and input gates (from `keyProg_reads`),
  which `zk_flock_countC_reg` needs, so that on each unit's wires the key program's evaluation is the unit's own
  (`Partition.correct_eval`);
* `rsZJ_wrongRegZ`: the registered commit strings are the public file's, so `wrongRegZ` is one set at every
  registration.
-/

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3 Audit Placed

/-- **A unit cut out as a rows unit reads only its own gates and input gates**, when every column of it that is no
computed column is on an input gate. -/
theorem readsOwn_of_rows {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {R : Fin n → Rows}
    (I : ∀ u, P.IsRowsUnit u (R u))
    (hread : ∀ u (y : Fin (R u).w), (∀ i, y ≠ (R u).comp i) → P.unit ((I u).wire y) = none) (u : Fin n) :
    P.ReadsOwn u := by
  intro h x hh hx
  obtain ⟨i, rfl⟩ := (I u).onto h hh
  rw [(I u).op i] at hx
  simp only [Audit.Op.args, List.mem_append, List.mem_map] at hx
  obtain ⟨y, rfl⟩ : ∃ y, (I u).wire y = x := by
    rcases hx with ⟨y, -, e⟩ | ⟨y, -, e⟩ <;> exact ⟨y, e⟩
  by_cases hcomp : ∃ j, y = (R u).comp j
  · obtain ⟨j, rfl⟩ := hcomp
    exact Or.inl ((I u).mem j)
  · exact Or.inr (hread u y fun j e => hcomp ⟨j, e⟩)

/-- **`keyProg`'s units read only their own gates and input gates.** -/
theorem keyProg_readsOwn (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ) (outs : Finset (Fin (keyN c hU n)))
    (u : Fin n) : ((keyProg c hU n).partition outs).ReadsOwn u :=
  readsOwn_of_rows (fun u => Const.Prog.instC (keyProg c hU n) outs u)
    (fun u y hy => keyProg_reads c hU n outs u y hy) u

end FlockSoundness.Discharge.Layout

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

/-- **The registered commit strings are the public file's**: the rows' value binding (`rsZJ`) reads the registration
only through the public file `pub` (`layZJ`'s population is `fun _ => pub`), so its `wrongRegZ` is one set at every
registration. -/
theorem rsZJ_wrongRegZ (dj : Finset (Fin n) → Lean.Json) {I : Inputs} {mPts : ℕ} {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit) {pub : Public}
    (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOk c = true) (R R' : Reg) (Xv : Fin (Layout.keyN c hU n) → Bool) :
    wrongRegZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
        ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope)) R Xv =
      wrongRegZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
        ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope)) R' Xv := by
  apply wrongRegZ_congr
  intro p
  rfl

/-- **One `--zk` sampled session of `J` tables is sound against the registration, at the coin server's custody.**
`zk_session_soundJ_custody` with "wrong" `wrongRegZ` at the prover's registration `reg σ`, read through the rows'
value binding (`rsZJ`): the registered commit strings are the public file's `b ‖ c` digests at rows and its output
words' leaves elsewhere. -/
theorem zk_session_soundJ_custody_reg
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOk c = true)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    (hTags : TagsOkZJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZJ (Reg := Reg) dj y₀ S R).length) wr kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZJ dj y₀) I wr (opnAtZJ (planZJ dj y₀) I dj) dg (decAtZJ (planZJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJ_live I ht dj y₀ wr _ hTags hCust.1 fun _ => drawOkZKJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (wrongRegZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
          ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope)) (reg σ)
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
            ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
          ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope))
          k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ hCR
  refine zk_headline_exec_tables execArith hs512 (planZJ dj y₀) I wr (opnAtZJ (planZJ dj y₀) I dj) dg
    (decAtZJ (planZJ dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJ_live I ht dj y₀ wr _ hTags hCust.1 fun _ => drawOkZKJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    _ _ ?_
  refine (zk_flock_countC_reg execArith hs512 _ (planZJ dj y₀) (tabZJ dj y₀) _
    (loweringSoundZK_of_place execArith _ (planZJ dj y₀) (tabZJ dj y₀)
      ((sitesZJ dj y₀ ht).placeKey hU (sitesZJ_parse dj y₀ ht hc) (Layout.outsOf c hU n)))
    (Layout.keyProg_readsOwn c hU n _)
    hRw hM hρ hr (jointZC_ksBound execArith execArith_correct hs512 (planZJ dj y₀) hk) ht' K₀ σ).trans ?_
  exact add_le_add (add_le_add (Tables.execOS_subset_miss_le kd hkd Lb src hA3 K₀)
    (ksAvgZC_le_strict execArith hs512 (planZJ dj y₀) k (reg σ) (cont σ) hCR)) le_rfl

end FlockSoundness.Discharge.Composed
