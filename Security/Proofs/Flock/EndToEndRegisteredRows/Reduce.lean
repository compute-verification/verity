import Proofs.Flock.EndToEndRegisteredRows.Statement
import Proofs.Flock.EndToEndRegistered
import Proofs.Flock.Soundness.Registered.ReadsAt

/-!
# `EndToEndRegisteredRows` from the rows' extraction bound

`EndToEndRegisteredRows_of_rowsLink` proves `Flock.Guarantees.EndToEndRegisteredRows` from one open piece,
`RowsLinkLe`: the bound with the registered reads' term removed, at outcomes whose registration is on the registrant's
leaves (`¬ rd.Off o.1`), and with only the input rows in the event. The rest is proved here:

* **output rows**: a drawn unit not wrong on the registration (`wrongRegZ`) is correct in the value layer, so on each of
  its committed wires the value layer is the circuit's evaluation on its inputs (`Partition.correct_eval`, with
  `Layout.keyProg_readsOwn`);
* **the registered reads' term**: at an accepted outcome whose reads the verifier owns, the registered check holds at
  the outcome's registration (`EndToEndRegisteredDrawn`'s second conjunct), so accepting while a read's leaf is off is
  at most `q_R²/2^513` (`Registered.ReadsX.prob_off_le_at`).

`RowsLinkLe` is `EndToEndRegisteredDrawn`'s first conjunct with the input rows added to "wrong" and `δ_rows` to the
bound: its proof is `zk_session_soundHJR_custody_drawn_reg`'s, at an analysis whose link term carries the rows finder
too (the note's proof plan).
-/

namespace FlockSoundness.Discharge.ZkReg

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **The rows' extraction bound** (open): in `EndToEndRegisteredRows`' setting, accepting while some drawn unit is
wrong on the registration, or, with the registration on the registrant's leaves at every owned read, carries at an
owned registered read a bit other than the registrant's row's, has probability at most `ε_ks + δ_link + δ_rows`. -/
def RowsLinkLe : Prop :=
  ∀ (n : ℕ) (Reg : Type) (own : Array Flock.Registered.Port)
    (I : Inputs) (ht : I.tags.typed = false)
    (c : Flock.Circuit) (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH c = true)
    (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (rd : FlockSoundness.Registered.ReadsX (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHJS dj y₀ ht hc hscope hU hpub))
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t')
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    (qF qS : ℝ),
    let L := Law.execOS (n := n) src (.subset kd) n
    let G := auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)
    let P := (keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)
    let vb := (rsZHJS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHJS_computes dj y₀ ht hc hscope hU hpub)
    let wo : vb.Pos → vb.Opening := fun p => ((p, rd.wv p, rd.ws p) : vb.Opening)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht hc (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags hCust.1
          fun _ => Discharge.ZkReg.drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    let Xv := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    LinkCRZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw M t' →
    RowsCRZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) rd.isRead wo k Rw t' →
    prob (fun o => LiveAcceptsZKR I own o ∧ ∃ u ∈ o.2.1,
        u ∈ wrongRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) Xv ∨
        (¬ rd.Off o.1 ∧ OwnReadsHJ I own c pub o.2.2 rd.isRead rd.port rd.readAt ∧
          ∃ g ∈ P.io u, rd.isRead (vb.pos g) ∧ Xv g ≠ vb.bit (vb.val (wo (vb.pos g))) g))
        G τ ≤
      ksAvgStrictZ (planZHJ dj y₀) L k qF qS (reg σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))

/-- **`EndToEndRegisteredRows` from the rows' extraction bound.** -/
theorem EndToEndRegisteredRows_of_rowsLink (hL : RowsLinkLe) : Flock.Guarantees.EndToEndRegisteredRows := by
  intro n Reg own I ht c hc hU pub hpub hscope kd hdiv dj S₀ d₀ y₀ rd wr dg N Ω _ _ Lb src hTags hCust k Rw M hk hRw
    hM ρ hρ hr t' ht' τ qF qS qR L G P vb wo σ hCR hA2 hRows hReg
  have hD := Flock.SecurityProofs.EndToEndRegisteredDrawn n Reg I ht own c hc hU pub hpub hscope kd hdiv dj S₀ d₀ y₀
    wr dg N Ω Lb src hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' τ qF qS hCR hA2
  have hLk := hL n Reg own I ht c hc hU pub hpub hscope kd hdiv dj S₀ d₀ y₀ rd wr dg N Ω Lb src hTags hCust k Rw M hk
    hRw hM ρ hρ hr t' ht' τ qF qS hCR hA2 hRows
  set Xv := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
  set own' := fun (o : Reg × Finset (Fin n) × Discharge.Exec.Transcript) =>
    OwnReadsHJ I own c pub o.2.2 rd.isRead rd.port rd.readAt
  have hReg' := rd.prob_off_le_at G τ (fun o => o.1) (fun o => LiveAcceptsZKR I own o ∧ own' o)
    (fun o => regPathsHJ I c pub o.2.2 rd.port) (fun o ho => hD.2 o ho.1 rd ho.2) hReg
  have hsub : ∀ o : Reg × Finset (Fin n) × Discharge.Exec.Transcript,
      (LiveAcceptsZKR I own o ∧ ∃ u ∈ o.2.1,
        u ∈ wrongRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) Xv ∨
        ¬ RowsRightZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ)
          (fun p => own' o ∧ rd.isRead p) wo k Rw u) →
      (LiveAcceptsZKR I own o ∧ ∃ u ∈ o.2.1,
        u ∈ wrongRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) Xv ∨
        (¬ rd.Off o.1 ∧ own' o ∧
          ∃ g ∈ P.io u, rd.isRead (vb.pos g) ∧ Xv g ≠ vb.bit (vb.val (wo (vb.pos g))) g)) ∨
      ((LiveAcceptsZKR I own o ∧ own' o) ∧ rd.Off o.1) := by
    rintro o ⟨hA, u, hu, hW | hR⟩
    · exact Or.inl ⟨hA, u, hu, Or.inl hW⟩
    by_cases hW : u ∈ wrongRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) Xv
    · exact Or.inl ⟨hA, u, hu, Or.inl hW⟩
    have hRR : RightRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) Xv u := by
      by_contra h
      exact hW ((mem_wrongRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb).2 h)
    have heval := P.correct_eval (Discharge.Layout.keyProg_readsOwn c hU n _ u) hRR.1
    obtain ⟨hOwn, g, hg, hrd, hne⟩ : own' o ∧
        ∃ g ∈ P.io u, rd.isRead (vb.pos g) ∧ Xv g ≠ vb.bit (vb.val (wo (vb.pos g))) g := by
      by_contra hno
      refine hR fun g hg => ⟨fun ⟨hOwn, hrd⟩ => ?_, heval g hg⟩
      by_contra hne
      exact hno ⟨hOwn, g, hg, hrd, hne⟩
    by_cases hoff : rd.Off o.1
    · exact Or.inr ⟨⟨hA, hOwn⟩, hoff⟩
    · exact Or.inl ⟨hA, u, hu, Or.inr ⟨hoff, hOwn, g, hg, hrd, hne⟩⟩
  exact (prob_mono hsub G τ).trans ((prob_or_le _ _ G τ).trans (add_le_add hLk hReg'))

end FlockSoundness.Discharge.ZkReg
