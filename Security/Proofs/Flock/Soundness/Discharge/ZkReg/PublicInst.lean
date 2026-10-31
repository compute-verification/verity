import Proofs.Flock.Soundness.Discharge.ZkReg.PublicHJ
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundInst

set_option Elab.async false

/-!
# One hidden-output `--zk --registered` sampled session of `J` tables, at the template instance's program

`checked_of_acceptsHJ`, `zk_session_soundHJR_custody_reg` and `zk_session_soundHJR_custody_drawn_reg` at `instProg c n`
in place of `Placed.keyProg c hU n` (`ZkHidden.SoundInst`), with the circuit conditions `hrf`, `hpr` and `hw` in place
of `scopeOkH`.

* `checked_of_acceptsHJI`: the registered reads are checked at every accepted outcome. The registered leaf at a row of
  `instLayoutH` is the public file's, as at `keyLayoutH`, so `checked_of_acceptsHJ`'s argument reads unchanged.
* `zk_session_soundHJR_custody_regI`, `zk_session_soundHJR_custody_drawn_regI`: the session's bound with "wrong"
  `wrongRegZ` at `rsZHJI`, and the registered check at every accepted outcome.
-/

namespace FlockSoundness.Discharge.ZkReg

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind ZkHidden HiddenExec
  HiddenDec DecodeZL Composed Placed StrictCR
open FlockSoundness.Discharge.Exec (Inputs recordDraw)
open FlockSoundness.Discharge.Inst
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

/-- **The registered reads are checked at every accepted hidden-output outcome of `J` tables, at `instProg`.** At an
outcome `verify --zk --registered F` accepts (`own` F's ports), each registered read of `layShHJI` that is the
verifier's own at the draw of the table its unit is drawn into (`OwnReadsHJ`) has its registered leaf open the root
`own` holds at the position `own` gives, through `regPathsHJ`. -/
theorem checked_of_acceptsHJI (I : Inputs) (ht : I.tags.typed = false) (own : Array Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hrf : Hidden.Any.RowsFitS c)
    (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)
    (hTags : TagsOkZHJ I) (o : Reg × Finset (Fin n) × Exec.Transcript) (hacc : LiveAcceptsZKR I own o)
    (rd : FlockSoundness.Registered.ReadsX (layShHJI (Reg := Reg) dj y₀ ht hc hrf (pub := pub))
      (rsShHJI dj y₀ ht hc hrf hpub hU hpr hw))
    (hown : OwnReadsHJ I own c pub o.2.2 rd.isRead rd.port rd.readAt) :
    rd.Checked o.1 (regPathsHJ I c pub o.2.2 rd.port) := by
  intro p hp
  obtain ⟨u, q, d, D, dp, idx, rfl, hmem, hfind, hd, hD, hdp, hidx, hu, hread⟩ := hown p hp
  obtain ⟨spec, sts, hs, hreg, -, -⟩ := hacc
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJH hTags.1 hd hs
  have hjs : (partOf I d u.val).val < sts.size := by rw [hsz]; exact (partOf I d u.val).isLt
  have hst := hstJ _ hjs
  have s := Hidden.setupHidden_spec' hst
  have hcst : sts[(partOf I d u.val).val].c = c := by
    have h1 := s.parseF ((FlockSoundness.Discharge.Exec.tagsAt_isTmpl _ _ _).trans (HmOut.parse_isTmpl hc))
    rw [parseH_tagsAt ((zkTags_typed I).trans ht), hc] at h1
    exact (Except.ok.inj h1).symm
  obtain ⟨pop', hpop', -, hdr⟩ := s.pub
  rw [hcst, loadPublicH_tagsAt, hpub] at hpop'
  cases hpop'
  have hdr' := hdr _ rfl
  rw [hcst, hdp] at hdr'
  have hpd : sts[(partOf I d u.val).val].pub = dp := (Except.ok.inj hdr').symm
  have hdpR : HmRow.drawn { c with outPorts := #[] } pub (drawJ I d (partOf I d u.val)) = .ok dp := hdp
  have hck := checkRegistered_ok (array_forM_ok hreg _ hjs)
  rw [hcst, hpd] at hck
  obtain ⟨tb, idx', htb, hidx', hports⟩ := check_tables hck
  rw [hidx] at hidx'
  cases hidx'
  obtain ⟨p', positions, paths, hp', hent, hall⟩ := checkPort_at (hports _ hmem)
  rw [hfind] at hp'
  cases hp'
  obtain ⟨D', hD', -, hn, -, -⟩ := Layout.drawn_spec hdpR
  rw [hD] at hD'
  cases hD'
  have hi : D.proved.idxOf u.val < D.proved.size := Array.idxOf_lt_length_of_mem hu
  obtain ⟨-, hreads, hopens⟩ := hall _ (by rw [hn]; exact hi)
  rw [hread] at hreads
  have hra : rd.readAt (.inl (u, q)) =
      positions.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) 0 := by
    simpa using hreads
  have hpath : regPathsHJ I c pub o.2.2 rd.port (.inl (u, q)) =
      paths.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) ByteArray.empty := by
    simp [regPathsHJ, hd, hD, hdp, htb, hent, Except.toOption]
  have hleaf : ((layShHJI (Reg := Reg) dj y₀ ht hc hrf (pub := pub)).regLeaf o.1 (.inl (u, q))).1 =
      Hm96.Default512.treeLeaf (pub.digest u.val q.val c.ports.size) := by
    show (H512 (K512.leafPrefix ++ (pub.digest u.val q.val c.ports.size).data.toList)).1 = _
    rw [H512_val]
    show Sha512.hash ⟨(Hm96.Default512.leafPrefix.data.toList ++ (pub.digest u.val q.val c.ports.size).data.toList).toArray⟩ =
      Sha512.hash (Hm96.Default512.leafPrefix ++ pub.digest u.val q.val c.ports.size)
    rw [← data_append, bytes_of_list]
  have hget : D.proved[D.proved.idxOf u.val]'hi = u.val := by
    have := Array.findIdx_getElem (xs := D.proved) (p := (· == u.val)) (w := hi)
    simp only [beq_iff_eq] at this
    exact this
  have hdig : pub.digest u.val q.val c.ports.size =
      Registered.tableRow dp tb q.val (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) := by
    have h1 := Hidden.drawn_digestW (Hidden.loadPublic_nil hpub) hdpR hD (Layout.proved_lt_population hD) hi q.isLt
    rw [hget] at h1
    rw [← h1]
    simp only [Public.digest, htb, Registered.tableRow, Registered.ref]
  show Registered.opensLeaf (rd.port (.inl (u, q))) _ (rd.readAt (.inl (u, q))) _ = true
  rw [hleaf, hra, hpath, hdig]
  exact hopens

/-- **One hidden-output `--zk --registered` sampled session of `J` tables is sound against the registration under
`live-verifier`, and its registered reads are checked, at `instProg`.** `zk_session_soundHJR_custody_reg` with the
template instance's program: "wrong" is `wrongRegZ` at the prover's registration `reg σ`, read through `rsZHJI`. -/
theorem zk_session_soundHJR_custody_regI
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hrf : Hidden.Any.RowsFitS c) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_liveI I ht hc dj y₀ wr _ hTags hCust.1
          fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZKR I own o ∧
        K₀ ≤ (wrongRegZ execArith ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          (reg σ)
          (XplurZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          k Rw M ρ t' (reg σ) (cont σ) ∧
    ∀ o : Reg × Finset (Fin n) × Exec.Transcript, LiveAcceptsZKR I own o →
      ∀ rd : FlockSoundness.Registered.ReadsX (layShHJI (Reg := Reg) dj y₀ ht hc hrf (pub := pub))
          (rsShHJI dj y₀ ht hc hrf hpub hU hpr hw),
        OwnReadsHJ I own c pub o.2.2 rd.isRead rd.port rd.readAt → rd.Checked o.1 (regPathsHJ I c pub o.2.2 rd.port) := by
  intro σ hCR
  refine ⟨?_, fun o hacc rd hown => checked_of_acceptsHJI I ht own dj y₀ hc hrf hU hpub hpr hw hTags o hacc rd hown⟩
  exact le_trans (prob_mono (q := fun o => LiveAcceptsZK I o ∧ _) (fun o h => ⟨acceptsZK_of_R h.1, h.2⟩) _ _)
    (zk_session_soundHJ_custody_regI I ht hc hU hpub hrf hpr hw kd hkd hdiv dj y₀ wr dg N Lb src hA3 hTags hCust
      hk hRw hM hρ hr ht' K₀ τ hCR)

/-- **One hidden-output `--zk --registered` sampled session of `J` tables is sound at the units it draws, and its
registered reads are checked, at `instProg`.** `zk_session_soundHJR_custody_drawn_reg` with the template instance's
program. The draw is the session's, so no law over it is assumed (no A3, no `kd ≤ n`). -/
theorem zk_session_soundHJR_custody_drawn_regI
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hrf : Hidden.Any.RowsFitS c) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)
    (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_liveI I ht hc dj y₀ wr _ hTags hCust.1
          fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZKR I own o ∧ ∃ u ∈ o.2.1,
        u ∈ wrongRegZ execArith ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          (reg σ)
          (XplurZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
            (reg σ) (cont σ) k Rw))
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          k Rw M ρ t' (reg σ) (cont σ) ∧
    ∀ o : Reg × Finset (Fin n) × Exec.Transcript, LiveAcceptsZKR I own o →
      ∀ rd : FlockSoundness.Registered.ReadsX (layShHJI (Reg := Reg) dj y₀ ht hc hrf (pub := pub))
          (rsShHJI dj y₀ ht hc hrf hpub hU hpr hw),
        OwnReadsHJ I own c pub o.2.2 rd.isRead rd.port rd.readAt → rd.Checked o.1 (regPathsHJ I c pub o.2.2 rd.port) := by
  intro σ hCR
  refine ⟨?_, fun o hacc rd hown => checked_of_acceptsHJI I ht own dj y₀ hc hrf hU hpub hpr hw hTags o hacc rd hown⟩
  exact le_trans (prob_mono (q := fun o => LiveAcceptsZK I o ∧ _) (fun o h => ⟨acceptsZK_of_R h.1, h.2⟩) _ _)
    (zk_session_soundHJ_custody_drawn_regI I ht hc hU hpub hrf hpr hw kd hdiv dj y₀ wr dg N Lb src hTags hCust
      hk hRw hM hρ hr ht' τ hCR)

end FlockSoundness.Discharge.ZkReg
