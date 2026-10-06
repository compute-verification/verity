import Proofs.Flock.Soundness.Discharge.ZkHidden.Sound
import Proofs.Flock.Soundness.Discharge.ZkReg.BindH
import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedH
import Proofs.Flock.Soundness.Discharge.ZkRegV2.BindH

/-!
# One hidden-output `--zk` sampled session with registered reads

`zk_session_soundHR`: `ZkHidden.zk_session_soundH` for `flock-verify verify --zk --registered F` sessions over hidden
outputs (`Stmt.setupHidden`, the default `--statement`), `own` F's ports, over a public file with or without shared-row
tables (no `pub.tables = none`, which would make the event empty: `Registered.check` refuses a registered port outside a
shared-row file's tables), at a circuit whose rows are whole-block `sha512/row/v2` bit rows with one output port per
output row (`ZkRegV2.scopeOkHV2`), the rows the registered check admits (`Registered.Port.fits`). It is to
`zk_session_soundH` what `zk_session_soundR` is to `zk_session_sound`. Its first conjunct is `zk_session_soundH`'s bound
at the event `LiveAcceptsZKR I own` and the rows `rsZHSV2`; its second is the registered check's consequence at every
accepted outcome: the shadow layout's registered reads that are the verifier's own at the outcome's draw (`OwnReadsH`)
are `Checked` at the outcome's registration, through the paths the drawn statement gives (`regPathsH`).
`checked_of_acceptsHV2` is `checked_of_acceptsH` at any rows of the shadow layout `layShHV2`: the registered check reads
the layout's leaves, not its rows' bytes.

What it rests on: `zk_session_soundH`'s assumptions, and nothing more for the second conjunct, which reads only the
executable's checks.
-/

namespace FlockSoundness.Discharge.ZkReg

variable {mPts n : ℕ} {Reg : Type}

section Checked

open Flock FlockLevel3 Refine ZkLink ZkExec ZkBind ZkHidden HiddenExec HiddenDec DecodeZL
open FlockSoundness.Discharge.Exec (Inputs recordDraw)

/-- **The registered reads are checked at every accepted hidden-output outcome**, at any rows `rows` of the shadow
layout `layShHV2`: `checked_of_acceptsH`, whose argument reads the layout's registered leaves only. -/
theorem checked_of_acceptsHV2 (I : Inputs) (ht : I.tags.typed = false) (own : Array Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : ZkRegV2.scopeOkHV2 c = true)
    (rows : Binding.HmRows (layShHV2 (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub)))
    (hTags : TagsOkH I) (o : Reg × Finset (Fin n) × Exec.Transcript) (hacc : LiveAcceptsZKR I own o)
    (rd : FlockSoundness.Registered.Reads (layShHV2 (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub)) rows)
    (hown : OwnReadsH own c pub o.2.2 rd.isRead rd.port rd.readAt) :
    rd.Checked o.1 (regPathsH c pub o.2.2 rd.port) := by
  intro p hp
  obtain ⟨u, q, d, D, dp, idx, rfl, hmem, hfind, hd, hD, hdp, hidx, hu, hread⟩ := hown p hp
  obtain ⟨spec, sts, hs, hreg, -, -⟩ := hacc
  have hh : (zkTags I).hiddenOutputs = true := (zkTags_hiddenOutputs I).trans hTags.1
  rcases setupTables_okH hh hs with ⟨-, st, hst, -, rfl⟩ | ⟨hc', -⟩
  swap
  · exact absurd hTags.2.2.2 (by change 1 < I.count at hc'; omega)
  change Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (recordDraw o.2.2.record)
    I.partition I.program = .ok st at hst
  rw [hd] at hst
  have s := Hidden.setupHidden_spec' hst
  have hcst : st.c = c := by
    have h1 := s.parse
    rw [hc] at h1
    exact (Except.ok.inj h1).symm
  obtain ⟨pop', hpop', -, hdr⟩ := s.pub
  rw [hcst, hpub] at hpop'
  cases hpop'
  have hdr' := hdr d rfl
  rw [hcst, hdp] at hdr'
  have hpd : st.pub = dp := (Except.ok.inj hdr').symm
  have hdpR : HmRow.drawn { c with outPorts := #[] } pub d = .ok dp := hdp
  have hck := checkRegistered_ok (array_forM_ok hreg 0 (by simp))
  simp only [List.getElem_toArray, List.getElem_cons_zero] at hck
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
  have hpath : regPathsH c pub o.2.2 rd.port (.inl (u, q)) =
      paths.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) ByteArray.empty := by
    simp [regPathsH, hd, hD, hdp, htb, hent, Except.toOption]
  have hleaf : ((layShHV2 (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub)).regLeaf o.1 (.inl (u, q))).1 =
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

end Checked

section Sound

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind ZkHidden HiddenExec
  HiddenDec Composed Placed StrictCR DecodeZL
open scoped ENNReal

/-- **One hidden-output `--zk --registered` sampled session over `sha512/row/v2` rows is sound, and its registered
reads are checked.** At a `scopeOkHV2` circuit, with the rows `rsZHSV2`. -/
theorem zk_session_soundHR
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : ZkRegV2.scopeOkHV2 c = true)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKH I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkH I) (hRec : RecordsLiveZK I dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planAtZKH dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (ZkRegV2.scopeOkHV2_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planAtZKH dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (ZkRegV2.scopeOkHV2_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsZKR I own o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
            ((rsZHSV2 dj y₀ ht hc hscope hU hpub).bindingZ (rsZHSV2_computes dj y₀ ht hc hscope hU hpub))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (ZkRegV2.scopeOkHV2_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (ZkRegV2.scopeOkHV2_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (ZkRegV2.scopeOkHV2_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZHSV2 dj y₀ ht hc hscope hU hpub).bindingZ (rsZHSV2_computes dj y₀ ht hc hscope hU hpub))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (ZkRegV2.scopeOkHV2_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_liveH I ht hc (ZkRegV2.scopeOkHV2_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)) τ)) ∧
    ∀ o : Reg × Finset (Fin n) × Exec.Transcript, LiveAcceptsZKR I own o →
      ∀ rd : FlockSoundness.Registered.Reads (layShHV2 (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
          (rsShHSV2 dj y₀ ht hc hscope hU hpub),
        OwnReadsH own c pub o.2.2 rd.isRead rd.port rd.readAt → rd.Checked o.1 (regPathsH c pub o.2.2 rd.port) := by
  refine ⟨?_, fun o hacc rd hown =>
    checked_of_acceptsHV2 I ht own dj y₀ hc hU hpub hscope _ hTags o hacc rd hown⟩
  have hdec := decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _
    (runsZC_liveH I ht hc (ZkRegV2.scopeOkHV2_spec hU hscope).2.1 dj y₀ wr _ tablesAtZKH_live hTags hRec hDraw)
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hc' := zk_flock_countC_execOS_atHSV2 dj y₀ ht hc hscope hU hpub hs512 σs ks hks St hS Kd pop Lb src hA3 hk
    hRw hM hρ hr ht' K₀ (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ) hCR (execAccept := fun o => o.1 = true)
    (StrictCR.TreeRewind.ofNever hnever) (ofNever_finderCR hs512.H hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hc'
  exact le_trans (prob_mono (fun o h => ⟨acceptsZK_of_R h.1, h.2⟩) _ _)
    (zk_headline_exec_on execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ _ _ hc')

end Sound

end FlockSoundness.Discharge.ZkReg
