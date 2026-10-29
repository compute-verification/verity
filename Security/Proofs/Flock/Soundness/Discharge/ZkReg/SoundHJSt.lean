import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedHJ
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundStmt

set_option Elab.async false

/-!
# The registered `--zk` session of `J` tables at the session's statement circuit

`checked_of_acceptsHJ` and `zk_session_soundHJR` read the session's circuit from `HmOut.parse`, which refuses a templated
file. Their `…St` forms here read it from `HmOut.parseStmt` (`hcS`, a templated file's committed circuit unrolled), with
the rows' fit `hv1 : RowsFitS c` given, at `ZkHidden`'s statement-circuit rows (`layShHJSt`, `rsShHJSt`, `rsZHJSt`), which
read no `pub.tables = none`, so a shared-row public file is in.

Each table's statement commits the committed circuit `cs` (`HmOut.unrolled cs = c`), and `checkRegistered` reads it
there. Unrolling changes only the unit net and the template (`Hidden.Committed.unrolled_eq`), and the registered check,
the public file's load and the draw read neither (`check_unrolledOf`, `loadPublic_unrolledOf`, `drawn_unrolledOf`), so the
check at `cs` is the check at `c`.
-/

namespace FlockSoundness.Discharge.ZkReg

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind ZkHidden HiddenExec
  HiddenDec Composed Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Exec (Inputs recordDraw)
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

section unrolled

variable (cs : Flock.Circuit) (nets : Array (String × Net))

theorem check_unrolledOf (pub : Public) (own : Option (Array Registered.Port)) (derived : Bool) :
    Registered.check { cs with nets := nets, tmpl := none } pub own derived = Registered.check cs pub own derived :=
  rfl

theorem loadPublic_unrolledOf (t : Tags) (pf : ByteArray) :
    HmOut.loadPublic t { cs with nets := nets, tmpl := none } pf = HmOut.loadPublic t cs pf :=
  rfl

theorem drawn_unrolledOf (pub : Public) (d : Lean.Json) :
    HmOut.drawn { cs with nets := nets, tmpl := none } pub d = HmOut.drawn cs pub d :=
  rfl

end unrolled

/-- **The registered reads are checked at every accepted hidden-output outcome of `J` tables**, at the session's
statement circuit (`checked_of_acceptsHJ` at `HmOut.parseStmt`). -/
theorem checked_of_acceptsHJSt (I : Inputs) (ht : I.tags.typed = false) (own : Array Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    {c : Flock.Circuit} (hcS : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c)
    (hv1 : Hidden.Any.RowsFitS c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hscope : Layout.scopeOkH c = true)
    (hTags : TagsOkZHJ I) (o : Reg × Finset (Fin n) × Exec.Transcript) (hacc : LiveAcceptsZKR I own o)
    (rd : FlockSoundness.Registered.ReadsX (layShHJSt (Reg := Reg) dj y₀ ht hcS hv1 hscope hU (pub := pub))
      (rsShHJSt dj y₀ ht hcS hv1 hscope hU hpub))
    (hown : OwnReadsHJ I own c pub o.2.2 rd.isRead rd.port rd.readAt) :
    rd.Checked o.1 (regPathsHJ I c pub o.2.2 rd.port) := by
  intro p hp
  obtain ⟨u, q, d, D, dp, idx, rfl, hmem, hfind, hd, hD, hdp, hidx, hu, hread⟩ := hown p hp
  obtain ⟨spec, sts, hs, hreg, -, -⟩ := hacc
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJH hTags.1 hd hs
  have hjs : (partOf I d u.val).val < sts.size := by rw [hsz]; exact (partOf I d u.val).isLt
  have hst := hstJ _ hjs
  have s := Hidden.setupHidden_spec' hst
  have sc := Hidden.setupHidden_specCC hst
  have htyJ : (tagsAt (zkTags I) I.count (partOf I d u.val).val).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ((zkTags_typed I).trans ht)
  have hcu : HmOut.unrolled sts[(partOf I d u.val).val].c = c := by
    have h1 := sc.parseStmt_eq htyJ
    rw [parseStmtH_tagsAt ((zkTags_typed I).trans ht), hcS] at h1
    exact (Except.ok.inj h1).symm
  obtain ⟨nets, he⟩ := sc.committed.unrolled_eq htyJ
  have hce : c = { sts[(partOf I d u.val).val].c with nets := nets, tmpl := none } := hcu.symm.trans he
  obtain ⟨pop', hpop', -, hdr⟩ := s.pub
  rw [loadPublicH_tagsAt, ← loadPublic_unrolledOf _ nets, ← hce, hpub] at hpop'
  cases hpop'
  have hdr' := hdr _ rfl
  rw [← drawn_unrolledOf _ nets, ← hce, hdp] at hdr'
  have hpd : sts[(partOf I d u.val).val].pub = dp := (Except.ok.inj hdr').symm
  have hdpR : HmRow.drawn { c with outPorts := #[] } pub (drawJ I d (partOf I d u.val)) = .ok dp := hdp
  have hck := checkRegistered_ok (array_forM_ok hreg _ hjs)
  rw [hpd, ← check_unrolledOf _ nets, ← hce] at hck
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
  have hleaf : ((layShHJSt (Reg := Reg) dj y₀ ht hcS hv1 hscope hU (pub := pub)).regLeaf o.1 (.inl (u, q))).1 =
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

/-- **One hidden-output `--zk --registered` sampled session of `J` tables is sound, and its registered reads are
checked**, at the session's statement circuit (`zk_session_soundHJR` at `HmOut.parseStmt`). -/
theorem zk_session_soundHJRSt
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's statement circuit, its rows' fit, and the public file
    {c : Flock.Circuit} (hcS : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c)
    (hv1 : Hidden.Any.RowsFitS c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Layout.scopeOkH c = true)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_liveSt I ht hcS (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags hRec hDraw)) τ
    -- cr/sha-512 for the knowledge finders of its simulation
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZKR I own o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJSt dj y₀ ht hcS hv1 hscope hU hpub).bindingZ (rsZHJSt_computes dj y₀ ht hcS hv1 hscope hU hpub))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJSt dj y₀ ht hcS hv1 hscope hU hpub).bindingZ (rsZHJSt_computes dj y₀ ht hcS hv1 hscope hU hpub))
          k Rw M ρ t' (reg σ) (cont σ) ∧
    ∀ o : Reg × Finset (Fin n) × Exec.Transcript, LiveAcceptsZKR I own o →
      ∀ rd : FlockSoundness.Registered.ReadsX (layShHJSt (Reg := Reg) dj y₀ ht hcS hv1 hscope hU (pub := pub))
          (rsShHJSt dj y₀ ht hcS hv1 hscope hU hpub),
        OwnReadsHJ I own c pub o.2.2 rd.isRead rd.port rd.readAt → rd.Checked o.1 (regPathsHJ I c pub o.2.2 rd.port) := by
  intro σ hCR
  refine ⟨?_, fun o hacc rd hown => checked_of_acceptsHJSt I ht own dj y₀ hcS hv1 hU hpub hscope hTags o hacc rd hown⟩
  exact le_trans (prob_mono (q := fun o => LiveAcceptsZK I o ∧ _) (fun o h => ⟨acceptsZK_of_R h.1, h.2⟩) _ _)
    (zk_session_soundHJSt I ht dj y₀ wr dg N hcS hv1 hU hpub hscope kd hkd Lb src hA3 hDraw hTags hRec hk hRw hM hρ hr
      ht' K₀ τ hCR)

end FlockSoundness.Discharge.ZkReg
