import Proofs.Flock.Soundness.Discharge.ZkReg.BindHJ
import Proofs.Flock.Soundness.Discharge.ZkReg.DefsHJ
import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedH

/-!
# The registered check at an accepted hidden-output `--zk --registered` outcome of `J` tables

`checked_of_acceptsHJ`: `checked_of_acceptsH` at `J` tables. `AcceptsZKR` runs `checkRegistered` on every table's
statement, and table `j`'s is setupHidden's under its tags at its part of the record's draw, `drawJ I d j`
(`ZkHidden.setupOf_tableJH`, from `setupTables_partsH`), over `HmOut.drawn c pub (drawJ I d j)`. A read `(u, q)` is at
the table `u` is drawn into (`OwnReadsHJ`), so that table's check is `checked_of_acceptsH`'s at its draw: the drawn
instance at `u`'s rank in the table's draw holds the population's row of `u` (`Hidden.drawn_digestW`), whose leaf is the
hidden layout's registered leaf at `(u, q)`, which reads the population alone, whatever the table.
-/

namespace FlockSoundness.Discharge.ZkReg

open Flock FlockLevel3 Refine ZkLink ZkExec ZkBind ZkHidden HiddenExec HiddenDec DecodeZL Composed
open FlockSoundness.Discharge.Exec (Inputs recordDraw)

variable {mPts n : ℕ} {Reg : Type}

/-- **The registered reads are checked at every accepted hidden-output outcome of `J` tables.** At an outcome
`verify --zk --registered F` accepts over hidden outputs (`own` F's ports), each registered read of the hidden shadow
layout of `J` tables that is the verifier's own at the draw of the table its unit is drawn into (`OwnReadsHJ`) has its
registered leaf open the root `own` holds at the position `own` gives, through `regPathsHJ`. -/
theorem checked_of_acceptsHJ (I : Inputs) (ht : I.tags.typed = false) (own : Array Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hscope : Layout.scopeOkH c = true)
    (hTags : TagsOkZHJ I) (o : Reg × Finset (Fin n) × Exec.Transcript) (hacc : LiveAcceptsZKR I own o)
    (rd : FlockSoundness.Registered.ReadsX (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHJS dj y₀ ht hc hscope hU hpub))
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
    have h1 := s.parse
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
  have hleaf : ((layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub)).regLeaf o.1 (.inl (u, q))).1 =
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

end FlockSoundness.Discharge.ZkReg
