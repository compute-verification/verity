import FlockSoundness.Discharge.ZkReg.BindH
import FlockSoundness.Discharge.ZkReg.DefsH
import FlockSoundness.Discharge.ZkReg.Checked
import FlockSoundness.Discharge.HiddenDec.Statements

/-!
# The registered check at an accepted hidden-output `--zk --registered` outcome

`checked_of_acceptsH`: `checked_of_accepts` at hidden outputs. The one table's statement is setupHidden's at the
record's draw `d` (`HiddenExec.setupTables_okH`), over `HmOut.drawn c pub d` (`Hidden.setupHidden_spec'`), which is
`HmRow.drawn` at the circuit without its output words, and the session's registered check is `Registered.check` on it
at the units `Registered.tableUnits` reads (`check_of_acceptsH`), so the rest is `checked_of_accepts`' proof: the drawn
instance at `u`'s rank holds the population's row of `u` (`Hidden.drawn_digestW`, from `Hidden.loadPublic_nil`), whose
leaf is the hidden layout's registered leaf at `(u, q)`.
-/

namespace FlockSoundness.Discharge.ZkReg

open Flock FlockLevel3 Refine ZkLink ZkExec ZkBind ZkHidden HiddenExec HiddenDec DecodeZL
open FlockSoundness.Discharge.Exec (Inputs recordDraw)

/-- **The registered check at an accepted hidden-output outcome**: the session's one statement `st` is setupHidden's at
the record's draw, and `Registered.check` accepts it against `own` at the units `Registered.tableUnits` reads for
table 0. -/
theorem check_of_acceptsH {I : Inputs} {own : Array Registered.Port} {t : Exec.Transcript} (hTags : TagsOkH I)
    (h : AcceptsZKR I own t) :
    ∃ st, Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (recordDraw t.record) I.partition
        I.program = .ok st ∧
      Registered.check st.c st.pub (some own) (Registered.derived st.partition)
        (fun _ => Registered.tableUnits I.publicFile (recordDraw t.record) I.count 0) = .ok () := by
  obtain ⟨spec, sts, hs, hreg, -, -⟩ := h
  have hh : (zkTags I).hiddenOutputs = true := (zkTags_hiddenOutputs I).trans hTags.1
  rcases setupTables_okH hh hs with ⟨-, st, hst, -, rfl⟩ | ⟨hc', -⟩
  swap
  · exact absurd hTags.2.2.2 (by change 1 < I.count at hc'; omega)
  have hck := Registered.checkSession_ok hreg 0 (by simp)
  have e : (#[st] : Array Stmt)[0]! = st := rfl
  rw [e] at hck
  exact ⟨st, hst, hck⟩

variable {mPts n : ℕ} {Reg : Type}

/-- **The registered reads are checked at every accepted hidden-output outcome.** At an outcome
`verify --zk --registered F` accepts over hidden outputs (`own` F's ports), each registered read of the hidden shadow
layout that is the verifier's own at the outcome's draw (`OwnReadsH`) has its registered leaf open the root `own` holds
at the position `own` gives, through `regPathsH`. -/
theorem checked_of_acceptsH (I : Inputs) (ht : I.tags.typed = false) (own : Array Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hscope : Layout.scopeOk c = true)
    (hTags : TagsOkH I) (o : Reg × Finset (Fin n) × Exec.Transcript) (hacc : LiveAcceptsZKR I own o)
    (rd : FlockSoundness.Registered.Reads (layShH (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHS dj y₀ ht hc hscope hU hpub))
    (hown : OwnReadsH own c pub o.2.2 rd.isRead rd.port rd.readAt) :
    rd.Checked o.1 (regPathsH c pub o.2.2 rd.port) := by
  intro p hp
  obtain ⟨u, q, d, D, units, rfl, hmem, hfind, hd, hD, hu, hun, hread⟩ := hown p hp
  obtain ⟨st, hst, hck⟩ := check_of_acceptsH hTags hacc
  rw [hd] at hst hck
  have s := Hidden.setupHidden_spec' hst
  have hcst : st.c = c := by
    have h1 := s.parse
    rw [hc] at h1
    exact (Except.ok.inj h1).symm
  obtain ⟨pop', hpop', -, hdr⟩ := s.pub
  rw [hcst, hpub] at hpop'
  cases hpop'
  have hdp := hdr d rfl
  rw [hcst] at hdp
  have hdpR : HmRow.drawn { c with outPorts := #[] } pub d = .ok st.pub := hdp
  rw [hcst] at hck
  obtain ⟨tb, idx, htb, hidx, hports⟩ := check_tables hck
  have hidx' : Registered.tableUnits I.publicFile (some d) I.count 0 = .ok idx := hidx
  obtain ⟨header', units', D', hh', hun', hD', rfl⟩ := Registered.tableUnits_one hTags.2.2.2 hidx'
  rw [loadPublic_header (Hidden.loadPublic_nil hpub)] at hh'
  cases hh'
  rw [hun] at hun'
  cases hun'
  rw [hD] at hD'
  cases hD'
  obtain ⟨p', positions, paths, hp', hent, hall⟩ := checkPort_at (hports _ hmem)
  rw [hfind] at hp'
  cases hp'
  obtain ⟨D', hD', -, hn, -, -⟩ := Layout.drawn_spec hdpR
  rw [hD] at hD'
  cases hD'
  have hi : D.proved.idxOf u.val < D.proved.size := Array.idxOf_lt_length_of_mem hu
  obtain ⟨-, hreads, hopens⟩ := hall _ (by rw [hn]; exact hi)
  rw [drawnUnit hu, hread] at hreads
  have hra : rd.readAt (.inl (u, q)) =
      positions.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) 0 := by
    simpa using hreads
  have hpath : regPathsH c pub o.2.2 rd.port (.inl (u, q)) =
      paths.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) ByteArray.empty := by
    simp [regPathsH, hd, hD, hdp, htb, hent, Except.toOption]
  have hleaf : ((layShH (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub)).regLeaf o.1 (.inl (u, q))).1 =
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
      Registered.tableRow st.pub tb q.val (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) := by
    have h1 := Hidden.drawn_digestW (Hidden.loadPublic_nil hpub) hdpR hD (Layout.proved_lt_population hD) hi q.isLt
    rw [hget] at h1
    rw [← h1]
    simp only [Public.digest, htb, Registered.tableRow, Registered.ref]
  show Registered.opensLeaf (rd.port (.inl (u, q))) _ (rd.readAt (.inl (u, q))) _ = true
  rw [hleaf, hra, hpath, hdig]
  exact hopens

end FlockSoundness.Discharge.ZkReg
