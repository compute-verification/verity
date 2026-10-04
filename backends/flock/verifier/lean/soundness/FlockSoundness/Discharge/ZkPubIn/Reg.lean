import FlockSoundness.Discharge.ZkPubIn.Bind
import FlockSoundness.Discharge.ZkPubIn.Setup
import FlockSoundness.Discharge.ZkReg.CheckedH

/-!
# The registered reads of a `verify --zk --registered R --public-inputs F` session

ZkReg's `OwnReadsH`, `regPathsH` and `checked_of_acceptsH` with the verifier's public inputs. With `own` given,
setupHidden's drawn statement is `HmIn.drawn c pi pub d` (the population's `in` words filtered to the draw), not
`HmOut.drawn c pub d`, so the registered paths are read there; the units are the population header's
(`Registered.tableUnits`), as at `OwnReadsH`.

* `acceptsZKP_of_RP`: `AcceptsZKRP I reg own` is `AcceptsZKP I own` with the registered check;
  `check_of_acceptsHP`: that check is `Registered.check` on the one table's statement.
* `OwnReadsHP`, `regPathsHP`: `OwnReadsH`, `regPathsH` at `HmIn.drawn c pi pub d`.
* `checked_of_acceptsHP`: the shadow layout's registered reads that are the verifier's own are `Checked` at every
  accepted outcome.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock FlockLevel3 Refine ZkLink ZkExec ZkBind ZkHidden HiddenExec HiddenDec DecodeZL ZkReg
open FlockSoundness.Discharge.Exec (Inputs recordDraw)

theorem acceptsZKP_of_RP {I : Inputs} {reg : Array Registered.Port} {own : ByteArray} {t : Exec.Transcript}
    (h : AcceptsZKRP I reg own t) : AcceptsZKP I own t := by
  obtain ⟨spec, sts, hs, -, hshape, hv⟩ := h
  exact ⟨spec, sts, hs, hshape, hv⟩

variable {n : ℕ} {Gate : Type}

/-- **The registered reads are the verifier's own at the record's draw, with public inputs**: `OwnReadsH`, at the
circuit with public input ports `pi` (the check reads the units from the population's header, whichever drawn statement
`pi` makes). -/
def OwnReadsHP (reg : Array Registered.Port) (c : Circuit) (_pi : HmIn.Ports) (pub : Public) (t : Exec.Transcript)
    (isRead : (Fin n × Fin c.ports.size) ⊕ Gate → Prop) (port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port)
    (readAt : (Fin n × Fin c.ports.size) ⊕ Gate → ℕ) : Prop :=
  OwnReadsH (n := n) reg c pub t isRead port readAt

/-- **The paths of the registered reads, with public inputs**: `regPathsH` at the drawn statement
`HmIn.drawn c pi pub d`. -/
def regPathsHP (c : Circuit) (pi : HmIn.Ports) (pub : Public) (t : Exec.Transcript)
    (port : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port) : (Fin n × Fin c.ports.size) ⊕ Gate → ByteArray
  | .inl (u, q) => Option.getD (do
      let d ← recordDraw t.record
      let D ← (Draw.UnitDraw.ofJson d).toOption
      let dp ← (HmIn.drawn c pi pub d).toOption
      let tb ← dp.tables
      let (_, paths) ← (Registered.entry dp (port (.inl (u, q)))).toOption
      return paths.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) ByteArray.empty)
      ByteArray.empty
  | .inr _ => ByteArray.empty

/-- **The registered check at an accepted outcome with public inputs**: the session's one statement `st` is
setupHidden's at the record's draw and `own`, and `Registered.check` accepts it against `reg` at the units
`Registered.tableUnits` reads for table 0. -/
theorem check_of_acceptsHP {I : Inputs} {reg : Array Registered.Port} {own : ByteArray} {t : Exec.Transcript}
    (hTags : TagsOkH I) (h : AcceptsZKRP I reg own t) :
    ∃ st, Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (recordDraw t.record)
        I.partition I.program (some own) = .ok st ∧
      Registered.check st.c st.pub (some reg) (Registered.derived st.partition)
        (fun _ => Registered.tableUnits I.publicFile (recordDraw t.record) I.count 0) = .ok () := by
  obtain ⟨spec, sts, hs, hreg, -, -⟩ := h
  have hh : (zkTags I).hiddenOutputs = true := (zkTags_hiddenOutputs I).trans hTags.1
  rcases setupTables_okHP hh hs with ⟨-, st, hst, -, rfl⟩ | ⟨hc', -⟩
  swap
  · exact absurd hTags.2.2.2 (by change 1 < I.count at hc'; omega)
  have hck := Registered.checkSession_ok hreg 0 (by simp)
  have e : (#[st] : Array Stmt)[0]! = st := rfl
  rw [e] at hck
  exact ⟨st, hst, hck⟩

variable {mPts : ℕ} {Reg : Type}

/-- **The registered reads are checked at every accepted outcome with public inputs** (`checked_of_acceptsH`). At an
outcome `verify --zk --registered R --public-inputs F` accepts (`reg` R's ports, `own` F's bytes), each registered read
of the shadow layout that is the verifier's own at the outcome's draw (`OwnReadsHP`) has its registered leaf open the
root `reg` holds at the position `reg` gives, through `regPathsHP`. -/
theorem checked_of_acceptsHP (I : Inputs) (ht : I.tags.typed = false) (reg : Array Registered.Port) (own : ByteArray)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKHP I own mPts S₀ (dj S₀))
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
    {pub : Public} (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub) (hscope : Layout.scopeOk c = true)
    (hTags : TagsOkH I) (o : Reg × Finset (Fin n) × Exec.Transcript) (hacc : LiveAcceptsZKRP I reg own o)
    (rd : FlockSoundness.Registered.Reads (layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub))
      (rsShHP dj y₀ ht hc hscope hU hpi hpub))
    (hown : OwnReadsHP reg c pi pub o.2.2 rd.isRead rd.port rd.readAt) :
    rd.Checked o.1 (regPathsHP c pi pub o.2.2 rd.port) := by
  intro p hp
  obtain ⟨u, q, d, D, units, rfl, hmem, hfind, hd, hD, hu, hun, hread⟩ := hown p hp
  obtain ⟨st, hst, hck⟩ := check_of_acceptsHP hTags hacc
  rw [hd] at hst hck
  obtain ⟨pi₀, s⟩ := setupHidden_specP hst
  have hcst : st.c = c := by
    have h1 := s.parse
    rw [hc] at h1
    exact (Except.ok.inj h1).symm
  have hpi₀ : pi₀ = pi := by
    have h1 := s.inParse
    rw [hcst, hpi] at h1
    exact (Except.ok.inj h1).symm
  subst hpi₀
  obtain ⟨pop', hpop', -, -, -, hdr⟩ := s.pub
  rw [hcst, hpub] at hpop'
  cases hpop'
  have hdp := hdr d rfl
  rw [hcst] at hdp
  have hdpR : HmRow.drawn { c with outPorts := #[("public inputs", pi₀.words)] } pub d = .ok st.pub := by
    have := hdp
    unfold HmIn.drawn at this
    simpa only [s.ports, Bool.false_eq_true, ↓reduceIte] using this
  have hload : HmRow.loadPublic (zkTags I) { c with outPorts := #[("public inputs", pi₀.words)] } I.publicFile =
      .ok pub := by
    simpa only [s.ports, Bool.false_eq_true, ↓reduceIte] using loadPublic_nilP hpub
  rw [hcst] at hck
  obtain ⟨tb, idx, htb, hidx, hports⟩ := check_tables hck
  have hidx' : Registered.tableUnits I.publicFile (some d) I.count 0 = .ok idx := hidx
  obtain ⟨header', units', D', hh', hun', hD', rfl⟩ := Registered.tableUnits_one hTags.2.2.2 hidx'
  rw [loadPublic_header hload] at hh'
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
  have hpath : regPathsHP c pi₀ pub o.2.2 rd.port (.inl (u, q)) =
      paths.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) ByteArray.empty := by
    simp [regPathsHP, hd, hD, hdp, htb, hent, Except.toOption]
  have hleaf : ((layShHP (Reg := Reg) dj y₀ ht hc hscope hU hpi (pub := pub)).regLeaf o.1 (.inl (u, q))).1 =
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
    have h1 := Hidden.drawn_digestW hload hdpR hD (Layout.proved_lt_population hD) hi q.isLt
    rw [hget] at h1
    rw [← h1]
    simp only [Public.digest, htb, Registered.tableRow, Registered.ref]
  show Registered.opensLeaf (rd.port (.inl (u, q))) _ (rd.readAt (.inl (u, q))) _ = true
  rw [hleaf, hra, hpath, hdig]
  exact hopens

end FlockSoundness.Discharge.ZkPubIn
