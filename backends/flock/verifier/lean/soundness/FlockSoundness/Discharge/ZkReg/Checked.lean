import FlockSoundness.Discharge.ZkReg.Bind
import FlockSoundness.Discharge.ZkReg.Defs
import FlockSoundness.Discharge.ZkExec.Setup
import FlockSoundness.Discharge.DecodeZL.Statements

/-!
# The registered check at an accepted `--zk --registered` outcome

`checked_of_accepts`: at an outcome `verify --zk --registered F` accepts, the shadow layout's registered reads that are
the verifier's own at the outcome's draw (`OwnReads`) are `Checked` at the outcome's registration, through `regPaths`.
The one table's statement is setupH's at the record's draw `d`, over `HmRow.drawn c pub d` (`setupH_drawn`), and the
session's registered check is `Registered.check` on it at the units `Registered.tableUnits` reads (`check_of_accepts`):
the population header's units at the draw's proved units (`FlockProofs.Registered.tableUnits_one`). It opens each drawn
instance's row of each of `own`'s ports at the position `own` gives for its unit, through the path the header gives
(`check_tables`, `checkPort_at`: `FlockProofs.Registered.check_ok` keeping the positions and paths). The drawn
instance at `u`'s rank holds the population's row of `u` (`Hidden.drawn_digestW`), whose leaf is the layout's
registered leaf at `(u, q)`.
-/

namespace FlockSoundness.Discharge.ZkReg

open Flock FlockLevel3 Refine ZkLink ZkExec ZkBind Integrate DecodeZL
open FlockSoundness.Discharge.Exec (Inputs recordDraw)

/-- `AcceptsZKR` is `AcceptsZK` with the registered check: it implies `AcceptsZK`. -/
theorem acceptsZK_of_R {I : Inputs} {own : Array Registered.Port} {t : Exec.Transcript} (h : AcceptsZKR I own t) :
    AcceptsZK I t := by
  obtain ⟨spec, sts, hs, -, hshape, hv⟩ := h
  exact ⟨spec, sts, hs, hshape, hv⟩

/-- **The registered check at an accepted outcome**: the session's one statement `st` is setupH's at the record's draw,
and `Registered.check` accepts it against `own` at the units `Registered.tableUnits` reads for table 0. -/
theorem check_of_accepts {I : Inputs} {own : Array Registered.Port} {t : Exec.Transcript} (hTags : TagsOk I)
    (h : AcceptsZKR I own t) :
    ∃ st, Stmt.setupH (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (recordDraw t.record) I.partition
        I.program = .ok st ∧
      Registered.check st.c st.pub (some own) (Registered.derived st.partition)
        (fun _ => Registered.tableUnits I.publicFile (recordDraw t.record) I.count 0) = .ok () := by
  obtain ⟨spec, sts, hs, hreg, -, -⟩ := h
  have hh' : (zkTags I).hiddenOutputs = false := (zkTags_hiddenOutputs I).trans hTags.1
  rcases Exec.setupTables_ok hh' hs with ⟨-, st, hst, -, rfl⟩ | ⟨hc', -⟩
  swap
  · exact absurd hTags.2.2.2 (by change 1 < I.count at hc'; omega)
  have hck := Registered.checkSession_ok hreg 0 (by simp)
  have e : (#[st] : Array Stmt)[0]! = st := by first | rfl | simp
  rw [e] at hck
  exact ⟨st, hst, hck⟩

/-- `Registered.check` accepting with the verifier's own `own`: the public file has tables `t`, the units are `idx`, and
`checkPort` accepts each of `own`'s ports. -/
theorem check_tables {c : Circuit} {pub : Public} {own : Array Registered.Port} {derived : Bool}
    {units : Unit → V (Array Nat)} (h : Registered.check c pub (some own) derived units = .ok ()) :
    ∃ t idx, pub.tables = some t ∧ units () = .ok idx ∧
      ∀ reg ∈ own, Registered.checkPort c pub t idx reg = .ok () := by
  unfold Registered.check at h
  simp only at h
  split at h
  · exact absurd (Registered.except_bind_ok h) fun ⟨_, h1, _⟩ => Registered.throw_ne_ok _ _ h1
  · split at h
    · obtain ⟨_, _, h⟩ := Registered.except_bind_ok h
      split at h
      · rename_i t ht
        obtain ⟨indices, hidx, h⟩ := Registered.except_bind_ok h
        exact ⟨t, indices, ht, hidx, fun reg hreg => Registered.forM_ok h reg (by simpa using hreg)⟩
      · exact absurd h (Registered.throw_ne_ok _ _)
    · exact absurd h (Registered.throw_ne_ok _ _)

/-- **`checkPort` accepting, with its positions and paths** (`FlockProofs.Registered.checkPort_ok`, keeping them): the
port is the circuit's first port `p` of its name, the header's entry gives `positions` and `paths`, and every instance's
ref names a row of its table, which opens the registered root at that row's position through that row's path, the
position `reg` gives for the instance's unit. -/
theorem checkPort_at {c : Circuit} {pub : Public} {t : Tables} {indices : Array Nat} {reg : Registered.Port}
    (h : Registered.checkPort c pub t indices reg = .ok ()) :
    ∃ p positions paths, (List.range c.ports.size).find? (fun p => c.ports[p]!.name == reg.name) = some p ∧
      Registered.entry pub reg = .ok (positions, paths) ∧ ∀ i < pub.n,
        Registered.ref t c.ports.size p i < t.counts.getD p 0 ∧
        reg.reads[indices.getD i 0]? = some (some (positions.getD (Registered.ref t c.ports.size p i) 0)) ∧
        Registered.opens reg (Registered.tableRow pub t p (Registered.ref t c.ports.size p i))
          (positions.getD (Registered.ref t c.ports.size p i) 0)
          (paths.getD (Registered.ref t c.ports.size p i) ByteArray.empty) = true := by
  unfold Registered.checkPort at h
  simp only at h
  split at h
  · rename_i p hp
    split at h
    · exact absurd (Registered.except_bind_ok h) fun ⟨_, h1, _⟩ => Registered.throw_ne_ok _ _ h1
    split at h
    · exact absurd (Registered.except_bind_ok h) fun ⟨_, h1, _⟩ => Registered.throw_ne_ok _ _ h1
    · obtain ⟨⟨positions, paths⟩, hent, h⟩ := Registered.except_bind_ok h
      simp only at h
      split at h
      · exact absurd (Registered.except_bind_ok h) fun ⟨_, h1, _⟩ => Registered.throw_ne_ok _ _ h1
      · split at h
        · exact absurd h (Registered.throw_ne_ok _ _)
        · rename_i hrow
          split at h
          · rename_i hinst
            refine ⟨p, positions, paths, hp, hent, fun i hi => ?_⟩
            have hi' := List.find?_eq_none.mp hinst i (List.mem_range.mpr hi)
            simp only [Bool.not_eq_true', Bool.and_eq_false_iff, decide_eq_false_iff_not, not_or,
              Classical.not_not] at hi'
            obtain ⟨hr, hreads⟩ := hi'
            have ho := List.find?_eq_none.mp hrow _ (List.mem_range.mpr hr)
            exact ⟨hr, hreads, by simpa using ho⟩
          · split at h
            · exact absurd (Registered.except_bind_ok h) fun ⟨_, h1, _⟩ => Registered.throw_ne_ok _ _ h1
            · split at h <;> exact absurd h (Registered.throw_ne_ok _ _)
  · exact absurd h (Registered.throw_ne_ok _ _)

set_option maxHeartbeats 2000000 in
/-- **The population's header is the public file's first line**: `HmRow.loadPublic` keeps the header it parses, which
is `Registered.publicHeader`'s. -/
theorem loadPublic_header {tags : Flock.Tags} {c : Circuit} {file : ByteArray} {pop : Public}
    (h : HmRow.loadPublic tags c file = .ok pop) : Registered.publicHeader file = .ok pop.header := by
  unfold HmRow.loadPublic at h
  unfold Registered.publicHeader
  cases hf : findByte file 10 with
  | none => simp only [hf] at h; exact absurd h (Refine.throw_ne_ok _ _)
  | some nl =>
    simp only [hf] at h ⊢
    cases hs : String.fromUTF8? (file.extract 0 nl) with
    | none => simp only [hs, throw_bind'] at h; exact absurd h (Refine.throw_ne_ok _ _)
    | some s =>
      simp only [hs, pure_bind] at h ⊢
      cases hp : Lean.Json.parse s with
      | error e => simp only [hp, throw_bind'] at h; exact absurd h (Refine.throw_ne_ok _ _)
      | ok j =>
        simp only [hp, pure_bind] at h ⊢
        dsimp only at h
        simp only [throw_bind'] at h
        repeat' walk_step
        all_goals
          simp only [pure, Except.pure, Except.ok.injEq] at h
          subst h
          rfl

/-- The drawn statement's unit at `u`'s rank among the draw's proved units is the population's unit of `u`. -/
theorem drawnUnit {proved units : Array ℕ} {u : ℕ} (hu : u ∈ proved) :
    (proved.map (units.getD · 0)).getD (proved.idxOf u) 0 = units.getD u 0 := by
  have hi : proved.idxOf u < proved.size := Array.idxOf_lt_length_of_mem hu
  have hget : proved[proved.idxOf u]'hi = u := by
    have := Array.findIdx_getElem (xs := proved) (p := (· == u)) (w := hi)
    simp only [beq_iff_eq] at this
    exact this
  simp only [Array.getD, Array.size_map, hi, ↓reduceDIte, Array.getElem_map, hget]

variable {mPts n : ℕ} {Reg : Type}

/-- **The registered reads are checked at every accepted outcome.** At an outcome `verify --zk --registered F`
accepts (`own` F's ports), each registered read of the shadow layout that is the verifier's own at the outcome's draw
(`OwnReads`) has its registered leaf open the root `own` holds at the position `own` gives, through `regPaths`. -/
theorem checked_of_accepts (I : Inputs) (ht : I.tags.typed = false) (own : Array Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hscope : Layout.scopeOk c = true)
    (hTags : TagsOk I) (o : Reg × Finset (Fin n) × Exec.Transcript) (hacc : LiveAcceptsZKR I own o)
    (rd : FlockSoundness.Registered.Reads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub))
      (rowsSh ht dj y₀ hc hU hpub hscope))
    (hown : OwnReads own c pub o.2.2 rd.isRead rd.port rd.readAt) :
    rd.Checked o.1 (regPaths c pub o.2.2 rd.port) := by
  intro p hp
  obtain ⟨u, q, d, D, units, rfl, hmem, hfind, hd, hD, hu, hun, hread⟩ := hown p hp
  obtain ⟨st, hst, hck⟩ := check_of_accepts hTags hacc
  rw [hd] at hst hck
  have hcst : st.c = c := by
    have h1 := (setupH_spec ((zkTags_typed I).trans ht) hst).1
    rw [hc] at h1
    exact (Except.ok.inj h1).symm
  obtain ⟨pop', hpop', hdr⟩ := Layout.setupH_drawn hst
  rw [hcst, hpub] at hpop'
  cases hpop'
  rw [hcst] at hdr hck
  obtain ⟨tb, idx, htb, hidx, hports⟩ := check_tables hck
  have hidx' : Registered.tableUnits I.publicFile (some d) I.count 0 = .ok idx := hidx
  obtain ⟨header', units', D', hh', hun', hD', rfl⟩ := Registered.tableUnits_one hTags.2.2.2 hidx'
  rw [loadPublic_header hpub] at hh'
  cases hh'
  rw [hun] at hun'
  cases hun'
  rw [hD] at hD'
  cases hD'
  obtain ⟨p', positions, paths, hp', hent, hall⟩ := checkPort_at (hports _ hmem)
  rw [hfind] at hp'
  cases hp'
  obtain ⟨D', hD', -, hn, -, -⟩ := Layout.drawn_spec hdr
  rw [hD] at hD'
  cases hD'
  have hi : D.proved.idxOf u.val < D.proved.size := Array.idxOf_lt_length_of_mem hu
  obtain ⟨-, hreads, hopens⟩ := hall _ (by rw [hn]; exact hi)
  rw [drawnUnit hu, hread] at hreads
  have hra : rd.readAt (.inl (u, q)) =
      positions.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) 0 := by
    simpa using hreads
  have hpath : regPaths c pub o.2.2 rd.port (.inl (u, q)) =
      paths.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) ByteArray.empty := by
    simp [regPaths, hd, hD, hdr, htb, hent, Except.toOption]
  have hleaf : ((laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).regLeaf o.1 (.inl (u, q))).1 =
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
    have h1 := Hidden.drawn_digestW hpub hdr hD (Layout.proved_lt_population hD) hi q.isLt
    rw [hget] at h1
    rw [← h1]
    simp only [Public.digest, htb, Registered.tableRow, Registered.ref]
  show Registered.opensLeaf (rd.port (.inl (u, q))) _ (rd.readAt (.inl (u, q))) _ = true
  rw [hleaf, hra, hpath, hdig]
  exact hopens

end FlockSoundness.Discharge.ZkReg
