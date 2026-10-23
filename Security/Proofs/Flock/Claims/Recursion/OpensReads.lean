import Proofs.Flock.Claims.Recursion.RecRecord
import Proofs.Flock.Claims.Recursion.DeployHonest
import Proofs.Flock.C3.VStarList.Accepts
import Proofs.Flock.C3.PubRows.Check
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Shared
import Proofs.Flock.Soundness.Discharge.Layout.SharedRows
import Proofs.Flock.Soundness.Discharge.Layout.QSort

/-!
# What a staged part's transcript shows is the record the verifier read in its own format

`PartOpensB` at `cmtTr recRead` asks the commit strings a part holds to be those its V\* statement's transcript shows,
read by `recRead`: the coin server's record in its own format (`Flock.RecRecord.readFull`). Every staged part carries
the coin server's transcript (`insO_staged_link`), and its recursive setup holds it at the verifier's inner shape
(`RecSetup.commits`), which reads it in that same format and holds what it reads there to what its JSON readers read
(`RecSetup.checkReads`, `FlockC3.VStarList.checkReads_ok`). So at a staged part (`staged_reads`) the rounds
`cmtTr recRead` reads are the record's (`Record.msgs`), and the record's link, rounds, tops and `c_k` are the JSON's the
setup pins: Hello's table, `root_B`, the link points, the schedule's rounds, and each rep's salted tops and `c_k`.

This is the transcript's half of `PartOpensB`. The other half is that a part's hm96 slot outputs are the public rows
its statement pins, and those rows are the JSON's (`RecVStar.checkPubRows`, `RecVStar.checkTopRows`). Its first steps
are here: the setup checks every table's rows and registered reads (`RecIns.setup_pubRows`,
`RecIns.setup_registered`), which give each registered port's rows as `fixedRow`'s at the positions its instances read
(`rowsAt`), and a drawn table's slot holds its instance's public `b ‖ c` (`bc_setupOne`), a table row
(`digest_tables`).
-/

namespace RecDeploy

open Flock
open Claims (Msg Out)
open Lean (Json)

theorem byteArray_toList_loop (b : ByteArray) (i : ℕ) (r : List UInt8) :
    ByteArray.toList.loop b i r = r.reverse ++ b.data.toList.drop i := by
  fun_induction ByteArray.toList.loop b i r with
  | case1 i r h ih =>
    rw [ih, List.reverse_cons, List.append_assoc, List.singleton_append]
    congr 1
    conv_rhs => rw [List.drop_eq_getElem_cons (by simpa using h)]
    simp [ByteArray.get!, h]
  | case2 i r h =>
    rw [List.drop_eq_nil_of_le (by simp; simpa using h), List.append_nil]

theorem byteArray_toList_eq (b : ByteArray) : b.toList = b.data.toList := by
  unfold ByteArray.toList
  rw [byteArray_toList_loop]
  simp

theorem toUTF8_of_fromUTF8? {b : ByteArray} {s : String} (h : String.fromUTF8? b = some s) : s.toUTF8 = b := by
  unfold String.fromUTF8? at h
  split at h
  · cases h; rfl
  · cases h

/-- **The commits the reader reads** are `commitsOf` at the statement's source, binding and inner shape. -/
theorem RecRaw.ins_commits {w : RecRaw} {s : RecIns} (h : w.ins = .ok s) :
    commitsOf w s.src s.bound s.inner = .ok s.commits := by
  unfold RecRaw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; assumption)

theorem commits_parse_error {sh : RecVStar.Inner} {tt : String} {e : String} {r : Except String (Option String)}
    {bound : Flock.V (Option String × Option RecVStar.StackedShape)} {record : String} {cs : RecVStar.InnerCommits}
    (hp : Json.parse tt = .error e) : RecSetup.commits (some sh) (.ok tt) r bound record ≠ .ok cs := by
  intro h
  unfold RecSetup.commits at h
  simp only [bind, Except.bind, pure, Except.pure, hp] at h
  repeat' split at h
  all_goals cases h

theorem commits_text_error {sh : RecVStar.Inner} {e : String} {r : Except String (Option String)}
    {bound : Flock.V (Option String × Option RecVStar.StackedShape)} {record : String} {cs : RecVStar.InnerCommits} :
    RecSetup.commits (some sh) (.error e) r bound record ≠ .ok cs := by
  intro h
  unfold RecSetup.commits at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals cases h

/-- **A transcript's commits at the verifier's inner shape read it in the coin server's own format**: where
`commitsOf` holds the transcript `t`, it is UTF-8 and JSON, the source has a record, and `checkReads` passed. -/
theorem commitsOf_reads {w : RecRaw} {src : Option RecCoins.Source}
    {bound : Flock.V (Option String × Option RecVStar.StackedShape)} {sh : RecVStar.Inner}
    {c : Option RecVStar.InnerCommits} {t : ByteArray} (ht : w.transcript = some t)
    (h : commitsOf w src bound (some sh) = .ok c) :
    ∃ cs tt tr record, c = some cs ∧ String.fromUTF8? t = some tt ∧ Json.parse tt = .ok tr ∧
      src.map (·.record) = some record ∧ RecSetup.checkReads sh tt tr record cs = .ok () := by
  unfold commitsOf at h
  rw [ht] at h
  cases hsrc : src.map (·.record) with
  | none => rw [hsrc] at h; cases h
  | some record =>
    rw [hsrc] at h
    dsimp only at h
    split at h
    · cases h
    cases hX : RecSetup.commits (some sh) (textOf "rec inner transcript" t)
        (w.registration.mapM (textOf "rec inner registration")) bound record with
    | error e => rw [hX] at h; cases h
    | ok cs =>
      rw [hX] at h
      cases h
      cases htx : String.fromUTF8? t with
      | none =>
        rw [textOf, htx] at hX
        exact absurd hX commits_text_error
      | some tt =>
        rw [textOf, htx] at hX
        cases hp : Json.parse tt with
        | error e => exact absurd hX (commits_parse_error hp)
        | ok tr => exact ⟨cs, tt, tr, record, rfl, rfl, hp, rfl, FlockC3.VStarList.reads_of_commits hp hX⟩

theorem ins_of_insO {x : Msg} {i : RecIns} (h : insO x = some i) : ∃ w, decodeRec x = some w ∧ w.ins = .ok i := by
  unfold insO at h
  obtain ⟨w, hw, h⟩ := Option.bind_eq_some_iff.1 h
  split_ifs at h
  · cases h
  exact ⟨w, hw, toOption_eq_some h⟩

/-- **A staged part's transcript, as the verifier read it in its own format**: at a part staged at `x`, the recursive
setup holds the coin server's transcript at the verifier's inner shape `sh` (`i.commits = some cs`), `checkReads`
passed on it, and the rounds `recRead` reads off the V\* statement are the record's (`Record.msgs`), the record
`readFull` reads. So `checkReads_ok` gives that record's link, rounds, tops and `c_k` as the JSON's. -/
theorem staged_reads {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) :
    ∃ (i : RecIns) (sh : RecVStar.Inner) (cs : RecVStar.InnerCommits) (tt : String) (tr : Json) (record : String)
      (r : Flock.RecRecord.Record),
      insO xO = some i ∧ i.inner = some sh ∧ i.commits = some cs ∧
      RecSetup.checkReads sh tt tr record cs = .ok () ∧ Flock.RecRecord.readFull tt.toUTF8.toList = some r ∧
      shownOf recRead x xO = some (r.msgs.map fun mc => (mc.1, mc.2.toArray)) := by
  obtain ⟨ms, i, _spec, _sts, rfl, hsu, -⟩ := hS
  obtain ⟨hins, -⟩ := setupO_some hsu
  obtain ⟨xr, -, hinner, -⟩ := insO_staged hins
  obtain ⟨wr, hd, htr, -, -⟩ := insO_staged_link hins
  obtain ⟨wr', hd', hw⟩ := ins_of_insO hins
  rw [hd] at hd'
  cases hd'
  have hc := RecRaw.ins_commits hw
  rw [hinner] at hc
  obtain ⟨cs, tt, tr, record, hcs, htx, -, -, hR⟩ := commitsOf_reads htr hc
  obtain ⟨_lk, r, _want, -, hr, -⟩ := FlockC3.VStarList.checkReads_ok hR
  refine ⟨i, xr.shape, cs, tt, tr, record, r, hins, hinner, hcs, hR, hr, ?_⟩
  have hb : (st.transcript x ms).data.toList = tt.toUTF8.toList := by
    rw [byteArray_toList_eq, toUTF8_of_fromUTF8? htx]
  simp only [shownOf, hd, htr, Option.bind_some, recRead, hb, hr, Option.map_some]

/-! ## What the setup pins of a part's rows -/

theorem RecIns.setup_pubRows {s : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts)) :
    sts.forM (RecVStar.checkPubRows · s.base.registered s.commits (s.inner.bind (·.sparse))) = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; assumption)

theorem forIn_ok_of {α : Type} {xs : Array α} {f : α → Flock.V Unit}
    {b : α → PUnit → Flock.V (ForInStep PUnit)} {u : PUnit}
    (hb : ∀ a v r, b a v = .ok r → f a = .ok () ∧ r = .yield PUnit.unit) (h : forIn xs PUnit.unit b = .ok u) : ∀ a ∈ xs, f a = .ok () := by
  rcases xs with ⟨l⟩
  simp only [List.forIn_toArray] at h
  intro a ha
  simp only [List.mem_toArray] at ha
  induction l generalizing u with
  | nil => cases ha
  | cons c l ih =>
    simp only [List.forIn_cons, bind, Except.bind] at h
    cases hc : b c PUnit.unit with
    | error e => rw [hc] at h; cases h
    | ok r =>
      rw [hc] at h
      rcases List.mem_cons.1 ha with rfl | ha
      · exact (hb _ _ _ hc).1
      · rw [(hb _ _ _ hc).2] at h
        exact ih h ha

/-- The recursive setup checks each table's registered reads (`checkRegistered`). -/
theorem RecIns.setup_registered {s : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (h : s.setup = .ok (spec, sts)) : ∀ st ∈ sts, checkRegistered st s.base.registered = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    refine forIn_ok_of ?_ (by assumption)
    intro a v r hr
    try dsimp only at hr
    split at hr
    · cases hr
    · rename_i u hu
      split at hr
      · cases hr
      · cases hr
        exact ⟨hu, rfl⟩

/-- A shared-row file's `b ‖ c` of instance `i`'s row `p` is its table row at the instance's ref. -/
theorem digest_tables {pub : Public} {t : Tables} (ht : pub.tables = some t) (i p np : ℕ) :
    pub.digest i p np = Registered.tableRow pub t p (Registered.ref t np p i) := by
  unfold Public.digest Registered.tableRow Registered.ref
  rw [ht]

/-- **A registered port's rows at a checked table**: where the table passed `checkPubRows` and `checkRegistered`
against the verifier's own reads `own`, a port `reg` of `own` registering a value `fixedRow` knows has a table `t`, a
circuit port `p` and an entry's positions, with every instance `i`'s ref a row of that table read at the position its
unit reads, and every row the value's `fixedRow` row at its position. -/
theorem rowsAt {st : Stmt} {own : Array Registered.Port} {cs : RecVStar.InnerCommits}
    {sparse : Option RecSparse.Shape} (hP : RecVStar.checkPubRows st (some own) (some cs) sparse = .ok ())
    (hR : checkRegistered st (some own) = .ok ()) {reg : Registered.Port} (hmem : reg ∈ own)
    (hfix : (RecVStar.fixedRow { reps := #[] } reg.value 0).isSome = true) :
    ∃ t idx p positions, st.pub.tables = some t ∧ Registered.unitIndices st.pub = .ok idx ∧
      (List.range st.c.ports.size).find? (fun p => st.c.ports[p]!.name == reg.name) = some p ∧
      positions.size = t.counts.getD p 0 ∧
      (∀ i < st.pub.n, Registered.ref t st.c.ports.size p i < positions.size ∧
        reg.reads[idx.getD i 0]? = some (some (positions.getD (Registered.ref t st.c.ports.size p i) 0))) ∧
      ∀ r < positions.size, ∃ w c, RecVStar.fixedRow cs reg.value (positions.getD r 0)
        ((RecSparse.params st.c.unit.name).or sparse) = some (.ok (w, c)) ∧ Registered.tableRow st.pub t p r = c := by
  obtain ⟨t, ht, p, positions, paths, hp, hent, hrows⟩ := FlockC3.PubRows.checkPubRows_rows hP hmem hfix
  obtain ⟨t', idx, ht', hidx, hports⟩ :=
    FlockSoundness.Discharge.ZkReg.check_tables (FlockSoundness.Discharge.ZkReg.checkRegistered_ok hR)
  rw [ht] at ht'
  cases ht'
  obtain ⟨p', positions', paths', hp', hent', hall⟩ := FlockSoundness.Discharge.ZkReg.checkPort_at (hports _ hmem)
  obtain ⟨p'', positions'', paths'', hp'', hent'', hsize⟩ := FlockC3.PubRows.checkPort_size (hports _ hmem)
  rw [hp] at hp' hp''
  cases hp'
  cases hp''
  rw [hent] at hent' hent''
  cases hent'
  cases hent''
  refine ⟨t, idx, p, positions, ht, hidx, hp, hsize, fun i hi => ?_, hrows⟩
  obtain ⟨hlt, hreads, -⟩ := hall i hi
  exact ⟨hsize ▸ hlt, hreads⟩

theorem regionsOf_mem (st : Stmt) :
    ∀ r ∈ st.regions.toList,
      FlockSoundness.Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ FlockSoundness.Refine.regionsOf st :=
  fun _ hr => List.mem_map_of_mem hr

open FlockSoundness.Discharge in
/-- **A drawn table's hm96 slot holds its instance's public `b ‖ c`**: at a statement `setupOne` sets up at a draw, in
either branch, a witness satisfying the table's model (at the regions the soundness proof reads) holds at instance
`o · g + g`'s port-`p` slot the statement's `b ‖ c` of that instance's row `p` (`Layout.bc_drawnS`,
`Hidden.bc_drawnW`, with `Hidden.drawn_digestW`). -/
theorem bc_setupOne {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {d : Json} {partition : Option Json} {program : Option ByteArray} {st : Stmt}
    (h : HiddenExec.setupOne tags cf pf coins tables (some d) partition program = .ok st)
    (L : FlockSoundness.Layout st) {z : FlockSoundness.Model.Witness (L.model (FlockSoundness.Refine.regionsOf st)).m}
    (hz : (L.model (FlockSoundness.Refine.regionsOf st)).Satisfies z)
    (o : Fin (2 ^ (L.model (FlockSoundness.Refine.regionsOf st)).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hg2 : g < 2 ^ st.c.g.log2) (hp : p < st.c.ports.size) (hr : o.val * st.c.g + g < st.pub.n) :
    FlockSoundness.Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) =>
        (L.model (FlockSoundness.Refine.regionsOf st)).addr o
          (Layout.inBlockOr st.c.kLog (Layout.bcAddr st.c g p j.val t.val))) =
      (st.pub.digest (o.val * st.c.g + g) p st.c.ports.size).data.toList := by
  unfold HiddenExec.setupOne at h
  split at h
  · have s := Hidden.setupHidden_spec' h
    obtain ⟨pop, hpop, -, hdr⟩ := s.pub
    have hdrawn : HmRow.drawn { st.c with outPorts := #[] } pop d = .ok st.pub := hdr d rfl
    obtain ⟨D, hD, -, hn, -⟩ := Layout.drawn_spec hdrawn
    have hlt := Layout.proved_lt_population hD
    have hr' : o.val * st.c.g + g < D.proved.size := by omega
    rw [Hidden.drawn_digestW (Hidden.loadPublic_nil hpop) hdrawn hD hlt hr' hp]
    exact Hidden.bc_drawnW h hpop hD hlt L _ (regionsOf_mem st) hz o hg hg2 hp hr'
  · obtain ⟨pop, hpop, hdr⟩ := Layout.setupH_drawn h
    obtain ⟨D, hD, -, hn, -⟩ := Layout.drawn_spec hdr
    have hlt := Layout.proved_lt_population hD
    have hr' : o.val * st.c.g + g < D.proved.size := by omega
    rw [Hidden.drawn_digestW hpop hdr hD hlt hr' hp]
    exact Layout.bc_drawnS h hpop hD hlt L _ (regionsOf_mem st) hz o hg hg2 hp hr'
