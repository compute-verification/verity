import Proofs.Flock.Claims.KS.JointJ
import Proofs.Flock.Claims.KS.TableBound

/-!
# Every table of the masked record

`JointJ` at the deployed masked (`--zk`) record, whose scope admits several tables. Table `j`'s word is its level-0
codeword (`wordLenJ`), the words together no longer than `N₀` (`sum_wordLenJ_le`), and its columns have its level-0
witness lanes (`wideJ`). An accepted run's columns at table `j` are read at the statement's table `j` (`acc_colsJ`), so
they have its width (`zk_colsOkJ`); and each is the witness lanes of an opened row whose level-0 opening against table
`j`'s `root_B` in the link passed (`zk_openingJ`, `zk_opening` at table `j`), so two accepted runs at one tape that
open one position of table `j` to two columns collide (`zk_conflictTiedJ`). The statement parses and its relation
splits by table wherever some run accepts (`zk_parses`, `zk_splits`). So `Flock.KS` at the masked record follows from
every table's bound alone (`zk_KS_of_tableBoundJ`), and that bound at tables with unopened positions from the bound at
full-width tables and `ExtendJ` (`zk_tableBoundJ_of_extend`).
-/

namespace FlockDeploy.KS

open Flock
open FlockSoundness.Refine (ebind_ok entries_eq_of_flat flat_data merkleCheck_cap)

local notation "𝔽" => FlockLevel3.GF128

/-! ## Table `j`'s word -/

/-- **Table `j`'s word's length**: its level-0 codeword length `2^d`; `0` past the statement's tables. -/
def wordLenJ (zk : Bool) (x : Claims.Msg) (j : ℕ) : ℕ :=
  match (tables zk x)[j]? with
  | some (_, st) => 2 ^ (if zk then Claims.Flock.zkLevel0 st.m else Claims.Flock.plainLevel0 st.m).d
  | none => 0

namespace Gen

variable [S : ZkSetup]

/-- **Table `j`'s word's length**: its level-0 codeword length `2^d`; `0` past the statement's tables. -/
def wordLenJ (zk : Bool) (x : Claims.Msg) (j : ℕ) : ℕ :=
  match (tables zk x)[j]? with
  | some (_, st) => 2 ^ (if zk then Claims.Flock.zkLevel0 st.m else Claims.Flock.plainLevel0 st.m).d
  | none => 0

end Gen

/-- **A column of table `j`'s code's width**: table `j`'s level-0 witness lanes (under `--zk`, without the extra
lanes). -/
def wideJ (zk : Bool) (x : Claims.Msg) (j : ℕ) (u : List 𝔽) : Prop :=
  match (tables zk x)[j]? with
  | some (_, st) => u.length = 2 ^ (if zk then Claims.Flock.zkLevel0 st.m else Claims.Flock.plainLevel0 st.m).lanes
  | none => False

namespace Gen

variable [S : ZkSetup]

/-- **A column of table `j`'s code's width**: table `j`'s level-0 witness lanes (under `--zk`, without the extra
lanes). -/
def wideJ (zk : Bool) (x : Claims.Msg) (j : ℕ) (u : List 𝔽) : Prop :=
  match (tables zk x)[j]? with
  | some (_, st) => u.length = 2 ^ (if zk then Claims.Flock.zkLevel0 st.m else Claims.Flock.plainLevel0 st.m).lanes
  | none => False

end Gen

theorem sum_range_getD_le {α : Type} (g : α → ℕ) : ∀ (l : List α) (n : ℕ),
    ∑ j ∈ Finset.range n, (l[j]?.map g).getD 0 ≤ (l.map g).sum
  | [], n => by simp
  | a :: l, 0 => by simp
  | a :: l, n + 1 => by
    rw [Finset.sum_range_succ']
    simp only [List.getElem?_cons_succ, List.getElem?_cons_zero, Option.map_some, Option.getD_some, List.map_cons,
      List.sum_cons]
    have := sum_range_getD_le g l n
    omega

namespace Gen

variable [S : ZkSetup]

/-- **The words together are no longer than `N₀`**, the tables' level-0 codeword lengths summed. -/
theorem sum_wordLenJ_le (zk : Bool) (x : Claims.Msg) (n : ℕ) :
    ((∑ j : Fin n, wordLenJ zk x j : ℕ) : ℝ) ≤ N₀ zk x := by
  set g : SessionSpec × Stmt → ℕ :=
    fun p => 2 ^ (if zk then Claims.Flock.zkLevel0 p.2.m else Claims.Flock.plainLevel0 p.2.m).d with hg
  have hw : ∀ j, wordLenJ zk x j = ((tables zk x)[j]?.map g).getD 0 := by
    intro j
    unfold wordLenJ
    cases (tables zk x)[j]? <;> rfl
  have hN : N₀ zk x = (((tables zk x).map g).sum : ℕ) := by
    unfold N₀
    rw [Nat.cast_list_sum, List.map_map]
    congr 1
    refine List.map_congr_left fun p _ => ?_
    simp [hg, tableLen]
  rw [hN, Fin.sum_univ_eq_sum_range (fun j => wordLenJ zk x j) n]
  simp only [hw]
  exact_mod_cast sum_range_getD_le g (tables zk x) n

end Gen

/-- **The words together are no longer than `N₀`**, the tables' level-0 codeword lengths summed. -/
theorem sum_wordLenJ_le (zk : Bool) (x : Claims.Msg) (n : ℕ) :
    ((∑ j : Fin n, wordLenJ zk x j : ℕ) : ℝ) ≤ N₀ zk x :=
  Gen.sum_wordLenJ_le (S := S₀) (zk := zk) (x := x) (n := n)

namespace Gen

variable [S : ZkSetup]

theorem wideJ_ne_zero (zk : Bool) (x : Claims.Msg) (j : ℕ) (u : List 𝔽) (_ : 0 < wordLenJ zk x j)
    (h : wideJ zk x j u) : u ≠ 0 := by
  unfold wideJ at h
  rintro rfl
  split at h
  · exact absurd h.symm (pow_ne_zero _ two_ne_zero)
  · exact h

end Gen

theorem wideJ_ne_zero (zk : Bool) (x : Claims.Msg) (j : ℕ) (u : List 𝔽) (_ : 0 < wordLenJ zk x j)
    (h : wideJ zk x j u) : u ≠ 0 :=
  Gen.wideJ_ne_zero (S := S₀) (zk := zk) (x := x) (j := j) (u := u) ‹_› (h := h)

/-! ## `ColsOkJ` -/

namespace Gen

variable [S : ZkSetup]

theorem tables_getElem? {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}
    {j : ℕ} (hs : setupOf zk x = some (s, spec, sts)) (hst : sts[j]? = some st) :
    (tables zk x)[j]? = some (spec, st) := by
  unfold tables
  rw [hs, List.getElem?_map, Array.getElem?_toList, hst]
  rfl

end Gen

theorem tables_getElem? {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}
    {j : ℕ} (hs : setupOf zk x = some (s, spec, sts)) (hst : sts[j]? = some st) :
    (tables zk x)[j]? = some (spec, st) :=
  Gen.tables_getElem? (S := S₀) (zk := zk) (x := x) (s := s) (spec := spec) (sts := sts) (st := st) (j := j) (hs := hs)
    (hst := hst)

namespace Gen

variable [S : ZkSetup]

/-- **An accepted run opens columns of the reader's table `j`**: each has the statement's table `j`'s level-0
width. -/
theorem acc_colsJ {zk : Bool} {H : Claims.Msg → Claims.Msg} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg}
    {ω : Ω} {r : Fin ((V zk).k x) → Coin} (h : (Claims.run H P (V zk) x ω r).out = .acc) {j q : ℕ} {u : List 𝔽}
    (hu : Claims.Flock.lookCol (cols zk (Claims.run H P (V zk) x ω r).seen j) q = some u) :
    ∃ spec st, (tables zk x)[j]? = some (spec, st) ∧
      u.length = 2 ^ (if zk then Claims.Flock.zkLevel0 st.m else Claims.Flock.plainLevel0 st.m).lanes := by
  obtain ⟨s, spec, sts, f, hs, hf, hx⟩ := acc_last h
  unfold cols at hu
  split at hu
  · rename_i rr hrr
    obtain ⟨f', s', spec', sts', st, hf', hs', hst, hm⟩ := runOf_m hrr
    rw [hf] at hf'
    cases hf'
    rw [hx, hs] at hs'
    cases hs'
    refine ⟨spec, st, tables_getElem? hs hst, ?_⟩
    obtain ⟨p, hp, rfl⟩ := exists_of_lookCol hu
    cases zk
    · simp only [Bool.false_eq_true, ↓reduceIte] at hp ⊢
      rw [plainRunCols_width hp, hm]
    · simp only [↓reduceIte] at hp ⊢
      rw [zkRunCols_width hp, hm]
  · simp [Claims.Flock.lookCol] at hu

end Gen

/-- **An accepted run opens columns of the reader's table `j`**: each has the statement's table `j`'s level-0
width. -/
theorem acc_colsJ {zk : Bool} {H : Claims.Msg → Claims.Msg} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg}
    {ω : Ω} {r : Fin ((V zk).k x) → Coin} (h : (Claims.run H P (V zk) x ω r).out = .acc) {j q : ℕ} {u : List 𝔽}
    (hu : Claims.Flock.lookCol (cols zk (Claims.run H P (V zk) x ω r).seen j) q = some u) :
    ∃ spec st, (tables zk x)[j]? = some (spec, st) ∧
      u.length = 2 ^ (if zk then Claims.Flock.zkLevel0 st.m else Claims.Flock.plainLevel0 st.m).lanes :=
  Gen.acc_colsJ (S := S₀) (zk := zk) (H := H) (Ω := Ω) (P := P) (x := x) (ω := ω) (r := r) (h := h) (j := j) (q := q)
    (u := u) (hu := hu)

namespace Gen

variable [S : ZkSetup]

/-- **`ColsOkJ` at the masked record**: an accepted run opens only columns of table `j`'s width at table `j`. -/
theorem zk_colsOkJ (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ColsOkJ H (deployZk rel) (wordLenJ true) (wideJ true) := by
  intro x Ω P ω r j q u hacc _ _ hu
  obtain ⟨spec, st, hh, hl⟩ := acc_colsJ (zk := true) hacc hu
  unfold wideJ
  rw [hh]
  exact hl

end Gen

/-- **`ColsOkJ` at the masked record**: an accepted run opens only columns of table `j`'s width at table `j`. -/
theorem zk_colsOkJ (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ColsOkJ H (deployZk rel) (wordLenJ true) (wideJ true) :=
  Gen.zk_colsOkJ (S := S₀) (H := H) (rel := rel)

/-! ## `ConflictTiedJ` -/

namespace Gen

variable [S : ZkSetup]

/-- **The reader's table `j` at a decided run**: what `runOf` takes, read off the decision's final message, setup
and live session. -/
theorem runOf_eqJ {zk : Bool} {seen : Claims.Seen} {f : Final} {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {lv : Live} {rr : Claims.Flock.ZkRun} {j : ℕ} (hf : lastFinal seen = some f)
    (hx : f.stmt = x) (hs : setupOf zk x = some (s, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hr : runOf zk seen j = some rr) :
    ∃ st extras, sts[j]? = some st ∧ (Setup.ofCircuit st).extra lv.points = .ok extras ∧
      rr = { m := (Setup.ofCircuit st).m, kLog := (Setup.ofCircuit st).kLog, claims := 2 + extras.size,
             hashLen := (Setup.ofCircuit st).merkle.digestLen, saltLen := saltLenOf zk (Setup.ofCircuit st),
             reps := [(lv.streams.getD (2 * j) #[], f.proofs.getD (2 * j) ByteArray.empty),
                      (lv.streams.getD (2 * j + 1) #[], f.proofs.getD (2 * j + 1) ByteArray.empty)] } := by
  unfold runOf at hr
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hr
  obtain ⟨f', hf', ⟨s', spec', sts'⟩, hs', st, hst, lv', hlv', extras, hext, rfl⟩ := hr
  rw [hf] at hf'
  cases hf'
  rw [hx, hs] at hs'
  cases hs'
  rw [hlv] at hlv'
  cases hlv'
  refine ⟨st, extras, hst, ?_, rfl⟩
  cases he : (Setup.ofCircuit st).extra lv.points with
  | error e => rw [he] at hext; cases hext
  | ok v => rw [he] at hext; cases hext; rfl

end Gen

/-- **The reader's table `j` at a decided run**: what `runOf` takes, read off the decision's final message, setup
and live session. -/
theorem runOf_eqJ {zk : Bool} {seen : Claims.Seen} {f : Final} {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {lv : Live} {rr : Claims.Flock.ZkRun} {j : ℕ} (hf : lastFinal seen = some f)
    (hx : f.stmt = x) (hs : setupOf zk x = some (s, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hr : runOf zk seen j = some rr) :
    ∃ st extras, sts[j]? = some st ∧ (Setup.ofCircuit st).extra lv.points = .ok extras ∧
      rr = { m := (Setup.ofCircuit st).m, kLog := (Setup.ofCircuit st).kLog, claims := 2 + extras.size,
             hashLen := (Setup.ofCircuit st).merkle.digestLen, saltLen := saltLenOf zk (Setup.ofCircuit st),
             reps := [(lv.streams.getD (2 * j) #[], f.proofs.getD (2 * j) ByteArray.empty),
                      (lv.streams.getD (2 * j + 1) #[], f.proofs.getD (2 * j + 1) ByteArray.empty)] } :=
  Gen.runOf_eqJ (S := S₀) (zk := zk) (seen := seen) (f := f) (x := x) (s := s) (spec := spec) (sts := sts) (lv := lv)
    (rr := rr) (j := j) (hf := hf) (hx := hx) (hs := hs) (hlv := hlv) (hr := hr)

namespace Gen

variable [S : ZkSetup]

/-- **An accepted masked run opens a column of table `j` only as a Merkle opening** of table `j`'s level 0 in its
own log: at the statement's setup, table `j`'s `--zk` schedule and the link of the tape's first message, a column `u`
the run's reader opens at `q` is the witness lanes of an opened row whose opening against `root_B[j]` of the link
passed, every SHA-512 input of it in the run's verifier log. -/
theorem zk_openingJ {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {r : Fin ((V true).k x) → Coin}
    {j q : ℕ} {u : List 𝔽} (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (hu : Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) q = some u) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ)
      (link : List ByteArray × List ByteArray) (row : Array F128),
      setupOf true x = some (s, spec, sts) ∧ sts[j]? = some st ∧ Zk.schedule st.m = .ok (sch, tPad) ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧
      Opening MerkleScheme.hm96Sha512 (sch.levels.getD 0 default)
        (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) link.1.toArray[j]!
        (Claims.run sha512 P (V true) x ω r).vLog q row ∧
      u = (row.toList.map FlockLevel3.GF128.of).take (2 ^ (sch.levels.getD 0 default).lanes) := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hs, hf, hlink, hc, hv⟩ := acc_within h
  generalize hL : (Claims.run sha512 P (V true) x ω r).vLog = L at hv ⊢
  obtain ⟨hx, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checks_ok hc
  obtain ⟨hroot, -⟩ := Claims.Bridge.Custody.custody_ok hcu
  unfold cols at hu
  split at hu
  swap
  · simp [Claims.Flock.lookCol] at hu
  rename_i rr hrr
  obtain ⟨st, extras, hst, hext, rfl⟩ := runOf_eqJ hf hx hs hlv hrr
  simp only [↓reduceIte] at hu
  have hmem := lookCol_mem hu
  obtain ⟨ρ, hρ, hp⟩ := List.mem_flatMap.1 (mem_of_distinctBelow hmem)
  obtain ⟨k, hk, rfl⟩ : ∃ k < 2,
      ρ = (lv.streams.getD (2 * j + k) #[], f.proofs.getD (2 * j + k) ByteArray.empty) := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hρ
    rcases hρ with rfl | rfl
    · exact ⟨0, by omega, rfl⟩
    · exact ⟨1, by omega, rfl⟩
  have hstsj : j < sts.size := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, -⟩
    exact h0
  have hstj : sts[j] = st := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, e⟩
    exact e
  have hj : j < (tabsOf sts).size := by simpa [tabsOf] using hstsj
  have htj : (tabsOf sts)[j] = Setup.ofCircuit st := by rw [tabsOf_getElem hj]; simp only [hstj]
  replace hv : Zk.verifyWithM (hashesIn (L.map Prod.fst)) spec (tabsOf sts) f.record f.proofs = .ok () := hv
  obtain ⟨sess, hdec, hsz, hreps⟩ :=
    FlockSoundness.ZkQuery.zkQueryRound_verifyWithM (hashesIn (L.map Prod.fst)) spec (tabsOf sts) f.record f.proofs hv
  obtain ⟨sch, tPad, hsch, hrep⟩ := hreps j hj
  obtain ⟨stream, extras', zp, hstream, hext', hzp, hlt, hcols, hmc, hcap⟩ := hrep k hk
  simp only [htj] at hsch hext' hzp hlt hcols hmc
  have hdec' : Record.decode spec f.record = .ok sess := by
    rw [← With.decodeWith_std]
    exact (With.decodeWith_sim sub (sub_hashes (L.map Prod.fst)) spec f.record) sess hdec
  obtain ⟨hrB, hpts, hlsz, hstreams⟩ := Claims.Bridge.Custody.decode_live hdec' hlv
  rw [hpts, hext] at hext'
  cases hext'
  have htab : j < spec.tables.size := by rw [← hsz]; exact hj
  obtain ⟨hk', hrs⟩ := Claims.Bridge.Custody.repStreams_getElem spec htab hk
  obtain ⟨y, hy, hfa⟩ := hstreams (2 * j + k) hk'
  rw [hrs] at hy
  have hys : y = stream := Option.some.inj (hy.symm.trans hstream)
  subst hys
  have hcoins := forall₂_coins (fun _ _ hab => hab.2.1) hfa
  have hcolsEq : Claims.Flock.zkRepColsOf
      (Claims.Flock.zkProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
        (Setup.ofCircuit st).merkle.digestLen (saltLenOf true (Setup.ofCircuit st)))
      (lv.streams.getD (2 * j + k) #[]) (f.proofs.getD (2 * j + k) ByteArray.empty) =
      (Claims.Flock.zkRepCols (sch.levels.getD 0 default)
        (y.rounds[Claims.Flock.zkQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0
          (2 + extras.size)]!).coins zp.proof.ligerito.initialProof.rows).map
        (fun p => (p.1, p.2.take (2 ^ (sch.levels.getD 0 default).lanes))) := by
    rw [← hcols]
    have e1 : lv.streams.getD (2 * j + k) #[] = lv.streams[2 * j + k]! := by
      rw [show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]
    have e2 : f.proofs.getD (2 * j + k) ByteArray.empty = f.proofs[2 * j + k]! := by
      rw [show ByteArray.empty = default from rfl, getD_eq_getElem!]
    rw [e1, e2]
    refine zkRepColsOf_congr _ _ fun i => ?_
    rw [← Array.getElem?_toList, ← Array.getElem?_toList]
    exact (hcoins i).symm
  have hp2 : (q, u) ∈ (Claims.Flock.zkRepCols (sch.levels.getD 0 default)
      (y.rounds[Claims.Flock.zkQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0
        (2 + extras.size)]!).coins zp.proof.ligerito.initialProof.rows).map
      (fun p => (p.1, p.2.take (2 ^ (sch.levels.getD 0 default).lanes))) := by
    rw [← hcolsEq]
    exact hp
  obtain ⟨⟨q', w⟩, hw, hqw⟩ := List.mem_map.1 hp2
  simp only [Prod.mk.injEq] at hqw
  obtain ⟨rfl, rfl⟩ := hqw
  obtain ⟨i, hi, -, hiq, hiu⟩ := mem_zkRepCols hw
  have hms : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 := by
    have := checks_hm96 hc (setupOf_some hs).2 (setupOf_zk hs) j hstsj
    rw [hstj] at this
    exact this
  rw [hms] at hmc
  have hruns := With.merkleCheckWith_sim (back (L.map Prod.fst))
    (With.opsWith_sim (back (L.map Prod.fst)) (back_hashes (L.map Prod.fst)) MerkleScheme.hm96Sha512)
    MerkleScheme.hm96Sha512 _ _ _ _ _ _ _ _ () hmc
  rw [opsWith_hm96] at hruns
  obtain ⟨hpure, hlog⟩ := merkleCheck_logs hruns
  obtain ⟨hcs, hcw⟩ := merkleCheck_cap hpure
  refine ⟨s, spec, sts, st, sch, tPad, link, zp.proof.ligerito.initialProof.rows[i]!, hs, hst, hsch, hlink,
    ⟨Or.inl rfl, zp.proof.cap, zp.proof.ligerito.initialProof.salts.getD i ByteArray.empty,
      sibsOf zp.proof.ligerito.initialProof.siblings
        ((sch.levels.getD 0 default).d - (sch.levels.getD 0 default).cap) i,
      hcs, hcw, ?_, ?_, merkleCheck_sibs _ _ _ _ _ _ _ _ _ hpure i, ?_⟩, by rw [hiu]⟩
  · rw [hcap, hrB, ← hroot, Array.toArray_toList]
  · rw [← hiq]
    exact merkleCheck_opens _ _ _ _ _ _ _ _ _ hpure i hi
  · rw [opensInsOf_hm96, ← hiq]
    exact hlog i hi

end Gen

/-- **An accepted masked run opens a column of table `j` only as a Merkle opening** of table `j`'s level 0 in its
own log: at the statement's setup, table `j`'s `--zk` schedule and the link of the tape's first message, a column `u`
the run's reader opens at `q` is the witness lanes of an opened row whose opening against `root_B[j]` of the link
passed, every SHA-512 input of it in the run's verifier log. -/
theorem zk_openingJ {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {r : Fin ((V true).k x) → Coin}
    {j q : ℕ} {u : List 𝔽} (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (hu : Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) q = some u) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ)
      (link : List ByteArray × List ByteArray) (row : Array F128),
      setupOf true x = some (s, spec, sts) ∧ sts[j]? = some st ∧ Zk.schedule st.m = .ok (sch, tPad) ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧
      Opening MerkleScheme.hm96Sha512 (sch.levels.getD 0 default)
        (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) link.1.toArray[j]!
        (Claims.run sha512 P (V true) x ω r).vLog q row ∧
      u = (row.toList.map FlockLevel3.GF128.of).take (2 ^ (sch.levels.getD 0 default).lanes) :=
  Gen.zk_openingJ (S := S₀) (Ω := Ω) (P := P) (x := x) (ω := ω) (r := r) (j := j) (q := q) (u := u) (h := h) (hu := hu)

namespace Gen

variable [S : ZkSetup]

/-- **`ConflictTiedJ` at the deployed masked (`--zk`) record**: two accepted runs at one tape that open different
columns at one position of one table collide in their joined verifier logs, at SHA-512. -/
theorem zk_conflictTiedJ (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ConflictTiedJ sha512 (deployZk rel) (wordLenJ true) := by
  intro x Ω P ω r r' j q u v hacc hacc' _ _ hu hv huv
  obtain ⟨s, spec, sts, st, sch, tPad, link, row, hs, hst, hsch, hlink, ho, hu'⟩ :=
    zk_openingJ (P := P) (r := r) hacc hu
  obtain ⟨s', spec', sts', st', sch', tPad', link', row', hs', hst', hsch', hlink', ho', hv'⟩ :=
    zk_openingJ (P := P) (r := r') hacc' hv
  rw [hs] at hs'
  cases hs'
  rw [hst] at hst'
  cases hst'
  rw [hsch] at hsch'
  cases hsch'
  rw [hlink] at hlink'
  cases hlink'
  exact tiedColl_of_openings ho ho' fun e => huv (by rw [hu', hv', e])

end Gen

/-- **`ConflictTiedJ` at the deployed masked (`--zk`) record**: two accepted runs at one tape that open different
columns at one position of one table collide in their joined verifier logs, at SHA-512. -/
theorem zk_conflictTiedJ (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ConflictTiedJ sha512 (deployZk rel) (wordLenJ true) :=
  Gen.zk_conflictTiedJ (S := S₀) (rel := rel)

/-! ## KS from every table's bound -/

namespace Gen

variable [S : ZkSetup]

/-- **The masked statement parses wherever some run accepts.** -/
theorem zk_parses (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    (∃ s, (deployZk rel).parse x = some s) ∨ Claims.Flock.NoAcc sha512 (deployZk rel).V x := by
  by_cases hno : Claims.Flock.NoAcc sha512 (V true) x
  · exact .inr hno
  · obtain ⟨s, hs, -⟩ := acc_inScope rel hno
    exact .inl ⟨s, hs⟩

end Gen

/-- **The masked statement parses wherever some run accepts.** -/
theorem zk_parses (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) :
    (∃ s, (deployZk rel).parse x = some s) ∨ Claims.Flock.NoAcc sha512 (deployZk rel).V x :=
  Gen.zk_parses (S := S₀) (rel := rel) (x := x)

namespace Gen

variable [S : ZkSetup]

/-- **The masked relation splits by table wherever some run accepts**: the statement is in scope there
(`acc_inScope`, `splits_of_inScope`). -/
theorem zk_splits (x : Claims.Msg) :
    Claims.Flock.Splits (deployZk Readable.relation) x ∨
      Claims.Flock.NoAcc sha512 (deployZk Readable.relation).V x := by
  by_cases hno : Claims.Flock.NoAcc sha512 (V true) x
  · exact .inr hno
  · obtain ⟨s, hs, hin⟩ := acc_inScope Readable.relation hno
    exact .inl (splits_of_inScope Claims.Flock.zkReadD (deployZk Readable.relation) rfl rfl rfl hs hin)

end Gen

/-- **The masked relation splits by table wherever some run accepts**: the statement is in scope there
(`acc_inScope`, `splits_of_inScope`). -/
theorem zk_splits (x : Claims.Msg) :
    Claims.Flock.Splits (deployZk Readable.relation) x ∨
      Claims.Flock.NoAcc sha512 (deployZk Readable.relation).V x :=
  Gen.zk_splits (S := S₀) (x := x)

/-- **The deployed masked (`--zk`) verifier: KS**, at SHA-512, from every table's bound alone, at every number of
tables. -/
theorem zk_KS_of_tableBoundJ
    (htab : Claims.Flock.TableBoundJ sha512 (deployZk Readable.relation) (wordLenJ true)
      (Claims.Flock.orZeroJ (wideJ true))) :
    Claims.Flock.KS sha512 (deployZk Readable.relation) :=
  zk_of_oneWait_sha512 Readable.relation zk_splits
    (Claims.Flock.oneWait_of_tableBoundJ sha512 (deployZk Readable.relation) (wordLenJ true)
      (fun x => sum_wordLenJ_le true x _) (fun x => εtable_nonneg true x) (V_k_pos true) (k_one_zk _)
      (zk_parses _) _ (fun _ _ => .inl rfl) (Claims.Flock.colsOkJ_orZero _ _ _ _ (zk_colsOkJ sha512 _))
      (zk_conflictTiedJ _) htab)

/-- **Every masked table's bound at tables with unopened positions**, from the bound at full-width tables and
`ExtendJ`. -/
theorem zk_tableBoundJ_of_extend (rel : Readable.Parsed → Readable.Relation)
    (hext : Claims.Flock.ExtendJ (deployZk rel) (wordLenJ true) (wideJ true))
    (htab : Claims.Flock.TableBoundJ sha512 (deployZk rel) (wordLenJ true) (wideJ true)) :
    Claims.Flock.TableBoundJ sha512 (deployZk rel) (wordLenJ true) (Claims.Flock.orZeroJ (wideJ true)) :=
  Claims.Flock.tableBoundJ_of_extend _ _ _ _ (wideJ_ne_zero true) (zk_colsOkJ sha512 rel) hext htab

end FlockDeploy.KS
