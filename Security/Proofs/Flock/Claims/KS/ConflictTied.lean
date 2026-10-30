import Proofs.Flock.Claims.KS.ConflictRun
import Proofs.Flock.Claims.DeployK

/-!
# Conflicts tied to collisions, at both records

`ConflictTied` (`KS.Joint`) at the deployed plain and masked records, at SHA-512. An accepted run opens a column at
`q` only where level 0's Merkle check of table 0 passed on that column's row at `q` (`plain_opening`,
`zk_opening`): its verdict accepts with SHA-512 answered only within the run's verifier log (`acc_within`), so the
query round's lemmas under every oracle place the reader's columns among the verifier's openings
(`plainQueryRound_verifyWithM`, `ZkQuery.zkQueryRound_verifyWithM`), the record's rep streams being the decoded
session's (`decode_live`); the check, carried back to `OC` (`back`), logged every SHA-512 input of the opening
(`merkleCheck_logs`); and the check's cap flattens to table 0's `root_B` in the link message (custody). An `Opening`
records that.

Two runs at one tape send one link message (`run_head`), so their openings at `q` climb to one cap: with different
rows they collide in the two runs' joined logs (`tiedColl_of_openings`, `tiedColl_of_opens`), which is
`ConflictTied` (`plain_conflictTied`, `zk_conflictTied`). That leaves `TableBoundθ` for KS at SHA-512
(`plain_KS_of_table`, `zk_KS_of_table`).

The plain opening holds of every column a rep opens, before `distinctBelow` keeps one per position (`plainRawCols`,
`plain_opening_raw`), so two such columns of one accepted run at one position that differ collide in its own log
(`plain_selfClash_tied`).
-/

namespace FlockDeploy.KS

open Flock
open FlockSoundness.Refine (ebind_ok entries_eq_of_flat flat_data merkleCheck_cap)

local notation "𝔽" => FlockLevel3.GF128

/-! ## An opening, and two of one position -/

/-- **An opening of position `q` to `row`** under `ms` (`hm96-sha512/v1`, or plain SHA-512 leaves) at level `lv`
against a cap that flattens to `root`, its siblings digests, with every SHA-512 input of the opening a query of `L`. -/
def Opening (ms : MerkleScheme) (lv : Flock.Level) (lanes : ℕ) (root : ByteArray) (L : List (Claims.Msg × Claims.Msg))
    (q : ℕ) (row : Array F128) : Prop :=
  (ms = MerkleScheme.hm96Sha512 ∨ ms = MerkleScheme.sha512 "flock-leaf/sha512-unsalted") ∧
  ∃ (cap : Array ByteArray) (salt : ByteArray) (sibs : List ByteArray),
    cap.size = 2 ^ lv.cap ∧ (∀ b ∈ cap.toList, b.size = 64) ∧ cap.foldl (· ++ ·) ByteArray.empty = root ∧
      opens ms cap lv.cap lv.d lanes q row salt sibs = true ∧ (∀ b ∈ sibs, b.size = 64) ∧
      ∀ b ∈ opensInsOf ms row salt q sibs, Claims.toMsg b ∈ L.map Prod.fst

/-- **Two openings of one position to different rows, against one root, collide in the joined logs.** -/
theorem tiedColl_of_openings {ms : MerkleScheme} {lv : Flock.Level} {lanes : ℕ} {root : ByteArray}
    {L₁ L₂ : List (Claims.Msg × Claims.Msg)} {q : ℕ} {row₁ row₂ : Array F128}
    (h₁ : Opening ms lv lanes root L₁ q row₁) (h₂ : Opening ms lv lanes root L₂ q row₂)
    (hr : row₁ ≠ row₂) : Claims.TiedColl sha512 (L₁ ++ L₂) := by
  obtain ⟨hms, cap₁, y₁, s₁, hc₁, hw₁, hf₁, ho₁, hs₁, hl₁⟩ := h₁
  obtain ⟨-, cap₂, y₂, s₂, hc₂, hw₂, hf₂, ho₂, hs₂, hl₂⟩ := h₂
  have hcap : cap₁ = cap₂ := by
    apply Array.ext'
    refine entries_eq_of_flat _ _ hw₁ hw₂ (by simp [hc₁, hc₂]) ?_
    have e := congrArg (fun b : ByteArray => b.data.toList) (hf₁.trans hf₂.symm)
    simp only [← Array.foldl_toList, flat_data] at e
    simpa using e
  subst hcap
  have hl₁' : ∀ b ∈ opensInsOf ms row₁ y₁ q s₁, Claims.toMsg b ∈ (L₁ ++ L₂).map Prod.fst :=
    fun b hb => by rw [List.map_append]; exact List.mem_append_left _ (hl₁ b hb)
  have hl₂' : ∀ b ∈ opensInsOf ms row₂ y₂ q s₂, Claims.toMsg b ∈ (L₁ ++ L₂).map Prod.fst :=
    fun b hb => by rw [List.map_append]; exact List.mem_append_right _ (hl₂ b hb)
  rcases hms with rfl | rfl
  · rw [opensInsOf_hm96] at hl₁' hl₂'
    exact tiedColl_of_opens hs₁ hs₂ ho₁ ho₂ hr hl₁' hl₂'
  · rw [opensInsOf_plain] at hl₁' hl₂'
    exact tiedColl_of_opensP hs₁ hs₂ ho₁ ho₂ hr hl₁' hl₂'

/-! ## The reader's columns, at the verifier's -/

theorem lookCol_mem {os : List (ℕ × List 𝔽)} {q : ℕ} {u : List 𝔽} (h : Claims.Flock.lookCol os q = some u) :
    (q, u) ∈ os := by
  unfold Claims.Flock.lookCol at h
  obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 h
  have h1 := List.find?_some hp
  simp only [decide_eq_true_eq] at h1
  rw [← h1]
  exact List.mem_of_find?_eq_some hp

/-- A column of one rep is an opened row at one of its positions. -/
theorem mem_zkRepCols {lv : Flock.Level} {coins : Array F128} {rows : Array (Array F128)} {q : ℕ} {u : List 𝔽}
    (h : (q, u) ∈ Claims.Flock.zkRepCols lv coins rows) :
    ∃ j < (positions lv coins).size, j < rows.size ∧ (positions lv coins)[j]! = q ∧
      u = rows[j]!.toList.map FlockLevel3.GF128.of := by
  unfold Claims.Flock.zkRepCols at h
  obtain ⟨⟨a, b⟩, hm, he⟩ := List.mem_map.1 h
  simp only [Prod.mk.injEq] at he
  obtain ⟨rfl, rfl⟩ := he
  obtain ⟨j, hj, hje⟩ := List.mem_iff_getElem.1 hm
  rw [List.getElem_zip] at hje
  simp only [Prod.mk.injEq] at hje
  obtain ⟨ha, hb⟩ := hje
  rw [List.length_zip] at hj
  have hj₁ : j < (positions lv coins).size := by simpa using (lt_min_iff.1 hj).1
  have hj₂ : j < rows.size := by
    have := (lt_min_iff.1 hj).2
    simp only [List.length_take, Array.length_toList] at this
    omega
  refine ⟨j, hj₁, hj₂, ?_, ?_⟩
  · rw [← ha, getElem!_pos _ j hj₁]
    simp
  · rw [← hb, getElem!_pos _ j hj₂]
    simp

theorem forall₂_coins {R : Flock.Round → Flock.Round → Prop} (hR : ∀ a b, R a b → a.coins = b.coins) :
    ∀ {l₁ l₂ : List Flock.Round}, List.Forall₂ R l₁ l₂ → ∀ j : ℕ,
      (l₁[j]?).map (fun rd : Flock.Round => rd.coins) = (l₂[j]?).map (fun rd : Flock.Round => rd.coins)
  | [], [], _, _ => rfl
  | a :: l₁, b :: l₂, .cons hab h, j => by
    cases j with
    | zero => simp [hR a b hab]
    | succ j => simpa using forall₂_coins hR h j

theorem plainRepColsOf_congr (sh : Shape) {a b : Array Flock.Round} (bytes : ByteArray)
    (h : ∀ j : ℕ, a[j]?.map (fun rd : Flock.Round => rd.coins) = b[j]?.map (fun rd : Flock.Round => rd.coins)) :
    Claims.Flock.plainRepColsOf sh a bytes = Claims.Flock.plainRepColsOf sh b bytes := by
  unfold Claims.Flock.plainRepColsOf
  rw [h]

theorem zkRepColsOf_congr (sh : Shape) {a b : Array Flock.Round} (bytes : ByteArray)
    (h : ∀ j : ℕ, a[j]?.map (fun rd : Flock.Round => rd.coins) = b[j]?.map (fun rd : Flock.Round => rd.coins)) :
    Claims.Flock.zkRepColsOf sh a bytes = Claims.Flock.zkRepColsOf sh b bytes := by
  unfold Claims.Flock.zkRepColsOf
  rw [h]

theorem getD_eq_getElem! {α : Type} [Inhabited α] (xs : Array α) (k : ℕ) : xs.getD k default = xs[k]! := by
  unfold Array.getD
  split <;> simp_all

theorem tabsOf_getElem {sts : Array Stmt} {j : ℕ} (h : j < (tabsOf sts).size) :
    (tabsOf sts)[j] = Setup.ofCircuit (sts[j]'(by simpa [tabsOf] using h)) := by
  simp [tabsOf]

theorem tabsOf_fold (sts : Array Stmt) : ∀ st ∈ tabsOf sts,
    (∀ a b lw rho ts v, st.fold.folded a b lw rho ts = .ok v → rho.size + 6 = st.fold.kLog) ∧
      st.fold.kLog = st.kLog := by
  intro st hst
  obtain ⟨st₀, -, rfl⟩ := Array.mem_map.1 hst
  exact ⟨fun _ _ _ _ _ _ h => Stmt.folded_size st₀ h, rfl⟩

/-- **The reader's table at a decided run**: what `runOf` takes, read off the decision's final message, setup and
live session. -/
theorem runOf_eq {zk : Bool} {seen : Claims.Seen} {f : Final} {x : Claims.Msg} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {lv : Live} {rr : Claims.Flock.ZkRun} (hf : lastFinal seen = some f) (hx : f.stmt = x)
    (hs : setupOf zk x = some (s, spec, sts)) (hlv : liveOf spec f.record = .ok lv) (hr : runOf zk seen 0 = some rr) :
    ∃ st extras, sts[0]? = some st ∧ (Setup.ofCircuit st).extra lv.points = .ok extras ∧
      rr = { m := (Setup.ofCircuit st).m, kLog := (Setup.ofCircuit st).kLog, claims := 2 + extras.size,
             hashLen := (Setup.ofCircuit st).merkle.digestLen, saltLen := saltLenOf zk (Setup.ofCircuit st),
             reps := [(lv.streams.getD 0 #[], f.proofs.getD 0 ByteArray.empty),
                      (lv.streams.getD 1 #[], f.proofs.getD 1 ByteArray.empty)] } := by
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

/-! ## The decoder's `root_B`, whatever `keep` -/

/-- **What the verifier's decoder reads of each table's `root_B`**, whatever the hashes and `keep` (S5's parse). -/
theorem decodeWithK_rootB (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {sess : RunRecord} (h : Record.decodeWithK keep hs spec text = .ok sess) :
    ∃ j l, Lean.Json.parse text = .ok j ∧ Record.fld j "link" = .ok l ∧
      List.Forall₂ (Claims.Bridge.Custody.RootField l) spec.tables.toList sess.rootB.toList := by
  unfold Record.decodeWithK Record.decodeJWithK at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  rename_i j hparse
  obtain ⟨j', hj, h⟩ := ebind_ok h
  cases hj
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    cases h
    refine ⟨_, _, by assumption, by assumption,
      List.Forall₂.imp (fun t rb h => ?_) (Claims.Bridge.Custody.mapM_forall₂ (by assumption))⟩
    beta_reduce at h
    obtain ⟨v, h1, h⟩ := ebind_ok h
    obtain ⟨s, h2, h⟩ := ebind_ok h
    obtain ⟨rb', h3, h⟩ := ebind_ok h
    repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (simp only [] at h))
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨v, s, h1, h2, h3⟩

/-- **The element-level decoder's `root_B` is `liveOf`'s**, whatever the hashes and `keep`. -/
theorem decodeWithK_rootB_live (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {text : String}
    {sess : RunRecord} {lv : Live} (hd : Record.decodeWithK keep hs spec text = .ok sess)
    (hl : liveOf spec text = .ok lv) : sess.rootB = lv.rootB := by
  obtain ⟨j, l, hp, hl1, hroot⟩ := decodeWithK_rootB keep hs hd
  obtain ⟨j', l', recs', hp', hl1', -, hroot', -, -⟩ := Claims.Bridge.Custody.liveOf_ok hl
  rw [hp] at hp'
  cases hp'
  rw [hl1] at hl1'
  cases hl1'
  exact Array.ext' (Claims.Bridge.Custody.forall₂_unique (R := Claims.Bridge.Custody.RootField l)
    (fun _ _ _ => Claims.Bridge.Custody.rootField_unique) hroot hroot')

/-! ## The query round, at the verdict read through `keep` -/

/-- **Level 0's query round in a session `verifyWithMK keep` accepts**, under any oracle and whatever `keep`:
`plainQueryRound_verifyWithM`'s conclusion, with the session `Record.decodeWithK keep` decodes. -/
theorem plainQueryRound_verifyWithMK (keep : Bool → Bool) (hs : Hashes Flock.V) (spec : SessionSpec)
    (tabs : Array Setup) (record : String) (proofs : Array ByteArray)
    (hf : ∀ st ∈ tabs, (∀ a b lw rho ts v, st.fold.folded a b lw rho ts = .ok v → rho.size + 6 = st.fold.kLog) ∧
      st.fold.kLog = st.kLog)
    (h : verifyWithMK keep hs spec tabs record proofs = .ok ()) :
    ∃ sess : RunRecord, Record.decodeWithK keep hs spec record = .ok sess ∧ tabs.size = spec.tables.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ sch : Flock.Schedule, fast100 tabs[i].m = some sch ∧ ∀ r < 2,
        ∃ (stream : StreamRec) (extras : Array Flock.Claim) (p : Proof),
          sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
          tabs[i].extra sess.points = .ok extras ∧
          decodeProof (FlockSoundness.Refine.shapeOf tabs[i] sch extras.size) proofs[2 * i + r]! = .ok p ∧
          Claims.Flock.plainQueryRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size) < stream.rounds.size ∧
          Claims.Flock.plainRepColsOf (Claims.Flock.plainProofShape tabs[i].m tabs[i].kLog (2 + extras.size)
              tabs[i].merkle.digestLen
              (if tabs[i].merkle.saltBytes = 0 && !tabs[i].openedSalts then none else some tabs[i].merkle.saltBytes))
              stream.rounds proofs[2 * i + r]! =
            Claims.Flock.zkRepCols (sch.levels.getD 0 default)
              (stream.rounds[Claims.Flock.plainQueryRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size)]!).coins
              p.ligerito.initialProof.rows ∧
          merkleCheckWith (tabs[i].merkle.opsWith hs) tabs[i].merkle p.cap (sch.levels.getD 0 default).cap
            (sch.levels.getD 0 default).d
            (positions (sch.levels.getD 0 default)
              (stream.rounds[Claims.Flock.plainQueryRound tabs[i].m tabs[i].kLog sch.k0 (2 + extras.size)]!).coins)
            p.ligerito.initialProof.rows p.ligerito.initialProof.siblings p.ligerito.initialProof.salts
            (2 ^ (sch.levels.getD 0 default).lanes) = .ok () ∧
          p.cap.foldl (· ++ ·) ByteArray.empty = sess.rootB[i]! := by
  obtain ⟨sess, schs, hsess, hsz, -, hall⟩ := verifyWithMK_tables keep hs h
  refine ⟨sess, hsess, hsz, fun i hi => ?_⟩
  obtain ⟨hi', c₀, c₁, hsch, h0, h1⟩ := hall i hi
  obtain ⟨hfi, hkl⟩ := hf tabs[i] (Array.getElem_mem hi)
  refine ⟨schs[i]'hi', hsch, fun r hr2 => ?_⟩
  have hr : ∃ c, verifyRepWith hs spec tabs[i] (schs[i]'hi') sess i r proofs[2 * i + r]! = .ok c := by
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · exact ⟨c₀, by simpa using h0⟩
    · exact ⟨c₁, h1⟩
  obtain ⟨c, hc⟩ := hr
  exact plainRepColsOf_verifyRepWith hs spec tabs[i] (schs[i]'hi') sess i r _ c hsch hfi hkl hc

/-! ## An accepted plain run's openings -/

/-- **The columns table 0's reps open in a plain record**, every one, before `distinctBelow` keeps one per position. -/
def plainRawCols (s : Claims.Seen) : List (ℕ × List 𝔽) :=
  match runOf false s 0 with
  | some r => r.reps.flatMap fun y => Claims.Flock.plainRepColsOf r.plainShape y.1 y.2
  | none => []

theorem mem_plainRawCols {s : Claims.Seen} {p : ℕ × List 𝔽} (h : p ∈ cols false s 0) : p ∈ plainRawCols s := by
  unfold cols at h
  unfold plainRawCols
  cases hr : runOf false s 0 with
  | none => simp [hr] at h
  | some r =>
    simp only [hr, Bool.false_eq_true, ↓reduceIte, Claims.Flock.plainRunCols] at h ⊢
    exact mem_of_distinctBelow h

/-- **A plain record whose verdict read through `keep` accepts within `L` opens a column only as a Merkle opening** of
table 0's level 0 in `L`, at a statement that sets up, a final message `seen` ends on and checks that pass: a column `u`
a rep opens at `q` is an opened row whose opening under table 0's scheme against `root_B[0]` of `link` passed. The
shared step of `plain_opening_raw` (`keep := id`), `plainE_opening_raw` (`fun _ => true`) and `plainK_opening_raw`. -/
theorem plain_opening_raw_of_verdictK (keep : Bool → Bool) {x : Claims.Msg} {seen : Claims.Seen}
    {L : List (Claims.Msg × Claims.Msg)} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final} {q : ℕ}
    {u : List 𝔽} (hs : setupOf false x = some (s, spec, sts)) (hf : lastFinal seen = some f)
    (hc : checks false s spec x link c₁ mids f = .ok ())
    (hv : verifyWithMK keep (hashesIn (L.map Prod.fst)) spec (tabsOf sts) f.record f.proofs = .ok ())
    (hu : (q, u) ∈ plainRawCols seen) :
    ∃ (st : Stmt) (sch : Flock.Schedule) (row : Array F128), sts[0]? = some st ∧ fast100 st.m = some sch ∧
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD 0 default) (2 ^ (sch.levels.getD 0 default).lanes)
        link.1.toArray[0]! L q row ∧
      u = row.toList.map FlockLevel3.GF128.of := by
  obtain ⟨hx, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checks_ok hc
  obtain ⟨hroot, -⟩ := Claims.Bridge.Custody.custody_ok hcu
  unfold plainRawCols at hu
  split at hu
  swap
  · simp at hu
  rename_i rr hrr
  obtain ⟨st, extras, hst, hext, rfl⟩ := runOf_eq hf hx hs hlv hrr
  obtain ⟨ρ, hρ, hp⟩ := List.mem_flatMap.1 hu
  obtain ⟨k, hk, rfl⟩ : ∃ k < 2, ρ = (lv.streams.getD k #[], f.proofs.getD k ByteArray.empty) := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hρ
    rcases hρ with rfl | rfl
    · exact ⟨0, by omega, rfl⟩
    · exact ⟨1, by omega, rfl⟩
  have hsts0 : 0 < sts.size := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, -⟩
    exact h0
  have hst0 : sts[0] = st := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, e⟩
    exact e
  have h0 : 0 < (tabsOf sts).size := by simpa [tabsOf] using hsts0
  have ht0 : (tabsOf sts)[0] = Setup.ofCircuit st := by rw [tabsOf_getElem h0]; simp only [hst0]
  obtain ⟨sess, hdec, hsz, hreps⟩ := plainQueryRound_verifyWithMK keep (hashesIn (L.map Prod.fst)) spec
    (tabsOf sts) f.record f.proofs (tabsOf_fold sts) hv
  obtain ⟨sch, hsch, hrep⟩ := hreps 0 h0
  obtain ⟨stream, extras', p, hstream, hext', hp', hlt, hcols, hmc, hcap⟩ := hrep k hk
  simp only [ht0] at hsch hext' hp' hlt hcols hmc
  have hrB := decodeWithK_rootB_live keep (hashesIn (L.map Prod.fst)) hdec hlv
  obtain ⟨hpts, -, hlsz, hstreams⟩ := liveOf_decodeWithK keep (hashesIn (L.map Prod.fst)) hlv hdec
  rw [← hpts, hext] at hext'
  cases hext'
  have htab : 0 < spec.tables.size := by rw [← hsz]; exact h0
  obtain ⟨hk', hrs⟩ := Claims.Bridge.Custody.repStreams_getElem spec htab hk
  have hk2 : 2 * 0 + k < lv.streams.size := by rw [hlsz]; exact hk'
  obtain ⟨y, hy, hfa⟩ := hstreams (2 * 0 + k) hk' hk2
  rw [hrs] at hy
  have hys : y = stream := Option.some.inj (hy.symm.trans hstream)
  subst hys
  have hcoins := forall₂_coins (fun _ _ hab => hab.2) hfa
  have hcolsEq : Claims.Flock.plainRepColsOf
      (Claims.Flock.plainProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
        (Setup.ofCircuit st).merkle.digestLen (saltLenOf false (Setup.ofCircuit st)))
      (lv.streams.getD k #[]) (f.proofs.getD k ByteArray.empty) =
      Claims.Flock.zkRepCols (sch.levels.getD 0 default)
        (y.rounds[Claims.Flock.plainQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0
          (2 + extras.size)]!).coins p.ligerito.initialProof.rows := by
    rw [← hcols]
    have e1 : lv.streams.getD k #[] = lv.streams[2 * 0 + k]! := by
      rw [show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]; simp
    have e2 : f.proofs.getD k ByteArray.empty = f.proofs[2 * 0 + k]! := by
      rw [show ByteArray.empty = default from rfl, getD_eq_getElem!]; simp
    rw [e1, e2]
    refine plainRepColsOf_congr _ _ fun j => ?_
    rw [getElem!_pos lv.streams (2 * 0 + k) hk2, ← Array.getElem?_toList, ← Array.getElem?_toList]
    exact hcoins j
  have hp2 : (q, u) ∈ Claims.Flock.zkRepCols (sch.levels.getD 0 default)
      (y.rounds[Claims.Flock.plainQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0
        (2 + extras.size)]!).coins p.ligerito.initialProof.rows := by
    rw [← hcolsEq]
    exact hp
  obtain ⟨j, hj, -, hjq, hju⟩ := mem_zkRepCols hp2
  have hms : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 ∨
      (Setup.ofCircuit st).merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
    rcases checks_leaf hc (setupOf_some hs).2 0 hsts0 with h | ⟨-, h⟩
    · rw [hst0] at h
      exact Or.inl h
    · rw [hst0] at h
      exact Or.inr h
  have hdl : (Setup.ofCircuit st).merkle.digestLen = 64 := by
    rcases hms with e | e <;> rw [e] <;> rfl
  have hruns := With.merkleCheckWith_sim (back (L.map Prod.fst))
    (With.opsWith_sim (back (L.map Prod.fst)) (back_hashes (L.map Prod.fst)) (Setup.ofCircuit st).merkle)
    (Setup.ofCircuit st).merkle _ _ _ _ _ _ _ _ () hmc
  obtain ⟨hpure, hlog⟩ := merkleCheck_logsOf hms hruns
  obtain ⟨hcs, hcw⟩ := FlockSoundness.Refine.merkleCheck_cap hpure
  refine ⟨st, sch, p.ligerito.initialProof.rows[j]!, hst, hsch,
    ⟨hms, p.cap, p.ligerito.initialProof.salts.getD j ByteArray.empty,
      sibsOf p.ligerito.initialProof.siblings ((sch.levels.getD 0 default).d - (sch.levels.getD 0 default).cap) j,
      hcs, fun b hb => (hcw b hb).trans hdl, ?_, ?_,
      fun b hb => (merkleCheck_sibs _ _ _ _ _ _ _ _ _ hpure j b hb).trans hdl, ?_⟩, hju⟩
  · rw [hcap, hrB, ← hroot, Array.toArray_toList]
  · rw [← hjq]
    exact merkleCheck_opens _ _ _ _ _ _ _ _ _ hpure j hj
  · rw [← hjq]
    exact hlog j hj

/-- **An accepted plain run's reps open a column only as a Merkle opening** of table 0's level 0 in its own log: at
the statement's setup, table 0's `fast100` schedule and the link of the tape's first message, a column `u` a rep opens
at `q` is an opened row whose opening under table 0's scheme against `root_B[0]` of the link passed, every SHA-512
input of it in the run's verifier log. -/
theorem plain_opening_raw {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V false).k x) → Coin} {q : ℕ} {u : List 𝔽} (h : (Claims.run sha512 P (V false) x ω r).out = .acc)
    (hu : (q, u) ∈ plainRawCols (Claims.run sha512 P (V false) x ω r).seen) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (st : Stmt) (sch : Flock.Schedule)
      (link : List ByteArray × List ByteArray) (row : Array F128),
      setupOf false x = some (s, spec, sts) ∧ sts[0]? = some st ∧ fast100 st.m = some sch ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD 0 default) (2 ^ (sch.levels.getD 0 default).lanes)
        link.1.toArray[0]! (Claims.run sha512 P (V false) x ω r).vLog q row ∧
      u = row.toList.map FlockLevel3.GF128.of := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hs, hf, hlink, hc, hv⟩ := acc_within h
  replace hv : verifyWithMK id (hashesIn ((Claims.run sha512 P (V false) x ω r).vLog.map Prod.fst)) spec (tabsOf sts)
      f.record f.proofs = .ok () := hv
  obtain ⟨st, sch, row, hst, hsch, hop, hu⟩ :=
    plain_opening_raw_of_verdictK id (L := (Claims.run sha512 P (V false) x ω r).vLog) hs hf hc hv hu
  exact ⟨s, spec, sts, st, sch, link, row, hs, hst, hsch, hlink, hop, hu⟩

/-- **An accepted plain run opens a column only as a Merkle opening** of table 0's level 0 in its own log: a column
`u` the run's reader opens at `q` is one a rep opens (`plain_opening_raw`). -/
theorem plain_opening {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {r : Fin ((V false).k x) → Coin}
    {q : ℕ} {u : List 𝔽} (h : (Claims.run sha512 P (V false) x ω r).out = .acc)
    (hu : Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) q = some u) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (st : Stmt) (sch : Flock.Schedule)
      (link : List ByteArray × List ByteArray) (row : Array F128),
      setupOf false x = some (s, spec, sts) ∧ sts[0]? = some st ∧ fast100 st.m = some sch ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD 0 default) (2 ^ (sch.levels.getD 0 default).lanes)
        link.1.toArray[0]! (Claims.run sha512 P (V false) x ω r).vLog q row ∧
      u = row.toList.map FlockLevel3.GF128.of :=
  plain_opening_raw h (mem_plainRawCols (lookCol_mem hu))

/-! ## An accepted masked run's openings -/

/-- **An accepted masked run opens a column only as a Merkle opening** of table 0's level 0 in its own log: at the
statement's setup, table 0's `--zk` schedule and the link of the tape's first message, a column `u` the run's reader
opens at `q` is the witness lanes of an opened row whose opening against `root_B[0]` of the link passed, every SHA-512
input of it in the run's verifier log. -/
theorem zk_opening {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {r : Fin ((V true).k x) → Coin}
    {q : ℕ} {u : List 𝔽} (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (hu : Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen 0) q = some u) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ)
      (link : List ByteArray × List ByteArray) (row : Array F128),
      setupOf true x = some (s, spec, sts) ∧ sts[0]? = some st ∧ Zk.schedule st.m = .ok (sch, tPad) ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧
      Opening MerkleScheme.hm96Sha512 (sch.levels.getD 0 default)
        (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) link.1.toArray[0]!
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
  obtain ⟨st, extras, hst, hext, rfl⟩ := runOf_eq hf hx hs hlv hrr
  simp only [↓reduceIte] at hu
  have hmem := lookCol_mem hu
  obtain ⟨ρ, hρ, hp⟩ := List.mem_flatMap.1 (mem_of_distinctBelow hmem)
  obtain ⟨k, hk, rfl⟩ : ∃ k < 2, ρ = (lv.streams.getD k #[], f.proofs.getD k ByteArray.empty) := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hρ
    rcases hρ with rfl | rfl
    · exact ⟨0, by omega, rfl⟩
    · exact ⟨1, by omega, rfl⟩
  have hsts0 : 0 < sts.size := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, -⟩
    exact h0
  have hst0 : sts[0] = st := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, e⟩
    exact e
  have h0 : 0 < (tabsOf sts).size := by simpa [tabsOf] using hsts0
  have ht0 : (tabsOf sts)[0] = Setup.ofCircuit st := by rw [tabsOf_getElem h0]; simp only [hst0]
  replace hv : Zk.verifyWithM (hashesIn (L.map Prod.fst)) spec (tabsOf sts) f.record f.proofs = .ok () := hv
  obtain ⟨sess, hdec, hsz, hreps⟩ :=
    FlockSoundness.ZkQuery.zkQueryRound_verifyWithM (hashesIn (L.map Prod.fst)) spec (tabsOf sts) f.record f.proofs hv
  obtain ⟨sch, tPad, hsch, hrep⟩ := hreps 0 h0
  obtain ⟨stream, extras', zp, hstream, hext', hzp, hlt, hcols, hmc, hcap⟩ := hrep k hk
  simp only [ht0] at hsch hext' hzp hlt hcols hmc
  have hdec' : Record.decode spec f.record = .ok sess := by
    rw [← With.decodeWith_std]
    exact (With.decodeWith_sim sub (sub_hashes (L.map Prod.fst)) spec f.record) sess hdec
  obtain ⟨hrB, hpts, hlsz, hstreams⟩ := Claims.Bridge.Custody.decode_live hdec' hlv
  rw [hpts, hext] at hext'
  cases hext'
  have htab : 0 < spec.tables.size := by rw [← hsz]; exact h0
  obtain ⟨hk', hrs⟩ := Claims.Bridge.Custody.repStreams_getElem spec htab hk
  obtain ⟨y, hy, hfa⟩ := hstreams (2 * 0 + k) hk'
  rw [hrs] at hy
  have hys : y = stream := Option.some.inj (hy.symm.trans hstream)
  subst hys
  have hcoins := forall₂_coins (fun _ _ hab => hab.2.1) hfa
  have hcolsEq : Claims.Flock.zkRepColsOf
      (Claims.Flock.zkProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
        (Setup.ofCircuit st).merkle.digestLen (saltLenOf true (Setup.ofCircuit st)))
      (lv.streams.getD k #[]) (f.proofs.getD k ByteArray.empty) =
      (Claims.Flock.zkRepCols (sch.levels.getD 0 default)
        (y.rounds[Claims.Flock.zkQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0
          (2 + extras.size)]!).coins zp.proof.ligerito.initialProof.rows).map
        (fun p => (p.1, p.2.take (2 ^ (sch.levels.getD 0 default).lanes))) := by
    rw [← hcols]
    have e1 : lv.streams.getD k #[] = lv.streams[2 * 0 + k]! := by
      rw [show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]; simp
    have e2 : f.proofs.getD k ByteArray.empty = f.proofs[2 * 0 + k]! := by
      rw [show ByteArray.empty = default from rfl, getD_eq_getElem!]; simp
    rw [e1, e2]
    refine zkRepColsOf_congr _ _ fun j => ?_
    rw [← Array.getElem?_toList, ← Array.getElem?_toList]
    exact (hcoins j).symm
  have hp2 : (q, u) ∈ (Claims.Flock.zkRepCols (sch.levels.getD 0 default)
      (y.rounds[Claims.Flock.zkQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0
        (2 + extras.size)]!).coins zp.proof.ligerito.initialProof.rows).map
      (fun p => (p.1, p.2.take (2 ^ (sch.levels.getD 0 default).lanes))) := by
    rw [← hcolsEq]
    exact hp
  obtain ⟨⟨q', w⟩, hw, hqw⟩ := List.mem_map.1 hp2
  simp only [Prod.mk.injEq] at hqw
  obtain ⟨rfl, rfl⟩ := hqw
  obtain ⟨j, hj, -, hjq, hju⟩ := mem_zkRepCols hw
  have hms : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 := by
    have := checks_hm96 hc (setupOf_some hs).2 (setupOf_zk hs) 0 hsts0
    rw [hst0] at this
    exact this
  rw [hms] at hmc
  have hruns := With.merkleCheckWith_sim (back (L.map Prod.fst))
    (With.opsWith_sim (back (L.map Prod.fst)) (back_hashes (L.map Prod.fst)) MerkleScheme.hm96Sha512)
    MerkleScheme.hm96Sha512 _ _ _ _ _ _ _ _ () hmc
  rw [opsWith_hm96] at hruns
  obtain ⟨hpure, hlog⟩ := merkleCheck_logs hruns
  obtain ⟨hcs, hcw⟩ := merkleCheck_cap hpure
  refine ⟨s, spec, sts, st, sch, tPad, link, zp.proof.ligerito.initialProof.rows[j]!, hs, hst, hsch, hlink,
    ⟨Or.inl rfl, zp.proof.cap, zp.proof.ligerito.initialProof.salts.getD j ByteArray.empty,
      sibsOf zp.proof.ligerito.initialProof.siblings
        ((sch.levels.getD 0 default).d - (sch.levels.getD 0 default).cap) j,
      hcs, hcw, ?_, ?_, merkleCheck_sibs _ _ _ _ _ _ _ _ _ hpure j, ?_⟩, by rw [hju]⟩
  · rw [hcap, hrB, ← hroot, Array.toArray_toList]
  · rw [← hjq]
    exact merkleCheck_opens _ _ _ _ _ _ _ _ _ hpure j hj
  · rw [opensInsOf_hm96, ← hjq]
    exact hlog j hj

/-! ## `ConflictTied` -/

/-- **`ConflictTied` at the deployed plain record**: two accepted runs at one tape that open different columns at one
position collide in their joined verifier logs, at SHA-512. -/
theorem plain_conflictTied (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ConflictTied sha512 (deployPlain rel) (wordLen false) := by
  intro x Ω P ω r r' q u v hacc hacc' _ hu hv huv
  obtain ⟨s, spec, sts, st, sch, link, row, hs, hst, hsch, hlink, ho, hu'⟩ :=
    plain_opening (P := P) (r := r) hacc hu
  obtain ⟨s', spec', sts', st', sch', link', row', hs', hst', hsch', hlink', ho', hv'⟩ :=
    plain_opening (P := P) (r := r') hacc' hv
  rw [hs] at hs'
  cases hs'
  rw [hst] at hst'
  cases hst'
  rw [hsch] at hsch'
  cases hsch'
  rw [hlink] at hlink'
  cases hlink'
  exact tiedColl_of_openings ho ho' fun e => huv (by rw [hu', hv', e])

theorem tiedColl_of_append_self {H : Claims.Msg → Claims.Msg} {l : List (Claims.Msg × Claims.Msg)}
    (h : Claims.TiedColl H (l ++ l)) : Claims.TiedColl H l := by
  obtain ⟨u, v, hc, hu, hv⟩ := h
  simp only [List.map_append, List.mem_append, or_self] at hu hv
  exact ⟨u, v, hc, hu, hv⟩

/-- **A self-clash among an accepted plain run's columns collides in its own log**: two columns table 0's reps open at
one position that differ give a collision among the run's verifier calls, at SHA-512. -/
theorem plain_selfClash_tied {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V false).k x) → Coin} {q : ℕ} {u v : List 𝔽} (h : (Claims.run sha512 P (V false) x ω r).out = .acc)
    (hu : (q, u) ∈ plainRawCols (Claims.run sha512 P (V false) x ω r).seen)
    (hv : (q, v) ∈ plainRawCols (Claims.run sha512 P (V false) x ω r).seen) (huv : u ≠ v) :
    Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω r).vLog := by
  obtain ⟨s, spec, sts, st, sch, link, row, hs, hst, hsch, hlink, ho, hu'⟩ := plain_opening_raw h hu
  obtain ⟨s', spec', sts', st', sch', link', row', hs', hst', hsch', hlink', ho', hv'⟩ := plain_opening_raw h hv
  rw [hs] at hs'
  cases hs'
  rw [hst] at hst'
  cases hst'
  rw [hsch] at hsch'
  cases hsch'
  rw [hlink] at hlink'
  cases hlink'
  exact tiedColl_of_append_self (tiedColl_of_openings ho ho' fun e => huv (by rw [hu', hv', e]))

/-- **`ConflictTied` at the deployed masked (`--zk`) record**: two accepted runs at one tape that open different
columns at one position collide in their joined verifier logs, at SHA-512. -/
theorem zk_conflictTied (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ConflictTied sha512 (deployZk rel) (wordLen true) := by
  intro x Ω P ω r r' q u v hacc hacc' _ hu hv huv
  obtain ⟨s, spec, sts, st, sch, tPad, link, row, hs, hst, hsch, hlink, ho, hu'⟩ :=
    zk_opening (P := P) (r := r) hacc hu
  obtain ⟨s', spec', sts', st', sch', tPad', link', row', hs', hst', hsch', hlink', ho', hv'⟩ :=
    zk_opening (P := P) (r := r') hacc' hv
  rw [hs] at hs'
  cases hs'
  rw [hst] at hst'
  cases hst'
  rw [hsch] at hsch'
  cases hsch'
  rw [hlink] at hlink'
  cases hlink'
  exact tiedColl_of_openings ho ho' fun e => huv (by rw [hu', hv', e])

/-! ## KS at SHA-512 from a fixed table's bound -/

/-- **The deployed plain verifier: KS**, at SHA-512, from a fixed table's bound alone. -/
theorem plain_KS_of_table (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployPlain rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployPlain rel).V x)
    (htab : Claims.Flock.TableBoundθ sha512 (deployPlain rel) (wordLen false) (Claims.Flock.orZero (wide false))) :
    Claims.Flock.KS sha512 (deployPlain rel) :=
  plain_KS_sha512 rel hone (plain_conflictTied rel) htab

/-- **The deployed masked (`--zk`) verifier: KS**, at SHA-512, from a fixed table's bound alone. -/
theorem zk_KS_of_table (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployZk rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployZk rel).V x)
    (htab : Claims.Flock.TableBoundθ sha512 (deployZk rel) (wordLen true) (Claims.Flock.orZero (wide true))) :
    Claims.Flock.KS sha512 (deployZk rel) :=
  zk_KS_sha512 rel hone (zk_conflictTied rel) htab

end FlockDeploy.KS
