import Proofs.Flock.Claims.Recursion.NextShaS17
import Proofs.Flock.Claims.Recursion.NextShaRep
import Proofs.Flock.Claims.Recursion.CallsAgree

/-!
# The last step's decision with no proof file decoded (`decisionAtS_eval`)

`decisionAtS` runs the plain verdict (`verifyJWithM`) on the proofs' files (`filesAt`) at the record's value with its
proof digests the files' (`shaSubJ`) and its coins the live coins (`coinSubJ`). Run against `H`, that verdict is
`verifyJPWith`: the same checks read off the decoded proofs, with no proof file decoded and no S17
(`decisionAtS_eval`), wherever each proof is shaped for its file (`FlockVBridge.Family.ProofShaped` at `shapeAt`) and
the rep streams' names are distinct (`NamesDistinct`).

* **The verdict on the decoded proofs** (`verifyRepDWith`, `verifyJPWith`): `Flock.verifyRepWith` and
  `Flock.verifyJWithM` with each rep's proof given, so neither decodes a proof file, and without S17, which compares the
  files' digests with the record's.
* **They agree** (`verifyRepWith_dec`, `verifyJWithM_eqP`): a rep's verdict on a file that decodes to `p` at the
  verdict's shape is its verdict on `p`; the whole verdict on files is `verifyJPWith` on the proofs where S17 passes and
  each rep agrees.
* **At the substituted value** (`decodeJWith_sub`, `verifyJWithM_sub`): the decoder's link points are the first coin's
  (`points_read`) and its proof digests are `shaArr`'s (`proof_{dk}` survives `coinSubJ`, `arr_coinSubJ_proof`), so
  every rep's file is read at `shapeAt` (`decode_filesAt`) and S17 passes (`s17_canonical`).
* **S17's two hypotheses** (`namesDistinct_of_setup`, `publicInputs_none_of_checksAtJ`): the file count is the
  verdict's own check before S17 (`proofs.size != want.size`), so it is no hypothesis; the rep streams' names are
  distinct at every statement that sets up with no public inputs (`KS.tables_nodup`, `KS.repNames_nodup`), and the
  scope check among `checksAtJ` refuses public inputs.
* **The decision** (`decisionAtS_evalJ`, `decisionAtS_eval`, `decisionAtS_ok_iff`): run against `H`, `decisionAtS` is
  `verifyJPWith` at `hOf H` then `checksAtJ`, and at a statement that sets up it accepts exactly where both pass.

**What stays.** Each rep's proof must be shaped for its file (`hsh`), a fact about the proofs the last message carries;
the reader parts read them, and no part's check is yet known to hold it. The decoder still reads `proof_{dk}` at the
substituted value, whose digests the decision computes from the files (`decisionAtS`'s `mapM`), and S17 then passes on
them by construction.
-/

namespace FlockDeploy

open Flock
open Lean (Json)
open FlockComplete.JsonRT (f128s_map array_mapM_ok_of dk_cases)

/-! ## The verdict on the decoded proofs -/

section Decoded

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT Flock.V M]

/-- `Flock.verifyRepWith` with the proof read by `dec` at the rep's shape. -/
def verifyRepDecWith (dec : Flock.Shape → M Flock.Proof) (hs : Flock.Hashes M) (spec : SessionSpec)
    (st : Flock.Setup) (sch : Flock.Schedule) (sess : Session) (i r : ℕ) : M ByteArray := do
  let name := s!"{spec.tables[i]!}/rep{r}"
  let some stream := sess.streams.find? (·.name == name) | throw s!"S11: no stream {name}"
  let extras ← st.extra sess.points
  let p ← dec (Flock.repFileShape st sch extras.size)
  verifyRepPWith hs spec st sch sess i r stream extras p

/-- **A rep's verdict on its decoded proof `p`**: the rep's stream (S11), the statement's extra claims, then
`Flock.verifyRepPWith` at `p`. No proof file is decoded. -/
def verifyRepDWith (hs : Flock.Hashes M) (spec : SessionSpec) (st : Flock.Setup) (sch : Flock.Schedule)
    (sess : Session) (i r : ℕ) (p : Flock.Proof) : M ByteArray := do
  let name := s!"{spec.tables[i]!}/rep{r}"
  let some stream := sess.streams.find? (·.name == name) | throw s!"S11: no stream {name}"
  let extras ← st.extra sess.points
  verifyRepPWith hs spec st sch sess i r stream extras p

/-- **The verdict on the decoded proofs `ps`**: `Flock.verifyJWithM` with rep `2 i + r`'s proof `ps[2 i + r]`
(`verifyRepDWith`), so no proof file is decoded, and without S17. -/
def verifyJPWith (hs : Flock.Hashes M) (spec : SessionSpec) (tabs : Array Flock.Setup) (j : Lean.Json)
    (ps : Array Flock.Proof) : M Unit := do
  if tabs.size != spec.tables.size || tabs.size == 0 then throw "setup: one statement per table expected"
  let sess ← Record.decodeJWith hs spec j
  for (st, p) in tabs.zip sess.publics do st.publics p
  let schs ← tabs.mapM fun st => match fast100 st.m with
    | some sch => pure sch
    | none => throw s!"m = {st.m}: no fast100 schedule"
  let want := spec.repStreams
  if ps.size != want.size then throw s!"{want.size} proofs expected"
  for i in [0:tabs.size] do
    let (st, sch) := (tabs[i]!, schs[i]!)
    let c0 ← verifyRepDWith hs spec st sch sess i 0 ps[2 * i]!
    let c1 ← verifyRepDWith hs spec st sch sess i 1 ps[2 * i + 1]!
    if c0 != c1 then throw s!"S18/R1: {spec.tables[i]!}: the reps' caps differ"

end Decoded

/-! ## Congruences in `V` -/

theorem vok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

theorem bind_congr_ok {α β : Type} {x : Flock.V α} {f g : α → Flock.V β} (h : ∀ a, x = .ok a → f a = g a) :
    x >>= f = x >>= g := by
  cases x with
  | error e => rfl
  | ok a => exact h a rfl

theorem bind_congr_left {α β : Type} {x y : Flock.V α} {f g : α → Flock.V β} (hxy : x = y)
    (h : ∀ a, y = .ok a → f a = g a) : x >>= f = y >>= g := by
  subst hxy
  exact bind_congr_ok h

theorem bind_of_ok {α β : Type} {x : Flock.V α} {a : α} (hx : x = .ok a) (f : α → Flock.V β) : x >>= f = f a := by
  subst hx
  rfl

theorem ite_congr_V {β : Type} {c c' : Prop} [Decidable c] [Decidable c'] {t e t' e' : Flock.V β} (hc : c = c')
    (ht : c → t = t') (he : ¬c → e = e') : ite c t e = ite c' t' e' := by
  subst hc
  by_cases h : c
  · rw [if_pos h, if_pos h]
    exact ht h
  · rw [if_neg h, if_neg h]
    exact he h

theorem list_forIn_congr {α : Type} : ∀ (l : List α) (f g : α → Unit → Flock.V (ForInStep Unit)),
    (∀ a ∈ l, f a () = g a ()) → forIn l () f = forIn l () g
  | [], _, _, _ => rfl
  | a :: l, f, g, h => by
    rw [List.forIn_cons, List.forIn_cons]
    refine bind_congr_left (h a List.mem_cons_self) fun r _ => ?_
    cases r with
    | done b => rfl
    | yield b => exact list_forIn_congr l f g fun a' ha' => h a' (List.mem_cons_of_mem _ ha')

theorem list_forIn_unit_ok {α : Type} : ∀ (l : List α) (f : α → Unit → Flock.V (ForInStep Unit)),
    (∀ a ∈ l, f a () = .ok (.yield ())) → forIn l () f = .ok ()
  | [], _, _ => rfl
  | a :: l, f, h => by
    rw [List.forIn_cons, h a List.mem_cons_self]
    exact list_forIn_unit_ok l f fun a' ha' => h a' (List.mem_cons_of_mem _ ha')

theorem range_forIn_congr {n : ℕ} {f g : ℕ → Unit → Flock.V (ForInStep Unit)}
    (h : ∀ i, i < n → f i () = g i ()) : forIn [0:n] () f = forIn [0:n] () g := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  refine list_forIn_congr _ f g fun i hi => h i ?_
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_one_sub_one, Nat.div_one, List.mem_range'] at hi
  omega

/-! ## The verdicts agree -/

theorem verifyRepWith_eqDec (hs : Flock.Hashes Flock.V) (spec : SessionSpec) (st : Flock.Setup)
    (sch : Flock.Schedule) (sess : Session) (i r : ℕ) (bytes : ByteArray) :
    verifyRepWith hs spec st sch sess i r bytes =
      verifyRepDecWith (fun sh => decodeProof sh bytes) hs spec st sch sess i r := by
  rw [verifyRepWith_split]
  rfl

theorem verifyRepDWith_eqDec (hs : Flock.Hashes Flock.V) (spec : SessionSpec) (st : Flock.Setup)
    (sch : Flock.Schedule) (sess : Session) (i r : ℕ) (p : Flock.Proof) :
    verifyRepDWith hs spec st sch sess i r p = verifyRepDecWith (fun _ => pure p) hs spec st sch sess i r := rfl

theorem verifyRepDecWith_congr (hs : Flock.Hashes Flock.V) (spec : SessionSpec) (st : Flock.Setup)
    (sch : Flock.Schedule) (sess : Session) (i r : ℕ) {dec dec' : Flock.Shape → Flock.V Flock.Proof}
    (h : ∀ extras, st.extra sess.points = .ok extras →
      dec (Flock.repFileShape st sch extras.size) = dec' (Flock.repFileShape st sch extras.size)) :
    verifyRepDecWith dec hs spec st sch sess i r = verifyRepDecWith dec' hs spec st sch sess i r := by
  unfold verifyRepDecWith
  dsimp only
  split
  · exact bind_congr_ok fun extras he => by rw [h extras he]
  · rfl

/-- **A rep's verdict on a file that decodes to `p` is its verdict on `p`**, the file decoded at the verdict's shape
(`Flock.repFileShape`) at the statement's extra claims. -/
theorem verifyRepWith_dec (hs : Flock.Hashes Flock.V) (spec : SessionSpec) (st : Flock.Setup)
    (sch : Flock.Schedule) (sess : Session) (i r : ℕ) {bytes : ByteArray} {p : Flock.Proof}
    (hdec : ∀ extras, st.extra sess.points = .ok extras →
      decodeProof (Flock.repFileShape st sch extras.size) bytes = .ok p) :
    verifyRepWith hs spec st sch sess i r bytes = verifyRepDWith hs spec st sch sess i r p := by
  rw [verifyRepWith_eqDec, verifyRepDWith_eqDec]
  exact verifyRepDecWith_congr hs spec st sch sess i r fun extras he => hdec extras he

theorem schs_of_mapM {f : Flock.Setup → Flock.V Flock.Schedule} {tabs : Array Flock.Setup}
    {schs : Array Flock.Schedule} (h : tabs.mapM f = .ok schs)
    (hf : ∀ st sch, f st = .ok sch → fast100 st.m = some sch) :
    ∀ i, i < tabs.size → fast100 tabs[i]!.m = some schs[i]! := by
  intro i hi
  obtain ⟨hl, hall⟩ := array_mapM_ok_inv f tabs schs h
  rw [getElem!_pos tabs i hi, getElem!_pos schs i (hl ▸ hi)]
  exact hf _ _ (hall i hi)

/-- **The verdict on files is `verifyJPWith` on the proofs**: in `V`, where as many files as proofs, S17 passing at
every decoded session with as many files as rep streams (`h17`), and each rep's verdict on its file its verdict on its
proof (`hrep`). -/
theorem verifyJWithM_eqP (hs : Flock.Hashes Flock.V) (spec : SessionSpec) (tabs : Array Flock.Setup)
    (j : Lean.Json) {proofs : Array ByteArray} {ps : Array Flock.Proof} (hsize : proofs.size = ps.size)
    (h17 : ∀ sess, Record.decodeJWith hs spec j = .ok sess → proofs.size = spec.repStreams.size →
      (sess.proofSha.size != spec.repStreams.size ||
        !spec.repStreams.all fun a => sess.proofSha.any fun x => x.1 == a.2.2) = false ∧
      ∀ q ∈ (spec.repStreams.zip proofs).toList, ∃ d, spec.dgWith hs q.2 = .ok d ∧
        ((sess.proofSha.find? (·.1 == q.1.2.2)).map (·.2) != some (hexOf d)) = false)
    (hrep : ∀ sess (schs : Array Flock.Schedule), Record.decodeJWith hs spec j = .ok sess →
      (∀ i, i < tabs.size → fast100 tabs[i]!.m = some schs[i]!) → tabs.size = spec.tables.size →
      proofs.size = spec.repStreams.size → ∀ i, i < tabs.size → ∀ r, r < 2 →
        verifyRepWith hs spec tabs[i]! schs[i]! sess i r proofs[2 * i + r]! =
          verifyRepDWith hs spec tabs[i]! schs[i]! sess i r ps[2 * i + r]!) :
    verifyJWithM hs spec tabs j proofs = verifyJPWith hs spec tabs j ps := by
  unfold verifyJWithM verifyJPWith
  dsimp only
  refine ite_congr_V rfl (fun _ => rfl) fun hc0 => ?_
  have ht : tabs.size = spec.tables.size := by
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, not_or, Decidable.not_not] at hc0
    exact hc0.1
  refine bind_congr_ok fun sess hsess => ?_
  refine bind_congr_ok fun _ _ => ?_
  refine bind_congr_ok fun schs hschs => ?_
  have hsch := schs_of_mapM hschs fun st sch hf => by
    split at hf
    · rename_i heq
      simp only [pure, Except.pure, Except.ok.injEq] at hf
      rw [← hf]
      exact heq
    · cases hf
  refine ite_congr_V (by rw [hsize]) (fun _ => rfl) fun hc1 => ?_
  have hn : proofs.size = spec.repStreams.size := by simpa [bne_iff_ne] using hc1
  obtain ⟨hA, hB⟩ := h17 sess hsess hn
  refine (if_neg ?_).trans ?_
  · intro hc
    exact absurd (hA.symm.trans hc) Bool.false_ne_true
  refine (bind_of_ok (a := ()) ?_ _).trans ?_
  · rw [← Array.forIn_toList]
    refine list_forIn_unit_ok _ _ fun a ha => ?_
    obtain ⟨d, hd, hne⟩ := hB a ha
    (simp only [hd, vok_bind, hne, Bool.false_eq_true, ite_false]) <;> rfl
  refine bind_congr_left (range_forIn_congr fun i hi => ?_) fun _ _ => rfl
  have e0 := hrep sess schs hsess hsch ht hn i hi 0 (by decide)
  have e1 := hrep sess schs hsess hsch ht hn i hi 1 (by decide)
  exact bind_congr_left e0 fun c0 _ => bind_congr_left e1 fun c1 _ => rfl

/-! ## The decoder at the substituted value -/

theorem proofKey_ne_streams (spec : SessionSpec) : s!"proof_{spec.dk}" ≠ "streams" := by
  rcases dk_cases spec with e | e <;> simp [e, toString]

theorem proofKey_ne_link (spec : SessionSpec) : s!"proof_{spec.dk}" ≠ "link" := by
  rcases dk_cases spec with e | e <;> simp [e, toString]

theorem draw_ne_proofKey (spec : SessionSpec) : "unit_draw" ≠ s!"proof_{spec.dk}" := by
  rcases dk_cases spec with e | e <;> simp [e, toString]

/-- The decoder's reads of the link points and the proof digests. -/
theorem decodeJWith_reads (hs : Flock.Hashes Flock.V) {spec : SessionSpec} {J : Json} {sess : Session}
    (h : Record.decodeJWith hs spec J = .ok sess) :
    ∃ (l : Json) (pa a : Array Json), Record.fld J "link" = .ok l ∧ Record.arr l "points" = .ok pa ∧
      pa.mapM (m := Flock.V) (fun p => match p.getArr? with
        | .ok a => Record.f128s a
        | .error _ => .error "record: a link point is not a list") = .ok sess.points ∧
      Record.arr J s!"proof_{spec.dk}" = .ok a ∧
      a.mapM (m := Flock.V) (fun p => do pure ((← Record.str p "stream"), (← Record.str p spec.dk))) =
        .ok sess.proofSha := by
  unfold Record.decodeJWith at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := FlockSoundness.Refine.ebind_ok h) | (split at h))
  all_goals
    first
    | (cases h; done)
    | (cases h
       exact ⟨_, _, _, by assumption, by assumption, by assumption, by assumption, by assumption⟩)

/-- **The decoder's link points at the coin-substituted value are the first coin's.** -/
theorem points_read {spec : SessionSpec} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {j l : Json}
    {pa : Array Json} {pts : Array (Array F128)} (hl : Record.fld (coinSubJ spec c₁ mids j) "link" = .ok l)
    (hpa : Record.arr l "points" = .ok pa)
    (hpts : pa.mapM (m := Flock.V) (fun p => match p.getArr? with
      | .ok a => Record.f128s a
      | .error _ => .error "record: a link point is not a list") = .ok pts) :
    pts = pointsOf spec c₁ := by
  simp only [coinSubJ, fillCoins, mapRounds, setPoints] at hl
  rw [fld_mapKey_ne _ _ (by decide)] at hl
  obtain ⟨l₀, -, rfl⟩ := fld_mapKey_self hl
  obtain rfl := arr_mapKey_const hpa
  have := array_mapM_ok_of (fun p => match p.getArr? with
    | .ok a => Record.f128s a
    | .error _ => .error "record: a link point is not a list") ((pointsOf spec c₁).map hexArr) (pointsOf spec c₁)
    (by simp) fun i h₁ h₂ => by
      simp only [Array.getElem_map, hexArr, Json.getArr?]
      exact f128s_map _
  rw [this] at hpts
  cases hpts
  rfl

/-- `coinSubJ` maps only the link and the streams, so `proof_{dk}` reads through it. -/
theorem arr_coinSubJ_proof (spec : SessionSpec) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (j : Json) :
    Record.arr (coinSubJ spec c₁ mids j) s!"proof_{spec.dk}" = Record.arr j s!"proof_{spec.dk}" := by
  simp only [coinSubJ, fillCoins, eraseCoins, mapRounds, setPoints]
  first
  | (rw [arr_mapKey_ne _ _ (proofKey_ne_streams spec), arr_mapKey_ne _ _ (proofKey_ne_link spec),
      arr_mapKey_ne _ _ (proofKey_ne_streams spec), arr_mapKey_ne _ _ (proofKey_ne_link spec)])
  | (simp only [arr_mapKey_ne _ _ (proofKey_ne_streams spec), arr_mapKey_ne _ _ (proofKey_ne_link spec)])

/-- **The decoder at the substituted value**: its link points are the first coin's, its proof digests `shaArr`'s. -/
theorem decodeJWith_sub (hs : Flock.Hashes Flock.V) {spec : SessionSpec} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {ds : Array ByteArray} {j : Json} {sess : Session}
    (h : Record.decodeJWith hs spec (coinSubJ spec c₁ mids (shaSubJ spec ds j)) = .ok sess) :
    sess.points = pointsOf spec c₁ ∧ sess.proofSha = (spec.repStreams.zip ds).map fun q => (q.1.2.2, hexOf q.2) := by
  obtain ⟨l, pa, a, hl, hpa, hpts, ha, hm⟩ := decodeJWith_reads hs h
  exact ⟨points_read hl hpa hpts, proofSha_read ((arr_coinSubJ_proof spec c₁ mids _).symm.trans ha) hm⟩

/-! ## S17's hypotheses from the statement -/

theorem namesDistinct_of_nodup {spec : SessionSpec} (h : (spec.repStreams.toList.map (·.2.2)).Nodup) :
    NamesDistinct spec := by
  intro i k hi hk e
  have e' : (spec.repStreams.toList.map (·.2.2))[i]'(by simpa using hi) =
      (spec.repStreams.toList.map (·.2.2))[k]'(by simpa using hk) := by simpa using e
  exact (List.Nodup.getElem_inj_iff h).1 e'

/-- **The rep streams' names are distinct** at a statement that sets up with no public inputs. -/
theorem namesDistinct_of_setup {s : Ins} {spec : SessionSpec} {sts : Array Stmt} (hs : s.setup = .ok (spec, sts))
    (hpi : s.read.publicInputs = none) : NamesDistinct spec :=
  namesDistinct_of_nodup (KS.repNames_nodup (KS.tables_nodup hs hpi))

/-- **Checks that pass carry no public inputs**: `checksAtJ` runs the scope check. -/
theorem publicInputs_none_of_checksAtJ {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalJ} {j' : Json}
    (h : checksAtJ s spec x link c₁ mids f j' = .ok ()) : s.read.publicInputs = none := by
  unfold checksAtJ at h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨⟨⟩, h5, -⟩ := FlockSoundness.Refine.ebind_ok h
  exact Claims.Bridge.Custody.publicInputs_none_of_check h5

/-! ## The verdict at the substituted value -/

/-- **The verdict at the substituted value is `verifyJPWith`**: at the files' digests through `h`, where each proof is
shaped for its file and the rep streams' names are distinct, the verdict on the proofs' files is the verdict on the
proofs, with no file decoded and no S17. -/
theorem verifyJWithM_sub (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Flock.Setup) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (j : Json) {ps : Array Flock.Proof}
    (hsh : ∀ (k : ℕ) (hk : k < ps.size),
      FlockVBridge.Family.ProofShaped (shapeAt tabs (pointsOf spec c₁) k) ps[k])
    (hnd : NamesDistinct spec) :
    verifyJWithM (Flock.Hashes.ofSha512 h : Flock.Hashes Flock.V) spec tabs
        (coinSubJ spec c₁ mids (shaSubJ spec ((filesAt tabs (pointsOf spec c₁) ps).map
          (FlockComplete.dgOf spec h)) j))
        (filesAt tabs (pointsOf spec c₁) ps) =
      verifyJPWith (Flock.Hashes.ofSha512 h : Flock.Hashes Flock.V) spec tabs
        (coinSubJ spec c₁ mids (shaSubJ spec ((filesAt tabs (pointsOf spec c₁) ps).map
          (FlockComplete.dgOf spec h)) j)) ps := by
  have hfs : (filesAt tabs (pointsOf spec c₁) ps).size = ps.size := by simp only [filesAt, Array.size_mapIdx]
  refine verifyJWithM_eqP _ spec tabs _ hfs (fun sess hsess hn => ?_)
    (fun sess schs hsess hsch ht hn i hi r hr => ?_)
  · obtain ⟨hA, hB⟩ := s17_canonical spec h (decodeJWith_sub _ hsess).2 hn hnd
    exact ⟨hA, fun q hq => ⟨_, FlockComplete.dgWith_ofSha512 spec h q.2, hB q hq⟩⟩
  · have hk : 2 * i + r < ps.size := by
      have := repStreams_size spec
      omega
    have hk' : 2 * i + r < (filesAt tabs (pointsOf spec c₁) ps).size := by omega
    refine verifyRepWith_dec _ spec _ _ sess i r fun extras he => ?_
    rw [(decodeJWith_sub _ hsess).1] at he
    rw [repFileShape_shapeAt hr (hsch i hi) he, getElem!_pos (filesAt tabs (pointsOf spec c₁) ps) (2 * i + r) hk',
      getElem!_pos ps (2 * i + r) hk]
    exact decode_filesAt hk (hsh _ hk)

/-! ## The decision -/

theorem liftV_run_eval {α : Type} (H : Claims.Msg → Claims.Msg) (v : Flock.V α) :
    (((monadLift v : Claims.OCV α).run).eval H).1 = v := by
  first
  | rfl
  | (cases v <;> rfl)

theorem evRun_bind {α β : Type} (H : Claims.Msg → Claims.Msg) (x : Claims.OCV α) (f : α → Claims.OCV β) :
    (((x >>= f).run).eval H).1 = (((x.run).eval H).1 >>= fun a => (((f a).run).eval H).1) := by
  simp only [ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  first
  | (cases hx : ((x.run).eval H).1 <;> rfl)
  | (split <;> rfl)

/-- **The verifier at a parsed value over `OC`, against `H`, is `verifyJWithM` at `hOf H`.** -/
theorem verifyJOC_eval (H : Claims.Msg → Claims.Msg) (spec : SessionSpec) (tabs : Array Flock.Setup) (j : Json)
    (proofs : Array ByteArray) :
    ((verifyJWithM Claims.Hashes.oc spec tabs j proofs).run.eval H).1 =
      verifyJWithM (Flock.Hashes.ofSha512 (Claims.hOf H) : Flock.Hashes Flock.V) spec tabs j proofs := by
  rw [← (Flock.With.verifyJWithM_sim (Claims.ocLog H) (Claims.ocLog_hashes H) spec tabs j proofs #[]).1]
  exact Flock.With.verifyJWithM_sim (Flock.With.Sim.forget _) (Flock.With.Sim.forget_logged _) spec tabs j proofs #[]

theorem drawOfJ_shaSubJ (spec : SessionSpec) (ds : Array ByteArray) (j : Json) :
    drawOfJ (shaSubJ spec ds j) = drawOfJ j := by
  unfold drawOfJ shaSubJ
  rw [getObjVal?_mapKey_ne _ _ (draw_ne_proofKey spec)]

/-- The checks after the verdict read the statement and the draw, which `shaSubJ` keeps, and no proof file. -/
theorem checksAtJ_shaSubJ (s : Ins) (spec : SessionSpec) (x : Claims.Msg) (link : List ByteArray × List ByteArray)
    (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (stmt : Claims.Msg) (ds : Array ByteArray) (j : Json)
    (bs : Array ByteArray) (j' : Json) :
    checksAtJ s spec x link c₁ mids ⟨stmt, shaSubJ spec ds j, bs⟩ j' =
      checksAtJ s spec x link c₁ mids ⟨stmt, j, #[]⟩ j' := by
  first
  | (simp only [checksAtJ, drawOfJ_shaSubJ]; done)
  | (unfold checksAtJ; rw [drawOfJ_shaSubJ])
  | (unfold checksAtJ; dsimp only; rw [drawOfJ_shaSubJ])

/-- **`decisionAtS` run against `H`**: the plain verdict at `hOf H` on the proofs' files at the substituted value,
then the checks after it. -/
theorem decisionAtS_evalJ (H : Claims.Msg → Claims.Msg) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    {stmt : Claims.Msg} {j : Json} {ps : Array Flock.Proof} :
    ((decisionAtS s spec sts x link c₁ mids ⟨stmt, j, ps⟩).run.eval H).1 =
      (verifyJWithM (Flock.Hashes.ofSha512 (Claims.hOf H) : Flock.Hashes Flock.V) spec (tabsOf sts)
          (coinSubJ spec c₁ mids (shaSubJ spec ((filesAt (tabsOf sts) (pointsOf spec c₁) ps).map
            (FlockComplete.dgOf spec (Claims.hOf H))) j))
          (filesAt (tabsOf sts) (pointsOf spec c₁) ps) >>= fun _ =>
        checksAtJ s spec x link c₁ mids ⟨stmt, j, #[]⟩
          (coinSubJ spec c₁ mids (shaSubJ spec ((filesAt (tabsOf sts) (pointsOf spec c₁) ps).map
            (FlockComplete.dgOf spec (Claims.hOf H))) j))) := by
  dsimp only [decisionAtS]
  rw [evOk_bind_fst (evOk_mapM_array (evOk_dg H spec) (filesAt (tabsOf sts) (pointsOf spec c₁) ps))]
  dsimp only [decisionAtP, decisionAtJ]
  rw [evRun_bind, verifyJOC_eval]
  refine bind_congr_ok fun _ _ => ?_
  exact (liftV_run_eval H _).trans (checksAtJ_shaSubJ s spec x link c₁ mids stmt _ j _ _)

/-- **The last step's decision with no proof file decoded**: run against `H`, where each proof is shaped for its file
and the rep streams' names are distinct, `decisionAtS` is the verdict on the decoded proofs (`verifyJPWith`, no S17),
then the checks after it. -/
theorem decisionAtS_eval (H : Claims.Msg → Claims.Msg) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    {stmt : Claims.Msg} {j : Json} {ps : Array Flock.Proof}
    (hsh : ∀ (k : ℕ) (hk : k < ps.size),
      FlockVBridge.Family.ProofShaped (shapeAt (tabsOf sts) (pointsOf spec c₁) k) ps[k])
    (hnd : NamesDistinct spec) :
    ((decisionAtS s spec sts x link c₁ mids ⟨stmt, j, ps⟩).run.eval H).1 =
      (verifyJPWith (Flock.Hashes.ofSha512 (Claims.hOf H) : Flock.Hashes Flock.V) spec (tabsOf sts)
          (coinSubJ spec c₁ mids (shaSubJ spec ((filesAt (tabsOf sts) (pointsOf spec c₁) ps).map
            (FlockComplete.dgOf spec (Claims.hOf H))) j)) ps >>= fun _ =>
        checksAtJ s spec x link c₁ mids ⟨stmt, j, #[]⟩
          (coinSubJ spec c₁ mids (shaSubJ spec ((filesAt (tabsOf sts) (pointsOf spec c₁) ps).map
            (FlockComplete.dgOf spec (Claims.hOf H))) j))) := by
  rw [decisionAtS_evalJ, verifyJWithM_sub (Claims.hOf H) spec (tabsOf sts) c₁ mids j hsh hnd]

/-- **The decision accepts exactly where the verdict on the decoded proofs and the checks after it pass**, at a
statement that sets up, where each proof is shaped for its file; S17's name hypothesis comes from the setup and the
scope check (`namesDistinct_of_setup`, `publicInputs_none_of_checksAtJ`). -/
theorem decisionAtS_ok_iff (H : Claims.Msg → Claims.Msg) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    {stmt : Claims.Msg} {j : Json} {ps : Array Flock.Proof} (hset : s.setup = .ok (spec, sts))
    (hsh : ∀ (k : ℕ) (hk : k < ps.size),
      FlockVBridge.Family.ProofShaped (shapeAt (tabsOf sts) (pointsOf spec c₁) k) ps[k]) :
    ((decisionAtS s spec sts x link c₁ mids ⟨stmt, j, ps⟩).run.eval H).1 = .ok () ↔
      verifyJPWith (Flock.Hashes.ofSha512 (Claims.hOf H) : Flock.Hashes Flock.V) spec (tabsOf sts)
          (coinSubJ spec c₁ mids (shaSubJ spec ((filesAt (tabsOf sts) (pointsOf spec c₁) ps).map
            (FlockComplete.dgOf spec (Claims.hOf H))) j)) ps = .ok () ∧
        checksAtJ s spec x link c₁ mids ⟨stmt, j, #[]⟩
          (coinSubJ spec c₁ mids (shaSubJ spec ((filesAt (tabsOf sts) (pointsOf spec c₁) ps).map
            (FlockComplete.dgOf spec (Claims.hOf H))) j)) = .ok () := by
  constructor
  · intro h
    have h' := h
    rw [decisionAtS_evalJ] at h'
    obtain ⟨_, -, hc⟩ := FlockSoundness.Refine.ebind_ok h'
    rw [decisionAtS_eval H s spec sts x link c₁ mids hsh
      (namesDistinct_of_setup hset (publicInputs_none_of_checksAtJ hc))] at h
    obtain ⟨⟨⟩, hv, -⟩ := FlockSoundness.Refine.ebind_ok h
    exact ⟨hv, hc⟩
  · rintro ⟨hv, hc⟩
    rw [decisionAtS_eval H s spec sts x link c₁ mids hsh
      (namesDistinct_of_setup hset (publicInputs_none_of_checksAtJ hc)), hv]
    exact hc

end FlockDeploy
