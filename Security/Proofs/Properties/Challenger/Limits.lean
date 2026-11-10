module

public import Properties.Challenger

/-!
# v2's challenger keys and the size of what its challenger reads

Each holds of whatever v2 parses, with no hypothesis on its verdict, so of every key and record a reading of the log
takes from v2's parameters and evidence.

* **The deployed key.** v2's parameters parse only with a challenger key at `Lms.deployed` (`KEY_PARAMS`), the
  parameters the game's signature verifier reads: `paramsOfJson_deployed`, from bytes `params_deployed`, and so the key
  v2's samples are read under (`oneStageZkSource`) is at them, `source_deployed`.
* **Coins within limits.** v2's evidence parses only with its coins within `LIMITS` (`evidenceOf_within`). In coins
  within limits `l`, `drawCoins` checks at most `signaturesAt l` signatures (`signatures_le`) and SHA-512 hashes at most
  `readsAt l` strings as it reads them (`readHashed_le`, counted through `readHashed_eq`'s parts). So one v2 record's
  reading costs at most a constant at `LIMITS` and `Lms.deployed` (`record_calls_le`).
-/

@[expose] public section

namespace Properties.Challenger.Limits

open Lean (Json)
open Spot Spot.Challenger

/-! ## The deployed key -/

theorem deployedKey_params {key : ChallengerKey} (h : OneStageZk.deployedKey key = true) :
    key.params = Lms.deployed := by
  rcases key with ⟨⟨N, W, H⟩, pub⟩
  simp only [OneStageZk.deployedKey, Lms.deployed, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
  rfl

/-- **Every challenger key v2's parameters name is at `Lms.deployed`**: they name one, and `paramsOfJson` refuses it
at other parameters (`KEY_PARAMS`). -/
theorem paramsOfJson_deployed {j : Json} {p : OneStageZk.Params} (h : OneStageZk.paramsOfJson j = .ok p) :
    ∃ key label, p.base.challenger = some (key, label) ∧ key.params = Lms.deployed := by
  unfold OneStageZk.paramsOfJson at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals try contradiction
  all_goals try (rename_i ht _ _ _ _; simp [throw, throwThe, MonadExceptOf.throw] at ht; done)
  rename_i hch hk _ _ _ _ _ _
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  exact ⟨_, _, hch, deployedKey_params (by simpa using hk)⟩

/-- **v2's evidence holds its coins within `LIMITS`**: `evidenceOf` refuses them past (`LIMITS.form`). -/
theorem evidenceOf_within {b : ByteArray} {e : OneStageZk.Evidence} (h : OneStageZk.evidenceOf b = .ok e) :
    withinAt LIMITS e.base.coins = true := by
  unfold OneStageZk.evidenceOf at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals try contradiction
  rename_i hw _ _ _ _
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  exact hw

/-! ## Coins within limits -/

theorem withinAt_spec {l : Limits} {coins : Json} (h : withinAt l coins = true) :
    arrSize (getD (getD coins "session") "log") ≤ l.log ∧ arrSize (getD (getD coins "session") "hellos") ≤ l.hellos ∧
      arrSize (getD coins "ledger") ≤ l.ledger ∧ arrSize (getD coins "pins") ≤ l.pins ∧
      arrSize (getD (getD coins "segment") "openings") ≤ l.openings ∧
      ∀ os, getD (getD coins "segment") "openings" = .arr os → ∀ o ∈ os.toList, arrSize (getD o "path") ≤ l.path := by
  simp only [withinAt, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨⟨hlog, hhel⟩, hled⟩, hpins⟩, hops⟩, hpath⟩ := h
  refine ⟨hlog, hhel, hled, hpins, hops, fun os hos o ho => ?_⟩
  rw [hos] at hpath
  simp only [Array.all_eq_true, decide_eq_true_eq] at hpath
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ho
  simpa using hpath i (by simpa using hi)

/-! ## Counting -/

theorem length_flatMap_le {α β : Type} (f : α → List β) (k : Nat) (hf : ∀ a, (f a).length ≤ k) (xs : List α) :
    (xs.flatMap f).length ≤ k * xs.length := by
  induction xs with
  | nil => simp
  | cons a rest ih =>
    simp only [List.flatMap_cons, List.length_append, List.length_cons, Nat.mul_succ]
    have := hf a
    omega

theorem length_flatten_filterMap_le {α β : Type} (f : α → Option (List β)) (k : Nat) (xs : List α)
    (hf : ∀ a ∈ xs, ∀ ys, f a = some ys → ys.length ≤ k) : (xs.filterMap f).flatten.length ≤ k * xs.length := by
  induction xs with
  | nil => simp
  | cons a rest ih =>
    have ih := ih fun b hb => hf b (List.mem_cons_of_mem a hb)
    rw [List.length_cons, Nat.mul_succ]
    cases hfa : f a with
    | none => simp only [List.filterMap_cons, hfa]; omega
    | some ys =>
      simp only [List.filterMap_cons, hfa, List.flatten_cons, List.length_append]
      have := hf a (List.mem_cons_self ..) ys hfa
      omega

theorem foldl_snd_length {α β γ : Type} (f : α × List β → γ → α × List β)
    (hf : ∀ acc x, (f acc x).2.length = acc.2.length + 1) (xs : List γ) (acc : α × List β) :
    (xs.foldl f acc).2.length = acc.2.length + xs.length := by
  induction xs generalizing acc with
  | nil => simp
  | cons x rest ih => simp only [List.foldl_cons, ih, hf, List.length_cons]; omega

/-! ## What the reader hashes, piece by piece -/

theorem taggedMessage_length (tag : String) (m : Json) : (taggedMessage tag m).length ≤ 1 := by
  unfold taggedMessage
  split <;> simp

theorem domainHashed_length (scheme : String) (epoch : Nat) (fs : Families) :
    (domainHashed scheme epoch fs).length ≤ 3 := by
  unfold domainHashed
  simp only [List.length_append]
  refine Nat.le_trans (Nat.add_le_add (Nat.add_le_add (taggedMessage_length _ _) (taggedMessage_length _ _))
    (?_ : _ ≤ 1)) (by omega)
  split <;> simp

theorem leafHashed_length (scheme : String) (dom : ByteArray) (pos : Nat) (coin salt : ByteArray) :
    (leafHashed scheme dom pos coin salt).length = 4 := rfl

theorem climbFrames_length (dom : ByteArray) (cur : ByteArray) (idx d : Nat) (sibs : List ByteArray) :
    (climbFrames dom cur idx d sibs).length = sibs.length := by
  induction sibs generalizing cur idx d with
  | nil => rfl
  | cons s rest ih => simp only [climbFrames, List.length_cons, ih]

theorem openingHashed_length (c : LiveCommitment) (o : LiveOpening) (label : String) (index : Int) :
    (openingHashed c o label index).length ≤ 7 + o.path.length := by
  unfold openingHashed
  simp only [List.length_append]
  have := domainHashed_length c.scheme c.epoch c.families
  split
  · simp only [List.length_append, leafHashed_length, climbFrames_length]
    omega
  · simp only [List.length_nil]
    omega

theorem recordHashed_length (I : ByteArray) (host : String) (c : LiveCommitment) :
    (recordHashed I host c).length ≤ 2 := by
  unfold recordHashed
  simp only [List.length_append]
  have := taggedMessage_length COMMITMENT_TAG c.toJson
  split <;> simp only [List.length_cons, List.length_nil] <;> omega

theorem closeHashed_length (I : ByteArray) (host : String) (c : LiveCommitment) (session : String) (lines : Nat)
    (chain : ByteArray) (ended : Option String) : (closeHashed I host c session lines chain ended).length ≤ 3 := by
  unfold closeHashed
  simp only [List.length_append]
  have := recordHashed_length I host c
  split <;> simp only [List.length_cons, List.length_nil] <;> omega

theorem logHashed_length (lines : List String) : (logHashed lines).length = lines.length + 2 := by
  unfold logHashed
  simp only [List.length_append, List.length_cons, List.length_nil]
  rw [foldl_snd_length _ (fun _ _ => by simp)]
  simp only [List.length_nil]
  omega

theorem ledgerHashed_length (I : ByteArray) (line : Json) : (ledgerHashed I line).length ≤ 2 := by
  unfold ledgerHashed
  split
  · simp
  · split
    · simp
    · simp only [List.length_append]
      refine Nat.le_trans (Nat.add_le_add (taggedMessage_length _ _) (?_ : _ ≤ 1)) (by omega)
      split <;> simp

theorem mapM_length {α β : Type} {f : α → Option β} {xs : List α} {ys : List β} (h : xs.mapM f = some ys) :
    ys.length = xs.length := by
  induction xs generalizing ys with
  | nil => simp at h; simp [← h]
  | cons x rest ih =>
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
      Option.some.injEq] at h
    obtain ⟨_, _, zs, hzs, rfl⟩ := h
    simp [ih hzs]

theorem ofJson_path {j : Json} {o : LiveOpening} (h : LiveOpening.ofJson j = .ok o) :
    o.path.length = arrSize (getD j "path") := by
  unfold LiveOpening.ofJson at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals try contradiction
  rename_i hp
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  split at hp
  · rename_i ps hj
    rw [hj, mapM_length hp]
    simp [arrSize]
  · contradiction

/-! ## `readHashed`, restated through its parts -/

/-- The strings of an array: how `readHashed` reads a log's lines. -/
def linesOf : Json → List String
  | .arr ls => ls.toList.filterMap fun | .str s => some s | _ => none
  | _ => []

/-- What `readHashed` hashes for the segment's openings, under the record's commitment `c`. -/
def openingsHashed (c : LiveCommitment) (label : String) (segment : Json) : List ByteArray :=
  match nat? (getD segment "start"), getD segment "openings" with
  | some start, .arr os => (os.toList.zipIdx.filterMap fun (o, i) =>
      (LiveOpening.ofJson o).toOption.map fun o => openingHashed c o label ((start + i : Nat) : Int)).flatten
  | _, _ => []

/-- What `readHashed` hashes for the session. -/
def sessionHashed (I : ByteArray) (label : String) (coins : Json) : List ByteArray :=
  let session := getD coins "session"
  match ChallengerRecord.ofJson (getD session "record") with
  | .error _ => []
  | .ok r =>
    recordHashed I r.host r.commitment ++
      (match ChallengerClose.ofJson (getD session "close") with
        | .ok c => closeHashed I r.host r.commitment c.session c.lines c.chain c.ended
        | .error _ => []) ++
      logHashed (linesOf (getD session "log")) ++ logHashed (linesOf (getD session "hellos")) ++
      openingsHashed r.commitment label (getD coins "segment")

/-- What `readHashed` hashes for the ledger. -/
def ledgerLinesHashed (I : ByteArray) (coins : Json) : List ByteArray :=
  match getD coins "ledger" with
  | .arr ls => ls.toList.flatMap (ledgerHashed I)
  | _ => []

/-- What `readHashed` hashes for the pins log. -/
def pinsHashed (coins : Json) : List ByteArray :=
  match getD coins "pins" with
  | .arr _ => logHashed (linesOf (getD coins "pins"))
  | _ => []

theorem readHashed_eq (key : ChallengerKey) (label : String) (coins : Json) :
    readHashed key label coins =
      sessionHashed (ident key.pub) label coins ++ ledgerLinesHashed (ident key.pub) coins ++ pinsHashed coins := rfl

theorem linesOf_length (j : Json) : (linesOf j).length ≤ arrSize j := by
  cases j with
  | arr ls => exact Nat.le_trans (List.length_filterMap_le ..) (by simp [arrSize])
  | _ => simp [linesOf]

theorem openingsHashed_length (c : LiveCommitment) (label : String) (segment : Json) {n p : Nat}
    (hops : arrSize (getD segment "openings") ≤ n)
    (hpath : ∀ os, getD segment "openings" = .arr os → ∀ o ∈ os.toList, arrSize (getD o "path") ≤ p) :
    (openingsHashed c label segment).length ≤ n * (7 + p) := by
  unfold openingsHashed
  split
  · rename_i start os _ hos
    have hpath := hpath os hos
    rw [hos] at hops
    simp only [arrSize] at hops
    refine Nat.le_trans (length_flatten_filterMap_le _ (7 + p) _ ?_) ?_
    · rintro ⟨o, i⟩ hmem ys hys
      simp only [Option.map_eq_some_iff] at hys
      obtain ⟨o', ho', rfl⟩ := hys
      cases hj : LiveOpening.ofJson o with
      | error => simp [hj, Except.toOption] at ho'
      | ok o'' =>
        simp only [hj, Except.toOption, Option.some.injEq] at ho'
        subst ho'
        have := ofJson_path hj
        have := hpath o (List.fst_mem_of_mem_zipIdx hmem)
        have := openingHashed_length c o'' label ((start + i : Nat) : Int)
        omega
    · rw [List.length_zipIdx, Array.length_toList, Nat.mul_comm]
      exact Nat.mul_le_mul_right _ hops
  · simp

theorem sessionHashed_length (l : Limits) (I : ByteArray) (label : String) (coins : Json)
    (h : withinAt l coins = true) : (sessionHashed I label coins).length ≤ 9 + l.log + l.hellos + l.openings * (7 + l.path) := by
  obtain ⟨hlog, hhel, -, -, hops, hpath⟩ := withinAt_spec h
  unfold sessionHashed
  dsimp only
  split
  · simp
  · rename_i r _
    simp only [List.length_append, logHashed_length]
    have := recordHashed_length I r.host r.commitment
    have := linesOf_length (getD (getD coins "session") "log")
    have := linesOf_length (getD (getD coins "session") "hellos")
    have := openingsHashed_length r.commitment label (getD coins "segment") hops hpath
    have : (match ChallengerClose.ofJson (getD (getD coins "session") "close") with
        | .ok c => closeHashed I r.host r.commitment c.session c.lines c.chain c.ended
        | .error _ => []).length ≤ 3 := by
      split
      · exact closeHashed_length ..
      · simp
    omega

theorem ledgerLinesHashed_length (l : Limits) (I : ByteArray) (coins : Json) (h : withinAt l coins = true) :
    (ledgerLinesHashed I coins).length ≤ 2 * l.ledger := by
  obtain ⟨-, -, hled, -⟩ := withinAt_spec h
  unfold ledgerLinesHashed
  split
  · rename_i ls hls
    rw [hls] at hled
    simp only [arrSize] at hled
    refine Nat.le_trans (length_flatMap_le _ 2 (ledgerHashed_length I) _) ?_
    simp only [Array.length_toList]
    omega
  · simp

theorem pinsHashed_length (l : Limits) (coins : Json) (h : withinAt l coins = true) :
    (pinsHashed coins).length ≤ 2 + l.pins := by
  obtain ⟨-, -, -, hpins, -⟩ := withinAt_spec h
  unfold pinsHashed
  split
  · rw [logHashed_length]
    have := linesOf_length (getD coins "pins")
    omega
  · simp

/-! ## The counts -/

/-- At most how many signatures `drawCoins` checks in coins within `l`: the record's, the close's and one per ledger
line. -/
def signaturesAt (l : Limits) : Nat := 2 + l.ledger

/-- At most how many strings SHA-512 hashes as `drawCoins` reads coins within `l`: the record's and close's five; each
release log line, each refused hello and each pins log line, and two more per log; each opening's domain, leaf and
path; and two per ledger line. -/
def readsAt (l : Limits) : Nat := 11 + l.log + l.hellos + l.openings * (7 + l.path) + 2 * l.ledger + l.pins

/-- **`drawCoins` checks at most `signaturesAt l` signatures in coins within `l`.** -/
theorem signatures_le (l : Limits) (key : ChallengerKey) (coins : Json) (h : withinAt l coins = true) :
    (signatures key coins).length ≤ signaturesAt l := by
  obtain ⟨-, -, hled, -⟩ := withinAt_spec h
  unfold signatures signaturesAt
  simp only [List.length_append]
  repeat' split
  all_goals simp only [List.length_append, List.length_cons, List.length_nil]
  all_goals first
    | omega
    | (rename_i ls hls
       rw [hls] at hled
       simp only [arrSize] at hled
       refine Nat.le_trans (Nat.add_le_add_left (List.length_filterMap_le ..) _) ?_
       simp only [Array.length_toList]
       omega)

/-- **SHA-512 hashes at most `readsAt l` strings as `drawCoins` reads coins within `l`.** -/
theorem readHashed_le (l : Limits) (key : ChallengerKey) (label : String) (coins : Json)
    (h : withinAt l coins = true) :
    (readHashed key label coins).length ≤ readsAt l := by
  rw [readHashed_eq]
  simp only [readsAt, List.length_append]
  have := sessionHashed_length l (ident key.pub) label coins h
  have := ledgerLinesHashed_length l (ident key.pub) coins h
  have := pinsHashed_length l coins h
  omega

/-! ## A v2 record -/

/-- Every challenger key v2's parameters name, decoded from bytes, is at `Lms.deployed`. -/
theorem params_deployed {a : ByteArray} {p : OneStageZk.Params}
    (h : (Core.canonicalJson a >>= OneStageZk.paramsOfJson) = .ok p) :
    ∃ key label, p.base.challenger = some (key, label) ∧ key.params = Lms.deployed := by
  cases hj : Core.canonicalJson a with
  | error => simp [hj, bind, Except.bind] at h
  | ok j =>
    simp only [hj, bind, Except.bind] at h
    exact paramsOfJson_deployed h

/-- The key v2's samples are read under (`oneStageZkSource`) is at `Lms.deployed`. -/
theorem source_deployed {a : ByteArray} {p : OneStageZk.Params} {c : OneStage.Commitment} {d : Draws}
    (hp : (Core.canonicalJson a >>= OneStageZk.paramsOfJson) = .ok p) (hd : oneStageZkSource p c = some d) :
    d.key.params = Lms.deployed := by
  obtain ⟨key, label, hch, hk⟩ := params_deployed hp
  unfold oneStageZkSource oneStageSource at hd
  rw [hch] at hd
  cases hw : p.base.verifier.windowRoot with
  | none => rw [hw] at hd; cases hd
  | some w => rw [hw] at hd; cases hd; exact hk

/-- **One v2 record's reading costs at most a constant**: where v2 decodes the parameters `a` and the evidence `b`
and reads the samples `d` under them, the signatures `drawCoins` checks in the evidence's coins, at `L` each under the
key's parameters, and the strings SHA-512 hashes as it reads them are at most `LIMITS`' counts at `Lms.deployed`. -/
theorem record_calls_le (L : Lms.Params → Nat) {a b : ByteArray} {p : OneStageZk.Params} {c : OneStage.Commitment}
    {e : OneStageZk.Evidence} {d : Draws} (hp : (Core.canonicalJson a >>= OneStageZk.paramsOfJson) = .ok p)
    (he : OneStageZk.evidenceOf b = .ok e) (hd : oneStageZkSource p c = some d) :
    (signatures d.key e.base.coins).length * L d.key.params + (readHashed d.key d.label e.base.coins).length ≤
      signaturesAt LIMITS * L Lms.deployed + readsAt LIMITS := by
  rw [source_deployed hp hd]
  have hw := evidenceOf_within he
  exact Nat.add_le_add (Nat.mul_le_mul_right _ (signatures_le _ _ _ hw)) (readHashed_le _ _ _ _ hw)

end Properties.Challenger.Limits
