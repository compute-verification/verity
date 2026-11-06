import Proofs.Flock.Claims.Recursion.DeployInputs
import Proofs.Flock.Claims.NumbersDeploy

/-!
# The outer verifier: `verify --zk` with the recursive inputs, in public-coin form

`VO` is the shipped verifier the compiled verifier runs once per V* part (`flock-verify verify --zk` with its recursive
flags) as a `Claims.Verifier`, built as `FlockDeploy.V true` is: the same messages and coins (the link message, the
rounds, the last message, each round answered by its live coin) and the same decision (`FlockDeploy.decision`: the
shipped verdict over `OC`, `Zk.verifyWithM Hashes.oc`, then the custody checks), at the recursive setup
(`RecIns.setup`) where `FlockDeploy.V` has Flock's (`Ins.setup`), which refuses every V* statement; then the verdict's
last refusals (`fullDrawO`: `RecSetup.fullDraw` at the record's draw, which refuses a recursive session without
`--rec-inner` and, under it, a draw short of every unit; then `RecSetup.innerDrawn`, which refuses a session over an
inner session whose record holds no draw; then `RecSetup.denseOnly`, which refuses a holo or sparse inner shape), which
`FlockDeploy.V` doesn't run.

`nextO_rej`, `nextO_coin` and `nextO_decide` are its three steps; at the SHA-512 oracle a decision accepts exactly when
the deployed `Zk.verify` accepts the record and proof files at the recursive setup's tables, the custody checks pass and
so do the verdict's last refusals (`fullDrawO`, `nextO_decide_sha512`).

`deployZkRec rel` is the record Flock's claims read at that verifier: `deployZk rel`'s code, decoder spec and numbers,
each read at the recursive setup's tables (`Readable.tableOf`, `colsO`, `εtableO`, `N₀O`, `weightO`). `N₀O` sums
every table's level-0 codeword length, as `FlockDeploy.N₀ true` does, so the tables' words together are no longer
than it (`sum_wordLenJO_le`, the `hN` a masked KS proof reads).
-/

namespace RecDeploy

open Flock
open FlockDeploy (Coin Act Final Ins linkOf finalOf decision outOf rounds checks tabsOf verdictWith sha512 hOf_sha512
  lastFinal liveOf saltLenOf tableErr tableLen tableWeight)

local notation "𝔽" => FlockLevel3.GF128

/-! ## The verifier -/

/-- `FlockDeploy.act`'s step at a session's setup: the last message of tag `0`, answer its coin; of tag `1`, after the
link message, decide; anything else, refuse. -/
def actAt (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (as : List Claims.Msg) (cs : List Coin) : Act :=
  match as.getLast?, cs.getLast? with
  | some (0 :: _), some c => .coin c
  | some a, _ =>
    match finalOf a, as, cs with
    | some f, a₁ :: rest, c₁ :: crest =>
      match linkOf a₁ with
      | some link => .decide s spec sts link c₁ ((rest.dropLast.map List.tail).zip crest.dropLast) f
      | none => .rej
    | _, _, _ => .rej
  | none, _ => .rej

/-- **The outer verifier's step**: no recursive setup, refuse; else `FlockDeploy.act`'s step at it. -/
def actO (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : Act :=
  match setupO x with
  | none => .rej
  | some (s, spec, sts) => actAt s.base spec sts as cs

/-- The verifier's own inner shape (`--rec-inner`) at the statement, where it reads. -/
def innerOfO (x : Claims.Msg) : Option RecVStar.Inner := (insO x).bind (·.inner)

/-- Whether the statement is a recursive session (`--rec-part`, `--rec-bindings` or `--rec-rows`), where it reads. -/
def recOfO (x : Claims.Msg) : Bool :=
  match insO x with
  | some s => s.part.isSome || s.bindings.isSome || s.rows.isSome
  | none => false

/-- Check 5's source at the statement (`RecCoins.filesInput`'s, with the inner session's record), where it reads. -/
def srcOfO (x : Claims.Msg) : Option RecCoins.Source := (insO x).bind (·.src)

/-- **The verdict's last refusals**, as `verifyCmd` calls them: `RecSetup.fullDraw` at the record's draw (a recursive
session without the verifier's inner shape, `RecSetup.noInner`, and under that shape a V* session whose record's draw is
not every unit of its population), then `RecSetup.innerDrawn` (a session over an inner session whose record holds no
draw), then `RecSetup.denseOnly` (a session at a holo or sparse inner shape, which `RecursiveKS` doesn't cover). -/
def fullDrawO (x : Claims.Msg) (f : Final) : Bool :=
  (do
    RecSetup.fullDraw (innerOfO x) (recOfO x) (Witness.recordDraw f.record) f.record
    RecSetup.innerDrawn (srcOfO x)
    RecSetup.denseOnly (innerOfO x)).isOk

/-- Where the verdict's last refusals pass, so does `RecSetup.denseOnly`, the last of them. -/
theorem fullDrawO_denseOnly {x : Claims.Msg} {f : Final} (h : fullDrawO x f = true) :
    RecSetup.denseOnly (innerOfO x) = .ok () := by
  unfold fullDrawO at h
  cases hd : RecSetup.denseOnly (innerOfO x) with
  | ok u => rfl
  | error e =>
    exfalso
    cases ha : RecSetup.fullDraw (innerOfO x) (recOfO x) (Witness.recordDraw f.record) f.record <;>
      cases hb : RecSetup.innerDrawn (srcOfO x) <;>
      simp [ha, hb, hd, bind, Except.bind, Except.isOk, Except.toBool] at h

/-- Where the verdict's last refusals pass, so does `RecSetup.fullDraw`, the first of them. -/
theorem fullDrawO_fullDraw {x : Claims.Msg} {f : Final} (h : fullDrawO x f = true) :
    (RecSetup.fullDraw (innerOfO x) (recOfO x) (Witness.recordDraw f.record) f.record).isOk = true := by
  unfold fullDrawO at h
  cases hf : RecSetup.fullDraw (innerOfO x) (recOfO x) (Witness.recordDraw f.record) f.record with
  | error e =>
    rw [hf] at h
    first | cases h | exact absurd h (by simp [bind, Except.bind, Except.isOk, Except.toBool])
  | ok u => rfl

/-- The outer verifier's next message: at a decision, the decision, then the verdict's last refusals (`fullDrawO`), which
make no SHA-512 call. -/
def nextO (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : Claims.OC Claims.Out :=
  match actO x as cs with
  | .rej => .ret .rej
  | .coin c => .ret (.msg c.bytes)
  | .decide s _ _ link c₁ mids f =>
    (outOf (decision true s x link c₁ mids f)).bind fun o => .ret (if fullDrawO x f then o else .rej)

/-- **The outer verifier in public-coin form**: `k(x)` is the link message, the session's recorded rounds and the last
message (one round where the statement does not set up). -/
def VO : Claims.Verifier Coin where
  k x := match setupO x with
    | some (_, spec, sts) => 2 + rounds true spec sts
    | none => 1
  next := nextO

/-! ## Its steps against `H` -/

theorem nextO_rej (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (h : actO x as cs = .rej) : (nextO x as cs).eval H = (.rej, []) := by
  simp only [nextO, h, Claims.OC.eval]

theorem nextO_coin (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {c : Coin} (h : actO x as cs = .coin c) : (nextO x as cs).eval H = (.msg c.bytes, []) := by
  simp only [nextO, h, Claims.OC.eval]

/-- At a decision, the step is the decision's, refused where the verdict's last refusals fail, with the decision's oracle
log. -/
theorem nextO_decide_eval (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : actO x as cs = .decide s spec sts link c₁ mids f) :
    (nextO x as cs).eval H =
      (if fullDrawO x f then ((outOf (decision true s x link c₁ mids f)).eval H).1 else .rej,
        ((outOf (decision true s x link c₁ mids f)).eval H).2) := by
  simp only [nextO, h, Claims.OC.eval_bind, Claims.OC.eval, List.append_nil]

/-- **The decision itself against `H`**: it accepts exactly when Flock's setup with SHA-512 replaced by `H` is `.ok`,
and at its session `Zk.verifyWith (hOf H)` and the custody checks are `.ok`. -/
theorem outO_decide (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (s : Ins) (link : List ByteArray × List ByteArray)
    (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) :
    ((outOf (decision true s x link c₁ mids f)).eval H).1 = .acc ↔
      ∃ spec' sts', FlockDeploy.setupWith (Claims.hOf H) s.read = .ok (spec', sts') ∧
        verdictWith true (Claims.hOf H) spec' (tabsOf sts') f.record f.proofs = .ok () ∧
          checks true s spec' x link c₁ mids f = .ok () := by
  obtain ⟨he, -⟩ := FlockDeploy.decision_eval H true s x link c₁ mids f
  simp only [outOf, Claims.OC.eval_bind, Claims.OC.eval]
  rw [he]
  cases hs : FlockDeploy.setupWith (Claims.hOf H) s.read with
  | error e => simp [bind, Except.bind]
  | ok p =>
    obtain ⟨spec', sts'⟩ := p
    cases hv : verdictWith true (Claims.hOf H) spec' (tabsOf sts') f.record f.proofs with
    | error e => simp [bind, Except.bind, hv]
    | ok u =>
      cases hc : checks true s spec' x link c₁ mids f with
      | error e => simp [bind, Except.bind, hv, hc]
      | ok u' => simp [bind, Except.bind, hv, hc]

/-- **The outer verifier's decision against `H`**: it accepts exactly when the decision does (`outO_decide`) and the
verdict's last refusals pass. -/
theorem nextO_decide (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : actO x as cs = .decide s spec sts link c₁ mids f) :
    ((nextO x as cs).eval H).1 = .acc ↔
      (∃ spec' sts', FlockDeploy.setupWith (Claims.hOf H) s.read = .ok (spec', sts') ∧
        verdictWith true (Claims.hOf H) spec' (tabsOf sts') f.record f.proofs = .ok () ∧
          checks true s spec' x link c₁ mids f = .ok ()) ∧ fullDrawO x f = true := by
  rw [nextO_decide_eval H h, ← outO_decide]
  by_cases hf : fullDrawO x f = true <;> simp [hf]

/-- A decision accepts or rejects. -/
theorem nextO_decide_out (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : actO x as cs = .decide s spec sts link c₁ mids f) :
    ((nextO x as cs).eval H).1 = .acc ∨ ((nextO x as cs).eval H).1 = .rej := by
  rw [nextO_decide_eval H h]
  simp only [outOf, Claims.OC.eval_bind, Claims.OC.eval]
  split
  · split <;> simp
  · simp

/-- **The decision's oracle log**: Flock's setup's `setupCallsOf (hOf H)` entries, then, where it is `.ok`, the
shipped verdict's `Zk.shaCalls (hOf H)` at its session. -/
theorem nextO_decide_log (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : actO x as cs = .decide s spec sts link c₁ mids f) :
    ((nextO x as cs).eval H).2.length =
      FlockDeploy.setupCallsOf (Claims.hOf H) s.read +
        match FlockDeploy.setupWith (Claims.hOf H) s.read with
        | .ok (spec', sts') => FlockDeploy.shaCallsOf true (Claims.hOf H) spec' (tabsOf sts') f.record f.proofs
        | .error _ => 0 := by
  obtain ⟨-, hl⟩ := FlockDeploy.decision_eval H true s x link c₁ mids f
  rw [nextO_decide_eval H h]
  simp only [outOf, Claims.OC.eval_bind, Claims.OC.eval, List.append_nil]
  exact hl

/-! ## At SHA-512 -/

/-- A decision is made only on a statement that sets up, at its recursive setup. -/
theorem actO_decide {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)}
    {f : Final} (h : actO x as cs = .decide s spec sts link c₁ mids f) :
    ∃ i, setupO x = some (i, spec, sts) ∧ i.base = s := by
  unfold actO at h
  split at h
  · cases h
  · rename_i i spec' sts' hs
    unfold actAt at h
    repeat' split at h
    all_goals first | (cases h; done) | (cases h; exact ⟨i, hs, rfl⟩)

/-- **At SHA-512 the shipped verdict is the deployed `Zk.verify`**, at every session the statement's recursive setup
sets up. -/
theorem verdictWithO_sha512 {x : Claims.Msg} {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (h : setupO x = some (i, spec, sts)) (record : String) (proofs : Array ByteArray) :
    verdictWith true (Claims.hOf sha512) spec (tabsOf sts) record proofs =
      Zk.verify spec (tabsOf sts) record proofs := by
  have hr := RecIns.setup_read (setupO_some h).2
  rw [hOf_sha512]
  exact (Zk.verify_eq_verifyWith_of_setupTables _ _ _ _ _ _ _ _ _ _ _ _ _ hr record proofs).symm

/-- On a decision, Flock's setup at the SHA-512 oracle is the recursive setup's session. -/
theorem actO_decide_setupWith {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)}
    {f : Final} (h : actO x as cs = .decide s spec sts link c₁ mids f) :
    FlockDeploy.setupWith (Claims.hOf sha512) s.read = .ok (spec, sts) := by
  obtain ⟨i, hi, rfl⟩ := actO_decide h
  rw [hOf_sha512, FlockDeploy.setupWith_sha512]
  exact RecIns.setup_read (setupO_some hi).2

/-- **The outer verifier's decision at SHA-512**: it accepts exactly when the deployed `verify --zk` with the recursive
inputs (`Zk.verify` at the recursive setup's tables) accepts the record and proof files, the custody checks pass, and
the verdict's last refusals pass. -/
theorem nextO_decide_sha512 {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)}
    {f : Final} (h : actO x as cs = .decide s spec sts link c₁ mids f) :
    ((nextO x as cs).eval sha512).1 = .acc ↔
      Zk.verify spec (tabsOf sts) f.record f.proofs = .ok () ∧ checks true s spec x link c₁ mids f = .ok () ∧
        fullDrawO x f = true := by
  obtain ⟨i, hi, -⟩ := actO_decide h
  rw [nextO_decide sha512 h, actO_decide_setupWith h, ← verdictWithO_sha512 hi]
  constructor
  · rintro ⟨⟨spec', sts', he, hv, hc⟩, hf⟩
    cases he
    exact ⟨hv, hc, hf⟩
  · rintro ⟨hv, hc, hf⟩
    exact ⟨⟨spec, sts, rfl, hv, hc⟩, hf⟩

/-! ## The columns a run opens -/

/-- **What the column reader takes of a run's table `j`**, as `FlockDeploy.runOf true` takes it, at the recursive
setup. -/
def runOfO (s : Claims.Seen) (j : ℕ) : Option Claims.Flock.ZkRun := do
  let f ← lastFinal s
  let (_, spec, sts) ← setupO f.stmt
  let st ← sts[j]?
  let tab := Setup.ofCircuit st
  let lv ← (liveOf spec f.record).toOption
  let extras ← (tab.extra lv.points).toOption
  pure { m := tab.m, kLog := tab.kLog, claims := 2 + extras.size, hashLen := tab.merkle.digestLen,
         saltLen := saltLenOf true tab,
         reps := [(lv.streams.getD (2 * j) #[], f.proofs.getD (2 * j) ByteArray.empty),
                  (lv.streams.getD (2 * j + 1) #[], f.proofs.getD (2 * j + 1) ByteArray.empty)] }

/-- **The opened level-0 columns of a run's table `j`**: its two reps' (`zkRunCols`); none where the run's last message
does not set up table `j`. -/
def colsO (s : Claims.Seen) (j : ℕ) : List (ℕ × List 𝔽) :=
  match runOfO s j with
  | some r => Claims.Flock.zkRunCols r
  | none => []

theorem colsO_nodup (s : Claims.Seen) (j : ℕ) : ((colsO s j).map Prod.fst).Nodup := by
  unfold colsO
  split
  · exact Claims.Flock.zkRunCols_nodup _
  · exact List.nodup_nil

/-! ## The numbers -/

/-- The session's tables with its spec, at the recursive setup; none where the statement does not set up. -/
def tablesO (x : Claims.Msg) : List (SessionSpec × Stmt) :=
  match setupO x with
  | some (_, spec, sts) => sts.toList.map fun st => (spec, st)
  | none => []

/-- **`ε_table`**: the session's tables' masked errors, summed (`FlockDeploy.tableErr true`). -/
noncomputable def εtableO (x : Claims.Msg) : ℝ := ((tablesO x).map fun p => tableErr true p.1 p.2).sum

/-- **`N₀`**: the session's tables' masked level-0 codeword lengths, summed (`FlockDeploy.tableLen true`), as
`FlockDeploy.N₀ true` sums them: table 0's `2^{d₀}` at one table. -/
noncomputable def N₀O (x : Claims.Msg) : ℝ := ((tablesO x).map fun p => tableLen true p.2).sum

/-- **Table `j`'s word's length**: its masked level-0 codeword length `2^d`; `0` past the statement's tables. -/
def wordLenJO (x : Claims.Msg) (j : ℕ) : ℕ :=
  match (tablesO x)[j]? with
  | some (_, st) => 2 ^ (Claims.Flock.zkLevel0 st.m).d
  | none => 0

private theorem sum_range_getD_le {α : Type} (g : α → ℕ) : ∀ (l : List α) (n : ℕ),
    ∑ j ∈ Finset.range n, (l[j]?.map g).getD 0 ≤ (l.map g).sum
  | [], n => by simp
  | a :: l, 0 => by simp
  | a :: l, n + 1 => by
    rw [Finset.sum_range_succ']
    simp only [List.getElem?_cons_succ, List.getElem?_cons_zero, Option.map_some, Option.getD_some, List.map_cons,
      List.sum_cons]
    have := sum_range_getD_le g l n
    omega

/-- **The words together are no longer than `N₀O`** (`FlockDeploy.KS.sum_wordLenJ_le` at the recursive setup): the
`hN` of `Claims.Flock.oneWait_of_tableBoundJR` at `deployZkRec`. -/
theorem sum_wordLenJO_le (x : Claims.Msg) (n : ℕ) : ((∑ j : Fin n, wordLenJO x j : ℕ) : ℝ) ≤ N₀O x := by
  set g : SessionSpec × Stmt → ℕ := fun p => 2 ^ (Claims.Flock.zkLevel0 p.2.m).d with hg
  have hw : ∀ j, wordLenJO x j = ((tablesO x)[j]?.map g).getD 0 := by
    intro j
    unfold wordLenJO
    cases (tablesO x)[j]? <;> rfl
  have hN : N₀O x = (((tablesO x).map g).sum : ℕ) := by
    unfold N₀O
    rw [Nat.cast_list_sum, List.map_map]
    congr 1
    refine List.map_congr_left fun p _ => ?_
    simp [hg, tableLen]
  rw [hN, Fin.sum_univ_eq_sum_range (fun j => wordLenJO x j) n]
  simp only [hw]
  exact_mod_cast sum_range_getD_le g (tablesO x) n

theorem N₀O_nonneg (x : Claims.Msg) : 0 ≤ N₀O x := by
  unfold N₀O
  exact List.sum_nonneg fun a ha => by
    obtain ⟨p, -, rfl⟩ := List.mem_map.1 ha
    unfold tableLen
    positivity

/-- **The weight `C` of `γ`**: the session's tables' masked weights, summed (`FlockDeploy.tableWeight true`). -/
noncomputable def weightO (x : Claims.Msg) : ℝ := ((tablesO x).map fun p => tableWeight true p.2).sum

theorem εtableO_nonneg (x : Claims.Msg) : 0 ≤ εtableO x := by
  unfold εtableO
  exact List.sum_nonneg fun a ha => by
    obtain ⟨p, -, rfl⟩ := List.mem_map.1 ha
    exact FlockDeploy.tableErr_nonneg true p.1 p.2

/-! ## The instance -/

/-- **The outer verifier**, as the record Flock's claims read: `deployZk rel`'s code and numbers at the recursive
setup, with `deployZk`'s honest prover and completeness error. -/
noncomputable def deployZkRec (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Deploy Coin (List 𝔽) Claims.Flock.ZkTrace where
  zk := true
  V := VO
  parse := Readable.parseOf rel readOfO
  scopeCheck := Readable.scopeCheckOf readOfO
  table := Readable.tableOf rel readOfO
  cols := colsO
  enc := Claims.Flock.zkEncD
  len := Claims.Flock.zkLenD
  θ := Claims.Flock.zkThetaD
  dec := Claims.GS.zkCountedRD
  read := Claims.Flock.readTables Claims.Flock.zkReadD
  εtable := εtableO
  N₀ := N₀O
  weight := weightO
  waits _ := Claims.Flock.Numbers.waits
  cap _ := Claims.Flock.Numbers.cap
  Tape := FlockComplete.TapeZk
  P := FlockComplete.ZkTables.honestPR
  δc := FlockComplete.ZkTables.δc

/-- **`ScopeMatches` for the outer verifier** (C's `scopeMatches` at its parser and scope check). -/
theorem scopeMatches_zkRec (rel : Readable.Parsed → Readable.Relation) : Claims.Flock.ScopeMatches (deployZkRec rel) :=
  Readable.scopeMatches (deployZkRec rel) rel readOfO rfl rfl (fun _ _ h => readOfO_zk h)

end RecDeploy
