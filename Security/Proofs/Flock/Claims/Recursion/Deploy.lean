import Proofs.Flock.Claims.Recursion.DeployOuter
import Definitions.Claims.Recursion

/-!
# The compiled verifier in public-coin form (`RecDeploy.V`) and its record (`deployRec`)

The shipped recursive verifier is the inner session, then `flock-verify verify --zk` once per V* part with that part's
recursive inputs (`--rec-inner`, `--rec-part J`, `--rec-bindings`, the inner statement's files, the inner session's
record and the coin server's transcript). `RecDeploy.V` is that as one `Claims.Verifier` over Flock's live coins
(`FlockDeploy.Coin`):

* **Statement.** `x` is the canonical bytes (`RecX.encode`) of what the verifier holds before the run: the inner
  statement's bytes (a `FlockDeploy.Raw` encoding: its circuit, public and partition files), the verifier's own inner
  shape (`--rec-inner`) and bindings (`--rec-bindings`), the circuit registration, and per V* part its Flock statement
  and `--rec-rows`. `parse` (`readX`) reads the inner statement's dimensions with the shipped `RecVStar.innerOf`, the
  form off the verifier's own inner shape, and refuses unless there is one part per V* statement of that shape
  (`RecVStar.statements`). It reads no circuit commitment: the shipped verifier has no private-circuit form of the
  compiled statement.
* **Rounds.** The inner session's `k_I = IV.k (x_I)` rounds, each message answered by its live coin, then each V*
  session in order, part `s` at its V* statement (`vstar`), over `VO.k` rounds, each step the outer verifier's (`VO`,
  `verify --zk` at the recursive setup). A session that accepts at its last round is answered `.msg []`, the last one
  `.acc`; the compiled verifier refuses a session that accepts before its last round, and a session whose last round
  ended on a coin rather than a decision (`decided`).
* **V\* statements.** Part `s`'s statement is what the prover staged after the inner rounds (`Stage`: the part's public
  file, the inner session's record, the transcript), in the part's recursive flags (`staged`). Its numbers (`numsOf`:
  `VO`'s rounds and `deployZkRec`'s errors, `N₀` and weight) are to be the ones `x` fixes (`nums`, those of a staged
  statement that sets up, when one does): the compiled verifier refuses a part whose staged statement has others, and
  `vstar` reads such a part as that statement (`vstarAt`), so every V* statement `vstar` gives has `nums`' numbers.

`next_rej`, `next_coin` and `next_sess` are the steps; `next_decide_sha512` is the acceptance theorem: at a session's
decision, the compiled verifier accepts the session exactly when it is the session's last round and the deployed
`Zk.verify` at the part's recursive setup accepts the record and proof files with the custody checks
(`FlockDeploy.checks`), the part's statement being the one the prover staged (`ract_sess`). `next_acc`: the whole run
accepts only at the last session's last round, where the outer verifier accepts.

`deployRec H IV ivScope st rd` is the record the recursion's claims read, at the inner verifier `IV`, its scope check,
the stager and the outer witnesses' readers (`Readers`: the inner messages and the SHA-512 evaluations they record);
`innerView` runs `IV` at `H` on those messages and the run's first `k_I` coins.
-/

namespace RecDeploy

open Flock
open FlockDeploy (Raw Coin Final Ins encNat encBytes encStr encOpt encList Rd rdTake rdNat rdBytes rdStr rdOpt rdList
  jsonOf checks tabsOf sha512)
open Lean (Json)

/-! ## The compiled verifier's statement -/

/-- A V* part as the compiled statement holds it: its Flock statement (its public file the prover's, staged) and the
verifier's own `--rec-rows`. -/
structure PartX where
  base : Raw
  rows : Option ByteArray

/-- **What the compiled verifier holds before the run**: the inner statement's bytes, the verifier's own inner shape
(`--rec-inner`) and bindings (`--rec-bindings`), the circuit registration (`--rec-inner-registration`), and the V*
parts. -/
structure RecX where
  inner : Claims.Msg
  recInner : ByteArray
  bindings : ByteArray
  registration : Option ByteArray
  parts : List PartX

/-- The compiled statement's format string, its first field. -/
def XFORMAT : String := "flock-verify-recursion/v1"

def PartX.encode (p : PartX) : Claims.Msg := encBytes ⟨p.base.encode.toArray⟩ ++ encOpt encBytes p.rows

/-- **The compiled statement's canonical bytes.** -/
def RecX.encode (w : RecX) : Claims.Msg :=
  encStr XFORMAT ++ encNat w.inner.length ++ w.inner ++ encBytes w.recInner ++ encBytes w.bindings ++
    encOpt encBytes w.registration ++ encList PartX.encode w.parts

def PartX.read : Rd PartX := do
  let b ← rdBytes
  let some base := FlockDeploy.decode b.data.toList | failure
  let rows ← rdOpt rdBytes
  pure ⟨base, rows⟩

def RecX.read : Rd RecX := do
  if (← rdStr) ≠ XFORMAT then failure
  let n ← rdNat
  let inner ← rdTake n
  let recInner ← rdBytes
  let bindings ← rdBytes
  let registration ← rdOpt rdBytes
  let parts ← rdList PartX.read
  pure ⟨inner, recInner, bindings, registration, parts⟩

/-- **The compiled statement's decoder**: `RecX.read` on all of `x`, kept only when its encoding is `x`. -/
def decodeX (x : Claims.Msg) : Option RecX :=
  match RecX.read.run x with
  | some (w, []) => if w.encode = x then some w else none
  | _ => none

theorem decodeX_encode {x : Claims.Msg} {w : RecX} (h : decodeX x = some w) : w.encode = x := by
  unfold decodeX at h
  split at h
  · split_ifs at h with he
    cases h
    exact he
  · cases h

theorem decodeX_injective {x y : Claims.Msg} {w : RecX} (hx : decodeX x = some w) (hy : decodeX y = some w) : x = y :=
  (decodeX_encode hx).symm.trans (decodeX_encode hy)

/-! ## Its parser -/

/-- The inner statement's first table, set up as `RecVStar.innerOf` sets it up: plain leaves on the coin server's
coins. -/
def innerStmt (ri : Raw) (partition : Option Json) : Option Stmt :=
  match Stmt.setupTables RecVStar.innerTags ri.circuit ri.publicFile Coins.os 1 (partition := partition)
      (program := ri.program) with
  | .ok (_, sts) => sts[0]?
  | .error _ => none

/-- Whether the RecSparse unit range's slots lie inside the block: `layoutFactors` at the inner statement's unit range,
as `RecCoins.checkInnerRange` runs it. -/
def inBlock (i : RecVStar.Inner) (st : Option Stmt) : Bool :=
  match i.sparse, st with
  | some sp, some st =>
    (do
      let rng ← RecCoins.unitRangeOf st
      RecCoins.layoutFactors sp (Array.replicate (st.c.kLog - 6) 0) (Array.replicate (st.c.kLog - 6) 0) rng).isOk
  | _, _ => false

/-- The compiled statement as read: its fields, the inner statement's, the verifier's own inner shape and the inner
statement's shape as `RecVStar.innerOf` reads it. -/
structure XR where
  w : RecX
  innerRaw : Raw
  partition : Option Json
  shape : RecVStar.Inner
  dims : RecVStar.Inner

/-- **The compiled statement's reader**: its bytes decode, the inner statement's decode and set up (`RecVStar.innerOf`),
the verifier's own inner shape reads (`RecVStar.Inner.ofJson`), and there is one part per V* statement of that shape. -/
def readX (x : Claims.Msg) : Option XR := do
  let w ← decodeX x
  let ri ← FlockDeploy.decode w.inner
  let shape ← (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" w.recInner)).toOption
  let partition ← (ri.partition.mapM (jsonOf "partition")).toOption
  let dims ← (RecVStar.innerOf ri.circuit ri.publicFile partition ri.program).toOption
  let vs ← (RecVStar.statements shape).toOption
  if w.parts.length ≠ vs.size then none
  pure ⟨w, ri, partition, shape, dims⟩

/-- The recursion's form, off the verifier's own inner shape. -/
def formOf (r : XR) : Claims.Recursion.Form where
  m := r.shape.m
  kLog := r.shape.kLog
  regions := r.shape.regions
  claims := r.shape.claims
  holo := r.shape.holo
  sparse := r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)
  stacked := r.shape.stacked.map fun sh => (sh.m, sh.claims)
  unitRangeInBlock := inBlock r.shape (innerStmt r.innerRaw r.partition)

/-- **The claims' parser**: the inner statement's bytes, its `(m, k_log, regions)` as `RecVStar.innerOf` reads them, no
circuit commitment, and the form. -/
def parse (x : Claims.Msg) : Option Claims.Recursion.Stmt :=
  (readX x).map fun r =>
    { inner := r.w.inner, innerDims := (r.dims.m, r.dims.kLog, r.dims.regions), circ := none, form := formOf r }

open Classical in
/-- **The claims' scope check**: the statement parses, its inner statement passes the inner verifier's scope check, and
its form is candidate 3's (`InScope`). -/
noncomputable def scopeCheck (ivScope : Claims.Msg → Bool) (x : Claims.Msg) : Bool :=
  match parse x with
  | some s => ivScope s.inner && decide (Claims.Recursion.InScope s)
  | none => false

/-- The inner statement's bytes. -/
def xIOf (x : Claims.Msg) : Claims.Msg := ((parse x).map Claims.Recursion.Stmt.inner).getD []

/-- The V* parts' count. -/
def nS (x : Claims.Msg) : ℕ :=
  match readX x with
  | some r => r.w.parts.length
  | none => 0

/-! ## The V* statements -/

/-- **What the prover stages for the V* parts** after the inner rounds, read off the compiled statement and the inner
rounds as the run shows them: each part's public file, the inner session's record and the coin server's transcript. -/
structure Stage where
  pub : Claims.Msg → List (Claims.Msg × Claims.Out) → ℕ → ByteArray
  record : Claims.Msg → List (Claims.Msg × Claims.Out) → ByteArray
  transcript : Claims.Msg → List (Claims.Msg × Claims.Out) → ByteArray

/-- Part `s`'s recursive statement: its Flock statement at the staged public file, the verifier's own inner shape,
`--rec-part s`, the bindings and rows, the inner statement's files, the staged record and transcript, and the
registration. -/
def partRaw (r : XR) (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) (s : ℕ) (p : PartX) :
    RecRaw where
  base := { p.base with publicFile := st.pub x ms s }
  recInner := some r.w.recInner
  recPart := some s
  recBindings := some r.w.bindings
  recRows := p.rows
  innerCircuit := some r.innerRaw.circuit
  innerPublic := some r.innerRaw.publicFile
  innerPartition := r.innerRaw.partition
  innerCoins := none
  innerSession := some (st.record x ms)
  transcript := some (st.transcript x ms)
  registration := r.w.registration

/-- **Part `s`'s statement as staged** after the inner rounds `ms`. -/
def staged (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) (s : ℕ) : Claims.Msg :=
  match readX x with
  | some r =>
    match r.w.parts[s]? with
    | some p => (partRaw r st x ms s p).encode
    | none => []
  | none => []

/-- A V* statement's numbers: `VO`'s rounds, and `deployZkRec`'s table error, `N₀`, weight, waits and truncation. -/
noncomputable def numsOf (xO : Claims.Msg) : Claims.Recursion.Nums :=
  ⟨εtableO xO, N₀O xO, weightO xO, VO.k xO, Claims.Flock.Numbers.waits, Claims.Flock.Numbers.cap⟩

open Classical in
/-- Inner rounds at which part `s`'s staged statement sets up, when there are any. -/
noncomputable def pick (st : Stage) (x : Claims.Msg) (s : ℕ) : List (Claims.Msg × Claims.Out) :=
  if h : ∃ ms, (setupO (staged st x ms s)).isSome then Classical.choose h else []

/-- **Part `s`'s numbers at `x`**: those of its staged statement at inner rounds where it sets up, when there are
any. -/
noncomputable def nums (st : Stage) (x : Claims.Msg) (s : ℕ) : Claims.Recursion.Nums :=
  numsOf (staged st x (pick st x s) s)

open Classical in
/-- **Part `s`'s V* statement**: the staged one where its numbers are `nums`', else the one at `pick`. -/
noncomputable def vstarAt (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) (s : ℕ) : Claims.Msg :=
  if numsOf (staged st x ms s) = nums st x s then staged st x ms s else staged st x (pick st x s) s

/-- **The V* statements of a run**, read off its inner rounds. -/
noncomputable def vstar (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) : List Claims.Msg :=
  (List.range (nS x)).map (vstarAt st x ms)

theorem numsOf_vstarAt (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) (s : ℕ) :
    numsOf (vstarAt st x ms s) = nums st x s := by
  unfold vstarAt
  split_ifs with h
  · exact h
  · rfl

theorem vstar_getD (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) {s : ℕ} (hs : s < nS x) :
    (vstar st x ms).getD s [] = vstarAt st x ms s := by
  simp [vstar, List.getD_eq_getElem?_getD, hs]

theorem vstar_length (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) :
    (vstar st x ms).length = nS x := by
  simp [vstar]

/-- **Every V* statement's rounds are its numbers'.** -/
theorem vstar_map_k (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) :
    (vstar st x ms).map VO.k = (List.range (nS x)).map fun s => (nums st x s).rounds := by
  unfold vstar
  rw [List.map_map]
  congr 1
  funext s
  exact congrArg Claims.Recursion.Nums.rounds (numsOf_vstarAt st x ms s)

/-! ## The verifier -/

/-- The inner rounds as a run shows them: each message with its coin's bytes. -/
def innerPairs (kI : ℕ) (as : List Claims.Msg) (cs : List Coin) : List (Claims.Msg × Claims.Out) :=
  (as.take kI).zip ((cs.take kI).map fun c => .msg c.bytes)

/-- The session of round `i` past the inner rounds, at the sessions' rounds `ks`: its index and its round in it. -/
def locate : List ℕ → ℕ → Option (ℕ × ℕ)
  | [], _ => none
  | k :: ks, i => if i < k then some (0, i) else (locate ks (i - k)).map fun p => (p.1 + 1, p.2)

theorem locate_lt : ∀ {ks : List ℕ} {i s j : ℕ}, locate ks i = some (s, j) → s < ks.length ∧ j < ks.getD s 0
  | [], _, _, _, h => by simp [locate] at h
  | k :: ks, i, s, j, h => by
    unfold locate at h
    split_ifs at h with hi
    · cases h
      simpa using hi
    · obtain ⟨⟨s', j'⟩, hl, he⟩ := Option.map_eq_some_iff.1 h
      cases he
      obtain ⟨h1, h2⟩ := locate_lt hl
      exact ⟨by simpa using h1, by simpa using h2⟩

/-- Whether a session's rounds `as`, `cs` at `xO` ended on a decision rather than a coin. -/
def decided (xO : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : Bool :=
  match actO xO as cs with
  | .decide .. => true
  | _ => false

/-- What the compiled verifier does at the prover's messages: refuse, answer an inner round's coin, or take session
`s`'s round `j` at its V* statement `xO`, on the session's messages and coins. -/
inductive RAct where
  | rej
  | coin (c : Coin)
  | sess (s j : ℕ) (xO : Claims.Msg) (as : List Claims.Msg) (cs : List Coin)

open Classical in
/-- **The compiled verifier's step** at the prover's messages `as` and the coins `cs`: a statement that doesn't parse,
refuse; an inner round, answer its coin; past them, the round's V* session, refused where the part's staged numbers
are not `x`'s, or, at its first round, where the session before it ended on a coin. -/
noncomputable def ract (IV : Claims.Verifier Coin) (st : Stage) (x : Claims.Msg) (as : List Claims.Msg)
    (cs : List Coin) : RAct :=
  match readX x with
  | none => .rej
  | some _ =>
    if as.length ≤ IV.k (xIOf x) then
      match cs.getLast? with
      | some c => .coin c
      | none => .rej
    else
      let kI := IV.k (xIOf x)
      let ms := innerPairs kI as cs
      let xs := vstar st x ms
      let ks := xs.map VO.k
      match locate ks (as.length - 1 - kI) with
      | none => .rej
      | some (s, j) =>
        let off := as.length - 1 - j
        let off' := off - ks.getD (s - 1) 0
        if numsOf (staged st x ms s) ≠ nums st x s then .rej
        else if j = 0 ∧ 0 < s ∧
            decided (xs.getD (s - 1) []) ((as.take off).drop off') ((cs.take off).drop off') = false then .rej
        else .sess s j (xs.getD s []) (as.drop off) (cs.drop off)

/-- A session's answer as the compiled verifier gives it: a message as is; an acceptance at the session's last round
`.acc` at the last session, else `.msg []`, and before its last round a refusal; a refusal as is. -/
def sessOut (last atEnd : Bool) : Claims.Out → Claims.Out
  | .msg m => .msg m
  | .acc => if atEnd then (if last then .acc else .msg []) else .rej
  | .rej => .rej

/-- The compiled verifier's next message. -/
noncomputable def next (IV : Claims.Verifier Coin) (st : Stage) (x : Claims.Msg) (as : List Claims.Msg)
    (cs : List Coin) : Claims.OC Claims.Out :=
  match ract IV st x as cs with
  | .rej => .ret .rej
  | .coin c => .ret (.msg c.bytes)
  | .sess s j xO as' cs' =>
    (nextO xO as' cs').bind fun o => .ret (sessOut (s + 1 == nS x) (j + 1 == VO.k xO) o)

/-- **The compiled verifier in public-coin form**: `k(x)` is the inner rounds and every V* session's rounds at `x`'s
numbers. -/
noncomputable def V (IV : Claims.Verifier Coin) (st : Stage) : Claims.Verifier Coin where
  k x := IV.k (xIOf x) + ((List.range (nS x)).map fun s => (nums st x s).rounds).sum
  next := next IV st

/-! ## What the record reads off a run -/

/-- An accepted session's answer, `.msg []` before the last session, as its verifier's `.acc`. -/
def fixLast (last : Bool) (k : ℕ) (seg : List (Claims.Msg × Claims.Out)) : List (Claims.Msg × Claims.Out) :=
  if seg.length == k && !last then
    match seg.getLast? with
    | some (a, .msg []) => seg.dropLast ++ [(a, .acc)]
    | _ => seg
  else seg

/-- A session's decision, off its last answer. -/
def segOut (seg : List (Claims.Msg × Claims.Out)) : Claims.Out :=
  match seg.getLast? with
  | some (_, .acc) => .acc
  | _ => .rej

/-- **Session `s`'s view in a run's view**: its rounds' messages, the inner rounds and the earlier sessions' rounds
after the V* statements read off the view's inner rounds, its acceptance shown as its verifier's; accepted where the
run is, else its last answer's decision. -/
noncomputable def session (IV : Claims.Verifier Coin) (st : Stage) (x : Claims.Msg) (s : ℕ) (v : Claims.Seen) :
    Claims.Seen :=
  let kI := IV.k (xIOf x)
  let xs := vstar st x (v.msgs.take kI)
  let off := kI + ((xs.take s).map VO.k).sum
  let k := VO.k (xs.getD s [])
  let seg := fixLast (s + 1 == nS x) k ((v.msgs.drop off).take k)
  ⟨seg, if v.out = .acc then .acc else segOut seg⟩

/-- What the outer witnesses give: the inner session's messages, and the SHA-512 evaluations session `s`'s witness
records. -/
structure Readers where
  msgs : Claims.Msg → List Claims.Flock.Wit → List Claims.Msg
  calls : Claims.Msg → ℕ → Claims.Flock.Wit → List (Claims.Msg × Claims.Msg)

/-- **The inner session's view**: `IV`'s run at `H` of the inner messages the outer witnesses give, at the run's first
`k_I` coins. -/
def innerView (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (rd : Readers) (x : Claims.Msg) {k : ℕ}
    (r : Fin k → Coin) (ws : List Claims.Flock.Wit) : Claims.Seen :=
  (Claims.run H (Claims.replay (rd.msgs x ws)) IV (xIOf x) () fun i => Claims.coinSeq r i.val).seen

/-- **An inner round's commit string to the zero digest** under a salt: hm96-sha512's `b ‖ c` at the pinned key
(`Hm96.Default512`), `c` the oracle's answer at the salt prefix and the salt. -/
def zeroCommit (_x : Claims.Msg) (_j : ℕ) (salt : Claims.Msg) : Claims.OC Claims.Msg :=
  .query (Claims.toMsg Hm96.Default512.saltPrefix ++ salt) fun c =>
    .ret (Claims.toMsg (ByteArray.mk ((Array.range 64).map fun i =>
      (ByteArray.mk (Array.replicate 64 0)).get! i ^^^ (Hm96.Default512.mask (Claims.ofMsg salt)).get! i)) ++ c)

/-! ## The record -/

/-- **The compiled verifier**, as the record the recursion's claims read: at the oracle its readers run the inner
verifier against (`H`), the inner verifier `IV` and its scope check, the stager and the outer witnesses' readers. -/
noncomputable def deployRec (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool)
    (st : Stage) (rd : Readers) : Claims.Recursion.Deploy Coin where
  V := V IV st
  parse := parse
  scopeCheck := scopeCheck ivScope
  vstar := vstar st
  nS := nS
  session := session IV st
  innerView x r ws := innerView H IV rd x r ws
  calls x ws := (ws.mapIdx fun s w => rd.calls x s w).flatten
  nums := nums st
  zeroCommit := zeroCommit
  saltLen := Hm96.sha512.saltBytes

theorem xI_deployRec (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool)
    (st : Stage) (rd : Readers) (x : Claims.Msg) :
    Claims.Recursion.xI (deployRec H IV ivScope st rd) x = xIOf x := rfl

/-! ## The verifier's steps against `H` -/

section Steps

variable {IV : Claims.Verifier Coin} {st : Stage} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}

theorem next_rej (H : Claims.Msg → Claims.Msg) (h : ract IV st x as cs = .rej) :
    (next IV st x as cs).eval H = (.rej, []) := by
  simp only [next, h, Claims.OC.eval]

/-- **An inner round is answered by its coin, with no query.** -/
theorem next_coin (H : Claims.Msg → Claims.Msg) {c : Coin} (h : ract IV st x as cs = .coin c) :
    (next IV st x as cs).eval H = (.msg c.bytes, []) := by
  simp only [next, h, Claims.OC.eval]

/-- **A session's round is the outer verifier's**, its answer as `sessOut` gives it and its oracle log the outer
verifier's. -/
theorem next_sess (H : Claims.Msg → Claims.Msg) {s j : ℕ} {xO : Claims.Msg} {as' : List Claims.Msg} {cs' : List Coin}
    (h : ract IV st x as cs = .sess s j xO as' cs') :
    (next IV st x as cs).eval H =
      (sessOut (s + 1 == nS x) (j + 1 == VO.k xO) ((nextO xO as' cs').eval H).1, ((nextO xO as' cs').eval H).2) := by
  simp only [next, h, Claims.OC.eval_bind, Claims.OC.eval, List.append_nil]

/-- **A session's round is part `s`'s, at its staged statement**: `s` is a part, `j` a round of its outer verifier,
and the V* statement is the one the prover staged after the inner rounds. -/
theorem ract_sess {s j : ℕ} {xO : Claims.Msg} {as' : List Claims.Msg} {cs' : List Coin}
    (h : ract IV st x as cs = .sess s j xO as' cs') :
    s < nS x ∧ j < VO.k xO ∧ xO = staged st x (innerPairs (IV.k (xIOf x)) as cs) s := by
  unfold ract at h
  split at h
  · cases h
  split_ifs at h with hk
  · split at h <;> cases h
  dsimp only at h
  split at h
  · cases h
  rename_i s' j' hl
  split_ifs at h with hn hd
  obtain ⟨rfl, rfl, rfl, rfl, rfl⟩ := RAct.sess.inj h
  obtain ⟨hs, hj⟩ := locate_lt hl
  rw [List.length_map, vstar_length] at hs
  have he : (vstar st x (innerPairs (IV.k (xIOf x)) as cs))[s']? =
      some (vstarAt st x (innerPairs (IV.k (xIOf x)) as cs) s') := by
    simp [vstar, hs]
  refine ⟨hs, ?_, ?_⟩
  · rw [List.getD_eq_getElem?_getD, List.getElem?_map, he] at hj
    rw [List.getD_eq_getElem?_getD, he]
    simpa using hj
  · rw [vstar_getD st x _ hs]
    simp [vstarAt, not_not.1 hn]

/-- **The compiled verifier accepts a run only at the last session's last round**, where the outer verifier accepts. -/
theorem next_acc (H : Claims.Msg → Claims.Msg) (h : ((next IV st x as cs).eval H).1 = .acc) :
    ∃ s j xO as' cs', ract IV st x as cs = .sess s j xO as' cs' ∧ s + 1 = nS x ∧ j + 1 = VO.k xO ∧
      ((nextO xO as' cs').eval H).1 = .acc := by
  cases hr : ract IV st x as cs with
  | rej => rw [next_rej H hr] at h; cases h
  | coin c => rw [next_coin H hr] at h; cases h
  | sess s j xO as' cs' =>
    rw [next_sess H hr] at h
    refine ⟨s, j, xO, as', cs', rfl, ?_⟩
    simp only at h
    cases ho : ((nextO xO as' cs').eval H).1 with
    | msg m => rw [ho] at h; cases h
    | rej => rw [ho] at h; cases h
    | acc =>
      rw [ho] at h
      simp only [sessOut] at h
      split_ifs at h with h1 h2
      exact ⟨by simpa using h2, by simpa using h1, rfl⟩

/-- **The acceptance theorem: at SHA-512 a session's decision is the shipped recursive check's.** Where session `s`'s
outer verifier decides, the compiled verifier accepts the session (`.acc` at the last session, `.msg []` before it)
exactly when this is the session's last round and the deployed `flock-verify verify --zk` with part `s`'s recursive
inputs, `Zk.verify` at the statement's recursive setup (`actO_decide`, `RecIns.setup`), accepts the record and proof
files with the custody checks; else it refuses. The statement is the part's as the prover staged it (`ract_sess`). -/
theorem next_decide_sha512 {s j : ℕ} {xO : Claims.Msg} {as' : List Claims.Msg} {cs' : List Coin} {i : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : ract IV st x as cs = .sess s j xO as' cs')
    (ha : actO xO as' cs' = .decide i spec sts link c₁ mids f) :
    ((next IV st x as cs).eval sha512).1 = (if s + 1 = nS x then .acc else .msg []) ↔
      j + 1 = VO.k xO ∧ Zk.verify spec (tabsOf sts) f.record f.proofs = .ok () ∧
        checks true i spec xO link c₁ mids f = .ok () := by
  rw [next_sess sha512 h]
  simp only
  rcases nextO_decide_out sha512 ha with ho | ho
  · have hv := (nextO_decide_sha512 ha).1 ho
    rw [ho]
    by_cases hj : j + 1 = VO.k xO <;> by_cases hl : s + 1 = nS x <;> simp [sessOut, hj, hl, hv]
  · have hv : ¬ (Zk.verify spec (tabsOf sts) f.record f.proofs = .ok () ∧
        checks true i spec xO link c₁ mids f = .ok ()) := fun hv => by
      have := (nextO_decide_sha512 ha).2 hv
      rw [ho] at this
      cases this
    rw [ho]
    constructor
    · intro h'
      by_cases hl : s + 1 = nS x <;> simp [sessOut, hl] at h'
    · rintro ⟨-, hv'⟩
      exact absurd hv' hv

end Steps

end RecDeploy
