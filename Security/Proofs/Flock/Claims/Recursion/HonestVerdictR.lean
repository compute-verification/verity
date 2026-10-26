import Proofs.Flock.Claims.Recursion.NextRTables
import Proofs.Flock.Claims.Complete.Replay.ProverOk
import Proofs.Flock.Claims.Recursion.DeployedR

/-!
# The honest run passes `nextR`'s checks

`nextR_sound` makes the deployed recursive verifier's last step accept from its reading (`ReadOkR`) and its five checks
(`VerdictR`: the setup, the proofs' shapes, the rep streams' names, the verdict on the decoded proofs at the reader's
record `recJ`, and the checks after it). Here they hold at the honest run: a plain completeness run
(`FlockComplete.Run`), whose prover sends round 0 from its tables' level-0 caps, each rep's rounds from its proof at
its ring switch's batched value (`Run.recTs`), and a last message the reader reads as the files' proofs.

* **The honest decision** (`honest_decR`): `decR` decides on the honest messages (`Run.recMsgs`) at `Run.recDec`, the
  link of the caps, the proofs' framed bytes and the reader's record at them. This is `sessionRun_decides_coins` with
  the decision `ReadsBack` names.
* **The checks** (`HonestSetup`, `HonestShaped`, `HonestNames`, `HonestVerdict`, `HonestAfter`, `HonestReads`): each of
  `VerdictR.At` and `ReadOkR`'s field read at `Run.recDec`. The setup is the case's (`honestSetup`), the names are
  distinct at a statement with no public inputs (`honestNames`), and each file decodes to a proof of its shape
  (`honestShaped`).
* **Acceptance** (`honest_holdsR_of`, `honest_nextR_of`): the checks at `Run.recDec` are `holdsR` at the honest
  messages, and with the reading they make `nextR` accept.
* **What the last message carries** (`readPs_opensOf`, `decR_opensOf`): the deployed reader (`rdR`) returns each proof's
  openings only (`opensOf`). `readRep` takes from the last message's proof exactly the fields no frame absorbs, which
  are its openings (`absorbedFrom_opensOf`), so `decR` decides the same at the proofs or their openings.
-/

namespace FlockComplete.Run

open Flock FlockDeploy

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-! ## What the honest prover sends -/

/-- The first coin, which the link points are read at. -/
def recC₁ (o : Run H rel) : Coin := Claims.coinSeq o.r 0

/-- Each table's level-0 cap, as the session's files carry it. -/
def recCaps (o : Run H rel) : Array (Array ByteArray) := fileCaps o.spec o.sts o.recC₁ o.proofs

/-- The reps' proofs, as the session's files decode. -/
def recPs (o : Run H rel) : Array Flock.Proof := filePs o.spec o.sts o.recC₁ o.proofs

/-- **The honest prover's batched value at rep `k`**: the ring switch's `T` at table `k / 2`'s rep `k % 2`. -/
def recTs (o : Run H rel) (k : ℕ) : F128 := (o.sw (k / 2) (k % 2)).1.1

/-- Rep `k`'s context at the honest round 0 and the coins `cs`. -/
def recCtx (o : Run H rel) (cs : List Coin) (k : ℕ) : RepCtx :=
  repCtx o.spec o.sts o.recC₁ (capRows o.recCaps) cs.tail.dropLast k

/-- **The honest messages**: round 0 the caps' rows, each rep's rounds its proof's rows at its batched value, then the
last message. -/
def recMsgs (o : Run H rel) (cs : List Coin) (last : Claims.Msg) : List Claims.Msg :=
  honestMsgs o.recCaps ((List.range o.recPs.size).map fun k => repRows (o.recCtx cs k) o.recPs[k]! (o.recTs k)) last

/-- The link the reader takes from round 0: each table's `root_B` its cap's nodes joined, no public words. -/
def recLink (o : Run H rel) : List ByteArray × List ByteArray :=
  ((o.recCaps.toList.map Array.toList).map (·.foldl (· ++ ·) ByteArray.empty),
    o.sts.toList.map fun _ => ByteArray.empty)

/-- The round messages the reader takes: the honest proofs' framed bytes. -/
def recFrames (o : Run H rel) (cs : List Coin) : List Claims.Msg :=
  ((List.range o.recPs.size).map fun k => framesOf (o.recCtx cs k) o.recPs[k]! (o.recTs k)).flatten

/-- **The honest decision**: the statement's setup, the honest link, the first coin, the framed rounds with their coins,
and the reader's record (`recJ`) at them with the files' proofs. -/
def recDec (o : Run H rel) (cs : List Coin) : DecR :=
  ⟨o.s, o.spec, o.sts, o.recLink, o.recC₁, (o.recFrames cs).zip cs.tail.dropLast,
    ⟨o.x, recJ o.spec o.sts o.recLink o.recC₁ (o.recFrames cs) cs.tail.dropLast o.recPs, o.recPs⟩⟩

/-- **The deployed verifier's coins at the run**: the run's coins in order, at least the reps' rounds and two. -/
def CoinsAt (o : Run H rel) (cs : List Coin) : Prop :=
  (∀ i < cs.length, cs.getD i default = Claims.coinSeq o.r i) ∧ (repLens o.spec o.sts).sum + 2 ≤ cs.length

theorem CoinsAt.head {o : Run H rel} {cs : List Coin} (h : o.CoinsAt cs) : cs.head? = some o.recC₁ := by
  cases cs with
  | nil => have := h.2; simp at this
  | cons c cs' =>
    have := h.1 0 (by simp)
    simp only [List.getD_cons_zero] at this
    simp [this, recC₁]

/-- **A plain run's tables are `TableOk`**, with at most 2,048 extra claims at the first coin's link points. -/
theorem recTabsOk (o : Run H rel) : ∀ (i : ℕ) (hi : i < o.sts.size), JsonRT.TableOk o.sts[i] ∧
    ∀ e, o.sts[i].extraClaims (pointsOf o.spec o.recC₁) = .ok e → e.size ≤ 2048 := by
  obtain ⟨st, hst, hok⟩ := o.tableOk
  obtain ⟨st', hst', hsp, -, hreg⟩ := o.schemeOk
  rw [hst] at hst'
  obtain rfl : st = st' := by simpa using hst'
  have key : JsonRT.TableOk st ∧ ∀ e, st.extraClaims (pointsOf o.spec o.recC₁) = .ok e → e.size ≤ 2048 := by
    refine ⟨hok, fun e he => ?_⟩
    rw [JsonRT.extraClaims_size he]
    simp only [pointsOf, Array.size_map, Array.size_range, hsp]
    show st.regions.size * 2 ≤ 2048
    omega
  intro i hi
  have hi0 : i = 0 := by simp [hst] at hi; omega
  subst hi0
  have e : o.sts[0] = st := by simp [hst]
  rw [e]
  exact key

/-- The case's setup at `hOf H`. -/
theorem recSetup (o : Run H rel) : setupWith (Claims.hOf H) o.s.read = .ok (o.spec, o.sts) :=
  o.ready_parts.2.2.2.1

/-- The case's statement carries no public inputs. -/
theorem recPublicInputs (o : Run H rel) : o.s.read.publicInputs = none := by
  have hck := o.scope_ok
  unfold Readable.Read.check at hck
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hck).1
  exact Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])

end FlockComplete.Run

namespace FlockDeploy

open Flock FlockComplete

section

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-! ## The last message's openings -/

open RecDeploy.DeployedR (opensOf)

/-- `absorbedFrom` takes from the last message's proof only its openings. -/
theorem absorbedFrom_opensOf (c : RepCtx) (a q : Flock.Proof) : absorbedFrom c a (opensOf q) = absorbedFrom c a q :=
  rfl

theorem readRep_opensOf (c : RepCtx) (rows : List (List ByteArray)) (q : Flock.Proof) :
    readRep c rows (opensOf q) = readRep c rows q := by
  simp only [readRep, absorbedFrom_opensOf]

/-- **`readPs` reads the same at the proofs' openings.** -/
theorem readPs_opensOf {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray}
    {rss : List (List ByteArray)} {cs : List Coin} {fl fl' : FinalP} (hps : fl.ps = fl'.ps.map opensOf) :
    readPs spec sts c₁ r₀ rss cs fl = readPs spec sts c₁ r₀ rss cs fl' := by
  have hsz : fl.ps.size = fl'.ps.size := by rw [hps, Array.size_map]
  have hk : ∀ k, readRep (repCtx spec sts c₁ r₀ cs k) ((cutLens (repLens spec sts) rss).getD k []) fl.ps[k]! =
      readRep (repCtx spec sts c₁ r₀ cs k) ((cutLens (repLens spec sts) rss).getD k []) fl'.ps[k]! := by
    intro k
    by_cases h : k < fl'.ps.size
    · have e : fl.ps[k]! = opensOf fl'.ps[k]! := by
        rw [getElem!_pos fl.ps k (by omega), getElem!_pos fl'.ps k h]
        simp only [hps, Array.getElem_map]
      rw [e]
      exact readRep_opensOf ..
    · rw [getElem!_neg fl.ps k (by omega), getElem!_neg fl'.ps k h]
  unfold readPs
  simp only [hsz, hk]

/-- **`decR` decides the same where the last message reads as the proofs' openings**, at the same statement. -/
theorem decR_opensOf {rd rd' : Claims.Msg → Option FinalP} {x last : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} {fl fl' : FinalP} (hl : as.getLast? = some last) (hrd : rd last = some fl)
    (hrd' : rd' last = some fl') (hst : fl.stmt = fl'.stmt) (hps : fl.ps = fl'.ps.map opensOf) :
    decR rd x as cs = decR rd' x as cs := by
  have hr : ∀ spec sts c₁ r₀ rss cs', readPs spec sts c₁ r₀ rss cs' fl = readPs spec sts c₁ r₀ rss cs' fl' :=
    fun _ _ _ _ _ _ => readPs_opensOf hps
  by_cases hx : fl'.stmt = x
  · have h1 : (rd last).filter (·.stmt == x) = some fl :=
      Option.filter_eq_some_iff.2 ⟨hrd, beq_iff_eq.2 (hst.trans hx)⟩
    have h2 : (rd' last).filter (·.stmt == x) = some fl' := Option.filter_eq_some_iff.2 ⟨hrd', beq_iff_eq.2 hx⟩
    simp only [decR, hl, h1, h2, hr, Option.bind_eq_bind, Option.bind_some]
  · have h1 : (rd last).filter (·.stmt == x) = none := by
      rw [hrd]
      simp [Option.filter, hst, hx]
    have h2 : (rd' last).filter (·.stmt == x) = none := by
      rw [hrd']
      simp [Option.filter, hx]
    simp only [decR, hl, h1, h2, Option.bind_eq_bind, Option.bind_some, Option.bind_none]

/-! ## The honest decision -/

open Rows in
/-- **`decR` decides on the honest messages at `Run.recDec`**: `sessionRun_decides_coins`, with the decision
`ReadsBack` names, where the last message reads as the files' proofs' openings (`opensOf`, what `rdR` returns) at
the run's statement, and the proofs are shaped for their files (`recShaped`). -/
theorem honest_decR (h64 : Claims.Out64 H) (o : Run H rel) {cs : List Coin} (hcs : o.CoinsAt cs)
    {rd : Claims.Msg → Option FinalP} {last : Claims.Msg} {fl : FinalP} (hrd : rd last = some fl)
    (hst : fl.stmt = o.x) (hps : fl.ps = o.recPs.map opensOf) (hsh : ShapedAt o.spec o.sts o.recC₁ o.recPs) :
    decR rd o.x (o.recMsgs cs last) cs = some (o.recDec cs) := by
  have hx := o.setup
  have htab := o.recTabsOk
  have hS := o.ran
  have hresp : ∀ i < cs.length, o.resp i = (cs.getD i default).bytes := by
    intro i hi
    simp only [Case.resp, hcs.1 i hi]
  have hc : o.resp 0 = o.recC₁.bytes := rfl
  have hfc := filesFinalC (resp := o.resp) h64 hc (setup_tables hx).2.symm htab hS hresp hcs.2
  obtain ⟨h64t, hts⟩ := setup_tables hx
  obtain ⟨hsz, hall⟩ := sessionS_reps h64 hc hts.symm htab hS (capRows o.recCaps) cs.tail.dropLast
  have hcaps : ∀ i < o.recCaps.size, o.recCaps[i]!.size = RecDeploy.level0Nodes o.sts[i]! := by
    intro i hi
    unfold Run.recCaps at hi ⊢
    rw [fileCaps_size] at hi
    have h := (hall (2 * i) (by omega)).2.2.2.2.1
    rw [show 2 * i / 2 = i by omega] at h
    simp only [fileCaps, JsonRT.range_map_getElem! _ hi]
    exact h
  have hrep : ∀ k < o.recPs.size, SchedOf (o.recCtx cs k) ∧ HonestRep (o.recCtx cs k) o.recPs[k]! ∧
      o.recPs[k]!.cap = o.recCaps[k / 2]! := by
    intro k hk
    unfold Run.recPs at hk
    rw [filePs_size] at hk
    have hf := hfc k hk
    rw [hsz] at hk
    obtain ⟨-, h2, h3, h4, -, -⟩ := hall k hk
    refine ⟨h2, h3 hf, ?_⟩
    unfold Run.recCaps
    simp only [fileCaps, JsonRT.range_map_getElem! _ (show k / 2 < o.sts.size by omega)]
    exact h4
  have hfit0 : RowsFit ((capRows o.recCaps).map nodeRow) := by
    refine capRows_fit (by unfold Run.recCaps; rw [fileCaps_size]; exact h64t) fun a ha => ?_
    obtain ⟨i, hi, rfl⟩ := Array.mem_map.1 ha
    have hi' := Array.mem_range.1 hi
    have hk2 : 2 * i / 2 < o.sts.size := by omega
    obtain ⟨-, hsc, hon, -, -, -⟩ := hall (2 * i) (by omega)
    have hok := htab (2 * i / 2) hk2
    exact rep_cap_le hsc (hon (hfc (2 * i) (by rw [hsz]; omega))) (repCtx_bounds hk2 hok.1 hok.2).1
  have hfit : ∀ k < o.recPs.size, ∀ rs ∈ repRows (o.recCtx cs k) o.recPs[k]! (o.recTs k), RowsFit rs := by
    intro k hk rs hrs
    unfold Run.recPs at hk
    rw [filePs_size, hsz] at hk
    have hk2 : k / 2 < o.sts.size := by omega
    obtain ⟨-, hsc, hon, -, -, -⟩ := hall k hk
    have hok := htab (k / 2) hk2
    obtain ⟨hd, hkl, hn⟩ := repCtx_bounds (r₀ := capRows o.recCaps) (cs := cs.tail.dropLast) hk2 hok.1 hok.2
    exact repRows_fit (o.recTs k) hsc (hon (hfc k (by rw [hsz]; omega))) hd hkl hn rs hrs
  have hl : (o.recMsgs cs last).getLast? = some last := by
    unfold Run.recMsgs honestMsgs
    exact List.getLast?_concat
  rw [decR_opensOf (rd' := fun _ => some { fl with ps := o.recPs }) (fl' := { fl with ps := o.recPs }) hl hrd rfl rfl
    hps]
  exact readsBack (fun _ => some { fl with ps := o.recPs }) o.x last cs o.s o.spec o.sts o.recC₁ o.recCaps o.recPs
    o.recTs { fl with ps := o.recPs } hx hcs.head (fileCaps_size ..) hcaps (by unfold Run.recPs; rw [filePs_size, hsz]) hrep hfit0 hfit rfl rfl
    hst hsh

/-! ## The checks at the honest decision -/

/-- **The statement sets up at `hOf H`.** -/
def HonestSetup (H : Claims.Msg → Claims.Msg) : Prop :=
  ∀ (o : Run H Readable.relation) (cs : List Coin), o.CoinsAt cs →
    VerdictR.setup.At H (o.recDec cs).s o.x (o.recDec cs).link (o.recDec cs).c₁ (o.recDec cs).mids (o.recDec cs).f

/-- **Each honest proof is shaped for its file.** -/
def HonestShaped (H : Claims.Msg → Claims.Msg) : Prop :=
  ∀ (o : Run H Readable.relation) (cs : List Coin), o.CoinsAt cs →
    VerdictR.shaped.At H (o.recDec cs).s o.x (o.recDec cs).link (o.recDec cs).c₁ (o.recDec cs).mids (o.recDec cs).f

/-- **The rep streams' names are distinct.** -/
def HonestNames (H : Claims.Msg → Claims.Msg) : Prop :=
  ∀ (o : Run H Readable.relation) (cs : List Coin), o.CoinsAt cs →
    VerdictR.names.At H (o.recDec cs).s o.x (o.recDec cs).link (o.recDec cs).c₁ (o.recDec cs).mids (o.recDec cs).f

/-- **The verdict passes on the honest proofs at the reader's record**: `verifyJPWith` at `jAt` of `recJ`. -/
def HonestVerdict (H : Claims.Msg → Claims.Msg) : Prop :=
  ∀ (o : Run H Readable.relation) (cs : List Coin), o.CoinsAt cs →
    VerdictR.verdict.At H (o.recDec cs).s o.x (o.recDec cs).link (o.recDec cs).c₁ (o.recDec cs).mids
      (o.recDec cs).f

/-- **The checks after the verdict pass at the reader's record**: `checksAtJ` at `jAt` of `recJ`. -/
def HonestAfter (H : Claims.Msg → Claims.Msg) : Prop :=
  ∀ (o : Run H Readable.relation) (cs : List Coin), o.CoinsAt cs →
    VerdictR.after.At H (o.recDec cs).s o.x (o.recDec cs).link (o.recDec cs).c₁ (o.recDec cs).mids (o.recDec cs).f

/-- **Custody reads the reader's record**: `ReadOkR`'s field read at `Run.recDec`. -/
def HonestReads (H : Claims.Msg → Claims.Msg) : Prop :=
  ∀ (o : Run H Readable.relation) (cs : List Coin), o.CoinsAt cs →
    ∃ lv, liveOfJ o.spec (coinSubJ o.spec (o.recDec cs).c₁ (o.recDec cs).mids (o.recDec cs).f.j) = .ok lv

theorem honestSetup : HonestSetup H := fun o _ _ => ⟨_, o.recSetup⟩

theorem honestNames : HonestNames H := by
  intro o cs _ spec' sts' hs
  change setupWith (Claims.hOf H) o.s.read = _ at hs
  rw [o.recSetup] at hs
  cases hs
  exact namesDistinct_of_setup (setupOf_some o.setup).2 o.recPublicInputs

/-- **Each of a session's files decodes to a proof of its shape.** -/
theorem sessionS_shaped (h64 : Claims.Out64 H) {resp : ℕ → Claims.Msg} {spec : SessionSpec} {sts : Array Stmt}
    {w : Claims.Flock.Wit} {ω : Tape} {c₁ : Coin} (hc : resp 0 = c₁.bytes) (hts : sts.size = spec.tables.size)
    (htab : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048)
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS H resp spec sts w ω = (.ok (rec, pfs), bs)) :
    ∀ k < 2 * sts.size,
      FlockVBridge.Family.ProofShaped (shapeAt (tabsOf sts) (pointsOf spec c₁) k) (filePs spec sts c₁ pfs)[k]! := by
  have htabs : (tabsOf sts).size = sts.size := by
    unfold tabsOf
    rw [Array.size_map]
  obtain ⟨schs, cms, hsc, hsz, hk⟩ := JsonRT.sessionP_repRuns H resp spec (tabsOf sts) (tracesOf sts w) ω
    (by rw [htabs, hts]) hS
  rw [htabs] at hsc hsz hk
  replace hsz : pfs.size = 2 * sts.size := hsz
  intro k hk'
  have hk2 : k / 2 < sts.size := by omega
  obtain ⟨e, pt, l, l', he, -, hrun, hf⟩ := hk k hk'
  obtain ⟨hs, hcm⟩ := hsc (k / 2) hk2
  rw [hc, JsonRT.ptsOf_coin] at he
  have hsh : shapeAt (tabsOf sts) (pointsOf spec c₁) k =
      FlockComplete.shapeOf (tabsOf sts)[k / 2]! schs[k / 2]! e.size := by
    simp only [shapeAt, hs, he, Option.getD_some]
    rfl
  have ht := tabsOf_get! hk2
  rw [ht] at hs hcm he hrun hf hsh
  have hp := JsonRT.repShapedOk h64 (htab _ hk2).1 hs resp _ hcm ((htab _ hk2).2 e he) _ _ _ l pt l' hrun
  have hdec : decodeProof (shapeAt (tabsOf sts) (pointsOf spec c₁) k) pfs[k]! = .ok pt.1 := by
    rw [hsh, show pfs[k]! = (rec, pfs).2[k]! from rfl, hf]
    exact FlockVBridge.Family.decodeProof_encodeProof _ _ hp
  rw [filePs_get (by rw [hsz]; exact hk') hdec, hsh]
  exact hp

theorem honestShaped (h64 : Claims.Out64 H) : HonestShaped H := by
  intro o cs _ spec' sts' hs k hk
  change setupWith (Claims.hOf H) o.s.read = _ at hs
  rw [o.recSetup] at hs
  cases hs
  have hx := o.setup
  obtain ⟨-, hts⟩ := setup_tables hx
  have hc : o.resp 0 = o.recC₁.bytes := rfl
  have hsz := (sessionS_reps h64 hc hts.symm o.recTabsOk o.ran [] []).1
  have hk' : k < 2 * o.sts.size := by
    have : k < o.recPs.size := hk
    unfold Run.recPs at this
    rwa [filePs_size, hsz] at this
  have h := sessionS_shaped h64 hc hts.symm o.recTabsOk o.ran k hk'
  show FlockVBridge.Family.ProofShaped _ o.recPs[k]
  rw [← getElem!_pos o.recPs k hk]
  exact h

/-- **The honest proofs are shaped for their files**, as `ShapedAt` states it. -/
theorem recShaped (h64 : Claims.Out64 H) (o : Run H rel) : ShapedAt o.spec o.sts o.recC₁ o.recPs := by
  intro k hk
  have hx := o.setup
  obtain ⟨-, hts⟩ := setup_tables hx
  have hc : o.resp 0 = o.recC₁.bytes := rfl
  have hsz := (sessionS_reps h64 hc hts.symm o.recTabsOk o.ran [] []).1
  have hk' : k < 2 * o.sts.size := by
    have : k < o.recPs.size := hk
    unfold Run.recPs at this
    rwa [filePs_size, hsz] at this
  have h := sessionS_shaped h64 hc hts.symm o.recTabsOk o.ran k hk'
  show FlockVBridge.Family.ProofShaped _ o.recPs[k]
  rw [← getElem!_pos o.recPs k hk]
  exact h

/-! ## Acceptance -/

/-- The deployed plain verifier answers a coin only to a last message of tag `0`. -/
theorem act_ne_coin {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {a : Claims.Msg}
    (hl : as.getLast? = some a) (h0 : a.head? ≠ some 0) (c : Coin) : act zk x as cs ≠ .coin c := by
  intro h
  unfold act at h
  split at h
  · cases h
  · split at h
    · rename_i heq _
      rw [hl] at heq
      cases heq
      exact h0 rfl
    · repeat' split at h
      all_goals cases h
    · cases h

theorem honest_actR (h64 : Claims.Out64 H) (o : Run H rel) {cs : List Coin} (hcs : o.CoinsAt cs)
    {rd : Claims.Msg → Option FinalP} {last : Claims.Msg} {fl : FinalP} (hrd : rd last = some fl)
    (h0 : last.head? ≠ some 0) (hst : fl.stmt = o.x) (hps : fl.ps = o.recPs.map opensOf) :
    actR rd o.x (o.recMsgs cs last) cs = .decide (o.recDec cs).s (o.recDec cs).spec (o.recDec cs).sts
      (o.recDec cs).link (o.recDec cs).c₁ (o.recDec cs).mids (o.recDec cs).f := by
  have hd := honest_decR h64 o hcs hrd hst hps (recShaped h64 o)
  have hl : (o.recMsgs cs last).getLast? = some last := by
    unfold Run.recMsgs honestMsgs
    exact List.getLast?_concat
  unfold actR
  cases ha : act false o.x (o.recMsgs cs last) cs with
  | coin c => exact absurd ha (act_ne_coin hl h0 c)
  | rej => simp only [hd]
  | decide => simp only [hd]

/-- **The checks at `Run.recDec` are `holdsR` at the honest messages.** -/
theorem honest_holdsR_of (h64 : Claims.Out64 H) (hV : HonestVerdict H) (hA : HonestAfter H)
    (o : Run H Readable.relation) {cs : List Coin} (hcs : o.CoinsAt cs) {rd : Claims.Msg → Option FinalP}
    {last : Claims.Msg} {fl : FinalP} (hrd : rd last = some fl) (h0 : last.head? ≠ some 0)
    (hst : fl.stmt = o.x) (hps : fl.ps = o.recPs.map opensOf) :
    ∀ q : VerdictR, holdsR H rd q o.x (o.recMsgs cs last) cs := by
  intro q
  unfold holdsR
  rw [honest_actR h64 o hcs hrd h0 hst hps]
  cases q
  · exact honestSetup o cs hcs
  · exact honestShaped h64 o cs hcs
  · exact honestNames o cs hcs
  · exact hV o cs hcs
  · exact hA o cs hcs

/-- **Custody's read at `Run.recDec` is `ReadOkR` at the honest messages.** -/
theorem honest_readOkR_of (h64 : Claims.Out64 H) (hR : HonestReads H) (o : Run H Readable.relation)
    {cs : List Coin} (hcs : o.CoinsAt cs) {rd : Claims.Msg → Option FinalP} {last : Claims.Msg} {fl : FinalP}
    (hrd : rd last = some fl) (h0 : last.head? ≠ some 0)
    (hst : fl.stmt = o.x) (hps : fl.ps = o.recPs.map opensOf) :
    ReadOkR rd o.x (o.recMsgs cs last) cs := by
  unfold ReadOkR
  rw [honest_actR h64 o hcs hrd h0 hst hps]
  exact hR o cs hcs

/-- **The honest run makes `nextR` accept**, given the verdict, the checks after it and custody's read at
`Run.recDec`. -/
theorem honest_nextR_of (h64 : Claims.Out64 H) (hV : HonestVerdict H) (hA : HonestAfter H) (hR : HonestReads H)
    (o : Run H Readable.relation) {cs : List Coin} (hcs : o.CoinsAt cs) {rd : Claims.Msg → Option FinalP}
    {last : Claims.Msg} {fl : FinalP} (hrd : rd last = some fl) (h0 : last.head? ≠ some 0)
    (hst : fl.stmt = o.x) (hps : fl.ps = o.recPs.map opensOf) :
    ((nextR rd o.x (o.recMsgs cs last) cs).eval H).1 = .acc :=
  nextR_sound H rd o.x _ cs (honest_readOkR_of h64 hR o hcs hrd h0 hst hps)
    (honest_holdsR_of h64 hV hA o hcs hrd h0 hst hps)

end

end FlockDeploy
