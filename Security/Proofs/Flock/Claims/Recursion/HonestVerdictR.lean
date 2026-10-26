import Proofs.Flock.Claims.Recursion.NextRTables
import Proofs.Flock.Claims.Complete.Replay.ProverOk

/-!
# The honest run passes `nextR`'s checks

`nextR_sound` makes the deployed recursive verifier's last step accept from its reading (`ReadOkR`) and its five checks
(`VerdictR`: the setup, the proofs' shapes, the rep streams' names, the verdict on the decoded proofs at the reader's
record `recJ`, and the checks after it). Here they hold at the honest run: a plain completeness run
(`FlockComplete.Run`), whose prover sends round 0 from its tables' level-0 caps, each rep's rounds from its proof at
its ring switch's batched value (`honestTs`), and a last message the reader reads as the files' proofs.

* **The honest decision** (`honest_decR`): `decR` decides on the honest messages (`honestMsgsR`) at `honestDec`, the
  link of the caps, the proofs' framed bytes and the reader's record at them. This is `sessionRun_decides_coins` with
  the decision `ReadsBack` names.
* **The checks** (`HonestSetup`, `HonestShaped`, `HonestNames`, `HonestVerdict`, `HonestAfter`, `HonestReads`): each of
  `VerdictR.At` and `ReadOkR`'s field read at `honestDec`.
* **Acceptance** (`honest_holdsR_of`, `honest_nextR_of`): the checks at `honestDec` are `holdsR` at the honest messages,
  and with the reading they make `nextR` accept.
-/

namespace FlockDeploy

open Flock FlockComplete

section

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-! ## What the honest prover sends -/

/-- The first coin, which the link points are read at. -/
def honestC₁ (o : Run H rel) : Coin := Claims.coinSeq o.r 0

/-- Each table's level-0 cap, as the session's files carry it. -/
def honestCaps (o : Run H rel) : Array (Array ByteArray) := fileCaps o.spec o.sts (honestC₁ o) o.proofs

/-- The reps' proofs, as the session's files decode. -/
def honestPs (o : Run H rel) : Array Flock.Proof := filePs o.spec o.sts (honestC₁ o) o.proofs

/-- **The honest prover's batched value at rep `k`**: the ring switch's `T` at table `k / 2`'s rep `k % 2`. -/
def honestTs (o : Run H rel) (k : ℕ) : F128 := (o.sw (k / 2) (k % 2)).1.1

/-- Rep `k`'s context at the honest round 0 and the coins `cs`. -/
def honestCtx (o : Run H rel) (cs : List Coin) (k : ℕ) : RepCtx :=
  repCtx o.spec o.sts (honestC₁ o) (capRows (honestCaps o)) cs.tail.dropLast k

/-- **The honest messages**: round 0 the caps' rows, each rep's rounds its proof's rows at its batched value, then the
last message. -/
def honestMsgsR (o : Run H rel) (cs : List Coin) (last : Claims.Msg) : List Claims.Msg :=
  honestMsgs (honestCaps o) ((List.range (honestPs o).size).map fun k =>
    repRows (honestCtx o cs k) (honestPs o)[k]! (honestTs o k)) last

/-- The link the reader takes from round 0: each table's `root_B` its cap's nodes joined, no public words. -/
def honestLink (o : Run H rel) : List ByteArray × List ByteArray :=
  (((honestCaps o).toList.map Array.toList).map (·.foldl (· ++ ·) ByteArray.empty),
    o.sts.toList.map fun _ => ByteArray.empty)

/-- The round messages the reader takes: the honest proofs' framed bytes. -/
def honestFrames (o : Run H rel) (cs : List Coin) : List Claims.Msg :=
  ((List.range (honestPs o).size).map fun k => framesOf (honestCtx o cs k) (honestPs o)[k]! (honestTs o k)).flatten

/-- **The honest decision**: the statement's setup, the honest link, the first coin, the framed rounds with their coins,
and the reader's record (`recJ`) at them with the files' proofs. -/
def honestDec (o : Run H rel) (cs : List Coin) : DecR :=
  ⟨o.s, o.spec, o.sts, honestLink o, honestC₁ o, (honestFrames o cs).zip cs.tail.dropLast,
    ⟨o.x, recJ o.spec o.sts (honestLink o) (honestC₁ o) (honestFrames o cs) cs.tail.dropLast (honestPs o),
      honestPs o⟩⟩

/-- **The deployed verifier's coins at the run**: the run's coins in order, at least the reps' rounds and two. -/
def CoinsAt (o : Run H rel) (cs : List Coin) : Prop :=
  (∀ i < cs.length, cs.getD i default = Claims.coinSeq o.r i) ∧ (repLens o.spec o.sts).sum + 2 ≤ cs.length

/-! ## The honest decision -/

/-- **A plain run's tables are `TableOk`**, with at most 2,048 extra claims at the first coin's link points. -/
theorem Run.tabsOk (o : Run H rel) : ∀ (i : ℕ) (hi : i < o.sts.size), JsonRT.TableOk o.sts[i] ∧
    ∀ e, o.sts[i].extraClaims (pointsOf o.spec (honestC₁ o)) = .ok e → e.size ≤ 2048 := by
  obtain ⟨st, hst, hok⟩ := o.tableOk
  obtain ⟨st', hst', hsp, -, hreg⟩ := o.schemeOk
  rw [hst] at hst'
  obtain rfl : st = st' := by simpa using hst'
  intro i hi
  have hi0 : i = 0 := by simp [hst] at hi; omega
  subst hi0
  simp only [hst]
  refine ⟨hok, fun e he => ?_⟩
  rw [JsonRT.extraClaims_size he]
  simp only [pointsOf, Array.size_map, Array.size_range, hsp]
  show st.regions.size * 2 ≤ 2048
  omega

theorem CoinsAt.head {o : Run H rel} {cs : List Coin} (h : CoinsAt o cs) : cs.head? = some (honestC₁ o) := by
  cases cs with
  | nil => have := h.2; omega
  | cons c cs' =>
    have := h.1 0 (by simp)
    simp only [List.getD_cons_zero] at this
    simp [this, honestC₁]

open Rows in
/-- **`decR` decides on the honest messages at `honestDec`**: `sessionRun_decides_coins`, with the decision
`ReadsBack` names, where the last message reads as the files' proofs at the run's statement. -/
theorem honest_decR (h64 : Claims.Out64 H) (o : Run H rel) {cs : List Coin} (hcs : CoinsAt o cs)
    {rd : Claims.Msg → Option FinalP} {last : Claims.Msg} {fl : FinalP} (hrd : rd last = some fl)
    (hst : fl.stmt = o.x) (hps : fl.ps = honestPs o) :
    decR rd o.x (honestMsgsR o cs last) cs = some (honestDec o cs) := by
  have hx := o.setup
  have htab := o.tabsOk
  have hS := o.ran
  have hresp : ∀ i < cs.length, (fun j => (Claims.coinSeq o.r j).bytes) i = (cs.getD i default).bytes := by
    intro i hi
    simp only [hcs.1 i hi]
  have hc : (fun j => (Claims.coinSeq o.r j).bytes) 0 = (honestC₁ o).bytes := rfl
  have hfc := filesFinalC h64 hc (setup_tables hx).2.symm htab hS hresp hcs.2
  obtain ⟨h64t, hts⟩ := setup_tables hx
  obtain ⟨hsz, hall⟩ := sessionS_reps h64 hc hts.symm htab hS (capRows (honestCaps o)) cs.tail.dropLast
  have hcaps : ∀ i < (honestCaps o).size, (honestCaps o)[i]!.size = RecDeploy.level0Nodes o.sts[i]! := by
    intro i hi
    unfold honestCaps at hi ⊢
    rw [fileCaps_size] at hi
    have h := (hall (2 * i) (by omega)).2.2.2.2.1
    rw [show 2 * i / 2 = i by omega] at h
    simp only [fileCaps, JsonRT.range_map_getElem! _ hi]
    exact h
  have hrep : ∀ k < (honestPs o).size, SchedOf (honestCtx o cs k) ∧ HonestRep (honestCtx o cs k) (honestPs o)[k]! ∧
      (honestPs o)[k]!.cap = (honestCaps o)[k / 2]! := by
    intro k hk
    unfold honestPs at hk
    rw [filePs_size] at hk
    have hf := hfc k hk
    rw [hsz] at hk
    obtain ⟨-, h2, h3, h4, -, -⟩ := hall k hk
    refine ⟨h2, h3 hf, ?_⟩
    unfold honestCaps
    simp only [fileCaps, JsonRT.range_map_getElem! _ (show k / 2 < o.sts.size by omega)]
    exact h4
  have hfit0 : RowsFit (capRows (honestCaps o)) := by
    refine capRows_fit (by unfold honestCaps; rw [fileCaps_size]; exact h64t) fun a ha => ?_
    obtain ⟨i, hi, rfl⟩ := Array.mem_map.1 ha
    have hi' := Array.mem_range.1 hi
    have hk2 : 2 * i / 2 < o.sts.size := by omega
    obtain ⟨-, hsc, hon, -, -, -⟩ := hall (2 * i) (by omega)
    have hok := htab (2 * i / 2) hk2
    exact rep_cap_le hsc (hon (hfc (2 * i) (by rw [hsz]; omega))) (repCtx_bounds hk2 hok.1 hok.2).1
  have hfit : ∀ k < (honestPs o).size, ∀ rs ∈ repRows (honestCtx o cs k) (honestPs o)[k]! (honestTs o k),
      RowsFit rs := by
    intro k hk rs hrs
    unfold honestPs at hk
    rw [filePs_size, hsz] at hk
    have hk2 : k / 2 < o.sts.size := by omega
    obtain ⟨-, hsc, hon, -, -, -⟩ := hall k hk
    have hok := htab (k / 2) hk2
    obtain ⟨hd, hkl, hn⟩ := repCtx_bounds (r₀ := capRows (honestCaps o)) (cs := cs.tail.dropLast) hk2 hok.1 hok.2
    exact repRows_fit (honestTs o k) hsc (hon (hfc k (by rw [hsz]; omega))) hd hkl hn rs hrs
  have h := readsBack rd o.x last cs o.s o.spec o.sts (honestC₁ o) (honestCaps o) (honestPs o) (honestTs o) fl hx
    hcs.head (fileCaps_size ..) hcaps (by unfold honestPs; rw [filePs_size, hsz]) hrep hfit0 hfit hrd hps
  rw [hst] at h
  exact h

/-! ## The checks at the honest decision -/

/-- **The statement sets up at `hOf H`.** -/
def HonestSetup (H : Claims.Msg → Claims.Msg) : Prop :=
  ∀ (o : Run H Readable.relation) (cs : List Coin), CoinsAt o cs →
    VerdictR.setup.At H (honestDec o cs).s o.x (honestDec o cs).link (honestDec o cs).c₁ (honestDec o cs).mids
      (honestDec o cs).f

/-- **Each honest proof is shaped for its file.** -/
def HonestShaped (H : Claims.Msg → Claims.Msg) : Prop :=
  ∀ (o : Run H Readable.relation) (cs : List Coin), CoinsAt o cs →
    VerdictR.shaped.At H (honestDec o cs).s o.x (honestDec o cs).link (honestDec o cs).c₁ (honestDec o cs).mids
      (honestDec o cs).f

/-- **The rep streams' names are distinct.** -/
def HonestNames (H : Claims.Msg → Claims.Msg) : Prop :=
  ∀ (o : Run H Readable.relation) (cs : List Coin), CoinsAt o cs →
    VerdictR.names.At H (honestDec o cs).s o.x (honestDec o cs).link (honestDec o cs).c₁ (honestDec o cs).mids
      (honestDec o cs).f

/-- **The verdict passes on the honest proofs at the reader's record**: `verifyJPWith` at `jAt` of `recJ`. -/
def HonestVerdict (H : Claims.Msg → Claims.Msg) : Prop :=
  ∀ (o : Run H Readable.relation) (cs : List Coin), CoinsAt o cs →
    VerdictR.verdict.At H (honestDec o cs).s o.x (honestDec o cs).link (honestDec o cs).c₁ (honestDec o cs).mids
      (honestDec o cs).f

/-- **The checks after the verdict pass at the reader's record**: `checksAtJ` at `jAt` of `recJ`. -/
def HonestAfter (H : Claims.Msg → Claims.Msg) : Prop :=
  ∀ (o : Run H Readable.relation) (cs : List Coin), CoinsAt o cs →
    VerdictR.after.At H (honestDec o cs).s o.x (honestDec o cs).link (honestDec o cs).c₁ (honestDec o cs).mids
      (honestDec o cs).f

/-- **Custody reads the reader's record**: `ReadOkR`'s field read at `honestDec`. -/
def HonestReads (H : Claims.Msg → Claims.Msg) : Prop :=
  ∀ (o : Run H Readable.relation) (cs : List Coin), CoinsAt o cs →
    ∃ lv, liveOfJ o.spec (coinSubJ o.spec (honestC₁ o) ((honestFrames o cs).zip cs.tail.dropLast)
      (honestDec o cs).f.j) = .ok lv

/-! ## Acceptance -/

theorem honest_actR (h64 : Claims.Out64 H) (o : Run H rel) {cs : List Coin} (hcs : CoinsAt o cs)
    {rd : Claims.Msg → Option FinalP} {last : Claims.Msg} {fl : FinalP} (hrd : rd last = some fl)
    (hst : fl.stmt = o.x) (hps : fl.ps = honestPs o) :
    actR rd o.x (honestMsgsR o cs last) cs = .decide (honestDec o cs).s (honestDec o cs).spec (honestDec o cs).sts
      (honestDec o cs).link (honestDec o cs).c₁ (honestDec o cs).mids (honestDec o cs).f :=
  actR_decide (honest_decR h64 o hcs hrd hst hps)

/-- **The checks at `honestDec` are `holdsR` at the honest messages.** -/
theorem honest_holdsR_of (h64 : Claims.Out64 H) (hS : HonestSetup H) (hSh : HonestShaped H) (hN : HonestNames H)
    (hV : HonestVerdict H) (hA : HonestAfter H) (o : Run H Readable.relation) {cs : List Coin}
    (hcs : CoinsAt o cs) {rd : Claims.Msg → Option FinalP} {last : Claims.Msg} {fl : FinalP}
    (hrd : rd last = some fl) (hst : fl.stmt = o.x) (hps : fl.ps = honestPs o) :
    ∀ q : VerdictR, holdsR H rd q o.x (honestMsgsR o cs last) cs := by
  intro q
  unfold holdsR
  rw [honest_actR h64 o hcs hrd hst hps]
  cases q
  · exact hS o cs hcs
  · exact hSh o cs hcs
  · exact hN o cs hcs
  · exact hV o cs hcs
  · exact hA o cs hcs

/-- **Custody's read at `honestDec` is `ReadOkR` at the honest messages.** -/
theorem honest_readOkR_of (h64 : Claims.Out64 H) (hR : HonestReads H) (o : Run H Readable.relation)
    {cs : List Coin} (hcs : CoinsAt o cs) {rd : Claims.Msg → Option FinalP} {last : Claims.Msg} {fl : FinalP}
    (hrd : rd last = some fl) (hst : fl.stmt = o.x) (hps : fl.ps = honestPs o) :
    ReadOkR rd o.x (honestMsgsR o cs last) cs := by
  unfold ReadOkR
  rw [honest_actR h64 o hcs hrd hst hps]
  exact hR o cs hcs

/-- **The honest run makes `nextR` accept**, given the checks at `honestDec`. -/
theorem honest_nextR_of (h64 : Claims.Out64 H) (hS : HonestSetup H) (hSh : HonestShaped H) (hN : HonestNames H)
    (hV : HonestVerdict H) (hA : HonestAfter H) (hR : HonestReads H) (o : Run H Readable.relation)
    {cs : List Coin} (hcs : CoinsAt o cs) {rd : Claims.Msg → Option FinalP} {last : Claims.Msg} {fl : FinalP}
    (hrd : rd last = some fl) (hst : fl.stmt = o.x) (hps : fl.ps = honestPs o) :
    ((nextR rd o.x (honestMsgsR o cs last) cs).eval H).1 = .acc :=
  nextR_sound H rd o.x _ cs (honest_readOkR_of h64 hR o hcs hrd hst hps)
    (honest_holdsR_of h64 hS hSh hN hV hA o hcs hrd hst hps)

end

end FlockDeploy
