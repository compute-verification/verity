import Properties.NetworkPeriod
import Properties.NetworkPeriod.Placed
import Proofs.Properties.NetworkScan
import Proofs.Flock.Claims.Sampling.Count
import Proofs.NetworkCertifier.SampledCharge
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Each form's `K` bounds `PeriodLeakAccepted`'s count: the proofs of `ClockKBoundsCount` and `RowsKBoundsCount`

Each factor is at most 2 to the power of the ceiling of its `log2` (`le_two_pow_clog2`), and `K`'s sums over the
connections and the windows (the compute audits, or the scans) are sums over `Fin m` and `Fin n` (`Array.ofFn`), so
`2 ^ K` is the product of the factors' bounds.

`RowsProgWired` and `RowsStateTied` read the check's refusals off its findings: on acceptance every finding is none
(`firstFinding_none`), each link-window's `wiredCheck` and `stateCheck` among them.

`ScanWindowsPlaced`'s `WrongBounds` holds at the bound the audit publishes, at strata that are the design's strata draw
(`wrongBoundsDrawn`, at `StrataDrawn`): a set of more blocks is accepted with probability at most
`miss(countBound + 1) < δ`.
-/

namespace Properties.NetworkPeriod

open NetworkCheck.Period

theorem le_two_pow_clog2 (x : ℕ) : x ≤ 2 ^ clog2 x := by
  unfold clog2
  split
  · simpa using ‹x ≤ 1›
  · have := Nat.lt_log2_self (n := x - 1)
    omega

theorem foldl_add (l : List ℕ) (a : ℕ) : l.foldl (· + ·) a = a + l.sum := by
  induction l generalizing a with
  | nil => simp
  | cons x l ih => simp only [List.foldl_cons, List.sum_cons, ih]; omega

theorem foldl_ofFn {n : ℕ} (g : Fin n → ℕ) : (Array.ofFn g).foldl (· + ·) 0 = ∑ i, g i := by
  rw [← Array.foldl_toList, Array.toList_ofFn, foldl_add, List.sum_ofFn, zero_add]

theorem capsTerm_ofFn {m : ℕ} (caps : Fin m → ℕ) : capsTerm (Array.ofFn caps) = ∑ j, clog2 (caps j + 1) := by
  unfold capsTerm
  rw [Array.map_ofFn]
  exact foldl_ofFn _

theorem K_ofFn (facts : Array Facts) (scan : Scan) (jitter windows : ℕ) {m n : ℕ} (caps : Fin m → ℕ)
    (w : Fin n → Facts) (tied faulty : Fin n → ℕ) (charged : Array Facts) :
    (rowsTerms facts scan (Array.ofFn fun i => (w i, tied i, faulty i)) (Array.ofFn caps) jitter windows charged).K =
      (∑ j, clog2 (caps j + 1)) + (∑ j, clog2 (caps j + 1)) +
        ∑ i, (scan.terms (w i) (tied i) (faulty i) jitter).bits := by
  simp only [Public.K, rowsTerms, capsTerm_ofFn, Array.map_ofFn]
  rw [foldl_ofFn]
  rfl

theorem caps_le {m : ℕ} (caps : Fin m → ℕ) : ∏ j, (caps j + 1) ≤ 2 ^ ∑ j, clog2 (caps j + 1) := by
  rw [← Finset.prod_pow_eq_pow_sum]
  exact Finset.prod_le_prod fun j _ => le_two_pow_clog2 _

theorem zip_ofFn {α β : Type} {n : ℕ} (f : Fin n → α) (g : Fin n → β) :
    (Array.ofFn f).zip (Array.ofFn g) = Array.ofFn fun i => (f i, g i) := by
  ext <;> simp

theorem clockK_ofFn (jitter syncs windows : ℕ) {m n : ℕ} (caps : Fin m → ℕ) (a : Fin n → Audit)
    (w : Fin n → Facts) :
    (terms (Array.ofFn a) (Array.ofFn w) (Array.ofFn caps) jitter syncs windows).K =
      clog2 (syncs ^ windows) + (∑ j, clog2 (caps j + 1)) + ∑ i, (AuditTerms.of (a i) (w i) jitter).bits := by
  simp only [Public.K, terms, capsTerm_ofFn, zip_ofFn, Array.map_ofFn]
  rw [foldl_ofFn]
  rfl

theorem clockKBoundsCount : ClockKBoundsCount := by
  intro jitter syncs windows m n caps a w
  rw [clockK_ofFn, pow_add, pow_add, mul_assoc]
  refine Nat.mul_le_mul (le_two_pow_clog2 _) (Nat.mul_le_mul (caps_le caps) ?_)
  rw [← Finset.prod_pow_eq_pow_sum]
  refine Finset.prod_le_prod fun i _ => ?_
  simp only [AuditTerms.bits, AuditTerms.of, add_zero, pow_add]
  exact Nat.mul_le_mul (Nat.mul_le_mul (Nat.mul_le_mul le_rfl (le_two_pow_clog2 _)) le_rfl) (le_two_pow_clog2 _)

theorem rowsKBoundsCount : RowsKBoundsCount := by
  intro facts scan jitter windows m n caps w tied faulty charged
  rw [K_ofFn, pow_add, pow_add, mul_assoc]
  refine Nat.mul_le_mul (caps_le caps) (Nat.mul_le_mul (caps_le caps) ?_)
  rw [← Finset.prod_pow_eq_pow_sum]
  refine Finset.prod_le_prod fun i _ => ?_
  simp only [AuditTerms.bits, Scan.terms, pow_add]
  exact Nat.mul_le_mul
    (Nat.mul_le_mul (Nat.mul_le_mul (Nat.mul_le_mul le_rfl (le_two_pow_clog2 _)) le_rfl) (le_two_pow_clog2 _)) le_rfl

/-! ## The rows form's scan reads a compute audit's slot entries -/

theorem firstFinding_none {fs : List (Option (Core.Verdict × String))} (h : firstFinding fs = none) :
    ∀ f ∈ fs, f = none := by
  unfold firstFinding at h
  split at h
  · cases h
  · intro f hf
    simpa using List.findSome?_eq_none_iff.mp h f hf

theorem wiredChecks_getElem? {p : Params} {c : Commitment} {s : Schedule} {sd : ByteArray} {scan : Scan}
    (hform : p.form = .rows s sd scan) {i : ℕ} {reg : Lean.Json} (hreg : c.windows[i]? = some reg) :
    (wiredChecks p c)[i]? = some (wiredCheck p s c.audits (p.first + i) reg) := by
  simp [wiredChecks, hform, Array.getElem?_zipIdx, hreg]

theorem rowsProgWired : RowsProgWired := by
  intro p c e pub prof h s sd scan hform i reg hreg
  have hmem := Array.mem_of_getElem? (wiredChecks_getElem? hform hreg)
  unfold check at h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  simp only [hform] at h
  split at h
  · cases h
  · cases h
  · rename_i hF
    have h0 := firstFinding_none hF _
      (List.mem_append_left _ (List.mem_append_right _ (Array.mem_toList_iff.mpr hmem)))
    unfold wiredCheck at h0
    split at h0
    · assumption
    · cases h0

/-- A link-window's window record whose `prog` is at the root `r` and the domain `d`. -/
def tiedWindow : Lean.Json :=
  Lean.Json.mkObj [("roots", Lean.Json.arr #[Lean.Json.mkObj [("port", "scan/prog0"), ("root", "r"), ("domain", "d")]])]

/-- A compute audit whose Program has one `SlotEntry_v1` template of one member, a pinned Definition (its closure
digest `slotEntryPins`' first), and the schedule of one slot a window (one lane, T = 1, `every` 1). -/
def slotAudit : Audit := ⟨0, 0, "", 1, 0, #[("SlotEntry_v1{KO=1}", 1, slotEntryPins[0]!)]⟩

def slotSchedule : Schedule := ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1⟩

/-- Its registration, with that template's entries port at the root `r` and the domain `d`. -/
def slotReg : Lean.Json :=
  Lean.Json.mkObj [("roots", Lean.Json.arr #[Lean.Json.mkObj
    [("port", "SlotEntry_v1{KO=1}/c0/o0"), ("leaves", Lean.toJson (1 : ℕ)), ("root", "r"), ("domain", "d")]])]

/-- Its honest side: a link-window whose scan's `prog` is at that port's root and domain is wired, and the slot audit
is tied to it, so its free inputs are in the link-window's term. -/
theorem rowsProgWired_honest :
    wired #[slotAudit] slotSchedule #[slotReg] tiedWindow = true ∧ tiedIn #[slotReg] #[tiedWindow] 0 = true := by
  decide +kernel

/-- A compute audit whose Program has one `SelectEntry_v1` template of one member, a pinned Definition (its closure
digest `selectEntryPins`' first), and its registration, with that template's entries port at the root `r` and the
domain `d`. -/
def selectAudit : Audit := ⟨0, 0, "", 1, 0, #[("SelectEntry_v1{KO=1}", 1, selectEntryPins[0]!)]⟩

def selectReg : Lean.Json :=
  Lean.Json.mkObj [("roots", Lean.Json.arr #[Lean.Json.mkObj
    [("port", "SelectEntry_v1{KO=1}/c0/o0"), ("leaves", Lean.toJson (1 : ℕ)), ("root", "r"), ("domain", "d")]])]

/-- Its honest side: a link-window whose scan's `prog` is at that port's root and domain is wired, and the audit is
tied to it. -/
theorem rowsProgWired_select :
    wired #[selectAudit] slotSchedule #[selectReg] tiedWindow = true ∧ tiedIn #[selectReg] #[tiedWindow] 0 = true := by
  decide +kernel

theorem tiedWindow_unwired : wired #[] slotSchedule #[] tiedWindow = false := by
  decide +kernel

/-- **No period check that accepts every period has `RowsProgWired`'s property** (its nontrivial lemma): such a check
accepts a rows-form period of one link-window with no audit, whose scan's `prog` is no audit's slot entries. -/
theorem rowsProgWired_nontrivial : ¬ RowsProgWiredOf acceptAllRows := by
  intro h
  let p : Params :=
    ⟨#[], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 1⟩
  have hw : wired #[] slotSchedule #[] tiedWindow = true :=
    h p ⟨#[], #[tiedWindow], .empty⟩ ⟨#[], #[]⟩ _ _ rfl slotSchedule .empty ⟨0, 0, 0, 0, 0⟩ rfl 0 tiedWindow rfl
  rw [tiedWindow_unwired] at hw
  cases hw

/-- `slotAudit` with its template's body another than every pinned one: a template under `SlotEntry_v1`'s name and
descriptor id, one member a slot, whose closure digest is none of `slotEntryPins`. -/
def renamedAudit : Audit := ⟨0, 0, "", 1, 0, #[("SlotEntry_v1{KO=1}", 1, "")]⟩

/-- A renamed body is no slot entries: the link-window `slotAudit` wires is unwired when its audit's template has
another body, with its port, root, domain and members unchanged. -/
theorem renamedAudit_unwired : wired #[renamedAudit] slotSchedule #[slotReg] tiedWindow = false := by
  decide +kernel

/-- **No period check that accepts every period has `RowsProgWired`'s property at a renamed body** (its nontrivial
lemma at the pin): such a check accepts a rows-form period whose scan's `prog` is at the root and domain of an audit's
`SlotEntry_v1` template's entries port, one member a slot, whose body is none of the pinned Definitions'. -/
theorem rowsProgWired_renamed : ¬ RowsProgWiredOf acceptAllRows := by
  intro h
  let p : Params := ⟨#[renamedAudit], #[], 0, 1, "", Lms.deployed, .empty, .empty, "",
    .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 1⟩
  have hw : wired #[renamedAudit] slotSchedule #[slotReg] tiedWindow = true :=
    h p ⟨#[slotReg], #[tiedWindow], .empty⟩ ⟨#[], #[]⟩ _ _ rfl slotSchedule .empty ⟨0, 0, 0, 0, 0⟩ rfl 0 tiedWindow rfl
  rw [renamedAudit_unwired] at hw
  cases hw

/-! ## The rows form's scan's state is its wired audit's `SCAN_STATE` -/

theorem stateChecks_getElem? {p : Params} {c : Commitment} {s : Schedule} {sd : ByteArray} {scan : Scan}
    (hform : p.form = .rows s sd scan) {i : ℕ} {reg : Lean.Json} (hreg : c.windows[i]? = some reg) :
    (stateChecks p c)[i]? = some (stateCheck p s c.audits (p.first + i) reg) := by
  simp [stateChecks, hform, Array.getElem?_zipIdx, hreg]

theorem rowsStateTied : RowsStateTied := by
  intro p c e pub prof h s sd scan hform i reg hreg
  have hmem := Array.mem_of_getElem? (stateChecks_getElem? hform hreg)
  unfold check at h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  simp only [hform] at h
  split at h
  · cases h
  · cases h
  · rename_i hF
    have h0 := firstFinding_none hF _ (List.mem_append_right _ (Array.mem_toList_iff.mpr hmem))
    unfold stateCheck at h0
    split at h0
    · assumption
    · cases h0

/-- The slot audit's registration with its `SCAN_STATE` registered at the root `root` and the domain `e`, beside its
entries port at `r` and `d`. -/
def stateReg (root : String) : Lean.Json :=
  Lean.Json.mkObj [
    ("registered", Lean.Json.arr #[Lean.Json.mkObj
      [("value", "verity/certifier/scan-state/v1"), ("root", root), ("domain", "e")]]),
    ("roots", Lean.Json.arr #[Lean.Json.mkObj
      [("port", "SlotEntry_v1{KO=1}/c0/o0"), ("leaves", Lean.toJson (1 : ℕ)), ("root", "r"), ("domain", "d")]])]

/-- A link-window whose scan's `prog` is at the slot audit's entries port's root and domain, and whose state, the tree
its units read and register their states in, is at the root `s` and the domain `e`. -/
def stateWindow : Lean.Json :=
  Lean.Json.mkObj [("roots", Lean.Json.arr #[
    Lean.Json.mkObj [("port", "scan/prog0"), ("root", "r"), ("domain", "d")],
    Lean.Json.mkObj [("port", "scan/state"), ("root", "s"), ("domain", "e")]])]

/-- Its honest side: a slot audit whose `SCAN_STATE` is the scan's state, at its root and domain, ties it. -/
theorem rowsStateTied_honest : stateTied #[slotAudit] slotSchedule #[stateReg "s"] stateWindow = true := by
  decide +kernel

/-- A slot audit whose `SCAN_STATE` skips or alters one of the scan's transitions: a tree of other rows, at another
root (`t`) than the scan's state's, so its Programs compute on states the scan never reached. -/
theorem stateWindow_skipped : stateTied #[slotAudit] slotSchedule #[stateReg "t"] stateWindow = false := by
  decide +kernel

/-- A link-window whose scan's `prog` is wired, but whose registration names no state (`tiedWindow`), is not tied. -/
theorem tiedWindow_unstated : stateTied #[slotAudit] slotSchedule #[stateReg "s"] tiedWindow = false := by
  decide +kernel

/-- **No period check that accepts every period has `RowsStateTied`'s property** (its nontrivial lemma): such a check
accepts a rows-form period of one link-window whose scan's `prog` is wired to the slot audit, and whose state is not
that audit's `SCAN_STATE`, which skips or alters a transition of the scan's. -/
theorem rowsStateTied_nontrivial : ¬ RowsStateTiedOf acceptAllRows := by
  intro h
  let p : Params :=
    ⟨#[slotAudit], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 1⟩
  have hs : stateTied #[slotAudit] slotSchedule #[stateReg "t"] stateWindow = true :=
    h p ⟨#[stateReg "t"], #[stateWindow], .empty⟩ ⟨#[], #[]⟩ _ _ rfl slotSchedule .empty ⟨0, 0, 0, 0, 0⟩ rfl 0
      stateWindow rfl
  rw [stateWindow_skipped] at hs
  cases hs

/-- Where a draw reads as strata, the bound the audit publishes is their count bound: `boundOf` takes `countBound` of
`strataOf`'s strata at every draw but a Bernoulli one, which reads as none. -/
theorem boundOf_of_strataOf {population : ℕ} {draw : Core.SamplingDesign} {strata : Array Core.Stratum}
    (h : SampledProofs.Bound.strataOf population draw = some strata) (δ : Core.Ratio) :
    SampledProofs.Bound.boundOf population draw δ = SampledProofs.Bound.countBound strata δ := by
  cases draw <;> simp_all [SampledProofs.Bound.boundOf, SampledProofs.Bound.strataOf]

/-- **`WrongBoundsDrawn`**: at strata that, then strata drawn whole, are the design's strata draw, every possible
incorrect set at `δ` holds at most `countBound strata δ` proof blocks, since `miss(countBound strata δ + 1) < δ`
(`countBound_miss_design`), and that is the bound the audit publishes (`boundOf_of_strataOf`). -/
theorem wrongBoundsDrawn : WrongBoundsDrawn := fun _ _ strata hs _ _ ns δ h hδ hδd _ hF =>
  (boundOf_of_strataOf hs δ).symm ▸ NetTiming.card_le_of_mem_faultySets _ _
    (Claims.Sampling.Count.countBound_miss_design strata ns h δ hδ hδd).le hF

theorem scanWindowsPlaced : ScanWindowsPlaced := by
  sorry

/-- A statement of one input gate, with `null`'s bytes as its descriptor and a design of no rule, which draws no slot,
so every set of slots is accepted with probability `1` (`Design.Pa`). -/
def oneInputStmt : Claims.Sampling.AuditStatement :=
  { Properties.NetworkScan.nullStmt with prog := ⟨[.input], []⟩, design := ⟨[]⟩ }

/-- `oneInputStmt`'s gate as a proof block of its own. -/
def oneInputPart : Claims.Sampling.Partition := ⟨[[[[0]]]]⟩

/-- A deployment that parses every message to `oneInputStmt` and partitions it as `oneInputPart`. -/
def oneInputDeploy : Claims.Sampling.Deploy Bool ℕ Unit :=
  { Properties.NetworkScan.acceptAllDeploy with
    parse := fun _ => some oneInputStmt, partition := fun _ => some oneInputPart }

/-- `oneInputStmt`'s one proof block is a possible incorrect set at `δ = 1/2`: its design draws nothing, so the block
is accepted with probability `1`. -/
theorem oneInput_faulty : ({0} : Finset ℕ) ∈ NetTiming.faultySets oneInputStmt.design oneInputPart (1 / 2) := by
  have hPa : Claims.Sampling.Design.Pa oneInputStmt.design oneInputPart {0} = 1 := rfl
  have hall : NetTiming.allSlots oneInputStmt.design oneInputPart = ∅ := rfl
  classical
  simp only [NetTiming.faultySets, Finset.mem_filter, Finset.mem_powerset]
  refine ⟨fun v hv => ?_, ?_, ?_⟩
  · rw [Finset.mem_singleton.1 hv, Finset.mem_range]
    decide
  · show (1 / 2 : ℝ) < Claims.Sampling.Design.Pa oneInputStmt.design oneInputPart {0}
    rw [hPa]
    norm_num
  · rw [hall]
    exact Finset.disjoint_empty_right _

/-- **`ScanWindowsPlaced`'s conclusion is not trivial** (the two-lemma rule's `_nontrivial`): at a scan C-Flock
canonicalizes as `null` and a deployment that parses every message to `oneInputStmt`, whose descriptor is that text,
partitioned as `oneInputPart`, with at most one incorrect proof block at `δ = 1/2` (`WrongBounds`), one window is not
placed at `β = 0`: its one block, possibly incorrect since the design draws nothing, commits its input gate. So the
conclusion fails where the deployment doesn't lay its statements out as Bits does (`StmtBits`) and the charge is below
a block's committed bits. -/
theorem scanWindowsPlaced_nontrivial :
    ∃ (j : Lean.Json) (c : String) (_ : Flock.canon j = .ok c) (D : Claims.Sampling.Deploy Bool ℕ Unit)
      (sw : Fin 1 → Claims.Sampling.AuditStatement) (Pw : Fin 1 → Claims.Sampling.Partition)
      (_ : ∀ i, ∃ x, D.parse x = some (sw i)) (_ : ∀ i, (⟨(sw i).desc.toArray⟩ : ByteArray) = c.toUTF8)
      (_ : ∀ i, D.partition (sw i) = some (Pw i)) (_ : ∀ i, WrongBounds (sw i) (Pw i) (1 / 2) 1),
      ¬ NetTiming.WindowsPlaced D (1 / 2) sw Pw fun _ => 0 := by
  refine ⟨.null, "null", Properties.NetworkScan.canon_null, oneInputDeploy, fun _ => oneInputStmt,
    fun _ => oneInputPart, fun _ => ⟨[], rfl⟩, fun _ => Properties.NetworkScan.bytes_of_data_toList "null".toUTF8,
    fun _ => rfl, fun _ => ?_, fun h => ?_⟩
  · unfold WrongBounds
    intro F hF
    classical
    simp only [NetTiming.faultySets, Finset.mem_filter, Finset.mem_powerset] at hF
    calc F.card ≤ (Finset.range oneInputPart.blocks.length).card := Finset.card_le_card hF.1
      _ = 1 := (Finset.card_range _).trans rfl
  · have h1 : NetTiming.faultyBits oneInputStmt.prog oneInputPart {0} = 1 := by
      simp only [NetTiming.faultyBits, Finset.sum_singleton]
      decide
    have h0 : NetTiming.faultyBits oneInputStmt.prog oneInputPart {0} ≤ 0 := h.2 0 {0} oneInput_faulty
    omega

/-- **`WrongBoundsDrawn`'s premise holds at a deployed law** (the two-lemma rule's `_honest`): on a layout's partition of
a population class of two proof blocks and an input class of one, the code's strata of a `subset:1` draw over the two
units (`SampledProofs.Bound.strataOf`), then the input class drawn whole, are the strata draw of the design a `subset:1`
draw runs: `subset 1`, then the input class `all`. -/
theorem wrongBoundsDrawn_honest :
    Claims.Sampling.StrataDrawn ⟨[.subset 1, .all]⟩ ⟨[[[[0], [1]]], [[[2]]]]⟩
      ((((SampledProofs.Bound.strataOf 2 (.subset 1)).getD #[]).toList.map fun x => (x.size, x.k)) ++
        [1].map fun m => (m, m)) := by
  refine ⟨⟨rfl, fun p hp => ?_⟩, [[(2, 1)], [(1, 1)]], rfl, fun p hp => ?_, rfl⟩
  · simp only [List.zip_cons_cons, List.zip_nil_left, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl
    · show 1 ≤ 1 ∧ 1 ≤ ([2] : List ℕ).sum
      decide
    · trivial
  · simp only [List.zip_cons_cons, List.zip_nil_left, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl <;> rfl

/-- The code's strata of no stratum, then any strata drawn whole, are not `oneInputStmt`'s strata draw on
`oneInputPart`: its design of no rule is not well formed on a partition of one class. -/
theorem oneInput_not_strataDrawn (ns : List ℕ) :
    ¬ Claims.Sampling.StrataDrawn oneInputStmt.design oneInputPart
      (((#[] : Array Core.Stratum).toList.map fun x => (x.size, x.k)) ++ ns.map fun m => (m, m)) := fun h => by
  obtain ⟨⟨hl, _⟩, _⟩ := h
  exact absurd hl (by decide)

/-- **`WrongBoundsDrawn`'s conclusion is not trivial** (the two-lemma rule's `_nontrivial`): a stratified draw of no
strata reads as no strata, and at `oneInputStmt`, whose design of no rule draws nothing, its one proof block is a
possible incorrect set at `δ = 1/2` (`oneInput_faulty`), so the bound the audit publishes for that draw, `0`, is no
`WrongBounds` bound there. Every other premise holds at `δ = ⟨1, 2⟩`, and the strata draw (`StrataDrawn`) fails,
whatever strata drawn whole follow (`oneInput_not_strataDrawn`). -/
theorem wrongBoundsDrawn_nontrivial :
    ∃ (strata : Array Core.Stratum) (_ : SampledProofs.Bound.strataOf 0 (.stratified #[]) = some strata)
      (s : Claims.Sampling.AuditStatement) (P : Claims.Sampling.Partition),
      ¬ WrongBounds s P (((⟨1, 2⟩ : Core.Ratio).num : ℝ) / (⟨1, 2⟩ : Core.Ratio).den)
        (SampledProofs.Bound.boundOf 0 (.stratified #[]) ⟨1, 2⟩) := by
  refine ⟨#[], rfl, oneInputStmt, oneInputPart, fun h => ?_⟩
  have h0 : SampledProofs.Bound.boundOf 0 (.stratified #[]) ⟨1, 2⟩ = 0 := by
    rw [boundOf_of_strataOf rfl]
    simp [SampledProofs.Bound.countBound]
  have hδ : (((⟨1, 2⟩ : Core.Ratio).num : ℝ) / (⟨1, 2⟩ : Core.Ratio).den) = 1 / 2 := by norm_num
  rw [h0, hδ] at h
  simpa using h {0} oneInput_faulty

end Properties.NetworkPeriod
