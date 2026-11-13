module

public import Properties.NetworkPeriod
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import all Init.Data.Array.Basic
import all Init.Data.Repr
import all Init.Data.Range.Basic
import all Init.Data.Array.DecidableEq
public import Mathlib.Algebra.GroupWithZero.Nat
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Each form's `K` bounds `MonitoringPeriodLeakAccepted`'s count: the proofs of `ClockKBoundsCount` and `RowsKBoundsCount`

Each factor is at most 2 to the power of the ceiling of its `log2` (`le_two_pow_clog2`), and `K`'s sums over the
connections and the windows (the compute audits, or the scans) are sums over `Fin m` and `Fin n` (`Array.ofFn`), so
`2 ^ K` is the product of the factors' bounds.

`RowsProgWired` and `RowsStateTied` read the check's refusals off its findings: on acceptance every finding is none
(`firstFinding_none`), each link-window's `wiredCheck` and `stateCheck` among them.

`RowsTiedCharged` reads the public the rows form accepts with (`check_rows`): with no finding, every audit's check gave
its facts and every link-window's its scan's, so the public's `audits` and `units` hold them position for position
(`getElem?_filterMap_of_isSome`), `charged` is `chargedOf` over the audits (`charged_eq`), and a link-window's term's
`faultyEntries` is `faultyEntries` of its registration. An audit's facts are `factsOf`'s, which refuses a profile
sampling more than its level holds, a stratified one with a stratum sampling more than it holds or whose strata's
sizes do not sum to the level's, and one that is another sample than its registration's (`factsOf_sample`), so facts
of a registered stratified sample that is sampled whole are of every stratum sampled whole (`strata_whole`).

`RegisteredSampleHeld` and `SampleBoundCharged` read the same public in both forms (`check_clock`, `check_rows`): what
`factsOf` gives of every answer (`factsOf_sample`: facts of the registered sample, with a count at least its bound,
the larger of the stated one and `sampleBound`) holds of each audit's facts and each scan's
(`rowsExecutionPhaseCheck_facts`), position for position (`answersAt`).

`RowsCapPublished` reads the public off the acceptance, in each form.
-/

@[expose] public section

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
def tiedExecutionPhase : Lean.Json :=
  Lean.Json.mkObj [("roots", Lean.Json.arr #[Lean.Json.mkObj [("port", "scan/prog0"), ("root", "r"), ("domain", "d")]])]

/-- A compute audit whose Program has one `SlotEntry_v1` template of one member, a pinned Definition (its closure
digest `slotEntryPins`' first), and the schedule of one slot a window (one lane, T = 1, `every` 1). -/
def slotAudit : Audit := ⟨0, 0, "", 1, 0, #[("SlotEntry_v1{KO=1}", 1, slotEntryPins[0]!)]⟩

def slotSchedule : Schedule := ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, none⟩

/-- Its registration, with that template's entries port at the root `r` and the domain `d`. -/
def slotReg : Lean.Json :=
  Lean.Json.mkObj [("roots", Lean.Json.arr #[Lean.Json.mkObj
    [("port", "SlotEntry_v1{KO=1}/c0/o0"), ("leaves", Lean.toJson (1 : ℕ)), ("root", "r"), ("domain", "d")]])]

/-- Its honest side: a link-window whose scan's `prog` is at that port's root and domain is wired, and the slot audit
is tied to it, so its free inputs are in the link-window's term. -/
theorem rowsProgWired_honest :
    wired #[slotAudit] slotSchedule #[slotReg] tiedExecutionPhase = true ∧ tiedIn #[slotReg] #[tiedExecutionPhase] 0 = true := by
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
    wired #[selectAudit] slotSchedule #[selectReg] tiedExecutionPhase = true ∧ tiedIn #[selectReg] #[tiedExecutionPhase] 0 = true := by
  decide +kernel

theorem tiedExecutionPhase_unwired : wired #[] slotSchedule #[] tiedExecutionPhase = false := by
  decide +kernel

/-- **No period check that accepts every period has `RowsProgWired`'s property** (its nontrivial lemma): such a check
accepts a rows-form period of one link-window with no audit, whose scan's `prog` is no audit's slot entries. -/
theorem rowsProgWired_nontrivial : ¬ RowsProgWiredOf acceptAllRows := by
  intro h
  let p : Params :=
    ⟨#[], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 1⟩
  have hw : wired #[] slotSchedule #[] tiedExecutionPhase = true :=
    h p ⟨#[], #[tiedExecutionPhase], .empty⟩ ⟨#[], #[]⟩ _ _ rfl slotSchedule .empty ⟨0, 0, 0, 0, 0⟩ rfl 0 tiedExecutionPhase rfl
  rw [tiedExecutionPhase_unwired] at hw
  cases hw

/-- `slotAudit` with its template's body another than every pinned one: a template under `SlotEntry_v1`'s name and
descriptor id, one member a slot, whose closure digest is none of `slotEntryPins`. -/
def renamedAudit : Audit := ⟨0, 0, "", 1, 0, #[("SlotEntry_v1{KO=1}", 1, "")]⟩

/-- A renamed body is no slot entries: the link-window `slotAudit` wires is unwired when its audit's template has
another body, with its port, root, domain and members unchanged. -/
theorem renamedAudit_unwired : wired #[renamedAudit] slotSchedule #[slotReg] tiedExecutionPhase = false := by
  decide +kernel

/-- **No period check that accepts every period has `RowsProgWired`'s property at a renamed body** (its nontrivial
lemma at the pin): such a check accepts a rows-form period whose scan's `prog` is at the root and domain of an audit's
`SlotEntry_v1` template's entries port, one member a slot, whose body is none of the pinned Definitions'. -/
theorem rowsProgWired_renamed : ¬ RowsProgWiredOf acceptAllRows := by
  intro h
  let p : Params := ⟨#[renamedAudit], #[], 0, 1, "", Lms.deployed, .empty, .empty, "",
    .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 1⟩
  have hw : wired #[renamedAudit] slotSchedule #[slotReg] tiedExecutionPhase = true :=
    h p ⟨#[slotReg], #[tiedExecutionPhase], .empty⟩ ⟨#[], #[]⟩ _ _ rfl slotSchedule .empty ⟨0, 0, 0, 0, 0⟩ rfl 0 tiedExecutionPhase rfl
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
def stateExecutionPhase : Lean.Json :=
  Lean.Json.mkObj [("roots", Lean.Json.arr #[
    Lean.Json.mkObj [("port", "scan/prog0"), ("root", "r"), ("domain", "d")],
    Lean.Json.mkObj [("port", "scan/state"), ("root", "s"), ("domain", "e")]])]

/-- Its honest side: a slot audit whose `SCAN_STATE` is the scan's state, at its root and domain, ties it. -/
theorem rowsStateTied_honest : stateTied #[slotAudit] slotSchedule #[stateReg "s"] stateExecutionPhase = true := by
  decide +kernel

/-- A slot audit whose `SCAN_STATE` skips or alters one of the scan's transitions: a tree of other rows, at another
root (`t`) than the scan's state's, so its Programs compute on states the scan never reached. -/
theorem stateExecutionPhase_skipped : stateTied #[slotAudit] slotSchedule #[stateReg "t"] stateExecutionPhase = false := by
  decide +kernel

/-- A link-window whose scan's `prog` is wired, but whose registration names no state (`tiedExecutionPhase`), is not tied. -/
theorem tiedExecutionPhase_unstated : stateTied #[slotAudit] slotSchedule #[stateReg "s"] tiedExecutionPhase = false := by
  decide +kernel

/-- **No period check that accepts every period has `RowsStateTied`'s property** (its nontrivial lemma): such a check
accepts a rows-form period of one link-window whose scan's `prog` is wired to the slot audit, and whose state is not
that audit's `SCAN_STATE`, which skips or alters a transition of the scan's. -/
theorem rowsStateTied_nontrivial : ¬ RowsStateTiedOf acceptAllRows := by
  intro h
  let p : Params :=
    ⟨#[slotAudit], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 1⟩
  have hs : stateTied #[slotAudit] slotSchedule #[stateReg "t"] stateExecutionPhase = true :=
    h p ⟨#[stateReg "t"], #[stateExecutionPhase], .empty⟩ ⟨#[], #[]⟩ _ _ rfl slotSchedule .empty ⟨0, 0, 0, 0, 0⟩ rfl 0
      stateExecutionPhase rfl
  rw [stateExecutionPhase_skipped] at hs
  cases hs

/-! ## The rows form charges a tied audit sampled in part -/

/-- Where every element maps to a value, `filterMap` keeps the positions. -/
theorem getElem?_filterMap_of_isSome {α β : Type} (g : α → Option β) (xs : Array α)
    (h : ∀ a ∈ xs, (g a).isSome = true) (i : ℕ) : (xs.filterMap g)[i]? = xs[i]?.bind g := by
  obtain ⟨l⟩ := xs
  simp only [List.filterMap_toArray, List.getElem?_toArray]
  have h' : ∀ a ∈ l, (g a).isSome = true := fun a ha => h a (by simpa using ha)
  clear h
  induction l generalizing i with
  | nil => simp
  | cons a l ih =>
    obtain ⟨b, hb⟩ := Option.isSome_iff_exists.mp (h' a (by simp))
    rw [List.filterMap_cons_some hb]
    cases i with
    | zero => simp [hb]
    | succ i => simpa using ih i (fun a ha => h' a (by simp [ha]))

/-- Where every element maps to a value, `filterMap` keeps the size. -/
theorem size_filterMap_of_isSome {α β : Type} (g : α → Option β) (xs : Array α)
    (h : ∀ a ∈ xs, (g a).isSome = true) : (xs.filterMap g).size = xs.size := by
  have h' : ∀ a ∈ xs.toList, (g a).isSome = true := fun a ha => h a (Array.mem_toList_iff.mp ha)
  rw [← Array.length_toList, Array.toList_filterMap, ← Array.length_toList (xs := xs)]
  generalize xs.toList = l at h'
  induction l with
  | nil => rfl
  | cons a l ih =>
    obtain ⟨b, hb⟩ := Option.isSome_iff_exists.mp (h' a (by simp))
    rw [List.filterMap_cons_some hb, List.length_cons, List.length_cons, ih (fun a ha => h' a (by simp [ha]))]

theorem findingOf_none {α : Type} {y : Found α} (h : findingOf y = none) : ∃ v, y = .ok v := by
  cases y with
  | error r => cases h
  | ok v => exact ⟨v, rfl⟩

/-- The rows form's link-windows' findings, as `check` computes them. -/
def rowsWindows (p : Params) (c : Commitment) (e : Evidence) (s : Schedule) (sd : ByteArray) :
    Array (Found (Option (Facts × ℕ))) :=
  (c.windows.zip e.windows).zipIdx.map fun ((reg, x), i) =>
    some <$> rowsExecutionPhaseCheck p s sd c.audits (p.first + i) reg x

/-- What acceptance in the rows form gives: the sizes, no finding of a link-window or an audit, and the public, the
rows form's (`rowsPublic`) with the schedule's response cap. -/
theorem check_rows {p : Params} {c : Commitment} {e : Evidence} {pub : Public} {prof : Profile} {s : Schedule}
    {sd : ByteArray} {scan : Scan} (hform : p.form = .rows s sd scan) (h : check p c e = .accept pub prof) :
    c.audits.size = p.audits.size ∧ e.windows.size = c.windows.size ∧ e.audits.size = c.audits.size ∧
      (∀ y ∈ rowsWindows p c e s sd, findingOf y = none) ∧ (∀ y ∈ auditChecks p c e, findingOf y = none) ∧
      pub = { rowsPublic p c (auditChecks p c e) scan ((rowsWindows p c e s sd).filterMap (·.toOption.join)) with
        maxTokensCap := p.form.maxTokensCap } := by
  unfold check at h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  rename_i h1 h2 h3 h4
  simp only [hform] at h
  split at h
  · cases h
  · cases h
  · rename_i hF
    have h0 := firstFinding_none hF
    refine ⟨by omega, by omega, by omega, ?_, ?_, ?_⟩
    · intro y hy
      exact h0 _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _
        (List.mem_map_of_mem (Array.mem_toList_iff.mpr hy))))))
    · intro y hy
      exact h0 _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _
        (List.mem_map_of_mem (Array.mem_toList_iff.mpr hy)))))
    · injection h with hp
      rw [hform]
      exact hp.symm

theorem auditChecks_getElem? {p : Params} {c : Commitment} {e : Evidence} {i : ℕ} {a : Audit} {reg : Lean.Json}
    {x : Core.Answer} (ha : p.audits[i]? = some a) (hreg : c.audits[i]? = some reg) (hx : e.audits[i]? = some x) :
    (auditChecks p c e)[i]? = some (auditCheck i a reg x) := by
  have hz : ((p.audits.zip c.audits).zip e.audits)[i]? = some ((a, reg), x) :=
    Array.getElem?_zip_eq_some.mpr ⟨Array.getElem?_zip_eq_some.mpr ⟨ha, hreg⟩, hx⟩
  simp [auditChecks, Array.getElem?_zipIdx, hz]

/-- An audit the check accepts has its answer's facts. -/
theorem auditCheck_facts {i : ℕ} {a : Audit} {reg : Lean.Json} {x : Core.Answer} {f : Facts}
    (h : auditCheck i a reg x = .ok f) : AuditFacts i reg x f := by
  unfold auditCheck at h
  cases ha : answered s!"audit {i}" reg x with
  | error r =>
    simp only [ha, bind, Except.bind] at h
    by_cases c1 : (reg.getObjValD "program" == Lean.Json.str a.program) = true <;>
      simp [c1, refuse] at h
  | ok pp =>
    obtain ⟨prof, pub⟩ := pp
    cases hf : factsOf s!"audit {i}" reg prof pub with
    | error r =>
      simp only [ha, hf, bind, Except.bind] at h
      by_cases c1 : (reg.getObjValD "program" == Lean.Json.str a.program) = true <;>
        simp [c1, refuse] at h
    | ok g =>
      refine ⟨prof, pub, ha, ?_⟩
      simp only [ha, hf, bind, Except.bind] at h
      by_cases c1 : (reg.getObjValD "program" == Lean.Json.str a.program) = true <;>
        by_cases c2 : g.units = a.units <;> simp_all [refuse, pure, Except.pure]

/-- The facts `factsOf` gives are of the sample the registration states, one its units hold, and charge it: it
refuses a profile that names more sampled than its level holds, one with a stratum that samples more than it holds,
one whose strata's sizes do not sum to the level's and one that is another sample than the registration's, and the
count it reads is the larger of the stated bound and the sample's (`sampleBound`). -/
theorem factsOf_sample {what : String} {reg : Lean.Json} {prof : Core.Integrity} {pub : Lean.Json} {f : Facts}
    (h : factsOf what reg prof pub = .ok f) : SampleHeld reg f ∧ SampleCharged reg f := by
  have one : ∀ (body : Core.Level → Found Facts) (alt : Found Facts),
      (match prof.levels with | #[l] => body l | _ => alt) = .ok f → alt = .ok f ∨ ∃ l, body l = .ok f := by
    intro body alt hm
    rcases hls : prof.levels with ⟨_ | ⟨l, _ | ⟨l', ls⟩⟩⟩ <;> rw [hls] at hm
    · exact .inl hm
    · exact .inr ⟨l, hm⟩
    · exact .inl hm
  unfold factsOf at h
  rcases one _ _ h with h | ⟨l, h⟩
  · cases h
  rcases hd : l.draw with _ | (p | k | ss | ⟨w, ws⟩) <;>
    simp only [hd, bind, Except.bind, pure, Except.pure, refuse] at h <;>
    (repeat' split at h) <;> cases h
  · obtain ⟨hn, hreg⟩ := ‹_ = l.size ∧ _›
    cases hn
    cases hreg
    exact ⟨⟨l.size, .subset k, by assumption, rfl, rfl, by assumption, .inl ⟨k, rfl, rfl⟩⟩,
      l.size, .subset k, by assumption, .inl ⟨k, rfl, Nat.le_max_right _ _⟩⟩
  · obtain ⟨hn, hreg⟩ := ‹_ = l.size ∧ _›
    cases hn
    cases hreg
    have hnone : ∀ st, ss.find? (fun st => decide (st.size < st.k)) = some st → False := by assumption
    have hsum : ss.foldl (fun n st => n + st.size) 0 = l.size := by assumption
    refine ⟨⟨l.size, .stratified ss, by assumption, rfl, rfl, by assumption,
        .inr ⟨ss, rfl, rfl, fun st m => ?_, hsum⟩⟩,
      l.size, .stratified ss, by assumption, .inr ⟨ss, rfl, Nat.le_max_right _ _⟩⟩
    have := Array.find?_eq_none.mp (Option.eq_none_iff_forall_ne_some.mpr hnone) st m
    simpa using this

/-- Strata that each sample no more than they hold, and in all as many as they hold, are each sampled whole. -/
theorem strata_whole (xs : List Core.Stratum) (hle : ∀ st ∈ xs, st.k ≤ st.size) (a b : ℕ) (hab : a ≤ b)
    (h : xs.foldl (fun n st => n + st.k) a = xs.foldl (fun n st => n + st.size) b) :
    a = b ∧ ∀ st ∈ xs, st.k = st.size := by
  induction xs generalizing a b with
  | nil => exact ⟨h, fun _ m => nomatch m⟩
  | cons x xs ih =>
    have hx := hle x List.mem_cons_self
    obtain ⟨e, r⟩ := ih (fun st m => hle st (List.mem_cons_of_mem _ m)) (a + x.k) (b + x.size) (by omega) h
    refine ⟨by omega, fun st m => ?_⟩
    rcases List.mem_cons.mp m with rfl | m
    · omega
    · exact r st m

/-- Facts of a registered stratified sample that is sampled whole are of every stratum sampled whole. -/
theorem sampleHeld_whole {reg : Lean.Json} {f : Facts} {n : ℕ} {ss : Array Core.Stratum} (h : SampleHeld reg f)
    (hs : sampleOf reg = some (n, .stratified ss)) (hw : f.drawn = f.units) : ∀ st ∈ ss, st.k = st.size := by
  obtain ⟨n', d, hs', hu, -, -, hd⟩ := h
  rw [hs] at hs'
  cases hs'
  rcases hd with ⟨k, hk, -⟩ | ⟨ss', hss, hdr, hle, hsum⟩
  · cases hk
  · cases hss
    have hk : ss.toList.foldl (fun n st => n + st.k) 0 = ss.toList.foldl (fun n st => n + st.size) 0 := by
      rw [Array.foldl_toList, Array.foldl_toList]
      omega
    exact fun st m => (strata_whole ss.toList (fun st m => hle st (Array.mem_toList_iff.mp m)) 0 0 (Nat.le_refl 0)
      hk).2 st (Array.mem_toList_iff.mpr m)

/-- The rows form's check of a scan, where it accepts, has the scan's answer's facts. -/
theorem rowsExecutionPhaseCheck_facts {p : Params} {s : Schedule} {sd : ByteArray} {audits : Array Lean.Json} {w : ℕ}
    {reg : Lean.Json} {x : ExecutionPhase} {u : Facts × ℕ} (h : rowsExecutionPhaseCheck p s sd audits w reg x = .ok u) :
    ∃ (what : String) (prof : Core.Integrity) (pub : Lean.Json), factsOf what reg prof pub = .ok u.1 := by
  unfold rowsExecutionPhaseCheck at h
  simp only [bind, Except.bind, pure, Except.pure, refuse] at h
  repeat' split at h
  all_goals cases h
  all_goals exact ⟨_, _, _, by assumption⟩

/-- A link-window with no finding has its scan's facts and its tied audits' free inputs. -/
theorem rowsWindows_ok {p : Params} {c : Commitment} {e : Evidence} {s : Schedule} {sd : ByteArray}
    (hws : ∀ y ∈ rowsWindows p c e s sd, findingOf y = none) :
    ∀ y ∈ rowsWindows p c e s sd, ∃ u, y = .ok (some u) := by
  intro y hy
  obtain ⟨v, rfl⟩ := findingOf_none (hws y hy)
  unfold rowsWindows at hy
  obtain ⟨z, -, hz⟩ := Array.mem_map.mp hy
  cases hr : rowsExecutionPhaseCheck p s sd c.audits (p.first + z.2) z.1.1 z.1.2 with
  | error r => simp [hr, Functor.map, Except.map] at hz
  | ok u => exact ⟨u, by simpa [hr, Functor.map, Except.map] using hz.symm⟩

/-- Facts of a registered sample, read at any δ, charge that sample's bound there: the reading's count is the larger
of the facts' own and the bound of the facts' design at that δ, and the design is the registered one. -/
theorem sampleCharged_readAt {reg : Lean.Json} {f : Facts} (h : SampleHeld reg f) (δ : Core.Ratio) :
    SampleCharged reg (f.readAt δ) := by
  obtain ⟨u, dr, δ₀, w, dw⟩ := f
  obtain ⟨n, d, hs, hu, hdesign, -, hd⟩ := h
  obtain rfl : u = n := hu
  obtain rfl : dw = d := hdesign
  refine ⟨_, _, hs, ?_⟩
  rcases hd with ⟨k, hk, -⟩ | ⟨ss, hss, -⟩
  · cases hk
    exact .inl ⟨k, rfl, Nat.le_max_right _ _⟩
  · cases hss
    exact .inr ⟨ss, rfl, Nat.le_max_right _ _⟩

/-- With every audit accepted, the audits the rows form charges are those the accepted facts name: tied and sampled in
part, each read at the period's own δ. -/
theorem charged_eq {p : Params} {c : Commitment} {s : Schedule} {sd : ByteArray} {scan : Scan}
    (hform : p.form = .rows s sd scan) {as : Array (Found Facts)} (has : ∀ y ∈ as, findingOf y = none) :
    (Array.range as.size).filterMap (chargedOf p c as) =
      tiedInPart (tiedIn c.audits c.windows) (as.filterMap (·.toOption)) := by
  have hall : ∀ y ∈ as, (Except.toOption y).isSome = true := by
    intro y hy
    obtain ⟨v, rfl⟩ := findingOf_none (has y hy)
    rfl
  unfold tiedInPart
  rw [size_filterMap_of_isSome _ _ hall]
  congr 1
  funext i
  rw [getElem?_filterMap_of_isSome _ _ hall]
  unfold chargedOf
  cases hi : as[i]? with
  | none => rfl
  | some y =>
    obtain ⟨f, rfl⟩ := findingOf_none (has y (Array.mem_of_getElem? hi))
    have hf : (some (Except.ok f : Found Facts)).bind Except.toOption = some f := rfl
    simp only [hform, tiedOf, hf, Option.filter]
    by_cases h : (tiedIn c.audits c.windows i && decide (f.drawn < f.units)) = true <;>
      simp only [h, ↓reduceIte] <;> rfl

theorem rowsTiedCharged : RowsTiedCharged := by
  intro p c e pub prof h s sd scan hform
  obtain ⟨hca, hew, hea, hws, has, rfl⟩ := check_rows hform h
  refine ⟨charged_eq hform has, ?_⟩
  intro i reg hreg htied
  obtain ⟨hlt, -⟩ := Array.getElem?_eq_some_iff.mp hreg
  have hi : i < p.audits.size := by omega
  have hx : e.audits[i]? = some (e.audits[i]'(by omega)) := Array.getElem?_eq_getElem (by omega)
  have hget := auditChecks_getElem? (Array.getElem?_eq_getElem hi) hreg hx
  obtain ⟨f, hf⟩ := findingOf_none (has _ (Array.mem_of_getElem? hget))
  rw [hf] at hget
  have hall : ∀ y ∈ auditChecks p c e, (Except.toOption y).isSome = true := by
    intro y hy
    obtain ⟨v, rfl⟩ := findingOf_none (has y hy)
    rfl
  have hfacts := auditCheck_facts hf
  have hheld : SampleHeld reg f := by
    obtain ⟨_, _, -, hfo⟩ := hfacts
    exact (factsOf_sample hfo).1
  refine ⟨_, f, hx, hfacts, ?_, ?_, ?_, ?_⟩
  · show ((auditChecks p c e).filterMap (·.toOption))[i]? = some f
    rw [getElem?_filterMap_of_isSome _ _ hall, hget]
    rfl
  · obtain ⟨_, _, -, hu, -, hle, -⟩ := hheld
    omega
  · exact fun hw _ _ hs => sampleHeld_whole hheld hs hw
  · intro hd
    have hch : chargedOf p c (auditChecks p c e) i = some (f.readAt tiedDelta) := by
      simp [chargedOf, hget, hform, tiedOf, htied, hd]
    refine ⟨?_, sampleCharged_readAt hheld tiedDelta, ?_⟩
    · show f.readAt tiedDelta ∈
        (Array.range (auditChecks p c e).size).filterMap (chargedOf p c (auditChecks p c e))
      exact Array.mem_filterMap.mpr ⟨i, Array.mem_range.mpr (Array.getElem?_eq_some_iff.mp hget).1, hch⟩
    · intro hwrong w wreg hw hc
      obtain ⟨hwlt, -⟩ := Array.getElem?_eq_some_iff.mp hw
      have hxw : e.windows[w]? = some (e.windows[w]'(by omega)) := Array.getElem?_eq_getElem (by omega)
      have hz : (c.windows.zip e.windows)[w]? = some (wreg, e.windows[w]'(by omega)) :=
        Array.getElem?_zip_eq_some.mpr ⟨hw, hxw⟩
      have hwget : (rowsWindows p c e s sd)[w]? =
          some (some <$> rowsExecutionPhaseCheck p s sd c.audits (p.first + w) wreg (e.windows[w]'(by omega))) := by
        simp [rowsWindows, Array.getElem?_zipIdx, hz]
      obtain ⟨u, hu⟩ := rowsWindows_ok hws _ (Array.mem_of_getElem? hwget)
      rw [hu] at hwget
      have hwall : ∀ y ∈ rowsWindows p c e s sd, (y.toOption.join).isSome = true := by
        intro y hy
        obtain ⟨u, rfl⟩ := rowsWindows_ok hws y hy
        rfl
      have hscan : ((rowsWindows p c e s sd).filterMap (·.toOption.join))[w]? = some u := by
        rw [getElem?_filterMap_of_isSome _ _ hwall, hwget]
        rfl
      refine ⟨scan.terms u.1 u.2 (faultyEntries p c (auditChecks p c e) scan wreg) p.jitter, ?_, ?_⟩
      · have hzip : (((rowsWindows p c e s sd).filterMap (·.toOption.join)).zip c.windows)[w]? = some (u, wreg) :=
          Array.getElem?_zip_eq_some.mpr ⟨hscan, hw⟩
        simp [rowsPublic, rowsTerms, hzip]
      · show faultyEntries p c (auditChecks p c e) scan wreg = scan.entries
        have hany : ((tiedAudits c.audits wreg).any fun i =>
            (chargedOf p c (auditChecks p c e) i).any (1 ≤ ·.wrong)) = true :=
          Array.any_eq_true'.mpr ⟨i, Array.mem_of_contains_eq_true hc, by simp [hch, hwrong]⟩
        simp [faultyEntries, hany]

/-- An answer of another protocol than one-stage's. -/
def otherAnswer : Core.Answer :=
  { protocol := "other", verdict := .accept, reason := none, digest := none, published := none, profile := none }

/-- The parameters of one compute audit, the slot audit, under `slotSchedule` and a scan of 7 slot entry bits. -/
def chargedParams : Params :=
  ⟨#[slotAudit], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", .rows slotSchedule .empty ⟨0, 0, 0, 0, 7⟩, 1⟩

/-- Its commitment: the slot audit's registration and the link-window whose scan's `prog` is its entries port. -/
def chargedCommitment : Commitment := ⟨#[slotReg], #[tiedExecutionPhase], .empty⟩

/-- Its honest side, at the slot audit tied to `tiedExecutionPhase`: sampled 1 of its 2 units with a bound of one incorrect
proof block, the audit is charged, read at the period's own δ, where the one unit it leaves may be incorrect, and its
link-window's `β_w` is the scan's every entry bit; sampled whole, it is not charged and `β_w` is 0. -/
theorem rowsTiedCharged_honest :
    (chargedOf chargedParams chargedCommitment #[.ok ⟨2, 1, ⟨0, 1⟩, 1, .subset 1⟩] 0).isSome = true ∧
      faultyEntries chargedParams chargedCommitment #[.ok ⟨2, 1, ⟨0, 1⟩, 1, .subset 1⟩] ⟨0, 0, 0, 0, 7⟩
        tiedExecutionPhase = 7 ∧
      (chargedOf chargedParams chargedCommitment #[.ok ⟨2, 2, ⟨0, 1⟩, 0, .subset 2⟩] 0).isNone = true ∧
      faultyEntries chargedParams chargedCommitment #[.ok ⟨2, 2, ⟨0, 1⟩, 0, .subset 2⟩] ⟨0, 0, 0, 0, 7⟩
        tiedExecutionPhase = 0 := by
  decide +kernel

/-- **No period check that accepts every period has `RowsTiedCharged`'s property** (its nontrivial lemma): such a check
accepts a rows-form period whose audit 0 is tied to its link-window's scan, and publishes no facts of it and no
charge. -/
theorem rowsTiedCharged_nontrivial : ¬ RowsTiedChargedOf acceptAllRows := by
  intro h
  let p : Params :=
    ⟨#[slotAudit], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 1⟩
  obtain ⟨-, h1⟩ := h p ⟨#[slotReg], #[tiedExecutionPhase], .empty⟩ ⟨#[otherAnswer], #[]⟩ _ _ rfl slotSchedule .empty
    ⟨0, 0, 0, 0, 0⟩ rfl
  obtain ⟨x, f, -, -, hf, -⟩ := h1 0 slotReg rfl rowsProgWired_honest.2
  simp [rowsTerms] at hf

/-! ## The published response cap -/

theorem rowsCapPublished : RowsCapPublished := by
  intro p c e pub prof h
  unfold check at h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  cases hf : p.form <;> simp only [hf] at h <;> split at h <;> cases h <;> simp [CapPublished, hf]

/-- **`RowsCapPublished`'s honest lemma**: the period check accepts a period under a schedule with a response cap,
`cappedPeriod` with nothing registered. A period with link-windows needs the network certifier's LMS signatures and
one-stage's answers, which the period's difftest builds and checks against the mirror. -/
theorem rowsCapPublished_honest :
    ∃ (pub : Public) (prof : Profile), check cappedPeriod ⟨#[], #[], .empty⟩ ⟨#[], #[]⟩ = .accept pub prof := by
  have hb : (match check cappedPeriod ⟨#[], #[], .empty⟩ ⟨#[], #[]⟩ with
      | .accept _ _ => true
      | _ => false) = true := by
    decide +kernel
  split at hb
  · exact ⟨_, _, ‹_›⟩
  · cases hb

/-- **`RowsCapPublished`'s nontrivial lemma**: the public `acceptAllRows` publishes for `cappedPeriod`, the rows form's
terms of no link-window, names no cap, and the period's schedule has one. -/
theorem rowsCapPublished_nontrivial :
    ¬ CapPublished cappedPeriod
      (rowsTerms #[] ⟨0, 0, 0, 0, 0⟩ #[] cappedPeriod.cells cappedPeriod.jitter cappedPeriod.windows) := by
  intro h
  cases h

/-- **No period check that accepts every period has `RowsCapPublished`'s property.** -/
theorem rowsCapPublished_acceptAll : ¬ RowsCapPublishedOf acceptAllRows :=
  fun h => rowsCapPublished_nontrivial (h cappedPeriod ⟨#[], #[], .empty⟩ ⟨#[], #[]⟩ _ _ rfl)

/-! ## Every answer is of its registered sample and charges its bound -/

/-- What acceptance in the clock form gives: the sizes, no finding of an audit, and the public, the clock form's
(`terms`) over the accepted audits' facts. -/
theorem check_clock {p : Params} {c : Commitment} {e : Evidence} {pub : Public} {prof : Profile}
    {ss : NetworkCertifier.SyncSet} (hform : p.form = .clock ss) (h : check p c e = .accept pub prof) :
    c.audits.size = p.audits.size ∧ e.audits.size = c.audits.size ∧
      (∀ y ∈ auditChecks p c e, findingOf y = none) ∧
      pub = { terms p.audits ((auditChecks p c e).filterMap (·.toOption)) p.cells p.jitter
        (ss.deltas.length * ss.rhos.length) p.windows with maxTokensCap := p.form.maxTokensCap } := by
  unfold check at h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  rename_i h1 h2 h3 h4
  simp only [hform] at h
  split at h
  · cases h
  · cases h
  · rename_i hF
    have h0 := firstFinding_none hF
    refine ⟨by omega, by omega, ?_, ?_⟩
    · intro y hy
      exact h0 _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _
        (List.mem_map_of_mem (Array.mem_toList_iff.mpr hy)))))
    · injection h with hp
      rw [hform]
      exact hp.symm

/-- What acceptance gives of the audits in either form: the sizes, no finding of an audit, and the public's audits'
facts, the accepted ones'. -/
theorem check_audits {p : Params} {c : Commitment} {e : Evidence} {pub : Public} {prof : Profile}
    (h : check p c e = .accept pub prof) :
    c.audits.size = p.audits.size ∧ e.audits.size = c.audits.size ∧
      (∀ y ∈ auditChecks p c e, findingOf y = none) ∧
      pub.audits = (auditChecks p c e).filterMap (·.toOption) := by
  cases hform : p.form with
  | clock ss =>
    obtain ⟨hca, hea, has, rfl⟩ := check_clock hform h
    exact ⟨hca, hea, has, rfl⟩
  | rows s sd scan =>
    obtain ⟨hca, -, hea, -, has, rfl⟩ := check_rows hform h
    exact ⟨hca, hea, has, rfl⟩

/-- What `factsOf` gives of every answer holds of an accepted period's public: of each audit's facts, which are its
term's in the clock form, and in the rows form of each scan's. -/
theorem answersAt {P : Lean.Json → Facts → Prop}
    (hP : ∀ (what : String) (reg : Lean.Json) (prof : Core.Integrity) (pub : Lean.Json) (f : Facts),
      factsOf what reg prof pub = .ok f → P reg f)
    {p : Params} {c : Commitment} {e : Evidence} {pub : Public} {prof : Profile}
    (h : check p c e = .accept pub prof) : AnswersAt P p c pub := by
  refine ⟨?_, ?_⟩
  · intro i reg hreg
    obtain ⟨hca, hea, has, haud⟩ := check_audits h
    obtain ⟨hlt, -⟩ := Array.getElem?_eq_some_iff.mp hreg
    have hi : i < p.audits.size := by omega
    have hx : e.audits[i]? = some (e.audits[i]'(by omega)) := Array.getElem?_eq_getElem (by omega)
    have hget := auditChecks_getElem? (Array.getElem?_eq_getElem hi) hreg hx
    obtain ⟨f, hf⟩ := findingOf_none (has _ (Array.mem_of_getElem? hget))
    rw [hf] at hget
    have hall : ∀ y ∈ auditChecks p c e, (Except.toOption y).isSome = true := by
      intro y hy
      obtain ⟨v, rfl⟩ := findingOf_none (has y hy)
      rfl
    have hfi : ((auditChecks p c e).filterMap (·.toOption))[i]? = some f := by
      rw [getElem?_filterMap_of_isSome _ _ hall, hget]
      rfl
    obtain ⟨_, _, -, hfo⟩ := auditCheck_facts hf
    refine ⟨f, haud ▸ hfi, hP _ _ _ _ _ hfo, ?_⟩
    intro ss hform
    obtain ⟨-, -, -, rfl⟩ := check_clock hform h
    have hz : (p.audits.zip ((auditChecks p c e).filterMap (·.toOption)))[i]? = some (p.audits[i], f) :=
      Array.getElem?_zip_eq_some.mpr ⟨Array.getElem?_eq_getElem hi, hfi⟩
    exact ⟨AuditTerms.of p.audits[i] f p.jitter, by simp [terms, hz], rfl⟩
  · intro s sd scan hform w reg hw
    obtain ⟨hca, hew, hea, hws, has, rfl⟩ := check_rows hform h
    obtain ⟨hwlt, -⟩ := Array.getElem?_eq_some_iff.mp hw
    have hxw : e.windows[w]? = some (e.windows[w]'(by omega)) := Array.getElem?_eq_getElem (by omega)
    have hz : (c.windows.zip e.windows)[w]? = some (reg, e.windows[w]'(by omega)) :=
      Array.getElem?_zip_eq_some.mpr ⟨hw, hxw⟩
    have hwget : (rowsWindows p c e s sd)[w]? =
        some (some <$> rowsExecutionPhaseCheck p s sd c.audits (p.first + w) reg (e.windows[w]'(by omega))) := by
      simp [rowsWindows, Array.getElem?_zipIdx, hz]
    obtain ⟨u, hu⟩ := rowsWindows_ok hws _ (Array.mem_of_getElem? hwget)
    have hrc : rowsExecutionPhaseCheck p s sd c.audits (p.first + w) reg (e.windows[w]'(by omega)) = .ok u := by
      cases hr : rowsExecutionPhaseCheck p s sd c.audits (p.first + w) reg (e.windows[w]'(by omega)) with
      | error r => simp [hr, Functor.map, Except.map] at hu
      | ok v => simpa [hr, Functor.map, Except.map] using hu
    rw [hu] at hwget
    have hwall : ∀ y ∈ rowsWindows p c e s sd, (y.toOption.join).isSome = true := by
      intro y hy
      obtain ⟨u, rfl⟩ := rowsWindows_ok hws y hy
      rfl
    have hscan : ((rowsWindows p c e s sd).filterMap (·.toOption.join))[w]? = some u := by
      rw [getElem?_filterMap_of_isSome _ _ hwall, hwget]
      rfl
    have hzip : (((rowsWindows p c e s sd).filterMap (·.toOption.join)).zip c.windows)[w]? = some (u, reg) :=
      Array.getElem?_zip_eq_some.mpr ⟨hscan, hw⟩
    obtain ⟨_, _, _, hfo⟩ := rowsExecutionPhaseCheck_facts hrc
    exact ⟨scan.terms u.1 u.2 (faultyEntries p c (auditChecks p c e) scan reg) p.jitter,
      by simp [rowsPublic, rowsTerms, hzip], hP _ _ _ _ _ hfo⟩

theorem registeredSampleHeld : RegisteredSampleHeld :=
  fun _ _ _ _ _ h => answersAt (fun _ _ _ _ _ hf => (factsOf_sample hf).1) h

theorem sampleBoundCharged : SampleBoundCharged :=
  fun _ _ _ _ _ h => answersAt (fun _ _ _ _ _ hf => (factsOf_sample hf).2) h

/-- **`RegisteredSampleHeld`'s honest lemma**: the period check accepts a period, `cappedPeriod` with nothing
registered, and of an answer whose profile is the sample its registration states, two strata of 8 units with 4 of
each sampled, the facts it reads are of the 16 units with 8 sampled. A period with an audit needs one-stage's answer
at its window record's digest, which the period's difftest builds and checks against the mirror. -/
theorem registeredSampleHeld_honest :
    (∃ (pub : Public) (prof : Profile), check cappedPeriod ⟨#[], #[], .empty⟩ ⟨#[], #[]⟩ = .accept pub prof) ∧
      (factsOf "audit 0" halvesReg halvesProfile (statedBound 0)).toOption.map (fun f => (f.units, f.drawn)) =
        some (16, 8) := by
  refine ⟨rowsCapPublished_honest, ?_⟩
  decide +kernel

/-- **`RegisteredSampleHeld`'s nontrivial lemma, at the level**: no facts are of a registration that states 3
sampled of 2 units, so a period check that accepts an audit registered so and publishes its facts as stated, 3 of 2
units sampled, fails `AnswersAt SampleHeld`. The registration states no stratum, so the other two clauses hold. -/
theorem registeredSampleHeld_nontrivial :
    ¬ AnswersAt SampleHeld cappedPeriod ⟨#[sampleReg 2 (subsetDesign 3)], #[], .empty⟩
      (statedPublic ⟨2, 3, ⟨1, 10⟩, 0, .subset 3⟩) := by
  intro h
  obtain ⟨f, -, ⟨n, d, hs, -, -, hle, hd⟩, -⟩ := h.1 0 (sampleReg 2 (subsetDesign 3)) rfl
  have hreg : sampleOf (sampleReg 2 (subsetDesign 3)) = some (2, .subset 3) := by decide +kernel
  rw [hreg] at hs
  cases hs
  rcases hd with ⟨k, hk, hdr⟩ | ⟨ss, hss, -⟩
  · cases hk
    omega
  · cases hss

/-- **`RegisteredSampleHeld` fails at a stratum sampled above its size**: no facts are of a registration whose
stratum `a` states 2 sampled of its 1 unit, in 3 units of which 2 are sampled and whose strata's sizes sum to 3, so
a period check that accepts an audit registered so fails `AnswersAt SampleHeld` on that clause alone. -/
theorem registeredSampleHeld_stratum :
    ¬ AnswersAt SampleHeld cappedPeriod ⟨#[sampleReg 3 (strataDesign 1 2 2 0)], #[], .empty⟩
      (statedPublic ⟨3, 2, ⟨1, 10⟩, 0, .stratified #[⟨"a", 1, 2⟩, ⟨"b", 2, 0⟩]⟩) := by
  intro h
  obtain ⟨f, -, ⟨n, d, hs, -, -, -, hd⟩, -⟩ := h.1 0 (sampleReg 3 (strataDesign 1 2 2 0)) rfl
  have hreg : sampleOf (sampleReg 3 (strataDesign 1 2 2 0)) = some (3, .stratified #[⟨"a", 1, 2⟩, ⟨"b", 2, 0⟩]) := by
    decide +kernel
  rw [hreg] at hs
  cases hs
  rcases hd with ⟨k, hk, -⟩ | ⟨ss, hss, -, hle, -⟩
  · cases hk
  · cases hss
    have := hle ⟨"a", 1, 2⟩ (by simp)
    simp at this

/-- **`RegisteredSampleHeld` fails at strata whose sizes do not sum to the units**: no facts are of a registration
of 2 units whose two strata hold 2 units each, 1 of each sampled, so no more sampled than the units and no stratum
above its size, and a period check that accepts an audit registered so fails `AnswersAt SampleHeld` on that clause
alone. -/
theorem registeredSampleHeld_sizes :
    ¬ AnswersAt SampleHeld cappedPeriod ⟨#[sampleReg 2 (strataDesign 2 1 2 1)], #[], .empty⟩
      (statedPublic ⟨2, 2, ⟨1, 10⟩, 0, .stratified #[⟨"a", 2, 1⟩, ⟨"b", 2, 1⟩]⟩) := by
  intro h
  obtain ⟨f, -, ⟨n, d, hs, -, -, -, hd⟩, -⟩ := h.1 0 (sampleReg 2 (strataDesign 2 1 2 1)) rfl
  have hreg : sampleOf (sampleReg 2 (strataDesign 2 1 2 1)) = some (2, .stratified #[⟨"a", 2, 1⟩, ⟨"b", 2, 1⟩]) := by
    decide +kernel
  rw [hreg] at hs
  cases hs
  rcases hd with ⟨k, hk, -⟩ | ⟨ss, hss, -, -, hsum⟩
  · cases hk
  · cases hss
    exact absurd hsum (by decide +kernel)

/-- **`SampleBoundCharged`'s honest lemma**: the period check accepts a period, `cappedPeriod` with nothing
registered, and of an answer for two registered strata of 4 of 8 units at δ = 1/10 whose public states a bound of 0,
the count it reads is 6: each stratum's own bound at δ = 1/20 is 3. A period with an audit needs one-stage's answer
at its window record's digest, which the period's difftest builds and checks against the mirror. -/
theorem sampleBoundCharged_honest :
    (∃ (pub : Public) (prof : Profile), check cappedPeriod ⟨#[], #[], .empty⟩ ⟨#[], #[]⟩ = .accept pub prof) ∧
      (factsOf "audit 0" halvesReg halvesProfile (statedBound 0)).toOption.map (·.wrong) = some 6 := by
  refine ⟨rowsCapPublished_honest, ?_⟩
  decide +kernel

/-- **`SampleBoundCharged`'s nontrivial lemma**: a period check that accepts an audit registered at two strata of 4
of 8 units and publishes its facts with a count of 5 at δ = 1/10, below the 6 its strata's bounds sum to, fails
`AnswersAt SampleCharged`. -/
theorem sampleBoundCharged_nontrivial :
    ¬ AnswersAt SampleCharged cappedPeriod ⟨#[halvesReg], #[], .empty⟩
      (statedPublic ⟨16, 8, ⟨1, 10⟩, 5, .stratified #[⟨"a", 8, 4⟩, ⟨"b", 8, 4⟩]⟩) := by
  intro h
  obtain ⟨f, hf, ⟨n, d, hs, hd⟩, -⟩ := h.1 0 halvesReg rfl
  obtain rfl : f = ⟨16, 8, ⟨1, 10⟩, 5, .stratified #[⟨"a", 8, 4⟩, ⟨"b", 8, 4⟩]⟩ := by
    simpa [statedPublic, rowsTerms] using hf.symm
  have hreg : sampleOf halvesReg = some (16, .stratified #[⟨"a", 8, 4⟩, ⟨"b", 8, 4⟩]) := by decide +kernel
  rw [hreg] at hs
  cases hs
  rcases hd with ⟨k, hk, -⟩ | ⟨ss, hss, hb⟩
  · cases hk
  · cases hss
    exact absurd hb (by decide +kernel)

/-- **No period check that accepts every period has `AnswersOf`'s property**, whatever it says of an answer: such a
check accepts a period with an audit and publishes no facts of it. -/
theorem answers_acceptAll (P : Lean.Json → Facts → Prop) : ¬ AnswersOf P acceptAllRows := by
  intro h
  obtain ⟨f, hf, -⟩ := (h cappedPeriod ⟨#[Lean.Json.null], #[], .empty⟩ ⟨#[], #[]⟩ _ _ rfl).1 0 _ rfl
  simp [rowsTerms] at hf

end Properties.NetworkPeriod
