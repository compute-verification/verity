module

public import Properties.NetworkPeriod
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import all Init.Data.Array.Basic
import all Init.Data.Repr
public import Mathlib.Algebra.GroupWithZero.Nat
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Each form's `K` bounds `PeriodLeakAccepted`'s count: the proofs of `ClockKBoundsCount` and `RowsKBoundsCount`

Each factor is at most 2 to the power of the ceiling of its `log2` (`le_two_pow_clog2`), and `K`'s sums over the
connections and the windows (the compute audits, or the scans) are sums over `Fin m` and `Fin n` (`Array.ofFn`), so
`2 ^ K` is the product of the factors' bounds.

`RowsProgWired` and `RowsStateTied` read the check's refusals off its findings: on acceptance every finding is none
(`firstFinding_none`), each link-window's `wiredCheck` and `stateCheck` among them.

`RowsTiedCharged` reads the public the rows form accepts with (`check_rows`): with no finding, every audit's check gave
its facts and every link-window's its scan's, so the public's `audits` and `units` hold them position for position
(`getElem?_filterMap_of_isSome`), `charged` is `chargedOf` over the audits (`charged_eq`), and a link-window's term's
`faultyEntries` is `faultyEntries` of its registration. An audit's facts are `factsOf`'s, which refuses a profile
sampling more than its level holds (`factsOf_sampled_le_units`). It also refuses a stratified profile with a stratum
sampling more than it holds, and one whose strata's sizes do not sum to the level's, so facts with `drawn = units` are
of a profile whose every stratum is sampled whole (`factsOf_stratified_whole`, by `strata_whole`).

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
      (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (Array.mem_toList_iff.mpr hmem))))
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

def slotSchedule : Schedule := ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, none⟩

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
    have h0 := firstFinding_none hF _
      (List.mem_append_left _ (List.mem_append_right _ (Array.mem_toList_iff.mpr hmem)))
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
    some <$> rowsWindowCheck p s sd c.audits (p.first + i) reg x

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
        (List.mem_append_left _ (List.mem_map_of_mem (Array.mem_toList_iff.mpr hy)))))))
    · intro y hy
      exact h0 _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _
        (List.mem_map_of_mem (Array.mem_toList_iff.mpr hy))))))
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
    cases hf : factsOf s!"audit {i}" prof pub with
    | error r =>
      simp only [ha, hf, bind, Except.bind] at h
      by_cases c1 : (reg.getObjValD "program" == Lean.Json.str a.program) = true <;>
        simp [c1, refuse] at h
    | ok g =>
      refine ⟨prof, pub, ha, ?_⟩
      simp only [ha, hf, bind, Except.bind] at h
      by_cases c1 : (reg.getObjValD "program" == Lean.Json.str a.program) = true <;>
        by_cases c2 : g.units = a.units <;> simp_all [refuse, pure, Except.pure]

/-- The facts `factsOf` gives sample no more than their level holds: it refuses a profile that names more. -/
theorem factsOf_sampled_le_units {what : String} {prof : Core.Integrity} {pub : Lean.Json} {f : Facts}
    (h : factsOf what prof pub = .ok f) : f.drawn ≤ f.units := by
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
  · simp only [bind, Except.bind, pure, Except.pure, refuse] at h
    repeat' split at h
    all_goals cases h
    all_goals assumption

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

/-- The facts `factsOf` gives of a stratified profile with `drawn = units` are of every stratum sampled whole: it
refuses a profile with a stratum that samples more than it holds, and one whose strata's sizes do not sum to the
level's. -/
theorem factsOf_stratified_whole {what : String} {prof : Core.Integrity} {pub : Lean.Json} {f : Facts}
    {l : Core.Level} {ss : Array Core.Stratum} (h : factsOf what prof pub = .ok f) (hl : prof.levels = #[l])
    (hs : l.draw = some (.stratified ss)) (hw : f.drawn = f.units) : ∀ st ∈ ss, st.k = st.size := by
  have one : ∀ (body : Core.Level → Found Facts) (alt : Found Facts),
      (match prof.levels with | #[l] => body l | _ => alt) = .ok f → body l = .ok f := by
    intro body alt hm
    rw [hl] at hm
    exact hm
  unfold factsOf at h
  replace h := one _ _ h
  simp only [hs, bind, Except.bind, pure, Except.pure, refuse] at h
  repeat' split at h
  all_goals cases h
  have hnone : ∀ st, ss.find? (fun st => decide (st.size < st.k)) = some st → False := by assumption
  have hsum : ss.foldl (fun n st => n + st.size) 0 = l.size := by assumption
  have hle : ∀ st ∈ ss.toList, st.k ≤ st.size := fun st m => by
    have := Array.find?_eq_none.mp (Option.eq_none_iff_forall_ne_some.mpr hnone) st (Array.mem_toList_iff.mp m)
    simpa using this
  have hk : ss.toList.foldl (fun n st => n + st.k) 0 = ss.toList.foldl (fun n st => n + st.size) 0 := by
    rw [Array.foldl_toList, Array.foldl_toList]
    exact hw.trans hsum.symm
  exact fun st m => (strata_whole ss.toList hle 0 0 (Nat.le_refl 0) hk).2 st (Array.mem_toList_iff.mpr m)

/-- A link-window with no finding has its scan's facts and its tied audits' free inputs. -/
theorem rowsWindows_ok {p : Params} {c : Commitment} {e : Evidence} {s : Schedule} {sd : ByteArray}
    (hws : ∀ y ∈ rowsWindows p c e s sd, findingOf y = none) :
    ∀ y ∈ rowsWindows p c e s sd, ∃ u, y = .ok (some u) := by
  intro y hy
  obtain ⟨v, rfl⟩ := findingOf_none (hws y hy)
  unfold rowsWindows at hy
  obtain ⟨z, -, hz⟩ := Array.mem_map.mp hy
  cases hr : rowsWindowCheck p s sd c.audits (p.first + z.2) z.1.1 z.1.2 with
  | error r => simp [hr, Functor.map, Except.map] at hz
  | ok u => exact ⟨u, by simpa [hr, Functor.map, Except.map] using hz.symm⟩

/-- With every audit accepted, the audits the rows form charges are those the accepted facts name: tied and sampled in
part. -/
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
    simp [hform, tiedOf, Except.toOption, Option.filter]

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
  refine ⟨_, f, hx, hfacts, ?_, ?_, ?_⟩
  · show ((auditChecks p c e).filterMap (·.toOption))[i]? = some f
    rw [getElem?_filterMap_of_isSome _ _ hall, hget]
    rfl
  · obtain ⟨_, _, -, hfo⟩ := hfacts
    exact factsOf_sampled_le_units hfo
  · intro hd
    have hch : chargedOf p c (auditChecks p c e) i = some f := by
      simp [chargedOf, hget, hform, tiedOf, htied, hd]
    refine ⟨?_, ?_⟩
    · show f ∈ (Array.range (auditChecks p c e).size).filterMap (chargedOf p c (auditChecks p c e))
      exact Array.mem_filterMap.mpr ⟨i, Array.mem_range.mpr (Array.getElem?_eq_some_iff.mp hget).1, hch⟩
    · intro hwrong w wreg hw hc
      obtain ⟨hwlt, -⟩ := Array.getElem?_eq_some_iff.mp hw
      have hxw : e.windows[w]? = some (e.windows[w]'(by omega)) := Array.getElem?_eq_getElem (by omega)
      have hz : (c.windows.zip e.windows)[w]? = some (wreg, e.windows[w]'(by omega)) :=
        Array.getElem?_zip_eq_some.mpr ⟨hw, hxw⟩
      have hwget : (rowsWindows p c e s sd)[w]? =
          some (some <$> rowsWindowCheck p s sd c.audits (p.first + w) wreg (e.windows[w]'(by omega))) := by
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
def chargedCommitment : Commitment := ⟨#[slotReg], #[tiedWindow], .empty⟩

/-- Its honest side, at the slot audit tied to `tiedWindow`: sampled 1 of its 2 units with a bound of one incorrect
proof block, the audit is charged and its link-window's `β_w` is the scan's every entry bit; sampled whole, it is not
charged and `β_w` is 0. -/
theorem rowsTiedCharged_honest :
    (chargedOf chargedParams chargedCommitment #[.ok ⟨2, 1, ⟨0, 1⟩, 1⟩] 0).isSome = true ∧
      faultyEntries chargedParams chargedCommitment #[.ok ⟨2, 1, ⟨0, 1⟩, 1⟩] ⟨0, 0, 0, 0, 7⟩ tiedWindow = 7 ∧
      (chargedOf chargedParams chargedCommitment #[.ok ⟨2, 2, ⟨0, 1⟩, 0⟩] 0).isNone = true ∧
      faultyEntries chargedParams chargedCommitment #[.ok ⟨2, 2, ⟨0, 1⟩, 0⟩] ⟨0, 0, 0, 0, 7⟩ tiedWindow = 0 := by
  decide +kernel

/-- **No period check that accepts every period has `RowsTiedCharged`'s property** (its nontrivial lemma): such a check
accepts a rows-form period whose audit 0 is tied to its link-window's scan, and publishes no facts of it and no
charge. -/
theorem rowsTiedCharged_nontrivial : ¬ RowsTiedChargedOf acceptAllRows := by
  intro h
  let p : Params :=
    ⟨#[slotAudit], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 1⟩
  obtain ⟨-, h1⟩ := h p ⟨#[slotReg], #[tiedWindow], .empty⟩ ⟨#[otherAnswer], #[]⟩ _ _ rfl slotSchedule .empty
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

end Properties.NetworkPeriod
