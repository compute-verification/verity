import Properties.NetworkPeriod
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Each form's `K` bounds `PeriodLeakAccepted`'s count: the proofs of `ClockKBoundsCount` and `RowsKBoundsCount`

Each factor is at most 2 to the power of the ceiling of its `log2` (`le_two_pow_clog2`), and `K`'s sums over the
connections and the windows (the compute audits, or the scans) are sums over `Fin m` and `Fin n` (`Array.ofFn`), so
`2 ^ K` is the product of the factors' bounds.

`RowsTiedDrawn`, `RowsProgWired` and `RowsStateTied` read the check's refusals off its findings: on acceptance every
finding is none (`firstFinding_none`), a tied audit's among them, and each link-window's `wiredCheck` and
`stateCheck`.
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
    (w : Fin n → Facts) (tied : Fin n → ℕ) :
    (rowsTerms facts scan (Array.ofFn fun i => (w i, tied i)) (Array.ofFn caps) jitter windows).K =
      (∑ j, clog2 (caps j + 1)) + (∑ j, clog2 (caps j + 1)) + ∑ i, (scan.terms (w i) (tied i) jitter).bits := by
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
  simp only [AuditTerms.bits, AuditTerms.of, pow_add]
  exact Nat.mul_le_mul (Nat.mul_le_mul (Nat.mul_le_mul le_rfl (le_two_pow_clog2 _)) le_rfl) (le_two_pow_clog2 _)

theorem rowsKBoundsCount : RowsKBoundsCount := by
  intro facts scan jitter windows m n caps w tied
  rw [K_ofFn, pow_add, pow_add, mul_assoc]
  refine Nat.mul_le_mul (caps_le caps) (Nat.mul_le_mul (caps_le caps) ?_)
  rw [← Finset.prod_pow_eq_pow_sum]
  refine Finset.prod_le_prod fun i _ => ?_
  simp only [AuditTerms.bits, Scan.terms, pow_add]
  exact Nat.mul_le_mul (Nat.mul_le_mul (Nat.mul_le_mul le_rfl (le_two_pow_clog2 _)) le_rfl) (le_two_pow_clog2 _)

/-! ## A tied audit draws every unit -/

theorem firstFinding_none {fs : List (Option (Core.Verdict × String))} (h : firstFinding fs = none) :
    ∀ f ∈ fs, f = none := by
  unfold firstFinding at h
  split at h
  · cases h
  · intro f hf
    simpa using List.findSome?_eq_none_iff.mp h f hf

theorem auditChecks_getElem? {p : Params} {c : Commitment} {e : Evidence} {i : ℕ} {a : Audit} {reg : Registration}
    {x : Core.Answer} (ha : p.audits[i]? = some a) (hreg : c.audits[i]? = some reg) (hx : e.audits[i]? = some x) :
    (auditChecks p c e)[i]? = some (auditCheck i a reg x (tiedOf p.form c i)) := by
  have hz : ((p.audits.zip c.audits).zip e.audits)[i]? = some ((a, reg), x) :=
    Array.getElem?_zip_eq_some.mpr ⟨Array.getElem?_zip_eq_some.mpr ⟨ha, hreg⟩, hx⟩
  simp [auditChecks, Array.getElem?_zipIdx, hz]

/-- On acceptance in the rows form, every audit's finding is its facts. -/
theorem auditChecks_ok {p : Params} {c : Commitment} {e : Evidence} {pub : Public} {prof : Profile}
    {s : Schedule} {sd : ByteArray} {scan : Scan} (hform : p.form = .rows s sd scan)
    (h : check p c e = .accept pub prof) : ∀ y ∈ auditChecks p c e, ∃ f, y = .ok f := by
  intro y hy
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
    have h0 := firstFinding_none hF (findingOf y) (List.mem_append_left _ (List.mem_append_left _
      (List.mem_append_right _ (List.mem_map_of_mem (Array.mem_toList_iff.mpr hy)))))
    cases y with
    | error r => cases h0
    | ok f => exact ⟨f, rfl⟩

/-- The refusal: an audit the check ties that it accepts drew every unit. -/
theorem auditCheck_tied {i : ℕ} {a : Audit} {reg : Registration} {x : Core.Answer} {f : Facts}
    (h : auditCheck i a reg x true = .ok f) : DrawnWhole i reg x := by
  unfold auditCheck at h
  cases ha : answered s!"audit {i}" reg x with
  | error r =>
    simp only [ha, bind, Except.bind] at h
    by_cases c1 : (reg.record.getObjValD "program" == Lean.Json.str a.program) = true <;>
      simp [c1, refuse] at h
  | ok pp =>
    obtain ⟨prof, pub⟩ := pp
    cases hf : factsOf s!"audit {i}" prof pub with
    | error r =>
      simp only [ha, hf, bind, Except.bind] at h
      by_cases c1 : (reg.record.getObjValD "program" == Lean.Json.str a.program) = true <;>
        simp [c1, refuse] at h
    | ok g =>
      refine ⟨prof, pub, g, ha, hf, ?_⟩
      simp only [ha, hf, bind, Except.bind] at h
      by_cases c1 : (reg.record.getObjValD "program" == Lean.Json.str a.program) = true <;>
        by_cases c2 : g.units = a.units <;> by_cases c3 : g.drawn = g.units <;>
          simp_all [refuse, pure, Except.pure]

/-- The refusal refuses no audit drawn whole: one passes the tied check as it passes the untied one. -/
theorem auditCheck_tied_whole {i : ℕ} {a : Audit} {reg : Registration} {x : Core.Answer} {f : Facts}
    (h : auditCheck i a reg x = .ok f) (hd : f.drawn = f.units) : auditCheck i a reg x true = .ok f := by
  unfold auditCheck at h ⊢
  cases ha : answered s!"audit {i}" reg x with
  | error r =>
    simp only [ha, bind, Except.bind] at h
    by_cases c1 : (reg.record.getObjValD "program" == Lean.Json.str a.program) = true <;>
      simp [c1, refuse] at h
  | ok pp =>
    obtain ⟨prof, pub⟩ := pp
    cases hf : factsOf s!"audit {i}" prof pub with
    | error r =>
      simp only [ha, hf, bind, Except.bind] at h
      by_cases c1 : (reg.record.getObjValD "program" == Lean.Json.str a.program) = true <;>
        simp [c1, refuse] at h
    | ok g =>
      simp only [ha, hf, bind, Except.bind] at h ⊢
      by_cases c1 : (reg.record.getObjValD "program" == Lean.Json.str a.program) = true <;>
        by_cases c2 : g.units = a.units <;> simp_all [refuse, pure, Except.pure]

theorem check_sizes {p : Params} {c : Commitment} {e : Evidence} {pub : Public} {prof : Profile}
    (h : check p c e = .accept pub prof) : c.audits.size = p.audits.size := by
  unfold check at h
  split at h
  · cases h
  split at h
  · cases h
  rename_i hs
  by_contra hc
  exact hs hc

theorem rowsTiedDrawn : RowsTiedDrawn := by
  intro p c e pub prof h s sd scan hform i reg x hreg hx htied
  obtain ⟨hlt, -⟩ := Array.getElem?_eq_some_iff.mp hreg
  have hi : i < p.audits.size := by rw [← check_sizes h]; exact hlt
  have hget := auditChecks_getElem? (Array.getElem?_eq_getElem hi) hreg hx
  obtain ⟨f, hf⟩ := auditChecks_ok hform h _ (Array.mem_of_getElem? hget)
  have ht : tiedOf p.form c i = true := by rw [hform]; exact htied
  rw [ht] at hf
  exact auditCheck_tied hf

/-! ## Its nontrivial lemma -/

/-- An audit's registration with an output port's entry, `u/o1` at the root `r` and the domain `d`. -/
def tiedAudit : Registration :=
  { record := Lean.Json.mkObj [("roots", Lean.Json.arr #[Lean.Json.mkObj
      [("port", "u/o1"), ("root", "r"), ("domain", "d")]])], digest := .empty }

theorem tiedWindow_ties : tiedIn #[tiedAudit] #[tiedWindow] 0 = true := by
  decide +kernel

/-- An answer of another protocol than one-stage's. -/
def otherAnswer : Core.Answer :=
  { protocol := "other", verdict := .accept, reason := none, digest := none, published := none, profile := none }

theorem otherAnswer_refused : (answered s!"audit {0}" tiedAudit otherAnswer).toBool = false := by
  decide +kernel

/-- **No period check that accepts every period has `RowsTiedDrawn`'s property** (its nontrivial lemma): such a check
accepts a rows-form period whose audit 0 is tied and answered by another protocol than one-stage's. -/
theorem rowsTiedDrawn_nontrivial : ¬ RowsTiedDrawnOf acceptAllRows := by
  intro h
  let s : Schedule := ⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩
  let p : Params := ⟨#[], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", .rows s .empty ⟨0, 0, 0, 0⟩, 1⟩
  obtain ⟨prof, pub, f, ha, -, -⟩ := h p ⟨#[tiedAudit], #[tiedWindow], .empty⟩ ⟨#[otherAnswer], #[]⟩ _ _ rfl s .empty
    ⟨0, 0, 0, 0⟩ rfl 0 tiedAudit otherAnswer rfl rfl tiedWindow_ties
  have := otherAnswer_refused
  rw [ha] at this
  simp [Except.toBool] at this

/-! ## The rows form's scan reads a compute audit's slot entries -/

theorem wiredChecks_getElem? {p : Params} {c : Commitment} {s : Schedule} {sd : ByteArray} {scan : Scan}
    (hform : p.form = .rows s sd scan) {i : ℕ} {reg : Registration} (hreg : c.windows[i]? = some reg) :
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

/-- `slotAudit`'s registration, with its template's entries port at the root `r` and the domain `d`. -/
def slotReg : Registration :=
  { record := Lean.Json.mkObj [("roots", Lean.Json.arr #[Lean.Json.mkObj
      [("port", "SlotEntry_v1{KO=1}/c0/o0"), ("root", "r"), ("domain", "d")]])], digest := .empty }

/-- Its honest side: a link-window whose scan's `prog` is at that port's root and domain is wired, and the slot audit
is tied to it, so its free inputs are in the link-window's term. -/
theorem rowsProgWired_honest :
    wired #[slotAudit] slotSchedule #[slotReg] tiedWindow = true ∧ tiedIn #[slotReg] #[tiedWindow] 0 = true := by
  decide +kernel

theorem tiedWindow_unwired : wired #[] slotSchedule #[] tiedWindow = false := by
  decide +kernel

/-- **No period check that accepts every period has `RowsProgWired`'s property** (its nontrivial lemma): such a check
accepts a rows-form period of one link-window with no audit, and there the link-window's scan's `prog` is no audit's
slot entries (`wired` fails at the period, its acceptance, its form and its link-window). -/
theorem rowsProgWired_nontrivial :
    ¬ RowsProgWiredOf acceptAllRows ∧
      (∃ pub prof, acceptAllRows (rowsParams #[]) (oneWindow #[] tiedWindow) ⟨#[], #[]⟩ = .accept pub prof) ∧
      (rowsParams #[]).form = .rows slotSchedule .empty ⟨0, 0, 0, 0⟩ ∧
      (oneWindow #[] tiedWindow).windows[0]? = some tiedWindow ∧
      ¬ wired (rowsParams #[]).audits slotSchedule (oneWindow #[] tiedWindow).audits tiedWindow = true := by
  have hw : ¬ wired (rowsParams #[]).audits slotSchedule (oneWindow #[] tiedWindow).audits tiedWindow = true := by
    intro hw
    change wired #[] slotSchedule #[] tiedWindow = true at hw
    rw [tiedWindow_unwired] at hw
    cases hw
  exact ⟨fun h => hw (h (rowsParams #[]) (oneWindow #[] tiedWindow) ⟨#[], #[]⟩ _ _ rfl slotSchedule .empty ⟨0, 0, 0, 0⟩
    rfl 0 tiedWindow rfl), ⟨_, _, rfl⟩, rfl, rfl, hw⟩

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
    .rows slotSchedule .empty ⟨0, 0, 0, 0⟩, 1⟩
  have hw : wired #[renamedAudit] slotSchedule #[slotReg] tiedWindow = true :=
    h p ⟨#[slotReg], #[tiedWindow], .empty⟩ ⟨#[], #[]⟩ _ _ rfl slotSchedule .empty ⟨0, 0, 0, 0⟩ rfl 0 tiedWindow rfl
  rw [renamedAudit_unwired] at hw
  cases hw

/-! ## The rows form's scan's state is its wired audit's `SCAN_STATE` -/

theorem stateChecks_getElem? {p : Params} {c : Commitment} {s : Schedule} {sd : ByteArray} {scan : Scan}
    (hform : p.form = .rows s sd scan) {i : ℕ} {reg : Registration} (hreg : c.windows[i]? = some reg) :
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
accepts a rows-form period of one link-window whose scan's `prog` is wired to the slot audit, and there the scan's
state is not that audit's `SCAN_STATE`, which skips or alters a transition of the scan's (`stateTied` fails at the
period, its acceptance, its form and its link-window). -/
theorem rowsStateTied_nontrivial :
    ¬ RowsStateTiedOf acceptAllRows ∧
      (∃ pub prof, acceptAllRows (rowsParams #[slotAudit]) (oneWindow #[stateReg "t"] stateWindow) ⟨#[], #[]⟩ =
        .accept pub prof) ∧
      (rowsParams #[slotAudit]).form = .rows slotSchedule .empty ⟨0, 0, 0, 0⟩ ∧
      (oneWindow #[stateReg "t"] stateWindow).windows[0]? = some stateWindow ∧
      ¬ stateTied (rowsParams #[slotAudit]).audits slotSchedule (oneWindow #[stateReg "t"] stateWindow).audits
        stateWindow = true := by
  have hs : ¬ stateTied (rowsParams #[slotAudit]).audits slotSchedule (oneWindow #[stateReg "t"] stateWindow).audits
      stateWindow = true := by
    intro hs
    change stateTied #[slotAudit] slotSchedule #[stateReg "t"] stateWindow = true at hs
    rw [stateWindow_skipped] at hs
    cases hs
  exact ⟨fun h => hs (h (rowsParams #[slotAudit]) (oneWindow #[stateReg "t"] stateWindow) ⟨#[], #[]⟩ _ _ rfl
    slotSchedule .empty ⟨0, 0, 0, 0⟩ rfl 0 stateWindow rfl), ⟨_, _, rfl⟩, rfl, rfl, hs⟩

end Properties.NetworkPeriod
