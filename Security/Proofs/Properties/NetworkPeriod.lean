import Properties.NetworkPeriod
import Proofs.NetworkCertifier.RecordRows
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Each form's `K` bounds `PeriodLeakAccepted`'s count: the proofs of `ClockKBoundsCount` and `RowsKBoundsCount`

Each factor is at most 2 to the power of the ceiling of its `log2` (`le_two_pow_clog2`), and `K`'s sums over the
connections and the windows (the compute audits, or the scans) are sums over `Fin m` and `Fin n` (`Array.ofFn`), so
`2 ^ K` is the product of the factors' bounds.

`RowsProgWired`, `RowsStateTied` and `RowsCircuitHeld` read the check's refusals off its findings: on acceptance every
finding is none (`firstFinding_none`), each phase's `wiredCheck`, `stateCheck` and `heldCheck` among them.
`rowsCircuitHeld_honest` and `rowsCircuitHeld_nontrivial` are `RowsCircuitHeld`'s witnesses: an empty rows-form period
the check accepts, where the hold is vacuous, beside a tied audit that runs the class circuit, and a period whose tied
audit runs another. That the check accepts a period with a held tied audit is the difftest's (`rows-honest`, through
Lean).
`RowsStateHeld` reads the accepted public off the check: on acceptance every phase has no finding, so each is its
scan's facts and its tied audits' charges (`rowsPhase_ok`), the scans keep the phases' indices (`filterMap_getElem?`),
and each link-window's term is its scan's free inputs and, by its scan's draw, the scan's slot entries
(`terms_whole`) or those charges (`tiedFree`, `terms_part`).

On acceptance a rows-form period has no `rowsWfCheck` finding (`rowsWf_of_accept`), so its frames, the network
certifier's of `B` bytes for its `lanes · windows · window` connections, read back (`cellRows_wf_of_rowsWf`,
`NetTiming.RowsFormat.Wf` at `NetTiming.cellRows`).
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

/-- A scan's term's bits, by its draw: the scan's free inputs, with its slot entries where every unit is proved and
else its tied audits' free inputs and the entry bits their wrong units can set, and the ceilings of its other
factors. -/
theorem terms_bits (scan : Scan) (f : Facts) (tied faulty jitter : ℕ) :
    (scan.terms f tied faulty jitter).bits =
      (scan.free + if f.whole then scan.entries else tied) + (if f.whole then 0 else faulty) +
        clog2 (atMost f.units f.wrong) + f.wrong * max scan.ingress scan.egress + clog2 (jitter ^ scan.steps) := by
  simp only [AuditTerms.bits, Scan.terms]
  split <;> omega

/-- A window's factors, its faulty sets `A` and jitter `J` each at most 2 to a ceiling, are at most 2 to the sum of
their exponents. -/
theorem factors_le {a e s b j A J : ℕ} (hA : A ≤ 2 ^ s) (hJ : J ≤ 2 ^ j) :
    2 ^ a * A * 2 ^ b * J * 2 ^ e ≤ 2 ^ (a + e + s + b + j) :=
  calc 2 ^ a * A * 2 ^ b * J * 2 ^ e ≤ 2 ^ a * 2 ^ s * 2 ^ b * 2 ^ j * 2 ^ e :=
        Nat.mul_le_mul (Nat.mul_le_mul (Nat.mul_le_mul (Nat.mul_le_mul le_rfl hA) le_rfl) hJ) le_rfl
    _ = 2 ^ (a + s + b + j + e) := by simp only [pow_add]
    _ ≤ 2 ^ (a + e + s + b + j) := Nat.pow_le_pow_right (by decide) (by omega)

theorem rowsKBoundsCount : RowsKBoundsCount := by
  intro facts scan jitter windows m n caps w tied faulty charged
  rw [K_ofFn, pow_add, pow_add, mul_assoc]
  refine Nat.mul_le_mul (caps_le caps) (Nat.mul_le_mul (caps_le caps) ?_)
  rw [← Finset.prod_pow_eq_pow_sum]
  refine Finset.prod_le_prod fun i _ => ?_
  simp only [terms_bits]
  exact factors_le (le_two_pow_clog2 _) (le_two_pow_clog2 _)

/-- A scan's term where every unit of the scan is proved: its free inputs are the scan's and it charges the scan's
slot entries. -/
theorem terms_whole {scan : Scan} {f : Facts} (tied faulty jitter : ℕ) (h : f.whole = true) :
    (scan.terms f tied faulty jitter).free = scan.free ∧ (scan.terms f tied faulty jitter).entries = scan.entries := by
  simp [Scan.terms, h]

/-- A scan's term where the scan is sampled in part: its free inputs are the scan's and its tied audits'. -/
theorem terms_part {scan : Scan} {f : Facts} (tied faulty jitter : ℕ) (h : f.whole = false) :
    (scan.terms f tied faulty jitter).free = scan.free + tied := by
  simp [Scan.terms, h]

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

/-- The slot audit's registration (`slotAudit`), with that template's entries port at the root `r` and the domain
`d`. -/
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
def selectAudit : Audit := ⟨0, 0, "", 1, 0, #[("SelectEntry_v1{KO=1}", 1, selectEntryPins[0]!)], none, none, 0⟩

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
    ⟨#[], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", none, .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 1⟩
  have hw : wired #[] slotSchedule #[] tiedWindow = true :=
    h p ⟨#[], #[tiedWindow], .empty⟩ ⟨#[], #[]⟩ _ _ rfl slotSchedule .empty ⟨0, 0, 0, 0, 0⟩ rfl 0 tiedWindow rfl
  rw [tiedWindow_unwired] at hw
  cases hw

/-- `slotAudit` with its template's body another than every pinned one: a template under `SlotEntry_v1`'s name and
descriptor id, one member a slot, whose closure digest is none of `slotEntryPins`. -/
def renamedAudit : Audit := ⟨0, 0, "", 1, 0, #[("SlotEntry_v1{KO=1}", 1, "")], none, none, 0⟩

/-- A renamed body is no slot entries: the link-window `slotAudit` wires is unwired when its audit's template has
another body, with its port, root, domain and members unchanged. -/
theorem renamedAudit_unwired : wired #[renamedAudit] slotSchedule #[slotReg] tiedWindow = false := by
  decide +kernel

/-- **No period check that accepts every period has `RowsProgWired`'s property at a renamed body** (its nontrivial
lemma at the pin): such a check accepts a rows-form period whose scan's `prog` is at the root and domain of an audit's
`SlotEntry_v1` template's entries port, one member a slot, whose body is none of the pinned Definitions'. -/
theorem rowsProgWired_renamed : ¬ RowsProgWiredOf acceptAllRows := by
  intro h
  let p : Params := ⟨#[renamedAudit], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", none,
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

/-- A link-window's state finding is none only when its scan's state is tied (`stateTied`) and its row 0 opened to the
empty state (`stateEmpty`). -/
theorem stateCheck_none {p : Params} {s : Schedule} {audits : Array Lean.Json} {w : ℕ} {reg : Lean.Json}
    (h : stateCheck p s audits w reg = none) : stateTied p.audits s audits reg = true ∧ stateEmpty s reg = true := by
  unfold stateCheck at h
  split at h
  · split at h
    · exact ⟨by assumption, by assumption⟩
    · cases h
  · cases h

/-- An accepted rows-form period's link-window has no state finding. -/
theorem stateCheck_of_accept {p : Params} {c : Commitment} {e : Evidence} {pub : Public} {prof : Profile}
    (h : check p c e = .accept pub prof) {s : Schedule} {sd : ByteArray} {scan : Scan}
    (hform : p.form = .rows s sd scan) {i : ℕ} {reg : Lean.Json} (hreg : c.windows[i]? = some reg) :
    stateCheck p s c.audits (p.first + i) reg = none := by
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
    exact firstFinding_none hF _
      (List.mem_append_left _ (List.mem_append_right _ (Array.mem_toList_iff.mpr hmem)))

theorem rowsStateTied : RowsStateTied := by
  intro p c e pub prof h s sd scan hform i reg hreg
  exact stateCheck_none (stateCheck_of_accept h hform hreg)

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

/-- The slot audit's schedule at T = 2, `every` 2: one slot a window, its two ingress units chained through the state's
leaf 1. -/
def stateSchedule : Schedule := ⟨0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 2⟩

/-- Its honest side: unit 1 reads `sin` at leaf 1, where unit 0 registers its `o1`, and the state is the slot audit's
`SCAN_STATE`. -/
theorem stateWindow_chained :
    stateTied #[slotAudit] stateSchedule #[stateReg "s"] (stateWindowOf (stateRun 1 1 2) (stateRun 1 0 1)) = true := by
  decide +kernel

/-- A registration whose units read their state off the chain: unit 1's `sin` at leaf 2, a leaf no unit registered,
and unit 0's `o1` at leaf 0, the empty state's, at the state's root and domain all the same. -/
def misindexedWindow : Lean.Json := stateWindowOf (stateRun 2 1 2) (stateRun 0 0 1)

theorem misindexedWindow_untied : stateTied #[slotAudit] stateSchedule #[stateReg "s"] misindexedWindow = false := by
  decide +kernel

/-- **The rows form refuses a scan whose units read their state off the chain**: a period whose one link-window's scan
is wired to the slot audit, its state that audit's `SCAN_STATE` by root and domain, but its registration's reads
misindexed (`misindexedWindow`), is accepted by no evidence (`rowsStateTied`). -/
theorem rowsStateTied_misindexed {p : Params} {sd : ByteArray} {scan : Scan} (e : Evidence) (pub : Public)
    (prof : Profile) (hform : p.form = .rows stateSchedule sd scan) (ha : p.audits = #[slotAudit]) :
    check p ⟨#[stateReg "s"], #[misindexedWindow], .empty⟩ e ≠ .accept pub prof := by
  intro hacc
  have hs : stateTied p.audits stateSchedule #[stateReg "s"] misindexedWindow = true :=
    (rowsStateTied p _ e pub prof hacc stateSchedule sd scan hform 0 misindexedWindow rfl).1
  rw [ha, misindexedWindow_untied] at hs
  cases hs

/-- **No period check that accepts every period has `RowsStateTied`'s property** (its nontrivial lemma): such a check
accepts a rows-form period of one link-window whose scan's `prog` is wired to the slot audit, and whose state is not
that audit's `SCAN_STATE`, which skips or alters a transition of the scan's. -/
theorem rowsStateTied_nontrivial : ¬ RowsStateTiedOf acceptAllRows := by
  intro h
  let p : Params :=
    ⟨#[slotAudit], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", none, .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 1⟩
  have hs : stateTied #[slotAudit] slotSchedule #[stateReg "t"] stateWindow = true :=
    (h p ⟨#[stateReg "t"], #[stateWindow], .empty⟩ ⟨#[], #[]⟩ _ _ rfl slotSchedule .empty ⟨0, 0, 0, 0, 0⟩ rfl 0
      stateWindow rfl).1
  rw [stateWindow_skipped] at hs
  cases hs

/-! ## The rows form's tied audits run the class circuit -/

theorem heldChecks_getElem? {p : Params} {c : Commitment} {s : Schedule} {sd : ByteArray} {scan : Scan}
    (hform : p.form = .rows s sd scan) {i : ℕ} {reg : Lean.Json} (hreg : c.windows[i]? = some reg) :
    (heldChecks p c)[i]? = some (heldCheck p c.audits (p.first + i) reg) := by
  simp [heldChecks, hform, Array.getElem?_zipIdx, hreg]

theorem rowsCircuitHeld : RowsCircuitHeld := by
  intro p c e pub prof h s sd scan hform i reg hreg
  have hmem := Array.mem_of_getElem? (heldChecks_getElem? hform hreg)
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
    unfold heldCheck at h0
    split at h0
    · assumption
    · cases h0

/-- A phase whose tied audit is registered under the class circuit `c` holds it. -/
theorem tiedWindow_held : held (some "c") #[heldReg "c"] tiedWindow = true := by
  decide +kernel

/-- A phase whose tied audit is registered under another Program than the class circuit `c` does not. -/
theorem tiedWindow_swapped : held (some "c") #[heldReg "d"] tiedWindow = false := by
  decide +kernel

/-- **`RowsCircuitHeld`'s honest lemma**: the check accepts a rows-form period of no phase under the class circuit
`c`, where the hold is vacuous, and a phase whose tied audit is registered under `c` holds it. That the check accepts
a period with a held tied audit is the difftest's (`rows-honest`, through Lean). -/
theorem rowsCircuitHeld_honest :
    check ⟨#[], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", some "c", .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 0⟩
        ⟨#[], #[], .empty⟩ ⟨#[], #[]⟩ =
      .accept (rowsTerms #[] ⟨0, 0, 0, 0, 0⟩ #[] #[] 1 0 #[]) { link := "", windows := 0 } ∧
    held (some "c") #[heldReg "c"] tiedWindow = true :=
  ⟨by
    simp [check, auditChecks, wiredChecks, stateChecks, heldChecks, firstFinding, rowsPublic, rowsWfCheck, rowsWf]
    rfl,
    tiedWindow_held⟩

/-- **`RowsCircuitHeld`'s nontrivial lemma**: `RowsCircuitHeldAt` fails at a rows-form period of one phase whose tied
audit is registered under another Program than the class circuit `c`, which a check that accepts everything
accepts. -/
theorem rowsCircuitHeld_nontrivial :
    ¬ RowsCircuitHeldAt
      ⟨#[slotAudit], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", some "c",
        .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 1⟩
      ⟨#[heldReg "d"], #[tiedWindow], .empty⟩ := by
  intro h
  have hh : held (some "c") #[heldReg "d"] tiedWindow = true :=
    h slotSchedule .empty ⟨0, 0, 0, 0, 0⟩ rfl 0 tiedWindow rfl
  rw [tiedWindow_swapped] at hh
  cases hh

/-! ## The rows form's tied audits read the scan's state at their slots' anchors -/

/-- A digit of radix 1 moves no index: leaving the digits of radix 1 out keeps `accum`. -/
theorem accum_filter (ps : List (ℕ × ℤ)) (i : ℕ) :
    SampledProofs.Wires.accum (ps.filter (·.1 != 1)) i = SampledProofs.Wires.accum ps i := by
  induction ps generalizing i with
  | nil => rfl
  | cons p ps ih =>
    obtain ⟨r, st⟩ := p
    by_cases hr : r = 1
    · subst hr
      rw [List.filter_cons_of_neg (p := fun x : ℕ × ℤ => x.1 != 1) (a := (1, st)) (l := ps) (by simp), ih]
      simp [SampledProofs.Wires.accum]
    · rw [List.filter_cons_of_pos (p := fun x : ℕ × ℤ => x.1 != 1) (a := (r, st)) (l := ps) (by simpa using hr)]
      simp only [SampledProofs.Wires.accum, ih]

/-- Leaving the digits of radix 1 out keeps the product of the radices. -/
theorem prodR_filter (ps : List (ℕ × ℤ)) :
    SampledProofs.Wires.prodR (ps.filter (·.1 != 1)) = SampledProofs.Wires.prodR ps := by
  induction ps with
  | nil => rfl
  | cons p ps ih =>
    obtain ⟨r, st⟩ := p
    by_cases hr : r = 1
    · subst hr
      rw [List.filter_cons_of_neg (p := fun x : ℕ × ℤ => x.1 != 1) (a := (1, st)) (l := ps) (by simp), ih]
      simp [SampledProofs.Wires.prodR]
    · rw [List.filter_cons_of_pos (p := fun x : ℕ × ℤ => x.1 != 1) (a := (r, st)) (l := ps) (by simpa using hr)]
      simp only [SampledProofs.Wires.prodR, List.foldr_cons] at ih ⊢
      rw [ih]

/-- A registered read the check takes for its slots' anchors' (`anchorRead`) reads the scan's state at them
(`AnchoredRead`). -/
theorem anchorRead_rows {s : Schedule} {c : NetworkCheck.Circuit.Wiring} {params : List (String × ℕ)}
    {x : NetworkCheck.Circuit.Read} (h : anchorRead s c params x = true) : AnchoredRead s c params x := by
  refine ⟨?_, fun W k hp => ?_⟩
  · cases hp : NetworkCheck.Circuit.portOf c x.pat.reader x.pat.port false with
    | none => simp [anchorRead, hp] at h
    | some _ => rfl
  · simp only [anchorRead, hp, Bool.and_eq_true, beq_iff_eq, and_assoc] at h
    obtain ⟨hk, hs, hc0, hc, hbase, hd, hl⟩ := h
    have hw : ∀ i, x.pat.writerAt i = ((i % (s.window / s.every) * (s.every * W) : ℕ) : ℤ) := by
      intro i
      rw [SampledProofs.Wires.Wire.writerAt, hbase, SampledProofs.Wires.Wire.pairs, ← accum_filter,
        List.filter_reverse, hd, anchorDigits, ← List.filter_reverse, accum_filter]
      simp [SampledProofs.Wires.accum]
    refine ⟨hk, hl, ?_, fun l _ b hb => ⟨?_, ?_⟩⟩
    · rw [SampledProofs.Wires.Wire.units, SampledProofs.Wires.Wire.pairs, ← prodR_filter, List.filter_reverse, hd,
        anchorDigits, ← List.filter_reverse, prodR_filter]
      simp [SampledProofs.Wires.prodR, Nat.mul_comm]
    · simp [SampledProofs.Wires.Wire.readerAt, hs, hc, hc0, Nat.mod_one]
    · rw [hw, Nat.mul_add_mod_of_lt hb, Nat.mul_assoc]

/-- An audit the check takes for anchored (`anchored`) with circuit record `r` reads the scan's state at its slots'
anchors (`AnchoredRecord`). -/
theorem anchored_record {s : Schedule} {a : Audit} {r : NetworkCheck.Circuit.Stated} (h : anchored s a = true)
    (hr : a.circuit = some r) : AnchoredRecord s r := by
  simp only [anchored, hr, Bool.and_eq_true] at h
  obtain ⟨hrd, htab⟩ := h
  refine ⟨by simpa using htab, fun x hx hp => anchorRead_rows ?_⟩
  exact List.all_eq_true.mp hrd x (List.mem_filter.mpr ⟨hx, beq_iff_eq.mpr hp⟩)

/-- **A link-window's state finding is none only when its tied audits read the scan's state at their slots'
anchors**: each audit tied to its scan (`tiedAudits`) with a circuit record reads, at unit `l · nb + b` (lane `l`'s
slot at boundary `b`, `wired`'s chunk `l` leaf `b`), row `b · every` of the scan's state, and declares no table over
it (`AnchoredRecord`). -/
theorem stateCheck_anchored {p : Params} {s : Schedule} {audits : Array Lean.Json} {w : ℕ} {reg : Lean.Json}
    (h : stateCheck p s audits w reg = none) :
    ∀ j ∈ tiedAudits audits reg, ∀ (a : Audit) (r : NetworkCheck.Circuit.Stated), p.audits[j]? = some a →
      a.circuit = some r → AnchoredRecord s r := by
  intro j hj a r ha hr
  have hat : anchoredAt p.audits s audits reg = true := by
    unfold stateCheck at h
    split at h
    · split at h
      · split at h
        · assumption
        · simp at h
      · cases h
    · cases h
  rw [anchoredAt, Array.all_eq_true'] at hat
  have hj' := hat j hj
  simp only [ha, Option.map_some, Option.getD_some] at hj'
  exact anchored_record hj' hr

theorem rowsStateAnchored : RowsStateAnchored := by
  intro p c e pub prof h s sd scan hform i reg hreg
  exact stateCheck_anchored (stateCheck_of_accept h hform hreg)

/-- **`RowsStateAnchored`'s honest lemma**: the check accepts a rows-form period of no phase, and a phase whose tied
audit's circuit record reads row `b` of the scan's state at unit `l · 2 + b` from base 0 (`anchorAudit 0`) is
anchored. -/
theorem rowsStateAnchored_honest :
    check ⟨#[], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", some "c", .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 0⟩
        ⟨#[], #[], .empty⟩ ⟨#[], #[]⟩ =
      .accept (rowsTerms #[] ⟨0, 0, 0, 0, 0⟩ #[] #[] 1 0 #[]) { link := "", windows := 0 } ∧
    anchoredAt #[anchorAudit 0] anchorSchedule #[heldReg "c"] tiedWindow = true :=
  ⟨rowsCircuitHeld_honest.1, by decide +kernel⟩

/-- **`RowsStateAnchored`'s nontrivial lemma**: `RowsStateAnchoredAt` fails at a rows-form period of one phase whose
tied audit's circuit record reads the scan's state from base 3 (`anchorAudit 3`), so lane 0's slot at boundary 0 reads
row 1, past its anchor's row 0, which a check that accepts everything accepts. -/
theorem rowsStateAnchored_nontrivial :
    ¬ RowsStateAnchoredAt
      ⟨#[anchorAudit 3], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", some "c",
        .rows anchorSchedule .empty ⟨0, 0, 0, 0, 0⟩, 1⟩
      ⟨#[heldReg "c"], #[tiedWindow], .empty⟩ := by
  intro h
  have hj : 0 ∈ tiedAudits #[heldReg "c"] tiedWindow := by
    rw [show tiedAudits #[heldReg "c"] tiedWindow = #[0] by decide +kernel]
    exact Array.mem_singleton.mpr rfl
  have hr := ((h anchorSchedule .empty ⟨0, 0, 0, 0, 0⟩ rfl 0 tiedWindow rfl 0 hj (anchorAudit 3)
    (anchorRecord (stateReadAt 3)) rfl rfl).2 (stateReadAt 3) (List.mem_singleton.mpr rfl) rfl).2 3 1
    (by decide +kernel)
  exact absurd (hr.2.2.2 0 (by decide +kernel) 0 (by decide +kernel)).2 (by decide +kernel)

/-! ## The rows form charges a tied audit its reads of the scan's state only where they are held -/

/-- A `filterMap` whose function keeps every element keeps each index. -/
theorem filterMap_getElem? {α β : Type} {g : α → Option β} {l : List α} (h : ∀ x ∈ l, (g x).isSome) (i : ℕ) :
    (l.filterMap g)[i]? = l[i]?.bind g := by
  induction l generalizing i with
  | nil => simp
  | cons x xs ih =>
    obtain ⟨y, hy⟩ := Option.isSome_iff_exists.mp (h x (by simp))
    cases i with
    | zero => simp [hy]
    | succ i => simpa [List.filterMap_cons, hy] using ih (fun z hz => h z (List.mem_cons_of_mem _ hz)) i

/-- A rows-form phase without a finding has its scan's facts and its tied audits' charges (`tiedFree`). -/
theorem rowsPhase_ok {p : Params} {s : Schedule} {sd : ByteArray} {c : Commitment} {q : (Lean.Json × Window) × ℕ}
    (h : findingOf (rowsPhase p s sd c q) = none) :
    ∃ f, rowsPhase p s sd c q = .ok (some (f, tiedFree p c.audits c.windows q.1.1)) := by
  unfold rowsPhase at h ⊢
  cases hr : rowsWindowCheck p s sd c.audits (p.first + q.2) q.1.1 q.1.2 with
  | error x =>
    rw [hr] at h
    change findingOf (Except.error x : Found (Option (Facts × ℕ))) = none at h
    cases h
  | ok f => exact ⟨f, rfl⟩

theorem rowsStateHeld : RowsStateHeld := by
  intro p c e pub prof h s sd scan hform i reg hreg
  unfold check at h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  rename_i hew
  split at h
  · cases h
  simp only [hform] at h
  split at h
  · cases h
  · cases h
  · rename_i hF
    injection h with hpub
    subst hpub
    have hok : ∀ w ∈ ((c.windows.zip e.windows).zipIdx.map (rowsPhase p s sd c)).toList, findingOf w = none :=
      fun w hw => firstFinding_none hF _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _
        (List.mem_append_left _ (List.mem_append_left _ (List.mem_map_of_mem hw))))))
    have hsome : ∀ w ∈ ((c.windows.zip e.windows).zipIdx.map (rowsPhase p s sd c)).toList,
        (w.toOption.join).isSome := by
      intro w hw
      have h0 := hok w hw
      rw [Array.toList_map, List.mem_map] at hw
      obtain ⟨q, _, rfl⟩ := hw
      obtain ⟨g, hg⟩ := rowsPhase_ok h0
      rw [hg]
      rfl
    obtain ⟨hi, -⟩ := Array.getElem?_eq_some_iff.mp hreg
    have hie : i < e.windows.size := by omega
    have hpair : ((c.windows.zip e.windows).zipIdx)[i]? = some ((reg, e.windows[i]), i) := by
      rw [Array.getElem?_zipIdx,
        (Array.getElem?_zip_eq_some (z := (reg, e.windows[i]))).mpr ⟨hreg, Array.getElem?_eq_getElem hie⟩]
      simp
    have hwsi : ((c.windows.zip e.windows).zipIdx.map (rowsPhase p s sd c))[i]? =
        some (rowsPhase p s sd c ((reg, e.windows[i]), i)) := by
      rw [Array.getElem?_map, hpair]
      rfl
    obtain ⟨f, hf⟩ := rowsPhase_ok (hok _ (Array.mem_toList_iff.mpr (Array.mem_of_getElem? hwsi)))
    generalize (c.windows.zip e.windows).zipIdx.map (rowsPhase p s sd c) = W at hok hsome hwsi ⊢
    have hscan : (W.filterMap fun x => x.toOption.join)[i]? = some (f, tiedFree p c.audits c.windows reg) := by
      rw [← Array.getElem?_toList, Array.toList_filterMap, filterMap_getElem? hsome, Array.getElem?_toList, hwsi]
      show (rowsPhase p s sd c ((reg, e.windows[i]), i)).toOption.join = _
      rw [hf]
      rfl
    have hz : ((W.filterMap fun x => x.toOption.join).zip c.windows)[i]? =
        some ((f, tiedFree p c.audits c.windows reg), reg) :=
      (Array.getElem?_zip_eq_some (z := ((f, tiedFree p c.audits c.windows reg), reg))).mpr ⟨hscan, hreg⟩
    have hu : (rowsPublic p c (auditChecks p c e) scan (W.filterMap fun x => x.toOption.join)).units[i]? =
        some (scan.terms f (tiedFree p c.audits c.windows reg)
          (faultyEntries p (partOf c (W.filterMap fun x => x.toOption.join)) (auditChecks p c e) scan reg)
          p.jitter) := by
      simp only [rowsPublic, rowsTerms, Array.getElem?_map, hz, Option.map_some]
    exact ⟨_, hu, fun hw => terms_whole (scan := scan) (f := f) _ _ _ hw,
      fun hw => terms_part (scan := scan) (f := f) (tiedFree p c.audits c.windows reg) _ _ hw⟩

/-- **`RowsStateHeld`'s honest lemma**: the check accepts a rows-form period of no phase, and an audit whose
`SCAN_STATE` is its one phase's scan's state, at its root and domain (`stateReg "s"`, `stateWindow`), is held there and
its phase charges it its `stateFree`. -/
theorem rowsStateHeld_honest :
    check ⟨#[], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", some "c", .rows slotSchedule .empty ⟨0, 0, 0, 0, 0⟩, 0⟩
        ⟨#[], #[], .empty⟩ ⟨#[], #[]⟩ =
      .accept (rowsTerms #[] ⟨0, 0, 0, 0, 0⟩ #[] #[] 1 0 #[]) { link := "", windows := 0 } ∧
    stateHeldIn #[stateReg "s"] #[stateWindow] 0 = true ∧
    tiedFree heldParams #[stateReg "s"] #[stateWindow] stateWindow = heldAudit.stateFree :=
  ⟨rowsCircuitHeld_honest.1, by decide +kernel, by decide +kernel⟩

/-- `heldPublic`'s term is of a scan sampled in part. -/
theorem heldPublic_part : ∀ t ∈ heldPublic.units[0]?, t.facts.whole = false ∧ t.free = 4 := by
  intro t ht
  simp [heldPublic] at ht
  subst ht
  exact ⟨by decide +kernel, rfl⟩

/-- **`RowsStateHeld`'s nontrivial lemma**: `RowsStateHeldAt` fails at a rows-form period of one phase whose tied
audit's `SCAN_STATE` is at another root than its scan's state (`stateReg "t"`), under a public whose term, of a scan
sampled in part, charges that audit its `stateFree`, 4 of its 9 free inputs, which a check that accepts everything
could publish. -/
theorem rowsStateHeld_nontrivial :
    ¬ RowsStateHeldAt heldParams ⟨#[stateReg "t"], #[stateWindow], .empty⟩ heldPublic := by
  intro h
  obtain ⟨t, ht, -, hfree⟩ := h slotSchedule .empty ⟨0, 0, 0, 0, 0⟩ rfl 0 stateWindow rfl
  obtain ⟨hpart, h4⟩ := heldPublic_part t ht
  have hfree := hfree hpart
  rw [h4] at hfree
  exact absurd hfree (by decide +kernel)

/-! ## The rows form's frames read back -/

/-- **`rowsWf` gives the deployed rows format's `Wf`**: at a schedule it passes over `windows` link-windows, the
network certifier's frames of `B` bytes for the `lanes · windows · window` connections (`NetTiming.cellRows`) read
back. -/
theorem cellRows_wf_of_rowsWf {s : Schedule} {windows : ℕ} (h : rowsWf s windows = true) {L K : ℕ} (link : Fin L)
    (T : ℕ)
    (jitter : NetTiming.WardenSchedule (Fin (s.lanes * (windows * s.window))) →
      (Fin (s.lanes * (windows * s.window)) → NetTiming.Arrivals (NetTiming.FrameCell (s.cellBytes - 9))) → ℕ → ℕ)
    (status : NetTiming.WardenSchedule (Fin (s.lanes * (windows * s.window))) →
      (Fin (s.lanes * (windows * s.window)) → NetTiming.Arrivals (NetTiming.FrameCell (s.cellBytes - 9))) → ℕ →
        Fin K → NetTiming.Status) :
    (NetTiming.cellRows s.cellBytes (s.lanes * (windows * s.window)) link T jitter status).Wf := by
  simp only [rowsWf, Bool.and_eq_true, decide_eq_true_eq] at h
  exact NetTiming.cellRows_wf h.1 h.2 link T jitter status

/-- **The check's acceptance gives `rowsWf`**: a rows-form period it accepts has no `rowsWfCheck` finding. -/
theorem rowsWf_of_accept {p : Params} {c : Commitment} {e : Evidence} {pub : Public} {prof : Profile}
    (h : check p c e = .accept pub prof) {s : Schedule} {sd : ByteArray} {scan : Scan}
    (hform : p.form = .rows s sd scan) : rowsWf s p.windows = true := by
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
    have h0 := firstFinding_none hF (rowsWfCheck p) (List.mem_append_left _ (List.mem_append_left _
      (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _ (by simp))))))
    simp only [rowsWfCheck, hform] at h0
    split at h0
    · assumption
    · cases h0

end Properties.NetworkPeriod
