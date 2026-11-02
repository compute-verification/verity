import Properties.NetworkPeriod
import Proofs.Properties.NetworkScan.Accept
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Each form's `K` bounds `PeriodLeakAccepted`'s count: the proofs of `ClockKBoundsCount` and `RowsKBoundsCount`

Each factor is at most 2 to the power of the ceiling of its `log2` (`le_two_pow_clog2`), and `K`'s sums over the
connections and the windows (the compute audits, or the scans) are sums over `Fin m` and `Fin n` (`Array.ofFn`), so
`2 ^ K` is the product of the factors' bounds.

`RowsProgWired` and `RowsStateTied` read the check's refusals off its findings: on acceptance every finding is none
(`firstFinding_none`), each link-window's `wiredCheck` and `stateCheck` among them.

`AuditFreeOfLayout` reads `auditOfProgram`'s steps one by one, each a bind (`bind_ok`), and keeps the pins, the circuit
record and the inputs it counts from them. Its honest lemma has the kernel evaluate `auditOfProgram` at an audit
without a circuit record (`decide +kernel`); with one, `entriesOf` compares `Json` by its `==`, a `partial def` no
proof evaluates.
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
    (r : Fin n → Lean.Json) (w : Fin n → Facts) :
    (terms (Array.ofFn a) (Array.ofFn r) (Array.ofFn w) (Array.ofFn caps) jitter syncs windows).K =
      clog2 (syncs ^ windows) + (∑ j, clog2 (caps j + 1)) + ∑ i, (AuditTerms.of (a i) (r i) (w i) jitter).bits := by
  simp only [Public.K, terms, capsTerm_ofFn, zip_ofFn, Array.map_ofFn]
  rw [foldl_ofFn]
  rfl

theorem clockKBoundsCount : ClockKBoundsCount := by
  intro jitter syncs windows m n caps a r w
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

/-- A compute audit whose Program has one template of one member, `SlotEntry_v1{KO=1}` at the closure digest `e`, and
whose circuit record, of SHA-512 `c`, registers that template as an entry template at that digest; and the schedule of
one slot a window (one lane, T = 1, `every` 1). -/
def slotAudit : Audit :=
  ⟨{}, 0, "", 1, 0, #[("SlotEntry_v1{KO=1}", 1, "e")], "c", { templates := #[("SlotEntry_v1{KO=1}", "e")] }⟩

def slotSchedule : Schedule := ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1⟩

/-- A registration whose window binds the circuit record of SHA-512 `circuit`, with template `key`'s entries port at
the root `r` and the domain `d`. -/
def entriesReg (key circuit : String) : Lean.Json :=
  Lean.Json.mkObj [
    ("roots", Lean.Json.arr #[Lean.Json.mkObj
      [("port", .str s!"{key}/c0/o0"), ("leaves", Lean.toJson (1 : ℕ)), ("root", "r"), ("domain", "d")]]),
    ("window", Lean.Json.mkObj [("circuit", .str circuit)])]

/-- The slot audit's registration, binding its circuit record. -/
def slotReg : Lean.Json := entriesReg "SlotEntry_v1{KO=1}" "c"

/-- Its honest side: a link-window whose scan's `prog` is at that port's root and domain is wired, and the slot audit
is tied to it, so its free inputs are in the link-window's term. -/
theorem rowsProgWired_honest :
    wired #[slotAudit] slotSchedule #[slotReg] tiedWindow = true ∧ tiedIn #[slotReg] #[tiedWindow] 0 = true := by
  decide +kernel

/-- A second model's slot entry template, of another vocabulary size and other stop ids, which no constant of the check
names. -/
def otherKey : String := "NetworkPeriodEntry_v1{V=512,STOP=[3,7]}"

/-- A compute audit serving the second model: its Program's one template of one member, `otherKey` at the closure digest
`f`, which its circuit record, of SHA-512 `c'`, registers as an entry template at that digest, with the vocabulary size
512 and the stop ids [3, 7] it binds. -/
def otherAudit : Audit := ⟨{}, 0, "", 1, 0, #[(otherKey, 1, "f")], "c'", ⟨#[(otherKey, "f")], 512, #[3, 7]⟩⟩

/-- Its honest side, with no change to the check: a link-window whose scan's `prog` is at the root and domain of the
entries port of the template its registered circuit record names is wired, and the audit is tied to it. -/
theorem rowsProgWired_other :
    wired #[otherAudit] slotSchedule #[entriesReg otherKey "c'"] tiedWindow = true ∧
      tiedIn #[entriesReg otherKey "c'"] #[tiedWindow] 0 = true := by
  decide +kernel

/-- Its tampered sides: the second model's link-window is unwired when its audit's template has another body than the
one its circuit record registers (closure digest `g`), and when the audit's registration binds another record (`c`)
than the one that registers the template, so a record made after the coins registers nothing. -/
theorem rowsProgWired_other_tampered :
    wired #[{ otherAudit with entries := #[(otherKey, 1, "g")] }] slotSchedule #[entriesReg otherKey "c'"]
        tiedWindow = false ∧
      wired #[otherAudit] slotSchedule #[entriesReg otherKey "c"] tiedWindow = false := by
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

/-- `slotAudit` with its template's body another than the one its circuit record registers: a template under the
registered descriptor id, one member a slot, whose closure digest is not the registered `e`. -/
def renamedAudit : Audit := { slotAudit with entries := #[("SlotEntry_v1{KO=1}", 1, "")] }

/-- A renamed body is no slot entries: the link-window `slotAudit` wires is unwired when its audit's template has
another body, with its port, root, domain and members unchanged. -/
theorem renamedAudit_unwired : wired #[renamedAudit] slotSchedule #[slotReg] tiedWindow = false := by
  decide +kernel

/-- **No period check that accepts every period has `RowsProgWired`'s property at a renamed body** (its nontrivial
lemma at the registered closure digest): such a check accepts a rows-form period whose scan's `prog` is at the root and
domain of an audit's template's entries port, one member a slot, whose body is not the one its registered circuit
record names. -/
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
entries port at `r` and `d`, binding its circuit record. -/
def stateReg (root : String) : Lean.Json :=
  Lean.Json.mkObj [
    ("registered", Lean.Json.arr #[Lean.Json.mkObj
      [("value", "verity/certifier/scan-state/v1"), ("root", root), ("domain", "e")]]),
    ("roots", Lean.Json.arr #[Lean.Json.mkObj
      [("port", "SlotEntry_v1{KO=1}/c0/o0"), ("leaves", Lean.toJson (1 : ℕ)), ("root", "r"), ("domain", "d")]]),
    ("window", Lean.Json.mkObj [("circuit", "c")])]

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

/-! ## `K`'s free inputs are counted from the registered circuits -/

open Properties.NetworkScan.Fits (bind_ok)

theorem mapError_ok {ε ε' α : Type} {f : ε → ε'} {x : Except ε α} {a : α} (h : x.mapError f = .ok a) :
    x = .ok a := by
  cases x with
  | error e => cases h
  | ok b => cases h; rfl

theorem auditFreeOfLayout : AuditFreeOfLayout := by
  intro a P ctx A h
  unfold auditOfProgram at h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨pins, hpins, h⟩ := bind_ok h
  obtain ⟨⟨circuit, registered, inputs⟩, ht, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  cases h
  refine ⟨fun r pins' hr hp => ?_, fun hn reg => ?_⟩
  · rw [hpins] at hp
    cases hp
    rw [hr] at ht
    obtain ⟨d, hd, ht⟩ := bind_ok ht
    obtain ⟨_, -, ht⟩ := bind_ok ht
    obtain ⟨i, hi, ht⟩ := bind_ok ht
    cases ht
    exact ⟨mapError_ok hd, mapError_ok hi⟩
  · cases hc : a.getObjVal? "circuit" with
    | ok r => rw [hc] at hn; cases hn
    | error e =>
      rw [hc] at ht
      cases ht
      simp [Audit.free, Inputs.heldBits]

/-- **`AuditFreeOfLayout`'s acceptance holds at an honest audit**: `auditOfProgram` accepts `oneTemplateAudit`, an
audit of `oneTemplate`'s two units over three steps without a circuit record, so the guarantee's hypothesis holds at
an audit an honest prover registers. -/
theorem auditFreeOfLayout_honest : (auditOfProgram oneTemplateAudit oneTemplate "audit").isOk = true := by
  decide +kernel

/-- **`AuditFreeOfLayoutAt` fails at an audit a verifier accepting every audit would take**: `oneFreeBit` claims one
free input bit of `noTemplate`, a Program with none, so its free count under any registration is not its Program's
input bits. -/
theorem auditFreeOfLayout_nontrivial : ¬ AuditFreeOfLayoutAt .null noTemplate "audit" oneFreeBit := by
  rintro ⟨-, h⟩
  exact absurd (h rfl .null) (by decide +kernel)

end Properties.NetworkPeriod
