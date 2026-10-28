import Mathlib
import Properties.CertifierDevice

/-!
# The certifier device, proved: loops, the certifier's operations, packed rows

A loop of the device's `Id.run do` code is a `forIn` over a list; `forIn_inv` carries an invariant through it. The
certifier's fault (`Certifier.doFault`) keeps the first reason, so a certifier and its faulted forms are closed under
faulting (`CertRel`). A row the certifier releases is `Packed`, and what a packed row puts on the link (`wire`) is
`r · B` bytes, the frames the code writes.
-/

namespace Properties.CertifierDevice.Proofs

open _root_.CertifierDevice
open NetworkCertifier (Reason Status)

/-! ## Loops -/

theorem forIn_inv {α β : Type _} (P : β → Prop) {l : List α} {b : β} {f : α → β → Id (ForInStep β)}
    (h0 : P b) (hf : ∀ a ∈ l, ∀ b, P b → P (f a b).run.value) : P (forIn l b f).run := by
  induction l generalizing b with
  | nil => simpa using h0
  | cons a l ih =>
    rw [List.forIn_cons]
    have h1 := hf a (by simp) b h0
    rcases h : (f a b).run with b' | b'
    · simp only [h, ForInStep.value] at h1
      have : f a b = .done b' := h
      simp only [this]; exact h1
    · simp only [h, ForInStep.value] at h1
      have : f a b = .yield b' := h
      simp only [this]
      exact ih h1 (fun a' ha' => hf a' (by simp [ha']))

/-! ## The certifier -/

/-- `c'` is `c` or `c` faulted. -/
def CertRel (c c' : Certifier) : Prop := c' = c ∨ ∃ r, c' = c.doFault r

theorem doFault_doFault (c : Certifier) (r r' : Reason) : (c.doFault r).doFault r' = c.doFault r := by
  unfold Certifier.doFault
  cases h : c.fault <;> simp [h]

theorem CertRel.refl (c : Certifier) : CertRel c c := Or.inl rfl

theorem CertRel.doFault {c c' : Certifier} (h : CertRel c c') (r : Reason) : CertRel c (c'.doFault r) := by
  rcases h with rfl | ⟨r', rfl⟩
  · exact Or.inr ⟨r, rfl⟩
  · exact Or.inr ⟨r', doFault_doFault c r' r⟩

theorem CertRel.trans {c c' c'' : Certifier} (h : CertRel c c') (h' : CertRel c' c'') : CertRel c c'' := by
  rcases h' with rfl | ⟨r, rfl⟩
  · exact h
  · exact h.doFault r

theorem doFault_link (c : Certifier) (r : Reason) : (c.doFault r).link = c.link := by
  unfold Certifier.doFault; split <;> rfl
theorem doFault_params (c : Certifier) (r : Reason) : (c.doFault r).params = c.params := by
  unfold Certifier.doFault; split <;> rfl
theorem doFault_window (c : Certifier) (r : Reason) : (c.doFault r).window = c.window := by
  unfold Certifier.doFault; split <;> rfl
theorem doFault_rows (c : Certifier) (r : Reason) : (c.doFault r).rows = c.rows := by
  unfold Certifier.doFault; split <;> rfl
theorem doFault_queue (c : Certifier) (r : Reason) :
    (c.doFault r).queue = c.queue ∨ (c.doFault r).queue = #[] := by
  unfold Certifier.doFault; split
  · exact Or.inr rfl
  · exact Or.inl rfl
theorem doFault_of_fault {c : Certifier} {r : Reason} (h : c.fault = some r) (r' : Reason) : c.doFault r' = c := by
  unfold Certifier.doFault; simp [h]
theorem doFault_fault_isSome (c : Certifier) (r : Reason) : (c.doFault r).fault.isSome := by
  unfold Certifier.doFault; cases h : c.fault <;> simp [h]
theorem doFault_fault (c : Certifier) (r : Reason) : (c.doFault r).fault = some (c.fault.getD r) := by
  unfold Certifier.doFault; cases h : c.fault <;> simp [h]

/-- What a certifier's operations keep, apart from the rows and the window: its link and parameters, frames of `B` bytes
in its queue, and no queue once faulted. -/
structure Shape (c : Certifier) (link : String) (p : Params) : Prop where
  link : c.link = link
  params : c.params = p
  sizes : ∀ f ∈ c.queue, f.size = p.frameBytes
  dropped : c.fault.isSome → c.queue = #[]

theorem Shape.doFault {c : Certifier} {link : String} {p : Params} (h : Shape c link p) (r : Reason) :
    Shape (c.doFault r) link p := by
  refine ⟨(doFault_link c r).trans h.link, (doFault_params c r).trans h.params, ?_, ?_⟩
  · unfold Certifier.doFault; split
    · simp
    · exact h.sizes
  · unfold Certifier.doFault; split
    · simp
    · intro hs; exact h.dropped hs

theorem CertRel.shape {c c' : Certifier} {link : String} {p : Params} (h : CertRel c c') (hs : Shape c link p) :
    Shape c' link p := by
  rcases h with rfl | ⟨r, rfl⟩
  · exact hs
  · exact hs.doFault r

theorem CertRel.link {c c' : Certifier} (h : CertRel c c') : c'.link = c.link := by
  rcases h with rfl | ⟨r, rfl⟩ <;> simp [doFault_link]
theorem CertRel.params {c c' : Certifier} (h : CertRel c c') : c'.params = c.params := by
  rcases h with rfl | ⟨r, rfl⟩ <;> simp [doFault_params]
theorem CertRel.window {c c' : Certifier} (h : CertRel c c') : c'.window = c.window := by
  rcases h with rfl | ⟨r, rfl⟩ <;> simp [doFault_window]
theorem CertRel.rows {c c' : Certifier} (h : CertRel c c') : c'.rows = c.rows := by
  rcases h with rfl | ⟨r, rfl⟩ <;> simp [doFault_rows]
theorem CertRel.of_fault {c c' : Certifier} {r : Reason} (h : CertRel c c') (hf : c.fault = some r) : c' = c := by
  rcases h with rfl | ⟨r', rfl⟩
  · rfl
  · exact doFault_of_fault hf r'

theorem enqueue_eq (c : Certifier) (frame : ByteArray) :
    c.enqueue frame = c ∨ (∃ r, c.enqueue frame = c.doFault r) ∨
      (c.fault = none ∧ frame.size = c.params.frameBytes ∧ c.queue.size < c.params.queueCapacity ∧
        c.enqueue frame = { c with queue := c.queue.push frame }) := by
  unfold Certifier.enqueue
  by_cases h1 : c.fault.isSome
  · simp [h1]
  · simp only [h1, Bool.false_eq_true, ite_false]
    by_cases h2 : frame.size ≠ c.params.frameBytes
    · exact Or.inr (Or.inl ⟨.malformed, by simp [h2]⟩)
    · by_cases h3 : c.queue.size ≥ c.params.queueCapacity
      · exact Or.inr (Or.inl ⟨.overflow, by simp [h2, h3]⟩)
      · refine Or.inr (Or.inr ⟨by simpa using h1, by simpa using h2, by omega, by simp [h2, h3]⟩)

theorem enqueue_link (c : Certifier) (f : ByteArray) : (c.enqueue f).link = c.link := by
  rcases enqueue_eq c f with h | ⟨r, h⟩ | ⟨-, -, -, h⟩ <;> rw [h]; all_goals simp [doFault_link]
theorem enqueue_params (c : Certifier) (f : ByteArray) : (c.enqueue f).params = c.params := by
  rcases enqueue_eq c f with h | ⟨r, h⟩ | ⟨-, -, -, h⟩ <;> rw [h]; all_goals simp [doFault_params]
theorem enqueue_window (c : Certifier) (f : ByteArray) : (c.enqueue f).window = c.window := by
  rcases enqueue_eq c f with h | ⟨r, h⟩ | ⟨-, -, -, h⟩ <;> rw [h]; all_goals simp [doFault_window]
theorem enqueue_rows (c : Certifier) (f : ByteArray) : (c.enqueue f).rows = c.rows := by
  rcases enqueue_eq c f with h | ⟨r, h⟩ | ⟨-, -, -, h⟩ <;> rw [h]; all_goals simp [doFault_rows]
theorem enqueue_of_fault {c : Certifier} {r : Reason} (h : c.fault = some r) (f : ByteArray) : c.enqueue f = c := by
  unfold Certifier.enqueue; simp [h]

theorem Shape.enqueue {c : Certifier} {link : String} {p : Params} (h : Shape c link p) (f : ByteArray) :
    Shape (c.enqueue f) link p := by
  rcases enqueue_eq c f with he | ⟨r, he⟩ | ⟨hn, hsz, -, he⟩ <;> rw [he]
  · exact h
  · exact h.doFault r
  · refine ⟨h.link, h.params, ?_, ?_⟩
    · intro g hg
      rcases Array.mem_push.1 hg with hg | rfl
      · exact h.sizes g hg
      · rw [hsz, h.params]
    · simp [hn]

/-! ## Rows -/

theorem mem_of_mem_extract {α : Type _} {a : Array α} {x : α} {i j : Nat} (h : x ∈ a.extract i j) : x ∈ a := by
  rw [← Array.mem_toList_iff, Array.toList_extract] at h
  exact Array.mem_toList_iff.1 (List.mem_of_mem_drop (List.mem_of_mem_take h))

theorem packed_replicate (p : Params) : Packed p (Array.replicate p.allocation none) :=
  ⟨#[], by simp, by simp, by simp⟩

theorem rowCount_packed (fs : Array ByteArray) (k : Nat) :
    rowCount (fs.map some ++ Array.replicate k none) = fs.size := by
  unfold rowCount
  rw [← Array.length_toList]
  simp [List.filter_map, Function.comp_def]

theorem filterMap_packed (fs : Array ByteArray) (k : Nat) :
    (fs.map some ++ Array.replicate k none).filterMap id = fs := by
  apply Array.ext'
  simp

theorem packed_tick {c : Certifier} {link : String} {p : Params} (h : Shape c link p) :
    Packed p ((c.queue.extract 0 (min c.queue.size c.params.allocation)).map some ++
      Array.replicate (c.params.allocation - min c.queue.size c.params.allocation) none) := by
  rw [h.params]
  refine ⟨c.queue.extract 0 (min c.queue.size p.allocation), ?_, ?_, ?_⟩
  · simp
  · intro f hf
    exact h.sizes f (mem_of_mem_extract hf)
  · simp

theorem empty_append' (b : ByteArray) : ByteArray.empty ++ b = b := by
  apply ByteArray.ext; simp

theorem wire_size {p : Params} {row : Slots} (h : Packed p row) : (wire p row).size = bucketBytes p := by
  obtain ⟨fs, hle, hsz, rfl⟩ := h
  unfold wire bucketBytes
  rw [rowCount_packed, filterMap_packed, ByteArray.size_append]
  have : (fs.foldl (· ++ ·) ByteArray.empty).size = fs.size * p.frameBytes := by
    rw [← Array.foldl_toList]
    have key : ∀ (l : List ByteArray) (acc : ByteArray), (∀ f ∈ l, f.size = p.frameBytes) →
        (l.foldl (· ++ ·) acc).size = acc.size + l.length * p.frameBytes := by
      intro l
      induction l with
      | nil => simp
      | cons f l ih =>
        intro acc hl
        rw [List.foldl_cons, ih _ (fun g hg => hl g (by simp [hg])), ByteArray.size_append,
          hl f (by simp)]
        simp [Nat.succ_mul]; ring
    rw [key _ _ (fun f hf => hsz f (by simpa using hf))]
    simp
  rw [this]
  change fs.size * p.frameBytes + (Array.replicate _ _).size = _
  rw [Array.size_replicate, ← Nat.add_mul, Nat.add_sub_cancel' hle]

/-- What the code writes for a row: the frames of its first `rowCount` slots, then the padding. -/
def codeBytes (p : Params) (row : Slots) : ByteArray :=
  (row.extract 0 (rowCount row)).foldl (fun acc f => acc ++ f.getD ByteArray.empty) ByteArray.empty ++
    ⟨Array.replicate ((p.allocation - rowCount row) * p.frameBytes) 0⟩

theorem codeBytes_packed {p : Params} {row : Slots} (h : Packed p row) : codeBytes p row = wire p row := by
  obtain ⟨fs, -, -, rfl⟩ := h
  unfold codeBytes wire
  rw [rowCount_packed, filterMap_packed]
  congr 1
  have : (fs.map some ++ Array.replicate (p.allocation - fs.size) none).extract 0 fs.size = fs.map some := by
    rw [Array.extract_append]; simp
  rw [this, Array.foldl_map]
  rfl

theorem wire_replicate (p : Params) :
    wire p (Array.replicate p.allocation none) = ⟨Array.replicate (bucketBytes p) 0⟩ := by
  have h := filterMap_packed #[] p.allocation
  have h' := rowCount_packed #[] p.allocation
  simp only [Array.map_empty, Array.empty_append, Array.size_empty] at h h'
  unfold wire bucketBytes
  rw [h, h']
  simp

end Properties.CertifierDevice.Proofs
