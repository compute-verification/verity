import Proofs.Flock.Claims.Recursion.VStarFit

/-!
# The deployed assigner fits the port inputs to the circuit (`fitPin`, `A0`)

The trace assigner (`VStarTrace.A`) places each port's rows and salts where `TraceOk.rows` holds: rows of the port's
words, salts of 192 bytes. The stager's rows from the inner prover's records (`InnerRec.rest`: the openings' fresh salts,
the running sums' salts, κ, eq(α), v) are whatever the records hold. The deployed assigner `A0` fits them first: each
port's rows to the port's words (16 bits a word, `fitRow`) and each salt to 192 bytes (`fitSalt`), then assigns. So
`TraceOk.rows` holds at `fitPin c pin` for every `pin` (`rows_fitPin`), and at inputs of the deployed sizes `fitPin`
changes nothing (`fitPortRow_of`), which is what the difftest against the deployed stager checks.
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)

/-- A salt's first 192 bytes, zero bytes past its end. -/
def fitSalt (b : ByteArray) : ByteArray :=
  ⟨(Array.range HmRow.SALT_BYTES).map fun i => b.data.getD i 0⟩

/-- A row fitted to `k` words and a 192-byte salt. -/
def fitPortRow (k : ℕ) (r : VStar.In.Row) : VStar.In.Row := ⟨fitRow k r.words, fitSalt r.salt⟩

/-- Port `nm`'s word count at circuit `c` (its first port of that name). -/
def portWords (c : Circuit) (nm : String) : ℕ :=
  (((c.ports.findIdx? (·.name == nm)).bind (c.ports[·]?)).map (·.words)).getD 0

/-- **The port inputs fitted to the circuit**: each port's rows to its words, each salt to 192 bytes. -/
def fitPin (c : Circuit) (pin : VStar.In.PartIn) : VStar.In.PartIn :=
  pin.map fun e => (e.1, { e.2 with rows := e.2.rows.map (fitPortRow (portWords c e.1)) })

/-- **The deployed assigner**: the trace assigner at the fitted port inputs. -/
def A0 (mask : ℕ → ℕ → Bool) (c : Circuit) (pin : VStar.In.PartIn) : Claims.Flock.Wit :=
  VStarTrace.A mask c (fitPin c pin)

theorem fitSalt_size (b : ByteArray) : (fitSalt b).size = HmRow.SALT_BYTES := by
  show ((Array.range HmRow.SALT_BYTES).map fun i => b.data.getD i 0).size = _
  rw [Array.size_map, Array.size_range]

theorem fitSalt_of {b : ByteArray} (h : b.size = HmRow.SALT_BYTES) : fitSalt b = b := by
  cases b with
  | mk d =>
    simp only [fitSalt, ByteArray.mk.injEq]
    apply Array.ext (by simpa [ByteArray.size] using h.symm)
    intro i h1 h2
    simp [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem h2]

theorem fitRow_of {k : ℕ} {ws : Array ℕ} (hk : ws.size = k) (hb : ∀ (j : ℕ) (h : j < ws.size), ws[j] < 65536) :
    fitRow k ws = ws := by
  subst hk
  apply Array.ext (by simp [fitRow])
  intro j h1 h2
  simp only [fitRow, List.getElem_toArray, List.getElem_take]
  rw [List.getElem_append_left (by simpa using h2)]
  simp [Nat.mod_eq_of_lt (hb j h2)]

/-- **A row of the deployed sizes is fitted unchanged.** -/
theorem fitPortRow_of {k : ℕ} {r : VStar.In.Row} (hk : r.words.size = k)
    (hb : ∀ (j : ℕ) (h : j < r.words.size), r.words[j] < 65536) (hs : r.salt.size = HmRow.SALT_BYTES) :
    fitPortRow k r = r := by
  cases r
  simp only [fitPortRow, fitRow_of hk hb, fitSalt_of hs]

/-- **Port `nm`'s row at instance `n`, fitted**: the stager's, fitted to the port's words. -/
theorem portRow_fitPin (c : Circuit) (pin : VStar.In.PartIn) (nm : String) (n : ℕ) :
    portRow (fitPin c pin) nm n = (portRow pin nm n).map (fitPortRow (portWords c nm)) := by
  unfold portRow fitPin
  rw [Array.find?_map]
  cases hf : pin.find? (fun e => e.1 == nm) with
  | none => simp [Function.comp_def, hf]
  | some e =>
    have he : e.1 = nm := by simpa using Array.find?_some hf
    simp only [Function.comp_def, hf, Option.map_some]
    cases hr : e.2.refs[n]? with
    | none => simp [hr]
    | some i => simp [hr, Array.getElem?_map, he]

/-- The first port of a name has that name's words. -/
theorem portWords_of {c : Circuit} {nm : String} {q : ℕ} (hq : c.ports.findIdx? (·.name == nm) = some q)
    (hqs : q < c.ports.size) : portWords c nm = c.ports[q].words := by
  unfold portWords
  rw [hq]
  simp [Array.getElem?_eq_getElem hqs]

/-- **`TraceOk.rows` at the fitted port inputs**, where each port's name gives its words (`portWords`): its rows are of
the port's words, its salts 192 bytes. -/
theorem rows_fitPin (c : Circuit) (pin : VStar.In.PartIn)
    (hw : ∀ q (hq : q < c.ports.size), portWords c c.ports[q].name = c.ports[q].words) :
    ∀ q (hq : q < c.ports.size) n r, portRow (fitPin c pin) c.ports[q].name n = some r →
      r.words.size = c.ports[q].words ∧ r.salt.size = HmRow.SALT_BYTES := by
  intro q hq n r h
  rw [portRow_fitPin] at h
  obtain ⟨r0, -, rfl⟩ := Option.map_eq_some_iff.1 h
  exact ⟨by simp [fitPortRow, fitRow_size, hw q hq], fitSalt_size _⟩

/-- **`TraceOk.refs` at the fitted port inputs**: `fitPin` keeps every port's refs. -/
theorem refs_fitPin {c : Circuit} {pin : VStar.In.PartIn}
    (h : ∀ x ∈ pin.toList, x.2.refs.size = VStar.Trace.count pin) :
    ∀ x ∈ (fitPin c pin).toList, x.2.refs.size = VStar.Trace.count (fitPin c pin) := by
  have hc : VStar.Trace.count (fitPin c pin) = VStar.Trace.count pin := by
    unfold VStar.Trace.count fitPin
    rw [Array.getElem?_map]
    cases pin[0]? <;> rfl
  intro x hx
  rw [hc]
  simp only [fitPin, Array.toList_map, List.mem_map] at hx
  obtain ⟨e, he, rfl⟩ := hx
  exact h e he

end RecDeploy.VStarBuild
