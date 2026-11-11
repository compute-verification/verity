module

public import Proofs.Flock.SpotFlock.SlotsAt
public import Proofs.Flock.SpotFlock.Facts
public import Proofs.Flock.SamplingGeneric.TwoCount

/-!
# At a program of proof blocks side by side, an address holds only under the statement's roots

`Addresses h` (`SlotsAt`) gives every gate one address that holds at every accepted run's checked unit that needs it.
An address holds through one of two branches: the address's port reads a registered value, under the roots round 2
reads that value under (`ReadsRegistered`, `readRoots`), or it reads the interior record, at a gate its replay block
doesn't commit (`ReadsInterior`). This file is what those two branches come to where the statement's proof blocks stand
side by side, each reading only its own gates and constants:

* `BlocksClosed C Pt`: every proof block of `Pt` reads only its own gates and constants, and holds no constant.
  `SideBySide h` is that at the code's statement: the members' programs (`bitsProgs h.ms`) and units
  (`membersPart h.ms`). It is a hypothesis here: nothing in this file derives it from how the members' programs are
  laid out.
* `needs_committed_of_closed`, `needs_boundary_of_closed`, `CheckedRun.needs_boundary`: there, a gate a proof block
  needs is returned or is an input, so it is on the boundary of every set of gates, its replay block's among them.
* `CheckedRun.roots_of_boundary`, `CheckedRun.not_readsInterior`: at a gate on its replay block's boundary, the roots
  the program opens it under are the statement's, and `ReadsInterior` fails.
* `roots_ne_irootsOf`, `CheckedRun.roots_ne`: the statement's roots are never a replay block's interior
  roots. The statement's are one message and four an entry of the boundary record (`portRoots`); a replay block's are
  two messages more, over the interior record, or none (`irootsOf`).
* `CheckedRun.addressHolds_reads`, `CheckedRun.addresses_reads`: so an address that holds at a needed gate names a
  registered read that round 2 reads under the statement's roots, and under no replay block's interior roots.
* `CheckedRun.not_addressHolds_of_interior`, `CheckedRun.not_addresses`: where, at one accepted run's checked unit and
  one gate it needs, every registered read named like a port whose row holds the gate's bit (`RowAt`) is read under
  the replay block's interior roots, no address holds at the gate, and `Addresses h` is false.
  `readRoots_of_readsInterior` names those reads: round 2 reads the value off the interior record (`readsInterior`).
* `CheckedRun.addresses_not_readsInterior`, `CheckedRun.not_addresses_of_readsInterior`: the same two in round 2's
  own terms: under `Addresses h` every needed gate has an address whose registered read round 2 doesn't read off the
  interior record, and one accepted run with a checked unit that needs a gate, all of whose session's registered reads
  round 2 reads off the interior record, refutes `Addresses h`.
* `CheckedRun.not_addresses_of_boundary`, `CheckedRun.not_addresses_of_input`,
  `CheckedRun.not_addresses_of_input_readsInterior`: the refutation at one gate with no hypothesis on the program: the
  gate is on its replay block's boundary, as every input gate is.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs membersPart)
open Lean (Json)

/-! ## A needed gate of a closed block is returned or is an input -/

/-- **The proof blocks stand side by side**: each gate of a proof block reads only gates of that block and constants,
and no proof block holds a constant. -/
def BlocksClosed (C : Prog) (Pt : Partition) : Prop :=
  (∀ b ∈ Pt.blocks, ∀ g ∈ b, ∀ x : BGate, C.gates[g]? = some x →
      ∀ i ∈ x.reads, i ∈ b ∨ ∃ c, C.gates[i]? = some (.const c)) ∧
    ∀ b ∈ Pt.blocks, ∀ g ∈ b, ∀ c, C.gates[g]? ≠ some (.const c)

/-- **Among proof blocks side by side, a gate a block needs is returned or is an input**: it is a gate of the block
that the block's boundary commits, and a gate outside the block that read it would be a gate of another block, which
reads only its own gates and constants. -/
theorem needs_committed_of_closed {C : Prog} {Pt : Partition} (hpart : Pt.Partitions C) (hcl : BlocksClosed C Pt)
    {u : ℕ} {bu : List ℕ} (hu : Pt.blocks[u]? = some bu) {g : ℕ} (hg : C.Needs bu g) :
    g ∈ C.outs ∨ C.gates[g]? = some .input := by
  obtain ⟨hc, hnc⟩ := hcl
  have hbu : bu ∈ Pt.blocks := List.mem_of_getElem? hu
  have hown : g ∈ bu ∧ C.Boundary bu g = true := by
    unfold Prog.Needs at hg
    rcases hg with hg | ⟨⟨g', hg', x, hx, hi⟩, hgb, hncg⟩
    · exact hg
    · rcases hc bu hbu g' hg' x hx g hi with hin | ⟨c, hcc⟩
      · exact absurd hin hgb
      · exact absurd hcc (hncg c)
  obtain ⟨hgb, hB⟩ := hown
  unfold Prog.Boundary at hB
  simp only [Bool.or_eq_true, List.contains_iff_mem, decide_eq_true_eq, List.any_eq_true, List.mem_range,
    Bool.and_eq_true] at hB
  rcases hB with (hout | hin) | ⟨g', hg', hg'u, hr⟩
  · exact Or.inl hout
  · exact Or.inr hin
  · exfalso
    obtain ⟨x, hx, hi⟩ := C.mem_reads_getD.1 hr
    have hg'P : g' ∈ Pt.gates := by
      refine hpart.2.2.1 g' hg' (fun c hc' => ?_) (fun hun => ?_)
      · rw [hx] at hc'
        obtain rfl := Option.some.inj hc'
        simp [BGate.reads] at hi
      · rw [hun.1] at hx
        obtain rfl := Option.some.inj hx
        simp [BGate.reads] at hi
    obtain ⟨v, hv, hgv⟩ := Claims.Sampling.exists_block hg'P
    have hbv : Pt.blocks[v]? = some (Pt.blocks.getD v []) := Claims.Sampling.getElem?_getD hv
    have hbvm : Pt.blocks.getD v [] ∈ Pt.blocks := List.mem_of_getElem? hbv
    rcases hc _ hbvm g' hgv x hx g hi with hin' | ⟨c, hcc⟩
    · have hvu : v = u := Claims.Sampling.eq_of_mem_of_nodup hpart.1 hbv hu hin' hgb
      have h1 : Pt.blocks.getD v [] = bu := by
        rw [hvu]
        exact Claims.Sampling.getD_of_getElem? hu
      rw [h1] at hgv
      exact hg'u hgv
    · exact hnc bu hbu g hgb c hcc

/-- **Among proof blocks side by side, a gate a block needs is on every set's boundary**: returned or an input
(`needs_committed_of_closed`), it is committed whatever the set of gates. -/
theorem needs_boundary_of_closed {C : Prog} {Pt : Partition} (hpart : Pt.Partitions C) (hcl : BlocksClosed C Pt)
    {u : ℕ} {bu : List ℕ} (hu : Pt.blocks[u]? = some bu) {g : ℕ} (hg : C.Needs bu g) (rb : List ℕ) :
    C.Boundary rb g = true := by
  rcases needs_committed_of_closed hpart hcl hu hg with hout | hin
  · exact Prog.boundary_of_out rb hout
  · exact Prog.boundary_of_input rb hin

/-! ## The statement's roots are no replay block's interior roots -/

/-- A replay block's interior roots are two messages and a record's roots, or none. -/
theorem length_irootsOf (m₁ : Msg) (ρ : ℕ) :
    (∃ rec, (irootsOf m₁ ρ).length = 4 * (tagged rec).length + 3) ∨ (irootsOf m₁ ρ).length = 0 := by
  unfold irootsOf
  cases jsonMsg m₁ with
  | none => exact Or.inr (by simp)
  | some rec =>
    refine Or.inl ⟨rec, ?_⟩
    have h4 := length_portRoots rec
    simp only [Option.map_some, Option.getD_some, List.length_cons]
    omega

/-- The roots a statement decodes to are its record's. -/
theorem decode_roots {x : Msg} {st : Roots} (hx : decode x = some st) : ∃ rec, st.roots = portRoots rec := by
  unfold decode at hx
  cases hj : jsonMsg x with
  | none => simp [hj] at hx
  | some rec => rw [hj] at hx; cases hx; exact ⟨rec, rfl⟩

/-- **The statement's roots are never a replay block's interior roots**: the two lists never have one length. -/
theorem roots_ne_irootsOf {x : Msg} {st : Roots} (hx : decode x = some st) (m₁ : Msg) (ρ : ℕ) :
    st.roots ≠ irootsOf m₁ ρ := by
  obtain ⟨rec, hrec⟩ := decode_roots hx
  intro heq
  have h1 := length_portRoots rec
  have h3 : (portRoots rec).length = (irootsOf m₁ ρ).length := by rw [← hrec, heq]
  rcases length_irootsOf m₁ ρ with ⟨rec', h2⟩ | h2 <;> omega

variable (h : Held)

/-- The roots of the statement a run's statement parses to are the roots it decodes to. -/
theorem parse_roots {x : Msg} {s : AuditStatement} (hs : parse h x = some s) :
    ∃ st, decode x = some st ∧ s.roots = st.roots := by
  unfold parse at hs
  cases hd : decode x with
  | none => simp [hd] at hs
  | some st => rw [hd] at hs; cases hs; exact ⟨st, rfl, rfl⟩

/-- **Round 2 reads a value off the interior record under the replay block's interior roots**: where the run's
statement decodes, its round-1 message is a record and its second stage's sample evaluates, a registered read of
value `v` at template `t`'s port `port` that round 2 reads off the interior record (`readsInterior`) is read under the
replay block's interior roots. -/
theorem readRoots_of_readsInterior {x m₁ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {st : Roots}
    {rec smp : Json}
    (hx : decode x = some st) (hm : jsonMsg m₁ = some rec) (hd : secondDraw h c₀ c₁ = some smp) (ρ t : ℕ)
    (port v : String) (hri : readsInterior (ownOf h st rec smp c₀) (h.names.getD t "") port v = true) :
    readRoots h x m₁ c₀ c₁ ρ t port v = irootsOf m₁ ρ := by
  unfold readRoots
  rw [hx, hm, hd]
  dsimp only
  exact if_pos hri

/-- **The code's proof blocks stand side by side**: at the members' programs side by side and the members' units,
each unit's gates read only that unit's gates and constants, and no unit holds a constant. -/
def SideBySide : Prop := BlocksClosed (bitsProgs h.ms) (membersPart h.ms)

namespace CheckedRun

variable {h} {r : CheckedRun h.stage}

/-! ## At a checked unit of an accepted run -/

/-- **`needs_boundary`**: where the code's proof blocks stand side by side, every gate a checked unit of an accepted
run needs is on the boundary of every set of gates. -/
theorem needs_boundary (hcl : SideBySide h) (hr : r.Ok h) {g : ℕ} (hg : r.s.prog.Needs r.b g) (rb : List ℕ) :
    r.s.prog.Boundary rb g = true := by
  obtain ⟨Pt, d, spec, hx, hP, -, hin, -, -, -, hb, -⟩ := hr
  have hprog : r.s.prog = bitsProgs h.ms := parse_prog h hx
  have hPt' : some (membersPart h.ms) = some Pt := hP
  have hPt : membersPart h.ms = Pt := Option.some.inj hPt'
  subst hPt
  have hcl' : BlocksClosed r.s.prog (membersPart h.ms) := by
    rw [hprog]
    exact hcl
  exact needs_boundary_of_closed hin.2.1 hcl' hb hg rb

/-- **At a gate on its replay block's boundary, the program opens under the statement's roots.** -/
theorem roots_of_boundary (hr : r.Ok h) {g : ℕ} (hB : r.s.prog.Boundary r.rb g = true) :
    r.roots h g = r.s.roots := by
  obtain ⟨Pt, d, spec, -, hP, h2, -, -, -, -, -, hρ, hrb, -⟩ := hr
  have he : r.roots h g =
      if g ∈ r.rb ∧ r.s.prog.Boundary r.rb g = false then iroots r.x r.seen r.ρ else r.s.roots :=
    Reveal.rootsAt_eq (partition h) (two h) (iroots []) hP h2 hρ hrb
  rw [he]
  exact if_neg fun hc => absurd (hB.symm.trans hc.2) (by decide)

/-- **`not_readsInterior`**: at a gate on its replay block's boundary no address reads the interior record. -/
theorem not_readsInterior {S : Address} {g : ℕ} (hB : r.s.prog.Boundary r.rb g = true) :
    ¬ r.ReadsInterior h S g := by
  rintro ⟨-, -, -, ⟨-, hf⟩, -⟩
  exact absurd (hB.symm.trans hf) (by decide)

/-- **`roots_ne`**: an accepted run's statement's roots are not its replay blocks' interior roots. -/
theorem roots_ne (hr : r.Ok h) (m₁ : Msg) (ρ : ℕ) : r.s.roots ≠ irootsOf m₁ ρ := by
  obtain ⟨_, _, _, hx, -⟩ := hr
  obtain ⟨st, hst, hroots⟩ := parse_roots h hx
  rw [hroots]
  exact SpotFlock.roots_ne_irootsOf hst m₁ ρ

/-- **An address that holds at a gate on its replay block's boundary reads a registered value under the statement's
roots**: the registered read named like the address's port is read by round 2 under the statement's roots, which are
no replay block's interior roots. -/
theorem addressHolds_reads_of_boundary (hr : r.Ok h) {S : Address} {g : ℕ}
    (hB : r.s.prog.Boundary r.rb g = true) (hH : r.AddressHolds h S g) :
    ∃ reg, regNamed r.ins r.sts (S.port r.u) = some reg ∧
      readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value = r.s.roots ∧
      readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value ≠ irootsOf r.m₁ r.ρ := by
  obtain ⟨-, -, hrd⟩ := hH
  rcases hrd with ⟨reg, hreg, -, hroots, -⟩ | hI
  · have he : readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value = r.s.roots :=
      hroots.symm.trans (roots_of_boundary hr hB)
    refine ⟨reg, hreg, he, ?_⟩
    rw [he]
    exact roots_ne hr r.m₁ r.ρ
  · exact absurd hI (not_readsInterior hB)

/-- **Where the code's proof blocks stand side by side, an address that holds at a needed gate reads a registered
value under the statement's roots**, and under no replay block's interior roots. -/
theorem addressHolds_reads (hcl : SideBySide h) (hr : r.Ok h) {S : Address} {g : ℕ}
    (hg : r.s.prog.Needs r.b g) (hH : r.AddressHolds h S g) :
    ∃ reg, regNamed r.ins r.sts (S.port r.u) = some reg ∧
      readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value = r.s.roots ∧
      readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value ≠ irootsOf r.m₁ r.ρ :=
  addressHolds_reads_of_boundary hr (needs_boundary hcl hr hg r.rb) hH

/-- **Under `Addresses`, side by side, every needed gate of every accepted run's checked unit has an address whose
registered read round 2 reads under the statement's roots.** -/
theorem addresses_reads (hcl : SideBySide h) (hA : Addresses h) (hr : r.Ok h) {g : ℕ}
    (hg : r.s.prog.Needs r.b g) :
    ∃ S : Address, r.AddressHolds h S g ∧ ∃ reg, regNamed r.ins r.sts (S.port r.u) = some reg ∧
      readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value = r.s.roots ∧
      readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value ≠ irootsOf r.m₁ r.ρ := by
  obtain ⟨S, hS⟩ := hA g
  have hH := hS r hr hg
  exact ⟨S, hH, addressHolds_reads hcl hr hg hH⟩

/-- **No address holds at a needed gate whose registered reads are under the interior roots**: side by side, at an
accepted run's checked unit and a gate it needs, an address whose port's registered reads round 2 reads under the
replay block's interior roots does not hold. -/
theorem not_addressHolds_of_interior (hcl : SideBySide h) (hr : r.Ok h) {S : Address} {g : ℕ}
    (hg : r.s.prog.Needs r.b g)
    (hI : ∀ reg, regNamed r.ins r.sts (S.port r.u) = some reg →
      readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value = irootsOf r.m₁ r.ρ) :
    ¬ r.AddressHolds h S g := by
  intro hH
  obtain ⟨reg, hreg, -, hne⟩ := addressHolds_reads hcl hr hg hH
  exact hne (hI reg hreg)

/-- **`Addresses` fails at a registration with a checked row under the interior roots**: where the code's proof
blocks stand side by side, one accepted run with a checked unit and a gate it needs, such that every registered read
named like a port whose row holds the gate's bit (`RowAt`) is read by round 2 under the replay block's interior
roots, refutes `Addresses h`. -/
theorem not_addresses (hcl : SideBySide h) (hr : r.Ok h) {g : ℕ} (hg : r.s.prog.Needs r.b g)
    (hI : ∀ S : Address, r.RowAt h S g → ∀ reg, regNamed r.ins r.sts (S.port r.u) = some reg →
      readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value = irootsOf r.m₁ r.ρ) :
    ¬ Addresses h := by
  intro hA
  obtain ⟨S, hS⟩ := hA g
  have hH := hS r hr hg
  exact not_addressHolds_of_interior hcl hr hg (hI S hH.2.1) hH

/-- **Under `Addresses`, side by side, round 2 reads no needed gate's row off the interior record**: at an accepted
run whose statement decodes, whose round-1 message is a record and whose second stage's sample evaluates, every gate
a checked unit needs has an address that holds, and the registered read named like its port is one round 2 doesn't
read off the interior record (`readsInterior`). -/
theorem addresses_not_readsInterior (hcl : SideBySide h) (hA : Addresses h) (hr : r.Ok h) {g : ℕ}
    (hg : r.s.prog.Needs r.b g) {st : Roots} {rec smp : Json} (hx : decode r.x = some st)
    (hm : jsonMsg r.m₁ = some rec) (hd : secondDraw h r.c₀ r.c₁ = some smp) :
    ∃ S : Address, r.AddressHolds h S g ∧ ∃ reg, regNamed r.ins r.sts (S.port r.u) = some reg ∧
      readsInterior (ownOf h st rec smp r.c₀) (h.names.getD (templateOf h.ms r.u) "") reg.name reg.value =
        false := by
  obtain ⟨S, hH, reg, hreg, -, hne⟩ := addresses_reads hcl hA hr hg
  refine ⟨S, hH, reg, hreg, eq_false_of_ne_true fun hri => ?_⟩
  exact hne (readRoots_of_readsInterior h hx hm hd r.ρ (templateOf h.ms r.u) reg.name reg.value hri)

/-- **`Addresses` fails where round 2 reads a checked unit's registered reads off the interior record**: where the
code's proof blocks stand side by side, one accepted run whose statement decodes, whose round-1 message is a record
and whose second stage's sample evaluates, with a checked unit that needs a gate and every one of whose session's
registered reads round 2 reads off the interior record (`readsInterior`), refutes `Addresses h`. -/
theorem not_addresses_of_readsInterior (hcl : SideBySide h) (hr : r.Ok h) {g : ℕ} (hg : r.s.prog.Needs r.b g)
    {st : Roots} {rec smp : Json} (hx : decode r.x = some st) (hm : jsonMsg r.m₁ = some rec)
    (hd : secondDraw h r.c₀ r.c₁ = some smp)
    (hI : ∀ q reg, regNamed r.ins r.sts q = some reg →
      readsInterior (ownOf h st rec smp r.c₀) (h.names.getD (templateOf h.ms r.u) "") reg.name reg.value = true) :
    ¬ Addresses h :=
  not_addresses hcl hr hg fun S _ reg hreg =>
    readRoots_of_readsInterior h hx hm hd r.ρ (templateOf h.ms r.u) reg.name reg.value (hI (S.port r.u) reg hreg)

/-! ## At an input gate, with no hypothesis on the program -/

/-- **`Addresses` fails at a gate on its replay block's boundary whose rows round 2 reads under the interior roots**:
one accepted run with a checked unit and a gate it needs, on its replay block's boundary, such that every registered
read named like a port whose row holds the gate's bit (`RowAt`) is read by round 2 under the replay block's interior
roots, refutes `Addresses h`. -/
theorem not_addresses_of_boundary (hr : r.Ok h) {g : ℕ} (hg : r.s.prog.Needs r.b g)
    (hB : r.s.prog.Boundary r.rb g = true)
    (hI : ∀ S : Address, r.RowAt h S g → ∀ reg, regNamed r.ins r.sts (S.port r.u) = some reg →
      readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value = irootsOf r.m₁ r.ρ) :
    ¬ Addresses h := by
  intro hA
  obtain ⟨S, hS⟩ := hA g
  have hH := hS r hr hg
  obtain ⟨reg, hreg, -, hne⟩ := addressHolds_reads_of_boundary hr hB hH
  exact hne (hI S hH.2.1 reg hreg)

/-- **`Addresses` fails at an input gate whose rows round 2 reads under the interior roots**: an input gate is on
every set's boundary (`Prog.boundary_of_input`), so the statement's program needs no hypothesis. -/
theorem not_addresses_of_input (hr : r.Ok h) {g : ℕ} (hg : r.s.prog.Needs r.b g)
    (hin : r.s.prog.gates[g]? = some .input)
    (hI : ∀ S : Address, r.RowAt h S g → ∀ reg, regNamed r.ins r.sts (S.port r.u) = some reg →
      readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value = irootsOf r.m₁ r.ρ) :
    ¬ Addresses h :=
  not_addresses_of_boundary hr hg (Prog.boundary_of_input r.rb hin) hI

/-- **`Addresses` fails where a checked unit needs an input gate and round 2 reads its session's registered reads off
the interior record**: one accepted run whose statement decodes, whose round-1 message is a record and whose second
stage's sample evaluates, with a checked unit that needs an input gate of the statement's program and every one of
whose session's registered reads round 2 reads off the interior record (`readsInterior`), refutes `Addresses h`. -/
theorem not_addresses_of_input_readsInterior (hr : r.Ok h) {g : ℕ} (hg : r.s.prog.Needs r.b g)
    (hin : r.s.prog.gates[g]? = some .input) {st : Roots} {rec smp : Json} (hx : decode r.x = some st)
    (hm : jsonMsg r.m₁ = some rec) (hd : secondDraw h r.c₀ r.c₁ = some smp)
    (hI : ∀ q reg, regNamed r.ins r.sts q = some reg →
      readsInterior (ownOf h st rec smp r.c₀) (h.names.getD (templateOf h.ms r.u) "") reg.name reg.value = true) :
    ¬ Addresses h :=
  not_addresses_of_input hr hg hin fun S _ reg hreg =>
    readRoots_of_readsInterior h hx hm hd r.ρ (templateOf h.ms r.u) reg.name reg.value (hI (S.port r.u) reg hreg)

end CheckedRun

end SpotFlock
