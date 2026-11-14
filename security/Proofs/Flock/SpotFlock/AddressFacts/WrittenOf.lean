module

public import Proofs.Flock.SpotFlock.AddressFacts.BitsOwn
public import Proofs.Flock.Soundness.Discharge.Hidden.Commit

/-!
# Each written bit inside its own output row

`writtenOwn_of_two`: `WrittenOwn` where the second stage's design evaluates (`two`), so `BitsOwn` and, with each member
naming its ports once, `BitsInPorts` rest on the design alone (`bitsInPorts_of_once`).

A written bit's commitment position is Bits' (`Spot.Bits.outPos`): bit `16 (x − lo) + t` of the output row `o` whose
run of output ports holds output word `x`, `lo` the run's first word and `t < 16`, the row port `nIn + o`. The design
evaluates only where every template's reference statement parses (`templateOk`), and the hidden-output parse ends in
`HmOut.checkRows`, flat or templated (`Hidden.Committed`): output row `o` is a port of the circuit, of its run's words
zero-padded to whole SHA-512 blocks, or of `bits` exactly `16` times its run's words (`Hidden.RowsOut`). With every port
of `16 · words` bits (`bits_of_two`), the row holds at least its run's words either way. `RowsOut` reads no net, so it
holds of the member's circuit, the parsed one unrolled (`Readable.Read.stmt`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-- **A written bit is inside its output row's port**, where the output rows pass `HmOut.checkRows` and every port is
a row of `16 · words` bits. -/
theorem outPos_own {c : _root_.Flock.Circuit} (hR : FlockSoundness.Discharge.Hidden.RowsOut c)
    (h16 : ∀ p, p < c.ports.size → c.ports[p]!.bits = 16 * c.ports[p]!.words) {u x b : ℕ} {r : Spot.Bits.RowPos}
    (hb : b < 16) (ho : Spot.Bits.outPos c u x b = some r) :
    ∃ q, q < c.ports.size ∧ c.ports[q]!.name = r.row ∧ r.bit < 16 * c.ports[q]!.words := by
  unfold Spot.Bits.outPos at ho
  dsimp only at ho
  obtain ⟨o, hf, rfl⟩ := Option.map_eq_some_iff.1 ho
  have hmem := List.mem_range.1 (List.mem_of_find?_eq_some hf)
  have hp := List.find?_some hf
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
  obtain ⟨hq, hrow⟩ := hR o hmem
  have hg : c.ports.getD (_root_.Flock.HmOut.nIn c + o) default = c.ports[_root_.Flock.HmOut.nIn c + o]! := by
    rw [getElem!_pos c.ports _ hq]
    simp [Array.getD, hq]
  rw [hg] at hrow
  have h1 := h16 _ hq
  refine ⟨_root_.Flock.HmOut.nIn c + o, hq, (congrArg _root_.Flock.Port.name hg).symm, ?_⟩
  show 16 * (x - (_root_.Flock.HmOut.outGroup c o).2.2.1) + b < 16 * c.ports[_root_.Flock.HmOut.nIn c + o]!.words
  rcases hrow with ⟨-, hw, -, -⟩ | ⟨-, hbits⟩
  · omega
  · omega

/-- **`RowsOut` of a circuit is `RowsOut` of it unrolled**: unrolling changes its nets and template only. -/
theorem rowsOut_unrolled {c : _root_.Flock.Circuit} (hR : FlockSoundness.Discharge.Hidden.RowsOut c) :
    FlockSoundness.Discharge.Hidden.RowsOut (_root_.Flock.HmOut.unrolled c) := by
  unfold _root_.Flock.HmOut.unrolled
  split
  · split
    · exact fun j hj => hR j hj
    · exact fun j hj => hR j hj
  · exact fun j hj => hR j hj

variable {h : Held}

/-- **A written bit is a member's returned output bit with Bits' commitment position.** -/
theorem outRowBit_of_mem {x t : ℕ} {r : Spot.Bits.RowPos} (hy : (x, t, r) ∈ writtenBits h.ms) :
    ∃ ht : t < h.ms.length, ∃ u k j b, b < 16 ∧ Spot.Bits.outRowBit h.ms[t].1 u k j b = some r := by
  unfold writtenBits at hy
  obtain ⟨m, hm, hy⟩ := List.mem_flatMap.1 hy
  have hm' := List.mem_range.1 hm
  have hci : h.ms.getD m default = h.ms[m] := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hm', Option.getD_some]
  dsimp only at hy
  simp only [hci] at hy
  obtain ⟨u, -, hy⟩ := List.mem_flatMap.1 hy
  obtain ⟨k, -, hy⟩ := List.mem_flatMap.1 hy
  obtain ⟨j, -, hy⟩ := List.mem_flatMap.1 hy
  obtain ⟨b, hb, hy⟩ := List.mem_filterMap.1 hy
  have hb' := List.mem_range.1 hb
  split at hy
  · simp only [Option.some.injEq, Prod.mk.injEq] at hy
    obtain ⟨-, rfl, rfl⟩ := hy
    exact ⟨hm', u, k, j, b, hb', by assumption⟩
  · cases hy

/-- **Where the design evaluates, each member's output rows pass `HmOut.checkRows`'s row check.** -/
theorem rowsOut_member {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) {t : ℕ} (ht : t < h.ms.length) :
    FlockSoundness.Discharge.Hidden.RowsOut (memberCircuit h t) := by
  have hok := templateOk_of_two h hd ht
  unfold templateOk at hok
  cases hr : FlockDeploy.readOf true (h.xS₀ t) with
  | none => rw [hr] at hok; cases hok
  | some r =>
    rw [hr] at hok
    simp only [Bool.and_eq_true] at hok
    obtain ⟨⟨⟨⟨⟨hho, hty⟩, hps⟩, -⟩, -⟩, -⟩ := hok
    have hty' : r.tags.typed = false := by simpa using hty
    have hm : memberCircuit h t = r.stmt := by simp [memberCircuit, stmtOf, hr]
    rw [hm]
    obtain ⟨c, hc⟩ : ∃ c, _root_.Flock.HmOut.parseCommit r.tags r.I.circuitFile r.I.tables = .ok c := by
      cases hD : _root_.Flock.Desc.parseStmt r.tags r.I.circuitFile r.I.tables with
      | error e =>
        rw [hD] at hps
        cases hps
      | ok D =>
        have h1 := _root_.Flock.Desc.parseStmt_stmt hD
        unfold _root_.Flock.HmOut.parseStmt at h1
        unfold _root_.Flock.HmOut.parseCommit
        cases hi : r.tags.isTmpl <;> simp only [hi, Bool.false_eq_true, ↓reduceIte] at h1 ⊢
        · exact ⟨_, h1⟩
        · cases hp : _root_.Flock.HmOut.parseTmpl r.tags r.I.circuitFile r.I.tables with
          | error e =>
            rw [hp] at h1
            cases h1
          | ok c => exact ⟨c, rfl⟩
    have hcirc : r.circ = c := by
      simp only [_root_.Flock.Readable.Read.circ, _root_.Flock.ProvedScope.circuitOf, hho, ↓reduceIte, hc]
    show FlockSoundness.Discharge.Hidden.RowsOut (_root_.Flock.HmOut.unrolled r.circ)
    rw [hcirc]
    apply rowsOut_unrolled
    rcases FlockSoundness.Discharge.Hidden.committed_of_commit hc with hC | hC
    · exact FlockSoundness.Discharge.Hidden.rowsOut_of_checkRows
        (FlockSoundness.Discharge.Hidden.parse_untyped hty' hC).2
    · obtain ⟨tu, c₀, -, hck, -, rfl⟩ := FlockSoundness.TmplValue.parseTmpl_cases hC
      exact fun j hj => FlockSoundness.Discharge.Hidden.rowsOut_of_checkRows hck j hj

/-- **`WrittenOwn` where the design evaluates.** -/
theorem writtenOwn_of_two {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) : WrittenOwn h := by
  rintro ⟨x, t, r⟩ hy
  obtain ⟨ht, u, k, j, b, hb, ho⟩ := outRowBit_of_mem hy
  have hm : memberCircuit h t = h.ms[t].1 := (by simp [Held.ms] : h.ms[t].1 = stmtOf (h.xS₀ t)).symm
  unfold Spot.Bits.outRowBit at ho
  obtain ⟨x', -, hx⟩ := Option.bind_eq_some_iff.1 ho
  rw [← hm] at hx
  exact outPos_own (rowsOut_member hd ht) (fun p hp => bits_of_two hd ht hp) hb hx

/-- **`BitsInPorts` where the design evaluates and each member names its ports once.** -/
theorem bitsInPorts_of_once {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) (hO : PortsOnce h) :
    BitsInPorts h :=
  bitsInPorts_of_two hd hO (writtenOwn_of_two hd)

end SpotFlock
