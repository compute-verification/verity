module

public import Proofs.Flock.SpotFlock.Code

/-!
# Every wired source bit has a gate to go to, decided once per template

`wiresMove` (`Code`) passes over every source bit of the members' programs: each one at a port a wire names and no
link names has a gate to go to (`wireTarget`). What it asks of a source bit reads only the bit's template, its row,
its bit of the row, and where the bit's unit sits in its replay block's cell. `wiresMoveRows` decides the same once
per template:

* for each row a unit of the template reads source bits at (`sourceRows`) that a wire names and no link names
  (`rowGoes`): the row has rows, the wire's writer names a template, no link names the writer's port, the reader's
  units are whole cells and the writer has as many cells of units, the rows' runs hold the reader's cell, and each
  run's last row is one of the rows a replay block holds, which are no more than the writer's cell (`cellGoes`). So
  every unit of the reader reads the row at a unit of the writer;
* for each bit one unit of the template reads there (`sourceCells`), the writer's unit writes that bit of its
  output row (`bitGoes`, `outBit`).

`Proofs.Flock.SpotFlock.WiresMoveRows` proves that it gives `wiresMove` where the design evaluates.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **The row and the bit of it that each input column of one unit of circuit `c` reads**, where its source is a
message bit (`Bits.srcKey`, `Bits.srcPos`): `sourceRows`, with each one's bit. A source gate's commitment position
(`Bits.rowBit`) is its column's source at its unit, so its row and bit are one of these whatever the unit. -/
def sourceCells (c : _root_.Flock.Circuit) : List (String × ℕ) :=
  let ix := Spot.Bits.Index.of c
  (List.range ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn).filterMap fun b =>
    (Spot.Bits.srcPos c 0 (Spot.Bits.srcKey c ix b)).map fun x => (x.row, x.bit)

/-- **Bit `bit` of the row named `wport` is one a unit of circuit `c` writes**, decided: `wport` names one of the
circuit's output row ports and the bit is inside that output port's words, sixteen bits a word. An output row's port
is padded to whole hash blocks, so the port's own `words` would say more than the unit writes. -/
def outBit (c : _root_.Flock.Circuit) (wport : String) (bit : ℕ) : Bool :=
  (List.range c.outPorts.size).any fun o =>
    (c.ports.getD (_root_.Flock.HmOut.nIn c + o) default).name == wport &&
      decide (bit < 16 * (c.outPorts.getD o default).2)

/-- **Wire `w`'s writer writes bit `bit` of its output row**, decided: the wire's writer names a template
(`heldWriter`) whose unit writes that bit of the wire's writer port (`outBit`). -/
def wireBit (w : Spot.SecondStage.Wire) (bit : ℕ) : Bool :=
  match heldWriter h w.writer with
  | some tw => outBit (memberCircuit h tw) w.wport bit
  | none => false

/-- **Bit `bit` of template `t`'s row `row` has a writer's bit to go to**, decided: where a wire names the row and no
link does, the wire's writer writes that bit (`wireBit`); true at every other row. -/
def bitGoes (t : ℕ) (row : String) (bit : ℕ) : Bool :=
  match heldLink h (h.names.getD t "") row, heldWire h (h.names.getD t "") row with
  | none, some w => wireBit h w bit
  | _, _ => true

/-- **Every unit of template `t` has a unit of template `tw` to read wire `w` at**, at rows `r`, decided: no link
names the writer's port; the reader's cell is not empty and its units are whole cells; the writer has as many cells
of units; the rows a replay block holds are no more than the writer's cell; the rows' runs hold the reader's cell,
each position once (`SecondStage.fits`); and each run's last row is one of the rows a replay block holds. -/
def cellGoes (t tw : ℕ) (w : Spot.SecondStage.Wire) (r : Spot.SecondStage.Rows) : Bool :=
  (heldLink h (h.names.getD tw "") w.wport).isNone && decide (0 < h.reg.ns.getD t 0) &&
    decide ((h.vus.getD t #[]).size % h.reg.ns.getD t 0 = 0) &&
    decide ((h.vus.getD t #[]).size / h.reg.ns.getD t 0 * h.reg.ns.getD tw 0 ≤ (h.vus.getD tw #[]).size) &&
    decide (r.per ≤ h.reg.ns.getD tw 0) && Spot.SecondStage.fits r.runs (h.reg.ns.getD t 0) &&
    r.runs.all fun x => decide (x.row + (x.count - 1) * x.step < r.per)

/-- **Every unit of template `t` has a writer's unit to read wire `w` at**, at its row `row`, decided: the row has
rows (`heldRows`), the wire's writer names a template (`heldWriter`), and each unit of `t` has a unit of it
(`cellGoes`). -/
def wireGoes (t : ℕ) (row : String) (w : Spot.SecondStage.Wire) : Bool :=
  match heldRows h (h.names.getD t "") row, heldWriter h w.writer with
  | some r, some tw => cellGoes h t tw w r
  | _, _ => false

/-- **Every unit of template `t` has a writer's unit to read its row `row` at**, decided: where a wire names the row
and no link does, `wireGoes`; true at every other row. -/
def rowGoes (t : ℕ) (row : String) : Bool :=
  match heldLink h (h.names.getD t "") row, heldWire h (h.names.getD t "") row with
  | none, some w => wireGoes h t row w
  | _, _ => true

/-- **Every wired source bit has a gate to go to, decided once per template** (`wiresMove`, which passes over every
source bit of the members' programs): for each template, each row its units read source bits at has a writer's unit
for each of its units (`rowGoes`), and each bit one unit reads there is one the writer's unit writes (`bitGoes`). It
reads the circuit files and the registered wiring, whatever the count of units. -/
def wiresMoveRows : Bool :=
  (List.range h.ms.length).all fun t =>
    ((sourceRows (memberCircuit h t)).all fun row => rowGoes h t row) &&
      (sourceCells (memberCircuit h t)).all fun rb => bitGoes h t rb.1 rb.2

end SpotFlock
