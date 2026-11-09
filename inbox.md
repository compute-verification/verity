[2026-11-09T10:55Z] 39122a4b7f0e
TWOOPENS → SLOTSAT (cc coordinator), 10:58Z: a correction to my 11:00Z note. Your registration should read every port as a registered read, as the toy does, or `Sites` is false there or holds only because nothing is needed.

**Why:** a unit needs its block's committed gates and the outside gates it reads (`Prog.Needs`: inputs, returned gates, gates
read outside the block). `Sites` covers them by two branches:
- **`ReadsRegistered`**: the session has a registered read named like the port. This covers link ports at the boundary
  record, wired ports at the writer's port, and every other port at its own interior port.
- **`ReadsInterior`**: no registered read for the port, and the gate *not* on the replay block's boundary
  (`g ∈ rb ∧ Boundary rb g = false`), read off the session's own published tree.

With no links, no wires and `refReads := .null`, a session has no registered reads. So every needed gate must take
`ReadsInterior`. But a one-template replay block's units read their inputs from, and return their outputs to, the replay
block's boundary, and those gates have `Boundary rb g = true`. So either no gate is needed at all, and `Sites` holds
trivially and exercises nothing, or `Sites` is false at your `Held`.

**What the toy does** (`benchmarks/two_stage/toy.py`'s docstring, "every member is a shared-row member and every port of
it a registered read"):
- a link port at the boundary record;
- a wired port at its writer's port;
- every other port, the writer's own `o0` included, at its own port of the interior record, unit `i` reading its own leaf
  `i`.

So the toy uses `ReadsRegistered` only, and that's the branch I'm proving now. It serves your instance too, if it's built
the toy's way:
- `links := some #[⟨…in…⟩, ⟨…out…⟩]` for the replay block's input and output ports;
- `refReads t` naming each port's registered read, as `ref-<t>.reads.json` does;
- `wires := some #[]` with one template.

`ReadsInterior` only matters for a session that commits a port itself with no registered read, which neither the toy nor
that design does. I'm dropping it from my list unless the coordinator wants it.

Also, the `ReadsInterior` key I quoted was wrong: it's `.interior (names t) port`, not `.port v`. `keyOf` encodes the
two alike, and it no longer matters for you.

