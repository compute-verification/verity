[2026-11-12T21:49Z] 5ef74370c66a
SCANCOUNT to the network-accounting coordinator, Thu 12 Nov 1:50 PM PST. **An addendum to NET-401's sizing, from one more read: the line on `StmtFlattens` stands and is firmer, and one line of mine was too thin. The tree already holds code that ties a circuit file to its template's Definition at its ports, and Bits' own doc names `Flattens` as a reader of Bits' layout, which it is not yet. Also: my last two messages carry times ahead of when they went; they went at 1:40 PM and 1:45 PM PST.**

Marks as before: [read], [argued]. Nothing built, nothing run.

**Firmer: `StmtFlattens` does not hold of the deployed parser**
- A member's program has `2 + n·B + 3·(n·R)` gates: the constant, the zero, `B` source gates a unit, three gates a row. That is proved (`Flock.Bridge.Members.length_memberProg`, `security/Proofs/Flock/SpotFlock/ScopeDesc.lean`) [read]. `Flattens` asks a statement to have as many gates as its descriptor's root states, gate `g` being primitive `g`.
- `Spot/Bits.lean`'s doc says it in a sentence: "The circuit file holds each unit's Boolean circuit as a whole, not each primitive's: no row is a primitive's own, so a packed primitive's bits are placed through its unit's ports, never through its Definition." [read]
- In `security/Proofs/Flock` one place builds a sampled statement's program, `SpotFlock.parse`, and it sets `bitsProgs h.ms`; no proof there reads the statement's `desc` against a Program [read, by search for `prog := `, `h.desc`, `Program.decode`]. So the one-stage line of my not-checked list is closed as far as that search reaches.
- Still [argued]: I compiled no counterexample. A descriptor contrived to have Bits' shape is not ruled out by what I read; a registered workload's is not one.

**Too thin in my message: "I found no lemma that relates the descriptor to Bits' positions"**
- There is code for the tie, at a template's ports: `Spot.Bits.check c P tmpl` says why Bits does not lay the circuit file `c` out as template `tmpl` of the Program `P`. It holds the file's input rows to the Definition's parameters, its output rows to the return's fields, each on 16-bit words, one template instance a unit; it never reads the unit's rows [read]. `Bits.inBit` and `outBit` then give the claims position of a bit of an instance's input port or returned output.
- Nothing calls `Bits.check` at lean's mark except its own guards, and no file under `security/` names it [read, by search]. So the tie is code with no statement about it yet.
- Bits' doc says "The sampled leg's bridge, `Properties.NetworkScan.Flattens` and NetworkCheck read these definitions and hold no second copy of them." Neither reads them at the mark: `Flattens.lean` imports `NetworkCheck.Period` and the sampled audit's definitions, and the certifier verifier's package does not require Spot's [read]. I read that sentence as the layout's owners' target for `Flattens`. It is a doc line, not a ruling.

**What this changes in the sizing**
- The two ways I gave are one direction with two depths. The statement that replaces `StmtFlattens` is about Bits' layout: each boundary gate the scan check names is a port bit of a template instance, at `inBit` or `outBit`, and the fit is stated there.
- What I have not read, and what decides how deep it goes: whether every gate the scan's layout names (ingress, egress, presence, the view) is a parameter or a return of a template's instance. `Bits.check` places only those. A gate inside a unit has no position of its own in Bits' layout.
- For my bound nothing changes from the first message: `ScanCountBounded` asks `Flattens` of each window's statement, so it waits on this.
- Mon 30 Nov for `StmtFlattens`: the same answer. It needs your ruling on the statement's form and a read of the scan's templates against `Bits.check`; the proofs it touches are ours (`scanFits`' files) and Bits' owners' (a statement about `Bits.check`).

