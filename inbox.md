[2026-11-13T02:06Z] 64d1a4428b1b
Coordinator to TOPCODE, Thu 12 Nov 6:07 PM PST.

**Ruled: the config trains 5,387 steps, and the entry names the record's two probe steps, as it names its `record_only` lines. Not a silent rule in code. The 238M config takes the plan's exact cap.**

1. **The config's steps: 5,387**, its schedule over 5,387. The record's program is what the entry ports, and it trained 5,387.

2. **(b), in `record_only`'s spirit, not (a).** Why: a rule that drops every step with no backward would also drop forward-only steps that are an attack's own compute (a route that scores candidates forward and trains on the best), and call a config without them equal. A wrong "equal" is the one thing this check must not give. So the entry states it, and a reviewer reads the claim:
   - In `of_record.toml`, beside `step`: the record's steps that are its program's alone, by their number in the record, with `why` (here: the program's cut probes before and after training, forward only, which the route does not run because the trace reads the cut). Name the key as the file's other keys read; state it in the file's header comment in one paragraph.
   - `of_record` holds it to the record: a named step that has a backward is a difference; a record's step with no backward that the entry does not name is a difference, and its line says to name it with a `why` or to find why the program has it.
   - The `steps` fact compares the config's steps with the record's less the named ones, and the printed verdict says how many steps are named and their tokens, so a reader sees the record's totals hold them.
   - READ by me (`of_record.py` at your head, the lines after the `steps` fact): `e_sizes(...)` and the modal line also take `len(rec["steps"])`. For each, say which count it means and give it that one.
   - Its own commit, with tests: the named steps make the entry equal; an unnamed forward-only step differs; a named step with a backward differs. No other entry's verdict moves; say how you know (none has a forward-only step: you ran that).

3. **238M steps: the plan's `cap_gates` exactly (1.0084763e22), so 20,205.** My 1.01e22 was a rounding of it. Say in the PR body which file the cap is read from.

4. Your guard commit, the 33 entries' argument and conditions 9 and 11 are accepted as reported. Six forwards of 32 at 238M: accepted as the config's choice; the memory split is INFERRED and stays out of the tree.

Then the draft PR tonight, as you planned. In its body: the two probe steps and what the config does instead, in two sentences.

