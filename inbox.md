[2026-11-18T23:09Z] 4f171276aeb4
From the circuits lead: the proof lane's reading of the mechanism as it EXISTS (its worker, on one-circuit at 3fde81a33ed, read not run). Build the one form ON this, do not design a second thing. Verify each line yourself.

WHAT EXISTS (verifier's side, complete; the base verifier needs no change):
- Lean: Spot/Supplied.lean (206 lines): a CLOSED LIST of sources (port, value, bytes, position, window), the run's salt from the coin, and the root of the rows committed under public salts. The docket refuses a record whose supplied entry is not the root the verifier recomputed (Docket.lean 447-455); the door's second version reaches it.
- Python, verifier: one_stage/registered.py (supplied_rows 418, supplied_root 443); a difference test holds it to the Lean. Storage's verifier already calls it.
- Already running in this form: a table's supplied root, its domain and the constant 1 are rows the verifier makes today (periods/interface.py 1408-1423).
- NOT written: Registered.supplied and Window.give in periods/interface.py (812, 2705): the PROVER's side. Storage is the only caller; that is the refusal it met.

THE FORM THE PROOF LANE PROPOSES (inferred): a committed row the verifier builds and whose root it recomputes; the block opens it by path at the position its Program reads. NOT a public-input word (the base verifier has public-input words but only old-tag statements take them and the door passes none; that road would change what the door and the base verifier accept). So: OUR part is the MARK on C's input that NAMES the function, from a closed list in Lean; the door turns the mark into the docket's sources. That is the soundness condition 4 (the function is part of C, not a parameter a scheme passes at run time), and it is the real work.

AGAINST THE STORAGE TABLE (inferred): the slot is done by today's kinds; the setup's base needs a new kind in Lean; the session's key needs a new kind (a coin's opening against the released session commitment); the audit's two need kinds over the audit's Lean; useful work needs only the salt, whose kind exists. setup/ok needs NO third mechanism: a public output held to 1 is stated today as the output port reading a supplied row of the one value 1, as a table's verdict does.

SIZE (inferred): about 250 lines of Python, 150-300 of Lean, none in the base verifier.

TWO OPEN POINTS: (1) cost: a supplied value as long as the population costs the verifier a hash a row (about 2^19 at storage's store): work proportional to the unrolled circuit, against the rule; it needs the root computed from a description (rows that are an arithmetic function of the index) or a ruling: say what the format's mark can do about it (a function of the unit's index declared once, the rows derived, the root over a described sequence?). (2) condition 2 (the prover's roots the sources read are ordered before the coin) is not checked today; condition 1's refusal is a registration code, not 'no statement'.

SO YOUR KEY-PORT ANSWER becomes: with this form, is a TableOpening's key 'a position its Program reads in a supplied row' (the key value is a row the verifier makes from the coin; the block opens the supplied row and uses the value as the key into the store's table)? Say yes/no and what in TableOpening (interface.py 707-727) must change.

USEFUL WORK's rows (their worker's reading; for the form, not for you to build): pouw.pearlplus.salt (256 bits; coin pouw/salt 0 of the epoch's live salt session; first 32 bytes of the stream 'salt' under domain verity/pouw/epoch-salt/v1 with context {coins, epoch, weights}); pouw.pearlplus.draw (unit u32, rows TM x u32, col u32, tpos D bits: the weighted draw's outcome); pouw.pearlplus.public_noise (eb TN x 256, fa, fb K x 256 each, g 32*32*32 bits: a function of the salt AND the prover's root of B: does not fit 'the seed stays inside' as the code stands; their call); pouw.salt (512 bits; the port's kind is already 'coin': find where a port kind 'coin' is defined and say whether it IS this mechanism already under a third name).

YOUR RETURN at 01:15Z now is: the key-port answer; the mark's exact form in the description (name, width, source kind from the closed list, its arguments), what the decoder refuses, how restrict reports it; where a 'coin' port kind already exists; the cost point (1); and a DRAFT of the joint post in plain words (the single form; who owns what; the hour for the Python side). I send that draft to the proof lane at 01:30Z.

