[2026-11-18T20:22Z] 401a9ce96227
From the proofs lead, 12:28 PM PST, a small one from lean's worker CUT on your Circuits/Lanes.lean, to do in passing on one-circuit: partLeafAt should read the decoder's Lane.leafAt (Program.lean, the arithmetic Part.get's grid case uses, kept as an Int as you need), and laneDigits should go unless laneWire still needs the digits themselves; the decoder keeps its one copy. Note lean's v2 decoder is merging into one-circuit this hour and changes Program.lean: merge first, then make this change, and try the file.

