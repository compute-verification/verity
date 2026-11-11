[2026-11-11T19:49Z] 79b4c1ede33f
From your coordinator (network-accounting), Wed 11 Nov 11:49 AM PST. No reply needed; carry on. Three facts from proofs on NET-297 (11:38 AM PST) that hold for the response tie you are writing. They are proofs' worker's read of main at cc6ee7446278, nothing run. Check each at the lines given before you rely on it.

1. Every proved unit of a session reads a row at every port of the verifier's registered file (Flock/Registered.lean:262-264, :294), and every port of the circuit is in the file (:364-375). So a response tie that reads b.stop in a stop class, and one in a class with no stop block, are two Definitions, two files and two sessions. Do not write one Definition with an optional stop port.

2. Which row a unit reads is the reads array of the verifier's own file. Every writer of it on main computes it from public structure (one_stage/registered.py:382-441, tie_service.py:437-453). That is the design's "block (class, r) reads row r on both sides". The file names a row, never a slot of a row, so a tie reads each certifier row whole.

3. complete is computed, not written: every entry of the stop record's verdicts has a verdict (Spot/Record.lean:380, :431; one_stage/audit.py:147, :155), and a sample gets one outcome (service.py:2257-2262). So a stop record's outcome is written after whatever turns its last entry, inside the proof deadline, or complete stays false for that window for good.

Also from proofs: Flock holds a port's root to nothing outside its file. Holding a root to another stream's record is the service's Python, as the ingress tie's verify does (tie_service.py:552-561).

Still open, and proofs answers on NET-297 by 2 PM PST today: where the two ties sit. If your work needs that answer before 2 PM, say so in your 5 PM report and go on with the parts that do not.

