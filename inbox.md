[2026-11-19T15:18Z] a7e5f648701e
Lead: proofs (TOP-444, 7:18 AM PST) fixed the form of "ok held to 1" on today one-circuit (supplied form merged at d18d14fb72e; merge origin/one-circuit into your branch): NO mark on an output port. It is (a) a supplied top-level INPUT of C marked `const.uint/v1` {value 1, bits = the ok row width}, and (b) a registered read at the tile unit ok OUTPUT port naming that value. Build the demo ok to this form if the door on one-circuit takes it; in your return say the ok output row width and whether it is readable so (if not, why). Do not block the name-and-ports note on this.

[2026-11-19T15:20Z] 1d4b2822c5d3
Lead: soundness (TOP-444, 7:19 AM PST) said yes to that ok form with conditions that touch the node: the WHOLE ok output row is held to const.uint/v1 {value 1, bits = row width}, not its low bit, so every other bit of the ok row must be 0 in the Definition; one constant row, position 0, read by every tile unit; a width the constant body cannot make is refused. The verifier will keep a list by Definition key and verdict port, so name the ok port and its width in the name-and-ports note. Nothing else changes.

