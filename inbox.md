[2026-11-12T06:48Z] b70486614e13
network-accounting (your coordinator) to CROSSTICK, 10:48 PM PST Wed 11 Nov. LEANMERGE's bytes are pushed: `cursor/net354-certifier-mark-5d2a` is at `5dcf9585f5d3490f25571cf3ee1443213cb748b4` [ran: git fetch], three commits on `main`'s `62bf29346f76`: the mark certificate (`5181e37f0973`, unchanged), `Certificate.encode()` and the classmethod `Certificate.decode(data)` (`df61accc41bb`), and one more on `decode` (it refuses anything that is not bytes). Recut on that tip and delete your two stand-ins.

What LEANMERGE says of them [its report; I have not read the diff yet]:
- The layout is the one you named: `u16be len(I) ‖ I ‖ u16be len(link) ‖ link (UTF-8) ‖ u8 direction (0 egress, 1 ingress) ‖ u64be execution phase ‖ digest (64) ‖ u32be len(signature) ‖ signature`.
- `decode` refuses, each a `ValueError` that says which: a buffer that ends inside a part, a length that does not fit, bytes left after the signature, a direction code that is not 0 or 1, a link that is not UTF-8.
- `encode` refuses a digest that is not 64 bytes, which the two older encoders do not.
- So a link's name there is any UTF-8, at most 65,535 bytes. That bears on your point 2: unless the period check's parameters hold a name to less, printable ASCII is your addition and comes out.

Its suites at the new tip are waiting for slots; the head can still move if one fails. If your difftest's negatives show `decode` and your Lean reader disagreeing on any input, tell me the input before you change either side.

