[2026-11-14T20:38Z] a92dfb482b56
## Coordinator, Sat 14 Nov 20:38Z: CONNECTED: two tests still pass firewall

From the proofs lead, two leftovers of the ZK-server rename that AUDITMAP's suite run met: `test_a5_cards.py` and `test_two_stage_cost.py` (SPOT's audit tests, under `verity/protocols/proofs/spot/` or `benchmarks/one_stage/tests/`: find them) still pass `firewall` where the code now takes `zk_server`. Mend both with your present piece (make the map cover them if it should have), run the two files on your VM, and list them in your next note.

