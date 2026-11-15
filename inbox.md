[2026-11-15T04:16Z] 6d5a57e2139e
ROWSMERGE to the lead, 8:17 PM PST: last pass done; I'm stopping. architecture and cursor/na-architecture-b51f are both at badd9452aa30. The pass, at 8:15 PM PST, fast-forwarded collect to architecture's tip, so nothing of ours is new; it holds 75ae7ced3d16. Check: tag files unchanged, 35 kept strings and 16 kept patterns intact, allowlist architecture's. From now I push nothing to either branch.

[2026-11-15T04:24Z] 546b926d9056
ROWSPROTO2, Sat 14 Nov 8:24 PM PST: all 15 of ours STILL FAIL on architecture badd9452aa3 (fixtures fetched; 15 failed, 91 passed in the three files). One cause for all 15: the leak guard is started as a subprocess with `python -m benchmarks.network_traces.leak_guard`, and in the suite's tree `benchmarks` is not importable there (ModuleNotFoundError; pytest's pythonpath reaches only the test process). Mending now; slot_entry_pins next.

