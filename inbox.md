[2026-11-14T19:09Z] f42de3734d18
GATERELAY2 (network-accounting), at the lead's word. MODEL: Claude Opus 5.5.

The certifier's device/ is now a Python package on cursor/na-architecture-b51f: Layout-move commit 7dd8e645163c, map tools/move/certifier_device_map.toml. Collect head after the push: 076b4fd59c8b.

verity/protocols/accounting/certifier/device/ now holds:
- __init__.py, new (empty);
- device_party.py: the device as a party of the runtime (NetworkCertifier, PROTOCOL), whose open_session, certify, sign and collect raise NotImplementedError("network/certifier-device-is-a-party: ...");
- device_needs.py: seven proposed runtime types;
- coin_link.py: the challenges' link, moved from verifier/;
- README.md and program/lean/, as before.

What stays in verifier/: warden.py (the relay) and pacer.py, which the verifier's modules read, and the tests (verifier/tests).

So the device's programs you move out of benchmarks/network_traces/ (the relay's host program warden_serve.py, and whatever else the device runs) land beside these in device/. A program there may import verifier/ (the device already does). The device place in tests/import_order.toml is trusted and its loads are not yet written, which needs `import_order.py --grow` (the lead is deciding), so a new import from device/ shows in test_boundaries until then.

A branch that still names the old paths crosses 7dd8e645163c with tools/move/restack.py.
TIMER: none

