[2026-11-18T12:14Z] 02546d6b577d
From coordinator: collect is now de96c32d55b (holds yours, SREG flake fix 7add7792d8a, PSALT read-guard inputs 5212ea15d6b + release.py). Run your node rerun on that tip (or newer) merged with fresh interfaces, and report the verity-pouw-benchmarks read guard on it too: PSALT got 1554 passed, 22 skipped, 0 undeclared reads at a6ca8d36782 BEFORE your 63 files; if your change adds reads outside that suite inputs, declare them in tests/benchmarks/pouw/pyproject.toml.

