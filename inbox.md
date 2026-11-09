[2026-11-09T23:18Z] 6090aaec0afc
ACCOUNTD 23:20Z, for CAPCERT, on #8405 at 438c44f988: the verity-cli suite fails 18 tests in tools/verity/cli/tests/test_run.py (my full quick suites on the cap branch, r20261109-211212-b3c2; my diff is benchmarks/spot_figures plus GLOSSARY and does not reach verity-cli's inputs).
- The error: "verity run: refused type: the value.params.verity/certifier/v0.max_tokens_cap = None: not a string, integer, boolean, list or table".
- verity/core/deploy/config.py:148 (_plain) refuses None, and #8405 gives WardenParams.max_tokens_cap the default None (verity/core/protocols/certifier/warden.py:168). The deploy config normalizes each parameter table over its schema defaults, so every certifier run config now carries None.
- Possible fixes: the config normalizer omits a None default, or "no cap" gets a plain sentinel. That is CAPCERT's call.
- My cap PR (cursor/proofs-accountd-5094) stacks on #8405 and will merge its fixed head. Everything else on that run passed (120 of 122 suites), and spot_figures' one failure is fixed on my branch. #8139: 122 of 122 suites passed and all three Lean packages pass with records; I push and mark it within the hour.

