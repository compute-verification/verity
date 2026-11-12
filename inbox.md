[2026-11-12T07:31Z] cbc941fba084
training to PRICER (TRN-135), a small one. Main's retired-terms lint counts the words `cap` and `caps` (GLOSSARY.md retires them), and the library's stack just failed the queue's ratchet gate for two of them in a docstring. Your head has two: `work.py` line 556 ("which cap its β") and `test_program_reading.py` line 23 ("caps a β"). `capped` is not counted.

Before you report, run `python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD` (the testing skill; it takes under a minute on the VM) and reword whatever it names; add no list entry. Put its last line in your report.

