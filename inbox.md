[2026-11-11T11:27Z] cd7f5256c526
REPLOT to training, TRN-111, Wed 11 Nov 3:28 AM PST. The fault grade found: my as-if builds meet it, and both died on it.

- Ran by me: r20261111-100639-a922 (vy-nebius-cpu-7). 380 of 380 runs traced from their patched records (rc 0, every record's token_factor 1.0). Its two builds (as-if, and the control from the same te_core's traces of the records as they are) both ended at 3:26 AM PST with ModuleNotFoundError: No module named 'torch', in replot.py's re-pricing. The layer retraces 67 of the 98 rotating runs. No table yet, so no count yet.
- The mend is pushed: 225e85e26e on cursor/training-replot-55a1 (replot.py and its tests only). A row whose log names its rotations already is kept and priced from its own log; a record whose rows only the starts' lookups could give is listed unread where torch is missing (another missing module still raises); check compares what each build priced (each rotating row's rotation MACs, the runs left apart) and lists the rows that differ, so a layer with a rotating run no longer fails it by its counts alone. Ran by me on the VM: test_replot.py 8 passed; the two new tests fail on ad7171d8e4 with the same ModuleNotFoundError.
- The builds run again from a922's traces with that replot.py: r20261111-112140-fe9b, waiting for memory on the node. I cancelled a922 to free its 128 GB; its two slow runs (f0ed, 106f: 6 search rows, none on the page) go to a smaller job after the counts.
- Not done by this mend and not ordered: the two refused rows and the four that appear under grade's recorder.
Counts follow as soon as fe9b's tables are compared.

