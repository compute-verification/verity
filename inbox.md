[2026-11-16T21:47Z] 73d6e90369cc
top, Mon 16 Nov 1:47 PM PST. I read pass/digest.py and pass/certify.py as you synced them at 21:42Z, beside PROSEONLY's finding (prose-not-inert.md, lists in proseonly/) and VLLMSTYLE's final (examples/README.md). certify.py compares what I wanted. The digest does not yet mark the six below; add each before you report, or say in the report that it was already in.

MARK UNTOUCHABLE:
1. Lock reads. _audit_names takes assumptions, domain, open, guarantees, ids, trust_gap, witnesses, unwitnessed, deployed, trust and the signatures, but not a lock's top-level "reads" (module -> definitions). Every definition named there keeps its doc comment byte for byte: tests/test_lean_packages.py::test_every_definition_a_guarantee_reads_says_what_it_means fails when one loses it, and the landing message shows Daniel that text beside the guarantee. PROSEONLY counts 4,129 definitions in 1,687 modules over 5 locks.
2. Stored whole-file digests. Every file in proseonly/stored-digest-sources.txt (106: 100 Lean, 3 lakefile.toml, lean-toolchain, one .sh, one fixture .py) is untouchable whole: tests/test_pins.py and the Lean check's compile-time digest fail on any byte.
3. Definition bodies. A function under @primitive(...) or @composite(...) keeps its docstring byte for byte (inbox 5, item 4); digest.py has no decorator match. Take proseonly/runtime-docstring-files.tsv (673 files) as a second source: every docstring in a file it lists is untouchable, in union with your own detection. Comments in those files may change.
4. The files hashed whole into a provenance digest (ref_prims.py, b1.py, prims.py, and what inbox 5 item 2 named): untouchable whole, by path, since uses_inspect only sees the reader.
5. tests/lint/allowlists/repo_python_verdicts.json pins 34 functions' text, docstring and comments included: every item inside a listed function is untouchable.
6. A Lean file's first /-! -/ is never deleted, only cut to one sentence: tests/lint/test_lean_bar.py's module-doc rule finds a file without one. certify.py refuses a Lean file whose base has a module doc and whose head has none. Later /-! -/ blocks in a file may go.

FOR pass/decider-brief.md, from VLLMSTYLE's eight rewrites:
7. No decision is a script's. Its scripted first cut of queue.py needed a hand fix in 32 of 138 edits and dropped two reasons a head is never run. A sentence that says why the code refuses, skips or never runs something stays.
8. A current limit is stated as a fact ("refuses a plain session past one table"), never "for now" or "until". A constant whose value a measurement chose says the property the value has, not the run that chose it.
9. A stub file (every body raises NotImplementedError) whose docstrings are the design of unbuilt code, such as adapters/vllm/verity_vllm/program/predict/class_window.py (230 prose lines), is cut to the rule like any file, and goes first on its owner's list with the git show BASE:PATH that recovers the text, so the owner can put the design in its issue.
10. Kept text is not restyled: double backticks, wrapping and wording of a kept sentence stay unless the rule needs them changed.

IN YOUR REPORT: say whether shell is in tonight's pass (certify.py refuses any file that is not Python or Lean), and how many items each of 1 to 6 marks on the whole tree.

