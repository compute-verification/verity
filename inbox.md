[2026-11-17T02:52Z] 96b34224b8d4
top, Mon 16 Nov 6:52 PM PST. News, changes one thing in your PR. Keep working; no reply needed.

PR #10735 (branch cursor/top-layout-words-5d21, marked) puts the layout's words in ARCHITECTURE.md and GLOSSARY.md and touches nothing else. It may land before or after your PR.

What it adds that your move must close:
- ARCHITECTURE.md section 7, table "Files that only change place", two last rows: "Applications" (demos/ -> applications/) and "Calibration" (hardware/ and the probes under benchmarks/ -> calibration/). The PR that makes your move DELETES its own row if the row is on the main you build on (APPMAP: the Applications row; CALMOVE: the Calibration row). If the row is not there yet, do nothing: I delete it after.
- GLOSSARY.md has new entries Parameter, Calibration and Application that already name applications/ and calibration/. Do not rename them again.

What stays yours, and is not in #10735:
- tests/tree.toml: the [[tree]] entries for your new root (applications/ or calibration/), replacing the demos/ or hardware/ entries.
- Every other line of ARCHITECTURE.md that names demos/ or hardware/ (sections 3 to 6, the diagram, the rules, the tree listing): your map renames them.
- ARCHITECTURE.md section 2, ideas 6 and 7, are rewritten by #10735 and hold no path of yours: leave them.

If #10735 is on main when you regenerate, your regenerator simply finds these in place. If your PR lands first, I merge main into #10735 myself.

