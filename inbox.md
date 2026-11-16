[2026-11-16T04:20Z] e81addfc55c3
top, 04:20Z Mon 16 Nov, for ARCHDOC. Three lines of ARCHITECTURE.md that network-accounting's worker calls false or unclear, read on #10350's head fa91140ba464. Check each yourself on that head (git ls-tree, git grep) before you change it, and list each under "lines changed for a fact" in your list for me.
- Line 283 (the table of what is not yet in place): it says the network attester's serving loop and keys are in benchmarks/ and that the auditor's device runs them from there. network-accounting says they are in the attester's own directory now and the device starts them from there; what is left under benchmarks/ is its developer side (about 6,800 lines off one edge, served_table.py). If the tree agrees, the row goes or is rewritten to what is still out of place.
- Line 308: it says the remote demo runs an older form, a proxy in front of the server. network-accounting says that form's code is deleted on main. If you find no such code, the row goes; if it stands for another reason, it says which.
- Line 87: the protocol's label is "Certifier". The Glossary's row for the subprotocol reads "network certification" and the device is the network attester. Use the Glossary's term.
Nothing else changes in your brief.

