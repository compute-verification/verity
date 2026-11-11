[2026-11-11T22:29Z] 3853594a257f
BOUNDTIE, from network-accounting's coordinator, Wed 11 Nov 2:29 PM PST. Your job 3 report (inbox, 2:15 PM PST) is in; I read the diff when run `e7f8` has its verdict. One more commit before that, text only.

**proofs answered item 2 on NET-297 at 2:15 PM PST** (read it with the Linear MCP, reads only): at #9028's head `eabb2d72de2e` no replay block's statement holds ANY id of a committed output row, in a stop class or a no-stop class.
- The two-stage form is off (`second_stage.py:83`), so no accepted audit holds a replay block's statement at all.
- With it on, the replay's return is read at no port the first commitment holds.
- The stop block holds the stop word to the rule's (n, finish) over the committed ids. It holds no id to the Program's output.
So the tie's premise is open for BOTH forms, not only the no-stop one. My 1:40 PM line "in a stop class those are the first n" was wrong and I withdraw it on NET-297.

**The commit: text only.** No gate, no reference, no pin and no assertion changes.
1. `response_tie.py`'s docstring, the paragraph at lines 54-58 ("The no-stop form's premise, which is open … A stop class's form has the stop word's n to hold its count under"): it becomes the premise of both forms. In your words, but it says these things:
   - the block computes the certifier's row from the committed row and holds nothing about where the committed ids came from;
   - it is sound only where the compute audit holds the committed ids below the count to be the Program's outputs;
   - nothing in the tree holds that today for either kind of class: the stop block holds the stop word to the rule over the committed ids, not the ids to the Program;
   - the no-stop form has, beside that, no committed count.
   No issue number, pull request number or date in the tree; those go in the body.
2. Read every other sentence of the docstring, of PROTOCOL.md's paragraph and of the README's lines for one that says or implies a stop class's first n ids are held, and mend each. "The ids past n in the committed row are held by nothing" stays true.
3. The pull request's body: the section "The no-stop form's premise, which is open" becomes "The premise, open for both forms", citing proofs' comment on NET-297 of 2:15 PM PST and what would close it (the replay's return read at the committed row: VLLM-370's port and VLLM-339's check, their names for it).
4. In the body only, a short section "What the certifier signs about a connection today", each line [read] with file and line: the lengths rows (`warden.py:689-691` hands each released connection's release, bytes in, `DATA` cells and whether it failed), the fails slot, and how each reaches a signed root. proofs asked what the certifier signs a connection: a no-stop class's count could then be the certifier's and not advice. Say what you read and nothing more.
5. Show the commit is text only [ran]: the Python files' syntax trees with docstrings stripped are equal before and after; the five circuit-check bindings' pins hold; the two test files pass on your VM; the README is at or under its cap.
6. NO new remote run. The run of record stays `r20261111-220825-e7f8` at `9100e6da14d1`; the body says the head differs from it in docstrings and Markdown only, with the check of point 5.

Report at 4:30 PM PST as before, with this commit's full sha.

