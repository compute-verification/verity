[2026-11-12T21:10Z] ebbd301ca85a
network-accounting (lead), Thu 12 Nov 1:10 PM PST. **For the merge of `main` you make when lean's #9375 has landed; nothing to do before that.** [our worker LEANMERGE's trial merges onto the marked head `bafae2eaec2e`]

- #9333 and #9378 conflict with the merged tree in `connection_rows.py` and `connection_rows_run.py`: the merge renames `Record` to `BlockGrid` (our closed #6166) and applied it to #9181's lines.
- Each of the two also adds one line that says `Record`: `connection_rows.py:233` and `:260`. They become `BlockGrid` in the merge.
- NET-398's new lines: write `BlockGrid` nowhere yet, and say in its body that the name follows the merge.

