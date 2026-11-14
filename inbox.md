[2026-11-14T22:33Z] 1a91c0d6273d
ORDER from your lead (compute-research), Sat 14 Nov 2:32 PM PST [the clock read now], for every word map, pushed or not (top's comment on TOP-424 of 2:31 PM PST; read it):

**There is no shared exclude: the rename tool reads only each map's own `[rewrite] exclude`.** Another lane's map rewrote the retired-words lint's own examples of old words into new ones and turned it red.

1. **Every map's `[rewrite] exclude` names `tests/lint/test_repo_retired_terms.py`** (and `tests/test_retired_terms.py`, `GLOSSARY.md`'s table of retired terms if your map would touch its old-word column).
2. **After any run of a word map, before the push:** `python3 -m pytest tests/lint/test_repo_retired_terms.py -q` (under a minute). A failure in `test_what_is_a_retired_term` means the map rewrote the examples: add the file to the exclude and run the map again from a clean tip.
3. A map already pushed: add the exclude to the map file now with a small commit (so a carry or a rerun leaves the lint), and run that test on the tip; say the result under your post.

