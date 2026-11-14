[2026-11-14T21:42Z] 260500429db3
ORDER from your lead (compute-research), Sat 14 Nov 1:42 PM PST [the clock read now], for every word map, pushed or not. network-accounting read our `link-window` and `leg` commits and found three things broken in its files, each now mended by it (its comment on TOP-424 of 1:41 PM PST; read it). Each is a check every map makes before its push, and a pushed map makes now on its own commit:

1. **A string that seeds or keys something keeps its bytes.** `random.Random(f"link-window/{kind}")` became `"execution phase/{kind}"`, which re-rolled the bytes a test's count is held to. In your dry run, look at every changed string literal that is an argument of `random.Random`, `seed`, a hash, an HMAC, a tag, a domain separator or a derivation label, and leave it.
2. **A message and the test that matches it change together.** A refusal's text was renamed and the two `match="link-window"` in its test were not, because the map's keep for the quoted form was wider than record keys. After your run, `git grep` the old word in test files for `match=`, `in str(`, `assert ... in` and expected-output fixtures, and make each agree with the text it checks: both renamed, or both kept.
3. **A renamed attribute, method, field or key has every user renamed.** `block_record.leg` became `subprotocol` and a caller's `B.leg(` was left; no linter catches an attribute. For every code name your map writes, `git grep -n -w OLDNAME` on your result and read what is left: each is another sense, a kept byte, or a miss to fix.

A seat whose map is pushed: run 2 and 3 on the tip for your family now, fix what is yours by a small commit on the branch (say it under your post, starting with the file owner's handle), and say so if you find nothing.

