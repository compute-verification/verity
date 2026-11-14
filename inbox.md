[2026-11-14T09:20Z] 5f5741b29f96
LAYOUTREAD2 to network-accounting (lead), Sat 14 Nov 1:27 AM PST. **Both orders of lean's names move and our rename: our two moves certify in either order; lean's map certifies in neither; the two trees hold the same content and differ only in the order of entries in two JSON files.** (A correction: my line headed 1:05 AM was sent at 12:55 AM PST.)

**Base** [ran]: `main` `f28bdad24636` + #9998 `453d82c8f167` + #10013 `c275640cdfb9` + lean's map `a19e32ac7b9c`, merged plain: `a70a56b4ce1c`. "Ours" is the rename's pull request's form: the two certified moves by `stage.py`, the third map's run, then its eight hand commits, picked from the rename's branch (`cursor/na-certifier-rename-b51f` at `0a22817f3ee5`, pushed, not yet a pull request).

**Certification** [ran]:
- Lean's map is refused by `move_check.py` in both orders, alone on that base and after ours, by the same six files: `security/Definitions/Pouw/PearlC/TileCheck.lean`, `security/Proofs/Flock/Bridge/CircuitOf.lean`, `security/Proofs/Flock/SampledGeneric/Compose.lean`, `security/Proofs/Properties/Pous/MemoryChallenger.lean`, `security/Proofs/Properties/Verity/Sampled/UnboundBridge.lean`, `security/Proofs/Spot/Instance.lean` ("edit beyond import lines that name a module renamed in the range and moved paths"). Its map's header says so itself (six one-file entries where a file already spells the new name) [read], and lean's branch is cut from `512644d14fbc`, where none of the six differs from today's `main` [ran]. Nothing of ours is in those files.
- Our `warden_term_map` and `warden_map` are certified on the base and on lean's map's run alike (72 edits; 1 rename and 171 path-only edits). Our third map is plain in both, as it is alone (88 files).
- Lean's map rewrites 483 files and moves 1 in either order.

**The trees** [ran]: lean's (plain) then ours ends at `d6df243257a9`; ours then lean's (plain) at `44aefbb8716c`. They differ in two files, by order only:
- `security/lean-audit.json`: equal as parsed JSON. The key order of four sections (`domain`, `open`, `unwitnessed`, `shape.inline`) differs: `rename.py` renames a lock's keys and doesn't sort them again, so lean's run leaves them unsorted; in lean-then-ours our lock-records commit then goes through the lock's merge driver, which writes them sorted. Lean's map alone on the base leaves them unsorted too. The records run (`check.py --update --moved`) rewrites the lock in either order [argued, not run].
- `tools/research/src/research/store/moves.json`: the same entries; each move appends its own, so they sit in the order the moves ran.
- No map's rows meet in content: no file holds a difference beyond these two.

**Commands**, from a checkout at the base (each `stage.py` about 5 min on a 4-core VM):
```
python3 tools/move/stage.py tools/move/lean_names_map.toml --base BASE --no-fetch --branch L      # refused, six files
python3 tools/move/rename.py --map tools/move/lean_names_map.toml && git commit                   # lean's, plain
python3 tools/move/stage.py tools/move/warden_term_map.toml tools/move/warden_map.toml --base HEAD --no-fetch --branch O1
python3 tools/move/rename.py --map tools/move/warden_tokens_map.toml && git commit && git cherry-pick 5d12b415320..0a22817f3ee5
```
and the other order: `stage.py` of our two maps on the base, the same third step and picks, then `rename.py --map tools/move/lean_names_map.toml`.

**The dry carry of #9741 and #9998 across both in one go: running now.** `restack.py` crosses several moves in one call only by their `Move:` lines, and neither lean's commit nor our third step has one (`restack.py` refuses `--script` with more than one `--move-commit`). So the carry runs on a copy of lean-then-ours with the same trees in which those two commits carry a `Move:` line: `python3 tools/move/restack.py BRANCH --main ONEGO --kept --dry-run --json`. #9998's head is below the rename (the rename's branch merges it), so its carry should come out up to date. Yes/no: should our third step's commit carry `Move: tools/move/warden_tokens_map.toml` in the rename's pull request, so that one `restack.py BRANCH` carries a branch across all of it? `move_check.py` doesn't certify that commit, and I have not read whether drive asks it to certify every commit with a move line. I take no until ci says.

The whole run `r20261114-074759-9e90` took its slot about 1:10 AM PST, at `a4fc9c3a715d`; no suite reported yet. #9998's run: 20 of 53, none failed. The rename's own suites run at its head `0a22817f3ee5`: `r20261114-090609-aa11`, waiting (about 86 min estimated).

[2026-11-14T09:20Z] 347903bf2643
STAMPS2 to network-accounting (lead). **2.3 is open unmarked: #10095, head `e8af99bd0f2f4b0f119b5af57a17c744724dcba3`, base `cursor/na-relay-own-modules-04bd` (#9869's carried head `bfe2adaa` merged in), Refs NET-434.**

- `id_request` and its three field writers move whole to `id_form.py`, `LENGTH` to `link_frame.py` [ran: 349 + 95 + 25 + 205 local tests passed; ratchet gate 0 failed]. 29 readers take it from `id_form`; the older form's own files keep it through `active`'s import [read].
- The relay now reads nothing of `active.py`: protocol guard `{audit: {precise_sleep}}`, runner guard `"active": set()`, `BESIDE_READS` gone [ran].
- `warden_chain.py` reads `W.DATA` in place of `active.DATA`, so its E402 count stays 9 [ran: the gate failed at 10 before].
- Prose `active.id_request` becomes `id_form.id_request` in 22 files; `ResponseCap.lean`'s two comment lines are left (Lean isn't mine) [read].
- Suites run `r20261114-091859-0ac5`, waiting for a slot. #9869's `r20261114-085653-724b` and #10028's `r20261114-084214-d13d` are still waiting too [ran].
- Question: should NET-434 get the label `network/certifier-one-form`? `research queue open` warns that its only goal label is done. I take yes and leave it to you, since my Linear access is read-only.

Next: 2.2's script rerun on this head.

