[2026-11-14T07:52Z] 8adc08e994d1
## Coordinator, Sat 14 Nov 07:52Z: new piece: flock-verify becomes verity-verify-proof

From your lead (proofs), Sat 14 Nov 07:56Z. A new piece, tonight, beside your two goals. top ruled (TOP-422, 11:28 PM PST) that the new names for `Flock.Firewall`, `FC_FIREWALL*` and `flock-verify` are proofs' to pick with lean. I picked them on LEAN-80 at 07:56Z:
- `Flock.Firewall` -> `Flock.ZkServer`; `flock-firewall` -> `flock-zk-server`; `FC_FIREWALL` and `FC_FIREWALL_*` -> `FC_ZK_SERVER` and `FC_ZK_SERVER_*` (suffix kept). These three ride lean's worker WORDS's map for `Firewall` (LEAN-80; its tools are in the notes under `lanes/lean/tools/lean80/` and `lean80-words/`). You write none of them.
- `flock-verify` (the Lean executable; 610 files spell it) -> `verity-verify-proof`, reached as `verity verify proof` (GLOSSARY.md line 243). YOURS, two changes:
  1. the rename's map, in the format of lean's maps (read NAMES's report and scripts in `lanes/lean/tools/lean80/` and `tools/move/`), on a branch of your own `cursor/proofs-auditmap-verify-proof-<hex>` from `origin/main`, NOT applied to `main` and with no pull request: it rides lean's one chain so every open branch crosses once. With a dry application and `move_check.py`'s collisions. Leave record keys, format constants and recorded statement strings where the Glossary row says "until a format version".
  2. a small pull request of its own on `main`: the `verity` command gains `verify proof`, which runs the Lean executable (under today's name until the map lands; the subcommand finds it by one constant the map then rewrites), with a test, and AGENTS.md / READMEs / CLI help rewritten wherever they tell a reader to run `flock-verify` by hand. `Lane proofs`, `Goal proofs/terminology-clean`.
FIRST WORD: a note `verify-proof-sized` by 11:30Z (3:30 AM PST): the count by kind of file (Lean, lakefiles, Python, Rust, shell, docs, fixtures, pins), which marked pull requests the map stops (`git merge-tree` against the marked heads of `research queue status`), what `tools/move` cannot express, and your hours for 1 and 2. I post the size on LEAN-80 from it.
Before any ready note: the ratchet gate from a worktree of `origin/main` (`python adapters/vllm/tools/ratchet_gate.py origin/main <full sha>`), its count in the note. Do not end your turn to wait for me.

[2026-11-14T07:56Z] c99e60973f42
## Coordinator, Sat 14 Nov 07:56Z: two lane rules changed: you mark your own; build on marked heads

From your lead (proofs), Sat 14 Nov 07:57Z (Fri 11:57 PM PST). TWO LANE RULES CHANGED TONIGHT. Read once, then re-run your command.

Daniel, through top (TOP-422, 11:13 PM PST): a step of a plan that nothing physical forces is deleted. What forces a wait: a machine held by a named run until a stated hour; a build, check or proof whose minutes are measured; work that is not written; a key, a click or a claim that is Daniel's. Nothing else.

1. YOU MARK YOUR OWN PULL REQUEST, in the turn all four hold:
   (a) your runs at that exact head passed on a node: the changed suites' quick tier, and the Lean check of each package the change touches;
   (b) the ratchet gate, run from a worktree of `origin/main` (`python adapters/vllm/tools/ratchet_gate.py origin/main <full sha>`), passes; or every site it prints is one top's ruling passes (a use of a name `main` defines; a new file's own namespace where the namespace is on `main`; a site carried by a move; an identifier holding `MonitoringPeriod`), and the body has a section `## Ratchet gate` with the count and those sites counted by name and kind. Your own words (prose, messages, a new name you could have chosen otherwise) are mended first, with `GLOSSARY.md`'s term. Never disguise a spelling;
   (c) no `lean-audit.json` record of a guarantee's statement or of an assumption changes its text (only `reads`, or a move under `--update --moved`), and nothing changes what an accepted audit means;
   (d) the body is true: what ran at this head, what did not, `Lane proofs`, its goal.
   The mark: from a worktree of `origin/main` at its tip, `python -m research queue ready <PR> --mark --by proofs`; read its whole output (a conflict with another marked pull request, a body with no goal) and act on it. Then the note `ready-<PR>-at-<head12>` the same minute, its first line "MARKED by <you> at HH:MMZ". I read after and withdraw the mark on a finding.
   If (c) fails, or you are unsure of it: do NOT mark. Write the ready note with both texts of each changed record; I read and mark.
   A head in a train being checked: push the new head only when its checks are green, and mark it the same minute.
2. YOU DO NOT WAIT FOR A LANDING. Where your next piece sits on a marked head, ours or another lane's, build it on that head now (a merge in scratch for another lane's head; a stacked pull request on ours) and mark it the hour its base lands. Tell me in a note what you built on.
3. You do not end your turn to wait for me. When a piece waits on a run, go on to the next; end the turn only when every piece waits on a run, naming each run and its ETA.
Unchanged: heavy work never on your VM; push only to your own branches; no force push; a checkpoint line every 60 minutes; never print a secret.

