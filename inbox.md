[2026-11-20T14:10Z] 0df92ce6e4c2
FROM top, Fri 20 Nov 6:10 AM PST. One more red on the tip that is mine and a ruling for it: `tests/test_tree.py` is red because `verity/lean_process.py` (added tonight by memory-accounting's panic rule; 100 lines; the one place every launch of a compiled Lean verifier in `verity` goes through, imported by proofs', compute's and memory's verifiers) has no place in `tests/tree.toml`. RULING: it stays where it is, directly under `verity/`, as shared machinery beside `errors.py` and `ports.py`. Add ONE entry to `tests/tree.toml`'s `verity/` section, after `verity/hooks.py`:

[[tree]]
path = "verity/lean_process.py"
means = "how a compiled Lean verifier is started and when its answer is read: the panic switch, a panic line refused, an undocumented exit refused"

and whatever count `test_tree.py` holds for that section, moved by one, on my word. Do not move the file. Put it in the same branch and pull request as the other ten.

