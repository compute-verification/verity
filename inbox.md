[2026-11-14T21:12Z] 2ef8b261e314
LEAD: ci queue notice for #10143: its base #10227 landed in main at 0c41579e1bef. Merge main into #10143 ONLY where no move lies between: check `git log --grep "^Move:\|^Layout-move:" <merge-base>..origin/main`; if a move commit is there that #10143 lacks, a plain merge is clean and WRONG: carry first with `python3 tools/move/restack.py cursor/vllm-wires-portrows-5e1a --main MOVE` (through `research run --on auto`, not on the VM), then merge main into the carried result, and check the tree differs from main only in your files. Then the gate and the mark.

