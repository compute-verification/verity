import sys

src, dst, suites_md, notrun_md = sys.argv[1:5]
s = open(src).read()


def rep(a, b):
    global s
    assert a in s, a[:80]
    s = s.replace(a, b)


rep("  - Its compare phase also records the `_C_stable_libtorch.abi3.so` the run loaded:",
    "  - It catches no exception wholesale. A phase that raises ends the script non-zero with its traceback in the run's log, and the `capture.json` it wrote before that phase says `ok: false`. The library record alone catches by name (`ImportError`, `OSError`), so the rows' verdict is still written when vLLM, its library or `cuobjdump` can't be read.\n"
    "  - Its compare phase also records the `_C_stable_libtorch.abi3.so` the run loaded:")

rep("Run `r20261112-192752-fca0`, at this PR's head `79cb0b3e28`, on card 3",
    "**At which head it ran.** No capture has run at this PR's head `0978f5af04`. The last ran at `79cb0b3e28`. "
    "`0978f5af04` changes only how the capture fails: the two blanket catches in its capture phase are gone, and the library record catches `ImportError` and `OSError` by name. "
    "It computes the same rows, comparisons and record fields. Where before a raising phase was recorded and the script went on, now the script stops there with its traceback, over a record that says `ok: false`. "
    "The capture of record runs at the head in the next card row.\n\n"
    "Run `r20261112-192752-fca0`, at `79cb0b3e28`, on card 3")

start = s.index("The head `79cb0b3e28` changes one word")
end = s.index("\n", start)
s = s[:start] + (
    "`79cb0b3e28` changes one word of the Definition's conformance text (`sm_90` to `cc 9.0`) and none of its words. "
    "With the sync, at `79cb0b3e28`, `r20261112-192652-7776` gave rc 0:\n\n"
    "~~~text\n"
    "circuit-check: 2 target(s), 0 failure(s), 0 unchecked (warnings, which never fail) on 0 Definition(s), report /workspace/research/runs/r20261112-192652-7776/out/circuit-check.json\n"
    "~~~\n\n"
    "The head `0978f5af04` changes no file circuit-check reads.") + s[end:]

rep("| `r20261112-192752-fca0` | `79cb0b3e28` (head) |", "| `r20261112-192752-fca0` | `79cb0b3e28` |")

start = s.index("- **`r20261112-192652-7776`, with the sync**")
end = s.index("\n- `r20261112-192536-de27`", start)
s = s[:start] + open(suites_md).read().rstrip("\n") + s[end:]

rep("On the VM, at the head `79cb0b3e28`:",
    "On the VM, at the head `0978f5af04`:\n\n"
    "- `python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD --json` (base `5f0c008aa0`): `failures: []`, 0 `# noqa` lines added and 0 removed. At `79cb0b3e28` it failed on the three `# noqa: BLE001` lines of the capture's catches.\n"
    "- `adapters/vllm/tests/properties/test_fp8_read_back_capture.py`: 47 passed.\n"
    "- `ruff check` on the capture file: passes.\n\n"
    "On the VM, at `79cb0b3e28`:")

start = s.index("## Not run\n")
end = s.index("<!-- CURSOR_AGENT_PR_BODY_END -->") if "<!-- CURSOR_AGENT_PR_BODY_END -->" in s else len(s)
s = s[:start] + open(notrun_md).read().rstrip("\n") + "\n\n" + s[end:]
open(dst, "w").write(s)
