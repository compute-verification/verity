#!/usr/bin/env python3
"""PR 9317's body after the capture of record at the head: `patch_body2.py POSTED.md OUT.md`.

POSTED.md is `gh pr view 9317 --json body -q .body`. Every edit is an exact replacement of text that must be there once.
"""

import sys

HEAD = "0978f5af04bb7a35269ce2fdf874f5a675539c97"
CAPTURE = "r20261113-031927-2fac"
SUITES = "r20261113-003940-657d"

EDITS = [
    (
        """**At which head it ran.** No capture has run at this PR's head `0978f5af04`. The last ran at `79cb0b3e28`. `0978f5af04` changes only how the capture fails: the two blanket catches in its capture phase are gone, and the library record catches `ImportError` and `OSError` by name. It computes the same rows, comparisons and record fields. Where before a raising phase was recorded and the script went on, now the script stops there with its traceback, over a record that says `ok: false`. The capture of record runs at the head in the next card row.

Run `r20261112-192752-fca0`, at `79cb0b3e28`, on card 3 with cores 34-43. The card is an H200 (capability 9.0), vLLM is `0.28.1rc1.dev472+gd9105ea80`, and the stand-in's attention is `FlashMLASparseImpl` over an `fp8_ds_mla` cache with 64-row blocks. The run exited 0.

`r20261112-174811-7ee1`, at the head's parent `185d1e34ae`, gave the same rows, counts and steps.
""",
        f"""**The capture of record ran at this PR's head**, `{HEAD}`: run `{CAPTURE}`, on card 3 with cores 30-39, in the row of Fri 13 Nov 03:15Z to 04:25Z. The card is an H200 (capability 9.0), vLLM is `0.28.1rc1.dev472+gd9105ea80`, and the stand-in's attention is `FlashMLASparseImpl` over an `fp8_ds_mla` cache with 64-row blocks. The run exited 0, and its `capture.json` and `compare.json` both say `ok: true`.

- **The lease.** One lease on card 3, held 48 s (03:20:01Z to 03:20:49Z) and declared with `--cores 30-39`: gpu-lease's line reads "CMD pinned to CPUs 30-39". The card held 0 MiB before and after. The compare ran after the lease on the same cores and ended at 03:23:50Z.
- **Against `r20261112-192752-fca0`, at `79cb0b3e28` on cores 34-43.** The two runs' records differ in one value of 4,701: the path of the Definition's module in `capture.json`, which holds the source SHA.
  - `compare.json` is the same on all 56 values, the library's sha256 and the coverage counts among them.
  - `capture.json` is the same on 4,644 of 4,645. That covers every step's batch and route, and the SHA-512 of the rows and of the words of each of the 132 calls outside the first layer.
  - The 22 saved first-layer calls and the 2 planted files are byte for byte the same.
- `0978f5af04` changes only how the capture fails: the two blanket catches in its capture phase are gone, and the library record catches `ImportError` and `OSError` by name. No phase raised in this run.
- `r20261112-174811-7ee1`, at `185d1e34ae`, gave the same rows, counts and steps.
""",
    ),
    (
        "`r20261112-174811-7ee1` was given the default wheel's library hash to check against and recorded `matches: false`; the run at the head was given the cu129 wheel's and recorded `matches: true`.",
        f"`r20261112-174811-7ee1` was given the default wheel's library hash to check against and recorded `matches: false`. `r20261112-192752-fca0` and the run at the head, `{CAPTURE}`, were given the cu129 wheel's and recorded `matches: true`.",
    ),
    (
        "- The listing was not made on node A. Its `cuobjdump` has no `nvdisasm` and exited 1. What the run shows on the node is the file's hash and that it registers the page kernel under the same name.",
        "- The listing was not made on node A. Its `cuobjdump` has no `nvdisasm` and exited 1, at the head too. What the run shows on the node is the file's hash and that it registers the page kernel under the same name.",
    ),
    (
        "Each condition is marked read (in the pin's source) or run (in `r20261112-192752-fca0`, the first attention layer, tensor-parallel 1, steps of at most 512 tokens).",
        f"Each condition is marked read (in the pin's source) or run (in `{CAPTURE}` at the head, and the same in `r20261112-192752-fca0`: the first attention layer, tensor-parallel 1, steps of at most 512 tokens).",
    ),
    (
        "The head `0978f5af04` changes no file circuit-check reads.\n",
        f"""The head `0978f5af04` changes no file circuit-check reads. So at the head, with the sync, `{SUITES}` gave rc 0 and reused both targets. No check ran anew there; the last that did is `r20261112-192652-7776`'s, at `79cb0b3e28` (2 checked, 0 reused).

~~~text
ok        definition Fp8DsMlaUpconvertRow_v1{{DN=128,DR=8}}  reused: every input its check read is unchanged (it took 0.3s)
ok        definition Fp8DsMlaUpconvertRow_v1{{DN=512,DR=64}}  reused: every input its check read is unchanged (it took 0.8s)
circuit-check: 2 target(s), 0 failure(s), 0 unchecked (warnings, which never fail) on 0 Definition(s), report /workspace/research/runs/{SUITES}/out/circuit-check.json
~~~
""",
    ),
    (
        "Card runs, on node A (`vy-spot-glm53`), card 3, cores 34-43, one lease each under `GPU_LEASE_WHO=circuits-fp8readback`. The card was idle at each start, so none waited.",
        "Card runs, on node A (`vy-spot-glm53`), card 3, one lease each under `GPU_LEASE_WHO=circuits-fp8readback`. The card was idle at each start, so none waited. Thursday's three ran on cores 34-43, pinned by `taskset` with no cores declared on the lease. Friday's ran on cores 30-39, declared on the lease with `--cores`.",
    ),
    (
        "| `r20261112-192752-fca0` | `79cb0b3e28` | 48 s | Exit 0. The same figures. Library `matches: true` against the cu129 wheel's hash. The compare ended at 19:32:13Z, off the card. |\n",
        f"""| `r20261112-192752-fca0` | `79cb0b3e28` | 48 s | Exit 0. The same figures. Library `matches: true` against the cu129 wheel's hash. The compare ended at 19:32:13Z, off the card. |
| `{CAPTURE}` | `0978f5af04` | 48 s | Exit 0, both records `ok: true`. Planted: 1,413 rows, 0 differ. Served: 127,274 rows over 154 calls and 53 steps, 0 differ; first layer 18,182 rows, 0 differ. Library `matches: true` against the cu129 wheel's hash. The compare ended at 03:23:50Z, off the card. |
""",
    ),
    (
        "  - `r20261112-222441-fadb`, `verity-vllm`, the try before it:",
        f"""  - `{SUITES}`, circuit-check on the sidecar and then `suites.py verity/definitions circuit-check`: rc 0. It waited 13m49s for slot r3 on vy-nebius-cpu-4 and held it from 00:53:49Z to the run's end at 01:53:28Z.
    - circuit-check on the sidecar: rc 0, 2 targets, 0 failures, 0 unchecked, both reused.
    - `verity/definitions`: **passed**, 986 passed, 0 failed, 0 errors, 2 skipped. Without the sync, at `ef693941ef`, it reported 942 passed and 17 skipped; I did not read which tests make up the difference.
    - The circuit-check suite (`tools/verity/circuit_check`): **passed**, 428 passed, 0 failed, 0 errors, 5 skipped.
  - `r20261112-222441-fadb`, `verity-vllm`, the try before it:""",
    ),
    (
        """- **The capture at the head `0978f5af04`.** It ran at `79cb0b3e28`, and the next card row (Fri 13 Nov 03:00Z to 04:25Z, card 3, cores 30-39) runs it at the head. Its lease now declares those cores (`--cores`). The three leases of today's runs declared none and were pinned only by `taskset`.
- **The `verity/definitions` and circuit-check suites with the sync**, at any head. They passed only in runs without it, at `ef693941ef`. The synced run that held them was cancelled before reporting them.
""",
        """- **The rest of `check`.** At the head I ran four suites (`repository`, `verity-vllm`, `verity/definitions` and the circuit-check suite) and circuit-check on this PR's sidecar. No other suite ran, nor `circuit-check --all`, nor the Lean check. This PR adds five files and one line of `catalog_ids.json`, all under `verity/definitions/` and `adapters/vllm/tests/`, and changes no Lean, Rust or lock file.
""",
    ),
]


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    body = open(src, encoding="utf-8").read()
    for old, new in EDITS:
        if body.count(old) != 1:
            sys.exit(f"patch_body2: {body.count(old)} matches, not 1, of: {old[:90]!r}")
        body = body.replace(old, new)
    if "### Mark" in body:
        sys.exit("patch_body2: the body holds a Mark section, which is the lead's")
    open(out, "w", encoding="utf-8").write(body)
    print(f"patch_body2: {len(EDITS)} edits, {len(body)} bytes -> {out}")


if __name__ == "__main__":
    main()
