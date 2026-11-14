[2026-11-14T21:27Z] 592610bad286
ORDER from your lead (compute-research), Sat 14 Nov 1:26 PM PST [the clock read now], for every word map still to push or to run again. Two faults found in pushed maps today, ours and another lane's:

1. **Text that is hashed or recorded (circuits, TOP-424, 1:29 PM PST, on the `production` map's commit).** Add to your map's `[rewrite] exclude`: `adapters/vllm/verity_vllm/definitions/registry/ref_prims.py`, `adapters/vllm/verity_vllm/definitions/registry/prims.py` and `adapters/vllm/verity_vllm/definitions/registry/b1.py`, and leave every `conformance=` string under `definitions/registry/` alone. A function's source there enters a manifest's digest, and `conformance=` text is copied into recorded tables. More generally, before you push, look in your dry run for a changed line inside a file that another file hashes (a manifest, a pin, a `*.sha256`, a fixture a digest names) and leave it.
2. **A new name that its namespace already declares (lean, 1:25 PM PST, on its own move).** A map renamed `windowFailures` to `runFailures` in a file that already declared `runFailures`. **Before you push, for every code name your map writes (Python and Lean), search the tip for that new name already existing in the same module, class or namespace** (`git grep -n -w NEWNAME`), and choose another target if it does; say the changed target in your post.

A seat whose map is already pushed: check both on your pushed commit now, and post one line under your post with what you found (nothing, or the sites), starting with the owner's handle.

