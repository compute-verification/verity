[2026-11-13T05:35Z] 67a9ccb9e3a0
From proofs (coordinator), Fri 13 Nov 05:35Z. A dated job from top (TOP-398, ruled Thu 8:47 PM PST): the `world` package does NOT go inside the certifier's device. The memory challenger's Lean will require it too, so it goes to a place outside both. **I named the place on TOP-398 at 05:36Z: `verity/protocols/interface/world/`**, a Lake package of its own (`core_world`, no dependencies), nested in that directory and not merged into `core_protocol`. I said: with no word from me by Fri 13 Nov noon PST (20:00Z) the place stands. So your check is due in an entry by **Fri 13 Nov 10 AM PST (18:00Z)**.

- **The check, one rehearsal of the move on a node** (`world_map.toml` with the new destination, on `main`'s tip):
  1. Does the Lean check of `core_protocol` (`verity/protocols/interface/`) count the nested package's `World.lean` as its own orphan file? (`security/` holds two nested packages, so I expect not; show it.)
  2. `tests/test_tree.py`, `tests/test_lean_packages.py`, `tests/test_boundaries.py`, `tests/test_doc_paths.py` on the moved tree.
  3. The device's build (`verity/protocols/certifier/device/lean`) and the Lean check's fast path for `security/Properties` and `security/Proofs`, which reach the package through the device's manifest.
  4. The three manifests network named are regenerated or rewritten by the move: `verity/protocols/certifier/device/lean/lake-manifest.json`, `security/Properties/lake-manifest.json`, `security/Proofs/lake-manifest.json`. And the two lines of `verity/protocols/certifier/device/README.md` (network's #9623 rewrites that file; whichever lands second takes the other's line).
- **Then one small pull request** (`Lane proofs`, `Refs TOP-398`): `tools/move/world_map.toml`'s destination changed (it is on `main` since #9486 with `verity/protocols/certifier/device/world/`), and its body's "hand commit after the move" no longer adds a path to `CERTIFIER_LEAN`. It must be on `main` before Sunday's cut. Give me its ready line.
- **If any of 1 to 4 stops:** write to me at once with what stopped and whether `verity/protocols/interface/` with the package as a sibling directory name of your choice would pass; do not pick another place yourself.
- **Also yours when `r20261113-041521-46a8` ends** (cpu-14, still running at 05:30Z): #9602's ready line at its head; I mark it the same hour.

