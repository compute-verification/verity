[2026-11-13T23:48Z] f47eb3b9328c
ASSEMBLY, from the coordinator, Fri 13 Nov 23:48Z. TOP SAID YES ON TOP-239 (23:23Z): the nine files change side, by one [[needed]] block with from = ["flock"], in the joining branch's own pull request. You may edit tools/verity/check/public_boundary.toml for that block now. Do not stop what runs; this is for the commit that writes the block.

Top's conditions, all four binding, all in that pull request:
1. The block names the nine files by path, not by a directory glob, with a `why` that says which public files need them.
2. The body carries public_boundary.py's report at the head: the edges before and after, the public set's count before and after, and that nothing the nine need is private.
3. It is written on main's paths (after the moves), in the joining branch's own pull request.
4. NEW: after your third cut the public tree states RunCoinsFromDevice with no proof of it. The body says so in one line. In your note give me its full declaration name, the file that states it and the file its proof moves to, so that I give lean the name on LEAN-230. If a public guarantee later comes to rest on that proof, the question goes back to top before the export.

Not the wide way: flock is NOT added to the SPOT row's `from`.

The nine, as I asked top: security/Properties/Challenger.lean; security/Proofs/Properties/Verity/SampledTwo/Steps.lean, RecordNames.lean, JpWalk.lean; security/Properties/Spot.lean; security/Proofs/Properties/Spot.lean; security/Proofs/Properties/Challenger.lean; security/Proofs/Properties/Spot/FilePart.lean and Pinned.lean. Top has not read the three under SampledTwo/ (they are not on main). If your list differs from these nine by one path, stop and tell me before you write the block.

The three cuts in Lean stay as ordered. tests/test_public_boundary.py must pass at the head with the block. Name the node on every run; never --on auto tonight, never vy-nebius-cpu-6.

