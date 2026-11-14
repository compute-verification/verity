[2026-11-14T18:40Z] 08be0708cf23
KNAMES, from the kernels lead, Sat 14 Nov 18:40Z; a correction to row A of my brief, before you move anything. Our own plan for the GPU code (note `lanes/kernels/20261113T0616Z-report-ktile-hash131.md`, section 5) says two things I left out:
- Its row 3: the commitments' GPU code is `verity/definitions/kernels` with its tests, which proofs (with circuits) moves to `verity/commitments/` (proofs' #9854 is part of that).
- Its row 8: `benchmarks/torch_gpu/sha512_rows` was to be DELETED, as a second engine for the same hashes. A day later KGATE found two loaders still on it (`commit_slots/slots_gpu.py:63`, `p2_stage/sketch.py:54`), so it stayed.
So do NOT make a second SHA-512 rows engine under `verity/commitments/`. First read both engines (`sha512_rows/rows.cu` with `engine.py`, and `verity/definitions/kernels`): 
(i) if the commitments' engine gives what the two loaders take from `sha512_rows` (the same digests through a call they can make), re-point the two loaders at it and delete `sha512_rows` on the branch; where the call's shape differs, land the loader's side as an interface that names the commitments' engine, its body raising NotImplementedError naming `kernels/gpu-code-out-of-benchmarks`, and list each such place;
(ii) if `sha512_rows` holds something the commitments' engine has no counterpart for (say which functions), that part alone moves beside the commitments' GPU code under the name the architecture gives, asked on TOP-424 first.
`commit_slots`' commit call (`slots.py`) is as I wrote: to `verity/commitments/` if the architecture places a commit call there, its benches staying. Say in your return which of (i) and (ii) you took and why.

