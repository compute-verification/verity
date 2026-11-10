[2026-11-10T02:32Z] 3a86123c6a46

## PROVECUT, Tue 10 Nov 0232Z: CIR-201 design (d), my side is on the issue; one ruling for Daniel

- **On CIR-201:** if C_R may be public, my history `params` and `pins` don't change. Only their caller does: roots
  from the interiors' registration (or the statement's), the replay's public bodies, and R's served strings as a held
  run. One-stage then needs no root inside the statement.
- **The ruling: may a warm request's prefix-hit length C_R be public to the verifier?** It shows how far R shares a
  prefix with another request, which is information about values, not protocol structure.
  - Public costs seconds of verifier CPU.
  - Private needs a statement proving the composed root: about 6.7×10^12 ANDs a sampled 128k request, 0.4 to 1.8
    H200-hours [estimate]. The committer would also pad the served table to a public length, and two pieces aren't
    built.
  Please take it to Daniel, or rule if it's yours. Until then, a warm request stays refused by name (#8391).

