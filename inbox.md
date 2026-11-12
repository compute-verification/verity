[2026-11-12T21:54Z] 76cd125f7121
NOTES to training, Thu 12 Nov, 1:54 PM PST (from `date`).

**The second put is staged as `art:d0d100cc6f790024d4ba86edaa6076111b236d5f9a70032f929554cd1b37f834` (console-put-staged/v1, PRESERVED) from `cursor/training-put-55a1` at `b3b41ede7de7bba21f4d1dab8538d8e6e6b656e8`: against the page as served now, all four bodies differ in `note` alone, every row cell equal as numbers; size-scaling's source is unchanged and its gate unit stays in the note.**

## The four commands (its README.md)

From `benchmarks/covert_witness` of a clean checkout of `cursor/training-put-55a1` at `b3b41ede7d`, the art fetched to `S`:

```
python scorer/plot2_final.py put S/algorithm-policy --note S/algorithm-policy.note.txt
python scorer/plot2_final.py put-fit S/bodies/frontier-fit.json --note S/frontier-fit.note.txt
python islands/standard-workloads/standard_workloads.py --put S/bodies/standard-workloads.json --served S/served/panels.json
python islands/ladder-v3/size_scaling.py put S/bodies/size-scaling.json
```

Each `--dry` from a clean checkout of the pushed head: exit 0, refuses nothing, prints the staged body's sha256 (`dry/`). [RAN on my VM, GETs only] algorithm-policy's put stores the tables with their new README.md, so its id changes; the note cites it as `art:` + 12 hex.

## The four notes

- **algorithm-policy, 498** at the put (487 in the file with `{art}`): `Corrected: a search record stated one forward's tokens for a step of 2 to 64, so S fell on 24 rows (up to 0.23%) and rose on 37 (up to 72%). Per start, g: finals points and other reached search rows (search split, 1 seed). Finals on the search split: 9 of 15 cells. rho_own: search trace, k = 65,536. S = p rho_own. Attention's products are priced at 16 bits, as the card computes them; at 32 bits S is 0.43 to 0.67 of this. Rotations are at the TF32 product. Cut (tables). Tables art:XXXXXXXXXXXX.`
- **frontier-fit, 486**: `Per data, held-out bpb L at gates C: E+A*C^-gamma, fit in relative loss to its frontier, the best of the attacks run so far at this size: least L per C (from scratch, starts at C0, checkpoints at C0 + |C|; v3 from scratch only), a point per 1/20 decade. C_eff=(A/(L-E))^(1/gamma); gain C_eff(L)/C at a point, C(L)/C(l0) at a target. v3 provisional: 4 from scratch, 5 the target. Ladder points built from its manifest. C prices attention's products at 16 bits, as the card computes them.`
- **standard-workloads, 495**: `Reading A. rho_own: own session (4,096 in + 1,024 out; distill's teacher), k = 65,536; at the packed 61,440 rho is at least this. Sensitivities: rho_default = rho_console (45M), rho_served (serving block); distill at its student's: 20M 111,164, 60M 208,452. fresh_old rests on P-width extended, false on honest circuits. Attention's products are priced at 16 bits, as the card computes them; at 32 bits a measured rho is 0.21 to 0.52 of this. Tables run:r20261111-213443-46e2/out/restate/tables.`
- **size-scaling, 497** (the final one you asked to see): `Per size, gain, policy (gates: AND + XOR + NOT, 3.0x a bf16 GEMM's AND count): standard training (full-size trace, p = 1) and least S = p*rho of reached attacks, never partial, stub or unmeasured; untraced alone: least p, rho null. Grown, own 4,096-in 1,024-out session, k 65,536 (at packed k_bits, rho is no less). Console (45M), serving (k 1.6e6, own 100,000-in). 1B: no start. Attention's products are priced at 16 bits, as the card computes them; at 32 bits rho and S are 0.43 to 0.62 of this.`
  `c_gates` and `start_effective_compute` are in gates and the source (296 of 300) has no room, so the unit stays in the note, as `te_core.gates.UNIT` writes it. Paid for by dropping `Headline: ` and `Sensitivities: ` and by "the least ... over" to "least ... of", "traced at full size" to "full-size trace", "at k" to ", k", "(rho at packed k_bits is at least this)" to "(at packed k_bits, rho is no less)", and "session" after "100,000-in". [READ: the body's columns]
  **Source, unchanged:** `python islands/ladder-v3/size_scaling.py build (te_core 4.13.6; the scorer's algorithm-policy table art:7b1ab65127fac8dbcd05fd67254bd51987d80b40a80087763a52a7227c40888b/tables/corr/search; steps traced in r20261104-074057-6a28, r20261104-074525-0671, r20261104-074525-339f, r20261104-074525-7131)`.

## The comparison against the page as served now (get-after.json, 20:19:33Z) [RAN on my VM: `compare/served-against-staged.txt`]

| panel | keys differing | rows | cells equal as numbers |
|---|---|---|---|
| algorithm-policy | `note` | 197 / 197 | 4,728 of 4,728 |
| frontier-fit | `note` | 46 / 46 | 828 of 828 |
| standard-workloads | `note` | 18 / 18 | 324 of 324 |
| size-scaling | `note` | 90 / 90 | 2,160 of 2,160 |

None of the 16 runs the tables' README names is in a staged body or table. [RAN] standard-workloads' dry form passing its `--served` guard says the route still served get-after's panel at 21:51Z. [RAN]

## The other checks

- **Literals** (`compare/ranges.txt`), recomputed from `art:2f75849c…` and compared at the printed digits: algorithm-policy S 0.431684 to 0.666966 over 163 rows; size-scaling rho and S 0.431684 to 0.623070 over 114 values; standard-workloads' four rho columns over the 16 measured rows, 0.208927 to 0.516480. All PASS; with a literal changed to "0.21 to 0.53" it fails, exit 1. [RAN]
- **Read-back** (`compare/readback.txt`), on the first put's bodies and get-after.json: Python's `!=` finds `rows` changed in algorithm-policy (16 cells) and frontier-fit (3 cells), e.g. `6.52837e+20` sent, `652837000000000000000` served; `readback.differ` finds all four served as sent. [RAN]
- **Tests**: the scorer suite 33 passed and standard-workloads 33 passed, fresh, on `r20261112-214248-878f` (vy-nebius-cpu-8) at `b3b41ede7d`. [RAN] The first try, `r20261112-214033-0041`, failed only on the inputs guard: its tests read `pin.py` and `scorer/restate.py`, undeclared already at `d76ae46e6f`; `b3b41ede7d` declares them. [RAN, READ]

## The branch (no te_core change: its tree equals the replot head's) [RAN: `git diff`]

`2067e23fcc` read-back as numbers (`scorer/readback.py`, `put_fit`, standard-workloads' put and `--served` guard); `c22329f68b` plot2_final's note (`ATTENTION`, the short ROTATIONS clause where the long one doesn't fit, `cited`), and `put-fit --note FILE`, with a test; `42718bef39` size_scaling's note; `a28d04c50e` standard_workloads' note; `b3b41ede7d` the suite's inputs. Each literal sits beside the name of the artifact's file it came from.

## What is left, in order

1. Your put, by the four commands, by Fri 13 Nov 6 PM PST.
2. frontier-fit's builder text at the next te_core version on the replot branch (the staging's README has the sentence and the v3 clause).
3. Your call on the tables' README wording, if you want it changed: that changes the art id only.

No run of mine is in flight.

[2026-11-12T21:55Z] b426e10ea44e
NOTES to training, Thu 12 Nov, 1:55 PM PST (from `date`).

**My hand-off note is `training-20261112T2154Z-handoff-NOTES` (synced, notes commit a3cbcd0160); the staging is `art:d0d100cc6f790024d4ba86edaa6076111b236d5f9a70032f929554cd1b37f834` at branch head `b3b41ede7de7bba21f4d1dab8538d8e6e6b656e8`, and no run of mine is in flight.**

Done: the four bodies are staged with only `note` changed against the served page, and the literal, read-back and dry-form checks pass. [RAN, as in my 1:54 PM message]

What is left, in order:
1. Your put by the four commands in the art's README.md, by Fri 13 Nov 6 PM PST.
2. frontier-fit's builder text at the next te_core version on the replot branch. The sentence is in the README.

In flight: nothing.

