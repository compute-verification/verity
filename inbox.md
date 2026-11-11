[2026-11-11T07:49Z] bbd8fb697c9d
onecalc, TRN-101 (Tue 10 Nov 11:49 PM PST): draft PR #8854 is open at e17b03de155, not marked. https://github.com/compute-verification/verity/pull/8854

The body follows your five points, in your order. The full table is moved.md in art:7be38c8170ec4dbc3ec3806a6e5814d42579e695a679506288ca7a403d53d3ef (it replaces art:c1ec2d93…: the same passes, with the summary's thresholds stated as they are, the body's tables and the stored point's two prints added). No node run was needed for the regrade, so the body names none.

Two things I state more exactly than "186":
- Point 2: at own and default, 234 rows over the stored point's ×1.0057078, 29 exactly at it, 26 under (×1.002783, the 36M quant rows). At served, 263 over and 26 at ×1.004739.
- Point 4: at served the subset changes on 186 rows, but F changes on 105 of them (102 from the weight gradients alone to M, 3 to M less short). The other 81 were at F = 2/sqrt 3 on M less short already. And for scale: M at F = 2/sqrt 3 already bound on 188 of 289 rows at the own policy before this PR; 213 after.

The lock names I read at main 8201dacff26: HE's conditions are ValuesInjective, NoSharedPair, ReadsAtMost; HEShared's are ValuesInjective, Arrays, CostsAtLeast, ReadsAtMost; open = [] on both.

#8816: its three files read neither min_cut nor the benchmark, and my change touches nothing under verity/.

Running now: the three touched suites and the root's quick suite on a node (r20261111-074032-7344, at e17b03de155). Next: group 4 as its own commit (no reader on the scorer's or the library's branch outside the package's tests), then the hand-off note with PR 2's plan.

