# Statement-binding fixtures for the direct Ligero GPU prover (`archive/direct/ligero`)

The red team's checklist (`note:r20-proof/red-team-vu/20260922T1042Z-report-redteam-vu`, F1 / F2 / F19) applied to the b-ligero prover,
lane `b-ligero2`.  `binding.json` is the output of

~~~text
python -m archive.direct.ligero.redteam_binding --device cuda --out binding.json      # vy-g3 (L40S)
~~~

| case | what | pre-fix (`main` 799ec36) | post-fix |
|---|---|---|---|
| `zero_witness_as_real_words` | a proof of two trivial VUs (`a = b = 0`, `y16 = 0`) presented as a proof of the first two real `vu-k1536` VUs | rejected (linear test: the pins are checked against the verifier's decodes at the opened columns) | rejected (column challenge mismatch: the statement is in every challenge) |
| `small_batch_vs_large_verifier` | an `l = 256` (2-VU) proof to the verifier configured for `l = 512` | rejected (`malformed proof`: `w`, `h`, `q`, opened columns are checked against the verifier's `Config`/`System` sizes, never read from the proof) | same |
| `adaptive_claim` | 21 real VUs proved honestly at `l = 2048`; the claimed BF16 words replaced by `y16 + delta` with `sum_j r_end[d, j] delta_j = 0 mod p` for the six challenge coordinates (`r_end` was a function of `root` alone; LLL on the kernel lattice, 1.2 s, `max |delta| = 352`) | **ACCEPTED -- forgery**: 21 wrong output words proved by an honest proof of the right ones | rejected (column challenge mismatch) |

The pre-fix transcript derived the challenges from `H(root)` and `H(root, w, h, q)`: the statement (operands,
claimed words, layout, code sizes, system) was never absorbed.  The verifier's chain-test scalar check
`l (q_0 + q_l) = r^T b` therefore bound the claimed words only through six BabyBear linear forms, and a prover who
has already produced an honest proof can move the claim inside their kernel (a lattice of determinant `<= p^6 ~ 2^186`
in dimension 21: shortest vectors have entries of a few hundred).  The zero-witness and size attacks were already
caught, because the pins and the sizes enter the verification equations directly; the checklist violation was the
adaptive one.

Fix (`protocol.statement_digest`): `H(tag, p, hash, system_id, l, n, D, t, mode, layout, a, b, y16 | c, y)` computed by
the verifier from its own configuration and inputs, absorbed into both challenge seeds; sizes from the verifier's
`Config`/`System`; the proof's hash name checked.  The pre-fix transcript is reachable only through the red-team hook
`protocol._BIND_STATEMENT = False`, which this fixture flips to demonstrate the forgery.
