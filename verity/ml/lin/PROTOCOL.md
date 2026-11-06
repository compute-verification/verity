# Lin v0

Lin proves matrix–vector proof blocks over committed tables, after FlashProve's linear layer (one shared opening of the
weights for every block a session proves), on SHA-512 and the challenger's live coins. v0 is the plain protocol: its
guarantee is soundness alone, and its messages reveal linear combinations of the committed values. Its verifier is
Python (`verity_lin.matvec.verify`); no protocol routes a proof block to it yet.

## 1. Field and code

F is the Goldilocks field, p = 2^64 − 2^32 + 1, its elements 8-byte little-endian words below p (one encoding each). E =
F[θ]/(θ^3 − 7) holds every coin; 7 generates F*, so it is no cube and the polynomial is irreducible; |E| = p^3.

A row of `width` elements (a power of two) is a polynomial's coefficients; its codeword is the polynomial's values at
the n-th roots of unity, ω = 7^((p−1)/n), n = width · blowup. The code has distance d = n − width + 1, and is F-linear, so
an E-row encodes coordinate by coordinate.

## 2. Tables

A table is r rows of `width` elements, encoded to U ∈ F^{r×n}. Column j is committed as one leaf of a frame-v3-sha512 tree
over the domain `CommitmentDomain(SHA-512(shape), -1, range-u64(0, n))`, `shape` the canonical JSON of
`{"format": "verity/lin/table/v0", "role", "rows", "width", "blowup"}`; the leaf's schema is `verity/lin/column/v0` and
its value a fresh 64-byte salt followed by the column's r elements. Columns open together: the count of distinct columns
(u32 big-endian), then each column ascending as its index (u64) and leaf value, then one frame-v3-sha512 multiproof
(`verity.ml.commitments.multiproof`: its length as u32, then its 64-byte nodes).

## 3. Sessions and coins

A session's transcript starts at SHA-512(`verity/lin/transcript/v0\0` ‖ statement digest), and each message (label,
data) moves it to SHA-512(`verity/lin/message/v0\0` ‖ previous ‖ u64be(len label) ‖ label ‖ u64be(len data) ‖ data).
Every coin is the challenger's, flipped after the transcript digest it follows is recorded and logged beside it. The
verifier reads coins from that log in order and refuses an entry whose digest or length isn't the one it rebuilt
(`coin-order`). Samples are drawn from the coin's raw bytes with no hash, by sampled proofs' exact rule
(`coins.OwnRandomness`, Lean's `Flock.Draw`): `uniform(m)` reads ⌈(bitlen(m) + 64)/8⌉-byte big-endian words and returns
the first below the largest multiple of m, mod m. An element of F is `uniform(p)`, one of E three of them, a column
index `uniform(n)`.

## 4. A linear claim

The claim is g·M·e = v on a committed table, g ∈ E^r, e ∈ E^width, v ∈ E (`linear`):

1. The prover sends w = g·M; the verifier refuses unless ⟨w, e⟩ = v (`claim`).
2. A coin draws γ, r independent uniform elements of E; the prover sends z = γ·M.
3. A coin draws t column indices (with replacement); the prover opens them (§2). The verifier refuses openings that don't
   parse (`encoding`), don't reconstruct the root (`column-path`) or aren't exactly the drawn columns
   (`column-position`), and unless, at each, Enc(w)[j] = Σ_i g_i U[i, j] (`column-claim`) and
   Enc(z)[j] = Σ_i γ_i U[i, j] (`column-proximity`).

Let e₀ = ⌊(d − 1)/3⌋, so 3e₀ < d. If U is e₀-far from every codeword matrix, then γ·U, a uniform word of U's row span over
E, is e₀-close to the code with probability at most d/|E| (Ligero's Lemma 4.5, [AHIV22], proved with no `sorry` as
`ProximityToRS.prob_of_bad_pts` in the ArkLib that `Security/Proofs` pins; U over F is e₀-far over E too), and each query
then catches Enc(z) ≠ γ·U with probability at least (e₀ + 1)/n. Otherwise U decodes uniquely to M*, and w ≠ g·M* makes
Enc(w) differ from g·U in at least d − e₀ columns. A claim false of M* is so accepted with probability at most
max(1 − (e₀ + 1)/n, 1 − (d − e₀)/n)^t + d/|E|. At blowup 4 that is (3/4)^t to first order; t = 241 (the default,
`matvec.queries(100, width, 4)`) gives 2^-100.

## 5. The matrix–vector statement

The statement (`matvec.Statement`, digest SHA-512 of its canonical JSON, format `verity/lin/matvec/v0`) names the weights
table W (`rows` × `width`), the sampled blocks' offsets o_u (u < n, each o_u + N ≤ rows), the output count N and the roots
of W, of the input table A (row u = a_u, n padded to a power of two) and of the output table Y (row u = y_u, N padded to
a power of two), with the code's blowup and t. It claims y_u[j] = Σ_k W[o_u + j, k] · a_u[k] in F for every u < n and j < N.

1. A coin β sets ρ_{u,j} = β^{u·N + j} (zero past n and N), and the prover sends C = Σ ρ_{u,j} y_u[j].
2. A sumcheck over (u, k), k the low log₂(width) variables, proves C = Σ G(u, k)·A(u, k), G(u, k) =
   Σ_j ρ_{u,j} W[o_u + j, k]: each round sends its degree-2 polynomial at 0, 1 and 2 and the verifier refuses unless the
   first two sum to the running claim (`sumcheck-round`); a coin binds the variable. The prover sends G̃(r) and Ã(r),
   refused unless their product is the last claim (`sumcheck-final`).
3. Three linear claims, in order: Y at g = (β^{uN})_u, e = (β^j)_j, v = C; A at g = eq(r_u, ·), e = eq(r_k, ·), v = Ã(r);
   W at g = c, e = eq(r_k, ·), v = G̃(r), with c_i = Σ_{o_u + j = i} eq(r_u, u)·ρ_{u,j}. W's claim opens the weights once
   for every block.

The session is accepted when every check passes and no message or coin is left over (`trailing`). A false statement is
accepted with probability at most (nN − 1)/|E| (batching) + 2·m/|E| (the m sumcheck rounds) + three linear claims' bound,
with the tables taken as the committed oracles; the binding of their roots rests on SHA-512's collision resistance.

**From F to the integers.** For int8 entries of W and a_u, |Σ_k W·a| ≤ 2^14 · width, so at width < 2^17 every sum is
below 2^31 < p/2 and the claim in F is the claim over the integers, which is the Boolean semantics of the exact s8
matmul (`verity.ml.circuits.ops.s8`) at those sizes. That the committed values are int8 and int32 images is not checked
here.
