# Candidate A — data-parallel GKR + logUp-GKR lookups + interactive Longfellow-style ZK layer

Status: **specification / proposed research**. Nothing here is implemented or measured. The interactive
adaptation of the Longfellow ZK layer (section 5), the merging of logUp-GKR claims into the checker GKR
(section 4.3), the depth reductions of section 5.4 marked **[proposal]** and the verifier-coin commitment of
section 5.5 are *proposals*; none is a drop-in component with a published proof for this composition.

**Label rule** (`backends/AGENTS.md`). A prover built from this document is `NON_ZK_PROOF_DIAGNOSTIC` until
sections 5, 6 and 8 are implemented and section 9 is reproduced by
`verity_numerical.security.accounting.account_candidate_a`. With sections 5.1–5.3 and 6–8 implemented but
**without step 0** (section 5.5) it is `COMPLETE_HVZK_BACKEND` — honest-verifier privacy, a milestone, never
`COMPLETE_ZK_BACKEND`. `COMPLETE_ZK_BACKEND` requires step 0 (or another justified mechanism) implemented, its
transcript order / commitments / binding / timing / abort rules / privacy argument / costs written down (5.5) and
red-team reviewed.

Text is tagged as in the handoff: **[source]** read in a cited document, **[derivation]** follows from sources by
a calculation shown or referenced here, **[proposal]** a design choice of this lane, **[assumption]** a parameter
or shape not yet fixed by another lane.

## 0. Sources (version read 2026-09-21)

| tag | document | used for |
|---|---|---|
| Tha13 | Thaler, *Time-Optimal Interactive Proofs for Circuit Evaluation*, arXiv:1304.3812v4 (8 Feb 2017), section 7 (7.1–7.3) | data-parallel GKR: copy index as extra sumcheck variables, prover cost \(O(S\cdot B)\) after \(O(S\log S)\) circuit-only preprocessing (Theorem 2) |
| Hab22 | Haböck, *Multivariate lookups based on logarithmic derivatives*, ePrint 2022/1530, revision 2025-02-20 | logUp: Lemma 5 (multiset ⇔ rational identity, needs \(\mathrm{char}\,F>N\)), Theorem 4 (soundness, challenge drawn from a set of size \(\ge|F|-|H|\)), Remark 3 (sampling outside the table range is for completeness only), section 3.4 (vector-valued lookups) |
| PH23 | Papini–Haböck, *Improving logarithmic derivative lookups using GKR*, ePrint 2023/1284, revision 2025-02-20 | fractional-sum GKR (section 3), Proposition 1 (GKR error), Proposition 2 (logUp-GKR error); the note calls itself *informal* and contains no ZK composition |
| FS24 | Frigo–shelat, *Anonymous credentials from ECDSA*, ePrint 2024/2010, revision 2026-04-27 | Protocol 2.5 (ZK by padded sumcheck transcript + Ligero on the decryption/verification circuit), Theorem 6 (honest-verifier ZK; knowledge soundness), section 2.2 Ligero bound used for parameters |
| libzk | IETF `draft-google-cfrg-libzk-02` (22 July 2026), sections 4–6 | Ligero tableau layout and commitment interface (4.2–4.3), prove/verify (4.4–4.5), quad circuit representation and in-circuit assertions (6.3), transcript encryption and deferred verification (6.5–6.7) |
| AHIV22 | Ames–Hazay–Ishai–Venkitasubramaniam, *Ligero*, ePrint 2022/1608 (2022-11-18; extended CCS 2017) | ZKIPCP (4.7), interactive variant via CRH (5.1), zero-knowledge condition Lemma 4.15, Theorem 4.7 and Appendix C bounds |
| HM96 | Halevi–Micali, *Practical and Provably-Secure Commitment Schemes from Collision-Free Hashing*, CRYPTO 1996, LNCS 1109 pp. 201–215 (author's PDF, sections 1.2, 3, 4 read) | statistically hiding, computationally binding string commitment from a collision-free hash: first scheme (section 3: \(L=4k+2n+4\) random bits \(x\), universal hash \(h\) with \(h(x)=m\), commit string \((h,\mathrm{MD}(x))\); Theorem 1: statistical distance \(<2^{-k}\)); \(O(k)\)-bit variant (section 4: commit to \(\mathrm{MD}(m)\), \(9k\)-bit commit string) |

Not read for this document (cited by name only): Goldreich–Kahan, *How to construct constant-round zero-knowledge
proof systems for NP*, J. Cryptology 9(3) 1996 — the verifier-coin-commitment technique and its simulator; the
simulator of section 5.5 is written out so that the argument does not rest on the citation.

## 1. Relation, shapes, public data

| symbol | value / meaning | public? |
|---|---|---|
| profile | `AMPERE_BF16_M16N8K16`, align `div` (`verity.ml.tc.silicon`) | yes |
| \(K\) | dot-product length; decision workload \(K=1536\) | yes |
| \(B\) | instances per proof (padded batch; real count also public); `N_subbatches` independently committed sub-batches → union bound (`union_over_proofs`) | yes |
| \(U=B\cdot K/16\) | transition units (one 16-product update each); \(K=1536\): \(U=96B\) | yes |
| \(b=\lceil\log_2 U\rceil\) | copy bits; \(B\in\{1,64,4096\}\Rightarrow b\in\{7,13,19\}\) | yes |
| \(C\) | fixed per-unit checker, layered quad circuit (libzk 6.3.2), layers \(L_0\) (assertions) … \(L_D\) (inputs), per-unit widths \(2^{s_i}\) | yes |
| \(x_u\in F_t^{2^{s_D}}\) | per-unit private inputs: the 34 words, the 225 hints (decode digits, flags, selectors, quotients/remainders, leading-bit positions, clamp/overflow flags), the 62 looked-up row columns; with `commit_query_wires` (default) also the 140 query wires | **no** |
| \(T_1..T_5\) | public lookup tables of the census (below): `T_range16`, `ALIGN`, `POW`, `LEAD`, `NORM` | yes |
| \(m_T\in\mathbb Z_{\ge0}^{|T|}\) | multiplicities of table \(T\) over the whole batch | **no** |
| census per unit (WP2 checker contracts, `verity_numerical.security.census.census_unit`) **[derivation from source]** | 358 `mul` (114 of them booleans), 137 `lin` (115 identities + 22 range-chunk splits), 306 lookups (284 sixteen-bit range chunks + 22 fixed-table rows); committed columns 321 = 34 words (\(c\), 32 operands, \(y\)) + 225 hints + 62 looked-up row columns; 244 products + 100 materialised range expressions as computed wires; multiplicative depth 5; 140 of the 306 lookup queries are on computed wires | yes |

**Census source (2026-09-21).** The counts are taken from the WP2 checker (`verity_numerical.checker`: `step.py`'s
composed unit executed once on symbolic values through the five predicates `rng`/`eq`/`boolean`/`lookup`/`lt_pow`,
hint widths from every gadget's `contract()`), **not** from `verity.ml.tc.relation.census`, which is forgeable as
written (red team: `magW` never range-checked; WP2 F1/F2: remainders and quotient bounds) and counts none of the
repairs. Conventions (`security.census` docstring): a product of two non-constant expressions is one `mul`;
`boolean` is one `mul`; a \(w\)-bit range check is \(\lceil w/16\rceil\) lookups into the one 16-bit table (a
\(w<16\) value is checked as \(x\cdot2^{16-w}\)) plus one `lin` split when \(w>16\); `lt_pow` is the reference
realisation \(x\cdot\mathrm{ipow}<2^B\) (one `mul` + one \(B\)-bit range; sound over Goldilocks since
\(p>2^{60}\ge2^{2B}\), checker F1 — a small field needs the direct realisation, not counted). Identical products
formed twice (`q*sr` in `div` and `quot`) are counted twice (upper count).

Lookups per unit by table: `T_range16` 284 (65 536 rows, 1 column), `ALIGN` 16 (512 rows, 4 columns), `POW` 2
(512, 3), `LEAD` 2 (31, 3), `NORM` 2 (32, 4). Multiplicity columns \(\sum_T|T|=66\,623\) (relation: 162 715). Range
checks per unit by width: 7-bit 36, 8-bit 66, 9-bit 26, 16-bit 112, 23-bit 2, 24-bit 5, 25-bit 8, 30-bit 6,
32-bit 1. Total lookups at \(K=1536,B=4096\): \(306\cdot393216=120.3\)M; `T_range16` has \(N=111.7\)M entries
(\(n_T=27\)). Hint widths (contract ranges): 114 one-bit (flags, selectors, signs), 2 five-bit, 34 seven-bit,
33 eight-bit, 2 nine-bit, 32 sixteen-bit, 2 twenty-three-bit, 2 twenty-four-bit, 4 twenty-five-bit.

**Diff against `relation.vu_census` (`census.diff_vs_relation`, `test_census.py`)**, checker − relation:

| | relation `mul`/`lin`/`lookup` | checker `mul`/`lin`/`lookup` | Δ |
|---|---|---|---|
| unit (16 products) | 146 / 104 / 172 | 358 / 137 / 306 | +212 / +33 / +134 |
| VU \(K=1536\) | 14 027 / 9 992 / 16 521 | 34 370 / 13 059 / 29 379 | +20 343 / +3 067 / +12 858 |

Not in the relation's count: the direct range of every remainder (`align_state.r`, `align_product.r`,
`normalise.r`) and of the normalised magnitude (`normalise.M`, the `magW` repair); the tight quotient bounds
(`quot` products); the decode of an arbitrary incoming accumulator word in every unit; the pack of the output
word in every unit; operand decode by hints + range checks (the checker uses no `T_bf16`, `T_pcls`, `T_exp`,
`T_iszero` tables — those lookups are replaced by 4 hints, 3 range checks, 2 `mul`, 2 `lin` per operand).

**Two unit variants (`census_unit`, `census_vu`).** The backend transition unit takes an arbitrary finite word
\(c\) (decode + pack inside; the row above). The VU (`check_vu_expr`) is 96 word-chained units from the constant
\(c_0=0\) plus the epilogue: 30 631 committed columns (A), 34 370 `mul`, 13 059 `lin`, 29 379 lookups, i.e.
\(96\times\)unit \(+2\) `mul` \(+6\) hints \(+3\) lookups: the variants differ by \(<0.01\%\) because `step.py`
decodes and packs the word in every unit. A state-chained VU (decode once, pass \((s,e,M)\)) would save
\(\approx14\) `mul`, 7 hints, 9 `lin`, 10 range chunks per inner unit (\(\approx4\%\)); it is not what `step.py`
composes and is not counted.

**Checker shape [derivation from the census; widths are estimates]** (`census.layer_shape`, `DEFAULT_LAYERS_A`):
libzk 6.3 layers are strict, so a value used \(j\) layers later is carried by copy gates; the width of the layer at
multiplicative depth \(d\) is taken as its products plus every value of lower depth (upper bound).

| layer (output first) | products | carried width per unit | \(2^{s_{out}}\) | reads \(2^{s_{in}}\) | holds lookup queries |
|---|---|---|---|---|---|
| assertions (137 `lin`) | 0 | 137 | \(2^8\) | \(2^{10}\) | no |
| depth 5 | 6 | 679 | \(2^{10}\) | \(2^{10}\) | yes |
| depth 4 | 4 | 673 | \(2^{10}\) | \(2^{10}\) | yes |
| depth 3 | 22 | 669 | \(2^{10}\) | \(2^{10}\) | yes |
| depth 2 | 114 | 647 | \(2^{10}\) | \(2^{10}\) | yes |
| depth 1 | 212 | 533 | \(2^{10}\) | \(2^9\) (321 inputs) | yes |

The message shape of every protocol step depends only on the public data of this table; no message length or
branch depends on a witness.

## 2. Fields

| role | choice | reason |
|---|---|---|
| trace field \(F_t\) (inputs, multiplicities, Reed–Solomon code) | Goldilocks \(p=2^{64}-2^{32}+1\) (**64-bit reference first**, handoff 7.2) | \(2^{32}\)-smooth multiplicative group gives RS length up to \(2^{32}\); every integer of the checker (\(|S|<2^{30}\), `lt_pow` products \(<2^{2B}\le2^{60}\), words \(<2^{32}\)) embeds without wrap: checker F1–F4 are satisfied by Goldilocks with the product realisation of `lt_pow` (`checker/__init__.py` section 3) **[source]** |
| challenge field \(F_c\) (all sumcheck/logUp/Ligero-combination randomness, pads) | \(F_{p^2}\) (\(\approx2^{128}\)) first meets \(2^{-80}\) and \(2^{-96}\); \(F_{p^3}\) (\(\approx2^{192}\)) first meets \(2^{-112}\) and \(2^{-128}\) | section 9.2: the logUp identity term \((N+|T|-1)/(|F_c|-|T|)\) floors at \(2^{-111.3}\) (\(B=1\)) … \(2^{-100.9}\) (\(B=4096\)) over \(F_{p^2}\) **[derivation]** |
| small-field study | Baby Bear \(15\cdot2^{27}+1\) / Mersenne-31 trace field, degree-6 extension for \(2^{-128}\) (degree 5 misses: lookup terms \(\approx2^{-127}\)) | \(|S|<2^{30}\) motivates it; logUp's \(\mathrm{char}\,F>N\) (Hab22 Lemma 5) caps total lookups per table at \(<2^{31}\): fine at \(B=4096\) (`T_range16` \(N=1.1\cdot10^8\)), fails at \(B=2^{20}\) (`test_small_field_study_limits`) |

Rule: \(F_c\) is an extension of \(F_t\); pads are \(F_c\) elements stored as \(\deg(F_c/F_t)\) trace-field
coordinates; an \(F_c\)-linear constraint on pads is \(\deg\) \(F_t\)-linear constraints on coordinates, an
\(F_c\)-product is \(\deg^2\) coordinate products plus linear combinations **[derivation]**.

## 3. Commitment interface (Ligero tableau, interactive, AHIV22 4.7 / libzk 4.2–4.3)

| parameter | value (\(2^{-128}\) set) | constraint (AHIV22 / libzk 4.2.1) |
|---|---|---|
| RS code | \([n,k]\) over \(F_t\), \(n=2^{14}\), \(k=2^{12}\), rate \(1/4\), distance \(d=n-k+1\) | \(n>2k\) (quadratic test), \(|F_t|\ge n+l\) |
| payload per row \(l\) | 3840 witness symbols | \(k\ge l+t\) so that any \(t\) opened columns are uniform (AHIV22 Lemma 4.15: \(k>l+t\); libzk: BLOCK = WR + NREQ with queries outside message positions) |
| opened columns \(t\) | 192 reference set (minimum 191, section 9.2); 121 reference / 120 minimum for \(2^{-80}\) | section 9 |
| proximity \(e\) | \((n-k)/2=6144\) (Appendix C regime \(e<d/2\)) | |
| blinding rows | one random codeword row per test (LDT row; linear-test row with payload summing to 0; quadratic-test row with zero payload) — libzk `layout_zk_rows` | |
| leaf \(j\) **[proposal, default]** | HM96 \(O(k)\)-bit commitment to column \(j\): \(s_j=H(T[0][j]\|\cdots\|T[\mathrm{NROW}][j])\), random \(x_j\in\{0,1\}^{6k+4}\), universal \(h_j\) with \(h_j(x_j)=s_j\), leaf \(=(h_j,\,H(x_j))\), \(k=160\); opening \(=\) column \(\|\,x_j\) | hiding of unopened columns is **statistical**, \(n\cdot2^{-160}=2^{-146}\) (HM96 Theorem 1, union over \(n\) leaves; calculator `zk` term); binding by collision resistance of \(H\). Alternative (unsalted \(H(\text{column})\), libzk `commit`) needs a hiding assumption on \(H\) beyond CRH (section 7) |
| committed elements, \(K=1536,B=4096\) | \(181.4\)M \(F_t\) elements = \(393216\cdot(321+140)\) inputs, hints, row columns and query wires \(+\) 66 623 multiplicities \(+\) 4 674 pads \(\times3\) coordinates → 47 231 rows, \(774\)M encoded symbols; \(126.3\)M without the query wires (`commit_query_wires=False`, depth 630) | calculator `parameters` |

Quadratic constraints are laid out libzk-style as aligned triples of rows (`layout_quadratic_rows`); Candidate A
has only the deferred products of section 6 (\(\approx4\) per checker layer, 3 per fractional-GKR layer, times
\(\deg^2\) coordinates), so alignment copies are negligible here (they are not for Candidate B).

## 4. Interactive reductions

### 4.1 Data-parallel GKR over the checker (Tha13 §7, libzk 6.3)

Layer \(i\) has per-unit wires \(V_i(g,c)\), \(g\in\{0,1\}^{s_i}\), \(c\in\{0,1\}^{b}\) the copy index. With the
libzk quad predicate \(Q_i(g,l,r)\) (additions encoded as quads with the constant-1 wire, so one predicate per layer):

\[
\widetilde V_i(z,\gamma)=\sum_{g,c,l,r} \mathrm{eq}(z,g)\,\mathrm{eq}(\gamma,c)\,\widetilde Q_i(g,l,r)\,\widetilde V_{i+1}(l,c)\,\widetilde V_{i+1}(r,c).
\]

**[derivation]** variable degrees \(g:2\), \(c:3\), \(l:2\), \(r:2\); rounds per layer \(s_i+b+2s_{i+1}\);
sumcheck error \((2s_i+3b+4s_{i+1})/|F_c|\). The wiring predicate is copy-independent (Tha13 §7.3), so the prover
tables of \(\widetilde Q_i\) are built once per circuit and shared by all \(U\) copies (Theorem 2 preprocessing).
Binding order **[proposal]**: \(g\) first, then \(c\), then \(l\), \(r\) — the \(c\) rounds are the ones with
\(U\)-parallel prover work, the \(l,r\) rounds are per-circuit only.

Two evaluation claims on layer \(i+1\) (\(\widetilde V_{i+1}(l^*,c^*)\), \(\widetilde V_{i+1}(r^*,c^*)\)) are
merged into one with a fresh \(\alpha\) (libzk 6.3.2; error \(1/|F_c|\)); in-circuit assertions of a layer are
folded with a fresh \(\beta\) (libzk 6.3.3; \(1/|F_c|\)). The top claim is \(\widetilde V_0(z_0)=0\) at a uniform
\(z_0\in F_c^{s_0+b}\): all \(2^{s_0}U\) assertion wires are zero (Schwartz–Zippel, \((s_0+b)/|F_c|\)).

### 4.2 Lookups: logUp-GKR per table (Hab22, PH23)

For table \(T\) with \(N_T=q_T U\) queries \(w_1..w_{N_T}\) (checker wires or inputs, **not separately
committed**) and multiplicities \(m_T\) (committed), the identity **[source, Hab22 Lemma 5]**

\[
\sum_{j=1}^{N_T}\frac{1}{z-w_j}=\sum_{v\in T}\frac{m_T(v)}{z-t_v}
\]

holds as rational functions iff the multisets agree (requires \(\mathrm{char}\,F_c>N_T\)). Vector-valued rows
(columns \(>1\)) are compressed with powers of a second challenge \(\beta_T\) (Hab22 3.4).

**[source, PH23 §3]** The fractional sum is proved by GKR over a hypercube of dimension
\(n_T=\lceil\log_2(N_T+|T|)\rceil\) with leaves \((p,q)\): witness side \((1,\,z-w_j(\beta))\), table side
\((-m_T(v),\,z-t_v(\beta))\), padding \((0,1)\). Layer \(k\) combines \(2^{k+1}\) fractions into \(2^k\) by
\((p,q)\mapsto(p_0q_1+p_1q_0,\;q_0q_1)\); each layer is a degree-3 sumcheck of \(k\) variables (error \(3k/|F_c|\))
followed by a two-point-to-one-point reduction and a \(\lambda\) combination of the \(p\) and \(q\) claims
(\(2/|F_c|\) per layer). The root value is the **cleared numerator** \(p_{\mathrm{root}}(z)=\sum_j p_j\prod_{i\ne j}q_i\);
the verifier asserts \(p_{\mathrm{root}}=0\). **No division occurs anywhere in the protocol**: the tree is
built from products and sums only (section 4.4).

**Challenge space and the conditioned bound.** \(z_T\) is uniform over \(F_c\setminus\mathrm{Img}_T\),
\(\mathrm{Img}_T=\{t_v(\beta_T):v\in T\}\) (public; \(|\mathrm{Img}_T|\le|T|\)), realised by the public redraw
rule of 5.1 step 2a. **[derivation]** \(p_{\mathrm{root}}(X)\) is a nonzero polynomial of degree
\(\le d_T=N_T+|T|-1\) whenever the multisets differ (Hab22 Lemma 5), so it has at most \(d_T\) roots in
\(F_c\setminus\mathrm{Img}_T\), and

\[
\Pr_{z\leftarrow F_c\setminus\mathrm{Img}_T}[p_{\mathrm{root}}(z)=0\mid\text{multisets differ}]
\;\le\;\frac{d_T}{|F_c|-|\mathrm{Img}_T|}\;\le\;\frac{N_T+|T|-1}{|F_c|-|T|}.
\]

This is the bound the calculator charges (`logup_identity_error(..., challenge_space="conditioned")`); it is
also the sampling set of Hab22 Protocol 2 (condition \(x\notin -t(H)\), Remark 3), so Theorem 4's
\(((M+1)|H|-1)/(|F|-|H|)\) is the same statement, not a whole-field theorem reused. The whole-field variant
(\(z\) uniform over \(F_c\), Hab22 Remark 3's overwhelmingly-complete protocol) gives \(d_T/|F_c|\); the two differ
by exactly \(d_T|T|/(|F_c|(|F_c|-|T|))\): at \(K=1536,B=4096\), `T_range16` (\(N_T=111.7\)M), \(F_c=F_{p^3}\), that is \(2^{-341.3}\)
absolute, a relative \(|T|/(|F_c|-|T|)=2^{-176}\) (`test_conditioned_challenge_space_vs_whole_field_theorem`).
For vector-valued rows the \(\beta\)-degree term \(d_T(\mathrm{cols}-1)/|F_c|\) is added **[derivation]**
(conditioning on \(\beta_T\), union bound).

Multiplicities are challenge-independent (counts of witness values) and are committed in message M1 **before**
any challenge. No inverse or product intermediate is committed (PH23's point); all \((p,q)\) layers are
computed by the prover on the fly in \(F_c\).

### 4.3 Linking lookups to the checker **[proposal]**

The final fractional-GKR claim for table \(T\) at \(\rho_T\in F_c^{n_T}\) splits linearly into
(a) a claim on \(\widetilde m_T\) at the table-side coordinates → one Ligero linear constraint on the committed
multiplicities (section 6), and (b) a claim
\(\sum_{g,c}L_T(g,c)\,V_{i(T)}(g,c)=\text{value}\) on the checker layer \(i(T)\) holding all query wires of
\(T\), where \(L_T\) is public (products of \(\mathrm{eq}\) factors and \(\beta_T\) powers).

**Where the queries are (census).** Of the 306 queries per unit, 166 are input-layer values (hints such as
`q`, `r`, `t`, `f`, `M`, `low`, the 33 words) and **140 are computed wires**: the `lt_pow` products
(\(r\cdot\mathrm{ipow}\), \(q\cdot\mathrm{pow}\), \((v-\mathrm{lo})\cdot\mathrm{ispan}\), \(r\cdot\mathrm{idiv}\)), the
`checked` expressions (\(t-nz\), \(M+z-1\), \(u+z-1\), \(v\), \(d_i=E-\mathrm{cand}_i\), `clamp_proof`,
`ovf_proof`) and the table keys \(d\), \(k\). They sit on every computed layer (depths 1–5).

**Layout rule [proposal], default `commit_query_wires`**: each of the 140 computed query wires is *also*
committed as an input column, bound to its expression by the identity the checker already states (the product
or linear identity is a gate of its layer either way), so every claim (b) is a Ligero linear constraint on the
committed \(x\) and the lookups never wait for the checker GKR. Cost: \(+140\) committed elements per unit
(\(321\to461\), \(+55\)M at \(B=4096\)). This satisfies handoff 6.2's "witnesses must agree on the shared query
values": the lookup argument reads the committed copy, and the checker binds the copy to the wire.
**Without it**, claim (b) is merged into the sumcheck of the layer holding the query wire (fresh coefficient,
\(1/|F_c|\) each, degrees unchanged), and every layer that holds a query — all five computed layers — follows the
lookups on the critical path (section 5.4: depth 630 instead of 381).

### 4.4 Every denominator, and why none can abort on witness data

| where | denominator | status |
|---|---|---|
| logUp identity (4.2), leaves \(q_j=z_T-w_j(\beta_T)\), \(q_v=z_T-t_v(\beta_T)\) | never inverted: the fractional GKR computes \(p_{\mathrm{root}},q_{\mathrm{root}}\) by products and sums (PH23 §3) | no division. Honest prover: \(w_j\in T\) and \(z_T\notin\mathrm{Img}_T\) ⇒ every \(q\ne0\), so \(p_{\mathrm{root}}=0\) exactly (completeness). Dishonest \(w_j=z_T\): some \(q_j=0\); the identity check \(p_{\mathrm{root}}=0\) is then the polynomial statement bounded in 4.2 (soundness), nothing to catch separately |
| verifier's evaluation of \(s_j(r_j)\) from \(s_j(0),s_j(1),s_j(2)[,s_j(3)]\) | Lagrange denominators \(\prod_{a\ne b}(a-b)\) over \(\{0,1,2,3\}\) | public constants, nonzero (\(\mathrm{char}>3\)) |
| \(\mathrm{eq}(z,\cdot)\), \(L_T\), \(\beta_T\) powers, \(\mathrm{Lagrange}\) weights of the padded transcript (FS24 §2.3) | none / public constants | public data only |
| Reed–Solomon encoding and the verifier's IRS/linear/quadratic tests | Vandermonde / NTT twiddles over the public evaluation domain (\(\eta_j\), \(\zeta_c\) distinct) | public constants |
| HM96 commitments (leaves, step-0 coins) | none | — |
| two-claims-to-one line reduction, \(\alpha,\beta,\gamma,\mu,\lambda\) combinations | none (multiplications by challenges) | — |
| deferred final-round identities (6) | none (products of padded values) | — |

There is no witness-dependent division in Candidate A; the only zero-test on witness data is the public
\(z_T\in\mathrm{Img}_T\) rule of step 2a, whose inputs (\(z_T\), \(\beta_T\), the table) are all public.

## 5. ZK layer: padded transcript + deferred Ligero verification, made interactive

FS24 Protocol 2.5 is already an interactive protocol (Fiat–Shamir is applied afterwards in FS24 §2.4); libzk
specifies only the non-interactive form. **[proposal]** We run Protocol 2.5's structure with live challenges and
three extensions: data-parallel GKR with \(F_t\) witness / \(F_c\) challenges, logUp-GKR fractional sumchecks
inside the padded transcript, and (optionally) verifier-coin commitments for malicious-verifier ZK. This is
proposed research: FS24 Theorem 6 is stated for honest-verifier ZK of the plain circuit case; nothing below has
a written proof for this composition (section 10).

### 5.1 Numbered transcript (\(K,B\), profile, \(C\), tables, code parameters, \(H\) fixed publicly beforehand)

Notation: `M` prover message, `C` verifier challenge (uniform in \(F_c\) unless stated). `⟨v⟩` denotes a padded
value \(v+\mathrm{pad}\) with a fresh committed pad. With step 0 every `C` line is an *opening* of the
corresponding coin commitment (5.5); without step 0 the coin is simply sent. Challenge **slots** are the `C`
lines instantiated (each \(r_j\), each \(\mu_k,\lambda_k\), …); `rounds.protocol` counts slots,
`rounds.sequential_depth` counts verifier messages on the critical path (5.4).

~~~text
 0  C  [proposal, malicious-verifier ZK; section 5.5]  V samples every coin of steps 2-5 and sends one HM96
       commitment per challenge slot (R = rounds.protocol slots).  Without step 0: COMPLETE_HVZK_BACKEND at most.
 1  M  com_1 = MerkleRoot_H( HM96-leaves( Encode(W_1) ) ),
       W_1 = (x_u)_{u<U} || (m_T)_T || coords(pad) || row randomizers || blinding rows.
       Everything in W_1 is challenge-independent: inputs+hints, ALL multiplicities, ALL pads for steps 2-4,
       Ligero row randomizers and the three blinding rows.  Fixed here: multiplicities, pad count P (public).
 2  --- lookups: one fractional GKR per table T (5 tables, independent challenges, run in parallel) ---
 2a C  for every T at once: z_T^(1), z_T^(2) (redraw slots, S = 2), beta_T if columns(T) > 1; z_0 of step 3a.
       Public rule: z_T := the first z_T^(i) not in Img_T = { t_v(beta_T) }; if all S lie in Img_T the session
       ends (public abort, probability (|T|/|F_c|)^S per table, calculator `completeness`).  No round trip.
 2b M  <q_root>;  p_root := 0 asserted (nothing sent).
 2c    for k = 1 .. n_T-1:
         M  <p_k(rho,0)>, <p_k(rho,1)>, <q_k(rho,0)>, <q_k(rho,1)>      (rho = current point, |rho| = k-1)
         C  mu_k, lambda_k               (must follow the four claims: the prover may not know them earlier)
         k rounds of degree-3 sumcheck:  M <s_j(0)>, <s_j(2)>, <s_j(3)>  ;  C r_j
 2d    leaf claims at rho_T: (a) m~_T(rho_T|table)  -> deferred linear constraint (step 6)
                             (b) sum_c beta^c w~^{(c)}(rho_T|witness) -> deferred linear constraint on x when
                                 the queries are committed (layout rule 4.3); else pending claim on layer i(T)
 3  --- checker GKR, layer by layer (in parallel with step 2 when no pending claim is outstanding) ---
 3a    claim V~_0(z_0) = 0, z_0 in F_c^{s_0+b} (issued in 2a).
 3b    for i = 0 .. D-1:
         C  alpha_i (two incoming claims -> one), beta_i (assertions), one coefficient per pending lookup
            claim on layer i                        (must follow the previous layer's <v_l>, <v_r>)
         s_i + b + 2 s_{i+1} rounds:  M <s_j(0)>, <s_j(2)> (+<s_j(3)> in the b degree-3 rounds);  C r_j
         M  <v_l> = <V~_{i+1}(l*,c*)>,  <v_r> = <V~_{i+1}(r*,c*)>
 4  C  gamma: input-layer claims (v_l, v_r, lookup claims on inputs) -> deferred linear constraints.
 5  --- deferred verification = Ligero ZKIPCP, interactive (AHIV22 4.7 steps 1-3) ---
 5a C  r in F_c^{rows} (explicit vector), r_add in F_c^{#linear}, r_q in F_c^{#quadratic}
 5b M  v = r^T U + u_0 in F_c^n ;  q_add(.) of degree < k+l-1 ;  p_0(.) of degree < 2k-1
 5c C  Q subset [n], |Q| = t
 5d M  columns T[.][j], j in Q, with HM96 openings x_j and Merkle paths
 6  V  checks: HM96 openings and Merkle paths; IRS test on v; sum_c q_add(zeta_c)=0 and column consistency
       (all linear constraints of section 6); p_0(zeta_c)=0 and column consistency (quadratic constraints);
       every step-0 opening verified as it arrived; accept/reject.
~~~

What is fixed when: inputs, hints, multiplicities, pads, randomizers, blinding rows — step 1, before any
challenge. No challenge-dependent value is ever committed; every challenge-dependent prover message is a padded
transcript element or a Ligero test response (blinded by the step-1 blinding rows). Verifier coins (with step
0) are fixed before step 1.

### 5.2 Masked values and how masks are bound

| value | mask | binding |
|---|---|---|
| every sumcheck evaluation \(s_j(\cdot)\) (steps 2c, 3b) | one \(F_c\) pad each, committed in W_1 | libzk 6.5: verifier's check \(s_j(0)+s_j(1)=s_{j-1}(r_{j-1})\) becomes a linear constraint on pads with public coefficients (Lagrange weights at \(r_{j-1}\)); accumulated symbolically per layer (FS24 §2.3) |
| layer evaluations \(v_l,v_r\); fractional-GKR \(p_k,q_k\) at two points; \(q_{\mathrm{root}}\) | one pad each | final-round identity \(s_{\text{last}}(r)=\widetilde Q\cdot v_l v_r\) (resp. \(p_0q_1+p_1q_0+\lambda q_0q_1\)) → FS24 Claim: linear constraints plus one quadratic constraint per product on pads (section 6) |
| leaf claims on \(\widetilde m_T\), input claims | pad each | linear constraint: ⟨public eq-vector, committed \(m_T\) or \(x\)⟩ − pad = public value |
| committed rows (inputs, multiplicities, pads) | \(k-l\) random symbols per row | any \(t\) columns uniform (AHIV22 Lemma 4.15) |
| Ligero responses \(v, q_{add}, p_0\) | blinding rows \(u_0, u_{add}, u_q\) | AHIV22 4.6.1–4.6.3 (affine tests, Appendix B.4–B.5) |
| unopened columns | HM96 leaf commitment, \(k=160\) | statistical, \(2^{-146}\) over all leaves (section 3) |

Pads are never reused: one per transcript element, indexed by (step, table/layer, round, evaluation point);
the index map is public and fixed by \((K,B,C)\). A pad reused across two elements would leak their difference.

### 5.3 Deferred checks (all evaluated inside the Ligero tests of step 5–6)

| check | Ligero constraint type | count (\(K=1536,B=4096\)) **[derivation]** |
|---|---|---|
| sumcheck round consistency \(s_j(0)+s_j(1)=s_{j-1}(r_{j-1})\) | linear over \(F_c\) → \(\deg\) linear over \(F_t\) | 1 569 |
| final-round identity per checker layer (\(\widetilde Q\, v_l v_r\)) | 1 quadratic + linear | 6 |
| final-round identity per fractional-GKR layer (3 products) | 3 quadratic + linear | \(3\sum_T(n_T-1)=315\) |
| claim merges (\(\alpha,\beta\), lookup coefficients, \(\gamma\), \(\mu,\lambda\)) | linear | one per coefficient |
| \(\widetilde m_T\) leaf claims | linear on committed \(m_T\) | 5 |
| input-layer claims | linear on committed \(x\) | 2 + input-layer lookups |
| \(p_{\mathrm{root}}=0\) | folded into layer-0 identity | 5 |

The verifier's own work: HM96 openings and Merkle paths for \(t\) columns, the IRS/linear/quadratic tests on
\(t\) columns (\(t\cdot\mathrm{rows}\) = 9.1M \(F_t\) elements opened at \(B=4096\)), evaluation of
\(\widetilde Q_i\) at the random points, and the public eq/Lagrange coefficient vectors. Proof size and
verifier time are secondary (handoff 2.1) and are reported, not optimised.

### 5.4 Protocol rounds vs. sequential challenge depth (latency axis)

Definitions (contract names). `rounds.protocol` = number of challenge slots of the transcript (every `C` line
instantiated; a naive one-challenge-per-message implementation needs this many verifier messages).
`rounds.sequential_depth` = verifier messages on the critical path after every available parallelisation: a
verifier message costs one network round trip only if the next prover message depends on it. The network floor
is `depth × RTT`; the final prover message (5d) adds a one-way latency not counted.

**Known techniques, always applied [source / immediate]:**
1. Independent sumchecks run in parallel: all 5 fractional GKRs and the checker GKR are independent instances
   (the lookups touch the checker only through the deferred constraints or a merged claim), so one verifier
   message carries all their round-\(j\) challenges and one prover message all their round-\(j\) polynomials
   (Tha13 §7 / libzk batch the *instances*; running independent protocols in lock-step is the standard
   observation that a public-coin verifier's coins for independent sub-protocols may be sent together).
2. Coins that do not depend on a prover message travel with the previous challenge: all \(z_T,\beta_T,z_0\) and
   the \(S\) redraw slots go in one message (2a); the redraw costs no trip.
3. Coins that **must** follow a prover message stay separate: \(\mu_k,\lambda_k\) after the four claims (else
   the prover chooses claims knowing the combination point — the two-claims-to-one reduction is sound only for
   a coin chosen after the claims), \(\alpha_i,\beta_i\) after \(v_l,v_r\), each \(r_j\) after \(s_j\), \(Q\)
   after the Ligero responses.

**Derivation of the baseline depth [derivation]**, \(K=1536,B=4096\), \(c=1\) variable per message, query wires
committed (layout rule 4.3), no step 0:

~~~text
fractional GKR of dimension n (per table, all tables in parallel):
    1 (mu_1, lambda_1 after the layer-1 claims) + sum_{k=1}^{n-1} (k sumcheck challenges + 1 (mu,lambda))
    = 1 + n(n-1)/2 + (n-1)                       n = 27 (T_range16, N = 111,673,344 + 65,536 <= 2^27):  378
checker GKR (6 layers, rounds s_out + 19 + 2 s_in = 47, 49, 49, 49, 49, 47):
    sum (rounds_i + 1) + 1 (gamma) = 296 + 1 = 297                                             (in parallel)
depth = 1 (message 2a: all z, beta, z_0) + max(378, 297) + 2 (Ligero r, Q) = 381
~~~

Without committed query wires every computed layer holds queries and waits for the lookups: depth
\(=1+\max(378,\,\text{assertion layer }48)+(5\text{ layers: }49+50+50+50+48)+1+2=630\).

**Proposals [proposal] (calculator `DepthOptions`), each with its cost:**

| reduction | mechanism | depth at \(B=4096\) | cost | status |
|---|---|---|---|---|
| \(c\) variables per message | prover sends the \(c\)-variate round polynomial (\((d+1)^c-1\) padded values); verifier binds \(c\) variables per message; per-message Schwartz–Zippel error \(cd/|F_c|\), total unchanged **[derivation]** | \(c=2\): 212; \(c=3\): 156 | pads \(\times(4^c-1)/(3c)\) (\(c=2\): \(2.5\times\), \(+7\)k pads); prover work in the sumcheck rounds \(\times\approx(d+1)^{c}/((d+1)c)\) (\(c=2\): \(\approx2\times\)) | folklore generalisation of the sumcheck round (no source read for it; derivation only) |
| table sharding \(s\) | split each table's \(N_T\) queries into \(2^s\) independent fractional GKRs with their own committed multiplicity vectors (challenge-independent); dimension \(\lceil\log_2(N_T/2^s+|T|)\rceil\); identity error charged per shard | \(s=5\): 300 (the checker chain, 297, is now the floor) | \((2^s-1)|T|\) extra committed multiplicities per sharded table (\(s=5\): \(+2.3\)M elements, \(+1.3\%\)); \(2^s\times\) the (padded) fractional transcript for that table (\(+3.3\) MB communication) | this lane's proposal |
| both | | \(c=2,s=5\): **158**; \(c=3,s=5\): 110 | sum of the above | |
| commit the 140 query wires (4.3) | default; listed here because it is the depth lever with the largest committed-element cost | 381 vs 630 | \(+55\)M committed elements (\(126.3\)M → \(181.4\)M, \(+44\%\)) | this lane's proposal |

Step 0 adds exactly one verifier message before com_1 (+1 depth) in every row.

**RTT sweep [derivation], \(K=1536,B=4096\), prover and verifier compute treated as given (\(t_{\text{compute}}\)),
network floor `depth × RTT` in seconds (`e2e_sweep`):**

| configuration | `rounds.protocol` | `rounds.sequential_depth` | `t.e2e.rtt_0ms` | `rtt_0.1ms` | `rtt_1ms` | `rtt_10ms` | `rtt_50ms` |
|---|---|---|---|---|---|---|---|
| A baseline (known techniques, query wires committed) | 1 589 | 381 | \(t_c\) | \(t_c+0.038\) | \(t_c+0.38\) | \(t_c+3.81\) | \(t_c+19.1\) |
| A without committed query wires (5 layers after the lookups) | 1 589 | 630 | \(t_c\) | \(+0.063\) | \(+0.63\) | \(+6.30\) | \(+31.5\) |
| A + \(c=2\) [proposal] | 1 589 | 212 | \(t_c\) | \(+0.021\) | \(+0.21\) | \(+2.12\) | \(+10.6\) |
| A + \(s=5\) [proposal] | 25 548 | 300 | \(t_c\) | \(+0.030\) | \(+0.30\) | \(+3.00\) | \(+15.0\) |
| A + \(c=2\), \(s=5\) [proposal] | 25 548 | 158 | \(t_c\) | \(+0.016\) | \(+0.16\) | \(+1.58\) | \(+7.9\) |
| A + \(c=3\), \(s=5\) [proposal] | 25 548 | 110 | \(t_c\) | \(+0.011\) | \(+0.11\) | \(+1.10\) | \(+5.5\) |
| A + step 0 (any row) | +1 | +1 | | | | | |
| B `helper_columns` (for comparison) | 7 | 3 | \(t_c\) | \(+0.0003\) | \(+0.003\) | \(+0.03\) | \(+0.15\) |
| B `gkr` mode | 1 569 | 381 | as A baseline | | | | |

At \(B=1\) and \(B=64\) the A baseline depth is 228 and 264 (`rounds_and_depth_a`): there the **checker chain**
(225, 261: six layers whose sumchecks have \(b=7,13\) copy variables) dominates the lookup chain (153, 231;
\(n_T=17,21\)). At \(B=4096\) the lookup chain dominates (378 vs 297). Whether the depth reductions are adopted is a
measurement question (`backends/AGENTS.md`: "no hard round budget yet").

### 5.5 Step 0: verifier-coin commitment for malicious-verifier privacy **[proposal]**

An implementation **without** this step is `COMPLETE_HVZK_BACKEND`, never `COMPLETE_ZK_BACKEND`.

**Commitment scheme.** HM96 first scheme with security parameter \(k_V=160\) for every coin slot \(i\) of
length \(\ell_i\) bits: V draws \(x_i\in\{0,1\}^{L_i}\), \(L_i=4k_V+2\ell_i+4\), a universal hash
\(h_i:\{0,1\}^{L_i}\to\{0,1\}^{\ell_i}\) with \(h_i(x_i)=\mathrm{coin}_i\), and sends
\(c_i=(h_i,\,H(x_i))\); the opening is \(x_i\) (the receiver recomputes \(\mathrm{coin}_i=h_i(x_i)\) and checks
\(H(x_i)\)). For long coins (the Ligero row-combination vector \(r\in F_c^{\mathrm{rows}}\), 630 KB; the query
set \(Q\)) the \(O(k)\) variant is used: commit to \(H(\mathrm{coin}_i)\) and open with \((\mathrm{coin}_i,x_i)\)
(HM96 section 4; 9k-bit commit string, 180 B). Statistically hiding: distance \(<2^{-k_V}\) per commitment
(HM96 Theorem 1), \(R\cdot2^{-160}=2^{-149.4}\) over the \(R=1589\) slots of the baseline (calculator term
`commitment_hash / verifier coin commitments`, statistical, **added to the soundness total**: it bounds what a
prover can learn about the coins). Computationally binding: two openings of one \(c_i\) are a collision of
\(H\) (HM96 section 3.2). Only collision resistance of \(H\) is used; no new assumption enters.

**What is committed.** Every coin of steps 2–5 in transcript order: for each table and redraw slot
\(z_T^{(i)}\), \(\beta_T\); \(z_0\); every \(\mu_k,\lambda_k\); every sumcheck challenge \(r_j\) (or the
\(c\)-tuple); every \(\alpha_i,\beta_i\) and lookup-merge coefficient; \(\gamma\); the Ligero \(r\) vector;
\(r_{add},r_q\) (as one uniform element \(\rho\) each when the powers form is used, section 6); \(Q\). The
slot list is a public function of \((K,B,C)\); its length is `rounds.protocol`.

**Transcript order.**

~~~text
 0   V -> P   (c_1, ..., c_R)                                  all coin commitments, before anything else
 1   P -> V   com_1                                            as 5.1 step 1
 i   V -> P   x_i  (opening of slot i, in the order of 5.1)    P checks H(x_i) = y_i, sets coin_i = h_i(x_i);
                                                               on failure P aborts (publicly attributable to V)
     P -> V   the prover message that depends on coin_i        never sent before the opening of coin_i verified
 ... every C line of 5.1 is such an opening; parallel slots (2a; the 5 tables' round-j coins) open together
 5d  P -> V   opened columns                                   after the opening of Q verified
~~~

**Binding requirement and why it suffices against an adaptive verifier.** A verifier who wants coin \(i\) to
depend on the prover's messages must open \(c_i\) to a value other than the one fixed in step 0, i.e. find
\(x_i'\ne x_i\) with \(H(x_i')=H(x_i)\). Under collision resistance this happens with negligible probability,
so every coin the prover accepts equals the one sampled before com_1: the verifier's only remaining freedom is
**whether** to open the next slot (abort) — the sampled coins themselves are a fixed function of the verifier's
initial randomness and are independent of every prover message. Nothing about the *distribution* of the
verifier's coins is required for privacy (a malicious verifier may pick them arbitrarily); privacy needs only
that they are fixed before the first prover message. Soundness needs the coins to be unpredictable to the
prover, which is what statistical hiding gives.

**Abort / restart rules.**
- P aborts iff an opening fails to verify (\(H(x_i)\ne y_i\)), a coin is malformed (not in its domain), or all
  \(S\) redraw slots of some table lie in \(\mathrm{Img}_T\). Each condition is computed from public data
  (\(c_i,x_i\), tables, \(\beta_T\)) only; none reads the witness. The prover **never** decides to abort on any
  other condition and never restarts within a session.
- V aborts iff a prover message fails a check. A new session recommits with fresh pads, randomizers and fresh
  coins; sessions are counted by the outer accounting (handoff 7.3).
- No restart-on-bad-challenge exists: the redraw of 2a is resolved inside the pre-issued slots.

**Privacy argument (sketch; the proof is an unproved obligation, section 10).** Let \(V^*\) be any verifier.
The view of \(V^*\) is (com_1, then padded/blinded prover messages, then \(t\) opened columns) for the coins it
committed to, or a prefix of it if it aborts. Simulator: (1) run \(V^*\) to obtain \((c_1..c_R)\); (2) send a
simulated com_1 (a commitment to a random tableau: rows of random symbols — indistinguishable from the real one
by the statistical hiding of the HM96 leaves and the uniformity of the \(t\) opened columns, AHIV22 Lemma
4.15); (3) receive \(V^*\)'s openings slot by slot; if \(V^*\) aborts, output the prefix; else all coins are now
known; (4) rewind \(V^*\) to after step 0, run the **honest-verifier simulator** of the underlying protocol
(AHIV22 Lemma 4.15 for the tableau and tests, FS24 Theorem 6 for the padded sumcheck transcript: every padded
value is uniform, the deferred constraints are satisfied by construction of the simulated pads) on the known
coins to produce com_1' and all prover messages, and replay; \(V^*\) must open the same coins (binding) or
abort; (5) if it aborts in the replay but not in the first run, repeat step 4 with fresh simulator randomness;
the expected number of repetitions is bounded as in Goldreich–Kahan (the abort probability depends on com_1
only through a statistically hiding commitment, so it differs negligibly between the first run and the
replay). Inherited from the HVZK proof: everything except steps (1)–(3) and (5); the new ingredients are the
binding of HM96 (computational) and the hiding of com_1 against the abort decision. The resulting notion is
computational zero knowledge under collision resistance, with statistical loss \(n\cdot2^{-160}\) (leaves) +
Ligero's \(0\) (Lemma 4.15 is perfect for \(k>l+t\)).

**Costs (\(K=1536,B=4096\), calculator `census_estimate_a(coin_commitment_bits=160)`):**

| item | value |
|---|---|
| additional verifier messages / depth | +1 (message 0); every other opening rides on the challenge message it replaces |
| additional communication | \(R\) commit strings + \(R\) openings \(\approx R\cdot(144+100)\) B \(=0.39\) MB for \(R=1589\) (74.7 MB → 75.1 MB, \(+0.5\%\)); long coins (\(r\), \(Q\)) add only their \(x_i\) |
| additional prover work | verify \(R\) openings: \(R\) hashes of \(\le 900\)-bit strings and \(R\) universal-hash evaluations — \(<10^7\) bit operations, below \(10^{-6}\) of the batch's \(1.2\cdot10^{11}\) field operations (section 9.4); no change to encoding, GKR or Ligero work |
| additional verifier work | sample coins up front (unchanged total), \(R\) commitments |
| soundness | \(+R\cdot2^{-160}=2^{-149.4}\), total unchanged at \(2^{-130.2}\) (`test_coin_commitment_step0_terms_and_label`) |

## 6. Constraint generation (libzk 6.7 adapted)

Public inputs to constraint generation: all challenges, all padded values, \(\widetilde Q_i\) at the challenge
points, table evaluations \(\widetilde t_v(\beta_T)\) at \(\rho_T\). Output: matrices \(A\) (linear) and triples
(quadratic) over \(F_c\) on the committed vector \(W_1\), expanded to \(F_t\) coordinates. Both parties compute
them; the prover needs them only for step 5b. Combination coefficients: \(r\) (IRS test) is an explicit uniform
vector; \(r_{add}\), \(r_q\) may be explicit (Candidate A has \(\approx10\)k linear and \(\approx1\)k quadratic
constraints) or powers of one uniform element each (`ligero_error(combination="powers")`, error
\((\#\text{constraints}-1)/|F_c|\) by Schwartz–Zippel **[derivation]**; required for Candidate B).

## 7. Hash properties assumed, and where

| property of \(H\) | used where | consequence if it fails |
|---|---|---|
| **collision resistance** (family chosen publicly before step 1, per AHIV22 5.1) | Merkle tree over HM96 leaves: binding of com_1; step 5d openings; binding of the step-0 coin commitments | soundness: a prover opening inconsistent columns (`commitment_hash` term, generic \(q^2/2^h\) for \(q\) hash evaluations — a modelling choice, section 9); ZK with step 0: a verifier re-opening a coin |
| **hiding of unopened columns** | HM96 leaf commitments (default, section 3): statistical \(<2^{-160}\) each, from collision resistance alone | ZK only; if the unsalted alternative \(H(\text{column})\) is used instead, a hiding assumption on \(H\) beyond CRH is required and must be stated by the implementation |
| **PRG** (hash-based expansion of a seed into pads, row randomizers, blinding rows, HM96 \(x_j\) on the GPU) | prover randomness generation | ZK only; at the ideal level all randomness is uniform (handoff 7.4). Recorded as a separate assumption |
| statistically hiding, computationally binding commitment for verifier coins (step 0) | HM96 from the same \(H\) | hiding failure → soundness (prover learns coins; charged \(R\,2^{-160}\)); binding failure → ZK (verifier adapts coins) |
| **not assumed**: random oracle, correlation intractability, extractability of \(H\) | — | Fiat–Shamir is not used; knowledge soundness of the argument is via AHIV22's extractor for the interactive variant, which uses collision resistance only (section 10 records it as unproved for *this* composition) |

## 8. Leakage rules

| leak channel | rule |
|---|---|
| branch-dependent message shapes (e.g. shift direction, zero/overflow flags) | every checker branch is fixed-size (selectors + both arms evaluated); number and degree of every sumcheck message is a function of \((K,B,C)\) only |
| batch partitioning by exponent / group state | forbidden; one batch, one GKR, one commitment; padding instances are public and prescribed (handoff 9.1). Sub-batches, if used, are a public split with `N_subbatches` reported and union-bounded |
| lookup multiplicities | committed, never sent; only \(\widetilde m_T(\rho)\) padded |
| logUp pole (\(z_T-w_j=0\)) | honest query values lie in \(T\), so a pole exists iff \(z_T\in\mathrm{Img}_T\): a **public** condition resolved by the pre-issued redraw slots (2a); the transcript records which slot was used deterministically; the prover never aborts on witness data (calculator `completeness`: redraw \(\sum_T|T|/|F_c|\approx2^{-176}\), abort \(\sum_T(|T|/|F_c|)^2\approx2^{-352}\)) |
| any other zero denominator | none exists (section 4.4) |
| restart on a bad challenge | forbidden. A session is aborted, never restarted with the same W_1; a new session recommits with fresh pads and randomizers (and the outer protocol counts sessions, handoff 7.3) |
| timing / message length | all messages fixed length; prover work per round data-independent (fixed-shape kernels) |
| verifier-adaptive challenges | with step 0, challenges are fixed before com_1 (5.5); without it, only honest-verifier ZK is claimed (`COMPLETE_HVZK_BACKEND`) |
| Merkle root of a deterministic layout | rows randomized (\(k-l\) fresh symbols); leaves are HM96 commitments (section 3) |
| Ligero responses \(v,q_{add},p_0\) | blinded (section 5.2); the linear/quadratic response polynomials have degree \(<k+l-1\), \(<2k-1\) and hide the payload by the blinding rows' randomness |

## 9. Soundness error decomposition (calculator output)

Soundness is a parameter (`backends/AGENTS.md`): the per-proof target is derived from the application budget
(`per_proof_target`), sub-batches are union-bounded (`union_over_proofs`). All figures: \(K=1536\), Goldilocks
trace field, Ligero \(n=2^{14},k=2^{12},l=3840\), hash 256-bit, adversary \(q=2^{60}\) hash evaluations unless
stated. Tests in `backends/numerical/tests/security/test_accounting.py`.

### 9.1 Reference decomposition, \(B=4096\)

| component | term | \(2^{-128}\) set: \(F_c=F_{p^3}\), \(t=192\), \(q=2^{60}\) | \(2^{-80}\) set: \(F_c=F_{p^2}\), \(t=121\), \(q=2^{64}\) |
|---|---|---|---|
| batching | \(z_0\) Schwartz–Zippel \((s_0+b)/|F_c|\); \(\alpha,\beta\) per layer; \(\gamma\); 5 lookup merges | \(2^{-186.7}\) | \(2^{-122.7}\) |
| sumcheck | \(\sum_i(2s_i+3b+4s_{i+1})/|F_c|\), 290 rounds (6 census layers) | \(2^{-182.6}\) | \(2^{-118.6}\) |
| lookup | \(\sum_T\big[(N_T+|T|-1)(\tfrac1{|F_c|-|T|}+\tfrac{\mathrm{cols}-1}{|F_c|}) + \tfrac{2(n_T-1)+1+\sum_{k<n_T}3k}{|F_c|}\big]\), conditioned challenge space; `T_range16` \(N=111.7\)M dominates | \(2^{-164.9}\) | \(2^{-100.9}\) |
| encoding/opening | \((1-e/n)^t+((k+l)/n)^t+(2k/n)^t+(n+3)/|F_c|\) (AHIV22 App. C, FS24 §2.2) | \(2^{-130.2}\) | \(2^{-82.0}\) |
| commitment/hash | \(q^2/2^{256}\); with step 0 also \(R\cdot2^{-160}=2^{-149.4}\) | \(2^{-136}\) | \(2^{-128}\) |
| **total** | | **\(2^{-130.2}\)** (\(B\in\{1,64,4096\}\) all pass) | **\(2^{-82.0}\)** |
| zk (not soundness) | HM96 leaves \(n\cdot2^{-160}\) | \(2^{-146}\) | \(2^{-146}\) |
| completeness (not soundness) | redraw \(\sum_T|T|/|F_c|\); abort \(\sum_T(|T|/|F_c|)^2\) | \(2^{-176.0}\); \(2^{-352}\) | \(2^{-112.0}\); \(2^{-224}\) |

The WP2 census moved the lookup term by \(2^{-0.1}\) (`T_range16` \(N\): 18.1M → 111.7M, but the 13 other
relation tables are gone) and the sumcheck term by \(2^{+0.8}\) (290 rounds instead of 154); no operating point
moved (9.2).

### 9.2 Operating points: which configuration first meets each (`first_configuration`, both candidates identical)

Sweep order: extension degree 2, then 3, then 4; within a degree the smallest \(t<k-l\). \(q=2^{60}\).

| operating point | first configuration | \(B=1\) | \(B=64\) | \(B=4096\) | why the smaller extension fails |
|---|---|---|---|---|---|
| \(2^{-80}\) (diagnostic) | **quadratic** \(F_{p^2}\), \(t=120\) | \(2^{-81.4}\) | \(2^{-81.4}\) | \(2^{-81.4}\) | — |
| \(2^{-96}\) (sweep) | **quadratic** \(F_{p^2}\), \(t=144\) | \(2^{-97.6}\) | \(2^{-97.6}\) | \(2^{-97.5}\) (A), \(2^{-97.2}\) (B) | — (lookup \(2^{-100.9}\) at \(B=4096\) still below) |
| \(2^{-112}\) (sweep) | **cubic** \(F_{p^3}\), \(t=167\) | \(2^{-113.2}\) | \(2^{-113.2}\) | \(2^{-113.2}\) | quadratic: lookup identity \(2^{-111.3}\) (\(B=1\)) … \(2^{-100.9}\) (\(B=4096\)) alone exceeds the target |
| \(2^{-128}\) (primary) | **cubic** \(F_{p^3}\), \(t=191\) (reference set \(t=192\)) | \(2^{-129.5}\) | \(2^{-129.5}\) | \(2^{-129.5}\) | quadratic: same floor, \(2^{-111.3}\) at best |

The quadratic Goldilocks extension cannot reach \(2^{-128}\) at any \(B\): the lookup identity term is set by the
\(65\,536\)-row range table (\(d_T/(|F_c|-|T|)\ge 2^{16}/2^{128}\)). The WP2 census did not move any first
configuration (`test_first_configuration_per_operating_point`, unchanged parameters). Rounds and committed elements scale linearly in
\(U\); soundness terms only logarithmically. The hash term is a modelling choice (adversary work \(q\)),
reported separately as `computational`; `security_bits(field)` is refused.

### 9.3 Sub-batches

A batch of 4096 proved as `N_subbatches` = 16 independent proofs of 256 multiplies every term by 16: a per-proof
\(2^{-130.2}\) becomes \(2^{-126.2}\) and misses \(2^{-128}\) (`test_sub_batches_are_union_bounded_mechanically`);
the per-VU cost includes every sub-batch.

### 9.4 Census, \(K=1536\) (`census_estimate_a` / `census_estimate_b`; **every row marked (est.) is an estimate
from the stated cost model**, `backends/AGENTS.md` columns)

Cost model (est.): \(F_{p^3}\) multiplication = 6 \(F_t\) multiplications; NTT butterfly = 1 mult-equivalent;
checker GKR 12 \(F_c\) mults per (wire, copy, layer); fractional GKR 15 \(F_c\) mults per leaf-equivalent;
Ligero tests \(3\cdot\mathrm{rows}\cdot n\cdot\deg\); memory passes: 3 over the committed matrix (read witness,
write codewords, hash) + 2 per sumcheck table; peak memory under the *retained* policy (codewords + witness +
largest live sumcheck table). B's rows: helpers by batch inversion, 3 \(F_c\) mults each, 2 passes.

Per-unit inputs to the model (WP2 census, section 1): A commits \(321+140\) elements per unit and computes 344
wires; B commits 1 064 per unit (3 aligned slots per `mul` + one per other column; layout **estimate**) plus 306
helper columns per unit in \(F_c\) (helper-column mode), with \(137+2\cdot358=853\) linear and 358 quadratic
constraints per unit. Primary configuration \(F_c=F_{p^3}\), \(t=192\) (\(2^{-128}\)); the \(2^{-80}\) diagnostic
(\(F_{p^2}\), \(t=121\)) is the last row group.

| column | A \(B=1\) | B \(B=1\) | A \(B=64\) | B \(B=64\) | A \(B=4096\) | B \(B=4096\) |
|---|---|---|---|---|---|---|
| materialized witness elements (est.; A: inputs + all wires + fractional-tree nodes; B: committed + helpers) | 1.68M | 0.46M | 31.4M | 12.4M | 2.01G | 780M |
| committed elements (`census.committed_elements`) | 116k | 457k | 2.91M | 12.4M | **181M** | **780M** |
| bytes encoded (`census.bytes_encoded`) | 4.5 MB | 16 MB | 100 MB | 425 MB | 6.19 GB | 26.6 GB |
| hash/commit work (`census.hash_bytes`, est.) | 7.9 MB | 23 MB | 103 MB | 432 MB | 6.19 GB | 26.6 GB |
| field operations (`census.field_ops`, \(F_t\) mult-eq., est.) | 45M | 39M | 1.90G | 1.03G | 121G | 64G |
| memory passes / bytes moved (`census.memory_passes` / `census.bytes_moved`, est.) | 25 / 43 MB | 5 / 56 MB | 25 / 1.53 GB | 5 / 1.47 GB | 25 / 97 GB | 5 / 92 GB |
| sequential challenge depth (`rounds.sequential_depth`; `rounds.protocol`) | 228; 579 | 3; 7 | 264; 968 | 3; 7 | 381; 1 589 | 3; 7 |
| communication (`transcript.bytes`, est.; P→V / V→P) | 0.99 MB (0.98 / 0.015) | 1.18 MB (1.18 / 0.003) | 2.16 MB (2.12 / 0.04) | 6.05 MB (5.97 / 0.08) | 74.7 MB (73.5 / 1.2) | 318 MB (313 / 4.9) |
| peak memory (retained, est.) | 6.9 MB | 20 MB | 223 MB | 525 MB | 14.0 GB | 32.8 GB |
| \(2^{-80}\) diagnostic (\(F_{p^2}\), \(t=121\)): committed / bytes encoded / field ops / peak | 114k / 4.3 MB / 25M / 6.3 MB | 361k / 12.7 MB / 26M / 15.6 MB | 2.90M / 100 MB / 1.02G / 190 MB | 10.5M / 359 MB / 0.71G / 443 MB | 181M / 6.19 GB / 65G / 11.9 GB | 659M / 22.5 GB / 45G / 27.8 GB |

Against the relation-based table (101M vs 405M at \(B=4096\)): A's committed elements rose \(1.8\times\) (321 + 140
query wires per unit instead of the assumed 256), B's \(1.9\times\) (1 064 slots + 306 helpers per unit instead of
512 + 172). The committed-elements ratio A:B is \(1:4.3\) at \(B=4096\) (\(1:4.0\) before), \(1:6.2\) without the
committed query wires (126M vs 780M, depth 630); it stays an explanatory variable of the encoding, hashing,
communication and memory columns. The field-operation column now favours B (A: \(2\times\) B): the checker GKR
runs over \(\approx3\,300\) carried values per unit per layer under the carry-everything width bound of section 1,
which is an upper bound a real layout would shrink; the depth column separates them in the other direction
(381 vs 3). No verdict is drawn here.

## 10. Unproved obligations (verbatim list for the coordinator)

1. Knowledge soundness of the interactive composition (data-parallel GKR + logUp-GKR + padded transcript +
   Ligero): FS24 Theorem 6 covers a plain layered circuit with the Ligero extractor; the fractional-GKR
   sub-transcripts, the lookup-claim merging of 4.3 and the \(F_t\)/\(F_c\) coordinate expansion are not covered.
2. Simulation (zero knowledge): only honest-verifier ZK is inherited (AHIV22 Lemma 4.15, FS24 Theorem 6). The
   verifier-coin-commitment step 0 (5.5) for malicious-verifier ZK is proposed with a simulator sketch, not
   proved, for this protocol; the Goldreich–Kahan expected-time analysis is cited, not read; its cost is
   estimated (5.5), not measured. An implementation without step 0 is `COMPLETE_HVZK_BACKEND`.
3. If the unsalted leaf hash is used instead of HM96 leaves (section 3), hiding of the Merkle leaves is an
   assumption on \(H\) beyond collision resistance.
4. The proximity-gap / IRS statement for a subfield RS code (\(F_t\)) tested with combination coefficients in
   the extension \(F_c\): the calculator charges the Appendix C field terms over \(|F_c|\); the corresponding
   lemma for this exact setting is not cited. When \(r_{add},r_q\) are powers of one element, the Schwartz–Zippel
   replacement of the linear/quadratic field terms is this lane's derivation, and the IRS test keeps uniform
   coefficients.
5. Extraction/composition with Verity's outer commitments (handoff 7.5): W_1 is a fresh commitment to the
   operands; equality with the values fixed by Verity's earlier commitments is a separate proof obligation, and
   the outer draft's extraction/composition requirements for the numerical backend remain unproved.
6. Integer semantics inside \(F_t\) (handoff 7.2): every range bound that makes a field congruence an integer
   equality is a checker obligation (WP2), not established here.
7. Application-level accounting: the number of sessions/batches an adversary may open multiplies the error
   budget (handoff 7.3); `per_proof_target` computes the per-proof allowance once that number is fixed.
8. The checker shape (`DEFAULT_LAYERS_A`) is derived from the WP2 census with carry-everything layer widths (an
   upper bound; the true widths depend on the WP3 layout); the layout rule of 4.3 (commit the 140 query wires) is a
   proposal with a stated cost, and the depth without it (630) applies if it is not adopted. The census itself
   counts the checker's *reference* realisation (`lt_pow` as a product, operand decode by hints); a backend that
   realises `Bf16Decode` by a 65 536-row table or shares the duplicated `q*sr` products changes the counts.
9. The depth reductions of 5.4 (\(c\) variables per message; table sharding) are this lane's proposals: their
   soundness derivations are in the table, their prover-cost factors are estimates, and neither has been measured.
10. The census figures of 9.4 marked (est.) come from the stated cost model, not from measurement.

## 11. Open items (questions, not decisions)

- Sequential depth: baseline 381 at \(B=4096\) (network floor 0.38 s at 1 ms RTT, 19.1 s at 50 ms) against
  Candidate B's 3; the proposals reach 158 (\(c=2,s=5\)) at \(+1.3\%\) committed elements and \(\approx2\times\)
  sumcheck-round prover work; the checker chain (297) is the floor of sharding alone. The measurements decide; no
  round budget is set.
- Whether to commit the 140 computed query wires per unit (depth 381, 181M committed) or not (depth 630, 126M).
- Whether the checker should realise `Bf16Decode` by one 65 536-row lookup per operand (the relation's `T_bf16`)
  instead of 4 hints + 3 range checks + 3 `mul` + 2 `lin`: per unit −96 `mul`, −96 range chunks, −64 `lin`,
  −32 committed columns, at the cost of a second 65 536-row multiplicity column and 32 more fixed-table
  lookups — a checker-lane question, not decided here.
- Whether the first complete backend ships as `COMPLETE_HVZK_BACKEND` (no step 0) with 5.5 as the documented
  path to `COMPLETE_ZK_BACKEND`.

## 12. The VU chain: 96 units + epilogue as one proof (implemented 2026-09-22, lane a-chain)

The Verity VU is `GemmCoordinate<1536>`: 96 word-chained transition units from the constant \(c_0=0\), then the
epilogue `Epilogue.check` (RN-even cast of the last FP32 word to the BF16 output word). Its public endpoints are
\(c_0=0\) and the output word \(y16\); the 95 intermediate words and every hint are private. Three ways to wire
the 96 units of one VU into the data-parallel GKR of section 4, with their costs at \(K=1536, B=4096\)
(\(U=393\,216\) units, \(468\) exported columns per unit, \(826\) wires, 6 checker layers) **[derivation]**:

| design | sequential depth (checker chain) | committed elements / VU | prover work (wire evaluations per layer sweep; phase-1 tables) | new code |
|---|---|---|---|---|
| (a) **deep circuit**: one circuit of 96 unit-blocks + epilogue, \(6\cdot96+2\) strict layers, data-parallel over \(B\) VUs (\(b=12\)) | \(\approx 578\) layers \(\times\) 47 rounds \(=27\,075\) | \(44\,747\) (the 95 intermediate words are wires, not columns: \(-190\)) | \(578\times4096\times79\,296=187\)G wire evaluations (strict layering carries every value of lower depth: width \(96\times826\)); tables 15.6 GB | new exporter for a 45k-column circuit, per-VU wiring tables |
| (b) **data-parallel units + linkage** *(implemented)*: \(U\) unit copies and \(B\) epilogue copies over **one** commitment; the linkage \(c_{v,0}=0\), \(c_{v,i}=y_{v,i-1}\), \(y32_v=y_{v,95}\) and the public words \(y16_v=\) recorded word are **linear constraints on the committed vector**, batched into the Ligero opening | \(+0\): checker chain 243 (6 layers, \(b=19\)), lookup chain 435 (`R16` has \(2^{28}\) leaves, below) — depth is set by the lookups exactly as before | \(96\cdot468+9=44\,937\) (the words are already the `x.c` / `y` columns of each unit; \(+9\) epilogue columns) | \(6\times393\,216\times826=1.95\)G; tables 15.6 GB; one extra \(O(U)\) pass for the linkage functional; the epilogue GKR (11 wires, 2 layers, \(B\) copies) is negligible | epilogue circuit export; multi-segment instance; one functional; no change to `gkr.rs`/`logup.rs` semantics |
| (c) **checkpoints every \(m\)**: commit every \(m\)-th word, prove the \(m\) units in between as a deep sub-circuit (\(6m\) layers) data-parallel over \(U/m\) copies | \(m=2\): 495; 4: 1 011; 8: 2 067; 16: 4 227; 32: 8 643; 96: (a) | \(44\,937-2\cdot96(1-1/m)\): \(m=2\): 44 841; 8: 44 769; 96: 44 747 | wire evaluations \(\times m\) (strict layers carry the sub-chain's \(m\times826\) values): 3.9G (m=2) … 187G (m=96) | as (a) plus a checkpoint layout |

The decisive fact is that the committed *word* is 1 element in a row of 468: checkpointing saves at most
\(190/44\,937=0.42\%\) of the committed elements and costs \(m\times\) the checker's prover work and depth.
(b) is the \(m=1\) point of (c), it needs the least new code, and its soundness argument is the one the Ligero
opening already carries. **(b) is implemented first**; (a)/(c) are not built (section 12.4 reports their cost
model; no measured point is expected to beat \(m=1\)).

### 12.1 Layout **[proposal, implemented]**

Two data-parallel *segments* over one committed vector \(X\):

~~~text
X = [ unit rows: U = B·96 rows × 468 columns, row v·96+i = unit i of VU v (VU-major) ]
    [ epilogue rows: B rows × 9 columns: one, y32, y16, hi, lo, hi2, lsb, carry, rem ]
    [ multiplicities of ALIGN, POW, LEAD, NORM, R16 over BOTH segments ]
~~~

Segment 0 is the exported unit circuit (section 1), segment 1 the exported epilogue circuit
(`gkr_export.vu.build_epilogue_circuit`: `Epilogue.check` verbatim — 5 sixteen-bit range chunks into `R16`
(`hi`, `lo`, `hi2·2`, `rem`, `hi+carry`), 2 booleans, 3 linear identities, and the assertion \(y16=hi+carry\)).
Each segment runs its own checker GKR (its own \(z_0,\gamma\), \(b_0=\lceil\log_2 U\rceil\), \(b_1=\lceil\log_2 B\rceil\));
the two are independent and run in lock-step (one verifier message carries both segments' round challenges), so
the checker depth is the unit segment's. The five logUp instances take the witness side of a table from both
segments concatenated (unit queries, then epilogue queries, then the table rows, then padding): the epilogue's
five `R16` queries per VU join the range-check multiset; `R16` therefore has \(352U+5B+65\,536=138\,498\,048\)
leaves at \(B=4096\), i.e. \(n_T=28\) (the census's 284 range queries per unit gave \(n_T=27\); the exporter's
352 come from two one-sided checks per non-power-of-two range and the materialised query wires).

### 12.2 Linkage and public endpoints as linear constraints **[proposal, implemented]**

Every claim the verifier is left with after the GKRs is a linear functional of \(X\) (section 4.3 layout rule:
input claims, lookup leaf claims). The chain adds \(U+2B\) more, all with **public** coefficients:

\[
X[(v,0),\,c]=0,\qquad X[(v,i),\,c]-X[(v,i-1),\,y]=0\ (1\le i<96),\qquad
X_{\mathrm{epi}}[v,\,y32]-X[(v,95),\,y]=0,\qquad X_{\mathrm{epi}}[v,\,y16]=\mathrm{word}_v .
\]

They are combined with geometric coefficients \(\rho_{\mathrm{geo}}^k\) (\(k\) the constraint index) into one
functional, which is batched with the \(2\cdot2+5\) GKR/lookup functionals by fresh \(\rho_f\) and opened once
(section 3, `ligero::prove_open`). Transcript: the \(\rho_f\) and \(\rho_{\mathrm{geo}}\) are drawn after the last
GKR message; the public words are absorbed into the transcript before any challenge (they are part of the
statement). Soundness **[derivation]**: if any linkage or endpoint constraint fails, the combined value is a
nonzero polynomial of degree \(<U+2B\) in \(\rho_{\mathrm{geo}}\): error \((U+2B-1)/|F_c|=2^{-173.4}\) at
\(B=4096\) over \(F_{p^3}\), plus \(1/|F_c|\) for its batching coefficient; the Ligero linear test then binds the
batched functional to the committed rows as before (the `encoding_opening` term is unchanged). Committing the
words twice (as `y` of unit \(i\) and `x.c` of unit \(i+1\)) is what makes the linkage linear; a state-chained
variant (pass \((s,e,M)\)) would save the decode/pack of the word (\(\approx4\%\), section 1) but is not what
`step.py` composes and is not built.

**Endpoints.** \(c_0=0\) is a constraint with constant 0 (no committed constant column is needed); the output words
are verifier inputs (\(B\) sixteen-bit words) and the only witness-independent data besides the circuits and the
batch shape. A `vu-k1536-neg` instance is a *claimed* word: with honest hints the epilogue's \(y16\) column must
equal the cast of \(c_{96}\) (assertion `y16`) **and** the public word (endpoint constraint), so the prover has no
accepting witness; an out-of-domain negative (non-finite operand, saturated intermediate) is rejected by the
unit circuit's range checks (`acc.t < 255`, `a[i].t < 255`) or already by `gen_step`/`check_step` at witness
generation.

### 12.3 Depth, rounds, what changed in the transcript

`rounds.sequential_depth` \(=1+\max(\text{lookup chain},\text{checker chain})+2\) (message 2a; the Ligero \(r\)
and \(Q\) coins; the batching coefficients ride on the \(r\) message). At \(B=4096\): lookup chain 435 (\(n_T=28\)),
checker chain 243 → **438**; the linkage adds no round trip. `rounds.protocol` grows by the epilogue segment's
sumcheck slots and by \(2\cdot2+5+1+1=11\) batching coins. Sharding `R16` (\(s=1\), section 5.4) would return
\(n_T\) to 27 (depth 409) for \(+65\,536\) committed multiplicities; not done.

### 12.4 Checkpoint interval \(m\): cost model (no measured points; 12.5 has the measured \(m=1\))

\(m\) = units between committed words (\(m=1\): every word committed = design (b); \(m=96\): design (a)).
Strict layering (libzk 6.3) carries every value of lower depth, so a sub-chain of \(m\) units is a layer of width
\(m\times826\); copies \(=U/m\). **[derivation, cost model of section 9.4]**

| \(m\) | committed / VU | checker layers | copies (\(b\)) | width / layer | rounds / layer | checker depth | wire evaluations | phase-1 tables |
|---|---|---|---|---|---|---|---|---|
| 1 | 44 937 | 6 | 393 216 (19) | 826 | 40 | 243 | 1.95G | 15.6 GB |
| 2 | 44 841 | 12 | 196 608 (18) | 1 652 | 41 | 495 | 3.90G | 15.6 GB |
| 4 | 44 793 | 24 | 98 304 (17) | 3 304 | 42 | 1 011 | 7.80G | 15.6 GB |
| 8 | 44 769 | 48 | 49 152 (16) | 6 608 | 43 | 2 067 | 15.6G | 15.6 GB |
| 16 | 44 757 | 96 | 24 576 (15) | 13 216 | 44 | 4 227 | 31.2G | 15.6 GB |
| 32 | 44 751 | 192 | 12 288 (14) | 26 432 | 45 | 8 643 | 62.4G | 15.6 GB |
| 96 | 44 747 | 576 | 4 096 (12) | 79 296 | 47 | 27 075 | 187G | 15.6 GB |

Committed elements move by \(<0.5\%\); prover work and depth grow linearly in \(m\). No point of the sweep
changes the picture, so none is implemented; the recorded runs of 12.5 are all \(m=1\).

### 12.5 Recorded runs

See `README.md` (section "Full VU runs") for the run ids, times, depth, proof bytes and memory of the
`vu-k1536` batches and the `vu-k1536-neg` rejections on vy-cpu2.
