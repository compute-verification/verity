"""WP2: the auxiliary-witness checker Check(x, y, h) for the frozen target's transition unit and, by chaining,
for the proof unit.

This is a REFERENCE: Python ints, clarity over speed, no field.  Its decomposition into gadgets, its hint layout
and its predicate names are one way to write the relation and are NOT canonical for the proof backends; what a
backend must preserve is the set of accepted (x, y) pairs (section 1) and, if it embeds the relation in a prime
field, the additional bounds of section 3.

The executable semantics it must agree with on the finite domain is ``verity.ml.circuits.silicon.silicon`` (``tc_dot`` =
two ``group_sum`` steps of the Ampere BF16 m16n8k16 model), then ``verity.ml.circuits.silicon.cast.f32_to_bf16`` for the VU.
Nothing here is derived from that module; ``reference.py`` re-states the same arithmetic parametrically so the
reduced-width searches have a reference, and the differential tests compare the checker to ``silicon`` itself.

Layout.  ``params.py`` the widths as data (``REAL``, ``TOY``, ``TINY``); ``tables.py`` the four lookup tables;
``ctx.py`` the five predicates on Python ints (``IntCtx``) and as z3 constraints (``Z3Ctx``); ``gadgets.py`` one
class per gadget with ``contract(p)`` (ranges as data), ``gen`` (honest hints) and ``check`` (predicates);
``step.py`` the unit and the VU; ``search.py`` the adversarial searches.

FP8 (silicon Phase 0).  ``Params`` carries two formats -- the operand's (``sig``, ``exp``, ``op_nan_rule``) and the
accumulator's (``acc_sig``, ``acc_exp``) -- and admits an adder narrower than the accumulator (``width <
acc_sig``, the *lossy* case of Hopper/Ada E4M3: 14 < 24).  ``Params.from_model(model)`` derives them from a
``verity.ml.circuits.silicon.models.GroupSum``; at ``AMPERE_BF16_M16N8K16`` it is ``REAL`` and nothing below changes (contracts,
hint layout and outputs at ``REAL``/``TOY``/``TINY`` are byte-identical to the BF16-only checker).  In the lossy
case one gadget joins the data flow before ``AlignState``: ``RescaleState`` (M = 2^drop P + r, r < 2^drop, P < 2^W),
the truncation of the incoming FP32 significand to the adder width that ``GroupSum.step`` performs; ``Normalise``
then produces a W-bit magnitude and scales it back up exactly (``norm_up``), since the truncations must precede
the scale-up and no longer fold into one floor division.  ``Bf16Decode`` (alias ``OperandDecode``) admits the
E4M3 rule (the all-ones exponent is finite unless the fraction is all ones: 448 is a value, ``0x7F``/``0xFF`` are the
NaNs) through one extra range check.  ``tests/checker/test_fp8.py`` compares the reference and the gadget relation
with ``model.step`` on ``HOPPER_E4M3_K32`` and ``ADA_E4M3_M16N8K32``.  The z3 group-step search and the v2
arithmetisation are BF16-only and say so.

Predicates.  Every gadget is written with five predicates only, so that a backend can read off what it must
provide:  ``rng(x, lo, hi)`` a range check on an expression; ``eq(a, b)`` an identity; ``boolean(b)``;
``lookup(TABLE, key)`` a row of a fixed table (POW, ALIGN, LEAD, NORM); ``lt_pow(x, 2^k, B)`` the variable-bound
range check x < 2^(B-k), realised in the integer model as ``rng(x * 2^k, 0, 2^B)``.

--------------------------------------------------------------------------------------------------------------------
1. The relation and the ∃h equivalence
--------------------------------------------------------------------------------------------------------------------

Unit.  x = (c, a[0..16), b[0..16)): an FP32 word and 32 BF16 words, every exponent field finite (the all-ones field
is outside the domain and is rejected by the decode gadgets).  y is an FP32 word.  h is the list of hints below.
Claim:  (∃h. Check(x, y, h) = 1)  ⇔  y = tc_dot(c, a, b).

Data flow (one group; the second group takes the first's output state as its state)::

    Fp32Decode  c -> (s_c, e_c, M_c)               hints s, t, f, nz, z
    Bf16Decode  a_i, b_i -> (s, m, e) each        hints s, t, f, nz
    Product     u_i = m_a m_b, e_i = e_a + e_b, s_i; zero product masked to the floor -132   hint z
    GroupMax    E = max(-132, e_c, e_0..e_7)      hints E, sel[0..10) (one-hot over the 9 candidates + floor)
    AlignState  M_c 2^1 = 2^d q + r, d = E - e_c   hints q, r          (W = 25, tight quotient bound)
    AlignProduct u_i = sr q + r, aligned = q sl   hints q, r per product (the trailing-zero specialisation)
    SignedSum   S = ±q_c ± q_0 .. ± q_7 = (1 - 2 sgn) v, |S| < 2^30 enforced   hints sgn, z
    LeadingBit  2^(l-1) <= v < 2^l               hint l
    Normalise   e' = max(E + l - 25, -126), M' = floor(v 2^(E - e' - 1))   hints clamp, M, r
    StateOut    zero cases -> +0 at the floor; overflow flag e' > 127       hints z, ovf
    Pack        the FP32 word: +0, the infinity of the first saturating group, or (s, e' + 126 + hidden, low)
                                                  hints hidden, low
    Epilogue    f32_to_bf16 RN-even on the VU's final word   hints hi, lo, hi2, lsb, carry, rem

Completeness (⇐).  ``gen_step`` computes, for every gadget, exactly the values its predicates name (the digits of
the word, the maximum and its first attaining index, the Euclidean quotient and remainder, the sign and
magnitude, the bit length, the clamp flag, the flags).  Each predicate is then true by the definition of the
value (a mixed-radix decomposition is unique; a Euclidean division satisfies 0 <= r < divisor; the bit length l of
v satisfies 2^(l-1) <= v < 2^l), and the outputs are the quantities ``group_sum`` computes, written as expressions:
the closed form M' = floor(v 2^(E - e' - 1)) with e' = max(E + l - 25, -126) is silicon's 25-bit normalisation
followed by its subnormal shift and its rescale to 24 bits, including the truncation-to-zero case (M' = 0 with
v != 0) and the exact-zero case (S = 0), both of which ``StateOut`` turns into the +0 state with the floor exponent
exactly as ``group_sum`` returns (0, 0) and ``fp32_term`` masks it.  ``Pack`` reproduces ``pack_fp32`` (the
hidden bit is 0 only at e' = -126 with M' < 2^23: a subnormal or zero; the saturation returns the infinity with
the saturating group's sum sign, as the early return in ``tc_dot``).  This is what the differential tests check:
structured, random, every all-finite golden record (all 360; 114 of them saturate to an infinity), and the
K = 1536 conformance vector for the VU.

Soundness (⇒).  Each gadget is a *function* of its inputs on the integers: whenever its predicates hold and its
inputs lie in its contract's input ranges, its outputs equal the mathematical values, whatever the hints.
Induction along the data flow (each gadget's output ranges are the next gadget's input ranges; the public words
are range-checked once by the unit) then gives y = tc_dot(x).  Per gadget:

  decode      s 2^31 + t 2^23 + f = c with s < 2, t < 255, f < 2^23 is the unique mixed-radix decomposition; nz is
              pinned by (1 - nz) t = 0 (t != 0 => nz = 1) and t - nz >= 0 (t = 0 => nz = 0); z by z M = 0 and
              M + z - 1 >= 0.  (BF16 likewise.)
  product     u, e_p, s_p are expressions; z is pinned as above.
  group_max   d_i = E - cand_i >= 0 for every candidate and E - floor >= 0 give E >= max; one-hot selection
              gives E = some candidate or the floor, so E <= max.  An excessive E has no attaining row.
  align       0 <= r < 2^d (``rem``) and 2^d q + r = P determine (q, r) uniquely (Euclidean division); with d >= W
              the table row forces q = 0 (P < 2^W), with d = 0 it forces r = 0.  The tight bound q < 2^(W-d)
              (``quot``) is implied over the integers (q = (P - r) / 2^d < 2^(W-d)); it is stated because a field
              needs it (section 3).  For products the ALIGN row makes the same argument on u alone; the aligned
              term q sl equals floor(u 2^10 / 2^d) because the low 10 bits of P = u 2^10 are zero.
  signed_sum  S is an expression of earlier outputs; v = (1 - 2 sgn) S with v in [0, 2^30) pins sgn unless
              S = 0, where sgn z = 0 with z pinned by z v = 0, v + z - 1 >= 0 forces sgn = 0.  The premise
              |S| <= (2^25 - 1) + 8 (2^26 - 1) < 2^30 is recomputed from the contracts by ``SignedSum.premise``;
              the checker refuses to run if it fails.
  leading_bit the LEAD rows [2^(l-1), 2^l) for l = 1..30 and {0} for l = 0 partition [0, 2^30); ``window`` puts v in
              row l, so l is v's bit length.
  normalise   ``clamp_proof`` is clamp (e_min - 1 - e_raw) + (1 - clamp) (e_raw - e_min) >= 0, whose two branches
              are exclusive on integers (one of e_min - 1 - e_raw, e_raw - e_min is negative), so clamp = [e_raw <
              e_min]; e' and k are then expressions; div M' + r = v mul with 0 <= r < div is a Euclidean division,
              so M' = floor(v mul / div) = floor(v 2^k).  Every |difference| in a flag proof is below ``cmp_rows``
              (512) by the interface ranges, which is what makes the two branches exclusive under a range check.
  state_out   z pinned as above; ovf by the same two-branch proof against e_max.
  pack        hidden 2^23 + low = M with hidden < 2, low < 2^23 is unique; the word is an expression.
  epilogue    hi 2^16 + lo = y32, 2 hi2 + lsb = hi, carry 2^16 + rem = lo + lsb + (2^15 - 1) are three unique
              decompositions; y16 = hi + carry, range-checked below 2^16 (an infinity rounds to itself; a NaN
              cannot reach the epilogue on the domain).

The VU.  ``check_vu`` chains K/16 units from c_0 = 0 through their output words and ends with ``Epilogue``.  An
infinite intermediate word has the all-ones exponent field and is rejected by the next unit's ``Fp32Decode``, as
``verity_sp1.proofs.families`` rejects it (``tc_dot_chain`` raises); an infinite final word is cast to the BF16
infinity.  Soundness of the chain is soundness of the units by induction on the chunk index.

Evidence (see ``search.py`` and the tests).  Exhaustive enumeration of every input x every joint hint assignment
per gadget at ``TINY`` widths: every accepted assignment agrees with the reference (multiple accepted assignments
exist only for ``group_max`` on ties, all with the same E); the composed unit at ``TINY`` under single-hint
mutation: no alternative hint value is accepted.  z3 at the real widths per gadget: UNSAT for "an accepted output
differs from the reference"; SAT as soon as any one of the named bounds (``attained``, ``onehot``, ``rem``,
``window``, the range of v, ``zero_sign``, ``clamp_proof``, ``ovf_proof``) is dropped, i.e. each of them is
load-bearing.  One full group step at the real widths with two independent hint copies on one concrete input:
UNSAT.  The negative tests mutate each gadget's hints in the handoff's named ways and check the rejecting
predicate.

--------------------------------------------------------------------------------------------------------------------
2. Unresolved obligations
--------------------------------------------------------------------------------------------------------------------

  U1  The soundness argument of section 1 is a per-gadget uniqueness argument plus an informal induction over the
      data flow; it is not machine-checked.  The evidence is enumeration at reduced widths and z3 per gadget at
      the real widths; there is no proof that ``TINY`` widths exhibit every case of the real widths.
  U2  The z3 query for one full group step with *symbolic* inputs (two hint copies, two outputs) returns UNKNOWN
      at 60 s locally and at the 1800 s timeout of the recorded pod run r20260921-232849-093c; only the
      concrete-input instance is UNSAT (0.3 s).  Step-level soundness at the real widths therefore rests on the
      per-gadget queries and the induction of section 1, not on a step-level solver result.
  U3  ``signed_sum`` read over F_(2^31 - 1) returns UNKNOWN at 20 s locally and at 300 s on the pod run (the
      sign-pinning argument needs p > 2^31 - 2, one below the modulus); listed as open, not as a finding.
  U4  The contracts' interface ranges (``Contract.inputs``/``outputs``) are consistent by construction of
      ``params.py`` and are exercised by every test, but there is no separate proof that each gadget's output range
      is contained in the next gadget's input range at every width; ``SignedSum.premise`` is the one bound that is
      recomputed and enforced at run time.
  U5  The checker follows ``silicon`` on the two open items of the discrepancy log that touch this arithmetic (the
      normalisation of the group sum, and the zero result): if the reference semantics moves, the checker's
      ``Normalise``/``StateOut`` closed forms and the differential tests move with it.
  U6  Equivalence is claimed on the finite domain only.  Non-finite operand or accumulator words are rejected by
      the decode gadgets (no h exists), which is a statement about the relation, not a proof that the VU's
      surrounding obligation (``families`` rejecting a non-finite intermediate) is discharged elsewhere.

--------------------------------------------------------------------------------------------------------------------
3. Field embedding: what a backend must bound in addition
--------------------------------------------------------------------------------------------------------------------

The integer model reads ``eq`` as an identity of integers.  Over F_p every ``eq`` is a congruence, so the
requirement everywhere is that both sides of every identity, on hints and inputs that pass their range checks,
are integers below p; and that every range check is on the canonical representative in [0, p).  The findings
(z3 over F_p in ``search.modular_queries``; each SAT reproduced in the results table) and the resulting list:

  F1  A variable-bound range check realised as a product wraps.  ``lt_pow(x, 2^k, B)`` as ``x 2^k < 2^B`` is
      sound over the integers but, read mod p, x 2^k can reach 2^(2B): SAT for ``align_state`` (B = 25) and
      ``leading_bit`` (B = 30) over a 40-bit prime, UNSAT over Goldilocks (2^64 - 2^32 + 1 > 2^60).  Rule: either
      p > 2^(2B) for every ``lt_pow`` (B = 30 here, so p > 2^60), or realise the variable bound directly (a lookup
      on (x, bound), or a bit decomposition of x masked by the bound).  With the direct realisation the 40-bit
      prime is UNSAT again (``leading_bit``, ideal bounds).
  F2  The tight quotient bound is load-bearing over F_p.  With ideal variable bounds over F_(2^31 - 1),
      ``align_state`` and ``align_product`` are UNSAT as stated and SAT without ``quot``: the reproducer is
      M = 0, d >= W, q = 63, r = 2^25 - 1, so that 63 2^25 + 2^25 - 1 = 2^31 - 1 ≡ 0.  Over the integers the same
      query is UNSAT (the bound is implied).  Rule: state and check q 2^d~ < 2^W (and q sr < 2^16) in the field.
  F3  ``normalise``'s reconstruction div M' + r reaches (2^24 - 1) 2^7 + 2^7 - 1 = 2^31 - 1: SAT over
      F_(2^31 - 1) with ideal bounds (v = 0, M' = 2^24 - 1, r = 127).  Rule: p > 2^31 - 1, or split the division.
  F4  A 32-bit word does not fit a 31-bit field: ``fp32_decode`` and ``epilogue`` are SAT over F_(2^31 - 1)
      (2^31 ≡ 1 lets the sign bit trade against the low bits), and ``pack``'s output is not injective there.
      Rule: p > 2^32 for words as single elements, else carry (s, t, f) or 16-bit limbs instead of the word.

Per gadget, the magnitudes a field must exceed (on in-range hints and inputs), and what to check on
representatives:

  bf16_decode   word < 2^16; hints s, t, f, nz and t - nz range-checked.
  fp32_decode   word < 2^32 (F4); M + z - 1 range-checked.
  product       u < 2^16; u + z - 1 range-checked.
  group_max     candidates and E within [-252, 259]: signed values, so either an offset representation or the
                usual signed reading with p > 2 cmp_rows; every d_i and E - floor range-checked below
                shift_rows = 512.
  align_state   q 2^d~ + r < 2^26 given ``rem`` and ``quot`` as ideal variable bounds (F1, F2); the POW row
                (2^d~, 2^(W-d~)) fits below 2^25.
  align_product q sr + r < 2^17 given the same two bounds; the ALIGN row (sl, sr, isr) below 2^16.
  signed_sum    S is signed with |S| < 2^30: the representative of v = (1 - 2 sgn) S must be checked in
                [0, 2^30) and p > 2^31 - 2 so that v ≡ ±S forces v = ±S (U3 for the boundary prime).
  leading_bit   (v - 2^(l-1)) 2^(30-l+1) < 2^30: ideal variable bound or p > 2^60 (F1).
  normalise     div M' + r = v mul < 2^31 (F3): p > 2^31 - 1; ``rem`` as ideal variable bound; ``clamp_proof``
                below cmp_rows.
  state_out     M + z - 1 and ``ovf_proof`` range-checked; nothing exceeds 2^24.
  pack          y < 2^32 (F4): p > 2^32 or limbs.
  epilogue      hi 2^16 + lo < 2^32 (F4); hi + carry range-checked below 2^16.

Goldilocks (2^64 - 2^32 + 1) satisfies every magnitude above, including F1 with the product realisation; a
31-bit or 40-bit field needs the direct realisation of every ``lt_pow`` (F1), keeps the tight quotient bound (F2)
and, at 31 bits, also needs the word split (F4) and a wider normalisation division (F3).
"""
from .params import REAL, TINY, TOY, TOY_LOSSY, Params, Reject
from .step import accepts_step, accepts_vu, check_step, check_vu, contracts, gen_step, gen_vu, step_output, vu_output

__all__ = ["REAL", "TINY", "TOY", "TOY_LOSSY", "Params", "Reject", "accepts_step", "accepts_vu", "check_step",
           "check_vu", "contracts", "gen_step", "gen_vu", "step_output", "vu_output"]
