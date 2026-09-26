"""The private-circuit track's prototype (Project store `docs/circuit-privacy.md`, Part I).

An inner Flock session (PR #83's `verity/flock-circuit` statement, run without zero knowledge) is verified inside V[B], a
fixed public verifier circuit that depends only on the public bounds B. V[B] is written as a program of batched gadget
operations (`program`), each a table of one Boolean gadget (`gadgets`); `vb` builds it from B alone, `coins` derives the
public inputs a verifier computes from the inner coins, and `witness` turns a recorded session and the private circuit C
into V[B]'s witness.
"""
