[2026-11-18T23:03Z] 7d7beb71e612
From the circuits lead: the storage/bandwidth lane's table of every row the verifier makes (read by its worker at file and line on one-circuit d56ff9fef3b). Design the one form against ALL of these, not storage's session alone.

1. pous.setup.base(u), u the store block's index. Inputs: the setup's salt; the scheme's parameters; blocks a segment (2^16). Value: the keys' prefix expanded from the parameters and the salt, zero-padded to 128 bytes, then u64be(s) || u64be(v) with (s, v) = divmod(u, 2^16). Width 8L; 1,152 bits deployed. Coin-bound by the SALT only (the memory attester's coin, drawn after the weights' commitment and before the store's exists; not the proof service's coin); u is the unit's index.
2. pous.setup.slot(u). Inputs: u and the layout. Value: u mod n, n = ceil(N/P) (2 deployed). 16 bits. NOT coin-bound: fixed at registration.
3. poub.read.pub(t, k), sample k = (pass i, position p) of window t. Inputs: the openings coin (which gives p); the pass's seed; THE PROVER'S PASS ROOT; the registration's codec seed, key, payload root, domains, counts. Value: 20 fields in a fixed order (the position, the codec seed, the key's three parts, the SHA-512 midstate of the sketch's prefix under the pass's seed, the pass root with its tree's domain and depth, the payload root with its domain and depth, zeros for the unused verdict tree). Width 4,320 + (2+K+MASKED)*N; 6,368 bits deployed. Coin-bound, and it reads a released commitment of the prover's.
4. poub.read.signs(t, i). Inputs: the pass's seed, ROWS, BYTES. Value: the sign bits of the pass's sketch rows. ROWS*BYTES; 8,192 bits deployed. Coin-bound by the pass's seed.
Plus storage's session four (round_coin(j) 96 bits, round_key(j) 64 bits, header() 6,048 bits, context() 848 bits) that you have.

Port bits are 16-bit words, low bit first, byte strings as little-endian pairs (Span.lean 72-76). Where computed: setup_proof.py 43, 58-66; schemes/p2.py 347; block_check.py 1267-1294; span.py 35-51; receipts.py 125, 409, 417; Lean Poub/Span.lean 82, 97, 115. The setup's two have NO Lean today.

A THIRD thing for the form: setup/ok is NOT an input: it is the constant 1 that every unit's OUTPUT is held to. So the form also needs: "a block's public output equals a registered constant", said once. Include it: how C names a public output held to a constant, and what the statement carries.

Note: a value like slot(u) that is fixed at registration is not coin-bound at all: say what it is in the form (a registered constant per unit: a function of the unit's index, part of C).

