# silicon_trail

How cheaply can a difference between two inputs absorbed by the logging chain be cancelled by differences absorbed
later? This directory holds the chain's step, the model of a difference through it, a search for the lightest
cancelling trail, and a direct count of collisions that checks the model. Python's standard library (numpy only for
`collide.py --exact`) and one C file. The SAT solver is fetched and built outside the tree by `solver.sh`.

## The step

    s_i = R(s_{i-1} xor P(x_i))

The state has 25w bits (w = 64: 1,600), a step absorbs 4w bits (256), and R is one round of Keccak-p[25w]. State bit
w(5y + x) + z is a[x, y, z]: bit z of lane (x, y). R = iota . chi . pi . rho . theta, applied in this order, indices
mod 5 in x and y and mod w in z:

- **theta:** a[x, y, z] ^= p[x-1, z] ^ p[x+1, z-1], where p[x, z] is the parity of column (x, z).
- **rho:** lane (x, y) is rotated towards higher z by r[x][y] mod w:

  | r[x][y] | y = 0 | y = 1 | y = 2 | y = 3 | y = 4 |
  |---|---|---|---|---|---|
  | x = 0 | 0 | 36 | 3 | 41 | 18 |
  | x = 1 | 1 | 44 | 10 | 45 | 2 |
  | x = 2 | 62 | 6 | 43 | 15 | 61 |
  | x = 3 | 28 | 55 | 25 | 21 | 56 |
  | x = 4 | 27 | 20 | 39 | 8 | 14 |

- **pi:** lane (x, y) moves to (y, 2x + 3y).
- **chi:** in each row (y, z), b[x] ^= (not b[x+1]) and b[x+2].
- **iota:** a constant is XORed into lane (0, 0).

P places the 4w input bits in four whole lanes, input bits wj .. wj + w - 1 in the j-th. There are two placements
(`keccak.PLACEMENTS`), and they are different chains:

- **plane:** lanes (0,0), (1,0), (2,0), (3,0), plane y = 0 of the round above; lane (4,0) takes nothing. **This is what
  the synthesized absorber does:** its 256 input bits enter state bits 0..255. `check_absorber.py` evaluates the
  absorber's Verilog and compares it with `keccak.step(..., "plane")`.
- **diag:** lanes (0,0), (1,1), (2,2), (3,3). This is Ketje Major's placement. Ketje v2 (Bertoni, Daemen, Peeters,
  Van Assche, Van Keer, "CAESAR submission: Ketje v2", sections 2.1 and 5.3) runs the twisted round pi . R . pi^-1 and
  absorbs into plane y = 0 of the twisted state, which in the coordinates above is this diagonal (`keccak.twisted_step`;
  `selfcheck.py` checks the equivalence).

**The round constant** is a parameter, and the absorber's Verilog takes it as an input port. Its value never matters
to which inputs collide: iota adds it to lane (0, 0), which both placements absorb into, so a chain with constant rc on
inputs (x_1, ..., x_n) is the chain with constant 0 on (x_1, x_2 xor rc, ..., x_n xor rc), with rc XORed into the final
state (`selfcheck.py`). The search and the counts use 0.

## The model

A difference a before a round passes L = pi . rho . theta unchanged in probability: b = L(a). Through chi each 5-bit
row of b goes to one of an affine set of output rows, all equally likely over a uniform state; the row's weight is
log2 of that set's size (0, 2, 3 or 4) and w(b) is the sum over rows (`diff.py`). A k-step cancelling trail is input
differences d_1 .. d_k with a_1 = P(d_1), b_i = L(a_i), c_i one of b_i's outputs, a_{i+1} = c_i xor P(d_{i+1}) and
c_{k-1} = P(d_k): k - 1 rounds, after which the last input difference removes what is left. Its weight is the sum of
the w(b_i), and its probability is 2^-weight if the rounds' conditions hold independently (round 1's factor is exact).

## Files

- `keccak.py`: the step, both placements, theta's and L's inverses, Keccak-f for the check against SHA-3.
- `diff.py`: chi's difference table by brute force, weights, and `Trail`, whose `check()` shares no code with the search.
- `satmodel.py`: a k-step cancelling trail of weight at most W as a CNF; also plain trails with nothing absorbed
  (`free`) and the active rows after one round (`rows`), which use the same encoding of the round.
- `search.py`: asks the solver about several W at once and narrows the least weight; decodes and checks every trail.
- `solver.sh`: fetches and builds Kissat 4.0.4 (MIT licence), pinned by commit, in a directory outside the tree.
- `collide.c`, `collide.py`: the chain run from random start states on two inputs a fixed difference apart, 512 start
  states a machine word, counting equal final states; `--forced` samples start states on which round 1 follows the
  trail; `--exact` enumerates all 2^25 start states at w = 1.
- `selfcheck.py`, `check_absorber.py --rtl DIR`: the checks of the step named above.

## What an answer means

A SAT answer is a trail, decoded and checked by `diff.Trail.check`. An UNSAT answer at W says the encoding has no
trail of weight at most W: it is as good as the encoding and the solver, and no proof from the solver is checked.
The encoding is checked by decoding every trail found and by reproducing published least weights of plain Keccak-f
trails (Mella, Daemen, Van Assche, IACR ToSC 2017(1): 8 for 2 rounds; 20, 24, 32, 32 for 3 rounds at widths 200, 400,
800, 1600; 46 for 4 rounds at width 200). There is no independent exhaustive search of the cancelling mode. A bound on
one trail's weight is not a bound on the collision probability, which adds up every trail with the same input
differences and can depend on the common input from step 2 on.

## Commands

    python3 selfcheck.py                                  # seconds
    python3 check_absorber.py --rtl DIR                   # DIR: the absorber's gov_ft.v and synth/keccak1600_rnd.v
    S=$(./solver.sh SCRATCH)                              # builds Kissat in SCRATCH, prints the binary's path

Every search is one call, and it keeps no checkpoint: to resume, run it again with `--lo` one more than the largest
weight already excluded. It stops at the minimum or after `--seconds` and writes what it reached to `--out`.

    python3 search.py --solver $S ARGS --lo LO --seconds SECONDS --jobs CORES --out OUT/NAME.json

| search | ARGS |
|---|---|
| 3 steps, full width, measured absorber | `--w 64 --placement plane --k 3 --hi 400` |
| 3 steps, full width, Ketje Major's placement | `--w 64 --placement diag --k 3 --hi 400` |
| 4 steps, full width | `--w 64 --placement plane --k 4 --hi 400`, and `--placement diag` |
| 5 steps, full width | `--w 64 --placement plane --k 5 --hi 400`, and `--placement diag` |
| least active rows after one round | `--w 64 --placement plane --mode rows --hi 40` with `--lo 1`, and `--placement diag` |
| widths 200 and 400 | `--w 8 --placement plane --k 3 --hi 150`, `--w 16 ... --hi 200`; likewise `--k 4`, `--k 5`, `diag` |
| the encoding against published plain trails | `--w 16 --mode free --rounds 3 --hi 40` (24), `--w 32 --mode free --rounds 3 --hi 40` (32), `--w 8 --mode free --rounds 4 --hi 60` (46), each from `--lo 8` |

A search takes all the cores it is given for minutes to hours, so it runs on a node and never on an agent VM. Name
the node: `research run --on auto` picks one itself.

    research run --on MACHINE --pool research --as LANE --project verity --source . --cwd source --mem-gb 16 --timeout T -- \
      bash -c 'set -e; d=$(mktemp -d); trap "rm -rf $d" EXIT; cd benchmarks/silicon_trail; S=$(./solver.sh $d);
               python3 search.py --solver $S ARGS --lo LO --seconds SECONDS --jobs 12 --out $OUT/NAME.json'

The collision count for a trail a search wrote:

    python3 collide.py --trail OUT/NAME.json --forced --x random --log2-trials 36 --jobs CORES --out OUT/collide-NAME.json

Without `--forced` it counts from uniform start states, which takes about 2^weight trials for one expected collision
and so reaches only light trails; `--x zero` and `--x fixed:SEED` hold the common input fixed instead of sampling it.
