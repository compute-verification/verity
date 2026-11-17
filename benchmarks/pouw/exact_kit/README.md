# The exact kit

`exact_kit` holds the primitives PoUW's theory and referee scripts share: a check registry that writes a run's verdict,
rank over GF(2), rotations of k-bit words, exact kernel searches over a packed domain, and a CNF over named variables
solved by pysat. It holds no theory: a referee and the author it checks may both use it, and neither re-runs the
other's argument. Its distribution, `verity-exact-kit`, is a workspace member, so a script anywhere in the workspace
imports it with no `sys.path` entry.

## Try it

From the repository root, take the rank of three rows over GF(2):

~~~sh
uv run python -c "from exact_kit.f2 import rank, from_matrix; print(rank(from_matrix([[1,0,1],[0,1,1],[1,1,0]])))"
~~~

It prints `2`: the third row is the sum of the first two.
