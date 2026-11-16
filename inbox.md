[2026-11-16T21:57Z] 9bdc4534d9b2
top, Mon 16 Nov 1:57 PM PST. Daniel, 1:51 PM PST, on the docstrings code reads as --help: "this seems fine? I don't care about modifying that." So they are IN tonight's pass. This replaces UNTOUCHABLE for one kind only: the module docstring of a Python file that reads its own __doc__ (521 non-test files on main e80c1a37ebb by my AST scan; 332 take __doc__.split("\n\n")[0], 10 split("\n")[0], 87 .splitlines()[0], 76 pass it bare, 8 .strip()).

a. It is never deleted and never left empty. Every one of those files calls a method on __doc__ or hands it to argparse, so a missing docstring is a crash when the command starts. digest.py marks it must_remain (a new mark, not untouchable); apply.py refuses a delete decision on it; certify.py refuses a head where a must_remain docstring is gone or empty.
b. The decider keeps its first paragraph as what --help prints: what the command does and its usage, for the person who runs it, with no memory. What follows the first paragraph is under the file rule and is cut, unless it is usage argparse does not already give (what it writes, exit codes).
c. A file whose code reads past the first paragraph or line keeps its module docstring UNTOUCHABLE: a subscript on a chain from __doc__ whose index is not 0 (split(...)[1], [2], [1:4], [-1]), a .partition, a % or .format, or a split on a sentence of the docstring. 17 lines on main do, among them tools/research/src/research/cli.py:233 (research run's description is paragraph 2), merge.py:2116, tools/check/post_train.py:388, pods/connect.py:318, digest.py:575, util_collect.py:160 (splits on a sentence). Find them by AST, not by my list.
d. "Text a test pins" still wins: a line a test quotes stays byte for byte.
e. Every other docstring read at run time stays UNTOUCHABLE as inbox 7 says (X.__doc__ of another object, a Definition's body, inspect.getsource or getdoc): those feed digests, not help.

Also his: the 34 functions tests/lint/allowlists/repo_python_verdicts.json pins by text are IN the pass (item 5 of inbox 7 is taken back; that test only warns). List them in your report: the list is mine and I refresh its digests after the pass.

If this moves your 2:50 PM machinery time, write the new time in your note; 3:15 PM PST is fine.

