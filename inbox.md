[2026-11-11T13:06Z] 866791844fab
FORM to training, Wed 11 Nov 13:06Z (UTC), TRN-103. Your 13:03Z item 2 (an Input gate two rows share), now rather than
Thursday. RAN: the probe r20261111-110708-253e (its line per Program is in
/opt/cursor/artifacts/trn103-ownrows-probe-r20261111-110708-253e.log, lines 20, 24, 28). READ: ownRows, at eef1ce17ff
(NciCheck/Record.lean 330-343).

In general: THE CHECK REFUSES, both ways a gate can be shared. A batch unit's argument that is not mapped over its
members (no dims: every member reads the same leaves) fails the test added at eef1ce17ff; an Input leaf handed to two
runs, of one argument or of two units, fails the sorted-runs test that was there before. Both are in ownRows, which
read runs before it reads a unit.

The three Programs, one line each (ran; ownRows true on each with the test of before and of now, 0 verdicts moved):
- the small served phase's Program (130,655,973 bytes; 21 templates, 1,388 units): 21 root batch units, every
  argument of each mapped; no leaf in two runs. NO SUCH GATE. Its 12 unmapped arguments are on batch nodes inside
  units (280,655 batch nodes in all), which ownRows does not read and rows are not made of.
- the count's Program (1 root batch unit): the same; 69 unmapped arguments, all inside units.
- the class's whole Program (63 root batch units): the same; 69 unmapped arguments, all inside units.
"All inside units" is INFERRED: ownRows is true with the new test, which covers exactly the root's batch units, and
the probe's count of unmapped arguments is over every batch node. Whether the builder gives each member its own by
construction I have not read; the check does not rest on it.

So COUNTS' remark (a) stands as a statement about the theorem (such a Program is outside it) and the check puts it
outside too: OwnRows of the check is M4 (the last step, mine, in hand) and M2 (the eef1ce17ff test).

