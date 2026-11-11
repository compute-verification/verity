[2026-11-11T13:45Z] 1e2d9341f96e
FORM to WIDTH and training, TRN-103 / TRN-121, 13:45Z (Wed 11 Nov). ResolveWithin.lean compiled and is pushed.

Branch cursor/record-property-laidout-3b1c. The commit that holds ONLY the ResolveWithin.lean change: d8d914e4c5 (on a8ed587ba8). Head after it: 0a3af3096b (COUNTS' fe99d83c7f69 merged: RecordLanding.lean as of the one proof, LeafWidthsType.lean; RecordCounts imports LeafWidthsType).
Run r20261111-133905-03cb on vy-nebius-cpu-1, verity lean try ResolveWithin.lean --siblings (ran by me, on the file as committed in d8d914e4c5): lean exit 0 in 23 s; none of its 11 theorems uses an axiom beyond propext, Classical.choice and Quot.sound; siblings: no clash among 2,844 modules imported (it lists the modules of that node's tree not built, unchecked). It compiled first time, so the names and statements are as my 13:39Z message gave them:
ADDED: gateStep_member, walk_within, gate_within. CHANGED (the leaf's bounds added to the ∃ i j, nothing else): run_within, ret_within, operands_within, ret_within_of.
All in namespace NciCheck.Properties.SecurityProofs.Resolve, inside the section with (P) (hroot) (hB) (hL); gateStep_member omits hroot and hB, so it is called gateStep_member P hL sc hd hp hk hm hc hlo hhi h, with {k m g} and {x} implicit:
  (hd : sc.d < P.defs.size) (hp : (P.defs[sc.d]!).prim = false) (hk : k < nodes.size) (hm : m < nodes[k]!.members)
  (hc : (P.defs[nodes[k]!.fn]!).prim = false)
  (hlo : sc.off + offsets[k]! + m * gates(callee) ≤ g) (hhi : g < sc.off + offsets[k]! + m * gates(callee) + gates(callee))
  (h : gateStep P g sc = .ok x) : x = .inr (P.child sc k m)
WIDTH: to use it without my other commits, cherry-pick d8d914e4c5 (one file) or take the file from it. Not in my file, yours: the induction along Good from the root, and the last step at the primitive node.
NOT CHECKED: the package root, check.py, the lock; a clash with a module that node had not built.

