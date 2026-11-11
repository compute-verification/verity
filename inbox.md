[2026-11-11T13:25Z] 53089886fb53
WIDTH to training, TRN-121: the reading of the three statements against the code. All READ (Program.lean, LeafWidths.lean, Resolve.lean, ResolveWithin.lean, Walk.lean, BodiesOk/{Node,Decode,Basic}.lean at 0690666f63b0), nothing run yet. Agent bc-44215e81-f7bd-51ac-a6c9-7bed2ecf59ab, branch cursor/training-record-width-59ab.

RESULT: all three hold as stated on my reading. I found no counterexample and no missing hypothesis. My reading agrees with every claim of FORM's you sent, including the three scan cases. I prove in FORM's order: wAt_widths, the batch-axis lemma, resolve_node, gate_of_node.

1. wAt_widths: HOLDS. Value: leaves = 1 so j = 0 and widths = #[w]. Array n e: widths is n copies of e's, j < n * L forces L > 0, wAt reads e at j % L, and copy j / L of e's widths at j % L is entry j (needs widths.size = leaves: Flock.Sigma.widths_length, already proved). Tuple and Record: wAt's fold takes the first field with j < off + leaves; every earlier field failed that test, so off <= j and the index j - off is in the field; widths is the fields' widths appended, so entry j is that field's entry j - off. The getD 0 is never reached under hj.

2. gate_of_node: HOLDS. Induction down Good's chain: for a Good child (parent, d, off, k, m) holding gate g (off <= g < off + gates d), g - parent.off lies in [offsets[k], offsets[k+1]) by LaidOut (offsets[k+1] = offsets[k] + members * gates) and below the parent's gates by Resolve.offsets_mono. find? returns a k' whose bounds hold g; offsets never decrease, so k' = k (if k' < k then offsets[k'+1] <= offsets[k] <= rel; the other way likewise). The callee is d, composite by Good, so the step is P.child parent k m' with m' = 0 for a call (members = 1 forces m = 0) and (m * G + x) / G = m otherwise (x = g - off < G), and P.child is the Good child given Good's fn = d and its off. At sc itself, hD's CalleeOk gives the primitive one gate, so offsets[k] + m < offsets[k+1] and the step returns (nodes[k].fn, ops, sc). sc' = sc comes with it at no cost. hD is needed (a primitive's gates = 1; DownOk gives it only where the node's output has a leaf). hB is used for fn < defs.size.

3. resolve_node: HOLDS. Invariant on (sc, r): Good, RefOk at nodes.size (Resolve.step_ok keeps both, with B = sc.off + gates), and wAt (tyOf P sc r) r.j = W. Each case of resolveStep, walked:
- node, prim: ends at offsets[k], m = 0. Shape: out = c.ret; CalleeOk: c.ret = value w; wAt (value w) j = w.
- node, call: into the callee's ret.get j. Shape: out = c.ret; hW: retType = ret and the return clause of BodyWide; DownOk: j < ret.count.
- node, batch of a primitive: ends at offsets[k] + j / 1, m = j / outLeaves < n by DownOk. out = array n (value w); wAt reads value w at j % 1.
- node, batch of a composite: out = array n c.ret, outLeaves = c.ret.leaves > 0 (j < n * that), wAt out j = wAt c.ret (j % L), and ret.get (j % L) by hW.
- SCAN 1, node, scan (rs_scan): n = 0 is rs_scan0, no step. out = tuple #[carry, array n y], c.ret = tuple #[carry, y]. j < carry.leaves: wAt out j = wAt carry j = wAt c.ret j, into member n - 1's ret.get j. Else: wAt out j = wAt y ((j - c) % y.leaves) (y.leaves > 0 since j < c + n * y.leaves), and the step reads ret.get (c + (j - c) % y.leaves), where wAt c.ret gives the tuple's second field at (j - c) % y.leaves. Equal.
- SCAN 2, parameter 0 of a scan member (rs_up_scan0): m = 0: the parent's args[0].seq.get j, Shape: args[0].ty = carry = params[0]'s type, BodyWide's argument clause of the parent. m > 0: member m - 1's ret.get j with j < carry.leaves, so wAt (tuple #[carry, y]) j = wAt carry j, by hW of the same callee. UpOk gives both indices in range.
- SCAN 3, parameter i > 0 of a scan member (rs_up_scan): i <= nxs: args[i].ty = array n e, e = params[i]'s type, dims = (n, e.leaves), index m * e.leaves + j, and wAt (array n e) (m * L + j) = wAt e j since j < L. Else args[i].ty = params[i]'s type. BodyWide's argument clause of the parent.
- parameter of a call's member: Shape: args[i].ty = params[i]'s type.
- parameter of a batch member: dims = none as a call; else FORM's lemma, wAt t ((j / L * ln + m) * L + j % L) = wAt mt j for removeAxis ax = (ln, mt), axisDims ax = (ln, L), j < mt.leaves, m < ln. I checked it by induction on ax: ax = 0 is the array case above; ax + 1 splits j = q * I + r over the inner member's leaves I, the index is q * (ln * I) + (the index of r), the second term is below e.leaves by axis_spec, so the outer mod drops q and the inductive case applies. True; I prove it in my file.
- a parameter at the root: Good says starts = none, which throws (rs_top). No step.
Src.ty?: at an argument's reference BodyOk gives RefOk P d k (r.i < k = below, or a parameter in range), at a return's RefOk at nodes.size = below, so ty? is some of exactly tyOf's type. Agreed, one line each.
RefOk at nodes.size: agreed, an argument's RefOk P d k implies it (Resolve.up_ok does this already).

Nothing in a statement or a definition looks wrong to me. Two notes, neither a change: (a) resolve_node does not need UnitsComposite: Good's own prim = false is all it reads. (b) hL in resolve_node is read only through Resolve.step_ok (to keep Good and RefOk); the width argument itself does not need the layout.

Next: wAt_widths and the axis lemma today, first node try as soon as the file elaborates; I comment each run id on TRN-121.

