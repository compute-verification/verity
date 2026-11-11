[2026-11-11T13:20Z] 796cf7e41662
FORM to training, TRN-103, 13:20Z (Wed 11 Nov). Your 13:18Z item 2, for WIDTH. READ, walked on paper, not run.

resolve_node's three scan cases need no hypothesis beyond the two the statement has:
- down through a scan node (rs_scan): hD's Shape for a scan (out = (carry, n of y), the callee's ret = (carry, y)) and hW of the callee (its return read as retType = ret). Leaf j < carry's leaves is leaf j of ret; a later leaf is leaf carry.leaves + (j - carry.leaves) % y.leaves of ret, the same width because an array's leaf is its element's.
- up from member m > 0's parameter 0 (rs_up_scan0): into member m - 1's return, leaf j of the carry: Shape's params[0] = carry and hW of the callee again.
- up from parameter i > 0 (rs_up_scan): Shape's second half (i ≤ nxs: the argument is n of the parameter's type, dims = (n, its leaves), so the index is m * L + j; else the argument's type is the parameter's) and BodyWide of the parent's argument.
n = 0 is rs_scan0: no step.

THE ONE LEMMA NOT IN THE TREE is the batch's, not a scan's: for t.removeAxis ax = some (ln, mt) and t.axisDims ax = some (ln, L), every j < mt.leaves and m < ln have Widths.wAt t ((j / L * ln + m) * L + j % L) = Widths.wAt mt j. BodiesOk.axis_spec (BodiesOk/Basic.lean 108) proves the index in range by induction on ax; this is the same induction with the width beside it. WIDTH proves it in its file; say 60 lines. It is easier after wAt_widths, so the order is wAt_widths, that lemma, resolve_node, gate_of_node.

Also: Widths.Src.ty? must give the source's type at every reference RefOk allows (parameters, and nodes below the bound): one line each from srcOf, but resolve_node needs it at each step.

I have moved no statement of groups 5 or 6: the scratch branch is still 0690666f63b0. The three Programs and a shared Input gate: sent 13:08Z. FINDING P (a primitive as a unit) went to you at about 13:21Z and bears on W too: resolve_node is stated from Good, which a composite unit has.

